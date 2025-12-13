import argparse
import gzip
import json
import math
import os
import re
import sys
from typing import Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Import project modules (same trick as in pretrain/train.py)
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sparse_attention.native_sparse_attention_pytorch.transformer import Transformer
from sparse_attention.native_sparse_attention_pytorch.compress_networks import (  # noqa: E402
    ConvLinearCompress,
    AttentionPool,
    GroupedMLP,
    MeanPoolCompress,
)


# ---------------------------------------------------------------------------
# Default model hyperparameters – must match pretrain/train.py
# ---------------------------------------------------------------------------

DEFAULT_NUM_TOKENS = 256
DEFAULT_DIM = 512
DEFAULT_DEPTH = 6
DEFAULT_HEADS = 8
DEFAULT_KV_HEADS = 4
DEFAULT_DIM_HEAD = 64

SLIDING_WINDOW_SIZE = 64
COMPRESS_BLOCK_SIZE = 16
COMPRESS_BLOCK_SLIDING_STRIDE = 8

FINE_BLOCK_SIZE = 16
NUM_FINE_SELECTED = 4

USE_DIFF_TOPK = True
QUERY_HEADS_SHARE_SELECTED_KV = True

USE_FLEX_SLIDING_WINDOW = True
USE_TRITON_NSA = True
USE_FLEX_FOR_FINE_SELECTION = False


MODEL_TYPE_TO_METHOD = {
    "full": None,
    "sparse_conv": "conv",
    "sparse_mlp": "mlp",
    "sparse_attn": "attn",
    "sparse_mean": "mean",
}


ENWIK8_PATH = os.path.join(PROJECT_ROOT, "pretrain", "data", "enwik8.gz")
CS441_TEST_PATH = os.path.join(PROJECT_ROOT, "data_collection", "cs441_synthetic_test.json")


def build_compress_module(method: str):
    if method == "mlp":
        return GroupedMLP(
            dim_head=DEFAULT_DIM_HEAD,
            compress_window_size=COMPRESS_BLOCK_SIZE,
            heads=DEFAULT_KV_HEADS,
        )
    if method == "conv":
        return ConvLinearCompress(
            heads=DEFAULT_KV_HEADS,
            dim_head=DEFAULT_DIM_HEAD,
            compress_window_size=COMPRESS_BLOCK_SIZE,
        )
    if method == "attn":
        return AttentionPool(
            dim_head=DEFAULT_DIM_HEAD,
            compress_window_size=COMPRESS_BLOCK_SIZE,
        )
    if method == "mean":
        return MeanPoolCompress(
            dim_head=DEFAULT_DIM_HEAD,
            compress_window_size=COMPRESS_BLOCK_SIZE,
        )
    raise ValueError(
        f"Unknown compress method '{method}'. "
        f"Valid options: conv, attn, mlp, mean."
    )


def build_model(model_type: str, device: str = "cuda") -> Transformer:
    if model_type not in MODEL_TYPE_TO_METHOD:
        raise ValueError(
            f"Unknown model_type='{model_type}'. "
            f"Choose from: {', '.join(MODEL_TYPE_TO_METHOD.keys())}"
        )

    use_sparse_attn = model_type != "full"
    compress_method = MODEL_TYPE_TO_METHOD[model_type]

    # For full attention, compress module will not be used, but we can still build one.
    if compress_method is None:
        compress_method = "conv"

    compress_module = build_compress_module(compress_method)

    model = Transformer(
        num_tokens=DEFAULT_NUM_TOKENS,
        dim=DEFAULT_DIM,
        depth=DEFAULT_DEPTH,
        heads=DEFAULT_HEADS,
        dim_head=DEFAULT_DIM_HEAD,
        kv_heads=DEFAULT_KV_HEADS,
        use_sparse_attn=use_sparse_attn,
        causal=True,
        use_flex_sliding_window=USE_FLEX_SLIDING_WINDOW,
        use_triton_fine_selection=USE_TRITON_NSA,
        use_flex_fine_selection=USE_FLEX_FOR_FINE_SELECTION,
        sparse_attn_kwargs=dict(
            sliding_window_size=SLIDING_WINDOW_SIZE,
            compress_block_size=COMPRESS_BLOCK_SIZE,
            compress_block_sliding_stride=COMPRESS_BLOCK_SLIDING_STRIDE,
            compress_mlp=compress_module,
            selection_block_size=FINE_BLOCK_SIZE,
            num_selected_blocks=NUM_FINE_SELECTED,
            use_diff_topk=USE_DIFF_TOPK,
            query_heads_share_selected_kv=QUERY_HEADS_SHARE_SELECTED_KV,
        ),
    )

    model.to(device)
    return model


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: str = "cuda"):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state_dict = state["model"]
    else:
        state_dict = state

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] Missing keys in state_dict: {missing}")
    if unexpected:
        print(f"[warn] Unexpected keys in state_dict: {unexpected}")


def load_enwik8_test(max_tokens: int | None = None) -> torch.Tensor:
    """Load the enwik8 validation portion as in-distribution test bytes."""
    if not os.path.isfile(ENWIK8_PATH):
        raise FileNotFoundError(f"enwik8.gz not found at {ENWIK8_PATH}")

    with gzip.open(ENWIK8_PATH) as f:
        data = np.frombuffer(f.read(int(95e6)), dtype=np.uint8).copy()
    _, np_valid = np.split(data, [int(90e6)])
    data_test = torch.from_numpy(np_valid).long()

    if max_tokens is not None and max_tokens > 0:
        # +1 so we still have at least one prediction target
        data_test = data_test[: max_tokens + 1]

    return data_test


def load_cs441_test(max_examples: int | None = None) -> Tuple[torch.Tensor, int]:
    """Load cs441_synthetic_test.json and convert to a continuous byte stream."""
    if not os.path.isfile(CS441_TEST_PATH):
        raise FileNotFoundError(f"cs441_synthetic_test.json not found at {CS441_TEST_PATH}")

    with open(CS441_TEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("cs441_synthetic_test.json should contain a list of objects.")

    if max_examples is not None and max_examples > 0:
        data = data[:max_examples]

    texts = []
    for item in data:
        q = item.get("question", "").strip()
        a = item.get("answer", "").strip()
        # Simple serialization: question + answer as plain text
        texts.append(q + "\nAnswer: " + a + "\n\n")

    full_text = "".join(texts)
    byte_array = full_text.encode("utf-8", errors="replace")
    tokens = torch.from_numpy(np.frombuffer(byte_array, dtype=np.uint8).copy()).long()
    return tokens, len(data)


def compute_ppl_on_tokens(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    seq_len: int,
    batch_size: int,
    device: str,
    name: str,
) -> Tuple[float, float, int]:
    """Compute perplexity on a flat byte token stream."""
    model.eval()

    total_len = int(tokens.size(0))
    if total_len <= seq_len:
        raise ValueError(f"{name}: token stream too short ({total_len}) for seq_len={seq_len}")

    num_chunks_total = (total_len - 1) // seq_len
    if num_chunks_total <= 0:
        raise ValueError(f"{name}: not enough tokens for even one chunk.")

    print(f"[{name}] total tokens: {total_len}, seq_len: {seq_len}, approx chunks: {num_chunks_total}")

    total_nll = 0.0
    total_tokens = 0
    num_chunks = 0

    batch: list[torch.Tensor] = []

    with torch.no_grad():
        for i in range(num_chunks_total):
            start = i * seq_len
            end = start + seq_len + 1
            if end > total_len:
                break

            chunk = tokens[start:end]  # (seq_len + 1,)
            batch.append(chunk)

            if len(batch) == batch_size:
                batch_tensor = torch.stack(batch, dim=0).to(device)
                loss = model(batch_tensor, return_loss=True)

                n_tokens = (batch_tensor.size(1) - 1) * batch_tensor.size(0)
                total_nll += loss.item() * n_tokens
                total_tokens += n_tokens
                num_chunks += batch_tensor.size(0)

                batch = []

        # process remaining batch
        if batch:
            batch_tensor = torch.stack(batch, dim=0).to(device)
            loss = model(batch_tensor, return_loss=True)

            n_tokens = (batch_tensor.size(1) - 1) * batch_tensor.size(0)
            total_nll += loss.item() * n_tokens
            total_tokens += n_tokens
            num_chunks += batch_tensor.size(0)

    if total_tokens == 0:
        raise RuntimeError(f"{name}: no tokens were evaluated.")

    avg_nll = total_nll / float(total_tokens)
    ppl = math.exp(avg_nll)

    print(f"[{name}] evaluated chunks: {num_chunks}, tokens: {total_tokens}")
    print(f"[{name}] avg NLL (nats): {avg_nll:.4f} | PPL: {ppl:.4f}")

    return ppl, avg_nll, total_tokens


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure in-distribution and out-of-distribution perplexity for pretrain models.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint (.pt) file saved by pretrain/train.py",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=list(MODEL_TYPE_TO_METHOD.keys()),
        help="Model type: full / sparse_conv / sparse_mlp / sparse_attn / sparse_mean",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=None,
        help=(
            "Sequence length used for evaluation windows. "
            "If omitted, will try to infer from checkpoint name (e.g. '..._seq4096_...')."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--max_id_tokens",
        type=int,
        default=1_000_000,
        help="Max number of in-distribution (enwik8) tokens to evaluate (approx).",
    )
    parser.add_argument(
        "--max_ood_examples",
        type=int,
        default=256,
        help="Max number of cs441_synthetic_test examples to evaluate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (default: cuda)",
    )
    return parser.parse_args()


def infer_seq_len_from_checkpoint(path: str) -> int | None:
    """
    Infer seq_len from checkpoint filename.
    Expected patterns: '..._seq512_...' or '..._seq4096_...'.
    """
    m = re.search(r"(?:^|[^0-9])seq(\d+)(?:[^0-9]|$)", os.path.basename(path))
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def main():
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available.")

    device = args.device
    torch.backends.cudnn.benchmark = True

    print(f"==> Loading model_type='{args.model_type}' from: {args.checkpoint}")
    model = build_model(args.model_type, device=device)
    load_checkpoint(model, args.checkpoint, device=device)

    seq_len = args.seq_len
    if seq_len is None:
        seq_len = infer_seq_len_from_checkpoint(args.checkpoint)
        if seq_len is None:
            raise ValueError(
                "seq_len was not provided and could not be inferred from checkpoint name. "
                "Please pass --seq_len explicitly."
            )

    print(f"==> In-distribution test on enwik8 (up to {args.max_id_tokens} tokens)")
    enwik8_tokens = load_enwik8_test(max_tokens=args.max_id_tokens)
    ppl_id, nll_id, tokens_id = compute_ppl_on_tokens(
        model=model,
        tokens=enwik8_tokens,
        seq_len=seq_len,
        batch_size=args.batch_size,
        device=device,
        name="in-distribution (enwik8)",
    )

    print(f"\n==> Out-of-distribution test on cs441_synthetic_test.json (up to {args.max_ood_examples} QA pairs)")
    cs_tokens, num_examples = load_cs441_test(max_examples=args.max_ood_examples)
    ppl_ood, nll_ood, tokens_ood = compute_ppl_on_tokens(
        model=model,
        tokens=cs_tokens,
        seq_len=seq_len,
        batch_size=args.batch_size,
        device=device,
        name="out-of-distribution (cs441)",
    )

    print("\n===== Perplexity Summary =====")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model type: {args.model_type} | seq_len: {seq_len} | batch_size: {args.batch_size}")
    print(f"In-distribution (enwik8):     PPL = {ppl_id:.4f} | avg NLL = {nll_id:.4f} | tokens = {tokens_id}")
    print(
        f"Out-of-distribution (cs441): PPL = {ppl_ood:.4f} | avg NLL = {nll_ood:.4f} | "
        f"tokens = {tokens_ood} | examples = {num_examples}"
    )
    print("==============================")


if __name__ == "__main__":
    main()


