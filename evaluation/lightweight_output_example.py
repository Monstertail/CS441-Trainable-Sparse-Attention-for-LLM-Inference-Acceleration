import argparse
import csv
import gzip
import json
import os
import re
import sys
from typing import Any

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

# Must match training hyperparameters
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

CS441_TEST_PATH = os.path.join(PROJECT_ROOT, "data_collection", "cs441_synthetic_test.json")
ENWIK8_PATH = os.path.join(PROJECT_ROOT, "pretrain", "data", "enwik8.gz")


def infer_seq_len_from_checkpoint(path: str) -> int | None:
    m = re.search(r"(?:^|[^0-9])seq(\d+)(?:[^0-9]|$)", os.path.basename(path))
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def infer_step_from_checkpoint(path: str) -> int | None:
    m = re.search(r"(?:^|[^0-9])step_(\d+)(?:[^0-9]|$)", os.path.basename(path))
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


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
    raise ValueError(f"Unknown compress method '{method}'")


def build_model(model_type: str, device: str = "cuda") -> Transformer:
    if model_type not in MODEL_TYPE_TO_METHOD:
        raise ValueError(f"Unknown model_type='{model_type}'")

    use_sparse_attn = model_type != "full"
    compress_method = MODEL_TYPE_TO_METHOD[model_type]
    if compress_method is None:
        compress_method = "conv"  # unused for full attention

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


def load_cs441_example(example_idx: int) -> dict[str, Any]:
    if not os.path.isfile(CS441_TEST_PATH):
        raise FileNotFoundError(f"cs441_synthetic_test.json not found at {CS441_TEST_PATH}")

    with open(CS441_TEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError("cs441_synthetic_test.json should be a non-empty list")

    idx = example_idx % len(data)
    item = data[idx]
    q = str(item.get("question", "")).strip()
    a = str(item.get("answer", "")).strip()
    return {"idx": idx, "question": q, "answer": a}


def load_enwik8_valid_bytes() -> bytes:
    """
    Load the enwik8 validation split bytes (same split as pretrain/train.py):
      - read first 95e6 bytes from enwik8.gz
      - train: first 90e6, valid: remaining
    """
    if not os.path.isfile(ENWIK8_PATH):
        raise FileNotFoundError(f"enwik8.gz not found at {ENWIK8_PATH}")

    with gzip.open(ENWIK8_PATH, "rb") as f:
        data = f.read(int(95e6))
    if len(data) < int(90e6) + 1:
        raise ValueError("enwik8 data too short; expected at least 90e6+1 bytes")
    valid = data[int(90e6) :]
    return valid


def slice_enwik8_prompt(valid_bytes: bytes, offset: int, length: int) -> tuple[int, bytes]:
    if length <= 0:
        raise ValueError("length must be > 0")
    n = len(valid_bytes)
    if n <= length + 1:
        raise ValueError(f"enwik8 valid split too short ({n}) for length={length}")
    start = offset % (n - length - 1)
    return start, valid_bytes[start : start + length]


def text_to_byte_tokens(s: str) -> torch.Tensor:
    b = s.encode("utf-8", errors="replace")
    # tokens must be [0,255]
    t = torch.frombuffer(b, dtype=torch.uint8).to(torch.int64)
    return t


def byte_tokens_to_text(t: torch.Tensor) -> str:
    b = bytes([int(x) & 0xFF for x in t.tolist()])
    return b.decode("utf-8", errors="replace")


def parse_args():
    ap = argparse.ArgumentParser(description="Load checkpoints and generate one cs441 example; append outputs to CSV")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=list(MODEL_TYPE_TO_METHOD.keys()),
    )
    ap.add_argument("--example_idx", type=int, default=0, help="Index into cs441_synthetic_test.json (OOD example)")
    ap.add_argument(
        "--id_offset",
        type=int,
        default=None,
        help="Start offset into enwik8 validation bytes for in-distribution prompt (default: derived from example_idx).",
    )
    ap.add_argument(
        "--id_prompt_bytes",
        type=int,
        default=512,
        help="Number of bytes (tokens) to use as in-distribution prompt from enwik8 valid split.",
    )
    ap.add_argument(
        "--datasets",
        type=str,
        default="both",
        choices=["both", "id", "ood"],
        help="Which dataset(s) to run: both / id / ood",
    )
    ap.add_argument("--gen_len", type=int, default=256, help="Number of tokens (bytes) to generate")
    ap.add_argument(
        "--max_prompt_bytes",
        type=int,
        default=2048,
        help="Truncate prompt to at most this many bytes (tokens) to avoid extremely long prompts",
    )
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    ap.add_argument("--filter_thres", type=float, default=0.9)
    ap.add_argument("--use_kv_cache", action="store_true", help="Enable KV cache during generation")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out_csv", type=str, required=True, help="Output CSV path (append)")
    return ap.parse_args()


def main():
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available")

    seq_len = infer_seq_len_from_checkpoint(args.checkpoint)
    step = infer_step_from_checkpoint(args.checkpoint)

    model = build_model(args.model_type, device=args.device)
    load_checkpoint(model, args.checkpoint, device=args.device)

    model.eval()
    torch.backends.cudnn.benchmark = True

    rows: list[dict[str, Any]] = []

    # -------- OOD (cs441) --------
    if args.datasets in ("both", "ood"):
        ex = load_cs441_example(args.example_idx)
        prompt_text = f"Question: {ex['question']}\nAnswer:"
        prompt_tokens = text_to_byte_tokens(prompt_text)

        if args.max_prompt_bytes is not None and args.max_prompt_bytes > 0:
            prompt_tokens = prompt_tokens[: args.max_prompt_bytes]

        prompt = prompt_tokens.unsqueeze(0).to(args.device)

        with torch.no_grad():
            gen_tokens = model.sample(
                prompt,
                seq_len=prompt.size(1) + args.gen_len,
                temperature=args.temperature,
                filter_thres=args.filter_thres,
                use_cache_kv=args.use_kv_cache,
            )

        gen_text = byte_tokens_to_text(gen_tokens[0].detach().cpu())

        rows.append(
            {
                "dataset": "ood",
                "checkpoint": args.checkpoint,
                "step": "" if step is None else step,
                "seq_len": "" if seq_len is None else seq_len,
                "model_type": args.model_type,
                "example_idx": ex["idx"],
                "id_offset": "",
                "question": ex["question"],
                "answer_gt": ex["answer"],
                "prompt": prompt_text,
                "gen_len": args.gen_len,
                "temperature": args.temperature,
                "use_kv_cache": int(args.use_kv_cache),
                "generated": gen_text,
            }
        )

    # -------- In-distribution (enwik8 valid) --------
    if args.datasets in ("both", "id"):
        valid_bytes = load_enwik8_valid_bytes()

        id_offset = args.id_offset
        if id_offset is None:
            # deterministic, derived from example_idx so it's easy to reproduce
            id_offset = int(args.example_idx) * 10000

        start, prompt_bytes = slice_enwik8_prompt(valid_bytes, offset=id_offset, length=args.id_prompt_bytes)

        # Use latin-1 to preserve bytes 1:1 in the CSV prompt text
        prompt_text = prompt_bytes.decode("latin-1", errors="replace")
        prompt_tokens = torch.frombuffer(prompt_bytes, dtype=torch.uint8).to(torch.int64)

        if args.max_prompt_bytes is not None and args.max_prompt_bytes > 0:
            prompt_tokens = prompt_tokens[: args.max_prompt_bytes]

        prompt = prompt_tokens.unsqueeze(0).to(args.device)

        with torch.no_grad():
            gen_tokens = model.sample(
                prompt,
                seq_len=prompt.size(1) + args.gen_len,
                temperature=args.temperature,
                filter_thres=args.filter_thres,
                use_cache_kv=args.use_kv_cache,
            )

        gen_text = byte_tokens_to_text(gen_tokens[0].detach().cpu())

        rows.append(
            {
                "dataset": "id",
                "checkpoint": args.checkpoint,
                "step": "" if step is None else step,
                "seq_len": "" if seq_len is None else seq_len,
                "model_type": args.model_type,
                "example_idx": "",
                "id_offset": start,
                "question": "",
                "answer_gt": "",
                "prompt": prompt_text,
                "gen_len": args.gen_len,
                "temperature": args.temperature,
                "use_kv_cache": int(args.use_kv_cache),
                "generated": gen_text,
            }
        )

    out_dir = os.path.dirname(os.path.abspath(args.out_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if not rows:
        raise RuntimeError("No outputs were generated (check --datasets)")

    write_header = not os.path.exists(args.out_csv) or os.path.getsize(args.out_csv) == 0
    with open(args.out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"[ok] appended output to: {args.out_csv}")


if __name__ == "__main__":
    main()
