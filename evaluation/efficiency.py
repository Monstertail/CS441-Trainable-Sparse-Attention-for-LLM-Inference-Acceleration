"""
Measure efficiency of Native Sparse Attention pretrain models.


"""

import argparse
import os
import re
import sys

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

# SLIDING_WINDOW_SIZE = 64
SLIDING_WINDOW_SIZE = 4
COMPRESS_BLOCK_SIZE = 16
COMPRESS_BLOCK_SLIDING_STRIDE = 8

FINE_BLOCK_SIZE = 16
NUM_FINE_SELECTED = 4

USE_DIFF_TOPK = True
QUERY_HEADS_SHARE_SELECTION = True

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


def print_config():
    """Print the key NSA / model hyperparameters used for this run."""
    # approximate number of tokens each query can attend to:
    # - sliding_window_size local tokens
    # - plus NUM_FINE_SELECTED fine blocks, each of size FINE_BLOCK_SIZE
    approx_tokens_per_query = SLIDING_WINDOW_SIZE + NUM_FINE_SELECTED * FINE_BLOCK_SIZE

    print("=== Efficiency test configuration (matching pretrain/train.py) ===")
    print(f"- num_tokens        : {DEFAULT_NUM_TOKENS}")
    print(f"- dim               : {DEFAULT_DIM}")
    print(f"- depth             : {DEFAULT_DEPTH}")
    print(f"- heads             : {DEFAULT_HEADS}")
    print(f"- kv_heads          : {DEFAULT_KV_HEADS}")
    print(f"- dim_head          : {DEFAULT_DIM_HEAD}")
    print(f"- sliding_window    : {SLIDING_WINDOW_SIZE}")
    print(f"- compress_block    : {COMPRESS_BLOCK_SIZE}")
    print(f"- compress_stride   : {COMPRESS_BLOCK_SLIDING_STRIDE}")
    print(f"- fine_block_size   : {FINE_BLOCK_SIZE}")
    print(f"- num_fine_selected : {NUM_FINE_SELECTED}")
    print(f"- use_diff_topk     : {USE_DIFF_TOPK}")
    print(f"- query_heads_share : {QUERY_HEADS_SHARE_SELECTION}")
    print(f"- use_triton_nsa    : {USE_TRITON_NSA}")
    print(f"- use_flex_sliding  : {USE_FLEX_SLIDING_WINDOW}")
    print(f"- use_flex_fine     : {USE_FLEX_FOR_FINE_SELECTION}")
    print(
        f"- approx tokens per query (sparse path): "
        f"{SLIDING_WINDOW_SIZE} (sliding) + "
        f"{NUM_FINE_SELECTED} * {FINE_BLOCK_SIZE} (fine) "
        f"≈ {approx_tokens_per_query} tokens"
    )
    print("===============================================================")


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
            query_heads_share_selected_kv=QUERY_HEADS_SHARE_SELECTION,
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


def measure_efficiency(
    model: torch.nn.Module,
    batch_size: int = 16,
    seq_len: int = 512,
    warmup_iters: int = 10,
    measure_iters: int = 100,
    device: str = "cuda",
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for efficiency measurement.")

    model.eval()

    # Prepare dummy input
    dummy_input = torch.randint(
        low=0,
        high=DEFAULT_NUM_TOKENS,
        size=(batch_size, seq_len),
        device=device,
        dtype=torch.long,
    )

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(dummy_input)
    torch.cuda.synchronize()

    # Reset memory stats before real measurement
    torch.cuda.reset_peak_memory_stats(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    with torch.no_grad():
        for _ in range(measure_iters):
            _ = model(dummy_input)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_per_batch_ms = elapsed_time_ms / float(measure_iters)

    tokens_per_batch = batch_size * seq_len
    tokens_per_sec = tokens_per_batch / (avg_time_per_batch_ms / 1000.0)

    peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2

    return avg_time_per_batch_ms, tokens_per_sec, peak_mem_mb


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure efficiency of full vs sparse attention models."
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
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for efficiency test",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=None,
        help=(
            "Sequence length for efficiency test. "
            "If omitted, will try to infer from checkpoint name (e.g. '..._seq4096_...')."
        ),
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=10,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--measure_iters",
        type=int,
        default=10,
        help="Number of timing iterations",
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

    # Show NSA / model config before building the model
    # print_config()

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

    avg_ms, tps, peak_mem = measure_efficiency(
        model=model,
        batch_size=args.batch_size,
        seq_len=seq_len,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
        device=device,
    )

    print(
        f"[{args.model_type}] "
        f"(seq_len={seq_len}) "
        f"Avg latency: {avg_ms:.2f} ms / batch | "
        f"Throughput: {tps:.2f} tokens/s | "
        f"Peak memory: {peak_mem:.2f} MB"
    )


if __name__ == "__main__":
    main()