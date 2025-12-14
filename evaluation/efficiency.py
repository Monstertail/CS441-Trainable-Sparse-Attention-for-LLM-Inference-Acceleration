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
    prompt_len: int | None = None,
    gen_len: int | None = None,
    use_kv_cache: bool = True,
    warmup_iters: int = 10,
    measure_iters: int = 100,
    device: str = "cuda",
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for efficiency measurement.")

    model.eval()

    if seq_len <= 1:
        raise ValueError(f"seq_len must be > 1, got {seq_len}")

    if prompt_len is None:
        prompt_len = max(1, seq_len // 2)
    if gen_len is None:
        gen_len = max(1, seq_len - prompt_len)

    if prompt_len <= 0 or gen_len <= 0:
        raise ValueError(f"prompt_len and gen_len must be > 0, got prompt_len={prompt_len}, gen_len={gen_len}")

    # Prepare a fixed token buffer so we can avoid per-step torch.cat allocations.
    # We will fill generated tokens in-place during decode.
    token_buffer = torch.randint(
        low=0,
        high=DEFAULT_NUM_TOKENS,
        size=(batch_size, prompt_len + gen_len),
        device=device,
        dtype=torch.long,
    )
    prompt = token_buffer[:, :prompt_len]

    # Full-attention path in this repo does NOT support a correct KV-cache inferencing path:
    # when cache is provided, Transformer.forward embeds only the last token, and full attention
    # would then attend over only that token (losing prompt context). So we must disable cache.
    if use_kv_cache and isinstance(model, Transformer) and not getattr(model, "use_sparse_attn", False):
        print("[warn] model is full attention; disabling KV cache for correctness in decode benchmark.")
        use_kv_cache = False

    # -----------------------
    # Prefill (prompt forward)
    # -----------------------

    # Warmup prefill
    with torch.no_grad():
        for _ in range(warmup_iters):
            if use_kv_cache:
                _ = model(prompt, return_cache=True)
            else:
                _ = model(prompt)
    torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    with torch.no_grad():
        for _ in range(measure_iters):
            if use_kv_cache:
                _ = model(prompt, return_cache=True)
            else:
                _ = model(prompt)
    end_event.record()
    torch.cuda.synchronize(device)

    elapsed_time_ms = start_event.elapsed_time(end_event)
    prefill_avg_ms_per_batch = elapsed_time_ms / float(measure_iters)

    prefill_tokens_per_batch = batch_size * prompt_len
    prefill_tokens_per_sec = prefill_tokens_per_batch / (prefill_avg_ms_per_batch / 1000.0)

    prefill_peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2

    # -----------------------
    # Decode (autoregressive)
    # -----------------------

    # Warmup decode
    with torch.no_grad():
        warm_steps = min(gen_len, 4)
        if warm_steps > 0:
            # prefill once to get cache (if enabled)
            cache = None
            if use_kv_cache:
                _, cache = model(prompt, return_cache=True)

            cur_len = prompt_len
            for _ in range(warm_steps):
                if use_kv_cache:
                    logits, cache = model(token_buffer[:, :cur_len], cache=cache, return_cache=True)
                    next_token = logits[:, -1].argmax(dim=-1)
                else:
                    logits = model(token_buffer[:, :cur_len])
                    next_token = logits[:, -1].argmax(dim=-1)
                token_buffer[:, cur_len] = next_token
                cur_len += 1

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    with torch.no_grad():
        for _ in range(measure_iters):
            # prefill once per measurement run to simulate real usage
            cache = None
            if use_kv_cache:
                _, cache = model(prompt, return_cache=True)

            cur_len = prompt_len
            for _step in range(gen_len):
                if use_kv_cache:
                    logits, cache = model(token_buffer[:, :cur_len], cache=cache, return_cache=True)
                    next_token = logits[:, -1].argmax(dim=-1)
                else:
                    logits = model(token_buffer[:, :cur_len])
                    next_token = logits[:, -1].argmax(dim=-1)

                if cur_len < token_buffer.size(1):
                    token_buffer[:, cur_len] = next_token
                cur_len += 1
    end_event.record()
    torch.cuda.synchronize(device)

    elapsed_time_ms = start_event.elapsed_time(end_event)
    decode_avg_ms_per_run = elapsed_time_ms / float(measure_iters)

    decode_tokens_per_run = batch_size * gen_len
    decode_tokens_per_sec = decode_tokens_per_run / (decode_avg_ms_per_run / 1000.0)

    decode_peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2

    return {
        "seq_len": seq_len,
        "prompt_len": prompt_len,
        "gen_len": gen_len,
        "use_kv_cache": use_kv_cache,
        "prefill_avg_ms_per_batch": prefill_avg_ms_per_batch,
        "prefill_tokens_per_sec": prefill_tokens_per_sec,
        "prefill_peak_mem_mb": prefill_peak_mem_mb,
        "decode_avg_ms_per_run": decode_avg_ms_per_run,
        "decode_tokens_per_sec": decode_tokens_per_sec,
        "decode_peak_mem_mb": decode_peak_mem_mb,
    }


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
        "--prompt_len",
        type=int,
        default=None,
        help="Prompt length for prefill. Default: seq_len//2",
    )
    parser.add_argument(
        "--gen_len",
        type=int,
        default=None,
        help="Generation length (number of decoded tokens). Default: seq_len - prompt_len",
    )
    parser.add_argument(
        "--no_kv_cache",
        action="store_true",
        help="Disable KV cache in decode benchmark (cache is only supported/correct for sparse attention in this repo).",
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=1,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--measure_iters",
        type=int,
        default=3,
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

    stats = measure_efficiency(
        model=model,
        batch_size=args.batch_size,
        seq_len=seq_len,
        prompt_len=args.prompt_len,
        gen_len=args.gen_len,
        use_kv_cache=not args.no_kv_cache,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
        device=device,
    )

    cache_str = "on" if stats["use_kv_cache"] else "off"
    print(f"[{args.model_type}] seq_len={stats['seq_len']} prompt_len={stats['prompt_len']} gen_len={stats['gen_len']} kv_cache={cache_str}")
    print(
        f"  Prefill: {stats['prefill_avg_ms_per_batch']:.2f} ms / batch | "
        f"{stats['prefill_tokens_per_sec']:.2f} tokens/s | "
        f"peak mem {stats['prefill_peak_mem_mb']:.2f} MB"
    )
    print(
        f"  Decode : {stats['decode_avg_ms_per_run']:.2f} ms / run (gen_len={stats['gen_len']}) | "
        f"{stats['decode_tokens_per_sec']:.2f} tokens/s | "
        f"peak mem {stats['decode_peak_mem_mb']:.2f} MB"
    )


if __name__ == "__main__":
    main()