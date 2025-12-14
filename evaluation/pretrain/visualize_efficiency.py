import argparse
import csv
import os
import re
from typing import Any


MODEL_TYPE_LABEL = {
    "full": "Full attention",
    "sparse_conv": "Sparse attention + ConvLinear",
    "sparse_mlp": "Sparse attention + GroupedMLP",
    "sparse_attn": "Sparse attention + AttentionPool",
    "sparse_mean": "Sparse attention + MeanPool",
}

MODEL_ORDER = ["full", "sparse_conv", "sparse_mlp", "sparse_attn", "sparse_mean"]

SEQ_STYLE = {
    512: {"linestyle": "-", "marker": "o"},
    4096: {"linestyle": "--", "marker": "s"},
}


def _as_int(v: str) -> int | None:
    try:
        return int(v)
    except Exception:
        return None


def _as_float(v: str) -> float | None:
    try:
        return float(v)
    except Exception:
        return None


def infer_step_from_checkpoint(path: str) -> int | None:
    base = os.path.basename(path)
    m = re.search(r"(?:^|[^0-9])step_(\d+)(?:[^0-9]|$)", base)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def read_rows(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["batch_size"] = _as_int(r.get("batch_size", ""))
            r["seq_len"] = _as_int(r.get("seq_len", ""))
            r["kv_cache"] = _as_int(r.get("kv_cache", ""))
            r["prefill_avg_ms_per_batch"] = _as_float(r.get("prefill_avg_ms_per_batch", ""))
            r["decode_avg_ms_per_run"] = _as_float(r.get("decode_avg_ms_per_run", ""))
            r["prefill_tokens_per_sec"] = _as_float(r.get("prefill_tokens_per_sec", ""))
            r["decode_tokens_per_sec"] = _as_float(r.get("decode_tokens_per_sec", ""))
            # for title
            ckpt = r.get("checkpoint", "")
            r["step"] = infer_step_from_checkpoint(ckpt) if ckpt else None
            rows.append(r)
    return rows


def _try_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as e:
        print(f"[warn] matplotlib not available; skip plotting ({e})")
        return None


def _method_order(keys: list[str]) -> list[str]:
    return [m for m in MODEL_ORDER if m in keys] + [m for m in sorted(keys) if m not in MODEL_ORDER]


def _method_colors(models: list[str]) -> dict[str, str]:
    palette = [
        "#4C78A8",  # blue
        "#F58518",  # orange
        "#54A24B",  # green
        "#E45756",  # red
        "#B279A2",  # purple
        "#FF9DA6",
        "#9D755D",
        "#BAB0AC",
    ]
    return {m: palette[i % len(palette)] for i, m in enumerate(models)}


def main():
    ap = argparse.ArgumentParser(description="Visualize efficiency_step*.csv as curves vs batch size")
    ap.add_argument("--csv", type=str, required=True, help="Path to efficiency_step*.csv")
    ap.add_argument(
        "--kv_cache",
        type=int,
        default=1,
        choices=[0, 1],
        help="Filter by kv_cache (0/1). Default: 1",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for figures (default: alongside the CSV)",
    )
    ap.add_argument(
        "--out_name",
        type=str,
        default=None,
        help="Output png filename (default: efficiency_curves_step{step}_kvcache{kv}.png)",
    )
    args = ap.parse_args()

    rows = read_rows(args.csv)
    if not rows:
        raise SystemExit(f"No rows found in: {args.csv}")

    filt = [r for r in rows if r.get("kv_cache") == args.kv_cache]
    if not filt:
        raise SystemExit("No rows after kv_cache filtering")

    # infer step (if unique)
    steps = sorted({r.get("step") for r in filt if r.get("step") is not None})
    step = steps[0] if len(steps) == 1 else (steps[-1] if steps else None)

    seq_lens = sorted({r.get("seq_len") for r in filt if r.get("seq_len") is not None})
    model_types = sorted({r.get("model_type") for r in filt if r.get("model_type")})
    batch_sizes = sorted({r.get("batch_size") for r in filt if r.get("batch_size") is not None})

    models_sorted = _method_order(model_types)
    colors = _method_colors(models_sorted)

    # index: (model_type, seq_len) -> {bs -> row}
    by_ms: dict[tuple[str, int], dict[int, dict[str, Any]]] = {}
    for r in filt:
        mt = r.get("model_type")
        sl = r.get("seq_len")
        bs = r.get("batch_size")
        if mt is None or sl is None or bs is None:
            continue
        by_ms.setdefault((mt, sl), {})[bs] = r

    plt = _try_import_matplotlib()
    if plt is None:
        return

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(args.csv)) or "."
    os.makedirs(out_dir, exist_ok=True)

    if args.out_name is None:
        out_name = "efficiency_curves"
        if step is not None:
            out_name += f"_step{step}"
        out_name += f"_kvcache{args.kv_cache}.png"
    else:
        out_name = args.out_name

    out_path = os.path.join(out_dir, out_name)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.5), sharex=True)

    panels = [
        (axes[0, 0], "prefill_avg_ms_per_batch", "Prefill time (ms / batch)", "Prefill time"),
        (axes[0, 1], "decode_avg_ms_per_run", "Decode time (ms / run)", "Decode time"),
        (axes[1, 0], "prefill_tokens_per_sec", "Prefill throughput (tokens/s)", "Prefill throughput"),
        (axes[1, 1], "decode_tokens_per_sec", "Decode throughput (tokens/s)", "Decode throughput"),
    ]

    for ax, key, ylabel, title in panels:
        for mt in models_sorted:
            for sl in seq_lens:
                points = by_ms.get((mt, sl), {})
                xs: list[int] = []
                ys: list[float] = []
                for bs in batch_sizes:
                    r = points.get(bs)
                    if not r:
                        continue
                    v = r.get(key)
                    if v is None:
                        continue
                    xs.append(bs)
                    ys.append(float(v))

                if not xs:
                    continue

                style = SEQ_STYLE.get(sl, {"linestyle": "-.", "marker": "^"})
                ax.plot(
                    xs,
                    ys,
                    color=colors.get(mt, "#333333"),
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    linewidth=2.0,
                    markersize=5,
                    alpha=0.95,
                )

        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.35)

    for ax in axes[1, :]:
        ax.set_xlabel("Batch size")
        ax.set_xticks(batch_sizes)

    # Legends: methods (colors) + seq_len (linestyle)
    method_handles = [
        plt.Line2D([0], [0], color=colors[m], lw=3, label=MODEL_TYPE_LABEL.get(m, m)) for m in models_sorted
    ]
    seq_handles = []
    for sl in seq_lens:
        st = SEQ_STYLE.get(sl, {"linestyle": "-.", "marker": "^"})
        seq_handles.append(
            plt.Line2D(
                [0],
                [0],
                color="#111111",
                lw=2,
                linestyle=st["linestyle"],
                marker=st["marker"],
                label=f"seq_len={sl}",
            )
        )

    subtitle_parts = []
    if step is not None:
        subtitle_parts.append(f"step={step}")
    subtitle_parts.append(f"kv_cache={args.kv_cache}")

    fig.suptitle("Efficiency vs batch size" + (" (" + ", ".join(subtitle_parts) + ")" if subtitle_parts else ""))

    # place legends below
    fig.legend(handles=method_handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.35, -0.02))
    fig.legend(handles=seq_handles, loc="lower center", ncol=len(seq_handles), frameon=False, bbox_to_anchor=(0.82, -0.02))

    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"[plot] saved: {out_path}")


if __name__ == "__main__":
    main()
