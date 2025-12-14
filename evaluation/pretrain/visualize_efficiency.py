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
            r["prompt_len"] = _as_int(r.get("prompt_len", ""))
            r["gen_len"] = _as_int(r.get("gen_len", ""))
            r["kv_cache"] = _as_int(r.get("kv_cache", ""))
            r["prefill_avg_ms_per_batch"] = _as_float(r.get("prefill_avg_ms_per_batch", ""))
            r["decode_avg_ms_per_run"] = _as_float(r.get("decode_avg_ms_per_run", ""))
            r["prefill_tokens_per_sec"] = _as_float(r.get("prefill_tokens_per_sec", ""))
            r["decode_tokens_per_sec"] = _as_float(r.get("decode_tokens_per_sec", ""))
            r["kv_cache_saving_ratio"] = _as_float(r.get("kv_cache_saving_ratio", ""))
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


def _kv_cache_label(v: int) -> str:
    return "enabled" if v == 1 else "disabled"


def plot_batch_curves(
    *,
    plt,
    out_path: str,
    step: int | None,
    kv_cache: int,
    seq_lens: list[int],
    models_sorted: list[str],
    colors: dict[str, str],
    batch_sizes: list[int],
    by_ms: dict[tuple[str, int], dict[int, dict[str, Any]]],
):
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
    subtitle_parts.append(f"kv_cache={_kv_cache_label(kv_cache)}")

    fig.suptitle("Efficiency vs batch size" + (" (" + ", ".join(subtitle_parts) + ")" if subtitle_parts else ""))

    # place legends below
    fig.legend(handles=method_handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.35, -0.02))
    fig.legend(handles=seq_handles, loc="lower center", ncol=len(seq_handles), frameon=False, bbox_to_anchor=(0.82, -0.02))

    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_decode_vs_prompt(
    *,
    plt,
    out_path: str,
    step: int | None,
    kv_cache: int,
    seq_len: int,
    fixed_batch_size: int,
    models_sorted: list[str],
    colors: dict[str, str],
    rows: list[dict[str, Any]],
):
    # filter
    filt = [
        r
        for r in rows
        if r.get("seq_len") == seq_len and r.get("batch_size") == fixed_batch_size and r.get("prompt_len") is not None
    ]
    if not filt:
        print(f"[warn] no rows for decode_vs_prompt (seq_len={seq_len}, batch_size={fixed_batch_size})")
        return False

    # index: (model_type, prompt_len) -> decode_tps
    by_mp: dict[tuple[str, int], float] = {}
    prompt_lens = sorted({r.get("prompt_len") for r in filt if r.get("prompt_len") is not None})
    for r in filt:
        mt = r.get("model_type")
        pl = r.get("prompt_len")
        v = r.get("decode_tokens_per_sec")
        if mt and pl is not None and v is not None:
            by_mp[(mt, pl)] = float(v)

    fig = plt.figure(figsize=(11.5, 4.6))
    ax = fig.add_subplot(1, 1, 1)

    for mt in models_sorted:
        xs: list[int] = []
        ys: list[float] = []
        for pl in prompt_lens:
            v = by_mp.get((mt, pl))
            if v is None:
                continue
            xs.append(pl)
            ys.append(v)
        if not xs:
            continue
        ax.plot(
            xs,
            ys,
            label=MODEL_TYPE_LABEL.get(mt, mt),
            color=colors.get(mt, "#333333"),
            linewidth=2.2,
            marker="o",
            markersize=4.5,
        )

    subtitle_parts = []
    if step is not None:
        subtitle_parts.append(f"step={step}")
    subtitle_parts.append(f"seq_len={seq_len}")
    subtitle_parts.append(f"batch_size={fixed_batch_size}")
    subtitle_parts.append(f"kv_cache={_kv_cache_label(kv_cache)}")

    ax.set_title("Decode throughput vs prompt length (" + ", ".join(subtitle_parts) + ")")
    ax.set_xlabel("Prompt length")
    ax.set_ylabel("Decode throughput (tokens/s)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_decode_vs_batch(
    *,
    plt,
    out_path: str,
    step: int | None,
    kv_cache: int,
    seq_len: int,
    fixed_prompt_len: int,
    models_sorted: list[str],
    colors: dict[str, str],
    rows: list[dict[str, Any]],
):
    filt = [
        r
        for r in rows
        if r.get("seq_len") == seq_len and r.get("prompt_len") == fixed_prompt_len and r.get("batch_size") is not None
    ]
    if not filt:
        print(f"[warn] no rows for decode_vs_batch (seq_len={seq_len}, prompt_len={fixed_prompt_len})")
        return False

    batch_sizes = sorted({r.get("batch_size") for r in filt if r.get("batch_size") is not None})
    by_mb: dict[tuple[str, int], float] = {}
    for r in filt:
        mt = r.get("model_type")
        bs = r.get("batch_size")
        v = r.get("decode_tokens_per_sec")
        if mt and bs is not None and v is not None:
            by_mb[(mt, bs)] = float(v)

    fig = plt.figure(figsize=(11.5, 4.6))
    ax = fig.add_subplot(1, 1, 1)

    for mt in models_sorted:
        xs: list[int] = []
        ys: list[float] = []
        for bs in batch_sizes:
            v = by_mb.get((mt, bs))
            if v is None:
                continue
            xs.append(bs)
            ys.append(v)
        if not xs:
            continue
        ax.plot(
            xs,
            ys,
            label=MODEL_TYPE_LABEL.get(mt, mt),
            color=colors.get(mt, "#333333"),
            linewidth=2.2,
            marker="o",
            markersize=4.5,
        )

    subtitle_parts = []
    if step is not None:
        subtitle_parts.append(f"step={step}")
    subtitle_parts.append(f"seq_len={seq_len}")
    subtitle_parts.append(f"prompt_len={fixed_prompt_len}")
    subtitle_parts.append(f"kv_cache={_kv_cache_label(kv_cache)}")

    ax.set_title("Decode throughput vs batch size (" + ", ".join(subtitle_parts) + ")")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Decode throughput (tokens/s)")
    ax.set_xticks(batch_sizes)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_decode_dashboard_2x2(
    *,
    plt,
    out_path: str,
    step: int | None,
    kv_cache: int,
    seq_len: int,
    fixed_batch_size: int,
    fixed_prompt_len: int,
    models_sorted: list[str],
    colors: dict[str, str],
    rows: list[dict[str, Any]],
):
    """
    2x2 dashboard:
      (0,0) decode throughput vs prompt_len (fixed batch size)
      (0,1) prefill/decode time ratio vs prompt_len (fixed batch size)
      (1,0) decode throughput vs batch size (fixed prompt_len)
      (1,1) KV cache saving ratio vs prompt_len (fixed batch size, same for all sparse variants theoretically)
    """
    kv_label = _kv_cache_label(kv_cache)

    rows_seq = [r for r in rows if r.get("seq_len") == seq_len]
    if not rows_seq:
        print(f"[warn] no rows for seq_len={seq_len}")
        return False

    # panel A/B/D use fixed batch size and vary prompt_len
    rows_pl = [r for r in rows_seq if r.get("batch_size") == fixed_batch_size and r.get("prompt_len") is not None]
    prompt_lens = sorted({r.get("prompt_len") for r in rows_pl if r.get("prompt_len") is not None})

    # panel C uses fixed prompt len and varies batch size
    rows_bs = [r for r in rows_seq if r.get("prompt_len") == fixed_prompt_len and r.get("batch_size") is not None]
    batch_sizes = sorted({r.get("batch_size") for r in rows_bs if r.get("batch_size") is not None})

    if not rows_pl:
        print(f"[warn] no rows for prompt sweep (seq_len={seq_len}, batch_size={fixed_batch_size})")
    if not rows_bs:
        print(f"[warn] no rows for batch sweep (seq_len={seq_len}, prompt_len={fixed_prompt_len})")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.5))
    ax_a, ax_b = axes[0, 0], axes[0, 1]
    ax_c, ax_d = axes[1, 0], axes[1, 1]

    # --- A: decode throughput vs prompt_len ---
    if rows_pl:
        by_mp_tps: dict[tuple[str, int], float] = {}
        by_mp_ratio: dict[tuple[str, int], float] = {}
        saving_by_prompt: dict[int, float] = {}

        for r in rows_pl:
            mt = r.get("model_type")
            pl = r.get("prompt_len")
            if not mt or pl is None:
                continue

            tps = r.get("decode_tokens_per_sec")
            pre_ms = r.get("prefill_avg_ms_per_batch")
            dec_ms = r.get("decode_avg_ms_per_run")
            if tps is not None:
                by_mp_tps[(mt, pl)] = float(tps)
            if pre_ms is not None and dec_ms is not None and float(dec_ms) > 0:
                by_mp_ratio[(mt, pl)] = float(pre_ms) / float(dec_ms)

            # saving ratio (same across methods in our theory; take first non-null)
            sr = r.get("kv_cache_saving_ratio")
            if sr is not None:
                saving_by_prompt.setdefault(pl, float(sr))

        for mt in models_sorted:
            xs: list[int] = []
            ys: list[float] = []
            for pl in prompt_lens:
                v = by_mp_tps.get((mt, pl))
                if v is None:
                    continue
                xs.append(pl)
                ys.append(v)
            if not xs:
                continue
            ax_a.plot(
                xs,
                ys,
                label=MODEL_TYPE_LABEL.get(mt, mt),
                color=colors.get(mt, "#333333"),
                linewidth=2.2,
                marker="o",
                markersize=4.5,
            )

        ax_a.set_title(f"Decode throughput vs prompt_len (seq_len={seq_len}, bs={fixed_batch_size}, kv_cache={kv_label})")
        ax_a.set_xlabel("Prompt length")
        ax_a.set_ylabel("Decode throughput (tokens/s)")
        ax_a.grid(True, linestyle="--", alpha=0.35)
        ax_a.legend(loc="best", frameon=False)

        # --- B: prefill/decode time ratio vs prompt_len ---
        for mt in models_sorted:
            xs: list[int] = []
            ys: list[float] = []
            for pl in prompt_lens:
                v = by_mp_ratio.get((mt, pl))
                if v is None:
                    continue
                xs.append(pl)
                ys.append(v)
            if not xs:
                continue
            ax_b.plot(
                xs,
                ys,
                label=MODEL_TYPE_LABEL.get(mt, mt),
                color=colors.get(mt, "#333333"),
                linewidth=2.2,
                marker="o",
                markersize=4.5,
            )

        ax_b.set_title("Prefill/Decode time ratio vs prompt_len")
        ax_b.set_xlabel("Prompt length")
        ax_b.set_ylabel("prefill_ms_per_batch / decode_ms_per_run")
        ax_b.grid(True, linestyle="--", alpha=0.35)

        # --- D: KV cache saving ratio vs prompt_len (theory) ---
        if saving_by_prompt:
            xs = sorted(saving_by_prompt.keys())
            ys = [saving_by_prompt[x] * 100.0 for x in xs]
            ax_d.plot(xs, ys, color="#111111", linewidth=2.4, marker="o", markersize=4.5)
            ax_d.set_ylim(0, 100)
        ax_d.set_title("KV-cache saving (theory) vs prompt_len")
        ax_d.set_xlabel("Prompt length")
        ax_d.set_ylabel("Saving ratio (%)")
        ax_d.grid(True, linestyle="--", alpha=0.35)

    # --- C: decode throughput vs batch size ---
    if rows_bs:
        by_mb: dict[tuple[str, int], float] = {}
        for r in rows_bs:
            mt = r.get("model_type")
            bs = r.get("batch_size")
            v = r.get("decode_tokens_per_sec")
            if mt and bs is not None and v is not None:
                by_mb[(mt, bs)] = float(v)

        for mt in models_sorted:
            xs: list[int] = []
            ys: list[float] = []
            for bs in batch_sizes:
                v = by_mb.get((mt, bs))
                if v is None:
                    continue
                xs.append(bs)
                ys.append(v)
            if not xs:
                continue
            ax_c.plot(
                xs,
                ys,
                label=MODEL_TYPE_LABEL.get(mt, mt),
                color=colors.get(mt, "#333333"),
                linewidth=2.2,
                marker="o",
                markersize=4.5,
            )
        ax_c.set_title(f"Decode throughput vs batch size (seq_len={seq_len}, prompt_len={fixed_prompt_len}, kv_cache={kv_label})")
        ax_c.set_xlabel("Batch size")
        ax_c.set_ylabel("Decode throughput (tokens/s)")
        ax_c.set_xticks(batch_sizes)
        ax_c.grid(True, linestyle="--", alpha=0.35)

    # overall title
    title_parts = []
    if step is not None:
        title_parts.append(f"step={step}")
    title_parts.append(f"seq_len={seq_len}")
    title_parts.append(f"kv_cache={kv_label}")
    fig.suptitle("Decode-focused efficiency dashboard (" + ", ".join(title_parts) + ")")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


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
    ap.add_argument(
        "--mode",
        type=str,
        default="decode",
        choices=["decode", "batch4", "all"],
        help=(
            "Visualization mode: "
            "'decode' plots decode throughput vs prompt_len and vs batch_size; "
            "'batch4' plots 2x2 (prefill/decode time+throughput) vs batch_size; "
            "'all' outputs everything."
        ),
    )
    ap.add_argument(
        "--decode_seq_len",
        type=int,
        default=4096,
        help="Seq length to use for decode-focused plots (default: 4096).",
    )
    ap.add_argument(
        "--fixed_batch_size",
        type=int,
        default=64,
        help="Batch size for decode throughput vs prompt_len plot (default: 64).",
    )
    ap.add_argument(
        "--fixed_prompt_len",
        type=int,
        default=3900,
        help="Prompt length for decode throughput vs batch size plot (default: 3900).",
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

    kv_label = _kv_cache_label(args.kv_cache)

    if args.mode in ("batch4", "all"):
        out_name = args.out_name
        if out_name is None:
            out_name = "efficiency_curves"
            if step is not None:
                out_name += f"_step{step}"
            out_name += f"_kvcache{kv_label}.png"
        out_path = os.path.join(out_dir, out_name)

        plot_batch_curves(
            plt=plt,
            out_path=out_path,
            step=step,
            kv_cache=args.kv_cache,
            seq_lens=seq_lens,
            models_sorted=models_sorted,
            colors=colors,
            batch_sizes=batch_sizes,
            by_ms=by_ms,
        )
        print(f"[plot] saved: {out_path}")

    if args.mode in ("decode", "all"):
        # Use raw filtered rows (already kv_cache filtered) for prompt_len plots
        decode_seq = args.decode_seq_len

        out_path = os.path.join(
            out_dir,
            f"decode_dashboard_seq{decode_seq}_bs{args.fixed_batch_size}_prompt{args.fixed_prompt_len}"
            + (f"_step{step}" if step is not None else "")
            + f"_kvcache{kv_label}.png",
        )
        ok = plot_decode_dashboard_2x2(
            plt=plt,
            out_path=out_path,
            step=step,
            kv_cache=args.kv_cache,
            seq_len=decode_seq,
            fixed_batch_size=args.fixed_batch_size,
            fixed_prompt_len=args.fixed_prompt_len,
            models_sorted=models_sorted,
            colors=colors,
            rows=filt,
        )
        if ok:
            print(f"[plot] saved: {out_path}")


if __name__ == "__main__":
    main()
