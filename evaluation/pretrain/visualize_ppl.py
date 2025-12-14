import argparse
import csv
import os
from typing import Any


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


def _fmt(v: Any, ndigits: int = 4) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.{ndigits}f}"
    return str(v)


def read_rows(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # normalize types for common fields
            r["step"] = _as_int(r.get("step", ""))
            r["seq_len"] = _as_int(r.get("seq_len", ""))
            r["batch_size"] = _as_int(r.get("batch_size", ""))
            r["use_kv_cache"] = _as_int(r.get("use_kv_cache", ""))
            r["ppl_id"] = _as_float(r.get("ppl_id", ""))
            r["ppl_ood"] = _as_float(r.get("ppl_ood", ""))
            r["nll_id"] = _as_float(r.get("nll_id", ""))
            r["nll_ood"] = _as_float(r.get("nll_ood", ""))
            r["tokens_id"] = _as_int(r.get("tokens_id", ""))
            r["tokens_ood"] = _as_int(r.get("tokens_ood", ""))
            r["ood_examples"] = _as_int(r.get("ood_examples", ""))
            rows.append(r)
    return rows


def to_markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines: list[str] = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(_fmt(x) for x in r) + " |")
    return "\n".join(lines)


MODEL_TYPE_LABEL = {
    "full": "Full attention",
    "sparse_conv": "Sparse attention + ConvLinear",
    "sparse_mlp": "Sparse attention + GroupedMLP",
    "sparse_attn": "Sparse attention + AttentionPool",
    "sparse_mean": "Sparse attention + MeanPool",
}

# Preferred plotting order
MODEL_ORDER = ["full", "sparse_conv", "sparse_mlp", "sparse_attn", "sparse_mean"]


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
    # stable palette (tab10-ish, manually specified for consistency)
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


def _plot_combined_2x2(
    out_dir: str,
    step: int | None,
    kv_cache: int | None,
    seq_lens: list[int],
    models: list[str],
    values: dict[tuple[str, int, str], float],
):
    """
    One figure with 2x2 subplots:
      rows: ID / OOD
      cols: seq_len (sorted)

    Each subplot: x-axis = methods (full + sparse variants), colored by method.
    """
    plt = _try_import_matplotlib()
    if plt is None:
        return

    os.makedirs(out_dir, exist_ok=True)

    seq_lens_sorted = sorted(seq_lens)
    if len(seq_lens_sorted) == 0:
        return

    models_sorted = _method_order(models)
    colors = _method_colors(models_sorted)

    ncols = min(2, len(seq_lens_sorted))
    nrows = 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(13, 7), sharey="row")
    if ncols == 1:
        # axes becomes shape (2,), normalize to (2,1)
        axes = [[axes[0]], [axes[1]]]

    def _subtitle(extra: str) -> str:
        parts = []
        if step is not None:
            parts.append(f"step={step}")
        if kv_cache is not None:
            parts.append(f"use_kv_cache={kv_cache}")
        if parts:
            return extra + " (" + ", ".join(parts) + ")"
        return extra

    for ci, sl in enumerate(seq_lens_sorted[:ncols]):
        # ID row
        ax = axes[0][ci]
        vals = [values.get((m, sl, "ppl_id")) for m in models_sorted]
        x = list(range(len(models_sorted)))
        bars = ax.bar(x, [v if v is not None else 0.0 for v in vals], color=[colors[m] for m in models_sorted])
        ax.set_title(_subtitle(f"ID (enwik8) @ seq_len={sl}"))
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_TYPE_LABEL.get(m, m) for m in models_sorted], rotation=18, ha="right")
        ax.set_ylabel("Perplexity (lower is better)")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        for b, v in zip(bars, vals):
            if v is None:
                continue
            ax.text(b.get_x() + b.get_width() / 2.0, b.get_height(), f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        # OOD row
        ax = axes[1][ci]
        vals = [values.get((m, sl, "ppl_ood")) for m in models_sorted]
        x = list(range(len(models_sorted)))
        bars = ax.bar(x, [v if v is not None else 0.0 for v in vals], color=[colors[m] for m in models_sorted])
        ax.set_title(_subtitle(f"OOD (cs441) @ seq_len={sl}"))
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_TYPE_LABEL.get(m, m) for m in models_sorted], rotation=18, ha="right")
        ax.set_ylabel("Perplexity (lower is better)")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        for b, v in zip(bars, vals):
            if v is None:
                continue
            ax.text(b.get_x() + b.get_width() / 2.0, b.get_height(), f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    # One legend for methods (colors)
    handles = [plt.Line2D([0], [0], color=colors[m], lw=8) for m in models_sorted]
    labels = [MODEL_TYPE_LABEL.get(m, m) for m in models_sorted]
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=(0, 0.05, 1, 1))

    out_name = "ppl_bars"
    if step is not None:
        out_name += f"_step{step}"
    if kv_cache is not None:
        out_name += f"_kvcache{kv_cache}"
    out_name += ".png"
    out_path = os.path.join(out_dir, out_name)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Visualize ppl_step*.csv as a markdown table")
    ap.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to ppl_step*.csv generated by evaluation/perplexity.py",
    )
    ap.add_argument(
        "--step",
        type=int,
        default=None,
        help="Filter by training step (default: use the max step in the CSV)",
    )
    ap.add_argument(
        "--kv_cache",
        type=int,
        default=None,
        choices=[0, 1],
        help="Filter by use_kv_cache (0/1). Default: no filter",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="If set, save bar charts (PNG) into this directory. Default: alongside the CSV file.",
    )
    ap.add_argument(
        "--plot_mode",
        type=str,
        default="combined",
        choices=["combined", "none"],
        help="Plot mode: 'combined' saves one 2x2 figure; 'none' disables plots.",
    )
    args = ap.parse_args()

    rows = read_rows(args.csv)
    if not rows:
        raise SystemExit(f"No rows found in: {args.csv}")

    steps = sorted({r.get("step") for r in rows if r.get("step") is not None})
    step = args.step if args.step is not None else (steps[-1] if steps else None)

    filt: list[dict[str, Any]] = []
    for r in rows:
        if step is not None and r.get("step") != step:
            continue
        if args.kv_cache is not None and r.get("use_kv_cache") != args.kv_cache:
            continue
        filt.append(r)

    if not filt:
        raise SystemExit("No rows after filtering; check --step / --kv_cache")

    seq_lens = sorted({r.get("seq_len") for r in filt if r.get("seq_len") is not None})
    model_types = sorted({r.get("model_type") for r in filt if r.get("model_type")})

    # map (model_type, seq_len) -> metrics
    m: dict[tuple[str, int], dict[str, Any]] = {}
    for r in filt:
        mt = r.get("model_type")
        sl = r.get("seq_len")
        if not mt or sl is None:
            continue
        m[(mt, sl)] = r

    # baseline (full) per seq_len for delta
    base: dict[int, dict[str, float]] = {}
    for sl in seq_lens:
        r = m.get(("full", sl))
        if r and r.get("ppl_id") is not None and r.get("ppl_ood") is not None:
            base[sl] = {"ppl_id": float(r["ppl_id"]), "ppl_ood": float(r["ppl_ood"])}

    headers = ["model_type"]
    for sl in seq_lens:
        headers += [
            f"ppl_id@{sl}",
            f"Δid@{sl}",
            f"ppl_ood@{sl}",
            f"Δood@{sl}",
        ]

    out_rows: list[list[Any]] = []
    # nicer ordering and names in table output
    ordered_model_types = [m for m in MODEL_ORDER if m in model_types] + [
        m for m in model_types if m not in MODEL_ORDER
    ]

    for mt in ordered_model_types:
        row: list[Any] = [MODEL_TYPE_LABEL.get(mt, mt)]
        for sl in seq_lens:
            r = m.get((mt, sl))
            ppl_id = r.get("ppl_id") if r else None
            ppl_ood = r.get("ppl_ood") if r else None

            d_id = None
            d_ood = None
            if ppl_id is not None and sl in base:
                d_id = float(ppl_id) - base[sl]["ppl_id"]
            if ppl_ood is not None and sl in base:
                d_ood = float(ppl_ood) - base[sl]["ppl_ood"]

            row += [ppl_id, d_id, ppl_ood, d_ood]
        out_rows.append(row)

    print(to_markdown_table(headers, out_rows))

    # ---- bar charts ----
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(args.csv)) or "."

    if args.plot_mode != "none":
        # collect values for combined figure
        values: dict[tuple[str, int, str], float] = {}
        for mt in ordered_model_types:
            for sl in seq_lens:
                r = m.get((mt, sl))
                if not r:
                    continue
                if r.get("ppl_id") is not None:
                    values[(mt, sl, "ppl_id")] = float(r["ppl_id"])
                if r.get("ppl_ood") is not None:
                    values[(mt, sl, "ppl_ood")] = float(r["ppl_ood"])

        _plot_combined_2x2(
            out_dir=out_dir,
            step=step,
            kv_cache=args.kv_cache,
            seq_lens=seq_lens,
            models=ordered_model_types,
            values=values,
        )


if __name__ == "__main__":
    main()
