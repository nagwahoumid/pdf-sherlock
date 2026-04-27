"""
plot_results.py - publication-quality charts for the PDF Sherlock evaluation.

Reads the three run-level CSVs produced by ``eval.py``
(``results_bm25.csv``, ``results_dense.csv``, ``results_hybrid.csv`` inside
``eval/``), aggregates the per-query metrics into per-mode means, and emits
three 300-DPI PNGs suitable for the master's dissertation:

    eval/chart_latency.png    - latency_mean_ms per mode with the 50 ms SLA
                                marked as a horizontal dashed line.
    eval/chart_precision.png  - mean Precision@k per mode.
    eval/chart_ranking.png    - mean MRR and mean NDCG@k side-by-side per mode.

Design notes
------------
- Uses ``matplotlib`` + ``seaborn`` with ``sns.set_theme(style="whitegrid")``
  for a clean, journal-ready look: subtle grid, no top/right spines,
  consistent Helvetica-ish default font stack.
- Palette uses a muted blue / slate / teal trio so the three retrieval
  systems stay visually distinct in both colour and greyscale prints:
      BM25   -> steel blue (#2F6690)   -> sparse / lexical
      Dense  -> neutral slate (#5B6B73) -> reference / non-sparse
      Hybrid -> teal          (#2A9D8F) -> the dissertation's headline system
- All figures are saved with ``plt.close(fig)`` and without ever calling
  ``plt.show()`` so the script is fully non-interactive and safe to run
  from a CI step or a notebook scheduler.

Usage
-----
    python plot_results.py
    python plot_results.py --input-dir eval --output-dir eval
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Force the headless 'Agg' backend BEFORE anything pulls in pyplot / seaborn.
#
# On macOS, matplotlib's default interactive backend is 'macosx', which spins
# up AppKit on import. When this script runs in a background process (CI,
# subprocess, notebook-less invocation) AppKit initialisation from a non-main
# thread or non-GUI session causes an immediate SIGABRT. 'Agg' is a pure
# CPU/PNG renderer with no GUI dependency and is the correct choice for a
# script that only writes PNG files.
#
# This block MUST stay above the ``import matplotlib.pyplot`` line: once
# pyplot loads, the backend is locked in.
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
except ImportError as exc:  # pragma: no cover - executed only when missing
    raise SystemExit(
        "plot_results.py needs matplotlib, seaborn, numpy and pandas.\n"
        "Install with:\n"
        "    pip install matplotlib seaborn\n"
        "or re-run: pip install -r requirements.txt"
    ) from exc


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ModeSpec:
    """Metadata for one retrieval system we want to plot."""

    key: str        # CSV filename stem (``bm25``, ``dense``, ``hybrid``)
    label: str      # Display name on axes / legend
    colour: str     # Hex colour for bars associated with this mode


# Ordering here determines the left-to-right order of bars in every chart.
MODES: List[ModeSpec] = [
    ModeSpec("bm25",   "BM25",   "#2F6690"),   # steel blue
    ModeSpec("dense",  "Dense",  "#5B6B73"),   # slate grey
    ModeSpec("hybrid", "Hybrid", "#2A9D8F"),   # teal
]

LATENCY_SLA_MS = 50.0  # Horizontal reference line on the latency chart.

# Global matplotlib font sizes. Centralised so every figure gets the same
# type treatment without scattering ``fontsize=...`` all over the place.
_RC_PARAMS = {
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.title_fontsize": 12,
    "figure.titlesize": 15,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


_REQUIRED_COLUMNS = {"P@k", "MRR", "Recall@k", "NDCG@k", "latency_mean_ms"}


def _load_mode_means(input_dir: Path) -> Dict[str, Dict[str, float]]:
    """
    Read the three results CSVs and return ``{mode_key: {metric: mean}}``.

    Each results CSV is expected to have one row per query with the columns
    listed in ``_REQUIRED_COLUMNS``. Missing files are a hard error so the
    dissertation figures are never produced from a partial run.
    """
    means: Dict[str, Dict[str, float]] = {}
    for mode in MODES:
        path = input_dir / f"results_{mode.key}.csv"
        if not path.exists():
            raise SystemExit(
                f"missing results file: {path}\n"
                "Run eval.py first to produce the per-mode CSVs."
            )
        df = pd.read_csv(path)
        missing = _REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise SystemExit(
                f"{path} is missing required columns: {sorted(missing)}\n"
                f"Found columns: {list(df.columns)}"
            )
        means[mode.key] = {
            metric: float(df[metric].mean()) for metric in _REQUIRED_COLUMNS
        }
        print(
            f"[plot] {mode.label:<6s} n={len(df):3d}  "
            f"P@k={means[mode.key]['P@k']:.3f}  "
            f"MRR={means[mode.key]['MRR']:.3f}  "
            f"Recall@k={means[mode.key]['Recall@k']:.3f}  "
            f"NDCG@k={means[mode.key]['NDCG@k']:.3f}  "
            f"latency={means[mode.key]['latency_mean_ms']:.1f} ms"
        )
    return means


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------


def _annotate_bars(ax: plt.Axes, fmt: str = "{:.2f}", offset: float = 0.01) -> None:
    """
    Place a small numeric label on top of every bar in ``ax``.

    ``offset`` is expressed as a fraction of the current y-axis span so the
    label sits a consistent visual distance above the bar regardless of the
    metric's magnitude (sub-unit for P@k/MRR, up-to-hundreds for latency).
    """
    y_min, y_max = ax.get_ylim()
    pad = (y_max - y_min) * offset
    for bar in ax.patches:
        height = bar.get_height()
        if np.isnan(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + pad,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=11,
            color="#333333",
        )


def _save(fig: plt.Figure, output_path: Path) -> None:
    """Save as 300 DPI PNG with a tight bounding box, then close the figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {output_path}")


# -----------------------------------------------------------------------------
# Individual charts
# -----------------------------------------------------------------------------


def plot_latency(means: Dict[str, Dict[str, float]], output_path: Path) -> None:
    """Bar chart of mean latency per mode, annotated with the 50 ms SLA line."""
    labels = [m.label for m in MODES]
    values = [means[m.key]["latency_mean_ms"] for m in MODES]
    colours = [m.colour for m in MODES]

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    bars = ax.bar(labels, values, color=colours, width=0.55, edgecolor="white")

    # Headroom: at least enough to clear both the tallest bar and the SLA line,
    # plus a margin for the numeric labels we place above each bar.
    top = max(max(values), LATENCY_SLA_MS) * 1.25
    ax.set_ylim(0, top)

    ax.axhline(
        LATENCY_SLA_MS,
        color="#C0392B",
        linestyle="--",
        linewidth=1.5,
        label=f"{LATENCY_SLA_MS:.0f} ms requirement",
    )
    # Inline annotation anchored to the right edge so the legend isn't
    # strictly necessary for readers skimming the figure.
    ax.text(
        len(labels) - 0.5,
        LATENCY_SLA_MS,
        f"  SLA {LATENCY_SLA_MS:.0f} ms",
        color="#C0392B",
        va="bottom",
        ha="right",
        fontsize=11,
    )

    ax.set_title("Mean Query Latency by Retrieval Mode")
    ax.set_xlabel("Retrieval mode")
    ax.set_ylabel("Mean latency (ms)")
    ax.legend(loc="upper left", frameon=False)
    _annotate_bars(ax, fmt="{:.1f} ms")

    _save(fig, output_path)
    _ = bars  # silence unused-variable lints


def plot_precision(means: Dict[str, Dict[str, float]], output_path: Path) -> None:
    """Bar chart of mean Precision@k per mode."""
    labels = [m.label for m in MODES]
    values = [means[m.key]["P@k"] for m in MODES]
    colours = [m.colour for m in MODES]

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.bar(labels, values, color=colours, width=0.55, edgecolor="white")
    ax.set_ylim(0, max(1.0, max(values) * 1.2))

    ax.set_title("Mean Precision@k by Retrieval Mode")
    ax.set_xlabel("Retrieval mode")
    ax.set_ylabel("Mean Precision@k")
    _annotate_bars(ax, fmt="{:.3f}")

    _save(fig, output_path)


def plot_ranking_quality(
    means: Dict[str, Dict[str, float]], output_path: Path
) -> None:
    """Grouped bar chart: mean MRR and mean NDCG@k side-by-side per mode."""
    labels = [m.label for m in MODES]
    x = np.arange(len(labels))
    width = 0.38

    mrr_vals = [means[m.key]["MRR"] for m in MODES]
    ndcg_vals = [means[m.key]["NDCG@k"] for m in MODES]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    # Paired colours: each mode keeps its base hue for MRR and a lighter
    # variant for NDCG, so you can tell both the mode and the metric apart
    # at a glance.
    base_colours = [m.colour for m in MODES]
    # The second hue per mode is the same colour at 55% opacity, applied via
    # ``matplotlib``'s RGBA tuple - equivalent to a tinted version.
    bars_mrr = ax.bar(
        x - width / 2, mrr_vals, width, color=base_colours, label="MRR",
        edgecolor="white",
    )
    bars_ndcg = ax.bar(
        x + width / 2, ndcg_vals, width,
        color=[_lighten(c, 0.45) for c in base_colours],
        label="NDCG@k", edgecolor="white",
    )

    # Extra headroom so the legend sits clear of the bar-top value labels.
    ax.set_ylim(0, max(1.15, max(mrr_vals + ndcg_vals) * 1.35))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Ranking Quality: MRR vs NDCG@k by Retrieval Mode")
    ax.set_xlabel("Retrieval mode")
    ax.set_ylabel("Score (higher is better)")
    ax.legend(loc="upper center", ncol=2, frameon=False)
    _annotate_bars(ax, fmt="{:.3f}")

    _save(fig, output_path)
    _ = bars_mrr, bars_ndcg


def _lighten(hex_colour: str, amount: float) -> str:
    """
    Return ``hex_colour`` blended toward white by ``amount`` (0..1).

    ``amount=0`` returns the original colour; ``amount=1`` returns white.
    Used to build a matching lighter shade for the secondary bars in the
    grouped ranking-quality chart so each mode is still identifiable.
    """
    amount = max(0.0, min(1.0, float(amount)))
    rgb = np.array(plt.matplotlib.colors.to_rgb(hex_colour))
    mixed = rgb + (1.0 - rgb) * amount
    return plt.matplotlib.colors.to_hex(mixed)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate 300-DPI publication charts from PDF Sherlock's "
            "per-mode evaluation CSVs."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("eval"),
        help="Directory containing results_{bm25,dense,hybrid}.csv (default: eval)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval"),
        help="Directory to write PNG charts into (default: eval)",
    )
    args = parser.parse_args()

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(_RC_PARAMS)

    means = _load_mode_means(args.input_dir)

    plot_latency(means, args.output_dir / "chart_latency.png")
    plot_precision(means, args.output_dir / "chart_precision.png")
    plot_ranking_quality(means, args.output_dir / "chart_ranking.png")

    print("\n[plot] all charts written successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
