"""
plot_latency_comparison.py

Cross-architecture latency comparison for the PDF Sherlock dissertation.

Produces a single publication-quality grouped bar chart comparing the mean
per-query latency of the three retrieval modes (BM25, Dense/FAISS, Hybrid/RRF)
measured on two hardware configurations:

    * Apple M2 (macOS)        - the primary development machine
    * AMD Ryzen 5 (Ubuntu)    - the Linux reference box

A red dashed line at 50 ms marks the product-level latency target.

Run:
    python plot_latency_comparison.py
    python plot_latency_comparison.py --output eval/chart_latency_comparison.png
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Force the headless 'Agg' backend BEFORE pyplot is imported.
# macOS's default 'macosx' backend spins up AppKit on import, which SIGABRTs
# when this script runs in a background / non-GUI context. Agg is a pure
# CPU/PNG renderer - exactly what we need for a write-only PNG script.
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# Data (hard-coded per the dissertation's measurement runs).
# -----------------------------------------------------------------------------
MODE_LABELS: list[str] = ["BM25", "Dense (FAISS)", "Hybrid (RRF)"]

APPLE_M2_MS: list[float] = [4.45, 60.77, 18.60]
AMD_RYZEN_MS: list[float] = [2.91, 16.23, 16.96]

SLA_MS: float = 50.0

# Publication-friendly palette: a deep navy-blue for Apple and a teal for
# AMD. Both stay distinct in greyscale prints and sit on the colour-blind-
# safe side of the spectrum (no red/green collision with the SLA line).
APPLE_COLOUR = "#1F3A93"   # deep blue
AMD_COLOUR   = "#2A9D8F"   # teal
SLA_COLOUR   = "#C0392B"   # warm red for the SLA threshold


def _annotate_bars(ax: plt.Axes, bars, values: list[float]) -> None:
    """Place the numeric value just above each bar."""
    y_min, y_max = ax.get_ylim()
    pad = (y_max - y_min) * 0.015
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + pad,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            color="#222222",
        )


def build_figure() -> plt.Figure:
    """Construct the grouped bar chart and return the Figure."""
    x = np.arange(len(MODE_LABELS))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.5, 5.25))

    bars_apple = ax.bar(
        x - width / 2, APPLE_M2_MS, width,
        color=APPLE_COLOUR, edgecolor="white",
        label="Apple M2 (macOS)",
    )
    bars_amd = ax.bar(
        x + width / 2, AMD_RYZEN_MS, width,
        color=AMD_COLOUR, edgecolor="white",
        label="AMD Ryzen 5 (Ubuntu)",
    )

    # Headroom: make sure the tallest bar, the SLA line, and the numeric
    # labels above the bars all comfortably fit.
    top = max(max(APPLE_M2_MS), max(AMD_RYZEN_MS), SLA_MS) * 1.22
    ax.set_ylim(0, top)

    ax.axhline(
        SLA_MS,
        color=SLA_COLOUR,
        linestyle="--",
        linewidth=1.6,
        label=f"{SLA_MS:.0f}ms Target Threshold",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(MODE_LABELS, fontsize=12)
    ax.set_ylabel("Mean Latency (ms)", fontsize=13)
    ax.set_xlabel("Retrieval mode", fontsize=13)
    ax.set_title(
        "Cross-Architecture Query Latency by Retrieval Mode",
        fontsize=14, pad=12,
    )

    # Subtle horizontal grid only: keeps the chart quantitative without
    # adding visual noise across the bars.
    ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.4)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.legend(loc="upper left", frameon=False, fontsize=11)

    _annotate_bars(ax, bars_apple, APPLE_M2_MS)
    _annotate_bars(ax, bars_amd, AMD_RYZEN_MS)

    fig.tight_layout()
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the cross-architecture latency comparison chart."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("chart_latency_comparison.png"),
        help="Output PNG path (default: chart_latency_comparison.png)",
    )
    args = parser.parse_args()

    fig = build_figure()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
