# Canonical owner: closed-loop-dbs-bench
"""Unified figure style for all Python-generated plots.

Single source of truth for colours, sizes, fonts, and formatting.
Import this module and call setup() before creating any figures.

Style targets CDC25 publication quality: no grid, minimal spines,
clean violins with inner box plots, smooth timeseries.

Usage:
    from dbs_bench.analysis.figure_style import setup, PALETTE, CONTROLLERS, SCENARIOS
    setup()
    fig, ax = plt.subplots(figsize=FIGSIZE["single"])
    ax.plot(x, y, color=CONTROLLERS["DCNN-MPC"]["color"],
            linestyle=CONTROLLERS["DCNN-MPC"]["ls"])
"""
from __future__ import annotations

from typing import Dict

import numpy as np

# ---------------------------------------------------------------------------
# Colour palette  (Tableau 10 Muted — colourblind-friendly)
#
# Mapping to semantic roles keeps every script consistent.
# TikZ equivalents are defined in report/figures/colors.tex
# ---------------------------------------------------------------------------

PALETTE = {
    "blue":    "#4E79A7",
    "orange":  "#F28E2B",
    "red":     "#E15759",
    "teal":    "#76B7B2",
    "green":   "#59A14F",
    "gold":    "#EDC948",
    "purple":  "#B07AA1",
    "brown":   "#9C755F",
    "grey":    "#BAB0AC",
    "black":   "#333333",
}

# Lighter tints for fills/shading (20% opacity equivalent on white)
PALETTE_LIGHT = {
    "blue":    "#D3DDE8",
    "orange":  "#FDE3C6",
    "red":     "#F6D1D2",
    "teal":    "#DBE9E8",
    "green":   "#D2E5CF",
    "gold":    "#FBF4D0",
    "purple":  "#E8DBEA",
    "brown":   "#E2DAD6",
}

# ---------------------------------------------------------------------------
# Semantic colour assignments
# ---------------------------------------------------------------------------

# Controller comparison plots
CONTROLLERS: Dict[str, dict] = {
    "BangBang":  {"color": PALETTE["blue"],    "ls": "-",  "marker": "o", "label": "Bang-Bang"},
    "PI":        {"color": PALETTE["orange"],   "ls": "--", "marker": "s", "label": "PI"},
    "Multi-step ARX": {"color": PALETTE["green"], "ls": "-.", "marker": "^", "label": "Multi-step ARX"},
    "DCNN-MPC":  {"color": PALETTE["red"],      "ls": "-",  "marker": "D", "label": "DCNN TMPC"},
    "Residual-DCNN-MPC": {"color": PALETTE["red"], "ls": "--", "marker": "d", "label": "Residual DCNN TMPC"},
    "Koopman-MPC": {"color": PALETTE["purple"], "ls": "-.", "marker": "P", "label": "Koopman MPC"},
}
CONTROLLER_ORDER = ["BangBang", "PI", "Multi-step ARX", "DCNN-MPC", "Residual-DCNN-MPC", "Koopman-MPC"]

# Drift-scenario plots
SCENARIOS: Dict[str, dict] = {
    "offset":     {"color": PALETTE["blue"],    "ls": "-",  "marker": "o", "label": "Offset"},
    "gain":       {"color": PALETTE["orange"],  "ls": "--", "marker": "s", "label": "Gain"},
    "frequency":  {"color": PALETTE["green"],   "ls": "-.", "marker": "^", "label": "Frequency"},
    "stim":       {"color": PALETTE["red"],     "ls": "-",  "marker": "D", "label": "Stim efficacy"},
    "medication": {"color": PALETTE["purple"],  "ls": "--", "marker": "v", "label": "Medication"},
}

# General-purpose semantic colours (reference lines, regions, etc.)
SEMANTIC = {
    "threshold":  PALETTE["grey"],       # beta threshold line
    "nominal":    PALETTE["blue"],       # nominal trajectory
    "tube":       PALETTE_LIGHT["blue"], # tube/confidence region fill
    "control":    PALETTE["teal"],       # control input signal
    "reference":  PALETTE["black"],      # setpoint / reference
}

# ---------------------------------------------------------------------------
# Line styles  (combine colour + linestyle + marker for accessibility)
# ---------------------------------------------------------------------------

LINESTYLES = ["-", "--", "-.", ":"]
MARKERS    = ["o", "s", "^", "D", "v", "P"]

# ---------------------------------------------------------------------------
# Figure sizing
#
# Report uses \geometry{margin=20mm} on A4 -> text width ~170 mm = 6.69 in.
# Golden ratio height: width / 1.618.
# ---------------------------------------------------------------------------

_TEXT_WIDTH_IN = 6.69   # A4 with 20 mm margins

FIGSIZE = {
    "single":    (_TEXT_WIDTH_IN, _TEXT_WIDTH_IN / 1.618),        # full-width, golden ratio
    "half":      (_TEXT_WIDTH_IN * 0.48, 2.6),                    # side-by-side subfigure (0.48\textwidth)
    "wide":      (_TEXT_WIDTH_IN, 3.2),                           # full-width, shorter
    "tall":      (_TEXT_WIDTH_IN, _TEXT_WIDTH_IN),                 # square-ish (multi-panel)
    "multi_2x1": (_TEXT_WIDTH_IN, _TEXT_WIDTH_IN * 0.55),         # 2 stacked subplots
    "multi_2x2": (_TEXT_WIDTH_IN, _TEXT_WIDTH_IN * 0.85),         # 2x2 grid
    "half_tall": (_TEXT_WIDTH_IN * 0.48, _TEXT_WIDTH_IN * 0.48 * 1.6),  # tall subfigure (0.48\textwidth)
}

# ---------------------------------------------------------------------------
# Matplotlib rc configuration — CDC25-inspired clean style
# ---------------------------------------------------------------------------

_RC_PARAMS = {
    # --- Typography (match XCharter in report) ---
    "font.family":      "serif",
    "font.serif":       ["XCharter", "DejaVu Serif"],
    "font.size":        9,
    "axes.labelsize":   10,
    "axes.titlesize":   11,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,

    # --- Lines & markers ---
    "lines.linewidth":  1.5,
    "lines.markersize": 4,
    "lines.solid_capstyle": "round",
    "patch.linewidth":  0.5,

    # --- Axes: clean, minimal (no grid, no top/right spines) ---
    "axes.linewidth":   0.6,
    "axes.edgecolor":   "#333333",
    "axes.grid":        False,
    "axes.axisbelow":   True,
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.prop_cycle":  None,  # set dynamically in setup()

    # --- Ticks: subtle, outward ---
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size":  3.5,
    "ytick.major.size":  3.5,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,

    # --- Legend: clean, subtle frame ---
    "legend.frameon":     True,
    "legend.fancybox":    False,
    "legend.edgecolor":   "#DDDDDD",
    "legend.framealpha":  0.95,
    "legend.borderpad":   0.4,
    "legend.handlelength": 1.5,

    # --- Figure ---
    "figure.figsize":   FIGSIZE["single"],
    "figure.dpi":       150,
    "figure.facecolor":  "white",
    "figure.constrained_layout.use": True,

    # --- Saving ---
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.03,
    "savefig.facecolor":  "white",
}


def setup():
    """Apply the unified figure style to matplotlib.

    Call once at the top of any plotting script:
        from dbs_bench.analysis.figure_style import setup
        setup()
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from cycler import cycler

    # Build colour cycle from the palette
    cycle_colors = [
        PALETTE["blue"], PALETTE["orange"], PALETTE["red"],
        PALETTE["teal"], PALETTE["green"], PALETTE["gold"],
        PALETTE["purple"], PALETTE["brown"],
    ]
    _RC_PARAMS["axes.prop_cycle"] = cycler(color=cycle_colors)

    plt.rcParams.update(_RC_PARAMS)


def save(fig, path, *, formats=("png", "pdf")):
    """Save figure in multiple formats with consistent settings.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    path : str or Path
        Base path without extension (e.g. "figures/fig1_comparison").
    formats : tuple of str
        File extensions to save. Default: PNG for preview + PDF for report.
    """
    from pathlib import Path as _Path
    p = _Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(p.with_suffix(f".{fmt}"))


# ---------------------------------------------------------------------------
# Helper: CDC25-style violin with inner box plot
# ---------------------------------------------------------------------------

def violin(ax, data, positions, colors, labels=None, width=0.7):
    """Draw CDC25-style violins with inner quartile box plots.

    Parameters
    ----------
    ax : matplotlib Axes
    data : list of arrays
        One array per violin.
    positions : array-like
        X positions for each violin.
    colors : list of str
        Fill colour for each violin.
    labels : list of str, optional
        X-tick labels.
    width : float
        Violin width.
    """
    parts = ax.violinplot(
        data,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=width,
    )

    # Style violin bodies
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[i % len(colors)])
        body.set_edgecolor(PALETTE["black"])
        body.set_linewidth(0.6)
        body.set_alpha(0.75)

    # Inner summary: thin vertical line (whiskers) + 3 horizontal bars
    # (min, median, max).  Matches CDC25 Fig 5 style — no white box.
    for i, d in enumerate(data):
        pos = positions[i]
        med = np.median(d)
        lo = np.min(d)
        hi = np.max(d)

        # Thin vertical whisker spanning full range
        ax.plot([pos, pos], [lo, hi], color=PALETTE["black"], lw=0.8, zorder=3)
        # Median bar (wider, thicker)
        bar_w = width * 0.15
        ax.plot([pos - bar_w, pos + bar_w], [med, med],
                color=PALETTE["black"], lw=2.0, solid_capstyle="butt", zorder=4)
        # Min / max caps (narrower)
        cap_w = width * 0.10
        ax.plot([pos - cap_w, pos + cap_w], [lo, lo],
                color=PALETTE["black"], lw=1.0, solid_capstyle="butt", zorder=3)
        ax.plot([pos - cap_w, pos + cap_w], [hi, hi],
                color=PALETTE["black"], lw=1.0, solid_capstyle="butt", zorder=3)

    if labels is not None:
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)

    return parts


# ---------------------------------------------------------------------------
# Helper: smooth timeseries for cleaner plots
# ---------------------------------------------------------------------------

def smooth(y, window=5):
    """Apply rolling mean to a signal for cleaner plotting.

    Parameters
    ----------
    y : array-like
        Input signal.
    window : int
        Smoothing window size (samples). Use 1 for no smoothing.

    Returns
    -------
    np.ndarray
        Smoothed signal (same length as input).
    """
    if window <= 1:
        return np.asarray(y)
    kernel = np.ones(window) / window
    # Pad to preserve length
    y = np.asarray(y, dtype=float)
    padded = np.pad(y, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")
