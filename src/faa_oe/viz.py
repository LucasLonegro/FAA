from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# Fixed categorical order (validated for CVD on adjacent pairs); never cycle past 8.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER = "#c3c2b7"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQUENTIAL = LinearSegmentedColormap.from_list("faa_blue", BLUE_RAMP)


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "axes.prop_cycle": mpl.cycler(color=SERIES),
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "lines.linewidth": 1.6,
            "legend.frameon": False,
            "figure.dpi": 110,
        }
    )


def small_multiples(n: int, ncols: int = 3, panel: tuple[float, float] = (4.2, 1.9), **kw) -> tuple[plt.Figure, list]:
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel[0] * ncols, panel[1] * nrows), squeeze=False, **kw)
    flat = list(axes.flat)
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


def stacked_columns(ax, x, layers: list, colors: list[str], width: float, labels: list[str] | None = None) -> None:
    """Stacked columns with a thin surface-coloured gap between segments and between neighbours."""
    bottom = None
    for i, (values, color) in enumerate(zip(layers, colors)):
        ax.bar(x, values, width=width, bottom=bottom, color=color, edgecolor=SURFACE, linewidth=0.8,
               label=labels[i] if labels else None)
        bottom = values if bottom is None else [b + v for b, v in zip(bottom, values)]
