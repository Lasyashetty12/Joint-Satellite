"""
m4_plot_style.py - one consistent figure style for every Member 4 plot (paper + dashboard).

Colours come from a validated, colour-blind-checked categorical palette
(slots 1-3 pass CVD separation; slot 3 is low-contrast, so plots using it always
carry a legend or direct label). Sequential = one hue (blue, light->dark).
Diverging = blue <-> red with a neutral grey midpoint.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]     # blue, orange, aqua (fixed order, never cycled)
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SHADE = "#e9e8e4"                              # eclipse / gap shading
ALERT = "#e34948"                              # reserved: anomalies / alarms only

SEQUENTIAL = LinearSegmentedColormap.from_list("m4_blue", ["#eef4fc", "#2a78d6", "#0d3a73"])
DIVERGING = LinearSegmentedColormap.from_list("m4_div", ["#2a78d6", "#f0efec", "#e34948"])


def apply() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8, "legend.frameon": False,
        "lines.linewidth": 1.2, "axes.prop_cycle": matplotlib.cycler(color=SERIES),
        "font.size": 9,
    })


def save(fig, path) -> str:
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(path)
