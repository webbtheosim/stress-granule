"""Shared plotting house style for the unified ANALYSIS_SM figure pipeline.

Support module imported by the SM figure scripts (``sm_plotting.py``,
``sm_isolation.py``, ``sm_parameters.py``) to standardise the look of every
panel to match ``si_observables.py`` / ``biopolymer_analysis.py``: square
2.30 in axes, DSM=#40641b / NDSM=#bfe49b / SG=#808080, dpi 400, seaborn
ticks/white, inward ticks, and fixed-point (no sci-notation) axis helpers.

Provides: the class colors/constants, ``apply_rc`` (rc params), ``make_axes``
(fixed-size axes), ``ticks_in``, the 2-significant-figure helpers
(``fmt_2sf_no_sci``, ``apply_y_no_sci``, ``ceil_2sf``), and ``savefig``.

The correlation-corrected statistics (``correlated_stats``,
``combine_class_means``) live on the compute side in
``analysis/sm/sm_common.py`` and are re-exported here for backward
compatibility, so callers may keep doing ``from sm_common import
correlated_stats`` alongside the style helpers.

Not runnable as a script; import the helpers from the SM figure scripts.
"""
import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# --- re-export the compute side from analysis/sm -----------------------------
# This file lives at <repo>/plotting/sm/, so the analysis package is two levels
# up then into analysis/ (and analysis/sm/ for same-name resolution).
_ANALYSIS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "analysis")
_ANALYSIS_SM_DIR = os.path.join(_ANALYSIS_DIR, "sm")
for _p in (os.path.abspath(_ANALYSIS_SM_DIR), os.path.abspath(_ANALYSIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Import the compute module by file path to avoid the name clash with this
# module (both are called ``sm_common``).
import importlib.util as _ilu
_compute_path = os.path.join(os.path.abspath(_ANALYSIS_SM_DIR), "sm_common.py")
_spec = _ilu.spec_from_file_location("sm_common_compute", _compute_path)
_compute = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_compute)
correlated_stats = _compute.correlated_stats
combine_class_means = _compute.combine_class_means

# ----------------------------------------------------------------------------
# House style
# ----------------------------------------------------------------------------
COLOR_SG = "#808080"
COLOR_DSM = "#40641b"     # dark green (dissolving, Y_)
COLOR_NDSM = "#bfe49b"    # light green (non-dissolving, N_)
CLASS_COLOR = {"DSM": COLOR_DSM, "NDSM": COLOR_NDSM, "SG": COLOR_SG}

DPI = 400
RDP_FIGSIZE = (3.30, 3.30)
# 2.30 x 2.30 in axes; a touch more left/bottom margin for axis labels.
AX_RECT = [0.62 / 3.30, 0.58 / 3.30, 2.30 / 3.30, 2.30 / 3.30]


def apply_rc():
    """Apply the shared seaborn ticks/white theme and font/linewidth rc params."""
    sns.set_theme(style="ticks")
    sns.set_style("white")
    plt.rc("font", size=10)
    plt.rc("axes", titlesize=10)
    plt.rc("axes", labelsize=10)
    plt.rc("xtick", labelsize=8)
    plt.rc("ytick", labelsize=8)
    plt.rc("legend", fontsize=8)
    plt.rc("axes", linewidth=2)


def make_axes(figsize=RDP_FIGSIZE, rect=AX_RECT):
    """Create a figure with a single fixed-size axes and return ``(fig, ax)``."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(rect)
    return fig, ax


def ticks_in(ax):
    """Draw inward ticks on all four spines and raise the spines above the data."""
    ax.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True,
                   direction="in", length=4, width=2)
    for spine in ax.spines.values():
        spine.set_zorder(3)


def fmt_2sf_no_sci(v, _pos=0):
    """Format ``v`` to two significant figures as a plain decimal (no sci-notation).

    Suitable as a Matplotlib ``FuncFormatter`` tick callback; ``_pos`` is the
    unused tick-position argument. Returns "" for non-finite values.
    """
    if not np.isfinite(v):
        return ""
    if v == 0:
        return "0"
    exp = math.floor(math.log10(abs(v)))
    factor = 10 ** (1 - exp)
    rounded = round(v * factor) / factor
    decimals = max(0, 1 - exp)
    return f"{rounded:.{decimals}f}"


def apply_y_no_sci(ax, y_low, y_high, nticks=5):
    """Set y-limits, ``nticks`` evenly spaced y-ticks, and the no-sci formatter."""
    ax.set_ylim(y_low, y_high)
    ax.set_yticks(np.linspace(y_low, y_high, nticks))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_2sf_no_sci))


def ceil_2sf(x):
    """Round ``x`` away from zero to two significant figures (1.0 for 0/non-finite)."""
    if not np.isfinite(x) or x == 0:
        return 1.0
    sign = 1 if x > 0 else -1
    ax = abs(x)
    exp = math.floor(math.log10(ax))
    mant = ax / (10 ** exp)
    rounded = math.ceil(mant * 10) / 10
    if rounded >= 10:
        rounded /= 10
        exp += 1
    return sign * rounded * (10 ** exp)


def savefig(fig, path):
    """Save ``fig`` to ``path`` at the module DPI (creating parents), then close it."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path
