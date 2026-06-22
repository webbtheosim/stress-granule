"""
Per-species biopolymer overlay plots and contact maps (pipeline Step 4).

Generates the per-biopolymer figures for the stress-granule paper from the
time-averaged outputs of the upstream pipeline. For each system category (SG
control, DSM, NDSM) it produces:
  - Radial density profiles (RDP) per biopolymer / RNA component, with sigmoid
    fits (via RDP_PLOT.RDP), plus DSM-SG / NDSM-SG difference and offset overlays.
  - Residue- and amino-acid-level contact-probability maps (CPMs) in several
    normalizations: raw K_chain, P_contact, C_total, contact membership, and
    contact enrichment (with log and z-score companions, using
    CONTACT_NORMALIZATION for chain-pair counts).
  - Small-molecule (SM) contact difference and membership/enrichment maps.
  - Three-way (SG vs DSM vs NDSM) and DSM-vs-NDSM component comparison overlays.

Pipeline role:
    Runs after average_simulations.py / max_cluster.py / system_analysis.py,
    reading their averaged ANALYSIS_*_AVE products; emits only figures (and the
    CSVs backing some contact maps).

Key inputs (read from the temperature root used as CWD / analysis root):
    - ANALYSIS_{SG,DSM,NDSM}_AVE/Density_Profile_*, PCA_*, Cluster_*, contact CSVs
    - CLI flags: --path, --folder, --T, --dt, --tmin, --tmax, [--plot-only]

Key outputs (under {analysis_root}/FIGURES/):
    - FIGURES/RDP/*                         density-profile overlays + diffs
    - FIGURES/RESIDUE_CONTACT_MAPS/*        residue/acid/SM contact maps

CLI:
    python biopolymer_analysis.py --path TEMP_300 --folder CLASSIFY --T 300 \
        --tmin 50 --dt 50 --tmax 2000
"""

import argparse
import os
import sys

# Allow this renderer, when run from the plotting/ folder, to import the shared
# compute/support modules that live in analysis/ (RDP_PLOT, CONTACT_NORMALIZATION).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "analysis"))

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import UnivariateSpline

import contact_normalization as CN
from rdp_plot import rdp

plt.rcParams["figure.max_open_warning"] = 0

RDP_XLIM = (0, 500)
RDP_LINEWIDTH = 2.0
RDP_MARKER_SIZE = 40
RDP_MARKER_EDGEWIDTH = 1.0
RDP_ERRORBAR_WIDTH = 1.0
RDP_POINT_STRIDE = 2
RESIDUE_CONTACT_MAP_FIGSIZE = (3.35, 3.00)
ACID_CONTACT_MAP_FIGSIZE = (3.35, 3.00)
# Fixed layout for square contact maps.
# figsize (3.35, 3.00) gives axes 2.00x2.00 in with left=0.62, right=0.73 in
# and top=bottom=0.50 in (axes vertically centered).
# Same axes size as contact_maps.py domain contact maps for panel assembly.
_CONTACT_AX_RECT = [0.62 / 3.35, 0.50 / 3.00, 2.00 / 3.35, 2.00 / 3.00]
# RDP plot layout: same 2.40×2.40 in axes, figsize (3.50, 2.92).
# equal left=right=0.55in margins → axes perfectly centred horizontally.
RDP_PLOT_FIGSIZE = (3.20, 3.20)
_RDP_AX_RECT = [0.50 / 3.20, 0.50 / 3.20, 2.20 / 3.20, 2.20 / 3.20]   # 2.20x2.20 in, centered both axes
_CONTACT_CBAR_PAD_IN = 0.035
_CONTACT_CBAR_W_IN = 0.13  # 2x thicker than 0.065
# Dynamic layout constants for SM diff contact maps (residue 7×n and acid 24×n).
_SM_BOTTOM_MARGIN_IN = 0.45 # space for x-axis labels
_SM_TOP_MARGIN_IN = 0.25  # room for cbar x10^n exponent label
_SM_CBAR_PAD_IN = 0.04
_SM_CBAR_W_IN = 0.10


def _safe_divide(numerator, denominator, fill_value=0.0):
    """Elementwise numerator/denominator, returning *fill_value* where the denominator is 0 or non-finite."""
    num_arr, den_arr = np.broadcast_arrays(
        np.asarray(numerator, dtype=float),
        np.asarray(denominator, dtype=float),
    )
    out = np.full(num_arr.shape, fill_value, dtype=float)
    return np.divide(num_arr, den_arr, out=out, where=np.isfinite(den_arr) & (den_arr != 0))


def _sanitize_heatmap_array(values, fill_value=0.0):
    """Return a float copy of *values* with NaN/+-inf replaced by *fill_value* for safe plotting."""
    arr = np.asarray(values, dtype=float).copy()
    return np.nan_to_num(arr, nan=fill_value, posinf=fill_value, neginf=fill_value)


def _round_1sig_floor(x):
    """Round x down (toward -inf) to 1 significant figure."""
    if x == 0:
        return 0.0
    exp = int(np.floor(np.log10(abs(x))))
    factor = 10.0 ** exp
    return np.floor(x / factor) * factor


def _round_1sig_ceil(x):
    """Round x up (toward +inf) to 1 significant figure."""
    if x == 0:
        return 0.0
    exp = int(np.floor(np.log10(abs(x))))
    factor = 10.0 ** exp
    return np.ceil(x / factor) * factor


def _setup_cbar(ax, linewidth=2):
    """Format colorbar: border, 5 ticks, rounded limits, math-text sci notation."""
    cbar = ax.collections[0].colorbar
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(linewidth)
    vmin, vmax = ax.collections[0].get_clim()
    tick_lo = _round_1sig_floor(vmin)
    tick_hi = _round_1sig_ceil(vmax)
    if abs(tick_hi - tick_lo) < 1e-30:
        tick_hi = tick_lo + 1.0
    ticks = np.linspace(tick_lo, tick_hi, 5)
    cbar.set_ticks(ticks)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter.set_useMathText(True)
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.yaxis.set_offset_position('right')
    cbar.ax.yaxis.get_offset_text().set_fontsize(8)
    for spine_name in ['top', 'bottom', 'left', 'right']:
        cbar.ax.spines[spine_name].set_visible(True)
        cbar.ax.spines[spine_name].set_color('black')
        cbar.ax.spines[spine_name].set_linewidth(linewidth)
    return cbar


def _round_2sig_floor(x):
    """Round x down (toward -inf) to 2 significant figures (0 maps to 0.0)."""
    if x == 0:
        return 0.0
    exp = int(np.floor(np.log10(abs(x))))
    factor = 10.0 ** (exp - 1)
    return np.floor(x / factor) * factor


def _round_2sig_ceil(x):
    """Round x up (toward +inf) to 2 significant figures (0 maps to 0.0)."""
    if x == 0:
        return 0.0
    exp = int(np.floor(np.log10(abs(x))))
    factor = 10.0 ** (exp - 1)
    return np.ceil(x / factor) * factor


def _setup_cbar_constrained(fig, ax, n_ticks=5, linewidth=2):
    """Attach a divider-appended colorbar to *ax* with 2-sig rounded limits and a shared x10^n exponent label."""
    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("right", size="8%", pad=0.08)
    mappable = ax.collections[0]
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    clim = mappable.get_clim()
    tick_lo = _round_2sig_floor(clim[0])
    tick_hi = _round_2sig_ceil(clim[1])
    if abs(tick_hi - tick_lo) < 1e-30:
        tick_hi = tick_lo + 1.0
    ticks = np.linspace(tick_lo, tick_hi, n_ticks)
    mappable.set_clim(tick_lo, tick_hi)
    cbar.set_ticks(ticks)
    max_abs_tick = max(abs(tick_lo), abs(tick_hi))
    if max_abs_tick > 0:
        common_exp = int(np.floor(np.log10(max_abs_tick)))
    else:
        common_exp = 0
    scale = 10.0 ** common_exp
    cbar.ax.yaxis.set_major_formatter(
        FuncFormatter(lambda val, pos, s=scale: f"{val / s:.1f}")
    )
    cbar.ax.text(
        0.5, 1.02, f"$\\times10^{{{common_exp}}}$",
        transform=cbar.ax.transAxes, ha='center', va='bottom', fontsize=8,
    )
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(linewidth)
    cbar.ax.tick_params(labelsize=8, width=linewidth, length=4)
    for spine_name in ['top', 'bottom', 'left', 'right']:
        cbar.ax.spines[spine_name].set_visible(True)
        cbar.ax.spines[spine_name].set_color('black')
        cbar.ax.spines[spine_name].set_linewidth(linewidth)
    return cbar


def _setup_cbar_fixed(fig, ax, ax_rect, n_ticks=5, linewidth=2,
                      cbar_pad_in=None, cbar_w_in=None):
    """Colorbar at a fixed physical width, independent of axes size."""
    pad = cbar_pad_in if cbar_pad_in is not None else _CONTACT_CBAR_PAD_IN
    w = cbar_w_in if cbar_w_in is not None else _CONTACT_CBAR_W_IN
    fig_w = fig.get_figwidth()
    cbar_left = ax_rect[0] + ax_rect[2] + pad / fig_w
    cbar_w = w / fig_w
    cbar_ax = fig.add_axes([cbar_left, ax_rect[1], cbar_w, ax_rect[3]])
    mappable = ax.collections[0]
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    clim = mappable.get_clim()
    tick_lo = _round_2sig_floor(clim[0])
    tick_hi = _round_2sig_ceil(clim[1])
    if abs(tick_hi - tick_lo) < 1e-30:
        tick_hi = tick_lo + 1.0
    ticks = np.linspace(tick_lo, tick_hi, n_ticks)
    mappable.set_clim(tick_lo, tick_hi)
    cbar.set_ticks(ticks)
    max_abs_tick = max(abs(tick_lo), abs(tick_hi))
    if max_abs_tick > 0:
        common_exp = int(np.floor(np.log10(max_abs_tick)))
    else:
        common_exp = 0
    scale = 10.0 ** common_exp
    cbar.ax.yaxis.set_major_formatter(
        FuncFormatter(lambda val, pos, s=scale: f"{val / s:.1f}")
    )
    cbar.ax.text(
        0.5, 1.02, f"$\\times10^{{{common_exp}}}$",
        transform=cbar.ax.transAxes, ha='center', va='bottom', fontsize=8,
    )
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(linewidth)
    cbar.ax.tick_params(labelsize=8, width=linewidth, length=4)
    for spine_name in ['top', 'bottom', 'left', 'right']:
        cbar.ax.spines[spine_name].set_visible(True)
        cbar.ax.spines[spine_name].set_color('black')
        cbar.ax.spines[spine_name].set_linewidth(linewidth)
    return cbar


_SM_AXIS_HEIGHT_IN = 2.00   # all SM/acid heatmaps share this axis height (matches full contact maps)
_SM_SIDE_MARGIN_IN = 0.80   # symmetric L/R margin so axes is centered horizontally
                            # wide enough for cbar tick labels + x10^n exponent on right
                            # (residue/acid name labels fit on left)
_SM_1D_AXIS_WIDTH_IN = _SM_AXIS_HEIGHT_IN / 7.0   # all 1D maps share this axis width (= residue 1D)


def _sm_compute_layout(n_rows, n_cols):
    """Compute (figsize, ax_rect) for SM/acid heatmaps.
    Axis height fixed at 2.40 in (matches RDP / contact map axes).
    For multi-column maps: cell side = 2.40 / n_rows so every cell is square.
    For 1D maps (n_cols==1): axis width is fixed to the residue-1D width (2.40/7)
    so acid-1D and residue-1D share the same axes thickness.
    Symmetric L/R margins so axes are centered horizontally.
    """
    axes_h = _SM_AXIS_HEIGHT_IN
    if n_cols == 1:
        axes_w = _SM_1D_AXIS_WIDTH_IN
    else:
        cell_side = axes_h / float(n_rows)
        axes_w = cell_side * float(n_cols)
    fig_w = _SM_SIDE_MARGIN_IN + axes_w + _SM_SIDE_MARGIN_IN
    fig_h = _SM_BOTTOM_MARGIN_IN + axes_h + _SM_TOP_MARGIN_IN
    rect = [
        _SM_SIDE_MARGIN_IN / fig_w,
        _SM_BOTTOM_MARGIN_IN / fig_h,
        axes_w / fig_w,
        axes_h / fig_h,
    ]
    return (fig_w, fig_h), rect


def _make_rdp_fig():
    """Return (fig, ax) with a 2.20×2.20 in square axes."""
    fig = plt.figure(figsize=RDP_PLOT_FIGSIZE)
    ax = fig.add_axes(_RDP_AX_RECT)
    return fig, ax


def _apply_contact_heatmap_ticks(ax):
    """Set contact-map tick style: outward bottom/left ticks only, length 4, width 2."""
    ax.tick_params(
        axis="both",
        which="major",
        top=False,
        bottom=True,
        left=True,
        right=False,
        labeltop=False,
        labelbottom=True,
        labelleft=True,
        labelright=False,
        direction="out",
        length=4,
        width=2,
        pad=1,
    )
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")


def _legend_entries(ax, entries, linewidth=2, fontsize=8, right_margin=0.06,
                    outside=False):
    """Draw legend-style colored line + black text labels, left-aligned as a group.

    *entries* is a list of ``(label, color, y_frac)`` tuples.

    When ``outside`` is False (default), the legend group sits inside the axes
    with the longest label's right edge ``right_margin`` from the right spine.

    When ``outside`` is True, the group is placed in the right-margin OUTSIDE
    the right spine: a short colored swatch from x=1.02 → 1.08 (axes fraction)
    followed by the label text at x=1.10. Caller must ``fig.savefig(...,
    bbox_inches='tight')`` so the canvas grows to include the outside legend.
    The axes box itself is unchanged.
    """
    if not entries:
        return
    if outside:
        for label, color, y_frac in entries:
            ax.annotate(
                "", xy=(1.02, y_frac), xytext=(1.08, y_frac),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-", color=color, linewidth=linewidth + 0.5),
                annotation_clip=False,
            )
            ax.text(1.10, y_frac, label, transform=ax.transAxes,
                    va='center', ha='left', fontsize=fontsize, color='black',
                    clip_on=False)
        return
    fig = ax.get_figure()
    bbox = ax.get_position()
    ax_width_pts = bbox.width * fig.get_figwidth() * 72.0
    handle_len = 2.0 * fontsize / ax_width_pts
    text_pad = 0.8 * fontsize / ax_width_pts
    max_text = max(len(lbl) for lbl, _, _ in entries) * 0.6 * fontsize / ax_width_pts
    x_line = 1.0 - right_margin - max_text - text_pad - handle_len
    x_text = x_line + handle_len + text_pad
    for label, color, y_frac in entries:
        ax.plot([x_line, x_line + handle_len], [y_frac, y_frac], color=color,
                linewidth=linewidth, transform=ax.transAxes, clip_on=False,
                solid_capstyle='round')
        ax.text(x_text, y_frac, label, transform=ax.transAxes, color='black',
                fontsize=fontsize, va='center', ha='left')


def _smooth_line_to_xlim(x, y, xlim, n=800, smoothing=1.0, k_max=5):
    """Return a smooth line evaluated across the full requested x-range."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    if np.count_nonzero(finite) == 0:
        return np.array([]), np.array([])

    x_arr = x_arr[finite]
    y_arr = y_arr[finite]
    order = np.argsort(x_arr)
    x_arr = x_arr[order]
    y_arr = y_arr[order]
    unique_x, unique_idx = np.unique(x_arr, return_index=True)
    unique_y = y_arr[unique_idx]

    xs = np.linspace(xlim[0], xlim[1], n)
    if len(unique_x) < 2:
        return xs, np.full_like(xs, unique_y[0], dtype=float)
    if len(unique_x) <= 5:
        return xs, np.interp(xs, unique_x, unique_y, left=unique_y[0], right=unique_y[-1])

    k = min(k_max, len(unique_x) - 1)
    spl = UnivariateSpline(unique_x, unique_y, k=k)
    spl.set_smoothing_factor(smoothing)
    return xs, spl(xs)


def _rdp_point_indices(length, limit=None):
    """Return strided RDP marker indices (step RDP_POINT_STRIDE) up to *length* (capped by *limit*)."""
    n = int(length) if limit is None else min(int(length), int(limit))
    if n <= 0:
        return np.array([], dtype=int)
    return np.arange(0, n, RDP_POINT_STRIDE, dtype=int)


def _rdp_plot_points(*arrays, limit=None):
    """Subsample several parallel arrays at the shared strided RDP marker indices."""
    if not arrays:
        return ()
    idx = _rdp_point_indices(len(arrays[0]), limit=limit)
    return tuple(np.asarray(arr)[idx] for arr in arrays)


class biopolymer_analysis():
    """Generate all per-biopolymer RDP overlays and residue/acid/SM contact maps for one temperature.

    Constructed with the data path, the analysis root (where averaged ANALYSIS_*_AVE
    inputs and FIGURES outputs live), and the temperature T. Exposes the gen_* /
    plot_* methods invoked from the __main__ driver to build the radial density
    profiles (with sigmoid fits and DSM/NDSM difference overlays), the
    contact-probability maps in their several normalizations (K_chain, P_contact,
    C_total, membership, enrichment), the small-molecule difference maps, and the
    SG/DSM/NDSM three-way comparison figures.

    Key attributes:
        path: TEMP_XXX data directory (also the analysis CWD).
        analysis_root: Root holding ANALYSIS_*_AVE inputs and the FIGURES/ tree.
        T: Simulation temperature in Kelvin.

    Usage:
        biopolymer_analysis(path, analysis_root, T).gen_residue_cpms()  # etc.
    """

    def __init__(self, path, analysis_root, T=300):
        """Store the data path, analysis root, and temperature; create output dirs.

        Args:
            path: TEMP_XXX data directory (used as the analysis CWD).
            analysis_root: Directory holding the averaged ANALYSIS_*_AVE inputs
                and the FIGURES/ output tree.
            T: Simulation temperature in Kelvin.
        """
        self.path = path
        self.analysis_root = analysis_root
        self.T = T
        self._ensure_dirs()

    def _ensure_dirs(self):
        """Create the FIGURES/ and IMAGES/ output subdirectories under analysis_root."""
        base = self.analysis_root
        dirs = [
            os.path.join(base, "FIGURES"),
            os.path.join(base, "FIGURES", "RDP"),
            os.path.join(base, "FIGURES", "RESIDUE_CONTACT_MAPS"),
            os.path.join(base, "FIGURES", "ACID_CONTACT_MAPS"),
            os.path.join(base, "FIGURES", "SM_CONTACT_MAPS"),
            os.path.join(base, "FIGURES", "PROPERTIES"),
            os.path.join(base, "IMAGES"),
            os.path.join(base, "IMAGES", "RDP"),
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    def _analysis_dir(self, folder):
        """Return the absolute path of *folder* under analysis_root."""
        return os.path.join(self.analysis_root, folder)

    def _analysis_file(self, folder, filename):
        """Return the absolute path of *filename* inside *folder* under analysis_root."""
        return os.path.join(self.analysis_root, folder, filename)

    def _available_conditions(self):
        """Return list of categories ("SG", "DSM", "NDSM") whose ANALYSIS_*_AVE
        folder is present and contains the universal precondition file
        BioPolNum_*.csv. SG-only temperatures (e.g. TEMP_290/295/305/310)
        return ["SG"]; standard DSM+NDSM temperatures return all three.
        """
        avail = []
        cands = [
            ("SG",   "ANALYSIS_SG_AVE",   "BioPolNum_sg_X.csv"),
            ("DSM",  "ANALYSIS_DSM_AVE",  "BioPolNum_DSM.csv"),
            ("NDSM", "ANALYSIS_NDSM_AVE", "BioPolNum_NDSM.csv"),
        ]
        for cat, folder, bionum in cands:
            if os.path.isfile(self._analysis_file(folder, bionum)):
                avail.append(cat)
        return avail

    def _sg_halfmax_radius_order(self, fallback_order):
        """Return SG species ordered by the outer radius where RDP falls to 50% of its peak."""
        preferred = ["FUS", "TIA1", "TDP43", "G3BP1", "PABP1", "TTP", "RNA"]
        ave_dir = self._analysis_dir("ANALYSIS_SG_AVE")
        ranked = []

        for sp in fallback_order:
            path = os.path.join(ave_dir, f"Density_Profile_{sp}_SG.csv")
            if not os.path.isfile(path):
                continue
            try:
                df = pd.read_csv(path)
                distance = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
                density = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
                mask = np.isfinite(distance) & np.isfinite(density)
                distance = distance[mask]
                density = density[mask]
                if distance.size < 2:
                    continue
                peak_idx = int(np.nanargmax(density))
                half_density = 0.5 * float(density[peak_idx])
                half_radius = np.nan
                for idx in range(peak_idx + 1, len(density)):
                    y0, y1 = density[idx - 1], density[idx]
                    if (y0 >= half_density >= y1) or (y1 >= half_density >= y0):
                        x0, x1 = distance[idx - 1], distance[idx]
                        half_radius = (
                            x0 + (half_density - y0) * (x1 - x0) / (y1 - y0)
                            if y1 != y0 else x1
                        )
                        break
                if np.isfinite(half_radius):
                    ranked.append((sp, float(half_radius)))
            except Exception as exc:
                print(f"[WARN] Could not read SG half-max radius for {sp}: {exc}")

        if len(ranked) == len(fallback_order):
            ordered = [sp for sp, _ in sorted(ranked, key=lambda item: item[1])]
            if "RNA" in ordered:
                ordered = [sp for sp in ordered if sp != "RNA"] + ["RNA"]
            return ordered

        return [sp for sp in preferred if sp in fallback_order]

    def _residue_source_order(self):
        """Return the 7 biopolymer species in their on-disk (source) row order."""
        return ["G3BP1", "PABP1", "TTP", "TIA1", "TDP43", "FUS", "RNA"]

    def _residue_plot_order(self):
        """Return the 7 species in SG half-max-radius display order."""
        return self._sg_halfmax_radius_order(self._residue_source_order())

    def _reorder_residue_rows(self, arr):
        """Reorder the rows of *arr* from source order to display order.

        Returns (reordered_array, display_order). Leaves *arr* untouched if its
        leading axis is not the 7-species axis.
        """
        source = self._residue_source_order()
        order = self._residue_plot_order()
        idx = [source.index(sp) for sp in order]
        arr = np.asarray(arr)
        if arr.shape[0] == len(source):
            arr = arr[idx, ...]
        return arr, order

    def _reorder_residue_matrix(self, arr):
        """Reorder both axes of a 7x7 *arr* from source order to display order.

        Returns (reordered_matrix, display_order); axes that are not the
        7-species axis are left as-is.
        """
        source = self._residue_source_order()
        order = self._residue_plot_order()
        idx = [source.index(sp) for sp in order]
        arr = np.asarray(arr)
        if arr.shape[0] == len(source):
            arr = arr[idx, ...]
        if arr.ndim >= 2 and arr.shape[1] == len(source):
            arr = arr[:, idx]
        return arr, order

    def gen_biopolymer_fitters(self, folder, sm):
        """Build the seven per-species RDP fitters (raw, un-normalized) for category *sm*.

        Args:
            folder: Unused; kept for call-site compatibility (the read folder is
                derived from *sm*).
            sm: Category / small-molecule identifier (e.g. "SG", "DSM",
                "dsm_anisomycin").

        Returns the fitters in the order
        (G3BP1, TDP43, PABP1, FUS, TIA1, TTP, RNA).
        """
        pca_file_A = self._analysis_file("ANALYSIS_SG_AVE", "PCA_Protein_sg_X.csv")
        cluster_file_A = self._analysis_file("ANALYSIS_SG_AVE", "Cluster_Protein_sg_X.csv")
        init = [80, 80, 200, 80]
        T = self.T

        # Determine which folder to read from based on sm
        # All files are now in ANALYSIS_*_AVE folders
        if sm == "SG" or sm == "sg_X":
            read_folder = "ANALYSIS_SG_AVE"
            sm_name = "SG"
        elif sm.upper() in ["DSM", "NDSM"]:
            # Aggregated DSM/NDSM are also in AVE folders now
            read_folder = "ANALYSIS_{}_AVE".format(sm.upper())
            sm_name = sm.upper()
        else:
            # Individual molecule (e.g., dsm_anisomycin, ndsm_dmso)
            category = sm.split("_")[0].upper()
            read_folder = "ANALYSIS_{}_AVE".format(category)
            sm_name = sm

        label = "G3BP1"
        print("{}    {}".format(sm,label))
        density_file = self._analysis_file(read_folder, f"Density_Profile_G3BP1_{sm_name}.csv")
        fitterG3BP1 = rdp(density_file, pca_file_A, cluster_file_A, T, init, label)

        label = "TDP43"
        print("{}    {}".format(sm, label))
        density_file = self._analysis_file(read_folder, f"Density_Profile_TDP43_{sm_name}.csv")
        fitterTDP43 = rdp(density_file, pca_file_A, cluster_file_A, T, init, label)

        label = "FUS"
        print("{}    {}".format(sm, label))
        density_file = self._analysis_file(read_folder, f"Density_Profile_FUS_{sm_name}.csv")
        fitterFUS = rdp(density_file, pca_file_A, cluster_file_A, T, init, label)

        label = "PABP1"
        print("{}    {}".format(sm, label))
        density_file = self._analysis_file(read_folder, f"Density_Profile_PABP1_{sm_name}.csv")
        fitterPABP1 = rdp(density_file, pca_file_A, cluster_file_A, T, init, label)

        label = "TIA1"
        print("{}    {}".format(sm, label))
        density_file = self._analysis_file(read_folder, f"Density_Profile_TIA1_{sm_name}.csv")
        fitterTIA1 = rdp(density_file, pca_file_A, cluster_file_A, T, init, label)

        label = "TTP"
        print("{}    {}".format(sm, label))
        density_file = self._analysis_file(read_folder, f"Density_Profile_TTP_{sm_name}.csv")
        fitterTTP = rdp(density_file, pca_file_A, cluster_file_A, T, init, label)

        label = "RNA"
        print("{}    {}".format(sm, label))
        density_file = self._analysis_file(read_folder, f"Density_Profile_RNA_{sm_name}.csv")
        fitterRNA = rdp(density_file, pca_file_A, cluster_file_A, T, init, label)

        return fitterG3BP1, fitterTDP43, fitterPABP1, fitterFUS, fitterTIA1, fitterTTP, fitterRNA

    def gen_biopolymer_fitters_normalized(self, folder, sm):
        """Like ``gen_biopolymer_fitters`` but with ``normalize=True``.

        Each species' density is scaled by its own maximum prior to fitting, so
        the resulting fitters can be overlaid on a common 0-1 axis.

        Args:
            folder: Unused; kept for call-site compatibility.
            sm: Category / small-molecule identifier (e.g. "SG", "DSM",
                "dsm_anisomycin").

        Returns the fitters in the order
        (G3BP1, TDP43, PABP1, FUS, TIA1, TTP, RNA).
        """
        pca_file_A = self._analysis_file("ANALYSIS_SG_AVE", "PCA_Protein_sg_X.csv")
        cluster_file_A = self._analysis_file("ANALYSIS_SG_AVE", "Cluster_Protein_sg_X.csv")
        init = [80, 80, 200, 80]
        T = self.T

        # Determine which folder to read from based on sm
        # All files are now in ANALYSIS_*_AVE folders
        if sm == "SG" or sm == "sg_X":
            read_folder = "ANALYSIS_SG_AVE"
            sm_name = "SG"
        elif sm.upper() in ["DSM", "NDSM"]:
            # Aggregated DSM/NDSM are also in AVE folders now
            read_folder = "ANALYSIS_{}_AVE".format(sm.upper())
            sm_name = sm.upper()
        else:
            # Individual molecule (e.g., dsm_anisomycin, ndsm_dmso)
            category = sm.split("_")[0].upper()
            read_folder = "ANALYSIS_{}_AVE".format(category)
            sm_name = sm

        label = "G3BP1"
        print("{}    {} (normalized)".format(sm,label))
        density_file = self._analysis_file(read_folder, f"Density_Profile_G3BP1_{sm_name}.csv")
        fitterG3BP1 = rdp(density_file, pca_file_A, cluster_file_A, T, init, label, normalize=True)

        label = "TDP43"
        print("{}    {} (normalized)".format(sm, label))
        density_file = self._analysis_file(read_folder, f"Density_Profile_TDP43_{sm_name}.csv")
        fitterTDP43 = rdp(density_file, pca_file_A, cluster_file_A, T, init, label, normalize=True)

        label = "FUS"
        print("{}    {} (normalized)".format(sm, label))
        density_file = self._analysis_file(read_folder, f"Density_Profile_FUS_{sm_name}.csv")
        fitterFUS = rdp(density_file, pca_file_A, cluster_file_A, T, init, label, normalize=True)

        label = "PABP1"
        print("{}    {} (normalized)".format(sm, label))
        density_file = self._analysis_file(read_folder, f"Density_Profile_PABP1_{sm_name}.csv")
        fitterPABP1 = rdp(density_file, pca_file_A, cluster_file_A, T, init, label, normalize=True)

        label = "TIA1"
        print("{}    {} (normalized)".format(sm, label))
        density_file = self._analysis_file(read_folder, f"Density_Profile_TIA1_{sm_name}.csv")
        fitterTIA1 = rdp(density_file, pca_file_A, cluster_file_A, T, init, label, normalize=True)

        label = "TTP"
        print("{}    {} (normalized)".format(sm, label))
        density_file = self._analysis_file(read_folder, f"Density_Profile_TTP_{sm_name}.csv")
        fitterTTP = rdp(density_file, pca_file_A, cluster_file_A, T, init, label, normalize=True)

        label = "RNA"
        print("{}    {} (normalized)".format(sm, label))
        density_file = self._analysis_file(read_folder, f"Density_Profile_RNA_{sm_name}.csv")
        fitterRNA = rdp(density_file, pca_file_A, cluster_file_A, T, init, label, normalize=True)

        return fitterG3BP1, fitterTDP43, fitterPABP1, fitterFUS, fitterTIA1, fitterTTP, fitterRNA

    def plot_biopolymer_curve_offset(self, fitter, ax1, fig, col, lab, offset, line_style, line_width,
                                     xlim=RDP_XLIM, show_errorbars=True, band_height=1.0):
        """Plot one biopolymer density curve on *ax1*, shifted up by *offset*.

        Draws the smoothed (spline) profile, a dashed vertical guide at the 50%
        midpoint concentration, the baseline at *offset*, and scatter+errorbars
        for the underlying data points (errorbars clipped to the [offset,
        offset+band_height] band). With ``lab == ""`` only the spline is drawn
        (used for unlabelled SG control overlays). Returns (fig, ax1).
        """
        sns.set_theme(style="ticks")
        sns.set_style('white')
        plt.rc('axes', titlesize=10)
        plt.rc('axes', labelsize=10)
        plt.rc('xtick', labelsize=10)
        plt.rc('ytick', labelsize=10)
        plt.rc('legend', fontsize=8)
        plt.rc('font', size=10)
        plt.rc('axes', linewidth=2)

        # Get data in xlim range
        x_min, x_max = xlim
        dist_arr = np.asarray(fitter.distances)
        dens_arr = np.asarray(fitter.densities)
        mask = (dist_arr >= x_min) & (dist_arr <= x_max)
        dist_clip = list(dist_arr[mask])
        dens_clip_raw = list(dens_arr[mask])

        if len(dist_clip) == 0:
            return fig, ax1

        if lab != "":
            xs, spl_vals = _smooth_line_to_xlim(dist_clip, dens_clip_raw, xlim)
            ax1.plot(xs, spl_vals + offset, color=col, label=lab, zorder=1, linewidth=line_width, linestyle=line_style, clip_on=True)

            # Dashed vertical line at midpoint concentration (enforce x > 50)
            mask_after_50 = xs > 50
            xs_after_50 = xs[mask_after_50]
            spl_vals_after_50 = spl_vals[mask_after_50]
            if len(xs_after_50) > 0:
                index_rel = (np.abs(spl_vals_after_50 - 0.5)).argmin()
                x_pos = xs_after_50[index_rel]
                spl_val_at_pos = spl_vals_after_50[index_rel]
                plt.plot([x_pos, x_pos], [offset, offset + spl_val_at_pos], color=col,
                         linewidth=RDP_LINEWIDTH, zorder=1, linestyle=(0, (2, 0.5)), clip_on=True)

            # Plot horizontal black line over full xlim range
            if offset != 0:
                ax1.plot([x_min, x_max], [offset, offset], color='k', linewidth=RDP_LINEWIDTH, clip_on=True)

            # Scatter ALL actual data points within xlim
            dens_clip = list(np.asarray(dens_clip_raw) + offset)
            err_clip = list(np.asarray(fitter.errors)[mask]) if hasattr(fitter, 'errors') else None
            if err_clip is not None:
                plot_x, plot_y, plot_err = _rdp_plot_points(dist_clip, dens_clip, err_clip)
            else:
                plot_x, plot_y = _rdp_plot_points(dist_clip, dens_clip)
                plot_err = None
            if show_errorbars and plot_err is not None:
                # Clip errorbars to band boundaries [offset, offset + band_height]
                plot_y_arr = np.asarray(plot_y, dtype=float)
                plot_err_arr = np.asarray(plot_err, dtype=float)
                band_lo = offset
                band_hi = offset + band_height
                yerr_lower = np.clip(np.minimum(plot_err_arr, plot_y_arr - band_lo), 0.0, None)
                yerr_upper = np.clip(np.minimum(plot_err_arr, band_hi - plot_y_arr), 0.0, None)
                ax1.errorbar(x=plot_x, y=plot_y_arr,
                             yerr=np.vstack([yerr_lower, yerr_upper]),
                             fmt="none", color=col,
                             zorder=9, capsize=2, elinewidth=RDP_ERRORBAR_WIDTH)
            sns.scatterplot(ax=ax1, x=plot_x, y=plot_y,
                          color=col, legend=False, s=RDP_MARKER_SIZE, edgecolor="k",
                          linewidth=RDP_MARKER_EDGEWIDTH, zorder=10, clip_on=False)
        else:
            # For control lines (no label), just plot spline
            xs, spl_vals = _smooth_line_to_xlim(dist_clip, dens_clip_raw, xlim)
            ax1.plot(xs, spl_vals + offset, color=col, label=lab, zorder=1, linewidth=line_width, linestyle=line_style, clip_on=True)

        ax1.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in',
                        length=4, width=2)

        ax1.set_xlim(xlim)
        ax1.set_xticks([0, 100, 200, 300, 400, 500])
        ax1.set_ylim(0, 7)
        for spine in ax1.spines.values():
            spine.set_zorder(3)

        return fig, ax1


    def plot_biopolymer_curve_raw(self, fitter, ax1, fig, col, lab, offset, line_style, line_width, xlim=RDP_XLIM):
        """Plot one raw (un-normalized, un-splined) density curve on *ax1*.

        Unlike ``plot_biopolymer_curve_offset`` this draws the densities as-is
        (extended to x=0) with scatter+errorbars when *lab* is non-empty, with
        no smoothing or midpoint guide. Returns (fig, ax1).
        """
        sns.set_theme(style="ticks")
        sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
        plt.rc('axes', titlesize=10)  # fontsize of the axes title
        plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
        plt.rc('legend', fontsize=8)  # legend fontsize
        plt.rc('font', size=10)  # controls default text sizes
        plt.rc('axes', linewidth=2)

        # clip to x in xlim range
        x_min, x_max = xlim
        dist_arr = np.asarray(fitter.distances)
        dens_arr = np.asarray(fitter.densities)
        mask = (dist_arr >= x_min) & (dist_arr <= x_max)
        distances = list(dist_arr[mask])
        densities = np.array(dens_arr[mask])

        if len(distances) == 0:
            return fig, ax1

        # Extend line to start at x_min so it reaches x=0
        plot_d, plot_rho = distances, densities
        if len(distances) > 0 and distances[0] > x_min:
            plot_d = [x_min] + distances
            plot_rho = np.concatenate([[densities[0]], densities])

        ax1.plot(plot_d, plot_rho, color=col, label=lab, zorder=1, linewidth=line_width, linestyle=line_style, clip_on=True)

        if lab != "":
            err = list(np.asarray(fitter.errors)[mask])
            plot_x, plot_y, plot_err = _rdp_plot_points(distances, densities, err)
            ax1.errorbar(x=plot_x, y=plot_y, yerr=plot_err,
                         fmt="none", color=col, zorder=9, capsize=2,
                         elinewidth=RDP_ERRORBAR_WIDTH)
            sns.scatterplot(ax=ax1, x=plot_x, y=plot_y,
                          color=col, legend=False, s=RDP_MARKER_SIZE, edgecolor="k",
                          linewidth=RDP_MARKER_EDGEWIDTH, zorder=10, clip_on=False)

        ax1.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in',
                        length=4, width=2)

        ax1.set_xlim(xlim)
        ax1.set_xticks([0, 100, 200, 300, 400, 500])
        ax1.set_ylim(0, 225)
        for spine in ax1.spines.values():
            spine.set_zorder(3)

        return fig, ax1

    def gen_biopolymer_plots(self, fitterG3BP1, fitterTDP43, fitterPABP1, fitterFUS, fitterTIA1, fitterTTP, fitterRNA, sm, xlim=RDP_XLIM):
        """Render the per-species NCC offset-stacked and raw RDP overlays for one category.

        Builds two figures for category *sm* ("SG", "DSM", or "NDSM"): a
        normalized offset stack (each species in its own [offset, offset+1]
        band, with the SM track at the bottom and dashed SG controls for
        DSM/NDSM) and a raw (un-normalized) overlay. Saves both under
        FIGURES/RDP/ as PROTEIN_ANALYSIS_NCC_{OFFSET,RAW}_{sm}.png.
        """
        col_pall = sns.color_palette("rocket", n_colors=14)
        col_pall2 = sns.color_palette(["#0066ff"], 1)
        col_pal_sm = sns.color_palette(["#40641b", "#bfe49b"], 2)

        sns.set_theme(style="ticks")
        sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
        plt.rc('axes', titlesize=10)  # fontsize of the axes title
        plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
        plt.rc('legend', fontsize=8)  # legend fontsize
        plt.rc('font', size=10)  # controls default text sizes
        plt.rc('axes', linewidth=2)

        fig1, ax1 = _make_rdp_fig()

        # Determine folder based on sm
        category = sm if sm in ["DSM", "NDSM"] else "SG"
        folder = f"ANALYSIS_{category}_AVE"

        # Build normalized fitters for the stacked panel
        nG3BP1, nTDP43, nPABP1, nFUS, nTIA1, nTTP, nRNA = self.gen_biopolymer_fitters_normalized(folder, sm)

        # If DSM/NDSM, compute normalized SM fitter and draw at bottom (0–1)
        if sm != "SG":
            # Use legacy style for SM curve at bottom (0–1) with spline/dashed midpoint
            label_sm = sm
            pca_file_A = self._analysis_file("ANALYSIS_SG_AVE", "PCA_Protein_sg_X.csv")
            cluster_file_A = self._analysis_file("ANALYSIS_SG_AVE", "Cluster_Protein_sg_X.csv")
            init = [80, 80, 200, 80]
            T = self.T
            density_file = self._analysis_file(f"ANALYSIS_{sm}_AVE", f"Density_Profile_SM_{sm}.csv")
            fitterSM = rdp(density_file, pca_file_A, cluster_file_A, T, init, label_sm, normalize=True)
            sm_col = col_pal_sm[0] if sm == "DSM" else col_pal_sm[1]
            fig1, ax1 = self.plot_biopolymer_curve_offset(fitterSM, ax1, fig1, sm_col, label_sm, 0, line_width=2, line_style='solid', xlim=xlim)

        base = 1 if sm != "SG" else 0

        species_fitters = {
            "TDP43": nTDP43,
            "FUS": nFUS,
            "TIA1": nTIA1,
            "G3BP1": nG3BP1,
            "PABP1": nPABP1,
            "TTP": nTTP,
            "RNA": nRNA,
        }
        species_colors = {
            "TDP43": col_pall[0],
            "FUS": col_pall[2],
            "TIA1": col_pall[4],
            "G3BP1": col_pall[6],
            "PABP1": col_pall[8],
            "TTP": col_pall[10],
            "RNA": col_pall2[0],
        }
        default_order = ["TDP43", "FUS", "TIA1", "G3BP1", "PABP1", "TTP", "RNA"]
        plot_order = self._sg_halfmax_radius_order(default_order)
        offset_by_species = {
            species: base + (len(plot_order) - 1 - idx)
            for idx, species in enumerate(plot_order)
        }

        for species in plot_order:
            fig1, ax1 = self.plot_biopolymer_curve_offset(
                species_fitters[species], ax1, fig1, species_colors[species], species,
                offset_by_species[species], line_width=2, line_style='solid', xlim=xlim,
            )

        # Legend labels placed after ylim is set (see below)

        if sm != "SG":

            # Overlay SG controls, normalized as well
            fitterG3BP1, fitterTDP43, fitterPABP1, fitterFUS, fitterTIA1, fitterTTP, fitterRNA = self.gen_biopolymer_fitters_normalized("ANALYSIS_SG_AVE", "SG")
            control_fitters = {
                "TDP43": fitterTDP43,
                "FUS": fitterFUS,
                "TIA1": fitterTIA1,
                "G3BP1": fitterG3BP1,
                "PABP1": fitterPABP1,
                "TTP": fitterTTP,
                "RNA": fitterRNA,
            }
            for species in plot_order:
                fig1, ax1 = self.plot_biopolymer_curve_offset(
                    control_fitters[species], ax1, fig1, species_colors[species], "",
                    offset_by_species[species], line_width=1.5, line_style=(0, (1, 1)),
                    xlim=xlim,
                )

        # Axis limits
        ax1.set_xlim(xlim)
        ax1.set_xticks([0, 100, 200, 300, 400, 500])
        if sm in ("DSM","NDSM"):
            ax1.set_ylim(0, 8)

        # Place labels at 50% of each offset band (band midpoint), outside axes.
        label_band_fraction = 0.50
        ylim_max = 8 if sm in ("DSM", "NDSM") else 7
        leg_entries = [
            (species, species_colors[species], (offset_by_species[species] + label_band_fraction) / ylim_max)
            for species in plot_order
        ]
        if sm in ("DSM", "NDSM"):
            sm_col = col_pal_sm[0] if sm == "DSM" else col_pal_sm[1]
            leg_entries.append((sm, sm_col, label_band_fraction / ylim_max))
        _legend_entries(ax1, leg_entries, right_margin=0.02, outside=True)

        # Save offset-stacked version
        fig1.savefig("{}/FIGURES/RDP/PROTEIN_ANALYSIS_NCC_OFFSET_{}.png".format(self.analysis_root,sm), format="png", dpi=400, bbox_inches='tight')

        # ---------------------------------
        # Also render non-offset (raw) stack
        # ---------------------------------
        fig2, ax2 = _make_rdp_fig()

        # Per-species vertical offsets for the raw stack (RNA at the bottom).
        if sm != "SG":
            offset_arr = np.arange(1, 8, 1)
        else:
            offset_arr = np.arange(0, 7, 1)

        # Regenerate raw fitters explicitly to ensure they're not normalized
        fitterG3BP1_raw, fitterTDP43_raw, fitterPABP1_raw, fitterFUS_raw, fitterTIA1_raw, fitterTTP_raw, fitterRNA_raw = self.gen_biopolymer_fitters(folder, sm)

        label = "TDP43"
        offset = offset_arr[6]
        col = col_pall[0]
        fig2, ax2 = self.plot_biopolymer_curve_raw(fitterTDP43_raw, ax2, fig2, col, label, offset, line_width=2, line_style='solid', xlim=xlim)

        label = "FUS"
        offset = offset_arr[5]
        col = col_pall[2]
        fig2, ax2 = self.plot_biopolymer_curve_raw(fitterFUS_raw, ax2, fig2, col, label, offset, line_width=2, line_style='solid', xlim=xlim)

        label = "TIA1"
        offset = offset_arr[4]
        col = col_pall[4]
        fig2, ax2 = self.plot_biopolymer_curve_raw(fitterTIA1_raw, ax2, fig2, col, label, offset, line_width=2, line_style='solid', xlim=xlim)

        label = "G3BP1"
        offset = offset_arr[3]
        col = col_pall[6]
        fig2, ax2 = self.plot_biopolymer_curve_raw(fitterG3BP1_raw, ax2, fig2, col, label, offset, line_width=2, line_style='solid', xlim=xlim)

        label = "PABP1"
        offset = offset_arr[2]
        col = col_pall[8]
        fig2, ax2 = self.plot_biopolymer_curve_raw(fitterPABP1_raw, ax2, fig2, col, label, offset, line_width=2, line_style='solid', xlim=xlim)

        label = "TTP"
        offset = offset_arr[1]
        col = col_pall[10]
        fig2, ax2 = self.plot_biopolymer_curve_raw(fitterTTP_raw, ax2, fig2, col, label, offset, line_width=2, line_style='solid', xlim=xlim)

        label = "RNA"
        offset = offset_arr[0]
        col = col_pall2[0]
        fig2, ax2 = self.plot_biopolymer_curve_raw(fitterRNA_raw, ax2, fig2, col, label, offset, line_width=2, line_style='solid', xlim=xlim)

        if sm != "SG":
            # Overlay SG controls, no labels
            fitterG3BP1, fitterTDP43, fitterPABP1, fitterFUS, fitterTIA1, fitterTTP, fitterRNA = self.gen_biopolymer_fitters("ANALYSIS_SG_AVE", "SG")

            label = ""
            offset = offset_arr[6]
            col = col_pall[0]
            fig2, ax2 = self.plot_biopolymer_curve_raw(fitterTDP43, ax2, fig2, col, label, offset, line_width=1.5, line_style=(0, (1, 1)), xlim=xlim)

            label = ""
            offset = offset_arr[5]
            col = col_pall[2]
            fig2, ax2 = self.plot_biopolymer_curve_raw(fitterFUS, ax2, fig2, col, label, offset, line_width=1.5, line_style=(0, (1, 1)), xlim=xlim)

            label = ""
            offset = offset_arr[4]
            col = col_pall[4]
            fig2, ax2 = self.plot_biopolymer_curve_raw(fitterTIA1, ax2, fig2, col, label, offset, line_width=1.5, line_style=(0, (1, 1)), xlim=xlim)

            label = ""
            offset = offset_arr[3]
            col = col_pall[6]
            fig2, ax2 = self.plot_biopolymer_curve_raw(fitterG3BP1, ax2, fig2, col, label, offset, line_width=1.5, line_style=(0, (1, 1)), xlim=xlim)

            label = ""
            offset = offset_arr[2]
            col = col_pall[8]
            fig2, ax2 = self.plot_biopolymer_curve_raw(fitterPABP1, ax2, fig2, col, label, offset, line_width=1.5, line_style=(0, (1, 1)), xlim=xlim)

            label = ""
            offset = offset_arr[1]
            col = col_pall[10]
            fig2, ax2 = self.plot_biopolymer_curve_raw(fitterTTP, ax2, fig2, col, label, offset, line_width=1.5, line_style=(0, (1, 1)), xlim=xlim)

            label = ""
            offset = offset_arr[0]
            col = col_pall2[0]
            fig2, ax2 = self.plot_biopolymer_curve_raw(fitterRNA, ax2, fig2, col, label, offset, line_width=1.5, line_style=(0, (1, 1)), xlim=xlim)

        # Axis limits (already set by plot function but redundantly set here)
        ax2.set_xlim(xlim)
        ax2.set_xticks([0, 100, 200, 300, 400, 500])
        ax2.set_ylim(0, 250)
        ax2.set_yticks([0, 50, 100, 150, 200, 250])

        # Legend for raw overlay plot
        handles, labels = ax2.get_legend_handles_labels()
        desired_order = plot_order
        lth = {l: h for h, l in zip(handles, labels) if l}
        oh = [lth[l] for l in desired_order if l in lth]
        ol = [l for l in desired_order if l in lth]
        leg = ax2.legend(oh, ol, loc='upper right', ncol=1,
                         borderpad=0.3, frameon=False)
        leg.get_frame().set_alpha(0)

        # Save raw version
        fig2.savefig("{}/FIGURES/RDP/PROTEIN_ANALYSIS_NCC_RAW_{}.png".format(self.analysis_root,sm), format="png", dpi=400)


    def gen_biopolymer_plots_with_rna_components(self, folder, sm, xlim=RDP_XLIM):
        """Protein/RNA NCC offset plot with A and UCG shown below the RNA track."""
        col_pall = sns.color_palette("rocket", n_colors=14)
        col_pal_rna = sns.color_palette(["#0066ff", "#99c2ff", "#38C7C5"], 3)
        col_pal_sm = sns.color_palette(["#40641b", "#bfe49b"], 2)

        sns.set_theme(style="ticks")
        sns.set_style('white')
        plt.rc('axes', titlesize=10)
        plt.rc('axes', labelsize=10)
        plt.rc('xtick', labelsize=10)
        plt.rc('ytick', labelsize=10)
        plt.rc('legend', fontsize=8)
        plt.rc('font', size=10)
        plt.rc('axes', linewidth=2)

        pca_protein = self._analysis_file("ANALYSIS_SG_AVE", "PCA_Protein_sg_X.csv")
        cluster_protein = self._analysis_file("ANALYSIS_SG_AVE", "Cluster_Protein_sg_X.csv")
        pca_rna = self._analysis_file("ANALYSIS_SG_AVE", "PCA_RNA_sg_X.csv")
        cluster_rna = self._analysis_file("ANALYSIS_SG_AVE", "Cluster_RNA_sg_X.csv")
        init = [80, 80, 200, 80]

        def rna_component_fitter(read_folder, name, label, sm_name):
            path = self._analysis_file(read_folder, f"Density_Profile_{name}_{sm_name}.csv")
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
            return rdp(path, pca_rna, cluster_rna, self.T, init, label, normalize=True)

        try:
            nG3BP1, nTDP43, nPABP1, nFUS, nTIA1, nTTP, nRNA = self.gen_biopolymer_fitters_normalized(folder, sm)
            fA = rna_component_fitter(folder, "ADENINE", "A", sm)
            fUCG = rna_component_fitter(folder, "UCG", "UCG", sm)
        except Exception as exc:
            print(f"[WARN] Skipping protein NCC RNA-component plot for {sm}: {exc}")
            return None

        species_fitters = {
            "FUS": nFUS,
            "TIA1": nTIA1,
            "TDP43": nTDP43,
            "G3BP1": nG3BP1,
            "PABP1": nPABP1,
            "TTP": nTTP,
            "RNA": nRNA,
            "A": fA,
            "UCG": fUCG,
        }
        species_colors = {
            "FUS": col_pall[2],
            "TIA1": col_pall[4],
            "TDP43": col_pall[0],
            "G3BP1": col_pall[6],
            "PABP1": col_pall[8],
            "TTP": col_pall[10],
            "RNA": col_pal_rna[0],
            "A": col_pal_rna[1],
            "UCG": col_pal_rna[2],
        }

        plot_order = self._sg_halfmax_radius_order(["TDP43", "FUS", "TIA1", "G3BP1", "PABP1", "TTP", "RNA"])
        top_to_bottom = [sp for sp in plot_order if sp != "RNA"] + ["RNA", "A", "UCG"]

        if sm != "SG":
            sm_path = self._analysis_file(folder, f"Density_Profile_SM_{sm}.csv")
            if os.path.isfile(sm_path):
                species_fitters[sm] = rdp(sm_path, pca_protein, cluster_protein, self.T, init, sm, normalize=True)
                species_colors[sm] = col_pal_sm[0] if sm == "DSM" else col_pal_sm[1]
                top_to_bottom.append(sm)
            else:
                print(f"[WARN] Missing SM density profile for {sm}: {sm_path}")

        n_bands = len(top_to_bottom)
        offsets = {label: n_bands - 1 - idx for idx, label in enumerate(top_to_bottom)}

        control_fitters = None
        if sm != "SG":
            try:
                cG3BP1, cTDP43, cPABP1, cFUS, cTIA1, cTTP, cRNA = self.gen_biopolymer_fitters_normalized("ANALYSIS_SG_AVE", "SG")
                control_fitters = {
                    "FUS": cFUS,
                    "TIA1": cTIA1,
                    "TDP43": cTDP43,
                    "G3BP1": cG3BP1,
                    "PABP1": cPABP1,
                    "TTP": cTTP,
                    "RNA": cRNA,
                    "A": rna_component_fitter("ANALYSIS_SG_AVE", "ADENINE", "", "SG"),
                    "UCG": rna_component_fitter("ANALYSIS_SG_AVE", "UCG", "", "SG"),
                }
            except Exception as exc:
                print(f"[WARN] Could not load SG controls for RNA-component plot for {sm}: {exc}")

        def _draw_rna_components(show_errorbars, filename):
            fig, ax = _make_rdp_fig()
            for label in top_to_bottom:
                self.plot_biopolymer_curve_offset(
                    species_fitters[label], ax, fig, species_colors[label], label, offsets[label],
                    line_width=2, line_style='solid', xlim=xlim,
                    show_errorbars=show_errorbars, band_height=1.0,
                )
            if control_fitters is not None:
                for label in top_to_bottom:
                    if label == sm:
                        continue
                    self.plot_biopolymer_curve_offset(
                        control_fitters[label], ax, fig, species_colors[label], "", offsets[label],
                        line_width=1.5, line_style=(0, (1, 1)), xlim=xlim,
                        show_errorbars=show_errorbars, band_height=1.0,
                    )

            ax.set_xlim(xlim)
            ax.set_xticks([0, 100, 200, 300, 400, 500])
            ax.set_ylim(0, n_bands)
            ax.set_yticks(np.arange(0, n_bands + 1, 1))
            for spine in ax.spines.values():
                spine.set_zorder(3)

            label_band_fraction = 0.50
            _legend_entries(ax, [
                (label, species_colors[label], (offsets[label] + label_band_fraction) / float(n_bands))
                for label in top_to_bottom
            ], right_margin=0.02, outside=True)

            fig.savefig(
                "{}/FIGURES/RDP/{}".format(self.analysis_root, filename),
                format="png", dpi=400, bbox_inches='tight',
            )
            return fig

        fig_err = _draw_rna_components(
            True,
            "PROTEIN_ANALYSIS_NCC_OFFSET_WITH_RNA_COMPONENTS_{}.png".format(sm),
        )
        _draw_rna_components(
            False,
            "PROTEIN_ANALYSIS_NCC_OFFSET_WITH_RNA_COMPONENTS_{}_NO_ERRORBARS.png".format(sm),
        )
        return fig_err

    def plot_rdp(self, sm, xlim=RDP_XLIM):
        """Render the total-system / protein / RNA RDP overlay for one category.

        For category *sm* fits and plots the SG (total), protein, and RNA
        radial density profiles on a shared mg/mL axis; for DSM/NDSM it adds the
        SM profile on a twin axis plus dashed SG-control curves. Saves
        FIGURES/RDP/{sm}_RDP.png (and, for SG, a protein/RNA crossover zoom).
        Returns the figure.
        """
        pca_file_A = self._analysis_file("ANALYSIS_SG_AVE", "PCA_Protein_sg_X.csv")
        cluster_file_A = self._analysis_file("ANALYSIS_SG_AVE", "Cluster_Protein_sg_X.csv")
        init = [80, 80, 200, 80]
        T = self.T

        label = "SG"
        print("{}    {}".format(sm,label))
        category = sm if sm in ["DSM", "NDSM"] else "SG"
        folder = f"ANALYSIS_{category}_AVE"
        density_file = self._analysis_file(folder, f"Density_Profile_SG_{sm}.csv")
        fitterSG = rdp(density_file, pca_file_A, cluster_file_A, T, init, label)

        label = "Protein"
        print("{}    {}".format(sm, label))
        density_file = self._analysis_file(folder, f"Density_Profile_Protein_{sm}.csv")
        fitterProtein = rdp(density_file, pca_file_A, cluster_file_A, T, init, label)

        label = "RNA"
        print("{}    {}".format(sm, label))
        density_file = self._analysis_file(folder, f"Density_Profile_RNA_{sm}.csv")
        fitterRNA = rdp(density_file, pca_file_A, cluster_file_A, T, init, label)

        if sm == "DSM" or sm == "NDSM":
            label = "RNA"
            print("{}    {}".format(sm, label))
            density_file = self._analysis_file(f"ANALYSIS_{sm}_AVE", f"Density_Profile_SM_{sm}.csv")
            fitterSM = rdp(density_file, pca_file_A, cluster_file_A, T, init, label)
            sg_control_density_file = self._analysis_file("ANALYSIS_SG_AVE", "Density_Profile_SG_SG.csv")
            protein_control_density_file = self._analysis_file("ANALYSIS_SG_AVE", "Density_Profile_Protein_SG.csv")
            rna_control_density_file = self._analysis_file("ANALYSIS_SG_AVE", "Density_Profile_RNA_SG.csv")

            fitter_SG_Control = rdp(sg_control_density_file, pca_file_A, cluster_file_A, T, init, label)
            fitter_Protein_Control = rdp(protein_control_density_file, pca_file_A, cluster_file_A, T, init, label)
            fitter_RNA_Control = rdp(rna_control_density_file, pca_file_A, cluster_file_A, T, init, label)

        else:
            fitterSM = None

        col_pal_sg = sns.color_palette(["#808080"], 1)[0]
        col_pal_protein = sns.color_palette(["#C7383A"], 1)[0]
        col_pal_rna = sns.color_palette(["#0066ff"], 1)[0]
        col_pal_sm = sns.color_palette(["#40641b", "#bfe49b"], 2)

        sns.set_theme(style="ticks")
        sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
        plt.rc('axes', titlesize=10)  # fontsize of the axes title
        plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
        plt.rc('legend', fontsize=8)  # legend fontsize
        plt.rc('font', size=10)  # controls default text sizes
        plt.rc('axes', linewidth=2)

        fig, ax1 = _make_rdp_fig()

        # Helper to filter fit curves to xlim
        def filter_fit(fit_x, fit_rho, xlim):
            fit_x_arr = np.asarray(fit_x)
            fit_rho_arr = np.asarray(fit_rho)
            mask = (fit_x_arr >= xlim[0]) & (fit_x_arr <= xlim[1])
            return fit_x_arr[mask], fit_rho_arr[mask]

        # Filter fit curves
        sg_fit_x, sg_fit_rho = filter_fit(fitterSG.fit_x, fitterSG.fit_rho, xlim)
        protein_fit_x, protein_fit_rho = filter_fit(fitterProtein.fit_x, fitterProtein.fit_rho, xlim)
        rna_fit_x, rna_fit_rho = filter_fit(fitterRNA.fit_x, fitterRNA.fit_rho, xlim)

        def _filter_scatter_err(fitter):
            d = np.asarray(fitter.distances)
            rho = np.asarray(fitter.densities)
            e = np.asarray(fitter.errors)
            m = (d >= xlim[0]) & (d <= xlim[1])
            sx, sy, se = _rdp_plot_points(d[m], rho[m], e[m])
            return sx, sy, sx, sy, se

        sx, sy, ex, ey, ee = _filter_scatter_err(fitterSG)
        line_sg, = ax1.plot(sg_fit_x, sg_fit_rho, color=col_pal_sg, label='SG', linewidth=2, zorder=2, clip_on=True)
        sns.scatterplot(ax=ax1, x=sx, y=sy, color=col_pal_sg, legend=False, s=40,
                        edgecolor="k", linewidth=1, zorder=10, clip_on=False)
        ax1.errorbar(x=ex, y=ey, yerr=ee, fmt="none", color=col_pal_sg, zorder=4, capsize=2)

        sx, sy, ex, ey, ee = _filter_scatter_err(fitterProtein)
        line_protein, = ax1.plot(protein_fit_x, protein_fit_rho, color=col_pal_protein, label='Protein', linewidth=2,
                 zorder=2, clip_on=True)
        sns.scatterplot(ax=ax1, x=sx, y=sy, color=col_pal_protein, legend=False,
                        s=40, edgecolor="k", linewidth=1, zorder=10, clip_on=False)
        ax1.errorbar(ex, ey, yerr=ee, fmt="none", color=col_pal_protein, zorder=4, capsize=2)

        sx, sy, ex, ey, ee = _filter_scatter_err(fitterRNA)
        line_rna, = ax1.plot(rna_fit_x, rna_fit_rho, color=col_pal_rna, label='RNA', linewidth=2, zorder=2, clip_on=True)
        sns.scatterplot(ax=ax1, x=sx, y=sy, color=col_pal_rna, legend=False, s=40,
                        edgecolor="k", linewidth=1, zorder=10, clip_on=False)
        ax1.errorbar(ex, ey, yerr=ee, fmt="none", color=col_pal_rna, zorder=4, capsize=2)

        if fitterSM is not None:
            ax2 = ax1.twinx()
            if "ND" not in sm:
                col = col_pal_sm[0]
            else:
                col = col_pal_sm[1]

            # Filter SM and control curves
            sm_fit_x, sm_fit_rho = filter_fit(fitterSM.fit_x, fitterSM.fit_rho, xlim)
            sg_ctrl_fit_x, sg_ctrl_fit_rho = filter_fit(fitter_SG_Control.fit_x, fitter_SG_Control.fit_rho, xlim)
            protein_ctrl_fit_x, protein_ctrl_fit_rho = filter_fit(fitter_Protein_Control.fit_x, fitter_Protein_Control.fit_rho, xlim)
            rna_ctrl_fit_x, rna_ctrl_fit_rho = filter_fit(fitter_RNA_Control.fit_x, fitter_RNA_Control.fit_rho, xlim)

            # Filter SM scatter points to xlim
            sm_mask = (np.asarray(fitterSM.distances) >= xlim[0]) & (np.asarray(fitterSM.distances) <= xlim[1])
            sm_distances_filtered, sm_densities_filtered = _rdp_plot_points(
                np.asarray(fitterSM.distances)[sm_mask],
                np.asarray(fitterSM.densities)[sm_mask],
            )
            sns.scatterplot(ax=ax2, x=sm_distances_filtered, y=sm_densities_filtered, color=col, legend=False, s=40,
                            edgecolor="k", linewidth=1, zorder=10, clip_on=False)
            # Filter errorbar to xlim
            sm_err_mask = (np.asarray(fitterSM.distances) >= xlim[0]) & (np.asarray(fitterSM.distances) <= xlim[1])
            sm_err_distances, sm_err_densities, sm_err_errors = _rdp_plot_points(
                np.asarray(fitterSM.distances)[sm_err_mask],
                np.asarray(fitterSM.densities)[sm_err_mask],
                np.asarray(fitterSM.errors)[sm_err_mask],
            )
            ax2.errorbar(sm_err_distances, sm_err_densities, yerr=sm_err_errors, fmt="none", color=col, zorder=4, capsize=2)
            line_sm, = ax2.plot(sm_fit_x, sm_fit_rho, color=col, label=sm, linewidth=2, zorder=2, clip_on=True)

            line_sg_ctrl, = ax1.plot(sg_ctrl_fit_x, sg_ctrl_fit_rho, color=col_pal_sg, label='SG Control',
                                     linewidth=1.5, zorder=1, linestyle=(0, (1, 1)), clip_on=True)

            line_protein_ctrl, = ax1.plot(protein_ctrl_fit_x, protein_ctrl_fit_rho, color=col_pal_protein, label='Protein Control',
                                          linewidth=1.5, zorder=1, linestyle=(0, (1, 1)), clip_on=True)

            line_rna_ctrl, = ax1.plot(rna_ctrl_fit_x, rna_ctrl_fit_rho, color=col_pal_rna,
                                      label='RNA Control',
                                      linewidth=1.5, zorder=1, linestyle=(0, (1, 1)), clip_on=True)


            _leg2 = ax2.get_legend()
            if _leg2 is not None:
                _leg2.remove()
            ax2.tick_params(left=False, right=True, top=False, bottom=False, labelbottom=False, direction='in',
                            length=4,
                            width=2)

            ax2.set_ylim(0.0, 1.0)

            ax1.tick_params(left=True, right=False, top=True, bottom=True, labelbottom=True, direction='in',
                            length=4,
                            width=2)
            ax1.set_ylim(0, 600)
            ax2.spines['bottom'].set_visible(False)
            ax2.spines['left'].set_visible(False)

        else:
            ax1.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in',
                            length=4,
                            width=2)
            ax1.set_ylim(0, 600)

        # Use xlim parameter
        ax1.set_xlim(xlim)
        ax1.set_xticks([0, 100, 200, 300, 400, 500])
        # Spine sits above fit lines (zorder<=2) but below scatter (zorder=10).
        # Tick marks are part of xaxis/yaxis (default zorder ~2.5); keep them below scatter.
        for spine in ax1.spines.values():
            spine.set_zorder(3)
        for line in ax1.get_xticklines() + ax1.get_yticklines():
            line.set_zorder(3)
        if fitterSM is not None:
            for spine in ax2.spines.values():
                spine.set_zorder(3)
            for line in ax2.get_xticklines() + ax2.get_yticklines():
                line.set_zorder(3)
            # twinx draws ax2 on top of ax1 by default. Promote ax1 above ax2 so
            # ax1 scatter (zorder=10) covers ax2's inward right tick marks at x=500.
            ax1.set_zorder(ax2.get_zorder() + 1)
            ax1.patch.set_visible(False)
            # Hide ax1's right spine so ax2's right spine (with SM labels) shows.
            # Without this, ax1's right spine would draw over ax2's SM scatter near x=500.
            ax1.spines['right'].set_visible(False)

        # Compose legend: include SM and controls when applicable
        if sm in ("DSM", "NDSM"):
            handles = [
                line_sg,
                line_sg_ctrl,
                line_protein,
                line_protein_ctrl,
                line_rna,
                line_rna_ctrl,
                line_sm,
            ]
            labels = [
                'SG',
                'SG Control',
                'Protein',
                'Protein Control',
                'RNA',
                'RNA Control',
                sm,
            ]
        else:
            handles = [line_sg, line_protein, line_rna]
            labels = ['SG', 'Protein', 'RNA']
        leg = ax1.legend(handles, labels, loc='upper right', ncol=1,
                         borderpad=0.3, frameon=False)

        plt.savefig("{}/FIGURES/RDP/{}_RDP.png".format(self.analysis_root,sm), format="png", dpi=400)
        if sm == "SG":
            try:
                self._plot_sg_rdp_crossover_zoom(
                    fitterSG,
                    fitterProtein,
                    fitterRNA,
                    self.analysis_root,
                    col_pal_sg,
                    col_pal_protein,
                    col_pal_rna,
                )
            except Exception as exc:
                print(f"SG RDP crossover zoom failed: {exc}")
        return fig

    def _plot_sg_rdp_crossover_zoom(self, fitterSG, fitterProtein, fitterRNA,
                                    analysis_root, col_pal_sg, col_pal_protein, col_pal_rna):
        """Save a +-50 A zoom of the SG RDP centered on the protein/RNA fit crossover.

        Locates where the protein and RNA fit curves cross in the dilute
        (low-density) regime and replots SG/protein/RNA over that window to
        FIGURES/RDP/SG_RDP_CROSSOVER_ZOOM.png.
        """
        def _clean_curve(x, y):
            x_arr = np.asarray(x, dtype=float)
            y_arr = np.asarray(y, dtype=float)
            mask = np.isfinite(x_arr) & np.isfinite(y_arr)
            x_arr = x_arr[mask]
            y_arr = y_arr[mask]
            if x_arr.size == 0:
                return x_arr, y_arr
            order = np.argsort(x_arr)
            return x_arr[order], y_arr[order]

        def _protein_rna_crossover_x():
            xp, yp = _clean_curve(fitterProtein.fit_x, fitterProtein.fit_rho)
            xr, yr = _clean_curve(fitterRNA.fit_x, fitterRNA.fit_rho)
            if xp.size < 2 or xr.size < 2:
                return 200.0
            lo = max(float(np.nanmin(xp)), float(np.nanmin(xr)))
            hi = min(float(np.nanmax(xp)), float(np.nanmax(xr)))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                return 200.0
            grid = np.linspace(lo, hi, 2000)
            diff = np.interp(grid, xp, yp) - np.interp(grid, xr, yr)
            finite = np.isfinite(diff)
            if not np.any(finite):
                return 200.0
            grid = grid[finite]
            diff = diff[finite]
            sign_changes = np.where(np.signbit(diff[:-1]) != np.signbit(diff[1:]))[0]
            candidates = []
            for idx in sign_changes:
                x0, x1 = grid[idx], grid[idx + 1]
                y0, y1 = diff[idx], diff[idx + 1]
                if abs(y1 - y0) < 1e-12:
                    x_cross = 0.5 * (x0 + x1)
                else:
                    x_cross = x0 - y0 * (x1 - x0) / (y1 - y0)
                p_cross = np.interp(x_cross, xp, yp)
                r_cross = np.interp(x_cross, xr, yr)
                candidates.append((x_cross, 0.5 * (p_cross + r_cross)))
            if candidates:
                low_density = [c for c in candidates if 0.0 <= c[1] <= 120.0]
                if low_density:
                    return float(max(low_density, key=lambda c: c[0])[0])
                return float(max(candidates, key=lambda c: c[0])[0])
            idx = int(np.nanargmin(np.abs(diff)))
            return float(grid[idx])

        center = _protein_rna_crossover_x()
        xlim_zoom = (max(0.0, center - 50.0), center + 50.0)

        sns.set_theme(style="ticks")
        sns.set_style('white')
        plt.rc('axes', titlesize=10)
        plt.rc('axes', labelsize=10)
        plt.rc('xtick', labelsize=10)
        plt.rc('ytick', labelsize=10)
        plt.rc('legend', fontsize=8)
        plt.rc('font', size=10)
        plt.rc('axes', linewidth=2)

        fig, ax1 = _make_rdp_fig()

        def _filter_fit(fitter):
            x_arr, y_arr = _clean_curve(fitter.fit_x, fitter.fit_rho)
            mask = (x_arr >= xlim_zoom[0]) & (x_arr <= xlim_zoom[1])
            return x_arr[mask], y_arr[mask]

        def _filter_scatter_err(fitter, n=24):
            d = np.asarray(fitter.distances[:n], dtype=float)
            rho = np.asarray(fitter.densities[:n], dtype=float)
            m = (d >= xlim_zoom[0]) & (d <= xlim_zoom[1])
            d_all = np.asarray(fitter.distances, dtype=float)
            rho_all = np.asarray(fitter.densities, dtype=float)
            e_all = np.asarray(fitter.errors, dtype=float)
            m_all = (d_all >= xlim_zoom[0]) & (d_all <= xlim_zoom[1])
            sx, sy = _rdp_plot_points(d[m], rho[m])
            ex, ey, ee = _rdp_plot_points(d_all[m_all], rho_all[m_all], e_all[m_all])
            return sx, sy, ex, ey, ee

        sg_fit_x, sg_fit_rho = _filter_fit(fitterSG)
        protein_fit_x, protein_fit_rho = _filter_fit(fitterProtein)
        rna_fit_x, rna_fit_rho = _filter_fit(fitterRNA)

        sx, sy, ex, ey, ee = _filter_scatter_err(fitterSG)
        line_sg, = ax1.plot(sg_fit_x, sg_fit_rho, color=col_pal_sg, label='SG', linewidth=2, zorder=1, clip_on=True)
        sns.scatterplot(ax=ax1, x=sx, y=sy, color=col_pal_sg, legend=False, s=40,
                        edgecolor="k", linewidth=1, zorder=10, clip_on=False)
        ax1.errorbar(x=ex, y=ey, yerr=ee, fmt="none", color=col_pal_sg, zorder=4, capsize=2)

        sx, sy, ex, ey, ee = _filter_scatter_err(fitterProtein)
        line_protein, = ax1.plot(protein_fit_x, protein_fit_rho, color=col_pal_protein, label='Protein',
                                 linewidth=2, zorder=1, clip_on=True)
        sns.scatterplot(ax=ax1, x=sx, y=sy, color=col_pal_protein, legend=False, s=40,
                        edgecolor="k", linewidth=1, zorder=10, clip_on=False)
        ax1.errorbar(ex, ey, yerr=ee, fmt="none", color=col_pal_protein, zorder=4, capsize=2)

        sx, sy, ex, ey, ee = _filter_scatter_err(fitterRNA)
        line_rna, = ax1.plot(rna_fit_x, rna_fit_rho, color=col_pal_rna, label='RNA',
                             linewidth=2, zorder=1, clip_on=True)
        sns.scatterplot(ax=ax1, x=sx, y=sy, color=col_pal_rna, legend=False, s=40,
                        edgecolor="k", linewidth=1, zorder=10, clip_on=False)
        ax1.errorbar(ex, ey, yerr=ee, fmt="none", color=col_pal_rna, zorder=4, capsize=2)

        ax1.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in',
                        length=4, width=2)
        ax1.set_xlim(xlim_zoom)
        ax1.set_ylim(0, 100)
        for spine in ax1.spines.values():
            spine.set_zorder(3)
        fig.legend([line_sg, line_protein, line_rna], ['SG', 'Protein', 'RNA'],
                   loc='upper right', ncol=1, bbox_to_anchor=(0.9, 0.85),
                   borderpad=0.3, frameon=False)
        fig.savefig("{}/FIGURES/RDP/SG_RDP_CROSSOVER_ZOOM.png".format(analysis_root),
                    format="png", dpi=400)
        plt.close(fig)

    def plot_res_cpm(self, res_contact_array, col, min, max, mid, figsize=RESIDUE_CONTACT_MAP_FIGSIZE):
        """Render a square 7x7 biopolymer-species contact heatmap (colormap *col*, [min,max]).

        Rows/cols are reordered into the SG half-max-radius display order and the
        y-axis inverted so the first species sits at the bottom. *mid* is accepted
        for signature parity with the other contact-map plotters but is unused
        (the colorbar limits come from min/max). Returns the figure.
        """
        plt.close('all')
        res_contact_array = _sanitize_heatmap_array(res_contact_array)
        res_contact_array, x_res_list = self._reorder_residue_matrix(res_contact_array)

        sns.set_theme(style="ticks")
        sns.set_style('white')
        plt.rc('axes', titlesize=10)
        plt.rc('axes', labelsize=10)
        plt.rc('xtick', labelsize=8)
        plt.rc('ytick', labelsize=8)
        plt.rc('legend', fontsize=8)
        plt.rc('font', size=10)
        plt.rc('axes', linewidth=2)

        y_res_list = x_res_list

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes(_CONTACT_AX_RECT)
        sns.heatmap(res_contact_array, xticklabels=x_res_list, yticklabels=y_res_list, square=True, cmap=col, cbar=False, ax=ax, vmin=min, vmax=max)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)
        _apply_contact_heatmap_ticks(ax)
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=45)
        for lbl in ax.get_xticklabels():
            lbl.set_ha('center')
            lbl.set_va('top')
            lbl.set_rotation_mode('default')
        for lbl in ax.get_yticklabels():
            lbl.set_ha('right')
            lbl.set_va('center')
            lbl.set_rotation_mode('default')
        # Invert y-axis so row 0 (FUS) sits at the bottom
        ax.invert_yaxis()
        _setup_cbar_fixed(fig, ax, _CONTACT_AX_RECT)

        return fig

    def _log_transform_cmap(self, array):
        """Return -ln(C / C_max) with the same mask as the input (NaN where non-positive)."""
        arr = np.asarray(array, dtype=float)
        finite_pos = np.isfinite(arr) & (arr > 0)
        if not np.any(finite_pos):
            return None, None
        max_val = float(np.nanmax(arr[finite_pos]))
        out = np.full_like(arr, np.nan, dtype=float)
        out[finite_pos] = -np.log(arr[finite_pos] / max_val)
        finite_out = np.isfinite(out)
        if not np.any(finite_out):
            return None, None
        log_max = float(np.nanmax(out[finite_out]))
        if not np.isfinite(log_max) or log_max <= 0:
            log_max = 1.0
        return out, log_max

    def _save_res_cpm_lin_log(self, array, output_path, col, min, max, mid):
        """Save residue contact map as linear (output_path) and -ln(C/C_max) (_LOG.png) variants.
        LOG variant uses Reds_r so dark = low value (strong contact / low free energy), matching domain CM."""
        fig = self.plot_res_cpm(array, col=col, min=min, max=max, mid=mid)
        plt.savefig(output_path, format="png", dpi=400)
        plt.close(fig)
        log_arr, log_max = self._log_transform_cmap(array)
        if log_arr is None:
            return
        fig = self.plot_res_cpm(log_arr, col="Reds_r", min=0.0, max=log_max, mid=log_max / 2.0)
        plt.savefig(output_path.replace(".png", "_LOG.png"), format="png", dpi=400)
        plt.close(fig)

    def _log_diff_cmap(self, arr_a, arr_b):
        """Return ln(A / B) in nats. Sign convention matches linear DIFF (A - B):
        positive when A has more contacts than B, negative when B has more.
        NaN where either input is non-positive.
        Note: uses raw values (no per-array max normalization) so absolute differences
        between conditions are preserved — unlike a difference of two -ln(X/X_max) terms,
        which would cancel out the per-condition scale.
        """
        a = np.asarray(arr_a, dtype=float)
        b = np.asarray(arr_b, dtype=float)
        finite = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
        out = np.full_like(a, np.nan, dtype=float)
        out[finite] = np.log(a[finite] / b[finite])
        finite_out = np.isfinite(out)
        if not np.any(finite_out):
            return None, None
        vmax = float(np.nanmax(np.abs(out[finite_out])))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        return out, vmax

    def _save_res_diff_log(self, arr_a, arr_b, output_path):
        """Save -ln(A/A_max) - -ln(B/B_max) residue contact map in nats (coolwarm centered at 0)."""
        diff, vmax = self._log_diff_cmap(arr_a, arr_b)
        if diff is None:
            return
        fig = self.plot_res_cpm(diff, col="coolwarm", min=-vmax, max=vmax, mid=0.0)
        plt.savefig(output_path, format="png", dpi=400)
        plt.close(fig)

    def _save_acid_diff_log(self, arr_a, arr_b, output_path):
        """Save -ln(A/A_max) - -ln(B/B_max) acid contact map in nats (coolwarm centered at 0)."""
        diff, vmax = self._log_diff_cmap(arr_a, arr_b)
        if diff is None:
            return
        fig = self.plot_acid_cpm(diff, col="coolwarm", min=-vmax, max=vmax, mid=0.0)
        plt.savefig(output_path, format="png", dpi=400)
        plt.close(fig)

    def _save_acid_cpm_lin_log(self, array, output_path, col, min, max, mid):
        """Save acid contact map as linear (output_path) and -ln(C/C_max) (_LOG.png) variants.
        LOG variant uses Reds_r so dark = low value (strong contact / low free energy), matching domain CM."""
        fig = self.plot_acid_cpm(array, col=col, min=min, max=max, mid=mid)
        plt.savefig(output_path, format="png", dpi=400)
        plt.close(fig)
        log_arr, log_max = self._log_transform_cmap(array)
        if log_arr is None:
            return
        fig = self.plot_acid_cpm(log_arr, col="Reds_r", min=0.0, max=log_max, mid=log_max / 2.0)
        plt.savefig(output_path.replace(".png", "_LOG.png"), format="png", dpi=400)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Helpers for the unified P_contact / K_chain scheme.
    # P_contact: per-pair contact probability in [0,1] -> -ln(P) is apparent
    #            free energy in nats (k_BT).
    # K_chain:   contacts per chain pair (a count, not a probability) -> log10(K)
    #            is a display transform only, NOT a free energy.
    # ------------------------------------------------------------------

    def _neg_log_abs(self, P):
        """Return -ln(P), NaN where P<=0 or non-finite. P is assumed to be a probability."""
        a = np.asarray(P, dtype=float)
        out = np.full_like(a, np.nan)
        mask = np.isfinite(a) & (a > 0)
        out[mask] = -np.log(a[mask])
        return out

    def _log10_arr(self, K):
        """Return log10(K), NaN where K<=0 or non-finite. Display transform for counts."""
        a = np.asarray(K, dtype=float)
        out = np.full_like(a, np.nan)
        mask = np.isfinite(a) & (a > 0)
        out[mask] = np.log10(a[mask])
        return out

    @staticmethod
    def _shared_vmax(*arrs):
        """Max over all finite entries across input arrays. Returns 1.0 if no finite data."""
        vmax = 0.0
        for a in arrs:
            a = np.asarray(a, dtype=float)
            finite = np.isfinite(a)
            if np.any(finite):
                v = float(np.nanmax(a[finite]))
                if v > vmax:
                    vmax = v
        return vmax if vmax > 0 else 1.0

    @staticmethod
    def _shared_vmin(*arrs):
        """Min over all finite entries across input arrays. Returns 0.0 if no finite data."""
        vmin = None
        for a in arrs:
            a = np.asarray(a, dtype=float)
            finite = np.isfinite(a)
            if np.any(finite):
                v = float(np.nanmin(a[finite]))
                vmin = v if vmin is None else min(vmin, v)
        return vmin if vmin is not None else 0.0

    @staticmethod
    def _sym_vmax_arr(*arrs):
        """Robust symmetric vmax for signed/difference maps: the 99th percentile
        of |value| pooled over all finite entries (not the absolute max), so a
        few extreme cells don't compress a heavy-tailed difference map into
        near-grey. For small maps (7x7, 24x24) the 99th percentile is essentially
        the max, so coarse difference maps are visually unchanged; the clip only
        bites on large, heavy-tailed maps. Returns 1.0 if no finite data."""
        pool = []
        for a in arrs:
            a = np.asarray(a, dtype=float)
            finite = np.isfinite(a)
            if np.any(finite):
                pool.append(np.abs(a[finite]))
        if not pool:
            return 1.0
        vmax = float(np.nanpercentile(np.concatenate(pool), 99))
        return vmax if vmax > 0 else 1.0

    def _find_cm_norm_with_sequences(self):
        """Locate the directory containing the *_seq.txt/RNA.txt sequence files.

        New runs write normalization maps under ``analysis_root/CM_NORM/MAPS``
        but the sequence inputs are committed under ``analysis/sequences``.
        """
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = [
            os.path.join(self.analysis_root, "CM_NORM"),
            os.path.join(repo_root, "analysis", "sequences"),
            os.path.join(os.getcwd(), "analysis", "sequences"),
            "CM_NORM",
        ]
        for cand in candidates:
            if (
                os.path.isfile(os.path.join(cand, "G3BP1_seq.txt"))
                and os.path.isfile(os.path.join(cand, "RNA.txt"))
            ):
                return cand
        raise FileNotFoundError(
            "Could not locate sequence directory containing G3BP1_seq.txt and RNA.txt; "
            f"searched: {candidates}"
        )

    def plot_res_sm_cpm(self, res_contact_array, col, min, max, mid, sm_list,
                        figsize=None, show_xticks=True):
        """Render an n_residue x len(sm_list) SM/biopolymer contact heatmap.

        Used for the 1D row vectors and the per-SM SM-contact maps. Rows are in
        residue source order, columns labelled by *sm_list*; cells are square via
        ``_sm_compute_layout`` so *figsize* and *mid* are accepted only for
        signature parity and are unused. When *show_xticks* is False (forced for
        single-column maps) the x labels are hidden. Returns the figure.
        """
        plt.close('all')
        res_contact_array = _sanitize_heatmap_array(res_contact_array)
        res_contact_array, x_res_list = self._reorder_residue_rows(res_contact_array)

        sns.set_theme(style="ticks")
        sns.set_style('white')
        plt.rc('axes', titlesize=10)
        plt.rc('axes', labelsize=10)
        plt.rc('xtick', labelsize=8)
        plt.rc('ytick', labelsize=8)
        plt.rc('legend', fontsize=8)
        plt.rc('font', size=10)
        plt.rc('axes', linewidth=2)

        n_rows, n_cols = res_contact_array.shape
        _figsize, ax_rect = _sm_compute_layout(n_rows, n_cols)
        if n_cols == 1:
            show_xticks = False
        fig = plt.figure(figsize=_figsize)
        ax = fig.add_axes(ax_rect)
        # Honour the requested colour range only when it is valid (some legacy
        # callers pass inverted/placeholder min,max that must fall back to autoscale).
        _vmin, _vmax = (min, max) if (min is not None and max is not None and min < max) else (None, None)
        sns.heatmap(res_contact_array, xticklabels=sm_list, yticklabels=x_res_list,
                    cmap=col, square=False, cbar=False, ax=ax, vmin=_vmin, vmax=_vmax)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)
        _apply_contact_heatmap_ticks(ax)
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=45)
        if not show_xticks:
            ax.tick_params(axis='x', which='both', bottom=False, top=False,
                           labelbottom=False, labeltop=False)
            ax.set_xticklabels([])
        for lbl in ax.get_xticklabels():
            lbl.set_ha('center')
            lbl.set_va('top')
            lbl.set_rotation_mode('default')
        for lbl in ax.get_yticklabels():
            lbl.set_ha('right')
            lbl.set_va('center')
            lbl.set_rotation_mode('default')
        # Invert y-axis so row 0 (first residue) sits at the bottom
        ax.invert_yaxis()
        _setup_cbar_fixed(fig, ax, ax_rect,
                          cbar_pad_in=_SM_CBAR_PAD_IN, cbar_w_in=_SM_CBAR_W_IN)
        return fig

    def gen_residue_cpms(self):
        """Generate the legacy 7x7 biopolymer-species residue contact maps.

        For each available condition writes individual maps (UNNORMALIZED and
        per-SYSTEM / per-CLUSTER normalized), their 1D row sums and -ln companions,
        and DSM-minus-NDSM difference maps, plus the per-SM SM-residue contact maps
        when DSM/NDSM data are present. Outputs go to
        FIGURES/RESIDUE_CONTACT_MAPS/ and FIGURES/SM_CONTACT_MAPS/.
        """
        conds = self._available_conditions()

        # Per-category arrays loaded only if the category is available.
        nBio_by = {}
        res_arr_by = {}
        if "SG" in conds:
            nBio_by["SG"] = np.loadtxt(self._analysis_file("ANALYSIS_SG_AVE", "BioPolNum_sg_X.csv"), delimiter=",", dtype=float)
            res_arr_by["SG"] = np.array(pd.read_csv(self._analysis_file("ANALYSIS_SG_AVE", "Residue_Contacts_Mean_sg_X.csv"), header=None))
        if "DSM" in conds:
            nBio_by["DSM"] = np.loadtxt(self._analysis_file("ANALYSIS_DSM_AVE", "BioPolNum_DSM.csv"), delimiter=",", dtype=float)
            res_arr_by["DSM"] = np.array(pd.read_csv(self._analysis_file("ANALYSIS_DSM_AVE", "Residue_Contacts_Mean_DSM.csv"), header=None))
        if "NDSM" in conds:
            nBio_by["NDSM"] = np.loadtxt(self._analysis_file("ANALYSIS_NDSM_AVE", "BioPolNum_NDSM.csv"), delimiter=",", dtype=float)
            res_arr_by["NDSM"] = np.array(pd.read_csv(self._analysis_file("ANALYSIS_NDSM_AVE", "Residue_Contacts_Mean_NDSM.csv"), header=None))
        nBio = np.loadtxt(os.path.join(self.analysis_root, "CM_NORM", "MAPS", "BioPolNum_SYSTEM.csv"), delimiter=",", dtype=float)

        # Backward-compatible aliases for code paths that historically used
        # explicit *_res_arr / nBio_* names. They're only referenced when the
        # owning condition is in `conds`, but defining them keeps later blocks
        # readable without renaming everything.
        sg_res_arr = res_arr_by.get("SG")
        dsm_res_arr = res_arr_by.get("DSM")
        ndsm_res_arr = res_arr_by.get("NDSM")
        nBio_sg = nBio_by.get("SG")
        nBio_dsm = nBio_by.get("DSM")
        nBio_ndsm = nBio_by.get("NDSM")

        # INDIVIDUAL MAPS
        # UNNORMALIZED
        if "SG" in conds:
            res_contact_array = np.divide(sg_res_arr, (np.sum(sg_res_arr)))
            self._save_res_cpm_lin_log(res_contact_array,
                "{}/FIGURES/RESIDUE_CONTACT_MAPS/Residue_SG_UNNORMALIZED_HeatMap.png".format(self.analysis_root),
                col="Reds", min=0, max=0.07, mid=0.035)

        if "DSM" in conds:
            res_contact_array = np.divide(dsm_res_arr, (np.sum(dsm_res_arr)))
            self._save_res_cpm_lin_log(res_contact_array,
                "{}/FIGURES/RESIDUE_CONTACT_MAPS/Residue_DSM_UNNORMALIZED_HeatMap.png".format(self.analysis_root),
                col="Reds", min=0, max=0.07, mid=0.035)

        if "NDSM" in conds:
            res_contact_array = np.divide(ndsm_res_arr, (np.sum(ndsm_res_arr)))
            self._save_res_cpm_lin_log(res_contact_array,
                "{}/FIGURES/RESIDUE_CONTACT_MAPS/Residue_NDSM_UNNORMALIZED_HeatMap.png".format(self.analysis_root),
                col="Reds", min=0, max=0.07, mid=0.035)

        # NORMALIZED SYSTEM
        if "SG" in conds:
            res_contact_array = np.divide(sg_res_arr, (np.sum(sg_res_arr)*nBio))
            self._save_res_cpm_lin_log(res_contact_array,
                "{}/FIGURES/RESIDUE_CONTACT_MAPS/Residue_SG_NORMALIZED_SYSTEM_HeatMap.png".format(self.analysis_root),
                col="Reds", min=0, max=0.0004, mid=0.0002)

        if "DSM" in conds:
            res_contact_array = np.divide(dsm_res_arr, (np.sum(dsm_res_arr)*nBio))
            self._save_res_cpm_lin_log(res_contact_array,
                "{}/FIGURES/RESIDUE_CONTACT_MAPS/Residue_DSM_NORMALIZED_SYSTEM_HeatMap.png".format(self.analysis_root),
                col="Reds", min=0, max=0.0004, mid=0.0002)

        if "NDSM" in conds:
            res_contact_array = np.divide(ndsm_res_arr, (np.sum(ndsm_res_arr)*nBio))
            self._save_res_cpm_lin_log(res_contact_array,
                "{}/FIGURES/RESIDUE_CONTACT_MAPS/Residue_NDSM_NORMALIZED_SYSTEM_HeatMap.png".format(self.analysis_root),
                col="Reds", min=0, max=0.0004, mid=0.0002)

        # NORMALIZED CLUSTER
        if "SG" in conds:
            res_contact_array = np.divide(sg_res_arr, (np.sum(sg_res_arr) * nBio_sg))
            self._save_res_cpm_lin_log(res_contact_array,
                "{}/FIGURES/RESIDUE_CONTACT_MAPS/Residue_SG_NORMALIZED_CLUSTER_HeatMap.png".format(self.analysis_root),
                col="Reds", min=0, max=0.0003, mid=0.00015)

        if "DSM" in conds:
            res_contact_array = np.divide(dsm_res_arr, (np.sum(dsm_res_arr) * nBio_dsm))
            self._save_res_cpm_lin_log(res_contact_array,
                "{}/FIGURES/RESIDUE_CONTACT_MAPS/Residue_DSM_NORMALIZED_CLUSTER_HeatMap.png".format(self.analysis_root),
                col="Reds", min=0, max=0.0003, mid=0.00015)

        if "NDSM" in conds:
            res_contact_array = np.divide(ndsm_res_arr, (np.sum(ndsm_res_arr) * nBio_ndsm))
            self._save_res_cpm_lin_log(res_contact_array,
                "{}/FIGURES/RESIDUE_CONTACT_MAPS/Residue_NDSM_NORMALIZED_CLUSTER_HeatMap.png".format(self.analysis_root),
                col="Reds", min=0, max=0.0003, mid=0.00015)

        # 1D INDIVIDUAL MAPS
        # 1D UNNORMALIZED
        if "SG" in conds:
            res_1d = np.divide(sg_res_arr, np.sum(sg_res_arr)).sum(axis=1).reshape(-1, 1)
            fig = self.plot_res_sm_cpm(res_1d, col="Reds", min=0, max=1, mid=0, sm_list=[""], figsize=(1.6, 3.2))
            plt.savefig("{}/FIGURES/RESIDUE_CONTACT_MAPS/Residue_SG_UNNORMALIZED_1D_HeatMap.png".format(
                self.analysis_root), format="png", dpi=400)

        if "DSM" in conds:
            res_1d = np.divide(dsm_res_arr, np.sum(dsm_res_arr)).sum(axis=1).reshape(-1, 1)
            fig = self.plot_res_sm_cpm(res_1d, col="Reds", min=0, max=1, mid=0, sm_list=[""], figsize=(1.6, 3.2))
            plt.savefig("{}/FIGURES/RESIDUE_CONTACT_MAPS/Residue_DSM_UNNORMALIZED_1D_HeatMap.png".format(
                self.analysis_root), format="png", dpi=400)

        if "NDSM" in conds:
            res_1d = np.divide(ndsm_res_arr, np.sum(ndsm_res_arr)).sum(axis=1).reshape(-1, 1)
            fig = self.plot_res_sm_cpm(res_1d, col="Reds", min=0, max=1, mid=0, sm_list=[""], figsize=(1.6, 3.2))
            plt.savefig("{}/FIGURES/RESIDUE_CONTACT_MAPS/Residue_NDSM_UNNORMALIZED_1D_HeatMap.png".format(
                self.analysis_root), format="png", dpi=400)

        # 1D NORMALIZED SYSTEM
        if "SG" in conds:
            res_1d = np.divide(sg_res_arr, np.sum(sg_res_arr) * nBio).sum(axis=1).reshape(-1, 1)
            fig = self.plot_res_sm_cpm(res_1d, col="Reds", min=0, max=1, mid=0, sm_list=[""], figsize=(1.6, 3.2))
            plt.savefig("{}/FIGURES/RESIDUE_CONTACT_MAPS/Residue_SG_NORMALIZED_SYSTEM_1D_HeatMap.png".format(
                self.analysis_root), format="png", dpi=400)

        if "DSM" in conds:
            res_1d = np.divide(dsm_res_arr, np.sum(dsm_res_arr) * nBio).sum(axis=1).reshape(-1, 1)
            fig = self.plot_res_sm_cpm(res_1d, col="Reds", min=0, max=1, mid=0, sm_list=[""], figsize=(1.6, 3.2))
            plt.savefig("{}/FIGURES/RESIDUE_CONTACT_MAPS/Residue_DSM_NORMALIZED_SYSTEM_1D_HeatMap.png".format(
                self.analysis_root), format="png", dpi=400)

        if "NDSM" in conds:
            res_1d = np.divide(ndsm_res_arr, np.sum(ndsm_res_arr) * nBio).sum(axis=1).reshape(-1, 1)
            fig = self.plot_res_sm_cpm(res_1d, col="Reds", min=0, max=1, mid=0, sm_list=[""], figsize=(1.6, 3.2))
            plt.savefig("{}/FIGURES/RESIDUE_CONTACT_MAPS/Residue_NDSM_NORMALIZED_SYSTEM_1D_HeatMap.png".format(
                self.analysis_root), format="png", dpi=400)

        # 1D NORMALIZED CLUSTER
        if "SG" in conds:
            res_1d = np.divide(sg_res_arr, np.sum(sg_res_arr) * nBio_sg).sum(axis=1).reshape(-1, 1)
            fig = self.plot_res_sm_cpm(res_1d, col="Reds", min=0, max=1, mid=0, sm_list=[""], figsize=(1.6, 3.2))
            plt.savefig("{}/FIGURES/RESIDUE_CONTACT_MAPS/Residue_SG_NORMALIZED_CLUSTER_1D_HeatMap.png".format(
                self.analysis_root), format="png", dpi=400)

        if "DSM" in conds:
            res_1d = np.divide(dsm_res_arr, np.sum(dsm_res_arr) * nBio_dsm).sum(axis=1).reshape(-1, 1)
            fig = self.plot_res_sm_cpm(res_1d, col="Reds", min=0, max=1, mid=0, sm_list=[""], figsize=(1.6, 3.2))
            plt.savefig("{}/FIGURES/RESIDUE_CONTACT_MAPS/Residue_DSM_NORMALIZED_CLUSTER_1D_HeatMap.png".format(
                self.analysis_root), format="png", dpi=400)

        if "NDSM" in conds:
            res_1d = np.divide(ndsm_res_arr, np.sum(ndsm_res_arr) * nBio_ndsm).sum(axis=1).reshape(-1, 1)
            fig = self.plot_res_sm_cpm(res_1d, col="Reds", min=0, max=1, mid=0, sm_list=[""], figsize=(1.6, 3.2))
            plt.savefig("{}/FIGURES/RESIDUE_CONTACT_MAPS/Residue_NDSM_NORMALIZED_CLUSTER_1D_HeatMap.png".format(
                self.analysis_root), format="png", dpi=400)

        def _sym_vmax(arr):
            finite = np.isfinite(arr)
            if not np.any(finite):
                return 1.0
            v = float(np.nanmax(np.abs(arr[finite])))
            return v if (np.isfinite(v) and v > 0) else 1.0

        # DIFFERENCE MAPS — require DSM and NDSM (and SG denominator)
        if {"SG", "DSM", "NDSM"}.issubset(conds):
            # UNNORMALIZED
            res_contact_array = np.divide(np.subtract(dsm_res_arr, ndsm_res_arr), sg_res_arr)
            _vmax = _sym_vmax(res_contact_array)
            fig = self.plot_res_cpm(res_contact_array, col="coolwarm", min=-_vmax, max=_vmax, mid=0.0)
            plt.savefig("{}/FIGURES/RESIDUE_CONTACT_MAPS/Residue_DIFF_UNNORMALIZED_HeatMap.png".format(
                self.analysis_root), format="png", dpi=400)
            self._save_res_diff_log(dsm_res_arr, ndsm_res_arr,
                "{}/FIGURES/RESIDUE_CONTACT_MAPS/Residue_DIFF_UNNORMALIZED_HeatMap_LOG.png".format(self.analysis_root))

            # UNNORMALIZED PERCENTAGE
            dsm_sum = np.divide(dsm_res_arr,np.sum(dsm_res_arr))
            ndsm_sum = np.divide(ndsm_res_arr,np.sum(ndsm_res_arr))
            sg_sum = np.divide(sg_res_arr,np.sum(sg_res_arr))
            res_contact_array = np.divide(np.subtract(dsm_sum, ndsm_sum), sg_sum)
            _vmax = _sym_vmax(res_contact_array)
            fig = self.plot_res_cpm(res_contact_array, col="coolwarm", min=-_vmax, max=_vmax, mid=0.0)
            plt.savefig("{}/FIGURES/RESIDUE_CONTACT_MAPS/Residue_DIFF_UNNORMALIZED_PERCENTAGE_HeatMap.png".format(
                self.analysis_root), format="png", dpi=400)
            self._save_res_diff_log(dsm_sum, ndsm_sum,
                "{}/FIGURES/RESIDUE_CONTACT_MAPS/Residue_DIFF_UNNORMALIZED_PERCENTAGE_HeatMap_LOG.png".format(self.analysis_root))

            # NORMALIZED SYSTEM
            sg_norm_arr = np.divide(sg_res_arr, (np.sum(sg_res_arr))*nBio)
            dsm_norm_arr = np.divide(dsm_res_arr, (np.sum(dsm_res_arr))*nBio)
            ndsm_norm_arr = np.divide(ndsm_res_arr, (np.sum(ndsm_res_arr))*nBio)
            res_contact_array = np.divide(np.subtract(dsm_norm_arr, ndsm_norm_arr), sg_norm_arr)
            _vmax = _sym_vmax(res_contact_array)
            fig = self.plot_res_cpm(res_contact_array, col="coolwarm", min=-_vmax, max=_vmax, mid=0.0)
            plt.savefig("{}/FIGURES/RESIDUE_CONTACT_MAPS/Residue_DIFF_NORMALIZED_SYSTEM_HeatMap.png".format(
                self.analysis_root), format="png", dpi=400)
            self._save_res_diff_log(dsm_norm_arr, ndsm_norm_arr,
                "{}/FIGURES/RESIDUE_CONTACT_MAPS/Residue_DIFF_NORMALIZED_SYSTEM_HeatMap_LOG.png".format(self.analysis_root))

            # NORMALIZED CLUSTER
            sg_norm_arr = np.divide(sg_res_arr, (np.sum(sg_res_arr)*nBio_sg))
            dsm_norm_arr = np.divide(dsm_res_arr, (np.sum(dsm_res_arr)*nBio_dsm))
            ndsm_norm_arr = np.divide(ndsm_res_arr, (np.sum(ndsm_res_arr)*nBio_ndsm))
            res_contact_array = np.divide(np.subtract(dsm_norm_arr, ndsm_norm_arr), sg_norm_arr)
            _vmax = _sym_vmax(res_contact_array)
            fig = self.plot_res_cpm(res_contact_array, col="coolwarm", min=-_vmax, max=_vmax, mid=0.0)
            plt.savefig("{}/FIGURES/RESIDUE_CONTACT_MAPS/Residue_DIFF_NORMALIZED_CLUSTER_HeatMap.png".format(
                self.analysis_root), format="png", dpi=400)
            self._save_res_diff_log(dsm_norm_arr, ndsm_norm_arr,
                "{}/FIGURES/RESIDUE_CONTACT_MAPS/Residue_DIFF_NORMALIZED_CLUSTER_HeatMap_LOG.png".format(self.analysis_root))

            # 1D DIFFERENCE MAPS
            # 1D DIFF UNNORMALIZED
            res_1d = np.divide(np.subtract(dsm_res_arr, ndsm_res_arr), sg_res_arr).sum(axis=1).reshape(-1, 1)
            fig = self.plot_res_sm_cpm(res_1d, col="Blues_r", min=0, max=1, mid=0, sm_list=[""], figsize=(1.6, 3.2))
            plt.savefig("{}/FIGURES/RESIDUE_CONTACT_MAPS/Residue_DIFF_UNNORMALIZED_1D_HeatMap.png".format(
                self.analysis_root), format="png", dpi=400)

            # 1D DIFF UNNORMALIZED PERCENTAGE
            dsm_sum = np.divide(dsm_res_arr, np.sum(dsm_res_arr))
            ndsm_sum = np.divide(ndsm_res_arr, np.sum(ndsm_res_arr))
            sg_sum = np.divide(sg_res_arr, np.sum(sg_res_arr))
            res_1d = np.divide(np.subtract(dsm_sum, ndsm_sum), sg_sum).sum(axis=1).reshape(-1, 1)
            fig = self.plot_res_sm_cpm(res_1d, col="coolwarm", min=0, max=1, mid=0, sm_list=[""], figsize=(1.6, 3.2))
            plt.savefig("{}/FIGURES/RESIDUE_CONTACT_MAPS/Residue_DIFF_UNNORMALIZED_PERCENTAGE_1D_HeatMap.png".format(
                self.analysis_root), format="png", dpi=400)

            # 1D DIFF NORMALIZED SYSTEM
            sg_norm_arr = np.divide(sg_res_arr, np.sum(sg_res_arr) * nBio)
            dsm_norm_arr = np.divide(dsm_res_arr, np.sum(dsm_res_arr) * nBio)
            ndsm_norm_arr = np.divide(ndsm_res_arr, np.sum(ndsm_res_arr) * nBio)
            res_1d = np.divide(np.subtract(dsm_norm_arr, ndsm_norm_arr), sg_norm_arr).sum(axis=1).reshape(-1, 1)
            fig = self.plot_res_sm_cpm(res_1d, col="coolwarm", min=0, max=1, mid=0, sm_list=[""], figsize=(1.6, 3.2))
            plt.savefig("{}/FIGURES/RESIDUE_CONTACT_MAPS/Residue_DIFF_NORMALIZED_SYSTEM_1D_HeatMap.png".format(
                self.analysis_root), format="png", dpi=400)

            # 1D DIFF NORMALIZED CLUSTER
            sg_norm_arr = np.divide(sg_res_arr, np.sum(sg_res_arr) * nBio_sg)
            dsm_norm_arr = np.divide(dsm_res_arr, np.sum(dsm_res_arr) * nBio_dsm)
            ndsm_norm_arr = np.divide(ndsm_res_arr, np.sum(ndsm_res_arr) * nBio_ndsm)
            res_1d = np.divide(np.subtract(dsm_norm_arr, ndsm_norm_arr), sg_norm_arr).sum(axis=1).reshape(-1, 1)
            fig = self.plot_res_sm_cpm(res_1d, col="Reds", min=0, max=1, mid=0, sm_list=[""], figsize=(1.6, 3.2))
            plt.savefig("{}/FIGURES/RESIDUE_CONTACT_MAPS/Residue_DIFF_NORMALIZED_CLUSTER_1D_HeatMap.png".format(
                self.analysis_root), format="png", dpi=400)

            # NORMALIZED CLUSTER (count)
            sg_norm_arr = nBio_sg
            dsm_norm_arr = nBio_dsm
            ndsm_norm_arr = nBio_ndsm
            res_contact_array = np.divide(np.subtract(dsm_norm_arr, ndsm_norm_arr), sg_norm_arr)
            fig = self.plot_res_cpm(res_contact_array, col="Reds", min=-0.25, max=0.25, mid=3)
            plt.savefig("{}/FIGURES/RESIDUE_CONTACT_MAPS/Residue_DIFF_COUNT_HeatMap.png".format(
                self.analysis_root), format="png", dpi=400)


            dsm_norm_arr = np.loadtxt(self._analysis_file("ANALYSIS_DSM_AVE", "BioNum_DSM.csv"), delimiter=",", dtype=float).transpose()
            ndsm_norm_arr = np.loadtxt(self._analysis_file("ANALYSIS_NDSM_AVE", "BioNum_NDSM.csv"), delimiter=",", dtype=float).transpose()
            sg_norm_arr = np.loadtxt(self._analysis_file("ANALYSIS_SG_AVE", "BioNum_sg_X.csv"), delimiter=",", dtype=float).transpose()
            dsm_norm_arr = np.divide(dsm_norm_arr, np.sum(dsm_norm_arr))
            ndsm_norm_arr = np.divide(ndsm_norm_arr, np.sum(ndsm_norm_arr))
            sg_norm_arr = np.divide(sg_norm_arr, np.sum(sg_norm_arr))
            sm_list = [""]
            res_contact_array = np.divide(np.subtract(dsm_norm_arr, ndsm_norm_arr), sg_norm_arr).reshape(-1, 1)
            fig = self.plot_res_sm_cpm(res_contact_array, col="coolwarm", min=0, max=8, mid=0, sm_list=sm_list, figsize=(1.6, 3.2))
            plt.savefig("{}/FIGURES/RESIDUE_CONTACT_MAPS/Residue_DIFF_COUNT_1D_HeatMap.png".format(
                self.analysis_root), format="png", dpi=400)

        # SM RESIDUE MAPS — only relevant when DSM/NDSM data are present
        if not {"DSM", "NDSM"}.issubset(conds):
            return

        # INDIVIDUAL SM MAPS
        df_res_contact = pd.read_csv(os.path.join(self.analysis_root, "RESULTS", "SM_CONTACT_MAPS", "SM_ResMap_Data.csv"))
        df_res_count = pd.read_csv(os.path.join(self.analysis_root, "RESULTS", "SM_CONTACT_MAPS", "SM_ResCount_Data.csv"))
        sm_list = [
            "D-1",
            "D-2",
            "D-3",
            "D-4",
            "D-5",
            "D-6",
            "D-7",
            "D-8",
            "D-9",
            "D-10",
            "N-1",
            "N-2",
            "N-3",
            "N-4",
            "N-5",
            "N-6",
            "N-7",
            "N-8",
            "N-9",
            "N-10",
            "DSM",
            "NDSM",
            "DSM_AVE",
            "NDSM_AVE"
        ]


        # UNNORMALIZED
        sm_res_contact_array = np.array(df_res_contact.iloc[:, 1:])
        file_name = "{}/FIGURES/SM_CONTACT_MAPS/SM_Residue_IND_UNNORMALIZED_HeatMap_OLD.png".format(
            self.analysis_root)
        fig = self.plot_res_sm_cpm(sm_res_contact_array, col="Reds", min=0, max=180, mid=90, sm_list=sm_list, figsize=(8.0, 3.5))
        plt.savefig(file_name, format="png", dpi=400)

        # SYSTEM NORMALIZED
        nprot = np.loadtxt(os.path.join(self.analysis_root, "CM_NORM", "MAPS", "BioNum_SYSTEM.csv"), delimiter=",", dtype=float).transpose().reshape(-1, 1)
        sm_res_contact_array = _safe_divide(np.array(df_res_contact.iloc[:, 1:]), nprot)
        file_name = "{}/FIGURES/SM_CONTACT_MAPS/SM_Residue_IND_NORMALIZED_SYSTEM_HeatMap_OLD.png".format(
            self.analysis_root)
        fig = self.plot_res_sm_cpm(sm_res_contact_array, col="Reds", min=0, max=8, mid=4, sm_list=sm_list, figsize=(8.0, 3.5))
        plt.savefig(file_name, format="png", dpi=400)

        # CLUSTER NORMALIZED
        sm_res_count_array = np.array(df_res_count.iloc[:, 1:])
        sm_res_contact_array = _safe_divide(np.array(df_res_contact.iloc[:, 1:]), sm_res_count_array)
        file_name = "{}/FIGURES/SM_CONTACT_MAPS/SM_Residue_IND_NORMALIZED_CLUSTER_HeatMap_OLD.png".format(
            self.analysis_root)
        fig = self.plot_res_sm_cpm(sm_res_contact_array, col="Reds", min=0, max=20, mid=10, sm_list=sm_list, figsize=(8.0, 3.5))
        plt.savefig(file_name, format="png", dpi=400)

        # DIFFERENCE SM RESIDUE MAPS
        if "DSM" not in df_res_contact.columns or "NDSM" not in df_res_contact.columns:
            print("[INFO] Skipping SM residue DIFF maps: DSM/NDSM aggregate columns absent in SM_ResMap_Data.csv")
            return

        # UNNORMALIZED
        sm_arr = _safe_divide(df_res_contact.loc[:, "DSM"], np.sum(df_res_contact.loc[:, "DSM"])) - _safe_divide(df_res_contact.loc[:, "NDSM"], np.sum(df_res_contact.loc[:, "NDSM"]))

        sm_res_contact_array = np.transpose(np.asarray([sm_arr]))
        sm_list = [""]
        file_name = "{}/FIGURES/SM_CONTACT_MAPS/SM_Residue_DIFF_UNNORMALIZED_HeatMap.png".format(
            self.analysis_root)
        fig = self.plot_res_sm_cpm(sm_res_contact_array, col="coolwarm", min=-0.4, max=-10, mid=0,
                                   sm_list=sm_list, show_xticks=False)
        plt.savefig(file_name, format="png", dpi=400)

        # SYSTEM NORMALIZED
        nprot = np.loadtxt(os.path.join(self.analysis_root, "CM_NORM", "MAPS", "BioNum_SYSTEM.csv"), delimiter=",", dtype=float).transpose()
        sm_arr = _safe_divide(df_res_contact.loc[:, "DSM"], nprot) - _safe_divide(
            df_res_contact.loc[:, "NDSM"], nprot)
        sm_res_contact_array = np.transpose(np.asarray([sm_arr]))
        file_name = "{}/FIGURES/SM_CONTACT_MAPS/SM_Residue_DIFF_NORMALIZED_SYSTEM_HeatMap.png".format(
            self.analysis_root)
        fig = self.plot_res_sm_cpm(sm_res_contact_array, col="Blues_r", min=-1.5, max=-0.5, mid=-1,
                                   sm_list=sm_list, show_xticks=False)
        plt.savefig(file_name, format="png", dpi=400)

        # CLUSTER NORMALIZED
        df_quant = pd.read_csv(os.path.join(self.analysis_root, "RESULTS", "SUMMARY", "Quant_Data.csv")).drop_duplicates()
        if (
            "DSM" not in df_res_contact.columns
            or "NDSM" not in df_res_contact.columns
            or df_quant[df_quant["Small Molecule ID"] == "DSM"].empty
            or df_quant[df_quant["Small Molecule ID"] == "NDSM"].empty
        ):
            print("[INFO] Skipping SM residue difference normalized maps: DSM/NDSM summary data are unavailable for this temperature")
            return
        dsm_conc = df_quant[df_quant["Small Molecule ID"]=="DSM"].loc[:,"$P_{SM}$"].values[0]
        ndsm_conc = df_quant[df_quant["Small Molecule ID"] == "NDSM"].loc[:,"$P_{SM}$"].values[0]

        sm_arr = _safe_divide(
            df_res_contact.loc[:, "DSM"],
            df_res_count.loc[:, "DSM"] * np.sum(df_res_contact.loc[:, "DSM"]),
        ) - _safe_divide(
            df_res_contact.loc[:, "NDSM"],
            df_res_count.loc[:, "NDSM"] * np.sum(df_res_contact.loc[:, "NDSM"]),
        )

        sm_res_contact_array = np.transpose(np.asarray([sm_arr]))

        x_res_list = ["", "", "", "", "", "", ""]
        file_name = "{}/FIGURES/SM_CONTACT_MAPS/SM_Residue_DIFF_NORMALIZED_CLUSTER_HeatMap.png".format(
            self.analysis_root)
        fig = self.plot_res_sm_cpm(sm_res_contact_array, col="Reds", min=-1.4, max=0, mid=0,
                                   sm_list=sm_list, show_xticks=False)
        plt.savefig(file_name, format="png", dpi=400)

    def plot_acid_cpm(self, acid_contact_array, col, min, max, mid, figsize=ACID_CONTACT_MAP_FIGSIZE):
        """Render a square 24x24 amino-acid / nucleotide-type contact heatmap.

        Rows/cols are permuted from LAMMPS atom-type order into the chemistry-
        grouped display order (``ACID_DISPLAY_PERM``) with a separator line
        between the 20 amino acids and 4 nucleotides. *mid* is accepted for
        signature parity but unused. Returns the figure.
        """
        plt.close('all')
        acid_contact_array = _sanitize_heatmap_array(acid_contact_array)
        acid_list = [
            'ARG',
            'HIS',
            'LYS',
            'ASP',
            'GLU',
            'SER',
            'THR',
            'ASN',
            'GLN',
            'CYS',
            'GLY',
            'PRO',
            'ALA',
            'VAL',
            'ILE',
            'LEU',
            'MET',
            'PHE',
            'TYR',
            'TRP',
            'A',
            'U',
            'C',
            'G'
        ]

        # Reorder rows/cols from LAMMPS atom-type order (0-indexed) to the
        # chemistry-grouped display order in acid_list via an explicit
        # permutation. The previous chained-swap implementation had a
        # subtle bug that left ILE/LEU labels visually swapped (display
        # 14 was LEU instead of ILE, display 15 was ILE instead of LEU).
        # LAMMPS protein types (1-20, 0-indexed 0-19):
        #   M=1,G=2,K=3,T=4,R=5,A=6,D=7,E=8,Y=9,V=10,
        #   I=11,Q=12,W=13,F=14,S=15,H=16,N=17,P=18,C=19,L=20
        # LAMMPS RNA types (21-24, 0-indexed 20-23): A=21,C=22,G=23,U=24
        # acid_list display order maps to LAMMPS 0-indexed:
        ACID_DISPLAY_PERM = [
            4, 15, 2, 6, 7, 14, 3, 16, 11, 18, 1, 17,
            5, 9, 10, 19, 0, 13, 8, 12, 20, 23, 21, 22,
        ]
        acid_contact_array = acid_contact_array[np.ix_(ACID_DISPLAY_PERM, ACID_DISPLAY_PERM)]


        sns.set_theme(style="ticks")
        sns.set_style('white')
        plt.rc('axes', titlesize=10)
        plt.rc('axes', labelsize=10)
        plt.rc('xtick', labelsize=8)
        plt.rc('ytick', labelsize=8)
        plt.rc('legend', fontsize=8)
        plt.rc('font', size=10)
        plt.rc('axes', linewidth=2)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes(_CONTACT_AX_RECT)
        sns.heatmap(acid_contact_array, xticklabels=acid_list, yticklabels=acid_list, square=True, cmap=col, cbar=False, ax=ax, vmin=min, vmax=max)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)
        _apply_contact_heatmap_ticks(ax)
        # Acid maps carry 24 densely-packed labels in a 2.40-in axis (~7.2 pt per
        # slot), so 8-pt labels overrun their slot and overlap. 6 pt fits cleanly
        # on both axes (matches the SM acid maps).
        ax.tick_params(axis='both', which='major', labelsize=6)
        # Separator between amino acids (cols/rows 0–19) and nucleic acids (20–23)
        ax.axvline(x=20, color='black', linewidth=1.5, zorder=3)
        ax.axhline(y=20, color='black', linewidth=1.5, zorder=3)
        _setup_cbar_fixed(fig, ax, _CONTACT_AX_RECT)

        return fig

    def plot_acid_sm_cpm(self, acid_contact_array, col, min, max, mid, sm_list,
                         figsize=None, show_xticks=True):
        """Render a 24 x len(sm_list) acid-type SM/1D contact heatmap.

        Rows are the 24 acid types in chemistry-grouped display order, columns
        labelled by *sm_list*; layout is computed by ``_sm_compute_layout`` so
        *figsize* and *mid* are accepted only for signature parity and are
        unused. *show_xticks* hides the x labels when False. Returns the figure.
        """
        plt.close('all')
        acid_contact_array = _sanitize_heatmap_array(acid_contact_array)
        acid_list = [
            'ARG',
            'HIS',
            'LYS',
            'ASP',
            'GLU',
            'SER',
            'THR',
            'ASN',
            'GLN',
            'CYS',
            'GLY',
            'PRO',
            'ALA',
            'VAL',
            'ILE',
            'LEU',
            'MET',
            'PHE',
            'TYR',
            'TRP',
            'A',
            'U',
            'C',
            'G'
        ]

        # Row-only permutation for the 1D / SM variant; same fix as the
        # 2D plot_acid_cpm above.
        ACID_DISPLAY_PERM = [
            4, 15, 2, 6, 7, 14, 3, 16, 11, 18, 1, 17,
            5, 9, 10, 19, 0, 13, 8, 12, 20, 23, 21, 22,
        ]
        acid_contact_array = acid_contact_array[ACID_DISPLAY_PERM, :]

        sns.set_theme(style="ticks")
        sns.set_style('white')
        plt.rc('axes', titlesize=10)
        plt.rc('axes', labelsize=10)
        plt.rc('xtick', labelsize=6)
        plt.rc('ytick', labelsize=6)
        plt.rc('legend', fontsize=8)
        plt.rc('font', size=10)
        plt.rc('axes', linewidth=2)

        n_rows, n_cols = acid_contact_array.shape
        _figsize, ax_rect = _sm_compute_layout(n_rows, n_cols)
        if n_cols == 1:
            show_xticks = False
        fig = plt.figure(figsize=_figsize)
        ax = fig.add_axes(ax_rect)
        _vmin, _vmax = (min, max) if (min is not None and max is not None and min < max) else (None, None)
        sns.heatmap(acid_contact_array, xticklabels=sm_list, yticklabels=acid_list,
                    cmap=col, square=False, cbar=False, ax=ax, vmin=_vmin, vmax=_vmax)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)
        _apply_contact_heatmap_ticks(ax)
        ax.tick_params(axis='x', rotation=90)
        ax.tick_params(axis='y', rotation=0)
        if not show_xticks:
            ax.tick_params(axis='x', which='both', bottom=False, top=False,
                           labelbottom=False, labeltop=False)
            ax.set_xticklabels([])
        _setup_cbar_fixed(fig, ax, ax_rect,
                          cbar_pad_in=_SM_CBAR_PAD_IN, cbar_w_in=_SM_CBAR_W_IN)
        return fig

    def gen_acid_cpms(self):
        """Generate the legacy 24x24 amino-acid / nucleotide contact maps.

        Mirrors ``gen_residue_cpms`` at acid-type resolution: per-condition
        individual maps (UNNORMALIZED / per-SYSTEM / per-CLUSTER), 1D sums and
        -ln companions, DSM-minus-NDSM difference maps, and the per-SM SM-acid
        maps when DSM/NDSM data are present. Bonded covalent neighbour counts
        (``BondNum_*``) are subtracted to match the legacy convention. Outputs
        go to FIGURES/ACID_CONTACT_MAPS/ and FIGURES/SM_CONTACT_MAPS/.
        """
        acid_list = [
            'ARG',
            'HIS',
            'LYS',
            'ASP',
            'GLU',
            'SER',
            'THR',
            'ASN',
            'GLN',
            'CYS',
            'GLY',
            'PRO',
            'ALA',
            'VAL',
            'ILE',
            'LEU',
            'MET',
            'PHE',
            'TYR',
            'TRP',
            'A',
            'U',
            'C',
            'G'
        ]

        conds = self._available_conditions()

        nAcid_by, nBond_by, acid_arr_by = {}, {}, {}
        if "SG" in conds:
            nAcid_by["SG"] = np.loadtxt(self._analysis_file("ANALYSIS_SG_AVE", "AcidPolNum_sg_X.csv"), delimiter=",", dtype=float)
            nBond_by["SG"] = np.loadtxt(self._analysis_file("ANALYSIS_SG_AVE", "BondNum_sg_X.csv"), delimiter=",", dtype=float)
            acid_arr_by["SG"] = np.array(pd.read_csv(self._analysis_file("ANALYSIS_SG_AVE", "Acid_Contacts_Mean_sg_X.csv"), header=None))
        if "DSM" in conds:
            nAcid_by["DSM"] = np.loadtxt(self._analysis_file("ANALYSIS_DSM_AVE", "AcidPolNum_DSM.csv"), delimiter=",", dtype=float)
            nBond_by["DSM"] = np.loadtxt(self._analysis_file("ANALYSIS_DSM_AVE", "BondNum_DSM.csv"), delimiter=",", dtype=float)
            acid_arr_by["DSM"] = np.array(pd.read_csv(self._analysis_file("ANALYSIS_DSM_AVE", "Acid_Contacts_Mean_DSM.csv"), header=None))
        if "NDSM" in conds:
            nAcid_by["NDSM"] = np.loadtxt(self._analysis_file("ANALYSIS_NDSM_AVE", "AcidPolNum_NDSM.csv"), delimiter=",", dtype=float)
            nBond_by["NDSM"] = np.loadtxt(self._analysis_file("ANALYSIS_NDSM_AVE", "BondNum_NDSM.csv"), delimiter=",", dtype=float)
            acid_arr_by["NDSM"] = np.array(pd.read_csv(self._analysis_file("ANALYSIS_NDSM_AVE", "Acid_Contacts_Mean_NDSM.csv"), header=None))
        nAcid = np.loadtxt(os.path.join(self.analysis_root, "CM_NORM", "MAPS", "AcidPolNum_SYSTEM.csv"), delimiter=",", dtype=float)
        nBond = np.loadtxt(os.path.join(self.analysis_root, "CM_NORM", "MAPS", "BondNum_SYSTEM.csv"), delimiter=",", dtype=float)

        sg_acid_arr  = acid_arr_by.get("SG")
        dsm_acid_arr = acid_arr_by.get("DSM")
        ndsm_acid_arr = acid_arr_by.get("NDSM")
        nAcid_sg, nAcid_dsm, nAcid_ndsm = nAcid_by.get("SG"), nAcid_by.get("DSM"), nAcid_by.get("NDSM")
        nBond_sg, nBond_dsm, nBond_ndsm = nBond_by.get("SG"), nBond_by.get("DSM"), nBond_by.get("NDSM")

        # INDIVIDUAL MAPS
        # UNNORMALIZED
        if "SG" in conds:
            acid_contact_array = np.divide(sg_acid_arr-nBond_sg, (np.sum(sg_acid_arr-nBond_sg)))
            self._save_acid_cpm_lin_log(acid_contact_array,
                "{}/FIGURES/ACID_CONTACT_MAPS/Acid_SG_UNNORMALIZED_HeatMap.png".format(self.analysis_root),
                col="Reds", min=0, max=0.012, mid=0.006)

        if "DSM" in conds:
            acid_contact_array = np.divide(dsm_acid_arr-nBond, (np.sum(dsm_acid_arr)))
            self._save_acid_cpm_lin_log(acid_contact_array,
                "{}/FIGURES/ACID_CONTACT_MAPS/Acid_DSM_UNNORMALIZED_HeatMap.png".format(self.analysis_root),
                col="Reds", min=0, max=0.012, mid=0.006)

        if "NDSM" in conds:
            acid_contact_array = np.divide(ndsm_acid_arr-nBond, (np.sum(ndsm_acid_arr)))
            self._save_acid_cpm_lin_log(acid_contact_array,
                "{}/FIGURES/ACID_CONTACT_MAPS/Acid_NDSM_UNNORMALIZED_HeatMap.png".format(self.analysis_root),
                col="Reds", min=0, max=0.012, mid=0.006)

        # NORMALIZED SYSTEM
        if "SG" in conds:
            acid_contact_array = np.divide(sg_acid_arr-nBond, (np.sum(sg_acid_arr) * nAcid))
            self._save_acid_cpm_lin_log(acid_contact_array,
                "{}/FIGURES/ACID_CONTACT_MAPS/Acid_SG_NORMALIZED_SYSTEM_HeatMap.png".format(self.analysis_root),
                col="Reds", min=0, max=0.6*10**-9, mid=0.3*10**-9)

        if "DSM" in conds:
            acid_contact_array = np.divide(dsm_acid_arr-nBond, (np.sum(dsm_acid_arr) * nAcid))
            self._save_acid_cpm_lin_log(acid_contact_array,
                "{}/FIGURES/ACID_CONTACT_MAPS/Acid_DSM_NORMALIZED_SYSTEM_HeatMap.png".format(self.analysis_root),
                col="Reds", min=0, max=0.6*10**-9, mid=0.3*10**-9)

        if "NDSM" in conds:
            acid_contact_array = np.divide(ndsm_acid_arr-nBond, (np.sum(ndsm_acid_arr) * nAcid))
            self._save_acid_cpm_lin_log(acid_contact_array,
                "{}/FIGURES/ACID_CONTACT_MAPS/Acid_NDSM_NORMALIZED_SYSTEM_HeatMap.png".format(self.analysis_root),
                col="Reds", min=0, max=0.6*10**-9, mid=0.3*10**-9)

        # NORMALIZED CLUSTER
        if "SG" in conds:
            acid_contact_array = np.divide(sg_acid_arr-nBond_sg, (np.sum(sg_acid_arr) * nAcid_sg))
            self._save_acid_cpm_lin_log(acid_contact_array,
                "{}/FIGURES/ACID_CONTACT_MAPS/Acid_SG_NORMALIZED_CLUSTER_HeatMap.png".format(self.analysis_root),
                col="Reds", min=0, max=1.5*10**-9, mid=0.75*10**-9)

        if "DSM" in conds:
            acid_contact_array = np.divide(dsm_acid_arr-nBond_dsm, (np.sum(dsm_acid_arr) * nAcid_dsm))
            self._save_acid_cpm_lin_log(acid_contact_array,
                "{}/FIGURES/ACID_CONTACT_MAPS/Acid_DSM_NORMALIZED_CLUSTER_HeatMap.png".format(self.analysis_root),
                col="Reds", min=0, max=1.5*10**-9, mid=0.75*10**-9)

        if "NDSM" in conds:
            acid_contact_array = np.divide(ndsm_acid_arr-nBond_ndsm, (np.sum(ndsm_acid_arr) * nAcid_ndsm))
            self._save_acid_cpm_lin_log(acid_contact_array,
                "{}/FIGURES/ACID_CONTACT_MAPS/Acid_NDSM_NORMALIZED_CLUSTER_HeatMap.png".format(self.analysis_root),
                col="Reds", min=0, max=1.5*10**-9, mid=0.75*10**-9)

        # 1D INDIVIDUAL MAPS
        # 1D UNNORMALIZED
        if "SG" in conds:
            acid_1d = np.divide(sg_acid_arr - nBond_sg, np.sum(sg_acid_arr - nBond_sg)).sum(axis=1).reshape(-1, 1)
            fig = self.plot_acid_sm_cpm(acid_1d, col="Reds", min=0, max=1, mid=0, sm_list=[""], figsize=(1.6, 3.2))
            plt.savefig("{}/FIGURES/ACID_CONTACT_MAPS/Acid_SG_UNNORMALIZED_1D_HeatMap.png".format(
                self.analysis_root), format="png", dpi=400)

        if "DSM" in conds:
            acid_1d = np.divide(dsm_acid_arr - nBond, np.sum(dsm_acid_arr)).sum(axis=1).reshape(-1, 1)
            fig = self.plot_acid_sm_cpm(acid_1d, col="Reds", min=0, max=1, mid=0, sm_list=[""], figsize=(1.6, 3.2))
            plt.savefig("{}/FIGURES/ACID_CONTACT_MAPS/Acid_DSM_UNNORMALIZED_1D_HeatMap.png".format(
                self.analysis_root), format="png", dpi=400)

        if "NDSM" in conds:
            acid_1d = np.divide(ndsm_acid_arr - nBond, np.sum(ndsm_acid_arr)).sum(axis=1).reshape(-1, 1)
            fig = self.plot_acid_sm_cpm(acid_1d, col="Reds", min=0, max=1, mid=0, sm_list=[""], figsize=(1.6, 3.2))
            plt.savefig("{}/FIGURES/ACID_CONTACT_MAPS/Acid_NDSM_UNNORMALIZED_1D_HeatMap.png".format(
                self.analysis_root), format="png", dpi=400)

        # 1D NORMALIZED SYSTEM
        if "SG" in conds:
            acid_1d = np.divide(sg_acid_arr - nBond, np.sum(sg_acid_arr) * nAcid).sum(axis=1).reshape(-1, 1)
            fig = self.plot_acid_sm_cpm(acid_1d, col="Reds", min=0, max=1, mid=0, sm_list=[""], figsize=(1.6, 3.2))
            plt.savefig("{}/FIGURES/ACID_CONTACT_MAPS/Acid_SG_NORMALIZED_SYSTEM_1D_HeatMap.png".format(
                self.analysis_root), format="png", dpi=400)

        if "DSM" in conds:
            acid_1d = np.divide(dsm_acid_arr - nBond, np.sum(dsm_acid_arr) * nAcid).sum(axis=1).reshape(-1, 1)
            fig = self.plot_acid_sm_cpm(acid_1d, col="Reds", min=0, max=1, mid=0, sm_list=[""], figsize=(1.6, 3.2))
            plt.savefig("{}/FIGURES/ACID_CONTACT_MAPS/Acid_DSM_NORMALIZED_SYSTEM_1D_HeatMap.png".format(
                self.analysis_root), format="png", dpi=400)

        if "NDSM" in conds:
            acid_1d = np.divide(ndsm_acid_arr - nBond, np.sum(ndsm_acid_arr) * nAcid).sum(axis=1).reshape(-1, 1)
            fig = self.plot_acid_sm_cpm(acid_1d, col="Reds", min=0, max=1, mid=0, sm_list=[""], figsize=(1.6, 3.2))
            plt.savefig("{}/FIGURES/ACID_CONTACT_MAPS/Acid_NDSM_NORMALIZED_SYSTEM_1D_HeatMap.png".format(
                self.analysis_root), format="png", dpi=400)

        # 1D NORMALIZED CLUSTER
        if "SG" in conds:
            acid_1d = np.divide(sg_acid_arr - nBond_sg, np.sum(sg_acid_arr) * nAcid_sg).sum(axis=1).reshape(-1, 1)
            fig = self.plot_acid_sm_cpm(acid_1d, col="Reds", min=0, max=1, mid=0, sm_list=[""], figsize=(1.6, 3.2))
            plt.savefig("{}/FIGURES/ACID_CONTACT_MAPS/Acid_SG_NORMALIZED_CLUSTER_1D_HeatMap.png".format(
                self.analysis_root), format="png", dpi=400)

        if "DSM" in conds:
            acid_1d = np.divide(dsm_acid_arr - nBond_dsm, np.sum(dsm_acid_arr) * nAcid_dsm).sum(axis=1).reshape(-1, 1)
            fig = self.plot_acid_sm_cpm(acid_1d, col="Reds", min=0, max=1, mid=0, sm_list=[""], figsize=(1.6, 3.2))
            plt.savefig("{}/FIGURES/ACID_CONTACT_MAPS/Acid_DSM_NORMALIZED_CLUSTER_1D_HeatMap.png".format(
                self.analysis_root), format="png", dpi=400)

        if "NDSM" in conds:
            acid_1d = np.divide(ndsm_acid_arr - nBond_ndsm, np.sum(ndsm_acid_arr) * nAcid_ndsm).sum(axis=1).reshape(-1, 1)
            fig = self.plot_acid_sm_cpm(acid_1d, col="Reds", min=0, max=1, mid=0, sm_list=[""], figsize=(1.6, 3.2))
            plt.savefig("{}/FIGURES/ACID_CONTACT_MAPS/Acid_NDSM_NORMALIZED_CLUSTER_1D_HeatMap.png".format(
                self.analysis_root), format="png", dpi=400)

        # DIFFERENCE MAPS — require SG+DSM+NDSM
        if {"SG", "DSM", "NDSM"}.issubset(conds):
            # UNNORMALIZED
            dsm_arr = dsm_acid_arr - nBond_dsm
            ndsm_arr = ndsm_acid_arr - nBond_ndsm
            sg_arr = sg_acid_arr - nBond_sg
            acid_contact_array = np.divide(np.subtract(dsm_arr, ndsm_arr), sg_arr)
            fig = self.plot_acid_cpm(acid_contact_array, col="coolwarm", min=-0.07, max=0.07, mid=0)
            plt.savefig("{}/FIGURES/ACID_CONTACT_MAPS/Acid_DIFF_UNNORMALIZED_HeatMap.png".format(
                self.analysis_root), format="png", dpi=400)
            self._save_acid_diff_log(dsm_arr, ndsm_arr,
                "{}/FIGURES/ACID_CONTACT_MAPS/Acid_DIFF_UNNORMALIZED_HeatMap_LOG.png".format(self.analysis_root))

            # NORMALIZED SYSTEM
            sg_norm_arr = np.divide(sg_acid_arr, (np.sum(sg_acid_arr)))
            dsm_norm_arr = np.divide(dsm_acid_arr, (np.sum(dsm_acid_arr)))
            ndsm_norm_arr = np.divide(ndsm_acid_arr, (np.sum(ndsm_acid_arr)))
            acid_contact_array = np.divide(np.subtract(dsm_norm_arr, ndsm_norm_arr), sg_norm_arr)
            fig = self.plot_acid_cpm(acid_contact_array, col="coolwarm", min=-0.07, max=0.07, mid=0)
            plt.savefig("{}/FIGURES/ACID_CONTACT_MAPS/Acid_DIFF_NORMALIZED_SYSTEM_HeatMap.png".format(
                self.analysis_root), format="png", dpi=400)
            self._save_acid_diff_log(dsm_norm_arr, ndsm_norm_arr,
                "{}/FIGURES/ACID_CONTACT_MAPS/Acid_DIFF_NORMALIZED_SYSTEM_HeatMap_LOG.png".format(self.analysis_root))

            sg_norm_arr = np.divide(sg_acid_arr - nBond_sg, (np.sum(sg_acid_arr - nBond_sg)))
            dsm_norm_arr = np.divide(dsm_acid_arr - nBond_dsm, (np.sum(dsm_acid_arr - nBond_dsm)))
            ndsm_norm_arr = np.divide(ndsm_acid_arr - nBond_ndsm, (np.sum(ndsm_acid_arr - nBond_ndsm)))
            acid_contact_array = np.divide(np.subtract(dsm_norm_arr, ndsm_norm_arr), sg_norm_arr)
            fig = self.plot_acid_cpm(acid_contact_array, col="coolwarm", min=0.3, max=1, mid=0.8)
            plt.savefig("{}/FIGURES/ACID_CONTACT_MAPS/Acid_DIFF_NORMALIZED_SYSTEM_NEW_HeatMap.png".format(
                self.analysis_root), format="png", dpi=400)
            self._save_acid_diff_log(dsm_norm_arr, ndsm_norm_arr,
                "{}/FIGURES/ACID_CONTACT_MAPS/Acid_DIFF_NORMALIZED_SYSTEM_NEW_HeatMap_LOG.png".format(self.analysis_root))

            # NORMALIZED CLUSTER
            sg_norm_arr = np.divide(sg_acid_arr-nBond_sg, (np.sum(sg_acid_arr-nBond_sg) * nAcid_sg))
            dsm_norm_arr = np.divide(dsm_acid_arr-nBond_dsm, (np.sum(dsm_acid_arr-nBond_dsm) * nAcid_dsm))
            ndsm_norm_arr = np.divide(ndsm_acid_arr-nBond_ndsm, (np.sum(ndsm_acid_arr-nBond_ndsm) * nAcid_ndsm))
            acid_contact_array = np.divide(np.subtract(dsm_norm_arr, ndsm_norm_arr), sg_norm_arr)
            fig = self.plot_acid_cpm(acid_contact_array, col="Reds", min=0.3, max=1, mid=0.8)
            plt.savefig("{}/FIGURES/ACID_CONTACT_MAPS/Acid_DIFF_NORMALIZED_CLUSTER_HeatMap.png".format(
                self.analysis_root), format="png", dpi=400)
            self._save_acid_diff_log(dsm_norm_arr, ndsm_norm_arr,
                "{}/FIGURES/ACID_CONTACT_MAPS/Acid_DIFF_NORMALIZED_CLUSTER_HeatMap_LOG.png".format(self.analysis_root))

            # 1D DIFF UNNORMALIZED
            dsm_arr = dsm_acid_arr - nBond_dsm
            ndsm_arr = ndsm_acid_arr - nBond_ndsm
            sg_arr = sg_acid_arr - nBond_sg
            acid_1d = np.divide(np.subtract(dsm_arr, ndsm_arr), sg_arr).sum(axis=1).reshape(-1, 1)
            fig = self.plot_acid_sm_cpm(acid_1d, col="coolwarm", min=0, max=1, mid=0, sm_list=[""], figsize=(1.6, 3.2))
            plt.savefig("{}/FIGURES/ACID_CONTACT_MAPS/Acid_DIFF_UNNORMALIZED_1D_HeatMap.png".format(
                self.analysis_root), format="png", dpi=400)

            # 1D DIFF NORMALIZED SYSTEM
            sg_norm_arr = np.divide(sg_acid_arr, np.sum(sg_acid_arr))
            dsm_norm_arr = np.divide(dsm_acid_arr, np.sum(dsm_acid_arr))
            ndsm_norm_arr = np.divide(ndsm_acid_arr, np.sum(ndsm_acid_arr))
            acid_1d = np.divide(np.subtract(dsm_norm_arr, ndsm_norm_arr), sg_norm_arr).sum(axis=1).reshape(-1, 1)
            fig = self.plot_acid_sm_cpm(acid_1d, col="coolwarm", min=0, max=1, mid=0, sm_list=[""], figsize=(1.6, 3.2))
            plt.savefig("{}/FIGURES/ACID_CONTACT_MAPS/Acid_DIFF_NORMALIZED_SYSTEM_1D_HeatMap.png".format(
                self.analysis_root), format="png", dpi=400)

            # 1D DIFF NORMALIZED SYSTEM NEW
            sg_norm_arr = np.divide(sg_acid_arr - nBond_sg, np.sum(sg_acid_arr - nBond_sg))
            dsm_norm_arr = np.divide(dsm_acid_arr - nBond_dsm, np.sum(dsm_acid_arr - nBond_dsm))
            ndsm_norm_arr = np.divide(ndsm_acid_arr - nBond_ndsm, np.sum(ndsm_acid_arr - nBond_ndsm))
            acid_1d = np.divide(np.subtract(dsm_norm_arr, ndsm_norm_arr), sg_norm_arr).sum(axis=1).reshape(-1, 1)
            fig = self.plot_acid_sm_cpm(acid_1d, col="coolwarm", min=0, max=1, mid=0, sm_list=[""], figsize=(1.6, 3.2))
            plt.savefig("{}/FIGURES/ACID_CONTACT_MAPS/Acid_DIFF_NORMALIZED_SYSTEM_NEW_1D_HeatMap.png".format(
                self.analysis_root), format="png", dpi=400)

            # 1D DIFF NORMALIZED CLUSTER
            sg_norm_arr = np.divide(sg_acid_arr - nBond_sg, np.sum(sg_acid_arr - nBond_sg) * nAcid_sg)
            dsm_norm_arr = np.divide(dsm_acid_arr - nBond_dsm, np.sum(dsm_acid_arr - nBond_dsm) * nAcid_dsm)
            ndsm_norm_arr = np.divide(ndsm_acid_arr - nBond_ndsm, np.sum(ndsm_acid_arr - nBond_ndsm) * nAcid_ndsm)
            acid_1d = np.divide(np.subtract(dsm_norm_arr, ndsm_norm_arr), sg_norm_arr).sum(axis=1).reshape(-1, 1)
            fig = self.plot_acid_sm_cpm(acid_1d, col="Reds", min=0, max=1, mid=0, sm_list=[""], figsize=(1.6, 3.2))
            plt.savefig("{}/FIGURES/ACID_CONTACT_MAPS/Acid_DIFF_NORMALIZED_CLUSTER_1D_HeatMap.png".format(
                self.analysis_root), format="png", dpi=400)

            # 1D ACID DIFF COUNT
            dsm_acid_num = np.loadtxt(self._analysis_file("ANALYSIS_DSM_AVE", "AcidNum_DSM.csv"), delimiter=",", dtype=float).transpose()
            ndsm_acid_num = np.loadtxt(self._analysis_file("ANALYSIS_NDSM_AVE", "AcidNum_NDSM.csv"), delimiter=",", dtype=float).transpose()
            sg_acid_num = np.loadtxt(self._analysis_file("ANALYSIS_SG_AVE", "AcidNum_sg_X.csv"), delimiter=",", dtype=float).transpose()
            dsm_acid_num = np.divide(dsm_acid_num, np.sum(dsm_acid_num))
            ndsm_acid_num = np.divide(ndsm_acid_num, np.sum(ndsm_acid_num))
            sg_acid_num = np.divide(sg_acid_num, np.sum(sg_acid_num))
            acid_1d = np.divide(np.subtract(dsm_acid_num, ndsm_acid_num), sg_acid_num).reshape(-1, 1)
            fig = self.plot_acid_sm_cpm(acid_1d, col="coolwarm", min=0, max=8, mid=0, sm_list=[""], figsize=(1.6, 3.2))
            plt.savefig("{}/FIGURES/ACID_CONTACT_MAPS/Acid_DIFF_COUNT_1D_HeatMap.png".format(
                self.analysis_root), format="png", dpi=400)

        # SM Acid Contact Array — only relevant if DSM/NDSM data are present
        if not {"DSM", "NDSM"}.issubset(conds):
            return

        df_acid_contact = pd.read_csv(os.path.join(self.analysis_root, "RESULTS", "SM_CONTACT_MAPS", "SM_AcidMap_Data.csv"))
        df_acid_count = pd.read_csv(os.path.join(self.analysis_root, "RESULTS", "SM_CONTACT_MAPS", "SM_AcidCount_Data.csv"))
        sm_list = [
            "D-1",
            "D-2",
            "D-3",
            "D-4",
            "D-5",
            "D-6",
            "D-7",
            "D-8",
            "D-9",
            "D-10",
            "N-1",
            "N-2",
            "N-3",
            "N-4",
            "N-5",
            "N-6",
            "N-7",
            "N-8",
            "N-9",
            "N-10",
            "DSM",
            "NDSM",
            "DSM_AVE",
            "NDSM_AVE"
        ]

        # SM ACID MAPS
        # INDIVIDUAL SM MAPS

        # UNNORMALIZED
        sm_acid_contact_array = _safe_divide(np.array(df_acid_contact.iloc[:, 1:]), np.sum(np.array(df_acid_contact.iloc[:, 1:])))
        file_name = "{}/FIGURES/SM_CONTACT_MAPS/SM_Acid_IND_UNNORMALIZED_HeatMap.png".format(
            self.analysis_root)
        fig = self.plot_acid_sm_cpm(sm_acid_contact_array, col="coolwarm", min=0, max=50, mid=25, sm_list=sm_list, figsize=(8.0, 3.5))
        plt.savefig(file_name, format="png", dpi=400)

        # SYSTEM NORMALIZED
        nprot = np.loadtxt(os.path.join(self.analysis_root, "CM_NORM", "MAPS", "AcidNum_SYSTEM.csv"), delimiter=",", dtype=float).transpose().reshape(-1, 1)
        sm_acid_contact_array = _safe_divide(np.array(df_acid_contact.iloc[:, 1:]), nprot)
        file_name = "{}/FIGURES/SM_CONTACT_MAPS/SM_Acid_IND_NORMALIZED_SYSTEM_HeatMap.png".format(
            self.analysis_root)
        fig = self.plot_acid_sm_cpm(sm_acid_contact_array, col="coolwarm", min=0, max=0.014, mid=0, sm_list=sm_list, figsize=(8.0, 3.5))
        plt.savefig(file_name, format="png", dpi=400)

        # CLUSTER NORMALIZED
        sm_acid_count_array = np.array(df_acid_count.iloc[:, 1:])
        sm_acid_contact_array = _safe_divide(np.array(df_acid_contact.iloc[:, 1:]), sm_acid_count_array)
        file_name = "{}/FIGURES/SM_CONTACT_MAPS/SM_Acid_IND_NORMALIZED_CLUSTER_HeatMap.png".format(
            self.analysis_root)
        fig = self.plot_acid_sm_cpm(sm_acid_contact_array, col="Reds", min=0, max=0.028, mid=0.014, sm_list=sm_list, figsize=(8.0, 3.5))
        plt.savefig(file_name, format="png", dpi=400)

        # DIFFERENCE SM ACID MAPS
        if "DSM" not in df_acid_contact.columns or "NDSM" not in df_acid_contact.columns:
            print("[INFO] Skipping SM acid DIFF maps: DSM/NDSM aggregate columns absent in SM_AcidMap_Data.csv")
            return
        # UNNORMALIZED
        sm_arr = _safe_divide(df_acid_contact.loc[:, "DSM"], np.sum(df_acid_contact.loc[:, "DSM"])) - _safe_divide(df_acid_contact.loc[:, "NDSM"], np.sum(df_acid_contact.loc[:, "NDSM"]))

        sm_acid_contact_array = np.transpose(np.asarray([sm_arr]))
        sm_list = [""]
        file_name = "{}/FIGURES/SM_CONTACT_MAPS/SM_Acid_DIFF_UNNORMALIZED_HeatMap.png".format(
            self.analysis_root)
        fig = self.plot_acid_sm_cpm(sm_acid_contact_array, col="coolwarm", min=-10, max=0, mid=0,
                                    sm_list=sm_list, show_xticks=False)
        plt.savefig(file_name, format="png", dpi=400)

        # SYSTEM NORMALIZED
        nprot = np.loadtxt(os.path.join(self.analysis_root, "CM_NORM", "MAPS", "AcidNum_SYSTEM.csv"), delimiter=",", dtype=float).transpose()
        sm_arr = _safe_divide(df_acid_contact.loc[:, "DSM"], nprot) - _safe_divide(
            df_acid_contact.loc[:, "NDSM"], nprot)
        sm_acid_contact_array = np.transpose(np.asarray([sm_arr]))
        file_name = "{}/FIGURES/SM_CONTACT_MAPS/SM_Acid_DIFF_NORMALIZED_SYSTEM_HeatMap.png".format(
            self.analysis_root)
        fig = self.plot_acid_sm_cpm(sm_acid_contact_array, col="Blues_r", min=-0.0026, max=-0.0012, mid=-0.0019,
                                    sm_list=sm_list, show_xticks=False)
        plt.savefig(file_name, format="png", dpi=400)

        # CLUSTER NORMALIZED
        df_quant = pd.read_csv(os.path.join(self.analysis_root, "RESULTS", "SUMMARY", "Quant_Data.csv")).drop_duplicates()
        if (
            "DSM" not in df_acid_contact.columns
            or "NDSM" not in df_acid_contact.columns
            or df_quant[df_quant["Small Molecule ID"] == "DSM"].empty
            or df_quant[df_quant["Small Molecule ID"] == "NDSM"].empty
        ):
            print("[INFO] Skipping SM acid difference normalized maps: DSM/NDSM summary data are unavailable for this temperature")
            return
        dsm_conc = df_quant[df_quant["Small Molecule ID"] == "DSM"].loc[:, "$P_{SM}$"].values[0]
        ndsm_conc = df_quant[df_quant["Small Molecule ID"] == "NDSM"].loc[:, "$P_{SM}$"].values[0]
        sm_arr = _safe_divide(
            df_acid_contact.loc[:, "DSM"],
            df_acid_count.loc[:, "DSM"] * np.sum(df_acid_contact.loc[:, "DSM"]),
        ) - _safe_divide(
            df_acid_contact.loc[:, "NDSM"],
            df_acid_count.loc[:, "NDSM"] * np.sum(df_acid_contact.loc[:, "NDSM"]),
        )
        sm_acid_contact_array = np.transpose(np.asarray([sm_arr]))
        x_res_list = ["", "", "", "", "", "", ""]
        file_name = "{}/FIGURES/SM_CONTACT_MAPS/SM_Acid_DIFF_NORMALIZED_CLUSTER_HeatMap.png".format(
            self.analysis_root)
        fig = self.plot_acid_sm_cpm(sm_acid_contact_array, col="Reds", min=-0.001, max=0.0, mid=0.0001,
                                    sm_list=sm_list, show_xticks=False)
        plt.savefig(file_name, format="png", dpi=400)


    # ======================================================================
    # New publication contact maps: P_contact (per-pair contact probability)
    # and K_chain (contacts per chain pair). Saved alongside the legacy
    # UNNORMALIZED/NORMALIZED_SYSTEM/NORMALIZED_CLUSTER maps.
    #
    # Math:
    #   P_AB  = <C_AB>_t / N_residue_pairs(A,B)   -> probability in [0,1]
    #   K_AB  = <C_AB>_t / N_chain_pairs(A,B)     -> contacts per chain pair
    #   N_residue_pairs / N_chain_pairs derived from BioPolNum_*.csv via
    #     contact_normalization.py (no RCC re-run required).
    #
    # File names (added — existing maps not modified):
    #   Residue_{SG,DSM,NDSM}_P_contact_HeatMap.png        (P linear)
    #   Residue_{SG,DSM,NDSM}_P_contact_LOG_HeatMap.png    (-ln(P), nats)
    #   Residue_{SG,DSM,NDSM}_K_chain_HeatMap.png          (K linear)
    #   Residue_{SG,DSM,NDSM}_K_chain_LOG_HeatMap.png      (log10(K), display)
    #   Residue_DIFF_P_contact_HeatMap.png                 (DeltaP = P_DSM - P_NDSM)
    #   Residue_DIFF_P_contact_LOGRATIO_HeatMap.png        (ln(P_DSM / P_NDSM))
    #   Residue_DIFF_K_chain_HeatMap.png                   (DeltaK)
    #   Residue_DIFF_K_chain_LOGRATIO_HeatMap.png          (ln(K_DSM / K_NDSM))
    #   Plus _1D_ variants for each, computed as row-mean.
    # ======================================================================

    def gen_residue_cpms_pcontact(self):
        """Generate biopolymer P_contact and K_chain residue maps (see banner above).

        For each condition saves per-pair contact probability P (and -ln P),
        contacts-per-chain-pair K (and log10 K), their 1D row means, and the
        DSM-vs-NDSM linear and log-ratio difference maps, into
        FIGURES/RESIDUE_CONTACT_MAPS/.
        """
        out_dir = "{}/FIGURES/RESIDUE_CONTACT_MAPS".format(self.analysis_root)
        os.makedirs(out_dir, exist_ok=True)

        conds = self._available_conditions()

        # Sequence files live in analysis/sequences; analysis-root CM_NORM holds
        # normalization maps generated by MaxCluster.
        cm_norm_dir = self._find_cm_norm_with_sequences()

        bionum_filename = {"SG": "BioPolNum_sg_X.csv", "DSM": "BioPolNum_DSM.csv", "NDSM": "BioPolNum_NDSM.csv"}
        res_filename    = {"SG": "Residue_Contacts_Mean_sg_X.csv",
                           "DSM": "Residue_Contacts_Mean_DSM.csv",
                           "NDSM": "Residue_Contacts_Mean_NDSM.csv"}

        Nchain, Nres, count = {}, {}, {}
        for c in conds:
            biopolnum_path = self._analysis_file(f"ANALYSIS_{c}_AVE", bionum_filename[c])
            Nchain[c], Nres[c] = CN.build_residue_normalizers(biopolnum_path)
            count[c] = np.array(pd.read_csv(self._analysis_file(f"ANALYSIS_{c}_AVE", res_filename[c]), header=None))

        # AVE Residue_Contacts_Mean files contain TIME-AVERAGED COUNTS, not
        # joint distributions despite the "Mean" name: AVERAGE_SIMULATIONS reads
        # per-window Residue_Contacts_Count_<sm>_<t>.csv and averages them. The
        # data is <C_AB>_t (mean contacts per frame per chain pair); it does
        # NOT sum to 1.

        P = {c: CN.safe_divide(count[c], Nres[c]) for c in conds}
        K = {c: CN.safe_divide(count[c], Nchain[c]) for c in conds}

        # Diagonal correction: upstream biopolymer 7x7 contact_array iterates
        # ORDERED (chain_a, chain_b) pairs (both (a,b) and (b,a)), so for self
        # species each unique pair contributes twice. Cross-pairs (A != B) are
        # not doubled (the matrix is symmetric across the diagonal so each
        # unique cross-pair contributes once to [A,B] and once to [B,A], which
        # are different cells). Divide the diagonal by 2 so K_chain[A,A] reads
        # as "fraction of unique self chain pairs in contact" ∈ [0, 1], matching
        # the cross-pair interpretation.
        for c in conds:
            np.fill_diagonal(K[c], np.diag(K[c]) / 2.0)

        nlnP = {c: self._neg_log_abs(P[c]) for c in conds}
        log10K = {c: self._log10_arr(K[c]) for c in conds}

        # Shared color limits so the available condition maps share a scale
        P_vmax = self._shared_vmax(*[P[c] for c in conds])
        nlnP_vmax = self._shared_vmax(*[nlnP[c] for c in conds])
        K_vmax = self._shared_vmax(*[K[c] for c in conds])
        log10K_vmin = self._shared_vmin(*[log10K[c] for c in conds])
        log10K_vmax = self._shared_vmax(*[log10K[c] for c in conds])

        for cond in conds:
            fig = self.plot_res_cpm(P[cond], col="Reds", min=0.0, max=P_vmax, mid=P_vmax / 2.0)
            plt.savefig(f"{out_dir}/Residue_{cond}_P_contact_HeatMap.png", format="png", dpi=400)
            plt.close(fig)

            fig = self.plot_res_cpm(nlnP[cond], col="Reds_r", min=0.0, max=nlnP_vmax, mid=nlnP_vmax / 2.0)
            plt.savefig(f"{out_dir}/Residue_{cond}_P_contact_LOG_HeatMap.png", format="png", dpi=400)
            plt.close(fig)

            fig = self.plot_res_cpm(K[cond], col="Reds", min=0.0, max=K_vmax, mid=K_vmax / 2.0)
            plt.savefig(f"{out_dir}/Residue_{cond}_K_chain_HeatMap.png", format="png", dpi=400)
            plt.close(fig)

            fig = self.plot_res_cpm(log10K[cond], col="Reds",
                                    min=log10K_vmin, max=log10K_vmax,
                                    mid=(log10K_vmin + log10K_vmax) / 2.0)
            plt.savefig(f"{out_dir}/Residue_{cond}_K_chain_LOG_HeatMap.png", format="png", dpi=400)
            plt.close(fig)

            # 1D row-mean variants
            for arr, suffix, cmap, vmin, vmax in [
                (P[cond], "P_contact", "Reds", 0.0, P_vmax),
                (nlnP[cond], "P_contact_LOG", "Reds_r", 0.0, nlnP_vmax),
                (K[cond], "K_chain", "Reds", 0.0, K_vmax),
                (log10K[cond], "K_chain_LOG", "Reds", log10K_vmin, log10K_vmax),
            ]:
                arr_1d = np.nanmean(arr, axis=1).reshape(-1, 1)
                fig = self.plot_res_sm_cpm(arr_1d, col=cmap, min=vmin, max=vmax,
                                           mid=(vmin + vmax) / 2.0,
                                           sm_list=[""], figsize=(1.6, 3.2))
                plt.savefig(f"{out_dir}/Residue_{cond}_{suffix}_1D_HeatMap.png", format="png", dpi=400)
                plt.close(fig)

        # DIFF maps: DSM vs NDSM. Linear and log-ratio for both P and K.
        if {"DSM", "NDSM"}.issubset(conds):
            dP = P["DSM"] - P["NDSM"]
            dK = K["DSM"] - K["NDSM"]
            lnratio_P = CN.log_ratio(P["DSM"], P["NDSM"])
            lnratio_K = CN.log_ratio(K["DSM"], K["NDSM"])

            diff_outputs = [
                (dP, "DIFF_P_contact"),
                (lnratio_P, "DIFF_P_contact_LOGRATIO"),
                (dK, "DIFF_K_chain"),
                (lnratio_K, "DIFF_K_chain_LOGRATIO"),
            ]
            for arr, suffix in diff_outputs:
                vmax = self._sym_vmax_arr(arr)
                fig = self.plot_res_cpm(arr, col="coolwarm", min=-vmax, max=vmax, mid=0.0)
                plt.savefig(f"{out_dir}/Residue_{suffix}_HeatMap.png", format="png", dpi=400)
                plt.close(fig)

                arr_1d = np.nanmean(arr, axis=1).reshape(-1, 1)
                fig = self.plot_res_sm_cpm(arr_1d, col="coolwarm", min=-vmax, max=vmax, mid=0.0,
                                           sm_list=[""], figsize=(1.6, 3.2))
                plt.savefig(f"{out_dir}/Residue_{suffix}_1D_HeatMap.png", format="png", dpi=400)
                plt.close(fig)

    # ======================================================================
    # C_total: raw time-averaged residue/acid contact count <C_AB> — the
    # "drivers of SG formation" view that includes stoichiometric weight.
    #
    # Relationship to the other two maps:
    #   C_total[A,B]   =   K_chain[A,B]   *   N_chain_pairs(A,B)
    #                  =   P_contact[A,B] *   N_residue_pairs(A,B)
    #
    # File names (added — existing maps not modified):
    #   Residue_{SG,DSM,NDSM}_C_total_HeatMap.png        (linear, contacts/frame)
    #   Residue_{SG,DSM,NDSM}_C_total_LOG_HeatMap.png    (-ln(C/C_max), nats relative)
    #   Residue_DIFF_C_total_HeatMap.png                 (linear DeltaC = C_DSM - C_NDSM)
    #   Residue_DIFF_C_total_LOGRATIO_HeatMap.png        (ln(C_DSM / C_NDSM), nats)
    #   Plus _1D_ variants for each.
    # ======================================================================

    def gen_residue_cpms_ctotal(self):
        """Generate biopolymer C_total (raw contact count) residue maps (see banner above).

        For each condition saves the raw time-averaged contact count C (and
        -ln(C/C_max)), the P_membership map C/sum(C), their 1D row means, and the
        DSM-vs-NDSM difference / log-ratio maps, into
        FIGURES/RESIDUE_CONTACT_MAPS/.
        """
        out_dir = "{}/FIGURES/RESIDUE_CONTACT_MAPS".format(self.analysis_root)
        os.makedirs(out_dir, exist_ok=True)

        conds = self._available_conditions()

        res_filename = {"SG": "Residue_Contacts_Mean_sg_X.csv",
                        "DSM": "Residue_Contacts_Mean_DSM.csv",
                        "NDSM": "Residue_Contacts_Mean_NDSM.csv"}

        # AVE Residue_Contacts_Mean files contain TIME-AVERAGED COUNTS <C_AB>_t,
        # not joint distributions. Same as gen_residue_cpms_pcontact.
        C = {c: np.array(pd.read_csv(self._analysis_file(f"ANALYSIS_{c}_AVE", res_filename[c]), header=None))
             for c in conds}

        # Relative -ln(C/C_max): use the maximum across all available conditions
        # so the colorbar zero is a shared reference (apparent free energy
        # relative to the strongest contact found in any of SG/DSM/NDSM).
        shared_C_max = self._shared_vmax(*[C[c] for c in conds])

        def neg_log_relative(arr, C_max=shared_C_max):
            a = np.asarray(arr, dtype=float)
            out = np.full_like(a, np.nan)
            mask = np.isfinite(a) & (a > 0)
            out[mask] = -np.log(a[mask] / C_max)
            return out

        nlnC = {c: neg_log_relative(C[c]) for c in conds}

        C_vmax = shared_C_max
        nlnC_vmax = self._shared_vmax(*[nlnC[c] for c in conds])

        for cond in conds:
            fig = self.plot_res_cpm(C[cond], col="Reds", min=0.0, max=C_vmax, mid=C_vmax / 2.0)
            plt.savefig(f"{out_dir}/Residue_{cond}_C_total_HeatMap.png", format="png", dpi=400)
            plt.close(fig)

            fig = self.plot_res_cpm(nlnC[cond], col="Reds_r", min=0.0, max=nlnC_vmax, mid=nlnC_vmax / 2.0)
            plt.savefig(f"{out_dir}/Residue_{cond}_C_total_LOG_HeatMap.png", format="png", dpi=400)
            plt.close(fig)

            for arr, suffix, cmap, vmin, vmax in [
                (C[cond], "C_total", "Reds", 0.0, C_vmax),
                (nlnC[cond], "C_total_LOG", "Reds_r", 0.0, nlnC_vmax),
            ]:
                arr_1d = np.nanmean(arr, axis=1).reshape(-1, 1)
                fig = self.plot_res_sm_cpm(arr_1d, col=cmap, min=vmin, max=vmax,
                                           mid=(vmin + vmax) / 2.0,
                                           sm_list=[""], figsize=(1.6, 3.2))
                plt.savefig(f"{out_dir}/Residue_{cond}_{suffix}_1D_HeatMap.png", format="png", dpi=400)
                plt.close(fig)

        # DIFF maps (DSM vs NDSM): linear DeltaC and log-ratio ln(C_DSM/C_NDSM)
        if {"DSM", "NDSM"}.issubset(conds):
            dC = C["DSM"] - C["NDSM"]
            lnratio_C = CN.log_ratio(C["DSM"], C["NDSM"])
            for arr, suffix in [(dC, "DIFF_C_total"), (lnratio_C, "DIFF_C_total_LOGRATIO")]:
                vmax = self._sym_vmax_arr(arr)
                fig = self.plot_res_cpm(arr, col="coolwarm", min=-vmax, max=vmax, mid=0.0)
                plt.savefig(f"{out_dir}/Residue_{suffix}_HeatMap.png", format="png", dpi=400)
                plt.close(fig)

                arr_1d = np.nanmean(arr, axis=1).reshape(-1, 1)
                fig = self.plot_res_sm_cpm(arr_1d, col="coolwarm", min=-vmax, max=vmax, mid=0.0,
                                           sm_list=[""], figsize=(1.6, 3.2))
                plt.savefig(f"{out_dir}/Residue_{suffix}_1D_HeatMap.png", format="png", dpi=400)
                plt.close(fig)

        # P_membership (= C / sum(C)) companion maps: per-condition normalized
        # so each cell is "fraction of all contacts contributed by pair (A,B)".
        # Sums to 1 per matrix; colorbars bounded in [0, max_P_membership].
        Pm = {c: (C[c] / C[c].sum() if C[c].sum() > 0 else C[c]) for c in conds}
        Pm_vmax = self._shared_vmax(*[Pm[c] for c in conds])

        for cond in conds:
            fig = self.plot_res_cpm(Pm[cond], col="Reds", min=0.0, max=Pm_vmax, mid=Pm_vmax / 2.0)
            plt.savefig(f"{out_dir}/Residue_{cond}_C_total_P_membership_HeatMap.png", format="png", dpi=400)
            plt.close(fig)
            arr_1d = np.nanmean(Pm[cond], axis=1).reshape(-1, 1)
            fig = self.plot_res_sm_cpm(arr_1d, col="Reds", min=0.0, max=Pm_vmax, mid=Pm_vmax / 2.0,
                                       sm_list=[""], figsize=(1.6, 3.2))
            plt.savefig(f"{out_dir}/Residue_{cond}_C_total_P_membership_1D_HeatMap.png", format="png", dpi=400)
            plt.close(fig)

        # DIFF in P_membership space: compositional shift between conditions
        if {"DSM", "NDSM"}.issubset(conds):
            dPm = Pm["DSM"] - Pm["NDSM"]
            vmax = self._sym_vmax_arr(dPm)
            fig = self.plot_res_cpm(dPm, col="coolwarm", min=-vmax, max=vmax, mid=0.0)
            plt.savefig(f"{out_dir}/Residue_DIFF_C_total_P_membership_HeatMap.png", format="png", dpi=400)
            plt.close(fig)
            arr_1d = np.nanmean(dPm, axis=1).reshape(-1, 1)
            fig = self.plot_res_sm_cpm(arr_1d, col="coolwarm", min=-vmax, max=vmax, mid=0.0,
                                       sm_list=[""], figsize=(1.6, 3.2))
            plt.savefig(f"{out_dir}/Residue_DIFF_C_total_P_membership_1D_HeatMap.png", format="png", dpi=400)
            plt.close(fig)

    def gen_acid_cpms_ctotal(self):
        """Generate acid-type C_total (raw contact count) maps.

        Acid-resolution analogue of ``gen_residue_cpms_ctotal``: per condition
        saves raw count C (with bonded neighbours subtracted) and -ln(C/C_max),
        the P_membership map C/sum(C), their 1D row means, and the DSM-vs-NDSM
        difference / log-ratio maps, into FIGURES/ACID_CONTACT_MAPS/.
        """
        out_dir = "{}/FIGURES/ACID_CONTACT_MAPS".format(self.analysis_root)
        os.makedirs(out_dir, exist_ok=True)

        conds = self._available_conditions()

        acid_filename = {"SG": "Acid_Contacts_Mean_sg_X.csv",
                         "DSM": "Acid_Contacts_Mean_DSM.csv",
                         "NDSM": "Acid_Contacts_Mean_NDSM.csv"}
        bond_filename = {"SG": "BondNum_sg_X.csv", "DSM": "BondNum_DSM.csv", "NDSM": "BondNum_NDSM.csv"}

        raw = {c: np.array(pd.read_csv(self._analysis_file(f"ANALYSIS_{c}_AVE", acid_filename[c]), header=None)) for c in conds}
        nBond = {c: np.loadtxt(self._analysis_file(f"ANALYSIS_{c}_AVE", bond_filename[c]), delimiter=",", dtype=float) for c in conds}

        # Subtract bonded peptide neighbours (intra-residue covalent bonds in
        # the type-pair counts) to match the legacy acid-map convention.
        C = {c: raw[c] - nBond[c] for c in conds}

        shared_C_max = self._shared_vmax(*[C[c] for c in conds])

        def neg_log_relative(arr, C_max=shared_C_max):
            a = np.asarray(arr, dtype=float)
            out = np.full_like(a, np.nan)
            mask = np.isfinite(a) & (a > 0)
            out[mask] = -np.log(a[mask] / C_max)
            return out

        nlnC = {c: neg_log_relative(C[c]) for c in conds}

        C_vmax = shared_C_max
        nlnC_vmax = self._shared_vmax(*[nlnC[c] for c in conds])

        for cond in conds:
            fig = self.plot_acid_cpm(C[cond], col="Reds", min=0.0, max=C_vmax, mid=C_vmax / 2.0)
            plt.savefig(f"{out_dir}/Acid_{cond}_C_total_HeatMap.png", format="png", dpi=400)
            plt.close(fig)

            fig = self.plot_acid_cpm(nlnC[cond], col="Reds_r", min=0.0, max=nlnC_vmax, mid=nlnC_vmax / 2.0)
            plt.savefig(f"{out_dir}/Acid_{cond}_C_total_LOG_HeatMap.png", format="png", dpi=400)
            plt.close(fig)

            for arr, suffix, cmap, vmin, vmax in [
                (C[cond], "C_total", "Reds", 0.0, C_vmax),
                (nlnC[cond], "C_total_LOG", "Reds_r", 0.0, nlnC_vmax),
            ]:
                arr_1d = np.nanmean(arr, axis=1).reshape(-1, 1)
                fig = self.plot_acid_sm_cpm(arr_1d, col=cmap, min=vmin, max=vmax,
                                            mid=(vmin + vmax) / 2.0,
                                            sm_list=[""], figsize=(1.6, 3.2))
                plt.savefig(f"{out_dir}/Acid_{cond}_{suffix}_1D_HeatMap.png", format="png", dpi=400)
                plt.close(fig)

        if {"DSM", "NDSM"}.issubset(conds):
            dC = C["DSM"] - C["NDSM"]
            lnratio_C = CN.log_ratio(C["DSM"], C["NDSM"])
            for arr, suffix in [(dC, "DIFF_C_total"), (lnratio_C, "DIFF_C_total_LOGRATIO")]:
                vmax = self._sym_vmax_arr(arr)
                fig = self.plot_acid_cpm(arr, col="coolwarm", min=-vmax, max=vmax, mid=0.0)
                plt.savefig(f"{out_dir}/Acid_{suffix}_HeatMap.png", format="png", dpi=400)
                plt.close(fig)

                arr_1d = np.nanmean(arr, axis=1).reshape(-1, 1)
                fig = self.plot_acid_sm_cpm(arr_1d, col="coolwarm", min=-vmax, max=vmax, mid=0.0,
                                            sm_list=[""], figsize=(1.6, 3.2))
                plt.savefig(f"{out_dir}/Acid_{suffix}_1D_HeatMap.png", format="png", dpi=400)
                plt.close(fig)

        # P_membership = C / sum(C) — fraction of acid contacts per pair-type.
        Pm = {c: (C[c] / C[c].sum() if C[c].sum() > 0 else C[c]) for c in conds}
        Pm_vmax = self._shared_vmax(*[Pm[c] for c in conds])

        for cond in conds:
            fig = self.plot_acid_cpm(Pm[cond], col="Reds", min=0.0, max=Pm_vmax, mid=Pm_vmax / 2.0)
            plt.savefig(f"{out_dir}/Acid_{cond}_C_total_P_membership_HeatMap.png", format="png", dpi=400)
            plt.close(fig)
            arr_1d = np.nanmean(Pm[cond], axis=1).reshape(-1, 1)
            fig = self.plot_acid_sm_cpm(arr_1d, col="Reds", min=0.0, max=Pm_vmax, mid=Pm_vmax / 2.0,
                                        sm_list=[""], figsize=(1.6, 3.2))
            plt.savefig(f"{out_dir}/Acid_{cond}_C_total_P_membership_1D_HeatMap.png", format="png", dpi=400)
            plt.close(fig)

        if {"DSM", "NDSM"}.issubset(conds):
            dPm = Pm["DSM"] - Pm["NDSM"]
            vmax = self._sym_vmax_arr(dPm)
            fig = self.plot_acid_cpm(dPm, col="coolwarm", min=-vmax, max=vmax, mid=0.0)
            plt.savefig(f"{out_dir}/Acid_DIFF_C_total_P_membership_HeatMap.png", format="png", dpi=400)
            plt.close(fig)
            arr_1d = np.nanmean(dPm, axis=1).reshape(-1, 1)
            fig = self.plot_acid_sm_cpm(arr_1d, col="coolwarm", min=-vmax, max=vmax, mid=0.0,
                                        sm_list=[""], figsize=(1.6, 3.2))
            plt.savefig(f"{out_dir}/Acid_DIFF_C_total_P_membership_1D_HeatMap.png", format="png", dpi=400)
            plt.close(fig)

    # ======================================================================
    # SM contact maps with the new normalization scheme.
    # Matches the residue/acid family conventions:
    #   ΔP_membership_SM[A] = (C_SM_DSM[A] / Σ_A C_SM_DSM) − (same for NDSM)
    #     -> compositional shift: where DSM redistributes its contacts vs NDSM
    #   Δ(C / N_residues)_SM[A] = (C_SM_DSM[A] / N_res_DSM[A]) − (same for NDSM)
    #     -> per-residue contact-rate shift: proxy for ΔP_contact at the SM level
    #
    # File names (added alongside legacy maps):
    #   SM_Residue_DIFF_P_membership_HeatMap.png      (biopolymer-level, [-vmax, vmax])
    #   SM_Residue_DIFF_K_per_chain_HeatMap.png       (biopolymer-level, contacts per A chain)
    #   SM_Acid_DIFF_P_membership_HeatMap.png         (AA-level)
    #   SM_Acid_DIFF_P_contact_HeatMap.png            (AA-level, per AA atom)
    # ======================================================================

    # ======================================================================
    # Per-residue-pair P_contact at the biopolymer level.
    #
    # The biopolymer 7x7 contact_array used by K_chain counts (chain_a, chain_b)
    # ordered pairs in contact (a chain-level binary event). To get the
    # literature-standard per-RESIDUE-pair contact probability, we use the
    # domain contact maps (which are LA x LB sequence-index matrices summed
    # over chain pairs per frame):
    #
    #   P_res_pair(A,B) = sum(domain_mat[A,B]) / (N_chain_pairs(A,B) * L_A * L_B)
    #
    # This is bounded in [0,1] for any A,B and removes BOTH chain count
    # stoichiometry AND chain length. -ln(P_res_pair) is the apparent contact
    # free energy in k_BT (nats) for an inter-chain residue-residue contact —
    # the observable that Mittag/Krainer/Lin/Lindorff-Larsen LLPS papers plot.
    #
    # File names (added — alongside K_chain and P_membership):
    #   Residue_{SG,DSM,NDSM}_P_residue_pair_HeatMap.png        (linear)
    #   Residue_{SG,DSM,NDSM}_P_residue_pair_LOG_HeatMap.png    (-ln(P), nats)
    #   Residue_DIFF_P_residue_pair_HeatMap.png                 (linear DeltaP)
    #   Residue_DIFF_P_residue_pair_LOGRATIO_HeatMap.png        (ln(P_DSM/P_NDSM))
    # ======================================================================

    def gen_residue_cpms_p_residue_pair(self):
        """Per-residue-pair contact probability at biopolymer level.

        Reads the existing AVE-level Domain_Contacts_Mean_*_{A}_{B}.csv files
        produced by contact_maps.py (residue-pair-resolved). No upstream re-run needed.
        """
        out_dir = "{}/FIGURES/RESIDUE_CONTACT_MAPS".format(self.analysis_root)
        os.makedirs(out_dir, exist_ok=True)

        # Species order matches BIOPOLYMER_ANALYSIS 7x7 maps (CN.SPECIES_ORDER):
        # G3BP1, PABP1, TTP, TIA1, TDP43, FUS, RNA
        species_to_bio = {  # 7x7 species name -> ARRAY species name (with "Protein" prefix where applicable)
            "G3BP1": "ProteinG3BP1", "PABP1": "ProteinPABP1", "TTP": "ProteinTTP",
            "TIA1": "ProteinTIA1", "TDP43": "ProteinTDP43", "FUS": "ProteinFUS",
            "RNA": "RNA",
        }

        EXCLUSION = 5  # bonded sequence-distance for intra exclusion

        def load_p_residue_pair(category, biopolnum_filename, variant):
            """Compute 7x7 P_residue_pair matrix for category and variant.
            variant: 'all' | 'inter' | 'intra'
            Numerator: collapse appropriate Domain_Contacts_Mean* file.
            Denominator: variant-specific residue-pair count.
            """
            prefix_map = {
                "all":   "Domain_Contacts_Mean",
                "inter": "Domain_Contacts_MeanInter",
                "intra": "Domain_Contacts_MeanIntraFiltered",
            }
            domain_prefix = prefix_map[variant]
            bio = np.loadtxt(self._analysis_file(f"ANALYSIS_{category}_AVE", biopolnum_filename),
                             delimiter=",", dtype=float)
            n_chains = CN.n_chains_from_biopolnum(bio)
            mat = np.full((7, 7), np.nan, dtype=float)
            cat_dir = self._analysis_dir(f"ANALYSIS_{category}_AVE")
            cat_tag = "sg_X" if category == "SG" else category
            for i, sp_i in enumerate(CN.SPECIES_ORDER):
                for j in range(i, 7):
                    sp_j = CN.SPECIES_ORDER[j]
                    bio_i = species_to_bio[sp_i]
                    bio_j = species_to_bio[sp_j]
                    L_i = CN.SPECIES_LENGTH[sp_i]
                    L_j = CN.SPECIES_LENGTH[sp_j]
                    # INTRA only exists for self-species (intra-chain by definition)
                    if variant == "intra" and i != j:
                        continue
                    candidates = [
                        os.path.join(cat_dir, f"{domain_prefix}_{cat_tag}_{bio_i}_{bio_j}.csv"),
                        os.path.join(cat_dir, f"{domain_prefix}_{cat_tag}_{bio_j}_{bio_i}.csv"),
                    ]
                    path = next((p for p in candidates if os.path.exists(p)), None)
                    if path is None:
                        continue
                    try:
                        dm = np.loadtxt(path, delimiter=",")
                    except Exception:
                        continue
                    total = float(np.nansum(dm))
                    # Variant-specific denominator
                    if i == j:
                        n = n_chains[i]
                        N_chain_pairs_inter = n * (n - 1) / 2.0  # unique chain pairs (inter)
                        # Count intra-chain (i,j) pairs at |i-j| > EXCLUSION within a single chain:
                        # = L*(L-1)/2 - bonded_within_5
                        bonded = 0
                        for d in range(1, EXCLUSION + 1):
                            bonded += max(0, L_i - d)
                        N_intra_filtered_per_chain = (L_i * (L_i - 1) / 2.0) - bonded
                        if variant == "inter":
                            denom = N_chain_pairs_inter * L_i * L_j
                        elif variant == "intra":
                            denom = n * N_intra_filtered_per_chain
                        else:  # all = inter + intra_filtered
                            denom = N_chain_pairs_inter * L_i * L_j + n * N_intra_filtered_per_chain
                    else:
                        # Cross-species: only inter exists; intra not defined here.
                        N_chain_pairs_inter = n_chains[i] * n_chains[j]
                        denom = N_chain_pairs_inter * L_i * L_j
                    if denom > 0:
                        mat[i, j] = total / denom
                        mat[j, i] = mat[i, j]
            return mat

        conds = self._available_conditions()
        bionum_filename = {"SG": "BioPolNum_sg_X.csv", "DSM": "BioPolNum_DSM.csv", "NDSM": "BioPolNum_NDSM.csv"}

        # Produce three variants: ALL (inter + intra_filtered), INTER (inter only),
        # INTRA (intra non-bonded only). File-suffix conventions match the acid maps.
        for v_suffix, v_key in [("ALL", "all"), ("INTER", "inter"), ("INTRA", "intra")]:
            try:
                P = {c: load_p_residue_pair(c, bionum_filename[c], v_key) for c in conds}
            except Exception as exc:
                print(f"[WARN] gen_residue_cpms_p_residue_pair {v_suffix}: failed loading domain maps: {exc}")
                continue

            P_global_max = self._shared_vmax(*[P[c] for c in conds])

            def neg_log_relative(arr, ref=P_global_max):
                a = np.asarray(arr, dtype=float)
                out = np.full_like(a, np.nan)
                mask = np.isfinite(a) & (a > 0) & (ref > 0)
                out[mask] = -np.log(a[mask] / ref)
                return out

            nlnP = {c: neg_log_relative(P[c]) for c in conds}

            P_vmax = P_global_max
            nlnP_vmax = self._shared_vmax(*[nlnP[c] for c in conds])

            for cond in conds:
                fig = self.plot_res_cpm(P[cond], col="Reds", min=0.0, max=P_vmax, mid=P_vmax / 2.0)
                plt.savefig(f"{out_dir}/Residue_{cond}_P_residue_pair_{v_suffix}_HeatMap.png", format="png", dpi=400)
                plt.close(fig)
                fig = self.plot_res_cpm(nlnP[cond], col="Reds_r", min=0.0, max=nlnP_vmax, mid=nlnP_vmax / 2.0)
                plt.savefig(f"{out_dir}/Residue_{cond}_P_residue_pair_{v_suffix}_LOG_HeatMap.png", format="png", dpi=400)
                plt.close(fig)

            # DIFF maps per variant
            if {"DSM", "NDSM"}.issubset(conds):
                dP = P["DSM"] - P["NDSM"]
                lnratio = CN.log_ratio(P["DSM"], P["NDSM"])
                for arr, suffix in [(dP, f"DIFF_P_residue_pair_{v_suffix}"),
                                    (lnratio, f"DIFF_P_residue_pair_{v_suffix}_LOGRATIO")]:
                    vmax = self._sym_vmax_arr(arr)
                    fig = self.plot_res_cpm(arr, col="coolwarm", min=-vmax, max=vmax, mid=0.0)
                    plt.savefig(f"{out_dir}/Residue_{suffix}_HeatMap.png", format="png", dpi=400)
                    plt.close(fig)

    def gen_sm_diff_pcontact(self):
        """Generate DSM-vs-NDSM SM-contact membership / per-residue-rate difference maps.

        At both biopolymer and amino-acid resolution, saves the DSM-minus-NDSM
        shift in SM-contact P_membership (fraction of SM contacts per target)
        and in the per-residue / per-atom contact rate, into
        FIGURES/SM_CONTACT_MAPS/. No-ops unless DSM and NDSM data are present.
        """
        out_dir = "{}/FIGURES/SM_CONTACT_MAPS".format(self.analysis_root)
        os.makedirs(out_dir, exist_ok=True)

        conds = self._available_conditions()
        if not {"DSM", "NDSM"}.issubset(conds):
            print(f"[SKIP] {self.__class__.__name__}.gen_sm_diff_pcontact: no DSM/NDSM data available")
            return

        # Biopolymer-level SM contact data
        df_res = pd.read_csv(os.path.join(self.analysis_root, "RESULTS", "SM_CONTACT_MAPS", "SM_ResMap_Data.csv"))
        df_res_count = pd.read_csv(os.path.join(self.analysis_root, "RESULTS", "SM_CONTACT_MAPS", "SM_ResCount_Data.csv"))
        if "DSM" not in df_res.columns or "NDSM" not in df_res.columns:
            print("[INFO] gen_sm_diff_pcontact: DSM/NDSM aggregate columns absent; skipping")
            return

        c_dsm = np.asarray(df_res.loc[:, "DSM"], dtype=float)
        c_ndsm = np.asarray(df_res.loc[:, "NDSM"], dtype=float)

        # ΔP_membership: fraction of all SM-residue contacts going to each biopolymer
        pm_dsm = c_dsm / c_dsm.sum() if c_dsm.sum() > 0 else c_dsm
        pm_ndsm = c_ndsm / c_ndsm.sum() if c_ndsm.sum() > 0 else c_ndsm
        d_pm = (pm_dsm - pm_ndsm).reshape(-1, 1)
        vmax = self._sym_vmax_arr(d_pm)
        fig = self.plot_res_sm_cpm(d_pm, col="coolwarm", min=-vmax, max=vmax, mid=0.0,
                                   sm_list=[""], show_xticks=False)
        plt.savefig(f"{out_dir}/SM_Residue_DIFF_P_membership_HeatMap.png", format="png", dpi=400)
        plt.close(fig)

        # Δ(contacts / N_residues): per-residue SM-contact rate shift (proxy for ΔP_contact)
        if "DSM" in df_res_count.columns and "NDSM" in df_res_count.columns:
            n_dsm = np.asarray(df_res_count.loc[:, "DSM"], dtype=float)
            n_ndsm = np.asarray(df_res_count.loc[:, "NDSM"], dtype=float)
            r_dsm = np.where(n_dsm > 0, c_dsm / n_dsm, np.nan)
            r_ndsm = np.where(n_ndsm > 0, c_ndsm / n_ndsm, np.nan)
            d_r = (r_dsm - r_ndsm).reshape(-1, 1)
            vmax = self._sym_vmax_arr(d_r)
            fig = self.plot_res_sm_cpm(d_r, col="coolwarm", min=-vmax, max=vmax, mid=0.0,
                                       sm_list=[""], show_xticks=False)
            plt.savefig(f"{out_dir}/SM_Residue_DIFF_K_per_chain_HeatMap.png", format="png", dpi=400)
            plt.close(fig)

        # AA-level SM contact data
        df_acid = pd.read_csv(os.path.join(self.analysis_root, "RESULTS", "SM_CONTACT_MAPS", "SM_AcidMap_Data.csv"))
        df_acid_count = pd.read_csv(os.path.join(self.analysis_root, "RESULTS", "SM_CONTACT_MAPS", "SM_AcidCount_Data.csv"))
        if "DSM" not in df_acid.columns or "NDSM" not in df_acid.columns:
            print("[INFO] gen_sm_diff_pcontact: AA-level DSM/NDSM aggregate columns absent; skipping AA maps")
            return

        ca_dsm = np.asarray(df_acid.loc[:, "DSM"], dtype=float)
        ca_ndsm = np.asarray(df_acid.loc[:, "NDSM"], dtype=float)

        pma_dsm = ca_dsm / ca_dsm.sum() if ca_dsm.sum() > 0 else ca_dsm
        pma_ndsm = ca_ndsm / ca_ndsm.sum() if ca_ndsm.sum() > 0 else ca_ndsm
        d_pma = (pma_dsm - pma_ndsm).reshape(-1, 1)
        vmax = self._sym_vmax_arr(d_pma)
        fig = self.plot_acid_sm_cpm(d_pma, col="coolwarm", min=-vmax, max=vmax, mid=0.0,
                                    sm_list=[""], show_xticks=False)
        plt.savefig(f"{out_dir}/SM_Acid_DIFF_P_membership_HeatMap.png", format="png", dpi=400)
        plt.close(fig)

        # Δ(contacts / N_atoms_of_AA_type): per-atom SM-contact probability shift
        if "DSM" in df_acid_count.columns and "NDSM" in df_acid_count.columns:
            na_dsm = np.asarray(df_acid_count.loc[:, "DSM"], dtype=float)
            na_ndsm = np.asarray(df_acid_count.loc[:, "NDSM"], dtype=float)
            pa_dsm = np.where(na_dsm > 0, ca_dsm / na_dsm, np.nan)
            pa_ndsm = np.where(na_ndsm > 0, ca_ndsm / na_ndsm, np.nan)
            d_pa = (pa_dsm - pa_ndsm).reshape(-1, 1)
            vmax = self._sym_vmax_arr(d_pa)
            fig = self.plot_acid_sm_cpm(d_pa, col="coolwarm", min=-vmax, max=vmax, mid=0.0,
                                        sm_list=[""], show_xticks=False)
            plt.savefig(f"{out_dir}/SM_Acid_DIFF_P_contact_HeatMap.png", format="png", dpi=400)
            plt.close(fig)

    def gen_sm_membership_enrichment(self):
        """SM-biopolymer contact membership f and contact enrichment ln(E)
        at both biopolymer (7-vector) and atom-type (24-vector) resolution.

        Math (per the user spec):
          - SM-Residue level: N_possible(SM, A) = N_SM * N_A * L_A. With
            constant N_SM the SM factor cancels in q, so:
              q[A]   = M_A / sum_A M_A   (M_A = N_A * L_A residues of species A)
              f[A]   = C_SM_A / sum_A C_SM_A
              ln E   = ln(f / q)
          - SM-Acid level: N_possible(SM, i) = N_SM * N_i, hence:
              q[i]   = N_i / sum_i N_i
              f[i]   = C_SM_i / sum_i C_SM_i
              ln E   = ln(f / q)

        Output files in FIGURES/SM_CONTACT_MAPS:
          SM_Residue_{cond}_CONTACT_MEMBERSHIP_HeatMap.png       (Reds, linear f)
          SM_Residue_{cond}_CONTACT_ENRICHMENT_LOG_HeatMap.png   (coolwarm @0, ln E)
          SM_Acid_{cond}_CONTACT_MEMBERSHIP_HeatMap.png
          SM_Acid_{cond}_CONTACT_ENRICHMENT_LOG_HeatMap.png
          SM_{Residue,Acid}_DIFF_CONTACT_MEMBERSHIP_HeatMap.png  (Δf, coolwarm @0)
          SM_{Residue,Acid}_DIFF_CONTACT_ENRICHMENT_LOG_HeatMap.png (Δln E, coolwarm @0)
        for cond in {DSM, NDSM} aggregate columns.
        """
        out_dir = f"{self.analysis_root}/FIGURES/SM_CONTACT_MAPS"
        os.makedirs(out_dir, exist_ok=True)

        conds = self._available_conditions()
        if not {"DSM", "NDSM"}.issubset(conds):
            print(f"[SKIP] {self.__class__.__name__}.gen_sm_membership_enrichment: no DSM/NDSM data available")
            return

        try:
            df_res = pd.read_csv(os.path.join(self.analysis_root, "RESULTS", "SM_CONTACT_MAPS", "SM_ResMap_Data.csv"))
            df_res_count = pd.read_csv(os.path.join(self.analysis_root, "RESULTS", "SM_CONTACT_MAPS", "SM_ResCount_Data.csv"))
        except Exception as exc:
            print(f"[INFO] gen_sm_membership_enrichment: missing SM_ResMap/Count_Data.csv: {exc}")
            return
        try:
            df_acid = pd.read_csv(os.path.join(self.analysis_root, "RESULTS", "SM_CONTACT_MAPS", "SM_AcidMap_Data.csv"))
            df_acid_count = pd.read_csv(os.path.join(self.analysis_root, "RESULTS", "SM_CONTACT_MAPS", "SM_AcidCount_Data.csv"))
        except Exception as exc:
            print(f"[INFO] gen_sm_membership_enrichment: missing SM_AcidMap/Count_Data.csv: {exc}")
            df_acid = None
            df_acid_count = None

        def _membership(c):
            c = np.asarray(c, dtype=float)
            s = float(np.nansum(c[np.isfinite(c)]))
            if s <= 0:
                return np.zeros_like(c)
            return c / s

        def _log_enrichment(c, n_possible):
            f = _membership(c)
            n = np.asarray(n_possible, dtype=float)
            sn = float(np.nansum(n[np.isfinite(n)]))
            out = np.full_like(f, np.nan, dtype=float)
            if sn <= 0:
                return out
            q = n / sn
            mask = np.isfinite(f) & np.isfinite(q) & (f > 0) & (q > 0)
            out[mask] = np.log(f[mask] / q[mask])
            return out

        def _save_panels(level, cond, c_vec, n_vec, plot_func):
            f = _membership(c_vec).reshape(-1, 1)
            lnE = _log_enrichment(c_vec, n_vec).reshape(-1, 1)
            f_vmax = float(np.nanmax(f)) if np.any(np.isfinite(f)) else 1e-6
            fig = plot_func(f, col="Reds", min=0.0, max=max(f_vmax, 1e-6), mid=f_vmax / 2.0,
                            sm_list=[""], show_xticks=False)
            plt.savefig(f"{out_dir}/SM_{level}_{cond}_CONTACT_MEMBERSHIP_HeatMap.png", format="png", dpi=400)
            plt.close(fig)
            lnE_abs = float(np.nanmax(np.abs(lnE))) if np.any(np.isfinite(lnE)) else 1e-6
            lnE_abs = max(lnE_abs, 1e-6)
            fig = plot_func(lnE, col="coolwarm", min=-lnE_abs, max=lnE_abs, mid=0.0,
                            sm_list=[""], show_xticks=False)
            plt.savefig(f"{out_dir}/SM_{level}_{cond}_CONTACT_ENRICHMENT_LOG_HeatMap.png", format="png", dpi=400)
            plt.close(fig)

        def _save_ctotal(level, c_d, c_n, plot_func):
            """C_total (raw SM-contact counts) maps: per-condition, raw difference
            (Delta C), and log-ratio ln(C_DSM/C_NDSM). The log-ratio is the SM-level
            equivalent of the old SM_*_DIFF_UNNORMALIZED net-contact-difference map."""
            cd = np.asarray(c_d, dtype=float)
            cn = np.asarray(c_n, dtype=float)
            finite_d = cd[np.isfinite(cd)]
            finite_n = cn[np.isfinite(cn)]
            cmax = max(float(np.nanmax(finite_d)) if finite_d.size else 0.0,
                       float(np.nanmax(finite_n)) if finite_n.size else 0.0, 1e-6)
            for cond, c in [("DSM", cd), ("NDSM", cn)]:
                fig = plot_func(c.reshape(-1, 1), col="Reds", min=0.0, max=cmax, mid=cmax / 2.0,
                                sm_list=[""], show_xticks=False)
                plt.savefig(f"{out_dir}/SM_{level}_{cond}_C_total_HeatMap.png", format="png", dpi=400)
                plt.close(fig)
            dC = (cd - cn).reshape(-1, 1)
            vmax = self._sym_vmax_arr(dC)
            fig = plot_func(dC, col="coolwarm", min=-vmax, max=vmax, mid=0.0,
                            sm_list=[""], show_xticks=False)
            plt.savefig(f"{out_dir}/SM_{level}_DIFF_C_total_HeatMap.png", format="png", dpi=400)
            plt.close(fig)
            lnr = CN.log_ratio(cd, cn).reshape(-1, 1)
            vmax2 = self._sym_vmax_arr(lnr)
            fig = plot_func(lnr, col="coolwarm", min=-vmax2, max=vmax2, mid=0.0,
                            sm_list=[""], show_xticks=False)
            plt.savefig(f"{out_dir}/SM_{level}_DIFF_C_total_LOGRATIO_HeatMap.png", format="png", dpi=400)
            plt.close(fig)

        # --- Biopolymer (residue / species) level ---
        if "DSM" in df_res.columns and "NDSM" in df_res.columns:
            n_dsm = np.asarray(df_res_count.loc[:, "DSM"], dtype=float) if "DSM" in df_res_count.columns else None
            n_ndsm = np.asarray(df_res_count.loc[:, "NDSM"], dtype=float) if "NDSM" in df_res_count.columns else None
            for cond, c_col, n_vec in [("DSM", "DSM", n_dsm), ("NDSM", "NDSM", n_ndsm)]:
                if n_vec is None:
                    continue
                c_vec = np.asarray(df_res.loc[:, c_col], dtype=float)
                _save_panels("Residue", cond, c_vec, n_vec, self.plot_res_sm_cpm)
            if n_dsm is not None and n_ndsm is not None:
                c_d = np.asarray(df_res.loc[:, "DSM"], dtype=float)
                c_n = np.asarray(df_res.loc[:, "NDSM"], dtype=float)
                df_pm = (_membership(c_d) - _membership(c_n)).reshape(-1, 1)
                df_lnE = (_log_enrichment(c_d, n_dsm) - _log_enrichment(c_n, n_ndsm)).reshape(-1, 1)
                vmax = self._sym_vmax_arr(df_pm)
                fig = self.plot_res_sm_cpm(df_pm, col="coolwarm", min=-vmax, max=vmax, mid=0.0,
                                            sm_list=[""], show_xticks=False)
                plt.savefig(f"{out_dir}/SM_Residue_DIFF_CONTACT_MEMBERSHIP_HeatMap.png", format="png", dpi=400)
                plt.close(fig)
                vmax2 = self._sym_vmax_arr(df_lnE)
                fig = self.plot_res_sm_cpm(df_lnE, col="coolwarm", min=-vmax2, max=vmax2, mid=0.0,
                                            sm_list=[""], show_xticks=False)
                plt.savefig(f"{out_dir}/SM_Residue_DIFF_CONTACT_ENRICHMENT_LOG_HeatMap.png", format="png", dpi=400)
                plt.close(fig)
                # SM-biopolymer C_total maps incl. DIFF_C_total_LOGRATIO (= old DIFF_UNNORMALIZED equiv.)
                _save_ctotal("Residue", c_d, c_n, self.plot_res_sm_cpm)

        # --- Acid (atom-type) level ---
        if df_acid is not None and df_acid_count is not None and \
           "DSM" in df_acid.columns and "NDSM" in df_acid.columns:
            na_dsm = np.asarray(df_acid_count.loc[:, "DSM"], dtype=float) if "DSM" in df_acid_count.columns else None
            na_ndsm = np.asarray(df_acid_count.loc[:, "NDSM"], dtype=float) if "NDSM" in df_acid_count.columns else None
            for cond, c_col, n_vec in [("DSM", "DSM", na_dsm), ("NDSM", "NDSM", na_ndsm)]:
                if n_vec is None:
                    continue
                c_vec = np.asarray(df_acid.loc[:, c_col], dtype=float)
                _save_panels("Acid", cond, c_vec, n_vec, self.plot_acid_sm_cpm)
            if na_dsm is not None and na_ndsm is not None:
                c_d = np.asarray(df_acid.loc[:, "DSM"], dtype=float)
                c_n = np.asarray(df_acid.loc[:, "NDSM"], dtype=float)
                df_pm = (_membership(c_d) - _membership(c_n)).reshape(-1, 1)
                df_lnE = (_log_enrichment(c_d, na_dsm) - _log_enrichment(c_n, na_ndsm)).reshape(-1, 1)
                vmax = self._sym_vmax_arr(df_pm)
                fig = self.plot_acid_sm_cpm(df_pm, col="coolwarm", min=-vmax, max=vmax, mid=0.0,
                                             sm_list=[""], show_xticks=False)
                plt.savefig(f"{out_dir}/SM_Acid_DIFF_CONTACT_MEMBERSHIP_HeatMap.png", format="png", dpi=400)
                plt.close(fig)
                vmax2 = self._sym_vmax_arr(df_lnE)
                fig = self.plot_acid_sm_cpm(df_lnE, col="coolwarm", min=-vmax2, max=vmax2, mid=0.0,
                                             sm_list=[""], show_xticks=False)
                plt.savefig(f"{out_dir}/SM_Acid_DIFF_CONTACT_ENRICHMENT_LOG_HeatMap.png", format="png", dpi=400)
                plt.close(fig)
                # SM-acid C_total maps incl. DIFF_C_total_LOGRATIO (= old DIFF_UNNORMALIZED equiv.)
                _save_ctotal("Acid", c_d, c_n, self.plot_acid_sm_cpm)

    def gen_acid_cpms_pcontact(self):
        """Per-atom-pair INTER-CHAIN contact probability at the AA-type level,
        derived from the residue-resolved Domain_Contacts_MeanInter maps. Only
        inter-chain atom-atom contacts contribute — intra-chain proximity
        (folded-domain structure, e.g. nearby CYS residues in RRMs; IDR
        compactness) is fully excluded. This is the literature-standard
        inter-molecular contact probability for LLPS condensate analysis
        (Joseph 2021 MPiPi, Krainer 2021 Mittag, Tesei 2025 CALVADOS).

        Math:
            P_contact[i,j] = N_inter_contacts[i,j]  /  N_inter_atom_pairs[i,j]

        where:
            N_inter_contacts = sum over species pairs (A,B) of
                               sum over (k,l) of M_inter_AB[k,l]
                               * (one-hot type indicators)
            N_inter_atom_pairs = N_atom_pairs_total - N_intra_atom_pairs
        """
        import re
        out_dir = "{}/FIGURES/ACID_CONTACT_MAPS".format(self.analysis_root)
        os.makedirs(out_dir, exist_ok=True)

        cm_norm_dir = self._find_cm_norm_with_sequences()

        # Parse FASTA -> per-position LAMMPS atom type for each species.
        types_per_species = {}
        for sp in CN.SPECIES_ORDER:
            path = os.path.join(cm_norm_dir, CN.SPECIES_SEQ_FILE[sp])
            seq = re.sub(r"[^A-Za-z]", "", open(path).read()).upper()
            mapping = CN.RNA_LETTER_TO_TYPE if sp == "RNA" else CN.PROTEIN_LETTER_TO_TYPE
            types_per_species[sp] = [mapping.get(c, None) for c in seq]

        def one_hot(types):
            L = len(types); X = np.zeros((CN.N_ATOM_TYPES, L))
            for i, t in enumerate(types):
                if t is not None: X[t - 1, i] = 1.0
            return X

        X_species = {sp: one_hot(types_per_species[sp]) for sp in CN.SPECIES_ORDER}

        SPECIES_BIONAME = {"G3BP1": "ProteinG3BP1", "PABP1": "ProteinPABP1",
                           "TTP": "ProteinTTP", "TIA1": "ProteinTIA1",
                           "TDP43": "ProteinTDP43", "FUS": "ProteinFUS", "RNA": "RNA"}

        EXCLUSION = 5  # sequence-distance exclusion for "bonded" pairs

        def derive_acid_contacts(category, domain_prefix):
            """Project the per-species-pair domain maps onto a 24x24 atom-type contact matrix."""
            cat_dir = self._analysis_dir(f"ANALYSIS_{category}_AVE")
            cat_tag = "sg_X" if category == "SG" else category
            acid = np.zeros((CN.N_ATOM_TYPES, CN.N_ATOM_TYPES))
            for i, A in enumerate(CN.SPECIES_ORDER):
                for j in range(i, 7):
                    B = CN.SPECIES_ORDER[j]
                    # For INTRA prefix, only self-species pairs have data (skip cross-species)
                    if domain_prefix.endswith("IntraFiltered") and A != B:
                        continue
                    bio_a, bio_b = SPECIES_BIONAME[A], SPECIES_BIONAME[B]
                    candidates = [
                        os.path.join(cat_dir, f"{domain_prefix}_{cat_tag}_{bio_a}_{bio_b}.csv"),
                        os.path.join(cat_dir, f"{domain_prefix}_{cat_tag}_{bio_b}_{bio_a}.csv"),
                    ]
                    path = next((p for p in candidates if os.path.exists(p)), None)
                    if path is None:
                        continue
                    try:
                        M = np.loadtxt(path, delimiter=",")
                    except Exception:
                        continue
                    L_a, L_b = CN.SPECIES_LENGTH[A], CN.SPECIES_LENGTH[B]
                    if M.shape != (L_a, L_b):
                        M = M.T
                        if M.shape != (L_a, L_b):
                            continue
                    contrib = X_species[A] @ M @ X_species[B].T
                    if A == B:
                        acid += contrib
                    else:
                        acid += contrib + contrib.T
            return acid

        def n_intra_all(n_chains_vec, comp):
            """Count all intra-chain atom-type pairs (24x24) summed over every chain."""
            N_intra = np.zeros((CN.N_ATOM_TYPES, CN.N_ATOM_TYPES))
            for sp_idx, sp in enumerate(CN.SPECIES_ORDER):
                counts = np.zeros(CN.N_ATOM_TYPES)
                _, sp_comp = comp[sp]
                for t, c in sp_comp.items():
                    counts[t - 1] = c
                per_chain = np.outer(counts, counts)
                np.fill_diagonal(per_chain, counts * np.maximum(counts - 1, 0) / 2.0)
                N_intra += n_chains_vec[sp_idx] * per_chain
            return N_intra

        def n_bonded_within_5(n_chains_vec):
            """Count bonded (sequence-distance <= EXCLUSION) intra-chain atom-type pairs (24x24)."""
            N_b = np.zeros((CN.N_ATOM_TYPES, CN.N_ATOM_TYPES))
            for sp_idx, sp in enumerate(CN.SPECIES_ORDER):
                L = CN.SPECIES_LENGTH[sp]
                Bmask = np.zeros((L, L))
                for d in range(1, EXCLUSION + 1):
                    for k in range(L - d):
                        Bmask[k, k + d] = 1.0
                        Bmask[k + d, k] = 1.0
                contrib = X_species[sp] @ Bmask @ X_species[sp].T
                np.fill_diagonal(contrib, np.diag(contrib) / 2.0)
                N_b += n_chains_vec[sp_idx] * contrib
            return N_b

        def build_pair_counts(biopolnum_path, variant):
            """Return N_pairs denominator for a given variant.
            variant: "all" -> total minus bonded(<=5)  (= inter + intra_filtered)
                     "inter" -> total minus all intra
                     "intra" -> intra_all minus bonded(<=5)  (= intra_filtered)
            """
            bio = np.loadtxt(biopolnum_path, delimiter=",", dtype=float)
            n_chains_vec = CN.n_chains_from_biopolnum(bio)
            comp = CN.load_all_species_composition(cm_norm_dir)
            n_atoms = CN.n_atoms_per_type_vector(n_chains_vec, comp)
            N_total = CN.n_atom_pairs_matrix(n_atoms)
            N_intra = n_intra_all(n_chains_vec, comp)
            N_bonded = n_bonded_within_5(n_chains_vec)
            if variant == "all":
                return N_total - N_bonded
            elif variant == "inter":
                return N_total - N_intra
            elif variant == "intra":
                return N_intra - N_bonded
            else:
                raise ValueError(f"Unknown variant {variant}")

        conds = self._available_conditions()
        biopolnum_path = {"SG": self._analysis_file("ANALYSIS_SG_AVE", "BioPolNum_sg_X.csv"),
                          "DSM": self._analysis_file("ANALYSIS_DSM_AVE", "BioPolNum_DSM.csv"),
                          "NDSM": self._analysis_file("ANALYSIS_NDSM_AVE", "BioPolNum_NDSM.csv")}

        VARIANTS = [
            ("ALL",   "Domain_Contacts_Mean",            "all"),
            ("INTER", "Domain_Contacts_MeanInter",       "inter"),
            ("INTRA", "Domain_Contacts_MeanIntraFiltered", "intra"),
        ]

        for v_suffix, domain_prefix, denom_variant in VARIANTS:
            acid_arr = {c: derive_acid_contacts(c, domain_prefix) for c in conds}
            Npairs = {c: build_pair_counts(biopolnum_path[c], denom_variant) for c in conds}

            P = {c: CN.safe_divide(acid_arr[c], Npairs[c]) for c in conds}

            P_global_max = self._shared_vmax(*[P[c] for c in conds])

            def neg_log_relative(arr, ref=P_global_max):
                a = np.asarray(arr, dtype=float)
                out = np.full_like(a, np.nan)
                mask = np.isfinite(a) & (a > 0) & (ref > 0)
                out[mask] = -np.log(a[mask] / ref)
                return out

            nlnP = {c: neg_log_relative(P[c]) for c in conds}

            P_vmax = P_global_max
            nlnP_vmax = self._shared_vmax(*[nlnP[c] for c in conds])

            for cond in conds:
                fig = self.plot_acid_cpm(P[cond], col="Reds", min=0.0, max=P_vmax, mid=P_vmax / 2.0)
                plt.savefig(f"{out_dir}/Acid_{cond}_P_contact_{v_suffix}_HeatMap.png", format="png", dpi=400)
                plt.close(fig)
                fig = self.plot_acid_cpm(nlnP[cond], col="Reds_r", min=0.0, max=nlnP_vmax, mid=nlnP_vmax / 2.0)
                plt.savefig(f"{out_dir}/Acid_{cond}_P_contact_{v_suffix}_LOG_HeatMap.png", format="png", dpi=400)
                plt.close(fig)

            # DIFF maps for this variant
            if {"DSM", "NDSM"}.issubset(conds):
                dP = P["DSM"] - P["NDSM"]
                lnratio_P = CN.log_ratio(P["DSM"], P["NDSM"])
                for arr, suffix in [(dP, f"DIFF_P_contact_{v_suffix}"),
                                    (lnratio_P, f"DIFF_P_contact_{v_suffix}_LOGRATIO")]:
                    vmax = self._sym_vmax_arr(arr)
                    fig = self.plot_acid_cpm(arr, col="coolwarm", min=-vmax, max=vmax, mid=0.0)
                    plt.savefig(f"{out_dir}/Acid_{suffix}_HeatMap.png", format="png", dpi=400)
                    plt.close(fig)


    def gen_residue_cpms_membership_enrichment(self):
        """Two-panel residue (7x7 biopolymer-species) maps in the same
        membership / enrichment framework as the acid maps. Filter variants
        ALL / INTER / INTRA are built by summing the per-chain-pair
        Domain_Contacts_{Mean, MeanInter, MeanIntraFiltered} CSV files
        to species level. Domain_Contacts_Mean is already CombinedFiltered
        (no bonded ≤5), so no bonded subtraction is applied here.
        n_residue_pairs_by_filter uses the ordered-position-pair convention
        to match the doubled upstream symmetric matrices.
        """
        out_dir = f"{self.analysis_root}/FIGURES/RESIDUE_CONTACT_MAPS"
        os.makedirs(out_dir, exist_ok=True)

        conds = self._available_conditions()
        all_biopolnum_paths = {
            "SG":   self._analysis_file("ANALYSIS_SG_AVE",   "BioPolNum_sg_X.csv"),
            "DSM":  self._analysis_file("ANALYSIS_DSM_AVE",  "BioPolNum_DSM.csv"),
            "NDSM": self._analysis_file("ANALYSIS_NDSM_AVE", "BioPolNum_NDSM.csv"),
        }
        biopolnum = {c: all_biopolnum_paths[c] for c in conds}
        n_chains = {c: CN.n_chains_from_biopolnum(np.loadtxt(p, delimiter=",", dtype=float)) for c, p in biopolnum.items()}

        SPECIES_BIONAME = {"G3BP1": "ProteinG3BP1", "PABP1": "ProteinPABP1",
                           "TTP": "ProteinTTP", "TIA1": "ProteinTIA1",
                           "TDP43": "ProteinTDP43", "FUS": "ProteinFUS", "RNA": "RNA"}

        def derive_residue_C(category, domain_prefix):
            """Sum per-chain-pair Domain_Contacts files into a symmetric 7x7
            species-level matrix in T_AB-per-cell convention (each unordered
            pair-type appears once on the diagonal, mirrored off-diagonal).
            """
            cat_dir = self._analysis_dir(f"ANALYSIS_{category}_AVE")
            cat_tag = "sg_X" if category == "SG" else category
            n_sp = len(CN.SPECIES_ORDER)
            M = np.zeros((n_sp, n_sp), dtype=float)
            for i, A in enumerate(CN.SPECIES_ORDER):
                for j in range(i, n_sp):
                    B = CN.SPECIES_ORDER[j]
                    if domain_prefix.endswith("IntraFiltered") and A != B:
                        continue
                    bio_a, bio_b = SPECIES_BIONAME[A], SPECIES_BIONAME[B]
                    cands = [
                        os.path.join(cat_dir, f"{domain_prefix}_{cat_tag}_{bio_a}_{bio_b}.csv"),
                        os.path.join(cat_dir, f"{domain_prefix}_{cat_tag}_{bio_b}_{bio_a}.csv"),
                    ]
                    path = next((p for p in cands if os.path.exists(p)), None)
                    if path is None:
                        continue
                    try:
                        dm = np.loadtxt(path, delimiter=",")
                    except Exception:
                        continue
                    L_a, L_b = CN.SPECIES_LENGTH[A], CN.SPECIES_LENGTH[B]
                    if dm.shape != (L_a, L_b):
                        dm = dm.T
                        if dm.shape != (L_a, L_b):
                            continue
                    s = float(dm.sum())
                    M[i, j] += s
                    if i != j:
                        M[j, i] += s
            return M

        def derive_residue_SEM(category, sem_prefix):
            """Sum per-pair SEMs in quadrature into a 7x7 species-level SEM matrix.
            For each unordered species pair (i, j), SEM_AB = sqrt(sum(SEM_ij^2))
            over all cells in all matching per-pair SEM files."""
            cat_dir = self._analysis_dir(f"ANALYSIS_{category}_AVE")
            cat_tag = "sg_X" if category == "SG" else category
            n_sp = len(CN.SPECIES_ORDER)
            M = np.zeros((n_sp, n_sp), dtype=float)
            for i, A in enumerate(CN.SPECIES_ORDER):
                for j in range(i, n_sp):
                    B = CN.SPECIES_ORDER[j]
                    if sem_prefix.endswith("IntraFiltered_") and A != B:
                        continue
                    bio_a, bio_b = SPECIES_BIONAME[A], SPECIES_BIONAME[B]
                    cands = [
                        os.path.join(cat_dir, f"{sem_prefix}{cat_tag}_{bio_a}_{bio_b}.csv"),
                        os.path.join(cat_dir, f"{sem_prefix}{cat_tag}_{bio_b}_{bio_a}.csv"),
                    ]
                    path = next((p for p in cands if os.path.exists(p)), None)
                    if path is None:
                        continue
                    try:
                        ds = np.loadtxt(path, delimiter=",")
                    except Exception:
                        continue
                    L_a, L_b = CN.SPECIES_LENGTH[A], CN.SPECIES_LENGTH[B]
                    if ds.shape != (L_a, L_b):
                        ds = ds.T
                        if ds.shape != (L_a, L_b):
                            continue
                    # Quadrature sum (variance addition over cells)
                    ds_sq = np.where(np.isfinite(ds), ds ** 2, 0.0)
                    s2 = float(np.sum(ds_sq))
                    M[i, j] += s2
                    if i != j:
                        M[j, i] += s2
            return np.sqrt(M)

        # NB: Domain_Contacts_Mean is already CombinedFiltered upstream
        # (Mean == MeanInter + MeanIntraFiltered), so summing it to the
        # species level already excludes bonded ≤5 contacts. No further
        # bonded subtraction needed.

        VARIANTS = [
            ("ALL",   "Domain_Contacts_Mean",            "Domain_Contacts_SEMCombinedFiltered_", "all"),
            ("INTER", "Domain_Contacts_MeanInter",       "Domain_Contacts_SEMInter_",            "inter"),
            ("INTRA", "Domain_Contacts_MeanIntraFiltered", "Domain_Contacts_SEMIntraFiltered_",  "intra"),
        ]
        for v_suffix, domain_prefix, sem_prefix, denom_variant in VARIANTS:
            C = {cond: derive_residue_C(cond, domain_prefix) for cond in conds}
            # Propagate per-pair SEMs into a 7x7 species-level SEM matrix
            SEM_C = {cond: derive_residue_SEM(cond, sem_prefix) for cond in conds}

            N_pos = {cond: CN.n_residue_pairs_by_filter(n_chains[cond], denom_variant)
                     for cond in conds}
            # Chain-pair null (treats chain length as intrinsic, normalises by
            # chain-pair multiplicity only — the right null for species×species
            # interaction propensity).
            N_chain = {cond: CN.n_chain_pairs_by_filter(n_chains[cond], denom_variant)
                       for cond in conds}

            f_maps = {c: CN.contact_membership(C[c]) for c in C}
            lnE_maps = {c: CN.log_enrichment(C[c], N_pos[c]) for c in C}
            E_maps = {c: CN.contact_enrichment(C[c], N_pos[c]) for c in C}
            # Chain-pair-normalised counterparts
            lnEc_maps = {c: CN.log_enrichment(C[c], N_chain[c]) for c in C}
            Ec_maps = {c: CN.contact_enrichment(C[c], N_chain[c]) for c in C}
            # SEM[ln X] ≈ SEM[X] / |X|; falls back to Poisson sqrt(C)/C if SEM_C invalid.
            sem_log_maps = {}
            for c in C:
                Cc = np.asarray(C[c], dtype=float)
                sem_c = np.asarray(SEM_C[c], dtype=float)
                if not np.any(np.isfinite(sem_c) & (sem_c > 0)):
                    sem_c = np.sqrt(np.maximum(np.abs(Cc), 0.0))
                with np.errstate(divide='ignore', invalid='ignore'):
                    sem_log_maps[c] = np.where((np.abs(Cc) > 0) & (sem_c > 0), sem_c / np.abs(Cc), np.nan)
            f_vmax = self._shared_vmax(*[f_maps[c] for c in conds])
            f_max = max((np.nanmax(f_maps[c]) if np.any(np.isfinite(f_maps[c])) else 0.0) for c in conds)

            for cond in conds:
                np.savetxt(f"{out_dir}/Residue_{cond}_CONTACT_MEMBERSHIP_{v_suffix}_HeatMap.csv",
                           f_maps[cond], delimiter=",")
                np.savetxt(f"{out_dir}/Residue_{cond}_CONTACT_ENRICHMENT_LOG_{v_suffix}_HeatMap.csv",
                           lnE_maps[cond], delimiter=",")
                fig = self.plot_res_cpm(f_maps[cond], col="Reds", min=0.0, max=f_vmax, mid=f_vmax / 2.0)
                plt.savefig(f"{out_dir}/Residue_{cond}_CONTACT_MEMBERSHIP_{v_suffix}_HeatMap.png", format="png", dpi=400)
                plt.close(fig)

                neglnf = np.full_like(f_maps[cond], np.nan)
                m = np.isfinite(f_maps[cond]) & (f_maps[cond] > 0) & (f_max > 0)
                neglnf[m] = -np.log(f_maps[cond][m] / f_max)
                nlnf_vmax = float(np.nanmax(neglnf)) if np.any(np.isfinite(neglnf)) else 1.0
                fig = self.plot_res_cpm(neglnf, col="Reds_r", min=0.0, max=nlnf_vmax, mid=nlnf_vmax / 2.0)
                plt.savefig(f"{out_dir}/Residue_{cond}_CONTACT_MEMBERSHIP_LOG_{v_suffix}_HeatMap.png", format="png", dpi=400)
                plt.close(fig)

                E_panel_max = max(2.0, float(np.nanmax(E_maps[cond])) if np.any(np.isfinite(E_maps[cond])) else 2.0)
                # Linear enrichment E = f/q is strictly non-negative (a magnitude),
                # so it uses a sequential Reds ramp. The diverging coolwarm map is
                # reserved for signed quantities (the LOG/ZSCORE/DIFF maps below).
                fig = self.plot_res_cpm(E_maps[cond], col="Reds", min=0.0, max=E_panel_max, mid=1.0)
                plt.savefig(f"{out_dir}/Residue_{cond}_CONTACT_ENRICHMENT_{v_suffix}_HeatMap.png", format="png", dpi=400)
                plt.close(fig)

                lnE = lnE_maps[cond]
                # Fixed colorbar cap (see acid generator for rationale).
                lnE_panel = 1.5
                fig = self.plot_res_cpm(lnE, col="coolwarm", min=-lnE_panel, max=lnE_panel, mid=0.0)
                plt.savefig(f"{out_dir}/Residue_{cond}_CONTACT_ENRICHMENT_LOG_{v_suffix}_HeatMap.png", format="png", dpi=400)
                plt.close(fig)

                # --- CHAIN-pair-normalised enrichment (treats chain length as
                # intrinsic biology, asks 'do chains pair up preferentially').
                # Same colourbar conventions as the residue-pair enrichment so
                # the two panels are visually directly comparable.
                lnEc = lnEc_maps[cond]
                np.savetxt(f"{out_dir}/Residue_{cond}_CONTACT_CHAIN_ENRICHMENT_LOG_{v_suffix}_HeatMap.csv",
                           lnEc, delimiter=",")
                fig = self.plot_res_cpm(lnEc, col="coolwarm", min=-lnE_panel, max=lnE_panel, mid=0.0)
                plt.savefig(f"{out_dir}/Residue_{cond}_CONTACT_CHAIN_ENRICHMENT_LOG_{v_suffix}_HeatMap.png", format="png", dpi=400)
                plt.close(fig)
                Ec = Ec_maps[cond]
                Ec_panel_max = max(2.0, float(np.nanmax(Ec)) if np.any(np.isfinite(Ec)) else 2.0)
                # Linear chain-pair enrichment is non-negative -> sequential Reds.
                fig = self.plot_res_cpm(Ec, col="Reds", min=0.0, max=Ec_panel_max, mid=1.0)
                plt.savefig(f"{out_dir}/Residue_{cond}_CONTACT_CHAIN_ENRICHMENT_{v_suffix}_HeatMap.png", format="png", dpi=400)
                plt.close(fig)
                # CHAIN_ENRICHMENT ZSCORE companion (same SEM as residue-pair
                # version since the numerator C is identical).
                z_lnEc = np.full_like(lnEc, np.nan, dtype=float)
                m_zc = np.isfinite(lnEc) & np.isfinite(sem_log_maps[cond]) & (sem_log_maps[cond] > 0)
                z_lnEc[m_zc] = lnEc[m_zc] / sem_log_maps[cond][m_zc]
                np.savetxt(f"{out_dir}/Residue_{cond}_CONTACT_CHAIN_ENRICHMENT_LOG_{v_suffix}_ZSCORE_HeatMap.csv",
                           z_lnEc, delimiter=",")
                fig = self.plot_res_cpm(z_lnEc, col="coolwarm", min=-5.0, max=5.0, mid=0.0)
                plt.savefig(f"{out_dir}/Residue_{cond}_CONTACT_CHAIN_ENRICHMENT_LOG_{v_suffix}_ZSCORE_HeatMap.png", format="png", dpi=400)
                plt.close(fig)

                # --- ZSCORE companion maps (additive; significance test) ---
                # z = ln(X) / SEM[ln X];  SEM[ln X] ≈ SEM[X] / |X|. Saturate at ±5σ.
                z_lnE = np.full_like(lnE, np.nan, dtype=float)
                sl = sem_log_maps[cond]
                m_z = np.isfinite(lnE) & np.isfinite(sl) & (sl > 0)
                z_lnE[m_z] = lnE[m_z] / sl[m_z]
                np.savetxt(f"{out_dir}/Residue_{cond}_CONTACT_ENRICHMENT_LOG_{v_suffix}_ZSCORE_HeatMap.csv",
                           z_lnE, delimiter=",")
                fig = self.plot_res_cpm(z_lnE, col="coolwarm", min=-5.0, max=5.0, mid=0.0)
                plt.savefig(f"{out_dir}/Residue_{cond}_CONTACT_ENRICHMENT_LOG_{v_suffix}_ZSCORE_HeatMap.png", format="png", dpi=400)
                plt.close(fig)
                z_lnf = np.full_like(neglnf, np.nan, dtype=float)
                m_zf = np.isfinite(neglnf) & np.isfinite(sl) & (sl > 0)
                z_lnf[m_zf] = neglnf[m_zf] / sl[m_zf]
                np.savetxt(f"{out_dir}/Residue_{cond}_CONTACT_MEMBERSHIP_LOG_{v_suffix}_ZSCORE_HeatMap.csv",
                           z_lnf, delimiter=",")
                fig = self.plot_res_cpm(z_lnf, col="coolwarm", min=-5.0, max=5.0, mid=0.0)
                plt.savefig(f"{out_dir}/Residue_{cond}_CONTACT_MEMBERSHIP_LOG_{v_suffix}_ZSCORE_HeatMap.png", format="png", dpi=400)
                plt.close(fig)

            # DSM vs NDSM DIFF panels — only when both treatments are present
            if {"DSM", "NDSM"}.issubset(conds):
                df_diff = f_maps["DSM"] - f_maps["NDSM"]
                df_vmax = self._sym_vmax_arr(df_diff)
                fig = self.plot_res_cpm(df_diff, col="coolwarm", min=-df_vmax, max=df_vmax, mid=0.0)
                plt.savefig(f"{out_dir}/Residue_DIFF_CONTACT_MEMBERSHIP_{v_suffix}_HeatMap.png", format="png", dpi=400)
                plt.close(fig)

                dlnE = lnE_maps["DSM"] - lnE_maps["NDSM"]
                dlnE_vmax = self._sym_vmax_arr(dlnE)
                fig = self.plot_res_cpm(dlnE, col="coolwarm", min=-dlnE_vmax, max=dlnE_vmax, mid=0.0)
                plt.savefig(f"{out_dir}/Residue_DIFF_CONTACT_ENRICHMENT_LOG_{v_suffix}_HeatMap.png", format="png", dpi=400)
                plt.close(fig)

                # Chain-pair-normalised DIFF (treats chain length as intrinsic)
                dlnEc = lnEc_maps["DSM"] - lnEc_maps["NDSM"]
                dlnEc_vmax = self._sym_vmax_arr(dlnEc)
                np.savetxt(f"{out_dir}/Residue_DIFF_CONTACT_CHAIN_ENRICHMENT_LOG_{v_suffix}_HeatMap.csv",
                           dlnEc, delimiter=",")
                fig = self.plot_res_cpm(dlnEc, col="coolwarm", min=-dlnEc_vmax, max=dlnEc_vmax, mid=0.0)
                plt.savefig(f"{out_dir}/Residue_DIFF_CONTACT_CHAIN_ENRICHMENT_LOG_{v_suffix}_HeatMap.png", format="png", dpi=400)
                plt.close(fig)

                # --- DIFF ZSCORE: significance of DSM vs NDSM at each species-pair cell ---
                sem_diff = np.sqrt(np.nan_to_num(sem_log_maps["DSM"], nan=0.0) ** 2 +
                                   np.nan_to_num(sem_log_maps["NDSM"], nan=0.0) ** 2)
                z_dlnE = np.full_like(dlnE, np.nan, dtype=float)
                m_zd = np.isfinite(dlnE) & np.isfinite(sem_diff) & (sem_diff > 0)
                z_dlnE[m_zd] = dlnE[m_zd] / sem_diff[m_zd]
                np.savetxt(f"{out_dir}/Residue_DIFF_CONTACT_ENRICHMENT_LOG_{v_suffix}_ZSCORE_HeatMap.csv",
                           z_dlnE, delimiter=",")
                fig = self.plot_res_cpm(z_dlnE, col="coolwarm", min=-5.0, max=5.0, mid=0.0)
                plt.savefig(f"{out_dir}/Residue_DIFF_CONTACT_ENRICHMENT_LOG_{v_suffix}_ZSCORE_HeatMap.png", format="png", dpi=400)
                plt.close(fig)

                # CHAIN_ENRICHMENT DIFF ZSCORE (uses the same SEM as ENRICHMENT
                # since the numerator C is the same; only the null model differs).
                z_dlnEc = np.full_like(dlnEc, np.nan, dtype=float)
                m_zdc = np.isfinite(dlnEc) & np.isfinite(sem_diff) & (sem_diff > 0)
                z_dlnEc[m_zdc] = dlnEc[m_zdc] / sem_diff[m_zdc]
                np.savetxt(f"{out_dir}/Residue_DIFF_CONTACT_CHAIN_ENRICHMENT_LOG_{v_suffix}_ZSCORE_HeatMap.csv",
                           z_dlnEc, delimiter=",")
                fig = self.plot_res_cpm(z_dlnEc, col="coolwarm", min=-5.0, max=5.0, mid=0.0)
                plt.savefig(f"{out_dir}/Residue_DIFF_CONTACT_CHAIN_ENRICHMENT_LOG_{v_suffix}_ZSCORE_HeatMap.png", format="png", dpi=400)
                plt.close(fig)
                # MEMBERSHIP_LOG DIFF: use log-ratio of f
                with np.errstate(divide='ignore', invalid='ignore'):
                    lnratio_f = np.where((f_maps["DSM"] > 0) & (f_maps["NDSM"] > 0),
                                         np.log(f_maps["DSM"] / f_maps["NDSM"]), np.nan)
                z_dlnf = np.full_like(lnratio_f, np.nan, dtype=float)
                m_zdf = np.isfinite(lnratio_f) & np.isfinite(sem_diff) & (sem_diff > 0)
                z_dlnf[m_zdf] = lnratio_f[m_zdf] / sem_diff[m_zdf]
                np.savetxt(f"{out_dir}/Residue_DIFF_CONTACT_MEMBERSHIP_LOG_{v_suffix}_ZSCORE_HeatMap.csv",
                           z_dlnf, delimiter=",")
                fig = self.plot_res_cpm(z_dlnf, col="coolwarm", min=-5.0, max=5.0, mid=0.0)
                plt.savefig(f"{out_dir}/Residue_DIFF_CONTACT_MEMBERSHIP_LOG_{v_suffix}_ZSCORE_HeatMap.png", format="png", dpi=400)
                plt.close(fig)

            # Per-treatment minus-SG panels — only when SG is present
            if "SG" in conds:
                for treat in [c for c in conds if c != "SG"]:
                    v = self._sym_vmax_arr(f_maps[treat] - f_maps["SG"])
                    fig = self.plot_res_cpm(f_maps[treat] - f_maps["SG"], col="coolwarm", min=-v, max=v, mid=0.0)
                    plt.savefig(f"{out_dir}/Residue_{treat}_minus_SG_CONTACT_MEMBERSHIP_{v_suffix}_HeatMap.png", format="png", dpi=400)
                    plt.close(fig)
                    v2 = self._sym_vmax_arr(lnE_maps[treat] - lnE_maps["SG"])
                    fig = self.plot_res_cpm(lnE_maps[treat] - lnE_maps["SG"], col="coolwarm", min=-v2, max=v2, mid=0.0)
                    plt.savefig(f"{out_dir}/Residue_{treat}_minus_SG_CONTACT_ENRICHMENT_LOG_{v_suffix}_HeatMap.png", format="png", dpi=400)
                    plt.close(fig)
                    v3 = self._sym_vmax_arr(lnEc_maps[treat] - lnEc_maps["SG"])
                    fig = self.plot_res_cpm(lnEc_maps[treat] - lnEc_maps["SG"], col="coolwarm", min=-v3, max=v3, mid=0.0)
                    plt.savefig(f"{out_dir}/Residue_{treat}_minus_SG_CONTACT_CHAIN_ENRICHMENT_LOG_{v_suffix}_HeatMap.png", format="png", dpi=400)
                    plt.close(fig)

    def gen_acid_cpms_membership_enrichment(self):
        """Two-panel acid (24x24) maps: contact membership f = C / sum(C)
        and contact enrichment E = f / q, where q = N_possible / sum(N_possible).
        Both numerator C and denominator N_possible use the same filter
        (ALL = inter + intra-non-bonded, INTER, INTRA) so the enrichment
        is a clean odds-ratio against random mixing at the cluster
        composition.

        Output filenames (out_dir = FIGURES/ACID_CONTACT_MAPS):
          Acid_{cond}_CONTACT_MEMBERSHIP_{filter}_HeatMap.png        (Reds, linear f)
          Acid_{cond}_CONTACT_MEMBERSHIP_LOG_{filter}_HeatMap.png    (Reds_r, -ln(f/f_max))
          Acid_{cond}_CONTACT_ENRICHMENT_{filter}_HeatMap.png        (coolwarm @1, E linear)
          Acid_{cond}_CONTACT_ENRICHMENT_LOG_{filter}_HeatMap.png    (coolwarm @0, ln(E))
          Acid_DIFF_CONTACT_MEMBERSHIP_{filter}_HeatMap.png          (coolwarm @0, Df)
          Acid_DIFF_CONTACT_ENRICHMENT_LOG_{filter}_HeatMap.png      (coolwarm @0, Dln(E))
        for cond in {SG, DSM, NDSM} and filter in {ALL, INTER, INTRA}.
        """
        import re
        out_dir = f"{self.analysis_root}/FIGURES/ACID_CONTACT_MAPS"
        os.makedirs(out_dir, exist_ok=True)

        cm_norm_dir = self._find_cm_norm_with_sequences()
        comp = CN.load_all_species_composition(cm_norm_dir)
        types_per_species = CN.load_per_position_types(cm_norm_dir)

        SPECIES_BIONAME = {"G3BP1": "ProteinG3BP1", "PABP1": "ProteinPABP1",
                           "TTP": "ProteinTTP", "TIA1": "ProteinTIA1",
                           "TDP43": "ProteinTDP43", "FUS": "ProteinFUS", "RNA": "RNA"}

        def one_hot(types, n_types=CN.N_ATOM_TYPES):
            L = len(types); X = np.zeros((n_types, L), dtype=float)
            for i, t in enumerate(types):
                if t is not None:
                    X[t - 1, i] = 1.0
            return X
        X_species = {sp: one_hot(types_per_species[sp]) for sp in CN.SPECIES_ORDER}

        def derive_C(category, domain_prefix):
            """Project the per-species-pair domain maps onto a 24x24 atom-type contact matrix."""
            cat_dir = self._analysis_dir(f"ANALYSIS_{category}_AVE")
            cat_tag = "sg_X" if category == "SG" else category
            acid = np.zeros((CN.N_ATOM_TYPES, CN.N_ATOM_TYPES))
            for i, A in enumerate(CN.SPECIES_ORDER):
                for j in range(i, 7):
                    B = CN.SPECIES_ORDER[j]
                    if domain_prefix.endswith("IntraFiltered") and A != B:
                        continue
                    bio_a, bio_b = SPECIES_BIONAME[A], SPECIES_BIONAME[B]
                    cands = [
                        os.path.join(cat_dir, f"{domain_prefix}_{cat_tag}_{bio_a}_{bio_b}.csv"),
                        os.path.join(cat_dir, f"{domain_prefix}_{cat_tag}_{bio_b}_{bio_a}.csv"),
                    ]
                    path = next((p for p in cands if os.path.exists(p)), None)
                    if path is None:
                        continue
                    try:
                        M = np.loadtxt(path, delimiter=",")
                    except Exception:
                        continue
                    L_a, L_b = CN.SPECIES_LENGTH[A], CN.SPECIES_LENGTH[B]
                    if M.shape != (L_a, L_b):
                        M = M.T
                        if M.shape != (L_a, L_b):
                            continue
                    contrib = X_species[A] @ M @ X_species[B].T
                    if A == B:
                        acid += contrib
                    else:
                        acid += contrib + contrib.T
            return acid

        def derive_SEM(category, sem_prefix):
            """Project per-pair SEMs onto the 24x24 acid matrix via variance
            propagation: var_acid = X_A @ SEM^2 @ X_B^T (X_species are one-hot
            so X**2 == X). Returns SEM matrix."""
            cat_dir = self._analysis_dir(f"ANALYSIS_{category}_AVE")
            cat_tag = "sg_X" if category == "SG" else category
            var_acid = np.zeros((CN.N_ATOM_TYPES, CN.N_ATOM_TYPES))
            for i, A in enumerate(CN.SPECIES_ORDER):
                for j in range(i, 7):
                    B = CN.SPECIES_ORDER[j]
                    if sem_prefix.endswith("IntraFiltered_") and A != B:
                        continue
                    bio_a, bio_b = SPECIES_BIONAME[A], SPECIES_BIONAME[B]
                    cands = [
                        os.path.join(cat_dir, f"{sem_prefix}{cat_tag}_{bio_a}_{bio_b}.csv"),
                        os.path.join(cat_dir, f"{sem_prefix}{cat_tag}_{bio_b}_{bio_a}.csv"),
                    ]
                    path = next((p for p in cands if os.path.exists(p)), None)
                    if path is None:
                        continue
                    try:
                        S = np.loadtxt(path, delimiter=",")
                    except Exception:
                        continue
                    L_a, L_b = CN.SPECIES_LENGTH[A], CN.SPECIES_LENGTH[B]
                    if S.shape != (L_a, L_b):
                        S = S.T
                        if S.shape != (L_a, L_b):
                            continue
                    S2 = np.where(np.isfinite(S), S ** 2, 0.0)
                    var_contrib = X_species[A] @ S2 @ X_species[B].T
                    if A == B:
                        var_acid += var_contrib
                    else:
                        var_acid += var_contrib + var_contrib.T
            return np.sqrt(var_acid)

        conds = self._available_conditions()
        all_biopolnum_files = {
            "SG":   self._analysis_file("ANALYSIS_SG_AVE",   "BioPolNum_sg_X.csv"),
            "DSM":  self._analysis_file("ANALYSIS_DSM_AVE",  "BioPolNum_DSM.csv"),
            "NDSM": self._analysis_file("ANALYSIS_NDSM_AVE", "BioPolNum_NDSM.csv"),
        }
        biopolnum_files = {c: all_biopolnum_files[c] for c in conds}
        n_chains = {}
        for cond, path in biopolnum_files.items():
            bio = np.loadtxt(path, delimiter=",", dtype=float)
            n_chains[cond] = CN.n_chains_from_biopolnum(bio)

        VARIANTS = [
            ("ALL",   "Domain_Contacts_Mean",            "Domain_Contacts_SEMCombinedFiltered_", "all"),
            ("INTER", "Domain_Contacts_MeanInter",       "Domain_Contacts_SEMInter_",            "inter"),
            ("INTRA", "Domain_Contacts_MeanIntraFiltered", "Domain_Contacts_SEMIntraFiltered_",  "intra"),
        ]

        # NB: Domain_Contacts_Mean is already CombinedFiltered upstream
        # (Mean == MeanInter + MeanIntraFiltered, no bonded ≤5 included),
        # so the projected acid matrix already excludes bonded contacts
        # for every filter. No bonded subtraction is needed; doing it
        # would double-subtract ~17-20% of contacts.

        for v_suffix, domain_prefix, sem_prefix, denom_variant in VARIANTS:
            C = {cond: derive_C(cond, domain_prefix) for cond in conds}
            SEM_C = {cond: derive_SEM(cond, sem_prefix) for cond in conds}
            N_pos = {cond: CN.n_atom_pairs_by_filter(n_chains[cond], comp, types_per_species, denom_variant)
                     for cond in conds}

            f_maps = {cond: CN.contact_membership(C[cond]) for cond in C}
            E_maps = {cond: CN.contact_enrichment(C[cond], N_pos[cond]) for cond in C}
            lnE_maps = {cond: CN.log_enrichment(C[cond], N_pos[cond]) for cond in C}
            # SEM[ln X] ≈ SEM[X] / |X|; fall back to Poisson sqrt(C)/C if SEM_C invalid.
            sem_log_maps = {}
            for cond in C:
                Cc = np.asarray(C[cond], dtype=float)
                sc = np.asarray(SEM_C[cond], dtype=float)
                if not np.any(np.isfinite(sc) & (sc > 0)):
                    sc = np.sqrt(np.maximum(np.abs(Cc), 0.0))
                with np.errstate(divide='ignore', invalid='ignore'):
                    sem_log_maps[cond] = np.where((np.abs(Cc) > 0) & (sc > 0), sc / np.abs(Cc), np.nan)
            # Fixed cap on the membership colorbar so the few abundance
            # supercells (GLY-GLY in ALL, GLY-A(RNA) in INTER) don't crush
            # the rest of the matrix to near-white. 0.5% saturates the top
            # ~3-5 cells (which are clearly off-scale and flagged by the
            # saturated colour) while making the bulk distribution readable.
            f_vmax = 5e-3
            E_vmax = self._shared_vmax(*[E_maps[c] for c in conds])
            lnE_max = self._shared_vmax(*[np.abs(lnE_maps[c]) for c in conds])
            f_max = max((np.nanmax(f_maps[c]) if np.any(np.isfinite(f_maps[c])) else 0.0) for c in conds)

            for cond in conds:
                # CSV sidecars for audit / downstream re-plotting
                np.savetxt(f"{out_dir}/Acid_{cond}_CONTACT_MEMBERSHIP_{v_suffix}_HeatMap.csv",
                           f_maps[cond], delimiter=",")
                np.savetxt(f"{out_dir}/Acid_{cond}_CONTACT_ENRICHMENT_LOG_{v_suffix}_HeatMap.csv",
                           lnE_maps[cond], delimiter=",")
                fig = self.plot_acid_cpm(f_maps[cond], col="Reds", min=0.0, max=f_vmax, mid=f_vmax / 2.0)
                plt.savefig(f"{out_dir}/Acid_{cond}_CONTACT_MEMBERSHIP_{v_suffix}_HeatMap.png", format="png", dpi=400)
                plt.close(fig)

                neglnf = np.full_like(f_maps[cond], np.nan)
                m = np.isfinite(f_maps[cond]) & (f_maps[cond] > 0) & (f_max > 0)
                neglnf[m] = -np.log(f_maps[cond][m] / f_max)
                nlnf_vmax = float(np.nanmax(neglnf)) if np.any(np.isfinite(neglnf)) else 1.0
                fig = self.plot_acid_cpm(neglnf, col="Reds_r", min=0.0, max=nlnf_vmax, mid=nlnf_vmax / 2.0)
                plt.savefig(f"{out_dir}/Acid_{cond}_CONTACT_MEMBERSHIP_LOG_{v_suffix}_HeatMap.png", format="png", dpi=400)
                plt.close(fig)

                E_panel_max = max(2.0, float(np.nanmax(E_maps[cond])) if np.any(np.isfinite(E_maps[cond])) else 2.0)
                # Linear enrichment is non-negative -> sequential Reds (coolwarm is
                # reserved for the signed LOG/ZSCORE/DIFF maps below).
                fig = self.plot_acid_cpm(E_maps[cond], col="Reds", min=0.0, max=E_panel_max, mid=1.0)
                plt.savefig(f"{out_dir}/Acid_{cond}_CONTACT_ENRICHMENT_{v_suffix}_HeatMap.png", format="png", dpi=400)
                plt.close(fig)

                lnE = lnE_maps[cond]
                # Fixed colorbar cap so the protein-protein block isn't
                # crushed by extreme RNA U/C/G depletion. |ln(E)|=1.5
                # corresponds to E in [0.22x, 4.5x] which is the
                # publishable LLPS dynamic range; cells outside saturate.
                lnE_panel = 1.5
                fig = self.plot_acid_cpm(lnE, col="coolwarm", min=-lnE_panel, max=lnE_panel, mid=0.0)
                plt.savefig(f"{out_dir}/Acid_{cond}_CONTACT_ENRICHMENT_LOG_{v_suffix}_HeatMap.png", format="png", dpi=400)
                plt.close(fig)

                # --- ZSCORE companion maps (additive; significance test) ---
                sl = sem_log_maps[cond]
                z_lnE = np.full_like(lnE, np.nan, dtype=float)
                m_z = np.isfinite(lnE) & np.isfinite(sl) & (sl > 0)
                z_lnE[m_z] = lnE[m_z] / sl[m_z]
                np.savetxt(f"{out_dir}/Acid_{cond}_CONTACT_ENRICHMENT_LOG_{v_suffix}_ZSCORE_HeatMap.csv",
                           z_lnE, delimiter=",")
                fig = self.plot_acid_cpm(z_lnE, col="coolwarm", min=-5.0, max=5.0, mid=0.0)
                plt.savefig(f"{out_dir}/Acid_{cond}_CONTACT_ENRICHMENT_LOG_{v_suffix}_ZSCORE_HeatMap.png", format="png", dpi=400)
                plt.close(fig)
                z_lnf = np.full_like(neglnf, np.nan, dtype=float)
                m_zf = np.isfinite(neglnf) & np.isfinite(sl) & (sl > 0)
                z_lnf[m_zf] = neglnf[m_zf] / sl[m_zf]
                np.savetxt(f"{out_dir}/Acid_{cond}_CONTACT_MEMBERSHIP_LOG_{v_suffix}_ZSCORE_HeatMap.csv",
                           z_lnf, delimiter=",")
                fig = self.plot_acid_cpm(z_lnf, col="coolwarm", min=-5.0, max=5.0, mid=0.0)
                plt.savefig(f"{out_dir}/Acid_{cond}_CONTACT_MEMBERSHIP_LOG_{v_suffix}_ZSCORE_HeatMap.png", format="png", dpi=400)
                plt.close(fig)

            # DIFFs (DSM - NDSM) — the main perturbation comparison
            if {"DSM", "NDSM"}.issubset(conds):
                df_diff = f_maps["DSM"] - f_maps["NDSM"]
                df_vmax = self._sym_vmax_arr(df_diff)
                fig = self.plot_acid_cpm(df_diff, col="coolwarm", min=-df_vmax, max=df_vmax, mid=0.0)
                plt.savefig(f"{out_dir}/Acid_DIFF_CONTACT_MEMBERSHIP_{v_suffix}_HeatMap.png", format="png", dpi=400)
                plt.close(fig)

                dlnE = lnE_maps["DSM"] - lnE_maps["NDSM"]
                dlnE_vmax = self._sym_vmax_arr(dlnE)
                fig = self.plot_acid_cpm(dlnE, col="coolwarm", min=-dlnE_vmax, max=dlnE_vmax, mid=0.0)
                plt.savefig(f"{out_dir}/Acid_DIFF_CONTACT_ENRICHMENT_LOG_{v_suffix}_HeatMap.png", format="png", dpi=400)
                plt.close(fig)

                # --- DIFF ZSCORE (DSM vs NDSM significance) ---
                sem_diff = np.sqrt(np.nan_to_num(sem_log_maps["DSM"], nan=0.0) ** 2 +
                                   np.nan_to_num(sem_log_maps["NDSM"], nan=0.0) ** 2)
                z_dlnE = np.full_like(dlnE, np.nan, dtype=float)
                m_zd = np.isfinite(dlnE) & np.isfinite(sem_diff) & (sem_diff > 0)
                z_dlnE[m_zd] = dlnE[m_zd] / sem_diff[m_zd]
                np.savetxt(f"{out_dir}/Acid_DIFF_CONTACT_ENRICHMENT_LOG_{v_suffix}_ZSCORE_HeatMap.csv",
                           z_dlnE, delimiter=",")
                fig = self.plot_acid_cpm(z_dlnE, col="coolwarm", min=-5.0, max=5.0, mid=0.0)
                plt.savefig(f"{out_dir}/Acid_DIFF_CONTACT_ENRICHMENT_LOG_{v_suffix}_ZSCORE_HeatMap.png", format="png", dpi=400)
                plt.close(fig)
                with np.errstate(divide='ignore', invalid='ignore'):
                    lnratio_f = np.where((f_maps["DSM"] > 0) & (f_maps["NDSM"] > 0),
                                         np.log(f_maps["DSM"] / f_maps["NDSM"]), np.nan)
                z_dlnf = np.full_like(lnratio_f, np.nan, dtype=float)
                m_zdf = np.isfinite(lnratio_f) & np.isfinite(sem_diff) & (sem_diff > 0)
                z_dlnf[m_zdf] = lnratio_f[m_zdf] / sem_diff[m_zdf]
                np.savetxt(f"{out_dir}/Acid_DIFF_CONTACT_MEMBERSHIP_LOG_{v_suffix}_ZSCORE_HeatMap.csv",
                           z_dlnf, delimiter=",")
                fig = self.plot_acid_cpm(z_dlnf, col="coolwarm", min=-5.0, max=5.0, mid=0.0)
                plt.savefig(f"{out_dir}/Acid_DIFF_CONTACT_MEMBERSHIP_LOG_{v_suffix}_ZSCORE_HeatMap.png", format="png", dpi=400)
                plt.close(fig)

            # SG-relative DIFFs for SI completeness
            if "SG" in conds:
                for treat in [c for c in conds if c != "SG"]:
                    df_sg = f_maps[treat] - f_maps["SG"]
                    v = self._sym_vmax_arr(df_sg)
                    fig = self.plot_acid_cpm(df_sg, col="coolwarm", min=-v, max=v, mid=0.0)
                    plt.savefig(f"{out_dir}/Acid_{treat}_minus_SG_CONTACT_MEMBERSHIP_{v_suffix}_HeatMap.png", format="png", dpi=400)
                    plt.close(fig)
                    dlnE_sg = lnE_maps[treat] - lnE_maps["SG"]
                    v2 = self._sym_vmax_arr(dlnE_sg)
                    fig = self.plot_acid_cpm(dlnE_sg, col="coolwarm", min=-v2, max=v2, mid=0.0)
                    plt.savefig(f"{out_dir}/Acid_{treat}_minus_SG_CONTACT_ENRICHMENT_LOG_{v_suffix}_HeatMap.png", format="png", dpi=400)
                    plt.close(fig)

    def plot_rdf(self, folder, sm, protein):
        """Plot radial distribution functions of *protein* against every other species.

        Reads RDF_{sm}.csv from *folder*, selects the columns pairing *protein*
        with each other biopolymer, and overlays them on a single axes saved to
        IMAGES/RDP/RDF_{sm}_{protein}.png.
        """
        df_rdf = pd.read_csv(self._analysis_file(folder, f"RDF_{sm}.csv")).iloc[1:,:]
        dist = df_rdf["Distance"]
        prot_list = ["TDP43","FUS","TIA1","G3BP1","RNA","PABP1","TTP"]
        rdf_list = []
        prot_list.remove(protein)
        labs = []
        for i in prot_list:
            labs.append("{}-{}".format(protein, i))
            for j in list(df_rdf.columns.tolist()):
                if protein in j and i in j:
                    rdf_list.append(df_rdf[j])

        col_pall = sns.color_palette("rocket", n_colors=14)

        sns.set_theme(style="ticks")
        sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
        plt.rc('axes', titlesize=10)  # fontsize of the axes title
        plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
        plt.rc('legend', fontsize=8)  # legend fontsize
        plt.rc('font', size=10)  # controls default text sizes
        plt.rc('axes', linewidth=2)
        fig, ax1 = _make_rdp_fig()

        for i in range(len(rdf_list)):
            ax1.plot(dist, rdf_list[i], label=labs[i], linewidth=2, zorder=1, color=col_pall[2*i], clip_on=True)
        leg = plt.figlegend(loc='upper right', ncol=1, bbox_to_anchor=(0.7, 0, 0.2, 0.85))
        leg.get_frame().set_alpha(0)
        file_name = "{}/IMAGES/RDP/RDF_{}_{}.png".format(self.analysis_root,sm,protein)
        ax1.set_ylabel("")
        ax1.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in',
                        length=4,
                        width=2)
        for spine in ax1.spines.values():
            spine.set_zorder(3)

        plt.savefig(file_name, format="png", dpi=400)


def _warn_missing_dirs(base_path):
    """Print a warning for each missing ANALYSIS_{SG,DSM,NDSM}_AVE input dir under *base_path*."""
    for cat in ["SG", "DSM", "NDSM"]:
        candidate = os.path.join(base_path, f"ANALYSIS_{cat}_AVE")
        if not os.path.exists(candidate):
            print(f"Warning: {candidate} not found — {cat} summary maps will be skipped.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate biopolymer/RDP plots from CLASSIFY outputs (matching SYSTEM_ANALYSIS.py CLI).'
    )
    parser.add_argument('--path', required=True, help='Path to TEMP_XXX directory (e.g., TEMP_300)')
    parser.add_argument('--folder', required=True, help='Output folder prefix (e.g., CLASSIFY)')
    parser.add_argument('--T', type=int, required=True, help='Simulation temperature in Kelvin')
    parser.add_argument('--dt', type=int, required=True, help='Cluster time stride (ns)')
    parser.add_argument('--tmin', type=int, required=True, help='Start of analysis window (ns)')
    parser.add_argument('--tmax', type=int, required=True, help='End of analysis window (ns)')
    parser.add_argument('--plot-only', action='store_true', help='Accepted for pipeline plot-only mode; this script already reads existing averaged outputs')
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"Error: Path {args.path} does not exist.")
        sys.exit(1)

    args.path = os.path.abspath(args.path)
    # Compute analysis root (e.g., TEMP_300/CLASSIFY_300_50_50_2000)
    analysis_root = os.path.join(args.path, f"{args.folder}_{args.T}_{args.dt}_{args.tmin}_{args.tmax}")
    if not os.path.exists(analysis_root):
        print(f"Error: Analysis root {analysis_root} does not exist.")
        sys.exit(1)
    _warn_missing_dirs(analysis_root)
    os.chdir(args.path)

    bio_analysis = biopolymer_analysis(args.path, analysis_root, T=args.T)

    def _run_with_logging(name, fn):
        try:
            print(f"Running {name}")
            fn()
            print(f"{name} completed successfully")
        except Exception as exc:
            print(f"{name} failed: {exc}")

    _run_with_logging("Residue contact CPMs", bio_analysis.gen_residue_cpms)
    _run_with_logging("Acid contact CPMs", bio_analysis.gen_acid_cpms)
    _run_with_logging("Residue P_contact / K_chain CPMs", bio_analysis.gen_residue_cpms_pcontact)
    _run_with_logging("Acid P_contact CPMs", bio_analysis.gen_acid_cpms_pcontact)
    _run_with_logging("Residue C_total CPMs", bio_analysis.gen_residue_cpms_ctotal)
    _run_with_logging("Acid C_total CPMs", bio_analysis.gen_acid_cpms_ctotal)
    _run_with_logging("SM DIFF P_contact / P_membership maps", bio_analysis.gen_sm_diff_pcontact)
    _run_with_logging("Per-residue-pair P biopolymer maps", bio_analysis.gen_residue_cpms_p_residue_pair)
    _run_with_logging("Acid membership/enrichment CPMs", bio_analysis.gen_acid_cpms_membership_enrichment)
    _run_with_logging("Residue membership/enrichment CPMs", bio_analysis.gen_residue_cpms_membership_enrichment)
    _run_with_logging("SM membership/enrichment CPMs", bio_analysis.gen_sm_membership_enrichment)

    # SG block
    try:
        sm = "SG"
        bio_analysis.plot_rdp(sm)
        # Biopolymer
        folder = "ANALYSIS_SG_AVE"
        fitterG3BP1, fitterTDP43, fitterPABP1, fitterFUS, fitterTIA1, fitterTTP, fitterRNA = bio_analysis.gen_biopolymer_fitters(folder, "SG")
        bio_analysis.gen_biopolymer_plots(fitterG3BP1, fitterTDP43, fitterPABP1, fitterFUS, fitterTIA1, fitterTTP, fitterRNA, "SG")
        bio_analysis.gen_biopolymer_plots_with_rna_components(folder, "SG")
        print("SG block completed successfully")
    except Exception as exc:
        print(f"SG block failed: {exc}")

    # DSM block
    try:
        sm = "DSM"
        bio_analysis.plot_rdp(sm)
        # Biopolymer
        folder = "ANALYSIS_DSM_AVE"
        fitterG3BP1, fitterTDP43, fitterPABP1, fitterFUS, fitterTIA1, fitterTTP, fitterRNA = bio_analysis.gen_biopolymer_fitters(folder, "DSM")
        bio_analysis.gen_biopolymer_plots(fitterG3BP1, fitterTDP43, fitterPABP1, fitterFUS, fitterTIA1, fitterTTP, fitterRNA, "DSM")
        bio_analysis.gen_biopolymer_plots_with_rna_components(folder, "DSM")
        print("DSM block completed successfully")
    except Exception as exc:
        print(f"DSM block failed: {exc}")

    # NDSM block
    try:
        sm = "NDSM"
        bio_analysis.plot_rdp(sm)
        # Biopolymer
        folder = "ANALYSIS_NDSM_AVE"
        fitterG3BP1, fitterTDP43, fitterPABP1, fitterFUS, fitterTIA1, fitterTTP, fitterRNA = bio_analysis.gen_biopolymer_fitters(folder, "NDSM")
        bio_analysis.gen_biopolymer_plots(fitterG3BP1, fitterTDP43, fitterPABP1, fitterFUS, fitterTIA1, fitterTTP, fitterRNA, "NDSM")
        bio_analysis.gen_biopolymer_plots_with_rna_components(folder, "NDSM")
        print("NDSM block completed successfully")
    except Exception as exc:
        print(f"NDSM block failed: {exc}")
