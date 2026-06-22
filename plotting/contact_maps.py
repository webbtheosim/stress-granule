"""
Domain/residue contact-map averaging and heatmap rendering (aggregate mode).

Pipeline step 6 of the stress-granule MD analysis pipeline. Runs after the
per-window domain-contact extraction (rcc_analysis.py), time-averages the
per-residue contact matrices for each system, aggregates them across the DSM and
NDSM compound sets, and renders the contact-map heatmaps used in the paper.

Purpose: turn raw per-window domain-contact counts into publication FULL_FULL and
per-pair heatmaps (raw C_total, contact membership/enrichment, and DSM/NDSM/SG
difference variants), with propagated per-cell SEM.

Inputs (read from the analysis run dir): ``ANALYSIS_{SG,DSM,NDSM}/`` containing
    Domain_Contacts_Total_<sm>_<ProteinA>_<ProteinB>_<frame>.csv[.npz]
plus the chain-pair normalization table BioPolNum_<tag>.csv and (optionally)
domains.csv for the colored domain-annotation bars.

Outputs:
    - ANALYSIS_*_AVE/Domain_Contacts_Mean_<TAG>_<ProteinA>_<ProteinB>.csv
      (time-averaged matrices; <TAG> is an SM name or a category name)
    - FIGURES/DOMAIN_CONTACT_MAPS/<CATEGORY>/Domain_Contact_Map_<P1>_<P2>.png
    - FIGURES/DOMAIN_CONTACT_MAPS/<CATEGORY>/Domain_Contact_Map_FULL_FULL.png
    - FIGURES/DOMAIN_CONTACT_MAPS/DIFF/ difference heatmaps + SEM/z-score CSVs

CLI:
    python contact_maps.py --path TEMP_300 --folder CLASSIFY --temp 300 \
        --tmin 50 --dt 50 --tmax 2000 [--no-avg | --plot-only]

``--no-avg``/``--plot-only`` regenerate plots from already-averaged matrices in
ANALYSIS_*_AVE without re-reading the raw per-window contacts.
"""

import sys
import os
# Allow this renderer, when run from the plotting/ folder, to import the shared
# compute/support modules that live in analysis/ (e.g. CONTACT_NORMALIZATION).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "analysis"))
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
from matplotlib.patches import Rectangle
import colorcet as cc
import colorsys
from matplotlib.ticker import FuncFormatter
from typing import Literal, Optional

_FIG_MAX_IN = 3.415
_PLOT_FIG_SIZE = (_FIG_MAX_IN, _FIG_MAX_IN)
_LEFT_MARGIN_IN = 0.62   # space for 8pt 45 deg-rotated species labels on left
_RIGHT_MARGIN_IN = 0.55  # room for cbar tick labels + x10^n exponent
                         # ax_w = 3.415 - 0.62 - (0.55+0.08+0.035+0.13) = 2.00 in
_TOP_MARGIN_IN = 0.42    # top_aux = 0.42 + 0.08 (bar) = 0.50 in -> matches bottom for vertical centering
_BOTTOM_MARGIN_IN = 0.50 # 8pt 45 deg-rotated species labels at bottom
_DOMAIN_BAR_IN = 0.08
_CBAR_WIDTH_IN = 0.13    # 2x thicker than 0.065
_CBAR_PAD_IN = 0.035
_AX_RECT = [0.12, 0.12, 0.68, 0.68]  # fallback for non-map line plots
_PAIR_TICK_COUNT = 8
_DOMAIN_FULL_TICK_FONTSIZE = 8
_DOMAIN_PAIR_TICK_FONTSIZE = 8
_DOMAIN_CBAR_FONTSIZE = 8


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
_DOMAIN_BAR_PAD_TOP = 0.0  # touching the plot border
_DOMAIN_BAR_PAD_BOTTOM = 0.0  # touching the plot border
_DOMAIN_BAR_FRAC = 0.08  # legacy fallback; contact-map bars use _DOMAIN_BAR_IN
_BAR_PAD = 0.0  # touching the plot border
_BAR_FRAC = 0.08

# -----------------------
# Helper configuration
# -----------------------

# Shared biopolymer order and lengths (must match file generation)
BIOPOLYMER_LIST = [
    "ProteinFUS", "ProteinTIA1", "ProteinTDP43",
    "ProteinG3BP1", "ProteinPABP1", "ProteinTTP", "RNA"
]
BIOPOLYMER_LENGTHS = [526, 386, 414, 466, 636, 326, 840]


def _ensure_dir(path):
    """Create *path* (and parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def _sm_category(sm_name: str) -> str:
    """Classify a small-molecule tag into its category: 'DSM', 'NDSM', or 'SG' (default)."""
    sm = sm_name.lower()
    if sm.startswith("dsm_"):
        return "DSM"
    if sm.startswith("ndsm_"):
        return "NDSM"
    return "SG"  # default


def reshape_to_subarrays(arr: np.ndarray, p: int, q: int) -> np.ndarray:
    """Extract inter-molecular contact submatrices and average across copies.

    Two operating modes depending on input shape:

    1. **Collapsed** (n == p, m == q): The input is already a single LA×LB
       sequence-position map (produced by the current gen_domain_contacts which
       collapses all copies internally).  Returned unchanged.

    2. **Copy-stacked** (n == k*p, m == l*q): The input contains k×l sub-blocks
       of size p×q, one per molecule-pair.  For same-species pairs (p == q),
       only off-diagonal sub-blocks (i ≠ j) are kept — these represent
       *inter*-molecular contacts — and averaged.  For different species all
       sub-blocks are inter-molecular by construction, so all are averaged.
    """
    arr = arr.copy()
    n, m = arr.shape
    # Current RCC domain-contact outputs are already collapsed LA×LB sequence maps.
    # Legacy ARRAY logic expected copy-stacked block matrices and, for same-species
    # pairs, averaged only the off-diagonal sub-blocks. If the input is already a
    # single LA×LB map, returning zeros here wipes the entire same-species block and
    # propagates a blank diagonal into FULL_FULL.
    if n == p and m == q:
        return arr
    if n % p != 0 or m % q != 0:
        raise ValueError(f"Dimensions mismatch: {n} % {p} != 0 or {m} % {q} != 0")

    num_subarrays_n = n // p
    num_subarrays_m = m // q
    subarrays_diff = []

    # Do not zero global diagonals here; we exclude same-molecule sub-blocks below when p == q

    for i in range(num_subarrays_n):
        for j in range(num_subarrays_m):
            sub_array = arr[i * p:(i + 1) * p, j * q:(j + 1) * q]
            if p == q:
                if i != j:  # inter-molecular only
                    subarrays_diff.append(sub_array)
            else:
                # Always inter-molecular for p != q
                subarrays_diff.append(sub_array)

    ave_diff = np.mean(np.array(subarrays_diff), axis=0) if subarrays_diff else np.zeros((p, q))
    return ave_diff


def _time_labels(k: int) -> list[str]:
    """Return candidate filename time tokens for frame *k* (e.g. ['50', '50.0']), de-duplicated."""
    labels = [str(k), f"{float(k):.1f}"]
    out = []
    for label in labels:
        if label not in out:
            out.append(label)
    return out


def _domain_contact_candidates(category_dir: str, sm: str, bio_i: str, bio_j: str, label: str,
                               prefix: str = "Domain_Contacts_Total_") -> list[str]:
    """Return the candidate .csv / .csv.npz paths for one domain-contact pair at one time label."""
    return [
        os.path.join(category_dir, f"{prefix}{sm}_{bio_i}_{bio_j}_{label}.csv"),
        os.path.join(category_dir, f"{prefix}{sm}_{bio_i}_{bio_j}_{label}.csv.npz"),
    ]


def _load_domain_contact_matrix(path_to_use: str) -> np.ndarray:
    """Load a domain-contact matrix from a .csv.npz (key arr_0) or comma-delimited .csv file."""
    if path_to_use.endswith(".npz"):
        with np.load(path_to_use) as data:
            return data["arr_0"]
    return np.loadtxt(path_to_use, delimiter=",", dtype=float)


def _raw_prefixes_for_pair(raw_prefix: str, bio_i: str, bio_j: str) -> list[str]:
    """Prefer exact combined-filtered contacts when available.

    The exact same-species map is computed upstream before sequence positions are
    collapsed: interchain contacts are kept intact, while only intrachain local
    sequence neighbors are excluded. Older Total files remain a fallback.
    """
    if raw_prefix == "Domain_Contacts_Total_":
        return ["Domain_Contacts_TotalCombinedFiltered_", raw_prefix]
    return [raw_prefix]


def _contact_map_layout(shape: tuple[int, int], *, square: bool,
                        has_x_bar: bool = True, has_y_bar: bool = True,
                        show_colorbar: bool = True) -> tuple[tuple[float, float], list[float]]:
    """Return figure size and axes rect with fixed-size side elements.

    The heatmap itself preserves the matrix aspect ratio. Square/self/FULL maps
    are exactly 3.2 x 3.2 in; rectangular pair maps cap the longer figure axis at
    3.2 in and shrink the other axis accordingly. Domain bars and colorbars use
    physical inch widths so non-square maps do not get visually thicker bars.
    """
    nrows, ncols = shape
    aspect = (float(ncols) / float(nrows)) if nrows > 0 else 1.0
    if not np.isfinite(aspect) or aspect <= 0:
        aspect = 1.0

    right_aux = _RIGHT_MARGIN_IN
    if has_y_bar:
        right_aux += _DOMAIN_BAR_IN
    if show_colorbar:
        right_aux += _CBAR_PAD_IN + _CBAR_WIDTH_IN

    bottom_aux = _BOTTOM_MARGIN_IN
    left = _LEFT_MARGIN_IN
    top = _TOP_MARGIN_IN + (_DOMAIN_BAR_IN if has_x_bar else 0.0)

    if square:
        # Width is the limiting dimension (left+right_aux > top+bottom_aux after margin reduction).
        # Compute fig_h independently so the figure is not unnecessarily tall.
        fig_w = _FIG_MAX_IN
        ax_w = ax_h = max(0.5, fig_w - left - right_aux)
        fig_h = top + ax_h + bottom_aux
    elif aspect >= 1.0:
        fig_w = _FIG_MAX_IN
        max_ax_w = fig_w - left - right_aux
        ax_w = max(0.5, max_ax_w)
        ax_h = ax_w / aspect
        fig_h = min(_FIG_MAX_IN, max(1.0, top + ax_h + bottom_aux))
        if top + ax_h + bottom_aux > fig_h:
            ax_h = max(0.5, fig_h - top - bottom_aux)
            ax_w = ax_h * aspect
    else:
        fig_h = _FIG_MAX_IN
        max_ax_h = fig_h - top - bottom_aux
        ax_h = max(0.5, max_ax_h)
        ax_w = ax_h * aspect
        fig_w = min(_FIG_MAX_IN, max(1.0, left + ax_w + right_aux))
        if left + ax_w + right_aux > fig_w:
            ax_w = max(0.5, fig_w - left - right_aux)
            ax_h = ax_w / aspect

    ax_rect = [left / fig_w, bottom_aux / fig_h, ax_w / fig_w, ax_h / fig_h]
    return (fig_w, fig_h), ax_rect


def _mean_prefixes_for_pair(bio_i: str, bio_j: str) -> list[str]:
    """Prefer pre-averaged exact combined-filtered maps in plot-only mode."""
    return ["Domain_Contacts_MeanCombinedFiltered_", "Domain_Contacts_Mean_"]


def _cbar_left_after_right_bar(ax, has_right_bar: bool) -> float:
    """Return the colorbar left edge (figure fraction) just right of *ax*, accounting for a right domain bar."""
    pos = ax.get_position()
    fig_w = ax.figure.get_figwidth()
    right_bar_width = (_DOMAIN_BAR_IN / fig_w) if has_right_bar else 0.0
    return pos.x1 + right_bar_width + (_CBAR_PAD_IN / fig_w)


def _create_bar_axis(ax, orientation: str, *, size_frac: float = _BAR_FRAC,
                     pad: float = _BAR_PAD, align: str = 'top'):
    """Add and return a thin tick-free axes flush to one edge of *ax* for a domain color bar.

    *orientation* selects the side ('top'/'bottom'/'left'/'right'); the bar uses a
    fixed physical thickness (_DOMAIN_BAR_IN) and shares the data limits of the
    matching *ax* axis. Raises ValueError for an unsupported orientation.
    """
    fig = ax.figure
    pos = ax.get_position()
    if orientation == 'top':
        height = _DOMAIN_BAR_IN / fig.get_figheight()
        bottom = pos.y1 + pad if align == 'top' else pos.y0 - pad - height
        max_top = 0.99
        min_bottom = 0.01
        if bottom + height > max_top:
            bottom = max_top - height
        if bottom < min_bottom:
            bottom = min_bottom
        bar_ax = fig.add_axes([pos.x0, bottom, pos.width, height])
        bar_ax.set_xlim(ax.get_xlim())
        bar_ax.set_ylim(0, 1)
    elif orientation == 'bottom':
        height = _DOMAIN_BAR_IN / fig.get_figheight()
        bottom = pos.y0 - pad - height
        min_bottom = 0.01
        if bottom < min_bottom:
            bottom = min_bottom
        bar_ax = fig.add_axes([pos.x0, bottom, pos.width, height])
        bar_ax.set_xlim(ax.get_xlim())
        bar_ax.set_ylim(0, 1)
    elif orientation == 'right':
        width = _DOMAIN_BAR_IN / fig.get_figwidth()
        left = pos.x1 + pad
        bar_ax = fig.add_axes([left, pos.y0, width, pos.height])
        bar_ax.set_ylim(ax.get_ylim())
        bar_ax.set_xlim(0, 1)
    elif orientation == 'left':
        width = _DOMAIN_BAR_IN / fig.get_figwidth()
        left = pos.x0 - pad - width
        min_left = 0.01
        if left < min_left:
            left = min_left
        bar_ax = fig.add_axes([left, pos.y0, width, pos.height])
        bar_ax.set_ylim(ax.get_ylim())
        bar_ax.set_xlim(0, 1)
    else:
        raise ValueError("Unsupported orientation for bar axis")
    bar_ax.set_facecolor('none')
    # Turn off ticks but keep spines controllable
    bar_ax.tick_params(left=False, right=False, top=False, bottom=False,
                       labelleft=False, labelright=False, labeltop=False, labelbottom=False)
    return bar_ax


def _add_domain_bar(fig, ax, bars, boundaries=None, orientation='top', align='top', spine_linewidth=2, flip_axis=False):
    """Draw the colored domain-annotation bar (rectangles + boundary lines) alongside *ax*.

    *bars* is a list of (start0, end0, color) spans in residue-axis index units;
    *boundaries* draws thin black separators at species/domain edges. *orientation*
    and *align* place the bar on the chosen edge, *flip_axis* reverses a vertical
    bar's direction, and *spine_linewidth* sets its border width. Returns the bar axes.
    """
    if orientation == 'top' and align == 'top':
        pad = _DOMAIN_BAR_PAD_TOP
    elif orientation == 'bottom' or align == 'bottom':
        pad = _DOMAIN_BAR_PAD_BOTTOM
    else:
        pad = _DOMAIN_BAR_PAD_TOP
    # Temporarily set rc linewidth to desired value before creating axes
    old_linewidth = plt.rcParams['axes.linewidth']
    plt.rc('axes', linewidth=spine_linewidth)
    bar_ax = _create_bar_axis(ax, orientation, size_frac=_DOMAIN_BAR_FRAC,
                              pad=pad, align=align)
    plt.rc('axes', linewidth=old_linewidth)  # Restore original
    if orientation in ('top', 'bottom'):
        x0, x1 = ax.get_xlim()
        bar_ax.set_xlim(x0, x1)
        for s0, e0, color in bars or []:
            start = s0 - 0.5
            width = (e0 - s0 + 1)
            bar_ax.add_patch(Rectangle((start, 0), width, 1, facecolor=color, edgecolor='none'))
        if boundaries is not None and len(boundaries) > 0:
            for b in boundaries:
                bar_ax.axvline(b - 0.5, color='k', linewidth=0.4, zorder=10)
        # Add borders to seamlessly join with axes
        # Must explicitly set each spine to override plt.rc defaults
        for spine_name in ['top', 'bottom', 'left', 'right']:
            bar_ax.spines[spine_name].set_visible(True)
            bar_ax.spines[spine_name].set_color('black')
            bar_ax.spines[spine_name].set_linewidth(spine_linewidth)
    else:
        y0, y1 = ax.get_ylim()
        # If flipping, invert the bar_ax y-axis by setting limits in reverse order
        if flip_axis:
            bar_ax.set_ylim(y1, y0)  # Reverse the limits to flip
        else:
            bar_ax.set_ylim(y0, y1)
        # Draw rectangles using the original coordinates - the inverted axis will flip them visually
        for s0, e0, color in bars or []:
            start = s0 - 0.5
            height = (e0 - s0 + 1)
            bar_ax.add_patch(Rectangle((0, start), 1, height, facecolor=color, edgecolor='none'))
        if boundaries is not None and len(boundaries) > 0:
            for b in boundaries:
                bar_ax.axhline(b - 0.5, color='k', linewidth=0.4, zorder=10)
        # Add borders to seamlessly join with axes
        # Must explicitly set each spine to override plt.rc defaults
        for spine_name in ['top', 'bottom', 'left', 'right']:
            bar_ax.spines[spine_name].set_visible(True)
            bar_ax.spines[spine_name].set_color('black')
            bar_ax.spines[spine_name].set_linewidth(spine_linewidth)
    return bar_ax


# -----------------------
# Domain annotation helpers
# -----------------------

_DOMAINS_CACHE = None  # filled on first use


def _find_domains_csv() -> Optional[str]:
    """Locate domains.csv according to the new repo layout.

    Search order:
      1) Environment override: DOMAINS_CSV
      2) Script directory: <repo>/ANALYSIS_FIG-.../domains.csv
      3) Current working directory: ./domains.csv (legacy)
    """
    # 1) Explicit override
    env_path = os.environ.get("DOMAINS_CSV")
    if env_path and os.path.isfile(env_path):
        return env_path

    # 2) Script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_domains = os.path.join(script_dir, "domains.csv")
    if os.path.isfile(script_domains):
        return script_domains

    # 3) Current working directory (legacy behavior)
    cwd_domains = os.path.join(os.getcwd(), "domains.csv")
    if os.path.isfile(cwd_domains):
        return cwd_domains

    return None


def _normalize_protein_name(name: str) -> str:
    """Map a domains.csv protein label (e.g. 'TDP-43') to the internal matrix name (e.g. 'ProteinTDP43')."""
    # Map CSV names to internal names used in matrices
    name = name.strip()
    mapping = {
        "TDP-43": "ProteinTDP43",
        "TDP43": "ProteinTDP43",
        "FUS": "ProteinFUS",
        "TIA1": "ProteinTIA1",
        "G3BP1": "ProteinG3BP1",
        "PABP1": "ProteinPABP1",
        "TTP": "ProteinTTP",
        "RNA": "RNA",
    }
    return mapping.get(name, name)


def _to_hex(rgb):
    """Convert an (r, g, b) tuple of 0-1 floats to a '#rrggbb' hex color string (values clamped)."""
    return "#%02x%02x%02x" % tuple(int(max(0, min(1, c)) * 255) for c in rgb[:3])


def _domain_color(label: str) -> str:
    """Apply Apple-style named colors for domain rectangles.

    Mapping requested:
    - Unstructured: IDR → clover; LCD → spring; Prion → honeydew
    - RGG bridge: seafoam
    - RBP: RRM → midnight; PABC → blueberry; TZF → sky; ZF → ice
    - RNA: Coding → lavender; Poly‑A (non-coding) → eggplant
    - Structured: Linker → cayenne; NTF2 → maraschino; NTD → tangerine
    - NA: dark grey
    """
    lab = (label or "").lower()

    APPLE = {
        # Greens
        "clover":    "#016100",  # vivid green
        "spring":    "#02bd00",  # bright spring green
        "honeydew":  "#bdff14",  # very light yellow‑green
        "seafoam":   "#03F987",  # seafoam/turquoise
        # Blues
        "midnight":  "#002EFF",  # deep navy
        "blueberry": "#001888",  # strong blue
        "sky":       "#6ACFFF",  # sky blue
        "ice":       "#68FDFF",  # very light blue
        # Purples
        "lavender":  "#D278FF",  # light purple
        "eggplant":  "#491A88",  # dark purple
        # Reds/Oranges
        "cayenne":   "#891100",  # dark red
        "maraschino":"#FF2101",  # bright red
        "tangerine": "#FF8802",  # orange
    }

    if lab == "na":
        return "#666666"

    # Unstructured
    
    if "idr" in lab:
        return APPLE["clover"]
    if ("lcd" in lab) or ("low complexity" in lab):
        return APPLE["spring"]
    if ("pld" in lab) or ("prion" in lab) or ("prion-like" in lab):
        return APPLE["honeydew"]
    
    # RGG bridge
    if "rgg" in lab:
        return APPLE["seafoam"]

    # RBP
    if "rrm" in lab:
        return APPLE["midnight"]
    if "pabc" in lab:
        return APPLE["blueberry"]
    if ("zinc" in lab) and ("finger" in lab):
        return APPLE["sky"]
    if ("tandem zinc" in lab) or ("tzf" in lab):
        return APPLE["ice"]

    # RNA pieces
    if "coding" in lab:
        return APPLE["lavender"]
    if "poly" in lab:
        return APPLE["eggplant"]

    # Structured
    if ("linker" in lab) or ("pro-rich" in lab) or ("pro rich" in lab):
        return APPLE["cayenne"]
    if "ntf2" in lab:
        return APPLE["maraschino"]
    if "ntd" in lab:
        return APPLE["tangerine"]

    # Fallback for unrecognized labels
    return "#666666"

def _load_domains() -> dict:
    """Parse domains.csv into {protein -> [(start, end, label), ...]}, cached after first call.

    Returns an empty dict if domains.csv is absent or unparseable, which disables
    the domain annotation bars.
    """
    global _DOMAINS_CACHE
    if _DOMAINS_CACHE is not None:
        return _DOMAINS_CACHE
    path = _find_domains_csv()
    domains = {}
    if path and os.path.exists(path):
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
            for _, row in df.iterrows():
                prot = _normalize_protein_name(str(row["Protein"]))
                rng = str(row["AminoAcids"]).strip()
                label = str(row["Domain"]).strip()
                if "-" in rng:
                    a, b = rng.split("-", 1)
                    try:
                        a = int(a)
                        b = int(b)
                        if a > b:
                            a, b = b, a
                    except Exception:
                        continue
                else:
                    try:
                        a = b = int(rng)
                    except Exception:
                        continue
                domains.setdefault(prot, []).append((a, b, label))
        except Exception:
            print(f"Warning: Failed to parse domains.csv at {path}; domain bars disabled.")
            domains = {}
    _DOMAINS_CACHE = domains
    return domains


def _interval_complement(intervals: list[tuple[int, int]], length: int) -> list[tuple[int, int]]:
    """Return the gaps (unannotated 'NA' spans) in [1, length] not covered by *intervals*.

    Inputs and outputs are 1-based inclusive ranges; overlapping intervals are merged first.
    """
    # intervals are 1-based inclusive; returns gaps also 1-based inclusive
    if length <= 0:
        return []
    if not intervals:
        return [(1, length)]
    ivals = sorted([(max(1, a), min(length, b)) for (a, b) in intervals if a <= length and b >= 1])
    merged = []
    for a, b in ivals:
        if not merged or a > merged[-1][1] + 1:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)
    gaps = []
    prev = 0
    for a, b in merged:
        if a > prev + 1:
            gaps.append((prev + 1, a - 1))
        prev = b
    if prev < length:
        gaps.append((prev + 1, length))
    return gaps


def _bars_for_biopolymer(bio: str, length: int) -> list[tuple[int, int, str]]:
    """Build the colored domain spans for one biopolymer across its residue axis.

    Returns a start-sorted list of (start0, end0, color) 0-based inclusive index
    spans covering both the annotated domains and the 'NA' gaps between them.
    """
    # Returns list of (start0, end0, color) 0-based inclusive indices across residue axis
    doms = _load_domains()
    entries = doms.get(bio, [])
    # Known domains
    dom_ivals = [(a, b) for (a, b, _lab) in entries]
    # Add NA gaps
    na_ivals = _interval_complement(dom_ivals, length)
    bars = []
    for a, b, lab in entries:
        color = _domain_color(lab)
        bars.append((max(0, a - 1), min(length - 1, b - 1), color))
    for a, b in na_ivals:
        color = _domain_color("NA")
        bars.append((max(0, a - 1), min(length - 1, b - 1), color))
    # Sort by start
    bars.sort(key=lambda x: x[0])
    return bars

def _discover_sms_with_domain_contacts(category_dir: str,
                                       prefix: str = "Domain_Contacts_Total_") -> list:
    """Return sorted list of SM tags that have Domain_Contacts files in a category dir.

    Looks for files matching <prefix><sm>_Protein*_*_*.(csv|csv.npz)
    and extracts the <sm> token.
    """
    patterns = [
        os.path.join(category_dir, f"{prefix}*_Protein*_*_*.csv"),
        os.path.join(category_dir, f"{prefix}*_Protein*_*_*.csv.npz"),
    ]
    files = []
    for pattern in patterns:
        files.extend(glob(pattern))
    sms = set()
    for fp in files:
        base = os.path.basename(fp)
        if not base.startswith(prefix):
            continue
        tail = base[len(prefix):]
        if tail.endswith(".csv.npz"):
            tail = tail[:-len(".csv.npz")]
        elif tail.endswith(".csv"):
            tail = tail[:-len(".csv")]
        tokens = tail.split("_")
        # Accumulate tokens for SM until we hit a biopolymer token (Protein* or RNA)
        sm_tokens = []
        for tok in tokens:
            if tok.startswith("Protein") or tok.startswith("RNA"):
                break
            sm_tokens.append(tok)
        if sm_tokens:
            sm = "_".join(sm_tokens)
            sms.add(sm)
    return sorted(sms)


def _discover_sms_from_ave_dir(ave_dir: str,
                               prefix: str = "Domain_Contacts_Mean_") -> list:
    """Discover SM tags from existing averaged CSVs in ave_dir.

    Expects files named: <prefix><sm>_<ProteinA>_<ProteinB>.csv
    Skips category-aggregate files like Domain_Contacts_Mean_DSM_...
    """
    pattern = os.path.join(ave_dir, f"{prefix}*_Protein*_*.*")
    files = glob(pattern)
    sms = set()
    for fp in files:
        base = os.path.basename(fp)
        if not base.startswith(prefix):
            continue
        tail = base[len(prefix):]
        tokens = tail.split("_")
        sm_tokens = []
        for tok in tokens:
            if tok.startswith("Protein") or tok.startswith("RNA"):
                break
            sm_tokens.append(tok)
        if not sm_tokens:
            continue
        sm = "_".join(sm_tokens)
        # Skip category aggregates
        if sm in {"SG", "DSM", "NDSM"}:
            continue
        sms.add(sm)
    return sorted(sms)


def _load_time_averaged_pair(category_dir: str, sm: str, bio_i: str, bio_j: str,
                             p: int, q: int, tmin: int, dt: int, tmax: int,
                             raw_prefix: str = "Domain_Contacts_Total_") -> tuple[np.ndarray, np.ndarray, int]:
    """Load and time-average Domain_Contacts for a single pair across frames.

    Returns (mean_matrix, sem_matrix, count) where count is the number of frames found.
    """
    mats = []
    for k in range(tmin, tmax, dt):
        transpose = False
        path_to_use = None
        for label in _time_labels(k):
            for prefix in _raw_prefixes_for_pair(raw_prefix, bio_i, bio_j):
                for file_path in _domain_contact_candidates(category_dir, sm, bio_i, bio_j, label, prefix=prefix):
                    if os.path.exists(file_path):
                        path_to_use = file_path
                        break
                if path_to_use is not None:
                    break
                for alt_path in _domain_contact_candidates(category_dir, sm, bio_j, bio_i, label, prefix=prefix):
                    if os.path.exists(alt_path):
                        path_to_use = alt_path
                        transpose = True
                        break
                if path_to_use is not None:
                    break
            if path_to_use is not None:
                break
        if path_to_use is None:
            continue
        try:
            arr = _load_domain_contact_matrix(path_to_use)
            if transpose:
                # Reverse-pair file: transpose back to (bio_i, bio_j) orientation.
                arr = arr.T
            ave_diff = reshape_to_subarrays(arr, p, q)
            mats.append(ave_diff)
        except Exception:
            continue
    count = len(mats)
    if count > 0:
        stack = np.stack(mats, axis=0)
        mean_arr_diff = np.nanmean(stack, axis=0)
        sem_arr_diff = (np.nanstd(stack, axis=0, ddof=1) / np.sqrt(count)
                        if count > 1 else np.full_like(mean_arr_diff, np.nan, dtype=float))
    else:
        mean_arr_diff = np.zeros((p, q))
        sem_arr_diff = np.full((p, q), np.nan, dtype=float)
    return mean_arr_diff, sem_arr_diff, count


def _save_pair_csv(out_dir: str, tag: str, bio_i: str, bio_j: str, mat: np.ndarray,
                   prefix: str = "Domain_Contacts_Mean_") -> None:
    """Save a per-pair matrix to <out_dir>/<prefix><tag>_<bio_i>_<bio_j>.csv (out_dir must already exist)."""
    # Do NOT create new AVE directories here; ensure we are writing into the existing run
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(f"Expected existing output dir not found: {out_dir}")
    out_path = os.path.join(out_dir, f"{prefix}{tag}_{bio_i}_{bio_j}.csv")
    np.savetxt(out_path, mat, delimiter=",")


def _aggregate_over_sms(per_sm_dicts: list[dict], bio_list: list, bio_lengths: list) -> dict:
    """Compute the unweighted mean across SMs for each (bio_i, bio_j) pair.

    *per_sm_dicts* is a list of dicts each mapping (bio_i, bio_j) -> matrix.
    Returns a dict mapping each pair present in at least one SM to its mean matrix.
    """
    agg = {}
    for i, bio_i in enumerate(bio_list):
        for j in range(i, len(bio_list)):
            bio_j = bio_list[j]
            mats = []
            for d in per_sm_dicts:
                key = (bio_i, bio_j)
                if key in d:
                    mats.append(d[key])
            if mats:
                agg[(bio_i, bio_j)] = np.mean(np.stack(mats, axis=0), axis=0)
    return agg


def _aggregate_over_sms_with_sem(per_sm_dicts: list[dict], bio_list: list, bio_lengths: list,
                                  per_sm_sem_dicts: Optional[list[dict]] = None) -> tuple[dict, dict]:
    """Aggregate per-SM matrices into a category-level mean and SEM.

    For N_SM == 1 (e.g. SG control with single sg_X trajectory): the aggregate SEM
    falls back to the per-SM time-window SEM (passed via per_sm_sem_dicts), since
    cross-SM SEM is undefined with n=1 sample.

    For N_SM >= 2 (e.g. DSM_AVG, NDSM_AVG): aggregate SEM is the cross-SM SEM
    (std/sqrt(n)) summed in quadrature with the mean intra-SM SEM, capturing both
    inter-trajectory (between-compound) and intra-trajectory (time-window)
    variability via a simple random-effects model. If per_sm_sem_dicts is None,
    falls back to cross-SM SEM only (legacy behavior).
    """
    mean = {}
    sem = {}
    for i, bio_i in enumerate(bio_list):
        for j in range(i, len(bio_list)):
            bio_j = bio_list[j]
            key = (bio_i, bio_j)
            mats = [d[key] for d in per_sm_dicts if key in d]
            if not mats:
                continue
            stack = np.stack(mats, axis=0)
            mean[key] = np.nanmean(stack, axis=0)
            n_sm = stack.shape[0]
            cross_sem = (np.nanstd(stack, axis=0, ddof=1) / np.sqrt(n_sm)
                         if n_sm > 1 else np.full_like(mean[key], np.nan, dtype=float))
            if per_sm_sem_dicts is not None:
                sem_mats = [d.get(key) for d in per_sm_sem_dicts if key in d]
                sem_mats = [s for s in sem_mats if s is not None]
                if sem_mats:
                    sem_stack = np.stack(sem_mats, axis=0)
                    if n_sm == 1:
                        # Single SM: aggregate SEM = the per-SM time-window SEM
                        sem[key] = sem_stack[0].copy()
                        continue
                    # Multi-SM: random-effects-style combination.
                    # mean intra-SM variance ~ <SEM^2>; convert to aggregate
                    # SEM-of-mean by dividing by sqrt(n_sm).
                    mean_intra_var = np.nanmean(sem_stack ** 2, axis=0)
                    intra_sem = np.sqrt(np.maximum(mean_intra_var / n_sm, 0.0))
                    # Combine cross-SM and intra-SM in quadrature
                    sem[key] = np.sqrt(np.nan_to_num(cross_sem, nan=0.0) ** 2 +
                                       np.nan_to_num(intra_sem, nan=0.0) ** 2)
                    # Restore NaN where everything was NaN
                    bad = ~(np.isfinite(cross_sem) | np.isfinite(intra_sem))
                    sem[key][bad] = np.nan
                    continue
            sem[key] = cross_sem
    return mean, sem


def _plot_contact_heatmap(array: np.ndarray, output_path: str,
                          symmetric: bool = False,
                          tick_positions: Optional[list] = None,
                          tick_labels: Optional[list] = None,
                          x_bars: Optional[list] = None,
                          y_bars: Optional[list] = None,
                          flip_y_bars: bool = False,
                          x_bar_labels: Optional[list] = None,
                          y_bar_labels: Optional[list] = None,
                          block_boundaries: Optional[list] = None,
                          invert_y: bool = True,
                          cbar_limits: tuple[float, float] = (0.0, 10.0),
                          show_colorbar: bool = True,
                          show_xticks: bool = True,
                          show_yticks: bool = True,
                          apply_neg_log_relative: bool = True,
                          cmap_name: Optional[str] = None) -> None:
    """Plot a domain contact heatmap.

    When apply_neg_log_relative is True (default, backward-compatible behavior),
    the input array is treated as raw contact counts and rendered as
    -ln(C / C_max) with the "Reds_r" colormap (dark = strong contact).

    When apply_neg_log_relative is False, the array is plotted as-is using
    cmap_name (default "Reds" if not specified, or "coolwarm" if the data
    crosses zero). Use this for P_contact, -ln(P_contact), and DIFF maps where
    the transform is already applied upstream.
    """
    sns.set_theme(style="ticks")
    sns.set_style('white')
    plt.rc('axes', titlesize=8)
    plt.rc('axes', labelsize=8)
    plt.rc('xtick', labelsize=_DOMAIN_PAIR_TICK_FONTSIZE)
    plt.rc('ytick', labelsize=_DOMAIN_PAIR_TICK_FONTSIZE)
    plt.rc('legend', fontsize=8)
    plt.rc('font', size=8)
    plt.rc('axes', linewidth=2)

    array_proc = np.asarray(array, dtype=float).copy()

    if apply_neg_log_relative:
        finite_mask = np.isfinite(array_proc)
        positive = array_proc[finite_mask & (array_proc > 0)]
        if positive.size == 0:
            min_positive = 1e-12
            max_val = 1e-12
        else:
            min_positive = np.min(positive)
            max_val = np.nanmax(array_proc)
        array_proc[finite_mask & (array_proc == 0)] = min_positive
        array_proc = -np.log(array_proc / max_val)
        default_cmap = "Reds_r"
    else:
        default_cmap = "Reds"

    plot_mask = ~np.isfinite(array_proc)

    is_square = array_proc.shape[0] == array_proc.shape[1]
    fig_size, ax_rect = _contact_map_layout(
        array_proc.shape,
        square=is_square,
        has_x_bar=bool(x_bars),
        has_y_bar=bool(y_bars),
        show_colorbar=show_colorbar,
    )

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_axes(ax_rect)
    cmap = plt.get_cmap(cmap_name if cmap_name else default_cmap).copy()
    cmap.set_bad(color="lightgrey")
    ax = sns.heatmap(
        array_proc,
        cmap=cmap,
        square=is_square,
        ax=ax,
        cbar=False,
        mask=plot_mask,
        vmin=cbar_limits[0], vmax=cbar_limits[1]
    )
    if is_square:
        ax.set_box_aspect(1)
    for side in ["top", "bottom", "left", "right"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color('black')
        ax.spines[side].set_linewidth(2)
    if show_colorbar:
        pos = ax.get_position()
        cbar_ax = fig.add_axes([
            _cbar_left_after_right_bar(ax, bool(y_bars)),
            pos.y0,
            _CBAR_WIDTH_IN / fig.get_figwidth(),
            pos.height,
        ])
        mappable = ax.collections[0]
        clim = mappable.get_clim()
        tick_lo = _round_2sig_floor(clim[0])
        tick_hi = _round_2sig_ceil(clim[1])
        if abs(tick_hi - tick_lo) < 1e-30:
            tick_hi = tick_lo + 1.0
        ticks = np.linspace(tick_lo, tick_hi, 5)
        mappable.set_clim(tick_lo, tick_hi)
        cbar = fig.colorbar(mappable, cax=cbar_ax)
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
        cbar.ax.text(0.5, 1.01, f"$\\times10^{{{common_exp}}}$",
            transform=cbar.ax.transAxes, ha='center', va='bottom', fontsize=_DOMAIN_CBAR_FONTSIZE)
        cbar.outline.set_edgecolor('black')
        cbar.outline.set_linewidth(2)
        cbar.ax.tick_params(labelsize=_DOMAIN_CBAR_FONTSIZE, width=2, length=4)
        for spine_name in ['top', 'bottom', 'left', 'right']:
            cbar.ax.spines[spine_name].set_visible(True)
            cbar.ax.spines[spine_name].set_color('black')
            cbar.ax.spines[spine_name].set_linewidth(2)
    ax.tick_params(axis='both', labelsize=_DOMAIN_PAIR_TICK_FONTSIZE)
    # Invert y-axis for non-FULL plots; keep as-is for FULL_FULL when invert_y=False
    if invert_y:
        ax.set_ylim(ax.get_ylim()[::-1])

    def _even_ticks(length: int, n: int = _PAIR_TICK_COUNT):
        """Return (cell-centered positions, integer labels) for *n* evenly spaced
        residue ticks over an axis of *length* cells."""
        if length <= 0:
            return [], []
        max_val = int(np.ceil(length))
        vals = np.linspace(0, max_val, n)
        vals_int = [int(round(v)) for v in vals]
        centers = [v + 0.5 for v in vals_int]
        tick_labels_str = [str(v) for v in vals_int]
        return centers, tick_labels_str

    # FULL_FULL maps use explicit species ticks; pair maps use numeric residue ticks.
    if symmetric and tick_positions is not None and tick_labels is not None:
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        # Rotate species labels by 45 degrees; center align to block centers
        ax.set_xticklabels(tick_labels, rotation=45, ha='center', va='top', rotation_mode='default')
        ax.set_yticklabels(tick_labels, rotation=45, ha='right', va='center', rotation_mode='default')
        # Match label size to other plots
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_fontsize(_DOMAIN_FULL_TICK_FONTSIZE)
        ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True,
                       top=False, labeltop=False)
        ax.xaxis.set_ticks_position('bottom')
        ax.xaxis.set_label_position('bottom')
        ax.set_xticklabels(tick_labels, rotation=45, ha='center', va='top',
                           rotation_mode='default', fontsize=_DOMAIN_FULL_TICK_FONTSIZE)
    else:
        y_centers, y_labels = _even_ticks(array_proc.shape[0])
        x_centers, x_labels = _even_ticks(array_proc.shape[1])
        if y_centers:
            ax.set_yticks(y_centers)
            ax.set_yticklabels(y_labels)
        if x_centers:
            ax.set_xticks(x_centers)
            ax.set_xticklabels(x_labels)
        ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True,
                       top=False, labeltop=False)
        ax.xaxis.set_ticks_position('bottom')
        ax.xaxis.set_label_position('bottom')

    # Tick marks 4pt long; pad=0 places label at the tick-mark tip,
    # matching x and y label distance from the spine.
    ax.tick_params(axis='both', which='major', pad=0, direction='out', length=4, width=2)
    ax.tick_params(axis='both', which='minor', pad=0, direction='out', length=2, width=2)
    if not show_xticks:
        ax.tick_params(axis='x', bottom=False, top=False, labelbottom=False, labeltop=False)
    if not show_yticks:
        ax.tick_params(axis='y', left=False, right=False, labelleft=False, labelright=False)

    # Draw thin boundaries at protein starts/ends if requested
    if block_boundaries:
        for b in sorted(set(block_boundaries)):
            ax.axvline(b - 0.5, color='k', linewidth=0.4, zorder=3)
            ax.axhline(b - 0.5, color='k', linewidth=0.4, zorder=3)

    # Invert y-axis so FUS sits at bottom-left and RNA at top-right
    ax.invert_yaxis()

    if x_bars:
        x_orient = 'top'
        xbar_ax = _add_domain_bar(fig, ax, x_bars, block_boundaries, orientation=x_orient,
                                   align='top')
        if x_bar_labels:
            x0, x1 = ax.get_xlim()
            pos_bar = xbar_ax.get_position()
            for s0, e0, label in x_bar_labels:
                xc = (s0 + e0) / 2.0
                frac = (xc - x0) / (x1 - x0) if x1 != x0 else 0.0
                xp = pos_bar.x0 + frac * pos_bar.width
                if x_orient == 'top':
                    yp = min(0.995, pos_bar.y1 + 0.006)
                else:
                    yp = max(0.0, pos_bar.y0 - 0.006)
                fig.text(xp, yp, label, ha='center', va='bottom', fontsize=8, rotation=45)
    if y_bars:
        y_orient = 'right'
        ybar_ax = _add_domain_bar(fig, ax, y_bars, block_boundaries,
                                  orientation=y_orient, flip_axis=(flip_y_bars and not invert_y))
        # For FULL_FULL (symmetric with explicit block boundaries), force inversion
        # so the right-side domain colors align with the species order.
        if symmetric and block_boundaries is not None:
            try:
                ybar_ax.invert_yaxis()
            except Exception:
                pass
        if y_bar_labels:
            for s0, e0, label in y_bar_labels:
                yc = (s0 + e0) / 2.0
                if y_orient == 'left':
                    ybar_ax.text(-0.1, yc, label, ha='center', va='center', rotation=45, fontsize=8)
                else:
                    ybar_ax.text(0.6, yc, label, ha='left', va='center', rotation=45, fontsize=8)
    # Axis label positioning: all domain maps use bottom x labels and left y labels.
    if symmetric and tick_positions is not None and tick_labels is not None:
        ax.tick_params(axis='x', labelbottom=True, labeltop=False, bottom=True, top=False)
        ax.tick_params(axis='y', labelleft=True, labelright=False, left=True, right=False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        for t in ax.get_xticklabels():
            t.set_rotation(45)
            t.set_ha('center')
            t.set_va('top')
            t.set_rotation_mode('default')
        for t in ax.get_yticklabels():
            t.set_rotation(45)
            t.set_ha('right')
            t.set_va('center')
            t.set_rotation_mode('default')
    else:
        ax.tick_params(axis='x', labelbottom=True, labeltop=False, bottom=True, top=False)
        ax.tick_params(axis='y', labelright=False, labelleft=True, right=False, left=True)
        ax.xaxis.set_ticks_position('bottom')
        ax.xaxis.set_label_position('bottom')

    fig.savefig(output_path, format="png", dpi=400)
    plt.close(fig)


def _flip_rna_axes(arr: np.ndarray, bio_i: str, bio_j: str) -> np.ndarray:
    """Reverse the RNA axis in the per-pair matrix so that RNA position 0
    (5' coding-region start) becomes the LAST RNA-axis index and position
    L_RNA-1 (3' poly-A end) becomes the FIRST. The result is that, when
    the per-pair block is placed into the FULL_FULL matrix with RNA as
    the last species, the poly-A tail sits AT THE BOUNDARY with the
    adjacent protein band (TTP) rather than at the far corner.

    Applied at plot time only; underlying Domain_Contacts_* CSVs keep
    natural 5'→3' indexing.
    """
    a = np.asarray(arr, dtype=float)
    if bio_i == "RNA" and bio_j == "RNA":
        return a[::-1, ::-1]
    if bio_i == "RNA":
        return a[::-1, :]
    if bio_j == "RNA":
        return a[:, ::-1]
    return a


def _flip_rna_bars(bars: list, bio: str, length: int) -> list:
    """Mirror domain-bar segments for the RNA axis so the sub-domain colour bar
    tracks `_flip_rna_axes`.

    The RNA matrix axis is reversed at plot time (poly-A tail moved to the
    inter-species boundary), but `_bars_for_biopolymer` always returns segments
    in natural 5'->3' order. Without this mirror the colour bar ends up inverted
    relative to the heatmap (poly-A at the boundary in the map but at the far end
    in the bar). For non-RNA species the bars are returned unchanged.
    """
    if bio != "RNA":
        return bars
    return [(length - 1 - e0, length - 1 - s0, color) for (s0, e0, color) in bars]


def _combine_to_symmetric(contact_matrices: dict, bio_list: list, bio_lengths: list) -> np.ndarray:
    """Assemble per-pair biopolymer matrices into one symmetric FULL_FULL block matrix.

    Each upper-triangle (bio_i, bio_j) block is placed at its concatenated species
    offsets (RNA axes flipped via _flip_rna_axes) and mirrored into the lower
    triangle so the result is square and symmetric.
    """
    dimensions = bio_lengths
    sum_dim = int(np.sum(np.array(dimensions)))
    combined_array = np.zeros((sum_dim, sum_dim))
    indices = np.cumsum([0] + dimensions)
    for i in range(len(dimensions)):
        for j in range(i, len(dimensions)):
            bio_i = bio_list[i]
            bio_j = bio_list[j]
            key = (bio_i, bio_j)
            if key not in contact_matrices:
                continue
            arr = contact_matrices[key]
            arr = _flip_rna_axes(arr, bio_i, bio_j)
            dim_i = dimensions[i]
            dim_j = dimensions[j]
            combined_array[indices[i]:indices[i] + dim_i,
                           indices[j]:indices[j] + dim_j] = arr
            combined_array[indices[j]:indices[j] + dim_j,
                           indices[i]:indices[i] + dim_i] = arr.T
    return combined_array


def _plot_from_contact_dict(contact_matrices: dict, out_dir: str, bio_list: list, bio_lengths: list,
                            show_colorbar: bool = True, show_yticks: bool = True,
                            pair_suffix: str = "", full_basename: str = "Domain_Contact_Map_FULL_FULL") -> np.ndarray:
    """Render and save the per-pair and combined FULL_FULL contact heatmaps for one category.

    Writes one heatmap per biopolymer pair plus the assembled symmetric FULL_FULL
    map (PNG + CSV) and a 1D stacked contact-probability plot into *out_dir*, and
    returns the combined FULL_FULL array. *pair_suffix* / *full_basename* select the
    output filename variant.
    """
    _ensure_dir(out_dir)
    # Pair plots
    for (bio_i, bio_j), mat in contact_matrices.items():
        name_i = bio_i.replace("Protein", "")
        name_j = bio_j.replace("Protein", "")
        output_path = os.path.join(out_dir, f"Domain_Contact_Map_{name_i}_{name_j}{pair_suffix}.png")
        mat_flipped = _flip_rna_axes(mat, bio_i, bio_j)
        symmetric = (bio_i == bio_j) and (mat_flipped.shape[0] == mat_flipped.shape[1])
        y_bars = _flip_rna_bars(_bars_for_biopolymer(bio_i, mat_flipped.shape[0]), bio_i, mat_flipped.shape[0])
        x_bars = _flip_rna_bars(_bars_for_biopolymer(bio_j, mat_flipped.shape[1]), bio_j, mat_flipped.shape[1])
        _plot_contact_heatmap(mat_flipped, output_path, symmetric=symmetric, x_bars=x_bars, y_bars=y_bars,
                              flip_y_bars=symmetric, invert_y=False)
    # Combined plot
    combined_array = _combine_to_symmetric(contact_matrices, bio_list, bio_lengths)
    output_path = os.path.join(out_dir, f"{full_basename}.png")
    # Build domain bars across concatenated axis
    offsets = np.cumsum([0] + bio_lengths)
    full_bars = []
    for i, bio in enumerate(bio_list):
        offset = int(offsets[i])
        bars = _flip_rna_bars(_bars_for_biopolymer(bio, bio_lengths[i]), bio, bio_lengths[i])
        for s0, e0, color in bars:
            full_bars.append((offset + s0, offset + e0, color))
    # Compute species-centered tick positions and labels
    centers = []
    labels = []
    for i, bio in enumerate(bio_list):
        start = int(offsets[i])
        end = int(offsets[i+1]) - 1
        centers.append((start + end) / 2.0 + 0.5)
        labels.append(bio.replace("Protein", ""))

    # Use identical bars for x and y in FULL_FULL and set species labels at block centers.
    # Local intrachain exclusions are applied upstream before same-species matrices
    # are collapsed, so interchain same-index contacts are retained exactly.
    _plot_contact_heatmap(combined_array, output_path, symmetric=True, x_bars=full_bars, y_bars=full_bars,
                          flip_y_bars=True, x_bar_labels=None, y_bar_labels=None,
                          block_boundaries=list(offsets), tick_positions=centers, tick_labels=labels,
                          invert_y=False, show_colorbar=show_colorbar, show_yticks=show_yticks)
    np.savetxt(os.path.join(out_dir, f"{full_basename}.csv"), combined_array, delimiter=",")

    # 1D stacked contact probability plot derived from FULL_FULL
    prob_out = os.path.join(out_dir, "Domain_Contact_Probabilities_FULL.png")
    try:
        _plot_stacked_contact_probabilities(
            combined_array,
            prob_out,
            bio_list,
            bio_lengths,
        )
    except Exception as e:
        print(f"Warning: failed to generate stacked contact probability plot: {e}")

    return combined_array


# ----------------------------------------------------------------------
# P_contact / K_chain domain maps (literature-aligned normalization).
# K_chain[i,j]   = <C_ij(t)>_t  (mean contacts per chain pair per frame
#                 between residue sequence position i in species A and
#                 sequence position j in species B; this is the existing
#                 raw map).
# P_contact[i,j] = K_chain[i,j] / N_chain_pairs(A,B)  (probability in [0,1]
#                 of contact for any random pair of chains A,B at the
#                 specified residue indices).
# -ln(P_contact) = apparent contact free energy in nats (k_BT).
# ----------------------------------------------------------------------

def _array_species_to_cn_label(bio_name: str) -> str:
    """Map contact_maps.py species name ('ProteinFUS', 'RNA', ...) to CN species label."""
    return bio_name.replace("Protein", "")


def _build_chain_pair_lookup(biopolnum_csv_path: str) -> dict:
    """Return dict (bio_i, bio_j) -> N_chain_pairs from BioPolNum CSV."""
    import contact_normalization as _CN
    bio = np.loadtxt(biopolnum_csv_path, delimiter=",", dtype=float)
    n_chains = _CN.n_chains_from_biopolnum(bio)
    pair_mat = _CN.n_chain_pairs_matrix(n_chains)  # 7x7 in CN order
    lookup = {}
    cn_order = _CN.SPECIES_ORDER  # ["G3BP1", "PABP1", "TTP", "TIA1", "TDP43", "FUS", "RNA"]
    cn_idx = {sp: i for i, sp in enumerate(cn_order)}
    for bio_i in BIOPOLYMER_LIST:
        for bio_j in BIOPOLYMER_LIST:
            li = _array_species_to_cn_label(bio_i)
            lj = _array_species_to_cn_label(bio_j)
            if li not in cn_idx or lj not in cn_idx:
                continue
            lookup[(bio_i, bio_j)] = float(pair_mat[cn_idx[li], cn_idx[lj]])
    return lookup


def _convert_kchain_to_pcontact_dict(kchain_dict: dict, chainpair_lookup: dict) -> dict:
    """Divide every mat in kchain_dict by N_chain_pairs(A,B) -> P_contact dict."""
    p_dict = {}
    for (bio_i, bio_j), mat in kchain_dict.items():
        denom = chainpair_lookup.get((bio_i, bio_j))
        if denom is None or not np.isfinite(denom) or denom <= 0:
            continue
        p_dict[(bio_i, bio_j)] = np.asarray(mat, dtype=float) / denom
    return p_dict


def _neg_ln_of_dict(p_dict: dict) -> dict:
    """Elementwise -ln(P) for each matrix in p_dict; NaN where P<=0 or non-finite."""
    out = {}
    for key, mat in p_dict.items():
        a = np.asarray(mat, dtype=float)
        m = np.full_like(a, np.nan, dtype=float)
        mask = np.isfinite(a) & (a > 0)
        m[mask] = -np.log(a[mask])
        out[key] = m
    return out


def _plot_pcontact_full(contact_matrices: dict, out_dir: str, bio_list: list, bio_lengths: list,
                       cmap_name: str, cbar_vmin: float, cbar_vmax: float,
                       full_basename: str, pair_suffix: str = "",
                       show_colorbar: bool = True, show_yticks: bool = True) -> None:
    """Plot pre-transformed P_contact-style maps (no internal log transform)."""
    _ensure_dir(out_dir)
    for (bio_i, bio_j), mat in contact_matrices.items():
        name_i = bio_i.replace("Protein", "")
        name_j = bio_j.replace("Protein", "")
        output_path = os.path.join(out_dir, f"Domain_Contact_Map_{name_i}_{name_j}{pair_suffix}.png")
        mat_flipped = _flip_rna_axes(mat, bio_i, bio_j)
        symmetric = (bio_i == bio_j) and (mat_flipped.shape[0] == mat_flipped.shape[1])
        y_bars = _flip_rna_bars(_bars_for_biopolymer(bio_i, mat_flipped.shape[0]), bio_i, mat_flipped.shape[0])
        x_bars = _flip_rna_bars(_bars_for_biopolymer(bio_j, mat_flipped.shape[1]), bio_j, mat_flipped.shape[1])
        _plot_contact_heatmap(
            mat_flipped, output_path, symmetric=symmetric, x_bars=x_bars, y_bars=y_bars,
            flip_y_bars=symmetric, invert_y=False,
            apply_neg_log_relative=False, cmap_name=cmap_name,
            cbar_limits=(cbar_vmin, cbar_vmax),
        )
    combined_array = _combine_to_symmetric(contact_matrices, bio_list, bio_lengths)
    output_path = os.path.join(out_dir, f"{full_basename}.png")
    offsets = np.cumsum([0] + bio_lengths)
    full_bars = []
    for i, bio in enumerate(bio_list):
        offset = int(offsets[i])
        bars = _flip_rna_bars(_bars_for_biopolymer(bio, bio_lengths[i]), bio, bio_lengths[i])
        for s0, e0, color in bars:
            full_bars.append((offset + s0, offset + e0, color))
    centers, labels = [], []
    for i, bio in enumerate(bio_list):
        start = int(offsets[i])
        end = int(offsets[i + 1]) - 1
        centers.append((start + end) / 2.0 + 0.5)
        labels.append(bio.replace("Protein", ""))
    _plot_contact_heatmap(
        combined_array, output_path, symmetric=True, x_bars=full_bars, y_bars=full_bars,
        flip_y_bars=True, x_bar_labels=None, y_bar_labels=None,
        block_boundaries=list(offsets), tick_positions=centers, tick_labels=labels,
        invert_y=False, show_colorbar=show_colorbar, show_yticks=show_yticks,
        apply_neg_log_relative=False, cmap_name=cmap_name,
        cbar_limits=(cbar_vmin, cbar_vmax),
    )
    np.savetxt(os.path.join(out_dir, f"{full_basename}.csv"), combined_array, delimiter=",")


def _dict_global_max(d: dict) -> float:
    """Return the maximum finite value across all matrices in *d* (>=positive), or 1.0 if none."""
    vmax = 0.0
    for mat in d.values():
        a = np.asarray(mat, dtype=float)
        finite = np.isfinite(a)
        if np.any(finite):
            v = float(np.nanmax(a[finite]))
            if v > vmax:
                vmax = v
    return vmax if vmax > 0 else 1.0


def _dict_sym_max(d: dict) -> float:
    """Robust symmetric vmax for signed/difference domain maps: the 99th
    percentile of |value| pooled across all finite cells (not the absolute max),
    so the top ~0.01% of outlier cells don't compress a heavy-tailed difference
    map (e.g. the 3594x3594 FULL_FULL DSM-NDSM diff, where >90% of cells sit
    below 10% of the max) into near-grey. Small maps: 99th pct ~= max, so coarse
    diffs are unchanged. Returns 1.0 if no finite data."""
    pool = []
    for mat in d.values():
        a = np.asarray(mat, dtype=float)
        finite = np.isfinite(a)
        if np.any(finite):
            pool.append(np.abs(a[finite]))
    if not pool:
        return 1.0
    vmax = float(np.nanpercentile(np.concatenate(pool), 99))
    return vmax if vmax > 0 else 1.0


def _generate_ctotal_domain_maps(agg_dict: dict, maps_dir: str,
                                 bio_list: list, bio_lengths: list,
                                 shared_C_max: float,
                                 shared_Pm_max: Optional[float] = None) -> dict:
    """Plot raw C_total (linear) and -ln(C/C_max) (log relative) domain maps,
    plus the P_membership = C / sum(C) variant which sums to 1 over the map
    and shows compositional fraction of total cluster contacts.

    Same data as the legacy Domain_Contact_Map_FULL_FULL but with explicit
    C_TOTAL naming so the three-family scheme (P_contact, K_chain, C_total)
    is exposed consistently. shared_C_max is the shared reference across
    SG/DSM/NDSM so colorbars are directly comparable.
    """
    if not agg_dict:
        return {}
    # Linear C_total (raw counts per frame, identical to agg_dict)
    _plot_pcontact_full(agg_dict, maps_dir, bio_list, bio_lengths,
                       cmap_name="Reds", cbar_vmin=0.0, cbar_vmax=shared_C_max,
                       full_basename="Domain_Contact_Map_FULL_FULL_C_TOTAL",
                       pair_suffix="_C_TOTAL")
    # Log relative -ln(C / C_max)
    nlnC_dict = {}
    for key, mat in agg_dict.items():
        a = np.asarray(mat, dtype=float)
        out = np.full_like(a, np.nan, dtype=float)
        mask = np.isfinite(a) & (a > 0)
        out[mask] = -np.log(a[mask] / shared_C_max)
        nlnC_dict[key] = out
    nlnC_vmax = _dict_global_max(nlnC_dict)
    _plot_pcontact_full(nlnC_dict, maps_dir, bio_list, bio_lengths,
                       cmap_name="Reds_r", cbar_vmin=0.0, cbar_vmax=nlnC_vmax,
                       full_basename="Domain_Contact_Map_FULL_FULL_C_TOTAL_LOG",
                       pair_suffix="_C_TOTAL_LOG")

    # P_membership = C / sum(C) per condition: compositional fraction map.
    total = 0.0
    for mat in agg_dict.values():
        a = np.asarray(mat, dtype=float)
        total += float(np.nansum(a[np.isfinite(a)]))
    if total > 0:
        Pm_dict = {key: np.asarray(mat, dtype=float) / total for key, mat in agg_dict.items()}
        Pm_vmax = shared_Pm_max if shared_Pm_max is not None else _dict_global_max(Pm_dict)
        _plot_pcontact_full(Pm_dict, maps_dir, bio_list, bio_lengths,
                           cmap_name="Reds", cbar_vmin=0.0, cbar_vmax=Pm_vmax,
                           full_basename="Domain_Contact_Map_FULL_FULL_C_TOTAL_P_MEMBERSHIP",
                           pair_suffix="_C_TOTAL_P_MEMBERSHIP")
    return agg_dict


def _generate_ctotal_difference_maps(c_dsm: dict, c_ndsm: dict,
                                     c_sem_dsm, c_sem_ndsm,
                                     out_dir: str, bio_list: list, bio_lengths: list) -> None:
    """DIFF maps in C_total space: linear DeltaC, log-ratio ln(C_DSM/C_NDSM),
    z-score on the log-ratio, plus DeltaP_membership for the compositional view."""
    common = set(c_dsm.keys()) & set(c_ndsm.keys())
    if not common:
        return
    dC = {}
    lnratio = {}
    zlr = {}
    for key in common:
        a = np.asarray(c_dsm[key], dtype=float)
        b = np.asarray(c_ndsm[key], dtype=float)
        dC[key] = a - b
        lr = np.full_like(a, np.nan, dtype=float)
        mask = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
        lr[mask] = np.log(a[mask] / b[mask])
        lnratio[key] = lr
        if c_sem_dsm and c_sem_ndsm and key in c_sem_dsm and key in c_sem_ndsm:
            sa = np.asarray(c_sem_dsm[key], dtype=float)
            sb = np.asarray(c_sem_ndsm[key], dtype=float)
            sem_lr = np.full_like(a, np.nan, dtype=float)
            valid = mask & np.isfinite(sa) & np.isfinite(sb) & (sa > 0) & (sb > 0)
            sem_lr[valid] = np.sqrt((sa[valid] / a[valid])**2 + (sb[valid] / b[valid])**2)
            z = np.full_like(a, np.nan, dtype=float)
            zvalid = valid & (sem_lr > 0) & np.isfinite(lr)
            z[zvalid] = lr[zvalid] / sem_lr[zvalid]
            zlr[key] = z
    for d, fname, pair_suf in [
        (dC, "Domain_Contact_Map_FULL_FULL_DIFF_C_TOTAL", "_DIFF_C_TOTAL"),
        (lnratio, "Domain_Contact_Map_FULL_FULL_DIFF_C_TOTAL_LOGRATIO", "_DIFF_C_TOTAL_LOGRATIO"),
    ]:
        vmax = _dict_sym_max(d)
        _plot_pcontact_full(d, out_dir, bio_list, bio_lengths,
                           cmap_name="coolwarm", cbar_vmin=-vmax, cbar_vmax=vmax,
                           full_basename=fname, pair_suffix=pair_suf)
    if zlr:
        # z-score data is written for availability but not plotted (consistent
        # with the other z-score CSV writers); only the C_total DIFF maps render.
        np.savetxt(os.path.join(out_dir, "Domain_Contact_Map_FULL_FULL_DIFF_C_TOTAL_ZSCORE.csv"),
                   _combine_to_symmetric(zlr, bio_list, bio_lengths), delimiter=",")

    # DIFF in P_membership space: compositional shift between conditions
    total_dsm = sum(float(np.nansum(np.asarray(c_dsm[k], dtype=float)[np.isfinite(c_dsm[k])])) for k in common)
    total_ndsm = sum(float(np.nansum(np.asarray(c_ndsm[k], dtype=float)[np.isfinite(c_ndsm[k])])) for k in common)
    if total_dsm > 0 and total_ndsm > 0:
        dPm = {}
        for key in common:
            a = np.asarray(c_dsm[key], dtype=float) / total_dsm
            b = np.asarray(c_ndsm[key], dtype=float) / total_ndsm
            dPm[key] = a - b
        vmax = _dict_sym_max(dPm)
        _plot_pcontact_full(dPm, out_dir, bio_list, bio_lengths,
                           cmap_name="coolwarm", cbar_vmin=-vmax, cbar_vmax=vmax,
                           full_basename="Domain_Contact_Map_FULL_FULL_DIFF_C_TOTAL_P_MEMBERSHIP",
                           pair_suffix="_DIFF_C_TOTAL_P_MEMBERSHIP")


def _build_n_chains_from_biopolnum(biopolnum_csv_path: str) -> dict:
    """Return {bio_name -> N_chains} for the biopolymer-species row order
    used by contact_maps.py (FUS, TIA1, TDP43, G3BP1, PABP1, TTP, RNA).
    BioPolNum diagonal stores N*(N-1)/2 so N = (1 + sqrt(1 + 8*diag))/2.
    """
    try:
        bio = np.loadtxt(biopolnum_csv_path, delimiter=",", dtype=float)
    except Exception:
        return {}
    # BioPolNum file order matches CONTACT_NORMALIZATION.SPECIES_ORDER:
    # G3BP1, PABP1, TTP, TIA1, TDP43, FUS, RNA
    SPECIES_ORDER_BIO = ["G3BP1", "PABP1", "TTP", "TIA1", "TDP43", "FUS", "RNA"]
    diag = np.diag(bio)
    n_chains = 0.5 * (1.0 + np.sqrt(np.maximum(1.0 + 8.0 * diag, 0.0)))
    out = {}
    for i, sp in enumerate(SPECIES_ORDER_BIO):
        out[f"Protein{sp}" if sp != "RNA" else "RNA"] = float(n_chains[i])
    return out


def _n_possible_domain_pair(bio_a: str, bio_b: str, L_a: int, L_b: int,
                             n_chains: dict, filter_name: str,
                             exclusion_bonded: int = 5) -> np.ndarray:
    """Per-residue-position N_possible(i, j) matrix for the (bio_a, bio_b)
    cell of the domain map, under the requested filter.

    Cases (per user spec):
      - A != B: N_possible(i, j) = N_A * N_B (all interchain, no bonded
        exclusion applies because chains are distinct).
      - A == A interchain: N_A * (N_A - 1) / 2 — valid for all (i, j)
        including i == j (chain1 residue i with chain2 residue i is a
        legitimate inter-chain contact pair).
      - A == A intra (non-local): N_A if |i - j| > exclusion_bonded and
        i != j, else 0.

    Filter dispatch:
      - "all":   inter + intra_nonlocal
      - "inter": inter only
      - "intra": intra_nonlocal only
    """
    N_a = float(n_chains.get(bio_a, 0.0))
    N_b = float(n_chains.get(bio_b, 0.0))
    out = np.zeros((L_a, L_b), dtype=float)
    if N_a <= 0 or N_b <= 0:
        return out

    if bio_a != bio_b:
        # Different species: all interchain, no intra.
        if filter_name in ("all", "inter"):
            out[:, :] = N_a * N_b
        # filter == "intra" => zero everywhere (kept zero).
        return out

    # Same species
    L = L_a
    inter_val = N_a * max(N_a - 1.0, 0.0) / 2.0
    if filter_name in ("all", "inter"):
        out[:, :] += inter_val
    if filter_name in ("all", "intra"):
        i_idx = np.arange(L)
        ii, jj = np.meshgrid(i_idx, i_idx, indexing="ij")
        nonlocal_mask = (np.abs(ii - jj) > exclusion_bonded)
        out += N_a * nonlocal_mask.astype(float)
    return out


def _generate_membership_enrichment_domain_maps(
        C_dict: dict, maps_dir: str,
        biopolnum_csv_path: str,
        bio_list: list, bio_lengths: list,
        filter_name: str,
        variant_suffix: str,
        SEM_C_dict: Optional[dict] = None) -> tuple[dict, dict]:
    """Produce two-panel domain maps for one condition + filter combo.

    Panel A: contact membership f[i, j] = C[i, j] / sum(C) across all
    pair-types and positions. Sums to 1 by construction.

    Panel B: contact enrichment ln(E[i, j]) = ln(f / q) where
    q[i, j] = N_possible[i, j] / sum(N_possible). Centered at 0 on a
    coolwarm colormap so positive = enriched, negative = depleted.

    Both numerator and denominator use the same filter (ALL / INTER /
    INTRA). Returns (membership_dict, log_enrichment_dict).
    """
    if not C_dict:
        return {}, {}
    n_chains = _build_n_chains_from_biopolnum(biopolnum_csv_path)
    if not n_chains:
        print(f"[WARN] _generate_membership_enrichment_domain_maps: empty n_chains for {biopolnum_csv_path}")
        return {}, {}

    # Sum C and N across all (bio_A, bio_B) cells. C_dict stores upper-
    # triangle pair-types only; for the FULL_FULL matrix display we
    # mirror cross-species cells into both [A,B] and [B,A] regions. To
    # keep f = C/sum(C) summing to 1 across the displayed full matrix,
    # both total_C and total_N must double-count cross-species (since
    # those cells appear twice in the plot). Without this, f sums to >1.
    total_C = 0.0
    for (bio_i, bio_j), mat in C_dict.items():
        a = np.asarray(mat, dtype=float)
        s = float(np.nansum(a[np.isfinite(a)]))
        total_C += s
        if bio_i != bio_j:
            total_C += s  # mirror cell
    if total_C <= 0:
        return {}, {}

    N_dict = {}
    total_N = 0.0
    for (bio_i, bio_j), mat in C_dict.items():
        L_i = bio_lengths[bio_list.index(bio_i)]
        L_j = bio_lengths[bio_list.index(bio_j)]
        N = _n_possible_domain_pair(bio_i, bio_j, L_i, L_j, n_chains, filter_name)
        N_dict[(bio_i, bio_j)] = N
        s = float(np.nansum(N))
        total_N += s
        if bio_i != bio_j:
            total_N += s  # mirror cell (same convention as total_C)

    if total_N <= 0:
        return {}, {}

    f_dict = {}
    lnE_dict = {}
    for (bio_i, bio_j), C_mat in C_dict.items():
        C_arr = np.asarray(C_mat, dtype=float)
        f = C_arr / total_C
        f_dict[(bio_i, bio_j)] = f
        q = N_dict[(bio_i, bio_j)] / total_N
        lnE = np.full_like(f, np.nan, dtype=float)
        mask = np.isfinite(f) & np.isfinite(q) & (f > 0) & (q > 0)
        lnE[mask] = np.log(f[mask] / q[mask])
        lnE_dict[(bio_i, bio_j)] = lnE

    # Fixed cap on the membership colorbar to keep the bulk-condensate
    # signal visible: a handful of hot cells (G3BP1-G3BP1 dimer core,
    # ARG-RNA cation-pi patch) otherwise saturate the colormap and wash
    # everything else to white. Cap = 99th percentile of non-zero cells
    # across all (bio_A, bio_B) blocks, so the top ~1% of cells saturate
    # at the bar endpoint while the rest of the matrix shows real detail.
    _f_nonzero = []
    for v in f_dict.values():
        a = np.asarray(v, dtype=float)
        _f_nonzero.append(a[np.isfinite(a) & (a > 0)])
    _f_all = np.concatenate(_f_nonzero) if _f_nonzero and all(len(x) > 0 for x in _f_nonzero) else np.array([])
    f_vmax = float(np.percentile(_f_all, 99)) if _f_all.size else _dict_global_max(f_dict)
    # Fixed cap on ln(E) colorbar (same rationale as acid level).
    lnE_abs_max = 1.5

    # --- Membership: linear f and log -ln(f/f_max) versions ----------
    _plot_pcontact_full(f_dict, maps_dir, bio_list, bio_lengths,
                       cmap_name="Reds", cbar_vmin=0.0, cbar_vmax=f_vmax,
                       full_basename=f"Domain_Contact_Map_FULL_FULL_CONTACT_MEMBERSHIP{variant_suffix}",
                       pair_suffix=f"_CONTACT_MEMBERSHIP{variant_suffix}")
    # Log membership: -ln(f / f_max) — emphasises rare contacts and apparent
    # contact free energy relative to the strongest cell across the matrix.
    f_max_global = max((float(np.nanmax(v)) if np.any(np.isfinite(v)) else 0.0) for v in f_dict.values())
    nlnf_dict = {}
    for key, v in f_dict.items():
        a = np.asarray(v, dtype=float)
        out = np.full_like(a, np.nan, dtype=float)
        mask = np.isfinite(a) & (a > 0) & (f_max_global > 0)
        out[mask] = -np.log(a[mask] / f_max_global)
        nlnf_dict[key] = out
    nlnf_vmax = max((float(np.nanmax(v)) if np.any(np.isfinite(v)) else 0.0) for v in nlnf_dict.values())
    _plot_pcontact_full(nlnf_dict, maps_dir, bio_list, bio_lengths,
                       cmap_name="Reds_r", cbar_vmin=0.0, cbar_vmax=max(nlnf_vmax, 1e-6),
                       full_basename=f"Domain_Contact_Map_FULL_FULL_CONTACT_MEMBERSHIP_LOG{variant_suffix}",
                       pair_suffix=f"_CONTACT_MEMBERSHIP_LOG{variant_suffix}")

    # --- Enrichment: linear E (coolwarm centered at 1) and log ln(E) ---
    E_dict = {}
    for key, ln_val in lnE_dict.items():
        a = np.asarray(ln_val, dtype=float)
        out = np.full_like(a, np.nan, dtype=float)
        m = np.isfinite(a)
        out[m] = np.exp(a[m])
        E_dict[key] = out
    # Linear-E vmax: cap at 2 * max(median enrichment) for clarity; same
    # interpretive range as the acid linear-E map (1.0 = random).
    E_vmax_data = max((float(np.nanmax(v)) if np.any(np.isfinite(v)) else 1.0) for v in E_dict.values())
    E_panel = max(2.0, min(E_vmax_data, 10.0))
    _plot_pcontact_full(E_dict, maps_dir, bio_list, bio_lengths,
                       cmap_name="coolwarm", cbar_vmin=0.0, cbar_vmax=E_panel,
                       full_basename=f"Domain_Contact_Map_FULL_FULL_CONTACT_ENRICHMENT{variant_suffix}",
                       pair_suffix=f"_CONTACT_ENRICHMENT{variant_suffix}")
    _plot_pcontact_full(lnE_dict, maps_dir, bio_list, bio_lengths,
                       cmap_name="coolwarm", cbar_vmin=-lnE_abs_max, cbar_vmax=lnE_abs_max,
                       full_basename=f"Domain_Contact_Map_FULL_FULL_CONTACT_ENRICHMENT_LOG{variant_suffix}",
                       pair_suffix=f"_CONTACT_ENRICHMENT_LOG{variant_suffix}")

    return f_dict, lnE_dict


def _generate_membership_enrichment_difference_maps(
        f_dsm: dict, lnE_dsm: dict, f_ndsm: dict, lnE_ndsm: dict,
        out_dir: str, bio_list: list, bio_lengths: list,
        variant_suffix: str) -> None:
    """Δf (linear, coolwarm @0) and Δln(E) (linear in log domain,
    coolwarm @0) for the DSM-vs-NDSM contrast."""
    common = set(f_dsm.keys()) & set(f_ndsm.keys())
    if not common:
        return
    df_dict = {}
    dlnE_dict = {}
    for key in common:
        a = np.asarray(f_dsm[key], dtype=float)
        b = np.asarray(f_ndsm[key], dtype=float)
        df_dict[key] = a - b
        la = np.asarray(lnE_dsm.get(key), dtype=float)
        lb = np.asarray(lnE_ndsm.get(key), dtype=float)
        out = np.full_like(la, np.nan, dtype=float)
        mask = np.isfinite(la) & np.isfinite(lb)
        out[mask] = la[mask] - lb[mask]
        dlnE_dict[key] = out
    df_max = _dict_sym_max(df_dict)
    dlnE_max = _dict_sym_max(dlnE_dict)
    # Linear delta-membership and delta-log-enrichment (canonical paper variants)
    _plot_pcontact_full(df_dict, out_dir, bio_list, bio_lengths,
                       cmap_name="coolwarm", cbar_vmin=-df_max, cbar_vmax=df_max,
                       full_basename=f"Domain_Contact_Map_FULL_FULL_DIFF_CONTACT_MEMBERSHIP{variant_suffix}",
                       pair_suffix=f"_DIFF_CONTACT_MEMBERSHIP{variant_suffix}")
    _plot_pcontact_full(dlnE_dict, out_dir, bio_list, bio_lengths,
                       cmap_name="coolwarm", cbar_vmin=-dlnE_max, cbar_vmax=dlnE_max,
                       full_basename=f"Domain_Contact_Map_FULL_FULL_DIFF_CONTACT_ENRICHMENT_LOG{variant_suffix}",
                       pair_suffix=f"_DIFF_CONTACT_ENRICHMENT_LOG{variant_suffix}")
    # Log-ratio of membership (SI variant): ln(f_DSM / f_NDSM). Stays
    # symmetric around 0 in log space; emphasises fold-change in rare contacts.
    lnratio_f_dict = {}
    for key in common:
        a = np.asarray(f_dsm[key], dtype=float)
        b = np.asarray(f_ndsm[key], dtype=float)
        out = np.full_like(a, np.nan, dtype=float)
        m = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
        out[m] = np.log(a[m] / b[m])
        lnratio_f_dict[key] = out
    lnratio_max = _dict_sym_max(lnratio_f_dict)
    _plot_pcontact_full(lnratio_f_dict, out_dir, bio_list, bio_lengths,
                       cmap_name="coolwarm", cbar_vmin=-lnratio_max, cbar_vmax=lnratio_max,
                       full_basename=f"Domain_Contact_Map_FULL_FULL_DIFF_CONTACT_MEMBERSHIP_LOG{variant_suffix}",
                       pair_suffix=f"_DIFF_CONTACT_MEMBERSHIP_LOG{variant_suffix}")
    # Linear Δ-enrichment: ΔE = E_DSM - E_NDSM (companion to Δln(E))
    dE_dict = {}
    for key in common:
        a = np.asarray(lnE_dsm.get(key), dtype=float)
        b = np.asarray(lnE_ndsm.get(key), dtype=float)
        out = np.full_like(a, np.nan, dtype=float)
        m = np.isfinite(a) & np.isfinite(b)
        out[m] = np.exp(a[m]) - np.exp(b[m])
        dE_dict[key] = out
    dE_max = _dict_sym_max(dE_dict)
    _plot_pcontact_full(dE_dict, out_dir, bio_list, bio_lengths,
                       cmap_name="coolwarm", cbar_vmin=-dE_max, cbar_vmax=dE_max,
                       full_basename=f"Domain_Contact_Map_FULL_FULL_DIFF_CONTACT_ENRICHMENT{variant_suffix}",
                       pair_suffix=f"_DIFF_CONTACT_ENRICHMENT{variant_suffix}")


def _gaussian_kernel1d(sigma: float, radius: Optional[int] = None) -> np.ndarray:
    """Return a normalized 1D Gaussian kernel for KDE-like smoothing on a grid."""
    if sigma is None or sigma <= 0:
        return np.array([1.0], dtype=float)
    if radius is None:
        radius = max(1, int(round(3.0 * float(sigma))))
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5 * (x / float(sigma)) ** 2)
    s = k.sum()
    if s > 0:
        k /= s
    return k


def _kde_smooth_1d(y: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """Gaussian KDE on a uniform grid implemented via convolution.

    - y is treated as a histogram over equally spaced positions (residues).
    - sigma in units of grid steps (residues).
    - Preserves non-negativity and renormalizes to unit area.
    """
    if y.size == 0:
        return y
    k = _gaussian_kernel1d(sigma)
    r = (len(k) - 1) // 2
    y = np.asarray(y, dtype=float)
    ypad = np.pad(y, pad_width=r, mode='reflect')
    ys = np.convolve(ypad, k, mode='valid')
    ys = np.clip(ys, a_min=0.0, a_max=None)
    s = ys.sum()
    if s > 0:
        ys = ys / s
    return ys


def _plot_stacked_contact_probabilities(
    full_mat: np.ndarray,
    output_path: str,
    bio_list: list[str],
    bio_lengths: list[int],
    normalize: Literal['area', 'max', 'mean_per_res'] = 'area',
    kde_sigma: float = 5.0,
) -> None:
    """Plot 7 stacked 1D contact probability profiles from FULL_FULL.

    - For each species i, collapse the FULL_FULL matrix across the species-i
      rows to obtain a 1D profile across the concatenated x-axis.
    - Smooth, normalize to highlight spikes, and stack with +1 vertical offset.
    - Fill the area with translucent color and draw species boundaries/labels.
    """
    sns.set_theme(style="ticks")
    sns.set_style('white')
    plt.rc('axes', titlesize=10)
    plt.rc('axes', labelsize=10)
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('legend', fontsize=8)
    plt.rc('font', size=10)
    plt.rc('axes', linewidth=2)

    total_len = int(np.sum(np.asarray(bio_lengths)))
    if full_mat.shape[0] != total_len or full_mat.shape[1] != total_len:
        raise ValueError("FULL_FULL matrix dimensions do not match bio_lengths sum")

    offsets = np.cumsum([0] + list(bio_lengths))
    x = np.arange(total_len, dtype=float)

    # Colors consistent with BIOPOLYMER_ANALYSIS stacked plots
    col_pall = sns.color_palette("rocket", n_colors=14)
    col_rna = sns.color_palette(["#0066ff"], 1)[0]
    # Map to BIOPOLYMER_LIST order
    color_map = {
        "ProteinTDP43": col_pall[0],
        "ProteinFUS":   col_pall[2],
        "ProteinTIA1":  col_pall[4],
        "ProteinG3BP1": col_pall[6],
        "ProteinPABP1": col_pall[8],
        "ProteinTTP":   col_pall[10],
        "RNA":          col_rna,
    }

    # Single axes with vertically offset rows (touching) in 0.0015 increments
    fig = plt.figure(figsize=_PLOT_FIG_SIZE)
    ax = fig.add_axes(_AX_RECT)
    ax.set_box_aspect(1)

    band_height = 3.0e-3
    ymin = 0.0
    ymax = band_height * float(len(bio_list))
    ax.set_ylim(ymin, ymax)

    # Compute and plot each species profile with offset by k*1e-3 (bottom to top)
    for idx, bio in enumerate(bio_list):
        start = int(offsets[idx])
        end = int(offsets[idx + 1])  # exclusive
        prof = np.sum(full_mat[start:end, :], axis=0).astype(float)
        if normalize == 'mean_per_res':
            nres = max(1, end - start)
            prof = prof / float(nres)
        total = prof.sum()
        if total > 0:
            prof /= total
        # Smooth using Gaussian KDE on the discrete residue grid
        prof_s = _kde_smooth_1d(prof, sigma=float(kde_sigma))
        if normalize == 'max':
            m = float(np.max(prof_s)) if prof_s.size else 0.0
            if m > 0:
                prof_s = prof_s / m

        band_index = len(bio_list) - 1 - idx
        base = band_index * band_height
        col = color_map.get(bio, '#333333')
        ax.fill_between(x, base, base + prof_s, color=col, alpha=0.35, linewidth=0)
        ax.plot(x, base + prof_s, color=col, linewidth=1.0)

    # Species block boundaries spanning full stacked height (on top of all content)
    for b in offsets:
        ax.axvline(b - 0.5, color='k', linewidth=0.4, zorder=1000)

    # Black horizontal lines at every 0.001
    for k in range(0, len(bio_list) + 1):
        y = k * band_height
        ax.axhline(y, color='black', linewidth=0.4, linestyle='-', zorder=90)

    # X ticks at species centers with labels centered on blocks
    centers = []
    labels = []
    for i, bio in enumerate(bio_list):
        start = int(offsets[i])
        end = int(offsets[i + 1]) - 1
        centers.append((start + end) / 2.0)
        labels.append(bio.replace("Protein", ""))
    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation=45, ha='center', color='black')

    # X/Y limits
    ax.set_xlim(-0.5, total_len - 0.5)
    yticks = [k * band_height for k in range(0, len(bio_list) + 1)]

    # All borders present
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(2)
    ax.tick_params(axis='both', which='major', pad=1, direction='out', length=4, width=2,
                   colors='black', labelsize=8)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='minor', direction='out', length=2, width=2, colors='black')
    # Suppress y-axis ticks on the data axis itself; handled by dedicated twin axes
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
    # Completely remove yticks from main axis to prevent any numeric labels
    ax.set_yticks([])
    # Ensure x-axis ticks are visible on bottom
    ax.tick_params(axis='x', which='major', bottom=True, top=False, labelbottom=True, labeltop=False)

    fig.canvas.draw()

    # Right y-axis for numeric probability ticks
    ax_right = ax.twinx()
    ax_right.set_ylim(ax.get_ylim())
    ax_right.set_yticks(yticks)
    ax_right.set_yticklabels([('{:.1f}'.format(v * 1e3)).rstrip('0').rstrip('.') for v in yticks])
    ax_right.minorticks_on()
    ax_right.tick_params(axis='y', which='major', pad=1, direction='out', length=4, width=2,
                         colors='black', labelsize=8, left=False, right=True, labelleft=False, labelright=True)
    ax_right.tick_params(axis='y', which='minor', direction='out', length=2, width=2,
                         colors='black', left=False, right=True)
    ax_right.spines['left'].set_visible(False)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['bottom'].set_visible(False)
    # Ensure right spine thickness matches data axis (but avoid double drawing)
    ax_right.spines['right'].set_visible(False)

    ax_right.text(1.02, 1.02, r"x10$^{-3}$", transform=ax_right.transAxes,
                  ha='left', va='bottom', color='black', fontsize=8)

    # Left axis for biopolymer names centered on rows - create new empty axis
    ax_pos = ax.get_position()
    ax_left = fig.add_axes([ax_pos.x0, ax_pos.y0, ax_pos.width, ax_pos.height], frameon=False)
    ax_left.set_xlim(ax.get_xlim())
    ax_left.set_ylim(ax.get_ylim())

    # Calculate protein label positions
    centers_y = [((len(bio_list) - 1 - i) + 0.5) * band_height for i in range(len(bio_list))]

    # Set only protein tick marks and labels explicitly
    ax_left.set_yticks(centers_y)
    ax_left.set_yticklabels([b.replace("Protein", "") for b in bio_list])

    # Configure tick parameters - only left ticks, no numeric ticks
    ax_left.tick_params(axis='y', which='major', pad=1, direction='out', length=4, width=2,
                        colors='black', labelsize=8, left=True, right=False, labelleft=True, labelright=False)
    ax_left.tick_params(axis='y', which='minor', left=False, right=False)
    ax_left.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=False)
    ax_left.minorticks_off()

    # Rotate labels
    for t in ax_left.get_yticklabels():
        t.set_rotation(45)
        t.set_va('center')
        t.set_ha('right')

    # Hide all spines
    for spine in ax_left.spines.values():
        spine.set_visible(False)

    # Domain rectangles (aligned with main axes) and global multiplier label
    try:
        full_bars = []
        for i, bio in enumerate(bio_list):
            offset = int(offsets[i])
            bars = _flip_rna_bars(_bars_for_biopolymer(bio, bio_lengths[i]), bio, bio_lengths[i])
            for s0, e0, color in bars:
                full_bars.append((offset + s0, offset + e0, color))
        bar_ax = _add_domain_bar(fig, ax, full_bars, offsets, spine_linewidth=2)
    except Exception as e:
        print(f"Warning: failed to add domain bar in probability plot: {e}")
        import traceback
        traceback.print_exc()

    fig.savefig(output_path, format='png', dpi=400)
    plt.close(fig)


def _save_difference_heatmap_with_csv(diff_array: np.ndarray, out_dir: str, stem: str,
                                      full_bars: list, offsets: list, centers: list, labels: list,
                                      symmetric_limit: Optional[float] = None,
                                      robust_percentile: Optional[float] = None) -> None:
    """Render a FULL_FULL difference heatmap and save the matching {stem}.png and {stem}.csv into *out_dir*."""
    output_path = os.path.join(out_dir, f"{stem}.png")
    _plot_difference_heatmap(
        diff_array, output_path, full_bars, offsets, centers, labels,
        symmetric_limit=symmetric_limit,
        robust_percentile=robust_percentile,
    )
    np.savetxt(os.path.join(out_dir, f"{stem}.csv"), diff_array, delimiter=",")


def _normalized_difference(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Elementwise numerator/denominator, returning 0 where the denominator is exactly 0."""
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    )


def _contact_positive_floor(*arrays: np.ndarray, percentile: float = 1.0) -> float:
    """Small data-driven pseudocount for contact ratios."""
    positives = []
    for arr in arrays:
        a = np.asarray(arr, dtype=float)
        finite_positive = a[np.isfinite(a) & (a > 0)]
        if finite_positive.size:
            positives.append(finite_positive)
    if not positives:
        return 1e-12
    floor = float(np.nanpercentile(np.concatenate(positives), percentile))
    return max(floor, 1e-12)


def _masked_log2_fold_change(numerator: np.ndarray, denominator: np.ndarray,
                             pseudocount: float) -> np.ndarray:
    """log2((numerator + eps)/(denominator + eps)), masking no-contact pixels."""
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        log2fc = np.log2((num + pseudocount) / (den + pseudocount))
    low_contact = np.maximum(num, den) <= pseudocount
    log2fc[low_contact | ~np.isfinite(log2fc)] = np.nan
    return log2fc


def _generate_difference_plots(combined_matrices: dict, out_dir: str, bio_list: list, bio_lengths: list) -> None:
    """Generate normalized and unnormalized FULL_FULL difference plots.

    Outputs include explicit DSM-SG and NDSM-SG unnormalized differences,
    SG-normalized relative differences, and DSM-NDSM comparisons. Legacy
    filenames are preserved for backward compatibility.
    """
    sg_mat = combined_matrices["SG"]
    dsm_mat = combined_matrices["DSM"]
    ndsm_mat = combined_matrices["NDSM"]

    dsm_minus_sg = dsm_mat - sg_mat
    ndsm_minus_sg = ndsm_mat - sg_mat
    dsm_minus_ndsm = dsm_mat - ndsm_mat

    dsm_minus_sg_norm = _normalized_difference(dsm_minus_sg, sg_mat)
    ndsm_minus_sg_norm = _normalized_difference(ndsm_minus_sg, sg_mat)
    dsm_ndsm_over_sg = _normalized_difference(dsm_minus_ndsm, sg_mat)
    log2fc_dsm_over_ndsm = _masked_log2_fold_change(
        dsm_mat, ndsm_mat,
        _contact_positive_floor(sg_mat, dsm_mat, ndsm_mat, percentile=1.0),
    )

    offsets = np.cumsum([0] + bio_lengths)
    full_bars = []
    for i, bio in enumerate(bio_list):
        offset = int(offsets[i])
        bars = _flip_rna_bars(_bars_for_biopolymer(bio, bio_lengths[i]), bio, bio_lengths[i])
        for s0, e0, color in bars:
            full_bars.append((offset + s0, offset + e0, color))
    centers = []
    labels = []
    for i, bio in enumerate(bio_list):
        start = int(offsets[i])
        end = int(offsets[i+1]) - 1
        centers.append((start + end) / 2.0 + 0.5)
        labels.append(bio.replace("Protein", ""))
    # Backward-compatible legacy filenames (unnormalized for DSM/NDSM, normalized for DSM-NDSM over SG)
    _save_difference_heatmap_with_csv(dsm_minus_sg, out_dir, "Domain_Contact_Map_FULL_FULL_DIFF_DSM", full_bars, offsets, centers, labels)
    _save_difference_heatmap_with_csv(ndsm_minus_sg, out_dir, "Domain_Contact_Map_FULL_FULL_DIFF_NDSM", full_bars, offsets, centers, labels)
    _save_difference_heatmap_with_csv(dsm_ndsm_over_sg, out_dir, "Domain_Contact_Map_FULL_FULL_DIFF_DSM_NDSM_over_SG", full_bars, offsets, centers, labels)

    # Explicit unnormalized and normalized outputs
    _save_difference_heatmap_with_csv(dsm_minus_sg, out_dir, "Domain_Contact_Map_FULL_FULL_DIFF_DSM_UNNORMALIZED", full_bars, offsets, centers, labels)
    _save_difference_heatmap_with_csv(ndsm_minus_sg, out_dir, "Domain_Contact_Map_FULL_FULL_DIFF_NDSM_UNNORMALIZED", full_bars, offsets, centers, labels)
    _save_difference_heatmap_with_csv(dsm_minus_ndsm, out_dir, "Domain_Contact_Map_FULL_FULL_DIFF_DSM_NDSM_UNNORMALIZED", full_bars, offsets, centers, labels)

    _save_difference_heatmap_with_csv(dsm_minus_sg_norm, out_dir, "Domain_Contact_Map_FULL_FULL_DIFF_DSM_NORMALIZED", full_bars, offsets, centers, labels)
    _save_difference_heatmap_with_csv(ndsm_minus_sg_norm, out_dir, "Domain_Contact_Map_FULL_FULL_DIFF_NDSM_NORMALIZED", full_bars, offsets, centers, labels)
    _save_difference_heatmap_with_csv(dsm_ndsm_over_sg, out_dir, "Domain_Contact_Map_FULL_FULL_DIFF_DSM_NDSM_NORMALIZED", full_bars, offsets, centers, labels)

    _save_difference_heatmap_with_csv(
        log2fc_dsm_over_ndsm, out_dir,
        "Domain_Contact_Map_FULL_FULL_LOG2FC_DSM_over_NDSM",
        full_bars, offsets, centers, labels,
        symmetric_limit=3.0,
    )

    # Generate probability difference plots
    try:
        output_path = os.path.join(out_dir, "Domain_Contact_Probabilities_DIFF_DSM.png")
        _plot_probability_difference(sg_mat, dsm_mat, output_path, bio_list, bio_lengths, "DSM")
    except Exception as e:
        print(f"Warning: failed to generate DSM probability difference plot: {e}")

    try:
        output_path = os.path.join(out_dir, "Domain_Contact_Probabilities_DIFF_NDSM.png")
        _plot_probability_difference(sg_mat, ndsm_mat, output_path, bio_list, bio_lengths, "NDSM")
    except Exception as e:
        print(f"Warning: failed to generate NDSM probability difference plot: {e}")


def _write_difference_sem_csvs(combined_matrices: dict, combined_sem_matrices: dict, out_dir: str) -> None:
    """Propagate per-cell SEM through the SG/DSM/NDSM difference maps and save SEM + z-score CSVs.

    Requires SG, DSM and NDSM entries in both inputs; writes the error-propagated
    SEM grids for the unnormalized and SG-normalized differences, plus the DSM-vs-NDSM
    z-score CSV. Data-only writer (no figures are produced).
    """
    required = {"SG", "DSM", "NDSM"}
    if not required.issubset(combined_matrices) or not required.issubset(combined_sem_matrices):
        return
    sg = combined_matrices["SG"]
    dsm = combined_matrices["DSM"]
    ndsm = combined_matrices["NDSM"]
    sg_sem = combined_sem_matrices["SG"]
    dsm_sem = combined_sem_matrices["DSM"]
    ndsm_sem = combined_sem_matrices["NDSM"]

    dsm_minus_sg_sem = np.sqrt(dsm_sem ** 2 + sg_sem ** 2)
    ndsm_minus_sg_sem = np.sqrt(ndsm_sem ** 2 + sg_sem ** 2)
    dsm_minus_ndsm_sem = np.sqrt(dsm_sem ** 2 + ndsm_sem ** 2)

    with np.errstate(divide="ignore", invalid="ignore"):
        dsm_norm_sem = np.sqrt((dsm_sem / sg) ** 2 + (dsm * sg_sem / (sg ** 2)) ** 2)
        ndsm_norm_sem = np.sqrt((ndsm_sem / sg) ** 2 + (ndsm * sg_sem / (sg ** 2)) ** 2)
        dsm_ndsm_norm_sem = np.sqrt(
            (dsm_sem / sg) ** 2
            + (ndsm_sem / sg) ** 2
            + ((dsm - ndsm) * sg_sem / (sg ** 2)) ** 2
        )
    for arr in (dsm_norm_sem, ndsm_norm_sem, dsm_ndsm_norm_sem):
        arr[~np.isfinite(arr)] = np.nan

    np.savetxt(os.path.join(out_dir, "Domain_Contact_Map_FULL_FULL_DIFF_DSM_SEM.csv"), dsm_minus_sg_sem, delimiter=",")
    np.savetxt(os.path.join(out_dir, "Domain_Contact_Map_FULL_FULL_DIFF_NDSM_SEM.csv"), ndsm_minus_sg_sem, delimiter=",")
    np.savetxt(os.path.join(out_dir, "Domain_Contact_Map_FULL_FULL_DIFF_DSM_NDSM_UNNORMALIZED_SEM.csv"), dsm_minus_ndsm_sem, delimiter=",")
    np.savetxt(os.path.join(out_dir, "Domain_Contact_Map_FULL_FULL_DIFF_DSM_NDSM_over_SG_SEM.csv"), dsm_ndsm_norm_sem, delimiter=",")
    np.savetxt(os.path.join(out_dir, "Domain_Contact_Map_FULL_FULL_DIFF_DSM_NORMALIZED_SEM.csv"), dsm_norm_sem, delimiter=",")
    np.savetxt(os.path.join(out_dir, "Domain_Contact_Map_FULL_FULL_DIFF_NDSM_NORMALIZED_SEM.csv"), ndsm_norm_sem, delimiter=",")
    np.savetxt(os.path.join(out_dir, "Domain_Contact_Map_FULL_FULL_DIFF_DSM_NDSM_NORMALIZED_SEM.csv"), dsm_ndsm_norm_sem, delimiter=",")

    with np.errstate(divide="ignore", invalid="ignore"):
        z_dsm_ndsm = (dsm - ndsm) / dsm_minus_ndsm_sem
    z_dsm_ndsm[~np.isfinite(z_dsm_ndsm)] = np.nan
    np.savetxt(os.path.join(out_dir, "Domain_Contact_Map_FULL_FULL_ZSCORE_DSM_NDSM.csv"), z_dsm_ndsm, delimiter=",")


def _write_pair_zscore_csvs(pair_matrices: dict, pair_sem_matrices: dict,
                            out_dir: str, bio_list: list) -> None:
    """Write pairwise DSM-vs-NDSM z-score CSVs for standard domain contact maps.

    Data-only writer (no figures are produced)."""
    required = {"DSM", "NDSM"}
    if not required.issubset(pair_matrices) or not required.issubset(pair_sem_matrices):
        return

    dsm_pairs = pair_matrices["DSM"]
    ndsm_pairs = pair_matrices["NDSM"]
    dsm_sem_pairs = pair_sem_matrices["DSM"]
    ndsm_sem_pairs = pair_sem_matrices["NDSM"]

    for i, bio_i in enumerate(bio_list):
        for j in range(i, len(bio_list)):
            bio_j = bio_list[j]
            key = (bio_i, bio_j)
            if key not in dsm_pairs or key not in ndsm_pairs:
                continue
            if key not in dsm_sem_pairs or key not in ndsm_sem_pairs:
                continue
            dsm = np.asarray(dsm_pairs[key], dtype=float)
            ndsm = np.asarray(ndsm_pairs[key], dtype=float)
            dsm_sem = np.asarray(dsm_sem_pairs[key], dtype=float)
            ndsm_sem = np.asarray(ndsm_sem_pairs[key], dtype=float)
            if dsm.shape != ndsm.shape or dsm.shape != dsm_sem.shape or dsm.shape != ndsm_sem.shape:
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                z = (dsm - ndsm) / np.sqrt(dsm_sem ** 2 + ndsm_sem ** 2)
            z[~np.isfinite(z)] = np.nan
            name_i = bio_i.replace("Protein", "")
            name_j = bio_j.replace("Protein", "")
            stem = f"Domain_Contact_Map_{name_i}_{name_j}_ZSCORE_DSM_NDSM"
            np.savetxt(os.path.join(out_dir, f"{stem}.csv"), z, delimiter=",")


def _plot_difference_heatmap(diff_array: np.ndarray, output_path: str, full_bars: list,
                              offsets: list, centers: list, labels: list,
                              symmetric_limit: Optional[float] = None,
                              robust_percentile: Optional[float] = None) -> None:
    """Plot a difference heatmap with coolwarm diverging colormap."""
    sns.set_theme(style="ticks")
    sns.set_style('white')
    plt.rc('axes', titlesize=8)
    plt.rc('axes', labelsize=8)
    plt.rc('xtick', labelsize=_DOMAIN_PAIR_TICK_FONTSIZE)
    plt.rc('ytick', labelsize=_DOMAIN_PAIR_TICK_FONTSIZE)
    plt.rc('legend', fontsize=8)
    plt.rc('font', size=8)
    plt.rc('axes', linewidth=2)

    diff_plot = np.asarray(diff_array, dtype=float).copy()
    finite = np.isfinite(diff_plot)
    if symmetric_limit is not None:
        max_abs = float(symmetric_limit)
    elif robust_percentile is not None and np.any(finite):
        max_abs = float(np.nanpercentile(np.abs(diff_plot[finite]), robust_percentile))
    else:
        max_abs = np.max(np.abs(diff_plot[finite])) if np.any(finite) else 1.0
    if not np.isfinite(max_abs) or max_abs <= 0:
        max_abs = 1.0
    vmin, vmax = -max_abs, max_abs

    fig_size, ax_rect = _contact_map_layout(
        diff_plot.shape,
        square=True,
        has_x_bar=True,
        has_y_bar=True,
        show_colorbar=True,
    )
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_axes(ax_rect)

    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color="lightgrey")
    ax = sns.heatmap(
        diff_plot,
        cmap=cmap,
        square=True,
        ax=ax,
        cbar=False,
        mask=~np.isfinite(diff_plot),
        vmin=vmin, vmax=vmax,
        center=0
    )
    ax.set_box_aspect(1)

    for side in ["top", "bottom", "left", "right"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color('black')
        ax.spines[side].set_linewidth(2)

    # Match FULL_FULL contact maps: draw biopolymer block separators at the
    # concatenated species boundaries so domain-level DIFF maps are readable.
    if offsets is not None:
        for b in sorted(set(np.asarray(offsets, dtype=float))):
            ax.axvline(b - 0.5, color='k', linewidth=0.4, zorder=3)
            ax.axhline(b - 0.5, color='k', linewidth=0.4, zorder=3)

    pos = ax.get_position()
    cbar_ax = fig.add_axes([
        _cbar_left_after_right_bar(ax, True),
        pos.y0,
        _CBAR_WIDTH_IN / fig.get_figwidth(),
        pos.height,
    ])
    mappable = ax.collections[0]
    tick_lo = _round_2sig_floor(vmin)
    tick_hi = _round_2sig_ceil(vmax)
    if abs(tick_hi - tick_lo) < 1e-30:
        tick_hi = tick_lo + 1.0
    ticks = np.linspace(tick_lo, tick_hi, 5)
    mappable.set_clim(tick_lo, tick_hi)
    cbar = fig.colorbar(mappable, cax=cbar_ax)
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
    cbar.ax.text(0.5, 1.01, f"$\\times10^{{{common_exp}}}$",
        transform=cbar.ax.transAxes, ha='center', va='bottom', fontsize=_DOMAIN_CBAR_FONTSIZE)
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(2)
    cbar.ax.tick_params(labelsize=_DOMAIN_CBAR_FONTSIZE, width=2, length=4)
    for spine_name in ['top', 'bottom', 'left', 'right']:
        cbar.ax.spines[spine_name].set_visible(True)
        cbar.ax.spines[spine_name].set_color('black')
        cbar.ax.spines[spine_name].set_linewidth(2)

    ax.set_xticks(centers)
    ax.set_yticks(centers)
    ax.set_xticklabels(labels, rotation=45, ha='center', va='top',
                       rotation_mode='default', fontsize=_DOMAIN_FULL_TICK_FONTSIZE)
    ax.set_yticklabels(labels, rotation=45, ha='right', va='center', rotation_mode='default')
    ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True,
                   top=False, labeltop=False)
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_label_position('bottom')
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontsize(_DOMAIN_FULL_TICK_FONTSIZE)
    ax.tick_params(axis='both', which='major', pad=1, direction='out', length=4, width=2)
    ax.tick_params(axis='y', left=True, labelleft=True, right=False, labelright=False)
    ax.yaxis.set_ticks_position('left')

    # Invert y-axis so FUS sits at bottom-left and RNA at top-right
    ax.invert_yaxis()

    # Add domain bars
    try:
        xbar_ax = _add_domain_bar(fig, ax, full_bars, offsets, orientation='top', align='top')
        ybar_ax = _add_domain_bar(fig, ax, full_bars, offsets, orientation='right', flip_axis=True)
    except Exception:
        pass

    fig.savefig(output_path, format='png', dpi=400)
    plt.close(fig)


def _plot_probability_difference(sg_mat: np.ndarray, sm_mat: np.ndarray, output_path: str,
                                  bio_list: list, bio_lengths: list, sm_name: str) -> None:
    """Plot probability differences overlaid on same axis (not stacked), with +1 offset.

    Similar to biopolymer_analysis.py raw diff plots but for contact probabilities.
    """
    sns.set_theme(style="ticks")
    sns.set_style('white')
    plt.rc('axes', titlesize=10)
    plt.rc('axes', labelsize=10)
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('legend', fontsize=8)
    plt.rc('font', size=10)
    plt.rc('axes', linewidth=2)

    total_len = int(np.sum(np.asarray(bio_lengths)))
    offsets = np.cumsum([0] + bio_lengths)
    x = np.arange(total_len, dtype=float)

    # Colors
    col_pall = sns.color_palette("rocket", n_colors=14)
    col_rna = sns.color_palette(["#0066ff"], 1)[0]
    color_map = {
        "ProteinTDP43": col_pall[0],
        "ProteinFUS":   col_pall[2],
        "ProteinTIA1":  col_pall[4],
        "ProteinG3BP1": col_pall[6],
        "ProteinPABP1": col_pall[8],
        "ProteinTTP":   col_pall[10],
        "RNA":          col_rna,
    }

    fig = plt.figure(figsize=_PLOT_FIG_SIZE)
    ax = fig.add_axes(_AX_RECT)
    ax.set_box_aspect(1)

    # Compute difference profiles for each species (SM - SG) with +1 offset
    for idx, bio in enumerate(bio_list):
        start = int(offsets[idx])
        end = int(offsets[idx + 1])
        sg_prof = np.sum(sg_mat[start:end, :], axis=0).astype(float)
        sm_prof = np.sum(sm_mat[start:end, :], axis=0).astype(float)

        # Normalize
        nres = max(1, end - start)
        sg_prof = sg_prof / float(nres)
        sm_prof = sm_prof / float(nres)
        total_sg = sg_prof.sum()
        total_sm = sm_prof.sum()
        if total_sg > 0:
            sg_prof /= total_sg
        if total_sm > 0:
            sm_prof /= total_sm

        # Compute difference and add offset of 1
        diff_prof = sm_prof - sg_prof + 1.0  # +1 offset to keep positive

        # Smooth
        diff_prof_s = _kde_smooth_1d(diff_prof, sigma=5.0)

        col = color_map.get(bio, '#333333')
        label = bio.replace("Protein", "")
        ax.plot(x, diff_prof_s, color=col, linewidth=1.5, label=label, alpha=0.8)

    # Species block boundaries
    for b in offsets:
        ax.axvline(b - 0.5, color='k', linewidth=0.4, zorder=1000)

    # Horizontal reference line at y=1 (zero difference)
    ax.axhline(y=1.0, color='grey', linewidth=1, linestyle='--', zorder=0, alpha=0.6)

    # X ticks
    centers = []
    tick_labels = []
    for i, bio in enumerate(bio_list):
        start = int(offsets[i])
        end = int(offsets[i + 1]) - 1
        centers.append((start + end) / 2.0)
        tick_labels.append(bio.replace("Protein", ""))
    ax.set_xticks(centers)
    ax.set_xticklabels(tick_labels, rotation=45, ha='center', color='black')
    ax.set_xlim(-0.5, total_len - 0.5)
    ax.set_ylim(0, 2.5)  # 1 ± some range for differences

    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(2)

    ax.tick_params(axis='both', which='major', pad=1, direction='out', length=4, width=2,
                   colors='black', labelsize=8)
    ax.legend(loc='best', fontsize=8, frameon=False)

    fig.savefig(output_path, format='png', dpi=400)
    plt.close(fig)


def run_aggregate_mode(base_path: str, tmin: int, dt: int, tmax: int, perform_avg: bool = True) -> None:
    """Average Domain_Contacts over time for SG/DSM/NDSM, aggregate by category, and plot heatmaps.

    - Writes time-averaged matrices into ANALYSIS_*_AVE
    - Writes aggregate heatmaps under FIGURES/DOMAIN_CONTACT_MAPS/<CATEGORY>
    """
    # Read inputs from current working directory (ANALYSIS_* live here)
    cwd = os.getcwd()
    categories = {
        "SG": os.path.join(cwd, "ANALYSIS_SG"),
        "DSM": os.path.join(cwd, "ANALYSIS_DSM"),
        "NDSM": os.path.join(cwd, "ANALYSIS_NDSM"),
    }

    # First pass: determine which categories will actually be processed
    categories_to_process = []
    for category, category_dir in categories.items():
        if not os.path.isdir(category_dir):
            continue
        # Check if there are actually domain contact files
        if perform_avg:
            sm_list = _discover_sms_with_domain_contacts(category_dir)
            if category == "SG" and not sm_list and os.path.isdir(category_dir):
                maybe_csv = os.path.join(category_dir, "Domain_Contacts_Total_sg_X_Protein*_*_*.csv")
                maybe_npz = os.path.join(category_dir, "Domain_Contacts_Total_sg_X_Protein*_*_*.csv.npz")
                if glob(maybe_csv) or glob(maybe_npz):
                    sm_list = ["sg_X"]
        else:
            ave_dir_existing = os.path.join(base_path, f"ANALYSIS_{category}_AVE")
            sm_list = _discover_sms_from_ave_dir(ave_dir_existing)

        if sm_list:
            categories_to_process.append(category)

    all_three_present = len(categories_to_process) == 3

    # Storage for combined matrices when generating difference plots
    combined_matrices = {}
    combined_sem_matrices = {}
    pair_matrices = {}
    pair_sem_matrices = {}
    # Storage for C_total matrices (= agg_dict raw counts) for C-space DIFF maps
    c_dict_by_category: dict = {}
    c_sem_dict_by_category: dict = {}
    # Storage for membership f and log-enrichment ln(E) per filter for DIFF maps
    membership_by_category: dict = {"ALL": {}, "INTER": {}, "INTRA": {}}
    log_enrichment_by_category: dict = {"ALL": {}, "INTER": {}, "INTRA": {}}
    # Storage for raw INTER and INTRA agg dicts (used to construct ALL = INTER + INTRA)
    inter_agg_by_category: dict = {}
    intra_agg_by_category: dict = {}
    # Parallel SEM storage per filter
    inter_sem_by_category: dict = {}
    intra_sem_by_category: dict = {}

    for category, category_dir in categories.items():
        if not os.path.isdir(category_dir):
            print(f"Skipping {category}: missing folder {category_dir}")
            continue

        # Discover SMs
        if perform_avg:
            sm_list = _discover_sms_with_domain_contacts(category_dir)
            # SG fallback: if none found but SG folder exists, try default tag
            if category == "SG" and not sm_list and os.path.isdir(category_dir):
                maybe_csv = os.path.join(category_dir, "Domain_Contacts_Total_sg_X_Protein*_*_*.csv")
                maybe_npz = os.path.join(category_dir, "Domain_Contacts_Total_sg_X_Protein*_*_*.csv.npz")
                if glob(maybe_csv) or glob(maybe_npz):
                    sm_list = ["sg_X"]
        else:
            ave_dir_existing = os.path.join(base_path, f"ANALYSIS_{category}_AVE")
            sm_list = _discover_sms_from_ave_dir(ave_dir_existing)

        if not sm_list:
            print(f"Skipping {category}: no Domain_Contacts files found")
            continue

        print(f"Processing {category}: {len(sm_list)} system(s)")

        # Output directories (must already exist for this run)
        ave_dir = os.path.join(base_path, f"ANALYSIS_{category}_AVE")
        maps_dir = os.path.join(base_path, "FIGURES/DOMAIN_CONTACT_MAPS", category)
        if not os.path.isdir(ave_dir):
            print(f"Skipping {category}: missing output dir {ave_dir} (will not create new AVE dir)")
            continue
        # Ensure category maps subfolder exists under existing FIGURES/DOMAIN_CONTACT_MAPS root
        if not os.path.isdir(maps_dir):
            try:
                os.makedirs(maps_dir, exist_ok=True)
                print(f"Created maps dir: {maps_dir}")
            except Exception as e:
                print(f"Skipping {category}: failed to create maps dir {maps_dir}: {e}")
                continue

        per_sm_dicts = []
        per_sm_sem_dicts = []  # parallel list of per-SM time-window SEMs
        if perform_avg:
            # Time-average per SM
            for sm in tqdm(sm_list, desc=f"{category} time-average"):
                sm_dict = {}
                sm_sem_dict = {}
                for i, bio_i in enumerate(BIOPOLYMER_LIST):
                    p = BIOPOLYMER_LENGTHS[i]
                    for j in range(i, len(BIOPOLYMER_LIST)):
                        bio_j = BIOPOLYMER_LIST[j]
                        q = BIOPOLYMER_LENGTHS[j]
                        mat, sem, count = _load_time_averaged_pair(category_dir, sm, bio_i, bio_j, p, q, tmin, dt, tmax)
                        if count > 0:
                            sm_dict[(bio_i, bio_j)] = mat
                            sm_sem_dict[(bio_i, bio_j)] = sem
                            _save_pair_csv(ave_dir, sm, bio_i, bio_j, mat)
                            _save_pair_csv(ave_dir, sm, bio_i, bio_j, mat, prefix="Domain_Contacts_MeanCombinedFiltered_")
                            _save_pair_csv(ave_dir, sm, bio_i, bio_j, sem, prefix="Domain_Contacts_SEM_")
                            _save_pair_csv(ave_dir, sm, bio_i, bio_j, sem, prefix="Domain_Contacts_SEMCombinedFiltered_")
                if sm_dict:
                    per_sm_dicts.append(sm_dict)
                    per_sm_sem_dicts.append(sm_sem_dict)
        else:
            # Load pre-averaged per-SM matrices (and SEMs) from AVE dir
            for sm in tqdm(sm_list, desc=f"{category} load-averaged"):
                sm_dict = {}
                sm_sem_dict = {}
                for i, bio_i in enumerate(BIOPOLYMER_LIST):
                    for j in range(i, len(BIOPOLYMER_LIST)):
                        bio_j = BIOPOLYMER_LIST[j]
                        mat = None
                        for mean_prefix in _mean_prefixes_for_pair(bio_i, bio_j):
                            fpath = os.path.join(ave_dir, f"{mean_prefix}{sm}_{bio_i}_{bio_j}.csv")
                            alt = os.path.join(ave_dir, f"{mean_prefix}{sm}_{bio_j}_{bio_i}.csv")
                            if os.path.exists(fpath):
                                try:
                                    mat = np.loadtxt(fpath, delimiter=",")
                                except Exception:
                                    mat = None
                            elif os.path.exists(alt):
                                try:
                                    mat = np.loadtxt(alt, delimiter=",")
                                    # transpose if needed
                                    if mat.shape == (BIOPOLYMER_LENGTHS[j], BIOPOLYMER_LENGTHS[i]):
                                        mat = mat.T
                                except Exception:
                                    mat = None
                            if mat is not None:
                                break
                        if mat is not None and mat.size > 0:
                            sm_dict[(bio_i, bio_j)] = mat
                            # Also try to load matching SEM file (NEW)
                            for sem_prefix in ("Domain_Contacts_SEMCombinedFiltered_", "Domain_Contacts_SEM_"):
                                fpath_s = os.path.join(ave_dir, f"{sem_prefix}{sm}_{bio_i}_{bio_j}.csv")
                                alt_s = os.path.join(ave_dir, f"{sem_prefix}{sm}_{bio_j}_{bio_i}.csv")
                                sem_mat = None
                                if os.path.exists(fpath_s):
                                    try:
                                        sem_mat = np.loadtxt(fpath_s, delimiter=",")
                                    except Exception:
                                        sem_mat = None
                                elif os.path.exists(alt_s):
                                    try:
                                        sem_mat = np.loadtxt(alt_s, delimiter=",")
                                        if sem_mat.shape == (BIOPOLYMER_LENGTHS[j], BIOPOLYMER_LENGTHS[i]):
                                            sem_mat = sem_mat.T
                                    except Exception:
                                        sem_mat = None
                                if sem_mat is not None and np.any(np.isfinite(sem_mat)):
                                    sm_sem_dict[(bio_i, bio_j)] = sem_mat
                                    break
                if sm_dict:
                    per_sm_dicts.append(sm_dict)
                    per_sm_sem_dicts.append(sm_sem_dict)

        # Aggregate across SMs for this category
        if not per_sm_dicts:
            print(f"No per-system matrices were generated for {category}. Skipping aggregation/plots.")
            continue

        agg_dict, agg_sem_dict = _aggregate_over_sms_with_sem(
            per_sm_dicts, BIOPOLYMER_LIST, BIOPOLYMER_LENGTHS,
            per_sm_sem_dicts=per_sm_sem_dicts if any(per_sm_sem_dicts) else None,
        )
        # Save aggregated matrices
        for (bio_i, bio_j), mat in agg_dict.items():
            _save_pair_csv(ave_dir, category, bio_i, bio_j, mat)
            _save_pair_csv(ave_dir, category, bio_i, bio_j, mat, prefix="Domain_Contacts_MeanCombinedFiltered_")
        for (bio_i, bio_j), sem in agg_sem_dict.items():
            _save_pair_csv(ave_dir, category, bio_i, bio_j, sem, prefix="Domain_Contacts_SEM_")
            _save_pair_csv(ave_dir, category, bio_i, bio_j, sem, prefix="Domain_Contacts_SEMCombinedFiltered_")

        # Plot aggregate heatmaps and combined matrix
        show_colorbar = True
        show_yticks = True
        combined_array = _plot_from_contact_dict(agg_dict, maps_dir, BIOPOLYMER_LIST, BIOPOLYMER_LENGTHS,
                                                 show_colorbar=show_colorbar, show_yticks=show_yticks)
        if agg_sem_dict:
            combined_sem_array = _combine_to_symmetric(agg_sem_dict, BIOPOLYMER_LIST, BIOPOLYMER_LENGTHS)
            np.savetxt(os.path.join(maps_dir, "Domain_Contact_Map_FULL_FULL_SEM.csv"), combined_sem_array, delimiter=",")

        # Chain-pair normalization table (BioPolNum_*.csv) used by the
        # membership/enrichment maps below. No RCC re-run required.
        biopolnum_tag = "sg_X" if category == "SG" else category
        biopolnum_csv = os.path.join(ave_dir, f"BioPolNum_{biopolnum_tag}.csv")
        if not os.path.exists(biopolnum_csv):
            print(f"[INFO] {biopolnum_csv} not found; skipping membership/enrichment maps for {category}")

        # Stash agg_dict (= C_total) for later cross-condition DIFF maps;
        # per-category C_total individual maps need a shared C_max which is
        # computed after the per-category loop.
        c_dict_by_category[category] = dict(agg_dict)
        c_sem_dict_by_category[category] = dict(agg_sem_dict) if agg_sem_dict else {}

        # Diagnostic companion families generated from upstream chain-aware counts:
        # - INTERCHAIN_ONLY: all intermolecular contacts, including same sequence
        #   index on different chains.
        # - INTRACHAIN_ONLY: same-chain contacts only, with local bonded sequence
        #   neighbors excluded before sequence-position collapse.
        variant_specs = [
            {
                "name": "interchain",
                "raw_prefix": "Domain_Contacts_TotalInter_",
                "mean_prefix": "Domain_Contacts_MeanInter_",
                "sem_prefix": "Domain_Contacts_SEMInter_",
                "pair_suffix": "_INTERCHAIN_ONLY",
                "full_basename": "Domain_Contact_Map_FULL_FULL_INTERCHAIN_ONLY",
                "same_species_only": False,
            },
            {
                "name": "intrachain",
                "raw_prefix": "Domain_Contacts_TotalIntraFiltered_",
                "mean_prefix": "Domain_Contacts_MeanIntraFiltered_",
                "sem_prefix": "Domain_Contacts_SEMIntraFiltered_",
                "pair_suffix": "_INTRACHAIN_ONLY",
                "full_basename": "Domain_Contact_Map_FULL_FULL_INTRACHAIN_ONLY",
                "same_species_only": True,
            },
        ]

        for spec in variant_specs:
            raw_prefix = spec["raw_prefix"]
            mean_prefix = spec["mean_prefix"]
            sem_prefix = spec["sem_prefix"]
            same_species_only = spec["same_species_only"]
            if perform_avg:
                sm_list_variant = _discover_sms_with_domain_contacts(category_dir, prefix=raw_prefix)
                if category == "SG" and not sm_list_variant and os.path.isdir(category_dir):
                    maybe_csv = os.path.join(category_dir, f"{raw_prefix}sg_X_Protein*_*_*.csv")
                    maybe_npz = os.path.join(category_dir, f"{raw_prefix}sg_X_Protein*_*_*.csv.npz")
                    if glob(maybe_csv) or glob(maybe_npz):
                        sm_list_variant = ["sg_X"]
            else:
                sm_list_variant = _discover_sms_from_ave_dir(ave_dir, prefix=mean_prefix)

            if not sm_list_variant:
                continue

            per_sm_dicts_variant = []
            per_sm_sem_dicts_variant = []  # parallel list of per-SM time-window SEMs
            if perform_avg:
                for sm in tqdm(sm_list_variant, desc=f"{category} time-average {spec['name']}"):
                    sm_dict = {}
                    sm_sem_dict = {}
                    for i, bio_i in enumerate(BIOPOLYMER_LIST):
                        p = BIOPOLYMER_LENGTHS[i]
                        for j in range(i, len(BIOPOLYMER_LIST)):
                            if same_species_only and i != j:
                                continue
                            bio_j = BIOPOLYMER_LIST[j]
                            q = BIOPOLYMER_LENGTHS[j]
                            mat, sem, count = _load_time_averaged_pair(
                                category_dir, sm, bio_i, bio_j, p, q, tmin, dt, tmax, raw_prefix=raw_prefix
                            )
                            if count > 0:
                                sm_dict[(bio_i, bio_j)] = mat
                                sm_sem_dict[(bio_i, bio_j)] = sem
                                _save_pair_csv(ave_dir, sm, bio_i, bio_j, mat, prefix=mean_prefix)
                                _save_pair_csv(ave_dir, sm, bio_i, bio_j, sem, prefix=sem_prefix)
                    if sm_dict:
                        per_sm_dicts_variant.append(sm_dict)
                        per_sm_sem_dicts_variant.append(sm_sem_dict)
            else:
                for sm in tqdm(sm_list_variant, desc=f"{category} load-averaged {spec['name']}"):
                    sm_dict = {}
                    sm_sem_dict = {}
                    for i, bio_i in enumerate(BIOPOLYMER_LIST):
                        for j in range(i, len(BIOPOLYMER_LIST)):
                            if same_species_only and i != j:
                                continue
                            bio_j = BIOPOLYMER_LIST[j]
                            fpath = os.path.join(ave_dir, f"{mean_prefix}{sm}_{bio_i}_{bio_j}.csv")
                            alt = os.path.join(ave_dir, f"{mean_prefix}{sm}_{bio_j}_{bio_i}.csv")
                            mat = None
                            if os.path.exists(fpath):
                                try:
                                    mat = np.loadtxt(fpath, delimiter=",")
                                except Exception:
                                    mat = None
                            elif os.path.exists(alt):
                                try:
                                    mat = np.loadtxt(alt, delimiter=",")
                                    if mat.shape == (BIOPOLYMER_LENGTHS[j], BIOPOLYMER_LENGTHS[i]):
                                        mat = mat.T
                                except Exception:
                                    mat = None
                            if mat is not None and mat.size > 0:
                                sm_dict[(bio_i, bio_j)] = mat
                                # Load matching SEM file (NEW)
                                fpath_s = os.path.join(ave_dir, f"{sem_prefix}{sm}_{bio_i}_{bio_j}.csv")
                                alt_s = os.path.join(ave_dir, f"{sem_prefix}{sm}_{bio_j}_{bio_i}.csv")
                                sem_mat = None
                                if os.path.exists(fpath_s):
                                    try:
                                        sem_mat = np.loadtxt(fpath_s, delimiter=",")
                                    except Exception:
                                        sem_mat = None
                                elif os.path.exists(alt_s):
                                    try:
                                        sem_mat = np.loadtxt(alt_s, delimiter=",")
                                        if sem_mat.shape == (BIOPOLYMER_LENGTHS[j], BIOPOLYMER_LENGTHS[i]):
                                            sem_mat = sem_mat.T
                                    except Exception:
                                        sem_mat = None
                                if sem_mat is not None and np.any(np.isfinite(sem_mat)):
                                    sm_sem_dict[(bio_i, bio_j)] = sem_mat
                    if sm_dict:
                        per_sm_dicts_variant.append(sm_dict)
                        per_sm_sem_dicts_variant.append(sm_sem_dict)

            if not per_sm_dicts_variant:
                continue

            agg_dict_variant, agg_sem_dict_variant = _aggregate_over_sms_with_sem(
                per_sm_dicts_variant, BIOPOLYMER_LIST, BIOPOLYMER_LENGTHS,
                per_sm_sem_dicts=per_sm_sem_dicts_variant if any(per_sm_sem_dicts_variant) else None,
            )
            for (bio_i, bio_j), mat in agg_dict_variant.items():
                _save_pair_csv(ave_dir, category, bio_i, bio_j, mat, prefix=mean_prefix)
            for (bio_i, bio_j), sem in agg_sem_dict_variant.items():
                _save_pair_csv(ave_dir, category, bio_i, bio_j, sem, prefix=sem_prefix)
            _plot_from_contact_dict(
                agg_dict_variant, maps_dir, BIOPOLYMER_LIST, BIOPOLYMER_LENGTHS,
                show_colorbar=show_colorbar, show_yticks=show_yticks,
                pair_suffix=spec["pair_suffix"],
                full_basename=spec["full_basename"]
            )
            # Stash per-filter agg dicts and SEMs for membership/enrichment maps.
            if spec["name"] == "interchain":
                inter_agg_by_category[category] = dict(agg_dict_variant)
                inter_sem_by_category[category] = dict(agg_sem_dict_variant) if agg_sem_dict_variant else {}
            elif spec["name"] == "intrachain":
                intra_agg_by_category[category] = dict(agg_dict_variant)
                intra_sem_by_category[category] = dict(agg_sem_dict_variant) if agg_sem_dict_variant else {}
            if agg_sem_dict_variant:
                combined_sem_variant = _combine_to_symmetric(
                    agg_sem_dict_variant, BIOPOLYMER_LIST, BIOPOLYMER_LENGTHS
                )
                np.savetxt(
                    os.path.join(maps_dir, f"{spec['full_basename']}_SEM.csv"),
                    combined_sem_variant,
                    delimiter=",",
                )

        # Store combined matrix for difference plots
        if all_three_present:
            combined_matrices[category] = combined_array
            pair_matrices[category] = dict(agg_dict)
            if agg_sem_dict:
                combined_sem_matrices[category] = _combine_to_symmetric(agg_sem_dict, BIOPOLYMER_LIST, BIOPOLYMER_LENGTHS)
                pair_sem_matrices[category] = dict(agg_sem_dict)

        # --- Membership / Enrichment maps for ALL / INTER / INTRA ---
        # Build C_ALL = C_INTER + C_INTRA so the ALL numerator matches the
        # filter-aware denominator (no bonded contamination).
        if os.path.exists(biopolnum_csv) and category in inter_agg_by_category:
            inter_d = inter_agg_by_category.get(category, {})
            intra_d = intra_agg_by_category.get(category, {})
            inter_s = inter_sem_by_category.get(category, {}) or {}
            intra_s = intra_sem_by_category.get(category, {}) or {}
            all_d = {}
            all_s = {}
            for key in set(inter_d.keys()) | set(intra_d.keys()):
                a = np.asarray(inter_d.get(key, 0.0), dtype=float)
                b = np.asarray(intra_d.get(key, 0.0), dtype=float)
                if isinstance(a, np.ndarray) and a.shape:
                    base = a
                    if isinstance(b, np.ndarray) and b.shape == a.shape:
                        base = base + b
                    all_d[key] = base
                elif isinstance(b, np.ndarray) and b.shape:
                    all_d[key] = b
                # Propagate SEM_ALL = sqrt(SEM_INTER^2 + SEM_INTRA^2)
                sa = inter_s.get(key); sb = intra_s.get(key)
                if sa is not None or sb is not None:
                    sa_arr = np.asarray(sa, dtype=float) if sa is not None else None
                    sb_arr = np.asarray(sb, dtype=float) if sb is not None else None
                    if sa_arr is not None and sb_arr is not None and sa_arr.shape == sb_arr.shape:
                        all_s[key] = np.sqrt(np.nan_to_num(sa_arr, nan=0.0) ** 2 +
                                             np.nan_to_num(sb_arr, nan=0.0) ** 2)
                    elif sa_arr is not None:
                        all_s[key] = sa_arr
                    elif sb_arr is not None:
                        all_s[key] = sb_arr
            sem_by_filter = {"ALL": all_s, "INTER": inter_s, "INTRA": intra_s}
            # Only the inter-chain membership/enrichment maps are published.
            for filt_name, filt_dict in [("INTER", inter_d)]:
                if not filt_dict:
                    continue
                v_suffix = f"_{filt_name}"
                try:
                    f_map, lnE_map = _generate_membership_enrichment_domain_maps(
                        filt_dict, maps_dir, biopolnum_csv,
                        BIOPOLYMER_LIST, BIOPOLYMER_LENGTHS,
                        filter_name=filt_name.lower(),
                        variant_suffix=v_suffix,
                        SEM_C_dict=sem_by_filter.get(filt_name),
                    )
                    membership_by_category[filt_name][category] = f_map
                    log_enrichment_by_category[filt_name][category] = lnE_map
                except Exception as exc:
                    print(f"[WARN] Membership/Enrichment {v_suffix} failed for {category}: {exc}")

    # Generate difference plots if all three categories are present
    if all_three_present and len(combined_matrices) == 3:
        diff_dir = os.path.join(base_path, "FIGURES/DOMAIN_CONTACT_MAPS", "DIFF")
        _ensure_dir(diff_dir)
        if os.path.isdir(diff_dir):
            print("\nGenerating difference plots (DSM-SG, NDSM-SG, (DSM-NDSM)/SG)...")
            _generate_difference_plots(combined_matrices, diff_dir, BIOPOLYMER_LIST, BIOPOLYMER_LENGTHS)
            _write_difference_sem_csvs(combined_matrices, combined_sem_matrices, diff_dir)
            _write_pair_zscore_csvs(pair_matrices, pair_sem_matrices, diff_dir, BIOPOLYMER_LIST)

            # Membership / Enrichment DIFF maps (DSM-NDSM); inter-chain only.
            for filt_name in ("INTER",):
                mem_by = membership_by_category.get(filt_name, {})
                lne_by = log_enrichment_by_category.get(filt_name, {})
                if "DSM" in mem_by and "NDSM" in mem_by:
                    try:
                        _generate_membership_enrichment_difference_maps(
                            mem_by["DSM"], lne_by.get("DSM", {}),
                            mem_by["NDSM"], lne_by.get("NDSM", {}),
                            diff_dir, BIOPOLYMER_LIST, BIOPOLYMER_LENGTHS,
                            variant_suffix=f"_{filt_name}",
                        )
                    except Exception as exc:
                        print(f"[WARN] Membership/Enrichment DIFF {filt_name} failed: {exc}")

            # C_total DIFF maps (DSM vs NDSM): linear DeltaC, log-ratio,
            # and z-score on log-ratio.
            if "DSM" in c_dict_by_category and "NDSM" in c_dict_by_category:
                print("Generating C_total difference plots (DSM vs NDSM)...")
                try:
                    _generate_ctotal_difference_maps(
                        c_dict_by_category["DSM"],
                        c_dict_by_category["NDSM"],
                        c_sem_dict_by_category.get("DSM"),
                        c_sem_dict_by_category.get("NDSM"),
                        diff_dir, BIOPOLYMER_LIST, BIOPOLYMER_LENGTHS,
                    )
                except Exception as exc:
                    print(f"[WARN] C_total DIFF map generation failed: {exc}")
        else:
            print(f"Skipping diff plots: missing dir {diff_dir}")

    # C_total individual maps per category, with shared color limits across
    # SG / DSM / NDSM so the colorbar is comparable. Runs after the per-category
    # loop so we know the global maximum.
    if c_dict_by_category:
        all_C_max = 0.0
        for c_dict in c_dict_by_category.values():
            v = _dict_global_max(c_dict)
            if v > all_C_max:
                all_C_max = v
        if all_C_max <= 0:
            all_C_max = 1.0
        # Shared P_membership max: max over P_membership_max(SG/DSM/NDSM)
        all_Pm_max = 0.0
        for c_dict in c_dict_by_category.values():
            total = 0.0
            for mat in c_dict.values():
                a = np.asarray(mat, dtype=float)
                total += float(np.nansum(a[np.isfinite(a)]))
            if total <= 0:
                continue
            Pm = {k: np.asarray(v, dtype=float) / total for k, v in c_dict.items()}
            v = _dict_global_max(Pm)
            if v > all_Pm_max:
                all_Pm_max = v
        if all_Pm_max <= 0:
            all_Pm_max = 1.0
        print("Generating C_total domain maps (per category)...")
        for category, c_dict in c_dict_by_category.items():
            maps_dir = os.path.join(base_path, "FIGURES/DOMAIN_CONTACT_MAPS", category)
            if os.path.isdir(maps_dir):
                try:
                    _generate_ctotal_domain_maps(
                        c_dict, maps_dir, BIOPOLYMER_LIST, BIOPOLYMER_LENGTHS,
                        shared_C_max=all_C_max,
                        shared_Pm_max=all_Pm_max,
                    )
                except Exception as exc:
                    print(f"[WARN] C_total domain map generation failed for {category}: {exc}")


def main():
    """Parse command line arguments and run analysis/aggregation.

    New CLI aligns with other scripts; computes analysis_root as:
      {path}/{folder}_{temp}_{dt}_{tmin}_{tmax}
    and uses it for all reads/writes. Inputs are expected in
      analysis_root/ANALYSIS_{SG|DSM|NDSM}
    and outputs are written to
      analysis_root/ANALYSIS_*_AVE and analysis_root/FIGURES/DOMAIN_CONTACT_MAPS
    """
    import argparse
    parser = argparse.ArgumentParser(description='Aggregate domain contact maps across SMs and time.')
    parser.add_argument('--path', required=True, help='Path to TEMP_XXX directory (e.g., TEMP_300)')
    parser.add_argument('--folder', default='CLASSIFY', help='Output folder prefix (default: CLASSIFY)')
    parser.add_argument('--temp', type=int, required=True, help='Temperature in Kelvin (e.g., 300)')
    parser.add_argument('--tmin', type=int, required=True, help='Start frame')
    parser.add_argument('--dt', type=int, required=True, help='Frame stride')
    parser.add_argument('--tmax', type=int, required=True, help='End frame')
    parser.add_argument(
        '--no-avg',
        action='store_true',
        help='Skip raw per-window time-averaging; use existing ANALYSIS_*_AVE averaged contact matrices',
    )
    parser.add_argument(
        '--plot-only',
        action='store_true',
        help='Regenerate FIGURES/DOMAIN_CONTACT_MAPS plots from saved ANALYSIS_*_AVE data only; no raw contact matrices are read',
    )
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"Error: Path {args.path} does not exist")
        sys.exit(1)

    # Normalize to absolute path for consistency
    args.path = os.path.abspath(args.path)

    base_path = os.path.join(
        args.path,
        f"{args.folder}_{args.temp}_{args.dt}_{args.tmin}_{args.tmax}"
    )
    if not os.path.exists(base_path):
        print(f"Error: Analysis root {base_path} does not exist")
        sys.exit(1)
    if int(args.tmin) >= int(args.tmax):
        print(f"Error: tmin ({args.tmin}) must be less than tmax ({args.tmax})")
        sys.exit(1)
    if int(args.dt) <= 0:
        print(f"Error: dt ({args.dt}) must be positive")
        sys.exit(1)

    # Ensure expected existing classification run dirs; do not create new ones
    if not os.path.isdir(base_path):
        print(f"Error: Classification run folder not found: {base_path}")
        sys.exit(1)
    maps_root = os.path.join(base_path, 'FIGURES/DOMAIN_CONTACT_MAPS')
    if not os.path.isdir(maps_root):
        try:
            os.makedirs(maps_root, exist_ok=True)
            print(f"Created maps root: {maps_root}")
        except Exception as exc:
            print(f"Error: failed to create maps folder {maps_root}: {exc}")
            sys.exit(1)

    # Inputs live under the temperature root (e.g., TEMP_300/ANALYSIS_SG). Use that as CWD.
    os.chdir(args.path)

    do_avg = not (args.no_avg or args.plot_only)
    if do_avg:
        print("[ARRAY] mode=full: reading raw Domain_Contacts_Total* windows and writing averaged matrices")
    else:
        print("[ARRAY] mode=plot-only: reading saved ANALYSIS_*_AVE Domain_Contacts_Mean* matrices")
    # Run in aggregate mode using ANALYSIS_SG/DSM/NDSM inputs from args.path, writing to base_path
    try:
        run_aggregate_mode(base_path, int(args.tmin), int(args.dt), int(args.tmax), perform_avg=do_avg)
    except Exception as exc:
        print(f"ARRAY aggregation failed: {exc}")
        sys.exit(1)


if __name__ == '__main__':
    main()
