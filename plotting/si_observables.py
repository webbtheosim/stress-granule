"""
si_observables.py — per-temperature SI plots for per-species observables.

For each per-species observable (R_g, R_h, Occ, r/R, l_conf, tau_conf, D_SE_GK)
this script produces two figures:
  (A) Grouped bars across the 7 biopolymer species, three bars per species
      (SG / DSM / NDSM) with colour-matched SEM error bars; tick labels
      centred under the middle (DSM) bar; no legend.
  (B) Single-axis violin plot styled like KMeans.plot_violin (width=0.35,
      saturation=100, linewidth=2, inner='box', cut=0). All 7 species on one
      axis, with the DSM (D1..D10) and NDSM (ND1..ND10) distributions dodged
      side-by-side per species, and the SG value shown as a horizontal grey
      segment per species.

All figures use the RDP layout: 3.20 x 3.20 in figure with a 2.20 x 2.20 in
square axes (centred). No axis labels, no titles. Y-axis from 0 to
ceil-2sf(max) with the top tick at the max. Y tick labels are never in
scientific notation in this SI/ output (fixed-point with 2 significant
figures of precision).

Role: figure renderer for the per-species SI panels (SI 5) — grouped bars and
single-axis violins per observable, plus a biopolymer-composition violin (the
S2.B-style count-ratio figure).

Key inputs: ``{path}/{folder}_{temp}_{dt}_{tmin}_{tmax}/RESULTS/SUMMARY/Quant_Data.csv``
and the ``ANALYSIS_*_AVE/BioNumDF_*.csv`` composition files, selected via the
``--temp``, ``--path``, ``--folder``, ``--tmin``, ``--dt``, ``--tmax`` flags.
Key outputs: PNG panels under ``{classify_dir}/{--out-subdir}`` (default
``FIGURES/SI/PER_SPECIES``).

Usage:
    python si_observables.py --temp 300
    python si_observables.py --temp 300 --folder CLASSIFY_CORRELATED
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------- Style (PLOT_STYLE.md defaults) ----------
sns.set_theme(style="ticks")
sns.set_style("white")
plt.rc("font", size=10)
plt.rc("axes", titlesize=10)
plt.rc("axes", labelsize=10)
plt.rc("xtick", labelsize=8)
plt.rc("ytick", labelsize=8)
plt.rc("legend", fontsize=8)
plt.rc("axes", linewidth=2)

COLOR_SG = "#808080"
COLOR_DSM = "#40641b"
COLOR_NDSM = "#bfe49b"

# RDP layout — match biopolymer_analysis.py RDP axes exactly (2.40 x 2.40 in).
RDP_FIGSIZE = (3.20, 3.20)
RDP_AX_RECT = [0.50 / 3.20, 0.50 / 3.20, 2.20 / 3.20, 2.20 / 3.20]

SPECIES_ORDER = ["TDP43", "FUS", "TIA1", "G3BP1", "PABP1", "TTP", "RNA"]

PER_SPECIES_OBSERVABLES: List[Tuple[str, str, str, str]] = [
    ("Rg",      r"$R_{g,{X}}$ A",                                 r"SIG$R_{g,{X}}$ A",                        r"$R_{g}$"),
    ("Rh",      r"$R_{h,{X}}$ A",                                 r"SIG$R_{h,{X}}$ A",                        r"$R_{h}$"),
    ("Occ",     r"$Occ_{{X}}$",                                   r"SIG$Occ_{{X}}$",                          r"Occupancy"),
    ("rOverR",  r"$r/R_{{X}}$",                                   r"SIG$r/R_{{X}}$",                          r"$r/R$"),
    ("lconf",   r"$l_{conf,{X}}$ A",                              r"SIG$l_{conf,{X}}$ A",                     r"$l_{conf}$"),
    ("tauconf", r"$\tau_{conf,{X}}$ $ns$",                        r"SIG$\tau_{conf,{X}}$ $ns$",               r"$\tau_{conf}$"),
    ("D_SE_GK", r"$D_{SE,GK,{X}}$ $\mu m^{2} / s$",               r"SIG$D_{SE,GK,{X}}$ $\mu m^{2} / s$",      r"$D_{SE,GK}$"),
]

DSM_IDS = [f"D{i}" for i in range(1, 11)]
NDSM_IDS = [f"ND{i}" for i in range(1, 11)]
SG_ID = "SG"
DSM_AVG_ID = "DSM_AVG"
NDSM_AVG_ID = "NDSM_AVG"

# Biopolymer-composition column order in BioNumDF_*.csv files.
# First entry "SG" is the cluster bead total; the other seven are species
# bead counts. Matches BIOPOLYMER_COMPOSITION_CHANGE.py from V1.
BIOPOLYMER_COMPOSITION_ORDER = ["SG", "TDP43", "FUS", "TIA1", "G3BP1", "PABP1", "TTP", "RNA"]


# ---------- helpers ----------
def _ceil_2sf(x: float) -> float:
    """Round ``x`` away from zero to 2 significant figures (axis-limit helper)."""
    if not np.isfinite(x):
        return 1.0
    if x == 0:
        return 0.0
    sign = 1 if x > 0 else -1
    ax = abs(x)
    exp = math.floor(math.log10(ax))
    mantissa = ax / (10 ** exp)
    if sign > 0:
        rounded = math.ceil(mantissa * 10) / 10
    else:
        rounded = math.floor(mantissa * 10) / 10
    if rounded >= 10:
        rounded /= 10
        exp += 1
    return sign * rounded * (10 ** exp)


def _fmt_2sf_no_sci(v: float, _pos: int = 0) -> str:
    """Format v to 2 sig figs as fixed-point (never scientific notation)."""
    if not np.isfinite(v):
        return ""
    if v == 0:
        return "0"
    exp = math.floor(math.log10(abs(v)))
    factor = 10 ** (1 - exp)
    rounded = round(v * factor) / factor
    decimals = max(0, 1 - exp)
    return f"{rounded:.{decimals}f}"


def _make_square_axes() -> Tuple[plt.Figure, plt.Axes]:
    """Create a figure with the fixed RDP-style square axes rectangle."""
    fig = plt.figure(figsize=RDP_FIGSIZE)
    ax = fig.add_axes(RDP_AX_RECT)
    return fig, ax


def _style_xticks_under_ticks(ax: plt.Axes, positions: np.ndarray, labels: List[str], rotation: float = 45) -> None:
    """Position x-tick labels centred under each tick (residue contact map style)."""
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=rotation)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("center")
        lbl.set_va("top")
        lbl.set_rotation_mode("default")


def _apply_y_axis_no_sci(ax: plt.Axes, y_low: float, y_high: float, nticks: int = 5,
                         x_ticks_outward: bool = False) -> None:
    """Set y-limits/ticks with fixed-point (non-scientific) labels and clear titles.

    Ticks point inward on all sides by default; when ``x_ticks_outward`` is True
    the x-axis ticks face outward so they stay visible below tall bars.
    """
    ax.set_ylim(y_low, y_high)
    ax.set_yticks(np.linspace(y_low, y_high, nticks))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_2sf_no_sci))
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    if x_ticks_outward:
        # y-axis stays inward; x-axis ticks point outward so they remain visible
        # below the bar baseline (otherwise inward x-ticks are hidden inside the bars).
        # Length deliberately longer than the y-axis ticks so they read clearly
        # below the spine even when bars are tall.
        ax.tick_params(axis="y", left=True, right=True, direction="in", length=4, width=2)
        ax.tick_params(axis="x", top=False, bottom=True, labelbottom=True,
                       direction="out", length=6, width=2)
    else:
        ax.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True,
                       direction="in", length=4, width=2)


def find_quant_data_csv(path: Path, folder: str, temp: int, tmin: int, dt: int, tmax: int) -> Path:
    """Build and verify the Quant_Data.csv path for a given run window.

    Raises ``FileNotFoundError`` if the expected
    ``{folder}_{temp}_{dt}_{tmin}_{tmax}/RESULTS/SUMMARY/Quant_Data.csv`` is absent.
    """
    classify = path / f"{folder}_{temp}_{dt}_{tmin}_{tmax}"
    candidate = classify / "RESULTS" / "SUMMARY" / "Quant_Data.csv"
    if not candidate.exists():
        raise FileNotFoundError(f"Could not find {candidate}")
    return candidate


def load_quant_data(csv_path: Path) -> pd.DataFrame:
    """Read Quant_Data.csv and index it by 'Small Molecule ID' (raises if missing)."""
    df = pd.read_csv(csv_path)
    if "Small Molecule ID" not in df.columns:
        raise ValueError(f"Missing 'Small Molecule ID' column in {csv_path}")
    return df.set_index("Small Molecule ID")


def fmt_column(template: str, species: str) -> str:
    """Substitute a species name into a ``{X}`` column-name template."""
    return template.replace("{X}", species)


def numeric_value(df: pd.DataFrame, row_id: str, col: str) -> float:
    """Safely read ``df.at[row_id, col]`` as float, returning NaN if absent/invalid."""
    if row_id not in df.index or col not in df.columns:
        return np.nan
    try:
        return float(df.at[row_id, col])
    except (TypeError, ValueError):
        return np.nan


def numeric_series(df: pd.DataFrame, row_ids: Iterable[str], col: str) -> np.ndarray:
    """Return the numeric values of one column over the given row ids as an array.

    Missing rows/columns yield an empty array; non-numeric entries become NaN.
    """
    rows = [r for r in row_ids if r in df.index]
    if not rows or col not in df.columns:
        return np.array([], dtype=float)
    return pd.to_numeric(df.loc[rows, col], errors="coerce").to_numpy(dtype=float)


# ---------- plots ----------
def plot_grouped_bars(df: pd.DataFrame, template: str, sig_template: str, output_path: Path) -> None:
    """Render grouped SG/DSM/NDSM bars (with SEM) across the 7 species and save it.

    Three bars per species (SG, DSM_AVG, NDSM_AVG) with black SEM error bars; the
    y-axis runs 0..ceil-2sf(max) and the figure is saved to ``output_path``.
    """
    species = SPECIES_ORDER
    n = len(species)
    x = np.arange(n, dtype=float)
    width = 0.27

    sg = np.array([numeric_value(df, SG_ID, fmt_column(template, sp)) for sp in species])
    dsm = np.array([numeric_value(df, DSM_AVG_ID, fmt_column(template, sp)) for sp in species])
    nd = np.array([numeric_value(df, NDSM_AVG_ID, fmt_column(template, sp)) for sp in species])
    sg_sem = np.array([numeric_value(df, SG_ID, fmt_column(sig_template, sp)) for sp in species])
    dsm_sem = np.array([numeric_value(df, DSM_AVG_ID, fmt_column(sig_template, sp)) for sp in species])
    nd_sem = np.array([numeric_value(df, NDSM_AVG_ID, fmt_column(sig_template, sp)) for sp in species])

    fig, ax = _make_square_axes()
    ax.bar(x - width, sg, width=width, color=COLOR_SG, edgecolor="k", linewidth=1.4)
    ax.bar(x,         dsm, width=width, color=COLOR_DSM, edgecolor="k", linewidth=1.4)
    ax.bar(x + width, nd, width=width, color=COLOR_NDSM, edgecolor="k", linewidth=1.4)
    # Black error bars on bars.
    for xi, v, s in zip(x - width, sg, sg_sem):
        if np.isfinite(v) and np.isfinite(s):
            ax.errorbar([xi], [v], yerr=[s], fmt="none", color="k", ecolor="k",
                        elinewidth=1, capsize=2, capthick=1)
    for xi, v, s in zip(x, dsm, dsm_sem):
        if np.isfinite(v) and np.isfinite(s):
            ax.errorbar([xi], [v], yerr=[s], fmt="none", color="k", ecolor="k",
                        elinewidth=1, capsize=2, capthick=1)
    for xi, v, s in zip(x + width, nd, nd_sem):
        if np.isfinite(v) and np.isfinite(s):
            ax.errorbar([xi], [v], yerr=[s], fmt="none", color="k", ecolor="k",
                        elinewidth=1, capsize=2, capthick=1)

    tops: List[float] = []
    for vals, sems in [(sg, sg_sem), (dsm, dsm_sem), (nd, nd_sem)]:
        for v, s in zip(vals, sems):
            if np.isfinite(v):
                tops.append(v + (s if np.isfinite(s) else 0.0))
    ymax = _ceil_2sf(max(tops)) if tops else 1.0

    _apply_y_axis_no_sci(ax, y_low=0.0, y_high=ymax, nticks=5, x_ticks_outward=True)
    _style_xticks_under_ticks(ax, x, species, rotation=45)

    if ax.get_legend() is not None:
        ax.get_legend().remove()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=400)
    plt.close(fig)


# Violin styling shared by the per-species violin figures.
VIOLIN_WIDTH = 0.85       # slightly < 1 so adjacent species leave a small x-gap
VIOLIN_OUTLINE_LW = 1.0   # violin body outline
IQR_BAR_LW = 2.5          # inner IQR bar (rounded caps)
WHISKER_LW = 0.9          # inner whisker line (slightly thinner)
MEDIAN_DOT_SIZE = 6       # white centre dot (points^2) — small
MEDIAN_DOT_EDGE_LW = 0.5


def _overlay_box_inner(ax, plot_df: pd.DataFrame, species_order: List[str],
                       hue_order: List[str], width: float = VIOLIN_WIDTH,
                       value_col: str = "value", species_col: str = "species",
                       hue_col: str = "class") -> None:
    """Manual inner box for hue-dodged violins (replaces seaborn inner='box').

    Per (species, hue) group draws a thin black whisker (1.5xIQR clipped to the
    data range), a slightly-thinner black IQR bar (Q1-Q3), and the median as a
    filled white dot — matching the requested SI styling. Dodge geometry follows
    seaborn 0.13: hue level i of n sits at x + (i - (n-1)/2) * width/n."""
    n_hue = len(hue_order)
    for xi, sp in enumerate(species_order):
        for hi, hcls in enumerate(hue_order):
            vals = plot_df[(plot_df[species_col] == sp) & (plot_df[hue_col] == hcls)][value_col]
            vals = vals.to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            offset = (hi - (n_hue - 1) / 2.0) * (width / n_hue)
            xc = xi + offset
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            iqr = q3 - q1
            if iqr > 0:
                lo = max(float(vals.min()), q1 - 1.5 * iqr)
                hi_w = min(float(vals.max()), q3 + 1.5 * iqr)
            else:
                lo, hi_w = float(vals.min()), float(vals.max())
            ax.plot([xc, xc], [lo, hi_w], color="k", linewidth=WHISKER_LW,
                    solid_capstyle="round", zorder=5)
            ax.plot([xc, xc], [q1, q3], color="k", linewidth=IQR_BAR_LW,
                    solid_capstyle="round", zorder=6)
            ax.scatter([xc], [med], s=MEDIAN_DOT_SIZE, facecolor="white",
                       edgecolor="k", linewidth=MEDIAN_DOT_EDGE_LW, zorder=7)


def plot_single_axis_violins(df: pd.DataFrame, template: str, sig_template: str, output_path: Path) -> None:
    """Single-axis violin plot styled to match KMeans.plot_violin."""
    species = SPECIES_ORDER

    records = []
    for sp in species:
        col = fmt_column(template, sp)
        for v in numeric_series(df, DSM_IDS, col):
            if np.isfinite(v):
                records.append({"species": sp, "class": "DSM", "value": v})
        for v in numeric_series(df, NDSM_IDS, col):
            if np.isfinite(v):
                records.append({"species": sp, "class": "NDSM", "value": v})
    plot_df = pd.DataFrame(records)

    fig, ax = _make_square_axes()
    if not plot_df.empty:
        palette = sns.color_palette([COLOR_DSM, COLOR_NDSM], 2)
        # Exact match for BIOPOLYMER_COMPOSITION_CHANGE.plot_violin (Figure S2.B):
        # width=1, density_norm='width' (every violin gets equal max width
        # regardless of sample density, so tightly clustered species are no
        # longer pinched into invisible slivers), linewidth=1 (thin outline),
        # inner='box', cut=0, dodge=True, saturation=100.
        sns.violinplot(
            ax=ax, data=plot_df, x="species", y="value", hue="class",
            order=species, hue_order=["DSM", "NDSM"],
            palette=palette, dodge=True, width=VIOLIN_WIDTH, density_norm="width", saturation=100,
            linewidth=VIOLIN_OUTLINE_LW, edgecolor="k", inner=None, cut=0,
        )
        _overlay_box_inner(ax, plot_df, species, ["DSM", "NDSM"], width=VIOLIN_WIDTH)

    # Per-species SG marker (short grey horizontal segment).
    sg_vals: List[float] = []
    for i, sp in enumerate(species):
        sg_val = numeric_value(df, SG_ID, fmt_column(template, sp))
        if np.isfinite(sg_val):
            ax.hlines(sg_val, i - 0.40, i + 0.40, color=COLOR_SG, linewidth=2, zorder=4)
            sg_vals.append(sg_val)

    all_top: List[float] = []
    if not plot_df.empty:
        all_top.extend(plot_df["value"].tolist())
    all_top.extend(sg_vals)
    ymax = _ceil_2sf(max(all_top)) if all_top else 1.0

    _apply_y_axis_no_sci(ax, y_low=0.0, y_high=ymax, nticks=5)
    _style_xticks_under_ticks(ax, np.arange(len(species), dtype=float), species, rotation=45)

    if ax.get_legend() is not None:
        ax.get_legend().remove()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=400)
    plt.close(fig)


def _read_bionumdf(path: Path) -> Optional[Dict[str, float]]:
    """Return {Biopolymer: Mean} dict from a BioNumDF_*.csv file, or None on failure."""
    try:
        d = pd.read_csv(path)
    except Exception as exc:
        print(f"[SI_OBSERVABLES] cannot read {path}: {exc}")
        return None
    if "Biopolymer" not in d.columns or "Mean" not in d.columns:
        return None
    return dict(zip(d["Biopolymer"].astype(str), pd.to_numeric(d["Mean"], errors="coerce")))


def plot_biopolymer_composition_violins(classify_dir: Path, output_path: Path) -> None:
    """Recreate Figure S2.B: per-species biopolymer count ratio relative to SG.

    For each DSM and NDSM compound c, plot
        ratio[s] = count_c[s] / count_SG[s]
    for s in {SG_total, TDP43, FUS, TIA1, G3BP1, PABP1, TTP, RNA}. Two violins
    per x-position (DSM dark green, NDSM light green), matching the style of
    BIOPOLYMER_COMPOSITION_CHANGE.plot_violin in the V1 analysis.
    """
    sg_path = classify_dir / "ANALYSIS_SG_AVE" / "BioNumDF_sg_X.csv"
    sg = _read_bionumdf(sg_path)
    if sg is None:
        print(f"[SI_OBSERVABLES] skip biopolymer composition: missing {sg_path}")
        return

    dsm_paths = sorted((classify_dir / "ANALYSIS_DSM_AVE").glob("BioNumDF_*.csv"))
    ndsm_paths = sorted((classify_dir / "ANALYSIS_NDSM_AVE").glob("BioNumDF_*.csv"))
    if not dsm_paths or not ndsm_paths:
        print("[SI_OBSERVABLES] skip biopolymer composition: missing DSM/NDSM BioNumDF files")
        return

    records: List[Dict[str, float]] = []
    for cls, paths in (("DSM", dsm_paths), ("NDSM", ndsm_paths)):
        for p in paths:
            comp = _read_bionumdf(p)
            if comp is None:
                continue
            for sp in BIOPOLYMER_COMPOSITION_ORDER:
                num = comp.get(sp, np.nan)
                den = sg.get(sp, np.nan)
                if np.isfinite(num) and np.isfinite(den) and den != 0:
                    records.append({"species": sp, "class": cls, "value": float(num) / float(den)})
    plot_df = pd.DataFrame(records)
    if plot_df.empty:
        print("[SI_OBSERVABLES] skip biopolymer composition: no finite values")
        return

    palette = sns.color_palette([COLOR_DSM, COLOR_NDSM], 2)
    fig, ax = _make_square_axes()
    sns.violinplot(
        ax=ax, data=plot_df, x="species", y="value", hue="class",
        order=BIOPOLYMER_COMPOSITION_ORDER, hue_order=["DSM", "NDSM"],
        palette=palette, dodge=True, width=VIOLIN_WIDTH, density_norm="width", saturation=100,
        linewidth=VIOLIN_OUTLINE_LW, edgecolor="k", inner=None, cut=0,
    )
    _overlay_box_inner(ax, plot_df, BIOPOLYMER_COMPOSITION_ORDER, ["DSM", "NDSM"], width=VIOLIN_WIDTH)

    # SG reference: ratio[s] = 1 for every species by construction.
    ax.axhline(1.0, color=COLOR_SG, linewidth=2, zorder=1)

    finite_vals = plot_df["value"].to_numpy(dtype=float)
    finite_vals = finite_vals[np.isfinite(finite_vals)]
    ymax = _ceil_2sf(max(np.max(finite_vals), 1.0)) if finite_vals.size else 1.0

    _apply_y_axis_no_sci(ax, y_low=0.0, y_high=ymax, nticks=5)
    _style_xticks_under_ticks(ax, np.arange(len(BIOPOLYMER_COMPOSITION_ORDER), dtype=float),
                              BIOPOLYMER_COMPOSITION_ORDER, rotation=45)

    if ax.get_legend() is not None:
        ax.get_legend().remove()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=400)
    plt.close(fig)


def main() -> None:
    """CLI entry point: locate Quant_Data.csv and render all SI per-species panels.

    For each available per-species observable writes the grouped-bar and violin
    PNGs, then the biopolymer-composition violin, into ``--out-subdir``.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--temp", type=int, required=True)
    p.add_argument("--path", default=None)
    p.add_argument("--folder", default="CLASSIFY")
    p.add_argument("--tmin", type=int, default=50)
    p.add_argument("--dt", type=int, default=50)
    p.add_argument("--tmax", type=int, default=2000)
    p.add_argument("--out-subdir", default="FIGURES/SI/PER_SPECIES")
    args = p.parse_args()

    if args.path is None:
        args.path = f"TEMP_{args.temp}"
    path = Path(args.path)
    csv_path = find_quant_data_csv(path, args.folder, args.temp, args.tmin, args.dt, args.tmax)
    classify_dir = csv_path.parent.parent.parent
    out_dir = classify_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_quant_data(csv_path)
    print(f"[SI_OBSERVABLES] {csv_path} -> {out_dir}")

    for slug, tmpl, sig_tmpl, _label in PER_SPECIES_OBSERVABLES:
        if not any(fmt_column(tmpl, sp) in df.columns for sp in SPECIES_ORDER):
            print(f"[SI_OBSERVABLES] skip {slug}: no columns")
            continue
        plot_grouped_bars(df, tmpl, sig_tmpl, out_dir / f"{slug}_grouped_bars.png")
        plot_single_axis_violins(df, tmpl, sig_tmpl, out_dir / f"{slug}_violins.png")
        print(f"[SI_OBSERVABLES] wrote {slug}_grouped_bars.png and {slug}_violins.png")

    plot_biopolymer_composition_violins(classify_dir, out_dir / "Biopolymer_Composition_violins.png")
    print("[SI_OBSERVABLES] wrote Biopolymer_Composition_violins.png")
    print("[SI_OBSERVABLES] done.")


if __name__ == "__main__":
    main()
