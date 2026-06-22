"""
si_temperature_trends.py — observables vs temperature, square RDP-style panels.

Reads Quant_Data.csv across temperatures and produces one figure per observable
(global observables) plus one figure per (per-species observable × species).

Each figure is a 3.20 in x 3.20 in panel with 2.20 in x 2.20 in square axes
matching the RDP layout. No axis labels, no titles. Legend top-right, 1 column,
3 rows (SG / DSM / NDSM) with combined marker+line entries matching the
projected-binodal phase-diagram legend. SG, DSM, NDSM are all solid lines; SG
is grey. Error bars are coloured to match the line and points they correspond
to.

The X-axis always shows all 7 simulated temperatures (285, 290, 295, 300, 305,
310, 315 K). The Y-axis spans floor-2sf(min) to ceil-2sf(max). Y tick labels
are normalised by a common 10^N exponent (KMeans-style) with the multiplier
"x10^N" annotated above the y-axis.

Two variants per observable:
  - absolute:  X vs T
  - delta:     (X_class - X_SG) vs T

Usable two ways:
  1. As a library, imported by phase_diagram.py.
  2. Standalone:
        python si_temperature_trends.py \
            --summary-glob 'TEMP_*/CLASSIFY_*_50_50_2000/RESULTS/SUMMARY/Quant_Data.csv' \
            --output-dir FIGURES_SI/temperature_trends_legacy
"""

from __future__ import annotations

import argparse
import glob as globlib
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

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

# RDP layout — match biopolymer_analysis.py RDP plots: 2.20x2.20 in axes,
# figsize 3.20x3.20, centered both axes (equal 0.50 in L/R/T/B margins).
RDP_FIGSIZE = (3.20, 3.20)
RDP_AX_RECT = [0.50 / 3.20, 0.50 / 3.20, 2.20 / 3.20, 2.20 / 3.20]

# Canonical x-axis temperatures (always show all 7).
# xlim padded to 280-320 with ticks at the 7 simulated temperatures.
TEMP_TICKS = [285, 290, 295, 300, 305, 310, 315]
TEMP_XLIM = (285.0, 315.0)

SPECIES_ORDER = ["TDP43", "FUS", "TIA1", "G3BP1", "PABP1", "TTP", "RNA"]

PER_SPECIES_OBSERVABLES: List[Tuple[str, str, str, str]] = [
    ("Rg",      r"$R_{g,{X}}$ A",                                 r"SIG$R_{g,{X}}$ A",                       r"$R_{g}$"),
    ("Rh",      r"$R_{h,{X}}$ A",                                 r"SIG$R_{h,{X}}$ A",                       r"$R_{h}$"),
    ("Occ",     r"$Occ_{{X}}$",                                   r"SIG$Occ_{{X}}$",                          r"Occupancy"),
    ("rOverR",  r"$r/R_{{X}}$",                                   r"SIG$r/R_{{X}}$",                          r"$r/R$"),
    ("lconf",   r"$l_{conf,{X}}$ A",                              r"SIG$l_{conf,{X}}$ A",                     r"$l_{conf}$"),
    ("tauconf", r"$\tau_{conf,{X}}$ $ns$",                        r"SIG$\tau_{conf,{X}}$ $ns$",               r"$\tau_{conf}$"),
    ("D_SE_GK", r"$D_{SE,GK,{X}}$ $\mu m^{2} / s$",               r"SIG$D_{SE,GK,{X}}$ $\mu m^{2} / s$",      r"$D_{SE,GK}$"),
]

# Global observables vs T. The starred (KMeans) entries are the aggregate
# features fed into the SMC classifier (PI_AUDIT_FIXES.AGG_POOL): P_SM, N_D,
# phi_D, phi_R, R_g, R_cond, W_interface, c_dense_SG_fit, c_dense_SG_calc,
# eta_GK, l_conf, Delta G_trans, gamma_1, gamma_2, gamma_ave. All 15 are
# included here so every classifier input has a temperature-trend panel.
# (c_dilute_SG_fit is kept as a non-KMeans reference companion to c_dense.)
GLOBAL_OBSERVABLES: List[Tuple[str, str, str, str]] = [
    ("P_SM",             r"$P_{SM}$",                        r"SIG$P_{SM}$",                        r"$P_{SM}$"),
    ("N_D",              r"$N_{D}$",                         r"SIG$N_{D}$",                         r"$N_{D}$"),
    ("phi_D",            r"$\phi_{D}$",                      r"SIG$\phi_{D}$",                      r"$\phi_{D}$"),
    ("phi_R",            r"$\phi_{R}$",                      r"SIG$\phi_{R}$",                      r"$\phi_{R}$"),
    ("R_g",              r"$R_{g}$",                         r"SIG$R_{g}$",                         r"$R_{g}$"),
    ("R_cond",           r"$R_{cond}$ $(\AA)$",              r"SIG$R_{cond}$ $(\AA)$",              r"$R_{cond}$"),
    ("W_interface",      r"$W_{interface}$ $(\AA)$",         r"SIG$W_{interface}$ $(\AA)$",         r"$W_{interface}$"),
    ("c_dense_SG_fit",   r'$c_{dense,SG,fit}$ $(mg/ml)$',   r'SIG$c_{dense,SG,fit}$ $(mg/ml)$',   r"$c_{dense,fit}$"),
    ("c_dense_SG_calc",  r'$c_{dense,SG,calc}$ $(mg/ml)$',  r'SIG$c_{dense,SG,calc}$ $(mg/ml)$',  r"$c_{dense,calc}$"),
    ("eta_GK",           r"$\eta_{GK}$ Pa s",                r"SIG$\eta_{GK}$ Pa s",                r"$\eta_{GK}$"),
    ("l_conf",           r"$l_{conf}$ A",                    r"SIG$l_{conf}$ A",                    r"$l_{conf}$"),
    ("delta_G_trans",    r"$\Delta G_{trans}$ $(kJ/mol)$",   r"SIG$\Delta G_{trans}$ $(kJ/mol)$",   r"$\Delta G_{trans}$"),
    ("gamma_1",          r"$\gamma_{1}$ $(mN/m)$",           r"SIG$\gamma_{1}$ $(mN/m)$",           r"$\gamma_{1}$"),
    ("gamma_2",          r"$\gamma_{2}$ $(mN/m)$",           r"SIG$\gamma_{2}$ $(mN/m)$",           r"$\gamma_{2}$"),
    ("gamma_ave",        r"$\gamma_ave$ $(mN/m)$",           r"SIG$\gamma_ave$ $(mN/m)$",           r"$\gamma_{ave}$"),
    # Non-KMeans reference companion to c_dense.
    ("c_dilute_SG_fit",  r'$c_{dilute,SG,fit}$ $(mg/ml)$',  r'SIG$c_{dilute,SG,fit}$ $(mg/ml)$',  r"$c_{dilute}$"),
]

SG_ID = "SG"
DSM_AVG_ID = "DSM_AVG"
NDSM_AVG_ID = "NDSM_AVG"


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


def _floor_2sf(x: float) -> float:
    """Round ``x`` toward zero to 2 significant figures (axis-limit helper)."""
    return -_ceil_2sf(-x)


def _make_square_axes() -> Tuple[plt.Figure, plt.Axes]:
    """Create a figure with the fixed RDP-style square axes rectangle."""
    fig = plt.figure(figsize=RDP_FIGSIZE)
    ax = fig.add_axes(RDP_AX_RECT)
    return fig, ax


def _apply_2sig_yticks_sci(ax: plt.Axes, y_low: float, y_high: float, n_ticks: int = 5) -> None:
    """KMeans-style: floor/ceil to 2 sig figs, common x10^N exponent
    annotated above the y-axis (upper-left of axes)."""
    if abs(y_high - y_low) < 1e-30:
        y_high = y_low + 1.0
    ax.set_ylim(y_low, y_high)
    ax.set_yticks(np.linspace(y_low, y_high, n_ticks))
    max_abs_tick = max(abs(y_low), abs(y_high))
    if max_abs_tick > 0:
        common_exp = int(math.floor(math.log10(max_abs_tick)))
    else:
        common_exp = 0
    scale = 10.0 ** common_exp
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _pos, s=scale: f"{val / s:.1f}"))
    if common_exp != 0:
        ax.annotate(f"$\\times10^{{{common_exp}}}$", xy=(0, 1), xycoords="axes fraction",
                    ha="left", va="bottom", fontsize=8)


def _style_x_temperature(ax: plt.Axes) -> None:
    """Apply the fixed temperature x-axis (limits, 7 ticks, inward ticks)."""
    ax.set_xlim(*TEMP_XLIM)
    ax.set_xticks(TEMP_TICKS)
    ax.set_xticklabels([str(t) for t in TEMP_TICKS])
    ax.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True,
                   direction="in", length=4, width=2)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("center")
        lbl.set_va("top")
        lbl.set_rotation_mode("default")


def _legend_best(ax: plt.Axes, handles: List) -> None:
    """Draw a frameless single-column 'best'-located legend from given handles."""
    ax.legend(
        handles=handles,
        frameon=False, loc="best",
        handletextpad=0.30, labelspacing=0.40,
        borderaxespad=0.25, ncol=1, fontsize=8,
    )


def _legend_handles_sg_dsm_ndsm(include_sg: bool = True) -> List[Line2D]:
    """Combined marker + solid line legend entries (SG grey, DSM/NDSM coloured)."""
    def _h(color: str, label: str) -> Line2D:
        return Line2D(
            [0], [0], marker="o", color=color, lw=1.8, linestyle="-",
            markerfacecolor=color, markeredgecolor="black",
            markeredgewidth=0.5, markersize=5.0, label=label,
        )
    handles: List[Line2D] = []
    if include_sg:
        handles.append(_h(COLOR_SG, "SG"))
    handles.append(_h(COLOR_DSM, "DSM"))
    handles.append(_h(COLOR_NDSM, "NDSM"))
    return handles


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with whitespace-collapsed, stripped column names."""
    new_cols = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    df = df.copy()
    df.columns = new_cols
    return df


def _fmt_column(template: str, species: str) -> str:
    """Substitute a species name into a ``{X}`` column-name template."""
    return template.replace("{X}", species)


def _num(df: pd.DataFrame, row_id: str, col: str) -> float:
    """Safely read ``df.at[row_id, col]`` as float, returning NaN if absent/invalid."""
    if col not in df.columns or row_id not in df.index:
        return np.nan
    try:
        return float(df.at[row_id, col])
    except (TypeError, ValueError):
        return np.nan


def _infer_temp(p: Path) -> Optional[float]:
    """Extract the temperature from a ``TEMP_<n>`` path component (None if absent)."""
    m = re.search(r"TEMP_(\d+)", str(p))
    return float(m.group(1)) if m else None


def _load(csv: Path) -> Optional[pd.DataFrame]:
    """Read one Quant_Data.csv indexed by 'Small Molecule ID', or None on failure."""
    try:
        df = pd.read_csv(csv)
    except Exception as exc:
        print(f"[SI_TEMP_TRENDS] cannot read {csv}: {exc}")
        return None
    if "Small Molecule ID" not in df.columns:
        print(f"[SI_TEMP_TRENDS] missing 'Small Molecule ID' in {csv}")
        return None
    return _normalise_columns(df.set_index("Small Molecule ID"))


def _gather(paths: Iterable[Path]) -> Dict[float, pd.DataFrame]:
    """Load all CSV paths into a {temperature: DataFrame} mapping (skips unreadable)."""
    out: Dict[float, pd.DataFrame] = {}
    for p in sorted(set(paths), key=lambda q: (_infer_temp(q) or 0.0, str(q))):
        T = _infer_temp(p)
        if T is None:
            continue
        df = _load(p)
        if df is not None:
            out[T] = df
    return out


def _trio(df: pd.DataFrame, value_col: str, sig_col: str) -> Tuple[float, float, float, float, float, float]:
    """Return (value, sem) pairs for the SG, DSM_AVG, and NDSM_AVG rows."""
    return (
        _num(df, SG_ID, value_col),       _num(df, SG_ID, sig_col),
        _num(df, DSM_AVG_ID, value_col),  _num(df, DSM_AVG_ID, sig_col),
        _num(df, NDSM_AVG_ID, value_col), _num(df, NDSM_AVG_ID, sig_col),
    )


def _draw_series(ax, t: np.ndarray, v: np.ndarray, sem: np.ndarray, color: str) -> None:
    """Plot one T-vs-value series: solid line, circle markers, and SEM error bars."""
    mask = np.isfinite(t) & np.isfinite(v)
    if not np.any(mask):
        return
    tm, vm = t[mask], v[mask]
    sm = sem[mask] if sem is not None else np.full_like(vm, np.nan)
    ax.plot(tm, vm, linestyle="-", color=color, lw=1.8, zorder=2, clip_on=False)
    ax.plot(tm, vm, marker="o", linestyle="None", color=color, markersize=5,
            markeredgecolor="black", markeredgewidth=0.5, zorder=3, clip_on=False)
    sem_mask = np.isfinite(sm)
    if np.any(sem_mask):
        ax.errorbar(tm[sem_mask], vm[sem_mask], yerr=sm[sem_mask],
                    fmt="none", color=color, ecolor=color,
                    elinewidth=1, capsize=2, capthick=1, zorder=2.5, clip_on=False)


def _gather_global_series(by_temp: Dict[float, pd.DataFrame], value_tmpl: str, sig_tmpl: str):
    """Assemble per-temperature SG/DSM/NDSM value+SEM arrays for a global observable.

    Returns ``(temps, sg, sg_sem, dsm, dsm_sem, nd, nd_sem)`` aligned to sorted temps.
    """
    temps = np.array(sorted(by_temp.keys()), dtype=float)
    arrs = [np.full_like(temps, np.nan) for _ in range(6)]
    vc = re.sub(r"\s+", " ", value_tmpl).strip()
    sc = re.sub(r"\s+", " ", sig_tmpl).strip()
    for i, T in enumerate(temps):
        df = by_temp[T]
        a, b, c, d, e, f = _trio(df, vc, sc)
        arrs[0][i], arrs[1][i] = a, b
        arrs[2][i], arrs[3][i] = c, d
        arrs[4][i], arrs[5][i] = e, f
    return (temps, *arrs)


def _gather_species_series(by_temp: Dict[float, pd.DataFrame], value_tmpl: str, sig_tmpl: str, species: str):
    """Like ``_gather_global_series`` but for one species' per-species observable.

    Returns ``(temps, sg, sg_sem, dsm, dsm_sem, nd, nd_sem)`` for the given species.
    """
    vc = re.sub(r"\s+", " ", _fmt_column(value_tmpl, species)).strip()
    sc = re.sub(r"\s+", " ", _fmt_column(sig_tmpl, species)).strip()
    temps = np.array(sorted(by_temp.keys()), dtype=float)
    arrs = [np.full_like(temps, np.nan) for _ in range(6)]
    for i, T in enumerate(temps):
        df = by_temp[T]
        a, b, c, d, e, f = _trio(df, vc, sc)
        arrs[0][i], arrs[1][i] = a, b
        arrs[2][i], arrs[3][i] = c, d
        arrs[4][i], arrs[5][i] = e, f
    return (temps, *arrs)


def _y_range(*arrs_with_sem: np.ndarray) -> Tuple[float, float]:
    """Compute [floor_2sf(min - sem), ceil_2sf(max + sem)] across all finite points."""
    tops: List[float] = []
    bots: List[float] = []
    pairs = list(zip(arrs_with_sem[0::2], arrs_with_sem[1::2]))
    for vals, sems in pairs:
        for v, s in zip(vals, sems):
            if not np.isfinite(v):
                continue
            err = s if np.isfinite(s) else 0.0
            tops.append(v + err)
            bots.append(v - err)
    if not tops:
        return 0.0, 1.0
    return _floor_2sf(min(bots)), _ceil_2sf(max(tops))


# ---------- plot functions ----------
def _plot_absolute_panel(
    temps: np.ndarray,
    sg: np.ndarray, sg_sem: np.ndarray,
    dsm: np.ndarray, dsm_sem: np.ndarray,
    nd: np.ndarray, nd_sem: np.ndarray,
    output_path: Path,
) -> None:
    """Render and save the absolute (X vs T) panel for SG, DSM, and NDSM series."""
    fig, ax = _make_square_axes()
    _draw_series(ax, temps, sg, sg_sem, COLOR_SG)
    _draw_series(ax, temps, dsm, dsm_sem, COLOR_DSM)
    _draw_series(ax, temps, nd, nd_sem, COLOR_NDSM)
    y_low, y_high = _y_range(sg, sg_sem, dsm, dsm_sem, nd, nd_sem)
    _apply_2sig_yticks_sci(ax, y_low, y_high, n_ticks=5)
    _style_x_temperature(ax)
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title("")
    _legend_best(ax, _legend_handles_sg_dsm_ndsm(include_sg=True))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=400)
    plt.close(fig)


def _plot_delta_panel(
    temps: np.ndarray,
    sg: np.ndarray, sg_sem: np.ndarray,
    dsm: np.ndarray, dsm_sem: np.ndarray,
    nd: np.ndarray, nd_sem: np.ndarray,
    output_path: Path,
) -> None:
    """Render and save the delta panel: (DSM-SG) and (NDSM-SG) vs T.

    SEM is propagated in quadrature from the class and SG uncertainties; a dashed
    zero line marks the SG reference.
    """
    d_dsm = dsm - sg
    d_nd = nd - sg
    s_dsm = np.where(np.isfinite(dsm_sem) & np.isfinite(sg_sem),
                     np.sqrt(np.nan_to_num(dsm_sem) ** 2 + np.nan_to_num(sg_sem) ** 2),
                     np.nan)
    s_nd = np.where(np.isfinite(nd_sem) & np.isfinite(sg_sem),
                    np.sqrt(np.nan_to_num(nd_sem) ** 2 + np.nan_to_num(sg_sem) ** 2),
                    np.nan)

    fig, ax = _make_square_axes()
    ax.axhline(0.0, color=COLOR_SG, lw=1, ls="--", zorder=1.5)
    _draw_series(ax, temps, d_dsm, s_dsm, COLOR_DSM)
    _draw_series(ax, temps, d_nd, s_nd, COLOR_NDSM)
    y_low, y_high = _y_range(d_dsm, s_dsm, d_nd, s_nd)
    _apply_2sig_yticks_sci(ax, y_low, y_high, n_ticks=5)
    _style_x_temperature(ax)
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title("")
    _legend_best(ax, _legend_handles_sg_dsm_ndsm(include_sg=False))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=400)
    plt.close(fig)


def plot_temperature_trends(csv_paths: Iterable[Path], output_dir: Path) -> int:
    """Render every global and per-species T-trend panel (absolute + delta).

    Loads all CSVs by temperature, then for each available observable writes the
    absolute and delta PNGs into ``output_dir``. Returns the number of
    temperatures processed.
    """
    by_temp = _gather(csv_paths)
    if not by_temp:
        print("[SI_TEMP_TRENDS] no usable Quant_Data.csv files found")
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[SI_TEMP_TRENDS] {len(by_temp)} temperatures: {sorted(by_temp.keys())}")

    for slug, vtmpl, stmpl, _label in GLOBAL_OBSERVABLES:
        any_match = any(re.sub(r"\s+", " ", vtmpl).strip() in df.columns for df in by_temp.values())
        if not any_match:
            print(f"[SI_TEMP_TRENDS] skip global {slug}: column not found")
            continue
        temps, sg, sg_sem, dsm, dsm_sem, nd, nd_sem = _gather_global_series(by_temp, vtmpl, stmpl)
        _plot_absolute_panel(temps, sg, sg_sem, dsm, dsm_sem, nd, nd_sem,
                             output_dir / f"global_{slug}_vs_T.png")
        _plot_delta_panel(temps, sg, sg_sem, dsm, dsm_sem, nd, nd_sem,
                          output_dir / f"global_{slug}_delta_vs_T.png")
        print(f"[SI_TEMP_TRENDS] wrote global_{slug}_vs_T.png and global_{slug}_delta_vs_T.png")

    for slug, vtmpl, stmpl, _label in PER_SPECIES_OBSERVABLES:
        any_match = False
        for df in by_temp.values():
            for sp in SPECIES_ORDER:
                if re.sub(r"\s+", " ", _fmt_column(vtmpl, sp)).strip() in df.columns:
                    any_match = True
                    break
            if any_match:
                break
        if not any_match:
            print(f"[SI_TEMP_TRENDS] skip per-species {slug}: columns not found")
            continue
        for sp in SPECIES_ORDER:
            temps, sg, sg_sem, dsm, dsm_sem, nd, nd_sem = _gather_species_series(by_temp, vtmpl, stmpl, sp)
            _plot_absolute_panel(temps, sg, sg_sem, dsm, dsm_sem, nd, nd_sem,
                                 output_dir / f"perspecies_{slug}_{sp}_vs_T.png")
            _plot_delta_panel(temps, sg, sg_sem, dsm, dsm_sem, nd, nd_sem,
                              output_dir / f"perspecies_{slug}_{sp}_delta_vs_T.png")
        print(f"[SI_TEMP_TRENDS] wrote perspecies_{slug}_<species>_vs_T.png and *_delta_vs_T.png (7 species each)")
    return len(by_temp)


def _expand_globs(patterns: Iterable[str]) -> List[Path]:
    """Expand glob patterns to a sorted, de-duplicated list of Paths."""
    out: List[Path] = []
    for pat in patterns:
        out.extend(Path(p) for p in globlib.glob(pat))
    return sorted(set(out))


def main() -> None:
    """CLI entry point: resolve CSVs from globs/paths and render all trend panels.

    Falls back to the default non-CORRELATED Quant_Data.csv glob if no inputs are
    supplied, then calls ``plot_temperature_trends`` into ``--output-dir``.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary-glob", action="append", default=None)
    p.add_argument("--csv", action="append", default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    csvs: List[Path] = []
    if args.summary_glob:
        csvs.extend(_expand_globs(args.summary_glob))
    if args.csv:
        csvs.extend(Path(c) for c in args.csv)
    if not csvs:
        csvs = _expand_globs(["TEMP_*/CLASSIFY_*_50_50_2000/RESULTS/SUMMARY/Quant_Data.csv"])
        csvs = [c for c in csvs if "CORRELATED" not in str(c)]
    n = plot_temperature_trends(csvs, args.output_dir)
    print(f"[SI_TEMP_TRENDS] processed {n} temperatures into {args.output_dir}")


if __name__ == "__main__":
    main()
