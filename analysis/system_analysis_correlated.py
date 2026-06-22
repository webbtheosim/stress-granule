#!/usr/bin/env python3
"""Correlated system-level analysis: apply correlation-corrected errors (Step 3).

Pipeline role
-------------
Correlated-pipeline analogue of ``system_analysis.py``. It runs the same
``system_analysis`` engine (imported from ``SYSTEM_ANALYSIS_FINAL``) over a *blocked*
analysis root produced by ``average_simulations_correlated.py``, then overwrites
the time-correlated observables in ``Quant_Data.csv`` with the
Flyvbjerg-Petersen correlation-corrected means and SEMs computed by
``block_correlation_diagnostics.py`` (read from ``CORRELATION_QUANT_DATA.csv``).
Per-system rows, class-average rows (DSM_AVG / NDSM_AVG), and per-segment
viscosity are all replaced with their corrected values; observables without a
native per-window/per-segment scalar are left as inherited single-run
aggregates, and a provenance sidecar labels every column accordingly.

Key inputs
----------
- A blocked analysis root (``--analysis-root`` or
  ``--path``/``--folder``/``--T``/``--dt``/``--tmin``/``--tmax``) with
  ``ANALYSIS_*_AVE`` outputs and ``RESULTS/CORRELATION_DIAGNOSTICS/`` CSVs
  (``CORRELATION_QUANT_DATA.csv``, ``WINDOW_SCALARS.csv``,
  ``VISCOSITY_SEGMENT_SCALARS.csv``, ``SUMMARY_COMPARISON.csv``,
  ``RECOMMENDED_SYSTEM_BLOCK_LAYOUT.csv``).
- CLI flags as above plus ``--c``, ``--plot-only``, species selectors, and the
  full diffusion/viscosity override flags.

Key outputs (under ``{analysis_root}/RESULTS/SUMMARY/`` and contact-map dirs)
- ``Quant_Data.csv`` (correlation-corrected) plus ``SG/DSM/NDSM_Quant_Data.csv``,
  ``Quant_Data_LONG.csv``, ``Quant_Data_CI95.csv`` (t-distribution CIs),
  ``Quant_Data_PROVENANCE.csv``, reportable diagnostics tables, and the
  combined DSM/NDSM standardized contact heatmaps.

Example invocation
-------------------
    python system_analysis_correlated.py \
        --path TEMP_300 --folder CLASSIFY_BLOCKED \
        --T 300 --dt 50 --tmin 50 --tmax 2000
"""
import argparse
import math
import os
import warnings

import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None

from average_simulations_correlated import DEFAULT_DSM_LIST, DEFAULT_NDSM_LIST, load_layout
from block_correlation_diagnostics import (
    extract_viscosity_scalars_for_system,
    extract_window_scalars_for_system,
)
from system_analysis import (
    system_analysis,
    _contact_1d_sem,
    _contact_matrix_sem,
    _fill_aggregate_spatial_from_members,
    _prepare_acid_rows,
    _prepare_acid_square,
    _prepare_residue_rows,
    _prepare_residue_square,
    _ratio_difference_sem,
    _read_contact_matrix,
)

warnings.filterwarnings("ignore")


SUMMARY_FOLDERS = [
    "IMAGES", "FIGURES", "RESULTS",
    "IMAGES/RDP", "IMAGES/RESIDUE_CONTACT_MAPS", "IMAGES/ACID_CONTACT_MAPS", "IMAGES/SM_CONTACT_MAPS", "IMAGES/DYNAMICS",
    "FIGURES/RDP", "FIGURES/RESIDUE_CONTACT_MAPS", "FIGURES/ACID_CONTACT_MAPS", "FIGURES/SM_CONTACT_MAPS", "FIGURES/PROPERTIES", "FIGURES/TIME",
    "RESULTS/RDP", "RESULTS/RESIDUE_CONTACT_MAPS", "RESULTS/ACID_CONTACT_MAPS", "RESULTS/SM_CONTACT_MAPS", "RESULTS/SUMMARY",
]

WINDOW_SAMPLING_FAMILY = "diffusion_segment"
VISCOSITY_SAMPLING_FAMILY = "viscosity_segment"

OBSERVABLE_TO_QUANT_COLUMN = {
    "c_dense_sg_fit": "$c_{dense,SG,fit}$ $(mg/ml)$",
    "c_dilute_sg_fit": "$c_{dilute,SG,fit}$ $(mg/ml)$",
    "c_dense_sg_calc": "$c_{dense,SG,calc}$ $(mg/ml)$",
    "c_dilute_sg_calc": "$c_{dilute,SG,calc}$ $(mg/ml)$",
    "P_sg": "$P_{SG}$",
    "R_cond_A": r"$R_{cond}$ $(\AA)$",
    "W_interface_A": r"$W_{interface}$ $(\AA)$",
    "gamma1_mN_m": r"$\gamma_{1}$ $(mN/m)$",
    "gamma2_mN_m": r"$\gamma_{2}$ $(mN/m)$",
    "gamma_ave_mN_m": r"$\gamma_ave$ $(mN/m)$",
    "deltaG_trans_kJ_mol": r"$\Delta G_{trans}$ $(kJ/mol)$",
    "c_dilute_sm_mg_ml": "$c_{dilute,SM}$ $(mg/ml)$",
    "c_dense_sm_mg_ml": "$c_{dense,SM}$ $(mg/ml)$",
    "P_sm": "$P_{SM}$",
    "phi_D": r"$\phi_{D}$",
    "N_D": "$N_{D}$",
    "R_g_A": "$R_{g}$",
    "phi_R": r"$\phi_{R}$",
    "Rg_conf_A": r"$R_{g,conf}$ A",
    "Rh_conf_A": r"$R_{h,conf}$ A",
    "eta_GK_Pa_s": r"$\eta_{GK}$ Pa s",
    "eta_GK_Theo_Pa_s": r"$\eta_{GK Theo}$ Pa s",
    "D_SE_GK_Rh_um2_s": r"$D_{SE,GK,Rh}$ $\mu m^{2} / s$",
    "D_SE_GK_Rg_um2_s": r"$D_{SE,GK,Rg}$ $\mu m^{2} / s$",
    "gamma1_protein_mN_m": r"$\gamma_{1,Protein}$ $(mN/m)$",
    "gamma2_protein_mN_m": r"$\gamma_{2,Protein}$ $(mN/m)$",
    "gamma_ave_protein_mN_m": r"$\gamma_{ave,Protein}$ $(mN/m)$",
    "gamma1_rna_mN_m": r"$\gamma_{1,RNA}$ $(mN/m)$",
    "gamma2_rna_mN_m": r"$\gamma_{2,RNA}$ $(mN/m)$",
    "gamma_ave_rna_mN_m": r"$\gamma_{ave,RNA}$ $(mN/m)$",
}

_PER_SPECIES_FOR_SPATIAL = ("G3BP1", "PABP1", "TIA1", "TTP", "FUS", "TDP43", "RNA")
for _sp in _PER_SPECIES_FOR_SPATIAL:
    OBSERVABLE_TO_QUANT_COLUMN[f"Rg_{_sp}"] = f"$R_{{g,{_sp}}}$ A"
    OBSERVABLE_TO_QUANT_COLUMN[f"Rh_{_sp}"] = f"$R_{{h,{_sp}}}$ A"
    OBSERVABLE_TO_QUANT_COLUMN[f"Occ_{_sp}"] = f"$Occ_{{{_sp}}}$"
    OBSERVABLE_TO_QUANT_COLUMN[f"r_over_R_{_sp}"] = f"$r/R_{{{_sp}}}$"
    OBSERVABLE_TO_QUANT_COLUMN[f"D_SE_GK_{_sp}_um2_s"] = f"$D_{{SE,GK,{_sp}}}$ $\\mu m^{{2}} / s$"


def ensure_dir(path: str) -> None:
    """Create ``path`` (and parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def resolve_analysis_root(args) -> str:
    """Return the absolute blocked analysis root from CLI args.

    Uses ``--analysis-root`` if given, otherwise derives
    ``{path}/{folder}_{T}_{dt}_{tmin}_{tmax}``; raises SystemExit if neither is
    fully specified.
    """
    if args.analysis_root:
        return os.path.abspath(args.analysis_root)
    if not all(v is not None for v in [args.path, args.folder, args.T, args.dt, args.tmin, args.tmax]):
        raise SystemExit("Pass either --analysis-root or the full --path/--folder/--T/--dt/--tmin/--tmax set")
    return os.path.abspath(os.path.join(args.path, f"{args.folder}_{args.T}_{args.dt}_{args.tmin}_{args.tmax}"))


def resolve_layout_csv(args, analysis_root: str) -> str:
    """Return the recommended block-layout CSV path (``--layout-csv`` or default).

    Falls back to
    ``RESULTS/CORRELATION_DIAGNOSTICS/RECOMMENDED_SYSTEM_BLOCK_LAYOUT.csv`` under
    the analysis root; raises SystemExit if it is missing.
    """
    if args.layout_csv:
        return os.path.abspath(args.layout_csv)
    candidate = os.path.join(analysis_root, "RESULTS", "CORRELATION_DIAGNOSTICS", "RECOMMENDED_SYSTEM_BLOCK_LAYOUT.csv")
    if not os.path.isfile(candidate):
        raise SystemExit(f"Layout CSV not found: {candidate}")
    return candidate


def load_system_list(analysis_root: str, filename: str, defaults):
    """Return system names from ``{analysis_root}/{filename}`` or ``defaults``."""
    path = os.path.join(analysis_root, filename)
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as handle:
            items = [line.strip() for line in handle if line.strip()]
        return items
    return list(defaults)


def build_analysis(args, analysis_root: str) -> system_analysis:
    """Construct the ``system_analysis`` engine with CLI-supplied diffusion/viscosity overrides.

    Assembles the diffusion and viscosity parameter dicts and optional species
    selectors from ``args`` and returns a ``system_analysis`` rooted at
    ``analysis_root``.
    """
    diffusion_overrides = {
        "downsample_stride": args.diff_downsample,
        "dt_diff": args.diff_dt_diff,
        "segments": args.diff_segments,
        "n_boot": args.diff_n_boot,
        "slope_iterations": args.diff_slope_iters,
        "dt_pts": args.diff_dt_pts,
        "slope_tol": args.diff_slope_tol,
        "min_diff_pts": args.diff_min_diff_pts,
        "slope_tol_plateau": args.diff_slope_plateau,
        "min_plateau_pts": args.diff_min_plateau_pts,
        "boot_r2": args.diff_boot_r2,
        "smooth_window_pts": args.diff_smooth_window,
        "smooth_polyorder": args.diff_smooth_polyorder,
        "seed": args.diff_seed,
        "min_segment_ns": args.diff_min_segment_ns,
        "min_primary_fraction": args.diff_min_primary_fraction,
        "origin_resample": None if args.diff_origin_resample is None else bool(args.diff_origin_resample),
        "origin_window_ns": args.diff_origin_window_ns,
        "origin_stride_ns": args.diff_origin_stride_ns,
        "origin_candidate_windows_ns": args.diff_origin_candidate_windows_ns,
        "origin_min_success_fraction": args.diff_origin_min_success_fraction,
        "origin_min_success_count": args.diff_origin_min_success_count,
        "origin_max_origins": args.diff_origin_max_origins,
    }
    viscosity_overrides = {
        "segments": args.visc_segments,
        "iterations": args.visc_iterations,
        "n_boot": args.visc_n_boot,
        "dt_unit": args.visc_dt_unit,
        "n_point": args.visc_n_point,
        "n_tau": args.visc_n_tau,
        "seed": args.visc_seed,
    }
    species_override = args.species if args.species else None
    species_all_override = args.species_all if args.species_all else None
    return system_analysis(
        diffusion_params=diffusion_overrides,
        viscosity_params=viscosity_overrides,
        species_to_analyze=species_override,
        species_for_global=species_all_override,
        data_root=analysis_root,
    )


def ensure_output_scaffold(analysis_root: str) -> None:
    """Create the standard IMAGES/FIGURES/RESULTS output subdirectories."""
    for folder in SUMMARY_FOLDERS:
        ensure_dir(os.path.join(analysis_root, folder))


def block_summary(values):
    """Return (mean, SEM, n_blocks) from block-averaged values.

    n_blocks is the effective degrees of freedom for the SEM.
    Downstream consumers can use this for t-distribution CIs:
        CI_95 = mean ± t_{n-1}(0.975) * SEM
    """
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return math.nan, math.nan, 0
    mean = float(np.mean(vals))
    n = int(vals.size)
    if n <= 1:
        return mean, math.nan, n
    sem = float(np.std(vals, ddof=1) / np.sqrt(n))
    return mean, sem, n


def flatten_windows(blocks):
    """Return the sorted unique set of window-start times across all blocks."""
    if not blocks:
        return []
    return sorted({int(w) for block in blocks for w in block})


def mean_from_values(values):
    """Return the mean over finite values (NaN if there are none)."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return math.nan
    return float(np.mean(vals))


def sem_from_values(values):
    """Return the standard error of the mean over finite values (NaN if <=1)."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size <= 1:
        return math.nan
    return float(np.std(vals, ddof=1) / np.sqrt(vals.size))


def load_window_scalars_df(analysis_root: str):
    """Load ``WINDOW_SCALARS.csv`` from the diagnostics dir (None if absent)."""
    csv_path = os.path.join(analysis_root, "RESULTS", "CORRELATION_DIAGNOSTICS", "WINDOW_SCALARS.csv")
    if not os.path.isfile(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def load_viscosity_scalars_df(analysis_root: str):
    """Load ``VISCOSITY_SEGMENT_SCALARS.csv`` from the diagnostics dir (None if absent)."""
    csv_path = os.path.join(analysis_root, "RESULTS", "CORRELATION_DIAGNOSTICS", "VISCOSITY_SEGMENT_SCALARS.csv")
    if not os.path.isfile(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def load_correlation_summary_df(analysis_root: str):
    """Load the correlation-corrected summary ``CORRELATION_QUANT_DATA.csv`` (None if absent).

    This table holds the per-observable corrected mean/SEM and block metadata
    that overwrite the inherited single-run values in Quant_Data.csv.
    """
    csv_path = os.path.join(analysis_root, "RESULTS", "CORRELATION_DIAGNOSTICS", "CORRELATION_QUANT_DATA.csv")
    if not os.path.isfile(csv_path):
        return None
    df = pd.read_csv(csv_path)
    for col in [
        "temperature_K",
        "corrected_mean",
        "corrected_sem",
        "corrected_block_size_units",
        "corrected_block_size_ns",
        "n_corrected_blocks",
        "n_offsets_total",
        "n_offsets_used",
        "offset_mean_average",
        "offset_mean_sd",
        "offset_sem_min",
        "offset_sem_median",
        "offset_sem_max",
        "n_blocks_min",
        "n_blocks_median",
        "n_blocks_max",
        "n_replicates",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_correlation_long_df(analysis_root: str):
    """Load the per-window long-format ``CORRELATION_QUANT_DATA_LONG.csv`` (None if absent)."""
    csv_path = os.path.join(analysis_root, "RESULTS", "CORRELATION_DIAGNOSTICS", "CORRELATION_QUANT_DATA_LONG.csv")
    if not os.path.isfile(csv_path):
        return None
    df = pd.read_csv(csv_path)
    for col in ["temperature_K", "window_index", "window_start_ns", "window_end_ns", "value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_optional_layout(layout_csv: str):
    """Load a block-layout CSV if it exists, else return an empty dict."""
    if not os.path.isfile(layout_csv):
        return {}
    return load_layout(layout_csv)


def load_summary_comparison_df(analysis_root: str):
    """Load the per-observable SUMMARY_COMPARISON.csv from correlation diagnostics.

    This contains tau_int_ns, n_recommended_blocks, g, and equilibration
    diagnostics for each observable per system.
    """
    csv_path = os.path.join(analysis_root, "RESULTS", "CORRELATION_DIAGNOSTICS", "SUMMARY_COMPARISON.csv")
    if not os.path.isfile(csv_path):
        return None
    df = pd.read_csv(csv_path)
    for col in ["tau_int_ns", "tau_int_windows", "g", "n_eff_pymbar",
                 "n_recommended_blocks", "t_equil_ns", "equil_n_discarded",
                 "equil_fraction_discarded", "equil_g", "equil_n_eff"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_system_blocks_df(analysis_root: str):
    """Load RECOMMENDED_SYSTEM_BLOCKS.csv with system-level n_blocks and tau_int."""
    csv_path = os.path.join(analysis_root, "RESULTS", "CORRELATION_DIAGNOSTICS", "RECOMMENDED_SYSTEM_BLOCKS.csv")
    if not os.path.isfile(csv_path):
        return None
    df = pd.read_csv(csv_path)
    for col in ["n_blocks", "tau_int_max_ns", "tau_int_median_ns",
                 "equil_t_max_ns", "equil_t_max_index"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_optional_diagnostics_csv(analysis_root: str, filename: str):
    """Load a named CSV from the diagnostics dir, or None if it does not exist."""
    csv_path = os.path.join(analysis_root, "RESULTS", "CORRELATION_DIAGNOSTICS", filename)
    if not os.path.isfile(csv_path):
        return None
    return pd.read_csv(csv_path)


def block_means_from_rows(df: pd.DataFrame, blocks, observable_key: str):
    """Return the per-block means of one observable from long-format rows.

    Builds a window->value map for ``observable_key`` and returns the mean over
    the finite members of each block.
    """
    sub = df[df["observable_key"] == observable_key][["window_start_ns", "value"]].copy()
    if sub.empty:
        return []
    sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
    window_map = {
        int(row.window_start_ns): float(row.value)
        for row in sub.itertuples(index=False)
        if np.isfinite(row.value)
    }
    out = []
    for block in blocks:
        vals = [window_map[int(w)] for w in block if int(w) in window_map]
        if vals:
            out.append(float(np.mean(vals)))
    return out


def lookup_corrected_summary_rows(correlation_summary_df, category: str, system_name: str, observable_keys, sampling_unit_family: str):
    """Select corrected-summary rows for one system, observables and sampling family.

    Returns the de-duplicated matching rows (last per observable_key), or None
    if the table is empty or nothing matches.
    """
    if correlation_summary_df is None or correlation_summary_df.empty:
        return None
    sub = correlation_summary_df[
        (correlation_summary_df["category"] == category)
        & (correlation_summary_df["system_name"] == system_name)
        & (correlation_summary_df["observable_key"].isin(observable_keys))
    ].copy()
    if sampling_unit_family is not None and "sampling_unit_family" in sub.columns:
        sub = sub[sub["sampling_unit_family"] == sampling_unit_family]
    if sub.empty:
        return None
    return sub.sort_values(["observable_key"]).drop_duplicates(subset=["observable_key"], keep="last")


def corrected_summary_scalar(rows_df, observable_key: str):
    """Return ``(corrected_mean, corrected_sem)`` for one observable, or (NaN, NaN)."""
    if rows_df is None or rows_df.empty:
        return math.nan, math.nan
    sub = rows_df[rows_df["observable_key"] == observable_key]
    if sub.empty:
        return math.nan, math.nan
    mean = float(pd.to_numeric(sub["corrected_mean"], errors="coerce").iloc[-1])
    sem = float(pd.to_numeric(sub["corrected_sem"], errors="coerce").iloc[-1])
    return mean, sem


def corrected_class_scalar(correlation_summary_df, category: str, system_names, observable_key: str, sampling_unit_family: str):
    """Return the class-average ``(mean, sem)`` of one corrected observable.

    Averages the per-system corrected means across ``system_names`` and reports
    the SEM across those system means.
    """
    if correlation_summary_df is None or correlation_summary_df.empty:
        return math.nan, math.nan
    sub = correlation_summary_df[
        (correlation_summary_df["category"] == category)
        & (correlation_summary_df["system_name"].isin(system_names))
        & (correlation_summary_df["observable_key"] == observable_key)
    ].copy()
    if sampling_unit_family is not None and "sampling_unit_family" in sub.columns:
        sub = sub[sub["sampling_unit_family"] == sampling_unit_family]
    means = pd.to_numeric(sub.get("corrected_mean", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
    means = means[np.isfinite(means)]
    return mean_from_values(means), sem_from_values(means)


def build_class_average_viscosity(correlation_summary_df, category: str, system_names):
    """Return class-average corrected viscosity ``(eta, eta_sem, theo, theo_sem)``.

    Returns None if neither the raw nor theoretical Green-Kubo viscosity is
    finite for the class.
    """
    eta_mean, eta_sem = corrected_class_scalar(correlation_summary_df, category, system_names, "eta_GK_Pa_s", VISCOSITY_SAMPLING_FAMILY)
    theo_mean, theo_sem = corrected_class_scalar(correlation_summary_df, category, system_names, "eta_GK_Theo_Pa_s", VISCOSITY_SAMPLING_FAMILY)
    if not any(np.isfinite(x) for x in [eta_mean, theo_mean]):
        return None
    return eta_mean, eta_sem, theo_mean, theo_sem


def sampling_family_for_observable(observable_key: str) -> str:
    """Map an observable key to its sampling-unit family (viscosity vs window)."""
    if str(observable_key).startswith("D_SE_GK_"):
        return VISCOSITY_SAMPLING_FAMILY
    if observable_key in {"eta_GK_Pa_s", "eta_GK_Theo_Pa_s"}:
        return VISCOSITY_SAMPLING_FAMILY
    return WINDOW_SAMPLING_FAMILY


def apply_corrected_summary_to_row(df: pd.DataFrame, row_name: str, category: str, system_name: str, correlation_summary_df) -> pd.DataFrame:
    """Overwrite one system's Quant_Data row with correlation-corrected values.

    For every mapped observable, writes the corrected mean into its Quant_Data
    column and the corrected SEM into the matching ``SIG`` column, always
    overwriting (NaN included) so corrected provenance is never mixed with
    legacy single-run values. Returns the modified DataFrame.
    """
    if df.empty or correlation_summary_df is None or correlation_summary_df.empty:
        return df
    row_mask = df["Small Molecule Name"] == row_name
    if not row_mask.any():
        return df
    row_idx = df.index[row_mask][-1]
    summary_rows = correlation_summary_df[
        (correlation_summary_df["category"] == category)
        & (correlation_summary_df["system_name"] == system_name)
    ].copy()
    if summary_rows.empty:
        return df
    for record in summary_rows.itertuples(index=False):
        quant_col = OBSERVABLE_TO_QUANT_COLUMN.get(str(record.observable_key))
        if not quant_col:
            continue
        if quant_col not in df.columns:
            df[quant_col] = np.nan
        corrected_mean = float(pd.to_numeric(pd.Series([record.corrected_mean]), errors="coerce").iloc[0])
        corrected_sem = float(pd.to_numeric(pd.Series([record.corrected_sem]), errors="coerce").iloc[0])
        # Always overwrite mapped correlated observables, including NaN. If a
        # correlated estimator fails, the table must show undefined rather than
        # silently retaining the legacy single-run value under corrected
        # provenance.
        if quant_col in df.columns:
            df.at[row_idx, quant_col] = corrected_mean
        sig_col = f"SIG{quant_col}"
        if sig_col not in df.columns:
            df[sig_col] = np.nan
        if sig_col in df.columns:
            df.at[row_idx, sig_col] = corrected_sem
    return df


def apply_corrected_class_stats_to_row(df: pd.DataFrame, row_name: str, category: str, system_names, correlation_summary_df) -> pd.DataFrame:
    """Overwrite a class-average row (DSM/NDSM) with corrected class statistics.

    For every mapped observable, writes the class-average corrected mean and the
    SEM across system means into the row and its ``SIG`` column (NaN means the
    class value is genuinely undefined). Returns the modified DataFrame.
    """
    if df.empty or correlation_summary_df is None or correlation_summary_df.empty:
        return df
    row_mask = df["Small Molecule Name"] == row_name
    if not row_mask.any():
        return df
    row_idx = df.index[row_mask][-1]
    for observable_key, quant_col in OBSERVABLE_TO_QUANT_COLUMN.items():
        mean_val, sem_val = corrected_class_scalar(
            correlation_summary_df,
            category,
            system_names,
            observable_key,
            sampling_family_for_observable(observable_key),
        )
        if quant_col not in df.columns:
            df[quant_col] = np.nan
        # Same rule for class rows: mapped correlated columns are authoritative.
        # NaN means the correlated class-level value is genuinely undefined.
        if quant_col in df.columns:
            df.at[row_idx, quant_col] = mean_val
        sig_col = f"SIG{quant_col}"
        if sig_col not in df.columns:
            df[sig_col] = np.nan
        if sig_col in df.columns:
            df.at[row_idx, sig_col] = sem_val
    return df


def append_class_average_row(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Append a class-average row (mean of values, SEM for ``SIG`` columns).

    Names the new row with ``label`` (e.g. ``DSM_AVG``), sets its Compound Class
    to the class prefix, and fills numeric columns with column means (``SIG``
    columns with the SEM of their base column). Returns the extended DataFrame.
    """
    if df.empty:
        return df
    source = df.copy()
    class_label = label.split("_")[0]
    row = {}
    for column_name in source.columns:
        if column_name in {"Small Molecule ID", "Small Molecule Name", "Compound Name"}:
            row[column_name] = label
            continue
        if column_name == "Compound Class":
            row[column_name] = class_label
            continue
        if column_name.startswith("SIG"):
            base_column = column_name[3:]
            if base_column in source.columns:
                row[column_name] = sem_from_values(pd.to_numeric(source[base_column], errors="coerce").to_numpy(dtype=float))
                continue
        numeric_vals = pd.to_numeric(source[column_name], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(numeric_vals).any():
            row[column_name] = mean_from_values(numeric_vals)
        else:
            row[column_name] = np.nan
    return pd.concat([source, pd.DataFrame([row], columns=source.columns)], ignore_index=True)


def annotate_quant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``quant_data_column`` / ``quant_data_sig_column`` from observable keys."""
    out = df.copy()
    out["quant_data_column"] = out["observable_key"].map(OBSERVABLE_TO_QUANT_COLUMN)
    out["quant_data_sig_column"] = out["quant_data_column"].apply(
        lambda x: f"SIG{x}" if isinstance(x, str) and x else math.nan
    )
    return out


def observable_family(observable_key: str) -> str:
    """Classify an observable key into its physical family label (for provenance)."""
    key = str(observable_key)
    if key.startswith("D_SE_GK_"):
        return "stokes_einstein_inferred_diffusion"
    if key in {"eta_GK_Pa_s", "eta_GK_Theo_Pa_s"}:
        return "viscosity_green_kubo"
    if key in {"Rg_conf_A", "Rh_conf_A"}:
        return "confinement_radius"
    if key in {"phi_D", "N_D", "R_g_A", "phi_R"}:
        return "cluster_scalar"
    if key.startswith(("Rg_", "Rh_", "Occ_", "r_over_R_")):
        return "species_spatial_scalar"
    if key.startswith("gamma"):
        return "interfacial_fluctuation"
    if key in {"c_dense_sg_fit", "c_dilute_sg_fit", "c_dense_sg_calc", "c_dilute_sg_calc", "P_sg", "R_cond_A", "W_interface_A", "deltaG_trans_kJ_mol", "c_dense_sm_mg_ml", "c_dilute_sm_mg_ml", "P_sm"}:
        return "phase_fit_scalar"
    return "other"


def default_provenance_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Fill default provenance columns (family, central/uncertainty source, scope).

    Adds the observable family and, based on the sampling-unit family, the
    central-value and uncertainty provenance labels plus the within-run
    correlation-corrected scope and ``n_replicates=1``.
    """
    out = df.copy()
    if "observable_key" in out.columns:
        out["observable_family"] = out["observable_key"].map(observable_family)
    if "sampling_unit_family" in out.columns:
        out["central_value_provenance"] = out.get("central_value_provenance", np.nan)
        out["uncertainty_provenance"] = out.get("uncertainty_provenance", np.nan)
        window_mask = out["sampling_unit_family"] == WINDOW_SAMPLING_FAMILY
        visc_mask = out["sampling_unit_family"] == VISCOSITY_SAMPLING_FAMILY
        dse_mask = out.get("observable_key", pd.Series(dtype=object)).astype(str).str.startswith("D_SE_GK_")
        out.loc[window_mask & out["central_value_provenance"].isna(), "central_value_provenance"] = "per_window_derived_scalar_mean"
        out.loc[visc_mask & out["central_value_provenance"].isna(), "central_value_provenance"] = "per_segment_green_kubo_scalar_mean"
        out.loc[dse_mask, "central_value_provenance"] = "per_segment_stokes_einstein_from_segment_eta_and_window_radius"
        out.loc[out["uncertainty_provenance"].isna(), "uncertainty_provenance"] = "all_offset_nonoverlapping_batch_means_median_sem"
    if "uncertainty_scope" not in out.columns:
        out["uncertainty_scope"] = "within_run_time_correlation_corrected"
    if "n_replicates" not in out.columns:
        out["n_replicates"] = 1
    return out


def build_observable_provenance_table(correlation_summary_df):
    """Build the per-observable provenance table (one row per observable/family).

    Returns a de-duplicated DataFrame mapping each observable to its Quant_Data
    column, family, and central/uncertainty provenance labels.
    """
    cols = [
        "sampling_unit_family",
        "observable_key",
        "observable_family",
        "quant_data_column",
        "quant_data_sig_column",
        "central_value_provenance",
        "uncertainty_provenance",
        "uncertainty_scope",
        "n_replicates",
    ]
    if correlation_summary_df is None or correlation_summary_df.empty:
        return pd.DataFrame(columns=cols)
    df = default_provenance_fields(annotate_quant_columns(correlation_summary_df))
    df = df[cols].drop_duplicates().sort_values(["sampling_unit_family", "observable_key"])
    return df.reset_index(drop=True)


def ci95_multiplier(n_samples):
    """Return the two-sided 95% Student-t multiplier for ``n_samples`` blocks.

    Uses t with ``n-1`` degrees of freedom (1.96 fallback if SciPy is missing);
    NaN for ``n<=1``.
    """
    try:
        n = int(round(float(n_samples)))
    except Exception:
        return math.nan
    if n <= 1:
        return math.nan
    if scipy_stats is None:
        return 1.96
    return float(scipy_stats.t.ppf(0.975, n - 1))


def build_quant_data_long_sidecar(correlation_long_df):
    """Build the ``Quant_Data_LONG.csv`` sidecar from the per-window long rows.

    Renames the diagnostic columns to the Quant_Data naming and annotates each
    row with its Quant_Data value/sig column; returns an empty framed schema if
    there is no input.
    """
    cols = [
        "temperature_K",
        "category",
        "Small Molecule Name",
        "Small Molecule ID",
        "sampling_unit_family",
        "observable_key",
        "quant_data_column",
        "quant_data_sig_column",
        "window_index",
        "window_start_ns",
        "window_end_ns",
        "sampling_unit_value",
        "fit_ok",
        "source_density_file",
        "source_pca_file",
        "source_cluster_file",
        "note",
    ]
    if correlation_long_df is None or correlation_long_df.empty:
        return pd.DataFrame(columns=cols)
    df = annotate_quant_columns(correlation_long_df)
    df = df.rename(
        columns={
            "system_name": "Small Molecule Name",
            "small_molecule_id": "Small Molecule ID",
            "value": "sampling_unit_value",
        }
    )
    for col in cols:
        if col not in df.columns:
            df[col] = math.nan
    return df[cols]


QUANT_DATA_METADATA_COLUMNS = {
    "Small Molecule ID",
    "Small Molecule Name",
    "Compound Name",
    "Compound Class",
    "Temperature K",
    "Temperature",
}


def build_quant_data_provenance_sidecar(df_all_clean):
    """Enumerate every Quant_Data.csv column and label provenance.

    The correlated Quant_Data.csv carries a mix of correlation-corrected
    values (for observables with per-window/per-segment sampling in the
    diagnostic pipeline) and inherited single-run aggregates (for
    observables such as the two-stage confinement fit, which still have no
    native per-window/per-segment scalar and therefore cannot be
    Flyvbjerg-Petersen corrected). This sidecar labels each column
    explicitly so downstream consumers never silently treat an inherited
    single-run SEM as a correlation-corrected one.
    """
    cols = [
        "quant_data_column",
        "column_kind",
        "provenance",
        "central_value_provenance",
        "uncertainty_provenance",
        "uncertainty_scope",
        "observable_key",
    ]
    if df_all_clean is None or df_all_clean.empty:
        return pd.DataFrame(columns=cols)
    corrected_value_cols = set(OBSERVABLE_TO_QUANT_COLUMN.values())
    corrected_sig_cols = {f"SIG{v}" for v in corrected_value_cols}
    inv_map = {v: k for k, v in OBSERVABLE_TO_QUANT_COLUMN.items()}

    def corrected_central_provenance(obs_key: str) -> str:
        key = str(obs_key)
        if key.startswith("D_SE_GK_"):
            return "per_segment_stokes_einstein_from_segment_eta_and_window_radius_then_flyvbjerg_petersen_superblock_mean"
        if key in {"eta_GK_Pa_s", "eta_GK_Theo_Pa_s"}:
            return "per_segment_green_kubo_scalar_then_flyvbjerg_petersen_superblock_mean"
        return "flyvbjerg_petersen_superblock_mean"

    rows = []
    for col in df_all_clean.columns:
        if col in QUANT_DATA_METADATA_COLUMNS:
            rows.append({
                "quant_data_column": col,
                "column_kind": "metadata",
                "provenance": "metadata",
                "central_value_provenance": "identifier",
                "uncertainty_provenance": "identifier",
                "uncertainty_scope": "n/a",
                "observable_key": "",
            })
            continue
        if col in corrected_value_cols:
            obs_key = inv_map.get(col, "")
            rows.append({
                "quant_data_column": col,
                "column_kind": "value",
                "provenance": "correlation_corrected",
                "central_value_provenance": corrected_central_provenance(obs_key),
                "uncertainty_provenance": "all_offset_nonoverlapping_batch_means_median_sem",
                "uncertainty_scope": "within_run_time_correlation_corrected",
                "observable_key": obs_key,
            })
            continue
        if col in corrected_sig_cols:
            base = col[3:]
            obs_key = inv_map.get(base, "")
            rows.append({
                "quant_data_column": col,
                "column_kind": "sigma",
                "provenance": "correlation_corrected",
                "central_value_provenance": corrected_central_provenance(obs_key),
                "uncertainty_provenance": "all_offset_nonoverlapping_batch_means_median_sem",
                "uncertainty_scope": "within_run_time_correlation_corrected",
                "observable_key": obs_key,
            })
            continue
        # Unmapped observable: inherited from the single-run legacy path.
        # Central value is the single-run aggregate (mean over chains or
        # fit parameter from chain-averaged data); uncertainty is the
        # corresponding within-run spread or fit covariance. Neither is
        # Flyvbjerg-Petersen corrected for time correlation.
        if col.startswith("SIG"):
            rows.append({
                "quant_data_column": col,
                "column_kind": "sigma",
                "provenance": "inherited_single_run",
                "central_value_provenance": "single_run_legacy_aggregate",
                "uncertainty_provenance": "single_run_within_chain_or_fit_covariance",
                "uncertainty_scope": "within_run_uncorrected_for_time_correlation",
                "observable_key": "",
            })
        else:
            rows.append({
                "quant_data_column": col,
                "column_kind": "value",
                "provenance": "inherited_single_run",
                "central_value_provenance": "single_run_legacy_aggregate",
                "uncertainty_provenance": "single_run_within_chain_or_fit_covariance",
                "uncertainty_scope": "within_run_uncorrected_for_time_correlation",
                "observable_key": "",
            })
    return pd.DataFrame(rows, columns=cols)


def build_quant_data_ci95_sidecar(correlation_summary_df, summary_comparison_df=None):
    """Build CI95 sidecar with t-distribution corrected confidence intervals.

    The CI uses the Student t-distribution with (n_blocks - 1) degrees of
    freedom, which is the correct distribution for small sample sizes
    (Flyvbjerg & Petersen, JCP 1989).  For n_blocks=4, the 95% CI multiplier
    is t_3(0.975)=3.18 vs. z(0.975)=1.96, a ~60% widening.

    Also enriches with tau_int and equilibration diagnostics from the
    SUMMARY_COMPARISON output of BLOCK_CORRELATION_DIAGNOSTICS.
    """
    cols = [
        "temperature_K",
        "category",
        "Small Molecule Name",
        "Small Molecule ID",
        "sampling_unit_family",
        "observable_key",
        "observable_family",
        "quant_data_column",
        "quant_data_sig_column",
        "corrected_mean",
        "corrected_sem",
        "ci95_t_multiplier",
        "ci95_low",
        "ci95_high",
        "n_corrected_blocks",
        "corrected_block_size_units",
        "corrected_block_size_ns",
        "corrected_method",
        "correlation_flag",
        "tau_int_ns",
        "tau_int_windows",
        "g_statistical_inefficiency",
        "t_equil_ns",
        "equil_fraction_discarded",
        "central_value_provenance",
        "uncertainty_provenance",
        "uncertainty_scope",
        "n_replicates",
    ]
    if correlation_summary_df is None or correlation_summary_df.empty:
        return pd.DataFrame(columns=cols)
    df = default_provenance_fields(annotate_quant_columns(correlation_summary_df))
    df = df.rename(columns={"system_name": "Small Molecule Name", "small_molecule_id": "Small Molecule ID"})
    multipliers = np.asarray([ci95_multiplier(x) for x in df.get("n_corrected_blocks", pd.Series(dtype=float))], dtype=float)
    means = pd.to_numeric(df.get("corrected_mean", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
    sems = pd.to_numeric(df.get("corrected_sem", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
    df["ci95_t_multiplier"] = multipliers
    df["ci95_low"] = means - multipliers * sems
    df["ci95_high"] = means + multipliers * sems

    # Enrich with tau_int and equilibration diagnostics from SUMMARY_COMPARISON
    if summary_comparison_df is not None and not summary_comparison_df.empty:
        sc = summary_comparison_df.rename(columns={"system_name": "Small Molecule Name", "small_molecule_id": "Small Molecule ID"})
        merge_keys = ["Small Molecule Name", "observable_key", "sampling_unit_family"]
        available_keys = [k for k in merge_keys if k in sc.columns and k in df.columns]
        if available_keys:
            enrich_cols = ["tau_int_ns", "tau_int_windows", "g", "t_equil_ns", "equil_fraction_discarded"]
            enrich_available = [c for c in enrich_cols if c in sc.columns]
            if enrich_available:
                merge_df = sc[available_keys + enrich_available].drop_duplicates(subset=available_keys, keep="last")
                df = df.merge(merge_df, on=available_keys, how="left", suffixes=("", "_diag"))
                if "g" in df.columns:
                    df = df.rename(columns={"g": "g_statistical_inefficiency"})

    for col in cols:
        if col not in df.columns:
            df[col] = math.nan
    return df[cols]


def prepare_reportable_diagnostics_df(df: pd.DataFrame) -> pd.DataFrame:
    """Annotate a diagnostics frame and rename id columns for the SUMMARY tables."""
    out = df.copy()
    if "observable_key" in out.columns:
        out = default_provenance_fields(annotate_quant_columns(out))
    if "system_name" in out.columns:
        out = out.rename(columns={"system_name": "Small Molecule Name"})
    if "small_molecule_id" in out.columns:
        out = out.rename(columns={"small_molecule_id": "Small Molecule ID"})
    return out


def write_reportable_diagnostics_tables(path: str, correlation_summary_df=None) -> None:
    """Copy the diagnostics CSVs (annotated) into ``RESULTS/SUMMARY/`` with report names.

    Also writes the observable provenance table.
    """
    diagnostics_map = [
        ("CORRELATION_INFERENCE_DIAGNOSTICS.csv", "Correlation_Inference_Diagnostics.csv"),
        ("EQUILIBRATION_AUDIT.csv", "Equilibration_Audit.csv"),
        ("EQUILIBRATION_SYSTEM_AUDIT.csv", "Equilibration_System_Audit.csv"),
        ("EQUILIBRATION_SENSITIVITY.csv", "Equilibration_Sensitivity.csv"),
        ("VISCOSITY_DIAGNOSTICS.csv", "Viscosity_Diagnostics.csv"),
        ("VISCOSITY_SEGMENT_DIAGNOSTICS.csv", "Viscosity_Segment_Diagnostics.csv"),
        ("VISCOSITY_RUNNING_INTEGRAL_SUMMARY.csv", "Viscosity_Running_Integral_Summary.csv"),
        ("STATISTICAL_POLICY.csv", "Statistical_Policy.csv"),
    ]
    for src_name, dst_name in diagnostics_map:
        df = load_optional_diagnostics_csv(path, src_name)
        if df is None:
            continue
        prepare_reportable_diagnostics_df(df).to_csv(
            os.path.join(path, "RESULTS", "SUMMARY", dst_name),
            index=False,
        )
    build_observable_provenance_table(correlation_summary_df).to_csv(
        os.path.join(path, "RESULTS", "SUMMARY", "Observable_Provenance.csv"),
        index=False,
    )


def build_sg_window_map(analysis_root: str, temperature: int, category: str, system_name: str, windows):
    """Return ``{window: {R_cond_A, W_interface_A}}`` by re-extracting SG window scalars.

    Used to supply the condensate radius/interface width that set the Green-Kubo
    volume in the on-the-fly viscosity fallback.
    """
    rows = extract_window_scalars_for_system(analysis_root, temperature, category, system_name, windows)
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    out = {}
    for window_start in windows:
        r_sub = df[(df["window_start_ns"] == window_start) & (df["observable_key"] == "R_cond_A")]
        w_sub = df[(df["window_start_ns"] == window_start) & (df["observable_key"] == "W_interface_A")]
        out[int(window_start)] = {
            "R_cond_A": float(pd.to_numeric(r_sub["value"], errors="coerce").iloc[0]) if not r_sub.empty else math.nan,
            "W_interface_A": float(pd.to_numeric(w_sub["value"], errors="coerce").iloc[0]) if not w_sub.empty else math.nan,
        }
    return out


def build_precomputed_viscosity(analysis_root: str, temperature: int, category: str, system_name: str, blocks, dt_ns: int, tmin_ns: int, tmax_ns: int, visc_cfg, correlation_summary_df=None, viscosity_scalars_df=None, viscosity_layout=None, window_scalars_df=None):
    """Return ``(eta, eta_sem, theo, theo_sem)`` precomputed viscosity for a system.

    Prefers the correlation-corrected summary; otherwise block-averages the
    diagnostic per-segment viscosity scalars (loaded or recomputed on the fly).
    Returns None if no finite viscosity can be obtained. This is passed into
    ``system_analysis.sg_sm_analysis_full`` so it does not recompute viscosity.
    """
    summary_rows = lookup_corrected_summary_rows(
        correlation_summary_df,
        category,
        system_name,
        ["eta_GK_Pa_s", "eta_GK_Theo_Pa_s"],
        VISCOSITY_SAMPLING_FAMILY,
    )
    eta_mean, eta_sem = corrected_summary_scalar(summary_rows, "eta_GK_Pa_s")
    theo_mean, theo_sem = corrected_summary_scalar(summary_rows, "eta_GK_Theo_Pa_s")
    if any(np.isfinite(x) for x in [eta_mean, theo_mean]):
        return eta_mean, eta_sem, theo_mean, theo_sem

    df = None
    segment_blocks = None
    if viscosity_layout:
        segment_blocks = viscosity_layout.get(category, {}).get(system_name)
    if viscosity_scalars_df is not None:
        df = viscosity_scalars_df[(viscosity_scalars_df["category"] == category) & (viscosity_scalars_df["system_name"] == system_name)].copy()
        if df[df["observable_key"].isin(["eta_GK_Pa_s", "eta_GK_Theo_Pa_s"])].empty:
            df = None
    if df is None:
        windows = flatten_windows(blocks)
        if not windows:
            return None
        sg_window_map = build_sg_window_map(analysis_root, temperature, category, system_name, windows)
        rows = extract_viscosity_scalars_for_system(
            analysis_root,
            temperature,
            category,
            system_name,
            windows,
            dt_ns,
            tmin_ns,
            tmax_ns,
            sg_window_map,
            visc_cfg,
        )
        if not rows:
            return None
        df = pd.DataFrame(rows)
    if not segment_blocks:
        segment_members = sorted({int(v) for v in pd.to_numeric(df["window_start_ns"], errors="coerce").dropna().tolist()})
        segment_blocks = [[member] for member in segment_members]
    eta_mean, eta_sem, _ = block_summary(block_means_from_rows(df, segment_blocks, "eta_GK_Pa_s"))
    theo_mean, theo_sem, _ = block_summary(block_means_from_rows(df, segment_blocks, "eta_GK_Theo_Pa_s"))
    if not any(np.isfinite(x) for x in [eta_mean, theo_mean]):
        return None
    return eta_mean, eta_sem, theo_mean, theo_sem


def summarize_and_write_master_tables(path, df_sg_clean, df_dsm_clean, df_ndsm_clean,
                                      df_res_contact_master, df_acid_contact_master,
                                      df_res_count_master, df_acid_count_master,
                                      correlation_summary_df=None, correlation_long_df=None,
                                      summary_comparison_df=None,
                                      dsm_names=None, ndsm_names=None):
    """Write all master SM-contact and Quant_Data tables (with sidecars).

    Adds DSM/NDSM class-average columns to the contact masters, writes the SM
    contact mean/SEM CSVs, concatenates the per-category Quant_Data into
    ``Quant_Data.csv``, and emits the LONG, CI95, provenance and reportable
    diagnostics sidecars.
    """
    d_cols_res = [c for c in df_res_contact_master.columns if c.startswith("D")]
    nd_cols_res = [c for c in df_res_contact_master.columns if c.startswith("ND")]
    if d_cols_res:
        df_res_contact_master["DSM_AVE"] = df_res_contact_master[d_cols_res].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    if nd_cols_res:
        df_res_contact_master["NDSM_AVE"] = df_res_contact_master[nd_cols_res].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    d_cols_acid = [c for c in df_acid_contact_master.columns if c.startswith("D")]
    nd_cols_acid = [c for c in df_acid_contact_master.columns if c.startswith("ND")]
    if d_cols_acid:
        df_acid_contact_master["DSM_AVE"] = df_acid_contact_master[d_cols_acid].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    if nd_cols_acid:
        df_acid_contact_master["NDSM_AVE"] = df_acid_contact_master[nd_cols_acid].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    d_cols_rescnt = [c for c in df_res_count_master.columns if c.startswith("D")]
    nd_cols_rescnt = [c for c in df_res_count_master.columns if c.startswith("ND")]
    if d_cols_rescnt:
        df_res_count_master["DSM_AVE"] = df_res_count_master[d_cols_rescnt].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    if nd_cols_rescnt:
        df_res_count_master["NDSM_AVE"] = df_res_count_master[nd_cols_rescnt].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    d_cols_acidcnt = [c for c in df_acid_count_master.columns if c.startswith("D")]
    nd_cols_acidcnt = [c for c in df_acid_count_master.columns if c.startswith("ND")]
    if d_cols_acidcnt:
        df_acid_count_master["DSM_AVE"] = df_acid_count_master[d_cols_acidcnt].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    if nd_cols_acidcnt:
        df_acid_count_master["NDSM_AVE"] = df_acid_count_master[nd_cols_acidcnt].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    df_res_contact_master.to_csv(f"{path}/RESULTS/SM_CONTACT_MAPS/SM_ResMap_Data.csv", index=False)
    df_acid_contact_master.to_csv(f"{path}/RESULTS/SM_CONTACT_MAPS/SM_AcidMap_Data.csv", index=False)
    df_res_count_master.to_csv(f"{path}/RESULTS/SM_CONTACT_MAPS/SM_ResCount_Data.csv", index=False)
    df_acid_count_master.to_csv(f"{path}/RESULTS/SM_CONTACT_MAPS/SM_AcidCount_Data.csv", index=False)

    def _write_sm_contact_sem_master(master_df, filename, sem_prefix, d_names, nd_names):
        if master_df.empty:
            return
        out = master_df.iloc[:, :1].copy()
        d_cols = [c for c in master_df.columns if c.startswith("D") and not c.startswith("DSM")]
        nd_cols = [c for c in master_df.columns if c.startswith("ND") and not c.startswith("NDSM")]
        for col, system_name in zip(d_cols, d_names):
            mat = _read_contact_matrix(f"{path}/RESULTS/SM_CONTACT_MAPS/{sem_prefix}_{system_name}.csv")
            if mat is not None:
                out[col] = np.asarray(mat, dtype=float).reshape(-1)
        for col, system_name in zip(nd_cols, nd_names):
            mat = _read_contact_matrix(f"{path}/RESULTS/SM_CONTACT_MAPS/{sem_prefix}_{system_name}.csv")
            if mat is not None:
                out[col] = np.asarray(mat, dtype=float).reshape(-1)
        if d_cols:
            out["DSM_AVE"] = master_df[d_cols].apply(pd.to_numeric, errors="coerce").sem(axis=1)
        if nd_cols:
            out["NDSM_AVE"] = master_df[nd_cols].apply(pd.to_numeric, errors="coerce").sem(axis=1)
        out.to_csv(f"{path}/RESULTS/SM_CONTACT_MAPS/{filename}", index=False)

    _write_sm_contact_sem_master(
        df_res_contact_master,
        "SM_ResMap_SEM_Data.csv",
        "SM_Residue_Contacts_SEM",
        dsm_names or list(DEFAULT_DSM_LIST),
        ndsm_names or list(DEFAULT_NDSM_LIST),
    )
    _write_sm_contact_sem_master(
        df_acid_contact_master,
        "SM_AcidMap_SEM_Data.csv",
        "SM_Acid_Contacts_SEM",
        dsm_names or list(DEFAULT_DSM_LIST),
        ndsm_names or list(DEFAULT_NDSM_LIST),
    )

    df_all = pd.concat([df_sg_clean, df_dsm_clean, df_ndsm_clean], ignore_index=True)
    df_all_clean = df_all.drop_duplicates(subset=["Small Molecule ID"]) if not df_all.empty else df_all

    df_dsm_clean.to_csv(f"{path}/RESULTS/SUMMARY/DSM_Quant_Data.csv", index=False)
    df_ndsm_clean.to_csv(f"{path}/RESULTS/SUMMARY/NDSM_Quant_Data.csv", index=False)
    df_all_clean.to_csv(f"{path}/RESULTS/SUMMARY/Quant_Data.csv", index=False)
    build_quant_data_long_sidecar(correlation_long_df).to_csv(f"{path}/RESULTS/SUMMARY/Quant_Data_LONG.csv", index=False)
    build_quant_data_ci95_sidecar(correlation_summary_df, summary_comparison_df).to_csv(f"{path}/RESULTS/SUMMARY/Quant_Data_CI95.csv", index=False)
    build_quant_data_provenance_sidecar(df_all_clean).to_csv(f"{path}/RESULTS/SUMMARY/Quant_Data_PROVENANCE.csv", index=False)
    write_reportable_diagnostics_tables(path, correlation_summary_df)


def write_combined_standardized_contact_plots(path: str, dsm_names, ndsm_names) -> None:
    """Write the combined DSM-minus-NDSM standardized contact-map CSVs (data-only).

    Computes the (DSM - NDSM)/SG standardized residue, acid and SM contact
    difference maps (with propagated SEM maps) and saves their reordered CSV
    outputs; no-op if either category list is empty.
    """
    if not dsm_names or not ndsm_names:
        print("[INFO] Skipping combined DSM/NDSM standardized contact data: missing one or both categories")
        return

    def _mean_standardized_contacts(tag_list, label, subdir, prefix):
        arrays = []
        for tag in tag_list:
            fn = f"{path}/RESULTS/{subdir}/{prefix}_{tag}.csv"
            if os.path.isfile(fn):
                arrays.append(np.array(pd.read_csv(fn, header=None), dtype=float))
            else:
                print(f"[WARN] Missing standardized contacts for {label} system '{tag}': {fn}")
        if not arrays:
            raise FileNotFoundError(f"No {prefix}_* files found for {label} systems in {path}/RESULTS/{subdir}")
        return np.nanmean(np.stack(arrays, axis=0), axis=0)

    def _sem_standardized_contacts(tag_list, subdir, prefix):
        arrays = []
        for tag in tag_list:
            fn = f"{path}/RESULTS/{subdir}/{prefix}_{tag}.csv"
            if os.path.isfile(fn):
                arrays.append(np.array(pd.read_csv(fn, header=None), dtype=float))
        return _contact_matrix_sem(arrays) if arrays else None

    try:
        ndsm_res_arr = _mean_standardized_contacts(ndsm_names, "NDSM", "RESIDUE_CONTACT_MAPS", "Residue_Contacts_Standardized_Mean")
        dsm_res_arr = _mean_standardized_contacts(dsm_names, "DSM", "RESIDUE_CONTACT_MAPS", "Residue_Contacts_Standardized_Mean")
        ndsm_res_sem = _sem_standardized_contacts(ndsm_names, "RESIDUE_CONTACT_MAPS", "Residue_Contacts_Standardized_Mean")
        dsm_res_sem = _sem_standardized_contacts(dsm_names, "RESIDUE_CONTACT_MAPS", "Residue_Contacts_Standardized_Mean")
        sg_res_arr = np.array(pd.read_csv(f"{path}/RESULTS/RESIDUE_CONTACT_MAPS/Residue_Contacts_Standardized_Mean_sg_X.csv", header=None), dtype=float)
        sg_res_sem = _read_contact_matrix(f"{path}/RESULTS/RESIDUE_CONTACT_MAPS/Residue_Contacts_Standardized_SEM_sg_X.csv")
    except Exception as exc:
        print(f"[INFO] Skipping combined residue standardized data: {exc}")
        return

    res_contact_delta = np.subtract(dsm_res_arr, ndsm_res_arr)
    res_contact_array = np.divide(res_contact_delta, sg_res_arr, out=np.zeros_like(res_contact_delta, dtype=float), where=sg_res_arr != 0)

    res_contact_array = _prepare_residue_square(res_contact_array)
    np.savetxt(f"{path}/RESULTS/RESIDUE_CONTACT_MAPS/Residue_DSM_NDSM_Difference_Standardized_HeatMap.csv", res_contact_array, delimiter=",")
    res_delta_sem = np.sqrt(dsm_res_sem ** 2 + ndsm_res_sem ** 2) if dsm_res_sem is not None and ndsm_res_sem is not None else None
    res_contact_sem = _ratio_difference_sem(res_contact_delta, res_delta_sem, sg_res_arr, sg_res_sem)
    if res_contact_sem is not None:
        np.savetxt(f"{path}/RESULTS/RESIDUE_CONTACT_MAPS/Residue_DSM_NDSM_Difference_Standardized_SEM_HeatMap.csv", res_contact_sem, delimiter=",")
        res_sem_1d = _prepare_residue_rows(_contact_1d_sem(res_contact_sem))
        np.savetxt(f"{path}/RESULTS/RESIDUE_CONTACT_MAPS/Residue_DSM_NDSM_Difference_Standardized_SEM_1D_HeatMap.csv", res_sem_1d, delimiter=",")

    try:
        ndsm_acid_arr = _mean_standardized_contacts(ndsm_names, "NDSM", "ACID_CONTACT_MAPS", "Acid_Contacts_Standardized_Mean")
        dsm_acid_arr = _mean_standardized_contacts(dsm_names, "DSM", "ACID_CONTACT_MAPS", "Acid_Contacts_Standardized_Mean")
        ndsm_acid_sem = _sem_standardized_contacts(ndsm_names, "ACID_CONTACT_MAPS", "Acid_Contacts_Standardized_Mean")
        dsm_acid_sem = _sem_standardized_contacts(dsm_names, "ACID_CONTACT_MAPS", "Acid_Contacts_Standardized_Mean")
        sg_acid_arr = np.array(pd.read_csv(f"{path}/RESULTS/ACID_CONTACT_MAPS/Acid_Contacts_Standardized_Mean_sg_X.csv", header=None), dtype=float)
        sg_acid_sem = _read_contact_matrix(f"{path}/RESULTS/ACID_CONTACT_MAPS/Acid_Contacts_Standardized_SEM_sg_X.csv")
    except Exception as exc:
        print(f"[INFO] Skipping combined acid standardized data: {exc}")
        return

    acid_contact_delta = np.subtract(dsm_acid_arr, ndsm_acid_arr)
    acid_contact_array = np.divide(acid_contact_delta, sg_acid_arr, out=np.zeros_like(acid_contact_delta, dtype=float), where=sg_acid_arr != 0)
    acid_contact_array = _prepare_acid_square(acid_contact_array)
    np.savetxt(f"{path}/RESULTS/ACID_CONTACT_MAPS/Acid_DSM_NDSM_Difference_Standardized_HeatMap.csv", acid_contact_array, delimiter=",")
    acid_delta_sem = np.sqrt(dsm_acid_sem ** 2 + ndsm_acid_sem ** 2) if dsm_acid_sem is not None and ndsm_acid_sem is not None else None
    acid_contact_sem = _ratio_difference_sem(acid_contact_delta, acid_delta_sem, sg_acid_arr, sg_acid_sem)
    if acid_contact_sem is not None:
        np.savetxt(f"{path}/RESULTS/ACID_CONTACT_MAPS/Acid_DSM_NDSM_Difference_Standardized_SEM_HeatMap.csv", acid_contact_sem, delimiter=",")
        acid_sem_1d = _prepare_acid_rows(_contact_1d_sem(acid_contact_sem))
        np.savetxt(f"{path}/RESULTS/ACID_CONTACT_MAPS/Acid_DSM_NDSM_Difference_Standardized_SEM_1D_HeatMap.csv", acid_sem_1d, delimiter=",")

    try:
        df_res_contact = pd.read_csv(f"{path}/RESULTS/SM_CONTACT_MAPS/SM_ResMap_Data.csv")
        df_acid_contact = pd.read_csv(f"{path}/RESULTS/SM_CONTACT_MAPS/SM_AcidMap_Data.csv")
    except Exception as exc:
        print(f"[INFO] Skipping combined SM contact data: {exc}")
        return

    df_res_contact_sem = None
    res_sem_path = f"{path}/RESULTS/SM_CONTACT_MAPS/SM_ResMap_SEM_Data.csv"
    if os.path.isfile(res_sem_path):
        df_res_contact_sem = pd.read_csv(res_sem_path)
        sm_res_sem_array = df_res_contact_sem.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        np.savetxt(f"{path}/RESULTS/SM_CONTACT_MAPS/SM_Residue_DSM_NDSM_SUMMARY_Standardized_SEM_HeatMap.csv", sm_res_sem_array, delimiter=",")

    df_acid_contact.to_csv(f"{path}/RESULTS/SM_CONTACT_MAPS/DF_Acid_DSM_NDSM_Difference_Standardized_HeatMap.csv", index=False)
    sm_acid_contact_array = np.array(df_acid_contact.iloc[:, 1:])
    np.savetxt(f"{path}/RESULTS/SM_CONTACT_MAPS/SM_Acid_DSM_NDSM_Difference_Standardized_HeatMap.csv", sm_acid_contact_array, delimiter=",")

    df_acid_contact_sem = None
    acid_sem_path = f"{path}/RESULTS/SM_CONTACT_MAPS/SM_AcidMap_SEM_Data.csv"
    if os.path.isfile(acid_sem_path):
        df_acid_contact_sem = pd.read_csv(acid_sem_path)
        sm_acid_sem_array = df_acid_contact_sem.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        np.savetxt(f"{path}/RESULTS/SM_CONTACT_MAPS/SM_Acid_DSM_NDSM_SUMMARY_Standardized_SEM_HeatMap.csv", sm_acid_sem_array, delimiter=",")

    if {"DSM", "NDSM"}.issubset(df_res_contact.columns):
        b = df_res_contact["DSM"] - df_res_contact["NDSM"]
    elif {"DSM_AVE", "NDSM_AVE"}.issubset(df_res_contact.columns):
        b = df_res_contact["DSM_AVE"] - df_res_contact["NDSM_AVE"]
    else:
        d_cols_res = [c for c in df_res_contact.columns if c.startswith("D")]
        nd_cols_res = [c for c in df_res_contact.columns if c.startswith("ND")]
        b = df_res_contact[d_cols_res].apply(pd.to_numeric, errors="coerce").mean(axis=1) - df_res_contact[nd_cols_res].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    sm_res_contact_array = _prepare_residue_rows(np.transpose(np.asarray([b.to_numpy(dtype=float)])))
    np.savetxt(f"{path}/RESULTS/SM_CONTACT_MAPS/SM_Residue_DSM_NDSM_SUMMARY_Difference_Standardized_HeatMap.csv", sm_res_contact_array, delimiter=",")
    if df_res_contact_sem is not None and {"DSM_AVE", "NDSM_AVE"}.issubset(df_res_contact_sem.columns):
        res_diff_sem = np.sqrt(
            pd.to_numeric(df_res_contact_sem["DSM_AVE"], errors="coerce").to_numpy(dtype=float) ** 2
            + pd.to_numeric(df_res_contact_sem["NDSM_AVE"], errors="coerce").to_numpy(dtype=float) ** 2
        ).reshape(-1, 1)
        np.savetxt(f"{path}/RESULTS/SM_CONTACT_MAPS/SM_Residue_DSM_NDSM_SUMMARY_Difference_Standardized_SEM_HeatMap.csv", res_diff_sem, delimiter=",")

    if {"DSM", "NDSM"}.issubset(df_acid_contact.columns):
        b = df_acid_contact["DSM"] - df_acid_contact["NDSM"]
    elif {"DSM_AVE", "NDSM_AVE"}.issubset(df_acid_contact.columns):
        b = df_acid_contact["DSM_AVE"] - df_acid_contact["NDSM_AVE"]
    else:
        d_cols_acid = [c for c in df_acid_contact.columns if c.startswith("D")]
        nd_cols_acid = [c for c in df_acid_contact.columns if c.startswith("ND")]
        b = df_acid_contact[d_cols_acid].apply(pd.to_numeric, errors="coerce").mean(axis=1) - df_acid_contact[nd_cols_acid].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    sm_acid_contact_array = np.transpose(np.asarray([b.to_numpy(dtype=float)]))
    np.savetxt(f"{path}/RESULTS/SM_CONTACT_MAPS/SM_Acid_DSM_NDSM_SUMMARY_Difference_Standardized_HeatMap.csv", sm_acid_contact_array, delimiter=",")
    if df_acid_contact_sem is not None and {"DSM_AVE", "NDSM_AVE"}.issubset(df_acid_contact_sem.columns):
        acid_diff_sem = np.sqrt(
            pd.to_numeric(df_acid_contact_sem["DSM_AVE"], errors="coerce").to_numpy(dtype=float) ** 2
            + pd.to_numeric(df_acid_contact_sem["NDSM_AVE"], errors="coerce").to_numpy(dtype=float) ** 2
        ).reshape(-1, 1)
        np.savetxt(f"{path}/RESULTS/SM_CONTACT_MAPS/SM_Acid_DSM_NDSM_SUMMARY_Difference_Standardized_SEM_HeatMap.csv", acid_diff_sem, delimiter=",")
    sm_acid_contact_array = _prepare_acid_rows(sm_acid_contact_array)
    np.savetxt(f"{path}/RESULTS/SM_CONTACT_MAPS/SM_Acid_DSM_NDSM_SUMMARY_Difference_Standardized_HeatMap_2.csv", sm_acid_contact_array, delimiter=",")


def main() -> None:
    """Run the correlated system-level analysis over a blocked analysis root.

    Builds the ``system_analysis`` engine, loads the diagnostics tables, processes SG /
    DSM / NDSM (and their class averages) via ``sg_sm_analysis_full`` with
    precomputed viscosity, overwrites each Quant_Data row with
    correlation-corrected values, and writes all master tables and combined
    contact CSVs. With ``--plot-only`` it only regenerates the combined
    standardized contact CSVs from existing RESULTS.
    """
    parser = argparse.ArgumentParser(description="Perform correlated system-level analysis on a blocked analysis root")
    parser.add_argument("--analysis-root", default=None, help="Explicit blocked analysis root")
    parser.add_argument("--path", default=None, help="Path to TEMP_XXX directory")
    parser.add_argument("--folder", default="CLASSIFY_BLOCKED", help="Output folder prefix used to derive the blocked analysis root")
    parser.add_argument("--layout-csv", default=None, help="Path to RECOMMENDED_SYSTEM_BLOCK_LAYOUT.csv")
    parser.add_argument("--T", type=int, required=True, help="Simulation temperature in Kelvin")
    parser.add_argument("--dt", type=int, required=True, help="Cluster time stride (ns)")
    parser.add_argument("--tmin", type=int, required=True, help="Start of analysis window (ns)")
    parser.add_argument("--tmax", type=int, required=True, help="End of analysis window (ns)")
    parser.add_argument("--c", type=float, default=1.0, help="Small molecule concentration label (default: 1.0)")
    parser.add_argument("--plot-only", action="store_true", help="Regenerate combined standardized contact CSVs from existing RESULTS without recomputing correlated tables")
    parser.add_argument("--species", nargs="+", default=None, help="Subset of species for diffusion (default: all)")
    parser.add_argument("--species-all", nargs="+", default=None, help="Subset of species for global ALL diffusion averages (default: all analyzed species)")
    parser.add_argument("--diff-downsample", type=int, default=None)
    parser.add_argument("--diff-dt-diff", type=float, default=None, help="Optional MSD coarse-graining width in ns before fitting (default: none; use native lag grid)")
    parser.add_argument("--diff-segments", type=int, default=None, help="Number of contiguous diffusion segments for RCC window bootstrap summaries (default: main-pipeline default)")
    parser.add_argument("--diff-n-boot", type=int, default=None, help="Number of diffusion bootstrap resamples over segment means (default: 10)")
    parser.add_argument("--diff-slope-iters", type=int, default=None)
    parser.add_argument("--diff-dt-pts", type=int, default=None)
    parser.add_argument("--diff-slope-tol", type=float, default=None)
    parser.add_argument("--diff-min-diff-pts", type=int, default=None)
    parser.add_argument("--diff-slope-plateau", type=float, default=None)
    parser.add_argument("--diff-min-plateau-pts", type=int, default=None)
    parser.add_argument("--diff-boot-r2", type=float, default=None)
    parser.add_argument("--diff-smooth-window", type=int, default=None)
    parser.add_argument("--diff-smooth-polyorder", type=int, default=None)
    parser.add_argument("--diff-seed", type=int, default=None)
    parser.add_argument("--diff-min-segment-ns", type=float, default=None)
    parser.add_argument("--diff-min-primary-fraction", type=float, default=None)
    parser.add_argument("--diff-origin-resample", type=int, choices=[0, 1], default=None)
    parser.add_argument("--diff-origin-window-ns", type=float, default=None)
    parser.add_argument("--diff-origin-stride-ns", type=float, default=None)
    parser.add_argument("--diff-origin-candidate-windows-ns", type=str, default=None)
    parser.add_argument("--diff-origin-min-success-fraction", type=float, default=None)
    parser.add_argument("--diff-origin-min-success-count", type=int, default=None)
    parser.add_argument("--diff-origin-max-origins", type=int, default=None)
    parser.add_argument("--visc-segments", type=int, default=None)
    parser.add_argument("--visc-iterations", type=int, default=None)
    parser.add_argument("--visc-n-boot", type=int, default=None)
    parser.add_argument("--visc-dt-unit", type=float, default=None)
    parser.add_argument("--visc-n-point", type=int, default=None)
    parser.add_argument("--visc-n-tau", type=int, default=None)
    parser.add_argument("--visc-seed", type=int, default=None)
    args = parser.parse_args()

    analysis_root = resolve_analysis_root(args)
    if not os.path.isdir(analysis_root):
        raise SystemExit(f"Analysis root not found: {analysis_root}")

    if args.plot_only:
        ensure_output_scaffold(analysis_root)
        write_combined_standardized_contact_plots(analysis_root, list(DEFAULT_DSM_LIST), list(DEFAULT_NDSM_LIST))
        return

    for cat in ["SG", "DSM", "NDSM"]:
        input_dir = os.path.join(analysis_root, f"ANALYSIS_{cat}")
        if not os.path.exists(input_dir):
            print(f"Warning: {input_dir} not found - {cat} analysis will be skipped")

    ensure_output_scaffold(analysis_root)
    analysis = build_analysis(args, analysis_root)
    layout = load_layout(resolve_layout_csv(args, analysis_root))
    window_scalars_df = load_window_scalars_df(analysis_root)
    viscosity_scalars_df = load_viscosity_scalars_df(analysis_root)
    correlation_summary_df = load_correlation_summary_df(analysis_root)
    correlation_long_df = load_correlation_long_df(analysis_root)
    summary_comparison_df = load_summary_comparison_df(analysis_root)
    system_blocks_df = load_system_blocks_df(analysis_root)
    viscosity_layout = load_optional_layout(
        os.path.join(analysis_root, "RESULTS", "CORRELATION_DIAGNOSTICS", "RECOMMENDED_VISCOSITY_SEGMENT_LAYOUT.csv")
    )

    c = args.c
    T = args.T
    dt = args.dt
    tmin = args.tmin
    tmax = args.tmax
    mol_name = "X"

    # The ground-truth DSM/NDSM partition is fixed by the simulation design.
    # Do NOT read dsm_list.txt / ndsm_list.txt here: those files are written by
    # kmeans.py as *predicted* cluster members, so reading them creates a
    # circular dependency that progressively prunes the analyzed systems.
    dsm_names = list(DEFAULT_DSM_LIST)
    ndsm_names = list(DEFAULT_NDSM_LIST)

    df_sg, df_res_contact_sg, df_acid_contact_sg, df_res_count_sg, df_acid_count_sg = analysis.define_pd()
    df_dsm, _, _, _, _ = analysis.define_pd()
    df_ndsm, _, _, _, _ = analysis.define_pd()
    _, df_res_contact_master, df_acid_contact_master, df_res_count_master, df_acid_count_master = analysis.define_pd()

    df_sg_clean = pd.DataFrame(columns=df_sg.columns)
    df_dsm_clean = pd.DataFrame(columns=df_dsm.columns)
    df_ndsm_clean = pd.DataFrame(columns=df_ndsm.columns)

    try:
        print("\n========== PROCESSING SG ==========")
        folder = "ANALYSIS_SG_AVE"
        name = "sg_X"
        blocks = layout.get("SG", {}).get(name)
        print(f"[SG] Loaded {len(blocks) if blocks else 0} coarse blocks")
        precomputed_viscosity = build_precomputed_viscosity(
            analysis_root,
            T,
            "SG",
            name,
            blocks,
            dt,
            tmin,
            tmax,
            analysis.viscosity_params,
            correlation_summary_df=correlation_summary_df,
            viscosity_scalars_df=viscosity_scalars_df,
            viscosity_layout=viscosity_layout,
            window_scalars_df=window_scalars_df,
        )
        print(f"[SG] Precomputed viscosity ready: {precomputed_viscosity is not None}")
        print("[SG] Entering sg_sm_analysis_full")
        dfs = analysis.sg_sm_analysis_full(
            analysis_root,
            folder,
            name,
            df_sg,
            mol_name,
            c,
            df_res_contact_sg,
            df_acid_contact_sg,
            df_res_count_sg,
            df_acid_count_sg,
            T,
            dt,
            tmin,
            tmax,
            diffusion_params=analysis.diffusion_params,
            viscosity_params=analysis.viscosity_params,
            species_to_analyze=args.species if args.species else None,
            precomputed_viscosity=precomputed_viscosity,
        )
        print("[SG] sg_sm_analysis_full returned")
        df_sg = dfs[0]
        df_res_contact_sg = dfs[1]
        df_acid_contact_sg = dfs[2]
        df_res_count_sg = dfs[3]
        df_acid_count_sg = dfs[4]
        df_sg = apply_corrected_summary_to_row(df_sg, name, "SG", name, correlation_summary_df)
        df_sg.to_csv(f"{analysis_root}/RESULTS/SUMMARY/SG_Quant_Data.csv", index=False)
        df_sg_clean = df_sg.drop_duplicates(subset=["Small Molecule ID"])
        print("SG PROCESSING SUCCESSFUL")
    except Exception as exc:
        print(f"SG PROCESSING FAILED: {exc}")

    try:
        if os.path.isdir(os.path.join(analysis_root, "ANALYSIS_DSM")):
            print("\n========== PROCESSING DSM ==========")
            folder = "ANALYSIS_DSM_AVE"
            for name in dsm_names:
                blocks = layout.get("DSM", {}).get(name)
                if not blocks:
                    continue
                print(name)
                precomputed_viscosity = build_precomputed_viscosity(
                    analysis_root,
                    T,
                    "DSM",
                    name,
                    blocks,
                    dt,
                    tmin,
                    tmax,
                    analysis.viscosity_params,
                    correlation_summary_df=correlation_summary_df,
                    viscosity_scalars_df=viscosity_scalars_df,
                    viscosity_layout=viscosity_layout,
                    window_scalars_df=window_scalars_df,
                )
                dfs = analysis.sg_sm_analysis_full(
                    analysis_root,
                    folder,
                    name,
                    df_dsm,
                    mol_name,
                    c,
                    df_res_contact_master,
                    df_acid_contact_master,
                    df_res_count_master,
                    df_acid_count_master,
                    T,
                    dt,
                    tmin,
                    tmax,
                    diffusion_params=analysis.diffusion_params,
                    viscosity_params=analysis.viscosity_params,
                    species_to_analyze=args.species if args.species else None,
                    precomputed_viscosity=precomputed_viscosity,
                )
                df_dsm = dfs[0]
                df_res_contact_master = dfs[1]
                df_acid_contact_master = dfs[2]
                df_res_count_master = dfs[3]
                df_acid_count_master = dfs[4]
                df_dsm = apply_corrected_summary_to_row(df_dsm, name, "DSM", name, correlation_summary_df)

            if not df_dsm.empty:
                df_dsm = df_dsm.sort_values(by="Small Molecule ID", ascending=False)
                df_dsm = append_class_average_row(df_dsm, "DSM_AVG")
                dsm_visc_avg = build_class_average_viscosity(correlation_summary_df, "DSM", dsm_names)

                dfs = analysis.sg_sm_analysis_full(
                    analysis_root,
                    folder,
                    "DSM",
                    df_dsm,
                    "DSM",
                    c,
                    df_res_contact_master,
                    df_acid_contact_master,
                    df_res_count_master,
                    df_acid_count_master,
                    T,
                    dt,
                    tmin,
                    tmax,
                    diffusion_params=analysis.diffusion_params,
                    viscosity_params=analysis.viscosity_params,
                    species_to_analyze=args.species if args.species else None,
                    precomputed_viscosity=dsm_visc_avg,
                )
                df_dsm = dfs[0]
                df_res_contact_master = dfs[1]
                df_acid_contact_master = dfs[2]
                df_res_count_master = dfs[3]
                df_acid_count_master = dfs[4]
                df_dsm = apply_corrected_class_stats_to_row(df_dsm, "DSM", "DSM", dsm_names, correlation_summary_df)
                df_dsm = _fill_aggregate_spatial_from_members(df_dsm, dsm_names, "DSM")
                df_dsm = _fill_aggregate_spatial_from_members(df_dsm, dsm_names, "DSM_AVG")
                df_dsm_clean = df_dsm.drop_duplicates(subset=["Small Molecule ID"])
                print("DSM PROCESSING SUCCESSFUL")
    except Exception as exc:
        print(f"DSM PROCESSING FAILED: {exc}")

    try:
        if os.path.isdir(os.path.join(analysis_root, "ANALYSIS_NDSM")):
            print("\n========== PROCESSING NDSM ==========")
            folder = "ANALYSIS_NDSM_AVE"
            for name in ndsm_names:
                blocks = layout.get("NDSM", {}).get(name)
                if not blocks:
                    continue
                print(name)
                precomputed_viscosity = build_precomputed_viscosity(
                    analysis_root,
                    T,
                    "NDSM",
                    name,
                    blocks,
                    dt,
                    tmin,
                    tmax,
                    analysis.viscosity_params,
                    correlation_summary_df=correlation_summary_df,
                    viscosity_scalars_df=viscosity_scalars_df,
                    viscosity_layout=viscosity_layout,
                    window_scalars_df=window_scalars_df,
                )
                dfs = analysis.sg_sm_analysis_full(
                    analysis_root,
                    folder,
                    name,
                    df_ndsm,
                    mol_name,
                    c,
                    df_res_contact_master,
                    df_acid_contact_master,
                    df_res_count_master,
                    df_acid_count_master,
                    T,
                    dt,
                    tmin,
                    tmax,
                    diffusion_params=analysis.diffusion_params,
                    viscosity_params=analysis.viscosity_params,
                    species_to_analyze=args.species if args.species else None,
                    precomputed_viscosity=precomputed_viscosity,
                )
                df_ndsm = dfs[0]
                df_res_contact_master = dfs[1]
                df_acid_contact_master = dfs[2]
                df_res_count_master = dfs[3]
                df_acid_count_master = dfs[4]
                df_ndsm = apply_corrected_summary_to_row(df_ndsm, name, "NDSM", name, correlation_summary_df)

            if not df_ndsm.empty:
                df_ndsm = df_ndsm.sort_values(by="Small Molecule ID", ascending=False)
                df_ndsm = append_class_average_row(df_ndsm, "NDSM_AVG")
                ndsm_visc_avg = build_class_average_viscosity(correlation_summary_df, "NDSM", ndsm_names)

                dfs = analysis.sg_sm_analysis_full(
                    analysis_root,
                    folder,
                    "NDSM",
                    df_ndsm,
                    "NDSM",
                    c,
                    df_res_contact_master,
                    df_acid_contact_master,
                    df_res_count_master,
                    df_acid_count_master,
                    T,
                    dt,
                    tmin,
                    tmax,
                    diffusion_params=analysis.diffusion_params,
                    viscosity_params=analysis.viscosity_params,
                    species_to_analyze=args.species if args.species else None,
                    precomputed_viscosity=ndsm_visc_avg,
                )
                df_ndsm = dfs[0]
                df_res_contact_master = dfs[1]
                df_acid_contact_master = dfs[2]
                df_res_count_master = dfs[3]
                df_acid_count_master = dfs[4]
                df_ndsm = apply_corrected_class_stats_to_row(df_ndsm, "NDSM", "NDSM", ndsm_names, correlation_summary_df)
                df_ndsm = _fill_aggregate_spatial_from_members(df_ndsm, ndsm_names, "NDSM")
                df_ndsm = _fill_aggregate_spatial_from_members(df_ndsm, ndsm_names, "NDSM_AVG")
                df_ndsm_clean = df_ndsm.drop_duplicates(subset=["Small Molecule ID"])
                print("NDSM PROCESSING SUCCESSFUL")
    except Exception as exc:
        print(f"NDSM PROCESSING FAILED: {exc}")

    try:
        summarize_and_write_master_tables(
            analysis_root,
            df_sg_clean,
            df_dsm_clean,
            df_ndsm_clean,
            df_res_contact_master,
            df_acid_contact_master,
            df_res_count_master,
            df_acid_count_master,
            correlation_summary_df=correlation_summary_df,
            correlation_long_df=correlation_long_df,
            summary_comparison_df=summary_comparison_df,
            dsm_names=dsm_names,
            ndsm_names=ndsm_names,
        )
    except Exception as exc:
        print(f"SM INTERDEPENDENCY BLOCK FAILED: {exc}")

    write_combined_standardized_contact_plots(
        analysis_root,
        dsm_names if not df_dsm_clean.empty else [],
        ndsm_names if not df_ndsm_clean.empty else [],
    )


if __name__ == "__main__":
    main()
