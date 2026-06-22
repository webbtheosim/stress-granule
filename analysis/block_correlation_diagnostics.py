#!/usr/bin/env python3
"""Time-correlation diagnostics and block-size recommendation (correlated pipeline).

Pipeline role
-------------
First stage of the correlated pipeline. For one temperature it re-extracts every
Quant_Data observable as a *per-window* (or per-stress-segment) scalar time
series across the 39 post-equilibration 50 ns windows, then quantifies the
within-run time correlation of each series and recommends a block size that
decorrelates it. Two complementary estimators are used: a Flyvbjerg-Petersen
(JCP 1989) superblock SEM plateau sweep and PYMBAR's statistical inefficiency
``g`` (Chodera JCTC 2016 for equilibration detection). The per-system block size
is the conservative maximum over the core (non-viscosity) observables. The
resulting block layouts drive the downstream blocked averaging
(``AVERAGE_SIMULATIONS_CORRELATED``, ``MAX_CLUSTER_CORRELATED``) and the
correlation-corrected means/SEMs applied in
``SYSTEM_ANALYSIS_FINAL_CORRELATED``.

Per-window scalars are derived by re-fitting RDP density profiles (phase
concentrations, R/W, surface tensions, deltaG), reading cluster scalars (phi_D,
N_D, R_g, phi_R), loading the COM sidecar for per-species Rg/Rh/occupancy/(r/R),
running segmentwise Green-Kubo viscosity from the stress tensor, and deriving
per-segment Stokes-Einstein diffusion from the segment eta and confinement
radii.

Key inputs
----------
- A standard per-window analysis root under ``--path`` with
  ``ANALYSIS_{SG,DSM,NDSM}/`` density/PCA/cluster CSVs, ``Stress_Tensor`` files,
  and COM/tracked-cluster NPZ sidecars.
- CLI flags: ``--path``, ``--folder``, ``--temp``, ``--tmin``, ``--dt``,
  ``--tmax``; plus block/pymbar/equilibration/viscosity tuning flags
  (``--max-superblock``, ``--min-superblocks``, ``--plateau-fraction``,
  ``--winsorize-pct``, ``--pymbar-conservative``, ``--visc-*``, etc.).

Key outputs (written under ``CORRELATION_{folder}_{temp}_{dt}_{tmin}_{tmax}/``)
- ``RESULTS/CORRELATION_DIAGNOSTICS/``: ``WINDOW_SCALARS.csv``,
  ``VISCOSITY_SEGMENT_SCALARS.csv``, ``CORRELATION_QUANT_DATA_LONG.csv``,
  ``SUPERBLOCK_DIAGNOSTICS.csv``, ``PYMBAR_DIAGNOSTICS.csv``,
  ``SUMMARY_COMPARISON.csv``, ``CORRELATION_QUANT_DATA.csv`` (corrected
  mean+SEM), ``RECOMMENDED_SYSTEM_BLOCKS.csv``,
  ``RECOMMENDED_SYSTEM_BLOCK_LAYOUT.csv``,
  ``RECOMMENDED_VISCOSITY_SEGMENT_LAYOUT.csv``, equilibration audits,
  ``STATISTICAL_POLICY.csv``, and more.
- ``FIGURES/CORRELATION_DIAGNOSTICS/``: per-observable SEM-vs-blocksize and ACF
  plots.

Example invocation
-------------------
    python block_correlation_diagnostics.py \
        --path TEMP_300 --folder CLASSIFY --temp 300 \
        --tmin 50 --dt 50 --tmax 2000
"""
import argparse
import contextlib
import glob
import io
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from acf import acf
from rdp import rdp
from viscosity import visc
from diffusion import (
    _stokes_einstein_D,
    _load_tracked_cluster_npz,
    classify_species,
    get_persistent_inside_outside,
    parse_rdp_com_series,
    rdp_com_sidecar_path,
)


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


PER_SPECIES_SPATIAL_SPECIES = ("G3BP1", "PABP1", "TIA1", "TTP", "FUS", "TDP43", "RNA")


DEFAULT_DSM_LIST = [
    "dsm_anisomycin",
    "dsm_daunorubicin",
    "dsm_dihydrolipoic_acid",
    "dsm_hydroxyquinoline",
    "dsm_lipoamide",
    "dsm_lipoic_acid",
    "dsm_mitoxantrone",
    "dsm_pararosaniline",
    "dsm_pyrivinium",
    "dsm_quinicrine",
]

DEFAULT_NDSM_LIST = [
    "ndsm_dmso",
    "ndsm_valeric_acid",
    "ndsm_ethylenediamine",
    "ndsm_propanedithiol",
    "ndsm_hexanediol",
    "ndsm_diethylaminopentane",
    "ndsm_aminoacridine",
    "ndsm_anthraquinone",
    "ndsm_acetylenapthacene",
    "ndsm_anacardic",
]

SMALL_MOLECULE_ID = {
    "sg_X": "SG",
    "ndsm_dmso": "ND1",
    "ndsm_valeric_acid": "ND2",
    "ndsm_ethylenediamine": "ND3",
    "ndsm_propanedithiol": "ND4",
    "ndsm_hexanediol": "ND5",
    "ndsm_diethylaminopentane": "ND6",
    "ndsm_aminoacridine": "ND7",
    "ndsm_anthraquinone": "ND8",
    "ndsm_acetylenapthacene": "ND9",
    "ndsm_anacardic": "ND10",
    "dsm_hydroxyquinoline": "D1",
    "dsm_lipoamide": "D2",
    "dsm_lipoic_acid": "D3",
    "dsm_dihydrolipoic_acid": "D4",
    "dsm_anisomycin": "D5",
    "dsm_pararosaniline": "D6",
    "dsm_pyrivinium": "D7",
    "dsm_quinicrine": "D8",
    "dsm_mitoxantrone": "D9",
    "dsm_daunorubicin": "D10",
}

QUANT_DATA_OBSERVABLES = {
    "c_dense_sg_fit",
    "c_dilute_sg_fit",
    "c_dense_sg_calc",
    "c_dilute_sg_calc",
    "P_sg",
    "R_cond_A",
    "W_interface_A",
    "gamma1_mN_m",
    "gamma2_mN_m",
    "gamma_ave_mN_m",
    "deltaG_trans_kJ_mol",
    "c_dense_sm_mg_ml",
    "c_dilute_sm_mg_ml",
    "P_sm",
    "phi_D",
    "N_D",
    "R_g_A",
    "phi_R",
    "Rg_conf_A",
    "Rh_conf_A",
    "eta_GK_Pa_s",
    "eta_GK_Theo_Pa_s",
    "D_SE_GK_Rh_um2_s",
    "D_SE_GK_Rg_um2_s",
    "gamma1_protein_mN_m",
    "gamma2_protein_mN_m",
    "gamma_ave_protein_mN_m",
    "gamma1_rna_mN_m",
    "gamma2_rna_mN_m",
    "gamma_ave_rna_mN_m",
} | {
    f"{prefix}_{sp}"
    for sp in PER_SPECIES_SPATIAL_SPECIES
    for prefix in ("Rg", "Rh", "Occ", "r_over_R")
} | {
    f"D_SE_GK_{sp}_um2_s"
    for sp in PER_SPECIES_SPATIAL_SPECIES
}

PER_SPECIES_SPATIAL_OBSERVABLES = {
    f"{prefix}_{sp}"
    for sp in PER_SPECIES_SPATIAL_SPECIES
    for prefix in ("Rg", "Rh", "Occ", "r_over_R")
}

VISCOSITY_OBSERVABLES = {
    "eta_GK_Pa_s",
    "eta_GK_Theo_Pa_s",
    "D_SE_GK_Rh_um2_s",
    "D_SE_GK_Rg_um2_s",
} | {
    f"D_SE_GK_{sp}_um2_s"
    for sp in PER_SPECIES_SPATIAL_SPECIES
}

CORE_RECOMMENDATION_OBSERVABLES = set(QUANT_DATA_OBSERVABLES) - VISCOSITY_OBSERVABLES

# ---------------------------------------------------------------------------
# Equilibration control observables (plan item 2)
# ---------------------------------------------------------------------------
# For LLPS single-trajectory analysis, the system-level equilibration audit
# should be driven only by core structural/phase observables that directly
# characterise condensate formation and stability.  Transport observables
# (diffusion, viscosity) and interfacial fluctuation observables (gamma, W)
# are too noisy to control the equilibration advisory cutoff for slow phase-
# separating CG systems with O(39) post-equilibration windows.
#
# Chodera detect_equilibration is used as a DIAGNOSTIC, not as an automatic
# truncation rule.  The production estimator retains fixed tmin as configured.
EQUILIBRATION_CONTROL_OBSERVABLES = {
    "c_dense_sg_fit",
    "c_dilute_sg_fit",
    "P_sg",
    "R_cond_A",
    "R_g_A",
    "phi_D",
    "N_D",
    "phi_R",
}

# Key observables for the equilibration sensitivity table (cutoff scan).
EQUILIBRATION_SENSITIVITY_OBSERVABLES = {
    "c_dense_sg_fit",
    "c_dilute_sg_fit",
    "R_cond_A",
    "R_g_A",
    "phi_D",
}

# ---------------------------------------------------------------------------
# Statistical policy metadata (plan item 1)
# ---------------------------------------------------------------------------
STATISTICAL_POLICY = {
    "uncertainty_scope": "within_run_time_correlation_corrected",
    "n_replicates": 1,
    "block_selection_method": "max(superblock_plateau_flyvbjerg_petersen, pymbar_statistical_inefficiency)",
    "system_block_rule": "max_over_core_recommendation_observables",
    "equilibration_policy": "fixed_tmin_with_chodera_diagnostic_audit",
    "sem_formula": "std(ddof=1)/sqrt(n_blocks)",
    "ci95_method": "student_t_distribution",
    "winsorization": "per_window_values_winsorized_at_5th_95th_percentile_before_block_analysis",
}

DEFAULT_EQUILIBRATION_SOFT_MAX_NS = 200.0
DEFAULT_EQUILIBRATION_HARD_FRACTION = 1.0 / 3.0
DEFAULT_EQUILIBRATION_SENSITIVITY_CUTOFFS = (50, 100, 150, 200)

DEFAULT_PLOT_OBSERVABLES = [
    "R_cond_A",
    "gamma_ave_mN_m",
    "c_dense_sg_fit",
    "c_dilute_sg_fit",
    "phi_D",
    "N_D",
    "R_g_A",
    "phi_R",
    "eta_GK_Pa_s",
    "eta_GK_Theo_Pa_s",
    "gamma_ave_protein_mN_m",
    "gamma_ave_rna_mN_m",
    "c_dense_sm_mg_ml",
    "c_dilute_sm_mg_ml",
    "P_sm",
]

WINDOW_INIT = {
    "SG": [250, 250, 200, 80],
    "Protein": [150, 150, 200, 80],
    "RNA": [80, 80, 200, 80],
    "SM": [0, 0.2, 200, 50],
}

RDP_WINDOW_QC = {
    # Interfaces wider than ~3x the fitted condensate radius are not
    # meaningful droplet profiles; they correspond to the flat/degenerate
    # windows that blow up the unbounded ERF fit.
    "max_w_over_r": 3.0,
    # For SG fits only, reject windows whose fitted condensate radius is
    # tiny relative to the cluster Rg (degenerate R~0 solution) or many
    # times larger than the cluster itself.
    "sg_min_phi_r": 0.05,
    "sg_max_phi_r": 3.0,
    # Extremely large fit/calc mismatches indicate a pathological fit even
    # if the raw parameters are finite.
    "max_dense_fit_over_calc": 100.0,
}

DEFAULT_VISCOSITY_CONFIG = {
    "segments": 20,
    "dt_unit": 2000000E-15,
    "n_point": 1000,
    "n_tau": 10,
}

class data_load_error(RuntimeError):
    """Raised when an input CSV is missing, empty, or unreadable."""


@dataclass
class pymbar_result:
    """Statistical-inefficiency diagnostics for one observable from PYMBAR."""
    available: bool
    method: str
    g: float
    tau_int_windows: float
    n_eff: float
    indices: List[int]
    mean_uncorrelated: float
    se_uncorrelated: float
    se_g_corrected: float
    message: str = ""


def ensure_dir(path: str) -> None:
    """Create ``path`` (and parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def safe_float(x) -> float:
    """Return ``float(x)`` or NaN if the conversion fails."""
    try:
        return float(x)
    except Exception:
        return math.nan


def sem_from_values(values: np.ndarray) -> float:
    """Return the standard error of the mean over finite values (NaN if <=1)."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size <= 1:
        return math.nan
    return float(np.std(vals, ddof=1) / np.sqrt(vals.size))


def std_from_values(values: np.ndarray) -> float:
    """Return the sample standard deviation over finite values (NaN if <=1)."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size <= 1:
        return math.nan
    return float(np.std(vals, ddof=1))


def mean_from_values(values: np.ndarray) -> float:
    """Return the mean over finite values (NaN if there are none)."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return math.nan
    return float(np.mean(vals))


def winsorize_values(values: np.ndarray, pct: float) -> np.ndarray:
    """Winsorize finite values at the pct/100-th and (100-pct)/100-th percentiles.

    Replaces extreme values with the corresponding percentile threshold.
    Non-finite values are left as-is.  If pct <= 0 or fewer than 3 finite
    values exist, returns values unchanged.
    """
    vals = np.array(values, dtype=float, copy=True)
    if pct <= 0:
        return vals
    finite_mask = np.isfinite(vals)
    finite_vals = vals[finite_mask]
    if finite_vals.size < 3:
        return vals
    lo = float(np.percentile(finite_vals, pct))
    hi = float(np.percentile(finite_vals, 100.0 - pct))
    clipped = np.clip(finite_vals, lo, hi)
    vals[finite_mask] = clipped
    return vals


def parse_cutoff_list(raw: str) -> List[int]:
    """Parse a comma-separated cutoff string into a sorted unique int list."""
    out: List[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(int(round(float(token))))
        except Exception:
            continue
    return sorted(set(out))


def classify_equilibration_audit(
    t_equil_ns: float,
    fraction_discarded: float,
    default_tmin_ns: float,
    soft_max_ns: float,
    hard_fraction: float,
) -> str:
    """Classify a detected equilibration time against the fixed tmin policy.

    Returns one of ``"indeterminate"``, ``"agree_with_default"``,
    ``"later_than_default_soft"`` or ``"later_than_default_hard"`` depending on
    whether the detected ``t_equil_ns`` and discarded fraction stay within the
    default tmin, the soft-max ns, and the hard discard fraction.
    """
    if not np.isfinite(t_equil_ns):
        return "indeterminate"
    frac = float(fraction_discarded) if np.isfinite(fraction_discarded) else math.nan
    if t_equil_ns <= float(default_tmin_ns) and (not np.isfinite(frac) or frac <= hard_fraction):
        return "agree_with_default"
    if t_equil_ns <= float(soft_max_ns) and (not np.isfinite(frac) or frac <= hard_fraction):
        return "later_than_default_soft"
    return "later_than_default_hard"


def summarize_running_integral(dt_ns: np.ndarray, integral: np.ndarray) -> Dict[str, float]:
    """Summarise a Green-Kubo running viscosity integral for convergence QC.

    Returns a dict with the final and abs-max integral values, the time to reach
    90% of the final value, and the late-time slope (absolute and relative) used
    to flag non-converged viscosity segments.
    """
    dt_arr = np.asarray(dt_ns, dtype=float)
    int_arr = np.asarray(integral, dtype=float)
    mask = np.isfinite(dt_arr) & np.isfinite(int_arr)
    dt_arr = dt_arr[mask]
    int_arr = int_arr[mask]
    if dt_arr.size == 0 or int_arr.size == 0:
        return {
            "eta_raw_final_Pa_s": math.nan,
            "eta_raw_absmax_Pa_s": math.nan,
            "time_to_90pct_final_ns": math.nan,
            "late_integral_slope_Pa_s_per_ns": math.nan,
            "late_integral_relative_slope": math.nan,
        }

    eta_final = float(int_arr[-1])
    eta_absmax = float(np.max(np.abs(int_arr)))
    time_to_90 = math.nan
    if np.isfinite(eta_final) and abs(eta_final) > 0:
        threshold = 0.9 * abs(eta_final)
        hit = np.where(np.abs(int_arr) >= threshold)[0]
        if hit.size:
            time_to_90 = float(dt_arr[int(hit[0])])

    late_slope = math.nan
    rel_slope = math.nan
    late_n = max(3, int(math.ceil(0.2 * dt_arr.size)))
    if dt_arr.size >= late_n:
        x = dt_arr[-late_n:]
        y = int_arr[-late_n:]
        if np.unique(x).size >= 2:
            try:
                late_slope = float(np.polyfit(x, y, 1)[0])
                if np.isfinite(eta_final) and abs(eta_final) > 1e-12:
                    duration = max(float(x[-1] - x[0]), 1e-12)
                    rel_slope = abs(late_slope) * duration / abs(eta_final)
            except Exception:
                late_slope = math.nan
                rel_slope = math.nan

    return {
        "eta_raw_final_Pa_s": eta_final,
        "eta_raw_absmax_Pa_s": eta_absmax,
        "time_to_90pct_final_ns": time_to_90,
        "late_integral_slope_Pa_s_per_ns": late_slope,
        "late_integral_relative_slope": rel_slope,
    }


def compute_all_offset_batch_estimator(
    values: np.ndarray,
    unit_starts: np.ndarray,
    unit_ends: np.ndarray,
    block_size: int,
    unit_span_ns: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    """Correlation-corrected mean/SEM via all-offset non-overlapping batch means.

    For a fixed block size, sweeps every starting offset (0..block_size-1),
    forms non-overlapping block means at each offset, and takes the median SEM
    across eligible offsets as the corrected SEM (the full-series mean is the
    corrected mean). Returns ``(offset_rows, block_rows, inference_row)`` of
    per-offset, per-block, and summary records for the diagnostics CSVs.
    """
    n = int(len(values))
    offset_rows: List[Dict[str, object]] = []
    block_rows: List[Dict[str, object]] = []
    eligible_sems: List[float] = []
    eligible_means: List[float] = []
    eligible_blocks: List[int] = []

    if n == 0 or block_size <= 0:
        return offset_rows, block_rows, {
            "corrected_mean": math.nan,
            "corrected_sem": math.nan,
            "n_corrected_blocks": 0,
            "n_offsets_total": 0,
            "n_offsets_used": 0,
            "offset_mean_average": math.nan,
            "offset_mean_sd": math.nan,
            "offset_sem_min": math.nan,
            "offset_sem_median": math.nan,
            "offset_sem_max": math.nan,
            "n_blocks_min": 0,
            "n_blocks_median": 0,
            "n_blocks_max": 0,
            "corrected_method": "mean_full + all_offset_batch_means_median_sem",
        }

    for offset in range(block_size):
        layout = block_layout(n, block_size, offset=offset)
        block_means = []
        used_n = 0
        for block_id, start_idx, end_idx in layout:
            vals = np.asarray(values[start_idx : end_idx + 1], dtype=float)
            win = np.asarray(unit_starts[start_idx : end_idx + 1], dtype=int)
            win_ends = np.asarray(unit_ends[start_idx : end_idx + 1], dtype=int)
            block_mean = mean_from_values(vals)
            block_means.append(block_mean)
            used_n += len(vals)
            block_duration_ns = int(win_ends[-1] - win[0]) if len(win_ends) else int(block_size * unit_span_ns)
            block_rows.append({
                "offset_windows": int(offset),
                "block_id": int(block_id),
                "block_start_window_index": int(start_idx),
                "block_end_window_index": int(end_idx),
                "block_start_ns": int(win[0]) if len(win) else math.nan,
                "block_end_ns": int(win_ends[-1]) if len(win_ends) else math.nan,
                "n_windows_in_block": int(len(win)),
                "block_duration_ns": block_duration_ns,
                "window_members": ";".join(str(x) for x in win),
                "block_mean": block_mean,
            })

        offset_mean = mean_from_values(np.asarray(block_means, dtype=float))
        offset_sem = sem_from_values(np.asarray(block_means, dtype=float))
        offset_n_blocks = int(len(layout))
        eligible = bool(offset_n_blocks >= 2 and np.isfinite(offset_sem))
        offset_rows.append({
            "offset_windows": int(offset),
            "n_raw_windows": int(n),
            "n_used_windows": int(used_n),
            "n_dropped_windows_total": int(n - used_n),
            "n_blocks": offset_n_blocks,
            "offset_mean": offset_mean,
            "offset_sem": offset_sem,
            "eligible_for_inference": eligible,
        })
        if eligible:
            eligible_sems.append(float(offset_sem))
            eligible_means.append(float(offset_mean))
            eligible_blocks.append(offset_n_blocks)

    corrected_mean = mean_from_values(np.asarray(values, dtype=float))
    if eligible_sems:
        corrected_sem = float(np.median(np.asarray(eligible_sems, dtype=float)))
        n_corrected_blocks = int(min(eligible_blocks))
        offset_mean_average = float(np.mean(np.asarray(eligible_means, dtype=float)))
        offset_mean_sd = float(np.std(np.asarray(eligible_means, dtype=float), ddof=1)) if len(eligible_means) > 1 else math.nan
        offset_sem_min = float(np.min(np.asarray(eligible_sems, dtype=float)))
        offset_sem_median = float(np.median(np.asarray(eligible_sems, dtype=float)))
        offset_sem_max = float(np.max(np.asarray(eligible_sems, dtype=float)))
        n_blocks_min = int(min(eligible_blocks))
        n_blocks_median = int(round(float(np.median(np.asarray(eligible_blocks, dtype=float)))))
        n_blocks_max = int(max(eligible_blocks))
    else:
        corrected_sem = math.nan
        n_corrected_blocks = int(max((row["n_blocks"] for row in offset_rows), default=0))
        offset_mean_average = math.nan
        offset_mean_sd = math.nan
        offset_sem_min = math.nan
        offset_sem_median = math.nan
        offset_sem_max = math.nan
        n_blocks_min = 0
        n_blocks_median = 0
        n_blocks_max = 0

    inference_row = {
        "corrected_mean": corrected_mean,
        "corrected_sem": corrected_sem,
        "n_corrected_blocks": n_corrected_blocks,
        "n_offsets_total": int(block_size),
        "n_offsets_used": int(len(eligible_sems)),
        "offset_mean_average": offset_mean_average,
        "offset_mean_sd": offset_mean_sd,
        "offset_sem_min": offset_sem_min,
        "offset_sem_median": offset_sem_median,
        "offset_sem_max": offset_sem_max,
        "n_blocks_min": n_blocks_min,
        "n_blocks_median": n_blocks_median,
        "n_blocks_max": n_blocks_max,
        "corrected_method": "mean_full + all_offset_batch_means_median_sem",
    }
    return offset_rows, block_rows, inference_row


def summarize_sensitivity_cutoff(
    values: np.ndarray,
    unit_starts: np.ndarray,
    unit_ends: np.ndarray,
    unit_span_ns: int,
    cutoff_ns: int,
    args,
) -> Dict[str, object]:
    """Recompute the corrected mean/SEM after discarding windows before a cutoff.

    Restricts the series to windows starting at >= ``cutoff_ns``, re-runs the
    superblock + pymbar block recommendation and the all-offset batch
    estimator, and returns a summary row for the equilibration-sensitivity table.
    """
    mask = np.asarray(unit_starts, dtype=float) >= float(cutoff_ns)
    values_sub = np.asarray(values, dtype=float)[mask]
    starts_sub = np.asarray(unit_starts, dtype=int)[mask]
    ends_sub = np.asarray(unit_ends, dtype=int)[mask]
    if values_sub.size == 0:
        return {
            "cutoff_ns": int(cutoff_ns),
            "n_units_retained": 0,
            "recommended_block_size_units": math.nan,
            "recommended_block_size_ns": math.nan,
            "n_corrected_blocks": 0,
            "corrected_mean": math.nan,
            "corrected_sem": math.nan,
        }

    _, summary_rows_sub, b_super_sub, _ = compute_superblock_tables(
        values_sub,
        starts_sub,
        max_block_size=args.max_superblock,
        min_superblocks=args.min_superblocks,
        plateau_fraction=args.plateau_fraction,
    )
    pymbar_sub = run_pymbar(values_sub, args.pymbar_conservative)
    b_pymbar_sub = int(max(1, math.ceil(pymbar_sub.g))) if (pymbar_sub.available and np.isfinite(pymbar_sub.g)) else 1
    valid_max_b_sub = max([int(r["superblock_size_windows"]) for r in summary_rows_sub], default=1)
    recommended_b_sub = max(b_super_sub or 1, b_pymbar_sub)
    recommended_b_sub = min(recommended_b_sub, valid_max_b_sub)
    _, _, inference_sub = compute_all_offset_batch_estimator(
        values_sub,
        starts_sub,
        ends_sub,
        recommended_b_sub,
        unit_span_ns,
    )
    return {
        "cutoff_ns": int(cutoff_ns),
        "n_units_retained": int(values_sub.size),
        "recommended_block_size_units": int(recommended_b_sub),
        "recommended_block_size_ns": int(recommended_b_sub * unit_span_ns),
        "n_corrected_blocks": int(inference_sub.get("n_corrected_blocks", 0)),
        "corrected_mean": inference_sub.get("corrected_mean", math.nan),
        "corrected_sem": inference_sub.get("corrected_sem", math.nan),
    }


def load_sm_lists(analysis_root: str, use_lists: bool) -> Tuple[List[str], List[str]]:
    """Return (dsm_list, ndsm_list) of small-molecule systems.

    With ``use_lists`` it reads ``dsm_list.txt`` / ``ndsm_list.txt`` from the
    analysis root when present; otherwise returns the hard-coded defaults.
    """
    if not use_lists:
        return list(DEFAULT_DSM_LIST), list(DEFAULT_NDSM_LIST)

    dsm_list = []
    ndsm_list = []
    dsm_fp = os.path.join(analysis_root, "dsm_list.txt")
    ndsm_fp = os.path.join(analysis_root, "ndsm_list.txt")
    if os.path.isfile(dsm_fp):
        with open(dsm_fp, "r", encoding="utf-8") as handle:
            dsm_list = [line.strip() for line in handle if line.strip()]
    if os.path.isfile(ndsm_fp):
        with open(ndsm_fp, "r", encoding="utf-8") as handle:
            ndsm_list = [line.strip() for line in handle if line.strip()]
    return dsm_list, ndsm_list


def time_candidates(window_start: int) -> List[str]:
    """Return candidate string spellings of a window-start time tag."""
    return [
        f"{window_start}",
        f"{int(window_start)}",
        f"{float(window_start):.1f}",
        f"{float(window_start)}",
    ]


def resolve_window_file(category_dir: str, prefix: str, system_name: str, window_start: int) -> str:
    """Locate ``{prefix}_{system}_{t}.csv`` for one window across time spellings.

    Returns the first existing candidate, else the preferred ``.1f`` path even
    if absent (callers test existence).
    """
    preferred = os.path.join(category_dir, f"{prefix}_{system_name}_{float(window_start):.1f}.csv")
    for tag in time_candidates(window_start):
        candidate = os.path.join(category_dir, f"{prefix}_{system_name}_{tag}.csv")
        if os.path.isfile(candidate):
            return candidate
    return preferred


def read_csv_nonempty(path: str) -> pd.DataFrame:
    """Read a CSV into a DataFrame, raising ``data_load_error`` if missing or empty."""
    if not os.path.isfile(path):
        raise data_load_error(f"missing file: {path}")
    if os.path.getsize(path) == 0:
        raise data_load_error(f"empty file: {path}")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise data_load_error(f"empty csv: {path}") from exc
    except Exception as exc:
        raise data_load_error(f"failed to read csv: {path}: {exc}") from exc
    if df.empty:
        raise data_load_error(f"empty dataframe: {path}")
    return df


def sanitize_density_for_rdp(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a density-profile frame into the (Distance, Density, Sigma) schema RDP expects."""
    x = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    y = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    if "Standard mean error" in df.columns:
        sigma = pd.to_numeric(df["Standard mean error"], errors="coerce")
    elif len(df.columns) >= 4:
        sigma = pd.to_numeric(df.iloc[:, 3], errors="coerce")
    elif len(df.columns) >= 3:
        sigma = pd.to_numeric(df.iloc[:, 2], errors="coerce")
    else:
        sigma = pd.Series(np.nan, index=df.index)

    sigma = sigma.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pd.DataFrame({
        "Distance": x,
        "Density": y,
        "Sigma": sigma,
    })


def sanitize_cluster_for_rdp(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a per-window cluster frame carries a scalar ``RG SEM`` column for RDP.

    If ``RG SEM`` is absent or non-finite, fills it from the SEM of the
    per-frame ``Largest Droplet Radius of Gyration`` values.
    """
    out = df.copy()
    if "RG SEM" in out.columns:
        first = safe_float(out["RG SEM"].iloc[0])
        if np.isfinite(first):
            out["RG SEM"] = first
            return out
    if "Largest Droplet Radius of Gyration" in out.columns:
        rg_vals = pd.to_numeric(out["Largest Droplet Radius of Gyration"], errors="coerce").to_numpy(dtype=float)
        rg_sem = sem_from_values(rg_vals)
    else:
        rg_sem = math.nan
    out["RG SEM"] = rg_sem
    return out


def safe_ratio(num: float, den: float) -> float:
    """Return ``num/den`` or NaN if either is non-finite or the denominator is 0."""
    if not np.isfinite(num) or not np.isfinite(den) or den == 0:
        return math.nan
    return float(num / den)


def _fmt_qc_float(x: float) -> str:
    """Format a float for QC reason strings (``"nan"`` when non-finite)."""
    if np.isfinite(x):
        return f"{float(x):.6g}"
    return "nan"


def _rdp_window_qc_reasons(fitter, label: str) -> List[str]:
    """Return human-readable QC failure reasons for one window's RDP fit.

    Applies the ``RDP_WINDOW_QC`` thresholds (non-positive R/W, excessive W/R,
    and for SG fits a degenerate R/Rg ratio or huge fit/calc density mismatch);
    an empty list means the window passed.
    """
    reasons: List[str] = []

    fit_r_raw = safe_float(getattr(fitter, "fit_R", math.nan))
    fit_w_raw = safe_float(getattr(fitter, "fit_W", math.nan))
    if not np.isfinite(fit_r_raw) or fit_r_raw <= 0.0:
        reasons.append(f"fit_R_raw={_fmt_qc_float(fit_r_raw)} <= 0")
    if not np.isfinite(fit_w_raw) or fit_w_raw <= 0.0:
        reasons.append(f"fit_W_raw={_fmt_qc_float(fit_w_raw)} <= 0")

    if np.isfinite(fit_r_raw) and fit_r_raw > 0.0 and np.isfinite(fit_w_raw) and fit_w_raw > 0.0:
        w_over_r = fit_w_raw / fit_r_raw
        if not np.isfinite(w_over_r) or w_over_r > float(RDP_WINDOW_QC["max_w_over_r"]):
            reasons.append(
                f"W_over_R={_fmt_qc_float(w_over_r)} > {RDP_WINDOW_QC['max_w_over_r']:.1f}"
            )

    if label == "SG" and np.isfinite(fit_r_raw) and fit_r_raw > 0.0:
        rg_cluster = abs(safe_float(getattr(fitter, "radius", math.nan)))
        if np.isfinite(rg_cluster) and rg_cluster > 0.0:
            phi_r = fit_r_raw / rg_cluster
            if phi_r < float(RDP_WINDOW_QC["sg_min_phi_r"]) or phi_r > float(RDP_WINDOW_QC["sg_max_phi_r"]):
                reasons.append(
                    f"phi_R={_fmt_qc_float(phi_r)} outside "
                    f"[{RDP_WINDOW_QC['sg_min_phi_r']:.2f}, {RDP_WINDOW_QC['sg_max_phi_r']:.1f}]"
                )

        c_dense_fit = abs(safe_float(getattr(fitter, "c_dense_fit", math.nan)))
        c_dense_calc = abs(safe_float(getattr(fitter, "c_dense_calc", math.nan)))
        if np.isfinite(c_dense_fit) and np.isfinite(c_dense_calc) and c_dense_calc > 0.0:
            dense_ratio = c_dense_fit / c_dense_calc
            if dense_ratio > float(RDP_WINDOW_QC["max_dense_fit_over_calc"]):
                reasons.append(
                    f"c_dense_fit/calc={_fmt_qc_float(dense_ratio)} > "
                    f"{RDP_WINDOW_QC['max_dense_fit_over_calc']:.0f}"
                )

    return reasons


def fit_rdp_window(
    density_path: str,
    pca_path: str,
    cluster_path: str,
    temperature: int,
    label: str,
    init: Sequence[float],
) -> Tuple[bool, Dict[str, float], str]:
    """Fit the RDP sigmoid model to one window's density/PCA/cluster files.

    Sanitises the inputs into temp CSVs, runs the ``RDP`` fitter, applies the
    window QC, and returns ``(ok, result_dict, note)`` where result_dict holds
    the dense/dilute concentrations, R/W, partition ratio, surface tensions and
    deltaG. On failure returns ``(False, {}, reason)``.
    """
    try:
        density_df = read_csv_nonempty(density_path)
        pca_df = read_csv_nonempty(pca_path)
        cluster_df = read_csv_nonempty(cluster_path)
    except data_load_error as exc:
        return False, {}, str(exc)

    with tempfile.TemporaryDirectory(prefix="corrdiag_") as tmpdir:
        density_tmp = os.path.join(tmpdir, "density.csv")
        pca_tmp = os.path.join(tmpdir, "pca.csv")
        cluster_tmp = os.path.join(tmpdir, "cluster.csv")
        sanitize_density_for_rdp(density_df).to_csv(density_tmp, index=False)
        pca_df.to_csv(pca_tmp, index=False)
        sanitize_cluster_for_rdp(cluster_df).to_csv(cluster_tmp, index=False)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                fitter = rdp(density_tmp, pca_tmp, cluster_tmp, temperature, list(init), label)
        except Exception as exc:
            return False, {}, f"RDP fit failed: {exc}"

    fit_ok = bool(
        np.isfinite(fitter.fit_R)
        and np.isfinite(fitter.fit_W)
        and (abs(fitter.fit_A) + abs(fitter.fit_B) + abs(fitter.fit_R) + abs(fitter.fit_W) > 0)
    )
    if not fit_ok:
        return False, {}, "RDP fit returned non-finite or zero parameters"

    qc_reasons = _rdp_window_qc_reasons(fitter, label)
    if qc_reasons:
        return False, {}, "RDP window QC failed: " + "; ".join(qc_reasons)

    c_dense_fit = abs(safe_float(fitter.c_dense_fit))
    c_dilute_fit = abs(safe_float(fitter.c_dilute_fit))
    c_dense_calc = abs(safe_float(fitter.c_dense_calc))
    c_dilute_calc = abs(safe_float(fitter.c_dilute_calc))
    p_ratio = safe_ratio(c_dense_fit, c_dilute_fit)

    result = {
        "fit_A": safe_float(fitter.fit_A),
        "fit_B": safe_float(fitter.fit_B),
        "R_cond_A": abs(safe_float(fitter.fit_R)),
        "W_interface_A": abs(safe_float(fitter.fit_W)),
        "c_dense_fit": c_dense_fit,
        "c_dilute_fit": c_dilute_fit,
        "c_dense_calc": c_dense_calc,
        "c_dilute_calc": c_dilute_calc,
        "P_ratio": p_ratio,
        "gamma1_mN_m": safe_float(fitter.st_1) * 1000.0,
        "gamma2_mN_m": safe_float(fitter.st_2) * 1000.0,
        "gamma_ave_mN_m": safe_float(fitter.st) * 1000.0,
        "deltaG_trans_kJ_mol": safe_float(fitter.dG) / 1000.0,
    }
    return True, result, ""


def extract_cluster_scalars(cluster_path: str, r_cond_a: float) -> Tuple[bool, Dict[str, float], str]:
    """Read per-window cluster scalars (phi_D, N_D, R_g, phi_R, masses, etc.).

    Averages the relevant cluster-table columns and derives phi_D (largest-
    droplet mass fraction) and phi_R (R_cond / R_g). Returns ``(ok, dict, note)``.
    """
    try:
        df = read_csv_nonempty(cluster_path)
    except data_load_error as exc:
        return False, {}, str(exc)

    def mean_col(name: str) -> float:
        """Mean of finite values in column ``name`` (NaN if column absent)."""
        if name not in df.columns:
            return math.nan
        return mean_from_values(pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float))

    total_mass = mean_col("Total Mass (mg)")
    total_chain_number = mean_col("Total Chain Number")
    rg = mean_col("Largest Droplet Radius of Gyration")
    n_d = mean_col("Number of Droplets")
    chains_largest = mean_col("Chains in Largest Droplet")
    mass_largest = mean_col("Mass of Largest Droplet (mg)")
    n_external = mean_col("Number of External Chains")
    mass_external = mean_col("Mass of External Chains")

    phi_d = safe_ratio(mass_largest, total_mass)
    phi_r = safe_ratio(r_cond_a, rg)

    result = {
        "total_mass_mg": total_mass,
        "total_chain_number": total_chain_number,
        "R_g_A": rg,
        "N_D": n_d,
        "chains_largest": chains_largest,
        "mass_largest_mg": mass_largest,
        "number_external": n_external,
        "mass_external_mg": mass_external,
        "phi_D": phi_d,
        "phi_R": phi_r,
    }
    return True, result, ""


def _resolve_msd_rdp_path(category_dir: str, system_name: str, temperature: int, category: str) -> Optional[str]:
    """Resolve the `*_msd_rdp.out.all` path for a system.

    Checks (1) the local analysis dir, (2) env-var ANALYSIS_MSD_ROOTS, and
    (3) the standard /projects cluster MSD mirror, mirroring
    ``ANALYSIS._resolve_msd_path``.
    """
    local_path = os.path.join(category_dir, f"{system_name}_msd_rdp.out.all")
    if os.path.isfile(local_path):
        return local_path
    roots: List[str] = []
    env_roots = os.environ.get("ANALYSIS_MSD_ROOTS")
    if env_roots:
        roots.extend(p for p in env_roots.split(os.pathsep) if p)
    roots.extend([
        os.path.join(_REPO_ROOT, "PYTHON_ANALYSIS"),
        os.path.join(_REPO_ROOT, "PYTHON_SIMULATIONS"),
    ])
    cat_upper = category.upper()
    for root in roots:
        if not os.path.isdir(root):
            continue
        direct = os.path.join(root, f"TEMP_{temperature}", cat_upper, "MSD", f"{system_name}_msd_rdp.out.all")
        if os.path.isfile(direct):
            return direct
    for root in roots:
        if not os.path.isdir(root):
            continue
        pattern = os.path.join(root, "**", "MSD", f"{system_name}_msd_rdp.out.all")
        for candidate in glob.glob(pattern, recursive=True):
            if str(temperature) in candidate:
                return os.path.abspath(candidate)
    return None


def _per_species_window_means(
    times_ns: np.ndarray,
    value_mat: np.ndarray,
    species_by_col: np.ndarray,
    window_start: float,
    window_end: float,
) -> Dict[str, float]:
    """Mean of `value_mat[time_in_window, :]` by species.

    value_mat shape: (n_time, n_res). species_by_col shape: (n_res,).
    Time window is [window_start, window_end).  Returns NaN for species
    with no finite samples.
    """
    mask = (times_ns >= float(window_start)) & (times_ns < float(window_end))
    out: Dict[str, float] = {}
    if not np.any(mask):
        for sp in PER_SPECIES_SPATIAL_SPECIES:
            out[sp] = math.nan
        return out
    sub = value_mat[mask]
    for sp in PER_SPECIES_SPATIAL_SPECIES:
        cols = species_by_col == sp
        if not np.any(cols):
            out[sp] = math.nan
            continue
        vals = sub[:, cols].astype(float)
        vals = vals[np.isfinite(vals)]
        out[sp] = float(vals.mean()) if vals.size else math.nan
    return out


def _window_mean_for_cols(
    times_ns: np.ndarray,
    value_mat: np.ndarray,
    col_mask: np.ndarray,
    window_start: float,
    window_end: float,
) -> float:
    """Mean of ``value_mat`` over the selected columns within a time window.

    Restricts rows to ``[window_start, window_end)`` and columns to ``col_mask``,
    then returns the mean of finite entries (NaN if none).
    """
    mask = (times_ns >= float(window_start)) & (times_ns < float(window_end))
    if not np.any(mask) or not np.any(col_mask):
        return math.nan
    sub = value_mat[mask][:, col_mask].astype(float)
    vals = sub[np.isfinite(sub)]
    return float(vals.mean()) if vals.size else math.nan


def extract_per_species_spatial_scalars_for_system(
    temp_root: str,
    temperature: int,
    category: str,
    system_name: str,
    windows: Sequence[int],
    dt_ns: int,
    tmin_ns: int,
    tmax_ns: int,
) -> List[Dict[str, object]]:
    """Per-window per-species Rg, Rh, occupancy, and r/R_cluster.

    Loads the COM sidecar (`*_msd_rdp_com.npz`) once per system and, when
    available, the tracked-cluster NPZ sidecars for inside-membership and
    condensate radial position.  For each analysis window [t, t+dt_ns),
    emits per-species mean values via ``add_long_rows`` with
    ``sampling_unit_family="diffusion_segment"``.  Observables missing
    from the sidecar (e.g. RNA Rh when the sidecar predates the Kirkwood
    addition) are emitted as NaN so downstream block correction still sees
    a uniform schema.
    """
    category_dir = os.path.join(temp_root, f"ANALYSIS_{category}")
    rows: List[Dict[str, object]] = []

    nan_observables = {
        "Rg_conf_A": math.nan,
        "Rh_conf_A": math.nan,
    }
    nan_observables.update({f"{prefix}_{sp}": math.nan
                       for sp in PER_SPECIES_SPATIAL_SPECIES
                       for prefix in ("Rg", "Rh", "Occ", "r_over_R")})

    def emit_nans(note: str, source: str = "") -> None:
        """Emit NaN per-species spatial rows for every window (sidecar missing)."""
        for i, window_start in enumerate(windows):
            window_end = int(window_start) + int(dt_ns)
            add_long_rows(
                rows,
                temperature,
                category,
                system_name,
                i,
                int(window_start),
                window_end,
                dict(nan_observables),
                False,
                "",
                "",
                source,
                note,
                sampling_unit_family="diffusion_segment",
            )

    msd_path = _resolve_msd_rdp_path(category_dir, system_name, temperature, category)
    if msd_path is None:
        emit_nans("no msd file; per-species spatial scalars skipped")
        return rows
    sidecar_path = rdp_com_sidecar_path(msd_path)
    if not os.path.isfile(sidecar_path):
        emit_nans(f"no com sidecar alongside {os.path.basename(msd_path)}", sidecar_path)
        return rows
    try:
        times_ns, com_A, rg_A, rh_A, resids, _ = parse_rdp_com_series(sidecar_path, t_start_ns=0.0)
    except Exception as exc:
        emit_nans(f"failed to read com sidecar: {exc}", sidecar_path)
        return rows

    resids_arr = np.asarray(resids, dtype=int)
    species_by_col = np.array([classify_species(int(r)) for r in resids_arr])
    max_resid = int(np.max(resids_arr)) if resids_arr.size else 0
    strict_inside_set = set()
    relaxed_inside_set = set()
    if max_resid > 0:
        strict_inside, _ = get_persistent_inside_outside(
            cluster_root=temp_root,
            tag=system_name,
            tmin=int(tmin_ns),
            dt=int(dt_ns),
            tmax=int(tmax_ns),
            n_res=max_resid,
            t_start_analysis_ns=float(tmin_ns),
            inside_fraction=1.0,
        )
        relaxed_inside, _ = get_persistent_inside_outside(
            cluster_root=temp_root,
            tag=system_name,
            tmin=int(tmin_ns),
            dt=int(dt_ns),
            tmax=int(tmax_ns),
            n_res=max_resid,
            t_start_analysis_ns=float(tmin_ns),
            inside_fraction=0.80,
        )
        strict_inside_set = set(strict_inside)
        relaxed_inside_set = set(relaxed_inside)
    strict_mask = np.array([int(r) in strict_inside_set for r in resids_arr], dtype=bool)
    relaxed_mask = np.array([int(r) in relaxed_inside_set for r in resids_arr], dtype=bool)

    # Tracked cluster NPZ for inside-mask and cluster COM/R_rms — optional.
    tracked = _load_tracked_cluster_npz(
        cluster_root=temp_root, tag=system_name,
        tmin=int(tmin_ns), dt=int(dt_ns), tmax=int(tmax_ns),
    )
    inside_aligned = None
    r_over_R_mat = None
    if tracked is not None and com_A is not None:
        sg_resids = np.asarray(tracked["sg_resids"], dtype=int)
        inside_all = np.concatenate(tracked["inside_masks"], axis=0)
        nF_tc = inside_all.shape[0]
        n_use = min(nF_tc, com_A.shape[0], times_ns.size)
        inside_all = inside_all[:n_use]
        com = com_A[:n_use]
        resid_to_col = {int(r): k for k, r in enumerate(sg_resids)}
        cols_map = np.array([resid_to_col.get(int(r), -1) for r in resids_arr], dtype=int)
        valid_map = cols_map >= 0
        if np.any(valid_map):
            inside_aligned_local = np.zeros((n_use, resids_arr.size), dtype=bool)
            inside_aligned_local[:, valid_map] = inside_all[:, cols_map[valid_map]]
            inside_aligned = inside_aligned_local
            n_inside_pf = inside_aligned.sum(axis=1)
            good_frames = n_inside_pf >= 3
            r_norm = np.full((n_use, resids_arr.size), math.nan, dtype=float)
            for t_idx in np.nonzero(good_frames)[0]:
                mask_t = inside_aligned[t_idx]
                positions = com[t_idx, mask_t]
                ccom = positions.mean(axis=0)
                d2 = ((positions - ccom) ** 2).sum(axis=1)
                Rrms = float(math.sqrt(d2.mean())) if d2.size else 0.0
                if Rrms <= 0.0 or not math.isfinite(Rrms):
                    continue
                d_all = np.linalg.norm(com[t_idx] - ccom, axis=1)
                r_norm[t_idx] = d_all / Rrms
            r_over_R_mat = r_norm
            # Align times_ns (trim to n_use) for window slicing of tracked-cluster values
            tracked_times_ns = times_ns[:n_use]
        else:
            inside_aligned = None

    for i, window_start in enumerate(windows):
        window_start_f = float(window_start)
        window_end_f = window_start_f + float(dt_ns)
        observable_map: Dict[str, float] = {}

        # System-level confinement radii: strict-inside persistent chains only,
        # matching the legacy D_SE,GK system-level source radii.
        observable_map["Rg_conf_A"] = _window_mean_for_cols(
            times_ns, rg_A, strict_mask, window_start_f, window_end_f
        )
        observable_map["Rh_conf_A"] = _window_mean_for_cols(
            times_ns, rh_A, strict_mask, window_start_f, window_end_f
        )

        # Per-species Rg and Rh: relaxed-inside (80%) persistent chains only,
        # matching the legacy per-species confinement radii semantics.
        for sp in PER_SPECIES_SPATIAL_SPECIES:
            cols_sp = (species_by_col == sp) & relaxed_mask
            observable_map[f"Rg_{sp}"] = _window_mean_for_cols(
                times_ns, rg_A, cols_sp, window_start_f, window_end_f
            )
            observable_map[f"Rh_{sp}"] = _window_mean_for_cols(
                times_ns, rh_A, cols_sp, window_start_f, window_end_f
            )

        # Occupancy and r/R: only if tracked cluster NPZ is present.
        if inside_aligned is not None:
            mask_w = (tracked_times_ns >= window_start_f) & (tracked_times_ns < window_end_f)
            inside_w = inside_aligned[mask_w]
            r_over_R_w = r_over_R_mat[mask_w] if r_over_R_mat is not None else None
            n_inside_pf_w = inside_w.sum(axis=1) if inside_w.size else np.array([], dtype=int)
            good_w = n_inside_pf_w >= 3 if inside_w.size else np.array([], dtype=bool)
            for sp in PER_SPECIES_SPATIAL_SPECIES:
                cols_sp = species_by_col == sp
                if not np.any(cols_sp) or inside_w.size == 0:
                    observable_map[f"Occ_{sp}"] = math.nan
                    observable_map[f"r_over_R_{sp}"] = math.nan
                    continue
                if not np.any(good_w):
                    observable_map[f"Occ_{sp}"] = math.nan
                    observable_map[f"r_over_R_{sp}"] = math.nan
                    continue
                occ_vals = inside_w[good_w][:, cols_sp].astype(float)
                observable_map[f"Occ_{sp}"] = float(occ_vals.mean()) if occ_vals.size else math.nan
                if r_over_R_w is not None:
                    masked = inside_w[:, cols_sp] & good_w[:, None]
                    sub = r_over_R_w[:, cols_sp]
                    vals = sub[masked]
                    vals = vals[np.isfinite(vals)]
                    observable_map[f"r_over_R_{sp}"] = float(vals.mean()) if vals.size else math.nan
                else:
                    observable_map[f"r_over_R_{sp}"] = math.nan
        else:
            for sp in PER_SPECIES_SPATIAL_SPECIES:
                observable_map[f"Occ_{sp}"] = math.nan
                observable_map[f"r_over_R_{sp}"] = math.nan

        note = f"com sidecar={os.path.basename(sidecar_path)}; tracked={'yes' if inside_aligned is not None else 'no'}"
        add_long_rows(
            rows,
            temperature,
            category,
            system_name,
            i,
            int(window_start),
            int(window_start) + int(dt_ns),
            observable_map,
            True,
            "",
            "",
            sidecar_path,
            note,
            sampling_unit_family="diffusion_segment",
        )
    return rows


def derive_stokes_einstein_segment_scalars(
    temperature: int,
    category: str,
    system_name: str,
    window_rows: Sequence[Dict[str, object]],
    viscosity_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Build native viscosity-segment D_SE observables from segment eta_GK.

    D_SE is not a native window observable. The clean correlated analogue is
    therefore a per-viscosity-segment derived scalar using the segment's raw
    eta_GK together with radius observables averaged over the same time span.
    """
    if not viscosity_rows:
        return []

    window_map = build_window_observable_map(window_rows)
    if not window_map:
        return []

    def _segment_window_mean(segment_start: int, segment_end: int, observable_key: str) -> float:
        """Mean of an observable over windows starting in [segment_start, segment_end)."""
        vals = []
        for window_start, obs_map in window_map.items():
            if int(window_start) < int(segment_start) or int(window_start) >= int(segment_end):
                continue
            val = safe_float(obs_map.get(observable_key, math.nan))
            if np.isfinite(val):
                vals.append(val)
        if not vals:
            return math.nan
        return float(np.mean(np.asarray(vals, dtype=float)))

    derived_rows: List[Dict[str, object]] = []
    for row in viscosity_rows:
        if str(row.get("observable_key", "")) != "eta_GK_Pa_s":
            continue
        eta = safe_float(row.get("value", math.nan))
        seg_start = int(row.get("window_start_ns", 0))
        seg_end = int(row.get("window_end_ns", seg_start))
        segment_index = int(row.get("window_index", 0))
        base_note = str(row.get("note", ""))
        fit_ok = bool(row.get("fit_ok", False)) and np.isfinite(eta) and eta > 0.0

        observable_map: Dict[str, float] = {}

        rg_conf = _segment_window_mean(seg_start, seg_end, "Rg_conf_A")
        rh_conf = _segment_window_mean(seg_start, seg_end, "Rh_conf_A")
        d_rh, _ = _stokes_einstein_D(eta, 0.0, rh_conf, 0.0, temperature)
        d_rg, _ = _stokes_einstein_D(eta, 0.0, rg_conf, 0.0, temperature)
        observable_map["D_SE_GK_Rh_um2_s"] = d_rh * 1e12 if np.isfinite(d_rh) else math.nan
        observable_map["D_SE_GK_Rg_um2_s"] = d_rg * 1e12 if np.isfinite(d_rg) else math.nan

        for sp in PER_SPECIES_SPATIAL_SPECIES:
            rh_sp = _segment_window_mean(seg_start, seg_end, f"Rh_{sp}")
            d_sp, _ = _stokes_einstein_D(eta, 0.0, rh_sp, 0.0, temperature)
            observable_map[f"D_SE_GK_{sp}_um2_s"] = d_sp * 1e12 if np.isfinite(d_sp) else math.nan

        note = (
            f"{base_note}; derived from per-segment eta_GK and "
            f"segment-averaged window confinement radii"
        )
        add_long_rows(
            derived_rows,
            temperature,
            category,
            system_name,
            segment_index,
            seg_start,
            seg_end,
            observable_map,
            fit_ok,
            "",
            "",
            str(row.get("source_cluster_file", "")),
            note,
            sampling_unit_family="viscosity_segment",
        )
    return derived_rows


def build_window_observable_map(rows: Sequence[Dict[str, object]]) -> Dict[int, Dict[str, float]]:
    """Index long-format scalar rows as ``{window_start: {observable_key: value}}``."""
    out: Dict[int, Dict[str, float]] = {}
    for row in rows:
        try:
            window_start = int(row["window_start_ns"])
            observable_key = str(row["observable_key"])
            value = safe_float(row["value"])
        except Exception:
            continue
        out.setdefault(window_start, {})[observable_key] = value
    return out


def read_stress_tensor_raw(stress_path: str, tmin_ns: int, tmax_ns: int) -> Tuple[np.ndarray, np.ndarray]:
    """Read the off-diagonal stress-tensor time series within [tmin, tmax) ns.

    Returns ``(time_ns, pxyz)`` where pxyz is the negated ``(P_xy, P_xz, P_yz)``
    columns, masked to finite rows inside the analysis window.
    """
    df = read_csv_nonempty(stress_path)
    time_ns = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float) * 20e-6
    pxy = -pd.to_numeric(df.iloc[:, 4], errors="coerce").to_numpy(dtype=float)
    pxz = -pd.to_numeric(df.iloc[:, 5], errors="coerce").to_numpy(dtype=float)
    pyz = -pd.to_numeric(df.iloc[:, 6], errors="coerce").to_numpy(dtype=float)
    pxyz = np.column_stack([pxy, pxz, pyz])
    mask = np.isfinite(time_ns) & np.all(np.isfinite(pxyz), axis=1) & (time_ns >= float(tmin_ns)) & (time_ns < float(tmax_ns))
    return time_ns[mask], pxyz[mask]


def compute_window_acf_from_pressure(p_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the off-diagonal stress autocorrelation averaged over xy/xz/yz.

    Returns ``(lags, acf)`` (lags in sample units) up to half the series length,
    or empty arrays for too-short input.
    """
    xyz = np.asarray(p_xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[0] < 4 or xyz.shape[1] != 3:
        return np.array([], dtype=float), np.array([], dtype=float)
    helper = acf()
    dt = np.arange(0, int(xyz.shape[0] / 2), 1, dtype=float)
    if dt.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    acf_vals = []
    for lag in dt.astype(int):
        if lag <= 1:
            acf_temp = (
                helper.multi_kernel(xyz[..., 0], lag).mean()
                + helper.multi_kernel(xyz[..., 1], lag).mean()
                + helper.multi_kernel(xyz[..., 2], lag).mean()
            ) / 3.0
        else:
            acf_temp = (
                helper.multi_kernel_block(xyz[..., 0], lag).mean()
                + helper.multi_kernel_block(xyz[..., 1], lag).mean()
                + helper.multi_kernel_block(xyz[..., 2], lag).mean()
            ) / 3.0
        acf_vals.append(acf_temp)
    return dt, np.asarray(acf_vals, dtype=float)


def extract_viscosity_scalars_for_system(
    temp_root: str,
    temperature: int,
    category: str,
    system_name: str,
    windows: Sequence[int],
    dt_ns: int,
    tmin_ns: int,
    tmax_ns: int,
    sg_window_map: Dict[int, Dict[str, float]],
    visc_cfg: Dict[str, float],
    return_diagnostics: bool = False,
):
    """Compute per-segment Green-Kubo viscosity scalars for one system.

    Reads the condensate-tracked stress tensor (or a fallback), converts to
    pressure using the mean SG condensate volume, splits into contiguous
    segments, and runs the Green-Kubo / Maxwell-mode fit per segment to emit
    ``eta_GK_Pa_s`` and ``eta_GK_Theo_Pa_s`` as long rows. When
    ``return_diagnostics`` is set, also returns per-segment diagnostic and
    running-integral summary rows; otherwise returns only the scalar rows.
    """
    category_dir = os.path.join(temp_root, f"ANALYSIS_{category}")
    rows: List[Dict[str, object]] = []
    diagnostic_rows: List[Dict[str, object]] = []
    integral_summary_rows: List[Dict[str, object]] = []
    segment_count = max(1, int(visc_cfg.get("segments", DEFAULT_VISCOSITY_CONFIG["segments"])))
    sm_id = SMALL_MOLECULE_ID.get(system_name, system_name)

    tracked_paths: List[str] = []
    for window_start in windows:
        tracked = resolve_window_file(category_dir, "Stress_Tensor_Tracked", system_name, int(window_start))
        if tracked is not None:
            tracked_paths.append(tracked)
    # Preserve order and uniqueness.
    tracked_paths = list(dict.fromkeys(tracked_paths))

    if tracked_paths:
        stress_sources = tracked_paths
        stress_path = ";".join(tracked_paths)
    else:
        fallback = resolve_window_file(category_dir, "Stress_Tensor", system_name, 0)
        stress_sources = [fallback] if fallback is not None else []
        stress_path = fallback

    def fallback_bounds() -> List[Tuple[int, int]]:
        """Return ``segment_count`` evenly spaced (start, end) ns bounds over [tmin, tmax]."""
        edges = np.linspace(float(tmin_ns), float(tmax_ns), segment_count + 1)
        out = []
        for i in range(segment_count):
            out.append((int(round(edges[i])), int(round(edges[i + 1]))))
        return out

    def emit_missing_rows(bounds: Sequence[Tuple[int, int]], note: str) -> None:
        """Emit NaN eta_GK rows (and diagnostics) for each segment when stress data is unusable."""
        for i, (segment_start, segment_end) in enumerate(bounds):
            add_long_rows(
                rows,
                temperature,
                category,
                system_name,
                i,
                segment_start,
                segment_end,
                {"eta_GK_Pa_s": math.nan, "eta_GK_Theo_Pa_s": math.nan},
                False,
                "",
                "",
                stress_path,
                note,
                sampling_unit_family="viscosity_segment",
            )
            if return_diagnostics:
                diagnostic_rows.append({
                    "temperature_K": temperature,
                    "category": category,
                    "system_name": system_name,
                    "small_molecule_id": sm_id,
                    "segment_index": int(i),
                    "segment_start_ns": int(segment_start),
                    "segment_end_ns": int(segment_end),
                    "segment_duration_ns": int(segment_end - segment_start),
                    "n_stress_points": 0,
                    "segment_count_requested": int(segment_count),
                    "mean_r_total_A": math.nan,
                    "volume_m3": math.nan,
                    "eta_GK_Pa_s": math.nan,
                    "eta_GK_Theo_Pa_s": math.nan,
                    "acf0": math.nan,
                    "fit_mae": math.nan,
                    "maxwell_mode_count": 0,
                    "theoretical_success": False,
                    "time_to_90pct_final_ns": math.nan,
                    "late_integral_slope_Pa_s_per_ns": math.nan,
                    "late_integral_relative_slope": math.nan,
                    "eta_raw_absmax_Pa_s": math.nan,
                    "note": note,
                })

    if not stress_sources or any(not os.path.isfile(path) for path in stress_sources):
        emit_missing_rows(fallback_bounds(), "missing stress tensor file")
        return (rows, diagnostic_rows, integral_summary_rows) if return_diagnostics else rows

    try:
        stress_time_parts: List[np.ndarray] = []
        stress_raw_parts: List[np.ndarray] = []
        for source in stress_sources:
            time_part, raw_part = read_stress_tensor_raw(source, tmin_ns, tmax_ns)
            if raw_part.size == 0:
                continue
            stress_time_parts.append(time_part)
            stress_raw_parts.append(raw_part)
        if stress_raw_parts:
            stress_time_ns = np.concatenate(stress_time_parts, axis=0)
            stress_raw = np.concatenate(stress_raw_parts, axis=0)
            order = np.argsort(stress_time_ns)
            stress_time_ns = stress_time_ns[order]
            stress_raw = stress_raw[order]
        else:
            stress_time_ns = np.array([], dtype=float)
            stress_raw = np.empty((0, 3), dtype=float)
    except Exception as exc:
        emit_missing_rows(fallback_bounds(), f"failed to read stress tensor: {exc}")
        return (rows, diagnostic_rows, integral_summary_rows) if return_diagnostics else rows

    if stress_raw.size == 0 or stress_raw.shape[0] < segment_count:
        emit_missing_rows(fallback_bounds(), f"insufficient stress points for {segment_count} segments")
        return (rows, diagnostic_rows, integral_summary_rows) if return_diagnostics else rows

    r_totals_a = []
    for sg_vals in sg_window_map.values():
        r_cond = safe_float(sg_vals.get("R_cond_A", math.nan))
        w_int = safe_float(sg_vals.get("W_interface_A", math.nan))
        r_total_a = r_cond + 0.5 * w_int
        if np.isfinite(r_total_a) and r_total_a > 0:
            r_totals_a.append(r_total_a)
    if not r_totals_a:
        emit_missing_rows(fallback_bounds(), "missing SG radius/window fit for viscosity")
        return (rows, diagnostic_rows, integral_summary_rows) if return_diagnostics else rows

    mean_r_total_a = float(np.mean(r_totals_a))
    vol_sys = 4.0 / 3.0 * math.pi * (mean_r_total_a * 1e-10) ** 3
    conv = 101325.0 * 1e-30 / vol_sys
    p_xyz = stress_raw * conv

    helper = acf()
    dt_idx, segment_acfs, segment_slices = helper.segment_acfs(p_xyz, segment_count)
    if dt_idx.size < 2 or segment_acfs.size == 0:
        emit_missing_rows(fallback_bounds(), f"failed to compute segment ACFs for {segment_count} segments")
        return (rows, diagnostic_rows, integral_summary_rows) if return_diagnostics else rows

    if stress_time_ns.size > 1:
        stress_dt_ns = float(np.median(np.diff(stress_time_ns)))
    else:
        stress_dt_ns = float(dt_ns)

    vsc = visc(category_dir, np.empty((0, 3)))
    for i, ((start_idx, stop_idx), acf_vals) in enumerate(zip(segment_slices, segment_acfs)):
        segment_start_ns = int(round(float(stress_time_ns[start_idx])))
        segment_end_ns = int(round(float(stress_time_ns[stop_idx - 1] + stress_dt_ns)))
        eta_raw, eta_theo, amp_opt, tau_opt, _, _, eta_diag = vsc.estimate_single_acf(
            dt=dt_idx,
            acf=np.asarray(acf_vals, dtype=float),
            vol=vol_sys,
            dt_unit=float(visc_cfg["dt_unit"]),
            n_point=int(visc_cfg["n_point"]),
            n_tau=int(visc_cfg["n_tau"]),
            T=temperature,
        )
        raw_dt_ns = np.asarray(eta_diag.get("raw_integral_dt", np.array([], dtype=float)), dtype=float) * stress_dt_ns
        raw_integral = np.asarray(eta_diag.get("raw_integral", np.array([], dtype=float)), dtype=float)
        integral_summary = summarize_running_integral(raw_dt_ns, raw_integral)
        note = (
            f"n_stress_points={int(stop_idx - start_idx)}; "
            f"segment_count={segment_count}; mean_r_total_A={mean_r_total_a:.3f}"
        )
        add_long_rows(
            rows,
            temperature,
            category,
            system_name,
            i,
            segment_start_ns,
            segment_end_ns,
            {"eta_GK_Pa_s": eta_raw, "eta_GK_Theo_Pa_s": eta_theo},
            bool(np.isfinite(eta_raw) or np.isfinite(eta_theo)),
            "",
            "",
            stress_path,
            note,
            sampling_unit_family="viscosity_segment",
        )
        if return_diagnostics:
            diagnostic_rows.append({
                "temperature_K": temperature,
                "category": category,
                "system_name": system_name,
                "small_molecule_id": sm_id,
                "segment_index": int(i),
                "segment_start_ns": int(segment_start_ns),
                "segment_end_ns": int(segment_end_ns),
                "segment_duration_ns": int(segment_end_ns - segment_start_ns),
                "n_stress_points": int(stop_idx - start_idx),
                "segment_count_requested": int(segment_count),
                "mean_r_total_A": float(mean_r_total_a),
                "volume_m3": float(vol_sys),
                "eta_GK_Pa_s": float(eta_raw) if np.isfinite(eta_raw) else math.nan,
                "eta_GK_Theo_Pa_s": float(eta_theo) if np.isfinite(eta_theo) else math.nan,
                "acf0": float(eta_diag.get("acf0", math.nan)) if np.isfinite(eta_diag.get("acf0", math.nan)) else math.nan,
                "fit_mae": float(eta_diag.get("fit_mae", math.nan)) if np.isfinite(eta_diag.get("fit_mae", math.nan)) else math.nan,
                "maxwell_mode_count": int(eta_diag.get("maxwell_mode_count", len(amp_opt) if len(amp_opt) else 0)),
                "theoretical_success": bool(eta_diag.get("theoretical_success", np.isfinite(eta_theo) and eta_theo > 0)),
                "time_to_90pct_final_ns": integral_summary["time_to_90pct_final_ns"],
                "late_integral_slope_Pa_s_per_ns": integral_summary["late_integral_slope_Pa_s_per_ns"],
                "late_integral_relative_slope": integral_summary["late_integral_relative_slope"],
                "eta_raw_absmax_Pa_s": integral_summary["eta_raw_absmax_Pa_s"],
                "note": note,
            })
            integral_summary_rows.append({
                "temperature_K": temperature,
                "category": category,
                "system_name": system_name,
                "small_molecule_id": sm_id,
                "segment_index": int(i),
                "segment_start_ns": int(segment_start_ns),
                "segment_end_ns": int(segment_end_ns),
                "segment_duration_ns": int(segment_end_ns - segment_start_ns),
                "eta_raw_final_Pa_s": integral_summary["eta_raw_final_Pa_s"],
                "eta_raw_absmax_Pa_s": integral_summary["eta_raw_absmax_Pa_s"],
                "time_to_90pct_final_ns": integral_summary["time_to_90pct_final_ns"],
                "late_integral_slope_Pa_s_per_ns": integral_summary["late_integral_slope_Pa_s_per_ns"],
                "late_integral_relative_slope": integral_summary["late_integral_relative_slope"],
                "n_running_integral_points": int(raw_integral.size),
            })

    return (rows, diagnostic_rows, integral_summary_rows) if return_diagnostics else rows


def add_long_rows(
    rows: List[Dict[str, object]],
    temperature: int,
    category: str,
    system_name: str,
    window_index: int,
    window_start: int,
    window_end: int,
    observable_map: Dict[str, float],
    fit_ok: bool,
    source_density: str,
    source_pca: str,
    source_cluster: str,
    note: str,
    sampling_unit_family: str = "diffusion_segment",
) -> None:
    """Append one long-format row per observable to ``rows`` (in place).

    Expands ``observable_map`` into the standard long schema (temperature,
    category, system, window indices, observable_key/value, provenance) tagged
    with the given ``sampling_unit_family``.
    """
    sm_id = SMALL_MOLECULE_ID.get(system_name, system_name)
    for observable_key, value in observable_map.items():
        rows.append({
            "temperature_K": temperature,
            "category": category,
            "system_name": system_name,
            "small_molecule_id": sm_id,
            "window_index": window_index,
            "window_start_ns": window_start,
            "window_end_ns": window_end,
            "observable_key": observable_key,
            "value": value,
            "fit_ok": bool(fit_ok),
            "sampling_unit_family": sampling_unit_family,
            "source_density_file": source_density,
            "source_pca_file": source_pca,
            "source_cluster_file": source_cluster,
            "note": note,
        })


def extract_window_scalars_for_system(
    temp_root: str,
    temperature: int,
    category: str,
    system_name: str,
    windows: Sequence[int],
) -> List[Dict[str, object]]:
    """Extract all per-window phase/cluster scalars for one system.

    For each window, fits the SG, Protein, RNA (and SM for DSM/NDSM) RDP
    profiles and reads the SG cluster scalars, emitting the full set of
    Quant_Data phase observables (concentrations, R/W, gammas, deltaG, phi_D,
    N_D, R_g, phi_R, ...) as long-format rows.
    """
    category_dir = os.path.join(temp_root, f"ANALYSIS_{category}")
    rows: List[Dict[str, object]] = []

    for i, window_start in enumerate(windows):
        window_end = window_start + int(windows[1] - windows[0]) if len(windows) > 1 else window_start

        sg_density = resolve_window_file(category_dir, "DensityProfile_SG", system_name, window_start)
        sg_pca = resolve_window_file(category_dir, "PCA_SG", system_name, window_start)
        sg_cluster = resolve_window_file(category_dir, "Cluster_SG", system_name, window_start)
        protein_density = resolve_window_file(category_dir, "DensityProfile_Protein", system_name, window_start)
        protein_pca = resolve_window_file(category_dir, "PCA_Protein", system_name, window_start)
        protein_cluster = resolve_window_file(category_dir, "Cluster_Protein", system_name, window_start)
        rna_density = resolve_window_file(category_dir, "DensityProfile_RNA", system_name, window_start)
        rna_pca = resolve_window_file(category_dir, "PCA_RNA", system_name, window_start)
        rna_cluster = resolve_window_file(category_dir, "Cluster_RNA", system_name, window_start)
        sm_density = resolve_window_file(category_dir, "DensityProfile_SM", system_name, window_start)

        sg_ok, sg_fit, sg_note = fit_rdp_window(sg_density, sg_pca, sg_cluster, temperature, "SG", WINDOW_INIT["SG"])
        add_long_rows(
            rows,
            temperature,
            category,
            system_name,
            i,
            window_start,
            window_end,
            {
                "c_dense_sg_fit": sg_fit.get("c_dense_fit", math.nan),
                "c_dilute_sg_fit": sg_fit.get("c_dilute_fit", math.nan),
                "c_dense_sg_calc": sg_fit.get("c_dense_calc", math.nan),
                "c_dilute_sg_calc": sg_fit.get("c_dilute_calc", math.nan),
                "P_sg": sg_fit.get("P_ratio", math.nan),
                "R_cond_A": sg_fit.get("R_cond_A", math.nan),
                "W_interface_A": sg_fit.get("W_interface_A", math.nan),
                "gamma1_mN_m": sg_fit.get("gamma1_mN_m", math.nan),
                "gamma2_mN_m": sg_fit.get("gamma2_mN_m", math.nan),
                "gamma_ave_mN_m": sg_fit.get("gamma_ave_mN_m", math.nan),
                "deltaG_trans_kJ_mol": sg_fit.get("deltaG_trans_kJ_mol", math.nan),
            },
            sg_ok,
            sg_density,
            sg_pca,
            sg_cluster,
            sg_note,
        )

        prot_ok, prot_fit, prot_note = fit_rdp_window(protein_density, protein_pca, protein_cluster, temperature, "Protein", WINDOW_INIT["Protein"])
        add_long_rows(
            rows,
            temperature,
            category,
            system_name,
            i,
            window_start,
            window_end,
            {
                "gamma1_protein_mN_m": prot_fit.get("gamma1_mN_m", math.nan),
                "gamma2_protein_mN_m": prot_fit.get("gamma2_mN_m", math.nan),
                "gamma_ave_protein_mN_m": prot_fit.get("gamma_ave_mN_m", math.nan),
            },
            prot_ok,
            protein_density,
            protein_pca,
            protein_cluster,
            prot_note,
        )

        rna_ok, rna_fit, rna_note = fit_rdp_window(rna_density, rna_pca, rna_cluster, temperature, "RNA", WINDOW_INIT["RNA"])
        add_long_rows(
            rows,
            temperature,
            category,
            system_name,
            i,
            window_start,
            window_end,
            {
                "gamma1_rna_mN_m": rna_fit.get("gamma1_mN_m", math.nan),
                "gamma2_rna_mN_m": rna_fit.get("gamma2_mN_m", math.nan),
                "gamma_ave_rna_mN_m": rna_fit.get("gamma_ave_mN_m", math.nan),
            },
            rna_ok,
            rna_density,
            rna_pca,
            rna_cluster,
            rna_note,
        )

        if category in {"DSM", "NDSM"}:
            sm_ok, sm_fit, sm_note = fit_rdp_window(sm_density, sg_pca, sg_cluster, temperature, "SM", WINDOW_INIT["SM"])
            add_long_rows(
                rows,
                temperature,
                category,
                system_name,
                i,
                window_start,
                window_end,
                {
                    "c_dense_sm_mg_ml": sm_fit.get("c_dense_fit", math.nan),
                    "c_dilute_sm_mg_ml": sm_fit.get("c_dilute_fit", math.nan),
                    "P_sm": sm_fit.get("P_ratio", math.nan),
                },
                sm_ok,
                sm_density,
                sg_pca,
                sg_cluster,
                sm_note,
            )

        cluster_ok, cluster_vals, cluster_note = extract_cluster_scalars(sg_cluster, sg_fit.get("R_cond_A", math.nan))
        add_long_rows(
            rows,
            temperature,
            category,
            system_name,
            i,
            window_start,
            window_end,
            cluster_vals if cluster_ok else {
                "phi_D": math.nan,
                "N_D": math.nan,
                "R_g_A": math.nan,
                "phi_R": math.nan,
                "chains_largest": math.nan,
                "mass_largest_mg": math.nan,
                "number_external": math.nan,
                "mass_external_mg": math.nan,
                "total_mass_mg": math.nan,
                "total_chain_number": math.nan,
            },
            cluster_ok,
            "",
            "",
            sg_cluster,
            cluster_note,
        )

    return rows


# ---------------------------------------------------------------------------
# Equilibration detection
# ---------------------------------------------------------------------------
# Implements the Chodera (JCTC 2016) automated equilibration detection via
# pymbar.timeseries.detect_equilibration.  The algorithm finds the index t*
# that maximises the effective number of uncorrelated samples:
#     n_eff(t*) = (N - t*) / g(t*)
# where g(t*) is the statistical inefficiency of the post-t* subseries.
# This is the standard recommended approach in Grossfield et al., "Best
# Practices for Quantification of Uncertainty and Sampling Quality in
# Molecular Simulations", Living J. Comp. Mol. Sci. 2019.
# ---------------------------------------------------------------------------

@dataclass
class equilibration_result:
    """Result of automated equilibration detection for one observable."""
    available: bool
    method: str
    t_equil_index: int          # index of first post-equilibration sample
    t_equil_ns: float           # equilibration time in ns
    g_equil: float              # statistical inefficiency of equilibrated portion
    n_eff_equil: float          # effective uncorrelated samples after equilibration
    n_total: int                # total number of windows
    n_discarded: int            # windows discarded as equilibration
    fraction_discarded: float   # n_discarded / n_total
    message: str = ""


def detect_equilibration(values: np.ndarray, window_starts_ns: np.ndarray, dt_ns: int) -> equilibration_result:
    """Detect equilibration using pymbar's detect_equilibration (Chodera JCTC 2016).

    Parameters
    ----------
    values : array of observable values, one per window
    window_starts_ns : corresponding window start times in ns
    dt_ns : window spacing in ns

    Returns
    -------
    ``equilibration_result`` with the detected equilibration index and diagnostics.
    """
    x = np.asarray(values, dtype=float)
    n = x.size
    if n == 0:
        return equilibration_result(False, "empty", 0, 0.0, math.nan, 0.0, 0, 0, 0.0, "no values")
    if n <= 2 or np.nanstd(x) == 0:
        return equilibration_result(True, "degenerate", 0, 0.0, 1.0, float(n), n, 0, 0.0,
                                   "constant or too-short series; no equilibration detected")
    try:
        from pymbar import timeseries
        t_equil, g_equil, n_eff_equil = timeseries.detect_equilibration(x)
        t_equil = int(t_equil)
        if t_equil >= n:
            t_equil = 0
        t_equil_ns = float(window_starts_ns[t_equil]) if t_equil < len(window_starts_ns) else float(t_equil * dt_ns)
        return equilibration_result(
            available=True,
            method="pymbar.detect_equilibration",
            t_equil_index=t_equil,
            t_equil_ns=t_equil_ns,
            g_equil=float(g_equil) if np.isfinite(g_equil) else math.nan,
            n_eff_equil=float(n_eff_equil) if np.isfinite(n_eff_equil) else math.nan,
            n_total=n,
            n_discarded=t_equil,
            fraction_discarded=float(t_equil / n),
        )
    except Exception as exc:
        return equilibration_result(False, "error", 0, 0.0, math.nan, 0.0, n, 0, 0.0, str(exc))


def compute_acf(values: np.ndarray, max_lag: int) -> List[Tuple[int, float]]:
    """Compute the normalised autocorrelation of a 1-D series up to ``max_lag``.

    Returns a list of ``(lag, acf)`` pairs (acf[0]=1) over the finite values.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return []
    if x.size == 1:
        return [(0, 1.0)]
    x_center = x - np.mean(x)
    denom = np.sum(x_center * x_center)
    if denom <= 0:
        return [(lag, 1.0 if lag == 0 else math.nan) for lag in range(min(max_lag, x.size - 1) + 1)]
    out = []
    lag_max = min(max_lag, x.size - 1)
    for lag in range(lag_max + 1):
        numer = np.sum(x_center[: x.size - lag] * x_center[lag:])
        out.append((lag, float(numer / denom)))
    return out


def run_pymbar(values: np.ndarray, conservative: bool) -> pymbar_result:
    """Compute PYMBAR statistical-inefficiency diagnostics for one series.

    Returns a ``pymbar_result`` with the statistical inefficiency ``g``, integrated
    autocorrelation time, effective sample count, decorrelated subsample indices,
    and both uncorrelated and g-corrected SEMs. Degrades gracefully (flagged
    unavailable) if PYMBAR is missing or the series is degenerate.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return pymbar_result(False, "unavailable", math.nan, math.nan, math.nan, [], math.nan, math.nan, math.nan, "no finite values")
    if n == 1 or np.nanstd(x) == 0:
        return pymbar_result(True, "degenerate", 1.0, 0.0, float(n), list(range(n)), float(np.mean(x)), 0.0, 0.0, "constant or singleton series")

    try:
        from pymbar import timeseries
    except Exception as exc:
        return pymbar_result(False, "missing", math.nan, math.nan, math.nan, [], math.nan, math.nan, math.nan, str(exc))

    try:
        g = float(timeseries.statistical_inefficiency(x, fast=False))
        if not np.isfinite(g) or g < 1.0:
            g = 1.0
        idx = list(timeseries.subsample_correlated_data(x, g=g, conservative=conservative))
        uncorr = x[idx] if idx else np.array([], dtype=float)
        mean_uncorr = mean_from_values(uncorr)
        se_uncorr = sem_from_values(uncorr)
        sd_full = std_from_values(x)
        se_g = float(sd_full * math.sqrt(g / n)) if np.isfinite(sd_full) else math.nan
        return pymbar_result(
            True,
            "pymbar.timeseries",
            g,
            0.5 * (g - 1.0),
            float(n / g),
            idx,
            mean_uncorr,
            se_uncorr,
            se_g,
            "",
        )
    except Exception as exc:
        return pymbar_result(False, "error", math.nan, math.nan, math.nan, [], math.nan, math.nan, math.nan, str(exc))


def compute_superblock_tables(
    values: np.ndarray,
    window_starts: np.ndarray,
    max_block_size: int,
    min_superblocks: int,
    plateau_fraction: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Optional[int], Optional[float]]:
    """Block-averaging analysis following Flyvbjerg & Petersen (JCP 1989).

    The standard error of the mean (SEM) for a correlated time series is
    underestimated by naive SEM = std/sqrt(N).  The Flyvbjerg-Petersen method
    groups consecutive samples into blocks of increasing size b and computes
    the block-averaged SEM at each b.  As b grows past the integrated
    autocorrelation time tau_int, block means become statistically
    independent, and SEM(b) plateaus at the correct value.

    This implementation:
      1. Sweeps block sizes b = 1 .. max_block_size.
      2. For each b, tests all offsets (0 .. b-1) to reduce bias from the
         starting index.  The median SEM across eligible offsets (those with
         at least min_superblocks blocks) is the representative SEM(b).
      3. Builds a monotonic running-maximum envelope over the eligible SEM(b)
         values.
      4. Selects the smallest b whose envelope reaches plateau_fraction (default
         95%) of the asymptotic plateau value.  This is the recommended block
         size.

    A complementary pymbar statistical-inefficiency estimate (g) is computed
    separately, and the final per-observable recommendation is:
        recommended_block = max(superblock_plateau, ceil(g))
    ensuring the block size is at least as large as what either method demands.

    The system-level block size for downstream averaging is then the maximum
    over all non-viscosity observables (conservative), guaranteeing that even
    the slowest-decorrelating observable is properly decorrelated.
    """
    offset_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    n = len(values)
    if n == 0:
        return offset_rows, summary_rows, None, math.nan

    max_b = max(1, min(max_block_size, n))
    for b in range(1, max_b + 1):
        se_vals = []
        eligible_vals = []
        for offset in range(b):
            n_superblocks = (n - offset) // b
            if n_superblocks < 2:
                continue
            used_n = n_superblocks * b
            used_values = values[offset : offset + used_n]
            used_windows = window_starts[offset : offset + used_n]
            block_means = used_values.reshape(n_superblocks, b).mean(axis=1)
            row = {
                "superblock_size_windows": b,
                "offset_windows": offset,
                "n_raw_windows": n,
                "n_used_windows": int(used_n),
                "n_superblocks": int(n_superblocks),
                "n_dropped_tail_windows": int(n - used_n),
                "first_window_start_ns": int(used_windows[0]) if used_windows.size else math.nan,
                "last_window_start_ns": int(used_windows[-1]) if used_windows.size else math.nan,
                "mean_used": mean_from_values(used_values),
                "mean_superblock_means": mean_from_values(block_means),
                "sd_superblock_means": std_from_values(block_means),
                "se_superblock": sem_from_values(block_means),
                "eligible_for_plateau": bool(n_superblocks >= min_superblocks),
            }
            offset_rows.append(row)
            if np.isfinite(row["se_superblock"]):
                se_vals.append(float(row["se_superblock"]))
                if n_superblocks >= min_superblocks:
                    eligible_vals.append(float(row["se_superblock"]))

        if not se_vals:
            continue

        use_vals = eligible_vals if eligible_vals else se_vals
        summary_rows.append({
            "superblock_size_windows": b,
            "n_offsets": len(se_vals),
            "n_offsets_eligible": len(eligible_vals),
            "se_superblock_mean": float(np.mean(se_vals)),
            "se_superblock_median": float(np.median(se_vals)),
            "se_superblock_min": float(np.min(se_vals)),
            "se_superblock_max": float(np.max(se_vals)),
            "se_for_recommendation": float(np.median(use_vals)),
            "eligible_for_plateau": bool(len(eligible_vals) > 0),
        })

    recommended_b = None
    plateau_value = math.nan
    eligible = [row for row in summary_rows if row["eligible_for_plateau"] and np.isfinite(row["se_for_recommendation"])]
    if eligible:
        envelope = []
        running = -np.inf
        for row in eligible:
            running = max(running, row["se_for_recommendation"])
            envelope.append(running)
        plateau_value = max(envelope)
        target = plateau_fraction * plateau_value
        for row, env in zip(eligible, envelope):
            if env >= target:
                recommended_b = int(row["superblock_size_windows"])
                break
        if recommended_b is None:
            recommended_b = int(eligible[-1]["superblock_size_windows"])
    elif summary_rows:
        recommended_b = int(summary_rows[-1]["superblock_size_windows"])

    return offset_rows, summary_rows, recommended_b, plateau_value


def block_layout(n_windows: int, block_size: int, offset: int = 0) -> List[Tuple[int, int, int]]:
    """Enumerate non-overlapping blocks as ``(block_id, start_idx, end_idx)``.

    Starting at ``offset``, returns contiguous ``block_size``-window blocks
    covering ``n_windows`` (the tail remainder is dropped).
    """
    out = []
    if block_size <= 0:
        return out
    n_blocks = (n_windows - offset) // block_size
    for block_id in range(n_blocks):
        start_idx = offset + block_id * block_size
        end_idx = start_idx + block_size - 1
        out.append((block_id, start_idx, end_idx))
    return out


def choose_family_recommended_block(per_observable_df: pd.DataFrame, observable_keys: Sequence[str], rule_label: str) -> Tuple[Optional[int], str]:
    """Return the conservative (max) recommended block size over a family.

    Selects the maximum ``recommended_block_size_windows`` among the given
    observable keys (or all observables if none match) and returns
    ``(block_size, rule_label)``.
    """
    if per_observable_df.empty:
        return None, "no observables"
    family = per_observable_df[per_observable_df["observable_key"].isin(observable_keys)].copy()
    valid = family if not family.empty else per_observable_df.copy()
    valid = valid[np.isfinite(valid["recommended_block_size_windows"])].copy()
    if valid.empty:
        return None, "no valid recommendations"
    block_size = int(valid["recommended_block_size_windows"].max())
    return block_size, rule_label


def choose_system_recommended_block(per_observable_df: pd.DataFrame) -> Tuple[Optional[int], str]:
    """Recommend the system block size: max over core (non-viscosity) observables."""
    return choose_family_recommended_block(
        per_observable_df,
        CORE_RECOMMENDATION_OBSERVABLES,
        "max_over_non_viscosity_quant_data_observables",
    )


def choose_viscosity_recommended_block(per_observable_df: pd.DataFrame) -> Tuple[Optional[int], str]:
    """Recommend the viscosity-segment block size: max over viscosity observables."""
    return choose_family_recommended_block(
        per_observable_df,
        VISCOSITY_OBSERVABLES,
        "max_over_viscosity_observables",
    )


def main() -> None:
    """Run correlation diagnostics for one temperature and write all outputs.

    Extracts per-window phase/cluster/spatial scalars and per-segment viscosity
    for every SG/DSM/NDSM system, runs the superblock + pymbar + Chodera
    diagnostics per observable, derives correlation-corrected means/SEMs and the
    recommended system/viscosity block layouts, and writes the diagnostics CSVs
    and figures. With ``--plot-only`` it only verifies existing outputs.
    """
    parser = argparse.ArgumentParser(description="Correlation diagnostics over 39 post-equilibration 50 ns windows")
    parser.add_argument("--path", required=True, help="Path to TEMP_XXX directory")
    parser.add_argument("--folder", default="CLASSIFY", help="Output folder prefix")
    parser.add_argument("--temp", type=int, required=True, help="Temperature in K")
    parser.add_argument("--tmin", type=int, required=True, help="Start of post-equilibration windowing (ns)")
    parser.add_argument("--dt", type=int, required=True, help="Window size / spacing (ns)")
    parser.add_argument("--tmax", type=int, required=True, help="End of analysis window (ns)")
    parser.add_argument("--use-lists", action="store_true", help="Load dsm_list.txt and ndsm_list.txt from the analysis root if present")
    parser.add_argument("--max-superblock", type=int, default=9, help="Maximum number of 50 ns windows per tested superblock")
    parser.add_argument("--min-superblocks", type=int, default=4, help="Minimum number of superblocks required for a block size to count toward recommendation")
    parser.add_argument("--plateau-fraction", type=float, default=0.95, help="Choose the smallest block size whose SEM reaches this fraction of the plateau SEM")
    parser.add_argument("--winsorize-pct", type=float, default=5.0, help="Winsorize per-window values at this percentile (0 to disable). Caps extreme per-window outliers before block averaging.")
    parser.add_argument("--acf-max-lag", type=int, default=15, help="Maximum ACF lag to export and plot")
    parser.add_argument("--pymbar-conservative", action="store_true", help="Use conservative=True in pymbar.timeseries.subsample_correlated_data")
    parser.add_argument("--equil-soft-max-ns", type=float, default=DEFAULT_EQUILIBRATION_SOFT_MAX_NS, help="Soft upper bound for LLPS equilibration audit classification")
    parser.add_argument("--equil-hard-fraction", type=float, default=DEFAULT_EQUILIBRATION_HARD_FRACTION, help="Fraction discarded above which equilibration advice is classified as hard")
    parser.add_argument("--equil-sensitivity-cutoffs", default=",".join(str(x) for x in DEFAULT_EQUILIBRATION_SENSITIVITY_CUTOFFS), help="Comma-separated fixed cutoff scan (ns) for equilibration sensitivity reporting")
    parser.add_argument("--plot-all-observables", action="store_true", help="Plot every supported observable instead of the default key subset")
    parser.add_argument("--plot-only", action="store_true", help="No-op for pipeline plot-only mode; exits after verifying diagnostics outputs exist")
    parser.add_argument("--out-root", default=None, help="Optional dedicated diagnostics output root")
    parser.add_argument("--skip-viscosity", action="store_true", help="Skip viscosity observable extraction")
    parser.add_argument("--visc-segments", type=int, default=DEFAULT_VISCOSITY_CONFIG["segments"], help="Number of contiguous stress segments for viscosity diagnostics")
    parser.add_argument("--visc-dt-unit", type=float, default=DEFAULT_VISCOSITY_CONFIG["dt_unit"], help="Stress sampling interval in seconds for viscosity")
    parser.add_argument("--visc-n-point", type=int, default=DEFAULT_VISCOSITY_CONFIG["n_point"], help="Number of log-spaced fit points for viscosity")
    parser.add_argument("--visc-n-tau", type=int, default=DEFAULT_VISCOSITY_CONFIG["n_tau"], help="Maximum number of Maxwell modes for viscosity fitting")
    args = parser.parse_args()
    sensitivity_cutoffs = sorted(set(parse_cutoff_list(args.equil_sensitivity_cutoffs) + [int(args.tmin)]))

    temp_root = os.path.abspath(args.path)
    analysis_root = os.path.join(temp_root, f"{args.folder}_{args.temp}_{args.dt}_{args.tmin}_{args.tmax}")
    diag_root = os.path.abspath(args.out_root) if args.out_root else os.path.join(
        temp_root,
        f"CORRELATION_{args.folder}_{args.temp}_{args.dt}_{args.tmin}_{args.tmax}",
    )
    results_root = os.path.join(diag_root, "RESULTS", "CORRELATION_DIAGNOSTICS")
    figures_root = os.path.join(diag_root, "FIGURES", "CORRELATION_DIAGNOSTICS")
    ensure_dir(results_root)
    ensure_dir(figures_root)
    ensure_dir(analysis_root)
    if args.plot_only:
        required = os.path.join(results_root, "RECOMMENDED_SYSTEM_BLOCK_LAYOUT.csv")
        if not os.path.isfile(required):
            raise SystemExit(f"Diagnostics output not found: {required}. Run full diagnostics first.")
        print(f"[plot-only] BLOCK_CORRELATION_DIAGNOSTICS has no standalone plot-only redraw path; existing diagnostics verified: {results_root}")
        return

    dsm_list, ndsm_list = load_sm_lists(analysis_root, args.use_lists)
    windows = list(range(args.tmin, args.tmax, args.dt))

    visc_cfg = dict(DEFAULT_VISCOSITY_CONFIG)
    visc_cfg.update({
        "segments": args.visc_segments,
        "dt_unit": args.visc_dt_unit,
        "n_point": args.visc_n_point,
        "n_tau": args.visc_n_tau,
    })

    systems_by_category: Dict[str, List[str]] = {}
    if os.path.isdir(os.path.join(temp_root, "ANALYSIS_SG")):
        systems_by_category["SG"] = ["sg_X"]
    if os.path.isdir(os.path.join(temp_root, "ANALYSIS_DSM")) and dsm_list:
        systems_by_category["DSM"] = dsm_list
    if os.path.isdir(os.path.join(temp_root, "ANALYSIS_NDSM")) and ndsm_list:
        systems_by_category["NDSM"] = ndsm_list

    if not systems_by_category:
        raise SystemExit("No ANALYSIS_{SG,DSM,NDSM} directories found under the supplied TEMP directory.")

    print(f"Writing correlation diagnostics to {diag_root}")
    print(f"Using windows: {windows[0]}..{windows[-1]} ns ({len(windows)} total windows)")

    window_rows: List[Dict[str, object]] = []
    viscosity_rows: List[Dict[str, object]] = []
    viscosity_segment_diag_rows: List[Dict[str, object]] = []
    viscosity_integral_summary_rows: List[Dict[str, object]] = []
    for category, systems in systems_by_category.items():
        print(f"Processing category {category} ({len(systems)} systems)")
        for system_name in systems:
            print(f"  extracting per-window scalars: {system_name}")
            phase_rows = extract_window_scalars_for_system(temp_root, args.temp, category, system_name, windows)
            window_rows.extend(phase_rows)
            sg_window_map = build_window_observable_map(phase_rows)
            print(f"  extracting per-window per-species Rg/Rh/Occ/r-over-R: {system_name}")
            spatial_rows = extract_per_species_spatial_scalars_for_system(
                temp_root,
                args.temp,
                category,
                system_name,
                windows,
                args.dt,
                args.tmin,
                args.tmax,
            )
            window_rows.extend(spatial_rows)
            if not args.skip_viscosity:
                print(f"  extracting segmentwise viscosity: {system_name}")
                visc_rows, visc_diag_rows, visc_integral_rows = extract_viscosity_scalars_for_system(
                    temp_root,
                    args.temp,
                    category,
                    system_name,
                    windows,
                    args.dt,
                    args.tmin,
                    args.tmax,
                    sg_window_map,
                    visc_cfg,
                    return_diagnostics=True,
                )
                dse_rows = derive_stokes_einstein_segment_scalars(
                    args.temp,
                    category,
                    system_name,
                    spatial_rows,
                    visc_rows,
                )
                viscosity_rows.extend(visc_rows)
                viscosity_rows.extend(dse_rows)
                viscosity_segment_diag_rows.extend(visc_diag_rows)
                viscosity_integral_summary_rows.extend(visc_integral_rows)

    window_df = pd.DataFrame(window_rows)
    if not window_df.empty:
        window_df.sort_values(["category", "system_name", "observable_key", "window_start_ns"], inplace=True)
    window_df.to_csv(os.path.join(results_root, "WINDOW_SCALARS.csv"), index=False)

    viscosity_df = pd.DataFrame(viscosity_rows)
    if not viscosity_df.empty:
        viscosity_df.sort_values(["category", "system_name", "observable_key", "window_start_ns"], inplace=True)
    viscosity_df.to_csv(os.path.join(results_root, "VISCOSITY_SEGMENT_SCALARS.csv"), index=False)

    viscosity_segment_diag_df = pd.DataFrame(viscosity_segment_diag_rows)
    if not viscosity_segment_diag_df.empty:
        viscosity_segment_diag_df.sort_values(["category", "system_name", "segment_index"], inplace=True)
    viscosity_segment_diag_df.to_csv(os.path.join(results_root, "VISCOSITY_SEGMENT_DIAGNOSTICS.csv"), index=False)

    viscosity_integral_summary_df = pd.DataFrame(viscosity_integral_summary_rows)
    if not viscosity_integral_summary_df.empty:
        viscosity_integral_summary_df.sort_values(["category", "system_name", "segment_index"], inplace=True)
    viscosity_integral_summary_df.to_csv(os.path.join(results_root, "VISCOSITY_RUNNING_INTEGRAL_SUMMARY.csv"), index=False)

    scalar_frames = [df for df in [window_df, viscosity_df] if not df.empty]
    if scalar_frames:
        correlation_long_df = pd.concat(scalar_frames, ignore_index=True)
        correlation_long_df.sort_values(
            ["category", "system_name", "sampling_unit_family", "observable_key", "window_start_ns"],
            inplace=True,
        )
    else:
        correlation_long_df = pd.DataFrame(
            columns=[
                "temperature_K",
                "category",
                "system_name",
                "small_molecule_id",
                "window_index",
                "window_start_ns",
                "window_end_ns",
                "observable_key",
                "value",
                "fit_ok",
                "sampling_unit_family",
                "source_density_file",
                "source_pca_file",
                "source_cluster_file",
                "note",
            ]
        )
    correlation_long_df.to_csv(os.path.join(results_root, "CORRELATION_QUANT_DATA_LONG.csv"), index=False)

    plot_observables = None if args.plot_all_observables else set(DEFAULT_PLOT_OBSERVABLES)
    superblock_offset_rows: List[Dict[str, object]] = []
    superblock_summary_rows: List[Dict[str, object]] = []
    acf_rows_all: List[Dict[str, object]] = []
    pymbar_rows: List[Dict[str, object]] = []
    pymbar_subsamples_rows: List[Dict[str, object]] = []
    observable_summary_rows: List[Dict[str, object]] = []
    recommended_block_rows: List[Dict[str, object]] = []
    recommended_scalar_rows: List[Dict[str, object]] = []
    inference_offset_rows: List[Dict[str, object]] = []
    inference_block_rows: List[Dict[str, object]] = []
    inference_summary_rows: List[Dict[str, object]] = []
    equilibration_rows: List[Dict[str, object]] = []
    equilibration_audit_rows: List[Dict[str, object]] = []
    equilibration_sensitivity_rows: List[Dict[str, object]] = []

    group_cols = ["temperature_K", "category", "system_name", "small_molecule_id", "observable_key", "sampling_unit_family"]
    for keys, group in correlation_long_df.groupby(group_cols, dropna=False):
        temperature, category, system_name, small_molecule_id, observable_key, sampling_unit_family = keys
        group = group.sort_values(["window_start_ns", "window_index"])
        finite = group[np.isfinite(group["value"])].copy()
        values_raw = finite["value"].to_numpy(dtype=float)
        values = winsorize_values(values_raw, args.winsorize_pct)
        unit_starts = finite["window_start_ns"].to_numpy(dtype=int)
        unit_ends = finite["window_end_ns"].to_numpy(dtype=int)
        if unit_ends.size > 0:
            unit_durations = unit_ends - unit_starts
            unit_durations = unit_durations[unit_durations > 0]
            unit_span_ns = int(round(float(np.median(unit_durations)))) if unit_durations.size else int(args.dt)
        else:
            unit_span_ns = int(args.dt)

        if values.size == 0:
            pymbar_res = run_pymbar(values, args.pymbar_conservative)
            pymbar_rows.append({
                "temperature_K": temperature,
                "category": category,
                "system_name": system_name,
                "small_molecule_id": small_molecule_id,
                "observable_key": observable_key,
                "sampling_unit_family": sampling_unit_family,
                "n_raw_windows": int(group.shape[0]),
                "n_finite_windows": 0,
                "n_raw_units": int(group.shape[0]),
                "n_finite_units": 0,
                "mean_full": math.nan,
                "sd_full": math.nan,
                "g": pymbar_res.g,
                "tau_int_windows": pymbar_res.tau_int_windows,
                "n_eff": pymbar_res.n_eff,
                "n_uncorrelated": len(pymbar_res.indices),
                "mean_uncorrelated": pymbar_res.mean_uncorrelated,
                "se_uncorrelated": pymbar_res.se_uncorrelated,
                "se_g_corrected": pymbar_res.se_g_corrected,
                "pymbar_available": pymbar_res.available,
                "pymbar_method": pymbar_res.method,
                "message": pymbar_res.message,
            })
            observable_summary_rows.append({
                "temperature_K": temperature,
                "category": category,
                "system_name": system_name,
                "small_molecule_id": small_molecule_id,
                "observable_key": observable_key,
                "sampling_unit_family": sampling_unit_family,
                "n_raw_windows": int(group.shape[0]),
                "n_finite_windows": 0,
                "n_raw_units": int(group.shape[0]),
                "n_finite_units": 0,
                "mean_full": math.nan,
                "se_superblock_b1": math.nan,
                "se_superblock_b2": math.nan,
                "se_superblock_b3": math.nan,
                "se_superblock_b4": math.nan,
                "se_superblock_b5": math.nan,
                "se_pymbar_uncorrelated": pymbar_res.se_uncorrelated,
                "se_pymbar_g_corrected": pymbar_res.se_g_corrected,
                "g": pymbar_res.g,
                "tau_int_windows": pymbar_res.tau_int_windows,
                "tau_int_ns": math.nan,
                "n_eff_pymbar": pymbar_res.n_eff,
                "recommended_block_superblock": math.nan,
                "recommended_block_pymbar": math.nan,
                "recommended_block_size_windows": math.nan,
                "recommended_block_size_ns": math.nan,
                "n_recommended_blocks": 0,
                "plateau_sem": math.nan,
                "correlation_flag": "no_finite_values",
                "t_equil_index": 0,
                "t_equil_ns": math.nan,
                "equil_n_discarded": 0,
                "equil_fraction_discarded": math.nan,
                "equil_g": math.nan,
                "equil_n_eff": math.nan,
                "control_equilibration_observable": bool(observable_key in EQUILIBRATION_CONTROL_OBSERVABLES),
            })
            equilibration_rows.append({
                "temperature_K": temperature,
                "category": category,
                "system_name": system_name,
                "small_molecule_id": small_molecule_id,
                "observable_key": observable_key,
                "sampling_unit_family": sampling_unit_family,
                "n_total_windows": 0,
                "t_equil_index": 0,
                "t_equil_ns": math.nan,
                "n_discarded": 0,
                "fraction_discarded": math.nan,
                "g_equil": math.nan,
                "n_eff_equil": math.nan,
                "equil_detection_available": False,
                "equil_detection_method": "no_finite_values",
                "equil_detection_message": "no finite values",
            })
            equilibration_audit_rows.append({
                "temperature_K": temperature,
                "category": category,
                "system_name": system_name,
                "small_molecule_id": small_molecule_id,
                "observable_key": observable_key,
                "sampling_unit_family": sampling_unit_family,
                "control_equilibration_observable": bool(observable_key in EQUILIBRATION_CONTROL_OBSERVABLES),
                "default_tmin_ns": float(args.tmin),
                "soft_max_ns": float(args.equil_soft_max_ns),
                "hard_fraction": float(args.equil_hard_fraction),
                "t_equil_ns": math.nan,
                "fraction_discarded": math.nan,
                "audit_flag": "indeterminate",
            })
            continue

        acf_vals = compute_acf(values, args.acf_max_lag)
        for lag, acf_val in acf_vals:
            acf_rows_all.append({
                "temperature_K": temperature,
                "category": category,
                "system_name": system_name,
                "small_molecule_id": small_molecule_id,
                "observable_key": observable_key,
                "sampling_unit_family": sampling_unit_family,
                "lag_windows": lag,
                "acf": acf_val,
            })

        # Chodera equilibration detection (JCTC 2016)
        equil_result = detect_equilibration(values, unit_starts, args.dt)
        equilibration_rows.append({
            "temperature_K": temperature,
            "category": category,
            "system_name": system_name,
            "small_molecule_id": small_molecule_id,
            "observable_key": observable_key,
            "sampling_unit_family": sampling_unit_family,
            "n_total_windows": equil_result.n_total,
            "t_equil_index": equil_result.t_equil_index,
            "t_equil_ns": equil_result.t_equil_ns,
            "n_discarded": equil_result.n_discarded,
            "fraction_discarded": equil_result.fraction_discarded,
            "g_equil": equil_result.g_equil,
            "n_eff_equil": equil_result.n_eff_equil,
            "equil_detection_available": equil_result.available,
            "equil_detection_method": equil_result.method,
            "equil_detection_message": equil_result.message,
        })
        equilibration_audit_rows.append({
            "temperature_K": temperature,
            "category": category,
            "system_name": system_name,
            "small_molecule_id": small_molecule_id,
            "observable_key": observable_key,
            "sampling_unit_family": sampling_unit_family,
            "control_equilibration_observable": bool(observable_key in EQUILIBRATION_CONTROL_OBSERVABLES),
            "default_tmin_ns": float(args.tmin),
            "soft_max_ns": float(args.equil_soft_max_ns),
            "hard_fraction": float(args.equil_hard_fraction),
            "t_equil_ns": equil_result.t_equil_ns,
            "fraction_discarded": equil_result.fraction_discarded,
            "audit_flag": classify_equilibration_audit(
                equil_result.t_equil_ns,
                equil_result.fraction_discarded,
                default_tmin_ns=float(args.tmin),
                soft_max_ns=float(args.equil_soft_max_ns),
                hard_fraction=float(args.equil_hard_fraction),
            ),
        })

        offset_rows, summary_rows, b_super, plateau_sem = compute_superblock_tables(
            values,
            unit_starts,
            max_block_size=args.max_superblock,
            min_superblocks=args.min_superblocks,
            plateau_fraction=args.plateau_fraction,
        )
        for row in offset_rows:
            row.update({
                "temperature_K": temperature,
                "category": category,
                "system_name": system_name,
                "small_molecule_id": small_molecule_id,
                "observable_key": observable_key,
                "sampling_unit_family": sampling_unit_family,
            })
        for row in summary_rows:
            row.update({
                "temperature_K": temperature,
                "category": category,
                "system_name": system_name,
                "small_molecule_id": small_molecule_id,
                "observable_key": observable_key,
                "sampling_unit_family": sampling_unit_family,
            })
        superblock_offset_rows.extend(offset_rows)
        superblock_summary_rows.extend(summary_rows)

        pymbar_res = run_pymbar(values, args.pymbar_conservative)
        pymbar_rows.append({
            "temperature_K": temperature,
            "category": category,
            "system_name": system_name,
            "small_molecule_id": small_molecule_id,
            "observable_key": observable_key,
            "sampling_unit_family": sampling_unit_family,
            "n_raw_windows": int(group.shape[0]),
            "n_finite_windows": int(finite.shape[0]),
            "n_raw_units": int(group.shape[0]),
            "n_finite_units": int(finite.shape[0]),
            "mean_full": mean_from_values(values),
            "sd_full": std_from_values(values),
            "g": pymbar_res.g,
            "tau_int_windows": pymbar_res.tau_int_windows,
            "n_eff": pymbar_res.n_eff,
            "n_uncorrelated": len(pymbar_res.indices),
            "mean_uncorrelated": pymbar_res.mean_uncorrelated,
            "se_uncorrelated": pymbar_res.se_uncorrelated,
            "se_g_corrected": pymbar_res.se_g_corrected,
            "pymbar_available": pymbar_res.available,
            "pymbar_method": pymbar_res.method,
            "message": pymbar_res.message,
        })
        for raw_idx in pymbar_res.indices:
            if raw_idx < len(unit_starts):
                pymbar_subsamples_rows.append({
                    "temperature_K": temperature,
                    "category": category,
                    "system_name": system_name,
                    "small_molecule_id": small_molecule_id,
                    "observable_key": observable_key,
                    "sampling_unit_family": sampling_unit_family,
                    "raw_window_index": int(raw_idx),
                    "raw_unit_index": int(raw_idx),
                    "window_start_ns": int(unit_starts[raw_idx]),
                    "value": float(values[raw_idx]),
                    "conservative": bool(args.pymbar_conservative),
                })

        b_pymbar = int(max(1, math.ceil(pymbar_res.g))) if (pymbar_res.available and np.isfinite(pymbar_res.g)) else 1
        valid_max_b = max([int(r["superblock_size_windows"]) for r in summary_rows], default=1)
        recommended_b = max(b_super or 1, b_pymbar)
        recommended_b = min(recommended_b, valid_max_b)
        if recommended_b == 1 and (b_super or 1) == 1 and (not pymbar_res.available or b_pymbar == 1):
            correlation_flag = "roughly_uncorrelated_at_native_unit"
        elif recommended_b <= 2:
            correlation_flag = "mild_correlation"
        else:
            correlation_flag = "strong_correlation"

        b_map = {int(r["superblock_size_windows"]): r for r in summary_rows}
        # Compute tau_int in physical units (ns) from pymbar's statistical inefficiency g
        # tau_int = 0.5 * (g - 1) in units of the sampling interval;  tau_int_ns = tau_int * dt
        tau_int_ns = pymbar_res.tau_int_windows * unit_span_ns if np.isfinite(pymbar_res.tau_int_windows) else math.nan
        n_recommended_blocks = len(values) // recommended_b if recommended_b > 0 else 0
        observable_summary_rows.append({
            "temperature_K": temperature,
            "category": category,
            "system_name": system_name,
            "small_molecule_id": small_molecule_id,
            "observable_key": observable_key,
            "sampling_unit_family": sampling_unit_family,
            "n_raw_windows": int(group.shape[0]),
            "n_finite_windows": int(finite.shape[0]),
            "n_raw_units": int(group.shape[0]),
            "n_finite_units": int(finite.shape[0]),
            "mean_full": mean_from_values(values),
            "se_superblock_b1": b_map.get(1, {}).get("se_superblock_median", math.nan),
            "se_superblock_b2": b_map.get(2, {}).get("se_superblock_median", math.nan),
            "se_superblock_b3": b_map.get(3, {}).get("se_superblock_median", math.nan),
            "se_superblock_b4": b_map.get(4, {}).get("se_superblock_median", math.nan),
            "se_superblock_b5": b_map.get(5, {}).get("se_superblock_median", math.nan),
            "se_pymbar_uncorrelated": pymbar_res.se_uncorrelated,
            "se_pymbar_g_corrected": pymbar_res.se_g_corrected,
            "g": pymbar_res.g,
            "tau_int_windows": pymbar_res.tau_int_windows,
            "tau_int_ns": tau_int_ns,
            "n_eff_pymbar": pymbar_res.n_eff,
            "recommended_block_superblock": b_super,
            "recommended_block_pymbar": b_pymbar,
            "recommended_block_size_windows": recommended_b,
            "recommended_block_size_ns": recommended_b * unit_span_ns,
            "n_recommended_blocks": n_recommended_blocks,
            "plateau_sem": plateau_sem,
            "correlation_flag": correlation_flag,
            "t_equil_index": equil_result.t_equil_index,
            "t_equil_ns": equil_result.t_equil_ns,
            "equil_n_discarded": equil_result.n_discarded,
            "equil_fraction_discarded": equil_result.fraction_discarded,
            "equil_g": equil_result.g_equil,
            "equil_n_eff": equil_result.n_eff_equil,
            "control_equilibration_observable": bool(observable_key in EQUILIBRATION_CONTROL_OBSERVABLES),
        })

        if values.size >= recommended_b and recommended_b > 0:
            layout = block_layout(len(unit_starts), recommended_b, offset=0)
            block_means = []
            for block_id, start_idx, end_idx in layout:
                vals = values[start_idx : end_idx + 1]
                win = unit_starts[start_idx : end_idx + 1]
                win_ends = unit_ends[start_idx : end_idx + 1]
                block_mean = mean_from_values(vals)
                block_means.append(block_mean)
                block_duration_ns = int(win_ends[-1] - win[0]) if len(win_ends) else int(recommended_b * unit_span_ns)
                recommended_scalar_rows.append({
                    "temperature_K": temperature,
                    "category": category,
                    "system_name": system_name,
                    "small_molecule_id": small_molecule_id,
                    "observable_key": observable_key,
                    "sampling_unit_family": sampling_unit_family,
                    "recommended_block_size_windows": recommended_b,
                    "recommended_block_size_ns": recommended_b * unit_span_ns,
                    "block_duration_ns": block_duration_ns,
                    "block_id": block_id,
                    "block_start_window_index": int(start_idx),
                    "block_end_window_index": int(end_idx),
                    "block_start_ns": int(win[0]),
                    "block_end_ns": int(win_ends[-1]),
                    "n_windows_in_block": int(len(win)),
                    "window_members": ";".join(str(x) for x in win),
                    "block_mean": block_mean,
                })
            recommended_block_rows.append({
                "temperature_K": temperature,
                "category": category,
                "system_name": system_name,
                "small_molecule_id": small_molecule_id,
                "observable_key": observable_key,
                "sampling_unit_family": sampling_unit_family,
                "recommended_block_size_windows": recommended_b,
                "recommended_block_size_ns": recommended_b * unit_span_ns,
                "n_recommended_blocks": len(layout),
                "recommended_mean": mean_from_values(np.asarray(block_means, dtype=float)),
                "recommended_se": sem_from_values(np.asarray(block_means, dtype=float)),
                "recommended_method": "offset0_block_layout_compatibility_only",
                "correlation_flag": correlation_flag,
            })

        if values.size >= recommended_b and recommended_b > 0:
            inf_offset_rows, inf_block_rows, inf_summary = compute_all_offset_batch_estimator(
                values,
                unit_starts,
                unit_ends,
                recommended_b,
                unit_span_ns,
            )
            for row in inf_offset_rows:
                row.update({
                    "temperature_K": temperature,
                    "category": category,
                    "system_name": system_name,
                    "small_molecule_id": small_molecule_id,
                    "observable_key": observable_key,
                    "sampling_unit_family": sampling_unit_family,
                    "corrected_block_size_units": int(recommended_b),
                    "corrected_block_size_ns": int(recommended_b * unit_span_ns),
                })
            for row in inf_block_rows:
                row.update({
                    "temperature_K": temperature,
                    "category": category,
                    "system_name": system_name,
                    "small_molecule_id": small_molecule_id,
                    "observable_key": observable_key,
                    "sampling_unit_family": sampling_unit_family,
                    "corrected_block_size_units": int(recommended_b),
                    "corrected_block_size_ns": int(recommended_b * unit_span_ns),
                })
            inference_offset_rows.extend(inf_offset_rows)
            inference_block_rows.extend(inf_block_rows)
            inference_summary_rows.append({
                "temperature_K": temperature,
                "category": category,
                "system_name": system_name,
                "small_molecule_id": small_molecule_id,
                "observable_key": observable_key,
                "sampling_unit_family": sampling_unit_family,
                "corrected_block_size_units": int(recommended_b),
                "corrected_block_size_ns": int(recommended_b * unit_span_ns),
                "correlation_flag": correlation_flag,
                "central_value_provenance": f"full_series_winsorized_{args.winsorize_pct:.0f}pct_mean" if args.winsorize_pct > 0 else "full_series_mean_from_native_sampling_units",
                "uncertainty_provenance": "all_offset_nonoverlapping_batch_means_median_sem",
                "uncertainty_scope": STATISTICAL_POLICY["uncertainty_scope"],
                "n_replicates": STATISTICAL_POLICY["n_replicates"],
                **inf_summary,
            })

        if sampling_unit_family == "diffusion_segment" and observable_key in EQUILIBRATION_SENSITIVITY_OBSERVABLES:
            for cutoff_ns in sensitivity_cutoffs:
                sens = summarize_sensitivity_cutoff(
                    values,
                    unit_starts,
                    unit_ends,
                    unit_span_ns,
                    cutoff_ns,
                    args,
                )
                equilibration_sensitivity_rows.append({
                    "temperature_K": temperature,
                    "category": category,
                    "system_name": system_name,
                    "small_molecule_id": small_molecule_id,
                    "observable_key": observable_key,
                    "sampling_unit_family": sampling_unit_family,
                    **sens,
                })

    superblock_offset_df = pd.DataFrame(superblock_offset_rows)
    superblock_summary_df = pd.DataFrame(superblock_summary_rows)
    acf_df = pd.DataFrame(acf_rows_all)
    pymbar_df = pd.DataFrame(pymbar_rows)
    pymbar_sub_df = pd.DataFrame(pymbar_subsamples_rows)
    observable_summary_df = pd.DataFrame(observable_summary_rows)
    recommended_block_df = pd.DataFrame(recommended_block_rows)
    recommended_scalar_df = pd.DataFrame(recommended_scalar_rows)
    inference_offset_df = pd.DataFrame(inference_offset_rows)
    inference_block_df = pd.DataFrame(inference_block_rows)
    correlation_summary_df = pd.DataFrame(inference_summary_rows)
    if not inference_offset_df.empty:
        inference_offset_df.sort_values(["category", "system_name", "observable_key", "offset_windows"], inplace=True)
    if not inference_block_df.empty:
        inference_block_df.sort_values(["category", "system_name", "observable_key", "offset_windows", "block_id"], inplace=True)
    if not correlation_summary_df.empty:
        correlation_summary_df.sort_values(["category", "system_name", "sampling_unit_family", "observable_key"], inplace=True)
    equilibration_audit_df = pd.DataFrame(equilibration_audit_rows)
    equilibration_sensitivity_df = pd.DataFrame(equilibration_sensitivity_rows)
    if not equilibration_sensitivity_df.empty:
        baseline_cutoff = int(args.tmin)
        if baseline_cutoff not in set(pd.to_numeric(equilibration_sensitivity_df["cutoff_ns"], errors="coerce").dropna().astype(int).tolist()):
            baseline_cutoff = int(pd.to_numeric(equilibration_sensitivity_df["cutoff_ns"], errors="coerce").min())
        base = equilibration_sensitivity_df[
            pd.to_numeric(equilibration_sensitivity_df["cutoff_ns"], errors="coerce") == baseline_cutoff
        ][["temperature_K", "category", "system_name", "small_molecule_id", "observable_key", "sampling_unit_family", "corrected_mean"]].copy()
        base = base.rename(columns={"corrected_mean": "baseline_mean"})
        merge_keys = ["temperature_K", "category", "system_name", "small_molecule_id", "observable_key", "sampling_unit_family"]
        equilibration_sensitivity_df = equilibration_sensitivity_df.merge(base, on=merge_keys, how="left")
        cur = pd.to_numeric(equilibration_sensitivity_df["corrected_mean"], errors="coerce")
        basev = pd.to_numeric(equilibration_sensitivity_df["baseline_mean"], errors="coerce")
        equilibration_sensitivity_df["baseline_cutoff_ns"] = baseline_cutoff
        equilibration_sensitivity_df["delta_from_baseline"] = cur - basev
        equilibration_sensitivity_df["relative_delta_from_baseline"] = np.where(
            np.isfinite(basev) & (np.abs(basev) > 1e-12),
            (cur - basev) / basev,
            math.nan,
        )

    system_recommendation_rows: List[Dict[str, object]] = []
    system_layout_rows: List[Dict[str, object]] = []
    equilibration_system_audit_rows: List[Dict[str, object]] = []
    if not observable_summary_df.empty:
        for (temperature, category, system_name, small_molecule_id), group in observable_summary_df.groupby(["temperature_K", "category", "system_name", "small_molecule_id"]):
            system_block_size, rule = choose_system_recommended_block(group)
            if system_block_size is None:
                continue
            n_core = group[
                group["observable_key"].isin(CORE_RECOMMENDATION_OBSERVABLES)
                & np.isfinite(group["recommended_block_size_windows"])
            ].shape[0]
            # Summarise τ_int over all core recommendation observables
            core_group = group[group["observable_key"].isin(CORE_RECOMMENDATION_OBSERVABLES)]
            tau_int_max_ns = float(core_group["tau_int_ns"].max()) if "tau_int_ns" in core_group.columns else math.nan
            tau_int_median_ns = float(core_group["tau_int_ns"].median()) if "tau_int_ns" in core_group.columns else math.nan
            n_system_blocks = len(windows) // system_block_size if system_block_size > 0 else 0
            # Equilibration audit: use only EQUILIBRATION_CONTROL_OBSERVABLES
            # (structural/phase observables), not the full core set which includes
            # noisy transport observables that can produce misleading late-equilibration
            # calls in slow LLPS systems with short post-equilibration windows.
            equil_group = group[group["observable_key"].isin(EQUILIBRATION_CONTROL_OBSERVABLES)]
            equil_max_ns = float(equil_group["t_equil_ns"].max()) if ("t_equil_ns" in equil_group.columns and not equil_group.empty) else math.nan
            equil_max_idx = int(equil_group["t_equil_index"].max()) if ("t_equil_index" in equil_group.columns and not equil_group.empty) else 0
            equil_frac_max = float(equil_group["equil_fraction_discarded"].max()) if ("equil_fraction_discarded" in equil_group.columns and not equil_group.empty) else math.nan
            system_audit_flag = classify_equilibration_audit(
                equil_max_ns,
                equil_frac_max,
                default_tmin_ns=float(args.tmin),
                soft_max_ns=float(args.equil_soft_max_ns),
                hard_fraction=float(args.equil_hard_fraction),
            )
            system_recommendation_rows.append({
                "temperature_K": temperature,
                "category": category,
                "system_name": system_name,
                "small_molecule_id": small_molecule_id,
                "recommended_block_size_windows": system_block_size,
                "recommended_block_size_ns": system_block_size * args.dt,
                "n_blocks": n_system_blocks,
                "recommendation_rule": rule,
                "n_observables_used": int(n_core),
                "tau_int_max_ns": tau_int_max_ns,
                "tau_int_median_ns": tau_int_median_ns,
                "equil_t_max_ns": equil_max_ns,
                "equil_t_max_index": equil_max_idx,
                "equil_fraction_max": equil_frac_max,
                "equilibration_system_audit_flag": system_audit_flag,
                "n_control_observables_used": int(equil_group.shape[0]),
            })
            equilibration_system_audit_rows.append({
                "temperature_K": temperature,
                "category": category,
                "system_name": system_name,
                "small_molecule_id": small_molecule_id,
                "default_tmin_ns": float(args.tmin),
                "soft_max_ns": float(args.equil_soft_max_ns),
                "hard_fraction": float(args.equil_hard_fraction),
                "n_control_observables_used": int(equil_group.shape[0]),
                "equil_t_max_ns": equil_max_ns,
                "equil_t_max_index": equil_max_idx,
                "equil_fraction_max": equil_frac_max,
                "system_audit_flag": system_audit_flag,
            })
            layout = block_layout(len(windows), system_block_size, offset=0)
            for block_id, start_idx, end_idx in layout:
                block_windows = windows[start_idx : end_idx + 1]
                system_layout_rows.append({
                    "temperature_K": temperature,
                    "category": category,
                    "system_name": system_name,
                    "small_molecule_id": small_molecule_id,
                    "recommended_block_size_windows": system_block_size,
                    "recommended_block_size_ns": system_block_size * args.dt,
                    "block_id": block_id,
                    "block_start_window_index": int(start_idx),
                    "block_end_window_index": int(end_idx),
                    "block_start_ns": int(block_windows[0]),
                    "block_end_ns": int(block_windows[-1] + args.dt),
                    "window_members": ";".join(str(x) for x in block_windows),
                })

    viscosity_recommendation_rows: List[Dict[str, object]] = []
    viscosity_layout_rows: List[Dict[str, object]] = []
    viscosity_diagnostics_rows: List[Dict[str, object]] = []
    if not viscosity_df.empty and not observable_summary_df.empty:
        viscosity_summary = observable_summary_df[observable_summary_df["observable_key"].isin(VISCOSITY_OBSERVABLES)].copy()
        for (temperature, category, system_name, small_molecule_id), group in viscosity_summary.groupby(["temperature_K", "category", "system_name", "small_molecule_id"]):
            viscosity_block_size, rule = choose_viscosity_recommended_block(group)
            if viscosity_block_size is None:
                continue
            unit_meta = (
                viscosity_df[
                    (viscosity_df["temperature_K"] == temperature)
                    & (viscosity_df["category"] == category)
                    & (viscosity_df["system_name"] == system_name)
                ][["window_index", "window_start_ns", "window_end_ns"]]
                .drop_duplicates()
                .sort_values(["window_start_ns", "window_index"])
                .reset_index(drop=True)
            )
            if unit_meta.empty:
                continue
            durations = (unit_meta["window_end_ns"] - unit_meta["window_start_ns"]).to_numpy(dtype=float)
            durations = durations[np.isfinite(durations) & (durations > 0)]
            unit_span_ns = int(round(float(np.median(durations)))) if durations.size else int(args.dt)
            n_visc = group[np.isfinite(group["recommended_block_size_windows"])].shape[0]
            viscosity_recommendation_rows.append({
                "temperature_K": temperature,
                "category": category,
                "system_name": system_name,
                "small_molecule_id": small_molecule_id,
                "recommended_block_size_windows": viscosity_block_size,
                "recommended_block_size_ns": viscosity_block_size * unit_span_ns,
                "recommendation_rule": rule,
                "n_observables_used": int(n_visc),
            })
            diag_sub = viscosity_segment_diag_df[
                (viscosity_segment_diag_df["temperature_K"] == temperature)
                & (viscosity_segment_diag_df["category"] == category)
                & (viscosity_segment_diag_df["system_name"] == system_name)
            ].copy() if not viscosity_segment_diag_df.empty else pd.DataFrame()
            corr_sub = correlation_summary_df[
                (correlation_summary_df["temperature_K"] == temperature)
                & (correlation_summary_df["category"] == category)
                & (correlation_summary_df["system_name"] == system_name)
                & (correlation_summary_df["observable_key"].isin(VISCOSITY_OBSERVABLES))
            ].copy() if not correlation_summary_df.empty else pd.DataFrame()

            eta_row = corr_sub[corr_sub["observable_key"] == "eta_GK_Pa_s"] if not corr_sub.empty else pd.DataFrame()
            theo_row = corr_sub[corr_sub["observable_key"] == "eta_GK_Theo_Pa_s"] if not corr_sub.empty else pd.DataFrame()
            eta_mean = float(pd.to_numeric(eta_row.get("corrected_mean", pd.Series(dtype=float)), errors="coerce").iloc[-1]) if not eta_row.empty else math.nan
            eta_sem = float(pd.to_numeric(eta_row.get("corrected_sem", pd.Series(dtype=float)), errors="coerce").iloc[-1]) if not eta_row.empty else math.nan
            eta_blocks = int(pd.to_numeric(eta_row.get("n_corrected_blocks", pd.Series(dtype=float)), errors="coerce").iloc[-1]) if not eta_row.empty and np.isfinite(pd.to_numeric(eta_row.get("n_corrected_blocks", pd.Series(dtype=float)), errors="coerce").iloc[-1]) else 0
            theo_mean = float(pd.to_numeric(theo_row.get("corrected_mean", pd.Series(dtype=float)), errors="coerce").iloc[-1]) if not theo_row.empty else math.nan
            theo_sem = float(pd.to_numeric(theo_row.get("corrected_sem", pd.Series(dtype=float)), errors="coerce").iloc[-1]) if not theo_row.empty else math.nan
            if diag_sub.empty:
                n_segments_total = 0
                n_segments_raw = 0
                n_segments_theo = 0
                mean_fit_mae = math.nan
                median_fit_mae = math.nan
                median_modes = math.nan
                mean_rel_slope = math.nan
            else:
                n_segments_total = int(diag_sub.shape[0])
                n_segments_raw = int(np.count_nonzero(np.isfinite(pd.to_numeric(diag_sub["eta_GK_Pa_s"], errors="coerce"))))
                theo_success = diag_sub["theoretical_success"].fillna(False).astype(bool) if "theoretical_success" in diag_sub.columns else pd.Series(dtype=bool)
                n_segments_theo = int(np.count_nonzero(theo_success)) if not theo_success.empty else 0
                fit_mae = pd.to_numeric(diag_sub.get("fit_mae", pd.Series(dtype=float)), errors="coerce")
                mean_fit_mae = float(fit_mae.mean()) if fit_mae.notna().any() else math.nan
                median_fit_mae = float(fit_mae.median()) if fit_mae.notna().any() else math.nan
                modes = pd.to_numeric(diag_sub.get("maxwell_mode_count", pd.Series(dtype=float)), errors="coerce")
                median_modes = float(modes.median()) if modes.notna().any() else math.nan
                rel_slope = pd.to_numeric(diag_sub.get("late_integral_relative_slope", pd.Series(dtype=float)), errors="coerce")
                mean_rel_slope = float(rel_slope.mean()) if rel_slope.notna().any() else math.nan
            raw_theo_ratio = math.nan
            if np.isfinite(eta_mean) and np.isfinite(theo_mean) and eta_mean > 0 and theo_mean > 0:
                raw_theo_ratio = float(max(eta_mean / theo_mean, theo_mean / eta_mean))
            viscosity_diagnostics_rows.append({
                "temperature_K": temperature,
                "category": category,
                "system_name": system_name,
                "small_molecule_id": small_molecule_id,
                "segment_count_requested": int(visc_cfg.get("segments", DEFAULT_VISCOSITY_CONFIG["segments"])),
                "segment_duration_median_ns": int(unit_span_ns),
                "n_segments_total": int(n_segments_total),
                "n_segments_valid_raw": int(n_segments_raw),
                "n_segments_valid_theoretical": int(n_segments_theo),
                "recommended_block_size_segments": int(viscosity_block_size),
                "recommended_block_size_ns": int(viscosity_block_size * unit_span_ns),
                "n_corrected_blocks": int(eta_blocks),
                "eta_GK_corrected_mean": eta_mean,
                "eta_GK_corrected_sem": eta_sem,
                "eta_GK_Theo_corrected_mean": theo_mean,
                "eta_GK_Theo_corrected_sem": theo_sem,
                "raw_theoretical_ratio_fold": raw_theo_ratio,
                "mean_fit_mae": mean_fit_mae,
                "median_fit_mae": median_fit_mae,
                "median_maxwell_mode_count": median_modes,
                "mean_late_integral_relative_slope": mean_rel_slope,
                "theoretical_success_fraction": float(n_segments_theo / n_segments_total) if n_segments_total > 0 else math.nan,
                "viscosity_audit_flag": (
                    "requires_manual_review"
                    if (
                        (np.isfinite(raw_theo_ratio) and raw_theo_ratio > 5.0)
                        or (np.isfinite(mean_rel_slope) and mean_rel_slope > 0.25)
                        or (n_segments_total > 0 and (float(n_segments_theo) / n_segments_total) < 0.5)
                    )
                    else "ok"
                ),
            })
            layout = block_layout(len(unit_meta), viscosity_block_size, offset=0)
            for block_id, start_idx, end_idx in layout:
                block_units = unit_meta.iloc[start_idx : end_idx + 1]
                block_starts = block_units["window_start_ns"].to_numpy(dtype=int)
                block_ends = block_units["window_end_ns"].to_numpy(dtype=int)
                viscosity_layout_rows.append({
                    "temperature_K": temperature,
                    "category": category,
                    "system_name": system_name,
                    "small_molecule_id": small_molecule_id,
                    "recommended_block_size_windows": viscosity_block_size,
                    "recommended_block_size_ns": viscosity_block_size * unit_span_ns,
                    "block_id": block_id,
                    "block_start_window_index": int(start_idx),
                    "block_end_window_index": int(end_idx),
                    "block_start_ns": int(block_starts[0]),
                    "block_end_ns": int(block_ends[-1]),
                    "window_members": ";".join(str(x) for x in block_starts),
                })

    equilibration_df = pd.DataFrame(equilibration_rows)
    equilibration_system_audit_df = pd.DataFrame(equilibration_system_audit_rows)
    viscosity_diagnostics_df = pd.DataFrame(viscosity_diagnostics_rows)
    policy_payload = dict(STATISTICAL_POLICY)
    policy_payload.update({
        "tmin_ns": int(args.tmin),
        "tmax_ns": int(args.tmax),
        "window_dt_ns": int(args.dt),
        "equilibration_soft_max_ns": float(args.equil_soft_max_ns),
        "equilibration_hard_fraction": float(args.equil_hard_fraction),
        "equilibration_sensitivity_cutoffs_ns": ",".join(str(x) for x in sensitivity_cutoffs),
        "viscosity_segments_requested": int(visc_cfg.get("segments", DEFAULT_VISCOSITY_CONFIG["segments"])),
    })
    policy_df = pd.DataFrame([{"key": key, "value": value} for key, value in policy_payload.items()])

    pd.DataFrame(system_recommendation_rows).to_csv(os.path.join(results_root, "RECOMMENDED_SYSTEM_BLOCKS.csv"), index=False)
    pd.DataFrame(system_layout_rows).to_csv(os.path.join(results_root, "RECOMMENDED_SYSTEM_BLOCK_LAYOUT.csv"), index=False)
    pd.DataFrame(viscosity_recommendation_rows).to_csv(os.path.join(results_root, "RECOMMENDED_VISCOSITY_BLOCKS.csv"), index=False)
    pd.DataFrame(viscosity_layout_rows).to_csv(os.path.join(results_root, "RECOMMENDED_VISCOSITY_SEGMENT_LAYOUT.csv"), index=False)
    superblock_offset_df.to_csv(os.path.join(results_root, "SUPERBLOCK_OFFSETS.csv"), index=False)
    superblock_summary_df.to_csv(os.path.join(results_root, "SUPERBLOCK_DIAGNOSTICS.csv"), index=False)
    inference_offset_df.to_csv(os.path.join(results_root, "INFERENCE_OFFSET_DIAGNOSTICS.csv"), index=False)
    inference_block_df.to_csv(os.path.join(results_root, "INFERENCE_BLOCK_SCALARS.csv"), index=False)
    correlation_summary_df.to_csv(os.path.join(results_root, "CORRELATION_INFERENCE_DIAGNOSTICS.csv"), index=False)
    acf_df.to_csv(os.path.join(results_root, "ACF_VALUES.csv"), index=False)
    pymbar_df.to_csv(os.path.join(results_root, "PYMBAR_DIAGNOSTICS.csv"), index=False)
    pymbar_sub_df.to_csv(os.path.join(results_root, "PYMBAR_SUBSAMPLES.csv"), index=False)
    observable_summary_df.to_csv(os.path.join(results_root, "SUMMARY_COMPARISON.csv"), index=False)
    recommended_block_df.to_csv(os.path.join(results_root, "RECOMMENDED_BLOCKS.csv"), index=False)
    recommended_scalar_df.to_csv(os.path.join(results_root, "RECOMMENDED_BLOCK_SCALARS.csv"), index=False)
    correlation_summary_df.to_csv(os.path.join(results_root, "CORRELATION_QUANT_DATA.csv"), index=False)
    equilibration_df.to_csv(os.path.join(results_root, "EQUILIBRATION_DIAGNOSTICS.csv"), index=False)
    equilibration_audit_df.to_csv(os.path.join(results_root, "EQUILIBRATION_AUDIT.csv"), index=False)
    equilibration_system_audit_df.to_csv(os.path.join(results_root, "EQUILIBRATION_SYSTEM_AUDIT.csv"), index=False)
    equilibration_sensitivity_df.to_csv(os.path.join(results_root, "EQUILIBRATION_SENSITIVITY.csv"), index=False)
    viscosity_diagnostics_df.to_csv(os.path.join(results_root, "VISCOSITY_DIAGNOSTICS.csv"), index=False)
    policy_df.to_csv(os.path.join(results_root, "STATISTICAL_POLICY.csv"), index=False)

    print("Correlation diagnostics complete.")
    print(f"Results written to: {results_root}")
    print(f"Figures written to: {figures_root}")


if __name__ == "__main__":
    main()
