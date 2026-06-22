"""System-level thermodynamics, dynamics and contact analysis (pipeline Step 3).

Pipeline role
-------------
Third stage of the standard analysis pipeline. After the per-frame extraction
(``RCC_ANALYSIS_FINAL``), time-window averaging (``AVERAGE_SIMULATIONS``) and
composition counting (``MaxCluster``), this script computes the physical
observables that populate the master results table for every SG / DSM / NDSM
system at one temperature. The core ``system_analysis`` class fits radial
density profiles (dense/dilute concentrations, condensate radius, interface width,
surface tension, transfer free energy), computes confined diffusion (D, tau,
confinement length, Stokes-Einstein viscosity) and Green-Kubo viscosity, builds
the per-species spatial descriptors, and writes the residue/acid/SM contact-map
data tables.

Key inputs
----------
- A ``{path}/{folder}_{T}_{dt}_{tmin}_{tmax}`` analysis root with the
  ``ANALYSIS_{SG,DSM,NDSM}_AVE`` window-averaged density/PCA/cluster/contact
  files and MSD/stress products (plus COM sidecars).
- CLI flags: ``--path``, ``--folder``, ``--T``, ``--dt``, ``--tmin``,
  ``--tmax``, ``--c``, species selectors, and the full
  ``--diff-*`` / ``--visc-*`` override flags.

Key outputs (under ``{path}/.../RESULTS/``)
- ``RESULTS/SUMMARY/Quant_Data.csv`` (and SG/DSM/NDSM variants): per-system and
  aggregate thermodynamic/dynamic metrics.
- Residue/acid/SM contact-map data CSVs.

This module is also imported as a library by ``SYSTEM_ANALYSIS_FINAL_CORRELATED``
(the correlated pipeline), which reuses ``system_analysis``.

Example invocation
-------------------
    python system_analysis.py \
        --path TEMP_300 --folder CLASSIFY \
        --T 300 --dt 50 --tmin 50 --tmax 2000
"""
import argparse
import glob
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd

from viscosity import visc
from rdp import rdp
from diffusion import DEFAULT_DIFFUSION_SPECIES, run_diffusion_confined, _stokes_einstein_D

warnings.filterwarnings("ignore")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_DIFFUSION_PARAMS = {
    "downsample_stride": 1,
    "dt_diff": None,
    "segments": 8,
    "n_boot": 10,
    "slope_iterations": 200,
    "dt_pts": 7,
    "slope_tol": 0.15,
    "min_diff_pts": 7,
    "slope_tol_plateau": 0.10,
    "min_plateau_pts": 7,
    "boot_r2": 0.90,
    "smooth_window_pts": 9,
    "smooth_polyorder": 2,
    "seed": 0,
    "min_diff_start_ns": 100.0,
    "min_diff_span_ns": 200.0,
    "max_diff_fit_fraction": 0.75,
    "linear_r2_min": 0.995,
    "min_plateau_span_ns": 200.0,
    "min_segment_ns": 200.0,
    "min_primary_fraction": 0.25,
    "origin_resample": True,
    "origin_window_ns": 1000.0,
    "origin_stride_ns": 200.0,
    "origin_candidate_windows_ns": (500.0, 750.0, 1000.0, 1250.0, 1500.0),
    "origin_min_success_fraction": 0.70,
    "origin_min_success_count": 3,
    "origin_max_origins": 12,
}

DEFAULT_VISCOSITY_PARAMS = {
    "segments": 20,
    "iterations": 10,
    "n_boot": 10,
    "dt_unit": 2000000E-15,
    "n_point": 1000,
    "n_tau": 10,
    "seed": 0,
}


def _time_labels(t):
    """Return the candidate string spellings (int then float) of a time tag."""
    labels = []
    try:
        tf = float(t)
        if tf.is_integer():
            labels.append(str(int(tf)))
        labels.append(str(tf))
    except Exception:
        labels.append(str(t))

    seen = set()
    ordered = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            ordered.append(label)
    return ordered


def _resolve_time_file(path_builder, t):
    """Return the first existing path produced by ``path_builder`` over time spellings."""
    for label in _time_labels(t):
        path = path_builder(label)
        if os.path.isfile(path):
            return path
    return None


def _split_contiguous_segments(items, n_segments):
    """Split a sequence into up to ``n_segments`` contiguous chunks (as lists)."""
    seq = [item for item in items]
    if not seq:
        return []
    n = max(1, min(int(n_segments), len(seq)))
    chunks = np.array_split(np.asarray(seq, dtype=object), n)
    return [[x.item() if hasattr(x, "item") else x for x in chunk] for chunk in chunks if len(chunk) > 0]


def _bootstrap_segment_means(values, n_boot=10, seed=0):
    """Return ``(mean, bootstrap_sem)`` of finite values via resampling the mean.

    SEM is the std (ddof=1) of ``n_boot`` bootstrap-resampled means; NaN when
    fewer than two finite values are present.
    """
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return math.nan, math.nan
    mean = float(np.mean(vals))
    if vals.size == 1:
        return mean, math.nan
    n_boot = max(1, int(n_boot))
    rng = np.random.RandomState(int(seed))
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = vals[rng.randint(0, vals.size, size=vals.size)]
        boot[i] = float(np.mean(sample))
    sem = float(np.std(boot, ddof=1)) if boot.size > 1 else math.nan
    return mean, sem


# Display reorderings applied to contact-map matrices before they are written
# to the result CSVs (consumed by the data reorder helpers below).
_RESIDUE_DISPLAY_ORDER = [4, 5, 3, 0, 6, 1, 2]
_ACID_DISPLAY_ORDER = [4, 15, 2, 6, 7, 14, 3, 16, 11, 18, 1, 17, 5, 9, 19, 10, 0, 13, 8, 12, 20, 21, 22, 23]
_DIFFUSION_SPECIES_ORDER = list(DEFAULT_DIFFUSION_SPECIES)
_DIFFUSION_PROTEIN_GROUP = tuple(sp for sp in DEFAULT_DIFFUSION_SPECIES if sp != "RNA")


def _copy_float_array(array):
    """Return a writable float64 copy of ``array``."""
    return np.array(array, dtype=float, copy=True)


def _reorder_square(array, order):
    """Reorder both rows and columns of a square matrix by ``order``."""
    arr = _copy_float_array(array)
    idx = np.asarray(order, dtype=int)
    return arr[np.ix_(idx, idx)]


def _reorder_rows(array, order):
    """Reorder the rows of a matrix by ``order`` (columns unchanged)."""
    arr = _copy_float_array(array)
    idx = np.asarray(order, dtype=int)
    return arr[idx, :]


def _prepare_residue_square(array):
    """Reorder a residue-residue contact matrix into the display species order."""
    return _reorder_square(array, _RESIDUE_DISPLAY_ORDER)


def _prepare_residue_rows(array):
    """Reorder the rows of a residue matrix into the display species order."""
    return _reorder_rows(array, _RESIDUE_DISPLAY_ORDER)


def _prepare_acid_square(array):
    """Reorder an acid-acid contact matrix into the display residue-type order."""
    return _reorder_square(array, _ACID_DISPLAY_ORDER)


def _prepare_acid_rows(array):
    """Reorder the rows of an acid matrix into the display residue-type order."""
    return _reorder_rows(array, _ACID_DISPLAY_ORDER)


# Species list must match ANALYSIS._CONFINED_SPECIES.
_AGGREGATE_SPATIAL_SPECIES = ["G3BP1", "PABP1", "TIA1", "TTP", "FUS", "TDP43", "RNA"]


def _read_contact_matrix(path: str):
    """Read a comma-delimited contact matrix CSV as a 2-D float array, or None."""
    if not os.path.isfile(path):
        return None
    try:
        return np.loadtxt(path, delimiter=",", dtype=float, ndmin=2)
    except Exception:
        try:
            return np.array(pd.read_csv(path, header=None), dtype=float)
        except Exception:
            return None


def _write_contact_sem_map(path: str, arr: np.ndarray) -> None:
    """Write a contact-map SEM matrix to a comma-delimited CSV."""
    np.savetxt(path, np.asarray(arr, dtype=float), delimiter=",")


def _contact_matrix_sem(mats):
    """Return the elementwise NaN-aware SEM over a list of contact matrices.

    NaN-filled when fewer than two matrices are supplied.
    """
    arr = np.stack([np.asarray(m, dtype=float) for m in mats], axis=0)
    if arr.shape[0] <= 1:
        return np.full(arr.shape[1:], np.nan, dtype=float)
    return np.nanstd(arr, axis=0, ddof=1) / math.sqrt(arr.shape[0])


def _ratio_difference_sem(sm_arr, sm_sem, ref_arr, ref_sem):
    """SEM for X/ref up to an additive constant, assuming independent maps."""
    if sm_sem is None or ref_sem is None:
        return None
    sm_arr = np.asarray(sm_arr, dtype=float)
    ref_arr = np.asarray(ref_arr, dtype=float)
    sm_sem = np.asarray(sm_sem, dtype=float)
    ref_sem = np.asarray(ref_sem, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sem = np.sqrt((sm_sem / ref_arr) ** 2 + (sm_arr * ref_sem / (ref_arr ** 2)) ** 2)
    sem[~np.isfinite(sem)] = np.nan
    return sem


def _contact_1d_sem(sem_arr):
    """Collapse a contact-map SEM matrix to a 1-D row SEM via quadrature sum."""
    if sem_arr is None:
        return None
    arr = np.asarray(sem_arr, dtype=float)
    return np.sqrt(np.nansum(arr ** 2, axis=1)).reshape(-1, 1)


def _write_residue_contact_map(path, name, prefix, contact_array, one_d=None):
    """Write the display-reordered residue contact map (square + 1-D row sum) CSVs."""
    square = _prepare_residue_square(contact_array)
    np.savetxt("{}/RESULTS/RESIDUE_CONTACT_MAPS/{}_{}.csv".format(path, prefix, name), square, delimiter=",")
    src = contact_array if one_d is None else one_d
    row_1d = _prepare_residue_rows(np.asarray(src, dtype=float).sum(axis=1).reshape(-1, 1))
    np.savetxt("{}/RESULTS/RESIDUE_CONTACT_MAPS/{}_{}_1D.csv".format(path, prefix, name), row_1d, delimiter=",")


def _write_acid_contact_map(path, name, prefix, contact_array, one_d=None):
    """Write the display-reordered acid contact map (square + 1-D row sum) CSVs."""
    square = _prepare_acid_square(contact_array)
    np.savetxt("{}/RESULTS/ACID_CONTACT_MAPS/{}_{}.csv".format(path, prefix, name), square, delimiter=",")
    src = contact_array if one_d is None else one_d
    row_1d = _prepare_acid_rows(np.asarray(src, dtype=float).sum(axis=1).reshape(-1, 1))
    np.savetxt("{}/RESULTS/ACID_CONTACT_MAPS/{}_{}_1D.csv".format(path, prefix, name), row_1d, delimiter=",")


def _fill_aggregate_spatial_from_members(df, member_names, agg_id):
    """Fill an aggregate row's per-species Occ / (r/R) from its member systems.

    Overwrites the aggregate (``agg_id``) row's occupancy and r/R columns (and
    their ``SIG`` columns) with the mean +/- SEM over the member system rows,
    since the aggregate pass lacks the tracked-cluster sidecar. Returns ``df``.
    """
    # The second sg_sm_analysis_full pass on the aggregate has no tracked_cluster
    # NPZ sidecar, so Occ_{sp} / r/R_{sp} come out NaN. Overwrite with mean ± SEM
    # of the per-system rows.
    if df is None or df.empty or "Small Molecule ID" not in df.columns:
        return df
    agg_mask = df["Small Molecule ID"].astype(str) == agg_id
    if not agg_mask.any():
        return df
    name_col = "Small Molecule Name" if "Small Molecule Name" in df.columns else None
    if name_col is None:
        return df
    member_mask = df[name_col].astype(str).isin(list(member_names))
    if not member_mask.any():
        return df
    member_rows = df.loc[member_mask]
    for sp in _AGGREGATE_SPATIAL_SPECIES:
        for col, sig_col in (
            (f"$Occ_{{{sp}}}$", f"SIG$Occ_{{{sp}}}$"),
            (f"$r/R_{{{sp}}}$", f"SIG$r/R_{{{sp}}}$"),
        ):
            if col not in df.columns:
                continue
            vals = pd.to_numeric(member_rows[col], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            mean_val = float(np.mean(vals))
            if vals.size >= 2:
                sem_val = float(np.std(vals, ddof=1) / math.sqrt(vals.size))
            else:
                sem_val = math.nan
            df.loc[agg_mask, col] = mean_val
            if sig_col in df.columns:
                df.loc[agg_mask, sig_col] = sem_val
    return df


class system_analysis():
    """System-level analysis engine for one temperature's SG/DSM/NDSM systems.

    Holds the diffusion and viscosity parameter sets (merged from defaults plus
    caller overrides), optional species selectors, and the resolved data/MSD
    roots, and caches diffusion/viscosity results. Its methods fit radial
    density profiles (concentrations, R/W, surface tension, transfer free
    energy), compute confined diffusion and Green-Kubo viscosity, derive
    per-species spatial descriptors, build the Quant_Data tables, and render the
    contact-map figures. The same engine is reused by the correlated pipeline.
    """
    def __init__(self, diffusion_params=None, viscosity_params=None, species_to_analyze=None, species_for_global=None, data_root=None, msd_roots=None):
        """Configure the engine with parameter sets, selectors, and data roots.

        Args:
            diffusion_params: Overrides merged onto ``DEFAULT_DIFFUSION_PARAMS``.
            viscosity_params: Overrides merged onto ``DEFAULT_VISCOSITY_PARAMS``.
            species_to_analyze: Optional subset of species for diffusion.
            species_for_global: Optional subset of species for global "ALL"
                diffusion averages.
            data_root: Absolute ``TEMP_XXX`` directory holding ``ANALYSIS_<cat>``
                inputs (required for path resolution).
            msd_roots: Extra roots searched for MSD products, ahead of the
                ``ANALYSIS_MSD_ROOTS`` env var and the in-repo defaults.
        """
        self.diffusion_params = self._merge_defaults(DEFAULT_DIFFUSION_PARAMS, diffusion_params)
        self.viscosity_params = self._merge_defaults(DEFAULT_VISCOSITY_PARAMS, viscosity_params)
        self.species_to_analyze = species_to_analyze
        # Optional subset of species for global "ALL" diffusion averages
        self.species_for_global = species_for_global
        self.diffusion_cache = {}
        self.viscosity_cache = {}
        # Absolute TEMP_XXX directory that stores ANALYSIS_<cat> inputs
        self.data_root = os.path.abspath(data_root) if data_root else None
        self.msd_roots = self._discover_msd_roots(msd_roots)
        self._msd_path_cache = {}

    @staticmethod
    def _merge_defaults(defaults, overrides):
        """Return ``defaults`` overlaid with non-None entries from ``overrides``."""
        cfg = dict(defaults) if defaults is not None else {}
        if overrides:
            for key, value in overrides.items():
                if value is not None:
                    cfg[key] = value
        return cfg

    def _discover_msd_roots(self, msd_roots=None):
        """Return the de-duplicated, existing MSD search roots.

        Combines explicit ``msd_roots``, the ``ANALYSIS_MSD_ROOTS`` env var, and
        the in-repo ``PYTHON_ANALYSIS`` / ``PYTHON_SIMULATIONS`` defaults.
        """
        roots = []
        if msd_roots:
            roots.extend(msd_roots)
        env_roots = os.environ.get("ANALYSIS_MSD_ROOTS")
        if env_roots:
            roots.extend([p for p in env_roots.split(os.pathsep) if p])
        for default_root in [os.path.join(_REPO_ROOT, "PYTHON_ANALYSIS"), os.path.join(_REPO_ROOT, "PYTHON_SIMULATIONS")]:
            if os.path.isdir(default_root):
                roots.append(default_root)
        seen = set()
        out = []
        for root in roots:
            root_abs = os.path.abspath(root)
            if root_abs in seen or not os.path.isdir(root_abs):
                continue
            seen.add(root_abs)
            out.append(root_abs)
        return out

    def _resolve_raw_folder(self, folder):
        """Return (base_name, absolute_path) for ANALYSIS_<cat> inputs."""
        base_folder = folder[:-4] if folder.endswith("_AVE") else folder
        if self.data_root is None:
            raise RuntimeError("Data root not set; pass TEMP_XXX via --path")
        return base_folder, os.path.join(self.data_root, base_folder)

    def _resolve_msd_path(self, folder, sm, T=None, suffix="_msd.out.all"):
        """Locate the MSD product file for one system, caching the result.

        Prefers a sibling file in the ``ANALYSIS_<cat>`` input folder, then a
        ``TEMP_{T}/<CAT>/MSD/`` path under each MSD root, then a recursive glob
        (temperature-filtered first, unfiltered as a last resort). Falls back to
        the local (possibly nonexistent) path so callers can report it.
        """
        base_folder, raw_folder_path = self._resolve_raw_folder(folder)
        local_path = os.path.join(raw_folder_path, f"{sm}{suffix}")
        if os.path.isfile(local_path):
            return local_path
        cache_key = (sm, T, suffix)
        if cache_key in self._msd_path_cache:
            return self._msd_path_cache[cache_key]
        matches = []
        if T is not None:
            analysis_cat = base_folder.replace("ANALYSIS_", "").replace("_AVE", "").upper()
            for root in self.msd_roots:
                direct = os.path.join(root, f"TEMP_{T}", analysis_cat, "MSD", f"{sm}{suffix}")
                if os.path.isfile(direct):
                    self._msd_path_cache[cache_key] = direct
                    return direct
        for root in self.msd_roots:
            pattern = os.path.join(root, "**", "MSD", f"{sm}{suffix}")
            for candidate in glob.glob(pattern, recursive=True):
                if T is not None and str(T) not in candidate:
                    continue
                matches.append(os.path.abspath(candidate))
        if not matches and T is not None:
            for root in self.msd_roots:
                pattern = os.path.join(root, "**", "MSD", f"{sm}{suffix}")
                matches.extend(os.path.abspath(candidate) for candidate in glob.glob(pattern, recursive=True))
        resolved = sorted(set(matches))[0] if matches else local_path
        self._msd_path_cache[cache_key] = resolved
        return resolved

    @staticmethod
    def _mean_sem(values):
        """Return ``(mean, SEM)`` of the finite entries (NaN SEM when < 2)."""
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return math.nan, math.nan
        mean = float(np.mean(arr))
        if arr.size == 1:
            return mean, math.nan
        sem = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
        return mean, sem

    def _average_cached_tuple(self, cache, tags, label):
        """Return the per-element NaN-mean of cached metric tuples over ``tags``."""
        values = []
        missing = []
        for tag in tags:
            if tag in cache:
                values.append(cache[tag])
            else:
                missing.append(tag)
        if missing:
            print(f"[WARN] Missing cached {label} metrics for: {', '.join(missing)}")
        if not values:
            raise RuntimeError(f"No cached {label} metrics available for aggregation.")
        arr = np.array(values, dtype=float)
        means = np.nanmean(arr, axis=0)
        return tuple(means.tolist())

    @staticmethod
    def _average_summary_lookup_entries(entries):
        """NaN-mean nested ``{key: {metric: value}}`` summary-lookup dicts."""
        keys = set()
        for entry in entries:
            keys.update(entry.keys())
        averaged = {}
        for key in keys:
            metric_keys = set()
            for entry in entries:
                metric_keys.update(entry.get(key, {}).keys())
            metric_avg = {}
            for metric_key in metric_keys:
                vals = []
                for entry in entries:
                    value = entry.get(key, {}).get(metric_key, math.nan)
                    try:
                        vals.append(float(value))
                    except Exception:
                        continue
                metric_arr = np.asarray(vals, dtype=float)
                metric_arr = metric_arr[np.isfinite(metric_arr)]
                metric_avg[metric_key] = float(np.mean(metric_arr)) if metric_arr.size else math.nan
            averaged[key] = metric_avg
        return averaged

    @staticmethod
    def _average_stats_dicts(entries):
        """Aggregate a list of diffusion-stats dicts into a single averaged dict.

        Numeric keys are NaN-averaged; string keys collapse to their common
        value or ``"Mixed"``; the nested ``summary_lookup`` is averaged in turn.
        """
        if not entries:
            raise RuntimeError("No cached diffusion stats available for aggregation.")
        averaged = {}
        numeric_keys = set()
        string_keys = set()
        for entry in entries:
            for key, value in entry.items():
                if key == "summary_lookup":
                    continue
                if isinstance(value, str):
                    string_keys.add(key)
                else:
                    numeric_keys.add(key)
        for key in numeric_keys:
            vals = []
            for entry in entries:
                value = entry.get(key, math.nan)
                try:
                    vals.append(float(value))
                except Exception:
                    continue
            arr = np.asarray(vals, dtype=float)
            arr = arr[np.isfinite(arr)]
            averaged[key] = float(np.mean(arr)) if arr.size else math.nan
        for key in string_keys:
            vals = [str(entry.get(key, "")).strip() for entry in entries if str(entry.get(key, "")).strip()]
            unique = sorted(set(vals))
            averaged[key] = unique[0] if len(unique) == 1 else "Mixed"
        summary_entries = [entry.get("summary_lookup", {}) for entry in entries if entry.get("summary_lookup")]
        averaged["summary_lookup"] = system_analysis._average_summary_lookup_entries(summary_entries) if summary_entries else {}
        return averaged

    def _average_cached_stats(self, cache, tags, label):
        """Average the cached stats dicts for ``tags`` (warns on any missing)."""
        values = []
        missing = []
        for tag in tags:
            if tag in cache:
                values.append(cache[tag])
            else:
                missing.append(tag)
        if missing:
            print(f"[WARN] Missing cached {label} stats for: {', '.join(missing)}")
        if not values:
            raise RuntimeError(f"No cached {label} stats available for aggregation.")
        return self._average_stats_dicts(values)

    def average_viscosity_metrics(self, tags):
        """Return the class-average viscosity tuple over the cached ``tags``."""
        return self._average_cached_tuple(self.viscosity_cache, tags, "viscosity")


    _CONFINED_SPECIES = ["G3BP1", "PABP1", "TIA1", "TTP", "FUS", "TDP43", "RNA"]

    _CONFINED_NAN_KEYS = [
        "l_A_mean", "l_A_sem", "tau_s_mean", "tau_s_sem",
        "D_cage_m2_s", "D_cage_m2_s_sem",
        "D_loglog_m2_s", "D_loglog_m2_s_sem",
        "plateau_start_ns", "plateau_cv", "n_plateau",
        "confinement_stage",
        "tau_lin_ns", "tau_lin_r2",
        "origin_resample_status", "origin_resample_window_ns",
        "origin_resample_stride_ns", "origin_resample_n_success",
        "origin_resample_n_total", "origin_resample_success_fraction",
        "origin_resample_l_sem_A", "origin_resample_tau_sem_ns",
        "origin_resample_D_cage_sem_m2_s",
        "origin_resample_sem_block_origins", "origin_resample_sem_block_ns",
        "origin_resample_sem_n_blocks", "origin_resample_min_plateau_window_ns",
        "l_A_chain_median", "l_A_chain_mean",
        "tau_s_chain_median", "tau_s_chain_mean",
        "D_cage_m2_s_chain_median", "D_cage_m2_s_chain_mean",
        "Rg_A_mean", "Rg_A_sem", "Rh_A_mean", "Rh_A_sem",
        "n_confined", "n_total_inside",
    ] + [f"l_conf_{sp}_A_mean" for sp in _CONFINED_SPECIES
    ] + [f"l_conf_{sp}_A_sem" for sp in _CONFINED_SPECIES
    ] + [f"tau_conf_{sp}_s_mean" for sp in _CONFINED_SPECIES
    ] + [f"tau_conf_{sp}_s_sem" for sp in _CONFINED_SPECIES
    ] + [f"n_conf_{sp}" for sp in _CONFINED_SPECIES
    ] + [f"Rg_{sp}_A_mean" for sp in _CONFINED_SPECIES
    ] + [f"Rg_{sp}_A_sem" for sp in _CONFINED_SPECIES
    ] + [f"Rh_{sp}_A_mean" for sp in _CONFINED_SPECIES
    ] + [f"Rh_{sp}_A_sem" for sp in _CONFINED_SPECIES
    ] + [f"Occ_{sp}_mean" for sp in _CONFINED_SPECIES
    ] + [f"Occ_{sp}_sem" for sp in _CONFINED_SPECIES
    ] + [f"r_over_R_{sp}_mean" for sp in _CONFINED_SPECIES
    ] + [f"r_over_R_{sp}_sem" for sp in _CONFINED_SPECIES]

    def calc_diffusion_confined(self, path, sm, folder, dt, T, tmin, tmax):
        """Two-stage confinement: plateau-first l_conf, then linearized tau."""
        if os.environ.get("SKIP_RCC_DIFFUSION", "0") == "1":
            print(f"[calc_diffusion_confined] SKIP_RCC_DIFFUSION=1; returning NaN for {sm}")
            return {k: math.nan for k in self._CONFINED_NAN_KEYS}
        msd_path = self._resolve_msd_path(folder, sm, T=T, suffix="_msd_rdp.out.all")
        if not os.path.isfile(msd_path):
            print(f"[calc_diffusion_confined] MSD file not found: {msd_path}")
            return {k: math.nan for k in self._CONFINED_NAN_KEYS}
        cfg = self.diffusion_params
        return run_diffusion_confined(
            msd_path=msd_path,
            cluster_root=self.data_root,
            tag=sm,
            temp_K=T,
            tmin=tmin,
            dt=dt,
            tmax=tmax,
            seed=cfg.get("seed", 0),
            origin_resample=cfg.get("origin_resample", True),
            origin_window_ns=cfg.get("origin_window_ns", 1000.0),
            origin_stride_ns=cfg.get("origin_stride_ns", 200.0),
            origin_candidate_windows_ns=cfg.get("origin_candidate_windows_ns"),
            origin_min_success_fraction=cfg.get("origin_min_success_fraction", 0.70),
            origin_min_success_count=cfg.get("origin_min_success_count", 3),
            origin_max_origins=cfg.get("origin_max_origins", 12),
        )

    @staticmethod
    def _build_diffusion_qc_columns(conf):
        """Extract the confinement QC fields from a result dict as Quant_Data columns."""
        return {
            "QC Confinement stage": conf.get("confinement_stage", ""),
            "QC Confinement n_total_inside": conf.get("n_total_inside", math.nan),
            "QC Confinement n_confined": conf.get("n_confined", math.nan),
            "QC Confinement n_plateau": conf.get("n_plateau", math.nan),
            "QC Confinement plateau_cv": conf.get("plateau_cv", math.nan),
            "QC Confinement origin_status": conf.get("origin_resample_status", ""),
            "QC Confinement origin_window_ns": conf.get("origin_resample_window_ns", math.nan),
            "QC Confinement origin_stride_ns": conf.get("origin_resample_stride_ns", math.nan),
            "QC Confinement origin_n_success": conf.get("origin_resample_n_success", math.nan),
            "QC Confinement origin_n_total": conf.get("origin_resample_n_total", math.nan),
            "QC Confinement origin_success_fraction": conf.get("origin_resample_success_fraction", math.nan),
            "QC Confinement origin_min_plateau_window_ns": conf.get("origin_resample_min_plateau_window_ns", math.nan),
            "QC Confinement origin_sem_block_ns": conf.get("origin_resample_sem_block_ns", math.nan),
        }

    def calc_visc(self, path, folder, sm, r, tmin, tmax, T, viscosity_params=None, dt_window=None):
        """Green-Kubo viscosity for one system from its condensate stress tensor.

        Reads the per-window tracked (or legacy) stress CSVs within
        ``[tmin, tmax]``, converts the off-diagonal stresses to Pa using the
        condensate volume from ``r`` (the RDP droplet radius R + 0.5 W, in
        Angstrom), and runs ``VSC.visc`` to bootstrap the Maxwell-mode fit.
        Returns ``(eta_raw, eta_raw_sem, eta_theo, eta_theo_sem)`` in Pa.s, or a
        NaN tuple when ``SKIP_VISCOSITY=1``.
        """
        if os.environ.get("SKIP_VISCOSITY", "0") == "1":
            print(f"[calc_visc] SKIP_VISCOSITY=1; returning NaN viscosity tuple for {sm}")
            return math.nan, math.nan, math.nan, math.nan
        vol_sys = 4 / 3 * math.pi * (r * 1e-10) ** 3  # m^3
        cfg = self._merge_defaults(self.viscosity_params, viscosity_params)
        segments = cfg["segments"]
        iterations = cfg["iterations"]
        n_boot = cfg["n_boot"]
        dt_unit = cfg["dt_unit"]
        n_point = cfg["n_point"]
        n_tau = cfg["n_tau"]

        _, raw_folder_path = self._resolve_raw_folder(folder)

        def _read_stress_csv(fpath, Pxyz, time):
            """Read a single stress CSV and append matching rows."""
            with open(fpath, "r") as file:
                for line in file.readlines()[1:]:
                    ln = line.split(",")
                    t = float(ln[0]) * 20E-6
                    if int(t) <= tmax and int(t) >= tmin:
                        time.append(float(ln[0]) * 20)
                        Pxy_sum = -float(ln[4])
                        Pxz_sum = -float(ln[5])
                        Pyz_sum = -float(ln[6])
                        conv = 101325.0 * 1e-30 / vol_sys
                        Pxyz.append([Pxy_sum * conv, Pxz_sum * conv, Pyz_sum * conv])

        def read_files(sm):
            Pxyz, time = [], []
            print('\nPreparing Pressure Tensor Array')

            # Try per-window tracked stress files (one per RCC window)
            tracked_files = []
            if dt_window is not None and dt_window > 0:
                for t_ns in range(0, int(tmax), int(dt_window)):
                    fname = _resolve_time_file(
                        lambda label: os.path.join(raw_folder_path, f"Stress_Tensor_Tracked_{sm}_{label}.csv"),
                        t_ns,
                    )
                    if fname is not None:
                        tracked_files.append(fname)

            if tracked_files:
                print(f"  Reading {len(tracked_files)} per-window tracked stress files")
                for fpath in tracked_files:
                    _read_stress_csv(fpath, Pxyz, time)
            else:
                # Fall back to single-file: tracked at label 0, then static, then legacy
                stress_file = _resolve_time_file(
                    lambda label: os.path.join(raw_folder_path, f"Stress_Tensor_Tracked_{sm}_{label}.csv"),
                    0,
                )
                if stress_file is None:
                    stress_file = _resolve_time_file(
                        lambda label: os.path.join(raw_folder_path, f"Stress_Tensor_{sm}_{label}.csv"),
                        0,
                    )
                if stress_file is None:
                    legacy_plain = os.path.join(raw_folder_path, f"Stress_Tensor_{sm}.csv")
                    stress_file = legacy_plain if os.path.isfile(legacy_plain) else None
                if stress_file is None:
                    raise FileNotFoundError(
                        f"Stress tensor file not found for {sm} in {raw_folder_path}; "
                        "checked tracked, windowed, and legacy stress tensor files"
                    )
                _read_stress_csv(stress_file, Pxyz, time)

            return np.array(Pxyz)

        Pxyz = read_files(sm)
        vsc = visc(path, Pxyz)
        eta_raw, eta_raw_sem, eta_theo, eta_theo_sem, amp_opts, tau_opts, y_log0s, dt_log = vsc.run_vsc(vol_sys=vol_sys,
                                                                                                      name=sm,
                                                                                                      segments=segments,
                                                                                                      iterations=iterations,
                                                                                                      n_boot=n_boot,
                                                                                                      dt_unit=dt_unit,
                                                                                                      n_point=n_point,
                                                                                                      n_tau=n_tau,
                                                                                                      T=T,
                                                                                                      seed=cfg["seed"])
        return eta_raw, eta_raw_sem, eta_theo, eta_theo_sem

    def define_pd(self):
        """Build the empty result DataFrames seeded with their row/column labels.

        Returns ``(df_sg, df_res_contact, df_acid_contact, df_res_count,
        df_acid_count)``: the per-system Quant_Data frame plus the residue/acid
        contact and count frames (pre-labelled with species/residue names and a
        column per SM tag) that are filled in by ``sg_sm_analysis_full``.
        """
        df_acid_contact = pd.DataFrame(columns=["Acid",
                                                "D1",
                                                "D2",
                                                "D3",
                                                "D4",
                                                "D5",
                                                "D6",
                                                "D7",
                                                "D8",
                                                "D9",
                                                "D10",
                                                "ND1",
                                                "ND2",
                                                "ND3",
                                                "ND4",
                                                "ND5",
                                                "ND6",
                                                "ND7",
                                                "ND8",
                                                "ND9",
                                                "ND10",
                                                ])

        df_acid_contact["Acid"] = ["Met",
                                   "Gly",
                                   "Lys",
                                   "Thr",
                                   "Arg",
                                   "Ala",
                                   "Asp",
                                   "Glu",
                                   "Tyr",
                                   "Val",
                                   "Leu",
                                   "Gln",
                                   "Trp",
                                   "Phe",
                                   "Ser",
                                   "His",
                                   "Asn",
                                   "Pro",
                                   "Cys",
                                   "Ile",
                                   "A",
                                   "C",
                                   "G",
                                   "U"]

        df_acid_count = df_acid_contact.copy()

        df_res_contact = pd.DataFrame(columns=["Residue",
                                               "D1",
                                               "D2",
                                               "D3",
                                               "D4",
                                               "D5",
                                               "D6",
                                               "D7",
                                               "D8",
                                               "D9",
                                               "D10",
                                               "ND1",
                                               "ND2",
                                               "ND3",
                                               "ND4",
                                               "ND5",
                                               "ND6",
                                               "ND7",
                                               "ND8",
                                               "ND9",
                                               "ND10",
                                               ])

        df_res_contact["Residue"] = ["G3BP1",
                                     "PABP1",
                                     "TTP",
                                     "TIA1",
                                     "TDP43",
                                     "FUS",
                                     "RNA"]

        df_res_count = df_res_contact.copy()

        df_sg = pd.DataFrame(columns=['Small Molecule ID',
                                   "Small Molecule Name",
                                    "Compound Name",
                                   "c_{SM}",
                                   "Compound Class",
                                   "D_Binary",
                                   "$Mass$ $(Da)$",
                                   "$c_{dense,SG,fit}$ $(mg/ml)$",
                                   "SIG$c_{dense,SG,fit}$ $(mg/ml)$",
                                   "$c_{dilute,SG,fit}$ $(mg/ml)$",
                                   "SIG$c_{dilute,SG,fit}$ $(mg/ml)$",
                                   "$c_{dense,SG,calc}$ $(mg/ml)$",
                                   "SIG$c_{dense,SG,calc}$ $(mg/ml)$",
                                   "$c_{dilute,SG,calc}$ $(mg/ml)$",
                                   "SIG$c_{dilute,SG,calc}$ $(mg/ml)$",
                                   "$P_{SG}$",
                                   "SIG$P_{SG}$",
                                   r"$R_{cond}$ $(\AA)$",
                                   r"SIG$R_{cond}$ $(\AA)$",
                                   r"$W_{interface}$ $(\AA)$",
                                   r"SIG$W_{interface}$ $(\AA)$",
                                   r"$\gamma_{1}$ $(mN/m)$",
                                   r"SIG$\gamma_{1}$ $(mN/m)$",
                                   r"$\gamma_{2}$ $(mN/m)$",
                                   r"SIG$\gamma_{2}$ $(mN/m)$",
                                   r"$\gamma_ave$ $(mN/m)$",
                                   r"SIG$\gamma_ave$ $(mN/m)$",
                                   r"$\Delta G_{trans}$ $(kJ/mol)$",
                                   r"SIG$\Delta G_{trans}$ $(kJ/mol)$",
                                   "$c_{dilute,SM}$ $(mg/ml)$",
                                   "$c_{dense,SM}$ $(mg/ml)$",
                                   "SIG$c_{dilute,SM}$ $(mg/ml)$",
                                   "SIG$c_{dense,SM}$ $(mg/ml)$",
                                   "$P_{SM}$",
                                   "SIG$P_{SM}$",
                                   r"$\phi_{D}$",
                                   r"SIG$\phi_{D}$",
                                   "$N_{D}$",
                                   "SIG$N_{D}$",
                                   "$R_{g}$",
                                   "SIG$R_{g}$",
                                   r"$\phi_{R}$",
                                   r"SIG$\phi_{R}$",
                                   r"$\eta_{GK}$ Pa s",
                                   r"SIG$\eta_{GK}$ Pa s"])

        return df_sg, df_res_contact, df_acid_contact, df_res_count, df_acid_count

    def sg_sm_analysis_full(self, path, folder, name, df_sg, mol_name, c, df_res_contact, df_acid_contact, df_res_count, df_acid_count, T, dt, tmin, tmax, diffusion_params=None, viscosity_params=None, species_to_analyze=None, precomputed_viscosity=None):
        """Run the full physical analysis for one system and append its result row.

        For system ``name`` (an SM tag such as ``sg_X``/``dsm_lipoamide`` or an
        aggregate ``DSM``/``NDSM``) this fits the SG/Protein/RNA/SM radial
        density profiles, derives dense/dilute concentrations, partition
        coefficients, condensate radius/interface width, surface tensions and
        transfer free energy; runs the two-stage confinement analysis and
        Green-Kubo viscosity (unless ``precomputed_viscosity`` is supplied);
        computes Stokes-Einstein diffusivities; and writes the standardized,
        difference, and SM contact-map CSVs.

        Args:
            path: Output analysis root (``{path}/{folder}_...``).
            folder: ``ANALYSIS_<cat>_AVE`` input subfolder for ``name``.
            name: System tag being analyzed.
            df_sg: Quant_Data frame to which the new row is appended.
            mol_name: Legacy molecule label (unused; kept for call compatibility).
            c: Small-molecule concentration label written to the row.
            df_res_contact, df_acid_contact, df_res_count, df_acid_count:
                Contact/count frames that gain a column for this SM.
            T, dt, tmin, tmax: Temperature (K) and analysis-window parameters.
            diffusion_params: Unused override hook (engine uses its own params).
            viscosity_params: Per-call viscosity overrides for ``calc_visc``.
            species_to_analyze: Unused override hook (kept for compatibility).
            precomputed_viscosity: Optional ``(eta, eta_sem, theo, theo_sem)`` to
                use instead of recomputing Green-Kubo viscosity (aggregate rows).

        Returns:
            ``(df_sg, df_res_contact, df_acid_contact, df_res_count,
            df_acid_count, fitterSG, fitterProtein, fitterRNA, fitterSM)`` — the
            updated frames followed by the four ``rdp`` fitters (``fitterSM`` is
            None for the SG control).
        """
        print("ANALYZING SYSTEM: {}".format(name))
        lab = ''.join(name.split("_STRESS")[0])

        sm_dict = {"sg_X": "SG",
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
                   "DSM": "DSM",
                   "NDSM": "NDSM"
                   }

        name_dict = {"sg_X": "SG",
                   "ndsm_dmso": "DMSO",
                   "ndsm_valeric_acid": "Valeric Acid",
                   "ndsm_ethylenediamine": "Ethylenediamine",
                   "ndsm_propanedithiol": "Propanedithiol",
                   "ndsm_hexanediol": "Hexanediol",
                   "ndsm_diethylaminopentane": "Diethylaminopentane",
                   "ndsm_aminoacridine": "Aminoacridine",
                   "ndsm_anthraquinone": "Anthraquinone",
                   "ndsm_acetylenapthacene": "Acetylenapthacene",
                   "ndsm_anacardic": "Anacardic Acid",

                   "dsm_hydroxyquinoline": "Hydroxyquinoline",
                   "dsm_lipoamide": "Lipoamide",
                   "dsm_lipoic_acid": "Lipoic Acid",
                   "dsm_dihydrolipoic_acid": "Dihydrolipoic Acid",
                   "dsm_anisomycin": "Anisomycin",
                   "dsm_pararosaniline": "Pararosaniline",
                   "dsm_pyrivinium": "Pyrivinium",
                   "dsm_quinicrine": "Quinicrine",
                   "dsm_mitoxantrone": "Mitoxantrone",
                   "dsm_daunorubicin": "Daunorubicin",
                   "DSM": "DSM",
                   "NDSM": "NDSM"
                   }

        sm_mass_dict = {"sg_X": 0,
                        "ndsm_dmso": 78.12922,
                        "ndsm_valeric_acid": 102.13350,
                        "ndsm_ethylenediamine": 104.15244,
                        "ndsm_propanedithiol": 108.21676,
                        "ndsm_hexanediol": 118.17638,
                        "ndsm_diethylaminopentane": 158.28774,
                        "ndsm_aminoacridine": 194.23610,
                        "ndsm_anthraquinone": 240.21536,
                        "ndsm_acetylenapthacene": 336.34452,
                        "ndsm_anacardic": 348.52712,

                        "dsm_hydroxyquinoline": 145.16089,
                        "dsm_lipoamide": 205.33365,
                        "dsm_lipoic_acid": 206.31838,
                        "dsm_dihydrolipoic_acid": 208.33432,
                        "dsm_anisomycin": 265.30973,
                        "dsm_pararosaniline": 287.36459,
                        "dsm_pyrivinium": 382.52926,
                        "dsm_quinicrine": 399.96460,
                        "dsm_mitoxantrone": 444.48836,
                        "dsm_daunorubicin": 527.52883,
                        "DSM": (145.16089+205.33365+206.31838+208.33432+265.30973+287.36459+382.52926+399.96460+444.48836+527.52883)/10,
                        "NDSM": (78.12922+102.13350+104.15244+108.21676+118.17638+158.28774+194.23610+240.21536+336.34452+348.52712)/10
                        }

        if name.split("_")[0] == "ndsm" or name=="NDSM":
            effect = "Non-Dissolving"
            d_bin = 0
        elif name.split("_")[0] == "dsm" or name=="DSM":
            effect = "Dissolving"
            d_bin = 1
        else:
            effect = "X"
            d_bin = 0

        name_sm = sm_dict[name]

        # Phase Analysis
        print("Stress Granule ANALYSIS")
        label = "SG"
        density_file_SG = "{}/{}/Density_Profile_SG_{}.csv".format(path, folder, lab)
        pca_file_SG = "{}/{}/PCA_SG_{}.csv".format(path, folder, lab)
        cluster_file_SG = "{}/{}/Cluster_SG_{}.csv".format(path, folder, lab)
        init = [250, 250, 200, 80]
        fitterSG = rdp(density_file_SG, pca_file_SG, cluster_file_SG, T, init, label)

        print("Protein ANALYSIS")
        label = "Protein"
        density_file_Protein = "{}/{}/Density_Profile_Protein_{}.csv".format(path, folder, lab)
        pca_file_Protein = "{}/{}/PCA_Protein_{}.csv".format(path, folder, lab)
        cluster_file_Protein = "{}/{}/Cluster_Protein_{}.csv".format(path, folder, lab)
        init = [150, 150, 200, 80]
        fitterProtein = rdp(density_file_Protein, pca_file_Protein, cluster_file_Protein, T, init, label)

        print("RNA ANALYSIS")
        label = "RNA"
        density_file_RNA = "{}/{}/Density_Profile_RNA_{}.csv".format(path, folder, lab)
        pca_file_RNA = "{}/{}/PCA_RNA_{}.csv".format(path, folder, lab)
        cluster_file_RNA = "{}/{}/Cluster_RNA_{}.csv".format(path, folder, lab)
        init = [80, 80, 200, 80]
        fitterRNA = rdp(density_file_RNA, pca_file_RNA, cluster_file_RNA, T, init, label)

        if effect != 'X':
            print("{} ANALYSIS".format(lab))
            label = "SM"
            density_file_SM = "{}/{}/Density_Profile_SM_{}.csv".format(path, folder, lab)
            init = [0, 0.2, 200, 50]
            fitterSM = rdp(density_file_SM, pca_file_SG, cluster_file_SG, T, init, label)
            c_dilute_sm = fitterSM.fit_rho[-1]
            c_dense_sm = fitterSM.fit_rho[0]

            c_dilute_sm_se = fitterSM.c_dilute_fit_se
            c_dense_sm_se = fitterSM.c_dense_fit_se

            if fitterSM.c_dilute_fit != 0:
                pc = np.abs(fitterSM.fit_rho[0] / fitterSM.fit_rho[-1])
                pc_se = np.sqrt((1 / c_dilute_sm) ** 2 * c_dense_sm_se ** 2 + (
                        -c_dense_sm / c_dilute_sm ** 2) ** 2 * c_dilute_sm_se ** 2)
            else:
                pc = 1
                pc_se = 0

        else:
            fitterSM = None
            c_dilute_sm = 0
            c_dense_sm = 0
            c_dilute_sm_se = 0
            c_dense_sm_se = 0
            pc = 0
            pc_se = 0

        c_dense_sg_fit = np.abs(fitterSG.c_dense_fit)
        c_dense_sg_fit_se = fitterSG.c_dense_fit_se
        c_dilute_sg_fit = np.abs(fitterSG.c_dilute_fit)
        c_dilute_sg_fit_se = fitterSG.c_dilute_fit_se
        c_dense_sg_calc = np.abs(fitterSG.c_dense_calc)
        c_dense_sg_calc_se = fitterSG.c_dense_calc_se
        c_dilute_sg_calc = np.abs(fitterSG.c_dilute_calc)
        c_dilute_sg_calc_se = fitterSG.c_dilute_calc_se

        p_SG = np.abs(c_dense_sg_fit) / np.abs(c_dilute_sg_fit)
        sig_p_SG = np.sqrt((1 / c_dilute_sg_fit) ** 2 * c_dense_sg_fit_se ** 2 + (
                -c_dense_sg_fit / c_dilute_sg_fit ** 2) ** 2 * c_dilute_sg_fit_se ** 2)

        R_droplet = np.abs(fitterSG.fit_R)
        sig_R = np.abs(fitterSG.se_R)
        W = np.abs(fitterSG.fit_W)
        sig_W = np.abs(fitterSG.se_W)

        st1 = fitterSG.st_1
        st2 = fitterSG.st_2
        st1_se = fitterSG.st_1_se
        st2_se = fitterSG.st_2_se
        st = fitterSG.st
        st_se = fitterSG.st_se
        dG = fitterSG.dG / 1000
        se_dG = np.abs(fitterSG.se_dG) / 1000

        # Clustering Analysis
        df_analysis_cluster = pd.read_csv("{}/{}/Cluster_SG_{}.csv".format(path, folder, lab))

        phi_d = float(df_analysis_cluster["Mass of Largest Droplet (mg)"].iloc[0]) / float(
            df_analysis_cluster["Total Mass (mg)"].iloc[0])
        num_clus = float(df_analysis_cluster["Number of Droplets"].iloc[0])
        rg = float(df_analysis_cluster["Largest Droplet Radius of Gyration"].iloc[0])

        phi_d_std = np.sqrt((1 / float(df_analysis_cluster["Total Mass (mg)"].iloc[0])) ** 2 * float(
            df_analysis_cluster["Mass Largest SEM"].iloc[0]) ** 2)
        num_clus_std = float(df_analysis_cluster["ND SEM"].iloc[0])
        rg_std = float(df_analysis_cluster["RG SEM"].iloc[0])

        print("PHI: " + str(phi_d) + " ± " + str(phi_d_std))
        print("CLUS: " + str(num_clus) + " ± " + str(num_clus_std))
        print("RG: " + str(rg) + " ± " + str(rg_std))

        phi_R = R_droplet / rg
        phi_R_std = np.sqrt((1 / rg) ** 2 * sig_R ** 2 + (-R_droplet / rg ** 2) ** 2 * rg_std ** 2)

        # Two-stage confinement analysis:
        #   Stage 1: l_conf from plateau detection (primary, robust)
        #   Stage 2: tau_conf from linearized approach-to-plateau (conditional)
        #   D_cage = l^2/(6*tau) only if both stages pass
        conf = self.calc_diffusion_confined(path, name, folder, dt, T, tmin, tmax)
        conf_stage = conf.get("confinement_stage", "failed")
        l_conf = conf["l_A_mean"]
        l_conf_sme = conf["l_A_sem"]
        tau_conf_s = conf["tau_s_mean"]
        tau_conf_sme_s = conf.get("tau_s_sem", math.nan)
        tau_conf_ns = tau_conf_s * 1e9 if np.isfinite(tau_conf_s) else math.nan
        tau_conf_sme_ns = tau_conf_sme_s * 1e9 if np.isfinite(tau_conf_sme_s) else math.nan
        D_cage_m2s = conf["D_cage_m2_s"]
        D_cage_sme_m2s = conf["D_cage_m2_s_sem"]
        D_cage_um2s = D_cage_m2s * 1e12 if np.isfinite(D_cage_m2s) else math.nan
        D_cage_sme_um2s = D_cage_sme_m2s * 1e12 if np.isfinite(D_cage_sme_m2s) else math.nan
        D_loglog_m2s = conf.get("D_loglog_m2_s", math.nan)
        D_loglog_sme_m2s = conf.get("D_loglog_m2_s_sem", math.nan)
        D_loglog_um2s = D_loglog_m2s * 1e12 if np.isfinite(D_loglog_m2s) else math.nan
        D_loglog_sme_um2s = D_loglog_sme_m2s * 1e12 if np.isfinite(D_loglog_sme_m2s) else math.nan
        Rg_conf_mean = conf["Rg_A_mean"]
        Rg_conf_sem = conf["Rg_A_sem"]
        Rh_conf_mean = conf["Rh_A_mean"]
        Rh_conf_sem = conf["Rh_A_sem"]
        # Chain heterogeneity medians (NaN if < 5 chains passed per-chain fits)
        l_conf_chain_med = conf.get("l_A_chain_median", math.nan)
        tau_conf_chain_med_s = conf.get("tau_s_chain_median", math.nan)
        tau_conf_chain_med_ns = tau_conf_chain_med_s * 1e9 if np.isfinite(tau_conf_chain_med_s) else math.nan
        D_cage_chain_med_m2s = conf.get("D_cage_m2_s_chain_median", math.nan)
        D_cage_chain_med_um2s = D_cage_chain_med_m2s * 1e12 if np.isfinite(D_cage_chain_med_m2s) else math.nan

        # Per-species confinement (comoving-frame pure model)
        # D_SE,GK per species is computed later (after eta_E is available)
        per_species_conf_columns = {}
        for sp in self._CONFINED_SPECIES:
            l_sp = conf.get(f"l_conf_{sp}_A_mean", math.nan)
            l_sp_sem = conf.get(f"l_conf_{sp}_A_sem", math.nan)
            tau_sp_s = conf.get(f"tau_conf_{sp}_s_mean", math.nan)
            tau_sp_sem_s = conf.get(f"tau_conf_{sp}_s_sem", math.nan)
            tau_sp_ns = tau_sp_s * 1e9 if np.isfinite(tau_sp_s) else math.nan
            tau_sp_sem_ns = tau_sp_sem_s * 1e9 if np.isfinite(tau_sp_sem_s) else math.nan
            rg_sp = conf.get(f"Rg_{sp}_A_mean", math.nan)
            rg_sp_sem = conf.get(f"Rg_{sp}_A_sem", math.nan)
            rh_sp = conf.get(f"Rh_{sp}_A_mean", math.nan)
            rh_sp_sem = conf.get(f"Rh_{sp}_A_sem", math.nan)
            occ_sp = conf.get(f"Occ_{sp}_mean", math.nan)
            occ_sp_sem = conf.get(f"Occ_{sp}_sem", math.nan)
            rr_sp = conf.get(f"r_over_R_{sp}_mean", math.nan)
            rr_sp_sem = conf.get(f"r_over_R_{sp}_sem", math.nan)
            per_species_conf_columns[f"$l_{{conf,{sp}}}$ A"] = l_sp
            per_species_conf_columns[f"SIG$l_{{conf,{sp}}}$ A"] = l_sp_sem
            per_species_conf_columns[f"$\\tau_{{conf,{sp}}}$ $ns$"] = tau_sp_ns
            per_species_conf_columns[f"SIG$\\tau_{{conf,{sp}}}$ $ns$"] = tau_sp_sem_ns
            per_species_conf_columns[f"$R_{{g,{sp}}}$ A"] = rg_sp
            per_species_conf_columns[f"SIG$R_{{g,{sp}}}$ A"] = rg_sp_sem
            per_species_conf_columns[f"$R_{{h,{sp}}}$ A"] = rh_sp
            per_species_conf_columns[f"SIG$R_{{h,{sp}}}$ A"] = rh_sp_sem
            per_species_conf_columns[f"$Occ_{{{sp}}}$"] = occ_sp
            per_species_conf_columns[f"SIG$Occ_{{{sp}}}$"] = occ_sp_sem
            per_species_conf_columns[f"$r/R_{{{sp}}}$"] = rr_sp
            per_species_conf_columns[f"SIG$r/R_{{{sp}}}$"] = rr_sp_sem
            # Store Rh for per-species D_SE,GK computation after eta_E is available
            per_species_conf_columns[f"_Rh_{sp}_A"] = rh_sp
            per_species_conf_columns[f"_Rh_{sp}_A_sem"] = rh_sp_sem

        print(f"Confinement (two-stage, {conf_stage}) for {name}:")
        print(f"  l_conf   = {l_conf:.2f} ± {l_conf_sme:.2f} Å  (plateau, CV={conf.get('plateau_cv', math.nan):.4f})")
        if np.isfinite(tau_conf_ns):
            tau_sem_str = f" ± {tau_conf_sme_ns:.1f}" if np.isfinite(tau_conf_sme_ns) else ""
            print(f"  tau_cage = {tau_conf_ns:.1f}{tau_sem_str} ns  (threshold crossing + bootstrap SEM)")
            print(f"  D_cage   = {D_cage_um2s:.3e} µm²/s  (local cage mobility)")
        else:
            print(f"  tau_cage = NaN (MSD never reached 0.632·l²)")
            print(f"  D_cage   = NaN")
        if np.isfinite(D_loglog_um2s):
            print(f"  D_loglog = {D_loglog_um2s:.3e} µm²/s  (V1/V2 log-log slope detection)")
        else:
            print(f"  D_loglog = NaN  (no diffusive window in log-log)")
        print(f"  chains   = {conf.get('n_confined', 0)}/{conf.get('n_total_inside', 0)} passed per-chain fits")

        # Viscosity Analysis
        # Use droplet radius from RDP fit (R + 0.5 W) in Angstroms
        r = float(fitterSG.fit_R + 0.5 * fitterSG.fit_W)

        if precomputed_viscosity is not None:
            eta_E, eta_E_sme, eta_theo, eta_theo_sme = precomputed_viscosity
        else:
            eta_E, eta_E_sme, eta_theo, eta_theo_sme = self.calc_visc(
                path,
                folder,
                name,
                r,
                tmin,
                tmax,
                T,
                viscosity_params=viscosity_params,
                dt_window=dt,
            )
        self.viscosity_cache[name] = (
            float(eta_E),
            float(eta_E_sme),
            float(eta_theo),
            float(eta_theo_sme),
        )

        print(f"Viscosity GK (Eta_Raw): {eta_E:.4e} ± {eta_E_sme:.4e} Pa·s")

        # Stokes-Einstein inferred diffusivity from eta_GK
        # D_{SE,GK,Rh} is the primary inferred condensate diffusivity.
        # D_{SE,GK,Rg} is a secondary sensitivity estimate.
        D_SE_GK_Rg, D_SE_GK_Rg_sem = _stokes_einstein_D(eta_E, eta_E_sme, Rg_conf_mean, Rg_conf_sem, T)
        D_SE_GK_Rh, D_SE_GK_Rh_sem = _stokes_einstein_D(eta_E, eta_E_sme, Rh_conf_mean, Rh_conf_sem, T)
        D_SE_GK_Rg_um2s = D_SE_GK_Rg * 1e12 if np.isfinite(D_SE_GK_Rg) else math.nan
        D_SE_GK_Rg_sme_um2s = D_SE_GK_Rg_sem * 1e12 if np.isfinite(D_SE_GK_Rg_sem) else math.nan
        D_SE_GK_Rh_um2s = D_SE_GK_Rh * 1e12 if np.isfinite(D_SE_GK_Rh) else math.nan
        D_SE_GK_Rh_sme_um2s = D_SE_GK_Rh_sem * 1e12 if np.isfinite(D_SE_GK_Rh_sem) else math.nan
        print(f"  D_SE,GK(Rh) = {D_SE_GK_Rh_um2s:.3e} µm²/s  (Stokes-Einstein with η_GK={eta_E:.3e}, Rh={Rh_conf_mean:.1f} Å)")
        print(f"  D_SE,GK(Rg) = {D_SE_GK_Rg_um2s:.3e} µm²/s  (sensitivity est. with Rg={Rg_conf_mean:.1f} Å)")

        # Per-species D_SE,GK = kT/(6πηRh_species) — Stokes-Einstein per species
        for sp in self._CONFINED_SPECIES:
            rh_sp = per_species_conf_columns.pop(f"_Rh_{sp}_A", math.nan)
            rh_sp_sem = per_species_conf_columns.pop(f"_Rh_{sp}_A_sem", math.nan)
            D_sp, D_sp_sem = _stokes_einstein_D(eta_E, eta_E_sme, rh_sp, rh_sp_sem, T)
            D_sp_um2s = D_sp * 1e12 if np.isfinite(D_sp) else math.nan
            D_sp_sme_um2s = D_sp_sem * 1e12 if np.isfinite(D_sp_sem) else math.nan
            per_species_conf_columns[f"$D_{{SE,GK,{sp}}}$ $\\mu m^{{2}} / s$"] = D_sp_um2s
            per_species_conf_columns[f"SIG$D_{{SE,GK,{sp}}}$ $\\mu m^{{2}} / s$"] = D_sp_sme_um2s

        # Surface Tension for Protein and RNA subsystems
        st1_protein = fitterProtein.st_1
        st2_protein = fitterProtein.st_2
        st1_se_protein = fitterProtein.st_1_se
        st2_se_protein = fitterProtein.st_2_se
        st_protein = fitterProtein.st
        st_se_protein = fitterProtein.st_se
        
        st1_rna = fitterRNA.st_1
        st2_rna = fitterRNA.st_2
        st1_se_rna = fitterRNA.st_1_se
        st2_se_rna = fitterRNA.st_2_se
        st_rna = fitterRNA.st
        st_se_rna = fitterRNA.st_se
        
        print("Surface Tension Protein (gamma_ave): " + str(st_protein * 1000) + " ± " + str(st_se_protein * 1000) + " mN/m")
        print("Surface Tension RNA (gamma_ave): " + str(st_rna * 1000) + " ± " + str(st_se_rna * 1000) + " mN/m")

        # Contact Analysis
        # SG Residue Contact Map
        res_contact_file = "{}/{}/Residue_Contacts_Mean_{}.csv".format(path, folder, name)
        arr = np.loadtxt(res_contact_file, delimiter=",", dtype=float)
        arr_sem = _read_contact_matrix("{}/{}/Residue_Contacts_SEM_{}.csv".format(path, folder, name))
        _write_residue_contact_map(path, name, "Residue_Contacts_Standardized_Mean", arr)
        if arr_sem is not None:
            _write_residue_contact_map(path, name, "Residue_Contacts_Standardized_SEM", arr_sem, one_d=_contact_1d_sem(arr_sem))

        # SG Acid Contact Map
        res_contact_file = "{}/{}/Acid_Contacts_Mean_{}.csv".format(path, folder, name)
        arr = np.loadtxt(res_contact_file, delimiter=",", dtype=float)
        arr_sem = _read_contact_matrix("{}/{}/Acid_Contacts_SEM_{}.csv".format(path, folder, name))
        _write_acid_contact_map(path, name, "Acid_Contacts_Standardized_Mean", arr)
        if arr_sem is not None:
            _write_acid_contact_map(path, name, "Acid_Contacts_Standardized_SEM", arr_sem, one_d=_contact_1d_sem(arr_sem))

        if "X" not in name:
            sg_res_contact_file = "{}/ANALYSIS_SG_AVE/Residue_Contacts_Mean_sg_X.csv".format(path)
            sg_res_arr = np.loadtxt(sg_res_contact_file, delimiter=",", dtype=float)
            sg_res_sem = _read_contact_matrix("{}/ANALYSIS_SG_AVE/Residue_Contacts_SEM_sg_X.csv".format(path))
            sg_acid_contact_file = "{}/ANALYSIS_SG_AVE/Acid_Contacts_Mean_sg_X.csv".format(path)
            sg_acid_arr = np.loadtxt(sg_acid_contact_file, delimiter=",", dtype=float)
            sg_acid_sem = _read_contact_matrix("{}/ANALYSIS_SG_AVE/Acid_Contacts_SEM_sg_X.csv".format(path))

            # SG SM Difference Contact Map
            sm_res_contact_file = "{}/{}/Residue_Contacts_Mean_{}.csv".format(path, folder, name)
            sm_res_arr = np.loadtxt(sm_res_contact_file, delimiter=",", dtype=float)
            sm_res_sem = _read_contact_matrix("{}/{}/Residue_Contacts_SEM_{}.csv".format(path, folder, name))

            arr = (sm_res_arr - sg_res_arr)/sg_res_arr
            _write_residue_contact_map(path, name, "Residue_Contacts_Difference_Mean", arr)
            arr_sem = _ratio_difference_sem(sm_res_arr, sm_res_sem, sg_res_arr, sg_res_sem)
            if arr_sem is not None:
                _write_residue_contact_map(path, name, "Residue_Contacts_Difference_SEM", arr_sem, one_d=_contact_1d_sem(arr_sem))

            # SG SM Acid Difference Contact Map
            sm_acid_contact_file = "{}/{}/Acid_Contacts_Mean_{}.csv".format(path, folder, name)
            sm_acid_arr = np.loadtxt(sm_acid_contact_file, delimiter=",", dtype=float)
            sm_acid_sem = _read_contact_matrix("{}/{}/Acid_Contacts_SEM_{}.csv".format(path, folder, name))
            arr = (sm_acid_arr - sg_acid_arr) / sg_acid_arr
            _write_acid_contact_map(path, name, "Acid_Contacts_Difference_Mean", arr)
            arr_sem = _ratio_difference_sem(sm_acid_arr, sm_acid_sem, sg_acid_arr, sg_acid_sem)
            if arr_sem is not None:
                _write_acid_contact_map(path, name, "Acid_Contacts_Difference_SEM", arr_sem, one_d=_contact_1d_sem(arr_sem))

            # SM Residue Contact Map
            res_contact_file = "{}/{}/Residue_SM_Contacts_Mean_{}.csv".format(path, folder, name)
            arr = np.loadtxt(res_contact_file, delimiter=",", dtype=float)
            df_res_contact[name_sm] = arr
            arr_sem = _read_contact_matrix("{}/{}/Residue_SM_Contacts_SEM_{}.csv".format(path, folder, name))
            if arr_sem is not None:
                _write_contact_sem_map("{}/RESULTS/SM_CONTACT_MAPS/SM_Residue_Contacts_SEM_{}.csv".format(path, name), arr_sem)

            # SG SM Acid Contact Map
            acid_contact_file = "{}/{}/Acid_SM_Contacts_Mean_{}.csv".format(path, folder, name)
            arr = np.loadtxt(acid_contact_file, delimiter=",", dtype=float)
            df_acid_contact[name_sm] = arr
            arr_sem = _read_contact_matrix("{}/{}/Acid_SM_Contacts_SEM_{}.csv".format(path, folder, name))
            if arr_sem is not None:
                _write_contact_sem_map("{}/RESULTS/SM_CONTACT_MAPS/SM_Acid_Contacts_SEM_{}.csv".format(path, name), arr_sem)
                np.savetxt(
                    "{}/RESULTS/SM_CONTACT_MAPS/SM_Acid_Contacts_{}_SEM.csv".format(path, name),
                    _prepare_acid_rows(arr_sem),
                    delimiter=",",
                )

            # SM Residue Count
            res_count_file = "{}/{}/BioNum_{}.csv".format(path, folder, name)
            arr = np.loadtxt(res_count_file, delimiter=",", dtype=float)
            df_res_count[name_sm] = arr

            # SM Acid Count Map
            acid_count_file = "{}/{}/AcidNum_{}.csv".format(path, folder, name)
            arr = np.loadtxt(acid_count_file, delimiter=",", dtype=float)
            df_acid_count[name_sm] = arr

        diffusion_qc_columns = self._build_diffusion_qc_columns(conf)

        # Create SM DataFrame Row
        tempDF = pd.DataFrame([{'Small Molecule ID': sm_dict[name],
                                "Small Molecule Name": name,
                                "Compound Name": name_dict[name],
                                "c_{SM}": c,
                                "Compound Class": effect,
                                "D_Binary": d_bin,
                                "$Mass$ $(Da)$": sm_mass_dict[name],
                                "$c_{dense,SG,fit}$ $(mg/ml)$": c_dense_sg_fit,
                                "SIG$c_{dense,SG,fit}$ $(mg/ml)$": c_dense_sg_fit_se,
                                "$c_{dilute,SG,fit}$ $(mg/ml)$": c_dilute_sg_fit,
                                "SIG$c_{dilute,SG,fit}$ $(mg/ml)$": c_dilute_sg_fit_se,
                                "$c_{dense,SG,calc}$ $(mg/ml)$": c_dense_sg_calc,
                                "SIG$c_{dense,SG,calc}$ $(mg/ml)$": c_dense_sg_calc_se,
                                "$c_{dilute,SG,calc}$ $(mg/ml)$": c_dilute_sg_calc,
                                "SIG$c_{dilute,SG,calc}$ $(mg/ml)$": c_dilute_sg_calc_se,
                                "$P_{SG}$": p_SG,
                                "SIG$P_{SG}$": sig_p_SG,
                                r"$R_{cond}$ $(\AA)$": R_droplet,
                                r"SIG$R_{cond}$ $(\AA)$": sig_R,
                                r"$W_{interface}$ $(\AA)$": W,
                                r"SIG$W_{interface}$ $(\AA)$": sig_W,
                                r"$\gamma_{1}$ $(mN/m)$": st1 * 1000,
                                r"SIG$\gamma_{1}$ $(mN/m)$": st1_se * 1000,
                                r"$\gamma_{2}$ $(mN/m)$": st2 * 1000,
                                r"SIG$\gamma_{2}$ $(mN/m)$": st2_se * 1000,
                                r"$\gamma_ave$ $(mN/m)$": st * 1000,
                                r"SIG$\gamma_ave$ $(mN/m)$": st_se * 1000,
                                r"$\Delta G_{trans}$ $(kJ/mol)$": dG,
                                r"SIG$\Delta G_{trans}$ $(kJ/mol)$": se_dG,
                                "$c_{dilute,SM}$ $(mg/ml)$": c_dilute_sm,
                                "$c_{dense,SM}$ $(mg/ml)$": c_dense_sm,
                                "SIG$c_{dilute,SM}$ $(mg/ml)$": c_dilute_sm_se,
                                "SIG$c_{dense,SM}$ $(mg/ml)$": c_dense_sm_se,
                                "$P_{SM}$": pc,
                                "SIG$P_{SM}$": pc_se,
                                r"$\phi_{D}$": phi_d,
                                r"SIG$\phi_{D}$": phi_d_std,
                                "$N_{D}$": num_clus,
                                "SIG$N_{D}$": num_clus_std,
                                "$R_{g}$": rg,
                                "SIG$R_{g}$": rg_std,
                                r"$\phi_{R}$": phi_R,
                                r"SIG$\phi_{R}$": phi_R_std,
                                r"$D_{SE,GK,Rh}$ $\mu m^{2} / s$": D_SE_GK_Rh_um2s,
                                r"SIG$D_{SE,GK,Rh}$ $\mu m^{2} / s$": D_SE_GK_Rh_sme_um2s,
                                r"$D_{SE,GK,Rg}$ $\mu m^{2} / s$": D_SE_GK_Rg_um2s,
                                r"SIG$D_{SE,GK,Rg}$ $\mu m^{2} / s$": D_SE_GK_Rg_sme_um2s,
                                r"$l_{conf}$ A": l_conf,
                                r"SIG$l_{conf}$ A": l_conf_sme,
                                r"$\tau_{cage}$ $ns$": tau_conf_ns,
                                r"SIG$\tau_{cage}$ $ns$": tau_conf_sme_ns,
                                r"$D_{cage}$ $\mu m^{2} / s$": D_cage_um2s,
                                r"SIG$D_{cage}$ $\mu m^{2} / s$": D_cage_sme_um2s,
                                r"$D_{loglog}$ $\mu m^{2} / s$": D_loglog_um2s,
                                r"SIG$D_{loglog}$ $\mu m^{2} / s$": D_loglog_sme_um2s,
                                r"$\eta_{GK}$ Pa s": eta_E,
                                r"SIG$\eta_{GK}$ Pa s": eta_E_sme,
                                r"$\gamma_{1,Protein}$ $(mN/m)$": st1_protein * 1000,
                                r"SIG$\gamma_{1,Protein}$ $(mN/m)$": st1_se_protein * 1000,
                                r"$\gamma_{2,Protein}$ $(mN/m)$": st2_protein * 1000,
                                r"SIG$\gamma_{2,Protein}$ $(mN/m)$": st2_se_protein * 1000,
                                r"$\gamma_{ave,Protein}$ $(mN/m)$": st_protein * 1000,
                                r"SIG$\gamma_{ave,Protein}$ $(mN/m)$": st_se_protein * 1000,
                                r"$\gamma_{1,RNA}$ $(mN/m)$": st1_rna * 1000,
                                r"SIG$\gamma_{1,RNA}$ $(mN/m)$": st1_se_rna * 1000,
                                r"$\gamma_{2,RNA}$ $(mN/m)$": st2_rna * 1000,
                                r"SIG$\gamma_{2,RNA}$ $(mN/m)$": st2_se_rna * 1000,
                                r"$\gamma_{ave,RNA}$ $(mN/m)$": st_rna * 1000,
                                r"SIG$\gamma_{ave,RNA}$ $(mN/m)$": st_se_rna * 1000,
                                }])

        # Include tail-completed theoretical viscosity columns in the row
        tempDF[r"$\eta_{GK Theo}$ Pa s"] = eta_theo
        tempDF[r"SIG$\eta_{GK Theo}$ Pa s"] = eta_theo_sme
        for column_name, value in per_species_conf_columns.items():
            tempDF[column_name] = value
        for column_name, value in diffusion_qc_columns.items():
            tempDF[column_name] = value
        df_sg = pd.concat([df_sg, tempDF], ignore_index=True)

        return df_sg, df_res_contact, df_acid_contact, df_res_count, df_acid_count, fitterSG, fitterProtein, fitterRNA, fitterSM


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Perform system-level analysis (diffusion, viscosity, contacts)')
    parser.add_argument('--path', required=True, help='Path to TEMP_XXX directory (e.g., TEMP_300)')
    parser.add_argument('--folder', required=True, help='Output folder prefix (e.g., CLASSIFY)')
    parser.add_argument('--T', type=int, required=True, help='Simulation temperature in Kelvin')
    parser.add_argument('--dt', type=int, required=True, help='Cluster time stride (ns)')
    parser.add_argument('--tmin', type=int, required=True, help='Start of analysis window (ns)')
    parser.add_argument('--tmax', type=int, required=True, help='End of analysis window (ns)')
    parser.add_argument('--c', type=float, default=1.0, help='Small molecule concentration label (default: 1.0)')
    parser.add_argument('--species', nargs='+', default=None, help='Subset of species for diffusion (default: all)')
    parser.add_argument('--species-all', nargs='+', default=None, help='Subset of species for global ALL diffusion averages (default: all analyzed species)')
    # Diffusion controls
    parser.add_argument('--diff-downsample', type=int, default=None, help='Downsample stride for MSD (default: 1)')
    parser.add_argument('--diff-dt-diff', type=float, default=None, help='Optional MSD coarse-graining width in ns before fitting (default: none; use native lag grid)')
    parser.add_argument('--diff-segments', type=int, default=None, help='Number of contiguous diffusion segments for RCC window bootstrap summaries (default: 8)')
    parser.add_argument('--diff-n-boot', type=int, default=None, help='Number of diffusion bootstrap resamples over segment means (default: 10)')
    parser.add_argument('--diff-slope-iters', type=int, default=None, help='Bootstrap iterations for diffusion fits (default: 200)')
    parser.add_argument('--diff-dt-pts', type=int, default=None, help='Log–log window length for diffusion (default: 5)')
    parser.add_argument('--diff-slope-tol', type=float, default=None, help='Slope tolerance for diffusion windows (default: 0.2)')
    parser.add_argument('--diff-min-diff-pts', type=int, default=None, help='Minimum points per diffusion window (default: 5)')
    parser.add_argument('--diff-slope-plateau', type=float, default=None, help='Slope tolerance for plateau windows (default: 0.2)')
    parser.add_argument('--diff-min-plateau-pts', type=int, default=None, help='Minimum points per plateau window (default: 5)')
    parser.add_argument('--diff-boot-r2', type=float, default=None, help='Minimum |r| for accepted diffusion fits (default: 0.8)')
    parser.add_argument('--diff-smooth-window', type=int, default=None, help='Savitzky-Golay window for MSD smoothing (default: 7)')
    parser.add_argument('--diff-smooth-polyorder', type=int, default=None, help='Savitzky-Golay polynomial order (default: 2)')
    parser.add_argument('--diff-seed', type=int, default=None, help='Bootstrap RNG seed for diffusion (default: 0)')
    parser.add_argument('--diff-min-segment-ns', type=float, default=None, help='Minimum segment duration for segmented diffusion (default: 200)')
    parser.add_argument('--diff-min-primary-fraction', type=float, default=None, help='Minimum finite-D fraction required for a primary diffusion summary (default: 0.25)')
    parser.add_argument('--diff-origin-resample', type=int, choices=[0, 1], default=None, help='Enable time-origin confinement resampling for MSD fits (1/0; default: 1)')
    parser.add_argument('--diff-origin-window-ns', type=float, default=None, help='Target subtrajectory duration for time-origin confinement resampling (default: 1000 ns)')
    parser.add_argument('--diff-origin-stride-ns', type=float, default=None, help='Spacing between time origins for confinement resampling (default: 200 ns)')
    parser.add_argument('--diff-origin-candidate-windows-ns', type=str, default=None, help='Comma-separated candidate subtrajectory durations for plateau-stability scan (default: 500,750,1000,1250,1500)')
    parser.add_argument('--diff-origin-min-success-fraction', type=float, default=None, help='Minimum successful-origin fraction for an accepted plateau block length (default: 0.70)')
    parser.add_argument('--diff-origin-min-success-count', type=int, default=None, help='Minimum successful origins for an accepted plateau block length (default: 3)')
    parser.add_argument('--diff-origin-max-origins', type=int, default=None, help='Maximum origins tested per candidate duration to cap runtime (default: 12)')
    # Viscosity controls
    parser.add_argument('--visc-segments', type=int, default=None, help='Number of segments for stress bootstrapping (default: 20)')
    parser.add_argument('--visc-iterations', type=int, default=None, help='Iterations per bootstrap segment (default: 10)')
    parser.add_argument('--visc-n-boot', type=int, default=None, help='Number of viscosity bootstraps (default: 10)')
    parser.add_argument('--visc-dt-unit', type=float, default=None, help='Time spacing between stress samples in seconds (default: 2e-9)')
    parser.add_argument('--visc-n-point', type=int, default=None, help='Number of log-space points for fits (default: 1000)')
    parser.add_argument('--visc-n-tau', type=int, default=None, help='Maximum Maxwell modes (default: 10)')
    parser.add_argument('--visc-seed', type=int, default=None, help='Bootstrap RNG seed for viscosity (default: 0)')

    args = parser.parse_args()

    # Validate path exists
    if not os.path.exists(args.path):
        print(f"Error: Path {args.path} does not exist")
        sys.exit(1)

    temp_root = os.path.abspath(args.path)

    # Check for input directories and warn if missing
    for cat in ['SG', 'DSM', 'NDSM']:
        input_dir = os.path.join(temp_root, f"ANALYSIS_{cat}")
        if not os.path.exists(input_dir):
            print(f"Warning: {input_dir} not found - {cat} analysis will be skipped")

    # Change to TEMP_* directory
    os.chdir(temp_root)

    # Output directory: {path}/{folder}_{T}_{dt}_{tmin}_{tmax}
    path = os.path.join(temp_root, f"{args.folder}_{args.T}_{args.dt}_{args.tmin}_{args.tmax}")
    dt = args.dt
    tmin = args.tmin
    tmax = args.tmax
    T = args.T  # Use temperature from command line for all physics calculations

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
    analysis = system_analysis(
        diffusion_params=diffusion_overrides,
        viscosity_params=viscosity_overrides,
        species_to_analyze=species_override,
        species_for_global=species_all_override,
        data_root=temp_root,
    )

    # Calculate Cluster, Phase, Dynamic Properties for a single SM System
    # Run Analysis

    c = args.c
    # Only create folders if they don't exist (preserve existing data from AVERAGE_SIMULATIONS)
    all_folders = [
        "IMAGES", "FIGURES", "RESULTS",
        "IMAGES/RDP", "IMAGES/RESIDUE_CONTACT_MAPS", "IMAGES/ACID_CONTACT_MAPS", "IMAGES/SM_CONTACT_MAPS", "IMAGES/DYNAMICS",
        "FIGURES/RDP", "FIGURES/RESIDUE_CONTACT_MAPS", "FIGURES/ACID_CONTACT_MAPS", "FIGURES/SM_CONTACT_MAPS", "FIGURES/PROPERTIES", "FIGURES/TIME",
        "RESULTS/RDP", "RESULTS/RESIDUE_CONTACT_MAPS", "RESULTS/ACID_CONTACT_MAPS", "RESULTS/SM_CONTACT_MAPS", "RESULTS/SUMMARY"
    ]
    for folder in all_folders:
        folder_path = "{}/{}".format(path, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    mol_name = "X"

    # Initialize empty dataframes for all categories
    df_sg, df_res_contact_sg, df_acid_contact_sg, df_res_count_sg, df_acid_count_sg = analysis.define_pd()
    df_dsm, _, _, _, _ = analysis.define_pd()
    df_ndsm, _, _, _, _ = analysis.define_pd()
    # Master SM contact/count frames, populated across the DSM and NDSM loops
    _, df_res_contact_master, df_acid_contact_master, df_res_count_master, df_acid_count_master = analysis.define_pd()

    df_sg_clean = pd.DataFrame(columns=df_sg.columns)
    df_dsm_clean = pd.DataFrame(columns=df_dsm.columns)
    df_ndsm_clean = pd.DataFrame(columns=df_ndsm.columns)
    has_dsm_input = os.path.isdir(os.path.join(path, "ANALYSIS_DSM_AVE"))
    has_ndsm_input = os.path.isdir(os.path.join(path, "ANALYSIS_NDSM_AVE"))
    dsm_names = [
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
    ndsm_names = [
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

    # SG Processing
    try:
        print("\n========== PROCESSING SG ==========")
        folder = "ANALYSIS_SG_AVE"
        name = "sg_X"
        df_sg, df_res_contact_sg, df_acid_contact_sg, df_res_count_sg, df_acid_count_sg = analysis.define_pd()

        print(name)
        dfs = analysis.sg_sm_analysis_full(
            path,
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
            diffusion_params=diffusion_overrides,
            viscosity_params=viscosity_overrides,
            species_to_analyze=species_override,
        )
        df_sg = dfs[0]
        df_res_contact_sg = dfs[1]
        df_acid_contact_sg = dfs[2]
        df_res_count_sg = dfs[3]
        df_acid_count_sg = dfs[4]

        df_sg.to_csv("{}/RESULTS/SUMMARY/SG_Quant_Data.csv".format(path), index=False)
        df_sg_clean = df_sg.drop_duplicates(subset=["Small Molecule ID"])
        print("SG PROCESSING SUCCESSFUL")
    except Exception as e:
        print(f"SG PROCESSING FAILED: {e}")
        df_sg_clean = pd.DataFrame(columns=df_sg.columns)

    # DSM Processing
    try:
        print("\n========== PROCESSING DSM ==========")
        if not has_dsm_input:
            print("[INFO] ANALYSIS_DSM not found; skipping DSM processing for this temperature")
            raise FileNotFoundError("ANALYSIS_DSM not found")
        df_dsm, _, _, _, _ = analysis.define_pd()
        folder = "ANALYSIS_DSM_AVE"

        for name in dsm_names:
            print(name)
            dfs = analysis.sg_sm_analysis_full(
                path,
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
                diffusion_params=diffusion_overrides,
                viscosity_params=viscosity_overrides,
                species_to_analyze=species_override,
            )
            df_dsm = dfs[0]
            df_res_contact_master = dfs[1]
            df_acid_contact_master = dfs[2]
            df_res_count_master = dfs[3]
            df_acid_count_master = dfs[4]

        df_dsm = df_dsm.sort_values(by="Small Molecule ID", ascending=False)

        mean_list = []
        for columnName in df_dsm:
            avg = pd.to_numeric(df_dsm[columnName], errors="coerce").mean()
            if math.isnan(avg):
                avg = np.nan
            mean_list.append(avg)

        avg_idx = len(df_dsm.index)
        df_dsm.loc[avg_idx] = mean_list
        for col in ("Small Molecule ID", "Small Molecule Name", "Compound Name",
                     "Compound Class", "D_Binary", "c_{SM}"):
            if col in df_dsm.columns:
                df_dsm.at[avg_idx, col] = "DSM_AVG"

        try:
            dsm_visc_avg = analysis.average_viscosity_metrics(dsm_names)
        except RuntimeError:
            dsm_visc_avg = None

        dfs = analysis.sg_sm_analysis_full(
            path,
            "ANALYSIS_DSM_AVE",
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
            diffusion_params=diffusion_overrides,
            viscosity_params=viscosity_overrides,
            species_to_analyze=species_override,
            precomputed_viscosity=dsm_visc_avg,
        )
        df_dsm = dfs[0]
        df_res_contact_master = dfs[1]
        df_acid_contact_master = dfs[2]
        df_res_count_master = dfs[3]
        df_acid_count_master = dfs[4]

        df_dsm = _fill_aggregate_spatial_from_members(df_dsm, dsm_names, "DSM")

        df_dsm_clean = df_dsm.drop_duplicates(subset=["Small Molecule ID"])
        print("DSM PROCESSING SUCCESSFUL")
    except Exception as e:
        if has_dsm_input:
            print(f"DSM PROCESSING FAILED: {e}")
        df_dsm_clean = pd.DataFrame(columns=df_dsm.columns)

    # NDSM Processing
    try:
        print("\n========== PROCESSING NDSM ==========")
        if not has_ndsm_input:
            print("[INFO] ANALYSIS_NDSM not found; skipping NDSM processing for this temperature")
            raise FileNotFoundError("ANALYSIS_NDSM not found")
        df_ndsm, _, _, _, _ = analysis.define_pd()
        folder = "ANALYSIS_NDSM_AVE"

        for name in ndsm_names:
            print(name)
            dfs = analysis.sg_sm_analysis_full(
                path,
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
                diffusion_params=diffusion_overrides,
                viscosity_params=viscosity_overrides,
                species_to_analyze=species_override,
            )
            df_ndsm = dfs[0]
            df_res_contact_master = dfs[1]
            df_acid_contact_master = dfs[2]
            df_res_count_master = dfs[3]
            df_acid_count_master = dfs[4]

        df_ndsm = df_ndsm.sort_values(by="Small Molecule ID", ascending=False)

        mean_list = []
        for columnName in df_ndsm:
            avg = pd.to_numeric(df_ndsm[columnName], errors="coerce").mean()
            if math.isnan(avg):
                avg = np.nan
            mean_list.append(avg)

        avg_idx = len(df_ndsm.index)
        df_ndsm.loc[avg_idx] = mean_list
        for col in ("Small Molecule ID", "Small Molecule Name", "Compound Name",
                     "Compound Class", "D_Binary", "c_{SM}"):
            if col in df_ndsm.columns:
                df_ndsm.at[avg_idx, col] = "NDSM_AVG"

        try:
            ndsm_visc_avg = analysis.average_viscosity_metrics(ndsm_names)
        except RuntimeError:
            ndsm_visc_avg = None

        dfs = analysis.sg_sm_analysis_full(
            path,
            "ANALYSIS_NDSM_AVE",
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
            diffusion_params=diffusion_overrides,
            viscosity_params=viscosity_overrides,
            species_to_analyze=species_override,
            precomputed_viscosity=ndsm_visc_avg,
        )
        df_ndsm = dfs[0]
        df_res_contact_master = dfs[1]
        df_acid_contact_master = dfs[2]
        df_res_count_master = dfs[3]
        df_acid_count_master = dfs[4]

        df_ndsm = _fill_aggregate_spatial_from_members(df_ndsm, ndsm_names, "NDSM")

        df_ndsm_clean = df_ndsm.drop_duplicates(subset=["Small Molecule ID"])
        print("NDSM PROCESSING SUCCESSFUL")
    except Exception as e:
        if has_ndsm_input:
            print(f"NDSM PROCESSING FAILED: {e}")
        df_ndsm_clean = pd.DataFrame(columns=df_ndsm.columns)

    # Save SM maps and combined summary tables
    try:
        # Derive DSM/NDSM averages for SM maps
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
            for col, system_name in zip([c for c in master_df.columns if c.startswith("D") and not c.startswith("DSM")], d_names):
                mat = _read_contact_matrix(f"{path}/RESULTS/SM_CONTACT_MAPS/{sem_prefix}_{system_name}.csv")
                if mat is not None:
                    out[col] = np.asarray(mat, dtype=float).reshape(-1)
            for col, system_name in zip([c for c in master_df.columns if c.startswith("ND") and not c.startswith("NDSM")], nd_names):
                mat = _read_contact_matrix(f"{path}/RESULTS/SM_CONTACT_MAPS/{sem_prefix}_{system_name}.csv")
                if mat is not None:
                    out[col] = np.asarray(mat, dtype=float).reshape(-1)
            d_cols = [c for c in master_df.columns if c.startswith("D") and not c.startswith("DSM")]
            nd_cols = [c for c in master_df.columns if c.startswith("ND") and not c.startswith("NDSM")]
            if d_cols:
                out["DSM_AVE"] = master_df[d_cols].apply(pd.to_numeric, errors="coerce").sem(axis=1)
            if nd_cols:
                out["NDSM_AVE"] = master_df[nd_cols].apply(pd.to_numeric, errors="coerce").sem(axis=1)
            out.to_csv(f"{path}/RESULTS/SM_CONTACT_MAPS/{filename}", index=False)

        _write_sm_contact_sem_master(
            df_res_contact_master,
            "SM_ResMap_SEM_Data.csv",
            "SM_Residue_Contacts_SEM",
            dsm_names,
            ndsm_names,
        )
        _write_sm_contact_sem_master(
            df_acid_contact_master,
            "SM_AcidMap_SEM_Data.csv",
            "SM_Acid_Contacts_SEM",
            dsm_names,
            ndsm_names,
        )

        df_all = pd.concat([df_sg_clean, df_dsm_clean, df_ndsm_clean], ignore_index=True)
        if not df_all.empty:
            df_all_clean = df_all.drop_duplicates(subset=["Small Molecule ID"])
        else:
            df_all_clean = df_all

        df_dsm_clean.to_csv(f"{path}/RESULTS/SUMMARY/DSM_Quant_Data.csv", index=False)
        df_ndsm_clean.to_csv(f"{path}/RESULTS/SUMMARY/NDSM_Quant_Data.csv", index=False)
        df_all_clean.to_csv(f"{path}/RESULTS/SUMMARY/Quant_Data.csv", index=False)
    except Exception as e:
        print(f"SM INTERDEPENDENCY BLOCK FAILED: {e}")

    if not has_dsm_input and not has_ndsm_input:
        print("[INFO] SG-only temperature detected; skipping SM contact-map summary stages")
        sys.exit(0)


    # Residue DSM / NDSM Difference Standardized Array
    # Build standardized mean contact arrays for DSM and NDSM by averaging
    # the per-SM standardized maps that were written earlier in the pipeline.
    dsm_names_for_std = [
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
    ndsm_names_for_std = [
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

    def _mean_standardized_contacts(tag_list, label):
        arrays = []
        for tag in tag_list:
            fn = "{}/RESULTS/RESIDUE_CONTACT_MAPS/Residue_Contacts_Standardized_Mean_{}.csv".format(
                path, tag
            )
            if os.path.isfile(fn):
                arrays.append(
                    np.array(pd.read_csv(fn, header=None), dtype=float)
                )
            else:
                print(f"[WARN] Missing standardized residue contacts for {label} system '{tag}': {fn}")
        if not arrays:
            raise FileNotFoundError(
                f"No Residue_Contacts_Standardized_Mean_* files found for {label} systems in {path}/RESULTS/RESIDUE_CONTACT_MAPS"
            )
        return np.nanmean(np.stack(arrays, axis=0), axis=0)

    def _sem_standardized_contacts(tag_list):
        arrays = []
        for tag in tag_list:
            fn = "{}/RESULTS/RESIDUE_CONTACT_MAPS/Residue_Contacts_Standardized_Mean_{}.csv".format(path, tag)
            if os.path.isfile(fn):
                arrays.append(np.array(pd.read_csv(fn, header=None), dtype=float))
        return _contact_matrix_sem(arrays) if arrays else None

    if has_dsm_input and has_ndsm_input and not df_dsm_clean.empty and not df_ndsm_clean.empty:
        ndsm_res_arr = _mean_standardized_contacts(ndsm_names_for_std, "NDSM")
        dsm_res_arr = _mean_standardized_contacts(dsm_names_for_std, "DSM")
        ndsm_res_sem = _sem_standardized_contacts(ndsm_names_for_std)
        dsm_res_sem = _sem_standardized_contacts(dsm_names_for_std)

        # sg_X standardized map exists explicitly
        sg_res_arr = np.array(
            pd.read_csv(
                "{}/RESULTS/RESIDUE_CONTACT_MAPS/Residue_Contacts_Standardized_Mean_sg_X.csv".format(path),
                header=None,
            ),
            dtype=float,
        )
        sg_res_sem = _read_contact_matrix("{}/RESULTS/RESIDUE_CONTACT_MAPS/Residue_Contacts_Standardized_SEM_sg_X.csv".format(path))

        res_contact_delta = np.subtract(dsm_res_arr, ndsm_res_arr)
        res_contact_array = np.divide(
            res_contact_delta,
            sg_res_arr,
            out=np.zeros_like(res_contact_delta, dtype=float),
            where=sg_res_arr != 0,
        )

        res_contact_array = _prepare_residue_square(res_contact_array)
        res_1d_data = _prepare_residue_rows(res_contact_array.sum(axis=1).reshape(-1, 1))
        np.savetxt("{}/RESULTS/RESIDUE_CONTACT_MAPS/Residue_DSM_NDSM_Difference_Standardized_1D_HeatMap.csv".format(path), res_1d_data, delimiter=",")

        # save the dataframe as a csv file
        np.savetxt("{}/RESULTS/RESIDUE_CONTACT_MAPS/Residue_DSM_NDSM_Difference_Standardized_HeatMap.csv".format(path), res_contact_array, delimiter=",")
        res_contact_sem = _ratio_difference_sem(dsm_res_arr - ndsm_res_arr, np.sqrt(dsm_res_sem ** 2 + ndsm_res_sem ** 2) if dsm_res_sem is not None and ndsm_res_sem is not None else None, sg_res_arr, sg_res_sem)
        if res_contact_sem is not None:
            np.savetxt("{}/RESULTS/RESIDUE_CONTACT_MAPS/Residue_DSM_NDSM_Difference_Standardized_SEM_HeatMap.csv".format(path), res_contact_sem, delimiter=",")
            res_sem_1d_data = _prepare_residue_rows(_contact_1d_sem(res_contact_sem).sum(axis=1).reshape(-1, 1))
            np.savetxt("{}/RESULTS/RESIDUE_CONTACT_MAPS/Residue_DSM_NDSM_Difference_Standardized_SEM_1D_HeatMap.csv".format(path), res_sem_1d_data, delimiter=",")
    else:
        print("[INFO] Skipping combined DSM/NDSM standardized residue contact data: missing one or both categories")




    # Acid DSM / NDSM Difference Standardized Array

    def _mean_standardized_acid_contacts(tag_list, label):
        arrays = []
        for tag in tag_list:
            fn = "{}/RESULTS/ACID_CONTACT_MAPS/Acid_Contacts_Standardized_Mean_{}.csv".format(
                path, tag
            )
            if os.path.isfile(fn):
                arrays.append(
                    np.array(pd.read_csv(fn, header=None), dtype=float)
                )
            else:
                print(f"[WARN] Missing standardized acid contacts for {label} system '{tag}': {fn}")
        if not arrays:
            raise FileNotFoundError(
                f"No Acid_Contacts_Standardized_Mean_* files found for {label} systems in {path}/RESULTS/ACID_CONTACT_MAPS"
            )
        return np.nanmean(np.stack(arrays, axis=0), axis=0)

    def _sem_standardized_acid_contacts(tag_list):
        arrays = []
        for tag in tag_list:
            fn = "{}/RESULTS/ACID_CONTACT_MAPS/Acid_Contacts_Standardized_Mean_{}.csv".format(path, tag)
            if os.path.isfile(fn):
                arrays.append(np.array(pd.read_csv(fn, header=None), dtype=float))
        return _contact_matrix_sem(arrays) if arrays else None

    if has_dsm_input and has_ndsm_input and not df_dsm_clean.empty and not df_ndsm_clean.empty:
        ndsm_acid_arr = _mean_standardized_acid_contacts(ndsm_names_for_std, "NDSM")
        dsm_acid_arr = _mean_standardized_acid_contacts(dsm_names_for_std, "DSM")
        ndsm_acid_sem = _sem_standardized_acid_contacts(ndsm_names_for_std)
        dsm_acid_sem = _sem_standardized_acid_contacts(dsm_names_for_std)
        sg_acid_arr = np.array(
            pd.read_csv(
                "{}/RESULTS/ACID_CONTACT_MAPS/Acid_Contacts_Standardized_Mean_sg_X.csv".format(path),
                header=None,
            ),
            dtype=float,
        )
        sg_acid_sem = _read_contact_matrix("{}/RESULTS/ACID_CONTACT_MAPS/Acid_Contacts_Standardized_SEM_sg_X.csv".format(path))

        acid_contact_delta = np.subtract(dsm_acid_arr, ndsm_acid_arr)
        acid_contact_array = np.divide(
            acid_contact_delta,
            sg_acid_arr,
            out=np.zeros_like(acid_contact_delta, dtype=float),
            where=sg_acid_arr != 0,
        )

        acid_contact_array = _prepare_acid_square(acid_contact_array)
        acid_1d_data = _prepare_acid_rows(acid_contact_array.sum(axis=1).reshape(-1, 1))
        np.savetxt("{}/RESULTS/ACID_CONTACT_MAPS/Acid_DSM_NDSM_Difference_Standardized_1D_HeatMap.csv".format(path), acid_1d_data, delimiter=",")

        # save the dataframe as a csv file
        np.savetxt("{}/RESULTS/ACID_CONTACT_MAPS/Acid_DSM_NDSM_Difference_Standardized_HeatMap.csv".format(path), acid_contact_array, delimiter=",")
        acid_contact_sem = _ratio_difference_sem(dsm_acid_arr - ndsm_acid_arr, np.sqrt(dsm_acid_sem ** 2 + ndsm_acid_sem ** 2) if dsm_acid_sem is not None and ndsm_acid_sem is not None else None, sg_acid_arr, sg_acid_sem)
        if acid_contact_sem is not None:
            np.savetxt("{}/RESULTS/ACID_CONTACT_MAPS/Acid_DSM_NDSM_Difference_Standardized_SEM_HeatMap.csv".format(path), acid_contact_sem, delimiter=",")
            acid_sem_1d_data = _prepare_acid_rows(_contact_1d_sem(acid_contact_sem).sum(axis=1).reshape(-1, 1))
            np.savetxt("{}/RESULTS/ACID_CONTACT_MAPS/Acid_DSM_NDSM_Difference_Standardized_SEM_1D_HeatMap.csv".format(path), acid_sem_1d_data, delimiter=",")
    else:
        print("[INFO] Skipping combined DSM/NDSM standardized acid contact data: missing one or both categories")




    # SM Residue Contact Array

    df_res_contact = pd.read_csv("{}/RESULTS/SM_CONTACT_MAPS/SM_ResMap_Data.csv".format(path))

    df_res_contact_sem = None
    if df_res_contact.shape[1] > 1:
        sm_res_contact_array = np.array(df_res_contact.iloc[:,1:])

        res_sem_path = "{}/RESULTS/SM_CONTACT_MAPS/SM_ResMap_SEM_Data.csv".format(path)
        if os.path.isfile(res_sem_path):
            df_res_contact_sem = pd.read_csv(res_sem_path)
            sm_res_sem_array = df_res_contact_sem.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            np.savetxt(
                "{}/RESULTS/SM_CONTACT_MAPS/SM_Residue_DSM_NDSM_SUMMARY_Standardized_SEM_HeatMap.csv".format(path),
                sm_res_sem_array,
                delimiter=",",
            )
    else:
        print("[INFO] Skipping SM residue summary data: no SM columns were produced for this temperature")



    # SM Acid Contact Array

    df_acid_contact = pd.read_csv("{}/RESULTS/SM_CONTACT_MAPS/SM_AcidMap_Data.csv".format(path))

    df_acid_contact_sem = None
    if df_acid_contact.shape[1] > 1:
        df_acid_contact.to_csv("{}/RESULTS/SM_CONTACT_MAPS/DF_Acid_DSM_NDSM_Difference_Standardized_HeatMap.csv".format(path), index=False)

        sm_acid_contact_array = np.array(df_acid_contact.iloc[:,1:])

        np.savetxt("{}/RESULTS/SM_CONTACT_MAPS/SM_Acid_DSM_NDSM_Difference_Standardized_HeatMap.csv".format(path), sm_acid_contact_array, delimiter=",")

        sm_acid_contact_array = _prepare_acid_rows(sm_acid_contact_array)

        acid_sem_path = "{}/RESULTS/SM_CONTACT_MAPS/SM_AcidMap_SEM_Data.csv".format(path)
        if os.path.isfile(acid_sem_path):
            df_acid_contact_sem = pd.read_csv(acid_sem_path)
            sm_acid_sem_array = df_acid_contact_sem.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            np.savetxt(
                "{}/RESULTS/SM_CONTACT_MAPS/SM_Acid_DSM_NDSM_SUMMARY_Standardized_SEM_HeatMap.csv".format(path),
                sm_acid_sem_array,
                delimiter=",",
            )

        # save the dataframe as a csv file
        np.savetxt("{}/RESULTS/SM_CONTACT_MAPS/SM_Acid_DSM_NDSM_Difference_Standardized_HeatMap_2.csv".format(path), sm_acid_contact_array, delimiter=",")
    else:
        print("[INFO] Skipping SM acid summary data: no SM columns were produced for this temperature")

    # SM Residue Summary Contact Array
    d_cols_res = [c for c in df_res_contact.columns if c.startswith("D")]
    nd_cols_res = [c for c in df_res_contact.columns if c.startswith("ND")]
    if {"DSM", "NDSM"}.issubset(df_res_contact.columns):
        b = df_res_contact["DSM"] - df_res_contact["NDSM"]
    elif {"DSM_AVE", "NDSM_AVE"}.issubset(df_res_contact.columns):
        b = df_res_contact["DSM_AVE"] - df_res_contact["NDSM_AVE"]
    elif d_cols_res and nd_cols_res:
        d_vals_res = df_res_contact[d_cols_res].apply(pd.to_numeric, errors="coerce")
        nd_vals_res = df_res_contact[nd_cols_res].apply(pd.to_numeric, errors="coerce")
        b = d_vals_res.mean(axis=1) - nd_vals_res.mean(axis=1)
    else:
        b = None
        print("[INFO] Skipping SM residue summary difference heatmaps: no DSM/NDSM summary columns were produced")
    if b is not None:
        sm_res_contact_array = np.transpose(np.asarray([b.to_numpy(dtype=float)]))
        sm_res_contact_array = _prepare_residue_rows(sm_res_contact_array)

        # save the dataframe as a csv file
        np.savetxt("{}/RESULTS/SM_CONTACT_MAPS/SM_Residue_DSM_NDSM_SUMMARY_Difference_Standardized_HeatMap.csv".format(path),
                   sm_res_contact_array, delimiter=",")
        if df_res_contact_sem is not None and {"DSM_AVE", "NDSM_AVE"}.issubset(df_res_contact_sem.columns):
            res_diff_sem = np.sqrt(
                pd.to_numeric(df_res_contact_sem["DSM_AVE"], errors="coerce").to_numpy(dtype=float) ** 2
                + pd.to_numeric(df_res_contact_sem["NDSM_AVE"], errors="coerce").to_numpy(dtype=float) ** 2
            ).reshape(-1, 1)
            np.savetxt(
                "{}/RESULTS/SM_CONTACT_MAPS/SM_Residue_DSM_NDSM_SUMMARY_Difference_Standardized_SEM_HeatMap.csv".format(path),
                res_diff_sem,
                delimiter=",",
            )

    # SM Acid Contact Array

    df_acid_contact = pd.read_csv("{}/RESULTS/SM_CONTACT_MAPS/SM_AcidMap_Data.csv".format(path))

    df_acid_contact.to_csv("{}/RESULTS/SM_CONTACT_MAPS/DF_Acid_DSM_NDSM_Difference_Standardized_HeatMap.csv".format(path), index=False)

    sm_acid_contact_array = np.array(df_acid_contact.iloc[:, 1:])

    np.savetxt("{}/RESULTS/SM_CONTACT_MAPS/SM_Acid_DSM_NDSM_Difference_Standardized_HeatMap.csv".format(path), sm_acid_contact_array,
               delimiter=",")

    # SM Acid Contact Summary Array
    d_cols_acid = [c for c in df_acid_contact.columns if c.startswith("D")]
    nd_cols_acid = [c for c in df_acid_contact.columns if c.startswith("ND")]
    if {"DSM", "NDSM"}.issubset(df_acid_contact.columns):
        b = df_acid_contact["DSM"] - df_acid_contact["NDSM"]
    elif {"DSM_AVE", "NDSM_AVE"}.issubset(df_acid_contact.columns):
        b = df_acid_contact["DSM_AVE"] - df_acid_contact["NDSM_AVE"]
    elif d_cols_acid and nd_cols_acid:
        d_vals_acid = df_acid_contact[d_cols_acid].apply(pd.to_numeric, errors="coerce")
        nd_vals_acid = df_acid_contact[nd_cols_acid].apply(pd.to_numeric, errors="coerce")
        b = d_vals_acid.mean(axis=1) - nd_vals_acid.mean(axis=1)
    else:
        b = None
        print("[INFO] Skipping SM acid summary difference heatmaps: no DSM/NDSM summary columns were produced")
    if b is not None:
        sm_acid_contact_array = np.transpose(np.asarray([b.to_numpy(dtype=float)]))

        np.savetxt("{}/RESULTS/SM_CONTACT_MAPS/SM_Acid_DSM_NDSM_SUMMARY_Difference_Standardized_HeatMap.csv".format(path),
                   sm_acid_contact_array, delimiter=",")
        if df_acid_contact_sem is not None and {"DSM_AVE", "NDSM_AVE"}.issubset(df_acid_contact_sem.columns):
            acid_diff_sem = np.sqrt(
                pd.to_numeric(df_acid_contact_sem["DSM_AVE"], errors="coerce").to_numpy(dtype=float) ** 2
                + pd.to_numeric(df_acid_contact_sem["NDSM_AVE"], errors="coerce").to_numpy(dtype=float) ** 2
            ).reshape(-1, 1)
            np.savetxt(
                "{}/RESULTS/SM_CONTACT_MAPS/SM_Acid_DSM_NDSM_SUMMARY_Difference_Standardized_SEM_HeatMap.csv".format(path),
                acid_diff_sem,
                delimiter=",",
            )

        sm_acid_contact_array = _prepare_acid_rows(sm_acid_contact_array)

        # save the dataframe as a csv file
        np.savetxt("{}/RESULTS/SM_CONTACT_MAPS/SM_Acid_DSM_NDSM_SUMMARY_Difference_Standardized_HeatMap_2.csv".format(path),
                   sm_acid_contact_array, delimiter=",")
