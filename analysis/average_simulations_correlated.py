#!/usr/bin/env python3
"""Block-layout-aware time-window averaging (correlated pipeline).

Pipeline role
-------------
Correlated-pipeline analogue of ``average_simulations.py`` (standard pipeline
Step 1). Instead of averaging every 50 ns window equally, it groups windows into
the correlation-corrected blocks recommended by
``block_correlation_diagnostics.py`` (``RECOMMENDED_SYSTEM_BLOCK_LAYOUT.csv``),
forms a per-block mean for every observable (density profiles, PCA shape, cluster
scalars, residue/acid/domain/SM contact matrices, G3BP1 diffusivity curves), and
then averages and takes the SEM across blocks. This yields a fresh blocked
``CLASSIFY_BLOCKED_*`` analysis root whose ``ANALYSIS_*_AVE`` outputs are
consumed downstream by the correlated SYSTEM/BIOPOLYMER/ARRAY/KMeans steps.

Key inputs
----------
- A standard (per-window) analysis root via ``--path`` containing
  ``ANALYSIS_{SG,DSM,NDSM}/`` with per-window CSV/NPZ products.
- ``RECOMMENDED_SYSTEM_BLOCK_LAYOUT.csv`` via ``--layout-csv`` or the standard
  diagnostics path written by ``block_correlation_diagnostics.py`` (sibling
  diagnostic CSVs are copied into the new root).
- CLI flags: ``--path``, ``--folder``, ``--temp``, ``--tmin``, ``--dt``,
  ``--tmax``, ``--out-root``, ``--overwrite``, ``--plot-only``; optional
  ``--layout-csv`` override.

Key outputs (written under the new blocked ``out_root``)
- Symlinks to source ``ANALYSIS_{cat}/`` plus populated ``ANALYSIS_{cat}_AVE/``
  with block-averaged ``Density_Profile_*``, ``PCA_*``, ``Cluster_*``,
  ``*_Contacts_Mean/SEM_*``, ``Domain_Contacts_Mean/SEM_*`` and
  ``G3BP1_Diffusivity_*`` files.
- ``dsm_list.txt`` / ``ndsm_list.txt``, the copied
  ``RESULTS/CORRELATION_DIAGNOSTICS/`` tables, and per-category aggregates.

Example invocation
-------------------
    python average_simulations_correlated.py \
        --path TEMP_300 \
        --folder CLASSIFY_CORRELATED --temp 300 --tmin 50 --dt 50 --tmax 2000 \
        --layout-csv TEMP_300/CORRELATION_CLASSIFY_300_50_50_2000/RESULTS/CORRELATION_DIAGNOSTICS/RECOMMENDED_SYSTEM_BLOCK_LAYOUT.csv \
        --overwrite
"""
import argparse
import math
import os
import re
import shutil
from glob import glob
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from average_simulations import generate_aggregate_rdp_inputs


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

BIOPOLYMER_DENSITY_LIST = [
    "RNA", "ADENINE", "UCG", "Protein", "G3BP1", "TDP43", "TTP", "TIA1", "FUS", "PABP1",
]

DOMAIN_BIOPOLYMER_LIST = [
    "ProteinTDP43", "ProteinFUS", "ProteinTIA1", "ProteinG3BP1", "RNA", "ProteinPABP1", "ProteinTTP",
]
DOMAIN_BIOPOLYMER_LENGTHS = [414, 526, 386, 466, 840, 636, 326]
RESIDUE_SM_SPECIES = [
    "ProteinG3BP1",
    "ProteinPABP1",
    "ProteinTTP",
    "ProteinTIA1",
    "ProteinTDP43",
    "ProteinFUS",
    "RNA",
]


def ensure_dir(path: str) -> None:
    """Create ``path`` (and parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def safe_float(x) -> float:
    """Return ``float(x)`` or NaN if the conversion fails."""
    try:
        return float(x)
    except Exception:
        return math.nan


def time_candidates(window_start: int) -> List[str]:
    """Return the candidate string spellings of a window-start time tag.

    Filenames use inconsistent integer/float time labels; this enumerates the
    spellings (``"50"``, ``"50.0"``, ...) tried when resolving a window file.
    """
    return [
        str(window_start),
        str(int(window_start)),
        f"{float(window_start):.1f}",
        str(float(window_start)),
    ]


def resolve_window_file(category_dir: str, prefix: str, system_name: str, window_start: int, suffix: str = ".csv") -> Optional[str]:
    """Locate ``{prefix}_{system}_{t}{suffix}`` for one window, trying time spellings.

    Returns the first existing path (or the preferred path if it exists), else
    None.
    """
    preferred = os.path.join(category_dir, f"{prefix}_{system_name}_{float(window_start):.1f}{suffix}")
    for tag in time_candidates(window_start):
        candidate = os.path.join(category_dir, f"{prefix}_{system_name}_{tag}{suffix}")
        if os.path.isfile(candidate):
            return candidate
    return preferred if os.path.isfile(preferred) else None


def reshape_to_subarrays(arr: np.ndarray, p: int, q: int) -> np.ndarray:
    """Average the pxq sub-blocks of a domain-contact matrix.

    Splits ``arr`` into p-by-q tiles and returns their elementwise mean. For
    square same-species pairs (p==q) the diagonal (i==j) tiles are excluded;
    for cross-species pairs all tiles are averaged. Returns a zero pxq matrix if
    no tiles qualify.
    """
    arr = arr.copy()
    n, m = arr.shape
    if n % p != 0 or m % q != 0:
        raise ValueError(f"Dimensions mismatch: {n} % {p} != 0 or {m} % {q} != 0")

    num_subarrays_n = n // p
    num_subarrays_m = m // q
    subarrays_diff = []

    for i in range(num_subarrays_n):
        for j in range(num_subarrays_m):
            sub_array = arr[i * p:(i + 1) * p, j * q:(j + 1) * q]
            if p == q:
                if i != j:
                    subarrays_diff.append(sub_array)
            else:
                subarrays_diff.append(sub_array)

    if not subarrays_diff:
        return np.zeros((p, q))
    return np.mean(np.array(subarrays_diff), axis=0)


def load_layout(layout_csv: str) -> Dict[str, Dict[str, List[List[int]]]]:
    """Parse a recommended block-layout CSV into nested block lists.

    Returns ``{category: {system_name: [[window_starts], ...]}}`` where each
    inner list is one block's member window-start times. Raises ValueError if
    required columns (category, system_name, block_id, window_members) are
    missing.
    """
    df = pd.read_csv(layout_csv)
    required = {"category", "system_name", "block_id", "window_members"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Layout CSV missing required columns: {sorted(missing)}")

    layout: Dict[str, Dict[str, List[List[int]]]] = {}
    for (category, system_name), group in df.groupby(["category", "system_name"], dropna=False):
        group = group.sort_values("block_id")
        blocks: List[List[int]] = []
        for _, row in group.iterrows():
            members = [int(x) for x in str(row["window_members"]).split(";") if str(x).strip()]
            if members:
                blocks.append(members)
        if blocks:
            layout.setdefault(str(category), {})[str(system_name)] = blocks
    return layout


def symlink_dir(src: str, dst: str) -> None:
    """Create a symlink ``dst`` -> absolute ``src`` unless ``dst`` already exists."""
    if os.path.lexists(dst):
        return
    os.symlink(os.path.abspath(src), dst)


def copy_if_exists(src: str, dst: str) -> None:
    """Copy ``src`` to ``dst`` (preserving metadata) only if ``src`` is a file."""
    if os.path.isfile(src):
        shutil.copy2(src, dst)


def average_vectors(block_vectors: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, int]:
    """Compute block-averaged mean, SEM, and effective sample count.

    Returns (mean, sem, n) where n is the number of independent samples
    (blocks or systems) used for the SEM denominator.  Downstream consumers
    should report n alongside any SEM so that t-distribution corrections
    can be applied for small n (Flyvbjerg & Petersen, JCP 1989).
    """
    if not block_vectors:
        raise ValueError("No block vectors provided")
    arr = np.stack(block_vectors, axis=0)
    n = arr.shape[0]
    mean = np.nanmean(arr, axis=0)
    if n > 1:
        sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(n)
    else:
        sem = np.full_like(mean, np.nan, dtype=float)
    return mean, sem, n


def read_density_profile(path: str) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """Read a radial density-profile CSV.

    Returns ``(distance, density, x_col_name, y_col_name)`` with the first two
    columns coerced to float.
    """
    df = pd.read_csv(path)
    x_col = df.columns[0]
    y_col = df.columns[1]
    x = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
    return x, y, x_col, y_col


def average_density_blocks(category_dir: str, system_name: str, prefix: str, blocks: Sequence[Sequence[int]]) -> Optional[pd.DataFrame]:
    """Block-average a system's radial density profiles onto a common grid.

    Averages per-window profiles within each block (dropping windows whose
    distance grid disagrees), then returns a DataFrame of distance, mean
    density and across-block SEM (``n_blocks`` in ``.attrs``). Returns None if
    no profile is found.
    """
    block_profiles: List[np.ndarray] = []
    distance = None
    x_col = "Distance from center of mass (A)"
    y_col = "Protein density (mg/mL)"

    for block in blocks:
        curves = []
        for window_start in block:
            fp = resolve_window_file(category_dir, f"DensityProfile_{prefix}", system_name, window_start)
            if fp is None:
                continue
            x, y, x_name, y_name = read_density_profile(fp)
            if distance is None:
                distance = x
                x_col = x_name
                y_col = y_name
            elif len(distance) != len(x) or not np.allclose(distance, x, equal_nan=True):
                continue
            curves.append(y)
        if curves:
            block_profiles.append(np.nanmean(np.stack(curves, axis=0), axis=0))

    if not block_profiles or distance is None:
        return None

    mean, sem, n_blocks = average_vectors(block_profiles)
    out = pd.DataFrame({
        x_col: distance,
        y_col: mean,
        "Standard mean error": sem,
    })
    out.attrs["n_blocks"] = n_blocks
    return out


def read_numeric_frame_mean(path: str) -> pd.Series:
    """Read a CSV and return the column-wise mean over its numeric columns."""
    df = pd.read_csv(path)
    return df.mean(numeric_only=True)


def average_pca_blocks(category_dir: str, system_name: str, prefix: str, blocks: Sequence[Sequence[int]]) -> Optional[pd.DataFrame]:
    """Block-average a system's per-window PCA shape tables.

    Returns one row per block (the within-block mean of the per-window PCA
    means), or None if no PCA file is found.
    """
    block_rows: List[pd.Series] = []
    cols = None
    for block in blocks:
        rows = []
        for window_start in block:
            fp = resolve_window_file(category_dir, f"PCA_{prefix}", system_name, window_start)
            if fp is None:
                continue
            row = read_numeric_frame_mean(fp)
            if cols is None:
                cols = list(row.index)
            row = row.reindex(cols)
            rows.append(row)
        if rows:
            block_rows.append(pd.concat(rows, axis=1).mean(axis=1))
    if not block_rows:
        return None
    df = pd.DataFrame(block_rows)
    if cols is not None:
        df = df.reindex(columns=cols)
    return df


def average_cluster_blocks(category_dir: str, system_name: str, prefix: str, blocks: Sequence[Sequence[int]]) -> Optional[pd.DataFrame]:
    """Block-average a system's per-window cluster scalar tables.

    Returns a one-row DataFrame of cross-block means plus across-block SEM
    columns (``RG SEM``, ``ND SEM``, ...) and ``n_blocks``; None if no cluster
    file is found.
    """
    block_rows: List[pd.Series] = []
    cols = None
    for block in blocks:
        rows = []
        for window_start in block:
            fp = resolve_window_file(category_dir, f"Cluster_{prefix}", system_name, window_start)
            if fp is None:
                continue
            row = read_numeric_frame_mean(fp)
            if cols is None:
                cols = list(row.index)
            row = row.reindex(cols)
            rows.append(row)
        if rows:
            block_rows.append(pd.concat(rows, axis=1).mean(axis=1))
    if not block_rows:
        return None

    block_df = pd.concat(block_rows, axis=1).T
    n_blocks = len(block_df)
    mean_row = block_df.mean(axis=0)
    out = mean_row.to_frame().T
    out["RG SEM"] = block_df["Largest Droplet Radius of Gyration"].sem() if "Largest Droplet Radius of Gyration" in block_df.columns else math.nan
    out["ND SEM"] = block_df["Number of Droplets"].sem() if "Number of Droplets" in block_df.columns else math.nan
    out["Chains Largest SEM"] = block_df["Chains in Largest Droplet"].sem() if "Chains in Largest Droplet" in block_df.columns else math.nan
    out["Mass Largest SEM"] = block_df["Mass of Largest Droplet (mg)"].sem() if "Mass of Largest Droplet (mg)" in block_df.columns else math.nan
    out["NE SEM"] = block_df["Number of External Chains"].sem() if "Number of External Chains" in block_df.columns else math.nan
    out["ME SEM"] = block_df["Mass of External Chains"].sem() if "Mass of External Chains" in block_df.columns else math.nan
    out["n_blocks"] = n_blocks
    return out


def read_matrix_csv(path: str) -> np.ndarray:
    """Read a headerless numeric CSV into a float ndarray."""
    return pd.read_csv(path, header=None).to_numpy(dtype=float)


def average_matrix_blocks(block_matrices: List[np.ndarray]) -> np.ndarray:
    """Return the elementwise mean over a list of equal-shape matrices."""
    return np.mean(np.stack(block_matrices, axis=0), axis=0)


def matrix_mean_sem(matrices: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return ``(mean, sem, n)`` elementwise over a list of matrices.

    SEM is NaN-filled when fewer than two matrices are supplied.
    """
    arr = np.stack([np.asarray(m, dtype=float) for m in matrices], axis=0)
    n = int(arr.shape[0])
    mean = np.nanmean(arr, axis=0)
    if n > 1:
        sem = np.nanstd(arr, axis=0, ddof=1) / math.sqrt(n)
    else:
        sem = np.full(arr.shape[1:], np.nan, dtype=float)
    return mean, sem, n


def average_contact_blocks(category_dir: str, system_name: str, prefix: str, blocks: Sequence[Sequence[int]]) -> Optional[np.ndarray]:
    """Block-average a system's ``{prefix}_Contacts_Count`` matrices (mean only).

    Returns the cross-block mean contact matrix, or None if none are found.
    """
    per_block: List[np.ndarray] = []
    for block in blocks:
        mats = []
        for window_start in block:
            fp = resolve_window_file(category_dir, f"{prefix}_Contacts_Count", system_name, window_start)
            if fp is None:
                continue
            mats.append(read_matrix_csv(fp))
        if mats:
            per_block.append(np.mean(np.stack(mats, axis=0), axis=0))
    if not per_block:
        return None
    return average_matrix_blocks(per_block)


def average_contact_blocks_with_sem(category_dir: str, system_name: str, prefix: str, blocks: Sequence[Sequence[int]]) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    """Block-average a system's ``{prefix}_Contacts_Count`` matrices with SEM.

    Returns ``(mean, sem, n_blocks)`` over the per-block means, or None if no
    contact files are found.
    """
    per_block: List[np.ndarray] = []
    for block in blocks:
        mats = []
        for window_start in block:
            fp = resolve_window_file(category_dir, f"{prefix}_Contacts_Count", system_name, window_start)
            if fp is None:
                continue
            mats.append(read_matrix_csv(fp))
        if mats:
            per_block.append(np.mean(np.stack(mats, axis=0), axis=0))
    if not per_block:
        return None
    return matrix_mean_sem(per_block)


def residue_sm_vector_for_window(category_dir: str, system_name: str, window_start: int) -> Optional[np.ndarray]:
    """Return the per-species residue-SM contact vector for one window.

    Prefers a legacy combined ``Residue_SM_Contacts_Count`` file; otherwise sums
    each per-species ``SM_Residue_Contacts_Count_<system>_<species>`` file over
    ``RESIDUE_SM_SPECIES``. Returns None if any species file is missing.
    """
    legacy = resolve_window_file(category_dir, "Residue_SM_Contacts_Count", system_name, window_start)
    if legacy is not None:
        vec = np.loadtxt(legacy, delimiter=",", dtype=float)
        return np.atleast_1d(vec).astype(float)

    per_species = []
    for species in RESIDUE_SM_SPECIES:
        fp = resolve_window_file(category_dir, f"SM_Residue_Contacts_Count_{system_name}_{species}", "", window_start)
        if fp is None:
            for tag in time_candidates(window_start):
                candidate = os.path.join(category_dir, f"SM_Residue_Contacts_Count_{system_name}_{species}_{tag}.csv")
                if os.path.isfile(candidate):
                    fp = candidate
                    break
        if fp is None:
            return None
        arr = np.loadtxt(fp, delimiter=",", dtype=float)
        arr = np.atleast_1d(arr)
        per_species.append(np.nansum(arr))
    return np.asarray(per_species, dtype=float)


def average_residue_sm_blocks(category_dir: str, system_name: str, blocks: Sequence[Sequence[int]]) -> Optional[np.ndarray]:
    """Block-average the per-species residue-SM contact vector (mean only).

    Returns the cross-block mean as a column vector, or None if no window
    yields a vector.
    """
    per_block: List[np.ndarray] = []
    for block in blocks:
        vectors = []
        for window_start in block:
            vec = residue_sm_vector_for_window(category_dir, system_name, window_start)
            if vec is not None:
                vectors.append(vec)
        if vectors:
            per_block.append(np.mean(np.stack(vectors, axis=0), axis=0))
    if not per_block:
        return None
    mean_vec = np.mean(np.stack(per_block, axis=0), axis=0)
    return mean_vec.reshape(-1, 1)


def average_residue_sm_blocks_with_sem(category_dir: str, system_name: str, blocks: Sequence[Sequence[int]]) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    """Block-average the per-species residue-SM contact vector with SEM.

    Returns ``(mean, sem, n_blocks)`` over per-block column vectors, or None if
    no window yields a vector.
    """
    per_block: List[np.ndarray] = []
    for block in blocks:
        vectors = []
        for window_start in block:
            vec = residue_sm_vector_for_window(category_dir, system_name, window_start)
            if vec is not None:
                vectors.append(vec)
        if vectors:
            per_block.append(np.mean(np.stack(vectors, axis=0), axis=0).reshape(-1, 1))
    if not per_block:
        return None
    return matrix_mean_sem(per_block)


def build_g3bp1_diffusivity_index(category_dir: str, system_name: str) -> Dict[int, List[str]]:
    """Index G3BP1 per-chain diffusivity CSVs by window-start time.

    Returns ``{window_start: [sorted file paths]}`` for all
    ``(Protein)G3BP1_Diffusivity_<system>_<t>_*.csv`` files in the directory.
    """
    pattern = re.compile(r"(?:G3BP1|ProteinG3BP1)_Diffusivity_{}_([0-9]+(?:\.[0-9]+)?)_[0-9]+\.csv$".format(re.escape(system_name)))
    out: Dict[int, List[str]] = {}
    if not os.path.isdir(category_dir):
        return out
    for name in os.listdir(category_dir):
        match = pattern.match(name)
        if not match:
            continue
        window_start = int(round(float(match.group(1))))
        out.setdefault(window_start, []).append(os.path.join(category_dir, name))
    for files in out.values():
        files.sort()
    return out


def average_diffusivity_blocks(category_dir: str, system_name: str, blocks: Sequence[Sequence[int]]) -> Optional[pd.DataFrame]:
    """Block-average G3BP1 MSD(t) / Rg(t) curves onto a common time grid.

    Within each block, pools all per-chain diffusivity frames and averages MSD
    and Rg by rounded time; then averages the block curves and reports SEM and
    sample count per time point. Returns a DataFrame (Time, MSD, MSD_SEM, Rg,
    Rg_SEM, N) or None if no curves are found.
    """
    diff_index = build_g3bp1_diffusivity_index(category_dir, system_name)
    block_curves = []
    for block in blocks:
        frames = []
        for window_start in block:
            for fp in diff_index.get(window_start, []):
                try:
                    df = pd.read_csv(fp)
                except Exception:
                    continue
                cols = df.columns
                if "Time (s)" in cols:
                    t = pd.to_numeric(df["Time (s)"], errors="coerce")
                elif "Time (ns)" in cols:
                    t = pd.to_numeric(df["Time (ns)"], errors="coerce") * 1e-9
                else:
                    continue
                if "MSD (m^2)" in cols:
                    msd = pd.to_numeric(df["MSD (m^2)"], errors="coerce")
                elif "MSD (um^2)" in cols:
                    msd = pd.to_numeric(df["MSD (um^2)"], errors="coerce") * 1e-12
                elif "MSD (um)" in cols:
                    msd = pd.to_numeric(df["MSD (um)"], errors="coerce") * 1e-12
                else:
                    continue
                rg = pd.to_numeric(df["Rg"], errors="coerce") if "Rg" in cols else pd.Series(np.nan, index=df.index)
                cur = pd.DataFrame({"Time (s)": t, "MSD (m^2)": msd, "Rg": rg})
                cur.loc[cur["MSD (m^2)"] == 0, "MSD (m^2)"] = np.nan
                cur.loc[cur["Rg"] == 0, "Rg"] = np.nan
                cur["Time_round"] = cur["Time (s)"].round(12)
                frames.append(cur)
        if not frames:
            continue
        all_df = pd.concat(frames, ignore_index=True)
        grouped = all_df.groupby("Time_round")
        mean = grouped.mean(numeric_only=True)
        mean["Time (s)"] = mean.index.values
        block_curves.append(mean[["Time (s)", "MSD (m^2)", "Rg"]].copy())

    if not block_curves:
        return None

    msd_cols = []
    rg_cols = []
    time_master = None
    for idx, df in enumerate(block_curves):
        cur = df.copy()
        cur.index = cur["Time (s)"].round(12)
        if time_master is None:
            time_master = cur.index.to_numpy(dtype=float)
        msd_cols.append(cur["MSD (m^2)"].rename(f"msd_{idx}"))
        rg_cols.append(cur["Rg"].rename(f"rg_{idx}"))

    msd_df = pd.concat(msd_cols, axis=1).sort_index()
    rg_df = pd.concat(rg_cols, axis=1).sort_index()
    out = pd.DataFrame({
        "Time (s)": msd_df.index.to_numpy(dtype=float),
        "MSD (m^2)": msd_df.mean(axis=1),
        "MSD_SEM": msd_df.sem(axis=1),
        "Rg": rg_df.mean(axis=1),
        "Rg_SEM": rg_df.sem(axis=1),
        "N": msd_df.count(axis=1),
    }).reset_index(drop=True).sort_values("Time (s)")
    return out


def stage_g3bp1_diffusivity_files(category_dir: str, system_name: str, blocks: Sequence[Sequence[int]], stage_dir: str) -> None:
    """Copy the raw per-chain G3BP1 diffusivity CSVs used by the blocks into ``stage_dir``."""
    ensure_dir(stage_dir)
    diff_index = build_g3bp1_diffusivity_index(category_dir, system_name)
    for block in blocks:
        for window_start in block:
            for fp in diff_index.get(window_start, []):
                dst = os.path.join(stage_dir, os.path.basename(fp))
                if not os.path.exists(dst):
                    shutil.copy2(fp, dst)


def discover_domain_pairs(category_dir: str, system_name: str) -> List[Tuple[str, str, int, int]]:
    """Enumerate available domain-contact biopolymer pairs for a system.

    Scans ``Domain_Contacts_Total_<system>_<bioI>_<bioJ>_<t>.csv[.npz]`` files
    and returns sorted ``(bio_i, bio_j, len_i, len_j)`` tuples (residue lengths
    from ``DOMAIN_BIOPOLYMER_LENGTHS``).
    """
    patterns = [
        os.path.join(category_dir, f"Domain_Contacts_Total_{system_name}_*.csv"),
        os.path.join(category_dir, f"Domain_Contacts_Total_{system_name}_*.csv.npz"),
    ]
    files: List[str] = []
    for pattern in patterns:
        files.extend(glob(pattern))
    pairs = set()
    bio_lengths = dict(zip(DOMAIN_BIOPOLYMER_LIST, DOMAIN_BIOPOLYMER_LENGTHS))
    for fp in files:
        base = os.path.basename(fp)
        if not base.startswith(f"Domain_Contacts_Total_{system_name}_"):
            continue
        if base.endswith(".csv.npz"):
            tail = base[len(f"Domain_Contacts_Total_{system_name}_"):-len(".csv.npz")]
        elif base.endswith(".csv"):
            tail = base[len(f"Domain_Contacts_Total_{system_name}_"):-len(".csv")]
        else:
            continue
        tokens = tail.split("_")
        if not tokens:
            continue
        time_idx = None
        for idx in range(len(tokens) - 1, -1, -1):
            try:
                float(tokens[idx])
                time_idx = idx
                break
            except Exception:
                continue
        if time_idx is None or time_idx < 2:
            continue
        bio_tokens = tokens[:time_idx]
        found = None
        for i in range(1, len(bio_tokens)):
            a = "_".join(bio_tokens[:i])
            b = "_".join(bio_tokens[i:])
            if a in bio_lengths and b in bio_lengths:
                found = (a, b)
                break
        if found is None:
            continue
        a, b = found
        pairs.add((a, b, bio_lengths[a], bio_lengths[b]))
    return sorted(pairs)


def load_domain_submatrix(category_dir: str, system_name: str, bio_i: str, bio_j: str, p: int, q: int, window_start: int) -> Optional[np.ndarray]:
    """Load and sub-block-average one domain-contact matrix for a window.

    Prefers the canonical ``bio_i_bio_j`` file; falls back to the transposed
    ``bio_j_bio_i`` file (transposing the data). Returns the pxq sub-block
    average via ``reshape_to_subarrays``, or None if no file exists.
    """
    canonical = None
    alternate = None
    for tag in time_candidates(window_start):
        candidates = [
            os.path.join(category_dir, f"Domain_Contacts_Total_{system_name}_{bio_i}_{bio_j}_{tag}.csv"),
            os.path.join(category_dir, f"Domain_Contacts_Total_{system_name}_{bio_i}_{bio_j}_{tag}.csv.npz"),
        ]
        alternates = [
            os.path.join(category_dir, f"Domain_Contacts_Total_{system_name}_{bio_j}_{bio_i}_{tag}.csv"),
            os.path.join(category_dir, f"Domain_Contacts_Total_{system_name}_{bio_j}_{bio_i}_{tag}.csv.npz"),
        ]
        if canonical is None:
            for c in candidates:
                if os.path.isfile(c):
                    canonical = c
                    break
        if alternate is None:
            for a in alternates:
                if os.path.isfile(a):
                    alternate = a
                    break
    path = canonical or alternate
    if path is None:
        return None
    if path.endswith(".npz"):
        with np.load(path) as data:
            arr = data["arr_0"]
    else:
        arr = np.loadtxt(path, delimiter=",", dtype=float)
    if path == alternate:
        arr = arr.T
    return reshape_to_subarrays(arr, p, q)


def average_domain_blocks(category_dir: str, system_name: str, blocks: Sequence[Sequence[int]]) -> Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray, int]]:
    """Block-average every domain-contact submatrix pair for a system.

    Returns ``{(bio_i, bio_j): (mean, sem, n_blocks)}`` over the per-block mean
    submatrices.
    """
    out: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray, int]] = {}
    for bio_i, bio_j, p, q in discover_domain_pairs(category_dir, system_name):
        per_block = []
        for block in blocks:
            mats = []
            for window_start in block:
                mat = load_domain_submatrix(category_dir, system_name, bio_i, bio_j, p, q, window_start)
                if mat is not None:
                    mats.append(mat)
            if mats:
                per_block.append(np.mean(np.stack(mats, axis=0), axis=0))
        if per_block:
            out[(bio_i, bio_j)] = matrix_mean_sem(per_block)
    return out


def write_matrix_csv(path: str, mat: np.ndarray) -> None:
    """Write a 2-D array to a comma-delimited headerless CSV."""
    np.savetxt(path, mat, delimiter=",")


def average_density_over_systems(ave_dir: str, systems: Sequence[str], prefix: str, tag: str) -> None:
    """Average per-system block-averaged density profiles into a category file.

    Writes ``Density_Profile_{prefix}_{tag}.csv`` with mean density and
    across-system SEM (falling back to the single system's SEM when n==1).
    """
    frames = []
    sem_frames = []
    distance = None
    x_col = "Distance from center of mass (A)"
    y_col = "Protein density (mg/mL)"
    for system_name in systems:
        fp = os.path.join(ave_dir, f"Density_Profile_{prefix}_{system_name}.csv")
        if not os.path.isfile(fp):
            continue
        df = pd.read_csv(fp)
        if distance is None:
            distance = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
            x_col = df.columns[0]
            y_col = df.columns[1]
        frames.append(pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(dtype=float))
        if "Standard mean error" in df.columns:
            sem_frames.append(pd.to_numeric(df["Standard mean error"], errors="coerce").to_numpy(dtype=float))
    if not frames or distance is None:
        return
    mean, sem, n_systems = average_vectors(frames)
    if n_systems == 1 and sem_frames:
        sem = sem_frames[0]
    out = pd.DataFrame({x_col: distance, y_col: mean, "Standard mean error": sem})
    out.attrs["n_systems"] = n_systems
    out.to_csv(os.path.join(ave_dir, f"Density_Profile_{prefix}_{tag}.csv"), index=False)


def average_cluster_over_systems(ave_dir: str, systems: Sequence[str], prefix: str, tag: str) -> None:
    """Average per-system cluster scalar rows into a category ``Cluster_{prefix}_{tag}.csv``.

    Reports the cross-system mean plus across-system SEM columns and
    ``n_systems``.
    """
    rows = []
    cols = None
    for system_name in systems:
        fp = os.path.join(ave_dir, f"Cluster_{prefix}_{system_name}.csv")
        if not os.path.isfile(fp):
            continue
        df = pd.read_csv(fp)
        if cols is None:
            cols = list(df.columns)
        rows.append(df.iloc[0])
    if not rows:
        return
    n_systems = len(rows)
    df = pd.DataFrame(rows)
    out = pd.DataFrame(df.mean(numeric_only=True)).transpose()
    out["RG SEM"] = df["Largest Droplet Radius of Gyration"].sem() if "Largest Droplet Radius of Gyration" in df.columns else math.nan
    out["ND SEM"] = df["Number of Droplets"].sem() if "Number of Droplets" in df.columns else math.nan
    out["Chains Largest SEM"] = df["Chains in Largest Droplet"].sem() if "Chains in Largest Droplet" in df.columns else math.nan
    out["Mass Largest SEM"] = df["Mass of Largest Droplet (mg)"].sem() if "Mass of Largest Droplet (mg)" in df.columns else math.nan
    out["NE SEM"] = df["Number of External Chains"].sem() if "Number of External Chains" in df.columns else math.nan
    out["ME SEM"] = df["Mass of External Chains"].sem() if "Mass of External Chains" in df.columns else math.nan
    out["n_systems"] = n_systems
    out.to_csv(os.path.join(ave_dir, f"Cluster_{prefix}_{tag}.csv"), index=False)


def average_pca_over_systems(ave_dir: str, systems: Sequence[str], prefix: str, tag: str) -> None:
    """Average per-system PCA shape tables into ``PCA_{prefix}_{tag}.csv``."""
    rows = []
    cols = None
    for system_name in systems:
        fp = os.path.join(ave_dir, f"PCA_{prefix}_{system_name}.csv")
        if not os.path.isfile(fp):
            continue
        row = pd.read_csv(fp).mean(numeric_only=True)
        if cols is None:
            cols = list(row.index)
        rows.append(row.reindex(cols))
    if not rows:
        return
    df = pd.DataFrame(rows)
    if cols is not None:
        df = df.reindex(columns=cols)
    df.to_csv(os.path.join(ave_dir, f"PCA_{prefix}_{tag}.csv"), index=False)


def average_matrix_over_systems(ave_dir: str, systems: Sequence[str], prefix: str, tag: str) -> None:
    """Average per-system contact matrices into category mean and SEM files.

    Writes ``{prefix}_{tag}.csv`` (mean) and a sibling ``*_SEM_{tag}.csv``
    (``_Mean``->``_SEM`` in the prefix).
    """
    mats = []
    for system_name in systems:
        fp = os.path.join(ave_dir, f"{prefix}_{system_name}.csv")
        if not os.path.isfile(fp):
            continue
        mats.append(read_matrix_csv(fp))
    if not mats:
        return
    mean, sem, _ = matrix_mean_sem(mats)
    write_matrix_csv(os.path.join(ave_dir, f"{prefix}_{tag}.csv"), mean)
    sem_prefix = prefix.replace("_Mean", "_SEM")
    write_matrix_csv(os.path.join(ave_dir, f"{sem_prefix}_{tag}.csv"), sem)


def average_domain_over_systems(ave_dir: str, systems: Sequence[str], tag: str) -> None:
    """Average per-system domain-contact matrices into category mean/SEM files.

    Groups ``Domain_Contacts_Mean_<system>_<bioI>_<bioJ>.csv`` by biopolymer
    pair and writes ``Domain_Contacts_Mean_{tag}_*`` and ``..._SEM_{tag}_*``.
    """
    pair_to_mats: Dict[Tuple[str, str], List[np.ndarray]] = {}
    pattern = os.path.join(ave_dir, "Domain_Contacts_Mean_*_*.csv")
    for system_name in systems:
        for fp in glob(os.path.join(ave_dir, f"Domain_Contacts_Mean_{system_name}_*.csv")):
            base = os.path.basename(fp)
            tail = base[len(f"Domain_Contacts_Mean_{system_name}_"):-len(".csv")]
            tokens = tail.split("_")
            found = None
            for i in range(1, len(tokens)):
                a = "_".join(tokens[:i])
                b = "_".join(tokens[i:])
                if a in DOMAIN_BIOPOLYMER_LIST and b in DOMAIN_BIOPOLYMER_LIST:
                    found = (a, b)
                    break
            if found is None:
                continue
            pair_to_mats.setdefault(found, []).append(np.loadtxt(fp, delimiter=","))
    for (bio_i, bio_j), mats in pair_to_mats.items():
        if mats:
            mean, sem, _ = matrix_mean_sem(mats)
            write_matrix_csv(os.path.join(ave_dir, f"Domain_Contacts_Mean_{tag}_{bio_i}_{bio_j}.csv"), mean)
            write_matrix_csv(os.path.join(ave_dir, f"Domain_Contacts_SEM_{tag}_{bio_i}_{bio_j}.csv"), sem)


def additive_conc(ave_dir: str, tag: str) -> None:
    """Recompute category SG / Protein / RNA density profiles additively.

    Sums the six protein density profiles (with error propagation) and the RNA
    profile to rewrite ``Density_Profile_{SG,Protein,RNA}_{tag}.csv`` so that
    SG = Protein + RNA on a common distance grid.
    """
    rna_fp = os.path.join(ave_dir, f"Density_Profile_RNA_{tag}.csv")
    if not os.path.isfile(rna_fp):
        return
    df_rna = pd.read_csv(rna_fp)
    distances = pd.to_numeric(df_rna.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    rna_density = pd.to_numeric(df_rna.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
    rna_sem = pd.to_numeric(df_rna["Standard mean error"], errors="coerce").to_numpy(dtype=float)

    protein_density = np.zeros_like(rna_density)
    protein_sem_sq = np.zeros_like(rna_density)
    for protein in ["G3BP1", "FUS", "PABP1", "TDP43", "TIA1", "TTP"]:
        fp = os.path.join(ave_dir, f"Density_Profile_{protein}_{tag}.csv")
        if not os.path.isfile(fp):
            continue
        df = pd.read_csv(fp)
        protein_density += pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
        protein_sem_sq += pd.to_numeric(df["Standard mean error"], errors="coerce").to_numpy(dtype=float) ** 2
    protein_sem = np.sqrt(protein_sem_sq)
    sg_density = protein_density + rna_density
    sg_sem = np.sqrt(protein_sem ** 2 + rna_sem ** 2)

    pd.DataFrame({
        "Distance from center of mass (A)": distances,
        "Protein density (mg/mL)": sg_density,
        "Standard mean error": sg_sem,
    }).to_csv(os.path.join(ave_dir, f"Density_Profile_SG_{tag}.csv"), index=False)

    pd.DataFrame({
        "Distance from center of mass (A)": distances,
        "Protein density (mg/mL)": protein_density,
        "Standard mean error": protein_sem,
    }).to_csv(os.path.join(ave_dir, f"Density_Profile_Protein_{tag}.csv"), index=False)

    pd.DataFrame({
        "Distance from center of mass (A)": distances,
        "Protein density (mg/mL)": rna_density,
        "Standard mean error": rna_sem,
    }).to_csv(os.path.join(ave_dir, f"Density_Profile_RNA_{tag}.csv"), index=False)


def build_system_averages(source_root: str, out_root: str, category: str, system_name: str, blocks: Sequence[Sequence[int]]) -> None:
    """Write all block-averaged ``_AVE`` outputs for one system.

    Block-averages density profiles, PCA, cluster scalars, residue/acid/SM
    contact maps (mean+SEM), G3BP1 diffusivity, and domain-contact submatrices,
    writing each into ``{out_root}/ANALYSIS_{category}_AVE/``.
    """
    raw_dir = os.path.join(source_root, f"ANALYSIS_{category}")
    ave_dir = os.path.join(out_root, f"ANALYSIS_{category}_AVE")
    ensure_dir(ave_dir)
    ensure_dir(os.path.join(ave_dir, "DIFFUSIVITY"))

    density_prefixes = ["Protein", "RNA", "SG"] + (["SM"] if category in {"DSM", "NDSM"} else [])
    for prefix in density_prefixes:
        df = average_density_blocks(raw_dir, system_name, prefix, blocks)
        if df is not None:
            df.to_csv(os.path.join(ave_dir, f"Density_Profile_{prefix}_{system_name}.csv"), index=False)

    for prefix in BIOPOLYMER_DENSITY_LIST:
        df = average_density_blocks(raw_dir, system_name, prefix, blocks)
        if df is not None:
            df.to_csv(os.path.join(ave_dir, f"Density_Profile_{prefix}_{system_name}.csv"), index=False)

    for prefix in ["SG", "Protein", "RNA"]:
        df = average_pca_blocks(raw_dir, system_name, prefix, blocks)
        if df is not None:
            df.to_csv(os.path.join(ave_dir, f"PCA_{prefix}_{system_name}.csv"), index=False)

    for prefix in ["SG", "Protein", "RNA"]:
        df = average_cluster_blocks(raw_dir, system_name, prefix, blocks)
        if df is not None:
            df.to_csv(os.path.join(ave_dir, f"Cluster_{prefix}_{system_name}.csv"), index=False)

    for prefix in ["Residue", "Acid"]:
        contact_stats = average_contact_blocks_with_sem(raw_dir, system_name, prefix, blocks)
        if contact_stats is not None:
            mat, sem, _ = contact_stats
            write_matrix_csv(os.path.join(ave_dir, f"{prefix}_Contacts_Mean_{system_name}.csv"), mat)
            write_matrix_csv(os.path.join(ave_dir, f"{prefix}_Contacts_SEM_{system_name}.csv"), sem)

    if category in {"DSM", "NDSM"}:
        residue_sm_stats = average_residue_sm_blocks_with_sem(raw_dir, system_name, blocks)
        if residue_sm_stats is not None:
            residue_sm, residue_sm_sem, _ = residue_sm_stats
            write_matrix_csv(os.path.join(ave_dir, f"Residue_SM_Contacts_Mean_{system_name}.csv"), residue_sm)
            write_matrix_csv(os.path.join(ave_dir, f"Residue_SM_Contacts_SEM_{system_name}.csv"), residue_sm_sem)
        acid_sm_stats = average_contact_blocks_with_sem(raw_dir, system_name, "Acid_SM", blocks)
        if acid_sm_stats is not None:
            acid_sm, acid_sm_sem, _ = acid_sm_stats
            write_matrix_csv(os.path.join(ave_dir, f"Acid_SM_Contacts_Mean_{system_name}.csv"), acid_sm)
            write_matrix_csv(os.path.join(ave_dir, f"Acid_SM_Contacts_SEM_{system_name}.csv"), acid_sm_sem)

    diff_df = average_diffusivity_blocks(raw_dir, system_name, blocks)
    if diff_df is not None:
        diff_df.to_csv(os.path.join(ave_dir, f"G3BP1_Diffusivity_{system_name}.csv"), index=False)
        stage_g3bp1_diffusivity_files(raw_dir, system_name, blocks, os.path.join(ave_dir, "DIFFUSIVITY"))

    domain_mats = average_domain_blocks(raw_dir, system_name, blocks)
    for (bio_i, bio_j), (mat, sem, _) in domain_mats.items():
        write_matrix_csv(os.path.join(ave_dir, f"Domain_Contacts_Mean_{system_name}_{bio_i}_{bio_j}.csv"), mat)
        write_matrix_csv(os.path.join(ave_dir, f"Domain_Contacts_SEM_{system_name}_{bio_i}_{bio_j}.csv"), sem)


def build_category_aggregates(out_root: str, category: str, systems: Sequence[str], tag: str) -> None:
    """Average all per-system ``_AVE`` outputs into category-level aggregates.

    Aggregates density, cluster, PCA, contact and domain products across the
    systems and recomputes the additive SG/Protein/RNA densities.
    """
    ave_dir = os.path.join(out_root, f"ANALYSIS_{category}_AVE")
    for prefix in ["SG", "Protein", "RNA", "SM"]:
        average_density_over_systems(ave_dir, systems, prefix, tag)
    for prefix in BIOPOLYMER_DENSITY_LIST:
        average_density_over_systems(ave_dir, systems, prefix, tag)

    for prefix in ["SG", "Protein", "RNA"]:
        average_cluster_over_systems(ave_dir, systems, prefix, tag)
        average_pca_over_systems(ave_dir, systems, prefix, tag)

    for prefix in ["Residue_Contacts_Mean", "Acid_Contacts_Mean", "Residue_SM_Contacts_Mean", "Acid_SM_Contacts_Mean"]:
        average_matrix_over_systems(ave_dir, systems, prefix, tag)

    average_domain_over_systems(ave_dir, systems, tag)
    additive_conc(ave_dir, tag)


def create_scaffold(out_root: str) -> None:
    """Create the standard FIGURES/IMAGES/RESULTS output-directory tree."""
    for folder in [
        "FIGURES", "FIGURES/RDP", "FIGURES/RESIDUE_CONTACT_MAPS", "FIGURES/ACID_CONTACT_MAPS",
        "FIGURES/SM_CONTACT_MAPS", "FIGURES/PROPERTIES", "FIGURES/TIME",
        "IMAGES", "IMAGES/RDP", "IMAGES/RESIDUE_CONTACT_MAPS", "IMAGES/ACID_CONTACT_MAPS",
        "IMAGES/SM_CONTACT_MAPS", "IMAGES/DYNAMICS",
        "RESULTS", "RESULTS/RDP", "RESULTS/RESIDUE_CONTACT_MAPS", "RESULTS/ACID_CONTACT_MAPS",
        "RESULTS/SM_CONTACT_MAPS", "RESULTS/SUMMARY", "RESULTS/CORRELATION_DIAGNOSTICS",
    ]:
        ensure_dir(os.path.join(out_root, folder))


def infer_system_lists(layout: Dict[str, Dict[str, List[List[int]]]]) -> Tuple[List[str], List[str]]:
    """Return (dsm, ndsm) system-name lists from the layout, or the defaults."""
    dsm = sorted(layout.get("DSM", {}).keys()) or list(DEFAULT_DSM_LIST)
    ndsm = sorted(layout.get("NDSM", {}).keys()) or list(DEFAULT_NDSM_LIST)
    return dsm, ndsm


def resolve_layout_csv(args, out_root: str) -> str:
    """Resolve the recommended block-layout CSV from CLI args or the output root.

    ``block_correlation_diagnostics.py`` writes diagnostics into
    ``{path}/CORRELATION_<source-folder>_<temp>_<dt>_<tmin>_<tmax>/``. For the
    standard correlated workflow this averaging step is launched with
    ``--folder CLASSIFY_CORRELATED`` after diagnostics were launched with
    ``--folder CLASSIFY``, so we also check the conventional sibling
    diagnostics root.
    """
    if args.layout_csv:
        candidate = os.path.abspath(args.layout_csv)
        if not os.path.isfile(candidate):
            raise SystemExit(f"Layout CSV not found: {candidate}")
        return candidate

    candidates = [
        os.path.join(out_root, "RESULTS", "CORRELATION_DIAGNOSTICS", "RECOMMENDED_SYSTEM_BLOCK_LAYOUT.csv"),
        os.path.join(
            os.path.abspath(args.path),
            f"CORRELATION_CLASSIFY_{args.temp}_{args.dt}_{args.tmin}_{args.tmax}",
            "RESULTS",
            "CORRELATION_DIAGNOSTICS",
            "RECOMMENDED_SYSTEM_BLOCK_LAYOUT.csv",
        ),
        os.path.join(
            os.path.abspath(args.path),
            f"CORRELATION_{args.folder}_{args.temp}_{args.dt}_{args.tmin}_{args.tmax}",
            "RESULTS",
            "CORRELATION_DIAGNOSTICS",
            "RECOMMENDED_SYSTEM_BLOCK_LAYOUT.csv",
        ),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise SystemExit(
        "Layout CSV not found. Pass --layout-csv explicitly or run "
        "block_correlation_diagnostics.py first. Checked:\n  - "
        + "\n  - ".join(candidates)
    )


def main() -> None:
    """Parse CLI args and build the blocked ``CLASSIFY_BLOCKED`` averaging root.

    Creates the output scaffold, symlinks the source ANALYSIS dirs, copies the
    diagnostics CSVs, then block-averages every system and category aggregate.
    With ``--plot-only`` it only verifies an existing output root.
    """
    parser = argparse.ArgumentParser(description="Build block-layout-aware ANALYSIS_*_AVE outputs from correlation diagnostics")
    parser.add_argument("--path", required=True, help="Path to TEMP_XXX directory")
    parser.add_argument("--layout-csv", default=None, help="Path to RECOMMENDED_SYSTEM_BLOCK_LAYOUT.csv")
    parser.add_argument("--folder", default="CLASSIFY_BLOCKED", help="Output folder prefix (default: CLASSIFY_BLOCKED)")
    parser.add_argument("--temp", type=int, required=True, help="Temperature in K")
    parser.add_argument("--tmin", type=int, required=True, help="Original post-equilibration tmin in ns")
    parser.add_argument("--dt", type=int, required=True, help="Original window spacing in ns")
    parser.add_argument("--tmax", type=int, required=True, help="Original tmax in ns")
    parser.add_argument("--out-root", default=None, help="Optional explicit output root")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output root if it already exists")
    parser.add_argument("--plot-only", action="store_true", help="No-op for this averaging script; exits after verifying the existing blocked output root")
    args = parser.parse_args()

    source_root = os.path.abspath(args.path)
    out_root = os.path.abspath(args.out_root) if args.out_root else os.path.join(
        source_root,
        f"{args.folder}_{args.temp}_{args.dt}_{args.tmin}_{args.tmax}",
    )

    if args.plot_only:
        if not os.path.isdir(out_root):
            raise SystemExit(f"Analysis root {out_root} does not exist; run full correlated averaging first.")
        print(f"[plot-only] AVERAGE_SIMULATIONS_CORRELATED has no plot products; existing root verified: {out_root}")
        return

    if os.path.exists(out_root):
        if not args.overwrite:
            raise SystemExit(f"Output root already exists: {out_root}. Pass --overwrite to replace it.")
        shutil.rmtree(out_root)

    ensure_dir(out_root)
    create_scaffold(out_root)

    for category in ["SG", "DSM", "NDSM"]:
        src = os.path.join(source_root, f"ANALYSIS_{category}")
        dst = os.path.join(out_root, f"ANALYSIS_{category}")
        if os.path.isdir(src):
            symlink_dir(src, dst)
        ensure_dir(os.path.join(out_root, f"ANALYSIS_{category}_AVE"))

    layout_csv = resolve_layout_csv(args, out_root)
    layout = load_layout(layout_csv)
    dsm_list, ndsm_list = infer_system_lists(layout)
    with open(os.path.join(out_root, "dsm_list.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(dsm_list) + ("\n" if dsm_list else ""))
    with open(os.path.join(out_root, "ndsm_list.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(ndsm_list) + ("\n" if ndsm_list else ""))

    diagnostics_src = os.path.dirname(os.path.abspath(layout_csv))
    diagnostics_dst = os.path.join(out_root, "RESULTS", "CORRELATION_DIAGNOSTICS")
    shutil.copy2(layout_csv, os.path.join(diagnostics_dst, "RECOMMENDED_SYSTEM_BLOCK_LAYOUT.csv"))
    sibling = os.path.join(diagnostics_src, "RECOMMENDED_SYSTEM_BLOCKS.csv")
    copy_if_exists(sibling, os.path.join(diagnostics_dst, "RECOMMENDED_SYSTEM_BLOCKS.csv"))
    for entry in sorted(os.listdir(diagnostics_src)):
        if not entry.endswith('.csv'):
            continue
        src = os.path.join(diagnostics_src, entry)
        dst = os.path.join(diagnostics_dst, entry)
        if os.path.isfile(src) and os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy2(src, dst)

    for category in ["SG", "DSM", "NDSM"]:
        systems = sorted(layout.get(category, {}).keys())
        if not systems:
            continue
        print(f"Building block-aware averages for {category} ({len(systems)} systems)")
        for system_name in systems:
            print(f"  {system_name}")
            build_system_averages(source_root, out_root, category, system_name, layout[category][system_name])

    if layout.get("SG"):
        build_category_aggregates(out_root, "SG", sorted(layout["SG"].keys()), "SG")
    if layout.get("DSM"):
        dsm_systems = sorted(layout["DSM"].keys())
        build_category_aggregates(out_root, "DSM", dsm_systems, "DSM")
        generate_aggregate_rdp_inputs(out_root, "DSM", dsm_systems, aggregate_tag="DSM")
    if layout.get("NDSM"):
        ndsm_systems = sorted(layout["NDSM"].keys())
        build_category_aggregates(out_root, "NDSM", ndsm_systems, "NDSM")
        generate_aggregate_rdp_inputs(out_root, "NDSM", ndsm_systems, aggregate_tag="NDSM")

    print(f"Blocked averaging root written to: {out_root}")


if __name__ == "__main__":
    main()
