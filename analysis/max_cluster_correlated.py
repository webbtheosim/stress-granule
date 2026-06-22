#!/usr/bin/env python3
"""Block-layout-aware MaxCluster composition counting (correlated pipeline).

Pipeline role
-------------
Correlated-pipeline analogue of ``max_cluster.py`` (standard pipeline Step 2).
It rebuilds the per-system biopolymer / amino-acid / nucleotide composition
normalisation tables, but instead of averaging over every 50 ns window it
averages over the correlation-corrected block layout recommended by
``block_correlation_diagnostics.py``. The block structure (which windows belong
to which block) is read from ``RECOMMENDED_SYSTEM_BLOCK_LAYOUT.csv``; per-block
means are formed first and then averaged across blocks, so the output reflects
the same coarse-graining used elsewhere in the correlated pipeline.

Key inputs
----------
- Raw per-window ``Max_Continuous_Cluster_<system>_<t>.txt`` membership files
  under ``{analysis_root}/ANALYSIS_{SG,DSM,NDSM}/``.
- ``RECOMMENDED_SYSTEM_BLOCK_LAYOUT.csv`` (the recommended block layout).
- Amino-acid / nucleotide sequences under ``analysis/sequences/`` by default
  (``*_seq.txt`` and ``RNA.txt``).
- CLI flags: ``--analysis-root`` OR the full
  ``--path/--folder/--temp/--tmin/--dt/--tmax`` set, plus ``--layout-csv`` and
  ``--plot-only``.

Key outputs (written under ``{analysis_root}/ANALYSIS_{cat}_AVE/`` and
``CM_NORM/MAPS/``)
- ``BioNumDF_<system>.csv``, ``BioNum_<system>.csv``, ``BioPolNum_<system>.csv``
- ``AcidNum_<system>.csv``, ``AcidPolNum_<system>.csv``, ``BondNum_<system>.csv``
- Per-category aggregates (tag ``DSM`` / ``NDSM``) and a generic
  ``*_SYSTEM.csv`` normalisation set.

Example invocation
-------------------
    python max_cluster_correlated.py \
        --path TEMP_300 --folder CLASSIFY_BLOCKED --temp 300 \
        --tmin 50 --dt 50 --tmax 2000 \
        --layout-csv .../RECOMMENDED_SYSTEM_BLOCK_LAYOUT.csv \
        --seq-dir analysis/sequences
"""
import argparse
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from average_simulations_correlated import (
    DEFAULT_DSM_LIST,
    DEFAULT_NDSM_LIST,
    load_layout,
    time_candidates,
)

DEFAULT_SEQ_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sequences")
SEQ_BASE_DIR = DEFAULT_SEQ_BASE_DIR
BIOPOLYMER_VECTOR_ORDER = ["G3BP1", "PABP1", "TTP", "TIA1", "TDP43", "FUS", "RNA"]
BIOPOLYMER_DF_ORDER = ["SG", "TDP43", "FUS", "TIA1", "G3BP1", "PABP1", "TTP", "RNA"]
GENERIC_SYSTEM_COUNTS = {
    0: 33,
    1: 16,
    2: 16,
    3: 16,
    4: 16,
    5: 16,
    6: 21,
}


def ensure_dir(path: str) -> None:
    """Create ``path`` (and parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def read_sequence(species: str, seq_dir: Optional[str] = None) -> List[str]:
    """Read one protein/RNA sequence from ``seq_dir`` as uppercase letters."""
    base = seq_dir or SEQ_BASE_DIR
    filename = "RNA.txt" if species == "RNA" else f"{species}_seq.txt"
    path = os.path.join(base, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing sequence file for {species}: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return [char.upper() for char in handle.read() if char.isalpha()]


def resolve_analysis_root(args) -> str:
    """Return the absolute blocked analysis root from CLI args.

    Uses ``--analysis-root`` if given, otherwise derives
    ``{path}/{folder}_{temp}_{dt}_{tmin}_{tmax}``; raises SystemExit if neither
    is fully specified.
    """
    if args.analysis_root:
        return os.path.abspath(args.analysis_root)
    if not all(v is not None for v in [args.path, args.folder, args.temp, args.dt, args.tmin, args.tmax]):
        raise SystemExit("Pass either --analysis-root or the full --path/--folder/--temp/--dt/--tmin/--tmax set")
    return os.path.abspath(
        os.path.join(
            args.path,
            f"{args.folder}_{args.temp}_{args.dt}_{args.tmin}_{args.tmax}",
        )
    )


def resolve_layout_csv(args, analysis_root: str) -> str:
    """Return the path to the recommended block-layout CSV.

    Uses ``--layout-csv`` if given, otherwise looks for
    ``RESULTS/CORRELATION_DIAGNOSTICS/RECOMMENDED_SYSTEM_BLOCK_LAYOUT.csv`` under
    the analysis root; raises SystemExit if it is missing.
    """
    if args.layout_csv:
        return os.path.abspath(args.layout_csv)
    candidate = os.path.join(
        analysis_root,
        "RESULTS",
        "CORRELATION_DIAGNOSTICS",
        "RECOMMENDED_SYSTEM_BLOCK_LAYOUT.csv",
    )
    if not os.path.isfile(candidate):
        raise SystemExit(f"Layout CSV not found: {candidate}")
    return candidate


def infer_system_lists(analysis_root: str, layout: Dict[str, Dict[str, List[List[int]]]]) -> Tuple[List[str], List[str]]:
    """Return (dsm_list, ndsm_list) of system names to process.

    Prefers ``dsm_list.txt`` / ``ndsm_list.txt`` in the analysis root, then the
    layout keys, then the hard-coded ``DEFAULT_DSM_LIST`` / ``DEFAULT_NDSM_LIST``.
    """
    dsm_file = os.path.join(analysis_root, "dsm_list.txt")
    ndsm_file = os.path.join(analysis_root, "ndsm_list.txt")

    if os.path.isfile(dsm_file):
        with open(dsm_file, "r", encoding="utf-8") as handle:
            dsm = [line.strip() for line in handle if line.strip()]
    else:
        dsm = sorted(layout.get("DSM", {}).keys()) or list(DEFAULT_DSM_LIST)

    if os.path.isfile(ndsm_file):
        with open(ndsm_file, "r", encoding="utf-8") as handle:
            ndsm = [line.strip() for line in handle if line.strip()]
    else:
        ndsm = sorted(layout.get("NDSM", {}).keys()) or list(DEFAULT_NDSM_LIST)

    return dsm, ndsm


def resolve_max_cluster_file(raw_dir: str, system_name: str, window_start: int) -> Optional[str]:
    """Locate the ``Max_Continuous_Cluster`` membership file for one window.

    Tries each time-label spelling and a legacy ``_acid``->``acid`` filename
    variant; returns the matching path or None.
    """
    for tag in time_candidates(window_start):
        primary = os.path.join(raw_dir, f"Max_Continuous_Cluster_{system_name}_{tag}.txt")
        legacy = os.path.join(raw_dir, f"Max_Continuous_Cluster_{system_name.replace('_acid', 'acid')}_{tag}.txt")
        if os.path.isfile(primary):
            return primary
        if os.path.isfile(legacy):
            return legacy
    return None


def parse_residue_list(path: str) -> Optional[List[int]]:
    """Parse the MDAnalysis ``resid ... or resid ...`` selection on line 1.

    Returns the list of integer residue ids forming the max continuous cluster,
    or None if the header is missing or unparseable.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            first = handle.readline().strip()
    except Exception:
        return None
    if not first or "resid" not in first:
        return None
    try:
        tokens = first.split(" or resid ")
        tokens[0] = tokens[0].split("resid ")[1]
        return [int(tok) for tok in tokens if str(tok).strip()]
    except Exception:
        return None


def residue_counts_from_members(residues: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Count cluster members per biopolymer species from their residue ids.

    Returns ``(vector, df_vec)`` where ``vector`` is the 7-species count in
    ``BIOPOLYMER_VECTOR_ORDER`` and ``df_vec`` is the 8-entry count (SG total
    plus 7 species) in ``BIOPOLYMER_DF_ORDER``. Residue-id ranges follow the
    fixed system layout (G3BP1<=33, PABP1<=49, ...).
    """
    counts = {
        "SG": 0.0,
        "TDP43": 0.0,
        "FUS": 0.0,
        "TIA1": 0.0,
        "G3BP1": 0.0,
        "PABP1": 0.0,
        "TTP": 0.0,
        "RNA": 0.0,
    }
    vector = np.zeros(7, dtype=float)

    for resid in residues:
        counts["SG"] += 1.0
        if resid <= 33:
            counts["G3BP1"] += 1.0
            vector[0] += 1.0
        elif resid <= 49:
            counts["PABP1"] += 1.0
            vector[1] += 1.0
        elif resid <= 65:
            counts["TIA1"] += 1.0
            vector[3] += 1.0
        elif resid <= 81:
            counts["TTP"] += 1.0
            vector[2] += 1.0
        elif resid <= 97:
            counts["FUS"] += 1.0
            vector[5] += 1.0
        elif resid <= 113:
            counts["TDP43"] += 1.0
            vector[4] += 1.0
        elif resid <= 135:
            counts["RNA"] += 1.0
            vector[6] += 1.0

    df_vec = np.array([
        counts["SG"],
        counts["TDP43"],
        counts["FUS"],
        counts["TIA1"],
        counts["G3BP1"],
        counts["PABP1"],
        counts["TTP"],
        counts["RNA"],
    ], dtype=float)
    return vector, df_vec


def sem_or_nan(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    """Return the NaN-aware SEM along ``axis``, or NaN if <=1 sample."""
    if arr.shape[axis] <= 1:
        shape = list(arr.shape)
        del shape[axis]
        return np.full(shape or (), np.nan, dtype=float)
    return np.nanstd(arr, axis=axis, ddof=1) / np.sqrt(arr.shape[axis])


def build_block_counts(raw_dir: str, system_name: str, blocks: Sequence[Sequence[int]]):
    """Average per-window species counts within each block, then across blocks.

    For each block, averages the per-window 7- and 8-entry count vectors; then
    returns a dict with the per-block stacks, cross-block mean/SEM (``mean7``,
    ``sem7``, ``mean8``, ``sem8``) and ``n_blocks``. Returns None if no block
    yields a valid window.
    """
    block_vectors: List[np.ndarray] = []
    block_df_vectors: List[np.ndarray] = []

    for block in blocks:
        window_vectors = []
        window_df_vectors = []
        for window_start in block:
            fp = resolve_max_cluster_file(raw_dir, system_name, int(window_start))
            if fp is None:
                continue
            residues = parse_residue_list(fp)
            if not residues or len(residues) <= 2:
                continue
            vec7, vec8 = residue_counts_from_members(residues)
            window_vectors.append(vec7)
            window_df_vectors.append(vec8)
        if window_vectors:
            block_vectors.append(np.mean(np.stack(window_vectors, axis=0), axis=0))
            block_df_vectors.append(np.mean(np.stack(window_df_vectors, axis=0), axis=0))

    if not block_vectors:
        return None

    arr7 = np.stack(block_vectors, axis=0)
    arr8 = np.stack(block_df_vectors, axis=0)
    n_blocks = arr7.shape[0]
    return {
        "block_vectors": arr7,
        "block_df_vectors": arr8,
        "mean7": np.mean(arr7, axis=0),
        "sem7": sem_or_nan(arr7, axis=0),
        "mean8": np.mean(arr8, axis=0),
        "sem8": sem_or_nan(arr8, axis=0),
        "n_blocks": n_blocks,
    }


def write_bionumdf(ave_dir: str, system_name: str, mean8: np.ndarray, sem8: np.ndarray, n_blocks: int = 0) -> None:
    """Write ``BioNumDF_<system>.csv`` with per-biopolymer mean, SEM, n_blocks."""
    df = pd.DataFrame({
        "Biopolymer": BIOPOLYMER_DF_ORDER,
        "Mean": mean8,
        "SEM": sem8,
        "n_blocks": n_blocks,
    })
    df.to_csv(os.path.join(ave_dir, f"BioNumDF_{system_name}.csv"), index=False)


def write_vector(path: str, values: np.ndarray) -> None:
    """Write a 1-D vector as a single headerless CSV row."""
    pd.DataFrame([values]).to_csv(path, index=False, header=False)


def gen_ave_biopolymer_ni_nj(biopolymer_num: Sequence[float]) -> np.ndarray:
    """Build the 7x7 pairwise-count matrix n_i*n_j (diagonal n_i*(n_i-1)/2).

    Gives the number of possible inter-biopolymer (off-diagonal) and
    intra-biopolymer (diagonal) contact pairs used to normalise contact maps.
    """
    n = 7
    bio_arr = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                bio_arr[i, j] = (biopolymer_num[i] * (biopolymer_num[j] - 1.0)) / 2.0
            else:
                bio_arr[i, j] = biopolymer_num[i] * biopolymer_num[j]
    return bio_arr


def gen_acid_contacts(biopolymer_num_dict: Sequence[float], seq_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Build amino-acid / nucleotide normalisation arrays from sequences.

    Given the 7-species biopolymer counts, reads each species' sequence and
    accumulates (a) the 24-type acid count vector weighted by chain copy number,
    (b) its outer product (acid-pair normalisation), and (c) the 24x24
    bonded-neighbour count matrix (i<j within 3 residues; 2 for RNA). Returns
    ``(acid_array, acid_pol, bond_array)``.
    """
    n = 24
    bond_array = np.zeros((n, n), dtype=float)
    acid_array = np.zeros((n,), dtype=float)

    aa_dict = {
        'M': 0, 'G': 1, 'K': 2, 'T': 3, 'R': 4, 'A': 5, 'D': 6, 'E': 7, 'Y': 8, 'V': 9,
        'I': 10, 'Q': 11, 'W': 12, 'F': 13, 'S': 14, 'H': 15, 'N': 16, 'P': 17, 'C': 18, 'L': 19,
    }
    na_dict = {'A': 20, 'C': 21, 'T': 22, 'U': 22, 'G': 23}

    biopolymer_num = {
        "G3BP1": float(biopolymer_num_dict[0]),
        "PABP1": float(biopolymer_num_dict[1]),
        "TTP": float(biopolymer_num_dict[2]),
        "TIA1": float(biopolymer_num_dict[3]),
        "TDP43": float(biopolymer_num_dict[4]),
        "FUS": float(biopolymer_num_dict[5]),
        "RNA": float(biopolymer_num_dict[6]),
    }

    for biopolymer in ["G3BP1", "TDP43", "FUS", "PABP1", "TIA1", "TTP"]:
        bio_array = np.zeros((n, n), dtype=float)
        na_array = np.zeros((n,), dtype=float)
        aa = read_sequence(biopolymer, seq_dir=seq_dir)

        for j in range(len(aa)):
            for k in range(j + 1, j + 4):
                try:
                    if aa_dict[aa[j]] <= aa_dict[aa[k]]:
                        bio_array[aa_dict[aa[j]], aa_dict[aa[k]]] += 1.0
                    else:
                        bio_array[aa_dict[aa[k]], aa_dict[aa[j]]] += 1.0
                except Exception:
                    pass

        for j in range(len(aa)):
            na_array[aa_dict[aa[j]]] += 1.0

        acid_array += na_array * biopolymer_num[biopolymer]
        bond_array += bio_array * biopolymer_num[biopolymer]

    rna_bond = np.zeros((n, n), dtype=float)
    rna_counts = np.zeros((n,), dtype=float)
    na = read_sequence("RNA", seq_dir=seq_dir)

    for j in range(len(na)):
        for k in range(j + 1, j + 3):
            try:
                if na_dict[na[j]] <= na_dict[na[k]]:
                    rna_bond[na_dict[na[j]], na_dict[na[k]]] += 1.0
                else:
                    rna_bond[na_dict[na[k]], na_dict[na[j]]] += 1.0
            except Exception:
                pass
    bond_array += rna_bond * biopolymer_num["RNA"]

    for j in range(len(na)):
        rna_counts[na_dict[na[j]]] += 1.0
    acid_array += rna_counts * biopolymer_num["RNA"]

    for i in range(n):
        for j in range(i + 1, n):
            bond_array[j][i] = bond_array[i][j]

    acid_pol = np.outer(acid_array, acid_array)
    return acid_array, acid_pol, bond_array


def average_system_files(ave_dir: str, systems: Sequence[str], prefix: str, tag: str) -> None:
    """Average per-system ``{prefix}_<system>.csv`` matrices into ``{prefix}_{tag}.csv``."""
    mats = []
    for system_name in systems:
        fp = os.path.join(ave_dir, f"{prefix}_{system_name}.csv")
        if not os.path.isfile(fp):
            continue
        mats.append(pd.read_csv(fp, header=None).to_numpy(dtype=float))
    if not mats:
        return
    mean_mat = np.mean(np.stack(mats, axis=0), axis=0)
    np.savetxt(os.path.join(ave_dir, f"{prefix}_{tag}.csv"), mean_mat, delimiter=",")


def build_system_outputs(analysis_root: str, category: str, system_name: str, blocks: Sequence[Sequence[int]]) -> None:
    """Write all normalisation CSVs for one system from its block layout.

    Produces ``BioNumDF``, ``BioNum``, ``BioPolNum``, ``AcidNum``,
    ``AcidPolNum`` and ``BondNum`` files in the category's ``_AVE`` dir; warns
    and returns if the system has no valid max-cluster windows.
    """
    raw_dir = os.path.join(analysis_root, f"ANALYSIS_{category}")
    ave_dir = os.path.join(analysis_root, f"ANALYSIS_{category}_AVE")
    ensure_dir(ave_dir)

    block_data = build_block_counts(raw_dir, system_name, blocks)
    if block_data is None:
        print(f"[WARN] No valid max-cluster windows for {system_name}")
        return

    write_bionumdf(ave_dir, system_name, block_data["mean8"], block_data["sem8"], block_data["n_blocks"])
    write_vector(os.path.join(ave_dir, f"BioNum_{system_name}.csv"), block_data["mean7"])
    np.savetxt(
        os.path.join(ave_dir, f"BioPolNum_{system_name}.csv"),
        gen_ave_biopolymer_ni_nj(block_data["mean7"]),
        delimiter=",",
    )

    acid_num, acid_pol, bond_num = gen_acid_contacts(block_data["mean7"])
    np.savetxt(os.path.join(ave_dir, f"AcidNum_{system_name}.csv"), acid_num, delimiter=",")
    np.savetxt(os.path.join(ave_dir, f"AcidPolNum_{system_name}.csv"), acid_pol, delimiter=",")
    np.savetxt(os.path.join(ave_dir, f"BondNum_{system_name}.csv"), bond_num, delimiter=",")


def build_category_outputs(analysis_root: str, category: str, systems: Sequence[str], tag: str) -> None:
    """Average each normalisation matrix across all systems in a category."""
    ave_dir = os.path.join(analysis_root, f"ANALYSIS_{category}_AVE")
    for prefix in ["BioNum", "BioPolNum", "AcidNum", "AcidPolNum", "BondNum"]:
        average_system_files(ave_dir, systems, prefix, tag)


def build_generic_system_norm(analysis_root: str) -> None:
    """Write whole-system (full-composition) normalisation tables to CM_NORM/MAPS.

    Uses the fixed ``GENERIC_SYSTEM_COUNTS`` (every chain present) rather than
    cluster membership, giving the denominator for full-system contact maps.
    """
    cm_norm_maps = os.path.join(analysis_root, "CM_NORM", "MAPS")
    ensure_dir(cm_norm_maps)

    generic_vec = np.array([
        GENERIC_SYSTEM_COUNTS[0],
        GENERIC_SYSTEM_COUNTS[1],
        GENERIC_SYSTEM_COUNTS[2],
        GENERIC_SYSTEM_COUNTS[3],
        GENERIC_SYSTEM_COUNTS[4],
        GENERIC_SYSTEM_COUNTS[5],
        GENERIC_SYSTEM_COUNTS[6],
    ], dtype=float)
    write_vector(os.path.join(cm_norm_maps, "BioNum_SYSTEM.csv"), generic_vec)
    np.savetxt(os.path.join(cm_norm_maps, "BioPolNum_SYSTEM.csv"), gen_ave_biopolymer_ni_nj(generic_vec), delimiter=",")
    acid_num, acid_pol, bond_num = gen_acid_contacts(generic_vec)
    np.savetxt(os.path.join(cm_norm_maps, "AcidNum_SYSTEM.csv"), acid_num, delimiter=",")
    np.savetxt(os.path.join(cm_norm_maps, "AcidPolNum_SYSTEM.csv"), acid_pol, delimiter=",")
    np.savetxt(os.path.join(cm_norm_maps, "BondNum_SYSTEM.csv"), bond_num, delimiter=",")


def main() -> None:
    """Parse CLI args and build blocked MaxCluster normalisation outputs.

    Resolves the analysis root and layout CSV, then writes per-system and
    per-category normalisation tables plus the generic system normalisation.
    With ``--plot-only`` it merely verifies the existing root (no plot products).
    """
    parser = argparse.ArgumentParser(description="Build block-layout-aware MaxCluster normalization outputs")
    parser.add_argument("--analysis-root", default=None, help="Explicit blocked analysis root")
    parser.add_argument("--path", default=None, help="Path to TEMP_XXX directory")
    parser.add_argument("--folder", default="CLASSIFY_BLOCKED", help="Output folder prefix used to derive the blocked analysis root")
    parser.add_argument("--temp", type=int, default=None, help="Temperature in K")
    parser.add_argument("--tmin", type=int, default=None, help="Original post-equilibration tmin in ns")
    parser.add_argument("--dt", type=int, default=None, help="Original window spacing in ns")
    parser.add_argument("--tmax", type=int, default=None, help="Original tmax in ns")
    parser.add_argument("--layout-csv", default=None, help="Path to RECOMMENDED_SYSTEM_BLOCK_LAYOUT.csv")
    parser.add_argument("--seq-dir", default=DEFAULT_SEQ_BASE_DIR,
                        help="Directory with <PROT>_seq.txt and RNA.txt sequence inputs")
    parser.add_argument("--plot-only", action="store_true", help="No-op for this count script; exits after verifying the existing analysis root")
    args = parser.parse_args()

    global SEQ_BASE_DIR
    SEQ_BASE_DIR = os.path.abspath(args.seq_dir)
    if not os.path.isdir(SEQ_BASE_DIR):
        raise SystemExit(f"Sequence directory not found: {SEQ_BASE_DIR}")

    analysis_root = resolve_analysis_root(args)
    if not os.path.isdir(analysis_root):
        raise SystemExit(f"Analysis root not found: {analysis_root}")
    if args.plot_only:
        print(f"[plot-only] MAX_CLUSTER_CORRELATED has no plot products; existing root verified: {analysis_root}")
        return
    layout_csv = resolve_layout_csv(args, analysis_root)
    layout = load_layout(layout_csv)
    dsm_list, ndsm_list = infer_system_lists(analysis_root, layout)

    for category in ["SG", "DSM", "NDSM"]:
        if not os.path.isdir(os.path.join(analysis_root, f"ANALYSIS_{category}")):
            print(f"[WARN] Missing raw folder for {category}; skipping")
            continue
        systems = sorted(layout.get(category, {}).keys())
        if not systems:
            print(f"[WARN] No layout entries for {category}; skipping")
            continue
        print(f"Building blocked MaxCluster outputs for {category} ({len(systems)} systems)")
        for system_name in systems:
            print(f"  {system_name}")
            build_system_outputs(analysis_root, category, system_name, layout[category][system_name])

    if layout.get("DSM"):
        build_category_outputs(analysis_root, "DSM", [sm for sm in dsm_list if sm in layout.get("DSM", {})], "DSM")
    if layout.get("NDSM"):
        build_category_outputs(analysis_root, "NDSM", [sm for sm in ndsm_list if sm in layout.get("NDSM", {})], "NDSM")

    build_generic_system_norm(analysis_root)
    print(f"Blocked MaxCluster outputs written under: {analysis_root}")


if __name__ == "__main__":
    main()
