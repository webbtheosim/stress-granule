"""
Contact map normalization helpers.

Provides per-residue-pair contact probability P_res and per-chain-pair contact
load K_chain from existing on-disk outputs:

  - Residue_Contacts_Count_*.csv  : <C_AB(t)>_t at biopolymer-species level (7x7)
  - Acid_Contacts_Count_*.csv     : <C_ij(t)>_t at AA/RNA atom-type level (24x24)
  - BioPolNum_*.csv               : cluster-averaged N_chains_A * N_chains_B (7x7)
  - analysis/sequences/{species}_seq.txt and RNA.txt : fixed per-species sequences

No RCC re-run required. All quantities derive from existing files.

Species order (matches rcc_analysis.py:1711 and max_cluster.py):
    0: G3BP1
    1: PABP1
    2: TTP
    3: TIA1
    4: TDP43
    5: FUS
    6: RNA

Atom-type order (matches BIOPOLYMER_ANALYSIS.plot_acid_cpm acid_list):
    1=ARG 2=HIS 3=LYS 4=ASP 5=GLU 6=SER 7=THR 8=ASN 9=GLN 10=CYS
    11=GLY 12=PRO 13=ALA 14=VAL 15=ILE 16=LEU 17=MET 18=PHE 19=TYR 20=TRP
    21=A(RNA) 22=U(RNA) 23=C(RNA) 24=G(RNA)
"""

import os
import re
import numpy as np


SPECIES_ORDER = ["G3BP1", "PABP1", "TTP", "TIA1", "TDP43", "FUS", "RNA"]
SPECIES_LENGTH = {"G3BP1": 466, "PABP1": 636, "TTP": 326, "TIA1": 386,
                  "TDP43": 414, "FUS": 526, "RNA": 840}
SPECIES_SEQ_FILE = {"G3BP1": "G3BP1_seq.txt", "PABP1": "PABP1_seq.txt",
                    "TTP": "TTP_seq.txt", "TIA1": "TIA1_seq.txt",
                    "TDP43": "TDP43_seq.txt", "FUS": "FUS_seq.txt",
                    "RNA": "RNA.txt"}

# LAMMPS atom-type assignments verified from the MPiPi topology file
# (./SIMULATION/Type_Protein_Pure/System_PURE_FUS/sys_FUS.data):
# matched by residue mass for unambiguous AAs, and disambiguated I vs L
# (both mass 113.18) by per-chain atom counts vs FASTA composition.
PROTEIN_LETTER_TO_TYPE = {
    'M': 1, 'G': 2, 'K': 3, 'T': 4, 'R': 5, 'A': 6, 'D': 7, 'E': 8, 'Y': 9, 'V': 10,
    'I': 11, 'Q': 12, 'W': 13, 'F': 14, 'S': 15, 'H': 16, 'N': 17, 'P': 18, 'C': 19, 'L': 20,
}
# RNA bases: types 21=A (329.20), 22=C (305.20), 23=G (345.20), 24=U (306.20),
# verified from residue mass. The FASTA file uses T-letters for some legacy
# reasons; treat T as U.
RNA_LETTER_TO_TYPE = {'A': 21, 'C': 22, 'G': 23, 'U': 24, 'T': 24}
N_ATOM_TYPES = 24


def parse_sequence(path, is_rna):
    """Read a sequence file, return (length, dict[lammps_type] -> count_per_chain)."""
    with open(path, "r") as f:
        raw = f.read()
    seq = re.sub(r"[^A-Za-z]", "", raw).upper()
    mapping = RNA_LETTER_TO_TYPE if is_rna else PROTEIN_LETTER_TO_TYPE
    composition = {}
    for ch in seq:
        if ch not in mapping:
            continue
        t = mapping[ch]
        composition[t] = composition.get(t, 0) + 1
    return len(seq), composition


def load_all_species_composition(cm_norm_dir):
    """Return dict: species -> (length, {lammps_type -> count_per_chain})."""
    out = {}
    for sp in SPECIES_ORDER:
        path = os.path.join(cm_norm_dir, SPECIES_SEQ_FILE[sp])
        L, comp = parse_sequence(path, is_rna=(sp == "RNA"))
        if L != SPECIES_LENGTH[sp]:
            raise ValueError(
                f"Sequence length mismatch for {sp}: file gives {L}, "
                f"expected {SPECIES_LENGTH[sp]}"
            )
        out[sp] = (L, comp)
    return out


def n_chains_from_biopolnum(biopolnum_matrix):
    """
    Solve N_chains_A from BioPolNum diagonal: diag[A,A] = N_A * (N_A - 1) / 2.
    Returns vector of length 7 in SPECIES_ORDER.

    Off-diagonal entries (N_A * N_B) are not used; the diagonal is sufficient.
    """
    mat = np.asarray(biopolnum_matrix, dtype=float)
    if mat.shape != (7, 7):
        raise ValueError(f"BioPolNum expected (7,7), got {mat.shape}")
    diag = np.diag(mat)
    n_chains = 0.5 * (1.0 + np.sqrt(np.maximum(1.0 + 8.0 * diag, 0.0)))
    return n_chains


def n_chain_pairs_matrix(n_chains):
    """
    7x7 matrix: N_chain_pairs(A,B) = N_A * N_B for A!=B, N_A*(N_A-1)/2 for A=B.
    Equivalent to recovering BioPolNum directly.
    """
    n = np.asarray(n_chains, dtype=float)
    out = np.outer(n, n)
    diag_self = n * np.maximum(n - 1.0, 0.0) / 2.0
    np.fill_diagonal(out, diag_self)
    return out


def n_residue_pairs_matrix(n_chains):
    """
    7x7 matrix: N_residue_pairs(A,B) at biopolymer-species level.
    Treats each chain of A as contributing L_A residues, and counts ordered
    (residue_a in any A chain, residue_b in any B chain) pairs:

        A != B:  N_pairs(A,B) = (N_A * L_A) * (N_B * L_B)
        A == B:  N_pairs(A,A) = (N_A * L_A) * (N_A * L_A - 1) / 2
    """
    n = np.asarray(n_chains, dtype=float)
    L = np.array([SPECIES_LENGTH[sp] for sp in SPECIES_ORDER], dtype=float)
    n_res = n * L
    out = np.outer(n_res, n_res)
    diag_self = n_res * np.maximum(n_res - 1.0, 0.0) / 2.0
    np.fill_diagonal(out, diag_self)
    return out


def n_atoms_per_type_vector(n_chains, composition_per_species):
    """
    Return a length-24 vector: N_atoms_of_type_i in cluster (cluster-averaged).

    For each LAMMPS type t in 1..24:
        N_atoms(t) = sum over species s of N_chains[s] * composition[s][t]
    """
    n_atoms = np.zeros(N_ATOM_TYPES, dtype=float)
    for s_idx, sp in enumerate(SPECIES_ORDER):
        _, comp = composition_per_species[sp]
        nc = float(n_chains[s_idx])
        for t, count_per_chain in comp.items():
            n_atoms[t - 1] += nc * count_per_chain
    return n_atoms


def n_atom_pairs_matrix(n_atoms_per_type):
    """
    24x24 matrix: N_atom_pairs(i,j) = N_i * N_j (i!=j), N_i*(N_i-1)/2 (i=j).
    """
    n = np.asarray(n_atoms_per_type, dtype=float)
    out = np.outer(n, n)
    diag_self = n * np.maximum(n - 1.0, 0.0) / 2.0
    np.fill_diagonal(out, diag_self)
    return out


def build_residue_normalizers(biopolnum_csv, cm_norm_dir=None):
    """
    Convenience: from a BioPolNum_*.csv path, return (n_chain_pairs, n_residue_pairs)
    both as 7x7 matrices in SPECIES_ORDER.

    cm_norm_dir is not needed for residue-level normalizers but is accepted
    for symmetry with the acid-level call.
    """
    bio = np.loadtxt(biopolnum_csv, delimiter=",", dtype=float)
    n_chains = n_chains_from_biopolnum(bio)
    return n_chain_pairs_matrix(n_chains), n_residue_pairs_matrix(n_chains)


def build_acid_normalizer(biopolnum_csv, cm_norm_dir):
    """
    From a BioPolNum_*.csv path and the CM_NORM directory holding sequence
    files, return a 24x24 N_atom_pairs matrix in the acid_list LAMMPS-type
    ordering used by BIOPOLYMER_ANALYSIS.plot_acid_cpm.
    """
    bio = np.loadtxt(biopolnum_csv, delimiter=",", dtype=float)
    n_chains = n_chains_from_biopolnum(bio)
    comp = load_all_species_composition(cm_norm_dir)
    n_atoms = n_atoms_per_type_vector(n_chains, comp)
    return n_atom_pairs_matrix(n_atoms)


def safe_divide(num, denom, zero_to=np.nan):
    """Elementwise num / denom, masking denom == 0 (or non-finite) to zero_to."""
    num = np.asarray(num, dtype=float)
    denom = np.asarray(denom, dtype=float)
    out = np.full_like(num, zero_to, dtype=float)
    mask = np.isfinite(denom) & (denom > 0)
    out[mask] = num[mask] / denom[mask]
    return out


def neg_log(arr):
    """Return -ln(arr) with arr <= 0 or non-finite mapped to NaN."""
    a = np.asarray(arr, dtype=float)
    out = np.full_like(a, np.nan, dtype=float)
    mask = np.isfinite(a) & (a > 0)
    out[mask] = -np.log(a[mask])
    return out


def log_ratio(arr_a, arr_b):
    """Return ln(A/B) elementwise with A,B > 0 required, else NaN."""
    a = np.asarray(arr_a, dtype=float)
    b = np.asarray(arr_b, dtype=float)
    out = np.full_like(a, np.nan, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    out[mask] = np.log(a[mask] / b[mask])
    return out


# ---------------------------------------------------------------------------
# Contact membership / enrichment helpers (the two-panel framework).
#
# Contact membership f_ij = C_ij / sum(C). Sums to 1 by construction; reads
# as "of all observed contacts in the cluster, what fraction are pair (i,j)".
# Linear panel = composition.
#
# Random-mixing expectation q_ij = N_possible_ij / sum(N_possible).
# Enrichment E_ij = f_ij / q_ij. Mean (weighted by q) = 1 by construction.
# Log enrichment ln(E_ij) is symmetric about 0 = "as expected from random".
#
# Filter consistency: whatever exclusion is applied to the numerator (C)
# MUST be applied to the denominator (N_possible). Helpers below preserve
# this invariant.
# ---------------------------------------------------------------------------


def contact_membership(C):
    """f_ij = C_ij / sum(C). NaN-safe; returns zeros if sum is zero."""
    C = np.asarray(C, dtype=float)
    s = float(np.nansum(C))
    if not np.isfinite(s) or s <= 0:
        return np.zeros_like(C)
    return C / s


def contact_enrichment(C, N_possible):
    """E_ij = f_ij / q_ij where q_ij = N_possible_ij / sum(N_possible).
    Equivalent to (C_ij / sum_C) / (N_possible_ij / sum_N). NaN where
    N_possible_ij <= 0 or non-finite. The same filter must be applied
    upstream to BOTH C and N_possible.
    """
    f = contact_membership(C)
    N = np.asarray(N_possible, dtype=float)
    sN = float(np.nansum(N))
    out = np.full_like(f, np.nan, dtype=float)
    if not np.isfinite(sN) or sN <= 0:
        return out
    q = N / sN
    mask = np.isfinite(q) & (q > 0) & np.isfinite(f)
    out[mask] = f[mask] / q[mask]
    return out


def log_enrichment(C, N_possible):
    """ln(E_ij). NaN where E_ij is non-finite or non-positive."""
    E = contact_enrichment(C, N_possible)
    out = np.full_like(E, np.nan, dtype=float)
    mask = np.isfinite(E) & (E > 0)
    out[mask] = np.log(E[mask])
    return out


# Bonded-pair counting at the AA-type level (24x24).
EXCLUSION_BONDED = 5


def load_per_position_types(cm_norm_dir, species_order=None):
    """Return {species -> list[LAMMPS_type|None]} reading sequence files
    from the sequence directory."""
    sp_order = species_order or SPECIES_ORDER
    types_per_species = {}
    for sp in sp_order:
        path = os.path.join(cm_norm_dir, SPECIES_SEQ_FILE[sp])
        with open(path, "r") as f:
            seq = re.sub(r"[^A-Za-z]", "", f.read()).upper()
        mapping = RNA_LETTER_TO_TYPE if sp == "RNA" else PROTEIN_LETTER_TO_TYPE
        types_per_species[sp] = [mapping.get(c, None) for c in seq]
    return types_per_species


def n_bonded_within_5_atom(n_chains_vec, types_per_species,
                            species_order=None, n_types=N_ATOM_TYPES):
    """24x24 count of atom pairs (i,j) that are within EXCLUSION_BONDED
    sequence positions on the SAME chain, summed across chains of every
    species. types_per_species: dict species -> list[LAMMPS type per position].
    """
    sp_order = species_order or SPECIES_ORDER
    N_b = np.zeros((n_types, n_types), dtype=float)
    for sp_idx, sp in enumerate(sp_order):
        types_list = types_per_species.get(sp)
        if types_list is None:
            continue
        L = len(types_list)
        X = np.zeros((n_types, L), dtype=float)
        for pos, t in enumerate(types_list):
            if t is None:
                continue
            X[t - 1, pos] = 1.0
        Bmask = np.zeros((L, L), dtype=float)
        for d in range(1, EXCLUSION_BONDED + 1):
            for k in range(L - d):
                Bmask[k, k + d] = 1.0
                Bmask[k + d, k] = 1.0
        contrib = X @ Bmask @ X.T
        np.fill_diagonal(contrib, np.diag(contrib) / 2.0)
        N_b += n_chains_vec[sp_idx] * contrib
    return N_b


def n_intra_total_atom(n_chains_vec, composition_per_species,
                        species_order=None, n_types=N_ATOM_TYPES):
    """24x24 count of ALL atom pairs (i,j) that lie on the SAME chain
    (any sequence separation, including bonded). Summed across all chains
    of every species. The total atom-pair count N_total then satisfies
        N_total = N_inter + N_intra_total
    so the three filter variants can be built as:
        all    = N_total - N_bonded
        inter  = N_total - N_intra_total
        intra  = N_intra_total - N_bonded
    """
    sp_order = species_order or SPECIES_ORDER
    N_intra = np.zeros((n_types, n_types), dtype=float)
    for sp_idx, sp in enumerate(sp_order):
        counts = np.zeros(n_types, dtype=float)
        _, sp_comp = composition_per_species[sp]
        for t, c in sp_comp.items():
            counts[t - 1] = c
        per_chain = np.outer(counts, counts)
        # Unordered self-pair correction on the diagonal
        np.fill_diagonal(per_chain, counts * np.maximum(counts - 1.0, 0.0) / 2.0)
        N_intra += n_chains_vec[sp_idx] * per_chain
    return N_intra


def n_atom_pairs_by_filter(n_chains_vec, composition_per_species,
                            types_per_species, variant,
                            species_order=None, n_types=N_ATOM_TYPES):
    """Filter-aware 24x24 atom-pair count matrix matching the variant
    applied to the numerator. variant in {"all", "inter", "intra"}.
    composition_per_species: {sp -> (length, {LAMMPS_type -> count})}
    types_per_species:        {sp -> list[LAMMPS_type per position]}
    """
    sp_order = species_order or SPECIES_ORDER
    n_atoms = n_atoms_per_type_vector(n_chains_vec, composition_per_species)
    N_total = n_atom_pairs_matrix(n_atoms)
    N_intra = n_intra_total_atom(n_chains_vec, composition_per_species,
                                  species_order=sp_order, n_types=n_types)
    N_bonded = n_bonded_within_5_atom(n_chains_vec, types_per_species,
                                        species_order=sp_order, n_types=n_types)
    if variant == "all":
        return N_total - N_bonded
    if variant == "inter":
        return N_total - N_intra
    if variant == "intra":
        return N_intra - N_bonded
    raise ValueError(f"Unknown filter variant {variant!r}")


def n_residue_pairs_by_filter(n_chains_vec, variant, species_order=None):
    """Filter-aware 7x7 species-level residue-pair count matrix using the
    ORDERED-position-pair convention to match the upstream symmetric
    Domain_Contacts_* L×L matrices (each unique unordered intra-chain pair
    appears at both M[i,j] and M[j,i], so summing the matrix gives 2× the
    unique count; the denominator below uses the corresponding 2× ordered
    count so per-cell ratios are correct).

    Same case logic as the user spec for residue 7x7 maps:
        - A != B: N_possible(A,B) = M_A * M_B   (all interchain, distinct chains).
        - A == B (interchain): N_inter(A,A) = (N_A choose 2) * L_A^2  [ordered]
        - A == B (intrachain nonlocal): N_A * L_A*(L_A-1) [ordered] minus the
          corresponding 2 * sum_{d=1..5}(L_A - d) ordered bonded-pair count.
    """
    sp_order = species_order or SPECIES_ORDER
    n_sp = len(sp_order)
    M = np.array([n_chains_vec[i] * SPECIES_LENGTH[sp_order[i]] for i in range(n_sp)], dtype=float)
    N_inter = np.outer(M, M).astype(float)
    for i, sp in enumerate(sp_order):
        N_i = float(n_chains_vec[i])
        L = float(SPECIES_LENGTH[sp])
        N_inter[i, i] = (N_i * max(N_i - 1.0, 0.0) / 2.0) * (L * L)
    N_intra = np.zeros_like(N_inter)
    N_bonded = np.zeros_like(N_inter)
    for i, sp in enumerate(sp_order):
        N_i = float(n_chains_vec[i])
        L = float(SPECIES_LENGTH[sp])
        # ORDERED conventions to match the doubled upstream symmetric matrix.
        total_intra_pairs_per_chain_ordered = L * (L - 1.0)
        bonded_pairs_per_chain_unordered = 0.0
        for d in range(1, EXCLUSION_BONDED + 1):
            bonded_pairs_per_chain_unordered += max(L - d, 0.0)
        bonded_pairs_per_chain_ordered = 2.0 * bonded_pairs_per_chain_unordered
        N_intra[i, i] = N_i * total_intra_pairs_per_chain_ordered
        N_bonded[i, i] = N_i * bonded_pairs_per_chain_ordered
    N_total = N_inter + N_intra
    if variant == "all":
        return N_total - N_bonded
    if variant == "inter":
        return N_inter
    if variant == "intra":
        return N_intra - N_bonded
    raise ValueError(f"Unknown filter variant {variant!r}")


def n_chain_pairs_by_filter(n_chains_vec, variant, species_order=None):
    """Filter-aware 7x7 species-level CHAIN-PAIR count matrix.

    Unlike n_residue_pairs_by_filter which carries the L_A * L_B chain-length
    factor in the null model, this returns just chain-pair multiplicities so
    enrichment against this null asks 'are species A and B chains preferentially
    paired up?' independent of their lengths. Use this when you want to map
    biopolymer-species interaction propensity at the species level, treating
    chain length as intrinsic biology rather than as opportunity.

      - inter variant: off-diag = n_A * n_B  (distinct interchain pairs across
        species), diag = n_A * (n_A - 1) / 2  (distinct same-species pairs).
      - intra variant: off-diag = 0  (intra-chain pairs cannot cross species),
        diag = n_A  (each chain pairs with itself for intra-chain contacts).
      - all variant: off-diag = n_A * n_B  (same as inter),
        diag = n_A * (n_A - 1) / 2 + n_A  (inter self + intra self).
    """
    sp_order = species_order or SPECIES_ORDER
    n_sp = len(sp_order)
    n_chains = np.asarray(n_chains_vec, dtype=float)
    if variant == "intra":
        N = np.zeros((n_sp, n_sp), dtype=float)
        for i in range(n_sp):
            N[i, i] = n_chains[i]
        return N
    N = np.outer(n_chains, n_chains).astype(float)
    for i in range(n_sp):
        ni = n_chains[i]
        if variant == "inter":
            N[i, i] = ni * max(ni - 1.0, 0.0) / 2.0
        elif variant == "all":
            N[i, i] = ni * max(ni - 1.0, 0.0) / 2.0 + ni
        else:
            raise ValueError(f"Unknown filter variant {variant!r}")
    return N
