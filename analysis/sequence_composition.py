#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sequence-composition counting for paper Figure 1, panels A and B (compute only).

Compute step for the Figure-1 composition panels. Reads the per-protein PR-DOS
files (which carry the one-letter residue identity in their second column) and
the RNA sequence, then counts the amino-acid / nucleotide composition. This
module is render-free (no matplotlib, no figure output): it only produces a tidy CSV
that the companion renderer ``plotting/composition_panels.py`` turns into the
figure.

Two composition views are produced:
  - Per-protein amino-acid counts (one row per protein x residue type), with each
    residue tagged by its physicochemical class (Electrostatic / Polar /
    Hydrophobic / Aromatic) -- this drives the colored bars of panel A.
  - Pooled nucleotide counts for the RNA chain (A/U/C/G; LAMMPS "T" is folded
    into "U") -- this drives panel B.

The residue-type ordering is fixed to the manuscript x-axis order
    R H K D E  S T N Q  C G P A V I L M  F Y W
and the nucleotides to
    A U C G.

Inputs (default ``--seq-dir`` = the shipped ``analysis/sequences/``):
  - ``<PROT>_PRDOS.csv``  : columns (residue index, one-letter code, chain, score)
                            for PROT in TDP43, TTP, TIA1, PABP1, G3BP1, FUS
  - ``RNA.txt``           : raw nucleotide string (newlines ignored)

Output:
  - ``<out>/sequence_composition.csv`` : tidy long-form table with columns
        molecule, kind ("protein"|"rna"), residue, residue_class, count
    Proteins contribute kind="protein" rows; the RNA chain contributes
    kind="rna" rows. Panel A reads the protein rows; panel B reads the rna rows.

CLI:
    python sequence_composition.py [--seq-dir DIR] [--out DIR]

Can also be imported; ``compute_composition()`` returns the tidy DataFrame and
``write_composition_csv()`` writes it.

Adapted (cleaned, de-interactive, processing/plotting split) from the original
``the original SequenceAnalysis.py``.

Created on Wed May 10 19:27:50 2023
@author: jaykaplan
"""

import os
import argparse

import pandas as pd


# Default location of the shipped sequence inputs (alongside this module).
DEFAULT_SEQ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sequences")

# Proteins rendered in panel A (order is cosmetic; rows are tagged by name).
PROTEINS = ["TDP43", "TTP", "TIA1", "PABP1", "G3BP1", "FUS"]

# Fixed amino-acid x-axis order for panel A (manuscript convention).
AA_ORDER = [
    "R", "H", "K", "D", "E",       # Electrostatic (charged)
    "S", "T", "N", "Q",            # Polar
    "C", "G", "P", "A", "V", "I", "L", "M",  # Hydrophobic
    "F", "Y", "W",                 # Aromatic
]

# Physicochemical class per residue, aligned position-for-position with AA_ORDER.
AA_CLASS = [
    "Electrostatic", "Electrostatic", "Electrostatic", "Electrostatic", "Electrostatic",
    "Polar", "Polar", "Polar", "Polar",
    "Hydrophobic", "Hydrophobic", "Hydrophobic", "Hydrophobic",
    "Hydrophobic", "Hydrophobic", "Hydrophobic", "Hydrophobic",
    "Aromatic", "Aromatic", "Aromatic",
]
AA_CLASS_BY_RESIDUE = dict(zip(AA_ORDER, AA_CLASS))

# Fixed nucleotide x-axis order for panel B.
NA_ORDER = ["A", "U", "C", "G"]


def count_protein_residues(prdos_path):
    """Count amino-acid occurrences for one protein from its PR-DOS file.

    The PR-DOS CSV stores the one-letter residue code in its second column
    (index 1); the disorder score columns are ignored here. Counts are returned
    in the fixed ``AA_ORDER`` so every protein yields the same 20 residue rows
    (zero-filled where a residue is absent).

    Args:
        prdos_path: Path to ``<PROT>_PRDOS.csv``.

    Returns:
        dict mapping one-letter residue code -> integer count.
    """
    counts = {aa: 0 for aa in AA_ORDER}
    df = pd.read_csv(prdos_path, header=None)
    for code in df.iloc[:, 1]:
        residue = str(code).strip()
        if residue in counts:
            counts[residue] += 1
    return counts


def count_rna_nucleotides(rna_path):
    """Count A/U/C/G occurrences in the RNA chain.

    The raw file may contain DNA-style "T"; following the original analysis it
    is folded into "U" (the simulations use RNA). Newlines are ignored.

    Args:
        rna_path: Path to ``RNA.txt`` (a raw nucleotide string).

    Returns:
        dict mapping nucleotide (A/U/C/G) -> integer count.
    """
    counts = {na: 0 for na in NA_ORDER}
    with open(rna_path, "r") as handle:
        sequence = handle.read().replace("\n", "")
    for base in sequence:
        base = base.strip()
        if base == "T":
            base = "U"
        if base in counts:
            counts[base] += 1
    return counts


def compute_composition(seq_dir=DEFAULT_SEQ_DIR):
    """Build the tidy long-form composition table for panels A and B.

    Args:
        seq_dir: Directory holding ``<PROT>_PRDOS.csv`` and ``RNA.txt``.

    Returns:
        pandas.DataFrame with columns
        ``molecule, kind, residue, residue_class, count`` -- one block of 20
        protein rows per protein (kind="protein") plus 4 RNA rows
        (kind="rna", residue_class="Nucleotide").
    """
    rows = []

    for protein in PROTEINS:
        prdos_path = os.path.join(seq_dir, "{}_PRDOS.csv".format(protein))
        if not os.path.isfile(prdos_path):
            print("[sequence_composition] WARNING: missing {}; skipping {}".format(
                prdos_path, protein))
            continue
        counts = count_protein_residues(prdos_path)
        for residue in AA_ORDER:
            rows.append({
                "molecule": protein,
                "kind": "protein",
                "residue": residue,
                "residue_class": AA_CLASS_BY_RESIDUE[residue],
                "count": counts[residue],
            })

    rna_path = os.path.join(seq_dir, "RNA.txt")
    if os.path.isfile(rna_path):
        na_counts = count_rna_nucleotides(rna_path)
        for base in NA_ORDER:
            rows.append({
                "molecule": "RNA",
                "kind": "rna",
                "residue": base,
                "residue_class": "Nucleotide",
                "count": na_counts[base],
            })
    else:
        print("[sequence_composition] WARNING: missing {}; panel B will be empty".format(
            rna_path))

    return pd.DataFrame(rows, columns=[
        "molecule", "kind", "residue", "residue_class", "count"])


def write_composition_csv(out_dir, seq_dir=DEFAULT_SEQ_DIR):
    """Compute the composition table and write it to ``sequence_composition.csv``.

    Args:
        out_dir: Directory to create/use for the CSV.
        seq_dir: Sequence-input directory (see ``compute_composition``).

    Returns:
        Path to the written CSV.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = compute_composition(seq_dir=seq_dir)
    csv_path = os.path.join(out_dir, "sequence_composition.csv")
    df.to_csv(csv_path, index=False)
    print("[sequence_composition] wrote {} ({} rows)".format(csv_path, len(df)))
    return csv_path


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Compute amino-acid / nucleotide composition for Figure 1 "
                    "panels A and B (writes a tidy CSV; no plotting).")
    parser.add_argument("--seq-dir", default=DEFAULT_SEQ_DIR,
                        help="Directory with <PROT>_PRDOS.csv and RNA.txt "
                             "(default: shipped analysis/sequences/).")
    parser.add_argument("--out", default=".",
                        help="Output directory for sequence_composition.csv "
                             "(default: current directory).")
    return parser.parse_args()


def main():
    args = _parse_args()
    write_composition_csv(args.out, seq_dir=args.seq_dir)


if __name__ == "__main__":
    main()
