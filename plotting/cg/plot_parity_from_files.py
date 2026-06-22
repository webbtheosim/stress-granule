"""
Generate mix and small-molecule parity plots from existing data files.

Thin command-line wrapper around ``compare_parameters.parity_comparer`` for
regenerating the CG parameterization parity figures (manuscript Fig S1) outside
the full pipeline, given parameter files that already exist on disk. Builds both
a heterotypic ("mix") comparer and a homotypic ("sm") comparer from the supplied
reference (JJ) and new (JK) files and writes their JJ-vs-JK panels.

Inputs (CLI flags): reference/new mix files, reference/new SM parameter CSVs,
and an optional SM parameters CSV controlling SM legend ordering / membership.
Outputs: ``{outdir}/{mix|sm}_{E,S,V,U,R}_JJ_vs_JK.png`` (written by
``parity_comparer.plot_all``).

CLI:
    python plot_parity_from_files.py --mix-ref JJ --mix-new JK \\
        --sm-ref SM_JJ --sm-new SM_JK --outdir OUT [--sm-parameters sm.csv]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Set

import pandas as pd

# Ensure the sibling plotting module (compare_parameters, now under plotting/cg/)
# is importable regardless of the current working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compare_parameters import parity_comparer


def _load_allowed_sm(sm_parameters_csv: Optional[Path]) -> Optional[Set[str]]:
    """Return the set of Biomolecule names to keep for the SM parity plots.

    Returns None (no filtering) if the path is falsy, missing, lacks a
    ``Biomolecule`` column, or is empty.
    """
    if not sm_parameters_csv:
        return None
    if not sm_parameters_csv.exists():
        return None
    df = pd.read_csv(sm_parameters_csv)
    if 'Biomolecule' not in df.columns:
        return None
    names = df['Biomolecule'].astype(str).tolist()
    return set(names) if names else None


def main() -> None:
    """Parse CLI arguments and write the mix and SM parity plots."""
    parser = argparse.ArgumentParser(description="Create mix and SM parity plots from existing data")
    parser.add_argument('--mix-ref', required=True, type=Path, help='Reference mix_params file (e.g., JJ)')
    parser.add_argument('--mix-new', required=True, type=Path, help='New mix_params file (e.g., JK)')
    parser.add_argument('--sm-ref', required=True, type=Path, help='Reference SM parameters CSV (e.g., JJ)')
    parser.add_argument('--sm-new', required=True, type=Path, help='New SM parameters CSV (e.g., JK)')
    parser.add_argument('--sm-parameters', type=Path, default=None,
                        help='Optional sm_parameters.csv to control SM legend ordering')
    parser.add_argument('--outdir', required=True, type=Path, help='Directory to write parity plots to')
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # Mix parity plots
    mix_comparer = parity_comparer('mix', str(args.mix_ref), str(args.mix_new))
    mix_comparer.plot_all(save=True, out_dir=str(outdir))

    # Small-molecule parity plots
    allowed = _load_allowed_sm(args.sm_parameters)
    sm_comparer = parity_comparer('sm', str(args.sm_ref), str(args.sm_new))
    sm_comparer.plot_all(save=True, out_dir=str(outdir), allowed=allowed)


if __name__ == '__main__':
    main()
