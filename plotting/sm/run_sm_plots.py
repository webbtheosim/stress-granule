#!/usr/bin/env python3
"""Thin CLI wrapper that runs the DSM/NDSM SM-isolation figure generator.

Convenience entry point around ``sm_plotting.generate_plots``: resolves the
DSM/NDSM analysis directories (from ``--base-dir`` or explicit overrides) and
the output directory, then renders the full SM self-aggregation figure set.

Inputs: ANALYSIS_DSM/ and ANALYSIS_NDSM/ under ``--base-dir`` (the per-window
cluster/RDP/RDF CSVs read by ``sm_plotting``).
Outputs: the aggregated/dsm/ndsm PNG panels under ``--output-dir``.

CLI:
    python run_sm_plots.py [--base-dir .] [--dsm-dir D] [--ndsm-dir N] \
        [--output-dir ANALYSIS_SM_PLOTS] [--dt-ns 1.0 --window-ns 20.0 \
        --trim-start 40.0 --trim-end 200.0 --coarsen-bins 4 --plot-stride-ns 5.0]
"""

import argparse
from pathlib import Path

from sm_plotting import generate_plots


def main():
    """Parse CLI arguments, resolve the cohort/output dirs, and run generate_plots."""
    parser = argparse.ArgumentParser(description="Run solubility plots for DSM and NDSM cohorts")
    parser.add_argument("--base-dir", type=Path, default=Path("."),
                        help="Root directory containing ANALYSIS_DSM and ANALYSIS_NDSM")
    parser.add_argument("--dsm-dir", type=Path, default=None,
                        help="Optional explicit path to DSM analysis directory")
    parser.add_argument("--ndsm-dir", type=Path, default=None,
                        help="Optional explicit path to NDSM analysis directory")
    parser.add_argument("--output-dir", type=Path, default=Path("ANALYSIS_SM_PLOTS"),
                        help="Directory where figures are written")
    parser.add_argument("--dt-ns", type=float, default=1.0,
                        help="Temporal resolution (ns) in cluster CSVs")
    parser.add_argument("--window-ns", type=float, default=20.0,
                        help="Window length represented by each CSV (ns)")
    parser.add_argument("--trim-start", type=float, default=40.0,
                        help="Discard data before this time (ns)")
    parser.add_argument("--trim-end", type=float, default=200.0,
                        help="Discard data after this time (ns)")
    parser.add_argument("--coarsen-bins", type=int, default=4,
                        help="Coarsen RDP by averaging every N consecutive bins")
    parser.add_argument("--plot-stride-ns", type=float, default=5.0,
                        help="Only plot cluster time series every N ns (reduce clutter)")
    args = parser.parse_args()

    base = args.base_dir
    dsm_dir = args.dsm_dir or base / "ANALYSIS_DSM"
    ndsm_dir = args.ndsm_dir or base / "ANALYSIS_NDSM"
    output_dir = args.output_dir if args.output_dir.is_absolute() else base / args.output_dir

    generate_plots(dsm_dir, ndsm_dir, output_dir,
                   dt_ns=args.dt_ns, window_ns=args.window_ns,
                   tmin=args.trim_start, tmax=args.trim_end,
                   coarsen_bins=args.coarsen_bins,
                   plot_stride_ns=args.plot_stride_ns)


if __name__ == "__main__":
    main()
