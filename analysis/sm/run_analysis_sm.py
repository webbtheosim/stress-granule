#!/usr/bin/env python
"""run_analysis_sm.py — single driver for the unified ANALYSIS_SM pipeline.

R^2 and parity reuse the original coarse-grained plotting code (plotting/cg/);
the interaction curves and isolation figures are styled to match the manuscript
SI exactly. The rendering stages live under plotting/sm/ (this driver lives under
analysis/sm/ and adds plotting/sm/ to sys.path):

  isolation   : sm_isolation  (SI Fig S1 D/E: green DSM/NDSM mean + SEM band +
                black-edged scatter; rocket per-molecule "_individual" plots)
                -> FIGURES/isolation/{aggregated,dsm,ndsm}
                data: <SM_300>/{DSM,NDSM}/ANALYSIS_SM/*.csv
  parameters  : r2_analysis + compare_parameters(ParityComparer mix & sm) + manuscript
                interaction curves  (R^2, JJ-vs-JK parity, WF curves)
                -> FIGURES/parameters
                data: cg_pipeline/ (R2_Dataset.csv, drug_parameters.csv, ...)

Run under a Python 3.10+ interpreter with pandas / scikit-learn / scipy / seaborn
(the coarse-grained plotting modules use PEP-604 annotations). Example:

  module load anaconda3/2025.12 && conda activate base
  python run_analysis_sm.py                 # both stages -> FIGURES/{isolation,parameters}
  python run_analysis_sm.py --isolation
  python run_analysis_sm.py --parameters
"""
import os
import sys
import argparse

# This driver lives at <repo>/analysis/sm/. The figure-rendering stages it calls
# (sm_isolation, sm_parameters) now live under <repo>/plotting/sm/, so put that
# directory on sys.path. Run from the repo root so default FIGURES/ output and
# the ./PYTHON_ANALYSIS data symlink resolve.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PLOTTING_SM = os.path.join(_REPO_ROOT, "plotting", "sm")
if _PLOTTING_SM not in sys.path:
    sys.path.insert(0, _PLOTTING_SM)
os.chdir(_REPO_ROOT)


def main():
    """Parse flags and run the isolation and/or parameterisation render stages."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--isolation", action="store_true", help="run isolation stage only")
    ap.add_argument("--parameters", action="store_true", help="run parameterisation stage only")
    ap.add_argument("--outdir", default="FIGURES", help="base output dir")
    args = ap.parse_args()

    do_iso = args.isolation or not (args.isolation or args.parameters)
    do_par = args.parameters or not (args.isolation or args.parameters)

    if do_iso:
        print("\n========== ISOLATION (SI Fig S1 D/E style: SEM band + scatter; rocket per-molecule) ==========")
        import sm_isolation
        made = sm_isolation.run(os.path.join(args.outdir, "isolation"))
        print(f"  {len(made)} isolation figures -> {os.path.join(args.outdir, 'isolation')} (aggregated/ dsm/ ndsm/)")

    if do_par:
        print("\n========== PARAMETERISATION (r2_analysis / ParityComparer / manuscript curves) ==========")
        import sm_parameters
        for m in sm_parameters.run(os.path.join(args.outdir, "parameters")):
            print("  ", m)

    print("\nDone.")


if __name__ == "__main__":
    main()
