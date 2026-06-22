"""Force-field parameterisation figures (SI Fig S1) for the ANALYSIS_SM pipeline.

R^2 (S1A) and JJ-vs-JK parity (S1C) delegate to the coarse-grained ``plotting/cg``
code (no reimplementation); the Wang-Frenkel interaction curves (S1B) copy the
manuscript style from SequenceAnalysis.py.

Run under the repository analysis environment with pandas / scikit-learn /
scipy / seaborn.

Figures and their data sources (the ground truth, after diagnosis):
  R2_Plot.png (S1A)              r2_analysis.run_r2_analysis on cg_pipeline/R2_Dataset.csv
                                 = the published Table S1 (features.txt run; eps 0.916,
                                 sigma 0.878, mu 0.969, r_c 0.856). [uploaded original data]
  mix_{E,S,V,U,R}_JJ_vs_JK.png   parity_comparer('mix') on RUN_JJ_PARITY_..._184243, which
  (S1C)                          reproduces the published MPiPi-comparison PCC
                                 (eps 0.59, sigma 0.86, r 0.81).
  sm_{E,S,V,U,R}_JJ_vs_JK.png    parity_comparer('sm') on the same run (controls, by species).
  MIX_PARAMS_DSM.png /           per-compound AVERAGED Wang-Frenkel homotypic curves,
  MIX_PARAMS_NDSM.png (S1B)      split DSM/NDSM (manuscript style: DSM=Blues, NDSM=rocket),
                                 from cg_pipeline/drug_parameters.csv [uploaded original data].

Note: the parity (a greedy-selection run) and the R^2 (a fixed features.txt run)
come from different model runs -- that is what reproduces both published value sets.
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# This file lives at <repo>/plotting/sm/, so the repo root is three levels up.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# After the repository reorganisation the coarse-grained plotting code lives in
# plotting/cg/ and its data (R2_Dataset.csv, drug_parameters.csv, mix_params_*)
# lives in cg_pipeline/.
_CG_CODE = os.path.join(_REPO, "plotting", "cg")
CG = os.path.join(_REPO, "cg_pipeline")

# Import the ORIGINAL coarse-grained plotting code verbatim (optional: if its
# heavy dependencies are unavailable the parity/R2 figures simply won't run, but
# importing this module must not fail).
if _CG_CODE not in sys.path:
    sys.path.insert(0, _CG_CODE)
if CG not in sys.path:
    sys.path.insert(0, CG)
try:
    import r2_analysis            # run_r2_analysis  -> R2_Plot.png
    import compare_parameters     # parity_comparer  -> mix_*/sm_*_JJ_vs_JK.png
except Exception as _exc:  # pragma: no cover
    r2_analysis = None
    compare_parameters = None
    print(f"[sm_parameters] WARNING: could not import CG plotting code ({_exc}); "
          f"R2/parity figures will be skipped.")

# The archived run that reproduced the published MPiPi-comparison PCC. After the
# reorganisation the mix-parameter files live directly in cg_pipeline/; fall back
# to that location if the original RUNS archive is absent.
RUN243 = os.path.join(CG, "RUNS", "RUN_JJ_PARITY_20251009_184243")
if not os.path.isdir(RUN243):
    RUN243 = CG


# ----------------------------------------------------------------------------
# Fig S1A: R^2 vs number of descriptors  (verbatim original code + data)
# ----------------------------------------------------------------------------
def fig_r2(outdir, r2_csv=None):
    """Render Fig S1A (R^2 vs descriptor count) via the CG ``r2_analysis`` code.

    Delegates to ``r2_analysis.run_r2_analysis`` on ``R2_Dataset.csv``. Returns
    the output PNG path, or None if the CG plotting code is unavailable.
    """
    if r2_analysis is None:
        print("[sm_parameters] CG plotting code unavailable; skipping R2 figure.")
        return None
    r2_csv = r2_csv or os.path.join(CG, "R2_Dataset.csv")
    r2_analysis.run_r2_analysis(r2_csv, outdir, figsize=_R2_FIGSIZE, ax_rect=_R2_AX_RECT)
    return os.path.join(outdir, "R2_Plot.png")


# ----------------------------------------------------------------------------
# Fig S1C: JJ-vs-JK parity, mix (heterotypic) + sm (homotypic)  (verbatim original)
# ----------------------------------------------------------------------------
# Parity panels: 1.6 x 1.6 in axes (matches the violins), centred in the figure
# with margins wide enough that the y-tick labels are not clipped.
_VIOLIN_FIGSIZE = (2.90, 2.70)
_VIOLIN_AX_RECT = [0.65 / 2.90, 0.55 / 2.70, 1.60 / 2.90, 1.60 / 2.70]

# R^2 panel: square 2.2 x 2.2 in axes (the RDP-plot size), centred.
_R2_FIGSIZE = (3.20, 3.20)
_R2_AX_RECT = [0.50 / 3.20, 0.50 / 3.20, 2.20 / 3.20, 2.20 / 3.20]


def fig_parity(outdir):
    """Render Fig S1C JJ-vs-JK parity panels (mix heterotypic + sm homotypic).

    Delegates to the CG ``compare_parameters.parity_comparer`` for both the
    ``mix`` and ``sm`` parameter sets. Returns the list of PNG paths written
    (empty if the CG plotting code is unavailable).
    """
    if compare_parameters is None:
        print("[sm_parameters] CG plotting code unavailable; skipping parity figures.")
        return []
    made = []
    # mix (heterotypic) parity — the high-point-count / higher-PCC plot
    compare_parameters.parity_comparer(
        "mix", os.path.join(RUN243, "mix_params_JJ.txt"), os.path.join(RUN243, "mix_params_JK.txt")
    ).plot_all(save=True, out_dir=outdir, figsize=_VIOLIN_FIGSIZE, ax_rect=_VIOLIN_AX_RECT)
    made += sorted(glob.glob(os.path.join(outdir, "mix_*_JJ_vs_JK.png")))
    # sm (homotypic) parity — controls, coloured by species
    compare_parameters.parity_comparer(
        "sm", os.path.join(CG, "SM_SM_Parameters_JJ"), os.path.join(RUN243, "SM_SM_Parameters_JK.csv")
    ).plot_all(save=True, out_dir=outdir, figsize=_VIOLIN_FIGSIZE, ax_rect=_VIOLIN_AX_RECT)
    made += sorted(glob.glob(os.path.join(outdir, "sm_*_JJ_vs_JK.png")))
    return made


# ----------------------------------------------------------------------------
# Fig S1B: Wang-Frenkel homotypic interaction curves, separate DSM / NDSM panels
# (manuscript style, copied from SequenceAnalysis.py)
# ----------------------------------------------------------------------------
def _wf_phi(eps, sig, v, mu, rc, r):
    """Wang-Frenkel homotypic potential (SequenceAnalysis.py / manuscript form)."""
    ratio = rc / sig
    denom = 2 * v * (np.power(ratio, 2 * mu) - 1)
    if abs(denom) < 1e-12:
        return np.zeros_like(r)
    alpha = 2 * v * np.power(ratio, 2 * mu) * np.power((1 + 2 * v) / denom, 2 * v + 1)
    phi = eps * alpha * (np.power(sig / r, 2 * mu) - 1) * np.power(np.power(rc / r, 2 * mu) - 1, 2 * v)
    return np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)


def _rocket_partition(cls, n):
    """DSM = purple/dark end of rocket; NDSM = orange/light end (matches SI_1 B)."""
    full = sns.color_palette("rocket", 20)
    return list(full[:n]) if cls == "DSM" else list(full[20 - n:])


def _interaction_panel(df_avg, cls, prefix, outpath):
    """One manuscript panel (SI_1 B): per-compound averaged homotypic curves.
    Axes 2.2 (x) x 0.8 (y) in, rocket partition (DSM purple / NDSM orange), thick
    smooth curves, no legend, no axis titles."""
    sns.set_theme(style="ticks")
    sns.set_style("white")
    plt.rc("axes", titlesize=10); plt.rc("axes", labelsize=10)
    plt.rc("xtick", labelsize=10); plt.rc("ytick", labelsize=10)
    plt.rc("font", size=10); plt.rc("axes", linewidth=2)
    fig = plt.figure(figsize=(2.92, 1.30))
    ax = fig.add_axes([0.52 / 2.92, 0.34 / 1.30, 2.20 / 2.92, 0.80 / 1.30])   # 2.2 x 0.8 in
    ax.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True,
                   direction="in", length=4, width=2)
    r = np.linspace(4, 22, 400)                                              # dense -> smooth
    cols = _rocket_partition(cls, max(len(df_avg), 1))
    for i, (_, row) in enumerate(df_avg.iterrows()):
        phi = _wf_phi(row["E"], row["S"], row["V"], row["U"], row["R"], r)
        ax.plot(r, phi, color=cols[i], linewidth=2.2)
    ax.set_ylim(-0.4, 0.6)
    ax.set_xlim(4, 20)
    ax.set_xticks(np.arange(4, 21, 4))
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title("")
    fig.savefig(outpath, dpi=400)
    plt.close(fig)
    return outpath


def fig_interaction(outdir):
    """SI Fig S1B: separate DSM / NDSM panels of per-compound AVERAGED homotypic
    interaction curves (mean over each compound's beads); thin ~3:1 panels with the
    rocket partition (DSM = purple end, NDSM = orange end)."""
    df = pd.read_csv(os.path.join(CG, "drug_parameters.csv"))
    avg = (df.groupby("Biomolecule")
             .agg({"E": "mean", "S": "mean", "V": "mean", "U": "mean", "R": "mean",
                   "Type": "first", "Molecule Number": "min"})
             .sort_values("Molecule Number"))
    dsm = avg[avg["Type"] == "DSM"]
    ndsm = avg[avg["Type"] == "NDSM"]
    return [
        _interaction_panel(dsm, "DSM", "D", os.path.join(outdir, "MIX_PARAMS_DSM.png")),
        _interaction_panel(ndsm, "NDSM", "ND", os.path.join(outdir, "MIX_PARAMS_NDSM.png")),
    ]


# ----------------------------------------------------------------------------
def run(outdir="FIGURES/parameters"):
    """Render all SI Fig S1 parameterisation panels into ``outdir``.

    Runs the R^2 (S1A), parity (S1C), and interaction-curve (S1B) figures and
    returns the list of PNG paths that were actually written.
    """
    os.makedirs(outdir, exist_ok=True)
    made = [fig_r2(outdir)]
    made += fig_parity(outdir)
    made += fig_interaction(outdir)
    return [m for m in made if m]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="FIGURES/parameters")
    a = ap.parse_args()
    for m in run(a.outdir):
        print("  ", m)
