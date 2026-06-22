"""si_tc_spectrum.py - per-compound apparent-Tc spectrum (DSM/NDSM), house style.

Reads the per-compound constrained binodal fits and renders a sorted horizontal
bar chart of the apparent critical temperature for each of the 20 small molecules,
coloured by experimental class, with the SG control Tc as a reference line. This
visualises the DSM/NDSM binary classification as a continuous Tc-shift spectrum
(supports the temperature subsection and the classifier's top feature).

Paper figure: SI 9 (per-compound apparent-Tc bar chart).

Inputs (relative to the run directory, no CLI flags):
  - PHASE_DIAGRAM_CORRELATED_RESULTS/RESULTS/PHASE_DIAGRAM/
        perturbation_individual_binodal_fit.csv  (per-compound Tc fits)
        control_binodal_fit.json                 (SG control Tc reference)

Output figure:
  - FIGURES_SI/temperature_trends_corr/perturbation_Tc_spectrum.png

Style matches si_temperature_trends.py / PLOT_STYLE.md (same colours, fonts,
linewidths, ticks-in).

Run with:
    python si_tc_spectrum.py
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ---- house style ----
sns.set_theme(style="ticks"); sns.set_style("white")
plt.rc("font", size=10); plt.rc("axes", titlesize=10); plt.rc("axes", labelsize=10)
plt.rc("xtick", labelsize=8); plt.rc("ytick", labelsize=8)
plt.rc("legend", fontsize=8); plt.rc("axes", linewidth=2)
COLOR_SG = "#808080"; COLOR_DSM = "#40641b"; COLOR_NDSM = "#bfe49b"

PD = Path("PHASE_DIAGRAM_CORRELATED_RESULTS/RESULTS/PHASE_DIAGRAM")
FIT_CSV = PD / "perturbation_individual_binodal_fit.csv"
CTRL_JSON = PD / "control_binodal_fit.json"
OUT = Path("FIGURES_SI/temperature_trends_corr/perturbation_Tc_spectrum.png")


def main() -> None:
    """Load the per-compound and control binodal fits, then draw and save the
    sorted apparent-Tc horizontal bar chart (DSM/NDSM coloured, SG control line)."""
    df = pd.read_csv(FIT_CSV)
    df = df[df["condition_name"].isin([f"D{i}" for i in range(1, 11)] +
                                      [f"ND{i}" for i in range(1, 11)])].copy()
    df = df.dropna(subset=["Tc_app_pert_ind"]).sort_values("Tc_app_pert_ind")
    with open(CTRL_JSON) as fh:
        sg_tc = json.load(fh)["default_fit"]["parameters"]["Tc_app_control"]

    names = df["condition_name"].tolist()
    tcs = df["Tc_app_pert_ind"].to_numpy(float)
    cols = [COLOR_DSM if t == "DSM" else COLOR_NDSM for t in df["condition_type"]]
    y = np.arange(len(names))

    # taller panel to fit 20 labels; same margins philosophy as RDP panels
    fig = plt.figure(figsize=(3.30, 4.60))
    ax = fig.add_axes([0.62 / 3.30, 0.50 / 4.60, 2.55 / 3.30, 3.95 / 4.60])
    ax.barh(y, tcs, color=cols, edgecolor="k", linewidth=1.0, zorder=3, height=0.72)
    ax.axvline(sg_tc, color=COLOR_SG, lw=2, ls="--", zorder=4)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7)
    ax.set_ylim(-0.7, len(names) - 0.3)
    ax.invert_yaxis()  # lowest Tc (most dissolving) at top
    ax.set_xlabel(r"Apparent $T_c$ (K)")
    ax.set_xlim(300, 420); ax.set_xticks([300, 330, 360, 390, 420])
    ax.tick_params(left=True, right=True, top=True, bottom=True, direction="in",
                   length=4, width=2)
    handles = [Patch(facecolor=COLOR_DSM, edgecolor="k", label="DSM"),
               Patch(facecolor=COLOR_NDSM, edgecolor="k", label="NDSM"),
               Line2D([0], [0], color=COLOR_SG, lw=2, ls="--", label="SG control")]
    ax.legend(handles=handles, frameon=False, loc="upper right",
              handletextpad=0.4, labelspacing=0.35, borderaxespad=0.3, fontsize=7)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=400)
    print("saved:", OUT, "| SG control Tc =", round(sg_tc, 2), "K")


if __name__ == "__main__":
    main()
