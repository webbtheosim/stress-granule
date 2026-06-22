"""Render S9 (global observable trends) and S10 (per-species occupancy & r/R)
as INDIVIDUAL panels in the *exact* original temperature-trend style
(si_temperature_trends.py), at violin-panel size (3.20x3.20 in, 2.20 axes).

This reuses the original helpers so the panels follow the house plotting rules:
  - tick marks inward on all four sides
  - y-limits floored/ceiled to 2 sig figs, 5 ticks, common x10^N exponent
  - x-axis fixed 285-315 K with 5 K increments
  - markers drawn clip_on=False (sit above the axes)
  - SG/DSM/NDSM legend (S9) or species legend (S10), frameon off, loc best
  - no axis titles (author adds those)

S9 : one panel per global observable + the Delta-phi_D efficacy panel.
S10: occupancy and r/R, one panel per system (SG/DSM/NDSM), species overlaid,
     shared y within each quantity.

Role: figure renderer for SI 9 (global observable trends + apparent-T_c
spectrum) and SI 10 (per-species occupancy and r/R).

Inputs: hardcoded ``TEMP_{T}/CLASSIFY_CORRELATED_{T}_50_50_2000/.../Quant_Data.csv``
for the 7 temperatures, plus ``PI_AUDIT_DATA/per_compound_tc_table.csv`` for the
T_c spectrum. Outputs: PNG panels under ``FIGURES_SI/temperature_figures/``.

This module runs its work at import/exec time (no ``main()``, no CLI args).
Exact invocation:
    python si_temp_figures.py
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from pathlib import Path

import si_temperature_trends as TT   # original style + data helpers

# Match the SI violin axes (KMeans._VP_FIG_SIZE / _VP_AX_RECT): 1.40 x 1.40 in
# axes inside a 2.20 x 2.20 in figure (vs the 2.20-in RDP axes used by default).
TT.RDP_FIGSIZE = (2.20, 2.20)
TT.RDP_AX_RECT = [0.40 / 2.20, 0.30 / 2.20, 1.40 / 2.20, 1.40 / 2.20]
# Fonts sized for the smaller axes (7 temperature ticks must fit). Set after the
# SI_TEMPERATURE_TRENDS import so these rc params win over any it applies.
plt.rc("xtick", labelsize=8); plt.rc("ytick", labelsize=8); plt.rc("legend", fontsize=6)
# No legends -- author adds them. Disable the S9 legend that _plot_*_panel draws.
TT._legend_best = lambda *args, **kwargs: None

OUT = Path("FIGURES_SI/temperature_figures"); OUT.mkdir(parents=True, exist_ok=True)
TEMPS = [285, 290, 295, 300, 305, 310, 315]
CSVS = [Path(f"TEMP_{T}/CLASSIFY_CORRELATED_{T}_50_50_2000/RESULTS/SUMMARY/Quant_Data.csv") for T in TEMPS]
BY = TT._gather(CSVS)

# ---------- S9: global observables ----------
RENAME = {"N_D": "k"}
for slug, vt, st, _short in TT.GLOBAL_OBSERVABLES:
    temps, sg, sgs, dsm, dsms, nd, nds = TT._gather_global_series(BY, vt, st)
    TT._plot_absolute_panel(temps, sg, sgs, dsm, dsms, nd, nds,
                            OUT / f"S9_{RENAME.get(slug, slug)}.png")
# Delta-phi_D (old "S10B") in the same delta style (DSM/NDSM vs control)
for slug, vt, st, _ in TT.GLOBAL_OBSERVABLES:
    if slug == "phi_D":
        temps, sg, sgs, dsm, dsms, nd, nds = TT._gather_global_series(BY, vt, st)
        TT._plot_delta_panel(temps, sg, sgs, dsm, dsms, nd, nds, OUT / "S9_delta_phi_D.png")
print(f"S9: wrote {len(TT.GLOBAL_OBSERVABLES)} global panels + S9_delta_phi_D")

# ---------- S10: per-species occupancy & r/R (species overlaid) ----------
SPECIES = TT.SPECIES_ORDER  # [TDP43, FUS, TIA1, G3BP1, PABP1, TTP, RNA]
_rk = sns.color_palette("rocket", n_colors=14)
SPECIES_COLORS = {"TDP43": _rk[0], "FUS": _rk[2], "TIA1": _rk[4],
                  "G3BP1": _rk[6], "PABP1": _rk[8], "TTP": _rk[10], "RNA": "#0066ff"}

def _species_legend(ax):
    """Draw a 2-column frameless legend with one marker+line entry per species."""
    handles = [Line2D([0], [0], marker="o", color=SPECIES_COLORS[s], lw=1.8, linestyle="-",
                      markerfacecolor=SPECIES_COLORS[s], markeredgecolor="black",
                      markeredgewidth=0.5, markersize=4, label=s) for s in SPECIES]
    ax.legend(handles=handles, frameon=False, loc="best", ncol=2, fontsize=5,
              handletextpad=0.25, labelspacing=0.25, columnspacing=0.6, borderaxespad=0.2)

def _occ_rr(value_tmpl, sig_tmpl, prefix):
    """Render the S10 per-species panels (one per SG/DSM/NDSM) for one quantity.

    Overlays all species on a shared y-range and writes
    ``S10_{prefix}_{system}.png`` for each of SG, DSM, and NDSM.
    """
    series = {sp: TT._gather_species_series(BY, value_tmpl, sig_tmpl, sp) for sp in SPECIES}
    # shared y across every species and every system
    yargs = []
    for sp in SPECIES:
        t, sv, ss, dv, ds, nv, ns = series[sp]
        yargs += [sv, ss, dv, ds, nv, ns]
    y_low, y_high = TT._y_range(*yargs)
    for sysname, (vi, si) in {"SG": (1, 2), "DSM": (3, 4), "NDSM": (5, 6)}.items():
        fig, ax = TT._make_square_axes()
        for sp in SPECIES:
            tup = series[sp]
            TT._draw_series(ax, tup[0], tup[vi], tup[si], SPECIES_COLORS[sp])
        TT._apply_2sig_yticks_sci(ax, y_low, y_high, n_ticks=5)
        TT._style_x_temperature(ax)
        ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title("")
        fig.savefig(OUT / f"S10_{prefix}_{sysname}.png", dpi=400); plt.close(fig)
    print(f"S10: wrote {prefix} (SG/DSM/NDSM), shared y=[{y_low:.3g},{y_high:.3g}]")

# templates from SI_TEMPERATURE_TRENDS.PER_SPECIES_OBSERVABLES
_OCC = next(o for o in TT.PER_SPECIES_OBSERVABLES if o[0] == "Occ")
_RR = next(o for o in TT.PER_SPECIES_OBSERVABLES if o[0] == "rOverR")
_occ_rr(_OCC[1], _OCC[2], "occ")
_occ_rr(_RR[1], _RR[2], "rR")

# ---------- T_c spectrum (S9: long vertical-bar rectangle below the 3 combined plots) ----------
tc_csv = Path("PI_AUDIT_DATA/per_compound_tc_table.csv")
if tc_csv.exists():
    df = pd.read_csv(tc_csv)
    dsm = df.loc[df.exp_class == "DSM", "dT_c_vs_SG"]
    ndsm = df.loc[df.exp_class == "NDSM", "dT_c_vs_SG"]
    dsm_sem = dsm.std(ddof=1) / np.sqrt(len(dsm))
    ndsm_sem = ndsm.std(ddof=1) / np.sqrt(len(ndsm))
    # order compounds by class then number: D1..D10 then ND1..ND10
    order = ([f"D{i}" for i in range(1, 11)] + [f"ND{i}" for i in range(1, 11)])
    ind = df.set_index("compound").loc[order].reset_index()
    # bars: [DSM mean, NDSM mean] then D1..D10, ND1..ND10 -- continuous, no gap
    labels = ["DSM", "NDSM"] + list(ind["compound"])
    vals = [dsm.mean(), ndsm.mean()] + list(ind["dT_c_vs_SG"])
    cls = ["DSM", "NDSM"] + list(ind["exp_class"])
    errs = [dsm_sem, ndsm_sem] + [np.nan] * len(ind)   # error bars where possible (means)
    colmap = {"DSM": TT.COLOR_DSM, "NDSM": TT.COLOR_NDSM}
    colors = [colmap[c] for c in cls]
    xs = np.arange(len(labels), dtype=float)            # continuous
    # geometry: axes 6.0 x 1.40 in (S9 axis height); extra bottom for x labels
    AX_W, AX_H = 6.1, 1.40
    L, R, TOP, BOT = 0.55, 0.12, 0.12, 0.40
    fig_w, fig_h = L + AX_W + R, BOT + AX_H + TOP
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([L / fig_w, BOT / fig_h, AX_W / fig_w, AX_H / fig_h])
    ax.bar(xs, vals, width=0.9, color=colors, edgecolor="black",
           linewidth=1.5, zorder=3)
    # SEM error bars on the class means (only place uncertainty is defined)
    me = [i for i, e in enumerate(errs) if np.isfinite(e)]
    ax.errorbar([xs[i] for i in me], [vals[i] for i in me],
                yerr=[errs[i] for i in me], fmt="none", ecolor="black",
                elinewidth=1.5, capsize=3, capthick=1.5, zorder=5)
    ax.set_ylim(-70, 70); ax.set_yticks([-70, -35, 0, 35, 70])
    ax.axhline(0.0, color=TT.COLOR_SG, lw=2.0, ls="-", zorder=4)  # SG control, centred
    ax.set_xlim(-0.7, xs[-1] + 0.7)
    for s in ax.spines.values():
        s.set_linewidth(2.0)
    # y: thick inward ticks both sides
    ax.tick_params(axis="y", direction="in", length=5, width=2,
                   left=True, right=True, labelsize=8)
    # x labels at the BOTTOM, centred on each tick; bottom ticks face OUT
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=6.5, ha="center")
    ax.tick_params(axis="x", which="major", direction="out", length=5, width=2,
                   bottom=True, top=False, labelbottom=True)
    # top x ticks face IN (no labels) via a twin x-axis
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(xs); ax_top.set_xticklabels([])
    ax_top.tick_params(axis="x", which="major", direction="in", length=5, width=2,
                       top=True, bottom=False, labeltop=False)
    for s in ax_top.spines.values():
        s.set_linewidth(2.0)
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title("")
    fig.savefig(OUT / "S9_Tc_spectrum.png", dpi=400); plt.close(fig)
    print("wrote S9_Tc_spectrum (bottom labels, top-in/bottom-out x ticks)")
