"""Per-species CORE composition shift, DSM vs NDSM, ACROSS-COMPOUND error bars.

For compound c, species i:
    f_i^c(core) = sum_{r<=Rcore} rho_i(r) / sum_{r<=Rcore} rho_total(r)
= species i's fraction of total biopolymer density in the condensate core.
Bar = mean_DSM f_i(core) - mean_NDSM f_i(core) (percentage points); error bar =
across-compound SEM (10 DSM, 10 NDSM) in quadrature. RNA is resolved into its
poly-A (A) and coding (U/C/G = UCG) nucleotide classes; 6 proteins + A + UCG is a
complete partition of the biopolymer composition (bars sum to ~0).

Paper figure: Fig 5 panel D (core-composition-shift bars).

Inputs (no CLI flags; ROOT is hard-coded to the 300 K correlated run):
  - TEMP_300/CLASSIFY_CORRELATED_300_50_50_2000/ANALYSIS_{DSM,NDSM}_AVE/
        Density_Profile_*_{dsm,ndsm}_<tag>.csv  (per-compound radial density profiles)

Outputs:
  - <ROOT>/RESULTS/RDP/CORE_COMPOSITION_SHIFT_DSM_NDSM.csv  (per-species sidecar)
  - <ROOT>/FIGURES/RDP/CORE_COMPOSITION_SHIFT_DSM_NDSM_BAR.png  (and /tmp copy)

Run with:
    python build_core_composition_shift.py
"""
import os, glob, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = "TEMP_300/CLASSIFY_CORRELATED_300_50_50_2000"
# Cutoff tied to the fitted SG condensate geometry: half-density radius
# R_cond = 138 A + interface width W = 72 A => r <= 210 A captures the ENTIRE
# condensate including its full interface (density reaches the dilute baseline
# by ~R_cond + W).
RCORE = 210.0

# Fixed order matching the RDP per-species figures (FUS ... RNA, A, UCG)
SPECIES = [("FUS","FUS"),("TIA1","TIA1"),("TDP43","TDP43"),("G3BP1","G3BP1"),
           ("PABP1","PABP1"),("TTP","TTP"),("RNA","RNA"),("A","ADENINE"),("UCG","UCG")]

# exact palette used by the RDP per-species figures
_rk = sns.color_palette("rocket", n_colors=14)
_rna = sns.color_palette(["#0066ff", "#99c2ff", "#38C7C5"], 3)
SPECIES_COLORS = {"FUS":_rk[2],"TIA1":_rk[4],"TDP43":_rk[0],"G3BP1":_rk[6],
                  "PABP1":_rk[8],"TTP":_rk[10],"RNA":_rna[0],"A":_rna[1],"UCG":_rna[2]}

def load(path):
    """Load a 2-column (radius, density) density-profile CSV, skipping the header,
    and return the radius and density arrays."""
    a = np.loadtxt(path, delimiter=",", skiprows=1)
    return a[:,0], a[:,1]

def tags(ave_dir, prefix):
    """List the per-compound file tags (e.g. 'dsm_D1') found in `ave_dir` for the
    given SM prefix, derived from the total-SG density-profile filenames."""
    fs = glob.glob(os.path.join(ROOT, ave_dir, f"Density_Profile_SG_{prefix}_*.csv"))
    return sorted(os.path.basename(f).replace("Density_Profile_SG_","").replace(".csv","") for f in fs)

def core_frac(ave_dir, tag, prof):
    """Fraction of total biopolymer density inside the core (r <= RCORE) carried by
    species `prof` for one compound `tag`. The species profile is interpolated onto
    the total-SG radial grid; returns NaN if either profile is missing or empty."""
    tp = os.path.join(ROOT, ave_dir, f"Density_Profile_SG_{tag}.csv")
    sp = os.path.join(ROOT, ave_dir, f"Density_Profile_{prof}_{tag}.csv")
    if not (os.path.isfile(tp) and os.path.isfile(sp)): return np.nan
    rt, dt = load(tp); rs, ds = load(sp)
    dsi = np.interp(rt, rs, ds); core = rt <= RCORE
    den = np.nansum(dt[core])
    return np.nansum(dsi[core])/den if (np.isfinite(den) and den>0) else np.nan

def fracs(ave_dir, prefix):
    """For every species, the array of per-compound core fractions over all tags in
    `ave_dir`. Returns (dict species_label -> fraction array, list of tags)."""
    tg = tags(ave_dir, prefix)
    return {lab: np.array([core_frac(ave_dir, t, prof) for t in tg], float) for lab, prof in SPECIES}, tg

# ---- Per-compound core fractions, then DSM-minus-NDSM shift per species ----
dsm, dt = fracs("ANALYSIS_DSM_AVE", "dsm")
ndsm, nt = fracs("ANALYSIS_NDSM_AVE", "ndsm")

# rows[i] = (species, mean_DSM_pct, mean_NDSM_pct, delta_pp, across-compound SEM,
#            z = delta/SEM, n_DSM, n_NDSM)
rows = []
for lab, _ in SPECIES:
    d = dsm[lab][np.isfinite(dsm[lab])]; n = ndsm[lab][np.isfinite(ndsm[lab])]
    md, mn = d.mean()*100, n.mean()*100
    sd = d.std(ddof=1)/np.sqrt(len(d))*100; sn = n.std(ddof=1)/np.sqrt(len(n))*100
    dl = md-mn; se = np.hypot(sd, sn); z = dl/se if se>0 else np.nan
    rows.append((lab, md, mn, dl, se, z, len(d), len(n)))

# (no value-sort: fixed SPECIES order to match the RDP per-species figures)

# CSV sidecar
csv_path = f"{ROOT}/RESULTS/RDP/CORE_COMPOSITION_SHIFT_DSM_NDSM.csv"
with open(csv_path, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["species","DSM_core_pct","NDSM_core_pct","delta_pp","sem_pp_across_compound","z","n_DSM","n_NDSM","Rcore_A"])
    for lab, md, mn, dl, se, z, nd, nn in rows:
        w.writerow([lab, f"{md:.4f}", f"{mn:.4f}", f"{dl:.4f}", f"{se:.4f}", f"{z:.3f}", nd, nn, RCORE])

# ---- RDP-matched panel: figsize 3.20x3.20, axis 2.20x2.20 centered ----
sns.set_theme(style="ticks"); sns.set_style("white")
plt.rc("font", size=10); plt.rc("axes", titlesize=10); plt.rc("axes", labelsize=10)
plt.rc("xtick", labelsize=8); plt.rc("ytick", labelsize=8); plt.rc("axes", linewidth=2)
RDP_FIGSIZE = (3.20, 3.20)
RDP_RECT = [0.50/3.20, 0.50/3.20, 2.20/3.20, 2.20/3.20]
fig = plt.figure(figsize=RDP_FIGSIZE)
ax = fig.add_axes(RDP_RECT)

labs = [r[0] for r in rows]; dl = [r[3] for r in rows]; se = [r[4] for r in rows]
x = np.arange(len(labs))
cols = [SPECIES_COLORS[l] for l in labs]
ax.bar(x, dl, yerr=se, color=cols, edgecolor="k", linewidth=1.0,
       error_kw=dict(ecolor="k", elinewidth=1.0, capsize=2, capthick=1.0), zorder=3)
ax.axhline(0, color="k", lw=2, zorder=4)
ax.set_xticks(x)
ax.set_xticklabels(labs, fontsize=8)
ax.set_ylim(-4, 4)
ax.set_yticks([-4, -2, 0, 2, 4])
ax.set_xlim(-0.6, len(labs)-0.4)
ax.tick_params(left=True, right=True, top=True, bottom=True, direction="in", length=4, width=2)
# x-tick labels centred on their tick, rotated 45 deg — exactly the contact-map style
ax.tick_params(axis='x', rotation=45)
for lbl in ax.get_xticklabels():
    lbl.set_ha('center')
    lbl.set_va('top')
    lbl.set_rotation_mode('default')
for outp in [f"{ROOT}/FIGURES/RDP/CORE_COMPOSITION_SHIFT_DSM_NDSM_BAR.png", "/tmp/core_comp_shift.png"]:
    fig.savefig(outp, dpi=400)
print("saved:", csv_path)
print(f"{'sp':>6} {'Δpp':>7} {'sem':>6} {'z':>6}")
for lab, md, mn, d2, s2, z2, nd, nn in rows:
    print(f"{lab:>6} {d2:>+7.2f} {s2:>6.2f} {z2:>+6.1f}{'  *' if abs(z2)>2 else ''}")
