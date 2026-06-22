"""Temperature contact-map FILMSTRIPS for SI.

S11 (SG control, 7 T): every panel is the DEVIATION FROM THE 300 K CONTROL
    map(T) - map(300 K), so the central panel is neutral and the temperature
    effect reads directly (the abundant resident species otherwise dominate and
    make all temperatures look identical).
      A domain        : domain ln-enrichment(T) - (300)
      B species       : MASKED chain ln-enrichment(T) - (300)
                        cells with < MIN_CHAINS in-cluster chains are greyed,
                        because a depleted species (TDP43: 15->1 chains over
                        285->315 K) makes the enrichment denominator collapse and
                        spuriously spike -- a low-count artifact, not physics.
      C acid fraction : acid membership(T) - (300)
      D acid enrich   : acid ln-enrichment(T) - (300)

S12 (DSM - NDSM difference, 3 T): the pre-computed class-difference maps.

All panels: coolwarm centred at 0, shared symmetric scale per row (99th-pct
clip so a single hot cell does not wash out the rest). Layout: a row of equal
square maps with small gaps between them, first map keeps y-tick labels, tick
labels pulled tight to the ticks, shared colour bar fully inside the right
margin (never clipped), x-tick labels on every map, no titles (author adds T).

Paper figures: SI 11 (S11, SG temperature filmstrips) and SI 12 (S12, DSM-NDSM
difference filmstrips). Inputs are the per-temperature CLASSIFY_CORRELATED contact
maps (domain, residue/species, and acid CSVs); outputs are PNG filmstrips written
to FIGURES_SI/temperature_figures/. Run as a script (no CLI flags):
    python si_contact_filmstrips.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import os, sys
# Allow this renderer, when run from the plotting/ folder, to import the shared
# compute/support modules that live in analysis/ (CONTACT_NORMALIZATION).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "analysis"))
import contact_normalization as CN

plt.rc("axes", linewidth=1.2)
OUT = Path("FIGURES_SI/temperature_figures"); OUT.mkdir(parents=True, exist_ok=True)

REF_T = 300
MIN_CHAINS = 5.0    # below this in-cluster chain count the enrichment is unreliable

ACID = ["ARG","HIS","LYS","ASP","GLU","SER","THR","ASN","GLN","CYS","GLY","PRO",
        "ALA","VAL","ILE","LEU","MET","PHE","TYR","TRP","A","U","C","G"]
ACID_ORD = list(range(24))
SPEC_BASE = ["G3BP1","PABP1","TTP","TIA1","TDP43","FUS","RNA"]   # == CN.SPECIES_ORDER
SPEC_ORD = [5, 3, 4, 0, 1, 2, 6]
SPEC_LAB = [SPEC_BASE[i] for i in SPEC_ORD]                       # FUS,TIA1,TDP43,G3BP1,PABP1,TTP,RNA
DOM_LEN = [526, 386, 414, 466, 636, 326, 840]                    # FUS,TIA1,TDP43,G3BP1,PABP1,TTP,RNA
DOM_LAB = ["FUS", "TIA1", "TDP43", "G3BP1", "PABP1", "TTP", "RNA"]
DOM_BND = np.cumsum(DOM_LEN)
DOM_CEN = DOM_BND - np.array(DOM_LEN) / 2.0

# 300 K is the subtraction reference (all-grey panel) -> not shown; it already
# appears in the main text and carries no information here. Applies to S11 AND
# S12 (both are now deviations from the 300 K control).
S11_TEMPS = [285, 290, 295, 305, 310, 315]
S12_TEMPS = [285, 315]
REF_T_S12 = 300

RES = "FIGURES/RESIDUE_CONTACT_MAPS"
ACI = "FIGURES/ACID_CONTACT_MAPS"


def B(T):
    """Base analysis directory (Path) for the CLASSIFY_CORRELATED run at temperature T."""
    return Path(f"TEMP_{T}/CLASSIFY_CORRELATED_{T}_50_50_2000")


def _csv(path):
    """Read a headerless numeric CSV into a float array, or None if it is absent."""
    return pd.read_csv(path, header=None).to_numpy(float) if path.exists() else None


def load_sq(T, subdir, fname, order):
    """Load a square contact matrix for temperature T and reorder its rows/columns
    by `order` (the display permutation). Returns None if missing or wrong shape."""
    M = _csv(B(T) / subdir / fname)
    if M is None or M.shape[0] != len(order):
        return None
    return M[np.ix_(order, order)]


def load_dom(T, sub, fname):
    """Load a per-residue domain contact matrix (no reordering) for temperature T."""
    return _csv(B(T) / sub / fname)


def chain_counts_disp(T):
    """In-cluster chain count per species at temperature T, in display order, used
    to mask low-count species cells. Returns None if the BioPolNum file is missing."""
    M = _csv(B(T) / "ANALYSIS_SG_AVE/BioPolNum_sg_X.csv")
    if M is None:
        return None
    return CN.n_chains_from_biopolnum(M)[SPEC_ORD]   # display order


def _round_1sig_ceil(x):
    """Round a positive magnitude up to 1 significant figure."""
    if not np.isfinite(x) or x <= 0:
        return 1.0
    e = int(np.floor(np.log10(x)))
    m = x / 10.0 ** e
    return float(np.ceil(m) * 10.0 ** e)


# ---------- one filmstrip ----------
FIG_W = 6.1      # default filmstrip width (S12); maps fill the row, square
S11_FIG_W = 6.6  # S11 is ~0.5 in wider (height grows so the maps stay square)
GAP_IN = 0.05   # small gap so maps read as separate panels
L_IN, RGT_IN, TOP_IN, BOT_IN = 0.60, 0.48, 0.30, 0.42   # TOP roomy so the colour-bar x10^n header is not clipped
CBAR_W = 0.065  # thin colour bar (50% of the previous 0.13)


def render(out_slug, mats, labels, *, domain=False, invert_y=False, label_fs=6,
           clip_pct=99.0, show_labels=True, fig_total_w=None, xrot=90, gap_in=None):
    """Render a single-row filmstrip of equal square maps sharing one symmetric
    coolwarm scale (99th-pct clipped, 1-sig-fig range) with a thin shared colour
    bar in the right margin, and save it as `<out_slug>.png`. `domain=True` uses an
    imshow layout with domain-boundary lines; otherwise a seaborn heatmap is used.
    Skips (prints) if any matrix is None."""
    if any(m is None for m in mats):
        print(f"  SKIP {out_slug}: missing matrices"); return
    stack = np.array(mats, dtype=float)
    finite = stack[np.isfinite(stack)]
    a = float(np.nanpercentile(np.abs(finite), clip_pct)) if finite.size else 1.0
    if a <= 0:
        a = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
    cmap = plt.get_cmap("coolwarm").copy(); cmap.set_bad("0.80")   # grey for masked
    v = _round_1sig_ceil(a)                                        # symmetric range, 1 sig fig
    vmin, vmax = -v, v

    _ha = "right" if xrot not in (0, 90) else "center"
    _rm = "anchor" if xrot not in (0, 90) else "default"
    n = len(mats)
    FW = fig_total_w if fig_total_w is not None else FIG_W
    G = gap_in if gap_in is not None else GAP_IN
    # square maps sized so the figure is exactly FW wide (height follows so maps stay square)
    map_in = (FW - L_IN - RGT_IN - (n - 1) * G) / n
    fig_w = FW
    fig_h = BOT_IN + map_in + TOP_IN
    fig = plt.figure(figsize=(fig_w, fig_h))
    for i, M in enumerate(mats):
        left = (L_IN + i * (map_in + G)) / fig_w
        ax = fig.add_axes([left, BOT_IN / fig_h, map_in / fig_w, map_in / fig_h])
        if domain:
            ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower",
                      aspect="equal", interpolation="nearest")
            for b in DOM_BND[:-1]:
                ax.axvline(b - 0.5, color="k", lw=0.4); ax.axhline(b - 0.5, color="k", lw=0.4)
            ax.set_xlim(-0.5, M.shape[1] - 0.5); ax.set_ylim(-0.5, M.shape[0] - 0.5)
            ax.set_xticks(DOM_CEN)
            ax.set_xticklabels(labels, rotation=xrot, fontsize=label_fs, ha=_ha, rotation_mode=_rm)
            if i == 0:
                ax.set_yticks(DOM_CEN); ax.set_yticklabels(labels, rotation=0, fontsize=label_fs)
            else:
                ax.set_yticks([])
        else:
            sns.heatmap(M, cmap=cmap, vmin=vmin, vmax=vmax, center=0.0, ax=ax,
                        cbar=False, square=True,
                        xticklabels=labels, yticklabels=(labels if i == 0 else False))
            if show_labels:
                ax.set_xticklabels(labels, rotation=xrot, fontsize=label_fs, ha=_ha, rotation_mode=_rm)
                if i == 0:
                    ax.set_yticklabels(labels, rotation=0, fontsize=label_fs)
            else:
                ax.set_xticks([]); ax.set_yticks([])
            if invert_y:
                ax.invert_yaxis()
        ax.tick_params(length=2.25, width=1.2, pad=0.6)   # ticks 50% longer, as thick as axes
        for s in ax.spines.values():
            s.set_visible(True); s.set_linewidth(1.2)
    # shared colour bar: thin, 5 ticks (max top, min bottom), scientific notation
    # with a common x10^N header and 1-sig-fig mantissas (house map style).
    cax_left = (L_IN + n * map_in + (n - 1) * G + 0.06) / fig_w
    cax = fig.add_axes([cax_left, BOT_IN / fig_h, CBAR_W / fig_w, map_in / fig_h])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                   norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
    ticks = np.linspace(vmin, vmax, 5)
    cexp = int(np.floor(np.log10(v))) if v > 0 else 0
    scale = 10.0 ** cexp
    cb.set_ticks(ticks)
    cb.ax.set_yticklabels([f"{t / scale:g}" for t in ticks])
    cb.ax.set_title(rf"$\times10^{{{cexp}}}$", fontsize=8, pad=2)
    cb.ax.tick_params(labelsize=8, length=3, width=1.2, pad=1)   # ticks 50% longer, as thick as axes
    cb.outline.set_linewidth(1.2)
    fig.savefig(OUT / f"{out_slug}.png", dpi=400); plt.close(fig)
    print(f"  wrote {out_slug}.png  ({fig_w:.2f} x {fig_h:.2f} in, {n} maps, map {map_in:.2f} in)")


def render_grid(out_slug, mats, labels, *, ncol=3, label_fs=8, clip_pct=99.0,
                invert_y=False, map_in=2.80):
    """Multi-row grid of square maps with ONE shared full-height colour bar.

    Used for the S11 acid map: the 24x24 acid grid needs big maps for font-8
    labels, so the 6 temperatures are laid out 3-across x 2-down rather than as
    a single thin row. y-labels on the left column, x-labels on the bottom row
    (the acids are identical for every panel), colour bar spans both rows.
    """
    if any(m is None for m in mats):
        print(f"  SKIP {out_slug}: missing matrices"); return
    stack = np.array(mats, dtype=float)
    finite = stack[np.isfinite(stack)]
    a = float(np.nanpercentile(np.abs(finite), clip_pct)) if finite.size else 1.0
    if a <= 0:
        a = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
    cmap = plt.get_cmap("coolwarm").copy(); cmap.set_bad("0.80")
    v = _round_1sig_ceil(a); vmin, vmax = -v, v

    n = len(mats)
    nrow = int(np.ceil(n / ncol))
    M = map_in                                  # square map
    L, TOPm, BOTm = 0.60, 0.30, 0.52            # L matches A/B left margin so map blocks align when stacked
    GAP_H, GAP_V = 0.14, 0.14
    CBAR_GAP, CBAR_W, R_PAD = 0.14, 0.11, 0.55
    fig_w = L + ncol * M + (ncol - 1) * GAP_H + CBAR_GAP + CBAR_W + R_PAD
    fig_h = TOPm + nrow * M + (nrow - 1) * GAP_V + BOTm
    fig = plt.figure(figsize=(fig_w, fig_h))
    for i, Mm in enumerate(mats):
        r, c = divmod(i, ncol)
        left = (L + c * (M + GAP_H)) / fig_w
        bottom = (BOTm + (nrow - 1 - r) * (M + GAP_V)) / fig_h   # row 0 on top
        ax = fig.add_axes([left, bottom, M / fig_w, M / fig_h])
        sns.heatmap(Mm, cmap=cmap, vmin=vmin, vmax=vmax, center=0.0, ax=ax,
                    cbar=False, square=True, xticklabels=labels, yticklabels=labels)
        if c == 0:
            ax.set_yticklabels(labels, rotation=0, fontsize=label_fs)
        else:
            ax.set_yticks([])
        if r == nrow - 1:
            ax.set_xticklabels(labels, rotation=90, fontsize=label_fs)
        else:
            ax.set_xticks([])
        if invert_y:
            ax.invert_yaxis()
        ax.tick_params(length=2.25, width=1.2, pad=0.6)
        for s in ax.spines.values():
            s.set_visible(True); s.set_linewidth(1.2)
    # one shared colour bar spanning the full height of both rows
    cb_left = (L + ncol * M + (ncol - 1) * GAP_H + CBAR_GAP) / fig_w
    cb_h = (nrow * M + (nrow - 1) * GAP_V) / fig_h
    cax = fig.add_axes([cb_left, BOTm / fig_h, CBAR_W / fig_w, cb_h])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                   norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
    ticks = np.linspace(vmin, vmax, 5)
    cexp = int(np.floor(np.log10(v))) if v > 0 else 0
    scale = 10.0 ** cexp
    cb.set_ticks(ticks)
    cb.ax.set_yticklabels([f"{t / scale:g}" for t in ticks])
    cb.ax.set_title(rf"$\times10^{{{cexp}}}$", fontsize=8, pad=4)
    cb.ax.tick_params(labelsize=8, length=3, width=1.2, pad=1)
    cb.outline.set_linewidth(1.2)
    fig.savefig(OUT / f"{out_slug}.png", dpi=400); plt.close(fig)
    print(f"  wrote {out_slug}.png  ({fig_w:.2f} x {fig_h:.2f} in, {n} maps grid {nrow}x{ncol}, map {M:.2f} in)")


def render_single(out_slug, mat, labels, *, vmin, vmax, domain=False, invert_y=False,
                  label_fs=6, show_colorbar=True, map_in=1.747):
    """One isolated square map at a GIVEN (shared) colour scale. The colour bar is
    optional so a pair of temperatures can share one bar placed on just one map."""
    if mat is None:
        print(f"  SKIP {out_slug}: missing matrix"); return
    cmap = plt.get_cmap("coolwarm").copy(); cmap.set_bad("0.80")
    # Always reserve the colour-bar gutter, even on the no-bar (285) map, so the
    # with-bar (315) and no-bar (285) PNGs are byte-for-byte the same canvas size
    # and identical axis rect -> scaling both to one slide width keeps the heatmap
    # axes EXACTLY the same size (the only difference is whether the bar is drawn).
    right = RGT_IN
    fig_w = L_IN + map_in + right
    fig_h = BOT_IN + map_in + TOP_IN
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([L_IN / fig_w, BOT_IN / fig_h, map_in / fig_w, map_in / fig_h])
    if domain:
        ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower",
                  aspect="equal", interpolation="nearest")
        for b in DOM_BND[:-1]:
            ax.axvline(b - 0.5, color="k", lw=0.4); ax.axhline(b - 0.5, color="k", lw=0.4)
        ax.set_xlim(-0.5, mat.shape[1] - 0.5); ax.set_ylim(-0.5, mat.shape[0] - 0.5)
        ax.set_xticks(DOM_CEN); ax.set_xticklabels(labels, rotation=90, fontsize=label_fs)
        ax.set_yticks(DOM_CEN); ax.set_yticklabels(labels, rotation=0, fontsize=label_fs)
    else:
        sns.heatmap(mat, cmap=cmap, vmin=vmin, vmax=vmax, center=0.0, ax=ax,
                    cbar=False, square=True, xticklabels=labels, yticklabels=labels)
        ax.set_xticklabels(labels, rotation=90, fontsize=label_fs)
        ax.set_yticklabels(labels, rotation=0, fontsize=label_fs)
        if invert_y:
            ax.invert_yaxis()
    ax.tick_params(length=2.25, width=1.2, pad=0.6)
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(1.2)
    if show_colorbar:
        cax = fig.add_axes([(L_IN + map_in + 0.06) / fig_w, BOT_IN / fig_h,
                            CBAR_W / fig_w, map_in / fig_h])
        cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                       norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
        ticks = np.linspace(vmin, vmax, 5)
        cexp = int(np.floor(np.log10(vmax))) if vmax > 0 else 0
        scale = 10.0 ** cexp
        cb.set_ticks(ticks)
        cb.ax.set_yticklabels([f"{t / scale:g}" for t in ticks])
        cb.ax.set_title(rf"$\times10^{{{cexp}}}$", fontsize=8, pad=2)
        cb.ax.tick_params(labelsize=8, length=3, width=1.2, pad=1)
        cb.outline.set_linewidth(1.2)
    fig.savefig(OUT / f"{out_slug}.png", dpi=400); plt.close(fig)
    print(f"  wrote {out_slug}.png  ({fig_w:.2f} x {fig_h:.2f} in, map {map_in:.2f} in, cbar={show_colorbar})")


def render_pair_split(base, mats, labels, *, domain=False, invert_y=False, label_fs=6,
                      clip_pct=99.0, map_in=1.747, temps=(285, 315)):
    """Split a 2-temperature S12 panel into two standalone maps that SHARE one
    colour scale; the colour bar is drawn only on the higher-temperature map."""
    if any(m is None for m in mats):
        print(f"  SKIP {base}: missing matrices"); return
    stack = np.array(mats, dtype=float)
    finite = stack[np.isfinite(stack)]
    a = float(np.nanpercentile(np.abs(finite), clip_pct)) if finite.size else 1.0
    if a <= 0:
        a = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
    v = _round_1sig_ceil(a); vmin, vmax = -v, v
    for k, (mat, T) in enumerate(zip(mats, temps)):
        is_hi = (k == len(mats) - 1)                 # colour bar only on the last (highest T)
        render_single(f"{base}_{T}", mat, labels, vmin=vmin, vmax=vmax, domain=domain,
                      invert_y=invert_y, label_fs=label_fs, show_colorbar=is_hi, map_in=map_in)


# ============================ S11: deviation from 300 K ============================
print("S11 (SG, 7 temps, minus 300 K control):")

# A domain: ln-enrichment(T) - (300)
DOM_F = "Domain_Contact_Map_FULL_FULL_CONTACT_ENRICHMENT_LOG_INTER.csv"
dom_ref = load_dom(REF_T, "FIGURES/DOMAIN_CONTACT_MAPS/SG", DOM_F)
dom_d = [None if (m := load_dom(T, "FIGURES/DOMAIN_CONTACT_MAPS/SG", DOM_F)) is None or dom_ref is None
         else m - dom_ref for T in S11_TEMPS]
render("S11_A_domain", dom_d, DOM_LAB, domain=True, label_fs=8, fig_total_w=S11_FIG_W)

# B species: MASKED chain ln-enrichment(T) - (300)
F_ENR = "Residue_SG_CONTACT_CHAIN_ENRICHMENT_LOG_INTER_HeatMap.csv"
def masked_enrich(T):
    """Species (chain) ln-enrichment map at T with low-count species cells masked to
    NaN (rows/cols whose in-cluster chain count is below MIN_CHAINS)."""
    M = load_sq(T, RES, F_ENR, SPEC_ORD)
    nc = chain_counts_disp(T)
    if M is None or nc is None:
        return None
    low = nc < MIN_CHAINS
    M = M.copy(); M[low[:, None] | low[None, :]] = np.nan
    return M
me_ref = masked_enrich(REF_T)
spec_d = [None if (m := masked_enrich(T)) is None or me_ref is None else m - me_ref
          for T in S11_TEMPS]
render("S11_B_species", spec_d, SPEC_LAB, invert_y=True, label_fs=8, clip_pct=100.0,
       fig_total_w=S11_FIG_W)

# C acid enrichment: ln-enrichment(T) - (300), laid out as a 2x3 grid so the 24
# acid labels fit at font 8. The acid FRACTION map is dropped: it is dominated by
# raw abundance (the A nucleotide swamps it) and carries the same temperature
# story; enrichment is the composition-normalised, more informative version and
# has no low-count artifact for acids (every acid type is abundant).
AE = "Acid_SG_CONTACT_ENRICHMENT_LOG_INTER_HeatMap.csv"
ae_ref = load_sq(REF_T, ACI, AE, ACID_ORD)
acid_e_d = [None if (m := load_sq(T, ACI, AE, ACID_ORD)) is None or ae_ref is None
            else m - ae_ref for T in S11_TEMPS]
# map-span (3 maps + 2 gaps) = 5.52 in, matching the domain/residue maps exactly;
# at 1.747 in/map the 24 acid labels need font 5 (font 8 would overlap).
render_grid("S11_C_acid_enrich", acid_e_d, ACID, ncol=3, label_fs=5, map_in=1.747)

# ============================ S12: DSM-NDSM difference, minus 300 K =================
# The 300 K class-difference is the reference: each panel is
# [DSM-NDSM](T) - [DSM-NDSM](300 K). The 300 K panel is all-zero (grey) so it is
# dropped, exactly like S11 -> S12 shows the temperature change of the contrast.
print("S12 (DSM-NDSM difference minus 300 K, 2 temps):")
def dom_diff(T):
    """DSM-minus-NDSM domain ln-enrichment difference map at temperature T."""
    return load_dom(T, "FIGURES/DOMAIN_CONTACT_MAPS/DIFF", "Domain_Contact_Map_FULL_FULL_DIFF_CONTACT_ENRICHMENT_LOG_INTER.csv")
def spec_diff(T):
    """DSM-minus-NDSM species (chain) ln-enrichment difference map at temperature T."""
    return load_sq(T, RES, "Residue_DIFF_CONTACT_CHAIN_ENRICHMENT_LOG_INTER_HeatMap.csv", SPEC_ORD)
def acid_diff(T, what):   # DSM - NDSM
    """DSM-minus-NDSM acid contact map of type `what` (e.g. ENRICHMENT_LOG) at T."""
    d = load_sq(T, ACI, f"Acid_DSM_CONTACT_{what}_INTER_HeatMap.csv", ACID_ORD)
    n = load_sq(T, ACI, f"Acid_NDSM_CONTACT_{what}_INTER_HeatMap.csv", ACID_ORD)
    return None if d is None or n is None else d - n

def _minus_ref(fn, T, ref):
    """Evaluate map-builder `fn` at T and at the reference temperature and return
    their difference (None if either is missing)."""
    a, b = fn(T), fn(ref)
    return None if a is None or b is None else a - b

# S12 maps are 1.747 in square (same axis size as the S11 maps), and each panel's
# two temperatures (285/315) are SPLIT into standalone maps for hand placement.
# Both temps share one scale; the colour bar is drawn only on the 315 K map.
# Acid is enrichment ONLY (the membership/fraction panel is dropped).
S12_MAP = 1.747

dom_ref12 = dom_diff(REF_T_S12); spec_ref12 = spec_diff(REF_T_S12)
dom_d = [None if (m := dom_diff(T)) is None or dom_ref12 is None else m - dom_ref12 for T in S12_TEMPS]
spec_d = [None if (m := spec_diff(T)) is None or spec_ref12 is None else m - spec_ref12 for T in S12_TEMPS]
acid_d = [_minus_ref(lambda t: acid_diff(t, "ENRICHMENT_LOG"), T, REF_T_S12) for T in S12_TEMPS]
render_pair_split("S12_A_domain", dom_d, DOM_LAB, domain=True, label_fs=8, map_in=S12_MAP, temps=S12_TEMPS)
render_pair_split("S12_B_species", spec_d, SPEC_LAB, invert_y=True, label_fs=8, map_in=S12_MAP, temps=S12_TEMPS)
render_pair_split("S12_C_acid_enrich", acid_d, ACID, label_fs=5, map_in=S12_MAP, temps=S12_TEMPS)
