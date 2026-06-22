"""SM self-aggregation figures (SI Fig S1 D/E) for the ANALYSIS_SM pipeline.

DSM/NDSM mean lines are green (#40641b / #bfe49b) with filled scatter markers
(black edge) and a SEM band (std-over-compounds / sqrt(N)); a grey dashed line
marks the all-monomeric baseline where it fits (cluster number -> 8324). No axis
titles; 5 ticks/axis; y limits floor->ceiling (2-sig-fig rounding).

Axes sizes: the time series (cluster number / fraction) use the RDP-plot size
(2.2 x 2.2 in); the RDP/RDF profiles use the violin size (1.6 x 1.6 in).

To reduce clutter / smooth: cluster series take every 5th frame; RDP/RDF profiles
are averaged over every 5 distance bins (which also tames the small-r shell noise).

dsm/ and ndsm/ additionally hold "_individual" plots with every molecule, coloured
by a rocket partition (DSM = purple end, NDSM = orange end; matches SI_1 panel B).

Run under Python 3.10+ with pandas/seaborn/matplotlib (conda base).
"""
import os
import re
import sys
import glob
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import sm_plotting as SP   # reuse the correct cluster loader

# This file lives at <repo>/plotting/sm/, so the repo root is three levels up.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_DSM = os.path.join(_REPO_ROOT, "PYTHON_ANALYSIS/SM_300/DSM/ANALYSIS_SM")
DEFAULT_NDSM = os.path.join(_REPO_ROOT, "PYTHON_ANALYSIS/SM_300/NDSM/ANALYSIS_SM")

COLOR = {"DSM": "#40641b", "NDSM": "#bfe49b"}
N_MOLECULES = 8324
DT_NS, WINDOW_NS, TMIN, TMAX, STRIDE = 1.0, 20.0, 0.0, 200.0, 5.0   # every 5th frame
COARSEN = 5                                                        # avg every 5 RDP/RDF bins

# Axes sizes (inches): cluster/PHI match the RDP plots; RDP/RDF match the violins.
AX_CLUSTER = 2.20
AX_RDP = 1.60


def _rocket_partition(cls, n):
    """DSM = purple/dark end of rocket; NDSM = orange/light end (matches SI_1 B)."""
    full = sns.color_palette("rocket", 20)
    return list(full[:n]) if cls == "DSM" else list(full[20 - n:])


def _round_2sf(x, up):
    """Round ``x`` to two significant figures, ceiling if ``up`` else floor."""
    if not np.isfinite(x) or x == 0:
        return 0.0
    sign = 1 if x > 0 else -1
    ax = abs(x)
    exp = math.floor(math.log10(ax))
    mant = ax / 10 ** exp
    r = (math.ceil(mant * 10) if up else math.floor(mant * 10)) / 10
    if r >= 10:
        r /= 10
        exp += 1
    return sign * r * 10 ** exp


def _rc():
    """Apply the shared seaborn ticks/white theme and font/linewidth rc params."""
    sns.set_theme(style="ticks")
    sns.set_style("white")
    plt.rc("font", size=10)
    plt.rc("axes", titlesize=10); plt.rc("axes", labelsize=10)
    plt.rc("xtick", labelsize=8); plt.rc("ytick", labelsize=8)
    plt.rc("legend", fontsize=8); plt.rc("axes", linewidth=2)


def _ax(ax_in, left=0.60, bottom=0.46, right=0.24, top=0.26):
    """Build a figure with a single ``ax_in`` (inch) square axes plus margins.

    Returns ``(fig, ax)`` sized so the axes is exactly ``ax_in`` x ``ax_in`` in.
    """
    fw, fh = ax_in + left + right, ax_in + bottom + top
    fig = plt.figure(figsize=(fw, fh))
    ax = fig.add_axes([left / fw, bottom / fh, ax_in / fw, ax_in / fh])
    return fig, ax


def _finish(ax, xlim, ylo, yhi, time_x):
    """Set x/y limits, 5 ticks/axis, sci-notation when extreme, and inward ticks.

    ``time_x`` pins the x-ticks to 0..200 ns; otherwise they span ``xlim``.
    Strips axis labels (panels are titleless).
    """
    ax.set_xlim(*xlim)
    ax.set_ylim(ylo, yhi)
    if time_x:
        ax.set_xticks(np.linspace(0, 200, 5))
    else:
        ax.set_xticks(np.linspace(xlim[0], xlim[1], 5))
    ax.set_yticks(np.linspace(ylo, yhi, 5))
    if abs(yhi) < 1e-2 or abs(yhi) >= 1e4:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True,
                   direction="in", length=4, width=2)
    for s in ax.spines.values():
        s.set_zorder(3)
    ax.set_xlabel(""); ax.set_ylabel("")


# ----------------------------------------------------------------------------
# RDP / RDF loaders -> one mean profile per compound (band over compounds)
# ----------------------------------------------------------------------------
def _parse_id(stem, prefix):
    """Parse a profile filename stem into ``(category, sm_name)``.

    Strips ``prefix`` and any trailing ``_t<start>`` window suffix; category is
    "DSM"/"NDSM" for ``dsm_``/``ndsm_`` tags, else the uppercased leading token.
    """
    tag = stem.replace(prefix, "")
    m = re.match(r"(.*)_t\d+$", tag)
    full = m.group(1) if m else tag
    cat = "DSM" if full.startswith("dsm_") else ("NDSM" if full.startswith("ndsm_") else full.split("_")[0].upper())
    sm = full.split("_", 1)[1] if "_" in full else full
    return cat, sm


def _load_profile(category_dir, prefix, value_col):
    """Load all ``prefix*`` profile CSVs and average windows within each compound.

    Returns a long DataFrame with columns distance/value/category/sm, one mean
    profile per (category, sm) over its windows. ``value_col`` selects the
    y-column (falls back to the second column if absent).
    """
    acc = {}
    for f in sorted(glob.glob(os.path.join(str(category_dir), prefix + "*.csv"))):
        cat, sm = _parse_id(Path(f).stem, prefix)
        t = pd.read_csv(f)
        dist = t.iloc[:, 0].to_numpy(float)
        col = value_col if value_col in t.columns else t.columns[1]
        val = pd.to_numeric(t[col], errors="coerce").to_numpy(float)
        acc.setdefault((cat, sm), []).append(pd.Series(val, index=dist))
    rows = []
    for (cat, sm), series in acc.items():
        mean = pd.concat(series, axis=1).mean(axis=1)
        for d, v in mean.items():
            rows.append({"distance": float(d), "value": float(v), "category": cat, "sm": sm})
    return pd.DataFrame(rows)


def _coarsen_profile(df, n=COARSEN):
    """Average every n distance bins per compound — fewer points and smooths the
    small-r shell-volume noise that otherwise makes the RDP look broken."""
    out = []
    for (cat, sm), g in df.groupby(["category", "sm"], sort=False):
        g = g.sort_values("distance").reset_index(drop=True)
        g = g.assign(_bin=g.index // n)
        a = g.groupby("_bin").agg(distance=("distance", "mean"), value=("value", "mean")).reset_index(drop=True)
        a["category"] = cat
        a["sm"] = sm
        out.append(a)
    return pd.concat(out, ignore_index=True) if out else df


# ----------------------------------------------------------------------------
# Class mean + SEM band + scatter (SI Fig S1 D/E style)
# ----------------------------------------------------------------------------
def _band(df, x, y, outpath, classes, sg_value=None, time_x=False, ax_in=AX_CLUSTER):
    """Render the class-mean +/- SEM band with scatter markers (SI Fig S1 D/E style).

    For each class in ``classes``, plots the across-compound mean of ``y`` vs
    ``x`` with a SEM band and edge-marked scatter points; an optional grey dashed
    ``sg_value`` baseline is added. Returns the output path, or None if there is
    no finite data to plot.
    """
    _rc()
    fig, ax = _ax(ax_in)
    order = [c for c in ("DSM", "NDSM") if c in classes]
    lo, hi = np.inf, -np.inf
    for cls in order:
        sub = df[df["category"] == cls]
        if sub.empty:
            continue
        g = sub.groupby(x)[y].agg(["mean", "sem"]).reset_index().sort_values(x)
        m, e = g["mean"].to_numpy(), np.nan_to_num(g["sem"].to_numpy())
        ax.fill_between(g[x], m - e, m + e, color=COLOR[cls], alpha=0.25, linewidth=0, zorder=1)
        ax.plot(g[x], m, color=COLOR[cls], linewidth=2, label=cls, zorder=2)
        ax.scatter(g[x], m, facecolor=COLOR[cls], edgecolor="k", linewidth=0.6, s=20, zorder=5)
        lo = min(lo, np.nanmin(m - e)); hi = max(hi, np.nanmax(m + e))
    if sg_value is not None:
        ax.axhline(sg_value, color="grey", linestyle="--", linewidth=1.5, zorder=0)
        lo = min(lo, sg_value); hi = max(hi, sg_value)
    if not np.isfinite(lo):
        plt.close(fig)
        return None
    ylo, yhi = _round_2sf(lo, up=False), _round_2sf(hi, up=True)
    if ylo == yhi:
        yhi = ylo + abs(ylo) * 0.1 + 1e-9
    xlim = (TMIN, TMAX) if time_x else (0, float(df["distance"].max()))
    _finish(ax, xlim, ylo, yhi, time_x)
    if len(order) > 1:
        leg = ax.legend(loc="upper right", frameon=False, handlelength=1.2, borderpad=0.2)
        leg.get_frame().set_alpha(0)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=400)
    plt.close(fig)
    return outpath


# ----------------------------------------------------------------------------
# All individual molecules, rocket partition (DSM purple / NDSM orange)
# ----------------------------------------------------------------------------
def _individual(df, x, y, outpath, cls, sg_value=None, time_x=False, ax_in=AX_CLUSTER):
    """Render one line per compound for a single class (rocket-partition colors).

    Plots every compound's mean ``y`` vs ``x`` for class ``cls`` using the rocket
    partition (DSM purple end / NDSM orange end), with an optional grey dashed
    ``sg_value`` baseline. Returns the output path, or None if the class is empty.
    """
    _rc()
    fig, ax = _ax(ax_in)
    sub = df[df["category"] == cls]
    if sub.empty:
        plt.close(fig)
        return None
    sms = sorted(sub["sm"].unique())
    cols = _rocket_partition(cls, len(sms))
    lo, hi = np.inf, -np.inf
    for i, sm in enumerate(sms):
        s = sub[sub["sm"] == sm].groupby(x)[y].mean().reset_index().sort_values(x)
        ax.plot(s[x], s[y], color=cols[i], linewidth=1.2, zorder=2)
        lo = min(lo, np.nanmin(s[y])); hi = max(hi, np.nanmax(s[y]))
    if sg_value is not None:
        ax.axhline(sg_value, color="grey", linestyle="--", linewidth=1.5, zorder=0)
        lo = min(lo, sg_value); hi = max(hi, sg_value)
    ylo, yhi = _round_2sf(lo, up=False), _round_2sf(hi, up=True)
    if ylo == yhi:
        yhi = ylo + abs(ylo) * 0.1 + 1e-9
    xlim = (TMIN, TMAX) if time_x else (0, float(df["distance"].max()))
    _finish(ax, xlim, ylo, yhi, time_x)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=400)
    plt.close(fig)
    return outpath


# ----------------------------------------------------------------------------
def run(outdir="FIGURES/isolation", dsm_dir=DEFAULT_DSM, ndsm_dir=DEFAULT_NDSM):
    """Render all SM-isolation panels (cluster, RDP number/mass, RDF) into ``outdir``.

    Loads and coarsens the DSM/NDSM cluster series and density/RDF profiles, then
    writes the aggregated/ class-mean band plots plus per-compound dsm/ and ndsm/
    "_individual" plots. Returns the list of PNG paths actually written.
    """
    cl = pd.concat([SP._load_cluster(Path(dsm_dir), DT_NS, WINDOW_NS, TMIN, TMAX),
                    SP._load_cluster(Path(ndsm_dir), DT_NS, WINDOW_NS, TMIN, TMAX)],
                   ignore_index=True)
    cl = SP._downsample_time(cl, "time_ns", STRIDE)                      # every 5th frame
    rdpN = _coarsen_profile(pd.concat([_load_profile(dsm_dir, "Density_Profile_SM_Number_", "SM number density (1/A^3)"),
                                       _load_profile(ndsm_dir, "Density_Profile_SM_Number_", "SM number density (1/A^3)")], ignore_index=True))
    rdpM = _coarsen_profile(pd.concat([_load_profile(dsm_dir, "Density_Profile_SM_Mass_", "SM mass density (mg/mL)"),
                                       _load_profile(ndsm_dir, "Density_Profile_SM_Mass_", "SM mass density (mg/mL)")], ignore_index=True))
    rdf = _coarsen_profile(pd.concat([_load_profile(dsm_dir, "RDF_SM_COM_", "g(r)"),
                                      _load_profile(ndsm_dir, "RDF_SM_COM_", "g(r)")], ignore_index=True))

    # (name, dataframe, x, y, sg baseline, time?, axes size)
    SERIES = [
        ("cluster_fraction_vs_time", cl, "time_ns", "LargestClusterFraction", None, True, AX_CLUSTER),
        ("cluster_number_vs_time", cl, "time_ns", "NumClusters", N_MOLECULES, True, AX_CLUSTER),
        ("rdp_number", rdpN, "distance", "value", None, False, AX_RDP),
        ("rdp_mass", rdpM, "distance", "value", None, False, AX_RDP),
        ("rdf", rdf, "distance", "value", None, False, AX_RDP),
    ]
    made = []
    for sub, classes in [("aggregated", ("DSM", "NDSM")), ("dsm", ("DSM",)), ("ndsm", ("NDSM",))]:
        d = os.path.join(outdir, sub)
        for name, dfo, x, y, sg, tx, ax_in in SERIES:
            made.append(_band(dfo, x, y, os.path.join(d, name + ".png"), classes,
                              sg_value=sg, time_x=tx, ax_in=ax_in))
            if sub in ("dsm", "ndsm"):
                made.append(_individual(dfo, x, y, os.path.join(d, name + "_individual.png"),
                                        classes[0], sg_value=sg, time_x=tx, ax_in=ax_in))
    return [m for m in made if m]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="FIGURES/isolation")
    a = ap.parse_args()
    for m in run(a.outdir):
        print("  ", m)
