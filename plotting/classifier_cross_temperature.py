"""Cross-temperature SMC classifier LOOCV comparison.

For each of the three temperatures (285, 300, 315 K) we run the same forward
feature selection the SMC uses (greedy by |Pearson r| with the DSM/NDSM label
over the aggregate-only pool AGG_POOL, best in-sample PCA+KMeans accuracy,
K=2..7, parsimony tie-break) to obtain that temperature's *own* selected
feature set. We then take each temperature's selected set — plus the overall
manuscript model [P_SM, N_D] — and evaluate every one of them by LEAVE-ONE-OUT
CV at all three temperatures.

This answers: "the selected values each temperature gives, applied to all of
them, and the overall one we selected." The diagonal (a set evaluated at the
temperature it was selected on) is the in-sample-optimal case; off-diagonal
bars show how well each selection transfers across temperature.

Paper figure: SI 8 (PCA eigenvalue / loadings panels and the LOOCV transfer bars).

Inputs: PI_AUDIT_FIXES (imported as `A`) supplies the per-compound feature frame
(`A._build_per_compound_frame`), the aggregate feature pool (`A.AGG_POOL`), the
overall manuscript feature set (`A.FIG4F_FEATURES`), the PCA+KMeans accuracy scorer
(`A._pca_kmeans_acc`), and the fixed-model leave-one-out pass (`A._loo_fixed_once`),
which in turn read the per-temperature CLASSIFY_CORRELATED Quant_Data tables.

Outputs (FIGURES_SI/classifier_cross_temperature/):
  - CLASSIFIER_CROSS_TEMPERATURE_LOOCV.png         grouped LOOCV bar plot
  - CLASSIFIER_CROSS_TEMPERATURE_LOOCV.csv         full model x eval-temp matrix
  - CLASSIFIER_CROSS_TEMPERATURE_EIGENVALUES.png   PCA eigenvalue scree plot
  - CLASSIFIER_CROSS_TEMPERATURE_LOADINGS_*.png    one PC1/PC2 loadings plot per set

Run with (no CLI flags):
    python classifier_cross_temperature.py
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import audit_fixes as A

TEMPS = [285, 300, 315]
OVERALL_FEATURES = list(A.FIG4F_FEATURES)          # [P_SM, N_D]
OVERALL_LABEL = "Union"
N_SEEDS = 20
SEL_SEEDS = range(10)                              # seeds averaged for selection

OUT_DIR = Path("FIGURES_SI/classifier_cross_temperature")

# Axis size shared by EVERY plot in this folder.
AX_IN = 2.00


def _make_axes(left: float, bottom: float, right: float, top: float):
    """Figure + axes with a fixed AX_IN x AX_IN drawing area and given margins
    (all in inches)."""
    fig_w = left + AX_IN + right
    fig_h = bottom + AX_IN + top
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([left / fig_w, bottom / fig_h, AX_IN / fig_w, AX_IN / fig_h])
    return fig, ax


# Bar colours by evaluation temperature, on a cool->warm scale
# (285 -> blue, 300 -> purple, 315 -> red).
TEMP_COLOR = {285: "#2c7fb8", 300: "#7b3294", 315: "#d7301f"}

# Short display names for features (matches classifier_loo_stability style).
_NICE = {
    "$P_{SM}$": "P_SM", "$N_{D}$": "N_D", "$\\phi_{D}$": "phi_D",
    "$\\phi_{R}$": "phi_R", "$R_{g}$": "R_g", "$R_{cond}$ $(\\AA)$": "R_cond",
    "$W_{interface}$ $(\\AA)$": "W_int", "$c_{dense,SG,fit}$ $(mg/ml)$": "c_dense_fit",
    "$c_{dense,SG,calc}$ $(mg/ml)$": "c_dense_calc", "$\\eta_{GK}$ Pa s": "eta_GK",
    "$l_{conf}$ A": "l_conf", "$\\Delta G_{trans}$ $(kJ/mol)$": "dG_trans",
    "$\\gamma_{1}$ $(mN/m)$": "gamma_1", "$\\gamma_{2}$ $(mN/m)$": "gamma_2",
    "$\\gamma_ave$ $(mN/m)$": "gamma_ave",
}
nice = lambda f: _NICE.get(f, f.strip("$").replace("\\", ""))


# --------------------------------------------------------------------------
def forward_select_full(temp: int) -> Tuple[List[str], float]:
    """Full-data forward selection at one temperature. Rank AGG_POOL by |corr|
    with the label, try top-K (K=2..7), pick the subset with the best
    seed-averaged in-sample PCA+KMeans accuracy; tie-break to FEWER features."""
    df = A._build_per_compound_frame(temp)
    pool = [f for f in A.AGG_POOL if f in df.columns]
    y = df["true_label"].to_numpy(int)
    X = df[pool].to_numpy(float)
    ok = np.all(np.isfinite(X), axis=1)
    Xok, yok = X[ok], y[ok]
    # per-feature |Pearson r|
    corrs = []
    for j, f in enumerate(pool):
        xj = Xok[:, j]
        r = 0.0 if xj.std() < 1e-9 else abs(np.corrcoef(xj, yok)[0, 1])
        corrs.append((f, r if np.isfinite(r) else 0.0))
    corrs.sort(key=lambda t: t[1], reverse=True)

    best_acc, best_K, best = -1.0, 99, []
    for K in range(2, 8):
        feats = [c[0] for c in corrs[:K]]
        Xt = df[feats].to_numpy(float)
        m = np.all(np.isfinite(Xt), axis=1)
        if m.sum() < 10:
            continue
        accs = [A._pca_kmeans_acc(Xt[m], y[m], seed=s) for s in SEL_SEEDS]
        acc = float(np.mean(accs))
        # higher accuracy wins; on (near) ties prefer fewer features
        if acc > best_acc + 1e-9 or (abs(acc - best_acc) <= 1e-9 and K < best_K):
            best_acc, best_K, best = acc, K, feats
    return best, best_acc


def loo_fixed_at_temp(features: List[str], temp: int) -> Dict[str, float]:
    """Fixed-model leave-one-out CV (no reselection) at a given temperature,
    averaged over N_SEEDS KMeans initialisations."""
    df = A._build_per_compound_frame(temp)
    y = df["true_label"].to_numpy(int)
    X = df[features].to_numpy(float)
    ok = np.all(np.isfinite(X), axis=1)
    Xo, yo = X[ok], y[ok]
    per_seed = [A._loo_fixed_once(Xo, yo, seed=s).mean() for s in range(N_SEEDS)]
    insample = [A._pca_kmeans_acc(Xo, yo, s) for s in range(N_SEEDS)]
    return {
        "loo_mean": float(np.mean(per_seed)),
        "loo_std": float(np.std(per_seed, ddof=1)) if N_SEEDS > 1 else 0.0,
        "insample_mean": float(np.mean(insample)),
        "n": int(len(yo)),
    }


# --------------------------------------------------------------------------
def _eig_spectrum(features: List[str], temp: int) -> np.ndarray:
    """Full PCA eigenvalue spectrum (covariance eigenvalues) of a feature set,
    on standardised per-compound data at one temperature."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    df = A._build_per_compound_frame(temp)
    X = df[features].to_numpy(float)
    ok = np.all(np.isfinite(X), axis=1)
    Xs = StandardScaler().fit_transform(X[ok])
    pca = PCA(n_components=Xs.shape[1], random_state=0).fit(Xs)
    return pca.explained_variance_


def plot_eigenvalues(selected: Dict[str, List[str]], model_keys: List[str]) -> None:
    """Scree plot: PCA eigenvalue spectrum for each temperature's selected set
    (evaluated at its own temperature) and for the union set (at 300 K)."""
    sns.set_style("white")
    fig, ax = _make_axes(left=0.62, bottom=0.55, right=0.20, top=0.20)

    max_k = 0
    for m in model_keys:
        temp = 300 if m == OVERALL_LABEL else int(m)
        eig = _eig_spectrum(selected[m], temp)
        x = np.arange(1, len(eig) + 1)
        max_k = max(max_k, len(eig))
        if m == OVERALL_LABEL:
            ax.plot(x, eig, marker="s", ls="--", color="black", lw=1.6,
                    markersize=5, markeredgecolor="black", label="Union",
                    zorder=6, clip_on=False)
        else:
            ax.plot(x, eig, marker="o", ls="-", color=TEMP_COLOR[temp], lw=1.6,
                    markersize=5, markeredgecolor="black", markeredgewidth=0.5,
                    label=f"{m} K", zorder=5, clip_on=False)

    ax.set_xticks(range(1, 8))
    ax.set_xlim(1, 7)
    ax.set_ylim(0, 5)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.tick_params(axis="both", labelsize=8, left=True, right=True, direction="in",
                   length=4, width=2)
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=1,
              handlelength=1.4, borderaxespad=0.4)
    # keep spines BELOW the markers so edge points render on top of the axes
    for s in ax.spines.values():
        s.set_linewidth(2)
        s.set_zorder(1)
    png_path = OUT_DIR / "CLASSIFIER_CROSS_TEMPERATURE_EIGENVALUES.png"
    fig.savefig(png_path, dpi=400)
    plt.close(fig)
    print(f"[wrote] {png_path}")


PC1_COLOR = "#404040"   # dark grey
PC2_COLOR = "#9a9a9a"   # light grey


# Display overrides for the PC-loading x-axis tick labels (proper LaTeX).
_PC_LABELS = {
    "$N_{D}$": "$k$",
    "$R_{g}$": "$R_{g}^{(sg)}$",
    "$\\eta_{GK}$ Pa s": "$\\eta_{GK}^{(sg)}$",
    "$\\gamma_ave$ $(mN/m)$": "$\\gamma$",
    "$\\phi_{D}$": "$\\phi_{bp}^{(sg)}$",
}


def _latex_symbol(feat: str) -> str:
    """Display LaTeX label for a feature on the PC-loading plots."""
    if feat in _PC_LABELS:
        return _PC_LABELS[feat]
    m = re.match(r"(\$[^$]*\$)", feat)
    s = m.group(1) if m else feat
    return s.replace("\\gamma_ave", "\\gamma_{ave}")


def _loadings(features: List[str], temp: int) -> np.ndarray:
    """PCA component loadings (n_comp x n_features) on standardised data."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    df = A._build_per_compound_frame(temp)
    X = df[features].to_numpy(float)
    ok = np.all(np.isfinite(X), axis=1)
    Xs = StandardScaler().fit_transform(X[ok])
    pca = PCA(n_components=min(2, Xs.shape[1]), random_state=0).fit(Xs)
    return pca.components_


def plot_loadings(selected: Dict[str, List[str]], model_keys: List[str]) -> None:
    """One dual-axis grouped-bar plot per feature set: |loading| of each feature
    on PC1 (left axis, dark grey) and PC2 (right axis, light grey)."""
    sns.set_style("white")
    for m in model_keys:
        temp = 300 if m == OVERALL_LABEL else int(m)
        feats = selected[m]
        comp = _loadings(feats, temp)
        pc1 = np.abs(comp[0])
        pc2 = np.abs(comp[1]) if comp.shape[0] > 1 else np.zeros_like(pc1)
        n = len(feats)
        x = np.arange(n)
        bw = 0.40

        # identical AX_IN x AX_IN axes for every set (bars scale to fit)
        fig, ax1 = _make_axes(left=0.62, bottom=0.85, right=0.62, top=0.20)
        ax2 = ax1.twinx()

        ax1.bar(x - bw / 2, pc1, width=bw, color=PC1_COLOR, edgecolor="k",
                linewidth=1.0, zorder=3, label="PC 1")
        ax2.bar(x + bw / 2, pc2, width=bw, color=PC2_COLOR, edgecolor="k",
                linewidth=1.0, zorder=3, label="PC 2")

        # black axes + tick marks; only the y tick LABELS carry the PC colour
        for axx, col in ((ax1, PC1_COLOR), (ax2, PC2_COLOR)):
            axx.set_ylim(0, 1)
            axx.set_yticks(np.linspace(0, 1, 6))
            axx.tick_params(axis="y", color="black", labelcolor=col,
                            direction="in", length=4, width=2, labelsize=8)
        ax1.set_xlim(-0.6, n - 0.4)
        ax1.set_xticks(x)
        # centre feature labels on each bar group, residue-contact-map style
        ax1.set_xticklabels([_latex_symbol(f) for f in feats], rotation=45,
                            ha="center", va="top", rotation_mode="default",
                            color="black", fontsize=8)
        ax1.tick_params(axis="x", length=0)

        ax1.set_axisbelow(False)
        for s in ax1.spines.values():
            s.set_linewidth(2)
            s.set_zorder(10)
        ax2.spines["right"].set_linewidth(2)
        ax2.spines["right"].set_zorder(10)

        tag = "UNION" if m == OVERALL_LABEL else f"{m}K"
        png = OUT_DIR / f"CLASSIFIER_CROSS_TEMPERATURE_LOADINGS_{tag}.png"
        fig.savefig(png, dpi=400)
        plt.close(fig)
        print(f"[wrote] {png}")


# --------------------------------------------------------------------------
def main() -> None:
    """Run the full cross-temperature analysis: forward-select a feature set at
    each temperature (plus the union set), evaluate every set by LOOCV at all
    three temperatures, and write the LOOCV CSV/bar plot, the PCA scree plot, and
    the per-set PC1/PC2 loading plots."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. per-temperature selected sets + the overall set
    selected: Dict[str, List[str]] = {}
    print("=== Per-temperature forward selection (AGG_POOL) ===")
    for T in TEMPS:
        feats, acc = forward_select_full(T)
        selected[f"{T}"] = feats
        print(f"  T={T}: {[nice(f) for f in feats]}  (in-sample {acc*100:.0f}%)")
    # Union of every temperature's selected feature set (first-appearance order)
    union_feats: List[str] = []
    for T in TEMPS:
        for f in selected[f"{T}"]:
            if f not in union_feats:
                union_feats.append(f)
    selected[OVERALL_LABEL] = union_feats
    print(f"  Union: {[nice(f) for f in union_feats]}")

    # model order: the three temp selections, then the union
    model_keys = [f"{T}" for T in TEMPS] + [OVERALL_LABEL]
    model_titles = {f"{T}": f"Selected @ {T} K" for T in TEMPS}
    model_titles[OVERALL_LABEL] = "Union"

    # 2. evaluate every model by LOOCV at every temperature
    rows = []
    loo = {m: {} for m in model_keys}        # loo[model][eval_temp] = mean
    loo_n = {m: {} for m in model_keys}      # loo[model][eval_temp] = n folds (compounds)
    for m in model_keys:
        feats = selected[m]
        for E in TEMPS:
            r = loo_fixed_at_temp(feats, E)
            loo[m][E] = r["loo_mean"]
            loo_n[m][E] = r["n"]
            rows.append({
                "model": m,
                "model_title": model_titles[m],
                "model_features": "; ".join(nice(f) for f in feats),
                "eval_temp_K": E,
                "loo_accuracy_pct": round(r["loo_mean"] * 100, 1),
                "loo_sd_pct": round(r["loo_std"] * 100, 1),
                "insample_pct": round(r["insample_mean"] * 100, 1),
                "n_compounds": r["n"],
            })
    mat = pd.DataFrame(rows)
    csv_path = OUT_DIR / "CLASSIFIER_CROSS_TEMPERATURE_LOOCV.csv"
    mat.to_csv(csv_path, index=False)
    print(f"\n[wrote] {csv_path}")
    print("\n=== LOOCV accuracy (%): rows=model, cols=eval temperature ===")
    pivot = mat.pivot(index="model_title", columns="eval_temp_K", values="loo_accuracy_pct")
    pivot = pivot.reindex([model_titles[m] for m in model_keys])
    print(pivot.to_string())

    # 3. grouped bar plot: groups = model (selected-at temperature + union),
    #    bars within each group = evaluation temperature.
    sns.set_style("white")
    n_groups = len(model_keys)
    n_bars = len(TEMPS)
    bar_w = 0.26
    group_gap = 0.46
    x0 = np.arange(n_groups) * (n_bars * bar_w + group_gap)
    y_bot, y_top = 40.0, 118.0

    # RDP-consistent square axes (AX_IN x AX_IN); legend sits in top margin
    fig, ax = _make_axes(left=0.50, bottom=0.50, right=0.30, top=0.45)

    for k, E in enumerate(TEMPS):
        vals = [loo[m][E] * 100 for m in model_keys]
        # binomial standard error of the LOOCV proportion: sqrt(p(1-p)/n)
        errs = [100.0 * math.sqrt(max(loo[m][E] * (1 - loo[m][E]), 0.0)
                                  / max(loo_n[m][E], 1)) for m in model_keys]
        xpos = x0 + k * bar_w
        ax.bar(xpos, vals, width=bar_w, color=TEMP_COLOR[E],
               edgecolor="k", linewidth=1.0, label=f"{E} K", zorder=3)
        # usual capped error-bar style
        ax.errorbar(xpos, vals, yerr=errs, fmt="none", ecolor="black",
                    elinewidth=1, capsize=2, capthick=1, zorder=4)

    ax.axhline(50, color="black", lw=1, ls="--", zorder=0)
    ax.set_ylim(y_bot, y_top)
    ax.set_yticks([40, 60, 80, 100])
    ax.set_xlim(x0[0] - bar_w - 0.10, x0[-1] + n_bars * bar_w + 0.10)
    ax.set_xticks(x0 + (n_bars - 1) * bar_w / 2)
    # x tick labels: selected-at temperature with units (and "Union" for the last)
    xt_labels = [f"{m} K" if m != OVERALL_LABEL else "Union" for m in model_keys]
    ax.set_xticklabels(xt_labels, fontsize=8)
    ax.tick_params(axis="y", labelsize=8, left=True, right=True, direction="in",
                   length=4, width=2)
    ax.tick_params(axis="x", length=0)
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=1,
              handlelength=1.2, borderaxespad=0.4)
    # draw the axis frame ON TOP of the bars (bars sit below the axes)
    ax.set_axisbelow(False)
    for s in ax.spines.values():
        s.set_linewidth(2)
        s.set_zorder(10)
    png_path = OUT_DIR / "CLASSIFIER_CROSS_TEMPERATURE_LOOCV.png"
    fig.savefig(png_path, dpi=400)
    plt.close(fig)
    print(f"[wrote] {png_path}")

    # 4. eigenvalue-decomposition (scree) plot of the feature sets
    plot_eigenvalues(selected, model_keys)

    # 5. per-set PCA loading bar plots (PC1 left axis, PC2 right axis)
    plot_loadings(selected, model_keys)


if __name__ == "__main__":
    main()
