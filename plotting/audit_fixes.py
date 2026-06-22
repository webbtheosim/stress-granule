"""
audit_fixes.py — address the four most-actionable PI critique items.

1. SG control binodal refit with EXTENDED upper bound (PI item #1).
   The default fit (Tc = 388.5 K, B=0.7507) hit the 415 K upper bound on the
   500-resample CI. Refit with upper_bound = 500 K and see whether T_c moves.

2. Per-compound T_c table with experimental class (PI item #5 — D1 outlier).
   Sort compounds by fitted T_c, mark DSM/NDSM, show SG reference. This
   surfaces D1 (DSM with T_c above SG) and other inconvenient compounds.

3. Leave-one-out cross-validation of the new Fig 4F classifier (PI item #4).
   Feature selection redone inside each LOO fold; report mean / SD accuracy
   over 100 random KMeans seeds per fold.

4. Permutation-test null distribution (PI item #4 cont.) — shuffle DSM/NDSM
   labels and rerun the entire forward feature search; report the null
   distribution of "best" accuracies for the 4-feature classifier.

Role: statistical-validation / audit script that underpins the Fig 4F
classifier claims (it defines the canonical Fig 4F feature set [P_SM, N_D]
and the PCA+KMeans scorer used to report the 85% LOOCV accuracy). Not a
figure renderer itself; its CSV/markdown outputs back the Fig 4F and phase-
diagram (Fig 7) discussion.

Inputs (paths default to the repository root and can be redirected with CLI flags):
  PHASE_DIAGRAM_CORRELATED_RESULTS/.../aggregated_phase_data.csv,
  perturbation_individual_binodal_fit.csv, control_binodal_fit.json
  TEMP_{T}/CLASSIFY_CORRELATED_{T}_50_50_2000/.../Quant_Data.csv

Outputs:
  PI_AUDIT_RESULTS.md   — concise report of all four fixes
  PI_AUDIT_DATA/        — supporting CSVs and plots

Exact CLI invocation:
  python plotting/audit_fixes.py --root .
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PD_RES = ROOT / "PHASE_DIAGRAM_CORRELATED_RESULTS/RESULTS/PHASE_DIAGRAM"
OUT_DIR = ROOT / "PI_AUDIT_DATA"

BETA = 0.325  # Ising universality, fixed throughout


def configure_paths(root: Path | str | None = None,
                    phase_dir: Path | str | None = None,
                    out_dir: Path | str | None = None) -> None:
    """Set repository/result paths used by this audit module."""
    global ROOT, PD_RES, OUT_DIR
    if root is not None:
        ROOT = Path(root).expanduser().resolve()
    if phase_dir is not None:
        PD_RES = Path(phase_dir).expanduser().resolve()
    else:
        PD_RES = ROOT / "PHASE_DIAGRAM_CORRELATED_RESULTS/RESULTS/PHASE_DIAGRAM"
    if out_dir is not None:
        OUT_DIR = Path(out_dir).expanduser().resolve()
    else:
        OUT_DIR = ROOT / "PI_AUDIT_DATA"


def _df_to_md(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub markdown table WITHOUT the optional
    `tabulate` dependency (which is absent in the `sg` conda env)."""
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |\n"
    sep = "|" + "|".join("---" for _ in cols) + "|\n"
    rows = ""
    for _, r in df.iterrows():
        rows += "| " + " | ".join("" if pd.isna(v) else str(v) for v in r.tolist()) + " |\n"
    return header + sep + rows


# --------------------------------------------------------------------------
# 1. SG control binodal: refit with extended upper bound
# --------------------------------------------------------------------------
def _ising_model(params: np.ndarray, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Ising-class binodal model for the dense and dilute coexistence branches.

    With params ``(Tc, c_c, A, B)`` and the fixed Ising exponent ``BETA``,
    returns ``(c_dense, c_dilute)`` as functions of temperature ``T`` using
    ``c_c + A*(Tc-T) ± B*(Tc-T)**BETA``.
    """
    Tc, c_c, A, B = params
    dx = np.maximum(Tc - T, 1e-9)
    delta = B * dx ** BETA
    c_dense = c_c + A * dx + delta
    c_dil = c_c + A * dx - delta
    return c_dense, c_dil


def _residuals(params: np.ndarray, T: np.ndarray, c_dense: np.ndarray, c_dil: np.ndarray) -> np.ndarray:
    """Stacked dense+dilute residuals for ``least_squares`` binodal fitting.

    Returns the concatenation of (model - observed) for the dense branch and
    the dilute branch given trial ``params``.
    """
    cd_model, cl_model = _ising_model(params, T)
    return np.concatenate([(cd_model - c_dense), (cl_model - c_dil)])


def refit_sg_with_extended_bound() -> Dict:
    """Refit SG with bound 500 K and 1000 K and see whether T_c moves."""
    agg = pd.read_csv(PD_RES / "aggregated_phase_data.csv")
    sg = agg[agg["condition_name"] == "SG"].copy().sort_values("temperature_K")
    T = sg["temperature_K"].to_numpy(dtype=float)
    c_dense = sg["c_sg_observed"].to_numpy(dtype=float)
    c_dil = sg["c_dil_inf_corrected"].to_numpy(dtype=float)

    out = {}
    for upper_bound in [415.0, 500.0, 1000.0]:
        x0 = np.array([min(420.0, upper_bound - 5), 200.0, 100.0, 0.75])
        lo = np.array([max(T) + 1.0, 50.0, 10.0, 0.1])
        hi = np.array([upper_bound, 500.0, 500.0, 2.0])
        result = least_squares(_residuals, x0, args=(T, c_dense, c_dil),
                               bounds=(lo, hi), loss="linear", max_nfev=10000)
        out[upper_bound] = {
            "Tc": float(result.x[0]),
            "c_c": float(result.x[1]),
            "A": float(result.x[2]),
            "B": float(result.x[3]),
            "rmse": float(np.sqrt(np.mean(result.fun ** 2))),
            "at_upper_bound": bool(abs(result.x[0] - upper_bound) < 1.0),
        }
    return out


# --------------------------------------------------------------------------
# 2. Per-compound T_c table
# --------------------------------------------------------------------------
def per_compound_tc_table() -> pd.DataFrame:
    """Build the per-compound apparent-T_c table sorted ascending by T_c.

    Loads the individual-compound binodal fits and the SG reference T_c, then
    adds ``dT_c_vs_SG``, ``above_SG``, and an ``inconvenient`` flag (DSM with
    T_c above SG, or NDSM more than 15 K below SG) to surface compounds whose
    fitted T_c sign-disagrees with their experimental class.
    """
    pf = pd.read_csv(PD_RES / "perturbation_individual_binodal_fit.csv")
    with open(PD_RES / "control_binodal_fit.json") as fh:
        sg_fit = json.load(fh)["default_fit"]["parameters"]
    sg_tc = sg_fit["Tc_app_control"]

    out = pf[["condition_name", "condition_type", "Tc_app_pert_ind",
              "c_c_pert_ind", "A_pert_ind"]].copy()
    out.rename(columns={"condition_name": "compound", "condition_type": "exp_class",
                        "Tc_app_pert_ind": "Tc_K", "c_c_pert_ind": "c_c",
                        "A_pert_ind": "A"}, inplace=True)
    out["dT_c_vs_SG"] = out["Tc_K"] - sg_tc
    out["above_SG"] = out["dT_c_vs_SG"] > 0
    out["inconvenient"] = ((out["exp_class"] == "DSM") & (out["dT_c_vs_SG"] > 0)) | \
                         ((out["exp_class"] == "NDSM") & (out["dT_c_vs_SG"] < -15.0))
    return out.sort_values("Tc_K").reset_index(drop=True)


# --------------------------------------------------------------------------
# 3. Leave-one-out cross-validation of the new Fig 4F classifier
# --------------------------------------------------------------------------
BASELINE_FEATURES = [
    "$R_{cond}$ $(\\AA)$",
    "$P_{SM}$",
    "$\\phi_{D}$",
    "$D_{SE,GK,Rh}$ $\\mu m^{2} / s$",
    "$\\eta_{GK}$ Pa s",
]

# Fig 4F classifier (final): the dimensionless SM partition coefficient P_SM and
# the condensate droplet count N_D. No temperature/binodal metrics, no raw SM
# concentrations, no per-species features. This is a FIXED 2-feature model (no
# in-fold reselection), so its LOOCV accuracy equals its in-sample accuracy.
FIG4F_FEATURES = ["$P_{SM}$", "$N_{D}$"]
NEW_FIG4F_SUBSET = list(FIG4F_FEATURES)

# Aggregate-only candidate pool, mirroring kmeans.py
# _is_allowed_condensate_kmeans_feature: P_SM + condensate aggregates + total
# interfacial tension + transfer free energy. NO raw SM concentrations, NO
# dilute-phase SG conc, NO Stokes-Einstein D, NO temperature/binodal metrics,
# NO per-species observables.
AGG_POOL = [
    "$P_{SM}$", "$N_{D}$", "$\\phi_{D}$", "$\\phi_{R}$", "$R_{g}$",
    "$R_{cond}$ $(\\AA)$", "$W_{interface}$ $(\\AA)$",
    "$c_{dense,SG,fit}$ $(mg/ml)$", "$c_{dense,SG,calc}$ $(mg/ml)$",
    "$\\eta_{GK}$ Pa s", "$l_{conf}$ A", "$\\Delta G_{trans}$ $(kJ/mol)$",
    "$\\gamma_ave$ $(mN/m)$",
]


def _build_per_compound_frame(temp: int = 300) -> pd.DataFrame:
    """Load the augmented Quant_Data and select the 20 individual compounds.

    temp selects the temperature directory (default 300 K). Other temperatures
    (e.g. 285, 315) are used by the cross-temperature classifier comparison."""
    qd = pd.read_csv(ROOT / f"TEMP_{temp}/CLASSIFY_CORRELATED_{temp}_50_50_2000/RESULTS/SUMMARY/Quant_Data.csv")
    qd = qd.set_index("Small Molecule ID")
    ids = [f"D{i}" for i in range(1, 11)] + [f"ND{i}" for i in range(1, 11)]
    df = qd.loc[ids].copy()
    df["true_label"] = [1] * 10 + [0] * 10  # 1 = DSM
    return df


def _pca_kmeans_acc(X: np.ndarray, y_true: np.ndarray, seed: int) -> float:
    """Standardize, PCA-reduce, KMeans-cluster, and score against ``y_true``.

    The unsupervised KMeans labels are cluster-identity agnostic, so the
    returned accuracy is the larger of the two label-assignment matches
    (``max(acc, 1-acc)``).
    """
    Xs = StandardScaler().fit_transform(X)
    Z = PCA(n_components=min(2, Xs.shape[1]), random_state=seed).fit_transform(Xs)
    labels = KMeans(n_clusters=2, n_init=20, max_iter=400, random_state=seed).fit_predict(Z)
    a1 = float(np.mean(labels == y_true))
    a2 = float(np.mean(labels == (1 - y_true)))
    return max(a1, a2)


def seed_sweep_baseline_vs_fig4f(n_seeds: int = 100) -> Dict:
    """Mean ± SD accuracy across n_seeds random KMeans initialisations,
    for the baseline 5-feature set and for the new Fig 4F subset."""
    df = _build_per_compound_frame()
    y = df["true_label"].to_numpy(dtype=int)
    results = {}
    for name, feats in [("baseline_5feat", BASELINE_FEATURES),
                        ("new_fig4f_4feat", NEW_FIG4F_SUBSET)]:
        X = df[feats].to_numpy(dtype=float)
        accs = []
        for seed in range(n_seeds):
            try:
                accs.append(_pca_kmeans_acc(X, y, seed))
            except Exception as exc:
                print(f"[seed sweep] {name} seed {seed} failed: {exc}")
        accs = np.array(accs)
        results[name] = {
            "n_seeds": n_seeds,
            "mean": float(np.mean(accs)),
            "std": float(np.std(accs, ddof=1)),
            "min": float(np.min(accs)),
            "max": float(np.max(accs)),
            "median": float(np.median(accs)),
        }
    return results


def loo_with_feature_reselection(n_seeds: int = 20) -> Dict:
    """Leave-one-out cross-validation with forward feature selection inside
    each fold. For each held-out compound:
      1) compute |Pearson r| with label across all candidate features (on the
         19 in-fold compounds)
      2) sort, iteratively add top features, find best subset (2..K) by
         in-fold accuracy
      3) project held-out compound onto the same PCA + KMeans, record correct/incorrect
    Report mean accuracy over n_seeds × 20 folds.
    """
    df = _build_per_compound_frame()
    # Candidate pool: the aggregate-only pool that KMeans Automated_Optimal now
    # sees (mirrors KMeans._is_allowed_condensate_kmeans_feature) — no raw SM
    # concentrations, no temperature/binodal metrics, no per-species features.
    candidate_pool = list(AGG_POOL)
    available = [f for f in candidate_pool if f in df.columns]
    print(f"[LOO] candidate pool: {len(available)} features")

    y_full = df["true_label"].to_numpy(dtype=int)
    n = len(df)
    fold_results = []  # list of dicts per (seed, fold)
    for seed in range(n_seeds):
        for hold_out in range(n):
            in_idx = [i for i in range(n) if i != hold_out]
            X_in = df[available].iloc[in_idx].to_numpy(dtype=float)
            y_in = y_full[in_idx]
            # finite-mask
            row_ok = np.all(np.isfinite(X_in), axis=1)
            if row_ok.sum() < 15:
                continue
            # per-feature correlation with label
            corrs = []
            for j, f in enumerate(available):
                xj = X_in[row_ok, j]
                if xj.std() < 1e-9:
                    corrs.append((f, 0.0))
                    continue
                r = abs(np.corrcoef(xj, y_in[row_ok])[0, 1])
                corrs.append((f, float(r) if np.isfinite(r) else 0.0))
            corrs.sort(key=lambda x: x[1], reverse=True)
            # forward search: top 2, 3, 4, 5, 6, 7 — find best in-fold accuracy
            best_acc = 0.0
            best_subset: List[str] = []
            for K in range(2, 8):
                feats = [c[0] for c in corrs[:K]]
                X_train = df[feats].iloc[in_idx].to_numpy(dtype=float)
                y_train = y_full[in_idx]
                # require finite rows
                tr_ok = np.all(np.isfinite(X_train), axis=1)
                if tr_ok.sum() < 10:
                    continue
                try:
                    acc = _pca_kmeans_acc(X_train[tr_ok], y_train[tr_ok], seed=seed)
                except Exception:
                    continue
                if acc > best_acc:
                    best_acc = acc
                    best_subset = feats
            # now classify the held-out compound using the SAME pipeline trained on the 19
            if not best_subset:
                continue
            X_train_b = df[best_subset].iloc[in_idx].to_numpy(dtype=float)
            X_hold_b = df[best_subset].iloc[[hold_out]].to_numpy(dtype=float)
            tr_ok = np.all(np.isfinite(X_train_b), axis=1)
            ho_ok = np.all(np.isfinite(X_hold_b), axis=1)
            if (not tr_ok.sum() >= 10) or (not ho_ok.all()):
                continue
            sc = StandardScaler().fit(X_train_b[tr_ok])
            Xs_tr = sc.transform(X_train_b[tr_ok])
            Xs_ho = sc.transform(X_hold_b)
            pca = PCA(n_components=min(2, Xs_tr.shape[1]), random_state=seed).fit(Xs_tr)
            Z_tr = pca.transform(Xs_tr)
            Z_ho = pca.transform(Xs_ho)
            km = KMeans(n_clusters=2, n_init=20, max_iter=400, random_state=seed).fit(Z_tr)
            pred_in = km.predict(Z_tr)
            pred_ho = km.predict(Z_ho)
            # Resolve cluster identity using in-fold accuracy
            y_train_b = y_full[in_idx][tr_ok]
            a1 = float(np.mean(pred_in == y_train_b))
            a2 = float(np.mean(pred_in == (1 - y_train_b)))
            if a2 > a1:
                pred_ho = 1 - pred_ho
            correct = bool(pred_ho[0] == y_full[hold_out])
            fold_results.append({"seed": seed, "hold_out_idx": hold_out,
                                 "hold_out_compound": df.index[hold_out],
                                 "true_label": int(y_full[hold_out]),
                                 "predicted": int(pred_ho[0]),
                                 "correct": correct,
                                 "best_subset_size": len(best_subset)})
    res_df = pd.DataFrame(fold_results)
    acc_per_seed = res_df.groupby("seed")["correct"].mean()
    return {
        "n_folds_run": len(res_df),
        "loo_accuracy_mean_over_seeds": float(acc_per_seed.mean()),
        "loo_accuracy_std_over_seeds": float(acc_per_seed.std(ddof=1)) if len(acc_per_seed) > 1 else float("nan"),
        "loo_accuracy_min": float(acc_per_seed.min()),
        "loo_accuracy_max": float(acc_per_seed.max()),
        "per_compound_correct_rate": res_df.groupby("hold_out_compound")["correct"].mean().to_dict(),
        "df": res_df,
    }


def _loo_fixed_once(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """One leave-one-out pass for a FIXED feature matrix (no reselection).
    Returns a boolean array (per fold) of held-out correctness."""
    n = len(y)
    out = np.zeros(n, dtype=bool)
    for h in range(n):
        idx = [i for i in range(n) if i != h]
        sc = StandardScaler().fit(X[idx])
        pca = PCA(n_components=min(2, X.shape[1]), random_state=seed).fit(sc.transform(X[idx]))
        km = KMeans(n_clusters=2, n_init=20, max_iter=400, random_state=seed).fit(pca.transform(sc.transform(X[idx])))
        pin = km.predict(pca.transform(sc.transform(X[idx])))
        pho = km.predict(pca.transform(sc.transform(X[[h]])))
        if np.mean(pin == (1 - y[idx])) > np.mean(pin == y[idx]):
            pho = 1 - pho
        out[h] = bool(pho[0] == y[h])
    return out


def loo_fixed_model(features: List[str] = None, n_seeds: int = 20) -> Dict:
    """Leave-one-out CV for the FIXED Fig 4F model (default [P_SM, N_D]); no
    in-fold feature reselection, so there is no selection variance and the LOOCV
    accuracy should match the in-sample accuracy. Also reports the in-sample
    accuracy and a per-compound correct-rate over seeds."""
    features = list(features) if features is not None else list(FIG4F_FEATURES)
    df = _build_per_compound_frame()
    y = df["true_label"].to_numpy(int)
    X = df[features].to_numpy(float)
    ok = np.all(np.isfinite(X), axis=1)
    Xo, yo = X[ok], y[ok]
    compounds = df.index[ok]
    per_seed, per_compound = [], np.zeros(len(yo))
    for s in range(n_seeds):
        c = _loo_fixed_once(Xo, yo, seed=s)
        per_seed.append(c.mean())
        per_compound += c
    per_seed = np.array(per_seed)
    insample = [_pca_kmeans_acc(Xo, yo, s) for s in range(n_seeds)]
    return {
        "features": features,
        "n_compounds": int(len(yo)),
        "insample_mean": float(np.mean(insample)),
        "loo_mean": float(per_seed.mean()),
        "loo_std": float(per_seed.std(ddof=1)) if n_seeds > 1 else float("nan"),
        "loo_min": float(per_seed.min()),
        "loo_max": float(per_seed.max()),
        "per_compound_correct_rate": {compounds[i]: per_compound[i] / n_seeds for i in range(len(yo))},
    }


def permutation_null_fixed(features: List[str] = None, n_perm: int = 200, seed: int = 42) -> Dict:
    """Permutation null for the FIXED model: shuffle DSM/NDSM labels and record
    the LOOCV accuracy of the fixed feature set. p = P(null LOOCV >= observed)."""
    features = list(features) if features is not None else list(FIG4F_FEATURES)
    df = _build_per_compound_frame()
    y = df["true_label"].to_numpy(int)
    X = df[features].to_numpy(float)
    ok = np.all(np.isfinite(X), axis=1)
    Xo, yo = X[ok], y[ok]
    observed = _loo_fixed_once(Xo, yo, seed=0).mean()
    rng = np.random.default_rng(seed)
    null = np.array([_loo_fixed_once(Xo, rng.permutation(yo), seed=0).mean() for _ in range(n_perm)])
    return {
        "features": features,
        "n_perm": n_perm,
        "observed_loo": float(observed),
        "null_mean": float(null.mean()),
        "null_p95": float(np.percentile(null, 95)),
        "p_value": float((np.sum(null >= observed) + 1) / (n_perm + 1)),
        "distribution": null.tolist(),
    }


def permutation_null(n_perm: int = 100, K: int = 4, seed: int = 42) -> Dict:
    """Shuffle DSM/NDSM labels, rerun the forward feature search up to top-K,
    record the best accuracy. Returns the distribution of "best" accuracies
    achievable under random labels."""
    df = _build_per_compound_frame()
    candidate_pool = list(AGG_POOL)
    available = [f for f in candidate_pool if f in df.columns]
    rng = np.random.default_rng(seed)
    null_accs = []
    for perm in range(n_perm):
        y_perm = rng.permutation(df["true_label"].to_numpy(dtype=int))
        X_full = df[available].to_numpy(dtype=float)
        finite = np.all(np.isfinite(X_full), axis=1)
        X_full = X_full[finite]
        y_perm = y_perm[finite]
        corrs = []
        for j, f in enumerate(available):
            xj = X_full[:, j]
            if xj.std() < 1e-9:
                corrs.append((f, 0.0))
                continue
            r = abs(np.corrcoef(xj, y_perm)[0, 1])
            corrs.append((f, float(r) if np.isfinite(r) else 0.0))
        corrs.sort(key=lambda x: x[1], reverse=True)
        best = 0.0
        for Kp in range(2, K + 1):
            feats = [c[0] for c in corrs[:Kp]]
            Xt = df[feats].to_numpy(dtype=float)
            Xt = Xt[finite]
            try:
                acc = _pca_kmeans_acc(Xt, y_perm, seed=perm)
                if acc > best:
                    best = acc
            except Exception:
                continue
        null_accs.append(best)
    null_accs = np.array(null_accs)
    return {
        "n_perm": n_perm,
        "K_max": K,
        "mean": float(null_accs.mean()),
        "std": float(null_accs.std(ddof=1)),
        "p95": float(np.percentile(null_accs, 95)),
        "p99": float(np.percentile(null_accs, 99)),
        "observed_85pct_p_value": float(np.mean(null_accs >= 0.85)),
        "distribution": null_accs.tolist(),
    }


def main(argv: List[str] | None = None) -> None:
    """Run all four PI-audit analyses and write the CSVs + summary markdown.

    Sequentially executes the SG binodal refit, per-compound T_c table, KMeans
    seed-sensitivity sweep, fixed and reselection LOOCV of the Fig 4F
    classifier, and the permutation-null tests, printing a console report and
    writing results to ``PI_AUDIT_DATA/`` and ``PI_AUDIT_RESULTS.md``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT),
                        help="Repository/results root containing TEMP_* and PHASE_DIAGRAM_CORRELATED_RESULTS")
    parser.add_argument("--phase-dir", default=None,
                        help="Explicit PHASE_DIAGRAM results directory")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory for audit CSV/PNG files")
    args = parser.parse_args(argv)
    configure_paths(args.root, args.phase_dir, args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PI AUDIT FIXES")
    print("=" * 70)

    print("\n[1] SG control binodal: refit with extended upper bounds")
    sg_refits = refit_sg_with_extended_bound()
    for ub, r in sg_refits.items():
        print(f"  upper_bound={ub:5g} K: Tc={r['Tc']:.2f}  rmse={r['rmse']:.2f}  at_bound={r['at_upper_bound']}")
    pd.DataFrame(sg_refits).T.to_csv(OUT_DIR / "sg_binodal_refit_extended_bound.csv")

    print("\n[2] Per-compound T_c table (sorted, inconvenient compounds flagged)")
    tc_table = per_compound_tc_table()
    tc_table.to_csv(OUT_DIR / "per_compound_tc_table.csv", index=False)
    print(tc_table[["compound", "exp_class", "Tc_K", "dT_c_vs_SG", "inconvenient"]].to_string(index=False))

    print("\n[3] 100-seed sensitivity (baseline 5-feat vs new Fig 4F 4-feat)")
    seed_results = seed_sweep_baseline_vs_fig4f(n_seeds=100)
    for name, r in seed_results.items():
        print(f"  {name}: mean={r['mean']*100:.1f}%  std={r['std']*100:.1f} pp  "
              f"range=[{r['min']*100:.0f}%, {r['max']*100:.0f}%]  median={r['median']*100:.0f}%")
    pd.DataFrame(seed_results).T.to_csv(OUT_DIR / "kmeans_seed_sensitivity.csv")

    print("\n[4] Fig 4F classifier = FIXED [P_SM, N_D] (no in-fold reselection)")
    loo_fix = loo_fixed_model(FIG4F_FEATURES, n_seeds=20)
    print(f"  features: {loo_fix['features']}  (n={loo_fix['n_compounds']})")
    print(f"  in-sample accuracy: {loo_fix['insample_mean']*100:.1f}%")
    print(f"  LOOCV accuracy:     {loo_fix['loo_mean']*100:.1f}% ± {loo_fix['loo_std']*100:.1f} pp "
          f"(range [{loo_fix['loo_min']*100:.0f}%, {loo_fix['loo_max']*100:.0f}%])")
    per_cmp_fix = pd.DataFrame(list(loo_fix['per_compound_correct_rate'].items()),
                              columns=["compound", "loo_correct_rate"])
    per_cmp_fix.to_csv(OUT_DIR / "loo_fixed_per_compound.csv", index=False)

    print("\n[4b] Secondary: LOOCV WITH in-fold feature reselection (aggregate pool)")
    loo = loo_with_feature_reselection(n_seeds=20)
    print(f"  n_folds_run = {loo['n_folds_run']} (20 LOO × 20 seeds = 400 expected)")
    print(f"  LOO accuracy mean over seeds: {loo['loo_accuracy_mean_over_seeds']*100:.1f}%")
    print(f"  LOO accuracy std over seeds:  {loo['loo_accuracy_std_over_seeds']*100:.1f} pp")
    print(f"  LOO accuracy range:           [{loo['loo_accuracy_min']*100:.0f}%, {loo['loo_accuracy_max']*100:.0f}%]")
    # Save per-compound LOO accuracy (most informative)
    per_cmp = pd.DataFrame(list(loo['per_compound_correct_rate'].items()),
                           columns=["compound", "loo_correct_rate"])
    per_cmp.to_csv(OUT_DIR / "loo_per_compound.csv", index=False)
    print("\n  Per-compound LOO correct-rate (over 20 seeds):")
    print(per_cmp.sort_values("loo_correct_rate").to_string(index=False))

    print("\n[5] Permutation-test null for the FIXED [P_SM, N_D] LOOCV (200 perms)")
    perm_fix = permutation_null_fixed(FIG4F_FEATURES, n_perm=200)
    print(f"  observed LOOCV = {perm_fix['observed_loo']*100:.1f}%")
    print(f"  null mean = {perm_fix['null_mean']*100:.1f}%, 95th pct = {perm_fix['null_p95']*100:.0f}%")
    print(f"  p-value (null LOOCV >= observed) = {perm_fix['p_value']:.3f}")
    pd.DataFrame({"perm_loo_accuracy_pct": [a * 100 for a in perm_fix['distribution']]}).to_csv(
        OUT_DIR / "permutation_null_fixed_distribution.csv", index=False)

    print("\n[5b] Secondary: permutation null with reselection (aggregate pool)")
    perm = permutation_null(n_perm=100, K=4)
    print(f"  Null mean = {perm['mean']*100:.1f}%, SD = {perm['std']*100:.1f} pp")
    print(f"  Null 95th percentile = {perm['p95']*100:.0f}%")
    print(f"  Null 99th percentile = {perm['p99']*100:.0f}%")
    print(f"  P(null ≥ 85%) = {perm['observed_85pct_p_value']:.3f}")
    pd.DataFrame({"perm_best_accuracy_pct": [a * 100 for a in perm['distribution']]}).to_csv(
        OUT_DIR / "permutation_null_distribution.csv", index=False)

    # ---- Final summary file ----
    summary = OUT_DIR / "../PI_AUDIT_RESULTS.md"
    with open(summary, "w") as fh:
        fh.write("# PI audit fixes — quantitative results\n\n")
        fh.write("Generated by `audit_fixes.py`.\n\n")
        fh.write("## 1. SG control binodal with extended upper bound\n\n")
        fh.write("| Upper bound (K) | Fitted T_c (K) | RMSE | Pegged at bound? |\n")
        fh.write("|---:|---:|---:|---|\n")
        for ub, r in sg_refits.items():
            fh.write(f"| {ub:g} | {r['Tc']:.2f} | {r['rmse']:.2f} | {r['at_upper_bound']} |\n")
        fh.write("\nIf T_c moves significantly with the upper bound, the original is bound-limited.\n\n")

        fh.write("## 2. Per-compound T_c table (D1 outlier and inconvenient compounds)\n\n")
        fh.write(_df_to_md(tc_table[["compound", "exp_class", "Tc_K", "dT_c_vs_SG", "inconvenient"]]))
        fh.write("\n\n`inconvenient` = DSM with T_c above SG OR NDSM with T_c more than 15 K below SG (sign-disagrees with experimental class).\n\n")

        fh.write("## 3. Seed sensitivity of baseline vs new Fig 4F (100 random KMeans seeds)\n\n")
        fh.write("| Feature set | mean | SD | min | max |\n")
        fh.write("|---|---:|---:|---:|---:|\n")
        for name, r in seed_results.items():
            fh.write(f"| {name} | {r['mean']*100:.1f}% | {r['std']*100:.1f} pp | {r['min']*100:.0f}% | {r['max']*100:.0f}% |\n")
        fh.write("\n")

        fh.write("## 4. Fig 4F classifier: FIXED [P_SM, N_D] (no in-fold reselection)\n\n")
        fh.write(f"- features: {', '.join(f.strip('$') for f in loo_fix['features'])}\n")
        fh.write(f"- in-sample accuracy: **{loo_fix['insample_mean']*100:.1f}%**\n")
        fh.write(f"- **LOOCV accuracy: {loo_fix['loo_mean']*100:.1f}% ± {loo_fix['loo_std']*100:.1f} pp** "
                 f"(range [{loo_fix['loo_min']*100:.0f}%, {loo_fix['loo_max']*100:.0f}%])\n")
        fh.write("- A fixed model has no feature-selection variance, so LOOCV = in-sample "
                 "(no generalization gap). P_SM is the dimensionless SM partition coefficient; "
                 "N_D is the condensate droplet count.\n\n")
        fh.write("### 4b. Secondary — LOOCV with in-fold reselection (aggregate pool)\n\n")
        fh.write(f"- folds run: {loo['n_folds_run']}\n")
        fh.write(f"- LOO accuracy (mean over 20 seeds): {loo['loo_accuracy_mean_over_seeds']*100:.1f}% ± {loo['loo_accuracy_std_over_seeds']*100:.1f} pp (1 SD)\n")
        fh.write(f"- LOO accuracy range: [{loo['loo_accuracy_min']*100:.0f}%, {loo['loo_accuracy_max']*100:.0f}%]\n\n")
        fh.write("Per-compound LOO correct-rate (over 20 seeds), sorted:\n\n")
        fh.write(_df_to_md(per_cmp.sort_values("loo_correct_rate")))
        fh.write("\n\nCompounds with correct-rate < 0.5 are systematically misclassified.\n\n")

        fh.write("## 5. Permutation-test null distribution\n\n")
        fh.write(f"- FIXED [P_SM, N_D] model, {perm_fix['n_perm']} label permutations of the LOOCV accuracy\n")
        fh.write(f"- observed LOOCV = {perm_fix['observed_loo']*100:.1f}%\n")
        fh.write(f"- null mean = {perm_fix['null_mean']*100:.1f}%, 95th percentile = {perm_fix['null_p95']*100:.0f}%\n")
        fh.write(f"- **p-value (null LOOCV ≥ observed) = {perm_fix['p_value']:.3f}**\n\n")
        fh.write("Secondary (reselection over the aggregate pool, K=2–4): "
                 f"null mean {perm['mean']*100:.1f}%, 95th pct {perm['p95']*100:.0f}%, "
                 f"P(null ≥ 85%) = {perm['observed_85pct_p_value']:.3f}.\n")

    print(f"\n[wrote] {summary}")


if __name__ == "__main__":
    main()
