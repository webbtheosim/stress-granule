"""Inject per-compound binodal-fit columns into Quant_Data.csv for KMeans.

Adds per-compound binodal-fit + dissolution-summary columns to Quant_Data.csv
so kmeans.py's Automated_Optimal feature selection considers them as candidates.

Backs up the original to Quant_Data.csv.PRE_BINODAL and writes the augmented
table back in place. Idempotent: if the columns already exist, they are
overwritten with the latest fit values.

New columns added:
  - "$T_{c,pert}$ $(K)$"            — per-compound apparent T_c (B-constrained)
  - "$c_{c,pert}$ $(mg/ml)$"        — per-compound critical concentration
  - "$A_{pert}$ $(mg/ml/K)$"        — per-compound diameter slope
  - "$\\bar{\\Delta c}$ $(mg/ml)$"  — mean binodal width across the 3 SMC-screen Ts
  - "$\\Delta\\Delta c_{rel}$"      — relative ΔΔc vs SG control
  - "$\\Delta\\Delta c_{abs}$ $(mg/ml)$" — absolute ΔΔc vs SG control
  - "$\\Delta c_{dil,\\inf,rel}$"   — relative dilute-side change

Row-by-row data sources:
  - SG row     ← control_binodal_fit.json default_fit
  - DSM_AVG / NDSM_AVG ← perturbation_average_binodal_fit.csv
  - D1..D10 / ND1..ND10 ← perturbation_individual_binodal_fit.csv
  - DSM / NDSM (aggregate-of-individuals): use class average (same as DSM_AVG / NDSM_AVG)

The binodal fit columns are per-compound (not per-T), so the same values are
injected into Quant_Data.csv at every simulated temperature.

Role: data-preparation step feeding the KMeans classifier (Fig 4F) — adds
binodal candidate features to Quant_Data.csv; not a figure renderer itself.

Inputs (paths default to the repository root and can be redirected with CLI flags):
  PHASE_DIAGRAM_CORRELATED_RESULTS/.../control_binodal_fit.json,
  perturbation_individual_binodal_fit.csv,
  perturbation_average_binodal_fit.csv, dissolution_summary.csv
  TEMP_{T}/CLASSIFY_CORRELATED_{T}_50_50_2000/.../Quant_Data.csv

Outputs: the augmented Quant_Data.csv (with a one-time .csv.PRE_BINODAL backup)
for T in {285, 300, 315}, or copied augmented CSVs under --out-dir.

Exact CLI invocation:
  python analysis/augment_binodal.py --root .
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PD_RES = ROOT / "PHASE_DIAGRAM_CORRELATED_RESULTS" / "RESULTS" / "PHASE_DIAGRAM"

NEW_COLS = [
    "$T_{c,pert}$ $(K)$",
    "$c_{c,pert}$ $(mg/ml)$",
    "$A_{pert}$ $(mg/ml/K)$",
    "$\\bar{\\Delta c}$ $(mg/ml)$",
    "$\\Delta\\Delta c_{rel}$",
    "$\\Delta\\Delta c_{abs}$ $(mg/ml)$",
    "$\\Delta c_{dil,\\inf,rel}$",
]


def _load_sources() -> Dict[str, Dict[str, float]]:
    """Return {compound_id: {new_col: value}} mapping."""
    # SG control fit (from JSON)
    with open(PD_RES / "control_binodal_fit.json") as fh:
        cb = json.load(fh)["default_fit"]["parameters"]
    sg_record = {
        "$T_{c,pert}$ $(K)$":         cb["Tc_app_control"],
        "$c_{c,pert}$ $(mg/ml)$":     cb["c_c_control"],
        "$A_{pert}$ $(mg/ml/K)$":     cb["A_control"],
        "$\\bar{\\Delta c}$ $(mg/ml)$": float("nan"),
        "$\\Delta\\Delta c_{rel}$":   0.0,
        "$\\Delta\\Delta c_{abs}$ $(mg/ml)$": 0.0,
        "$\\Delta c_{dil,\\inf,rel}$": 0.0,
    }

    # Per-compound fits
    pf = pd.read_csv(PD_RES / "perturbation_individual_binodal_fit.csv").set_index("condition_name")
    ds = pd.read_csv(PD_RES / "dissolution_summary.csv")
    ds_cond = ds[ds["summary_level"] == "condition"].set_index("condition_name")

    out: Dict[str, Dict[str, float]] = {"SG": sg_record}
    individual = [f"D{i}" for i in range(1, 11)] + [f"ND{i}" for i in range(1, 11)]
    for cid in individual:
        if cid not in pf.index or cid not in ds_cond.index:
            print(f"[augment] WARN missing fit data for {cid}; skipping")
            continue
        out[cid] = {
            "$T_{c,pert}$ $(K)$":         float(pf.at[cid, "Tc_app_pert_ind"]),
            "$c_{c,pert}$ $(mg/ml)$":     float(pf.at[cid, "c_c_pert_ind"]),
            "$A_{pert}$ $(mg/ml/K)$":     float(pf.at[cid, "A_pert_ind"]),
            "$\\bar{\\Delta c}$ $(mg/ml)$":      float(ds_cond.at[cid, "mean_Delta_c"]),
            "$\\Delta\\Delta c_{rel}$":         float(ds_cond.at[cid, "mean_delta_Delta_c_rel"]),
            "$\\Delta\\Delta c_{abs}$ $(mg/ml)$": float(ds_cond.at[cid, "mean_delta_Delta_c_abs"]),
            "$\\Delta c_{dil,\\inf,rel}$":       float(ds_cond.at[cid, "mean_delta_c_dil_inf_rel"]),
        }

    # Class averages (DSM_AVG, NDSM_AVG): from average binodal fit
    avg = pd.read_csv(PD_RES / "perturbation_average_binodal_fit.csv").set_index("condition_type")
    avg_ds = pd.read_csv(PD_RES / "dissolution_summary.csv")
    avg_ds = avg_ds[(avg_ds["summary_level"] == "condition") &
                    (avg_ds["condition_name"].isin(["DSM_AVG", "NDSM_AVG"]))].set_index("condition_name")
    for cls, cid in [("DSM", "DSM_AVG"), ("NDSM", "NDSM_AVG")]:
        if cls not in avg.index:
            print(f"[augment] WARN missing {cls} avg fit; skipping {cid}")
            continue
        out[cid] = {
            "$T_{c,pert}$ $(K)$":         float(avg.at[cls, "Tc_app_pert_avg"]),
            "$c_{c,pert}$ $(mg/ml)$":     float(avg.at[cls, "c_c_pert_avg"]),
            "$A_{pert}$ $(mg/ml/K)$":     float(avg.at[cls, "A_pert_avg"]),
            "$\\bar{\\Delta c}$ $(mg/ml)$":      float(avg_ds.at[cid, "mean_Delta_c"]) if cid in avg_ds.index else float("nan"),
            "$\\Delta\\Delta c_{rel}$":         float(avg_ds.at[cid, "mean_delta_Delta_c_rel"]) if cid in avg_ds.index else float("nan"),
            "$\\Delta\\Delta c_{abs}$ $(mg/ml)$": float(avg_ds.at[cid, "mean_delta_Delta_c_abs"]) if cid in avg_ds.index else float("nan"),
            "$\\Delta c_{dil,\\inf,rel}$":       float(avg_ds.at[cid, "mean_delta_c_dil_inf_rel"]) if cid in avg_ds.index else float("nan"),
        }
        # Mirror onto the "DSM" / "NDSM" aggregate rows (which are pipeline aggregates without proper fit)
        agg_id = cls
        out[agg_id] = out[cid].copy()

    return out


def configure_paths(root: Path | str | None = None,
                    phase_dir: Path | str | None = None) -> None:
    """Set repository/result paths used by this augmentation step."""
    global ROOT, PD_RES
    if root is not None:
        ROOT = Path(root).expanduser().resolve()
    if phase_dir is not None:
        PD_RES = Path(phase_dir).expanduser().resolve()
    else:
        PD_RES = ROOT / "PHASE_DIAGRAM_CORRELATED_RESULTS" / "RESULTS" / "PHASE_DIAGRAM"


def augment_quant_data(temp: int, in_place: bool = True, output_dir: Path | None = None) -> Path:
    """Inject the binodal columns (and SIG companions) into one temperature's CSV.

    Backs up Quant_Data.csv to ``.csv.PRE_BINODAL`` (once), looks up each row's
    compound id in the source mapping, fills the new columns in place, writes the
    table back, and returns its path. Idempotent on re-run.
    """
    qd_path = ROOT / f"TEMP_{temp}/CLASSIFY_CORRELATED_{temp}_50_50_2000/RESULTS/SUMMARY/Quant_Data.csv"
    if not in_place:
        if output_dir is None:
            raise ValueError("output_dir is required when in_place=False")
        output_dir.mkdir(parents=True, exist_ok=True)
        qd_target = output_dir / f"Quant_Data_{temp}K_BINODAL.csv"
    else:
        qd_target = qd_path
    backup = qd_path.with_suffix(".csv.PRE_BINODAL")
    if not qd_path.exists():
        raise FileNotFoundError(qd_path)
    if in_place and not backup.exists():
        shutil.copyfile(qd_path, backup)
        print(f"[backup] {backup}")
    elif in_place:
        print(f"[backup] already exists, leaving in place: {backup}")

    df = pd.read_csv(qd_path)
    if "Small Molecule ID" not in df.columns:
        raise ValueError(f"Missing 'Small Molecule ID' in {qd_path}")

    sources = _load_sources()
    # Initialise the new columns with NaN; also create paired SIG{col} companions
    # required by kmeans.py's _drop_incomplete_kmeans_features filter. For the
    # binodal fit and dissolution-summary metrics there is no per-compound SEM
    # (single fit per compound), so the SIG columns are placeholders (0.0).
    for col in NEW_COLS:
        df[col] = float("nan")
        df["SIG" + col] = 0.0
    # Inject per-row
    n_filled = 0
    for idx, row in df.iterrows():
        cid = str(row["Small Molecule ID"]).strip()
        if cid in sources:
            for col, val in sources[cid].items():
                df.at[idx, col] = val
            n_filled += 1
    # Rows without a matching source id keep their NaN columns: kmeans.py's
    # candidate filter only cares about non-NaN values, and the SG row is
    # intentionally left NaN for $\bar{\Delta c}$.
    print(f"[augment] filled {n_filled} rows of {qd_path.name} with binodal columns "
          f"({len(NEW_COLS)} new columns + {len(NEW_COLS)} SIG companions)")

    df.to_csv(qd_target, index=False)
    return qd_target


def main() -> None:
    """Augment Quant_Data.csv with binodal columns for each relevant temperature."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT), help="Repository/results root containing TEMP_*")
    parser.add_argument("--phase-dir", default=None, help="Explicit PHASE_DIAGRAM results directory")
    parser.add_argument("--temps", nargs="+", type=int, default=[285, 300, 315],
                        help="Temperatures to augment")
    parser.add_argument("--out-dir", default=None,
                        help="Write augmented CSV copies here instead of modifying Quant_Data.csv in place")
    args = parser.parse_args()
    configure_paths(args.root, args.phase_dir)
    in_place = args.out_dir is None
    output_dir = None if in_place else Path(args.out_dir).expanduser().resolve()
    for T in args.temps:
        out = augment_quant_data(T, in_place=in_place, output_dir=output_dir)
        print(f"[wrote] {out}")


if __name__ == "__main__":
    main()
