#!/usr/bin/env python3
"""
Projected SG biopolymer phase-diagram analysis.

This module reads existing Quant_Data summary outputs from this repository,
infers temperature from the TEMP_<K> path token, applies a Gibbs-Thomson-style
curvature correction to the dilute SG biopolymer branch, fits an apparent
control binodal, and quantifies DSM/NDSM dissolution shifts.

Important scientific notes implemented here:
- dilute concentration is corrected early using a Gibbs-Thomson-style curvature correction
- dense branch is intentionally left uncorrected as an approximation
- the resulting phase diagram is a projected SG biopolymer mixture binodal
- the fitted Tc is an apparent critical temperature
- DSM/NDSM apparent critical temperatures are qualitative if fitted from only 3 temperatures
- Delta_c is the primary perturbation/dissolution metric
- RCC-derived Meff_SG is exact for the model-level SG species-weighted average chain molar mass
- the approximation enters later through Veff_SG = Meff_SG / c_sg, which is only an apparent molar-volume surrogate

Role: produces the projected binodal and apparent-Tc phase diagram (paper
Fig 7) and supporting temperature-trend SI panels (SI 7, 9). Driven by the
``phase_diagram`` class.

Key inputs: per-temperature ``Quant_Data.csv`` files matched by --summary-glob
(or a single --input-csv); optional cluster file for Meff_SG. Key outputs:
binodal-fit CSV/JSON tables, dissolution summaries, and figures written under
--output-dir (RESULTS/PHASE_DIAGRAM and FIGURES/PHASE_DIAGRAM).

Example usage:
    python phase_diagram.py \
      --summary-glob 'TEMP_*/CLASSIFY_*/RESULTS/SUMMARY/Quant_Data.csv' \
      --output-dir PHASE_DIAGRAM_RESULTS \
      --beta-critical 0.325 \
      --fit-perturbed-tc \
      --robust-fit-sensitivity
"""

from __future__ import annotations

import argparse
import glob as globlib
import io
import json
import math
import re
import warnings
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import OptimizeWarning, least_squares
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from kmeans import summary_plots

R_GAS = 8.314462618
N_AVOGADRO = 6.02214076e23
DEFAULT_CONTROL_ID = "SG"
DEFAULT_REL_FLOOR = 0.05
DEFAULT_MONO_SIGMA_FACTOR = 1.0
DEFAULT_MEFF_TOL_REL = 1e-3
DEFAULT_BOOTSTRAP_SAMPLES = 500
DEFAULT_RANDOM_SEED = 12345
DEFAULT_DILUTE_NEGATIVE_FLOOR = -0.05
DEFAULT_INTERFACE_THRESHOLD = 0.75
DEFAULT_CURVATURE_MC_SAMPLES = 4000
DEFAULT_MIN_PERTURBED_TC_TEMPS = 5
PHASE_COLOR_SG = "#808080"
PHASE_COLOR_DSM = "#40641b"
PHASE_COLOR_NDSM = "#bfe49b"
PHASE_MARKER_SIZE_SG = 36          # 25% smaller than legacy 48
PHASE_MARKER_SIZE_AVG = 36         # 25% smaller than legacy 48
PHASE_MARKER_SIZE_INDIVIDUAL = 8
# Match biopolymer_analysis.py RDP plots: 2.20x2.20 in axes, figsize 3.20x3.20.
RDP_STYLE_FIGSIZE = (3.20, 3.20)
RDP_STYLE_AX_RECT = [0.50 / 3.20, 0.50 / 3.20, 2.20 / 3.20, 2.20 / 3.20]

SUMMARY_COLUMN_MAP = {
    "small_molecule_id": "Small Molecule ID",
    "small_molecule_name": "Small Molecule Name",
    "compound_name": "Compound Name",
    "compound_class": "Compound Class",
    "c_sg": "$c_{dense,SG,fit}$ $(mg/ml)$",
    "c_sg_std": "SIG$c_{dense,SG,fit}$ $(mg/ml)$",
    "c_dil": "$c_{dilute,SG,fit}$ $(mg/ml)$",
    "c_dil_std": "SIG$c_{dilute,SG,fit}$ $(mg/ml)$",
    "c_sg_calc": "$c_{dense,SG,calc}$ $(mg/ml)$",
    "c_dil_calc": "$c_{dilute,SG,calc}$ $(mg/ml)$",
    "R_sg_A": "$R_{cond}$ $(\\AA)$",
    "R_sg_std_A": "SIG$R_{cond}$ $(\\AA)$",
    "W_sg_A": "$W_{interface}$ $(\\AA)$",
    "W_sg_std_A": "SIG$W_{interface}$ $(\\AA)$",
    "gamma_mN_m": "$\\gamma_ave$ $(mN/m)$",
    "gamma_std_mN_m": "SIG$\\gamma_ave$ $(mN/m)$",
    "gamma1_mN_m": "$\\gamma_{1}$ $(mN/m)$",
    "gamma2_mN_m": "$\\gamma_{2}$ $(mN/m)$",
}


@dataclass
class phase_diagram_config:
    """All tunable parameters and I/O paths for one phase-diagram analysis run.

    Populated from CLI arguments by ``build_arg_parser``/``main``; passed to
    ``phase_diagram``. Fields cover input selection, the Meff_SG source,
    the Gibbs-Thomson/curvature correction, binodal-fit options (robustness,
    bootstrap, beta exponent), and the KMeans temperature-transfer sub-analysis.
    """
    summary_glob: Optional[List[str]]
    input_csv: Optional[Path]
    output_dir: Path
    meff_sg: str
    cluster_file_for_meff: Optional[Path]
    meff_sg_scale_sensitivity: List[float] = field(default_factory=list)
    beta_critical: float = 0.325
    fit_perturbed_tc: bool = False
    include_raw_dilute: bool = False
    interface_threshold: float = DEFAULT_INTERFACE_THRESHOLD
    curvature_threshold: float = 0.1
    include_flagged_fit_sensitivity: bool = False
    control_reference_sensitivity: bool = False
    delta_c_ctrl_epsilon: float = 1e-3
    robust_fit_sensitivity: bool = False
    conc_source: str = "calc"
    temperature_from: str = "path"
    temperature_column: Optional[str] = None
    control_id: str = DEFAULT_CONTROL_ID
    include_category_aggregate_rows: bool = False
    rel_floor: float = DEFAULT_REL_FLOOR
    monotonicity_sigma_factor: float = DEFAULT_MONO_SIGMA_FACTOR
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES
    random_seed: int = DEFAULT_RANDOM_SEED
    meff_tolerance_rel: float = DEFAULT_MEFF_TOL_REL
    c_dil_negative_floor: float = DEFAULT_DILUTE_NEGATIVE_FLOOR
    robust_loss: str = "huber"
    robust_f_scale: float = 1.0
    plot_title_prefix: str = ""
    run_kmeans_temperature_transfer: bool = True
    kmeans_random_seed: int = 42
    curvature_mc_samples: int = DEFAULT_CURVATURE_MC_SAMPLES
    min_perturbed_tc_temperatures: int = DEFAULT_MIN_PERTURBED_TC_TEMPS


@dataclass
class fit_result:
    """Outcome of a single binodal least-squares fit.

    Records success/fallback status, the fitted parameters and RMSE, the loss
    and residual-scaling mode used, the eligible vs. excluded temperatures, and
    any warnings raised during fitting.
    """
    success: bool
    fit_kind: str
    loss: str
    used_fallback: bool
    parameters: Dict[str, float]
    rmse: float
    n_points: int
    n_residuals: int
    message: str
    eligible_temperatures_K: List[float]
    excluded_temperatures_K: List[float]
    include_flagged_points: bool
    residual_scale_mode: str
    warnings: List[str] = field(default_factory=list)


class phase_diagram:
    """Driver for the projected-binodal / apparent-Tc phase-diagram analysis.

    Holds the run configuration and output directories and orchestrates the full
    pipeline via ``run()``: load summary inputs, resolve Meff_SG, apply the
    dilute-branch curvature correction, aggregate replicates, fit the control and
    perturbed binodals, quantify DSM/NDSM dissolution shifts, optionally run the
    KMeans temperature-transfer sub-analysis, and write all tables and figures.
    """

    def __init__(self, config: phase_diagram_config) -> None:
        """Store the config and create the RESULTS/FIGURES output directories."""
        self.config = config
        self.result_dir = config.output_dir / "RESULTS" / "PHASE_DIAGRAM"
        self.figure_dir = config.output_dir / "FIGURES" / "PHASE_DIAGRAM"
        self.kmeans_result_dir = self.result_dir / "KMEANS_TEMPERATURE_TRANSFER"
        self.kmeans_figure_dir = self.figure_dir / "KMEANS_TEMPERATURE_TRANSFER"
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self.kmeans_result_dir.mkdir(parents=True, exist_ok=True)
        self.kmeans_figure_dir.mkdir(parents=True, exist_ok=True)
        self.manifest: Dict[str, Any] = {
            "config": self._jsonable(asdict(config)),
            "summary_files": [],
            "temperature_mapping": [],
            "meff_inference": {},
            "warnings": [],
            "kmeans_temperature_transfer": {},
        }

    def run(self) -> None:
        """Execute the full phase-diagram pipeline and write all tables, figures, and the manifest."""
        raw_df = self.load_inputs()
        meff_info = self.resolve_meff_sg(raw_df)
        processed_df = self.preprocess_phase_rows(raw_df, meff_info["meff_g_mol"])
        aggregated_df = self.aggregate_replicates(processed_df)
        aggregated_df = self.apply_control_shape_diagnostics(aggregated_df)
        control_bundle = self.fit_control_bundle(aggregated_df)
        perturb_df, perturb_summary = self.compute_perturbation_metrics(aggregated_df, control_bundle)
        dissolution_summary = self.build_dissolution_summary(perturb_df, perturb_summary)
        control_curve_df = self.build_control_curve_table(control_bundle, aggregated_df)
        sensitivity_df = self.run_meff_scale_sensitivity(aggregated_df)

        self.write_outputs(
            processed_df=processed_df,
            aggregated_df=perturb_df,
            control_bundle=control_bundle,
            control_curve_df=control_curve_df,
            perturb_summary=perturb_summary,
            dissolution_summary=dissolution_summary,
            sensitivity_df=sensitivity_df,
        )
        self.make_plots(perturb_df, control_bundle, control_curve_df, perturb_summary)
        if self.config.run_kmeans_temperature_transfer:
            self.run_temperature_transfer_kmeans()
        self.write_manifest()

    # ------------------------------------------------------------------
    # Input discovery and loading
    # ------------------------------------------------------------------
    def load_inputs(self) -> pd.DataFrame:
        """Load all input rows from a single --input-csv or from every file matched by --summary-glob."""
        if self.config.input_csv:
            return self.load_input_csv(self.config.input_csv)
        summary_files = self.discover_summary_files()
        if not summary_files:
            raise FileNotFoundError("No input files found from --summary-glob")
        rows: List[Dict[str, Any]] = []
        for summary_path in summary_files:
            rows.extend(self.load_summary_table(summary_path))
        if not rows:
            raise RuntimeError("No usable rows were found in any Quant_Data.csv input")
        return pd.DataFrame(rows)

    def discover_summary_files(self) -> List[Path]:
        """Resolve --summary-glob patterns to a sorted list of Quant_Data.csv paths (records them in the manifest)."""
        if not self.config.summary_glob:
            raise ValueError("Either --summary-glob or --input-csv is required")
        patterns = self.config.summary_glob or []
        if isinstance(patterns, str):
            patterns = [patterns]
        files = sorted({Path(p) for pattern in patterns for p in globlib.glob(pattern)})
        if not files:
            raise FileNotFoundError(f"No files matched summary glob: {self.config.summary_glob}")
        self.manifest["summary_files"] = [str(p.resolve()) for p in files]
        return files

    def load_input_csv(self, path: Path) -> pd.DataFrame:
        """Load one CSV, dispatching to the Quant_Data-style or tidy-CSV parser based on its columns."""
        df = pd.read_csv(path)
        if SUMMARY_COLUMN_MAP["small_molecule_id"] in df.columns:
            temp, token, source = self.infer_temperature_from_path(path)
            rows = self.load_summary_table(path, override_df=df, override_temp=(temp, token, source))
            return pd.DataFrame(rows)
        return self.load_tidy_csv(df, path)

    def load_tidy_csv(self, df: pd.DataFrame, path: Path) -> pd.DataFrame:
        """Parse a tidy (long-format) CSV into the internal phase-row schema, deriving concentrations and row semantics."""
        required = {"condition_name", "condition_type"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Input CSV is neither Quant_Data-like nor tidy; missing columns: {sorted(missing)}")
        out = df.copy()
        if self.config.temperature_column and self.config.temperature_column in out.columns:
            out["temperature_K"] = pd.to_numeric(out[self.config.temperature_column], errors="coerce")
        if "temperature_K" not in out.columns:
            if self.config.temperature_from == "column":
                raise ValueError("temperature_K or --temperature-column is required for tidy mode")
            temp, token, source = self.infer_temperature_from_path(path)
            out["temperature_K"] = temp
            out["temperature_source"] = source
            out["temperature_token"] = token
            out["temperature_path"] = str(path)
        else:
            out["temperature_K"] = pd.to_numeric(out["temperature_K"], errors="coerce")
            if "temperature_source" not in out.columns:
                out["temperature_source"] = "column"
            if "temperature_token" not in out.columns:
                out["temperature_token"] = self.config.temperature_column or "temperature_K"
            if "temperature_path" not in out.columns:
                out["temperature_path"] = str(path)
        out["source_file"] = str(path.resolve())
        out["source_mode"] = "tidy_csv"
        if "summary_root" not in out.columns:
            out["summary_root"] = str(path.parent.resolve())
        if "uncertainty_source_mode" not in out.columns:
            out["uncertainty_source_mode"] = self.infer_uncertainty_source_mode(path)
        if "uncertainty_interpretation" not in out.columns:
            out["uncertainty_interpretation"] = "approximate_analysis_sigma"
        if "replicate_id" not in out.columns:
            out["replicate_id"] = np.arange(len(out), dtype=int)
        if "condition_subtype" not in out.columns:
            out["condition_subtype"] = out["condition_type"].astype(str)
        if "small_molecule_id" not in out.columns:
            out["small_molecule_id"] = out["condition_name"]
        if "small_molecule_name" not in out.columns:
            out["small_molecule_name"] = out["condition_name"]
        if "compound_name" not in out.columns:
            out["compound_name"] = out["condition_name"]
        if "compound_class" not in out.columns:
            out["compound_class"] = out["condition_type"]

        def choose_column(frame: pd.DataFrame, names: Sequence[str]) -> pd.Series:
            for name in names:
                if name in frame.columns:
                    return pd.to_numeric(frame[name], errors="coerce")
            return pd.Series(np.nan, index=frame.index, dtype=float)

        beta_fit = choose_column(out, ["beta_fit", "beta", "beta_sg_fit"])
        alpha_fit = choose_column(out, ["alpha_fit", "alpha", "alpha_sg_fit"])
        c_sg = choose_column(out, ["c_sg", "c_dense", "c_dense_sg", "c_dense_sg_fit"])
        c_dil = choose_column(out, ["c_dil", "c_dilute", "c_dilute_sg", "c_dilute_sg_fit"])
        c_sg = c_sg.where(np.isfinite(c_sg), beta_fit + alpha_fit)
        c_dil = c_dil.where(np.isfinite(c_dil), beta_fit - alpha_fit)

        out["beta_fit"] = beta_fit
        out["alpha_fit"] = alpha_fit
        out["c_sg"] = c_sg
        out["c_dil"] = c_dil
        out["c_sg_std"] = choose_column(out, ["c_sg_std", "c_dense_std", "c_dense_sg_std"])
        out["c_dil_std"] = choose_column(out, ["c_dil_std", "c_dilute_std", "c_dilute_sg_std"])
        out["R_sg_A"] = choose_column(out, ["R_sg_A", "R_sg", "R_cond_A"])
        out["R_sg_std_A"] = choose_column(out, ["R_sg_std_A", "R_sg_std", "R_cond_std_A"])
        out["W_sg_A"] = choose_column(out, ["W_sg_A", "W_sg", "W_interface_A"])
        out["W_sg_std_A"] = choose_column(out, ["W_sg_std_A", "W_sg_std", "W_interface_std_A"])
        out["gamma_mN_m"] = choose_column(out, ["gamma_mN_m", "gamma", "gamma_ave", "gamma_ave_mN_m"])
        out["gamma_std_mN_m"] = choose_column(out, ["gamma_std_mN_m", "gamma_std", "gamma_ave_std_mN_m"])
        out["Meff_SG_g_mol"] = choose_column(out, ["Meff_SG_g_mol", "Meff_SG", "meff_sg"])

        row_semantic_type = []
        is_control = []
        is_average_row = []
        is_individual_sm = []
        average_row_source = []
        average_row_membership = []
        for _, row in out.iterrows():
            condition_name = self.safe_str(row.get("condition_name", ""))
            cond_type = self.safe_str(row.get("condition_type", ""))
            semantic = self.safe_str(row.get("row_semantic_type", ""))
            if not semantic:
                if condition_name == self.config.control_id or cond_type == "control":
                    semantic = "individual_compound"
                elif condition_name in {"DSM_AVG", "NDSM_AVG"}:
                    semantic = "arithmetic_class_average"
                elif condition_name in {"DSM", "NDSM"}:
                    semantic = "category_aggregate"
                else:
                    semantic = "individual_compound"
            row_semantic_type.append(semantic)
            ctrl = condition_name == self.config.control_id or cond_type == "control"
            is_control.append(ctrl)
            is_average_row.append(semantic == "arithmetic_class_average")
            is_individual_sm.append((semantic == "individual_compound") and (cond_type in {"DSM", "NDSM"}))
            if semantic == "arithmetic_class_average":
                average_row_source.append(self.safe_str(row.get("average_row_source", "input_csv_arithmetic_average")))
                average_row_membership.append(self.safe_str(row.get("average_row_membership", cond_type)))
            elif semantic == "category_aggregate":
                average_row_source.append(self.safe_str(row.get("average_row_source", "input_csv_category_aggregate")))
                average_row_membership.append(self.safe_str(row.get("average_row_membership", cond_type)))
            else:
                average_row_source.append(self.safe_str(row.get("average_row_source", "")))
                average_row_membership.append(self.safe_str(row.get("average_row_membership", "")))
        out["row_semantic_type"] = row_semantic_type
        out["is_control"] = is_control
        out["is_average_row"] = is_average_row
        out["is_individual_sm"] = is_individual_sm
        out["average_row_source"] = average_row_source
        out["average_row_membership"] = average_row_membership
        return out

    def load_summary_table(
        self,
        summary_path: Path,
        override_df: Optional[pd.DataFrame] = None,
        override_temp: Optional[Tuple[float, str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Map one Quant_Data.csv into a list of internal phase-row dicts, attaching the inferred temperature."""
        df = override_df if override_df is not None else pd.read_csv(summary_path)
        required_columns = [
            SUMMARY_COLUMN_MAP["small_molecule_id"],
            SUMMARY_COLUMN_MAP["c_sg"],
            SUMMARY_COLUMN_MAP["c_dil"],
            SUMMARY_COLUMN_MAP["R_sg_A"],
            SUMMARY_COLUMN_MAP["W_sg_A"],
        ]
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required Quant_Data columns in {summary_path}: {missing}")
        if override_temp is None:
            temperature_K, temperature_token, temperature_source = self.infer_temperature_from_path(summary_path)
        else:
            temperature_K, temperature_token, temperature_source = override_temp
        summary_root = summary_path.parents[2]
        rows: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            mapped = self.map_quant_data_row(
                row=row,
                summary_path=summary_path,
                summary_root=summary_root,
                temperature_K=temperature_K,
                temperature_token=temperature_token,
                temperature_source=temperature_source,
            )
            if mapped is not None:
                rows.append(mapped)
        self.manifest["temperature_mapping"].append(
            {
                "summary_file": str(summary_path.resolve()),
                "temperature_K": temperature_K,
                "temperature_source": temperature_source,
                "temperature_token": temperature_token,
            }
        )
        return rows

    def infer_temperature_from_path(self, path: Path) -> Tuple[float, str, str]:
        """Infer the temperature (K) from the ``TEMP_<n>`` token in a path; raises if absent."""
        if self.config.temperature_from == "column":
            if self.config.temperature_column is None:
                raise ValueError("--temperature-column is required when --temperature-from column")
            raise ValueError("Column-based temperature inference is only supported for tidy CSV mode")
        text = str(path)
        match = re.search(r"TEMP_(\d+)", text)
        if not match:
            raise ValueError(f"Could not infer temperature from path: {path}")
        token = match.group(0)
        return float(match.group(1)), token, "path"

    def infer_uncertainty_source_mode(self, path: Path) -> str:
        """Classify a path's uncertainty provenance (e.g. correlated vs. plain) from its name."""
        text = str(path).lower()
        if "blocked" in text or "correlated" in text or "corr" in text:
            return "correlated"
        return "legacy"

    def map_quant_data_row(
        self,
        row: pd.Series,
        summary_path: Path,
        summary_root: Path,
        temperature_K: float,
        temperature_token: str,
        temperature_source: str,
    ) -> Optional[Dict[str, Any]]:
        """Convert one Quant_Data.csv row into the internal phase-row dict, or None if it should be skipped."""
        sm_id = str(row.get(SUMMARY_COLUMN_MAP["small_molecule_id"], "")).strip()
        if not sm_id:
            return None
        if not self.keep_row(sm_id):
            return None

        condition_type, condition_subtype, row_semantic_type = self.classify_row(sm_id)
        out: Dict[str, Any] = {
            "condition_name": sm_id,
            "condition_type": condition_type,
            "condition_subtype": condition_subtype,
            "temperature_K": temperature_K,
            "temperature_source": temperature_source,
            "temperature_token": temperature_token,
            "temperature_path": str(summary_path.resolve()),
            "source_file": str(summary_path.resolve()),
            "source_mode": "summary_quant_data",
            "summary_root": str(summary_root.resolve()),
            "uncertainty_source_mode": self.infer_uncertainty_source_mode(summary_path),
            "uncertainty_interpretation": "approximate_analysis_sigma",
            "small_molecule_id": sm_id,
            "small_molecule_name": self.safe_str(row.get(SUMMARY_COLUMN_MAP["small_molecule_name"], "")),
            "compound_name": self.safe_str(row.get(SUMMARY_COLUMN_MAP["compound_name"], "")),
            "compound_class": self.safe_str(row.get(SUMMARY_COLUMN_MAP["compound_class"], "")),
            "is_control": sm_id == self.config.control_id,
            "is_average_row": row_semantic_type == "arithmetic_class_average",
            "average_row_source": "",
            "average_row_membership": "",
            "row_semantic_type": row_semantic_type,
            "is_individual_sm": row_semantic_type == "individual_compound" and condition_type in {"DSM", "NDSM"},
        }

        if row_semantic_type == "arithmetic_class_average":
            out["average_row_source"] = "arithmetic_mean_over_individual_sm_rows"
            out["average_row_membership"] = condition_type
        elif row_semantic_type == "category_aggregate":
            out["average_row_source"] = "category_aggregate_row_from_pipeline"
            out["average_row_membership"] = condition_type

        string_fields = {
            "small_molecule_id",
            "small_molecule_name",
            "compound_name",
            "compound_class",
        }
        for internal, column in SUMMARY_COLUMN_MAP.items():
            if internal in string_fields:
                out[internal] = self.safe_str(row.get(column, ""))
            else:
                out[internal] = self.safe_float(row.get(column, math.nan))

        if self.config.conc_source == "calc":
            if math.isfinite(out.get("c_sg_calc", math.nan)):
                out["c_sg"] = out["c_sg_calc"]
            if math.isfinite(out.get("c_dil_calc", math.nan)):
                out["c_dil"] = out["c_dil_calc"]

        return out

    def keep_row(self, sm_id: str) -> bool:
        """Return whether a small-molecule id should be retained for the analysis."""
        if sm_id == self.config.control_id:
            return True
        if sm_id in {"DSM_AVG", "NDSM_AVG"}:
            return True
        if sm_id in {"DSM", "NDSM"}:
            return self.config.include_category_aggregate_rows
        if re.fullmatch(r"D\d+", sm_id) or re.fullmatch(r"ND\d+", sm_id):
            return True
        if sm_id.startswith("dsm_") or sm_id.startswith("ndsm_"):
            return True
        return False

    def classify_row(self, sm_id: str) -> Tuple[str, str, str]:
        """Return the (semantic type, condition type, membership) classification for a small-molecule id."""
        if sm_id == self.config.control_id:
            return "control", "control", "individual_compound"
        if sm_id == "DSM_AVG":
            return "DSM", "average", "arithmetic_class_average"
        if sm_id == "NDSM_AVG":
            return "NDSM", "average", "arithmetic_class_average"
        if sm_id == "DSM":
            return "DSM", "aggregate", "category_aggregate"
        if sm_id == "NDSM":
            return "NDSM", "aggregate", "category_aggregate"
        if re.fullmatch(r"D\d+", sm_id):
            return "DSM", "individual", "individual_compound"
        if re.fullmatch(r"ND\d+", sm_id):
            return "NDSM", "individual", "individual_compound"
        if sm_id.startswith("dsm_"):
            return "DSM", "individual", "individual_compound"
        if sm_id.startswith("ndsm_"):
            return "NDSM", "individual", "individual_compound"
        return "unknown", "unknown", "individual_compound"

    # ------------------------------------------------------------------
    # Meff inference
    # ------------------------------------------------------------------
    def resolve_meff_sg(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Determine the effective SG chain molar mass Meff_SG (explicit value or auto-inferred from a cluster file)."""
        if self.config.meff_sg.lower() != "auto":
            value = float(self.config.meff_sg)
            info = {
                "meff_g_mol": value,
                "source": "cli_override",
                "inferred_values_g_mol": [],
                "warning": "",
            }
            self.manifest["meff_inference"] = self._jsonable(info)
            return info

        if "Meff_SG_g_mol" in df.columns:
            rowwise = pd.to_numeric(df["Meff_SG_g_mol"], errors="coerce")
            rowwise = rowwise[np.isfinite(rowwise) & (rowwise > 0)]
            if not rowwise.empty:
                values = rowwise.to_numpy(dtype=float)
                meff = float(np.median(values))
                rel_spread = float((np.max(values) - np.min(values)) / abs(meff)) if meff != 0 else math.inf
                warning = ""
                if rel_spread > self.config.meff_tolerance_rel:
                    warning = (
                        f"Row-wise Meff_SG values vary across input rows (relative spread {rel_spread:.3e}); "
                        f"using the median fallback value {meff:.6g} g/mol for missing rows"
                    )
                    self.manifest["warnings"].append(warning)
                info = {
                    "meff_g_mol": meff,
                    "source": "rowwise_input_median_fallback",
                    "inferred_values_g_mol": [{"row_index": int(i), "meff_g_mol": float(v)} for i, v in rowwise.items()],
                    "relative_spread": rel_spread,
                    "warning": warning,
                }
                self.manifest["meff_inference"] = self._jsonable(info)
                return info

        cluster_files: List[Path] = []
        if self.config.cluster_file_for_meff:
            cluster_files.append(self.config.cluster_file_for_meff)
        else:
            roots = sorted({Path(str(x)) for x in df["summary_root"].dropna().unique().tolist()})
            for root in roots:
                cluster_files.extend(self.cluster_candidates_from_summary_root(root))

        inferred: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for cluster_path in cluster_files:
            if not cluster_path.exists():
                continue
            key = str(cluster_path.resolve())
            if key in seen:
                continue
            seen.add(key)
            value = self.infer_meff_sg_from_cluster(cluster_path)
            if value is not None:
                inferred.append({"cluster_file": key, "meff_g_mol": value})

        if not inferred:
            raise FileNotFoundError(
                "Automatic Meff_SG inference failed: no usable Cluster_SG_sg_X.csv was found. "
                "Provide --meff-sg or --cluster-file-for-meff."
            )

        values = np.array([item["meff_g_mol"] for item in inferred], dtype=float)
        meff = float(np.median(values))
        rel_spread = float((np.max(values) - np.min(values)) / abs(meff)) if meff != 0 else math.inf
        warning = ""
        if rel_spread > self.config.meff_tolerance_rel:
            warning = (
                f"Inferred Meff_SG varies across available SG cluster files (relative spread {rel_spread:.3e}); "
                f"using the median value {meff:.6g} g/mol"
            )
            self.manifest["warnings"].append(warning)
        info = {
            "meff_g_mol": meff,
            "source": "cluster_inference_median",
            "inferred_values_g_mol": inferred,
            "relative_spread": rel_spread,
            "warning": warning,
            "scientific_note": (
                "RCC-derived Meff_SG is exact for the model-level SG species-weighted average chain molar mass; "
                "the approximation enters later through Veff_SG = Meff_SG / c_sg."
            ),
        }
        self.manifest["meff_inference"] = self._jsonable(info)
        return info

    def cluster_candidates_from_summary_root(self, summary_root: Path) -> List[Path]:
        """List candidate Cluster_*.csv files under a summary root for Meff_SG inference."""
        candidates = [
            summary_root / "ANALYSIS_SG_AVE" / "Cluster_SG_sg_X.csv",
            summary_root / "ANALYSIS_SG_AVE" / "Cluster_SG_SG.csv",
        ]
        if "RESULTS" in summary_root.parts:
            idx = summary_root.parts.index("RESULTS")
            alt_root = Path(*summary_root.parts[:idx])
            candidates.extend(
                [
                    alt_root / "ANALYSIS_SG_AVE" / "Cluster_SG_sg_X.csv",
                    alt_root / "ANALYSIS_SG_AVE" / "Cluster_SG_SG.csv",
                ]
            )
        return candidates

    def infer_meff_sg_from_cluster(self, cluster_path: Path) -> Optional[float]:
        """Compute the species-weighted average chain molar mass (g/mol) from a cluster composition file."""
        try:
            df = pd.read_csv(cluster_path)
        except Exception as exc:
            self.manifest["warnings"].append(f"Failed to read cluster file for Meff_SG: {cluster_path} ({exc})")
            return None
        if "Total Mass (mg)" not in df.columns or "Total Chain Number" not in df.columns:
            self.manifest["warnings"].append(
                f"Cluster file missing Total Mass / Total Chain Number columns: {cluster_path}"
            )
            return None
        total_mass_mg = self.safe_float(df["Total Mass (mg)"].mean())
        total_chain_number = self.safe_float(df["Total Chain Number"].mean())
        if not math.isfinite(total_mass_mg) or not math.isfinite(total_chain_number) or total_chain_number <= 0:
            self.manifest["warnings"].append(f"Invalid Total Mass / Total Chain Number in {cluster_path}")
            return None
        m_per_chain_g = total_mass_mg * 1e-3 / total_chain_number
        return float(m_per_chain_g * N_AVOGADRO)

    # ------------------------------------------------------------------
    # Preprocessing and uncertainty propagation
    # ------------------------------------------------------------------
    def preprocess_phase_rows(self, df: pd.DataFrame, meff_sg_g_mol: float) -> pd.DataFrame:
        """Apply the Gibbs-Thomson dilute-branch curvature correction and derive per-row concentrations/flags."""
        out = df.copy()
        if "Meff_SG_g_mol" in out.columns:
            out["Meff_SG_g_mol"] = pd.to_numeric(out["Meff_SG_g_mol"], errors="coerce")
            out["Meff_SG_g_mol"] = out["Meff_SG_g_mol"].where(
                np.isfinite(out["Meff_SG_g_mol"]) & (out["Meff_SG_g_mol"] > 0),
                meff_sg_g_mol,
            )
        else:
            out["Meff_SG_g_mol"] = meff_sg_g_mol

        # Preserve observed semantic names for output clarity.
        out["c_sg_observed"] = pd.to_numeric(out["c_sg"], errors="coerce")
        out["c_dil_observed"] = pd.to_numeric(out["c_dil"], errors="coerce")

        warnings_col: List[str] = []
        clamped = []
        hard_invalid = []
        for _, row in out.iterrows():
            row_warnings: List[str] = []
            c_dil = self.safe_float(row["c_dil_observed"])
            invalid = False
            if math.isfinite(c_dil) and self.config.c_dil_negative_floor <= c_dil < 0.0:
                c_dil = 0.0
                row_warnings.append("clamped_small_negative_c_dil_to_zero")
            elif math.isfinite(c_dil) and c_dil < self.config.c_dil_negative_floor:
                invalid = True
                row_warnings.append("hard_invalid_negative_c_dil")
            clamped.append(c_dil)

            if not math.isfinite(self.safe_float(row["c_sg_observed"])) or self.safe_float(row["c_sg_observed"]) <= 0:
                invalid = True
                row_warnings.append("hard_invalid_c_sg")
            if not math.isfinite(self.safe_float(row["R_sg_A"])) or self.safe_float(row["R_sg_A"]) <= 0:
                invalid = True
                row_warnings.append("hard_invalid_R_sg")
            if not math.isfinite(self.safe_float(row["W_sg_A"])) or self.safe_float(row["W_sg_A"]) < 0:
                invalid = True
                row_warnings.append("hard_invalid_W_sg")
            if not math.isfinite(self.safe_float(row["gamma_mN_m"])) or self.safe_float(row["gamma_mN_m"]) < 0:
                invalid = True
                row_warnings.append("hard_invalid_gamma")
            if not math.isfinite(self.safe_float(row["temperature_K"])) or self.safe_float(row["temperature_K"]) <= 0:
                invalid = True
                row_warnings.append("hard_invalid_temperature")

            hard_invalid.append(invalid)
            warnings_col.append(";".join(row_warnings))

        out["c_dil_observed"] = clamped
        out["flag_invalid_input"] = hard_invalid
        out["warning_text"] = warnings_col

        out["gamma_SI"] = pd.to_numeric(out["gamma_mN_m"], errors="coerce") * 1e-3
        out["R_sg_SI_m"] = pd.to_numeric(out["R_sg_A"], errors="coerce") * 1e-10
        out["W_sg_SI_m"] = pd.to_numeric(out["W_sg_A"], errors="coerce") * 1e-10
        out["c_sg_SI_kg_m3"] = pd.to_numeric(out["c_sg_observed"], errors="coerce")
        out["c_dil_SI_kg_m3"] = pd.to_numeric(out["c_dil_observed"], errors="coerce")
        out["Meff_SG_SI_kg_mol"] = pd.to_numeric(out["Meff_SG_g_mol"], errors="coerce") * 1e-3

        out["Veff_SG_m3_mol"] = out["Meff_SG_SI_kg_mol"] / out["c_sg_SI_kg_m3"]
        out["curvature_smallness"] = (
            2.0 * out["gamma_SI"] * out["Veff_SG_m3_mol"] / (out["R_sg_SI_m"] * R_GAS * out["temperature_K"])
        )
        out["c_dil_inf_corrected"] = out["c_dil_observed"] * np.exp(-out["curvature_smallness"])
        out["Delta_c_observed"] = out["c_sg_observed"] - out["c_dil_inf_corrected"]
        out["c_bar_observed"] = 0.5 * (out["c_sg_observed"] + out["c_dil_inf_corrected"])
        out["interface_ratio"] = out["W_sg_A"] / out["R_sg_A"]
        out["flag_large_interface_ratio"] = out["interface_ratio"] >= self.config.interface_threshold
        out["flag_large_curvature"] = out["curvature_smallness"] >= self.config.curvature_threshold
        out["flag_negative_delta_c"] = out["Delta_c_observed"] <= 0
        out["flag_nonmonotone_control_dilute"] = False
        out["flag_nonmonotone_control_delta_c"] = False

        invalid_correction = ~np.isfinite(out["c_dil_inf_corrected"]) | ~np.isfinite(out["curvature_smallness"])
        out.loc[invalid_correction, "flag_invalid_input"] = True
        out.loc[invalid_correction, "warning_text"] = out.loc[invalid_correction, "warning_text"].astype(str) + ";invalid_curvature_correction"

        out = self.propagate_uncertainties(out)
        return out

    def propagate_uncertainties(self, df: pd.DataFrame) -> pd.DataFrame:
        """Propagate per-row concentration and shape uncertainties through the derived quantities."""
        out = df.copy()
        out["c_sg_std"] = pd.to_numeric(out["c_sg_std"], errors="coerce")
        out["c_dil_std"] = pd.to_numeric(out["c_dil_std"], errors="coerce")
        out["R_sg_std_A"] = pd.to_numeric(out["R_sg_std_A"], errors="coerce")
        out["W_sg_std_A"] = pd.to_numeric(out["W_sg_std_A"], errors="coerce")
        out["gamma_std_mN_m"] = pd.to_numeric(out["gamma_std_mN_m"], errors="coerce")

        out["gamma_std_SI"] = out["gamma_std_mN_m"] * 1e-3
        out["R_sg_std_SI_m"] = out["R_sg_std_A"] * 1e-10

        out["Veff_SG_std_m3_mol"] = np.where(
            (out["c_sg_observed"] > 0) & np.isfinite(out["c_sg_std"]) & np.isfinite(out["Veff_SG_m3_mol"]),
            out["Veff_SG_m3_mol"] * out["c_sg_std"] / out["c_sg_observed"],
            np.nan,
        )

        gamma_term = self._safe_relative(out["gamma_std_SI"], out["gamma_SI"])
        veff_term = self._safe_relative(out["Veff_SG_std_m3_mol"], out["Veff_SG_m3_mol"])
        r_term = self._safe_relative(out["R_sg_std_SI_m"], out["R_sg_SI_m"])
        out["curvature_smallness_std"] = out["curvature_smallness"] * np.sqrt(gamma_term**2 + veff_term**2 + r_term**2)

        # First-order linear propagation through exp(-x) is an approximation.
        # For highly fluctuating small droplets, symmetric propagation underestimates
        # the upper positive skew that would arise from a fuller log-normal treatment.
        rel_dil = self._safe_relative(out["c_dil_std"], out["c_dil_observed"])
        out["c_dil_inf_std"] = out["c_dil_inf_corrected"] * np.sqrt(rel_dil**2 + out["curvature_smallness_std"].fillna(0.0) ** 2)
        zero_dil = out["c_dil_observed"] == 0
        out.loc[zero_dil & np.isfinite(out["c_dil_std"]), "c_dil_inf_std"] = np.exp(-out.loc[zero_dil, "curvature_smallness"]) * out.loc[zero_dil, "c_dil_std"]
        out.loc[zero_dil & ~np.isfinite(out["c_dil_std"]), "c_dil_inf_std"] = np.nan

        out["Delta_c_std"] = np.sqrt(out["c_sg_std"].fillna(0.0) ** 2 + out["c_dil_inf_std"].fillna(0.0) ** 2)
        out["c_bar_std"] = 0.5 * np.sqrt(out["c_sg_std"].fillna(0.0) ** 2 + out["c_dil_inf_std"].fillna(0.0) ** 2)

        ci_low_cols = [
            "curvature_smallness_ci95_low",
            "Delta_c_ci95_low",
            "c_bar_ci95_low",
            "c_dil_inf_ci95_low",
        ]
        ci_high_cols = [
            "curvature_smallness_ci95_high",
            "Delta_c_ci95_high",
            "c_bar_ci95_high",
            "c_dil_inf_ci95_high",
        ]
        median_cols = [
            "curvature_smallness_mc_median",
            "Delta_c_mc_median",
            "c_bar_mc_median",
            "c_dil_inf_mc_median",
        ]
        for column in ci_low_cols + ci_high_cols + median_cols:
            out[column] = math.nan

        rng = np.random.default_rng(self.config.random_seed)
        for idx, row in out.iterrows():
            summary = self._row_mc_curvature_summary(row, rng)
            if not summary:
                continue
            for key, value in summary.items():
                out.at[idx, key] = value
        return out

    def _row_mc_curvature_summary(
        self,
        row: pd.Series,
        rng: np.random.Generator,
    ) -> Dict[str, float]:
        """Monte-Carlo summary of the curvature-corrected dilute concentration for one row (mean/SEM/quantiles)."""
        samples = max(int(self.config.curvature_mc_samples), 0)
        if samples < 2:
            return {}

        temp_K = self.safe_float(row.get("temperature_K", math.nan))
        meff_g_mol = self.safe_float(row.get("Meff_SG_g_mol", math.nan))
        c_sg = self.safe_float(row.get("c_sg_observed", math.nan))
        c_dil = self.safe_float(row.get("c_dil_observed", math.nan))
        R_sg_A = self.safe_float(row.get("R_sg_A", math.nan))
        gamma_mN_m = self.safe_float(row.get("gamma_mN_m", math.nan))
        if not all(math.isfinite(value) for value in [temp_K, meff_g_mol, c_sg, c_dil, R_sg_A, gamma_mN_m]):
            return {}
        if temp_K <= 0 or meff_g_mol <= 0 or c_sg <= 0 or R_sg_A <= 0 or gamma_mN_m < 0:
            return {}

        c_sg_draw = self._draw_truncated_normal(
            rng,
            c_sg,
            self.safe_float(row.get("c_sg_std", math.nan)),
            samples,
            lower=1e-12,
        )
        c_dil_draw = self._draw_truncated_normal(
            rng,
            max(c_dil, 0.0),
            self.safe_float(row.get("c_dil_std", math.nan)),
            samples,
            lower=0.0,
        )
        R_draw = self._draw_truncated_normal(
            rng,
            R_sg_A,
            self.safe_float(row.get("R_sg_std_A", math.nan)),
            samples,
            lower=1e-12,
        )
        gamma_draw = self._draw_truncated_normal(
            rng,
            gamma_mN_m,
            self.safe_float(row.get("gamma_std_mN_m", math.nan)),
            samples,
            lower=0.0,
        )

        meff_kg_mol = meff_g_mol * 1e-3
        gamma_si = gamma_draw * 1e-3
        R_si = R_draw * 1e-10
        Veff = meff_kg_mol / c_sg_draw
        curvature = 2.0 * gamma_si * Veff / (R_si * R_GAS * temp_K)
        c_dil_inf = c_dil_draw * np.exp(-curvature)
        delta_c = c_sg_draw - c_dil_inf
        c_bar = 0.5 * (c_sg_draw + c_dil_inf)

        alpha = 0.025
        return {
            "curvature_smallness_std": float(np.std(curvature, ddof=1)),
            "curvature_smallness_mc_median": float(np.quantile(curvature, 0.5)),
            "curvature_smallness_ci95_low": float(np.quantile(curvature, alpha)),
            "curvature_smallness_ci95_high": float(np.quantile(curvature, 1.0 - alpha)),
            "c_dil_inf_std": float(np.std(c_dil_inf, ddof=1)),
            "c_dil_inf_mc_median": float(np.quantile(c_dil_inf, 0.5)),
            "c_dil_inf_ci95_low": float(np.quantile(c_dil_inf, alpha)),
            "c_dil_inf_ci95_high": float(np.quantile(c_dil_inf, 1.0 - alpha)),
            "Delta_c_std": float(np.std(delta_c, ddof=1)),
            "Delta_c_mc_median": float(np.quantile(delta_c, 0.5)),
            "Delta_c_ci95_low": float(np.quantile(delta_c, alpha)),
            "Delta_c_ci95_high": float(np.quantile(delta_c, 1.0 - alpha)),
            "c_bar_std": float(np.std(c_bar, ddof=1)),
            "c_bar_mc_median": float(np.quantile(c_bar, 0.5)),
            "c_bar_ci95_low": float(np.quantile(c_bar, alpha)),
            "c_bar_ci95_high": float(np.quantile(c_bar, 1.0 - alpha)),
        }

    @staticmethod
    def _draw_truncated_normal(
        rng: np.random.Generator,
        mean: float,
        sigma: float,
        size: int,
        lower: float,
    ) -> np.ndarray:
        """Draw truncated-normal samples for the curvature Monte-Carlo (returns the mean if inputs are non-finite)."""
        if not math.isfinite(mean):
            return np.full(size, np.nan, dtype=float)
        if not math.isfinite(sigma) or sigma <= 0:
            return np.full(size, max(mean, lower), dtype=float)
        draws = rng.normal(loc=mean, scale=sigma, size=size)
        return np.maximum(draws, lower)

    @staticmethod
    def _safe_relative(num: pd.Series, den: pd.Series) -> pd.Series:
        """Element-wise num/den with non-finite/zero-denominator entries set to NaN."""
        num_arr = pd.to_numeric(num, errors="coerce")
        den_arr = pd.to_numeric(den, errors="coerce")
        out = pd.Series(np.nan, index=num_arr.index, dtype=float)
        valid = np.isfinite(num_arr) & np.isfinite(den_arr) & (den_arr > 0)
        out.loc[valid] = num_arr.loc[valid] / den_arr.loc[valid]
        return out

    # ------------------------------------------------------------------
    # Aggregation and diagnostics
    # ------------------------------------------------------------------
    def aggregate_replicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate replicate rows per (condition, temperature) into mean values with propagated uncertainties."""
        group_cols = [
            "condition_name",
            "condition_type",
            "condition_subtype",
            "temperature_K",
            "row_semantic_type",
        ]
        numeric_cols = [
            "c_sg_observed",
            "c_dil_observed",
            "R_sg_A",
            "W_sg_A",
            "gamma_mN_m",
            "Meff_SG_g_mol",
            "Veff_SG_m3_mol",
            "curvature_smallness",
            "c_dil_inf_corrected",
            "Delta_c_observed",
            "c_bar_observed",
            "interface_ratio",
            "c_sg_std",
            "c_dil_std",
            "R_sg_std_A",
            "W_sg_std_A",
            "gamma_std_mN_m",
            "c_dil_inf_std",
            "Delta_c_std",
            "c_bar_std",
        ]
        records: List[Dict[str, Any]] = []
        for key, grp in df.groupby(group_cols, dropna=False):
            record: Dict[str, Any] = {col: val for col, val in zip(group_cols, key)}
            record["n_reps"] = int(len(grp))
            for col in numeric_cols:
                series = pd.to_numeric(grp[col], errors="coerce")
                if series.notna().sum() > 1:
                    record[col] = float(series.mean())
                    record[f"{col}_sample_std"] = float(series.std(ddof=1))
                elif series.notna().sum() == 1:
                    record[col] = float(series.dropna().iloc[0])
                    record[f"{col}_sample_std"] = math.nan
                else:
                    record[col] = math.nan
                    record[f"{col}_sample_std"] = math.nan

            # Prefer replicate sample std when available; otherwise keep imported analysis sigma.
            for value_col, sigma_col in [
                ("c_sg_observed", "c_sg_std"),
                ("c_dil_observed", "c_dil_std"),
                ("R_sg_A", "R_sg_std_A"),
                ("W_sg_A", "W_sg_std_A"),
                ("gamma_mN_m", "gamma_std_mN_m"),
                ("c_dil_inf_corrected", "c_dil_inf_std"),
                ("Delta_c_observed", "Delta_c_std"),
                ("c_bar_observed", "c_bar_std"),
            ]:
                sample_std = record.get(f"{value_col}_sample_std", math.nan)
                if math.isfinite(sample_std):
                    record[sigma_col] = sample_std

            record["small_molecule_id"] = self._mode_or_first(grp["small_molecule_id"])
            record["small_molecule_name"] = self._mode_or_first(grp["small_molecule_name"])
            record["compound_name"] = self._mode_or_first(grp["compound_name"])
            record["compound_class"] = self._mode_or_first(grp["compound_class"])
            record["is_control"] = bool(grp["is_control"].any())
            record["is_average_row"] = bool(grp["is_average_row"].any())
            record["average_row_source"] = self._mode_or_first(grp["average_row_source"])
            record["average_row_membership"] = self._mode_or_first(grp["average_row_membership"])
            record["is_individual_sm"] = bool(grp["is_individual_sm"].any())
            record["temperature_source"] = self._mode_or_first(grp["temperature_source"])
            record["temperature_token"] = self._mode_or_first(grp["temperature_token"])
            record["temperature_path"] = self._mode_or_first(grp["temperature_path"])
            record["source_mode"] = self._mode_or_first(grp["source_mode"])
            record["summary_root"] = self._join_unique(grp["summary_root"])
            record["source_file"] = self._join_unique(grp["source_file"])
            record["uncertainty_source_mode"] = self._mode_or_mixed(grp["uncertainty_source_mode"])
            record["uncertainty_interpretation"] = "approximate_analysis_sigma"
            record["flag_invalid_input"] = bool(grp["flag_invalid_input"].any())
            record["flag_large_interface_ratio"] = bool(grp["flag_large_interface_ratio"].any())
            record["flag_large_curvature"] = bool(grp["flag_large_curvature"].any())
            record["flag_negative_delta_c"] = bool(grp["flag_negative_delta_c"].any())
            record["flag_nonmonotone_control_dilute"] = False
            record["flag_nonmonotone_control_delta_c"] = False
            record["warning_text"] = self._join_unique(grp["warning_text"])
            records.append(record)
        out = pd.DataFrame(records)
        out = out.sort_values(["condition_type", "condition_name", "temperature_K"]).reset_index(drop=True)
        return out

    def apply_control_shape_diagnostics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add control-branch shape diagnostics (e.g. paired sigmas) used by the binodal fit."""
        out = df.copy()
        ctrl = out[(out["is_control"])].sort_values("temperature_K")
        if ctrl.empty:
            raise RuntimeError("No control SG rows were found after filtering")
        idx = ctrl.index.to_list()
        for i in range(1, len(ctrl)):
            prev = ctrl.iloc[i - 1]
            curr = ctrl.iloc[i]
            dilute_diff = self.safe_float(curr["c_dil_inf_corrected"]) - self.safe_float(prev["c_dil_inf_corrected"])
            dilute_sigma = self._pair_sigma(curr.get("c_dil_inf_std", math.nan), prev.get("c_dil_inf_std", math.nan))
            if math.isfinite(dilute_diff) and dilute_diff < -self.config.monotonicity_sigma_factor * dilute_sigma:
                out.at[idx[i], "flag_nonmonotone_control_dilute"] = True
            delta_diff = self.safe_float(curr["Delta_c_observed"]) - self.safe_float(prev["Delta_c_observed"])
            delta_sigma = self._pair_sigma(curr.get("Delta_c_std", math.nan), prev.get("Delta_c_std", math.nan))
            if math.isfinite(delta_diff) and delta_diff > self.config.monotonicity_sigma_factor * delta_sigma:
                out.at[idx[i], "flag_nonmonotone_control_delta_c"] = True
        return out

    @staticmethod
    def _pair_sigma(a: float, b: float) -> float:
        """Combine two per-branch sigmas into a single paired uncertainty (quadrature when both finite)."""
        if math.isfinite(a) and math.isfinite(b):
            return math.sqrt(a * a + b * b)
        return 0.0

    # ------------------------------------------------------------------
    # Control fitting
    # ------------------------------------------------------------------
    def fit_control_bundle(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Fit the control (SG) binodal and its fallback/resampled variants, returning the fit bundle."""
        control_all = df[df["is_control"]].copy().sort_values("temperature_K")
        if control_all.empty:
            raise RuntimeError("No control SG rows available for fitting")
        fit_mask_base = ~control_all["flag_invalid_input"]
        fit_mask_main = fit_mask_base & (~control_all["flag_large_interface_ratio"])
        control_fit = control_all.loc[fit_mask_main].copy()
        fit_filter_mode = "invalid_and_interface_filtered"
        if len(control_fit) < 3:
            control_fit = control_all.loc[fit_mask_base].copy()
            fit_filter_mode = "invalid_only_relaxed_from_interface_filter"
            self.manifest["warnings"].append(
                "Main phase-diagram control fit relaxed broad-interface exclusion because "
                f"only {int(fit_mask_main.sum())} SG control points passed "
                f"W/R < {self.config.interface_threshold:.3f}. "
                f"Using {int(fit_mask_base.sum())} finite SG control points instead."
            )
        if len(control_fit) < 3:
            raise RuntimeError(
                "Need at least 3 valid control SG rows after relaxing broad-interface exclusion"
            )

        default_fit = self.fit_control_binodal(control_fit, loss="linear", fit_kind="default", include_flagged_points=False)
        robust_fit = None
        if self.config.robust_fit_sensitivity:
            robust_fit = self.fit_control_binodal(control_fit, loss=self.config.robust_loss, fit_kind="robust_sensitivity", include_flagged_points=False)

        flagged_fit = None
        if self.config.include_flagged_fit_sensitivity:
            flagged_df = control_all[~control_all["flag_invalid_input"]].copy()
            if not flagged_df.empty:
                flagged_fit = self.fit_control_binodal(flagged_df, loss="linear", fit_kind="include_flagged_sensitivity", include_flagged_points=True)

        resample_df, resample_summary = self.resample_control_fit(control_fit, default_fit)
        return {
            "default": default_fit,
            "robust": robust_fit,
            "include_flagged": flagged_fit,
            "resamples": resample_df,
            "resample_summary": resample_summary,
            "control_points": control_all,
            "control_fit_points": control_fit,
            "fit_filter_mode": fit_filter_mode,
        }

    def fit_control_binodal(
        self,
        df: pd.DataFrame,
        loss: str,
        fit_kind: str,
        include_flagged_points: bool,
    ) -> fit_result:
        """Least-squares fit of the apparent control binodal (Tc, c_c, A, B) to the SG dense/dilute branches."""
        temperatures = pd.to_numeric(df["temperature_K"], errors="coerce").to_numpy(dtype=float)
        c_sg = pd.to_numeric(df["c_sg_observed"], errors="coerce").to_numpy(dtype=float)
        c_dil_inf = pd.to_numeric(df["c_dil_inf_corrected"], errors="coerce").to_numpy(dtype=float)
        sigma_dense = pd.to_numeric(df.get("c_sg_std", np.nan), errors="coerce").to_numpy(dtype=float)
        sigma_dil = pd.to_numeric(df.get("c_dil_inf_std", np.nan), errors="coerce").to_numpy(dtype=float)

        finite = np.isfinite(temperatures) & np.isfinite(c_sg) & np.isfinite(c_dil_inf)
        if finite.sum() < 3:
            raise RuntimeError(f"Need at least 3 finite control points for {fit_kind} fit")

        temperatures = temperatures[finite]
        c_sg = c_sg[finite]
        c_dil_inf = c_dil_inf[finite]
        sigma_dense = sigma_dense[finite]
        sigma_dil = sigma_dil[finite]
        max_T = float(np.max(temperatures))
        delta_c = c_sg - c_dil_inf
        c_bar = 0.5 * (c_sg + c_dil_inf)

        c_c_init = float(np.median(c_sg))
        A_init = max(float(np.nanmax(delta_c)), 1e-6)
        Tc_init = max_T + 15.0
        B_init = 0.0
        bounds_lower = np.array([max_T + 0.1, 1e-12, 1e-12, -1e6], dtype=float)
        bounds_upper = np.array([max_T + 100.0, 1e6, 1e6, 1e6], dtype=float)
        starts = [
            np.array([Tc_init, c_c_init, A_init, B_init], dtype=float),
            np.array([max_T + 10.0, float(np.mean(c_bar)), max(A_init * 0.5, 1e-6), -0.1], dtype=float),
            np.array([max_T + 25.0, float(np.max(c_sg)), max(A_init * 1.2, 1e-6), 0.1], dtype=float),
        ]

        def residuals(params: np.ndarray) -> np.ndarray:
            Tc, c_c, A, B = params
            dense_model, dil_model = self.control_branch_model(temperatures, Tc, c_c, A, B)
            sigma_eff_dense = self.effective_sigma(c_sg, sigma_dense)
            sigma_eff_dil = self.effective_sigma(c_dil_inf, sigma_dil)
            r_dense = (c_sg - dense_model) / sigma_eff_dense
            r_dil = (c_dil_inf - dil_model) / sigma_eff_dil
            return np.concatenate([r_dense, r_dil])

        best = None
        fit_warnings: List[str] = []
        for start in starts:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", OptimizeWarning)
                    res = least_squares(
                        residuals,
                        x0=start,
                        bounds=(bounds_lower, bounds_upper),
                        loss=loss,
                        f_scale=self.config.robust_f_scale,
                        max_nfev=20000,
                    )
                if not res.success:
                    fit_warnings.append(f"least_squares reported non-success for start {start.tolist()}: {res.message}")
                    continue
                if best is None or np.sum(res.fun ** 2) < np.sum(best.fun ** 2):
                    best = res
            except (ValueError, OptimizeWarning, FloatingPointError) as exc:
                fit_warnings.append(f"fit start {start.tolist()} failed: {exc}")

        used_fallback = False
        message = "success"
        parameters: Dict[str, float]
        rmse: float
        if best is None:
            used_fallback = True
            message = "simultaneous fit failed; using Delta_c + c_bar fallback"
            parameters, rmse, fallback_warnings = self.fit_control_fallback(temperatures, delta_c, c_bar, sigma_dense, sigma_dil)
            fit_warnings.extend(fallback_warnings)
        else:
            Tc, c_c, A, B = best.x.tolist()
            if abs(Tc - bounds_upper[0]) < 1e-6:
                fit_warnings.append(
                    "Tc_app_control reached the upper fit bound; the critical temperature is underconstrained by the available coexistence points"
                )
            if abs(Tc - bounds_lower[0]) < 1e-6:
                fit_warnings.append(
                    "Tc_app_control reached the lower fit bound; the critical temperature is underconstrained by the available coexistence points"
                )
            parameters = {"Tc_app_control": Tc, "c_c_control": c_c, "A_control": A, "B_control": B, "beta_fixed": self.config.beta_critical}
            rmse = float(np.sqrt(np.mean(best.fun ** 2)))
            message = str(best.message)

        excluded: List[float] = []
        return fit_result(
            success=True,
            fit_kind=fit_kind,
            loss=loss,
            used_fallback=used_fallback,
            parameters=parameters,
            rmse=rmse,
            n_points=int(len(temperatures)),
            n_residuals=int(2 * len(temperatures)),
            message=message,
            eligible_temperatures_K=sorted([float(x) for x in temperatures]),
            excluded_temperatures_K=excluded,
            include_flagged_points=include_flagged_points,
            residual_scale_mode="branch_balanced_normalized",
            warnings=fit_warnings,
        )

    def fit_control_fallback(
        self,
        temperatures: np.ndarray,
        delta_c: np.ndarray,
        c_bar: np.ndarray,
        sigma_dense: np.ndarray,
        sigma_dil: np.ndarray,
    ) -> Tuple[Dict[str, float], float, List[str]]:
        """Two-stage fallback fit (diameter then width) used when the joint control binodal fit fails."""
        warnings_out: List[str] = []
        max_T = float(np.max(temperatures))
        sigma_delta = np.sqrt(np.nan_to_num(sigma_dense, nan=0.0) ** 2 + np.nan_to_num(sigma_dil, nan=0.0) ** 2)
        sigma_bar = 0.5 * sigma_delta

        def delta_resid(params: np.ndarray) -> np.ndarray:
            Tc, A = params
            model = A * np.maximum(Tc - temperatures, 1e-12) ** self.config.beta_critical
            return (delta_c - model) / self.effective_sigma(delta_c, sigma_delta)

        delta_fit = least_squares(
            delta_resid,
            x0=np.array([max_T + 15.0, max(np.nanmax(delta_c), 1e-6)], dtype=float),
            bounds=(np.array([max_T + 0.1, 1e-12]), np.array([max_T + 100.0, 1e6])),
            loss="linear",
            max_nfev=20000,
        )
        params_tc = float(delta_fit.x[0])
        delta_A = float(delta_fit.x[1])

        def bar_resid_local(params: np.ndarray) -> np.ndarray:
            c_c, B = params
            model = c_c + B * np.maximum(params_tc - temperatures, 0.0)
            return (c_bar - model) / self.effective_sigma(c_bar, sigma_bar)

        bar_fit = least_squares(
            bar_resid_local,
            x0=np.array([float(np.median(c_bar)), 0.0], dtype=float),
            bounds=(np.array([1e-12, -1e6]), np.array([1e6, 1e6])),
            loss="linear",
            max_nfev=20000,
        )
        c_c = float(bar_fit.x[0])
        B = float(bar_fit.x[1])
        rmse = float(np.sqrt(np.mean(np.concatenate([delta_fit.fun, bar_fit.fun]) ** 2)))
        return (
            {
                "Tc_app_control": params_tc,
                "c_c_control": c_c,
                "A_control": delta_A,
                "B_control": B,
                "beta_fixed": self.config.beta_critical,
            },
            rmse,
            warnings_out,
        )

    def resample_control_fit(self, control_fit_df: pd.DataFrame, fit_result: fit_result) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Bootstrap-resample the control fit to produce parameter confidence intervals."""
        rng = np.random.default_rng(self.config.random_seed)
        c_sg = control_fit_df["c_sg_observed"].to_numpy(dtype=float)
        c_dil_raw = control_fit_df["c_dil_observed"].to_numpy(dtype=float)
        r_sg = control_fit_df["R_sg_A"].to_numpy(dtype=float)
        gamma = control_fit_df["gamma_mN_m"].to_numpy(dtype=float)
        meff = control_fit_df["Meff_SG_g_mol"].to_numpy(dtype=float)
        temps = control_fit_df["temperature_K"].to_numpy(dtype=float)
        sig_dense = control_fit_df["c_sg_std"].to_numpy(dtype=float)
        sig_dil_raw = control_fit_df["c_dil_std"].to_numpy(dtype=float)
        sig_r = control_fit_df["R_sg_std_A"].to_numpy(dtype=float)
        sig_gamma = control_fit_df["gamma_std_mN_m"].to_numpy(dtype=float)
        rows: List[Dict[str, Any]] = []
        success = 0
        for i in range(self.config.bootstrap_samples):
            draw_dense = np.maximum(
                c_sg + np.where(np.isfinite(sig_dense), rng.normal(0.0, sig_dense), 0.0),
                1e-12,
            )
            draw_dil_raw = np.maximum(
                c_dil_raw + np.where(np.isfinite(sig_dil_raw), rng.normal(0.0, sig_dil_raw), 0.0),
                0.0,
            )
            draw_r = np.maximum(
                r_sg + np.where(np.isfinite(sig_r), rng.normal(0.0, sig_r), 0.0),
                1e-12,
            )
            draw_gamma = np.maximum(
                gamma + np.where(np.isfinite(sig_gamma), rng.normal(0.0, sig_gamma), 0.0),
                0.0,
            )
            draw_veff = (meff * 1e-3) / draw_dense
            draw_curvature = 2.0 * (draw_gamma * 1e-3) * draw_veff / ((draw_r * 1e-10) * R_GAS * temps)
            draw_dil = draw_dil_raw * np.exp(-draw_curvature)
            sample = control_fit_df.copy()
            sample["c_sg_observed"] = draw_dense
            sample["c_dil_observed"] = draw_dil_raw
            sample["R_sg_A"] = draw_r
            sample["gamma_mN_m"] = draw_gamma
            sample["Veff_SG_m3_mol"] = draw_veff
            sample["curvature_smallness"] = draw_curvature
            sample["c_dil_inf_corrected"] = draw_dil
            sample["Delta_c_observed"] = draw_dense - draw_dil
            sample["c_bar_observed"] = 0.5 * (draw_dense + draw_dil)
            try:
                res = self.fit_control_binodal(sample, loss="linear", fit_kind="bootstrap", include_flagged_points=False)
                success += 1
                row = {"sample_index": i, "fit_success": True}
                row.update(res.parameters)
                row["rmse"] = res.rmse
                rows.append(row)
            except Exception as exc:
                rows.append({"sample_index": i, "fit_success": False, "error": str(exc)})
        resample_df = pd.DataFrame(rows)
        param_rows = resample_df[resample_df["fit_success"]]
        summary: Dict[str, Any] = {
            "n_requested": self.config.bootstrap_samples,
            "n_success": int(success),
            "n_failed": int(self.config.bootstrap_samples - success),
            "success_fraction": float(success / self.config.bootstrap_samples) if self.config.bootstrap_samples > 0 else math.nan,
        }
        tc_lower_bound = float(np.max(temps) + 0.1) if len(temps) else math.nan
        tc_upper_bound = float(np.max(temps) + 100.0) if len(temps) else math.nan
        for col in ["Tc_app_control", "c_c_control", "A_control", "B_control"]:
            if col in param_rows.columns and not param_rows.empty:
                entry: Dict[str, Any] = {
                    "median": float(param_rows[col].median()),
                    "mean": float(param_rows[col].mean()),
                    "ci95_low": float(param_rows[col].quantile(0.025)),
                    "ci95_high": float(param_rows[col].quantile(0.975)),
                }
                if col == "Tc_app_control" and math.isfinite(tc_lower_bound) and math.isfinite(tc_upper_bound):
                    values = pd.to_numeric(param_rows[col], errors="coerce").to_numpy(dtype=float)
                    at_upper = int(np.sum(np.isclose(values, tc_upper_bound, atol=1e-6)))
                    at_lower = int(np.sum(np.isclose(values, tc_lower_bound, atol=1e-6)))
                    entry["lower_fit_bound_K"] = tc_lower_bound
                    entry["upper_fit_bound_K"] = tc_upper_bound
                    entry["n_at_lower_bound"] = at_lower
                    entry["n_at_upper_bound"] = at_upper
                    entry["fraction_at_lower_bound"] = float(at_lower / len(values)) if len(values) else math.nan
                    entry["fraction_at_upper_bound"] = float(at_upper / len(values)) if len(values) else math.nan
                    entry["interval_status"] = (
                        "truncated_by_fit_bounds" if (at_upper > 0 or at_lower > 0) else "regular"
                    )
                summary[col] = entry
        return resample_df, summary

    def effective_sigma(self, values: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Return a regularized per-point sigma (floored, finite) for use as least-squares weights."""
        values = np.asarray(values, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        rel_floor = self.config.rel_floor * np.maximum(np.abs(values), 1e-6)
        sigma_eff = np.where(np.isfinite(sigma) & (sigma > 0), sigma, rel_floor)
        sigma_eff = np.maximum(sigma_eff, rel_floor)
        sigma_eff[~np.isfinite(sigma_eff)] = 1.0
        return sigma_eff

    def control_branch_model(self, T: np.ndarray, Tc: float, c_c: float, A: float, B: float) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the dense and dilute control-binodal branches at temperatures T for given parameters."""
        delta = np.maximum(Tc - T, 1e-12)
        order = A * delta ** self.config.beta_critical
        cbar = c_c + B * delta
        dense = cbar + 0.5 * order
        dilute = cbar - 0.5 * order
        return dense, dilute

    def build_control_curve_table(self, control_bundle: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
        """Build a tabulated dense/dilute control binodal curve over a temperature grid for plotting."""
        fit_variants = [("default", control_bundle.get("default"))]
        if control_bundle.get("robust") is not None:
            fit_variants.append(("robust_sensitivity", control_bundle.get("robust")))
        if control_bundle.get("include_flagged") is not None:
            fit_variants.append(("include_flagged_sensitivity", control_bundle.get("include_flagged")))
        T_obs = df[df["is_control"]]["temperature_K"].dropna().to_numpy(dtype=float)
        T_min = float(np.min(T_obs)) if len(T_obs) else 280.0
        T_max = float(np.max(T_obs)) if len(T_obs) else 320.0
        rows: List[Dict[str, Any]] = []
        for label, fit in fit_variants:
            if fit is None:
                continue
            Tc = fit.parameters["Tc_app_control"]
            c_c = fit.parameters["c_c_control"]
            A = fit.parameters["A_control"]
            B = fit.parameters["B_control"]
            T_grid = np.linspace(T_min, Tc, 250)
            dense, dilute = self.control_branch_model(T_grid, Tc, c_c, A, B)
            for T, d, l in zip(T_grid, dense, dilute):
                rows.append(
                    {
                        "fit_kind": label,
                        "temperature_K": float(T),
                        "c_sg_control_fit": float(d),
                        "c_dil_control_fit": float(l),
                        "Delta_c_control_fit": float(d - l),
                        "c_bar_control_fit": float(0.5 * (d + l)),
                    }
                )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Perturbation metrics
    # ------------------------------------------------------------------
    def compute_perturbation_metrics(self, df: pd.DataFrame, control_bundle: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compute per-compound dissolution metrics (Delta_c and relative shifts) against the control fit."""
        out = df.copy()
        control_points = out[out["is_control"]].set_index("temperature_K")
        fit = control_bundle["default"]
        fit_curve_lookup = self.build_control_fit_lookup(fit, out)

        delta_rel = []
        delta_rel_std = []
        delta_abs = []
        dil_rel = []
        dil_rel_std = []
        unstable = []
        for _, row in out.iterrows():
            if row["is_control"]:
                delta_rel.append(0.0)
                delta_rel_std.append(0.0)
                delta_abs.append(0.0)
                dil_rel.append(0.0)
                dil_rel_std.append(0.0)
                unstable.append(False)
                continue
            T = self.safe_float(row["temperature_K"])
            if T not in control_points.index:
                delta_rel.append(math.nan)
                delta_rel_std.append(math.nan)
                delta_abs.append(math.nan)
                dil_rel.append(math.nan)
                dil_rel_std.append(math.nan)
                unstable.append(True)
                continue
            ctrl_row = control_points.loc[T]
            delta_ctrl = self.safe_float(ctrl_row["Delta_c_observed"])
            delta_pert = self.safe_float(row["Delta_c_observed"])
            delta_abs_val = delta_pert - delta_ctrl
            delta_abs.append(delta_abs_val)
            stable = math.isfinite(delta_ctrl) and abs(delta_ctrl) >= self.config.delta_c_ctrl_epsilon
            unstable.append(not stable)
            if stable:
                rel = delta_abs_val / delta_ctrl
                delta_rel.append(rel)
                delta_rel_std.append(
                    self.relative_difference_sigma(
                        num=delta_pert,
                        num_sigma=self.safe_float(row.get("Delta_c_std", math.nan)),
                        den=delta_ctrl,
                        den_sigma=self.safe_float(ctrl_row.get("Delta_c_std", math.nan)),
                    )
                )
            else:
                delta_rel.append(math.nan)
                delta_rel_std.append(math.nan)

            c_dil_ctrl = self.safe_float(ctrl_row["c_dil_inf_corrected"])
            c_dil_pert = self.safe_float(row["c_dil_inf_corrected"])
            if math.isfinite(c_dil_ctrl) and abs(c_dil_ctrl) >= self.config.delta_c_ctrl_epsilon:
                dil_rel_val = (c_dil_pert - c_dil_ctrl) / c_dil_ctrl
                dil_rel.append(dil_rel_val)
                dil_rel_std.append(
                    self.relative_difference_sigma(
                        num=c_dil_pert,
                        num_sigma=self.safe_float(row.get("c_dil_inf_std", math.nan)),
                        den=c_dil_ctrl,
                        den_sigma=self.safe_float(ctrl_row.get("c_dil_inf_std", math.nan)),
                    )
                )
            else:
                dil_rel.append(math.nan)
                dil_rel_std.append(math.nan)

            if self.config.control_reference_sensitivity:
                fit_dense, fit_dil = fit_curve_lookup(T)
                out.loc[out.index == row.name, "Delta_c_control_fit_reference"] = fit_dense - fit_dil
                out.loc[out.index == row.name, "c_dil_control_fit_reference"] = fit_dil

        out["delta_Delta_c_rel"] = delta_rel
        out["delta_Delta_c_rel_std"] = delta_rel_std
        out["delta_Delta_c_abs"] = delta_abs
        out["delta_c_dil_inf_rel"] = dil_rel
        out["delta_c_dil_inf_rel_std"] = dil_rel_std
        out["relative_metric_unstable"] = unstable

        records: List[Dict[str, Any]] = []
        pert_rows = out[~out["is_control"]].copy()
        for condition_name, grp in pert_rows.groupby("condition_name"):
            record: Dict[str, Any] = {
                "condition_name": condition_name,
                "condition_type": self._mode_or_first(grp["condition_type"]),
                "row_semantic_type": self._mode_or_first(grp["row_semantic_type"]),
                "n_temperatures": int(len(grp)),
                "mean_Delta_c": float(pd.to_numeric(grp["Delta_c_observed"], errors="coerce").mean()),
                "mean_delta_Delta_c_rel": float(pd.to_numeric(grp.loc[~grp["relative_metric_unstable"], "delta_Delta_c_rel"], errors="coerce").mean()),
                "mean_delta_Delta_c_abs": float(pd.to_numeric(grp["delta_Delta_c_abs"], errors="coerce").mean()),
                "mean_delta_c_dil_inf_rel": float(pd.to_numeric(grp.loc[~grp["relative_metric_unstable"], "delta_c_dil_inf_rel"], errors="coerce").mean()),
                "n_stable_relative_points": int((~grp["relative_metric_unstable"]).sum()),
                "fit_status": "not_attempted_by_policy",
                "Tc_app_constrained": math.nan,
                "Tc_app_constrained_shift_vs_control": math.nan,
                "A_pert": math.nan,
                "fit_warning": "",
            }
            if self.config.fit_perturbed_tc and len(grp) >= self.config.min_perturbed_tc_temperatures:
                fit_record = self.fit_constrained_perturbation_tc(grp, control_bundle["default"])
                record.update(fit_record)
            elif self.config.fit_perturbed_tc:
                record["fit_warning"] = (
                    f"Perturbed Tc fits disabled by policy for fewer than "
                    f"{self.config.min_perturbed_tc_temperatures} temperatures"
                )
            records.append(record)
        summary_df = pd.DataFrame(records)
        if not summary_df.empty:
            summary_df = summary_df.sort_values(["condition_type", "mean_delta_Delta_c_rel", "condition_name"], ascending=[True, True, True]).reset_index(drop=True)
            summary_df["dissolution_rank"] = summary_df["mean_delta_Delta_c_rel"].rank(method="dense", ascending=True)
        return out, summary_df

    def build_control_fit_lookup(self, fit: fit_result, df: pd.DataFrame):
        """Return a callable mapping temperature to the control (dense, dilute) concentrations for a fit."""
        Tc = fit.parameters["Tc_app_control"]
        c_c = fit.parameters["c_c_control"]
        A = fit.parameters["A_control"]
        B = fit.parameters["B_control"]

        def lookup(T: float) -> Tuple[float, float]:
            dense, dilute = self.control_branch_model(np.array([T], dtype=float), Tc, c_c, A, B)
            return float(dense[0]), float(dilute[0])

        return lookup

    def fit_average_binodal(
        self,
        df: pd.DataFrame,
        condition_type: str,
        control_fit: Optional[fit_result] = None,
        fit_method: str = "diameter_slope_constrained",
    ) -> Optional[fit_result]:
        """Binodal Ising fit for the DSM_AVE / NDSM_AVE points.

        fit_method:
          - "unconstrained"             : 4-param fit (Tc, c_c, A, B all free).
                                          2 DoF on 3-temp data.
          - "diameter_slope_constrained" : B fixed at control (Welsh 2022).
                                          3 DoF on 3-temp data. Default.
          - "bremer_pappu"               : c_c AND B fixed at control
                                          (Bremer/Pappu 2022 Nat Chem).
                                          4 DoF on 3-temp data. Strongest prior.
        Returns a fit_result with parameter keys renamed to *_pert_avg and a
        'fit_method' tag matching the input.
        """
        avg = df[
            (df.get("row_semantic_type") == "arithmetic_class_average")
            & (df.get("condition_type") == condition_type)
        ].copy()
        if avg.empty or control_fit is None:
            return None
        temperatures = pd.to_numeric(avg["temperature_K"], errors="coerce").to_numpy(dtype=float)
        c_sg = pd.to_numeric(avg["c_sg_observed"], errors="coerce").to_numpy(dtype=float)
        c_dil = pd.to_numeric(avg.get("c_dil_inf_corrected", np.nan), errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(temperatures) & np.isfinite(c_sg) & np.isfinite(c_dil)
        if finite.sum() < 3:
            return None
        max_T = float(np.max(temperatures[finite]))

        if fit_method == "unconstrained":
            try:
                base = self.fit_control_binodal(
                    avg, loss="linear", fit_kind=f"{condition_type.lower()}_average_unconstrained",
                    include_flagged_points=False,
                )
            except Exception:
                return None
            Tc = float(base.parameters.get("Tc_app_control", math.nan))
            c_c = float(base.parameters.get("c_c_control", math.nan))
            A = float(base.parameters.get("A_control", math.nan))
            B = float(base.parameters.get("B_control", math.nan))
            renamed = {
                "Tc_app_pert_avg": Tc, "c_c_pert_avg": c_c, "A_pert_avg": A, "B_pert_avg": B,
                "beta_fixed": self.config.beta_critical,
                "condition_type": condition_type, "fit_method": "unconstrained",
            }
            return fit_result(
                success=base.success, fit_kind=base.fit_kind, loss=base.loss,
                used_fallback=base.used_fallback, parameters=renamed,
                rmse=base.rmse, n_points=base.n_points, n_residuals=base.n_residuals,
                message=base.message,
                eligible_temperatures_K=base.eligible_temperatures_K,
                excluded_temperatures_K=base.excluded_temperatures_K,
                include_flagged_points=base.include_flagged_points,
                residual_scale_mode=base.residual_scale_mode,
                warnings=list(base.warnings),
            )

        if fit_method == "diameter_slope_constrained":
            constrained = self._fit_avg_binodal_diameter_constrained(
                temperatures[finite], c_sg[finite], c_dil[finite],
                control_fit=control_fit, max_T=max_T,
            )
        elif fit_method == "bremer_pappu":
            constrained = self._fit_avg_binodal_bremer_pappu(
                temperatures[finite], c_sg[finite], c_dil[finite],
                control_fit=control_fit, max_T=max_T,
            )
        else:
            raise ValueError(f"Unknown fit_method {fit_method!r}")
        if constrained is None:
            return None

        renamed_params = {
            "Tc_app_pert_avg": constrained.parameters.get("Tc_app_control"),
            "c_c_pert_avg": constrained.parameters.get("c_c_control"),
            "A_pert_avg": constrained.parameters.get("A_control"),
            "B_pert_avg": constrained.parameters.get("B_control"),
            "beta_fixed": constrained.parameters.get("beta_fixed", self.config.beta_critical),
            "condition_type": condition_type,
            "fit_method": fit_method,
        }
        return fit_result(
            success=constrained.success,
            fit_kind=constrained.fit_kind,
            loss=constrained.loss,
            used_fallback=constrained.used_fallback,
            parameters=renamed_params,
            rmse=constrained.rmse,
            n_points=constrained.n_points,
            n_residuals=constrained.n_residuals,
            message=constrained.message,
            eligible_temperatures_K=constrained.eligible_temperatures_K,
            excluded_temperatures_K=constrained.excluded_temperatures_K,
            include_flagged_points=constrained.include_flagged_points,
            residual_scale_mode=constrained.residual_scale_mode,
            warnings=list(constrained.warnings),
        )

    def fit_individual_binodal(
        self,
        df: pd.DataFrame,
        condition_name: str,
        control_fit: Optional[fit_result] = None,
        fit_method: str = "diameter_slope_constrained",
    ) -> Optional[fit_result]:
        """Per-compound binodal fit. Supports the same three fit_methods
        as fit_average_binodal: 'unconstrained', 'diameter_slope_constrained',
        'bremer_pappu'.
        """
        sub = df[(df.get("condition_name") == condition_name)
                 & (df.get("row_semantic_type") == "individual_compound")].copy()
        if sub.empty or control_fit is None:
            return None
        temperatures = pd.to_numeric(sub["temperature_K"], errors="coerce").to_numpy(dtype=float)
        c_sg = pd.to_numeric(sub["c_sg_observed"], errors="coerce").to_numpy(dtype=float)
        c_dil = pd.to_numeric(sub.get("c_dil_inf_corrected", np.nan), errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(temperatures) & np.isfinite(c_sg) & np.isfinite(c_dil)
        if finite.sum() < 3:
            return None
        max_T = float(np.max(temperatures[finite]))
        if fit_method == "unconstrained":
            try:
                return self.fit_control_binodal(
                    sub, loss="linear",
                    fit_kind=f"{condition_name}_unconstrained",
                    include_flagged_points=False,
                )
            except Exception:
                return None
        if fit_method == "diameter_slope_constrained":
            return self._fit_avg_binodal_diameter_constrained(
                temperatures[finite], c_sg[finite], c_dil[finite],
                control_fit=control_fit, max_T=max_T,
            )
        if fit_method == "bremer_pappu":
            return self._fit_avg_binodal_bremer_pappu(
                temperatures[finite], c_sg[finite], c_dil[finite],
                control_fit=control_fit, max_T=max_T,
            )
        raise ValueError(f"Unknown fit_method {fit_method!r}")

    def _fit_avg_binodal_bremer_pappu(
        self,
        temperatures: np.ndarray,
        c_sg: np.ndarray,
        c_dil: np.ndarray,
        control_fit: fit_result,
        max_T: float,
    ) -> Optional[fit_result]:
        """Bremer-Pappu (Nat Chem 2022) constrained binodal: c_c AND B
        fixed at control values; only Tc and A are free. Coupled dense +
        dilute residuals.

        2 free params from 6 data points = 4 DoF. Strongest prior of the
        three options. Justification: small chemical perturbations change
        interaction strength (Tc, A) but not polymer architecture (c_c,
        diameter slope B).
        """
        c_c_fixed = float(control_fit.parameters["c_c_control"])
        B_fixed = float(control_fit.parameters["B_control"])
        A_init = max(float(np.nanmax(c_sg - c_dil)), 1e-6)

        def residuals(params: np.ndarray) -> np.ndarray:
            Tc, A = params
            dense, dilute = self.control_branch_model(temperatures, Tc, c_c_fixed, A, B_fixed)
            sig_d = self.effective_sigma(c_sg, np.full_like(c_sg, np.nan))
            sig_l = self.effective_sigma(c_dil, np.full_like(c_dil, np.nan))
            return np.concatenate([(c_sg - dense) / sig_d, (c_dil - dilute) / sig_l])

        starts = [
            np.array([max_T + 20.0, A_init], dtype=float),
            np.array([max_T + 5.0, max(A_init * 0.5, 1e-6)], dtype=float),
            np.array([max_T + 50.0, max(A_init * 1.5, 1e-6)], dtype=float),
        ]
        bounds_lo = np.array([max_T + 0.1, 1e-12])
        bounds_hi = np.array([max_T + 200.0, 1e6])
        best = None
        for start in starts:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", OptimizeWarning)
                    res = least_squares(
                        residuals, x0=start,
                        bounds=(bounds_lo, bounds_hi),
                        loss="linear", max_nfev=20000,
                    )
                if not res.success:
                    continue
                if best is None or np.sum(res.fun ** 2) < np.sum(best.fun ** 2):
                    best = res
            except (ValueError, OptimizeWarning, FloatingPointError):
                continue
        if best is None:
            return None
        Tc, A = best.x.tolist()
        params = {
            "Tc_app_control": float(Tc),
            "c_c_control": c_c_fixed,
            "A_control": float(A),
            "B_control": B_fixed,
            "beta_fixed": self.config.beta_critical,
        }
        return fit_result(
            success=True,
            fit_kind="bremer_pappu",
            loss="linear",
            used_fallback=False,
            parameters=params,
            rmse=float(np.sqrt(np.mean(best.fun ** 2))),
            n_points=int(len(temperatures)),
            n_residuals=int(2 * len(temperatures)),
            message=str(best.message),
            eligible_temperatures_K=sorted([float(x) for x in temperatures]),
            excluded_temperatures_K=[],
            include_flagged_points=False,
            residual_scale_mode="branch_balanced_normalized",
            warnings=["Bremer-Pappu: c_c and B pinned to control values"],
        )

    def _fit_avg_binodal_diameter_constrained(
        self,
        temperatures: np.ndarray,
        c_sg: np.ndarray,
        c_dil: np.ndarray,
        control_fit: fit_result,
        max_T: float,
    ) -> Optional[fit_result]:
        """Diameter-slope-constrained binodal (Welsh/Mittag 2022 Mol Cell):
        only the rectilinear-diameter slope B is fixed at the control value.
        Tc, c_c, and A are all free. Coupled dense + dilute residuals.

        Rationale: B is the parameter most data-hungry to pin down (a slow
        linear temperature dependence in the diameter), so borrowing it
        from the well-determined control fit is the lightest possible
        constraint. c_c and A reflect actual perturbation-induced shifts
        in critical density and order-parameter amplitude.

        3 free params from 6 data points (3 T x 2 branches) = 3 DoF.
        """
        B_fixed = float(control_fit.parameters["B_control"])
        c_c_init = float(np.median(c_sg))
        A_init = max(float(np.nanmax(c_sg - c_dil)), 1e-6)

        def residuals(params: np.ndarray) -> np.ndarray:
            Tc, c_c, A = params
            dense, dilute = self.control_branch_model(temperatures, Tc, c_c, A, B_fixed)
            sig_d = self.effective_sigma(c_sg, np.full_like(c_sg, np.nan))
            sig_l = self.effective_sigma(c_dil, np.full_like(c_dil, np.nan))
            return np.concatenate([(c_sg - dense) / sig_d, (c_dil - dilute) / sig_l])

        starts = [
            np.array([max_T + 20.0, c_c_init, A_init], dtype=float),
            np.array([max_T + 5.0, c_c_init, max(A_init * 0.5, 1e-6)], dtype=float),
            np.array([max_T + 50.0, c_c_init * 1.1, max(A_init * 1.5, 1e-6)], dtype=float),
        ]
        bounds_lo = np.array([max_T + 0.1, 1e-12, 1e-12])
        bounds_hi = np.array([max_T + 200.0, 1e6, 1e6])
        best = None
        for start in starts:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", OptimizeWarning)
                    res = least_squares(
                        residuals, x0=start,
                        bounds=(bounds_lo, bounds_hi),
                        loss="linear", max_nfev=20000,
                    )
                if not res.success:
                    continue
                if best is None or np.sum(res.fun ** 2) < np.sum(best.fun ** 2):
                    best = res
            except (ValueError, OptimizeWarning, FloatingPointError):
                continue
        if best is None:
            return None
        Tc, c_c, A = best.x.tolist()
        params = {
            "Tc_app_control": float(Tc),
            "c_c_control": float(c_c),
            "A_control": float(A),
            "B_control": B_fixed,
            "beta_fixed": self.config.beta_critical,
        }
        return fit_result(
            success=True,
            fit_kind="diameter_slope_constrained",
            loss="linear",
            used_fallback=False,
            parameters=params,
            rmse=float(np.sqrt(np.mean(best.fun ** 2))),
            n_points=int(len(temperatures)),
            n_residuals=int(2 * len(temperatures)),
            message=str(best.message),
            eligible_temperatures_K=sorted([float(x) for x in temperatures]),
            excluded_temperatures_K=[],
            include_flagged_points=False,
            residual_scale_mode="branch_balanced_normalized",
            warnings=["Diameter-slope-constrained: B pinned to control value (Welsh/Mittag 2022)"],
        )

    def build_average_curve(
        self,
        fit: fit_result,
        T_min: float,
        T_max_hint: float,
    ) -> pd.DataFrame:
        """Return a dataframe with dense/dilute branches for an average fit
        on a fine T grid from T_min to Tc_app_pert_avg.
        """
        Tc = float(fit.parameters["Tc_app_pert_avg"])
        c_c = float(fit.parameters["c_c_pert_avg"])
        A = float(fit.parameters["A_pert_avg"])
        B = float(fit.parameters["B_pert_avg"])
        T_lo = float(min(T_min, T_max_hint))
        T_hi = float(Tc)
        if T_hi <= T_lo:
            return pd.DataFrame()
        T_grid = np.linspace(T_lo, T_hi, 250)
        dense, dilute = self.control_branch_model(T_grid, Tc, c_c, A, B)
        return pd.DataFrame(
            {
                "temperature_K": T_grid,
                "c_dense_fit": dense,
                "c_dilute_fit": dilute,
            }
        )

    def fit_constrained_perturbation_tc(self, grp: pd.DataFrame, control_fit: fit_result) -> Dict[str, Any]:
        """Fit a constrained apparent Tc for one perturbation group from its Delta_c-vs-temperature trend."""
        valid = grp[(~grp["flag_invalid_input"]) & np.isfinite(grp["Delta_c_observed"])].copy().sort_values("temperature_K")
        if len(valid) < 3:
            return {
                "fit_status": "insufficient_points",
                "fit_warning": "Need at least 3 valid temperatures for constrained apparent Tc fit",
            }
        temperatures = valid["temperature_K"].to_numpy(dtype=float)
        delta_obs = valid["Delta_c_observed"].to_numpy(dtype=float)
        sigma_delta = valid["Delta_c_std"].to_numpy(dtype=float)
        max_T = float(np.max(temperatures))

        def residuals(params: np.ndarray) -> np.ndarray:
            A_pert, Tc_app = params
            model = A_pert * np.maximum(Tc_app - temperatures, 1e-12) ** self.config.beta_critical
            sigma_eff = self.effective_sigma(delta_obs, sigma_delta)
            return (delta_obs - model) / sigma_eff

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                res = least_squares(
                    residuals,
                    x0=np.array([max(float(np.nanmax(delta_obs)), 1e-6), max_T + 10.0], dtype=float),
                    bounds=(np.array([1e-12, max_T + 0.1]), np.array([1e6, max_T + 100.0])),
                    loss="linear",
                    max_nfev=20000,
                )
            if not res.success:
                return {
                    "fit_status": "failed",
                    "fit_warning": res.message,
                }
            A_pert = float(res.x[0])
            Tc_app = float(res.x[1])
            fit_warning = ""
            if abs(Tc_app - (max_T + 100.0)) < 1e-6:
                fit_warning = "Tc_app_constrained reached upper fit bound; perturbation critical temperature is underconstrained"
            return {
                "fit_status": "success",
                "fit_warning": fit_warning,
                "A_pert": A_pert,
                "Tc_app_constrained": Tc_app,
                "Tc_app_constrained_shift_vs_control": Tc_app - control_fit.parameters["Tc_app_control"],
            }
        except (ValueError, OptimizeWarning, FloatingPointError) as exc:
            return {
                "fit_status": "failed",
                "fit_warning": str(exc),
            }

    def relative_difference_sigma(self, num: float, num_sigma: float, den: float, den_sigma: float) -> float:
        """Propagate uncertainty for a relative difference (num-den)/den given both values and sigmas."""
        if not (math.isfinite(num) and math.isfinite(den) and den != 0):
            return math.nan
        if not (math.isfinite(num_sigma) and math.isfinite(den_sigma)):
            return math.nan
        return math.sqrt((num_sigma / den) ** 2 + ((num * den_sigma) / (den * den)) ** 2)

    # ------------------------------------------------------------------
    # Summaries and outputs
    # ------------------------------------------------------------------
    def build_dissolution_summary(self, df: pd.DataFrame, perturb_summary: pd.DataFrame) -> pd.DataFrame:
        """Assemble the per-condition dissolution summary table from the perturbation metrics."""
        rows: List[Dict[str, Any]] = []
        if not perturb_summary.empty:
            for _, row in perturb_summary.iterrows():
                rows.append({"summary_level": "condition", **row.to_dict()})
        group_df = df[(~df["is_control"]) & (df["row_semantic_type"] == "individual_compound")].copy()
        if not group_df.empty:
            stable = group_df[~group_df["relative_metric_unstable"]].copy()
            grouped = stable.groupby(["condition_type", "temperature_K"], dropna=False)
            for (condition_type, temperature_K), grp in grouped:
                rows.append(
                    {
                        "summary_level": "condition_type_temperature",
                        "condition_name": f"{condition_type}_GROUP",
                        "condition_type": condition_type,
                        "temperature_K": temperature_K,
                        "mean_Delta_c": float(pd.to_numeric(grp["Delta_c_observed"], errors="coerce").mean()),
                        "mean_delta_Delta_c_rel": float(pd.to_numeric(grp["delta_Delta_c_rel"], errors="coerce").mean()),
                        "mean_delta_Delta_c_abs": float(pd.to_numeric(grp["delta_Delta_c_abs"], errors="coerce").mean()),
                        "mean_delta_c_dil_inf_rel": float(pd.to_numeric(grp["delta_c_dil_inf_rel"], errors="coerce").mean()),
                        "n_points": int(len(grp)),
                    }
                )
        return pd.DataFrame(rows)

    def run_meff_scale_sensitivity(self, aggregated_df: pd.DataFrame) -> pd.DataFrame:
        """Re-run the analysis across Meff_SG scale factors to report sensitivity of the outputs."""
        if not self.config.meff_sg_scale_sensitivity:
            return pd.DataFrame()
        rows: List[Dict[str, Any]] = []
        for scale in self.config.meff_sg_scale_sensitivity:
            scaled = aggregated_df.copy()
            scaled["Meff_SG_g_mol"] = scaled["Meff_SG_g_mol"] * scale
            scaled["Meff_SG_SI_kg_mol"] = scaled["Meff_SG_g_mol"] * 1e-3
            scaled["Veff_SG_m3_mol"] = scaled["Meff_SG_SI_kg_mol"] / scaled["c_sg_observed"]
            scaled["curvature_smallness"] = (
                2.0 * scaled["gamma_mN_m"] * 1e-3 * scaled["Veff_SG_m3_mol"] / (scaled["R_sg_A"] * 1e-10 * R_GAS * scaled["temperature_K"])
            )
            scaled["c_dil_inf_corrected"] = scaled["c_dil_observed"] * np.exp(-scaled["curvature_smallness"])
            scaled["Delta_c_observed"] = scaled["c_sg_observed"] - scaled["c_dil_inf_corrected"]
            scaled["c_bar_observed"] = 0.5 * (scaled["c_sg_observed"] + scaled["c_dil_inf_corrected"])
            scaled = self.apply_control_shape_diagnostics(scaled)
            try:
                fit_bundle = self.fit_control_bundle(scaled)
                fit = fit_bundle["default"]
                rows.append(
                    {
                        "meff_scale_factor": scale,
                        "fit_success": True,
                        **fit.parameters,
                        "rmse": fit.rmse,
                    }
                )
            except Exception as exc:
                rows.append({"meff_scale_factor": scale, "fit_success": False, "error": str(exc)})
        return pd.DataFrame(rows)

    def write_outputs(
        self,
        processed_df: pd.DataFrame,
        aggregated_df: pd.DataFrame,
        control_bundle: Dict[str, Any],
        control_curve_df: pd.DataFrame,
        perturb_summary: pd.DataFrame,
        dissolution_summary: pd.DataFrame,
        sensitivity_df: pd.DataFrame,
    ) -> None:
        """Write all processed tables, fit results, summaries, and the manifest to the RESULTS directory."""
        processed_df = self._add_csat_alias_columns(processed_df)
        aggregated_df = self._add_csat_alias_columns(aggregated_df)
        control_curve_df = self._add_csat_alias_columns(control_curve_df)
        processed_df.to_csv(self.result_dir / "processed_phase_data.csv", index=False)
        aggregated_df.to_csv(self.result_dir / "aggregated_phase_data.csv", index=False)
        control_curve_df.to_csv(self.result_dir / "control_binodal_curve.csv", index=False)
        perturb_summary.to_csv(self.result_dir / "perturbation_fit_summary.csv", index=False)
        dissolution_summary.to_csv(self.result_dir / "dissolution_summary.csv", index=False)
        control_bundle["resamples"].to_csv(self.result_dir / "control_fit_resamples.csv", index=False)
        if not sensitivity_df.empty:
            sensitivity_df.to_csv(self.result_dir / "meff_scale_sensitivity.csv", index=False)

        control_json = {
            "default_fit": self._fit_result_dict(control_bundle["default"]),
            "robust_fit": self._fit_result_dict(control_bundle.get("robust")),
            "include_flagged_fit": self._fit_result_dict(control_bundle.get("include_flagged")),
            "resample_summary": self._jsonable(control_bundle["resample_summary"]),
            "fit_filter_mode": control_bundle.get("fit_filter_mode", ""),
            "scientific_notes": [
                "dilute concentration is corrected early using a Gibbs-Thomson-style curvature correction",
                "dense branch is intentionally left uncorrected as an approximation",
                "the resulting phase diagram is a projected SG biopolymer mixture binodal",
                "the fitted Tc is an apparent critical temperature",
                "DSM/NDSM apparent critical temperatures are disabled by policy unless enough temperatures are available for a minimally constrained fit",
                "Delta_c is the primary perturbation/dissolution metric",
                "RCC-derived Meff_SG is exact for the model-level SG species-weighted average chain molar mass",
                "the approximation enters later through Veff_SG = Meff_SG / c_sg, which is only an apparent molar-volume surrogate",
                "Gibbs-Thomson-corrected dilute-branch uncertainties are propagated with Monte Carlo sampling of c_sg, c_dil, R_cond, and gamma rather than only first-order symmetric propagation",
            ],
        }
        with (self.result_dir / "control_binodal_fit.json").open("w", encoding="utf-8") as fh:
            json.dump(self._jsonable(control_json), fh, indent=2)
        self.write_manifest()

    def _fit_result_dict(self, fit: Optional[fit_result]) -> Optional[Dict[str, Any]]:
        """Serialize a fit_result to a plain dict (or None) for JSON/manifest output."""
        if fit is None:
            return None
        return self._jsonable(asdict(fit))

    @staticmethod
    def _add_csat_alias_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Add c_sat alias columns to a dataframe for downstream/legacy column-name compatibility."""
        out = df.copy()
        if "c_dil_observed" in out.columns and "C_sat_raw" not in out.columns:
            out["C_sat_raw"] = out["c_dil_observed"]
        if "c_dil_inf_corrected" in out.columns and "C_sat_corrected" not in out.columns:
            out["C_sat_corrected"] = out["c_dil_inf_corrected"]
        if "c_dil_inf_std" in out.columns and "C_sat_corrected_std" not in out.columns:
            out["C_sat_corrected_std"] = out["c_dil_inf_std"]
        if "c_dil_control_fit" in out.columns and "C_sat_fit" not in out.columns:
            out["C_sat_fit"] = out["c_dil_control_fit"]
        return out

    def write_manifest(self) -> None:
        """Write the run manifest (config, inputs, temperature mapping, warnings) to input_manifest.json."""
        with (self.result_dir / "input_manifest.json").open("w", encoding="utf-8") as fh:
            json.dump(self._jsonable(self.manifest), fh, indent=2)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def make_plots(
        self,
        df: pd.DataFrame,
        control_bundle: Dict[str, Any],
        control_curve_df: pd.DataFrame,
        perturb_summary: pd.DataFrame,
    ) -> None:
        """Render the two paper phase-diagram (binodal) figures.

        Only the two binodal panels used by the paper are emitted:
          - class-average overlay (SG + DSM_AVE + NDSM_AVE),
          - the same panel plus the individual per-compound binodal curves.
        All accompanying data/CSV computation is preserved in the variant
        helper. No other figures are produced.
        """
        self.plot_perturbation_binodal(df, control_bundle, control_curve_df, mode="averages")
        self.plot_perturbation_binodal(df, control_bundle, control_curve_df, mode="all")

    # ------------------------------------------------------------------
    # Cross-temperature KMeans analysis
    # ------------------------------------------------------------------
    def run_temperature_transfer_kmeans(self) -> None:
        """Run the cross-temperature KMeans transfer analysis: per-T best features, transfer matrix, and pooled search."""
        contexts = self._build_kmeans_temperature_contexts()
        if len(contexts) < 2:
            warning = (
                "Skipping KMeans temperature-transfer analysis: need at least two "
                "temperatures with valid DSM/NDSM KMeans outputs."
            )
            self.manifest["warnings"].append(warning)
            self.manifest["kmeans_temperature_transfer"]["status"] = "skipped"
            self.manifest["kmeans_temperature_transfer"]["reason"] = warning
            return

        temperatures = sorted(contexts)
        best_feature_rows: List[Dict[str, Any]] = []
        transfer_rows: List[Dict[str, Any]] = []
        for source_temp in temperatures:
            source_context = contexts[source_temp]
            best_feature_rows.append({
                "Temperature_K": source_temp,
                "Summary_Root": str(source_context["summary_root"]),
                "Best_Accuracy_Percent": source_context["best_accuracy_percent"],
                "Best_Feature_Count": len(source_context["best_features"]),
                "Best_Features": " | ".join(source_context["best_features"]),
                "Source_KMeans_File": str(source_context["kmeans_iteration_path"]),
            })
            for target_temp in temperatures:
                try:
                    result = self._evaluate_kmeans_feature_set(
                        contexts[target_temp],
                        source_context["best_features"],
                        context_label=f"transfer_{int(source_temp)}_to_{int(target_temp)}",
                    )
                except ValueError as exc:
                    transfer_rows.append({
                        "Source_Temperature_K": source_temp,
                        "Target_Temperature_K": target_temp,
                        "Source_Feature_Count": len(source_context["best_features"]),
                        "Requested_Features": " | ".join(source_context["best_features"]),
                        "Usable_Features": "",
                        "Usable_Feature_Count": 0,
                        "Accuracy_Percent": math.nan,
                        "Balanced_Accuracy_Percent": math.nan,
                        "DSM_Recall_Percent": math.nan,
                        "NDSM_Correct_Percent": math.nan,
                        "False_Positive_Percent": math.nan,
                        "False_Negative_Percent": math.nan,
                        "Silhouette": math.nan,
                        "Error": str(exc),
                    })
                    continue
                transfer_rows.append({
                    "Source_Temperature_K": source_temp,
                    "Target_Temperature_K": target_temp,
                    "Source_Feature_Count": len(source_context["best_features"]),
                    "Requested_Features": " | ".join(source_context["best_features"]),
                    "Usable_Features": " | ".join(result["usable_features"]),
                    "Usable_Feature_Count": len(result["usable_features"]),
                    "Accuracy_Percent": result["accuracy_percent"],
                    "Balanced_Accuracy_Percent": result["balanced_accuracy_percent"],
                    "DSM_Recall_Percent": result["dsm_recall_percent"],
                    "NDSM_Correct_Percent": result["ndsm_correct_percent"],
                    "False_Positive_Percent": result["false_positive_percent"],
                    "False_Negative_Percent": result["false_negative_percent"],
                    "Silhouette": result["silhouette"],
                    "Error": "",
                })

        best_feature_df = pd.DataFrame(best_feature_rows).sort_values("Temperature_K")
        transfer_df = pd.DataFrame(transfer_rows).sort_values(
            ["Source_Temperature_K", "Target_Temperature_K"]
        )
        best_feature_df.to_csv(
            self.kmeans_result_dir / "kmeans_temperature_best_features.csv",
            index=False,
        )
        transfer_df.to_csv(
            self.kmeans_result_dir / "kmeans_temperature_transfer_matrix.csv",
            index=False,
        )

        union_features = self._ordered_unique(
            feature
            for temp in temperatures
            for feature in contexts[temp]["best_features"]
        )
        pooled_context = self._build_pooled_kmeans_context(contexts)

        union_rows: List[Dict[str, Any]] = []
        for target_temp in temperatures:
            result = self._evaluate_kmeans_feature_set(
                contexts[target_temp],
                union_features,
                context_label=f"all_temp_features_{int(target_temp)}",
            )
            union_rows.append({
                "Dataset": self._format_kmeans_dataset_label(target_temp),
                "Temperature_K": target_temp,
                "Accuracy_Percent": result["accuracy_percent"],
                "Balanced_Accuracy_Percent": result["balanced_accuracy_percent"],
                "DSM_Recall_Percent": result["dsm_recall_percent"],
                "NDSM_Correct_Percent": result["ndsm_correct_percent"],
                "False_Positive_Percent": result["false_positive_percent"],
                "False_Negative_Percent": result["false_negative_percent"],
                "Silhouette": result["silhouette"],
                "Usable_Feature_Count": len(result["usable_features"]),
                "Usable_Features": " | ".join(result["usable_features"]),
            })

        pooled_union = self._evaluate_kmeans_feature_set(
            pooled_context,
            union_features,
            context_label="all_temp_features_pooled",
        )
        union_rows.append({
            "Dataset": "All",
            "Temperature_K": math.nan,
            "Accuracy_Percent": pooled_union["accuracy_percent"],
            "Balanced_Accuracy_Percent": pooled_union["balanced_accuracy_percent"],
            "DSM_Recall_Percent": pooled_union["dsm_recall_percent"],
            "NDSM_Correct_Percent": pooled_union["ndsm_correct_percent"],
            "False_Positive_Percent": pooled_union["false_positive_percent"],
            "False_Negative_Percent": pooled_union["false_negative_percent"],
            "Silhouette": pooled_union["silhouette"],
            "Usable_Feature_Count": len(pooled_union["usable_features"]),
            "Usable_Features": " | ".join(pooled_union["usable_features"]),
        })
        union_df = pd.DataFrame(union_rows)
        union_df.to_csv(
            self.kmeans_result_dir / "kmeans_all_temperature_feature_union_results.csv",
            index=False,
        )
        self._write_kmeans_projection_bundle(
            stem="kmeans_all_temperature_feature_union",
            result=pooled_union,
            plotter=pooled_context["plotter"],
        )

        pooled_search_df, pooled_best = self._run_pooled_kmeans_greedy_search(pooled_context)
        pooled_search_df.to_csv(
            self.kmeans_result_dir / "kmeans_all_temperature_pooled_search.csv",
            index=False,
        )
        pooled_best_summary = pd.DataFrame([{
            "Accuracy_Percent": pooled_best["accuracy_percent"],
            "Balanced_Accuracy_Percent": pooled_best["balanced_accuracy_percent"],
            "DSM_Recall_Percent": pooled_best["dsm_recall_percent"],
            "NDSM_Correct_Percent": pooled_best["ndsm_correct_percent"],
            "False_Positive_Percent": pooled_best["false_positive_percent"],
            "False_Negative_Percent": pooled_best["false_negative_percent"],
            "Silhouette": pooled_best["silhouette"],
            "Feature_Count": len(pooled_best["usable_features"]),
            "Features": " | ".join(pooled_best["usable_features"]),
        }])
        pooled_best_summary.to_csv(
            self.kmeans_result_dir / "kmeans_all_temperature_pooled_best_subset.csv",
            index=False,
        )
        self._write_kmeans_projection_bundle(
            stem="kmeans_all_temperature_pooled_best_subset",
            result=pooled_best,
            plotter=pooled_context["plotter"],
        )

        pooled_best_rows: List[Dict[str, Any]] = []
        for target_temp in temperatures:
            result = self._evaluate_kmeans_feature_set(
                contexts[target_temp],
                pooled_best["usable_features"],
                context_label=f"pooled_best_subset_{int(target_temp)}",
            )
            pooled_best_rows.append({
                "Dataset": self._format_kmeans_dataset_label(target_temp),
                "Temperature_K": target_temp,
                "Accuracy_Percent": result["accuracy_percent"],
                "Balanced_Accuracy_Percent": result["balanced_accuracy_percent"],
                "DSM_Recall_Percent": result["dsm_recall_percent"],
                "NDSM_Correct_Percent": result["ndsm_correct_percent"],
                "False_Positive_Percent": result["false_positive_percent"],
                "False_Negative_Percent": result["false_negative_percent"],
                "Silhouette": result["silhouette"],
                "Usable_Feature_Count": len(result["usable_features"]),
                "Usable_Features": " | ".join(result["usable_features"]),
            })
        pooled_best_rows.append({
            "Dataset": "All",
            "Temperature_K": math.nan,
            "Accuracy_Percent": pooled_best["accuracy_percent"],
            "Balanced_Accuracy_Percent": pooled_best["balanced_accuracy_percent"],
            "DSM_Recall_Percent": pooled_best["dsm_recall_percent"],
            "NDSM_Correct_Percent": pooled_best["ndsm_correct_percent"],
            "False_Positive_Percent": pooled_best["false_positive_percent"],
            "False_Negative_Percent": pooled_best["false_negative_percent"],
            "Silhouette": pooled_best["silhouette"],
            "Usable_Feature_Count": len(pooled_best["usable_features"]),
            "Usable_Features": " | ".join(pooled_best["usable_features"]),
        })
        pooled_best_eval_df = pd.DataFrame(pooled_best_rows)
        pooled_best_eval_df.to_csv(
            self.kmeans_result_dir / "kmeans_all_temperature_pooled_best_subset_by_temperature.csv",
            index=False,
        )

        feature_set_records: List[Dict[str, Any]] = []
        for source_temp in temperatures:
            feature_set_records.append({
                "label": f"{int(source_temp)} best",
                "stem": f"kmeans_feature_set_{int(source_temp)}_best",
                "features": list(contexts[source_temp]["best_features"]),
                "plotter": contexts[source_temp]["plotter"],
            })
        feature_set_records.extend([
            {
                "label": "Union",
                "stem": "kmeans_feature_set_union",
                "features": list(union_features),
                "plotter": pooled_context["plotter"],
            },
            {
                "label": "Best overall",
                "stem": "kmeans_feature_set_best_overall",
                "features": list(pooled_best["usable_features"]),
                "plotter": pooled_context["plotter"],
            },
        ])
        feature_set_accuracy_rows: List[Dict[str, Any]] = []
        for feature_set in feature_set_records:
            rows: List[Dict[str, Any]] = []
            for target_temp in temperatures:
                target_context = contexts[target_temp]
                result = self._evaluate_kmeans_feature_set(
                    target_context,
                    feature_set["features"],
                    context_label=f"{feature_set['stem']}_{int(target_temp)}",
                )
                dataset_label = self._format_kmeans_dataset_label(target_temp)
                rows.append({
                    "Feature_Set": feature_set["label"],
                    "Dataset": dataset_label,
                    "Temperature_K": target_temp,
                    "Accuracy_Percent": result["accuracy_percent"],
                    "Balanced_Accuracy_Percent": result["balanced_accuracy_percent"],
                    "DSM_Recall_Percent": result["dsm_recall_percent"],
                    "NDSM_Correct_Percent": result["ndsm_correct_percent"],
                    "False_Positive_Percent": result["false_positive_percent"],
                    "False_Negative_Percent": result["false_negative_percent"],
                    "Silhouette": result["silhouette"],
                    "Usable_Feature_Count": len(result["usable_features"]),
                    "Usable_Features": " | ".join(result["usable_features"]),
                    "Usable_Features_Display": " | ".join(
                        self._format_kmeans_feature_display_labels(
                            target_context["plotter"],
                            result["usable_features"],
                        )
                    ),
                })
                scatter_stem = f"{feature_set['stem']}_{int(target_temp)}K"
                self._write_kmeans_projection_scatter_bundle(scatter_stem, result)

            pooled_result = self._evaluate_kmeans_feature_set(
                pooled_context,
                feature_set["features"],
                context_label=f"{feature_set['stem']}_all",
            )
            rows.append({
                "Feature_Set": feature_set["label"],
                "Dataset": "All",
                "Temperature_K": math.nan,
                "Accuracy_Percent": pooled_result["accuracy_percent"],
                "Balanced_Accuracy_Percent": pooled_result["balanced_accuracy_percent"],
                "DSM_Recall_Percent": pooled_result["dsm_recall_percent"],
                "NDSM_Correct_Percent": pooled_result["ndsm_correct_percent"],
                "False_Positive_Percent": pooled_result["false_positive_percent"],
                "False_Negative_Percent": pooled_result["false_negative_percent"],
                "Silhouette": pooled_result["silhouette"],
                "Usable_Feature_Count": len(pooled_result["usable_features"]),
                "Usable_Features": " | ".join(pooled_result["usable_features"]),
                "Usable_Features_Display": " | ".join(
                    self._format_kmeans_feature_display_labels(
                        feature_set["plotter"],
                        pooled_result["usable_features"],
                    )
                ),
            })
            self._write_kmeans_projection_scatter_bundle(f"{feature_set['stem']}_All", pooled_result)

            feature_set_df = pd.DataFrame(rows)
            feature_set_df.to_csv(
                self.kmeans_result_dir / f"{feature_set['stem']}_accuracy_by_temperature.csv",
                index=False,
            )
            feature_set_accuracy_rows.extend(rows)

        feature_set_accuracy_df = pd.DataFrame(feature_set_accuracy_rows)
        feature_set_accuracy_df.to_csv(
            self.kmeans_result_dir / "kmeans_feature_set_accuracy_comparison.csv",
            index=False,
        )

        self.manifest["kmeans_temperature_transfer"] = {
            "status": "completed",
            "temperatures_K": temperatures,
            "union_feature_count": len(union_features),
            "union_features": union_features,
            "pooled_best_feature_count": len(pooled_best["usable_features"]),
            "pooled_best_features": pooled_best["usable_features"],
        }

    def _build_kmeans_temperature_contexts(self) -> Dict[float, Dict[str, Any]]:
        """Build the per-temperature KMeans contexts (plotter, data, best feature row) used by the transfer analysis."""
        contexts: Dict[float, Dict[str, Any]] = {}
        for record in self.manifest.get("temperature_mapping", []):
            temperature_K = self.safe_float(record.get("temperature_K"))
            if not math.isfinite(temperature_K):
                continue
            if temperature_K in contexts:
                continue
            summary_file = Path(str(record["summary_file"]))
            summary_root = summary_file.parents[2]

            with redirect_stdout(io.StringIO()):
                plotter = summary_plots(str(summary_root))
            plotter._write_kmeans_preprocessing_report = lambda: None

            raw_df = plotter.df_og.copy()
            if "D_Binary" not in raw_df.columns:
                self.manifest["warnings"].append(
                    f"Skipping {summary_root}: Quant_Data.csv lacks D_Binary for KMeans transfer."
                )
                continue
            labels = pd.to_numeric(raw_df["D_Binary"], errors="coerce")
            usable_raw = raw_df[labels.isin([0, 1])].copy()
            usable_raw = usable_raw[
                ~usable_raw["Small Molecule ID"].astype(str).isin(
                    [self.config.control_id, "DSM", "DSM_AVG", "NDSM", "NDSM_AVG"]
                )
            ].copy()
            class_counts = pd.to_numeric(usable_raw["D_Binary"], errors="coerce").value_counts()
            if usable_raw.empty or len(class_counts) < 2 or int(class_counts.min()) < 2:
                self.manifest["warnings"].append(
                    f"Skipping {summary_root}: insufficient DSM/NDSM rows for temperature-transfer KMeans."
                )
                continue

            with redirect_stdout(io.StringIO()):
                plotter.clean_df()
            df = plotter.df.copy()
            df = df[df["Small Molecule ID"].astype(str) != self.config.control_id].copy()
            df = df[pd.to_numeric(df["D_Binary"], errors="coerce").isin([0, 1])].copy().reset_index(drop=True)
            best_row = self._load_best_kmeans_feature_row(summary_root)
            if best_row is None:
                self.manifest["warnings"].append(
                    f"Skipping {summary_root}: missing usable KMeans_Iteration_Results.csv."
                )
                continue

            contexts[temperature_K] = {
                "temperature_K": temperature_K,
                "summary_root": summary_root,
                "plotter": plotter,
                "df": df,
                "best_features": best_row["features"],
                "best_accuracy_percent": best_row["accuracy_percent"],
                "kmeans_iteration_path": best_row["path"],
            }
        return contexts

    def _load_best_kmeans_feature_row(self, summary_root: Path) -> Optional[Dict[str, Any]]:
        """Load the best feature set/accuracy row from a temperature's KMeans_Iteration_Results.csv."""
        results_path = summary_root / "RESULTS" / "SUMMARY" / "KMeans_Iteration_Results.csv"
        if not results_path.exists():
            return None
        df = pd.read_csv(results_path)
        if df.empty or "Accuracy" not in df.columns or "Usable_Feature_List" not in df.columns:
            return None
        ranked = df.copy()
        ranked["Accuracy"] = pd.to_numeric(ranked["Accuracy"], errors="coerce")
        if "Usable_Features_Count" in ranked.columns:
            ranked["Usable_Features_Count"] = pd.to_numeric(
                ranked["Usable_Features_Count"], errors="coerce"
            )
        else:
            ranked["Usable_Features_Count"] = math.nan
        if "Features_Count" in ranked.columns:
            ranked["Features_Count"] = pd.to_numeric(ranked["Features_Count"], errors="coerce")
        else:
            ranked["Features_Count"] = math.nan
        ranked = ranked.dropna(subset=["Accuracy"])
        if ranked.empty:
            return None
        ranked = ranked.sort_values(
            ["Accuracy", "Usable_Features_Count", "Features_Count"],
            ascending=[False, True, True],
            na_position="last",
        )
        row = ranked.iloc[0]
        features = self._parse_feature_list(row["Usable_Feature_List"])
        if len(features) < 2:
            return None
        return {
            "path": results_path,
            "features": features,
            "accuracy_percent": self.safe_float(row["Accuracy"]),
        }

    @staticmethod
    def _format_kmeans_dataset_label(temperature_K: float) -> str:
        """Format a temperature as a ``"<K> K"`` dataset label for KMeans output tables."""
        return f"{int(temperature_K)} K"

    @staticmethod
    def _format_kmeans_feature_display_labels(
        plotter: summary_plots,
        features: Sequence[str],
    ) -> List[str]:
        """Map raw feature column names to their human-readable plot labels via the plotter."""
        return [plotter.variable_plot_name_map.get(feature, feature) for feature in features]

    @staticmethod
    def _parse_feature_list(text: Any) -> List[str]:
        """Parse a ``"|"``-separated feature-list cell into a list of stripped feature names."""
        if text is None:
            return []
        try:
            if pd.isna(text):
                return []
        except Exception:
            pass
        return [token.strip() for token in str(text).split("|") if token.strip()]

    @staticmethod
    def _ordered_unique(items: Iterable[str]) -> List[str]:
        """Return the unique items preserving first-seen order."""
        out: List[str] = []
        seen: set[str] = set()
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out

    def _build_pooled_kmeans_context(self, contexts: Dict[float, Dict[str, Any]]) -> Dict[str, Any]:
        """Build a pooled KMeans context combining all temperatures for the union/greedy analyses."""
        temperatures = sorted(contexts)
        pooled_frames: List[pd.DataFrame] = []
        for temperature_K in temperatures:
            frame = contexts[temperature_K]["df"].copy()
            frame["Temperature_K"] = temperature_K
            pooled_frames.append(frame)
        pooled_df = pd.concat(pooled_frames, ignore_index=True, sort=False)
        pooled_plotter = contexts[temperatures[0]]["plotter"]
        return {
            "temperature_K": math.nan,
            "summary_root": self.kmeans_result_dir,
            "plotter": pooled_plotter,
            "df": pooled_df,
        }

    def _evaluate_kmeans_feature_set(
        self,
        context: Dict[str, Any],
        feature_list: Sequence[str],
        context_label: str,
    ) -> Dict[str, Any]:
        """Score a feature set on one context: PCA+KMeans accuracy, balanced accuracy, recalls, and silhouette."""
        plotter = context["plotter"]
        df = context["df"].copy().reset_index(drop=True)
        with redirect_stdout(io.StringIO()):
            features, usable_features = plotter._prepare_feature_matrix(
                df,
                list(feature_list),
                context=context_label,
            )

        labels = pd.to_numeric(df["D_Binary"], errors="coerce").astype(int).to_numpy()
        scaled_features = StandardScaler().fit_transform(features)
        pca = PCA(n_components=2)
        pca_coordinates = pca.fit_transform(scaled_features)

        kmeans = KMeans(
            init="random",
            n_clusters=2,
            n_init=20,
            max_iter=400,
            random_state=self.config.kmeans_random_seed,
        )
        raw_predictions = kmeans.fit_predict(pca_coordinates)
        acc_forward = float(np.mean(raw_predictions == labels))
        acc_flipped = float(np.mean((1 - raw_predictions) == labels))
        predictions = 1 - raw_predictions if acc_flipped > acc_forward else raw_predictions

        tp = int(np.sum((predictions == 1) & (labels == 1)))
        tn = int(np.sum((predictions == 0) & (labels == 0)))
        fp = int(np.sum((predictions == 1) & (labels == 0)))
        fn = int(np.sum((predictions == 0) & (labels == 1)))
        n_dsm = max(int(np.sum(labels == 1)), 1)
        n_ndsm = max(int(np.sum(labels == 0)), 1)
        dsm_recall = 100.0 * tp / n_dsm
        ndsm_correct = 100.0 * tn / n_ndsm
        accuracy = 100.0 * float(np.mean(predictions == labels))
        balanced_accuracy = 0.5 * (dsm_recall + ndsm_correct)
        false_positive = 100.0 * fp / n_ndsm
        false_negative = 100.0 * fn / n_dsm
        silhouette = math.nan
        if len(np.unique(raw_predictions)) > 1 and len(raw_predictions) > 2:
            silhouette = float(silhouette_score(scaled_features, raw_predictions))

        predicted_labels = np.where(predictions == 1, "DSM", "NDSM")
        true_labels = np.where(labels == 1, "DSM", "NDSM")
        projection_df = pd.DataFrame(
            pca_coordinates,
            columns=["Principle Component 1", "Principle Component 2"],
        )
        projection_df["Predicted Label"] = predicted_labels
        projection_df["True Label"] = true_labels
        projection_df["Small Molecule ID"] = df["Small Molecule ID"].astype(str).tolist()
        projection_df["Small Molecule Name"] = df["Small Molecule Name"].astype(str).tolist()
        if "Temperature_K" in df.columns:
            projection_df["Temperature_K"] = pd.to_numeric(df["Temperature_K"], errors="coerce")
            projection_df["Point Label"] = projection_df["Small Molecule ID"]
        else:
            projection_df["Temperature_K"] = context.get("temperature_K", math.nan)
            projection_df["Point Label"] = projection_df["Small Molecule ID"]

        loadings = pd.DataFrame(
            pca.components_,
            index=["PC1", "PC2"],
            columns=usable_features,
        )
        return {
            "usable_features": usable_features,
            "accuracy_percent": accuracy,
            "balanced_accuracy_percent": balanced_accuracy,
            "dsm_recall_percent": dsm_recall,
            "ndsm_correct_percent": ndsm_correct,
            "false_positive_percent": false_positive,
            "false_negative_percent": false_negative,
            "silhouette": silhouette,
            "projection": projection_df,
            "loadings": loadings,
        }

    def _run_pooled_kmeans_greedy_search(
        self,
        pooled_context: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Greedy forward feature search on the pooled context, recording accuracy at each step."""
        plotter = pooled_context["plotter"]
        df = pooled_context["df"].copy().reset_index(drop=True)
        candidate_features = [
            column for column in plotter._get_kmeans_feature_candidates()
            if column in df.columns
        ]
        candidate_df = df.loc[:, candidate_features].apply(pd.to_numeric, errors="coerce")
        candidate_df, _ = plotter._drop_incomplete_kmeans_features(df, candidate_df)
        labels = pd.to_numeric(df["D_Binary"], errors="coerce").astype(int).to_numpy()

        correlations: List[Tuple[str, float, float]] = []
        for feature in candidate_df.columns:
            series = candidate_df[feature]
            std = series.std(skipna=True)
            if np.isnan(std) or std == 0:
                continue
            corr = float(np.corrcoef(series.values, labels)[0, 1])
            if not math.isfinite(corr):
                corr = 0.0
            correlations.append((feature, abs(corr), corr))
        correlations.sort(key=lambda item: item[1], reverse=True)
        if len(correlations) < 2:
            raise ValueError("Not enough pooled KMeans features remain after strict value/SIG filtering.")

        iteration_rows: List[Dict[str, Any]] = []
        best_result: Optional[Dict[str, Any]] = None
        current_features: List[str] = []
        for feature, abs_corr, raw_corr in correlations:
            current_features.append(feature)
            if len(current_features) < 2:
                continue
            try:
                result = self._evaluate_kmeans_feature_set(
                    pooled_context,
                    current_features,
                    context_label=f"pooled_top_{len(current_features)}",
                )
            except ValueError:
                continue
            iteration_rows.append({
                "Requested_Feature_Count": len(current_features),
                "Usable_Feature_Count": len(result["usable_features"]),
                "Added_Feature": feature,
                "Added_Feature_Abs_Correlation": abs_corr,
                "Added_Feature_Raw_Correlation": raw_corr,
                "Accuracy_Percent": result["accuracy_percent"],
                "Balanced_Accuracy_Percent": result["balanced_accuracy_percent"],
                "DSM_Recall_Percent": result["dsm_recall_percent"],
                "NDSM_Correct_Percent": result["ndsm_correct_percent"],
                "False_Positive_Percent": result["false_positive_percent"],
                "False_Negative_Percent": result["false_negative_percent"],
                "Silhouette": result["silhouette"],
                "Requested_Features": " | ".join(current_features),
                "Usable_Features": " | ".join(result["usable_features"]),
            })
            if best_result is None:
                best_result = result
                continue
            if result["accuracy_percent"] > best_result["accuracy_percent"]:
                best_result = result
                continue
            if (
                math.isclose(result["accuracy_percent"], best_result["accuracy_percent"], rel_tol=0.0, abs_tol=1e-12)
                and len(result["usable_features"]) < len(best_result["usable_features"])
            ):
                best_result = result

        if best_result is None:
            raise ValueError("Pooled KMeans greedy search failed to identify any usable feature subset.")
        return pd.DataFrame(iteration_rows), best_result

    def _write_kmeans_projection_bundle(
        self,
        stem: str,
        result: Dict[str, Any],
        plotter: summary_plots,
    ) -> None:
        """Write a KMeans result's projection and loadings CSVs for one context."""
        projection_path = self.kmeans_result_dir / f"{stem}_projection.csv"
        loadings_path = self.kmeans_result_dir / f"{stem}_loadings.csv"
        summary_path = self.kmeans_result_dir / f"{stem}_summary.csv"

        result["projection"].to_csv(projection_path, index=False)
        result["loadings"].to_csv(loadings_path)
        pd.DataFrame([{
            "Accuracy_Percent": result["accuracy_percent"],
            "Balanced_Accuracy_Percent": result["balanced_accuracy_percent"],
            "DSM_Recall_Percent": result["dsm_recall_percent"],
            "NDSM_Correct_Percent": result["ndsm_correct_percent"],
            "False_Positive_Percent": result["false_positive_percent"],
            "False_Negative_Percent": result["false_negative_percent"],
            "Silhouette": result["silhouette"],
            "Feature_Count": len(result["usable_features"]),
            "Features": " | ".join(result["usable_features"]),
        }]).to_csv(summary_path, index=False)

    def _write_kmeans_projection_scatter_bundle(
        self,
        stem: str,
        result: Dict[str, Any],
    ) -> None:
        """Write a KMeans result's projection/loadings CSVs for one context."""
        result["projection"].to_csv(self.kmeans_result_dir / f"{stem}_projection.csv", index=False)
        pd.DataFrame([{
            "Accuracy_Percent": result["accuracy_percent"],
            "Balanced_Accuracy_Percent": result["balanced_accuracy_percent"],
            "DSM_Recall_Percent": result["dsm_recall_percent"],
            "NDSM_Correct_Percent": result["ndsm_correct_percent"],
            "False_Positive_Percent": result["false_positive_percent"],
            "False_Negative_Percent": result["false_negative_percent"],
            "Silhouette": result["silhouette"],
            "Feature_Count": len(result["usable_features"]),
            "Features": " | ".join(result["usable_features"]),
        }]).to_csv(self.kmeans_result_dir / f"{stem}_summary.csv", index=False)

    @staticmethod
    def _apply_phase_plot_style(font_size: int = 10) -> None:
        """Set the shared seaborn/matplotlib rc style for the phase-diagram figures."""
        sns.set_theme(style="ticks")
        sns.set_style("white")
        plt.rc("axes", titlesize=10)
        plt.rc("axes", labelsize=10)
        plt.rc("xtick", labelsize=font_size)
        plt.rc("ytick", labelsize=font_size)
        plt.rc("legend", fontsize=8)
        plt.rc("font", size=font_size)
        plt.rc("axes", linewidth=2)

    @staticmethod
    def _style_phase_axes(ax: plt.Axes) -> None:
        """Apply the house phase-diagram axes styling (inward ticks, line widths) to an axis."""
        ax.tick_params(
            left=True,
            right=True,
            top=True,
            bottom=True,
            labelbottom=True,
            direction="in",
            length=4,
            width=2,
        )
        xticks = np.asarray(ax.get_xticks(), dtype=float)
        yticks = np.asarray(ax.get_yticks(), dtype=float)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xticks = xticks[np.isfinite(xticks) & (xticks >= min(xlim)) & (xticks <= max(xlim))]
        yticks = yticks[np.isfinite(yticks) & (yticks >= min(ylim)) & (yticks <= max(ylim))]
        if xticks.size >= 2:
            ax.set_xlim(float(xticks[0]), float(xticks[-1]))
        if yticks.size >= 2:
            ax.set_ylim(float(yticks[0]), float(yticks[-1]))

    @staticmethod
    def _draw_phase_xerr(
        ax: plt.Axes,
        x: float,
        y: float,
        err: float,
        *,
        color: str,
        linewidth: float = 1.0,
        alpha: float = 1.0,
        zorder: float = 3.0,
    ) -> None:
        """RDP-style capped horizontal error bar: capsize=2, elinewidth=1.0,
        capthick=1.0 (matches BIOPOLYMER_ANALYSIS.RDP_ERRORBAR_WIDTH=1.0).
        """
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(err) and err > 0):
            return
        ax.errorbar(
            x,
            y,
            xerr=err,
            fmt="none",
            ecolor=color,
            elinewidth=linewidth,
            capsize=2,
            capthick=linewidth,
            alpha=alpha,
            zorder=zorder,
            clip_on=False,
        )

    @staticmethod
    def _style_projected_binodal_axes(ax: plt.Axes) -> None:
        """Apply the fixed axis limits/ticks for the projected-binodal (concentration vs temperature) panel."""
        ax.set_xlim(0.0, 600.0)
        ax.set_xticks([0.0, 150.0, 300.0, 450.0, 600.0])
        ax.set_ylim(280.0, 420.0)
        ax.set_yticks([280.0, 300.0, 320.0, 340.0, 360.0, 380.0, 400.0, 420.0])
        ax.set_box_aspect(1.0)

    def plot_perturbation_binodal(
        self,
        df: pd.DataFrame,
        control_bundle: Dict[str, Any],
        control_curve_df: pd.DataFrame,
        mode: str,
    ) -> None:
        """Render one paper perturbation-binodal figure (mode="averages" or "all")."""
        if mode not in {"averages", "all"}:
            raise ValueError(f"Unknown perturbation plot mode: {mode}")
        self._apply_phase_plot_style(font_size=10)
        fig = plt.figure(figsize=RDP_STYLE_FIGSIZE)
        ax = fig.add_axes(RDP_STYLE_AX_RECT)
        ind_recs, avg_recs = self._render_unified_binodal_panel(
            ax, df, control_bundle, control_curve_df,
            mode=mode, with_errorbars=False,
        )

        stem = f"perturbation_projected_binodal_{mode}"
        aliases: List[str] = [stem]
        if mode == "all":
            aliases.append("perturbation_projected_binodal")
        if mode == "averages":
            aliases.append("sg_projected_binodal_with_averages")
        self.save_figure(fig, aliases)

        if avg_recs:
            pd.DataFrame(avg_recs).to_csv(
                self.result_dir / "perturbation_average_binodal_fit.csv", index=False)
        if ind_recs:
            pd.DataFrame(ind_recs).to_csv(
                self.result_dir / "perturbation_individual_binodal_fit.csv", index=False)

    def _scatter_binodal_pair(
        self,
        ax: plt.Axes,
        dense_x: float,
        dilute_x: float,
        temp_y: float,
        *,
        color: str,
        size: float,
        linewidth: float = 0.5,
        zorder: float = 4.0,
    ) -> None:
        """Scatter a dense/dilute concentration pair at one temperature on the unified binodal panel."""
        if not math.isfinite(temp_y):
            return
        xs: List[float] = []
        if math.isfinite(dense_x):
            xs.append(dense_x)
        if math.isfinite(dilute_x):
            xs.append(dilute_x)
        if not xs:
            return
        ax.scatter(
            xs, [temp_y] * len(xs),
            s=size, c=color, marker="o",
            edgecolors="black", linewidths=linewidth,
            zorder=zorder, clip_on=False,
        )

    def _render_unified_binodal_panel(
        self,
        ax: plt.Axes,
        df: pd.DataFrame,
        control_bundle: Dict[str, Any],
        control_curve_df: pd.DataFrame,
        *,
        mode: str,
        with_errorbars: bool,
        fit_method: str = "diameter_slope_constrained",
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Canonical projected-binodal panel renderer.

        mode:
          - "control_only"       : SG control only.
          - "averages"           : SG control + DSM_AVE + NDSM_AVE.
          - "individuals"        : SG control + 10 DSM compounds + 10 NDSM
                                   compounds (per-compound binodal fits).
          - "all"                : SG control + 20 individual + 2 average.

        Returns (individual_records, average_records) so callers can save
        accompanying CSVs.
        """
        Tc_ctrl = float(control_bundle["default"].parameters["Tc_app_control"])
        T_obs_min = float(df["temperature_K"].min()) if "temperature_K" in df else 280.0

        # mode → translate to fine-grained booleans for SG / per-class avgs / per-class individuals
        if isinstance(mode, dict):
            show_sg = bool(mode.get("show_sg", True))
            show_dsm_avg = bool(mode.get("show_dsm_avg", False))
            show_dsm_ind = bool(mode.get("show_dsm_ind", False))
            show_ndsm_avg = bool(mode.get("show_ndsm_avg", False))
            show_ndsm_ind = bool(mode.get("show_ndsm_ind", False))
            errorbars_avg_override = mode.get("errorbars_avg", None)
            errorbars_ind_override = mode.get("errorbars_ind", None)
        else:
            show_sg = True
            show_dsm_avg = mode in {"averages", "all"}
            show_dsm_ind = mode in {"individuals", "all"}
            show_ndsm_avg = mode in {"averages", "all"}
            show_ndsm_ind = mode in {"individuals", "all"}
            errorbars_avg_override = None
            errorbars_ind_override = None

        # Automatic error-bar logic:
        #   - Averages get error bars when shown.
        #   - Individuals get error bars only when NO averages are shown
        #     (individual-only panels). Otherwise individuals stay clean
        #     so the averaged signal isn't visually drowned out.
        any_avg_shown = show_dsm_avg or show_ndsm_avg
        any_ind_shown = show_dsm_ind or show_ndsm_ind
        errorbars_avg = with_errorbars if errorbars_avg_override is None else errorbars_avg_override
        if any_avg_shown and errorbars_avg_override is None:
            errorbars_avg = True
        errorbars_ind = (not any_avg_shown) and any_ind_shown if errorbars_ind_override is None else errorbars_ind_override

        # --- Control SG binodal: single solid grey closed curve ---
        if show_sg:
            control_curve = control_curve_df[control_curve_df["fit_kind"] == "default"]
            if not control_curve.empty:
                xs, ys = self._closed_binodal_xy(control_curve)
                ax.plot(xs, ys, color=PHASE_COLOR_SG, lw=1.8, zorder=2.0)
            ax.axhline(Tc_ctrl, color=PHASE_COLOR_SG, ls="--", lw=1.0, zorder=1.5)

        # For per-section conditionals below
        show_individuals = show_dsm_ind or show_ndsm_ind
        show_averages = show_dsm_avg or show_ndsm_avg

        pert = df[~df["is_control"]].copy() if (show_individuals or show_averages) else df.iloc[0:0]
        individuals = pert[pert.get("row_semantic_type", "") == "individual_compound"]
        averages = pert[pert.get("row_semantic_type", "") == "arithmetic_class_average"]

        individual_records: List[Dict[str, Any]] = []
        avg_fit_records: List[Dict[str, Any]] = []

        # --- Per-compound (individual) binodal fits ---
        if show_individuals:
            for condition_name, grp in individuals.groupby("condition_name"):
                ctype = self.safe_str(grp["condition_type"].iloc[0])
                if ctype == "DSM" and not show_dsm_ind:
                    continue
                if ctype == "NDSM" and not show_ndsm_ind:
                    continue
                color = PHASE_COLOR_DSM if ctype == "DSM" else PHASE_COLOR_NDSM
                fit = self.fit_individual_binodal(
                    grp, str(condition_name), control_fit=control_bundle.get("default"),
                    fit_method=fit_method)
                if fit is None:
                    continue
                Tc_ind = float(fit.parameters.get("Tc_app_control", math.nan))
                fit_for_curve = fit_result(
                    success=fit.success, fit_kind=fit.fit_kind, loss=fit.loss,
                    used_fallback=fit.used_fallback,
                    parameters={
                        "Tc_app_pert_avg": fit.parameters.get("Tc_app_control"),
                        "c_c_pert_avg": fit.parameters.get("c_c_control"),
                        "A_pert_avg": fit.parameters.get("A_control"),
                        "B_pert_avg": fit.parameters.get("B_control"),
                    },
                    rmse=fit.rmse, n_points=fit.n_points, n_residuals=fit.n_residuals,
                    message=fit.message,
                    eligible_temperatures_K=fit.eligible_temperatures_K,
                    excluded_temperatures_K=fit.excluded_temperatures_K,
                    include_flagged_points=fit.include_flagged_points,
                    residual_scale_mode=fit.residual_scale_mode,
                    warnings=fit.warnings,
                )
                curve = self.build_average_curve(fit_for_curve, T_obs_min, Tc_ctrl)
                if curve.empty:
                    continue
                xs, ys = self._closed_binodal_xy(curve)
                ax.plot(xs, ys, color=color, lw=0.7, alpha=0.55, zorder=2.2)
                individual_records.append({
                    "condition_name": condition_name,
                    "condition_type": ctype,
                    "Tc_app_pert_ind": Tc_ind,
                    "c_c_pert_ind": float(fit.parameters.get("c_c_control", math.nan)),
                    "A_pert_ind": float(fit.parameters.get("A_control", math.nan)),
                    "B_pert_ind": float(fit.parameters.get("B_control", math.nan)),
                })

        # --- Class-average binodal fits ---
        if show_averages:
            for condition_type, color in [("DSM", PHASE_COLOR_DSM), ("NDSM", PHASE_COLOR_NDSM)]:
                if condition_type == "DSM" and not show_dsm_avg:
                    continue
                if condition_type == "NDSM" and not show_ndsm_avg:
                    continue
                avg_fit = self.fit_average_binodal(
                    df, condition_type, control_fit=control_bundle.get("default"),
                    fit_method=fit_method)
                if avg_fit is None:
                    continue
                curve = self.build_average_curve(avg_fit, T_obs_min, Tc_ctrl)
                if curve.empty:
                    continue
                xs, ys = self._closed_binodal_xy(curve)
                ax.plot(xs, ys, color=color, lw=1.8, zorder=2.6)
                Tc_avg = float(avg_fit.parameters["Tc_app_pert_avg"])
                ax.axhline(Tc_avg, color=color, ls="--", lw=1.0, zorder=1.6)
                avg_fit_records.append({"condition_type": condition_type, **avg_fit.parameters})

        # --- Control SG scatter ---
        if show_sg:
            ctrl = df[df["is_control"]].sort_values("temperature_K")
            for _, row in ctrl.iterrows():
                d = self.safe_float(row.get("c_sg_observed", math.nan))
                l = self.safe_float(row.get("c_dil_inf_corrected", math.nan))
                t = self.safe_float(row.get("temperature_K", math.nan))
                if errorbars_avg:
                    self._draw_phase_xerr(ax, d, t,
                                          self.safe_float(row.get("c_sg_std", math.nan)),
                                          color=PHASE_COLOR_SG, zorder=3.8)
                    self._draw_phase_xerr(ax, l, t,
                                          self.safe_float(row.get("c_dil_inf_std", math.nan)),
                                          color=PHASE_COLOR_SG, zorder=3.8)
                self._scatter_binodal_pair(ax, d, l, t,
                                           color=PHASE_COLOR_SG,
                                           size=PHASE_MARKER_SIZE_SG,
                                           linewidth=0.5, zorder=4.0)

        # --- Individual scatter (per-class boolean) ---
        if show_individuals:
            for _, row in individuals.iterrows():
                ctype = self.safe_str(row.get("condition_type", ""))
                if ctype == "DSM" and not show_dsm_ind:
                    continue
                if ctype == "NDSM" and not show_ndsm_ind:
                    continue
                color = PHASE_COLOR_DSM if ctype == "DSM" else PHASE_COLOR_NDSM
                d = self.safe_float(row.get("c_sg_observed", math.nan))
                l = self.safe_float(row.get("c_dil_inf_corrected", math.nan))
                t = self.safe_float(row.get("temperature_K", math.nan))
                if errorbars_ind:
                    self._draw_phase_xerr(ax, d, t,
                                          self.safe_float(row.get("c_sg_std", math.nan)),
                                          color=color, zorder=3.6)
                    self._draw_phase_xerr(ax, l, t,
                                          self.safe_float(row.get("c_dil_inf_std", math.nan)),
                                          color=color, zorder=3.6)
                self._scatter_binodal_pair(ax, d, l, t, color=color,
                                           size=PHASE_MARKER_SIZE_INDIVIDUAL,
                                           linewidth=0.3, zorder=3.5)

        # --- Average scatter (per-class boolean; error bars if averages-mode) ---
        if show_averages:
            for _, row in averages.iterrows():
                ctype = self.safe_str(row.get("condition_type", ""))
                if ctype == "DSM" and not show_dsm_avg:
                    continue
                if ctype == "NDSM" and not show_ndsm_avg:
                    continue
                color = PHASE_COLOR_DSM if ctype == "DSM" else PHASE_COLOR_NDSM
                d = self.safe_float(row.get("c_sg_observed", math.nan))
                l = self.safe_float(row.get("c_dil_inf_corrected", math.nan))
                t = self.safe_float(row.get("temperature_K", math.nan))
                if errorbars_avg:
                    self._draw_phase_xerr(ax, d, t,
                                          self.safe_float(row.get("c_sg_std", math.nan)),
                                          color=color, zorder=4.6)
                    self._draw_phase_xerr(ax, l, t,
                                          self.safe_float(row.get("c_dil_inf_std", math.nan)),
                                          color=color, zorder=4.6)
                self._scatter_binodal_pair(ax, d, l, t, color=color,
                                           size=PHASE_MARKER_SIZE_AVG,
                                           linewidth=0.5, zorder=4.7)

        # Axis labels intentionally left blank; only ticks + axis frame are kept.
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        self._style_phase_axes(ax)
        self._style_projected_binodal_axes(ax)

        return individual_records, avg_fit_records

    @staticmethod
    def _closed_binodal_xy(curve: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Concatenate dilute (ascending T) + dense (descending T) branches
        into a single closed-bell curve traced as one solid line.
        """
        ordered = curve.sort_values("temperature_K")
        T_arr = ordered["temperature_K"].to_numpy(dtype=float)
        if "c_dense_fit" in ordered.columns:
            c_dense = ordered["c_dense_fit"].to_numpy(dtype=float)
            c_dil = ordered["c_dilute_fit"].to_numpy(dtype=float)
        else:
            c_dense = ordered["c_sg_control_fit"].to_numpy(dtype=float)
            c_dil = ordered["c_dil_control_fit"].to_numpy(dtype=float)
        xs = np.concatenate([c_dil, c_dense[::-1]])
        ys = np.concatenate([T_arr, T_arr[::-1]])
        return xs, ys

    def save_figure(self, fig: plt.Figure, stem: str | Sequence[str]) -> None:
        """Save a figure to the phase-diagram figure directory under one or more stems and close it."""
        stems = [stem] if isinstance(stem, str) else list(stem)
        for item in stems:
            fig.savefig(self.figure_dir / f"{item}.png", dpi=400)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def safe_float(value: Any) -> float:
        """Coerce a value to float, returning NaN on failure."""
        try:
            if pd.isna(value):
                return math.nan
        except Exception:
            pass
        try:
            return float(value)
        except Exception:
            return math.nan

    @staticmethod
    def safe_str(value: Any) -> str:
        """Coerce a value to a stripped string, returning '' for None/NaN."""
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass
        return str(value)

    @staticmethod
    def _mode_or_first(series: pd.Series) -> str:
        """Return the most common string in a series, falling back to the first non-empty value."""
        vals = [str(v) for v in series.dropna().tolist() if str(v)]
        if not vals:
            return ""
        mode = pd.Series(vals).mode()
        return str(mode.iloc[0]) if not mode.empty else vals[0]

    @staticmethod
    def _mode_or_mixed(series: pd.Series) -> str:
        """Return the single unique string in a series, or 'mixed' when several distinct values are present."""
        vals = sorted({str(v) for v in series.dropna().tolist() if str(v)})
        if not vals:
            return ""
        if len(vals) == 1:
            return vals[0]
        return "mixed"

    @staticmethod
    def _join_unique(series: pd.Series) -> str:
        """Return the sorted unique non-empty strings of a series joined by a separator."""
        vals = sorted({str(v) for v in series.dropna().tolist() if str(v)})
        return ";".join(vals)

    @staticmethod
    def _jsonable(obj: Any) -> Any:
        """Recursively convert an object (dicts, Paths, numpy scalars, etc.) into JSON-serializable form."""
        if isinstance(obj, dict):
            return {str(k): phase_diagram._jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [phase_diagram._jsonable(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            value = obj.item()
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return value
        if isinstance(obj, float):
            return obj if math.isfinite(obj) else None
        if isinstance(obj, Path):
            return str(obj)
        return obj


def parse_scale_list(text: Optional[str]) -> List[float]:
    """Parse a comma-separated string of Meff_SG scale factors into a list of floats."""
    if not text:
        return []
    out = []
    for token in text.split(","):
        token = token.strip()
        if token:
            out.append(float(token))
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser defining every phase_diagram.py CLI option."""
    parser = argparse.ArgumentParser(description="Projected SG biopolymer phase-diagram analysis")
    parser.add_argument("--summary-glob", nargs="+", default=None, help="Glob or explicit list of Quant_Data.csv summary files")
    parser.add_argument("--input-csv", default=None, help="Optional direct input CSV (Quant_Data-like or tidy)")
    parser.add_argument("--output-dir", required=True, help="Output directory root")
    parser.add_argument("--meff-sg", default="auto", help="Meff_SG in g/mol, or 'auto'")
    parser.add_argument("--cluster-file-for-meff", default=None, help="Optional specific Cluster_SG_sg_X.csv for Meff inference")
    parser.add_argument("--meff-sg-scale-sensitivity", default="", help="Comma-separated Meff scale factors for sensitivity reruns")
    parser.add_argument("--beta-critical", type=float, default=0.325, help="Fixed apparent critical exponent")
    parser.add_argument("--fit-perturbed-tc", action="store_true", help="Attempt constrained qualitative apparent-Tc fits for perturbations")
    parser.add_argument("--include-raw-dilute", action="store_true", help="Overlay raw dilute branch on control binodal plot")
    parser.add_argument("--interface-threshold", type=float, default=DEFAULT_INTERFACE_THRESHOLD, help="W/R threshold for broad-interface exclusion")
    parser.add_argument("--curvature-threshold", type=float, default=0.1, help="Curvature exponent diagnostic threshold")
    parser.add_argument("--include-flagged-fit-sensitivity", action="store_true", help="Also attempt a control fit including flagged broad-interface points")
    parser.add_argument("--control-reference-sensitivity", action="store_true", help="Also record sensitivity metrics against the fitted control curve")
    parser.add_argument("--delta-c-ctrl-epsilon", type=float, default=1e-3, help="Stability threshold for relative metrics when control Delta_c is small")
    parser.add_argument("--robust-fit-sensitivity", action="store_true", help="Also run a robust-loss control-fit sensitivity analysis")
    parser.add_argument("--conc-source", choices=["fit", "calc"], default="calc", help="Use fit or calc SG concentration branch")
    parser.add_argument("--temperature-from", choices=["path", "column"], default="path", help="How to infer temperature")
    parser.add_argument("--temperature-column", default=None, help="Temperature column to use in tidy mode")
    parser.add_argument("--control-id", default=DEFAULT_CONTROL_ID, help="Control SG row identifier")
    parser.add_argument("--include-category-aggregate-rows", action="store_true", help="Include DSM/NDSM category aggregate rows in processed outputs")
    parser.add_argument("--rel-floor", type=float, default=DEFAULT_REL_FLOOR, help="Relative floor used in branch-balanced residual scaling")
    parser.add_argument("--monotonicity-sigma-factor", type=float, default=DEFAULT_MONO_SIGMA_FACTOR, help="Tolerance factor for monotonicity diagnostics")
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES, help="Number of control-fit resamples")
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed for resampling")
    parser.add_argument("--curvature-mc-samples", type=int, default=DEFAULT_CURVATURE_MC_SAMPLES, help="Monte Carlo samples for nonlinear Gibbs-Thomson uncertainty propagation")
    parser.add_argument("--min-perturbed-tc-temperatures", type=int, default=DEFAULT_MIN_PERTURBED_TC_TEMPS, help="Minimum number of temperatures required before fitting a perturbation apparent Tc")
    parser.add_argument("--meff-tolerance-rel", type=float, default=DEFAULT_MEFF_TOL_REL, help="Relative tolerance for multi-root Meff consistency check")
    parser.add_argument("--c-dil-negative-floor", type=float, default=DEFAULT_DILUTE_NEGATIVE_FLOOR, help="Hard-invalid floor for negative dilute concentration")
    parser.add_argument("--robust-loss", choices=["huber", "cauchy", "soft_l1"], default="huber", help="Robust loss for sensitivity fit")
    parser.add_argument("--robust-f-scale", type=float, default=1.0, help="f_scale parameter for robust least_squares sensitivity fit")
    parser.add_argument("--plot-title-prefix", default="", help="Optional prefix for figure titles")
    parser.add_argument(
        "--skip-kmeans-temperature-transfer",
        action="store_true",
        help="Skip the cross-temperature KMeans transfer, union-feature, and pooled-feature analyses.",
    )
    parser.add_argument(
        "--kmeans-random-seed",
        type=int,
        default=42,
        help="Random seed used for cross-temperature KMeans evaluations inside the phase-diagram workflow.",
    )
    return parser


def main() -> None:
    """CLI entry point: parse arguments into a phase_diagram_config and run phase_diagram."""
    args = build_arg_parser().parse_args()
    config = phase_diagram_config(
        summary_glob=args.summary_glob,
        input_csv=Path(args.input_csv) if args.input_csv else None,
        output_dir=Path(args.output_dir),
        meff_sg=str(args.meff_sg),
        cluster_file_for_meff=Path(args.cluster_file_for_meff) if args.cluster_file_for_meff else None,
        meff_sg_scale_sensitivity=parse_scale_list(args.meff_sg_scale_sensitivity),
        beta_critical=args.beta_critical,
        fit_perturbed_tc=args.fit_perturbed_tc,
        include_raw_dilute=args.include_raw_dilute,
        interface_threshold=args.interface_threshold,
        curvature_threshold=args.curvature_threshold,
        include_flagged_fit_sensitivity=args.include_flagged_fit_sensitivity,
        control_reference_sensitivity=args.control_reference_sensitivity,
        delta_c_ctrl_epsilon=args.delta_c_ctrl_epsilon,
        robust_fit_sensitivity=args.robust_fit_sensitivity,
        conc_source=args.conc_source,
        temperature_from=args.temperature_from,
        temperature_column=args.temperature_column,
        control_id=args.control_id,
        include_category_aggregate_rows=args.include_category_aggregate_rows,
        rel_floor=args.rel_floor,
        monotonicity_sigma_factor=args.monotonicity_sigma_factor,
        bootstrap_samples=args.bootstrap_samples,
        random_seed=args.random_seed,
        curvature_mc_samples=args.curvature_mc_samples,
        min_perturbed_tc_temperatures=args.min_perturbed_tc_temperatures,
        meff_tolerance_rel=args.meff_tolerance_rel,
        c_dil_negative_floor=args.c_dil_negative_floor,
        robust_loss=args.robust_loss,
        robust_f_scale=args.robust_f_scale,
        plot_title_prefix=args.plot_title_prefix,
        run_kmeans_temperature_transfer=not args.skip_kmeans_temperature_transfer,
        kmeans_random_seed=args.kmeans_random_seed,
    )
    analysis = phase_diagram(config)
    analysis.run()


if __name__ == "__main__":
    main()
