"""
PCA + KMeans clustering and observable plots (pipeline Step 7).

Final stage of the stress-granule MD analysis pipeline. Reads the per-temperature
master results table and uses it to (a) draw per-observable comparison figures
(violin plots of DSM vs NDSM with the SG control as a reference line) and (b) run
an unsupervised PCA -> KMeans classifier that separates dissolving (DSM) from
non-dissolving (NDSM) small molecules. An automated, correlation-ranked greedy
feature search identifies the observable subset that best separates the two
classes; only physics-validated condensate-aggregate observables are eligible
(MSD-based/cage/diagnostic transport metrics and per-species observables are
excluded). The published Fig 4 classifier is the fixed two-feature [P_SM, N_D]
model evaluated elsewhere; this script produces the exploratory clustering,
loadings, and scatter/violin panels.

Pipeline role:
    Runs after system_analysis.py (which writes Quant_Data.csv).

Key inputs:
    - {path}/{folder}_{T}_{dt}_{tmin}_{tmax}/RESULTS/SUMMARY/Quant_Data.csv
    - parameters.csv (only for the optional k_means_parameters route)
    - CLI flags: --path, --folder, --T, --dt, --tmin, --tmax, [--plot-only]

Key outputs (under the analysis root):
    - FIGURES/PROPERTIES/*_VP.png         per-observable violin plots
    - FIGURES/PROPERTIES/KMeans*.png      PCA scatter, loadings, annotated scatter
    - RESULTS/SUMMARY/Feature_Correlations.csv,
      KMeans_Feature_Selection_Ranking.csv, KMeans_Single_Feature_Results.csv,
      KMeans_Iteration_Results.csv, KMeans_Preprocessing_Report.csv
    - dsm_predicted{suffix}.txt / ndsm_predicted{suffix}.txt predicted class lists
    - RESULTS/SUMMARY/KMeans_Skipped.txt  (when <2 labeled classes are present)

CLI:
    python kmeans.py --path TEMP_300 --folder CLASSIFY --T 300 \
        --tmin 50 --dt 50 --tmax 2000 [--plot-only]
"""

import sys
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Publication layout constants. Axes sizes match RDP / contact map panels.
# Each fig has symmetric L/R margins so the axes spine is horizontally centered.
_VP_FIG_SIZE = (2.20, 2.20)
_VP_AX_RECT = [0.40 / 2.20, 0.30 / 2.20, 1.40 / 2.20, 1.40 / 2.20]   # ax 1.4 x 1.4 in
_PCA_BAR_FIG_SIZE = (3.50, 3.20)
_PCA_BAR_AX_RECT = [0.55 / 3.50, 0.65 / 3.20, 2.40 / 3.50, 2.40 / 3.20]   # ax 2.4 x 2.4 in
_SCATTER_FIG_SIZE = (3.50, 3.20)
_SCATTER_AX_RECT = [0.55 / 3.50, 0.45 / 3.20, 2.40 / 3.50, 2.40 / 3.20]   # ax 2.4 x 2.4 in (square, matches PCA bars)


def _make_fixed_axes(figsize, rect):
    """Create a figure with a single fixed-size axes placed at *rect* (axes fraction)."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(rect)
    return fig, ax


def _apply_2sig_yticks(ax, ymin, ymax, n_ticks=6, annotate_exp=True):
    """6 y-ticks with 2-sig-fig floor/ceil bounds and shared ×10^n exponent."""
    tick_lo = _round_2sig_floor(ymin)
    tick_hi = _round_2sig_ceil(ymax)
    if abs(tick_hi - tick_lo) < 1e-30:
        tick_hi = tick_lo + 1.0
    ticks = np.linspace(tick_lo, tick_hi, n_ticks)
    ax.set_ylim(tick_lo, tick_hi)
    ax.set_yticks(ticks)
    max_abs_tick = max(abs(tick_lo), abs(tick_hi))
    if max_abs_tick > 0:
        common_exp = int(np.floor(np.log10(max_abs_tick)))
    else:
        common_exp = 0
    scale = 10.0 ** common_exp
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda val, pos, s=scale: f"{val / s:.1f}")
    )
    if annotate_exp:
        ax.annotate(f"$\\times10^{{{common_exp}}}$", xy=(0, 1), xycoords='axes fraction',
                    ha='left', va='bottom', fontsize=8)


def _round_2sig_floor(x):
    """Round x down (toward -inf) to 2 significant figures (0 maps to 0.0)."""
    if x == 0:
        return 0.0
    exp = int(np.floor(np.log10(abs(x))))
    factor = 10.0 ** (exp - 1)
    return np.floor(x / factor) * factor


def _round_2sig_ceil(x):
    """Round x up (toward +inf) to 2 significant figures (0 maps to 0.0)."""
    if x == 0:
        return 0.0
    exp = int(np.floor(np.log10(abs(x))))
    factor = 10.0 ** (exp - 1)
    return np.ceil(x / factor) * factor


class summary_plots():
    """Load Quant_Data.csv and generate observable comparison figures + PCA/KMeans classification.

    Discovers the numeric observable columns in the per-temperature master table,
    maps them to plot-friendly names, and exposes the violin-plot, bar-plot, and
    PCA -> KMeans clustering routines used to compare DSM vs NDSM small molecules
    against the SG control. Includes a correlation-ranked greedy feature search
    (automated_feature_search) restricted to physics-validated condensate
    observables. Constructed with the analysis-root path containing
    RESULTS/SUMMARY/Quant_Data.csv.
    """

    def __init__(self, path):
        """Load Quant_Data.csv from *path* and set up the observable column maps.

        Args:
            path: Analysis root containing ``RESULTS/SUMMARY/Quant_Data.csv``.

        Discovers the numeric observable columns, builds the variable -> plot-name
        map (preferring legacy names), and records the KMeans-excluded set.
        """
        self.path = path
        self.kmeans_preprocessing_records = []
        self.df_og = pd.read_csv("{}/RESULTS/SUMMARY/Quant_Data.csv".format(path), delimiter=",")
        self._resolved_column_cache = {}
        legacy_variables = [
                     "$Mass$ $(Da)$",
                     "$c_{dense,SG,fit}$ $(mg/ml)$",
                     "$c_{dilute,SG,fit}$ $(mg/ml)$",
                     "$c_{dense,SG,calc}$ $(mg/ml)$",
                     "$c_{dilute,SG,calc}$ $(mg/ml)$",
                     "$P_{SG}$",
                     "$R_{cond}$ $(\AA)$",
                     "$W_{interface}$ $(\AA)$",
                     "$\gamma_{1}$ $(mN/m)$",
                     "$\gamma_{2}$ $(mN/m)$",
                     "$\gamma_ave$ $(mN/m)$",
                     "$\Delta G_{trans}$ $(kJ/mol)$",
                     "$c_{dilute,SM}$ $(mg/ml)$",
                     "$c_{dense,SM}$ $(mg/ml)$",
                     "$P_{SM}$",
                     "$\phi_{D}$",
                     "$N_{D}$",
                     "$R_{g}$",
                     "$\phi_{R}$",
                     "$\eta_{GK}$ Pa s",
                     "$D_{SE,GK,Rh}$ $\mu m^{2} / s$",
                     "$D_{SE,GK,Rg}$ $\mu m^{2} / s$",
                     "$l_{conf}$ A",
                     "$\\tau_{cage}$ $ns$",
                     "$D_{cage}$ $\mu m^{2} / s$",
                     "$D_{loglog}$ $\mu m^{2} / s$",
                     "$\gamma_{1,Protein}$ $(mN/m)$",
                     "$\gamma_{2,Protein}$ $(mN/m)$",
                     "$\gamma_{ave,Protein}$ $(mN/m)$",
                     "$\gamma_{1,RNA}$ $(mN/m)$",
                     "$\gamma_{2,RNA}$ $(mN/m)$",
                     "$\gamma_{ave,RNA}$ $(mN/m)$",
                     "$\\eta_{GK Theo}$ Pa s",
                     "$l_{conf,G3BP1}$ A",
                     "$l_{conf,PABP1}$ A",
                     "$l_{conf,TIA1}$ A",
                     "$l_{conf,TTP}$ A",
                     "$l_{conf,FUS}$ A",
                     "$l_{conf,TDP43}$ A",
                     "$l_{conf,RNA}$ A",
                     "$\\tau_{conf,G3BP1}$ $ns$",
                     "$\\tau_{conf,PABP1}$ $ns$",
                     "$\\tau_{conf,TIA1}$ $ns$",
                     "$\\tau_{conf,TTP}$ $ns$",
                     "$\\tau_{conf,FUS}$ $ns$",
                     "$\\tau_{conf,TDP43}$ $ns$",
                     "$\\tau_{conf,RNA}$ $ns$",
                     "$R_{g,G3BP1}$ A",
                     "$R_{g,PABP1}$ A",
                     "$R_{g,TIA1}$ A",
                     "$R_{g,TTP}$ A",
                     "$R_{g,FUS}$ A",
                     "$R_{g,TDP43}$ A",
                     "$R_{g,RNA}$ A",
                     "$R_{h,G3BP1}$ A",
                     "$R_{h,PABP1}$ A",
                     "$R_{h,TIA1}$ A",
                     "$R_{h,TTP}$ A",
                     "$R_{h,FUS}$ A",
                     "$R_{h,TDP43}$ A",
                     "$R_{h,RNA}$ A",
                     "$D_{SE,GK,G3BP1}$ $\\mu m^{2} / s$",
                     "$D_{SE,GK,PABP1}$ $\\mu m^{2} / s$",
                     "$D_{SE,GK,TIA1}$ $\\mu m^{2} / s$",
                     "$D_{SE,GK,TTP}$ $\\mu m^{2} / s$",
                     "$D_{SE,GK,FUS}$ $\\mu m^{2} / s$",
                     "$D_{SE,GK,TDP43}$ $\\mu m^{2} / s$",
                     "$D_{SE,GK,RNA}$ $\\mu m^{2} / s$",
                     "$Occ_{G3BP1}$",
                     "$Occ_{PABP1}$",
                     "$Occ_{TIA1}$",
                     "$Occ_{TTP}$",
                     "$Occ_{FUS}$",
                     "$Occ_{TDP43}$",
                     "$Occ_{RNA}$",
                     "$r/R_{G3BP1}$",
                     "$r/R_{PABP1}$",
                     "$r/R_{TIA1}$",
                     "$r/R_{TTP}$",
                     "$r/R_{FUS}$",
                     "$r/R_{TDP43}$",
                     "$r/R_{RNA}$",
                     ]
        legacy_names = [
                 "Mass",
                 "c_dense_fit",
                 "c_dilute_fit",
                 "c_dense_calc",
                 "c_dilute_calc",
                 "P_SG",
                 "R_cond",
                 "W_int",
                 "gamma_1",
                 "gamma_2",
                 "gamma_ave",
                 "Delta_G",
                 "c_dilute_SM",
                 "c_dense_SM",
                 "P_SM",
                 "phi_D",
                 "N_D",
                 "R_g",
                 "phi_R",
                 "eta_GK",
                 "D_SE_GK_Rh",
                 "D_SE_GK_Rg",
                 "l_conf",
                 "tau_cage",
                 "D_cage",
                 "D_loglog",
                 "gamma_1_Protein",
                 "gamma_2_Protein",
                 "gamma_ave_Protein",
                 "gamma_1_RNA",
                 "gamma_2_RNA",
                 "gamma_ave_RNA",
                 "eta_GK_Theo",
                 "l_conf_G3BP1",
                 "l_conf_PABP1",
                 "l_conf_TIA1",
                 "l_conf_TTP",
                 "l_conf_FUS",
                 "l_conf_TDP43",
                 "l_conf_RNA",
                 "tau_conf_G3BP1",
                 "tau_conf_PABP1",
                 "tau_conf_TIA1",
                 "tau_conf_TTP",
                 "tau_conf_FUS",
                 "tau_conf_TDP43",
                 "tau_conf_RNA",
                 "Rg_G3BP1",
                 "Rg_PABP1",
                 "Rg_TIA1",
                 "Rg_TTP",
                 "Rg_FUS",
                 "Rg_TDP43",
                 "Rg_RNA",
                 "Rh_G3BP1",
                 "Rh_PABP1",
                 "Rh_TIA1",
                 "Rh_TTP",
                 "Rh_FUS",
                 "Rh_TDP43",
                 "Rh_RNA",
                 "D_SE_GK_G3BP1",
                 "D_SE_GK_PABP1",
                 "D_SE_GK_TIA1",
                 "D_SE_GK_TTP",
                 "D_SE_GK_FUS",
                 "D_SE_GK_TDP43",
                 "D_SE_GK_RNA",
                 "Occ_G3BP1",
                 "Occ_PABP1",
                 "Occ_TIA1",
                 "Occ_TTP",
                 "Occ_FUS",
                 "Occ_TDP43",
                 "Occ_RNA",
                 "r_over_R_G3BP1",
                 "r_over_R_PABP1",
                 "r_over_R_TIA1",
                 "r_over_R_TTP",
                 "r_over_R_FUS",
                 "r_over_R_TDP43",
                 "r_over_R_RNA",
                 ]

        preferred_name_map = {}
        for variable, name in zip(legacy_variables, legacy_names):
            try:
                preferred_name_map[self._resolve_column_name(variable)] = name
            except KeyError:
                print(f"[KMeans] Skipping unavailable summary column during setup: {variable}")

        self.variables = self._discover_observable_columns()
        self.names = self._build_plot_names(self.variables, preferred_name_map)
        self.variable_plot_name_map = dict(zip(self.variables, self.names))
        self.kmeans_excluded_variables = {"$Mass$ $(Da)$"}
        print(f"[KMeans] Using {len(self.variables)} non-SIG observable columns from Quant_Data.csv")

    @staticmethod
    def _normalize_feature_label(label):
        """Canonicalize a LaTeX column label for alias-tolerant comparison
        (collapse doubled backslashes, unify the ``{GK Theo}`` token, squeeze
        whitespace)."""
        text = str(label).strip()
        while "\\\\" in text:
            text = text.replace("\\\\", "\\")
        text = re.sub(r"\{GK,\s*Theo\}", "{GK Theo}", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def _resolve_column_name(self, requested):
        """Map a requested LaTeX label to the matching Quant_Data.csv column,
        tolerating formatting drift via ``_normalize_feature_label``. Caches hits
        and raises ``KeyError`` if no column matches."""
        if requested in self._resolved_column_cache:
            return self._resolved_column_cache[requested]
        if requested in self.df_og.columns:
            self._resolved_column_cache[requested] = requested
            return requested

        requested_norm = self._normalize_feature_label(requested)
        for column in self.df_og.columns:
            if self._normalize_feature_label(column) == requested_norm:
                self._resolved_column_cache[requested] = column
                if column != requested:
                    print(f"[KMeans] Resolved column alias: {requested} -> {column}")
                return column

        raise KeyError(f"Column not found in Quant_Data.csv: {requested}")

    def _resolve_feature_list(self, feature_list, allow_missing=False):
        """Resolve a list of requested labels to de-duplicated column names.

        Returns ``(resolved, missing)``. Unresolvable labels go into ``missing``;
        if any are missing and ``allow_missing`` is False, raises ``KeyError``.
        """
        resolved = []
        missing = []
        seen = set()
        for requested in feature_list:
            try:
                column = self._resolve_column_name(requested)
            except KeyError:
                missing.append(requested)
                continue
            if column in seen:
                continue
            seen.add(column)
            resolved.append(column)
        if missing and not allow_missing:
            raise KeyError(f"Columns not found in Quant_Data.csv: {missing}")
        return resolved, missing

    def _discover_observable_columns(self):
        """Return the numeric observable columns of Quant_Data.csv, dropping ID/
        metadata columns and any SIG (uncertainty) or ``QC`` quality-flag column."""
        excluded_columns = {
            "Small Molecule ID",
            "Small Molecule Name",
            "Compound Name",
            "Compound Class",
            "D_Binary",
            "ID",
            "Time",
            "Timestep",
        }

        observable_columns = []
        for column in self.df_og.columns:
            if column in excluded_columns or "SIG" in str(column) or str(column).startswith("QC "):
                continue
            numeric_values = pd.to_numeric(self.df_og[column], errors="coerce")
            if numeric_values.notna().any():
                observable_columns.append(column)
        return observable_columns

    @staticmethod
    def _sanitize_feature_name(label):
        """Convert a LaTeX column label into a filesystem-safe plot name
        (spell out Greek macros, strip math markup, collapse to underscores)."""
        sanitized = str(label)
        replacements = {
            "\\Delta": "Delta",
            "\\gamma": "gamma",
            "\\eta": "eta",
            "\\phi": "phi",
            "\\tau": "tau",
            "\\mu": "mu",
            "\\AA": "A",
        }
        for old, new in replacements.items():
            sanitized = sanitized.replace(old, new)
        sanitized = sanitized.replace("$", "")
        sanitized = sanitized.replace("{", "")
        sanitized = sanitized.replace("}", "")
        sanitized = re.sub(r"[^A-Za-z0-9]+", "_", sanitized)
        sanitized = re.sub(r"_+", "_", sanitized).strip("_")
        return sanitized or "observable"

    def _build_plot_names(self, variables, preferred_name_map):
        """Assign each observable a short plot name (preferred legacy name when
        available, else a sanitized one), suffixing ``_N`` to break collisions."""
        names = []
        name_counts = {}
        for variable in variables:
            base_name = preferred_name_map.get(variable, self._sanitize_feature_name(variable))
            count = name_counts.get(base_name, 0)
            name_counts[base_name] = count + 1
            if count:
                names.append(f"{base_name}_{count + 1}")
            else:
                names.append(base_name)
        return names

    def _cleanup_generated_plots(self, suffix):
        """Delete any pre-existing FIGURES/PROPERTIES files ending in *suffix* so a
        rerun does not leave stale panels behind."""
        plot_dir = os.path.join(self.path, "FIGURES", "PROPERTIES")
        if not os.path.isdir(plot_dir):
            return
        for filename in os.listdir(plot_dir):
            if filename.endswith(suffix):
                try:
                    os.remove(os.path.join(plot_dir, filename))
                except OSError:
                    pass

    @staticmethod
    def _is_invalid_kmeans_feature(column_name):
        """Return True if column represents a physically invalid or diagnostic-only observable.

        Excludes all MSD-based diffusion/confinement fit artifacts (lab-frame LAMMPS,
        RCC segmented, tracked aliases), derived cage quantities, and diagnostic
        metrics that are not independent physical observables.  Only validated
        transport (eta_GK, D_SE_GK), thermodynamic, structural, and true
        confinement (l_conf) observables pass through.
        """
        # Normalize: strip backslashes and dollar signs for substring matching
        norm = column_name.replace("\\", "").replace("$", "")
        invalid_markers = [
            "D_{LAMMPS",          # LAMMPS lab-frame D (raw, all, per-species)
            "tau_{LAMMPS",        # LAMMPS tau (raw, all, per-species)
            "l_{Cond",            # l_Cond from 3-param model (all sources + per-species)
            "eta_{D",             # Stokes-Einstein eta from invalid D fits (all sources)
            "D_{RCC",             # RCC D (full, segmented)
            "tau_{RCC",           # RCC tau (full, segmented)
            "D_{tracked",         # tracked D aliases (MSD-based)
            "tau_{tracked",       # tracked tau aliases
            "l_{tracked",         # tracked l aliases
            "D_{cage",            # cage mobility (derived, not transport)
            "tau_{cage",          # cage relaxation time (diagnostic)
            "D_{loglog",          # log-log slope D (unreliable in confinement)
            "l_{conf,chain",      # per-chain l_conf median (diagnostic)
            "eta_{GK Theo",       # theoretical GK viscosity (less reliable)
            "eta_{GK,Theo",       # alt formatting
            "R^2_{conf",          # fit quality metric, not physics
            "R^{2}_{conf",        # alt formatting
        ]
        for marker in invalid_markers:
            if marker in norm:
                return True
        # Exclude Mass (not a simulation observable)
        if "Mass" in norm and "Da" in norm:
            return True
        return False

    @staticmethod
    def _is_allowed_condensate_kmeans_feature(column_name):
        """Whitelist observables that describe the condensate itself.

        Built as deny-then-allow so the candidate pool spans the full set of
        condensate structural / interfacial / thermodynamic observables (close to
        the pre-whitelist baseline), minus the families excluded by analysis
        decision.

        Excluded:
          * physics-invalid MSD/cage/diagnostic transport (_is_invalid_kmeans_feature)
          * raw small-molecule concentrations (c_SM, c_dilute,SM, c_dense,SM) —
            mass concentrations tied to loading, not comparable between compounds.
            P_SM is KEPT (allowed below): it is the dimensionless partition
            coefficient (a number-density ratio in which molecular mass cancels),
            and it is informative that DSMs reach a similar/lower partition
            coefficient despite a higher dilute-phase concentration.
          * dilute-phase SG concentrations and P_SG — noisy dilute estimates.
          * Stokes-Einstein D_SE,GK transport proxies (per-species and aggregate).
          * cross-T binodal-screen metrics (T_c,pert, c_c,pert, A_pert, Delta-c-bar,
            DDc_rel/abs, Dc_dil_inf_rel) — temperature-derived, not for single-T (300 K)
            clustering.

          * per-species structural observables (l_conf, tau_conf, R_g, R_h, Occ,
            r/R per biopolymer; per-biopolymer gamma) — excluded by analysis
            decision; the classifier uses condensate-aggregate observables only.

        Allowed: condensate-aggregate size/shape (R_cond, W_interface, R_g, phi_D,
        N_D, phi_R), dense-phase concentration, viscosity (eta_GK), aggregate
        confinement length (l_conf), TOTAL interfacial tension (gamma_1/2/ave),
        transfer free energy (Delta G_trans), and the SM partition coefficient (P_SM).
        """
        if summary_plots._is_invalid_kmeans_feature(column_name):
            return False

        norm = column_name.replace("\\", "").replace("$", "")

        # --- explicit exclusions ---
        # Raw SM mass concentrations (c_SM, c_dilute,SM, c_dense,SM) are tied to
        # loading; reject them but KEEP P_SM (dimensionless partition coefficient).
        if "SM" in norm and not norm.startswith("P_{SM"):
            return False
        if "dilute" in norm or norm.startswith("P_{SG"):   # dilute-phase SG conc + P_SG
            return False
        if "D_{SE" in norm:                                # Stokes-Einstein transport proxies
            return False
        binodal_markers = ("{c,pert", "A_{pert", "bar{Delta c", "DeltaDelta c", "c_{dil,inf,rel")
        if any(m in norm for m in binodal_markers):        # cross-T binodal screen metrics
            return False

        text = summary_plots._normalize_feature_label(column_name)

        # --- allowed condensate observables ---
        exact_allowed = {
            "$c_{dense,SG,fit}$ $(mg/ml)$",
            "$c_{dense,SG,calc}$ $(mg/ml)$",
            "$R_{cond}$ $(\\AA)$",
            "$W_{interface}$ $(\\AA)$",
            "$\\phi_{D}$",
            "$N_{D}$",
            "$R_{g}$",
            "$\\phi_{R}$",
            "$\\eta_{GK}$ Pa s",
            "$l_{conf}$ A",
            "$\\Delta G_{trans}$ $(kJ/mol)$",
            "$P_{SM}$",
        }
        normalized_exact_allowed = {
            summary_plots._normalize_feature_label(item)
            for item in exact_allowed
        }
        if text in normalized_exact_allowed:
            return True

        # TOTAL interfacial tension only (gamma_{1}, gamma_{2}, gamma_ave); the
        # per-biopolymer variants carry ",Protein"/",RNA" and are excluded.
        if norm.startswith("gamma_") and "Protein" not in norm and "RNA" not in norm:
            return True

        return False

    def _get_kmeans_feature_candidates(self):
        """Return the observable columns eligible for the KMeans feature search:
        those not in the hard exclusion set and passing the condensate whitelist."""
        return [col for col in self.variables
                if col not in self.kmeans_excluded_variables
                and self._is_allowed_condensate_kmeans_feature(col)]

    @staticmethod
    def _should_use_log_scale(series, ratio_threshold=500):
        """Return True if *series* is strictly positive and spans at least
        *ratio_threshold* (max/min), i.e. wide enough to warrant a log y-axis."""
        finite_values = pd.to_numeric(series, errors="coerce").dropna()
        if finite_values.empty or (finite_values <= 0).any():
            return False
        positive_min = finite_values.min()
        positive_max = finite_values.max()
        if positive_min <= 0:
            return False
        return (positive_max / positive_min) >= ratio_threshold

    def _set_violin_axis_limits(self, ax, series, y_line, forced_limits=None):
        """Set the violin-plot y-axis from *series* (plus the SG reference
        *y_line*). Uses a log scale when warranted, otherwise 6 evenly spaced
        ticks with a shared x10^n exponent; *forced_limits* overrides the bounds.
        """
        finite_values = pd.to_numeric(series, errors="coerce").dropna()
        extra_values = []
        if pd.notna(y_line):
            extra_values.append(float(y_line))
        if extra_values:
            finite_values = pd.concat([finite_values, pd.Series(extra_values)], ignore_index=True)
        if finite_values.empty:
            return

        if self._should_use_log_scale(finite_values):
            positive_values = finite_values[finite_values > 0]
            ax.set_yscale("log")
            ax.set_ylim(positive_values.min() / 1.4, positive_values.max() * 1.4)
            return

        ymin = float(finite_values.min())
        ymax = float(finite_values.max())
        tick_lo = _round_2sig_floor(ymin)
        tick_hi = _round_2sig_ceil(ymax)
        if forced_limits is not None:
            # Explicit (lo, hi) override; bypasses 2-sig rounding so the 6 evenly
            # spaced ticks land on clean values (avoids duplicate labels, e.g. ΔG).
            tick_lo, tick_hi = float(forced_limits[0]), float(forced_limits[1])
        if abs(tick_hi - tick_lo) < 1e-30:
            tick_hi = tick_lo + 1.0
        ticks = np.linspace(tick_lo, tick_hi, 6)
        ax.set_ylim(tick_lo, tick_hi)
        ax.set_yticks(ticks)
        max_abs_tick = max(abs(tick_lo), abs(tick_hi))
        if max_abs_tick > 0:
            common_exp = int(np.floor(np.log10(max_abs_tick)))
        else:
            common_exp = 0
        scale = 10.0 ** common_exp
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda val, pos, s=scale: f"{val / s:.1f}")
        )
        ax.annotate(f"$\\times10^{{{common_exp}}}$", xy=(0, 1), xycoords='axes fraction',
                     ha='left', va='bottom', fontsize=8)

    def _write_kmeans_preprocessing_report(self):
        """Write the accumulated per-context feature-preprocessing records to
        RESULTS/SUMMARY/KMeans_Preprocessing_Report.csv (no-op if none)."""
        if not self.kmeans_preprocessing_records:
            return
        report_path = os.path.join(self.path, "RESULTS", "SUMMARY", "KMeans_Preprocessing_Report.csv")
        pd.DataFrame(self.kmeans_preprocessing_records).to_csv(report_path, index=False)

    def _drop_incomplete_kmeans_features(self, df, features_df):
        """Drop feature columns lacking full value/SIG coverage (any NaN value, or
        a missing/NaN companion ``SIG`` column). Returns the filtered frame and a
        list of dropped-column descriptions with reasons."""
        dropped_incomplete = []
        for column in list(features_df.columns):
            values = pd.to_numeric(features_df[column], errors="coerce")
            sigma_column = "SIG" + column
            sigma_missing = sigma_column not in df.columns
            sigma_has_nan = False
            if not sigma_missing:
                sigma_values = pd.to_numeric(df[sigma_column], errors="coerce")
                sigma_has_nan = sigma_values.isna().any()
            value_has_nan = values.isna().any()
            if value_has_nan or sigma_missing or sigma_has_nan:
                reasons = []
                if value_has_nan:
                    reasons.append("value_nan")
                if sigma_missing:
                    reasons.append("sig_missing")
                elif sigma_has_nan:
                    reasons.append("sig_nan")
                dropped_incomplete.append(f"{column} ({','.join(reasons)})")
                features_df = features_df.drop(columns=[column])
        return features_df, dropped_incomplete

    def _prepare_feature_matrix(self, df, feature_list, context, min_features=2):
        """Build a clean numeric feature matrix from *df* for the requested
        *feature_list*.

        Resolves column aliases, coerces to numeric, drops all-NaN and
        value/SIG-incomplete columns, and requires no residual NaNs and at least
        *min_features* usable columns (raises ``ValueError`` otherwise). Logs a
        preprocessing record under *context*. Returns ``(features_array,
        usable_feature_names)``.
        """
        resolved_features, missing_features = self._resolve_feature_list(feature_list, allow_missing=True)
        if missing_features:
            print(f"[KMeans] {context}: requested features missing from Quant_Data.csv: {missing_features}")
        if len(resolved_features) < min_features:
            raise ValueError(
                f"{context}: need at least {min_features} usable features after resolving column aliases; "
                f"requested={feature_list}, resolved={resolved_features}"
            )

        features_df = df.loc[:, resolved_features].apply(pd.to_numeric, errors="coerce")
        requested_features = list(feature_list)
        dropped_all_nan = [column for column in resolved_features if features_df[column].isna().all()]
        if dropped_all_nan:
            print(f"[KMeans] {context}: dropping all-NaN features: {dropped_all_nan}")
            features_df = features_df.drop(columns=dropped_all_nan)

        features_df, dropped_incomplete = self._drop_incomplete_kmeans_features(df, features_df)
        if dropped_incomplete:
            print(f"[KMeans] {context}: dropping incomplete features lacking full value/SIG coverage: {dropped_incomplete}")

        usable_features = list(features_df.columns)
        if len(usable_features) < min_features:
            raise ValueError(
                f"{context}: need at least {min_features} usable features after dropping incomplete features; "
                f"requested={requested_features}, usable={usable_features}"
            )

        missing_entries = int(features_df.isna().sum().sum())
        if missing_entries:
            raise ValueError(
                f"{context}: feature matrix still contains {missing_entries} missing entries after strict "
                f"value/SIG filtering; usable={usable_features}"
            )
        features = features_df.to_numpy(dtype=float, copy=True)

        self.kmeans_preprocessing_records.append({
            "Context": context,
            "Requested_Feature_Count": len(requested_features),
            "Usable_Feature_Count": len(usable_features),
            "Dropped_All_NaN_Features": " | ".join(dropped_all_nan),
            "Dropped_Incomplete_Features": " | ".join(dropped_incomplete),
            "Imputed_Value_Count": missing_entries,
            "Requested_Features": " | ".join(requested_features),
            "Usable_Features": " | ".join(usable_features),
        })
        self._write_kmeans_preprocessing_report()
        return features, usable_features

    def clean_df(self):
        """Build the per-system working frames from Quant_Data.csv.

        Drops aggregate rows (DSM/NDSM and their *_AVG), de-duplicates, relabels
        the Compound Class to DSM/NDSM, and coerces observables to numeric. Sets
        ``self.df`` (per-system), ``self.df_filtered`` (DSM+NDSM only), and the
        class-wise ``self.df_mean`` / ``self.df_std`` tables (rows DSM, NDSM, SG).
        """
        df = self.df_og.copy()

        df = df[df['Small Molecule ID'] != "DSM"]
        df = df[df['Small Molecule ID'] != "DSM_AVG"]
        df = df[df['Small Molecule ID'] != "NDSM"]
        df = df[df['Small Molecule ID'] != "NDSM_AVG"]

        df = df.drop_duplicates()

        df = df.copy()
        df.loc[:, "Compound Class"] = df["Compound Class"].replace({
            'Dissolving': "DSM",
            'Non-Dissolving': "NDSM",
        })
        for column in self.variables:
            df.loc[:, column] = pd.to_numeric(df[column], errors="coerce")

        features = df.loc[:, ["Compound Class"] + self.variables]

        df_mean = features.groupby("Compound Class", observed=False).mean()
        df_std = features.groupby("Compound Class", observed=False).std()

        df_mean["ID"] = ["DSM", "NDSM", "SG"]
        df_std["ID"] = ["DSM", "NDSM", "SG"]

        self.df = df
        categories_to_plot = ["DSM", "NDSM"]
        self.df_filtered = df[df['Compound Class'].isin(categories_to_plot)]
        self.df_mean = df_mean
        self.df_std = df_std


    def plot_mean(self):
        """Save one class-mean bar plot per observable (DSM/NDSM/SG with std-dev
        error bars) to FIGURES/PROPERTIES/{name}_AVG.png."""
        col_pall = sns.color_palette(["#808080","#bfe49b","#40641b"], 3)
        df = self.df
        df_mean = self.df_mean
        df_std = self.df_std
        variables = self.variables

        # Mean Calculation
        for var in range(len(self.variables)):
            sns.set_theme(style="ticks")
            sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
            plt.rc('axes', titlesize=10)  # fontsize of the axes title
            plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
            plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
            plt.rc('legend', fontsize=8)  # legend fontsize
            plt.rc('font', size=10)  # controls default text sizes
            plt.rc('axes', linewidth=2)

            fig, ax1 = plt.subplots(figsize=(2.8, 2.8))
            plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
            sns.barplot(ax=ax1, data=df_mean, x="ID", y=variables[var], hue="ID", palette=col_pall,
                             dodge=False,
                             width=0.5, saturation=100, linewidth=2, edgecolor="k")
            plotline, caps, barlinecols = ax1.errorbar(df_mean["ID"], df_mean[variables[var]], yerr=df_std[variables[var]], fmt="none", color='k', elinewidth=1, capsize=2, capthick=1)
            plt.setp(barlinecols[0], capstyle='round')
            ax1.get_legend().remove()
            ax1.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in',
                            length=4,
                            width=2)
            ax1.set_xlabel("")
            ax1.set_ylabel("")
            plt.tight_layout()
            plt.savefig("{}/FIGURES/PROPERTIES/{}_AVG.png".format(self.path, self.names[var]), format="png", dpi=400)
            plt.close(fig)

    def plot_full(self):
        """Save one per-system bar plot per observable (every small molecule,
        colored by class, with SIG error bars when present) to
        FIGURES/PROPERTIES/{name}_ALL.png."""
        col_pall = sns.color_palette(["#808080","#bfe49b","#40641b"], 3)
        for var in range(len(self.variables)):
            sns.set_theme(style="ticks")
            sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
            plt.rc('axes', titlesize=10)  # fontsize of the axes title
            plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
            plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
            plt.rc('legend', fontsize=8)  # legend fontsize
            plt.rc('font', size=10)  # controls default text sizes
            plt.rc('axes', linewidth=2)

            fig, ax1 = plt.subplots(figsize=(2.8, 2.8))
            plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
            sns.barplot(ax=ax1, data=self.df, x="Small Molecule ID", y=self.variables[var], hue="Compound Class", palette=col_pall,
                        dodge=False,
                        width=0.5, saturation=100, linewidth=2, edgecolor="k")
            sigma_column = "SIG" + self.variables[var]
            if sigma_column in self.df.columns:
                plotline, caps, barlinecols = ax1.errorbar(self.df["Small Molecule ID"], self.df[self.variables[var]],
                                                           yerr=self.df[sigma_column], fmt="none", color='k', elinewidth=1,
                                                           capsize=2, capthick=1)
                plt.setp(barlinecols[0], capstyle='round')
            ax1.get_legend().remove()
            ax1.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in',
                            length=4,
                            width=2)
            plt.xticks(rotation=90)
            ax1.set_xlabel("")
            ax1.set_ylabel("")
            plt.tight_layout()
            plt.savefig("{}/FIGURES/PROPERTIES/{}_ALL.png".format(self.path, self.names[var]), format="png", dpi=400)
            plt.close(fig)

    def plot_violin(self):
        """Save one DSM-vs-NDSM violin plot per observable, with the SG class mean
        drawn as a horizontal reference line, to FIGURES/PROPERTIES/{name}_VP.png.
        Pre-existing ``_VP.png`` panels are cleared first."""
        self._cleanup_generated_plots("_VP.png")
        for var in range(len(self.variables)):
            col_pal_sm = sns.color_palette(["#40641b","#bfe49b"], 2)
            variable = self.variables[var]
            plot_df = self.df_filtered.loc[:, ["Compound Class", variable]].copy()
            plot_df.loc[:, variable] = pd.to_numeric(plot_df[variable], errors="coerce")
            plot_df = plot_df.dropna(subset=[variable])
            if plot_df.empty:
                print(f"[KMeans] Skipping violin plot with no finite values: {variable}")
                continue

            sns.set_theme(style="ticks")
            sns.set_style('white')
            plt.rc('axes', titlesize=10)
            plt.rc('axes', labelsize=10)
            plt.rc('xtick', labelsize=10)
            plt.rc('ytick', labelsize=10)
            plt.rc('legend', fontsize=8)
            plt.rc('font', size=10)
            plt.rc('axes', linewidth=2)

            fig, ax1 = _make_fixed_axes(_VP_FIG_SIZE, _VP_AX_RECT)
            plt.rc('xtick', labelsize=10)
            plt.rc('ytick', labelsize=10)
            sns.violinplot(ax=ax1, data=plot_df, x="Compound Class",
                            y=variable, hue="Compound Class", palette=col_pal_sm,
                            dodge=False, width=0.35, saturation=100,
                           linewidth=2, edgecolor="k", inner="box",
                           cut=0, order=["DSM", "NDSM"])
            y_line = self.df_mean.loc[self.df_mean["ID"]=="SG", variable].iloc[0]
            plt.axhline(y=y_line, color=sns.color_palette(["#808080"], 1)[0], linewidth=2)
            ax1.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in',
                            length=4,
                            width=2)
            ax1.set_xlabel("")
            ax1.set_ylabel("")
            # Per-observable y-limit overrides (clean, non-duplicate tick labels).
            # ΔG_trans ≈ -12.7..-14.7 kJ/mol: force -16..-11 so the ÷10¹ ticks read
            # -1.6,-1.5,-1.4,-1.3,-1.2,-1.1 instead of a duplicated -1.4,-1.4.
            _forced = {"Delta_G": (-16.0, -11.0)}.get(self.names[var])
            self._set_violin_axis_limits(ax1, plot_df[variable], y_line, forced_limits=_forced)
            if ax1.get_legend() is not None:
                ax1.get_legend().remove()
            plt.savefig("{}/FIGURES/PROPERTIES/{}_VP.png".format(self.path, self.names[var]), format="png", dpi=400)
            plt.close(fig)


    def k_means_simulation(self, feature_list, suffix=""):
        """Run the StandardScaler -> PCA(2) -> KMeans(2) classifier on *feature_list*
        and emit all panels and prediction lists for one feature set.

        Aligns the unsupervised cluster labels to the true DSM/NDSM labels, prints
        accuracy/silhouette diagnostics, and writes the PCA loading bars, the
        (annotated) PC1-PC2 scatter, and the ``dsm_predicted{suffix}.txt`` /
        ``ndsm_predicted{suffix}.txt`` class lists. *suffix* disambiguates outputs.
        """
        df = self.df.iloc[1:].copy()
        df_part1 = df.iloc[0:10]  # Rows 2-11 (Python indexing: includes 2-10)
        df_part2 = df.iloc[10:20]  # Rows 11-20 (Python indexing: includes 11-19)

        # Swap and reassemble the DataFrame
        df = pd.concat([df_part1, df_part2]).reset_index(drop=True)
        filtered_feature_list = [feature for feature in feature_list if feature not in self.kmeans_excluded_variables]
        if len(filtered_feature_list) != len(feature_list):
            print(
                f"[KMeans] Excluding non-simulation features from PCA/KMeans: "
                f"{sorted(set(feature_list) - set(filtered_feature_list))}"
            )

        features, feature_labels = self._prepare_feature_matrix(
            df,
            filtered_feature_list,
            context=f"k_means_simulation{suffix or '_default'}",
        )

        labels = list(df.loc[:, "D_Binary"].astype(int))

        kmeans = KMeans(
            init="random",
            n_clusters=2,
            n_init=20,
            max_iter=400,
            random_state=None)

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        pca = PCA(n_components=2)
        pca_features = pd.DataFrame(pca.fit_transform(scaled_features),
                                    columns=["Principle Component 1", "Principle Component 2"])

        kmeans.fit(pca_features)
        loadings = pd.DataFrame(pca.components_, columns=feature_labels)

        sns.set_theme(style="ticks")
        sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
        plt.rc('axes', titlesize=10)  # fontsize of the axes title
        plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
        plt.rc('legend', fontsize=8)  # legend fontsize
        plt.rc('font', size=10)  # controls default text sizes
        plt.rc('axes', linewidth=2)

        print(loadings)

        feature_display_labels = [self.variable_plot_name_map.get(label, label) for label in feature_labels]

        fig, ax1 = _make_fixed_axes(_PCA_BAR_FIG_SIZE, _PCA_BAR_AX_RECT)
        plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
        sns.barplot(ax=ax1, x=feature_labels, y=np.abs(loadings.iloc[0]),
                    dodge=True,
                    width=0.6, saturation=100, color = 'grey', edgecolor ="k")
        ax1.set_ylim(0, 1)
        ax1.set_yticks(np.linspace(0, 1, 6))
        ax1.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in',
                        length=4,
                        width=2)
        ax1.set_xlabel("")
        ax1.set_ylabel("")
        plt.setp(ax1.get_xticklabels(), rotation=35, ha='right', rotation_mode='anchor')
        plt.savefig("{}/FIGURES/PROPERTIES/KMeans{}_PCA1_Loading.png".format(self.path, suffix), format="png", dpi=400)
        plt.close(fig)

        sns.set_theme(style="ticks")
        sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
        plt.rc('axes', titlesize=10)  # fontsize of the axes title
        plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
        plt.rc('legend', fontsize=8)  # legend fontsize
        plt.rc('font', size=10)  # controls default text sizes
        plt.rc('axes', linewidth=2)

        fig, ax1 = _make_fixed_axes(_PCA_BAR_FIG_SIZE, _PCA_BAR_AX_RECT)
        plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
        sns.barplot(ax=ax1, x=feature_labels, y=np.abs(loadings.iloc[1]),
                    dodge=True,
                    width=0.6, saturation=100, color = 'grey', edgecolor ="k")
        ax1.set_ylim(0, 1)
        ax1.set_yticks(np.linspace(0, 1, 6))
        ax1.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in',
                        length=4,
                        width=2)
        ax1.set_xlabel("")
        ax1.set_ylabel("")
        plt.setp(ax1.get_xticklabels(), rotation=35, ha='right', rotation_mode='anchor')
        plt.savefig("{}/FIGURES/PROPERTIES/KMeans{}_PCA2_Loading.png".format(self.path, suffix), format="png", dpi=400)
        plt.close(fig)


        print(f"Lowest SSE: {kmeans.inertia_:.6f}")
        ndsm_center = kmeans.cluster_centers_[0]
        dsm_center = kmeans.cluster_centers_[1]
        print(f"DSM Center: pca1={dsm_center[0]:.6f}; pca2={dsm_center[1]:.6f}")
        print(f"NDSM Center: pca1={ndsm_center[0]:.6f}; pca2={ndsm_center[1]:.6f}")

        pred_labs = []
        true_labs = []

        predictions = list(kmeans.labels_)
        acc1 = np.mean(np.array(predictions) == np.array(labels))
        acc2 = np.mean((1 - np.array(predictions)) == np.array(labels))
        if acc2 > acc1:
            predictions = [1 - p for p in predictions]

        for i in range(len(predictions)):
            if predictions[i] == 1:
                pred_labs.append("DSM")
            else:
                pred_labs.append("NDSM")
            if labels[i] == 1:
                true_labs.append("DSM")
            else:
                true_labs.append("NDSM")

        fpr = 0
        fnr = 0
        mcr = 0
        ccr = 0
        for i in range(len(predictions)):
            if predictions[i] != labels[i]:
                mcr += 1
                if predictions[i] == 1:
                    fpr += 1
                elif predictions[i] == 0:
                    fnr += 1
            else:
                ccr += 1

        fpr = fpr / (len(predictions) / 2) * 100
        fnr = fnr / (len(predictions) / 2) * 100
        mcr = mcr / len(predictions) * 100
        ccr = ccr / len(predictions) * 100

        print(f"False Positive Rate: {fpr:.6f}")
        print(f"False Negative Rate: {fnr:.6f}")
        print(f"Mis-classification Rate: {mcr:.6f}")
        print(f"Classification Rate: {ccr:.6f}")
        score = silhouette_score(scaled_features, kmeans.labels_)
        print(f"Silhouette Score: {score:.6f}")

        pca_features["Predicted Label"] = list(pred_labs)
        pca_features["True Label"] = list(true_labs)
        pca_features["Small Molecule ID"] = list(df["Small Molecule ID"])
        pca_features["Small Molecule Name"] = list(df["Small Molecule Name"])
        print(pca_features)



        sns.set_theme(style="ticks")
        sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
        plt.rc('axes', titlesize=10)  # fontsize of the axes title
        plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
        plt.rc('legend', fontsize=8)  # legend fontsize
        plt.rc('font', size=10)  # controls default text sizes
        plt.rc('axes', linewidth=2)

        scatter_xlim = (-3, 3)
        scatter_ylim = (-2, 4)

        fig, ax1 = _make_fixed_axes(_SCATTER_FIG_SIZE, _SCATTER_AX_RECT)
        col_pall_sm = sns.color_palette(["#40641b", "#bfe49b"], 2)

        sns.scatterplot(
            x=pca_features.loc[:, "Principle Component 1"],
            y=pca_features.loc[:, "Principle Component 2"],
            hue=pca_features.loc[:, "Predicted Label"],
            style=pca_features.loc[:, "True Label"],
            palette=col_pall_sm,
            s=80,
            ax=ax1,
        )
        ax1.set_xlim(scatter_xlim)
        ax1.set_xticks(np.linspace(scatter_xlim[0], scatter_xlim[1], 7))
        _apply_2sig_yticks(ax1, scatter_ylim[0], scatter_ylim[1], n_ticks=6, annotate_exp=False)
        ax1.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in', length=4,
                        width=2)
        ax1.legend(loc='upper right', ncol=2, frameon=False, fontsize=8)
        ax1.set_xlabel("")
        ax1.set_ylabel("")
        if ax1.get_legend() is not None:
            ax1.get_legend().remove()
        plt.savefig("{}/FIGURES/PROPERTIES/KMeans{}.png".format(self.path, suffix), format="png", dpi=400)
        plt.close(fig)

        df_annotate = self.df.set_index("Small Molecule ID").iloc[1:,:]
        plt.rc('font', size=6)  # controls default text sizes
        fig, ax1 = _make_fixed_axes(_SCATTER_FIG_SIZE, _SCATTER_AX_RECT)
        sns.scatterplot(
            x=pca_features.loc[:, "Principle Component 1"],
            y=pca_features.loc[:, "Principle Component 2"],
            hue=pca_features.loc[:, "Predicted Label"],
            style=pca_features.loc[:, "True Label"],
            palette=col_pall_sm,
            s=80,
            ax=ax1,
        )
        ax1.set_xlim(scatter_xlim)
        ax1.set_xticks(np.linspace(scatter_xlim[0], scatter_xlim[1], 7))
        _apply_2sig_yticks(ax1, scatter_ylim[0], scatter_ylim[1], n_ticks=6, annotate_exp=False)
        ax1.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in', length=4,
                        width=2)
        ax1.legend(loc='upper right', ncol=2, frameon=False, fontsize=8)
        ax1.set_xlabel("")
        ax1.set_ylabel("")
        if ax1.get_legend() is not None:
            ax1.get_legend().remove()
        for i, txt in enumerate(df_annotate.index):
            ax1.annotate(
                txt,
                xy=(pca_features.iloc[i, 0], pca_features.iloc[i, 1]),
                xytext=(-1, 1),
                textcoords="offset points",
                ha="right",
                va="bottom",
            )
        plt.savefig("{}/FIGURES/PROPERTIES/KMeans{}_Annotated.png".format(self.path, suffix), format="png", dpi=400)
        plt.close(fig)

        dsm_list = []
        ndsm_list = []

        pca_sorted = pca_features.sort_values(by="Predicted Label")

        # NOTE: write to *_predicted.txt (not *_list.txt) so we do not
        # overwrite the input system lists consumed by SYSTEM_ANALYSIS_FINAL_*.
        dsm_list = list(pca_sorted[pca_sorted["Predicted Label"] == "DSM"]["Small Molecule Name"])
        with open('{}/dsm_predicted{}.txt'.format(self.path, suffix), 'w') as f:
            for item in dsm_list:
                f.write(f"{item}\n")

        ndsm_list = list(pca_sorted[pca_sorted["Predicted Label"] == "NDSM"]["Small Molecule Name"])
        with open('{}/ndsm_predicted{}.txt'.format(self.path, suffix), 'w') as f:
            for item in ndsm_list:
                f.write(f"{item}\n")

    def automated_feature_search(self):
        """Greedy correlation-ranked feature search over the condensate observable
        pool, then full visualization of the best subset.

        Ranks candidate features by absolute correlation to the DSM/NDSM labels,
        runs single-feature and cumulative (top-k) PCA+KMeans accuracy sweeps,
        writes the ranking / single-feature / iteration CSVs to RESULTS/SUMMARY,
        and calls ``k_means_simulation`` on the highest-accuracy subset
        (suffix ``_Automated_Optimal``).
        """
        print("\n=== Automated Feature Ranking & Selection ===")
        # 1. Prepare Data
        df = self.df.iloc[1:].copy()
        df_part1 = df.iloc[0:10]
        df_part2 = df.iloc[10:20]
        df = pd.concat([df_part1, df_part2]).reset_index(drop=True)

        labels = df["D_Binary"].values.astype(int)

        # 2. Extract and Rank Features (physics-validated only)
        candidate_features = self._get_kmeans_feature_candidates()
        excluded_physics = [col for col in self.variables if self._is_invalid_kmeans_feature(col)]
        if excluded_physics:
            print(f"[KMeans] Excluded {len(excluded_physics)} physically invalid features "
                  f"(MSD-based D/tau/l/eta, cage, diagnostic):")
            for col in excluded_physics:
                print(f"  - {col}")
        print(f"[KMeans] {len(candidate_features)} physics-validated candidate features remain")
        candidate_df = df.loc[:, candidate_features].apply(pd.to_numeric, errors="coerce")
        candidate_df, dropped_incomplete = self._drop_incomplete_kmeans_features(df, candidate_df)
        if dropped_incomplete:
            print(
                "[KMeans] automated_feature_search: excluding features without full value/SIG coverage: "
                f"{dropped_incomplete}"
            )
        features_list = list(candidate_df.columns)

        feature_stats = {}
        correlations = []
        for f in features_list:
            series = candidate_df[f]
            std = series.std(skipna=True)
            if np.isnan(std) or std == 0:
                continue
            variance = series.var(skipna=True)
            corr = np.corrcoef(series.values, labels)[0, 1]
            if np.isnan(corr):
                corr = 0
            feature_stats[f] = {
                "Raw_Mean": float(series.mean(skipna=True)),
                "Raw_Std": float(std),
                "Raw_Variance": float(variance),
            }
            correlations.append((f, abs(corr), corr))

        correlations.sort(key=lambda x: x[1], reverse=True)

        # Save rankings
        print("\n--- Feature Ranking Used by Greedy Selection ---")
        print("[KMeans] Selection order is descending absolute correlation to DSM/NDSM labels; "
              "raw variance is reported for audit only because features are StandardScaler-normalized before KMeans.")
        ranking_data = []
        for idx, (f, abs_corr, corr) in enumerate(correlations):
            stats = feature_stats[f]
            print(
                f"{idx+1}. {f:30} | Abs Corr: {abs_corr:.3f} | Raw Corr: {corr:.3f} | "
                f"Raw Var: {stats['Raw_Variance']:.3g}"
            )
            ranking_data.append({
                "Rank": idx+1,
                "Feature": f,
                "Abs_Correlation": abs_corr,
                "Raw_Correlation": corr,
                **stats,
            })

        ranking_df = pd.DataFrame(ranking_data)
        ranking_df.to_csv(f"{self.path}/RESULTS/SUMMARY/Feature_Correlations.csv", index=False)
        ranking_df.to_csv(f"{self.path}/RESULTS/SUMMARY/KMeans_Feature_Selection_Ranking.csv", index=False)

        print("\n--- Single-Observable KMeans Test (StandardScaler -> KMeans) ---")
        single_feature_results = []
        for idx, (feature_name, abs_corr, corr) in enumerate(correlations):
            series = candidate_df[feature_name]
            X_single = series.to_numpy(dtype=float).reshape(-1, 1)
            X_single_scaled = StandardScaler().fit_transform(X_single)

            kmeans_single = KMeans(init="random", n_clusters=2, n_init=20, max_iter=400, random_state=42)
            preds = kmeans_single.fit_predict(X_single_scaled)

            acc1 = np.mean(preds == labels)
            acc2 = np.mean((1 - preds) == labels)
            acc = max(acc1, acc2)

            print(
                f"{idx+1:2}. {feature_name:30} | Abs Corr: {abs_corr:.3f} | "
                f"Single-Feature Accuracy: {acc*100:.1f}%"
            )
            single_feature_results.append({
                "Rank": idx + 1,
                "Feature": feature_name,
                "Abs_Correlation": abs_corr,
                "Raw_Correlation": corr,
                "Single_Feature_Accuracy": acc * 100,
            })

        pd.DataFrame(single_feature_results).to_csv(
            f"{self.path}/RESULTS/SUMMARY/KMeans_Single_Feature_Results.csv",
            index=False,
        )

        # 3. Iterative Testing
        print("\n--- Iterative PCA KMeans Test (StandardScaler -> PCA -> KMeans) ---")
        best_acc = 0
        best_subset = []

        iteration_results = []
        current_features = []

        for i in range(len(correlations)):
            current_features.append(correlations[i][0])
            if len(current_features) < 2:
                continue

            try:
                X, usable_features = self._prepare_feature_matrix(
                    df,
                    current_features,
                    context=f"automated_feature_search_top_{len(current_features)}",
                )
            except ValueError as exc:
                print(f"[KMeans] Skipping top {len(current_features)} feature set: {exc}")
                continue

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            kmeans = KMeans(init="random", n_clusters=2, n_init=20, max_iter=400, random_state=42)
            preds = kmeans.fit_predict(X_pca)

            acc1 = np.mean(preds == labels)
            acc2 = np.mean((1 - preds) == labels)
            acc = max(acc1, acc2)

            print(
                f"Top {len(current_features):2} features | Used {len(usable_features):2} | "
                f"Accuracy: {acc*100:.1f}% | Added: {current_features[-1]}"
            )
            iteration_results.append({
                "Features_Count": len(current_features),
                "Usable_Features_Count": len(usable_features),
                "Added_Feature": current_features[-1],
                "Accuracy": acc * 100,
                "Feature_List": " | ".join(current_features),
                "Usable_Feature_List": " | ".join(usable_features),
            })

            if acc >= best_acc:
                best_acc = acc
                best_subset = list(usable_features)

        pd.DataFrame(iteration_results).to_csv(f"{self.path}/RESULTS/SUMMARY/KMeans_Iteration_Results.csv", index=False)

        print(f"\n=> HIGHEST ACCURACY ACHIEVED: {best_acc*100:.1f}% with Top {len(best_subset)} features:")
        print(best_subset)
        if len(best_subset) < 2:
            raise ValueError("No usable feature subset remained for automated KMeans after NaN preprocessing.")

        # 4. Generate Visualizations for Best Subset
        print("\nRunning full visualization pipeline for the automated optimal subset...")
        self.k_means_simulation(best_subset, suffix="_Automated_Optimal")

    def k_means_parameters(self):
        """Alternative classifier driven by small-molecule force-field parameters.

        Reads ``parameters.csv`` (per-biomolecule E, S, V, R, U, M), averages and
        class-reduces them, runs StandardScaler -> PCA(2) -> KMeans(2) on the
        (E, S, U, R) descriptors, and saves per-parameter bar plots plus the
        annotated PARAM_KMeans scatter under FIGURES/PROPERTIES. Standalone
        exploratory route, not part of the default pipeline run.
        """
        parameters = pd.read_csv("parameters.csv").loc[:, ["Biomolecule", "E", "S", "V", "R", "U", "M"]]

        ave_parameters = parameters.groupby("Biomolecule",observed=False).mean()

        df_annotate = self.df.set_index("Small Molecule ID")

        sm_dict = {"sg_X": "SG",
                   "ndsm_dmso": "ND1",
                   "ndsm_valeric_acid": "ND2",
                   "ndsm_ethylenediamine": "ND3",
                   "ndsm_propanedithiol": "ND4",
                   "ndsm_hexanediol": "ND5",
                   "ndsm_diethylaminopentane": "ND6",
                   "ndsm_aminoacridine": "ND7",
                   "ndsm_anthraquinone": "ND8",
                   "ndsm_acetylenapthacene": "ND9",
                   "ndsm_anacardic": "ND10",

                   "dsm_hydroxyquinoline": "D1",
                   "dsm_lipoamide": "D2",
                   "dsm_lipoic": "D3",
                   "dsm_lipoic_acid": "D3",
                   "dsm_dihydrolipoic": "D4",
                   "dsm_dihydrolipoic_acid": "D4",
                   "dsm_anisomycin": "D5",
                   "dsm_pararosaniline": "D6",
                   "dsm_pyrivinium": "D7",
                   "dsm_quinicrine": "D8",
                   "dsm_mitoxantrone": "D9",
                   "dsm_daunorubicin": "D10",
                   "DSM": "DSM",
                   "NDSM": "NDSM"
                   }

        ids = []
        dis = []
        for index, row in ave_parameters.iterrows():
            if "dsm" in index:
                ids.append(sm_dict[index])
                if "ndsm" in index:
                    dis.append(0)
                else:
                    dis.append(1)

        ave_parameters["Molecule ID"] = ids

        ave_parameters["Category"] = dis

        print(ave_parameters)

        mean_parameters = ave_parameters.loc[:, ["E", "S", "V", "U", "R", "M", "Category"]].groupby("Category",observed=False).mean()

        mean_parameters["Molecule ID"] = ["NDSM", "DSM"]

        mean_parameters["Category"] = [0, 1]

        print(mean_parameters)

        variables = ["E",
                     "S",
                     "V",
                     "U",
                     "R"]

        col_pall1 = sns.color_palette("rocket", n_colors=3)
        col_pall2 = sns.color_palette("Blues", n_colors=1)
        col_pall = [list(col_pall1[0]), list(col_pall1[2])]

        for var in range(len(variables)):
            sns.set_theme(style="ticks")
            sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
            plt.rc('axes', titlesize=10)  # fontsize of the axes title
            plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
            plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
            plt.rc('legend', fontsize=8)  # legend fontsize
            plt.rc('font', size=10)  # controls default text sizes
            plt.rc('axes', linewidth=2)

            fig, ax1 = plt.subplots(figsize=(3.6, 3.6))
            plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
            sns.barplot(ax=ax1, data=mean_parameters, x="Molecule ID", y=variables[var], hue="Molecule ID",
                        palette=col_pall,
                        dodge=False,
                        width=0.5, saturation=100)
            ax1.get_legend().remove()
            ax1.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in',
                            length=4,
                            width=2)
            ax1.set_xlabel("")
            ax1.set_ylabel("")
            plt.savefig("{}/FIGURES/PROPERTIES/{}.png".format(self.path, variables[var]), format="png", dpi=400)
            plt.close(fig)

        features = ave_parameters.loc[:, ["E",
                                          "S",
                                          "U",
                                          "R"]]

        features = np.array(features)

        labels = list((ave_parameters.loc[:, "Category"]))

        kmeans = KMeans(
            init="random",
            n_clusters=2,
            n_init=20,
            max_iter=400,
            random_state=None)

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        pca = PCA(n_components=2)
        pca_features = pd.DataFrame(pca.fit_transform(scaled_features),
                                    columns=["Principle Component 1", "Principle Component 2"])

        kmeans.fit(pca_features)

        print(f"Lowest SSE: {kmeans.inertia_:.6f}")
        ndsm_center = kmeans.cluster_centers_[0]
        dsm_center = kmeans.cluster_centers_[1]
        print(f"DSM Center: pca1={dsm_center[0]:.6f}; pca2={dsm_center[1]:.6f}")
        print(f"NDSM Center: pca1={ndsm_center[0]:.6f}; pca2={ndsm_center[1]:.6f}")

        print(list(kmeans.labels_))
        print(labels)
        pred_labs = []
        true_labs = []

        predictions = list(kmeans.labels_)
        acc1 = np.mean(np.array(predictions) == np.array(labels))
        acc2 = np.mean((1 - np.array(predictions)) == np.array(labels))
        if acc2 > acc1:
            predictions = [1 - p for p in predictions]

        for i in range(len(predictions)):
            if predictions[i] == 1:
                pred_labs.append("DSM")
            else:
                pred_labs.append("NDSM")
            if labels[i] == 1:
                true_labs.append("DSM")
            else:
                true_labs.append("NDSM")

        fpr = 0
        fnr = 0
        mcr = 0
        ccr = 0
        for i in range(len(predictions)):
            if predictions[i] != labels[i]:
                mcr += 1
                if predictions[i] == 1:
                    fpr += 1
                elif predictions[i] == 0:
                    fnr += 1
            else:
                ccr += 1

        fpr = fpr / (len(predictions) / 2) * 100
        fnr = fnr / (len(predictions) / 2) * 100
        mcr = mcr / len(predictions) * 100
        ccr = ccr / len(predictions) * 100

        print(f"False Positive Rate: {fpr:.6f}")
        print(f"False Negative Rate: {fnr:.6f}")
        print(f"Mis-classification Rate: {mcr:.6f}")
        print(f"Classification Rate: {ccr:.6f}")

        pca_features["Predicted Cluster"] = list(pred_labs)
        pca_features["True Label"] = list(true_labs)
        pca_features["Small Molecule ID"] = list()
        print(pca_features)

        score = silhouette_score(scaled_features, kmeans.labels_)
        print(f"Silhouette Score: {score:.6f}")

        sns.set_theme(style="ticks")
        sns.set_style('white')  # darkgrid, white grid, dark, white and ticks
        plt.rc('axes', titlesize=10)  # fontsize of the axes title
        plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
        plt.rc('legend', fontsize=8)  # legend fontsize
        plt.rc('font', size=10)  # controls default text sizes
        plt.rc('axes', linewidth=2)

        fig, ax1 = plt.subplots(figsize=(3.6, 3.6))
        col_pall1 = sns.color_palette("rocket", n_colors=3)

        sns.scatterplot(
            x=pca_features.loc[:, "Principle Component 1"],
            y=pca_features.loc[:, "Principle Component 2"],
            hue=pca_features.loc[:, "Predicted Cluster"],
            style=pca_features.loc[:, "True Label"],
            palette=col_pall1,
            s=80,
        )

        _xl = (-3, 3)
        _yl = (-2, 4)
        ax1.set_xlim(_xl)
        ax1.set_ylim(_yl)
        ax1.set_xticks(np.linspace(_xl[0], _xl[1], 7))
        ax1.set_yticks(np.linspace(_yl[0], _yl[1], 7))
        ax1.tick_params(left=True, right=True, top=True, bottom=True, labelbottom=True, direction='in', length=4,
                        width=2)
        ax1.legend(loc='upper right', frameon=False, fontsize=8)
        for i, txt in enumerate(df_annotate.index):
            if "ND" in txt:
                txt = "N" + str(txt[-1])
                if "10" in txt:
                    txt = "N10"
            ax1.annotate(
                txt,
                xy=(pca_features.iloc[i, 0], pca_features.iloc[i, 1]),
                xytext=(-1, 1),
                textcoords="offset points",
                ha="right",
                va="bottom",
            )
        plt.savefig("{}/FIGURES/PROPERTIES/PARAM_KMeans.png".format(self.path), format="png", dpi=400)
        plt.close(fig)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='KMeans clustering and violin plots from Quant_Data.csv')
    parser.add_argument('--path', required=True, help='Path to TEMP_XXX directory (e.g., TEMP_300)')
    parser.add_argument('--folder', required=True, help='Output folder prefix (e.g., CLASSIFY)')
    parser.add_argument('--T', type=int, required=True, help='Simulation temperature in Kelvin')
    parser.add_argument('--dt', type=int, required=True, help='Cluster time stride (ns)')
    parser.add_argument('--tmin', type=int, required=True, help='Start of analysis window (ns)')
    parser.add_argument('--tmax', type=int, required=True, help='End of analysis window (ns)')
    parser.add_argument('--plot-only', action='store_true', help='Accepted for pipeline plot-only mode; reads existing Quant_Data.csv')
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"Error: Path {args.path} does not exist")
        sys.exit(1)

    # Normalize to absolute path for consistency
    args.path = os.path.abspath(args.path)

    # Compute analysis root (where RESULTS/SUMMARY/Quant_Data.csv lives)
    analysis_root = os.path.join(
        args.path,
        f"{args.folder}_{args.T}_{args.dt}_{args.tmin}_{args.tmax}"
    )

    if not os.path.exists(analysis_root):
        print(f"Error: Analysis root {analysis_root} does not exist")
        sys.exit(1)

    quant_data_path = os.path.join(analysis_root, "RESULTS", "SUMMARY", "Quant_Data.csv")
    if not os.path.exists(quant_data_path):
        print(f"Error: Quant_Data.csv not found at {quant_data_path}")
        print("Make sure SYSTEM_ANALYSIS.py has been run first.")
        sys.exit(1)

    quant_df = pd.read_csv(quant_data_path)
    label_series = pd.to_numeric(quant_df.get("D_Binary"), errors="coerce")
    valid_mask = label_series.isin([0, 1])
    if valid_mask.sum() < 2 or label_series.loc[valid_mask].nunique() < 2:
        print("[KMeans] Skipping clustering: fewer than two labeled DSM/NDSM classes are available at this temperature")
        skip_path = os.path.join(analysis_root, "RESULTS", "SUMMARY", "KMeans_Skipped.txt")
        with open(skip_path, "w", encoding="utf-8") as fh:
            fh.write("Skipped KMeans because fewer than two labeled DSM/NDSM classes were available in Quant_Data.csv.\n")
        sys.exit(0)

    path = analysis_root

    try:
        classification = summary_plots(path)

        classification.clean_df()
        classification.plot_violin()
        classification.automated_feature_search()
        print("KMeans analysis completed successfully")
    except Exception as exc:
        print(f"KMeans analysis failed: {exc}")
        sys.exit(1)
