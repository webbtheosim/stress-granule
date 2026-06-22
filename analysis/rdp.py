"""
Radial density profile (RDP) sigmoid fitting and condensate thermodynamics.

Support module imported by the analysis pipeline (e.g. ``system_analysis.py``)
to characterize a single condensate from its precomputed radial density profile,
PCA shape eigenvalues, and cluster statistics. The ``RDP`` class fits the
error-function (erf) interface model and derives dense/dilute concentrations,
transfer free energy, surface tension, and radius of gyration with propagated
uncertainties. This is the primary variant with full (A,B) covariance
propagation; companion files are ``rdp_plot.py`` (plotting) and ``rdp_time.py``
(temporal windows).

Interface model:  rho(r) = B - A * erf((r - R) / (sqrt(2) * W))
    A : half the dense-minus-dilute density step (mg/mL)
    B : profile midpoint density (mg/mL)
    R : interface radius (Angstrom)
    W : interface width (Angstrom)

Key inputs (CSV paths): radial density profile (r, rho, sigma), PCA eigenvalues
(l1, l2, l3), and cluster statistics (droplet mass, external mass, Rg).

Key outputs (instance attributes): fitted (A,B,R,W) with standard errors;
c_dense / c_dilute (fit- and geometry-based, mg/mL); transfer free energy
``dG`` (J/mol) with SEM; surface tensions ``st_1``/``st_2``/``st`` (N/m, from
shape fluctuations with K1=15kT/16pi, K2=45kT/16pi); and Rg.

Not runnable as a script; instantiate ``RDP`` from the pipeline.
"""

import math
import warnings

import numpy as np
import pandas as pd
import scipy.special
from scipy.optimize import curve_fit, OptimizeWarning

class rdp:
    """Fit a condensate radial density profile and derive its thermodynamics."""

    def __init__(self, density_file, pca_file, cluster_file, T, init, label):
        """Load inputs and run the full fit/derivation chain for one condensate.

        Args:
            density_file: CSV path with columns (radius A, density mg/mL, sigma).
            pca_file: CSV path with PCA shape eigenvalues (columns l1, l2, l3).
            cluster_file: CSV path with cluster statistics from max_cluster.py.
            T: Temperature in Kelvin.
            init: Initial guess sequence for the erf parameters (A, B, R, W).
            label: Species label ("Protein", "RNA", ...) selecting the
                cytoplasmic reference concentration used for c_dilute.
        """
        kb = 1.3806 * 10 ** (-23)

        density_profile = pd.read_csv(density_file)
        self.distances = density_profile.iloc[:, 0].tolist()
        densities = np.array(density_profile.iloc[:, 1].tolist())
        self.densities = list(densities)
        self.sig = density_profile.iloc[:, 2].tolist()
        self.errors = list(np.array(density_profile.iloc[:, 2].tolist()))
        # Keep the full profile for plotting/output, but match the legacy
        # fitting convention by excluding the first radial bin from the fit.
        if len(self.distances) > 1:
            self.fit_distances = self.distances[1:]
            self.fit_densities = self.densities[1:]
            self.fit_sig = self.sig[1:]
        else:
            self.fit_distances = self.distances
            self.fit_densities = self.densities
            self.fit_sig = self.sig

        self.fit_x = []
        self.st1_std = 0
        self.st2_std = 0

        pca = pd.read_csv(pca_file)
        self.l1 = pca['l1'].tolist()
        self.l2 = pca['l2'].tolist()
        self.l3 = pca['l3'].tolist()

        cluster = pd.read_csv(cluster_file)
        self.cluster_mass = cluster["Mass of Largest Droplet (mg)"].mean()
        self.outer_mass = cluster["Mass of External Chains"].mean()
        self.radius = cluster["Largest Droplet Radius of Gyration"].mean()
        rg_sem = np.nan
        if "RG SEM" in cluster.columns:
            rg_sem = pd.to_numeric(cluster["RG SEM"], errors="coerce").iloc[0]
        if not np.isfinite(rg_sem):
            rg_series = pd.to_numeric(cluster["Largest Droplet Radius of Gyration"], errors="coerce")
            rg_sem = float(rg_series.sem()) if rg_series.notna().sum() > 1 else math.nan
        self.radius_se = float(rg_sem) if np.isfinite(rg_sem) else math.nan

        self.label = label

        self.fit_rho = []

        self.fit_A = 0.0
        self.se_A = 0

        self.fit_B = 0.0
        self.se_B = 0

        self.fit_R = 0.0
        self.se_R = 0

        self.fit_W = 0.0
        self.se_W = 0

        self.cov_AB = 0.0

        self.c_dilute_fit = 0.0
        self.c_dilute_calc = 0.0

        self.dG = 0.0
        self.se_dG = 0.0

        self.RG = 0
        self.se_RG = 0
        self.calc_rad()

        self.fit_density(T, init)

        fit_x_max = max(float(self.distances[-1]), 500.0)
        self.fit_x = np.linspace(0, fit_x_max, 500)
        self.fit_rho = self.ERF(self.fit_x, self.fit_A, self.fit_B, self.fit_R, self.fit_W)

        self.c_dense_fit = abs(self.fit_B + self.fit_A)
        # c_dense = A+B: var(A+B) = var(A) + var(B) + 2*cov(A,B)
        self.c_dense_fit_se = np.sqrt(max(0.0, self.se_A**2 + self.se_B**2 + 2.0 * self.cov_AB))

        self.c_dilute_fit = np.abs(self.fit_B - self.fit_A)
        # c_dilute = |B-A|: var(B-A) = var(A) + var(B) - 2*cov(A,B)
        self.c_dilute_fit_se = np.sqrt(max(0.0, self.se_A**2 + self.se_B**2 - 2.0 * self.cov_AB))

        self.c_dense_calc = 0
        self.c_dense_calc_se = 0
        self.calc_dense()

        self.c_dilute_calc = 0
        self.c_dilute_calc_se = 0
        self.calc_dilute()

        if self.c_dense_fit > 0 and self.c_dilute_fit > 0:
            self.dG = kb * T * np.log(self.c_dilute_fit / self.c_dense_fit) * 6.022 * 10 ** 23
            # Full propagation from the fitted (A,B) covariance, because
            # c_dense = A+B and c_dilute = |B-A| share the same A and B so
            # treating c_dense_fit_se and c_dilute_fit_se as independent
            # double-counts their shared contribution. Writing
            #   g(A,B) = ln(|B-A|) - ln(A+B)
            #   dg/dA  = -sign(B-A)/|B-A| - 1/(A+B)
            #   dg/dB  =  sign(B-A)/|B-A| - 1/(A+B)
            # Var(g) = (dg/dA)^2 Var(A) + (dg/dB)^2 Var(B) + 2 dg/dA dg/dB Cov(A,B)
            NA = 6.022 * 10 ** 23
            kT_NA = kb * T * NA
            diff = self.fit_B - self.fit_A
            sign = 1.0 if diff >= 0 else -1.0
            inv_dilute = sign / self.c_dilute_fit
            inv_dense = 1.0 / self.c_dense_fit
            dg_dA = -inv_dilute - inv_dense
            dg_dB = inv_dilute - inv_dense
            var_g = (
                dg_dA ** 2 * self.se_A ** 2
                + dg_dB ** 2 * self.se_B ** 2
                + 2.0 * dg_dA * dg_dB * self.cov_AB
            )
            var_g = max(0.0, float(var_g))
            self.se_dG = kT_NA * np.sqrt(var_g)

        self.st_1 = 0
        self.st_1_se = 0
        self.st_2 = 0
        self.st_2_se = 0
        self.surface_tension(T)

    def ERF(self, r, A, B, R, W):
        """Error-function interface model ``rho(r) = B - A*erf((r-R)/(sqrt(2)*W))``
        evaluated at radius ``r`` (Angstrom); returns density in mg/mL."""
        y = B - A * scipy.special.erf((r - R) / (np.sqrt(2) * W))
        return y

    def calc_dense(self):
        """Set the geometry-based dense concentration ``c_dense_calc`` (mg/mL) as
        droplet mass over the sphere volume at radius R+W, with propagated SEM."""
        # Legacy convention: r = R + W (Angstrom -> cm); volume in cm^3 = mL
        r = (self.fit_R + self.fit_W) * 1e-8
        r_std = np.sqrt((1e-8) ** 2 * self.se_R ** 2 + (1e-8) ** 2 * self.se_W ** 2)
        vol = (4.0 / 3.0) * np.pi * (r ** 3)
        vol_std = np.sqrt((4 * np.pi * r ** 2) ** 2 * r_std ** 2)
        self.c_dense_calc = (self.cluster_mass) / vol
        self.c_dense_calc_se = np.sqrt(((-self.cluster_mass) / (vol ** 2)) ** 2 * vol_std ** 2)

    def calc_dilute(self):
        """Set the geometry-based dilute concentration ``c_dilute_calc`` (mg/mL)
        as external-chain mass over the volume outside the droplet, with SEM. The
        total system volume is inferred from the label-specific cytoplasmic
        reference density ``c_cyt``."""
        c_cyt = 120
        if self.label == "Protein":
            c_cyt = 108
        elif self.label == "RNA":
            c_cyt = 12
        r = (self.fit_R + self.fit_W) * 1e-8
        r_std = np.sqrt((1e-8) ** 2 * self.se_R ** 2 + (1e-8) ** 2 * self.se_W ** 2)

        vol_clus = (4.0 / 3.0) * np.pi * (r ** 3)
        vol_sys = np.mean(self.outer_mass + self.cluster_mass) / c_cyt
        vol = np.abs(vol_sys - vol_clus)
        vol_std = np.sqrt(((-4 * np.pi * r ** 2) ** 2 * r_std ** 2))

        self.c_dilute_calc = np.mean(self.outer_mass) / vol
        self.c_dilute_calc_se = np.sqrt((-(self.outer_mass) / (vol ** 2)) ** 2 * vol_std ** 2)

    def calc_rad(self):
        """Copy the cluster radius of gyration (and its SEM) into ``RG``/``se_RG``."""
        self.RG = self.radius
        self.se_RG = self.radius_se

    def fit_density(self, T, init):
        """Fit the erf model to the radial density profile, storing (A,B,R,W) and
        their standard errors plus the A-B covariance.

        Uses a bounded, sigma-weighted (with a tail sigma floor) nonlinear
        least-squares fit, followed by the legacy unweighted unbounded refit
        seeded from ``init`` that supersedes the bounded result when it succeeds.
        """
        x_data = np.asarray([x * 1 for x in self.fit_distances], dtype=float)
        y_data = np.asarray([x * 1 for x in self.fit_densities], dtype=float)
        sig_raw = np.asarray([x * 1 for x in self.fit_sig], dtype=float)
        bnds = [[0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]]

        # Sigma floor: the dilute tail produces vanishingly small SEMs
        # (~1e-5 mg/mL) that otherwise dominate chi^2 and collapse the
        # weighted fit to R=0, W>>interface. Floor at max(1e-3 * max(sig),
        # 1e-3 * max(rho)) so tail points get a physically reasonable weight.
        sig_floor = max(1e-3 * float(np.nanmax(sig_raw)) if np.any(np.isfinite(sig_raw)) else 0.0,
                         1e-3 * float(np.nanmax(np.abs(y_data))) if np.any(np.isfinite(y_data)) else 0.0)
        sig = np.where(np.isfinite(sig_raw) & (sig_raw > 0), np.maximum(sig_raw, sig_floor), np.nan)
        atmpt = bool(np.all(np.isfinite(sig)))

        try:
            # Stage 1: bounded nonlinear LS (weighted when sigma is valid).
            parameters = None
            covariance = None
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", OptimizeWarning)
                    if atmpt:
                        parameters, covariance = curve_fit(
                            f=self.ERF,
                            xdata=x_data,
                            ydata=y_data,
                            p0=init,
                            sigma=sig,
                            bounds=bnds,
                            maxfev=40000,
                        )
                    else:
                        parameters, covariance = curve_fit(
                            f=self.ERF,
                            xdata=x_data,
                            ydata=y_data,
                            p0=init,
                            bounds=bnds,
                            maxfev=40000,
                        )
            except Exception:
                parameters, covariance = None, None

            # Stage 2 (V1 legacy behavior): unweighted unbounded refit seeded
            # from init. If this succeeds it supersedes the bounded result;
            # otherwise retain the bounded one. The bounded fit can collapse
            # to a degenerate R~0, W>>interface when the tail weights are
            # pathological, so we always attempt the unbounded refit.
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", OptimizeWarning)
                    params_u, cov_u = curve_fit(
                        f=self.ERF,
                        xdata=x_data,
                        ydata=y_data,
                        p0=init,
                        maxfev=40000,
                    )
                if np.all(np.isfinite(params_u)) and np.all(np.isfinite(cov_u)):
                    parameters, covariance = params_u, cov_u
            except Exception:
                pass

            if parameters is None or covariance is None:
                raise RuntimeError("ERF fit failed at both bounded and unbounded stages")

            self.fit_A = parameters[0]
            self.fit_B = parameters[1]
            self.fit_R = parameters[2]
            self.fit_W = parameters[3]

            std = np.sqrt((np.diag(covariance)))
            self.se_A = std[0]
            self.se_B = std[1]
            self.se_R = std[2]
            self.se_W = std[3]
            self.cov_AB = covariance[0, 1]

        except Exception:
            pass

    def surface_tension(self, T):
        """Compute interfacial tensions from PCA shape fluctuations of the droplet.

        Maps per-frame PCA eigenvalues to semi-axis deviations about the
        spherical radius R, forms the two shape-mode ensemble averages, and sets
        ``st_1`` = K1/<sum (da+db)^2>, ``st_2`` = K2/<sum (da-db)^2> and their
        mean ``st`` (all in N/m), with SEMs propagated from the between-frame
        spread. K1 = 15kT/16pi, K2 = 45kT/16pi.
        """
        kb = 1.3806 * 10 ** (-23)

        R = self.fit_R * 10 ** (-10)
        l1 = self.l1
        l2 = self.l2
        l3 = self.l3

        a = []
        b = []
        c = []

        for i in range(len(l1)):
            L1 = l1[i] * 10 ** (-10)
            L2 = l2[i] * 10 ** (-10)
            L3 = l3[i] * 10 ** (-10)
            if not (np.isfinite(L1) and np.isfinite(L2) and np.isfinite(L3)):
                continue
            if L1 <= 0 or L2 <= 0 or L3 <= 0:
                continue
            try:
                a.append((R * L1 ** (1 / 3)) / ((L2 * L3) ** (1 / 6)))
                b.append((R * L2 ** (1 / 3)) / ((L1 * L3) ** (1 / 6)))
                c.append((R * L3 ** (1 / 3)) / ((L1 * L2) ** (1 / 6)))
            except Exception:
                continue

        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        da = a - R
        db = b - R
        dc = c - R

        # Build per-frame ensemble measures to retain correlations between terms
        term_sum_plus = np.square(da + db) + np.square(da + dc) + np.square(db + dc)
        term_sum_minus = np.square(da - db) + np.square(da - dc) + np.square(db - dc)

        # Filter invalid values
        mask = np.isfinite(term_sum_plus) & np.isfinite(term_sum_minus)
        term_sum_plus = term_sum_plus[mask]
        term_sum_minus = term_sum_minus[mask]
        n = term_sum_plus.size
        if term_sum_plus.size == 0 or term_sum_minus.size == 0:
            self.st_1 = np.nan
            self.st_2 = np.nan
            self.st = np.nan
            self.st_1_se = np.nan
            self.st_2_se = np.nan
            self.st_se = np.nan
            return
        ensemble1 = term_sum_plus.mean()
        ensemble2 = term_sum_minus.mean()

        # Standard errors (of the mean) for ensemble measures.
        # NOTE: frames are correlated in time, and the three shape-mode
        # contributions (da+db, da+dc, db+dc) are not statistically
        # independent modes either. The SEM computed here therefore
        # captures between-frame spread only -- it is NOT a rigorous
        # absolute uncertainty. Use gamma as a comparative interfacial
        # descriptor across conditions; absolute uncertainty requires
        # the correlated-pipeline Flyvbjerg-Petersen superblock SEM
        # applied per-window to the gamma time series.
        if n > 1:
            ensemble1_se = term_sum_plus.std(ddof=1) / np.sqrt(n)
            ensemble2_se = term_sum_minus.std(ddof=1) / np.sqrt(n)
        else:
            ensemble1_se = np.nan
            ensemble2_se = np.nan
        ensemble1_var = ensemble1_se ** 2
        ensemble2_var = ensemble2_se ** 2

        # Interfacial tensions
        K1 = (15 * kb * T) / (16 * np.pi)
        K2 = (45 * kb * T) / (16 * np.pi)
        eps = 1e-30
        if ensemble1 <= eps or ensemble2 <= eps:
            self.st_1 = np.nan
            self.st_2 = np.nan
            self.st = np.nan
        else:
            self.st_1 = K1 / ensemble1
            self.st_2 = K2 / ensemble2
            self.st = (self.st_1 + self.st_2) / 2.0

        # Error propagation: var(f(ensemble)) ≈ (df/dx)^2 var(x)
        if ensemble1 > eps:
            d1 = -K1 / (ensemble1 ** 2)
            var_st1 = (d1 ** 2) * ensemble1_var
        else:
            var_st1 = np.nan
        if ensemble2 > eps:
            d2 = -K2 / (ensemble2 ** 2)
            var_st2 = (d2 ** 2) * ensemble2_var
        else:
            var_st2 = np.nan

        # Store standard errors directly for clarity
        self.st_1_se = np.sqrt(var_st1) if np.isfinite(var_st1) else np.nan
        self.st_2_se = np.sqrt(var_st2) if np.isfinite(var_st2) else np.nan
        # For mean of two (assume independence): var(mean) = (var1 + var2) / 4
        self.st_se = np.sqrt((var_st1 + var_st2) / 4.0) if (np.isfinite(var_st1) and np.isfinite(var_st2)) else np.nan
