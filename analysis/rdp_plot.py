"""
Radial density profile (RDP) fitting variant used for plotting.

Plotting-oriented companion to ``rdp.py``. The ``RDP`` class here mirrors the
core erf-interface fit and thermodynamic derivations but adds an optional
``normalize`` flag (rescale densities by their maximum, matching legacy
``RDP_NORMALIZE`` behaviour) so overlay plots can be drawn on a common scale.
Imported by the figure-generation code rather than run as a script.

Interface model:  rho(r) = B - A * erf((r - R) / (sqrt(2) * W)).

Key inputs (CSV paths): radial density profile, PCA eigenvalues, cluster stats.
Key outputs (instance attributes): fitted (A,B,R,W) with SEs, dense/dilute
concentrations, transfer free energy ``dG``, surface tensions, Rg, and the
``normalization_factor`` applied to the densities.
"""

import math

import numpy as np
import pandas as pd
import scipy.special
from scipy.optimize import curve_fit

class rdp:
    """RDP erf fit with optional density normalization for overlay plotting."""

    def __init__(self, density_file, pca_file, cluster_file, T, init, label, normalize=False):
        """Fit radial density profiles and derive thermodynamic properties.

        Parameters
        ----------
        density_file : str
            CSV containing radial density profile.
        pca_file : str
            PCA eigenvalues CSV (three columns).
        cluster_file : str
            Cluster statistics CSV produced by max_cluster.py.
        T : float
            Temperature in Kelvin.
        init : sequence[float]
            Initial guess for (A, B, R, W) parameters.
        label : str
            Human readable label used in logs/plots.
        normalize : bool, optional
            When True, densities are rescaled by their maximum value. Matches
            the legacy `RDP_NORMALIZE` behaviour. False keeps raw densities as
            previously provided by `RDP_PLOT`.
        """
        kb = 1.3806 * 10 ** (-23)

        self.normalize = bool(normalize)

        density_profile = pd.read_csv(density_file)
        self.distances = density_profile.iloc[:, 0].tolist()

        raw_densities = np.array(density_profile.iloc[:, 1].tolist(), dtype=float)
        raw_sig = np.array(density_profile.iloc[:, 2].tolist(), dtype=float)

        scale = 1.0
        if self.normalize:
            max_dens = np.max(np.abs(raw_densities))
            if max_dens > 0:
                scale = max_dens
            else:
                # Avoid divide-by-zero; fall back to raw magnitudes
                scale = 1.0

        self.normalization_factor = scale

        self.densities = (raw_densities / scale).tolist()
        self.sig = raw_sig.tolist()
        self.errors = (raw_sig / scale).tolist()

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
        self.c_dense_fit_se = np.sqrt(self.se_A**2 + self.se_B**2)

        self.c_dilute_fit = np.abs(self.fit_B - self.fit_A)
        self.c_dilute_fit_se = np.sqrt(self.se_A**2 + self.se_B**2)

        self.c_dense_calc = 0
        self.c_dense_calc_se = 0
        self.calc_dense()

        self.c_dilute_calc = 0
        self.c_dilute_calc_se = 0
        self.calc_dilute()

        if self.c_dense_fit > 0:
            self.dG = kb * T * np.log(self.c_dilute_fit / self.c_dense_fit) * 6.022 * 10 ** 23
            # Error propagation: ∂ΔG/∂c_dilute = kT/c_dilute; ∂ΔG/∂c_dense = -kT/c_dense (squared terms both positive)
            self.se_dG = np.sqrt((kb * T / self.c_dilute_fit * 6.022 * 10 ** 23) ** 2 * self.c_dilute_fit_se ** 2 + (
                    kb * T / self.c_dense_fit * 6.022 * 10 ** 23) ** 2 * self.c_dense_fit_se ** 2)

        self.st_1 = 0
        self.st_1_se = 0
        self.st_2 = 0
        self.st_2_se = 0
        self.surface_tension(T)

        print("A: " + str(self.fit_A) + "\tStandard Deviation: " + str(self.se_A))
        print("B: " + str(self.fit_B) + "\tStandard Deviation: " + str(self.se_B))
        print("R: " + str(self.fit_R) + "\tStandard Deviation: " + str(self.se_R))
        print("W: " + str(self.fit_W) + "\tStandard Deviation: " + str(self.se_W))

        print("c_dense_fit: " + str(self.c_dense_fit) + "\tStandard Deviation: " + str(self.c_dense_fit_se))
        print("c_dense_calc: " + str(self.c_dense_calc) + "\tStandard Deviation: " + str(self.c_dense_calc_se))
        print("c_dilute_fit: " + str(self.c_dilute_fit) + "\tStandard Deviation: " + str(self.c_dilute_fit_se))
        print("c_dilute_calc: " + str(self.c_dilute_calc) + "\tStandard Deviation: " + str(self.c_dilute_calc_se))

        print("dG: " + str(self.dG) + "\tStandard Deviation: " + str(self.se_dG))

        print("st_1: " + str(self.st_1) + "\tStandard Deviation: " + str(self.st_1_se))
        print("st_2: " + str(self.st_2) + "\tStandard Deviation: " + str(self.st_2_se))
        print("st: " + str(self.st) + "\tStandard Deviation: " + str(self.st_se))

        print("Radius Gyration: " + str(self.RG) + "\tStandard Deviation: " + str(self.se_RG))

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
        as external-chain mass over the volume outside the droplet, with SEM,
        using the label-specific cytoplasmic reference density ``c_cyt``."""
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
        """Fit the erf model to the (optionally normalized) density profile,
        storing (A,B,R,W) and standard errors. Uses a bounded sigma-weighted fit
        followed by the legacy unweighted refit seeded from ``init``."""
        x_data = [x * 1 for x in self.distances]
        y_data = [x * 1 for x in self.densities]
        sig = [x * 1 for x in self.errors]
        bnds = [[0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]]

        try:
            atmpt = True
            for i in sig:
                if i == 0 or not np.isfinite(i):
                    atmpt = False

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

            # Final unweighted refit to stabilize parameters (legacy V2 behavior).
            try:
                parameters, covariance = curve_fit(
                    f=self.ERF,
                    xdata=x_data,
                    ydata=y_data,
                    p0=init,
                )
            except Exception:
                pass

            self.fit_A = parameters[0]
            self.fit_B = parameters[1]
            self.fit_R = parameters[2]
            self.fit_W = parameters[3]

            std = np.sqrt((np.diag(covariance)))
            self.se_A = std[0]
            self.se_B = std[1]
            self.se_R = std[2]
            self.se_W = std[3]

        except Exception:
            print("Failed")

    def surface_tension(self, T):
        """Compute interfacial tensions from PCA shape fluctuations: ``st_1`` =
        K1/<sum (da+db)^2>, ``st_2`` = K2/<sum (da-db)^2>, and their mean ``st``
        (N/m), with SEMs from the between-frame spread. K1=15kT/16pi, K2=45kT/16pi."""
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

        n = len(da)
        term_sum_plus = np.square(da + db) + np.square(da + dc) + np.square(db + dc)
        term_sum_minus = np.square(da - db) + np.square(da - dc) + np.square(db - dc)

        # Filter invalid entries
        mask = np.isfinite(term_sum_plus) & np.isfinite(term_sum_minus)
        term_sum_plus = term_sum_plus[mask]
        term_sum_minus = term_sum_minus[mask]
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

        ensemble1_se = term_sum_plus.std(ddof=1) / np.sqrt(max(term_sum_plus.size, 1))
        ensemble2_se = term_sum_minus.std(ddof=1) / np.sqrt(max(term_sum_minus.size, 1))
        ensemble1_var = ensemble1_se ** 2
        ensemble2_var = ensemble2_se ** 2

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

        self.st_1_se = np.sqrt(var_st1) if np.isfinite(var_st1) else np.nan
        self.st_2_se = np.sqrt(var_st2) if np.isfinite(var_st2) else np.nan
        self.st_se = np.sqrt((var_st1 + var_st2) / 4.0) if (np.isfinite(var_st1) and np.isfinite(var_st2)) else np.nan
