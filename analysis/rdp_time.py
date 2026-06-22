"""
Radial density profile (RDP) erf fitting for individual time windows.

Temporal companion to ``rdp.py`` used by ``time_analysis.py`` to fit the
condensate interface within a single time window. Unlike ``rdp.py``/``rdp_plot.py``,
the ``RDP`` class here receives the radial density profile as an in-memory
DataFrame (not a CSV path) and uses a data-adaptive initial guess so the fit
tracks BOTH a decreasing (normal condensate) and an increasing (dissolving
small molecule, depleted from the core) profile, allowing A < 0.

Interface model:  rho(r) = B - A * erf((r - R) / (sqrt(2) * W)).

Key inputs: density DataFrame (radius A, density mg/mL, sigma); PCA eigenvalues
CSV; cluster statistics CSV; temperature; label.
Key outputs (instance attributes): fitted (A,B,R,W) with SEs, dense/dilute
concentrations, transfer free energy ``dG``, surface tensions, and Rg.

Not runnable as a script; instantiate ``RDP`` from the temporal pipeline.
"""

import math

import numpy as np
import pandas as pd
import scipy.special
from scipy.optimize import curve_fit

class rdp:
    """Per-window RDP erf fit accepting an in-memory profile (rising or falling)."""

    def __init__(self, density, pca_file, cluster_file, T, init, label):
        """Run the fit/derivation chain for one time window.

        Args:
            density: DataFrame with columns (radius A, density mg/mL, sigma).
            pca_file: CSV path with PCA shape eigenvalues (l1, l2, l3).
            cluster_file: CSV path with cluster statistics from max_cluster.py.
            T: Temperature in Kelvin.
            init: Initial guess (A, B, R, W); note ``fit_density`` overrides it
                with a data-adaptive guess.
            label: Species label selecting the c_dilute reference concentration.
        """
        kb = 1.3806 * 10 ** (-23)

        density_profile = density
        self.distances = density_profile.iloc[0:, 0].tolist()
        densities = np.array(density_profile.iloc[0:, 1].tolist())
        self.densities = list(densities)
        self.sig = density_profile.iloc[0:, 2].tolist()
        self.errors = list(np.array(density_profile.iloc[0:, 2].tolist()))

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

        self.fit_x = np.linspace(0, self.distances[-1], 400)
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
        """Fit the erf model to this window's profile, storing (A,B,R,W) and SEs.

        Builds a data-adaptive initial guess (so both falling condensate and
        rising dissolving-SM profiles are tracked, permitting A < 0), runs a
        bounded sigma-weighted fit, then a legacy unweighted refit accepted only
        when its parameters remain physically sane.
        """
        x_data = np.asarray(self.distances, dtype=float)
        y_data = np.asarray(self.densities, dtype=float)
        sig = np.asarray(self.errors, dtype=float)

        # Data-adaptive initial guess. Works at any density scale (biopolymer
        # ~hundreds of mg/mL, small molecule ~0.3) and for BOTH a decreasing
        # condensate profile and an INCREASING one (a dissolving small molecule is
        # depleted from the condensate core, so its density rises with r). The fixed
        # biopolymer-scaled init=[80,80,200,80] otherwise drove the SM fit to a flat
        # line (R->0), so it never tracked the data.
        span = max(float(x_data[-1] - x_data[0]), 1.0)
        A0 = (y_data[0] - y_data[-1]) / 2.0       # >0 condensate (falls), <0 dissolving (rises)
        B0 = (np.nanmax(y_data) + np.nanmin(y_data)) / 2.0
        R0 = float(x_data[int(0.4 * len(x_data))])
        W0 = span / 8.0
        p0 = [A0, B0, R0, W0]

        # Allow A<0 so an increasing (dissolving-SM) profile can be fit rather than
        # collapsing to flat. B,R,W stay non-negative as before.
        bnds = [[-np.inf, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]]

        try:
            atmpt = True
            for i in sig:
                if i == 0 or not np.isfinite(i):
                    atmpt = False

            if atmpt:
                parameters, covariance = curve_fit(
                    f=self.ERF, xdata=x_data, ydata=y_data, p0=p0,
                    sigma=sig, bounds=bnds, maxfev=40000,
                )
            else:
                parameters, covariance = curve_fit(
                    f=self.ERF, xdata=x_data, ydata=y_data, p0=p0,
                    bounds=bnds, maxfev=40000,
                )

            # Final unweighted refit to stabilize parameters (legacy V2 behavior),
            # seeded from the bounded result and only accepted if it stays sane.
            try:
                p2, c2 = curve_fit(f=self.ERF, xdata=x_data, ydata=y_data, p0=parameters)
                if np.all(np.isfinite(p2)) and p2[1] >= 0 and p2[2] >= 0 and p2[3] > 0:
                    parameters, covariance = p2, c2
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

        term_sum_plus = np.square(da + db) + np.square(da + dc) + np.square(db + dc)
        term_sum_minus = np.square(da - db) + np.square(da - dc) + np.square(db - dc)
        mask = np.isfinite(term_sum_plus) & np.isfinite(term_sum_minus)
        term_sum_plus = term_sum_plus[mask]
        term_sum_minus = term_sum_minus[mask]
        n = term_sum_plus.size
        if n == 0:
            self.st_1 = np.nan
            self.st_2 = np.nan
            self.st = np.nan
            self.st_1_se = np.nan
            self.st_2_se = np.nan
            self.st_se = np.nan
            return

        ensemble1 = term_sum_plus.mean()
        ensemble2 = term_sum_minus.mean()

        if n > 1:
            ensemble1_se = term_sum_plus.std(ddof=1) / np.sqrt(n)
            ensemble2_se = term_sum_minus.std(ddof=1) / np.sqrt(n)
        else:
            ensemble1_se = np.nan
            ensemble2_se = np.nan
        ensemble1_var = ensemble1_se ** 2
        ensemble2_var = ensemble2_se ** 2

        K1 = (15 * kb * T) / (16 * np.pi)
        K2 = (45 * kb * T) / (16 * np.pi)
        eps = 1e-30
        if ensemble1 <= eps or ensemble2 <= eps:
            self.st_1 = np.nan
            self.st_2 = np.nan
            self.st = np.nan
            var_st1 = np.nan
            var_st2 = np.nan
        else:
            self.st_1 = K1 / ensemble1
            self.st_2 = K2 / ensemble2
            self.st = (self.st_1 + self.st_2) / 2.0
            d1 = -K1 / (ensemble1 ** 2)
            d2 = -K2 / (ensemble2 ** 2)
            var_st1 = (d1 ** 2) * ensemble1_var
            var_st2 = (d2 ** 2) * ensemble2_var

        self.st_1_se = np.sqrt(var_st1) if np.isfinite(var_st1) else np.nan
        self.st_2_se = np.sqrt(var_st2) if np.isfinite(var_st2) else np.nan
        self.st_se = np.sqrt((var_st1 + var_st2) / 4.0) if (np.isfinite(var_st1) and np.isfinite(var_st2)) else np.nan
