"""
Green-Kubo shear viscosity from the condensate stress autocorrelation.

Support module imported by the thermodynamics/dynamics pipeline
(``system_analysis.py``) and by ``rdp.py``. Given the off-diagonal stress
tensor of the condensate, it builds bootstrapped autocorrelation functions (via
``acf.py``), fits a multi-mode Maxwell relaxation model, and integrates the
Green-Kubo relation to obtain the viscosity.

Green-Kubo prefactor: ``k_conv = dt_unit * V / (k_B * T)`` (dimensionally Pa*s),
where ``V`` is the droplet volume in m^3, ``dt_unit`` the stress sample interval
in seconds, and ``T`` the temperature in Kelvin. Two estimates are produced:
    eta_raw  : cumulative trapezoidal integral of the ACF * k_conv (Pa*s)
    eta_theo : sum(amplitude * tau) of the fitted Maxwell modes * acf0 * k_conv

Key inputs:
    Pxyz : array (n_frames, 3) of the three off-diagonal stress components,
           pre-converted to Pa, with the condensate-volume normalization applied.

Key outputs (per run): bootstrap mean and SEM of eta_raw and eta_theo (Pa*s),
and fitted Maxwell amplitudes/relaxation times.

Runnable directly for a single test system (see ``__main__``); ordinarily used
as an imported helper.
"""

import numpy as np
import scipy.integrate as spi
from scipy import interpolate
from scipy.optimize import curve_fit
import math
from acf import acf
from tqdm import tqdm

class visc:
    """Green-Kubo viscosity estimator from a condensate stress-tensor series."""

    def __init__(self, path, Pxyz):
        """Store the output ``path`` and the off-diagonal stress array ``Pxyz``
        (shape ``(n_frames, 3)``, already converted to Pa)."""
        self.Pxyz = Pxyz
        self.path = path

    def run_vsc(self, vol_sys, name, segments, iterations, n_boot, dt_unit, n_point, n_tau, T, seed=0):
        """Bootstrap the Green-Kubo viscosity.

        Args:
            vol_sys: Droplet volume in m^3 (enters the Green-Kubo prefactor).
            name: System tag (retained for call-signature compatibility).
            segments: Number of contiguous ACF segments for bootstrapping.
            iterations: Bootstrap iterations used to build the ACF ensemble.
            n_boot: Number of bootstrap viscosity estimates.
            dt_unit: Stress-sample time interval in seconds.
            n_point: Number of log-spaced points on the ACF fitting grid.
            n_tau: Maximum number of Maxwell relaxation modes attempted.
            T: Temperature in Kelvin.
            seed: RNG seed for reproducible bootstrapping.

        Returns:
            Tuple ``(eta_raw_mean, eta_raw_sem, eta_theo_mean, eta_theo_sem,
            amp_opts, tau_opts, y_log0s, dt_log)`` with viscosities in Pa*s.
        """
        acf_calc = acf()
        dt, acfs_bootstrap = acf_calc.get_boot_data(self.Pxyz, segments, iterations, seed=seed)

        print('\nCalculating the Viscosity from the Green-Kubo Relation')

        eta_raw, eta_theo, amp_opts, tau_opts, y_log0s, dt_log, dt, acf = self.get_visco(dt=dt, acfs=acfs_bootstrap, vol=vol_sys, dt_unit=dt_unit,
                                                                                           n_point=n_point, n_tau=n_tau, n_boot=n_boot, T=T, seed=seed)

        # Statistics
        # Raw: include all finite values (do not clip negatives)
        eta_raw_mean, eta_raw_sem = self._mean_sem(eta_raw, min_val=None)
        # Fit/Theo: keep only positive, finite estimates (failed fits excluded)
        theo_success = np.isfinite(eta_theo) & (eta_theo > 0)
        eta_theo_mean, eta_theo_sem = self._mean_sem(eta_theo, min_val=0)

        print("Viscosity GK (Eta_Raw): " + str(eta_raw_mean) + " ± " + str(eta_raw_sem) + " Pa s")
        print("Viscosity GK (Eta_Theo): " + str(eta_theo_mean) + " ± " + str(eta_theo_sem) + " Pa s")
        print(f"Bootstrap success (theo): {theo_success.sum()}/{n_boot}")

        return eta_raw_mean, eta_raw_sem, eta_theo_mean, eta_theo_sem, amp_opts, tau_opts, y_log0s, dt_log

    def maxwell_model(self, x, *params):
        """Multi-mode Maxwell relaxation model: sum of ``a_i * exp(-x / b_i)``
        over (amplitude ``a``, relaxation time ``b``) parameter pairs."""
        y = np.zeros_like(x)

        for i in range(0, len(params), 2):
            a = params[i]
            b = params[i + 1]

            y += a * np.exp(- x / b)

        return y

    def estimate_single_acf(self, dt, acf, vol, dt_unit, n_point, n_tau, T):
        """Estimate eta_raw and eta_theo (Pa*s) for one (non-bootstrapped) ACF.

        Integrates the ACF directly (eta_raw) and fits 1..``n_tau`` Maxwell modes
        to the normalized ACF on a log-spaced grid, taking the best fit for
        eta_theo. Returns ``(eta_raw, eta_theo, amp_opt, tau_opt, acf_log0,
        dt_log, diagnostics)``; viscosities are NaN when the fit/integral fails.
        """
        # Green-Kubo prefactor: V/(k_B T)
        kb = 1.3806e-23
        k_conv = (dt_unit * vol) / (kb * T)
        dt = np.asarray(dt, dtype=float)
        acf = np.asarray(acf, dtype=float)
        diagnostics = {
            "fit_mae": np.nan,
            "maxwell_mode_count": 0,
            "acf0": np.nan,
            "theoretical_success": False,
            "raw_integral": np.array([], dtype=float),
            "raw_integral_dt": np.array([], dtype=float),
        }
        if dt.size == 0 or acf.size == 0 or dt.size != acf.size:
            return np.nan, np.nan, np.array([]), np.array([]), np.nan, np.array([]), diagnostics

        dt_positive = dt[dt > 0]
        if dt_positive.size == 0:
            return np.nan, np.nan, np.array([]), np.array([]), np.nan, np.array([]), diagnostics

        dt_min_pos = float(np.min(dt_positive))
        dt_max = float(np.max(dt)) if len(dt) else dt_min_pos * 10
        dt_log = np.logspace(np.log10(max(dt_min_pos, 1e-6)), np.log10(max(dt_max, dt_min_pos * 10)), n_point)
        acf0 = float(acf[0]) if len(acf) > 0 else np.nan
        diagnostics["acf0"] = acf0

        try:
            integral_raw = spi.cumulative_trapezoid(acf, dt) * k_conv
            eta_raw = float(integral_raw[-1]) if len(integral_raw) > 0 else np.nan
            diagnostics["raw_integral"] = np.asarray(integral_raw, dtype=float)
            diagnostics["raw_integral_dt"] = np.asarray(dt[1:], dtype=float)
        except Exception:
            eta_raw = np.nan

        if not np.isfinite(acf0) or acf0 == 0.0:
            return eta_raw, np.nan, np.array([]), np.array([]), np.nan, dt_log, diagnostics

        try:
            acf_spline = interpolate.InterpolatedUnivariateSpline(dt, acf)
            acf_log = acf_spline(dt_log)
            acf_norm = acf_log / acf0
        except Exception:
            return eta_raw, np.nan, np.array([]), np.array([]), np.nan, dt_log, diagnostics

        errs = []
        tau_seq = []
        amp_seq = []
        for n_tau_temp in range(1, n_tau + 1):
            amps0 = np.full(n_tau_temp, 1.0 / max(n_tau_temp, 1))
            taus0 = np.logspace(np.log10(dt_log[0]), np.log10(dt_log[-1]), n_tau_temp)
            initial_params = np.empty(n_tau_temp * 2)
            initial_params[0::2] = amps0
            initial_params[1::2] = taus0
            lb = np.empty_like(initial_params)
            ub = np.empty_like(initial_params)
            lb[0::2] = 0.0
            ub[0::2] = 2.0
            lb[1::2] = dt_log[0] / 10.0
            ub[1::2] = dt_log[-1] * 10.0
            try:
                params, _ = curve_fit(
                    self.maxwell_model,
                    dt_log,
                    acf_norm,
                    p0=initial_params,
                    bounds=(lb, ub),
                    maxfev=20000,
                )
                fit_norm = self.maxwell_model(dt_log, *params)
                err = np.mean(np.abs(fit_norm - (acf_log / acf0)))
                errs.append(err)
                amp_seq.append(params[0::2])
                tau_seq.append(params[1::2])
            except Exception:
                continue

        if len(errs) == 0:
            return eta_raw, np.nan, np.array([]), np.array([]), float(acf_log[0]), dt_log, diagnostics

        idx = int(np.argmin(errs))
        amp_opt = amp_seq[idx]
        tau_opt = tau_seq[idx]
        eta_theo = float(np.sum(amp_opt * tau_opt) * acf0 * k_conv)
        diagnostics["fit_mae"] = float(errs[idx]) if np.isfinite(errs[idx]) else np.nan
        diagnostics["maxwell_mode_count"] = int(len(amp_opt))
        diagnostics["theoretical_success"] = bool(np.isfinite(eta_theo) and eta_theo > 0)
        return eta_raw, eta_theo, amp_opt, tau_opt, float(acf_log[0]), dt_log, diagnostics

    def get_visco(self, dt, acfs, vol, dt_unit, n_point, n_tau, n_boot, T, seed=0):
        """Compute per-bootstrap eta_raw and eta_theo over a pool of ACFs.

        Draws ``n_boot`` ACFs (with replacement) from ``acfs``, integrating each
        for eta_raw and fitting the best Maxwell model for eta_theo. Returns
        ``(eta_raw, eta_theo, amp_opts, tau_opts, y_log0s, dt_log, dt, acf)``
        with the per-bootstrap viscosity arrays in Pa*s.
        """
        # Green-Kubo prefactor: V/(k_B T)
        kb = 1.3806e-23
        k_conv = (dt_unit * vol) / (kb * T)
        # Build a positive time grid for fitting (exclude dt=0)
        dt_min_pos = float(np.min(np.asarray(dt)[np.asarray(dt) > 0])) if np.any(np.asarray(dt) > 0) else 1.0
        dt_max = float(np.max(dt)) if len(dt) else dt_min_pos * 10
        dt_log = np.logspace(np.log10(max(dt_min_pos, 1e-6)), np.log10(max(dt_max, dt_min_pos*10)), n_point)
        eta_raw = np.zeros(n_boot)
        eta_theo = np.zeros(n_boot)
        rng = np.random.RandomState(seed)
        y_log0s = np.zeros(n_boot)
        amp_opts = []
        tau_opts = []

        # Guard: if no bootstrap ACFs were produced, return NaN-filled arrays
        # rather than crashing in rng.randint(0, 0).
        n_acfs = int(len(acfs)) if acfs is not None else 0
        if n_acfs == 0 or n_boot <= 0:
            eta_raw = np.full(max(n_boot, 0), np.nan, dtype=float)
            eta_theo = np.full(max(n_boot, 0), np.nan, dtype=float)
            y_log0s = np.full(max(n_boot, 0), np.nan, dtype=float)
            return eta_raw, eta_theo, amp_opts, tau_opts, y_log0s, dt_log, dt, np.array([])

        for i in tqdm(range(n_boot), desc="Running Viscosity Bootstrap", unit="iterations"):
            n = int(rng.randint(0, n_acfs))
            acf = acfs[n]
            acf0 = float(acf[0]) if len(acf) > 0 else np.nan
            if not np.isfinite(acf0) or acf0 == 0.0:
                # Degenerate bootstrap sample; skip fit/theo, still compute raw
                acf_spline = interpolate.InterpolatedUnivariateSpline(dt, acf)
                acf_log = acf_spline(dt_log)
                acf_norm = acf_log  # placeholder, won't be used
            else:
                acf_spline = interpolate.InterpolatedUnivariateSpline(dt, acf)
                acf_log = acf_spline(dt_log)
                # Normalize with true t=0 amplitude, not the first positive time
                acf_norm = acf_log / acf0

            errs = []
            tau_seq = []
            amp_seq = []
            for n_tau_temp in range(1, n_tau+1):
                # Robust initialization: equal amplitudes, taus log-spaced over fit grid
                amps0 = np.full(n_tau_temp, 1.0 / max(n_tau_temp, 1))
                taus0 = np.logspace(np.log10(dt_log[0]), np.log10(dt_log[-1]), n_tau_temp)
                initial_params = np.empty(n_tau_temp * 2)
                initial_params[0::2] = amps0
                initial_params[1::2] = taus0
                # Bounds: amplitudes in [0, 2], taus in [dt_min/10, dt_max*10]
                lb = np.empty_like(initial_params)
                ub = np.empty_like(initial_params)
                lb[0::2] = 0.0
                ub[0::2] = 2.0
                lb[1::2] = dt_log[0] / 10.0
                ub[1::2] = dt_log[-1] * 10.0
                try:
                    params, _ = curve_fit(self.maxwell_model,
                                          dt_log,
                                          acf_norm,
                                          p0=initial_params,
                                          bounds=(lb, ub),
                                          maxfev=20000)
                    fit_norm = self.maxwell_model(dt_log, *params)
                    err = np.mean(np.abs(fit_norm - (acf_log / acf0)))
                    errs.append(err)
                    amp_seq.append(params[0::2])
                    tau_seq.append(params[1::2])
                except Exception:
                    continue

            if len(errs) > 0:
                idx = int(np.argmin(errs))
                amp_opt = amp_seq[idx]
                tau_opt = tau_seq[idx]
                amp_opts.append(amp_opt)
                tau_opts.append(tau_opt)
                eta_theo[i] = np.sum(amp_opt * tau_opt) * acf0 * k_conv
            else:
                eta_theo[i] = np.nan
            # Always compute raw
            integral_raw = spi.cumulative_trapezoid(acf, dt) * k_conv
            eta_raw[i] = integral_raw[-1]
            y_log0s[i] = acf_log[0]


        return eta_raw, eta_theo, amp_opts, tau_opts, y_log0s, dt_log, dt, acf

    @staticmethod
    def _mean_sem(arr, min_val=0):
        """Return ``(mean, SEM)`` over finite entries of ``arr`` (optionally only
        those greater than ``min_val``); SEM is NaN for a single sample."""
        values = np.asarray(arr)
        if min_val is not None:
            mask = np.isfinite(values) & (values > min_val)
        else:
            mask = np.isfinite(values)
        values = values[mask]
        if values.size == 0:
            return np.nan, np.nan
        mean = np.mean(values)
        if values.size > 1:
            sem = np.std(values, ddof=1) / np.sqrt(values.size)
        else:
            # SEM is undefined on a single sample; NaN is honest, 0 is not.
            sem = np.nan
        return mean, sem

if __name__ == '__main__':
    r = 300
    T=300
    sigma = 8
    folder = "ANALYSIS_SG"
    sm = "sg_X_0"
    #folder = "ANALYSIS_DSM_AVE"
    #sm = "dsm_daunorubicin"
    segments = 20
    iterations = 10
    n_boot = 10
    dt_unit = 2000000E-15
    n_point = 1000
    n_tau = 26

    Pxyz, time = [], []
    vol_sys = 4 / 3 * math.pi * (r*10**-10) ** 3
    stress_file = "{}/Stress_Tensor_{}.csv".format(folder, sm)
    with open(stress_file, "r") as file:
        print('\nPreparing Pressure Tensor Array')
        for line in file.readlines()[1:]:
            ln = line.split(",")
            time.append(float(ln[0]) * 20)
            Pxy = -float(ln[4])
            Pxz = -float(ln[5])
            Pyz = -float(ln[6])
            Pxyz.append([Pxy, Pxz, Pyz])
    # Stress_Tensor stores summed per-atom stress-volume terms in atm*Angstrom^3.
    # Convert to pressure in Pa using the same droplet-volume normalization as the
    # main pipeline before applying the Green-Kubo prefactor.
    conv = 101325.0 * 1e-30 / vol_sys
    Pxyz = np.array(Pxyz) * conv
    path = "CLASS"
    vsc = visc(path, Pxyz)
    eta_raw, eta_raw_sem, eta_theo, eta_theo_sem, amp_opts, tau_opts, y_log0s, dt_log = vsc.run_vsc(vol_sys=vol_sys, name=sm, segments=segments,
                                                                                                  iterations=iterations, n_boot=n_boot,
                                                                                                  dt_unit=dt_unit, n_point=n_point,
                                                                                                  n_tau=n_tau, T=T)
