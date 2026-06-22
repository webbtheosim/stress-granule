#!/usr/bin/env python3
"""
Per-chain confined-diffusion analysis from condensate-comoving MSD.

Support module imported by the analysis pipeline (``system_analysis.py``
and ``block_correlation_diagnostics.py``) to characterize biopolymer mobility
inside a stress-granule condensate. The condensate translation is removed from
the per-residue COM trajectory to form a comoving MSD, which is fit with the
confined model MSD(t) = l^2*(1 - exp(-t/tau)) via a robust two-stage estimator
(plateau detection for l, then threshold crossing for tau). The local cage
mobility D_cage = l^2/(6*tau), Stokes-Einstein viscosities, and per-species
confinement/spatial observables are derived from there. ``run_diffusion_confined``
is the main entry point.

Inputs (read from disk):
    - RCC MSD output ``<system_tag>_msd_rdp.out.all`` (or LAMMPS
      ``<system_tag>_msd.out.all``: ``step n_residues`` header rows followed by
      per-residue ``resid ... COMMSD_x COMMSD_y COMMSD_z``).
    - COM trajectory sidecar ``<system_tag>_msd_rdp_com.npz`` (times, per-residue
      COM/Rg/Rh), preferred over the precomputed MSD when present.
    - Radius of gyration ``cluster_root/ANALYSIS_<PREFIX>/RG_<system_tag>_rg.out.all``.
    - Cluster membership: ``Tracked_Cluster_<tag>_<t>.npz`` sidecars (preferred)
      or legacy ``Max_Continuous_Cluster_<tag>_<t>.txt`` files.

Internal units:
    - time: ns,  MSD: A^2,  D: A^2/ns  (converted to SI for the returned dict).

Outputs:
    - Per-chain CSV (``*_per_chain_diffusion_confined.csv``) with l, tau, D_cage,
      Rg, Rh, species, and Inside/Outside flags, plus an optional time-origin QC
      CSV. ``run_diffusion_confined`` also returns a dict of system-level and
      per-species summaries matching the Quant_Data convention.

Not runnable as a script; call ``run_diffusion_confined`` from the pipeline.
"""


import math
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress

DEFAULT_DIFFUSION_SPECIES = ("G3BP1", "PABP1", "TIA1", "TTP", "FUS", "TDP43", "RNA")
PROTEIN_DIFFUSION_SPECIES = tuple(sp for sp in DEFAULT_DIFFUSION_SPECIES if sp != "RNA")
# For condensate-internal transport we want chains that remain in the SG for most
# of the analyzed time window. Strict 1.00 discards dissolving systems entirely
# (e.g., pyrivinium, anacardic) where chains exit/re-enter; 0.95 recovers those
# while still requiring the chain be inside in all but ~2 of 40 windows.
DEFAULT_INSIDE_OCCUPANCY = 0.95
DEFAULT_OUTSIDE_OCCUPANCY = 0.10
DEFAULT_ALLOW_AMBIGUOUS_FALLBACK = False
DEFAULT_MIN_DIFF_START_NS = 10.0       # V1/V2: 0; keep small guard for initial noise
DEFAULT_MIN_DIFF_SPAN_NS = 50.0        # V1/V2: 0; insist on 50 ns minimum span
DEFAULT_MAX_DIFF_FIT_FRACTION = 0.90   # V1/V2: no limit; allow fitting most of trajectory
DEFAULT_LINEAR_R2_MIN = 0.80           # V1/V2: boot_r2=0.8 (~r2>0.64); 0.80 = decent quality
DEFAULT_MIN_PLATEAU_SPAN_NS = 50.0     # V1/V2: 0; keep small guard
DEFAULT_MIN_SEGMENT_NS = 200.0
DEFAULT_MIN_PRIMARY_FRACTION = 0.25
DEFAULT_ORIGIN_RESAMPLE = True
DEFAULT_ORIGIN_WINDOW_NS = 1000.0
DEFAULT_ORIGIN_STRIDE_NS = 200.0
DEFAULT_ORIGIN_CANDIDATE_WINDOWS_NS = (500.0, 750.0, 1000.0, 1250.0, 1500.0)
DEFAULT_ORIGIN_MIN_SUCCESS_FRACTION = 0.70
DEFAULT_ORIGIN_MIN_SUCCESS_COUNT = 3
DEFAULT_ORIGIN_MAX_ORIGINS = 12

# ---------------------------------------------------------------------------
# Data structures and helpers
# ---------------------------------------------------------------------------


@dataclass
class chain_result:
    """Per-chain diffusion/confinement result.

    Bundles a single chain's identity (resid, resname, species, Inside/Outside
    location) with its fitted transport observables and uncertainties: diffusion
    coefficient ``D`` (A^2/ns), confinement length ``l`` (A), cage relaxation
    time ``tau`` (s), Stokes-Einstein viscosities from Rg and Kirkwood Rh (Pa*s),
    the radii Rg/Rh (A), and the fit-type labels for the diffusion and plateau
    estimators.
    """
    resid: int
    resname: str
    species: str
    location: str  # "Inside" or "Outside"
    D_A2_ns: float
    D_sem_A2_ns: float
    l_A: float
    l_sem_A: float
    tau_s: float
    tau_sem_s: float
    eta_Pa_s: float        # Stokes-Einstein viscosity using Rg
    eta_sem_Pa_s: float
    eta_Rh_Pa_s: float     # Stokes-Einstein viscosity using Kirkwood Rh
    eta_Rh_sem_Pa_s: float
    Rg_A: float
    Rg_sem_A: float
    Rh_A: float            # Kirkwood hydrodynamic radius
    Rh_sem_A: float
    type_diffusion: str
    type_plateau: str


def _stokes_einstein_D(
    eta_Pa_s: float,
    eta_sem_Pa_s: float,
    R_A: float,
    R_sem_A: float,
    T: float,
) -> Tuple[float, float]:
    """Stokes-Einstein diffusion coefficient from viscosity and radius.

    D = kB*T / (6*pi*eta*R), with eta in Pa.s and R in m.
    Returns (D_m2_s, D_sem_m2_s).

    Inverse Stokes-Einstein relation.  Use when eta_GK is the primary
    transport observable and D is inferred.
    """
    kb = 1.380649e-23  # J/K
    try:
        R_m = R_A * 1e-10 if math.isfinite(R_A) else math.nan
        if not (math.isfinite(eta_Pa_s) and eta_Pa_s > 0.0
                and math.isfinite(R_m) and R_m > 0.0):
            return math.nan, math.nan
        D_val = (kb * T) / (6.0 * math.pi * eta_Pa_s * R_m)
        rel2 = 0.0
        if math.isfinite(eta_sem_Pa_s) and eta_sem_Pa_s > 0.0:
            rel2 += (eta_sem_Pa_s / eta_Pa_s) ** 2
        if math.isfinite(R_sem_A) and R_sem_A > 0.0:
            R_sem_m = R_sem_A * 1e-10
            if R_sem_m > 0.0:
                rel2 += (R_sem_m / R_m) ** 2
        D_sem_val = abs(D_val) * math.sqrt(rel2) if rel2 > 0.0 else math.nan
        return float(D_val), float(D_sem_val)
    except Exception:
        return math.nan, math.nan


def _time_labels(t) -> List[str]:
    """Return de-duplicated string spellings of a time value ``t`` (e.g. integer
    and float forms) to try when resolving time-stamped filenames."""
    labels: List[str] = []
    try:
        tf = float(t)
        if tf.is_integer():
            labels.append(str(int(tf)))
        labels.append(str(tf))
    except Exception:
        labels.append(str(t))

    seen = set()
    ordered: List[str] = []
    for label in labels:
        if label not in seen:
            ordered.append(label)
            seen.add(label)
    return ordered


def _resolve_time_file(path_builder, t) -> Optional[str]:
    """Return the first existing file produced by ``path_builder(label)`` over the
    candidate time spellings of ``t``, or None if none exist."""
    for label in _time_labels(t):
        path = path_builder(label)
        if os.path.isfile(path):
            return path
    return None


def _msd_fft_total_from_positions(pos_A: np.ndarray) -> np.ndarray:
    """
    Vectorized unbiased MSD from positions using FFT.

    Args:
        pos_A: (n_time, n_series, 3) Cartesian positions in Angstrom.

    Returns:
        Total MSD (x+y+z) with shape (n_time, n_series) in Angstrom^2.
    """
    pos = np.asarray(pos_A, dtype=float)
    if pos.ndim != 3 or pos.shape[2] != 3:
        raise ValueError("pos_A must have shape (n_time, n_series, 3)")
    if pos.shape[0] < 1:
        raise ValueError("pos_A must contain at least one time point")

    T = pos.shape[0]
    n_series = pos.shape[1]
    if T == 1:
        return np.zeros((1, n_series), dtype=float)

    def _autocorr_fft(x: np.ndarray) -> np.ndarray:
        """Unbiased (lag-count normalized) autocorrelation of each column via FFT."""
        nfft = 1 << (2 * T - 1).bit_length()
        fx = np.fft.rfft(x, n=nfft, axis=0)
        ac = np.fft.irfft(fx * np.conj(fx), n=nfft, axis=0)[:T]
        norm = (T - np.arange(T)).astype(float)[:, None]
        return ac / norm

    sq = np.sum(pos * pos, axis=2)
    sq_pad = np.vstack([sq, np.zeros((1, n_series), dtype=float)])
    s2 = sum(_autocorr_fft(pos[:, :, d]) for d in range(3))

    q = 2.0 * np.sum(sq, axis=0)
    s1 = np.zeros((T, n_series), dtype=float)
    for m in range(T):
        q = q - sq_pad[m - 1] - sq_pad[T - m]
        s1[m, :] = q / float(T - m)

    msd_tot = s1 - 2.0 * s2
    msd_tot[0, :] = 0.0
    return np.maximum(msd_tot, 0.0)


def _window_points_from_duration(times_ns: np.ndarray, duration_ns: float, min_points: int) -> int:
    """Convert a target window ``duration_ns`` into a number of frame points,
    using the median frame spacing in ``times_ns`` and clamping to at least
    ``min_points``."""
    t = np.asarray(times_ns, dtype=float)
    if t.size <= 1:
        return max(2, int(min_points))
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0.0)]
    if dt.size == 0:
        return max(2, int(min_points))
    dt_med = float(np.median(dt))
    if not math.isfinite(dt_med) or dt_med <= 0.0:
        return max(2, int(min_points))
    pts = int(math.ceil(float(duration_ns) / dt_med)) + 1
    return max(int(min_points), pts)


# ---------------------------------------------------------------------------
# Confined diffusion model: MSD(t) = l^2 * (1 - exp(-t / tau))
# ---------------------------------------------------------------------------


def _pure_confined_model(t, l_sq, tau):
    """MSD(t) = l^2*(1 - exp(-t/tau)).  Pure confinement, no transport term.

    For droplet-geometry condensates where no diffusive regime exists
    in the condensate-comoving MSD.  Fit parameters:
      l  = confinement / localisation length (sqrt(l_sq))
      tau = cage-escape / local relaxation timescale
    Derived:  D_cage = l^2 / (6*tau)  (local cage mobility, NOT long-time D).
    """
    return l_sq * (1.0 - np.exp(-t / tau))


def _detect_msd_plateau(
    times_ns: np.ndarray,
    msd_A2: np.ndarray,
    t_skip_ns: float = 5.0,
    late_frac_start: float = 0.30,
    late_frac_end: float = 0.85,
    cv_max: float = 0.20,
) -> Tuple[float, float, float, float, int]:
    """Detect MSD plateau directly — Stage 1 of two-stage confinement analysis.

    Uses the median MSD in the late-time range [late_frac_start, late_frac_end]
    of the lag-time axis as a robust plateau estimate.  The last ~15% of lag
    times are excluded because ensemble-averaged MSD has fewer independent
    window pairs there and becomes noisy.

    This approach is more robust than log-log slope detection because:
    - Not sensitive to non-monotonic fluctuations (e.g. overshoot/rebound)
    - Not sensitive to noisy upswing at extreme lag times
    - Median is robust to outliers

    Plateau existence is verified by the coefficient of variation (CV):
    if CV > cv_max, the data does not have a clean plateau.

    Parameters
    ----------
    times_ns : array  — lag times
    msd_A2   : array  — ensemble-averaged comoving MSD (Å²)
    t_skip_ns : float — skip early points
    late_frac_start : float — start of plateau window as fraction of time range
    late_frac_end : float — end of plateau window (excludes noisy tail)
    cv_max : float — max coefficient of variation for valid plateau

    Returns
    -------
    (l_A, l_sem_A, plateau_start_ns, plateau_cv, n_plateau)
    All NaN/0 if no plateau detected.
    """
    NAN5P = (math.nan, math.nan, math.nan, math.nan, 0)
    t = np.asarray(times_ns, dtype=float)
    y = np.asarray(msd_A2, dtype=float)

    mask = np.isfinite(t) & np.isfinite(y) & (t > t_skip_ns) & (y > 0.0)
    t = t[mask]
    y = y[mask]
    n = t.size
    if n < 30:
        return NAN5P

    # Define plateau measurement window in the late-time range,
    # excluding the noisy tail
    t_range = t[-1] - t[0]
    t_plat_lo = t[0] + late_frac_start * t_range
    t_plat_hi = t[0] + late_frac_end * t_range
    plat_mask = (t >= t_plat_lo) & (t <= t_plat_hi)
    plat_msd = y[plat_mask]

    if plat_msd.size < 20:
        return NAN5P

    # Robust plateau value: median (insensitive to outliers / local bumps)
    msd_plateau = float(np.median(plat_msd))
    plat_std = float(np.std(plat_msd, ddof=1)) if plat_msd.size > 1 else 0.0
    plat_cv = plat_std / msd_plateau if msd_plateau > 0 else math.nan

    if not math.isfinite(plat_cv) or plat_cv > cv_max:
        return NAN5P

    # SEM of plateau MSD (standard error of the mean in the window)
    plat_sem = plat_std / math.sqrt(plat_msd.size) if plat_msd.size > 1 else math.nan

    # l_conf = sqrt(MSD_plateau)
    l_A = math.sqrt(msd_plateau)
    # Propagate SEM:  l = sqrt(MSD) → δl = δMSD / (2l)
    l_sem_A = plat_sem / (2.0 * l_A) if (l_A > 0 and math.isfinite(plat_sem)) else math.nan

    # Plateau onset: first time MSD reaches 90% of plateau value
    onset_val = 0.90 * msd_plateau
    above = np.where(y >= onset_val)[0]
    plateau_start_ns = float(t[above[0]]) if above.size > 0 else float(t_plat_lo)

    return l_A, l_sem_A, plateau_start_ns, plat_cv, int(plat_msd.size)


def _fit_crossover_tau(
    times_ns: np.ndarray,
    msd_A2: np.ndarray,
    l_sq: float,
    t_skip_ns: float = 5.0,
    frac_max: float = 0.90,
    r2_min: float = 0.85,
) -> Tuple[float, float, float]:
    """Estimate cage-escape timescale τ from linearized approach-to-plateau.

    Stage 2 of two-stage confinement analysis.  With l_conf² fixed from
    plateau detection, the pure confinement model gives:

        log(1 - MSD(t)/l²) = -t/τ

    Fit the left side vs t (linear regression) in the pre-plateau region
    where MSD < frac_max * l².

    Parameters
    ----------
    l_sq : float — plateau MSD value (l_conf² in Å²), from Stage 1.
    frac_max : float — only use points where MSD < frac_max * l².
    r2_min : float — reject fit if R² below this.

    Returns
    -------
    (tau_ns, tau_sem_ns, fit_r2)   — all NaN if fit fails.
    """
    NAN3 = (math.nan, math.nan, math.nan)
    if not (math.isfinite(l_sq) and l_sq > 0):
        return NAN3

    t = np.asarray(times_ns, dtype=float)
    y = np.asarray(msd_A2, dtype=float)

    # Select pre-plateau region: t > t_skip, MSD > 0, MSD < frac_max * l²
    mask = (np.isfinite(t) & np.isfinite(y)
            & (t > t_skip_ns) & (y > 0.0) & (y < frac_max * l_sq))
    t_sel = t[mask]
    y_sel = y[mask]
    if t_sel.size < 10:
        return NAN3

    # Linearized variable: z = log(1 - MSD/l²)
    ratio = 1.0 - y_sel / l_sq
    # Guard: ratio must be positive for log
    pos = ratio > 0.01
    t_fit = t_sel[pos]
    z_fit = np.log(ratio[pos])
    if t_fit.size < 10:
        return NAN3

    # Linear regression: z = -t/τ  (force through origin? No — allow intercept
    # to absorb early-time deviations from single-exponential)
    try:
        result = linregress(t_fit, z_fit)
    except Exception:
        return NAN3

    slope = result.slope       # should be negative: -1/τ
    slope_se = result.stderr
    r2 = result.rvalue ** 2

    if slope >= 0 or r2 < r2_min:
        return NAN3

    tau_ns = -1.0 / slope
    # δτ/τ = δslope/|slope|  (propagation from τ = -1/slope)
    tau_sem_ns = tau_ns * (slope_se / abs(slope)) if slope_se > 0 else math.nan

    return tau_ns, tau_sem_ns, r2


def _tau_threshold_crossing(
    times_ns: np.ndarray,
    msd_A2: np.ndarray,
    l_sq: float,
    threshold_frac: float = 0.6321,
    plat_std_A2: float = math.nan,
) -> Tuple[float, float]:
    """Cage-escape time from threshold crossing: τ = t where MSD first reaches
    (1-1/e) * l² ≈ 0.632 * l².

    Model-free standard from colloidal glass physics.  Always gives a number
    when a plateau exists and the MSD rises through the threshold.  Handles
    overshoot/rebound naturally (uses first crossing).

    Analytic SEM (when ``plat_std_A2`` is supplied):
        σ_τ ≈ plat_std / |dMSD/dt|_{t=τ}
    The per-time-point MSD noise level (plat_std, measured from the plateau
    window std) is mapped through the local MSD slope to yield an honest
    uncertainty on the crossing time.  This is preferred over the linearised-
    fit slope SE, which overestimates τ_SEM because log(1−MSD/l²) diverges
    as MSD approaches l².

    Returns (tau_ns, tau_sem_ns), with tau_sem_ns = NaN when plat_std_A2 is
    NaN or the local slope cannot be estimated.
    """
    if not (math.isfinite(l_sq) and l_sq > 0):
        return math.nan, math.nan
    target = threshold_frac * l_sq
    t = np.asarray(times_ns, dtype=float)
    y = np.asarray(msd_A2, dtype=float)
    above = np.where((y >= target) & np.isfinite(y))[0]
    if above.size == 0:
        return math.nan, math.nan
    idx = int(above[0])
    # Linear interpolation between idx-1 and idx for sub-bin precision
    if idx > 0 and y[idx - 1] < target and np.isfinite(y[idx - 1]):
        frac = (target - y[idx - 1]) / (y[idx] - y[idx - 1])
        tau_ns = float(t[idx - 1] + frac * (t[idx] - t[idx - 1]))
    else:
        tau_ns = float(t[idx])

    tau_sem_ns = math.nan
    if math.isfinite(plat_std_A2) and plat_std_A2 > 0:
        # Local slope dMSD/dt at the crossing via a short-window linear fit.
        i0 = max(0, idx - 3)
        i1 = min(len(t) - 1, idx + 3)
        if i1 - i0 >= 3:
            try:
                lr = linregress(t[i0:i1 + 1], y[i0:i1 + 1])
                slope = float(lr.slope)
                if math.isfinite(slope) and slope > 0:
                    tau_sem_ns = float(plat_std_A2 / slope)
            except Exception:
                pass
    return tau_ns, tau_sem_ns


def _dl_random_effects_pool(
    values: np.ndarray,
    sems: np.ndarray,
) -> Tuple[float, float, float, int]:
    """DerSimonian-Laird random-effects meta-analysis pooling.

    Combines N per-chain (or per-unit) estimates with within-unit uncertainties
    into a single pooled estimate whose SEM captures BOTH within-unit fit error
    AND between-unit heterogeneity.  Standard in meta-analysis (DerSimonian &
    Laird, Controlled Clinical Trials 1986) and polymer / single-particle
    tracking analyses (Saxton, Biophys J 1997; Weeks et al., Science 2000).

    Model: value_i = μ + ε_i + u_i, with ε_i ~ N(0, sem_i²) within-unit noise
    and u_i ~ N(0, τ²) between-unit heterogeneity.  τ² is estimated from the
    weighted heterogeneity statistic Q:

        w_i = 1 / sem_i²
        μ̂_FE = Σ w_i·v_i / Σ w_i          (fixed-effects estimate)
        Q = Σ w_i·(v_i - μ̂_FE)²           (Cochran Q)
        τ² = max(0, (Q - (k-1)) / (Σw_i - Σw_i²/Σw_i))   (DL estimator)
        w_i* = 1 / (sem_i² + τ²)
        μ̂ = Σ w_i*·v_i / Σ w_i*
        SEM(μ̂) = 1 / √(Σ w_i*)

    Edge cases:
      - k=0 (no valid units):  returns (nan, nan, nan, 0)
      - k=1:  μ̂ = v_1, SEM = sem_1, τ² = 0   (no heterogeneity estimable)
      - all sem_i = 0 or nan:  fall back to unweighted mean with
        SEM = std(values, ddof=1)/√k  (classical Student t, reduces to a
        single-measurement NaN for k=1 — caller should pre-filter sems)

    Returns:
        (pooled_mean, pooled_sem, tau_sq_between, k_used)
    """
    v = np.asarray(values, dtype=float)
    s = np.asarray(sems, dtype=float)
    mask = np.isfinite(v) & np.isfinite(s) & (s > 0.0)
    v = v[mask]
    s = s[mask]
    k = int(v.size)
    if k == 0:
        return math.nan, math.nan, math.nan, 0
    if k == 1:
        return float(v[0]), float(s[0]), 0.0, 1

    w = 1.0 / (s ** 2)
    sum_w = float(w.sum())
    mu_fe = float((w * v).sum() / sum_w)
    Q = float((w * (v - mu_fe) ** 2).sum())
    c = sum_w - float((w ** 2).sum()) / sum_w
    tau_sq = max(0.0, (Q - (k - 1)) / c) if c > 0 else 0.0
    w_star = 1.0 / (s ** 2 + tau_sq)
    sum_ws = float(w_star.sum())
    mu = float((w_star * v).sum() / sum_ws)
    sem = float(1.0 / math.sqrt(sum_ws)) if sum_ws > 0 else math.nan
    return mu, sem, tau_sq, k


def _bootstrap_two_stage(
    times_ns: np.ndarray,
    msd_mat_A2: np.ndarray,
    chain_idx: np.ndarray,
    t_skip_ns: float = 5.0,
    n_boot: int = 200,
    seed: int = 0,
    cv_max: float = 0.20,
) -> Tuple[float, float, float, float]:
    """Bootstrap SEM for l_conf and tau_conf by resampling chains.

    Resamples which chains contribute to the species-averaged MSD,
    recomputes the chain-averaged MSD, and re-runs the two-stage
    plateau + threshold-crossing analysis.  The standard deviation
    of the bootstrap distribution is the standard error.

    Returns (l_sem_A, tau_sem_ns, l_boot_mean, tau_boot_mean).
    All NaN if insufficient successful resamples.
    """
    n_chains = chain_idx.size
    if n_chains < 2:
        return math.nan, math.nan, math.nan, math.nan

    rng = np.random.RandomState(seed)
    l_samples: List[float] = []
    tau_samples: List[float] = []

    for _ in range(n_boot):
        idx_b = rng.choice(n_chains, n_chains, replace=True)
        msd_b = np.nanmean(msd_mat_A2[:, chain_idx[idx_b]], axis=1)
        ts_b = fit_confined_two_stage(times_ns, msd_b, t_skip_ns=t_skip_ns, cv_max=cv_max)
        if math.isfinite(ts_b["l_A"]):
            l_samples.append(ts_b["l_A"])
        if math.isfinite(ts_b["tau_ns"]):
            tau_samples.append(ts_b["tau_ns"])

    l_sem = float(np.std(l_samples, ddof=1)) if len(l_samples) >= 10 else math.nan
    tau_sem = float(np.std(tau_samples, ddof=1)) if len(tau_samples) >= 10 else math.nan
    l_mean = float(np.mean(l_samples)) if l_samples else math.nan
    tau_mean = float(np.mean(tau_samples)) if tau_samples else math.nan
    return l_sem, tau_sem, l_mean, tau_mean


def _combine_sem_quadrature(*sems: float) -> float:
    """Combine independent SEM components in quadrature."""
    vals: List[float] = []
    for sem in sems:
        try:
            val = float(sem)
        except Exception:
            continue
        if math.isfinite(val) and val > 0.0:
            vals.append(val)
    if not vals:
        return math.nan
    return float(math.sqrt(sum(v * v for v in vals)))


def _block_corrected_origin_sem(values: Sequence[float]) -> Tuple[float, int, int]:
    """Flyvbjerg-Petersen style batch SEM for correlated origin estimates.

    The origin-started subtrajectory fits are ordered in time and are not fully
    independent when adjacent windows overlap.  We therefore sweep contiguous
    batch sizes, compute the SEM of batch means, and use the maximal SEM as a
    conservative corrected uncertainty.  The reported block size is the first
    batch size whose SEM reaches 95% of that maximum.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n < 3:
        return math.nan, 0, 0

    sem_rows: List[Tuple[int, int, float]] = []
    for block_size in range(1, max(1, n // 2) + 1):
        n_blocks = n // block_size
        if n_blocks < 2:
            continue
        trimmed = arr[:n_blocks * block_size]
        block_means = trimmed.reshape(n_blocks, block_size).mean(axis=1)
        sem = float(block_means.std(ddof=1) / math.sqrt(n_blocks))
        if math.isfinite(sem) and sem > 0.0:
            sem_rows.append((block_size, n_blocks, sem))

    if not sem_rows:
        return math.nan, 0, 0

    max_sem = max(row[2] for row in sem_rows)
    target = 0.95 * max_sem
    for block_size, n_blocks, sem in sem_rows:
        if sem >= target:
            return float(max_sem), int(block_size), int(n_blocks)
    block_size, n_blocks, _ = sem_rows[-1]
    return float(max_sem), int(block_size), int(n_blocks)


def _parse_origin_candidate_windows(candidate_windows_ns) -> List[float]:
    """Normalize candidate time-origin window lengths (ns) into a sorted list of
    unique positive floats, accepting a delimited string, an iterable, or None
    (which falls back to ``DEFAULT_ORIGIN_CANDIDATE_WINDOWS_NS``)."""
    if candidate_windows_ns is None:
        return list(DEFAULT_ORIGIN_CANDIDATE_WINDOWS_NS)
    if isinstance(candidate_windows_ns, str):
        parts = re.split(r"[,;\s]+", candidate_windows_ns.strip())
        values = [float(p) for p in parts if p]
    else:
        values = [float(v) for v in candidate_windows_ns]
    clean = sorted({v for v in values if math.isfinite(v) and v > 0.0})
    return clean or list(DEFAULT_ORIGIN_CANDIDATE_WINDOWS_NS)


def _origin_start_stop_indices(
    times_ns: np.ndarray,
    window_ns: float,
    stride_ns: float,
    max_origins: Optional[int] = None,
) -> List[Tuple[int, int, float]]:
    """Enumerate sliding time-origin windows over ``times_ns``.

    Returns a list of ``(start_index, stop_index, start_time_ns)`` tuples for
    windows of length ``window_ns`` advanced by ``stride_ns`` (each requiring at
    least 30 frames), optionally thinned to at most ``max_origins`` evenly
    spaced windows.
    """
    t = np.asarray(times_ns, dtype=float)
    if t.size < 30 or not (math.isfinite(window_ns) and window_ns > 0.0):
        return []
    try:
        stride_val = float(stride_ns)
    except Exception:
        stride_val = math.nan
    stride = stride_val if math.isfinite(stride_val) and stride_val > 0.0 else window_ns
    start_limit = float(t[-1] - window_ns)
    if start_limit < float(t[0]) - 1e-9:
        return []

    starts: List[Tuple[int, int, float]] = []
    requested_start = float(t[0])
    while requested_start <= start_limit + 1e-9:
        i0 = int(np.searchsorted(t, requested_start, side="left"))
        if i0 >= t.size - 29:
            break
        stop_time = float(t[i0] + window_ns)
        i1 = int(np.searchsorted(t, stop_time, side="right"))
        if i1 - i0 >= 30:
            starts.append((i0, i1, float(t[i0])))
        requested_start += stride

    if max_origins is not None and int(max_origins) > 0 and len(starts) > int(max_origins):
        keep = np.linspace(0, len(starts) - 1, int(max_origins), dtype=int)
        starts = [starts[int(i)] for i in np.unique(keep)]
    return starts


def _origin_resample_nan_result(status: str = "not_run") -> Dict[str, object]:
    """Return the time-origin resample result dict with all metrics set to NaN/0
    and the given ``status``, used when resampling is skipped or fails."""
    return {
        "origin_resample_status": status,
        "origin_resample_window_ns": math.nan,
        "origin_resample_stride_ns": math.nan,
        "origin_resample_n_success": 0,
        "origin_resample_n_total": 0,
        "origin_resample_success_fraction": math.nan,
        "origin_resample_l_sem_A": math.nan,
        "origin_resample_tau_sem_ns": math.nan,
        "origin_resample_D_cage_sem_m2_s": math.nan,
        "origin_resample_sem_block_origins": math.nan,
        "origin_resample_sem_block_ns": math.nan,
        "origin_resample_sem_n_blocks": math.nan,
        "origin_resample_min_plateau_window_ns": math.nan,
    }


def _time_origin_resample_two_stage(
    times_ns: np.ndarray,
    com_pos_A: np.ndarray,
    center_mask: np.ndarray,
    inside_idx_arr: np.ndarray,
    t_skip_ns: float = 5.0,
    cv_max: float = 0.20,
    candidate_windows_ns=None,
    stride_ns: float = DEFAULT_ORIGIN_STRIDE_NS,
    min_success_fraction: float = DEFAULT_ORIGIN_MIN_SUCCESS_FRACTION,
    min_success_count: int = DEFAULT_ORIGIN_MIN_SUCCESS_COUNT,
    max_origins: int = DEFAULT_ORIGIN_MAX_ORIGINS,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Resample confined-MSD fits over trajectory time origins.

    The central confinement estimate remains the full-trajectory FFT MSD.  This
    routine asks a separate question: how stable are l_conf, tau_conf and D_cage
    if the same analysis is repeated on long subtrajectories starting at
    different time origins?  Candidate window lengths are scanned to identify
    the shortest block with reliable plateau detection; SEMs are then corrected
    by batch means over ordered origin estimates to avoid treating overlapping
    origins as independent.
    """
    if com_pos_A is None or inside_idx_arr.size == 0:
        return _origin_resample_nan_result("missing_com_or_inside"), []

    t = np.asarray(times_ns, dtype=float)
    pos = np.asarray(com_pos_A, dtype=float)
    if t.size < 30 or pos.ndim != 3 or pos.shape[0] != t.size:
        return _origin_resample_nan_result("insufficient_timeseries"), []

    total_span_ns = float(t[-1] - t[0])
    candidates = [
        w for w in _parse_origin_candidate_windows(candidate_windows_ns)
        if w <= total_span_ns + 1e-9
    ]
    if not candidates:
        return _origin_resample_nan_result("no_candidate_window_fits_trajectory"), []

    records: List[Dict[str, object]] = []
    summaries: List[Dict[str, object]] = []
    for window_ns in candidates:
        starts = _origin_start_stop_indices(t, window_ns, stride_ns, max_origins=max_origins)
        fit_rows: List[Dict[str, object]] = []
        for i0, i1, start_ns in starts:
            seg_times = t[i0:i1] - t[i0]
            seg_pos = pos[i0:i1, :, :]
            try:
                seg_msd = build_comoving_msd_from_com_series(seg_pos, center_mask=center_mask)
                with np.errstate(invalid="ignore", divide="ignore"):
                    seg_avg = np.nanmean(seg_msd[:, inside_idx_arr], axis=1)
                fit = fit_confined_two_stage(seg_times, seg_avg, t_skip_ns=t_skip_ns, cv_max=cv_max)
            except Exception:
                fit = {
                    "l_A": math.nan,
                    "tau_ns": math.nan,
                    "D_cage_m2_s": math.nan,
                    "plateau_cv": math.nan,
                    "plateau_start_ns": math.nan,
                    "n_plateau": 0,
                    "stage": "failed",
                }
            success = bool(math.isfinite(fit.get("l_A", math.nan)) and fit.get("stage") != "failed")
            row = {
                "record_type": "origin_fit",
                "window_ns": float(window_ns),
                "origin_start_ns": float(start_ns),
                "origin_stop_ns": float(t[i1 - 1]),
                "n_points": int(i1 - i0),
                "success": success,
                "stage": fit.get("stage", "failed"),
                "l_A": fit.get("l_A", math.nan),
                "tau_ns": fit.get("tau_ns", math.nan),
                "D_cage_m2_s": fit.get("D_cage_m2_s", math.nan),
                "plateau_cv": fit.get("plateau_cv", math.nan),
                "plateau_start_ns": fit.get("plateau_start_ns", math.nan),
                "n_plateau": fit.get("n_plateau", 0),
            }
            records.append(row)
            fit_rows.append(row)

        n_total = len(starts)
        n_success = int(sum(bool(r["success"]) for r in fit_rows))
        success_fraction = float(n_success / n_total) if n_total else math.nan
        summary = {
            "window_ns": float(window_ns),
            "n_total": int(n_total),
            "n_success": int(n_success),
            "success_fraction": success_fraction,
            "fit_rows": fit_rows,
        }
        summaries.append(summary)
        records.append({
            "record_type": "candidate_summary",
            "window_ns": float(window_ns),
            "origin_start_ns": math.nan,
            "origin_stop_ns": math.nan,
            "n_points": math.nan,
            "success": n_success >= int(min_success_count)
                       and math.isfinite(success_fraction)
                       and success_fraction >= float(min_success_fraction),
            "stage": "summary",
            "l_A": math.nan,
            "tau_ns": math.nan,
            "D_cage_m2_s": math.nan,
            "plateau_cv": math.nan,
            "plateau_start_ns": math.nan,
            "n_plateau": math.nan,
            "n_total": int(n_total),
            "n_success": int(n_success),
            "success_fraction": success_fraction,
        })

    accepted = [
        s for s in summaries
        if s["n_success"] >= int(min_success_count)
        and math.isfinite(float(s["success_fraction"]))
        and float(s["success_fraction"]) >= float(min_success_fraction)
    ]
    if accepted:
        selected = min(accepted, key=lambda s: float(s["window_ns"]))
        status = "ok"
        min_plateau_window_ns = float(selected["window_ns"])
    else:
        usable = [s for s in summaries if s["n_success"] >= int(min_success_count)]
        if not usable:
            return _origin_resample_nan_result("no_successful_origin_window"), records
        selected = max(usable, key=lambda s: (float(s["success_fraction"]), int(s["n_success"]), -float(s["window_ns"])))
        status = "below_success_threshold"
        min_plateau_window_ns = math.nan

    fit_rows = [r for r in selected["fit_rows"] if bool(r["success"])]
    l_vals = [float(r["l_A"]) for r in fit_rows if math.isfinite(float(r["l_A"]))]
    tau_vals = [float(r["tau_ns"]) for r in fit_rows if math.isfinite(float(r["tau_ns"]))]
    D_vals = [float(r["D_cage_m2_s"]) for r in fit_rows if math.isfinite(float(r["D_cage_m2_s"]))]

    l_sem, l_block, l_nblocks = _block_corrected_origin_sem(l_vals)
    tau_sem, tau_block, tau_nblocks = _block_corrected_origin_sem(tau_vals)
    D_sem, D_block, D_nblocks = _block_corrected_origin_sem(D_vals)
    block_candidates = [b for b in (l_block, tau_block, D_block) if b and b > 0]
    nblock_candidates = [b for b in (l_nblocks, tau_nblocks, D_nblocks) if b and b > 0]
    sem_block_origins = max(block_candidates) if block_candidates else math.nan
    try:
        stride_val = float(stride_ns)
    except Exception:
        stride_val = math.nan
    sem_block_ns = (float(sem_block_origins * stride_val)
                    if math.isfinite(float(sem_block_origins)) and math.isfinite(stride_val)
                    else math.nan)
    sem_n_blocks = min(nblock_candidates) if nblock_candidates else math.nan

    result = _origin_resample_nan_result(status)
    result.update({
        "origin_resample_window_ns": float(selected["window_ns"]),
        "origin_resample_stride_ns": stride_val,
        "origin_resample_n_success": int(selected["n_success"]),
        "origin_resample_n_total": int(selected["n_total"]),
        "origin_resample_success_fraction": float(selected["success_fraction"]),
        "origin_resample_l_sem_A": l_sem,
        "origin_resample_tau_sem_ns": tau_sem,
        "origin_resample_D_cage_sem_m2_s": D_sem,
        "origin_resample_sem_block_origins": sem_block_origins,
        "origin_resample_sem_block_ns": sem_block_ns,
        "origin_resample_sem_n_blocks": sem_n_blocks,
        "origin_resample_min_plateau_window_ns": min_plateau_window_ns,
    })
    return result, records


def loglog_multiwindow_D(
    times_ns: np.ndarray,
    msd_A2: np.ndarray,
    dt_pts: int = 5,
    slope_tol: float = 0.25,
    min_diff_pts: int = 5,
    slope_iterations: int = 200,
    boot_r2: float = 0.80,
    t_skip_ns: float = 5.0,
    seed: int = 0,
) -> Tuple[float, float]:
    """V1/V2-style log-log slope detection for early-time D.

    Finds windows where d(log MSD)/d(log t) ~ 1 (diffusive regime),
    then extracts D = slope/6 from real-space linear regression in those
    windows via bootstrap.

    Returns (D_A2_ns, D_sem_A2_ns) in A^2/ns; NaN if no diffusive region.
    """
    t = np.asarray(times_ns, float)
    y = np.asarray(msd_A2, float)
    mask = (t > t_skip_ns) & (y > 0) & np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]
    if t.size < min_diff_pts:
        return math.nan, math.nan

    x_log = np.log10(t)
    y_log = np.log10(y)

    # Detect diffusive regions in log-log space (slope ~ 1)
    diff_regions: List[Tuple[int, int]] = []
    start = 0
    end = dt_pts
    while end <= len(x_log):
        try:
            sl = float(linregress(x_log[start:end], y_log[start:end]).slope)
        except Exception:
            sl = math.nan
        if not math.isnan(sl) and abs(sl - 1.0) <= slope_tol and (end - start) >= min_diff_pts:
            if diff_regions and start <= diff_regions[-1][1]:
                diff_regions[-1] = (diff_regions[-1][0], end)
            else:
                diff_regions.append((start, end))
        start += 1
        end += 1

    if not diff_regions:
        return math.nan, math.nan

    # Bootstrap real-space slopes from diffusive regions
    rng = np.random.RandomState(seed)
    all_slopes: List[float] = []
    for r0, r1 in diff_regions:
        xd, yd = t[r0:r1], y[r0:r1]
        xl, yl = x_log[r0:r1], y_log[r0:r1]
        for _ in range(slope_iterations):
            try:
                idx = rng.choice(len(xd), len(xd), replace=True)
                lm_log = linregress(xl[idx], yl[idx])
                if abs(lm_log.rvalue) > boot_r2 and abs(lm_log.slope - 1.0) <= slope_tol:
                    lm_real = linregress(xd[idx], yd[idx])
                    if lm_real.slope > 0:
                        all_slopes.append(lm_real.slope)
            except Exception:
                continue

    if not all_slopes:
        return math.nan, math.nan

    a = np.array(all_slopes, float)
    D = float(a.mean() / 6.0)
    D_sem = float(a.std(ddof=1) / (6.0 * math.sqrt(len(a)))) if a.size > 1 else math.nan
    return D, D_sem


def fit_confined_two_stage(
    times_ns: np.ndarray,
    msd_A2: np.ndarray,
    t_skip_ns: float = 5.0,
    cv_max: float = 0.20,
) -> Dict[str, float]:
    """Two-stage confinement analysis — robust primary estimator.

    Stage 1:  l_conf from direct MSD plateau detection (median in late-time
              window, robust to non-monotonic fluctuations and tail noise).
    Stage 2:  τ_cage from threshold crossing — the first time MSD reaches
              (1-1/e) · l² ≈ 0.632 · l².  Model-free, standard in glass
              physics (Doliwa & Heuer PRE 2000; Weeks et al. Science 2000).
    Derived:  D_cage = l²/(6τ) — local cage mobility, NOT long-time D.

    Both stages always produce a value when a plateau exists; there are no
    R² gates or conditional fits.  The linearized τ fit (_fit_crossover_tau)
    is retained as a diagnostic but is not the headline τ.

    cv_max controls plateau-quality gating: for system-wide chain-averaged
    MSD (~100 chains) use 0.20; for per-species MSD with small chain counts
    the intrinsic noise is ~sqrt(N_sys/N_sp)× higher, so pass a looser value
    (e.g. 1.0) so fits are not excluded for noise reasons alone.  The
    ``plateau_cv`` field in the result always carries the observed value so
    downstream code can filter by quality.

    Returns dict with keys:
        l_A, l_sem_A           — confinement length from plateau (Å)
        plateau_start_ns       — MSD onset of 90% of plateau
        plateau_cv             — coefficient of variation in plateau window
        n_plateau              — points in plateau window
        tau_ns, tau_sem_ns     — cage-escape time and analytic SEM (ns).
                                 SEM from plat_std / |dMSD/dt|_{t=τ}: the
                                 per-point plateau noise propagated through
                                 the local slope at the crossing.
        D_cage_m2_s, D_cage_sem_m2_s  — local cage mobility (m²/s), with
                                 full error propagation from l and τ.
        tau_lin_ns, tau_lin_sem_ns, tau_lin_r2 — diagnostic linearised fit
                                 (retained for comparison; its SEM is not
                                 the primary τ uncertainty).
        stage                  — "plateau+tau" or "failed"
    """
    nan_result = {
        "l_A": math.nan, "l_sem_A": math.nan,
        "plateau_start_ns": math.nan, "plateau_cv": math.nan, "n_plateau": 0,
        "tau_ns": math.nan, "tau_sem_ns": math.nan,
        "D_cage_m2_s": math.nan, "D_cage_sem_m2_s": math.nan,
        "tau_lin_ns": math.nan, "tau_lin_sem_ns": math.nan, "tau_lin_r2": math.nan,
        "stage": "failed",
    }

    # Stage 1: plateau detection
    l_A, l_sem_A, plat_start, plat_cv, n_plat = _detect_msd_plateau(
        times_ns, msd_A2, t_skip_ns=t_skip_ns, cv_max=cv_max,
    )
    if not math.isfinite(l_A):
        return nan_result

    l_sq = l_A ** 2
    result = dict(nan_result)
    result["l_A"] = l_A
    result["l_sem_A"] = l_sem_A
    result["plateau_start_ns"] = plat_start
    result["plateau_cv"] = plat_cv
    result["n_plateau"] = n_plat

    # Stage 2: τ_cage from threshold crossing (model-free) with analytic SEM.
    # plat_std (per-point plateau MSD noise) is recoverable from plat_cv × l_sq,
    # because plat_cv = plat_std / msd_plateau and msd_plateau = l² for the
    # median-plateau estimator.
    plat_std = plat_cv * l_sq if math.isfinite(plat_cv) else math.nan
    tau_ns, tau_sem_ns = _tau_threshold_crossing(
        times_ns, msd_A2, l_sq, plat_std_A2=plat_std,
    )
    result["tau_ns"] = tau_ns
    if math.isfinite(tau_sem_ns) and tau_sem_ns > 0:
        result["tau_sem_ns"] = tau_sem_ns

    if math.isfinite(tau_ns) and tau_ns > 0:
        result["stage"] = "plateau+tau"
        # D_cage = l²/(6τ) in Å²/ns → m²/s
        D_cage_A2_ns = l_sq / (6.0 * tau_ns)
        D_cage_m2_s = D_cage_A2_ns * 1e-11
        # Propagate both l and τ uncertainties
        rel2 = 0.0
        if math.isfinite(l_sem_A) and l_sem_A > 0:
            rel2 += (2.0 * l_sem_A / l_A) ** 2
        if math.isfinite(tau_sem_ns) and tau_sem_ns > 0:
            rel2 += (tau_sem_ns / tau_ns) ** 2
        D_cage_sem = D_cage_m2_s * math.sqrt(rel2) if rel2 > 0 else math.nan
        result["D_cage_m2_s"] = D_cage_m2_s
        result["D_cage_sem_m2_s"] = D_cage_sem

    # Diagnostic: linearized τ fit (retained for comparison only; no longer
    # used as τ_sem proxy because its slope-SE overestimates uncertainty
    # near the plateau).
    tau_lin, tau_lin_sem, tau_lin_r2 = _fit_crossover_tau(
        times_ns, msd_A2, l_sq, t_skip_ns=t_skip_ns, r2_min=0.0,
    )
    result["tau_lin_ns"] = tau_lin
    result["tau_lin_sem_ns"] = tau_lin_sem
    result["tau_lin_r2"] = tau_lin_r2

    return result


def fit_confined_msd(
    times_ns: np.ndarray,
    msd_A2: np.ndarray,
    t_skip_ns: float = 5.0,
    r2_min: float = 0.85,
) -> Tuple[float, float, float, float, float]:
    """
    Fit MSD(t) = l^2*(1-exp(-t/tau)) to condensate-comoving MSD.

    Pure confinement model for droplet-geometry condensates where no
    diffusive regime exists in the comoving MSD.

    NOTE: This is the global nonlinear fit, retained for per-chain fits.
    For the headline chain-averaged result, use fit_confined_two_stage()
    which is more robust (plateau-first, then linearized tau).

    Fit parameters:
      - l (Å): confinement / localisation length
      - tau (ns): cage-escape / local relaxation timescale

    Derived (by caller): D_cage = l^2/(6*tau), local cage mobility.

    Returns:
        (l_A, l_sem_A, tau_ns, tau_sem_ns, fit_r2)
        All NaN if the fit fails or R^2 < r2_min.
    """
    NAN5 = (math.nan,) * 5
    t = np.asarray(times_ns, dtype=float)
    y = np.asarray(msd_A2, dtype=float)

    mask = np.isfinite(t) & np.isfinite(y) & (t > t_skip_ns) & (y > 0.0)
    t = t[mask]
    y = y[mask]
    if t.size < 15:
        return NAN5

    # Initial guesses from MSD shape
    # Plateau ~ early quasi-plateau (use values between 20-40% of time range)
    n = t.size
    i_lo, i_hi = n // 5, 2 * n // 5
    plateau_guess = float(np.median(y[i_lo:i_hi])) if i_hi > i_lo else float(np.median(y))
    if plateau_guess <= 0.0:
        plateau_guess = float(np.median(y))
    if plateau_guess <= 0.0:
        return NAN5

    # tau guess: time to reach 63% of early plateau
    frac63 = 0.632 * plateau_guess
    above = np.where(y >= frac63)[0]
    tau_guess = float(t[above[0]]) if above.size > 0 else float(t[n // 4])
    tau_guess = max(tau_guess, 1.0)

    try:
        popt, pcov = curve_fit(
            _pure_confined_model,
            t, y,
            p0=[plateau_guess, tau_guess],
            bounds=(
                [0.0, 1.0],
                [plateau_guess * 20.0, float(t[-1]) * 5.0],
            ),
            maxfev=20000,
        )
    except Exception:
        return NAN5

    l_sq, tau_ns = float(popt[0]), float(popt[1])
    if l_sq <= 0.0 or tau_ns <= 0.0:
        return NAN5

    y_pred = _pure_confined_model(t, l_sq, tau_ns)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    if r2 < r2_min:
        return NAN5

    perr = np.sqrt(np.diag(pcov))
    l_sq_sem, tau_sem_ns = float(perr[0]), float(perr[1])

    l_A = math.sqrt(l_sq)
    l_sem_A = l_sq_sem / (2.0 * l_A) if l_A > 0.0 else math.nan

    return l_A, l_sem_A, tau_ns, tau_sem_ns, r2


def per_chain_confined(
    times_ns: np.ndarray,
    msd_chain_A2: np.ndarray,
    Rg_A: float,
    Rg_sem_A: float,
    Rh_A: float,
    Rh_sem_A: float,
    T: float,
    t_skip_ns: float = 5.0,
    r2_min: float = 0.85,
) -> Dict[str, float]:
    """Per-chain pure confinement fit.

    Returns dict with l, tau, D_cage (= l^2/(6*tau)), Rg, Rh, fit_r2, fit_type.
    D_cage is local cage mobility, NOT long-time self-diffusion.
    """
    l_A, l_sem, tau_ns, tau_sem_ns, r2 = fit_confined_msd(
        times_ns, msd_chain_A2, t_skip_ns=t_skip_ns, r2_min=r2_min,
    )
    fit_type = "Confined" if math.isfinite(l_A) else "Rejected"

    tau_s = tau_ns * 1e-9 if math.isfinite(tau_ns) else math.nan
    tau_sem_s = tau_sem_ns * 1e-9 if math.isfinite(tau_sem_ns) else math.nan

    # D_cage = l^2 / (6*tau)  — local cage mobility in A^2/ns
    if math.isfinite(l_A) and l_A > 0.0 and math.isfinite(tau_ns) and tau_ns > 0.0:
        D_cage_A2_ns = l_A ** 2 / (6.0 * tau_ns)
        rel2 = 0.0
        if math.isfinite(l_sem) and l_sem > 0.0:
            rel2 += (2.0 * l_sem / l_A) ** 2
        if math.isfinite(tau_sem_ns) and tau_sem_ns > 0.0:
            rel2 += (tau_sem_ns / tau_ns) ** 2
        D_cage_sem_A2_ns = D_cage_A2_ns * math.sqrt(rel2) if rel2 > 0.0 else math.nan
        D_cage_m2_s = D_cage_A2_ns * 1e-11
        D_cage_sem_m2_s = D_cage_sem_A2_ns * 1e-11 if math.isfinite(D_cage_sem_A2_ns) else math.nan
    else:
        D_cage_m2_s = math.nan
        D_cage_sem_m2_s = math.nan

    return {
        "l_A": l_A,
        "l_A_sem": l_sem,
        "tau_s": tau_s,
        "tau_s_sem": tau_sem_s,
        "D_cage_m2_s": D_cage_m2_s,
        "D_cage_m2_s_sem": D_cage_sem_m2_s,
        "Rg_A": float(Rg_A),
        "Rg_A_sem": float(Rg_sem_A),
        "Rh_A": float(Rh_A),
        "Rh_A_sem": float(Rh_sem_A),
        "fit_r2": r2,
        "fit_type": fit_type,
    }


def run_diffusion_confined(
    msd_path: str,
    cluster_root: str,
    tag: str,
    temp_K: float,
    tmin: int,
    dt: int,
    tmax: int,
    t_skip_ns: float = 5.0,
    r2_min: float = 0.85,
    seed: int = 0,
    origin_resample: bool = DEFAULT_ORIGIN_RESAMPLE,
    origin_window_ns: float = DEFAULT_ORIGIN_WINDOW_NS,
    origin_stride_ns: float = DEFAULT_ORIGIN_STRIDE_NS,
    origin_candidate_windows_ns=None,
    origin_min_success_fraction: float = DEFAULT_ORIGIN_MIN_SUCCESS_FRACTION,
    origin_min_success_count: int = DEFAULT_ORIGIN_MIN_SUCCESS_COUNT,
    origin_max_origins: int = DEFAULT_ORIGIN_MAX_ORIGINS,
) -> Dict[str, object]:
    """
    Confined-diffusion estimator: fit MSD(t) = l^2*(1-exp(-t/tau)) per chain.

    Uses the same COM sidecar / comoving MSD infrastructure as run_diffusion
    but applies the single-exponential confined model instead of multiwindow
    linear fitting.

    The system-level l_conf/tau/D_cage central values are computed from the
    full available trajectory.  When COM sidecars exist, long subtrajectory
    time-origin resampling is used only for uncertainty/QC: it scans candidate
    block lengths to identify a stable plateau length, then adds a
    batch-corrected origin-start SEM to the existing chain-bootstrap SEM.

    Returns dict matching the Quant_Data output convention.
    """
    if not os.path.isfile(msd_path):
        raise FileNotFoundError(f"Confined MSD file not found: {msd_path}")

    print(f"[DIFFUSION-CONFINED] Using MSD file: {msd_path}")
    t_run0 = time.time()

    msd_format = "rdp" if msd_path.endswith("_msd_rdp.out.all") else "lammps"

    com_pos_A = None
    rg_samples_A = None
    rh_samples_A = None
    if msd_format == "rdp":
        sidecar_path = rdp_com_sidecar_path(msd_path)
        if os.path.isfile(sidecar_path):
            times_ns, com_pos_A, rg_samples_A, rh_samples_A, resids, resnames = parse_rdp_com_series(
                sidecar_path, t_start_ns=float(tmin),
            )
            print(f"[DIFFUSION-CONFINED] Using RCC COM trajectory sidecar: {sidecar_path}")
        else:
            times_ns, msd_mat_A2, resids, resnames, _ = parse_rdp_msd(
                msd_path, t_start_ns=float(tmin), downsample_stride=1,
            )
            com_pos_A = None
            print("[DIFFUSION-CONFINED] No COM sidecar; using precomputed MSD.")
    else:
        times_ns, com_pos_A, resids, resnames = parse_lammps_com_series(
            msd_path, t_start_ns=float(tmin),
        )

    resids_arr = np.asarray(resids, dtype=int)

    # RG / RH arrays
    mapping_mean: Dict[int, float] = {}
    mapping_sem: Dict[int, float] = {}
    if rg_samples_A is None and com_pos_A is not None:
        analysis_prefix = tag.split("_")[0].upper()
        rg_out_path = os.path.join(
            cluster_root, f"ANALYSIS_{analysis_prefix}", f"RG_{tag}_rg.out.all",
        )
        if os.path.isfile(rg_out_path):
            rg_override = parse_rg_out_all(rg_out_path, float(tmin), float(tmax), float(dt))
            if rg_override is not None:
                override_resids, override_mean_vals, override_sem_vals = rg_override
                mapping_mean = {int(r): float(v) for r, v in zip(override_resids, override_mean_vals)}
                mapping_sem = {int(r): float(v) for r, v in zip(override_resids, override_sem_vals)}

    if rg_samples_A is None:
        keep_mask = np.array([int(r) in mapping_mean for r in resids_arr], dtype=bool)
        if not keep_mask.any():
            keep_mask = np.ones(resids_arr.size, dtype=bool)
    else:
        keep_mask = np.ones(resids_arr.size, dtype=bool)

    resids_arr = resids_arr[keep_mask]
    if com_pos_A is not None:
        com_pos_A = np.asarray(com_pos_A, dtype=float)[:, keep_mask, :]

    if rg_samples_A is not None:
        rg_samples_A = np.asarray(rg_samples_A, dtype=float)[:, keep_mask]
        Rg_A = np.nanmean(rg_samples_A, axis=0)
        Rg_sem_A = (np.nanstd(rg_samples_A, axis=0, ddof=1) / math.sqrt(rg_samples_A.shape[0])
                     if rg_samples_A.shape[0] > 1 else np.full(Rg_A.shape, np.nan))
    else:
        Rg_A = np.array([mapping_mean.get(int(r), math.nan) for r in resids_arr])
        Rg_sem_A = np.array([mapping_sem.get(int(r), math.nan) for r in resids_arr])

    if rh_samples_A is not None:
        rh_samples_A = np.asarray(rh_samples_A, dtype=float)[:, keep_mask]
        Rh_A = np.nanmean(rh_samples_A, axis=0)
        Rh_sem_A = (np.nanstd(rh_samples_A, axis=0, ddof=1) / math.sqrt(rh_samples_A.shape[0])
                     if rh_samples_A.shape[0] > 1 else np.full(Rh_A.shape, np.nan))
    else:
        Rh_A = np.full(Rg_A.shape, np.nan)
        Rh_sem_A = np.full(Rg_A.shape, np.nan)

    # Window restriction
    window_end = float(tmax - tmin)
    if window_end > 0.0:
        win_mask = times_ns <= window_end + 1e-9
        times_ns = times_ns[win_mask]
        if com_pos_A is not None:
            com_pos_A = com_pos_A[win_mask, :, :]

    # Inside/outside classification
    max_resid = int(np.max(resids_arr)) if resids_arr.size else 0
    inside_resids, outside_resids = get_persistent_inside_outside(
        cluster_root=cluster_root, tag=tag, tmin=tmin, dt=dt, tmax=tmax,
        n_res=max_resid, t_start_analysis_ns=float(tmin),
    )
    inside_set = set(inside_resids)

    # Build comoving MSD
    center_mask = None
    if com_pos_A is not None:
        center_mask = np.array([int(r) in inside_set for r in resids_arr], dtype=bool)
        if not np.any(center_mask):
            center_mask = np.ones(resids_arr.size, dtype=bool)
        msd_mat_A2 = build_comoving_msd_from_com_series(com_pos_A, center_mask=center_mask)
    # else: msd_mat_A2 already loaded from precomputed file

    nsteps, nres = msd_mat_A2.shape
    print(f"[DIFFUSION-CONFINED] Fitting {nres} chains, {nsteps} lag points, t_skip={t_skip_ns} ns")

    # Per-chain confined fits
    rows: List[Dict[str, object]] = []
    for j in range(nres):
        resid = int(resids_arr[j])
        location = "Inside" if resid in inside_set else "Outside"
        res = per_chain_confined(
            times_ns, msd_mat_A2[:, j],
            Rg_A=float(Rg_A[j]), Rg_sem_A=float(Rg_sem_A[j]),
            Rh_A=float(Rh_A[j]), Rh_sem_A=float(Rh_sem_A[j]),
            T=temp_K, t_skip_ns=t_skip_ns, r2_min=r2_min,
        )
        res["resid"] = resid
        res["location"] = location
        res["species"] = classify_species(resid)
        rows.append(res)

    df = pd.DataFrame(rows)

    # Write per-chain CSV
    out_dir = os.path.dirname(msd_path)
    base = os.path.basename(msd_path)
    if base.endswith("_msd_rdp.out.all"):
        tag_out = base[:-len("_msd_rdp.out.all")]
    elif base.endswith("_msd.out.all"):
        tag_out = base[:-len("_msd.out.all")]
    else:
        tag_out = os.path.splitext(base)[0]
    csv_path = os.path.join(out_dir, f"{tag_out}_per_chain_diffusion_confined.csv")
    df.to_csv(csv_path, index=False)
    print(f"[DIFFUSION-CONFINED] Per-chain CSV: {csv_path}")

    # --- Per-chain aggregation (heterogeneity summaries) ---
    inside_df = df[df["location"] == "Inside"].copy()
    n_fit = int(inside_df["fit_type"].eq("Confined").sum())
    n_total = len(inside_df)
    print(f"[DIFFUSION-CONFINED] Inside chains: {n_fit}/{n_total} Confined fits accepted")

    def _agg(col):
        """Return (mean, SEM, median) over the finite values of inside_df[col]."""
        vals = inside_df[col].dropna()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return math.nan, math.nan, math.nan
        return float(vals.mean()), float(vals.std(ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else math.nan, float(vals.median())

    l_chain_mean, _, l_chain_med = _agg("l_A")
    tau_chain_mean, _, tau_chain_med = _agg("tau_s")
    Dcage_chain_mean, _, Dcage_chain_med = _agg("D_cage_m2_s")
    rg_mean, rg_sem, _ = _agg("Rg_A")
    rh_mean, rh_sem, _ = _agg("Rh_A")

    # --- Two-stage confinement analysis (primary headline result) ---
    # Stage 1: l_conf from plateau detection (robust)
    # Stage 2: tau_conf from linearized approach-to-plateau (conditional)
    inside_idx_arr = np.array([j for j in range(nres) if int(resids_arr[j]) in inside_set])
    if inside_idx_arr.size > 0:
        msd_avg_inside = np.nanmean(msd_mat_A2[:, inside_idx_arr], axis=1)
        two_stage = fit_confined_two_stage(
            times_ns, msd_avg_inside, t_skip_ns=t_skip_ns,
        )
    else:
        two_stage = fit_confined_two_stage(np.array([0.0]), np.array([0.0]))

    l_avg = two_stage["l_A"]
    l_avg_sem = two_stage["l_sem_A"]
    tau_avg_ns = two_stage["tau_ns"]
    tau_avg_s = tau_avg_ns * 1e-9 if math.isfinite(tau_avg_ns) else math.nan
    D_cage_avg_m2_s = two_stage["D_cage_m2_s"]
    D_cage_avg_sem_m2_s = two_stage["D_cage_sem_m2_s"]
    stage = two_stage["stage"]

    # Bootstrap SEM for headline l_conf and tau_conf (resample chains)
    tau_avg_sem_ns = two_stage.get("tau_sem_ns", math.nan)
    tau_avg_sem_s = tau_avg_sem_ns * 1e-9 if math.isfinite(tau_avg_sem_ns) else math.nan
    if inside_idx_arr.size >= 2:
        l_boot_sem, tau_boot_sem_ns, _, _ = _bootstrap_two_stage(
            times_ns, msd_mat_A2, inside_idx_arr,
            t_skip_ns=t_skip_ns, n_boot=200, seed=seed,
        )
        # Prefer chain-bootstrap SEM for tau when available; otherwise keep the
        # threshold-crossing analytic SEM from the full MSD.
        if math.isfinite(tau_boot_sem_ns):
            tau_avg_sem_ns = tau_boot_sem_ns
        tau_avg_sem_s = tau_avg_sem_ns * 1e-9 if math.isfinite(tau_avg_sem_ns) else math.nan
        # For l_conf, keep the larger of plateau SEM and bootstrap SEM
        if math.isfinite(l_boot_sem):
            l_avg_sem = max(l_avg_sem, l_boot_sem) if math.isfinite(l_avg_sem) else l_boot_sem

    origin_summary = _origin_resample_nan_result("disabled")
    origin_records: List[Dict[str, object]] = []
    if origin_resample and com_pos_A is not None and center_mask is not None and inside_idx_arr.size > 0:
        origin_candidates = origin_candidate_windows_ns
        if origin_candidates is None:
            origin_candidates = list(DEFAULT_ORIGIN_CANDIDATE_WINDOWS_NS)
        origin_candidates = set(_parse_origin_candidate_windows(origin_candidates))
        try:
            target_origin_window = float(origin_window_ns)
        except Exception:
            target_origin_window = math.nan
        if math.isfinite(target_origin_window) and target_origin_window > 0.0:
            origin_candidates.add(target_origin_window)
        origin_candidates = sorted(origin_candidates)
        origin_summary, origin_records = _time_origin_resample_two_stage(
            times_ns=times_ns,
            com_pos_A=com_pos_A,
            center_mask=center_mask,
            inside_idx_arr=inside_idx_arr,
            t_skip_ns=t_skip_ns,
            cv_max=0.20,
            candidate_windows_ns=origin_candidates,
            stride_ns=origin_stride_ns,
            min_success_fraction=origin_min_success_fraction,
            min_success_count=origin_min_success_count,
            max_origins=origin_max_origins,
        )
        if origin_records:
            origin_csv_path = os.path.join(out_dir, f"{tag_out}_time_origin_confinement.csv")
            pd.DataFrame(origin_records).to_csv(origin_csv_path, index=False)
            print(f"[DIFFUSION-CONFINED] Time-origin confinement CSV: {origin_csv_path}")
    elif origin_resample:
        origin_summary = _origin_resample_nan_result("missing_com_or_inside")

    l_origin_sem = origin_summary.get("origin_resample_l_sem_A", math.nan)
    tau_origin_sem_ns = origin_summary.get("origin_resample_tau_sem_ns", math.nan)
    D_origin_sem_m2_s = origin_summary.get("origin_resample_D_cage_sem_m2_s", math.nan)
    if math.isfinite(l_origin_sem):
        l_avg_sem = _combine_sem_quadrature(l_avg_sem, l_origin_sem)
    if math.isfinite(tau_origin_sem_ns):
        tau_avg_sem_ns = _combine_sem_quadrature(tau_avg_sem_ns, tau_origin_sem_ns)
    tau_avg_sem_s = tau_avg_sem_ns * 1e-9 if math.isfinite(tau_avg_sem_ns) else math.nan

    # Recompute D_cage SEM with the final l/tau SEMs, then add the independent
    # time-origin sensitivity term if it exists.
    D_prop_sem_m2_s = math.nan
    if (math.isfinite(tau_avg_ns) and tau_avg_ns > 0
            and math.isfinite(l_avg) and l_avg > 0
            and math.isfinite(D_cage_avg_m2_s)):
        rel_l = (2.0 * l_avg_sem / l_avg) if math.isfinite(l_avg_sem) else 0.0
        rel_tau = (tau_avg_sem_ns / tau_avg_ns) if math.isfinite(tau_avg_sem_ns) else 0.0
        D_prop_sem_m2_s = abs(D_cage_avg_m2_s) * math.sqrt(rel_l**2 + rel_tau**2) if (rel_l > 0 or rel_tau > 0) else math.nan
    D_cage_avg_sem_m2_s = _combine_sem_quadrature(D_prop_sem_m2_s, D_origin_sem_m2_s)

    print(f"[DIFFUSION-CONFINED] Two-stage result ({stage}):")
    tau_sem_str = f" ± {tau_avg_sem_ns:.1f}" if math.isfinite(tau_avg_sem_ns) else ""
    print(f"  l_conf    = {l_avg:.1f} ± {l_avg_sem:.1f} Å  (plateau CV={two_stage['plateau_cv']:.4f}, "
          f"n_plateau={two_stage['n_plateau']}, start={two_stage['plateau_start_ns']:.0f} ns)")
    if math.isfinite(tau_avg_ns):
        print(f"  tau_cage  = {tau_avg_ns:.1f}{tau_sem_str} ns  (threshold crossing + chain/time-origin SEM)")
        print(f"  D_cage    = {D_cage_avg_m2_s * 1e12:.3e} µm²/s")
    else:
        print(f"  tau_cage  = NaN (MSD never reached 0.632·l²)")
    origin_status = origin_summary.get("origin_resample_status", "not_run")
    if origin_status != "disabled":
        print("  origin QC = "
              f"{origin_status}, window={origin_summary.get('origin_resample_window_ns', math.nan):.0f} ns, "
              f"success={origin_summary.get('origin_resample_n_success', 0)}/"
              f"{origin_summary.get('origin_resample_n_total', 0)}, "
              f"min_plateau_window={origin_summary.get('origin_resample_min_plateau_window_ns', math.nan):.0f} ns")
    tau_lin = two_stage.get("tau_lin_ns", math.nan)
    tau_lin_r2 = two_stage.get("tau_lin_r2", math.nan)
    if math.isfinite(tau_lin):
        print(f"  tau_lin   = {tau_lin:.1f} ns  (diagnostic linearized fit, R²={tau_lin_r2:.4f})")

    # Log-log slope D (V1/V2 style): find early-time windows where d(log MSD)/d(log t) ~ 1
    if inside_idx_arr.size > 0:
        D_loglog_A2_ns, D_loglog_sem_A2_ns = loglog_multiwindow_D(
            times_ns, msd_avg_inside, t_skip_ns=t_skip_ns, seed=seed,
        )
    else:
        D_loglog_A2_ns = D_loglog_sem_A2_ns = math.nan
    D_loglog_m2_s = D_loglog_A2_ns * 1e-11 if math.isfinite(D_loglog_A2_ns) else math.nan
    D_loglog_sem_m2_s = D_loglog_sem_A2_ns * 1e-11 if math.isfinite(D_loglog_sem_A2_ns) else math.nan
    if math.isfinite(D_loglog_m2_s):
        print(f"  D_loglog  = {D_loglog_m2_s*1e12:.3e} µm²/s  (V1/V2-style log-log slope detection)")
    else:
        print(f"  D_loglog  = NaN  (no diffusive window found in log-log)")

    # --- Per-species confinement: per-chain two-stage fits + DL pooling ---
    # Each chain is fit independently by the two-stage plateau + threshold-
    # crossing algorithm, yielding (l_i, l_sem_i, tau_i, tau_sem_i) pairs.
    # The per-chain pairs are then pooled with DerSimonian–Laird random-
    # effects meta-analysis (DerSimonian & Laird, Controlled Clinical Trials
    # 1986; Higgins & Thompson, Stat Med 2002), which combines within-chain
    # fit uncertainty and between-chain heterogeneity τ² into a single SEM:
    #     SEM_pooled² = 1 / Σ 1/(sem_i² + τ²)
    # This unifies the SEM definition across all N_sp:
    #   - N_sp = 1: SEM = fit SEM of the single chain (no heterogeneity estimable)
    #   - N_sp ≥ 2: SEM ≈ 1/√(Σ w_i*) where w_i* includes heterogeneity
    # Literature standard in single-particle tracking and polymer transport
    # (Saxton, Biophys J 1997; Weeks et al., Science 2000; Doliwa & Heuer,
    # PRE 2000).  Replaces the ad-hoc n-branch chain-averaging + bootstrap
    # logic, which produced inconsistent SEMs across species with different
    # chain counts.
    # CV gate is loosened to 1.0: a single chain's MSD plateau window has
    # ~sqrt(N_sys)× the noise of the system-wide chain-averaged MSD, so the
    # strict cv_max=0.20 tuned for the system-level fit would reject
    # physically-valid plateaus purely on noise.  Chains without a valid
    # plateau are dropped (contribute nothing to μ̂ or τ²).  The observed
    # plateau CV is still available in the per-chain result for downstream QC.
    _SPECIES_ORDER = ["G3BP1", "PABP1", "TIA1", "TTP", "FUS", "TDP43", "RNA"]
    _SPECIES_INSIDE_FRAC = 0.80
    _SPECIES_CV_MAX = 1.0
    relaxed_inside, _ = get_persistent_inside_outside(
        cluster_root=cluster_root, tag=tag, tmin=tmin, dt=dt, tmax=tmax,
        n_res=max_resid, t_start_analysis_ns=float(tmin),
        inside_fraction=_SPECIES_INSIDE_FRAC,
    )
    relaxed_inside_set = set(relaxed_inside)
    per_species = {}
    print(f"[DIFFUSION-CONFINED] Per-species confinement (chain-averaged MSD → two-stage fit, "
          f"occupancy≥{_SPECIES_INSIDE_FRAC:.0%}, {len(relaxed_inside_set)} chains):")
    for sp in _SPECIES_ORDER:
        sp_inside_idx = np.array([j for j in range(nres)
                                  if int(resids_arr[j]) in relaxed_inside_set
                                  and classify_species(int(resids_arr[j])) == sp])
        n_chains = sp_inside_idx.size
        per_species[f"n_conf_{sp}"] = n_chains

        # Per-species Rg and Rh (from all relaxed-inside chains of this species)
        if n_chains >= 1:
            rg_sp_vals = Rg_A[sp_inside_idx]
            rh_sp_vals = Rh_A[sp_inside_idx]
            rg_sp_fin = rg_sp_vals[np.isfinite(rg_sp_vals)]
            rh_sp_fin = rh_sp_vals[np.isfinite(rh_sp_vals)]
            per_species[f"Rg_{sp}_A_mean"] = float(rg_sp_fin.mean()) if rg_sp_fin.size else math.nan
            per_species[f"Rg_{sp}_A_sem"] = (float(rg_sp_fin.std(ddof=1) / math.sqrt(rg_sp_fin.size))
                                              if rg_sp_fin.size > 1 else math.nan)
            per_species[f"Rh_{sp}_A_mean"] = float(rh_sp_fin.mean()) if rh_sp_fin.size else math.nan
            per_species[f"Rh_{sp}_A_sem"] = (float(rh_sp_fin.std(ddof=1) / math.sqrt(rh_sp_fin.size))
                                              if rh_sp_fin.size > 1 else math.nan)
        else:
            per_species[f"Rg_{sp}_A_mean"] = math.nan
            per_species[f"Rg_{sp}_A_sem"] = math.nan
            per_species[f"Rh_{sp}_A_mean"] = math.nan
            per_species[f"Rh_{sp}_A_sem"] = math.nan

        if n_chains >= 1:
            # Per-chain two-stage fits: collect (value, SEM) pairs.
            l_vals: List[float] = []
            l_sems: List[float] = []
            tau_vals: List[float] = []
            tau_sems: List[float] = []
            for ci in sp_inside_idx:
                ts_i = fit_confined_two_stage(
                    times_ns, msd_mat_A2[:, ci],
                    t_skip_ns=t_skip_ns, cv_max=_SPECIES_CV_MAX,
                )
                if math.isfinite(ts_i["l_A"]):
                    l_vals.append(ts_i["l_A"])
                    l_sems.append(ts_i["l_sem_A"])
                if math.isfinite(ts_i["tau_ns"]):
                    tau_vals.append(ts_i["tau_ns"])
                    tau_sems.append(ts_i.get("tau_sem_ns", math.nan))

            # DerSimonian–Laird random-effects pooling.
            l_sp, l_sp_sem, l_tau_sq, k_l = _dl_random_effects_pool(
                np.array(l_vals, dtype=float), np.array(l_sems, dtype=float),
            )
            tau_sp_ns, tau_sp_sem_ns, tau_tau_sq, k_tau = _dl_random_effects_pool(
                np.array(tau_vals, dtype=float), np.array(tau_sems, dtype=float),
            )

            # Fallback: if DL pooling failed for l (k_l == 0) because every
            # chain's plateau was too noisy, take the simple mean + SE of
            # whatever per-chain values are finite even without usable SEMs.
            if k_l == 0 and len(l_vals) > 0:
                lv = np.array(l_vals, dtype=float)
                lv = lv[np.isfinite(lv)]
                if lv.size > 0:
                    l_sp = float(lv.mean())
                    l_sp_sem = (float(lv.std(ddof=1) / math.sqrt(lv.size))
                                if lv.size > 1 else math.nan)
                    k_l = int(lv.size)
            if k_tau == 0 and len(tau_vals) > 0:
                tv = np.array(tau_vals, dtype=float)
                tv = tv[np.isfinite(tv)]
                if tv.size > 0:
                    tau_sp_ns = float(tv.mean())
                    tau_sp_sem_ns = (float(tv.std(ddof=1) / math.sqrt(tv.size))
                                     if tv.size > 1 else math.nan)
                    k_tau = int(tv.size)

            per_species[f"l_conf_{sp}_A_mean"] = l_sp
            per_species[f"l_conf_{sp}_A_sem"] = l_sp_sem
            tau_sp_s = tau_sp_ns * 1e-9 if math.isfinite(tau_sp_ns) else math.nan
            tau_sp_sem_s = tau_sp_sem_ns * 1e-9 if math.isfinite(tau_sp_sem_ns) else math.nan
            per_species[f"tau_conf_{sp}_s_mean"] = tau_sp_s
            per_species[f"tau_conf_{sp}_s_sem"] = tau_sp_sem_s

            rh_sp = per_species[f"Rh_{sp}_A_mean"]
            l_str = f"{l_sp:.1f}" if math.isfinite(l_sp) else "NaN"
            l_sem_str = f"{l_sp_sem:.1f}" if math.isfinite(l_sp_sem) else "NaN"
            tau_str = f"{tau_sp_ns:.1f}" if math.isfinite(tau_sp_ns) else "NaN"
            tau_sem_str = f" ± {tau_sp_sem_ns:.1f}" if math.isfinite(tau_sp_sem_ns) else ""
            fit_tag = f"DL pool [k_l={k_l}/{n_chains}, k_tau={k_tau}/{n_chains}]"
            print(f"  {sp:6s}: n={n_chains:3d} chains, "
                  f"l_conf={l_str} ± {l_sem_str} Å, "
                  f"tau_conf={tau_str}{tau_sem_str} ns, "
                  f"Rh={rh_sp:.1f} Å  ({fit_tag})")
        else:
            per_species[f"l_conf_{sp}_A_mean"] = math.nan
            per_species[f"l_conf_{sp}_A_sem"] = math.nan
            per_species[f"tau_conf_{sp}_s_mean"] = math.nan
            per_species[f"tau_conf_{sp}_s_sem"] = math.nan
            rh_sp = per_species[f"Rh_{sp}_A_mean"]
            rh_str = f"Rh={rh_sp:.1f} Å" if math.isfinite(rh_sp) else "Rh=NaN"
            print(f"  {sp:6s}: n={n_chains:3d} chains — skipped (n=0), {rh_str}")

    # --- Per-species spatial observables: cluster occupancy and radial position ---
    # These are independent of the MSD fit and always computed (even when confinement
    # fit rejects a species, e.g. TDP43). They capture whether a species is a core
    # scaffold, shell-adjacent, interfacial, or coacervate-like component of the SG.
    occ_chain, r_over_R_chain = compute_spatial_occupancy_radial(
        cluster_root=cluster_root, tag=tag, tmin=tmin, dt=dt, tmax=tmax,
        com_pos_A=com_pos_A, resids_arr=resids_arr, times_ns=times_ns,
    )
    print("[DIFFUSION-CONFINED] Per-species spatial occupancy / radial position:")
    for sp in _SPECIES_ORDER:
        sp_idx = np.array([j for j in range(nres)
                           if classify_species(int(resids_arr[j])) == sp])
        if sp_idx.size == 0:
            per_species[f"Occ_{sp}_mean"] = math.nan
            per_species[f"Occ_{sp}_sem"] = math.nan
            per_species[f"r_over_R_{sp}_mean"] = math.nan
            per_species[f"r_over_R_{sp}_sem"] = math.nan
            continue
        occ_sp = occ_chain[sp_idx]
        rr_sp = r_over_R_chain[sp_idx]
        occ_fin = occ_sp[np.isfinite(occ_sp)]
        rr_fin = rr_sp[np.isfinite(rr_sp)]
        occ_mean = float(occ_fin.mean()) if occ_fin.size else math.nan
        occ_sem = (float(occ_fin.std(ddof=1) / math.sqrt(occ_fin.size))
                   if occ_fin.size > 1 else math.nan)
        rr_mean = float(rr_fin.mean()) if rr_fin.size else math.nan
        rr_sem = (float(rr_fin.std(ddof=1) / math.sqrt(rr_fin.size))
                  if rr_fin.size > 1 else math.nan)
        per_species[f"Occ_{sp}_mean"] = occ_mean
        per_species[f"Occ_{sp}_sem"] = occ_sem
        per_species[f"r_over_R_{sp}_mean"] = rr_mean
        per_species[f"r_over_R_{sp}_sem"] = rr_sem
        occ_str = f"{100*occ_mean:5.1f}%" if math.isfinite(occ_mean) else "  NaN "
        rr_str = f"{rr_mean:.2f}" if math.isfinite(rr_mean) else " NaN"
        print(f"  {sp:6s}: n={sp_idx.size:3d}  occupancy={occ_str}  r/R_cluster={rr_str}")

    elapsed = time.time() - t_run0
    print(f"[DIFFUSION-CONFINED] Complete in {elapsed:.1f} s")

    result = {
        # Primary: two-stage confinement analysis
        "l_A_mean": l_avg,
        "l_A_sem": l_avg_sem,
        "tau_s_mean": tau_avg_s,
        "tau_s_sem": tau_avg_sem_s,
        "D_cage_m2_s": D_cage_avg_m2_s,
        "D_cage_m2_s_sem": D_cage_avg_sem_m2_s,
        "plateau_start_ns": two_stage["plateau_start_ns"],
        "plateau_cv": two_stage["plateau_cv"],
        "n_plateau": two_stage["n_plateau"],
        "confinement_stage": stage,
        # Log-log slope D (V1/V2 style early-time diffusive regime)
        "D_loglog_m2_s": D_loglog_m2_s,
        "D_loglog_m2_s_sem": D_loglog_sem_m2_s,
        # Diagnostic: linearized tau fit
        "tau_lin_ns": tau_lin,
        "tau_lin_r2": tau_lin_r2,
        # Chain-level heterogeneity (from per-chain global fits — secondary)
        "l_A_chain_median": l_chain_med if n_fit >= 5 else math.nan,
        "l_A_chain_mean": l_chain_mean if n_fit >= 5 else math.nan,
        "tau_s_chain_median": tau_chain_med if n_fit >= 5 else math.nan,
        "tau_s_chain_mean": tau_chain_mean if n_fit >= 5 else math.nan,
        "D_cage_m2_s_chain_median": Dcage_chain_med if n_fit >= 5 else math.nan,
        "D_cage_m2_s_chain_mean": Dcage_chain_mean if n_fit >= 5 else math.nan,
        # Radii (per-chain averages, always available)
        "Rg_A_mean": rg_mean,
        "Rg_A_sem": rg_sem,
        "Rh_A_mean": rh_mean,
        "Rh_A_sem": rh_sem,
        # Metadata
        "n_confined": n_fit,
        "n_total_inside": n_total,
        "csv_path": csv_path,
    }
    result.update(origin_summary)
    # Per-species confinement (comoving-frame pure model)
    result.update(per_species)
    return result


def parse_rg_out_all(
    rg_path: str,
    tmin_ns: float,
    tmax_ns: float,
    dt_ns: float,
) -> Optional[Tuple[List[int], np.ndarray, np.ndarray]]:
    """Parse RG output file (RG_<tag>_rg.out.all) and compute block-averaged per-chain Rg.

    Rg is averaged over blocks of width ``dt_ns`` between ``tmin_ns`` and ``tmax_ns``
    (both in ns). SEM is computed across those block means, mirroring the
    time-block averaging used elsewhere in the analysis pipeline.

    Returns:
        (resid_list, mean_Rg_A, sem_Rg_A) where Rg is in Å.
    """
    if not os.path.isfile(rg_path):
        return None

    times: List[float] = []
    blocks: List[List[float]] = []
    resid_list: Optional[List[int]] = None

    with open(rg_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) >= 2:
            try:
                time_ns = float(parts[0])
                nrows = int(parts[1])
            except ValueError:
                i += 1
                continue
            i += 1
            block: List[float] = []
            block_resids: List[int] = []
            for _ in range(nrows):
                if i >= len(lines):
                    break
                cols = lines[i].split()
                i += 1
                if len(cols) < 4:
                    continue
                try:
                    resid_in = int(cols[1])
                    rg_val = float(cols[3])
                except ValueError:
                    continue
                block_resids.append(resid_in)
                block.append(rg_val)
            if block:
                if resid_list is None:
                    resid_list = block_resids.copy()
                elif resid_list != block_resids:
                    # Remap to the first block's residue ordering
                    mapping = {r: g for r, g in zip(block_resids, block)}
                    block = [mapping.get(r, math.nan) for r in resid_list]
                times.append(time_ns)
                blocks.append(block)
        else:
            i += 1

    if not blocks or resid_list is None:
        return None

    arr = np.array(blocks, float)  # shape (n_times, n_residues)
    times_arr = np.array(times, float)

    # Select analysis window
    if tmax_ns > tmin_ns:
        win_mask = (times_arr >= tmin_ns) & (times_arr <= tmax_ns)
    else:
        # If tmax not meaningful, just use everything after tmin
        win_mask = times_arr >= tmin_ns

    times_sel = times_arr[win_mask]
    arr_sel = arr[win_mask, :]
    if arr_sel.size == 0 or times_sel.size == 0:
        return None

    # Block-average in time with block width dt_ns (if provided)
    if dt_ns and dt_ns > 0.0:
        block_means: List[np.ndarray] = []
        t0 = float(tmin_ns)
        t_end = float(tmax_ns) if tmax_ns > tmin_ns else float(times_sel[-1])
        # Small epsilon to ensure last block is included
        eps = 1e-8
        while t0 <= t_end + eps:
            t1 = t0 + dt_ns
            mask = (times_sel >= t0) & (times_sel < t1)
            if mask.any():
                block_means.append(np.nanmean(arr_sel[mask, :], axis=0))
            t0 += dt_ns
        if not block_means:
            return None
        blocks_for_stats = np.vstack(block_means)
    else:
        blocks_for_stats = arr_sel

    mean_vals = np.nanmean(blocks_for_stats, axis=0)
    if blocks_for_stats.shape[0] > 1:
        sem_vals = np.nanstd(blocks_for_stats, axis=0, ddof=1) / math.sqrt(blocks_for_stats.shape[0])
    else:
        sem_vals = np.full_like(mean_vals, np.nan, dtype=float)

    return resid_list, mean_vals, sem_vals


def get_lammps_species_name(resid: int) -> str:
    """
    Return short species name based on LAMMPS resid.
    
    G3BP1:   1-33
    PABP1:   34-49
    TIA1:    50-65
    TTP:     66-81
    FUS:     82-97
    TDP43:   98-113
    RNA:     114-134
    """
    if 1 <= resid <= 33:
        return "G3BP1"
    elif 34 <= resid <= 49:
        return "PABP1"
    elif 50 <= resid <= 65:
        return "TIA1"
    elif 66 <= resid <= 81:
        return "TTP"
    elif 82 <= resid <= 97:
        return "FUS"
    elif 98 <= resid <= 113:
        return "TDP43"
    elif 114 <= resid <= 134:
        return "RNA"
    else:
        return "Unknown"


def parse_lammps_com_series(
    msd_path: str,
    t_start_ns: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse raw LAMMPS chunk output into COM coordinate time series.

    The file stores chunk COM coordinates in the last three columns under the
    ``c_commsd`` labels. Those coordinates let us remove condensate drift
    explicitly and recompute a condensate-comoving self-MSD, which is more
    appropriate for LLPS transport than the lab-frame chunk MSD.

    Returns:
        times_ns: (n_steps,) rebased to 0 after ``t_start_ns`` trimming
        com_pos_A: (n_steps, n_res, 3) chunk COM coordinates in Angstrom
        resids: (n_res,)
        resnames: (n_res,)
    """
    if not os.path.isfile(msd_path):
        raise FileNotFoundError(msd_path)

    times: List[float] = []
    blocks: List[List[List[float]]] = []
    resid_list: Optional[List[int]] = None

    with open(msd_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            step = int(parts[0])
            nrows = int(parts[1])
            time_ns = step * 20.0 * 1e-6
            i += 1
            rows_pos: List[List[float]] = []
            rows_resid: List[int] = []
            for _ in range(nrows):
                if i >= len(lines):
                    break
                cols = lines[i].split()
                if len(cols) < 8:
                    i += 1
                    continue
                try:
                    resid = int(cols[0])
                    x = float(cols[-3])
                    y = float(cols[-2])
                    z = float(cols[-1])
                except (ValueError, IndexError):
                    i += 1
                    continue
                rows_resid.append(resid)
                rows_pos.append([x, y, z])
                i += 1
            if rows_pos:
                times.append(time_ns)
                blocks.append(rows_pos)
                if resid_list is None:
                    resid_list = rows_resid
        else:
            i += 1

    if not blocks or resid_list is None:
        raise RuntimeError(f"No COM coordinate data parsed from LAMMPS file: {msd_path}")

    nmin = min(len(r) for r in blocks)
    blocks = [r[:nmin] for r in blocks]
    com_pos_A = np.array(blocks, float)
    times_ns = np.array(times, float)
    resids = np.array(resid_list[:nmin], int)
    resnames = np.array([get_lammps_species_name(r) for r in resids], dtype=object)

    if t_start_ns > 0.0:
        idx_start = np.searchsorted(times_ns, t_start_ns)
        if idx_start >= len(times_ns):
            raise ValueError(f"t_start_ns={t_start_ns} ns is beyond available data (max={times_ns[-1]:.1f} ns)")
        times_ns = times_ns[idx_start:]
        com_pos_A = com_pos_A[idx_start:, :, :]
        times_ns = times_ns - times_ns[0]

    return times_ns, com_pos_A, resids, resnames


def parse_rdp_msd(
    msd_path: str,
    t_start_ns: float = 0.0,
    downsample_stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse <system_name>_msd_rdp.out.all produced by RDP_FINAL.calc_diffusivities().

    Args:
        msd_path: Path to RDP MSD file
        t_start_ns: Start time in ns (discard data before this, default 0.0)
        downsample_stride: Keep every Nth point (1=no downsampling, 2=half points, etc.)

    Returns:
        times_ns: (n_steps,) - time axis starting from 0 after discarding equilibration
        msd_mat_A2: (n_steps, n_res)
        resids: (n_res,)
        resnames: (n_res,)
        Rg_A: (n_res,)  mean Rg per residue in Å
    """
    if not os.path.isfile(msd_path):
        raise FileNotFoundError(msd_path)

    times: List[float] = []
    blocks: List[List[float]] = []
    resid_list: Optional[List[int]] = None
    resname_list: Optional[List[str]] = None
    rg_list: Optional[List[float]] = None

    with open(msd_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) >= 2:
            try:
                time_ns = float(parts[0])
                nrows = int(parts[1])
            except ValueError:
                i += 1
                continue
            i += 1
            rows_msd: List[float] = []
            rows_resid: List[int] = []
            rows_resname: List[str] = []
            rows_rg: List[float] = []
            for _ in range(nrows):
                if i >= len(lines):
                    break
                cols = lines[i].split()
                if len(cols) < 5:
                    i += 1
                    continue
                try:
                    resid = int(cols[1])
                    resname = cols[2]
                    msd_tot = float(cols[3])
                    rg = float(cols[4])
                except ValueError:
                    i += 1
                    continue
                rows_resid.append(resid)
                rows_resname.append(resname)
                rows_msd.append(msd_tot)
                rows_rg.append(rg)
                i += 1
            if rows_msd:
                times.append(time_ns)
                blocks.append(rows_msd)
                if resid_list is None:
                    resid_list = rows_resid
                    resname_list = rows_resname
                    rg_list = rows_rg
        else:
            i += 1

    if not blocks or resid_list is None or resname_list is None or rg_list is None:
        raise RuntimeError(f"No MSD data parsed from {msd_path}")

    nmin = min(len(r) for r in blocks)
    blocks = [r[:nmin] for r in blocks]
    msd_mat_A2 = np.array(blocks, float)  # (n_steps, n_res)
    times_ns = np.array(times, float)
    resids = np.array(resid_list[:nmin], int)
    resnames = np.array(resname_list[:nmin], str)
    Rg_A = np.array(rg_list[:nmin], float)

    # Discard equilibration period (t < t_start_ns)
    if t_start_ns > 0.0:
        idx_start = np.searchsorted(times_ns, t_start_ns)
        if idx_start >= len(times_ns):
            raise ValueError(f"t_start_ns={t_start_ns} ns is beyond available data (max={times_ns[-1]:.1f} ns)")
        times_ns = times_ns[idx_start:]
        msd_mat_A2 = msd_mat_A2[idx_start:, :]
        # Reset time axis to start at 0 and re-base MSD so that MSD(0) = 0
        # The COM MSD in the RDP file is cumulative from the original t=0,
        # so when we change the time origin we must also subtract the value
        # at the new origin, exactly as we do for the LAMMPS MSD.
        times_ns = times_ns - times_ns[0]
        msd_mat_A2 = msd_mat_A2 - msd_mat_A2[0:1, :]

    # Downsample if requested
    if downsample_stride > 1:
        times_ns = times_ns[::downsample_stride]
        msd_mat_A2 = msd_mat_A2[::downsample_stride, :]

    return times_ns, msd_mat_A2, resids, resnames, Rg_A


def rdp_com_sidecar_path(msd_path: str) -> str:
    """Map an RDP MSD output path to its center-of-mass sidecar ``*_msd_rdp_com.npz``."""
    if msd_path.endswith("_msd_rdp.out.all"):
        return msd_path[:-len("_msd_rdp.out.all")] + "_msd_rdp_com.npz"
    return os.path.splitext(msd_path)[0] + "_msd_rdp_com.npz"


def parse_rdp_com_series(
    sidecar_path: str,
    t_start_ns: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse native-grid RCC COM trajectory sidecar written alongside
    ``*_msd_rdp.out.all``.

    Returns:
        times_ns: (n_steps,) rebased to 0 after trimming
        com_pos_A: (n_steps, n_res, 3)
        rg_samples_A: (n_steps, n_res)
        rh_samples_A: (n_steps, n_res) — Kirkwood Rh; NaN array if absent
        resids: (n_res,)
        resnames: (n_res,)
    """
    if not os.path.isfile(sidecar_path):
        raise FileNotFoundError(sidecar_path)

    with np.load(sidecar_path, allow_pickle=False) as data:
        times_ns = np.asarray(data["times_ns"], dtype=float)
        com_pos_A = np.asarray(data["com_A"], dtype=float)
        rg_samples_A = np.asarray(data["rg_A"], dtype=float)
        resids = np.asarray(data["resids"], dtype=int)
        resnames = np.asarray(data["resnames"]).astype(str)
        # Backward compatibility: rh_A may not exist in older sidecars
        if "rh_A" in data:
            rh_samples_A = np.asarray(data["rh_A"], dtype=float)
        else:
            rh_samples_A = np.full_like(rg_samples_A, math.nan)

    if t_start_ns > 0.0:
        idx_start = np.searchsorted(times_ns, t_start_ns)
        if idx_start >= len(times_ns):
            raise ValueError(f"t_start_ns={t_start_ns} ns is beyond available data (max={times_ns[-1]:.1f} ns)")
        times_ns = times_ns[idx_start:]
        com_pos_A = com_pos_A[idx_start:, :, :]
        rg_samples_A = rg_samples_A[idx_start:, :]
        rh_samples_A = rh_samples_A[idx_start:, :]
        times_ns = times_ns - times_ns[0]

    return times_ns, com_pos_A, rg_samples_A, rh_samples_A, resids, resnames


def build_comoving_msd_from_com_series(
    com_pos_A: np.ndarray,
    center_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convert residue COM coordinates into condensate-comoving MSD curves.

    Args:
        com_pos_A: (n_time, n_res, 3) residue COM coordinates in Angstrom.
        center_mask: optional boolean mask of length n_res selecting the residues
            that define the condensate translation to remove. If omitted or empty,
            all residues are used.
    """
    pos = np.asarray(com_pos_A, dtype=float)
    if pos.ndim != 3 or pos.shape[2] != 3:
        raise ValueError("com_pos_A must have shape (n_time, n_res, 3)")
    if pos.shape[1] == 0:
        return np.empty((pos.shape[0], 0), dtype=float)

    if center_mask is None:
        center_mask = np.ones(pos.shape[1], dtype=bool)
    else:
        center_mask = np.asarray(center_mask, dtype=bool)
        if center_mask.size != pos.shape[1]:
            raise ValueError("center_mask length must match number of residues")
        if not np.any(center_mask):
            center_mask = np.ones(pos.shape[1], dtype=bool)

    center_traj = np.mean(pos[:, center_mask, :], axis=1, keepdims=True)
    pos_rel = pos - center_traj
    return _msd_fft_total_from_positions(pos_rel)


def _load_tracked_cluster_npz(
    cluster_root: str,
    tag: str,
    tmin: int,
    dt: int,
    tmax: int,
) -> Optional[dict]:
    """Try to load Tracked_Cluster NPZ sidecars for all analysis windows.

    Returns a dict with:
        sg_resids: int array of biopolymer resid axis
        inside_masks: list of (n_frames_in_window, n_res) bool arrays
        n_windows: number of windows loaded
    or None if no sidecars found.
    """
    analysis_prefix = tag.split("_")[0].upper()
    analysis_dir = os.path.join(cluster_root, "ANALYSIS_{}".format(analysis_prefix))

    start_t = int(float(tmin))
    sg_resids = None
    inside_masks = []
    n_loaded = 0

    for t_ns in range(start_t, int(tmax) + 1, int(dt)):
        fname = _resolve_time_file(
            lambda label: os.path.join(analysis_dir, f"Tracked_Cluster_{tag}_{label}.npz"),
            t_ns,
        )
        if fname is None:
            continue
        data = np.load(fname, allow_pickle=False)
        if sg_resids is None:
            sg_resids = data["sg_resids"]
        inside_masks.append(data["inside_mask"])  # (n_frames, n_res) bool
        n_loaded += 1

    if n_loaded == 0:
        return None
    return {
        "sg_resids": sg_resids,
        "inside_masks": inside_masks,
        "n_windows": n_loaded,
    }


def compute_spatial_occupancy_radial(
    cluster_root: str,
    tag: str,
    tmin: int,
    dt: int,
    tmax: int,
    com_pos_A: np.ndarray,
    resids_arr: np.ndarray,
    times_ns: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-chain cluster-occupancy fraction and mean normalized radial position r/R_cluster.

    - occ[j]    = fraction of frames chain j is inside the tracked cluster
                  (after equilibration cut; frames where cluster has < 3 members are skipped).
    - r_over_R[j] = mean (over frames where chain j is inside) of |COM_j − COM_cluster| / R_rms,
                    where R_rms is the RMS radial spread of the currently-inside chains
                    (a cluster-size proxy analogous to Rg_condensate).

    Returns (occ, r_over_R), each shape (n_res,). NaN if no data.
    """
    n_res = resids_arr.size
    occ = np.full(n_res, math.nan)
    r_over_R = np.full(n_res, math.nan)

    npz = _load_tracked_cluster_npz(cluster_root, tag, tmin, dt, tmax)
    if npz is None or com_pos_A is None:
        return occ, r_over_R

    # Concatenate per-window inside_masks into a single (nF_tc, n_sg) bool array.
    sg_resids = np.asarray(npz["sg_resids"], dtype=int)
    inside_all = np.concatenate(npz["inside_masks"], axis=0)
    nF_tc = inside_all.shape[0]

    # Align time axes: COM sidecar and tracked-cluster series share 1 ns cadence.
    # Use the shorter of the two, from the front.
    n_use = min(nF_tc, com_pos_A.shape[0], times_ns.size)
    inside_all = inside_all[:n_use]
    com = com_pos_A[:n_use]

    # Map resids_com → column in inside_all.
    resid_to_col = {int(r): i for i, r in enumerate(sg_resids)}
    cols = np.array([resid_to_col.get(int(r), -1) for r in resids_arr], dtype=int)
    valid_map = cols >= 0
    if not np.any(valid_map):
        return occ, r_over_R

    inside_aligned = np.zeros((n_use, n_res), dtype=bool)
    inside_aligned[:, valid_map] = inside_all[:, cols[valid_map]]

    # Per-frame cluster COM and R_rms (over currently-inside chains in this residue subset).
    n_inside_pf = inside_aligned.sum(axis=1)
    good_frames = n_inside_pf >= 3
    if not np.any(good_frames):
        return occ, r_over_R

    r_norm = np.full((n_use, n_res), math.nan)
    for t_idx in np.nonzero(good_frames)[0]:
        mask = inside_aligned[t_idx]
        positions = com[t_idx, mask]
        ccom = positions.mean(axis=0)
        d2 = ((positions - ccom) ** 2).sum(axis=1)
        Rrms = float(math.sqrt(d2.mean())) if d2.size else 0.0
        if Rrms <= 0.0 or not math.isfinite(Rrms):
            continue
        d_all = np.linalg.norm(com[t_idx] - ccom, axis=1)
        r_norm[t_idx] = d_all / Rrms

    n_good = int(good_frames.sum())
    for j in range(n_res):
        if not valid_map[j]:
            continue
        occ[j] = float(inside_aligned[good_frames, j].mean()) if n_good > 0 else math.nan
        mask_j = inside_aligned[:, j] & good_frames
        if mask_j.any():
            vals = r_norm[mask_j, j]
            vals = vals[np.isfinite(vals)]
            r_over_R[j] = float(vals.mean()) if vals.size else math.nan

    return occ, r_over_R


def get_persistent_inside_outside(
    cluster_root: str,
    tag: str,
    tmin: int,
    dt: int,
    tmax: int,
    n_res: int,
    t_start_analysis_ns: float = 0.0,
    inside_fraction: float = DEFAULT_INSIDE_OCCUPANCY,
    outside_fraction: float = DEFAULT_OUTSIDE_OCCUPANCY,
) -> Tuple[List[int], List[int]]:
    """
    Determine residues that are persistently inside the cluster and persistently outside.

    Prefers Tracked_Cluster NPZ sidecars (per-frame membership) when available,
    falling back to legacy Max_Continuous_Cluster text files (per-window membership).

    The occupancy policy classifies a residue as:
      - Inside if it belongs to the tracked condensate in at least
        ``inside_fraction`` of analyzed windows.
      - Outside if it belongs in at most ``outside_fraction`` of windows.
      - Both/ambiguous otherwise.

    When NPZ sidecars are available, a window counts as "inside" for a residue
    if that residue is present in at least 95% of the tracked frames within
    that window (matching the persistence threshold used by RCC_ANALYSIS).

    Args:
        t_start_analysis_ns: Only consider cluster files with t >= this value (default 0.0).
        inside_fraction: occupancy threshold for persistent-inside classification.
        outside_fraction: occupancy threshold for persistent-outside classification.
    """
    if n_res <= 0:
        return [], []

    all_resids = set(range(1, n_res + 1))

    # Try NPZ sidecars first (per-frame tracked membership)
    npz_data = _load_tracked_cluster_npz(cluster_root, tag, tmin, dt, tmax)
    if npz_data is not None:
        sg_resids = npz_data["sg_resids"]
        resid_to_col = {int(r): j for j, r in enumerate(sg_resids)}
        counts = {resid: 0 for resid in all_resids}
        n_windows = 0
        frame_threshold_frac = 0.95  # same as RCC persistence threshold

        for mask in npz_data["inside_masks"]:
            # mask shape: (n_frames_in_window, n_res)
            n_frames = mask.shape[0]
            frame_threshold = max(1, int(math.ceil(frame_threshold_frac * n_frames)))
            frame_counts = np.sum(mask, axis=0)  # per-resid frame count
            n_windows += 1
            for resid in all_resids:
                col = resid_to_col.get(resid)
                if col is not None and frame_counts[col] >= frame_threshold:
                    counts[resid] += 1

        if n_windows > 0:
            inside_threshold = max(1, int(math.ceil(float(inside_fraction) * n_windows)))
            outside_threshold = int(math.floor(float(outside_fraction) * n_windows))
            inside = sorted(resid for resid, count in counts.items() if count >= inside_threshold)
            outside = sorted(resid for resid, count in counts.items() if count <= outside_threshold)
            return inside, outside

    # Legacy fallback: Max_Continuous_Cluster text files
    analysis_prefix = tag.split("_")[0].upper()
    analysis_dir = os.path.join(cluster_root, "ANALYSIS_{}".format(analysis_prefix))
    start_t = int(max(float(tmin), float(t_start_analysis_ns)))

    counts = {resid: 0 for resid in all_resids}
    n_windows = 0
    for t_ns in range(start_t, tmax + 1, dt):
        fname = _resolve_time_file(
            lambda label: os.path.join(analysis_dir, f"Max_Continuous_Cluster_{tag}_{label}.txt"),
            t_ns,
        )
        if fname is None:
            continue
        with open(fname, "r") as f:
            txt = f.read()
        nums = re.findall(r"resid\s+(\d+)", txt)
        if not nums:
            nums = re.findall(r"\b(\d+)\b", txt)
        inner = {int(x) for x in nums}
        n_windows += 1
        for resid in inner:
            if resid in counts:
                counts[resid] += 1

    if n_windows == 0:
        return [], []

    inside_threshold = max(1, int(math.ceil(float(inside_fraction) * n_windows)))
    outside_threshold = int(math.floor(float(outside_fraction) * n_windows))

    inside = sorted(resid for resid, count in counts.items() if count >= inside_threshold)
    outside = sorted(resid for resid, count in counts.items() if count <= outside_threshold)
    return inside, outside


def classify_species(resid: int) -> str:
    """Map residue index to short species name."""
    return get_lammps_species_name(resid)
