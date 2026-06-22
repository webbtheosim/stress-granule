"""sm_common.py (compute) — correlation-corrected statistics for the unified
ANALYSIS_SM pipeline.

CORRELATION-CORRECTED UNCERTAINTY (identical method to the main pipeline):
imports BLOCK_CORRELATION_DIAGNOSTICS and reuses run_pymbar (statistical
inefficiency g), compute_superblock_tables (Flyvbjerg-Petersen plateau), and
compute_all_offset_batch_estimator (all-offset batch SEM). Block size is
max(superblock_plateau, ceil(g)); CI95 is Student-t with df=n_blocks-1.
This is the same STATISTICAL_POLICY used for Quant_Data.csv.

The house-style plotting helpers that previously lived alongside these
statistics now live in ``plotting/sm/sm_common.py`` (they re-export the
functions below).
"""
import os
import sys
import math
import numpy as np

# --- make the repo-root modules importable (BLOCK_CORRELATION_DIAGNOSTICS) -----
# This file lives at <repo>/analysis/sm/, so the repo root is three levels up.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# BLOCK_CORRELATION_DIAGNOSTICS lives under analysis/.
_ANALYSIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, _ANALYSIS_DIR)

try:
    import block_correlation_diagnostics as _BCD
    _HAVE_BCD = True
except Exception as _exc:  # pragma: no cover
    _BCD = None
    _HAVE_BCD = False
    print(f"[sm_common] WARNING: could not import BLOCK_CORRELATION_DIAGNOSTICS ({_exc}); "
          f"falling back to naive SEM.")

try:
    from scipy.stats import t as _student_t
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ----------------------------------------------------------------------------
# Correlation-corrected statistics (same policy as the main pipeline)
# ----------------------------------------------------------------------------
def correlated_stats(values, conservative=True, max_block=None,
                     min_superblocks=2, plateau_fraction=0.95):
    """Correlation-corrected mean / SEM / CI95 for a single time-correlated series.

    Mirrors BLOCK_CORRELATION_DIAGNOSTICS:
      - g  = pymbar statistical inefficiency (run_pymbar)
      - b_FP = Flyvbjerg-Petersen superblock plateau (compute_superblock_tables)
      - block = max(b_FP, ceil(g))                      (conservative)
      - mean/SEM via all-offset batch estimator (median SEM across offsets)
      - CI95 = t_{0.975, df=n_blocks-1} * SEM

    Returns dict with mean, sem, ci95, g, block_size, n_blocks, method.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    out = {"mean": math.nan, "sem": math.nan, "ci95": math.nan,
           "g": math.nan, "block_size": 0, "n_blocks": 0, "n": int(n),
           "method": "correlation_corrected_flyvbjerg_petersen_pymbar"}
    if n == 0:
        return out
    if n == 1 or np.nanstd(v) == 0:
        out.update(mean=float(v.mean()), sem=0.0, ci95=0.0, g=1.0,
                   block_size=1, n_blocks=int(n))
        return out

    if not _HAVE_BCD:
        # graceful naive fallback
        mean = float(v.mean())
        sem = float(np.std(v, ddof=1) / math.sqrt(n))
        out.update(mean=mean, sem=sem, ci95=1.96 * sem, g=math.nan,
                   block_size=1, n_blocks=int(n), method="naive_sem_fallback")
        return out

    starts = np.arange(n, dtype=int)
    ends = starts.copy()

    # 1) pymbar statistical inefficiency
    pj = _BCD.run_pymbar(v, conservative)
    g = pj.g if (getattr(pj, "available", False) and np.isfinite(getattr(pj, "g", math.nan))) else 1.0

    # 2) Flyvbjerg-Petersen superblock plateau
    maxb = max_block if max_block else max(2, n // 4)
    try:
        _, _, rec_b, _ = _BCD.compute_superblock_tables(
            v, starts, int(maxb), int(min_superblocks), float(plateau_fraction))
    except Exception:
        rec_b = 1
    rec_b = int(rec_b) if rec_b else 1

    # 3) conservative block size
    block = max(rec_b, int(math.ceil(g)) if np.isfinite(g) else 1, 1)
    block = max(1, min(block, n // 2 if n >= 2 else 1))

    # 4) all-offset batch estimator
    _, _, inf = _BCD.compute_all_offset_batch_estimator(v, starts, ends, block, 1)
    mean = float(inf.get("corrected_mean", v.mean()))
    sem = float(inf.get("corrected_sem", math.nan))
    nb = int(inf.get("n_corrected_blocks", 0))
    if not np.isfinite(sem):  # fall back to g-corrected SEM if batch estimator degenerate
        sd = float(np.std(v, ddof=1))
        sem = sd * math.sqrt(g / n) if np.isfinite(g) else sd / math.sqrt(n)
        nb = max(1, int(math.floor(n / max(g, 1.0))))

    if nb > 1 and np.isfinite(sem):
        if _HAVE_SCIPY:
            ci95 = float(_student_t.ppf(0.975, df=nb - 1)) * sem
        else:
            ci95 = 1.96 * sem
    else:
        ci95 = math.nan

    out.update(mean=mean, sem=sem, ci95=ci95, g=float(g),
               block_size=int(block), n_blocks=int(nb))
    return out


def combine_class_means(per_compound_means):
    """Class-level mean and SEM across independent compounds.

    Each compound mean is already correlation-corrected within-run, so the
    class average is the simple mean and the class SEM is the between-compound
    std/sqrt(k) (the main pipeline's class-average rule: 'SEM across system
    corrected means'). Returns (mean, sem, ci95, k).
    """
    m = np.asarray([x for x in per_compound_means if np.isfinite(x)], dtype=float)
    k = m.size
    if k == 0:
        return math.nan, math.nan, math.nan, 0
    if k == 1:
        return float(m[0]), math.nan, math.nan, 1
    mean = float(np.mean(m))
    sem = float(np.std(m, ddof=1) / math.sqrt(k))
    if _HAVE_SCIPY:
        ci95 = float(_student_t.ppf(0.975, df=k - 1)) * sem
    else:
        ci95 = 1.96 * sem
    return mean, sem, ci95, k
