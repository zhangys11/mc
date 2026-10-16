"""
core.py -- shared Monte Carlo kernel for the 14-test panel (COST-D-26-00503R1 revision).

This module is the long-lived home of the test panel used by the revision of
"Finite-Sample Adequacy and Robustness of Null Distributions".  It is imported by
``rev_study1_ext.py`` (t1), ``rev_alpha_dependence.py`` (t1), ``rev_sensitivity.py``
(t2) and ``rev_power.py`` (t3).

Design rules (see docs/DESIGN_CONTRACT.md, Sections 1-3):

* The ten archive generators (T1--T10) are imported *unchanged* from ``experiments.py``
  and re-exported here.  Nothing in this module re-implements an existing statistic, so
  the RNG stream of the archived Study 1 is reproducible bit-for-bit (see
  ``rev_study1_ext.py``).
* Every test has a registry entry :data:`TESTS` describing its theoretical null
  distribution, the rejection rule and whether the reference is sample-size independent.
* :func:`_evaluate` is the single evaluation path used by every study, so that
  ``D_KS`` and ``\\hat\\alpha`` have one definition across the whole paper:

      ks      = sup_x |F_hat_N(x) - F_0(x)|          (Kolmogorov-Smirnov distance)
      type1   = (1/N) * sum_j 1{stat_j in rejection region at level alpha}
      ci_lo/hi= Wilson 95% score interval for type1

* New statistics (T11--T13) use *sample-size independent* reference distributions.
  Where a statistic must be standardized first, the standardization is documented in the
  generator docstring and in :data:`TESTS`.

Conventions inherited from ``experiments.py`` (do not change silently)
---------------------------------------------------------------------
* ``N = 10_000`` replications and seed ``2024`` for the published grids.
* Significance threshold for ``D_KS``: ``1.358 / sqrt(N)`` (95% KS critical value).
* ``n^*_alpha``: smallest ``n`` with ``|alpha_hat - alpha| < 0.01`` (operational rule).
* ``n^*_KS``: smallest ``n`` with ``D_KS < 1.358/sqrt(N)``.
* Wilson 95% interval is the archive :func:`experiments.wilson_ci`, re-exported here.

Author: t1-new-tests (task-1).
"""

from __future__ import annotations

import functools
import json
import os
import platform
import subprocess
import sys
import time
import warnings

import numpy as np
from scipy import stats
from scipy import integrate
from scipy.special import gammaln, kv
from scipy.optimize import brentq

# Headless-safe: experiments.py imports matplotlib.pyplot at import time.
os.environ.setdefault("MPLBACKEND", "Agg")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. Archive generators and metric conventions (imported, never re-implemented)
# ---------------------------------------------------------------------------
# The archived generator module is ``experiments.py`` in the submission layout.  In the
# public repository it is shipped as ``archive_experiments.py`` inside ``code/`` so that
# the file does not shadow the identically named script at the paper directory's root,
# which is the frozen original-submission record.  Import whichever is present, under the
# canonical name, so that every downstream import keeps working unchanged.
try:  # pragma: no cover - layout-dependent
    import experiments  # noqa: F401  (submission layout)
except ModuleNotFoundError:  # pragma: no cover - layout-dependent
    import archive_experiments as experiments  # noqa: F401
    sys.modules["experiments"] = experiments

from experiments import (  # noqa: E402  (import after MPLBACKEND is set)
    mc_t_test,
    mc_anova,
    mc_kruskal_wallis,
    mc_chisq_gof,
    mc_bartlett,
    mc_fligner_killeen,
    mc_sign_test,
    mc_cochran_q,
    mc_median_test,
    mc_hotelling_t2,
    compute_ks,
    compute_kl,
    get_n_total,
)
from experiments import wilson_ci  # re-exported unchanged (identical semantics)
from experiments import compute_type1_error as _archive_type1_error

__all__ = [
    "TESTS", "REFERENCE_DISTS",
    "mc_t_test", "mc_anova", "mc_kruskal_wallis", "mc_chisq_gof", "mc_bartlett",
    "mc_fligner_killeen", "mc_sign_test", "mc_cochran_q", "mc_median_test",
    "mc_hotelling_t2", "mc_kpss", "mc_hoeffding_d", "mc_interaction_h",
    "mc_t_test_ar1", "mc_sign_test_ar1", "mc_anova_clustered",
    "mc_kruskal_wallis_clustered", "mc_cochran_q_corr",
    "compute_ks", "compute_kl", "ks_distance", "wilson_ci", "compute_type1_error",
    "reject_rate", "critical_value", "reference_cdf", "reference_sf", "_evaluate",
    "evaluate_multi_alpha", "cvm_limit_cdf", "cvm_limit_ppf", "cvm_limit_quantile",
    "hoeffding_z_sf", "hoeffding_z_ppf", "_hoeffding_ref_moments", "run_convergence", "make_meta",
    "save_results", "update_results", "SEVERITY_HOOKS", "n_star_alpha", "n_star_ks", "_selftest",
]

SEED = 2024
N_MC = 10_000
ALPHAS = (0.01, 0.05, 0.10)
KS_THRESHOLD = 1.358 / np.sqrt(N_MC)  # 0.013580 (archive convention)
EVAL_BUDGET = "-0.01"  # n*_alpha band: |alpha_hat - alpha| < 0.01


# ---------------------------------------------------------------------------
# 2. Reference distributions that scipy does not provide
# ---------------------------------------------------------------------------

def cvm_limit_cdf(x):
    """Exact CDF of the Cramer-von Mises limit law (Brownian-bridge functional).

    The limit distribution of the KPSS level-stationarity statistic is
    ``W^2 = int_0^1 V(r)^2 dr`` with ``V`` a standard Brownian bridge, i.e. the
    Cramer-von Mises limit law (also the limit law of the CvM goodness-of-fit
    statistic).  Its CDF is evaluated with the exact series of Csorgo & Faraway
    (1996, JRSS-B 58, 221-234), as implemented in the R package ``goftest``
    (``Vinf`` in ``R/cramer.R``):

        F(x) = 1/(pi^(3/2) sqrt(x)) * sum_{k>=0} Gamma(k+1/2)/Gamma(k+1)
                        * sqrt(4k+1) * exp(-q_k) * K_{1/4}(q_k),
        q_k  = (4k+1)^2 / (16 x)

    (using ``(-1)^k C(-1/2, k) = Gamma(k+1/2)/(sqrt(pi) Gamma(k+1))``).
    Published critical values: x(0.90)=0.34730, x(0.95)=0.46136, x(0.975)=0.58061,
    x(0.99)=0.74346; cross-checked against ``goftest::pCvM`` in
    ``rev_study1_ext.py --validate``.

    Parameters
    ----------
    x : float or array_like
        Non-negative argument(s).

    Returns
    -------
    numpy.ndarray or float
        ``P(W^2 <= x)``.
    """
    x = np.asarray(x, dtype=float)
    scalar = x.ndim == 0
    x = np.atleast_1d(x)
    out = np.zeros_like(x)
    pos = x > 0
    xp = x[pos]
    total = np.zeros_like(xp)
    for k in range(0, 300):
        a = 4.0 * k + 1.0
        q = a * a / (16.0 * xp)
        logt = (gammaln(k + 0.5) - gammaln(k + 1.0) + 0.5 * np.log(a)
                - 1.5 * np.log(np.pi) - 0.5 * np.log(xp) - q + np.log(kv(0.25, q)))
        term = np.exp(logt)
        total += term
        if k > 5 and np.max(term) < 1e-15:
            break
    out[pos] = np.clip(total, 0.0, 1.0)
    return float(out[0]) if scalar else out


def cvm_limit_ppf(q):
    """UPPER-tail quantile of the Cramer-von Mises limit law.

    .. warning::
       The argument ``q`` is an **upper-tail probability**, following the convention of
       :func:`reference_ppf` and of every ``type1 == 'upper_tail'`` entry in
       :data:`TESTS`: the return value is ``inf{x : P(W^2 > x) <= q}``.  Callers that
       want the ordinary (lower) quantile must pass ``1 - p``.  Use
       :func:`cvm_limit_quantile` if a lower-tail quantile is what you want.

    Parameters
    ----------
    q : float
        Upper-tail probability in (0, 1); ``q = 0.05`` gives the 5% critical value of
        the KPSS level-stationarity test.

    Returns
    -------
    float
        ``inf{x : cvm_limit_cdf(x) >= 1 - q}``, obtained by Brent root finding.
    """
    q = float(q)
    if not 0.0 < q < 1.0:
        raise ValueError("q must lie in (0, 1)")
    lo, hi = 1e-12, 100.0
    return float(brentq(lambda z: cvm_limit_cdf(z) - (1.0 - q), lo, hi, xtol=1e-12))


def cvm_limit_quantile(p):
    """Ordinary (lower-tail) quantile of the Cramer-von Mises limit law.

    ``p = 0.95`` returns 0.46136, the tabulated 5% upper critical value of the KPSS
    level-stationarity test (KPSS 1992, Table 1).
    """
    p = float(p)
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie in (0, 1)")
    return float(brentq(lambda z: cvm_limit_cdf(z) - p, 1e-12, 100.0, xtol=1e-12))


# Hmisc::hoeffd asymptotic P-value table (Hollander & Wolfe 1973, pp. 228-235, 423;
# reproduced verbatim from Hmisc 5.3-0, R/hoeffd.s).  Values are survival
# probabilities P(Z > z) for the scaled Hoeffding statistic
#     z = 0.5 * pi^4 * n * (D_Hmisc + 1/(36 n)),   D_Hmisc = 30 * d_Hoeffding,
# evaluated at z = 1.1, 1.15, ..., 5.0, 5.5, 6.0, ..., 8.5.
_HOEFFD_Z_GRID = np.concatenate([np.arange(1.1, 5.0 + 1e-9, 0.05),
                                 np.arange(5.5, 8.5 + 1e-9, 0.5)])
_HOEFFD_Z_TABLE = np.array([
    5297, 4918, 4565, 4236, 3930,
    3648, 3387, 3146, 2924, 2719, 2530, 2355,
    2194, 2045, 1908, 1781, 1663, 1554, 1453,
    1359, 1273, 1192, 1117, 1047, 982, 921,
    864, 812, 762, 716, 673, 633, 595,
    560, 527, 496, 467, 440, 414, 390,
    368, 347, 327, 308, 291, 274, 259,
    244, 230, 217, 205, 194, 183, 173,
    163, 154, 145, 137, 130, 123, 116,
    110, 104, 98, 93, 87, 83, 78,
    74, 70, 66, 63, 59, 56, 53,
    50, 47, 45, 42, 25, 14, 8,
    5, 3, 2, 1]) / 10_000.0
assert _HOEFFD_Z_GRID.size == _HOEFFD_Z_TABLE.size == 86


def hoeffding_z_sf(z):
    """Upper-tail probability of the scaled Hoeffding statistic.

    ``P(Z > z)`` where ``Z`` is the scaled Hoeffding statistic used by the canonical
    R implementation ``Hmisc::hoeffd`` (Hollander & Wolfe 1973 table for the
    asymptotically equivalent Blum-Kiefer-Rosenblatt statistic).  This is the
    n-independent reference distribution adopted for T12; the scaling
    ``z = 0.5 pi^4 n (D + 1/(36n))`` is documented in :func:`mc_hoeffding_d`.

    Parameters
    ----------
    z : float or array_like
        Scaled statistic value(s).

    Returns
    -------
    numpy.ndarray or float
        Survival probability (clipped to [1e-8, 1]).
    """
    z = np.asarray(z, dtype=float)
    scalar = z.ndim == 0
    z = np.atleast_1d(z)
    # Inside the tabulated range the table is strictly decreasing on an equispaced
    # grid, so linear interpolation preserves monotonicity and the interpolant is a
    # valid survival function there.
    # THIS FUNCTION IS NOT A DISTRIBUTION FUNCTION AND MUST NOT BE USED AS ONE.
    # Without the flat left clamp below, its value at z -> -inf is 0.4703 rather than 1,
    # because the published extrapolation branch is discontinuous with the table; with the
    # clamp it is piecewise flat and still not a probability law on the line.  It is kept
    # for exactly two purposes: computing the decision-level critical value, and a software
    # diagnostic (``ks_tab``) that records how far an empirical CDF lies from the published
    # approximation as shipped.  Any KS distance meant as a distance to a probability
    # distribution must use hoeffding_z_cdf.
    #
    # The published table is an upper-tail p-value approximation tabulated only for
    # z >= 1.1.  Its own extrapolation branch is NOT continuous with the table (at
    # z = 1.1 it returns 0.4095 against a tabulated 0.5297), so the combined function
    # increases with z and is not a valid CDF.  It is therefore used here ONLY for the
    # decision level, which lies well inside the tabulated range, and never as a
    # full-distribution reference.  See hoeffding_z_cdf for the latter.
    sf = np.interp(z, _HOEFFD_Z_GRID, _HOEFFD_Z_TABLE)
    above = z > _HOEFFD_Z_GRID[-1]
    if np.any(above):
        sf = np.where(above, np.exp(0.3885037 - 1.164879 * z), sf)
    below = z < _HOEFFD_Z_GRID[0]
    if np.any(below):
        sf = np.where(below, _HOEFFD_Z_TABLE[0], sf)
    sf = np.clip(sf, 1e-8, 1.0)
    return float(sf[0]) if scalar else sf


def hoeffding_z_cdf(x):
    """Full-distribution reference CDF for T12, from an independent realisation of the law.

    The scaled Hoeffding statistic is distribution-free, so its limit law can be
    estimated once, at large ``n``, independently of every study run.  This function
    interpolates the monotone quantile grid of that realisation (stored in
    ``rev_results_t12_law.json``, ``n_law = 2000``, ``N = 200,000``, seed independent of
    the studies) and is the reference used for the KS distance of T12.  Unlike the
    published p-value table it is monotone, normalized and defined on the whole real
    line, and its own Monte Carlo error is reported in the manuscript.

    Args:
        x: value(s) of the scaled statistic.

    Returns:
        numpy.ndarray or float: the CDF value(s).
    """
    x = np.asarray(x, dtype=float)
    scalar = x.ndim == 0
    x = np.atleast_1d(x)
    if _HOEFFD_LAW is None:
        raise RuntimeError("hoeffding_z_cdf requires rev_results_t12_law.json")
    p, q = _HOEFFD_LAW
    # p_grid is increasing in [0, 1] and q_grid increasing in z, so np.interp on
    # (q, p) yields a monotone CDF; values outside the grid are clamped.
    cdf = np.interp(x, q, p)
    cdf = np.clip(cdf, 0.0, 1.0)
    return float(cdf[0]) if scalar else cdf


@functools.lru_cache(maxsize=1)
def _hoeffding_ref_moments():
    """Mean and standard deviation of the T12 reference law.

    Taken from the same independently estimated limit law that supplies the KS reference
    (``rev_results_t12_law.json``).  The published tail table cannot be used for this:
    it is an upper-tail approximation defined only above its first tabulated value, so
    inverting it over the whole unit interval is not possible and its "moments" would not
    exist.  Earlier versions of this function did exactly that and failed as soon as the
    survival function was restricted to its valid range.

    With the quantile grid ``Q(p)`` on ``p in [0, 1]``, ``E[Q] = int Q(p) dp`` and
    ``E[Q^2] = int Q(x)^2 dx = int 2 Q(p) q(p) dp``, the second form avoiding a division
    by a density that is zero in the flat left tail.

    Returns:
        tuple of float: ``(mean, sd)`` of the reference distribution.

    Raises:
        RuntimeError: if the limit-law grid is not on disk.
    """
    if _HOEFFD_LAW is None:
        raise RuntimeError(
            "_hoeffding_ref_moments needs rev_results_t12_law.json; run "
            "code/gen_t12_law.py to produce it")
    p_grid, q_grid = _HOEFFD_LAW
    mean = float(np.trapezoid(q_grid, p_grid))
    m2 = float(np.trapezoid(2.0 * q_grid * q_grid, p_grid))
    return mean, float(np.sqrt(max(m2 - mean * mean, 0.0)))


def hoeffding_z_ppf(alpha):
    """Upper quantile of the published scaled-Hoeffding tail table.

    Returns the smallest ``z`` with ``sf(z) <= alpha``.  The published table is an
    upper-tail approximation tabulated, after its own left extension, only up to a
    survival probability of 0.5297 at ``z = 1.1``; for ``alpha`` above that value no such
    ``z`` exists and the quantile is undefined.  Callers that need the full distribution
    (moments, KS distances, confidence sets) must use :func:`hoeffding_z_cdf`, which is
    built from the independently estimated limit law.

    Args:
        alpha: upper-tail probability, must be below the table's maximum.

    Returns:
        float: the quantile.

    Raises:
        ValueError: if ``alpha`` exceeds the largest survival probability the published
            table defines, in which case no quantile exists.
    """
    alpha = float(alpha)
    lo, hi = -60.0, 60.0
    if alpha >= float(_HOEFFD_Z_TABLE[0]):
        raise ValueError(
            f"hoeffding_z_ppf({alpha}) is undefined: the published tail table reaches "
            f"only {_HOEFFD_Z_TABLE[0]:.4f} at z = {_HOEFFD_Z_GRID[0]}; use "
            "hoeffding_z_cdf for the full distribution")
    if hoeffding_z_sf(hi) > alpha:
        return float(hi)
    return float(brentq(lambda z: hoeffding_z_sf(z) - alpha, lo, hi, xtol=1e-10))


def _load_hoeffding_law():
    """Load the independent limit-law quantile grid for T12, if it is on disk."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    for base in (here, root, os.path.join(root, "results")):
        path = os.path.join(base, "rev_results_t12_law.json")
        if os.path.exists(path):
            break
    else:
        return None
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    return np.asarray(d["p_grid"], dtype=float), np.asarray(d["z_grid"], dtype=float)


_HOEFFD_LAW = _load_hoeffding_law()

REFERENCE_DISTS = {
    # name -> (cdf, ppf, n_independent, description); ppf takes an UPPER-tail probability
    "cvm_limit": (cvm_limit_cdf, cvm_limit_ppf, True,
                  "Cramer-von Mises limit law (Brownian-bridge L2 norm)"),
    "hoeffding_law": (hoeffding_z_cdf, None, True,
                      "Independently estimated limit law of the scaled Hoeffding "
                      "statistic; full-distribution reference for T12"),
    "hoeffding_z": (lambda z: 1.0 - hoeffding_z_sf(z), hoeffding_z_ppf, True,
                    "Hollander-Wolfe / Hmisc p-value table for the scaled Hoeffding D; "
                    "decision-level reference only, not a valid CDF"),
}


def _selftest(quick=True, seed=SEED):
    """Guard rails for the bespoke reference distributions.

    Runs at the start of every study script so that a convention slip (for example
    passing a lower-tail quantile where an upper-tail quantile is expected) can never
    silently invalidate a results file.  Returns a dict of checks; raises AssertionError
    on failure.

    Parameters
    ----------
    quick : bool
        If True, only the analytic checks are run (fast, no randomness).  If False, also
        run an MC null check of the T11 and T12 rejection rates.
    seed : int
        Seed for the optional MC checks.

    Returns
    -------
    dict
        ``{check_name: {'value': ..., 'target': ..., 'ok': bool}}``.
    """
    checks = {}

    def chk(name, value, target, tol):
        ok = bool(np.isfinite(value)) and abs(value - target) <= tol
        checks[name] = {"value": float(value), "target": float(target),
                        "tol": float(tol), "ok": ok}
        assert ok, f"core self-test failed: {name}: {value} vs {target} (tol {tol})"

    # --- T11: the documented convention is UPPER-tail --------------------------
    chk("cvm_ppf(0.05) == KPSS 5% critical value 0.46136",
        cvm_limit_ppf(0.05), 0.46136, 1e-3)
    chk("cvm_ppf(0.10) == KPSS 10% critical value 0.34730",
        cvm_limit_ppf(0.10), 0.34730, 2e-3)
    chk("cvm_cdf(cvm_ppf(0.05)) == 0.95",
        cvm_limit_cdf(cvm_limit_ppf(0.05)), 0.95, 1e-6)
    chk("cvm_quantile(0.95) == cvm_ppf(0.05)",
        cvm_limit_quantile(0.95), cvm_limit_ppf(0.05), 1e-9)
    # published CvM-limit CDF values (goftest::pCvM cross-check, see rev_study1_ext)
    for x, f in [(0.05, 0.12371907), (0.1, 0.41512656), (0.2, 0.73252957),
                 (0.5, 0.96016678), (1.0, 0.99753955)]:
        chk(f"cvm_cdf({x}) matches goftest", float(cvm_limit_cdf(x)), f, 2e-7)
    # --- T12: upper-tail as well ----------------------------------------------
    chk("hoeffding_z_sf is decreasing in z",
        float(hoeffding_z_sf(3.0) < hoeffding_z_sf(2.0)), 1.0, 0.0)
    chk("hoeffding_z_ppf(0.05) inverts sf",
        float(hoeffding_z_sf(hoeffding_z_ppf(0.05))), 0.05, 1e-9)

    # --- consistency of the registry with the evaluation path ------------------
    for tid in ("T11_kpss", "T12_hoeffding_d", "T13_interaction"):
        crit, spec = critical_value(tid, 0.05)
        checks[f"{tid}: critical value at alpha=0.05"] = {
            "value": float(crit), "spec": spec, "ok": True}
    # T12 is the one test whose decision reference and full-distribution reference differ.
    # The decision reference is the published n-free tail table; the KS reference is the
    # independently simulated quantile grid.  Both must exist and must agree at the
    # decision level, and the simulated grid must be a valid distribution function.
    assert TESTS["T12_hoeffding_d"].get("crit_null") == "hoeffding_z", \
        "T12 must decide from the published tail table"
    assert TESTS["T12_hoeffding_d"]["null"] == "hoeffding_law", \
        "T12 must measure KS against the simulated reference"
    assert _HOEFFD_LAW is not None, \
        "T12 KS reference needs rev_results_t12_law.json (run code/gen_t12_law.py)"
    # Independent check: the two references must agree at the decision level.  This
    # compares the published TABLE against the SIMULATED grid, not a function with itself.
    _tab_crit = hoeffding_z_ppf(0.05)
    _law_crit = float(np.interp(1.0 - 0.05, _HOEFFD_LAW[0], _HOEFFD_LAW[1]))
    chk("T12 decision level: published table agrees with the simulated reference",
        _law_crit / _tab_crit, 1.0, 0.012)
    # And the limit-law CDF must be a valid distribution function: monotone on [0, 1].
    _z = np.linspace(0.0, 9.0, 2001)
    _c = hoeffding_z_cdf(_z)
    chk("hoeffding_z_cdf is monotone and normalized",
        float(np.all(np.diff(_c) >= -1e-12) and _c.min() >= -1e-12 and _c.max() <= 1 + 1e-12), 1.0, 0.0)

    # --- D_KS convention must match scipy.stats.kstest (archive convention) ----
    rng = np.random.default_rng(11)
    for n_ks, dname, dargs in ((777, "t", (9,)), (1500, "chi2", (3,)), (400, "norm", ())):
        sample = getattr(stats, dname).rvs(size=n_ks, random_state=rng, *dargs) if dargs \
            else getattr(stats, dname).rvs(size=n_ks, random_state=rng)
        ks_mine = ks_distance(sample, dname, dargs)
        ks_scipy = float(stats.kstest(sample, dname, args=dargs).statistic)
        chk(f"ks_distance == scipy.kstest ({dname}, N={n_ks})", ks_mine, ks_scipy, 1e-12)

    # --- rank-invariance / canonical-algorithm checks for T12 ------------------
    d_perfect = hoeffding_d(np.arange(1.0, 21.0), np.arange(1.0, 21.0))
    chk("Hoeffding D = 1/30 for a perfect monotone relation (Hmisc D = 1.0)",
        30.0 * d_perfect, 1.0, 1e-12)

    if not quick:
        np.random.seed(seed)
        sv = mc_kpss(100, 20_000)
        r = reject_rate(sv, "T11_kpss", 0.05)
        checks["T11 alpha_hat at n=100, N=20000 (iid normal)"] = {
            "value": float(r), "target": 0.05, "tol": 0.01, "ok": bool(abs(r - 0.05) < 0.01)}
        assert abs(r - 0.05) < 0.01, f"T11 MC self-test failed: alpha_hat={r}"
        z = mc_hoeffding_d(200, 5_000)
        r2 = reject_rate(z, "T12_hoeffding_d", 0.05)
        checks["T12 alpha_hat at n=200, N=5000 (independence)"] = {
            "value": float(r2), "target": 0.05, "tol": 0.02, "ok": bool(abs(r2 - 0.05) < 0.02)}
        assert abs(r2 - 0.05) < 0.02, f"T12 MC self-test failed: alpha_hat={r2}"
    return checks


def reference_cdf(x, name, args=()):
    """CDF of reference ``name`` at ``x`` (scipy name or a key of REFERENCE_DISTS)."""
    if name in REFERENCE_DISTS:
        return REFERENCE_DISTS[name][0](x)
    return getattr(stats, name).cdf(x, *args)


def reference_sf(x, name, args=()):
    """Survival function of reference ``name`` at ``x``."""
    if name in REFERENCE_DISTS:
        if name == "hoeffding_z":
            return hoeffding_z_sf(x)
        return 1.0 - REFERENCE_DISTS[name][0](x)
    return getattr(stats, name).sf(x, *args)


def reference_ppf(q, name, args=()):
    """Quantile function (q = upper-tail probability) of reference ``name``."""
    if name in REFERENCE_DISTS:
        return REFERENCE_DISTS[name][1](q)
    return getattr(stats, name).isf(q, *args)


# ---------------------------------------------------------------------------
# 3. Evaluation: one definition of D_KS, alpha_hat and the rejection rule
# ---------------------------------------------------------------------------

def ks_distance(stat_values, null, args=()):
    """Kolmogorov-Smirnov distance between the empirical and reference CDFs.

    Parameters
    ----------
    stat_values : array_like
        Monte Carlo sample of the test statistic (finite values only are used).
    null : str
        scipy distribution name (``'t'``, ``'f'``, ``'chi2'``, ``'norm'``) or a
        custom key of :data:`REFERENCE_DISTS` (``'cvm_limit'``, ``'hoeffding_z'``).
    args : tuple
        Reference parameters.  For the classical panel these may depend on ``n``
        (exact finite-sample null); for T11--T13 they are empty by construction.

    Returns
    -------
    float
        ``sup_x |F_hat_N(x) - F_0(x)|``.
    """
    sv = np.asarray(stat_values, dtype=float)
    sv = np.sort(sv[np.isfinite(sv)])
    if sv.size == 0:
        return float("nan")
    n = sv.size
    if null == "binom":
        n_binom, p_binom = args
        xs = np.arange(0, int(n_binom) + 1)
        ecdf = np.searchsorted(sv, xs, side="right") / n
        return float(np.max(np.abs(ecdf - stats.binom.cdf(xs, n_binom, p_binom))))
    # sup_x |F_hat_N(x) - F_0(x)| for a step function F_hat_N: the supremum is attained
    # either just after a jump, F_hat = i/n, or just before the next jump, F_hat = (i-1)/n
    # (this is exactly the scipy.stats.kstest convention, so D_KS matches the archive).
    cdf = reference_cdf(sv, null, args)
    i = np.arange(1, n + 1) / n
    d_plus = np.max(i - cdf)
    d_minus = np.max(cdf - (i - 1.0 / n))
    return float(max(d_plus, d_minus))


def critical_value(test_id, alpha=0.05, ref_args=None):
    """Theoretical critical value(s) of the test's reference distribution at ``alpha``.

    Returns ``(crit, spec)`` where ``spec`` is the tail specification actually used
    (``'two_sided'``, ``'upper_tail'``, ``'lower_tail'`` or ``'binom'``).  For
    ``'binom'`` the value returned is the tuple ``(lower, upper)`` of the two-sided
    rule used by the archive (``binom.ppf(alpha/2)`` / ``binom.ppf(1 - alpha/2)``).
    The *same* nominal alpha is used for the critical value and for the reference
    CDF in :func:`_evaluate`.
    """
    t = TESTS[test_id]
    # ``crit_null`` (when present) names the law used for the DECISION level, which may
    # differ from the full-distribution reference: for T12 the published p-value table
    # is the appropriate critical value, while the KS distance uses the independently
    # estimated limit law because the table is not a valid CDF.
    null = t.get("crit_null", t["null"])
    args = ref_args if ref_args is not None else t.get("crit_null_args", t.get("null_args", ()))
    spec = t["type1"]
    if spec == "binom":
        n_binom, p_binom = args
        return (stats.binom.ppf(alpha / 2.0, n_binom, p_binom),
                stats.binom.ppf(1.0 - alpha / 2.0, n_binom, p_binom)), spec
    if spec == "two_sided":
        return reference_ppf(alpha / 2.0, null, args), spec
    if spec == "upper_tail":
        return reference_ppf(alpha, null, args), spec
    if spec == "lower_tail":
        return reference_ppf(1.0 - alpha, null, args), spec
    raise ValueError(f"unknown type1 spec {spec!r} for {test_id}")


def reject_rate(stat_values, test_id, alpha=0.05, ref_args=None):
    """Empirical rejection rate at nominal level ``alpha`` using the theoretical rule.

    Parameters
    ----------
    stat_values : array_like
        Monte Carlo sample of the statistic.
    test_id : str
        Key of :data:`TESTS`.
    alpha : float
        Nominal level; the identical value is used for the critical value and for the
        reference CDF.
    ref_args : tuple, optional
        Reference parameters (defaults to the registry entry).

    Returns
    -------
    float
        ``(1/N) sum_j 1{stat_j in rejection region}``.
    """
    sv = np.asarray(stat_values, dtype=float)
    sv = sv[np.isfinite(sv)]
    if sv.size == 0:
        return float("nan")
    crit, spec = critical_value(test_id, alpha, ref_args)
    if spec == "binom":
        lower, upper = crit
        return float(np.mean((sv < lower) | (sv > upper)))
    if spec == "two_sided":
        return float(np.mean(np.abs(sv) > crit))
    if spec == "upper_tail":
        return float(np.mean(sv > crit))
    if spec == "lower_tail":
        return float(np.mean(sv < crit))
    raise ValueError(spec)


def compute_type1_error(stat_values, dist_name, df_args, alpha=0.05, two_sided=False):
    """Superset of :func:`experiments.compute_type1_error`.

    The archive signature and semantics are preserved exactly (so existing callers keep
    working); the only extensions are (i) a non-default ``alpha`` and (ii) the custom
    reference families of :data:`REFERENCE_DISTS`.  For the archive combinations
    (``'t'``/``'f'``/``'chi2'``/``'binom'``/``'norm'``) the result equals the archive
    function to machine precision.
    """
    if dist_name == "binom":
        n_binom, p_binom = df_args
        lower = stats.binom.ppf(alpha / 2.0, n_binom, p_binom)
        upper = stats.binom.ppf(1.0 - alpha / 2.0, n_binom, p_binom)
        return float(np.mean((stat_values < lower) | (stat_values > upper)))
    if dist_name in REFERENCE_DISTS:
        crit = reference_ppf(alpha, dist_name)
        return float(np.mean(np.asarray(stat_values) > crit))
    return _archive_type1_error(stat_values, dist_name, df_args,
                                alpha=alpha, two_sided=two_sided)


def _evaluate(stat_values, test_id, alpha=0.05, n=None, ref_args=None):
    """Evaluate one Monte Carlo sample against the theoretical null of ``test_id``.

    Parameters
    ----------
    stat_values : array_like
        Sample of the test statistic under the configuration being studied.
    test_id : str
        Key of :data:`TESTS`.
    alpha : float
        Nominal level for the Type I error rate (identical value used for the
        critical value and the reference CDF).
    n : int, optional
        Per-group sample size; needed only for tests whose *theoretical* null is the
        exact finite-sample law (T1, T2, T7, T10).
    ref_args : tuple, optional
        Explicit reference parameters, overriding the registry lookup.

    Returns
    -------
    dict
        ``{'ks': float, 'type1': float, 'ci_lo': float, 'ci_hi': float,
        'n_effective': int, 'critical_value': float or tuple, 'alpha': float}``.
    """
    t = TESTS[test_id]
    if ref_args is None:
        ref_args = t["ref_args_fn"](n) if t.get("ref_args_fn") is not None else t.get("null_args", ())
    sv = np.asarray(stat_values, dtype=float)
    sv = sv[np.isfinite(sv)]
    ks = ks_distance(sv, t["null"], ref_args)
    type1 = reject_rate(sv, test_id, alpha, ref_args)
    ci_lo, ci_hi = wilson_ci(type1, sv.size)
    crit, _ = critical_value(test_id, alpha, ref_args)
    return {"ks": float(ks), "type1": float(type1),
            "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
            "n_effective": int(sv.size), "critical_value": crit, "alpha": float(alpha)}


def evaluate_multi_alpha(stat_values, test_id, alphas=ALPHAS, n=None, ref_args=None):
    """Evaluate the same Monte Carlo sample at several nominal levels."""
    return {f"{a:.2f}": _evaluate(stat_values, test_id, a, n=n, ref_args=ref_args)
            for a in alphas}


def n_star_alpha(type1_by_n, ns=None, alpha=0.05, tol=0.01):
    """Smallest ``n`` with ``|alpha_hat - alpha| < tol`` (archive operational rule)."""
    if ns is None:
        ns = sorted(type1_by_n)
    for nn in ns:
        v = type1_by_n.get(nn, type1_by_n.get(str(nn)))
        if v is not None and np.isfinite(v) and abs(v - alpha) < tol:
            return nn
    return ">%s" % max(ns) if ns else None


def n_star_ks(ks_by_n, ns=None, threshold=KS_THRESHOLD):
    """Smallest ``n`` with ``D_KS < 1.358/sqrt(N)`` (archive operational rule)."""
    if ns is None:
        ns = sorted(ks_by_n)
    for nn in ns:
        v = ks_by_n.get(nn, ks_by_n.get(str(nn)))
        if v is not None and np.isfinite(v) and v < threshold:
            return nn
    return ">%s" % max(ns) if ns else None


# ---------------------------------------------------------------------------
# 4. New test statistics (T11--T13)
# ---------------------------------------------------------------------------

def kpss_statistic(y, lags=None):
    """KPSS (1992) level-stationarity statistic for one series.

    Implementation (Kwiatkowski, Phillips, Schmidt & Shin 1992,
    doi:10.1016/0304-4076(92)90104-Y, eq. 10-12):

    1. Residuals from the level regression: ``e_t = y_t - mean(y)``.
    2. Cumulative residuals ``S_t = sum_{i<=t} e_i``.
    3. Long-run variance with the Bartlett (Newey-West) kernel

           s2(l) = gamma_0 + 2 * sum_{j=1..l} (1 - j/(l+1)) * gamma_j,
           gamma_j = (1/n) * sum_{t=j+1..n} e_t e_{t-j},

    4. ``LM = (1/n^2) * sum_t S_t^2 / s2(l)``.

    Bandwidth rule: ``l = floor(4 * (n/100)^(1/4))`` (KPSS 1992, p. 164; the
    Schwert rule), i.e. l = 1, 1, 2, 2, 3, 3, 4, 5 for n = 5, 10, 20, 30, 50,
    100, 200, 500.  The estimator uses the same lags as the HAC variance, which is
    the convention of the original paper; ``lags`` can be overridden.

    Parameters
    ----------
    y : array_like
        One time series of length n (mean-zero under the level-stationary null).
    lags : int, optional
        Bartlett lag truncation; default the Schwert rule above.

    Returns
    -------
    float
        The LM statistic (0 if the long-run variance estimate is non-positive).
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 4:
        return np.nan
    e = y - y.mean()
    S = np.cumsum(e)
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** 0.25))
    lags = max(0, min(int(lags), n - 2))
    gamma0 = float(np.dot(e, e) / n)
    s2 = gamma0
    for j in range(1, lags + 1):
        w = 1.0 - j / (lags + 1.0)
        s2 += 2.0 * w * float(np.dot(e[j:], e[:-j]) / n)
    if not np.isfinite(s2) or s2 <= 0:
        return 0.0
    return float(np.sum(S ** 2) / (n ** 2 * s2))


def mc_kpss(n, N, kind="iid", rho=0.0, delta=1.0, rng=None, lags=None, **kwargs):
    """Study-1/Study-2 generator for T11 (KPSS level stationarity).

    Null configuration (``kind='iid'``): ``y_t ~ N(0, 1)`` i.i.d., which is level
    stationary, so nominal rejection rates are Type I error rates.  The theoretical
    reference is the n-independent Cramer-von Mises limit law (see
    :func:`cvm_limit_cdf`); **no standardization of the statistic is applied**, because
    the LM statistic already converges to that law without rescaling.

    Scenario variants (violations of the null *or* of the i.i.d. assumption):

    * ``'ar1'``   : stationary AR(1), ``y_t = rho y_{t-1} + sqrt(1-rho^2) e_t`` with a
      stationary start ``y_0 ~ N(0, 1)``, so the marginal variance is 1 and H0 (level
      stationarity) remains TRUE while the series is serially dependent.  The scenario
      therefore tests whether the long-run-variance correction restores the limit law
      under dependence the null permits.
    * ``'expon'`` / ``'tdist'`` : skewed / heavy-tailed i.i.d. innovations.
    * ``'trend'`` : ``y_t = delta * t/n + e_t`` -> trend stationarity, so level
      stationarity is false (a genuine power scenario, not a size scenario).
    * ``'rw'``    : ``y_t = cumsum(e_t)`` (unit root) -> H0 false (power scenario).

    Parameters
    ----------
    n : int
        Series length.
    N : int
        Number of Monte Carlo replications.
    kind : str
        One of ``'iid'``, ``'ar1'``, ``'expon'``, ``'tdist'``, ``'trend'``, ``'rw'``.
    rho : float
        AR(1) coefficient for ``kind='ar1'``.
    delta : float
        Trend size (in innovation standard deviations) for ``kind='trend'``.
    rng : numpy.random.Generator, optional
        Random source; defaults to the legacy global NumPy stream so that the archive
        seed convention (``np.random.seed(2024)``) still applies.
    lags : int, optional
        Bartlett lag truncation (default: Schwert rule).

    Returns
    -------
    numpy.ndarray, shape (N,)
        KPSS LM statistics.
    """
    randn = rng.standard_normal if rng is not None else np.random.standard_normal
    if kind == "iid":
        Y = randn((N, n))
    elif kind == "ar1":
        Y = np.empty((N, n))
        # Stationary initial value: with innovations scaled by sqrt(1 - rho^2), the
        # stationary marginal variance is 1, so y_0 ~ N(0, 1).  The earlier version
        # divided by sqrt(1 - rho^2) here, giving the series an initial variance of
        # 1/(1 - rho^2) and a non-stationary start-up transient.
        Y[:, 0] = randn(N)
        for t in range(1, n):
            Y[:, t] = rho * Y[:, t - 1] + np.sqrt(1.0 - rho ** 2) * randn(N)
    elif kind == "expon":
        Y = (rng.exponential(1.0, (N, n)) if rng is not None
             else np.random.exponential(1.0, (N, n))) - 1.0
    elif kind == "tdist":
        Y = (rng.standard_t(5, (N, n)) if rng is not None
             else np.random.standard_t(5, (N, n)))
    elif kind == "trend":
        e = randn((N, n))
        Y = delta * (np.arange(n) / n)[None, :] + e
    elif kind == "rw":
        Y = np.cumsum(randn((N, n)), axis=1)
    else:
        raise ValueError(f"unknown KPSS kind {kind!r}")
    return np.array([kpss_statistic(Y[i], lags=lags) for i in range(N)])


def _joint_rank(x, y):
    """Hmisc mid-rank + bivariate (joint) rank, vectorised.

    Returns ``(rx, ry, rj)`` where ``rx``/``ry`` are mid-ranks (1-based) and
    ``rj_i = 1 + sum_{j != i} cx_ij cy_ij`` with ``cx_ij = 1{x_j < x_i}`` (0.5 on ties)
    and ``cy_ij`` likewise -- exactly the ``jrank`` routine of ``Hmisc::hoeffd``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    cx = (x[None, :] < x[:, None]).astype(float) + 0.5 * (x[None, :] == x[:, None])
    cy = (y[None, :] < y[:, None]).astype(float) + 0.5 * (y[None, :] == y[:, None])
    rj = 1.0 + (cx * cy).sum(axis=1) - 0.25  # the j == i term equals 0.5 * 0.5
    rx = stats.rankdata(x, method="average")
    ry = stats.rankdata(y, method="average")
    return rx, ry, rj


def hoeffding_d(x, y):
    """Hoeffding's (1948) D statistic, exactly as in ``Hmisc::hoeffd`` (times 1/30).

    The canonical Fortran/Ratfor implementation (Hmisc 5.3-0, ``src/ratfor/hoeffd.r``)
    computes

        d = [q - 2(n-2) r + (n-2)(n-3) s] / [n(n-1)(n-2)(n-3)(n-4)],

    with ``q = sum (rx-1)(rx-2)(ry-1)(ry-2)``, ``r = sum (rx-2)(ry-2)(rj-1)`` and
    ``s = sum (rj-1)(rj-2)``, where ``rx``/``ry`` are mid-ranks and ``rj`` the bivariate
    ("joint") rank.  ``Hmisc::hoeffd`` reports ``30 d`` (range -0.5..1.0 without ties);
    this function returns ``d``.  The sample-size independent reference used here is the
    scaled statistic ``z = 0.5 pi^4 (n d + 1/36)`` (see :func:`mc_hoeffding_d`).

    Parameters
    ----------
    x, y : array_like
        Equal-length samples.

    Returns
    -------
    float
        ``d`` (T12's raw statistic).  Requires ``n >= 5``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 5:
        return np.nan
    rx, ry, rj = _joint_rank(x, y)
    q = np.sum((rx - 1.0) * (rx - 2.0) * (ry - 1.0) * (ry - 2.0))
    r = np.sum((rx - 2.0) * (ry - 2.0) * (rj - 1.0))
    s = np.sum((rj - 1.0) * (rj - 2.0))
    return float((q - 2.0 * (n - 2.0) * r + (n - 2.0) * (n - 3.0) * s)
                 / (n * (n - 1.0) * (n - 2.0) * (n - 3.0) * (n - 4.0)))


def hoeffding_z(d, n):
    """Scale Hoeffding's ``d`` to the n-free reference scale used by Hmisc/Wolfe.

    ``z = 0.5 * pi^4 * (n d + 1/36)``, i.e. exactly Hmisc's
    ``z = 0.5 pi^4 n (d + 1/(36 n))`` applied to the *raw* statistic ``d`` returned by
    :func:`hoeffding_d` (note: ``Hmisc::hoeffd`` reports ``30 d``, so its printed ``D``
    must be divided by 30 before applying this scaling).  The ``+1/(36n)`` centering
    correction is part of the published Hmisc/Hollander-Wolfe scaling, not an ad-hoc
    adjustment; the resulting ``z`` has a fixed asymptotic law
    (:func:`hoeffding_z_sf`) for every ``n``.

    Parameters
    ----------
    d : float or array_like
        Raw Hoeffding statistic returned by :func:`hoeffding_d`.
    n : int
        Sample size used to compute ``d``.

    Returns
    -------
    float or numpy.ndarray
        Scaled statistic ``z``.
    """
    d = np.asarray(d, dtype=float)
    return 0.5 * np.pi ** 4 * (n * d + 1.0 / 36.0)


def mc_hoeffding_d(n, N, dep="indep", rho=0.5, rng=None, **kwargs):
    """Study-1/Study-2 generator for T12 (Hoeffding's D independence test).

    Null configuration (``dep='indep'``): ``X ~ N(0,1)`` and ``Y ~ N(0,1)`` independent,
    so the nominal rejection rate is a Type I error rate.  The statistic is *raw*
    Hoeffding ``d`` (:func:`hoeffding_d`); the standardization

        z = 0.5 * pi^4 * (n d + 1/36)

    (Hmisc/Hollander-Wolfe scaling, see :func:`hoeffding_z`) makes the reference
    distribution sample-size independent.  The rejection rule is upper-tail at level
    ``alpha`` using the published asymptotic law of ``z`` (:func:`hoeffding_z_sf`).

    Scenario variants (violations of independence; H0 false -> power scenarios):

    * ``'linear'``  : ``(X, Y)`` bivariate normal with correlation ``rho``.
    * ``'quad'``    : ``Y = X^2 - 1 + noise`` (symmetric, zero *linear* correlation but
      strong dependence) -- the case Hoeffding's D is designed to detect.
    * ``'expon'``   : independent but heavily skewed marginals (``X, Y ~ Exp(1)-1``):
      a *size* scenario that checks whether the rank-based null survives non-normality.

    Parameters
    ----------
    n : int
        Sample size (both variables).
    N : int
        Number of Monte Carlo replications.
    dep : str
        One of ``'indep'``, ``'linear'``, ``'quad'``, ``'expon'``.
    rho : float
        Correlation for ``dep='linear'``.
    rng : numpy.random.Generator, optional
        Random source (defaults to the legacy global stream).

    Returns
    -------
    numpy.ndarray, shape (N,)
        Scaled statistics ``z``.
    """
    randn = rng.standard_normal if rng is not None else np.random.standard_normal
    if dep == "indep":
        X, Y = randn((N, n)), randn((N, n))
    elif dep == "linear":
        Z = randn((N, n))
        X = randn((N, n))
        Y = rho * X + np.sqrt(1.0 - rho ** 2) * Z
    elif dep == "quad":
        X = randn((N, n))
        Y = X ** 2 - 1.0 + 0.5 * randn((N, n))
    elif dep == "expon":
        expo = rng.exponential if rng is not None else np.random.exponential
        X, Y = expo(1.0, (N, n)) - 1.0, expo(1.0, (N, n)) - 1.0
    else:
        raise ValueError(f"unknown Hoeffding dep {dep!r}")
    out = np.empty(N)
    for i in range(N):
        out[i] = hoeffding_z(hoeffding_d(X[i], Y[i]), n)
    return out


# --- T13: interaction test (Welchowski & Edelmann 2024, doi:10.3390/make6020061) ---

LGBM_PARAMS = dict(
    n_estimators=100,      # boosting rounds
    num_leaves=7,          # small trees: depth-3-equivalent capacity
    learning_rate=0.1,
    min_child_samples=20,
    subsample=1.0,
    colsample_bytree=1.0,
    reg_lambda=0.0,
    verbose=-1,
    n_jobs=1,              # parallelism is across MC replications, not inside LightGBM
)
N_TRAIN_T13 = 1000         # fixed training-set size (documented; see mc_interaction_h)
PD_GRID_M_T13 = 50         # PD grid resolution per covariate (diagnostic method only)
PD_M_T13 = 100             # reference points for the unbiased MC partial-dependence estimator


def _interaction_core(Xtr, ytr, Xte, params, grid_m, seed, pd_method="mc", pd_m=PD_M_T13):
    """One replication of the interaction test: fit LightGBM, build PD functions, return z.

    Returns ``(z_iad, h2_fp)`` where

    * ``z_iad`` is the Pitman statistic of Welchowski & Edelmann (2024, eq. 24) applied to
      the interaction difference ``IAD = Cov(z1, z2)`` with
      ``z1 = f_hat + g_hat``, ``z2 = f_hat - g_hat`` and
      ``g_hat(x) = PD(x1) + PD(x2)`` (the model restricted to the no-interaction null;
      the empty-set PD is a constant and cancels in the centering);
    * ``h2_fp`` is the Friedman & Popescu (2008) H^2 statistic
      ``sum_i (f_i - g_i)^2 / sum_i f_i^2`` on centered predictions, reported as a
      secondary diagnostic because H^2 has no model-agnostic asymptotic null.

    Partial dependence is estimated with ``pd_method``:

    * ``'mc'`` (default): unbiased Monte-Carlo average of ``f_hat`` over ``pd_m``
      reference rows drawn without replacement from the training covariates, i.e.
      ``PD_1(x) = (1/pd_m) sum_k f_hat(x, x_k2)``.  Cost ``n_test * pd_m`` predictions.
      The PD noise is independent of the response and therefore makes the statistic
      slightly *conservative* (it inflates ``Var(g_hat)``), never liberal.
    * ``'grid'``: deterministic ``grid_m x grid_m`` grid with linear interpolation
      (cheap, but the smoothing bias deflates ``Var(g_hat)`` and inflates the statistic;
      kept only for the estimator-comparison diagnostic).
    * ``'exact'``: full sample average over all ``n_train`` training rows
      (cost ``n_test * n_train`` predictions; used as the unbiased reference in the
      diagnostic at small ``N``).
    """
    import lightgbm as lgb

    model = lgb.LGBMRegressor(random_state=seed, **params).fit(Xtr, ytr)
    f = model.predict(Xte)
    if pd_method == "grid":
        g = np.linspace(float(Xtr[:, 0].min()), float(Xtr[:, 0].max()), grid_m)
        A, B = np.meshgrid(g, g, indexing="ij")
        Z = np.column_stack([A.ravel(), B.ravel()])
        S = model.predict(Z).reshape(grid_m, grid_m)
        pd1 = np.interp(Xte[:, 0], g, S.mean(axis=1))
        pd2 = np.interp(Xte[:, 1], g, S.mean(axis=0))
    else:
        if pd_method == "exact":
            ref = Xtr
        elif pd_method == "mc":
            rng = np.random.default_rng([seed, 31337])
            m = int(min(pd_m, Xtr.shape[0]))
            ref = Xtr[rng.choice(Xtr.shape[0], size=m, replace=False)]
        else:
            raise ValueError(f"unknown pd_method {pd_method!r}")
        n_te, m_ref = Xte.shape[0], ref.shape[0]
        M1 = np.repeat(Xte[:, 0][:, None], m_ref, axis=1).ravel()
        M2 = np.tile(ref[:, 1], n_te)
        pd1 = model.predict(np.column_stack([M1, M2])).reshape(n_te, m_ref).mean(axis=1)
        M2b = np.repeat(Xte[:, 1][:, None], m_ref, axis=1).ravel()
        M1b = np.tile(ref[:, 0], n_te)
        pd2 = model.predict(np.column_stack([M1b, M2b])).reshape(n_te, m_ref).mean(axis=1)
    g_hat = pd1 + pd2
    z1 = f + g_hat
    z2 = f - g_hat
    z3 = (z1 - z1.mean()) * (z2 - z2.mean())
    m = z3.mean()
    var = float(np.mean((z3 - m) ** 2))
    n_te = Xte.shape[0]
    z_iad = np.sqrt(n_te) * m / np.sqrt(max(var, 1e-300)) if n_te > 3 else np.nan
    fc = f - f.mean()
    gc = g_hat - g_hat.mean()
    denom = float(np.sum(fc ** 2))
    h2 = float(np.sum((fc - gc) ** 2) / denom) if denom > 0 else np.nan
    return z_iad, h2


def _t13_chunk(args):
    """Worker for parallel T13 replication chunks (module level: picklable)."""
    (n_test, reps, seed, kind, params, grid_m, n_train, rho, pd_method, pd_m) = args
    out_z = np.empty(len(reps))
    out_h2 = np.empty(len(reps))
    for k, j in enumerate(reps):
        rng = np.random.default_rng([seed, int(j)])
        if kind == "additive":
            X = rng.random((n_train, 2))
            f_true = np.sin(np.pi * X[:, 0]) + 1.5 * X[:, 1]
            y = f_true + 0.5 * rng.standard_normal(n_train)
            Xte = rng.random((n_test, 2))
        elif kind == "interaction":
            X = rng.random((n_train, 2))
            f_true = np.sin(np.pi * X[:, 0]) + 1.5 * X[:, 1] + rho * (
                (X[:, 0] - 0.5) * (X[:, 1] - 0.5)) / 0.25
            y = f_true + 0.5 * rng.standard_normal(n_train)
            Xte = rng.random((n_test, 2))
        elif kind == "corrX":
            # correlated covariates (additive truth): size scenario for the learner
            # Cholesky factor L of [[1, rho], [rho, 1]] is [[1, 0], [rho, sqrt(1-rho^2)]],
            # so Z @ L.T has covariance L L^T and the *latent* correlation is exactly rho.
            # The earlier version multiplied by the transpose of the wrong matrix, giving
            # latent variances 1 + rho^2 and 1 - rho^2 and a correlation of rho/sqrt(1+rho^2).
            L = np.array([[1.0, 0.0], [rho, np.sqrt(max(1.0 - rho ** 2, 0.0))]])
            Z = rng.standard_normal((n_train, 2))
            X = stats.norm.cdf(Z @ L.T)
            y = np.sin(np.pi * X[:, 0]) + 1.5 * X[:, 1] + 0.5 * rng.standard_normal(n_train)
            Zt = rng.standard_normal((n_test, 2))
            Xte = stats.norm.cdf(Zt @ L.T)
        else:
            raise ValueError(kind)
        out_z[k], out_h2[k] = _interaction_core(X, y, Xte, params, grid_m,
                                                seed=seed + 7919 * int(j),
                                                pd_method=pd_method, pd_m=pd_m)
    return out_z, out_h2


def mc_interaction_h(n, N, kind="additive", rho=0.0, rng=None, n_jobs=1,
                     n_train=N_TRAIN_T13, params=None, grid_m=PD_GRID_M_T13,
                     return_h2=False, seed=None, pd_method="mc", pd_m=PD_M_T13,
                     **kwargs):
    """Generator for T13, the interaction-effect test of a prediction model.

    Statistic and reference (Welchowski & Edelmann 2024,
    doi:10.3390/make6020061, the paper cited by the referee, which supplies the
    model-agnostic asymptotic null missing from Friedman & Popescu 2008):

    * learner: LightGBM 4.6.0 gradient boosting with the documented fixed
      hyperparameters :data:`LGBM_PARAMS` (100 trees, 7 leaves, learning rate 0.1,
      min_child_samples 20).  No tuning per replication: the benchmark measures the
      *calibration* of a fixed, documented pipeline.
    * partial dependence: unbiased Monte-Carlo estimator over ``pd_m = 100`` reference
      rows (``pd_method='mc'``, default); ``'grid'`` and ``'exact'`` exist only for the
      estimator-comparison diagnostic in ``t13_split_diagnostic``.
    * training/test split: ``n_train = 1000`` fixed (``N_TRAIN_T13``) and test size
      ``n_test = n``; the asymptotic theory requires the training size to grow faster
      than the test size, so the ratio only improves as ``n`` grows.
    * ``z = sqrt(n) * mean(z3) / sqrt(mean((z3 - mean(z3))^2))`` evaluated on the test
      data, where ``z3 = (z1 - mean z1)(z2 - mean z2)``, ``z1 = f_hat + g_hat``,
      ``z2 = f_hat - g_hat`` and ``g_hat = PD(x1) + PD(x2)`` is the fitted model
      restricted to the no-interaction null.
    * reference distribution: standard normal, two-sided.  The scale is therefore
      sample-size independent by construction (the ``sqrt(n)`` and the self-normalizing
      denominator carry the scaling).

    Null configuration (``kind='additive'``): ``y = sin(pi x1) + 1.5 x2 + N(0, 0.5^2)``
    (nonlinear but additive, explained variance 0.5), ``x1, x2 ~ U(0,1)`` independent.
    Scenario variants: ``'interaction'`` (true multiplicative interaction, power
    scenario) and ``'corrX'`` (correlated covariates, size scenario).

    Reproducibility: every replication has its own deterministic seed derived from
    ``seed`` and the global replication index, so results are independent of ``n_jobs``
    and of the chunking.

    Parameters
    ----------
    n : int
        Test-set size (the "sample size" of the benchmark).
    N : int
        Number of Monte Carlo replications.
    kind : str
        ``'additive'`` (H0), ``'interaction'`` (H1), ``'corrX'`` (H0, correlated X).
    rho : float
        Interaction amplitude (``'interaction'``) or covariate correlation (``'corrX'``).
    n_jobs : int
        Worker processes (default 1).
    n_train : int
        Fixed training-set size.
    params : dict, optional
        Override of :data:`LGBM_PARAMS`.
    grid_m : int
        PD grid resolution.
    return_h2 : bool
        Also return the Friedman-Popescu H^2 diagnostic (same fits, no extra cost).
    seed : int, optional
        Base seed (default :data:`SEED`).
    rng : numpy.random.Generator, optional
        Accepted for interface compatibility; T13 uses the deterministic per-replication
        seeding described above instead of a shared stream.

    Returns
    -------
    numpy.ndarray, shape (N,)
        Test statistics ``z``; if ``return_h2=True``, the tuple ``(z, h2)``.
    """
    seed = SEED if seed is None else int(seed)
    params = dict(LGBM_PARAMS if params is None else params)
    n = int(n)
    reps = np.arange(N)
    if n_jobs and n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        chunks = np.array_split(reps, int(n_jobs))
        jobs = [(n, list(c), seed, kind, params, grid_m, int(n_train), rho, pd_method, pd_m)
                for c in chunks if len(c) > 0]
        with ProcessPoolExecutor(max_workers=int(n_jobs)) as ex:
            res = list(ex.map(_t13_chunk, jobs))
        z = np.concatenate([r[0] for r in res])
        h2 = np.concatenate([r[1] for r in res])
        order = np.concatenate([c for c in chunks if len(c) > 0])
        idx = np.argsort(order)
        z, h2 = z[idx], h2[idx]
    else:
        z, h2 = _t13_chunk((n, list(reps), seed, kind, params, grid_m, int(n_train), rho,
                            pd_method, pd_m))
    return (z, h2) if return_h2 else z


# ---------------------------------------------------------------------------
# 5. Dependence generators (Referee 1 comment 4)
# ---------------------------------------------------------------------------

def mc_t_test_ar1(n, N, rho=0.0, rng=None, **kwargs):
    """One-sample t statistic for stationary AR(1) data (mean 0).

    ``y_t = rho y_{t-1} + sqrt(1-rho^2) e_t``, so ``E[y_t] = 0`` and H0 of the
    one-sample t test is *true*: any rejection is a Type I error.  The statistic is the
    ordinary ``t = mean(y) / (sd(y)/sqrt(n))`` computed as in
    :func:`experiments.mc_t_test` (ddof=1, zero-variance guard).

    Returns
    -------
    numpy.ndarray, shape (N,)
        t statistics.
    """
    randn = rng.standard_normal if rng is not None else np.random.standard_normal
    Y = np.empty((N, n))
    Y[:, 0] = randn(N)
    for t in range(1, n):
        Y[:, t] = rho * Y[:, t - 1] + np.sqrt(max(1.0 - rho ** 2, 0.0)) * randn(N)
    xbar = Y.mean(axis=1)
    s = Y.std(axis=1, ddof=1)
    s[s == 0] = 1e-12
    return xbar / (s / np.sqrt(n))


def mc_sign_test_ar1(n, N, rho=0.0, rng=None, **kwargs):
    """Sign-test count ``N+`` for stationary AR(1) data with median 0.

    The marginals are symmetric about zero, so each sign is marginally a fair coin, but
    the signs are NOT independent under serial dependence: the autocorrelation depends on
    the time lag, the signs are therefore not exchangeable in the sense the binomial null
    requires, and equal signs cluster in runs.  ``N+`` is consequently over-dispersed
    relative to ``Bin(n, 1/2)`` and the test is badly liberal at large ``rho`` -- measured,
    not assumed (see the dependence results of the revision).

    Returns
    -------
    numpy.ndarray, shape (N,)
        Counts of positive observations.
    """
    randn = rng.standard_normal if rng is not None else np.random.standard_normal
    Y = np.empty((N, n))
    Y[:, 0] = randn(N)
    for t in range(1, n):
        Y[:, t] = rho * Y[:, t - 1] + np.sqrt(max(1.0 - rho ** 2, 0.0)) * randn(N)
    return (Y > 0).sum(axis=1)


def _make_clusters(k, c, m, N, icc, dist="normal", rng=None):
    """Clustered data: k groups x c clusters x m observations, ICC(1) = icc."""
    randn = rng.standard_normal if rng is not None else np.random.standard_normal
    sb = np.sqrt(max(icc, 0.0))
    se = np.sqrt(max(1.0 - icc, 0.0))
    b = sb * randn((N, k, c, 1))
    if dist == "normal":
        e = se * randn((N, k, c, m))
    elif dist == "expon":
        e = se * ((rng.exponential(1.0, (N, k, c, m)) if rng is not None
                   else np.random.exponential(1.0, (N, k, c, m))) - 1.0)
    else:
        e = se * randn((N, k, c, m))
    return b + e  # (N, k, c, m)


def mc_anova_clustered(k, c, m, N, icc=0.0, dist="normal", rng=None, **kwargs):
    """One-way ANOVA F statistic for group-clustered data (T2 dependence scenario).

    Design: ``k`` groups, each containing ``c`` clusters of ``m`` observations,
    ``Y_gji = b_gj + e_gji`` with ``b ~ N(0, icc)`` and ``e ~ N(0, 1-icc)``.  The group
    means are equal under H0; positive ICC inflates the between-group mean square while
    the within-group residual variance is unchanged, so the naive ``F`` distribution
    ``F(k-1, kcm-k)`` no longer holds.  Under H0 the rejection rate is a Type I error
    rate (the design effect is ``1 + (m-1) icc``).

    Returns
    -------
    numpy.ndarray, shape (N,)
        F statistics.
    """
    Y = _make_clusters(k, c, m, N, icc, dist, rng)
    gm = Y.mean(axis=(2, 3))
    grand = Y.mean(axis=(1, 2, 3))
    SSTR = c * m * ((gm - grand[:, None]) ** 2).sum(axis=1)
    MSTR = SSTR / (k - 1)
    SSE = ((Y - gm[:, :, None, None]) ** 2).sum(axis=(1, 2, 3))
    MSE = SSE / (k * (c * m - 1))
    MSE[MSE == 0] = 1e-15
    return MSTR / MSE


def mc_kruskal_wallis_clustered(k, c, m, N, icc=0.0, dist="normal", rng=None, **kwargs):
    """Kruskal-Wallis H for group-clustered data (T3 dependence scenario).

    Same design as :func:`mc_anova_clustered`; ranks are computed within each
    replication across all ``k c m`` observations (average ranks, as in the archive
    generator).
    """
    Y = _make_clusters(k, c, m, N, icc, dist, rng).reshape(N, k, c * m)
    out = np.empty(N)
    n_tot = k * c * m
    for i in range(N):
        ranks = stats.rankdata(Y[i].ravel())
        H = 0.0
        for j in range(k):
            Rj = ranks[j * c * m:(j + 1) * c * m].sum()
            H += Rj ** 2 / (c * m)
        out[i] = 12.0 / (n_tot * (n_tot + 1)) * H - 3.0 * (n_tot + 1)
    return out


def mc_cochran_q_corr(k, n, N, icc=0.0, rng=None, **kwargs):
    """Cochran's Q for block-correlated dichotomous data (T8 dependence scenario).

    ``k`` treatments, ``n`` blocks.  With ``icc = 0`` the responses are i.i.d.
    Bernoulli(1/2) (the archive configuration).  With ``icc > 0`` a symmetric
    beta-binomial block structure is used, ``p_i ~ Beta(a, a)`` with
    ``a = (1-icc)/(2 icc)`` (so the within-block ICC of the binary responses is exactly
    ``icc``) and ``X_ij | p_i ~ Bernoulli(p_i)``.  Every marginal remains 1/2, so the
    null of equal treatment probabilities is true by symmetry and any excess rejection
    is a Type I error.  The beta-binomial is the canonical overdispersion model beyond
    the additive block effect that Cochran's Q already absorbs.

    Returns
    -------
    numpy.ndarray, shape (N,)
        Cochran Q statistics.
    """
    if icc <= 0:
        X = (rng.binomial(1, 0.5, (N, n, k)) if rng is not None
             else np.random.binomial(1, 0.5, (N, n, k)))
    else:
        a = (1.0 - icc) / (2.0 * icc)
        if rng is not None:
            P = rng.beta(a, a, size=(N, n, 1))
            X = (rng.random((N, n, k)) < P).astype(int)
        else:
            P = np.random.beta(a, a, size=(N, n, 1))
            X = (np.random.random((N, n, k)) < P).astype(int)
    col = X.sum(axis=1)
    row = X.sum(axis=2)
    num = (k - 1) * (k * (col ** 2).sum(axis=1) - col.sum(axis=1) ** 2)
    den = k * row.sum(axis=1) - (row ** 2).sum(axis=1)
    den = np.where(den == 0, 1e-15, den)
    return num / den


# ---------------------------------------------------------------------------
# 6. Registry
# ---------------------------------------------------------------------------

def _n_total_13(test_id, n, k_default=3):
    """Total sample size entering each test at per-group size ``n`` (14-test panel)."""
    if test_id == "T11_kpss":
        return n                      # series length
    if test_id == "T12_hoeffding_d":
        return 2 * n                  # bivariate sample
    if test_id == "T13_interaction":
        return N_TRAIN_T13 + n        # training set + test set
    return get_n_total(test_id, n, k_default)


# ---------------------------------------------------------------------------
# T14 -- Szekely-Rizzo distance-correlation t-test (high-dimensional independence)
# ---------------------------------------------------------------------------

def _pairwise_dist(A):
    """Euclidean distance matrix of the rows of ``A``."""
    diff = A[:, None, :] - A[None, :, :]
    return np.sqrt((diff * diff).sum(-1))


def _astar(d):
    """Corrected doubly centered distance matrix A* of Szekely & Rizzo (2013).

    Port of ``energy::Astar`` (R package ``energy`` 1.7-12), the ``A*`` matrix of
    the JMVA 2013 paper::

        A*_ij = (n/(n-1)) * [ a_ij - a_i. - a_.j + a_.. - a_ij/n ],  i != j,
        A*_ii = (n/(n-1)) * (a_i. - a_..).

    Args:
        d: ``(n, n)`` distance matrix.

    Returns:
        numpy.ndarray: the ``(n, n)`` corrected doubly centered matrix.
    """
    n = d.shape[0]
    m = d.mean(axis=1)
    M = d.mean()
    A = d - m[:, None] - m[None, :] + M
    A = A - d / n
    np.fill_diagonal(A, m - M)
    return (n / (n - 1)) * A


def _bcdcor(A, B):
    """Bias-corrected distance correlation of two distance matrices.

    Port of ``energy::BCDCOR``.  Returns ``(bcR, n)``.  ``bcR`` is signed and is
    therefore *not* square-rooted, unlike the classical (non-corrected) dCor.

    Args:
        A, B: ``(n, n)`` distance matrices of the two samples.

    Returns:
        tuple: ``(bcR, n)``, with ``bcR = nan`` when either distance variance is
        non-positive.
    """
    n = A.shape[0]
    AA = _astar(A)
    BB = _astar(B)
    k = n / (n - 2.0)
    XY = (AA * BB).sum() - k * np.diag(AA * BB).sum()
    XX = (AA * AA).sum() - k * np.diag(AA * AA).sum()
    YY = (BB * BB).sum() - k * np.diag(BB * BB).sum()
    if XX <= 0 or YY <= 0:
        return np.nan, n
    return XY / np.sqrt(XX * YY), n


def dcor_t(X, Y):
    """Distance-correlation t statistic of Szekely & Rizzo (2013).

    ``T = sqrt(M-1) * bcR / sqrt(1 - bcR^2)`` with ``M = n(n-3)/2``, referred to
    Student's t with ``M - 1 = n(n-3)/2 - 1`` degrees of freedom, upper tail.
    Verified against ``energy::dcorT`` to 3e-11 on five configurations
    (``verify_dcor_against_R.py``).

    Args:
        X, Y: ``(n, d)`` arrays, observations in rows; the dimensions may differ
            between the two samples.

    Returns:
        tuple: ``(T, df)``; ``T`` is ``nan`` when the correction is undefined.
    """
    n = X.shape[0]
    R, n = _bcdcor(_pairwise_dist(X), _pairwise_dist(Y))
    M = n * (n - 3) / 2.0
    df = M - 1.0
    if not np.isfinite(R) or M <= 1.0:
        return np.nan, df
    if abs(R) >= 1.0:
        return (np.inf if R > 0 else -np.inf), df
    return np.sqrt(M - 1.0) * R / np.sqrt(1.0 - R * R), df


def dcor_t_ref_args(n):
    """Reference parameters of T14 at sample size ``n``: the t degrees of freedom.

    Args:
        n: number of observation pairs.

    Returns:
        tuple: ``(df,)`` with ``df = n(n-3)/2 - 1``.
    """
    return (n * (n - 3) / 2.0 - 1.0,)


def mc_dcor_t(n, N, d=10, dep="indep", rho=0.5, rng=None, **kwargs):
    """Study-1/Study-2 generator for T14 (distance-correlation t-test).

    Null configuration (``dep='indep'``): the rows of ``X`` are i.i.d.
    ``N(0, I_d)``, the rows of ``Y`` are i.i.d. ``N(0, I_d)``, and ``X`` and ``Y``
    are independent.  The reference law is Student's t with ``n(n-3)/2 - 1``
    degrees of freedom and the test rejects in the upper tail, since large
    positive ``bcR`` indicates dependence.  The reference depends on ``n``, but
    only through its degrees of freedom, which is explicit and documented.

    Dependence *within* one vector is not dependence between the vectors and does
    not violate the null: ``eta`` sets the within-vector correlation
    ``Sigma_ab = eta^|a-b|`` while keeping ``X`` and ``Y`` independent, so it is a
    nuisance configuration rather than an alternative.  Genuine alternatives are
    ``dep='linear'`` (``Y = rho X + noise``) and ``dep='nonlinear'``
    (``Y[,0] = X[,0]^2 - 1 + rho * noise``, a dependence with zero linear
    correlation).

    Args:
        n: number of observation pairs.
        N: number of Monte Carlo replications.
        d: dimension of each vector.
        dep: ``'indep'``, ``'linear'`` or ``'nonlinear'``.
        rho: strength of the dependence for the non-null settings.
        eta: within-vector correlation parameter (keyword; default 0).
        rng: ``numpy.random.Generator``; defaults to the legacy global stream.

    Returns:
        numpy.ndarray of shape ``(N,)`` of t statistics.
    """
    eta = float(kwargs.get("eta", 0.0))
    if rng is None:
        rng = np.random

    def draw(m, k):
        Z = rng.standard_normal((m, k))
        if eta > 0.0:
            idx = np.arange(k)
            Sig = eta ** np.abs(idx[:, None] - idx[None, :])
            L = np.linalg.cholesky(Sig + 1e-12 * np.eye(k))
            Z = Z @ L.T
        return Z

    out = np.empty(N)
    for i in range(N):
        X = draw(n, d)
        if dep == "indep":
            Y = draw(n, d)
        elif dep == "linear":
            Y = rho * X + np.sqrt(max(1.0 - rho * rho, 0.0)) * draw(n, d)
        elif dep == "nonlinear":
            Y = draw(n, d)
            # Y[,0] = (X[,0]^2 - 1) * a + noise, so that rho 1 -> 0 moves the
            # alternative from strong quadratic dependence to independence.
            a = rho
            Y[:, 0] = a * (X[:, 0] ** 2 - 1.0) + Y[:, 0]
        else:
            raise ValueError(f"unknown dep {dep!r}")
        out[i] = dcor_t(X, Y)[0]
    return out


TESTS = {
    # --- archived panel (T1-T10): theoretical null may be the exact finite-sample law ---
    "T1_t_test": dict(
        label="Student t", family="t", null="t", null_args=(),
        ref_args_fn=lambda n: (n - 1,), type1="two_sided",
        n_independent_ref=False,
        note="exact t(n-1) null under normality; KS uses the exact finite-sample reference"),
    "T2_anova": dict(
        label="ANOVA F", family="F", null="f", null_args=(),
        ref_args_fn=lambda n: (2, 3 * n - 3), type1="upper_tail",
        n_independent_ref=False, note="F(k-1, N-k), k=3 equal groups"),
    "T3_kruskal_wallis": dict(
        label="Kruskal-Wallis", family="chi2", null="chi2", null_args=(2,),
        type1="upper_tail", n_independent_ref=True, note="asymptotic chi2(k-1), k=3"),
    "T4_chisq_gof": dict(
        label="Chi-sq GOF", family="chi2", null="chi2", null_args=(0,),
        ref_args_fn=lambda n: (5,), type1="upper_tail", n_independent_ref=True,
        note="k=6 categories, sample size max(5n, 30)"),
    "T5_bartlett": dict(
        label="Bartlett", family="chi2", null="chi2", null_args=(2,),
        type1="upper_tail", n_independent_ref=True, note="asymptotic chi2(k-1), k=3"),
    "T6_fligner_killeen": dict(
        label="Fligner-Killeen", family="chi2", null="chi2", null_args=(2,),
        type1="upper_tail", n_independent_ref=True, note="asymptotic chi2(k-1), k=3"),
    "T7_sign_test": dict(
        label="Sign test", family="binomial", null="binom", null_args=(),
        ref_args_fn=lambda n: (n, 0.5), type1="binom", n_independent_ref=False,
        note="exact b(n, 1/2); discreteness floor documented in the paper"),
    "T8_cochran_q": dict(
        label="Cochran Q", family="chi2", null="chi2", null_args=(2,),
        type1="upper_tail", n_independent_ref=True, note="asymptotic chi2(k-1), k=3"),
    "T9_median_test": dict(
        label="Median test", family="chi2", null="chi2", null_args=(2,),
        type1="upper_tail", n_independent_ref=True, note="asymptotic chi2(k-1), k=3"),
    "T10_hotelling": dict(
        label="Hotelling T2", family="F", null="f", null_args=(),
        ref_args_fn=lambda n: (2, n - 2), type1="upper_tail",
        n_independent_ref=False, note="exact F(p, n-p) for p=2 after transformation"),
    # --- new tests (T11-T13): sample-size independent reference by construction ---
    "T11_kpss": dict(
        label="KPSS", family="CvM limit", null="cvm_limit", null_args=(),
        type1="upper_tail", n_independent_ref=True,
        note="LM statistic vs Cramer-von Mises limit law; Bartlett/Newey-West HAC with "
             "Schwert bandwidth l = floor(4 (n/100)^(1/4)); no standardization needed"),
    "T12_hoeffding_d": dict(
        label="Hoeffding D", family="Hmisc/BKR asymptotic", null="hoeffding_law", null_args=(),
        crit_null="hoeffding_z", crit_null_args=(),
        type1="upper_tail", n_independent_ref=False,
        note="statistic standardized as z = 0.5 pi^4 (n d + 1/36). The KS distance uses "
             "an independently estimated limit law; the decision level uses the published "
             "Hollander-Wolfe/Hmisc tail table, whose own extrapolation branch is not "
             "continuous with the table and is therefore not a valid CDF."),
    "T14_dcor_t": dict(
        label="dCor t-test", family="Student t (high-dim)", null="t", null_args=(),
        ref_args_fn=dcor_t_ref_args, type1="upper_tail",
        n_independent_ref=False,
        note=("Szekely-Rizzo bias-corrected distance-correlation t-test; the "
              "reference is Student t with n(n-3)/2-1 df, so it depends on n "
              "through the degrees of freedom only.")),
    "T13_interaction": dict(
        label="Interaction IAD", family="normal (self-normalized)", null="norm", null_args=(0.0, 1.0),
        type1="two_sided", n_independent_ref=True,
        note="Pitman z of the interaction difference (Welchowski & Edelmann 2024) with a "
             "fixed LightGBM learner; standardized by construction, reference N(0,1)"),
}

# --- adapter for t2-sensitivity: run T11-T13 inside the severity ladders ---

def _parent_draw(n, dist, rng):
    """Draw n i.i.d. draws from a standardized parent family (mean 0, variance 1)."""
    name, par = dist
    if name == "norm":
        return rng.standard_normal(n)
    if name == "exp":
        return rng.exponential(1.0, n) - 1.0
    if name == "chi2":
        return (rng.chisquare(par, n) - par) / np.sqrt(2.0 * par)
    if name == "lognorm":
        s = float(par)
        m = np.exp(s * s / 2.0)
        v = (np.exp(s * s) - 1.0) * np.exp(s * s)
        return (rng.lognormal(0.0, s, n) - m) / np.sqrt(v)
    if name == "t":
        return rng.standard_t(par, n) / np.sqrt(par / (par - 2.0))
    raise ValueError(f"unknown parent family {name!r}")


def _hook_kpss(spec, N, rng):
    """Severity-ladder adapter for T11 (KPSS).

    Args:
        spec: ladder cell spec; ``spec['dist']`` is a ``(kind, par)`` parent
            family and ``spec['ns'][0]`` is the series length.
        N: number of replications.
        rng: ``numpy.random.Generator`` used for the draws.

    Returns:
        np.ndarray of shape (N,) of KPSS LM statistics, already on the scale of
        the Cramer-von Mises limit law (no standardization needed).
    """
    dist = spec.get("dist", ("norm", None))
    n = int(spec["ns"][0])
    e = np.empty((N, n))
    for i in range(N):
        e[i] = _parent_draw(n, dist, rng)
    return np.array([kpss_statistic(e[i]) for i in range(N)])


def _hook_hoeffding(spec, N, rng):
    """Severity-ladder adapter for T12 (Hoeffding's D).

    Args:
        spec: ladder cell spec; ``spec['ns'][0]`` is the bivariate sample size.
        N: number of replications.
        rng: ``numpy.random.Generator`` used for the draws.

    Returns:
        np.ndarray of shape (N,) of the standardized statistic
        z = 0.5 pi^4 (n d + 1/36), whose reference law is the scaled Hoeffding
        asymptotic law.
    """
    dist = spec.get("dist", ("norm", None))
    n = int(spec["ns"][0])
    out = np.empty(N)
    for i in range(N):
        x = _parent_draw(n, dist, rng)
        y = _parent_draw(n, dist, rng)
        out[i] = hoeffding_z(hoeffding_d(x, y), n)
    return out


def _hook_interaction(spec, N, rng):
    """Severity-ladder adapter for T13 (interaction / IAD).

    Args:
        spec: ladder cell spec; ``spec['ns'][0]`` is the test-set size, while the
            training set is fixed at ``N_TRAIN_T13``.
        N: number of replications.
        rng: ``numpy.random.Generator``; one seed is drawn from it and handed to
            the LightGBM-based generator so that the cell stays reproducible.

    Returns:
        np.ndarray of shape (N,) of the self-normalized Pitman z statistic, to be
        compared with a two-sided N(0,1) reference.
    """
    dist = spec.get("dist", ("norm", None))
    n = int(spec["ns"][0])
    seed = int(rng.integers(1, 2 ** 31 - 1))
    z, h2 = mc_interaction_h(n, N, kind="additive", seed=seed, return_h2=True)
    return z


SEVERITY_HOOKS = {
    "T11_kpss": dict(
        label="KPSS",
        rule=("cvm_limit",),
        gen=_hook_kpss,
        note="upper-tail rejection against the Cramer-von Mises limit law; spec['ns'][0] "
             "is the series length"),
    "T12_hoeffding_d": dict(
        label="Hoeffding D",
        rule=("hoeffding_z",),
        gen=_hook_hoeffding,
        note="upper-tail rejection against the scaled Hoeffding asymptotic law; "
             "spec['ns'][0] is the bivariate sample size"),
    "T13_interaction": dict(
        label="Interaction IAD",
        rule=("norm",),
        gen=_hook_interaction,
        note="two-sided N(0,1) rejection; spec['ns'][0] is the test-set size, the "
             "training set is fixed at N_TRAIN_T13=1000"),
}


# ---------------------------------------------------------------------------
# 7. Study runners
# ---------------------------------------------------------------------------

def make_meta(seed=SEED, extra=None):
    """Reproducibility block attached to every results file.

    Returns
    -------
    dict
        ``{seed, numpy, scipy, python, git_hash, timestamp}`` (+ ``extra``).
    """
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                           cwd=os.path.dirname(os.path.abspath(__file__)),
                                           stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_hash = "unknown"
    meta = {
        "seed": int(seed),
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "python": platform.python_version(),
        "git_hash": git_hash,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_mc": N_MC,
        "ks_threshold": KS_THRESHOLD,
        "alphas": list(ALPHAS),
    }
    if extra:
        meta.update(extra)
    return meta


def save_results(obj, path):
    """Write a results dict to JSON (NaN preserved, as in the archive results file)."""
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=float)
    return path


def update_results(path, **keys):
    """Merge ``keys`` into an existing results file (or create it), atomically.

    Used so that ``rev_study1_ext.py`` (Study 1 / null calibration) and
    ``rev_alpha_dependence.py`` (alpha sweep / dependence) can both contribute to
    ``rev_results_t1.json`` without either overwriting the other's section: the merge is
    a *top-level* update, so a partial-stage run refreshes only the blocks it produced.
    Pass ``stage="<name>"`` to append the block provenance to ``meta.stages_run``.

    Parameters
    ----------
    path : str
        Results file.
    **keys
        Sections to set or replace.

    Returns
    -------
    dict
        The merged result object.
    """
    obj = {}
    if os.path.exists(path):
        with open(path) as fh:
            obj = json.load(fh)
    stages = list(obj.get("meta", {}).get("stages_run", []))
    for k, v in keys.items():
        if k == "stage":
            continue
        obj[k] = v                      # top-level merge: untouched sections preserved
    if "stage" in keys:
        stages.append(keys["stage"])
        obj.setdefault("meta", {})["stages_run"] = stages
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, default=float)
    os.replace(tmp, path)
    return obj


def run_convergence(test_id, ns, N=N_MC, alphas=ALPHAS, gen=None, ref_args_fn=None,
                    seed=None, verbose=False):
    """Sweep ``n`` for one test and evaluate the MC null sample at every ``alpha``.

    Parameters
    ----------
    test_id : str
        Key of :data:`TESTS`.
    ns : sequence of int
        Per-group sample sizes.
    N : int
        Monte Carlo replications (identical across tests so that ``D_KS`` is comparable).
    alphas : sequence of float
        Nominal levels; the *same* sample is evaluated at each level.
    gen : callable, optional
        ``gen(n, N) -> array`` returning the statistic under the *null* configuration.
        Defaults to a dispatcher that mirrors ``experiments.run_study1`` exactly.
    ref_args_fn : callable, optional
        ``n -> tuple`` of reference parameters, overriding the registry.
    seed : int, optional
        If given, re-seeds the legacy global NumPy stream before the sweep.
    verbose : bool
        Print one line per sample size.

    Returns
    -------
    dict
        ``{test_id, ns, n_total, N, ks, type1: {alpha: [...]}, ci_lo, ci_hi, critical}``.
    """
    if seed is not None:
        np.random.seed(seed)
    gen = gen or default_generator(test_id)
    out = {"label": TESTS[test_id]["label"], "ns": list(ns), "n_total": [],
           "N": int(N), "ks": [], "ks_p": [],
           "type1": {f"{a:.2f}": [] for a in alphas},
           "ci_lo": {f"{a:.2f}": [] for a in alphas},
           "ci_hi": {f"{a:.2f}": [] for a in alphas},
           "critical": {f"{a:.2f}": [] for a in alphas}}
    for n in ns:
        t0 = time.time()
        try:
            sv = gen(n, N)
            rargs = ref_args_fn(n) if ref_args_fn is not None else None
            ev = evaluate_multi_alpha(sv, test_id, alphas, n=n, ref_args=rargs)
            out["ks"].append(ev[f"{alphas[1]:.2f}"]["ks"])
            ks_p = ks_distance_p(sv, test_id, rargs)
            out["ks_p"].append(ks_p)
            for a in alphas:
                key = f"{a:.2f}"
                out["type1"][key].append(ev[key]["type1"])
                out["ci_lo"][key].append(ev[key]["ci_lo"])
                out["ci_hi"][key].append(ev[key]["ci_hi"])
                c = ev[key]["critical_value"]
                out["critical"][key].append(list(c) if isinstance(c, tuple) else c)
        except Exception as exc:  # keep the grid running, record the failure
            out["ks"].append(float("nan"))
            out["ks_p"].append(float("nan"))
            for a in alphas:
                key = f"{a:.2f}"
                out["type1"][key].append(float("nan"))
                out["ci_lo"][key].append(float("nan"))
                out["ci_hi"][key].append(float("nan"))
                out["critical"][key].append(None)
            if verbose:
                print(f"    [ERR n={n}: {exc}]", flush=True)
        out["n_total"].append(_n_total_13(test_id, n))
        if verbose:
            print(f"    n={n:<4d} ks={out['ks'][-1]:.4f} "
                  f"a05={out['type1']['0.05'][-1]:.4f} ({time.time() - t0:.1f}s)", flush=True)
    return out


def ks_distance_p(stat_values, test_id, ref_args=None):
    """p-value of the one-sample KS test against the theoretical null (diagnostic only)."""
    t = TESTS[test_id]
    sv = np.asarray(stat_values, dtype=float)
    sv = sv[np.isfinite(sv)]
    if sv.size < 2 or t["null"] in REFERENCE_DISTS or t["null"] == "binom":
        return float("nan")
    args = ref_args if ref_args is not None else t.get("null_args", ())
    try:
        return float(stats.kstest(sv, t["null"], args=args).pvalue)
    except Exception:
        return float("nan")


def default_generator(test_id):
    """Default null-configuration generator, mirroring ``experiments.run_study1``.

    The call sequence is identical to ``experiments.run_study1`` (same generators, same
    arguments, same order), so evaluating T1-T10 with this dispatcher under
    ``np.random.seed(2024)`` reproduces the archived Study 1 numbers bit-for-bit; T11-T13
    are appended after the archived stream.
    """
    k = 3
    if test_id == "T1_t_test":
        return lambda n, N: mc_t_test(n, N)
    if test_id == "T2_anova":
        return lambda n, N: mc_anova(k, n, N)[0]
    if test_id == "T3_kruskal_wallis":
        return lambda n, N: mc_kruskal_wallis(k, n, N)
    if test_id == "T4_chisq_gof":
        return lambda n, N: mc_chisq_gof(6, max(n * 5, 30), N)
    if test_id == "T5_bartlett":
        return lambda n, N: mc_bartlett(k, n, N)
    if test_id == "T6_fligner_killeen":
        return lambda n, N: mc_fligner_killeen(k, n, N)
    if test_id == "T7_sign_test":
        return lambda n, N: mc_sign_test(n, N)
    if test_id == "T8_cochran_q":
        return lambda n, N: mc_cochran_q(k, n, N)
    if test_id == "T9_median_test":
        return lambda n, N: mc_median_test(k, n, N)
    if test_id == "T10_hotelling":
        return lambda n, N: mc_hotelling_t2(2, n, N)[0]
    if test_id == "T11_kpss":
        return lambda n, N: mc_kpss(n, N)
    if test_id == "T12_hoeffding_d":
        return lambda n, N: mc_hoeffding_d(n, N)
    if test_id == "T13_interaction":
        return lambda n, N: mc_interaction_h(n, N, n_jobs=1)
    raise KeyError(test_id)


PANEL_13 = list(TESTS.keys())
PANEL_10 = PANEL_13[:10]
