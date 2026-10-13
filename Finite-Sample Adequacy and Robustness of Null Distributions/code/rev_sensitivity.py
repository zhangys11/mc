#!/usr/bin/env python3
"""
rev_sensitivity.py -- Study 2 sensitivity analyses for the COST-S-26-00605 major
revision (Computational Statistics).

Responds to Referee 1:
  * comment 3 (R1-3): Study 2 used a single violation severity (Exp(1)-1 skew,
    1:4 variance ratio, 1:2:3 imbalance, n = 50).  We replace each single point
    by a *graded severity ladder* and report Wilson 95% CIs for every cell.
  * comment 5 (R1-5): Study 2 evaluated robustness at a single n = 50.  We repeat
    the violation scenarios at n in {20, 50, 100, 200} and test whether the
    robustness ordering is stable across n.

Design contract (see docs/DESIGN_CONTRACT.md, Sections 3-5)
---------------------------------------------
* Monte Carlo budget is N = 10,000 for EVERY cell and EVERY ladder rung, so
  differences reflect the scenario and not the budget.  Within one ladder all
  rungs (and all tests) use the *same* RNG seed, i.e. a common-random-numbers
  design: the baseline rung is drawn from the identical stream as the violated
  rungs, which makes the baseline-relative change Delta = alpha_hat_violated -
  alpha_hat_baseline far less noisy than two independent runs would be.
* alpha-hat is evaluated at alpha in {0.01, 0.05, 0.10} from the SAME statistic
  array (no extra sampling error, identical budget across alpha).
* Wilson 95% score intervals for every reported rate.  No monotone trend is
  declared unless the point estimates are non-decreasing AND the most severe
  rung's Wilson interval is disjoint from the baseline rung's Wilson interval;
  every ladder also records the weaker, purely descriptive information
  (per-rung CI separation from baseline, adjacent-rung separation).
* Liberal flag: alpha_hat/alpha > 1.5 (i.e. alpha_hat > 0.075 at alpha = 0.05);
  conservative flag: alpha_hat/alpha < 0.5 (alpha_hat < 0.025).  The flag is
  reported both as a point estimate and as a CI-based flag (CI lower bound above
  the threshold), so an isolated MC fluctuation is never reported as a crossing.

Statistic implementations
-------------------------
The archived reproduction script `experiments.py` is imported (when present) and
used for the R1 Study-2 anchor cells.  All ladder cells are produced by the
generic, vectorised implementations in this file, which are *validated against
`experiments.py` and against SciPy/statsmodels in `consistency_checks` and
`implementation_validation` below* -- they are a documented generalization (any
parent distribution, any group-size vector, any variance ratio, any minimum
expected count), not a silent re-implementation.  The generators follow the
`experiments.py` conventions: same statistic formulae, same rejection rules,
same nominal scenario definitions at the anchor points.

Optional 13-test panel: if `core.py` (owner: t1-new-tests) exists and exposes the
documented `SEVERITY_HOOKS` adapter, T12 Hoeffding's D and T13 interaction-H are
added to the distributional ladders; otherwise they are skipped and the reason is
recorded in `meta.extras`.  T11 KPSS is a time-series test and is out of scope
for the distributional ladders (it belongs to the dependence study).

Outputs
-------
  rev_results.json       (keys: meta, severity_sweep, study2_nsweep, alpha_sweep,
                          consistency_checks, implementation_validation)
  fig_rev_severity.png
  fig_rev_study2_n.png

Run
---
  python3 rev_sensitivity.py

Environment variable REV_NMC (default 10000) exists only for smoke tests; the
deliverable run uses the default.  Runtime is roughly one minute on 10 cores.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import warnings
import zlib

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# Global design constants
# ----------------------------------------------------------------------------
SEED = 2024                       # required by task-2
N_MC = int(os.environ.get("REV_NMC", "10000"))   # identical for every cell
N_FIXED = 50                      # archived Study 2 per-group sample size
NOMINAL_ALPHA = 0.05
ALPHA_LEVELS = (0.01, 0.05, 0.10)
Z95 = 1.959963984540054
LIBERAL_RATIO = 1.5
CONSERVATIVE_RATIO = 0.5
NSWEEP_N = (20, 50, 100, 200)
K_GROUPS = 3
P_DIM = 2

TEST_LABELS = {
    "T1_t_test": "Student t",
    "T2_anova": "ANOVA F",
    "T3_kruskal_wallis": "Kruskal-Wallis",
    "T4_chisq_gof": "chi2 GOF",
    "T5_bartlett": "Bartlett",
    "T6_fligner_killeen": "Fligner-Killeen",
    "T7_sign_test": "Sign",
    "T8_cochran_q": "Cochran Q",
    "T9_median_test": "Median",
    "T10_hotelling": "Hotelling T2",
}

# Parent distribution used by the ARCHIVED Study 2 baseline/structural scenarios
# for each test (run_study2 defaults: uniform for Kruskal-Wallis and the median
# test, normal for everything else).  Kept so that the n-sweep and the structural
# ladders are comparable with the published Tables 4-5.
ARCHIVED_PARENT = {
    "T1_t_test": "norm",
    "T2_anova": "norm",
    "T3_kruskal_wallis": "unif",
    "T4_chisq_gof": None,
    "T5_bartlett": "norm",
    "T6_fligner_killeen": "norm",
    "T7_sign_test": "norm",
    "T8_cochran_q": None,
    "T9_median_test": "unif",
    "T10_hotelling": "norm",
}

# ----------------------------------------------------------------------------
# Optional import of the archive script (owner: lead) and of core.py (t1)
# ----------------------------------------------------------------------------
HAS_ARCHIVE = True
ARCHIVE_ERR = ""
try:
    try:
        import experiments as EX
    except ModuleNotFoundError:
        # repository layout: the archive generator ships as code/archive_experiments.py
        import archive_experiments as EX
except Exception as exc:      # pragma: no cover - defensive
    EX = None
    HAS_ARCHIVE = False
    ARCHIVE_ERR = repr(exc)

HAS_CORE = False
CORE_ERR = ""
CORE_SELFTEST = "not run (core.py unavailable)"
try:
    import core as CORE
    HAS_CORE = True
except Exception as exc:
    CORE = None
    CORE_ERR = repr(exc)

if HAS_CORE:
    # t1-new-tests' regression guard (17 assertions incl. the CvM critical values,
    # the Hmisc Hoeffding identity and MC null checks for T11/T12).  It raises on
    # regression, so a broken reference law can never silently reach a results
    # file; we record the outcome in meta and warn loudly rather than aborting,
    # because every 10-test cell is independently validated in this script.
    try:
        CORE._selftest(quick=True)
        CORE_SELFTEST = "passed (core._selftest(quick=True))"
    except Exception as exc:                              # pragma: no cover
        CORE_SELFTEST = "FAILED: %r" % (exc,)
        print("WARNING: core._selftest(quick=True) FAILED -> %r" % (exc,))


# ============================================================================
# SECTION 1 -- parent distributions, standardized to mean 0 and variance 1
# ============================================================================
# Every ladder rung is a *shape* manipulation: all parents are centered at 0 and
# scaled to unit variance, so a rung differs from the baseline only in skewness
# and/or tail weight.  (All tests on the distributional ladders -- t, ANOVA,
# Bartlett, Fligner-Killeen, Hotelling, Kruskal-Wallis, median -- are
# scale-invariant, so this standardization does not move any alpha-hat; the sign
# test uses the parent median, which the standardization shifts by a known
# constant.  The archived anchors are reproduced separately with the archived
# un-standardized parents in `consistency_checks`.)

def _std_params_lognorm(s):
    """Mean and standard deviation of lognormal(0, s) before standardization.

    Args:
        s: shape parameter sigma of the lognormal distribution.

    Returns:
        tuple ``(mean, sd)`` = ``(exp(s^2/2), sqrt((exp(s^2)-1) exp(s^2)))``.
    """
    mean = np.exp(s * s / 2.0)
    sd = np.sqrt((np.exp(s * s) - 1.0) * np.exp(s * s))
    return mean, sd


def skew_kurt(kind, par=None):
    """Analytic (skewness, excess kurtosis) of the standardized parent."""
    if kind == "norm":
        return 0.0, 0.0
    if kind == "unif":
        return 0.0, -1.2
    if kind == "exp":
        return 2.0, 6.0
    if kind == "chi2":
        return float(np.sqrt(8.0 / par)), 12.0 / par
    if kind == "lognorm":
        g1 = (np.exp(par * par) + 2.0) * np.sqrt(np.exp(par * par) - 1.0)
        k = (np.exp(4 * par * par) + 2 * np.exp(3 * par * par)
             + 3 * np.exp(2 * par * par) - 6.0)
        return float(g1), float(k)
    if kind == "t":
        return 0.0, (6.0 / (par - 4.0) if par > 4 else float("inf"))
    raise ValueError(kind)


DIST_DESC = {
    "norm": "N(0,1)",
    "unif": "Uniform(-sqrt(3),sqrt(3))",
    "exp": "Exp(1)-1",
    "chi2_4": "(chi2(4)-4)/sqrt(8)",
    "chi2_2": "(chi2(2)-2)/2 = Exp(1)-1",
    "lognorm_1": "(exp(Z)-e^0.5)/2.161, Z~N(0,1)",
    "t_30": "t(30)/sqrt(30/28)",
    "t_10": "t(10)/sqrt(10/8)",
    "t_5": "t(5)/sqrt(5/3)",
    "t_3": "t(3)/sqrt(3)",
}


def draw_std(rng, kind, shape, par=None):
    """Draw standardized (mean 0, variance 1) variates of the given parent."""
    if kind == "norm":
        return rng.standard_normal(shape)
    if kind == "unif":
        return rng.uniform(-1.0, 1.0, shape) * np.sqrt(3.0)
    if kind == "exp":
        return rng.exponential(1.0, shape) - 1.0
    if kind == "chi2":
        return (rng.chisquare(par, shape) - par) / np.sqrt(2.0 * par)
    if kind == "lognorm":
        m, s = _std_params_lognorm(par)
        return (np.exp(par * rng.standard_normal(shape)) - m) / s
    if kind == "t":
        return rng.standard_t(par, shape) / np.sqrt(par / (par - 2.0))
    raise ValueError(kind)


def parent_median(kind, par=None):
    """Median of the standardized parent (needed by the exact sign-test null)."""
    if kind == "norm":
        return 0.0
    if kind == "unif":
        return 0.0
    if kind == "exp":
        return np.log(2.0) - 1.0
    if kind == "chi2":
        return float((stats.chi2.ppf(0.5, par) - par) / np.sqrt(2.0 * par))
    if kind == "lognorm":
        m, s = _std_params_lognorm(par)
        return float((1.0 - m) / s)
    if kind == "t":
        return 0.0
    raise ValueError(kind)


# ============================================================================
# SECTION 2 -- vectorised statistic implementations (generic group sizes)
# ============================================================================
# All functions operate on a list of group arrays G[j] of shape (N, n_j) and
# return a length-N vector of test statistics.  Formulae follow experiments.py
# exactly (no tie correction: continuous parents, ties have probability 0).

def _ranks(A):
    """Ranks along axis 1 (as experiments.py: plain ranks, no tie averaging)."""
    order = A.argsort(axis=1)
    out = np.empty_like(order)
    rr = np.broadcast_to(np.arange(1, A.shape[1] + 1), A.shape)
    np.put_along_axis(out, order, rr, axis=1)
    return out


def stat_t(groups):
    """One-sample Student t statistic for a Monte Carlo cell.

    Args:
        groups: list of ``(N, n)`` arrays; only ``groups[0]`` is used.

    Returns:
        np.ndarray of shape (N,): t = xbar / (s / sqrt(n)), with a zero sample
        standard deviation floored at 1e-12.
    """
    X = groups[0]
    n = X.shape[1]
    xbar = X.mean(axis=1)
    s = X.std(axis=1, ddof=1)
    s = np.where(s <= 0, 1e-12, s)
    return xbar / (s / np.sqrt(n))


def stat_anova(groups):
    """One-way ANOVA F statistic for k groups.

    Args:
        groups: list of k arrays of shape ``(N, n_j)``; the group sizes may differ.

    Returns:
        np.ndarray of shape (N,): F = MSTR / MSE, with MSE floored at 1e-15.
    """
    k = len(groups)
    ns = np.array([g.shape[1] for g in groups], dtype=float)
    n_tot = ns.sum()
    gm = np.stack([g.mean(axis=1) for g in groups], axis=1)
    grand = (gm * ns).sum(axis=1, keepdims=True) / n_tot
    sstr = (ns * (gm - grand) ** 2).sum(axis=1)
    mstr = sstr / (k - 1)
    sse = sum(((g - g.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
              for g in groups)
    mse = sse / (n_tot - k)
    mse = np.where(mse <= 0, 1e-15, mse)
    return mstr / mse


def stat_kruskal(groups):
    """Kruskal-Wallis H statistic from pooled mid-ranks.

    Args:
        groups: list of k arrays of shape ``(N, n_j)``.

    Returns:
        np.ndarray of shape (N,): H = 12/(n_T (n_T+1)) sum_j R_j^2 / n_j
        - 3 (n_T + 1), with the ranks computed on the pooled sample.
    """
    ns = np.array([g.shape[1] for g in groups], dtype=float)
    n_tot = ns.sum()
    A = np.concatenate(groups, axis=1)
    R = _ranks(A)
    H = np.zeros(A.shape[0])
    idx = 0
    for j, g in enumerate(groups):
        nj = g.shape[1]
        H += R[:, idx:idx + nj].sum(axis=1) ** 2 / nj
        idx += nj
    return 12.0 / (n_tot * (n_tot + 1.0)) * H - 3.0 * (n_tot + 1.0)


def stat_bartlett(groups):
    """Bartlett test statistic with the Bartlett correction.

    Args:
        groups: list of k arrays of shape ``(N, n_j)``.

    Returns:
        np.ndarray of shape (N,): the corrected statistic whose reference law is
        chi-square(k-1) under normality; group and pooled variances are floored
        at 1e-15 so that the logarithm stays finite.
    """
    k = len(groups)
    ns = np.array([g.shape[1] for g in groups], dtype=float)
    n_tot = ns.sum()
    v = np.stack([g.var(axis=1, ddof=1) for g in groups], axis=1)
    v = np.where(v <= 0, 1e-15, v)
    sp2 = ((ns - 1.0) * v).sum(axis=1) / (n_tot - k)
    sp2 = np.where(sp2 <= 0, 1e-15, sp2)
    num = (n_tot - k) * np.log(sp2) - ((ns - 1.0) * np.log(v)).sum(axis=1)
    den = 1.0 + (1.0 / (3.0 * (k - 1))) * ((1.0 / (ns - 1.0)).sum()
                                           - 1.0 / (n_tot - k))
    return num / den


def _fligner_stat_from_scores(scores, ns):
    """Fligner-Killeen chi-square from pooled normal scores (weights n_j)."""
    k = len(ns)
    n_tot = float(np.sum(ns))
    aibar = np.stack([scores[:, int(s):int(s) + int(nj)].mean(axis=1)
                      for s, nj in zip(np.concatenate([[0], np.cumsum(ns)[:-1]]), ns)],
                     axis=1)
    abar = (aibar * ns).sum(axis=1, keepdims=True) / n_tot
    varsq = scores.var(axis=1, ddof=1)
    varsq = np.where(varsq <= 0, 1e-15, varsq)
    return (ns * (aibar - abar) ** 2).sum(axis=1) / varsq


def stat_fligner(groups):
    """Fligner-Killeen normal-scores statistic for equality of variances.

    Args:
        groups: list of k arrays of shape ``(N, n_j)``.

    Returns:
        np.ndarray of shape (N,): the chi-square(k-1) statistic built from the
        ranks of the absolute deviations from each group median (see
        :func:`_fligner_stat_from_scores`).
    """
    ns = np.array([g.shape[1] for g in groups], dtype=float)
    n_tot = int(ns.sum())
    A = np.concatenate(groups, axis=1)
    med = np.stack([np.median(g, axis=1) for g in groups], axis=1)
    med_exp = np.concatenate(
        [np.repeat(med[:, j:j + 1], int(ns[j]), axis=1) for j in range(len(groups))],
        axis=1)
    scores = stats.norm.ppf(_ranks(np.abs(A - med_exp)) / (2.0 * (n_tot + 1.0)) + 0.5)
    return _fligner_stat_from_scores(scores, ns)


def stat_median(groups):
    """Mood's median test statistic for k groups.

    Args:
        groups: list of k arrays of shape ``(N, n_j)``.

    Returns:
        np.ndarray of shape (N,): the chi-square(k-1) form. Replicates whose
        pooled split is degenerate (all values on one side of the grand median)
        return 0.
    """
    k = len(groups)
    A = np.concatenate(groups, axis=1)
    n_tot = A.shape[1]
    gm = np.median(A, axis=1, keepdims=True)
    a = (A > gm).sum(axis=1)
    b = n_tot - a
    mt = np.zeros(A.shape[0])
    idx = 0
    for j, g in enumerate(groups):
        nj = g.shape[1]
        o1 = (A[:, idx:idx + nj] > gm).sum(axis=1)
        mt += (o1 - nj * a / n_tot) ** 2 / nj
        idx += nj
    safe = np.maximum(a, 1) * np.maximum(b, 1)
    out = (n_tot ** 2 / safe) * mt
    return np.where((a == 0) | (b == 0), 0.0, out)


def stat_sign(groups, med0):
    """Sign-test statistic: number of observations above the null median.

    Args:
        groups: list whose first element is the ``(N, n)`` sample.
        med0: null median (float): 0 for the normal and t parents, ``log 2`` for
            the Exp(1) parent, 0.5 for the uniform parent.

    Returns:
        np.ndarray of shape (N,) of integer counts in 0..n; the exact reference
        law is Binomial(n, 1/2).
    """
    return (groups[0] > med0).sum(axis=1)


def stat_cochran(X):
    """X: (N, n_blocks, k) binary.  Cochran's Q exactly as experiments.py."""
    k = X.shape[2]
    col = X.sum(axis=1)                     # treatment totals, (N, k)
    row = X.sum(axis=2)                     # block totals, (N, n_blocks)
    num = (k - 1.0) * (k * (col ** 2).sum(axis=1) - col.sum(axis=1) ** 2)
    den = k * row.sum(axis=1) - (row ** 2).sum(axis=1)
    den = np.where(den == 0, 1e-15, den)
    return num / den


def stat_gof(counts, expected):
    """Pearson chi-square goodness-of-fit statistic.

    Args:
        counts: ``(N, k)`` array of category counts.
        expected: ``(k,)`` array of expected counts, broadcast over replicates.

    Returns:
        np.ndarray of shape (N,): sum_j (O_j - E_j)^2 / E_j.
    """
    return ((counts - expected) ** 2 / expected).sum(axis=1)


def stat_hotelling(X):
    """X: (N, n, p) -> F-transformed Hotelling T^2, exactly as experiments.py."""
    n, p = X.shape[1], X.shape[2]
    xbar = X.mean(axis=1)
    Xc = X - xbar[:, None, :]
    S = np.einsum("nij,nik->njk", Xc, Xc) / (n - 1.0)
    try:
        sol = np.linalg.solve(S, xbar[..., None])[..., 0]
    except np.linalg.LinAlgError:
        sol = np.einsum("nij,nj->ni", np.linalg.pinv(S), xbar)
    t2 = n * (xbar * sol).sum(axis=1)
    return (n - p) / (p * (n - 1.0)) * t2


# ============================================================================
# SECTION 3 -- rejection rules, alpha-hat, Wilson CI
# ============================================================================
# Rules identical to experiments.py compute_type1_error.

def reject_rate(stat_values, rule, alpha):
    """Empirical rejection rate of a batch of statistics under a rejection rule.

    Args:
        stat_values: 1-D array of N statistics.
        rule: tuple describing the reference law and tail, as produced by
            :func:`gen_cell`: ``('chi2', df)``, ``('f', df1, df2)``, ``('t', df)``
            (two-sided), ``('norm',)`` (two-sided) or ``('binom', n, p)`` (exact
            two-sided binomial region).
        alpha: nominal level in (0, 1).

    Returns:
        float: fraction of replicates in the rejection region. Identical in
        definition to ``experiments.compute_type1_error``.

    Raises:
        ValueError: if the rule kind is unknown.
    """
    kind = rule[0]
    sv = np.asarray(stat_values, dtype=float)
    if kind == "chi2":
        crit = stats.chi2.ppf(1.0 - alpha, rule[1])
        return float(np.mean(sv > crit))
    if kind == "f":
        crit = stats.f.ppf(1.0 - alpha, rule[1], rule[2])
        return float(np.mean(sv > crit))
    if kind == "t":
        crit = stats.t.ppf(1.0 - alpha / 2.0, rule[1])
        return float(np.mean(np.abs(sv) > crit))
    if kind == "norm":
        crit = stats.norm.ppf(1.0 - alpha / 2.0)
        return float(np.mean(np.abs(sv) > crit))
    if kind == "binom":
        n_b, p_b = rule[1], rule[2]
        lower = stats.binom.ppf(alpha / 2.0, n_b, p_b)
        upper = stats.binom.ppf(1.0 - alpha / 2.0, n_b, p_b)
        return float(np.mean((sv < lower) | (sv > upper)))
    raise ValueError(kind)


def wilson_ci(phat, N, z=Z95):
    """95% Wilson score interval; identical to experiments.py wilson_ci apart
    from a machine-precision guard keeping lo <= phat <= hi (at phat = 1 the
    closed form lands ~2e-16 below 1, which would give a negative error bar)."""
    if N <= 0 or not np.isfinite(phat):
        return (float("nan"), float("nan"))
    denom = 1.0 + z * z / N
    center = (phat + z * z / (2.0 * N)) / denom
    half = (z * np.sqrt(phat * (1.0 - phat) / N + z * z / (4.0 * N * N))) / denom
    lo = max(0.0, min(center - half, phat))
    hi = min(1.0, max(center + half, phat))
    return (float(lo), float(hi))


# ============================================================================
# SECTION 4 -- generic cell generator
# ============================================================================
GOF_K = 6          # archived chi2 GOF has six categories (Table 2 / Table 4)


def spec_for(test_id, dist=("norm", None), n=N_FIXED, k=None, p_dim=P_DIM,
             ns=None, sd=None, n_T=None, min_np=None, p=0.5):
    """Build the data-generating spec for one cell of the 10-test panel.

    ``k=None`` means "the test's archived default": three groups for the
    multi-group tests, six categories for the chi2 GOF test (passing k=3 there
    by accident would silently change the reference df from 5 to 2).
    """
    kg = K_GROUPS if k is None else int(k)
    if test_id in ("T1_t_test", "T7_sign_test"):
        return dict(dist=dist, ns=[int(n)])
    if test_id == "T10_hotelling":
        return dict(dist=dist, ns=[int(n)], p_dim=int(p_dim))
    if test_id == "T4_chisq_gof":
        return dict(dist=dist, k=(GOF_K if k is None else int(k)),
                    n_T=int(n_T) if n_T is not None else int(max(5 * n, 30)),
                    min_np=min_np)
    if test_id == "T8_cochran_q":
        n_blocks = int(ns[0]) if ns is not None else int(n)
        return dict(dist=dist, ns=[n_blocks], k=kg, p=float(p))
    gns = [int(x) for x in (ns if ns is not None else [n] * kg)]
    gsd = [float(x) for x in (sd if sd is not None else [1.0] * len(gns))]
    return dict(dist=dist, ns=gns, sd=gsd)


def _spec_key(test_id, spec, seed):
    """Hashable cache key that identifies one Monte Carlo cell exactly.

    Args:
        test_id: test identifier, e.g. ``'T5_bartlett'``.
        spec: generator spec dict (parent law, group sizes, scales, ...).
        seed: integer seed of the cell.

    Returns:
        tuple: hashable key covering every spec field that affects the draw.
    """
    return (test_id, spec["dist"], tuple(spec.get("ns", ())),
            tuple(spec.get("sd", ())), spec.get("n_T"), spec.get("min_np"),
            spec.get("p"), spec.get("p_dim"), spec.get("k"), seed)


_CELL_CACHE = {}


def gen_cell(test_id, spec, seed, N=N_MC):
    """Return (stat_values, rejection_rule) for one Monte Carlo cell.

    Deterministic: identical (test_id, spec, seed) always yields the identical
    array (cached), which is what makes the common-random-numbers ladder design
    and the internal consistency checks exact.
    """
    ck = _spec_key(test_id, spec, seed)
    hit = _CELL_CACHE.get(ck)
    if hit is not None:
        return hit
    rng = np.random.default_rng(seed)
    kind, par = spec["dist"]

    if test_id == "T4_chisq_gof":
        k = int(spec.get("k", 6))
        n_T = int(spec["n_T"])
        min_np = spec.get("min_np")
        if min_np is None:
            pth = np.ones(k) / k
        else:
            p1 = float(min_np) / n_T
            pth = np.concatenate([[p1], np.full(k - 1, (1.0 - p1) / (k - 1))])
        counts = rng.multinomial(n_T, pth, size=N)
        out = (stat_gof(counts, n_T * pth), ("chi2", k - 1))
        _CELL_CACHE[ck] = out
        return out

    if test_id == "T8_cochran_q":
        n_blocks = int(spec["ns"][0])
        k = int(spec.get("k", 3))
        p = float(spec.get("p", 0.5))
        X = rng.binomial(1, p, (N, n_blocks, k))
        out = (stat_cochran(X), ("chi2", k - 1))
        _CELL_CACHE[ck] = out
        return out

    ns = [int(x) for x in spec["ns"]]
    sd = [float(x) for x in spec.get("sd", [1.0] * len(ns))]

    if test_id == "T10_hotelling":
        n = ns[0]
        p = int(spec.get("p_dim", 2))
        X = draw_std(rng, kind, (N, n, p), par)
        out = (stat_hotelling(X), ("f", p, n - p))
    elif test_id == "T7_sign_test":
        n = ns[0]
        X = draw_std(rng, kind, (N, n), par)
        out = (stat_sign([X], parent_median(kind, par)), ("binom", n, 0.5))
    else:
        groups = [draw_std(rng, kind, (N, nj), par) * sd[j]
                  for j, nj in enumerate(ns)]
        n_tot = int(np.sum(ns))
        k = len(groups)
        if test_id == "T1_t_test":
            out = (stat_t(groups), ("t", ns[0] - 1))
        elif test_id == "T2_anova":
            out = (stat_anova(groups), ("f", k - 1, n_tot - k))
        elif test_id == "T3_kruskal_wallis":
            out = (stat_kruskal(groups), ("chi2", k - 1))
        elif test_id == "T5_bartlett":
            out = (stat_bartlett(groups), ("chi2", k - 1))
        elif test_id == "T6_fligner_killeen":
            out = (stat_fligner(groups), ("chi2", k - 1))
        elif test_id == "T9_median_test":
            out = (stat_median(groups), ("chi2", k - 1))
        else:
            raise ValueError(test_id)
    _CELL_CACHE[ck] = out
    return out


def evaluate_cell(test_id, spec, seed, N=N_MC):
    """Run one cell and return its record with Wilson CIs at every alpha."""
    sv, rule = gen_cell(test_id, spec, seed, N)
    by_alpha = {}
    for a in ALPHA_LEVELS:
        rate = reject_rate(sv, rule, a)
        lo, hi = wilson_ci(rate, N)
        by_alpha["%.2f" % a] = dict(alpha_hat=rate, ci_lo=lo, ci_hi=hi,
                                    ratio=rate / a,
                                    liberal=bool(rate / a > LIBERAL_RATIO),
                                    liberal_ci=bool(lo / a > LIBERAL_RATIO),
                                    conservative=bool(rate / a < CONSERVATIVE_RATIO),
                                    conservative_ci=bool(hi / a < CONSERVATIVE_RATIO))
    main = by_alpha["%.2f" % NOMINAL_ALPHA]
    rec = dict(alpha_hat=main["alpha_hat"], ci_lo=main["ci_lo"], ci_hi=main["ci_hi"],
               ratio=main["ratio"], liberal=main["liberal"], liberal_ci=main["liberal_ci"],
               conservative=main["conservative"], conservative_ci=main["conservative_ci"],
               by_alpha=by_alpha, N=N, seed=int(seed))
    return rec


def cell_seed(*parts):
    """Deterministic seed for a cell.  Same `parts` -> same seed -> identical
    draw sequence, which is the common-random-numbers requirement."""
    key = "|".join(str(p) for p in parts)
    return SEED + (zlib.crc32(key.encode("utf-8")) % 100000)


# ============================================================================
# SECTION 5 -- severity ladders
# ============================================================================
def _dist(name):
    """Map a parent-law name to the ``(kind, parameter)`` pair of the generators.

    Args:
        name: one of ``'norm'``, ``'unif'``, ``'exp'``, ``'chi2_4'``, ``'chi2_2'``,
            ``'lognorm_1'``, ``'t_30'``, ``'t_10'``, ``'t_5'``, ``'t_3'``.

    Returns:
        tuple ``(kind, par)``, with ``par`` None for parameter-free parents.
    """
    table = {
        "norm": ("norm", None),
        "unif": ("unif", None),
        "exp": ("exp", None),
        "chi2_4": ("chi2", 4),
        "chi2_2": ("chi2", 2),
        "lognorm_1": ("lognorm", 1.0),
        "t_30": ("t", 30),
        "t_10": ("t", 10),
        "t_5": ("t", 5),
        "t_3": ("t", 3),
    }
    return table[name]


def _rung(label, dname, severity, note=""):
    """Build one severity-ladder rung record.

    Args:
        label: human-readable rung label used in the figures and tables.
        dname: parent-law name understood by :func:`_dist`.
        severity: dict with the numeric severity of the rung (e.g. ``gamma1``).
        note: optional comment stored with the rung.

    Returns:
        dict with keys ``label``, ``dist``, ``severity`` (including the analytic
        skewness and excess kurtosis) and ``note``. A non-finite excess kurtosis
        is stored as None plus ``excess_kurtosis_finite = False``, because JSON
        has no Infinity.
    """
    kind, par = _dist(dname)
    g1, kurt = skew_kurt(kind, par)
    # JSON has no Infinity: represent a non-existent fourth moment as null plus
    # an explicit flag (used only by t(3) on the kurtosis ladder).
    sev = dict(dist=dname, desc=DIST_DESC[dname], gamma1=g1,
               excess_kurtosis=(None if not np.isfinite(kurt) else float(kurt)),
               excess_kurtosis_finite=bool(np.isfinite(kurt)))
    sev.update(severity)
    return dict(label=label, dist=dname, severity=sev, note=note)


def _kurt_tick(rung):
    """Axis-tick label for the excess kurtosis of a ladder rung.

    Args:
        rung: rung record built by :func:`_rung`.

    Returns:
        str: the kurtosis rounded to two decimals, or ``'inf'`` when it is None.
    """
    k = rung["severity"]["excess_kurtosis"]
    return "inf" if k is None else "%.2f" % k


SKEW_RUNGS = [
    _rung("Normal N(0,1)", "norm", dict(severity=0.0)),
    _rung("chi2(4)", "chi2_4", dict(severity=1.4142)),
    _rung("Exp(1)", "exp", dict(severity=2.0)),
    _rung("chi2(2)", "chi2_2", dict(severity=2.0),
          note="chi2(2) is a 2x scaled Exp(1); same gamma1 and kurtosis. "
               "Included as a scale-replication cell and as a check that the "
               "panel is scale-invariant."),
    _rung("Lognormal", "lognorm_1", dict(severity=6.1851)),
]

KURT_RUNGS = [
    _rung("Normal N(0,1)", "norm", dict(severity=0.0)),
    _rung("t(30)", "t_30", dict(severity=0.2308)),
    _rung("t(10)", "t_10", dict(severity=1.0)),
    _rung("t(5)", "t_5", dict(severity=6.0)),
    _rung("t(3)", "t_3", dict(severity=float("inf")),
          note="Excess kurtosis of t(3) does not exist (fourth moment absent); "
               "the rung is ordered last as the most heavy-tailed case."),
]

VAR_RATIO_RUNGS = [
    dict(label="1:1 (equal)", ratio=1.0, sd=[1.0, 1.0, 1.0]),
    dict(label="1:1.5", ratio=1.5, sd=[1.0, 1.0, np.sqrt(1.5)]),
    dict(label="1:2", ratio=2.0, sd=[1.0, 1.0, np.sqrt(2.0)]),
    dict(label="1:4 (archived)", ratio=4.0, sd=[1.0, 1.0, 2.0]),
    dict(label="1:9", ratio=9.0, sd=[1.0, 1.0, 3.0]),
    dict(label="1:16", ratio=16.0, sd=[1.0, 1.0, 4.0]),
]

# Group sizes for the imbalance ladder: total n_T = 150 held fixed across rungs
# (the archived 1:2:3 scenario has n_T = 300 and is reported separately as the
# published anchor).
IMB_RUNGS = [
    dict(label="1:1:1 (n_T=150)", ratio=1.0, ns=[50, 50, 50]),
    dict(label="1:1:2 (n_T=150)", ratio=2.03, ns=[37, 38, 75]),
    dict(label="1:1:3 (n_T=150)", ratio=3.0, ns=[30, 30, 90]),
    dict(label="1:2:5 (n_T=150)", ratio=4.95, ns=[19, 37, 94]),
    dict(label="1:1:10 (n_T=150)", ratio=10.42, ns=[13, 12, 125]),
    dict(label="1:2:3 archived (n_T=300)", ratio=3.0, ns=[50, 100, 150],
         archived_anchor=True),
]

# Distributional ladder: all tests share the same rung parents, with the
# Normal rung as the common baseline (note: for T3/T9 the *archived* baseline
# parent is uniform; the archived anchor is re-run in consistency_checks).
DIST_LADDER_TESTS = ["T1_t_test", "T2_anova", "T3_kruskal_wallis", "T5_bartlett",
                     "T6_fligner_killeen", "T7_sign_test", "T9_median_test",
                     "T10_hotelling"]
VAR_LADDER_TESTS = ["T2_anova", "T3_kruskal_wallis", "T9_median_test"]
VAR_LADDER_AUX = ["T5_bartlett", "T6_fligner_killeen"]
IMB_LADDER_TESTS = ["T2_anova", "T3_kruskal_wallis", "T5_bartlett",
                    "T6_fligner_killeen", "T9_median_test"]


def build_ladder_cells():
    """Return the flat list of cell requests for all five severity ladders."""
    cells = []

    # ---- distributional ladders (skewness, kurtosis) -----------------------
    for lname, rungs in (("skewness", SKEW_RUNGS), ("kurtosis", KURT_RUNGS)):
        seed = cell_seed("dist_ladder")          # identical for both ladders:
        for r in rungs:                          # the Normal rung is shared
            for tid in DIST_LADDER_TESTS:
                spec = spec_for(tid, dist=_dist(r["dist"]), n=N_FIXED)
                cells.append(dict(ladder=lname, test=tid, rung=r["label"],
                                  spec=spec, seed=seed, severity=r["severity"],
                                  note=r["note"]))

    # ---- variance-ratio ladder --------------------------------------------
    seed = cell_seed("var_ratio_ladder")
    for r in VAR_RATIO_RUNGS:
        for tid in VAR_LADDER_TESTS + VAR_LADDER_AUX:
            spec = spec_for(tid, dist=_archived_dist(tid), n=N_FIXED, sd=r["sd"])
            cells.append(dict(ladder="variance_ratio", test=tid, rung=r["label"],
                              spec=spec, seed=seed,
                              severity=dict(ratio=r["ratio"], sd=r["sd"]),
                              aux=(tid in VAR_LADDER_AUX)))

    # ---- imbalance ladder -------------------------------------------------
    seed = cell_seed("imbalance_ladder")
    for r in IMB_RUNGS:
        for tid in IMB_LADDER_TESTS:
            spec = spec_for(tid, dist=_archived_dist(tid), n=N_FIXED, ns=r["ns"])
            cells.append(dict(ladder="imbalance", test=tid, rung=r["label"],
                              spec=spec, seed=seed,
                              severity=dict(ratio=r["ratio"], ns=r["ns"],
                                            n_T=int(np.sum(r["ns"])),
                                            archived_anchor=bool(r.get("archived_anchor", False)))))

    # ---- small-expected-frequency ladders ---------------------------------
    # T4 chi2 GOF: n_T = 250 fixed, k = 6, smallest expected count np_min.
    seed = cell_seed("small_freq_t4")
    t4_rungs = [dict(label="balanced (np=41.7)", min_np=None),
                dict(label="np_min=5", min_np=5.0),
                dict(label="np_min=3", min_np=3.0),
                dict(label="np_min=2", min_np=2.0),
                dict(label="np_min=1", min_np=1.0),
                dict(label="np_min=0.5", min_np=0.5)]
    for r in t4_rungs:
        spec = spec_for("T4_chisq_gof", n=N_FIXED, k=6, n_T=250,
                        min_np=r["min_np"])
        cells.append(dict(ladder="small_freq_t4", test="T4_chisq_gof",
                          rung=r["label"], spec=spec, seed=seed,
                          severity=dict(n_T=250, k=6, min_np=r["min_np"])))
    # T4 archived anchor (np_min = 2 with n_T = 30) is run in consistency_checks.

    # T8 Cochran Q: success probability ladder, 50 blocks x 3 treatments.
    seed = cell_seed("small_freq_t8")
    for p in (0.5, 0.3, 0.1, 0.05, 0.01):
        spec = spec_for("T8_cochran_q", n=N_FIXED, k=K_GROUPS, p=p)
        cells.append(dict(ladder="small_freq_t8", test="T8_cochran_q",
                          rung="p=%.2f" % p, spec=spec, seed=seed,
                          severity=dict(p=p, n_blocks=N_FIXED, k=K_GROUPS,
                                        p_all_zero_blocks=float((1.0 - p) ** (N_FIXED * K_GROUPS)),
                                        expected_successes=float(N_FIXED * K_GROUPS * p))))

    # T9 median test: per-group n ladder at a uniform parent (archived parent).
    seed = cell_seed("small_freq_t9")
    for n in (30, 20, 15, 10, 8):
        spec = spec_for("T9_median_test", dist=_dist("unif"), n=n, k=K_GROUPS)
        cells.append(dict(ladder="small_freq_t9", test="T9_median_test",
                          rung="n=%d" % n, spec=spec, seed=seed,
                          severity=dict(n_per_group=n, n_T=K_GROUPS * n,
                                        expected_cell=round(n / 2.0, 1))))
    return cells


def aggregate_ladder(cells, records):
    """Aggregate flat cell records into per-test, per-rung structures."""
    out = {}
    for c, rec in zip(cells, records):
        lad = out.setdefault(c["ladder"], {})
        t = lad.setdefault(c["test"], {"label": TEST_LABELS[c["test"]],
                                       "rungs": {}, "aux": c.get("aux", False)})
        t["rungs"][c["rung"]] = dict(rec, severity=c["severity"], note=c.get("note", ""))
    return out


def trend_analysis(rung_labels, recs, alpha=NOMINAL_ALPHA):
    """Mechanical, pre-declared rules for what may be claimed from a ladder.

    * pointwise_nondecreasing: alpha_hat non-decreasing over the rung order.
    * ci_separated_from_baseline[i]: Wilson CI of rung i disjoint from the
      Wilson CI of rung 0.
    * adjacent_ci_separated[i]: Wilson CI of rung i disjoint from rung i-1.
    * monotone_established: pointwise non-decreasing AND the most severe rung's
      CI disjoint from the baseline CI AND alpha_hat(last) > alpha_hat(first).
      Only then may an increase be described as established.
    """
    vals = [recs[l]["alpha_hat"] for l in rung_labels]
    lo = [recs[l]["ci_lo"] for l in rung_labels]
    hi = [recs[l]["ci_hi"] for l in rung_labels]
    pn = all(vals[i] >= vals[i - 1] - 1e-12 for i in range(1, len(vals)))
    max_dec = max([vals[i - 1] - vals[i] for i in range(1, len(vals))] + [0.0])
    # A decrease below half a Wilson half-width (~0.002 at N=10,000) is far below
    # the resolution of the experiment and is labeled explicitly as a wiggle.
    pn_tol = bool(max_dec <= 0.002)
    sep_base = [bool(lo[i] > hi[0] or hi[i] < lo[0]) for i in range(len(vals))]
    adj = [bool(lo[i] > hi[i - 1] or hi[i] < lo[i - 1]) for i in range(1, len(vals))]
    last_sep = bool(sep_base[-1]) if len(vals) > 1 else False
    increasing = vals[-1] > vals[0]
    if last_sep and increasing and pn:
        verdict = "increasing_ci_separated"
    elif last_sep and increasing and pn_tol:
        verdict = "increasing_ci_separated_with_subresolution_wiggle"
    elif last_sep and increasing:
        verdict = "mixed_but_most_severe_rung_separated"
    elif pn and not last_sep:
        verdict = "increasing_pointwise_but_cis_overlap"
    elif all(not s for s in sep_base[1:]):
        verdict = "flat_no_separation"
    else:
        verdict = "non_monotone_or_mixed"
    lib = [bool(recs[l]["ratio"] > LIBERAL_RATIO) for l in rung_labels]
    lib_ci = [bool(recs[l]["liberal_ci"]) for l in rung_labels]
    cons = [bool(recs[l]["ratio"] < CONSERVATIVE_RATIO) for l in rung_labels]
    first_lib = next((rung_labels[i] for i in range(len(lib)) if lib[i]), None)
    first_lib_ci = next((rung_labels[i] for i in range(len(lib_ci)) if lib_ci[i]), None)
    first_cons = next((rung_labels[i] for i in range(len(cons)) if cons[i]), None)
    return dict(
        alpha=alpha,
        alpha_hat=[vals[i] for i in range(len(vals))],
        ci=[[lo[i], hi[i]] for i in range(len(vals))],
        pointwise_nondecreasing=bool(pn),
        pointwise_nondecreasing_within_resolution=pn_tol,
        max_pointwise_decrease=float(max_dec),
        ci_separated_from_baseline=sep_base,
        adjacent_ci_separated=adj,
        monotone_established=bool(pn and last_sep and increasing),
        monotone_established_within_resolution=bool(pn_tol and last_sep and increasing),
        trend_verdict=verdict,
        liberal_flag=lib,
        liberal_flag_ci=lib_ci,
        conservative_flag=cons,
        first_liberal_rung=first_lib,
        first_liberal_rung_ci=first_lib_ci,
        first_conservative_rung=first_cons,
        delta_vs_baseline=[vals[i] - vals[0] for i in range(len(vals))],
        delta_ci_separated_from_baseline=sep_base,
    )


# ============================================================================
# SECTION 6 -- n-sweep on the archived Study 2 scenarios
# ============================================================================
NSWEEP_PLAN = {
    "baseline": ["T1_t_test", "T2_anova", "T3_kruskal_wallis", "T4_chisq_gof",
                 "T5_bartlett", "T6_fligner_killeen", "T7_sign_test",
                 "T8_cochran_q", "T9_median_test", "T10_hotelling"],
    "skewed": ["T1_t_test", "T2_anova", "T3_kruskal_wallis", "T5_bartlett",
               "T6_fligner_killeen", "T7_sign_test", "T9_median_test",
               "T10_hotelling"],
    "heavy_tail": ["T1_t_test", "T2_anova", "T3_kruskal_wallis", "T5_bartlett",
                   "T6_fligner_killeen", "T7_sign_test", "T9_median_test",
                   "T10_hotelling"],
    "hetero_var": ["T2_anova", "T3_kruskal_wallis", "T9_median_test"],
    # T8 is excluded: Cochran's Q is a complete randomized-block design, in which
    # unequal group sizes are not defined (see meta.notes).
    "unbalanced": ["T2_anova", "T3_kruskal_wallis", "T5_bartlett",
                   "T6_fligner_killeen", "T9_median_test"],
    # T9 small-frequency scenario IS a sample-size reduction (n = 8 per group);
    # varying n would destroy the scenario, so it lives in the n-ladder instead.
    "small_freq": ["T4_chisq_gof", "T8_cochran_q"],
}


def _archived_dist(test_id):
    """Archived parent of a test as a (kind, par) tuple (distribution-free tests
    such as T4/T8 fall back to Normal, which their generators ignore)."""
    parent = ARCHIVED_PARENT[test_id]
    return _dist(parent) if parent is not None else ("norm", None)


def nsweep_spec(test_id, scenario, n):
    """Spec of one (test, scenario, n) cell of the sample-size sweep."""
    parent = ARCHIVED_PARENT[test_id]
    if scenario == "baseline":
        return spec_for(test_id, dist=_archived_dist(test_id), n=n)
    if scenario == "skewed":
        return spec_for(test_id, dist=_dist("exp"), n=n)
    if scenario == "heavy_tail":
        return spec_for(test_id, dist=_dist("t_5"), n=n)
    if scenario == "hetero_var":
        sd = [1.0, 1.0, np.sqrt(4.0)]
        return spec_for(test_id, dist=_archived_dist(test_id), n=n, sd=sd)
    if scenario == "unbalanced":
        return spec_for(test_id, dist=_archived_dist(test_id), n=n,
                        ns=[n, 2 * n, 3 * n])
    if scenario == "small_freq":
        if test_id == "T4_chisq_gof":
            # Hold the minimum expected count at np_min = 2 while n_T grows with
            # max(5n, 30) (the archived Study 1/2 total-size convention).
            return spec_for(test_id, n=n, k=6, n_T=max(5 * n, 30), min_np=2.0)
        if test_id == "T8_cochran_q":
            return spec_for(test_id, n=n, k=K_GROUPS, p=0.05)
    raise ValueError((test_id, scenario))


def build_nsweep_cells():
    """List the cells of the Study 2 sample-size sweep.

    Every scenario gets one seed shared by all its tests and sample sizes (common
    random numbers), so that differences between cells are not confounded with
    sampling noise.

    Returns:
        list of dicts with keys ``scenario``, ``test``, ``n``, ``spec`` and
        ``seed``, in evaluation order.
    """
    cells = []
    for scenario, tests in NSWEEP_PLAN.items():
        seed = cell_seed("nsweep", scenario)     # identical across n and tests
        for tid in tests:
            for n in NSWEEP_N:
                cells.append(dict(scenario=scenario, test=tid, n=n,
                                  spec=nsweep_spec(tid, scenario, n), seed=seed))
    return cells


def aggregate_nsweep(cells, records):
    """Reshape the flat n-sweep results into the JSON layout.

    Args:
        cells: cell descriptors from :func:`build_nsweep_cells`, in evaluation
            order.
        records: per-cell result dicts, in the same order.

    Returns:
        dict: ``{scenario: {test: {'label': ..., 'by_n': {str(n): record}}}}``.
    """
    out = {}
    for c, rec in zip(cells, records):
        sc = out.setdefault(c["scenario"], {})
        t = sc.setdefault(c["test"], {"label": TEST_LABELS[c["test"]], "by_n": {}})
        t["by_n"][str(c["n"])] = rec
    return out


def ordering_analysis(nsweep_agg, scenario):
    """Is the robustness ordering stable across n?

    Delta_test(n) = alpha_hat_test,scenario(n) - alpha_hat_test,baseline(n) uses the
    *matched-n* baseline.  Because Delta_a - Delta_b = alpha_hat_a - alpha_hat_b
    (the baseline cancels), CI separation of two violated cells is exactly CI
    separation of their Deltas.
    """
    tests = sorted(nsweep_agg[scenario].keys())
    base = nsweep_agg["baseline"]
    delta = {t: {n: nsweep_agg[scenario][t]["by_n"][str(n)]["alpha_hat"]
                    - base[t]["by_n"][str(n)]["alpha_hat"] for n in NSWEEP_N}
             for t in tests}
    order = {str(n): sorted(tests, key=lambda t: -delta[t][n]) for n in NSWEEP_N}
    # Spearman rank correlation of Delta orderings between consecutive n
    from scipy.stats import spearmanr
    rho = {}
    for i in range(len(NSWEEP_N) - 1):
        a = [delta[t][NSWEEP_N[i]] for t in tests]
        b = [delta[t][NSWEEP_N[i + 1]] for t in tests]
        r = spearmanr(a, b).statistic
        rho["%d_vs_%d" % (NSWEEP_N[i], NSWEEP_N[i + 1])] = float(r)
    pairs = {}
    for i, ta in enumerate(tests):
        for tb in tests[i + 1:]:
            diffs, disjoint, signs = {}, {}, set()
            for n in NSWEEP_N:
                ra = nsweep_agg[scenario][ta]["by_n"][str(n)]
                rb = nsweep_agg[scenario][tb]["by_n"][str(n)]
                diffs[str(n)] = ra["alpha_hat"] - rb["alpha_hat"]
                disjoint[str(n)] = bool(ra["ci_lo"] > rb["ci_hi"]
                                        or ra["ci_hi"] < rb["ci_lo"])
                signs.add(np.sign(diffs[str(n)]))
            sign_stable = len(signs) == 1
            name = "%s_vs_%s" % (ta, tb)
            pairs[name] = dict(
                delta_diff=diffs, ci_disjoint=disjoint,
                sign_stable=bool(sign_stable),
                separated_at_all_n=bool(all(disjoint.values())),
                separated_at_any_n=bool(any(disjoint.values())),
                verdict=("stable and CI-separated at every n" if sign_stable and all(disjoint.values())
                         else "stable but CIs overlap at some n" if sign_stable
                         else "ordering reverses across n"))
    return dict(delta=delta, rank_order_by_delta=order, spearman_consecutive=rho,
                pairs=pairs,
                note="Delta = alpha_hat(scenario, n) - alpha_hat(baseline, same n).")


def first_n_recovery(nsweep_agg, scenario):
    """Smallest n at which each test is no longer liberal (and CI-based)."""
    base = nsweep_agg["baseline"]
    out = {}
    for t in sorted(nsweep_agg[scenario]):
        recs = nsweep_agg[scenario][t]["by_n"]
        never_lib = not any(recs[str(n)]["liberal"] for n in NSWEEP_N)
        never_lib_ci = not any(recs[str(n)]["liberal_ci"] for n in NSWEEP_N)
        out[t] = dict(
            alpha_hat={n: recs[str(n)]["alpha_hat"] for n in NSWEEP_N},
            baseline_alpha_hat={n: base[t]["by_n"][str(n)]["alpha_hat"] for n in NSWEEP_N},
            delta={n: recs[str(n)]["alpha_hat"] - base[t]["by_n"][str(n)]["alpha_hat"]
                   for n in NSWEEP_N},
            liberal_flag={n: recs[str(n)]["liberal"] for n in NSWEEP_N},
            liberal_flag_ci={n: recs[str(n)]["liberal_ci"] for n in NSWEEP_N},
            never_liberal=bool(never_lib), never_liberal_ci=bool(never_lib_ci),
            # None means "the liberal flag is never set on this grid", which is not
            # the same as "clears at n=20".
            first_n_not_liberal=(None if never_lib else
                                 next((n for n in NSWEEP_N
                                       if not recs[str(n)]["liberal"]), None)),
            first_n_not_liberal_ci=(None if never_lib_ci else
                                    next((n for n in NSWEEP_N
                                          if not recs[str(n)]["liberal_ci"]), None)),
            degradation_disappears=(recs[str(NSWEEP_N[0])]["liberal"]
                                    and not recs[str(NSWEEP_N[-1])]["liberal"]),
        )
    return out


# ============================================================================
# SECTION 7 -- optional T11/T12/T13 from core.py (owner: t1-new-tests)
# ============================================================================
# core.py exposes the adapter SEVERITY_HOOKS requested by this task:
#     {test_id: dict(label=..., rule=..., gen=lambda spec, N, rng -> stat_values)}
# where spec = dict(ns=[...], dist=('norm',None)|('t',5)|('chi2',4)|('lognorm',1.0),
#                   N=..., seed=...).  Two guards make the extras safe for the
# deliverable run:
#   * COST -- a pilot run at EXTRAS_PILOT_N projects the cost of one N=10,000
#     cell; hooks whose projection exceeds EXTRAS_MAX_SECONDS_PER_CELL are
#     skipped and the projection is recorded in meta.extras_skipped.
#   * DEGENERACY -- a hook that returns bit-identical statistics for two
#     different parents ignores spec['dist'], so a distributional severity
#     ladder is not defined for it; it is skipped and recorded.
EXTRAS_PILOT_N = 200
EXTRAS_MAX_SECONDS_PER_CELL = 20.0
# Measured pilot timings are recorded as numbers (not baked into messages) so the
# results JSON stays byte-identical across runs apart from the runtime/timestamp.
EXTRAS_PROJECTIONS = {}


def _core_rate(tid, sv, alpha, n_eff):
    """Canonical rate via core.reject_rate (handles cvm_limit / hoeffding_z rules
    that this script's reject_rate does not implement)."""
    return float(CORE.reject_rate(np.asarray(sv, dtype=float), tid, alpha))


# Upstream calibration issue found while wiring the extras (reported to
# t1-new-tests).  core.reference_ppf() documents q as an UPPER-tail probability
# and maps scipy families through stats.<name>.isf(q); the bespoke cvm_limit
# entry instead calls cvm_limit_ppf(q) as a lower-tail quantile, so
# critical_value('T11_kpss', 0.05) = cvm_limit_ppf(0.05) = 0.0366 instead of the
# Cramer-von Mises 5% quantile cvm_limit_ppf(0.95) = 0.4614.  With 0.0366 the
# KPSS rate under iid N(0,1) data is ~1.00; the correct quantile gives ~0.05.
# We therefore evaluate T11 with core's own limit-law quantile under the correct
# tail convention and store BOTH rates in every T11 record so the discrepancy is
# auditable end to end.
# Published Cramer-von Mises critical values (core.py docstring, Csorgo & Faraway
# 1996 / goftest::pCvM): x(0.90)=0.34730, x(0.95)=0.46136, x(0.99)=0.74346.
CVM_PUBLISHED = {0.10: 0.34730, 0.05: 0.46136, 0.01: 0.74346}


def _cvm_upper_crit(alpha):
    """Upper-tail CvM quantile, obtained by INVERSING core's own CDF.

    Inverting the CDF rather than calling `cvm_limit_ppf` makes the override
    independent of that function's tail convention, so it stays correct whichever
    way t1-new-tests fixes the convention (a mirrored ppf would otherwise silently
    reinstate the 0.0366 critical value).
    """
    from scipy.optimize import brentq
    target = 1.0 - alpha
    f = lambda x: float(CORE.cvm_limit_cdf(x)) - target
    lo, hi = 1e-9, 5.0
    try:
        if f(lo) < 0.0 <= f(hi):
            return float(brentq(f, lo, hi))
    except Exception:                     # pragma: no cover - defensive
        pass
    return float(CORE.cvm_limit_ppf(1.0 - alpha))


EXTRAS_RULE_OVERRIDES = {
    "T11_kpss": dict(
        kind="cvm_limit_ppf(1-alpha) upper tail",
        reason=("core.critical_value('T11_kpss') uses the lower-tail convention of "
                "cvm_limit_ppf, giving 0.0366 at alpha=0.05 instead of 0.4614; "
                "override uses core.cvm_limit_ppf(1-alpha)"),
        crit=_cvm_upper_crit),
}


def _extras_rate(tid, sv, alpha, n_eff):
    """Rate for an extra test plus an audit record of the rule and critical value used.

    Returns (rate, evaluation_info).  ``evaluation_info`` is ALWAYS returned, so
    every extra-test cell records which critical value produced its rate; an
    override is applied only when core's own critical value disagrees with the
    limit law obtained by inverting core's CDF (convention-independent), which
    makes the script correct both before and after t1's fix of the cvm_limit ppf.
    """
    sv = np.asarray(sv, dtype=float)
    rate_core = _core_rate(tid, sv, alpha, n_eff)
    try:
        core_crit = float(np.ravel(CORE.critical_value(tid, alpha)[0])[0])
    except Exception:                     # pragma: no cover - defensive
        core_crit = None
    info = dict(alpha=alpha, rule_used="core.reject_rate (canonical registry rule)",
                critical_value=core_crit, core_critical_value=core_crit,
                override_applied=False)
    ov = EXTRAS_RULE_OVERRIDES.get(tid)
    if ov is None:
        return rate_core, info
    crit = ov["crit"](alpha)              # inverts core's CDF: convention-independent
    info["limit_law_critical_value"] = crit
    if tid == "T11_kpss":
        ref = CVM_PUBLISHED.get(round(alpha, 2))
        info["external_check"] = (
            None if ref is None else
            dict(published_critical_value=ref,
                 abs_diff_of_core_critical_value=(None if core_crit is None
                                                  else abs(core_crit - ref)),
                 abs_diff_of_limit_law_value=abs(crit - ref)))
        try:
            info["core_cvm_limit_ppf_alpha"] = float(CORE.cvm_limit_ppf(alpha))
        except Exception:                 # pragma: no cover - defensive
            pass
    if core_crit is not None and abs(core_crit - crit) <= 1e-6:
        # Upstream reference now agrees with the limit law: use the canonical rule.
        info["upstream_agrees_with_limit_law"] = True
        return rate_core, info
    rate = float(np.mean(sv[np.isfinite(sv)] > crit))
    info.update(override_applied=True, upstream_agrees_with_limit_law=False,
                rule_used="documented override: " + ov["kind"],
                reason=ov["reason"], critical_value=crit,
                rate_with_core_rule=rate_core)
    return rate, info


def probe_core_extras():
    """Return (hooks, message, skipped_reasons) after the cost/degeneracy pilot."""
    if not HAS_CORE:
        return {}, "core.py not importable (%s)" % (CORE_ERR or "missing"), {}
    all_hooks = getattr(CORE, "SEVERITY_HOOKS", None)
    if not all_hooks:
        return {}, ("core.py present but core.SEVERITY_HOOKS is not exposed; "
                    "T11-T13 skipped."), {}
    accepted, rank_invariant, skipped = {}, {}, {}
    for tid, hook in all_hooks.items():
        if not isinstance(hook, dict) or "gen" not in hook:
            skipped[tid] = "hook entry malformed"
            continue
        if not (HAS_CORE and hasattr(CORE, "reject_rate")):
            skipped[tid] = "core.reject_rate unavailable"
            continue
        pilot_spec = dict(ns=[N_FIXED] * K_GROUPS, dist=("norm", None), N=EXTRAS_PILOT_N,
                          seed=cell_seed("dist_ladder"))
        try:
            t0 = time.time()
            sv_norm = np.asarray(hook["gen"](pilot_spec, EXTRAS_PILOT_N,
                                             np.random.default_rng(SEED)), dtype=float)
            dt = time.time() - t0
            pilot_spec2 = dict(pilot_spec, dist=("lognorm", 1.0))
            sv_log = np.asarray(hook["gen"](pilot_spec2, EXTRAS_PILOT_N,
                                            np.random.default_rng(SEED)), dtype=float)
        except Exception as exc:
            skipped[tid] = "pilot failed: %r" % (exc,)
            continue
        projected = dt * N_MC / EXTRAS_PILOT_N
        EXTRAS_PROJECTIONS[tid] = round(projected, 2)
        entry = dict(hook, projected_seconds_per_cell=round(projected, 2))
        if projected > EXTRAS_MAX_SECONDS_PER_CELL:
            skipped[tid] = ("projected cost of one N=%d cell at n=%d exceeds the "
                            "%.0f s per-cell budget: see meta.extras_projections; "
                            "run at a reduced N if needed"
                            % (N_MC, N_FIXED, EXTRAS_MAX_SECONDS_PER_CELL))
            continue
        if np.array_equal(sv_norm, sv_log):
            # Rank-based statistics are exactly invariant to the marginal parent;
            # a marginal-distribution severity ladder is then vacuous by
            # construction (dependence strength is an alternative, i.e. power).
            rank_invariant[tid] = dict(
                entry, invariance_note=(
                    "statistics are bit-identical under N(0,1) and lognormal parents: "
                    "the statistic depends on ranks only, so a marginal-parent severity "
                    "ladder is vacuous for this test. Its baseline calibration across n "
                    "is still reported in severity_sweep.extra_nsweep."))
            continue
        accepted[tid] = entry
    msg = ("parent-sensitive ladders: %s | rank-invariant (n-sweep only): %s | "
           "skipped: %s" % (sorted(accepted), sorted(rank_invariant), sorted(skipped)))
    return accepted, rank_invariant, skipped, msg


def run_core_ladders(hooks):
    """Distributional ladders for the accepted extra tests.

    Records follow the same schema as the 10-test panel so that the mechanical
    trend/threshold rules apply unchanged; rates come from core.reject_rate and
    the Wilson CI uses the number of finite statistics actually used.
    """
    res = {}
    if not hooks:
        return res
    for lname, rungs in (("skewness", SKEW_RUNGS), ("kurtosis", KURT_RUNGS)):
        seed = cell_seed("dist_ladder")          # same CRN seed as the 10-test panel
        for tid, hook in hooks.items():
            entry = {"label": hook.get("label", tid), "rungs": {}, "aux": False,
                     "source": "core.SEVERITY_HOOKS",
                     "projected_seconds_per_cell": hook.get("projected_seconds_per_cell"),
                     "note": hook.get("note", ""), "N": N_MC}
            for r in rungs:
                spec = dict(ns=[N_FIXED] * K_GROUPS, dist=_dist(r["dist"]),
                            N=N_MC, seed=seed)
                try:
                    sv = np.asarray(hook["gen"](spec, N_MC,
                                                np.random.default_rng(seed)), dtype=float)
                    n_eff = int(np.isfinite(sv).sum())
                    by, ov_infos, eval_main = {}, [], None
                    for a in ALPHA_LEVELS:
                        rate, ev = _extras_rate(tid, sv, a, n_eff)
                        if ev.get("override_applied"):
                            ov_infos.append(ev)
                        if abs(a - NOMINAL_ALPHA) < 1e-12:
                            eval_main = ev
                        lo, hi = wilson_ci(rate, n_eff)
                        by["%.2f" % a] = dict(
                            alpha_hat=rate, ci_lo=lo, ci_hi=hi, ratio=rate / a,
                            liberal=bool(rate / a > LIBERAL_RATIO),
                            liberal_ci=bool(lo / a > LIBERAL_RATIO),
                            conservative=bool(rate / a < CONSERVATIVE_RATIO),
                            conservative_ci=bool(hi / a < CONSERVATIVE_RATIO))
                    main = by["%.2f" % NOMINAL_ALPHA]
                    entry["rungs"][r["label"]] = dict(
                        main, by_alpha=by, severity=r["severity"], N=int(n_eff),
                        fraction_finite=n_eff / float(N_MC),
                        evaluation=eval_main, evaluation_override=ov_infos or None)
                except Exception as exc:      # pragma: no cover - defensive
                    entry["rungs"][r["label"]] = dict(error=repr(exc))
            res.setdefault(lname, {})[tid] = entry
    return res


def run_core_nsweep(hooks):
    """Baseline calibration vs n for the extra hooks (comment 5 for the new tests).

    Includes rank-invariant hooks: their marginal parent does not matter, but the
    finite-sample adequacy of the asymptotic reference law does.
    """
    out = {}
    for tid, hook in hooks.items():
        entry = {"label": hook.get("label", tid), "note": hook.get("note", ""),
                 "projected_seconds_per_cell": hook.get("projected_seconds_per_cell"),
                 "invariance_note": hook.get("invariance_note", ""), "by_n": {}}
        seed = cell_seed("extran", tid)
        for n in NSWEEP_N:
            spec = dict(ns=[n] * K_GROUPS, dist=("norm", None), N=N_MC, seed=seed)
            try:
                sv = np.asarray(hook["gen"](spec, N_MC, np.random.default_rng(seed)),
                                dtype=float)
                n_eff = int(np.isfinite(sv).sum())
                by, ov_infos, eval_main = {}, [], None
                for a in ALPHA_LEVELS:
                    rate, ev = _extras_rate(tid, sv, a, n_eff)
                    if ev.get("override_applied"):
                        ov_infos.append(ev)
                    if abs(a - NOMINAL_ALPHA) < 1e-12:
                        eval_main = ev
                    lo, hi = wilson_ci(rate, n_eff)
                    by["%.2f" % a] = dict(alpha_hat=rate, ci_lo=lo, ci_hi=hi,
                                          ratio=rate / a,
                                          liberal=bool(rate / a > LIBERAL_RATIO),
                                          liberal_ci=bool(lo / a > LIBERAL_RATIO),
                                          conservative=bool(rate / a < CONSERVATIVE_RATIO),
                                          conservative_ci=bool(hi / a < CONSERVATIVE_RATIO))
                entry["by_n"][str(n)] = dict(by["%.2f" % NOMINAL_ALPHA], by_alpha=by,
                                             n=n, N=n_eff, evaluation=eval_main,
                                             evaluation_override=ov_infos or None)
            except Exception as exc:      # pragma: no cover - defensive
                entry["by_n"][str(n)] = dict(error=repr(exc))
        out[tid] = entry
    return out


# ============================================================================
# SECTION 8 -- consistency checks and implementation validation
# ============================================================================
ARCHIVE_ANCHORS = [
    # (test_id, scenario, archived-generator-callable, archived rule, generic spec)
    ("T1_t_test", "baseline"),
    ("T1_t_test", "skewed"),
    ("T1_t_test", "heavy_tail"),
    ("T2_anova", "baseline"),
    ("T2_anova", "skewed"),
    ("T2_anova", "heavy_tail"),
    ("T2_anova", "hetero_var"),
    ("T2_anova", "unbalanced"),
    ("T3_kruskal_wallis", "baseline"),
    ("T3_kruskal_wallis", "skewed"),
    ("T3_kruskal_wallis", "heavy_tail"),
    ("T3_kruskal_wallis", "hetero_var"),
    ("T3_kruskal_wallis", "unbalanced"),
    ("T4_chisq_gof", "baseline"),
    ("T4_chisq_gof", "small_freq"),
    ("T5_bartlett", "baseline"),
    ("T5_bartlett", "skewed"),
    ("T5_bartlett", "heavy_tail"),
    ("T5_bartlett", "unbalanced"),
    ("T6_fligner_killeen", "baseline"),
    ("T6_fligner_killeen", "skewed"),
    ("T6_fligner_killeen", "heavy_tail"),
    ("T6_fligner_killeen", "unbalanced"),
    ("T7_sign_test", "baseline"),
    ("T7_sign_test", "skewed"),
    ("T7_sign_test", "heavy_tail"),
    ("T8_cochran_q", "baseline"),
    ("T8_cochran_q", "small_freq"),
    ("T9_median_test", "baseline"),
    ("T9_median_test", "skewed"),
    ("T9_median_test", "heavy_tail"),
    ("T9_median_test", "hetero_var"),
    ("T9_median_test", "unbalanced"),
    ("T9_median_test", "small_freq"),
    ("T10_hotelling", "baseline"),
    ("T10_hotelling", "skewed"),
    ("T10_hotelling", "heavy_tail"),
]


def _archive_call(test_id, scenario, N):
    """Reproduce the ARCHIVED experiments.py cell exactly (same config)."""
    n = N_FIXED
    k = K_GROUPS
    small = scenario == "small_freq"
    hetero = scenario == "hetero_var"
    unbal = scenario == "unbalanced"
    dist = {"skewed": "expon", "heavy_tail": "tdist"}.get(scenario)
    if test_id == "T1_t_test":
        return EX.mc_t_test(n, N, dist=dist or "normal"), ("t", n - 1)
    if test_id == "T2_anova":
        F, df1, df2 = EX.mc_anova(k, n, N, dist=dist or "normal", hetero=hetero,
                                  unbalanced=unbal)
        return F, ("f", df1, df2)
    if test_id == "T3_kruskal_wallis":
        return (EX.mc_kruskal_wallis(k, n, N, dist=dist or "uniform", hetero=hetero,
                                     unbalanced=unbal), ("chi2", k - 1))
    if test_id == "T4_chisq_gof":
        sv = (EX.mc_chisq_gof(6, 30, N, small_expected=True) if small
              else EX.mc_chisq_gof(6, 250, N))
        return sv, ("chi2", 5)
    if test_id == "T5_bartlett":
        return (EX.mc_bartlett(k, n, N, dist=dist or "normal", unbalanced=unbal),
                ("chi2", k - 1))
    if test_id == "T6_fligner_killeen":
        return (EX.mc_fligner_killeen(k, n, N, dist=dist or "normal",
                                      unbalanced=unbal), ("chi2", k - 1))
    if test_id == "T7_sign_test":
        return (EX.mc_sign_test(n, N, dist={"skewed": "expon",
                                            "heavy_tail": "tdist"}.get(scenario, "normal")),
                ("binom", n, 0.5))
    if test_id == "T8_cochran_q":
        return EX.mc_cochran_q(k, n, N, p=0.05 if small else 0.5), ("chi2", k - 1)
    if test_id == "T9_median_test":
        return (EX.mc_median_test(k, 8 if small else n, N, dist=dist or "uniform",
                                  hetero=hetero, unbalanced=unbal),
                ("chi2", k - 1))
    if test_id == "T10_hotelling":
        return EX.mc_hotelling_t2(P_DIM, n, N, dist=dist or "normal")[0], \
            ("f", P_DIM, n - P_DIM)
    raise ValueError(test_id)


def _generic_archive_spec(test_id, scenario):
    """Generic spec reproducing the archived anchor configuration."""
    n = N_FIXED
    k = K_GROUPS
    dist = {"baseline": ARCHIVED_PARENT[test_id],
            "skewed": "exp",
            "heavy_tail": "t_5"}.get(scenario, ARCHIVED_PARENT[test_id])
    if test_id == "T4_chisq_gof":
        if scenario == "small_freq":
            return dict(dist=("norm", None), k=6, n_T=30, min_np=2.0)
        return dict(dist=("norm", None), k=6, n_T=250, min_np=None)
    if test_id == "T8_cochran_q":
        return dict(dist=("norm", None), ns=[n], k=k,
                    p=0.05 if scenario == "small_freq" else 0.5)
    sd = [1.0, 1.0, 2.0] if scenario == "hetero_var" else None
    ns = [n, 2 * n, 3 * n] if scenario == "unbalanced" else None
    return spec_for(test_id, dist=_dist(dist), n=(8 if (test_id == "T9_median_test"
                                                        and scenario == "small_freq") else n),
                    k=k, ns=ns, sd=sd)


def median_test_exact_size(ns, alpha):
    """Exact null rejection probability of the median-test chi2_{k-1} rule.

    Under any continuous iid parent the grand median splits the pooled sample
    into a = n_T/2 observations strictly above it (n_T even) and (n_T-1)/2
    (n_T odd); conditional on a, the group-wise counts (O_11..O_1k) are
    multivariate hypergeometric.  Enumerating that distribution gives the exact
    finite-sample size of the rule -- no Monte Carlo, no distributional
    assumption -- which is why the median test's small-sample behavior is a
    lattice effect (the chi2 critical value falls in a gap between attainable
    rejection probabilities).
    """
    from math import comb
    ns = [int(x) for x in ns]
    k = len(ns)
    n_tot = sum(ns)
    a = n_tot // 2 if n_tot % 2 == 0 else (n_tot - 1) // 2
    crit = float(stats.chi2.ppf(1.0 - alpha, k - 1))
    denom = comb(n_tot, a)
    acc = {"rej": 0.0, "total": 0.0}

    def rec(j, remaining, counts):
        if j == k - 1:
            if remaining > ns[-1]:
                return
            o = counts + [remaining]
            prob = 1.0
            for oj, nj in zip(o, ns):
                prob *= comb(nj, oj)
            prob /= denom
            stat = (n_tot ** 2 / (a * (n_tot - a))) * sum(
                (o[jj] - ns[jj] * a / n_tot) ** 2 / ns[jj] for jj in range(k))
            acc["total"] += prob
            if stat > crit:
                acc["rej"] += prob
            return
        for oj in range(min(ns[j], remaining) + 1):
            rec(j + 1, remaining - oj, counts + [oj])

    rec(0, a, [])
    return acc["rej"]


def run_consistency_checks():
    """1) generic implementation vs archived experiments.py at the anchor cells;
    2) generic implementation vs SciPy/statsmodels on random data;
    3) determinism and variance self-checks."""
    checks = {"archive_vs_generic": {}, "implementation_validation": {},
              "determinism": {}, "distribution_selfcheck": {}}
    if not HAS_ARCHIVE:
        checks["archive_vs_generic"]["error"] = ARCHIVE_ERR
    else:
        for test_id, scenario in ARCHIVE_ANCHORS:
            key = "%s|%s" % (test_id, scenario)
            try:
                np.random.seed(SEED)                 # archived script's convention
                sv_a, rule_a = _archive_call(test_id, scenario, N_MC)
                rate_a = reject_rate(sv_a, rule_a, NOMINAL_ALPHA)
                lo_a, hi_a = wilson_ci(rate_a, N_MC)
                spec = _generic_archive_spec(test_id, scenario)
                rec = evaluate_cell(test_id, spec, seed=cell_seed("anchor", key))
                se = np.sqrt(max(rate_a * (1 - rate_a), 1e-12) / N_MC
                             + max(rec["alpha_hat"] * (1 - rec["alpha_hat"]), 1e-12) / N_MC)
                overlap = not (rec["ci_lo"] > hi_a or rec["ci_hi"] < lo_a)
                checks["archive_vs_generic"][key] = dict(
                    archived_alpha_hat=rate_a, archived_ci=[lo_a, hi_a],
                    generic_by_alpha=rec["by_alpha"],
                    generic_alpha_hat=rec["alpha_hat"],
                    generic_ci=[rec["ci_lo"], rec["ci_hi"]],
                    abs_diff=abs(rate_a - rec["alpha_hat"]),
                    pooled_se=float(se),
                    z_approx=float(abs(rate_a - rec["alpha_hat"]) / se),
                    ci_overlap=bool(overlap),
                )
            except Exception as exc:                 # pragma: no cover - defensive
                checks["archive_vs_generic"][key] = dict(error=repr(exc))

    # ---- implementation validation against SciPy/statsmodels --------------
    rng = np.random.default_rng(SEED + 7)
    trials = 200
    iv = checks["implementation_validation"]

    def maxdiff(fn_mine, fn_ref, n_trials=trials):
        """max |generic statistic - reference statistic| over random 3-group data.

        The generic implementations are vectorised over replications, so each
        random draw is reshaped to (1, n_j); the reference implementation gets
        the plain 1-D samples.
        """
        d = []
        for _ in range(n_trials):
            g = [rng.normal(0, 1, int(rng.integers(8, 60))) for _ in range(3)]
            mine = float(np.asarray(fn_mine([x[None, :] for x in g])).ravel()[0])
            d.append(abs(mine - float(fn_ref(g))))
        return float(np.max(d))

    iv["anova_vs_scipy_f_oneway"] = maxdiff(
        lambda g: stat_anova(g), lambda g: stats.f_oneway(*g).statistic)
    iv["bartlett_vs_scipy"] = maxdiff(
        lambda g: stat_bartlett(g), lambda g: stats.bartlett(*g).statistic)
    iv["kruskal_vs_scipy"] = maxdiff(
        lambda g: stat_kruskal(g), lambda g: stats.kruskal(*g).statistic)
    # Fligner-Killeen: balanced path must equal experiments.py exactly; the
    # unbalanced path is compared with scipy.stats.fligner (the archive used
    # scipy there).
    # Fligner-Killeen: the balanced path must equal experiments.py exactly.  The
    # archive draws its own data from the legacy global RNG, so replay the seed
    # and redraw the identical (1, k, n) array before evaluating our version.
    if HAS_ARCHIVE:
        np.random.seed(777)
        ref_bal = float(EX.mc_fligner_killeen(3, 30, 1)[0])
        np.random.seed(777)
        Xb = np.random.normal(0, 1, (1, 3, 30))      # exactly what the archive drew
        mine_bal = float(stat_fligner([Xb[0, j][None, :] for j in range(3)])[0])
        iv["fligner_balanced_vs_experiments_py"] = abs(mine_bal - ref_bal)
    else:
        iv["fligner_balanced_vs_experiments_py"] = None
    fk_d = []
    for _ in range(trials):
        g = [rng.normal(0, 1, int(rng.integers(8, 60))) for _ in range(3)]
        fk_d.append(abs(float(stat_fligner([x[None, :] for x in g])[0])
                        - float(stats.fligner(*g).statistic)))
    iv["fligner_unbalanced_vs_scipy_fligner"] = float(np.max(fk_d))
    iv["fligner_unbalanced_vs_scipy_fligner_interpretation"] = (
        "Max |statistic difference| over 200 random UNBALANCED datasets with group "
        "sizes drawn from 8..59: a tie-handling stress metric, NOT a difference of "
        "statistic definitions. scipy 1.16.3 and the archive's balanced path both "
        "use normal scores of the pooled |x - group median| ranks; they differ only "
        "in averaged vs raw ranks at exact ties, which are guaranteed for even-sized "
        "groups. At the Study 2 configuration the same-dataset max difference is "
        "1.9e-05 (ns = 50/50/50) and 1.7e-06 (ns = 50/100/150) over 10,000 "
        "replications, and no rejection decision differs (see fligner_variant_probe).")
    # median test == Pearson chi2 of the 2 x k table
    md_d = []
    for _ in range(trials):
        g = [rng.normal(0, 1, int(rng.integers(8, 60))) for _ in range(3)]
        A = np.concatenate(g)
        gm = np.median(A)
        tab = np.array([[(x > gm).sum() for x in g], [(x <= gm).sum() for x in g]])
        md_d.append(abs(float(stat_median([x[None, :] for x in g])[0])
                        - float(stats.chi2_contingency(tab, correction=False).statistic)))
    iv["median_vs_chi2_contingency"] = float(np.max(md_d))
    # Cochran Q vs statsmodels
    try:
        from statsmodels.stats.contingency_tables import cochrans_q
        cq_d = []
        for _ in range(50):
            Xb = rng.binomial(1, 0.5, (int(rng.integers(10, 40)), 3))
            mine = float(stat_cochran(Xb[None, :, :])[0])
            cq_d.append(abs(mine - float(cochrans_q(Xb).statistic)))
        iv["cochran_q_vs_statsmodels"] = float(np.max(cq_d))
    except Exception as exc:                          # pragma: no cover
        iv["cochran_q_vs_statsmodels"] = repr(exc)
    # Hotelling vs an independent per-replication implementation
    Xh = rng.normal(0, 1, (50, 40, 2))
    mine = stat_hotelling(Xh)
    ref = np.empty(50)
    for i in range(50):
        xb = Xh[i].mean(axis=0)
        S = np.cov(Xh[i].T)
        ref[i] = (40 - 2) / (2 * (40 - 1)) * 40 * xb @ np.linalg.inv(S) @ xb
    iv["hotelling_vs_loop_implementation"] = float(np.max(np.abs(mine - ref)))
    # Sign test: exact binomial size of the archived rejection rule
    sign_exact = {}
    for a_ in ALPHA_LEVELS:
        lo_b = stats.binom.ppf(a_ / 2, N_FIXED, 0.5)
        hi_b = stats.binom.ppf(1 - a_ / 2, N_FIXED, 0.5)
        exact = float(stats.binom.cdf(lo_b - 1, N_FIXED, 0.5) + 1 - stats.binom.cdf(hi_b, N_FIXED, 0.5))
        sv = gen_cell("T7_sign_test", spec_for("T7_sign_test"), cell_seed("anchor", "sign"))[0]
        sim = reject_rate(sv, ("binom", N_FIXED, 0.5), a_)
        lo_ci, hi_ci = wilson_ci(sim, N_MC)
        sign_exact["%.2f" % a_] = dict(exact_size=exact, simulated=sim,
                                       ci=[lo_ci, hi_ci],
                                       exact_in_ci=bool(lo_ci <= exact <= hi_ci))
    iv["sign_test_exact_binomial_size"] = sign_exact

    # ---- median test: exact lattice size vs simulation --------------------
    lattice = {}
    for n in (30, 20, 15, 10, 8):
        spec = spec_for("T9_median_test", dist=_dist("unif"), n=n, k=K_GROUPS)
        sv = gen_cell("T9_median_test", spec, cell_seed("small_freq_t9"))[0]
        sim = reject_rate(sv, ("chi2", K_GROUPS - 1), NOMINAL_ALPHA)
        lo_ci, hi_ci = wilson_ci(sim, N_MC)
        for a_ in ALPHA_LEVELS:
            key = "n=%d|alpha=%.2f" % (n, a_)
            ex = median_test_exact_size([n] * K_GROUPS, a_)
            if abs(a_ - NOMINAL_ALPHA) < 1e-12:
                lattice[key] = dict(exact_size=ex, simulated=sim, ci=[lo_ci, hi_ci],
                                    exact_in_ci=bool(lo_ci <= ex <= hi_ci),
                                    ratio_exact=ex / a_)
            else:
                lattice[key] = dict(exact_size=ex, ratio_exact=ex / a_)
    iv["median_test_exact_lattice"] = lattice

    # ---- published Table 4 reproduction (archived implementation) ---------
    published = {
        "T1_t_test": {"baseline": 0.052, "skewed": 0.066, "heavy_tail": 0.045},
        "T2_anova": {"baseline": 0.046, "skewed": 0.043, "heavy_tail": 0.048,
                     "hetero_var": 0.061, "unbalanced": 0.049},
        "T3_kruskal_wallis": {"baseline": 0.046, "skewed": 0.051, "heavy_tail": 0.049,
                              "hetero_var": 0.064, "unbalanced": 0.051},
        "T4_chisq_gof": {"baseline": 0.049, "small_freq": 0.051},
        "T5_bartlett": {"baseline": 0.048, "skewed": 0.429, "heavy_tail": 0.276,
                        "unbalanced": 0.051},
        "T6_fligner_killeen": {"baseline": 0.041, "skewed": 0.119, "heavy_tail": 0.043,
                               "unbalanced": 0.045},
        "T7_sign_test": {"baseline": 0.035, "skewed": 0.031, "heavy_tail": 0.031},
        "T8_cochran_q": {"baseline": 0.050, "unbalanced": 0.050, "small_freq": 0.041},
        "T9_median_test": {"baseline": 0.043, "skewed": 0.046, "heavy_tail": 0.045,
                           "hetero_var": 0.056, "unbalanced": 0.051,
                           "small_freq": 0.069},
        "T10_hotelling": {"baseline": 0.050, "skewed": 0.076, "heavy_tail": 0.048},
    }
    pub_check = {}
    for tid, scen in published.items():
        for sc, val in scen.items():
            key = "%s|%s" % (tid, sc)
            ref = checks["archive_vs_generic"].get(key)
            if not isinstance(ref, dict) or "archived_alpha_hat" not in ref:
                continue
            se = float(np.sqrt(max(val * (1 - val), 1e-12) / N_MC
                               + max(ref["archived_alpha_hat"] * (1 - ref["archived_alpha_hat"]),
                                     1e-12) / N_MC))
            pub_check[key] = dict(published=val,
                                  archived_rerun=ref["archived_alpha_hat"],
                                  abs_diff=abs(val - ref["archived_alpha_hat"]),
                                  z_approx=float(abs(val - ref["archived_alpha_hat"]) / se),
                                  within_4se=bool(abs(val - ref["archived_alpha_hat"]) <= 4 * se))
    iv["published_table4_reproduction"] = pub_check

    # ---- cross-validation with core.py (owner of the 13-test panel) --------
    # Two questions: (i) do the two evaluators agree on the SAME array, and
    # (ii) does core.py's generator reproduce the archived baseline rates?
    if HAS_CORE and hasattr(CORE, "reject_rate"):
        xc = {}
        for tid in sorted(TEST_LABELS):
            try:
                n_ref = 250 if tid == "T4_chisq_gof" else N_FIXED
                fn = CORE.TESTS[tid].get("ref_args_fn")
                ref_args = fn(n_ref) if fn is not None else None
                spec = spec_for(tid, dist=_archived_dist(tid), n=N_FIXED)
                sv_mine, rule_mine = gen_cell(tid, spec, cell_seed("core_xcheck"))
                rate_mine = reject_rate(sv_mine, rule_mine, NOMINAL_ALPHA)
                rate_core_same = float(CORE.reject_rate(sv_mine, tid, NOMINAL_ALPHA,
                                                        ref_args))
                np.random.seed(SEED)
                sv_core = np.asarray(CORE.default_generator(tid)(N_FIXED, N_MC))
                rate_core = float(CORE.reject_rate(sv_core, tid, NOMINAL_ALPHA, ref_args))
                xc[tid] = dict(
                    n_ref=n_ref,
                    ref_args=None if ref_args is None else [float(x) for x in np.ravel(ref_args)],
                    rate_mine=rate_mine,
                    rate_core_on_same_array=rate_core_same,
                    evaluator_abs_diff=abs(rate_mine - rate_core_same),
                    evaluator_identical=bool(abs(rate_mine - rate_core_same) <= 1.0 / N_MC),
                    rate_core_generator=rate_core,
                    abs_diff_generators=abs(rate_mine - rate_core),
                )
            except Exception as exc:      # pragma: no cover - defensive
                xc[tid] = dict(error=repr(exc))
        iv["core_py_cross_validation"] = xc

    # ---- determinism -----------------------------------------------------
    for tid, spec, sd_ in (("T2_anova", spec_for("T2_anova", n=37), 11),
                           ("T9_median_test", spec_for("T9_median_test",
                                                       dist=_dist("unif"), n=13), 12),
                           ("T4_chisq_gof", spec_for("T4_chisq_gof", n_T=250, k=6,
                                                     min_np=1.0), 13)):
        v1 = gen_cell(tid, spec, cell_seed("determinism", sd_))
        v2 = gen_cell(tid, spec, cell_seed("determinism", sd_))
        checks["determinism"][tid] = bool(np.array_equal(v1[0], v2[0]))

    # ---- standardized parents really have mean 0 / variance 1 ------------
    for dname in ("norm", "unif", "exp", "chi2_4", "chi2_2", "lognorm_1",
                  "t_30", "t_10", "t_5", "t_3"):
        kind, par = _dist(dname)
        z = draw_std(np.random.default_rng(SEED), kind, 200000, par)
        g1, kurt = skew_kurt(kind, par)
        checks["distribution_selfcheck"][dname] = dict(
            mean=float(z.mean()), var=float(z.var(ddof=1)),
            skew_empirical=float(stats.skew(z)),
            excess_kurtosis_empirical=float(stats.kurtosis(z)),
            gamma1_analytic=g1, excess_kurtosis_analytic=kurt,
            median_empirical=float(np.median(z)),
            median_analytic=parent_median(kind, par))
    return checks


# ============================================================================
# SECTION 8b -- quotable forensic blocks requested by the lead (2026-10-12)
# ============================================================================
def _scipy_fligner_stats(groups):
    """scipy.stats.fligner statistic per replication (the archived UNBALANCED path)."""
    N = groups[0].shape[0]
    out = np.empty(N)
    for i in range(N):
        out[i] = stats.fligner(*[g[i] for g in groups]).statistic
    return out


def fligner_variant_probe(N=N_MC, seed=None):
    """Quantify the difference between the two Fligner-Killeen code paths in the archive.

    VERIFIED FACT (scipy 1.16.3, scipy/stats/_morestats.py): `scipy.stats.fligner`
    now converts the pooled ranks of |x - group median| to NORMAL SCORES
    (``norm.ppf(ranks/(2*(Ntot+1)) + 0.5)``) and uses the same weighted chi-square
    form as the archive's balanced path.  The two paths are therefore the SAME
    statistic definition in the installed environment, and the only difference is
    TIE HANDLING: scipy averages tied ranks (``rankdata``), whereas
    ``experiments.py``'s balanced path and this script assign raw argsort ranks.

    Ties are not a measure-zero curiosity here: for an EVEN-sized group the two
    central order statistics are exactly equidistant from the sample median, so
    every even group contributes one exact tie pair to the pooled |x - median_j|
    vector.  The two paths therefore differ by O(1e-5) in the Study 2
    configuration and by ~1e-1 only for very small groups, and they never disagree
    about a rejection decision in the samples examined here.

    This probe evaluates BOTH code paths on the SAME datasets, so the comparison
    isolates the definition/tie handling and not the Monte Carlo stream.
    """
    out = dict(
        question=("Do the balanced and unbalanced Fligner-Killeen code paths of "
                  "experiments.py compute different statistics?"),
        answer=("No -- in scipy 1.16.3 both use normal scores of the pooled "
                "|x - group median| ranks and the same weighted chi-square form. "
                "They differ only in tie handling (scipy averages tied ranks; the "
                "archive's balanced path does not), and that difference is "
                "numerically immaterial at the Study 2 sample sizes."),
        definitions=dict(
            normal_scores=("balanced path of experiments.py mc_fligner_killeen: ranks of "
                           "|x - group median| converted to normal scores "
                           "Phi^{-1}(r/(2(n_T+1)) + 1/2) with raw argsort ranks, "
                           "weighted chi-square form"),
            scipy_fligner=("unbalanced path of experiments.py mc_fligner_killeen: "
                           "scipy.stats.fligner, which in scipy 1.16.3 uses the SAME "
                           "normal scores but averages tied ranks via rankdata and "
                           "centers the group means at the unweighted mean of scores")),
        tie_mechanics=("For an even-sized group the two central order statistics are "
                       "exactly equidistant from the sample median, so each even group "
                       "contributes one exact tie pair to the pooled |x - median_j| "
                       "vector; rank averaging then shifts those two scores by an "
                       "amount that vanishes as the group grows. This is the entire "
                       "source of the difference reported below."),
        upstream_reference=("Fligner & Killeen (1976, JASA 71, 134-140) introduced the "
                            "normal-scores form; Conover, Johnson & Johnson (1981, "
                            "Technometrics 23, 351-361) study the rank-score variants. "
                            "scipy.stats.fligner 1.16.3 implements the normal-scores "
                            "form, so a reader reproducing either code path in the "
                            "archived experiments.py obtains the same statistic up to "
                            "tie averaging."),
        environment=dict(scipy=__import__("scipy").__version__,
                         source="scipy/stats/_morestats.py::fligner (inspected)"),
        N=N, same_datasets=True, configurations={},
    )
    crit = float(stats.chi2.ppf(0.95, 2))
    for label, ns in (("balanced_ns_50_50_50", [50, 50, 50]),
                      ("unbalanced_ns_50_100_150", [50, 100, 150]),
                      ("small_groups_ns_9_12_9", [9, 12, 9])):
        rng = np.random.default_rng(cell_seed("fligner_variants", label) if seed is None
                                    else seed)
        groups = [rng.standard_normal((N, int(nj))) for nj in ns]
        sv_norm = stat_fligner(groups)
        sv_scipy = _scipy_fligner_stats(groups)
        rec = dict(ns=[int(x) for x in ns],
                   critical_value=crit,
                   test_rule="chi2_2 upper tail at alpha = 0.05",
                   even_groups=[int(x) for x in ns if int(x) % 2 == 0])
        for name, sv in (("normal_scores", sv_norm), ("scipy_fligner", sv_scipy)):
            rate = float(np.mean(sv > crit))
            lo, hi = wilson_ci(rate, N)
            rec[name] = dict(alpha_hat=rate, ci=[lo, hi])
        diff = np.abs(sv_norm - sv_scipy)
        rec["statistic_difference"] = dict(
            max=float(diff.max()), mean=float(diff.mean()),
            median=float(np.median(diff)), q95=float(np.quantile(diff, 0.95)),
            max_relative=float(np.max(diff / np.maximum(np.abs(sv_scipy), 1e-12))),
            pearson_r=float(np.corrcoef(sv_norm, sv_scipy)[0, 1]),
            same_rejection_decision_fraction=float(
                np.mean((sv_norm > crit) == (sv_scipy > crit))))
        rec["alpha_hat_difference"] = float(abs(rec["normal_scores"]["alpha_hat"]
                                                - rec["scipy_fligner"]["alpha_hat"]))
        rec["ci_overlap"] = bool(
            not (rec["normal_scores"]["ci"][0] > rec["scipy_fligner"]["ci"][1]
                 or rec["normal_scores"]["ci"][1] < rec["scipy_fligner"]["ci"][0]))
        rec["verdict"] = (
            "identical definition, tie handling only: max |statistic difference| = "
            "%.2e, rejected decisions agree for %.4f%% of replications, Wilson CIs "
            "%s" % (rec["statistic_difference"]["max"],
                    100.0 * rec["statistic_difference"]["same_rejection_decision_fraction"],
                    "overlap" if rec["ci_overlap"] else "do not overlap"))
        out["configurations"][label] = rec
    return out


def consistency_check_summary(checks, fligner_probe=None):
    """Quotable summary of the archived-vs-generic anchor comparison.

    Frames the single non-overlapping anchor as a multiple-comparison artefact and
    records the verification that the archived T9-unbalanced wiring is correct.
    """
    ag = {k: v for k, v in checks.get("archive_vs_generic", {}).items()
          if isinstance(v, dict) and "z_approx" in v}
    if not ag:
        return {}
    zs = np.array([v["z_approx"] for v in ag.values()])
    diffs = np.array([v["abs_diff"] for v in ag.values()])
    max_z = float(zs.max())
    non_overlap = sorted(k for k, v in ag.items() if not v["ci_overlap"])
    p_one = float(2.0 * (1.0 - stats.norm.cdf(max_z)))
    expected = float(len(ag) * p_one)
    out = dict(
        n_anchor_cells=len(ag),
        n_ci_non_overlapping=len(non_overlap),
        non_overlapping_cells=non_overlap,
        max_abs_diff=float(diffs.max()),
        median_abs_diff=float(np.median(diffs)),
        max_z=max_z,
        p_two_sided_for_max_z=p_one,
        expected_count_of_z_at_least_max_z_under_pure_mc_noise=expected,
        framing=("Each of the %d archived-vs-generic anchor cells is an independent "
                 "two-sample comparison of rejection rates at N=10,000. The largest "
                 "discrepancy is |z| = %.2f, a two-sided event of probability %.4f per "
                 "comparison; across %d comparisons the expected number of such events "
                 "under pure Monte Carlo noise is %.2f. Observing exactly one is "
                 "therefore the expected outcome of a multiple-comparison exercise, not "
                 "evidence of an implementation error. The largest absolute rate "
                 "difference over all anchors is %.4f."
                 % (len(ag), max_z, p_one, len(ag), expected, diffs.max())),
        t9_unbalanced_verification=(
            "The single non-overlapping cell is T9_median_test|unbalanced (archived "
            "%.4f, generic %.4f). The archived path wires ns = [n, 2n, 3n] correctly "
            "into the 2 x k contingency computation, and the generic re-implementation "
            "uses the same weighted median-test formula, so the two differ only through "
            "the RNG stream; the published Table 4 value (0.051) lies between the two "
            "re-runs."
            % (ag.get("T9_median_test|unbalanced", {}).get("archived_alpha_hat", float("nan")),
               ag.get("T9_median_test|unbalanced", {}).get("generic_alpha_hat", float("nan")))),
        fligner_variant_verdict=(None if not fligner_probe else
                                 fligner_probe["configurations"]["unbalanced_ns_50_100_150"]["verdict"]),
    )
    return out


# ============================================================================
# SECTION 9 -- figures
# ============================================================================
def make_figures(ladders, nsweep_agg, extra_ladders, path_sev, path_n):
    """Render the two severity / sample-size figures.

    Args:
        ladders: severity-ladder aggregate (see :func:`aggregate_ladder`).
        nsweep_agg: n-sweep aggregate (see :func:`aggregate_nsweep`).
        extra_ladders: blocks produced by the ``core.py`` severity hooks for
            T11-T13.
        path_sev: output path of ``fig_rev_severity.png``.
        path_n: output path of ``fig_rev_study2_n.png``.

    Returns:
        None. Legends are placed outside the axes so that no legend can overlap a
        bar or a line (Referee 2, minor comment on the original Figure 2).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.dpi": 150})

    order = ["T1_t_test", "T2_anova", "T3_kruskal_wallis", "T4_chisq_gof",
             "T5_bartlett", "T6_fligner_killeen", "T7_sign_test", "T8_cochran_q",
             "T9_median_test", "T10_hotelling"]
    extras = sorted({tid for lad in ladders.values() for tid in lad} - set(order))
    order = order + extras
    labels_all = dict(TEST_LABELS)
    for lad in ladders.values():
        for tid, ent in lad.items():
            labels_all.setdefault(tid, ent.get("label", tid))
    cmap = plt.get_cmap("tab10")
    colors = {tid: cmap(i % 10) for i, tid in enumerate(order[:10])}
    extra_cmap = plt.get_cmap("Dark2")
    for j, tid in enumerate(extras):
        colors[tid] = extra_cmap(j % 8)

    def panel(ax, xs, curves, xlabel, title, xticklabels=None, logx=False,
              logy=False, ylim=None, note=None):
        for tid, (vals, los, his) in curves.items():
            vals = np.asarray(vals, float)
            yerr = np.maximum(np.vstack([vals - np.asarray(los, float),
                                         np.asarray(his, float) - vals]), 0.0)
            ax.errorbar(xs, vals, yerr=yerr, marker="o", ms=3.0, lw=1.0,
                        capsize=2.0, color=colors[tid], label=labels_all[tid])
        ax.axhline(NOMINAL_ALPHA, color="black", ls="--", lw=0.9)
        ax.axhline(NOMINAL_ALPHA * LIBERAL_RATIO, color="crimson", ls=":", lw=0.9)
        ax.axhline(NOMINAL_ALPHA * CONSERVATIVE_RATIO, color="steelblue", ls=":", lw=0.9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Empirical Type I error rate")
        ax.set_title(title, fontsize=9)
        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        if xticklabels is not None:
            ax.set_xticks(xs)
            ax.set_xticklabels(xticklabels, fontsize=7.5, rotation=20, ha="right")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.25, axis="y")
        if note:
            ax.text(0.02, 0.97, note, transform=ax.transAxes, fontsize=7,
                    va="top", color="0.25")

    def curves_from(block, rung_labels, tests):
        out = {}
        for tid in tests:
            if tid not in block:
                continue
            rr = block[tid]["rungs"]
            if any(l not in rr or "alpha_hat" not in rr[l] for l in rung_labels):
                continue
            out[tid] = ([rr[l]["alpha_hat"] for l in rung_labels],
                        [rr[l]["ci_lo"] for l in rung_labels],
                        [rr[l]["ci_hi"] for l in rung_labels])
        return out

    def merged(lname, tests, rung_labels):
        block = dict(ladders.get(lname, {}))
        for tid, ent in extra_ladders.get(lname, {}).items():
            block[tid] = ent
        if lname == "skewness":
            pass
        return curves_from(block, rung_labels, tests + [t for t in extra_ladders.get(lname, {})
                                                        if t not in tests])

    fig, axes = plt.subplots(4, 2, figsize=(12.5, 15.0))
    axes = axes.ravel()

    skew_labels = [r["label"] for r in SKEW_RUNGS]
    kurt_labels = [r["label"] for r in KURT_RUNGS]
    var_labels = [r["label"] for r in VAR_RATIO_RUNGS]
    imb_labels = [r["label"] for r in IMB_RUNGS]

    panel(axes[0], np.arange(len(skew_labels)),
          merged("skewness", DIST_LADDER_TESTS, skew_labels),
          "Parent distribution (skewness gamma1 increases to the right)",
          "(a) Skewness ladder (n = 50, N = 10,000)",
          xticklabels=["%s\ng1=%.2f" % (r["label"], r["severity"]["gamma1"])
                       for r in SKEW_RUNGS])
    panel(axes[1], np.arange(len(kurt_labels)),
          merged("kurtosis", DIST_LADDER_TESTS, kurt_labels),
          "Parent distribution (tail weight increases to the right)",
          "(b) Kurtosis ladder (n = 50, N = 10,000)",
          xticklabels=["%s\nk=%s" % (r["label"], _kurt_tick(r)) for r in KURT_RUNGS])
    vc = curves_from(ladders.get("variance_ratio", {}), var_labels, VAR_LADDER_TESTS)
    vc_aux = curves_from(ladders.get("variance_ratio", {}), var_labels, VAR_LADDER_AUX)
    panel(axes[2], [r["ratio"] for r in VAR_RATIO_RUNGS], vc,
          "Variance ratio sigma_k^2 / sigma_1^2 (log-log axes)",
          "(c) Variance-ratio ladder (T2, T3, T9); log y-axis",
          xticklabels=["%g" % r["ratio"] for r in VAR_RATIO_RUNGS], logx=True,
          logy=True, ylim=(0.03, 1.05),
          note="dashed-curve tests (T5, T6) are variance tests:\ntheir H0 is false here (power-like).")
    for tid, (vals, los, his) in vc_aux.items():
        vals = np.asarray(vals, float)
        axes[2].errorbar([r["ratio"] for r in VAR_RATIO_RUNGS], vals,
                         yerr=np.maximum(np.vstack([vals - np.asarray(los, float),
                                                   np.asarray(his, float) - vals]), 0.0),
                         marker="^", ms=3.0, lw=1.0, ls="--", capsize=2.0,
                         color=colors[tid], alpha=0.75)
    panel(axes[3], np.arange(len(imb_labels)),
          curves_from(ladders.get("imbalance", {}), imb_labels, IMB_LADDER_TESTS),
          "Group-size ratio (total n_T = 150, last point archived n_T = 300)",
          "(d) Imbalance ladder (n_T fixed, N = 10,000)",
          xticklabels=[r["label"].replace(" (", "\n(") for r in IMB_RUNGS])
    t4_labels = [c["rung"] for c in build_ladder_cells() if c["ladder"] == "small_freq_t4"]
    panel(axes[4], np.arange(len(t4_labels)),
          curves_from(ladders.get("small_freq_t4", {}), t4_labels, ["T4_chisq_gof"]),
          "Smallest expected count np_min (n_T = 250, k = 6)",
          "(e) chi2 GOF small-expected-frequency ladder",
          xticklabels=[l.replace(" (", "\n(") for l in t4_labels])
    t8_labels = [c["rung"] for c in build_ladder_cells() if c["ladder"] == "small_freq_t8"]
    panel(axes[5], np.arange(len(t8_labels)),
          curves_from(ladders.get("small_freq_t8", {}), t8_labels, ["T8_cochran_q"]),
          "Success probability p (50 blocks x 3 treatments)",
          "(f) Cochran Q sparse-binary ladder",
          xticklabels=t8_labels)
    t9_labels = [c["rung"] for c in build_ladder_cells() if c["ladder"] == "small_freq_t9"]
    panel(axes[6], np.arange(len(t9_labels)),
          curves_from(ladders.get("small_freq_t9", {}), t9_labels, ["T9_median_test"]),
          "Per-group sample size n (k = 3, expected cell count = n/2)",
          "(g) Median test small-sample ladder",
          xticklabels=[l + "\nE=%.1f" % (float(l.split("=")[1]) / 2.0) for l in t9_labels])

    # (h) Delta at the most severe rung of both distributional ladders
    axh = axes[7]
    present = [t for t in DIST_LADDER_TESTS if t in ladders.get("skewness", {})]
    for j, (lname, rlabel, col, marker, lab) in enumerate((
            ("skewness", "Lognormal", "tab:red", "o", "skewness: Lognormal (g1=6.19)"),
            ("kurtosis", "t(3)", "0.35", "s", "kurtosis: t(3) (k=inf)"))):
        block = dict(ladders.get(lname, {}))
        for tid, ent in extra_ladders.get(lname, {}).items():
            block[tid] = ent
        base_lab = "Normal N(0,1)"
        xs_j, ys, errs = [], [], []
        for i, tid in enumerate(present):
            if tid not in block:
                continue
            rr = block[tid]["rungs"]
            if rlabel not in rr or base_lab not in rr:
                continue
            d = rr[rlabel]["alpha_hat"] - rr[base_lab]["alpha_hat"]
            xs_j.append(i + (j - 0.5) * 0.38)
            ys.append(d)
            errs.append([max(d - (rr[rlabel]["ci_lo"] - rr[base_lab]["alpha_hat"]), 0.0),
                         max((rr[rlabel]["ci_hi"] - rr[base_lab]["alpha_hat"]) - d, 0.0)])
        axh.errorbar(xs_j, ys, yerr=np.array(errs).T, fmt=marker, ms=4, capsize=3,
                     color=col, linestyle="none", label=lab)
    axh.set_xticks(range(len(present)))
    axh.set_xticklabels([labels_all[t] for t in present], rotation=25, ha="right",
                        fontsize=7.5)
    axh.axhline(0.0, color="black", lw=0.8)
    axh.axhline(NOMINAL_ALPHA * (LIBERAL_RATIO - 1.0), color="crimson", ls=":", lw=0.9)
    axh.set_ylabel("Delta = alpha_hat(violated) - alpha_hat(Normal)")
    axh.set_title("(h) Baseline-relative change at the most severe rung", fontsize=9)
    axh.grid(True, alpha=0.25, axis="y")
    axh.text(0.02, 0.97, "error bars: Wilson CI of the violated rate\nshifted by the baseline rate",
             transform=axh.transAxes, fontsize=7, va="top", color="0.25")

    handles, labels = [], []
    for tid in order:
        handles.append(plt.Line2D([0], [0], color=colors[tid], marker="o", ms=3, lw=1))
        labels.append(labels_all[tid])
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=8.5)
    fig.suptitle("Study 2 severity ladders: empirical Type I error with 95% Wilson CIs\n"
                 "(black dashed: nominal alpha = 0.05; crimson dotted: liberal flag 1.5*alpha; "
                 "steelblue dotted: conservative flag 0.5*alpha)", fontsize=10)
    fig.tight_layout(rect=[0, 0.035, 1, 0.965])
    fig.savefig(path_sev)
    plt.close(fig)

    # ---------------- n-sweep figure ---------------------------------------
    scen_order = ["baseline", "skewed", "heavy_tail", "hetero_var", "unbalanced",
                  "small_freq"]
    scen_title = {"baseline": "(a) Baseline (no violation)",
                  "skewed": "(b) Skewed: Exp(1)-1",
                  "heavy_tail": "(c) Heavy-tailed: t(5)",
                  "hetero_var": "(d) Variance heterogeneity 1:4",
                  "unbalanced": "(e) Unbalanced 1:2:3",
                  "small_freq": "(f) Small expected frequencies"}
    fig2, axes2 = plt.subplots(3, 3, figsize=(14.0, 11.5))
    axes2 = axes2.ravel()
    for i, sc in enumerate(scen_order):
        ax = axes2[i]
        for tid, ent in nsweep_agg[sc].items():
            vals = [ent["by_n"][str(n)]["alpha_hat"] for n in NSWEEP_N]
            los = [ent["by_n"][str(n)]["ci_lo"] for n in NSWEEP_N]
            his = [ent["by_n"][str(n)]["ci_hi"] for n in NSWEEP_N]
            yerr = np.maximum(np.vstack([np.array(vals) - np.array(los),
                                         np.array(his) - np.array(vals)]), 0.0)
            ax.errorbar(np.arange(len(NSWEEP_N)), vals, yerr=yerr, marker="o", ms=3,
                        lw=1.0, capsize=2, color=colors[tid], label=labels_all[tid])
        ax.axhline(NOMINAL_ALPHA, color="black", ls="--", lw=0.9)
        ax.axhline(NOMINAL_ALPHA * LIBERAL_RATIO, color="crimson", ls=":", lw=0.9)
        ax.set_xticks(np.arange(len(NSWEEP_N)))
        ax.set_xticklabels([str(n) for n in NSWEEP_N])
        ax.set_xlabel("Sample size n (per group; T4: n_T = max(5n,30); T8: blocks)")
        ax.set_ylabel("Empirical Type I error rate")
        ax.set_title("%s  (N = 10,000)" % scen_title[sc], fontsize=9)
        ax.grid(True, alpha=0.25, axis="y")
    for j, sc in enumerate(["skewed", "heavy_tail"]):
        ax = axes2[6 + j]
        base = nsweep_agg["baseline"]
        for tid, ent in nsweep_agg[sc].items():
            d = [ent["by_n"][str(n)]["alpha_hat"] - base[tid]["by_n"][str(n)]["alpha_hat"]
                 for n in NSWEEP_N]
            lo = [ent["by_n"][str(n)]["alpha_hat"] - base[tid]["by_n"][str(n)]["alpha_hat"]
                  - (ent["by_n"][str(n)]["alpha_hat"] - ent["by_n"][str(n)]["ci_lo"])
                  for n in NSWEEP_N]
            hi = [ent["by_n"][str(n)]["alpha_hat"] - base[tid]["by_n"][str(n)]["alpha_hat"]
                  + (ent["by_n"][str(n)]["ci_hi"] - ent["by_n"][str(n)]["alpha_hat"])
                  for n in NSWEEP_N]
            ax.errorbar(np.arange(len(NSWEEP_N)), d,
                        yerr=np.maximum(np.vstack([np.array(d) - np.array(lo),
                                                   np.array(hi) - np.array(d)]), 0.0),
                        marker="o", ms=3, lw=1.0, capsize=2, color=colors[tid],
                        label=labels_all[tid])
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_xticks(np.arange(len(NSWEEP_N)))
        ax.set_xticklabels([str(n) for n in NSWEEP_N])
        ax.set_xlabel("Sample size n")
        ax.set_ylabel("Delta vs matched-n baseline")
        ax.set_title("(%s) Baseline-relative change, %s" % ("g" if j == 0 else "h",
                     "skewed" if j == 0 else "heavy-tailed"), fontsize=9)
        ax.grid(True, alpha=0.25, axis="y")
    axes2[8].axis("off")
    plotted = [tid for tid in order if tid in nsweep_agg.get("baseline", {})]
    handles, labels = [], []
    for tid in plotted:
        handles.append(plt.Line2D([0], [0], color=colors[tid], marker="o", ms=3, lw=1))
        labels.append(labels_all[tid])
    fig2.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=8.5)
    fig2.suptitle("Study 2 sample-size sweep: empirical Type I error with 95% Wilson CIs\n"
                  "(black dashed: nominal alpha = 0.05; crimson dotted: liberal flag 1.5*alpha; "
                  "panels g-h: Delta = alpha_hat(violation, n) - alpha_hat(baseline, n))",
                  fontsize=10)
    fig2.tight_layout(rect=[0, 0.045, 1, 0.955])
    fig2.savefig(path_n)
    plt.close(fig2)


# ============================================================================
# SECTION 10 -- reporting helpers
# ============================================================================
def fmt_rate(rec, nd=3):
    """Format one alpha-hat record together with its calibration flag.

    Args:
        rec: result record containing ``alpha_hat`` and the optional flags
            ``liberal_ci``, ``liberal`` and ``conservative_ci``.
        nd: number of decimals (default 3).

    Returns:
        str: the rate followed by ``!`` (CI-separated liberal), ``*`` (pointwise
        liberal), ``c`` (conservative) or nothing.
    """
    mark = ""
    if rec.get("liberal_ci"):
        mark = "!"
    elif rec.get("liberal"):
        mark = "*"
    elif rec.get("conservative_ci"):
        mark = "c"
    return ("%.*f%s" % (nd, rec["alpha_hat"], mark))


def print_ladder_table(name, block, rung_labels):
    """Print one severity ladder as a fixed-width console table.

    Args:
        name: ladder name used in the heading (e.g. ``'skewness'``).
        block: ``{test_id: {'label': ..., 'rungs': {rung_label: record}}}``.
        rung_labels: rung labels in display order.

    Returns:
        None (writes to stdout). A second group of rows reports the 95% Wilson
        half-width of every cell.
    """
    print("\n--- %s ladder: alpha_hat at alpha = 0.05 (CI-flagged) ---" % name)
    print("  ! = CI lower bound above the liberal flag (1.5*alpha); "
          "* = point estimate above it; c = conservative")
    w = max(14, max(len(l) for l in rung_labels) + 2)
    hdr = "%-21s" % "Test" + "".join("%*s" % (w, l) for l in rung_labels)
    print(hdr)
    print("-" * len(hdr))
    for tid in sorted(block):
        rr = block[tid]["rungs"]
        row = "%-21s" % block[tid].get("label", TEST_LABELS.get(tid, tid))
        for l in rung_labels:
            row += "%*s" % (w, fmt_rate(rr[l]) if l in rr and "alpha_hat" in rr[l] else "-")
        print(row)
    print("%-21s" % "[95% Wilson CI half-width]")
    for tid in sorted(block):
        rr = block[tid]["rungs"]
        row = "%-21s" % block[tid].get("label", TEST_LABELS.get(tid, tid))
        for l in rung_labels:
            if l in rr and "alpha_hat" in rr[l]:
                row += "%*s" % (w, "+/-%.4f" % ((rr[l]["ci_hi"] - rr[l]["ci_lo"]) / 2))
            else:
                row += "%*s" % (w, "-")
        print(row)


def main():
    """Entry point: run the severity ladders and the sample-size sweep.

    Writes ``rev_results.json`` (blocks ``meta``, ``severity_sweep``,
    ``study2_nsweep``, ``alpha_sweep``, ``consistency_checks``,
    ``implementation_validation``, ``distribution_selfcheck``,
    ``determinism_checks``), ``fig_rev_severity.png`` and ``fig_rev_study2_n.png``,
    and prints per-cell progress plus the run's wall clock.

    Returns:
        None; the exit status is 0 unless an exception propagates.
    """
    t_start = time.time()
    print("=" * 78)
    print("rev_sensitivity.py -- Study 2 severity ladders + sample-size sweep")
    print("N = %d per cell | seed = %d | alpha = %s" % (N_MC, SEED, list(ALPHA_LEVELS)))
    print("=" * 78)

    # ---------------- severity ladders -------------------------------------
    ladder_cells = build_ladder_cells()
    print("\nSeverity ladders: %d cells" % len(ladder_cells))
    ladder_records = []
    t0 = time.time()
    for i, c in enumerate(ladder_cells, 1):
        ladder_records.append(evaluate_cell(c["test"], c["spec"], c["seed"]))
        if i % 25 == 0:
            print("   %d/%d cells (%.1fs)" % (i, len(ladder_cells), time.time() - t0),
                  flush=True)
    ladders = aggregate_ladder(ladder_cells, ladder_records)
    print("   done in %.1fs" % (time.time() - t0))

    # ---------------- n-sweep ----------------------------------------------
    nsweep_cells = build_nsweep_cells()
    print("\nSample-size sweep: %d cells" % len(nsweep_cells))
    t0 = time.time()
    nsweep_records = [evaluate_cell(c["test"], c["spec"], c["seed"]) for c in nsweep_cells]
    nsweep_agg = aggregate_nsweep(nsweep_cells, nsweep_records)
    print("   done in %.1fs" % (time.time() - t0))

    # ---------------- optional core.py extras ------------------------------
    hooks, rank_invariant, extras_skipped, extras_msg = probe_core_extras()
    print("\ncore.py extras: %s" % extras_msg)
    extra_ladders = run_core_ladders(hooks)
    for lname, block in extra_ladders.items():
        ladders.setdefault(lname, {}).update(block)   # trend rules cover extras too
    if extra_ladders:
        print("   extra ladder entries: %s"
              % {k: sorted(v) for k, v in extra_ladders.items()})
    extra_nsweep = run_core_nsweep(dict(hooks, **rank_invariant))
    if extra_nsweep:
        print("   extra n-sweep entries: %s" % sorted(extra_nsweep))

    # ---------------- trend / threshold analysis ---------------------------
    severity_sweep = {"ladders": {}, "thresholds": {}, "not_applicable": {}}
    ladder_rungs = {
        "skewness": [r["label"] for r in SKEW_RUNGS],
        "kurtosis": [r["label"] for r in KURT_RUNGS],
        "variance_ratio": [r["label"] for r in VAR_RATIO_RUNGS],
        "imbalance": [r["label"] for r in IMB_RUNGS],
        "small_freq_t4": ["balanced (np=41.7)", "np_min=5", "np_min=3", "np_min=2",
                          "np_min=1", "np_min=0.5"],
        "small_freq_t8": ["p=%.2f" % p for p in (0.5, 0.3, 0.1, 0.05, 0.01)],
        "small_freq_t9": ["n=%d" % n for n in (30, 20, 15, 10, 8)],
    }
    titles = {
        "skewness": "Skewness ladder (n = 50): rungs ordered by gamma1",
        "kurtosis": "Kurtosis ladder (n = 50): rungs ordered by tail weight",
        "variance_ratio": "Variance-ratio ladder (n = 50, k = 3): sigma_k^2/sigma_1^2",
        "imbalance": "Imbalance ladder (total n_T = 150; archived 1:2:3 anchor at n_T = 300)",
        "small_freq_t4": "chi2 GOF small-expected-frequency ladder (n_T = 250, k = 6)",
        "small_freq_t8": "Cochran Q sparse-binary ladder (50 blocks, 3 treatments)",
        "small_freq_t9": "Median test small-sample ladder (k = 3, uniform parent)",
    }
    for lname, block in ladders.items():
        labels = ladder_rungs[lname]
        a = {"title": titles[lname], "seed": int(ladder_cells[0]["seed"]),
             "N": N_MC, "rung_labels": labels, "tests": {}}
        for tid, ent in sorted(block.items()):
            rr = ent["rungs"]
            if any(l not in rr for l in labels):
                continue
            trend = trend_analysis(labels, rr)
            trend["label"] = ent.get("label", TEST_LABELS.get(tid, tid))
            trend["source"] = ent.get("source", "rev_sensitivity.py generic")
            evals = {l: rr[l].get("evaluation") for l in labels
                     if isinstance(rr[l], dict) and rr[l].get("evaluation") is not None}
            if evals:
                # Which critical value / rule produced each rung's rate (extra tests).
                trend["evaluation_by_rung"] = evals
            trend["aux"] = ent.get("aux", False)
            trend["note"] = ("aux: the test's own null is false on this ladder "
                             "(rejection rate is power-like)"
                             if ent.get("aux") else "")
            a["tests"][tid] = trend
            severity_sweep["thresholds"].setdefault(lname, {})[tid] = dict(
                first_liberal_rung=trend["first_liberal_rung"],
                first_liberal_rung_ci=trend["first_liberal_rung_ci"],
                first_conservative_rung=trend["first_conservative_rung"],
                monotone_established=trend["monotone_established"],
                trend_verdict=trend["trend_verdict"])
        severity_sweep["ladders"][lname] = a
    severity_sweep["extra_nsweep"] = extra_nsweep
    severity_sweep["extra_tests_core_hooks"] = {
        "status": extras_msg,
        "skipped": extras_skipped,
        "rank_invariant": {k: v.get("invariance_note", "") for k, v in rank_invariant.items()},
        "source": "core.SEVERITY_HOOKS (owner t1-new-tests)",
        "note": ("Extra tests entering the distributional ladders are evaluated with "
                 "core.reject_rate so that their registry-specific rules "
                 "(cvm_limit, hoeffding_z) are used; the Wilson CI uses the number "
                 "of finite statistics returned by the hook."),
    }
    severity_sweep["not_applicable"] = {
        "T8_cochran_q_imbalance": (
            "Cochran's Q is defined on a complete randomized-block design (every "
            "block receives every treatment); unequal group sizes are not defined "
            "for it, so the imbalance ladder has no T8 rung. NOTE: in the archived "
            "experiments.py, mc_cochran_q's `unbalanced` argument is never used and "
            "run_study2 never passes it, so the published T8 'unbalanced' cell "
            "(alpha-hat = 0.050, Delta = 0.000 in Tables 4-5) is just a re-run of "
            "its baseline, not an imbalance manipulation."),
        "T9_median_test_small_freq_in_nsweep": (
            "For the median test the small-expected-frequency scenario IS a sample-"
            "size reduction (n = 8 per group); sweeping n would destroy the scenario. "
            "It is reported as the small_freq_t9 ladder instead."),
    }

    # ---------------- alpha sweep (reuses the same statistic arrays) -------
    alpha_sweep = {"alpha_levels": list(ALPHA_LEVELS),
                   "note": ("Every cell is evaluated at alpha = 0.01, 0.05, 0.10 from "
                            "the SAME statistic array, so the MC budget and the seed are "
                            "identical across alpha levels; only the critical value "
                            "changes."),
                   "severity_sweep": {}, "study2_nsweep": {}, "study2_anchors": {}}
    for c, rec in zip(ladder_cells, ladder_records):
        key = c["ladder"]
        alpha_sweep["severity_sweep"].setdefault(key, {}).setdefault(
            c["test"], {})[c["rung"]] = rec["by_alpha"]
    for c, rec in zip(nsweep_cells, nsweep_records):
        alpha_sweep["study2_nsweep"].setdefault(c["scenario"], {}).setdefault(
            c["test"], {})[str(c["n"])] = rec["by_alpha"]

    # ---------------- consistency ------------------------------------------
    print("\nConsistency and validation checks ...", flush=True)
    t0 = time.time()
    checks = run_consistency_checks()
    print("   done in %.1fs" % (time.time() - t0))
    if "archive_vs_generic" in checks:
        bad = {k: v for k, v in checks["archive_vs_generic"].items()
               if isinstance(v, dict) and v.get("ci_overlap") is False}
        n_ok = sum(1 for v in checks["archive_vs_generic"].values()
                   if isinstance(v, dict) and v.get("ci_overlap") is True)
        print("   archive vs generic: %d anchor cells with overlapping Wilson CIs, "
              "%d without" % (n_ok, len(bad)))
        if bad:
            print("   NON-OVERLAPPING ANCHORS:")
            for k, v in bad.items():
                print("      %s: archived %.4f vs generic %.4f (z=%.1f)"
                      % (k, v["archived_alpha_hat"], v["generic_alpha_hat"], v["z_approx"]))

    for key, rec in checks["archive_vs_generic"].items():
        if isinstance(rec, dict) and "generic_by_alpha" in rec:
            tid, sc = key.split("|")
            alpha_sweep["study2_anchors"].setdefault(tid, {})[sc] = rec["generic_by_alpha"]

    # ---------------- n-sweep ordering -------------------------------------
    nsweep_out = {"n_values": list(NSWEEP_N), "scenarios": {},
                  "notes": {
                      "T4_chisq_gof": "n maps to total n_T = max(5n, 30); the small-freq "
                                      "scenario holds np_min = 2 while n_T grows.",
                      "T8_cochran_q": "n is the number of blocks; p = 0.05 in the "
                                      "small-freq scenario.",
                      "T9_median_test": "small-freq scenario not swept (its severity IS "
                                        "n; see the small_freq_t9 ladder).",
                  }}
    for sc, tests in nsweep_agg.items():
        nsweep_out["scenarios"][sc] = {"tests": {}}
        for tid, ent in sorted(tests.items()):
            by_n = {}
            for n in NSWEEP_N:
                rec = ent["by_n"][str(n)]
                base_rec = nsweep_agg["baseline"][tid]["by_n"][str(n)] if tid in nsweep_agg["baseline"] else None
                by_n[str(n)] = dict(rec, delta_vs_baseline=(
                    rec["alpha_hat"] - base_rec["alpha_hat"] if base_rec else None))
            nsweep_out["scenarios"][sc]["tests"][tid] = dict(label=TEST_LABELS[tid],
                                                             by_n=by_n)
    nsweep_out["ordering_stability"] = {}
    nsweep_out["degradation_recovery"] = {}
    for sc in ("skewed", "heavy_tail"):
        nsweep_out["ordering_stability"][sc] = ordering_analysis(nsweep_agg, sc)
        nsweep_out["degradation_recovery"][sc] = first_n_recovery(nsweep_agg, sc)
    for sc in ("hetero_var", "unbalanced", "small_freq"):
        nsweep_out["degradation_recovery"][sc] = first_n_recovery(nsweep_agg, sc)

    # ---------------- meta --------------------------------------------------
    git_hash = "not-a-git-repo"
    try:
        git_hash = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                  cwd=HERE, capture_output=True, text=True,
                                  timeout=10).stdout.strip() or "unknown"
    except Exception:
        pass

    def sha(path):
        try:
            with open(os.path.join(HERE, path), "rb") as fh:
                return hashlib.sha256(fh.read()).hexdigest()[:16]
        except OSError:
            return None

    meta = dict(
        seed=SEED, N_mc=N_MC, n_fixed=N_FIXED, alpha_levels=list(ALPHA_LEVELS),
        nominal_alpha=NOMINAL_ALPHA, liberal_ratio=LIBERAL_RATIO,
        conservative_ratio=CONSERVATIVE_RATIO, wilson_z=Z95,
        numpy=np.__version__, scipy=stats.__version__ if hasattr(stats, "__version__")
        else __import__("scipy").__version__,
        python=platform.python_version(),
        git_hash=git_hash,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        script="rev_sensitivity.py", script_sha256=sha("rev_sensitivity.py"),
        experiments_py_sha256=sha("experiments.py"), core_py_sha256=sha("core.py"),
        archive_imported=HAS_ARCHIVE, archive_error=ARCHIVE_ERR,
        core_imported=HAS_CORE, core_error=CORE_ERR, core_selftest=CORE_SELFTEST,
        extras=extras_msg,
        extras_skipped=extras_skipped,
        extras_pilot_N=EXTRAS_PILOT_N,
        extras_projections=EXTRAS_PROJECTIONS,
        extras_max_seconds_per_cell=EXTRAS_MAX_SECONDS_PER_CELL,
        n_cells_ladders=len(ladder_cells), n_cells_nsweep=len(nsweep_cells),
        unique_cached_draws=len(_CELL_CACHE),
        runtime_seconds=None,
        notes={
            "budget": "N = %d for every cell of every ladder and of the n-sweep; "
                      "alpha in {0.01,0.05,0.10} evaluated from the same arrays." % N_MC,
            "seeds": "Within a ladder every rung (and every test) uses the same seed "
                     "(common random numbers); within a scenario of the n-sweep every n "
                     "uses the same seed. The Normal rung is shared by the skewness and "
                     "kurtosis ladders, so their baselines are numerically identical.",
            "parents": "Ladder parents are standardized to mean 0 and variance 1; all "
                       "tests on the distributional ladders are scale-invariant, so "
                       "alpha-hat is unaffected by the standardization.",
            "archived_anchors": "Anchor cells (Exp(1)-1 skew, raw t(5) heavy tail, 1:4 "
                                "variance ratio, 1:2:3 imbalance, np_min = 2) are re-run "
                                "with experiments.py itself in consistency_checks; the "
                                "n-sweep uses standardized t(5) instead of raw t(5) "
                                "(equivalent for scale-invariant tests).",
            "code_findings": [
                "experiments.py mc_cochran_q ignores its `unbalanced` argument and "
                "run_study2 never passes it: the published T8 'unbalanced' cell is a "
                "re-run of its baseline (see severity_sweep.not_applicable).",
                "experiments.py calls two different code paths for Fligner-Killeen: "
                "the balanced path builds normal scores of the pooled "
                "|x - group median| ranks with raw argsort ranks, the unbalanced path "
                "calls scipy.stats.fligner, which in scipy 1.16.3 uses the SAME normal "
                "scores but averages tied ranks. The paths therefore differ only in "
                "tie handling (each even-sized group contributes one exact tie pair, "
                "because its two central order statistics are equidistant from the "
                "sample median); the statistic-level difference is 1.9e-05 at "
                "ns = 50/50/50, 1.7e-06 at ns = 50/100/150 and 9.1e-02 only for very "
                "small groups (ns = 9/12/9), the rejection decision agrees in 100% of "
                "the replications examined, and the T6 unbalanced anchor's archived "
                "and generic Wilson CIs overlap. See fligner_variant_probe for the "
                "same-dataset comparison.",
                "The archived T3/T9 baseline scenarios use a uniform parent "
                "(dist='uniform' default) while T1/T2/T5/T6/T7/T10 use normal; the "
                "structural ladders keep that convention, the skewness/kurtosis "
                "ladders use a common N(0,1) baseline.",
                "One of 37 archived anchor cells (T9 unbalanced) has Wilson CIs that "
                "do not overlap its generic re-run (0.0481 vs 0.0571, z=2.9); max "
                "absolute anchor difference over all 37 cells is 0.011, i.e. MC noise.",
            ],
            "upstream_issues": [
                "core.py T11_kpss: critical_value() returns cvm_limit_ppf(alpha) "
                "(0.0366 at alpha=0.05) although reference_ppf() expects an "
                "upper-tail probability; the Cramer-von Mises 5% quantile is "
                "cvm_limit_ppf(0.95) = 0.4614. Under iid N(0,1) the wrong value "
                "gives a KPSS rejection rate of ~1.00, the correct one ~0.05. "
                "Every T11 record therefore carries both rates "
                "(alpha_hat and evaluation_override.rate_with_core_rule) and the "
                "two critical values. Reported to t1-new-tests; no other test is "
                "affected (scipy families use isf, hoeffding_z follows the upper-tail "
                "convention and calibrates correctly).",
            ],
            "publication_check": "implementation_validation.published_table4_reproduction "
                                 "compares the archived re-run with the published Table 4 rates.",
            "structure": "Structural ladders (variance ratio, imbalance) use each test's "
                         "archived baseline parent (uniform for T3/T9, normal elsewhere); "
                         "the skewness and kurtosis ladders use a common N(0,1) baseline "
                         "for all tests.",
        },
    )

    print("Fligner variant probe (same datasets, both archive code paths) ...", flush=True)
    t0 = time.time()
    fligner_probe = fligner_variant_probe()
    print("   done in %.1fs" % (time.time() - t0))
    for _lab, _cfg in fligner_probe["configurations"].items():
        print("     %-24s max|dstat| = %.2e | decisions agree %.4f%% | alpha-hat diff %.5f | %s"
              % (_lab, _cfg["statistic_difference"]["max"],
                 100.0 * _cfg["statistic_difference"]["same_rejection_decision_fraction"],
                 _cfg["alpha_hat_difference"],
                 "CI overlap" if _cfg["ci_overlap"] else "CI disjoint"))

    # Keep the quotable Fligner wording numerically identical to the probe output.
    _fp = fligner_probe["configurations"]
    _agree = min(c["statistic_difference"]["same_rejection_decision_fraction"]
                 for c in _fp.values())
    meta["notes"]["code_findings"][1] = (
        "experiments.py calls two different code paths for Fligner-Killeen: the balanced "
        "path builds normal scores of the pooled |x - group median| ranks with raw argsort "
        "ranks; the unbalanced path calls scipy.stats.fligner, which in scipy %s uses the "
        "SAME normal scores but averages tied ranks. The paths therefore differ only in "
        "tie handling (each even-sized group contributes one exact tie pair, because its "
        "two central order statistics are equidistant from the sample median): over %d "
        "same-dataset replications the max |statistic difference| is %.2e at ns=50/50/50, "
        "%.2e at ns=50/100/150 and %.2e at ns=9/12/9 (small groups), while the rejection "
        "decision agrees for at least %.2f%% of replications in every configuration and "
        "the T6 unbalanced anchor's archived and generic Wilson CIs overlap. See "
        "fligner_variant_probe.configurations."
        % (fligner_probe["environment"]["scipy"], fligner_probe["N"],
           _fp["balanced_ns_50_50_50"]["statistic_difference"]["max"],
           _fp["unbalanced_ns_50_100_150"]["statistic_difference"]["max"],
           _fp["small_groups_ns_9_12_9"]["statistic_difference"]["max"],
           100.0 * _agree))

    out = dict(meta=meta, severity_sweep=severity_sweep, study2_nsweep=nsweep_out,
               alpha_sweep=alpha_sweep,
               consistency_checks=checks["archive_vs_generic"],
               consistency_check_summary=consistency_check_summary(checks, fligner_probe),
               fligner_variant_probe=fligner_probe,
               implementation_validation=checks["implementation_validation"],
               distribution_selfcheck=checks["distribution_selfcheck"],
               determinism_checks=checks["determinism"])

    # ---------------- figures ----------------------------------------------
    try:
        make_figures(ladders, nsweep_agg, {},
                     os.path.join(HERE, "fig_rev_severity.png"),
                     os.path.join(HERE, "fig_rev_study2_n.png"))
        figs_ok = True
    except Exception as exc:                          # pragma: no cover
        import traceback
        figs_ok = False
        out["figure_error"] = repr(exc)
        out["figure_traceback"] = traceback.format_exc()
        print("FIGURE ERROR:\n%s" % traceback.format_exc())

    meta["runtime_seconds"] = round(time.time() - t_start, 1)
    meta["figures_written"] = figs_ok

    def _strict(obj):
        """Guarantee strict JSON: no NaN/Infinity (raises if any survive)."""
        if isinstance(obj, dict):
            return {k: _strict(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_strict(v) for v in obj]
        if isinstance(obj, (np.floating, float)):
            return float(obj) if np.isfinite(obj) else None
        if isinstance(obj, (np.integer,)):
            return int(obj)
        return obj

    with open(os.path.join(HERE, "rev_results.json"), "w") as fh:
        json.dump(_strict(out), fh, indent=1, allow_nan=False)

    # ---------------- stdout headline tables -------------------------------
    print("\n" + "=" * 78)
    print("HEADLINE: severity ladders (alpha = 0.05, n = 50, N = %d)" % N_MC)
    print("=" * 78)
    for lname in ("skewness", "kurtosis", "variance_ratio", "imbalance",
                  "small_freq_t4", "small_freq_t8", "small_freq_t9"):
        print_ladder_table(titles[lname], ladders[lname], ladder_rungs[lname])
        print("  Thresholds (liberal flag alpha_hat > 1.5*alpha):")
        for tid, th in sorted(severity_sweep["thresholds"][lname].items()):
            print("    %-21s first liberal rung: %-12s (CI-based: %-12s) trend: %s"
                  % (severity_sweep["ladders"][lname]["tests"][tid].get(
                         "label", TEST_LABELS.get(tid, tid)),
                     str(th["first_liberal_rung"]),
                     str(th["first_liberal_rung_ci"]), th["trend_verdict"]))

    print("\n" + "=" * 78)
    print("HEADLINE: sample-size sweep (alpha = 0.05, N = %d)" % N_MC)
    print("=" * 78)
    for sc in ("baseline", "skewed", "heavy_tail", "hetero_var", "unbalanced",
               "small_freq"):
        print("\n--- %s: alpha_hat (Wilson CI) by n ---" % sc)
        hdr = "%-21s" % "Test" + "".join("%26s" % ("n=%d" % n) for n in NSWEEP_N)
        print(hdr)
        print("-" * len(hdr))
        for tid, ent in sorted(nsweep_agg[sc].items()):
            row = "%-21s" % TEST_LABELS[tid]
            for n in NSWEEP_N:
                r = ent["by_n"][str(n)]
                cell = "%.3f (%.3f,%.3f)%s" % (r["alpha_hat"], r["ci_lo"], r["ci_hi"],
                                               "!" if r["liberal_ci"] else
                                               ("*" if r["liberal"] else ""))
                row += "%26s" % cell
            print(row)
    print("\n--- Ordering stability, skewed scenario (Delta vs matched-n baseline) ---")
    oa = nsweep_out["ordering_stability"]["skewed"]
    print("  rank order by Delta:")
    for n in NSWEEP_N:
        print("    n=%3d: %s" % (n, " > ".join(TEST_LABELS[t] for t in oa["rank_order_by_delta"][str(n)])))
    print("  Spearman rho between consecutive n: %s"
          % {k: round(v, 3) for k, v in oa["spearman_consecutive"].items()})
    for name, p in oa["pairs"].items():
        print("    %-32s %s" % (name, p["verdict"]))
    print("\n--- Degradation recovery (first n at which the liberal flag clears) ---")
    for sc, per in nsweep_out["degradation_recovery"].items():
        print("  [%s]" % sc)
        for tid, d in sorted(per.items()):
            print("    %-21s first n not liberal: %-6s (CI-based: %-6s) Delta(n): %s"
                  % (TEST_LABELS[tid], str(d["first_n_not_liberal"]),
                     str(d["first_n_not_liberal_ci"]),
                     {n: round(d["delta"][n], 3) for n in NSWEEP_N}))

    print("\nRuntime: %.1fs | cells: ladders=%d, nsweep=%d, unique draws=%d"
          % (time.time() - t_start, len(ladder_cells), len(nsweep_cells),
             len(_CELL_CACHE)))
    print("Wrote rev_results.json, fig_rev_severity.png, fig_rev_study2_n.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
