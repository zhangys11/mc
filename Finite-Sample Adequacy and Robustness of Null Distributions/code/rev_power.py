#!/usr/bin/env python3
"""
rev_power.py -- Study 4: power companion study for COST-S-26-00605 (major revision).

Referee 1, comment 2: the practical guidelines must distinguish Type I error
robustness (calibration) from overall test performance (discrimination), because
Studies 1-3 evaluate only D_KS and alpha-hat under H0.  This script supplies the
missing power analysis and, crucially, separates two quantities:

  * NOMINAL-alpha power -- rejection rate using the *theoretical* critical value
    (chi2_{k-1}, F_{k-1,k(n-1)}, t_{n-1}, binom(n,.5)).  This is what a
    practitioner actually experiences, but for a test whose size is distorted it
    is CONTAMINATED by that distortion: part of the apparent "power" is spurious
    rejection under H0 (Bartlett under skewness is the textbook case).

  * SIZE-ADJUSTED power -- rejection rate using a critical value recalibrated to
    the empirical ceil((1-alpha)N)-th order statistic of the *same* statistic
    under H0, with the same parent distribution, sample size and replication
    count.  This holds the empirical size at alpha for every test and is the
    only quantity on which tests may be compared fairly.

Uncertainty is reported throughout:
  * every rate carries a 95% Wilson score interval;
  * the size-adjusted critical value carries a distribution-free 95%
    order-statistic interval, and the power range induced by that
    critical-value uncertainty is reported separately;
  * pairwise orderings rest on a CI for the *difference* (paired McNemar-layout
    interval, since all tests of a family run on identical data sets,
    cross-checked against Newcombe's unpaired hybrid score interval; the wider
    of the two is used).  Any difference whose interval covers zero is reported
    as "inconclusive" -- never as an ordering.

Design (alpha = 0.05, N = 10,000 replications, seed 2024):

  family "location3"  k = 3 balanced groups, n per group in {20, 50, 100}
                      tests: ANOVA F (T2), Kruskal-Wallis (T3), median test (T9)
  family "location1"  one sample, n in {20, 50, 100}
                      tests: one-sample t (T1), sign test (T7)
  family "scale"      k = 3 balanced groups, n in {20, 50, 100}
                      tests: Bartlett (T5), Fligner-Killeen (T6)
                      alternatives: sd ratio rho in {1.5, 2, 4}
  parents             normal, t(5) rescaled to unit variance, Exp(1)-1, and a
                      10%-contaminated normal 0.9*N(0,1)+0.1*N(0,5^2), all with
                      mean 0 and variance 1 so that Cohen's d is comparable.
  effect sizes        d in {0.2, 0.5, 0.8}; for k = 3, equally spaced means at
                      (-d/2, 0, +d/2) so that (max mu_j - min mu_j)/sigma = d.

Outputs (see docs/DESIGN_CONTRACT.md, Section 6):
    rev_results_power.json   key "study4_power" (+ top-level "recommendation_matrix")
    fig_rev_power.png

Run:  python3 rev_power.py                 # full run, N = 10,000
      python3 rev_power.py --nmc 500       # fast smoke test (too noisy to interpret)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import platform
import time
import zlib

import numpy as np
import scipy
from scipy import stats

# ----------------------------------------------------------------------------
# Global configuration
# ----------------------------------------------------------------------------
SEED = 2024
ALPHA = 0.05
N_MC = 10000
Z95 = 1.959963984540054

# Output location. Every script in this package writes its results file to the package
# root by default, so the manuscript's verification suite finds it in one place.  Pass
# ``--outdir DIR`` (registered with argparse in ``main``) to redirect, which is what the
# reproduction vignette does when it runs a script in a scratch directory.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = _ROOT
WORKDIR = _ROOT


def _apply_outdir(path):
    """Redirect the results directory, creating it if necessary.

    Args:
        path: directory to write results into; may be relative to the working directory.

    Returns:
        str: the absolute path that results will be written to.
    """
    global OUT_DIR, WORKDIR
    OUT_DIR = WORKDIR = os.path.abspath(path)
    os.makedirs(OUT_DIR, exist_ok=True)
    return OUT_DIR
JSON_PATH = os.path.join(OUT_DIR, 'rev_results_power.json')
FIG_PATH = os.path.join(OUT_DIR, 'fig_rev_power.png')

NS = (20, 50, 100, 200)
EFFECTS = (0.2, 0.5, 0.8)          # Cohen's d
# sd(max)/sd(min) for the scale family.  1.5, 2 and 4 are the ratios requested
# by the revision plan; 1.25 is added because at n = 50 the larger ratios
# saturate both tests at power ~ 1, where the comparison carries no information.
RHOS = (1.25, 1.5, 2.0, 4.0)
K3 = 3                             # number of groups in the k-group families
DISTS = ('normal', 'tdist', 'expon', 'contam')

# Cells in which the *maintained null hypothesis itself* is false, so that the
# test is not merely miscalibrated but invalid: the sign test tests
# median(X) = 0, whereas Exp(1)-1 has median log(2) - 1 = -0.307.
NULL_INVALID = {
    ('T7_sign_test', 'expon'): (
        'H0: median(X) = 0 is false for Exp(1)-1 (median log(2)-1 = -0.307), so the '
        'sign test is not merely miscalibrated but invalid here; its alpha-hat and '
        'both power columns are uninterpretable as calibration or discrimination'),
}

# Published Study 2 null rejection rates (experiment_results.json, n = 50,
# N = 10,000) used as an end-to-end calibration reference for this file.
ARCHIVE_NULL = {
    'T1_t_test':          {'normal': 0.0520, 'expon': 0.0660, 'tdist': 0.0450},
    'T2_anova':           {'normal': 0.0461, 'expon': 0.0434, 'tdist': 0.0477},
    'T3_kruskal_wallis':  {'uniform': 0.0460, 'expon': 0.0510, 'tdist': 0.0490},
    'T5_bartlett':        {'normal': 0.0479, 'expon': 0.4288, 'tdist': 0.2760},
    'T6_fligner_killeen': {'normal': 0.0410, 'expon': 0.1191, 'tdist': 0.0425},
    'T7_sign_test':       {'normal': 0.0352, 'expon': 0.0307, 'tdist': 0.0312},
    'T9_median_test':     {'uniform': 0.0434, 'expon': 0.0457, 'tdist': 0.0453},
}

# teanmate t1-new-tests owns core.py (docs/DESIGN_CONTRACT.md, Section 2); import it when it
# exists, otherwise reproduce the two helpers locally so this script never
# depends on another member's write scope.
try:                                                            # pragma: no cover
    import core as _core
    _CORE_SOURCE = 'core.py'
except Exception:                                               # pragma: no cover
    _core = None
    _CORE_SOURCE = None

try:
    try:
        import experiments as _archive
        _ARCHIVE_SOURCE = 'experiments.py'
    except ModuleNotFoundError:
        # repository layout: the archive generator ships as code/archive_experiments.py
        import archive_experiments as _archive
        _ARCHIVE_SOURCE = 'archive_experiments.py'
except Exception:                                               # pragma: no cover
    _archive = None
    _ARCHIVE_SOURCE = None


# ----------------------------------------------------------------------------
# Statistical helpers
# ----------------------------------------------------------------------------
def wilson_ci_from_rate(phat, N, z=Z95):
    """95% Wilson score interval for an empirical rate phat from N trials."""
    phat = float(phat)
    if N <= 0 or not np.isfinite(phat):
        return (float('nan'), float('nan'))
    denom = 1.0 + z * z / N
    center = (phat + z * z / (2.0 * N)) / denom
    half = (z * np.sqrt(phat * (1.0 - phat) / N + z * z / (4.0 * N * N))) / denom
    lo, hi = max(0.0, center - half), min(1.0, center + half)
    # guarantee the interval contains phat exactly (avoids 1e-16 negative
    # error bars when the empirical rate is 0 or 1)
    if phat >= 1.0:
        hi = 1.0
    if phat <= 0.0:
        lo = 0.0
    return (lo, max(lo, hi))


def wilson_ci_from_counts(k, N, z=Z95):
    """95% Wilson interval from a rejection count k out of N."""
    return wilson_ci_from_rate(k / N if N > 0 else float('nan'), N, z)


def newcombe_diff_ci(k1, n1, k2, n2, z=Z95):
    """Newcombe's hybrid score interval (method 10) for p1 - p2, unpaired."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson_ci_from_counts(k1, n1, z)
    l2, u2 = wilson_ci_from_counts(k2, n2, z)
    d = p1 - p2
    lo = d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (max(-1.0, lo), min(1.0, hi))


def paired_diff_ci(b01, b10, N, z=Z95):
    """Continuity-corrected Wald interval for a difference of two proportions
    estimated on the *same* N replications (McNemar layout).

    b01 = #(A rejects, B does not), b10 = #(A does not, B rejects).
    Deliberately conservative (continuity correction, Wald form).
    """
    d = (b01 - b10) / N
    var = (b01 + b10 - (b01 - b10) ** 2 / N) / (N * N)
    se = np.sqrt(max(var, 0.0))
    half = z * se + 1.0 / N
    return (max(-1.0, d - half), min(1.0, d + half))


def order_stat_quantile_ci(sorted_vals, p, alpha=ALPHA):
    """Distribution-free 95% CI for the p-quantile from order statistics:
    P(X_(lo) <= q_p <= X_(hi)) >= 1 - alpha, ranks from the binomial."""
    N = sorted_vals.size
    lo_r = min(max(int(stats.binom.ppf(alpha / 2.0, N, p)) + 1, 1), N)
    hi_r = min(max(int(stats.binom.ppf(1.0 - alpha / 2.0, N, p)) + 1, 1), N)
    return float(sorted_vals[lo_r - 1]), float(sorted_vals[hi_r - 1])


def empirical_critical_value(null_stats, alpha=ALPHA):
    """Critical value whose H0 exceedance rate is (as close as possible to)
    alpha: the ceil((1-alpha) N)-th order statistic."""
    return empirical_quantile(null_stats, 1.0 - alpha)


def empirical_quantile(vals, p):
    """The ceil(p N)-th order statistic of `vals` (an empirical p-quantile)."""
    s = np.sort(np.asarray(vals, dtype=float))
    N = s.size
    idx = min(max(int(np.ceil(p * N)) - 1, 0), N - 1)
    return float(s[idx]), s


def calibration_transform(tid, stat, n):
    """Map a test statistic to the quantity whose upper tail the nominal rule
    actually tests, so that the calibrated rule is the SAME procedure with a
    recalibrated critical value.

    * upper-tail rules (F, chi-square): the statistic itself;
    * two-sided symmetric rules (t): |statistic|;
    * the two-sided sign test: |S - n/2|.

    Working on the transformed quantity keeps the nominal and calibrated
    rejection regions nested: a liberal test (alpha-hat > alpha) necessarily has
    a larger calibrated threshold, hence lower size-adjusted power, and a
    conservative test the reverse.  Splitting alpha/2 into each empirical tail
    instead would break that nesting, because the empirical null need not be
    symmetric.
    """
    rule = TESTS4[tid]['rule']
    if rule == 't_two_sided':
        return np.abs(stat), 'abs(stat)'
    if rule == 'binom_two_sided':
        return np.abs(stat - n / 2.0), 'abs(stat - n/2)'
    return stat, 'stat'


def calibrate_rule(calib_stat, alpha=ALPHA):
    """Upper-tail calibration of the transformed statistic: the threshold is the
    empirical (1-alpha) quantile of `calib_stat` under H0."""
    cv, s = empirical_quantile(calib_stat, 1.0 - alpha)
    cv_lo, cv_hi = order_stat_quantile_ci(s, 1.0 - alpha)
    return dict(cv=cv, cv_ci=[cv_lo, cv_hi])


def calibrated_rejection(calib_stat, calib):
    """Rejection indicator for a calibrated rule, plus the power range induced
    by the 95% CI of the calibrated threshold.

    Returns (reject, band_lo, band_hi); the band is the rejection rate computed
    with the threshold at the extremes of its confidence interval.
    """
    rej = calib_stat > calib['cv']
    band_lo = float(np.mean(calib_stat > calib['cv_ci'][1]))     # larger threshold
    band_hi = float(np.mean(calib_stat > calib['cv_ci'][0]))     # smaller threshold
    return rej, band_lo, band_hi


def _entropy(*keys):
    """Deterministic integer entropy for SeedSequence from arbitrary labels."""
    out = [SEED]
    for key in keys:
        if isinstance(key, (int, np.integer)):
            out.append(int(key))
        elif isinstance(key, (float, np.floating)):
            out.append(int(round(float(key) * 1000)))
        else:
            out.append(int(zlib.crc32(str(key).encode('utf8'))))
    return out


def cell_rng(*keys):
    """Order-independent deterministic RNG for one simulation cell."""
    return np.random.default_rng(np.random.SeedSequence(_entropy(*keys)))


# ----------------------------------------------------------------------------
# Data-generating processes
# ----------------------------------------------------------------------------
def parent_sample(rng, dist, shape):
    """Draw from a parent distribution with mean 0 and variance 1.

    Standardisation makes Cohen's d = shift/sigma comparable across parents:
    t(5) has variance 5/3 and the 10%-contaminated normal has variance
    0.9*1 + 0.1*25 = 3.4, so both are rescaled; Exp(1)-1 already has mean 0 and
    variance 1.
    """
    if dist == 'normal':
        return rng.standard_normal(shape)
    if dist == 'tdist':
        return rng.standard_t(5, shape) / np.sqrt(5.0 / 3.0)
    if dist == 'expon':
        return rng.exponential(1.0, shape) - 1.0
    if dist == 'contam':
        mask = rng.random(shape) < 0.10
        z = rng.standard_normal(shape)
        z[mask] *= 5.0
        return z / np.sqrt(3.4)
    if dist == 'uniform':                # only used by the archive calibration check
        return rng.uniform(-1.0, 1.0, shape)
    raise ValueError(f'unknown parent distribution: {dist}')


def draw_batch(family, dist, n, N, kind, effect, stream):
    """Vectorised draw of N replicated data sets for one cell.

    family : 'location3' | 'location1' | 'scale'
    kind   : 'null' | 'alt'
    effect : None, Cohen's d (location families) or sd ratio rho (scale family)
    """
    rng = cell_rng(family, dist, n, effect if effect is not None else 'null', stream)
    if family == 'location3':
        X = parent_sample(rng, dist, (N, K3, n))
        if kind == 'alt' and effect:
            off = (np.arange(K3) - (K3 - 1) / 2.0) * (effect / (K3 - 1))
            X += off[None, :, None]
        return X
    if family == 'location1':
        X = parent_sample(rng, dist, (N, n))
        if kind == 'alt' and effect:
            X += effect
        return X
    if family == 'scale':
        X = parent_sample(rng, dist, (N, K3, n))
        if kind == 'alt' and effect:
            sds = np.ones(K3)
            sds[-1] = effect
            X *= sds[None, :, None]
        return X
    raise ValueError(family)


# ----------------------------------------------------------------------------
# Test statistics
#
# These are vectorised over the first axis (MC replications) and mirror the
# archive generators in experiments.py exactly; t1-new-tests asked for the
# `_fast` suffix so they cannot be confused with the archive `mc_*`.  The
# same-data equivalence against independent implementations is verified by
# run_equivalence_check().
# ----------------------------------------------------------------------------
def _ordinal_ranks_along_last(a):
    """Ranks 1..m along the last axis.

    The parents used here are continuous, so ties have probability zero and
    ordinal ranks coincide with the average ranks of scipy.stats.rankdata used
    by the archive generator (checked by run_equivalence_check).
    """
    order = np.argsort(a, axis=-1, kind='stable')
    ranks = np.empty(order.shape, dtype=float)
    m = a.shape[-1]
    rr = np.broadcast_to(np.arange(1, m + 1, dtype=float), a.shape)
    np.put_along_axis(ranks, order, rr, axis=-1)
    return ranks


def mc_t_test_fast(X):
    """One-sample t statistic.  X: (N, n).  Archive: experiments.mc_t_test."""
    n = X.shape[-1]
    s = np.maximum(X.std(axis=-1, ddof=1), 1e-300)
    return X.mean(axis=-1) / (s / np.sqrt(n))


def mc_sign_test_fast(X):
    """Number of observations above the null median 0.  X: (N, n).

    Identical to the archive mc_sign_test with median_theory = 0 (normal and
    t-distributed parents); for Exp parents the archive compares against
    log(2) on Exp(1), which is handled explicitly in run_calibration_check.
    """
    return (X > 0.0).sum(axis=-1)


def mc_anova_fast(X):
    """One-way ANOVA F statistic (balanced).  X: (N, k, n)."""
    N, k, n = X.shape
    gm = X.mean(axis=2)
    grand = X.mean(axis=(1, 2))
    sstr = n * ((gm - grand[:, None]) ** 2).sum(axis=1)
    sse = ((X - gm[:, :, None]) ** 2).sum(axis=(1, 2))
    mse = np.maximum(sse / (k * (n - 1)), 1e-300)
    return (sstr / (k - 1)) / mse


def mc_kruskal_wallis_fast(X):
    """Kruskal-Wallis H statistic (balanced).  X: (N, k, n)."""
    N, k, n = X.shape
    ntot = k * n
    ranks = _ordinal_ranks_along_last(X.reshape(N, ntot))
    rbar = ranks.reshape(N, k, n).mean(axis=2)
    return (12.0 / (ntot * (ntot + 1.0))) * n * ((rbar - (ntot + 1.0) / 2.0) ** 2).sum(axis=1)


def mc_median_test_fast(X):
    """Mood's median test statistic.  X: (N, k, n)."""
    N, k, n = X.shape
    ntot = k * n
    flat = X.reshape(N, ntot)
    med = np.median(flat, axis=1, keepdims=True)
    above = flat > med
    a = above.sum(axis=1).astype(float)
    b = ntot - a
    counts = above.reshape(N, k, n).sum(axis=2).astype(float)
    mt = ((counts - n * a[:, None] / ntot) ** 2 / n).sum(axis=1)
    return np.where((a > 0) & (b > 0), (ntot ** 2 / np.maximum(a * b, 1.0)) * mt, 0.0)


def mc_bartlett_fast(X):
    """Bartlett statistic with the standard Bartlett correction.  X: (N, k, n)."""
    N, k, n = X.shape
    ntot = k * n
    v = np.maximum(X.var(axis=2, ddof=1), 1e-300)
    sp2 = np.maximum(((n - 1) * v).sum(axis=1) / (ntot - k), 1e-300)
    num = (ntot - k) * np.log(sp2) - (n - 1) * np.log(v).sum(axis=1)
    den = 1.0 + (1.0 / (3.0 * (k - 1))) * (k / (n - 1) - 1.0 / (ntot - k))
    return num / den


def mc_fligner_killeen_fast(X, rank_convention='ordinal'):
    """Fligner-Killeen statistic: ANOVA on van der Waerden scores of the
    group-median-centred absolute deviations.  X: (N, k, n).

    Mirrors the balanced branch of experiments.mc_fligner_killeen, which ranks
    with np.argsort (ordinal ranks).  For even n the two central order
    statistics are exactly equidistant from the sample median, so the absolute
    deviations contain exact ties; scipy.stats.fligner resolves them with
    mid-ranks (rankdata).  `rank_convention='ordinal'` (default) reproduces the
    archive exactly, 'average' gives the mid-rank variant; the practical impact
    is quantified by fk_tie_ranks_sensitivity().
    """
    N, k, n = X.shape
    ntot = k * n
    med = np.median(X, axis=2, keepdims=True)
    z = np.abs(X - med).reshape(N, ntot)
    if rank_convention == 'average':
        ranks = np.array([stats.rankdata(row) for row in z], dtype=float)
    else:
        ranks = _ordinal_ranks_along_last(z)
    scores = stats.norm.ppf(ranks / (2.0 * (ntot + 1.0)) + 0.5)
    sg = scores.reshape(N, k, n)
    abar = scores.mean(axis=1)
    aibar = sg.mean(axis=2)
    varsq = np.maximum(scores.var(axis=1, ddof=1), 1e-300)
    return (n * ((aibar - abar[:, None]) ** 2).sum(axis=1)) / varsq


def fk_tie_ranks_sensitivity(N=N_MC, n=50):
    """Practical impact of the ordinal versus mid-rank tie convention in the
    Fligner-Killeen statistic (exact ties arise for even n).  Reports the
    change in the H0 rejection rate and in the H1 power at sd ratio 2."""
    rows = []
    for parent in ('normal', 'expon'):
        X0 = draw_batch('scale', parent, n, N, 'null', None, 'fk-tie-h0')
        X1 = draw_batch('scale', parent, n, N, 'alt', 2.0, 'fk-tie-h1')
        for tag, XX in (('H0', X0), ('H1 sd ratio 2', X1)):
            s_ord = mc_fligner_killeen_fast(XX, 'ordinal')
            s_avg = mc_fligner_killeen_fast(XX, 'average')
            cv_ord, _ = empirical_critical_value(s_ord)
            cv_avg, _ = empirical_critical_value(s_avg)
            rows.append(dict(parent=parent, scenario=tag, N=N, n=n,
                             max_stat_difference=float(np.max(np.abs(s_ord - s_avg))),
                             cv_ordinal=cv_ord, cv_average=cv_avg,
                             rate_ordinal=float(np.mean(s_ord > cv_ord)),
                             rate_average=float(np.mean(s_avg > cv_avg)),
                             rate_difference=float(np.mean(s_ord > cv_ord) - np.mean(s_avg > cv_avg))))
    return dict(rows=rows,
                max_rate_difference=float(max(abs(r['rate_difference']) for r in rows)))


# ----------------------------------------------------------------------------
# Test registry and rejection rules
# ----------------------------------------------------------------------------
TESTS4 = {
    'T1_t_test': dict(label='Student t', family='location1',
                      rule='t_two_sided', df=lambda n: n - 1),
    'T7_sign_test': dict(label='Sign test', family='location1',
                         rule='binom_two_sided', df=lambda n: (n, 0.5), discrete=True),
    'T2_anova': dict(label='ANOVA F', family='location3',
                     rule='f', df=lambda n: (K3 - 1, K3 * (n - 1))),
    'T3_kruskal_wallis': dict(label='Kruskal-Wallis', family='location3',
                              rule='chi2', df=lambda n: K3 - 1),
    'T9_median_test': dict(label='Median test', family='location3',
                           rule='chi2', df=lambda n: K3 - 1, discrete=True),
    'T5_bartlett': dict(label='Bartlett', family='scale',
                        rule='chi2', df=lambda n: K3 - 1),
    'T6_fligner_killeen': dict(label='Fligner-Killeen', family='scale',
                               rule='chi2', df=lambda n: K3 - 1),
}

# rules whose nominal critical region is two-sided; the calibrated region must
# be two-sided as well (see calibrate_rule)
TWO_SIDED_RULES = ('t_two_sided', 'binom_two_sided')

FAMILY_TESTS = {
    'location3': ['T2_anova', 'T3_kruskal_wallis', 'T9_median_test'],
    'location1': ['T1_t_test', 'T7_sign_test'],
    'scale': ['T5_bartlett', 'T6_fligner_killeen'],
}

FAST_STAT = {
    'T1_t_test': mc_t_test_fast,
    'T7_sign_test': mc_sign_test_fast,
    'T2_anova': mc_anova_fast,
    'T3_kruskal_wallis': mc_kruskal_wallis_fast,
    'T9_median_test': mc_median_test_fast,
    'T5_bartlett': mc_bartlett_fast,
    'T6_fligner_killeen': mc_fligner_killeen_fast,
}

# Independent reference implementations (scipy), used on identical data to
# verify the vectorised statistics above.  For k-sample inputs the scipy
# functions are axis-aware when given axis=1 (elementwise over replications);
# median_test and fligner are not, and are evaluated replication by replication.
REF_STAT = {
    'T1_t_test': lambda X: stats.ttest_1samp(X, 0.0, axis=-1).statistic,
    'T2_anova': lambda X: stats.f_oneway(*[X[:, j, :] for j in range(X.shape[1])],
                                         axis=1).statistic,
    'T3_kruskal_wallis': lambda X: stats.kruskal(*[X[:, j, :] for j in range(X.shape[1])],
                                                 axis=1).statistic,
    'T9_median_test': lambda X: np.array(
        [stats.median_test(*rows, ties='below').statistic for rows in X]),
    'T5_bartlett': lambda X: stats.bartlett(*[X[:, j, :] for j in range(X.shape[1])],
                                            axis=1).statistic,
    'T6_fligner_killeen': lambda X: np.array(
        [stats.fligner(*rows, center='median').statistic for rows in X]),
    'T7_sign_test': lambda X: (X > 0.0).sum(axis=-1),
}

# Pairs whose power ordering is scientifically interesting.  All pairs run on
# identical data sets (common random numbers), so the paired interval applies.
CONTRAST_PAIRS = [
    ('scale', 'T5_bartlett', 'T6_fligner_killeen'),
    ('location3', 'T2_anova', 'T3_kruskal_wallis'),
    ('location3', 'T3_kruskal_wallis', 'T9_median_test'),
    ('location3', 'T2_anova', 'T9_median_test'),
    ('location1', 'T1_t_test', 'T7_sign_test'),
]


def critical_value_theoretical(tid, n, alpha=ALPHA):
    """The theoretical alpha-level critical value (the archive's rule)."""
    rule, df = TESTS4[tid]['rule'], TESTS4[tid]['df'](n)
    if rule == 't_two_sided':
        return float(stats.t.ppf(1.0 - alpha / 2.0, df))
    if rule == 'f':
        return float(stats.f.ppf(1.0 - alpha, df[0], df[1]))
    if rule == 'chi2':
        return float(stats.chi2.ppf(1.0 - alpha, df))
    if rule == 'binom_two_sided':
        nn, p = df
        return [float(stats.binom.ppf(alpha / 2.0, nn, p)),
                float(stats.binom.ppf(1.0 - alpha / 2.0, nn, p))]
    raise ValueError(rule)


def reject_nominal(stat, rule, df, alpha=ALPHA):
    """Rejection indicator using the theoretical critical value."""
    if rule == 't_two_sided':
        return np.abs(stat) > stats.t.ppf(1.0 - alpha / 2.0, df)
    if rule == 'f':
        return stat > stats.f.ppf(1.0 - alpha, df[0], df[1])
    if rule == 'chi2':
        return stat > stats.chi2.ppf(1.0 - alpha, df)
    if rule == 'binom_two_sided':
        nn, p = df
        lo = int(stats.binom.ppf(alpha / 2.0, nn, p))
        hi = int(stats.binom.ppf(1.0 - alpha / 2.0, nn, p))
        return (stat < lo) | (stat > hi)
    raise ValueError(rule)


# ----------------------------------------------------------------------------
# One cell: H0 calibration + alternative power
# ----------------------------------------------------------------------------
def null_cell(family, dist, n, N):
    """Calibrate every test of `family` on one shared H0 sample."""
    X = draw_batch(family, dist, n, N, 'null', None, 'null')
    out = {}
    for tid in FAMILY_TESTS[family]:
        spec = TESTS4[tid]
        stat = FAST_STAT[tid](X)
        nom = reject_nominal(stat, spec['rule'], spec['df'](n))
        cstat, cstat_label = calibration_transform(tid, stat, n)
        calib = calibrate_rule(cstat)
        adj, _, _ = calibrated_rejection(cstat, calib)
        out[tid] = dict(
            stat=stat,
            alpha_hat=float(np.mean(nom)),
            alpha_ci=list(wilson_ci_from_counts(int(nom.sum()), N)),
            calib=calib,
            calib_label=cstat_label,
            alpha_hat_at_calibrated=float(np.mean(adj)),
        )
    return out


def alt_cell(family, dist, n, N, effect, nulls):
    """Power of every test of `family` under one alternative."""
    X = draw_batch(family, dist, n, N, 'alt', effect, 'alt')
    out = {}
    for tid in FAMILY_TESTS[family]:
        spec = TESTS4[tid]
        stat = FAST_STAT[tid](X)
        nom = reject_nominal(stat, spec['rule'], spec['df'](n))
        cstat, _ = calibration_transform(tid, stat, n)
        adj, band_lo, band_hi = calibrated_rejection(cstat, nulls[tid]['calib'])
        band = [band_lo, band_hi]
        w_lo, w_hi = wilson_ci_from_counts(int(adj.sum()), N)
        out[tid] = dict(
            reject_nominal=nom,
            reject_adjusted=adj,
            power_nominal=float(np.mean(nom)),
            power_nominal_ci=list(wilson_ci_from_counts(int(nom.sum()), N)),
            power_adjusted=float(np.mean(adj)),
            power_adjusted_ci=list(wilson_ci_from_counts(int(adj.sum()), N)),
            power_adjusted_ci_cv=band,
            # conservative envelope: union of the two uncertainty sources,
            # NOT a nominal 95% interval (documented as such)
            power_adjusted_ci_combined=[min(w_lo, band[0]), max(w_hi, band[1])],
        )
    return out


def contrast_row(family, parent, scenario, n, N, test_a, test_b,
                 nom_flags, adj_flags, alpha_nom, alpha_adj):
    """Difference of two rejection rates in one scenario, nominal and adjusted."""
    row = dict(family=family, parent=parent, scenario=scenario, n=n,
               test_a=test_a, test_b=test_b,
               test_a_label=TESTS4[test_a]['label'], test_b_label=TESTS4[test_b]['label'])
    for tag, flags, alphas in (('nominal', nom_flags, alpha_nom),
                               ('size_adjusted', adj_flags, alpha_adj)):
        A, B = flags[test_a], flags[test_b]
        ka, kb = int(A.sum()), int(B.sum())
        b01 = int(np.sum(A & ~B))
        b10 = int(np.sum(~A & B))
        p_lp, p_hp = paired_diff_ci(b01, b10, N)
        lo_u, hi_u = newcombe_diff_ci(ka, N, kb, N)
        lo, hi = min(p_lp, lo_u), max(p_hp, hi_u)          # conservative union
        if lo > 0:
            verdict = f'{test_a}_higher'
        elif hi < 0:
            verdict = f'{test_b}_higher'
        else:
            verdict = 'inconclusive'
        row[tag] = dict(
            p_a=ka / N, p_b=kb / N,
            p_a_ci=list(wilson_ci_from_counts(ka, N)),
            p_b_ci=list(wilson_ci_from_counts(kb, N)),
            diff=ka / N - kb / N,
            diff_ci_paired=[float(p_lp), float(p_hp)],
            diff_ci_newcombe=[float(lo_u), float(hi_u)],
            diff_ci_conservative=[lo, hi],
            b01=b01, b10=b10,
            verdict=verdict,
            alpha_a=alphas[test_a], alpha_b=alphas[test_b],
        )
    return row


# ----------------------------------------------------------------------------
# Main study
# ----------------------------------------------------------------------------
def build_design():
    """List the alternative-data cells of Study 4.

    Returns:
        list of tuples ``(family, dist, n, effect, kind)``, where ``family`` is
        ``'location3'``, ``'location1'`` or ``'scale'``, ``dist`` is a parent-law
        name, ``n`` the sample size, ``effect`` the standardised effect size
        (kind ``'d'``) or the standard-deviation ratio (kind ``'rho'``). The grid
        is the cross product of the module constants.
    """
    design = []
    for family in ('location3', 'location1'):
        for dist in DISTS:
            for n in NS:
                for d in EFFECTS:
                    design.append((family, dist, n, d, 'd'))
    for dist in DISTS:
        for n in NS:
            for rho in RHOS:
                design.append(('scale', dist, n, rho, 'rho'))
    return design


def run_study4(N=N_MC, verbose=True):
    """Run the whole Study 4 grid: null calibration first, then the alternatives.

    Args:
        N: Monte Carlo replications per cell (default 10,000).
        verbose: print per-cell progress.

    Returns:
        dict with ``cells`` (one record per test x parent x n x effect, holding
        nominal and size-adjusted power with combined Wilson intervals),
        ``contrasts`` (paired comparisons between competing tests), ``null_store``
        (the calibrated null cells), ``design`` (the cell list) and ``runtime``
        (seconds).
    """
    t0 = time.time()
    cells, contrasts = [], []
    null_store = {}

    for family in ('location3', 'location1', 'scale'):
        for dist in DISTS:
            for n in NS:
                null_store[(family, dist, n)] = null_cell(family, dist, n, N)
                if verbose:
                    alphas = ', '.join(
                        f"{TESTS4[t]['label']}={null_store[(family, dist, n)][t]['alpha_hat']:.3f}"
                        for t in FAMILY_TESTS[family])
                    print(f'  [H0] {family:9s} {dist:7s} n={n:3d}  {alphas}')
    if verbose:
        print(f'  -- H0 calibration done in {time.time() - t0:.1f}s')

    design = build_design()
    for family, dist, n, effect, kind in design:
        nulls = null_store[(family, dist, n)]
        alts = alt_cell(family, dist, n, N, effect, nulls)
        scenario = f'd={effect}' if kind == 'd' else f'rho={effect}'
        for tid in FAMILY_TESTS[family]:
            spec = TESTS4[tid]
            nc, ac = nulls[tid], alts[tid]
            a_lo, a_hi = nc['alpha_ci']
            size_valid = bool(a_lo <= ALPHA <= a_hi)
            if a_lo > ALPHA:
                distortion = 'liberal'          # nominal-alpha power overstates discrimination
            elif a_hi < ALPHA:
                distortion = 'conservative'     # nominal-alpha power understates discrimination
            else:
                distortion = 'none'
            h0_note = NULL_INVALID.get((tid, dist))
            cells.append(dict(
                family=family, test=tid, test_label=spec['label'], parent=dist, n=n,
                scenario=scenario, effect=effect, effect_kind=kind, N=N, alpha=ALPHA,
                alpha_hat=nc['alpha_hat'], alpha_hat_ci=nc['alpha_ci'],
                size_valid=size_valid,
                size_distortion=distortion,
                # the nominal-alpha power of a size-distorted test is not a clean
                # measure of discrimination; flag it mechanically
                nominal_power_contaminated=not size_valid,
                null_hypothesis_valid=h0_note is None,
                null_hypothesis_note=h0_note,
                critical_value_theoretical=critical_value_theoretical(tid, n),
                calibration_statistic=nc['calib_label'],
                cv_adjusted=nc['calib']['cv'], cv_adjusted_ci=nc['calib']['cv_ci'],
                # for the two-sided rules the calibrated threshold, expressed back
                # on the original statistic's scale: reject when
                # stat < lower or stat > upper
                cv_adjusted_equivalent_thresholds=(
                    [((n / 2.0) if spec['rule'] == 'binom_two_sided' else 0.0)
                     - nc['calib']['cv'],
                     ((n / 2.0) if spec['rule'] == 'binom_two_sided' else 0.0)
                     + nc['calib']['cv']]
                    if spec['rule'] in TWO_SIDED_RULES else None),
                alpha_hat_at_adjusted_cv=nc['alpha_hat_at_calibrated'],
                power_nominal=ac['power_nominal'], power_nominal_ci=ac['power_nominal_ci'],
                power_adjusted=ac['power_adjusted'],
                power_adjusted_ci=ac['power_adjusted_ci'],
                power_adjusted_ci_cv=ac['power_adjusted_ci_cv'],
                power_adjusted_ci_combined=ac['power_adjusted_ci_combined'],
            ))

        for fam, ta, tb in CONTRAST_PAIRS:
            if fam != family:
                continue
            contrasts.append(contrast_row(
                family, dist, scenario, n, N, ta, tb,
                {t: alts[t]['reject_nominal'] for t in (ta, tb)},
                {t: alts[t]['reject_adjusted'] for t in (ta, tb)},
                {t: nulls[t]['alpha_hat'] for t in (ta, tb)},
                {t: nulls[t]['alpha_hat_at_calibrated'] for t in (ta, tb)}))

    if verbose:
        print(f'  -- alternatives done; total {time.time() - t0:.1f}s')
    return dict(cells=cells, contrasts=contrasts, null_store=null_store,
                design=design, runtime=time.time() - t0)


# ----------------------------------------------------------------------------
# Verification 1: same-data equivalence of the vectorised statistics against
# independent reference implementations (scipy), plus z-tests against the
# published archive rates.
# ----------------------------------------------------------------------------
def run_equivalence_check(N=2000, n=50):
    """Two checks, both requested by t1-new-tests:

    (a) SAME-DATA equivalence: for each test and parent, draw one batch and
        compare the vectorised statistic against an independent implementation
        (scipy.stats) on identical data.  Reports max |difference|.
    (b) ALPHA-HAT comparison against the archive generator running in-process
        (same configuration as the published Study 2, n = 50).  Because the
        archive uses the legacy global RNG stream and this script uses
        independent PCG64 streams, the comparison is a two-sample Monte Carlo
        comparison; the discrepancy is therefore standardised
        (z = diff / sqrt(2 p (1-p) / N)), not compared to a fixed tolerance
        that would be inside Monte Carlo noise.
    """
    same_data = []
    for parent in ('normal', 'expon', 'tdist'):
        batches = {
            'location3': draw_batch('location3', parent, n, N, 'null', None, f'equiv-{parent}'),
            'location1': draw_batch('location1', parent, n, N, 'null', None, f'equiv1-{parent}'),
            'scale': draw_batch('scale', parent, n, N, 'null', None, f'equiv2-{parent}'),
        }
        for tid, fam, data in [(t, 'location3', batches['location3'])
                               for t in FAMILY_TESTS['location3']] + \
                              [(t, 'location1', batches['location1'])
                               for t in FAMILY_TESTS['location1']] + \
                              [(t, 'scale', batches['scale'])
                               for t in FAMILY_TESTS['scale']]:
            mine = FAST_STAT[tid](data)
            ref = np.asarray(REF_STAT[tid](data), dtype=float)
            same_data.append(dict(
                test=tid, label=TESTS4[tid]['label'], parent=parent, N=N,
                max_abs_diff=float(np.max(np.abs(mine - ref))),
                max_rel_diff=float(np.max(np.abs(mine - ref) / np.maximum(np.abs(ref), 1e-12))),
            ))

    archive_alpha = []
    if _archive is not None:
        archive_specs = {
            'T1_t_test': lambda N: _archive.mc_t_test(n, N, dist='normal'),
            'T2_anova': lambda N: _archive.mc_anova(K3, n, N, dist='normal')[0],
            'T3_kruskal_wallis': lambda N: _archive.mc_kruskal_wallis(K3, n, N, dist='uniform'),
            'T5_bartlett': lambda N: _archive.mc_bartlett(K3, n, N, dist='normal'),
            'T9_median_test': lambda N: _archive.mc_median_test(K3, n, N, dist='uniform'),
            'T6_fligner_killeen': lambda N: _archive.mc_fligner_killeen(K3, n, N, dist='normal'),
            'T7_sign_test': lambda N: _archive.mc_sign_test(n, N, dist='normal'),
        }
        for tid, fn in archive_specs.items():
            family = TESTS4[tid]['family']
            parent = 'uniform' if tid in ('T3_kruskal_wallis', 'T9_median_test') else 'normal'
            np.random.seed(SEED)                       # archive's global-stream convention
            a_stat = fn(N)
            a_rej = reject_nominal(a_stat, TESTS4[tid]['rule'], TESTS4[tid]['df'](n))
            a_hat = float(np.mean(a_rej))
            X = draw_batch(family, parent, n, N, 'null', None, f'equiv2-{tid}')
            f_rej = reject_nominal(FAST_STAT[tid](X), TESTS4[tid]['rule'], TESTS4[tid]['df'](n))
            f_hat = float(np.mean(f_rej))
            p_pool = 0.5 * (a_hat + f_hat)
            se = np.sqrt(2.0 * p_pool * (1.0 - p_pool) / N) if 0 < p_pool < 1 else np.nan
            z = (f_hat - a_hat) / se if se and np.isfinite(se) and se > 0 else np.nan
            archive_alpha.append(dict(
                test=tid, label=TESTS4[tid]['label'], parent=parent, N=N,
                alpha_fast=f_hat, alpha_archive_generator=a_hat,
                abs_diff=abs(f_hat - a_hat), z=float(z),
                pass_3sd=bool(np.isfinite(z) and abs(z) <= 3.0),
                published_study2_value=ARCHIVE_NULL[tid].get('uniform' if parent == 'uniform'
                                                             else 'normal'),
            ))
    core_check = None
    if _core is not None:
        try:                                     # pragma: no cover - depends on core.py
            probes = [(0.05, 10000), (0.037, 2500), (0.43, 10000), (0.001, 500)]
            diffs = []
            for phat, nn in probes:
                c_lo, c_hi = _core.wilson_ci(phat, nn)
                m_lo, m_hi = wilson_ci_from_rate(phat, nn)
                diffs.append(max(abs(c_lo - m_lo), abs(c_hi - m_hi)))
            core_check = dict(module=_CORE_SOURCE, probes=len(probes),
                              max_wilson_ci_difference=float(max(diffs)),
                              agrees=bool(max(diffs) < 1e-12))
        except Exception as exc:                 # pragma: no cover
            core_check = dict(module=_CORE_SOURCE, agrees=None, error=repr(exc))

    return dict(
        same_data=same_data,
        max_abs_diff_all=float(max(r['max_abs_diff'] for r in same_data)),
        core_module_check=core_check,
        archive_alpha=archive_alpha,
        archive_n_pass=sum(r['pass_3sd'] for r in archive_alpha),
        archive_n_compared=len(archive_alpha),
        note=('Same-data differences are floating-point level; the archive alpha-hat '
              'comparison is a two-sample Monte Carlo comparison and is judged by a '
              '3-sigma z-criterion, because a fixed tolerance of 0.005 is inside the '
              'Monte Carlo noise at N=2000 for large rates (Bartlett under skewness).'),
    )


# ----------------------------------------------------------------------------
# Verification 2: end-to-end reproduction of the published Study 2 rates
# ----------------------------------------------------------------------------
def run_calibration_check(N=N_MC, n=50):
    """Reproduce the manuscript's Study 2 null alpha-hat values (n = 50) with
    the vectorised generators, using the same parent distributions the archive
    used for each scenario."""
    rows = []
    parents = {
        'T1_t_test': [('normal', 'baseline'), ('expon', 'skewed'), ('tdist', 'heavy_tail')],
        'T2_anova': [('normal', 'baseline'), ('expon', 'skewed'), ('tdist', 'heavy_tail')],
        'T3_kruskal_wallis': [('uniform', 'baseline (uniform parent)'), ('expon', 'skewed'),
                              ('tdist', 'heavy_tail')],
        'T5_bartlett': [('normal', 'baseline'), ('expon', 'skewed'), ('tdist', 'heavy_tail')],
        'T6_fligner_killeen': [('normal', 'baseline'), ('expon', 'skewed'), ('tdist', 'heavy_tail')],
        'T7_sign_test': [('normal', 'baseline'), ('expon', 'skewed'), ('tdist', 'heavy_tail')],
        'T9_median_test': [('uniform', 'baseline (uniform parent)'), ('expon', 'skewed'),
                           ('tdist', 'heavy_tail')],
    }
    for tid, plist in parents.items():
        family, spec = TESTS4[tid]['family'], TESTS4[tid]
        for dist, tag in plist:
            rng = cell_rng('calib', tid, dist, n)
            if family == 'location1':
                if tid == 'T7_sign_test' and dist == 'expon':
                    # the archive's mc_sign_test compares Exp(1) against log(2)
                    X = rng.exponential(1.0, (N, n))
                    stat = (X > np.log(2.0)).sum(axis=1)
                else:
                    X = parent_sample(rng, dist, (N, n))
                    stat = FAST_STAT[tid](X)
            else:
                X = (rng.uniform(-1.0, 1.0, (N, K3, n)) if dist == 'uniform'
                     else parent_sample(rng, dist, (N, K3, n)))
                stat = FAST_STAT[tid](X)
            rej = reject_nominal(stat, spec['rule'], spec['df'](n))
            a = float(np.mean(rej))
            k = int(rej.sum())
            lo, hi = wilson_ci_from_counts(k, N)
            ref = ARCHIVE_NULL[tid][dist]
            p_pool = 0.5 * (a + ref)
            se = np.sqrt(2.0 * p_pool * (1.0 - p_pool) / N) if 0 < p_pool < 1 else np.nan
            z = (a - ref) / se if np.isfinite(se) and se > 0 else np.nan
            rows.append(dict(test=tid, label=spec['label'], parent=dist, scenario=tag,
                             alpha_hat=a, alpha_hat_ci=[lo, hi], n_used=N,
                             archive_alpha_hat=ref, difference=a - ref,
                             z_two_sample=float(z),
                             consistent_3sd=bool(np.isfinite(z) and abs(z) <= 3.0)))
    n_ok = sum(r['consistent_3sd'] for r in rows)
    return dict(n_compared=len(rows), n_consistent_3sd=n_ok,
                all_consistent=bool(n_ok == len(rows)), rows=rows)


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def fmt_ci(ci):
    """Format a confidence interval as ``[lo,hi]`` with three decimals.

    Args:
        ci: pair ``(lo, hi)``.

    Returns:
        str, e.g. ``'[0.123,0.988]'``.
    """
    return f'[{ci[0]:.3f},{ci[1]:.3f}]'


def scale_rows(cells, n, N):
    """Bartlett vs Fligner-Killeen at one sample size, all parents and sd ratios."""
    rows = []
    for dist in DISTS:
        for rho in RHOS:
            sc = f'rho={rho}'
            b = next((c for c in cells if c['test'] == 'T5_bartlett' and c['parent'] == dist
                      and c['n'] == n and c['scenario'] == sc), None)
            f = next((c for c in cells if c['test'] == 'T6_fligner_killeen' and c['parent'] == dist
                      and c['n'] == n and c['scenario'] == sc), None)
            if b is None or f is None:
                continue
            rows.append(dict(parent=dist, sd_ratio=rho, n=n, N=N,
                             bartlett=dict(alpha_hat=b['alpha_hat'], alpha_ci=b['alpha_hat_ci'],
                                           power_nominal=b['power_nominal'],
                                           power_nominal_ci=b['power_nominal_ci'],
                                           power_adjusted=b['power_adjusted'],
                                           power_adjusted_ci=b['power_adjusted_ci_combined'],
                                           contaminated=b['nominal_power_contaminated']),
                             fligner_killeen=dict(alpha_hat=f['alpha_hat'], alpha_ci=f['alpha_hat_ci'],
                                                  power_nominal=f['power_nominal'],
                                                  power_nominal_ci=f['power_nominal_ci'],
                                                  power_adjusted=f['power_adjusted'],
                                                  power_adjusted_ci=f['power_adjusted_ci_combined'],
                                                  contaminated=f['nominal_power_contaminated']),
                             nominal_difference=b['power_nominal'] - f['power_nominal'],
                             adjusted_difference=b['power_adjusted'] - f['power_adjusted']))
    return rows


def print_headline(cells, contrasts, n=50, N=N_MC):
    """Print the headline Bartlett-versus-Fligner-Killeen table.

    Args:
        cells: cell records returned by :func:`run_study4`.
        contrasts: contrast records returned by :func:`run_study4`.
        n: sample size of the table (default 50).
        N: replication count, used in the column labels.

    Returns:
        None (writes to stdout).
    """
    print('\n' + '=' * 112)
    print(f'HEADLINE: Bartlett (T5) vs Fligner-Killeen (T6), n = {n}, alpha = 0.05')
    print('=' * 112)
    print(f"{'parent':7s} {'sd-ratio':>8s} | {'a-hatB':>7s} {'a-hatF':>7s} | "
          f"{'powNom B (95% CI)':>21s} {'powNom F (95% CI)':>21s} | "
          f"{'powAdj B (95% CI)':>21s} {'powAdj F (95% CI)':>21s}")
    for row in scale_rows(cells, n, N):
        b, f = row['bartlett'], row['fligner_killeen']
        print(f"{row['parent']:7s} {row['sd_ratio']:8.1f} | {b['alpha_hat']:7.3f} {f['alpha_hat']:7.3f} | "
              f"{b['power_nominal']:6.3f} {fmt_ci(b['power_nominal_ci']):>13s} "
              f"{f['power_nominal']:6.3f} {fmt_ci(f['power_nominal_ci']):>13s} | "
              f"{b['power_adjusted']:6.3f} {fmt_ci(b['power_adjusted_ci']):>13s} "
              f"{f['power_adjusted']:6.3f} {fmt_ci(f['power_adjusted_ci']):>13s}"
              f"{'  [CONTAMINATED]' if b['contaminated'] else ''}")
    print('\nPaired difference Bartlett - Fligner-Killeen '
          '(conservative union of paired McNemar and unpaired Newcombe 95% CIs):')
    print(f"{'parent':7s} {'sd-ratio':>8s} | {'diff nominal (95% CI)':>34s} {'verdict':>22s} | "
          f"{'diff size-adjusted (95% CI)':>34s} {'verdict':>22s}")
    for c in contrasts:
        if c['test_a'] != 'T5_bartlett' or c['n'] != n:
            continue
        dn, da = c['nominal'], c['size_adjusted']
        rho = c['scenario'].split('=')[1]
        print(f"{c['parent']:7s} {float(rho):8.1f} | "
              f"{dn['diff']:+8.3f} {fmt_ci(dn['diff_ci_conservative']):>24s} {dn['verdict']:>22s} | "
              f"{da['diff']:+8.3f} {fmt_ci(da['diff_ci_conservative']):>24s} {da['verdict']:>22s}")


# ----------------------------------------------------------------------------
# Rank-reversal summary (Bartlett vs Fligner-Killeen)
# ----------------------------------------------------------------------------
def rank_reversal_summary(cells, contrasts):
    """Does the Type I error ranking (and the nominal-alpha power ranking) agree
    with the size-adjusted power ranking?  Computed, never asserted."""
    index = {(c['test'], c['parent'], c['n'], c['scenario']): c for c in cells}
    rows = []
    for con in contrasts:
        if con['test_a'] != 'T5_bartlett' or con['test_b'] != 'T6_fligner_killeen':
            continue
        key = ('T5_bartlett', con['parent'], con['n'], con['scenario'])
        b = index.get(key)
        f = index.get(('T6_fligner_killeen',) + key[1:])
        if b is None or f is None:
            continue
        nom, adj = con['nominal'], con['size_adjusted']
        rows.append(dict(
            parent=con['parent'], n=con['n'], sd_ratio=b['effect'], scenario=con['scenario'],
            alpha_bartlett=b['alpha_hat'], alpha_bartlett_ci=b['alpha_hat_ci'],
            alpha_fligner=f['alpha_hat'], alpha_fligner_ci=f['alpha_hat_ci'],
            size_excess_bartlett=b['alpha_hat'] - f['alpha_hat'],
            size_valid_bartlett=b['size_valid'], size_valid_fligner=f['size_valid'],
            nominal_difference=nom['diff'], nominal_difference_ci=nom['diff_ci_conservative'],
            nominal_verdict=nom['verdict'],
            size_adjusted_difference=adj['diff'],
            size_adjusted_difference_ci=adj['diff_ci_conservative'],
            size_adjusted_verdict=adj['verdict'],
            verdict_reversal=bool(nom['verdict'] != adj['verdict']),
            nominal_favours_bartlett=bool(nom['verdict'] == 'T5_bartlett_higher'),
            adjusted_favours_fligner=bool(adj['verdict'] == 'T6_fligner_killeen_higher'),
        ))
    summary = dict(
        n_cells=len(rows),
        n_verdict_reversals=sum(r['verdict_reversal'] for r in rows),
        n_bartlett_size_inflated=sum((not r['size_valid_bartlett']) for r in rows),
        n_nominal_favours_bartlett_but_adjusted_does_not=sum(
            r['nominal_favours_bartlett'] and not
            (r['size_adjusted_verdict'] == 'T5_bartlett_higher') for r in rows),
        n_adjusted_favours_fligner=sum(r['adjusted_favours_fligner'] for r in rows),
        n_size_valid_bartlett_cells=sum(r['size_valid_bartlett'] for r in rows),
    )
    summary['n_size_valid_bartlett_cells_where_bartlett_wins_adjusted'] = sum(
        1 for r in rows if r['size_valid_bartlett']
        and r['size_adjusted_verdict'] == 'T5_bartlett_higher')
    summary['n_size_valid_bartlett_cells_where_fligner_wins_adjusted'] = sum(
        1 for r in rows if r['size_valid_bartlett']
        and r['size_adjusted_verdict'] == 'T6_fligner_killeen_higher')
    return dict(rows=rows, summary=summary)


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
def make_figure(res, path=FIG_PATH, n=50, N=N_MC):
    """Render the six-panel power figure.

    Args:
        res: result dict returned by :func:`run_study4`.
        path: output path (default ``fig_rev_power.png``).
        n: sample size shown in the location and scale panels.
        N: replication count used in the title.

    Returns:
        str: the path that was written.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'figure.dpi': 150, 'savefig.dpi': 150})

    cells = res['cells']
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.4))
    ax = axes.ravel()

    def cell(test, parent, nn, scenario):
        for c in cells:
            if c['test'] == test and c['parent'] == parent and c['n'] == nn \
                    and c['scenario'] == scenario:
                return c
        return None

    scale_style = {
        'T5_bartlett': dict(color='#c0392b', marker='o', label='Bartlett'),
        'T6_fligner_killeen': dict(color='#2471a3', marker='s', label='Fligner-Killeen'),
    }
    # ---- (a)-(c): scale family, power vs sd ratio, nominal vs size-adjusted
    for ax_i, parent in enumerate(('normal', 'expon', 'tdist')):
        a = ax[ax_i]
        for tid, st in scale_style.items():
            for mode, alpha_line, lw, mfc in (('nominal', 0.40, 1.3, 'none'),
                                              ('adjusted', 0.95, 1.8, st['color'])):
                xs, ys, es = [], [], []
                for rho in RHOS:
                    c = cell(tid, parent, n, f'rho={rho}')
                    if c is None:
                        continue
                    xs.append(rho)
                    if mode == 'nominal':
                        ys.append(c['power_nominal'])
                        ci = c['power_nominal_ci']
                    else:
                        ys.append(c['power_adjusted'])
                        ci = c['power_adjusted_ci_combined']
                    es.append([ys[-1] - ci[0], ci[1] - ys[-1]])
                es = np.maximum(np.array(es).T, 0.0)
                a.errorbar(xs, ys, yerr=es, color=st['color'], marker=st['marker'],
                           ls=st['ls'] if 'ls' in st else '-', lw=lw, ms=4.5,
                           alpha=alpha_line, mfc=mfc, capsize=2.5, elinewidth=0.9,
                           label=f"{st['label']} ({'nominal' if mode == 'nominal' else 'size-adj.'})")
        # multiplicative spacing: a log axis separates the sd ratios 1.25, 1.5,
        # 2 and 4 without the tick labels colliding
        a.set_xscale('log')
        a.set_xticks(list(RHOS))
        a.set_xticklabels([f'{r:g}' for r in RHOS])
        a.minorticks_off()
        a.set_xlim(min(RHOS) / 1.12, max(RHOS) * 1.10)
        a.set_xlabel(r'sd ratio $\sigma_{\max}/\sigma_{\min}$')
        a.set_ylabel(f'power (n = {n} per group)')
        a.set_title(f'({"abc"[ax_i]}) Scale tests, {parent} parent', fontsize=9)
        a.set_ylim(-0.02, 1.02)
        a.grid(alpha=0.25, lw=0.5)
        # place the legend where the curves are not: at sd ratio >= 2 both tests
        # are above 0.9, so the lower-right corner is empty in every panel
        a.legend(fontsize=6.8, loc='lower right', framealpha=0.92, ncol=2,
                 columnspacing=0.7, handlelength=1.8, borderpad=0.3)
    ax[0].axhline(ALPHA, color='grey', ls=':', lw=1.0)
    ax[0].text(2.0, ALPHA + 0.04, r'$\alpha=0.05$', fontsize=7, color='grey')
    ax[0].text(0.02, 0.98, 'shaded = size-adjusted\nfaded = nominal-$\\alpha$',
               transform=ax[0].transAxes, fontsize=6.5, color='0.35', va='top',
               bbox=dict(fc='white', ec='none', alpha=1.0, pad=2.0))

    # ---- (d) calibration-discrimination plane (scale family, sd ratio 2)
    a = ax[3]
    for tid, st in scale_style.items():
        xs, xe, ys, ye = [], [], [], []
        for parent in DISTS:
            c = cell(tid, parent, n, 'rho=2.0')
            if c is None:
                continue
            xs.append(c['alpha_hat'])
            xe.append([c['alpha_hat'] - c['alpha_hat_ci'][0], c['alpha_hat_ci'][1] - c['alpha_hat']])
            ys.append(c['power_nominal'])
            ye.append([c['power_nominal'] - c['power_nominal_ci'][0],
                       c['power_nominal_ci'][1] - c['power_nominal']])
            a.annotate('', xy=(c['alpha_hat'], c['power_adjusted']),
                       xytext=(c['alpha_hat'], c['power_nominal']),
                       arrowprops=dict(arrowstyle='->', color=st['color'], lw=1.1, alpha=0.85))
            a.plot([c['alpha_hat']], [c['power_adjusted']], marker=st['marker'],
                   color=st['color'], mfc='white', ms=6, ls='none')
        a.errorbar(xs, ys, xerr=np.maximum(np.array(xe).T, 0.0),
                   yerr=np.maximum(np.array(ye).T, 0.0), fmt=st['marker'],
                   color=st['color'], ms=5, capsize=2.5, elinewidth=0.9, ls='none',
                   label=st['label'])
    a.axvline(ALPHA, color='grey', ls=':', lw=1.0)
    a.axhline(ALPHA, color='grey', ls=':', lw=1.0)
    a.set_xlabel(r'empirical size $\hat{\alpha}$ under $H_0$')
    a.set_ylabel('power at sd ratio 2')
    a.set_title(f'(d) Calibration vs discrimination, scale tests (n = {n})', fontsize=9)
    a.grid(alpha=0.25, lw=0.5)
    a.legend(fontsize=6.8, loc='lower right', framealpha=0.92, handlelength=1.8, borderpad=0.3)
    a.text(0.02, 0.02, 'filled = nominal-$\\alpha$ power\nhollow = size-adjusted power',
           transform=a.transAxes, fontsize=6.5, color='0.35', ha='left', va='bottom',
           bbox=dict(fc='white', ec='none', alpha=1.0, pad=2.0))

    # ---- (e) three-group location tests, size-adjusted power vs Cohen's d
    a = ax[4]
    loc_style = {'T2_anova': ('#1e8449', 'o', 'ANOVA'),
                 'T3_kruskal_wallis': ('#7d3c98', '^', 'Kruskal-Wallis'),
                 'T9_median_test': ('#b9770e', 'v', 'Median')}
    for parent, ls in (('normal', '-'), ('tdist', '--'), ('contam', ':')):
        for tid, (col, mk, lab) in loc_style.items():
            xs, ys, es = [], [], []
            for d in EFFECTS:
                c = cell(tid, parent, n, f'd={d}')
                if c is None:
                    continue
                xs.append(d)
                ys.append(c['power_adjusted'])
                ci = c['power_adjusted_ci_combined']
                es.append([ys[-1] - ci[0], ci[1] - ys[-1]])
            a.errorbar(xs, ys, yerr=np.maximum(np.array(es).T, 0.0), color=col, marker=mk,
                       ls=ls, lw=1.4,
                       ms=4.5, alpha=0.9, capsize=2.5, elinewidth=0.9,
                       label=f'{lab} ({parent})')
    a.set_xlabel(r"Cohen's d (range of group means / $\sigma$)")
    a.set_ylabel(f'size-adjusted power (n = {n} per group)')
    a.set_title('(e) Three-group location tests', fontsize=9)
    a.set_xticks(list(EFFECTS))
    a.set_ylim(-0.02, 1.02)
    a.grid(alpha=0.25, lw=0.5)
    # legend in the empty lower-right corner: at d = 0.8 every curve is above 0.8
    a.legend(fontsize=6.0, loc='lower left', bbox_to_anchor=(0.40, 0.01), framealpha=0.92,
             ncol=2, handlelength=1.6, columnspacing=0.8, borderpad=0.3)

    # ---- (f) one-sample location tests
    a = ax[5]
    one_style = {'T1_t_test': ('#1a5276', 'o', 't-test'),
                 'T7_sign_test': ('#943126', 'v', 'Sign test')}
    for parent, ls in (('normal', '-'), ('tdist', '--'), ('contam', ':')):
        for tid, (col, mk, lab) in one_style.items():
            xs, ys, es = [], [], []
            for d in EFFECTS:
                c = cell(tid, parent, n, f'd={d}')
                if c is None:
                    continue
                xs.append(d)
                ys.append(c['power_adjusted'])
                ci = c['power_adjusted_ci_combined']
                es.append([ys[-1] - ci[0], ci[1] - ys[-1]])
            a.errorbar(xs, ys, yerr=np.maximum(np.array(es).T, 0.0), color=col, marker=mk,
                       ls=ls, lw=1.4,
                       ms=4.5, alpha=0.9, capsize=2.5, elinewidth=0.9,
                       label=f'{lab} ({parent})')
    ex = cell('T7_sign_test', 'expon', n, 'd=0.5')
    if ex is not None:
        a.text(0.02, 0.02, f"Expo(1)$-1$ omitted: null median $\\neq 0$,\n"
                           f"$\\hat{{\\alpha}}$ = {ex['alpha_hat']:.2f} for the sign test",
               transform=a.transAxes, fontsize=6.5, color='0.35', va='bottom')
    a.set_xlabel(r"Cohen's d (shift / $\sigma$)")
    a.set_ylabel(f'size-adjusted power (n = {n})')
    a.set_title('(f) One-sample location tests', fontsize=9)
    a.set_xticks(list(EFFECTS))
    a.set_ylim(-0.02, 1.02)
    a.grid(alpha=0.25, lw=0.5)
    a.legend(fontsize=6.1, loc='center right', framealpha=0.92, ncol=1,
             handlelength=1.6, borderpad=0.3)

    fig.suptitle(r'Study 4: nominal-$\alpha$ versus size-adjusted power '
                 f'(N = {N:,} replications, $\\alpha$ = 0.05, 95% CIs)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.962))
    fig.savefig(path)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def main():
    """Command-line entry point of the power study.

    Parses ``--nmc``, ``--no-fig`` and ``--skip-verification``, runs the
    same-data equivalence check, the in-process calibration check against the
    archived generators, and the Study 4 grid; writes
    ``rev_results_power.json`` (and ``fig_rev_power.png`` unless suppressed) and
    prints the headline tables.

    Returns:
        None.
    """
    ap = argparse.ArgumentParser(description='Study 4 power companion (R1 comment 2)')
    ap.add_argument("--outdir", default=None,
                    help="directory for results files (default: the package root); "
                         "created if it does not exist")
    ap.add_argument('--nmc', type=int, default=N_MC, help='MC replications (default 10000)')
    ap.add_argument('--no-fig', action='store_true', help='skip figure generation')
    ap.add_argument('--skip-verification', action='store_true',
                    help='skip the equivalence / calibration checks')
    args = ap.parse_args()
    if args.outdir:
        _apply_outdir(args.outdir)
    nmc = args.nmc
    t_start = time.time()

    print('=' * 112)
    print(f'STUDY 4: power companion | alpha={ALPHA} | N={nmc} | seed={SEED} | '
          f'core.py={_CORE_SOURCE} archive={_ARCHIVE_SOURCE}')
    print('=' * 112)

    verification = dict(equivalence=None, calibration=None)
    if not args.skip_verification:
        print('\n(a) SAME-DATA equivalence of vectorised statistics vs scipy (N=2000, n=50)')
        equiv = run_equivalence_check(N=2000, n=50)
        for r in equiv['same_data']:
            print(f"    {r['label']:16s} {r['parent']:7s} max|diff|={r['max_abs_diff']:.3e} "
                  f"max rel={r['max_rel_diff']:.3e}")
        print(f"    --> max |difference| across all tests = {equiv['max_abs_diff_all']:.3e}")
        print('(b) alpha-hat vs archive generators in-process (N=2000, n=50, 3-sigma rule)')
        for r in equiv['archive_alpha']:
            flag = 'OK ' if r['pass_3sd'] else 'DIFF'
            print(f"    {flag} {r['label']:16s} {r['parent']:8s} "
                  f"fast={r['alpha_fast']:.4f} archive={r['alpha_archive_generator']:.4f} "
                  f"|d|={r['abs_diff']:.4f} z={r['z']:+.2f}")
        print(f"    --> {equiv['archive_n_pass']}/{equiv['archive_n_compared']} within 3 sigma")
        print(f'\n(c) end-to-end check vs published Study 2 rates (N={nmc}, n=50)')
        calib = run_calibration_check(N=nmc, n=50)
        for r in calib['rows']:
            flag = 'OK ' if r['consistent_3sd'] else 'DIFF'
            print(f"    {flag} {r['label']:16s} {r['parent']:8s} "
                  f"alpha_hat={r['alpha_hat']:.4f} {fmt_ci(r['alpha_hat_ci'])} "
                  f"vs published {r['archive_alpha_hat']:.4f} "
                  f"(diff {r['difference']:+.4f}, z={r['z_two_sample']:+.2f})")
        print(f"    --> {calib['n_consistent_3sd']}/{calib['n_compared']} published rates "
              f"reproduced within 3 sigma")
        print('\n(d) Fligner-Killeen tie convention: ordinal ranks (archive) vs mid-ranks (scipy)')
        fk_tie = fk_tie_ranks_sensitivity(N=nmc, n=50)
        for r in fk_tie['rows']:
            print(f"    {r['parent']:7s} {r['scenario']:16s} "
                  f"max|dstat|={r['max_stat_difference']:.2e} "
                  f"rate(ordinal)={r['rate_ordinal']:.4f} rate(average)={r['rate_average']:.4f} "
                  f"diff={r['rate_difference']:+.5f}")
        print(f"    --> max |rate difference| = {fk_tie['max_rate_difference']:.5f} "
              f"(exact ties arise because n is even)")
        verification = dict(equivalence=equiv, calibration=calib, fk_tie_ranks=fk_tie)

    print('\nH0 calibration + alternatives')
    res = run_study4(N=nmc, verbose=True)

    headline = scale_rows(res['cells'], 50, nmc)

    rec = [{k: c[k] for k in ('family', 'test', 'test_label', 'parent', 'n', 'scenario',
                              'effect', 'effect_kind', 'alpha', 'alpha_hat', 'alpha_hat_ci',
                              'size_valid', 'size_distortion', 'nominal_power_contaminated',
                              'cv_adjusted',
                              'calibration_statistic', 'cv_adjusted', 'cv_adjusted_ci',
                              'cv_adjusted_equivalent_thresholds',
                              'alpha_hat_at_adjusted_cv', 'power_nominal',
                              'power_nominal_ci', 'power_adjusted', 'power_adjusted_ci',
                              'power_adjusted_ci_cv', 'power_adjusted_ci_combined')}
           for c in res['cells']]

    # ---- computed (not asserted) reading of the Bartlett vs Fligner-Killeen case
    scale_contrasts = [c for c in res['contrasts'] if c['test_a'] == 'T5_bartlett']
    flips = [c for c in scale_contrasts
             if c['nominal']['verdict'] != c['size_adjusted']['verdict']]
    misleading = [c for c in flips if c['nominal']['verdict'] == 'T5_bartlett_higher']
    counterexamples = [dict(parent=c['parent'], n=c['n'], scenario=c['scenario'],
                            verdict_nominal=c['nominal']['verdict'],
                            verdict_size_adjusted=c['size_adjusted']['verdict'],
                            diff_nominal=c['nominal']['diff'],
                            diff_nominal_ci=c['nominal']['diff_ci_conservative'],
                            diff_adjusted=c['size_adjusted']['diff'],
                            diff_adjusted_ci=c['size_adjusted']['diff_ci_conservative'],
                            alpha_bartlett=c['nominal']['alpha_a'],
                            alpha_fligner=c['nominal']['alpha_b']) for c in flips]
    notes = [
        (f"{c['scenario']}, parent={c['parent']}, n={c['n']}: nominal-alpha verdict "
         f"'{c['nominal']['verdict']}' but size-adjusted verdict "
         f"'{c['size_adjusted']['verdict']}' (nominal diff {c['nominal']['diff']:+.3f} "
         f"[{c['nominal']['diff_ci_conservative'][0]:+.3f}, "
         f"{c['nominal']['diff_ci_conservative'][1]:+.3f}]; size-adjusted diff "
         f"{c['size_adjusted']['diff']:+.3f} "
         f"[{c['size_adjusted']['diff_ci_conservative'][0]:+.3f}, "
         f"{c['size_adjusted']['diff_ci_conservative'][1]:+.3f}])")
        for c in flips]
    notes.append(
        'The nominal-alpha power of a test whose size CI excludes 0.05 is flagged '
        'nominal_power_contaminated = true; such numbers must not be used to rank tests.')
    notes.append(
        'For the discrete statistics (sign test, median test) the calibrated critical '
        'value attains a size strictly below 0.05 (see adjusted_size_attainment), so '
        'their size-adjusted power is mildly conservative and part of any gap to a '
        'continuous competitor reflects discreteness, not discrimination.')
    notes.append(
        'power_adjusted_ci_combined is the union of the Wilson interval for the power '
        'estimate and the power range induced by the 95% CI of the calibrated critical '
        'value; it is a conservative envelope, not a nominal 95% interval.')

    reversals = rank_reversal_summary(res['cells'], res['contrasts'])

    # Size attained by the calibrated critical value.  For continuous statistics
    # it is exactly alpha; for discrete statistics (sign test, median test) the
    # calibrated critical value attains a size <= alpha, so part of any power
    # deficit of those tests is conservatism rather than discrimination.
    attainment = {}
    for c in res['cells']:
        attainment.setdefault(c['test'], []).append(c['alpha_hat_at_adjusted_cv'])
    adjusted_size_attainment = {
        t: dict(label=TESTS4[t]['label'], discrete=bool(TESTS4[t].get('discrete', False)),
                attained_min=min(v), attained_max=max(v),
                attains_nominal=bool(abs(max(v) - ALPHA) < 1e-9), n_cells=len(v))
        for t, v in attainment.items()}

    meta = dict(
        seed=SEED, alpha=ALPHA, N_mc=nmc,
        numpy=np.__version__, scipy=scipy.__version__, python=platform.python_version(),
        git_hash='n/a (not a git repository)',
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        script='rev_power.py', core_module=_CORE_SOURCE, archive_module=_ARCHIVE_SOURCE,
        runtime_seconds=round(time.time() - t_start, 1),
    )

    payload = dict(
        meta=meta,
        study4_power=dict(
            design=dict(
                alpha=ALPHA, N_mc=nmc, seed=SEED, sample_sizes=list(NS),
                effects_d=list(EFFECTS), sd_ratios=list(RHOS), parents=list(DISTS),
                effect_definition=("Cohen's d = (max mu_j - min mu_j)/sigma for the k=3 "
                                   "families (equally spaced means at (-d/2, 0, +d/2)); "
                                   "d = delta/sigma for the one-sample family; the scale "
                                   "family uses the sd ratio sigma_max/sigma_min"),
                parents_definition=dict(
                    normal='N(0,1)',
                    tdist='t(5)/sqrt(5/3) (unit variance)',
                    expon='Exp(1)-1 (mean 0, variance 1, skewness 2)',
                    contam='0.9*N(0,1)+0.1*N(0,25), rescaled to unit variance'),
                size_adjustment=('the size-adjusted test is the SAME procedure with its '
                                 'critical value recalibrated to attain alpha exactly: the '
                                 'threshold is the empirical (1-alpha) quantile of the '
                                 'calibration statistic under H0 with the same parent, n '
                                 'and N, where the calibration statistic is the statistic '
                                 'itself for upper-tail rules (F, chi-square), |statistic| '
                                 'for the two-sided t test and |S - n/2| for the two-sided '
                                 'sign test. This keeps the nominal and calibrated '
                                 'rejection regions nested, so a liberal test necessarily '
                                 'loses power when size-adjusted and a conservative test '
                                 'necessarily gains it. A distribution-free 95% '
                                 'order-statistic CI for the threshold is reported together '
                                 'with the power range it induces'),
                common_random_numbers=('all tests of a family are evaluated on identical '
                                       'simulated data sets; pairwise differences therefore '
                                       'use a paired McNemar-layout interval, cross-checked '
                                       'against Newcombe unpaired, and the wider of the two '
                                       'is used for the verdict'),
                comparison_rule=('a power ordering is claimed only when the conservative '
                                 '95% difference CI excludes 0; otherwise the verdict is '
                                 '"inconclusive"'),
                size_distortion_direction=('size_distortion is "liberal" when the 95% CI of '
                                           'alpha-hat lies entirely above 0.05 (nominal-alpha '
                                           'power overstates discrimination), "conservative" '
                                           'when entirely below (nominal-alpha power '
                                           'understates discrimination), else "none"; '
                                           'nominal_power_contaminated is true whenever the '
                                           'CI excludes 0.05 in either direction'),
            ),
            tests={k: dict(label=v['label'], family=v['family'], rule=v['rule'],
                           discrete=bool(v.get('discrete', False)),
                           theoretical_df=str(v['df'](50))) for k, v in TESTS4.items()},
            verification=verification,
            cells=res['cells'],
            contrasts=res['contrasts'],
            headline_scale_table=headline,
            rank_reversal=reversals,
            adjusted_size_attainment=adjusted_size_attainment,
            recommendation_matrix=rec,
            counterexamples=counterexamples,
            notes=notes,
        ),
        recommendation_matrix=rec,
    )

    with open(JSON_PATH, 'w') as fh:
        json.dump(payload, fh, indent=1)

    if not args.no_fig:
        make_figure(res, N=nmc)

    print_headline(res['cells'], res['contrasts'], n=50, N=nmc)
    print(f'\nWrote {JSON_PATH}')
    if not args.no_fig:
        print(f'Wrote {FIG_PATH}')
    s = reversals['summary']
    print(f'Cells: {len(res["cells"])}, contrasts: {len(res["contrasts"])}')
    print(f'Scale-family verdict reversals (nominal vs size-adjusted): '
          f'{s["n_verdict_reversals"]}/{s["n_cells"]}')
    print(f'  of which nominal favours Bartlett but size-adjusted does not: '
          f'{s["n_nominal_favours_bartlett_but_adjusted_does_not"]}')
    print(f'  cells where Bartlett is size-valid: {s["n_size_valid_bartlett_cells"]} '
          f'(Bartlett wins size-adjusted in '
          f'{s["n_size_valid_bartlett_cells_where_bartlett_wins_adjusted"]}, '
          f'Fligner-Killeen in '
          f'{s["n_size_valid_bartlett_cells_where_fligner_wins_adjusted"]})')
    print(f'Total runtime: {time.time() - t_start:.1f}s')


if __name__ == '__main__':
    main()
