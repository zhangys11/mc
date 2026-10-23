"""
Edgeworth mechanisms in finite-sample null calibration — single experiment script.

This is the complete computational companion to `paper.md`. It is self-contained
(numpy, scipy, matplotlib) and has two halves:

  Part 1 — the eleven-test Monte Carlo suite
      Ten univariate tests (t, ANOVA F, Kruskal-Wallis, Pearson chi-square GOF,
      Bartlett, Fligner-Killeen, sign, Cochran's Q, median, Hotelling T^2) plus
      Box's M (T11): null-distribution convergence rates (KS distance vs n),
      Type I error under distributional violations, the three corrections
      (B*, mid-p, empirical lattice critical values), Monte Carlo precision
      requirements, and the real-data illustrations.

  Part 2 — exact Box's M calibration under a multivariate normal parent
      The matrix-Dirichlet representation of Box's M, its closed-form cumulant
      generating function, exact cumulants, and critical values by the
      Lugannani-Rice saddlepoint or by deterministic inversion of the CGF.
      Also the exact size of the classical chi-square and Bartlett-corrected
      rules, i.e. what the classical reference costs.

Every number quoted in the paper is produced by this script: the simulation
tables from Part 1 (committed results file), the exact tables from Part 2 (no
simulation at all).

Usage
-----
    python3 experiment.py                    # full run: Part 1 + Part 2, writes results and figures
    python3 experiment.py --exact-null       # only Part 2 (seconds), merges into the results file
    python3 experiment.py --figures-only     # regenerate figures from the committed results file
    python3 experiment.py --smoke            # tiny-N smoke test of both halves (minutes)

Outputs: results/experiment_results.json, results/exact_null_reference.json
(input, optional), figures/*.png.
"""

import numpy as np
from scipy import stats
from scipy.special import comb
from scipy import special as sp
from scipy.optimize import brentq
import warnings
import json
import os
import time

SEED = 2024
# Bootstrap benchmark settings.  The parametric bootstrap is reported as a
# benchmark only; it enters no table and no figure, so a moderate number of
# resamples is enough to place it to within about 0.015.
BOOT_B_TEST = 200
BOOT_B_BOOT = 100
np.random.seed(SEED)
# Numerical warnings are evidence, not noise, and are deliberately left enabled.
# Where a configuration legitimately has no defined statistic (e.g. Cochran's Q
# on a saturated subject x condition table) the code returns NaN explicitly
# instead of suppressing the warning.

# Resolved from this file, so the script runs from any working directory.
# Outputs: results/ for the committed results files, figures/ for the figures
# that paper.md embeds.
_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_HERE, 'results')
FIG_DIR = os.path.join(_HERE, 'figures')
EXACT_REF = os.path.join(OUT_DIR, 'exact_null_reference.json')


def _rel(path):
    """Project-relative path for logs and the committed results file.

    The manuscript is double-blind, so no local absolute path may appear in
    run.log or experiment_results.json.  Paths outside the project are reduced
    to their basename."""
    try:
        rel = os.path.relpath(os.path.abspath(str(path)), _HERE)
    except (TypeError, ValueError):
        return os.path.basename(str(path))
    return rel if not rel.startswith('..') else os.path.basename(str(path))

try:
    import matplotlib
    if __name__ == '__main__':
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FixedFormatter, FormatStrFormatter
    plt.rcParams.update({'font.size': 9, 'figure.dpi': 150, 'savefig.dpi': 150,
                         'figure.figsize': (10, 6)})
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ============================================================
# SECTION 1: Test Statistic Generators
# ============================================================

def mc_t_test(n, N, dist='normal', **kw):
    if dist == 'normal':
        X = np.random.normal(0, 1, (N, n))
    elif dist == 'expon':
        X = np.random.exponential(1, (N, n)) - 1
    elif dist == 'tdist':
        X = np.random.standard_t(5, (N, n))
    else:
        X = np.random.normal(0, 1, (N, n))
    xbar = X.mean(axis=1)
    s = X.std(axis=1, ddof=1)
    s[s == 0] = 1e-12
    return xbar / (s / np.sqrt(n))

def mc_anova(k, n, N, dist='normal', hetero=False, unbalanced=False):
    if unbalanced:
        ns = [n, 2*n, 3*n][:k]
        while len(ns) < k:
            ns.append(n)
        max_n = max(ns)
        F_arr = np.empty(N)
        for i in range(N):
            groups = []
            for j in range(k):
                if dist == 'normal':
                    sd = 2.0 if (hetero and j == k-1) else 1.0
                    groups.append(np.random.normal(0, sd, ns[j]))
                elif dist == 'expon':
                    groups.append(np.random.exponential(1, ns[j]) - 1)
                elif dist == 'tdist':
                    groups.append(np.random.standard_t(5, ns[j]))
                else:
                    groups.append(np.random.normal(0, 1, ns[j]))
            all_data = np.concatenate(groups)
            grand_mean = all_data.mean()
            n_total = sum(ns)
            sstr = sum(ns[j] * (groups[j].mean() - grand_mean)**2 for j in range(k))
            sse = sum(((groups[j] - groups[j].mean())**2).sum() for j in range(k))
            mstr = sstr / (k - 1)
            mse = sse / (n_total - k)
            F_arr[i] = mstr / max(mse, 1e-15)
        df1, df2 = k-1, sum(ns)-k
        return F_arr, df1, df2
    else:
        if dist == 'normal':
            if hetero:
                sds = np.ones(k)
                sds[-1] = 2.0
                X = np.random.normal(0, 1, (N, k, n)) * sds[None, :, None]
            else:
                X = np.random.normal(0, 1, (N, k, n))
        elif dist == 'expon':
            X = np.random.exponential(1, (N, k, n)) - 1
        elif dist == 'tdist':
            X = np.random.standard_t(5, (N, k, n))
        else:
            X = np.random.normal(0, 1, (N, k, n))
        gm = X.mean(axis=2)
        grand = X.mean(axis=(1,2))
        SSTR = n * ((gm - grand[:, None])**2).sum(axis=1)
        MSTR = SSTR / (k-1)
        SSE = ((X - gm[:,:,None])**2).sum(axis=(1,2))
        MSE = SSE / (k*(n-1))
        MSE[MSE == 0] = 1e-15
        df1, df2 = k-1, k*(n-1)
        return MSTR / MSE, df1, df2

def mc_kruskal_wallis(k, n, N, dist='uniform', hetero=False, unbalanced=False):
    Hs = np.empty(N)
    if unbalanced:
        ns = [n, 2*n, 3*n][:k]
        while len(ns) < k:
            ns.append(n)
    else:
        ns = [n] * k
    n_total = sum(ns)
    for i in range(N):
        groups = []
        for j in range(k):
            if dist == 'uniform':
                sd = 2.0 if (hetero and j == k-1) else 1.0
                groups.append(np.random.uniform(-sd, sd, ns[j]))
            elif dist == 'normal':
                sd = 2.0 if (hetero and j == k-1) else 1.0
                groups.append(np.random.normal(0, sd, ns[j]))
            elif dist == 'expon':
                groups.append(np.random.exponential(1, ns[j]) - 1)
            elif dist == 'tdist':
                groups.append(np.random.standard_t(5, ns[j]))
            else:
                groups.append(np.random.uniform(-1, 1, ns[j]))
        all_data = np.concatenate(groups)
        ranks = stats.rankdata(all_data)
        idx = 0
        H = 0
        for j in range(k):
            R_j = ranks[idx:idx+ns[j]].sum()
            H += R_j**2 / ns[j]
            idx += ns[j]
        H = 12 / (n_total * (n_total + 1)) * H - 3 * (n_total + 1)
        Hs[i] = H
    return Hs

def mc_chisq_gof(k, sample_size, N, small_expected=False):
    if small_expected:
        p_theory = np.ones(k) / k
        p_theory[0] = 3.0 / (sample_size + 3*(k-1))
        p_theory[1:] = (1 - p_theory[0]) / (k - 1)
    else:
        p_theory = np.ones(k) / k
    samples = np.random.multinomial(sample_size, p_theory, size=N)
    expected = sample_size * p_theory
    chisqs = ((samples - expected)**2 / expected).sum(axis=1)
    return chisqs

def mc_bartlett(k, n, N, dist='normal', unbalanced=False):
    if unbalanced:
        ns = [n, 2*n, 3*n][:k]
        while len(ns) < k:
            ns.append(n)
        BTs = np.empty(N)
        for i in range(N):
            vars_i = []
            for j in range(k):
                if dist == 'normal':
                    g = np.random.normal(0, 1, ns[j])
                elif dist == 'expon':
                    g = np.random.exponential(1, ns[j]) - 1
                elif dist == 'tdist':
                    g = np.random.standard_t(5, ns[j])
                else:
                    g = np.random.normal(0, 1, ns[j])
                vars_i.append(g.var(ddof=1))
            n_total = sum(ns)
            sp2 = sum((ns[j]-1)*vars_i[j] for j in range(k)) / (n_total - k)
            if sp2 <= 0:
                BTs[i] = 0
                continue
            num = (n_total - k) * np.log(sp2) - sum((ns[j]-1)*np.log(max(vars_i[j],1e-15)) for j in range(k))
            den = 1 + (1/(3*(k-1))) * (sum(1/(ns[j]-1) for j in range(k)) - 1/(n_total-k))
            BTs[i] = num / den
        return BTs
    else:
        if dist == 'normal':
            X = np.random.normal(0, 1, (N, k, n))
        elif dist == 'expon':
            X = np.random.exponential(1, (N, k, n)) - 1
        elif dist == 'tdist':
            X = np.random.standard_t(5, (N, k, n))
        else:
            X = np.random.normal(0, 1, (N, k, n))
        vars_i = X.var(axis=2, ddof=1)
        n_total = k * n
        sp2 = ((n-1) * vars_i).sum(axis=1) / (n_total - k)
        sp2[sp2 <= 0] = 1e-15
        vars_i[vars_i <= 0] = 1e-15
        num = (n_total - k) * np.log(sp2) - (n-1) * np.log(vars_i).sum(axis=1)
        den = 1 + (1/(3*(k-1))) * (k/(n-1) - 1/(n_total-k))
        return num / den


def mc_bartlett_corrected(k, n, N, dist='normal', unbalanced=False, return_b=False):
    """kurtosis-corrected Bartlett: B* = B / (1 + pooled_excess_kurtosis / 2).

    Data are generated once and both B and the pooled excess kurtosis are computed
    from them, so B and B* are paired by construction.  With return_b=True the pair
    (B, B*) is returned instead of B* alone, which is what the paired study needs."""
    if not unbalanced:
        if dist == 'normal':
            X = np.random.normal(0, 1, (N, k, n))
        elif dist == 'expon':
            X = np.random.exponential(1, (N, k, n)) - 1
        elif dist == 'tdist':
            X = np.random.standard_t(5, (N, k, n))
        else:
            X = np.random.normal(0, 1, (N, k, n))
        vars_i = X.var(axis=2, ddof=1)
        n_total = k * n
        sp2 = ((n-1) * vars_i).sum(axis=1) / (n_total - k)
        sp2[sp2 <= 0] = 1e-15
        vars_i[vars_i <= 0] = 1e-15
        num = (n_total - k) * np.log(sp2) - (n-1) * np.log(vars_i).sum(axis=1)
        den = 1 + (1/(3*(k-1))) * (k/(n-1) - 1/(n_total-k))
        B = num / den
        mu = X.mean(axis=2, keepdims=True)
        m2 = ((X - mu)**2).mean(axis=2)
        m4 = ((X - mu)**4).mean(axis=2)
        g2_j = m4 / np.maximum(m2**2, 1e-15) - 3
        g2_p = g2_j.mean(axis=1)
    else:
        ns = [n, 2*n, 3*n][:k]
        while len(ns) < k:
            ns.append(n)
        B = np.empty(N)
        g2_p = np.empty(N)
        for i in range(N):
            groups = []
            for j in range(k):
                if dist == 'normal':
                    g = np.random.normal(0, 1, ns[j])
                elif dist == 'expon':
                    g = np.random.exponential(1, ns[j]) - 1
                elif dist == 'tdist':
                    g = np.random.standard_t(5, ns[j])
                else:
                    g = np.random.normal(0, 1, ns[j])
                groups.append(g)
            vars_i = np.array([g.var(ddof=1) for g in groups])
            n_total = sum(ns)
            sp2 = sum((ns[j]-1)*vars_i[j] for j in range(k)) / (n_total - k)
            if sp2 <= 0:
                B[i] = 0; g2_p[i] = 0; continue
            num = (n_total - k)*np.log(sp2) - sum((ns[j]-1)*np.log(max(vars_i[j],1e-15)) for j in range(k))
            den = 1 + (1/(3*(k-1)))*(sum(1/(ns[j]-1) for j in range(k)) - 1/(n_total-k))
            B[i] = num / den
            g2s = []
            for g in groups:
                m2 = ((g - g.mean())**2).mean()
                m4 = ((g - g.mean())**4).mean()
                g2s.append(m4/(m2**2)-3 if m2>1e-15 else 0)
            g2_p[i] = np.mean(g2s)
    denom = np.maximum(1 + g2_p / 2, 1e-6)
    return (B, B / denom) if return_b else B / denom


def fligner_killeen_stats(X):
    """Vectorised Fligner-Killeen statistic for a (N, k, n) array.

    Matches scipy.stats.fligner on each replication: the absolute deviations from
    the group medians are ranked with average ranks (ties matter for even n, where
    the two central order statistics are equidistant from the median)."""
    N, k, n = X.shape
    Ntot = k * n
    med = np.median(X, axis=2, keepdims=True)
    z = np.abs(X - med).reshape(N, Ntot)
    ranks = stats.rankdata(z, axis=1)
    scores = stats.norm.ppf(ranks / (2 * (Ntot + 1.0)) + 0.5)
    sg = scores.reshape(N, k, n)
    abar = scores.mean(axis=1)
    aibar = sg.mean(axis=2)
    varsq = scores.var(axis=1, ddof=1)
    varsq[varsq <= 0] = 1e-15
    return (n * ((aibar - abar[:, None]) ** 2).sum(axis=1)) / varsq


def mc_fligner_killeen(k, n, N, dist='normal', unbalanced=False):
    if not unbalanced:
        if dist == 'normal':
            X = np.random.normal(0, 1, (N, k, n))
        elif dist == 'expon':
            X = np.random.exponential(1, (N, k, n)) - 1
        elif dist == 'tdist':
            X = np.random.standard_t(5, (N, k, n))
        else:
            X = np.random.normal(0, 1, (N, k, n))
        return fligner_killeen_stats(X)
    FKs = np.empty(N)
    ns = [n, 2*n, 3*n][:k]
    while len(ns) < k:
        ns.append(n)
    for i in range(N):
        groups = []
        for j in range(k):
            if dist == 'normal':
                groups.append(np.random.normal(0, 1, ns[j]))
            elif dist == 'expon':
                groups.append(np.random.exponential(1, ns[j]) - 1)
            elif dist == 'tdist':
                groups.append(np.random.standard_t(5, ns[j]))
            else:
                groups.append(np.random.normal(0, 1, ns[j]))
        stat, _ = stats.fligner(*groups)
        FKs[i] = stat
    return FKs

def mc_sign_test(n, N, dist='normal'):
    if dist == 'expon':
        X = np.random.exponential(1, (N, n))
        median_theory = np.log(2)
    elif dist == 'normal':
        X = np.random.normal(0, 1, (N, n))
        median_theory = 0.0
    elif dist == 'tdist':
        X = np.random.standard_t(5, (N, n))
        median_theory = 0.0
    elif dist == 'uniform':
        X = np.random.uniform(0, 1, (N, n))
        median_theory = 0.5
    elif dist == 'discrete':
        X = np.random.randint(1, 7, (N, n)).astype(float)
        median_theory = 3.5
    else:
        X = np.random.normal(0, 1, (N, n))
        median_theory = 0.0
    N_plus = (X > median_theory).sum(axis=1)
    return N_plus

def mc_sign_pvalues(n, N, dist='normal'):
    """Paired sign-test p-values from one set of N+ counts.

    Returns (p_exact, p_midp), both two-sided, computed from the same replications.
    """
    N_plus = mc_sign_test(n, N, dist=dist).astype(int)
    cdf = stats.binom.cdf
    sf = stats.binom.sf
    pmf = stats.binom.pmf
    p_exact = 2.0 * np.minimum(cdf(N_plus, n, 0.5), sf(N_plus - 1, n, 0.5))
    p_midp = 2.0 * np.minimum(cdf(N_plus - 1, n, 0.5) + 0.5 * pmf(N_plus, n, 0.5),
                             sf(N_plus, n, 0.5) + 0.5 * pmf(N_plus, n, 0.5))
    return np.clip(p_exact, 0.0, 1.0), np.clip(p_midp, 0.0, 1.0)


def mc_cochran_q(k, n, N, p=0.5):
    """N Monte Carlo draws of Cochran's Q for a randomized-block design.

    Design: n subjects (blocks) x k dichotomous conditions.  There is **no**
    unequal-group-size version of this test -- Q is defined on a complete
    subject x condition table, so the former ``unbalanced`` argument did nothing
    and has been removed (a block design cannot be "unbalanced" in the one-way
    sense used by the other tests; see ``run_study2``).

    Replications whose denominator vanishes (all subject totals equal, e.g. all
    responses identical) have no defined statistic and are returned as NaN.
    Callers must drop them and report the fraction that was defined.
    """
    X = np.random.binomial(1, p, (N, n, k))
    col_sums = X.sum(axis=1)
    row_sums = X.sum(axis=2)
    num = (k - 1) * (k * (col_sums ** 2).sum(axis=1) - col_sums.sum(axis=1) ** 2)
    den = k * row_sums.sum(axis=1) - (row_sums ** 2).sum(axis=1)
    out = np.full(N, np.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out

def mc_median_test(k, n, N, dist='uniform', hetero=False, unbalanced=False):
    MTs = np.empty(N)
    if unbalanced:
        ns = [n, 2*n, 3*n][:k]
        while len(ns) < k:
            ns.append(n)
    else:
        ns = [n] * k
    n_total = sum(ns)
    for i in range(N):
        groups = []
        for j in range(k):
            if dist == 'uniform':
                sd = 2.0 if (hetero and j == k-1) else 1.0
                groups.append(np.random.uniform(-sd, sd, ns[j]))
            elif dist == 'normal':
                sd = 2.0 if (hetero and j == k-1) else 1.0
                groups.append(np.random.normal(0, sd, ns[j]))
            elif dist == 'expon':
                groups.append(np.random.exponential(1, ns[j]) - 1)
            elif dist == 'tdist':
                groups.append(np.random.standard_t(5, ns[j]))
            else:
                groups.append(np.random.uniform(-1, 1, ns[j]))
        all_data = np.concatenate(groups)
        grand_median = np.median(all_data)
        a = (all_data > grand_median).sum()
        b = n_total - a
        if a == 0 or b == 0:
            MTs[i] = 0
            continue
        idx = 0
        mt = 0
        for j in range(k):
            O1j = (groups[j] > grand_median).sum()
            mt += (O1j - ns[j]*a/n_total)**2 / ns[j]
            idx += ns[j]
        MTs[i] = (n_total**2 / (a * b)) * mt
    return MTs

def mc_hotelling_t2(p_dim, n, N, dist='normal'):
    T2s = np.empty(N)
    for i in range(N):
        if dist == 'normal':
            X = np.random.normal(0, 1, (n, p_dim))
        elif dist == 'expon':
            X = np.random.exponential(1, (n, p_dim)) - 1
        elif dist == 'tdist':
            X = np.random.standard_t(5, (n, p_dim))
        else:
            X = np.random.normal(0, 1, (n, p_dim))
        xbar = X.mean(axis=0)
        S = np.cov(X.T)
        if p_dim == 1:
            S = np.array([[S]]) if np.ndim(S) == 0 else S
        try:
            S_inv = np.linalg.inv(S)
            T2s[i] = n * xbar @ S_inv @ xbar
        except np.linalg.LinAlgError:
            T2s[i] = 0
    F_trans = (n - p_dim) / (p_dim * (n - 1)) * T2s
    return F_trans, p_dim, n - p_dim

def mc_clt(n, N, dist='expon'):
    if dist == 'expon':
        X = np.random.exponential(1, (N, n))
        mu, sigma = 1.0, 1.0
    elif dist == 'uniform':
        X = np.random.uniform(-1, 1, (N, n))
        mu, sigma = 0.0, np.sqrt(1/3)
    elif dist == 'poisson':
        X = np.random.poisson(1, (N, n)).astype(float)
        mu, sigma = 1.0, 1.0
    elif dist == 'bernoulli':
        X = np.random.binomial(1, 0.5, (N, n)).astype(float)
        mu, sigma = 0.5, 0.5
    elif dist == 'tampered_dice':
        probs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]
        vals = np.array([1,2,3,4,5,6], dtype=float)
        X = np.random.choice(vals, size=(N, n), p=probs)
        mu = (vals * probs).sum()
        sigma = np.sqrt(((vals - mu)**2 * probs).sum())
    else:
        X = np.random.exponential(1, (N, n))
        mu, sigma = 1.0, 1.0
    xbars = X.mean(axis=1)
    z = (xbars - mu) / (sigma / np.sqrt(n))
    return z

# ============================================================
# SECTION 2: Divergence Measures
# ============================================================

def compute_ks(stat_values, dist_name, *args):
    return stats.kstest(stat_values, dist_name, args=args)

def wilson_ci(t1e, N, z=1.959963984540054):
    """95% Wilson score interval for an empirical rejection rate t1e from N trials."""
    if N <= 0 or np.isnan(t1e):
        return (np.nan, np.nan)
    phat = t1e
    denom = 1 + z**2 / N
    center = (phat + z**2 / (2*N)) / denom
    half = (z * np.sqrt(phat*(1-phat)/N + z**2/(4*N**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def fit_convergence_rate(ks_arr, ns_arr):
    """Fit log(KS) = a + b * log(n); return the exponent b and its standard error.
    The Edgeworth-smooth prediction is b ~ -0.5; lattice tests deviate."""
    ks = np.array(ks_arr, dtype=float)
    ns = np.array(ns_arr, dtype=float)
    mask = ~np.isnan(ks) & (ks > 1e-14) & (ns > 0)
    if mask.sum() < 3:
        return np.nan, np.nan
    log_ks = np.log(ks[mask])
    log_ns = np.log(ns[mask])
    p, cov = np.polyfit(log_ns, log_ks, 1, cov=True)
    b = p[0]
    se_b = np.sqrt(cov[0, 0])
    return b, se_b


def compute_type1_error(stat_values, dist_name, df_args, alpha=0.05, two_sided=False):
    if dist_name == 'binom':
        n_binom, p_binom = df_args
        lower = stats.binom.ppf(alpha/2, n_binom, p_binom)
        upper = stats.binom.ppf(1 - alpha/2, n_binom, p_binom)
        return np.mean((stat_values < lower) | (stat_values > upper))
    if two_sided:
        if dist_name == 't':
            crit = stats.t.ppf(1 - alpha/2, *df_args)
            return np.mean(np.abs(stat_values) > crit)
        elif dist_name == 'norm':
            crit = stats.norm.ppf(1 - alpha/2)
            return np.mean(np.abs(stat_values) > crit)
    else:
        if dist_name == 'f':
            crit = stats.f.ppf(1 - alpha, *df_args)
        elif dist_name == 'chi2':
            crit = stats.chi2.ppf(1 - alpha, *df_args)
        elif dist_name == 'norm':
            crit = stats.norm.ppf(1 - alpha/2)
            return np.mean(np.abs(stat_values) > crit)
        else:
            return np.nan
        return np.mean(stat_values > crit)


def compute_type1_error_midp_sign(stat_values, n, alpha=0.05):
    """Two-sided mid-p Type I error for the sign test.
    stat_values = N_plus counts (integers 0..n).  Uses Lancaster's (1961) mid-p:
    p_mid = 2 * min( P(X < N+) + 0.5*P(X=N+),  P(X > N+) + 0.5*P(X=N+)) < alpha."""
    N = len(stat_values)
    reject = np.zeros(N, dtype=bool)
    for i, np_val in enumerate(stat_values):
        x = int(np_val)
        pl = stats.binom.cdf(x - 1, n, 0.5) + 0.5 * stats.binom.pmf(x, n, 0.5)
        pu = stats.binom.sf(x, n, 0.5) + 0.5 * stats.binom.pmf(x, n, 0.5)
        reject[i] = 2 * min(pl, pu) < alpha
    return reject.mean()


def compute_kl(stat_values, dist_name, df_args, n_bins=80):
    eps = 1e-12
    if dist_name == 'binom':
        n_binom, p_binom = df_args
        vals = np.arange(0, n_binom + 1)
        emp_pmf = np.bincount(stat_values.astype(int), minlength=n_binom+1) / len(stat_values)
        theo_pmf = stats.binom.pmf(vals, n_binom, p_binom)
        mask = (emp_pmf > eps) & (theo_pmf > eps)
        if mask.sum() < 2:
            return np.nan
        return float(np.sum(emp_pmf[mask] * np.log(emp_pmf[mask] / theo_pmf[mask])))
    sv = stat_values[np.isfinite(stat_values)]
    if len(sv) < 50:
        return np.nan
    try:
        kde = stats.gaussian_kde(sv, bw_method='silverman')
    except Exception:
        return np.nan
    lo = np.percentile(sv, 0.5)
    hi = np.percentile(sv, 99.5)
    x_grid = np.linspace(lo, hi, 500)
    p_emp = kde(x_grid)
    dist_obj = getattr(stats, dist_name) if dist_name != 'norm' else stats.norm
    if dist_name == 'norm':
        q_theo = dist_obj.pdf(x_grid)
    else:
        q_theo = dist_obj.pdf(x_grid, *df_args)
    mask = (p_emp > eps) & (q_theo > eps)
    if mask.sum() < 10:
        return np.nan
    dx = x_grid[1] - x_grid[0]
    kl = float(np.sum(p_emp[mask] * np.log(p_emp[mask] / q_theo[mask]) * dx))
    return max(kl, 0.0)

# ============================================================
# SECTION 3: Study 1 - Finite-sample convergence
# ============================================================

def run_study1(N_mc=10000):
    print("=" * 60)
    print("STUDY 1: Finite-sample convergence")
    print("=" * 60)
    sample_sizes = [5, 10, 20, 30, 50, 100, 200, 500]
    k_default = 3
    results = {}

    tests = {
        'T1_t_test': {'label': 'Student t'},
        'T2_anova': {'label': 'ANOVA F'},
        'T3_kruskal_wallis': {'label': 'Kruskal-Wallis'},
        'T4_chisq_gof': {'label': 'Chi-sq GOF'},
        'T5_bartlett': {'label': 'Bartlett'},
        'T6_fligner_killeen': {'label': 'Fligner-Killeen'},
        'T7_sign_test': {'label': 'Sign test'},
        'T8_cochran_q': {'label': 'Cochran Q'},
        'T9_median_test': {'label': 'Median test'},
        'T10_hotelling': {'label': 'Hotelling T2'},
    }

    for test_id, info in tests.items():
        results[test_id] = {'ks': [], 'kl': [], 'type1': [], 'ci_lo': [], 'ci_hi': [],
                            'q95_emp': [], 'q95_theory': [], 'frac_defined': [],
                            'ns': [], 'N': N_mc}
        print(f"\n  {info['label']}...", end=' ', flush=True)
        t0 = time.time()

        for n in sample_sizes:
            N_eff = N_mc
            try:
                if test_id == 'T1_t_test':
                    sv = mc_t_test(n, N_mc)
                    ks_s, ks_p = compute_ks(sv, 't', n-1)
                    t1e = compute_type1_error(sv, 't', (n-1,), two_sided=True)
                    kl_val = compute_kl(sv, 't', (n-1,))
                    q95_t = stats.t.ppf(0.975, n-1)
                    q95_e = np.percentile(np.abs(sv), 95)

                elif test_id == 'T2_anova':
                    sv, df1, df2 = mc_anova(k_default, n, N_mc)
                    ks_s, ks_p = compute_ks(sv, 'f', df1, df2)
                    t1e = compute_type1_error(sv, 'f', (df1, df2))
                    kl_val = compute_kl(sv, 'f', (df1, df2))
                    q95_t = stats.f.ppf(0.95, df1, df2)
                    q95_e = np.percentile(sv, 95)

                elif test_id == 'T3_kruskal_wallis':
                    sv = mc_kruskal_wallis(k_default, n, N_mc)
                    ks_s, ks_p = compute_ks(sv, 'chi2', k_default-1)
                    t1e = compute_type1_error(sv, 'chi2', (k_default-1,))
                    kl_val = compute_kl(sv, 'chi2', (k_default-1,))
                    q95_t = stats.chi2.ppf(0.95, k_default-1)
                    q95_e = np.percentile(sv, 95)

                elif test_id == 'T4_chisq_gof':
                    k_gof = 6
                    sample_n = max(n * 5, 30)
                    sv = mc_chisq_gof(k_gof, sample_n, N_mc)
                    ks_s, ks_p = compute_ks(sv, 'chi2', k_gof-1)
                    t1e = compute_type1_error(sv, 'chi2', (k_gof-1,))
                    kl_val = compute_kl(sv, 'chi2', (k_gof-1,))
                    q95_t = stats.chi2.ppf(0.95, k_gof-1)
                    q95_e = np.percentile(sv, 95)

                elif test_id == 'T5_bartlett':
                    sv = mc_bartlett(k_default, n, N_mc)
                    ks_s, ks_p = compute_ks(sv, 'chi2', k_default-1)
                    t1e = compute_type1_error(sv, 'chi2', (k_default-1,))
                    kl_val = compute_kl(sv, 'chi2', (k_default-1,))
                    q95_t = stats.chi2.ppf(0.95, k_default-1)
                    q95_e = np.percentile(sv, 95)

                elif test_id == 'T6_fligner_killeen':
                    sv = mc_fligner_killeen(k_default, n, N_mc)
                    ks_s, ks_p = compute_ks(sv, 'chi2', k_default-1)
                    t1e = compute_type1_error(sv, 'chi2', (k_default-1,))
                    kl_val = compute_kl(sv, 'chi2', (k_default-1,))
                    q95_t = stats.chi2.ppf(0.95, k_default-1)
                    q95_e = np.percentile(sv, 95)

                elif test_id == 'T7_sign_test':
                    sv = mc_sign_test(n, N_mc)
                    t1e = compute_type1_error(sv, 'binom', (n, 0.5))
                    d_max = 0
                    for x_val in range(n+1):
                        emp_cdf = np.mean(sv <= x_val)
                        theo_cdf = stats.binom.cdf(x_val, n, 0.5)
                        d_max = max(d_max, abs(emp_cdf - theo_cdf))
                    ks_s = d_max
                    ks_p = np.nan
                    kl_val = compute_kl(sv, 'binom', (n, 0.5))
                    q95_t = stats.binom.ppf(0.95, n, 0.5)
                    q95_e = np.percentile(sv, 95)

                elif test_id == 'T8_cochran_q':
                    sv_all = mc_cochran_q(k_default, n, N_mc)
                    frac_def = float(np.mean(np.isfinite(sv_all)))
                    sv = sv_all[np.isfinite(sv_all)]
                    if sv.size < 100:
                        # too few defined blocks to estimate anything
                        ks_s, t1e, kl_val = np.nan, np.nan, np.nan
                        q95_t = stats.chi2.ppf(0.95, k_default-1)
                        q95_e = np.nan
                    else:
                        ks_s, ks_p = compute_ks(sv, 'chi2', k_default-1)
                        t1e = compute_type1_error(sv, 'chi2', (k_default-1,))
                        kl_val = compute_kl(sv, 'chi2', (k_default-1,))
                        q95_t = stats.chi2.ppf(0.95, k_default-1)
                        q95_e = np.percentile(sv, 95)
                    results[test_id]['frac_defined'].append(frac_def)
                    N_eff = int(sv.size)

                elif test_id == 'T9_median_test':
                    sv = mc_median_test(k_default, n, N_mc)
                    ks_s, ks_p = compute_ks(sv, 'chi2', k_default-1)
                    t1e = compute_type1_error(sv, 'chi2', (k_default-1,))
                    kl_val = compute_kl(sv, 'chi2', (k_default-1,))
                    q95_t = stats.chi2.ppf(0.95, k_default-1)
                    q95_e = np.percentile(sv, 95)

                elif test_id == 'T10_hotelling':
                    p_dim = 2
                    if n <= p_dim + 1:
                        ks_s, t1e, kl_val = np.nan, np.nan, np.nan
                    else:
                        N_ht = N_mc
                        sv, dfn, dfd = mc_hotelling_t2(p_dim, n, N_ht)
                        ks_s, ks_p = compute_ks(sv, 'f', dfn, dfd)
                        t1e = compute_type1_error(sv, 'f', (dfn, dfd))
                        kl_val = compute_kl(sv, 'f', (dfn, dfd))
                        q95_t = stats.f.ppf(0.95, dfn, dfd)
                        q95_e = np.percentile(sv, 95)

                results[test_id]['ks'].append(float(ks_s))
                results[test_id]['kl'].append(float(kl_val) if not np.isnan(kl_val) else np.nan)
                results[test_id]['type1'].append(float(t1e))
                ci_lo, ci_hi = wilson_ci(t1e, N_eff)
                results[test_id]['ci_lo'].append(float(ci_lo))
                results[test_id]['ci_hi'].append(float(ci_hi))
                results[test_id]['q95_emp'].append(float(q95_e) if not np.isnan(q95_e) else np.nan)
                results[test_id]['q95_theory'].append(float(q95_t))
                if len(results[test_id]['frac_defined']) < len(results[test_id]['ns']) + 1:
                    results[test_id]['frac_defined'].append(1.0)
                results[test_id]['ns'].append(n)
            except Exception as e:
                print(f"[ERR n={n}: {e}]", end=' ')
                results[test_id]['ks'].append(np.nan)
                results[test_id]['kl'].append(np.nan)
                results[test_id]['type1'].append(np.nan)
                results[test_id]['ci_lo'].append(np.nan)
                results[test_id]['ci_hi'].append(np.nan)
                results[test_id]['q95_emp'].append(np.nan)
                results[test_id]['q95_theory'].append(np.nan)
                results[test_id]['frac_defined'].append(np.nan)
                results[test_id]['ns'].append(n)

        elapsed = time.time() - t0
        print(f"({elapsed:.1f}s)")

    print("\n--- Study 1 Results: Type I Error Rate ---")
    header = f"{'Test':<20}" + "".join(f"{'n='+str(n):>8}" for n in sample_sizes)
    print(header)
    print("-" * len(header))
    for test_id, info in tests.items():
        row = f"{info['label']:<20}"
        for v in results[test_id]['type1']:
            row += f"{v:>8.4f}" if not np.isnan(v) else f"{'N/A':>8}"
        print(row)

    print("\n--- Study 1 Results: KS Distance ---")
    print(header)
    print("-" * len(header))
    for test_id, info in tests.items():
        row = f"{info['label']:<20}"
        for v in results[test_id]['ks']:
            row += f"{v:>8.4f}" if not np.isnan(v) else f"{'N/A':>8}"
        print(row)

    ks_threshold = 1.358 / np.sqrt(N_mc)
    print(f"\n  KS adequacy threshold (95% KS critical value at N={N_mc}): {ks_threshold:.4f}")
    n_star = {}
    n_star_ks = {}
    for test_id, info in tests.items():
        t1es = results[test_id]['type1']
        kss = results[test_id]['ks']
        ns_list = results[test_id]['ns']
        found = None
        for nn, t1e in zip(ns_list, t1es):
            if not np.isnan(t1e) and abs(t1e - 0.05) < 0.01:
                found = nn
                break
        n_star[test_id] = found if found else '>500'
        found_ks = None
        for nn, ks in zip(ns_list, kss):
            if not np.isnan(ks) and ks < ks_threshold:
                found_ks = nn
                break
        n_star_ks[test_id] = found_ks if found_ks else '>500'
        print(f"  {info['label']}: n*(alpha) = {n_star[test_id]}, n*(KS) = {n_star_ks[test_id]}")
    results['_ks_threshold'] = ks_threshold
    results['_n_star_ks'] = n_star_ks

    fitted_b = {}
    fitted_b_se = {}
    print("\n--- Study 1: Fitted convergence exponents ---")
    for test_id, info in tests.items():
        b, se = fit_convergence_rate(results[test_id]['ks'], results[test_id]['ns'])
        fitted_b[test_id] = float(b) if not np.isnan(b) else b
        fitted_b_se[test_id] = float(se) if not np.isnan(se) else se
        results[test_id]['b'] = fitted_b[test_id]
        results[test_id]['b_se'] = fitted_b_se[test_id]
        print(f"  {info['label']}: fitted b = {fitted_b[test_id]:.3f} (SE = {fitted_b_se[test_id]:.4f})" if not np.isnan(b) else f"  {info['label']}: N/A")
    results['_fitted_b'] = fitted_b
    results['_fitted_b_se'] = fitted_b_se

    # Publication figures are generated after all studies are collected.

    return results, n_star

# ============================================================
# SECTION 4: Study 2 - Robustness
# ============================================================

def run_study2(N_mc=10000, n_fixed=50):
    print("\n" + "=" * 60)
    print("STUDY 2: Robustness under assumption violations")
    print("=" * 60)
    k = 3
    results = {}

    scenarios = [
        ('baseline', {}),
        ('skewed', {'dist': 'expon'}),
        ('heavy_tail', {'dist': 'tdist'}),
        ('hetero_var', {'hetero': True}),
        ('unbalanced', {'unbalanced': True}),
        ('small_freq', {'small_freq': True}),
    ]

    test_configs = {
        'T1_t_test': {
            'label': 'Student t',
            'applicable': ['baseline', 'skewed', 'heavy_tail'],
            'run': lambda sc, N: (
                mc_t_test(n_fixed, N, dist=sc.get('dist', 'normal')),
                't', (n_fixed-1,), True
            ),
        },
        'T2_anova': {
            'label': 'ANOVA F',
            'applicable': ['baseline', 'skewed', 'heavy_tail', 'hetero_var', 'unbalanced'],
            'run': lambda sc, N: (
                mc_anova(k, n_fixed, N, dist=sc.get('dist', 'normal'),
                         hetero=sc.get('hetero', False), unbalanced=sc.get('unbalanced', False))[0],
                'f',
                mc_anova(k, n_fixed, 10, dist='normal',
                         unbalanced=sc.get('unbalanced', False))[1:],
                False
            ),
        },
        'T3_kruskal_wallis': {
            'label': 'Kruskal-Wallis',
            'applicable': ['baseline', 'skewed', 'heavy_tail', 'hetero_var', 'unbalanced'],
            'run': lambda sc, N: (
                mc_kruskal_wallis(k, n_fixed, N, dist=sc.get('dist', 'uniform'),
                                  hetero=sc.get('hetero', False), unbalanced=sc.get('unbalanced', False)),
                'chi2', (k-1,), False
            ),
        },
        'T4_chisq_gof': {
            'label': 'Chi-sq GOF',
            'applicable': ['baseline', 'small_freq'],
            'run': lambda sc, N: (
                mc_chisq_gof(6, 30, N, small_expected=True) if sc.get('small_freq')
                else mc_chisq_gof(6, 250, N),
                'chi2', (5,), False
            ),
        },
        'T5_bartlett': {
            'label': 'Bartlett',
            'applicable': ['baseline', 'skewed', 'heavy_tail', 'unbalanced'],
            'run': lambda sc, N: (
                mc_bartlett(k, n_fixed, N, dist=sc.get('dist', 'normal'),
                            unbalanced=sc.get('unbalanced', False)),
                'chi2', (k-1,), False
            ),
        },
        'T6_fligner_killeen': {
            'label': 'Fligner-Killeen',
            'applicable': ['baseline', 'skewed', 'heavy_tail', 'unbalanced'],
            'run': lambda sc, N: (
                mc_fligner_killeen(k, n_fixed, N, dist=sc.get('dist', 'normal'),
                                    unbalanced=sc.get('unbalanced', False)),
                'chi2', (k-1,), False
            ),
        },
        'T7_sign_test': {
            'label': 'Sign test',
            'applicable': ['baseline', 'skewed', 'heavy_tail'],
            'run': lambda sc, N: (
                mc_sign_test(n_fixed, N,
                             dist={'expon': 'expon', 'tdist': 'tdist'}.get(sc.get('dist', ''), 'normal')),
                'binom', (n_fixed, 0.5), False
            ),
        },
        'T8_cochran_q': {
            'label': 'Cochran Q',
            # no 'unbalanced' scenario: Q is defined on a complete block design,
            # so an unequal-group-size manipulation has no meaning for this test
            'applicable': ['baseline', 'small_freq'],
            'run': lambda sc, N: (
                mc_cochran_q(k, n_fixed, N, p=(0.05 if sc.get('small_freq') else 0.5)),
                'chi2', (k-1,), False
            ),
        },
        'T9_median_test': {
            'label': 'Median test',
            'applicable': ['baseline', 'skewed', 'heavy_tail', 'hetero_var', 'unbalanced', 'small_freq'],
            'run': lambda sc, N: (
                mc_median_test(k, (8 if sc.get('small_freq') else n_fixed), N,
                               dist=sc.get('dist', 'uniform'),
                               hetero=sc.get('hetero', False), unbalanced=sc.get('unbalanced', False)),
                'chi2', (k-1,), False
            ),
        },
        'T10_hotelling': {
            'label': 'Hotelling T2',
            'applicable': ['baseline', 'skewed', 'heavy_tail'],
            'run': lambda sc, N: (
                mc_hotelling_t2(2, n_fixed, N, dist=sc.get('dist', 'normal'))[0],
                'f', (2, n_fixed-2), False
            ),
        },
    }

    ci_results = {}
    for test_id, cfg in test_configs.items():
        results[test_id] = {}
        ci_results[test_id] = {}
        print(f"\n  {cfg['label']}...", end=' ', flush=True)
        t0 = time.time()
        for sc_name, sc_params in scenarios:
            if sc_name not in cfg['applicable']:
                results[test_id][sc_name] = np.nan
                ci_results[test_id][sc_name] = [np.nan, np.nan]
                continue
            try:
                sv, dist_name, df_args, two_sided = cfg['run'](sc_params, N_mc)
                if dist_name == 'binom':
                    t1e = compute_type1_error(sv, 'binom', df_args)
                else:
                    t1e = compute_type1_error(sv, dist_name, df_args, two_sided=two_sided)
                results[test_id][sc_name] = float(t1e)
                lo, hi = wilson_ci(t1e, len(sv))
                ci_results[test_id][sc_name] = [float(lo), float(hi)]
            except Exception as e:
                results[test_id][sc_name] = np.nan
                ci_results[test_id][sc_name] = [np.nan, np.nan]
                print(f"[ERR {sc_name}: {e}]", end=' ')
        elapsed = time.time() - t0
        print(f"({elapsed:.1f}s)")

    corr_results = {}
    corr_ci = {}
    print("\n  --- Correction runs ---")
    for test_id, corr_label in [('T5_bartlett', 'Bartlett* (kurtosis-corr.)'), ('T7_sign_test', 'Sign (mid-p)')]:
        corr_results[test_id] = {}
        corr_ci[test_id] = {}
        print(f"  {corr_label}...", end=' ', flush=True)
        ct0 = time.time()
        for sc_name, sc_params in scenarios:
            if sc_name not in test_configs[test_id]['applicable']:
                corr_results[test_id][sc_name] = np.nan
                corr_ci[test_id][sc_name] = [np.nan, np.nan]
                continue
            try:
                if test_id == 'T5_bartlett':
                    sv = mc_bartlett_corrected(k, n_fixed, N_mc,
                        dist=sc_params.get('dist', 'normal'),
                        unbalanced=sc_params.get('unbalanced', False))
                    t1e = compute_type1_error(sv, 'chi2', (k-1,))
                elif test_id == 'T7_sign_test':
                    sv = mc_sign_test(n_fixed, N_mc,
                        dist={'expon':'expon','tdist':'tdist'}.get(sc_params.get('dist',''), 'normal'))
                    t1e = compute_type1_error_midp_sign(sv, n_fixed)
                corr_results[test_id][sc_name] = float(t1e)
                lo, hi = wilson_ci(t1e, N_mc)
                corr_ci[test_id][sc_name] = [float(lo), float(hi)]
            except Exception as e:
                corr_results[test_id][sc_name] = np.nan
                corr_ci[test_id][sc_name] = [np.nan, np.nan]
                print(f"[ERR {sc_name}: {e}]", end=' ')
        print(f"({time.time()-ct0:.1f}s)")

    print("\n--- Study 2 Results: Empirical Type I Error (α=0.05) ---")
    sc_names = [s[0] for s in scenarios]
    header = f"{'Test':<20}" + "".join(f"{s:>12}" for s in sc_names)
    print(header)
    print("-" * len(header))
    for test_id, cfg in test_configs.items():
        row = f"{cfg['label']:<20}"
        for sc_name in sc_names:
            v = results[test_id].get(sc_name, np.nan)
            if np.isnan(v):
                row += f"{'—':>12}"
            elif abs(v - 0.05) > 0.025:
                row += f"{v:>11.4f}*"
            else:
                row += f"{v:>12.4f}"
        print(row)

    # Compute bias and absolute error
    alpha = 0.05
    bias = {}; abs_err = {}
    corr_bias = {}; corr_abs = {}
    for test_id in results:
        bias[test_id] = {}
        abs_err[test_id] = {}
        for sc_name in results[test_id]:
            v = results[test_id][sc_name]
            if v is not None and not np.isnan(v):
                bias[test_id][sc_name] = float(v - alpha)
                abs_err[test_id][sc_name] = float(abs(v - alpha))
            else:
                bias[test_id][sc_name] = None
                abs_err[test_id][sc_name] = None
    for test_id in corr_results:
        corr_bias[test_id] = {}
        corr_abs[test_id] = {}
        for sc_name in corr_results[test_id]:
            v = corr_results[test_id][sc_name]
            if v is not None and not np.isnan(v):
                corr_bias[test_id][sc_name] = float(v - alpha)
                corr_abs[test_id][sc_name] = float(abs(v - alpha))
            else:
                corr_bias[test_id][sc_name] = None
                corr_abs[test_id][sc_name] = None

    # Compute calibration_improvement for Bartlett and Sign corrections
    alpha_target = 0.05
    calib_improve = {}
    for test_id in ['T5_bartlett', 'T7_sign_test']:
        calib_improve[test_id] = {}
        for sc_name in results[test_id]:
            if sc_name not in corr_results[test_id]:
                continue
            before = results[test_id].get(sc_name)
            after = corr_results[test_id].get(sc_name)
            if before is not None and after is not None and not np.isnan(before) and not np.isnan(after):
                err_before = abs(before - alpha_target)
                err_after = abs(after - alpha_target)
                if err_before > 1e-10:
                    calib_improve[test_id][sc_name] = float((err_before - err_after) / err_before)
                else:
                    calib_improve[test_id][sc_name] = 0.0
            else:
                calib_improve[test_id][sc_name] = None

    return results, ci_results, corr_results, corr_ci, bias, abs_err, corr_bias, corr_abs, calib_improve

# ============================================================
# SECTION 5: Study 3 - MC precision
# ============================================================

def _paired_difference(rej_b, rej_a, alphas):
    """Paired rejection indicators -> per-alpha sizes, difference and McNemar CI.

    The difference follows the paper's definition,
    Delta = corrected rejection rate - uncorrected rejection rate, i.e. it is
    reported as ``size_after - size_before``.  A negative Delta therefore means
    that the correction reduces the rejection rate (an improvement when the
    uncorrected test is liberal), and a positive Delta means that it increases
    the rejection rate.
    """
    out = {}
    N = int(len(rej_b(alphas[0])))
    for a in alphas:
        rb = rej_b(a)
        ra = rej_a(a)
        b = int(np.sum(rb & ~ra))     # rejects before, not after
        c = int(np.sum(~rb & ra))     # rejects after, not before
        diff = (c - b) / N            # corrected minus uncorrected
        disc = (b + c) / N
        se = float(np.sqrt(max(disc - diff ** 2, 0.0) / N))
        out[f"{a:g}"] = dict(size_before=float(rb.mean()), size_after=float(ra.mean()),
                             diff=float(diff), se=se,
                             ci_lo=float(diff - 1.96 * se), ci_hi=float(diff + 1.96 * se),
                             discordant_before_only=b, discordant_after_only=c, N=N)
    return out


def run_study2_pairing(N_mc=10000, n_fixed=50, alphas=(0.01, 0.025, 0.05, 0.1)):
    """Paired comparison of each correction with the statistic it replaces.

    The uncorrected and corrected statistics are computed from the *same*
    replications, so the comparison is a paired proportion problem rather than a
    difference of two independent Monte Carlo batches.  Reported per nominal level:
    the two sizes, their difference, and a McNemar (normal-approximation) 95%
    confidence interval for the difference.
    """
    k = 3
    q_bart = stats.chi2.ppf(1.0 - np.array(alphas), k - 1)
    scenarios = [
        ('baseline', {}),
        ('skewed', {'dist': 'expon'}),
        ('heavy_tail', {'dist': 'tdist'}),
        ('unbalanced', {'unbalanced': True}),
    ]
    out = {'alphas': list(alphas), 'N': N_mc, 'n': n_fixed, 'bartlett': {}, 'sign': {}}
    print("\n" + "=" * 60)
    print("STUDY 2b: paired correction comparison (Bartlett, sign test)")
    print("=" * 60)

    for sc_name, sc in scenarios:
        B, Bstar = mc_bartlett_corrected(k, n_fixed, N_mc,
                                         dist=sc.get('dist', 'normal'),
                                         unbalanced=sc.get('unbalanced', False),
                                         return_b=True)
        rej_b = lambda a, B=B: B > stats.chi2.ppf(1.0 - a, k - 1)
        rej_a = lambda a, Bs=Bstar: Bs > stats.chi2.ppf(1.0 - a, k - 1)
        out['bartlett'][sc_name] = _paired_difference(rej_b, rej_a, alphas)
        print(f"  Bartlett [{sc_name}]...", end=' ', flush=True)

    for sc_name, sc in scenarios:
        if sc_name == 'unbalanced':
            continue                      # sign test has no unbalanced design
        p_exact, p_midp = mc_sign_pvalues(n_fixed, N_mc,
                                          dist={'expon': 'expon', 'tdist': 'tdist'}.get(
                                              sc.get('dist', ''), 'normal'))
        out['sign'][sc_name] = _paired_difference(
            lambda a, p=p_exact: p < a, lambda a, p=p_midp: p < a, alphas)
    print("  done")
    return out


def run_power_study(N_mc=5000, n_fixed=50, alpha=0.05, k=3):
    """Power at alpha = 0.05, raw and size-adjusted.

    Deterministic alternatives: a mean shift for the t test, a three-level mean
    pattern for ANOVA, unequal variances for Bartlett, and a median shift for the
    sign test.

    For each statistic three independent samples are drawn: an alternative sample
    (for power), a *calibration* null sample (used to build the randomised test)
    and a second, independent *evaluation* null sample (used to measure the
    realised size of the calibrated rule).  Keeping calibration and evaluation
    separate matters for discrete statistics: if the same null draws are used for
    both, the calibrated rule has realised size 5% by construction and the
    reported size is not evidence of calibration.  The nominal size-adjusted level
    is exactly 5% by construction of the randomised test; the reported
    ``size_adjusted`` column is the separately measured realised level.
    """
    out = {'alpha': alpha, 'N': N_mc, 'N_calibration': N_mc, 'N_evaluation': N_mc,
           'n': n_fixed, 'tests': {}}
    print("\n" + "=" * 60)
    print("STUDY 5: power (raw and size-adjusted; independent calibration sample)")
    print("=" * 60)
    rng = np.random.default_rng(SEED + 101)

    def _pvalues(stat, reference, two_sided=False):
        if two_sided:
            return 2.0 * reference.sf(np.abs(stat))
        return reference.sf(stat)

    def _randomised(p_cal, p_obs, u):
        """Randomised p-value using the independent calibration sample.

        P_{cal}(p < p_obs) + U * P_{cal}(p = p_obs); under H0 this is
        Uniform(0,1), also when the p-value has atoms."""
        pc = np.sort(np.asarray(p_cal))
        n = pc.size
        less = np.searchsorted(pc, p_obs, side='left') / n
        leq = np.searchsorted(pc, p_obs, side='right') / n
        return less + u * (leq - less)

    def _adjusted(p_cal, p_alt, p_eval):
        u_alt = rng.random(np.size(p_alt))
        u_eval = rng.random(np.size(p_eval))
        return (dict(power_raw=float(np.mean(p_alt < alpha)),
                     size_raw=float(np.mean(p_eval < alpha)),
                     power_size_adjusted=float(np.mean(
                         _randomised(p_cal, p_alt, u_alt) < alpha)),
                     size_adjusted=float(np.mean(
                         _randomised(p_cal, p_eval, u_eval) < alpha))),
                u_alt, u_eval)

    # --- one-sample t: mean shift delta = 0.3 sd
    delta = 0.3
    X = rng.normal(delta, 1.0, (N_mc, n_fixed))
    tstat = X.mean(axis=1) / (X.std(axis=1, ddof=1) / np.sqrt(n_fixed))
    Xc = rng.normal(0.0, 1.0, (N_mc, n_fixed))
    Xe = rng.normal(0.0, 1.0, (N_mc, n_fixed))
    tc = Xc.mean(axis=1) / (Xc.std(axis=1, ddof=1) / np.sqrt(n_fixed))
    te = Xe.mean(axis=1) / (Xe.std(axis=1, ddof=1) / np.sqrt(n_fixed))
    ref = stats.t(n_fixed - 1)
    p_alt, p_cal, p_eval = _pvalues(tstat, ref, True), _pvalues(tc, ref, True), _pvalues(te, ref, True)
    adj, _, _ = _adjusted(p_cal, p_alt, p_eval)
    out['tests']['T1_t_test'] = dict(alternative=f'mean shift {delta}',
                                     power=adj['power_raw'], size=adj['size_raw'],
                                     power_size_adjusted=adj['power_size_adjusted'],
                                     size_adjusted=adj['size_adjusted'])

    # --- ANOVA: means 0, 0.3, 0.6 with unit sd
    means = np.linspace(0.0, 0.6, k)
    def _anova_F(Xa):
        gm = Xa.mean(axis=(1, 2), keepdims=True)
        ssb = n_fixed * ((Xa.mean(axis=2) - gm[:, :, 0]) ** 2).sum(axis=1)
        ssw = ((Xa - Xa.mean(axis=2, keepdims=True)) ** 2).sum(axis=(1, 2))
        return (ssb / (k - 1)) / (ssw / (k * (n_fixed - 1)))
    F = _anova_F(rng.normal(means[None, :, None], 1.0, (N_mc, k, n_fixed)))
    Fc = _anova_F(rng.normal(0.0, 1.0, (N_mc, k, n_fixed)))
    Fe = _anova_F(rng.normal(0.0, 1.0, (N_mc, k, n_fixed)))
    ref = stats.f(k - 1, k * (n_fixed - 1))
    p_alt, p_cal, p_eval = _pvalues(F, ref), _pvalues(Fc, ref), _pvalues(Fe, ref)
    adj, _, _ = _adjusted(p_cal, p_alt, p_eval)
    out['tests']['T2_anova'] = dict(alternative='group means 0, 0.3, 0.6',
                                    power=adj['power_raw'], size=adj['size_raw'],
                                    power_size_adjusted=adj['power_size_adjusted'],
                                    size_adjusted=adj['size_adjusted'])

    # --- Bartlett: variances 1, 1.5, 2; same parent (normal) for the null
    sds = np.sqrt(np.array([1.0, 1.5, 2.0])[:k])
    def _bartlett_pair(Xa):
        vars_i = Xa.var(axis=2, ddof=1)
        n_total = k * n_fixed
        sp2 = ((n_fixed - 1) * vars_i).sum(axis=1) / (n_total - k)
        num = (n_total - k) * np.log(sp2) - (n_fixed - 1) * np.log(vars_i).sum(axis=1)
        den = 1 + (1 / (3 * (k - 1))) * (k / (n_fixed - 1) - 1 / (n_total - k))
        Bv = num / den
        mu = Xa.mean(axis=2, keepdims=True)
        m2 = ((Xa - mu) ** 2).mean(axis=2)
        m4 = ((Xa - mu) ** 4).mean(axis=2)
        g2 = (m4 / np.maximum(m2 ** 2, 1e-15) - 3).mean(axis=1)
        return Bv, Bv / np.maximum(1 + g2 / 2, 1e-6)
    B, Bstar = _bartlett_pair(rng.normal(size=(N_mc, k, n_fixed)) * sds[None, :, None])
    B0, Bstar0 = _bartlett_pair(rng.normal(size=(N_mc, k, n_fixed)))
    B0e, Bstar0e = _bartlett_pair(rng.normal(size=(N_mc, k, n_fixed)))
    ref = stats.chi2(k - 1)
    adj, u_alt, u_eval = _adjusted(_pvalues(B0, ref), _pvalues(B, ref), _pvalues(B0e, ref))
    adj_c = dict(power_raw=float(np.mean(_pvalues(Bstar, ref) < alpha)),
                 size_raw=float(np.mean(_pvalues(Bstar0e, ref) < alpha)),
                 power_size_adjusted=float(np.mean(
                     _randomised(_pvalues(Bstar0, ref), _pvalues(Bstar, ref), u_alt) < alpha)),
                 size_adjusted=float(np.mean(
                     _randomised(_pvalues(Bstar0, ref), _pvalues(Bstar0e, ref), u_eval) < alpha)))
    out['tests']['T5_bartlett'] = dict(
        alternative='variances 1, 1.5, 2',
        power=adj['power_raw'], size=adj['size_raw'],
        power_corrected=adj_c['power_raw'], size_corrected=adj_c['size_raw'],
        power_size_adjusted=adj['power_size_adjusted'],
        power_size_adjusted_corrected=adj_c['power_size_adjusted'],
        size_adjusted=adj['size_adjusted'],
        size_adjusted_corrected=adj_c['size_adjusted'])

    # --- sign test: median shift, normal parent
    shift = 0.3
    def _sign_p(Np):
        pe = 2.0 * np.minimum(stats.binom.cdf(Np, n_fixed, 0.5),
                              stats.binom.sf(Np - 1, n_fixed, 0.5))
        pm = 2.0 * np.minimum(stats.binom.cdf(Np - 1, n_fixed, 0.5)
                              + 0.5 * stats.binom.pmf(Np, n_fixed, 0.5),
                              stats.binom.sf(Np, n_fixed, 0.5)
                              + 0.5 * stats.binom.pmf(Np, n_fixed, 0.5))
        return np.clip(pe, 0, 1), np.clip(pm, 0, 1)
    Np = (rng.normal(shift, 1.0, (N_mc, n_fixed)) > 0).sum(axis=1)
    Np0 = (rng.normal(0.0, 1.0, (N_mc, n_fixed)) > 0).sum(axis=1)
    Np0e = (rng.normal(0.0, 1.0, (N_mc, n_fixed)) > 0).sum(axis=1)
    p_exact, p_mid = _sign_p(Np)
    p0_exact, p0_mid = _sign_p(Np0)
    p0e_exact, p0e_mid = _sign_p(Np0e)
    adj, u_alt, u_eval = _adjusted(p0_exact, p_exact, p0e_exact)
    adj_m = dict(power_raw=float(np.mean(p_mid < alpha)),
                 size_raw=float(np.mean(p0e_mid < alpha)),
                 power_size_adjusted=float(np.mean(
                     _randomised(p0_mid, p_mid, u_alt) < alpha)),
                 size_adjusted=float(np.mean(
                     _randomised(p0_mid, p0e_mid, u_eval) < alpha)))
    out['tests']['T7_sign_test'] = dict(
        alternative=f'median shift {shift}',
        power=adj['power_raw'], size=adj['size_raw'],
        power_corrected=adj_m['power_raw'], size_corrected=adj_m['size_raw'],
        power_size_adjusted=adj['power_size_adjusted'],
        power_size_adjusted_corrected=adj_m['power_size_adjusted'],
        size_adjusted=adj['size_adjusted'],
        size_adjusted_corrected=adj_m['size_adjusted'])

    for name, rec in out['tests'].items():
        line = f"  {name:18s} raw={rec['power']:.3f}"
        if 'power_corrected' in rec:
            line += f"/{rec['power_corrected']:.3f}"
        line += f"  adjusted={rec['power_size_adjusted']:.3f}"
        if 'power_size_adjusted_corrected' in rec:
            line += f"/{rec['power_size_adjusted_corrected']:.3f}"
        line += f"  realised size={rec['size_adjusted']:.3f}"
        if 'size_adjusted_corrected' in rec:
            line += f"/{rec['size_adjusted_corrected']:.3f}"
        print(line)
    return out


def _draw_mv_shape(spec, n, p, rng):
    """One (n, p) group with the requested parent and a specified covariance."""
    if spec == 'normal':
        return rng.normal(size=(n, p))
    if spec == 'lognormal':
        return np.exp(rng.normal(size=(n, p)))
    if spec == 'lognormal_scaled':
        # covariance = I
        return np.exp(rng.normal(size=(n, p))) / np.sqrt(np.e ** 2 - np.e)
    if spec == 't5_scaled':
        # multivariate t_5 scaled to covariance = I
        Z = rng.normal(size=(n, p))
        chi2 = rng.chisquare(5, size=n)
        return (Z / np.sqrt(chi2 / 5.0)[:, None]) / np.sqrt(5.0 / 3.0)
    raise ValueError(f'unknown multivariate parent: {spec}')


def run_bootstrap_comparison(N_outer=1000, B=200):
    """Compare four distinct resampling schemes for Box's M under the null.

    The schemes are (i) a parametric bootstrap from the pooled normal, (ii) a
    within-group bootstrap of centred observations -- resampling each group's own
    residuals, which does not impose covariance homogeneity, (iii) a
    **null-imposed** bootstrap that standardises each group's residuals by its own
    covariance, resamples them, and rescales them to the pooled covariance, so the
    bootstrap null imposes equal covariances while retaining each group's
    higher-order shape, and (iv) a permutation of the pooled observations.

    Five null cells are used: three in which both groups share a parent, and two
    in which the groups have *equal covariance but different distributions*
    (normal vs scaled lognormal, normal vs scaled t5).  The latter are the cells
    in which permutation is expected to break down (exchangeability fails) while
    the null-imposed scheme remains the appropriate object.
    """
    cells = [
        dict(label='MVN p=2,k=2,n=10', parents=['normal', 'normal'], p=2, k=2, n=10,
             family='same parent'),
        dict(label='MVN p=2,k=2,n=30', parents=['normal', 'normal'], p=2, k=2, n=30,
             family='same parent'),
        dict(label='lognormal p=2,k=2,n=30', parents=['lognormal', 'lognormal'], p=2, k=2, n=30,
             family='same parent'),
        dict(label='normal vs lognormal, equal cov, n=30',
             parents=['normal', 'lognormal_scaled'], p=2, k=2, n=30,
             family='equal covariance, different shape'),
        dict(label='normal vs t5, equal cov, n=30',
             parents=['normal', 't5_scaled'], p=2, k=2, n=30,
             family='equal covariance, different shape'),
    ]
    out = {'N_outer': N_outer, 'B': B, 'alpha': 0.05,
           'note': ('equal-covariance cells: both groups have covariance I but '
                    'different distributions'), 'cells': {}}
    print("\n" + "=" * 60)
    print("STUDY 6: resampling comparison for Box's M")
    print("=" * 60)
    rng = np.random.default_rng(20241022)
    for cell in cells:
        label, p_dim, k, n = cell['label'], cell['p'], cell['k'], cell['n']

        def draw():
            return [_draw_mv_shape(spec, n, p_dim, rng) for spec in cell['parents']]

        pvals = {m: np.empty(N_outer) for m in
                 ('parametric', 'within_centred', 'null_imposed', 'permutation')}
        for r in range(N_outer):
            groups = draw()
            _, M_obs, _, _ = box_m_statistic(groups)
            sizes = [g.shape[0] for g in groups]
            means = [g.mean(axis=0) for g in groups]

            # pooled covariance used by the parametric and null-imposed schemes
            X = np.vstack(groups)
            Sp = (X - X.mean(axis=0)).T @ (X - X.mean(axis=0)) / (X.shape[0] - k)
            Lp = np.linalg.cholesky(Sp + 1e-12 * np.eye(p_dim))

            # (i) parametric bootstrap from the pooled normal
            pvals['parametric'][r] = (1.0 + np.sum(
                [box_m_statistic([rng.normal(size=(s, p_dim)) @ Lp.T for s in sizes])[1] >= M_obs
                 for _ in range(B)])) / (B + 1.0)

            # (ii) within-group bootstrap of centred observations
            def centred_rep():
                parts = []
                for g, m in zip(groups, means):
                    idx = rng.integers(0, g.shape[0], g.shape[0])
                    parts.append(m + (g - m)[idx])
                return parts
            pvals['within_centred'][r] = (1.0 + np.sum(
                [box_m_statistic(centred_rep())[1] >= M_obs for _ in range(B)])) / (B + 1.0)

            # (iii) null-imposed bootstrap: within-group standardised residuals
            # rescaled to the common pooled covariance
            std_resid = []
            for g, m in zip(groups, means):
                Sg = (g - m).T @ (g - m) / (g.shape[0] - 1)
                Lg = np.linalg.cholesky(Sg + 1e-12 * np.eye(p_dim))
                std_resid.append((g - m) @ np.linalg.inv(Lg).T)

            def null_imposed_rep():
                parts = []
                for R in std_resid:
                    idx = rng.integers(0, R.shape[0], R.shape[0])
                    parts.append(R[idx] @ Lp.T)
                return parts
            pvals['null_imposed'][r] = (1.0 + np.sum(
                [box_m_statistic(null_imposed_rep())[1] >= M_obs
                 for _ in range(B)])) / (B + 1.0)

            # (iv) permutation of the pooled observations
            pooled = np.vstack(groups)

            def perm_rep():
                idx = rng.permutation(pooled.shape[0])
                parts, i0 = [], 0
                for s in sizes:
                    parts.append(pooled[idx[i0:i0 + s]]); i0 += s
                return parts
            pvals['permutation'][r] = (1.0 + np.sum(
                [box_m_statistic(perm_rep())[1] >= M_obs for _ in range(B)])) / (B + 1.0)

        cell_out = {'parents': cell['parents'], 'family': cell['family'],
                    'p': p_dim, 'k': k, 'n': n, 'type1': {}}
        for m, pv in pvals.items():
            cell_out['type1'][m] = float(np.mean(pv < 0.05))
        out['cells'][label] = cell_out
        print(f"  {label:38s}: " + ", ".join(f"{m}={cell_out['type1'][m]:.3f}"
                                             for m in cell_out['type1']))
    return out


def run_ks_resolution(study1, N_big=50000, n_list=(10, 50, 200), N_huge=200000, n_huge=(10, 50)):
    """Recompute the KS distance for the discrete tests with a larger Monte Carlo N.

    The generator is seeded explicitly so that these numbers do not depend on how
    many random draws earlier studies consumed.

    The point is to separate the population CDF distance from the Monte Carlo
    resolution: if D_KS at N = 50,000 is much smaller than at N = 10,000 for a given
    n, then the smaller-N value was dominated by simulation noise, and the n*_KS
    threshold of Study 1 is a resolution scale rather than a property of the test.
    """
    np.random.seed(SEED + 202)
    specs = {
        'T3_kruskal_wallis': lambda n, N: (mc_kruskal_wallis(3, n, N), 'chi2', (2,)),
        'T4_chisq_gof': lambda n, N: (mc_chisq_gof(6, max(5 * n, 30), N), 'chi2', (5,)),
        'T6_fligner_killeen': lambda n, N: (mc_fligner_killeen(3, n, N), 'chi2', (2,)),
        'T8_cochran_q': lambda n, N: (mc_cochran_q(3, n, N), 'chi2', (2,)),
        'T9_median_test': lambda n, N: (mc_median_test(3, n, N), 'chi2', (2,)),
    }
    out = {'N_big': N_big, 'threshold_big': 1.358 / np.sqrt(N_big),
           'N_huge': N_huge, 'threshold_huge': 1.358 / np.sqrt(N_huge),
           'tests': {t: {'ns': [], 'ks_big': [], 'ks_10k': [],
                         'ns_huge': [], 'ks_huge': []} for t in specs}}
    print("\n" + "=" * 60)
    print(f"STUDY 1b: KS resolution at N={N_big}")
    print("=" * 60)
    for test_id, fn in specs.items():
        ns = study1[test_id]['ns']
        ones = study1[test_id]['ks']
        for n in n_list:
            if n not in ns:
                continue
            sv, dist_name, df_args = fn(n, N_big)
            sv = np.asarray(sv)[~np.isnan(sv)]   # e.g. Cochran's Q undefined reps
            ks, _ = compute_ks(sv, dist_name, *df_args)
            out['tests'][test_id]['ns'].append(int(n))
            out['tests'][test_id]['ks_big'].append(float(ks))
            out['tests'][test_id]['ks_10k'].append(float(ones[ns.index(n)]))
        print(f"  {test_id}: {[round(x,4) for x in out['tests'][test_id]['ks_big']]}")
        for n in n_huge:
            if n not in ns:
                continue
            sv, dist_name, df_args = fn(n, N_huge)
            sv = np.asarray(sv)[~np.isnan(sv)]
            ks, _ = compute_ks(sv, dist_name, *df_args)
            out['tests'][test_id]['ns_huge'].append(int(n))
            out['tests'][test_id]['ks_huge'].append(float(ks))
        print(f"  {test_id} @N={N_huge}: "
              f"{[round(x,4) for x in out['tests'][test_id]['ks_huge']]}")
    return out


def run_study3(n_fixed=50, n_repeats=30):
    print("\n" + "=" * 60)
    print("STUDY 3: MC precision requirements")
    print("=" * 60)
    N_values = [100, 500, 1000, 1500, 2000, 3000, 5000, 10000, 50000]
    k = 3
    results = {}

    test_funcs = {
        'T1_t_test': ('Student t', lambda N: mc_t_test(n_fixed, N), 't', (n_fixed-1,), True, None),
        'T2_anova': ('ANOVA F', lambda N: mc_anova(k, n_fixed, N)[0], 'f', (k-1, k*(n_fixed-1)), False, None),
        'T3_kruskal_wallis': ('Kruskal-Wallis', lambda N: mc_kruskal_wallis(k, n_fixed, N), 'chi2', (k-1,), False, 10000),
        'T4_chisq_gof': ('Chi-sq GOF', lambda N: mc_chisq_gof(6, 250, N), 'chi2', (5,), False, None),
        'T5_bartlett': ('Bartlett', lambda N: mc_bartlett(k, n_fixed, N), 'chi2', (k-1,), False, None),
        'T6_fligner_killeen': ('Fligner-Killeen', lambda N: mc_fligner_killeen(k, n_fixed, N), 'chi2', (k-1,), False, None),
        'T7_sign_test': ('Sign test', lambda N: mc_sign_test(n_fixed, N), 'binom', (n_fixed, 0.5), False, None),
        'T8_cochran_q': ('Cochran Q', lambda N: mc_cochran_q(k, n_fixed, N), 'chi2', (k-1,), False, None),
        'T9_median_test': ('Median test', lambda N: mc_median_test(k, n_fixed, N), 'chi2', (k-1,), False, 10000),
        'T10_hotelling': ('Hotelling T2', lambda N: mc_hotelling_t2(2, n_fixed, N)[0], 'f', (2, n_fixed-2), False, 10000),
    }

    for test_id, (label, gen_fn, dist_name, df_args, two_sided, max_N) in test_funcs.items():
        results[test_id] = {'N_values': [], 'mean_t1e': [], 'std_t1e': [], 'cv_t1e': []}
        print(f"\n  {label}...", end=' ', flush=True)
        t0 = time.time()
        for Nmc in N_values:
            if max_N is not None and Nmc > max_N:
                continue
            t1e_list = []
            for rep in range(n_repeats):
                sv = gen_fn(Nmc)
                if dist_name == 'binom':
                    t1e = compute_type1_error(sv, 'binom', df_args)
                else:
                    t1e = compute_type1_error(sv, dist_name, df_args, two_sided=two_sided)
                t1e_list.append(t1e)
            t1e_arr = np.array(t1e_list)
            m = t1e_arr.mean()
            s = t1e_arr.std()
            cv = s / m if m > 0 else np.nan
            results[test_id]['N_values'].append(Nmc)
            results[test_id]['mean_t1e'].append(float(m))
            results[test_id]['std_t1e'].append(float(s))
            results[test_id]['cv_t1e'].append(float(cv))
        elapsed = time.time() - t0
        print(f"({elapsed:.1f}s)")

    print("\n--- Study 3 Results: CV of Type I Error ---")
    header = f"{'Test':<15}" + "".join(f"{'N='+str(n):>10}" for n in N_values)
    print(header)
    print("-" * len(header))
    for test_id, (label, _, _, _, _, _) in test_funcs.items():
        row = f"{label:<15}"
        for cv in results[test_id]['cv_t1e']:
            row += f"{cv:>10.4f}"
        print(row)

    n_star_mc = {}
    for test_id, (label, _, _, _, _, _) in test_funcs.items():
        found = None
        for j, (Nmc, cv) in enumerate(zip(results[test_id]['N_values'], results[test_id]['cv_t1e'])):
            if cv < 0.10:
                found = Nmc
                break
        n_star_mc[test_id] = found if found else '>50000'
        print(f"  {label}: N* = {n_star_mc[test_id]}")

    return results, n_star_mc

# ============================================================
# SECTION 6: Auto-generate markdown tables from results
# ============================================================

TEST_ORDER = ['T1_t_test', 'T2_anova', 'T3_kruskal_wallis', 'T4_chisq_gof', 'T5_bartlett',
              'T6_fligner_killeen', 'T7_sign_test', 'T8_cochran_q', 'T9_median_test', 'T10_hotelling']
TEST_LABELS = {
    'T1_t_test': 'T1: Student\'s t', 'T2_anova': 'T2: ANOVA F', 'T3_kruskal_wallis': 'T3: Kruskal-Wallis',
    'T4_chisq_gof': 'T4: Pearson\'s $\\chi^2$ GOF', 'T5_bartlett': 'T5: Bartlett\'s',
    'T6_fligner_killeen': 'T6: Fligner-Killeen', 'T7_sign_test': 'T7: Sign test',
    'T8_cochran_q': 'T8: Cochran\'s Q', 'T9_median_test': 'T9: Median test', 'T10_hotelling': 'T10: Hotelling\'s $T^2$',
}

def _fmt(v, nd=3):
    return '—' if (v is None or (isinstance(v, float) and np.isnan(v))) else f'{v:.{nd}f}'

def generate_tables(results, out_path):
    """Write results/generated_tables.md, numbered as in paper.md.

    Every numeric table of the paper is generated here from the committed results
    file.  Tables 1, 2, 7 and 18 in paper.md are editorial (list of tests, summary
    of corrections, scenario definitions, practitioner guidelines) and have no
    machine-generated counterpart; all other tables are reproduced below under
    their paper numbers.
    """
    s1 = results['study1']
    s2 = results['study2']
    s2_ci = results['study2_ci']
    corr_results = results['study2_corrections']
    corr_ci = results['study2_corrections_ci']
    s3 = results['study3']
    n_star_mc = results['study3_nstar_mc']
    n_star_alpha = results['study1_nstar_alpha']
    n_star_ks = results['study1_nstar_ks']
    fitted_b = results['study1_fitted_b']
    fitted_b_se = results['study1_fitted_b_se']
    ks_thr = results['study1_ks_threshold']
    paired = results['study2_paired']
    power = results['power']
    ks_res = results['ks_resolution']
    boot = results['bootstrap_comparison']
    mv = results['study_mv']
    real = results['real_data_case_studies']
    exn = results['exact_null']

    L = []
    L.append('# Auto-generated result tables\n')
    L.append('Generated by `experiment.py`; table numbers match `paper.md`. '
             'Monte Carlo studies use N=10,000 replications unless a header says otherwise; '
             f'Study 1 KS adequacy threshold = {ks_thr:.4f}; intervals are 95% Wilson score intervals '
             'unless a header says McNemar or bootstrap. '
             'Tables 1, 2, 7 and 18 of the paper are editorial and are not reproduced here; '
             'Table A1 (Appendix A) is generated from the exact fourth-cumulant weights.\n')

    # ---------------- Table 3 (paper): lattice critical values ----------------
    L.append('\n## Table 3: Empirical 95% quantiles of the lattice-test statistics vs. theoretical $\\chi^2(2)$\n')
    L.append('| Test | n=5 | n=10 | n=20 | n=30 | n=50 |')
    L.append('| --- | --- | --- | --- | --- | --- |')
    for tid, tl in [('T8_cochran_q', "Cochran's $Q$"), ('T9_median_test', 'Median test')]:
        qe, qt, nv = s1[tid]['q95_emp'], s1[tid]['q95_theory'], s1[tid]['ns']
        row = [tl]
        for target_n in (5, 10, 20, 30, 50):
            if target_n in nv:
                i = nv.index(target_n)
                row.append('N/A' if (qe[i] is None or (isinstance(qe[i], float) and np.isnan(qe[i])))
                           else f'{qe[i]:.2f} ({qt[i]:.2f})')
            else:
                row.append('—')
        L.append('| ' + ' | '.join(row) + ' |')

    # ---------------- Table 4 (paper): exact cumulants ----------------
    L.append('\n## Table 4: Exact cumulants of $M_{\\text{raw}}$ under multivariate normality, '
             'against their classical limits\n')
    L.append('| $n$ | $k$ | $p$ | exact $\\mathbb{E}[M_{\\text{raw}}]$ | $\\nu$ | exact Var | $2\\nu$ | ratio |')
    L.append('| --- | --- | --- | --- | --- | --- | --- | --- |')
    for c in exn['cumulants']:
        L.append(f"| {c['n']} | {c['k']} | {c['p']} | {c['mean']:.4f} | {c['nu']:g} | "
                 f"{c['var']:.4f} | {2*c['nu']:g} | {c['var_over_2nu']:.3f} |")

    # ---------------- Tables 5 and 6 (paper): exact critical values / size ----------------
    cells_by = {(c['p'], c['n'], c['k']): c for c in exn['cells']}
    L.append('\n## Table 5: 95% critical values of $M_{\\text{raw}}$ under multivariate normality\n')
    L.append('| $p$ | $n$ | $k$ | $\\nu$ | saddlepoint | exact inversion | Monte Carlo | $\\chi^2_\\nu$ | '
             '$\\chi^2$ error |')
    L.append('| --- | --- | --- | --- | --- | --- | --- | --- | --- |')
    for p, n, k in exn['paper_crit']:
        c = cells_by[(p, n, k)]
        crit = c['crit']['0.05']
        inv = crit['inversion']
        mc = crit.get('mc')
        # the paper reports the chi-square error against the Monte Carlo reference
        ref = mc if mc is not None else inv
        err = 100.0 * (crit['chi2'] / ref - 1.0)
        mc_s = '—' if mc is None else f'{mc:.4f}'
        L.append(f"| {p} | {n} | {k} | {c['nu']:g} | {crit['lr']:.4f} | {inv:.4f} | {mc_s} | "
                 f"{crit['chi2']:.4f} | {err:.2f}% |")
    L.append('\n## Table 6: Exact size at $\\alpha=0.05$, computed from (5.4) with no simulation\n')
    L.append('| $p$ | $n$ | $k$ | $\\nu$ | $\\chi^2_\\nu$ on $M_{\\text{raw}}$ | '
             '$\\chi^2$ on $M=(1-c)M_{\\text{raw}}$ | saddlepoint rule |')
    L.append('| --- | --- | --- | --- | --- | --- | --- |')
    for p, n, k in exn['paper_size']:
        c = cells_by[(p, n, k)]
        sz = c['size']['0.05']
        L.append(f"| {p} | {n} | {k} | {c['nu']:g} | {sz['chi2_raw']:.4f} | "
                 f"{sz['chi2_bartlett']:.4f} | {sz['saddlepoint']:.4f} |")

    # ---------------- Table 8 (paper): n* and exponent ----------------
    L.append('\n## Table 8: Minimum adequate sample size and convergence exponent\n')
    L.append('| Test | $n^*_\\alpha$ | $n^*_{KS}$ | Fitted $\\hat b$ (pred. $\\approx-0.5$) | 95% CI for $\\hat b$ |')
    L.append('| --- | --- | --- | --- | --- |')
    t_val = stats.t.ppf(0.975, 6)  # 8 sample sizes -> 6 df
    for tid in TEST_ORDER:
        b = fitted_b.get(tid, np.nan)
        se = fitted_b_se.get(tid, np.nan)
        b_str = _fmt(b, 3) if not (isinstance(b, float) and np.isnan(b)) else 'N/A'
        if not (isinstance(b, float) and np.isnan(b)) and not (isinstance(se, float) and np.isnan(se)):
            lo, hi = b - t_val * se, b + t_val * se
            ci_str = f'[{lo:.3f}, {hi:.3f}]'
        else:
            ci_str = '—'
        L.append(f'| {TEST_LABELS[tid]} | {n_star_alpha.get(tid, "—")} | '
                 f'{n_star_ks.get(tid, "—")} | {b_str} | {ci_str} |')
    # T11 (Box's M, MVN p=2,k=2 as the representative configuration): recompute the
    # OLS slope and its t(4) interval from the stored KS distances
    t11 = results['study_mv'].get('T11_box_m_normal_p2_k2')
    if t11:
        x = np.log(np.asarray(t11['ns'], float))
        y = np.log(np.asarray(t11['ks_M'], float))
        A = np.vstack([np.ones_like(x), x]).T
        coef, res, _, _ = np.linalg.lstsq(A, y, rcond=None)
        dof = len(x) - 2
        sig2 = res[0] / dof if len(res) else np.nan
        se = float(np.sqrt(np.diag(np.linalg.inv(A.T @ A))[1] * sig2))
        tt = stats.t.ppf(0.975, dof)
        L.append(f"| T11  Box's $M$ | 10 | 10 | {coef[1]:.3f} | "
                 f"[{coef[1]-tt*se:.3f}, {coef[1]+tt*se:.3f}] |")

    # ---------------- Table 9 (paper): Type I error under violations ----------------
    sc_cols = ['baseline', 'skewed', 'heavy_tail', 'hetero_var', 'unbalanced', 'small_freq']
    sc_hdr = ['Baseline', 'Skewed', 'Heavy-tailed', 'Var. het.', 'Unbalanced', 'Small exp. freq.']
    L.append('\n## Table 9: Empirical Type I error (bias $\\hat\\alpha-0.05$ in parentheses) under '
             'assumption violations ($\\alpha=0.05$, $n=50$, $N=10{,}000$)\n')
    L.append('| Test | ' + ' | '.join(sc_hdr) + ' |')
    L.append('| --- |' + ' --- |' * len(sc_cols))
    for tid in TEST_ORDER:
        row = [TEST_LABELS[tid]]
        for sc in sc_cols:
            v = s2.get(tid, {}).get(sc)
            ci = s2_ci.get(tid, {}).get(sc, [np.nan, np.nan])
            if v is None or (isinstance(v, float) and np.isnan(v)):
                row.append('—')
            else:
                row.append(f'{v:.3f} ({v-0.05:+.3f}) [{ci[0]:.3f}, {ci[1]:.3f}]')
        L.append('| ' + ' | '.join(row) + ' |')
    L.append('\n### Table 9 — correction columns (Bartlett $B^*$, sign mid-$p$)\n')
    L.append('| Test | Baseline (corr.) | Skewed (corr.) | Heavy-tailed (corr.) |')
    L.append('| --- | --- | --- | --- |')
    for tid in ['T5_bartlett', 'T7_sign_test']:
        row = [TEST_LABELS[tid] + ' (corrected)']
        for sc in ['baseline', 'skewed', 'heavy_tail']:
            v = corr_results.get(tid, {}).get(sc)
            ci = corr_ci.get(tid, {}).get(sc, [np.nan, np.nan])
            if v is None or (isinstance(v, float) and np.isnan(v)):
                row.append('—')
            else:
                row.append(f'{v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]')
        L.append('| ' + ' | '.join(row) + ' |')

    # ---------------- Table 10 (paper): MC precision ----------------
    L.append('\n## Table 10: Minimum MC replications $N^*$ for CV < 0.10\n')
    L.append('| Test | $N^*$ | CV at N=1,000 | CV at N=10,000 |')
    L.append('| --- | --- | --- | --- |')
    for tid in TEST_ORDER:
        Ns = s3[tid]['N_values']
        cvs = s3[tid]['cv_t1e']
        cv_at = dict(zip(Ns, cvs))
        L.append(f'| {TEST_LABELS[tid]} | {n_star_mc.get(tid, "—")} | '
                 f'{_fmt(cv_at.get(1000), 3)} | {_fmt(cv_at.get(10000), 3)} |')

    # ---------------- Table 11 (paper): real-data summary ----------------
    L.append('\n## Table 11: Summary of real-data case studies\n')
    L.append('| Case | Dataset | Test | Key parameter | p-value | Conclusion at $\\alpha=0.05$ |')
    L.append('| --- | --- | --- | --- | ---: | --- |')
    c1, c2, c3 = real['case1_malic'], real['case2_alcohol'], real['case3_mtcars']
    L.append(f"| 1 | Wine `malic_acid` | Bartlett | $\\hat\\gamma_2={c1['gamma2_pool']:.2f}$ | "
             f"{c1['p_B']:.4f} | Reject (strong) |")
    L.append(f"| 1 | Wine `malic_acid` | $B^*$ | $1+\\hat\\gamma_2/2={c1['correction_factor']:.2f}$ | "
             f"{c1['p_B_star']:.4f} | Reject (marginal) |")
    L.append(f"| 2 | Wine `alcohol` | Bartlett | $\\hat\\gamma_2={c2['gamma2_pool']:.2f}$ | "
             f"{c2['p_B']:.4f} | Do not reject |")
    L.append(f"| 2 | Wine `alcohol` | $B^*$ | $1+\\hat\\gamma_2/2={c2['correction_factor']:.2f}$ | "
             f"{c2['p_B_star']:.4f} | Do not reject |")
    L.append(f"| 3 | mtcars ($n={c3['n']}$), $H_0$: median={c3['H0_median']:g} | sign (exact) | "
             f"$N^+={c3['N_plus']}/{c3['n_eff']}$ | {c3['p_exact']:.4f} | {'Do not reject' if c3['conclusion_exact'] == 'not reject' else 'Reject'} |")
    L.append(f"| 3 | mtcars ($n={c3['n']}$), $H_0$: median={c3['H0_median']:g} | sign (mid-$p$) | "
             f"$N^+={c3['N_plus']}/{c3['n_eff']}$ | {c3['p_midp']:.4f} | {'Do not reject' if c3['conclusion_midp'] == 'not reject' else 'Reject'} |")

    # ---------------- Table 12 (paper): where M* fails ----------------
    L.append('\n## Table 12: Where the plug-in adjustment $M^*$ fails '
             '($H_0$ true in every cell, $N=10{,}000$, nominal 0.05)\n')
    L.append('| configuration | median denom. | $P(\\text{denom}\\le0)$ | $P(\\text{denom}<0.25)$ | '
             'defined | clipped | unclipped | constrained |')
    L.append('| --- | --- | --- | --- | --- | --- | --- | --- |')
    mv_rows = [('MVN', 'normal', 2, 2, 10), ('MVN', 'normal', 2, 2, 30), ('MVN', 'normal', 2, 2, 100),
               ('MVN', 'normal', 3, 3, 10), ('MVN', 'normal', 3, 3, 30), ('MVN', 'normal', 3, 3, 100),
               ('$t_5$', 't5', 2, 2, 10), ('$t_5$', 't5', 2, 2, 100),
               ('lognormal', 'lognormal', 3, 3, 10), ('lognormal', 'lognormal', 3, 3, 100)]
    for disp, dist, p, k, n in mv_rows:
        key = f'T11_box_m_{dist}_p{p}_k{k}'
        rec = mv[key]
        i = int(np.where(np.array(rec['ns']) == n)[0][0])
        L.append(f"| {disp}, $p={p},k={k},n={n}$ | {rec['median_denom'][i]:.2f} | "
                 f"{100*rec['frac_denom_le0'][i]:.1f}% | {100*rec['frac_denom_lt_quarter'][i]:.1f}% | "
                 f"{100*rec['frac_defined'][i]:.1f}% | {rec['type1_Mstar'][i]:.2f} | "
                 f"{rec['type1_Mstar_unclipped'][i]:.2f} | {rec['type1_Mstar_floor025'][i]:.2f} |")

    # ---------------- Table 13 (paper): paired comparison ----------------
    L.append('\n## Table 13: Paired size comparison (same replications), $n=50$, $N=10{,}000$. '
             '$\\Delta$ is corrected minus uncorrected\n')
    L.append('| correction / scenario | $\\alpha=0.01$ | $\\alpha=0.025$ | $\\alpha=0.05$ | '
             '$\\alpha=0.10$ | $\\Delta$ at 0.05 [95% CI] |')
    L.append('| --- | --- | --- | --- | --- | --- |')
    pair_rows = [('bartlett', 'baseline', 'Bartlett $B\\to B^*$, normal'),
                 ('bartlett', 'skewed', 'Bartlett $B\\to B^*$, Exp'),
                 ('bartlett', 'heavy_tail', 'Bartlett $B\\to B^*$, $t_5$'),
                 ('bartlett', 'unbalanced', 'Bartlett $B\\to B^*$, unbalanced'),
                 ('sign', 'baseline', 'sign exact $\\to$ mid-$p$, normal'),
                 ('sign', 'skewed', 'sign exact $\\to$ mid-$p$, Exp'),
                 ('sign', 'heavy_tail', 'sign exact $\\to$ mid-$p$, $t_5$')]
    for fam, sc, lab in pair_rows:
        rec = paired[fam][sc]
        cell = rec['0.05']
        cells_txt = ' | '.join(f"{rec[f'{a:g}']['size_before']:.3f} → "
                               f"{rec[f'{a:g}']['size_after']:.3f}"
                               for a in paired['alphas'])
        L.append(f"| {lab} | {cells_txt} | {cell['diff']:+.3f} "
                 f"[{cell['ci_lo']:+.3f}, {cell['ci_hi']:+.3f}] |")

    # ---------------- Table 14 (paper): attainable sizes of the sign-test rules ----------------
    n_sig = paired.get('n', 50)
    kk = np.arange(n_sig + 1)
    pmf = stats.binom.pmf(kk, n_sig, 0.5)
    cdf = np.cumsum(pmf)
    sf = 1 - cdf + pmf
    p_ex = np.clip(2 * np.minimum(cdf, sf), 0, 1)
    p_mid = np.clip(2 * np.minimum(cdf - 0.5 * pmf, sf - 0.5 * pmf), 0, 1)
    L.append(f'\n## Table 14: Exact attainable sizes of the two sign-test rules, n={n_sig} '
             '(binomial, no simulation)\n')
    L.append('| nominal $\\alpha$ | exact attainable size | mid-$p$ attainable size | '
             'abs. dev. (exact) | abs. dev. (mid-$p$) |')
    L.append('| --- | --- | --- | --- | --- |')
    for a in paired['alphas']:
        se = float(pmf[p_ex <= a].sum())
        sm = float(pmf[p_mid <= a].sum())
        L.append(f'| {a:.3f} | {se:.5f} | {sm:.5f} | {abs(se-a):.5f} | {abs(sm-a):.5f} |')

    # ---------------- Table 15 (paper): size and power ----------------
    L.append('\n## Table 15: Size and power at $\\alpha=0.05$ for four tests with a specified alternative\n')
    L.append('| test | alternative | size (uncorr.) | power (raw) | power (size-adj.) | '
             'size (corr.) | power, corr. (raw) | power, corr. (size-adj.) |')
    L.append('| --- | --- | --- | --- | --- | --- | --- | --- |')
    def _prow(tid, label):
        rec = power['tests'][tid]
        return (f"| {label} | {rec['alternative']} | {rec['size']:.3f} | {rec['power']:.3f} | "
                f"{rec['power_size_adjusted']:.3f} | "
                + ('— | — | — |' if 'power_corrected' not in rec else
                   f"{rec['size_corrected']:.3f} | {rec['power_corrected']:.3f} | "
                   f"{rec['power_size_adjusted_corrected']:.3f} |"))
    L.append(_prow('T1_t_test', 'T1: $t$'))
    L.append(_prow('T2_anova', 'T2: ANOVA $F$'))
    L.append(_prow('T5_bartlett', 'T5: Bartlett'))
    L.append(_prow('T7_sign_test', 'T7: sign test'))
    pt = power['tests']
    def _realised(tid):
        rec = pt[tid]
        s = f"{rec['size_adjusted']:.3f}"
        if 'size_adjusted_corrected' in rec:
            s += f"/{rec['size_adjusted_corrected']:.3f}"
        return s
    L.append(f"\nThe size-adjusted column uses a randomised test of nominal level exactly 5% "
             f"(boundary atoms split with a U(0,1) draw) calibrated on an independent null sample of "
             f"N={power.get('N_calibration', power['N']):,}; the realised level of the calibrated rule "
             f"is measured on a second, independent null sample. Realised size (uncorrected/corrected): "
             f"T1 {_realised('T1_t_test')}, T2 {_realised('T2_anova')}, "
             f"T5 {_realised('T5_bartlett')}, T7 {_realised('T7_sign_test')}.")
    L.append('Rows whose corrected column is absent have no correction.')

    # ---------------- Table 16 (paper): KS resolution ----------------
    L.append('\n## Table 16: KS distance at three Monte Carlo resolutions\n')
    L.append('| test | $n=10$: $10^4$ / $5\\cdot10^4$ / $2\\cdot10^5$ | '
             '$n=50$: $10^4$ / $5\\cdot10^4$ / $2\\cdot10^5$ | $n=200$: $10^4$ / $5\\cdot10^4$ |')
    L.append('| --- | --- | --- | --- |')
    tid_labels = [('T3_kruskal_wallis', 'T3: Kruskal–Wallis'), ('T4_chisq_gof', 'T4: $\\chi^2$ GOF'),
                  ('T6_fligner_killeen', 'T6: Fligner–Killeen'), ('T8_cochran_q', "T8: Cochran's $Q$"),
                  ('T9_median_test', 'T9: median test')]
    for tid, lab in tid_labels:
        rec = ks_res['tests'][tid]
        parts = []
        for n in (10, 50, 200):
            vals = []
            if n in rec['ns']:
                vals.append(f"{rec['ks_10k'][rec['ns'].index(n)]:.4f}")
            else:
                vals.append('—')
            if n in rec['ns']:
                vals.append(f"{rec['ks_big'][rec['ns'].index(n)]:.4f}")
            else:
                vals.append('—')
            if n in rec.get('ns_huge', []):
                vals.append(f"{rec['ks_huge'][rec['ns_huge'].index(n)]:.4f}")
            else:
                vals.append('—')
            parts.append(' / '.join(vals))
        L.append(f"| {lab} | {parts[0]} | {parts[1]} | {parts[2]} |")

    # ---------------- Table 17 (paper): resampling schemes ----------------
    L.append('\n## Table 17: Resampling schemes for Box\'s $M$ under the null '
             f"($\\alpha=0.05$, $N={boot['N_outer']}$ outer replications, "
             f"$B={boot['B']}$ resamples)\n")
    L.append('| null cell | parametric (pooled normal) | within-group, centred | '
             'null-imposed (common covariance) | permutation |')
    L.append('| --- | --- | --- | --- | --- |')
    for label, cell in boot['cells'].items():
        t = cell['type1']
        L.append(f"| {label} | {t['parametric']:.3f} | {t['within_centred']:.3f} | "
                 f"{t['null_imposed']:.3f} | {t['permutation']:.3f} |")
    L.append('\nThe last two cells have equal covariance but *different* distributions, so '
             'exchangeability fails there and permutation is not expected to be valid; they are the '
             'cells that discriminate the null-imposed scheme from permutation.')

    # ---------------- Table A1 (Appendix A): Box's M asymptotic weights ----------------
    bw = results.get('box_m_weights', {})
    if bw:
        L.append('\n## Table A1: Asymptotic weight decomposition of the Box\'s $M$ limit '
                 '(per contrast; multiply each weight by $(1-c)$ for the Bartlett-corrected $M$)\n')
        L.append('| parent | $p=2$: weights (multiplicity) | $p=3$: weights (multiplicity) | '
                 'sum, $p=2$ | sum, $p=3$ | equal weights? |')
        L.append('| --- | --- | --- | --- | --- | --- |')
        def _wg(rec):
            def _fmt(w):
                return f"{int(round(w))}" if abs(w - round(w)) < 1e-9 else f"{w:.2f}"
            return ', '.join(f"{_fmt(g['weight'])} ($\\times${g['multiplicity']})"
                             for g in rec['weight_groups'])
        for dist in ('normal', 't5', 'lognormal'):
            a = bw['weights'][f'p2_{dist}']
            b = bw['weights'][f'p3_{dist}']
            eq = 'yes' if a['equal_weights'] else 'no'
            L.append(f"| {dist} | {_wg(a)} | {_wg(b)} | {a['sum_per_contrast']:.2f} | "
                     f"{b['sum_per_contrast']:.2f} | {eq} |")
        L.append('\nFor $k$ groups each per-contrast weight is repeated $k-1$ times, so the limit '
                 'law has $\\nu=p(p+1)(k-1)/2$ terms. Under normality all weights equal 1 and the '
                 'sum is $\\nu$; under $t_5$ and lognormal the weights differ, so no scalar multiple '
                 'of $M$ is $\\chi^2_\\nu$.')

    with open(out_path, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print(f'  Saved {os.path.basename(out_path)}')


# ============================================================
# SECTION 6B: Real-data case studies
# ============================================================

def pooled_excess_kurtosis_real(groups):
    g2s = []
    for g in groups:
        g = np.asarray(g, dtype=float)
        m2 = ((g - g.mean())**2).mean()
        m4 = ((g - g.mean())**4).mean()
        g2s.append(m4 / max(m2**2, 1e-15) - 3)
    return np.mean(g2s)

def bartlett_statistic_real(groups):
    k = len(groups)
    vars_i = np.array([np.asarray(g, dtype=float).var(ddof=1) for g in groups])
    ns = np.array([len(g) for g in groups])
    n_total = sum(ns)
    sp2 = sum((ns[j]-1)*vars_i[j] for j in range(k)) / (n_total - k)
    if sp2 <= 0:
        return np.nan
    num = (n_total - k)*np.log(sp2) - sum((ns[j]-1)*np.log(max(vars_i[j], 1e-15)) for j in range(k))
    den = 1 + (1/(3*(k-1)))*(sum(1/(ns[j]-1) for j in range(k)) - 1/(n_total-k))
    return num / den

def sign_test_pvalue_real(data, median0):
    data = np.asarray(data, dtype=float)
    n = len(data)
    n_plus = (data > median0).sum()
    ties = (data == median0).sum()
    n_eff = n - ties
    n_p = n_plus
    p_lower = stats.binom.cdf(min(n_p, n_eff - n_p), n_eff, 0.5)
    return min(2 * p_lower, 1.0), n_eff, n_p

def sign_test_midp_real(data, median0):
    data = np.asarray(data, dtype=float)
    n = len(data)
    n_plus = (data > median0).sum()
    ties = (data == median0).sum()
    n_eff = n - ties
    n_p = n_plus
    pl = stats.binom.cdf(n_p - 1, n_eff, 0.5) + 0.5 * stats.binom.pmf(n_p, n_eff, 0.5)
    pu = stats.binom.sf(n_p, n_eff, 0.5) + 0.5 * stats.binom.pmf(n_p, n_eff, 0.5)
    return min(2 * min(pl, pu), 1.0)

def run_real_data_case_studies():
    """Real-data case studies using public datasets (bundled, no download).
    Case 1: Wine malic_acid — Bartlett vs. kurtosis-corrected B*
    Case 2: Wine alcohol — near-normal contrast (B* ≈ B)
    Case 3: mtcars mpg — sign test lattice floor (exact vs. mid-p flip)
    Returns a dict with results suitable for the paper."""
    print("\n" + "=" * 60)
    print("REAL-DATA CASE STUDIES")
    print("=" * 60)

    results = {}

    # --- Load datasets ---
    # Wine (sklearn, bundled)
    from sklearn.datasets import load_wine
    wine = load_wine()

    # mtcars mpg (embedded, 1974 Motor Trend, public-domain)
    mtcars_mpg = np.array([
        21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2,
        17.8, 16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 32.4, 30.4, 33.9,
        21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26.0, 30.4, 15.8, 19.7,
        15.0, 21.4
    ])

    # -- Case 1: malic_acid (skewed, pooled γ₂ ≈ 1.66) --
    print("\n  Case 1: Wine malic_acid — Bartlett vs B* (smooth kurtosis mechanism)")
    idx_malic = 1
    groups_malic = [wine.data[wine.target == c, idx_malic] for c in range(3)]
    ek_m = pooled_excess_kurtosis_real(groups_malic)
    B_m = bartlett_statistic_real(groups_malic)
    Bs_m = B_m / max(1 + ek_m/2, 0.01)
    p_B_m = 1 - stats.chi2.cdf(B_m, len(groups_malic)-1)
    p_Bs_m = 1 - stats.chi2.cdf(Bs_m, len(groups_malic)-1)
    print(f"    γ̂₂={ek_m:.3f}, B={B_m:.3f} (p={p_B_m:.5f}), B*={Bs_m:.3f} (p={p_Bs_m:.5f})")

    results['case1_malic'] = {
        'gamma2_pool': float(ek_m), 'B': float(B_m), 'p_B': float(p_B_m),
        'B_star': float(Bs_m), 'p_B_star': float(p_Bs_m),
        'correction_factor': float(1 + ek_m/2)
    }

    # -- Case 2: alcohol (near-normal, γ₂ ≈ -0.22) --
    print("  Case 2: Wine alcohol — near-normal contrast (B* ≈ B)")
    idx_alc = 0
    groups_alc = [wine.data[wine.target == c, idx_alc] for c in range(3)]
    ek_a = pooled_excess_kurtosis_real(groups_alc)
    B_a = bartlett_statistic_real(groups_alc)
    Bs_a = B_a / max(1 + ek_a/2, 0.01)
    p_B_a = 1 - stats.chi2.cdf(B_a, len(groups_alc)-1)
    p_Bs_a = 1 - stats.chi2.cdf(Bs_a, len(groups_alc)-1)
    print(f"    γ̂₂={ek_a:.3f}, B={B_a:.3f} (p={p_B_a:.5f}), B*={Bs_a:.3f} (p={p_Bs_a:.5f})")

    results['case2_alcohol'] = {
        'gamma2_pool': float(ek_a), 'B': float(B_a), 'p_B': float(p_B_a),
        'B_star': float(Bs_a), 'p_B_star': float(p_Bs_a),
        'correction_factor': float(1 + ek_a/2)
    }

    # -- Case 3: mtcars sign test, H₀: median = 16 --
    print("  Case 3: mtcars sign test — lattice floor (exact vs. mid-p flip)")
    n_mt = len(mtcars_mpg)
    H0_val = 16.0
    p_ex, n_eff, n_p = sign_test_pvalue_real(mtcars_mpg, H0_val)
    p_md = sign_test_midp_real(mtcars_mpg, H0_val)
    print(f"    n={n_mt}, H₀: median={H0_val}, N⁺={n_p}/{n_eff}, "
          f"exact p={p_ex:.5f}, mid-p={p_md:.5f}")

    # Find the gap in achievable α levels (largest ≤ 0.05 and next > 0.05)
    alpha_gap_lo = 0.0
    alpha_gap_hi = 1.0
    for k in range(0, n_mt + 1):
        prob = 2 * stats.binom.cdf(min(k, n_mt - k), n_mt, 0.5)
        if prob <= 0.05 and prob > alpha_gap_lo:
            alpha_gap_lo = prob
        if prob > 0.05 and prob < alpha_gap_hi:
            alpha_gap_hi = prob

    results['case3_mtcars'] = {
        'n': n_mt, 'H0_median': H0_val, 'N_plus': int(n_p), 'n_eff': int(n_eff),
        'p_exact': float(p_ex), 'p_midp': float(p_md),
        'alpha_gap_lo': float(alpha_gap_lo) if alpha_gap_lo else None,
        'alpha_gap_hi': float(alpha_gap_hi) if alpha_gap_hi else None,
        'conclusion_exact': 'reject' if p_ex <= 0.05 else 'not reject',
        'conclusion_midp': 'reject' if p_md <= 0.05 else 'not reject',
    }

    results['_timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    return results

# ============================================================
# SECTION 7: Multivariate extension — Box's M test (T11) with
# Mardia multivariate kurtosis correction
# ============================================================

def mardia_multivariate_kurtosis(X):
    """
    Compute Mardia's multivariate sample kurtosis b_{2,p}.
    X: (n, p) numpy array
    Returns: b_{2,p} (scalar)
    """
    n, p = X.shape
    Xc = X - X.mean(axis=0)
    S = (Xc.T @ Xc) / n  # ML covariance (divisor n)
    try:
        invS = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        invS = np.linalg.pinv(S)
    D2 = np.sum((Xc @ invS) * Xc, axis=1)  # squared Mahalanobis distances
    b2p = np.mean(D2**2)
    return b2p

def pooled_mardia_kurtosis(groups):
    """
    Pooled Mardia kurtosis across k groups.
    groups: list of (n_j, p) numpy arrays
    """
    total_n = sum(g.shape[0] for g in groups)
    b2p = sum(g.shape[0] * mardia_multivariate_kurtosis(g) for g in groups) / total_n
    return b2p

def box_m_statistic(groups):
    """
    Compute Box's M statistic for equality of covariance matrices.
    groups: list of (n_j, p) numpy arrays
    Returns: M_raw, M (Bartlett-corrected), c, nu (degrees of freedom)
    """
    k = len(groups)
    p = groups[0].shape[1]
    n_j = np.array([g.shape[0] for g in groups], dtype=float)
    n = n_j.sum()

    # Sample covariance matrices (divisor n_j - 1)
    S_j = []
    for g in groups:
        Xc = g - g.mean(axis=0)
        S_j.append((Xc.T @ Xc) / (g.shape[0] - 1))

    # Pooled covariance matrix
    S_p = sum((n_j[i] - 1) * S_j[i] for i in range(k)) / (n - k)

    # Raw M statistic
    _, logdet_p = np.linalg.slogdet(S_p)
    M_raw = (n - k) * logdet_p
    for i in range(k):
        _, logdet_j = np.linalg.slogdet(S_j[i])
        M_raw -= (n_j[i] - 1) * logdet_j

    # Bartlett correction factor (Box, 1949)
    c = ((2*p**2 + 3*p - 1) / (6 * (p+1) * (k-1))) * \
        (np.sum(1.0 / (n_j - 1)) - 1.0 / (n - k))

    M = max((1 - c) * M_raw, 0.0)

    # Degrees of freedom
    nu = p * (p + 1) * (k - 1) / 2

    return M_raw, M, c, nu

def box_m_kurtosis_corrected(groups):
    """
    Compute kurtosis-corrected Box's M statistic M*.
    Uses pooled Mardia multivariate kurtosis.
    Returns: M_star, b2p_pooled, denom, nu
    """
    _, M, _, nu = box_m_statistic(groups)
    p = groups[0].shape[1]

    b2p = pooled_mardia_kurtosis(groups)
    # Denominator: 1 + (b_{2,p} - p(p+2)) / 2
    denom = 1.0 + (b2p - p * (p + 2)) / 2.0
    # Floor at small positive value to prevent division by zero
    denom = max(denom, 1e-6)

    M_star = M / denom
    return M_star, b2p, denom, nu

def box_m_kurtosis_corrected_bc(groups):
    """
    Compute kurtosis-corrected Box's M statistic M* using the bias-corrected
    pooled Mardia multivariate kurtosis b_{2,p} (finite-sample correction
    E[b_{2,p}] = p(p+2)(n-1)/(n+1) under normality, so raw values are
    multiplied by (n+1)/(n-1)).
    Returns: M_star_bc, b2p_pooled_bc, denom, nu
    """
    _, M, _, nu = box_m_statistic(groups)
    p = groups[0].shape[1]

    # Pool the per-group bias-corrected kurtosis values
    total_n = 0.0
    b2p_sum = 0.0
    for g in groups:
        _, b2p_bc = mardia_b2p_bc(g)
        if np.isnan(b2p_bc):
            continue
        b2p_sum += g.shape[0] * b2p_bc
        total_n += g.shape[0]
    b2p = b2p_sum / total_n if total_n > 0 else np.nan
    denom = 1.0 + (b2p - p * (p + 2)) / 2.0
    # Floor at small positive value to prevent division by zero
    denom = max(denom, 1e-6)

    M_star = M / denom
    return M_star, b2p, denom, nu

def _gen_mv_group(p, n, dist):
    """Generate one (n, p) group under H0. All groups share the same
    mean/covariance structure, so H0 (equal covariance matrices) holds while
    non-normality is present for the non-normal settings."""
    if dist == 'normal':
        return np.random.normal(0, 1, (n, p))
    elif dist == 't5':
        # Multivariate t with df=5: Z / sqrt(chi2/df), Z ~ MVN(0, I_p)
        Z = np.random.normal(0, 1, (n, p))
        chi2 = np.random.chisquare(5, n)
        return Z / np.sqrt(chi2 / 5.0)[:, None]
    elif dist == 'lognormal':
        # Multivariate lognormal: exp(X), X ~ MVN(0, I_p)
        return np.exp(np.random.normal(0, 1, (n, p)))
    raise ValueError(f'unknown multivariate distribution: {dist}')

def mardia_b2p_bc(X):
    """Bias-corrected Mardia multivariate kurtosis for a single group.

    Uses the finite-sample multiplicative correction based on the known
    expectation under multivariate normality: E[b_{2,p}] = p(p+2) * (n-1)/(n+1).
    The kurtosis is computed with the ML covariance (divisor n) as in
    ``mardia_multivariate_kurtosis``, which is the convention for which the
    expectation above (and hence the (n+1)/(n-1) correction) holds.
    Returns (b2p_raw, b2p_bc).  Returns (nan, nan) if the sample is degenerate.
    """
    n, p = X.shape
    if n <= max(p + 2, 5):
        return np.nan, np.nan
    Xc = X - X.mean(axis=0)
    S = (Xc.T @ Xc) / n  # ML covariance (divisor n), matching Mardia's definition
    try:
        Sinv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    D2 = np.einsum('ij,ij->i', Xc @ Sinv, Xc)   # squared Mahalanobis distances
    b2p_raw = np.mean(D2 ** 2)
    # Finite-sample multiplicative correction
    b2p_bc = b2p_raw * (n + 1.0) / (n - 1.0)
    return b2p_raw, b2p_bc

def mc_box_m(p, k, n, N, dist='normal'):
    """N Monte Carlo replications of Box's M and of the Mardia-kurtosis
    "corrections" under H0 (equal covariance matrices).

    M* is the plug-in M/(1 + (b_2p - p(p+2))/2) exactly as a user would apply
    it: the denominator is floored at 1e-6, so a negative or near-zero
    denominator produces an enormous statistic.  To separate the sources of
    failure we also return the un-clipped statistic (NaN where the denominator
    is non-positive), a constrained control M/max(denominator, 0.25), and the
    denominator itself.

    Returns a dict of arrays:
        M                Bartlett-corrected statistic (clipped at 0 by box_m_statistic)
        Mstar            plug-in M* as implemented (denominator floored at 1e-6)
        Mstar_bc         bias-corrected kurtosis version, same flooring
        Mstar_raw        M/denominator, NaN where denominator <= 0
        Mstar_bc_raw     bias-corrected version, NaN where denominator <= 0
        Mstar_floor025   M/max(denominator, 0.25): a correction that can never
                         inflate by more than a factor 4
        denom, denom_bc  the two denominators, before any flooring
        b2p              pooled Mardia kurtosis
    """
    out = {k2: np.empty(N) for k2 in
           ('M', 'Mstar', 'Mstar_bc', 'Mstar_raw', 'Mstar_bc_raw', 'Mstar_floor025',
            'denom', 'denom_bc', 'b2p')}
    nu = None
    for i in range(N):
        groups = [_gen_mv_group(p, n, dist) for _ in range(k)]
        _, M, _, nu = box_m_statistic(groups)
        b2p = pooled_mardia_kurtosis(groups)
        denom = 1.0 + (b2p - p * (p + 2)) / 2.0
        b2p_bc = sum(g.shape[0] * mardia_b2p_bc(g)[1] for g in groups) / sum(
            g.shape[0] for g in groups)
        denom_bc = 1.0 + (b2p_bc - p * (p + 2)) / 2.0
        Mstar, _, _, _ = box_m_kurtosis_corrected(groups)
        Mstar_bc, _, _, _ = box_m_kurtosis_corrected_bc(groups)
        out['M'][i] = M
        out['Mstar'][i] = Mstar
        out['Mstar_bc'][i] = Mstar_bc
        out['Mstar_raw'][i] = M / denom if denom > 0 else np.nan
        out['Mstar_bc_raw'][i] = M / denom_bc if denom_bc > 0 else np.nan
        out['Mstar_floor025'][i] = M / max(denom, 0.25)
        out['denom'][i] = denom
        out['denom_bc'][i] = denom_bc
        out['b2p'][i] = b2p
    out['nu'] = nu
    return out

def bootstrap_box_m_type1(p, k, n, dist='normal', B_test=None, B_boot=None, alpha=0.05):
    """Parametric-bootstrap Type I error of the standard Bartlett-corrected
    Box's M test under H0 (equal covariance matrices).

    For each of B_test datasets drawn under H0, the observed M is compared
    with B_boot parametric resamples from MVN(0, S_pooled); the reported Type I
    error is the fraction of bootstrap p-values below alpha.
    """
    B_test = BOOT_B_TEST if B_test is None else B_test
    B_boot = BOOT_B_BOOT if B_boot is None else B_boot
    rejects = 0
    for _ in range(B_test):
        groups = [_gen_mv_group(p, n, dist) for _ in range(k)]
        _, M_obs, _, _ = box_m_statistic(groups)
        # Pooled covariance under H0 (identical for every group)
        n_j = np.array([g.shape[0] for g in groups], dtype=float)
        n_tot = n_j.sum()
        S_pool = sum((n_j[i] - 1) * ((g - g.mean(axis=0)).T @ (g - g.mean(axis=0))) / (g.shape[0] - 1)
                     for i, g in enumerate(groups)) / (n_tot - k)
        cnt = 0
        for _ in range(B_boot):
            bg = [np.random.multivariate_normal(np.zeros(p), S_pool, size=n) for _ in range(k)]
            _, M_b, _, _ = box_m_statistic(bg)
            cnt += (M_b >= M_obs)
        p_val = (1.0 + cnt) / (1.0 + B_boot)
        rejects += (p_val < alpha)
    return rejects / B_test

def run_heterokurtosis_check(N_mc=20000, n=50, alpha=0.05):
    """Bartlett B and the kurtosis-corrected B* when variances are equal but
    the group kurtoses are not.

    Three groups with unit variance: N(0,1), t_5/sqrt(5/3) and Exp(1)-1.
    The null of Bartlett's test (equal variances) holds, while the group excess
    kurtoses are 0, 6 and 6.  The plug-in denominator uses the average kurtosis.
    Two homokurtosis cells are reported as references.  The point is to check
    whether B* remains a first-order improvement away from homokurtosis: it does
    remove most of the inflation, but it does not restore calibration.
    """
    rng = np.random.default_rng(SEED + 303)
    k = 3

    def _bartlett(Xa):
        N, kk, nn = Xa.shape
        vars_i = Xa.var(axis=2, ddof=1)
        sp2 = ((nn - 1) * vars_i).sum(axis=1) / (kk * nn - kk)
        num = (kk * nn - kk) * np.log(sp2) - (nn - 1) * np.log(vars_i).sum(axis=1)
        den = 1 + (1 / (3 * (kk - 1))) * (kk / (nn - 1) - 1 / (kk * nn - kk))
        B = num / den
        mu = Xa.mean(axis=2, keepdims=True)
        m2 = ((Xa - mu) ** 2).mean(axis=2)
        m4 = ((Xa - mu) ** 4).mean(axis=2)
        g2 = m4 / np.maximum(m2 ** 2, 1e-15) - 3
        return B, B / (1 + g2.mean(axis=1) / 2), g2.mean(axis=1)

    def _gen(spec):
        groups = []
        for s in spec:
            if s == 'normal':
                groups.append(rng.normal(size=(N_mc, n)))
            elif s == 't5':
                groups.append(rng.standard_t(5, size=(N_mc, n)) / np.sqrt(5.0 / 3.0))
            elif s == 'exp':
                groups.append(rng.exponential(size=(N_mc, n)) - 1.0)
        return np.stack(groups, axis=1)

    cells = {
        'all_normal': ['normal'] * k,
        'all_exponential': ['exp'] * k,
        'heterokurtosis': ['normal', 't5', 'exp'],
    }
    q = float(stats.chi2.ppf(1.0 - alpha, k - 1))
    out = {'N': N_mc, 'n': n, 'alpha': alpha, 'cells': {}}
    for name, spec in cells.items():
        B, Bstar, g2 = _bartlett(_gen(spec))
        out['cells'][name] = dict(
            parents=spec,
            mean_excess_kurtosis=float(np.mean(g2)),
            type1_B=float(np.mean(B > q)),
            type1_B_star=float(np.mean(Bstar > q)),
            mean_B=float(np.mean(B)),
            mean_B_star=float(np.mean(Bstar)))
    print("\n" + "=" * 60)
    print("HETEROKURTOSIS CHECK: B vs B* with equal variances, unequal kurtoses")
    print("=" * 60)
    for name, rec in out['cells'].items():
        print(f"  {name:16s}: mean gamma2={rec['mean_excess_kurtosis']:.2f}, "
              f"B={rec['type1_B']:.3f}, B*={rec['type1_B_star']:.3f}")
    return out


def box_m_asymptotic_weights(p, dist='normal', nu_t=5.0):
    """Asymptotic weights of Box's M_raw limit Q = sum_j w_j chi^2_{1,j}.

    Under H0 with equal group sizes the linear term of the log-determinant
    expansion cancels and, with H_j = sqrt(n_j-1) Sigma^{-1/2}(S_j-Sigma)Sigma^{-1/2},

        M_raw ~ (1/2) [ sum_j tr(H_j^2) - (1/k) tr((sum_j H_j)^2) ],

    a quadratic form in a Gaussian vector whose covariance is
    Omega_{ab,cd} = c^{-2} [ kappa_{abcd} + c^2 (delta_{ac}delta_{bd}
    + delta_{ad}delta_{bc}) ].  For Sigma = c I the eigenvalues of Omega
    restricted to symmetric matrices give the weights; each weight is repeated
    (k-1) times.  Only fourth cumulants enter, which is consistent with the
    fourth-moment condition of Proposition 4.

    Parent conventions in this script (important: mixed cumulants are NOT
    universal constants of the marginal distribution):

    * normal -- Gaussian, kappa_aaaa = kappa_{aabb} = 0.
    * t5 -- ONE shared chi-square scale for all p coordinates (multivariate t
      with identity scale), so coordinates are dependent and
      kappa_{aabb} = 2 c^2 / (nu - 4), kappa_aaaa = 6 nu^2/((nu-2)^2(nu-4)).
    * lognormal -- independent coordinates X_a = exp(Z_a), Z_a iid N(0,1).
      Hence kappa_{aabb} = kappa_{abab} = kappa_{abba} = 0 for a != b.  The
      diagonal cumulant must be computed from CENTRED moments because
      E[X] = e^{1/2} != 0:
          kappa_aaaa = E[(X-mu)^4] - 3 c^2
                     = e^8 - 4 e^5 + 6 e^3 - 3 e^2 - 3 c^2.
      (Using the raw moment e^8, or a raw-product K22 = e^4 - c^2, is wrong
      and inflates the diagonal weight.)

    Returns per-contrast weights (k=2, i.e. one contrast) and their grouping.
    """
    e = np.e
    if dist == 'normal':
        c, K4, K22 = 1.0, 0.0, 0.0
    elif dist == 't5':
        # Multivariate t from a single shared chi-square scale; coordinates
        # are dependent, so the mixed cumulant K22 is non-zero.
        nu = float(nu_t)
        c = nu / (nu - 2.0)
        K4 = 6.0 * nu ** 2 / ((nu - 2.0) ** 2 * (nu - 4.0))
        K22 = 2.0 * c ** 2 / (nu - 4.0)
    elif dist == 'lognormal':
        # Independent coordinates: exp(N(0,1)) independently in each column,
        # so all mixed fourth cumulants vanish.  Centring matters because the
        # lognormal mean mu = e^{1/2} is not zero.
        c = e ** 2 - e
        mu = np.sqrt(e)                    # E[X] = e^{1/2}
        m2 = e ** 2                        # E[X^2]
        m3 = e ** 4.5                      # E[X^3]
        m4 = e ** 8                        # E[X^4]
        central_m4 = m4 - 4.0 * m3 * mu + 6.0 * m2 * mu ** 2 - 3.0 * mu ** 4
        K4 = central_m4 - 3.0 * c ** 2
        K22 = 0.0
    else:
        raise ValueError(f'unknown parent: {dist}')
    C_all = K4 - 3.0 * K22

    def kappa(a, b, cc, d):
        d_ab_cd = 1.0 if (a == b and cc == d) else 0.0
        d_ac_bd = 1.0 if (a == cc and b == d) else 0.0
        d_ad_bc = 1.0 if (a == d and b == cc) else 0.0
        d_all = 1.0 if (a == b == cc == d) else 0.0
        return K22 * d_ab_cd + K22 * (d_ac_bd + d_ad_bc) + C_all * d_all

    Omega = np.zeros((p * p, p * p))
    for a in range(p):
        for b in range(p):
            for cc in range(p):
                for d in range(p):
                    d_ac_bd = 1.0 if (a == cc and b == d) else 0.0
                    d_ad_bc = 1.0 if (a == d and b == cc) else 0.0
                    Omega[a * p + b, cc * p + d] = (
                        kappa(a, b, cc, d) + c * c * (d_ac_bd + d_ad_bc)) / (c * c)

    basis = []
    for a in range(p):
        for b in range(a, p):
            v = np.zeros(p * p)
            if a == b:
                v[a * p + b] = 1.0
            else:
                v[a * p + b] = 1.0 / np.sqrt(2.0)
                v[b * p + a] = 1.0 / np.sqrt(2.0)
            basis.append(v)
    E = np.array(basis).T
    lam = np.linalg.eigvalsh(E.T @ Omega @ E)
    w = np.sort(lam / 2.0)[::-1]
    groups = []
    for val in w:
        if groups and abs(val - groups[-1][0]) < 1e-8:
            groups[-1] = (groups[-1][0], groups[-1][1] + 1)
        else:
            groups.append((float(val), 1))
    return dict(p=p, dist=dist, weights_per_contrast=[float(x) for x in w],
                weight_groups=[dict(weight=round(v, 6), multiplicity=m)
                               for v, m in groups],
                sum_per_contrast=float(w.sum()),
                equal_weights=bool(np.allclose(w, w[0], rtol=1e-9)),
                max_min_ratio=float(w.max() / w.min()))


def run_weight_decomposition():
    """Asymptotic weight decomposition of Box's M_raw for the three parents."""
    out = {'nu_t': 5.0, 'parents': ['normal', 't5', 'lognormal'], 'p_values': [2, 3],
           'weights': {}}
    for p in (2, 3):
        for dist in out['parents']:
            out['weights'][f'p{p}_{dist}'] = box_m_asymptotic_weights(p, dist,
                                                                     nu_t=out['nu_t'])
    return out


def run_multivariate_simulations(N_mc=10000, alpha=0.05):
    """Multivariate study of Box's M test (T11) for equality of covariance
    matrices across k groups, and its Mardia-kurtosis-corrected version M*.

    Null settings (H0 true, covariances equal):
      - MVN(0, I_p)                  -> baseline
      - multivariate t_5             -> heavy-tailed
      - multivariate lognormal       -> skewed
    Grid: p in {2, 3}, k in {2, 3}, n in {10, 15, 20, 30, 50, 100}.

    Output per configuration: empirical rejection rates at alpha for M, M* and
    the bias-corrected M*_BC, a parametric-bootstrap Type I error for the
    standard M test, KS distance of the empirical null CDFs to chi2_nu, fitted
    convergence exponents, and convergence figures (KS vs n) in the Study-1
    format."""
    sample_sizes = [10, 15, 20, 30, 50, 100]
    dims = [2, 3]
    kvals = [2, 3]
    dist_configs = {
        'normal': 'Multivariate normal',
        't5': 'Multivariate t (df=5)',
        'lognormal': 'Multivariate lognormal',
    }

    results = {}

    for dist, dist_label in dist_configs.items():
        print("\n" + "=" * 60)
        print(f"STUDY MV: Box's M test (T11) — {dist_label}")
        print("=" * 60)
        for p in dims:
            for k in kvals:
                key = f'T11_box_m_{dist}_p{p}_k{k}'
                results[key] = {'ns': [], 'type1_M': [], 'type1_Mstar': [],
                                'type1_Mstar_bc': [], 'type1_boot': [],
                                'type1_Mstar_unclipped': [], 'type1_Mstar_floor025': [],
                                'type1_Mstar_bc_unclipped': [],
                                'frac_denom_le0': [], 'frac_denom_lt_quarter': [],
                                'frac_clipped': [], 'frac_defined': [], 'frac_defined_bc': [],
                                'median_denom': [], 'median_denom_bc': [],
                                'ci_lo_M': [], 'ci_hi_M': [], 'ci_lo_Mstar': [], 'ci_hi_Mstar': [],
                                'ci_lo_Mstar_bc': [], 'ci_hi_Mstar_bc': [],
                                'ks_M': [], 'ks_Mstar': [], 'ks_Mstar_bc': [],
                                'b_M': None, 'b_Mstar': None, 'b_Mstar_bc': None,
                                'nu': None, 'N': N_mc, 'dist': dist_label, 'p': p, 'k': k}
                print(f"\n  {dist_label}: p={p}, k={k} "
                      f"(nu={p*(p+1)*(k-1)/2})...", end=' ', flush=True)
                t0 = time.time()
                for n in sample_sizes:
                    if n <= p:
                        # Box's M requires n_j > p; skip degenerate configurations
                        results[key]['ns'].append(n)  # append n, not NaN, to indicate it was attempted
                        for field in ('type1_M', 'type1_Mstar', 'type1_Mstar_bc', 'type1_boot',
                                      'type1_Mstar_unclipped', 'type1_Mstar_floor025',
                                      'type1_Mstar_bc_unclipped',
                                      'frac_denom_le0', 'frac_denom_lt_quarter',
                                      'frac_clipped', 'frac_defined', 'frac_defined_bc',
                                      'median_denom', 'median_denom_bc',
                                      'ci_lo_M', 'ci_hi_M', 'ci_lo_Mstar', 'ci_hi_Mstar',
                                      'ci_lo_Mstar_bc', 'ci_hi_Mstar_bc',
                                      'ks_M', 'ks_Mstar', 'ks_Mstar_bc'):
                            results[key][field].append(np.nan)
                        continue
                    sim = mc_box_m(p, k, n, N_mc, dist=dist)
                    M_arr = sim['M']
                    nu = sim['nu']
                    results[key]['nu'] = float(nu)
                    results[key]['ns'].append(n)
                    ks_M, _ = compute_ks(M_arr, 'chi2', nu)
                    ks_Ms, _ = compute_ks(sim['Mstar'], 'chi2', nu)
                    ks_Ms_bc, _ = compute_ks(sim['Mstar_bc'], 'chi2', nu)
                    t1e_M = compute_type1_error(M_arr, 'chi2', (nu,), alpha=alpha)
                    t1e_Ms = compute_type1_error(sim['Mstar'], 'chi2', (nu,), alpha=alpha)
                    t1e_Ms_bc = compute_type1_error(sim['Mstar_bc'], 'chi2', (nu,), alpha=alpha)
                    results[key]['type1_M'].append(float(t1e_M))
                    results[key]['type1_Mstar'].append(float(t1e_Ms))
                    results[key]['type1_Mstar_bc'].append(float(t1e_Ms_bc))
                    # ---- diagnostics that separate the four failure sources ----
                    q_crit = float(stats.chi2.ppf(1.0 - alpha, nu))
                    den = sim['denom']
                    den_bc = sim['denom_bc']
                    defined = np.isfinite(sim['Mstar_raw'])
                    defined_bc = np.isfinite(sim['Mstar_bc_raw'])
                    results[key]['frac_denom_le0'].append(float(np.mean(den <= 0)))
                    results[key]['frac_denom_lt_quarter'].append(float(np.mean(den < 0.25)))
                    results[key]['frac_clipped'].append(float(np.mean(den < 1e-6)))
                    results[key]['frac_defined'].append(float(np.mean(defined)))
                    results[key]['frac_defined_bc'].append(float(np.mean(defined_bc)))
                    results[key]['median_denom'].append(float(np.median(den)))
                    results[key]['median_denom_bc'].append(float(np.median(den_bc)))
                    results[key]['type1_Mstar_unclipped'].append(
                        float(np.mean(sim['Mstar_raw'][defined] > q_crit)) if defined.any() else np.nan)
                    results[key]['type1_Mstar_bc_unclipped'].append(
                        float(np.mean(sim['Mstar_bc_raw'][defined_bc] > q_crit))
                        if defined_bc.any() else np.nan)
                    results[key]['type1_Mstar_floor025'].append(
                        float(np.mean(sim['Mstar_floor025'] > q_crit)))
                    results[key]['type1_boot'].append(np.nan)  # filled after the MC loop
                    lo, hi = wilson_ci(t1e_M, N_mc)
                    results[key]['ci_lo_M'].append(float(lo))
                    results[key]['ci_hi_M'].append(float(hi))
                    lo, hi = wilson_ci(t1e_Ms, N_mc)
                    results[key]['ci_lo_Mstar'].append(float(lo))
                    results[key]['ci_hi_Mstar'].append(float(hi))
                    lo, hi = wilson_ci(t1e_Ms_bc, N_mc)
                    results[key]['ci_lo_Mstar_bc'].append(float(lo))
                    results[key]['ci_hi_Mstar_bc'].append(float(hi))
                    results[key]['ks_M'].append(float(ks_M))
                    results[key]['ks_Mstar'].append(float(ks_Ms))
                    results[key]['ks_Mstar_bc'].append(float(ks_Ms_bc))
                # Bootstrap calibration of the standard M test (after the MC loop,
                # since it is independent of the N_mc replications above)
                for idx, n in enumerate(results[key]['ns']):
                    if n <= p:
                        continue  # already NaN
                    t1e_boot = bootstrap_box_m_type1(p, k, int(n), dist=dist,
                                                     alpha=alpha)
                    results[key]['type1_boot'][idx] = float(t1e_boot)
                b_M, _ = fit_convergence_rate(results[key]['ks_M'], results[key]['ns'])
                b_Ms, _ = fit_convergence_rate(results[key]['ks_Mstar'], results[key]['ns'])
                b_Ms_bc, _ = fit_convergence_rate(results[key]['ks_Mstar_bc'], results[key]['ns'])
                results[key]['b_M'] = float(b_M) if not np.isnan(b_M) else None
                results[key]['b_Mstar'] = float(b_Ms) if not np.isnan(b_Ms) else None
                results[key]['b_Mstar_bc'] = float(b_Ms_bc) if not np.isnan(b_Ms_bc) else None
                elapsed = time.time() - t0
                print(f"({elapsed:.1f}s)")

    # ---- summary tables ----
    print("\n--- Study MV: Empirical rejection rates at alpha=0.05 ---")
    header = f"{'Config':<32}" + "".join(f"{'n='+str(n):>8}" for n in sample_sizes)
    print(header)
    print("-" * len(header))
    for key in results:
        row = f"{key:<32}"
        for v in results[key]['type1_M']:
            row += f"{v:>8.4f}" if not np.isnan(v) else f"{'N/A':>8}"
        print(row + "   [M]")
        row = f"{key:<32}"
        for v in results[key]['type1_Mstar']:
            row += f"{v:>8.4f}" if not np.isnan(v) else f"{'N/A':>8}"
        print(row + "   [M*]")
        row = f"{key:<32}"
        for v in results[key]['type1_Mstar_bc']:
            row += f"{v:>8.4f}" if not np.isnan(v) else f"{'N/A':>8}"
        print(row + "   [M*_BC]")
        row = f"{key:<32}"
        for v in results[key]['type1_boot']:
            row += f"{v:>8.4f}" if not np.isnan(v) else f"{'N/A':>8}"
        print(row + "   [boot]")

    print("\n--- Study MV: KS distance vs chi2_nu ---")
    print(header)
    print("-" * len(header))
    for key in results:
        row = f"{key:<32}"
        for v in results[key]['ks_M']:
            row += f"{v:>8.4f}" if not np.isnan(v) else f"{'N/A':>8}"
        print(row + "   [M]")
        row = f"{key:<32}"
        for v in results[key]['ks_Mstar']:
            row += f"{v:>8.4f}" if not np.isnan(v) else f"{'N/A':>8}"
        print(row + "   [M*]")
        row = f"{key:<32}"
        for v in results[key]['ks_Mstar_bc']:
            row += f"{v:>8.4f}" if not np.isnan(v) else f"{'N/A':>8}"
        print(row + "   [M*_BC]")

    print("\n--- Study MV: Fitted convergence exponents b ---")
    for key in results:
        bm = results[key]['b_M']
        bs = results[key]['b_Mstar']
        bb = results[key]['b_Mstar_bc']
        bm_s = f"{bm:.3f}" if bm is not None else "N/A"
        bs_s = f"{bs:.3f}" if bs is not None else "N/A"
        bb_s = f"{bb:.3f}" if bb is not None else "N/A"
        print(f"  {key}: b(M) = {bm_s}, b(M*) = {bs_s}, b(M*_BC) = {bb_s}")

    # Publication figures are generated after all multivariate configurations are collected.

    return results

# ============================================================
# PUBLICATION FIGURES
# ============================================================

COLORS = {
    'blue': '#2F6DB5',
    'orange': '#D97706',
    'green': '#4C956C',
    'purple': '#6F4AA8',
    'red': '#C95D63',
    'gray': '#667085',
    'lightgray': '#D1D5DB',
    'dark': '#1F2937',
    'bg': '#F8FAFC',
    'blue_fill': '#F4F8FD',
    'orange_fill': '#FFF8F1',
    'purple_fill': '#F8F4FC',
    'fig3_orange': '#E64B19',
    'fig3_orange_edge': '#B83C10',
    'fig3_blue': '#4C4BC6',
    'fig3_blue_edge': '#3535A7',
    'fig3_green_band': '#DCEBC3',
    'fig3_green_edge': '#B8CE98',
    'fig3_grid': '#C9CDD2',
    'fig3_ref': '#666666',
    'fig3_connector': '#7A7A7A',
    'fig3_text': '#111111',
}

TEST_LABELS = {
    'T1_t_test': r'T1  Student $t$',
    'T2_anova': 'T2  ANOVA',
    'T3_kruskal_wallis': 'T3  Kruskal-Wallis',
    'T4_chisq_gof': r'T4  Pearson $\chi^2$ GOF',
    'T5_bartlett': 'T5  Bartlett',
    'T6_fligner_killeen': 'T6  Fligner-Killeen',
    'T7_sign_test': 'T7  Sign',
    'T8_cochran_q': r"T8  Cochran's $Q$",
    'T9_median_test': 'T9  Median',
    'T10_hotelling': r'T10 Hotelling $T^2$',
}

def _setup_style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 9.2,
        'axes.titlesize': 10.5,
        'axes.labelsize': 9.5,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'legend.fontsize': 8.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.55,
        'grid.alpha': 0.18,
        'savefig.dpi': 300,
        'figure.dpi': 150,
    })

def _save(fig, out_dir, stem):
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, stem + '.png'), bbox_inches='tight', dpi=320)
    fig.savefig(os.path.join(out_dir, stem + '.pdf'), bbox_inches='tight')
    plt.close(fig)


def _add_fig1_box(ax, x, y, w, h, title, body, edge, fill,
                  title_fs=10.0, body_fs=8.8):
    """Rounded rectangle with a bold title and multiline body (Fig. 1)."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        linewidth=1.35,
        edgecolor=edge,
        facecolor=fill,
        transform=ax.transAxes,
        zorder=2,
    )
    ax.add_patch(box)

    ax.text(
        x + 0.5*w, y + h - 0.16*h, title,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=title_fs, fontweight="bold",
        color=edge,
        zorder=3,
    )

    ax.text(
        x + 0.06*w, y + h - 0.39*h, body,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=body_fs,
        color=COLORS["dark"],
        linespacing=1.35,
        zorder=3,
    )
    return box


def _add_fig1_arrow(ax, x1, y1, x2, y2, color, lw=1.45,
                    dashed=False, mutation_scale=12):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        linestyle="--" if dashed else "-",
        color=color,
        transform=ax.transAxes,
        zorder=4,
        connectionstyle="arc3,rad=0",
    )
    ax.add_patch(arrow)
    return arrow


def _add_fig1_row_label(ax, label_x, y, h, letter, line1, line2, color):
    ax.text(
        label_x, y + h - 0.015, letter,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=18, fontweight="bold",
        color=color,
    )
    ax.text(
        label_x, y + h - 0.072, f"{line1}\n{line2}",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=8.9, fontweight="bold",
        color=color,
        linespacing=1.15,
    )


def fig1_mechanism_framework(out_dir):
    """Conceptual figure: the three mechanisms, plus the orthogonal null-status axis.

    Deliberately built from fixed-position text rather than auto-wrapped callouts,
    so that the layout does not depend on the matplotlib text-metric details.
    """
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(12.6, 7.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    for text, x in [("STATISTIC STRUCTURE", 21),
                    ("MECHANISM: WHAT SETS THE ERROR", 50),
                    ("CALIBRATION RESPONSE", 82)]:
        ax.text(x, 95.5, text, ha='center', va='center', fontsize=10.5, fontweight='bold')

    rows = [
        ("A", "Standardised mean\n$H(\\bar Z)$,  $\\nabla H\\neq 0$",
         "Edgeworth term of order $n^{-1/2}$;\ncoefficient from third cumulants.\nFor non-linear $H$ it is not\nthe parent skewness.",
         "No general correction:\nthe prediction is a rate,\nnot a calibrated statistic.",
         "#1f4e79", "#e9f1f9"),
        ("B", "Quadratic form / LRT\n$\\nabla H=0$ at $H_0$",
         "Limit $Q=\\frac{1}{2}\\sum_j\\lambda_j\\chi^2_{1,j}$;\nweights from second- and\nfourth-moment structure.",
         "All weights equal: scalar\ncorrection possible ($B^*$).\nUnequal weights: no scalar\ncorrection (Box's $M$).",
         "#6a4c93", "#f2ecf8"),
        ("C", "Discrete support\nlattice / rank / table",
         "Continuity term at $n^{-1/2}$.\nExact constant for the symmetric\nbinomial: $D_{KS}=\\varphi(0)n^{-1/2}$.\nOther discrete tests: empirical.",
         "mid-$p$ for the sign test;\nempirical critical values for\nthe lattice $\\chi^2$ tests.",
         "#b45f06", "#fdf2e3"),
    ]
    for (tag, s1, s2, s3, colour, fill), y in zip(rows, (73.0, 47.5, 22.0)):
        ax.add_patch(Rectangle((2, y - 10.5), 96, 21, facecolor=fill, edgecolor=colour, lw=1.6))
        ax.text(6.0, y, tag, fontsize=20, fontweight='bold', color=colour,
                ha='center', va='center')
        ax.text(21, y, s1, ha='center', va='center', fontsize=9.2, linespacing=1.45)
        ax.text(50, y, s2, ha='center', va='center', fontsize=9.2, linespacing=1.45)
        ax.text(82, y, s3, ha='center', va='center', fontsize=9.2, linespacing=1.45)

    ax.add_patch(Rectangle((2, 0.8), 96, 10.5, facecolor='#f3f3f3', edgecolor='#777',
                           lw=1.2, linestyle='--'))
    ax.text(50, 7.4,
            "Orthogonal axis - NULL STATUS: exact under the ideal assumptions for $t$, $F$, $T^2$ and Box's $M$",
            ha='center', va='center', fontsize=8.6)
    ax.text(50, 3.6,
            "(matrix-Dirichlet representation), and for the binomial sign test; asymptotic elsewhere.",
            ha='center', va='center', fontsize=8.6)
    fig.tight_layout()
    _save(fig, out_dir, 'fig1_mechanism_framework')


def _nstar_to_numeric(v):
    s = str(v)
    if s.startswith('>'):
        return 650.0, True
    return float(s), False


def fig2_univariate_convergence(results, out_dir):
    _setup_style()
    fig = plt.figure(figsize=(7.25, 5.55))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.38)
    ax1=fig.add_subplot(gs[0,0]); ax2=fig.add_subplot(gs[0,1])

    # Panel A: asymptotic tests only
    smooth=['T3_kruskal_wallis','T4_chisq_gof','T6_fligner_killeen']
    lattice=['T8_cochran_q','T9_median_test']
    smooth_cols=[COLORS['blue'],COLORS['green'],COLORS['purple']]
    lattice_cols=[COLORS['orange'],COLORS['red']]
    for key,c in zip(smooth,smooth_cols):
        v=results['study1'][key]
        ax1.plot(v['ns'],v['ks'],marker='o',ms=4.2,lw=1.55,color=c,label=TEST_LABELS[key])
    for key,c in zip(lattice,lattice_cols):
        v=results['study1'][key]
        ax1.plot(v['ns'],v['ks'],marker='s',ms=4.1,lw=1.55,ls='--',color=c,label=TEST_LABELS[key])
    nref=np.array([5,500.0])
    # Choose anchor to show slope, not fit a particular test.
    ax1.plot(nref,0.17*(nref/5)**(-0.5),ls=':',lw=1.5,color=COLORS['gray'],label=r'$n^{-1/2}$ reference')
    thr=float(results['study1_ks_threshold'])
    ax1.axhline(thr,color=COLORS['gray'],lw=1.0,ls='-.')
    ax1.text(7.2,thr*1.08,'MC KS floor',fontsize=7.5,color=COLORS['gray'])
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_xlabel('Sample size per group, $n$ (T4 uses total $\\max(5n,30)$)')
    ax1.set_ylabel('KS distance to reference null')
    ax1.set_title('A  KS convergence',loc='left',fontweight='bold')
    ax1.grid(True,which='major',axis='both')
    ax1.legend(frameon=False,loc='upper right',ncol=1,fontsize=7.25)

    # Panel B: n* comparison
    order=list(TEST_LABELS.keys())
    y=np.arange(len(order))[::-1]
    alpha_vals=[]; ks_vals=[]; ks_gt=[]
    for k in order:
        av,_=_nstar_to_numeric(results['study1_nstar_alpha'][k]); kv,gt=_nstar_to_numeric(results['study1_nstar_ks'][k])
        alpha_vals.append(av); ks_vals.append(kv); ks_gt.append(gt)
    for yi,a,k in zip(y,alpha_vals,ks_vals):
        ax2.plot([a,k],[yi,yi],color=COLORS['lightgray'],lw=1.4,zorder=1)
    ax2.scatter(alpha_vals,y,s=28,marker='o',color=COLORS['green'],label=r'$n^*_{\alpha}$',zorder=3)
    ax2.scatter(ks_vals,y,s=31,marker='s',color=COLORS['purple'],label=r'$n^*_{KS}$',zorder=3)
    for yi,x,gt in zip(y,ks_vals,ks_gt):
        if gt:
            ax2.annotate('>500',(x,yi),xytext=(3,0),textcoords='offset points',va='center',fontsize=7.0,color=COLORS['purple'])
    ax2.set_xscale('log'); ax2.set_xlim(4,900)
    ax2.set_yticks(y); ax2.set_yticklabels([TEST_LABELS[k] for k in order],fontsize=7.3)
    ax2.set_xlabel('First $n$ meeting the empirical criterion')
    ax2.set_title('B  Tail vs full-distribution adequacy',loc='left',fontweight='bold')
    ax2.grid(True,axis='x')
    ax2.legend(frameon=False,loc='lower right')
    _save(fig,out_dir,'fig2_univariate_convergence')


def _fig3_bar_value_labels(ax, bars, values, color, decimals=3, dy=0.006):
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + dy,
            f"{value:.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=11,
            color=color,
            fontweight="bold",
            clip_on=False,
        )


def _fig3_style_type1_axis(ax, ylim, yticks):
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_ylabel(
        "Empirical Type I error",
        fontsize=13,
        fontweight="bold",
        labelpad=11,
    )
    ax.grid(
        axis="y",
        color=COLORS["fig3_grid"],
        linestyle=(0, (2.5, 3.0)),
        linewidth=0.8,
        alpha=0.95,
        zorder=0,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="both", labelsize=11, width=0.8, length=4)


def _fig3_type1_reference(ax, xmin, xmax):
    ax.axhspan(
        0.04, 0.06,
        facecolor=COLORS["fig3_green_band"],
        edgecolor="none",
        alpha=0.68,
        zorder=0.5,
    )
    ax.axhline(
        0.05,
        color=COLORS["fig3_ref"],
        linewidth=1.5,
        linestyle=(0, (1.2, 2.0)),
        zorder=1,
    )
    ax.set_xlim(xmin, xmax)


def fig3_univariate_corrections(results, out_dir):
    """Figure 3: Bartlett correction (A) and sign-test lattice attainability (B).

    Panel B is computed directly from the binomial null at n=50, so it has no
    simulation noise; the exact values reproduce Table S7.
    """
    _setup_style()
    fig = plt.figure(figsize=(10.2, 4.15))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.30)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    # ---------------- A: Bartlett correction ----------------
    scenarios = [('baseline', 'Normal'), ('skewed', 'Exponential'),
                 ('heavy_tail', r'$t_5$')]
    xA = np.arange(len(scenarios), dtype=float)
    before = np.array([results['study2']['T5_bartlett'][k] for k, _ in scenarios])
    after = np.array([results['study2_corrections']['T5_bartlett'][k] for k, _ in scenarios])
    bw = 0.34
    b1 = axA.bar(xA - bw/2, before, width=bw, color=COLORS['fig3_orange'],
                 edgecolor=COLORS['fig3_orange_edge'], linewidth=0.8, zorder=3)
    b2 = axA.bar(xA + bw/2, after, width=bw, color=COLORS['fig3_blue'],
                 edgecolor=COLORS['fig3_blue_edge'], linewidth=0.8, zorder=3)
    for bars, vals in ((b1, before), (b2, after)):
        for bar, v in zip(bars, vals):
            axA.text(bar.get_x() + bar.get_width()/2, v + 0.008, f'{v:.3f}',
                     ha='center', va='bottom', fontsize=9, fontweight='bold',
                     color=COLORS['dark'], clip_on=False)
    axA.axhline(0.05, color=COLORS['fig3_ref'], lw=1.3, ls=(0, (1.2, 2.0)), zorder=1)
    axA.axhspan(0.04, 0.06, color=COLORS['fig3_green_band'], alpha=.55, zorder=0)
    axA.set_xticks(xA)
    axA.set_xticklabels([lab for _, lab in scenarios])
    axA.set_ylim(0, 0.48)
    axA.set_ylabel('Empirical Type I error')
    axA.set_title('A  Bartlett correction', loc='left', fontweight='bold', fontsize=10.5)
    axA.grid(True, axis='y', color=COLORS['fig3_grid'], ls=(0, (2.5, 3.0)), lw=.8)
    axA.legend(handles=[Patch(facecolor=COLORS['fig3_orange'], edgecolor=COLORS['fig3_orange_edge'], label=r'Bartlett $B$'),
                        Patch(facecolor=COLORS['fig3_blue'], edgecolor=COLORS['fig3_blue_edge'], label=r'Corrected $B^{*}$')],
               frameon=False, loc='upper left', ncol=1, fontsize=8.5)

    # ---------------- B: sign-test attainable size ----------------
    n_sig = 50
    kk = np.arange(n_sig + 1)
    pmf = stats.binom.pmf(kk, n_sig, 0.5)
    cdf = np.cumsum(pmf)
    sf = 1 - cdf + pmf
    p_ex = np.clip(2 * np.minimum(cdf, sf), 0, 1)
    p_mid = np.clip(2 * np.minimum(cdf - 0.5 * pmf, sf - 0.5 * pmf), 0, 1)
    alphas = np.array([0.010, 0.025, 0.050, 0.100])
    exact = np.array([pmf[p_ex <= a].sum() for a in alphas])
    midp = np.array([pmf[p_mid <= a].sum() for a in alphas])
    axB.plot([0, 0.13], [0, 0.13], ls=':', lw=1.2, color=COLORS['gray'], label='Nominal', zorder=1)
    axB.plot(alphas, exact, 'o-', color=COLORS['fig3_orange'], lw=1.5, ms=5.0,
             label='Exact', zorder=3)
    axB.plot(alphas, midp, 's--', color=COLORS['fig3_blue'], lw=1.5, ms=5.0,
             label=r'Mid-$p$', zorder=3)
    for a, e, m in zip(alphas, exact, midp):
        if a < 0.05:
            continue
        axB.annotate(f'{e:.3f}', (a, e), xytext=(0, -12), textcoords='offset points',
                     ha='center', fontsize=7.5, color=COLORS['fig3_orange_edge'])
        axB.annotate(f'{m:.3f}', (a, m), xytext=(0, 7), textcoords='offset points',
                     ha='center', fontsize=7.5, color=COLORS['fig3_blue_edge'])
    axB.set_xlim(0, 0.115)
    axB.set_ylim(0, 0.13)
    axB.set_xticks(alphas)
    axB.set_xticklabels([f'{a:.3f}' for a in alphas])
    axB.set_xlabel(r'Nominal significance level $\alpha$')
    axB.set_ylabel('Attained size')
    axB.set_title(r'B  Sign-test attainable size ($n=50$)', loc='left',
                  fontweight='bold', fontsize=10.5)
    axB.grid(True, color=COLORS['fig3_grid'], ls=(0, (2.5, 3.0)), lw=.8)
    axB.legend(frameon=False, loc='upper left', fontsize=8.5)

    fig.subplots_adjust(left=0.080, right=0.985, bottom=0.205, top=0.865, wspace=0.30)
    _save(fig, out_dir, 'fig3_univariate_corrections')

def _aggregate_mv(results, dist, metric='type1_M'):
    mats=[]
    ns=None
    for p in [2,3]:
        for k in [2,3]:
            v=results['study_mv'][f'T11_box_m_{dist}_p{p}_k{k}']
            ns=np.array(v['ns'],float)
            mats.append(np.array(v[metric],float))
    A=np.vstack(mats)
    return ns,np.nanmedian(A,axis=0),np.nanmin(A,axis=0),np.nanmax(A,axis=0)


def fig4_multivariate_boxm(results,out_dir):
    _setup_style()
    fig,axes=plt.subplots(2,3,figsize=(7.25,5.35),sharex='col')
    dists=[('normal','MV normal'),('t5',r'Multivariate $t_5$'),('lognormal','Multivariate lognormal')]
    for j,(dist,label) in enumerate(dists):
        ax=axes[0,j]
        ns,m,lo,hi=_aggregate_mv(results,dist,'type1_M')
        ax.fill_between(ns,lo,hi,color=COLORS['red'],alpha=.10,linewidth=0)
        ax.plot(ns,m,'o-',color=COLORS['red'],lw=1.55,ms=4,label='$M$')
        ns,ms,los,his=_aggregate_mv(results,dist,'type1_Mstar')
        ax.fill_between(ns,los,his,color=COLORS['blue'],alpha=.10,linewidth=0)
        ax.plot(ns,ms,'s--',color=COLORS['blue'],lw=1.55,ms=3.9,label='$M^*$')
        ns,ms_bc,los_bc,his_bc=_aggregate_mv(results,dist,'type1_Mstar_bc')
        ax.fill_between(ns,los_bc,his_bc,color=COLORS['green'],alpha=.10,linewidth=0)
        ax.plot(ns,ms_bc,'^-.',color=COLORS['green'],lw=1.55,ms=4.2,label=r'$M^*_{\mathrm{BC}}$')
        ax.axhline(.05,color=COLORS['gray'],lw=1,ls=':')
        ax.fill_between([ns.min(),ns.max()],0.04,0.06,color=COLORS['green'],alpha=.07,linewidth=0)
        ax.set_xscale('log'); ax.set_ylim(0,1.0); ax.grid(True,axis='y')
        ax.set_title(label,fontweight='bold')
        if j==0: ax.set_ylabel('Type I error')
        if j==2:
            handles, labels = ax.get_legend_handles_labels()
            handles.append(Patch(facecolor=COLORS['gray'], alpha=.15,
                                 label='Range across $(p,k)$ designs'))
            ax.legend(handles=handles, frameon=False, loc='center right', fontsize=7.4)
        ax.text(0.02,0.96,chr(ord('A')+j),transform=ax.transAxes,va='top',fontweight='bold',fontsize=10.5)

        axb=axes[1,j]
        ns,m,lo,hi=_aggregate_mv(results,dist,'ks_M')
        axb.fill_between(ns,lo,hi,color=COLORS['red'],alpha=.10,linewidth=0)
        axb.plot(ns,m,'o-',color=COLORS['red'],lw=1.55,ms=4,label='$M$')
        ns,ms,los,his=_aggregate_mv(results,dist,'ks_Mstar')
        axb.fill_between(ns,los,his,color=COLORS['blue'],alpha=.10,linewidth=0)
        axb.plot(ns,ms,'s--',color=COLORS['blue'],lw=1.55,ms=3.9,label='$M^*$')
        ns,ms_bc,los_bc,his_bc=_aggregate_mv(results,dist,'ks_Mstar_bc')
        axb.fill_between(ns,los_bc,his_bc,color=COLORS['green'],alpha=.10,linewidth=0)
        axb.plot(ns,ms_bc,'^-.',color=COLORS['green'],lw=1.55,ms=4.2,label=r'$M^*_{\mathrm{BC}}$')
        axb.axhline(float(results['study1_ks_threshold']),color=COLORS['gray'],lw=1,ls=':')
        axb.set_xscale('log'); axb.set_yscale('log'); axb.grid(True,which='major',axis='y')
        if j==0: axb.set_ylabel('KS distance')
        axb.text(0.02,0.96,chr(ord('D')+j),transform=axb.transAxes,va='top',fontweight='bold',fontsize=10.5)
    # single shared x-label: this figure is about Box's M only, so the
    # "T4 uses total max(5n,30)" note of Figure 2 does not belong here
    fig.supxlabel('Sample size per group, $n$', y=0.025)
    fig.subplots_adjust(top=.93,bottom=.135,left=.09,right=.985,wspace=.26,hspace=.28)
    _save(fig,out_dir,'fig4_multivariate_boxm')


def generate_publication_figures(results, out_dir=None):
    """Generate Figures 2-4 from a committed results dictionary.

    Figure 1 is the static framework graphic figures/framework.png and is not
    regenerated here.  This function intentionally separates plotting from
    simulation so figures can be regenerated quickly and deterministically
    from experiment_results.json.  PNG (320 dpi) and vector PDF versions are
    written for every generated figure.
    """
    if not HAS_MPL:
        raise RuntimeError("matplotlib is required to generate figures")
    if out_dir is None:
        out_dir = FIG_DIR
    fig2_univariate_convergence(results,out_dir)
    fig3_univariate_corrections(results,out_dir)
    fig4_multivariate_boxm(results,out_dir)
    print(f"  Saved publication figures to {_rel(out_dir)}")


# ============================================================
# MAIN
# ============================================================


# --------------------------------------------------------------------------
# Reproducibility: manifest, environment record, and result-schema assertions
# --------------------------------------------------------------------------
SCHEMA_VERSION = '2.0'


def _env_record():
    import platform
    rec = {'python': platform.python_version(),
           'numpy': np.__version__,
           'platform': platform.platform()}
    try:
        import scipy
        rec['scipy'] = scipy.__version__
    except Exception:
        pass
    try:
        import matplotlib
        rec['matplotlib'] = matplotlib.__version__
    except Exception:
        pass
    try:
        import sklearn
        rec['scikit-learn'] = sklearn.__version__
    except Exception:
        pass
    return rec


def _file_sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git_commit():
    import subprocess
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        out = subprocess.run(['git', '-C', here, 'rev-parse', 'HEAD'],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def write_manifest(path, results):
    """Write a manifest recording exactly how the results file was produced."""
    import datetime
    man = {
        'schema_version': SCHEMA_VERSION,
        'generated_at': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
        'git_commit': _git_commit(),
        'seed': SEED,
        'environment': _env_record(),
        'script': 'experiment.py',
        'results_file': os.path.basename(path),
        'study1_tests': sorted(k for k in results['study1'] if not k.startswith('_')),
        'study_mv_configs': sorted(results['study_mv']),
        'exact_null_validation': results['exact_null']['validation'],
        'bootstrap_benchmark': {'B_test': BOOT_B_TEST, 'B_boot': BOOT_B_BOOT,
                                'note': 'parametric bootstrap benchmark; not used by any table or figure'},
        'script_sha256': _file_sha256(os.path.abspath(__file__)),
        'results_sha256': _file_sha256(path),
    }
    with open(os.path.join(os.path.dirname(path) or '.', 'run_manifest.json'), 'w') as fh:
        json.dump(man, fh, indent=2)
    print(f"wrote run_manifest.json")


def check_results(results):
    """Fail loudly if the committed results do not have the schema the paper reads.

    This is the guard against the earlier situation in which the JSON silently
    lacked fields (q95_emp, q95_theory, fitted_b_se) that the tables needed.
    """
    problems = []
    s1 = results['study1']
    if len(s1) != 10:
        problems.append(f"study1 has {len(s1)} tests, expected 10")
    for name, rec in s1.items():
        for field in ('ks', 'type1', 'ns', 'q95_emp', 'q95_theory', 'frac_defined'):
            if field not in rec:
                problems.append(f"study1[{name}] missing {field}")
            elif len(rec[field]) != len(rec['ns']):
                problems.append(f"study1[{name}].{field} length {len(rec[field])} != ns {len(rec['ns'])}")
    for field in ('study1_fitted_b', 'study1_fitted_b_se', 'study2', 'study3', 'study_mv',
                  'exact_null', 'study2_paired', 'power', 'ks_resolution',
                  'bootstrap_comparison', 'box_m_weights', 'heterokurtosis_check'):
        if field not in results:
            problems.append(f"top level missing {field}")
    paired = results.get('study2_paired', {})
    if paired:
        for family in ('bartlett', 'sign'):
            if not paired.get(family):
                problems.append(f"study2_paired[{family}] empty")
            else:
                for sc, rec in paired[family].items():
                    for a, cell in rec.items():
                        for kk in ('size_before', 'size_after', 'diff', 'se'):
                            if kk not in cell:
                                problems.append(f"study2_paired[{family}][{sc}][{a}] missing {kk}")
        if len(paired.get('alphas', [])) < 3:
            problems.append("study2_paired: fewer than three nominal levels")
    if results.get('power'):
        for test_id, rec in results['power']['tests'].items():
            for field in ('power', 'size', 'power_size_adjusted', 'size_adjusted'):
                if field not in rec:
                    problems.append(f"power[{test_id}] missing {field}")
    if results.get('ks_resolution'):
        res = results['ks_resolution']
        for test_id, rec in res['tests'].items():
            if len(rec['ns']) != len(rec['ks_big']) or len(rec['ns']) != len(rec['ks_10k']):
                problems.append(f"ks_resolution[{test_id}] length mismatch")
            if len(rec.get('ns_huge', [])) != len(rec.get('ks_huge', [])):
                problems.append(f"ks_resolution[{test_id}] huge-N length mismatch")
    boot = results.get('bootstrap_comparison')
    if boot:
        if len(boot.get('cells', {})) < 5:
            problems.append("bootstrap_comparison: fewer than five null cells")
        if boot.get('N_outer', 0) < 500 or boot.get('B', 0) < 150:
            problems.append("bootstrap_comparison: N_outer/B below the paper's design")
        for label, cell in boot['cells'].items():
            for m in ('parametric', 'within_centred', 'null_imposed', 'permutation'):
                if m not in cell.get('type1', {}):
                    problems.append(f"bootstrap_comparison[{label}] missing {m}")
    if len(results.get('study_mv', {})) != 12:
        problems.append(f"study_mv has {len(results.get('study_mv', {}))} configurations, expected 12")
    mv = results.get('study_mv', {})
    for key, rec in mv.items():
        for field in ('type1_M', 'type1_Mstar', 'frac_denom_le0', 'frac_defined',
                      'type1_Mstar_unclipped', 'type1_Mstar_floor025'):
            if field not in rec:
                problems.append(f"study_mv[{key}] missing {field}")
    val = results.get('exact_null', {}).get('validation', {})
    if not val:
        problems.append("exact_null.validation missing")
    elif val['max_rel_lr_vs_inv'] > 5e-3:
        problems.append(f"exact_null: saddlepoint vs inversion {val['max_rel_lr_vs_inv']:.2e} > 5e-3")
    if problems:
        for pr in problems:
            print('  SCHEMA FAIL:', pr)
        raise SystemExit(f"{len(problems)} results-schema problems")
    print("results schema: OK")


def run_all_experiments():
    np.random.seed(SEED)
    print("Starting experiments...")
    print(f"Output directory: {_rel(OUT_DIR)}")
    t_total = time.time()

    s1_results, n_star = run_study1(N_mc=10000)
    ks_thr = s1_results.pop('_ks_threshold')
    n_star_ks = s1_results.pop('_n_star_ks')
    fitted_b = s1_results.pop('_fitted_b')
    fitted_b_se = s1_results.pop('_fitted_b_se')
    s2_results, s2_ci, corr_results, corr_ci, s2_bias, s2_abs_err, s2_corr_bias, s2_corr_abs_err, s2_calib_improve = run_study2(N_mc=10000, n_fixed=50)
    s3_results, n_star_mc = run_study3(n_fixed=50, n_repeats=30)
    real_data_results = run_real_data_case_studies()
    mv_results = run_multivariate_simulations(N_mc=10000)
    box_m_weights = run_weight_decomposition()
    heterokurtosis = run_heterokurtosis_check()
    paired_results = run_study2_pairing(N_mc=10000, n_fixed=50)
    power_results = run_power_study(N_mc=5000, n_fixed=50)
    ks_resolution = run_ks_resolution(s1_results, N_big=50000, n_list=(10, 50, 200))
    bootstrap_comparison = run_bootstrap_comparison(N_outer=1000, B=200)

    def _clean_list(vv):
        return [float(x) if not (isinstance(x, float) and np.isnan(x)) else None for x in vv]

    all_results = {
        'study1': {tid: {kk: (_clean_list(vv) if isinstance(vv, list) else vv)
                         for kk, vv in s1_results[tid].items()}
                   for tid in s1_results},
        'study1_nstar_alpha': {k: str(v) for k, v in n_star.items()},
        'study1_nstar_ks': {k: str(v) for k, v in n_star_ks.items()},
        'study1_ks_threshold': ks_thr,
        'study2': {tid: {kk: (float(vv) if not (isinstance(vv, float) and np.isnan(vv)) else None)
                         for kk, vv in s2_results[tid].items()}
                   for tid in s2_results},
        'study1_fitted_b': {k: v if isinstance(v, str) else float(v) for k, v in fitted_b.items()},
        'study1_fitted_b_se': {k: v if isinstance(v, str) else float(v) for k, v in fitted_b_se.items()},
        'study2_ci': s2_ci,
        'study2_corrections': {tid: {kk: (float(vv) if not (isinstance(vv, float) and np.isnan(vv)) else None)
                                     for kk, vv in corr_results[tid].items()} for tid in corr_results},
        'study2_corrections_ci': corr_ci,
        'study2_bias': s2_bias,
        'study2_abs_error': s2_abs_err,
        'study2_corrections_bias': s2_corr_bias,
        'study2_corrections_abs_error': s2_corr_abs_err,
        'study2_calibration_improvement': s2_calib_improve,
        'study3': {k: v for k, v in s3_results.items()},
        'study3_nstar_mc': {k: str(v) for k, v in n_star_mc.items()},
        'study_mv': {tid: {kk: (_clean_list(vv) if isinstance(vv, list) else vv)
                           for kk, vv in mv_results[tid].items()}
                     for tid in mv_results},
        'box_m_weights': box_m_weights,
        'heterokurtosis_check': heterokurtosis,
        'real_data_case_studies': {k: v for k, v in real_data_results.items() if not k.startswith('_')},
        'study2_paired': paired_results,
        'power': power_results,
        'ks_resolution': ks_resolution,
        'bootstrap_comparison': bootstrap_comparison,
    }

    # Part 2: the exact Box's M calibration needs no simulation; merge it in.
    all_results['exact_null'] = run_exact_null()

    results_path = os.path.join(OUT_DIR, 'experiment_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to experiment_results.json")
    check_results(all_results)
    write_manifest(results_path, all_results)
    generate_tables(all_results, os.path.join(OUT_DIR, 'generated_tables.md'))

    # Generate the publication figures from the exact committed results.
    generate_publication_figures(all_results, FIG_DIR)

    elapsed_total = time.time() - t_total
    print(f"\nTotal time: {elapsed_total:.1f}s")


# ============================================================================
# Part 2 -- EXACT Box's M CALIBRATION UNDER A MULTIVARIATE NORMAL PARENT
# ============================================================================
#
# Nothing below is Monte Carlo.  Under H0 with a normal parent the group
# Wisharts satisfy W_j = A^{1/2} B_j A^{1/2} with (B_j) matrix-Dirichlet and
# independent of A, so M_raw is a function of the B_j alone and its cumulant
# generating function is elementary.  Differentiating it gives exact cumulants;
# inverting it along the saddlepoint contour gives exact tail probabilities.
#
# Conventions: k groups, n_j observations in group j, m_j = n_j - 1,
# q0 = sum_j m_j, nu = p(p+1)(k-1)/2, M = (1-c) M_raw with Box's c.
#
_SQ2PI = np.sqrt(2.0 * np.pi)


# --------------------------------------------------------------------------- CGF core
def logGamma_p(a, p):
    """log Gamma_p(a) = (p(p-1)/4) log pi + sum_{i=0}^{p-1} log Gamma(a - i/2).

    Real or complex a (scipy.special.loggamma supports both).
    """
    a = np.asarray(a)
    return 0.25 * p * (p - 1) * np.log(np.pi) + sum(sp.loggamma(a - 0.5 * i) for i in range(p))


def _Lam(u, d, p):
    """log E |W_p(d,I)/d|^u, analytically continued."""
    return p * u * np.log(2.0 / d) + logGamma_p(d / 2.0 + u, p) - logGamma_p(d / 2.0, p)


def _dLam(u, d, p, r):
    """r-th derivative of Lam(.;d,p) w.r.t. its argument."""
    if r == 0:
        return _Lam(u, d, p)
    s = sum(sp.polygamma(r - 1, d / 2.0 + u - 0.5 * i) for i in range(p))
    return (p * np.log(2.0 / d) if r == 1 else 0.0) + s


class CGF:
    """A cumulant generating function on the strip (t_lo, t_hi)."""

    def __init__(self, K, K1=None, K2=None, t_hi=np.inf, t_lo=-np.inf, name='', dK=None):
        self.K = K
        self.t_hi, self.t_lo, self.name = t_hi, t_lo, name
        self._dK = dK or {}
        self.K1 = K1
        self.K2 = K2

    def deriv(self, t, r=1):
        if r in self._dK:
            return self._dK[r](t)
        raise NotImplementedError

    def mean(self):
        return self.K1(0.0)

    def var(self):
        return self.K2(0.0)

    def cumulant(self, r):
        return self.deriv(0.0, r)


def boxm_cgf(m, q0, p, k):
    """Exact CGF of Box's M_raw under MVN, balanced design (m = n-1, q0 = k m)."""
    t_hi = (m - p + 1) / (2.0 * m)          # branch point from the group term

    def K(t):
        return k * _Lam(-m * t, m, p) - _Lam(-q0 * t, q0, p)

    dK = {}
    for r in (1, 2, 3, 4):
        def make(r):
            f = lambda t: k * (-m) ** r * _dLam(-m * t, m, p, r) \
                - (-q0) ** r * _dLam(-q0 * t, q0, p, r)
            return f
        dK[r] = make(r)
    return CGF(K, dK[1], dK[2], t_hi, -np.inf, f'BoxM(m={m},q0={q0},p={p},k={k})', dK)


def boxm_cgf_unbalanced(m_j, p, k=None):
    """Exact CGF of Box's M_raw under MVN for ARBITRARY group sizes.

    m_j is the sequence of group degrees of freedom n_j - 1, q0 = sum_j m_j.  With
    (B_1..B_k) ~ matrix-Dirichlet(m_1/2, ..., m_k/2) independent of A ~ W_p(q0, I),

        K(t) = sum_j Lam(-m_j t; m_j, p) - Lam(-q0 t; q0, p),

    and this is the object used here.  The balanced case k*Lam(-mt;m,p) - Lam(-q0t;q0,p)
    is exactly the special case m_j == m, with no additional constant needed; verified to
    machine precision against boxm_cgf.

    The deterministic normalising term p( sum_j m_j log m_j - q0 log q0 ) needs no separate
    handling: it is already contained in the two Lam terms, whose normalisation contributes
    -p t sum_j m_j log(2/m_j) + p t q0 log(2/q0) = p t (sum_j m_j log m_j - q0 log q0).
    The earlier `include_shift` option added it a second time and has been removed; adding
    it made the mean of M_raw negative, which is impossible.

    This arbitrary-m_j form is what the near-exact / product-of-Betas route does not
    cover: those constructions are derived for equal n_j.
    """
    m_j = np.atleast_1d(np.asarray(m_j, dtype=float))
    k = len(m_j) if k is None else k
    q0 = float(m_j.sum())
    if m_j.min() <= 0:
        raise ValueError('every m_j must be positive')
    t_hi = float(np.min((m_j - p + 1.0) / (2.0 * m_j)))

    def K(t):
        return sum(_Lam(-mi * t, mi, p) for mi in m_j) - _Lam(-q0 * t, q0, p)

    dK = {}
    for r in (1, 2, 3, 4):
        def make(r):
            return lambda t: (sum((-mi) ** r * _dLam(-mi * t, mi, p, r) for mi in m_j)
                              - (-q0) ** r * _dLam(-q0 * t, q0, p, r))
        dK[r] = make(r)
    return CGF(K, dK[1], dK[2], t_hi, -np.inf,
               f'BoxM(m_j={m_j.tolist()},q0={q0:g},p={p},k={k})', dK)


def boxm_cumulants_unbalanced(m_j, p, R=4):
    """Exact cumulants of M_raw for arbitrary group sizes, from the CGF.

    The r-th cumulant is r! times the coefficient of t^r in K(t); the deterministic
    normalising constant contributes to the first cumulant only, and is excluded by
    default so that the values match the Monte Carlo convention for M_raw.
    """
    m_j = np.atleast_1d(np.asarray(m_j, dtype=float))
    q0 = float(m_j.sum())
    out = []
    for r in range(1, R + 1):
        # NB: the CGF built from Lam already contains the deterministic
        # p(sum m_j log m_j - q0 log q0) term (the two Lam terms differ from
        # the Gamma_p ratios by exactly that constant).  Adding a separate
        # "shift" here would count it twice; the former include_shift option
        # did that and has been removed.
        s = (sum((-mi) ** r * _dLam(0.0, mi, p, r) for mi in m_j)
             - (-q0) ** r * _dLam(0.0, q0, p, r))
        out.append(s)
    return out


def chi2_cgf(nu):
    dK = {1: lambda t: nu / (1.0 - 2.0 * t),
          2: lambda t: 2.0 * nu / (1.0 - 2.0 * t) ** 2,
          3: lambda t: 8.0 * nu / (1.0 - 2.0 * t) ** 3,
          4: lambda t: 48.0 * nu / (1.0 - 2.0 * t) ** 4}
    return CGF(lambda t: -0.5 * nu * np.log(1.0 - 2.0 * t), dK[1], dK[2], 0.5, -np.inf,
               f'chi2({nu})', dK)


def exp_cgf():
    dK = {1: lambda t: 1.0 / (1.0 - t),
          2: lambda t: 1.0 / (1.0 - t) ** 2,
          3: lambda t: 2.0 / (1.0 - t) ** 3,
          4: lambda t: 6.0 / (1.0 - t) ** 4}
    return CGF(lambda t: -np.log(1.0 - t), dK[1], dK[2], 1.0, -np.inf, 'Exp(1)', dK)


def boxm_cumulants(m, q0, p, k, R=4):
    """Exact cumulants of M_raw straight from the CGF."""
    return [k * (-m) ** r * _dLam(0.0, m, p, r) - (-q0) ** r * _dLam(0.0, q0, p, r)
            for r in range(1, R + 1)]


# ------------------------------------------------------------------------ saddlepoint
def _solve_saddle(cg, x, xtol=1e-14):
    """Solve K'(t) = x on the whole strip (t may be either sign)."""
    if np.isfinite(cg.t_hi):
        hi = cg.t_hi - max(1e-13, 1e-13 * abs(cg.t_hi))
        if cg.K1(hi) < x:
            raise ValueError('x beyond the domain of K')
    else:
        hi = 1.0
        while cg.K1(hi) < x:
            hi *= 2.0
            if hi > 1e12:
                raise ValueError('no bracket for the saddlepoint')
    lo = -1.0
    if np.isfinite(cg.t_lo):
        lo = cg.t_lo + max(1e-13, 1e-13 * abs(cg.t_lo))
    else:
        while cg.K1(lo) > x:
            lo *= 2.0
            if abs(lo) > 1e12:
                raise ValueError('no lower bracket for the saddlepoint')
    if lo >= 0.0 or hi <= 0.0:
        lo, hi = min(lo, -1e-9), max(hi, 1e-9)
    return brentq(lambda t: cg.K1(t) - x, lo, hi, xtol=xtol, rtol=8.9e-16)


def _reflected(cg):
    return CGF(lambda z: cg.K(-z), lambda t: -cg.K1(-t), lambda t: cg.K2(-t),
               -cg.t_lo, -cg.t_hi, cg.name + ' (reflected)',
               {r: (lambda t, r=r: (-1) ** r * cg.deriv(-t, r)) for r in cg._dK})


def tail_lr(cg, x, correction='upper'):
    """First-order Lugannani-Rice upper tail P(X >= x).

    w = sgn(t)*sqrt(2(t x - K(t))),  u = t sqrt(K''(t)),  K'(t) = x,
    P(X >= x) = 1 - Phi(w) + phi(w) (1/u - 1/w).

    correction='lead' reproduces the (wrong) sign used by boxm/dev/sp_validate.py:
    P(X >= x) = 1 - Phi(w) + phi(w) (1/w - 1/u).  Kept only for the report.
    """
    mu = cg.mean()
    if x < mu:
        return 1.0 - tail_lr(_reflected(cg), -x, correction)
    t = _solve_saddle(cg, x)
    Kt = cg.K(t)
    w = np.sqrt(max(2.0 * (t * x - Kt), 0.0))
    u = t * np.sqrt(max(cg.K2(t), 1e-300))
    if abs(w) < 1e-7:
        # removable singularity at x = mu:  1/u - 1/w -> -K'''(0) / (6 K''(0)^{3/2})
        lim = -cg.deriv(0.0, 3) / (6.0 * cg.K2(0.0) ** 1.5)
        lim = lim if correction == 'upper' else -lim
        return float(0.5 + lim / _SQ2PI)
    corr = (1.0 / u - 1.0 / w) if correction == 'upper' else (1.0 / w - 1.0 / u)
    return float(sp.ndtr(-w) + np.exp(-0.5 * w * w) / _SQ2PI * corr)


def _bromwich(cg, x, c, s_mult, h_div=8.0, d_fac=0.25, nmax=2_000_000):
    """One truncated trapezoid evaluation of the exact Bromwich integral.

    P(X>=x) = e^{K(c)-cx} (1/2pi) int e^{K(c+is)-K(c)-isx} ds/(c+is),  c = t_hat.

    The trapezoid step must resolve the nearest singularity of the integrand in the
    complex s-plane: the pole of 1/z at s = i c (distance |c|) and the branch point of
    K at s = i(t_hi - c) (distance t_hi - c).  The rule error there is ~exp(-2pi d/h),
    so h is tied to d = min(|c|, t_hi - c).  Truncation at |s| = S is separate and is
    handled by the caller (power-law decay, Richardson in S).
    """
    Kc = cg.K(c)
    pref = np.exp(Kc - c * x)
    sd = 1.0 / np.sqrt(max(cg.K2(c), 1e-300))
    d = abs(c)
    if np.isfinite(cg.t_hi):
        d = min(d, cg.t_hi - c)
    h = min(sd / h_div, d_fac * max(d, 1e-300))
    S = s_mult * sd
    n = int(np.ceil(S / h))
    if n > nmax:
        n = nmax
    s = np.arange(-n, n + 1) * h
    z = c + 1j * s
    integ = np.exp(cg.K(z) - Kc - 1j * s * x) / z
    wt = np.ones_like(s)
    wt[0] = wt[-1] = 0.5
    return pref * (integ.real * wt).sum() * h / (2 * np.pi), 2 * n + 1


def tail_invert(cg, x, s_mult=100.0, refine=True):
    """Exact P(X >= x) by numerical inversion of the CGF (Bromwich contour at the saddle).

    Deterministic quadrature, no simulation.  Three truncations S, 2S, 4S are used to
    estimate the power-law truncation error and Richardson-extrapolate it away.
    """
    mu = cg.mean()
    if abs(x - mu) <= 1e-4 * max(1.0, abs(mu)):
        # near the mean the saddle sits on the pole z=0 and the trapezoid step would
        # have to go to zero.  Evaluate at four neighbouring points (where the
        # quadrature is well conditioned) and interpolate cubically.
        d = 2e-2 * max(1.0, abs(mu))
        xs = np.array([mu - 2 * d, mu - d, mu + d, mu + 2 * d])
        vs = np.array([tail_invert(cg, xi, s_mult, refine) for xi in xs])
        return float(np.polyval(np.polyfit(xs, vs, 3), x))
    c = _solve_saddle(cg, x)
    if c > 0:
        v1 = _bromwich(cg, x, c, s_mult)[0]
        if not refine:
            return v1
        v2 = _bromwich(cg, x, c, 2 * s_mult)[0]
        v3 = _bromwich(cg, x, c, 4 * s_mult)[0]
        d1, d2 = v1 - v2, v2 - v3
        if abs(d2) > 1e-15 * max(abs(v3), 1e-300) and 1.0 < d1 / d2 < 1e6:
            q = float(np.log2(d1 / d2))
            out = v3 + d2 / (2.0 ** q - 1.0)
            if abs(out - v3) < 0.5 * abs(v3):
                return out
        return v3
    # lower tail: the pole at z=0 lies to the right of the contour; complement
    return 1.0 - tail_invert(_reflected(cg), -x, s_mult, refine)


def critical_value(tail, alpha, lo=1e-9, hi=1.0, n=200):
    """Solve tail(x) = alpha (tail is decreasing in x)."""
    for _ in range(n):
        if tail(hi) < alpha:
            break
        hi *= 2.0
    else:
        raise RuntimeError('failed to bracket above')
    for _ in range(n):
        if tail(lo) > alpha:
            break
        lo *= 0.5
    else:
        raise RuntimeError('failed to bracket below')
    return brentq(lambda xx: tail(xx) - alpha, lo, hi, xtol=1e-14, rtol=8.9e-16)


# ------------------------------------------------------------------------------ MC
def mc_boxm(n, p, k, N, seed=20261013, chunk_mb=64.0):
    """Exact-null Monte Carlo of M_raw (centred data => W_j ~ W_p(n-1,I))."""
    rng = np.random.default_rng(seed)
    m, q0 = n - 1, k * (n - 1)
    out = np.empty(N)
    step = max(1, int(chunk_mb * 1e6 / (8.0 * m * p * k)))
    done = 0
    while done < N:
        B = min(step, N - done)
        W = np.empty((B, k, p, p))
        for j in range(k):
            Z = rng.standard_normal((B, m, p))
            W[:, j] = np.einsum('bmi,bmj->bij', Z, Z)
        A = W.sum(axis=1)
        Mr = q0 * np.linalg.slogdet(A / q0)[1] \
            - m * sum(np.linalg.slogdet(W[:, j] / m)[1] for j in range(k))
        out[done:done + B] = Mr
        done += B
    return out


# -------------------------------------------------------------------------- selftest
def _step1(verbose=True):
    """Step 1: chi-square and exponential; LR vs exact vs exact inversion."""
    ok = True
    lines = []
    worst_lr = 0.0
    worst_inv = 0.0
    worst_lr_lead = 0.0
    for nu in (1, 3, 6, 12):
        cg = chi2_cgf(nu)
        for alpha in (0.5, 0.1, 0.05, 0.01):
            x = 2.0 * float(sp.gammaincinv(0.5 * nu, 1.0 - alpha))   # exact chi2_ nu quantile
            exact = alpha
            lr = tail_lr(cg, x)
            lr_bad = tail_lr(cg, x, correction='lead')
            inv = tail_invert(cg, x)
            worst_lr = max(worst_lr, abs(lr - exact))
            worst_inv = max(worst_inv, abs(inv - exact))
            worst_lr_lead = max(worst_lr_lead, abs(lr_bad - exact))
            if verbose:
                lines.append(f'  chi2_{nu:<3d} alpha={alpha:<5.2f} x={x:9.4f} '
                             f'LR={lr:.6e} (err {lr-exact:+.1e})  '
                             f'LR_lead={lr_bad:.6e} (err {lr_bad-exact:+.1e})  '
                             f'invert={inv:.6e} (err {inv-exact:+.1e})')
    # critical values
    for nu in (1, 3, 12):
        cg = chi2_cgf(nu)
        for alpha in (0.05, 0.01):
            xt = 2.0 * float(sp.gammaincinv(0.5 * nu, 1.0 - alpha))
            xl = critical_value(lambda xx: tail_lr(cg, xx), alpha, 1e-8, 10.0)
            xi = critical_value(lambda xx: tail_invert(cg, xx), alpha, 1e-8, 10.0)
            if verbose:
                lines.append(f'  crit chi2_{nu:<3d} alpha={alpha:<5.2f} true={xt:9.5f} '
                             f'LR={xl:9.5f} ({(xl/xt-1)*100:+.3f}%)  '
                             f'invert={xi:9.5f} ({(xi/xt-1)*100:+.4f}%)')
    if verbose:
        print('\n'.join(lines))
    print(f'  STEP 1 chi-square: max |LR - exact| = {worst_lr:.2e}   '
          f'max |inversion - exact| = {worst_inv:.2e}   '
          f'max |LR(lead sign) - exact| = {worst_lr_lead:.2e}')
    ok &= worst_inv < 1e-4
    ok &= worst_lr_lead > 10.0 * worst_lr        # the lead's sign really is the wrong one
    # exponential
    cg = exp_cgf()
    print('  Exp(1):  x      exact        LR(+)        rel err      LR_lead      rel err'
          '      inversion    rel err')
    worst_inv_exp = 0.0
    for x in (0.5, 1.0, 2.0, 5.0, 8.0, 12.0):
        exact = np.exp(-x)
        lr = tail_lr(cg, x)
        lrb = tail_lr(cg, x, correction='lead')
        inv = tail_invert(cg, x)
        worst_inv_exp = max(worst_inv_exp, abs(inv / exact - 1))
        print(f'          {x:5.2f}  {exact:.6e}  {lr:.6e}  {(lr/exact-1)*100:+8.3f}%  '
              f'{lrb:.6e}  {(lrb/exact-1)*100:+9.1f}%  {inv:.6e}  {(inv/exact-1)*100:+.2e}%')
    ok &= worst_inv_exp < 1e-2
    print(f'  STEP 1 Exp(1): max relative error of the inversion = {worst_inv_exp:.2e}')
    return bool(ok), worst_lr, worst_inv


def _step2(verbose=True):
    """Step 2: re-derive/verify the CGF: finite differences, cumulants, MC."""
    ok = True
    h = 1e-5
    print('  finite differences of K(t) vs analytic derivatives:')
    for (n, p, k) in [(10, 2, 2), (20, 3, 2)]:
        m, q0 = n - 1, k * (n - 1)
        cg = boxm_cgf(m, q0, p, k)
        e1 = e2 = 0.0
        for t in (-0.1, 0.0, 0.05):
            fd1 = (cg.K(t + h) - cg.K(t - h)) / (2 * h)
            fd2 = (cg.K(t + h) - 2 * cg.K(t) + cg.K(t - h)) / h ** 2
            e1 = max(e1, abs(fd1 - cg.deriv(t, 1)))
            e2 = max(e2, abs(fd2 - cg.deriv(t, 2)) / max(1.0, abs(cg.deriv(t, 2))))
        print(f'    n={n} p={p} k={k}: max rel err  K\'={e1:.2e}  K\'\'={e2:.2e}')
        ok &= (e1 < 1e-6) and (e2 < 1e-4)
    # MC moments
    print('  MC (correctly paired; centred data) vs exact cumulants:')
    print(f'    {"cell":>12s} {"MC mean":>10s} {"+-":>8s} {"K\'(0)":>10s} | '
          f'{"MC var":>10s} {"+-":>8s} {"K\'\'(0)":>10s} | {"nu":>5s} {"2nu":>6s}')
    for (n, p, k) in [(10, 2, 2), (20, 2, 2), (50, 2, 2), (20, 3, 2), (30, 3, 3)]:
        m, q0 = n - 1, k * (n - 1)
        nu = p * (p + 1) * (k - 1) / 2.0
        M = mc_boxm(n, p, k, 200000)
        cg = boxm_cgf(m, q0, p, k)
        mu, var = M.mean(), M.var(ddof=1)
        se_m = M.std(ddof=1) / np.sqrt(M.size)
        se_v = var * np.sqrt(2.0 / M.size)
        print(f'    n{n}p{p}k{k}'.rjust(14) + f'{mu:11.4f} {se_m:8.4f} {cg.mean():10.4f} | '
              f'{var:10.4f} {se_v:8.4f} {cg.var():10.4f} | {nu:5.1f} {2*nu:6.1f}')
        ok &= abs(mu - cg.mean()) < 5 * se_m + 1e-6
        ok &= abs(var - cg.var()) < 5 * se_v + 1e-6
    # MGF spot check (the strongest test)
    print('  MGF test  E[e^{t M_raw}] vs exp(K(t))  (t = 0.05):')
    from scipy import stats as _st
    for (n, p, k) in [(10, 2, 2), (30, 3, 3)]:
        m, q0 = n - 1, k * (n - 1)
        M = mc_boxm(n, p, k, 200000)
        cg = boxm_cgf(m, q0, p, k)
        mgf = np.exp(0.05 * M).mean()
        mgf_se = np.exp(0.05 * M).std(ddof=1) / np.sqrt(M.size)
        th = np.exp(cg.K(0.05))
        print(f'    n={n} p={p} k={k}: MC={mgf:.5f} (+-{mgf_se:.5f})  exp(K(0.05))={th:.5f}  '
              f'z={(mgf-th)/mgf_se:+.2f}')
        # the independence-assuming (wrong) CGF for contrast
        cg_w = CGF(lambda t: _Lam(q0 * t, q0, p) + k * _Lam(-m * t, m, p),
                   None, None, np.inf, -np.inf, 'wrong')
        print(f'      wrong CGF exp(K_w(0.05)) = {np.exp(cg_w.K(0.05)):.5f}')
        ok &= abs(mgf - th) < 5 * mgf_se + 1e-9
    return bool(ok)


def paper_table(alphas=(0.05, 0.01)):
    """Deterministic critical values and exact sizes for the paper (no simulation).

    M_raw = q0 log|S_p| - sum_j m log|S_j|,  M = (1-c) M_raw,  c as in the paper (3.1).
    """
    print('MVN null.  crit_*: critical value of M_raw.  size_*: exact size from the CGF.')
    print(f'{"p":>2s} {"n":>4s} {"k":>2s} {"nu":>4s} | ' +
          ' | '.join(f'{("a=%g"%a):>38s}' for a in alphas))
    hdr = 'crit_LR  crit_exact  crit_chi2  size_chi2raw  size_chi2corr'
    print(f'{"":17s} | ' + ' | '.join(f'{hdr:>38s}' for _ in alphas))
    for p in (2, 3, 5):
        for n in (20, 50, 100):
            for k in (2, 3):
                m, q0 = n - 1, k * (n - 1)
                nu = p * (p + 1) * (k - 1) / 2.0
                c = ((2 * p * p + 3 * p - 1) / (6 * (p + 1) * (k - 1))) * (k / m - 1.0 / q0)
                cg = boxm_cgf(m, q0, p, k)
                out = []
                for a in alphas:
                    crit_lr = critical_value(lambda x: tail_lr(cg, x), a, 1e-6, 20.0)
                    crit_inv = critical_value(lambda x: tail_invert(cg, x), a, 1e-6, 20.0)
                    chi = 2.0 * float(sp.gammaincinv(0.5 * nu, 1.0 - a))
                    out.append(f'{crit_lr:8.4f}  {crit_inv:10.4f}  {chi:9.4f}  '
                               f'{tail_invert(cg, chi):11.4f}  '
                               f'{tail_invert(cg, chi / (1 - c)):12.4f}')
                print(f'{p:2d} {n:4d} {k:2d} {nu:4.0f} | ' + ' | '.join(f'{o:>38s}' for o in out))


# ---------------------------------------------------------------- paper tables
def _chi2_quantile(nu, alpha):
    """Quantile of chi-square(nu) at upper tail alpha."""
    return 2.0 * float(sp.gammaincinv(0.5 * nu, 1.0 - alpha))


def _box_c(p, n, k):
    """Box's Bartlett-type factor c for a balanced design."""
    m = n - 1
    q0 = k * m
    return ((2.0 * p * p + 3.0 * p - 1.0) / (6.0 * (p + 1.0) * (k - 1.0))) * (k / m - 1.0 / q0)


EXACT_GRID = [(p, n, k) for p in (2, 3, 5) for n in (20, 50, 100) for k in (2, 3)]
PAPER_CUMMULANTS = [(10, 2, 2), (20, 2, 2), (50, 2, 2), (100, 2, 2), (30, 3, 3)]
PAPER_CRIT = [(2, 20, 2), (2, 50, 2), (2, 100, 2), (3, 20, 2), (3, 50, 2),
              (3, 100, 3), (5, 20, 2), (5, 20, 3), (5, 50, 3)]
PAPER_SIZE = [(2, 20, 2), (2, 20, 3), (2, 50, 2), (2, 100, 2), (3, 20, 2),
              (3, 20, 3), (3, 100, 3), (5, 20, 2), (5, 20, 3)]


def run_exact_null(reference_path=None, alphas=(0.05, 0.01), verbose=True):
    """Exact cumulants, critical values and sizes of Box's M under MVN.

    Returns a dict with the numbers behind Table 4 (exact cumulants),
    Table 5 (critical values) and Table 6 (exact size), plus the validation
    metrics quoted in Section 5.5.
    """
    if reference_path is None:
        reference_path = EXACT_REF
    ref = None
    if reference_path and os.path.exists(reference_path):
        with open(reference_path) as fh:
            ref = json.load(fh)
    if verbose:
        print("=" * 78)
        print("PART 2 -- exact Box's M calibration (multivariate normal parent)")
        print(f"Monte Carlo reference: {_rel(reference_path) if ref else 'NOT FOUND (MC columns skipped)'}")
        print("=" * 78)

    # --- exact cumulants (Table 4)
    cumulants = []
    for (n, k, p) in PAPER_CUMMULANTS:
        m = n - 1
        q0 = k * m
        nu = p * (p + 1) * (k - 1) / 2.0
        c1, c2 = boxm_cumulants(m, q0, p, k, R=2)
        cumulants.append(dict(n=n, k=k, p=p, nu=nu, mean=float(c1), var=float(c2),
                              var_over_2nu=float(c2 / (2.0 * nu))))
    if verbose:
        print("\nTable 4  exact cumulants of M_raw")
        print(f'{"n":>4s} {"k":>2s} {"p":>2s} {"E[M]":>9s} {"nu":>5s} {"Var":>9s} {"2nu":>5s} {"ratio":>6s}')
        for r in cumulants:
            print(f'{r["n"]:4d} {r["k"]:2d} {r["p"]:2d} {r["mean"]:9.4f} {r["nu"]:5.0f} '
                  f'{r["var"]:9.4f} {2*r["nu"]:5.0f} {r["var_over_2nu"]:6.3f}')

    # --- critical values (Table 5) and validation over the full grid
    cells = {}
    lr_vs_inv, lr_vs_mc, inv_vs_mc, chi_vs_mc = [], [], [], []
    for (p, n, k) in EXACT_GRID:
        m, q0 = n - 1, k * (n - 1)
        nu = p * (p + 1) * (k - 1) / 2.0
        cg = boxm_cgf(m, q0, p, k)
        rec = dict(p=p, n=n, k=k, nu=nu, crit={}, size={})
        for a in alphas:
            key = f"{p}|{n}|{k}"
            crit_lr = critical_value(lambda x, cg=cg: tail_lr(cg, x), a, 1e-6, 20.0)
            crit_inv = critical_value(lambda x, cg=cg: tail_invert(cg, x), a, 1e-6, 20.0)
            chi2q = _chi2_quantile(nu, a)
            rec["crit"][f"{a:g}"] = dict(lr=float(crit_lr), inversion=float(crit_inv),
                                         chi2=float(chi2q))
            lr_vs_inv.append(abs(crit_lr - crit_inv) / crit_inv)
            if ref and key in ref.get("cells", {}):
                mc = ref["cells"][key][f"q{int(a*1000):03d}"]
                rec["crit"][f"{a:g}"]["mc"] = float(mc)
                lr_vs_mc.append(abs(crit_lr - mc) / mc)
                inv_vs_mc.append(abs(crit_inv - mc) / mc)
                chi_vs_mc.append(abs(chi2q - mc) / mc)
        c = _box_c(p, n, k)
        for a in (0.05,):
            chi2q = _chi2_quantile(nu, a)
            rec["size"][f"{a:g}"] = dict(
                chi2_raw=float(tail_invert(cg, chi2q)),
                chi2_bartlett=float(tail_invert(cg, chi2q / (1.0 - c))),
                saddlepoint=float(tail_invert(cg, rec["crit"][f"{a:g}"]["lr"])),
            )
        cells[(p, n, k)] = rec

    def _mx(v):
        return float(max(v)) if v else None

    validation = dict(
        n_pairs_lr_vs_inv=len(lr_vs_inv),
        max_rel_lr_vs_inv=_mx(lr_vs_inv),
        median_rel_lr_vs_inv=float(sorted(lr_vs_inv)[len(lr_vs_inv) // 2]) if lr_vs_inv else None,
        n_pairs_vs_mc=len(lr_vs_mc),
        max_rel_lr_vs_mc=_mx(lr_vs_mc),
        max_rel_inv_vs_mc=_mx(inv_vs_mc),
        max_rel_chi2_vs_mc=_mx(chi_vs_mc),
    )

    if verbose:
        print("\nTable 5  95% critical values of M_raw")
        print(f'{"p":>2s} {"n":>4s} {"k":>2s} {"nu":>4s} {"saddle":>9s} {"exact":>9s} '
              f'{"MC":>9s} {"chi2":>9s} {"chi2 err":>9s}')
        for (p, n, k) in PAPER_CRIT:
            r = cells[(p, n, k)]
            c5 = r["crit"]["0.05"]
            mc = c5.get("mc")
            mc_s = f'{mc:9.4f}' if mc else f'{"-":>9s}'
            err = f'{(c5["chi2"]-mc)/mc*100:+8.2f}%' if mc else f'{"-":>9s}'
            print(f'{p:2d} {n:4d} {k:2d} {r["nu"]:4.0f} {c5["lr"]:9.4f} '
                  f'{c5["inversion"]:9.4f} {mc_s} {c5["chi2"]:9.4f} {err}')
        print("\nTable 6  exact size at alpha = 0.05")
        print(f'{"p":>2s} {"n":>4s} {"k":>2s} {"nu":>4s} {"chi2 raw":>9s} '
              f'{"chi2 Bartlett":>13s} {"saddlepoint":>12s}')
        for (p, n, k) in PAPER_SIZE:
            r = cells[(p, n, k)]
            s = r["size"]["0.05"]
            print(f'{p:2d} {n:4d} {k:2d} {r["nu"]:4.0f} {s["chi2_raw"]:9.4f} '
                  f'{s["chi2_bartlett"]:13.4f} {s["saddlepoint"]:12.4f}')
        print("\nvalidation")
        print(f'  saddlepoint vs exact inversion : max {100*validation["max_rel_lr_vs_inv"]:.3f}% '
              f'(median {100*validation["median_rel_lr_vs_inv"]:.4f}%), '
              f'{validation["n_pairs_lr_vs_inv"]} pairs')
        if validation["n_pairs_vs_mc"]:
            print(f'  saddlepoint vs Monte Carlo      : max {100*validation["max_rel_lr_vs_mc"]:.2f}%')
            print(f'  exact inversion vs Monte Carlo  : max {100*validation["max_rel_inv_vs_mc"]:.2f}%')
            print(f'  chi-square reference vs MC      : max {100*validation["max_rel_chi2_vs_mc"]:.2f}%')
        print("=" * 78)

    return dict(
        alphas=list(alphas),
        reference=_rel(reference_path) if ref else None,
        cumulants=cumulants,
        cells=[dict(p=p, n=n, k=k, **{k2: v for k2, v in cells[(p, n, k)].items()
                                      if k2 not in ("p", "n", "k")})
               for (p, n, k) in EXACT_GRID],
        paper_crit=list(PAPER_CRIT),
        paper_size=list(PAPER_SIZE),
        validation=validation,
    )



def build_exact_reference(path=None, N=400_000, alphas=(0.05, 0.01), verbose=True):
    """Rebuild the Monte Carlo reference used to validate the exact critical values.

    Independent of every method under test: it simulates M_raw directly from
    centred normal data (see mc_boxm) and stores the upper quantiles.  Runtime
    is a minute or two; the committed copy in results/ is what --exact-null uses
    by default.
    """
    if path is None:
        path = EXACT_REF
    cells = {}
    for (p, n, k) in EXACT_GRID:
        t0 = time.time()
        x = mc_boxm(n, p, k, N)
        rec = {f"q{int(a * 1000):03d}": float(np.quantile(x, 1 - a)) for a in alphas}
        rec["mean"] = float(x.mean())
        rec["var"] = float(x.var(ddof=1))
        rec["N"] = N
        rec["seconds"] = round(time.time() - t0, 1)
        cells[f"{p}|{n}|{k}"] = rec
        if verbose:
            print(f"  p={p} n={n} k={k}: {rec['seconds']:.1f}s")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"N": N, "alphas": list(alphas), "cells": cells}, fh, indent=1)
    print(f"wrote {_rel(path)}")
    return path


# ============================================================================
# MAIN
# ============================================================================
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Edgeworth null-calibration study: simulations, "
                                             "exact Box's M calibration, and figures.")
    ap.add_argument('--figures-only', action='store_true',
                    help='regenerate the figures from the committed results file, no simulation')
    ap.add_argument('--exact-null', action='store_true',
                    help="run only the exact Box's M calibration (deterministic) and merge it "
                         'into the results file')
    ap.add_argument('--smoke', action='store_true',
                    help='tiny-replication run of both halves, for a syntax/runtime check')
    ap.add_argument('--results-file', default=os.path.join(OUT_DIR, 'experiment_results.json'),
                    help='results JSON used with --figures-only / --exact-null')
    ap.add_argument('--fig-dir', default=FIG_DIR, help='figure output directory')
    ap.add_argument('--reference', default=EXACT_REF,
                    help='Monte Carlo reference JSON for the exact calibration')
    ap.add_argument('--extra-studies', action='store_true',
                    help='recompute Study 1 (CI convention) and the extra studies 1b/2b/4b, '
                         'merge them into the committed results, and refresh tables/manifest')
    ap.add_argument('--rebuild-reference', action='store_true',
                    help='rebuild the Monte Carlo reference used to validate the exact route')
    args = ap.parse_args()

    if args.figures_only:
        with open(args.results_file) as fh:
            committed = json.load(fh)
        generate_publication_figures(committed, args.fig_dir)
        return

    if args.extra_studies:
        with open(args.results_file) as fh:
            results = json.load(fh)

        def _fresh_study1():
            np.random.seed(SEED)
            r, nstar = run_study1(N_mc=10000)
            thr = r.pop('_ks_threshold')
            nstar_ks = r.pop('_n_star_ks')
            fb = r.pop('_fitted_b')
            fse = r.pop('_fitted_b_se')
            return r, nstar, nstar_ks, fb, fse, thr

        def _clean(values):
            return [float(x) if not (isinstance(x, float) and np.isnan(x)) else None
                    for x in values]

        s1, n_star, n_star_ks, fitted_b, fitted_b_se, ks_thr = _fresh_study1()
        results['study1'] = {tid: {kk: (_clean(vv) if isinstance(vv, list) else vv)
                                   for kk, vv in s1[tid].items()} for tid in s1}
        results['study1_nstar_alpha'] = n_star
        results['study1_nstar_ks'] = n_star_ks
        results['study1_ks_threshold'] = ks_thr
        results['study1_fitted_b'] = fitted_b
        results['study1_fitted_b_se'] = fitted_b_se
        results['study2_paired'] = run_study2_pairing(N_mc=10000, n_fixed=50)
        results['power'] = run_power_study(N_mc=5000, n_fixed=50)
        results['ks_resolution'] = run_ks_resolution(s1, N_big=50000, n_list=(10, 50, 200))
        results['box_m_weights'] = run_weight_decomposition()
        results['heterokurtosis_check'] = run_heterokurtosis_check()
        results['bootstrap_comparison'] = run_bootstrap_comparison(N_outer=1000, B=200)
        with open(args.results_file, 'w') as fh:
            json.dump(results, fh, indent=2, default=str)
        check_results(results)
        write_manifest(args.results_file, results)
        generate_tables(results, os.path.join(OUT_DIR, 'generated_tables.md'))
        generate_publication_figures(results, args.fig_dir)
        print("extra studies merged")
        return

    if args.exact_null:
        results = {}
        if os.path.exists(args.results_file):
            with open(args.results_file) as fh:
                results = json.load(fh)
        results['exact_null'] = run_exact_null(args.reference)
        with open(args.results_file, 'w') as fh:
            json.dump(results, fh, indent=2, default=str)
        check_results(results)
        write_manifest(args.results_file, results)
        print(f"merged the exact-null block into {_rel(args.results_file)}")
        return

    if args.rebuild_reference:
        build_exact_reference(args.reference)
        return

    if args.smoke:
        print("smoke run: 200 replications per cell (results are NOT written)")
        run_study1(N_mc=200)
        run_multivariate_simulations(N_mc=200)
        ex = run_exact_null(args.reference, alphas=(0.05,))
        assert ex['validation']['max_rel_lr_vs_inv'] < 1e-2
        print("smoke run OK")
        return

    run_all_experiments()


if __name__ == '__main__':
    main()
