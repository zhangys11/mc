"""
Finite-Sample Adequacy and Robustness of Null Distributions
A Systematic Monte Carlo Study Across Ten Hypothesis Tests

Experiment script for generating all results, figures, and tables.
"""

import numpy as np
from scipy import stats
from scipy.special import comb
import warnings
import json
import os
import time

warnings.filterwarnings('ignore')
np.random.seed(2024)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = OUT_DIR

try:
    import matplotlib
    if __name__ == '__main__':
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
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

def mc_fligner_killeen(k, n, N, dist='normal', unbalanced=False):
    FKs = np.empty(N)
    if unbalanced:
        ns = [n, 2*n, 3*n][:k]
        while len(ns) < k:
            ns.append(n)
    else:
        ns = [n] * k
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

def mc_sign_test(n, N, dist='expon'):
    if dist == 'expon':
        X = np.random.exponential(1, (N, n))
        median_theory = np.log(2)
    elif dist == 'normal':
        X = np.random.normal(0, 1, (N, n))
        median_theory = 0.0
    elif dist == 'uniform':
        X = np.random.uniform(0, 1, (N, n))
        median_theory = 0.5
    elif dist == 'discrete':
        X = np.random.randint(1, 7, (N, n)).astype(float)
        median_theory = 3.5
    else:
        X = np.random.exponential(1, (N, n))
        median_theory = np.log(2)
    N_plus = (X > median_theory).sum(axis=1)
    return N_plus

def mc_cochran_q(k, n, N, p=0.5, unbalanced=False):
    Ts = np.empty(N)
    X = np.random.binomial(1, p, (N, n, k))
    col_sums = X.sum(axis=1)
    row_sums = X.sum(axis=2)
    num = (k-1) * (k * (col_sums**2).sum(axis=1) - col_sums.sum(axis=1)**2)
    den = k * row_sums.sum(axis=1) - (row_sums**2).sum(axis=1)
    den[den == 0] = 1e-15
    return num / den

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
        results[test_id] = {'ks': [], 'kl': [], 'type1': [], 'ns': []}
        print(f"\n  {info['label']}...", end=' ', flush=True)
        t0 = time.time()

        for n in sample_sizes:
            try:
                if test_id == 'T1_t_test':
                    sv = mc_t_test(n, N_mc)
                    ks_s, ks_p = compute_ks(sv, 't', n-1)
                    t1e = compute_type1_error(sv, 't', (n-1,), two_sided=True)
                    kl_val = compute_kl(sv, 't', (n-1,))

                elif test_id == 'T2_anova':
                    sv, df1, df2 = mc_anova(k_default, n, N_mc)
                    ks_s, ks_p = compute_ks(sv, 'f', df1, df2)
                    t1e = compute_type1_error(sv, 'f', (df1, df2))
                    kl_val = compute_kl(sv, 'f', (df1, df2))

                elif test_id == 'T3_kruskal_wallis':
                    N_kw = min(N_mc, 5000) if n >= 100 else N_mc
                    sv = mc_kruskal_wallis(k_default, n, N_kw)
                    ks_s, ks_p = compute_ks(sv, 'chi2', k_default-1)
                    t1e = compute_type1_error(sv, 'chi2', (k_default-1,))
                    kl_val = compute_kl(sv, 'chi2', (k_default-1,))

                elif test_id == 'T4_chisq_gof':
                    k_gof = 6
                    sample_n = max(n * 5, 30)
                    sv = mc_chisq_gof(k_gof, sample_n, N_mc)
                    ks_s, ks_p = compute_ks(sv, 'chi2', k_gof-1)
                    t1e = compute_type1_error(sv, 'chi2', (k_gof-1,))
                    kl_val = compute_kl(sv, 'chi2', (k_gof-1,))

                elif test_id == 'T5_bartlett':
                    sv = mc_bartlett(k_default, n, N_mc)
                    ks_s, ks_p = compute_ks(sv, 'chi2', k_default-1)
                    t1e = compute_type1_error(sv, 'chi2', (k_default-1,))
                    kl_val = compute_kl(sv, 'chi2', (k_default-1,))

                elif test_id == 'T6_fligner_killeen':
                    N_fk = min(N_mc, 3000)
                    sv = mc_fligner_killeen(k_default, n, N_fk)
                    ks_s, ks_p = compute_ks(sv, 'chi2', k_default-1)
                    t1e = compute_type1_error(sv, 'chi2', (k_default-1,))
                    kl_val = compute_kl(sv, 'chi2', (k_default-1,))

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

                elif test_id == 'T8_cochran_q':
                    sv = mc_cochran_q(k_default, n, N_mc)
                    ks_s, ks_p = compute_ks(sv, 'chi2', k_default-1)
                    t1e = compute_type1_error(sv, 'chi2', (k_default-1,))
                    kl_val = compute_kl(sv, 'chi2', (k_default-1,))

                elif test_id == 'T9_median_test':
                    N_mt = min(N_mc, 5000) if n >= 100 else N_mc
                    sv = mc_median_test(k_default, n, N_mt)
                    ks_s, ks_p = compute_ks(sv, 'chi2', k_default-1)
                    t1e = compute_type1_error(sv, 'chi2', (k_default-1,))
                    kl_val = compute_kl(sv, 'chi2', (k_default-1,))

                elif test_id == 'T10_hotelling':
                    p_dim = 2
                    if n <= p_dim + 1:
                        ks_s, t1e, kl_val = np.nan, np.nan, np.nan
                    else:
                        N_ht = min(N_mc, 5000)
                        sv, dfn, dfd = mc_hotelling_t2(p_dim, n, N_ht)
                        ks_s, ks_p = compute_ks(sv, 'f', dfn, dfd)
                        t1e = compute_type1_error(sv, 'f', (dfn, dfd))
                        kl_val = compute_kl(sv, 'f', (dfn, dfd))

                results[test_id]['ks'].append(float(ks_s))
                results[test_id]['kl'].append(float(kl_val) if not np.isnan(kl_val) else np.nan)
                results[test_id]['type1'].append(float(t1e))
                results[test_id]['ns'].append(n)
            except Exception as e:
                print(f"[ERR n={n}: {e}]", end=' ')
                results[test_id]['ks'].append(np.nan)
                results[test_id]['kl'].append(np.nan)
                results[test_id]['type1'].append(np.nan)
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

    n_star = {}
    for test_id, info in tests.items():
        t1es = results[test_id]['type1']
        ns_list = results[test_id]['ns']
        found = None
        for j, (nn, t1e) in enumerate(zip(ns_list, t1es)):
            if not np.isnan(t1e) and abs(t1e - 0.05) < 0.01:
                found = nn
                break
        n_star[test_id] = found if found else '>500'
        print(f"  {info['label']}: n* = {n_star[test_id]}")

    if HAS_MPL:
        fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
        for test_id, info in tests.items():
            ns_arr = np.array(results[test_id]['ns'])
            ks_arr = np.array(results[test_id]['ks'])
            kl_arr = np.array(results[test_id]['kl'])
            t1_arr = np.array(results[test_id]['type1'])
            mask = ~np.isnan(ks_arr)
            axes[0].plot(ns_arr[mask], ks_arr[mask], 'o-', label=info['label'], markersize=3)
            mask_kl = ~np.isnan(kl_arr)
            axes[1].plot(ns_arr[mask_kl], kl_arr[mask_kl], 'o-', label=info['label'], markersize=3)
            mask2 = ~np.isnan(t1_arr)
            axes[2].plot(ns_arr[mask2], t1_arr[mask2], 'o-', label=info['label'], markersize=3)
        axes[0].set_ylabel('KS Distance')
        axes[0].set_title('Study 1: Convergence of Empirical Null Distribution')
        axes[0].legend(fontsize=7, ncol=3)
        axes[0].set_xscale('log')
        axes[0].grid(True, alpha=0.3)
        axes[1].set_ylabel('KL Divergence')
        axes[1].legend(fontsize=7, ncol=3)
        axes[1].set_xscale('log')
        axes[1].grid(True, alpha=0.3)
        axes[2].axhline(y=0.05, color='black', linestyle='--', linewidth=1, label='Nominal α=0.05')
        axes[2].axhspan(0.04, 0.06, alpha=0.1, color='green', label='±0.01 band')
        axes[2].set_ylabel('Empirical Type I Error')
        axes[2].set_xlabel('Sample size n')
        axes[2].legend(fontsize=7, ncol=3)
        axes[2].set_xscale('log')
        axes[2].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'fig_study1_convergence.png'))
        plt.show()
        plt.close()
        print(f"  Saved fig_study1_convergence.png")

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
                mc_kruskal_wallis(k, n_fixed, min(N, 5000), dist=sc.get('dist', 'uniform'),
                                  hetero=sc.get('hetero', False), unbalanced=sc.get('unbalanced', False)),
                'chi2', (k-1,), False
            ),
        },
        'T4_chisq_gof': {
            'label': 'Chi-sq GOF',
            'applicable': ['baseline'],
            'run': lambda sc, N: (
                mc_chisq_gof(6, 250, N),
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
                mc_fligner_killeen(k, n_fixed, min(N, 3000), dist=sc.get('dist', 'normal'),
                                    unbalanced=sc.get('unbalanced', False)),
                'chi2', (k-1,), False
            ),
        },
        'T7_sign_test': {
            'label': 'Sign test',
            'applicable': ['baseline', 'skewed', 'heavy_tail'],
            'run': lambda sc, N: (
                mc_sign_test(n_fixed, N, dist={'expon': 'expon', 'tdist': 'normal'}.get(sc.get('dist',''), 'expon')),
                'binom', (n_fixed, 0.5), False
            ),
        },
        'T8_cochran_q': {
            'label': 'Cochran Q',
            'applicable': ['baseline', 'unbalanced'],
            'run': lambda sc, N: (
                mc_cochran_q(k, n_fixed, N),
                'chi2', (k-1,), False
            ),
        },
        'T9_median_test': {
            'label': 'Median test',
            'applicable': ['baseline', 'skewed', 'heavy_tail', 'hetero_var', 'unbalanced'],
            'run': lambda sc, N: (
                mc_median_test(k, n_fixed, min(N, 5000), dist=sc.get('dist', 'uniform'),
                               hetero=sc.get('hetero', False), unbalanced=sc.get('unbalanced', False)),
                'chi2', (k-1,), False
            ),
        },
        'T10_hotelling': {
            'label': 'Hotelling T2',
            'applicable': ['baseline', 'skewed', 'heavy_tail'],
            'run': lambda sc, N: (
                mc_hotelling_t2(2, n_fixed, min(N, 5000), dist=sc.get('dist', 'normal'))[0],
                'f', (2, n_fixed-2), False
            ),
        },
    }

    for test_id, cfg in test_configs.items():
        results[test_id] = {}
        print(f"\n  {cfg['label']}...", end=' ', flush=True)
        t0 = time.time()
        for sc_name, sc_params in scenarios:
            if sc_name not in cfg['applicable']:
                results[test_id][sc_name] = np.nan
                continue
            try:
                sv, dist_name, df_args, two_sided = cfg['run'](sc_params, N_mc)
                if dist_name == 'binom':
                    t1e = compute_type1_error(sv, 'binom', df_args)
                else:
                    t1e = compute_type1_error(sv, dist_name, df_args, two_sided=two_sided)
                results[test_id][sc_name] = float(t1e)
            except Exception as e:
                results[test_id][sc_name] = np.nan
                print(f"[ERR {sc_name}: {e}]", end=' ')
        elapsed = time.time() - t0
        print(f"({elapsed:.1f}s)")

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

    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(12, 6))
        sc_names = [s[0] for s in scenarios]
        sc_labels = {'baseline': 'Baseline', 'skewed': 'Skewed', 'heavy_tail': 'Heavy-tailed',
                     'hetero_var': 'Var. heterogeneity', 'unbalanced': 'Unbalanced'}
        x = np.arange(len(sc_names))
        test_ids_list = [tid for tid in test_configs]
        n_tests = len(test_ids_list)
        width = 0.8 / n_tests
        for i, tid in enumerate(test_ids_list):
            vals = []
            for sc_name in sc_names:
                v = results[tid].get(sc_name, np.nan)
                vals.append(v if not np.isnan(v) else 0)
            mask = np.array([not np.isnan(results[tid].get(sc, np.nan)) for sc in sc_names])
            vals_arr = np.array(vals)
            ax.bar(x[mask] + i * width, vals_arr[mask], width * 0.9, label=test_configs[tid]['label'])
        ax.axhline(y=0.05, color='red', linestyle='--', linewidth=1, label='α=0.05')
        ax.set_ylabel('Empirical Type I Error Rate')
        ax.set_title('Study 2: Type I Error Under Assumption Violations (n=50, α=0.05)')
        ax.set_xticks(x + width * n_tests / 2)
        ax.set_xticklabels([sc_labels.get(s, s) for s in sc_names])
        ax.legend(fontsize=7, ncol=4, loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'fig_study2_robustness.png'))
        plt.show()
        plt.close()
        print(f"  Saved fig_study2_robustness.png")

    return results

# ============================================================
# SECTION 5: Study 3 - MC precision
# ============================================================

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
        'T6_fligner_killeen': ('Fligner-Killeen', lambda N: mc_fligner_killeen(k, n_fixed, N), 'chi2', (k-1,), False, 10000),
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

    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(10, 5))
        for test_id, (label, _, _, _, _, _) in test_funcs.items():
            ax.plot(results[test_id]['N_values'], results[test_id]['cv_t1e'],
                    'o-', label=label, markersize=4)
        ax.axhline(y=0.10, color='red', linestyle='--', linewidth=1, label='CV = 0.10 threshold')
        ax.set_xscale('log')
        ax.set_xlabel('MC Replications (N)')
        ax.set_ylabel('CV of Empirical Type I Error')
        ax.set_title('Study 3: MC Precision vs Replication Count (n=50)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'fig_study3_precision.png'))
        plt.show()
        plt.close()
        print(f"  Saved fig_study3_precision.png")

    return results, n_star_mc

# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("Starting experiments...")
    print(f"Output directory: {OUT_DIR}")
    t_total = time.time()

    s1_results, n_star = run_study1(N_mc=10000)
    s2_results = run_study2(N_mc=10000, n_fixed=50)
    s3_results, n_star_mc = run_study3(n_fixed=50, n_repeats=30)

    all_results = {
        'study1': {k: {kk: [float(x) if not np.isnan(x) else None for x in vv]
                        if isinstance(vv, list) else vv
                        for kk, vv in v.items()}
                   for k, v in s1_results.items()},
        'study1_nstar': {k: str(v) for k, v in n_star.items()},
        'study2': {k: {kk: float(vv) if not np.isnan(vv) else None
                        for kk, vv in v.items()}
                   for k, v in s2_results.items()},
        'study3': {k: v for k, v in s3_results.items()},
        'study3_nstar_mc': {k: str(v) for k, v in n_star_mc.items()},
    }

    with open(os.path.join(OUT_DIR, 'experiment_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to experiment_results.json")

    elapsed_total = time.time() - t_total
    print(f"\nTotal time: {elapsed_total:.1f}s")
