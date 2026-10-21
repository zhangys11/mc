#!/usr/bin/env python3
"""Monte Carlo verification of the null distribution of Box's M statistic.

This script accompanies ``mc.samplings.Box_M_Test`` and answers two questions:

1. **Is the normal-theory reference adequate?**  Under multivariate normality
   the Bartlett-corrected statistic ``(1 - c) M`` should be distributed as
   ``chi2(nu)`` with ``nu = (k - 1) p (p + 1) / 2``.  We simulate the null
   distribution over a grid of ``(k, p, n)``, measure the Kolmogorov-Smirnov
   distance to the reference and the empirical type I error at the nominal
   levels, and compare with the uncorrected statistic.

2. **What happens under non-normality when H0 still holds?**  All simulated
   groups share the same covariance matrix, so H0 is true by construction, but
   the marginal population is ``t``, exponential, chi-square or lognormal.
   The sample covariances then fluctuate with the fourth moments, and the
   nominal ``chi2(nu)`` reference is expected to break down.  The asymptotic
   null law is a weighted sum of chi-squares,

       T = (1 - c) M  ->  0.5 * sum_i lambda_i * chi2(k - 1),

   where the ``lambda_i`` are the eigenvalues of the fourth-moment matrix
   ``Omega`` with ``Omega[(a,b),(c,d)] = E[x_a x_b x_c x_d] - delta_ab delta_cd``
   (independent, standardized margins).  We verify this limit by simulation.
   For the normal case ``Omega`` has ``p(p + 1)/2`` eigenvalues equal to 2 and
   the limit collapses to ``chi2(nu)`` -- a useful analytic consistency check.

The script writes ``results/box_m_null_results.json`` and three figures under
``figures/``.  Run it from the repository root::

    python3 box_m/study_box_m_null.py            # full run, ~3-4 minutes
    python3 box_m/study_box_m_null.py --quick    # reduced run, ~20 seconds
    python3 box_m/study_box_m_null.py --help
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parents[1]      # "Finite-Sample Adequacy ..." package root
REPO_ROOT = HERE.parents[2]         # mc-tk repository root (holds src/mc)
sys.path.insert(0, str(REPO_ROOT / "src"))

from mc.samplings import Box_M_Test  # noqa: E402

SEED = 2024
N_MC = 10_000
# Defaults captured before ``main`` mutates the module-level values, so that a
# second ``main`` call in the same interpreter starts from the documented
# defaults instead of inheriting a previous ``--quick``/``--seed`` setting.
DEFAULT_SEED = SEED
DEFAULT_N_MC = N_MC
ALPHAS = (0.01, 0.05)
KS_THRESHOLD = 1.358 / np.sqrt(N_MC)   # 95% KS critical value at N_MC
N_LIMIT = 200_000                      # draws used to represent the limiting law

OUT_DIR = HERE
RESULTS_DIR = OUT_DIR / "results"
FIGURES_DIR = OUT_DIR / "figures"

PARENTS = ("normal", "t", "exp", "chi2", "lognorm")
PARENT_LABELS = {
    "normal": "normal",
    "t": "Student t(8)",
    "exp": "exponential",
    "chi2": "chi-square(1)",
    "lognorm": "lognormal(0.5)",
}


# ---------------------------------------------------------------------------
# theory: fourth-moment matrix and the weighted chi-square limit
# ---------------------------------------------------------------------------

def excess_kurtosis(parent):
    """Excess kurtosis of the standardized marginal used by ``Box_M_Test``.

    All non-normal configurations in this study use the class defaults
    (t: df=8, chi2: df=1, lognorm: sigma=0.5), so the values below match
    ``Box_M_Test``'s defaults.
    """
    if parent == "normal":
        return 0.0
    if parent == "t":
        return 6.0 / (8.0 - 4.0)          # df = 8 by default
    if parent == "exp":
        return 6.0
    if parent == "chi2":
        return 12.0                       # chi-square(1)
    if parent == "lognorm":
        s = 0.5
        return (np.exp(4 * s ** 2) + 2 * np.exp(3 * s ** 2)
                + 3 * np.exp(2 * s ** 2) - 6.0)
    raise ValueError(parent)


def omega_matrix(p, kurt):
    """Asymptotic covariance of ``vec(x x' - I)`` for independent margins.

    ``kurt[a]`` is the excess kurtosis of variable ``a``.  ``Omega`` is the
    matrix whose eigenvalues drive the weighted chi-square null law.
    """
    size = p * p
    omega = np.zeros((size, size))
    for a in range(p):
        for b in range(p):
            i = a * p + b
            for c in range(p):
                for d in range(p):
                    j = c * p + d
                    # E[x_a x_b x_c x_d]: zero unless every index pairs up.
                    if a == b and c == d:
                        moment = (kurt[a] + 3.0) if a == c else 1.0
                    elif a == c and b == d:
                        moment = (kurt[a] + 3.0) if a == b else 1.0
                    elif a == d and b == c:
                        moment = (kurt[a] + 3.0) if a == b else 1.0
                    else:
                        moment = 0.0
                    omega[i, j] = moment - (1.0 if a == b else 0.0) * \
                        (1.0 if c == d else 0.0)
    return omega


def weighted_limit(p, kurt):
    """Return ``(Omega, eigenvalues)`` of the asymptotic null law.

    The limit is ``0.5 * sum_i lambda_i * chi2(k - 1)``.
    """
    omega = omega_matrix(p, kurt)
    eigenvalues = np.linalg.eigvalsh(omega)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    return omega, eigenvalues


def simulate_weighted_limit(eigenvalues, k, n_draws, rng):
    """Draw from ``0.5 * sum lambda_i chi2(k - 1)``."""
    draws = np.zeros(n_draws)
    for lam in eigenvalues:
        draws += 0.5 * lam * rng.chisquare(k - 1, n_draws)
    return draws


def limit_moments(eigenvalues, k):
    """Mean and variance of the weighted chi-square limit, and Satterthwaite fit."""
    mean = 0.5 * (k - 1) * eigenvalues.sum()
    var = 0.5 * (k - 1) * (eigenvalues ** 2).sum()
    scale = var / (2.0 * mean)
    dof = 2.0 * mean ** 2 / var
    return float(mean), float(var), float(scale), float(dof)


def wilson_ci(phat, n, z=1.959963984540054):
    """Wilson score interval, as used elsewhere in the repository."""
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return float(centre - half), float(centre + half)


# ---------------------------------------------------------------------------
# Monte Carlo runs
# ---------------------------------------------------------------------------

def run_config(k, p, ns, parent, N, seed, parent_args=None):
    """Simulate ``Box_M_Test`` once and return ``(model, corrected, raw)``."""
    model = Box_M_Test(k=k, p=p, n=ns, N=N, parent=parent,
                       parent_args=parent_args, seed=seed)
    mc = model.simulate(progress=False)
    return model, mc["statistic"], mc["M"]


def normal_adequacy(N, seed=SEED):
    """Grid over (k, p, n) under multivariate normality."""
    rows = []
    cases = [(k, p, n) for k in (2, 3, 5) for p in (2, 3, 4)
             for n in (10, 20, 50, 100, 200)]
    for index, (k, p, n) in enumerate(cases):
        model, corrected, raw = run_config(k, p, [n] * k, "normal", N,
                                           seed + index)
        nu = model.dof
        row = {
            "k": k, "p": p, "n": n, "group_sizes": list(model.group_sizes),
            "dof": nu, "N": N, "correction": model.correction,
            "mean": float(np.mean(corrected)),
            "var": float(np.var(corrected, ddof=1)),
            "mean_chi2": float(nu), "var_chi2": float(2 * nu),
            "ks_chi2": float(stats.kstest(
                corrected, stats.chi2.cdf, args=(nu,)).statistic),
            "ks_chi2_uncorrected": float(stats.kstest(
                raw, stats.chi2.cdf, args=(nu,)).statistic),
        }
        for alpha in ALPHAS:
            critical = stats.chi2.ppf(1.0 - alpha, nu)
            size = float(np.mean(corrected > critical))
            low, high = wilson_ci(size, N)
            row["type1_{:.2f}".format(alpha)] = size
            row["type1_{:.2f}_ci".format(alpha)] = [low, high]
            row["type1_{:.2f}_uncorrected".format(alpha)] = float(
                np.mean(raw > critical))
        rows.append(row)
        print("  normal k={} p={} n={:3d}: D_KS={:.4f}  size={:.3f}".format(
            k, p, n, row["ks_chi2"], row["type1_0.05"]))
    return rows


def nonnormal_null(N, n_limit, seed=SEED):
    """Same covariance matrix, non-normal margins: nominal chi-square breaks."""
    rows = []
    cases = [(parent, k, p, n) for parent in PARENTS
             for k in (2, 3) for p in (2, 3) for n in (20, 50, 200)]
    # Large-n points document convergence to the asymptotic weighted law.
    cases += [(parent, 3, 2, 1000)
              for parent in ("t", "exp", "chi2", "lognorm")]

    rng = np.random.default_rng(seed)
    for index, (parent, k, p, n) in enumerate(cases):
        model, corrected, _ = run_config(k, p, [n] * k, parent, N,
                                         seed + 1000 + index)
        nu = model.dof
        kurt = excess_kurtosis(parent)
        _omega, eigenvalues = weighted_limit(p, [kurt] * p)
        mean_th, var_th, scale_th, dof_th = limit_moments(eigenvalues, k)
        limit_draws = simulate_weighted_limit(eigenvalues, k, n_limit, rng)

        row = {
            "parent": parent, "k": k, "p": p, "n": n,
            "group_sizes": list(model.group_sizes), "dof": nu, "N": N,
            "excess_kurtosis": float(kurt),
            "eigenvalues": [float(x) for x in eigenvalues],
            "mean_theory": mean_th, "var_theory": var_th,
            "mean": float(np.mean(corrected)),
            "var": float(np.var(corrected, ddof=1)),
            "ks_chi2": float(stats.kstest(
                corrected, stats.chi2.cdf, args=(nu,)).statistic),
            "ks_weighted_limit": float(stats.ks_2samp(
                corrected, limit_draws).statistic),
            "ks_scaled_chi2": float(stats.kstest(
                corrected,
                lambda x, a=scale_th, b=dof_th: stats.chi2.cdf(x / a, b)).statistic),
            "scaled_scale": scale_th, "scaled_dof": dof_th,
        }
        for alpha in ALPHAS:
            critical = stats.chi2.ppf(1.0 - alpha, nu)
            weighted_critical = float(np.quantile(limit_draws, 1.0 - alpha))
            size = float(np.mean(corrected > critical))
            size_weighted = float(np.mean(corrected > weighted_critical))
            row["type1_{:.2f}".format(alpha)] = size
            row["type1_{:.2f}_ci".format(alpha)] = list(wilson_ci(size, N))
            row["type1_{:.2f}_weighted".format(alpha)] = size_weighted
            row["critical_{:.2f}_weighted".format(alpha)] = weighted_critical
        rows.append(row)
        print("  {:<8s} k={} p={} n={:4d}: mean={:7.2f} (lim {:7.2f})  "
              "size={:.3f}  D_chi2={:.4f}  D_lim={:.4f}".format(
                  parent, k, p, n, row["mean"], mean_th, row["type1_0.05"],
                  row["ks_chi2"], row["ks_weighted_limit"]))
    return rows


def unbalanced_groups(N, seed=SEED):
    """Unequal group sizes under normality: the correction must still work."""
    cases = [
        (3, 2, [20, 20, 20]),
        (3, 2, [15, 25, 40]),
        (3, 3, [20, 30, 60]),
        (2, 2, [12, 80]),
        (3, 2, [10, 10, 100]),
    ]
    rows = []
    for index, (k, p, sizes) in enumerate(cases):
        model, corrected, _ = run_config(k, p, sizes, "normal", N,
                                         seed + 2000 + index)
        nu = model.dof
        row = {
            "k": k, "p": p, "group_sizes": sizes, "dof": nu, "N": N,
            "correction": model.correction,
            "mean": float(np.mean(corrected)),
            "var": float(np.var(corrected, ddof=1)),
            "ks_chi2": float(stats.kstest(
                corrected, stats.chi2.cdf, args=(nu,)).statistic),
        }
        for alpha in ALPHAS:
            critical = stats.chi2.ppf(1.0 - alpha, nu)
            size = float(np.mean(corrected > critical))
            row["type1_{:.2f}".format(alpha)] = size
            row["type1_{:.2f}_ci".format(alpha)] = list(wilson_ci(size, N))
        rows.append(row)
        print("  unbalanced n={}: D_KS={:.4f}  size={:.3f}".format(
            sizes, row["ks_chi2"], row["type1_0.05"]))
    return rows


def consistency_checks():
    """Analytic checks that tie the implementation to established results."""
    checks = {}

    # (a) p = 1: Box's M must reduce exactly to the Bartlett statistic.
    rng = np.random.default_rng(7)
    groups = [rng.standard_normal((25, 1)) for _ in range(3)]
    box = Box_M_Test(k=3, p=1, n=25, N=10).statistic(groups)
    data = np.concatenate([g.ravel() for g in groups]).reshape(3, 25)
    n, k = 25, 3
    variances = np.var(data, axis=1, ddof=1)
    pooled = np.sum((n - 1) * variances) / (k * n - k)
    bartlett = (k * n - k) * np.log(pooled) - (n - 1) * np.sum(np.log(variances))
    bartlett_c = (1.0 / (3.0 * (k - 1))) * (k / (n - 1) - 1.0 / (k * n - k))
    checks["p1_reduces_to_bartlett"] = {
        "box_M": box["M"], "bartlett": float(bartlett),
        "abs_diff": float(abs(box["M"] - bartlett)),
        "box_c": box["c"], "bartlett_c": float(bartlett_c),
        "pass": bool(abs(box["M"] - bartlett) < 1e-10
                     and abs(box["c"] - bartlett_c) < 1e-12),
    }

    # (b) Under normality Omega has p(p+1)/2 eigenvalues equal to 2, hence the
    #     weighted limit collapses to chi2(nu). Check numerically.
    collapse = {}
    for p in (2, 3, 4):
        _omega, eigenvalues = weighted_limit(p, [0.0] * p)
        expected = 2.0 * p * (p + 1) / 2.0
        collapse["p{}".format(p)] = {
            "eigenvalues": np.round(np.sort(eigenvalues), 10)[::-1].tolist(),
            "sum": float(eigenvalues.sum()), "expected_sum": float(expected),
            "pass": bool(np.allclose(eigenvalues, 2.0)
                         and len(eigenvalues) == p * (p + 1) // 2),
        }
    checks["normal_limit_is_chi2"] = collapse

    # (c) The class summary agrees with a direct simulation of a small case.
    model = Box_M_Test(k=3, p=2, n=30, N=4000, seed=11)
    summary = model.summary(progress=False)
    checks["summary_smoke_test"] = {
        "dof": summary["dof"], "ks_chi2": summary["ks_chi2"],
        "type1": summary["type1"],
        "pass": bool(summary["dof"] == 6 and summary["ks_chi2"] < 0.05),
    }
    return checks


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote", path.relative_to(PACKAGE_ROOT))
    return path


def figure_normal_adequacy(rows):
    """D_KS and type I error versus n under multivariate normality."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    colours = {2: "#1f77b4", 3: "#d95f02", 4: "#2ca02c"}
    for p in (2, 3, 4):
        sub = [r for r in rows if r["k"] == 3 and r["p"] == p]
        sub.sort(key=lambda r: r["n"])
        ns = [r["n"] for r in sub]
        axes[0].plot(ns, [r["ks_chi2"] for r in sub], "o-", ms=4,
                     color=colours[p], label="p = {}".format(p))
        sizes = [r["type1_0.05"] for r in sub]
        axes[1].plot(ns, sizes, "o-", ms=4, color=colours[p],
                     label="p = {}".format(p))
    axes[0].axhline(KS_THRESHOLD, color="0.4", ls="--", lw=1,
                    label="95% KS threshold")
    axes[0].set(xscale="log", xlabel="n (per group)",
                ylabel="D_KS vs $\\chi^2(\\nu)$",
                title="Normal populations: KS distance to $\\chi^2(\\nu)$ (k = 3)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].axhline(0.05, color="0.4", ls="--", lw=1, label="nominal 0.05")
    axes[1].fill_between([8, 250], 0.05 - 1.96 * np.sqrt(0.05 * 0.95 / N_MC),
                         0.05 + 1.96 * np.sqrt(0.05 * 0.95 / N_MC),
                         color="0.85", zorder=0)
    axes[1].set(xscale="log", xlabel="n (per group)",
                ylabel="empirical type I error at 5%",
                title="Normal populations: type I error (k = 3)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, "fig_box_m_normal_adequacy.png")


def figure_nonnormal(rows):
    """Size inflation and the weighted chi-square limit under non-normality."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9))

    colours = {"normal": "0.2", "t": "#1f77b4", "exp": "#d95f02",
               "chi2": "#7570b3", "lognorm": "#2ca02c"}
    for parent in PARENTS:
        sub = [r for r in rows if r["parent"] == parent
               and r["k"] == 3 and r["p"] == 2]
        sub.sort(key=lambda r: r["n"])
        axes[0].plot([r["n"] for r in sub], [r["type1_0.05"] for r in sub],
                     "o-", ms=4, color=colours[parent],
                     label=PARENT_LABELS[parent])
    axes[0].axhline(0.05, color="0.4", ls="--", lw=1)
    axes[0].set(xscale="log", xlabel="n (per group)",
                ylabel="empirical type I error at 5%",
                title="Same covariance, different marginals\n(k = 3, p = 2)")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    # Distances at a common n: nominal chi-square vs weighted limit vs
    # moment-matched scaled chi-square.
    at_n = 200
    parents = list(PARENTS)
    width = 0.26
    xpos = np.arange(len(parents))
    for offset, key, colour, label in [
            (-width, "ks_chi2", "#d62728", "nominal $\\chi^2(\\nu)$"),
            (0.0, "ks_weighted_limit", "#1f77b4", "weighted $\\chi^2$ limit"),
            (width, "ks_scaled_chi2", "#2ca02c", "scaled $\\chi^2$ (2-moment)")]:
        vals = []
        for parent in parents:
            match = [r for r in rows if r["parent"] == parent
                     and r["k"] == 3 and r["p"] == 2 and r["n"] == at_n]
            vals.append(match[0][key] if match else np.nan)
        axes[1].bar(xpos + offset, vals, width, color=colour, label=label,
                    edgecolor="white")
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels([PARENT_LABELS[p] for p in parents],
                            rotation=25, ha="right", fontsize=7)
    axes[1].set(ylabel="D_KS", title="Distance to the reference law\n(n = 200)")
    axes[1].legend(fontsize=6.5)
    axes[1].grid(alpha=0.3, axis="y")

    # Density overlay for one strongly non-normal configuration.
    parent, k, p, n = "exp", 3, 2, 200
    model, corrected, _ = run_config(k, p, [n] * k, parent, N_MC, SEED + 5000)
    nu = model.dof
    kurt = excess_kurtosis(parent)
    _omega, eigenvalues = weighted_limit(p, [kurt] * p)
    _mean_th, _var_th, scale_th, dof_th = limit_moments(eigenvalues, k)
    rng = np.random.default_rng(SEED + 5001)
    limit_draws = simulate_weighted_limit(eigenvalues, k, 100_000, rng)

    axes[2].hist(corrected, bins=80, density=True, color="0.75",
                 edgecolor="white", linewidth=0.2, label="MC null")
    grid = np.linspace(0.01, np.quantile(corrected, 0.995), 400)
    axes[2].plot(grid, stats.chi2.pdf(grid, nu), color="#d62728", lw=1.6,
                 label="$\\chi^2(\\nu={})$".format(nu))
    axes[2].plot(grid, stats.chi2.pdf(grid / scale_th, dof_th) / scale_th,
                 color="#2ca02c", lw=1.4, ls="--",
                 label="scaled $\\chi^2$")
    kde = stats.gaussian_kde(limit_draws, bw_method=0.08)
    axes[2].plot(grid, kde(grid), color="#1f77b4", lw=1.6,
                 label="weighted limit")
    axes[2].set(xlabel="$(1-c)M$", ylabel="density",
                title="Exponential margins, k = 3, p = 2, n = 200")
    axes[2].legend(fontsize=7)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    return _save(fig, "fig_box_m_nonnormal.png")


def figure_limit_convergence(rows):
    """MC mean/variance approach the fourth-moment limit as n grows."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    colours = {"t": "#1f77b4", "exp": "#d95f02", "chi2": "#7570b3",
               "lognorm": "#2ca02c"}
    for parent, colour in colours.items():
        sub = [r for r in rows if r["parent"] == parent
               and r["k"] == 3 and r["p"] == 2]
        sub.sort(key=lambda r: r["n"])
        ns = np.array([r["n"] for r in sub])
        axes[0].plot(ns, [r["mean"] / r["mean_theory"] for r in sub], "o-",
                     ms=4, color=colour, label=PARENT_LABELS[parent])
        axes[1].plot(ns, [r["var"] / r["var_theory"] for r in sub], "o-",
                     ms=4, color=colour, label=PARENT_LABELS[parent])
    for ax, ylabel, title in [
            (axes[0], "E[(1-c)M] / limit mean",
             "Mean approaches the weighted-$\\chi^2$ limit"),
            (axes[1], "Var[(1-c)M] / limit variance",
             "Variance approaches the weighted-$\\chi^2$ limit")]:
        ax.axhline(1.0, color="0.4", ls="--", lw=1)
        ax.set(xscale="log", xlabel="n (per group)", ylabel=ylabel, title=title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, "fig_box_m_limit_convergence.png")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    """Command-line entry point."""
    global N_MC, KS_THRESHOLD, SEED

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=DEFAULT_N_MC,
                        help="MC replications per configuration (default %d)"
                             % DEFAULT_N_MC)
    parser.add_argument("--n-limit", type=int, default=N_LIMIT,
                        help="draws representing the limiting law (default %d)"
                             % N_LIMIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="master seed (default %d)" % DEFAULT_SEED)
    parser.add_argument("--quick", action="store_true",
                        help="fast smoke run: N = 2000")
    parser.add_argument("--no-figures", action="store_true",
                        help="skip figure generation")
    args = parser.parse_args(argv)

    n_mc = 2000 if args.quick else args.N
    n_limit = 20_000 if args.quick else args.n_limit

    N_MC = n_mc
    KS_THRESHOLD = 1.358 / np.sqrt(N_MC)
    SEED = args.seed

    started = time.time()
    print("Box's M null-distribution study")
    print("  N_MC = {}  N_LIMIT = {}  seed = {}".format(n_mc, n_limit, SEED))

    print("\n[1/4] Normal populations: KS distance and type I error")
    normal_rows = normal_adequacy(n_mc, seed=args.seed)

    print("\n[2/4] Non-normal populations with H0 true")
    nonnormal_rows = nonnormal_null(n_mc, n_limit, seed=args.seed)

    print("\n[3/4] Unequal group sizes")
    unbalanced_rows = unbalanced_groups(n_mc, seed=args.seed)

    print("\n[4/4] Analytic consistency checks")
    checks = consistency_checks()
    for name, value in checks.items():
        if name == "normal_limit_is_chi2":
            print("  {}: {}".format(name,
                                    {k: v["pass"] for k, v in value.items()}))
        else:
            print("  {}: pass = {}".format(name, value.get("pass")))

    figures = []
    if not args.no_figures:
        print("\nFigures")
        figures.append(str(figure_normal_adequacy(normal_rows).relative_to(PACKAGE_ROOT)))
        figures.append(str(figure_nonnormal(nonnormal_rows).relative_to(PACKAGE_ROOT)))
        figures.append(str(figure_limit_convergence(nonnormal_rows).relative_to(PACKAGE_ROOT)))

    normal_final = [r for r in normal_rows if r["n"] >= 50]
    nonnormal_at = {(r["parent"], r["n"], r["k"], r["p"]): r
                    for r in nonnormal_rows}
    headline = {
        "normal_max_ks_n_ge_50": max(r["ks_chi2"] for r in normal_final),
        "normal_size_range_n_ge_50": [
            min(r["type1_0.05"] for r in normal_final),
            max(r["type1_0.05"] for r in normal_final)],
        "nonnormal_size_at_n_200": {
            parent: nonnormal_at[(parent, 200, 3, 2)]["type1_0.05"]
            for parent in PARENTS},
        "weighted_limit_ks_at_n_200": {
            parent: nonnormal_at[(parent, 200, 3, 2)]["ks_weighted_limit"]
            for parent in PARENTS},
        "nonnormal_size_at_n_1000_weighted": {
            parent: nonnormal_at[(parent, 1000, 3, 2)]["type1_0.05_weighted"]
            for parent in ("t", "exp", "chi2", "lognorm")},
    }

    payload = {
        "meta": {
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "platform": platform.platform(),
            "N_MC": n_mc,
            "N_LIMIT": n_limit,
            "seed": SEED,
            "quick": bool(args.quick),
            "ks_threshold": KS_THRESHOLD,
            "runtimes_seconds": round(time.time() - started, 1),
            "script": "extensions/box_m/study_box_m_null.py",
        },
        "normal_adequacy": normal_rows,
        "nonnormal_null": nonnormal_rows,
        "unbalanced": unbalanced_rows,
        "checks": checks,
        "headline": headline,
        "figures": figures,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # A smoke run must not clobber the canonical full-run artifact.
    out = RESULTS_DIR / ("box_m_null_results_quick.json" if args.quick
                         else "box_m_null_results.json")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nwrote", out.relative_to(PACKAGE_ROOT))
    print("elapsed: {:.1f} s".format(time.time() - started))

    print("\nHeadline")
    print("  normal, n >= 50: max D_KS = {:.4f}; type I error range = "
          "[{:.3f}, {:.3f}]".format(
              headline["normal_max_ks_n_ge_50"], *headline["normal_size_range_n_ge_50"]))
    print("  non-normal, size at n = 200 (k=3, p=2):")
    for parent in PARENTS:
        print("    {:<8s} size = {:.3f}   D_KS to weighted limit = {:.4f}".format(
            parent, headline["nonnormal_size_at_n_200"][parent],
            headline["weighted_limit_ks_at_n_200"][parent]))
    print("  non-normal, size at n = 1000 with the weighted-limit critical "
          "value (nominal 5%):")
    for parent, size in headline["nonnormal_size_at_n_1000_weighted"].items():
        print("    {:<8s} size = {:.3f}".format(parent, size))
    return 0


if __name__ == "__main__":
    sys.exit(main())
