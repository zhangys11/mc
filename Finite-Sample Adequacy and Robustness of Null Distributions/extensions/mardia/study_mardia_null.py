#!/usr/bin/env python3
"""Monte Carlo verification of the null distributions of Mardia's tests.

Mardia's multivariate skewness and kurtosis statistics are the two classical
moment-based tests of multivariate normality.  This script verifies, by Monte
Carlo, the two asymptotic normal-theory references used by the test

    n b_{1,p} / 6  ->  chi2( p (p + 1) (p + 2) / 6 )
    Z_K = (b_{2,p} - p (p + 2)) / sqrt(8 p (p + 2) / n)  ->  N(0, 1),

and quantifies how far the finite-sample null distributions are from those
references.  The second point matters because the kurtosis statistic has a
known finite-sample mean under normality,

    E[b_{2,p}] = p (p + 2) (n - 1) / (n + 1),

so the nominal N(0, 1) reference is neither centred nor unit-variance at small
n.  The script reports both biases and the corresponding empirical rejection
rates.

It also runs the same diagnostics on non-normal populations (multivariate t,
independent exponential margins) to show how the moment estimates behave under
alternatives and to verify the multivariate-t kurtosis formula

    beta_{2,p} = p (p + 2) (nu - 2) / (nu - 4)

for nu > 4.

Writes ``results/mardia_null_results.json`` and three figures under
``figures/``.  Run from the repository root::

    python3 mardia/study_mardia_null.py            # full run, ~2-3 minutes
    python3 mardia/study_mardia_null.py --quick    # smoke run, ~15 seconds
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

from mc.samplings import Mardia_Test  # noqa: E402

SEED = 2024
N_MC = 10_000
# Defaults captured before ``main`` mutates the module-level values, so that a
# second ``main`` call in the same interpreter starts from the documented
# defaults instead of inheriting a previous ``--quick``/``--seed`` setting.
DEFAULT_SEED = SEED
DEFAULT_N_MC = N_MC
ALPHAS = (0.01, 0.05)
KS_THRESHOLD = 1.358 / np.sqrt(N_MC)   # 95% KS critical value at N_MC

OUT_DIR = HERE
RESULTS_DIR = OUT_DIR / "results"
FIGURES_DIR = OUT_DIR / "figures"

PARENTS = ("normal", "t", "exp")
PARENT_LABELS = {
    "normal": "normal",
    "t": "multivariate t",
    "exp": "exponential margins",
}


# ---------------------------------------------------------------------------
# population moments of the non-normal alternatives
# ---------------------------------------------------------------------------

def theoretical_moments(parent, p, parent_args=None):
    """Return ``(beta1, beta2)`` of the parent population.

    For the multivariate t, ``beta2 = p(p+2)(nu-2)/(nu-4)`` with the true
    covariance matrix.  For independent standardized margins, the joint
    moments factorize: ``beta1 = p * gamma1^2`` and
    ``beta2 = p(p+2) + p * gamma2`` where ``gamma1``, ``gamma2`` are the
    marginal skewness and excess kurtosis.
    """
    args = dict(parent_args or {})
    if parent == "normal":
        return 0.0, float(p * (p + 2))
    if parent == "t":
        nu = float(args.get("df", 10.0))
        return 0.0, float(p * (p + 2) * (nu - 2.0) / (nu - 4.0))
    if parent == "exp":
        gamma1, gamma2 = 2.0, 6.0
        return float(p * gamma1 ** 2), float(p * (p + 2) + p * gamma2)
    if parent == "chi2":
        df = float(args.get("df", 1.0))
        gamma1, gamma2 = np.sqrt(8.0 / df), 12.0 / df
        return float(p * gamma1 ** 2), float(p * (p + 2) + p * gamma2)
    if parent == "lognorm":
        sigma = float(args.get("sigma", 0.5))
        gamma1 = (np.exp(sigma ** 2) + 2.0) * np.sqrt(np.exp(sigma ** 2) - 1.0)
        gamma2 = (np.exp(4 * sigma ** 2) + 2.0 * np.exp(3 * sigma ** 2)
                  + 3.0 * np.exp(2 * sigma ** 2) - 6.0)
        return float(p * gamma1 ** 2), float(p * (p + 2) + p * gamma2)
    raise ValueError(parent)


def wilson_ci(phat, n, z=1.959963984540054):
    """Wilson score interval."""
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return float(centre - half), float(centre + half)


# ---------------------------------------------------------------------------
# Monte Carlo runs
# ---------------------------------------------------------------------------

def run_config(n, p, N, parent="normal", parent_args=None, seed=SEED):
    """Simulate one configuration and return the model and its MC arrays."""
    model = Mardia_Test(n=n, p=p, N=N, parent=parent,
                        parent_args=parent_args, seed=seed)
    return model, model.simulate(progress=False)


def normal_calibration(N, seed=SEED):
    """Null calibration over a (p, n) grid under multivariate normality."""
    cases = []
    for p in (1, 2, 3, 5, 10):
        for n in (10, 20, 50, 100, 200):
            if n > p + 1:          # the class requires n > p + 1
                cases.append((p, n))

    rows = []
    for index, (p, n) in enumerate(cases):
        model, mc = run_config(n, p, N, seed=seed + index)
        dof = model.dof_skew
        skew = mc["skew"]
        kurt = mc["kurt"]
        b2 = mc["b2"]
        expected_finite = model.expected_kurtosis_finite
        se_b2 = float(np.std(b2, ddof=1) / np.sqrt(N))
        predicted_mean_z = float(
            (expected_finite - p * (p + 2)) / np.sqrt(8.0 * p * (p + 2) / n))

        row = {
            "p": p, "n": n, "N": N, "dof_skew": dof,
            "mean_skew": float(np.mean(skew)),
            "var_skew": float(np.var(skew, ddof=1)),
            "mean_chi2": float(dof), "var_chi2": float(2 * dof),
            "ks_skew": float(stats.kstest(
                skew, stats.chi2.cdf, args=(dof,)).statistic),
            "mean_b2": float(np.mean(b2)),
            "sd_b2": float(np.std(b2, ddof=1)),
            "se_b2": se_b2,
            "expected_b2": float(p * (p + 2)),
            "expected_b2_finite": float(expected_finite),
            "b2_bias_in_se": float((np.mean(b2) - expected_finite) / se_b2),
            "mean_z": float(np.mean(kurt)),
            "sd_z": float(np.std(kurt, ddof=1)),
            "se_mean_z": float(np.std(kurt, ddof=1) / np.sqrt(N)),
            "predicted_mean_z": predicted_mean_z,
            "ks_kurt": float(stats.kstest(kurt, stats.norm.cdf).statistic),
        }
        for alpha in ALPHAS:
            key = "{:.2f}".format(alpha)
            skew_critical = stats.chi2.ppf(1.0 - alpha, dof)
            size = float(np.mean(skew > skew_critical))
            row["type1_skew_" + key] = size
            row["type1_skew_" + key + "_ci"] = list(wilson_ci(size, N))

            kurt_critical = stats.norm.ppf(1.0 - alpha / 2.0)
            size = float(np.mean(np.abs(kurt) > kurt_critical))
            row["type1_kurt_" + key] = size
            row["type1_kurt_" + key + "_ci"] = list(wilson_ci(size, N))
            row["type1_kurt_upper_" + key] = float(
                np.mean(kurt > stats.norm.ppf(1.0 - alpha)))
            # Diagnostic: the exact finite-sample mean is known, so centring
            # removes the mean bias; the variance miscalibration remains.
            centred = kurt - predicted_mean_z
            row["type1_kurt_centred_" + key] = float(
                np.mean(np.abs(centred) > kurt_critical))
        rows.append(row)
        print("  normal p={:2d} n={:3d}: E[skew]={:6.2f} (dof {:2d})  "
              "E[b2]={:7.2f} (exact {:7.2f})  meanZ={:6.2f} sdZ={:5.2f}  "
              "size skew={:.3f} kurt={:.3f}".format(
                  p, n, row["mean_skew"], dof, row["mean_b2"],
                  expected_finite, row["mean_z"], row["sd_z"],
                  row["type1_skew_0.05"], row["type1_kurt_0.05"]))
    return rows


def nonnormal_alternatives(N, seed=SEED):
    """Moment estimates and power for non-normal parents (p = 2 and 3)."""
    rows = []
    cases = []
    for parent, parent_args in [
            ("t", {"df": 5.0}), ("t", {"df": 10.0}), ("t", {"df": 30.0}),
            ("exp", None)]:
        for p in (2, 3):
            for n in (50, 200):
                cases.append((parent, parent_args, p, n))

    for index, (parent, parent_args, p, n) in enumerate(cases):
        model, mc = run_config(n, p, N, parent, parent_args,
                               seed + 1000 + index)
        beta1, beta2 = theoretical_moments(parent, p, parent_args)
        b1 = mc["b1"]
        b2 = mc["b2"]
        skew = mc["skew"]
        kurt = mc["kurt"]
        row = {
            "parent": parent, "parent_args": dict(parent_args or {}),
            "p": p, "n": n, "N": N,
            "beta1_theory": beta1, "beta2_theory": beta2,
            "mean_b1": float(np.mean(b1)), "mean_b2": float(np.mean(b2)),
            "ratio_beta2": float(np.mean(b2) / beta2),
            "mean_z": float(np.mean(kurt)),
            "sd_z": float(np.std(kurt, ddof=1)),
        }
        for alpha in ALPHAS:
            key = "{:.2f}".format(alpha)
            row["power_skew_" + key] = float(np.mean(
                skew > stats.chi2.ppf(1.0 - alpha, model.dof_skew)))
            row["power_kurt_" + key] = float(np.mean(
                np.abs(kurt) > stats.norm.ppf(1.0 - alpha / 2.0)))
            row["power_kurt_upper_" + key] = float(np.mean(
                kurt > stats.norm.ppf(1.0 - alpha)))
        rows.append(row)
        print("  {:<4s} p={} n={:3d}: E[b1]={:7.2f} (theory {:6.2f})  "
              "E[b2]={:7.2f} (theory {:6.2f})  meanZ={:8.2f}  "
              "power skew={:.3f} kurt={:.3f}".format(
                  parent, p, n, row["mean_b1"], beta1, row["mean_b2"], beta2,
                  row["mean_z"], row["power_skew_0.05"],
                  row["power_kurt_0.05"]))
    return rows


def t_limit_convergence(N, seed=SEED):
    """Multivariate-t kurtosis formula: E[b2] -> p(p+2)(nu-2)/(nu-4)."""
    rows = []
    for index, nu in enumerate((5.0, 10.0, 30.0)):
        for n in (200, 500, 1000):
            reps = N if n <= 500 else max(400, N // 5)
            _model, mc = run_config(n, 2, reps, "t", {"df": nu},
                                    seed + 2000 + index * 10 + n)
            _beta1, beta2 = theoretical_moments("t", 2, {"df": nu})
            row = {
                "df": nu, "p": 2, "n": n, "N": reps,
                "beta2_theory": beta2,
                "mean_b2": float(np.mean(mc["b2"])),
                "ratio": float(np.mean(mc["b2"]) / beta2),
            }
            rows.append(row)
            print("  t(df={:4.1f}) n={:4d} N={:4d}: E[b2]={:7.3f}  "
                  "theory={:7.3f}  ratio={:.3f}".format(
                      nu, n, reps, row["mean_b2"], beta2, row["ratio"]))
    return rows


def consistency_checks():
    """Analytic checks on the implementation."""
    checks = {}

    # (a) p = 1: the Mardia statistics reduce to the univariate ones.
    rng = np.random.default_rng(7)
    x = rng.standard_normal(60)
    result = Mardia_Test.statistics(x[:, None])
    centered = x - x.mean()
    m2 = np.mean(centered ** 2)
    m3 = np.mean(centered ** 3)
    m4 = np.mean(centered ** 4)
    g1 = m3 / m2 ** 1.5
    checks["p1_reduction"] = {
        "mardia_skew": result["skew"],
        "n_g1_squared_over_6": float(60 * g1 ** 2 / 6.0),
        "mardia_b2": result["b2"],
        "m4_over_m2_squared": float(m4 / m2 ** 2),
        "pass": bool(abs(result["skew"] - 60 * g1 ** 2 / 6.0) < 1e-10
                     and abs(result["b2"] - m4 / m2 ** 2) < 1e-10),
    }

    # (b) univariate exact finite-sample mean of the skewness statistic:
    #     E[n g1^2 / 6] = n (n - 2) / ((n + 1)(n + 3)).
    rows = []
    for n in (20, 50):
        model, mc = run_config(n, 1, 20_000, seed=3000 + n)
        theory = n * (n - 2.0) / ((n + 1.0) * (n + 3.0))
        observed = float(np.mean(mc["skew"]))
        se = float(np.std(mc["skew"], ddof=1) / np.sqrt(len(mc["skew"])))
        rows.append({"n": n, "observed": observed, "theory": float(theory),
                     "bias_in_se": float((observed - theory) / se)})
    checks["p1_skew_finite_sample_mean"] = {
        "rows": rows,
        "pass": bool(all(abs(row["bias_in_se"]) < 4 for row in rows)),
    }

    # (c) summary smoke test
    model = Mardia_Test(n=50, p=2, N=4000, seed=11)
    summary = model.summary(progress=False)
    checks["summary_smoke_test"] = {
        "dof_skew": summary["dof_skew"],
        "ks_skew": summary["ks_skew"],
        "ks_kurt": summary["ks_kurt"],
        "pass": bool(summary["dof_skew"] == 4
                     and 0.0 < summary["ks_skew"] < 0.3
                     and 0.0 < summary["ks_kurt"] < 0.4),
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


def figure_skew_null(rows):
    """Skewness null: mean/dof and type I error versus n."""
    colours = {1: "#666666", 2: "#1f77b4", 3: "#d95f02", 5: "#2ca02c",
               10: "#9467bd"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for p in (1, 2, 3, 5, 10):
        sub = sorted([r for r in rows if r["p"] == p], key=lambda r: r["n"])
        ns = [r["n"] for r in sub]
        axes[0].plot(ns, [r["mean_skew"] / r["dof_skew"] for r in sub],
                     "o-", ms=4, color=colours[p], label="p = {}".format(p))
        axes[1].plot(ns, [r["type1_skew_0.05"] for r in sub], "o-", ms=4,
                     color=colours[p], label="p = {}".format(p))
    axes[0].axhline(1.0, color="0.4", ls="--", lw=1)
    axes[0].set(xscale="log", xlabel="n",
                ylabel="E[n b1/6] / dof",
                title="Skewness statistic: mean / dof")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    axes[1].axhline(0.05, color="0.4", ls="--", lw=1, label="nominal 0.05")
    band = 1.96 * np.sqrt(0.05 * 0.95 / N_MC)
    axes[1].fill_between([8, 250], 0.05 - band, 0.05 + band,
                         color="0.85", zorder=0)
    axes[1].set(xscale="log", xlabel="n",
                ylabel="empirical type I error at 5%",
                title="Skewness test: type I error")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, "fig_mardia_skew_null.png")


def figure_kurt_null(rows):
    """Kurtosis null: exact finite-sample mean, Z moments and size."""
    colours = {1: "#666666", 2: "#1f77b4", 3: "#d95f02", 5: "#2ca02c",
               10: "#9467bd"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))

    grid = np.linspace(8.0, 200.0, 200)
    axes[0].plot(grid, (grid - 1.0) / (grid + 1.0), color="0.4", ls="--",
                 lw=1, label="$(n-1)/(n+1)$")
    for p in (1, 2, 3, 5, 10):
        sub = sorted([r for r in rows if r["p"] == p], key=lambda r: r["n"])
        axes[0].plot([r["n"] for r in sub],
                     [r["mean_b2"] / r["expected_b2"] for r in sub],
                     "o-", ms=4, color=colours[p], label="p = {}".format(p))
    axes[0].set(xscale="log", xlabel="n",
                ylabel="E[b2] / [p(p+2)]",
                title="Finite-sample mean of b2")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    for p in (1, 2, 3, 5, 10):
        sub = sorted([r for r in rows if r["p"] == p], key=lambda r: r["n"])
        axes[1].plot([r["n"] for r in sub], [r["mean_z"] for r in sub],
                     "o-", ms=4, color=colours[p], label="mean, p={}".format(p))
        axes[2].plot([r["n"] for r in sub], [r["sd_z"] for r in sub],
                     "s--", ms=3, color=colours[p], label="sd, p={}".format(p))
    axes[1].axhline(0.0, color="0.4", ls="--", lw=1)
    axes[1].set(xscale="log", xlabel="n", ylabel="E[Z_K]",
                title="Kurtosis Z: finite-sample mean")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=6)
    axes[2].axhline(1.0, color="0.4", ls="--", lw=1)
    axes[2].set(xscale="log", xlabel="n", ylabel="sd(Z_K)",
                title="Kurtosis Z: finite-sample sd")
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=6)
    fig.tight_layout()
    return _save(fig, "fig_mardia_kurt_null.png")


def parent_key(row):
    """Short key of a non-normal parent, e.g. 't5' or 'exp'."""
    if row["parent"] == "t":
        return "t{:.0f}".format(row["parent_args"]["df"])
    return row["parent"]


PARENT_KEYS = ("t5", "t10", "t30", "exp")
PARENT_KEY_LABELS = {"t5": "t(5)", "t10": "t(10)", "t30": "t(30)",
                     "exp": "exp"}


def figure_nonnormal(rows, limit_rows):
    """Moment estimates and power under non-normal parents."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9))
    colours = {"t5": "#1f77b4", "t10": "#d95f02", "t30": "#7570b3",
               "exp": "#2ca02c"}

    at_n = 200
    width = 0.38
    xpos = np.arange(len(PARENT_KEYS))
    for offset, key, colour, lab in [
            (-width / 2, "power_skew_0.05", "#d62728", "skewness test"),
            (width / 2, "power_kurt_0.05", "#1f77b4", "kurtosis test")]:
        vals = []
        for parent in PARENT_KEYS:
            match = [r for r in rows
                     if parent_key(r) == parent and r["p"] == 2
                     and r["n"] == at_n]
            vals.append(match[0][key] if match else np.nan)
        axes[0].bar(xpos + offset, vals, width, color=colour, label=lab,
                    edgecolor="white")
    axes[0].set_xticks(xpos)
    axes[0].set_xticklabels([PARENT_KEY_LABELS[k] for k in PARENT_KEYS])
    axes[0].set(ylim=(0, 1.05), ylabel="rejection rate at 5%",
                title="Power against non-normal populations\n(p = 2, n = 200)")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3, axis="y")

    for parent, colour in colours.items():
        sub = sorted([r for r in rows
                      if parent_key(r) == parent and r["p"] == 2],
                     key=lambda r: r["n"])
        axes[1].plot([r["n"] for r in sub], [r["ratio_beta2"] for r in sub],
                     "o-", ms=4, color=colour,
                     label=PARENT_KEY_LABELS[parent])
    axes[1].axhline(1.0, color="0.4", ls="--", lw=1)
    axes[1].set(xscale="log", xlabel="n", ylabel="E[b2] / beta2",
                title="Kurtosis estimate vs population value\n(p = 2)")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)

    for nu, colour in [(5.0, "#1f77b4"), (10.0, "#d95f02"),
                       (30.0, "#7570b3")]:
        sub = sorted([r for r in limit_rows if r["df"] == nu],
                     key=lambda r: r["n"])
        axes[2].plot([r["n"] for r in sub], [r["ratio"] for r in sub],
                     "o-", ms=4, color=colour,
                     label="t({:.0f})".format(nu))
    axes[2].axhline(1.0, color="0.4", ls="--", lw=1)
    axes[2].set(xscale="log", xlabel="n", ylabel="E[b2] / beta2",
                title="t populations: convergence to\n"
                      "$p(p+2)(\\nu-2)/(\\nu-4)$ (p = 2)")
    axes[2].legend(fontsize=7)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    return _save(fig, "fig_mardia_nonnormal.png")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    """Command-line entry point."""
    global N_MC, KS_THRESHOLD

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=DEFAULT_N_MC,
                        help="MC replications per configuration (default %d)"
                             % DEFAULT_N_MC)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="master seed (default %d)" % DEFAULT_SEED)
    parser.add_argument("--quick", action="store_true",
                        help="fast smoke run: N = 2000")
    parser.add_argument("--no-figures", action="store_true",
                        help="skip figure generation")
    args = parser.parse_args(argv)

    N_MC = 2000 if args.quick else args.N
    KS_THRESHOLD = 1.358 / np.sqrt(N_MC)

    started = time.time()
    print("Mardia's tests: null-distribution study")
    print("  N_MC = {}  seed = {}".format(N_MC, args.seed))

    print("\n[1/4] Multivariate normal: null calibration")
    normal_rows = normal_calibration(N_MC, seed=args.seed)

    print("\n[2/4] Non-normal alternatives: moments and power")
    nonnormal_rows = nonnormal_alternatives(N_MC, seed=args.seed)

    print("\n[3/4] Multivariate t: kurtosis formula at large n")
    limit_rows = t_limit_convergence(N_MC, seed=args.seed)

    print("\n[4/4] Analytic consistency checks")
    checks = consistency_checks()

    checks["b2_exact_finite_sample_mean"] = {
        "max_deviation_in_se": max(abs(r["b2_bias_in_se"]) for r in normal_rows),
        "pass": bool(max(abs(r["b2_bias_in_se"]) for r in normal_rows) < 4.0),
    }
    checks["z_exact_finite_sample_mean"] = {
        "max_deviation_in_se": max(
            abs(r["mean_z"] - r["predicted_mean_z"]) / r["se_mean_z"]
            for r in normal_rows),
        "pass": bool(max(
            abs(r["mean_z"] - r["predicted_mean_z"]) / r["se_mean_z"]
            for r in normal_rows) < 4.0),
    }
    for name, value in checks.items():
        print("  {}: pass = {}".format(name, value.get("pass")))

    figures = []
    if not args.no_figures:
        print("\nFigures")
        figures.append(str(figure_skew_null(normal_rows).relative_to(PACKAGE_ROOT)))
        figures.append(str(figure_kurt_null(normal_rows).relative_to(PACKAGE_ROOT)))
        figures.append(str(figure_nonnormal(nonnormal_rows, limit_rows)
                           .relative_to(PACKAGE_ROOT)))

    p2 = [r for r in normal_rows if r["p"] == 2]
    p2 = sorted(p2, key=lambda r: r["n"])
    large = [r for r in normal_rows if r["n"] == 200]
    headline = {
        "normal_size_range_skew_0.05": [
            min(r["type1_skew_0.05"] for r in normal_rows),
            max(r["type1_skew_0.05"] for r in normal_rows)],
        "normal_size_range_kurt_0.05": [
            min(r["type1_kurt_0.05"] for r in normal_rows),
            max(r["type1_kurt_0.05"] for r in normal_rows)],
        "p2_kurtosis_bias_by_n": {
            str(r["n"]): {"mean_z": r["mean_z"], "sd_z": r["sd_z"],
                          "size_two_sided": r["type1_kurt_0.05"]}
            for r in p2},
        "b2_finite_mean_max_deviation_in_se": max(
            abs(r["b2_bias_in_se"]) for r in normal_rows),
        "z_finite_mean_max_deviation_in_se": max(
            abs(r["mean_z"] - r["predicted_mean_z"]) / r["se_mean_z"]
            for r in normal_rows),
        "p10n20_kurtosis_sizes": {
            "naive_two_sided": [r["type1_kurt_0.05"] for r in normal_rows
                                if r["p"] == 10 and r["n"] == 20][0],
            "upper_tail": [r["type1_kurt_upper_0.05"] for r in normal_rows
                           if r["p"] == 10 and r["n"] == 20][0],
            "centred_two_sided": [r["type1_kurt_centred_0.05"]
                                  for r in normal_rows
                                  if r["p"] == 10 and r["n"] == 20][0],
        },
        "n200_sd_z_range": [
            min(r["sd_z"] for r in large), max(r["sd_z"] for r in large)],
        "t_beta2_ratio_at_n1000": {
            "{:.0f}".format(r["df"]): r["ratio"]
            for r in limit_rows if r["n"] == 1000},
    }
    # Power at n = 200, p = 2, labelled by parent.
    power = {}
    for r in nonnormal_rows:
        if r["p"] != 2 or r["n"] != 200:
            continue
        key = PARENT_KEY_LABELS[parent_key(r)]
        power[key] = {"skew": r["power_skew_0.05"],
                      "kurtosis": r["power_kurt_0.05"]}
    headline["nonnormal_power_at_n200_p2"] = power

    payload = {
        "meta": {
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "platform": platform.platform(),
            "N_MC": N_MC,
            "seed": args.seed,
            "quick": bool(args.quick),
            "ks_threshold": KS_THRESHOLD,
            "runtimes_seconds": round(time.time() - started, 1),
            "script": "extensions/mardia/study_mardia_null.py",
        },
        "normal_calibration": normal_rows,
        "nonnormal_alternatives": nonnormal_rows,
        "t_limit_convergence": limit_rows,
        "checks": checks,
        "headline": headline,
        "figures": figures,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # A smoke run must not clobber the canonical full-run artifact.
    out = RESULTS_DIR / ("mardia_null_results_quick.json" if args.quick
                         else "mardia_null_results.json")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nwrote", out.relative_to(PACKAGE_ROOT))
    print("elapsed: {:.1f} s".format(time.time() - started))

    print("\nHeadline")
    print("  size at 5%: skewness [{:.3f}, {:.3f}], kurtosis [{:.3f}, {:.3f}]"
          .format(*headline["normal_size_range_skew_0.05"],
                  *headline["normal_size_range_kurt_0.05"]))
    print("  max |E[b2] - p(p+2)(n-1)/(n+1)| = {:.2f} MC SE"
          .format(headline["b2_finite_mean_max_deviation_in_se"]))
    print("  sd(Z_K) at n = 200: [{:.3f}, {:.3f}]"
          .format(*headline["n200_sd_z_range"]))
    print("  E[b2]/beta2 at n = 1000 for t: {}".format(
        {k: round(v, 3) for k, v in headline["t_beta2_ratio_at_n1000"].items()}))
    print("  power at n = 200, p = 2:")
    for key, value in headline["nonnormal_power_at_n200_p2"].items():
        print("    {:<7s} skew = {:.3f}  kurtosis = {:.3f}".format(
            key, value["skew"], value["kurtosis"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
