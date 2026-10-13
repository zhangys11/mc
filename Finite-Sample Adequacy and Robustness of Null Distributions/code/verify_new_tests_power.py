#!/usr/bin/env python3
"""Independent power measurement for the two new continuous-statistic tests (T11, T12).

Referee 1's objection was that a calibration study without a power counterpart cannot
support a recommendation. Study 4 (`rev_power.py`) covers the seven classical tests;
this script supplies the missing counterpart for the two new tests, using the SAME
nominal alpha, the SAME Monte Carlo budget and the SAME Wilson-interval convention,
and reporting nominal-alpha and size-adjusted power side by side.

Run:  python3 verify_new_tests_power.py
Writes: rev_results_newtests_power.json
"""
from __future__ import annotations

import json
import platform
import time

import numpy as np

import core

N = 10_000
SEED = 2024
ALPHA = 0.05
NS = [20, 50, 100, 200]


def wilson(k, n, z=1.959963984540054):
    """Wilson score interval for a binomial count.

    Args:
        k: number of successes.
        n: number of trials; ``n = 0`` returns ``(nan, nan)``.
        z: normal quantile (default 1.959963984540054 for 95%).

    Returns:
        tuple ``(lo, hi)`` clipped to [0, 1], with the same convention as
        ``experiments.wilson_ci`` and ``core.wilson_ci``.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rate_at(stats, crit):
    """Rejection rate above a critical value, with its Wilson interval.

    Args:
        stats: 1-D array of statistics.
        crit: critical value (upper tail).

    Returns:
        tuple ``(rate, (lo, hi))`` where ``rate = mean(stats > crit)`` and the
        interval is the 95% Wilson interval for that count.
    """
    k = int(np.sum(stats > crit))
    return k / len(stats), wilson(k, len(stats))


def main():
    """Run the power grid for T11 and T12 and write the results file.

    For every sample size in ``NS`` the script measures the size under the null
    (``alpha_hat_H0``), recalibrates the critical value so that the size is
    exactly 0.05, and reports nominal-alpha and size-adjusted power with Wilson
    intervals for the alternative designs of each test.

    Returns:
        None. Writes ``rev_results_newtests_power.json``.
    """
    rng = np.random.default_rng(SEED)
    out = {"meta": {"seed": SEED, "N": N, "alpha": ALPHA, "ns": NS,
                    "numpy": np.__version__, "python": platform.python_version(),
                    "command": "python3 verify_new_tests_power.py"},
           "T11_kpss": {}, "T12_hoeffding_d": {}}
    t0 = time.time()

    # ---------------- T11: level stationarity vs unit root ----------------
    crit11, spec11 = core.critical_value("T11_kpss", ALPHA)
    for n in NS:
        # H0: level-stationary iid noise. H1: random walk (unit root).
        s0 = core.mc_kpss(n, N, kind="iid", rng=np.random.default_rng(SEED + n))
        s1 = core.mc_kpss(n, N, kind="rw", rng=np.random.default_rng(SEED + 1000 + n))
        # size-adjusted threshold: empirical 0.95 quantile of the H0 sample
        adj = float(np.quantile(s0, 1 - ALPHA))
        a0, ci0 = rate_at(s0, crit11)
        p_nom, p_nom_ci = rate_at(s1, crit11)
        p_adj, p_adj_ci = rate_at(s1, adj)
        out["T11_kpss"][f"n={n}"] = {
            "critical_theoretical": crit11, "tail_spec": spec11,
            "alpha_hat_H0": a0, "alpha_hat_H0_ci": ci0,
            "critical_size_adjusted": adj,
            "power_nominal": p_nom, "power_nominal_ci": p_nom_ci,
            "power_size_adjusted": p_adj, "power_size_adjusted_ci": p_adj_ci,
        }
        print(f"T11 n={n:<4d} ahat={a0:.3f} crit*={adj:.4f} "
              f"power_nom={p_nom:.3f} [{p_nom_ci[0]:.3f},{p_nom_ci[1]:.3f}] "
              f"power_adj={p_adj:.3f} [{p_adj_ci[0]:.3f},{p_adj_ci[1]:.3f}]", flush=True)

    # ---------------- T12: independence vs dependence ----------------
    crit12, spec12 = core.critical_value("T12_hoeffding_d", ALPHA)
    for n in NS:
        s0 = core.mc_hoeffding_d(n, N, dep="indep", rng=np.random.default_rng(SEED + n))
        adj = float(np.quantile(s0, 1 - ALPHA))
        a0, ci0 = rate_at(s0, crit12)
        row = {"critical_theoretical": crit12, "tail_spec": spec12,
               "alpha_hat_H0": a0, "alpha_hat_H0_ci": ci0,
               "critical_size_adjusted": adj}
        for dep, label in [("linear", "power_linear"), ("quad", "power_quadratic")]:
            try:
                s1 = core.mc_hoeffding_d(n, N, dep=dep, rho=0.5,
                                         rng=np.random.default_rng(SEED + 2000 + n))
            except TypeError:
                s1 = core.mc_hoeffding_d(n, N, dep=dep,
                                         rng=np.random.default_rng(SEED + 2000 + n))
            p_nom, p_nom_ci = rate_at(s1, crit12)
            p_adj, p_adj_ci = rate_at(s1, adj)
            row[label] = p_nom
            row[label + "_ci"] = p_nom_ci
            row[label.replace("power", "power_size_adjusted")] = p_adj
            row[label.replace("power", "power_size_adjusted") + "_ci"] = p_adj_ci
        out["T12_hoeffding_d"][f"n={n}"] = row
        print(f"T12 n={n:<4d} ahat={a0:.3f} linear={row['power_linear']:.3f} "
              f"quadratic={row['power_quadratic']:.3f}", flush=True)

    out["meta"]["wall_clock_s"] = round(time.time() - t0, 1)
    with open("rev_results_newtests_power.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote rev_results_newtests_power.json in {out['meta']['wall_clock_s']}s")


if __name__ == "__main__":
    main()
