#!/usr/bin/env python3
"""Generate ``rev_results_dcor_local.json``: local power of T14 against a signal
confined to ``k_signal`` of ``d`` coordinates.

The distance-correlation statistic aggregates all coordinates with equal weight, so
whether the dependence is spread across coordinates or concentrated in a few of them
matters to its power.  This script measures that directly by placing the quadratic
signal ``Y_j = a (X_j^2 - 1) + noise`` in exactly ``k_signal`` coordinates, with
``a = rho``, and leaving the others independent, at a fixed amplitude.

Run from the package root:

    python3 code/gen_dcor_local.py

Writes ``rev_results_dcor_local.json`` to the package root.  Consumed by
``code/verify_manuscript.py`` and reported in Section 5.1.1b of the manuscript.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 2024
N = 10_000
AMPLITUDE = 1.0    # signal amplitude a; Var(a(X^2-1)) = 2a^2 against unit noise


def local_power(n, d, k_signal, N=N, a=AMPLITUDE, seed=SEED):
    """Rejection rate of T14 when the quadratic signal occupies ``k_signal`` coordinates.

    Args:
        n: sample size per replication.
        d: ambient dimension of both vectors.
        k_signal: number of coordinates of ``Y`` that carry the signal.
        N: Monte Carlo replications.
        a: signal amplitude; the signal has variance 2a^2 against unit noise.
        seed: random seed.

    Returns:
        dict: the design and the measured power.
    """
    rng = np.random.default_rng(seed)
    out = np.empty(N)
    for i in range(N):
        X = rng.standard_normal((n, d))
        Y = rng.standard_normal((n, d))
        for j in range(k_signal):
            Y[:, j] = a * (X[:, j] ** 2 - 1.0) + Y[:, j]
        out[i] = core.dcor_t(X, Y)[0]
    crit = core.critical_value("T14_dcor_t", 0.05, ref_args=core.dcor_t_ref_args(n))[0]
    return {"power": float(np.mean(out > crit)), "crit": float(crit)}


def main():
    """Run the local-power grid and write the results file."""
    t0 = time.time()
    rows = []
    for d in (2, 10, 50):
        for k_signal in (1, d):
            for n in (20, 100, 200):
                r = local_power(n, d, k_signal)
                rows.append({"d": d, "k_signal": k_signal, "n": n,
                             "power": r["power"], "critical_value": r["crit"]})
                print(f"  d={d:<3} k_signal={k_signal:<3} n={n:<4} "
                      f"power={r['power']:.4f}")
    doc = {"meta": {"seed": SEED, "N": N, "amplitude": AMPLITUDE,
                    "wall_clock_s": round(time.time() - t0, 1),
                    "script": "code/gen_dcor_local.py",
                    "note": ("local power of T14 against a quadratic signal confined to "
                             "k_signal of the d coordinates, at amplitude a = 1.0 so that the "
                             "signal variance 2a^2 = 2 is twice the noise variance; sparse "
                             "k_signal=1 is a hard alternative and is reported as such")},
           "rows": rows}
    out = os.path.join(ROOT, "rev_results_dcor_local.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    print(f"wrote {out} ({len(rows)} rows, {doc['meta']['wall_clock_s']}s)")


if __name__ == "__main__":
    main()
