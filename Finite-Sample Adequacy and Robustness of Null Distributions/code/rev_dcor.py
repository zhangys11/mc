#!/usr/bin/env python3
"""T14: the Szekely-Rizzo distance-correlation t-test as a fourteenth panel member.

Referee 2's second major comment named the distance-correlation test of independence
(doi:10.1016/j.jmva.2013.02.012) as a representative procedure for testing whether
two sets of variables are independent.  This script integrates it into the framework
on exactly the same terms as every other test: the Study 1 convergence grid at
N = 10,000, a null-calibration diagnostic, and a power measurement, all at alpha in
{0.01, 0.05, 0.10}.  It also probes the two configurations that matter specifically
for this test:

* **dimension.** The published asymptotic theory is a high-dimensional limit, so we
  report d = 2, 10 and 50 separately instead of pooling them; d = 2 is labeled a
  stress case rather than assumed to satisfy the approximation.
* **within-vector correlation.** Correlated coordinates inside one vector are a
  nuisance configuration, not a violation of independence between the two vectors.
  We verify that the test's size is unaffected by eta, which distinguishes it from
  a genuine dependence alternative.

The statistic is verified against the reference implementation ``energy::dcorT``
(R package energy 1.7-12) by ``verify_dcor_against_R.py``; the agreement is 3e-11.

Run from the package root:  python3 code/rev_dcor.py [--quick]
Writes rev_results_dcor.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time

import numpy as np

import core

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "rev_results_dcor.json")
SEED = 2024
NS = [20, 50, 100, 200, 500]
DS = [2, 10, 50]
ALPHAS = [0.01, 0.05, 0.10]


def main() -> None:
    """Run the T14 study: convergence by dimension, nuisance check, and power.

    Returns:
        None.  Writes ``rev_results_dcor.json``.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="reduced budget smoke run")
    args = ap.parse_args()
    N = 500 if args.quick else 10_000
    ns = NS[:3] if args.quick else NS
    ds = [2, 10] if args.quick else DS
    t0 = time.time()
    core._selftest(quick=True)

    study1 = {}
    for d in ds:
        for n in ns:
            row = core.run_convergence(
                "T14_dcor_t", [n], N=N, alphas=ALPHAS,
                gen=lambda n_, N_, n=n, d=d: core.mc_dcor_t(
                    n_, N_, d=d, dep="indep",
                    rng=np.random.default_rng(SEED + 100 * n + d)))
            study1[f"d={d}|n={n}"] = row
            print(f"  d={d:<3d} n={n:<4d} KS={row['ks'][0]:.4f} "
                  f"a05={row['type1']['0.05'][0]:.4f} a01={row['type1']['0.01'][0]:.4f} "
                  f"a10={row['type1']['0.10'][0]:.4f}", flush=True)

    # nuisance: correlation within each vector must not affect size
    nuisance = {}
    for eta in (0.0, 0.5):
        for n in (ns[0], ns[len(ns) // 2] if len(ns) > 1 else ns[0]):
            sv = core.mc_dcor_t(n, N, d=ds[-1], dep="indep", eta=eta,
                                rng=np.random.default_rng(SEED + 7 * n))
            ev = core._evaluate(sv, "T14_dcor_t", alpha=0.05, n=n)
            nuisance[f"eta={eta}|n={n}"] = ev
            print(f"  nuisance eta={eta} n={n}: a05={ev['type1']:.4f}", flush=True)

    # power
    power = {}
    for dep in ("linear", "nonlinear"):
        for n in (ns[0], ns[len(ns) // 2] if len(ns) > 1 else ns[0]):
            for rho in (0.3, 0.6):
                sv = core.mc_dcor_t(n, N, d=ds[-1], dep=dep, rho=rho,
                                    rng=np.random.default_rng(SEED + 31 * n))
                ev = core._evaluate(sv, "T14_dcor_t", alpha=0.05, n=n)
                power[f"{dep}|rho={rho}|n={n}"] = ev
                print(f"  power {dep} rho={rho} n={n}: {ev['type1']:.4f}", flush=True)

    out = {
        "meta": {"seed": SEED, "N": N, "ns": ns, "dims": ds, "alphas": ALPHAS,
                 "numpy": np.__version__, "python": platform.python_version(),
                 "script": "code/rev_dcor.py", "quick": bool(args.quick),
                 "wall_clock_s": round(time.time() - t0, 1)},
        "reference": ("Student t with n(n-3)/2-1 degrees of freedom, upper tail; "
                      "statistic verified against energy::dcorT (energy 1.7-12) to 3e-11"),
        "study1": study1, "nuisance_within_vector": nuisance, "power": power,
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {OUT} in {out['meta']['wall_clock_s']}s")


if __name__ == "__main__":
    main()
