#!/usr/bin/env python3
"""Generate ``rev_results_t14_study3.json``: the Study 3 Monte Carlo precision sweep
extended to the distance-correlation test.

Study 3 estimates, for each test, the coefficient of variation of the empirical Type I
error rate across independent runs, as a function of the replication count.  This script
produces that curve for T14 at ``d = 50``, the dimension at which its reference law is
valid, and reports the smallest grid point whose CV falls below 0.10.

Run from the package root:

    python3 code/gen_t14_study3.py

Writes ``rev_results_t14_study3.json`` to the package root.  Consumed by
``code/verify_manuscript.py`` and reported in Section 5.3 of the manuscript.
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
N_SAMPLE = 50
D = 50
RUNS = 30
GRID = (100, 500, 1000, 1500, 2000, 3000, 5000, 10_000)
TARGET_CV = 0.10


def main():
    """Run the precision sweep and write the results file."""
    t0 = time.time()
    cells = {}
    for N in GRID:
        alphas = []
        for run in range(RUNS):
            rng = np.random.default_rng(SEED + 1009 * run)
            sv = core.mc_dcor_t(N_SAMPLE, N, d=D, dep="indep", rng=rng)
            sv = sv[np.isfinite(sv)]
            ev = core._evaluate(sv, "T14_dcor_t", alpha=0.05, n=N_SAMPLE,
                                ref_args=core.dcor_t_ref_args(N_SAMPLE))
            alphas.append(ev["type1"])
        m, sd = float(np.mean(alphas)), float(np.std(alphas, ddof=1))
        cells[str(N)] = {"mean": m, "std": sd, "cv": sd / m if m > 0 else float("nan")}
        print(f"  N={N:<6d} mean={m:.4f} sd={sd:.4f} CV={cells[str(N)]['cv']:.4f}")
    n_star = next((N for N in GRID if cells[str(N)]["cv"] < TARGET_CV), None)
    doc = {"meta": {"seed": SEED, "n": N_SAMPLE, "d": D, "runs": RUNS,
                    "grid": list(GRID), "wall_clock_s": round(time.time() - t0, 1),
                    "script": "code/gen_t14_study3.py"},
           "cells": cells, "n_star_mc": n_star,
           "note": ("Study 3 extended to T14 at d = 50, the dimension at which its "
                    "reference is valid; n_star_mc is the smallest grid point whose CV "
                    "falls below 0.10")}
    out = os.path.join(ROOT, "rev_results_t14_study3.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    print(f"wrote {out} (n*_MC = {n_star}, {doc['meta']['wall_clock_s']}s)")


if __name__ == "__main__":
    main()
