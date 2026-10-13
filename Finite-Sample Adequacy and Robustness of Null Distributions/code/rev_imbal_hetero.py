#!/usr/bin/env python3
"""Imbalance x heteroscedasticity: the interaction the separate ladders cannot isolate.

Referee 1 (comment 3) asked whether the reported robustness patterns survive milder
and more severe violations, and our imbalance and variance-ratio ladders hold the
other factor fixed.  The classical literature, however, locates the danger of
unequal group sizes precisely in its interaction with unequal variances, so this
script crosses the two: for each allocation ratio and each variance vector it
reports the one-way ANOVA F and Kruskal-Wallis rejection rates at nominal 0.05,
with Wilson intervals, at a fixed total sample size.

Design
------
* total n_T = 150 (allocation ratios below are exact for n_T = 150):
  (50,50,50), (37.5,50,62.5) -> scaled to (38,50,62), (30,50,70), (25,50,75), (14,50,86)
* variance vectors, normal parents with equal means: (1,1,1), (1,1,4), (4,1,1),
  (1,4,1), (1,1,9), (9,1,1) -- i.e. the large variance is placed with the smallest,
  the middle and the largest group, which is the comparison the literature flags.
* N = 10,000 replications per cell, seed 2024, common random numbers across cells.

Run from the package root:  python3 code/rev_imbal_hetero.py
Writes rev_results_imbal_hetero.json
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

import rev_sensitivity as S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "rev_results_imbal_hetero.json")
N = 10_000
SEED = 2024
ALPHA = 0.05
N_T = 150
RATIOS = {"1:1:1": (50, 50, 50), "1:1.3:1.6": (38, 50, 62), "1:1.7:2.3": (30, 50, 70),
          "1:2:3": (25, 50, 75), "1:3.6:6.1": (15, 50, 85)}
VARS = {"1,1,1": (1.0, 1.0, 1.0), "1,1,4": (1.0, 1.0, 4.0), "4,1,1": (4.0, 1.0, 1.0),
        "1,4,1": (1.0, 4.0, 1.0), "1,1,9": (1.0, 1.0, 9.0), "9,1,1": (9.0, 1.0, 1.0)}


def main() -> None:
    """Run the crossed allocation x variance design and write the results file.

    Returns:
        None.  Writes ``rev_results_imbal_hetero.json``.
    """
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    crit_f = float(S.__dict__.get("Z95", 1.959963984540054))  # placeholder, replaced below
    from scipy import stats
    k = 3
    cells = {}
    for rname, ns in RATIOS.items():
        for vname, sds in VARS.items():
            groups = [rng.standard_normal((N, ns[j])) * np.sqrt(sds[j]) for j in range(k)]
            F = S.stat_anova(groups)
            H = S.stat_kruskal(groups)
            df1, df2 = k - 1, sum(ns) - k
            crit_f = float(stats.f.ppf(1 - ALPHA, df1, df2))
            crit_h = float(stats.chi2.ppf(1 - ALPHA, k - 1))
            cell = {}
            for label, stat, crit in (("anova", F, crit_f), ("kruskal", H, crit_h)):
                a = float(np.mean(stat > crit))
                lo, hi = S.wilson_ci(a, N)
                cell[label] = {"alpha_hat": a, "ci": [lo, hi],
                               "liberal": bool(a / ALPHA > 1.5),
                               "ci_liberal": bool(lo / ALPHA > 1.5)}
            cells[f"{rname}|{vname}"] = {"ns": list(ns), "variances": list(sds), **cell}
            print(f"  {rname:12s} var={vname:6s} ANOVA={cell['anova']['alpha_hat']:.4f} "
                  f"KW={cell['kruskal']['alpha_hat']:.4f}", flush=True)
    out = {"meta": {"seed": SEED, "N": N, "alpha": ALPHA, "n_total": N_T,
                    "wall_clock_s": round(time.time() - t0, 1),
                    "script": "code/rev_imbal_hetero.py"},
           "critical_values": {"anova_F": crit_f, "kruskal_chi2": float(stats.chi2.ppf(1 - ALPHA, k - 1))},
           "ratios": RATIOS, "variances": VARS, "cells": cells}
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {OUT} in {out['meta']['wall_clock_s']}s")


if __name__ == "__main__":
    main()
