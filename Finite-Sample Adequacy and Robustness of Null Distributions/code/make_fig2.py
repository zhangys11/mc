#!/usr/bin/env python3
"""Regenerate Figure 2 (Study 2 Type I error under assumption violations) with the
legend outside the plotting area.

Referee 2 (minor comment m13) reported that one bar overlapped the legend in the
original Figure 2: the legend sat at ``loc='upper left'`` in exactly the corner
where Bartlett's rate under skewness (0.429) rises above every other bar.  This
script redraws the panel from the archived results with the legend placed above
the axes, and additionally draws Wilson intervals on every bar so that the panel
is consistent with Table 5.

Run from the package root:  python3 code/make_fig2.py
Writes: figures/fig_study2_robustness.png
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCEN = ["baseline", "skewed", "heavy_tail", "hetero_var", "unbalanced", "small_freq"]
SCEN_LABELS = {"baseline": "Baseline", "skewed": "Skewed", "heavy_tail": "Heavy-tailed",
               "hetero_var": "Var. heterog.", "unbalanced": "Unbalanced",
               "small_freq": "Small exp. freq."}
ORDER = ["T5_bartlett", "T6_fligner_killeen", "T10_hotelling", "T1_t_test", "T2_anova",
         "T3_kruskal_wallis", "T4_chisq_gof", "T7_sign_test", "T8_cochran_q", "T9_median_test"]
SHORT = {"T5_bartlett": "Bartlett", "T6_fligner_killeen": "Fligner-Killeen",
         "T10_hotelling": "Hotelling T2", "T1_t_test": "Student t", "T2_anova": "ANOVA F",
         "T3_kruskal_wallis": "Kruskal-Wallis", "T4_chisq_gof": "chi2 GOF",
         "T7_sign_test": "Sign", "T8_cochran_q": "Cochran Q", "T9_median_test": "Median"}
ALPHA = 0.05


def main() -> None:
    """Regenerate Figure 2 in ``figures/`` with the legend outside the axes.

    Returns:
        None.  Writes ``figures/fig_study2_robustness.png``.
    """
    with open(os.path.join(ROOT, "experiment_results.json"), encoding="utf-8") as fh:
        res = json.load(fh)["study2"]
    ci = json.load(open(os.path.join(ROOT, "experiment_results.json"),
                        encoding="utf-8"))["study2_ci"]

    fig, ax = plt.subplots(figsize=(10, 5.4))
    x = range(len(SCEN))
    n_tests = len(ORDER)
    width = 0.8 / n_tests
    for i, tid in enumerate(ORDER):
        xs, ys, lo, hi = [], [], [], []
        for j, sc in enumerate(SCEN):
            v = res[tid].get(sc)
            if v is None:
                continue
            xs.append(j + i * width)
            ys.append(v)
            a, b = ci[tid][sc]
            lo.append(max(0.0, v - a))
            hi.append(max(0.0, b - v))
        ax.bar(xs, ys, width * 0.9, label=SHORT[tid], yerr=[lo, hi],
               capsize=1.5, error_kw={"linewidth": 0.6, "ecolor": "0.25"})
    ax.axhline(ALPHA, color="red", linestyle="--", linewidth=1.0,
               label=r"$\alpha = 0.05$")
    ax.set_ylabel("Empirical Type I error rate")
    # Title above the legend, and the legend above the axes: with the legend in a
    # reserved band the referee's bar/legend overlap cannot recur at any bar height.
    ax.set_title("Study 2: Type I error under assumption violations "
                 r"($n = 50$, $N = 10{,}000$)", pad=30)
    ax.set_xticks([j + width * n_tests / 2 for j in x])
    ax.set_xticklabels([SCEN_LABELS[s] for s in SCEN])
    ax.grid(True, alpha=0.3, axis="y")
    # Legend above the axes: the referee's overlap cannot recur, whatever the bar heights.
    ax.legend(fontsize=7.5, ncol=6, loc="lower center", bbox_to_anchor=(0.5, 1.005),
              frameon=False, columnspacing=1.0, handlelength=1.4)
    fig.tight_layout()
    out = os.path.join(ROOT, "figures", "fig_study2_robustness.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
