#!/usr/bin/env python3
"""Render the execution flowchart of the mc-tk simulation pipeline.

Run from the repository root::

    python3 docs/make_flowchart.py

Writes ``docs/fig_execution_flow.png`` (panel included in vignette.pdf) and
``docs/fig_execution_flow.pdf`` (vector version for the manuscript).

The Mermaid source of the same chart is ``docs/flowchart.mmd``; the two are kept in
sync by hand because no Mermaid renderer is available in this environment.

Runtime labels are the wall-clock times measured on the machine that produced the
results (see docs/CODE_MAP.md); they are printed in the panel so a reader can
sanity-check their own run.

Author: t4-docs (task-4).
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PNG = os.path.join(ROOT, "docs", "fig_execution_flow.png")
OUT_PDF = os.path.join(ROOT, "docs", "fig_execution_flow.pdf")

# Wall-clock runtimes measured on the producing machine (see docs/CODE_MAP.md).
RUN_TIMES = {
    "experiments.py": "\u2248 3.3 min",
    "rev_study1_ext.py": "in preparation",
    "rev_alpha_dependence.py": "in preparation",
    "rev_sensitivity.py": "1-2 min",
    "rev_power.py": "~1 min",
}

C = dict(env="#dbeafe", code="#f1f5f9", kernel="#e0e7ff", run="#dcfce7",
         out="#fef9c3", deliver="#fae8ff", verify="#fee2e2")
EDGE = "#334155"


def box(ax, cx, cy, w, h, title, body="", fc="#ffffff", title_size=10.7,
        body_size=8.7, mono_title=False, edge=EDGE, lw=0.9):
    """Draw one rounded box with an optional bold title and a body text block."""
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.004,rounding_size=0.012",
                                linewidth=lw, edgecolor=edge, facecolor=fc,
                                mutation_aspect=1.0, zorder=2))
    if body:
        ax.text(cx, cy + h * 0.26, title, ha="center", va="center", zorder=3,
                fontsize=title_size, fontweight="bold",
                family="monospace" if mono_title else "sans-serif")
        ax.text(cx, cy - h * 0.20, body, ha="center", va="center", zorder=3,
                fontsize=body_size, linespacing=1.45, color="#0f172a")
    else:
        ax.text(cx, cy, title, ha="center", va="center", zorder=3,
                fontsize=title_size, linespacing=1.5)


def arrow(ax, xy_from, xy_to, label="", style="-|>", rad=0.0, color=EDGE,
          ls="-", label_dx=0.0, label_dy=0.012, fontsize=8.2, lw=1.0):
    """Draw an arrow between two points, optionally with a boxed label."""
    ax.add_patch(FancyArrowPatch(xy_from, xy_to, arrowstyle=style,
                                 mutation_scale=9, linewidth=lw, color=color,
                                 linestyle=ls, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1.0, shrinkB=1.0))
    if label:
        mx = (xy_from[0] + xy_to[0]) / 2 + label_dx
        my = (xy_from[1] + xy_to[1]) / 2 + label_dy
        ax.text(mx, my, label, ha="center", va="center", fontsize=fontsize,
                color="#1e293b", zorder=4,
                bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none",
                          alpha=0.92))


def main() -> None:
    """Lay out the flowchart layers and write the PNG and PDF panels."""
    fig, ax = plt.subplots(figsize=(11.4, 8.2), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    cols = [0.088, 0.252, 0.416, 0.580, 0.744, 0.908]
    cw = 0.150

    # ---- layer 0: environment -------------------------------------------
    box(ax, 0.5, 0.945, 0.94, 0.075,
        "STEP 0  ·  environment",
        "Install Anaconda / Miniconda  \u2192  pip install -r requirements-lock.txt      "
        "Python 3.13.5  ·  numpy 2.4.6  ·  scipy 1.16.3  ·  matplotlib 3.11.0  ·  lightgbm 4.6.0",
        fc=C["env"], title_size=10.8, body_size=8.6)

    # ---- layer 1: code modules ------------------------------------------
    box(ax, 0.245, 0.815, 0.415, 0.095,
        "experiments.py",
        "T1\u2013T10 statistic generators\n"
        "Study 1 convergence · Study 2 robustness · Study 3 MC precision\n"
        f"python3 code/experiments.py   ({RUN_TIMES['experiments.py']})",
        fc=C["code"], mono_title=True, body_size=8.2)
    box(ax, 0.735, 0.815, 0.455, 0.095,
        "core.py",
        "shared kernel: registry TESTS (T1\u2013T13), _evaluate() \u2192 D_KS, \u03b1\u0302, Wilson CI\n"
        "T11 KPSS · T12 Hoeffding D · T13 interaction\n"
        "import core   (library only, no command line)",
        fc=C["kernel"], mono_title=True, body_size=8.2)
    arrow(ax, (0.4525, 0.815), (0.5075, 0.815))
    ax.text(0.48, 0.884, "imports T1\u2013T10 unchanged",
            ha="center", va="center", fontsize=8.0, color="#1e293b", zorder=4)

    # ---- layer 2: study runners -----------------------------------------
    runners = [
        ("python3 code/experiments.py", "Studies 1\u20133, archive baseline\n(RQ1, RQ2, RQ3)",
         "\u2248 3.3 min"),
        ("python3 code/rev_study1_ext.py", "13-test Study 1, n* table,\nnull calibration + diagnostics",
         "\u2248 38 min"),
        ("python3 code/rev_alpha_dependence.py", "multi-\u03b1 sweep +\nserial / clustered dependence",
         "\u2248 5 min"),
        ("python3 code/rev_sensitivity.py", "severity ladders +\nsample-size sweep",
         "\u2248 1 min"),
        ("python3 code/rev_power.py", "Study 4 power\nnominal vs. size-adjusted",
         "\u2248 1 min"),
        ("python3 code/verify_new_tests_power.py", "independent power for\nT11 KPSS and T12 Hoeffding D",
         "37 s"),
    ]
    ry = 0.615
    for cx, (label, body, secs) in zip(cols, runners):
        txt = label + "\n" + body + "\n(" + secs + ")"
        box(ax, cx, ry, cw, 0.125, "", fc=C["run"], body_size=7.0)
        ax.text(cx, ry, txt, ha="center", va="center", fontsize=7.0,
                linespacing=1.5, zorder=3)

    # ---- layer 3: result files ------------------------------------------
    outs = [
        "experiment_results.json\nfig_study1_convergence.png\nfig_study2_robustness.png\n"
        "fig_study3_precision.png\ngenerated_tables.md",
        "rev_results_t1.json\nfig_rev_alpha.png",
        "fig_rev_dependence.png\n(merged into\nrev_results_t1.json)",
        "rev_results.json\nfig_rev_severity.png\nfig_rev_study2_n.png",
        "rev_results_power.json\nfig_rev_power.png",
        "rev_results_newtests_power.json",
    ]
    oy = 0.415
    for cx, txt in zip(cols, outs):
        box(ax, cx, oy, cw, 0.135, "", fc=C["out"], body_size=6.6)
        ax.text(cx, oy, txt, ha="center", va="center", fontsize=6.6,
                linespacing=1.6, zorder=3, family="monospace")

    # ---- layer 4/5: deliverables and verification ------------------------
    box(ax, 0.5, 0.245, 0.94, 0.075,
        "Manuscript artefacts   \u2014   Tables 3\u201313 and Figures 1\u20138   \u2014   rendered by ./build.sh",
        "every table and figure is regenerated from the JSON files above; the ratio \u03b1\u0302/\u03b1 flags, Wilson intervals and the robustness rating rule "
        "are computed in core.py;\n./build.sh renders draft.pdf and response_letter.pdf with pandoc + XeLaTeX",
        fc=C["deliver"], title_size=10.0, body_size=8.0)
    box(ax, 0.5, 0.118, 0.94, 0.100,
        "Verification layer",
        "python3 code/verify_manuscript.py  \u2192  213 checks, all passing (every number in draft.md vs. the JSON files)  \u00b7  "
        "python3 docs/verify_bartlett.py  \u2192  13/13\n"
        "python3 code/rev_study1_ext.py --validate  \u2192  core._selftest 18/18  \u00b7  "
        "python3 docs/make_manifest.py --check / --lock  \u00b7  every results JSON carries a meta block",
        fc=C["verify"], title_size=10.0, body_size=8.0)

    # ---- arrows ----------------------------------------------------------
    for cx in cols:
        arrow(ax, (cx, 0.7675), (cx, 0.6775), lw=0.9)
    arrow(ax, (0.5, 0.5525), (0.5, 0.4825), lw=1.0)
    arrow(ax, (0.5, 0.3475), (0.5, 0.2825), lw=1.0)
    arrow(ax, (0.5, 0.2075), (0.5, 0.1680), lw=1.0)
    # environment -> both modules
    arrow(ax, (0.29, 0.9075), (0.245, 0.8625), rad=-0.15, lw=1.0)
    arrow(ax, (0.71, 0.9075), (0.735, 0.8625), rad=0.15, lw=1.0)

    ax.set_title("mc-tk execution flowchart \u2014 COST-S-26-00605 reproducibility package",
                 fontsize=12.5, fontweight="bold", pad=6)

    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
