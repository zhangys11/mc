#!/usr/bin/env python3
"""Generate the class diagram shown in the repository README.

The diagram shows ``McBase`` and the three public class modules.  Class names
are read from each module's ``__all__`` so the figure stays in sync when a class
is added.  Run from the repository root::

    python3 doc/make_class_diagram.py

It writes ``class diagram.jpg`` in the repository root (the file referenced by
``README.md``).  Only matplotlib is required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mc import distributions, experiments, samplings  # noqa: E402

OUT = ROOT / "class diagram.jpg"

MODULES = [
    ("mc.distributions", distributions),
    ("mc.samplings", samplings),
    ("mc.experiments", experiments),
]

EDGE = "#000000"
FONT = "DejaVu Sans Mono"


def draw_box(ax, x0, y0, width, height, text, fontsize, linestyle="-"):
    ax.add_patch(Rectangle((x0, y0), width, height, fill=False,
                           edgecolor=EDGE, linewidth=1.1, linestyle=linestyle))
    ax.text(x0 + width / 2.0, y0 + height / 2.0, text,
            ha="center", va="center", fontsize=fontsize,
            family=FONT, color=EDGE)


def main():
    fig, ax = plt.subplots(figsize=(12.0, 7.2))
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(0.0, 100.0)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Root class.
    draw_box(ax, 36.0, 89.5, 28.0, 7.5, "mc._mcbase.McBase", fontsize=12)

    # Connector: root -> bus -> the three modules.
    ax.plot([50.0, 50.0], [89.5, 85.5], color=EDGE, lw=1.0)
    ax.plot([17.0, 83.0], [85.5, 85.5], color=EDGE, lw=1.0)
    for cx in (17.0, 50.0, 83.0):
        ax.plot([cx, cx], [85.5, 80.0], color=EDGE, lw=1.0)

    # Module boxes.
    box_width = 27.0
    box_top = 80.0
    box_bottom = 12.0
    centers = (17.0, 50.0, 83.0)
    for center, (module_name, module) in zip(centers, MODULES):
        x0 = center - box_width / 2.0
        ax.add_patch(Rectangle((x0, box_bottom), box_width,
                               box_top - box_bottom, fill=False,
                               edgecolor=EDGE, linewidth=1.1))
        # Dotted label for the module itself.
        ax.add_patch(Rectangle((center - 10.5, 70.5), 21.0, 7.0, fill=False,
                               edgecolor=EDGE, linewidth=1.0, linestyle=":"))
        ax.text(center, 74.0, module_name, ha="center", va="center",
                fontsize=11, family=FONT, color=EDGE)

        names = list(module.__all__)
        top = 66.0
        step = min(3.9, (top - 16.0) / max(1, len(names) - 1))
        height = (len(names) - 1) * step
        start = top - max(0.0, (top - 16.0 - height) / 2.0)
        for index, name in enumerate(names):
            ax.text(center, start - index * step, name, ha="center",
                    va="center", fontsize=10.5, family=FONT, color=EDGE)

    fig.savefig(OUT, dpi=100, bbox_inches="tight", facecolor="white",
                pil_kwargs={"quality": 95})
    plt.close(fig)
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
