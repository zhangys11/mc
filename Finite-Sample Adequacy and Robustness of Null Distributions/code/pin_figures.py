#!/usr/bin/env python3
"""Move each figure's numbered caption inside its LaTeX float.

Pandoc puts an image into a ``figure`` environment and uses the Markdown image's
alternative text as the ``\\caption``. The real numbered caption then follows the float as
an ordinary italic paragraph, so LaTeX may float the image away from it and the document
ends up with two captions for one figure (a short unnumbered one inside the float and the
real one outside). In the previous build this put Figure 4's caption on page 33 and its
image on page 34.

This pass rewrites the generated LaTeX so that each float carries the real numbered
caption, and the duplicate outside paragraph is removed. It runs on the .tex file between
pandoc and xelatex.

Usage:
    python3 code/pin_figures.py path/to/file.tex
"""
from __future__ import annotations

import re
import sys

# A figure float: \begin{figure}...\end{figure}, plus the caption paragraph that follows.
FLOAT = re.compile(
    r"\\begin\{figure\}(?P<body>.*?)\\end\{figure\}"
    r"\s*\n\s*\\emph\{(?P<caption>Figure\s+\d+.*?)\}\s*\n",
    re.DOTALL,
)
CAPTION_IN_FLOAT = re.compile(r"\\caption\{.*?\}\s*$", re.DOTALL)


def fix(text: str) -> tuple[str, int]:
    """Return the rewritten LaTeX and the number of figures corrected.

    Args:
        text: the LaTeX document produced by pandoc.

    Returns:
        tuple: the new document text and the count of floats that were rewritten.
    """
    count = 0

    def repl(match: re.Match) -> str:
        nonlocal count
        body = match.group("body")
        caption = " ".join(match.group("caption").split())
        # LaTeX numbers the float itself, so drop the number already present in the text;
        # otherwise the caption reads "Figure 4: Figure 4: ...".
        caption = re.sub(r"^Figure\s+\d+\s*:\s*", "", caption)
        # drop pandoc's alt-text caption and the source-code comments it leaves behind
        body = CAPTION_IN_FLOAT.sub("", body).rstrip()
        count += 1
        return f"\\begin{{figure}}[H]\n{body}\n\\caption{{{caption}}}\n\\end{{figure}}\n"

    return FLOAT.sub(repl, text), count


def main(argv: list[str]) -> int:
    """Rewrite the file in place.

    Args:
        argv: ``[script, path]``.

    Returns:
        int: 0 on success, 1 when the argument is missing.
    """
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[-1])
        return 1
    path = argv[1]
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    new, n = fix(text)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(new)
    print(f"pinned {n} figures to their captions in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
