#!/usr/bin/env python3
"""Automated cross-reference audit for the manuscript, the supplement and the letters.

Every figure and table reference must resolve, and the caption order must match the
numbering.  This is the check that the manual passes kept missing: a figure can be moved
or a table renumbered while some reference elsewhere still points at the old number, and
the manuscript then cites a table it does not contain.

Run from the package root:

    python3 code/verify_crossrefs.py
"""
from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name):
    """Read a package file.

    Args:
        name: path relative to the package root.

    Returns:
        str: the file contents.
    """
    with open(os.path.join(ROOT, name), encoding="utf-8") as fh:
        return fh.read()


def audit(doc, label, caps_fig, caps_tab, extra_tabs=0, max_fig=None, max_tab=None):
    """Check that every reference in ``doc`` resolves to a caption in order.

    Args:
        doc: document text.
        label: name used in the report.
        caps_fig: figure numbers as they appear, in document order.
        caps_tab: table numbers as they appear, in document order.
        extra_tabs: number of tables that live in another file and may be cited.

    Returns:
        list of str: one message per problem found.
    """
    problems = []
    n_fig = len(caps_fig) if max_fig is None else max_fig
    n_tab = (len(caps_tab) + extra_tabs) if max_tab is None else max_tab
    if caps_fig and caps_fig != list(range(1, len(caps_fig) + 1)):
        problems.append(f"{label}: figure captions not in order: {caps_fig}")
    if caps_tab != list(range(1, len(caps_tab) + 1)):
        problems.append(f"{label}: table captions not in order: {caps_tab}")
    for m in re.finditer(r"Figures? (\d+)(?:\s+and\s+(\d+))?", doc):
        for g in (1, 2):
            if m.group(g) and int(m.group(g)) > n_fig:
                line = doc[:m.start()].count("\n") + 1
                problems.append(f"{label}: line {line} cites Figure {m.group(g)} but only "
                                f"{n_fig} figures exist")
    for m in re.finditer(r"Tables? (\d+)(?:\s+and\s+(\d+))?", doc):
        for g in (1, 2):
            if m.group(g) and int(m.group(g)) > n_tab:
                line = doc[:m.start()].count("\n") + 1
                problems.append(f"{label}: line {line} cites Table {m.group(g)} but only "
                                f"{n_tab} tables exist")
    # every figure must be cited at least once outside its own caption
    if caps_fig:
        body = re.sub(r"^\*Figure \d+:.*$", "", doc, flags=re.M)
        for n in range(1, n_fig + 1):
            if not re.search(rf"Figures? {n}\b", body):
                problems.append(f"{label}: Figure {n} is never cited in the text")
    return problems


def main():
    """Run the audit and report.

    The manuscript, the response letter and the supplement are submission materials and
    are not distributed with the reproduction package, so this audit is skipped with an
    explicit message when they are absent rather than failing.  What remains checkable in
    the package alone is covered by ``verify_manuscript.py``'s result-file checks,
    ``docs/verify_bartlett.py`` and ``docs/make_manifest.py --check``.

    Returns:
        int: 0 if no problems or if the text is not present, 1 otherwise.
    """
    if not os.path.exists(os.path.join(ROOT, "draft.md")):
        print("cross-reference audit SKIPPED: draft.md is not part of this package "
              "(submission material). Run it from the submission directory instead.")
        return 0
    draft = load("draft.md")
    supp = load("supplementary_material.md")
    letter = load("response_letter.md")

    figs = [int(x) for x in re.findall(r"^\*Figure (\d+):", draft, re.M)]
    tabs = [int(x) for x in re.findall(r"^\*\*Table (\d+):", draft, re.M)]
    supp_tabs = len(re.findall(r"^## S\d+\.", supp, re.M))

    problems = []
    problems += audit(draft, "draft.md", figs, tabs, extra_tabs=supp_tabs)
    # the letter cites manuscript figures and tables freely, so only check for
    # references beyond what exists anywhere
    # The letter may cite any figure or table of the manuscript, so its bounds are the
    # manuscript's, and it carries no captions of its own.
    letter_figs = [int(x) for x in re.findall(r"^\*Figure (\d+):", letter, re.M)]
    problems += audit(letter, "response_letter.md", letter_figs, [], max_fig=figs[-1],
                      max_tab=tabs[-1])

    # named figures must sit in the section that discusses them
    secs = [(m.start(), m.group(0)) for m in
            re.finditer(r"^#{2,3} 5\.\d.*$|^## 6\..*$", draft, re.M)]
    for i, (pos, name) in enumerate(secs):
        end = secs[i + 1][0] if i + 1 < len(secs) else len(draft)
        body = draft[pos:end]
        has_fig = re.findall(r"^\*Figure (\d+):", body, re.M)
        if not has_fig and not name.strip().startswith("## 6"):
            problems.append(f"draft.md: {name.strip()[:48]} contains no figure")

    if problems:
        print(f"cross-reference audit FAILED ({len(problems)} problems)")
        for p in problems:
            print("  -", p)
        return 1
    print(f"cross-reference audit PASSED: {len(figs)} figures, {len(tabs)} tables, "
          f"{supp_tabs} supplement sections, all references resolve and are cited")
    return 0


if __name__ == "__main__":
    sys.exit(main())
