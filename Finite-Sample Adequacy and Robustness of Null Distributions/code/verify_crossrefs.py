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


# --- subject consistency ---------------------------------------------------------------
# A reference can resolve to a real object and still point at the wrong one, which the
# existence checks above cannot see. Each topic below is owned by exactly one object, so a
# sentence that names the topic and a different object is flagged. Only topics with a
# single unambiguous owner are listed; anything that several figures legitimately share is
# left out, which keeps the check free of false positives.
TOPIC_OWNER = [
    # each topic below has exactly one owning object, so naming it beside another object
    # is a mismatch. Topics that several figures share (Study 2 scenarios, nominal level)
    # are deliberately absent: they cannot be attributed uniquely and would produce noise.
    ("coefficient of variation", "Figure 3"),
    ("cv-versus-n", "Figure 3"),
    ("severity ladder", "Figure 4"),
    ("sample sizes at which each criterion", "Table 4"),
    ("baseline-relative change", "Table 7"),
    ("imbalance crossed with heteroscedasticity", "Table 8"),
    ("$\\times$ heteroscedasticity", "Table 8"),
    ("power comparison", "Figure 8"),
]

# Concepts the objects are definitely not about, checked per clause.
WRONG_TOPIC = {
    "Table 6":  ("precision", "replication count", "coefficient of variation"),
    "Table 7":  ("precision", "replication count", "coefficient of variation"),
    "Table 8":  ("precision", "coefficient of variation", "cv-versus-n", "nominal level"),
    "Table 10": ("precision", "coefficient of variation"),
    "Table 11": ("nominal level", "goodness-of-fit"),
    "Figure 3": ("nominal level", "severity ladder", "crossed"),
    "Figure 4": ("nominal level", "sample-size sensitivity"),
    "Figure 5": ("severity ladder", "nominal level"),
    "Figure 6": ("sample-size sensitivity", "severity ladder"),
}


def _clauses(line, label):
    """Yield the clause of ``line`` around each occurrence of ``label``.

    Args:
        line: one line of prose.
        label: the object name to look for, for example ``"Table 8"``.

    Yields:
        tuple: ``(start_index, clause_text)``.
    """
    for hit in re.finditer(r"\b" + re.escape(label) + r"\b", line):
        left = max([line.rfind(ch, 0, hit.start())
                    for ch in (".", "!", "?", ";", ",")] + [-1])
        rights = [r for r in (line.find(ch, hit.end()) for ch in (";", ",")) if r > 0]
        right = min(rights) if rights else len(line)
        yield left, " ".join(line[left + 1:right].split())


def subject_warnings(text):
    """Report lines that name a topic beside an object that does not own it.

    These are advisory, not failures. A line that discusses two objects together is
    perfectly legitimate -- "Table 6 gives the rates ... and Table 7 reports the same rates
    as baseline-relative changes" names a Table 7 topic on a line that cites Table 6 -- so
    a hit means "check this by eye", not "this is wrong". The check still earns its place
    because a reference pointed at the wrong object does show up here, and nothing else in
    this suite can see that.

    Prose only: a Markdown table may mention a table number in a heading, and the rows
    below it can hold any vocabulary at all, so lines containing a pipe are skipped.

    Args:
        text: the manuscript Markdown.

    Returns:
        list: one advisory message per candidate line.
    """
    problems = []
    labels = {owner for _, owner in TOPIC_OWNER} | set(WRONG_TOPIC)
    for line_no, line in enumerate(text.split("\n"), start=1):
        if "|" in line:
            continue
        low = line.lower()
        # Compare against the whole line rather than one clause: the topic and the
        # reference are often in different clauses of the same sentence, as in
        # "... the CV-versus-N curves (Figure 3) and the CV values ... (Table 8)".
        for topic, owner in TOPIC_OWNER:
            if topic not in low:
                continue
            for label in labels:
                if label == owner:
                    continue
                if re.search(r"\b" + re.escape(label.lower()) + r"\b", low):
                    problems.append(
                        f"draft.md:{line_no} '{topic}' named on a line that cites {label}, "
                        f"but it belongs to {owner}")
        for label, forbidden in WRONG_TOPIC.items():
            for _, clause in _clauses(line, label):
                for bad in forbidden:
                    if bad in clause.lower():
                        problems.append(
                            f"draft.md:{line_no} {label} referenced in a clause about "
                            f"'{bad}': '{clause[:80]}'")
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

    # advisory: a reference can resolve and still point at the wrong object
    warnings = subject_warnings(draft)

    if problems:
        print(f"cross-reference audit FAILED ({len(problems)} problems)")
        for p in problems:
            print("  -", p)
        return 1
    print(f"cross-reference audit PASSED: {len(figs)} figures, {len(tabs)} tables, "
          f"{supp_tabs} supplement sections, all references resolve and are cited")
    if warnings:
        print(f"  {len(warnings)} advisory subject warnings (review by eye; a line may "
              f"legitimately discuss two objects):")
        for w in warnings:
            print("    ?", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
