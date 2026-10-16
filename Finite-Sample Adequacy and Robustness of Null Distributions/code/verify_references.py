#!/usr/bin/env python3
"""Bidirectional citation check for the manuscript.

Every reference-list entry must be cited in the text, every in-text citation must appear
in the list, and the list must be in alphabetical order.  The check is deliberately
conservative about matching: a citation counts only when a surname is followed by the
matching year within a short window, which tolerates the "Author (year)" and
"(Author, year)" styles but not a bare mention of a name.

Run from the package root:

    python3 code/verify_references.py
"""
from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Tokens that look like a citation to the pattern matcher but are not author names.
KNOWN_NON_CITATIONS = {"ks", "cvm", "anova", "icc", "hmisc", "d", "t", "f", "q", "mt", "cv", "bfk",
                       "table", "figure", "section", "equation", "panel", "n", "r", "p", "see"}



def main() -> int:
    """Run the check and report.

    Returns:
        int: 0 when the reference apparatus is consistent, 1 otherwise.
    """
    path = os.path.join(ROOT, "draft.md")
    if not os.path.exists(path):
        print("reference check SKIPPED: draft.md is submission material and is not "
              "distributed with this package")
        return 0
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    if "## References" not in text:
        print("reference check FAILED: no reference list found")
        return 1
    cut = text.index("## References")
    body, refsec = text[:cut], text[cut:]

    entries = []
    all_surnames = set()
    for line in refsec.split("\n"):
        line = line.strip()
        if "," not in line or "(" not in line:
            continue
        surname = line[:line.index(",")].strip()          # text before the first comma
        rest = line[line.index(","):]
        ym = re.search(r"\((\d{4})\)", rest)
        if not surname or not ym:
            continue
        if not re.match(r"[A-Z\u00c0-\u017f]", surname):
            continue                                      # not a reference entry
        entries.append((surname, ym.group(1), line))
        # every surname in the entry, so a citation to a co-author resolves as well
        head = line[:line.index("(")]
        for nm in re.finditer(r"([A-Z\u00c0-\u017f][\w\'\u2019-]+),", head):
            all_surnames.add(nm.group(1).lower().replace("-", ""))
        all_surnames.add(surname.lower().replace("-", ""))
    problems = []
    # forward: every entry is cited
    for surname, year, line in entries:
        cited = any(year in body[hit.start():hit.start() + 170]
                    for hit in re.finditer(re.escape(surname), body))
        if not cited:
            problems.append(f"reference not cited in text: {surname} ({year})")
    # backward: every in-text citation must resolve to the list.  A citation group is
    # split on ";" into works and then at each year, so that "A et al., 1981; B, 2020"
    # yields two pairs, and a phrase such as "a Bartlett kernel and the rule of
    # Schwert, 1989" yields only the one work it actually cites.
    cited_pairs = set()
    for grp in re.finditer(r"\(\s*[A-Z\u00c0-\u017f][^()]{0,240}?\)", body):
        inner = grp.group(0)
        if "=" in inner or "\\" in inner or "$" in inner:
            continue                              # a formula, not a citation group
        for work in re.split(r";", inner.strip("() ")):
            for chunk in re.finditer(r"([^,;()]{1,120}?),\s*(\d{4})[a-z]?", work):
                lead = chunk.group(1)
                names = re.findall(r"([A-Z\u00c0-\u017f][\w\'\u2019-]+)", lead)
                if names:
                    # the work cited is the name closest to the year
                    cited_pairs.add((names[-1].lower().rstrip("'\u2019s").replace("-", ""), chunk.group(2)))
    # narrative "Author (year)" and "Author and Author (year)" forms
    for m in re.finditer(r"([A-Z\u00c0-\u017f][\w\'\u2019-]+)(?:\s+(?:and|&)\s+[A-Z][\w-]+)?\s*\((\d{4})\)", body):
        cited_pairs.add((m.group(1).lower().rstrip("'\u2019s").replace("-", ""), m.group(2)))
    for nm, yr in sorted(cited_pairs, key=lambda t: (t[0], t[1] or "")):
        if nm.replace("-", "") not in all_surnames and nm not in KNOWN_NON_CITATIONS:
            problems.append(f"citation missing from the list: {nm} ({yr})")
    # alphabetical
    keys = [s.lower() for s, _, _ in entries]
    if keys != sorted(keys):
        problems.append("reference list is not in alphabetical order")
    if problems:
        print(f"reference check FAILED ({len(problems)} problems)")
        for p in sorted(set(problems)):
            print("  -", p)
        return 1
    print(f"reference check PASSED: {len(entries)} entries, all cited, "
          f"all citations listed, alphabetical order")
    return 0


if __name__ == "__main__":
    sys.exit(main())
