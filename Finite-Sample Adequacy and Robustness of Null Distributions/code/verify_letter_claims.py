#!/usr/bin/env python3
"""Adversarial check of the response letter against the working tree.

Every concrete artefact the letter claims to have produced must exist in the
manuscript, the supplementary material or the reproduction package. This is the
guard against the failure mode of claiming a fix that is not on disk, which
happened once during the revision (Figure 2 was asserted as regenerated while the
shipped file was still the original).

Run from the package root:  python3 code/verify_letter_claims.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    """Check each claimed artefact; return 0 if all are present.

    Returns:
        int: 0 when every claim is backed by a file or passage, 1 otherwise.
    """
    if not os.path.exists(os.path.join(ROOT, "response_letter.md")):
        print("letter-claim audit SKIPPED: response_letter.md is not part of this package "
              "(submission material). Run it from the submission directory instead.")
        return 0
    letter = open(os.path.join(ROOT, "response_letter.md"), encoding="utf-8").read()
    draft = open(os.path.join(ROOT, "draft.md"), encoding="utf-8").read()
    supp = open(os.path.join(ROOT, "supplementary_material.md"), encoding="utf-8").read()

    def size(rel, minimum=1000):
        p = os.path.join(ROOT, rel)
        return os.path.exists(p) and os.path.getsize(p) >= minimum

    checks = [
        ("Table 1 positioning table", "**Table 1:" in draft and "Positioning relative" in draft),
        ("prior-work subsection in §1", "Relation to prior comparative" in draft),
        ("§2.1 independence", "### 2.1 The independence assumption" in draft),
        ("independence column in Table 2", "Independence assumption" in draft),
        ("T11 KPSS in the panel", "T11" in draft and "KPSS" in draft),
        ("T12 Hoeffding in the panel", "Hoeffding" in draft),
        ("T13 interaction-difference test in the panel", "Interaction difference" in draft and "interaction-difference test of Welchowski" in draft),
        ("T14 distance correlation in the panel", "Distance correlation $t$" in draft),
        ("§3.4 sensitivity design", "### 3.4 Sensitivity analyses" in draft),
        ("§3.5 dependence and power design", "### 3.5 Dependence scenarios" in draft),
        ("robustness components defined in §3.3", "Definition of the robustness components" in draft and "$A_\\mathrm{dist}$" in draft),
        ("relative n*_alpha criterion", "le 0.2" in draft and "is *relative*" in draft),
        ("Binomial(N, pi) model", "Binomial}(N, \\pi)" in draft),
        ("severity ladders in §5.4", "S2: Violation severity" in draft),
        ("crossed design (Table 10)", "Imbalance crossed with heteroscedasticity" in draft),
        ("dependence results in §5.5", "### 5.5 Dependence" in draft),
        ("power study in §5.6", "### 5.6 Study 4" in draft),
        ("T12 reference diagnostic (Table 5)", "Hoeffding's $D$ (T12) against an independently simulated reference" in draft),
        ("T13 split diagnostic (Supplementary Material §S9)", "## S9." in supp and "§S9" in draft),
        ("T14 dimension finding (§5.1.1b)", "5.1.1b" in draft),
        ("Wilson endpoints in supplement §S4", "§S4" in draft and "## S4." in supp),
        ("crossed grid in supplement §S5", "## S5." in supp),
        ("alpha grid in supplement §S6", "## S6." in supp),
        ("n-sweep grid in supplement §S7", "## S7." in supp),
        ("full positioning table in supplement §S8", "## S8." in supp),
        ("vignette.pdf", size("vignette.pdf")),
        ("requirements-lock.txt", size("requirements-lock.txt", 100)),
        ("docs/CODE_MAP.md", size("docs/CODE_MAP.md")),
        ("docs/FUNCTION_REFERENCE.md", size("docs/FUNCTION_REFERENCE.md")),
        ("docs/fig_execution_flow.pdf", size("docs/fig_execution_flow.pdf")),
        ("code/verify_manuscript.py", os.path.exists(os.path.join(ROOT, "code/verify_manuscript.py"))),
        ("code/verify_dcor_against_R.py", os.path.exists(os.path.join(ROOT, "code/verify_dcor_against_R.py"))),
        ("R cross-check recorded in the package",
         "verify_dcor_against_R" in draft or "verify_dcor_against_R" in letter
         or "verify_dcor_against_R" in open(os.path.join(ROOT, "vignette.md"), encoding="utf-8").read()
         or "verify_dcor_against_R" in open(os.path.join(ROOT, "docs", "CODE_MAP.md"), encoding="utf-8").read()),
        ("single consolidated manuscript", os.path.exists(os.path.join(ROOT, "draft.md"))),
    ]
    bad = []
    for name, ok in checks:
        print(f"  {'OK  ' if ok else 'FAIL'} {name}")
        if not ok:
            bad.append(name)
    print(f"\n{len(checks) - len(bad)}/{len(checks)} letter claims verified on disk")
    if bad:
        print("FAILED:", bad)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
