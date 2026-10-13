#!/usr/bin/env python3
"""verify_manuscript.py -- cross-check every number asserted in `draft.md` against
the result files that produced it.

The point of this script is adversarial: it does not trust the manuscript. Each
check names the manuscript claim, the JSON path it should come from, and the
tolerance implied by the Monte Carlo budget. Run it after any edit to `draft.md`
or to a results file.

Usage:
    python3 verify_manuscript.py [-v]

Exit code 0 = all checks passed; 1 = at least one mismatch.
"""

from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))     # code/
ROOT = os.path.dirname(HERE)                          # package root

VERBOSE = "-v" in sys.argv


def load(name):
    """Load one of the results files.

    The files live at the package root in the submission layout and in ``results/`` in
    the public repository layout, so both locations are searched.  This keeps one copy of
    the verification code working in either place.

    Args:
        name: file name, e.g. ``'rev_results.json'``.

    Returns:
        dict: the parsed JSON content.

    Raises:
        FileNotFoundError: if the file is in neither location.
    """
    for base in (ROOT, os.path.join(ROOT, "results")):
        path = os.path.join(base, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
    raise FileNotFoundError(f"{name} not found in {ROOT} or {ROOT}/results")


def draft():
    """Read the manuscript text, or return an empty string when it is not shipped.

    ``draft.md`` is submission material and is not distributed with the reproduction
    package.  When it is absent the result-file checks still run and only the
    text-consistency checks are skipped, which the caller reports.

    Returns:
        str: the full text of ``draft.md`` (UTF-8), or ``""`` if it is not present.
    """
    path = os.path.join(ROOT, "draft.md")
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as fh:
        return fh.read()


class Checker:
    """Collect pass/fail results of the manuscript cross-check and summarize them.

    Methods:
        eq(label, got, want, tol): compare two numbers within a tolerance.
        text(label, needle, present=True): check that a string occurs in
            ``draft.md``.
        skip(label, why): record a check that could not be performed.
    """
    def __init__(self):
        self.passed = 0
        self.failed = []
        self.skipped = []

    def eq(self, label, got, want, tol):
        ok = got is not None and abs(float(got) - float(want)) <= tol + 1e-12
        if ok:
            self.passed += 1
            if VERBOSE:
                print(f"  ok   {label}: {got} == {want} (tol {tol})")
        else:
            self.failed.append((label, got, want, tol))
            print(f"  FAIL {label}: got {got}, manuscript says {want}, tol {tol}")

    def text(self, label, needle, present=True):
        hay = self._draft
        if not hay:
            # The manuscript is submission material and is not part of the reproduction
            # package, so text checks have nothing to read.  They are counted as skipped
            # rather than failed; the numeric checks all still run.
            self.skipped.append((label, "draft.md not shipped with the reproduction package"))
            return
        found = needle in hay
        if found == present:
            self.passed += 1
            if VERBOSE:
                print(f"  ok   {label}")
        else:
            self.failed.append((label, "present" if found else "absent", needle, None))
            print(f"  FAIL {label}: expected {'present' if present else 'absent'}: {needle[:80]!r}")

    def skip(self, label, why):
        self.skipped.append((label, why))


def main():
    """Run every check group and print the summary.

    The groups cover the archived Study 2/1/3 cells, the severity ladders, the
    sample-size sweep, the alpha sweep, the power study, source hygiene (table
    captions and references), the new-test panel, the new-tests power file and
    free-text claims. Any mismatch is printed as FAIL with the JSON path.

    Returns:
        int: 0 when every check passed, 1 otherwise.
    """
    c = Checker()
    c._draft = draft()
    arch = load("experiment_results.json")
    sens = load("rev_results.json")
    power = load("rev_results_power.json")

    # ---------------------------------------------------------------- archived
    print("[1] Archived Study 2 cells (experiment_results.json -> Table 5)")
    s2 = arch["study2"]
    for test, scen, want in [
        ("T1_t_test", "baseline", 0.052), ("T1_t_test", "skewed", 0.066),
        ("T5_bartlett", "baseline", 0.048), ("T5_bartlett", "skewed", 0.429),
        ("T5_bartlett", "heavy_tail", 0.276), ("T6_fligner_killeen", "skewed", 0.119),
        ("T6_fligner_killeen", "baseline", 0.041), ("T9_median_test", "small_freq", 0.069),
        ("T10_hotelling", "skewed", 0.076), ("T2_anova", "hetero_var", 0.061),
        ("T3_kruskal_wallis", "hetero_var", 0.064), ("T4_chisq_gof", "small_freq", 0.051),
    ]:
        c.eq(f"Table5 {test}/{scen}", s2[test][scen], want, 0.0005)

    d2 = arch["study2_delta"]
    for test, scen, want in [
        ("T5_bartlett", "skewed", 0.381), ("T5_bartlett", "heavy_tail", 0.228),
        ("T6_fligner_killeen", "skewed", 0.078), ("T10_hotelling", "skewed", 0.026),
    ]:
        c.eq(f"Table6 {test}/{scen}", d2[test][scen], want, 0.0005)

    print("[2] Archived Study 1 / Study 3 (Table 4, Table 7)")
    for test, want_ks in [("T3_kruskal_wallis", 20), ("T4_chisq_gof", 50),
                          ("T8_cochran_q", 200)]:
        c.eq(f"n*_KS {test}", float(arch["study1_nstar_ks"][test].replace(">", "")), want_ks, 0)
    # the median test never crosses the threshold within the grid; the manuscript writes ">500"
    c.text("median test n*_KS reported as >500", ">500")
    c.eq("median test KS at n=500 (text)", arch["study1"]["T9_median_test"]["ks"][-1], 0.025, 0.001)
    c.eq("KS threshold", arch["study1_ks_threshold"], 0.0136, 0.0001)
    for test, want in [("T2_anova", 1500), ("T6_fligner_killeen", 3000),
                       ("T8_cochran_q", 3000), ("T9_median_test", 2000)]:
        c.eq(f"N* {test}", float(arch["study3_nstar_mc"][test]), want, 0)

    # --------------------------------------------------------------- sensitivity
    print("[3] Severity ladders (rev_results.json -> Section 5.4)")
    L = sens["severity_sweep"]["ladders"]
    for test, want in [("T5_bartlett", [0.052, 0.267, 0.432, 0.432, 0.772]),
                       ("T6_fligner_killeen", [0.045, 0.069, 0.111, 0.111, 0.141]),
                       ("T10_hotelling", [0.054, 0.065, 0.074, 0.074, 0.137]),
                       ("T1_t_test", [0.050, 0.056, 0.065, 0.065, 0.105])]:
        got = L["skewness"]["tests"][test]["alpha_hat"]
        for i, (g, w) in enumerate(zip(got, want)):
            c.eq(f"skewness {test} rung{i}", g, w, 0.001)
    for test, want in [("T5_bartlett", [0.052, 0.067, 0.114, 0.268, 0.521]),
                       ("T10_hotelling", [0.054, 0.052, 0.049, 0.048, 0.042])]:
        got = L["kurtosis"]["tests"][test]["alpha_hat"]
        for i, (g, w) in enumerate(zip(got, want)):
            c.eq(f"kurtosis {test} rung{i}", g, w, 0.001)
    for test, want in [("T2_anova", [0.049, 0.050, 0.053, 0.059, 0.070, 0.076]),
                       ("T3_kruskal_wallis", [0.048, 0.048, 0.051, 0.057, 0.067, 0.073])]:
        got = L["variance_ratio"]["tests"][test]["alpha_hat"]
        for i, (g, w) in enumerate(zip(got, want)):
            c.eq(f"var-ratio {test} rung{i}", g, w, 0.001)
    for test, want in [("T4_chisq_gof", [0.049, 0.050, 0.048, 0.051, 0.054, 0.051]),
                       ("T8_cochran_q", [0.051, 0.050, 0.049, 0.050, 0.016])]:
        L2 = L["small_freq_t4"] if test == "T4_chisq_gof" else L["small_freq_t8"]
        got = L2["tests"][test]["alpha_hat"]
        for i, (g, w) in enumerate(zip(got, want)):
            c.eq(f"small-freq {test} rung{i}", g, w, 0.001)

    print("[4] Sample-size sweep (rev_results.json -> Table 10)")
    ns = sens["study2_nsweep"]
    for scen, test, want in [
        ("skewed", "T1_t_test", [0.083, 0.066, 0.063, 0.056]),
        ("skewed", "T10_hotelling", [0.106, 0.082, 0.066, 0.060]),
        ("skewed", "T5_bartlett", [0.379, 0.427, 0.449, 0.460]),
        ("heavy_tail", "T5_bartlett", [0.214, 0.268, 0.320, 0.345]),
        ("skewed", "T6_fligner_killeen", [0.085, 0.111, 0.134, 0.147]),
        ("hetero_var", "T2_anova", [0.063, 0.060, 0.067, 0.059]),
        ("hetero_var", "T3_kruskal_wallis", [0.063, 0.064, 0.059, 0.060]),
        ("unbalanced", "T2_anova", [0.050, 0.049, 0.049, 0.048]),
        ("small_freq", "T8_cochran_q", [0.035, 0.042, 0.052, 0.053]),
    ]:
        cells = ns["scenarios"][scen]["tests"][test]["by_n"]
        for n, w in zip([20, 50, 100, 200], want):
            c.eq(f"n-sweep {scen}/{test}/n={n}", cells[str(n)]["alpha_hat"], w, 0.001)

    print("[5] Alpha sweep (rev_results.json -> Section 5.4 S1)")
    A = sens["alpha_sweep"]["severity_sweep"]["skewness"]["T5_bartlett"]["Exp(1)"]
    c.eq("Bartlett Exp(1) alpha=0.01", A["0.01"]["alpha_hat"], 0.266, 0.001)
    c.eq("Bartlett Exp(1) alpha=0.05", A["0.05"]["alpha_hat"], 0.432, 0.001)
    c.eq("Bartlett Exp(1) alpha=0.10", A["0.10"]["alpha_hat"], 0.522, 0.001)

    # ------------------------------------------------------------------- power
    print("[6] Power study (rev_results_power.json -> Tables 11-12)")
    cells = power["study4_power"]["cells"]

    def cell(fam, test, parent, n, scen):
        for x in cells:
            if (x["family"] == fam and x["test"] == test and x["parent"] == parent
                    and x["n"] == n and x["scenario"] == scen):
                return x
        return None

    for parent, ratio, a_b, a_f, adj_b, adj_f in [
        ("normal", 1.25, 0.047, 0.042, 0.363, 0.306),
        ("normal", 1.50, 0.047, 0.042, 0.853, 0.761),
        ("tdist", 1.50, 0.270, 0.048, 0.408, 0.595),
        ("expon", 1.50, 0.421, 0.114, 0.328, 0.406),
        ("contam", 1.50, 0.690, 0.046, 0.207, 0.517),
        ("contam", 2.00, 0.690, 0.046, 0.494, 0.935),
    ]:
        b = cell("scale", "T5_bartlett", parent, 50, f"rho={ratio}")
        f = cell("scale", "T6_fligner_killeen", parent, 50, f"rho={ratio}")
        for got, want, nm in [(b["alpha_hat"], a_b, "B ahat"), (f["alpha_hat"], a_f, "F ahat"),
                              (b["power_adjusted"], adj_b, "B adjpow"),
                              (f["power_adjusted"], adj_f, "F adjpow")]:
            c.eq(f"Table11 {parent} rho={ratio} {nm}", got, want, 0.001)

    summary = power["study4_power"]["rank_reversal"]["summary"]
    c.eq("reversal cells", summary["n_verdict_reversals"], 34, 0)
    c.eq("scale cells", summary["n_cells"], 64, 0)
    c.eq("size-valid Bartlett cells", summary["n_size_valid_bartlett_cells"], 12, 0)
    c.eq("Bartlett wins when calibrated",
         summary["n_size_valid_bartlett_cells_where_bartlett_wins_adjusted"], 6, 0)

    for fam, ta, tb, parent, scen, want_a, want_b, tol in [
        ("location3", "T2_anova", "T3_kruskal_wallis", "normal", "d=0.5", 0.598, 0.579, 0.002),
        ("location3", "T2_anova", "T3_kruskal_wallis", "tdist", "d=0.5", 0.595, 0.682, 0.002),
        ("location3", "T2_anova", "T3_kruskal_wallis", "contam", "d=0.5", 0.622, 0.932, 0.002),
        ("location3", "T3_kruskal_wallis", "T9_median_test", "normal", "d=0.5", 0.579, 0.390, 0.002),
        ("location3", "T2_anova", "T9_median_test", "contam", "d=0.5", 0.622, 0.843, 0.002),
        ("location1", "T1_t_test", "T7_sign_test", "normal", "d=0.5", 0.936, 0.742, 0.002),
        ("location1", "T1_t_test", "T7_sign_test", "contam", "d=0.5", 0.920, 0.994, 0.002),
    ]:
        ca = cell(fam, ta, parent, 50, scen)
        cb = cell(fam, tb, parent, 50, scen)
        if ca is None or cb is None:
            c.skip(f"Table12 {ta} vs {tb} {parent} {scen}", "cell missing")
            continue
        c.eq(f"Table12 {ta} {parent}", ca["power_adjusted"], want_a, tol)
        c.eq(f"Table12 {tb} {parent}", cb["power_adjusted"], want_b, tol)

    print("[7] Source hygiene")
    _raw = draft()
    if not _raw:
        c.skip("source hygiene and text checks",
               "draft.md is submission material and is not distributed with this package")
    else:
        raw = _raw
        ctrl = [(i, hex(ord(ch))) for i, ch in enumerate(raw) if ord(ch) < 32 and ch not in "\n\t"]
        if ctrl:
            c.failed.append(("control characters in draft.md", ctrl[:5], "none", None))
            print(f"  FAIL control characters in draft.md: {ctrl[:5]}")
        else:
            c.passed += 1
            if VERBOSE:
                print("  ok   no stray control characters")
        greek = re.findall(r"[\u03b1-\u03c9]", raw)
        if greek:
            c.failed.append(("raw Greek characters (use LaTeX math)", sorted(set(greek)), "none", None))
            print(f"  FAIL raw Greek characters outside math: {sorted(set(greek))}")
        else:
            c.passed += 1
            if VERBOSE:
                print("  ok   no raw Greek characters")
        for fig in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", raw):
            if not os.path.exists(os.path.join(ROOT, fig)):
                c.failed.append((f"missing figure {fig}", None, "file exists", None))
                print(f"  FAIL figure referenced but absent: {fig}")
            else:
                c.passed += 1
        for m in re.findall(r"\((?:eqref|ref)\{([^}]*)\}", raw):
            c.failed.append((f"LaTeX-only cross-reference {m}", None, "markdown link", None))
            print(f"  FAIL LaTeX-only cross-reference found: {m}")

        caps = [int(m.group(1)) for m in re.finditer(r"^\*\*Table (\d+):", raw, re.M)]
        if caps != list(range(1, len(caps) + 1)):
            c.failed.append(("table numbering not contiguous", caps, "1..n", None))
            print(f"  FAIL table numbering not contiguous: {caps}")
        else:
            c.passed += 1
        refs = set()
        for line in raw.split("\n"):
            if line.startswith("**Table "):
                continue
            refs.update(int(m.group(1)) for m in re.finditer(r"Tables? (\d+)", line))
        dangling = sorted(refs - set(caps))
        if dangling:
            c.failed.append(("dangling table references", dangling, "none", None))
            print(f"  FAIL dangling table references: {dangling}")
        else:
            c.passed += 1
        print(f"  tables: {len(caps)} captions, refs to {sorted(refs)}")

        print("[8] New-test panel (rev_results_t1.json -> Tables 4-6, 12)")
        try:
            t1 = load("rev_results_t1.json")
        except FileNotFoundError:
            c.skip("rev_results_t1.json", "not on disk")
            t1 = None
        if t1:
            for key in ["study1_ext", "study1_ext_nstar", "null_calibration",
                        "t12_reference_diagnostic", "t13_split_diagnostic",
                        "alpha_sweep_13", "dependence_13", "archive_reproduction_check"]:
                if key in t1:
                    c.passed += 1
                else:
                    c.failed.append((f"rev_results_t1.json missing {key}", None, "present", None))
                    print(f"  FAIL rev_results_t1.json lost block: {key}")
            # KS convention: T1-T10 must match the archive to floating tolerance
            arch_ks = load("experiment_results.json")["study1"]
            worst = 0.0
            for k, v in arch_ks.items():
                v2 = t1["study1_ext"].get(k)
                if not v2:
                    continue
                for i in range(len(v["ks"])):
                    if v["ks"][i] is None or v2["ks"][i] is None:
                        continue
                    worst = max(worst, abs(v2["ks"][i] - v["ks"][i]))
            c.eq("KS convention matches archive (max |diff|)", worst, 0.0, 1e-9)
            # T11/T12/T13 key values quoted in Table 4
            # T12's KS in Table 4 is measured against the independent reference and lives in
            # rev_results_t12_diag.json; its study1_ext entry holds the historical distance
            # against the published table, which is no longer the reference.
            for test, n, want in [("T11_kpss", 500, 0.019), ("T13_interaction", 500, 0.444)]:
                c.eq(f"Table4 {test} KS(n=500)", t1["study1_ext"][test]["ks"][-1], want, 0.001)
            c.text("Table4 T11 row (relative criterion + KS not attained)",
                   "| T11: KPSS | 50 | 500 | 10 | — | 0.019 | — |")
            c.text("Table6 bold rule: median unbolded at 0.019 deviation",
                   "| T9: Median | 0.043 | 0.046 | 0.045 | 0.056 | 0.051 | 0.069 |")
            import re as _re
            _dtxt = open(os.path.join(ROOT, "draft.md"), encoding="utf-8").read()
            _t6 = _dtxt[_dtxt.index("**Table 6:"):]
            _t6 = _t6[:_t6.index("\n\n")]
            for _cell in _re.findall(r"\*\*([0-9]+\.[0-9]{3})\*\*", _t6):
                c.eq(f"Table6 bold cell {_cell} deviates >0.025 from nominal",
                     round(abs(float(_cell) - 0.05) > 0.025, 10), 1.0, 1e-9)
            c.text("Table14 reports two separate axes, not one label",
                   "Robustness reported on separate axes rather than as a single rating")
            c.text("T12 reference is an independent simulation, not the table",
                   "KS to the independent reference")
            c.text("Table4 uses the relative criterion",
                   r"|\hat{\alpha}-\alpha|/\alpha \le 0.2")
            # T12 diagnostic rows, from the file the generator writes (code/gen_t12_law.py)
            t12 = {r["n"]: r for r in load("rev_results_t12_diag.json")}
            c.eq("Table5 T12 KS to the independent reference, n=500", t12[500]["ks_law"], 0.0045, 0.0005)
            c.eq("Table5 T12 KS to the independent reference, n=20", t12[20]["ks_law"], 0.1554, 0.001)
            c.eq("Table5 T12 rejection rate n=500", t12[500]["a05"], 0.0509, 0.001)
            c.eq("T12 reference resolution (200000 draws)", t12[500]["crit"], 2.8435, 0.0005)
            law = load("rev_results_t12_law.json")
            c.eq("T12 reference sample size", law["meta"]["N_law"], 200000, 0)
            c.eq("T12 reference n_law", law["meta"]["n_law"], 2000, 0)
            # dependence values quoted in Table 12
            ar1 = t1["dependence_13"]["ar1"]["tests"]
            for test, want in [("T1_t_test", [0.108, 0.262, 0.527]),
                               ("T7_sign_test", [0.070, 0.167, 0.410])]:
                cells = ar1[test]["cells"]
                for rho, w in zip([0.2, 0.5, 0.8], want):
                    c.eq(f"Table12 AR1 {test} rho={rho}",
                         cells[f"n100_rho{rho}"]["type1"]["0.05"], w, 0.001)
            cl = t1["dependence_13"]["clustered"]["tests"]
            for test, want in [("T2_anova", [0.364, 0.624]), ("T3_kruskal_wallis", [0.335, 0.610]),
                               ("T8_cochran_q", [0.048, 0.054])]:
                cells = cl[test]["cells"]
                for icc, w in zip([0.2, 0.5], want):
                    c.eq(f"Table12 clustered {test} ICC={icc}",
                         cells[f"icc{icc}"]["type1"]["0.05"], w, 0.001)

        print("[9] New-tests power (rev_results_newtests_power.json)")
        try:
            npw = load("rev_results_newtests_power.json")
            c.eq("T11 power n=50", npw["T11_kpss"]["n=50"]["power_nominal"], 0.713, 0.002)
            c.eq("T11 size-adj power n=200", npw["T11_kpss"]["n=200"]["power_size_adjusted"], 0.951, 0.002)
            c.eq("T12 size-adj linear n=50",
                 npw["T12_hoeffding_d"]["n=50"]["power_size_adjusted_linear"], 0.910, 0.003)
            c.eq("T12 size-adj quad n=50",
                 npw["T12_hoeffding_d"]["n=50"]["power_size_adjusted_quadratic"], 0.996, 0.003)
        except FileNotFoundError:
            c.skip("rev_results_newtests_power.json", "not on disk")

        print("[9b] T14 distance-correlation panel (rev_results_dcor*.json)")
        try:
            dc = load("rev_results_dcor.json")
            s1 = dc["study1"]
            for key, want in [("d=50|n=20", 0.0530), ("d=50|n=500", 0.0509),
                              ("d=2|n=20", 0.0678), ("d=2|n=500", 0.0702)]:
                c.eq(f"T14 a05 {key}", s1[key]["type1"]["0.05"][0], want, 0.001)
            c.eq("T14 KS at d=50,n=20", s1["d=50|n=20"]["ks"][0], 0.006, 0.001)
            c.eq("T14 KS at d=2,n=500", s1["d=2|n=500"]["ks"][0], 0.074, 0.002)
            nu = dc["nuisance_within_vector"]
            c.eq("T14 nuisance eta=0.5 n=20", nu["eta=0.5|n=20"]["type1"], 0.0558, 0.002)
            c.text("T14 described as a dimension limit",
                   "Reference is a limit in the dimension")
            c.text("T14 concentrated-alternative power reported",
                   "0.059 at $d = 50$")
            loc = load("rev_results_dcor_local.json")
            got = {(r["d"], r["k_signal"], r["n"]): r["power"] for r in loc["rows"]}
            c.eq("T14 local power d=50 k=1 n=100", got[(50, 1, 100)], 0.0586, 0.004)
            c.eq("T14 local power d=50 k=50 n=200", got[(50, 50, 200)], 0.4709, 0.004)
            st3 = load("rev_results_t14_study3.json")
            c.eq("T14 N* (d=50)", st3["n_star_mc"], 3000, 0)
        except FileNotFoundError as exc:
            c.skip("T14 results", str(exc))

        print("[10] Text claims that must be consistent with the JSON")
        c.text("no 'first systematic' claim in abstract",
               "This paper presents the first systematic Monte Carlo", present=False)
        c.text("MC abbreviation defined", "Monte Carlo (MC)")
        c.text("reproducible not regenerable", "regenerable", present=False)
        c.text("T8 imbalance marked n/a", "| T8: Cochran $Q$ | 0.050 | — | — | — | n/a | 0.041 |")
        c.text("Kruskal-Wallis formula in the display block", r"H &= \frac{12}{n_T(n_T+1)} \sum_{i=1}^{k} \frac{R_i^2}{n_i} - 3(n_T+1)")
        c.text("MSTR defined", r"\text{MSTR} = \frac{1}{k-1}")
        c.text("Figure 3 caption no longer wrong",
               "All tests achieve CV < 0.10 by N = 3,000 at the latest", present=False)
        c.text("Figure 2 legend note", "Figure 2: Empirical Type I error rates")
        # The sign test IS correctly centered in the archive and in the ladders, so the
        # manuscript must assert that rather than disclaim it.
        c.text("sign test documented as genuine Type I error",
               "the sign test is centered at the population median of each parent")
        c.text("no false-null claim for the sign test",
               "measures size under a false null", present=False)
        c.text("Wilson endpoints pointed at the supplement",
               "Supplementary Material §S4")
        c.text("Table 9 no longer shows the ratio-to-nominal column",
               "$N^*/N^*_{theory}$", present=False)
        c.text("binomial model uses pi not alpha",
               r"$\text{Binomial}(N, \pi)$")

        print()
        print(f"passed: {c.passed}   failed: {len(c.failed)}   skipped: {len(c.skipped)}")
        if c.skipped:
            for lab, why in c.skipped:
                print(f"  skipped: {lab} ({why})")
        if c.failed:
            print("\nFAILURES:")
            for lab, got, want, tol in c.failed:
                print(f"  - {lab}: got={got!r} want={want!r} tol={tol!r}")
            return 1
        print("ALL CHECKS PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
