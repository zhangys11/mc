#!/usr/bin/env python3
"""Worked example: reproduce one published number from the paper end to end.

Run from the repository root::

    python3 docs/verify_bartlett.py

Target number
-------------
Bartlett's test (T5) under the skewed parent Exp(1)-1, k = 3 groups of n = 50,
N = 10,000 MC replications.  The manuscript reports alpha-hat = 0.429 (Table 5 and
the abstract).  The stored machine value is 0.4288 in ``experiment_results.json``
(``study2.T5_bartlett.skewed``), produced by the archived ``experiments.py``.

What this script does
---------------------
1. Re-runs the *same* scenario through the revision pipeline of
   ``rev_sensitivity.py`` with the seed that produced ``rev_results.json``, and
   checks the printed rate and Wilson interval against the JSON bit for bit
   (exact float equality).  This is the quickest end-to-end check of the pipeline
   (about one second).
2. Re-runs the scenario independently through the shared kernel ``core.py``
   (``core.mc_bartlett`` + ``core._evaluate``) with a *different* seed, which
   exercises the public evaluation path used by every study.
3. Collects the three independent Monte Carlo realisations of the same scenario
   that exist in the results files -- the archive run, the revision n-sweep, and
   the power-study calibration check -- and verifies that they are statistically
   consistent (two-sample z on the difference of two proportions, 3-sd band).
   They are *different random samples*, so they agree to within Monte Carlo error
   rather than exactly; the vignette says so explicitly.
4. Checks the two closed-form constants quoted in the manuscript: the KS
   adequacy threshold 1.358/sqrt(N) = 0.01358 at N = 10,000, and the Wilson
   interval half-width near alpha = 0.05.

Exit status 0 means every check passed; any mismatch raises and exits non-zero.

Author: t4-docs (task-4).
"""

from __future__ import annotations

import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def results_path(name):
    """Locate a results file in either supported layout.

    The submission layout keeps results at the package root; the public repository keeps
    them in ``results/``.  Both are accepted so one copy of this script works in either.

    Args:
        name: results file name.

    Returns:
        str: absolute path to the file.

    Raises:
        FileNotFoundError: if the file is in neither location.
    """
    for base in (ROOT, os.path.join(ROOT, "results")):
        path = os.path.join(base, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{name} not found in {ROOT} or {ROOT}/results")
sys.path.insert(0, os.path.join(ROOT, "code"))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402

# The archive module ships as code/archive_experiments.py in the repository layout and as
# experiments.py next to this file in the submission layout; accept either.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _base in (_HERE, _ROOT, os.path.join(_ROOT, "code")):
    if os.path.exists(os.path.join(_base, "experiments.py")) or \
       os.path.exists(os.path.join(_base, "archive_experiments.py")):
        sys.path.insert(0, _base)
        break
import scipy  # noqa: E402

import core  # noqa: E402

TEST_ID = "T5_bartlett"
N_MC = 10_000
ALPHA = 0.05

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    """Record and print one PASS/FAIL check; the exit status depends on these."""
    checks.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def z_two_proportions(p1: float, n1: int, p2: float, n2: int) -> float:
    """Two-sample z statistic for the difference of two independent proportions."""
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return 0.0 if se == 0 else (p1 - p2) / se


def main() -> int:
    """Run the four verification blocks and return the process exit status."""
    print("=" * 74)
    print("Worked example: Bartlett Type I error under the skewed parent Exp(1)-1")
    print("=" * 74)
    print(f"python {sys.version.split()[0]}   numpy {np.__version__}   "
          f"scipy {scipy.__version__}")
    print(f"core.py seed={core.SEED}  N_MC={core.N_MC}  alpha={ALPHA}")

    # ------------------------------------------------------------------
    # 1. Exact reproduction through rev_sensitivity.py (the t2 pipeline)
    # ------------------------------------------------------------------
    print("\n[1] Exact reproduction through rev_sensitivity.py (committed seed)")
    import rev_sensitivity as rs

    spec = rs.nsweep_spec(TEST_ID, "skewed", 50)
    seed = rs.cell_seed("nsweep", "skewed")
    stats, rule = rs.gen_cell(TEST_ID, spec, seed, N_MC)
    alpha_hat = rs.reject_rate(stats, rule, ALPHA)
    ci_lo, ci_hi = rs.wilson_ci(alpha_hat, N_MC)
    print(f"    spec        : {spec}")
    print(f"    cell seed   : {seed}   (rev_sensitivity.cell_seed('nsweep', 'skewed'))")
    print(f"    N           : {N_MC}")
    print(f"    alpha_hat   : {alpha_hat:.4f}")
    print(f"    95% Wilson  : [{ci_lo:.6f}, {ci_hi:.6f}]")

    with open(results_path("rev_results.json")) as fh:
        rev = json.load(fh)
    ref = rev["study2_nsweep"]["scenarios"]["skewed"]["tests"][TEST_ID]["by_n"]["50"]
    check("rev_results.json  study2_nsweep ... by_n['50'].alpha_hat == printed value",
          alpha_hat == ref["alpha_hat"],
          f"{alpha_hat} == {ref['alpha_hat']}  (exact float equality)")
    check("...ci_lo matches the stored Wilson interval",
          abs(ci_lo - ref["ci_lo"]) < 1e-12, f"{ci_lo:.12f}")
    check("...ci_hi matches the stored Wilson interval",
          abs(ci_hi - ref["ci_hi"]) < 1e-12, f"{ci_hi:.12f}")
    check("...stored seed matches the seed used here", seed == ref["seed"],
          f"{seed} == {ref['seed']}")

    # ------------------------------------------------------------------
    # 2. Independent route through the shared kernel core.py
    # ------------------------------------------------------------------
    print("\n[2] Independent route through the shared kernel core.py (fresh seed)")
    kernel_seed = 7
    np.random.seed(kernel_seed)
    kernel_stats = core.mc_bartlett(3, 50, N_MC, dist="expon")
    res = core._evaluate(kernel_stats, TEST_ID, ALPHA)
    print(f"    np.random.seed({kernel_seed}); core.mc_bartlett(3, 50, {N_MC}, "
          f"dist='expon'); core._evaluate(..., '{TEST_ID}', {ALPHA})")
    print(f"    type1       : {res['type1']:.4f}   95% Wilson: "
          f"[{res['ci_lo']:.6f}, {res['ci_hi']:.6f}]")
    print(f"    KS distance : {res['ks']:.4f}   critical value chi2_0.95(2) = "
          f"{res['critical_value']:.6f}")
    z_kernel = z_two_proportions(res["type1"], N_MC, ref["alpha_hat"], N_MC)
    check("kernel run agrees with rev_results.json within 3 Monte Carlo sd",
          abs(z_kernel) < 3.0, f"z = {z_kernel:+.3f}")

    # The archive script's global stream: the same generator called on its own
    # after np.random.seed(2024) -- the archive script's seed -- gives a different
    # realisation, because in the archive run the stream has already been consumed
    # by Study 1 before Study 2 draws Bartlett's data.  Both values are correct.
    np.random.seed(core.SEED)
    standalone = core._evaluate(core.mc_bartlett(3, 50, N_MC, dist="expon"),
                                TEST_ID, ALPHA)
    print(f"    archive generator alone: np.random.seed({core.SEED}); "
          f"core.mc_bartlett(3, 50, {N_MC}, dist='expon')")
    print(f"    type1       : {standalone['type1']:.4f}   -- NOT the archived 0.4288, "
          f"because the archive stream is consumed by Study 1 first")
    z_stand = z_two_proportions(standalone["type1"], N_MC, 0.4288, N_MC)
    check("standalone archive call is within 3 Monte Carlo sd of the archived value",
          abs(z_stand) < 3.0, f"z = {z_stand:+.3f}")

    # ------------------------------------------------------------------
    # 3. All independent realisations of the same scenario
    # ------------------------------------------------------------------
    print("\n[3] Independent Monte Carlo realisations of the same scenario")
    with open(results_path("experiment_results.json")) as fh:
        archive = json.load(fh)
    with open(results_path("rev_results_power.json")) as fh:
        power = json.load(fh)

    a_arch = archive["study2"][TEST_ID]["skewed"]
    row = next(r for r in power["study4_power"]["verification"]["calibration"]["rows"]
               if r["test"] == TEST_ID and r["parent"] == "expon")
    a_pow = row["alpha_hat"]

    print(f"    archive  experiment_results.json : study2.{TEST_ID}.skewed"
          f"                 = {a_arch}")
    print(f"    revision rev_results.json        : study2_nsweep ... by_n['50']"
          f"          = {ref['alpha_hat']}")
    print(f"    power    rev_results_power.json  : verification.calibration.rows"
          f"       = {a_pow}")
    print(f"    kernel   this script (seed {kernel_seed})"
          f"                                    = {res['type1']:.4f}")

    z_arch_pow = z_two_proportions(a_arch, N_MC, a_pow, N_MC)
    z_arch_rev = z_two_proportions(a_arch, N_MC, ref["alpha_hat"], N_MC)
    print(f"    archive vs power  : z = {z_arch_pow:+.3f}"
          f"   (stored by rev_power.py: {row['z_two_sample']:+.3f})")
    print(f"    archive vs n-sweep: z = {z_arch_rev:+.3f}")
    check("archive realisation agrees with the power study's calibration check "
          "within 3 sd", abs(z_arch_pow) < 3.0, f"z = {z_arch_pow:+.3f}")
    check("archive realisation agrees with the revision n-sweep within 3 sd",
          abs(z_arch_rev) < 3.0, f"z = {z_arch_rev:+.3f}")
    check("rev_power.json stores the archive value and its difference correctly",
          abs(row["archive_alpha_hat"] - a_arch) < 1e-12
          and abs(row["difference"] - (a_pow - a_arch)) < 1e-12,
          f"archive_alpha_hat={row['archive_alpha_hat']}, "
          f"difference={row['difference']:.4f}")
    check("the manuscript's rounded value 0.429 equals round(0.4288, 3)",
          round(a_arch, 3) == 0.429, f"round({a_arch}, 3) = {round(a_arch, 3)}")

    # ------------------------------------------------------------------
    # 4. Closed-form constants quoted in the manuscript
    # ------------------------------------------------------------------
    print("\n[4] Closed-form constants used throughout the paper")
    lo, hi = core.wilson_ci(ALPHA, N_MC)
    print(f"    core.wilson_ci(0.05, 10000)      = ({lo:.10f}, {hi:.10f})")
    print(f"    half-width at alpha = 0.05       = {(hi - lo) / 2:.6f}")
    print(f"    core.KS_THRESHOLD                = {core.KS_THRESHOLD:.10f}")
    print(f"    1.358 / sqrt(10000)              = {1.358 / math.sqrt(N_MC):.10f}")
    check("Wilson interval is centered on 0.05 and contains it",
          lo < ALPHA < hi, f"[{lo:.6f}, {hi:.6f}]")
    check("Wilson half-width is about 0.004 (manuscript says approx. 0.004)",
          abs((hi - lo) / 2 - 0.004) < 0.0006, f"half-width = {(hi - lo) / 2:.6f}")
    check("KS threshold equals 1.358/sqrt(N) to machine precision",
          core.KS_THRESHOLD == 1.358 / math.sqrt(N_MC),
          f"{core.KS_THRESHOLD!r}")

    failed = [c for c in checks if not c[1]]
    print("\n" + "=" * 74)
    print(f"{len(checks) - len(failed)}/{len(checks)} checks passed")
    if failed:
        for name, _, detail in failed:
            print(f"  FAILED: {name} {detail}")
        return 1
    print("Worked example reproduced: Bartlett alpha-hat = 0.4267 "
          "(rev_results.json, n = 50, N = 10,000, skewed parent),")
    print("consistent with the 0.4288 archive value reported as 0.429 in the paper.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
