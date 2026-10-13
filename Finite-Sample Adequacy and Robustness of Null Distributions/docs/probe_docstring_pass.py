#!/usr/bin/env python3
"""Behavioral fingerprint of the pipeline, used to prove that a documentation-only
edit changed nothing that the scripts compute.

Run from the repository root::

    python3 docs/probe_docstring_pass.py

It imports every study and verification script, calls a fixed set of functions with
fixed inputs and seeds, and prints one SHA-256 over all results. The digest is
identical before and after a docstring pass (the AST digest proves the *code* is
unchanged; this probe additionally proves the *behavior* is unchanged, including
the RNG conventions).

Nothing is written to disk.

Author: t4-docs (task-4).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))

import numpy as np  # noqa: E402

import core  # noqa: E402
import rev_alpha_dependence as rad  # noqa: E402
import rev_power as rp  # noqa: E402
import rev_sensitivity as rs  # noqa: E402
import rev_study1_ext as rse  # noqa: E402
import verify_manuscript as vm  # noqa: E402
import verify_new_tests_power as vnp  # noqa: E402


def digest_array(a) -> str:
    """Short SHA-256 of an array's float64 bytes, used as a behavioral fingerprint.

    Args:
        a: array-like of numbers.

    Returns:
        str: the first 24 hex characters of the SHA-256 of the float64 buffer.
    """
    a = np.asarray(a, dtype=np.float64)
    return hashlib.sha256(a.tobytes()).hexdigest()[:24]


def main() -> int:
    """Collect the fingerprint of every script and print its SHA-256.

    Returns:
        int: process exit status (0 unless a probe call raises).
    """
    fp: dict[str, object] = {}

    # ---- rev_sensitivity.py: seeded cells and statistic constructors --------
    fp["cell_seed.nsweep_skewed"] = rs.cell_seed("nsweep", "skewed")
    fp["cell_seed.dist_ladder"] = rs.cell_seed("dist_ladder")
    fp["wilson_ci"] = [float(x) for x in rs.wilson_ci(0.05, 10_000)]
    fp["std_params_lognorm(1.0)"] = [float(x) for x in rs._std_params_lognorm(1.0)]
    fp["skew_kurt(lognorm,1.0)"] = [float(x) for x in rs.skew_kurt("lognorm", 1.0)]
    fp["_dist(t_5)"] = list(rs._dist("t_5"))
    fp["_kurt_tick(last_skew_rung)"] = rs._kurt_tick(rs.SKEW_RUNGS[-1])
    fp["n_nsweep_cells"] = len(rs.build_nsweep_cells())
    fp["n_ladder_cells"] = len(rs.build_ladder_cells())
    fp["fmt_rate"] = rs.fmt_rate({"alpha_hat": 0.0671, "liberal_ci": True})

    spec = rs.nsweep_spec("T5_bartlett", "skewed", 50)
    stats, rule = rs.gen_cell("T5_bartlett", spec, rs.cell_seed("nsweep", "skewed"), 500)
    fp["gen_cell.bartlett_skewed_n50_N500"] = digest_array(stats)
    fp["reject_rate.bartlett_skewed_n50_N500"] = rs.reject_rate(stats, rule, 0.05)

    rng = np.random.default_rng(11)
    groups = [rng.normal(size=(64, 12)), rng.normal(size=(64, 15)), rng.normal(size=(64, 9))]
    for name, fn in (("stat_t", rs.stat_t), ("stat_anova", rs.stat_anova),
                     ("stat_kruskal", rs.stat_kruskal), ("stat_bartlett", rs.stat_bartlett),
                     ("stat_fligner", rs.stat_fligner), ("stat_median", rs.stat_median)):
        fp[name] = digest_array(fn(groups))
    fp["stat_sign"] = digest_array(rs.stat_sign(groups, 0.0))
    fp["stat_gof"] = digest_array(rs.stat_gof(np.array([[1, 2, 3]]), np.array([2.0, 2.0, 2.0])))
    fixed = np.linspace(-3, 3, 101)
    fp["reject_rate.rules"] = [
        rs.reject_rate(fixed, ("chi2", 2), 0.05),
        rs.reject_rate(fixed, ("f", 2, 10), 0.05),
        rs.reject_rate(fixed, ("t", 10), 0.05),
        rs.reject_rate(fixed, ("norm",), 0.05),
        rs.reject_rate(np.arange(0, 21), ("binom", 20, 0.5), 0.05),
    ]
    fp["aggregate_nsweep"] = rs.aggregate_nsweep(
        [{"scenario": "s", "test": "T1_t_test", "n": 20}],
        [{"alpha_hat": 0.05}])

    # ---- rev_power.py -------------------------------------------------------
    fp["build_design.len"] = len(rp.build_design())
    fp["build_design.head"] = rp.build_design()[:2]
    fp["fmt_ci"] = rp.fmt_ci((0.12345, 0.98765))
    fp["scale_rows.empty"] = rp.scale_rows([], 50, 10_000)

    # ---- core.py severity hooks --------------------------------------------
    spec_norm = {"dist": ("norm", None), "ns": [20]}
    fp["hook_kpss"] = digest_array(core._hook_kpss(spec_norm, 20, np.random.default_rng(7)))
    fp["hook_hoeffding"] = digest_array(
        core._hook_hoeffding(spec_norm, 20, np.random.default_rng(7)))
    fp["hook_interaction"] = digest_array(
        core._hook_interaction(spec_norm, 2, np.random.default_rng(1)))
    fp["SEVERITY_HOOKS"] = sorted(core.SEVERITY_HOOKS)
    fp["TESTS"] = sorted(core.TESTS)

    # ---- entry points exist and are callable --------------------------------
    fp["has_main"] = {m.__name__: callable(getattr(m, "main", None))
                      for m in (rs, rp, rse, rad, vnp, vm)}

    # ---- verify_new_tests_power helpers -------------------------------------
    fp["vnp.wilson"] = [float(x) for x in vnp.wilson(437, 10_000)]
    rate, ci = vnp.rate_at(np.array([0.0, 1.0, 2.0, 3.0]), 1.0)
    fp["vnp.rate_at"] = [float(rate), [float(x) for x in ci]]

    # ---- read-only verification scripts, run as subprocesses ----------------
    for script in ("verify_manuscript.py", "rev_study1_ext.py"):
        cmd = [sys.executable, os.path.join(ROOT, "code", script)]
        if script == "rev_study1_ext.py":
            cmd.append("--validate")
        out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout
        fp[f"stdout:{script}"] = hashlib.sha256(out.encode()).hexdigest()[:24]

    blob = json.dumps(fp, sort_keys=True, default=str).encode()
    print(json.dumps(fp, indent=1, sort_keys=True, default=str)[:1800])
    print("\nfingerprint SHA-256:", hashlib.sha256(blob).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
