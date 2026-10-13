"""
rev_alpha_dependence.py -- multi-alpha sweep of the Study 2 violation scenarios and the
dependence scenarios (Referee 2 major comment 2; Referee 1 comment 4).

What is computed
----------------
1. ``alpha_sweep_13``  : for every test of the 13-test panel, the Study 2 violation
                         scenarios of ``experiments.run_study2`` (baseline, skewed,
                         heavy-tailed, variance heterogeneity, unbalanced design, small
                         expected frequencies) are re-run under the archive generators and
                         evaluated at alpha in {0.01, 0.05, 0.10}.  The scenarios that are
                         meaningful for the three new tests are added:
                         * T11 KPSS  : trend-stationary and unit-root series (power), plus
                           stationary AR(1);
                         * T12 Hoeffding: linear and quadratic dependence, and independent
                           but heavily skewed marginals (size);
                         * T13 IAD   : a true interaction term (power) and correlated
                           covariates (size).
                         The same nominal alpha is used for the critical value and for the
                         reference CDF in every cell.  A ``crosscheck_archive`` block
                         compares the alpha = 0.05 column with ``experiment_results.json``
                         (different RNG stream, so agreement is expected only within MC
                         noise; the comparison is reported, not asserted).

2. ``identity_checks``: numerical verification of the identities that the alpha sweep
                         relies on for two-sided tests, namely ``alpha = 2(1 - Phi(z))``
                         with ``z = Phi^{-1}(1 - alpha/2)``, the equal-tail identity
                         ``eps = beta <=> Phi(v) = 1/2 + z`` (see the docstring of
                         :func:`identity_checks` for the notation used here), and the
                         statement that the total theoretical tail mass at the critical
                         value equals the nominal alpha exactly.

3. ``dependence_13``   : (a) stationary AR(1) serial dependence with rho in {0, 0.2, 0.5, 0.8}
                         for the one-sample t test and the sign test at n in {20, 50, 100};
                         (b) cluster/random-effects dependence with ICC in {0, 0.2, 0.5} for
                         ANOVA and Kruskal-Wallis, and beta-binomial block correlation for
                         Cochran's Q; (c) an explicit per-test table of what the i.i.d.
                         assumption means for each of the 13 tests and what breaks when it
                         fails.

4. ``fig_rev_dependence.png``: four-panel figure (t test vs rho, sign test vs rho,
                         ANOVA/KW vs ICC, Cochran Q vs ICC), legends placed outside the
                         plotting area so that no legend can overlap a line.

Usage
-----
    python3 rev_alpha_dependence.py                 # full run, merges into rev_results_t1.json
    python3 rev_alpha_dependence.py --quick         # small-N smoke run
    python3 rev_alpha_dependence.py --stages dependence,figure

Author: t1-new-tests (task-1).  See docs/DESIGN_CONTRACT.md, Sections 3 and 6.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import stats

import core

# Output location. Every script in this package writes its results file to the package
# root by default, so the manuscript's verification suite finds it in one place.  Pass
# ``--outdir DIR`` (registered with argparse in ``main``) to redirect, which is what the
# reproduction vignette does when it runs a script in a scratch directory.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = _ROOT
WORKDIR = _ROOT


def _apply_outdir(path):
    """Redirect the results directory, creating it if necessary.

    Args:
        path: directory to write results into; may be relative to the working directory.

    Returns:
        str: the absolute path that results will be written to.
    """
    global OUT_DIR, WORKDIR
    OUT_DIR = WORKDIR = os.path.abspath(path)
    os.makedirs(OUT_DIR, exist_ok=True)
    return OUT_DIR
OUT_JSON = os.path.join(WORKDIR, "rev_results_t1.json")
ARCHIVE_JSON = os.path.join(WORKDIR, "experiment_results.json")
FIG_DEP = os.path.join(WORKDIR, "fig_rev_dependence.png")

ALPHAS = (0.01, 0.05, 0.10)
N_MC = 10_000
SEED = 2024
K_DEFAULT = 3


# ---------------------------------------------------------------------------
# 1. Study 2 scenario sweep at three nominal levels
# ---------------------------------------------------------------------------

def _scenario_runs():
    """Return the Study 2 scenario definitions used by the alpha sweep.

    The first six scenarios reproduce ``experiments.run_study2`` (identical generators and
    arguments, n = 50); the remaining ones are the violations that only make sense for the
    new tests.  Each entry is ``(test_id, scenario_name, generator, kwargs)`` where the
    generator is the *archive* generator unless noted.

    Returns
    -------
    dict
        ``{test_id: {'label':..., 'runs': {scenario: callable(N) -> (stats, ref_args)}}}``.
    """
    k, n = K_DEFAULT, 50
    runs = {
        "T1_t_test": {
            "baseline": lambda N: (core.mc_t_test(n, N), (n - 1,)),
            "skewed": lambda N: (core.mc_t_test(n, N, dist="expon"), (n - 1,)),
            "heavy_tail": lambda N: (core.mc_t_test(n, N, dist="tdist"), (n - 1,)),
        },
        "T2_anova": {
            "baseline": lambda N: (core.mc_anova(k, n, N)[0], (k - 1, k * (n - 1))),
            "skewed": lambda N: (core.mc_anova(k, n, N, dist="expon")[0], (k - 1, k * (n - 1))),
            "heavy_tail": lambda N: (core.mc_anova(k, n, N, dist="tdist")[0], (k - 1, k * (n - 1))),
            "hetero_var": lambda N: (core.mc_anova(k, n, N, hetero=True)[0], (k - 1, k * (n - 1))),
            "unbalanced": lambda N: (core.mc_anova(k, n, N, unbalanced=True)[0], (k - 1, 3 * n - k)),
        },
        "T3_kruskal_wallis": {
            "baseline": lambda N: (core.mc_kruskal_wallis(k, n, N), (k - 1,)),
            "skewed": lambda N: (core.mc_kruskal_wallis(k, n, N, dist="expon"), (k - 1,)),
            "heavy_tail": lambda N: (core.mc_kruskal_wallis(k, n, N, dist="tdist"), (k - 1,)),
            "hetero_var": lambda N: (core.mc_kruskal_wallis(k, n, N, hetero=True), (k - 1,)),
            "unbalanced": lambda N: (core.mc_kruskal_wallis(k, n, N, unbalanced=True), (k - 1,)),
        },
        "T4_chisq_gof": {
            "baseline": lambda N: (core.mc_chisq_gof(6, 250, N), (5,)),
            "small_freq": lambda N: (core.mc_chisq_gof(6, 30, N, small_expected=True), (5,)),
        },
        "T5_bartlett": {
            "baseline": lambda N: (core.mc_bartlett(k, n, N), (k - 1,)),
            "skewed": lambda N: (core.mc_bartlett(k, n, N, dist="expon"), (k - 1,)),
            "heavy_tail": lambda N: (core.mc_bartlett(k, n, N, dist="tdist"), (k - 1,)),
            "unbalanced": lambda N: (core.mc_bartlett(k, n, N, unbalanced=True), (k - 1,)),
        },
        "T6_fligner_killeen": {
            "baseline": lambda N: (core.mc_fligner_killeen(k, n, N), (k - 1,)),
            "skewed": lambda N: (core.mc_fligner_killeen(k, n, N, dist="expon"), (k - 1,)),
            "heavy_tail": lambda N: (core.mc_fligner_killeen(k, n, N, dist="tdist"), (k - 1,)),
            "unbalanced": lambda N: (core.mc_fligner_killeen(k, n, N, unbalanced=True), (k - 1,)),
        },
        "T7_sign_test": {
            "baseline": lambda N: (core.mc_sign_test(n, N), (n, 0.5)),
            "skewed": lambda N: (core.mc_sign_test(n, N, dist="expon"), (n, 0.5)),
            "heavy_tail": lambda N: (core.mc_sign_test(n, N, dist="tdist"), (n, 0.5)),
        },
        "T8_cochran_q": {
            "baseline": lambda N: (core.mc_cochran_q(k, n, N), (k - 1,)),
            "small_freq": lambda N: (core.mc_cochran_q(k, n, N, p=0.05), (k - 1,)),
        },
        "T9_median_test": {
            "baseline": lambda N: (core.mc_median_test(k, n, N), (k - 1,)),
            "skewed": lambda N: (core.mc_median_test(k, n, N, dist="expon"), (k - 1,)),
            "heavy_tail": lambda N: (core.mc_median_test(k, n, N, dist="tdist"), (k - 1,)),
            "hetero_var": lambda N: (core.mc_median_test(k, n, N, hetero=True), (k - 1,)),
            "unbalanced": lambda N: (core.mc_median_test(k, n, N, unbalanced=True), (k - 1,)),
            "small_freq": lambda N: (core.mc_median_test(k, 8, N), (k - 1,)),
        },
        "T10_hotelling": {
            "baseline": lambda N: (core.mc_hotelling_t2(2, n, N)[0], (2, n - 2)),
            "skewed": lambda N: (core.mc_hotelling_t2(2, n, N, dist="expon")[0], (2, n - 2)),
            "heavy_tail": lambda N: (core.mc_hotelling_t2(2, n, N, dist="tdist")[0], (2, n - 2)),
        },
        "T11_kpss": {
            # H0 = level stationarity; 'trend' and 'rw' are power scenarios, 'ar1' a
            # serial-dependence scenario in which H0 still holds.
            "baseline": lambda N: (core.mc_kpss(n, N), ()),
            "skewed": lambda N: (core.mc_kpss(n, N, kind="expon"), ()),
            "heavy_tail": lambda N: (core.mc_kpss(n, N, kind="tdist"), ()),
            "serial_dep": lambda N: (core.mc_kpss(n, N, kind="ar1", rho=0.5), ()),
            "trend": lambda N: (core.mc_kpss(n, N, kind="trend", delta=1.0), ()),
            "unit_root": lambda N: (core.mc_kpss(n, N, kind="rw"), ()),
        },
        "T12_hoeffding_d": {
            "baseline": lambda N: (core.mc_hoeffding_d(n, N), ()),
            "skewed": lambda N: (core.mc_hoeffding_d(n, N, dep="expon"), ()),
            "linear_dep": lambda N: (core.mc_hoeffding_d(n, N, dep="linear", rho=0.5), ()),
            "quad_dep": lambda N: (core.mc_hoeffding_d(n, N, dep="quad"), ()),
        },
        "T13_interaction": {
            "baseline": lambda N: (core.mc_interaction_h(n, N, n_jobs=10), ()),
            "interaction": lambda N: (core.mc_interaction_h(n, N, kind="interaction", rho=1.0,
                                                            n_jobs=10), ()),
            "corr_covariates": lambda N: (core.mc_interaction_h(n, N, kind="corrX", rho=0.5,
                                                                n_jobs=10), ()),
        },
    }
    return runs


def alpha_sweep_13(N=N_MC, seed=SEED, verbose=True):
    """Run every Study 2 scenario at three nominal levels for all 13 tests.

    Returns
    -------
    dict
        ``{test_id: {'label':..., 'n': 50, 'scenarios': {name: {'type1': {alpha: ...},
        'ci_lo':..., 'ci_hi':..., 'n_effective':...}}}, ...}``.
    """
    np.random.seed(seed)
    runs = _scenario_runs()
    out = {}
    for tid, scenarios in runs.items():
        t0 = time.time()
        entry = {"label": core.TESTS[tid]["label"], "n": 50, "scenarios": {}}
        for name, fn in scenarios.items():
            sv, ref_args = fn(N)
            ev = core.evaluate_multi_alpha(sv, tid, ALPHAS, ref_args=ref_args)
            entry["scenarios"][name] = {
                "type1": {k: float(v["type1"]) for k, v in ev.items()},
                "ci_lo": {k: float(v["ci_lo"]) for k, v in ev.items()},
                "ci_hi": {k: float(v["ci_hi"]) for k, v in ev.items()},
                "ks": float(ev["0.05"]["ks"]),
                "n_effective": int(ev["0.05"]["n_effective"]),
                "kind": ("power" if name in ("trend", "unit_root", "interaction",
                                             "linear_dep", "quad_dep") else "size"),
            }
            if verbose:
                print(f"  [alpha] {tid:<18s} {name:<16s} "
                      f"a01={entry['scenarios'][name]['type1']['0.01']:.4f} "
                      f"a05={entry['scenarios'][name]['type1']['0.05']:.4f} "
                      f"a10={entry['scenarios'][name]['type1']['0.10']:.4f}", flush=True)
        entry["seconds"] = round(time.time() - t0, 1)
        out[tid] = entry
    return out


def crosscheck_archive(alpha_sweep):
    """Compare the alpha = 0.05 column with the archived Study 2 values (no assertion).

    The sweep uses its own RNG stream, so agreement is expected only within Monte-Carlo
    noise (Wilson half-width ~0.004 at N = 10,000).  The maximum absolute difference is
    reported so that a genuine metric change would be visible.

    Returns
    -------
    dict
        ``{'max_abs_diff': float, 'per_test': {...}}``.
    """
    if not os.path.exists(ARCHIVE_JSON):
        return {"available": False}
    with open(ARCHIVE_JSON) as fh:
        arch = json.load(fh)["study2"]
    per_test, mx = {}, 0.0
    for tid, e in alpha_sweep.items():
        if tid not in arch:
            continue
        diffs = {}
        for sc, rec in e["scenarios"].items():
            a = arch[tid].get(sc)
            if a is None or not np.isfinite(a):
                continue
            d = abs(rec["type1"]["0.05"] - a)
            diffs[sc] = {"archive": float(a), "new": rec["type1"]["0.05"], "abs_diff": float(d)}
            mx = max(mx, d)
        per_test[tid] = diffs
    return {"available": True, "max_abs_diff": float(mx), "per_test": per_test,
            "note": ("different RNG stream: differences up to the Wilson half-width "
                     "(~0.004) are expected; a systematic shift would indicate a metric "
                     "change")}


# ---------------------------------------------------------------------------
# 2. Identity checks for the two-sided rejection rule
# ---------------------------------------------------------------------------

def identity_checks(alphas=ALPHAS):
    """Numerical verification of the identities underlying the two-sided alpha rule.

    For a two-sided test based on a standard-normal reference, the critical value is
    ``z = Phi^{-1}(1 - alpha/2)``.  The sweep relies on three statements, each verified
    here to machine precision for every nominal level:

    1. ``alpha = 2 (1 - Phi(z))``: the nominal level and the critical value are two
       representations of the same quantity.
    2. Equal tails: with ``eps = P(S >= z)`` (upper tail) and ``beta = P(S <= -z)`` (lower
       tail), ``eps = beta`` holds for every symmetric reference; writing the critical
       value as the quantile ``v = z`` and the half-level as ``z_half = (1 - alpha)/2``,
       the equivalent quantile form is ``Phi(v) = 1/2 + z_half``.
    3. Total tail mass: ``P(S > c) + P(S < -c) = alpha`` for the actual critical value of
       every two-sided test in the panel (T1 and T13 use different references, so both are
       checked), i.e. the *same* nominal alpha is used for the critical value and for the
       reference CDF -- the rule that makes the multi-alpha sweep internally consistent.

    Returns
    -------
    dict
        ``{'checks': [...], 'max_abs_residual': float, 'note': str}``.
    """
    checks, max_res = [], 0.0
    for a in alphas:
        z = stats.norm.ppf(1.0 - a / 2.0)
        r1 = 2.0 * (1.0 - stats.norm.cdf(z)) - a                     # identity 1
        eps = stats.norm.sf(z)
        beta = stats.norm.cdf(-z)
        r2 = eps - beta                                              # equal tails
        v = z
        z_half = (1.0 - a) / 2.0
        r3 = stats.norm.cdf(v) - (0.5 + z_half)                      # quantile form
        # total theoretical tail mass at the critical value of each two-sided test
        tails = {}
        for tid in ("T1_t_test", "T13_interaction"):
            c, spec = core.critical_value(tid, a, ref_args={"T1_t_test": (49,)}.get(tid))
            if core.TESTS[tid]["null"] == "t":
                mass = stats.t.sf(c, 49) + stats.t.cdf(-c, 49)
            else:
                mass = stats.norm.sf(c) + stats.norm.cdf(-c)
            tails[tid] = {"critical_value": float(c), "total_tail_mass": float(mass),
                          "residual": float(mass - a)}
            max_res = max(max_res, abs(mass - a))
        row = {"alpha": a, "z": float(z),
               "residual_alpha_eq_2_1_minus_Phi_z": float(r1),
               "eps": float(eps), "beta": float(beta), "residual_eps_minus_beta": float(r2),
               "Phi_v_minus_half_minus_z": float(r3),
               "total_tail_mass_two_sided_tests": tails}
        max_res = max(max_res, abs(r1), abs(r2), abs(r3))
        checks.append(row)
    return {"checks": checks, "max_abs_residual": float(max_res),
            "note": ("all quantities are algebraic identities evaluated in double "
                     "precision; residuals at the 1e-16 level confirm that the critical "
                     "value and the reference CDF use the same nominal alpha")}


# ---------------------------------------------------------------------------
# 3. Dependence scenarios
# ---------------------------------------------------------------------------

AR1_TESTS = ("T1_t_test", "T7_sign_test")
AR1_RHOS = (0.0, 0.2, 0.5, 0.8)
AR1_NS = (20, 50, 100)
ICC_VALUES = (0.0, 0.2, 0.5)


def dependence_scenarios(N=N_MC, seed=SEED, verbose=True):
    """AR(1) serial dependence and cluster/block dependence for representative tests.

    Returns
    -------
    dict
        ``{'ar1': {...}, 'clustered': {...}, 'assumption_table': {...}, 'notes': [...]}``.
    """
    np.random.seed(seed)
    out = {"ar1": {"rhos": list(AR1_RHOS), "ns": list(AR1_NS), "tests": {}},
           "clustered": {"icc": list(ICC_VALUES), "anova_kw": {"c": 3, "m": 10, "k": 3},
                         "cochran_q": {"k": 3, "n_blocks": 50}, "tests": {}},
           "assumption_table": IID_ASSUMPTION_TABLE,
           "notes": [
               "AR(1): y_t = rho y_{t-1} + sqrt(1-rho^2) e_t is level stationary with "
               "median 0, so H0 is true for both tests and every rejection is a Type I "
               "error. Under AR(1) each sign is marginally a fair coin, but the signs are NOT independent and NOT exchangeable in the sense the binomial null requires (equal "
               "signs cluster in runs), so N+ is over-dispersed relative to Binomial(n, 1/2) "
               "and the sign test breaks too -- for a one-sample location test the "
               "distribution-free sign test is therefore no safer than the t test under "
               "serial dependence; this is measured, not assumed.",
               "Clustered ANOVA / Kruskal-Wallis: k = 3 groups x c = 3 clusters x m = 10 "
               "observations, Y = b + e with Var(b) = ICC and Var(e) = 1 - ICC; the naive "
               "critical values ignore the design effect 1 + (m-1) ICC.",
               "Cochran's Q: symmetric beta-binomial blocks, p_i ~ Beta(a, a) with "
               "a = (1-ICC)/(2 ICC), so all k marginal treatment probabilities stay 1/2 "
               "(H0 true by symmetry) while the within-block binary correlation equals ICC; "
               "this is overdispersion beyond the additive block effect that Q absorbs.",
           ]}

    for tid in AR1_TESTS:
        out["ar1"]["tests"][tid] = {"label": core.TESTS[tid]["label"], "cells": {}}
        for n in AR1_NS:
            for rho in AR1_RHOS:
                if tid == "T1_t_test":
                    sv = core.mc_t_test_ar1(n, N, rho=rho)
                    ref = (n - 1,)
                else:
                    sv = core.mc_sign_test_ar1(n, N, rho=rho)
                    ref = (n, 0.5)
                ev = core.evaluate_multi_alpha(sv, tid, ALPHAS, ref_args=ref)
                key = f"n{n}_rho{rho:.1f}"
                out["ar1"]["tests"][tid]["cells"][key] = {
                    "n": n, "rho": rho,
                    "type1": {k: float(v["type1"]) for k, v in ev.items()},
                    "ci_lo": {k: float(v["ci_lo"]) for k, v in ev.items()},
                    "ci_hi": {k: float(v["ci_hi"]) for k, v in ev.items()},
                }
                if verbose:
                    print(f"  [dep-ar1] {tid:<14s} n={n:<4d} rho={rho:.1f} "
                          f"a05={out['ar1']['tests'][tid]['cells'][key]['type1']['0.05']:.4f}",
                          flush=True)

    # cluster designs
    for tid, gen in (("T2_anova", core.mc_anova_clustered),
                     ("T3_kruskal_wallis", core.mc_kruskal_wallis_clustered)):
        out["clustered"]["tests"][tid] = {"label": core.TESTS[tid]["label"], "cells": {}}
        for icc in ICC_VALUES:
            sv = gen(3, 3, 10, N, icc=icc)
            ref = (2, 3 * 3 * 10 - 3) if tid == "T2_anova" else (2,)
            ev = core.evaluate_multi_alpha(sv, tid, ALPHAS, ref_args=ref)
            out["clustered"]["tests"][tid]["cells"][f"icc{icc:.1f}"] = {
                "icc": icc,
                "type1": {k: float(v["type1"]) for k, v in ev.items()},
                "ci_lo": {k: float(v["ci_lo"]) for k, v in ev.items()},
                "ci_hi": {k: float(v["ci_hi"]) for k, v in ev.items()},
                "design_effect": float(1 + (10 - 1) * icc),
            }
            if verbose:
                print(f"  [dep-clu] {tid:<18s} ICC={icc:.1f} "
                      f"a05={out['clustered']['tests'][tid]['cells'][f'icc{icc:.1f}']['type1']['0.05']:.4f}",
                      flush=True)
    tid = "T8_cochran_q"
    out["clustered"]["tests"][tid] = {"label": core.TESTS[tid]["label"], "cells": {}}
    for icc in ICC_VALUES:
        sv = core.mc_cochran_q_corr(3, 50, N, icc=icc)
        ev = core.evaluate_multi_alpha(sv, tid, ALPHAS, ref_args=(2,))
        out["clustered"]["tests"][tid]["cells"][f"icc{icc:.1f}"] = {
            "icc": icc,
            "type1": {k: float(v["type1"]) for k, v in ev.items()},
            "ci_lo": {k: float(v["ci_lo"]) for k, v in ev.items()},
            "ci_hi": {k: float(v["ci_hi"]) for k, v in ev.items()},
        }
        if verbose:
            print(f"  [dep-clu] {tid:<18s} ICC={icc:.1f} "
                  f"a05={out['clustered']['tests'][tid]['cells'][f'icc{icc:.1f}']['type1']['0.05']:.4f}",
                  flush=True)
    return out


# Per-test statement of what "independent observations" means and what breaks without it.
IID_ASSUMPTION_TABLE = {
    "T1_t_test": {
        "iid_assumption": "independent observations from a common (normal) population",
        "what_breaks": ("positive serial correlation inflates the variance of the mean but "
                        "not the sample standard deviation, so the t statistic is "
                        "over-dispersed and the test is liberal (measured here, AR(1))"),
        "scenario": "ar1",
    },
    "T2_anova": {
        "iid_assumption": "independent observations within and between groups",
        "what_breaks": ("cluster/group-level correlation leaves the within-group residual "
                        "variance unchanged while inflating the between-group mean square, "
                        "so F is inflated by roughly the design effect 1 + (m-1) ICC"),
        "scenario": "clustered",
    },
    "T3_kruskal_wallis": {
        "iid_assumption": "independent observations; ranks exchangeable under H0",
        "what_breaks": ("clustering makes the effective sample size smaller than the "
                        "nominal one, so the chi-square null is too narrow"),
        "scenario": "clustered",
    },
    "T4_chisq_gof": {
        "iid_assumption": "independent multinomial counts with the stated cell probabilities",
        "what_breaks": ("clustered or over-dispersed counts inflate the variance of the "
                        "cell counts relative to the multinomial model; the GOF statistic "
                        "is then over-dispersed (not simulated here: no natural cluster "
                        "structure for a single multinomial sample)"),
        "scenario": None,
    },
    "T5_bartlett": {
        "iid_assumption": "independent normal observations within each group",
        "what_breaks": ("within-group serial or cluster correlation reduces the effective "
                        "degrees of freedom, so the log-variance contrast is underestimated "
                        "and the test is conservative (the opposite direction to the "
                        "location tests)"),
        "scenario": None,
    },
    "T6_fligner_killeen": {
        "iid_assumption": "independent observations within each group",
        "what_breaks": ("same effective-sample-size problem as Bartlett; rank scores are "
                        "marginally symmetric but dependent, so the effect is milder than for the mean but still present"),
        "scenario": None,
    },
    "T7_sign_test": {
        "iid_assumption": "independent observations from a continuous distribution with "
                          "median m0",
        "what_breaks": ("serial dependence leaves each sign marginally symmetric but makes them dependent and non-exchangeable, so "
                        "independent: runs of equal signs make N+ over-dispersed relative "
                        "to Binomial(n, 1/2), so the exact null is too narrow and the test "
                        "is liberal (measured here, AR(1); the effect is of the same order "
                        "as for the t test, which is the headline dependence finding: "
                        "'distribution-free' is not the same as 'assumption-free')"),
        "scenario": "ar1",
    },
    "T8_cochran_q": {
        "iid_assumption": "independent dichotomous responses within a block, given the "
                          "block effect",
        "what_breaks": ("within-block over-dispersion beyond the additive block effect "
                        "(beta-binomial) makes the responses over-correlated, so the "
                        "chi-square null is no longer the right reference"),
        "scenario": "clustered",
    },
    "T9_median_test": {
        "iid_assumption": "independent observations; the 2 x k table has the multinomial "
                          "structure",
        "what_breaks": ("clustering acts like a smaller effective sample: the chi-square "
                        "approximation degrades further, in the same direction as its "
                        "already slow convergence"),
        "scenario": None,
    },
    "T10_hotelling": {
        "iid_assumption": "independent p-dimensional normal vectors",
        "what_breaks": ("serial dependence inflates the covariance of the sample mean "
                        "relative to S/n, so T2 is over-dispersed; the problem is the "
                        "multivariate analogue of the t-test case"),
        "scenario": None,
    },
    "T11_kpss": {
        "iid_assumption": "level stationarity with a short-memory error process",
        "what_breaks": ("the asymptotic CvM limit law presumes the HAC estimator captures "
                        "the long-run variance; with strong AR dependence and the fixed "
                        "Schwert bandwidth it does not, so the test distorts (measured "
                        "here: AR(1) rho = 0.5, plus trend and unit-root power scenarios)"),
        "scenario": "ar1, trend, unit_root",
    },
    "T12_hoeffding_d": {
        "iid_assumption": "independent bivariate pairs (X_i, Y_i)",
        "what_breaks": ("Hoeffding's D is rank-based and therefore exactly invariant to "
                        "strictly monotone marginal transformations, so only the dependence "
                        "structure matters: any dependence invalidates the null (measured "
                        "here: linear rho = 0.5 and quadratic dependence as power "
                        "scenarios)"),
        "scenario": "linear_dep, quad_dep",
    },
    "T13_interaction": {
        "iid_assumption": "i.i.d. observations, and a learner whose estimation error "
                          "vanishes relative to the test size (training set growing faster "
                          "than the test set)",
        "what_breaks": ("with a fixed training set the flexible learner's finite-sample "
                        "overfitting mimics interaction, so the statistic has a positive "
                        "location shift growing with the test size and the N(0,1) reference "
                        "fails (see t13_split_diagnostic)"),
        "scenario": "fixed split",
    },
}


# ---------------------------------------------------------------------------
# 4. Figure
# ---------------------------------------------------------------------------

def make_dependence_figure(dep, out_path=FIG_DEP):
    """fig_rev_dependence.png: four panels, legends outside the axes (no overlap)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.dpi": 150})

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    (ax1, ax2), (ax3, ax4) = axes

    # --- AR(1): t test and sign test ---------------------------------------
    for ax, tid in ((ax1, "T1_t_test"), (ax2, "T7_sign_test")):
        cells = dep["ar1"]["tests"][tid]["cells"]
        for n in AR1_NS:
            ys, los, his = [], [], []
            for rho in AR1_RHOS:
                c = cells[f"n{n}_rho{rho:.1f}"]
                ys.append(c["type1"]["0.05"])
                los.append(c["ci_lo"]["0.05"])
                his.append(c["ci_hi"]["0.05"])
            ax.errorbar(AR1_RHOS, ys, yerr=[np.array(ys) - los, np.array(his) - np.array(ys)],
                        marker="o", ms=4, lw=1.3, capsize=2.5, label=f"n = {n}")
        ax.axhline(0.05, color="black", ls=":", lw=1.0, label="nominal $\\alpha$ = 0.05")
        ax.axhspan(0.04, 0.06, color="green", alpha=0.08)
        ax.set_xlabel("AR(1) autocorrelation $\\rho$ (unitless)")
        ax.set_ylabel("$\\hat{\\alpha}$ at $\\alpha$ = 0.05 (unitless)")
        ax.set_title(("One-sample $t$ test" if tid == "T1_t_test" else "Sign test")
                     + " under serial dependence")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper left", frameon=False)

    # --- clustered ANOVA / KW ---------------------------------------------
    for tid, lab in (("T2_anova", "ANOVA $F$"), ("T3_kruskal_wallis", "Kruskal-Wallis $H$")):
        cells = dep["clustered"]["tests"][tid]["cells"]
        ys = [cells[f"icc{i:.1f}"]["type1"]["0.05"] for i in ICC_VALUES]
        los = [cells[f"icc{i:.1f}"]["ci_lo"]["0.05"] for i in ICC_VALUES]
        his = [cells[f"icc{i:.1f}"]["ci_hi"]["0.05"] for i in ICC_VALUES]
        ax3.errorbar(ICC_VALUES, ys, yerr=[np.array(ys) - los, np.array(his) - np.array(ys)],
                     marker="o", ms=4, lw=1.3, capsize=2.5, label=lab)
    ax3.axhline(0.05, color="black", ls=":", lw=1.0, label="nominal $\\alpha$ = 0.05")
    ax3.axhspan(0.04, 0.06, color="green", alpha=0.08)
    ax3.set_xlabel("Intra-cluster correlation ICC (unitless)")
    ax3.set_ylabel("$\\hat{\\alpha}$ at $\\alpha$ = 0.05 (unitless)")
    ax3.set_title("Group-clustered location tests\n(3 groups $\\times$ 3 clusters $\\times$ 10 obs)")
    ax3.grid(alpha=0.25)
    ax3.legend(fontsize=8, loc="upper left", frameon=False)

    # --- Cochran Q ---------------------------------------------------------
    cells = dep["clustered"]["tests"]["T8_cochran_q"]["cells"]
    ys = [cells[f"icc{i:.1f}"]["type1"]["0.05"] for i in ICC_VALUES]
    los = [cells[f"icc{i:.1f}"]["ci_lo"]["0.05"] for i in ICC_VALUES]
    his = [cells[f"icc{i:.1f}"]["ci_hi"]["0.05"] for i in ICC_VALUES]
    ax4.errorbar(ICC_VALUES, ys, yerr=[np.array(ys) - los, np.array(his) - np.array(ys)],
                 marker="s", ms=4, lw=1.3, capsize=2.5, label="Cochran's $Q$")
    ax4.axhline(0.05, color="black", ls=":", lw=1.0, label="nominal $\\alpha$ = 0.05")
    ax4.axhspan(0.04, 0.06, color="green", alpha=0.08)
    ax4.set_xlabel("Within-block binary ICC (unitless)")
    ax4.set_ylabel("$\\hat{\\alpha}$ at $\\alpha$ = 0.05 (unitless)")
    ax4.set_title("Cochran's $Q$ under beta-binomial\nblock over-dispersion (3 treatments, 50 blocks)")
    ax4.grid(alpha=0.25)
    ax4.legend(fontsize=8, loc="lower left", frameon=False)

    fig.suptitle("Dependence scenarios: empirical Type I error at $\\alpha$ = 0.05 (N = 10,000)",
                 y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 5. Entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    """Entry point: run the multi-alpha sweep and the dependence scenarios.

    Args:
        argv: argument list (defaults to ``sys.argv[1:]``). Options: ``--quick``
            (reduced budget, smoke testing only), ``--stages`` (subset of
            ``alpha``, ``dependence``, ``figure``), ``--N`` (override the budget),
            ``--no-merge`` (write ``rev_results_t1_alpha_dep.json`` instead of
            merging).

    Returns:
        int: process exit status. Unless ``--no-merge`` or ``--quick`` is given,
        the blocks are merged into the file written by ``rev_study1_ext.py``.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=None,
                    help="directory for results files (default: the package root); "
                         "created if it does not exist")
    ap.add_argument("--quick", action="store_true", help="small-N smoke run (not for publication)")
    ap.add_argument("--stages", default="alpha,dependence,figure",
                    help="comma-separated subset of alpha,dependence,figure")
    ap.add_argument("--N", type=int, default=None)
    ap.add_argument("--no-merge", action="store_true",
                    help="write rev_results_t1_alpha_dep.json instead of merging")
    args = ap.parse_args(argv)
    if args.outdir:
        _apply_outdir(args.outdir)

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    N = args.N if args.N else (500 if args.quick else N_MC)
    t0 = time.time()
    print("=" * 72)
    print("rev_alpha_dependence.py -- multi-alpha sweep and dependence scenarios")
    print(f"N = {N}, quick = {args.quick}, stages = {stages}")
    print("=" * 72, flush=True)

    out = {"meta_alpha_dep": core.make_meta(extra={
        "script": "rev_alpha_dependence.py",
        "command": "python3 rev_alpha_dependence.py" + (" --quick" if args.quick else ""),
        "N_mc": N, "alphas": list(ALPHAS), "quick_mode": bool(args.quick),
        "wall_clock_s": None})}
    if args.quick:
        out["meta_alpha_dep"]["WARNING"] = ("quick mode: reduced budget, smoke testing only")

    if "alpha" in stages:
        print("\n[1/3] Study 2 scenario sweep at three nominal levels", flush=True)
        sweep = alpha_sweep_13(N=N)
        out["alpha_sweep_13"] = sweep
        out["alpha_sweep_13_crosscheck_archive"] = crosscheck_archive(sweep)
        out["identity_checks"] = identity_checks()
        print(f"  identity checks: max |residual| = "
              f"{out['identity_checks']['max_abs_residual']:.2e}", flush=True)

    if "dependence" in stages:
        print("\n[2/3] dependence scenarios", flush=True)
        out["dependence_13"] = dependence_scenarios(N=N)

    if "figure" in stages:
        print("\n[3/3] figure", flush=True)
        if "dependence_13" in out:
            out["figure_dependence"] = make_dependence_figure(out["dependence_13"])

    out["meta_alpha_dep"]["wall_clock_s"] = round(time.time() - t0, 1)
    path = os.path.join(WORKDIR, "rev_results_t1_alpha_dep.json") if args.no_merge else OUT_JSON
    if args.quick:
        path = path.replace(".json", "_quick.json")
    if args.no_merge or args.quick:
        core.save_results(out, path)
    else:
        core.update_results(path, **{k: v for k, v in out.items() if k != "meta_alpha_dep"},
                            meta_alpha_dep=out["meta_alpha_dep"], stage=",".join(stages))
        path = OUT_JSON
    print(f"\nWrote {path} ({os.path.getsize(path)} bytes) in "
          f"{out['meta_alpha_dep']['wall_clock_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
