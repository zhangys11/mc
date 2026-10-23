"""Stage 1 diagnostics for the mechanism-aware calibration programme.

This is an EXPLORATORY prototype: it is *not* part of the committed
experiment pipeline and is *not* used by paper.md.  It implements the three
diagnostic quantities discussed for Route A:

    R_B : dispersion of asymptotic quadratic-form weights (shape mismatch)
    R_C : attainable-size error of a non-randomised discrete test
    R_A : leading Edgeworth CDF error of a standardised smooth statistic
          (analytic one-sample-t case only in this prototype)

and a retrospective table built from results/experiment_results.json.

Run
---
    python3 diagnostics.py

which prints the tables and writes results/diagnostics_stage1.md.
"""
from __future__ import annotations

import json
import math
import os
from math import comb

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results", "experiment_results.json")
OUT = os.path.join(ROOT, "results", "diagnostics_stage1.md")


# ============================================================================
# R_B: spectrum of a weighted chi-square reference law
# ============================================================================
def weight_spectrum(weights, baseline=1.0):
    """Diagnostics for weights $w_1,\\ldots,w_r$ of $\\sum_j w_j\\chi^2_{1,j}$.

    R_B is the coefficient of variation of the weights,

        R_B = r * sum(w^2) / (sum w)^2 - 1,

    which is 0 iff all weights are equal.  ``mean_shift`` is
    mean(w)/baseline - 1, the leading mean-scale mismatch of the quadratic
    form relative to the equal-weight reference (baseline=1 by default).
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.size == 0:
        raise ValueError("weights must be a non-empty 1-D array")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("weights must have positive sum")
    r = int(w.size)
    r_b = float(r * np.sum(w * w) / total ** 2 - 1.0)
    mean = total / r
    return dict(
        r=r,
        total=total,
        mean=mean,
        mean_shift=float(mean / baseline - 1.0),
        r_b=max(r_b, 0.0) if r_b > -1e-12 else r_b,
        max_min_ratio=float(w.max() / w.min()) if w.min() > 0 else math.inf,
        equal_weights=bool(np.allclose(w, w[0])),
        weights=[float(x) for x in w],
    )


def bartlett_weights(gamma2):
    """Per-contrast weights of Bartlett's statistic under excess kurtosis gamma2.

    The leading kurtosis distortion multiplies every chi-square weight by
    (1 + gamma2/2), so the spectrum is *equal* even when gamma2 != 0; what
    changes is the mean scale.  This is the counter-example to using R_B as
    the only quadratic-form diagnostic.
    """
    return np.array([1.0 + 0.5 * float(gamma2)])


# ============================================================================
# R_C: attainable-size error of discrete tests
# ============================================================================
def attainable_size_error_upper(support, pmf, alpha=0.05):
    """R_C for an upper-tail *conservative* rejection rule T >= c.

    A non-randomised exact test is conventionally calibrated from below, so
    the attainable size is the largest size not exceeding alpha (unlike the
    mid-p rule, which may exceed alpha).  `R_C` is alpha minus that size.
    """
    support = np.asarray(support, dtype=float)
    pmf = np.asarray(pmf, dtype=float)
    if support.ndim != 1 or support.size != pmf.size:
        raise ValueError("support and pmf must be 1-D arrays of equal length")
    if pmf.sum() <= 0:
        raise ValueError("pmf must have positive mass")
    pmf = pmf / pmf.sum()
    order = np.argsort(support)
    support = support[order]
    pmf = pmf[order]
    tail = np.cumsum(pmf[::-1])[::-1]          # P(T >= support[i])
    feasible = np.where(tail <= alpha)[0]
    if feasible.size:
        i = int(feasible[0])                   # smallest c with size <= alpha
        size = float(tail[i])
    else:
        i = 0
        size = 0.0
    return dict(
        R_C=float(alpha - size),
        attainable=size,
        threshold=float(support[i]),
        n_support=int(support.size),
    )


def sign_test_diagnostics(n, alpha=0.05):
    """Exact and mid-p attainable sizes for the two-sided sign test.

    Under H0 the number of positive signs is Bin(n, 1/2).  The exact rule
    rejects on {T <= c} U {T >= n-c}; the mid-p rule rejects when the mid-p
    value is at most alpha.
    """
    n = int(n)
    if n < 1:
        raise ValueError("n must be positive")
    support = np.arange(n + 1)
    pmf = stats.binom.pmf(support, n, 0.5)
    cdf = np.cumsum(pmf)

    # Exact rule: largest attainable size not exceeding alpha (conservative).
    best = None
    for c in range(n // 2):
        size = float(2.0 * cdf[c])
        if size <= alpha:
            if best is None or size > best[1]:
                best = (alpha - size, size, c)
    if best is None:
        best = (alpha, 0.0, 0)

    # two-sided mid-p: 2 * min(P(T<k)+P(T=k)/2, P(T>k)+P(T=k)/2)
    lower = np.where(support > 0, cdf[np.maximum(support - 1, 0)], 0.0)
    upper = 1.0 - cdf
    mid = 2.0 * np.minimum(lower + 0.5 * pmf, upper + 0.5 * pmf)
    reject = mid <= alpha
    mid_size = float(pmf[reject].sum())
    return dict(
        n=n,
        alpha=alpha,
        exact_size=float(best[1]),
        exact_R_C=float(best[0]),
        exact_c=int(best[2]),
        midp_size=mid_size,
        midp_R_C=float(abs(mid_size - alpha)),
        midp_reject_min=int(support[reject].min()) if reject.any() else None,
        midp_reject_max=int(support[reject].max()) if reject.any() else None,
    )


def _bounded_compositions(total, k, maxv):
    """Yield k-tuples of integers in [0, maxv] summing to total."""
    if k == 1:
        if 0 <= total <= maxv:
            yield (total,)
        return
    for first in range(min(maxv, total) + 1):
        for rest in _bounded_compositions(total - first, k - 1, maxv):
            yield (first,) + rest


def median_test_pmf(k, n):
    """Exact null distribution of the k-sample median-test statistic.

    Balanced design with n observations per group.  The grand-median cell
    counts are multivariate hypergeometric; the statistic is the one used in
    experiment.py (McCullagh-style quadratic form, upper tail only).
    """
    k = int(k)
    n = int(n)
    if k < 2 or n < 1:
        raise ValueError("k >= 2 and n >= 1 required")
    total = k * n
    above = total // 2 if total % 2 == 0 else (total - 1) // 2
    below = total - above
    if above == 0 or below == 0:
        raise ValueError("degenerate median split")
    denom = comb(total, above)
    pmf = {}
    for counts in _bounded_compositions(above, k, n):
        ways = 1
        for oj in counts:
            ways *= comb(n, oj)
        prob = ways / denom
        stat = (total ** 2 / (above * below)) * sum(
            (oj - n * above / total) ** 2 / n for oj in counts
        )
        pmf[stat] = pmf.get(stat, 0.0) + prob
    support = np.array(sorted(pmf))
    masses = np.array([pmf[s] for s in support])
    return support, masses


def median_test_diagnostics(k, n, alpha=0.05):
    support, pmf = median_test_pmf(k, n)
    rec = attainable_size_error_upper(support, pmf, alpha)
    rec.update(k=k, n=n, alpha=alpha)
    return rec


# ============================================================================
# R_A: leading Edgeworth CDF error (analytic one-sample-t case)
# ============================================================================
def t_edgeworth_coefficient(gamma1, x):
    """Leading coefficient c_A(x) for the one-sample t statistic.

    For iid data with skewness gamma1,
        P(T_n <= x) = Phi(x) + n^{-1/2} c_A(x) + O(n^{-1}),
        c_A(x) = gamma1 * (2 x^2 + 1) * phi(x) / 6.
    The coefficient is even in x, so for a two-sided test with symmetric
    critical values the O(n^{-1/2}) size error cancels; R_A below measures
    the CDF-level error, not the two-sided size error.
    """
    return float(gamma1) * (2.0 * x * x + 1.0) * stats.norm.pdf(x) / 6.0


def t_edgeworth_diagnostic(gamma1, n, x):
    """R_A = |c_A(x)| / sqrt(n) at a single quantile x."""
    c = t_edgeworth_coefficient(gamma1, x)
    return dict(
        x=float(x),
        gamma1=float(gamma1),
        coefficient=float(c),
        R_A=float(abs(c) / math.sqrt(n)),
    )


# ============================================================================
# Simulation-calibrated size-error scaling (T2 / T10)
# ============================================================================
def simulate_size_curve(test, parent="expon", n_grid=(20, 50, 100, 200),
                        N_mc=20000, alpha=0.05, seed=12345):
    """Simulate the nominal-reference size of T2/T10 and fit a power law.

    This is *simulation calibration under a specified parent*, not an
    estimator from the observed sample.  It is included to check the
    *regime* of the size error (O(n^{-1/2}) vs O(n^{-1})) before attempting
    a sample-based R_A for ratio statistics.
    """
    import experiment as ex
    import numpy as _np
    from scipy import stats as _st

    _np.random.seed(int(seed))
    sizes = []
    for n in n_grid:
        if test == "T2_anova":
            stat, df1, df2 = ex.mc_anova(3, int(n), int(N_mc), dist=parent)
            pvals = _st.f.sf(stat, df1, df2)
        elif test == "T10_hotelling":
            stat, df1, df2 = ex.mc_hotelling_t2(2, int(n), int(N_mc), dist=parent)
            pvals = _st.f.sf(stat, df1, df2)
        else:
            raise ValueError("test must be 'T2_anova' or 'T10_hotelling'")
        sizes.append(float((pvals <= alpha).mean()))
    errors = [s - alpha for s in sizes]
    logs = [(float(n), e) for n, e in zip(n_grid, errors) if e > 0]
    slope = None
    if len(logs) >= 2:
        x = _np.log([n for n, _ in logs])
        y = _np.log([e for _, e in logs])
        slope = float(_np.polyfit(x, y, 1)[0])
    return dict(test=test, parent=parent, n_grid=[float(n) for n in n_grid],
                N_mc=int(N_mc), alpha=float(alpha), sizes=sizes,
                errors=errors, slope=slope)


# ============================================================================
# Retrospective Stage 1 table from the committed results
# ============================================================================
def _load_results(path=RESULTS):
    with open(path) as fh:
        return json.load(fh)


def _observed_study2(results, test, scenario):
    return float(results["study2"][test][scenario])


def retrospective_table(results=None):
    """Return the Markdown text of the Stage 1 retrospective diagnostic table."""
    results = results or _load_results()
    L = []
    L.append("# Stage 1 diagnostic prototype (exploratory, not part of paper.md)")
    L.append("")
    L.append("Generated from `results/experiment_results.json`.  These diagnostics are "
             "prototypes for Route A, not validated selection rules.")
    L.append("")

    # --- Box's M weight spectra at p=2, k=2 -------------------------------
    L.append("## 1. R_B and mean-shift diagnostics for Box's $M$ weights ($p=2$, $k=2$)")
    L.append("")
    L.append("| parent | per-contrast weights | $R_B$ | mean shift | observed size at $n=50$ |")
    L.append("|---|---|---:|---:|---:|")
    for dist in ("normal", "t5", "lognormal"):
        rec = results["box_m_weights"]["weights"][f"p2_{dist}"]
        spec = weight_spectrum(rec["weights_per_contrast"])
        obs = results["study_mv"][f"T11_box_m_{dist}_p2_k2"]
        idx = obs["ns"].index(50.0)
        L.append(
            f"| {dist} | {', '.join(f'{w:.3f}' for w in spec['weights'])} | "
            f"{spec['r_b']:.4f} | {spec['mean_shift']:.3f} | "
            f"{obs['type1_M'][idx]:.4f} |"
        )
    L.append("")
    L.append("The lognormal row is the corrected independent-coordinate calculation "
             "(56.468, 56.468, 1); its R_B is 0.474 and its mean is 37.98 times the "
             "normal mean.  The t5 row has a moderate shape mismatch but a large mean "
             "shift (11/3 = 3.667).  R_B alone therefore separates shape mismatch, not "
             "the total calibration error.")
    L.append("")

    # --- Bartlett: equal weights but non-zero mean shift -------------------
    L.append("## 2. Bartlett's test: equal weights, scale error only")
    L.append("")
    L.append("| parent | excess kurtosis $\\gamma_2$ | per-contrast weight | $R_B$ | mean shift | observed size at $n=50$ |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for name, gamma2, scenario in (("normal", 0.0, "baseline"),
                                   ("exponential", 6.0, "skewed"),
                                   ("$t_5$", 6.0, "heavy_tail")):
        w = bartlett_weights(gamma2)
        spec = weight_spectrum(w)
        obs = _observed_study2(results, "T5_bartlett", scenario)
        L.append(
            f"| {name} | {gamma2:.1f} | {w[0]:.3f} | {spec['r_b']:.4f} | "
            f"{spec['mean_shift']:.3f} | {obs:.4f} |"
        )
    L.append("")
    L.append("Bartlett is the key counter-example to using R_B alone: its weights remain "
             "equal under non-normality, so R_B = 0 and yet the size error is enormous "
             "(0.429 under the exponential parent; the t5 value differs because higher-order terms differ).  The scalar kurtosis correction is a "
             "mean-scale fix; a spectral diagnostic must therefore be paired with an "
             "R_B-like mean-shift diagnostic.")
    L.append("")

    # --- Sign test R_C -----------------------------------------------------
    L.append("## 3. Sign-test attainable-size error R_C")
    L.append("")
    L.append("| $n$ | exact attainable size | exact $R_C$ | mid-$p$ attainable size | mid-$p$ $R_C$ |")
    L.append("|---:|---:|---:|---:|---:|")
    for n in (20, 30, 50, 100):
        rec = sign_test_diagnostics(n, 0.05)
        L.append(
            f"| {n} | {rec['exact_size']:.5f} | {rec['exact_R_C']:.5f} | "
            f"{rec['midp_size']:.5f} | {rec['midp_R_C']:.5f} |"
        )
    L.append("")
    L.append("At $n=50$ the largest exact attainable size not exceeding 0.05 is 0.03284 ($R_C=0.01716$), "
             "while mid-$p$ reaches 0.06491 ($R_C=0.01491$); observed exact-sign sizes in "
             "the simulation are 0.0307-0.0352 and mid-$p$ sizes are 0.058-0.069, within "
             "Monte Carlo error of these lattice values.")
    L.append("")

    # --- Median test R_C ---------------------------------------------------
    L.append("## 4. Median-test attainable-size error R_C (balanced design)")
    L.append("")
    L.append("| $k$ | $n$ | attainable size | $R_C$ | critical value |")
    L.append("|---:|---:|---:|---:|---:|")
    for k, n in ((3, 10), (4, 10), (4, 20)):
        rec = median_test_diagnostics(k, n, 0.05)
        L.append(
            f"| {k} | {n} | {rec['attainable']:.5f} | {rec['R_C']:.5f} | "
            f"{rec['threshold']:.3f} |"
        )
    L.append("")
    L.append("Median-test R_C is design-dependent (0.0037-0.0158 in these small balanced designs); the exact "
             "distribution is hypergeometric.  Cochran's $Q$ and the rank tests need their "
             "own exact enumeration and are left pending in this prototype.")
    L.append("")

    # --- T1 Edgeworth diagnostic ------------------------------------------
    L.append("## 5. T1 $t$-test: analytic Edgeworth diagnostic $R_A$")
    L.append("")
    L.append("| parent | $\\gamma_1$ | $n$ | $x$ | $c_A(x)$ | $R_A$ | observed two-sided size at $n=50$ |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, gamma1, x, scenario in (
        ("normal", 0.0, 1.96, "baseline"),
        ("exponential", 2.0, 1.96, "skewed"),
        ("$t_5$", 0.0, 1.96, "heavy_tail"),
    ):
        rec = t_edgeworth_diagnostic(gamma1, 50, x)
        obs = _observed_study2(results, "T1_t_test", scenario)
        L.append(
            f"| {name} | {gamma1:.1f} | 50 | {x:.3f} | {rec['coefficient']:+.5f} | "
            f"{rec['R_A']:.5f} | {obs:.4f} |"
        )
    L.append("")
    L.append("For the two-sided $t$-test the leading coefficient is even in $x$, so the "
             "O($n^{-1/2}$) size errors of the two symmetric tails cancel; $R_A$ is a "
             "CDF-level diagnostic, not a two-sided size prediction.  A size-oriented "
             "version needs the next-order term or the one-sided tail functional.")
    L.append("")

    # --- Simulation scaling for T2/T10 -----------------------------------
    L.append("## 6. Simulation-calibrated size-error scaling for ratio statistics")
    L.append("")
    L.append("| test | $n$ | nominal-reference size | error |")
    L.append("|---|---:|---:|---:|")
    sim_rows = [
        simulate_size_curve("T2_anova", N_mc=20000),
        simulate_size_curve("T10_hotelling", N_mc=20000),
    ]
    for rec in sim_rows:
        label = {"T2_anova": "T2 ANOVA $F$", "T10_hotelling": "T10 Hotelling $T^2$"}[rec["test"]]
        for n, size, err in zip(rec["n_grid"], rec["sizes"], rec["errors"]):
            L.append(f"| {label} | {int(n)} | {size:.4f} | {err:+.4f} |")
    L.append("")
    slopes = {r["test"]: r["slope"] for r in sim_rows}
    L.append("Fitted log-log slopes on the positive errors: T2 ANOVA "
             + ("not identifiable (errors change sign)" if slopes["T2_anova"] is None
                else f"{slopes['T2_anova']:.2f}")
             + f"; T10 Hotelling {slopes['T10_hotelling']:.2f}.  "
             "Under the exponential parent the ANOVA size error is small and non-monotone "
             "in this range, whereas the Hotelling $T^2$ error decays roughly like "
             "$n^{-1}$, not $n^{-1/2}$.  The latter is consistent with the parity "
             "cancellation of the leading Edgeworth term in regime B; a naive $R_A$ "
             "designed for regime A would misclassify it.")
    L.append("")

    # --- Coverage table ----------------------------------------------------
    L.append("## 7. Regime coverage of the prototype")
    L.append("")
    L.append("| test | regime | diagnostic implemented here | status |")
    L.append("|---|---|---|---|")
    rows = [
        ("T1 t", "A", "$R_A$ (analytic, one-sample $t$)", "implemented"),
        ("T2 ANOVA $F$", "B", "$R_A$/$R_B$", "theory pending (quadratic ratio)"),
        ("T3 Kruskal-Wallis", "C", "$R_C$", "exact enumeration pending"),
        ("T4 $\\chi^2$ GOF", "C", "$R_C$", "exact multinomial pending"),
        ("T5 Bartlett", "B (+scale)", "$R_B$, mean shift", "implemented"),
        ("T6 Fligner-Killeen", "C", "$R_C$", "not exact; parent sensitivity"),
        ("T7 Sign", "C", "$R_C$", "implemented"),
        ("T8 Cochran's $Q$", "C", "$R_C$", "exact enumeration pending"),
        ("T9 Median", "C", "$R_C$", "implemented (balanced design)"),
        ("T10 Hotelling", "B", "$R_B$", "theory pending"),
        ("T11 Box's $M$", "B", "$R_B$, mean shift", "implemented"),
    ]
    for test, regime, diag, status in rows:
        L.append(f"| {test} | {regime} | {diag} | {status} |")
    L.append("")
    L.append("The prototype deliberately does not claim a validated selector.  Its purpose "
             "is to show that the regime-specific diagnostics are computable and that they "
             "carry signal before investing in threshold tuning, sample splitting and "
             "out-of-sample validation.")
    return "\n".join(L) + "\n"


def write_retrospective_table(path=OUT):
    text = retrospective_table()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)
    return path


def main():
    print(retrospective_table())
    path = write_retrospective_table()
    print(f"wrote {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
