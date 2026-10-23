"""Cheap regression tests for experiment.py (run: python3 -m pytest -q tests).

They cover the pieces a reviewer is most likely to check and that are cheap to run:
the block-design statistic, the exact CGF of Box's M, the two tail formulas, the
Mardia bias correction, and the results schema.
"""
import json
import os
import sys

import numpy as np
import pytest
from scipy import special as sp
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import experiment as ex  # noqa: E402


# --------------------------------------------------------------------- Cochran's Q
def test_cochran_q_has_no_unbalanced_argument():
    with pytest.raises(TypeError):
        ex.mc_cochran_q(3, 20, 10, unbalanced=True)


def test_cochran_q_undefined_reps_are_nan_not_huge():
    q = ex.mc_cochran_q(3, 5, 2000)
    assert np.isfinite(q).mean() > 0.95          # almost all defined at n=5, k=3
    assert np.nanmax(q) < 1e6                    # no 1e-15-denominator blow-up
    # the defined values must be genuine Q values: 0 <= Q, and bounded by k*(n-1)
    assert np.nanmin(q) >= 0.0
    assert np.nanmax(q) <= 3 * (5 - 1)


# ----------------------------------------------------------------- exact CGF
def test_balanced_and_unbalanced_cgf_agree():
    p, k, m = 3, 3, 19
    q0 = k * m
    cg_bal = ex.boxm_cgf(m, q0, p, k)
    cg_unb = ex.boxm_cgf_unbalanced([m] * k, p)
    for r in (1, 2):
        a, b = cg_bal.deriv(0.0, r), cg_unb.deriv(0.0, r)
        assert abs(a - b) <= 1e-9 * max(1.0, abs(a))


def test_unbalanced_cumulants_do_not_add_the_shift_twice():
    """The Lam-based CGF already contains the deterministic p(sum m log m - q0 log q0)
    term; a separate shift would double it.  The mean must be positive and match the
    balanced CGF at the balanced design."""
    m, p, k = 9, 2, 2
    mean_unb = ex.boxm_cumulants_unbalanced([m, m], p, R=1)[0]
    mean_bal = ex.boxm_cumulants(m, k * m, p, k, R=1)[0]
    assert mean_unb > 0
    assert abs(mean_unb - mean_bal) < 1e-8


def test_cgf_evaluates_and_reduces_at_p1():
    # at p = 1 the statistic is Bartlett's; the CGF must be finite and increasing
    cg = ex.boxm_cgf(19, 2 * 19, 1, 2)
    assert np.isfinite(cg.K(0.0))
    assert cg.K1(0.0) > 0
    assert cg.K2(0.0) > 0


# ------------------------------------------------------- tail approximations
@pytest.mark.parametrize("nu", [3.0, 12.0])
@pytest.mark.parametrize("x", [4.0, 8.0])
def test_tail_inversion_matches_chi2(nu, x):
    got = ex.tail_invert(ex.chi2_cgf(nu), x)
    want = float(sp.chdtrc(nu, x))
    assert abs(got - want) < 1e-4


@pytest.mark.parametrize("nu", [3.0, 12.0])
def test_saddlepoint_matches_chi2(nu):
    cg = ex.chi2_cgf(nu)
    for x in (4.0, 8.0):
        got = ex.tail_lr(cg, x)
        want = float(sp.chdtrc(nu, x))
        assert abs(got - want) < 2e-2


def test_paper_equation_56_is_the_code_formula():
    """Eq. (5.6) in paper.md: P = (1/2pi) int_{-inf}^{inf} exp(K(c+is)-(c+is)x)/(c+is) ds.
    Check the real-part/one-sided form the code uses against the full-line integral."""
    cg = ex.chi2_cgf(3.0)
    x = 4.0
    c = ex._solve_saddle(cg, x)
    s = np.linspace(1e-6, 60.0, 200001)
    z = c + 1j * s
    integ = np.exp(cg.K(z) - z * x) / z
    one_sided = np.trapezoid(integ.real, s) / np.pi
    two_sided = np.trapezoid(np.concatenate([integ[::-1].conj(), integ]), np.concatenate([-s[::-1], s])) / (2 * np.pi)
    assert abs(one_sided - two_sided) < 1e-3
    assert abs(one_sided - ex.tail_invert(cg, x)) < 1e-3


# ------------------------------------------------------- Mardia bias correction
def test_mardia_bias_correction_matches_normal_expectation():
    rng = np.random.default_rng(11)
    p, n = 2, 300
    X = rng.normal(size=(n, p))
    _, b2p_bc = ex.mardia_b2p_bc(X)
    # E[b2p_bc] = p(p+2) under normality; loose MC tolerance
    assert abs(b2p_bc - p * (p + 2)) < 0.15 * p * (p + 2)


# ---------------------------------------------------------- multivariate study
def test_mc_box_m_returns_diagnostics():
    sim = ex.mc_box_m(2, 2, 10, 50)
    for key in ("M", "Mstar", "Mstar_bc", "Mstar_raw", "Mstar_floor025", "denom", "denom_bc"):
        assert key in sim and len(sim[key]) == 50
    # the raw version must be undefined exactly where the denominator is non-positive
    assert np.all(np.isnan(sim["Mstar_raw"]) == (sim["denom"] <= 0))
    # the floored version must be finite everywhere
    assert np.all(np.isfinite(sim["Mstar_floor025"]))


# ------------------------------------------------- paired sign (P0-1 review fix)
def test_paired_difference_is_corrected_minus_uncorrected():
    """Delta must follow paper.md: corrected minus uncorrected rejection rate."""
    out = ex.run_study2_pairing(N_mc=400, n_fixed=50)
    for family in ("bartlett", "sign"):
        for sc, rec in out[family].items():
            for a, cell in rec.items():
                assert abs(cell["diff"] - (cell["size_after"] - cell["size_before"])) < 1e-12
    skew = out["bartlett"]["skewed"]["0.05"]
    assert skew["size_after"] < skew["size_before"]      # B* repairs the Exp inflation
    assert skew["diff"] < 0
    sign = out["sign"]["baseline"]["0.05"]
    assert sign["size_after"] > sign["size_before"]      # mid-p overshoots
    assert sign["diff"] > 0


# ------------------------------------------- Proposition 1 counterexample (P0-2)
def test_nonlinear_counterexample_first_order_term():
    """For H(u)=u+c u^2 and a normal parent,
    P(sqrt(n) H(Xbar) <= x) = Phi(x) - n^{-1/2} c x^2 phi(x) + o(n^{-1/2}).
    The earlier tail-integral expression c phi(x) E[Z^2 1{Z<=x}] is wrong.
    """
    from scipy import stats as st
    rng = np.random.default_rng(5)
    c, x0, n, N = 1.0, 1.0, 2000, 400_000
    Z = rng.normal(size=N)
    T = Z + c / np.sqrt(n) * Z ** 2
    got = float(np.mean(T <= x0))
    want = st.norm.cdf(x0) - c * x0 ** 2 * st.norm.pdf(x0) / np.sqrt(n)
    assert abs(got - want) < 1.5e-3
    wrong = c * st.norm.pdf(x0) * (st.norm.cdf(x0) - x0 * st.norm.pdf(x0))
    assert abs(got - (st.norm.cdf(x0) + wrong)) > 0.1


# -------------------------------- Proposition 5 parity cancellation (P0-3)
def test_quadratic_form_first_order_correction_cancels():
    """For T_n = n Xbar^2 the O(n^{-1/2}) Edgeworth CDF term cancels between
    +sqrt(x) and -sqrt(x) because the third-cumulant correction is odd and the
    acceptance region is symmetric.  Under a skewed parent the empirical CDF at
    n=2000 must therefore agree with 2 Phi(sqrt(x))-1 to well within the
    n^{-1/2} scale."""
    from scipy import stats as st
    # exact parity identity: int_{-a}^{a} phi He_3 = 0
    a = 1.2
    z = np.linspace(-a, a, 400001)
    he3 = z ** 3 - 3 * z
    assert abs(np.trapezoid(st.norm.pdf(z) * he3, z)) < 1e-10
    # Monte Carlo with a skewed parent
    rng = np.random.default_rng(9)
    n, N, chunk = 2000, 300_000, 10_000
    hits = 0
    for _ in range(N // chunk):
        X = rng.exponential(1.0, size=(chunk, n)) - 1.0
        hits += int(np.sum(n * X.mean(axis=1) ** 2 <= 1.0))
    got = hits / N
    ref = 2 * st.norm.cdf(1.0) - 1
    assert abs(got - ref) < 4e-3     # n^{-1/2}=0.022; no term of that order survives


# ------------------------------------------------ mid-p attainable sizes (P0-4)
def test_midp_attainable_sizes_at_n50():
    from scipy import stats as st
    n = 50
    k = np.arange(n + 1)
    pmf = st.binom.pmf(k, n, 0.5)
    cdf = np.cumsum(pmf)
    sf = 1 - cdf + pmf
    p_ex = np.clip(2 * np.minimum(cdf, sf), 0, 1)
    p_mid = np.clip(2 * np.minimum(cdf - 0.5 * pmf, sf - 0.5 * pmf), 0, 1)
    sizes = {}
    for a in (0.01, 0.025, 0.05, 0.10):
        sizes[a] = (float(pmf[p_ex <= a].sum()), float(pmf[p_mid <= a].sum()))
    assert abs(sizes[0.05][0] - 0.03284) < 1e-4
    assert abs(sizes[0.05][1] - 0.06491) < 1e-4
    # mid-p is closer to the nominal level in absolute deviation for alpha >= 0.025
    for a in (0.025, 0.05, 0.10):
        assert abs(sizes[a][1] - a) < abs(sizes[a][0] - a)
    # but the exact rule never exceeds alpha
    for a in (0.01, 0.025, 0.05, 0.10):
        assert sizes[a][0] <= a + 1e-12


# ------------------------- studentised and centred bootstraps are identical
def test_studentised_within_group_bootstrap_is_the_centred_one():
    """Scaling residuals by a group's own Cholesky factor and multiplying the same
    factor back is the identity; the two schemes therefore cannot provide
    independent evidence, so the resampling table counts them as one method."""
    rng = np.random.default_rng(17)
    g = rng.normal(size=(30, 2))
    m = g.mean(axis=0)
    S = (g - m).T @ (g - m) / (g.shape[0] - 1)
    L = np.linalg.cholesky(S)
    R = (g - m) @ np.linalg.inv(L).T
    idx = rng.integers(0, g.shape[0], g.shape[0])
    centred = m + (g - m)[idx]
    studentised = m + R[idx] @ L.T
    assert np.max(np.abs(centred - studentised)) < 1e-12


# ------------------------------------------- generated tables and run log (D-5, D-6)
def test_generated_tables_use_the_paper_numbering(tmp_path):
    """generated_tables.md must cover the machine-generated paper tables under the
    paper's own numbers (Tables 1, 2, 7 and 18 are editorial and are excluded)."""
    with open(os.path.join(ROOT, "results", "experiment_results.json")) as fh:
        R = json.load(fh)
    out = tmp_path / "generated_tables.md"
    ex.generate_tables(R, str(out))
    txt = out.read_text()
    for n in (3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17):
        assert f"## Table {n}:" in txt, f"generated tables missing Table {n}"
    assert "T11  Box's $M$" in txt          # the T11 row of Table 8
    assert "## Table A1:" in txt            # appendix weight decomposition
    assert "## Table 18:" not in txt        # editorial, not generated


def test_run_log_records_a_complete_run():
    """results/run.log must be a complete run, not a truncated one (D-6)."""
    with open(os.path.join(ROOT, "results", "run.log")) as fh:
        log = fh.read()
    assert "results schema: OK" in log
    assert "Total time:" in log


# ------------------------------------------------ round-5 regressions
def test_fligner_killeen_matches_scipy():
    """The custom FK statistic must agree with scipy.stats.fligner to machine
    precision; ties in |x - median| matter for even n."""
    rng = np.random.default_rng(41)
    for n in (25, 30, 50):
        for kind in ("normal", "expon", "t5"):
            if kind == "normal":
                X = rng.normal(size=(4, 3, n))
            elif kind == "expon":
                X = rng.exponential(size=(4, 3, n)) - 1.0
            else:
                X = rng.standard_t(5, size=(4, 3, n))
            got = ex.fligner_killeen_stats(X)
            want = np.array([st.fligner(*[X[i, j] for j in range(3)])[0] for i in range(4)])
            assert np.max(np.abs(got - want)) < 1e-10


def test_box_m_weight_decomposition():
    """Under normality every weight is 1 and the sum is nu; the shared-scale
    t5 and independent-coordinate lognormal parents have unequal weights."""
    w = ex.run_weight_decomposition()
    for p in (2, 3):
        norm = w["weights"][f"p{p}_normal"]
        assert norm["equal_weights"]
        assert np.allclose(norm["weights_per_contrast"], 1.0)
        assert abs(norm["sum_per_contrast"] - p * (p + 1) / 2.0) < 1e-9
        for dist in ("t5", "lognormal"):
            rec = w["weights"][f"p{p}_{dist}"]
            assert not rec["equal_weights"]
            assert rec["max_min_ratio"] > 1.5
    assert abs(w["weights"]["p2_t5"]["sum_per_contrast"] - 11.0) < 1e-9

    # Regression for the lognormal generator: the coordinates are independent,
    # so kappa_aabb = 0 and the diagonal weight uses the CENTRED fourth
    # cumulant.  The raw-moment shortcut used to give 68.57/67.07/2.50.
    e = np.e
    c = e ** 2 - e
    k4 = e ** 8 - 4 * e ** 5 + 6 * e ** 3 - 3 * e ** 2 - 3 * c ** 2
    w_diag = 0.5 * (k4 / c ** 2 + 2.0)
    for p in (2, 3):
        rec = w["weights"][f"p{p}_lognormal"]
        want = [w_diag] * p + [1.0] * (p * (p - 1) // 2)
        assert np.allclose(sorted(rec["weights_per_contrast"]), sorted(want),
                           rtol=1e-12, atol=1e-12)
        assert abs(rec["sum_per_contrast"] - sum(want)) < 1e-9
        groups = {g["weight"]: g["multiplicity"] for g in rec["weight_groups"]}
        assert groups[round(w_diag, 6)] == p
        assert groups[1.0] == p * (p - 1) // 2
    assert abs(w["weights"]["p2_lognormal"]["sum_per_contrast"] - 113.936392) < 1e-5
    assert abs(w["weights"]["p3_lognormal"]["sum_per_contrast"] - 172.404588) < 1e-5


def test_lattice_bound_is_attained_near_the_jump():
    """Prop 3: for even n the bound is approached at x=0; for odd n at x=+/-1/sqrt(n)."""
    for n in (25, 26, 50, 51):
        k = np.arange(n + 1)
        z = (k - n / 2.0) / (np.sqrt(n) / 2.0)
        F = np.cumsum(st.binom.pmf(k, n, 0.5))
        err = np.abs(F - st.norm.cdf(z))
        j = int(np.argmax(err))
        if n % 2 == 0:
            assert abs(z[j]) < 1e-12
        else:
            assert abs(abs(z[j]) - 1.0 / np.sqrt(n)) < 1e-9


def test_power_uses_independent_calibration_sample():
    out = ex.run_power_study(N_mc=1000, n_fixed=30)
    assert out["N_calibration"] == 1000 and out["N_evaluation"] == 1000
    for rec in out["tests"].values():
        assert 0 <= rec["size_adjusted"] <= 1
        assert 0 <= rec["power_size_adjusted"] <= 1
    # the randomised rule is calibrated: its realised size is close to 5%
    for tid in ("T1_t_test", "T5_bartlett", "T7_sign_test"):
        assert abs(out["tests"][tid]["size_adjusted"] - 0.05) < 0.02


def test_committed_outputs_have_no_local_paths():
    """run.log and the results file must not leak a local absolute path
    (double-blind submission)."""
    with open(os.path.join(ROOT, "results", "experiment_results.json")) as fh:
        text = fh.read()
    with open(os.path.join(ROOT, "results", "run.log")) as fh:
        log = fh.read()
    for blob in (text, log):
        assert "/Users/" not in blob
        assert "论文" not in blob
    ref = json.loads(text)["exact_null"]["reference"]
    assert ref is not None and not ref.startswith("/")


def test_fligner_killeen_inflation_is_type1_error():
    """In the exponential cell every group has the same parent, so H0 holds and the
    rejection rate is a Type I error; it is inflated by the finite-sample chi-square
    reference, not by a violated null."""
    np.random.seed(123)
    q = ex.mc_fligner_killeen(3, 50, 5000, dist='expon')
    rate = float(np.mean(q > st.chi2.ppf(0.95, 2)))
    assert rate > 0.07        # nominal 0.05; the paper reports ~0.12


def test_heterokurtosis_block():
    """B* improves but does not calibrate when group kurtoses differ."""
    with open(os.path.join(ROOT, "results", "experiment_results.json")) as fh:
        R = json.load(fh)
    h = R["heterokurtosis_check"]["cells"]["heterokurtosis"]
    assert h["type1_B"] > h["type1_B_star"]
    assert h["type1_B_star"] > 0.05


# ---------------------------------------------------------------- results schema
def test_committed_results_have_the_schema_the_paper_reads():
    path = os.path.join(ROOT, "results", "experiment_results.json")
    with open(path) as fh:
        results = json.load(fh)
    ex.check_results(results)          # raises SystemExit if a field is missing
    assert len(results["exact_null"]["cells"]) == 18


# --------------------------------------------------- theory-property regressions
def test_quadratic_form_needs_the_half_factor():
    """P0-1: Q = (1/2) W^T A W with A = Sigma = 1 is (1/2) chi^2_1, not chi^2_1.
    The chi-square condition is that (1/2) Sigma^{1/2} A Sigma^{1/2} is a projection."""
    rng = np.random.default_rng(3)
    W = rng.normal(size=200000)
    Q = 0.5 * W ** 2
    from scipy import stats as st
    # mean and CDF at 1 must match (1/2) chi^2_1, not chi^2_1
    assert abs(Q.mean() - 0.5) < 0.01
    assert abs(np.mean(Q <= 1.0) - st.chi2.cdf(2.0, 1)) < 0.01
    assert abs(np.mean(Q <= 1.0) - st.chi2.cdf(1.0, 1)) > 0.1


def test_symmetric_binomial_lattice_constant():
    """P0-3: for the standardised symmetric binomial, D_KS * sqrt(n) -> phi(0)."""
    from scipy import stats as st
    for n in (200, 400, 800):
        k = np.arange(0, n + 1)
        z = (k - n / 2) / np.sqrt(n / 4)
        F = np.cumsum(st.binom.pmf(k, n, 0.5))
        D = np.max(np.abs(F - st.norm.cdf(z)))
        assert abs(D * np.sqrt(n) - st.norm.pdf(0)) < 2e-3


def test_kruskal_wallis_is_distribution_free_under_the_null():
    """P1.1: KW has the same null under a normal and an exponential parent."""
    np.random.seed(11)
    a = ex.mc_kruskal_wallis(4, 50, 2000, dist='normal')
    np.random.seed(11)
    b = ex.mc_kruskal_wallis(4, 50, 2000, dist='expon')
    q = 7.8147
    assert abs(np.mean(a > q) - np.mean(b > q)) < 0.02


def test_paper_figure_references_exist():
    """Every image referenced by paper.md must exist in figures/."""
    import re as _re
    paper = open(os.path.join(ROOT, "paper.md")).read()
    refs = _re.findall(r"!\[[^\]]*\]\(([^)]+)\)", paper)
    assert refs, "no image references found"
    for ref in refs:
        assert os.path.isfile(os.path.join(ROOT, ref)), f"missing figure: {ref}"


def test_frozen_headline_numbers():
    """End-to-end freeze: the committed results still contain the numbers the paper quotes."""
    import json as _json
    with open(os.path.join(ROOT, "results", "experiment_results.json")) as fh:
        R = _json.load(fh)
    s2 = R["study2"]["T5_bartlett"]
    assert abs(s2["skewed"] - 0.429) < 0.002
    assert abs(s2["heavy_tail"] - 0.276) < 0.002
    corr = R["study2_corrections"]["T5_bartlett"]
    assert abs(corr["skewed"] - 0.133) < 0.003
    v = R["exact_null"]["validation"]
    assert v["max_rel_lr_vs_inv"] < 5e-3
    assert abs(v["max_rel_chi2_vs_mc"] - 0.1454) < 5e-3


# ------------------------------------------------- paired / power / resolution studies
def test_paired_returns_paired_sizes_and_ci():
    out = ex.run_study2_pairing(N_mc=400, n_fixed=20)
    assert set(out["alphas"]) == {0.01, 0.025, 0.05, 0.1}
    for family in ("bartlett", "sign"):
        for sc, rec in out[family].items():
            cell = rec["0.05"]
            assert 0.0 <= cell["size_before"] <= 1.0
            assert 0.0 <= cell["size_after"] <= 1.0
            assert cell["discordant_before_only"] + cell["discordant_after_only"] <= cell["N"]
            assert cell["ci_lo"] <= cell["diff"] <= cell["ci_hi"]


def test_power_is_in_unit_interval_and_corrected_bartlett_reported():
    out = ex.run_power_study(N_mc=400, n_fixed=20)
    assert out["tests"]["T1_t_test"]["power"] > 0
    assert 0 < out["tests"]["T5_bartlett"]["power_corrected"] < 1
    assert "power_corrected" in out["tests"]["T7_sign_test"]


def test_ks_resolution_keys_and_lengths():
    fake = {t: {"ns": [10, 50], "ks": [0.1, 0.05]} for t in
            ("T3_kruskal_wallis", "T4_chisq_gof", "T6_fligner_killeen",
             "T8_cochran_q", "T9_median_test")}
    out = ex.run_ks_resolution(fake, N_big=4000, n_list=(10, 50))
    for t, rec in out["tests"].items():
        assert len(rec["ns"]) == len(rec["ks_big"]) == len(rec["ks_10k"]) == 2


def test_paired_headline_numbers_frozen():
    """The paired study behind Table 13 / Table 15 / Table 16 stays consistent with the paper."""
    import json as _json
    with open(os.path.join(ROOT, "results", "experiment_results.json")) as fh:
        R = _json.load(fh)
    pair = R["study2_paired"]
    skew = pair["bartlett"]["skewed"]["0.05"]
    assert abs(skew["size_before"] - 0.427) < 0.01
    # Delta is corrected minus uncorrected, so a reduction is negative
    assert abs(skew["diff"] + 0.293) < 0.01
    assert skew["ci_lo"] < skew["diff"] < skew["ci_hi"]
    sign = pair["sign"]["baseline"]["0.1"]
    assert sign["diff"] > 0            # mid-p overshoots at alpha = 0.10
    power = R["power"]["tests"]
    assert abs(power["T1_t_test"]["power"] - 0.554) < 0.02
    assert power["T7_sign_test"]["power_corrected"] > power["T7_sign_test"]["power"]
    ks = R["ks_resolution"]["tests"]["T9_median_test"]
    assert abs(ks["ks_big"][0] - ks["ks_10k"][0]) < 0.01   # population distance, not noise


def test_power_reports_size_adjusted_values():
    out = ex.run_power_study(N_mc=1000, n_fixed=30)
    for rec in out["tests"].values():
        assert 0 <= rec["power_size_adjusted"] <= 1
        # the randomised rule attains its nominal level in an independent sample
        assert abs(rec["size_adjusted"] - 0.05) < 0.025
        assert 0 <= rec["size"] <= 1
    assert "power_size_adjusted_corrected" in out["tests"]["T5_bartlett"]


def test_bootstrap_comparison_shape_and_range():
    out = ex.run_bootstrap_comparison(N_outer=6, B=5)
    assert len(out["cells"]) == 5
    assert sum("equal cov" in lab for lab in out["cells"]) == 2
    methods = {"parametric", "within_centred", "null_imposed", "permutation"}
    for label, cell in out["cells"].items():
        assert set(cell["type1"]) == methods
        for v in cell["type1"].values():
            assert 0 <= v <= 1


def test_ks_resolution_has_large_n_column():
    fake = {t: {"ns": [10], "ks": [0.1]} for t in
            ("T3_kruskal_wallis", "T4_chisq_gof", "T6_fligner_killeen",
             "T8_cochran_q", "T9_median_test")}
    out = ex.run_ks_resolution(fake, N_big=2000, n_list=(10,), N_huge=3000, n_huge=(10,))
    for rec in out["tests"].values():
        assert rec["ns_huge"] == [10]
        assert len(rec["ks_huge"]) == 1


def test_power_and_resampling_headline_numbers_frozen():
    """Values behind Tables 15 and 17 stay consistent with the committed results."""
    import json as _json
    with open(os.path.join(ROOT, "results", "experiment_results.json")) as fh:
        R = _json.load(fh)
    pw = R["power"]["tests"]
    assert abs(pw["T5_bartlett"]["power"] - 0.568) < 0.02
    assert abs(pw["T5_bartlett"]["power_corrected"] - 0.598) < 0.02
    # once size-adjusted, the corrected Bartlett power falls below the uncorrected one
    assert pw["T5_bartlett"]["power_size_adjusted_corrected"] < pw["T5_bartlett"]["power_size_adjusted"]
    # at a common actual size the two sign-test rules coincide
    assert abs(pw["T7_sign_test"]["power_size_adjusted"]
               - pw["T7_sign_test"]["power_size_adjusted_corrected"]) < 0.01
    # the exact rule is conservative and the mid-p rule anti-conservative at nominal 5%
    assert pw["T7_sign_test"]["size"] < 0.04
    assert pw["T7_sign_test"]["size_corrected"] > 0.06
    boot = R["bootstrap_comparison"]["cells"]
    mvn = boot["MVN p=2,k=2,n=30"]["type1"]
    assert 0.02 < mvn["parametric"] < 0.12
    assert mvn["within_centred"] < 0.02
    assert 0.01 < mvn["null_imposed"] < 0.15
    logn = boot["lognormal p=2,k=2,n=30"]["type1"]
    assert logn["parametric"] > 0.4
    assert logn["null_imposed"] < 0.3      # null-imposed repairs most of the failure
    assert logn["permutation"] < 0.10
    # equal covariance but different shapes: exchangeability fails, so permutation
    # is visibly inflated while the null-imposed scheme is closer but not perfect
    shape = boot["normal vs lognormal, equal cov, n=30"]["type1"]
    assert shape["permutation"] > 0.2
    assert shape["null_imposed"] < shape["parametric"]
    ks = R["ks_resolution"]["tests"]["T9_median_test"]
    assert abs(ks["ks_huge"][0] - ks["ks_10k"][0]) < 0.01   # population distance, stable
