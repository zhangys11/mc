"""Tests for the exploratory Stage 1 diagnostics (not used by paper.md)."""
import math

import numpy as np
import pytest

import diagnostics as dg


def test_weight_spectrum_identities():
    rec = dg.weight_spectrum([2.0, 2.0, 2.0])
    assert rec["r_b"] == 0.0
    assert rec["mean_shift"] == pytest.approx(1.0)
    assert rec["equal_weights"]

    rec = dg.weight_spectrum([56.46819608815576, 56.46819608815576, 1.0])
    assert rec["r_b"] == pytest.approx(0.4740, abs=1e-3)
    assert rec["mean_shift"] == pytest.approx(36.979, abs=1e-3)
    assert not rec["equal_weights"]

    # weights of a weighted chi-square are scale-free in R_B
    a = dg.weight_spectrum([1.0, 2.0, 3.0])["r_b"]
    b = dg.weight_spectrum([10.0, 20.0, 30.0])["r_b"]
    assert a == pytest.approx(b)
    with pytest.raises(ValueError):
        dg.weight_spectrum([])


def test_bartlett_has_equal_weights_but_nonzero_mean_shift():
    for gamma2 in (0.0, 6.0, -1.0):
        w = dg.bartlett_weights(gamma2)
        rec = dg.weight_spectrum(w)
        assert rec["r_b"] == 0.0
        assert rec["mean_shift"] == pytest.approx(gamma2 / 2.0)


def test_sign_test_attainable_sizes_match_table14():
    rec = dg.sign_test_diagnostics(50, 0.05)
    assert rec["exact_size"] == pytest.approx(0.03284, abs=1e-5)
    assert rec["exact_R_C"] == pytest.approx(0.01716, abs=1e-5)
    assert rec["midp_size"] == pytest.approx(0.06491, abs=1e-5)
    assert rec["midp_R_C"] == pytest.approx(0.01491, abs=1e-5)
    # At n=100 the mid-p rule improves the attainable-size error.
    rec100 = dg.sign_test_diagnostics(100, 0.05)
    assert rec100["midp_R_C"] < rec100["exact_R_C"]


def test_median_test_pmf_sums_to_one_and_has_known_mean():
    for k, n in ((3, 10), (4, 10), (4, 20)):
        support, pmf = dg.median_test_pmf(k, n)
        assert pmf.sum() == pytest.approx(1.0, abs=1e-12)
        total = k * n
        expected_mean = k * (total - n) / (total - 1.0)
        assert float(np.sum(support * pmf)) == pytest.approx(expected_mean, rel=1e-10)


def test_median_test_attainable_size_is_conservative():
    rec = dg.median_test_diagnostics(4, 20, 0.05)
    assert 0.0 <= rec["attainable"] <= 0.05
    assert rec["R_C"] == pytest.approx(0.05 - rec["attainable"])
    assert rec["threshold"] > 0


def test_t_edgeworth_coefficient():
    x = 1.96
    c = dg.t_edgeworth_coefficient(2.0, x)
    assert c == pytest.approx(2.0 * (2 * x * x + 1) * math.exp(-x * x / 2)
                              / math.sqrt(2 * math.pi) / 6.0, rel=1e-12)
    assert c > 0
    assert dg.t_edgeworth_coefficient(0.0, x) == 0.0
    rec = dg.t_edgeworth_diagnostic(2.0, 50, x)
    assert rec["R_A"] == pytest.approx(abs(c) / math.sqrt(50))


def test_simulate_size_curve_smoke():
    rec = dg.simulate_size_curve("T10_hotelling", n_grid=(20, 40), N_mc=2000, seed=1)
    assert rec["test"] == "T10_hotelling"
    assert len(rec["sizes"]) == len(rec["errors"]) == 2
    assert all(0.0 <= s <= 1.0 for s in rec["sizes"])
    assert rec["slope"] is None or rec["slope"] < 0


def test_retrospective_table(tmp_path):
    out = tmp_path / "diagnostics.md"
    dg.write_retrospective_table(str(out))
    text = out.read_text()
    for needle in ("Stage 1 diagnostic prototype", "56.468", "0.4740",
                   "0.03284", "0.06491", "0.16915", "0.02392", "T11 Box's $M$"):
        assert needle in text, needle
    # the prototype must not be presented as a validated selector
    assert "not validated selection rules" in text
