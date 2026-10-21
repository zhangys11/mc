# Box's M null-distribution study

Monte Carlo verification of the null distribution of the Box's M statistic
implemented in [`mc.samplings.Box_M_Test`](../../../src/mc/samplings/_box_m_test.py).
The scripts put the repository's `src/` on `sys.path`, so they run from a clone
without an installation; they use the toolkit's runtime dependencies (`numpy`,
`scipy`, `matplotlib`, `tqdm`).

Box's M tests the homogeneity of covariance matrices across `k` groups of
`p`-variate observations:

```
H0: Sigma_1 = Sigma_2 = ... = Sigma_k
M  = (N - k) ln|S_p| - sum_j (n_j - 1) ln|S_j|
```

Under multivariate normality the Bartlett-corrected statistic is approximately
`chi2(nu)` with `nu = (k - 1) p (p + 1) / 2` and

```
c = (2p^2 + 3p - 1) / (6 (p + 1) (k - 1))
    * ( sum_j 1/(n_j - 1) - 1/(N - k) ).
```

The point of the study is that this normal-theory reference is only as good as
the normal assumption. Two questions are answered by simulation:

1. **Adequacy under normality.** Does `(1 - c) M` follow `chi2(nu)` over a grid
   of `k`, `p` and `n`, and how much does the Bartlett correction matter?
2. **Breakdown under non-normality.** When all groups still share the same
   covariance matrix but the marginal population is non-normal, the sample
   covariances fluctuate with the fourth moments, so the null law is a weighted
   sum of chi-squares rather than a chi-square. The study measures the size
   inflation, derives the asymptotic weights from the fourth-moment matrix
   `Omega`, and verifies the weighted limit by simulation.

## Running

Run from the repository root (the script locates `src/` itself):

```bash
python3 "Finite-Sample Adequacy and Robustness of Null Distributions/extensions/box_m/study_box_m_null.py"
python3 "Finite-Sample Adequacy and Robustness of Null Distributions/extensions/box_m/study_box_m_null.py" --quick
python3 "Finite-Sample Adequacy and Robustness of Null Distributions/extensions/box_m/study_box_m_null.py" --no-figures
```

or from the package root (`Finite-Sample Adequacy and Robustness of Null Distributions/`):

```bash
python3 extensions/box_m/study_box_m_null.py              # full run, ~1 min
python3 extensions/box_m/study_box_m_null.py --quick      # smoke run, ~11 s
python3 extensions/box_m/study_box_m_null.py --no-figures # results only
```

The full run writes `results/box_m_null_results.json` (all rows, the analytic
checks and the headline numbers) and three figures under `figures/`; a
`--quick` smoke run writes `results/box_m_null_results_quick.json` instead, so
it cannot overwrite the canonical full-run artifact. The defaults are
`N = 10,000` replications per configuration, `200,000` draws for the limiting
law, and seed `2024`; the full run takes about a minute on one core (57 s in the
committed run).

## What is measured

For every configuration the script records

* the sample mean and variance of `(1 - c) M` and their large-`n` limits;
* `D_KS`, the Kolmogorov-Smirnov distance between the empirical null law and
  the nominal `chi2(nu)` reference, and the same distance for the uncorrected
  statistic `M`;
* the empirical type I error at the 1% and 5% levels with Wilson intervals;
* for non-normal populations, `D_KS` against the weighted chi-square limit and
  against a two-moment (Satterthwaite) scaled chi-square, the eigenvalues of
  `Omega`, and the size obtained with the weighted-limit critical value.

### The weighted chi-square limit

For independent standardized margins with excess kurtoses `kappa_a`, the
asymptotic covariance of `vec(x x' - I)` is

```
Omega[(a,b),(c,d)] = E[x_a x_b x_c x_d] - delta_ab delta_cd,
```

with `E[x_a x_b x_c x_d]` assembled from the pairings of the indices (the
automatic function `omega_matrix` in the study script builds it). The leading
term of the expansion of `M` is a quadratic form in the sample covariances, so

```
(1 - c) M  ->  0.5 * sum_i lambda_i * chi2(k - 1),
```

where `lambda_i` are the positive eigenvalues of `Omega`. For the normal
population `Omega` has `p (p + 1) / 2` eigenvalues equal to 2 and the limit
collapses to `chi2(nu)`; the script checks this numerically for `p = 2, 3, 4`
(`checks.normal_limit_is_chi2`). For example, exponential margins with
`k = 3`, `p = 2` give eigenvalues `(8, 8, 2)`, so the limit is
`4 chi2(2) + 4 chi2(2) + 1 chi2(2)`, not a scaled `chi2(6)`. The class's
`p = 1` case must reduce exactly to Bartlett's numerator `M` and to its
correction constant `c`, which the script verifies to `3.6e-15`
(`checks.p1_reduces_to_bartlett`).  The *corrected* statistics differ at
`O(c^2)`: the Bartlett class divides by `1 + c` while Box's M multiplies by
`1 - c`.

## Results

All numbers below are from the default run (`N = 10,000`, seed 2024); the full
table is in `results/box_m_null_results.json`.

### Normal populations: the chi-square reference is adequate

| n (per group) | median `D_KS` | max `D_KS` | size at 5% | size at 1% |
|---|---|---|---|---|
| 10  | 0.0098 | 0.0300 | 0.048--0.061 | 0.010--0.014 |
| 20  | 0.0086 | 0.0134 | 0.045--0.052 | 0.009--0.012 |
| 50  | 0.0087 | 0.0136 | 0.046--0.054 | 0.008--0.011 |
| 100 | 0.0104 | 0.0152 | 0.045--0.051 | 0.008--0.011 |
| 200 | 0.0086 | 0.0133 | 0.046--0.051 | 0.007--0.010 |

(45 configurations over `k in {2, 3, 5}`, `p in {2, 3, 4}`, `n in {10, 20, 50,
100, 200}`.) The 95% KS threshold at `N = 10,000` is `1.358/sqrt(N) = 0.0136`;
for `n >= 20` three of 36 configurations sit marginally above it and the maximum
is 0.0152, which is what 36 simultaneous 5% tests produce. Type I errors stay
within `[0.045, 0.054]` at the 5% level. The Bartlett correction is what makes
this work at small `n`: to first order, the uncorrected null law of `M` has mean
`nu (1 + c)` (exact normal-theory means differ by `O(n^-2)`), and the empirical
size at `n = 10` reaches 0.295 instead of 0.05.

### Non-normal populations with H0 true: the reference breaks down

Covariance matrices are identical by construction; only the marginal
distribution changes (`k = 3`, `p = 2`, so `nu = 6`).

| marginals | excess kurtosis | size at n=200 | `D_KS` vs `chi2(6)` | `D_KS` vs weighted limit | size at n=1000 with weighted critical value |
|---|---|---|---|---|---|
| normal          | 0.0  | 0.050 | 0.008 | 0.009 | -- |
| Student t(8)    | 1.5  | 0.196 | 0.243 | 0.026 | 0.049 |
| exponential     | 6.0  | 0.597 | 0.590 | 0.032 | 0.048 |
| chi-square(1)   | 12.0 | 0.800 | 0.755 | 0.044 | 0.045 |
| lognormal(0.5)  | 5.9  | 0.563 | 0.563 | 0.059 | 0.048 |

The empirical mean of `(1 - c) M` is far from `nu` and tracks the weighted
limit: e.g. exponential has mean 17.28 at `n = 200` against the limit 18.0
(nominal `nu = 6`). `D_KS` against the weighted limit decreases roughly as
`n^(-1/2)` and reaches 0.010--0.019 at `n = 1000`; using the weighted-limit 5%
critical value then restores the size to 0.045--0.049. A single scaled
chi-square `a * chi2(b)` (two-moment Satterthwaite) is a good but not exact
description (`D_KS` 0.027--0.063 at `n = 200` versus 0.009 for the exact
normal case): the null law is genuinely a weighted sum with unequal weights,
not the nominal chi-square multiplied by a constant.

### Unequal group sizes

The correction keeps working when the groups are unbalanced
(`D_KS <= 0.0141`, size in `[0.045, 0.056]` for
`n = (20,20,20)`, `(15,25,40)`, `(20,30,60)`, `(12,80)`, `(10,10,100)`).

## Files

```
study_box_m_null.py            the study (CLI, no arguments required)
results/box_m_null_results.json  every row + analytic checks + metadata
results/box_m_null_results_quick.json  optional smoke-run output (not committed)
figures/fig_box_m_normal_adequacy.png    D_KS and size vs n under normality
figures/fig_box_m_nonnormal.png          size inflation; distances; density overlay
figures/fig_box_m_limit_convergence.png  mean/variance vs the weighted-chi2 limit
```

## Caveats

* The weighted-chi-square result is an *asymptotic* statement about the leading
  quadratic term of `M`. Finite-sample deviations of order `n^(-1/2)` are
  visible in the non-normal rows (mean ratios 0.66--0.98 at `n = 20--200`), and
  they are why the weighted critical value is conservative at `n = 20--50` and
  accurate by `n = 1000` for these populations.
* The derivation assumes independent, standardized marginals; this is the
  simplest way to keep `Sigma_j = I` for every group while changing only the
  fourth moments. Any common linear transformation of these data leaves `M`
  unchanged, so the simulation covers that family too.
* `N = 10,000` gives a KS standard error of about `1.4e-3` at the centre of the
  distribution and a type I standard error of `2.2e-3` at the 5% level; the
  Wilson intervals are included in the results file.
