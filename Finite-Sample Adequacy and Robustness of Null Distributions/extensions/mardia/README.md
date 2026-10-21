# Mardia's tests: null-distribution study

Monte Carlo verification of the null distributions of Mardia's multivariate
skewness and kurtosis tests, implemented in
[`mc.samplings.Mardia_Test`](../../../src/mc/samplings/_mardia_test.py).  The scripts
put `../src` on `sys.path`, so they run from a clone without an installation;
they use the toolkit's runtime dependencies (`numpy`, `scipy`, `matplotlib`,
`tqdm`).

Mardia (1970) generalises skewness and kurtosis to `p`-variate data.  With
`D^2 = (X - mu)' Sigma^{-1} (X - mu)` the squared Mahalanobis distance,

```
beta_{1,p} = E[ ((X - mu)' Sigma^{-1} (Y - mu))^3 ]     (skewness)
beta_{2,p} = E[ (D^2)^2 ]                                (kurtosis)
```

and a multivariate normal population has `beta_{1,p} = 0`,
`beta_{2,p} = p(p+2)`.  For a multivariate t with `nu > 4` degrees of freedom,
standardized to its true covariance matrix, `beta_{2,p} = p(p+2)(nu-2)/(nu-4)`.

The sample statistics use the covariance matrix with divisor `n` (as in the
class), `S = (1/n) sum_i (X_i - Xbar)(X_i - Xbar)'`, and
`d_ij = (X_i - Xbar)' S^{-1} (X_j - Xbar)`:

```
b_{1,p} = (1/n^2) sum_i sum_j d_ij^3,      b_{2,p} = (1/n) sum_i d_ii^2,
n b_{1,p} / 6  ->  chi2( p(p+1)(p+2)/6 ),
Z_K = (b_{2,p} - p(p+2)) / sqrt(8 p(p+2)/n)  ->  N(0, 1).
```

## Running

Run from the repository root (the script locates `src/` itself):

```bash
python3 "Finite-Sample Adequacy and Robustness of Null Distributions/extensions/mardia/study_mardia_null.py"
python3 "Finite-Sample Adequacy and Robustness of Null Distributions/extensions/mardia/study_mardia_null.py" --quick
python3 "Finite-Sample Adequacy and Robustness of Null Distributions/extensions/mardia/study_mardia_null.py" --no-figures
```

or from the package root (`Finite-Sample Adequacy and Robustness of Null Distributions/`):

```bash
python3 extensions/mardia/study_mardia_null.py            # full run, ~2.5 min
python3 extensions/mardia/study_mardia_null.py --quick    # smoke run, ~30 s
python3 extensions/mardia/study_mardia_null.py --no-figures
```

The full run writes `results/mardia_null_results.json`; a `--quick` smoke run
writes `results/mardia_null_results_quick.json` instead, so it cannot overwrite
the canonical full-run artifact.  The full run uses `N = 10,000` replications
per configuration (2,000 for the `n = 1000` multivariate-t points, which are
expensive because every replication forms an `n x n` matrix of Mahalanobis
products) and seed `2024`; see the results file for every row and the metadata.
The analytic consistency checks use fixed internal seeds and are therefore not
affected by `--seed`.

## What the study measures

* **Null calibration under multivariate normality** on a grid of
  `p in {1,2,3,5,10}` and `n in {10,20,50,100,200}` (the grid keeps
  `n > p + 1`, so for example `p = 10` starts at `n = 20`):
  the mean and variance of `n b1/6` against `chi2(dof)`, the mean and standard
  deviation of `Z_K` against `N(0,1)`, KS distances, and empirical type I
  errors at 1% and 5%.
* **The exact finite-sample mean of the kurtosis statistic.**  Under
  multivariate normality `E[b_{2,p}] = p(p+2)(n-1)/(n+1)`, so the asymptotic
  `Z_K` is biased.  The script checks this formula against the Monte Carlo
  mean and reports the bias in standard errors.
* **Analytic consistency checks**: with `p = 1` the Mardia statistics reduce to
  the univariate ones (`n b1/6 = n g1^2/6`, `b2 = m4/m2^2`); for `p = 1` the
  finite-sample mean of the skewness statistic is
  `n(n-2)/((n+1)(n+3))`, which the simulation reproduces.
* **Non-normal populations**: multivariate t with 5, 10 and 30 degrees of
  freedom and independent exponential margins.  The script estimates the
  population `beta1`, `beta2`, the rejection rates (power) of both tests and
  the convergence of `E[b2]` to `p(p+2)(nu-2)/(nu-4)`.

## Results

### Under multivariate normality

Selected rows (`N = 10,000`, seed 2024).  `size skew` is the upper-tail
rejection rate of the skewness test; `size kurt` is the two-sided rate of the
kurtosis test.  Both use the nominal 5% reference.

| p | n | E[n b1/6]/dof | size skew | E[b2]/[p(p+2)] | E[Z_K] | sd(Z_K) | size kurt | size kurt (upper) |
|---|---|---|---|---|---|---|---|---|
| 2  | 10  | 0.627 | 0.004 | 0.818 | -0.58 | 0.432 | 0.000 | 0.000 |
| 2  | 50  | 0.915 | 0.045 | 0.962 | -0.27 | 0.854 | 0.019 | 0.027 |
| 2  | 200 | 0.975 | 0.051 | 0.991 | -0.12 | 0.969 | 0.039 | 0.045 |
| 3  | 200 | 0.979 | 0.049 | 0.990 | -0.20 | 0.958 | 0.042 | 0.040 |
| 10 | 20  | 0.847 | 0.000 | 0.905 | -1.65 | 0.356 | **0.199** | 0.000 |
| 10 | 200 | 0.982 | 0.044 | 0.990 | -0.56 | 0.920 | 0.062 | 0.013 |

The asymptotic references are the correct limits but convergence is slow and
the finite-sample null distributions are far from them:

* The skewness statistic has mean below `dof` (by 37% at `p=2,n=10`, 2.5% at
  `p=2,n=200`) and is strongly under-dispersed at small `n` (its variance is
  0.40 of `2 dof` at `p=2,n=10`), so the 5% skewness test is conservative:
  `size` ranges over 0.000--0.051 on the grid, is 0.027 at `p=1,n=20`, and
  reaches 0.05 only for large `n`.  At larger `n` the variance slightly exceeds
  `2 dof` (up to 1.13 of it), but the mean stays below `dof`.
* The kurtosis statistic `Z_K` has a strongly negative finite-sample mean and
  `sd(Z_K) < 1`; the exact finite-sample mean formula
  `E[b2] = p(p+2)(n-1)/(n+1)` matches the simulation to within 2.4 Monte Carlo
  standard errors.  The corresponding `E[Z_K]` check is algebraically the same
  statement (the two JSON values agree to roundoff), so it is one piece of
  evidence, not two.
* The two-sided kurtosis test can therefore **over-reject badly**: at
  `p = 10, n = 20` its size is 0.199, almost entirely from the lower tail
  (the upper-tail size is 0.000), because `E[Z_K] = -1.65`.  Centring `Z_K` by
  the exact finite-sample mean removes that rejection (size 0.000) but leaves
  the test ultra-conservative, since the finite-sample variance is only
  `0.356^2` of the nominal one.  This is the finite-sample/Edgeworth problem on
  a second, independent test family.
* The KS distance to the nominal references at `n = 200` is 0.016--0.029 for the
  skewness statistic (`p <= 3`) and 0.09--0.24 for `Z_K` (0.096 at `p = 2`);
  the kurtosis reference is still far from adequate at the end of the grid.

### Non-normal populations

| population | p | n | E[b1] (theory) | E[b2] (theory) | power skew | power kurtosis |
|---|---|---|---|---|---|---|
| t(5)  | 2 | 50  | 2.09 (0) | 11.89 (24.00) | 0.507 | 0.568 |
| t(5)  | 2 | 200 | 1.65 (0) | 15.76 (24.00) | 0.776 | 0.989 |
| t(10) | 2 | 200 | 0.34 (0) | 10.22 (10.67) | 0.390 | 0.730 |
| t(30) | 2 | 200 | 0.15 (0) | 8.47 (8.62)  | 0.115 | 0.173 |
| exp   | 2 | 200 | 7.59 (8.00) | 17.70 (20.00) | 1.000 | 1.000 |

* The multivariate-t kurtosis formula is supported by the large-`n` run:
  at `n = 1000`, `E[b2]/beta2` is 0.818 for `t(5)`, 0.992 for `t(10)` and
  0.995 for `t(30)`.  For `t(10)` and `t(30)` this is a verification within
  Monte Carlo error; for `t(5)` it is only a slow-convergence observation,
  because `b2` has infinite variance when `nu <= 8` (so no finite Monte Carlo
  standard error exists).  The `t(5)` sample kurtosis converges very slowly: the fourth moment is
  finite, but the higher moments that control its sampling variability are not
  (the variance of `b2` needs eighth moments, which require `nu > 8`), so the
  finite-sample bias decays very slowly; at `n = 200` the ratio is still 0.659.  This is exactly the kind of
  finite-sample (Edgeworth-regime) effect that the correction terms have to
  capture.
* Both tests pick up the alternatives: exponential margins (which are skewed,
  `beta1 = 8` for `p = 2`) are rejected with power 1.0 by both tests at
  `n = 200`; `t(5)` is rejected with power 0.776 (skewness) and 0.989
  (kurtosis); `t(30)`, which is close to normal, is much harder (0.115, 0.173).

## Derivation of the finite-sample mean of b2

With `S` using divisor `n`, the diagonal Mahalanobis products satisfy
`d_ii = (n-1) B_i` with `B_i ~ Beta(p/2, (n-p-1)/2)`.  Hence

```
E[b2] = (1/n) sum_i E[d_ii^2]
      = (n-1)^2 E[B^2]
      = (n-1)^2 * p(p+2) / ((n-1)(n+1))
      = p(p+2)(n-1)/(n+1).
```

This gives `E[Z_K] = (p(p+2)(n-1)/(n+1) - p(p+2)) / sqrt(8p(p+2)/n)`, which is
about `-sqrt(p(p+2)/(2n))` for large `n` and explains both the sign and the
magnitude of the observed bias.  The script verifies both formulas against the
simulation.

## Files

```
study_mardia_null.py               the study (CLI, no arguments required)
results/mardia_null_results.json   all rows + analytic checks + metadata
figures/fig_mardia_skew_null.png   skewness null: mean/dof and type I error
figures/fig_mardia_kurt_null.png   kurtosis null: E[b2], E[Z_K], sd(Z_K)
figures/fig_mardia_nonnormal.png   power, moment estimates, t-limit convergence
```

## Caveats

* The references are asymptotic: `n b1/6 -> chi2` and `Z_K -> N(0,1)`.  The
  study deliberately keeps the class's canonical (uncorrected) statistics;
  the comparison with the exact finite-sample mean is a diagnostic, not a
  modified test.  The class's `expected_kurtosis_finite` is the normal-theory
  expectation and is not the expectation of `b2` under the non-normal parents.
* The `p` values in the grid are at most 10 and the grid keeps `n >= p + 2`
  (the smallest grid value is `n = 10`); the over-rejection at `p = 10, n = 20`
  is therefore an extreme corner of the `p/n` regime, but it is a legitimate
  corner and is reported as such.
* Monte Carlo standard errors: with `N = 10,000` the type I standard error is
  `2.2e-3` at the 5% level and the KS standard error is about `1.4e-3`.
  Wilson intervals are included in the results file for the null type-I rates
  only; the power rows of the non-normal study do not carry intervals.
* The multivariate t is standardized to its true covariance matrix, so the
  identity covariance assumption of the class holds for every parent.
