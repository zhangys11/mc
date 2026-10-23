# Stage 1 diagnostic prototype (exploratory, not part of paper.md)

Generated from `results/experiment_results.json`.  These diagnostics are prototypes for Route A, not validated selection rules.

## 1. R_B and mean-shift diagnostics for Box's $M$ weights ($p=2$, $k=2$)

| parent | per-contrast weights | $R_B$ | mean shift | observed size at $n=50$ |
|---|---|---:|---:|---:|
| normal | 1.000, 1.000, 1.000 | 0.0000 | 0.000 | 0.0543 |
| t5 | 5.000, 3.000, 3.000 | 0.0661 | 2.667 | 0.2969 |
| lognormal | 56.468, 56.468, 1.000 | 0.4740 | 36.979 | 0.7458 |

The lognormal row is the corrected independent-coordinate calculation (56.468, 56.468, 1); its R_B is 0.474 and its mean is 37.98 times the normal mean.  The t5 row has a moderate shape mismatch but a large mean shift (11/3 = 3.667).  R_B alone therefore separates shape mismatch, not the total calibration error.

## 2. Bartlett's test: equal weights, scale error only

| parent | excess kurtosis $\gamma_2$ | per-contrast weight | $R_B$ | mean shift | observed size at $n=50$ |
|---|---:|---:|---:|---:|---:|
| normal | 0.0 | 1.000 | 0.0000 | 0.000 | 0.0479 |
| exponential | 6.0 | 4.000 | 0.0000 | 3.000 | 0.4288 |
| $t_5$ | 6.0 | 4.000 | 0.0000 | 3.000 | 0.2760 |

Bartlett is the key counter-example to using R_B alone: its weights remain equal under non-normality, so R_B = 0 and yet the size error is enormous (0.429 under the exponential parent; the t5 value differs because higher-order terms differ).  The scalar kurtosis correction is a mean-scale fix; a spectral diagnostic must therefore be paired with an R_B-like mean-shift diagnostic.

## 3. Sign-test attainable-size error R_C

| $n$ | exact attainable size | exact $R_C$ | mid-$p$ attainable size | mid-$p$ $R_C$ |
|---:|---:|---:|---:|---:|
| 20 | 0.04139 | 0.00861 | 0.04139 | 0.00861 |
| 30 | 0.04277 | 0.00723 | 0.04277 | 0.00723 |
| 50 | 0.03284 | 0.01716 | 0.06491 | 0.01491 |
| 100 | 0.03520 | 0.01480 | 0.05689 | 0.00689 |

At $n=50$ the largest exact attainable size not exceeding 0.05 is 0.03284 ($R_C=0.01716$), while mid-$p$ reaches 0.06491 ($R_C=0.01491$); observed exact-sign sizes in the simulation are 0.0307-0.0352 and mid-$p$ sizes are 0.058-0.069, within Monte Carlo error of these lattice values.

## 4. Median-test attainable-size error R_C (balanced design)

| $k$ | $n$ | attainable size | $R_C$ | critical value |
|---:|---:|---:|---:|---:|
| 3 | 10 | 0.03418 | 0.01582 | 7.200 |
| 4 | 10 | 0.03679 | 0.01321 | 8.800 |
| 4 | 20 | 0.04629 | 0.00371 | 8.000 |

Median-test R_C is design-dependent (0.0037-0.0158 in these small balanced designs); the exact distribution is hypergeometric.  Cochran's $Q$ and the rank tests need their own exact enumeration and are left pending in this prototype.

## 5. T1 $t$-test: analytic Edgeworth diagnostic $R_A$

| parent | $\gamma_1$ | $n$ | $x$ | $c_A(x)$ | $R_A$ | observed two-sided size at $n=50$ |
|---|---:|---:|---:|---:|---:|---:|
| normal | 0.0 | 50 | 1.960 | +0.00000 | 0.00000 | 0.0524 |
| exponential | 2.0 | 50 | 1.960 | +0.16915 | 0.02392 | 0.0663 |
| $t_5$ | 0.0 | 50 | 1.960 | +0.00000 | 0.00000 | 0.0455 |

For the two-sided $t$-test the leading coefficient is even in $x$, so the O($n^{-1/2}$) size errors of the two symmetric tails cancel; $R_A$ is a CDF-level diagnostic, not a two-sided size prediction.  A size-oriented version needs the next-order term or the one-sided tail functional.

## 6. Simulation-calibrated size-error scaling for ratio statistics

| test | $n$ | nominal-reference size | error |
|---|---:|---:|---:|
| T2 ANOVA $F$ | 20 | 0.0452 | -0.0048 |
| T2 ANOVA $F$ | 50 | 0.0502 | +0.0002 |
| T2 ANOVA $F$ | 100 | 0.0483 | -0.0018 |
| T2 ANOVA $F$ | 200 | 0.0494 | -0.0006 |
| T10 Hotelling $T^2$ | 20 | 0.1011 | +0.0511 |
| T10 Hotelling $T^2$ | 50 | 0.0807 | +0.0307 |
| T10 Hotelling $T^2$ | 100 | 0.0649 | +0.0149 |
| T10 Hotelling $T^2$ | 200 | 0.0577 | +0.0077 |

Fitted log-log slopes on the positive errors: T2 ANOVA not identifiable (errors change sign); T10 Hotelling -0.83.  Under the exponential parent the ANOVA size error is small and non-monotone in this range, whereas the Hotelling $T^2$ error decays roughly like $n^{-1}$, not $n^{-1/2}$.  The latter is consistent with the parity cancellation of the leading Edgeworth term in regime B; a naive $R_A$ designed for regime A would misclassify it.

## 7. Regime coverage of the prototype

| test | regime | diagnostic implemented here | status |
|---|---|---|---|
| T1 t | A | $R_A$ (analytic, one-sample $t$) | implemented |
| T2 ANOVA $F$ | B | $R_A$/$R_B$ | theory pending (quadratic ratio) |
| T3 Kruskal-Wallis | C | $R_C$ | exact enumeration pending |
| T4 $\chi^2$ GOF | C | $R_C$ | exact multinomial pending |
| T5 Bartlett | B (+scale) | $R_B$, mean shift | implemented |
| T6 Fligner-Killeen | C | $R_C$ | not exact; parent sensitivity |
| T7 Sign | C | $R_C$ | implemented |
| T8 Cochran's $Q$ | C | $R_C$ | exact enumeration pending |
| T9 Median | C | $R_C$ | implemented (balanced design) |
| T10 Hotelling | B | $R_B$ | theory pending |
| T11 Box's $M$ | B | $R_B$, mean shift | implemented |

The prototype deliberately does not claim a validated selector.  Its purpose is to show that the regime-specific diagnostics are computable and that they carry signal before investing in threshold tuning, sample splitting and out-of-sample validation.
