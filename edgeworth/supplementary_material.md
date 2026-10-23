# Supplementary Material

**Finite-Sample Null Calibration Across Eleven Statistical Tests: Mechanisms, Corrections, and Limits**

This supplement contains the simulation configurations, algorithmic validation details, and descriptive summaries that are referenced from the main text but are not needed for the main narrative. Table numbering is S1-S9; the corresponding main-text tables are numbered 1-8, and Table A1 remains in Appendix A.

| Supplement table | Content | Main-text pointer |
|---|---|---|
| S1 | Empirical lattice-test quantiles | §4.3 |
| S2 | Exact critical values of Box's $M$ vs saddlepoint/inversion/Monte Carlo | §5.5 |
| S3 | Assumption-violation simulation design | §6.2 |
| S4 | Fitted convergence exponents and adequacy thresholds | §7.1 |
| S5 | Monte Carlo replication requirements | §7.3 |
| S6 | Real-data case study summary | §7.4 |
| S7 | Exact attainable sizes of the sign-test rules | §7.6 |
| S8 | Discrete-test KS distances at larger simulation budgets | §7.6 |
| S9 | Full denominator diagnostics for $M^*$ | §7.5 |

**Table S1: Empirical 95% quantiles of the lattice-test statistics vs. theoretical $\chi^2(2)=5.99$.**

| Test | $n=5$ | $n=10$ | $n=20$ | $n=30$ | $n=50$ |
|---|---|---|---|---|---|
| Cochran's $Q$ | 5.20 (5.99) | 5.43 (5.99) | 6.00 (5.99) | 6.08 (5.99) | 5.89 (5.99) |
| Median test | 6.96 (5.99) | 5.60 (5.99) | 5.20 (5.99) | 5.60 (5.99) | 5.92 (5.99) |

The lookup values were estimated and evaluated using the same simulation batch; discrete ties further prevent exact 5% calibration. They are therefore descriptive, not externally validated critical values; the calibration evidence in Table 4 is an in-sample movement toward nominal, not a validated size guarantee. An independent calibration set or conditional randomisation would be needed before operational use; bootstrap or permutation calibration (Efron and Tibshirani, 1993; Hall, 1992) would avoid the lookup altogether.

**Table S2: $95\%$ critical values of $M_{\text{raw}}$ under multivariate normality.**

| $p$ | $n$ | $k$ | $\nu$ | saddlepoint | exact inversion | Monte Carlo | $\chi^2_\nu$ | $\chi^2$ error |
|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 2 | 3 | 8.2951 | 8.2870 | 8.2889 | 7.8147 | $-5.72\%$ |
| 2 | 50 | 2 | 3 | 7.9991 | 7.9914 | 7.9976 | 7.8147 | $-2.29\%$ |
| 2 | 100 | 2 | 3 | 7.9088 | 7.9012 | 7.8951 | 7.8147 | $-1.02\%$ |
| 3 | 20 | 2 | 6 | 13.7881 | 13.7852 | 13.7560 | 12.5916 | $-8.46\%$ |
| 3 | 50 | 2 | 6 | 13.0292 | 13.0255 | 12.9974 | 12.5916 | $-3.12\%$ |
| 3 | 100 | 3 | 12 | 21.3399 | 21.3383 | 21.3058 | 21.0261 | $-1.31\%$ |
| 5 | 20 | 2 | 15 | 29.2182 | 29.2183 | 29.2147 | 24.9958 | $-14.44\%$ |
| 5 | 20 | 3 | 30 | 50.2579 | 50.2580 | 50.2769 | 43.7730 | $-12.94\%$ |
| 5 | 50 | 3 | 30 | 46.0269 | 46.0264 | 45.9500 | 43.7730 | $-4.74\%$ |

The $\chi^2_\nu$ critical value is too small in every listed design, causing liberal tests. Errors are largest when $p$ is large relative to group size.

**Table S3: Assumption violation scenarios.**

| Violation | Tests run under it | Implementation | Caveat |
|---|---|---|---|
| Skewed and leptokurtic ($\gamma_1=2, \gamma_2=6$) | T1, T2, T5, T10 (parametric); T3, T6, T7, T9 (inert or approximate) | $N(0,1) \to \text{Exp}(1)-1$ | Kruskal–Wallis is exactly distribution-free, so for T3 this scenario changes nothing; for T6 the statistic ranks deviations from estimated medians and is only asymptotically distribution-free, so the change is real; T7 has a binomial null |
| Symmetric heavy-tailed ($\gamma_2=6$) | same set | $N(0,1) \to t(5)$ | as above |
| Variance heterogeneity | T2, T3, T9 | $\sigma_1^2=1,\ \sigma_k^2=4$ | for T3 and T9 the null of equal distributions is now **false**, so a rejection is a correct rejection (power), not a size error |
| Unbalanced design | T2, T3, T5, T6, T9 | Group sizes 1:2:3 | Cochran's $Q$ has no unbalanced version — it is defined on a complete subject $\times$ condition table — so T8 is not run here |
| Small expected frequencies | T4, T8, T9 | T4: $n p_j$ as low as 2; T8: $p=0.05$; T9: $n=8$/group | — |

**Table S4: Fitted convergence exponents $\hat b$ and adequacy thresholds.**

| Test                 | Mechanism            | Fitted $\hat b$ | 95% CI for $\hat b$ | $n^*_\alpha$ | $n^*_{KS}$ |
| -------------------- | -------------------- | --------------- | ------------------- | ------------ | ---------- |
| T1: Student's $t$    | A (mean), exact      | 0.050           | [−0.142, 0.241]     | 5            | 5          |
| T2: ANOVA $F$        | B (quadratic), exact | −0.081          | [−0.237, 0.075]     | 5            | 5          |
| T3: Kruskal-Wallis   | C (rank)             | −0.348          | [−0.600, −0.096]    | 5            | 20         |
| T4: $\chi^2$ GOF     | C (multinomial)      | −0.430          | [−0.603, −0.257]    | 5            | 50         |
| T5: Bartlett         | B + kurtosis scale   | −0.060          | [−0.225, 0.105]     | 5            | 5          |
| T6: Fligner-Killeen  | C (rank)             | −0.643          | [−0.936, −0.350]    | 30           | 100        |
| T7: Sign test        | C (lattice sum), exact | 0.057         | [−0.140, 0.254]     | 20           | 5          |
| T8: Cochran's $Q$    | C (block count)      | −0.582          | [−0.679, −0.484]    | 5            | 200        |
| T9: Median test      | C (count)            | −0.524          | [−0.590, −0.459]    | 20           | $>500$     |
| T10: Hotelling $T^2$ | B (quadratic), exact | −0.051          | [−0.213, 0.112]     | 5            | 5          |
| T11: Box's $M$       | B (quadratic), exact under MVN | 0.094  | [−0.306, 0.494] | 10 | 10 |

*Notes:* $\hat b$ is fitted across the stated sample-size grid; $n^*$ values and intervals are empirical. Entries at the KS floor have no interpretable convergence slope.

**Table S5: Minimum $N^*$ for CV $<0.10$.**

| Test | $N^*$ | CV at $N=1{,}000$ | CV at $N=10{,}000$ |
|---|---|---|---|
| T1: $t$ | 3,000 | 0.122 | 0.050 |
| T2: ANOVA | 1,500 | 0.138 | 0.043 |
| T3: K-W | 2,000 | 0.142 | 0.040 |
| T4: $\chi^2$ GOF | 1,500 | 0.130 | 0.044 |
| T5: Bartlett | 2,000 | 0.169 | 0.040 |
| T6: F-K | 1,500 | 0.144 | 0.039 |
| T7: Sign | 5,000 | 0.164 | 0.056 |
| T8: Cochran | 1,500 | 0.158 | 0.047 |
| T9: Median | 3,000 | 0.138 | 0.046 |
| T10: Hotelling | 2,000 | 0.100 | 0.043 |

**Table S6: Summary of real-data case studies.**

| Case | Dataset | Test | Key parameter | p-value | Conclusion at $\alpha=0.05$ |
|---|---|---|---|---:|---|
| 1 | Wine `malic_acid` ($n=178$) | Bartlett | $\hat\gamma_2=1.66$ | 0.0022 | Reject (strong) |
| 1 | Wine `malic_acid` ($n=178$) | $B^*$ (kurtosis-corr.) | $1+\hat\gamma_2/2=1.83$ | 0.0354 | Reject (marginal) |
| 2 | Wine `alcohol` ($n=178$) | Bartlett | $\hat\gamma_2=-0.22$ | 0.4496 | Do not reject |
| 2 | Wine `alcohol` ($n=178$) | $B^*$ (kurtosis-corr.) | $1+\hat\gamma_2/2=0.89$ | 0.4081 | Do not reject |
| 3 | mtcars ($n=32$), $H_0$: median=16 | Sign test (exact) | $N^+=22/32$ | 0.0501 | Do not reject |
| 3 | mtcars ($n=32$), $H_0$: median=16 | Sign test (mid-$p$) | $N^+=22/32$ | 0.0351 | Reject |

**Table S7: Exact attainable sizes of the two sign-test rules at $n=50$ (binomial, no simulation).**

| nominal $\alpha$ | exact attainable size | mid-$p$ attainable size | abs. dev. (exact) | abs. dev. (mid-$p$) |
|---|---|---|---|---|
| 0.010 | 0.00660 | 0.00660 | 0.00340 | 0.00340 |
| 0.025 | 0.01535 | 0.03284 | 0.00965 | 0.00784 |
| 0.050 | 0.03284 | 0.06491 | 0.01716 | 0.01491 |
| 0.100 | 0.06491 | 0.11892 | 0.03509 | 0.01892 |

**Table S8: Discrete-test KS distances at $N=10^4$, $5\times10^4$ and (where available) $2\times10^5$.** The table shows whether observed deviations persist as the Monte Carlo floor declines.

| test | $n=10$: $10^4$ / $5\cdot10^4$ / $2\cdot10^5$ | $n=50$: $10^4$ / $5\cdot10^4$ / $2\cdot10^5$ | $n=200$: $10^4$ / $5\cdot10^4$ |
|---|---|---|---|
| T3: Kruskal–Wallis | 0.0224 / 0.0162 / 0.0174 | 0.0095 / 0.0032 / 0.0035 | 0.0059 / 0.0061 |
| T4: $\chi^2$ GOF | 0.0299 / 0.0262 / 0.0278 | 0.0091 / 0.0063 / 0.0052 | 0.0066 / 0.0043 |
| T6: Fligner–Killeen | 0.0407 / 0.0348 / 0.0354 | 0.0163 / 0.0122 / 0.0142 | 0.0059 / 0.0064 |
| T8: Cochran's $Q$ | 0.0596 / 0.0620 / 0.0618 | 0.0280 / 0.0197 / 0.0215 | 0.0101 / 0.0082 |
| T9: median test | 0.2261 / 0.2272 / 0.2281 | 0.0842 / 0.0878 / 0.0886 | 0.0356 / 0.0327 |

**Table S9: Normal and non-normal denominator diagnostics for $M^*$ at nominal 0.05 ($N=10{,}000$).** 'Clipped' floors the denominator at $10^{-6}$; 'unclipped' conditions on positivity; 'constrained' floors it at 0.25. The 'defined' column is the share of positive denominators.

| configuration | median denom. | $P(\text{denom}\le0)$ | $P(\text{denom}<0.25)$ | defined | clipped | unclipped | constrained |
|---|---|---|---|---|---|---|---|
| MVN, $p=2,k=2,n=10$ | 0.21 | 26.5% | 53.8% | 73.5% | 0.61 | 0.47 | 0.44 |
| MVN, $p=2,k=2,n=30$ | 0.70 | 0.7% | 8.3% | 99.3% | 0.22 | 0.21 | 0.20 |
| MVN, $p=2,k=2,n=100$ | 0.90 | 0.0% | 0.0% | 100.0% | 0.09 | 0.09 | 0.09 |
| MVN, $p=3,k=3,n=10$ | -0.38 | 83.3% | 93.7% | 16.7% | 0.98 | 0.88 | 0.93 |
| MVN, $p=3,k=3,n=30$ | 0.49 | 10.1% | 27.9% | 89.9% | 0.56 | 0.51 | 0.55 |
| MVN, $p=3,k=3,n=100$ | 0.83 | 0.0% | 1.0% | 100.0% | 0.22 | 0.22 | 0.22 |
| $t_5$, $p=2,k=2,n=10$ | 0.58 | 9.3% | 24.2% | 90.7% | 0.46 | 0.41 | 0.41 |
| $t_5$, $p=2,k=2,n=100$ | 3.41 | 0.0% | 0.0% | 100.0% | 0.01 | 0.01 | 0.01 |
| lognormal, $p=3,k=3,n=10$ | 0.92 | 6.9% | 14.5% | 93.1% | 0.79 | 0.77 | 0.79 |
| lognormal, $p=3,k=3,n=100$ | 21.38 | 0.0% | 0.0% | 100.0% | 0.00 | 0.00 | 0.00 |
