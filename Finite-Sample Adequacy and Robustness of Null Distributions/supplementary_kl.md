# Supplementary Material: Kullback-Leibler Divergence Results

## S1. Overview

This supplement reports the Kullback-Leibler (KL) divergence results for Study 1 (finite-sample convergence). As discussed in the main text (§3.2), KL divergence is relegated to Supplementary Material due to a known boundary bias in the Gaussian kernel density estimator when applied to distributions with non-negative support ($\chi^2$, $F$). Section S3 provides a detailed account of this issue.

## S2. KL Divergence Values (Study 1)

Table S1 reports $D_{KL}(\hat{f} \parallel f_0)$ computed via Gaussian KDE with Silverman's bandwidth, for the baseline configuration ($N = 10{,}000$ MC replications, $k = 3$ groups where applicable). A value of 0.0000 indicates the KL divergence was numerically indistinguishable from zero at the grid resolution used.

**Table S1: KL divergence under baseline conditions ($N = 10{,}000$)**

| Test | $n=5$ | $n=10$ | $n=20$ | $n=30$ | $n=50$ | $n=100$ | $n=200$ | $n=500$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1: Student's t | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0007 | 0.0000 | 0.0005 |
| T2: ANOVA F | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| T3: Kruskal-Wallis | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| T4: $\chi^2$ GOF | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| T5: Bartlett's | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| T6: Fligner-Killeen | 0.0742 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| T7: Sign test | 0.0003 | 0.0003 | 0.0007 | 0.0013 | 0.0010 | 0.0021 | 0.0031 | 0.0046 |
| T8: Cochran's Q | 0.0172 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| T9: Median test | 0.2564 | 0.1536 | 0.0056 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| T10: Hotelling's $T^2$ | 0.2073 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

**Key observations:**

- Only four test-configuration pairs yield a non-zero KL value at $n \geq 10$: none. All non-zero values occur at $n = 5$ (Median test: 0.2564; Hotelling's $T^2$: 0.2073; Fligner-Killeen: 0.0742; Cochran's Q: 0.0172; Sign test: 0.0003) or reflect numerical noise at larger $n$ (e.g., Student's t at $n=100$: 0.0007).

- The KL divergence drops to zero far more quickly than the KS distance. For example, Cochran's Q has KL = 0.0172 at $n=5$ and KL = 0.0000 at $n=10$, yet its KS distance remains at 0.060 at $n=10$ and does not fall below the 0.0136 adequacy threshold until $n=200$ (see Table 3, main text). This illustrates precisely why KL is not a reliable convergence metric in this setting: the density estimate is so badly biased near the boundary that the KL integral fails to detect distributional differences that the KS distance and Type I error rate both capture.

- The sign test's KL values actually *increase* with $n$ (0.0003 at $n=5$ to 0.0046 at $n=500$), despite the fact that the binomial PMF is exact and the distributional fit is perfect (KS = 0.005–0.008 across all $n$). This artefact arises because the discrete-to-continuous KDE approximation introduces spurious density differences that grow with the number of support points.

**Conclusion:** The KL divergence values in Table S1 are not interpretable as genuine measures of distributional discrepancy. The systematic boundary bias for $\chi^2$ and $F$ distributions renders the KL estimates unreliable for cross-test comparisons, and the metric's rapid collapse to zero fails to capture the slow convergence behaviour — particular for the median test and Cochran's Q — that the KS distance reveals convincingly.

## S3. Boundary Bias in Gaussian KDE for KL Estimation

### S3.1 Mechanism

The Gaussian KDE estimate at point $x$ is:

$$\hat{f}_h(x) = \frac{1}{N h} \sum_{i=1}^{N} \phi\!\left(\frac{x - X_i}{h}\right)$$

where $\phi$ is the standard normal density (symmetric, support $(-\infty, \infty)$).

For any observation $X_i$, the proportion of its kernel that spills into the negative half-line is:

$$\int_{-\infty}^{0} \frac{1}{h}\,\phi\!\left(\frac{x - X_i}{h}\right) dx = \Phi\!\left(-\frac{X_i}{h}\right)$$

where $\Phi$ is the standard normal CDF. Each observation contributes mass 1 (normalised); the fraction $\Phi(-X_i/h)$ leaks to $x < 0$.

### S3.2 Numerical Example: $\chi^2(2)$

Six of the ten tests (T3–T6, T8, T9) have $\chi^2$ null distributions supported on $[0,\infty)$. For $\chi^2(2)$ with $N = 10{,}000$, Silverman's bandwidth is:

$$h = 1.06 \cdot \hat{\sigma} \cdot N^{-1/5}$$

With $\sigma = \sqrt{4} = 2$:

$$h \approx 1.06 \times 2 \times 10000^{-0.2} = 0.336$$

Leakage fractions for observations at various distances from the boundary:

| Observation $X_i$ | Leakage $\Phi(-X_i/h)$ | Interpretation |
|:---:|:---:|:---|
| 0.05 | 0.44 | 44% of kernel mass spills into $x < 0$ |
| 0.5 | 0.068 | 6.8% leakage |
| 2.0 | $\approx 0$ | negligible |

### S3.3 Consequences

| Effect | Cause |
|--------|-------|
| KDE produces $\hat{f}(x) > 0$ for some $x < 0$ | Kernels of observations near zero extend into the negative half-line |
| True density $f_0(x) = 0$ for all $x < 0$ | $\chi^2$ and $F$ have support $[0, \infty)$ |
| Density near $x = 0$ is underestimated | Probability mass leaked to $x < 0$ must come from somewhere; the normalisation constraint depresses the estimate on $[0, \infty)$ |
| Bias is worst for small degrees of freedom | For $\chi^2(2)$, the density is unbounded at $x = 0$ with $f_0(0) = 0.5$; observations pile densely near the boundary, maximising total leakage |

### S3.4 Possible Remedies

| Method | Drawback |
|--------|----------|
| **Reflection method**: reflect observations at $x = 0$, estimate on $(-\infty, \infty)$, fold back | Distorts the density shape near the boundary |
| **Logarithmic transformation**: transform $X_i \to \ln(X_i)$, KDE on $(-\infty, \infty)$, back-transform | Back-transformation introduces Jacobian instabilities near zero |
| **$k$-nearest-neighbour KL estimator** (Wang et al., 2009) | Avoids bandwidth selection and respects bounded support, but introduces its own bias for small $n$ |
| **Boundary-corrected kernels** (beta, gamma kernels) | Kernel family choice is distribution-specific; loses generality across the ten tests |

Given that (a) the KS distance already provides a non-parametric, distribution-free, boundary-unbiased CDF-level comparison, and (b) the empirical Type I error rate provides the most practically relevant tail-level diagnostic, we elected not to pursue a repaired KL estimate. The values in Table S1 are reported for completeness at the standard Gaussian KDE, without correction.

## Reference

Wang, Q., Kulkarni, S. R., & Verdú, S. (2009). Divergence estimation for multidimensional densities via $k$-nearest-neighbor distances. *IEEE Transactions on Information Theory*, 55(5), 2392–2405.
