# Simulation design contract

This file states the conventions that every script in `code/` relies on. It was
an internal working document during the revision; what remains here is only the
part that is load-bearing for reading or extending the code.

## 1. Test generators

```python
def mc_<name>(<params>, N, **scenario_kwargs) -> np.ndarray   # N statistics
```

Each generator returns an array of `N` statistic values drawn under the
configuration named by its `scenario_kwargs` (for example `kind='iid'` for KPSS,
`dep='indep'` for Hoeffding's D). Generators take an explicit `rng`
(`numpy.random.Generator`) where they are used by the revision scripts, so any
single cell is reproducible in isolation; the archived `experiments.py`
generators draw from the legacy global NumPy stream and are therefore
reproducible only by running that script in order.

## 2. Registry

```python
TESTS['T11_kpss'] = dict(
    label='KPSS',
    family='CvM limit',          # null-distribution family, for grouping in figures
    null='cvm_limit',            # name of the reference CDF/PPF
    null_args=(),                # arguments of the reference law (must be n-independent)
    type1='upper_tail',          # 'upper_tail' | 'lower_tail' | 'two_sided' | 'binom'
)
```

Rules:

* **The theoretical null must not depend on the sample size.** If a statistic's
  finite-sample null depends on `n`, standardise it first and document the
  standardisation in the generator docstring and in the registry entry. Hoeffding's
  D is standardised this way; KPSS needs no rescaling because its Lagrange-multiplier
  statistic already converges to the parameter-free Cramér-von Mises law.
* **Quantile conventions differ by law and are asserted, not assumed.**
  `reference_ppf(q, ...)` takes an **upper-tail probability** for every reference
  (`stats.*.isf` for scipy-backed laws, `hoeffding_z_ppf` for T12). `cvm_limit_ppf`
  follows the same convention; `cvm_limit_quantile` is the ordinary lower quantile
  and exists so the two can never be confused. `core._selftest` asserts the
  identity of every bespoke reference against an independent implementation and
  must be run before any study.
* The rejection rule for a given nominal level is derived from `type1`, and the
  *same* nominal level is applied to the critical value and to the reference CDF
  used for the KS distance.

## 3. Evaluation

There is exactly one evaluation path, `core._evaluate(stat_values, test_id, alpha)`,
returning

```python
dict(ks=..., type1=..., ci_lo=..., ci_hi=...)
```

so that `D_KS` and `\hat\alpha` have a single definition for every test, every
scenario, and both the null and the power studies. `D_KS` is
`sup_x |F_hat_N(x) - F_0(x)|` computed as `max(D+, D-)` including the left-limit
term, matching `scipy.stats.kstest`; the archive uses `kstest` directly, and the
revision kernel is asserted against it to 1e-12.

## 4. Monte Carlo budget and seeds

* `N = 10,000` replications for every cell of every published grid, so that
  cross-test comparisons are not confounded by unequal simulation resolution.
* Seed `2024` for the published grids. Within a severity ladder all rungs and all
  tests share one seed (common random numbers), which makes the baseline-relative
  change `Delta = alpha_hat_violated - alpha_hat_baseline` a paired comparison.
* The KS adequacy threshold is the two-sided 95% critical value at the fixed
  budget, `1.358 / sqrt(N)`, and is meaningful only relative to that budget.

## 5. Operational criteria

* `n*_alpha`: smallest `n` such that `|alpha_hat - alpha| < 0.01` holds at that `n`
  **and at every larger `n` in the grid** (the monotone reading used in Table 4 of
  the manuscript; the first-crossing reading is also reported where the two differ,
  because the difference is informative for lattice-valued statistics).
* `n*_KS`: smallest `n` with `D_KS` below the threshold above; "not attained"
  within the grid otherwise.
* `N*`: smallest `N` with `CV(alpha_hat) < 0.10` across 30 independent runs.

## 6. Results files

Every results file carries a `meta` block with the seed, the package versions
actually used, and the script that wrote it. The revision scripts write their
output **beside the script**, which is why the code lives in `code/` and the
published results files sit in the package root; `code/` is the working directory
for a reproduction run.

| file | written by | principal keys |
|---|---|---|
| `experiment_results.json` | `code/experiments.py` | `study1`, `study2`, `study3`, `study1_nstar_*`, `study3_nstar_mc` |
| `rev_results.json` | `code/rev_sensitivity.py` | `severity_sweep`, `study2_nsweep`, `alpha_sweep`, `fligner_variant_probe`, `implementation_validation` |
| `rev_results_t1.json` | `code/rev_study1_ext.py`, merged by `code/rev_alpha_dependence.py` | `study1_ext`, `study1_ext_nstar`, `null_calibration`, `t12_reference_diagnostic`, `t13_split_diagnostic`, `alpha_sweep_13`, `dependence_13` |
| `rev_results_power.json` | `code/rev_power.py` | `study4_power`, `recommendation_matrix` |
| `rev_results_newtests_power.json` | `code/verify_new_tests_power.py` | `T11_kpss`, `T12_hoeffding_d` |

## 7. Figures

`figures/` holds the published panels. Style is fixed across the package
(`font.size = 9`, `dpi = 150`) so that panels can be compared directly; every
panel carries axis labels with units and no legend may overlap a data mark
(Referee 2 raised this about the original Figure 2). The scripts write new
figures beside themselves in `code/`; move a regenerated panel into `figures/`
before rebuilding the manuscript.
