# Code map — the mc-tk reproducibility package

Manuscript: **COST-D-26-00503R1**, *Finite-Sample Adequacy and Robustness of Null
Distributions: A Systematic Monte Carlo Study Across Thirteen Hypothesis Tests*
(*Computational Statistics*, major revision).

This file answers Referee 2's request that "contents of each code file in the
repository should be explained": one row per file with its purpose, inputs,
outputs, runtime and dependencies, followed by a short section per file. The
function-level companion is [`FUNCTION_REFERENCE.md`](FUNCTION_REFERENCE.md),
generated from the sources by `docs/make_function_reference.py`: **all 178
top-level functions of the eight pipeline scripts carry a docstring** (100%
coverage), as do every `docs/` tool. The final documentation pass over the frozen
revision files was verified to be behaviour-neutral twice over — the code-only AST
digest of each file is unchanged, and the behavioural fingerprint in
`docs/probe_docstring_pass.py` (`31e176014cfb0ad90c8a4b9bd214f5cc4d2a038d7b0b286f3eb12843f653c1db`)
is identical before and after.

All runtimes below were measured on the machine described in
[`ENVIRONMENT.txt`](ENVIRONMENT.txt) (macOS 26.6.2, arm64, 10 cores, CPython
3.13.5, Anaconda). Runtimes quoted for the two long revision runs are the values
those runs wrote into their own `meta` blocks.

### Run order

```
core._selftest                     # analytic guard rails (runs inside every script)
  └─ rev_study1_ext.py             # writes rev_results_t1.json, fig_rev_alpha.png
       └─ rev_alpha_dependence.py  # merges alpha_sweep_13 + dependence_13 into the same file
  └─ rev_sensitivity.py            # independent: rev_results.json + 2 figures
  └─ rev_power.py                  # independent: rev_results_power.json + fig_rev_power.png
  └─ verify_new_tests_power.py     # independent power for T11/T12 (uses core.py only)
  └─ experiments.py                # archive Study 1–3 (writes the R1 files)
     └─ verify_manuscript.py       # read-only: cross-checks draft.md against all JSON files
```

`rev_study1_ext.py` and `rev_alpha_dependence.py` write the *same* results file and
must run in that order, because the second merges its blocks into the file the
first created (`core.update_results`). All other scripts are order-independent.

---

## 1. Scripts

| File | Role | How to run | Wall clock (measured) | Writes |
|---|---|---|---|---|
| `experiments.py` | Archive reproduction script: statistic generators T1–T10, divergence measures, Study 1–3 runners, table generator. Produced the original submission's numbers. | `python3 code/experiments.py` | **196 s** (205 s on a second run) | `experiment_results.json`, `fig_study1_convergence.png`, `fig_study2_robustness.png`, `fig_study3_precision.png`, `generated_tables.md` |
| `core.py` | Shared kernel of the revision: registry `TESTS` (T1–T13), single evaluation path `_evaluate()` → $D_{KS}$, $\hat\alpha$, Wilson CI; generators for T11 KPSS, T12 Hoeffding $D$, T13 interaction; dependence generators; `_selftest()` guard rails; `make_meta`/`save_results`/`update_results`. Imports the T1–T10 generators from `experiments.py` unchanged. | `import core` (library, no CLI); `python3 -c "import core; core._selftest(quick=False)"` | import ≈ 1 s; `_selftest` 18 analytic checks in ≈ 1 s, 20 checks in ≈ 9 s with `quick=False` | — |
| `rev_study1_ext.py` | Study 1 grid for the full 13-test panel at three nominal levels, `n*` table, Monte-Carlo null calibration of T11–T13, the T12 reference-law diagnostic, the T13 split diagnostic, and `fig_rev_alpha.png`. | `python3 code/rev_study1_ext.py` (stages `validate,study1,calib,t12diag,t13diag,figure`); `--validate` runs only the verification suite; `--quick` smoke run | **2302 s** (≈38 min) end to end; the recorded study-1/calibration block reports `wall_clock_s = 1601.1`; `--validate` 2 s | `rev_results_t1.json`, `fig_rev_alpha.png` |
| `rev_alpha_dependence.py` | Study 2 scenario sweep at $\alpha \in \{0.01, 0.05, 0.10\}$ for all 13 tests, the tail-mass identity checks, and the dependence designs (AR(1) serial, clustered/ICC, beta-binomial blocks) with `fig_rev_dependence.png`. | `python3 code/rev_alpha_dependence.py` (stages `alpha,dependence,figure`) | **288 s** (≈4.8 min; `meta_alpha_dep.wall_clock_s`) | merges `alpha_sweep_13`, `alpha_sweep_13_crosscheck_archive`, `identity_checks`, `dependence_13` into `rev_results_t1.json`; `fig_rev_dependence.png` |
| `rev_sensitivity.py` | Severity ladders (skewness, kurtosis, variance ratio, imbalance, small expected frequencies) at $n=50$ and the Study 2 sample-size sweep $n \in \{20,50,100,200\}$, both with Wilson intervals and common random numbers. | `python3 code/rev_sensitivity.py` (takes **no** command-line options: any argument is ignored and the full run starts) | **76 s** (58–76 s across reruns; `meta.runtime_seconds`) | `rev_results.json`, `fig_rev_severity.png`, `fig_rev_study2_n.png` |
| `rev_power.py` | Study 4: nominal-$\alpha$ vs. size-adjusted power under location, scale and dependence alternatives, with equivalence and calibration self-checks. | `python3 code/rev_power.py` (options `--nmc`, `--no-fig`, `--skip-verification`) | **53 s** (`meta.runtime_seconds`) | `rev_results_power.json`, `fig_rev_power.png` |
| `verify_new_tests_power.py` | Independent power measurement for the two new continuous tests (T11 KPSS, T12 Hoeffding $D$): nominal-$\alpha$ and size-adjusted power at $n \in \{20,50,100,200\}$, $N=10{,}000$, same Wilson convention as Study 4. | `python3 code/verify_new_tests_power.py` | **37 s** (`meta.wall_clock_s`) | `rev_results_newtests_power.json` |
| `verify_manuscript.py` | Adversarial cross-check: every number asserted in `draft.md` is compared with the JSON path that should produce it, at the tolerance implied by the Monte Carlo budget. Read-only; exit code 1 on any mismatch. | `python3 code/verify_manuscript.py [-v]` | **0.3 s** | — (prints `passed: N failed: 0`) |
| `build.sh` | Renders `draft.pdf` and `response_letter.pdf` with pandoc + XeLaTeX (`./build.sh [all\|draft\|letter]`). | `./build.sh` | one pandoc run per document, seconds | `draft.pdf`, `response_letter.pdf` |
| `latex_preamble.tex` | XeLaTeX preamble included by `build.sh`: `unicode-math`, longtable/booktabs, float, and section-numbering suppression so the manuscript renders like a journal article. | included automatically | — | — |

Direct third-party imports: `numpy`, `scipy` and `matplotlib` for the study
scripts; `lightgbm` additionally for T13 inside `core.py`. Nothing else is
imported. The optional studies under `extensions/` are the one exception: they
import `mc-tk` (`mc.samplings.Box_M_Test`, `mc.samplings.Mardia_Test`) and are
outside the manuscript pipeline described in this file. Consistent with Section
4 of the manuscript, no script of the fourteen-test panel imports the `mc-tk`
package: the numbers come from these
standalone scripts, and `mc-tk` 1.1.0 is the separately versioned library.
`pandas`, `scikit-learn`, `statsmodels` and `shap` are installed in the same
Anaconda environment but are **not** used by this project.

## 2. Result files (data)

| File | Produced by | Top-level contents | Consumed by |
|---|---|---|---|
| `experiment_results.json` | `experiments.py` | `study1` (per test: `ks`, `kl`, `type1`, `ci_lo`, `ci_hi`, `ns`, `n_total`, `N`), `study1_nstar_alpha`, `study1_nstar_ks`, `study1_ks_threshold`, `study2` (per test × scenario rate), `study2_ci`, `study2_delta`, `study3` (`N_values`, `mean_t1e`, `std_t1e`, `cv_t1e`), `study3_nstar_mc` | Tables 4–7 and Figures 1–3 |
| `rev_results_t1.json` | `rev_study1_ext.py`, then `rev_alpha_dependence.py` (merge) | `meta` + `meta_alpha_dep`, `verification` (`selftest`, `cvm_limit_vs_goftest`, `cvm_limit_critical_values`, `hoeffding_port`), `selftest` (18 checks), `study1_ext` (13 tests × 8 sample sizes × 3 levels), `study1_ext_nstar`, `archive_reproduction_check`, `null_calibration` (T11/T12/T13 vs their reference laws), `t12_reference_diagnostic`, `t13_split_diagnostic`, `alpha_sweep_13` (+ `alpha_sweep_13_crosscheck_archive`), `identity_checks`, `dependence_13` (`ar1`, `clustered`, `assumption_table`), `figure_alpha`, `figure_dependence`, `notes` | Tables 4–6 and 12 (13-test panel), the $\alpha$-sweep of §5.4, the dependence section §5.5 and Figures 4/6 |
| `rev_results.json` | `rev_sensitivity.py` | `meta`, `severity_sweep` (`ladders`, `thresholds`, `not_applicable`, `extra_nsweep`), `study2_nsweep` (`n_values`, `scenarios`, `ordering_stability`, `degradation_recovery`), `alpha_sweep`, `consistency_checks`, `implementation_validation`, `distribution_selfcheck`, `determinism_checks` | Tables 8–11, the severity and $n$-sweep panels, and the mechanical robustness ratings |
| `rev_results_power.json` | `rev_power.py` | `meta`, `study4_power` (`design`, `tests`, `verification`, 368 `cells`, 256 `contrasts`, `headline_scale_table`, `rank_reversal`, `adjusted_size_attainment`, `counterexamples`, `notes`), `recommendation_matrix` | Table 12, Figure 8 and the calibration-vs-discrimination discussion |
| `rev_results_newtests_power.json` | `verify_new_tests_power.py` | `meta` (`seed`, `N`, `alpha`, `ns`, versions, `wall_clock_s`), `T11_kpss`, `T12_hoeffding_d` (per $n$: nominal and size-adjusted power with Wilson intervals) | The power row/paragraph for the new tests |
| `rev_results_t1_quick.json` | `rev_study1_ext.py --quick` | Reduced-budget smoke output | **Not part of the published results**; a development artefact, deliberately excluded from `docs/MANIFEST.sha256` |

## 3. Figures

| File | Produced by | Shows | Size |
|---|---|---|---|
| `fig_study1_convergence.png` | `experiments.py` | $D_{KS}$ and $\hat\alpha$ vs. $n$ for T1–T10, with the 0.0136 KS adequacy line | 221 KB |
| `fig_study2_robustness.png` | `experiments.py` | $\hat\alpha$ under baseline and five violations, $n=50$, $\alpha=0.05$ | 55 KB |
| `fig_study3_precision.png` | `experiments.py` | CV of $\hat\alpha$ vs. $N$ with the 0.10 target line | 152 KB |
| `fig_rev_alpha.png` | `rev_study1_ext.py` | $\hat\alpha$ vs. $n$ for all 13 tests at $\alpha \in \{0.01, 0.05, 0.10\}$ | 181 KB |
| `fig_rev_dependence.png` | `rev_alpha_dependence.py` | Four panels: $t$ test and sign test vs. $\rho$; ANOVA/Kruskal–Wallis and Cochran $Q$ vs. ICC | 236 KB |
| `fig_rev_severity.png` | `rev_sensitivity.py` | $\hat\alpha$ with Wilson intervals along each severity ladder | 418 KB |
| `fig_rev_study2_n.png` | `rev_sensitivity.py` | $\hat\alpha$ vs. $n$ for every Study 2 scenario | 369 KB |
| `fig_rev_power.png` | `rev_power.py` | Nominal vs. size-adjusted power | 343 KB |
| `docs/fig_execution_flow.png` / `.pdf` | `docs/make_flowchart.py` | The execution flowchart reproduced in the vignette | 373 KB / 64 KB |

## 4. Supporting files

| File | Purpose |
|---|---|
| `requirements.txt` | Direct dependencies with lower bounds (`>=`) and the reason each is needed |
| `requirements-lock.txt` | Exact pins (`name==version`) of the runtime closure, read from the live environment |
| `docs/pip_freeze_full.txt` | Complete `pip freeze` of the producing environment (local-path entries removed and documented) |
| `docs/ENVIRONMENT.txt` | OS, CPU, Python, pip and named-package versions |
| `docs/CODE_MAP.md` | This file |
| `docs/FUNCTION_REFERENCE.md` | Generated per-function reference (signature, line, docstring summary) |
| `docs/REPO_LAYOUT.md` | Repository layout, licensing and Data Availability wording (R2-M3) |
| `docs/verify_bartlett.py` | Worked example: reproduces the published Bartlett rate from the pipeline end to end (13 checks) |
| `docs/probe_experiments.py` | Determinism probe for `experiments.py` (SHA-256 over ten generated statistic arrays) |
| `docs/probe_docstring_pass.py` | Behavioural fingerprint of all eight pipeline scripts, used to prove a documentation-only edit changed nothing |
| `docs/run_verification.sh` | Runs every command quoted in `vignette.md` and writes `docs/verification_log.txt` |
| `docs/verification_log.txt` | Output of the above: each command, its output and its exit status |
| `docs/flowchart.mmd` | Mermaid source of the execution flowchart |
| `docs/make_lock.py` | Regenerates the lock files from the live environment |
| `docs/make_manifest.py` | Writes/verifies `docs/MANIFEST.sha256`; `--lock` checks the environment against the lock file |
| `docs/make_flowchart.py` | Renders the execution flowchart |
| `docs/make_function_reference.py` | Generates `docs/FUNCTION_REFERENCE.md` |
| `docs/MANIFEST.sha256` | SHA-256 of every pipeline script, results file, figure, build file and `docs/` tool (38 files) |
| `docs/fig_execution_flow.png`, `docs/fig_execution_flow.pdf` | Rendered flowchart panel |

---

## 5. File-by-file notes

### `experiments.py` — archive reproduction script (~200 s)

* **Sections.** 1 test-statistic generators T1–T10; 2 divergence measures
  (`compute_ks`, `compute_kl`, `compute_type1_error`, `wilson_ci`); 3 study
  runners (`run_study1`, `run_study2`, `run_study3`) and `generate_tables`;
  4 the `__main__` block that sequences them and then re-runs three key Study 1
  configurations under five seeds (2024–2028) as a stability check.
* **Inputs.** None: every parameter is set in the script (sample-size grid
  `[5, 10, 20, 30, 50, 100, 200, 500]`, $N=10{,}000$, six Study 2 scenarios,
  Study 3 grid up to $N=50{,}000$). Output directory = the script's own directory.
* **Outputs.** The five artefacts listed in §1. All outputs are written next to
  the script, so run it in a copy of the directory if you do not want to
  overwrite the committed results.
* **RNG convention.** `np.random.seed(2024)` is executed once at import and every
  generator draws from that single global stream. A stored number is reproducible
  only by running the whole script in the original order; calling one generator on
  its own with `np.random.seed(2024)` produces a *different* (equally valid)
  realisation. See §6 below and §8.1 of `vignette.md`.
* **Verification anchors.** `study2.T5_bartlett.skewed = 0.4288`,
  `study2.T9_median_test.small_freq = 0.0691`, `study1_ks_threshold = 0.01358`,
  console ends with `Total time: ...s`, and `docs/probe_experiments.py` prints the
  combined digest `3824e5fe16564e764fec478cac090ffa7d6cd483e555c20c533d813a1ce5a7aa`.
* **Reproduced in the revision.** A full re-run in a scratch directory takes 196 s
  and reproduces every integer rejection rate exactly; 29 of the Study 1 KS/KL
  distances differ in the 14th significant digit (max relative difference
  $2.9\times10^{-12}$) because of floating-point reduction order. Two reruns on
  this machine were byte-identical to each other (SHA-256 `7fb7080f…`); the
  committed file from the original submission is `6d533636…`.
* **Docstring pass (revision).** File SHA-256 went from
  `6cd9a353fb90301d176e0e8031ca52e567866923af3305614728e2293758aa8c` to
  `41782dd05bf280fd7663c9a4683ee665c65c73d154451c8eb41821ead363923e`; the
  code-only AST digest is unchanged
  (`42c614978d1dc6e2dc4a6e2b0894eb696a028878617fcd7bbc5c507428794872`), and the RNG
  probe digest is unchanged, so no drawn number changed.

**Note on the `script_sha256` fields.** Results files record the hash of the script
that wrote them *at the moment of the run* (`meta.experiments_py_sha256` in
`rev_results.json`, `meta.script_sha256` in `rev_results.json` and
`rev_results_t1.json`). Those hashes therefore refer to the pre-docstring versions
of `experiments.py` and `rev_sensitivity.py`. The code is identical — only
docstring constants were added, as the AST digests above show — and every value in
those files is reproduced exactly by the current scripts, which
`docs/verify_bartlett.py` and `verify_manuscript.py` check on every run.

### `core.py` — shared kernel

* **Registry.** `TESTS[test_id]` holds the label, family, theoretical null, the
  rejection rule (`two_sided` / `upper_tail` / `binom`) and whether the reference
  law is sample-size independent. `SEVERITY_HOOKS` lets the severity ladders of
  `rev_sensitivity.py` drive T11–T13.
* **Single evaluation path.** `_evaluate(stat_values, test_id, alpha)` returns
  `{ks, type1, ci_lo, ci_hi, n_effective, critical_value, alpha}`; every study uses
  it, so $D_{KS}$ and $\hat\alpha$ have one definition across the paper.
* **`_selftest(quick=True)`.** Guard rails for the bespoke reference laws, run at
  the start of every study script: the Cramér–von Mises limit quantiles against the
  published KPSS critical values, the limit CDF against an independent series
  expansion (`goftest`), monotonicity of the Hoeffding survival function, and (with
  `quick=False`) a Monte-Carlo check of the T11/T12 rejection rates. It raises on
  failure, so a convention slip cannot silently invalidate a results file.
* **Results plumbing.** `make_meta()` writes the `{seed, numpy, scipy, python,
  git_hash, timestamp, …}` block; `save_results()` writes a file; `update_results()`
  merges blocks into an existing file — this is what lets
  `rev_alpha_dependence.py` add its sections to the file written by
  `rev_study1_ext.py` without discarding anything.
* **Conventions.** `SEED = 2024`, `N_MC = 10_000`, `ALPHAS = (0.01, 0.05, 0.10)`,
  `KS_THRESHOLD = 1.358/sqrt(N) = 0.01358`, $n^*_\alpha$ band
  $|\hat\alpha-\alpha| < 0.01$. $D_{KS}$ is computed with the same `max(D+, D-)`
  convention as `scipy.stats.kstest`, which `_selftest` asserts to $10^{-12}$.
* **Verification anchors.** `core.wilson_ci(0.05, 10000) = (0.0458984827, 0.0544471158)`;
  `core.KS_THRESHOLD == 1.358/sqrt(10000)`; `rev_results_t1.json → selftest` reports
  18/18 `ok: true`.

### `rev_study1_ext.py` — 13-test Study 1, multi-$\alpha$, null calibration (2302 s)

* **Blocks written.** `verification` (+ `selftest`), `study1_ext`,
  `study1_ext_nstar`, `archive_reproduction_check`, `null_calibration`,
  `t12_reference_diagnostic`, `t13_split_diagnostic`, `figure_alpha`.
* **Identical RNG stream for T1–T10.** The ten archived tests are drawn with the
  archived generators from the archived seed, so the $\alpha=0.05$ column is
  bit-identical to `experiment_results.json`; the script asserts this in
  `archive_reproduction_check` rather than assuming it.
* **Null calibration.** For T11, T12 and T13 the block re-derives the rejection
  rate at all three nominal levels against the analytic/limiting reference
  (KS distance, $\hat\alpha$, moment checks), which is the evidence that each new
  null distribution is the right one.
* **T12 diagnostic.** Shows that the finite-sample law of the scaled statistic
  matches an independently estimated limit law to Monte-Carlo resolution for
  $n \ge 50$, while the published coarse tail table differs by an order of
  magnitude more — i.e. T12's large $D_{KS}$ is a property of the published table,
  not of a non-converging statistic.
* **T13 diagnostic.** Separates learner overfitting from the partial-dependence
  approximation by comparing the grid estimator with (`mc_vs_exact_at_largest_n =
  0.0725`) and without the deterministic grid (`grid_inflation_at_largest_n =
  0.187`); the drift survives both corrections, so T13's liberality is attributed
  to the fixed flexible learner.
* **Modes.** `--validate` (2 s) runs the verification suite and writes nothing;
  `--quick` writes `rev_results_t1_quick.json`, which is a smoke test only.
* **Verification anchors.** `meta.core_selftest = "passed"`; `selftest` 18/18
  `ok`; `verification.hoeffding_port.max_abs_diff = 0.0`, i.e. the vectorised
  Hoeffding implementation reproduces a literal transcription of the Hmisc ratfor
  algorithm exactly, with and without ties, over the tabulated sample sizes.

### `rev_alpha_dependence.py` — multi-$\alpha$ sweep and dependence (288 s)

* **Blocks written** (merged into `rev_results_t1.json`): `meta_alpha_dep`,
  `alpha_sweep_13`, `alpha_sweep_13_crosscheck_archive`, `identity_checks`,
  `dependence_13`, `figure_dependence`.
* **$\alpha$ sweep.** Every Study 2 scenario of the 13-test panel at
  $\alpha \in \{0.01, 0.05, 0.10\}$, with the new tests extended by
  scenario-appropriate designs (T11: stationary AR(1), trend-stationary and
  unit-root series; T12: linear/quadratic dependence and skewed independent
  marginals; T13: true interaction and correlated covariates). The same nominal
  level is used for the critical value and the reference CDF in every cell.
* **Cross-check.** `crosscheck_archive` compares the $\alpha=0.05$ column with
  `experiment_results.json`; because the RNG stream differs, agreement is expected
  only within Monte Carlo noise, and the comparison is reported rather than
  asserted.
* **Identity checks.** Numerical verification of $\alpha = 2(1-\Phi(z))$ and of the
  equal-tail identity used for the two-sided tests, with the maximum absolute
  residual recorded in `identity_checks.max_abs_residual`.
* **Dependence.** `ar1` ($\rho \in \{0, 0.2, 0.5, 0.8\}$, $n \in \{20,50,100\}$ for
  the $t$ test and the sign test), `clustered` (ICC $\in \{0,0.2,0.5\}$ for ANOVA
  and Kruskal–Wallis; beta-binomial block correlation for Cochran's $Q$), and an
  explicit per-test `assumption_table` stating what i.i.d. means for each of the
  thirteen tests.
* **Modes.** `--no-merge` writes `rev_results_t1_alpha_dep.json` instead of
  merging; `--quick` appends `_quick` to the output name.

### `rev_sensitivity.py` — severity ladders and sample-size sweep (~76 s)

* **Design.** 156 ladder cells plus 144 $n$-sweep cells, all at $N=10{,}000$ and
  all three $\alpha$ levels; common random numbers (one seed per ladder and per
  scenario) so that differences between rungs are not confounded with sampling
  noise; a 333-entry draw cache makes repeated cells exact.
* **Outputs.** `rev_results.json` plus the two severity figures.
* **Self-checks stored in the file.** `consistency_checks` re-runs the archived
  anchor cells with `experiments.py` itself and compares (37 cells),
  `implementation_validation` compares each statistic against an independent
  implementation (`scipy.stats.f_oneway`, `bartlett`, `kruskal`, `fligner`,
  `statsmodels.stats.contingency_tables.cochran_q`, `chi2_contingency`, …),
  `distribution_selfcheck` verifies the analytic moments of every parent law, and
  `determinism_checks` confirms that re-running a cell yields an identical array.
* **Verification anchor.** `study2_nsweep.scenarios.skewed.tests.T5_bartlett.by_n["50"].alpha_hat = 0.4267`
  with seed 19204 — reproduced bit-for-bit by `docs/verify_bartlett.py`.

### `rev_power.py` — Study 4 power companion (53 s)

* **Design.** 368 cells and 256 contrasts; $n \in \{20,50,100\}$ (plus 200 for the
  scale family), $d \in \{0.2,0.5,0.8\}$, SD ratios $\{1.25,1.5,2,4\}$, four parent
  laws (normal, $t(5)$, Exp(1)−1, contaminated), common random numbers across
  competitors, and the same evaluation code as the null studies.
* **Outputs.** `rev_results_power.json` and `fig_rev_power.png`.
* **Self-checks.** `study4_power.verification.equivalence` shows that the power
  path and the null path agree to machine precision on the same data (max
  absolute difference $\le 2\times10^{-13}$ for the continuous tests), and
  `study4_power.verification.calibration` re-runs the archived Study 2 anchor
  rates and compares them with a two-sample $z$ statistic (21/21 consistent at
  $3\sigma$; Bartlett skewed: 0.4329 vs. archive 0.4288, $z=0.585$).

### `verify_new_tests_power.py` — independent power for T11/T12 (37 s)

* Supplies the power counterpart for the two new continuous tests with the same
  nominal level, Monte Carlo budget and Wilson convention as Study 4, reporting
  nominal-$\alpha$ and size-adjusted power side by side for
  $n \in \{20,50,100,200\}$.
* Deterministic: `np.random.default_rng(2024)`. The recorded run took 37.3 s
  (`meta.wall_clock_s`).

### `verify_manuscript.py` — adversarial cross-check of `draft.md` (0.3 s)

* Every number asserted in the manuscript is compared with the JSON path that
  should produce it, at the tolerance implied by the Monte Carlo budget; text
  claims are checked by substring. Ten groups of checks cover the archived
  Study 2/1/3 cells, the severity ladders, the $n$-sweep, the $\alpha$ sweep, the
  power study, source hygiene (table captions and references), the new-test panel,
  the new-test power and free-text claims.
* Read-only, no dependencies beyond the standard library, exit code 1 on any
  mismatch. The recorded run reports
  `passed: 213   failed: 0   skipped: 0` in 0.3 s.

### `build.sh`, `latex_preamble.tex` — manuscript build

* `./build.sh` renders `draft.pdf` and `response_letter.pdf` with
  `pandoc --pdf-engine=xelatex --include-in-header=latex_preamble.tex`;
  `./build.sh draft|letter` builds one of them. The preamble adds `unicode-math`,
  longtable/booktabs, float, and suppresses section numbering so the output reads
  like a journal manuscript.

### `docs/verify_bartlett.py` — worked example (1 s)

* Re-runs Bartlett's skewed scenario through `rev_sensitivity.py` with the
  committed seed and compares the printed rate and Wilson interval with
  `rev_results.json` (exact float equality), then re-derives the same scenario
  through `core.py` with a fresh seed, collects every independent realisation
  stored in the results files, and checks the two closed-form constants.
  13/13 checks pass; exit status 0.
* **Why it matters.** It is the one-command answer to "is this package alive on my
  machine?" — the runtime is about one second and every check prints PASS/FAIL.

### `docs/probe_docstring_pass.py` — behavioural fingerprint (2 s)

Imports all eight pipeline scripts, exercises a fixed set of functions with fixed
seeds (seeded cells, statistic constructors, rejection rules, the three `core.py`
severity hooks, the Wilson helpers, and the two read-only verification scripts as
subprocesses), and prints one SHA-256 over the results. Used to prove that the
docstring pass over the frozen files changed no computed value: the digest is
`31e1760…` before and after. Nothing is written to disk, so it is safe to run on a
frozen tree.

### `docs/probe_experiments.py` — determinism probe (2 s)

Replays a fixed prefix of the archive RNG stream through all ten generators and
prints a combined SHA-256. Used to prove that the docstring pass over
`experiments.py` changed no drawn number: the digest is identical before and
after (`3824e5fe…`), and the code-only AST digest is
`42c614978d1dc6e2dc4a6e2b0894eb696a028878617fcd7bbc5c507428794872` in both
versions.

---

## 6. RNG and determinism rules

1. **Archive script** (`experiments.py`): one global `np.random` stream, seeded
   once at import with 2024. Reproducible by re-running the whole script in order;
   **not** reproducible by calling a single generator, because the stream position
   depends on everything drawn before it in the script.
2. **Revision scripts** (`core.py`, `rev_*.py`, `verify_new_tests_power.py`): every
   cell takes an explicit seed (`cell_seed(...)` in `rev_sensitivity.py`,
   `np.random.default_rng(seed)` in `core.py` and in the new-test scripts), so any
   single configuration can be reproduced on its own. Two runs of the same cell
   return bit-identical arrays (`determinism_checks` in `rev_results.json`).
3. **Two realisations of the same configuration differ by Monte Carlo noise** of
   order $\sqrt{\alpha(1-\alpha)/N} \approx 0.0022$ at $\alpha=0.05$,
   $N=10{,}000$. Example: Bartlett under skewness is 0.4288 in the archive run,
   0.4267 in the $n$-sweep and 0.4329 in the power study — all three are correct,
   and the differences are within $1\sigma$ ($z = 0.30$ and $z = 0.59$).
4. **Reported value rule.** A reported value is reproducible bit-for-bit by
   re-running the script that produced it with the stated seed; standalone calls
   to individual generators are reproducible only under the RNG-state conventions
   above. `verify_manuscript.py` enforces the rule for the manuscript text.

## Revision studies added after the first integration pass

| file | purpose | command | runtime | outputs |
|---|---|---|---|---|
| `code/rev_dcor.py` | T14, the Szekely-Rizzo distance-correlation t-test: Study 1 convergence at $d \in \{2, 10, 50\}$, the within-vector-correlation nuisance check, and power against linear and nonlinear dependence | `python3 code/rev_dcor.py` | 1103 s | `rev_results_dcor.json` |
| `code/verify_dcor_against_R.py` | Verified cross-check of the T14 statistic against `energy::dcorT` (R package `energy` 1.7-12); calls R on identical data and reports the agreement (4.5e-13) | `python3 code/verify_dcor_against_R.py` | 2 s | stdout |
| `code/rev_imbal_hetero.py` | Crossed allocation x variance design at fixed $n_T = 150$ (Table 10 of the manuscript), the interaction the separate ladders cannot isolate | `python3 code/rev_imbal_hetero.py` | 5 s | `rev_results_imbal_hetero.json` |
| `code/make_fig2.py` | Regenerates Figure 2 with the legend outside the plotting area and Wilson intervals on every bar (Referee 2, m13) | `python3 code/make_fig2.py` | 1 s | `figures/fig_study2_robustness.png` |

| `code/verify_letter_claims.py` | Adversarial check that every artefact the response letter claims to have produced exists in the working tree (the guard against claiming a fix not on disk) | `python3 code/verify_letter_claims.py` | 0.2 s | stdout |

| `code/gen_dcor_local.py` | Generates `rev_results_dcor_local.json`: local power of T14 against a quadratic signal confined to `k_signal` of `d` coordinates, amplitude `a = 1` (signal variance 2 against unit noise). Reported in §5.1.1b and Supplementary Material §S10 | `python3 code/gen_dcor_local.py` | ~215 s | `rev_results_dcor_local.json` |
| `code/gen_t14_study3.py` | Generates `rev_results_t14_study3.json`: the Study 3 Monte Carlo precision sweep extended to T14 at `d = 50`, 30 independent runs, CV of the rejection rate against the replication grid. Reported in §5.3 | `python3 code/gen_t14_study3.py` | ~215 s | `rev_results_t14_study3.json` |

| `code/gen_t12_law.py` | Generates `rev_results_t12_law.json` (one independent realisation of the scaled-Hoeffding limit law, `n_law = 2,000`, `N = 200,000`, seed 20241231; ~110 min) and `rev_results_t12_diag.json` (KS to that reference and the published-table rejection rates at each sample size). Reported in §2.5, §5.1.1 and Table 5 | `python3 code/gen_t12_law.py --diag-only` reuses the committed reference and rebuilds only the diagnostic | ~110 min full / ~4 min diag | `rev_results_t12_law.json`, `rev_results_t12_diag.json` |
| `code/rev_imbal_hetero.py` | Generates `rev_results_imbal_hetero.json`: the crossed allocation x variance design of Table 8 | `python3 code/rev_imbal_hetero.py` | ~3 min | `rev_results_imbal_hetero.json` |
