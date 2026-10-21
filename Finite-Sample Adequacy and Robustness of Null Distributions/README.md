# Finite-Sample Adequacy and Robustness of Null Distributions

Reproduction package for the manuscript *Finite-Sample Adequacy and Robustness of Null
Distributions: A Systematic Monte Carlo Study Across Fourteen Hypothesis Tests*
(COST-D-26-00503R1, *Computational Statistics*, major revision).

The package accompanies the `mc-tk` toolkit in this repository (see the
[repository README](../../README.md)). The toolkit provides the Monte Carlo primitives;
this directory contains the study itself, its results, and the checks that tie the two
together.

## What this package can and cannot do

Everything below runs offline from a clone. The distinction between the two halves of the
table is the point of this section: most of the study re-runs in seconds, but two
analyses are long enough that their results are **committed rather than regenerated**, and
saying so is more useful than implying that one command reproduces every number.

| Analysis | How it is obtained | Cost to regenerate |
|---|---|---|
| Studies 1–3 of the original submission | `python3 experiments.py` (frozen original-submission script) | ~3 min |
| Study 1 grid, multi-level grid, T11/T12 null calibration | `python3 code/rev_study1_ext.py` | ~25 min |
| Study 2 severity ladders and sample-size sweep | `python3 code/rev_sensitivity.py` | ~20 min |
| Crossed allocation × variance design | `python3 code/rev_imbal_hetero.py` | ~3 min |
| Dependence designs (serial and clustered) | `python3 code/rev_alpha_dependence.py` | ~20 min |
| Study 4 power companion | `python3 code/rev_power.py` | ~30 min |
| Distance-correlation test (T14), convergence and nuisance | `python3 code/rev_dcor.py` | ~10 min |
| T14 local power against a concentrated signal | `python3 code/gen_dcor_local.py` | ~4 min |
| T14 Study 3 precision curve | `python3 code/gen_t14_study3.py` | ~4 min |
| KPSS / T13 generator correction, re-measured | `python3 code/rev_alpha_dependence.py` and the `gen_*` scripts | ~5 min |
| **T12 reference law** | **committed** (`results/rev_results_t12_law.json`) | **~110 min** (`python3 code/gen_t12_law.py`) |

The T12 reference is a Monte Carlo realisation at `n_law = 2,000` with `N = 200,000`
draws. Its own sampling error is reported in the manuscript alongside every distance
computed against it, which is why committing it is legitimate rather than a shortcut: the
error is disclosed, not hidden. `python3 code/gen_t12_law.py --diag-only` rebuilds the
diagnostic table from the committed reference in about four minutes.

## Quick start

```bash
python3 -m pip install -r docs/requirements-lock.txt   # or the repo-level environment
python3 docs/run_verification.sh                       # ~2 min, read-only, prints a summary
```

`docs/run_verification.sh` runs twelve read-only checks and writes
`docs/verification_log.txt`. It does not modify the package, so the log is a record of the
committed state rather than of a rebuild. To regenerate the derived artefacts (the pinned
lock, the manifest, the flowchart, the function reference), run `docs/regenerate.sh`
first.

## Layout

```
experiments.py            frozen original-submission script (byte-identical to the version
                          published with the first submission; do not edit)
experiment_results.json   frozen original-submission results
fig_study1/2/3_*.png      frozen original-submission figures
generated_tables.md       frozen original-submission tables
supplementary_kl.md       supplementary material, Sections S1-S10
vignette.md / .pdf        reproduction vignette: environment, flowchart, per-file guide,
                          step-by-step commands with runtimes, a worked example that
                          reproduces one published number, and how to verify a run
code/                     revision-grounded analysis scripts
  core.py                 shared kernel: registry of the 14 tests, the evaluation path,
                          the T12 and T14 reference machinery, and a 20-item self-test
  archive_experiments.py  the frozen archive script as imported by core.py (the file at
                          the package root has the same content under the canonical name)
  rev_*.py                the revision studies
  gen_*.py                generators for the committed long-run results
  verify_*.py             the automated checks
results/                  every results file, one per script, with its seed and runtime
figures/                  published panels
docs/                     CODE_MAP.md, DESIGN_CONTRACT.md, FUNCTION_REFERENCE.md,
                          ENVIRONMENT.txt, MANIFEST.sha256, the verification tooling,
                          the execution flowchart
extensions/               optional later studies (Box's M, Mardia); not part of the
                          fourteen-test manuscript and not covered by the manifest
  box_m/                  Box's M null-distribution study
  mardia/                 Mardia skewness/kurtosis null-distribution study
```

## Optional extension studies

`extensions/box_m/` and `extensions/mardia/` are two later Monte Carlo studies of the
null distributions of Box's M test and Mardia's multivariate skewness/kurtosis tests.
They are **not part of the fourteen-test manuscript**: they are not referenced by
`code/verify_manuscript.py`, are excluded from `docs/MANIFEST.sha256`, and are not
needed to reproduce any published number. They live here so that the toolkit repository
has a single home for its null-distribution studies.

Unlike the manuscript pipeline, they import the `mc-tk` classes
(`mc.samplings.Box_M_Test` and `mc.samplings.Mardia_Test`), so run them from a clone of
the repository with the `src/` tree present (they require the current `mc-tk` 1.2.0+,
which is where these classes were added; the frozen fourteen-test pipeline continues to
use the historical 1.1.0 environment recorded in `docs/ENVIRONMENT.txt`):

| Study | Command (from this package root) | Runtime |
|---|---|---|
| Box's M | `python3 extensions/box_m/study_box_m_null.py` | ~1 min |
| Mardia | `python3 extensions/mardia/study_mardia_null.py` | ~2.5 min |

Each directory has its own `README.md`, `results/` and `figures/`.  A `--quick` smoke
run writes `*_quick.json` and does not overwrite the committed full-run results.

## Verification

Three kinds of check are shipped, and they fail for different reasons:

1. **Kernel self-test** — `core._selftest()`, 20 checks covering the D_KS convention
   (asserted against `scipy.stats.kstest` to 1e-12), the Wilson interval, the reference
   CDFs, and the fact that the T12 decision reference and its full-distribution reference
   are distinct objects.
2. **Results-file consistency** — `code/verify_manuscript.py` checks every quoted number
   against the results file that produced it, and `docs/make_manifest.py --check` verifies
   the SHA-256 manifest.
3. **Independent implementations** — `docs/verify_bartlett.py` reproduces one published
   number from scratch through a different code path (13/13 checks), and
   `code/verify_dcor_against_R.py` compares the T14 statistic against `energy::dcorT` in
   R, agreeing to `4.5e-13` on identical data.

The manuscript, the response letter and the cover letter are **submission material and are
not distributed here**. Two checks (`code/verify_crossrefs.py`,
`code/verify_letter_claims.py`) audit the text of those documents; when they are absent
the checks report that they are skipped and exit successfully rather than failing.

## Environment

`docs/ENVIRONMENT.txt` records the hardware and threading configuration;
`docs/requirements-lock.txt` pins the Python dependencies; `docs/pip_freeze_full.txt` is
the full freeze. The T14 cross-check additionally needs R with the `energy` package
(version 1.7-12).

## Licence

Apache 2.0, as the rest of this repository.
