# Repository layout — a dedicated home for the paper's code (R2-M3)

Referee 2, major comment 3:

> The supplied hyperlink to GitHub covers also other work unrelated to this project
> (for example Galton board experiment or clip experiment to produce a Zipf
> distribution). It would be clearer to have a separate web page that only
> contains the code used in this study.

This file records the fix that was adopted — a **dedicated project folder inside
the mc-tk repository that contains only this study's material**, and nothing else —
and the exact wording for the Data Availability statement and the response letter.
A separate repository with a Zenodo DOI remains the recommended upgrade if the
authors want a citable archive independent of the toolkit repository.

---

## 1. Why the current link is ambiguous

`https://github.com/zhangys11/mc` is the home of the **mc-tk toolkit**
(`pip install mc-tk`), a general-purpose Monte Carlo package that also hosts
unrelated experiments (Galton board, Zipf/clip experiment, …). The manuscript
currently uses that URL for two different purposes at once:

* as the *software* reference for mc-tk (Section 4), and
* as the *data/code availability* location for the simulation study.

The referee is right that the second use is not acceptable: a reader following the
link cannot tell which files produced the paper.

## 2. Adopted solution: a dedicated project folder

The study now lives in its own folder of the mc-tk repository, containing only this
paper's material:

```
https://github.com/zhangys11/mc/tree/main/Finite-Sample%20Adequacy%20and%20Robustness%20of%20Null%20Distributions
```

The toolkit repository is cited only as the *software* reference for Section 4 of
the manuscript (`https://github.com/zhangys11/mc`), and the generic repository root
is no longer used as the data-availability location. The same folder is shipped as
Supplementary Material, so the materials survive independently of the repository.

Two properties make this satisfy the referee:

* the folder contains **no file from another project** — the Galton-board and
  Zipf/clip experiments remain in their own directories of the repository; and
* the link is self-contained and directly usable: someone following it sees the
  scripts, the results, this vignette, the code map and the pinned environment, and
  nothing else.

An `extensions/` subtree now contains two later, optional studies (Box's M and
Mardia's multivariate normality tests).  They are **not** part of the manuscript, are
excluded from `docs/MANIFEST.sha256` and from every verification script, and are kept
here only so the toolkit repository has one home for its null-distribution studies.
A reader interested strictly in the paper can ignore `extensions/`; the manuscript-facing
material described in this file is unchanged.

Recommended upgrade when time permits: create a standalone repository
`https://github.com/zhangys11/mc-tk-null-distributions`, push the same folder, tag
it (`paper-comstat-v1.0`) and archive the tag on Zenodo for a DOI. That makes the
"only this study" property structural and gives the paper a permanent, citable
data-availability URI; the checklist in §7 lists the steps.

## 3. Proposed file layout

```
mc-tk-null-distributions/
├── README.md                     # what this is, how to reproduce in 5 commands
├── LICENSE                       # MIT for code (see §4)
├── LICENSE-data                  # CC BY 4.0 for results/tables/figures
├── CITATION.cff                  # citation metadata (paper + version DOI)
├── requirements.txt              # direct dependencies, >= pins
├── requirements-lock.txt         # exact pins used for the published results
│
├── vignette.pdf                  # the referee-requested vignette (this package)
├── vignette.md                   # its Markdown source
│
├── experiments.py                # archive script: T1–T10, Study 1–3  (R1 baseline)
├── core.py                       # shared 13-test kernel (registry, _evaluate)
├── rev_study1_ext.py             # Study 1 with 13 tests + null calibration
├── rev_alpha_dependence.py       # multi-alpha + dependence designs
├── rev_sensitivity.py            # severity ladders + Study 2 n-sweep
├── rev_power.py                  # Study 4 power companion
│
├── results/
│   ├── experiment_results.json   # archive Study 1–3 numbers
│   ├── rev_results.json          # sensitivity ladders + n-sweep
│   ├── rev_results_t1.json       # 13-test Study 1, alpha sweep, dependence
│   └── rev_results_power.json    # Study 4
├── figures/
│   ├── fig_study1_convergence.png
│   ├── fig_study2_robustness.png
│   ├── fig_study3_precision.png
│   ├── fig_rev_severity.png
│   ├── fig_rev_study2_n.png
│   ├── fig_rev_alpha.png
│   ├── fig_rev_dependence.png
│   ├── fig_rev_power.png
│   └── fig_execution_flow.png    # flowchart panel of the vignette
│
└── docs/
    ├── CODE_MAP.md               # file-by-file purpose, I/O, runtime, dependencies
    ├── FUNCTION_REFERENCE.md     # generated per-function reference
    ├── REPO_LAYOUT.md            # this file
    ├── ENVIRONMENT.txt           # OS/CPU/Python/package versions
    ├── pip_freeze_full.txt       # complete freeze of the producing environment
    ├── MANIFEST.sha256           # checksums of every script, result and figure
    ├── verify_bartlett.py        # worked example (one published number)
    ├── probe_experiments.py      # determinism probe for experiments.py
    ├── make_lock.py              # regenerate the lock files
    ├── make_manifest.py          # write/verify the checksum manifest
    ├── make_flowchart.py         # render the execution flowchart
    ├── make_function_reference.py
    └── fig_execution_flow.pdf
```

Notes.

* In the flat layout used in the working directory, the results and figures sit
  next to the scripts (the archive script writes next to itself). If the
  `results/` and `figures/` split above is adopted, adjust `OUT_DIR`/`FIG_DIR` in
  `experiments.py` and the output paths of the `rev_*.py` scripts, or simply keep
  the flat layout — it is what the published results correspond to.
* `README.md` should open with the five-command quick start from the vignette
  (§2 of `vignette.md`) and the one-second check
  `python3 docs/verify_bartlett.py`.
* No file from another project may live in this repository. `mc-tk` is referenced
  as software (with its own version and DOI), not vendored.

## 4. Licensing

| Content | Recommended licence | Rationale |
|---|---|---|
| Python code (`*.py`) | MIT | Permissive, matches the mc-tk toolkit licence; maximises re-use, which is exactly what the referee asks for |
| Results (`*.json`) | CC BY 4.0 | Attribution-only; allows re-analysis and redistribution |
| Figures and tables (`*.png`, `generated_tables.md`) | CC BY 4.0 | Standard for scholarly figures |
| Manuscript text (`vignette.md`, `vignette.pdf`) | CC BY 4.0 | Consistent with the journal's open-access option; copyright remains with the author |

Third-party components are **not** redistributed: `numpy`, `scipy`, `matplotlib`
(BSD-3-Clause) and `lightgbm` (MIT) are declared in `requirements.txt` and
installed from PyPI. The vignette states this explicitly so that the licence
boundary is unambiguous.

## 5. Data Availability statement — agreed wording

The revised manuscript points at the dedicated project folder (Section 4 of the
manuscript uses the same URL):

> **Data Availability Statement.** All code, simulation output and figure sources
> needed to reproduce every number, table and figure in this article are openly
> available in the project folder of the mc-tk repository that contains this study
> and nothing else:
> `https://github.com/zhangys11/mc/tree/main/Finite-Sample%20Adequacy%20and%20Robustness%20of%20Null%20Distributions`,
> and as Supplementary Material accompanying this article. The folder contains the
> simulation scripts, the committed results files in JSON format, the figure
> sources, a PDF vignette describing the pipeline, a file-by-file code map, exact
> dependency versions (`requirements-lock.txt`), a checksum manifest and a
> one-command verification tool.

**Open item for the authors:** the Data Availability paragraph at the end of
`draft.md` (currently "*… openly available in our mc-tk GitHub repository at
https://github.com/zhangys11/mc …*") still gives the **repository root**, which is
precisely the link the referee objected to. It should be replaced by the statement
above, or at least point at the project folder. Section 4 of the manuscript already
uses the folder URL, so the two statements are currently inconsistent.

If a Zenodo DOI is obtained, add it in parentheses after the URL and switch the
first option below to the standalone-repository form:

> … openly available at `https://github.com/zhangys11/mc-tk-null-distributions`
> (tag `paper-comstat-v1.0`; archived at `https://doi.org/10.5281/zenodo.XXXXXXX`).

The link `https://github.com/zhangys11/mc` should survive only where the toolkit
itself is meant (Section 4's software reference).

## 6. Text for the point-by-point response (R2-M3)

> **Comment.** *The supplied hyperlink to GitHub covers also other work unrelated
> to this project … It would be clearer to have a separate web page that only
> contains the code used in this study.*
>
> **Response.** We agree. The previous link pointed to our general-purpose mc-tk
> toolkit repository, which also hosts unrelated experiments. The code, results
> and documentation for this study are now in a dedicated project folder that
> contains nothing else
> (`https://github.com/zhangys11/mc/tree/main/Finite-Sample%20Adequacy%20and%20Robustness%20of%20Null%20Distributions`),
> and the same folder is supplied as Supplementary Material. The toolkit repository
> is now cited only as the software reference for Section 4. The new repository contains a PDF
> vignette (`vignette.pdf`), exact dependency versions (`requirements.txt`,
> `requirements-lock.txt`, plus the full `pip freeze` of the producing machine), a
> file-by-file code map (`docs/CODE_MAP.md`) and a generated function reference
> (`docs/FUNCTION_REFERENCE.md`), an execution flowchart (Figure R3 of the
> vignette), step-by-step instructions for readers who do not use Python
> (Section 7 of the vignette), a SHA-256 manifest of every artefact
> (`docs/MANIFEST.sha256`) and a one-command worked example
> (`python3 docs/verify_bartlett.py`) that reproduces a published number from the
> committed results file in about one second.

## 7. Repository hygiene checklist

- [ ] Repository contains only this study's files (no Galton board, no Zipf clip).
- [ ] `README.md` states the paper title, the five-command quick start and the
      verification command.
- [ ] Tag the exact state that produced the submitted results
      (`git tag -a paper-comstat-v1.0`), and reference the tag in the paper.
- [ ] Release the tag on Zenodo; put the DOI in the Data Availability statement
      (`CITATION.cff` carries the same metadata).
- [ ] `docs/MANIFEST.sha256` regenerated after the final commit, so the checksums
      refer to the released files.
- [ ] `requirements-lock.txt` regenerated on the machine that produced the final
      results (it is environment-derived, never hand-edited).
- [ ] Licence files present (`LICENSE`, `LICENSE-data`).
- [ ] No credentials, absolute private paths or editor metadata committed.
- [ ] Issue tracker enabled and the vignette's troubleshooting section links to it.
