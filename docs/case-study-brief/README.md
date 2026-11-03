# Case-study briefs

Two regenerable documents about the [case studies](../../case_study/README.md). Neither is the
white paper (`docs/whitepaper/`), and neither claims more than the evidence it is built from.

| File | Audience | What it covers |
| --- | --- | --- |
| `brief.pdf` | anyone | Only the cases where our evolved models are a **clear winner**: nothing else we ran is competitive on that case's business scorecard. It shows longer evolutions, where the best runs started and where they ended up, and how every alternative scored. |
| `campaign-technical.pdf` | technical reviewers | The full record of all 14 cases: every arm, every alternative, the leakage audit and the leaderboard (validation stage, single holdout). |

## How the numbers in `brief.pdf` are produced

- **Train-only cross-validation.** Every score is the mean over the 5 folds of the dataset's
  `train.csv`: the fold partition in `prepared.json` (cuts 0–4, fold key `<id>:20260907`).
  - Our programs are searched with `case_study/bin/extend_case.sh --cuts 5`. Each fold runs its own
    evolution, candidates are cross-measured, and a program is ranked by its mean over all 5 folds.
  - Each alternative is fitted on 4 folds and scored on the fifth, 5 times. Scoring uses the same
    `jopedime fitness evaluate` and the case's own `objective.json`.
  - `test.csv` is never read.
- **Clear winner:** our typical run (the median over 6 runs: 2 seeding strategies × 3 random
  seeds) beats the best of every alternative's 5-fold mean, and the best run also wins fold by
  fold.
- **Where a run started and ended up:**
  - the start is the best program in the run's first generation;
  - the end is the best program the run ever measured;
  - the final champion is the program the last round selected.
  - All are recomputed from the per-fold observations, and each round's value is checked against
    its receipt.

## Build

```sh
make -C docs/case-study-brief              # brief.pdf (build-winners.sh, then latexmk)
make -C docs/case-study-brief technical    # campaign-technical.pdf (build-data.sh, then latexmk)
```

**Requirements:** bash, jq, gawk and a TeX distribution with latexmk, pgfplots, siunitx and
tcolorbox.

**Inputs** are read only:

- `JOPE_CASE_OUT` (default `~/jope-data/case_study`): receipts and per-fold observations.
- `JOPE_CASE_BASELINES` (default `~/jope-data/resfit-2071/baselines`): `<case>/summary_cv.csv` for
  `brief.pdf` and `<case>/summary.csv` for the technical record.
- `BRIEF_WINNERS` (default `09 02 12`) and `BRIEF_CV_STAMP` (default: the latest `extended-*-cv5`
  runs).

## What is generated

`winners-data/` (for `brief.pdf`):

- `NN-chains.csv`: per run, the start, the end, the final champion, rounds and evaluations.
- `NN-evo.csv` and `NN-evo-rmse.csv`: best-so-far per generation, one column per run.
- `NN-bench.csv`: every alternative's 5-fold mean, plus our start and our end.
- `NN-folds.csv`: our best run against the best alternative, fold by fold.
- `alternatives.csv`, `macros.tex`, and `inputs.txt` (every evidence file read).

`data/` (for `campaign-technical.pdf`): see the comments in `build-data.sh`.
