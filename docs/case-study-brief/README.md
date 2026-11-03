# Case-study brief

`brief.pdf` is a short, regenerable brief that introduces the strongest
[case studies](../../case_study/README.md) and their results. It covers resource-aware program
search on real data, compared with the best alternatives we could run. It is **not** the white
paper (`docs/whitepaper/`) and makes no claim beyond the evidence it is built from.

**Status: interim (validation stage).** Every number is measured on the cut-0 holdout of each
dataset's `train.csv`. The pre-declared single look at the test split has not been taken. Rebuild
the brief after it is taken; a claim is final only if the test look agrees.

## Build

```sh
make -C docs/case-study-brief            # data, then the PDF
make -C docs/case-study-brief data       # data/ only
BRIEF_CASES="03 09 02 01 07" make -C docs/case-study-brief    # the featured cases, in order
```

**Requirements:** bash, jq, gawk and a TeX distribution with latexmk, pgfplots, siunitx and
tcolorbox.

**Inputs** are read only and default to this host's evidence tree:

- `JOPE_CASE_OUT` (default `~/jope-data/case_study`): receipts, observations and extended rounds.
- `JOPE_CASE_BASELINES` (default `~/jope-data/resfit-2071/baselines`): `<case>/summary.csv` for
  the alternatives, scored by `jopedime fitness evaluate`.

## What is generated

`build-data.sh` writes everything under `data/`. The brief's numbers come only from there; none is
typed by hand.

- `champions.csv`, `baselines.csv` and `NN-table.tex`: per-case champions and alternatives.
- `NN-*.csv` and `NN-plots.tex`: chart series. Only non-empty series are plotted, so every legend
  entry matches its marks.
- `evolution.csv` and `evo/NN.csv`: per generation, every run's best new program and its
  best-so-far (global best) on the case's business objective, for every case. Extended chains
  number their generations continuously across rounds.
- `evolution-checks.csv`: each run's final best-so-far compared with `receipt.json`
  `.search.best.validation_objective`. Each main run must match exactly. An extended chain may
  hold its global best from an earlier round, which is marked `earlier-round`.
- `leaderboard.md` and `leaderboard-rows.tex`: the output of `case_study/bin/leaderboard.sh`.
- `provenance.csv`: host, source commit and binary hash of every run read.
- `inputs.txt`: every evidence file read. The digest printed in the brief hashes these files.
