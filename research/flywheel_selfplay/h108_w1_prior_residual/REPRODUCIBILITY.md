# H108 reproduction index

This file indexes the compact, committed record. The experiment report and full
metrics remain in [README.md](README.md) and [metrics.json](metrics.json); no
source tree or checkpoint is copied into this folder.

## Protocol and inputs

- **Protocol:** W1 prior replacement / frozen-DOPE neural residual; the full
  arm definitions, slate contract, loss objective, endpoints, and H70 gate are
  recorded in the README. There were 45 neural cells: 3 arms × 5 outer folds ×
  3 seeds. Training slate: 192 rows (4 revealed anchors + 188 unrevealed);
  evaluation: full pool.
- **Dataset and split:** 1,219 DL-Train contexts / 205 families; five disjoint
  41-family folds, fold seed `4900`. The fold-manifest SHA-256 is
  `dd211134bba127960b1612687478064969dca422e35dfa889790760f316961c1`.
  Exact context/family/fold IDs and original-precision vectors are in the
  version-pinned `w1_per_seed_family_vectors.json` named in README §6; no
  outer-test, DL-Val, holdout, or sealed data was read.
- **Seeds:** fit-only pilot `700`; neural-cell seeds `701`, `702`, `703`;
  outer fold seed `4900`; bootstrap seed `20261031` (10,000 paired-family
  draws). Frozen DOPE SHA-256:
  `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`.

## Source, metrics, and loss record

- **Run:** `w1_gpu_20261031T123502Z_326209`. The merged readout/report PR is
  [#1036](https://github.com/neverhuman/JopeDime/pull/1036), reviewed source
  head `2bccebd6107d296d3318854ec61f26054112b334`, merge
  `d1364192a657b755cfa5e4dc39f6df093cd554f8`. The executed Phase 1–3 runner
  SHA-256 is `e4531b98f29106eb96cf4e636ae6ba1f553d11020e3dabde6a7adae84bee48c8`;
  the Phase 4 readout runner SHA-256 is
  `a7a2996734e48c77176108a4cd00429560fcba918ff6664a1f79ecdb843788fe`.
- **Elo:** use the separate Track 2 Verified U4 and Track 1 Legacy tables in
  README/metrics; the official H70 decision is `NO_ADVANCE`. Elo is reported,
  not AUC.
- **Loss availability:** `metrics.json.neural_cells[]` retains each of the 45
  cells' `loss_first` and `loss_last` values (plus endpoint steps), but the
  archived compact receipt contains no per-step training series or validation
  loss series. These endpoints are not a loss curve; no intermediate values
  are interpolated or inferred. The full run/source objects remain at their
  versioned S3 identities in README §6; no checkpoint is placed in Git.

## Reproduction

Use the exact-version S3 manifest and objects listed in README §6, then the
source hashes above. The S3 custody section records the run outputs and replay
vectors; Phases 1–3 shared qualification-lock custody is explicitly
**unverified**, and the result remains `NO_ADVANCE`.
