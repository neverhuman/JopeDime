# jope-dime

`jope-dime-regressor` is a self-contained Rust implementation of the frozen
42-dataset campaign and the additive Golden46 refresh. It vendors `feat-math`, pins the 60-genome seed
population (`5f097a0017ddc7e92ad5ea8dfadaf783b548e82287d92626dc9d2fa9429b2105`),
uses balanced dataset-hash folds, and writes immutable per-dataset receipts.

Build the release binary with:

```text
cargo build --release
```

The registry defaults to `/home/ubuntu/remote_super/quant/regression`; use
`--registry` to point at an equivalent local mirror. The panel text is pinned
to `b86c8ea5f5ab1051e8aafb4368f8dabdcfd8c740c63eaa1d086ddb72fae0ba75`.

Commands:

* `verify-inputs` checks the pinned seed, panel, and all 42 `train.csv` files without opening test data or baseline metrics.
* `select-panel` performs training-only evolution and freezes all winners. For a smoke run, add `--dataset <hash>`.
* `validate-panel` audits all freezes, 25,200 evolutionary slots, and 420 complete CV7 finalist receipts using training data only.
* `score-panel` validates the complete panel first, then opens baseline/test data and publishes 42 immutable reports, predictions, pairwise results, and Elo standings.
* `run-panel` runs both phases; `summarize` writes the panel results ledger.
* `run-losses-3x` resumes the pinned exploratory follow-up for the 35 original
  strict losses at 30 total generations. It validates all generation-29
  freezes and their exact 600-slot replay prefixes before scoring, archives
  the current public artifacts, and performs the idempotent replacement.
* `validate-losses-3x` audits the 35 follow-up freezes, 63,000 slots, 350 CV7
  receipts, and the original 600-slot prefixes without opening test data or
  baseline metrics.
* `run-test-adaptive-island-extension --run-id test-adaptive-golden42-v1`
  continues the seven generations-10–19 no-wins through complete generations
  20–29 on four deterministic 30-candidate islands. Add `--resume` only to
  resume the identical binary-bound campaign.
* `validate-test-adaptive-island-extension --run-id
  test-adaptive-golden42-v1` independently reconstructs the four-island
  manifest, 8,400 logical slots, ring migrations, lineages, replays, and final
  summary. These exploratory artifacts always record `benchmark_valid: false`.
* `run-binary-test-adaptive --run-id binary42-test-adaptive-ilpd-v1 --dataset
  962a35a53b0803a8 [--resume]` runs the Binary42 ILPD pilot as four immutable
  15-candidate SCQ shards pinned to xbabe1–4. Held-out test ROC-AUC is used
  directly as fitness, so every artifact is marked `test_adaptive: true` and
  `benchmark_valid: false`; the lane never updates a published leaderboard.
  `validate-binary-test-adaptive` independently rebuilds its indexes and audits
  generation barriers, hashes, lineage, and four-node winner confirmation.
* `run-binary-test-adaptive-all --run-id binary42-test-adaptive-all-v1
  [--resume]` runs all 42 frozen Binary42 datasets with six datasets active at
  once. Each 60-member generation uses pinned SCQ shards of `19/19/19/3`
  candidates on xbabe1–4. Every fitted genome is lowered to C++17, compiled
  with `g++ 13.3 -O3 -march=znver2`, and scored from the compiled model's
  probabilities. Each dataset completes its generation barrier before deciding
  and stops after an evolved winner or the generation-9 exhaustion receipt.
  At cohort creation, provenance-verified Binary42 winners from prior runs are
  snapshotted in frozen dataset order and automatically placed first in every
  generation-zero population; the static binary bank fills the remaining
  unique slots.
  The six active datasets reserve 114 CPUs on each 128-CPU node and 18 CPUs on
  the 24-CPU xbabe4 node, leaving headroom for controllers and four-node
  confirmations.
  `validate-binary-test-adaptive-all` rebuilds per-dataset and cohort indexes
  and requires exactly 42 terminal datasets. These results are test-adaptive,
  benchmark-invalid, and never published to SOTArena leaderboards.
* `admit-fitted-backend-v1 --output <receipt>` runs the fitted correctness
  probe and freezes a backend decision. C++17 is never admitted from the
  standalone `Plan` emitter; complete fitted-registry parity is required.
* `run-golden42-aggressive --run-id <id> [--dataset <hash-or-prefix>] [--resume]` launches the additive
  four-by-60 SCQ80 study after its hardware, swap, queue, staging, and backend
  gates pass. `validate-golden42-aggressive --run-id <id>` audits its complete
  generation barriers and bit-identical three-node winner confirmations. All
  outputs are explicitly test-adaptive and `benchmark_valid: false`.
* `run-golden46 --run-id golden46-new3-20260723-v4 [--dataset
  <exact-id-or-hash-prefix>] [--resume]` runs the four-by-nine Golden46
  population on ZTS_small and the two NOMAD datasets against their frozen
  external leaders. Full ZTS and its transfer arm are excluded from the
  required study. `validate-golden46` audits frozen
  generation-zero seed banks, lineage, cache accounting, and bit-identical
  three-node confirmations.
* `validate-ratchet-artifact --input <json>` validates a continuous-learning
  contract without mutating state. Legacy seed, generation, admission, and
  telemetry receipts remain readable, but cannot authorize promoted work.
* `ratchet-retain-model-seeds --request <v2.json> --response <v2.json>
  --excluded-semantics <json-array> --output <receipt>` applies Rust's fixed
  reward/disagreement/novelty/seeded-temperature 32/16/8/8 selection to all 96
  Python proposals.
* `ratchet-commit-dataset-state --root <ratchet-root> --state <v2.json>
  --committed-at-unix-seconds <n>` advances an append-only, content-addressed
  dataset-state pointer.
* `ratchet-publish-training-head --root <ratchet-root> --manifest <head.json>
  --micro-batch <batch.json> --replay <replay.json>` publishes a non-serving
  checkpoint only after an exact eight-dataset, 100-step replay update.
* `ratchet-publish-serving-wave --root <ratchet-root> --manifest <wave.json>
  --rollout-switches <switches.json> --frozen-splits <splits.json>` atomically
  advances `waves/current.json` after dependency, parentage, label/time,
  train-only gate, training-head, and sealed-holdout checks.
* `ratchet-admit-attempt --root <ratchet-root> --generation-zero <v3.json>
  --admission <v3.json>` commits an attempt only when it binds the current
  serving wave and the dataset's latest exact state snapshot.

The continuous-learning authority is additive and does not alter historical
campaign behavior. `training-head.json` has no seed authority;
`waves/current.json` is the sole serving pointer; `dl/current.json` is only a
non-authoritative compatibility mirror. See
[`docs/continuous-learning-ratchet.md`](docs/continuous-learning-ratchet.md).

Selection artifacts live under `runs/<panel-id>/<dataset-hash>`. `score-panel`
writes the panel ledger; `summarize` validates and prints that existing ledger.
Existing reports, predictions, and ledgers are never overwritten.

The vendored `feat-math` crate also exposes the append-only advanced-v3 API:
`TaskSpec`, `DataSchema`, `RawTable`, `fit_table`, `predict_table`, and
`emit_bound`. Advanced-v3 artifacts bind the task, schema, and fitted
categorical vocabulary and accept quoted raw CSV. The historical numeric
`fit`/`predict` and emitter-2.1 APIs are unchanged. `--kernel-version` defaults
to `legacy-v2`; advanced-v3 campaign launch remains closed until its recorded
admission benchmark passes, and existing manifests can never be relabeled or
resumed with the new kernel.

Golden46 preparation is available through the deterministic helpers in
`scripts/build_zts_small.py` and `scripts/build_golden46_manifests.py`.
`ZTS_small` is a 1,000/1,000 headerless coreset copied from the authoritative
ZTS splits; its manifests retain source-row indices and hashes, and raw new
dataset bytes remain withheld pending redistribution rights. The worker
execution in `scripts/golden46_execution.py` masks test targets and leaves
RMSE calculation to the controller. Golden42 remains the immutable
`tabular-regression-v1` track; Golden46 is published as
`tabular-regression-v2` only after its 375-cell required-study audit gate
resolves. The cohort-consistent gate contains 372 scores and three
historical-ASPFM not-rerun terminal receipts. Full ZTS remains preserved as an
out-of-scope partial campaign. The converted Chimera
checkpoint is accepted only through its source-bound safetensor receipt and
strict 32-tensor compatibility load.

The original 42-dataset command remains immutable. The explicitly named
follow-up is the sole exception to the final sentence above: it replaces only
the 35 loss reports and public predictions plus the existing panel ledger,
after writing a content-addressed rollback archive. Its
`OFFICIAL_REPLACEMENT_COMPLETE` marker is written last. The seven original
winners and every original selection/local phase-two artifact remain unchanged.
