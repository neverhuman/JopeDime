# Frozen challenge regressors

This directory holds General Regressor Challenge entrants written in safe Rust under the
contract in `docs/challenges/frontier-regression-master-spec.md`. Each entrant is one
`RegressorApi` source file. Its training table carries the target in the last column; its
prediction input is features only.

`jope-bench` compiles every entrant into its binary. `src/regressors.rs` runs each one through
the same `jope.bench-adapter/v1` process contract as the XGBoost, LightGBM, CatBoost and
random-forest adapters in `../adapters/`. Entrants are therefore screened, deepened and
Elo-laddered on identical cuts by the same evaluator:

```sh
jope-bench run --datasets <tree> --out <reports> --entrant Opus5p5_Extra \
  --adapter "xgb=python3 crates/jope-bench/adapters/xgboost_adapter.py"
jope-bench regressor-adapter Opus5p5_Extra --version
jope-bench regressor-adapter Opus5p5_Extra [--seed N] train.csv predict.csv out.csv regression
```

The seed defaults to the bench fold seed (`20260907`). Entrants are regression-only. `run`
attaches them to regression datasets alone, and the adapter process refuses any other task.

## Freeze rule

An entrant's files are never edited. `src/regressors.rs` pins each source's SHA-256, and the
`frozen_sources_match_their_pinned_hashes_and_manifests` test fails on any drift. Workspace
lints are relaxed on the module declaration (`clippy::all`, `unused_assignments`,
`rustfmt::skip`) so they do not force edits to the source. A corrected or improved source is a
**new entrant**: new file, new name, new hash. The original and its results stay recorded.

## Entrants

### `Opus5p5_Extra`: CBRS, Cross-fitted Boosted-Ridge Stack

| File | SHA-256 |
| --- | --- |
| `Opus5p5_Extra.rs` (13.1, 66,536 bytes, 2,284 lines) | `2e2352f8ac7087076ef60e0408dfee5319a8d7aa8aaac8d657bc67f07f755834` |
| `Opus5p5_Extra.submission.json` (13.2) | `6e356ced71c307ca8994ae9fab1205923f4a639202f80233b7a01b66e25b899f` |
| `Opus5p5_Extra.design.md` (13.3, 2,030 words) | `0297dc240df5dc966a3dc2b6d20c45830cb8255b17bc023f93f7f9bc2c0520a9` |

**Authoring.** Configured model `claude-opus-5-5`, in the `reasoning_only` mode. Nothing was
compiled, executed, tested or tuned on data before the freeze. The source and manifest were
emitted in one conversation message (2026-10-21, before 06:24Z). The checked-in `.rs` and
`.json` were extracted from that message and compared byte-for-byte with it before any compile.
`design.md` was written after the freeze and before the first compile. Repository code was not
read until the source was already frozen.

**Post-freeze admission.** These are organizer actions on 2026-10-21. They supplied no feedback
to the design, and no byte was changed.

- **Compile:** `rustc 1.97.1 --edition=2021 --crate-type rlib -C opt-level=3 -C codegen-units=1 -D unsafe-code`
  exited 0 on the first attempt with two `unused_assignments` warnings, a never-read initial
  `rounds_f = 0.0` in the two work planners.
- **Contract checks:** 35 of 35 master-spec O6 checks passed on tiny invented arrays, run by
  [`admission/Opus5p5_Extra.contract-driver.rs`](admission/Opus5p5_Extra.contract-driver.rs)
  (log: [`admission/Opus5p5_Extra.contract.txt`](admission/Opus5p5_Extra.contract.txt)). They
  cover shape and overflow errors, infinite and non-finite inputs, zero features, one and two
  rows, constant targets, all-missing columns and rows, repeated rows, signed zeros, lifecycle
  errors, identical results across fresh instances and repeated predictions, batched and
  permuted prediction, and refit isolation.
- **Informational smoke:** the driver's invented `y = 3a + sin 2b + step(c)` data, on a 2-vCPU
  cloud container with `-C opt-level=3`. This is not a qualification or an admission gate.

  | Rows | Features | Fit | Predict (n/4 rows) | Held-out RMSE | Mean-baseline RMSE |
  | ---: | ---: | ---: | ---: | ---: | ---: |
  | 2,000 | 8 | 3.1 s | 0.13 s | 45.5 | 3,515.5 |
  | 20,000 | 20 | 27.4 s | 1.4 s | 37.0 | 3,624.6 |
  | 100,000 | 10 | 31.3 s | 3.0 s | 46.6 | 3,678.8 |

**Disclosed limitations** (from `design.md` §8):

- When tree-feature screening is active (more than 2,000 eligible features, or binned cells
  above 6·10⁸), it uses all working-row labels. In that regime, its out-of-fold blend estimates
  are not fully clean.
- Out-of-fold estimates also reuse the rows that chose early stopping.
- Predictions are capped at 10% beyond the training target range.

**Authoring cost.** This was recorded for the O3 authorship receipt. The pre-freeze turns
reached this record only as imported history with no usage telemetry, so those figures are
estimates. Post-freeze figures are measured from the session transcript.

| Phase | Wall time | Tokens |
| --- | --- | --- |
| Prompt to frozen working source (estimated) | About 25 min (likely 15–40). Only the final source message is bounded: at most 6.3 min, from 06:17:44Z to before 06:24Z. | About 75k output, likely 50–100k, including reasoning. The final 68.5k-character message alone is about 20k. At least one earlier, incomplete attempt preceded it ("continuing in smaller pieces"). Input was about 30–80k, mostly the roughly 16k-token spec re-read each turn. |
| Freeze, admission and bench integration (measured) | 06:40:49Z to about 07:05Z, about 25 min | 94k+ output (27k reasoning), 245k new input. Repeated context reads of 19.5M+ cached tokens are not new tokens. |

The source needed zero post-freeze edits, so the frozen message is the final working source.

## Integration review correction

The transport rejects nonempty CSV cells for zero-feature prediction rows. The
run --seed value is forwarded to the learner, including its version probe;
registered entrant adapters use the existing positive, finite timeout validation.
These wrapper corrections preserve the frozen learner source and its identity.
