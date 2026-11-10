# Native AutoML regressors

Four original safe Rust regressors, with no dependencies beyond the standard library, are
registered in `jope_automl::REGRESSORS`:

| Id | Source | SHA-256 |
| --- | --- | --- |
| `GPT6p1_Sol_Max` | [`src/GPT6p1_Sol_Max.rs`](src/GPT6p1_Sol_Max.rs) | `fe5bb21e…f75a8` |
| `Gemini4_argon_high` | [`src/Gemini4_argon_high.rs`](src/Gemini4_argon_high.rs) | `d5cd85fc…053d3` |
| `grok4p7_xhigh` | [`src/grok4p7_xhigh.rs`](src/grok4p7_xhigh.rs) | `8e648cef…664f8` |
| `kimik3p1_max` | [`src/kimik3p1_max.rs`](src/kimik3p1_max.rs) | `0c211694…6ceb3` |

`kimik3p1_max` is model release **1.0.0** (`KIMIK3P1_MAX_VERSION`). The receipt is
[submissions/kimik3p1_max/RELEASE.json](submissions/kimik3p1_max/RELEASE.json).
That version is the regressor release. The workspace crate version stays `1.0.0`.

Each regressor's entire learning procedure lives in its own file. That file is included
byte-for-byte and compiles independently:

```sh
rustc --edition=2021 --crate-name gpt6p1_sol_max --crate-type rlib \
  -C opt-level=3 -C codegen-units=1 -D unsafe-code \
  crates/jope-automl/src/GPT6p1_Sol_Max.rs -o libgpt6p1_sol_max.rlib
rustc --edition=2021 --crate-name gemini4_argon_high --crate-type rlib \
  -C opt-level=3 -C codegen-units=1 -D unsafe-code \
  crates/jope-automl/src/Gemini4_argon_high.rs -o libgemini4_argon_high.rlib
rustc --edition=2021 --crate-name kimik3p1_max --crate-type rlib \
  -C opt-level=3 -C codegen-units=1 -D unsafe-code \
  crates/jope-automl/src/kimik3p1_max.rs -o libkimik3p1_max.rlib
```

`GPT6p1_Sol_Max` was frozen before repository-code inspection or numerical execution.
`Gemini4_argon_high` was supplied by the owner on 2026-10-22. It replaces the earlier
`gemini-argon-4` candidate as the Gemini entrant, and its authoring conditions are recorded as
unknown. `kimik3p1_max` was supplied by the owner on 2026-11-07 as a single file without an
authoring manifest; its authoring conditions are likewise recorded as unknown.

## Using a regressor

AutoML callers can enumerate `jope_automl::REGRESSORS`, resolve `definition(id)` and call
`fit_rows(id, training_features, training_targets, seed)`. Each returned `FittedRegressor` owns
independent state and accepts features only through `predict_rows`.

- **Transport.** The adapter appends training labels as the final column, preserves NaN missing
  features and adds no numerical preprocessing.
- **Non-finite output.** It refuses a non-finite prediction with an error, because a frozen source
  is never patched.
- **Exact challenge API.** Also available directly: `Gpt6p1SolMax` with `RegressorApi`,
  `Gemini4ArgonHigh` with `Gemini4ArgonHighApi`, and `KimiK3p1Max` with `KimiK3p1MaxApi`.

The existing same-cut benchmark accepts any of these candidates alongside its existing baseline,
program and external-adapter methods:

```sh
cargo run --release -p jope-bench -- run \
  --datasets /path/to/catalog --out /path/to/reports \
  --native GPT6p1_Sol_Max --native Gemini4_argon_high --native kimik3p1_max \
  --seed 20260907 --folds 5 --top 20 --workers 1

cargo run --release -p jope-bench -- ladder \
  --reports /path/to/reports --track-root /path/to/arena \
  --track native-regressors
```

- `--native` is explicit and repeatable; unknown names fail before fitting.
- A method is recorded as `native.<id>`, with its source SHA-256 and seed in the report.
- It is fitted anew on each cut's training rows, then scored and pooled by the existing benchmark.
- Non-regression datasets are skipped when a native regressor is requested.
- Historical default rosters, frozen panel plans, GP grammars and same-cut Elo calculations keep
  their existing definitions.

The same-cut integration is trusted in-process product code. It does not implement the isolated,
CPU-, memory- and time-enforced challenge supervisor described by the master specification. A
competition organizer must provide that supervisor separately.

## AutoML Elo

[`assets/automl-elo/v1`](../../assets/automl-elo/v1/SOURCES.md) records the four-entrant AutoML
Elo: these two native regressors plus `Opus5p5_Extra` and `GPT6_Astra_Max`.

- **Panel.** 64 dataset families × 3 folds × 3 seeds, with one CPU and 8 GiB per fit.
- **Ratings.** Opus5p5_Extra 1641, **Gemini4_argon_high 1561**, GPT6p1_Sol_Max 1456,
  GPT6_Astra_Max 1342, each with a 95% family-bootstrap interval.
- **Caveat.** Gemini4_argon_high's RMSE differences from the other entrants are not significant
  after Holm adjustment.
- **Figures.** Ratings, pairwise win shares and source sizes are in
  [`figures/`](../../assets/automl-elo/v1/figures).

[`assets/automl-elo/v2`](../../assets/automl-elo/v2/SOURCES.md) refits the v1 panel with
`grok4p7_xhigh` added. The other four entrants are not refitted; their ratings move because
Bradley-Terry is recomputed on the five-player pool. v1 is unchanged.

[`assets/automl-elo/v3`](../../assets/automl-elo/v3/SOURCES.md) refits the same panel with
`kimik3p1_max` added to the five-entrant pool (the v1 four plus `grok4p7_xhigh`). No earlier
entrant is refitted; ratings move only because Bradley-Terry is recomputed on six players, and
the analysis first reproduces the five-entrant result exactly. v1 is unchanged.

- **Ratings.** Opus5p5_Extra 1691, Gemini4_argon_high 1603, **kimik3p1_max 1511**,
  GPT6p1_Sol_Max 1498, GPT6_Astra_Max 1389, grok4p7_xhigh 1308.
- **Caveat.** Kimi and Sol are level within their intervals: Kimi wins 55% of families against
  Sol, but Sol's geometric-mean RMSE is lower (ratio 0.764, not significant after Holm
  adjustment). Gemini's lower RMSE than Kimi's is significant after Holm adjustment.

## Submission records

| | `GPT6p1_Sol_Max` | `Gemini4_argon_high` | `kimik3p1_max` |
| --- | --- | --- | --- |
| Manifest | [submission.json](submissions/GPT6p1_Sol_Max/submission.json) | [submission.json](submissions/Gemini4_argon_high/submission.json) | [submission.json](submissions/kimik3p1_max/submission.json) |
| Design | [design.md](submissions/GPT6p1_Sol_Max/design.md) | [design.md](submissions/Gemini4_argon_high/design.md) | [design.md](submissions/kimik3p1_max/design.md) |
| Authorship and session receipt | [authorship.json](submissions/GPT6p1_Sol_Max/authorship.json) | [authorship.json](submissions/Gemini4_argon_high/authorship.json) | [authorship.json](submissions/kimik3p1_max/authorship.json) |
| Verification | [verification.json](submissions/GPT6p1_Sol_Max/verification.json) | [verification.json](submissions/Gemini4_argon_high/verification.json) | [verification.json](submissions/kimik3p1_max/verification.json) |

They distinguish source authoring from later correctness checks. Prior project context existed
when `GPT6p1_Sol_Max` was written, so it is not represented as a pristine fresh-session primary
entrant.

## Verification

```sh
cargo test -p jope-automl --locked
cargo test -p jope-bench --lib native::tests --locked
```

These checks cover target placement, lifecycle, deterministic refits, prediction batching and row
locality, missing, degenerate and extreme inputs, source identity, registration, train/test
separation and same-cut routing. They impose no accuracy threshold.

The frozen sources are excluded from automatic formatting and Clippy style rewrites; adapters
follow the workspace lint policy.
