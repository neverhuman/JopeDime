# Campaigns

> **Status on `main`:** this page describes the target design from the port plan. Run `jope --help` on `main` for the verbs that exist today. There is no `jope learn`, `jope hwm`, `jope scores` or `jope arena`; the learning store is the separate `jope-learning` binary, with `hwm` and `scores` subcommands, and the arena is the separate `jope-arena` binary. See the README's "What this SHA actually does".

`crates/jope-campaigns` (package `jope-dime-regressor` until the WP5 rename) is the benchmark
side of the repository: the programs that run jope-dime against fixed panels, keep their receipts
and produce the numbers the SOTArena scores and Elo ladders are made of. Plan sections:
"Inventory" (the root crate row), "CI and testing" (resource gates), red-team blockers B1, B6–B8,
B27; decision [ADR-0004](adr/ADR-0004-evaluator-identity-known-table.md).

## What the crate runs

| Family | Source | What it does |
|---|---|---|
| Golden42 / Golden46 panels | `src/golden42_v3.rs`, `src/main.rs` (`run-golden46`, ledger validators), `manifests/panel.txt`, `manifests/panel-analysis.v1.json` | The frozen regression panels: **Golden42 = SOTArena `tabular-regression-v1`** (42 datasets), **Golden46 = `tabular-regression-v2`** (46). A panel run searches per-dataset programs under the exact-CV protocol, writes a per-cell ledger and a max-manifest (`jope-dime.golden42-v3-max-manifest.v1`), and is what the Elo reporter reads |
| Exact CV / binary CV | `src/commands/{exact_cv,exact_cv_v2,binary_cv}.rs`, `seeds/generation-zero{,-binary}.txt` (B6/B7 — `include_bytes!` inputs, hence crate-local) | The evaluation protocol every score is measured under: fixed folds, fixed seeds, receipts with the evaluator identity |
| Test-adaptive | `src/test_adaptive/`, `manifests/binary42-test-adaptive.v1.json`, `fixtures/binary42-historical-reports/v1` (B8) | Adaptive test-time protocols and their historical report replays |
| DL best5 / budget-first | `src/jope_dlgp_best5/`, `tests/dlgp_best5.rs`, `tests/dlgp_budget_campaign.rs`, `manifests/dlgp-budget-first/` | The deep-learning-proposed program campaigns (policy proposals → search refinement) |
| Generalized search, search v3 | `src/jope_generalized_gp/`, `src/search_v3/`, `tests/{generalized_gp,search_v3}.rs`, `src/bin/search-v3.rs` | The multi-task generalisation and the v3 search driver |
| Hyperion gate / router, encoder v2 | `src/hyperion_{gate,router}.rs`, `src/encoder_v2_*.rs`, `manifests/hyperion-*`, `tests/{hyperion_gate,encoder_v2_recovery,router_contracts_v2}.rs` | The routing/gating campaigns that decide which encoder composition serves which task |
| Evaluation identity | `src/evaluation_v2.rs`, `known-evaluator-identities.json` | `JopeDimeEvaluatorIdentity.v1` hashes ~30 feat-math sources plus the manifest and lockfile; receipts (`JopeDimeSemanticEvaluationReceipt.v2`) validate under the current identity **or** any row of the append-only known table (ADR-0004), so the workspace merge and the WP2 feat-math changes did not invalidate history |
| Train-only study / red-team | `src/commands/train_only_*.rs` | The train-only leakage studies; read the Elo-reporter root (`SOTARENA_REPORTER_ROOT` after WP4) |
| Corpus eval, canary, capacity, causal tools, reporting | `src/bin/{corpus_eval,jope_canary_executor,jope_capacity,jope_causal_*,jope_reporting,…}.rs` | Campaign-side operational programs (55 binaries today; one binary with argv[0] dispatch after WP5) |
| Self-play control plane (moving to `jope-selfplay` with WP5) | `src/selfplay_*.rs`, `src/ratchet/`, `src/selfplay_shadow/`, `src/bin/jope_selfplay_*.rs`, `manifests/selfplay/`, `deploy/systemd/selfplay/*.service` (39 units), `tests/systemd_unit_paths.rs` | The lineage-A flywheel of [ADR-0005](adr/ADR-0005-canonical-learning-loop.md): queue → freshness → supply → ingestion → corpus-freezer → trainer-watchdog → promotion; kept with its tests, not scheduled by default |

Crate-local `manifests/` (B1: `include_str!("../../manifests/selfplay/scq-nodes.v3.toml")` and
friends), `seeds/`, `fixtures/` and `tests/fixtures/` stay next to the code because the code
includes them at compile time or resolves them through `CARGO_MANIFEST_DIR`. The one oversize
manifest (`manifests/multiclass/mc-union-runnable272-v1.json`, 21.4 MB) lives in the learning
store, not in git.

## Golden42 / Golden46 and the xbabe2-only resource gates

`src/golden42_v3.rs` was written for one machine. Its constants are the gate:

```rust
const WORKERS: usize = 80;
const NODE_LEASE: &str = "…/jope-golden42-controller-node-leases/xbabe2-v3.lock";   // → $JOPE_STATE with WP4
const MIN_RAM: u64 = 64 * 1024 * 1024 * 1024;   // 64 GiB
const MIN_DISK: u64 = 100 * 1024 * 1024 * 1024; // 100 GiB
```

plus `taskset` CPU pinning (`golden42_v3.rs:1150,1556`) and, in the Python tooling,
`taskset -c 60,123`. A panel run therefore **never executes on a hosted runner** (B27): the code
sits behind an explicit `xbabe2` marker, `.config/nextest.toml`'s `pr` profile excludes
`package(jope-campaigns)` entirely (the crate is `cargo check --all-targets`-ed on PRs and its
tests run in the `main` profile on xbabe2), and the bounded substitutes are:

- nightly `campaign-smoke` (WP8): one Golden46 panel cell, 200 evaluations, compared with
  `assets/campaigns/baseline.json` within tolerance;
- weekly `regression-suite` (WP12): one Golden42 panel cell (42 datasets × 1 seed, RMSE within
  1e-9 of `assets/regression/golden42-baseline.json`), encoder parity, policy parity, restore drill.

Full historical campaigns stay reproducible on demand from the learning store (`jope learn pull`
+ the campaign program), which is the plan's non-goal made explicit: no re-running of history on
every change.

## Running a campaign locally (CPU, small)

```bash
# the crate builds and its tests compile since B6 (the recovered exact-cv fixture is committed)
cargo check -p jope-dime-regressor --all-targets --locked
cargo test  -p jope-dime-regressor --locked --lib          # main tier on xbabe2; slow locally
cargo run   -p jope-dime-regressor --locked --bin search-v3 -- --help
```

Datasets come from `JOPE_REGISTRY` (`jope datasets fetch golden42 --print-registry`, WP-D4) and
the Elo-reporter data from `SOTARENA_REPORTER_ROOT` ([ADR-0003](adr/ADR-0003-runtime-roots.md)).
Scores and receipts a campaign produces are appended to the learning store's `scores/ledger`
(`jope scores append`, WP11) and published to SOTArena through `jope arena` (WP12,
`docs/arena.md`). The 55 historical program names, their new homes and their tests are enumerated
row by row in `docs/parity/capability-matrix.md` (PR
[#12](https://github.com/veox-ai/JopeDime/pull/12)).

## Targeted evolution of one dataset (`jope pipeline evolve`, CPU)

The panels above measure many datasets under one protocol. A targeted campaign drives **one**
chosen dataset to better GP programs over rounds and turns each round's champion into the next
round's starting point. Everything it runs exists today in `jope` and `jope-learning`; nothing is
scheduled, uploaded or promoted.

```bash
jope pipeline evolve --dataset datasets/test/uci-parkinsons-telemonitoring --rounds 3 --control \
  --cuts 3 --budget evaluations=200,seconds=600 --threads 8 --policy skip --emit rust \
  --output <campaign>
jope pipeline lineage verify --lineage <campaign>/lineage.json --require-bred-candidates
for r in round-001 round-002 round-003 control; do
  jope-learning hwm ingest --source pipeline <campaign>/$r/receipt.json
  jope-learning bank add --receipt <campaign>/$r/receipt.json --lineage <campaign>/lineage.json
done                                  # the bank accepts only HWM-improved champions
jope-learning bank export --dataset <dataset_hash> --out seeds.json
jope pipeline evolve --seeds seeds.json ...            # the next campaign starts from the bank
jope pipeline score --receipt <campaign>/round-003/receipt.json --split test   # once per genome
```

- **Rounds** share one prepared set of cuts (fixed fold seed and cut count), so their mean
  held-out RMSEs are directly comparable; round r > 1 is seeded from round r-1's champion and
  ranked runners-up and can never end worse than it. `--control` repeats the final round's
  configuration unseeded, which separates what seeding carried forward from what the search
  found on its own.
- **The learning store** is written only by the separate `jope-learning` steps: the HWM ledger
  records initialized / improved / tie / not-improved under a pipeline comparison identity (never
  teaching-eligible), and the champion bank keeps improved champions as `jope.pipeline-seeds/v1`
  targets for later campaigns.
- **The sealed test split** is read only by `jope pipeline score`, once per (dataset, genome),
  and never feeds selection, seeding or the bank.
- **Cores**: `--threads` and `--islands` multiply into the worker count, which is refused above
  the CPUs the process may use or an explicit `--cpu-ceiling`; no decision depends on it.

Command reference: [`docs/pipeline.md`](pipeline.md) ("Targeted evolution", "Test-split
scoring") and [`docs/learning.md`](learning.md) ("Pipeline champions", "The champion bank").
