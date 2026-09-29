# Jope Lambda Watch

A small Rust terminal dashboard for Lambda GP telemetry. Runs over SSH, reads local snapshots, and makes **no AWS API calls**. Bright cyan, mint, magenta and amber distinguish throughput, GP results, costs and budget exposure.

## Run on your Mac

Watch the coordinator's receipts on xbabe2:

```sh
rtk proxy ssh -t xbabe2 '~/JopeDime/.agent/experiments/lambda-watch/watch'
```

The existing campaign is closed, so this correctly displays **CLOSED**. To watch its actual recorded burst unfold:

```sh
rtk proxy ssh -t xbabe2 '~/JopeDime/.agent/experiments/lambda-watch/watch --replay --speed 1'
```

Replay is always labeled. It makes no Lambda invocations and spends no AWS request fees. Use `--speed 10` to accelerate it. Each viewer owns a separate snapshot, so simultaneous terminals cannot overwrite each other's live/replay state.

Keys: **↑/↓** or **j/k** select a view; **/** searches dataset, fold, task or ID; **f** shows failures; **Enter** opens scrollable full GP/KPI details; **Space** freezes the display; **?** shows help; **q** exits. In details, arrows/PageUp/PageDown scroll and Enter/Esc returns. The viewer cannot start, stop, retry or change training. Selection stays attached to a view as other results arrive.

## What the screen means

- In flight means **coordinator requests**, not measured simultaneous Lambda handlers. Their clocks and queue delays differ.
- Three independent throughput traces show call completions, dataset-event completions, and observed candidate fits, using one-second UTC buckets. The rightmost bucket is still filling. Completion receipts can arrive after the execution they describe.
- Dataset-event total counts qualified successful searches; its failures remain separate. The dataset completion graph includes terminal dataset events, including failures.
- The budget tile carries the entire original campaign's **$13.6535432** conservative reservations against the **$100** allowance. Observed Lambda/S3 charges are scoped to the selected phase. They are not an AWS invoice and do not release reservations. Missing bills remain unknown.
- Per-view KPIs use the recorded pilot objective: RMSE, logloss or macro-F1. Scores are never ranked across different datasets or metrics. A new attempt resets that view's quality curve. The native snapshot contract also accepts AUC and other explicitly named objectives.
- GP trees and program hashes correspond to the best observed candidate under that view's selection metric. These are adaptive selection scores, not held-out claims. The legacy run is a twelve-fit, capped GP screening protocol, not the planned 100-generation full-data campaign.
- Memory is the AWS reported environment peak per tier, not a per-model allocation measurement. p50/p95 describe observed handler durations, with a bounded 8,192-duration sample window.
- Stale, malformed, regressed or disconnected feeds are visible. A heartbeat proves the adapter is alive; **event age** separately shows whether work is arriving. Unknown calls are not fabricated successes or forever-running requests.

## Source placement and configuration

All code is under the canonical repository on xbabe2:

| Component | Location relative to `~/JopeDime/` |
|---|---|
| Monitor, Rust source and tests | `.agent/experiments/lambda-watch/` |
| Existing Rust Lambda training | `.agent/experiments/lambda-csv/rust/src/gp.rs` and `gp_v2.rs` |
| Existing dispatch, reservations and archival controls | `.agent/experiments/lambda-csv/gp/` |
| Full-data evolution plan and discovery tools | `.agent/experiments/gp-evolution-100/` |

The existing training code and closed campaign were not modified or relaunched. This is a standalone experimental crate, separate from the product Cargo workspace and other active PR owners. It is not a merged product feature.

Private `.env` is mode 0600, ignored by Git, and contains **paths only**. Start from `.env.example` when creating a separate configuration. The viewer needs no AWS keys. Training continues to use its existing external AWS credential provider and private environment configuration.

State, temporary viewer snapshots, logs, QA evidence and historical data live outside the repository in `/home/ubuntu/scratch/jopedime-lambda-watch-20260928/`. Stopping a viewer stops only its adapter; retained session snapshots/logs allow inspection.

## Live integration

For the existing receipt-based coordinator, change `JOPE_WATCH_RUN` and related paths in a separate env file, then run `watch --env /path/to/config`. The adapter notices request starts on its next poll, and candidate/model details when the invocation's durable receipt arrives. It reads existing files and the SQLite ledger in read-only mode. It does not change the frozen training worker.

For the planned large evolution controller, **publish compact snapshots directly** using [TELEMETRY.md](TELEMETRY.md), then run:

```sh
rtk proxy ssh -t xbabe2 '~/JopeDime/.agent/experiments/lambda-watch/watch --snapshot-only --env /path/to/config'
```

This uses only the Rust viewer and avoids the Python compatibility adapter. The all-fold population controller is not yet implemented; this monitor does not claim that campaign is running. Aggregation belongs in that controller, not in each terminal. Never scan millions of S3 keys to redraw a screen.

Full snapshots are read only when replaced. Unchanged sources use a tiny heartbeat sidecar; the viewer decays idle rates to zero without rereading models or whole snapshots. No polling of CloudWatch and no network client/AWS SDK is linked into the Rust binary.

## Build and verify

On xbabe2, from this directory, `./build` uses the pinned lockfile, cached dependencies, two build jobs, and one reused target directory. Runtime dependencies: a terminal plus the compiled binary; the compatibility launcher/feed use Python's standard library. Pillow and pyte are QA-only tools, not runtime requirements. The installed QA dependency directory is `/home/ubuntu/scratch/jopedime-lambda-watch-20260928/qa-deps` (pyte 0.8.2); set `PYTHONPATH` to it when running the PTY verification scripts.

Verification includes Rust reducer/layout tests, Python accounting/admission tests, real corpus receipts, a replay of 896 actual burst calls / 10,752 candidate fits, and PTY interaction/resize/partial-write/restart/stale/terminal-restoration tests. `PERFORMANCE.json` records measured binary size and resource use; source/evidence hashes accompany delivery.

The legacy adapter's initial read decodes old result bundles including models and has a larger startup memory footprint than the Rust viewer. Full-scale producers should use the direct snapshot contract.
