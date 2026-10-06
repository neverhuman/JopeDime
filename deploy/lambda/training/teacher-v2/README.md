# S1 teacher engine and fixed-budget conformance

Read [SCHEMA.md](SCHEMA.md) for the current S1 protocol and the retained historical
record. The owner selected new search with inner-CV-only selection, 96 G0 slots
and 18 equal generations of 64 attempts per context. Codex root owns the core
landing; Claude dltrain-impl-v1 retains input preparation and its pilot.

Run CPU conformance from the repository root, reusing its target:

```sh
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --offline --locked --manifest-path deploy/lambda/training/teacher-v2/rust/lambda-worker/Cargo.toml --target-dir target
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --offline --locked --manifest-path deploy/lambda/training/teacher-v2/rust/island-engine/Cargo.toml --target-dir target
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --offline --locked --manifest-path deploy/lambda/training/teacher-v2/rust/s1-proposal/Cargo.toml --target-dir target
python3 deploy/lambda/training/teacher-v2/test_s1_budget.py
python3 deploy/lambda/training/teacher-v2/test_materialize_g0.py
python3 deploy/lambda/training/teacher-v2/qualify_s1.py --out /custody/s1-mutants
```

The mutation qualifier compiles the real source with one dependency violation
at a time and executes the exact intended test. It requires an assertion failure
or a declared receiving-path expectation, and rejects unrelated fixture panics
and Python ERROR outcomes. Each output directory must be fresh, so failed runs
cannot leave an earlier success receipt behind. A compile error, stale
binary, zero-test result or unrelated failure cannot count as a killed mutant.
It also checks channel-4 test-based and unbounded continuation mutations. Generated
mutant files live under the reused target; tracked sources and historical copies
are untouched. Hosted hygiene runs the budget and G0 tests, all three standalone
Rust workspaces through the existing Lambda lane, and the mutation qualifier.

## Two-fold S1 and G0 preparation

The October 6 S1-FOLDS-RULING selects explicit `jope.s1-two-fold/v1`.
Send `fold_protocol=jope.s1-two-fold/v1` in a worker request, use `--fit-s1`
for a local evaluation, and start the engine with `--s1-two-fold`.
The worker decodes only outer-training rows and emits two inner-CV folds without
official-test products. The engine requires both folds to beat their training-only
baselines and a geometric-mean RMSE ratio at most 0.90. Without these explicit
options the existing seven-fold path retains its five-of-seven gate. A requested
resume file must exist; it never silently starts a fresh search.

Prepare G0 using a pinned proposal executable built from its Cargo artifact:

```sh
python3 deploy/lambda/training/teacher-v2/materialize_g0.py \
  --context CONTEXT_SHA256 --input ADMITTED_TWO_FOLD.jtt1 \
  --proposal PROPOSAL_BINARY --proposal-sha256 PROPOSAL_SHA256 --out NEW_PREPARED_DIR
```

This emits 96 fresh context-seeded prior programs, a training-only sample manifest,
64-by-OOF-row projection and dependency manifest. Existing output directories
refuse. Proposal regeneration is bounded separately and consumes no fitting slots.
The admitted supervisor must reserve and charge all G0 fits, including failures and
retries, before dispatch; this preparation command does not dispatch them.

After receiving exactly 96 terminal evaluation records in seed order, score them:

```sh
ISLAND_BINARY --score-g0 --context CONTEXT_SHA256 \
  --sample NEW_PREPARED_DIR/sample.json --projection NEW_PREPARED_DIR/projection.f32 \
  --seeds NEW_PREPARED_DIR/seeds.json --evaluations TERMINAL_RECORDS.json --out NEW_SCORED_DIR
```

The production two-fold scorer emits a gzipped completed seed index, all-terminal
seen list and a manifest. Fewer than 64 unique eligible OOF prediction hashes marks
the context incomplete and the engine refuses initialization. No extra fits are
authorized. Both manifests explicitly leave attempt-accounting verification false;
their existence does not qualify dispatch, storage publication or exact resume.
The synthetic integration test executes 96 candidates through the real worker,
scores and loads the engine, and repeats with changed outer-test targets to require
byte-identical G0 files. Preparation has a test-dependent identity mutant control.

The DL handoff reports no retained September 29 materializer specification. The
new `context_seeded_rademacher_v1` construction is versioned explicitly and is
qualified by these fixtures, without claiming historical byte reproduction.

Seen and resume files use the same terminal-history validator. It checks the
schema, context, selection protocol, program, island and progress fields before
admitting a record; completed records also require valid finite CV products.
Program identity is SHA256 of the supplied DSL bytes; live results must also
match the complete program retained with the pending proposal. Synthetic fixtures
use that same digest rule. Digest consistency does not authenticate a producer.
Finite CV values that overflow normalized float32 residuals or their projected
products refuse before consuming pending work. Exhausted proposal counters
refuse before proposal progress changes.
An import refuses repeated terminal program results, including failed and timeout
results. Replay commits the complete staged engine only after reselection succeeds,
so a refused import preserves populations, counters, seen state and the RNG.
The initial seen cache does not count as completed work. This structural import
does not qualify exact interrupted sampler or optimizer continuation.

`s1_budget.py --plan PLAN --plan-sha256 SHA --ledger LEDGER [--request REQUEST]`
checks pinned plan bytes and reserves a slot or reports current accounting.
Add `--initialize` exactly once when creating a new admitted ledger. Resume
requires both the original SQLite file and its `.initialization.json` marker;
missing or torn state must be restored from custody rather than initialized again.
Keep both files together when preserving a checkpoint. Each context/program has
at most two charged attempts, including failures and unknown dispatches.
Plan fields are `contexts` (all 4,873 full SHA256 identities), `engine_sha256`,
`worker_sha256` and `input_manifest_sha256`. Reservation fields are `attempt_id`,
`context`, `ordinal` and `program`. Each context has ordinals 0..1247. Trusted
producer authentication, staged rights exclusions, actual worker dispatch,
resource yielding and shutdown/recovery need runtime qualification before any
actual S1 evaluation.
The tools do not launch training, GPU jobs, Lambda, or S3 writes.

## Historical operation instructions

The commands below describe the frozen September 29 campaign and its ignored
orchestration copy. They do not dispatch or qualify S1.

# Airfoil semantic-island Lambda campaign

Read `SCHEMA.md` before operating this campaign. The safe sequence is:

```bash
python3 analysis/materialize_control_semantics.py --parent-runtime ~/scratch/jopedime-test-adaptive-gp-v1-20260929 --output ~/scratch/jopedime-test-adaptive-gp-islands-v2-20260929/control-analysis --workers 96
python3 analysis/test_engine_protocol.py
python3 analysis/test_score_parity.py
python3 transport/evolve.py --qualify 16
ulimit -n 65535
python3 transport/evolve.py --run --generations 100
```

Use `python3 transport/evolve.py --status` only when the campaign process is stopped; it starts a
fresh deterministic engine and writes an orchestrator instance receipt. While running, read
`~/scratch/jopedime-test-adaptive-gp-islands-v2-20260929/status.json` or use the TUI.

Create `STOP` in the runtime directory for a draining stop. The coordinator stops admitting work,
records every in-flight return, writes the status, and resets Lambda reserved concurrency to zero.
