# S1 teacher engine and fixed-budget conformance

Read [SCHEMA.md](SCHEMA.md) for the current S1 protocol and the retained historical
record. The owner selected new search with inner-CV-only selection, 96 G0 slots
and 18 equal generations of 64 attempts per context. Codex root owns the core
landing; Claude dltrain-impl-v1 retains input preparation and its pilot.

Run CPU conformance from the repository root, reusing its target:

```sh
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --offline --locked --manifest-path deploy/lambda/training/teacher-v2/rust/lambda-worker/Cargo.toml --target-dir target
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --offline --locked --manifest-path deploy/lambda/training/teacher-v2/rust/island-engine/Cargo.toml --target-dir target
python3 deploy/lambda/training/teacher-v2/test_s1_budget.py
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
are untouched. Hosted hygiene runs the budget tests, both standalone Rust
workspaces through the existing Lambda lane, and the mutation qualifier.

Seen and resume files use the same terminal-history validator. It checks the
schema, context, selection protocol, program, island and progress fields before
admitting a record; completed records also require valid finite CV products.
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
