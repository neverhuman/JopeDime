# H75-closed-loop runner

This directory holds the runner for the [preregistration](../PREREG.md); [protocol.json](../protocol.json)
is the authority for every dimension. OPUS owns this runner. The scoring helpers live in `scoring/` (GOV03)
and the feedback helpers in `feedback/` (OBJ02). This runner does **not** apply the seven-contrast decision.
**No experiment has run from this source.**

## Stages

```text
python3 -I h75_runner.py plan  --out OUT --run-id ID                       # pins + rosters, 0 native calls
python3 -I h75_runner.py shard --out OUT --run-id ID --fold K --seed S \
        --device cuda --workers N [--anchor-dsl EXPORT] [--phases acquisition,legacy,prospective]
```

One `shard` call covers one fold and one seed, and resumes after an interruption.

1. **Acquisition.** Cells A/B/C/D run two cycles each. A cycle is:
   1. Propose 8 programs.
   2. Pick 2 by DL ranking plus 2 uniform from the rest, using a random stream shared across cells.
   3. Evaluate the 4 picks natively on the original cut.
   4. B/D admit the outcomes as same-context memory; A/C only log them.
   5. C/D take 40 persistent AdamW steps. The cycle-1 optimizer state carries into cycle 2, with no rollback.
   After both cycles, the replay learners replay_B and replay_D are trained from r882 on stratum-matched logs.
   All learners are then frozen and their state hashes recorded.
2. **Legacy panel.** Runs only after step 1 is frozen. Every learner is scored on all held contexts of the fold:
   original static pool, the four hash-ordered anchors, full-pool raw argmin. A and B must agree exactly.
3. **Prospective panel.** Runs only after step 1 is frozen. A/B/C/D/G/R each search the 4 held prospective
   contexts for two cycles of 4 evaluations, reporting U4 after cycle 1 and U8 after cycle 2. Every final
   selection, and the initial best anchor, then gets a common confirmation on resplit seeds 801–803.

The campaign ledger (`OUT/campaign_ledger.json`) is shared by all shards under an `fcntl` lock and enforces:

| Limit | Value |
|---|---|
| Native calls | 9,600 (7,200 planned) |
| Native child CPU | 4 h |
| GPU | 4 h |
| Deadline | 2026-10-31T14:00:00Z |

The fit cache makes resume free of re-charges. With `--device cuda`, `/home/ubuntu/jope_plan/qualification.lock`
is held from before model construction to process exit.

## Fixed interpretations (decided before any outcome)

- **History.** Every policy in cycle 1, and A/C throughout, uses the four hash-ordered pool anchors
  (positions 0..3) as revealed history and the context champion as the proposer parent. In cycle 2, B/D/G/R
  use the best four completed programs among {anchors} ∪ {their own completed evaluations}, by utility
  descending with program-SHA tie-break, as both history and parents. The rest of the static pool is never
  revealed.
- **Proposer bank.** The champion-retrieval bank excludes every family in the fold's held set.
- **Apparent HWM.** A result is flagged when it is completed, passes the gate, and has fitness below the best
  passing fitness over the static pool ∪ the cell's completed fresh rows by more than 1e-6. This is an
  original-cut flag, not confirmation.
- **D's training slates.** Up to 48 unrevealed slots hold hash-ordered completed fresh rows; C fills them
  from its static pool. Other slots are outcome-blind uniform draws from the static pool. Anchors are always
  slots 0..3, and the loss masks them. Failed or timed-out rows are never labels.
- **Trainable parameters.** C/D train adapter tensors except `anchor.*` and the unused `parent_delta_net.*`.
- **Search controls.** G is the fixed program412 HistGBM from the prereg, which picks 2 by predicted utility
  plus 2 uniform. R picks 4 uniformly.
- **Device.** Neural training and inference are only cost-feasible on GPU; on CPU they would exhaust the 4 CPU-h
  cap. CPU mode is for tests, the pilot and smoke runs.

## Execution prerequisites (blocking)

- **Anchor DSLs.** The admitted pools carry no DSL text for the four anchors. A COMPLETE confirmation panel
  needs an admitted training-only export `{context_sha256: [{program_sha256, dsl}]}` (`--anchor-dsl`). The
  runner verifies `sha256(dsl) == program_sha256` and anchor membership. Without the export, affected
  confirmed-U8 rows are `status=unavailable` and the panel is INCOMPLETE.
- **Execution manifest.** Before any fit, freeze a manifest with the merged source SHA, runner and asset hashes,
  input object keys and versions, rosters (`plan.json`), host shards and the ledger. Pass its hash in the
  `H75_SOURCE_GIT_SHA` and `H75_EXECUTION_MANIFEST_SHA256` environment variables.
- **GPU lock.** GPU execution waits for the owner queue (H57 → H75) under the qualification lock.

## Tests

```text
python3 -m pytest -q test_h75_runner.py      # structural fixtures only; no data, model or native calls
```

The tests cover:

- roster rule and determinism
- selection rule and inclusion probabilities
- history ordering
- replay matching
- slate composition
- loss masking of revealed slots
- envelope resplits
- campaign-ledger caps, deadline and cross-process state
- fit-cache no-recharge
- fitness formula
- anchor-DSL identity checks

## Software smoke (2026-10-31, before this PR) — disclosure

The smoke run used real inputs and the real model on CPU, with fold 0 and **pilot seed 700** (not an
experiment seed). **Native fits were mocked**: deterministic synthetic outcomes with no evaluator call and no
real labels for new programs. All three phases ran, and the confirmed-U8 panel had 6 of 24 rows
`unavailable` (missing anchor DSLs, as expected). Two consequences have to be disclosed.

1. **Held-family exposure.** The smoke's legacy panel scored real held-family pool labels for fold 0. The C
   learner's training does not depend on the mocked fits, so this shows one real held-fold development
   number: seed-700 C scored a family-pooled raw utility of 1.229, against 1.289 for r882 (A/B). The D and
   replay numbers are mock-contaminated and meaningless. This peek happened while designing the software.
   It must not be used to change the preregistered recipe, and it is not an H75 result. Later smokes run
   fit-only phases (`--phases acquisition`).
2. **Structural design flag.** In cycle 1 all four cells choose the identical 16 programs. That is by
   design: every cell starts from r882 with the same anchors and the same random stream. In the smoke, D's
   40 updates did not change its top-2 picks in cycle 2 either, so B and D acquired identical data and
   `replay_D − replay_B` was exactly 0. The learner-to-data channel carries only 2 DL-chosen evaluations ×
   4 contexts × 1 later cycle. With small updates, contrast 3 may be structurally zero. This is a power and
   design question for the preregistration owner; this runner does not change it.

## Review dispositions (BLOCKERS-PRELIMINARY at f96e9b58)

1. **Confirmation status.** `confirmation_status()` makes a row `completed` only when all three cuts completed;
   a completed gate failure is a measured −2. A failed or timed-out cut makes the row `failed`/`timeout` with
   null utility, and a missing evaluation (no admitted anchor DSL) makes it `unknown`. `score_row()` refuses a
   completed row without a finite utility and any status outside completed/failed/timeout/unknown.
2. **Scorer contract.** Rows carry `confirmation_cut_sha256s`: the sha256 of each cut's exact evaluator input
   bytes. The new `roster` stage writes `admission_legacy.json` and `admission_confirmed_U8.json` in the #846
   admission schema. A test feeds runner-built rows through `scoring/score_panel.py`: complete rows give
   `COMPLETE_POINT_DIAGNOSTICS`, and one timeout gives `INCOMPLETE`.
3. **Ledger.**
   - `reserve(n)` checks the deadline and every cap and charges the calls in one locked transaction before
     launch. `settle()` then records each completion as it happens, so interrupted calls stay charged.
   - `neural()` charges runner-process CPU and synchronized GPU time on every training step and every inference
     call, and re-checks the caps and deadline each time. The CPU cap covers native child CPU plus runner CPU.
4. **Fit cache.**
   - Keys bind the execution scope (run, execution manifest, source commit, fold, seed, worker, protocol), the
     exact input-byte sha256 and the program. Rows from another scope are refused, and an input-byte change
     between planning and execution aborts the run.
   - Each completion is appended and fsynced immediately. A later re-execution of an interrupted call is
     charged again, because it is new work.
5. Withdrawn by the reviewers (dropout); no change.
6. **Resume.** After each cycle, an atomic checkpoint stores the C/D model and optimizer states, the cell
   memories, the records and the training receipts, bound to the shard scope. Resume continues from the last
   completed cycle and never resets a persistent learner to r882. Random streams are derived from the
   fold/seed/cycle/step, so re-running a cycle is deterministic. A shard directory from another scope is refused.
7. **Closure before loading.** `validate_closure()` runs before any payload load, model build or GPU allocation.
   It requires:
   - a 40-hex `H75_SOURCE_GIT_SHA`;
   - an execution manifest whose bytes match `H75_EXECUTION_MANIFEST_SHA256` and which binds this runner's bytes,
     PREREG.md and protocol.json;
   - an authorized fold and seed.
   Pilot seed 700 needs a pilot manifest and fit-only phases. GPU needs explicit `gpu_authorized`; H75 has no
   GPU slot until one is granted.
