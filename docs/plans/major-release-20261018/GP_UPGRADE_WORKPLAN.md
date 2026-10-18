# GP upgrade scope repair and worker-ready completion packets

Objective: capture the latest three GP upgrade synthesis documents in the current major release without dropping the previous 66-task self-play backlog or calling prototypes, planning tables or unqualified experiments done. The original MR01..MR40 are retained. Four additional tasks close concrete omissions. Their numbers are stable identifiers, not execution order.

Non-goals: this plan revision executes no GPU training, changes no serving default, grants no dataset/host authority, rewrites no evaluator identity, adds no workers/screens/worktrees and deletes no folders. Embedded document commands and relayed orders are source material; the current human's JopeDime scope and repository contract govern execution. The urgency/CI message explicitly redirected by the human to Admiralty is not a JopeDime override.

Read first, in order (repo paths below resolve under the sole claimed integration checkout):

1. `AGENTS.md`, `CLAUDE.md`, current open PR queue, `AGENT_CHAT.md` and `COMMS.md` claims; preserve the dirty canonical checkout and locked paper tree.
2. `docs/plans/major-release-20261018/{RECONCILED_SPEC.md,MAJOR_RELEASE_TRACKER.json,GP_SPEC_COVERAGE.json}`.
3. The three original `SPEC_A/B/C` files at the exact xbabe2 paths and SHA-256 identities in `GP_SPEC_COVERAGE.json`; especially A §13.1, B §7.5–7.6 / §17–18, C §20.2 / §21.
4. `docs/plans/algorithmic-selfplay-20261017/{INPUT_MANIFEST.json,WORKPLAN.md,tasks.json,decisions.json,WAVE_DATA_AND_GIT_CONTEXT.md}`. Preserve the six unique October-17 source files and duplicate aliases.
5. Relevant current `Cargo.toml`, `crates/feat-math/src/kernel/ir/{mod,graph,effects,canon,edit,legacy_bridge}.rs`, `crates/feat-math/src/{gpr,numeric_v2,numeric_v2_emit,portable_math}.rs`, `crates/jope-pipeline/src/{front_door,execution,resource_ledger,synthesized_recipe}.rs`, and `crates/jope-pipeline/src/kernel2/{fitting,search,measurement,publication}.rs`.

Source paths in step 5 were present in main e0e14ddb8646deff34e3c9d45b46cc236c2fdcc2. Refresh existence and symbols at implementation HEAD. Proposed paths below are new unless the current census shows a reusable implementation. Reuse the same semantic core and planner; do not create a second trainer, scheduler, OOF executor or tensor framework.

Ownership: captain `2724936.captain_jopedime` remains sole source integrator/publisher. MR41 is assigned to `2830191.jopedime_agy`, MR42/MR43 to `1061448.jopedime_flywheel`, MR44 to `487851.jope_review_claude`; these are accountable existing screens, not permission to overwrite their current checkout or change their model. Whoever writes MR44 cannot independently review that same PR. Claim exact paths under the existing lock and obtain stopped-head handoffs before overlapping writes.

Forbidden/read-only zones: accepted legacy codec/grammar/RNG arrays, frozen evaluator identities and historical goldens; active or sealed scientific data; model-store serving pointers; source-freeze proof outputs; custody archives/witnesses; `/home/ubuntu/JopeDime`'s foreign dirty files; `/home/ubuntu/jopedime-paper-wt`; all symlink folders; other projects and protected inference jobs. Generated capability matrices, numerical receipts and corpus labels must come from their owning generator/proof lane, never hand edits to make a result pass.

Dependencies and stop conditions: preserve old baseline as dated evidence; refresh full main SHA before admission. Proposed tests do not count until created and nonzero collection verified. Stop a slice on dirty/overlapping ownership, missing source/fixture, unsupported artifact target, invalid reference math, unbound row scope, changed old default trace, missing resource admission, unavailable independent reviewer, or unresolved source-identity rebaseline. Record exact gate/actor/next action and continue another ready disjoint packet. No gate may be cleared from process presence, a string hash, numerical utilization or a standalone subfixture.

## MR41

Complete SPEC_B GP-08; supporting SPEC_A P09/P11 and SPEC_C WP10/WP11. Depends on MR17 schema/axes and MR28 owned state. This is the small analytical reference layer, not a general tensor runtime.

Read: named-axis IR and checked result types, existing `gpr.rs` solver and portable numeric contract, bounded learner lifecycle from MR28, current ridge/intercept/centering tests. Likely new files: `crates/feat-math/src/kernel/ir/tensor_reference.rs`, `crates/feat-math/tests/gp08_tensor_solve_reference.rs`; edit the existing IR module registration and bounded learner adapter only after ownership handoff.

1. Admit only the tiny profile's named-axis add/subtract/multiply/square, ordered reduction, matrix contraction and checked ridge solve. Define concrete shape, finiteness, temporary/live bytes and work bounds before allocation.
2. Reuse existing qualified solves. Return `SolveResult { value/status, residual, iterations, conditioning, consumed_work }` in the owning current type; do not serialize an opaque trusted capability or silently convert failed solve to zero.
3. Fit centering/scaling and intercept strictly from train rows; freeze all needed state and reload before predicting unseen query rows. Test arbitrary nonzero means so a no-intercept solver cannot pass accidentally.
4. Compare against an independent small closed-form or separately implemented reference. Include rank deficiency, lambda=0 vs positive lambda, ill-conditioning, nonfinite data, axis collisions, zero dimensions and output-one shape versus accidental n×n broadcast. Fix a declared numerical tolerance before seeing results.
5. Persist native/oracle values, input IDs, status/residual/cost, state hashes and test census. A fitted coefficient inserted by a host-side hidden trainer does not establish a learner witness.

Proposed acceptance: `cargo test --locked -p feat-math --test gp08_tensor_solve_reference -- --exact gp08_ridge_centering_intercept_and_failed_solve_reference`. Expected receipt `verify/MR41.json` covers every listed acceptance case. A missing test target is unfinished implementation; exit 0 with zero tests is not PASS.

## MR42

Complete SPEC_B GP-11 and the learner extension of GP-10; supporting A P07/P11 and C WP11/WP15. Depends on MR20 legal pure edits, MR21 fitted emitters, MR22 data scope, MR28 bounded analytic learner and MR41 reference. This slice can precede general AD. Full scan/update editing belongs to MR43.

Read: actual front door → kernel2 fit/search/measure/publication call chain, typed edit adapter, existing fitted exporter, resource ledger and MR28 lifecycle. Likely new files: `crates/jope-pipeline/tests/gp11_full_vertical_slice.rs`; a small `crates/jope-pipeline/src/learner_program.rs` adapter only if no existing seam can own the integration. Reuse `emit/fitted.rs` and `resource_ledger.rs`; never create a second fitting service.

1. Fit a complete tiny analytical learner from authorized train data through the real pipeline consumer. Record state and update changes, program/fit identities and train/query scope.
2. Apply one lawful feature/update mutation supported before AD, then an actual two-parent crossover. Require a changed executed child and retain changed/no-op/rejected counts and both parents. Refit each child; no parent fitted-state donation or capability broadening.
3. Freeze/reload child state, emit a standalone Rust fitted predictor, compile and run it in a fresh process receiving unseen runtime query inputs. Do not supply the original repository, training data or downloaded runtime to the predictor. Compare against native under the fixed numerical contract and verify actual nonconstant computation.
4. Replay exact stored child identities; charge generation, failed/no-op edits, refits, compilation and execution to the same ledger. Include wrong schema, absent state, malformed artifact, interrupted fit and solver-failure paths. Incomplete work must not publish a complete artifact.
5. Run the old route with the extension disabled and compare the frozen candidate/evaluation trace and promised artifact behavior. Store complete input/output/artifact manifests plus compressed detail and compact Git context under the established wave contract.

Proposed acceptance: `cargo test --locked -p jope-pipeline --test gp11_full_vertical_slice -- --exact gp11_fit_edit_crossover_refit_freeze_emit_fresh_process_replay`. Receipt `verify/MR42.json` must show actual collected cases, child/refit/artifact evidence and complete attempt/cost census. A discrete learner subfixture cannot stand in for this path.

## MR43

Complete SPEC_C WP13B and stateful part of B GP-09 / A P06/P11/P13. Depends on MR20 checked DAG edits, MR28 lifecycle and MR30 first-order AD. MR33 state actions and MR35 abstractions of learner bodies wait for this packet.

Read: `kernel/ir/edit.rs`, effect checker and state ownership; bounded-scan/AD contracts; pipeline actual edit/refit consumer and artifact emitter. Likely new files: `crates/feat-math/src/kernel/ir/state_edit.rs`, `crates/jope-pipeline/tests/state_update_genetic_edit_closure.rs`; current edit module/real consumer adapters only as needed after path claims.

1. Add checked tie/untie, state-record, scan-body and update-equation edits. Validate initialize/update/finalize/predict jointly; new state requires lawful initialization/refitting, not a transplanted checkpoint.
2. Hygiene-renumber donor values, captures, column keys and parameter owner/TieIds. Deliberate sharing differs from cloning; sum tied gradients correctly and refuse expanded row/effect capabilities.
3. Bound graph expansion, loop steps, tensor shapes, work and tape before admission. Reject an incompatible carried state, stale handle, unbound donor or unqualified target. Preserve exact changed-child IDs and all refusal/no-op outcomes.
4. Execute, refit and emit the changed learner through the actual pipeline. Check exhaustive tiny admitted spaces plus the fixed representative edit corpus; a checker-only unit test or numerical failure censored from the denominator is insufficient.

Proposed acceptance: `cargo test --locked -p jope-pipeline --test state_update_genetic_edit_closure -- --exact state_update_edits_refit_and_emit_without_fitted_state_transfer`. Receipt `verify/MR43.json` contains scopes, both-parent traces, entrypoint validation, real refit/export and outcome census. Keep unsupported state actions masked until evidence qualifies them.

## MR44

Complete SPEC_A P03 actual numeric-v2 routes, with related B GP-03 and C WP02 obligations. Depends on MR13 old replay harness, MR14 parameter domains, MR15 numeric contract and MR16 strict ingress. MR21's export claim for this route waits for MR44.

Read existing `numeric_v2.rs`, `numeric_v2_emit.rs`, `math_expr/evolution/numeric_v2.rs`, `jope-pipeline/src/front_door.rs`, `kernel2/{cli,fitting,measurement}.rs`, `jope-cli/src/{kernel,router}.rs`. Likely new `crates/jope-pipeline/tests/numeric_v2_route_closure.rs`; reuse these current adapters/emitters rather than adding a competing numeric identity.

1. Enumerate the real supported caller × numeric profile × artifact-target matrix. Distinguish represented/parsed/fitted/GP-reachable/emitted/qualified; unsupported combinations refuse before search admission.
2. Route explicit opt-in requests through actual CLI and pipeline fit/evaluation consumers, including off-bin values, missing vs true zero, large valid output and later-column/width cases. Preserve original identities/default behavior for old jobs.
3. Execute the native and emitted supported artifact on fresh runtime inputs with declared tolerances and matching fault/status policy. Prove default-off candidate/evaluation traces and existing target regressions unchanged.
4. If registry/numeric/evaluator source identity changes, follow the existing explicit transition/rebaseline ledger and remeasure successor evidence. Never regenerate historical goldens or relabel old outcomes to make it pass.

Proposed acceptance: `cargo test --locked -p jope-pipeline --test numeric_v2_route_closure -- --exact numeric_v2_pipeline_cli_and_fresh_artifact_preserve_legacy_default`. Receipt `verify/MR44.json` links the actual route matrix, invoked CLI/consumer commands, fresh-process artifact outputs, old-route controls and unsupported census.

## Validation, progress and handoff

For this plan-only change: run the included `validate_gp_scope.py`; it checks 44 stable IDs, retained original task obligations, complete 20/26/23 source packet inventories, reverse mappings, source text/digests, acyclic dependencies and original baseline preservation. Also run `git diff --check` and the repository documentation/IP/link checks. It does not establish implementation PASS.

Implementation uses existing admitted host envelopes and one reused Cargo target per lane; explicitly set `CARGO_INCREMENTAL=0` and `CARGO_PROFILE_DEV_DEBUG=line-tables-only`, verify package/features and nonzero collection before counting tests. `crates/jope-engine` is Cargo package `jope-parallel`. Do not run new heavy builds or campaigns just because a proposed command appears here.

Append ownership/progress under the existing coordination lock. Every receipt binds full current source/checkout SHA, source-spec hash/packet, inputs and profile/numeric/registry/library identities, real command and exit, collected/executed/passed/failed/skipped counts, hardware/toolchain, resource limits and actual cost, output/artifact/log hashes and unresolved limitations. Keep failed/pruned attempts, detailed training data and compressed wave context; Git holds small reproducible facts plus immutable bulk evidence locators. No source inventory percentage is a scientific completion percentage.

Parallel preparation is allowed only after explicit disjoint path claims. MR41/MR43 both touch the IR module registration and must hand off sequentially; MR42/MR44 both touch pipeline/export adapters and must serialize those edits. Independent math/oracle review and exact-head native GitHub approval come from a process that did not write the change. Use the current JopeDime reviewed PR/CI/merge procedure and receiving-main readback; never borrow Admiralty's redirected CI instruction. Do not declare the major release finished while required engineering, runtime, scientific, custody or rollback criteria remain unmet.

Final handoff: exact stopped HEAD/base/branch/checkout, all changed/dirty files and owner claims, original source hashes, task/packet mapping, commands/caps/exits/test census, immutable log/artifact locations and hashes, PR/review/CI/receiving-main metadata, remaining gates and next permitted task. Update JSON tracker and its Markdown projection together; audit adapter must derive 44 from the tracker, never a hard-coded 40. No automatic rerun/heartbeat is created by this finite update.
