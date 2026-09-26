# W5 — selection replay, lineage, fit/score identities and feasibility

Branch `rr/w5-selection` (base `de747c363`, the `feat/regression-remediation` head at
assignment). Owner: Claude implementation subagent, lane `.agent/worktrees/demo`.
Observed states: **reproduced** (where marked), **implemented**, **fixture-proven**
locally through the `cfg(test)` fit harness (real authority service, service-owned
workers, CLI-shaped clients, reviewed synthetic issuer). Nothing here is `CI-verified`,
`live-qualified`, a production activation, or a closure of any register entry; the
findings stay open with their original hashes. Production fitting remains fenced:
no authority, witness, service, native SCQ, fit-boundary or `main.rs` product code
changed, and no test-only activation reached a product path.

Receipts (full source SHA, diff, nextest logs) are retained outside the tree under
`/home/ubuntu/jope-rescue-20260921/regression-remediation-20260926/claude-w0-w12/runs/`.

## W5-T06 — independent selector and stop replay (commit `90c3ab336`)

- **New capability.** `gp::selection_replay` (+ read-only binary
  `jope-gp-selection-replay --root R --output O`) re-derives the measured winner, the
  complete distinct ranking and the terminal stop reason from the durable ledger,
  under the frozen contract `parallel.gp_selection_contract.v1`, without calling the
  searcher's ranking. It reads `search-request.json`, `observations.jsonl`,
  `observations.json`, `progress.json`, `search.json` and each measured row's original
  completion, fit receipt, model and output.
- **Contract.** Objective `metrics.rmse × target_scale` (scale from the request-bound
  cut preprocessing), minimised. A row is measured only with a null original failure,
  one complete finite non-negative measurement of the frozen fold and verified
  completion evidence. Identical predictions form one entry (earliest position);
  ties resolve by IEEE total order then earliest position; an inherited continuation
  incumbent occupies position -1. Stop: never beyond `evaluations`; no dispatch after a
  row completed at or past `seconds`; every post-row proposal refused; the reason is
  exactly the evaluation budget, the deadline, or one slot's 10,000-proposal cap.
  Predicted, proposal, reward and rank scores in any field are never read.
- **Defect reproduced and fixed.** A resumed search adopted journaled rows without
  the live admission rule: a journaled RMSE rewritten to `0.0` was published as
  `best_validation_rmse` for `(pipe X (ridge))` (`runs/w5-t06-resume-repro-20260926T055746Z`).
  Resume now applies the finite/non-negative rule and requires each current-protocol
  row to equal the output its original completion attests, before any dispatch.
- **Tests.** Pure adversarial suite (`gp::selection_replay::tests`: nonfinite/NaN
  strings/negative/overflowing/incomplete measurements, failed rows with quality,
  delayed/unobserved outcomes, reordered/duplicated/foreign rows, semantic duplicates
  with different measurements, predicted scores reversed against measured order,
  stop budget/deadline/cap/late dispatch). Real-fit harness: agreement with
  `search.json` on real searches, across every crash/resume boundary and under
  reversed critic weights; tamper suite (swapped rows, forged RMSE, revived failure,
  published winner/score, torn tail, stray late results reported but never ranked).

## W5-T01 — immutable candidate lineage envelope (commit `578b41b64`)

- **Envelope.** Each pre-dispatch attempt records `parallel.gp_candidate_lineage.v1`
  (original proposal; source: seed slot with the critic weight marked as estimating
  the original only, fixed reference, generic bootstrap generator draw, native
  candidate, or variation event with operator, parent ledger positions and stream
  words; at most one explicit adaptation; executable recipe). It is bound by the
  original completion's attempt digest and repeated in the ledger row.
- **Defect reproduced and fixed (second implicit adaptation).** Generic bootstrap
  programs were projected when listed and projected again at dispatch; the recorded
  original was the adapted program (`runs/w5-t01-bootstrap-repro-20260926T060448Z`).
  The search now dispatches the single adaptation of the original generator draw.
- **Verifier.** `gp::lineage::verify_search` re-derives every original, the single
  projection, the uniform parent draws, parent eligibility and operator outputs, binds
  envelopes to attempts/completions, links the replayed winner to its fold fit and
  model, and `verify_final_refit` links a fitted-view export to exactly that fitted
  identity. Parent swaps, omitted/contradictory variation events, recipe substitution,
  double adaptation, unauthenticated seeds/vocabulary and same-recipe other-state
  refits refuse. Variation children leave every parent's stored model reproducing its
  recorded predictions.

## W5-T03 — fit identity separated from score identity (commit `8d25493c5`)

- **New safeguard.** Every measurement carries `parallel.gp_score_identity.v1`: the
  authority fit key and model (the reusable fit identity) plus the scored rows,
  training rows, preprocessing, task, metric, units, evaluator protocol and
  prediction digest. The replay verifies it against the frozen fold and the row's
  original fit receipt and model.
- **Proof.** Same training rows with a different validation membership reuse the fit
  (no new physical fit; `execution_accounting` reports the reuse) but are rescored on
  their own rows: the stored model reproduces the new predictions and RMSE, and a
  score carried from the other membership is refused.

## W5-T04 — conservative feasibility classification (commit `c83b7b805`)

- **New capability.** `feat_math::feasibility::classify(program, shape, limits)`
  (protocol `feat_math.feasibility.v1`) propagates generated and output width
  intervals through every stage with the operators' own parameter clamps and checked
  arithmetic, and bounds the simultaneously live f64 cells (caller rows, current
  matrix, generated and merged rows, fitted state; the ridge head's Gram matrix).
  Verdicts: `proven_feasible` (everything covered, every upper bound within the
  limit), `proven_outside_limits` (a lower bound or an overflowing lower bound
  exceeds the limit, or the fitter refuses the program/shape), `unknown` with reasons
  (uncovered operator, head or program form, or an upper bound that may exceed).
  Covered resource: materialised cells (memory); compute time is not covered.
  Default limit: 2^28 cells (the evaluator's 2 GiB sealed-view budget).
- **Coverage.** 36 transform operators and the ridge/logistic head. Uncovered:
  other heads (gpr, krr, gbm families, ensembles, stack, portfolio), symbolic and
  explicit-portfolio forms — these return `unknown`, which is never authorisation.
- **Soundness tests.** For every covered operator over widths 1…600 (including the
  512 `MAX_WIDTH` edge) the measured generated and output widths lie inside the
  intervals, and a per-thread counting allocator shows the measured peak bytes of fit
  plus apply within the stated bound (2× vector growth, row headers and scratch
  allowed) and above the generated-rows lower bound. Adversarial dimensions and
  overflow (u64 extremes, 50,000 × 512 × 32 one-hot expansion before truncation)
  refuse without shape-proportional allocation; edge widths (512 columns, one
  column into pair/ratio/fft/poly stages, 32-bin one-hot) are feasible and fit.
- **Search integration (diagnostic only).** Each pre-dispatch attempt records the full
  classification for the cut's declared shape and each ledger row carries the verdict
  and reasons beside its observed outcome, under one rule for every arm. Dispatch is
  unchanged: a refusal policy changes search feasibility policy and therefore needs a
  new protocol identity and owner decision (S05-I5). A real-fit test records
  feasible/refused/unknown estimates beside a successful fit, a deterministic fit
  failure and an uncovered head.

## W5-T02 — frozen selection rules (partial slice; commit adding this file)

- Every new `search-request.json` freezes `selection_contract`
  (`parallel.gp_selection_contract.v1`) and `proposal_cap_per_slot` (10,000, the
  searcher's own loop bound, now one shared constant). The existing request-equality
  check therefore refuses a resume under other objective/evidence/tie/stop rules
  instead of silently restarting, before any new dispatch; the replay refuses a
  request frozen under another contract. Seed, preparation/cut (folds), evaluator
  and random protocol were already frozen and resume-checked.
- Draws, unique programs, completed evaluations, physical fit starts and reused
  completions are already separate published counters (`proposal_attempts`,
  `unique_programs`, `completed_evaluations`, `execution_accounting`).
- **Paired-arm manifest (branch `rr/w5-selection-2`, new capability).**
  `gp::paired::compare(root, arms, declared)` (`parallel.gp_paired_comparison.v1`)
  replays every arm independently, requires the arms' frozen search requests to be
  identical except for the declared arm difference (which must actually differ),
  requires one feasibility rule across arms, and refuses a program whose
  measurement differs between arms. Per arm it records the stop reason, replayed
  winner, non-overlapping counters (proposal draws, rejected proposals, unique
  programs, completed, measured, failed and semantically duplicate evaluations,
  recorded physical fit starts, reused completions, unknown start outcomes) and a
  generation-zero inventory (initial slots evaluated, by origin, and slots refilled
  by variation); shared programs whose outcome diverged (for example a timeout in
  one arm only) are listed, not averaged away. Test
  `a_paired_comparison_names_its_sole_arm_difference_and_refuses_any_other`:
  swapping only origin labels leaves every shared measurement, winner and budget
  counter identical (the second arm's fits are recorded as reuse, not new starts);
  undeclared, overdeclared, empty and duplicate-label comparisons refuse.
- Not done: order-balanced paired receipts from real runs (W10).
- The same commit hardens the replay's proposal-accounting check against an
  adversarial rejection counter at `u64::MAX` (saturating arithmetic, no overflow).

## Verification summary

Per commit, retained under `runs/w5-*`: `-p jope-parallel --lib -E 'test(/gp::/)'`
(31 → 41 tests, all passing at each step) and `--test view_contracts --test
gp_contracts --test fit_boundary` (41/41 at each step); `-p feat-math --lib -E
'test(/feasibility/)'` 4/4. A full `-p feat-math` run after W5-T04 passed 434/435;
`golden::fast_profile_portfolio_head` hit nextest's 360 s limit while other agents'
test suites loaded the host (that integration binary does not contain the new
`cfg(test)` allocator or use the new module).

## W5-T05 — reservation and supervised termination (branch `rr/w5-selection-2`)

Base `64e56b410` (the integrated `feat/regression-remediation` head). Fixture
faults are injected by the `cfg(test)` harness only (`JOPE_FIXTURE_EVALUATOR_FAULT`
on a search client, inherited by its evaluator children): `block`, `fork_block`,
`fork_exit`, `oom`.

- **Defects reproduced and fixed** (`runs/w5-t05-supervision-repro-20260926T064427Z`,
  three failing reproductions):
  1. *Dispatch race.* A second searcher on the same output, started while the first
     had reserved and dispatched candidate 0, recorded that reservation as an
     unknown outcome, journaled it and published `search.json`; the first then
     appended a second row 0 and overwrote the publication. The search now takes an
     exclusive, crash-released `flock` lease on `search.lease` in the output before
     reading any resume state; a concurrent searcher is refused before any
     dispatch, and a completed search stays closed.
  2. *Blocked fitter with helpers.* A timed-out evaluator was killed alone; a
     process it had started kept running after the search returned. Evaluators
     now lead their own process group; on expiry the whole group is killed.
  3. *Helpers after a normal exit.* Processes left by a successful evaluator
     outlived it. The controller now observes the exit without reaping (`waitid`
     with `WNOWAIT`, so the group id cannot be reused), kills the group, reaps,
     and records a census.
- **Terminal disposition.** `search.json` records `stop_reason`
  (`evaluation_budget`, `search_deadline`, `proposal_space_exhausted`); the
  independent replay refuses a publication whose stop reason differs from its own
  stop decision. Each completion records
  `process_group: {isolated, members_at_exit, members_after_drain}`.
- **Safeguard demonstrated (no defect):** an evaluator exceeding its 8 GiB
  address-space limit aborts into a contained `worker_exit` failure row, never a
  measurement; with no measured candidate the search publishes a typed no-result.
- **Tests:** `a_blocked_fitter_and_its_helpers_are_killed_at_the_original_deadline`,
  `helpers_left_by_a_finished_evaluator_are_drained_before_the_next_dispatch`,
  `a_second_searcher_cannot_dispatch_into_a_live_search`,
  `an_evaluator_allocation_failure_is_a_contained_failed_row` (real service fits,
  `/proc` no-orphan census).
- **Residual risk:** if the controller itself is killed, `PR_SET_PDEATHSIG`
  stops the evaluator but not processes it started; the product evaluator does not
  start any today. Service-side worker custody is the authority service's
  (unchanged). Distributed reservation is W7.

## W5-T07 — sealed selection, final refit and export handoff

- **Defect reproduced and fixed** (`runs/w5-t07-handoff-repro-20260926T065444Z`):
  the cycle pipeline's predictor handoff exported `search.json`'s `best_program`
  as the selected predictor without replaying the selection. With the summary
  forged to name another evaluated program, it exported `(pipe X (ridge))` instead
  of the replayed winner. The handoff now lives in `gp::handoff::export_selected`
  (not feature-gated, so the harness can run it; the `neural`-gated pipeline only
  calls it) and exports only the sealed selection.
- **Seal** (`parallel.gp_selection_seal.v1`, `*.seal.json` beside the predictor):
  status (`selected`, `inherited_incumbent`, `no_measured_candidate`), the replayed
  winner's executable recipe and original, source, ancestry and objective, its
  fold fit completion and model, the replay's stop reason, and the authorised
  final-fit scope `selected_cut_training_rows` (training rows only; the cut's
  validation labels and the original final test are recorded as outside every
  fit). A recorded seal is valid only while it equals the seal recomputed from the
  search's evidence (independent replay plus verified lineage).
- **Handoff** (`parallel.gp_export_handoff.v1`, `*.handoff.json`): fits exactly the
  sealed recipe on exactly that scope, requires the export's fitted identity to be
  the selected fold fit (`lineage::verify_final_refit`), and records the result.
  A failed or mismatched export records `export_failed`, keeps the sealed recipe
  and fold model as the selected fitted state and exports nothing else; recorded
  outcomes are terminal and re-verified on resume.
- **Tests:** `the_predictor_handoff_exports_only_the_sealed_replayed_selection`
  (forged summary refused, honest path exports the winner on its training rows,
  resume reuses the recorded handoff) and
  `a_sealed_selection_refuses_swaps_and_keeps_the_selection_when_export_fails`
  (another candidate's fold model or fit completion, another recipe, another
  scope, held-out labels in the fit, another status; a missing lineage event;
  a fixture export failure keeps the selection and attempts no other program).
- Serving packages (Python/Rust/C++) and their unsupported-export statuses remain
  W9; the fitted-view export is the only export this handoff performs.

## Open items and risks

- Production activation verifiers remain unimplemented and require an owner/security
  decision; nothing here enables production fitting.
- `docs/internal/core-ip/tools/verify_test_inventory.py` already failed on the base
  (three unclassified W4 targets); the new `jope-gp-selection-replay` binary adds a
  fourth entry for the inventory owner.
- The searcher's internal parent pool keeps an inherited incumbent outside semantic
  de-duplication; the published winner is unaffected and the lineage verifier mirrors
  it. W5-T02/T05/T07 remain largely planned (see the task file).
- Scope limits: the replay refuses legacy ledgers without the durable search clock,
  and the lineage verifier refuses rows without envelopes (searches started before
  these commits); the lineage verifier takes the seed population and vocabulary as
  explicit inputs, authenticated only by the request digests. The native
  continuation paths (inherited incumbent, `continuation_mutate`) are covered by pure
  tests only; the harness cannot yet run a native genome task.
- Format additions: ledger rows now carry `lineage`, `feasibility` and
  `metrics.score_identity`; attempts carry `lineage` and `feasibility`; requests
  carry `selection_contract` and `proposal_cap_per_slot`. Generic bootstrap rows now
  record their true generator draw in `proposal_before_projection`. Consumers read
  these files as JSON values; in-flight searches from older binaries were already
  unresumable because the evaluator digest is part of the request.
- The feat-math `cfg(test)` counting allocator applies to the whole feat-math
  lib-test binary (accounting only; the full `-p feat-math` run is retained).
