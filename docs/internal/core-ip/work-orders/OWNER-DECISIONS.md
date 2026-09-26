# Owner decisions: the core-IP remediation's pending policy choices

**Decision of 2026-09-26.** The owner delegated every pending policy choice the core-IP remediation surfaced and asked for the conservative, fail-closed option in each case, recorded so that it can be reversed. Every entry below is therefore **adopted by owner delegation 2026-09-26**. None is an agent's own policy call.

**Scope.**
- D1–D16 are the sixteen decisions listed in the body of PR #111 ("Owner decisions surfaced (not decided here)"), with the same numbers.
- D17–D19 surfaced after PR #111 merged. The W7 `machine_id` decision was raised in both places and is recorded once, as D16.

**How to read an entry.** Each entry gives:
- the question;
- the options;
- the adopted option;
- whether behaviour changed;
- where it is enforced (file, function, test);
- how to reverse it.

"On main" means `origin/main` at `8a55c4fef`. Entries that name `rr2/integration-a` (bench lane) or `rr2/integration-c` point at unmerged local branches; their enforcement lands with those branches.

**Reversal rule.** Reversing a decision is an owner action, never an agent's. The reversal must:
- be recorded here as a new dated line under the entry;
- follow the execution contract: a failing reproduction, a fix at the receiving path, and positive and negative controls;
- where the entry says so, take a new versioned identity (policy, protocol or release) so that evidence produced under the old rule is never reinterpreted.

## Summary

| # | Decision | Adopted option | Behaviour |
|---|---|---|---|
| D1 | W3-T07: an ineligible HWM ledger tip sets the bar | Keep: the tip still sets the bar | unchanged; recorded on `rr2/integration-a` |
| D2 | W1-T03: exceptions to preparation checks | No exception mechanism | unchanged |
| D3 | W3-T06: repairing hand-edited or old-build stores | Only `hwm recover --apply`, which refuses forged pointers | unchanged here; verb on `rr2/integration-a` |
| D4 | W3-T02/T03: requalifying pre-provenance qualifications | None; they stay quarantined | refusal on `rr2/integration-a` |
| D5 | W1-T02: unresolved relations and unknown exposure | Never-seen-family claims stay blocked | unchanged |
| D6 | W8-T02/T05: outbox limits and cadence | The proposed values | adopted on `rr2/integration-a` |
| D7 | W5-T04: refusing proven-infeasible candidates | Feasibility stays diagnostic | unchanged |
| D8 | W5-T02: the slot-keyed stream protocol | Keep v3; no cross-protocol comparison or resume | unchanged |
| D9 | W6: dispatch hold after read-only validation | Keep the current order | unchanged |
| D10 | Python migration | Port live tools; archive unused legacy/research Python | executed separately |
| D11 | W8-T04: a divergent mirror pointer | Keep failing every push until reconciled | unchanged |
| D12 | W6-T06: legacy parents | Warm start only; no exact resume or optimizer carry | **changed** (`84ebe917f`) |
| D13 | W2 G-08: unknown receipt fields | Pin the receipt field set; refuse anything else | **changed** (`48c89cc0b`) |
| D14 | W2: `component_bundle` value derived from the local asset | Keep the single pinned value; refuse any other | unchanged |
| D15 | Policy-worker rebuild and re-pin | The producer path keeps refusing until a rebuilt, re-pinned worker exists | unchanged; refusal verified |
| D16 | W7: `machine_id` added in place to the v1 schemas | Keep in place; a v1 document without it is refused | unchanged |
| D17 | W8: what one update is | One update = one checkpoint window | adopted on `rr2/integration-a` |
| D18 | `jope-learning` dev-dependency on `jope-parallel` | Tests only; the library never links the engine | recorded on `rr2/integration-a` |
| D19 | W9-T02: a final whole-data refit | Not authorized; exports serve the selected fold fit | enforced on `rr2/integration-c` |

---

## D1 — W3-T07: an ineligible HWM ledger tip still sets the bar

- **Question.** A ledger tip that can no longer teach still defines the high-water mark. Examples are a legacy pre-provenance acceptance, or a tip whose retained evidence was lost or corrupted. Should it?
- **Options.**
  - (a) Keep the tip as the bar. An eligible teacher that does not strictly beat it is `not-improved`, so that comparison has no teacher until one does.
  - (b) Eligibility-first selection: the best eligible teacher sets the bar. This is the independent replay's rule, and it needs a new policy version.
- **Adopted.** (a), adopted by owner delegation 2026-09-26. `freeze_teaching` still never lists an ineligible tip, so the tip grants no authority. No weaker candidate is ever admitted.
- **Behaviour.** Unchanged.
- **Enforced.**
  - `crates/jope-learning/src/hwm.rs` `ingest` (compares with the tip) and `decide`;
  - `crates/jope-learning/src/eligibility.rs` `freeze_teaching` (excludes it);
  - test `eligibility::tests::matrix` (W3-T07).
  - Recorded as adopted in `evidence/W03-ingress-map.md` on `rr2/integration-a` (`6252d7ef1`).
- **Reverse.** Implement eligibility-first selection under a new HWM policy version, and replay existing ledgers under the version they were written with.

## D2 — W1-T03: no exception mechanism for preparation checks

- **Question.** Preparation checks fail closed. Is there a declared exception for:
  - legacy pointer views whose stored preprocessing differs from a refit;
  - an exact identity feature (`y = x`);
  - time-series cuts that map two rows to one backing row?
- **Options.**
  - (a) No exceptions: each case stays refused, and needs a new prepared identity.
  - (b) A declared, versioned exception mechanism.
- **Adopted.** (a), adopted by owner delegation 2026-09-26.
- **Behaviour.** Unchanged (fail-closed).
- **Enforced.** `crates/jope-engine/src/prepare.rs`:
  - `ensure_fitted_on_membership`: stored preprocessing must equal a refit on the training rows;
  - `reject_target_copies`: a feature equal to the target;
  - `check_cut_membership`: one backing row on both sides of a cut, and duplicate-expanding joins.

  Tests in `crates/jope-engine/tests/preparation_semantics.rs`:
  - `preparation_state_fitted_beyond_training_membership_is_rejected`;
  - `renamed_exact_target_copy_is_rejected`;
  - `one_backing_row_on_both_sides_of_a_cut_is_rejected`;
  - `duplicate_expanding_training_membership_is_rejected`.
- **Reverse.** Add a declared exception (for example a declared identity transformation) under a new preparation protocol version, with a new prepared identity for every view it admits. Never an override flag on an existing identity.

## D3 — W3-T06: the repair path for hand-edited or old-build stores

- **Question.** Ingest refuses any touched pointer that holds a record the ledger never accepted there. How are such stores repaired?
- **Options.**
  - (a) Only `jope-learning hwm recover --apply`. It rolls a lagging ledger head and lagging pointers forward from the ledger, and refuses (leaves untouched, exit 1) any pointer holding a record the ledger never accepted there.
  - (b) A procedure that can rewrite or adopt such pointers.
- **Adopted.** (a), adopted by owner delegation 2026-09-26. A forged or hand-edited pointer is never adopted. It stays visible and refused.
- **Behaviour.** Unchanged on main. The verb is on `rr2/integration-a` (`fc8dfb199`).
- **Enforced.**
  - On main: `crates/jope-learning/src/hwm.rs` `resolve` (the refusal at ingest) and `recover`;
  - tests `hwm::tests::a_pointer_the_ledger_never_accepted_is_refused_before_any_write` and `hwm::tests::recovery_rolls_lagging_pointers_forward_and_leaves_unaccepted_ones`.
  - On `rr2/integration-a`: `hwm::recover_plan`, and `jope-learning hwm recover [--apply]`.
- **Reverse.** A repair that adopts a pointer the ledger never accepted would need an owner-authorized, logged procedure that appends a new ledger decision through the ordinary receiving path, never an in-place edit.

## D4 — W3-T02/T03: requalification of pre-provenance qualifications

- **Question.** Qualifications stored before context provenance existed are quarantined. Is there a versioned requalification procedure?
- **Options.**
  - (a) None: they stay quarantined and are never upgraded.
  - (b) A versioned requalification procedure.
- **Adopted.** (a), adopted by owner delegation 2026-09-26.
- **Behaviour.** Unchanged on main. An explicit requalification refusal is being implemented on `rr2/integration-a` (bench lane).
- **Enforced.** On main: `crates/jope-learning/src/eligibility/tests/provenance.rs` `legacy_qualification_without_provenance_stays_quarantined_and_is_not_upgraded` (W3-T03).
- **Reverse.** A versioned requalification procedure that re-runs qualification with verified context provenance, and records a new qualification rather than upgrading the stored one.

## D5 — W1-T02: unresolved relations and unknown exposure

- **Question.** Fourteen catalog-name relations are unresolved, and all ten members have unknown encoder, teacher and checkpoint exposure. Can never-seen-family claims be made?
- **Options.**
  - (a) Keep never-seen-family claims blocked until an independent review resolves relations and exposure. Each resolution creates a new release identity.
  - (b) Resolve relations from catalog names.
- **Adopted.** (a), adopted by owner delegation 2026-09-26.
- **Behaviour.** Unchanged.
- **Enforced.**
  - `crates/jope-datasets/src/regression_release.rs` `FamilyRegistry::validate` and `FamilyRegistry::never_seen_claim_blockers`;
  - the frozen `crates/jope-datasets/releases/regression-10-v1/`;
  - `crates/jope-datasets/tests/regression_release.rs`.
- **Reverse.** An independent review resolves each relation or exposure with evidence, and the release is re-derived under a new identity (`regression-10-v2` or later). The v1 release is never edited.

## D6 — W8-T02/T05: outbox limits and cadence

- **Question.** Are the proposed outbox limits and cadence approved?
  - Limits: pause new work at 7,500 pending entries or 768 MiB; refuse every append at 10,000 or 1 GiB.
  - Cadence: flush at 60 s or 1 MiB; checkpoint every 100 updates or 5 minutes, and at shutdown.
- **Options.**
  - (a) Adopt the proposed values as explicit defaults.
  - (b) Keep them owner-pending, so that no product path uses them.
- **Adopted.** (a), adopted by owner delegation 2026-09-26. They stay explicit configuration: a configuration file may set other values, and its reports then say they are not the adopted defaults.
- **Behaviour.** Adopted on `rr2/integration-a` (`6252d7ef1`). On main the constants are still named `PROPOSED`, and no product path uses them.
- **Enforced.** On `rr2/integration-a`:
  - `crates/jope-learning/src/progress/` `Limits::ADOPTED`, `Policy::ADOPTED` and `Config::adopted()`, with the report marker `adopted-by-owner-delegation-2026-09-26`;
  - test `progress::tests::product::outbox_limits_and_cadence_are_explicit_adopted_configuration`;
  - recorded in `evidence/W08-durability.md`.
- **Reverse.** Change the constants, give the report marker a new dated approval, and record it here.

## D7 — W5-T04: feasibility stays diagnostic

- **Question.** Should candidates classified proven-infeasible be refused at dispatch?
- **Options.**
  - (a) Diagnostic only: record the verdict beside each outcome, and leave dispatch and the search protocol unchanged.
  - (b) Refuse at dispatch. That changes search policy and needs a new protocol identity.
- **Adopted.** (a), adopted by owner delegation 2026-09-26.
- **Behaviour.** Unchanged.
- **Enforced.**
  - `crates/feat-math/src/feasibility.rs` `classify` (`feat_math.feasibility.v1`), recorded per attempt by the GP search;
  - tests `feasibility::tests` and `gp::fit_contract_tests::feasibility_estimates_are_recorded_beside_observed_outcomes_under_one_rule`.
- **Reverse.** Define a refusal policy under a new search protocol identity. Searches under the old identity are neither resumed nor compared under the new one.

## D8 — W5-T02: the slot-keyed stream protocol

- **Question.** GP random draws are now keyed by slot and by draw within the slot (`parallel.gp_random_streams.v3`). New searches at the same seed differ from v2 searches, and this build cannot resume a v2 search. Is that accepted?
- **Options.**
  - (a) Keep v3. Never compare across stream protocols, and refuse to resume a v2 search rather than reinterpreting it.
  - (b) Restore v2 for comparability, which brings back the defect where one refused proposal shifts every later slot's draws.
- **Adopted.** (a), adopted by owner delegation 2026-09-26.
- **Behaviour.** Unchanged.
- **Enforced.**
  - `crates/jope-engine/src/gp.rs` and `gp/lineage.rs` `RANDOM_PROTOCOL`, frozen as `random_protocol` in each search request and checked on resume;
  - the paired comparison (`gp::paired::compare`) requires identical frozen requests apart from the declared arm difference;
  - tests `gp::fit_contract_tests::a_refused_proposal_in_one_slot_does_not_shift_another_slots_random_streams` and `a_resume_under_other_selection_rules_is_refused_not_restarted`.
- **Reverse.** A new stream protocol identity. Evidence under v2 or v3 keeps its own identity and is never pooled across protocols.

## D9 — W6: the controller's dispatch-hold order

- **Question.** The GPU dispatch hold now runs after read-only input validation (topology, replay/continuation policy against the corpus index, run-directory state, step window). It still precedes the cancellation check, the GPU lease, telemetry admission, any output, `run.json` and every rank. An ungranted malformed request therefore gets its own refusal instead of the hold's. Keep this order?
- **Options.**
  - (a) Keep the current order.
  - (b) Move the hold back to the first statement, so that nothing is read before it.
- **Adopted.** (a), adopted by owner delegation 2026-09-26. Both orders refuse every ungranted request, and neither writes, leases, locks or spawns anything before the hold. The current order only reads.
  - Note for the owner: (b) is stricter about *reading* caller-named inputs before a grant. The delegation named no behaviour change here, so none was made.
- **Behaviour.** Unchanged.
- **Enforced.**
  - `crates/jope-engine/src/controller.rs` `train_controlled`;
  - tests `replay_training::ungranted_valid_policy_is_still_refused_by_the_dispatch_hold` and `gpu_entry_guards::public_gpu_routes_reject_caller_declarations_before_input_or_cuda_access`;
  - `evidence/W06-training-boundary.md` "Controller input-validation order".
- **Reverse.** Move the hold before the validation. The negative test must then show that an ungranted request reads no input.

## D10 — The Python migration

- **Question.** Should the repo-wide Python migration go ahead (the Python audit's steps 2–5)?
- **Options.**
  - (a) Port the live tools to Rust, bash or jq, and archive the unused legacy and research Python.
  - (b) Keep the Python in place.
- **Adopted.** (a), adopted by owner delegation 2026-09-26. It is being executed separately, including the amendments to `CONTRIBUTING.md:60-62` and `docs/architecture.md:6-8`.
- **Behaviour.** No change in this branch. PR #111 already adds no Python.
- **Enforced.** By that separate migration. The owner's language rule (Rust/Vite/TypeScript/React) applies meanwhile.
- **Reverse.** An owner decision to retain named Python, recorded here with the files it covers.

## D11 — W8-T04: a divergent mirror pointer

- **Question.** A divergent mirror pointer now fails every push until it is reconciled, instead of being forgotten after one attempt. Keep this?
- **Options.**
  - (a) Keep failing every push until an explicit reconciliation.
  - (b) Let a push proceed past the divergence.
- **Adopted.** (a), adopted by owner delegation 2026-09-26. No automatic resolution, and no override.
- **Behaviour.** Unchanged.
- **Enforced.**
  - `crates/jope-learning/src/hwm.rs` `push_mirror`, `settle_owed` and `check_pointer_ancestor`;
  - tests in `crates/jope-learning/tests/mirror_reconcile.rs`: `a_diverged_mirror_is_refused_and_nothing_is_written`, `a_pointer_from_a_chain_this_host_does_not_have_is_refused` and `a_pointer_write_lost_after_the_head_moved_is_still_owed`.
  - The conflict-reconciliation verb `jope-learning progress reconcile [--apply]` is on `rr2/integration-a` (`fc8dfb199`).
- **Reverse.** An owner-authorized reconciliation rule, applied through the ordinary receiving path and recorded here. Never a flag that skips the divergence check.

## D12 — W6-T06: legacy parents

- **Question.** A parent without an objective identity was accepted for exact resume and optimizer carry when both legacy identifiers matched, recorded as `legacy_ids_only`. The accepted step-60000 checkpoint is such a parent. Keep that?
- **Options.**
  - (a) Keep accepting legacy parents for exact resume and optimizer carry.
  - (b) Refuse legacy parents for exact resume and optimizer carry. Allow them only as a weight warm start (optimizer reset, nothing carried), recorded as `legacy_ids_only`.
- **Adopted.** (b), adopted by owner delegation 2026-09-26.
- **Behaviour.** **Changed**, commit `84ebe917f` on `rr2/policy-defaults`. Reproduced first (`pd-w6t06-legacy-pre-20260926T223055Z`: 4 of 9 tests failed because a legacy parent was accepted). Post-fix: `pd-w6t06-legacy-post-20260926T223205Z`, 35/35.
- **Enforced.**
  - `crates/jope-engine/src/training/continuation.rs` `check_parent_partition`, the guard `rank_impl` runs before CUDA. The refusal reads: "a parent without an objective identity continues only as a weight warm start (optimizer reset); … is refused".
  - Tests in `crates/jope-engine/tests/continuation_modes.rs` (feature `neural`, CPU):
    - `a_legacy_parent_continues_only_as_a_weight_warm_start`;
    - `the_accepted_step_60000_checkpoint_continues_as_a_weight_warm_start`, which binds the real accepted `state.json` by its SHA-256;
    - the updated `exact_resume_refuses_every_undeclared_change` and `a_carry_transition_allows_only_declared_changes`.
  - Record: `evidence/W06-training-boundary.md` (e).
- **Consequence.** The native loop still freezes an optimizer-carry policy for its training blocks, so a native-loop block from the step-60000 checkpoint is now refused at the trainer. Continuing from step 60000 works as a weight warm start. A native-loop warm-start window needs a new native-loop protocol identity, because the loop's window accounting assumes a carried optimizer.
- **Reverse.** Remove the `legacy_ids_only` refusal in `check_parent_partition`, and restore the two updated assertions and the two new tests to their accepting form.

## D13 — W2 G-08: unknown receipt fields

- **Question.** The `jope.checkpoint_export.v1` receipt was parsed as untyped JSON, so unknown fields were accepted. Should they be?
- **Options.**
  - (a) Keep accepting unknown fields.
  - (b) Pin the accepted receipt's field set and refuse any receipt carrying fields outside it.
- **Adopted.** (b), adopted by owner delegation 2026-09-26.
- **The accepted receipt.** It is the `step-60000.receipt.json` that jain-datasets `tools/ot-export` 0.1.0 wrote for the released weights. Its `policy.output.sha256` is the W2-T02 weights pin `4b4b5a40…`. On this host it is `/mnt/fast-scratch/jope-migration-qualification/claude-impl-20260917/ot-export-out/step-60000.receipt.json`, SHA-256 `b50150ea…`.
- **Challenger receipts.** These come from this repository's native challenger exporter (`jope-parallel export-policy`, DEMO-3). They are pinned to that exporter's exact field set, which is the accepted set without `context_transforms`, plus `policy.architecture` and `relative_change`.
  - Pinning challengers to the accepted set would refuse every real challenger export. The decision did not ask for that.
  - A re-export of the accepted checkpoint by `jope-parallel export-policy` carries those challenger fields, so it is refused *as the accepted receipt*.
- **Behaviour.** **Changed**, commit `48c89cc0b` on `rr2/policy-defaults`. Reproduced first (`pd-w2g08-fields-pre-20260926T223836Z`: 0 of 3 tests passed because every unpinned field was accepted, including one added to the real accepted receipt). Post-fix:
  - jope-policy 70/70;
  - jope-pipeline `neural-search` 127/127;
  - the real receipt and weights, in a release build: 2/2 (`pd-w2g08-host-real-asset-20260926T224254Z`).
- **Enforced.**
  - `crates/jope-policy/src/config.rs`: `ACCEPTED_RECEIPT_FIELDS`, `CHALLENGER_RECEIPT_FIELDS` and `refuse_unpinned_fields`.
  - The check is called in `CheckpointBinding::parse` after the rejected-checkpoint refusal and before any identity is read. It is therefore reached through `from_receipt_bytes`, `from_challenger_receipt_bytes` and `for_proposal`, which every policy-worker and controller entry point uses.
  - Tests in `crates/jope-policy/tests/jope_dime.rs`:
    - `an_accepted_receipt_carrying_an_unpinned_field_is_refused`;
    - `a_challenger_receipt_carrying_an_unpinned_field_is_refused`;
    - the ignored host test `the_real_accepted_receipt_carries_only_pinned_fields`.
  - Record: `evidence/W02-asset-dependency-table.md` §12.
- **Reverse.** Extend or remove the pinned field set in `config.rs`. Adding a field for a new exporter version should come with a new receipt schema version, rather than widening v1.

## D14 — W2: the encoder `component_bundle` value

- **Question.** The pinned encoder identity's `component_bundle` value (`985074846c…`) was derived from the local pinned asset (`jain-starforge/artifacts/model-bundle.v1.json`, file SHA-256 `5e7884ec…`), not from a published release record. Accept it?
- **Options.**
  - (a) Keep the single pinned value, which every earlier producer recorded, and refuse any other identity. Do not describe it as release-published until a release record publishes the component bundle.
  - (b) Refuse every CPU context until a release record exists.
- **Adopted.** (a), adopted by owner delegation 2026-09-26.
- **Behaviour.** Unchanged (fail-closed toward every other identity).
- **Enforced.**
  - `crates/jope-policy/src/context_producer.rs` `EncoderIdentity::accepted()` and its receivers;
  - test `every_encoder_identity_field_is_pinned_to_the_release_decision`;
  - the ignored host test `accepted_encoder_identity_rederives_from_the_pinned_component_bundle`.
  - `rr2/integration-c` (`e6c479ea5`) also binds it in the versioned semantic manifest.
- **Reverse.** Publish the component bundle in a release record and pin that record, which is a new identity. Any other value needs a new release identity.

## D15 — Rebuilding and re-pinning the policy worker

- **Question.** The controller now requires a proposal ledger in every policy-producer artifact. Should the pinned `jope-policy-producer` be rebuilt and re-pinned now?
- **Options.**
  - (a) Do not rebuild or re-pin now. The producer path keeps refusing until a rebuilt, re-pinned worker exists.
  - (b) Rebuild and re-pin now.
- **Adopted.** (a), adopted by owner delegation 2026-09-26. No binary was rebuilt or re-pinned.
- **Behaviour.** Unchanged. The refusal was verified: the only pinned worker on this host predates the ledger and cannot emit one, and a rebuild without a re-pin fails the pin checks. See `evidence/W04-forward-contract.md` §13 (host run `pd-d15-policy-worker-host-20260926T224418Z`).
- **Enforced.** `crates/jope-pipeline/src/neural_search.rs`:
  - `run_neural`: "policy producer artifact lacks its proposal ledger", for an artifact from a worker that predates the ledger;
  - `verify_policy_inputs`: "policy executable differs from installation pin", and the installation-pin equality, for a rebuilt worker that has not been re-pinned.

  Tests: `producer::tests::historical_producer_artifact_has_no_ledger_evidence` (a pre-ledger artifact carries no ledger) and `producer::tests::producer_artifact_carries_a_reconciled_proposal_ledger`.
- **Reverse.** An owner-authorized rebuild, followed by a re-pin of `policy.program.sha256` in the installation and in each campaign configuration, recorded with the new binary's hash and source commit.

## D16 — W7: `machine_id` added in place to the v1 bounded-run and census schemas

- **Question.** `machine_id` was added to `jope.native_bounded_run.v1` and `jope.native_host_census.v1` in place, before any live use, rather than as v2. Keep it?
- **Options.**
  - (a) Keep it in place. Every v1 document must carry `machine_id`, and one without it is refused, never defaulted.
  - (b) Cut v2 and restore v1 without the field.
- **Adopted.** (a), adopted by owner delegation 2026-09-26. No live v1 document existed when the field was added. If one written before the change is ever found, it is refused.
- **Behaviour.** Unchanged.
- **Enforced.**
  - `crates/jope-engine/src/native_loop/bounded/census.rs` `HostIdentity`: `machine_id` is required and every type is `deny_unknown_fields`;
  - `bounded/contract.rs` and `bounded/rendezvous.rs` compare it;
  - `crates/jope-engine/tests/bounded_run.rs` preflight, census and execution-binding cases.
- **Reverse.** Cut v2 schemas that carry the field, and restore v1 byte-for-byte without it.

## D17 — W8: one update is one checkpoint window

- **Question.** What does one W8 update count?
- **Options.**
  - (a) One window of optimizer steps between two published W8 checkpoints; `consumed_rows` is `(step − previous step) × global_batch`.
  - (b) One optimizer step.
- **Adopted.** (a), adopted by owner delegation 2026-09-26.
- **Behaviour.** Defined on `rr2/integration-a` (bench lane).
- **Enforced.** On `rr2/integration-a`: `update_index` in `crates/jope-learning/src/progress/`, recorded in `evidence/W08-durability.md` (`6252d7ef1`).
- **Reverse.** Redefine the update under a new progress schema version, and never renumber published updates.

## D18 — The `jope-learning` dev-dependency on `jope-parallel`

- **Question.** May `jope-learning` depend on `jope-parallel` (the engine)?
- **Options.**
  - (a) A dev-dependency only. The W8 product tests check their engine fixtures with the engine's own `corpus_publication::verify` and `checkpoint_recovery::reconcile`, and the library never links the engine.
  - (b) No dependency at all: independent fixtures only.
  - (c) A normal dependency.
- **Adopted.** (a), adopted by owner delegation 2026-09-26. A normal (library) dependency stays refused.
- **Behaviour.** Recorded on `rr2/integration-a`.
- **Enforced.**
  - On `rr2/integration-a`: `crates/jope-learning/Cargo.toml` `[dev-dependencies]`, used only by `src/progress/tests/product.rs`; recorded in `evidence/W08-durability.md`.
  - Check: `cargo tree -p jope-learning -e normal -i jope-parallel` must find nothing.
- **Reverse.** Remove the dev-dependency and replace those checks with independent fixtures (b). A normal dependency needs its own owner decision.

## D19 — W9-T02: a final whole-data refit

- **Question.** Is a final whole-data refit of the selected recipe authorized for export?
- **Options.**
  - (a) No. Exports serve the selected recipe's existing fold fit, `final_refit_sha256` is `null`, and the package cannot be presented as the final model.
  - (b) Define and authorize a whole-data refit identity that exports bind.
- **Adopted.** (a), adopted by owner delegation 2026-09-26.
- **Behaviour.** Enforced on `rr2/integration-c` (`a884631a4`). On main no refit identity exists, and W9-T02 records that the exported subject is a fold fit.
- **Enforced.** On `rr2/integration-c`, `crates/jope-bench/src/panel/fitted_export_run.rs`:
  - `PackageRecord.final_refit_sha256` is always `None`;
  - `FittedSubject` has kind `selected_fold_fit`;
  - `release` is `not-decided`.
- **Reverse.** Define a whole-data refit protocol and identity, authorize it, and bind it in a new package record version. Existing records keep `null`.
