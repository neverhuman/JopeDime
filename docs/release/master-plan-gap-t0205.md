# Master-plan release gap audit — t0205

**Observed current main:** `d0bfa72a6c8270653b277ebd87025089ee3ddeeb` (2026-10-31 UTC)
**Scope:** all 137 indexed cards P001–P137 in `~/jope_plan/master/master_plan.md`.
**Result:** 131 DONE, 4 GATED, 2 MISSING.

## How to read the classifications

- **DONE** means the indexed task deliverable has task-specific merged evidence on current `origin/main`. It does not assert that every downstream runtime, scientific, platform, or release qualification is complete.
- **GATED** means some task-specific work or a bounded component is merged, but the indexed card's completion criterion is still not evidenced. The exact remaining gate and available CPU-only preparation are listed below.
- **MISSING** means the indexed owner deliverable/decision has no merged artifact or completed decision record. These two are OWNER-only decisions; this lane cannot substitute a CPU implementation for them.

The dated baseline inventory `STATUS_VS_MAIN_t2136.md` had 131 indexed DONE rows and four ACTIVE plus two QUEUED. Its 131 cited merge commits were resolved and verified as ancestors of refreshed `origin/main`; PR state and the four active rows were rechecked. The old `master_plan.md` source-control anchor (`5c8a167…`, shown in its historical plan text) is stale. The live main SHA above is the comparison point. The current open PR queue was checked: #935 and #936; neither overlaps this report. #931 is already merged.

The lane nudge's “~124/137 done” is an approximate release-readiness framing, not the indexed task-delivery count. This audit reports the exact card-based count and keeps merged implementation separate from qualification. In particular, #793, #801, #797 and #813 are merged bounded components/preparation; they do not close the outstanding card criteria described below.

## Incomplete indexed cards: gates and CPU-only preparation

| Task | Merged state | Gate preventing DONE | CPU-only preparation available now |
|---|---|---|---|
| DSC-06 | Proposer CPU source merged (#793). The generated-only 10,000-proposal fixture is explicitly not the matched evaluated/adaptively admitted population. | Demonstrate operator-family frequencies within ±2 percentage points on the training-matched evaluated population, with deterministic seed and filter evidence. The RES-04 proposal/evaluation population snapshot and exact replay identity are required; generated-only counts do not satisfy M3. | Freeze a source/input manifest for the eligible RES-04 snapshot when supplied; prepare a deterministic replay and frequency-summary command with explicit population/hash checks. Existing synthetic controls can validate the summarizer only, not close M3. |
| QLT-03 | CPU record schema/writer/Python reader component merged (#801); synthetic Rust→Python inventory matching is evidenced. | Indexed card also requires an actual recorded run read back with matching counts and a release build with no `research` command. Engine capture/outbox and feature-gated CLI are deferred to QLT-03-F01/F02; no actual engine-run/release-command proof is recorded. | Define and validate the producer-to-reader fixture contract, count invariants, and release-feature command-absence check. Run those CPU checks after the deferred CLI/capture paths land; do not count synthetic reader controls as an engine run. |
| QLT-04 | CPU preparation report merged (#797); no Q1 outcome/default decision. | Exact preregistered Q1 execution and complete report with OWNER-recorded guidance default and hashed run receipts. Task card requires GPU/`qualification.lock`; owner authorization, qualified production backend/evaluator/cohort and stronger-model execution remain prerequisites. | CPU-only validate preregistration/receipt schemas, input identity pins, report completeness checklist, and analysis code on synthetic fixtures. Do not acquire the lock, run Q1, access protected cohorts, or infer a default. |
| REL-07 | Release-gate inventory/preparation merged (#813); planner reports all 29 gates not qualified/not run. | Execute all 28 release gates on exact signed bytes, assemble the hashed qualification receipt/release notes, qualify native platform/client/reference-laptop requirements, record Q1/default, and obtain OWNER approval. #813 is an inventory, not gate execution or a signed release. | Reconcile CPU-only gate evidence references and validate receipt/schema/planner behavior against existing receipts. Keep every absent gate as null/not-qualified; no signing, publication, protected-data access, GPU, or release claim. |
| GOV-01 | No task-specific merged brief or signature. | OWNER must sign the program brief (including explicit policy/waiver decisions called out by the plan). | No substitute preparation closes an OWNER decision. CPU-only preparation can assemble a source-linked list of unresolved fields for the owner; do not infer or sign decisions. |
| GOV-02 | No task-specific merged OD decision record. | OWNER must decide OD1–OD3, OD5, OD6, and OD8; latest owner record keeps GOV-01/02 parked for Jepson. | No substitute preparation closes an OWNER decision. CPU-only preparation can map each OD to its existing source/options; do not select defaults or record a decision. |

## Per-card inventory

Evidence below comes from the task-specific merged-PR/merge-ancestry audit. For GATED and MISSING rows, the detailed gate above supersedes stale `master_plan.md` status labels.

| File | Task | State | Current-main evidence / reason |
|---|---|---|---|
| `T001-API-01-jope-api-crate-and-core-types-contract-freeze.md` | API-01 | **DONE** | PR #467 → `0cbac52869a6`. |
| `T002-GOV-03-freeze-the-v1-h40b-checkpoint.md` | GOV-03 | **DONE** | PR #573 → `d8383a65bc0b`. |
| `T003-RES-01-inventory-the-exact-custody-closure-that-produce.md` | RES-01 | **DONE** | PR #587 → `6d9d6403fc2d`. |
| `T004-API-02-error-catalog-and-rfc-9457-problem-type.md` | API-02 | **DONE** | PR #481 → `21b81f909550`. |
| `T005-API-03-id-newtypes.md` | API-03 | **DONE** | PR #500 → `e0ae51daaeb5`. |
| `T006-API-04-decimal-string-u64-serde.md` | API-04 | **DONE** | PR #501 → `787774792fe3`. |
| `T007-API-06-register-new-crates-in-the-workspace-and-ci-scop.md` | API-06 | **DONE** | PR #499 → `6923cbd2b3cd`; PR #493 → `67c0dd8de279`. |
| `T008-DSC-02-random-stratified-splits-k-fold-rung-0-subsample.md` | DSC-02 | **DONE** | PR #510 → `fe9cc58bddab`. |
| `T009-DSC-04-elite-pool-data-structure.md` | DSC-04 | **DONE** | PR #508 → `009de907e9c1`. |
| `T010-DSC-05-effort-table-budget-accounting-patience.md` | DSC-05 | **DONE** | PR #502 → `c1912ed1b1a2`. |
| `T011-GOV-04-put-spec-review-and-plan-in-docs-product-with-a-.md` | GOV-04 | **DONE** | PR #570 → `1129b01b01d4`; PR #464 → `9d323c4906d0`. |
| `T012-PUB-02-public-names-for-stage-operations.md` | PUB-02 | **DONE** | PR #490 → `9039d2eb17c0`. |
| `T013-PUB-03-public-names-for-heads.md` | PUB-03 | **DONE** | PR #520 → `4e03eed3441e`. |
| `T014-PUB-04-public-names-for-parameters.md` | PUB-04 | **DONE** | PR #548 → `7190d28942bd`; PR #529 → `09cf7a71d363`. |
| `T015-QLT-05-reference-datasets-for-tests-and-d2.md` | QLT-05 | **DONE** | PR #472 → `f2958205f1a1`. |
| `T016-RES-03-freeze-guard-refuse-missing-buffers-f09.md` | RES-03 | **DONE** | PR #460 → `3e589ec38209`. |
| `T017-RES-08-fused-bundle-json-schema.md` | RES-08 | **DONE** | PR #462 → `39a3e6451513`. |
| `T018-RNK-11-remove-c-from-the-cpu-closure-candle-patch-ban-l.md` | RNK-11 | **DONE** | PR #463 → `e741273de7a2`. |
| `T019-SVC-07-config-toml-loader-with-precedence-and-origins.md` | SVC-07 | **DONE** | PR #504 → `cc187b643de5`. |
| `T020-SVC-08-jopedime-home-resolver-in-jope-paths.md` | SVC-08 | **DONE** | PR #461 → `5035756434b7`. |
| `T021-TEL-01-fix-cpu-math-f06.md` | TEL-01 | **DONE** | PR #465 → `6bfd1c823d71`. |
| `T022-TEL-02-sink-parquet-feature-in-jope-telemetry.md` | TEL-02 | **DONE** | PR #492 → `83cc5ed2eb05`. |
| `T023-TEL-03-allowed-cpus-from-cgroup-and-affinity.md` | TEL-03 | **DONE** | PR #480 → `edd3d28bac08`; PR #477 → `a8e0a642d46d`. |
| `T024-TEL-07-per-language-lexical-token-counters-for-tok-s.md` | TEL-07 | **DONE** | PR #503 → `4444374af032`. |
| `T025-TUI-05-theme-tokens-color-detection-sanitizer.md` | TUI-05 | **DONE** | PR #475 → `1a30a8ffd22e`. |
| `T026-TUI-08-syntax-highlighter-for-three-languages.md` | TUI-08 | **DONE** | PR #483 → `772f8d5af2e4`. |
| `T027-TUI-10-pty-test-harness.md` | TUI-10 | **DONE** | PR #506 → `069d200319ae`; PR #489 → `682bca7a1fdd`. |
| `T028-WRK-02-length-prefixed-frame-codec.md` | WRK-02 | **DONE** | PR #505 → `fb5ee1822745`. |
| `T029-WRK-03-cross-platform-per-pid-cpu-and-rss-sampler.md` | WRK-03 | **DONE** | PR #514 → `771755480af1`. |
| `T030-DSC-01-public-per-candidate-api-in-feat-math.md` | DSC-01 | **DONE** | PR #470 → `64d457939b21`. |
| `T031-GOV-01-sign-the-program-owner-brief.md` | GOV-01 | **MISSING** | No task-specific merged PR, open PR, active claim, blocker, or platform park located. |
| `T032-GOV-02-decide-od1-od3-od5-od6-od8.md` | GOV-02 | **MISSING** | No task-specific merged PR, open PR, active claim, blocker, or platform park located. |
| `T033-MCP-01-conformance-spike-rmcp-vs-existing-dispatcher-od.md` | MCP-01 | **DONE** | PR #474 → `a78816fe2987`. |
| `T034-OBJ-02-jope-classification-fitness-v1-schema-framework-.md` | OBJ-02 | **DONE** | PR #468 → `3dc85dfa4b30`. |
| `T035-RES-02-vendor-custody-files-and-write-a-hash-verifier.md` | RES-02 | **DONE** | PR #641 → `ab31d7ec4c5c`. |
| `T036-API-05-openapi-generator-and-ci-drift-check.md` | API-05 | **DONE** | PR #571 → `2acab554224f`. |
| `T037-API-08-registry-consistency-tests.md` | API-08 | **DONE** | PR #524 → `6d50eac5d1dc`. |
| `T038-BAT-05-exit-codes-idempotency-key-receipt-printing.md` | BAT-05 | **DONE** | PR #521 → `0fb9dc3ab277`. |
| `T039-OBJ-01-wire-regression-objectives-into-evaluation.md` | OBJ-01 | **DONE** | PR #516 → `7d09c9b0f5d0`. |
| `T040-OBJ-03-remaining-binary-metrics-against-golden-values.md` | OBJ-03 | **DONE** | PR #561 → `13b73b90bf65`; PR #482 → `38b254f1be3c`. |
| `T041-OBJ-05-order-dependent-refusal-and-fitting-disclosure.md` | OBJ-05 | **DONE** | PR #546 → `5614f6fed615`. |
| `T042-PUB-05-phrase-renderer-and-banned-word-lint.md` | PUB-05 | **DONE** | PR #539 → `4c9fd776d7c6`. |
| `T043-QLT-02-held-out-cohort-manifest.md` | QLT-02 | **DONE** | PR #636 → `db36f04108c6`. |
| `T044-RNK-02-port-proj16.md` | RNK-02 | **DONE** | PR #642 → `8e8d499f8834`. |
| `T045-SVC-06-daemon-json-writer-and-reader-with-owner-only-pe.md` | SVC-06 | **DONE** | PR #522 → `6b301777af3d`. |
| `T046-SVC-10-get-v1-system-handler.md` | SVC-10 | **DONE** | PR #554 → `123c585328b5`. |
| `T047-SVC-12-dataset-profile-and-target-suggestion.md` | SVC-12 | **DONE** | PR #578 → `0757055e515d`. |
| `T048-TEL-05-nvml-device-mapping-and-unavailable-reasons.md` | TEL-05 | **DONE** | PR #476 → `3115dfed0a2f`. |
| `T049-TEL-06-warning-catalog.md` | TEL-06 | **DONE** | PR #518 → `80b954a2bf83`. |
| `T050-TUI-03-command-parser.md` | TUI-03 | **DONE** | PR #526 → `123349f75278`. |
| `T051-TUI-04-completion-engine.md` | TUI-04 | **DONE** | PR #517 → `dd81b891bcdd`. |
| `T052-TUI-06-transcript-card-renderers.md` | TUI-06 | **DONE** | PR #525 → `cf6078317fbc`. |
| `T053-TUI-07-status-and-activity-lines.md` | TUI-07 | **DONE** | PR #523 → `d60ec583c5fd`. |
| `T054-TUI-11-recorded-event-fixtures-and-fake-transport.md` | TUI-11 | **DONE** | PR #530 → `13dfc08c15ae`. |
| `T055-API-07-operation-registry-17-ops-12-tools-24-commands.md` | API-07 | **DONE** | PR #487 → `cef579478940`. |
| `T056-CLT-01-client-transports-inprocess-oneshot-and-http-ure.md` | CLT-01 | **DONE** | PR #544 → `fc4e21b17f53`. |
| `T057-DSC-03-grouped-and-time-ordered-folds-leakage-audit-hoo.md` | DSC-03 | **DONE** | PR #549 → `9bd867876ef0`. |
| `T058-DSC-11-f64-prediction-path-and-non-finite-failure-f23.md` | DSC-11 | **DONE** | PR #478 → `f94ebc0f07ed`. |
| `T059-EXP-01-capability-registry-and-proposer-filter.md` | EXP-01 | **DONE** | PR #496 → `40b33a0e8af9`; PR #348 → `cc21d667775d`; PR #345 → `aad22f87121d`. |
| `T060-QLT-01-q1-preregistration-and-harness-od7.md` | QLT-01 | **DONE** | PR #506 → `069d200319ae`; PR #497 → `8f417243b407`. |
| `T061-RES-04-document-the-training-proposer-and-its-operator-.md` | RES-04 | **DONE** | PR #657 → `bdf3549620cb`. |
| `T062-RES-05-surrogate-recipe-rung-0-measurement-and-revealed.md` | RES-05 | **DONE** | PR #697 → `7ca59e1e9052`; PR #670 → `864915caced7`. |
| `T063-RNK-01-bundle-loader-and-verification.md` | RNK-01 | **DONE** | PR #565 → `262ede2ffdde`. |
| `T064-SVC-01-store-redb-content-addressed-blobs-publish-order.md` | SVC-01 | **DONE** | PR #527 → `2661cd01b26a`. |
| `T065-SVC-02-idempotency-committed-atomically-with-effects.md` | SVC-02 | **DONE** | PR #540 → `6f1dfe8cab40`. |
| `T066-SVC-03-run-controller-six-states-revisions-fencing.md` | SVC-03 | **DONE** | PR #509 → `47a35eda5a2e`. |
| `T067-SVC-04-event-log-publish-after-commit-sse-long-poll.md` | SVC-04 | **DONE** | PR #543 → `d112464133c9`. |
| `T068-SVC-05-axum-router-bearer-auth-host-check-stubs.md` | SVC-05 | **DONE** | PR #519 → `cd3bf1f432ec`. |
| `T069-SVC-11-dataset-registration-and-the-jmat-matrix-format.md` | SVC-11 | **DONE** | PR #541 → `fea8754eb4ad`. |
| `T070-TUI-01-terminal-skeleton-inline-viewport-event-loop-res.md` | TUI-01 | **DONE** | PR #552 → `fd3013c753ab`. |
| `T071-TUI-02-composer-and-editing.md` | TUI-02 | **DONE** | PR #576 → `6e0739b45f60`. |
| `T072-WRK-01-worker-pool-limits-fencing-parent-death-safety.md` | WRK-01 | **DONE** | PR #555 → `993ce5f700a4`. |
| `T073-DSC-08-discovery-loop-heuristic-admission.md` | DSC-08 | **DONE** | PR #630 → `0c233e18b426`; PR #660 → `b2faf8ce9327`. |
| `T074-RES-06-deterministic-fused-bundle-exporter.md` | RES-06 | **DONE** | PR #667 → `7b30157ca228`. |
| `T075-RES-07-golden-parity-fixtures.md` | RES-07 | **DONE** | PR #726 → `f53163b36dd0`. |
| `T076-RNK-08-adapter-10tok-predict-costs.md` | RNK-08 | **DONE** | PR #692 → `024f82890ab5`. |
| `T077-BAT-01-serve-cli-wiring.md` | BAT-01 | **DONE** | PR #645 → `259691938d41`. |
| `T078-BAT-02-run-status-events.md` | BAT-02 | **DONE** | PR #607 → `dbbba6cc8364`. |
| `T079-BAT-03-control-restart-runs-code.md` | BAT-03 | **DONE** | PR #644 → `6b3543553fb7`. |
| `T080-BAT-04-export-predict-dataset-objective-gc.md` | BAT-04 | **DONE** | PR #628 → `dc23fef636df`. |
| `T081-BAT-07-lite-build-features-and-ci-job.md` | BAT-07 | **DONE** | PR #586 → `c801cffce100`. |
| `T082-CLT-03-upload-download-and-safe-atomic-extraction-to-ou.md` | CLT-03 | **DONE** | PR #572 → `62327dd06f1f`. |
| `T083-CLT-04-contract-test-harness-across-transports.md` | CLT-04 | **DONE** | PR #563 → `f6c19b01c1cf`; PR #562 → `5cd93e214724`. |
| `T084-DSC-07-port-select-landmark-preferred-pools.md` | DSC-07 | **DONE** | PR #664 → `edb7c86dc70d`. |
| `T085-REL-01-link-audit-script-and-ci-job.md` | REL-01 | **DONE** | PR #488 → `f95b42f50a2c`. |
| `T086-RES-09-rust-golden-fixture-loader-for-tests.md` | RES-09 | **DONE** | PR #743 → `a3e79ae62654`; PR #711 → `1ceadf03bc4c`. |
| `T087-RNK-06-lm48-projection.md` | RNK-06 | **DONE** | PR #735 → `cc5670d144e6`. |
| `T088-RNK-09-champion-bank-loader-and-nearest-retrieval.md` | RNK-09 | **DONE** | PR #742 → `952883072113`; PR #694 → `d3b589181be6`. |
| `T089-SVC-13-delete-operations-and-store-gc.md` | SVC-13 | **DONE** | PR #559 → `810a3fc34af0`. |
| `T090-CLT-02-engine-discovery-and-cross-platform-auto-start.md` | CLT-02 | **DONE** | PR #637 → `11286a848dca`. |
| `T091-DSC-06-proposer-with-training-matched-operator-mix.md` | DSC-06 | **GATED** | sol_v1_dsc08 owns CPU proposer source; matched M3 replay remains unqualified. |
| `T092-EXP-02-python-fitted-state-glue.md` | EXP-02 | **DONE** | PR #553 → `aac568984b53`. |
| `T093-EXP-03-rust-fitted-state-glue.md` | EXP-03 | **DONE** | PR #556 → `ab374912c3c3`. |
| `T094-MCP-02-mcp-adapter-with-12-tools.md` | MCP-02 | **DONE** | PR #582 → `ff127e4d61cb`. |
| `T095-OBJ-04-cross-fitted-platt-calibration-in-the-model-obje.md` | OBJ-04 | **DONE** | PR #494 → `498d346e1692`. |
| `T096-PUB-01-public-projection-and-claim-levels.md` | PUB-01 | **DONE** | PR #558 → `ad2c16437db6`. |
| `T097-RNK-03-feat412-featurizer-skeleton-and-first-feature-fa.md` | RNK-03 | **DONE** | PR #689 → `0d8810b61195`; PR #678 → `ef5dcead50d7`. |
| `T098-RNK-04-one-task-per-remaining-feature-family.md` | RNK-04 | **DONE** | PR #749 → `a733a608abe9`; PR #715 → `ba6e7f638a73`; PR #703 → `d7d7c8eef75c`. |
| `T099-RNK-05-full-landmark-recipe-port.md` | RNK-05 | **DONE** | PR #740 → `3a1adc560686`; PR #701 → `cc0296a1dd8b`; PR #687 → `76f89a4da9c9`. |
| `T100-RNK-07-stage-1b-token-projection.md` | RNK-07 | **DONE** | PR #733 → `a29a5ba3da6f`; PR #700 → `9fd2fae21c5a`. |
| `T101-SVC-09-engine-lock-idle-exit-serve-status-stop.md` | SVC-09 | **DONE** | PR #583 → `a5f5e3e598cf`. |
| `T102-TEL-04-engine-telemetry-service.md` | TEL-04 | **DONE** | PR #580 → `b22b932f869d`. |
| `T103-WRK-04-resource-governor.md` | WRK-04 | **DONE** | PR #623 → `8ae12728208f`. |
| `T104-RNK-10-learned-guidance-in-the-loop.md` | RNK-10 | **DONE** | PR #755 → `f738b81453be`. |
| `T105-DSC-10-warm-start-fork.md` | DSC-10 | **DONE** | PR #659 → `4abe7be83c83`. |
| `T106-EXP-04-regression-calibration-in-all-three-emitters.md` | EXP-04 | **DONE** | PR #581 → `a9a304d1a60e`; PR #579 → `dee223831e14`. |
| `T107-EXP-05-platt-calibration-in-all-three-emitters.md` | EXP-05 | **DONE** | PR #588 → `36a09d561e42`. |
| `T108-EXP-06-typescript-d-ts-generator.md` | EXP-06 | **DONE** | PR #577 → `981edf75df53`. |
| `T109-EXP-07-self-check-and-check-rows.md` | EXP-07 | **DONE** | PR #574 → `cfd1bac4d7f6`. |
| `T110-EXP-09-export-bundle-assembly.md` | EXP-09 | **DONE** | PR #585 → `024b39c905bb`. |
| `T111-MCP-03-legacy-tool-aliases.md` | MCP-03 | **DONE** | PR #621 → `183e9a10d48a`. |
| `T112-MCP-04-client-setup-docs-and-scenario-scripts.md` | MCP-04 | **DONE** | PR #589 → `b966c319662d`. |
| `T113-PUB-06-t2-sentinel-harness-across-every-surface.md` | PUB-06 | **DONE** | PR #709 → `362085ec9f05`. |
| `T114-BAT-06-in-process-batch-lifecycle-and-engine-busy.md` | BAT-06 | **DONE** | PR #646 → `4e6a834794d0`. |
| `T115-DSC-09-finalization-refits-and-model-artifact.md` | DSC-09 | **DONE** | PR #766 → `33ac6687c102`; PR #665 → `32c93be9b633`. |
| `T116-EXP-08-parity-ci-harness-with-real-toolchains.md` | EXP-08 | **DONE** | PR #594 → `d3b05b8b47bd`. |
| `T117-INT-01-surface-equivalence-and-golden-path-suites.md` | INT-01 | **DONE** | PR #745 → `782fabb91949`; PR #680 → `47f9c0845b90`. |
| `T118-INT-02-reconnect-determinism-across-reconnect-system-cr.md` | INT-02 | **DONE** | PR #724 → `5ad6f5e05fc2`. |
| `T119-QLT-03-research-record-mode.md` | QLT-03 | **GATED** | sol_v1_api01 owns research-record source; producer/durable-capture follow-ups remain. |
| `T120-REL-02-fused-cpu-fused-cuda-profiles-retire-full.md` | REL-02 | **DONE** | PR #603 → `192bf9ec2269`. |
| `T121-TUI-09-draft-run-spec-interrupts-quit-attach-replay.md` | TUI-09 | **DONE** | PR #705 → `a9b7c8fe6037`; PR #684 → `1febaec8470b`. |
| `T122-QLT-04-run-q1-and-record-the-decision.md` | QLT-04 | **GATED** | luna_v1_rnk02 owns active CPU-prep branch; final Q1 remains unrun/unsigned. |
| `T123-REL-03-size-budget-entry.md` | REL-03 | **DONE** | PR #682 → `830eb84270b3`. |
| `T124-REL-05-product-docs-with-executed-examples.md` | REL-05 | **DONE** | PR #753 → `e931225338b4`. |
| `T125-REL-06-d2-startup-and-first-model-benchmark.md` | REL-06 | **DONE** | PR #671 → `d99dbd3408fe`. |
| `T126-MCP-05-named-client-runs.md` | MCP-05 | **DONE** | PR #737 → `49b87906a6ec`. |
| `T127-REL-04-clean-host-matrix-on-three-oses.md` | REL-04 | **DONE** | PR #718 → `1243345ef4ec`. |
| `T128-RNK-12-cuda-parity-and-placement.md` | RNK-12 | **DONE** | PR #767 → `8b030971186b`. |
| `T129-REL-07-qualification-receipt-and-final-gate-rerun.md` | REL-07 | **GATED** | sol_v1_gov03 owns CPU receipt pre-stage; runtime qualification remains pending. |
| `T130-HWV-01-golden45-elo-for-h38-f2-and-h40b.md` | HWV-01 | **DONE** | PR #769 → `98ed3425ad6a`. |
| `T131-HWV-02-recover-h29-h34-and-h38-h45-evidence.md` | HWV-02 | **DONE** | PR #479 → `2a18420a59e6`. |
| `T132-HWV-03-replicate-h03-readout-aware-summaries.md` | HWV-03 | **DONE** | PR #612 → `b84549e04ff1`. |
| `T133-HWV-04-run-h09-dsl-residual-around-tree-baseline.md` | HWV-04 | **DONE** | PR #534 → `7796207d3424`. |
| `T134-HWV-05-h27-fingerprint-ranker-ablation.md` | HWV-05 | **DONE** | PR #774 → `3bcca9cc4214`. |
| `T135-HWV-06-flywheel-reuse-qualified-contexts.md` | HWV-06 | **DONE** | PR #535 → `eeabea7fa8fb`. |
| `T136-JRB-01-joint-regbin-learning-check.md` | JRB-01 | **DONE** | PR #739 → `aeb07146c930`; PR #734 → `076d8ab0b630`. |
| `T137-JRB-02-per-task-elo-and-percentile-store.md` | JRB-02 | **DONE** | PR #697 → `7ca59e1e9052`; PR #606 → `fc9d37100f8f`. |
## Release-readiness boundary

This is a source-control and task-evidence inventory, not a release approval. Merged task deliverables elsewhere in the 137-row index can still have open product qualification requirements recorded in their task/status evidence. No GPU, sealed/holdout data, corpus payload, signing, publication, model training, DOPE change, or GitHub CI was used for this audit. No Cargo checks were needed for this documentation-only change.
