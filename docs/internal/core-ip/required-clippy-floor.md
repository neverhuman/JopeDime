# Required Clippy floor remediation

This change addresses the 51 diagnostic entries inherited by the assigned cleanup lane. The inventory preserves repeated compiler emissions: C003 and C044 report the same worker fields in different targets. It is a receiving-API and lint cleanup; serialized protocols, objective coefficients, data scopes, admitted resource limits and checkpoint compatibility checks are preserved.

The baseline is `aac714fd56e28a5d0f1523c3959940ffc6f48a0d`. The corrected source commit is `5bcb402a54f1b2d313c89073e50fb2448fc7f5f2`, based on updated main `988b3f32eb53d72271d8216e438e659f3a7db502` with the independently landed authority clock fix preserved. `just required` failed with exit 101 at Clippy before its Rust/Python test stages. Clearing the initial library barrier exposes additional campaign binary diagnostics, so the assigned 51 are not an exhaustive workspace count. The full required gate must continue to be reported as failing until those remaining targets and every later stage pass.

The original `grammar.rs` is a permanently frozen historical source. The active module is `grammar/automaton.rs`, which retains its implementation with the two equivalent lint rewrites. The public module and test identities stay the same; corpus and teaching identities hash the active source as well as the historical bytes. The frozen grammar, vocabulary and parity fixtures remain strict.

Argument bundles group values that already travel together. They are input carriers, not authenticated receipts: authority still validates fitting arrays, live runtime/process identity and observed completion. The authorized completion fixture additionally checks distinct retained output, model, prediction and metric bytes and hashes. Rust callers in this repository migrate together; these non-published crates do not establish compatibility for external Rust consumers.

The CUDA holder now destroys its stream before its context while its driver library remains alive. Its process-lifetime singleton, device admission, compute implementation and CPU fallback are unchanged. A destructor implementation and CPU compilation do not qualify a live GPU path.

## Assigned diagnostic inventory

Locations below identify the retained pre-fix report. Closure requires the strict receiving/library checks plus the normal test and exact-head CI evidence; a changed source declaration alone is insufficient.

| ID | Baseline location | Diagnostic | Remediation |
| --- | --- | --- | --- |
| C001 | `crates/jope-engine/src/grammar.rs:120` | `int_plus_one` | Bound-preserving depth comparison and parser while-let. |
| C002 | `crates/jope-engine/src/corpus.rs:42` | `dead_code` | Test-only convenience method; grouped sources; direct borrow; Entry insertion preserves first label and duplicate count. |
| C003 | `crates/jope-engine/src/smartcluster_worker.rs:335` | `dead_code` | Neural-only fields/accessors use the existing neural feature; CPU stop/custody storage and receiving checks remain active. |
| C004 | `crates/jope-engine/src/smartcluster_worker.rs:343` | `dead_code` | Neural-only fields/accessors use the existing neural feature; CPU stop/custody storage and receiving checks remain active. |
| C005 | `crates/jope-engine/src/checkpoint_recovery.rs:94` | `too_many_arguments` | TrainingTopology groups world, microbatch and global batch without changing comparisons. |
| C006 | `crates/jope-engine/src/controller.rs:154` | `too_many_arguments` | ProposalOptions/TrainingOptions group inputs; parent-death fixture explicitly waits for its child. |
| C007 | `crates/jope-engine/src/controller.rs:373` | `too_many_arguments` | ProposalOptions/TrainingOptions group inputs; parent-death fixture explicitly waits for its child. |
| C008 | `crates/jope-engine/src/controller.rs:401` | `too_many_arguments` | ProposalOptions/TrainingOptions group inputs; parent-death fixture explicitly waits for its child. |
| C009 | `crates/jope-engine/src/controller.rs:479` | `too_many_arguments` | ProposalOptions/TrainingOptions group inputs; parent-death fixture explicitly waits for its child. |
| C010 | `crates/jope-engine/src/dashboard.rs:2575` | `too_many_arguments` | DashboardData groups the same render inputs in their original order. |
| C011 | `crates/jope-engine/src/dashboard.rs:2758` | `too_many_arguments` | DashboardData groups the same render inputs in their original order. |
| C012 | `crates/jope-engine/src/dashboard.rs:2809` | `too_many_arguments` | DashboardData groups the same render inputs in their original order. |
| C013 | `crates/jope-engine/src/fit_boundary.rs:210` | `too_many_arguments` | FitPolicies groups existing policy references. |
| C014 | `crates/jope-engine/src/gp.rs:1584` | `too_many_arguments` | SearchOptions groups the existing request controls; fit/accounting/selection bodies are preserved. |
| C015 | `crates/jope-engine/src/gp.rs:1607` | `too_many_arguments` | SearchOptions groups the existing request controls; fit/accounting/selection bodies are preserved. |
| C016 | `crates/jope-engine/src/gp.rs:1636` | `too_many_arguments` | SearchOptions groups the existing request controls; fit/accounting/selection bodies are preserved. |
| C017 | `crates/jope-engine/src/gp.rs:1729` | `too_many_arguments` | SearchOptions groups the existing request controls; fit/accounting/selection bodies are preserved. |
| C018 | `crates/jope-engine/src/prepare.rs:287` | `manual_clamp` | Equivalent positive-range clamp and exact tuple type alias. |
| C019 | `crates/jope-engine/src/prepare.rs:288` | `manual_clamp` | Equivalent positive-range clamp and exact tuple type alias. |
| C020 | `crates/jope-engine/src/prepare.rs:722` | `type_complexity` | Equivalent positive-range clamp and exact tuple type alias. |
| C021 | `crates/jope-engine/src/corpus.rs:518` | `too_many_arguments` | Test-only convenience method; grouped sources; direct borrow; Entry insertion preserves first label and duplicate count. |
| C022 | `crates/jope-engine/src/corpus.rs:618` | `needless_borrow` | Test-only convenience method; grouped sources; direct borrow; Entry insertion preserves first label and duplicate count. |
| C023 | `crates/jope-engine/src/corpus.rs:686` | `too_many_arguments` | Test-only convenience method; grouped sources; direct borrow; Entry insertion preserves first label and duplicate count. |
| C024 | `crates/jope-engine/src/corpus.rs:710` | `too_many_arguments` | Test-only convenience method; grouped sources; direct borrow; Entry insertion preserves first label and duplicate count. |
| C025 | `crates/jope-engine/src/corpus.rs:887` | `map_entry` | Test-only convenience method; grouped sources; direct borrow; Entry insertion preserves first label and duplicate count. |
| C026 | `crates/jope-engine/src/corpus_publication.rs:72` | `too_many_arguments` | PublicationSources groups existing input paths. |
| C027 | `crates/jope-engine/src/eval_registry/authority.rs:266` | `too_many_arguments` | FitInputs and CompletionArtifacts group payloads; verified process/runtime/fit and completion checks remain in their receiving constructors. |
| C028 | `crates/jope-engine/src/eval_registry/authority.rs:296` | `too_many_arguments` | FitInputs and CompletionArtifacts group payloads; verified process/runtime/fit and completion checks remain in their receiving constructors. |
| C029 | `crates/jope-engine/src/eval_registry/authority.rs:3919` | `too_many_arguments` | FitInputs and CompletionArtifacts group payloads; verified process/runtime/fit and completion checks remain in their receiving constructors. |
| C030 | `crates/jope-engine/src/eval_registry/authority.rs:4001` | `too_many_arguments` | FitInputs and CompletionArtifacts group payloads; verified process/runtime/fit and completion checks remain in their receiving constructors. |
| C031 | `crates/jope-engine/src/eval_registry/service.rs:1951` | `byte_char_slices` | Exact byte literal and positive-divisor is_multiple_of. |
| C032 | `crates/jope-engine/src/eval_registry/service.rs:2025` | `manual_is_multiple_of` | Exact byte literal and positive-divisor is_multiple_of. |
| C033 | `crates/jope-engine/src/grammar.rs:659` | `while_let_loop` | Bound-preserving depth comparison and parser while-let. |
| C034 | `crates/jope-engine/src/label_admission.rs:429` | `too_many_arguments` | AdmissionInputs groups vocabulary, expected identity and training rows. |
| C035 | `crates/jope-engine/src/task_contract.rs:94` | `manual_is_multiple_of` | Positive-divisor is_multiple_of preserves existing limit/refusal checks. |
| C036 | `crates/jope-engine/src/task_contract.rs:179` | `manual_is_multiple_of` | Positive-divisor is_multiple_of preserves existing limit/refusal checks. |
| C037 | `crates/jope-engine/src/task_contract.rs:183` | `manual_is_multiple_of` | Positive-divisor is_multiple_of preserves existing limit/refusal checks. |
| C038 | `crates/jope-engine/src/teaching_store.rs:139` | `type_complexity` | GroupLabels/PublishRequest and exact tuple aliases; qualification and complete receipt checks remain active. |
| C039 | `crates/jope-engine/src/teaching_store.rs:129` | `too_many_arguments` | GroupLabels/PublishRequest and exact tuple aliases; qualification and complete receipt checks remain active. |
| C040 | `crates/jope-engine/src/teaching_store.rs:274` | `too_many_arguments` | GroupLabels/PublishRequest and exact tuple aliases; qualification and complete receipt checks remain active. |
| C041 | `crates/jope-engine/src/teaching_store.rs:303` | `too_many_arguments` | GroupLabels/PublishRequest and exact tuple aliases; qualification and complete receipt checks remain active. |
| C042 | `crates/jope-engine/src/teaching_store.rs:328` | `too_many_arguments` | GroupLabels/PublishRequest and exact tuple aliases; qualification and complete receipt checks remain active. |
| C043 | `crates/jope-engine/src/teaching_store.rs:1040` | `type_complexity` | GroupLabels/PublishRequest and exact tuple aliases; qualification and complete receipt checks remain active. |
| C044 | `crates/jope-engine/src/smartcluster_worker.rs:335` | `dead_code` | Neural-only fields/accessors use the existing neural feature; CPU stop/custody storage and receiving checks remain active. |
| C045 | `crates/jope-campaigns/src/hyperion_gate.rs:3671` | `too_many_arguments` | FoldEncodingScope groups unchanged fitting scope identities. |
| C046 | `crates/jope-engine/src/controller.rs:1342` | `zombie_processes` | ProposalOptions/TrainingOptions group inputs; parent-death fixture explicitly waits for its child. |
| C047 | `crates/jope-engine/src/native_loop.rs:1843` | `too_many_arguments` | ParentEvidence groups unchanged checkpoint/train/comparison evidence. |
| C048 | `crates/jope-engine/src/native_observer/science.rs:496` | `too_many_arguments` | GraphLocation groups unchanged root/state/worker. |
| C049 | `crates/jope-campaigns/src/recovery_contracts.rs:1220` | `too_many_arguments` | RollbackBundleEvidence groups unchanged bundle hashes, files and artifacts. |
| C050 | `crates/jope-engine/src/eval_registry/authority.rs:4089` | `too_many_arguments` | FitInputs and CompletionArtifacts group payloads; verified process/runtime/fit and completion checks remain in their receiving constructors. |
| C051 | `crates/jope-campaigns/src/selfplay_shadow/policy.rs:392` | `too_many_arguments` | EvaluationIdentity groups evaluator and protocol. |

## Additional reached surfaces

The default workspace traversal also revealed five Hyperion library diagnostics. The progress-event fields and patch quantization inputs are grouped without changing projection, fitting, row alignment or numeric operations. The explicit floating-point comparison preserves refusal for NaN and non-positive bootstrap bounds. CPU-only candidate construction still returns exactly the CPU candidate; CUDA candidate order and load policy remain unchanged. Padding uses the same repeated byte.

The resident-protocol integration fixture imports the public production library module instead of compiling a private copy. That source file contains no separate test declarations. The unused `Fixture::strict_receipt` helper in `corpus_incremental` had no callers; its removal does not remove a test or the separate admitted teaching-receipt helper.

## Verification and evidence scope

Use the repository's unchanged `just required` gate. Focused strict lint commands are `cargo clippy -p jope-parallel --all-targets --locked -- -D warnings`, `cargo clippy -p jope-dime-regressor --lib --locked -- -D warnings` and `cargo clippy -p hyperion --all-targets --locked -- -D warnings`. Neural-feature compilation is a separate required check for the newly gated fields. SDK discovery does not establish compilation, linking or device execution.

Run original authority failure, identity, custody, deadline and authenticated service fixtures together with the authorized payload-retention test. Preserve the existing PR-profile exclusions and per-test timeouts. Run the ordinary product PR tier and compatibility/historical suites, without regenerating behavioral golden fixtures. Source-bound lineage identities must be issued by the existing identity-only compatibility producer from a committed source head, with an append-only historical transition and an explicit PR-body disclosure.

The first draft receiving-path batch selected 773 tests and reached its original 900-second batch limit: 235 passed, two were terminated and 536 were unexecuted; 18 were excluded by the existing profile. This is incomplete evidence, not a passed test lane. Subsequent complete runs retain that outcome and their own source identity, counts, exclusions, timestamps, resource limits and exit status.

Local source, logs, inventories and independent review receipts are retained in the owning lane's external evidence store. Hosted checks must bind the final full head, run ID and attempt. Fixture proof, hosted CI, live GPU/distributed qualification and serving promotion remain separate states. This change authorizes none of the latter operations.
