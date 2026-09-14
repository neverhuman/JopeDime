# Admission and completion integration — 2026-09-10

Owner: Codex admission lane. Local source implementation only; this document is not a deployment, corpus-admission, training, or promotion receipt.

## Production contract

`teaching_store::publish_admitted` publishes `strict_midrank_v4` under a new store lineage. Legacy `corpus::build*` and `teaching_store::publish` reject production use; reference exports remain available with explicit qualification. Existing observations and already published stores are not rewritten.

The frozen `parallel.label_admission.v2` policy must bind:

- `evaluator`, `evaluator_sha256`, and `evaluator_protocol` (`featmath_training_fold_normalized_v1`).
- `runtime_sha256` and `numerical_contract_sha256` from the approved loaded numerical runtime.
- Workspace-relative `authority_root`, `authority_witness`, and the approved `authority_instance`.

Missing or malformed fields, missing authority state, a changed approved instance, or unprovable committed authority membership fail closed. Collection tooling obtains these identities from explicit `fit_*` controller/task-manifest fields; it does not invent them.

## Original fit and evaluation evidence

Each observation carries a content reference in `clean_completion`. Its `parallel.clean_completion.v3` artifact binds the original attempt, program and evaluation identities, evaluator, original process exit/signal/deadline, evaluation elapsed time, and original result artifact. Successful labels require an observed exit 0 without signal or deadline violation, and exact agreement between retained result JSON and observation metrics.

The wrapper also references the original `parallel.fit_completion.v2` receipt and original retained model. The native receipt is checked against the approved authority terminal entry, original consumed token and process identity, retained journal and independent high-water witness. Self-hashed JSON is insufficient. The retained model must still match its original digest; missing artifacts do not authorize refitting.

The numerical training identity is recomputed from ordered prepared f64 features and targets once per admitted context. It is distinct from the prepared CSV-byte hash. Admission compares the original fit identity's preprocessing, feature view, fit seed, runtime, and numerical contract. Regression is the only admitted task. Actual prepared views and preprocessing are verified; sealed meta-test contexts are rejected before loading their rows.

`runtime_seconds` is the original evaluation-wrapper elapsed time. `numerical_runtime_seconds` must equal the original native fit elapsed time and supplies the derived runtime target. Unknown/unavailable durations remain null. Failure storage preserves evaluation and numerical elapsed/censor fields separately. A result file cannot clear failure or timeout.

The Python label boundary invokes the pinned native `verify-fit-completion` command to verify approved authority membership, then compares the returned native receipt. Its tests mock that subprocess to isolate Python contract logic; native authority admission is tested separately with committed synthetic fixtures.

Direct-proposal evaluation now freezes the approved admission policy before dispatch and retains its own original supervisor attempt, observed exit/signal, absolute deadline, output and clean-completion wrapper. A valid-looking score cannot clear a failed exit or timeout, including a process that exits successfully during timeout cleanup. Quality requires declared regression/normalized-RMSE units and the existing committed-native completion/model verification. CSV-backed prepared views independently reconstruct the native ordered-f64 training digest; compact views are unavailable before dispatch until an independently verified numerical-input/query protocol supports them.

Direct receipts distinguish logical slots, dispatched requests, verified physical starts, registry reuse, and unknown consumption. They report `physical_evaluations: null` when consumption is unresolved rather than equating unique requests with fits. Verified starts require original native task attribution to the current output path; reuse does not add a physical start. Reservations cover fitting, cleanup and completion verification without refunding ambiguous consumption. Failed or unknown selections remain in nomination panels.

This direct path still lacks a separately qualified registered prediction/query receipt and complete pre-outcome selection evidence. Its receipts explicitly declare `registered_query_receipt_qualified: false`; they cannot satisfy production promotion. Positive Python branch tests mock committed verification explicitly, while fake executable results, uncommitted/missing evidence, wrong units, late exits and task-attribution conflicts produce no quality. These tests do not perform numerical fits.

## Derived labels and publication

Exact original f64 RMSE ties share midranks, including zero RMSE. No f32 rounding establishes ties. Historical observations retain their bytes; changed derived labels acquire a new label policy and store binding.

Store bindings include corpus, grammar, teaching store, task contract, preparation, encoding, manifest, admission, and native authority sources. The manifest exposes the semantic dependency digest. Changed semantics require a new store lineage.

All supported source occurrences are retained in admitted label provenance. Conflicting success/failure or score evidence quarantines the program before deduplication. Registry reuse rows are retained in a separate table and cannot create teaching observations or fresh-growth counts. The current conservative policy also excludes fit reuse that lacks a separately qualified new-query evaluation receipt.

## Verification and remaining gates

`parallel/runs/admission-integration-20260910-local/complete.json` is the source/binary-bound local test receipt when its status is passed. It includes synthetic authority-backed publication, fabricated-receipt rejection, unit/cut/runtime conflicts, failed/unknown outcomes, exact ties, cache exclusion, legacy production fencing, reference-export parity, and Python consumer checks. Test binaries have debug symbols stripped solely to reduce hashing cost; their hashes and original build paths are recorded. No numerical fits or GPU tasks are part of this lane's test protocol. Earlier interrupted or intermediate checks are not represented as final passes.

The test-only authority fixture explicitly synthesizes a committed model/exit observation; it tests admission membership and data handling. Original Child exit observation is covered by the separate native registry subprocess tests. Full release CI, representative-history performance, historical coverage/import, live publication, GPU operation, actual fresh optimizer exposure, scientific qualification and both learning cycles remain separate release-owner gates.
