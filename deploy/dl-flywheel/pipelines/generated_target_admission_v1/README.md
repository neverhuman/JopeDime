# Successful generated variant target-reference export: source revision 2

Revision 1 is held for the two findings recorded in `provenance/v1-hold.json`; its exact packet and qualification attempts are preserved as historical evidence. This sibling revision pins the campaign weight exactly and closes the offline runner's fixture, source, import and result gates. Older PASS receipts do not qualify this revision.

This is a local, source-only first step for closing the generated-result feedback gap. It exports exact, verified result references for **already admitted, authoritative train-role variant contexts** into an append-only copy of the current variant extension. It does not prepare tensors or train a model. No live result has been exported with this packet.

The result is a canonical JSON byte string whose SHA-256 identifies the immutable target-reference manifest. The caller may retain these bytes create-only in its separately owned proposal directory. There is deliberately no publishing, database writer, broker, evaluator, optimizer, checkpoint, service, or epoch-boundary adoption entry point here.

## Scope and stop conditions

- Read only a separately captured, standalone physical SQLite backup. Require its full SHA, exact config, event count/head, observation count and batch count. This package does not capture the live database or take its writer lock. Never pass a live WAL main file to `immutable=1`; a separate qualified owner must produce the backup at one SQLite snapshot and close it. Any WAL, SHM or rollback-journal sidecar is refused.
- Require exact prior extension, objective, original native training-normalizer bytes/fit roster, KumoSmall-only 2,051-coordinate roster and model-weight identity. Preserve every previous context, non-result field, and ordered reference prefix.
- Only `published_verified` or `cached_verified` variant rows can be candidates. Check the emission's exact context, program, protocol, source campaign, parent lineage, inherited role and phase. Native-context results remain explicitly unsupported by this packet.
- Fetch each new wrapper and all seven artifact roles at its exact bucket/key/version/hash/length. Require the pinned, reviewed current verifier's structured identity-bearing return. A Boolean success, extra invented target fields, wrong context, stale version, bad bytes, parity rejection or changed role fails the entire export; no partial success manifest is returned.
- Export no validation target. Preserve validation, pending, refusal and duplicate denominators. Unknown/leased/unsupported attempts are never retried or turned into failure labels.
- Canonical identity is the existing evaluation key over the complete `Context` and canonical program SHA. Multiple emissions of the same canonical target contribute one result reference and retain all their source emission IDs. A conflicting reference for an accepted or previous canonical key fails closed. This conservative v1 does not resolve alternative equivalent wrapper versions.
- This package has no new q formula. `numeric_targets_materialized=false`, `variant_ce_teachers=0`, `optimizer_updates=0`, `admission_committed=false` and `source_optimizer_commits_verified=false` remain explicit. Exporting a successful result is not evidence that its originating optimizer batch committed, that the result became an optimizer target, or that later training consumed it.

## Read-first files and boundaries

1. `common.py`: exact JSON and references, source/loaded-function checks, monotonic budget.
2. `cutoff.py:capture_cutoff`: hash-bound immutable SQLite verification, event replay, materialized state and ACK closure.
3. `export_verified.py:export_targets`: bounded role/identity/closure verification and full census.
4. `transition.py:validate_extension_transition`: pure append-only old-task/reference contract.
5. `tests/reference_outbox.py`: exact historical outbox implementation copied only for synthetic fixtures. Its constructor creates fixture databases under temporary directories; production exporter code never imports it.
6. `tests/fixture_callbacks.py`: explicitly synthetic contract adapters, not native score verification.
7. `run_offline.py`: isolated, portable runner for this family's new fixtures only.

No existing source, source manifest, CI classification, runtime, live database, artifact, or checkpoint is edited by this proposal. Root owns review and additive PR integration. A later production callback adapter, live cutoff capture, numerical preparation, and repeating epoch-boundary adoption must each be separately qualified. Do not add these implicitly to this packet.

## APIs and immutable inputs

```python
cutoff_raw = capture_cutoff(
    absolute_standalone_backup,
    expected_sha256=backup_sha256,
    expected_config=exact_outbox_config,
    expected_head={
        "events": event_count, "head_sha256": event_head,
        "observations": observation_count, "batches": batch_count,
    },
)

manifest_raw = export_targets(
    request_raw,
    expected_request_sha256=request_sha256,
    cutoff_raw=cutoff_raw,
    predecessor_raw=previous_extension_bytes,
    objective_raw=current_objective_bytes,
    normalizer_raw=original_preprocessing_bytes,
    get_version=reviewed_read_version,
    canonicalize=reviewed_canonical_program,
    verify=reviewed_successful_variant_verifier,
    authority=reviewed_current_role_authority,
)
```

The closed request schema is `jope.generated-train-target-export-request/v1`. It contains the four input SHA fields (`cutoff`, `predecessor`, `objective`, `normalizer` with `_sha256` suffix); `native_train_context_ids_sha256`; exact `construction_sources.labels` and `.compose` descriptors; roster `{name: "kumo-s-only-v1", dimension: 2051, weight_sha256}` with the **only permitted weight SHA** `fc672f536501d64356036050ac2acea7e46927f7b9ad38366abe048f88e7d423`; all prior admitted descriptors; allowed source campaign SHA values; authority source descriptors; four callback bindings; and finite limits. `tests/test_export.py:ExportTests.build` provides a complete **synthetic-only** request example.

Each source descriptor is `{path, sha256}`. Each callback binding is `{path, sha256, qualname, dependencies}`. Callbacks must be top-level Python functions without closures or runtime defaults. The exporter checks the defining file, qualified name, structural equality of the loaded function's code to compilation of the pinned bytes, and the declared dependency file hashes before and after use. It does not sandbox Python or attest arbitrary module globals, native modules, dependency execution state or external services. A source/runtime-qualified adapter and an independently approved request remain prerequisites; merely pinning an arbitrary callback does not qualify its behavior.

The `get_version(ref)` callback returns `(exact_bytes, actual_version_id)` and may only perform bounded exact-version reads. Its adapter must enforce its own network timeout and response-size limit **before buffering the body**. The exporter checks budget before dispatch and immediately after return, then checks exact bytes/version/SHA. In-process cooperative deadlines cannot interrupt a hung callback. Future runtime qualification therefore also requires an external hard deadline and memory cap; this packet contains no service launcher.

The `canonicalize(raw_program)` callback must use the pinned native canonicalization contract without evaluating or fitting the program. Different raw whitespace may map to one canonical DSL. The `verify(ctx, dsl, entry, bodies)` callback must invoke the existing qualified variant `program_gate.verified_measurement` against a read-only body store containing exactly the seven already fetched artifacts; preserve the pinned native score canonicalizer. Its return must have exactly:

`evaluation_key`, `program_sha256`, `context_id`, `measurement`, `status`, `teaching_eligible`, `evaluation_sha256`, `artifact_versions`, `inner_fold_rmse`, `canonical_score_row_sha256`.

The exporter independently joins each identity, artifact version, finite two-fold RMSE and score-row SHA list. The callback remains responsible for the existing native input/target/fold reconstruction, emitted Rust parity and canonical score semantics. The synthetic callbacks in this package do not provide those scientific checks and must never be supplied to a live request.

The authority callback independently reads the exact pinned current role sources, validates each original admitted descriptor and parent role, and returns `jope.generated-target-current-authority/v1`, `state=passed`, current timestamp, request SHA, source pins, complete descriptors/parent roles, objective/normalizer/fit-roster pins and roster. The exporter rejects bare success, stale/future proof, missing roles, role drift and mismatched bindings. It calls authority before selection and again before producing output. A source-pinned callback that simply echoes a request is not a production authority check; the fixture callback does that only to exercise the return contract.

## Existing target semantics retained for the later preparation step

The current variant composer keeps `teacher_rule=disabled_for_unpromoted_variants/v1` and CE disabled. Its native mean baseline is reconstructed from the opposite inner fold's actual targets; the current labels implementation derives epsilon, normalized log-RMSE quality, clipping/quantization and support masks. The existing objective remains `native-support-mask-v1` with coefficients CE 1, BT .25, list .10, quality .25, feasibility .10 and runtime .05. This exporter binds those files and the complete current objective/preprocessing bytes; it does not copy, recompute or modify those equations.

For a previously admitted result, the exporter re-reads its exact wrapper/program bytes only to construct the canonical duplicate index. It preserves the exact prior reference and relies on the independently qualified predecessor for that old result's full seven-artifact custody. It does not claim to requalify every prior artifact. Every newly appended result gets all seven exact reads and the pinned structured verifier.

After separate successful numerical preparation, real q/ranking targets can affect the quality head and shared representation. This proposal itself provides no gradient or learning-gain evidence, no reward-weighted actor update, no CE promotion and no RL training. A later version must prove physical checkpoint/AdamW/RNG continuity, immutable prepared-candidate identity, retained old membership/roles and actual epoch-boundary adoption before any optimizer claim.

## Limits, census and failure interpretation

The first export permits at most 16 selected contexts, 64 unique new rows, 4,096 artifact reads, 512 MiB of declared attempted artifact bytes and 120 seconds. Per-artifact caps follow the existing seven-role verifier. Source files, JSON inputs, cutoff rows/events and manifest output also have hard bounds. The cutoff payload bound is applied per table plus final output; it is not an RSS guarantee. All reads are synchronous; no retries or absence caching occur here.

Selection follows the cutoff's deterministic emission-ID order, not measured reward. Canonical aliases are resolved before the new-target count limit. Exceeding the selection count gives an explicit pending reason while preserving the denominator. Exhausting the exact-read byte/count/time budget, an integrity check or a callback raises and produces no manifest. A caller must preserve a separate failure receipt and the original immutable inputs; it must not relabel this as a partial verified export.

The final manifest schema is `jope.generated-train-target-reference-manifest/v1`. It binds request/cutoff head/predecessor/objective/normalizer/construction/roster; preserves the new extension and its SHA; contains one canonical row per appended target, both role proofs and artifact read counters; and accounts for every captured emission once in `appended`, `duplicates`, `validation`, `pending` or `refused`. Validation entries are excluded without re-fetching their result bodies; their census is source-state evidence, not fresh scientific requalification.

## Offline qualification and handoff

Run only this new family:

```sh
rtk proxy python3 -B deploy/dl-flywheel/pipelines/generated_target_admission_v1/run_offline.py \
  --receipt /absolute/new-owned-directory/qualification.json
```

The runner restores imported package names and `sys.path`, disables bytecode writes, rejects duplicate fixture identities, requires the exact frozen 28 fixture identities and count, and checks all 11 Python source pins plus exact imported module origins before and after execution. `SOURCE_PINS.json` is itself bound by the external source packet; it is not a signature or a substitute for reviewing the source packet. Expected failures, unexpected successes, skips, ordinary failures/errors, count drift, source drift, import-origin drift and precompiled package imports are all failures. It records timing and test text in JSON. No raw `.log` artifact is required. All SQLite mutations in fixtures are confined to freshly created temporary databases; exporter SQL is read-only. Tests use the copied exact historical outbox implementation to produce schema-real events and ACKs, with fake S3/native bodies explicitly labeled synthetic.

Completion for this source packet means the new fixture receipt passes and independent review is hash-bound. Root then adds only this package's production/research classifications, one test family and its exact new fixture IDs without repeating old suites. Live capture/export/preparation/admission remain unqualified and unexecuted. Keep failed qualification attempts and final receipts separately; do not rewrite their historical pins.
