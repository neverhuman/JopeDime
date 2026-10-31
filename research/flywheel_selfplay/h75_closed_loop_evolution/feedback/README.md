# H75 completed-outcome feedback writer

CPU support owned by `sol_v1_obj02` under `obj02_t2312.md`, following ASTRA's
[feedback contract](../INTERFACES.md) and [preregistration](../PREREG.md).
OPUS owns runner integration and canonical feature joining; GOV03 owns scoring.
This helper writes target metadata. It does not train, select a slate, load
datasets/checkpoints/tensors, invoke an evaluator, use a GPU, acquire
`qualification.lock`, or update a shared HWM index.

`build_feedback(records, admission)` returns program-keyed records in separate
run/cell/fold/seed/context partitions. It retains **all** supplied execution
identities, including completed negatives and execution errors, and refuses a
missing member of the independently supplied execution ledger. Exact retries
deduplicate; conflicting payloads for the same stable record ID refuse.

Completed measured rows map raw `fitness` to the target when the gate passes,
otherwise `2.0`. Original fitness, logratio and status remain unchanged.
Only D's completed rows with feature provenance are `eligible_for_loss`.
A/C are shadow records; B is search memory only. Failed/timeout/unknown rows
and completed rows missing feature provenance have `target=null`, no loss
eligibility and `memory_use=audit_only`. Finite sentinels cannot create a
completed status. An apparent HWM means completed gate pass below the supplied
same-context pre-cycle incumbent minus `1e-6`; it is not confirmation or promotion.

## Frozen admission sidecar

The caller independently prepares and freezes one JSON admission object per
run/cell/fold/seed. It contains:

- `schema=jope.h75-closed-loop.feedback-admission/v1`; `run_id`, full
  `source_git_sha`, `prereg_sha256`, `execution_manifest_sha256`, `cell`, `fold`,
  `seed`, `dope_sha256`, `fold_manifest_sha256`, and `selection_policy`.
- `forbidden_families` and `forbidden_alias_components`: explicit nonempty
  scoring-family/alias rosters, never guessed from filenames.
- `fit_contexts`: context SHA → DL-Train `role`, `family_id`, `family_fold`,
  `alias_component`, `table_sha256`, `original_rows_sha256`, `cut_id`,
  `evaluator_sha256`, `vocabulary_sha256`, `evaluation_protocol`, `programs`,
  and `pre_cycle_best_fitness`.
- `programs` maps every admitted proposal **and parent** program SHA to its
  canonical feature-source SHA, or null when feature provenance is unavailable.
  It contains hashes, not feature arrays. `pre_cycle_best_fitness` maps the
  cycle string (`"1"`/`"2"`) to that context's original-cut incumbent.
- `source_objects`: exact `{key,version_id,sha256}` provenance allowlist.
  `record_sha256s`: complete stable execution ID → canonical outcome SHA256.
  This ledger includes errors and negatives, not just winners.
- `outcome_exports`: independently admitted clean JSONL byte SHA256s. The CLI
  checks this allowlist and validates admission **before opening outcome bytes**.

Outcome fields are exactly the clean required v1 fields in INTERFACES.md.
Additional fields, including legacy outer-test fields, refuse. Preserve costs
and other native diagnostics in the execution/source artifacts referenced by
the pinned manifest; supply a separately admitted clean export to this helper.
It never fetches or opens legacy mixed-score bodies and then filters them.

Canonical record SHA256 is over UTF-8
`json.dumps(row,sort_keys=True,separators=(",",":"),allow_nan=False)`.
DSL hashes are over the original UTF-8 DSL string, with no normalization.
Duplicate JSON fields and nonfinite JSON constants refuse. The admission copy
is isolated from caller mutation; the output includes its canonical digest,
the actual preregistration digest and this writer's source-file digest.

Frozen pins are DOPE `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`,
fold manifest `dd211134bba127960b1612687478064969dca422e35dfa889790760f316961c1`,
and evaluator `e3be938644722d14a1c1a0e82168518838c4d174cfe031076a339745c48870a8`.
The caller pins the current preregistration in admission; row-to-manifest
differences refuse. `reference_protocol_sha256` records the initial contract's
`d0083efb6dab8fa9ce2f065da37471b20a180b21119da2c3ea1d3c7b312c8c5f`
as a reference, so a schedule-only document amendment does not silently choose
or authorize any GPU slot. Seeds/cycles/folds and feedback semantics remain
those of the accepted v1 interface.

## CPU use and acceptance

From the repository root, with explicitly admitted paths and byte hashes:

```sh
nice -n 10 python3 -B research/flywheel_selfplay/h75_closed_loop_evolution/feedback/feedback.py \
  --admission /admitted/admission.json --admission-sha256 ADMISSION_SHA256 \
  --records /admitted/clean-outcomes.jsonl --records-sha256 OUTCOME_EXPORT_SHA256 \
  --output /owned/new-feedback.json
```

The POSIX writer fsyncs a private temporary file, installs the complete artifact
with a create-only hard link and fsyncs its directory. Existing files or symlink
targets are never overwritten; output reuse refuses with exit 2. Exact record
retry IDs stay stable. OPUS must verify an existing artifact's pinned bytes
when resuming, not invent new outcome identities to bypass output refusal.

```sh
nice -n 10 python3 -B -m unittest discover \
  -s research/flywheel_selfplay/h75_closed_loop_evolution/feedback -p 'test_*.py' -v
```

The tiny synthetic development fixtures verify software only. Real H75 outcomes
require the caller's admitted S3-derived data and execution manifests. No study
result, optimizer update, Elo, selection lift or data qualification is implied.
The helper validates supplied metadata; it cannot independently establish its
creator's hidden source-data closure or verify remote object bytes. OPUS must
join canonical feat412/proj16 by program identity, check tensor alignment and
keep fresh targets out of revealed training anchors/feature channels. Actual
training exposure and the separately preregistered common-learner replay remain
runner obligations; `eligible_for_loss` describes direct D-cell feedback only.
