# Producer → export → consumer contract (draft v1)

This implements the handoff detail missing from H88/H89, not its execution.
All identities below must be explicit immutable content bindings. A path name,
numeric signature, non-null metric or schema-valid JSON is not provenance.

## 1. Produce a training-only evidence bundle

The producer/export owner supplies a create-only, versioned S3 manifest and
SHA-256-pinned objects under the authorized W3 experiment prefix. Its manifest
must bind these fields and their schema/version meanings before any row import:

| Manifest field | Required contents |
| --- | --- |
| `producer` | Run ID, exact source/evaluator/vocabulary/metric-gate hashes, command, input identities and execution interval; for a derivation, proof that this run used these bytes, plus derivation code and output hashes |
| `closure` | Exact H65 fold identity; per-context original training table and row-membership/cut hashes; current family/table/alias exclusion inventory and reviewer receipt; zero DL-Val, outer, sealed and holdout rows |
| `pool_inventory` | Every expected context, pool byte hash, row count and ordered-program hash, bound to the admitted H50 source identities; a changed cohort requires an explicit amendment/comparability receipt |
| `objects` | Bucket/key/non-null VersionId/SHA-256/byte count for allowlisted completion rows, numerical arrays, producer evidence and join receipt; exact-version restore hashes |
| `census` | All expected rows partitioned into completed-pass, completed-gate-fail, timeout, error and unknown; measured/missing counts for each component, all exclusions and duplicate/foreign/missing join counts |
| `review` | Reviewer/source SHA, admission disposition and evidence hashes; no `admitted=true` supplied by an unchecked producer is sufficient |

Use [completion-row.schema.json](completion-row.schema.json) for each candidate
row. `row_index` is zero-based in the original frozen pool. `completion_basis`
references a **field-allowlisted** authoritative event or verified run-bound
derivation receipt, with an exact event/row locator. Unknown rows may have null
basis. The reference's shape does not grant permission to fetch it; the separate
closure review must approve the exact object first. This schema selects S3
custody; an approved mirror must be rebound to the same verified bytes before
use. Do not pass a raw mixed-role source to this consumer.

## 2. Join and census before admitting components

The future consumer must strictly parse JSON (reject duplicate keys, NaN and
Infinity), validate every row, then do these checks in this order:

1. Verify the manifest and every approved input object's exact version, hash and
   byte count. Verify producer run binding and current role/alias closure.
2. Compare the complete expected context/pool inventory to the pinned fold/H50
   inventory. For each pool require exactly `n` row records with indices `0..n-1`.
   Join by `(context_sha256, pool_sha256, row_index, program_sha256)`. Recompute
   the ordered-program hash using the existing pool export's canonical encoding;
   freeze that encoding in the manifest, never infer it from arbitrary JSON.
3. Reject duplicate keys, foreign contexts/programs, inconsistent family/cut/
   evaluator/vocabulary/metric identities, mismatched order/hash, and missing
   census entries. A known missing producer event becomes an explicit `unknown`
   row with null components; a missing export row is a failed export, not zero.
   Retries require one prospectively specified event-selection rule and retained
   attempt accounting; do not choose the best observed outcome per program.
4. Verify every completed label against its exact producer evidence. A source
   code path capable of producing values does not prove that this run completed
   that path. Signature-only H86 labels must be rejected as completion bases.
5. Compare every admitted component to its allowlisted source value and record
   masks and exclusions. Enforce finite numbers outside JSON Schema too.
   Reject any numerical/component mismatch. Only then freeze the export hash,
   join/census receipt and reviewer disposition for the training worker.

The original frozen cohort census is 1,219 contexts / 205 families / 1,404,287
pool rows. These are inventory expectations, **not completed-row counts**.
If rows are excluded, retain the full census and report effective exposure by
context/family/fold/status/component. Do not silently restrict comparisons to
surviving families or renormalize into a full-panel result. No common support
means untested; an effective-cohort change needs a comparability decision before
training. Current closure can invalidate a historical row even if its old
metadata called it DL-Train.

## 3. Target admission rules

| Recorded execution state | Permitted supervision |
| --- | --- |
| Completed pass | Recorded gate=true; measured finite logratio; passing residual only when both fitness and logratio are actually measured |
| Completed gate failure | Recorded gate=false and actually measured finite logratio; never reconstruct quality or passing residual from a failure penalty |
| Timeout/error/unknown | No W3 quality, residual or gate target; keep all training component masks false and values null |

These are permissions for components that exist, not a requirement to invent a
missing metric. A completed row can lack a component; mask it. Partial values
from failed executions may remain in approved producer evidence but are excluded
from this candidate supervision contract. Unknown is neither gate=false nor a
zero target. H86 masks remain diagnostics beside this ledger, never its source
of completion truth. Preserve original gate and utility semantics throughout.

## 4. Bind the downstream implementation and readout

H65 `load_winner_config` accepts any JSON object; `make_plan` emits
`preparation_only=true` and `gpu_or_native_fit_requested=false`. Before extending
it, require a measured accepted W1 result, exact frozen winner/model/recipe hash
and the H59 no-admission disposition. A hash of an opaque config is not a winner.
Register W3 a/b/c/d, width/compute-matched scalar and shuffled-auxiliary controls,
fit-family-only calibration, seeds 701–703, five folds and 40 terminal updates.
Record actual optimizer updates/changed parameters and paired training exposure,
including exclusions. Prove loss-unit changes preserve zero-update selections.

After an admitted run, produce H74 evidence with `closure.admitted_export_sha256`,
`fold_manifest_sha256`, `winner_config_sha256` and `h59_disposition_sha256`
bound to the reviewed objects. Convert context/seed outcomes into the required
complete 205-family paired panel without dropping failures or missing outcomes.
Run H74's `--evidence` check at the pinned implementation. Its output deliberately
has `source_provenance_verified=false` and `launch_authorized=false`; passing it
cannot replace the upstream provenance review or authorize another run.

Before launching, reconcile actual cumulative resource usage with the existing
W3 ceiling/deadline and qualification policy. This document reserves no compute,
creates no GPU queue ticket and changes no lock owner.

## Acceptance cases for the future admission validator

| Case | Required outcome |
| --- | --- |
| Verified completed pass with both metrics | Admit only measured components; bind row and source receipt |
| Verified completed gate failure with logratio | Admit measured quality plus false gate; mask passing residual |
| Complete census containing explicit unknowns | Retain rows, mask all their supervision; report lost support |
| H86 value signature relabeled as completion | Reject completion admission |
| Missing/duplicate/foreign export row, wrong hash/order/version | Reject export before training |
| Mixed-role object or unresolved current alias membership | Refuse payload admission |
| NaN/Infinity, duplicated JSON key, invented false/zero target | Reject |
| Changed cohort or post-outcome retry selection | Require preregistered comparability/selection disposition; no automatic admission |

The schema checks only some shape cases. These semantic cases are specifications
for the eventual implementation and remain unexecuted on real rows.
