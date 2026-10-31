# H111 — W7 common-support matching preparation

**CPU preparation only; every real common-learner receipt remains pending.**
This extends merged [H104](../h104_w7_acquisition_replay_prep/README.md), following
owner `rd_t0540.md`. W3 stays parked. H104 already supplied the acquisition
schedule; this change implements its missing family/status count matcher without
selecting a winning learner or acquiring new outcomes.

## Implemented boundary

[`matching.py`](matching.py) accepts a verified 20-fit-family cell and only these
attempt metadata fields: held fold, seed, arm, family, context/program SHA, explicit completion
status and completion-receipt SHA. Outcome/utility fields are rejected.
For each family × completed-pass/completed-gate-fail stratum, it takes the
minimum count across the three H104 memories, choosing by program SHA. Different
policies need not acquire identical programs. It reports every selected program,
matched count, discarded completed record and excluded timeout/error/unknown.
Selection ignores utility and preserves status balance within each family.

A complete cell must account for all 240 acquisition attempts: 20 families ×
four attempts × three arms. Missing attempts produce PENDING, with provisional
matching and `selection_final=false`. Retries/duplicate programs, a fifth attempt,
foreign fold/seed cells, families/contexts or malformed identities are refused. Explicit unknowns
are excluded, never zero-filled or treated as measured gate failure. Once the
census is complete, fewer than eight common families is UNTESTED; the original
batch-of-eight requirement cannot be reduced to obtain a favorable subset.
Otherwise the output is **CANDIDATE_MATCH**, never an execution or advance pass.

Every output states `source_provenance_verified=false`,
`cost_update_matching_verified=false`, `replay_schedule_implemented=false`,
`execution_admitted=false` and `advance_admitted=false`. Receipt hashes are syntax
and identity fields, not authenticated completion evidence. The caller must
separately establish exact-version producer evidence, current training-role and
alias closure, original rows/cuts/evaluator/vocabulary, and matched costs before
using a selection. A synthetic or forged receipt can satisfy a hash shape; this
helper deliberately cannot authenticate it. H88/H89/H107 provenance restrictions
remain applicable. No model, numerical outcome array, S3 client or training mode
exists in the matcher.

## Real metadata dry run

[`prepare.py`](prepare.py) verifies exact H104 source/config/fold byte hashes,
then calls H104's existing fold-closure validator and schedule builder. It emits
15 fold/seed cell contracts and 45 **pending** arm receipt slots, with unknown
matching counts and null receipt pins. The inherited five-fold manifest still
contains 1,219 contexts and 205 families; no held-family rows enter a fit cell.
The original H104 plan reproduces byte-for-byte at SHA-256
`28481a3b4be7bc28bb5711b88eb370684b9a0a3263fe387c603edee483bd75d2`.
The output distinguishes that file hash from its compact canonical-object hash.
The metadata-only S3 fold object was fetched at its recorded VersionId, SHA-256
verified and compared byte-for-byte with the merged H104 copy.

Four deterministic host shards (xbabe1 fold0, xbabe2 fold1, xbabe3 fold2,
xbabe4 Linux Docker folds3/4) concatenate exactly to the complete local cells
and pending panel. All 15 empty acquisition ledgers remain PENDING with 240
missing attempts each. No pending receipt was manufactured from synthetic tests.
[`dry_run_summary.json`](dry_run_summary.json) records input/source pins, hashes,
capacity-guarded host receipts and null scientific metrics.

Run from the repository root with an existing external evidence directory:

```sh
nice -n 15 python3 research/flywheel_selfplay/h111_w7_common_support/prepare.py \
  --dry-run --output "$W7_EVIDENCE_DIR/h111-matching-plan.json"
nice -n 15 python3 -m pytest -q -p no:cacheprovider \
  research/flywheel_selfplay/h111_w7_common_support/test_matching.py
```

The output path must not exist. `prepare.py` reads only committed identity
metadata, accepts no outcome ledger, and opens no network or dataset connection.
The pure matching API is available for later integration after separate admission.

## Remaining execution work and limits

The strong fold-excluded learner, initialization/recipe pins, admitted completed
outcomes, original/current role closure, finite archive, family-uniform replay
sampler, matched fit-cost/update/exposure receipts, timing/power pilot and full
scoring gate remain unimplemented or unbound here. Matching counts does not match
costs or train a common learner. An eight-family support count alone does not
establish the 40-update sampler's feasibility. No partial family/seed panel may
advance. H104's uniform4 acquisition interpretation still needs recipe freeze.

The original W7 ceilings stay 4,800 native fits, 14,400 CPU-seconds and 7,200
GPU-seconds, with 3,600 acquisition fits planned across all folds. This prep
reserves no additional fits and resets no campaign deadline or consumption.
Remaining authorized resources are unknown. DOPE stays frozen; no GPU/lock,
live-worker, model/weight, protected-row or serving action occurs.

Validation: 32 focused synthetic controls pass; 15 real-metadata empty-ledger
controls preserve PENDING across all hosts. These are integrity proofs, not
scientific observations. Actual fits, optimizer updates and GPU use are zero;
Elo, val_reg_inner and U4 remain null. See the READY PR for exact-head checks,
S3 custody and the separately disclosed unchanged-main source-inventory failure.

## Acquisition, archive and replay receipt preregistration

[PREREG.md](PREREG.md) binds the H104 design to eight required producer artifacts per arm/cell. [pending_manifest.json](pending_manifest.json) enumerates all 45 PENDING slots and exact source/S3 metadata identities; [manifest.schema.json](manifest.schema.json) describes this preparation format. Archive retention is separate from H104’s implemented acquisition rule. All receipt references remain null, costs/deadline remain unknown, and schema validity grants no execution or advancement. No new wave ID, fits, updates or GPU allocation is introduced.

Validation for this continuation: the JSON Schema and pending manifest pass Draft 2020-12 validation. Four deterministic CPU host shards reconstruct the H104/H111 family/context bindings, cover all 45 unique arm slots exactly once and reconcile 3,600 planned fits with the full manifest. All 360 producer references remain null. Exact-version S3 fold metadata matches the committed copy. Measured worker CPU is 0.120360475 seconds; transport and local checks are not included. Actual fits/updates/GPU remain zero. The READY PR records exact-head checks and verified S3 custody.
