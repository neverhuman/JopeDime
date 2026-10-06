# Publish S1 GP high-water-mark evidence

The additive adapter `deploy/dl-trainonly/hwm_publish.py` stores the S1
collector's measured selection results in the learning bucket. It reuses the
frozen, reviewed #224 `Publisher`, `PublishLedger`, create-only S3 guard, shared
40 GB byte-budget ledger and exact-VersionId readback verification. It does not
change the frozen lane manifest, encoding publisher or running jobs.

The publication is **progress evidence only**. It issues no teacher grant,
compares no new HWM challenger, changes no serving checkpoint and asserts no
transfer improvement. A `selection_reached` result retains the measured
inner-CV geometric mean normalized RMSE ratio and the separate selection
fitness. A `g0_not_ready` result is a terminal record, never an HWM. Unresolved
G0 attempt-accounting and sampling-resume qualifications remain explicit.

## Source and scope

The canonical current source is the DLTrain `receipts/trainonly-v2/s1-prod`
collector: its frozen `ledger.header.json`, two append-only final ledgers,
hash-matched immutable `contexts/<context_sha256>/final.json` and opaque sealed
summary hashes. The initial and joined input lists are authenticated from the
header and committed join records. Dataset family and permitted use are checked
against the separately pinned P3 registry, assignment and denylist. Unknown,
heldout and sealed families fail. Development contexts retain `selection_only`.
The S1 stripped-input digest, two-fold protocol, cut identity, admitted binaries,
selected DSL digest, scores and charged/finished attempts must agree across the
independent input and collector records.

Only an explicit JSON metadata projection is uploaded. No JTT1, tar members,
row arrays, labels, predictions, outer scores, original final JSON or arbitrary
source strings are uploaded. Summary bytes are hashed without extraction.
Unknown source fields are omitted; a populated selection test-score field is
refused. Every path component is opened with `O_NOFOLLOW`; missing artifacts,
changed dependencies, ambiguous families and torn ledger snapshots fail closed.
The bytes decoded and hashed are the bytes consumed.

Production results use `experiments/gp-policy-dl/v2-trainonly/gp-hwm/`; terminal
records use `gp-terminal/`. Research results, including their completion
manifests, use the corresponding paths **inside** `v2-trainonly/research-only/`.
Content-addressed keys preserve dataset, fold and context identity; payloads
preserve the full cut identity. Different cuts never inherit an earlier score.
Each rights-specific manifest contains exact object digests, sizes and S3
VersionIds. Dependencies are verified before its immutable manifest is written.
There is no mutable `latest` pointer or overwrite/delete fallback.

The historical `research_2k/per_fold_hwm` campaign selected on outer-test fitness.
Its 2,834 views and 600 archived engine snapshots cannot be requalified as
train-only by stripping fields. The `hwm-extra` file contains evaluator attempt
receipts, not one HWM per row. Their inventory remains historical custody; this
adapter refuses that source schema. Any historical S3 preservation needs its
own reviewed diagnostic disposition. This publication is not a complete replay
package for private fitted artifacts: original source custody and fitting inputs
remain separate prerequisites.

## Freeze, dry-run and publish

Use the locked Python environment with the `dl-custody` extra. Keep all plans,
receipts and resumable state outside working trees and outside input custody.
The example shell variables name paths; they do not acquire credentials or a
compute grant.

```bash
python deploy/dl-trainonly/hwm_publish.py freeze \
  --source "$S1_COLLECTOR" --header-sha256 "$ADMITTED_HEADER_SHA256" \
  --output "$EVIDENCE/plan.json"
sha256sum "$EVIDENCE/plan.json"
python deploy/dl-trainonly/hwm_publish.py publish \
  --plan "$EVIDENCE/plan.json" --plan-sha256 "$FROZEN_PLAN_SHA256" \
  --dry-run --budget-snapshot "$V2_BUDGET_LEDGER" \
  --work-dir "$EVIDENCE/dry-run" --receipt "$EVIDENCE/dry-run.json"
```

The CLI requires the plan's exact SHA, supplied by the operator's frozen
publication request. The ledger snapshot binds an exact newline-terminated
prefix; later producer appends are allowed, but cannot silently join this set.
Changed earlier bytes or final artifacts fail. Freeze a new plan for new results.
Dry-run builds no AWS client and charges only a copy of the existing budget
ledger. It exercises the same create-only guard and exact-version readback in a
fresh in-memory bucket. Use a fresh dry-run work directory each time: an
in-memory bucket does not survive process exit.

After review and merge, the authorized DLTrain publisher uses `--real` instead
of `--dry-run`, omits `--budget-snapshot`, and supplies a separate durable HWM
work directory. Output paths containing parent traversal or symlinks are refused
before client or ledger construction; state and receipts must remain outside
input custody. It shares the existing guarded v2 byte budget, while its local
object ledger is separate from the active encoding publish ledger. This command
uses the established #224 credential path; it does not provision credentials or
change their permissions. ROOT's implementation qualification authorizes only
the dry-run. DLTrain owns the subsequent pre-approved real publication.

A successful object is acknowledged only after exact-VersionId GET verifies
its digest and size. The durable local ledger records intent before PUT and
verification afterward. Resume uses the same frozen plan and real work directory,
with a new receipt filename. Verified objects are read back without another PUT;
interrupted writes retry create-only and accept existing bytes only after pinned
verification. Conflicts, corruption or budget exhaustion stop publication and
retain a terminal failure receipt; no incomplete manifest is emitted. Preflight
refusals occur before client/ledger construction and are recorded by the invoking
command's exit status and stderr. Do not erase those receipts or restart with an
empty real budget ledger.

## Verification

```bash
python -m unittest discover -s deploy/dl-trainonly -p test_hwm_publish.py -v
python deploy/dl-trainonly/run_lane_offline.py --output "$EVIDENCE/offline"
just required
```

The new synthetic module is explicitly mapped to the guarded offline Python
lane. Existing host exclusions and lane limits remain unchanged. A local dry-run
and passing fixtures establish implementation behavior; real S3 persistence and
restoration require DLTrain's separate retained live publication receipts.
