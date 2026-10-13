# Native roster ingest (GB-01)

This module exports a local, manifest-last data candidate from pinned native
inventory, frozen input bundles and copied versioned S3 JSON receipts. It never
writes to S3, executes programs, computes embeddings, changes a live database,
or admits examples to a learner.

The plan schema is `jope.native-ingest-plan/v1`, with `non_qualifying: true` and
`sources` containing exact `{path, sha256, bytes}` references for:
`inventory`, `meta`, `features`, `gid`, `bundle_manifest`, `parents`, `winners`,
`target`, and `object_index`. The object index contains captured `program_result`,
`backlog_terminal`, and `backlog_current_hwm` entries. Each copied object carries
its immutable `{bucket, key, version_id, sha256, bytes}` reference and local `copy`.
Missing captures carry `state: unmeasured`; they never create zero rewards.

Only native identities with their recorded dataset version, cut and typed outer
fold join retained features. The frozen C15 parent registry supplies partition
and family membership; the inventory's current exclusions take precedence.
An S1 final receipt can identify a *candidate* for a formerly NULL context ID;
the original inventory ID remains NULL until its input/lineage binding exists.
A transformed dataset, an epoch-0 context, and a native context are separate
identities. P3 family IDs do not supply missing C15 lineage bindings.

Historical HWM DSL is exported as historical CE evidence. Additional native
program wrappers preserve all seven immutable artifact references, but their
artifact bodies and teaching eligibility remain unmeasured. `q`, `q_opt`,
`q_grid`, actor targets, and newly qualified counts remain NULL. All output is
`learner_admitted: false`; a current-selection receipt alone grants no admission.

## Commands

From the repository root, use the repository Python environment:

```sh
uv run --project python --no-sync python deploy/dl-flywheel/ingest_joined_roster/run_offline.py --receipt target/ingest-offline.json
```

To build from an already captured and pinned plan:

```sh
PYTHONPATH=deploy/dl-flywheel uv run --project python --no-sync python -m ingest_joined_roster --plan "$PLAN" --plan-sha256 "$PLAN_SHA256" --plan-bytes "$PLAN_BYTES" --output "$OUTPUT" --receipt "$RECEIPT"
```

Run host scratch commands in a `systemd-run --user --scope` with explicit CPU,
memory, swap and wall-time caps. Bulk outputs belong in lane custody outside
the Git working tree. S3 callers construct `ReadOnlyS3` with a bounded prefix,
deadline, request count and byte budget. The adapter only dispatches LIST,
HEAD and version-pinned GET, and returns a request/byte receipt. No credentials
are required by the offline fixtures or the build from copied inputs.

## Publication and consumption

`publish` checks every input's length and SHA before joining, rejects conflicting
duplicate evaluations, and preserves immutable source aliases. Local copy paths
and identical duplicate index entries do not change output identity; the CLI
receipt preserves the physical input pins separately.

The writer verifies and fsyncs members, then links the complete final
`manifest.json` last. Uncommitted output and prepared manifests are not usable.
An existing incompatible or corrupt output is refused. A final directory-fsync
failure removes only the newly created commit link and preserves prepared bytes.
Train and validation vectors occupy separate packs. Each row carries its own
SHA and exact vector offset, vector SHA and pack SHA.

Consumers call `read_committed(output, expected_manifest_sha256)`. It verifies
the pinned final manifest, all member and row hashes, vector bindings, counts
and train/validation parent disjointness before returning data candidates.
This validation supplies no optimizer, HWM, seal or promotion authority.

## Retained real-data proof and remaining data gap

The first successful packed proof on base
`c18582136d85feae07fceaa25a2eb608e9903e10` produced manifest
`919c4913d67707c9541ffdd2798f40149109026e37e068183e4c6767987cfff2`:

| Measurement | Result |
|---|---:|
| Retained train / validation contexts | 6,857 / 263 |
| Train / validation C15 parents | 1,014 / 54 |
| Parent overlap | 0 |
| Backlog runs retained explicitly | 5,461 |
| Backlog runs with current S3 selection metadata | 425 |
| Validation contexts with at least two distinct native wrapper candidates | 263 |
| Qualified new programs / actor-target contexts | NULL / NULL |

Source inputs are pinned in the run receipt, including native inventory
`2daff0b17e5bb0c330bddcedde5b9b29f7001811f04df5f3f3cc1f4cd9dd5597`,
bundle manifest `bb6b706cd7db29c9feb2e83afae9743655fb8a8a5ee39b00be45eb239075f2f4`,
and object index `54d19249c678c459512b3939b20746da65f8943f712574f8f2f76d0fce765f70`.
The capture copied 8,893 versioned JSON bodies in 13,239 read requests and
21,262,371 verified bytes, with zero remote writes and no binary GETs or
program execution. The roster rerun produced the identical manifest SHA.

Durable evidence is under `.agent/rlfw-coord/lanes/CX-MET/ingest/`, especially
`s3-capture/receipt.json`, `real-build-v3-receipt.json`, and `joined-roster-v2/`.
These are dated, NON-QUALIFYING data proofs. The 5,461-run gap remains open:
4,581 have historical G0-not-ready terminals and 880 lacked a published index
in the frozen inventory. No registered input/2051-vector/C15-lineage closure
for these runs has been established. The 425 newer selections do not close
that requirement. Likewise, 263 multi-program wrappers do not establish
263 contexts with qualified multi-program teaching measurements.
