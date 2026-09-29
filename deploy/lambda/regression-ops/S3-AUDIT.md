# S3 audit and evolution storage decision

Audit date: 2026-09-28. Current-key census ran 21:10:51–21:14:45 UTC. It is a scan interval, not a transactional snapshot. Historical object versions and individual retention/legal holds were not exhaustively inventoried.

**Keep the useful existing design. Add a versioned experimental layer; do not reorganize old data in place.** Content-addressed objects, version-pinned references, immutable manifests, lineage and explicit scoring contexts are already the right foundations. The largest gains come from better indexes, publication boundaries and compact query tables—not renaming every S3 folder.

## What is actually present

| Primary bucket | Current objects | Bytes | Main purpose observed |
|---|---:|---:|---|
| veox-jopedime-datasets-use2 | 147,170 | 114,406,782,723 | Source corpus plus substantial historical research output |
| veox-jopedime-learning-use2 | 647,908 | 431,993,485,251 | Mostly staging lineage, plus recent experiments |
| veox-jopedime-models-use2 | 967 | 3,929,105,273 | Model bundles, blobs and staging |
| veox-jopedime-archive-use2 | 180,547 | 33,078,118,944 | Source archives and historical node evidence |
| **Total** | **976,592** | **583,407,492,191** | **583.41 GB decimal** |

Learning staging contains 645,281 objects / 430.65 GB. Its lineage-A, lineage-B and mirror-weight prefixes remain historical custody; their presence does not mean canonical learning admission. The current census found no populated canonical score/HWM roots. Existing experiment prefixes include the completed Lambda pilot and GP discovery artifacts.

The dataset bucket now contains 130,687 research objects / 100.05 GB, including BeyondArena and corpus-benchmark research. This differs materially from the older repository inventory that described a mostly 3.23 GB source corpus. Keep the old paths readable and document their status; moving them would break existing receipts without improving scientific validity.

The audit also inspected root prefixes of the west-region replicas and separate dev/prod/sealed custody buckets. Those descendants were not fully censused. The learning west-region replica was empty at the root listing. Do not substitute a custody bucket for a product source bucket.

## Compatibility findings

The audit retrieved 3,913 small JSON/JSONL source objects with no fetch errors, retaining original bytes, source version IDs, byte lengths and SHA-256 receipts. Most were dataset manifests; the remaining samples covered approximately 30 schema identities across legacy learning, experiment commits, model bundles and research protocols. This is representative content inspection, not a claim that all 976,592 payloads were decoded.

The important contracts remain intact:

- `jope.score/v1` has a closed metric enum. MAE, MAPE, RMSLE, fold vectors and importance-stability measurements cannot be smuggled into that schema. They belong in a versioned companion envelope.
- Legacy score hashes use native Rust serialization with the hash field blanked. One field description says “without” the field, which conflicts with the implementation. The implementation governs; generic Python JSON is not a substitute.
- `jope.hwm/v1` comparison identity, sealed-data rules and strict-improvement semantics remain unchanged. New experimental elites are not canonical HWM promotions.
- Legacy observations explicitly distinguish adaptive held-out-cut teachers, normalized versus original target units, program hashes and semantic hashes. Preserve those distinctions when importing.
- The existing TUI snapshot schema remains unchanged. A new producer supplies the existing viewer; snapshots are disposable views, never the scientific ledger.

Five core new JSON schemas validate GP evaluation envelopes, fitted folds, invocation packs, context HEADs and context events. The unchanged legacy score/HWM schemas are copied with hashes into the registry. A synthetic compatibility fixture passed schema and native Rust Value serialization checks. Native ScoreRecord deserialization and canonical score export are **not** enabled or claimed.

## Better scaling without a disruptive migration

New data uses:

```text
s3://veox-jopedime-learning-use2/experiments/regression-evolution/v1/
  objects/<kind>/sha256/<content-hash>.<format>
  attempts/<attempt-id>.json.gz
  contexts/<context-hash>/HEAD.json
  campaigns/<campaign-id>/protocol.json
  campaigns/<campaign-id>/manifests/<manifest-hash>.json
  objects/query-<table>/sha256/<content-hash>.parquet
  campaigns/<campaign-id>/COMMIT.json
```

The logical dataset/fold/generation organization belongs in manifests and tables. Physical payloads are shared by content hash, so a model does not get copied into five different hierarchy branches. A query should filter a compact table and follow exact references, rather than list S3 once per tree.

Each context HEAD is a small conditional-write pointer to immutable admission/completion events. Every 64 revisions produces an index checkpoint. This avoids repeatedly versioning the entire growing seen-program map. Missing chain objects fail closed. A live S3 qualification with 16 concurrent claimants admitted exactly one claim; restart and completed-claim checks admitted none.

Invocation packs combine models, predictions, fold memberships and outcomes into bounded compressed objects. Parquet tables provide the narrow columns needed for dashboards and neural training. Preserve all successful candidates, including losers and successful partial folds; an elite-only archive cannot support unbiased retrospective analysis or learning which proposals fail.

Publish manifests only after payload verification. Final COMMIT is create-only. Mutable convenience pointers are explicitly named and updated conditionally. Versioning alone is neither a concurrency lock nor a retention guarantee.

## Historical reuse and duplicate prevention

The historical index contains **8,345,677 program records** and **10,544,544 dataset/program pairs**. It includes the genebank, every one of the 60,276 listed lineage-B observation files, frozen discovery programs and completed Lambda candidate records. The genebank's complete local SHA matches its S3 source metadata; lineage-B files matched authenticated listing sizes and ETags, with local SHA receipts retained.

Current search excludes historical raw and canonical program fingerprints, then atomically claims each new `(evaluation-context, program)` identity before dispatch. Failed and unknown claims remain seen; losing an HTTP response does not authorize duplicate training. Completed results are reusable only after archive identity and closure checks.

**Remaining limitation:** opaque archives and every separate custody bucket have not been exhaustively decoded. This is strong duplicate prevention over the indexed history, not a proven account-wide absence of every old tree. No legacy fitness was admitted as an exact cache hit: its original full evaluation context was incomplete. Historical CV champions can seed mutations without pretending their old scores are measurements under the new protocol.

## Scientific data contract

Store distinct labels for inner CV, training-only confirmation and outer development results. Keep each metric's units, direction, fold memberships, scorer identity and exposure history. The optimizer sees CV only. Outer development outcomes can later train a separate neural prediction head; once used that way, they are no longer an unbiased test of that learned policy.

The historical genebank supports this separation. Of 54,825 dataset/program pairs with one unique value per CV/test tier, 2,286 had positive CV R² and nonpositive test R²; 811 had the reverse. Another 1,296 pairs had multiple values and remain explicitly ambiguous. These are adaptively selected retrospective records with incomplete fold/evaluator equivalence—not a clean generalization study.

The preferred neural target is a vector: CV skill, development skill, generalization gap, uncertainty/reliability and compute cost. Preserve raw vectors so later training can choose a scalarization without rewriting history. Keep negative examples and failed programs. Family/alias separation is more valuable than randomly dividing rows from the same source dataset among meta-training and meta-test.

Current fitness uses the mean log CV RMSE-to-mean-baseline ratio plus 0.2 times its fold SD. That SD is descriptive; correlated folds do not justify treating it as an independent-sample confidence interval. A separate Rust assessment tool now restores archived fitted fold models and computes permutation importance without fitting again. Five representative generation-2 leaders passed prediction replay and importance assessment, with all their features covered. This is qualification, not campaign-wide VIP coverage. Outer-development scoring is not implemented in this campaign.

## Durability and cost controls

The primary buckets have versioning and Object Lock enabled, but no default retention was shown. Individual holds/retention were not exhaustively read. The dataset bucket has no configured replication. Learning replication covers `hwm/`, `waves/`, `checkpoints/serving/` and `training-head`, **not** experiments, scores or staging. New experiment data therefore does not automatically gain a west-region replica.

Recommendation: define a recovery objective and restore drill for committed experiment manifests and their reachable payloads, then add targeted replication/retention through a separately documented infrastructure change. Preserve staging as historical custody. Do not apply a broad lifecycle rule that moves active query tables or unresolved experiment payloads into an inaccessible tier.

The runtime is an ARM ZIP deployment: 3,071,568-byte executable, 1,645,898-byte ZIP. No ECR is required. The Lambda role has logs-only permissions; signed input/output capabilities bind one exact experiment request. The private env file contains profile and paths, not keys.

The approved Lambda ceiling is $1,000 total, including $13.6535432 prior reservations. Local admission reserves the full invocation timeout using a conservative rate and keeps uncertain liabilities. The existing monthly AWS budget is amended to $1,000 with its dispatcher stop action at $800; AWS budget reporting is delayed and is not an instantaneous account-wide hard cap. Qualification used a separate 25 GiB storage envelope. The recorded production amendments increase that to 128 GiB and require at least 200 GiB free host space before new fits. S3 and logging costs remain outside the Lambda ceiling. Production uses 512 controller workers and up to 896 reserved Lambda executions; reservation is capacity, not measured simultaneous activity.

Each completed child fold is fsynced locally; successful invocation acknowledgment follows S3 archive upload. A catastrophic runtime or upload failure can still leave an unresolved attempt without a recoverable model. Such gaps are reported and blocked from automatic retraining, rather than declared successful custody.

## Evidence and follow-through

Source lives under `~/JopeDime/.agent/experiments/regression-lambda-v3/`; post-freeze reporting tools use the adjacent `regression-lambda-v3-ops` directory. Raw evidence is under `~/scratch/jopedime-regression-lambda-v3-20260928/`. No old research prefix, canonical score ledger, HWM, accepted policy, Git index or product branch was modified.

The accompanying frozen schema-evolution document records the original contract. SCHEMA-ADDENDUM.md corrects the proposed versus actual physical table paths, documents operational additions and identifies which contracts have machine validation. The frozen worker plan describes qualification limits; resource amendments govern production. Campaign execution and qualification results are separate dated receipts; deployment alone is not proof of completed evolution.

## Next improvements worth making

Keep the immutable payload/manifests architecture. For repeated large campaigns, compact the small live Parquet shards into larger immutable partitions, with a new catalog pointing to the compacted view and old segment lineage retained. Partition by campaign and a bounded dataset-hash bucket; avoid one tiny partition per tree or fold. Benchmark scan size and latency before choosing file sizes or adding a table format. Existing packs stay authoritative.

The current bounded campaign manifest enumerates its object references directly. At materially larger scale, replace that with a versioned root referencing bounded manifest pages. Do not grow a single mutable master JSON or rename source objects to achieve organization. Introduce this as a new manifest schema with a reader test; the current v1 reader and earlier snapshots remain valid.

Add a dataset-family/alias registry and a teacher-admission table before training a meta-model. Neither an attractive CV score nor a complete S3 archive demonstrates independent generalization. Add algorithm-aware memory routing after enough measured duration/peak-memory outcomes accumulate; the current tier selector uses input size only. Finally, consider streaming each completed fold to durable storage, so an invocation-level upload failure cannot strand all its already-completed models.
