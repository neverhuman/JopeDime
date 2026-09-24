# Dated data estate, recovery evidence and binary formats

This appendix draws mainly on four dated metadata captures made on 2026-09-23; [§](#recorded-copy-and-retention-gaps) also uses the DATA-01 author's copy-risk reconciliation, whose sizes partly come from a 2026-09-20 census. None of the four captures read a dataset payload, sealed label or model weight body. Compare figures within one capture; the captures differ in time and scope.

| Capture | Time (UTC) | Scope | Where |
|---|---|---|---|
| Product local roots | 16:29:27 | Six named product roots; `lstat` metadata plus two cached dataset manifests | [§](#product-local-roots-and-the-local-dataset-cache) |
| Product S3 buckets | 16:39:11 | Three product buckets; current objects only | [§](#product-s3-buckets) |
| DATA-01 bucket extension | configuration 17:30:13–17:30:19; learning-store prefix listings 17:30:32–17:33:50; bucket listings 17:34:15–17:37:10; assembled 17:37:54 | Configuration of nine buckets, targeted `veox-jopedime-learning-use2` prefixes, and current-object listings of seven additional or reconciled buckets | [§](#additional-archive-replica-and-custody-metadata) |
| DATA-01 host roots | scans started 17:30:58–17:33:14; report 18:54:01 | Twelve named host roots; `find -xdev` metadata | [§](#additional-host-data-roots) |

Both DATA-01 captures reach this manual only as an aggregate subset; [§](#additional-and-replica-buckets) records its release status.

## Configured product stores and local roots

### Access method, store defaults and path conventions

AWS metadata access used profile `veox-jepson-prod` and verified account `714044927654` with STS. No credentials are included in the receipts. Current-object listings exclude object versions and deleted objects. ETags are retained as concurrency guards for metadata downloads; they are not assumed to be SHA-256 content hashes. Catalog/collection/bundle manifest downloads use the exact listed ETag and verify size; their bytes receive local SHA-256 pins. No S3 dataset blobs, sealed tables, or model weight bodies were fetched.

`jope-datasets` defaults to `s3://veox-jopedime-datasets-use2`; `jope-models` defaults to `s3://veox-jopedime-models-use2`. The known learning store is `s3://veox-jopedime-learning-use2`; the current `jope-learning` mirror resolver uses `--store`, then `JOPE_LEARNING_URL`, and otherwise no remote mirror. It must not be described as an automatic remote default. No JOPE path/store overrides were set in this audit process.

There are two source-level path conventions to name explicitly: dataset cache resolution defaults to `${XDG_CACHE_HOME:-~/.cache}/jope`, while general `jope_paths::jope_home()` resolves a marked checkout. General mutable state defaults to `${XDG_STATE_HOME:-~/.local/state}/jope`. Use an explicit root or the owning command’s resolver; do not assume the shared variable name makes the defaults identical.

### Product S3 buckets

| S3 bucket / prefix | Current listed objects | Current listed logical bytes |
|---|---:|---:|
| `veox-jopedime-datasets-use2` | 15,442 | 3,234,922,515 |
| ↳ `blobs/` | 11,577 | 3,224,104,137 |
| ↳ `catalog/` | 2 | 5,363,942 |
| ↳ `collections/` | 4 | 94,356 |
| ↳ `datasets/` | 3,859 | 5,360,080 |
| `veox-jopedime-learning-use2` | 645,281 | 430,645,459,168 |
| ↳ `staging/` | 645,281 | 430,645,459,168 |
| `veox-jopedime-models-use2` | 967 | 3,929,105,273 |
| ↳ `blobs/` | 21 | 1,461,921,570 |
| ↳ `bundles/` | 3 | 32,045 |
| ↳ `staging/` | 943 | 2,467,151,658 |

The original three bucket listings completed with zero payload-body reads. The learning bucket contains current `staging/lineage-a/`, `staging/lineage-b/` and `staging/mirror-weights/` prefixes; the later DATA-01 extension below supplies measured and explicitly derived prefix counts. Current listed learning bytes are 430,645,459,168. A separate CloudWatch daily statistic at 2026-09-23T00:00:00Z reports 645,281 objects and 430,732,116,688 StandardStorage bytes. It is a dated metric with a different scope/capture time; no exact reconciliation with the current listing is claimed. Paginated S3 listing is an observation over an interval, not an atomic bucket snapshot.

### Product local roots and the local dataset cache

| Local root | Present | Regular files | Logical bytes | Unique-inode logical bytes |
|---|---|---:|---:|---:|
| `/home/ubuntu/.cache/jope` | True | 170 | 36,754,881 | 35,585,244 |
| `/home/ubuntu/.local/state/jope` | False | 0 | 0 | 0 |
| `/home/ubuntu/JopeDime/datasets/test` | True | 40 | 1,096,413 | 1,096,413 |
| `/home/ubuntu/JopeDime/datasets/by-id` | False | 0 | 0 | 0 |
| `/home/ubuntu/JopeDime/models` | True | 5 | 33,687 | 33,687 |
| `/home/ubuntu/JopeDime/state` | False | 0 | 0 | 0 |

The local inventory uses `lstat`, does not follow symlinks, and counts hardlinked payload bytes once in the unique-inode column. It covers the named product roots, not every archive, research checkout or build directory on the host. The standard local state roots were absent. That does not imply the host has no learning evidence: campaign and recovery receipts live in explicit custody paths outside those defaults.

The local dataset cache has two cataloged regression datasets, `0014b9dad5725a1e` (2,000 rows, 16 features) and `006bca140f93e411` (2,000 rows, 32 features), each with 1,000 declared training and 1,000 declared test rows. Their payloads were not reopened for this inventory. The committed historical `datasets/test/` collection is 10 datasets/40 files/1,096,413 B; it is distinct from the frozen thirty-dataset cache.

## Additional archive, replica and custody metadata

### Additional and replica buckets

A separate DATA-01 metadata capture at 2026-09-23T17:37:54Z extends the earlier three-store observation. It used the same verified AWS account and read current-object listings and bucket configuration only. The frozen panel and its row counts are unchanged. The author released the packet at 19:30:23 UTC (manifest `4bebee5a…`, 148 entries), but its handoff records a failed sealed-name gate and `cleared_for_handoff: false`. The only gate hit is in one raw file. A clarification recorded at 21:05:48 UTC (`RELEASE-CLARIFICATION.md`, `380f3402…`, covered by manifest `SHA256SUMS.release-2`, `a4b0c436…`) limits the release to eight pinned files: seven aggregate files and the DOCFIX-01 patch, which is reserved for a separate follow-up documentation PR. Codex applied that patch, with bounded corrections, in documentation-correction PR86 (head `0079ceddb`), which merged at 2026-09-24T00:04:12Z as main `2b375ef11`. Everything under `raw/` and `scratch/`, and the full handoff, stays quarantined in custody. This edition imports only Codex's two-file aggregate projection (stopped subset `558548be…`), which the clarification confirms lies inside the released scope: `data-estate-extension-summary.json` (`e4dbe075…`) and its source/scope pointer `data-estate-extension-release.json` (`366fa000…`). This is an aggregate subset, not a claim that the whole packet passed its release gates.

| Additional or reconciled S3 bucket | Current objects | Current logical bytes |
|---|---:|---:|
| `veox-jopedime-archive-use2` | 180,547 | 33,078,118,944 |
| `veox-jopedime-learning-usw2` | 0 | 0 |
| `veox-jopedime-models-usw2` | 967 | 3,929,105,273 |
| `veox-jopedime-models-use2` | 967 | 3,929,105,273 |
| `jopedime-714044927654-us-east-2-dev` | 186,305 | 118,116,270,003 |
| `jopedime-714044927654-us-east-2-prod` | 186,304 | 117,847,288,926 |
| `jopedime-714044927654-us-east-2-sealed` | 8,679 | 246,060,447,894 |

The models buckets agree in total and reported prefix counts and bytes; this does not prove key-by-key or payload identity. The learning replica is empty at this capture. Its source's four enabled replication filters are `hwm/`, `waves/`, `checkpoints/serving/` and the prefix `training-head`; none covers the current `staging/` objects. This is an observed configuration/coverage gap, not evidence that a matching replication rule failed. A `HeadBucket` probe for `veox-jopedime-datasets-usw2` returned HTTP 404 in this account/context; no dataset replica listing was obtained.

### Learning-store prefixes

| Learning-store prefix | Objects | Logical bytes | Basis |
|---|---:|---:|---|
| `staging/lineage-b/accepted60000-restored/` | 225,207 | 110,918,425,513 | measured |
| `staging/lineage-a/encodings/` | 419,644 | 149,840,032,105 | derived |
| `staging/lineage-a/preserved-archives-20260908-F-D-E/` | 34 | 141,775,927,725 | measured |
| `staging/lineage-a/genebank/` | 5 | 1,809,949,972 | measured |
| `staging/mirror-weights/` | 391 | 26,301,123,853 | measured |

The encodings row is a residual: the earlier full-bucket total minus four later measured sibling prefixes. It was not independently re-listed in full, and the sum closes by construction across capture times. Keep this distinction when comparing inventories. The accepted-60000 restored prefix's 225,207 objects and 110,918,425,513 B also match the retained local upload inventory by file count and size; that comparison is not a fresh content restore test.

### Versioning and Object Lock

All nine inspected buckets report versioning enabled. The six `veox-jopedime-*` buckets report Object Lock enabled with no default retention; the separate dev/prod/sealed family returned no Object Lock configuration. Neither result establishes retention for each object or preservation of every prior version. This listing excludes old versions and delete markers. Sealed storage contributes aggregate counts and bytes only; no sealed object body was fetched. Object/file counts are not counts of unique datasets, admitted teachers or recoverable checkpoints.

## Additional host data roots

### Root inventory and totals

The DATA-01 report generated at 2026-09-23T18:54:01Z summarizes metadata scans starting at 17:30–17:33 UTC across twelve named roots with `find -xdev` metadata only. Its following apparent-byte totals count each regular-file path, including hardlinks. They are not additive with the earlier product-root table or the S3 copies, and do not measure memory needed to train.

| Local root | Regular-file paths | Apparent bytes | Allocated bytes, including directories | Scan |
|---|---:|---:|---:|---|
| `/home/ubuntu/remote_super/quant` | 6,127,909 | 740,214,173,579 | 755,552,710,656 | completed |
| `/home/ubuntu/jope-flywheel` | 465,739 | 161,547,128,345 | 162,954,956,800 | completed |
| `/home/ubuntu/preserved-archives` | 23 | 141,775,926,746 | 141,776,224,256 | completed |
| `/home/ubuntu/jope-preservation` | 390,889 | 250,487,278,093 | 250,592,124,928 | completed |
| `/home/ubuntu/jope-evidence` | 236,213 | 123,899,684,744 | 125,110,591,488 | completed |
| `/home/ubuntu/jope-migration-evidence` | 234,158 | 33,597,305,065 | 35,018,276,864 | partial: 15 unreadable directories |
| `/home/ubuntu/jope-dime-mirror` | 201,251 | 23,067,601,488 | 23,633,518,592 | completed |
| `/home/ubuntu/jope-rescue-20260921` | 70,442 | 12,508,351,289 | 12,767,522,816 | completed |
| `/home/ubuntu/golden42` | 37,612 | 1,749,309,751 | 1,849,475,072 | completed |
| `/home/ubuntu/sotarena` | 22,602 | 705,696,476 | 776,736,768 | completed |
| `/home/ubuntu/hyperion/weights` | 4 | 253,108,234 | 253,132,800 | completed |
| `/home/ubuntu/.cache/jope` | 170 | 36,754,881 | 37,158,912 | completed |

Across these roots the scan observed 7,787,012 regular-file paths, 1,489,842,318,691 apparent bytes and 1,510,322,429,952 allocated bytes. Removing the recorded within-root and cross-root hardlink overcount gives 1,472,779,971,117 apparent bytes; content-identical copies on different inodes remain counted. Fifteen permission-denied directories under `jope-migration-evidence` are missing from these totals. Several roots were changing during the scan, including the custody directory holding the inventory itself.

### Filesystem capacity and dataset-shaped directories

All twelve roots resolve to the same filesystem. Its contemporaneous readback was 7,936,183,619,584 B total, 6,815,282,892,800 B used and 1,040,879,616,000 B available (969.395 GiB); these are dated observations, not current capacity guarantees. Multiple local paths on that filesystem do not provide independent failure-domain copies. Count/size matches with upload receipts identify recovery candidates; they do not authorize deletion or replace exact-content restore verification.

The `remote_super/quant` scan found 2,322 top-level 16-hex-ID directories with all three `train.csv`, `test.csv` and `meta.json` markers. The corrected `synth*` classification contains 25,197 `synth_<16hex>` dataset directories with those direct markers, plus two containers named `synthetic` and `synthetic_packed`. The dataset-directory class contains 86,923 files and 21,494,046,579 B; it includes more than the three payload markers per dataset. These are name/shape observations, not proof of unique, valid or train-admitted datasets. No rows, feature distributions or label bodies were opened. A pathname containing `sealed` is only a naming statistic and cannot identify every sealed/test-label payload.

## Recorded copy and retention gaps

### Material without an established remote copy

The author reconciled the surveyed product/custody listings, upload receipts and local indexes. The following sizes describe material for which that reconciliation did not establish a remote copy. They do not prove absence from every bucket, machine or archive. Counts overlap other tables and risk classes; never sum them as unique at-risk bytes.

| Material | Recorded files | Logical bytes | Evidence basis |
|---|---:|---:|---|
| T3 synthetic-regression train/meta/test payload | 75,591 | 21,471,473,310 | September 20 census; 25,197 dataset groups |
| T3 synthetic-binary payload | 305,284 | 135,798,140,876 | September 20 census |
| Synthetic source data | 101,687 | 126,856,964,874 | September 20 census |
| Binary generator script | 1 | 102,068 | September 23 file metadata |
| Node-archive mirror classes without a recorded S3 match | 394,460 | 75,989,854,135 | September 23 sizes joined to retained SHA-256 indexes |
| Quant model tree | 650,015 | 78,566,865,846 | September 23 file metadata; includes potentially replaceable cache |
| Migration evidence outside the uploaded node-high-value subtree | 53,638 | 22,715,493,488 | September 23 metadata; 15 unreadable directories excluded |

Payload sizes above include aggregate withheld/test-file sizes from historical metadata; this audit did not open those bodies. Documentation records the T3 regression generator as lost; this is a retained provenance claim, not a fresh exhaustive search. The binary generator has no second copy established in the surveyed records. The 21,471,473,310 B payload figure is the older three-file census, whereas the 21,494,046,579 B current directory class also includes other files: these are different scopes. Model, research, journal, evidence and other retained-root risks are summarized by all 19 source risk IDs in the companion JSON; missing counts remain null. Seven retained roots were not rescanned.

### Recorded custody risks and disk capacity

The empty surveyed learning replica and uncovered `staging/` prefix leave cross-region protection unestablished for the current learning objects. Counts matching between model buckets do not establish a successful content restore. Bucket Object Lock without default retention does not prove any object's retention or legal hold; custody buckets have no Object Lock configuration. The retained custody status also records a packer stop for exhausted disk reserve. At the dated local readback, `df` reported 87% used; capacity must be rechecked before admitting later workloads. These findings call for verified preservation and owner decisions about replication/retention. They authorize neither deletion nor new cloud writes, and they do not change the original training or panel budgets. Proposed rules: keep a single learning-store writer until an atomic server-side pointer update is qualified; backup, replication and retention operations remain separately authorized.

## Recovery evidence: what is preserved, and what was lost

### Lineages A and B and the staging store

Two historical lineages must remain separate. Lineage A contains July–September flywheel material, old winner/genebank state and preserved archives. Lineage B contains the native corpus/training chain used by the retained step-60000 checkpoint. `ADR-0005` and `docs/learning.md` describe their intended migration and their observed losses; those dated records do not authorize silently promoting old labels under the current ASM rules.

The learning S3 store is a staging estate: at the 2026-09-23 listings all 645,281 objects in `veox-jopedime-learning-use2` were under `staging/`. Targeted metadata listings found `staging/lineage-a/`, `staging/lineage-b/accepted60000-restored/` and `staging/mirror-weights/`. [§](#product-s3-buckets) and [§](#learning-store-prefixes) record the bucket census and the per-prefix counts. Object upload time is not the time a target was discovered or consumed. Five small metadata objects were fetched with an ETag precondition and hashed: the encoding caveats, frozen slice listing, archive catalog, genebank deprecation note and accepted60000 materialization intent. Their exact keys, byte lengths, ETags, upload times and fetch hashes are in `metadata/fetched-metadata.json` in custody.

### Lineage-A selections, archives, lost records and exposed score tiers

The frozen July 25 `slice500` listing chose the first 500 bytewise-sorted names from 25,031 directories, excluding 186 incoming entries and hidden packed storage. That is a reproducible selection of candidate datasets, not proof of 500 admitted teachers. The retained encoding caveat says the historical binary target encoder used a regression-flavored checkpoint: binary values were in range, but that did not establish classification training qualification. The same note describes historical target-vector fusion and encoder pins. It must not be recast as the current verified classification-context contract.

Lineage A preserves archives for snapshots `004754860`, `008903292`, `009658180` and `009915070`. The archives' compressed byte sizes and checksums identify preservation containers, not in-memory training size. This audit did not extract them. The historical records declare the strict-test-exact database, pair registry, later immutable checkpoint descendants, acceptance pointers, sealed-consumption receipts and task-Elo registry lost. A directory number or an old selected pointer cannot recreate that missing evidence. The old genebank has an improvement-event schema with task, dataset, score tier, metric, evaluator, source and ingestion time, but lacks the complete cut/vocabulary/protocol identity required by the current HWM. Its 22,798 improvement-event rows are not 22,798 currently admissible teachers; its timestamps include backfill ingestion.

The exact-current HWM identity and current admission requirements are explained in [§](teachers.md#the-admission-guard-label-exposure-and-hwm-identity) and [§](learning.md#what-better-and-better-must-mean). The historical `test_exact` score tier is optimization-exposed development evidence when used to choose programs. Renaming the tier does not restore independence.

## Retained corpus sizes

### Six retained corpora

The following table comes from six directly read local index/provenance files. Snapshot names are identifiers, not a reconstructed wall-clock axis. Several publications have no independently recorded timestamp in their index. These are retained historical corpora, not six steps in one proved monotonic update sequence. Counts must not be summed: corpora overlap.

| Corpus | Records | Contexts | Dataset/family counts | Groups | Record bytes | Audit-label bytes |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `qualification-v1` | 9,955 | 3,500 | 35 / 29 | 530 | 5,296,060 | 2,466,332 |
| `qualification-v2` | 85,498 | 3,500 | 35 / 29 | 834 | 45,484,936 | 25,540,156 |
| `qualification-v3` | 422,840 | 7,000 | 35 / 29 | 1,320 | 224,950,880 | 131,176,452 |
| `cohort64-qualification-v1` | 779,158 | 8,799 | 64 / 64 | 1,620 | 414,512,056 | 243,431,656 |
| `dataset-inclusion-20260909` | 5,538,508 | 8,904 | 82 / 82 | 6,815 | 2,946,486,256 | 1,753,858,051 |
| Accepted60000 closure, `snapshot-000097` | 10,214,332 | 8,894 | 80 / 80 | 7,877 | 5,434,024,624 | 3,227,289,461 |

All six use 532-byte packed records. The first five indexes use `parallel.teaching_corpus.v1`; the accepted closure uses `parallel.teaching_shards.v1`. Their indexes alone range from 1,807,418 to 98,330,379 bytes. The accepted sharded index is 60,589,288 bytes. Large JSON metadata remains a real part of startup work even when target records are binary. Full paths, byte counts, schemas and SHA-256 values are in `corpus-snapshots.csv` and `local-corpus-metadata.json`.

### Snapshot 421 and snapshot 97 accounting

A separate September 11 audit captured snapshot `000421` with 8,899,150 records and 24,313,972 observations seen; its original index hash is `74d643be…`. That is a retained audit observation, not a seventh newly payload-verified corpus. The later accepted `000097` belongs to a different recovery run: 97 is not chronologically earlier than 421. The apparent count difference cannot be called a count of newly discovered unique teachers without joining actual program/context identities and admission events.

For accepted snapshot 97, metadata records 34,321,997 observations seen, 109,797 failed programs, 488,028 withheld meta-test observations, 1,338,630 observations without a matching context, 405,322 exact-parameter tokenization rejections and 112,903 out-of-vocabulary parameter rejections. These diagnostics describe the packer's accounting categories; do not assume they form a disjoint partition that can simply be subtracted from the observation total.

## Binary format: the exact packed teaching record

### Record layout

The Rust writer and reader agree on the following fixed-width layout. Offsets are byte offsets; all multibyte numbers are little-endian.

| Offset | Bytes | Stored field | Meaning |
| ---: | ---: | --- | --- |
| 0 | 256 | `128 × u16` token IDs | Exact representable grammar sequence, with padded slots |
| 256 | 256 | `128 × u16` legal-set IDs | In `teaching_shards.v1`, group-local index into that shard’s own `legal_sets` for each sequence position |
| 512 | 2 | `u16` length | Actual token sequence length |
| 514 | 2 | `u16` zero | Reserved space |
| 516 | 4 | `u32` context ID | Index into the shared context metadata/vector table |
| 520 | 4 | `f32` value | Score-derived training value |
| 524 | 4 | `f32` percentile | Ranking supervision under the declared label policy |
| 528 | 4 | `f32` log-runtime | Natural logarithm of one plus runtime |
| Total | **532** | | |

### Packing semantics, legal sets and label policies

This is binary packing, not general-purpose compression. Every record reserves 128 token slots and 128 legal-state slots. A short program still occupies 532 bytes on disk. Context vectors are shared by context identity. In the accepted sharded format, each group stores its own legal-set table in `shard.json` and its index entry; legal IDs are local to that table. Loading remaps them into one global table (63 distinct sets in the external accepted-corpus audit), shared in host memory and on the GPU. The older monolithic `teaching_corpus.v1` format instead has a top-level legal table. Do not interpret a raw shard ID as a global ID. `groups.bin` stores two little-endian u64 values per group—start record and count—so a group entry is 16 bytes. The JSON index binds context, vocabulary, record hashes and offsets; each immutable shard also has a manifest and an audit-label sidecar.

Original floating-point measurements and exact program provenance are retained separately from the float32 training targets. Historical `reference_v1` orders original f64 RMSEs, uses a median-scaled score value, and assigns ordinal percentiles. The stored value is `1 / (1 + RMSE / max(upper_middle_RMSE, 1e-8))`; the percentile is `1 - i/(n-1)` in the ordered group. Here `upper_middle_RMSE` is the sorted original RMSE at zero-based index `floor(n/2)`, including for an even-sized group; it is not an average of the two middle scores. Its tie behavior is not the later midrank policy. `frozen_z_v1` preserves the first admitted value/percentile for an existing program while new programs receive their initial labels. `strict_midrank_v4` introduces exact-original-precision tie handling and the strict admission contract. These names are different target semantics, not interchangeable storage optimizations. Repacking cannot silently relabel historical targets or snap unsupported program parameters to convenient vocabulary tokens.

### Verified loading into seal-protected shared memory

The training reader's verified path copies and hashes the actual record bytes into seal-protected shared memory (a `memfd` carrying write, grow, shrink and seal-lock seals), or obtains a verified shared file carrying those seals, then uses a read-only memory mapping. These Linux file seals, unrelated to sealed evaluation data, prevent later writes, growth and truncation. This protects a running trainer from source-file mutation and mmap truncation faults. It is **not** a promise of zero-copy disk startup: verification and the initial sealed-memory copy have costs. Index, record and context identities remain content-based; a filesystem timestamp is not a substitute for them.

### What the accepted-corpus packing audit measured

#### Occupancy, padding and index size

The external verifier reports scanning all 10,214,332 records in 7,877 accepted-corpus shards, checking recorded payload/sidecar hashes and layout invariants. Its corpus identity is the same `1c840597…` bound in [§](data.md#record-split-and-the-meta-test-closure-question). This manual independently verifies its selected metadata-file hashes and recomputes the following arithmetic; it does not claim a second scan of the original record or label bodies.

| Quantity | Result | Evidence boundary |
| --- | ---: | --- |
| Occupied token slots | 205,657,518 of 1,307,434,496 | Sum of the external length histogram |
| Mean / maximum sequence length | 20.1342 / 54 tokens | Accepted snapshot only; not a future grammar bound |
| Padding fraction | 84.2701% | `1 - occupied_slots / (records × 128)` |
| Pretty-printed `legal_sets` values | 37,691,003 bytes, 62.2074% of the 60,589,288-byte index | Raw stored JSON spans; compact serialization is a different 8,335,358 bytes |
| Original record arrays | 5,434,024,624 bytes | 10,214,332 × 532 |
| Proposed 180-byte record arrays | 1,838,579,760 bytes | Layout calculation; excludes indexes, contexts and runtime overhead |
| Proposed binary index | 1,906,735 bytes | Estimated table components; no qualified reader or writer |

#### Compression, the proposed 180-byte layout and label precision

The verifier also reports an actual external `zstd -3 -T1` size experiment: original records compressed separately by group occupy **292,918,135 bytes**, audit-label group frames **331,375,483 bytes**, and one compressed index **5,371,779 bytes**. These named components total **629,665,397 bytes**, versus **8,721,903,373 bytes** uncompressed. This subtotal excludes shard manifests, `groups.bin`, contexts, vocabulary, checkpoint and other provenance; it is not a complete deployable-store size or a host/GPU memory measurement. Experimental 180-byte group records compressed to 259,362,502 bytes in that external experiment. No loader-throughput, optimizer or fitted-search benefit was measured by this metadata audit.

Status: proposed. The owner's packing order follows phase profiling: first a verified Python packed-reader bridge, then immutable shard reuse with bounded staging, then compression, and a narrower layout only if measurements justify it. The 180-byte layout is such a narrower layout: `56 × u16` tokens + `56 × u8` local legal-set IDs + three float32 labels: 112 + 56 + 12 bytes. It would reconstruct length and context from the token prefix and authenticated group metadata. The observed maximum length rounds to 56, and the current loss path already trims each batch to a multiple of eight. Those facts motivate the proposal; they **do not prove bit-identical training**. A production codec must preserve the existing 128-token grammar limit through a versioned extension or verified fallback layout; the observed 54-token maximum does not authorize a smaller grammar cap or truncation. It must reject inconsistent widths and legal-ID overflow, preserve group/sampler order, RNG sequence and all identity/provenance, reconstruct identical legal masks and every optimizer input, and pass record round trips plus fixed-batch token/mask/context/label, loss, gradient, update and reload/continuation parity. Changing physical bytes also changes the current raw-index corpus digest; compatibility needs an explicit verified migration, not an inherited old hash.

Keep the numerical policy separate from byte layout. The external all-pairs calculation reports that float16 creates new value ties in **0.377392%** of all within-group record pairs and **0.289905%** of teacher-versus-other pairs. These have different denominators from adjacent-score pairs. Quantizing labels therefore changes ranking supervision. Preserve float32 training labels and contexts, and original-precision audit observations; compression of their bytes does not require reducing their precision.
