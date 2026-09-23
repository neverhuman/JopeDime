# Data estate, statistics and measured evidence

Observation date: 2026-09-23T16:41:48.371336+00:00. Frozen source: `68f56404c3e199e06413bfd645d3fe0803d7b96f`. These are observations at that source and time. They do not turn stored metadata, a source packet, or historical results into a current operational qualification.

## What is measured and what remains pending

The current source has a successful operational proof on `230_machine_cpu`: the verified encoder produced context; the neural producer supplied actual generation-zero fits; a fitted descendant followed; two fitted refits each returned 34 development predictions. The neural refit did not beat the matched generic arm. This proof uses the accepted checkpoint, not a newly trained experimental export. Training, final installed warm demonstrations, and the unchanged thirty-dataset panel remain unstarted at this observation. No current thirty-dataset ranking or Elo exists.

Every number in this chapter comes from a pinned manifest, prepared development table, immutable report, committed historical snapshot, filesystem metadata, or authenticated AWS metadata request. The audit did not run a fit, load a model, open sealed target bodies, or execute a report command.

The machine-readable companions are `dataset-inventory.json` (all thirty datasets, 522 feature records, complete available upstream metadata and detailed development statistics), `performance-evidence.json.gz` (measured current origin plus historical score cells, timing and ladders), `catalog-summary.json`, `local-estate.json`, `s3-estate.json`, and `metadata-receipts/`. The catalog JSONL retains every available per-dataset catalog field for all 3,859 corpus datasets. No claim is made to enumerate every mathematically possible statistic.

## Frozen thirty-dataset panel

Panel `asm-private-30-v1` uses ten regression, ten binary, and ten multiclass datasets from PMLB revision `7c1f4bdc00136dc2e55c87fa6b8ba6e8af6d1a68`. Manifest SHA-256: `ba7dace4cd535687002cc95538dbc7f7aaf77ec973177c7159b3c6d22471eee0`. It was selected before measuring ASM outcomes, excludes the historical favorable test-10 collection, and cannot replace a dataset because a method failed or lost.

Local preparation: `/home/ubuntu/.cache/jope/asm-private-30-v1/`. `development/` contains the permitted development tables; `splits/` contains original row/group/fold assignments; `metadata/` contains pinned upstream YAML; `downloads/` contains the original compressed archives; `sealed/` contains withheld partitions. The audit opened development, assignment, and metadata files only. It inspected sizes and expected hashes of raw and sealed files without opening their bodies.

There are **55,029 total rows**, **44,048 development rows**, and **10,981 sealed rows** according to the frozen assignments. Feature counts sum to 522 across datasets; this is not a union of unique semantic variables. Development data contains 1,612 missing feature cells in 302 rows. All thirty development, split-assignment and metadata file SHA-256 values match the frozen manifest. Every row ID is unique within its dataset, and every group stays wholly within one original development fold or sealed partition.

| Dataset | Task | Total rows | Development | Sealed metadata | Numeric features | Categorical features | Missing development cells | Fold sizes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 197_cpu_act | regression | 8192 | 6553 | 1639 | 21 | 0 | 0 | 1311,1311,1311,1310,1310 |
| 225_puma8NH | regression | 8192 | 6553 | 1639 | 8 | 0 | 0 | 1311,1311,1311,1310,1310 |
| 230_machine_cpu | regression | 209 | 169 | 40 | 6 | 0 | 0 | 34,32,37,33,33 |
| 522_pm10 | regression | 500 | 402 | 98 | 7 | 0 | 0 | 81,81,91,67,82 |
| 529_pollen | regression | 3848 | 3078 | 770 | 4 | 0 | 0 | 616,616,616,615,615 |
| 579_fri_c0_250_5 | regression | 250 | 200 | 50 | 5 | 0 | 0 | 40,40,40,40,40 |
| auto_insurance_price | regression | 201 | 162 | 39 | 13 | 10 | 10 | 35,32,31,31,33 |
| 690_visualizing_galaxy | regression | 323 | 258 | 65 | 4 | 0 | 0 | 52,52,52,51,51 |
| 712_chscase_geyser1 | regression | 222 | 172 | 50 | 2 | 0 | 0 | 32,34,38,35,33 |
| 503_wind | regression | 6574 | 5238 | 1336 | 14 | 0 | 0 | 1066,1074,1030,1032,1036 |
| breast_cancer_wisconsin_diagnostic | binary | 569 | 455 | 114 | 30 | 0 | 0 | 91,91,91,91,91 |
| credit_approval_germany | binary | 1000 | 800 | 200 | 3 | 17 | 0 | 160,160,160,160,160 |
| congressional_voting_records | binary | 435 | 347 | 88 | 0 | 16 | 0 | 77,57,64,84,65 |
| haberman | binary | 306 | 245 | 61 | 3 | 0 | 0 | 49,49,46,50,51 |
| ionosphere | binary | 351 | 281 | 70 | 32 | 2 | 0 | 55,57,56,57,56 |
| sonar | binary | 208 | 166 | 42 | 60 | 0 | 0 | 34,33,33,33,33 |
| spambase | binary | 4601 | 3723 | 878 | 57 | 0 | 0 | 696,781,710,789,747 |
| phoneme | binary | 5404 | 4330 | 1074 | 5 | 0 | 0 | 866,867,866,870,861 |
| tic_tac_toe | binary | 958 | 766 | 192 | 0 | 9 | 0 | 154,153,153,153,153 |
| horse_colic_surgery | binary | 368 | 292 | 76 | 7 | 13 | 1583 | 58,61,58,58,57 |
| iris | multiclass | 150 | 122 | 28 | 4 | 0 | 0 | 25,23,25,25,24 |
| wine_recognition | multiclass | 178 | 142 | 36 | 13 | 0 | 0 | 29,29,28,28,28 |
| car_evaluation | multiclass | 1728 | 1382 | 346 | 0 | 6 | 0 | 277,277,276,276,276 |
| balance_scale | multiclass | 625 | 500 | 125 | 0 | 4 | 0 | 100,100,100,100,100 |
| dermatology | multiclass | 366 | 292 | 74 | 1 | 33 | 0 | 59,59,58,58,58 |
| segmentation | multiclass | 2310 | 1848 | 462 | 18 | 1 | 0 | 364,372,369,373,370 |
| vehicle | multiclass | 846 | 677 | 169 | 18 | 0 | 0 | 135,136,135,136,135 |
| optdigits | multiclass | 5620 | 4496 | 1124 | 51 | 13 | 0 | 900,899,899,899,899 |
| penguins | multiclass | 344 | 275 | 69 | 5 | 2 | 19 | 55,55,55,55,55 |
| tae | multiclass | 151 | 124 | 27 | 3 | 2 | 0 | 20,25,27,30,22 |

Regression uses RMSE; binary uses AUROC with minimization loss `1−AUROC`; multiclass uses probability-matrix log loss with the frozen class ordering. Features retain numeric/categorical type and missing cells; original labels are not converted to manufacture a task. Screen all applicable methods on original fold zero, freeze five recipes, confirm across all five development folds, and confirm ASM additionally if it missed the five. These optimization-visible folds remain development evidence. Sealed evaluation is separate.

Known date columns and identical-feature groups are retained by the prepared split. Missing original subject/time identifiers remain a limitation. A grouped split cannot establish independence when the source never supplied those identifiers. Checkpoint exposure is unknown for these datasets; an unseen-family generalization claim is not supported.

### Panel storage

| File class | Logical bytes | Content read in this audit |
|---|---:|---|
| development | 17,923,164 | Yes; metadata/development statistics and SHA-256 |
| metadata | 82,681 | Yes; metadata/development statistics and SHA-256 |
| sealed | 4,518,613 | No; size and expected identity only |
| source_archive | 2,075,982 | No; size and expected identity only |
| split_assignments | 9,812,165 | Yes; metadata/development statistics and SHA-256 |

File sizes are logical bytes, not compressed-memory use or total disk allocation. The raw archive sizes sum to 2,075,982 B, matching the frozen declared download byte counts; content hashes of those raw archives were not revalidated here because the archives also contain sealed targets. The prepared sealed file hashes are declared frozen identities, not freshly verified content checks in this audit.

## Configured product stores and local roots

AWS metadata access used profile `veox-jepson-prod` and verified account `714044927654` with STS. No credentials are included in the receipts. Current-object listings exclude object versions and deleted objects. ETags are retained as concurrency guards for metadata downloads; they are not assumed to be SHA-256 content hashes. Catalog/collection/bundle manifest downloads use the exact listed ETag and verify size; their bytes receive local SHA-256 pins. No S3 dataset blobs, sealed tables, or model weight bodies were fetched.

`jope-datasets` defaults to `s3://veox-jopedime-datasets-use2`; `jope-models` defaults to `s3://veox-jopedime-models-use2`. The known learning store is `s3://veox-jopedime-learning-use2`; the current `jope-learning` mirror resolver uses `--store`, then `JOPE_LEARNING_URL`, and otherwise no remote mirror. It must not be described as an automatic remote default. No JOPE path/store overrides were set in this audit process.

There are two source-level path conventions to name explicitly: dataset cache resolution defaults to `${XDG_CACHE_HOME:-~/.cache}/jope`, while general `jope_paths::jope_home()` resolves a marked checkout. General mutable state defaults to `${XDG_STATE_HOME:-~/.local/state}/jope`. Use an explicit root or the owning command’s resolver; do not assume the shared variable name makes the defaults identical.

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

### Additional archive, replica and custody metadata

A separate DATA-01 metadata capture at 2026-09-23T17:37:54Z extends the earlier three-store observation. It used the same verified AWS account and read current-object listings and bucket configuration only. The frozen panel and its row counts are unchanged. The author stopped the packet at 19:30 UTC. This edition imports only `data-estate-extension-summary.json` and its exact source/scope pointer, `data-estate-extension-release.json`. Raw inventories, the full packet and the separate DOCFIX-01 patch remain in custody. This is an aggregate subset, not a claim that the whole packet passed its release gates.

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

| Learning-store prefix | Objects | Logical bytes | Basis |
|---|---:|---:|---|
| `staging/lineage-b/accepted60000-restored/` | 225,207 | 110,918,425,513 | measured |
| `staging/lineage-a/encodings/` | 419,644 | 149,840,032,105 | derived |
| `staging/lineage-a/preserved-archives-20260908-F-D-E/` | 34 | 141,775,927,725 | measured |
| `staging/lineage-a/genebank/` | 5 | 1,809,949,972 | measured |
| `staging/mirror-weights/` | 391 | 26,301,123,853 | measured |

The encodings row is a residual: the earlier full-bucket total minus four later measured sibling prefixes. It was not independently re-listed in full, and the sum closes by construction across capture times. Keep this distinction when comparing inventories. The accepted-60000 restored prefix's 225,207 objects and 110,918,425,513 B also match the retained local upload inventory by file count and size; that comparison is not a fresh content restore test.

All nine inspected buckets report versioning enabled. The six `veox-jopedime-*` buckets report Object Lock enabled with no default retention; the separate dev/prod/sealed family returned no Object Lock configuration. Neither result establishes retention for each object or preservation of every prior version. This listing excludes old versions and delete markers. Sealed storage contributes aggregate counts and bytes only; no sealed object body was fetched. Object/file counts are not counts of unique datasets, admitted teachers or recoverable checkpoints.

| Local root | Present | Regular files | Logical bytes | Unique-inode logical bytes |
|---|---|---:|---:|---:|
| `/home/ubuntu/.cache/jope` | True | 170 | 36,754,881 | 35,585,244 |
| `/home/ubuntu/.local/state/jope` | False | 0 | 0 | 0 |
| `/home/ubuntu/JopeDime/datasets/test` | True | 40 | 1,096,413 | 1,096,413 |
| `/home/ubuntu/JopeDime/datasets/by-id` | False | 0 | 0 | 0 |
| `/home/ubuntu/JopeDime/models` | True | 5 | 33,687 | 33,687 |
| `/home/ubuntu/JopeDime/state` | False | 0 | 0 | 0 |

The local inventory uses `lstat`, does not follow symlinks, and counts hardlinked payload bytes once in the unique-inode column. It covers the named product roots, not every archive, research checkout or build directory on the host. The standard local state roots were absent. That does not imply the host has no learning evidence: campaign and recovery receipts live in explicit custody paths outside those defaults.

### Additional host data roots

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

All twelve roots resolve to the same filesystem. Its contemporaneous readback was 7,936,183,619,584 B total, 6,815,282,892,800 B used and 1,040,879,616,000 B available (969.395 GiB); these are dated observations, not current capacity guarantees. Multiple local paths on that filesystem do not provide independent failure-domain copies. Count/size matches with upload receipts identify recovery candidates; they do not authorize deletion or replace exact-content restore verification.

The `remote_super/quant` scan found 2,322 top-level 16-hex-ID directories with all three `train.csv`, `test.csv` and `meta.json` markers. The corrected `synth*` classification contains 25,197 `synth_<16hex>` dataset directories with those direct markers, plus two containers named `synthetic` and `synthetic_packed`. The dataset-directory class contains 86,923 files and 21,494,046,579 B; it includes more than the three payload markers per dataset. These are name/shape observations, not proof of unique, valid or train-admitted datasets. No rows, feature distributions or label bodies were opened. A pathname containing `sealed` is only a naming statistic and cannot identify every sealed/test-label payload.

### Recorded copy and retention gaps

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

The empty surveyed learning replica and uncovered `staging/` prefix leave cross-region protection unestablished for the current learning objects. Counts matching between model buckets do not establish a successful content restore. Bucket Object Lock without default retention does not prove any object's retention or legal hold; custody buckets have no Object Lock configuration. The retained custody status also records a packer stop for exhausted disk reserve. At the dated local readback, `df` reported 87% used; capacity must be rechecked before admitting later workloads. These findings call for verified preservation and owner decisions about replication/retention. They authorize neither deletion nor new cloud writes, and they do not change the original training or panel budgets.


The local dataset cache has two cataloged regression datasets, `0014b9dad5725a1e` (2,000 rows, 16 features) and `006bca140f93e411` (2,000 rows, 32 features), each with 1,000 declared training and 1,000 declared test rows. Their payloads were not reopened for this inventory. The committed historical `datasets/test/` collection is 10 datasets/40 files/1,096,413 B; it is distinct from the frozen thirty-dataset cache.

### S3 corpus catalog

The pinned current catalog is `ab9fda8d2dea46067b70a42812e9d3c1d9dc6ba025780100df34377e81aa1120`: 3,859 distinct dataset hashes, all marked real, 6,208,051 declared rows. Shape metadata sums to 265,105 features and 4,628 categorical features; sums across datasets are not deduplicated semantic quantities. The catalog contains no task explicitly named multiclass; 322 entries are multi-label, which is a different task and must not be silently relabeled.

| Catalog task | Datasets | Total declared rows | Training rows | Test rows metadata only | Feature-count range | Row-count range |
|---|---:|---:|---:|---:|---|---|
| binary | 843 | 1,490,965 | 892,307 | 598,658 | [1, 2000] | [10, 69624] |
| multi_label | 322 | 786,436 | 555,756 | 230,680 | [1, 20531] | [23, 50000] |
| regression | 2,611 | 3,025,922 | 1,898,974 | 1,126,948 | [0, 1688] | [10, 50000] |
| timeseries | 83 | 904,728 | 775,519 | 129,209 | [2, 86] | [81, 49999] |

**Catalog discrepancy:** the stored registry collection description says 1,416 unknown licenses. Counting the current catalog’s actual license-status fields gives **3,755 unknown and 104 recorded**. This chapter uses the counted fields and retains the stale description as provenance. All 2 collections remain private: `registry-real-v1` has 3,859 members and `test-10` has 10; every listed member occurs in the catalog. No payload or training eligibility validation was performed simply by counting them.

Seven regression catalog entries declare zero features. These are retained anomalies, not automatically usable supervised tasks: `5ffbe1543f683e4d`, `66fea7ec7dffa02a`, `72235646c52be632`, `7d1497edfe70495d`, `d12486ba35015a09`, `e1c1ff8cdf4ee36f`, `f773853db5b3dc9c`. Corpus metadata lacks feature distributions, missingness, actual class balance and row-level dependence evidence for many entries; those statistics remain unknown here.

The three current model bundle manifests are `hyperion-v7`, `jope-prime-gen31-v2`, and `policy-step-60000`. Their existence and object sizes do not establish new training, inference parity or serving promotion. The model/training chapter describes their roles and acceptance boundaries. Historical lineage-A material in learning storage cannot replace its missing canonical training-state chain.

## Measured current-source operational proof

| Quantity | Measured result |
|---|---|
| Source | `68f56404c3e199e06413bfd645d3fe0803d7b96f` |
| Dataset | `230_machine_cpu`, regression |
| Immutable build wall seconds (preparation; not a cold-install measurement) |170.4488760890672|
| Eight-case native context qualification wall seconds (separate preparation) |29.248627813067287|
| Run wall seconds |239.14312197000254|
| Commands started / finished |964 /964|
| Completed / failed command outcomes |834 /130|
| Train OOF / diagnostic / refit-screen commands |960 /2 /2|
| Unique complete fitted predictor artifacts |832|
| Unique reusable recipes |164|
| Successful final refits |2, each 34 predictions|
| Neural development RMSE |166.3181717454848|
| Generic development RMSE |155.25909998168862|
| Generic minus neural loss |−11.059071763796169; neural did not win|
| Known worker CPU seconds |181.220773|
| Sum of worker wall seconds |202.266164543|
| Exact supervisor aggregate CPU seconds |270.889403|
| Unknown worker CPU / wall counters |0 /0|
| Teachers admitted from this proof |0|

The supervisor figure is the exact `CPU_USAGE_NSEC=270889403000` event from local PID 1 systemd, linked to the precise supervisor unit, source, launch receipt and original experiment window. It includes controller/setup threads and child workers and excludes the outer launcher, immutable build and external preparation. It overlaps the worker CPU total; **never add them**. The rounded systemd stderr message is only diagnostic. No exact controller/setup overhead is derived by subtracting a rounded total. Isolated controller CPU, GPU seconds, token usage and currency cost remain unknown in the report.

| Denominator | Worker CPU seconds | Supervisor aggregate CPU seconds | Amortized run wall seconds |
|---|---:|---:|---:|
| One unique fitted predictor |0.2178134291|0.3255882248|0.2874316370|
| One unique recipe |1.1050047134|1.6517646524|1.4581897681|

These are accounting ratios for one retained run, not extrapolated throughput or equal-quality claims. A recipe and a fitted predictor differ: the same recipe can create distinct fitted artifacts on separate folds. Unique artifact bytes do not prove behavioral uniqueness. Every failed fit remains in the inventory; failed or unavailable commands are not silently counted as zero cost.

The supplied one-hour/ten-million-token scenario has 3,600 seconds and 10,000,000 tokens by stipulation. This run used 239.143 seconds of wall time but has no measured token or monetary receipt. A dollar comparison, token savings claim, or one-hour quality extrapolation would require actual token usage, contemporaneous input/output/cache rates and an equivalent task/quality protocol. None is invented here.

The preceding d0 source run failed after 194.789382994 seconds at scorer replay. All 964 commands had terminal receipts, with 834 completed and 130 failed. Its exact supervisor CPU was 252.614611 seconds and worker CPU 181.863951 seconds. It remains a failed reproducer; the old journal and deadline were not repaired or reopened.

## Existing historical rankings and Elo

Six committed snapshots are available and their ladder and score SHA-256 values were verified against retained bytes. No new Elo computation occurred in this audit. The snapshots use their own method pools and dataset protocols, so their Elo numbers cannot be compared across snapshots as a universal score.

| Snapshot | Rankable rows | Methods | Datasets | Interpretation |
|---|---:|---:|---:|---|
| same-cut-v1-screen |110|11|10|historical_same_cut_reproducible_snapshot_not_current_ASM 30|
| same-cut-v1-deep |50|11|10|historical_same_cut_reproducible_snapshot_not_current_ASM 30|
| same-cut-v2-screen |150|15|10|historical_same_cut_reproducible_snapshot_not_current_ASM 30|
| same-cut-v2-deep |50|15|10|historical_same_cut_reproducible_snapshot_not_current_ASM 30|
| historical-sotarena-published |8763|305|45|historical_published_best_known_test_selected_not_benchmark_valid|
| historical-sotarena-benchmark-valid |8718|304|45|historical_corrected_public_development|

Same-cut v1 covers native heads plus resampled search; v2 adds XGBoost 2.1.4, LightGBM 4.6.0, CatBoost 1.2.10 and scikit-learn 1.7.2 random forest under Python 3.12.3. Both use the favorable historical test-10 regression selection, seed 20260907, five folds, top-five deepening and 32×8 search. Deep rows include only screened finalists; coverage must be printed with rankings. Their timing fields are then-recorded wall times under uncontrolled host load, not the present isolated-worker timing protocol.

The published SOTArena snapshot puts historical `jope-dime` at rank 1/Elo 2259.8 with 11 wins and 45/45 coverage. All 45 rows are test-selected: 42 best-test evolution-envelope rows plus 3 test-adaptive accepted replays. The corrected benchmark-valid snapshot excludes those 45; **that rank 1 is not a held-out ASM claim**. The corrected snapshot has 304 methods/8,718 rankable rows. Its fixed-seed historical `jope-dime-regressor` entry is rank 31/Elo 1687.0, 1 win, 42/45 coverage; TabFM Ensemble is rank 1/Elo 1934.9. It remains historical public-development evidence.

For plots, use `performance-evidence.json.gz`: 6 complete ladder row sets; 520 same-cut score cells, including missing/rejected statuses; 20 per-dataset reports with exact cut identities, source heads and screen/deep timings; 8,763 historical SOTArena score rows; and the 2 current-origin refit loss rows. Do not blend these into one chart without explicit separate protocol panels. The current frozen 30 matrix has no completed cells yet; represent it as pending, not a fabricated ranking.

### How the retained Elo scale is constructed

`crates/jope-arena/src/elo.rs` fits regularized Bradley–Terry ratings. Each dataset contributes one equal-weight comparison for each pair of methods with available scores. The preferred direction is taken from the metric; RMSE comparisons minimize. A tie has outcome 0.5 when `|a−b| ≤ 1e−12 × max(|a|,|b|,1)`; otherwise the better score receives 1 and the other 0. Missing cells create no comparison and must not become invented defeats.

The expected pairwise outcome is `P(A beats B)=1/(1+10^((rating_B−rating_A)/400))`. The committed implementation centers at 1500, uses an L2 regularization coefficient of 1e−4, caps diagonal-Newton iterations at 250, clamps each iteration's rating update to ±40, and stops when the largest update falls below 1e−8. It preserves sorted dataset/method-pair accumulation and the historical numerical summation policy for reproduction. This describes the source algorithm; the audit did not rerun it or infer convergence where the snapshot does not report it.

A rating difference summarizes outcomes within one comparison pool. It does not encode an absolute RMSE difference, probability of unseen-dataset superiority, confidence interval, or universal ranking across tasks. A method observed only on its screened favorable subset needs its coverage displayed beside its rating. The historical `afm.lite-race.v1.resampled` row describes that search procedure; it does not prove the current neural ASM model.
