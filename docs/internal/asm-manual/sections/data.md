# Data, splits, groups, families and permitted input boundaries

Observation date: 2026-09-23T16:41:48.371336+00:00. Frozen source: `68f56404c3e199e06413bfd645d3fe0803d7b96f`, the reviewed PR83 head; `main` commit `a75e543d2166` has the same tree. These are observations at that source and time. They do not turn stored metadata, a source packet, or historical results into a current operational qualification.

Every number in this chapter comes from a pinned manifest, prepared development table, immutable report, committed historical snapshot, filesystem metadata, or authenticated AWS metadata request. The audit did not run a fit, load a model, open sealed target bodies, or execute a report command.

## The frozen thirty-dataset panel: identity, totals, groups and immutability

Panel `asm-private-30-v1` uses ten regression, ten binary, and ten multiclass datasets from PMLB revision `7c1f4bdc00136dc2e55c87fa6b8ba6e8af6d1a68`. Manifest SHA-256: `ba7dace4cd535687002cc95538dbc7f7aaf77ec973177c7159b3c6d22471eee0`. It was selected before measuring ASM outcomes, excludes the historical favorable test-10 collection, and cannot replace a dataset because a method failed or lost.

There are **55,029 total rows**, **44,048 development rows**, and **10,981 sealed rows** according to the frozen assignments. Feature counts sum to 522 across datasets; this is not a union of unique semantic variables. Development data contains 1,612 missing feature cells in 302 rows. All thirty development, split-assignment and metadata file SHA-256 values match the frozen manifest. Every row ID is unique within its dataset, and every group stays wholly within one original development fold or sealed partition.

The panel is immutable once comparison begins. Its identity covers dataset metadata, original labels and types, row IDs, group membership, original development folds, sealed membership and prepared data hashes. Do not replace difficult datasets, reshuffle rows, map categories to arbitrary numeric order, drop missing rows or substitute a different target to make a task executable. Missing source subject or time identifiers remain limitations even when duplicate-feature groups are protected.

Known date columns and identical-feature groups are retained by the prepared split. Missing original subject/time identifiers remain a limitation. A grouped split cannot establish independence when the source never supplied those identifiers. Checkpoint exposure is unknown for these datasets; an unseen-family generalization claim is not supported.

## Local preparation and panel storage

Local preparation: `/home/ubuntu/.cache/jope/asm-private-30-v1/`. `development/` contains the permitted development tables; `splits/` contains original row/group/fold assignments; `metadata/` contains pinned upstream YAML; `downloads/` contains the original compressed archives; `sealed/` contains withheld partitions. The audit opened development, assignment, and metadata files only. It inspected sizes and expected hashes of raw and sealed files without opening their bodies.

| File class | Logical bytes | Content read in this audit |
|---|---:|---|
| development | 17,923,164 | Yes; metadata/development statistics and SHA-256 |
| metadata | 82,681 | Yes; metadata/development statistics and SHA-256 |
| sealed | 4,518,613 | No; size and expected identity only |
| source_archive | 2,075,982 | No; size and expected identity only |
| split_assignments | 9,812,165 | Yes; metadata/development statistics and SHA-256 |

File sizes are logical bytes, not compressed-memory use or total disk allocation. The raw archive sizes sum to 2,075,982 B, matching the frozen declared download byte counts; content hashes of those raw archives were not revalidated here because the archives also contain sealed targets. The prepared sealed file hashes are declared frozen identities, not freshly verified content checks in this audit.

## The broader S3 corpus catalog

### Catalog composition by task

The pinned current catalog is `ab9fda8d2dea46067b70a42812e9d3c1d9dc6ba025780100df34377e81aa1120`: 3,859 distinct dataset hashes, all marked real, 6,208,051 declared rows. Shape metadata sums to 265,105 features and 4,628 categorical features; sums across datasets are not deduplicated semantic quantities. The catalog contains no task explicitly named multiclass; 322 entries are multi-label, which is a different task and must not be silently relabeled.

| Catalog task | Datasets | Total declared rows | Training rows | Test rows metadata only | Feature-count range | Row-count range |
|---|---:|---:|---:|---:|---|---|
| binary | 843 | 1,490,965 | 892,307 | 598,658 | [1, 2000] | [10, 69624] |
| multi_label | 322 | 786,436 | 555,756 | 230,680 | [1, 20531] | [23, 50000] |
| regression | 2,611 | 3,025,922 | 1,898,974 | 1,126,948 | [0, 1688] | [10, 50000] |
| timeseries | 83 | 904,728 | 775,519 | 129,209 | [2, 86] | [81, 49999] |

### Licence metadata, anomalies and the corpus figure

**Catalog discrepancy:** the stored registry collection description says 1,416 unknown licenses. Counting the current catalog’s actual license-status fields gives **3,755 unknown and 104 recorded**. This chapter uses the counted fields and retains the stale description as provenance. All 2 collections remain private: `registry-real-v1` has 3,859 members and `test-10` has 10; every listed member occurs in the catalog. No payload or training eligibility validation was performed simply by counting them.

Seven regression catalog entries declare zero features. These are retained anomalies, not automatically usable supervised tasks: `5ffbe1543f683e4d`, `66fea7ec7dffa02a`, `72235646c52be632`, `7d1497edfe70495d`, `d12486ba35015a09`, `e1c1ff8cdf4ee36f`, `f773853db5b3dc9c`. Corpus metadata lacks feature distributions, missingness, actual class balance and row-level dependence evidence for many entries; those statistics remain unknown here.

The broader catalog has 3,859 real datasets and 6,208,051 metadata-reported rows. It is a different collection from the frozen panel. Its `multi_label` task is not interchangeable with admitted multiclass classification. Per-payload statistics for the entire corpus were not recomputed.

![Corpus task counts and license metadata completeness](../figures/inventory-corpus.png)

The catalog has 3,755 entries with unknown license metadata and 104 with a recorded license. The collection's older prose count of 1,416 unknown licenses is inconsistent with this current pinned catalog. This is a metadata quality gap; neither the chart nor a repository-level license resolves each dataset's provenance. See [§](data-estate.md) for exact storage bytes, prefixes and observation scope.

## The historical input closure and the meta-test question

### Record split and the meta-test closure question

The accepted corpus's record split is precise: 9,382,874 `meta_train` records in 6,836 shards and 831,458 `meta_validation` records in 1,041 shards. There are no `meta_test` record shards. The context index nevertheless contains 6,962 training, 1,054 validation and **878 meta-test context descriptors**. Current legacy `training.rs` stages every context in that index before sampling records. Consequently, withholding meta-test record labels is not a proof that sealed families were excluded from the entire historical encoder/trainer input closure. This is a concrete remaining closure audit, not a claim that this documentation audit opened sealed vectors or labels.

A separate external historical-inventory report also discloses parsing manifests for eight datasets marked `meta_test` **and hashing their source CSVs**; its shorter coordination entry mentioned manifests only. That report says all eight have lineage-A `test_exact` high-water marks. This audit did not reopen those paths or inspect their labels. The disclosure does not establish whether labels reached lineage-B training. Before reuse or continuation under the current policy, audit the complete encoder, context-staging, target-admission and sampler input closure, and resolve the old exposed-HWM lineage. Neither zero meta-test record shards nor an old qualification flag closes that question.

The accepted index SHA `1c840597…`, vocabulary `8d55fee4…` and context index `d77e856b…` bind the preserved checkpoint lineage. Its label policy is explicitly `reference_v1`. An old `qualification: true` flag means qualification under that historical implementation and policy. It does not retroactively satisfy current finite train-side OOF improvement over a trivial baseline, improvement on four of five folds, exact source closure and non-exposure requirements.

### Audit the entire input closure before continuation

Before any future continuation, audit the **entire** input closure, including every context descriptor and staged tensor, not only training records. The retained snapshot 97 index lists 8,894 contexts, including 878 `meta_test` descriptors, while record rows are train/validation only. Current legacy training stages all indexed contexts before sampling. Absence of sealed record rows therefore does not prove strict sealed-family exclusion. Do not blindly continue that corpus under the new closure policy; preserve the metadata observation and resolve it without opening sealed payloads. See [§](#record-split-and-the-meta-test-closure-question) and [context staging source](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-engine/src/training.rs#L1179).

[§](#record-split-and-the-meta-test-closure-question) records unresolved historical input closure: meta-test context descriptors were staged by the legacy trainer, while the external author disclosed meta-test manifest inspection and source-CSV hashing alongside old exposed high-water marks. Do not treat these findings as proof of training-label leakage or dismiss them because the packed record split excludes meta-test rows. Resolve the complete allowed-input closure and current teacher eligibility before continuing the old corpus under the new policy.
