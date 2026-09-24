# Data, splits, groups, families and permitted input boundaries

Observation date: 2026-09-23T16:41:48.371336+00:00. Frozen source: `68f56404c3e199e06413bfd645d3fe0803d7b96f`, the reviewed PR83 head; `main` commit `a75e543d2166` has the same tree. These are observations at that source and time. They do not turn stored metadata, a source packet, or historical results into a current operational qualification.

Every number in this chapter comes from a pinned manifest, prepared development table, immutable report, committed historical snapshot, filesystem metadata, or authenticated AWS metadata request. The audit did not run a fit, load a model, open sealed target bodies, or execute a report command.

## The frozen thirty-dataset panel: identity, totals, groups and immutability

Panel `asm-private-30-v1` uses ten regression, ten binary, and ten multiclass datasets from PMLB revision `7c1f4bdc00136dc2e55c87fa6b8ba6e8af6d1a68`. Manifest SHA-256: `ba7dace4cd535687002cc95538dbc7f7aaf77ec973177c7159b3c6d22471eee0`. It was selected before measuring ASM outcomes, excludes the historical favorable test-10 collection, and cannot replace a dataset because a method failed or lost.

There are **55,029 total rows**, **44,048 development rows**, and **10,981 sealed rows** according to the frozen assignments. Feature counts sum to 522 across datasets; this is not a union of unique semantic variables. Development data contains 1,612 missing feature cells in 302 rows. All thirty development, split-assignment and metadata file SHA-256 values match the frozen manifest. Every row ID is unique within its dataset, and every group stays wholly within one original development fold or sealed partition.

The panel is immutable once comparison begins. Its identity covers dataset metadata, original labels and types, row IDs, group membership, original development folds, sealed membership and prepared data hashes. Do not replace difficult datasets, reshuffle rows, map categories to arbitrary numeric order, drop missing rows or substitute a different target to make a task executable. Missing source subject or time identifiers remain limitations even when duplicate-feature groups are protected.

Known date columns and identical-feature groups are retained by the prepared split. Missing original subject/time identifiers remain a limitation. A grouped split cannot establish independence when the source never supplied those identifiers. Checkpoint exposure is unknown for these datasets; an unseen-family generalization claim is not supported. Pretrained-encoder exposure is also unknown, and the panel remains an adaptive development domain ([§](#three-family-level-domains)).

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

## Family-level domains and the permitted input closure

This section states the owner's edition-2 rules for which data may reach teaching, training, selection and evaluation. They are requirements, not results. Each carries one status as of 2026-09-23 about 23:55 UTC: available, implemented but unqualified, pending integration or proposed.

### Three family-level domains

Domains are assigned to dataset families, not to rows, files or split names. A split name alone cannot prove an unseen family ([§](learning.md#freezing-the-experiment-and-choosing-families)).

| Domain | Permitted use | Status |
|---|---|---|
| Teacher/train | Train-side evidence may produce eligible teachers and policy-training rows | Pending integration: the stopped `training64` packet partitions admitted families into `train` and `development` sides only (pending `crates/jope-cli/src/train/corpus.rs:196-264`) |
| Adaptive development | Labels may guide screening, selection, confirmation and model monitoring; results are adaptive, not independent | Implemented but unqualified: every panel report at `a75e543d2` records `confirmation_role: "adaptive_development"` (`crates/jope-bench/src/panel/campaign.rs:894`); no full panel has run |
| Evaluation outside training and selection | Families whose labels informed no teaching, training, selection or monitoring | Proposed for this ASM path. The legacy engine assigns a sealed meta-test family partition and withholds its observations from corpus records, but the legacy trainer stages all indexed descriptors, including the 878 meta-test descriptors; that does not establish their exclusion from the worker input closure |

The frozen panel `asm-private-30-v1` remains an adaptive development domain. Its development labels drive the screen, freeze and confirm procedure ([§](benchmark.md#adaptive-development-exposure)), and a panel family that supplies teachers must additionally be reported as exposed ([§](learning.md#freezing-the-experiment-and-choosing-families)). Because domains are family-level, withholding the panel's sealed rows does not place its families in the third domain. Prior exposure is unknown in two separate places: the accepted policy checkpoint, which `optimize plan` records as `checkpoint_training_exposure: "unknown"` (`crates/jope-bench/src/panel/optimize_cli.rs:128`), and the fixed pretrained encoder assets that produce the proposal context ([§](architecture.md#component-dimensions-and-assembly)). Pretrained-encoder exposure remains unknown; neither unknown may be reported as zero.

### Hashes, eligibility and the complete input closure

A hash is not eligibility. A digest authenticates bytes; it does not decide whether those bytes may be used. The legacy trainer at `a75e543d2` checks every indexed context's bytes against its recorded `sha256` digest and expected size, then stages all of them on the device (`crates/jope-engine/src/training.rs:1187-1203`), so the 878 `meta_test` descriptors in the retained index would pass that integrity check ([§](#record-split-and-the-meta-test-closure-question)). Eligibility comes from each input's admission record and family domain ([§](teachers.md#the-admission-guard-label-exposure-and-hwm-identity)), never from a matching digest.

The permitted input closure covers everything a teaching or training worker can read, directly or transitively: records, context descriptors and tensors, fitted and frozen transforms, and all assets, including encoder and policy weights, vocabulary, admission manifests and checkpoints. Its parts have different status:

| Closure requirement | Status as of 2026-09-23 about 23:55 UTC |
|---|---|
| Native projection of admitted contexts | Pending integration: the stopped `training64` preparation requires the requested contexts to cover exactly the admitted comparisons and writes only the selected row contexts (pending `crates/jope-cli/src/train/corpus.rs:572-575`, `:600-635`) |
| One authenticated inventory of every permitted worker read and its role | Proposed: the same packet's Python campaign opens the complete context index, encoder, vocabulary and admission file, and validates only the contexts that retained rows reference (pending `python/src/jope/policy/training_campaign.py:298-363`) |
| Rejection of unused forbidden descriptors before loading | Proposed: neither current nor pending source rejects extra context-index entries, and the legacy native route still stages every indexed context |

An unused forbidden descriptor, such as a `meta_test` or evaluation-domain context that no admitted row references, must be rejected before any model, tensor or corpus is loaded; leaving it unsampled is not enough. The tests this requires do not exist yet: an injected unused forbidden context, an unlisted reachable artifact, invariance under mutation of an excluded artifact, and invariance of preprocessing and context under changed holdout labels. Source: the read-only boundary investigation `plan-execution-20260923/boundary-resume-siblings.md` in the ASM custody root.

### Datasets outside the frozen panel

Arbitrary ingestion is separate from the frozen panel. Status: proposed. No arbitrary-dataset ingestion command exists at `a75e543d2`: the one-dataset ASM front door takes a dataset already present in the frozen panel ([§](search.md#front-door-and-safe-metadata-checks)), and a fetchable catalog file is not automatically an admitted ASM input ([§](installation.md#command-surfaces-and-the-stopped-integration-order)). A separate authenticated ingestion request must preserve the new dataset's labels, categories, missingness and groups and assign its family to a domain; it never edits the frozen manifest ([§](search.md#what-is-missing-for-a-sustained-incumbent-seeded-dataset-attack)). Results on an ingested dataset are not frozen-panel results, and frozen-panel results do not qualify an ingested dataset.
