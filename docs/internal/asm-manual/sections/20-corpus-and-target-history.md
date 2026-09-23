# Corpus and target history

The retained evidence shows both substantial historical training corpora and some improving historical GP targets. It does **not** show a newly qualified ASM teaching loop, a current thirty-dataset result, or a causal advantage from neural training. Those are different claims with different evidence requirements.

This chapter answers two questions: which historical corpora and target records survive, and how the historical target collections and their observed scores changed. What one training example contains is described in [§](10-teachers-and-targets.md#a-dataset-is-not-one-training-example), and how the implementation stores and loads packed records in [§](11-corpora-and-packing.md). This manual's audit read metadata, manifests and development-score receipts. It did not read sealed labels, model tensors, packed target-record bodies or the large audit-label sidecars, and it ran no training or fits. External packing, inventory and seed audits report results that depend on payloads this audit did not open; those results are attributed separately below and were checked here through retained metadata, scripts and arithmetic, without replaying the payload reads. File-metadata observations, externally reported payload checks and source-derived calculations remain distinct evidence levels.

## What is preserved, and what was lost

### Lineages A and B and the staging store

Two historical lineages must remain separate. Lineage A contains July–September flywheel material, old winner/genebank state and preserved archives. Lineage B contains the native corpus/training chain used by the retained step-60000 checkpoint. `ADR-0005` and `docs/learning.md` describe their intended migration and their observed losses; those dated records do not authorize silently promoting old labels under the current ASM rules.

The learning S3 store is a staging estate: at the 2026-09-23 listings all 645,281 objects in `veox-jopedime-learning-use2` were under `staging/`. Targeted metadata listings found `staging/lineage-a/`, `staging/lineage-b/accepted60000-restored/` and `staging/mirror-weights/`. [§](C-data-estate.md#product-s3-buckets) and [§](C-data-estate.md#learning-store-prefixes) record the bucket census and the per-prefix counts. Object upload time is not the time a target was discovered or consumed. Five small metadata objects were fetched with an ETag precondition and hashed: the encoding caveats, frozen slice listing, archive catalog, genebank deprecation note and accepted60000 materialization intent. Their exact keys, byte lengths, ETags, upload times and fetch hashes are in `metadata/fetched-metadata.json` in custody.

### Lineage-A selections, archives, lost records and exposed score tiers

The frozen July 25 `slice500` listing chose the first 500 bytewise-sorted names from 25,031 directories, excluding 186 incoming entries and hidden packed storage. That is a reproducible selection of candidate datasets, not proof of 500 admitted teachers. The retained encoding caveat says the historical binary target encoder used a regression-flavored checkpoint: binary values were in range, but that did not establish classification training qualification. The same note describes historical target-vector fusion and encoder pins. It must not be recast as the current verified classification-context contract.

Lineage A preserves archives for snapshots `004754860`, `008903292`, `009658180` and `009915070`. The archives' compressed byte sizes and checksums identify preservation containers, not in-memory training size. This audit did not extract them. The historical records declare the strict-test-exact database, pair registry, later immutable checkpoint descendants, acceptance pointers, sealed-consumption receipts and task-Elo registry lost. A directory number or an old selected pointer cannot recreate that missing evidence. The old genebank has an improvement-event schema with task, dataset, score tier, metric, evaluator, source and ingestion time, but lacks the complete cut/vocabulary/protocol identity required by the current HWM. Its 22,798 improvement-event rows are not 22,798 currently admissible teachers; its timestamps include backfill ingestion.

The exact-current HWM identity and current admission requirements are explained in [§](10-teachers-and-targets.md#the-admission-guard-label-exposure-and-hwm-identity) and [§](13-flywheel.md#four-linked-improvements-and-the-comparison-identity). The historical `test_exact` score tier is optimization-exposed development evidence when used to choose programs. Renaming the tier does not restore independence.

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

### Record split and the meta-test closure question

The accepted corpus's record split is precise: 9,382,874 `meta_train` records in 6,836 shards and 831,458 `meta_validation` records in 1,041 shards. There are no `meta_test` record shards. The context index nevertheless contains 6,962 training, 1,054 validation and **878 meta-test context descriptors**. Current legacy `training.rs` stages every context in that index before sampling records. Consequently, withholding meta-test record labels is not a proof that sealed families were excluded from the entire historical encoder/trainer input closure. This is a concrete remaining closure audit, not a claim that this documentation audit opened sealed vectors or labels.

A separate external historical-inventory report also discloses parsing manifests for eight datasets marked `meta_test` **and hashing their source CSVs**; its shorter coordination entry mentioned manifests only. That report says all eight have lineage-A `test_exact` high-water marks. This audit did not reopen those paths or inspect their labels. The disclosure does not establish whether labels reached lineage-B training. Before reuse or continuation under the current policy, audit the complete encoder, context-staging, target-admission and sampler input closure, and resolve the old exposed-HWM lineage. Neither zero meta-test record shards nor an old qualification flag closes that question.

The accepted index SHA `1c840597…`, vocabulary `8d55fee4…` and context index `d77e856b…` bind the preserved checkpoint lineage. Its label policy is explicitly `reference_v1`. An old `qualification: true` flag means qualification under that historical implementation and policy. It does not retroactively satisfy current finite train-side OOF improvement over a trivial baseline, improvement on four of five folds, exact source closure and non-exposure requirements.

## Did GP targets get better?

### Carried-forward scores and the admitted receipt inventory

**Some recorded carried-forward best scores became lower.** These are historical development outcomes from searches sharing incumbent and teacher seeds. They are not independent discoveries by each arm, and their no-worsening pattern is partly structural. Corpus growth alone would not establish even this narrower result.

The audit enumerated 29,744 `search.json` paths under `/home/ubuntu/jope-node-mirror/results`, which resolves into `/home/ubuntu/jain-split/jain-datasets/.runtime/node-archive-mirror-20260914/results`. This source covers generations 0–70. The audit rejected 3,387 paths outside the accepted context index's `meta_train` dataset/cut allowlist **before opening their score files**. The remaining 26,357 receipts identify adaptive development-validation RMSE, prepared data, vocabulary, evaluator, execution protocol and completion metadata. Each source file and observation ledger has a distinct recorded digest. Their recorded completion times span September 12, 2026 18:37:13 UTC to September 14, 2026 13:37:43 UTC. These times are receipt metadata, not a new replay of the fits.

Only 3,530 of the 26,357 permitted receipts are also present in the separately preserved `accepted60000-restored` tree, all byte-identical; the other 22,827 receipts, including every permitted generation 10–70 receipt, are absent from that tree. A custody archive preserves all **26,357 original search receipts**, totaling **62,271,855 uncompressed bytes**. A separate streaming verification checked every member name, size and original SHA256 without reopening the source bodies. `original-target-receipts-custody.json` pins the archive and verification. The archive is a second local copy on the same filesystem, not an off-host backup. The separate seed bodies used by the external carry-forward analysis are outside that archive.

### First-to-last comparison at 1,000 evaluations

For a readable comparison, the table restricts each repeated comparison to exactly 1,000 completed evaluations, matching its recorded dataset/family, cut, prepared SHA, vocabulary SHA, evaluator SHA/protocol, execution-order protocol and run. It compares the first retained generation with the last retained generation within that group. Lower RMSE is better. Improvement and worsening use exact `<` and `>` comparisons of the parsed reported scores; unchanged means exact equality, with no epsilon. The analysis script applies no additional cumulative minimum, but the underlying search already carries incumbent/teacher programs forward. Thus the reported per-generation best usually behaves like an inherited incumbent rather than a fresh independent draw, but it is not strictly monotone. In the 9,536 adjacent 1,000-evaluation transitions where the previous best was re-seeded, the recorded best stayed equal 9,406 times, fell 130 times and never rose; all 27 rises occur in the 76 transitions where it was not re-seeded. The score field is `best_validation_rmse` under the historical `featmath_training_fold_normalized_v1` evaluator; the plots do not claim original target units.

| Historical search arm | Repeated arm groups | Last lower | Same | Last higher | Median relative reduction, all groups (unchanged included) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Neural | 51 | 19 | 32 | 0 | 0% |
| Generic | 54 | 22 | 32 | 0 | 0% |
| Shuffled | 54 | 23 | 31 | 0 | 0% |

![First-to-last carried-forward best scores at 1,000 evaluations; the arms share seeds. The 64 lower arm groups cover 23 distinct dataset/cuts.](../figures/target-quality-progress-counts.png)

The 159 arm groups cover **55 distinct dataset/cut pairs**. The 64 lower arm groups (19 + 22 + 23) cover only **23 distinct pairs**; 22 of those 23 end at the same score in every available arm. These counts were independently recomputed from the published comparison CSV. The denominators and shared seed pools make the arm counts dependent; one discovery can later appear in multiple arms.

The separately released external seed audit reports `seed_population_supplied=true` and `teacher_first_v1` for all 26,357 receipts. The previous recorded best appears in the next seed set in 25,790 of 25,985 transitions; 26,051 reported winners (98.84%) are already in that search's seed population. Incumbent, retained-teacher and exploration seed sets are identical across arms in all 8,785 multi-arm dataset/cut/generation groups checked. Those seed-body counts are externally reported and metadata-checked here, not a new local seed-payload replay. They explain why zero first-to-last worsening is not independent evidence of stability. A lower inherited score is also not a new discovery by the receiving arm. `carry-forward-audit.json` preserves these evidence boundaries.

This table is therefore not a paired policy-ranking result. Random seeds change between generations, elapsed time varies, and this audit did not requalify the actual CPU/memory/hardware enforcement or complete build closure of the historical searches. Matching these recorded fields is narrower than the complete current experiment identity. The recorded generation manifests declare 600 seconds and 1,000 evaluations for the illustrative example; those declarations do not establish equal consumed wall time. Checkpoint association is available by generation, but exact exposure of each improved target to later optimizer batches remains unjoined.

### The pine-chromium example, attribution limits and published data

A concrete example is `mendeley-6bspjkt7bg-pine-chromium`, dataset `b68222ad60398ea8`, cut 1. Across generations 0–70 the recorded best development RMSE changes from **0.009749860799638275 to 0.003185684481133936**, a **67.3258% relative reduction**. All three arms finish at the same score. The neural arm first records that final value at generation 43; generic and shuffled do so at generation 42. The external seed audit identifies the final program outside the generic and shuffled seed sets at generation 42, then as an `incumbent_cut_teacher` seed in all three arms at generation 43. The neural arm's later equality is carry-forward, not evidence that it independently discovered that program. This identifies generation-level observations; it does not establish which of two same-generation arms discovered the program first in wall-clock time. The recorded best is not monotone here either: generic reports 0.007209 at generation 29 and 0.009750 at generation 30, and generic and shuffled both rise from 0.004698 at generation 39 to 0.006747 at generation 40; the neural arm first reports 0.007209 at generation 39. The plot data include all 213 generation/arm pairs, with no interpolated missing generation. This example was selected after inspection as the largest first-to-last relative improvement in the neural 1,000-evaluation subset; it is illustrative, not representative.

![Post-hoc pine-chromium cut 1 history with shared teachers: generic/shuffled record the final program outside their seeds at generation 42; all arms inherit it at generation 43.](../figures/target-quality-pine-example.png)

Relative improvement is `(first_RMSE - last_RMSE) / first_RMSE`. The percentage is meaningful within that dataset/cut. Raw RMSEs from unlike target scales are not averaged to claim overall progress. A running-best curve, if separately shown, must be labelled as such: its monotonicity follows partly from selection. Better retained minima do not prove that a trained neural proposal policy caused the gain. Shared incumbent/teacher seed pools and carry-forward are established in the external audit; search history and ordinary exploration also prevent attribution to policy learning. Count a new search discovery separately from a copied incumbent: authenticate the full seed set, identify the first measured program absent from it, and retain the discovering arm, generation, context, program and parent identities. The aggregate table above does not provide that full per-program discovery ledger.

The small publication files are `target-quality-1000-summary.csv`, `target-quality-1000-comparisons.csv` and `target-quality-pine-chromium-cut1.csv`. Full 26,357-row extracted evidence stays in custody, with its hash in `historical-target-quality-summary.json`; the manual need not embed the 19.5 MB full extraction. This analysis reads development scores only and creates no new teacher eligibility, promotion or benchmark-valid result.

### Paired final-search outcomes within the admitted development inventory

A metadata-only comparison pairs arms **within the same generation**, keeping the recorded run, dataset/family, cut, prepared SHA, vocabulary SHA, evaluator SHA/protocol, execution-order protocol and exactly 1,000 completed evaluations equal. It uses the existing 26,357 admitted receipts, not a newly expanded input boundary. Scores are finite positive `best_validation_rmse`; lower is better. Each comparison counts exact score equality, with no epsilon.

| Neural compared with | Paired observations | Neural lower | Equal | Neural higher | Geometric mean neural/control RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Generic | 3,268 | 24 | 3,227 | 17 | 1.0001869094 |
| Shuffled | 3,253 | 24 | 3,213 | 16 | 1.0000854976 |

These are repeated adaptive **final-search** observations across generations 0–70, not measurements of initial neural seed quality. The generic comparison covers 52 dataset/cut pairs from 30 families; the shuffled comparison covers 52 pairs from 31 families. Reported wall times differ in every pair. Families, contexts and generations repeat; the arms share incumbent, retained-teacher and exploration seeds, and the search can recover from poor initial proposals. The paired rows are not independent arm discoveries. No independent-sample confidence interval, causal advantage or failure-inclusive benchmark claim follows. The ratios are descriptive and close to one; they do not establish that training improved routing.

The external report's broader routing extract has 42,863 rows. Only 3,530 match this chapter's admitted receipt inventory by retained path and all checked score, prepared-data, vocabulary, evaluator, protocol, completed-evaluation and wall-time fields. The other 39,333 are outside that existing inventory; their eligibility remains unqualified here, **not classified as all sealed**. No original score path was reopened for this join. The broader campaign ratios and p-values are therefore not imported. The compact `packing-routing-audit.json` pins inputs, matching fields, arithmetic and exclusions.

To establish a closed improvement loop rather than these separate observations, join: original search/fit completion → exact program and comparison identity → admission event and immutable target record → corpus membership → actual sampler exposure and optimizer continuation → later policy proposal/fit under the same comparison protocol. The historical logs in [§](19-training-history.md) show training activity; the target receipts here show some improving, largely carried-forward development scores. That complete causal/eligibility join is still missing.
