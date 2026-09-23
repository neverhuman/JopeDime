# Training examples, target history and binary packing

The retained evidence shows both substantial historical training corpora and some improving historical GP targets. It does **not** show a newly qualified ASM teaching loop, a current thirty-dataset result, or a causal advantage from neural training. Those are different claims with different evidence requirements.

This chapter answers three questions: what one training example contains; how the historical target collections and their observed scores changed; and how efficiently the implementation stores and loads them. The original manual audit read metadata, manifests and development-score receipts. It did not read sealed labels, model tensors, packed target-record bodies or the large audit-label sidecars, and it ran no training or fits. A later external packing audit reports checks over the accepted corpus payloads; the separately attributed results below were checked here through its retained metadata, scripts and arithmetic, without replaying those payload reads. File-metadata observations, externally reported payload checks, source-derived size calculations and proposed formats remain distinct evidence levels.

## 15.1 A dataset is not one training example

A training context describes a particular dataset, split and encoding. Many executable programs can share that context. Each program supplies a separate target sequence and measured supervision. The useful accounting units are therefore:

| Unit | Meaning | Why the distinction matters |
| --- | --- | --- |
| Dataset/family | A problem and its related datasets | Family partitions control generalization; row counts alone do not identify independence. |
| Cut/context | A dataset split plus prepared-data and encoder identities | The same dataset can have many contexts. Reusing a context must preserve its actual split and fitting rows. |
| Program observation | A measured evaluation of a concrete executable recipe | Failed, duplicated, incompatible or exposed observations need not become training examples. |
| Admitted target record | A representable program and its supervision for one context | Repeated observations of the same program/cut do not automatically earn repeated votes. |
| Corpus snapshot | An immutable collection of target records and identities | Its record count is not the number of optimizer updates or unique datasets. |
| Presentation | A sampled training example used in an update | Sampling revisits records. Millions of presentations do not mean millions of newly discovered targets. |

The legacy Rust context is 2,048 little-endian float32 values, or 8,192 bytes. The model receives a 1,536-value dataset lane and a separate 512-value target-statistics lane. The historical encoder recipe described the dataset lane as Hyperion 512 plus Foundation/PCA 1,024. These vectors are shared by context identity; they are not repeated inside every 532-byte target record. The program target is a grammar-token sequence, with the legal-token set for each step, context ID, score-derived value, percentile and log-runtime supervision.

A concrete metadata binding in the accepted step-60000 corpus connects dataset `031201c6895f724b`, family `rdatasets-agridat-kalamkar-potato-uniformity`, cut 0 and `meta_train` to a prepared view and context vector. Its context SHA begins `6f5fdd79…`, prepared-view SHA `811feabd…`, and content-addressed shard contains 24,231 records. The full paths and hashes are in `accepted-corpus-binding.json`. This audit verified that metadata relation without opening the encoded vector or target records.

The learning chain is:

1. Pin the raw dataset, family partition and actual fitting/development row assignments.
2. Fit preprocessing and encoding only within the permitted input closure; pin their bytes and versions.
3. Generate a concrete executable program, fit it and retain its outcome, measurement, resource use and evaluator identity.
4. Apply the declared admission policy; reject failed, incompatible, exposed or unsupported targets.
5. Tokenize the exact program and retain original observations beside the training representation.
6. Sample the admitted record, update parameters, and prove checkpoint reload/continuation.
7. Evaluate the new policy through actual proposals and fits under the comparison protocol.

A file at step 5 does not prove steps 6–7. Conversely, a lower optimizer loss does not establish better executable proposals.

## 15.2 What is preserved, and what was lost

Two historical lineages must remain separate. Lineage A contains July–September flywheel material, old winner/genebank state and preserved archives. Lineage B contains the native corpus/training chain used by the retained step-60000 checkpoint. `ADR-0005` and `docs/learning.md` describe their intended migration and their observed losses; those dated records do not authorize silently promoting old labels under the current ASM rules.

The learning S3 store is still a staging estate. Targeted metadata listings found `staging/lineage-a/`, `staging/lineage-b/accepted60000-restored/` and `staging/mirror-weights/`. The broader data chapter records the full bucket census. Object upload time is not the time a target was discovered or consumed. Five small metadata objects were fetched with an ETag precondition and hashed: the encoding caveats, frozen slice listing, archive catalog, genebank deprecation note and accepted60000 materialization intent. Their exact keys, byte lengths, ETags, upload times and fetch hashes are in `metadata/fetched-metadata.json` in custody.

The frozen July 25 `slice500` listing chose the first 500 bytewise-sorted names from 25,031 directories, excluding 186 incoming entries and hidden packed storage. That is a reproducible selection of candidate datasets, not proof of 500 admitted teachers. The retained encoding caveat says the historical binary target encoder used a regression-flavored checkpoint: binary values were in range, but that did not establish classification training qualification. The same note describes historical target-vector fusion and encoder pins. It must not be recast as the current verified classification-context contract.

Lineage A preserves archives for snapshots `004754860`, `008903292`, `009658180` and `009915070`. The archives' compressed byte sizes and checksums identify preservation containers, not in-memory training size. This audit did not extract them. The historical records declare the strict-test-exact database, pair registry, later immutable checkpoint descendants, acceptance pointers, sealed-consumption receipts and task-Elo registry lost. A directory number or an old selected pointer cannot recreate that missing evidence. The old genebank has an improvement-event schema with task, dataset, score tier, metric, evaluator, source and ingestion time, but lacks the complete cut/vocabulary/protocol identity required by the current HWM. Its 22,798 improvement-event rows are not 22,798 currently admissible teachers; its timestamps include backfill ingestion.

The exact-current HWM identity and current admission requirements are explained in chapters 04 and 17. The historical `test_exact` score tier is optimization-exposed development evidence when used to choose programs. Renaming the tier does not restore independence.

## 15.3 Retained corpus sizes

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

A separate September 11 audit captured snapshot `000421` with 8,899,150 records and 24,313,972 observations seen; its original index hash is `74d643be…`. That is a retained audit observation, not a seventh newly payload-verified corpus. The later accepted `000097` belongs to a different recovery run: 97 is not chronologically earlier than 421. The apparent count difference cannot be called a count of newly discovered unique teachers without joining actual program/context identities and admission events.

For accepted snapshot 97, metadata records 34,321,997 observations seen, 109,797 failed programs, 488,028 withheld meta-test observations, 1,338,630 observations without a matching context, 405,322 exact-parameter tokenization rejections and 112,903 out-of-vocabulary parameter rejections. These diagnostics describe the packer's accounting categories; do not assume they form a disjoint partition that can simply be subtracted from the observation total.

The accepted corpus's record split is precise: 9,382,874 `meta_train` records in 6,836 shards and 831,458 `meta_validation` records in 1,041 shards. There are no `meta_test` record shards. The context index nevertheless contains 6,962 training, 1,054 validation and **878 meta-test context descriptors**. Current legacy `training.rs` stages every context in that index before sampling records. Consequently, withholding meta-test record labels is not a proof that sealed families were excluded from the entire historical encoder/trainer input closure. This is a concrete remaining closure audit, not a claim that this documentation audit opened sealed vectors or labels.

A separate external historical-inventory report also discloses parsing manifests for eight datasets marked `meta_test` **and hashing their source CSVs**; its shorter coordination entry mentioned manifests only. That report says all eight have lineage-A `test_exact` high-water marks. This audit did not reopen those paths or inspect their labels. The disclosure does not establish whether labels reached lineage-B training. Before reuse or continuation under the current policy, audit the complete encoder, context-staging, target-admission and sampler input closure, and resolve the old exposed-HWM lineage. Neither zero meta-test record shards nor an old qualification flag closes that question.

The accepted index SHA `1c840597…`, vocabulary `8d55fee4…` and context index `d77e856b…` bind the preserved checkpoint lineage. Its label policy is explicitly `reference_v1`. An old `qualification: true` flag means qualification under that historical implementation and policy. It does not retroactively satisfy current finite train-side OOF improvement over a trivial baseline, improvement on four of five folds, exact source closure and non-exposure requirements.

## 15.4 Did GP targets get better?

**Some recorded carried-forward best scores became lower.** These are historical development outcomes from searches sharing incumbent and teacher seeds. They are not independent discoveries by each arm, and their no-worsening pattern is partly structural. Corpus growth alone would not establish even this narrower result.

The audit enumerated 29,744 `search.json` paths under `/home/ubuntu/jope-node-mirror/results`, which resolves into `/home/ubuntu/jain-split/jain-datasets/.runtime/node-archive-mirror-20260914/results`. This source covers generations 0–70. The later cross-check found only 3,530 of the 26,357 permitted receipts in the separately preserved `accepted60000-restored` tree, all byte-identical: 22,827 original receipts, including every permitted generation 10–70 receipt, were missing from that preservation tree. A subsequent custody archive now preserves all **26,357 original search receipts**, totaling **62,271,855 uncompressed bytes**. A separate streaming verification checked every member name, size and original SHA256 without reopening the source bodies. `original-target-receipts-custody.json` pins the archive and verification. This closes the identified search-receipt single-location gap through a second local copy on the same filesystem; it is not an off-host backup. The separate seed bodies used by the external carry-forward analysis are outside that archive. It rejected 3,387 paths outside the accepted context index's `meta_train` dataset/cut allowlist **before opening their score files**. The remaining 26,357 receipts identify adaptive development-validation RMSE, prepared data, vocabulary, evaluator, execution protocol and completion metadata. Each source file and observation ledger has a distinct recorded digest. Their recorded completion times span September 12, 2026 18:37:13 UTC to September 14, 2026 13:37:43 UTC. These times are receipt metadata, not a new replay of the fits.

For a readable comparison, the table restricts each repeated comparison to exactly 1,000 completed evaluations, matching its recorded dataset/family, cut, prepared SHA, vocabulary SHA, evaluator SHA/protocol, execution-order protocol and run. It compares the first retained generation with the last retained generation within that group. Lower RMSE is better. Improvement and worsening use exact `<` and `>` comparisons of the parsed reported scores; unchanged means exact equality, with no epsilon. The analysis script applies no additional cumulative minimum, but the underlying search already carries incumbent/teacher programs forward. Thus the reported per-generation best behaves like an inherited incumbent, rather than a fresh independent draw. The score field is `best_validation_rmse` under the historical `featmath_training_fold_normalized_v1` evaluator; the plots do not claim original target units.

| Historical search arm | Repeated arm groups | Last lower | Same | Last higher | Median relative reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Neural | 51 | 19 | 32 | 0 | 0% |
| Generic | 54 | 22 | 32 | 0 | 0% |
| Shuffled | 54 | 23 | 31 | 0 | 0% |

![First-to-last carried-forward best scores at 1,000 evaluations; the arms share seeds. The 64 lower arm groups cover 23 distinct dataset/cuts.](../figures/target-quality-progress-counts.png)

The 159 arm groups cover **55 distinct dataset/cut pairs**. The 64 lower arm groups (19 + 22 + 23) cover only **23 distinct pairs**; 22 of those 23 end at the same score in every available arm. These counts were independently recomputed from the published comparison CSV. The denominators and shared seed pools make the arm counts dependent; one discovery can later appear in multiple arms.

The separately released external seed audit reports `seed_population_supplied=true` and `teacher_first_v1` for all 26,357 receipts. The previous recorded best appears in the next seed set in 25,790 of 25,985 transitions; 26,051 reported winners (98.84%) are already in that search's seed population. Incumbent, retained-teacher and exploration seed sets are identical across arms in all 8,785 multi-arm dataset/cut/generation groups checked. Those seed-body counts are externally reported and metadata-checked here, not a new local seed-payload replay. They explain why zero first-to-last worsening is not independent evidence of stability. A lower inherited score is also not a new discovery by the receiving arm. `carry-forward-audit.json` preserves these evidence boundaries.

This table is therefore not a paired policy-ranking result. Random seeds change between generations, elapsed time varies, and this audit did not requalify the actual CPU/memory/hardware enforcement or complete build closure of the historical searches. Matching these recorded fields is narrower than the complete current experiment identity. The recorded generation manifests declare 600 seconds and 1,000 evaluations for the illustrative example; those declarations do not establish equal consumed wall time. Checkpoint association is available by generation, but exact exposure of each improved target to later optimizer batches remains unjoined.

A concrete example is `mendeley-6bspjkt7bg-pine-chromium`, dataset `b68222ad60398ea8`, cut 1. Across generations 0–70 the recorded best development RMSE changes from **0.009749860799638275 to 0.003185684481133936**, a **67.3258% relative reduction**. All three arms finish at the same score. The neural arm first records that final value at generation 43; generic and shuffled do so at generation 42. The external seed audit identifies the final program outside the generic and shuffled seed sets at generation 42, then as an `incumbent_cut_teacher` seed in all three arms at generation 43. The neural arm's later equality is carry-forward, not evidence that it independently discovered that program. This identifies generation-level observations; it does not establish which of two same-generation arms discovered the program first in wall-clock time. The plot data include all 213 generation/arm pairs, with no interpolated missing generation. This example was selected after inspection as the largest first-to-last relative improvement in the neural 1,000-evaluation subset; it is illustrative, not representative.

![Post-hoc pine-chromium cut 1 history with shared teachers: generic/shuffled record the final program outside their seeds at generation 42; all arms inherit it at generation 43.](../figures/target-quality-pine-example.png)

Relative improvement is `(first_RMSE - last_RMSE) / first_RMSE`. The percentage is meaningful within that dataset/cut. Raw RMSEs from unlike target scales are not averaged to claim overall progress. A running-best curve, if separately shown, must be labelled as such: its monotonicity follows partly from selection. Better retained minima do not prove that a trained neural proposal policy caused the gain. Shared incumbent/teacher seed pools and carry-forward are established in the external audit; search history and ordinary exploration also prevent attribution to policy learning. Count a new search discovery separately from a copied incumbent: authenticate the full seed set, identify the first measured program absent from it, and retain the discovering arm, generation, context, program and parent identities. The aggregate table above does not provide that full per-program discovery ledger.

The small publication files are `target-quality-1000-summary.csv`, `target-quality-1000-comparisons.csv` and `target-quality-pine-chromium-cut1.csv`. Full 26,357-row extracted evidence stays in custody, with its hash in `historical-target-quality-summary.json`; the manual need not embed the 19.5 MB full extraction. This analysis reads development scores only and creates no new teacher eligibility, promotion or benchmark-valid result.

### Paired final-search outcomes within the admitted development inventory

A later metadata-only comparison pairs arms **within the same generation**, keeping the recorded run, dataset/family, cut, prepared SHA, vocabulary SHA, evaluator SHA/protocol, execution-order protocol and exactly 1,000 completed evaluations equal. It uses the existing 26,357 admitted receipts, not a newly expanded input boundary. Scores are finite positive `best_validation_rmse`; lower is better. Each comparison counts exact score equality, with no epsilon.

| Neural compared with | Paired observations | Neural lower | Equal | Neural higher | Geometric mean neural/control RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Generic | 3,268 | 24 | 3,227 | 17 | 1.0001869094 |
| Shuffled | 3,253 | 24 | 3,213 | 16 | 1.0000854976 |

These are repeated adaptive **final-search** observations across generations 0–70, not measurements of initial neural seed quality. The generic comparison covers 52 dataset/cut pairs from 30 families; the shuffled comparison covers 52 pairs from 31 families. Reported wall times differ in every pair. Families, contexts and generations repeat; the arms share incumbent, retained-teacher and exploration seeds, and the search can recover from poor initial proposals. The paired rows are not independent arm discoveries. No independent-sample confidence interval, causal advantage or failure-inclusive benchmark claim follows. The ratios are descriptive and close to one; they do not establish that training improved routing.

The external report's broader routing extract has 42,863 rows. Only 3,530 match this chapter's admitted receipt inventory by retained path and all checked score, prepared-data, vocabulary, evaluator, protocol, completed-evaluation and wall-time fields. The other 39,333 are outside that existing inventory; their eligibility remains unqualified here, **not classified as all sealed**. No original score path was reopened for this join. The broader campaign ratios and p-values are therefore not imported. The compact `packing-routing-audit.json` pins inputs, matching fields, arithmetic and exclusions.

To establish a closed improvement loop rather than these separate observations, join: original search/fit completion → exact program and comparison identity → admission event and immutable target record → corpus membership → actual sampler exposure and optimizer continuation → later policy proposal/fit under the same comparison protocol. The historical logs and chapter 14 show training activity; the target receipts here show some improving development scores. That complete causal/eligibility join is still missing.

## 15.5 The exact packed record

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

This is binary packing, not general-purpose compression. Every record reserves 128 token slots and 128 legal-state slots. A short program still occupies 532 bytes on disk. Context vectors are shared by context identity. In the accepted sharded format, each group stores its own legal-set table in `shard.json` and its index entry; legal IDs are local to that table. Loading remaps them into one global table (63 distinct sets in the external accepted-corpus audit), shared in host memory and on the GPU. The older monolithic `teaching_corpus.v1` format instead has a top-level legal table. Do not interpret a raw shard ID as a global ID. `groups.bin` stores two little-endian u64 values per group—start record and count—so a group entry is 16 bytes. The JSON index binds context, vocabulary, record hashes and offsets; each immutable shard also has a manifest and an audit-label sidecar.

Original floating-point measurements and exact program provenance are retained separately from the float32 training targets. Historical `reference_v1` orders original f64 RMSEs, uses a median-scaled score value, and assigns ordinal percentiles. The stored value is `1 / (1 + RMSE / max(upper_middle_RMSE, 1e-8))`; the percentile is `1 - i/(n-1)` in the ordered group. Here `upper_middle_RMSE` is the sorted original RMSE at zero-based index `floor(n/2)`, including for an even-sized group; it is not an average of the two middle scores. Its tie behavior is not the later midrank policy. `frozen_z_v1` preserves the first admitted value/percentile for an existing program while new programs receive their initial labels. `strict_midrank_v4` introduces exact-original-precision tie handling and the strict admission contract. These names are different target semantics, not interchangeable storage optimizations. Repacking cannot silently relabel historical targets or snap unsupported program parameters to convenient vocabulary tokens.

The training reader's verified path copies and hashes the actual record bytes into sealed shared storage, or obtains a verified shared sealed file, then uses a read-only memory mapping. Linux file seals prevent later writes, growth and truncation. This protects a running trainer from source-file mutation and mmap truncation faults. It is **not** a promise of zero-copy disk startup: verification and the initial sealed-memory copy have costs. Index, record and context identities remain content-based; a filesystem timestamp is not a substitute for them.

### What the accepted-corpus packing audit measured

The external verifier reports scanning all 10,214,332 records in 7,877 accepted-corpus shards, checking recorded payload/sidecar hashes and layout invariants. Its corpus identity is the same `1c840597…` bound above. This manual independently verifies its selected metadata-file hashes and recomputes the following arithmetic; it does not claim a second scan of the original record or label bodies.

| Quantity | Result | Evidence boundary |
| --- | ---: | --- |
| Occupied token slots | 205,657,518 of 1,307,434,496 | Sum of the external length histogram |
| Mean / maximum sequence length | 20.1342 / 54 tokens | Accepted snapshot only; not a future grammar bound |
| Padding fraction | 84.2701% | `1 - occupied_slots / (records × 128)` |
| Pretty-printed `legal_sets` values | 37,691,003 bytes, 62.2074% of the 60,589,288-byte index | Raw stored JSON spans; compact serialization is a different 8,335,358 bytes |
| Original record arrays | 5,434,024,624 bytes | 10,214,332 × 532 |
| Proposed 180-byte record arrays | 1,838,579,760 bytes | Layout calculation; excludes indexes, contexts and runtime overhead |
| Proposed binary index | 1,906,735 bytes | Estimated table components; no qualified reader or writer |

The verifier also reports an actual external `zstd -3 -T1` size experiment: original records compressed separately by group occupy **292,918,135 bytes**, audit-label group frames **331,375,483 bytes**, and one compressed index **5,371,779 bytes**. These named components total **629,665,397 bytes**, versus **8,721,903,373 bytes** uncompressed. This subtotal excludes shard manifests, `groups.bin`, contexts, vocabulary, checkpoint and other provenance; it is not a complete deployable-store size or a host/GPU memory measurement. Experimental 180-byte group records compressed to 259,362,502 bytes in that external experiment. No loader-throughput, optimizer or fitted-search benefit was measured by this metadata audit.

The proposed 180-byte layout is `56 × u16` tokens + `56 × u8` local legal-set IDs + three float32 labels: 112 + 56 + 12 bytes. It would reconstruct length and context from the token prefix and authenticated group metadata. The observed maximum length rounds to 56, and the current loss path already trims each batch to a multiple of eight. Those facts motivate the proposal; they **do not prove bit-identical training**. A production codec must preserve the existing 128-token grammar limit through a versioned extension or verified fallback layout; the observed 54-token maximum does not authorize a smaller grammar cap or truncation. It must reject inconsistent widths and legal-ID overflow, preserve group/sampler order, RNG sequence and all identity/provenance, reconstruct identical legal masks and every optimizer input, and pass record round trips plus fixed-batch token/mask/context/label, loss, gradient, update and reload/continuation parity. Changing physical bytes also changes the current raw-index corpus digest; compatibility needs an explicit verified migration, not an inherited old hash.

Keep the numerical policy separate from byte layout. The external all-pairs calculation reports that float16 creates new value ties in **0.377392%** of all within-group record pairs and **0.289905%** of teacher-versus-other pairs. These have different denominators from adjacent-score pairs. Quantizing labels therefore changes ranking supervision. Preserve float32 training labels and contexts, and original-precision audit observations; compression of their bytes does not require reducing their precision.

## 15.6 Where reuse helps, and where full work remains

The on-disk teaching store can reuse unchanged verified shards **within an unchanged store binding**. In reviewed source 68, a group's key includes binding, context, programs and lineage. The binding includes twelve semantic source files and the SHA256 of `/proc/self/exe`. A changed packer executable digest therefore rejects the old store binding and changes every group key, requiring a new store/full disk repack even if the logical records are unchanged. A byte-identical reproducible rebuild does not change that digest. This disk invalidation is separate from GPU cache invalidation. New ledger contributions mark affected groups dirty; changed or reordered historical source inputs require reconciliation or a new store rather than silently editing history. Publication uses immutable artifacts and atomic replacement of the published index. Tokenization caches are bounded and exact; invalid exact parameters remain rejected.

Snapshot 97's historical `packing-timing.json` records **7,859 reused groups and 18 rebuilt groups**, from 51 new ledgers. Attribute these figures to its legacy writer: the retained receipt has the four-field legacy format, whereas reviewed source 68 writes seven fields and has the stronger executable/semantic/lineage binding. The old reuse observation is not a qualification of the current writer. That is 99.7715% reuse by group count, not a measured 99.7715% reduction in bytes, CPU or latency. The September 11 snapshot 421 audit recorded 7,863 reused and 14 rebuilt groups. Neither receipt by itself measures end-to-end pack time or memory. The accepted snapshot's 7,877 record files all exist at the declared lengths in this metadata audit, totaling 5,434,024,624 bytes; their label sidecars add 3,227,289,461 bytes. Sidecars preserve auditability but are not free storage.

The native trainer implementation maintains two different device caches. Its current direct dispatch remains held behind the original-GPU-grant requirement described in chapter 16; the following source behavior is not permission to launch it:

| Cache | Reuse identity | Behavior when identity changes |
| --- | --- | --- |
| Context vectors | Context-index SHA | Release prior tensor; read/hash every indexed 2,048-float context and stage the new table |
| Token/legal/context-ID tensors | Whole corpus SHA | Release prior tensors; traverse **every** target record and rebuild the full tables |

Tokens and legal IDs are each stored on the device as 16-bit signed tensors with all unsigned bits preserved; after gathering a minibatch, the code converts IDs to int64 and masks with `0xffff`. Context IDs are int64. The legal mask table is shared and boolean. For each minibatch, the sequence length is trimmed to the longest selected record, rounded up to a multiple of eight and capped at 128. Thus the implementation already avoids recomputing grammar masks per optimizer step and reduces unnecessary sequence work within a batch.

For 10,214,332 records, token-plus-legal identifiers alone are `N × 128 × 4 = 5,229,737,984` bytes. Context IDs add `N × 8 = 81,714,656` bytes. The 8,894 shared float32 contexts would occupy 72,859,648 bytes. These are layout calculations, not measured GPU peaks; they exclude the legal mask table, model, optimizer, activations, batch conversions, allocator overhead and host staging. The emitted `resident_identifier_bytes` counter covers only the token/legal identifier term, not complete device use.

A newly published corpus SHA invalidates the entire token/legal/context-ID cache even when most disk groups were reused under an unchanged binding. The historical 7,859-of-7,877 receipt illustrates disk reuse under the legacy writer; a changed current packer executable additionally forces full disk repacking as described above. There is no qualified incremental device-shard cache in this path. Disk-level incremental packing therefore does not guarantee low-latency continual training. Repeated trainer processes also cannot rely on process-local tensor residency. This is a concrete optimization opportunity to measure before changing the format.

## 15.7 Python training and the pending bounded campaign

The current Python `train_v3.py` path accepts JSONL `TrainingRow` objects or an in-memory list. It materializes the input rows, partitions them before grammar/reward preparation, and precomputes token IDs and legal masks for train/validation rows once before the epoch loop. It also supports a vocabulary-bound persistent row-artifact cache. It is inaccurate to say that Python reparses or retokenizes every example on every optimizer step.

However, the pending `training64` campaign's `admitted_rows` path still reads JSONL into memory and writes a per-attempt `corpus.jsonl`, then invokes `train_policy_v3_from_jsonl`. That stopped source packet does not add a reader for the Rust `teaching_shards.v1` packed store. Its stricter context/task/admission checks must be integrated and qualified before any real category training. The existence of an efficient historical Rust corpus does not prove that the pending Python route can ingest ten million examples within the current twenty-minute category window or 64 GiB host limit. No such performance run was authorized or performed for this chapter.

## 15.8 Concrete improvements, with acceptance criteria

The useful next change is evidence-led rather than another format by name. These are proposals, not implemented delivery claims.

1. **Measure the actual preparation path.** Retain separate wall time and peak host/device memory for metadata parsing, admission validation, tokenization, hashing/sealing, shard loading, host construction and device transfer. Measure cold and reused cases on the same admitted corpus identity. Include controller/setup cost; do not equate one identifier-array counter with total memory or CPU.
2. **Give Python a verified packed-reader bridge if those measurements justify it.** Preserve exact tokens, legal sets, float semantics, task/class routing, contexts, splits, original observations and vocabulary/admission pins. Prove JSONL/packed equivalence on admitted fixtures, corruption and changed-input rejection, exact recipe round trips, reload and continuation. A converter that copies old `qualification: true` fields without rechecking the current teacher contract is not sufficient.
3. **Reuse immutable device shards carefully.** Key reusable material by content digest and explicit vocabulary/grammar/context identities; measure only changed shards transferred. Global legal-set remapping, context IDs, balanced sampling, frozen labels and optimizer/sampler resume must remain reproducible. Appending raw old IDs into a new global table is unsafe without that mapping contract.
4. **Evaluate padding and compression separately.** Current u16 IDs and shared contexts already save space. The accepted-corpus audit above gives a measured padding opportunity and externally measured compressed byte counts. Length bucketing or a narrower layout still requires parity and measured GPU throughput. Compressed archive size is not a loader-memory result; preserve bounded random access, decompressed and stored-byte hashes, corruption rejection and deterministic sampling before adopting it. Keep the 180-byte record and binary-index estimates labelled as unqualified proposals.
5. **Close the evidence chain before promoting the result.** Prove an eligible new target entered the corpus, was actually sampled, changed parameters, survived reload and improved later bounded executable-search behavior. Keep resource deadlines and sealed boundaries unchanged.

Relevant source contracts are `crates/jope-engine/src/corpus.rs`, `teaching_store.rs`, `sealed_storage.rs` and `training.rs`; Python uses `python/src/jope/schema.py` and `policy/train_v3.py`. Existing source tests cover byte-identical reuse, changed inputs, frozen labels, original-precision midranks, interruption and immutable verified records. This documentation audit inspected those tests but did not run them. `source-pins.json` binds the inspected source and pending overlay; `HANDOFF.json` binds the chapter and its data.
