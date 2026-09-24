# Teacher admission, HWM identities and corpus updates

## A dataset is not one training example

### Accounting units

A training context describes a particular dataset, split and encoding. Many executable programs can share that context. Each program supplies a separate target sequence and measured supervision. The useful accounting units are therefore:

| Unit | Meaning | Why the distinction matters |
| --- | --- | --- |
| Dataset/family | A problem and its related datasets | Family partitions control generalization; row counts alone do not identify independence. |
| Cut/context | A dataset split plus prepared-data and encoder identities | The same dataset can have many contexts. Reusing a context must preserve its actual split and fitting rows. |
| Program observation | A measured evaluation of a concrete executable recipe | Failed, duplicated, incompatible or exposed observations need not become training examples. |
| Admitted target record | A representable program and its supervision for one context. Under the pending contract, a **teacher** is such a record that passed the `jope.teaching-guard/v1` admission ([§](#the-admission-guard-label-exposure-and-hwm-identity)) and entered a frozen corpus | Repeated observations of the same program/cut do not automatically earn repeated votes. |
| Corpus snapshot | An immutable collection of target records and identities | Its record count is not the number of optimizer updates or unique datasets. |
| Presentation | A sampled training example used in an update | Sampling revisits records. Millions of presentations do not mean millions of newly discovered targets. |

### Context vectors and program targets

The legacy Rust context is 2,048 little-endian float32 values, or 8,192 bytes. The model receives a 1,536-value dataset lane and a separate 512-value target-statistics lane. The historical encoder recipe described the dataset lane as Hyperion 512 plus Foundation/PCA 1,024. Despite its name, the dataset lane is not label-free: in the current context producer the 1,024 Foundation values are computed from the features together with the standardized training targets, and only the 512 Hyperion values use the features alone ([§](architecture.md#component-dimensions-and-assembly)). These vectors are shared by context identity; they are not repeated inside every 532-byte target record. The program target is a grammar-token sequence, with the legal-token set for each step, context ID, score-derived value, percentile and log-runtime supervision.

A concrete metadata binding in the accepted step-60000 corpus connects dataset `031201c6895f724b`, family `rdatasets-agridat-kalamkar-potato-uniformity`, cut 0 and `meta_train` to a prepared view and context vector. Its context SHA begins `6f5fdd79…`, prepared-view SHA `811feabd…`, and content-addressed shard contains 24,231 records. The full paths and hashes are in `accepted-corpus-binding.json`. This audit verified that metadata relation without opening the encoded vector or target records.

### The learning chain

The learning chain is:

1. Pin the raw dataset, family partition and actual fitting/development row assignments.
2. Fit preprocessing and encoding only within the permitted input closure; pin their bytes and versions. The closure covers records, contexts, transforms and every asset, and a forbidden descriptor must be rejected before loading even when no admitted row uses it. At the retained PR83/stopped-overlay snapshot, that rejection is **proposed**: neither source rejects extra context-index entries, and the legacy trainer at `a75e543d2` stages every indexed context after only a digest and size check ([§](data.md#hashes-eligibility-and-the-complete-input-closure)).
3. Generate a concrete executable program, fit it and retain its outcome, measurement, resource use and evaluator identity.
4. Apply the declared admission policy; reject failed, incompatible, exposed or unsupported targets.
5. Tokenize the exact program and retain original observations beside the training representation.
6. Sample the admitted record, update parameters, and prove checkpoint reload/continuation.
7. Evaluate the new policy through actual proposals and fits under the comparison protocol.

A file at step 5 does not prove steps 6–7. Conversely, a lower optimizer loss does not establish better executable proposals.

The chain yields four distinct claims, and the owner requires them to stay separate (relayed 2026-09-23T22:27:11Z). Each needs its own evidence; none implies the next:

| Claim | Established by | Not established by |
| --- | --- | --- |
| Eligible teacher | A record that passed the preregistered `jope.teaching-guard/v1` admission under its complete comparison identity (step 4) | A content hash, a saved recipe, a completed run or a historical HWM entry |
| New discovery | A program absent from the authenticated initial seed population of the search that found it, with its discovering arm, generation and first measurement retained ([§](learning.md#what-to-retain-for-each-cycle)) | A carried incumbent, a re-measured seed or a duplicate of an existing record |
| Actually consumed information | Sampler and optimizer receipts showing the record sampled in committed optimizer updates (step 6); changed parameters are a separate LOOP-01 link ([§](learning.md#loop-01-the-central-regression-milestone)) | Corpus membership or a manifest row |
| Improved policy | A matched comparison of parent and challenger checkpoints with actually fitted proposals under the same protocol and budget (step 7; [§](learning.md#the-closure-ledger)) | A lower training loss, a new checkpoint file or the teacher's own development score |

For the new ASM path none of the four is established yet: zero teachers have been admitted under the new contract ([§](#present-teacher-readiness-and-the-smallest-real-path)), and the pending campaign records no initial-population or discovery identity. Reporting the four separately is a **proposed** rule.

## How a fitted recipe becomes an eligible teacher

### The preregistered freeze and train-side inner split

The pending eligibility path freezes independent facts before optimization: original dataset and prepared-data identity, ordered rows and groups, target role, exact inner-fold assignment, evaluator and vocabulary identities, numerical policy, baseline and worker/resource pins. This freeze cannot be created retrospectively from a winning candidate's own output. A historical successful fit is not equivalent to preregistered teaching evidence.

The train-side inner split uses five whole-group folds. Groups are assigned deterministically by largest size first, with lexical tie-breaking and balancing. No group's rows are split simply to fill a fold. The cut's held-out development fold remains outside the search's inner fitting/selection closure.

### The admission guard, label exposure and HWM identity

Each candidate must produce complete authenticated OOF predictions for all five inner folds. Scores are recomputed from retained predictions, exact targets and ordered class probabilities. The trivial baseline is fitted from each fold's training rows: target mean for regression, class frequencies for classification. The guard uses mean fold loss and admits only if:

- all five fold losses and baseline losses are finite;
- the mean candidate loss is below the mean baseline loss by more than `1e-12 + 1e-6 * abs(mean_baseline_loss)`;
- at least four of five folds each improve by more than `1e-12 + 1e-6 * abs(that fold's baseline_loss)`;
- actual worker requests, process outcomes, output hashes, fitted artifacts, seeds, source, resources and deadlines match the original freeze.

The criteria are intentionally stronger than “a run completed” or “the score field is positive.” Constant/no-signal predictors, missing folds, reordered rows, changed class order, stale scores, unauthenticated prediction blobs and unknown execution accounting cannot provide teaching authority. A failed admission is retained as evidence rather than discarded. A hash is not eligibility: a content digest proves that bytes are unchanged, not that the record passed this guard under the right comparison identity with permitted labels. In the pending overlay, verifying a retained admission report re-derives the qualification from the learning state (`eligibility::revalidate_qualification`) and refuses any mismatch, rather than trusting a stored `qualified` flag (pending `crates/jope-bench/src/panel/teaching.rs:730-736`).

Optimization-visible labels initially called `test` become development labels, with a durable record of that exposure. `sealed` and `meta_test` inputs are refused for optimization/teaching. The HWM identity includes the full comparison context, so an improvement under a different evaluator, split, vocabulary, preprocessing or numerical protocol cannot displace an unrelated incumbent. These details belong to the pending eligibility/HWM correction, not a claim that every historical HWM record already meets this rule.

Sources: pending `crates/jope-learning/src/eligibility.rs:170`, `:253`, `:277`, `:427`, `:700`, `:1023`; `crates/jope-learning/src/eligibility/inner.rs`; `crates/jope-learning/src/record.rs`; `crates/jope-pipeline/src/search_plan.rs`.

## From optimization to reusable knowledge

The pending strict HWM key includes task, dataset hash, cut identity, evaluator, vocabulary and protocol. Scores from different identities cannot overwrite each other or supply training feedback. A recipe must show finite train-side out-of-fold improvement over a trivial baseline and improve on four of five folds before teaching admission. No-signal data should be refused by this gate. An optimization-visible test label is development evidence even if its filename says otherwise.

Typed motifs can reuse eligible program fragments and macros while retaining provenance. The prepared motif library does not yet establish the production eligible-state reader, `motifs mine` integration or operational reuse. A new operator requires compatibility tests and an ablation under the unchanged comparison protocol. Repeatedly selecting the best development result can improve development performance without proving improved sealed generalization.

## Present teacher readiness and the smallest real path

The metadata readiness audit verified 28 source-family evidence files and the retained original process/context receipts. The inspected historical origin runs at `55e9ed688faf` and `3fd7d5b2ebb6` had no preregistered teaching freeze or vocabulary pins; they cannot be upgraded after the fact. The operational origin proof at `68f56404` (the reviewed PR83 head, merged on 2026-09-23 as main `a75e543d2` with the identical tree) also was not a teaching campaign. There is therefore no currently demonstrated eligible real corpus for the new ASM admission-and-training proof. This does not negate the historical training corpus or the optimizer work that produced the retained model.

The prospective small pair selected from frozen provenance/size, before observing losses, is:

| Dataset | Evidenced family | Cut 0 fitting rows (development fold 0 held out) |
|---|---|---:|
| `230_machine_cpu` | UCI computer-hardware relative-performance source | 135 |
| `712_chscase_geyser1` | Chatterjee casebook geyser source | 140 |

Both must independently produce eligible recipes. The minimum of ten completed inner fits is only the five-fold×two-family floor for one successful recipe per family. It is not a replacement for the specified matched search and does not promise that either family will improve.

If exactly these two families qualify, seed 7 assigns `230_machine_cpu` to the policy-validation families and `712_chscase_geyser1` to the policy-training families under the current partition rule, leaving a single training family. If more families qualify, use the entire admitted set and let the native family partition determine membership. Do not omit inconvenient eligible comparisons, force the two-family allocation, or reuse the frozen panel's (`asm-private-30-v1`) deadline for a separate trial. Historical checkpoint exposure remains a separately reported unknown; family provenance alone does not prove zero prior exposure. Keep three family-level domains distinct: teacher and training families; adaptive development families, where the frozen thirty-dataset panel remains; and evaluation families outside both training and selection. Exposure of the pretrained encoder components to any family also remains unknown.

Both prospective families are small regression datasets of the frozen panel itself (169 and 172 development rows, [§](datasets.md#per-dataset-summary)); no real binary or multiclass family is prospective. With exactly these two families, the first training campaign is a **mechanics qualification**: it can show optimizer updates, reload, continuation and consumption of a new checkpoint, but it cannot test whether training improves proposals, and its export must not be compared with step 60,000 as a flywheel challenger. That campaign trains experimental regression lineage v1 (256×2, new initialization), a width-256, two-layer model family separate from the accepted 512×6 step-60000 lineage ([§](training.md#what-the-pending-campaign-trains)). If either dataset is used to manufacture teachers, later panel evaluations on it are labelled exposed/adaptive ([§](learning.md#freezing-the-experiment-and-choosing-families)); see also OD-8 in [§](plan.md#proposed-owner-decisions-and-external-dependencies).

Source evidence: custody `agent-training-readiness/readiness-v3.json`, SHA256 `9197900d4b696811909c4ba4e779e6c5d488569dd9c360165cc55fd43bf8c735`. Its metadata inspection does not itself admit training or re-run numerical qualification.

## Prepare a teaching campaign before it starts

Runbook summary (status as of 2026-09-23 about 23:55 UTC: **pending integration**):

| Runbook part | Teaching campaign |
| --- | --- |
| Purpose and status | Preregister train-side teaching freezes, run a bounded search, and admit each arm's selected recipe into the learning state only if it passes `jope.teaching-guard/v1`. **Pending integration:** the preregistration and admission code belongs to the stopped training64 overlay, and the configuration at main `a75e543d2` has no teaching member. Zero teachers have been admitted |
| Prerequisites and resources | The final reviewed build and its receipts ([§](#final-build-and-the-teaching-configuration)); a `teaching` member whose `state_root` and `prepared_root` are absolute, whose state root lies outside `/tmp` and outside Git checkouts, whose panel file is the approved frozen manifest and whose label role maps to development; a separately declared, approved resource allocation; no exclusive CPU CI running |
| Manifest-creation command | `benchmark plan` writes the comparator plan; `optimize plan` or `optimize proof-plan` writes the ASM plan and, in the pending overlay, preregisters the `jope.inner-teaching-registry/v1` freezes before the final experiment identity. No `jope optimize` subcommand, at `a75e543d2` or in the pending overlay, writes the complete configuration or its `teaching` member |
| Execution | `optimize run` ([§](#plan-run-and-recover-a-teaching-campaign)). Admission runs inside the controller before outer-refit scheduling and consumes completed receipts only; it starts no fit and no clock |
| Artifacts and success | In the journal, a `jope.inner-teaching-admission/v1` report at `teaching-admission/<key>.json` when every selected arm's entry qualified or the registration was unavailable, otherwise a `teaching-inventory/<key>-<hash>.json` record. Success is at least one qualified record per family, re-verified from the retained evidence |
| Failure and resume | `optimize run` resumes the original journal within its window. A non-terminal admission failure is inventory that revalidation may retry within the original experiment; a failed admission is retained, never deleted |
| Limitations | The two prospective families are frozen-panel regression datasets, so the first campaign is a mechanics qualification ([§](#present-teacher-readiness-and-the-smallest-real-path)); admission considers only each arm's selected recipe; the real admission path has not run |

Sources: pending `crates/jope-bench/src/panel/teaching.rs:48-67`, `:123-125`, `:537-623`, `:754-756` and `crates/jope-bench/src/panel/optimize_cli.rs:13-53` in the training overlay replay (`agent-training-integration/refresh-68f56404/replay-001/source-text` under the ASM custody root).

### Final build and the teaching configuration

The final reviewed build must supply its CLI, worker inventory, source manifest, runtime manifests, component assets and qualification receipts. Record the actual built source SHA; do not substitute the latest checkout hash for the binary's source. Recheck that no exclusive CPU CI is running before opening an authorized CPU campaign.

In the pending overlay, the `teaching` member of the complete ASM configuration has this shape (the configuration at main `a75e543d2` has no teaching member):

```json
{
  "state_root": "/durable/private/learning-state",
  "prepared_root": "/durable/private/prepared-panel",
  "panel": {"path": "/source/benchmarks/panel.json", "sha256": "EXACT_PANEL_FILE_SHA256"},
  "original_label_role": "development"
}
```

The capitalized value is a placeholder, not a valid digest. The complete configuration also requires exact source/build/runtime/checkpoint/vocabulary/context fields; the small object above cannot be passed as the whole configuration. Planning preregisters the teaching freezes before final experiment identity, so worker evidence can bind them.

### Plan, run and recover a teaching campaign

The following shell variables must already name authenticated artifacts. Commands are interface examples and allocate no budget merely by appearing here:

```sh
"$JOPE_FINAL_CLI" benchmark plan \
  --manifest "$JOPE_PANEL" --datasets "$JOPE_PREPARED" \
  --source-sha "$JOPE_FINAL_SOURCE" --runtimes "$JOPE_RUNTIMES" \
  --output "$JOPE_CAMPAIGN/comparator-plan.json"

"$JOPE_FINAL_CLI" optimize plan \
  --base-plan "$JOPE_CAMPAIGN/comparator-plan.json" \
  --base-plan-sha256 "$JOPE_BASE_PLAN_ID" \
  --configuration "$JOPE_TEACHING_CONFIGURATION" \
  --configuration-sha256 "$JOPE_CONFIGURATION_FILE_SHA" \
  --output "$JOPE_CAMPAIGN/asm-plan.json"
```

Use the semantic plan identity emitted by native planning for `--base-plan-sha256` and `--plan-sha256`. A JSON file's byte hash and its semantic identity need not match. Configuration/resource arguments explicitly refer to their required file hashes.

For a separately admitted small proof, substitute this planning command. `proof-plan` selects one regression dataset from the verified frozen base panel, replaces the resources with the explicitly pinned proof resource manifest, and preregisters teaching only for that selected dataset:

```sh
"$JOPE_FINAL_CLI" optimize proof-plan \
  --base-plan "$JOPE_CAMPAIGN/comparator-plan.json" \
  --base-plan-sha256 "$JOPE_BASE_PLAN_ID" \
  --configuration "$JOPE_TEACHING_CONFIGURATION" \
  --configuration-sha256 "$JOPE_CONFIGURATION_FILE_SHA" \
  --dataset "$JOPE_SELECTED_DATASET" \
  --resources "$JOPE_PROOF_RESOURCES" \
  --resources-sha256 "$JOPE_PROOF_RESOURCES_SHA" \
  --output "$JOPE_CAMPAIGN/asm-plan.json"
```

A second family requires its own actual preregistered proof plan and journal, with its separately declared approved resource allocation. Both may point at the same learning state and frozen30 metadata. Neither proof changes or spends an unstarted official panel's immutable window. Proposed example durations in earlier queue documents were not resource admissions.

Run and recover using the same immutable plan and original journal:

```sh
"$JOPE_FINAL_CLI" optimize run \
  --plan "$JOPE_CAMPAIGN/asm-plan.json" --plan-sha256 "$JOPE_ASM_PLAN_ID" \
  --manifest "$JOPE_PANEL" --datasets "$JOPE_PREPARED" \
  --journal "$JOPE_CAMPAIGN/journal"

"$JOPE_FINAL_CLI" optimize report \
  --plan "$JOPE_CAMPAIGN/asm-plan.json" --plan-sha256 "$JOPE_ASM_PLAN_ID" \
  --manifest "$JOPE_PANEL" --datasets "$JOPE_PREPARED" \
  --journal "$JOPE_CAMPAIGN/journal"
```

Sources: `crates/jope-bench/src/panel/optimize_cli.rs:13` and `crates/jope-bench/src/panel/origin_proof.rs` (the `optimize plan`, `proof-plan`, `run` and `report` front door is present at main `a75e543d2`, where it makes no teaching call; the pending overlay adds the `teaching::preregister` call to `optimize plan` and `proof-plan` and the new `crates/jope-bench/src/panel/teaching.rs` that defines it); current `crates/jope-bench/src/panel/campaign_cli.rs:18`.

## Corpus updates: reuse, device caches and loading

### On-disk shard reuse

The on-disk teaching store can reuse unchanged verified shards **within an unchanged store binding**. In reviewed source `68f56404` (merged as main `a75e543d2` with the identical tree), a group's key includes binding, context, programs and lineage. The binding includes twelve semantic source files and the SHA256 of `/proc/self/exe`. A changed packer executable digest therefore rejects the old store binding and changes every group key, requiring a new store/full disk repack even if the logical records are unchanged. A byte-identical reproducible rebuild does not change that digest. This disk invalidation is separate from GPU cache invalidation. New ledger contributions mark affected groups dirty; changed or reordered historical source inputs require reconciliation or a new store rather than silently editing history. Publication uses immutable artifacts and atomic replacement of the published index. Tokenization caches are bounded and exact; invalid exact parameters remain rejected.

Snapshot 97's historical `packing-timing.json` records **7,859 reused groups and 18 rebuilt groups**, from 51 new ledgers. Attribute these figures to its legacy writer: the retained receipt has the four-field legacy format, whereas that source writes seven fields and has the stronger executable/semantic/lineage binding. The old reuse observation is not a qualification of the current writer. That is 99.7715% reuse by group count, not a measured 99.7715% reduction in bytes, CPU or latency. The September 11 snapshot 421 audit recorded 7,863 reused and 14 rebuilt groups. Neither receipt by itself measures end-to-end pack time or memory. The accepted snapshot's 7,877 record files all exist at the declared lengths in this metadata audit, totaling 5,434,024,624 bytes; their label sidecars add 3,227,289,461 bytes. Sidecars preserve auditability but are not free storage.

### The two device caches in the native trainer

The native trainer implementation maintains two different device caches. Its current direct dispatch remains held behind the original-GPU-grant requirement described in [§](training.md#the-dispatch-hold-resource-incompatibility-and-valid-topologies); the following source behavior is not permission to launch it:

| Cache | Reuse identity | Behavior when identity changes |
| --- | --- | --- |
| Context vectors | Context-index SHA | Release prior tensor; read/hash every indexed 2,048-float context and stage the new table |
| Token/legal/context-ID tensors | Whole corpus SHA | Release prior tensors; traverse **every** target record and rebuild the full tables |

Tokens and legal IDs are each stored on the device as 16-bit signed tensors with all unsigned bits preserved; after gathering a minibatch, the code converts IDs to int64 and masks with `0xffff`. Context IDs are int64. The legal mask table is shared and boolean. For each minibatch, the sequence length is trimmed to the longest selected record, rounded up to a multiple of eight and capped at 128. Thus the implementation already avoids recomputing grammar masks per optimizer step and reduces unnecessary sequence work within a batch.

### Identifier memory and whole-corpus cache invalidation

For 10,214,332 records, token-plus-legal identifiers alone are `N × 128 × 4 = 5,229,737,984` bytes. Context IDs add `N × 8 = 81,714,656` bytes. The 8,894 shared float32 contexts would occupy 72,859,648 bytes. These are layout calculations, not measured GPU peaks; they exclude the legal mask table, model, optimizer, activations, batch conversions, allocator overhead and host staging. The emitted `resident_identifier_bytes` counter covers only the token/legal identifier term, not complete device use.

A newly published corpus SHA invalidates the entire token/legal/context-ID cache even when most disk groups were reused under an unchanged binding. The historical 7,859-of-7,877 receipt illustrates disk reuse under the legacy writer; a changed current packer executable additionally forces full disk repacking as described above. There is no qualified incremental device-shard cache in this path. Disk-level incremental packing therefore does not guarantee low-latency continual training. Repeated trainer processes also cannot rely on process-local tensor residency. This is a concrete optimization opportunity to measure before changing the format.

### Python training input and the pending bounded campaign

The current Python `train_v3.py` path accepts JSONL `TrainingRow` objects or an in-memory list. It materializes the input rows, partitions them before grammar/reward preparation, and caches each train/validation row's token IDs and sparse per-position legal-ID lists once before the epoch loop. It still rebuilds a dense boolean legal mask of shape `[batch, sequence − 1, vocabulary size]` on the host for every batch before moving it to the device. It also supports a vocabulary-bound persistent row-artifact cache. It is inaccurate to say that Python reparses or retokenizes every example on every optimizer step.

However, the pending `training64` campaign's `admitted_rows` path still reads JSONL into memory and writes a per-attempt `corpus.jsonl`, then invokes `train_policy_v3_from_jsonl`. That stopped source packet does not add a reader for the Rust `teaching_shards.v1` packed store. Its stricter context/task/admission checks must be integrated and qualified before any real category training. The existence of an efficient historical Rust corpus does not prove that the pending Python route can ingest ten million examples within the current twenty-minute category window or 64 GiB host limit. No such performance run was authorized or performed for this chapter.

## Concrete corpus improvements, with acceptance criteria

The useful next change is evidence-led rather than another format by name. These are proposals (status: **proposed**), not implemented delivery claims. The owner's order (relayed 2026-09-23T22:27:11Z) is: profile the phases; then a verified Python packed-reader bridge; then immutable shard reuse with bounded staging; then compression; and a narrower record layout only if measurements justify it. Every step preserves float32 supervision, the legal-set mappings and the 128-token grammar limit, and proves record, batch, loss, gradient, update and resume parity ([§](data-estate.md#compression-the-proposed-180-byte-layout-and-label-precision)).

1. **Measure the actual preparation path.** Retain separate wall time and peak host/device memory for metadata parsing, admission validation, tokenization, hashing/sealing, shard loading, host construction and device transfer. Measure cold and reused cases on the same admitted corpus identity. Include controller/setup cost; do not equate one identifier-array counter with total memory or CPU.
2. **Give Python a verified packed-reader bridge if those measurements justify it.** Preserve exact tokens, legal sets, float semantics, task/class routing, contexts, splits, original observations and vocabulary/admission pins. Prove JSONL/packed equivalence on admitted fixtures, corruption and changed-input rejection, exact recipe round trips, reload and continuation. A converter that copies old `qualification: true` fields without rechecking the current teacher contract is not sufficient.
3. **Then reuse immutable shards with bounded staging.** Key reusable material by content digest and explicit vocabulary/grammar/context identities; measure only changed shards transferred, and stage them through a bounded, measured host buffer rather than rebuilding the whole table. Global legal-set remapping, context IDs, balanced sampling, frozen labels and optimizer/sampler resume must remain reproducible. Appending raw old IDs into a new global table is unsafe without that mapping contract.
4. **Then evaluate compression.** Current u16 IDs and shared contexts already save space. The accepted-corpus audit ([§](data-estate.md#what-the-accepted-corpus-packing-audit-measured)) gives externally measured compressed byte counts. Compressed archive size is not a loader-memory result; preserve bounded random access, decompressed and stored-byte hashes, corruption rejection and deterministic sampling before adopting it.
5. **Adopt a narrower layout only if measured.** The same audit gives a measured padding opportunity. Length bucketing or a narrower layout still requires parity and measured GPU throughput. Keep the 180-byte record and binary-index estimates labelled as unqualified proposals.
6. **Close the evidence chain before promoting the result.** Prove an eligible new target entered the corpus, was actually sampled, changed parameters, survived reload and improved later bounded executable-search behavior. Keep resource deadlines and sealed boundaries unchanged.

Keep exactly one learning-store writer until an atomic server-side pointer update is qualified; this manual records no such qualification. The on-disk packer already takes an exclusive, non-blocking `writer.lock` on its store directory and refuses a second concurrent packer (`crates/jope-engine/src/teaching_store.rs:377-385` at `a75e543d2`), but that local lock does not coordinate writers of a remote learning store. Backup, replication and retention operations on learning storage remain separately authorized; a corpus update must not start them.

Relevant source contracts are `crates/jope-engine/src/corpus.rs`, `teaching_store.rs`, `sealed_storage.rs` and `training.rs`; Python uses `python/src/jope/schema.py` and `policy/train_v3.py`. Existing source tests cover byte-identical reuse, changed inputs, frozen labels, original-precision midranks, interruption and immutable verified records. This documentation audit inspected those tests but did not run them. `source-pins.json` binds the inspected source and pending overlay; `HANDOFF.json` binds the chapter and its data.
