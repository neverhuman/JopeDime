# Training corpora: packed records, reuse and loading

## The exact packed record

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

The external verifier reports scanning all 10,214,332 records in 7,877 accepted-corpus shards, checking recorded payload/sidecar hashes and layout invariants. Its corpus identity is the same `1c840597…` bound in [§](20-corpus-and-target-history.md#record-split-and-the-meta-test-closure-question). This manual independently verifies its selected metadata-file hashes and recomputes the following arithmetic; it does not claim a second scan of the original record or label bodies.

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

The proposed 180-byte layout is `56 × u16` tokens + `56 × u8` local legal-set IDs + three float32 labels: 112 + 56 + 12 bytes. It would reconstruct length and context from the token prefix and authenticated group metadata. The observed maximum length rounds to 56, and the current loss path already trims each batch to a multiple of eight. Those facts motivate the proposal; they **do not prove bit-identical training**. A production codec must preserve the existing 128-token grammar limit through a versioned extension or verified fallback layout; the observed 54-token maximum does not authorize a smaller grammar cap or truncation. It must reject inconsistent widths and legal-ID overflow, preserve group/sampler order, RNG sequence and all identity/provenance, reconstruct identical legal masks and every optimizer input, and pass record round trips plus fixed-batch token/mask/context/label, loss, gradient, update and reload/continuation parity. Changing physical bytes also changes the current raw-index corpus digest; compatibility needs an explicit verified migration, not an inherited old hash.

Keep the numerical policy separate from byte layout. The external all-pairs calculation reports that float16 creates new value ties in **0.377392%** of all within-group record pairs and **0.289905%** of teacher-versus-other pairs. These have different denominators from adjacent-score pairs. Quantizing labels therefore changes ranking supervision. Preserve float32 training labels and contexts, and original-precision audit observations; compression of their bytes does not require reducing their precision.

## Where reuse helps, and where full work remains

### On-disk shard reuse

The on-disk teaching store can reuse unchanged verified shards **within an unchanged store binding**. In reviewed source `68f56404` (merged as main `a75e543d2` with the identical tree), a group's key includes binding, context, programs and lineage. The binding includes twelve semantic source files and the SHA256 of `/proc/self/exe`. A changed packer executable digest therefore rejects the old store binding and changes every group key, requiring a new store/full disk repack even if the logical records are unchanged. A byte-identical reproducible rebuild does not change that digest. This disk invalidation is separate from GPU cache invalidation. New ledger contributions mark affected groups dirty; changed or reordered historical source inputs require reconciliation or a new store rather than silently editing history. Publication uses immutable artifacts and atomic replacement of the published index. Tokenization caches are bounded and exact; invalid exact parameters remain rejected.

Snapshot 97's historical `packing-timing.json` records **7,859 reused groups and 18 rebuilt groups**, from 51 new ledgers. Attribute these figures to its legacy writer: the retained receipt has the four-field legacy format, whereas that source writes seven fields and has the stronger executable/semantic/lineage binding. The old reuse observation is not a qualification of the current writer. That is 99.7715% reuse by group count, not a measured 99.7715% reduction in bytes, CPU or latency. The September 11 snapshot 421 audit recorded 7,863 reused and 14 rebuilt groups. Neither receipt by itself measures end-to-end pack time or memory. The accepted snapshot's 7,877 record files all exist at the declared lengths in this metadata audit, totaling 5,434,024,624 bytes; their label sidecars add 3,227,289,461 bytes. Sidecars preserve auditability but are not free storage.

### The two device caches in the native trainer

The native trainer implementation maintains two different device caches. Its current direct dispatch remains held behind the original-GPU-grant requirement described in [§](12-training-the-policy.md#the-dispatch-hold-resource-incompatibility-and-valid-topologies); the following source behavior is not permission to launch it:

| Cache | Reuse identity | Behavior when identity changes |
| --- | --- | --- |
| Context vectors | Context-index SHA | Release prior tensor; read/hash every indexed 2,048-float context and stage the new table |
| Token/legal/context-ID tensors | Whole corpus SHA | Release prior tensors; traverse **every** target record and rebuild the full tables |

Tokens and legal IDs are each stored on the device as 16-bit signed tensors with all unsigned bits preserved; after gathering a minibatch, the code converts IDs to int64 and masks with `0xffff`. Context IDs are int64. The legal mask table is shared and boolean. For each minibatch, the sequence length is trimmed to the longest selected record, rounded up to a multiple of eight and capped at 128. Thus the implementation already avoids recomputing grammar masks per optimizer step and reduces unnecessary sequence work within a batch.

### Identifier memory and whole-corpus cache invalidation

For 10,214,332 records, token-plus-legal identifiers alone are `N × 128 × 4 = 5,229,737,984` bytes. Context IDs add `N × 8 = 81,714,656` bytes. The 8,894 shared float32 contexts would occupy 72,859,648 bytes. These are layout calculations, not measured GPU peaks; they exclude the legal mask table, model, optimizer, activations, batch conversions, allocator overhead and host staging. The emitted `resident_identifier_bytes` counter covers only the token/legal identifier term, not complete device use.

A newly published corpus SHA invalidates the entire token/legal/context-ID cache even when most disk groups were reused under an unchanged binding. The historical 7,859-of-7,877 receipt illustrates disk reuse under the legacy writer; a changed current packer executable additionally forces full disk repacking as described above. There is no qualified incremental device-shard cache in this path. Disk-level incremental packing therefore does not guarantee low-latency continual training. Repeated trainer processes also cannot rely on process-local tensor residency. This is a concrete optimization opportunity to measure before changing the format.

## Python training and the pending bounded campaign

The current Python `train_v3.py` path accepts JSONL `TrainingRow` objects or an in-memory list. It materializes the input rows, partitions them before grammar/reward preparation, and caches each train/validation row's token IDs and sparse per-position legal-ID lists once before the epoch loop. It still rebuilds a dense boolean legal mask of shape `[batch, sequence − 1, vocabulary size]` on the host for every batch before moving it to the device. It also supports a vocabulary-bound persistent row-artifact cache. It is inaccurate to say that Python reparses or retokenizes every example on every optimizer step.

However, the pending `training64` campaign's `admitted_rows` path still reads JSONL into memory and writes a per-attempt `corpus.jsonl`, then invokes `train_policy_v3_from_jsonl`. That stopped source packet does not add a reader for the Rust `teaching_shards.v1` packed store. Its stricter context/task/admission checks must be integrated and qualified before any real category training. The existence of an efficient historical Rust corpus does not prove that the pending Python route can ingest ten million examples within the current twenty-minute category window or 64 GiB host limit. No such performance run was authorized or performed for this chapter.

## Concrete improvements, with acceptance criteria

The useful next change is evidence-led rather than another format by name. These are proposals, not implemented delivery claims.

1. **Measure the actual preparation path.** Retain separate wall time and peak host/device memory for metadata parsing, admission validation, tokenization, hashing/sealing, shard loading, host construction and device transfer. Measure cold and reused cases on the same admitted corpus identity. Include controller/setup cost; do not equate one identifier-array counter with total memory or CPU.
2. **Give Python a verified packed-reader bridge if those measurements justify it.** Preserve exact tokens, legal sets, float semantics, task/class routing, contexts, splits, original observations and vocabulary/admission pins. Prove JSONL/packed equivalence on admitted fixtures, corruption and changed-input rejection, exact recipe round trips, reload and continuation. A converter that copies old `qualification: true` fields without rechecking the current teacher contract is not sufficient.
3. **Reuse immutable device shards carefully.** Key reusable material by content digest and explicit vocabulary/grammar/context identities; measure only changed shards transferred. Global legal-set remapping, context IDs, balanced sampling, frozen labels and optimizer/sampler resume must remain reproducible. Appending raw old IDs into a new global table is unsafe without that mapping contract.
4. **Evaluate padding and compression separately.** Current u16 IDs and shared contexts already save space. The accepted-corpus audit ([§](#what-the-accepted-corpus-packing-audit-measured)) gives a measured padding opportunity and externally measured compressed byte counts. Length bucketing or a narrower layout still requires parity and measured GPU throughput. Compressed archive size is not a loader-memory result; preserve bounded random access, decompressed and stored-byte hashes, corruption rejection and deterministic sampling before adopting it. Keep the 180-byte record and binary-index estimates labelled as unqualified proposals.
5. **Close the evidence chain before promoting the result.** Prove an eligible new target entered the corpus, was actually sampled, changed parameters, survived reload and improved later bounded executable-search behavior. Keep resource deadlines and sealed boundaries unchanged.

Relevant source contracts are `crates/jope-engine/src/corpus.rs`, `teaching_store.rs`, `sealed_storage.rs` and `training.rs`; Python uses `python/src/jope/schema.py` and `policy/train_v3.py`. Existing source tests cover byte-identical reuse, changed inputs, frozen labels, original-precision midranks, interruption and immutable verified records. This documentation audit inspected those tests but did not run them. `source-pins.json` binds the inspected source and pending overlay; `HANDOFF.json` binds the chapter and its data.
