# Encodings and the proposal context

## Regression context production

### Original data and preprocessing

The context input binds prepared-data hash, exact training-content hash, ordered fitting-row hash, cut identity and preprocessing hash. It requires at least eight training rows and a nonempty declared feature schema. Numeric versus categorical kinds come from the schema rather than guesses based on strings. Numeric nulls are imputed using training-only medians, with a zero fallback for an entirely missing column; the recorded training transform contains its fitted location/scale. For categories, every present string is a value, including empty text or strings spelling `NA` or `null`. Only an absent value is missing. Original target strings remain separately authenticated before numerical regression parsing.

This context preprocessing is for the proposal encoder. It does not authorize discarding the original schema or replacing a fitted predictor's own training-only preprocessing. Source: `crates/jope-policy/src/context_producer/input.rs:84`.

### Component dimensions and assembly

The current CPU producer uses three components with fixed pretrained assets and fixed transforms:

| Component | Production | Raw output | Contribution to policy context |
|---|---|---:|---:|
| Hyperion | Training feature columns only (no label), quantized from fitted quantile bounds; three embedding sweeps, scales 4/7/11/15, batch 1024, internal setting `target_chunks_per_sweep=1` (a chunking setting, not the label) | 128 values per row | Mean, population standard deviation, 10th and 90th percentile per coordinate: 512 |
| Foundation | Feature columns plus the training targets as in-context labels, prepared with the native deterministic recipe; target normalization computed in f64 after the native f32 cast | 2048 | Frozen mean-centering, learned 2048→1024 projection, then frozen normalization: 1024 |
| Target | Training targets, already standardized in preparation, then min–max rescaled to [0,1] by the training minimum and maximum (span 1 when constant); two frozen native target windows with seed 17 | 512 | Frozen normalization: 512 |

Assembly concatenates normalized Hyperion512 + projected Foundation1024 + Target512 and clips the full 2048-vector to [-32,32]. Only the Hyperion block is label-free: Foundation is conditioned on the training targets, so the first 1,536 values, which the policy configuration still calls `dataset_dim`, include target-conditioned values. It checks all component dimensions and finite values. The source explicitly checks that the Hyperion row width is 128; confusing row width with pooled512 changes the model input meaning. Sources: `crates/jope-policy/src/context_producer.rs:188`, `:278`, `:356`, `:399`, `:422`.

The deterministic foundation preparation selects at most 768 context rows and 256 query rows before `encode_standardized`; that encoder adds no further cap. Hyperion receives every supplied training row and feature column, with no row/column subsampling in this producer. The serialized input is capped at 128 MiB and every context job remains subject to the fixed 30-second worker deadline. On the eight final reference jobs at source `68f56404`, Hyperion-stage elapsed time ranged from 6.143 to 14.287 seconds; whole-job wall time ranged from 11.731 to 20.755 seconds. These small-case measurements do not establish coverage for wider or larger cuts. A timeout leaves the context unavailable; it does not permit a new deadline or silent subsampling.

The frozen numerical qualification contract is unchanged, but the implementation includes both execution and numerical corrections. Hyperion uses an embeddings-only sweep that skips the unused confidence head. Foundation uses the separate `encode_standardized` path: train-side feature moments accumulate in f64 before the retained f32 transform, and p10/p90 use Torch 2.7 FP32 rank and stable-lerp arithmetic. Ordinary `encode` retains its legacy feature-moment and percentile arithmetic. The standardized route consumes all rows supplied by the preceding deterministic foundation preparation without additional caps. Cached-decoder fixtures separately check small random-model operations/full-prefix agreement within 1e-4, row/padding behavior and sampling-token/RNG agreement. They do not establish that the Candle CPU policy port matches the trained tch forward pass: the source states that logits and rewards from the port are unverified until reference fixtures exist (`crates/jope-policy/src/lib.rs:84-92`). Native CUDA policy parity is also not established; the `cuda` feature has never been compiled (`lib.rs:93-95`). Sources: `crates/jope-policy/src/context_producer.rs:238`; `crates/jope-encoders/src/foundation.rs:345`; `crates/jope-encoders/src/foundation/support.rs:123`; `crates/jope-encoders/src/model/tensor_ops.rs:114`; `crates/jope-policy/src/model/cached.rs:270`; `crates/jope-policy/src/decode.rs:485`.

### Qualification and admission

The CPU regression encoder has a stronger boundary than “the vector has 2048 values.” Eight fixed native-reference cases span two families and cuts 0–3. Qualification checks finite FP32 component and final-context values with maximum absolute difference at most 0.0003, exact asset/source/program identities, and actual bounded executions. Consumption binds those eight jobs plus the actual dataset context job: nine independently authenticated process proofs. The reference registry/index and transform hashes are pinned in source.

The eight retained reference outputs at source `68f56404` have final-context maximum absolute differences from 5.3644180e-5 to 2.8765202e-4. The worst case is `cardata-blackmore`, dataset `55e9dfc7a86361d7`, cut 0: 95.884% of the unchanged 3e-4 tolerance. Across all eight cases, maximum component differences are 4.127622e-6 for Hyperion, 2.908707e-5 for foundation and 3.911555e-7 for target. The frozen projection and projected-scale division can amplify small component differences in the assembled vector. Coverage is limited to two numeric meta-train families: blackmore, 629–630 training rows with 4 features, and `cardata-ces11`, 800 rows with 8 features, each at cuts 0–3. Changes to arithmetic, libraries, build/ISA flags or summation order require requalification; this gate is not evidence for wide or category-heavy contexts. Never relax its tolerance to make a build pass.

`VerifiedCpuContext` has private fields and can only be constructed by the verification path outside explicit test support. A syntactically valid JSON vector cannot create admission authority. Sources: `crates/jope-policy/src/context_producer/admission.rs:4`, `:115`, `:234`, `:297`; `crates/jope-pipeline/src/cpu_context_bridge.rs`.

## Context worker contract

In the bounded ASM flow, encoder weights are loaded inside `jope-context-worker` (`jope-policy`, feature `context-producer`). `--version` prints `jope-context-worker/1`. Its exact eight named arguments are:

```text
jope-context-worker --mode produce|qualify-case --input ABSOLUTE_PATH --input-sha256 HEX64 --assets ABSOLUTE_ENCODER_FILES_JSON --assets-sha256 HEX64 --source-sha40 HEX40 --producer-sha256 HEX64 --output ABSOLUTE_PATH
```

Extra arguments such as `--dataset-root` are refused. The child requires OMP, OPENBLAS, MKL, BLIS, NUMEXPR and RAYON thread limits of 1, `VECLIB_MAXIMUM_THREADS=1` and `CUDA_VISIBLE_DEVICES=-1`; it authenticates its own executable and the component closure before model loading. The pinned bundle manifest begins `5e7884ec…`; each inventoried file/notice is verified with a 512 MiB per-file limit. Outputs are create-new, mode 0600 and fsynced. Stdout records `jope-context-worker/1 <output_sha256>`. A failed `qualify-case` comparison retains `<stem>.unqualified.json`, exits nonzero and does not create the requested successful output.

For current `jope optimize`, input derives from `dataset.cut(0)` fitting rows and is retained at `<journal>/asm-context-inputs/<input_sha256>.json`. Admitted output is `<journal>/asm-contexts/<neural-arm-id>/context.f32` plus `bundle.json`; the vector contains 2,048 little-endian float32 values. Invoke this through the supervised planner/controller contract, not as an unsupervised shortcut.

## Stored model assets and the active encoder distinction

### Bundle manifests and declared asset sizes

The metadata-only S3 audit on 2026-09-23 fetched the three bundle manifests below and pinned their returned bytes, ETags and lengths. No model bodies were downloaded for this chapter. These are declared object sizes from the retrieved manifests; they do not measure resident memory, compute cost or a new numerical qualification.

| Bundle asset | Bytes | Role in this chapter |
|---|---:|---|
| `policy-step-60000/policy.safetensors` | 111,199,088 | Accepted policy export identity, consumed only with its checkpoint binding |
| `policy-step-60000/context_transforms.safetensors` | 8,413,808 | Accepted fixed context assembly transforms |
| `jope-prime-gen31-v2/model-bundle/hyperion_v5_arm_b.safetensors` | 6,650,692 | Exact Hyperion weights pinned by the current CPU context producer |
| `jope-prime-gen31-v2/starforge/starlight_core_regressor.safetensors` | 233,183,532 | Foundation component; manifest describes512 hidden features pooled into2048 |
| `jope-prime-gen31-v2/model-bundle/target_encoder_fused_v1…safetensors` | 78,615,176 | Target component; manifest describes normalized64+64+384 fused representation |
| `hyperion-v7/model.safetensors` | 246,456,988 | Separate stored bundle, not the accepted context producer's pinned weights |
| `jope-prime-gen31-v2/model-bundle/jope_policy.safetensors` | 108,846,812 | Separate gen31 policy artifact; not step-60000 |

The three current model bundle manifests are `hyperion-v7`, `jope-prime-gen31-v2`, and `policy-step-60000`. Their existence and object sizes do not establish new training, inference parity or serving promotion. [§](02-reading-this-manual.md#the-identity-tuple) gives their roles among the product identities, and [§](02-reading-this-manual.md#release-and-installation-are-independent-from-model-promotion) gives the acceptance boundaries. Historical lineage-A material in learning storage cannot replace its missing canonical training-state chain.

### Encoder identity pitfalls and provenance declarations

The storage name `hyperion-v7` must not be confused with the CPU implementation's `V7Model` Rust type. The actual current producer pins weight hash `88b9e834…`, which matches the v5 arm-B asset, with row width 128. The separate stored v7 manifest describes dimension512, eight heads,12 layers and six bands. Replacing the pinned asset based on its newer-looking name changes the encoder identity and invalidates compatibility.

Two identity pitfalls matter when maintaining this boundary. Hyperion `load_settings` reports a legacy 16-hex identifier, not a SHA-256 digest. The producer therefore authenticates manifest bytes before and after the path-based settings load, authenticates weights before and after `V7Model::load_cpu`, and requires embedding width 128. A legacy identifier must never be compared with or substituted for a cryptographic admission pin. Separately, ordinary `jope encode` resolves `HYPERION_V7_WEIGHTS` as a **bundle directory**, else `$JOPE_HOME/models/bundles/hyperion-v7` (relative to the working directory when unset). The default stored bundle is 512-dimensional and differs from the pinned v5-arm-B producer; both manifests use the loader string `version: v7`, so that string cannot establish checkpoint identity. `jope encode` is not the qualified 2048-value ASM context interface. Current documentation drift remains at `docs/models.md:105`, which supplies a file instead of the expected directory, and the default-v5 comment in `crates/hyperion/src/hyperion.rs:28`. Use exact asset hashes and the qualified context-worker flow.

The retrieved manifests also contain internal/redistribution/provenance declarations that vary by artifact. They are retained evidence about the bundle metadata, not proof that every object is approved for every distribution. The active private run's exact release/input inventory remains the authority for which files it used; S3 presence alone does not establish serving eligibility. Sources: `crates/jope-policy/src/context_producer.rs:41`; custody `metadata-receipts/veox-jopedime-models-use2--bundles--policy-step-60000--manifest.json`, `…--hyperion-v7--manifest.json`, and `…--jope-prime-gen31-v2--manifest.json`.

## Experimental classification context

Current source contains the pure class-partition context representation. The pending training overlay adds the bounded feature producer, process admission and operational proposal wiring. Classification does not parse class names as numbers or reuse the regression target encoder by ordinal coding.

The feature blocks remain Hyperion512 and projected Foundation1024. The pending producer explicitly conditions the foundation path on neutral zero targets and labels this an experimental feature-only probe. Each class gets a centroid over the 128-dimensional training row encodings. The target block contains, per coordinate, the mean, standard deviation, minimum and maximum across class centroids, totaling 512. The centroids are unweighted, so class frequencies never enter the block. Classes are visited in first-row order, so a bijective renaming preserves even floating-point summation order. Original labels still appear in authenticated input identity, even though their spelling cannot influence numerical pooling.

The resulting lineage is `asm.experimental-class-partition-pool/v1`; `policy_compatible` is false. The false marker means it cannot enter the accepted regression policy loader. A declared binary input must have exactly two classes; multiclass requires at least three, and the fitting data must contain the declared classes. Missing classes are an explicit refusal. Sources: `crates/jope-pipeline/src/search_context.rs:278`; pending `crates/jope-policy/src/context_producer/classification.rs:1`.

At source `68f56404`, classification has no neural arm: planning records the accepted regression-only policy as unavailable for binary/multiclass data. The available `asm.generic` classification sampler uses `logit_ridge` for binary or `multiclass_ridge` for multiclass, with l2 in [1e-4, 10], preceded by no transform, `standardize`, `winsor_clip`, or both. Advanced-v3 mutation/crossover preserve that head family and allow at most 4 transforms. A classification ASM result at this source would therefore be a ridge-head transform search; the pending experimental neural consumer is a separate capability.
