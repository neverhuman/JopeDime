# Architecture, encodings, GP representation and model lineages

**Algorithm Synthesis Model (ASM) learns to generate and optimize executable algorithms from data.** The implementation combines a learned proposal policy with actual fitted evaluation and genetic variation. A generated program is a candidate recipe; a fitted predictor is that recipe plus the state learned from a specific training partition. A checkpoint is the proposal model. Those three artifacts have different identities and different evidence requirements.

This chapter describes source commit `68f56404c3e199e06413bfd645d3fe0803d7b96f`, the reviewed PR83 head. PR83 merged on 2026-09-23 at 16:34:47Z as `main` commit `a75e543d2166017cb0139e0a251d51d9a0c98e08`, which has the same tree (`176b75bc36d76c7ce4eaeb7234677ec013468889`), so the current-source references below also describe `main`. As of 2026-09-23 about 23:55 UTC, the only later merge to `main` was PR85, edition 1 of this manual, which produced `main` `bdfc95c63d905c647e41175e79dc30cdc8600287` at 22:33:36 UTC. PR85 changed only files under `docs/` plus `README.md` and `AGENT_CHAT.md`, and no file under `crates/` or `python/`, so the source references below still describe `main`. Changes described as **pending training overlay** refer to the stopped 64-file custody patch `595824ef6eb8e15872be058668503cedad1dd9a5ff98ac4700c6f713251d384c`. That patch includes experimental regression consumption; it is not in this source commit. Current-source references below are repository paths with source line numbers. Overlay references are explicitly marked and use the stopped overlay's line numbers.

## System at a glance: flow diagram and component map

![ASM implementation and pending learning loop](../figures/architecture-flow.svg)

The solid path has bounded regression execution evidence. Dashed stages require the named runtime acceptance; this diagram does not claim an already qualified self-improving training loop.

| Component | Main responsibility | Durable boundary |
|---|---|---|
| `jope-datasets` | Frozen dataset metadata, source and split identity | Content-addressed manifests, source hashes and exact row assignments |
| `jope-bench::panel` | Development-data preparation, methods, metrics, screen/freeze/confirm protocol, ASM orchestration | Campaign plan, prepared data, fits, frozen selections and reports |
| `jope-pipeline` | Search identity, bounded execution, original deadlines, process and output authentication | Started/finished records, arm windows, observations and receipts |
| `jope-policy` | Accepted regression checkpoint binding, CPU context producer, grammar-constrained proposals and learned ranking | Checkpoint receipt, vocabulary, qualified context and proposal evidence |
| `feat-math` | Executable recipe representation, variation and fitting | Canonical recipe identity, fitted state and prediction output |
| `jope-learning` | HWM and score ledgers; pending stricter teaching admission | Comparison identity, immutable blobs, hash-chained events and frozen teaching manifests |
| Python policy package | Existing trainable policy model; pending bounded training campaign and experimental proposal consumer | Model/optimizer checkpoints, architecture, task/vocabulary binding and training proof |
| Native CLI | Current planning/execution front doors; pending `train` orchestration | Pinned requests, source/runtime inventories and supervised execution |

The ordinary CLI and `asm` feature graphs in the stopped follow-on were inspected as manifest text: they have no edge to the engine, Torch, CUDA bindings or a local dependency cycle. That is a source graph check, not a compilation result. Python training runs as a supervised external process; native orchestration does not require linking Torch into the CLI.

## The learning cycle: what is proven and what remains

The most recent bounded origin proof at this source completed its run command in 239.143 seconds; independent report replay took 18.719 seconds. It demonstrated neural-origin fitting and evolution on one regression dataset, `230_machine_cpu`. On the same 34 held development rows the neural arm's refit scored an RMSE of 166.32 against 155.26 for the generic arm (`neural_win: false`). The arms had equal declared allowances, not equal fit time: context encoding and policy production ran inside the neural arm's clock, the neural arm stopped at its original deadline and the generic arm at its evaluation limit ([§](scheduling.md#arm-order-and-clocks)). This one-dataset result is therefore not a matched quality comparison. It did not execute the complete thirty-dataset demonstration, create current eligible teachers, train a new policy, or qualify a newly trained policy for serving. [§](status.md#canonical-proof-record-and-per-arm-facts) gives its exact receipts and denominators.

1. Preserve the frozen data schema, original target strings, category strings, missingness and row/group assignments.
2. Derive the proposal context using only the allowed fitting rows. Record preprocessing and encoder identities, then verify actual context production and the numerical qualification required by that context lineage.
3. Use the chosen checkpoint to sample in a bounded number of rounds, stopping once 64 distinct valid executable recipes exist; the first 64 in draw order form the neural pool, and any shortfall is filled as described in step 4. Score exactly the 64 pool programs with the mean of the four learned reward heads. That score orders the pool's slots and records `rank_log_weight`; it does not select or reject programs. This stage has no access to held-out scoring labels or fitted GP scores.
4. Place a 64-slot policy pool plus 16 exploration candidates into the neural arm's generation zero; preserve any policy-pool shortfall as `neural_rejection_fallback`, never as a genuine neural sample. Compare with an 80-candidate generic arm under the same declared budgets and executable grammar policy. Context encoding and policy production run inside the neural arm's clock, so equal declared budgets are not equal fit time.
5. Actually fit candidates on inner training folds and score their out-of-fold predictions. The measured outcomes, not model-predicted quality, determine the search's parents and winner.
6. Generate mutation and crossover descendants with explicit parent identities. Preserve duplicate/cache handling, rejected variations, unavailable methods and every fit failure.
7. Freeze a recipe and evaluate the unchanged development protocol. Its fitted fold artifacts are separate from its reusable recipe.
8. In the pending teaching path, admit a recipe only after its independently preregistered train-side OOF evidence beats a trivial baseline under the frozen numerical rule.
9. Freeze the eligible corpus, train an explicitly experimental policy, prove real updates and continuation, and export a pinned checkpoint. The pending campaign trains a new, smaller Python `JopePolicyModelV3` (width 256, two layers, four heads); it is a different architecture from the accepted native step-60000 model (width 512, six layers), not a continuation of it ([§](training.md#what-the-pending-campaign-trains)). This manual names that line **experimental regression lineage v1 (256×2, new initialization)**; [§](#model-lineages) keeps it separate from the accepted 512×6 lineage-B model.
10. Consume that exact new checkpoint in a subsequent bounded experiment. A successful export alone is insufficient evidence for this final step.

Steps 1–7 have current bounded regression evidence at the scope described above. The stricter teaching/training/experimental consumption integration is stopped source work awaiting the next reviewed lifecycle and actual runtime acceptance.

Steps 8–10 support four distinct claims, and evidence for one never establishes another (owner rule):

| Claim | What must establish it | Status |
|---|---|---|
| Eligible teacher | Admission under the preregistered train-side guard of step 8, with the full comparison identity; a hash, a stored score or a corpus row is not eligibility | Pending integration |
| New discovery | A program absent from the authenticated initial population of the search that found it, retained with its discovering arm and ancestry; a discovery cannot silently seed another arm | Proposed |
| Actually consumed information | Committed optimizer updates that measurably sampled the new rows, with that exposure recorded | Proposed |
| Improved policy | A matched downstream fitted comparison of the trained checkpoint with its parent under the same budgets, as in step 10 | Proposed |

The preregistered train-side guard exists only in the stopped overlay, and the demonstrated teacher count under it remains zero in the inspected evidence ([§](training.md#what-can-be-claimed-now)). A source investigation of that overlay found no initial-population or discovery identity and no per-update fresh-exposure record in the pending campaign (`plan-execution-20260923/boundary-resume-siblings.md` under the ASM custody root, [§](installation.md#physical-layout-and-identities)).

The owner's rules, also proposed, fix three family-level domains: teacher/train families, adaptive development families, and evaluation families kept outside training and selection. The frozen thirty-dataset panel remains adaptive development ([§](benchmark.md#adaptive-development-exposure)), and the exposure of the pretrained encoders to any family remains unknown. The permitted input closure covers records, contexts, transforms and all assets; an unused forbidden descriptor must be rejected before loading, not merely left unsampled ([§](data.md#audit-the-entire-input-closure-before-continuation)). The same investigation found that the stopped overlay's native family partitioning has only train and development domains, and that its Python campaign does not reject extra context-index entries.

Remaining architectural acceptance includes real eligible-teacher collection, actual new-parameter training/reload/continuation, subsequent fitted use of that newly trained checkpoint, classification operational qualification, complete fitted-state parity across supported deployment backends, the unchanged full30 panel, and measured warm/cold installation demonstrations. Reusable motif mining must retain full comparison identity and eligibility; new operators need ablation and compatibility tests. GPU inference portability remains future work. Numerical parity of the Candle CPU policy port with the trained tch forward pass is not established: the source states that logits and rewards from the port are unverified until reference fixtures exist, and its `cuda` feature has never been compiled (`crates/jope-policy/src/lib.rs:84-95`).

## GP representation: recipes, fitted predictors and the four artifact types

The private program `(pipe X (standardize) (ridge))` is a small executable algorithm description. It takes a feature table, fits standardization from training rows, then fits a ridge-regression head. It is an ordered typed recipe, rather than an already fitted set of coefficients. A different transform or head changes the algorithm. GP search works within its admitted grammar and task constraints; the presence of a code emitter does not grant arbitrary-language or arbitrary-operator coverage.

Keep these artifact types distinct:

1. **Private recipe:** the lossless canonical program and its configuration. It can be fitted again on new training data and can carry private discovery provenance.
2. **Native fitted predictor:** preprocessing state, category dictionaries, imputation statistics, transforms, calibration and head parameters learned from one exact fit. It predicts without training again.
3. **Emitted recipe:** a program in another language with `fit` and `predict` behavior. Running it on training data fits a new instance of the algorithm. Agreement between these refits tests recipe/runtime equivalence.
4. **Fitted inference bundle:** an allowlisted implementation of one exact already fitted predictor. It contains the selected inference state and an input boundary; qualification compares its predictions against the original saved native artifact.

## Encodings and the proposal context

### Regression context production

#### Original data and preprocessing

The context input binds prepared-data hash, exact training-content hash, ordered fitting-row hash, cut identity and preprocessing hash. It requires at least eight training rows and a nonempty declared feature schema. Numeric versus categorical kinds come from the schema rather than guesses based on strings. Numeric nulls are imputed using training-only medians, with a zero fallback for an entirely missing column; the recorded training transform contains its fitted location/scale. For categories, every present string is a value, including empty text or strings spelling `NA` or `null`. Only an absent value is missing. Original target strings remain separately authenticated before numerical regression parsing.

This context preprocessing is for the proposal encoder. It does not authorize discarding the original schema or replacing a fitted predictor's own training-only preprocessing. Source: `crates/jope-policy/src/context_producer/input.rs:84`.

#### Component dimensions and assembly

The current CPU producer uses three components with fixed pretrained assets and fixed transforms:

| Component | Production | Raw output | Contribution to policy context |
|---|---|---:|---:|
| Hyperion | Training feature columns only (no label), quantized from fitted quantile bounds; three embedding sweeps, scales 4/7/11/15, batch 1024, internal setting `target_chunks_per_sweep=1` (a chunking setting, not the label) | 128 values per row | Mean, population standard deviation, 10th and 90th percentile per coordinate: 512 |
| Foundation | Feature columns plus the training targets as in-context labels, prepared with the native deterministic recipe; target normalization computed in f64 after the native f32 cast | 2048 | Frozen mean-centering, learned 2048→1024 projection, then frozen normalization: 1024 |
| Target | Training targets, already standardized in preparation, then min–max rescaled to [0,1] by the training minimum and maximum (span 1 when constant); two frozen native target windows with seed 17 | 512 | Frozen normalization: 512 |

Assembly concatenates normalized Hyperion512 + projected Foundation1024 + Target512 and clips the full 2048-vector to [-32,32]. Only the Hyperion block is label-free: Foundation is conditioned on the training targets, so the first 1,536 values, which the policy configuration still calls `dataset_dim`, include target-conditioned values. It checks all component dimensions and finite values. The source explicitly checks that the Hyperion row width is 128; confusing row width with pooled512 changes the model input meaning. Sources: `crates/jope-policy/src/context_producer.rs:188`, `:278`, `:356`, `:399`, `:422`.

The deterministic foundation preparation selects at most 768 context rows and 256 query rows before `encode_standardized`; that encoder adds no further cap. Hyperion receives every supplied training row and feature column, with no row/column subsampling in this producer. The serialized input is capped at 128 MiB and every context job remains subject to the fixed 30-second worker deadline. On the eight final reference jobs at source `68f56404`, Hyperion-stage elapsed time ranged from 6.143 to 14.287 seconds; whole-job wall time ranged from 11.731 to 20.755 seconds. These small-case measurements do not establish coverage for wider or larger cuts. A timeout leaves the context unavailable; it does not permit a new deadline or silent subsampling.

The frozen numerical qualification contract is unchanged, but the implementation includes both execution and numerical corrections. Hyperion uses an embeddings-only sweep that skips the unused confidence head. Foundation uses the separate `encode_standardized` path: train-side feature moments accumulate in f64 before the retained f32 transform, and p10/p90 use Torch 2.7 FP32 rank and stable-lerp arithmetic. Ordinary `encode` retains its legacy feature-moment and percentile arithmetic. The standardized route consumes all rows supplied by the preceding deterministic foundation preparation without additional caps. Cached-decoder fixtures separately check small random-model operations/full-prefix agreement within 1e-4, row/padding behavior and sampling-token/RNG agreement. They do not establish that the Candle CPU policy port matches the trained tch forward pass: the source states that logits and rewards from the port are unverified until reference fixtures exist (`crates/jope-policy/src/lib.rs:84-92`). Native CUDA policy parity is also not established; the `cuda` feature has never been compiled (`lib.rs:93-95`). Sources: `crates/jope-policy/src/context_producer.rs:238`; `crates/jope-encoders/src/foundation.rs:345`; `crates/jope-encoders/src/foundation/support.rs:123`; `crates/jope-encoders/src/model/tensor_ops.rs:114`; `crates/jope-policy/src/model/cached.rs:270`; `crates/jope-policy/src/decode.rs:485`.

#### Qualification and admission

The CPU regression encoder has a stronger boundary than “the vector has 2048 values.” Eight fixed native-reference cases span two families and cuts 0–3. Qualification checks finite FP32 component and final-context values with maximum absolute difference at most 0.0003, exact asset/source/program identities, and actual bounded executions. Consumption binds those eight jobs plus the actual dataset context job: nine independently authenticated process proofs. The reference registry/index and transform hashes are pinned in source.

The eight retained reference outputs at source `68f56404` have final-context maximum absolute differences from 5.3644180e-5 to 2.8765202e-4. The worst case is `cardata-blackmore`, dataset `55e9dfc7a86361d7`, cut 0: 95.884% of the unchanged 3e-4 tolerance. Across all eight cases, maximum component differences are 4.127622e-6 for Hyperion, 2.908707e-5 for foundation and 3.911555e-7 for target. The frozen projection and projected-scale division can amplify small component differences in the assembled vector. Coverage is limited to two numeric meta-train families: blackmore, 629–630 training rows with 4 features, and `cardata-ces11`, 800 rows with 8 features, each at cuts 0–3. Changes to arithmetic, libraries, build/ISA flags or summation order require requalification; this gate is not evidence for wide or category-heavy contexts. Never relax its tolerance to make a build pass.

`VerifiedCpuContext` has private fields and can only be constructed by the verification path outside explicit test support. A syntactically valid JSON vector cannot create admission authority. Sources: `crates/jope-policy/src/context_producer/admission.rs:4`, `:115`, `:234`, `:297`; `crates/jope-pipeline/src/cpu_context_bridge.rs`.

### Experimental classification context

Current source contains the pure class-partition context representation. The pending training overlay adds the bounded feature producer, process admission and operational proposal wiring. Classification does not parse class names as numbers or reuse the regression target encoder by ordinal coding.

The feature blocks remain Hyperion512 and projected Foundation1024. The pending producer explicitly conditions the foundation path on neutral zero targets and labels this an experimental feature-only probe. Each class gets a centroid over the 128-dimensional training row encodings. The target block contains, per coordinate, the mean, standard deviation, minimum and maximum across class centroids, totaling 512. The centroids are unweighted, so class frequencies never enter the block. Classes are visited in first-row order, so a bijective renaming preserves even floating-point summation order. Original labels still appear in authenticated input identity, even though their spelling cannot influence numerical pooling.

The resulting lineage is `asm.experimental-class-partition-pool/v1`; `policy_compatible` is false. The false marker means it cannot enter the accepted regression policy loader. A declared binary input must have exactly two classes; multiclass requires at least three, and the fitting data must contain the declared classes. Missing classes are an explicit refusal. Sources: `crates/jope-pipeline/src/search_context.rs:278`; pending `crates/jope-policy/src/context_producer/classification.rs:1`.

At source `68f56404`, classification has no neural arm: planning records the accepted regression-only policy as unavailable for binary/multiclass data. The available `asm.generic` classification sampler uses `logit_ridge` for binary or `multiclass_ridge` for multiclass, with l2 in [1e-4, 10], preceded by no transform, `standardize`, `winsor_clip`, or both. Advanced-v3 mutation/crossover preserve that head family and allow at most 4 transforms. A classification ASM result at this source would therefore be a ridge-head transform search; the pending experimental neural consumer is a separate capability.

## The accepted regression model: lineage B, step 60,000

### Checkpoint binding and configuration

The accepted lineage is the recovered lineage-B step-60000 checkpoint. The binding verifies the state, original model, exported tensors, configuration, parameter digest and vocabulary. The rejected step-69000 identity is explicitly refused. A self-consistent receipt around unrelated weights cannot impersonate the accepted model. The model loader rejects missing, extra, incorrectly shaped or non-FP32 tensors.

| Accepted-model property | Value or behavior |
|---|---|
| Dataset context (historical name of `dataset_dim`) | 1536 FP32 values: Hyperion 512 from features only plus Foundation 1024 conditioned on the training targets, so not label-free |
| Target context | 512 FP32 values; all 512 retained by the configuration (`target_keep_dim` 512), although most of them may be constant zero (`python/src/jope/policy/train_v3.py:919-920`) |
| Combined context | 2048 FP32 values; 8192 bytes in little-endian vector form |
| Decoder width | 512 |
| Decoder depth | 6 layers |
| Attention heads | 4 |
| Feed-forward width | 2048 |
| Token vocabulary | 401 output tokens |
| Embedding rows | 657, including the configuration's 256 extra rows |
| Maximum configured sequence | 128 tokens; positional table includes the context position |
| Training dropout field | 0.1; disabled for inference |
| Reward ensemble | 4 heads |
| Tensor inventory | 136 tensors for this configuration, including unused inference heads and scalar loss weights |
| Current Rust arithmetic | Candle CPU FP32 |
| Native CUDA BF16 policy parity | Not established by the current CPU inference implementation |

Sources: `crates/jope-policy/src/config.rs:21`, `config.rs:39`, `crates/jope-policy/src/model.rs:1`, `model.rs:41`, `crates/jope-policy/src/lib.rs:133`.

### Decoder and reward architecture

The decoder and reward paths each encode context with a two-layer MLP: 2048→512, GELU, LayerNorm, 512→512, GELU, LayerNorm. The decoder prepends a projected context token to the token embeddings, adds positional embeddings, applies causal self-attention with padding-key masking through pre-normalized layers, normalizes the result and projects through the tied token embedding table narrowed to the output vocabulary.

The reward path averages the program-token embeddings over non-padding positions, with no positional embedding, combines that mean with a separately encoded context, and evaluates four reward MLPs. The combined representation contains the program embedding, its elementwise interaction with a context gate, and an absolute-difference block. A separate runtime head emits a positive runtime estimate through softplus. Learned reward is a proposal-ranking signal; it is not a measured dataset loss.

**Reward pooling is order-blind.** The mean has no positional input. For one context, two programs whose tokens form the same multiset, for example the same operators or parameter values in a different order, receive the same pooled representation and therefore the same four reward scores, up to floating-point summation order. The decoder still conditions on token order; only the learned ranking signal is affected. At this source that signal orders the neural pool's slots and records `rank_log_weight`; it does not select programs (step 3 of [§](#the-learning-cycle-what-is-proven-and-what-remains)). The pooling is the same in the Candle port (`crates/jope-policy/src/model.rs:14`, `:555-567`), in the native tch model it mirrors (`crates/jope-engine/src/neural.rs:496-505`), and in the Python `JopePolicyModelV3` that the pending campaign trains (`python/src/jope/policy/model_v3.py:391-404`; the stopped overlay's copy keeps it). The Python module docstring (`model_v3.py:7-9`) still describes a bidirectional reward encoder with attention pooling; the code discards that encoder, recording that it collapsed (`:258`), and mean-pools instead.

Native source also contains an isolated experimental `OrderAwareRewardEncoder` (`parallel.reward_program_attention_2x256x4.v1`), which adds positional embeddings and two attention layers. Its source comment states that production `Policy` does not construct it and that a separately qualified reranker must opt in with a new artifact; at this source only its unit tests use it (`crates/jope-engine/src/neural.rs:180-263`; `crates/jope-engine/tests/reward_sequence.rs:7-24`). Status: implemented but unqualified.

**Owner rule (proposed): measure before changing the architecture.** First measure how often distinct candidate programs collide on one token multiset, and the ranking regret those collisions cause. Defer architecture changes, including adopting an order-aware reward input, so that the first causal learning experiment keeps changes to architecture, objective, corpus and sampler distinguishable.

### Accepted identities and the product restriction

The source records these accepted identities:

| Identity | SHA256 |
|---|---|
| State | `8ac80c269cc47c253b6f53fa3bc1eb75bb2fcda0bdace3ff8554be0175c400d6` |
| Original model | `053df4f53b84d2ebe7c14fbf907a149760af3d11de4a57b5ff89be8c2d3e8129` |
| Canonical parameter values | `9c7057e7f6a9e348f030da73f03522977b8c08915d3a221d7d5ab2e4213fcd92` |
| Configuration | `2508fc8b8268ea000141c9b119ba3da4e777c506c366168912a88d03482939bf` |
| Original checkpoint vocabulary | `8d55fee4366acd2c175cf55bfd769cebc904770f5227a6fd50a613723dc85787` |
| Executable product restriction | `570b2e5e8d859a73b888f5d0bee27f59da0e2376536babce24c241604f818e81` |

The product restriction preserves the 401-token table while limiting executable heads. It removes ten head operators (`extra_trees`, `gam_boost`, `logit_ridge`, the four multiclass heads, `portfolio`, `ridge_gcv` and `tab_boost`), and the vocabulary has no operator token for `pls` or `hurdle2`. The regression proof bound both arms to this restricted vocabulary, and mutation replaces a head only with `histgbm`, `ridge`, `ridge_gcv`, `elasticnet`, `histgbm2`, `gpr` or `ksmooth`; crossover keeps a parent's head. Six of the seventeen registered regression recipes (`tab_boost`, `extra_trees`, `gam_boost`, `pls`, `hurdle2` and `portfolio`) are therefore unreachable by any ASM operator at this source, and `ridge_gcv` is reachable only through mutation. Sources: `crates/jope-engine/assets/grammar/featmath-jain-v1.json` (`restriction.removed_ops`); `crates/feat-math/src/evolve.rs:29-39`, `:1480-1492`; `crates/jope-bench/src/panel/optimize.rs:254`; `crates/jope-bench/src/panel/methods.rs:60-89`. A token-compatible restriction is different from inventing a new vocabulary or silently snapping parameters to nearby bins.

## Model lineages

A lineage is a chain of checkpoints sharing one architecture and parameter ancestry. A higher step number, a newer file or a shared vocabulary does not place a checkpoint in the accepted line. The table covers three lineage-B checkpoints and the planned experimental line, as of 2026-09-23 about 23:55 UTC.

| Checkpoint or line | Architecture | Role and status | Identity and sources |
|---|---|---|---|
| Lineage B, step 60,000 | Native tch `model.ot`: width 512, six layers, four heads, feed-forward 2048 | The accepted regression model and the only identity the current consumer admits. Available. | State `8ac80c26…`, pinned with its model, parameter, configuration and vocabulary hashes (`crates/jope-policy/src/lib.rs:133-150`, `:161`; `crates/jope-policy/src/config.rs:289-325`); [§](#checkpoint-binding-and-configuration) |
| Lineage B, step 60,100 | Same as step 60,000 | Historical hardware continuation: the September 14 qualification run carried AdamW for 100 updates from step 60,000 on the unchanged corpus, with zero fresh programs, at world 1 × microbatch 256 × accumulation 8. It is neither a serving model nor a production continuation, and the accepted binding refuses any step other than 60,000. | Final state `745be4b5…`; [§](training-history.md#the-rtx-step-60100-qualification-run) |
| Lineage B, step 69,000 | Same as step 60,000 | Endpoint of the last retained A100 block, steps 66,001–69,000. Unaccepted: an export receipt naming its state or model hash is refused before any other check. | State `58c4a1f6…`, model `f3694654…` (`crates/jope-policy/src/lib.rs:152-159`, `:249-253`; `crates/jope-policy/src/config.rs:177-182`); [§](training-history.md#a100-checkpoints-and-the-accepted-step-60000) |
| Experimental regression lineage v1 (256×2, new initialization) | Python `JopePolicyModelV3`: width 256, two layers, four heads, feed-forward 1024 | The line the stopped training64 campaign would train, starting from a new initialization. No real campaign has run with it. Pending integration. | Stopped patch `595824ef…`; [§](training.md#what-the-pending-campaign-trains) |

The two architectures are not interchangeable. The Python path cannot load step 60,000, and the accepted loader rejects the smaller model's tensor shapes, so lineage v1 is a new line rather than a continuation of lineage B. Compare the two only through matched downstream fits under one protocol, never through loss values ([§](training.md#what-the-pending-campaign-trains)). Training or exporting a lineage-v1 checkpoint does not move the accepted serving pointer ([§](training.md#plan-run-resume-verify-and-export)).

Historical lineage A and the legacy Python gen31 and round-1 runs are recorded in [§](training-history.md#dated-lineage-and-evidence-coverage); the current accepted consumer admits none of their checkpoints.

Owner naming rules for lineage v1 (proposed):

- The new initialization is `M_init`. It is named `M0` only after it has actually trained on its first frozen corpus, `C0`, and that run has passed the pending campaign's verification, which includes positive optimizer updates, changed parameter digests, exact reload, an actual resumed optimizer state, increasing step counters and at least two committed passes ([§](training.md#plan-run-resume-verify-and-export)).
- Passes through one corpus are continuation, not generations: the pending plan's two passes through `C0` (`--passes 2`) do not make two generations. A new generation needs the LOOP-01 chain: a new eligible target, actual exposure, changed parameters, compatible reload and improved fitted search.
- The pending corpus request can name an optional Python initialization checkpoint ([§](training.md#build-the-real-corpus-from-admitted-evidence)); the campaign then uses a weight warm start, and otherwise a cold initialization. This manual retains cold initialization as the plan for lineage v1. Warm-start support is an unselected interface option, not a qualified lineage start; the owner has not ruled whether lineage v1 admits a warm start, and no prohibition is inferred. Before any admitted run, record which form `M_init` actually took.
- Step 60,100 remains historical hardware continuation and step 69,000 remains unaccepted; neither seeds lineage v1.
