# W2-T01 accepted asset and consumer dependency table

**Task.** W2-T01, "Trace accepted assets and consumers" ([W02 tasks](../W02-model-encoder-semantics.tasks.md)). Owner: lead to record.

**Method.** A read-only source trace (Read plus grep) of the `bench` lane.
- Nothing was built, tested or executed.
- No asset was hashed, and `verify_inventory.py` was not run.
- No state beyond `planned` is claimed for any W2 task.
- Recorded receipts (`data/tensors.json`, `data/inventory-proof.json`, `data/encoder-assets.json`) were read but not re-verified.

**Source base.** Branch `feat/regression-remediation` at `a748a496982ef907aa060444e6462e73ea7dbdb9`.
- The trace began at `0e2ca312b31afd1221b9948ab277160ea2d23929`.
- Among the cited files, only `crates/jope-policy/src/{grammar,decode,producer}.rs` changed between the two commits. Their anchors below were re-read at `a748a4969`.
- Every other anchor is unchanged between the two commits (`git diff --stat`).

## 1. Conventions

**Who issues the expected value.**

| Code | Issuer |
|---|---|
| **C** | A constant compiled into reviewed source. Today this is the only "release decision" any receiving loader consults. Its authority rests on the reviewed source plus the controller's pin of the executable (`producer_sha256` / `expected_program_sha256`). |
| **P** | A pin from the caller or controller plan, independent of the artifact being admitted. |
| **R** | A field of the per-request `producer::Request`. It is validated against C where stated. |
| **S** | The candidate's own sidecar or receipt: self-attested. |
| **—** | No expected value. |

**How the bytes are consumed.**

| Code | Meaning |
|---|---|
| **VB** | Validated bytes: the buffer that is parsed is the buffer that was hashed and compared. |
| **RV** | Re-verified re-read: a later read re-hashes exactly the bytes it returns. |
| **CL** | Check-then-load: the hash is taken on one open, and the content is parsed from another open by path. |
| **UN** | No identity check. |

## 2. Static release identity vs request-specific identity

### 2.1 Static release identity

This identity is spread across compile-time constants. No release manifest ties it together (see G-01).

| Component | Identity (issuer) | Declared at | Receiving check |
|---|---|---|---|
| step-60000 `state.json` | `8ac80c26…` (C) | `jope-policy/src/lib.rs:141-142` | `config.rs:268-270`, `:428-431`; `lib.rs:206-209` |
| `model.ot` | `053df4f5…` (C) | `lib.rs:144-145` | `config.rs:432-436` |
| step | 60000 (C) | `lib.rs:147` | `config.rs:422-426` |
| Parameter hash (sorted names + LE f32) | `9c7057e7…` (C) | `lib.rs:152-153` | `config.rs:437-440`; recomputed from the loaded tensors at `model.rs:234-238` and `lib.rs:210-213` |
| Config (`PolicyConfig::regression(401)` layout) | `2508fc8b…` (C) | `lib.rs:158-159`; `config.rs:94-98` | Claimed and recomputed: `config.rs:441-450`; `lib.rs:214-223` |
| Rejected step-69000 | state `58c4a1f6…`, model `f3694654…` (C) | `lib.rs:161-167` | `config.rs:256-260`; `lib.rs:257-264`; `policy_export.rs:210-213`, `:251-254` |
| Vocabulary `featmath.json` | `8d55fee4…` (C) | `lib.rs:169-170` | `grammar.rs:918-963`; `lib.rs:231-249`; `config.rs:451-455` |
| Product restriction `featmath-jain-v1` | `570b2e5e…` (C) | `lib.rs:174-179` | Same as above |
| `policy.safetensors` file | `4b4b5a40…` (C) | `config.rs:121-122` (private); also as literals in `producer.rs:104` and `jope-bench/.../optimize.rs:259-262` | Enforced only by some callers (G-02) |
| Context transforms | `1b779c6f…` (C) | `context_producer.rs:38-39`; duplicated at `jope-pipeline/src/search_context.rs:10-11` and `jope-policy-producer.rs:149` | `context_producer.rs:107-112`, `:398-401`, `:376-382` |
| Assembly spec | `6582df23…` (C) | `context_producer.rs:40-41`; duplicated at `search_context.rs:8-9` and `jope-policy-producer.rs:136` | `admission.rs:521`, `:612`; `search_context.rs:252`. What this hash covers is **not established in this repository** (§7). |
| Hyperion weights / manifest | `88b9e834…` / `ebb745f6…` (C) | `context_producer.rs:42-43` | CL windows (G-10a, G-10b) |
| Hyperion seed and sweep plan (3 sweeps; scales 4/7/11/15; batch 1024; 1 chunk) | code literal | `context_producer.rs:44`, `:243-250` | Pinned only via the producer binary hash |
| Component bundle manifest (`model-bundle.v1.json`) | `5e7884ec…` (C) | `jope-encoders/src/bundle.rs:68-69`; equals `worker.rs:10-11` | `bundle/loading/loaded.rs:7-12` |
| Foundation `starlight_core_regressor.safetensors` | `f663ae49…`, 233,183,532 bytes, 504 tensors, inventory `f4939f17…` (C) | `bundle.rs:60-67` | `bundle/loading/manifest.rs:126-160`; file bytes `integrity.rs:3-32` |
| Target `target_encoder_fused_v1` | artifact `be96adc6…`, metadata `3849615a…` (C, taken from the pinned manifest). These values are as recorded in `core-ip/data/encoder-assets.json:59-64`; not re-hashed here. | — | Seed 17, windows 2×32, normalization and oracle recipe `fee6643c…`: `bundle/validation/target.rs:56-71`, `:104-111`, `:211-216` |
| Foundation seed and context/query rows (20260707; 768/256) | code literal | `foundation.rs:18-20` | Producer binary hash only |
| Label normalizations (Foundation f64 standardization; Target min–max) | code | `context_producer.rs:264-270`, `:280-287` | Producer binary hash only |
| Target window seed | 17 (code) | `context_producer.rs:288` | Cross-checked against bundle metadata at `:135` |
| Qualification policy | policy string, registry `22b5de5f…`, native index `6c81da15…`, `MAX_ABS=3e-4`, 8 cases / 2 families / cuts 0–3 (C) | `admission.rs:4-12`, `:503-507`, `:660-663` | `admission.rs:500-717` |

### 2.2 Request-specific identity

| Dimension | Carrier (anchor) | Bound by |
|---|---|---|
| Dataset / training scope (context) | `TrainingIdentity{dataset,prepared,training,fitting_rows,cut<5}` (`context_producer/input.rs:5-25`). `training_sha256` covers rows including labels (`:79-93`). Fitting-row order is checked at `:105-114`. | P: `AdmissionPins.training/input/preprocessing` (`admission.rs:111-113`, `:340-345`); `AdmissionExpectation` (`jope-pipeline/src/cpu_context_bridge.rs:270-311`) |
| Per-request fitted preprocessing | Impute, mean and scale per column; target mean and scale (`input.rs:121-216`). Recorded as `preprocessing_sha256` (`context_producer.rs:193`), recomputed at `:372-375`. | P (above) |
| Policy request scope | `prepared/training/cut/context_sha256` (`producer.rs:16-35`) | Against the qualified context (`jope-policy-producer.rs:104-114`) or the native receipt (`:124-133`). The controller checks it at `neural_search.rs:1400-1403`. |
| Teaching scope | `encoder_receipt::Scope`: comparison, family, folds, fitting/forbidden members (`jope-learning/src/encoder_receipt.rs:48-55`); `Inputs` (`:69-74`) | TeachingBinding pins (P, `:237-245`) |
| RNG | Policy sampling, fallback and exploration seeds (R, `producer.rs:25-27`; derived at `seeds.rs:446-453`); ChaCha8 fallback (`producer.rs:498-506`) | R. The encoder RNGs are static (Hyperion 20260623, Foundation 20260707, Target 17); only the row count `n` changes splits and windows. |
| Device / backend | Admission is CPU-only: thread and GPU environment enforced at `worker.rs:192-209`; `BACKEND` checked at `context_producer.rs:350-354`. `CudaProducer` is benchmark-only (`:591-633`). The policy `load_bound(.., device)` (`model.rs:215`, `:239-243`) does not record the device in `BoundIdentity` (`:117-123`). Ranking backend string: `ranking.rs:24`, checked in `read_claims` (`producer.rs:271-311`). Training uses `Device::Cuda(rank)` (`training.rs:1058`). | The device is **absent** from `BoundIdentity` and from the training resident-cache keys (G-11) |

## 3. Loader and entrypoint inventory

| ID | Entrypoint (anchor) | Bytes opened | Identity check: expected value (issuer) | Mode | Consumer |
|---|---|---|---|---|---|
| E01 | `CheckpointBinding::for_proposal` / `parse` (`jope-policy/src/config.rs:229-238`, `:240-336`); `require_accepted_identity` (`:421-457`) | Export receipt `jope.checkpoint_export.v1`, as caller-supplied bytes | Rejected identity refused first (`:256-260`). Consistency checks on S only: `accepted_identity==true` and `expected_state_sha256==state` (`:262-267`); `parameter_sha256_recomputed==parameter_sha256` (`:282-286`); the `verified_bindings` list (`:287-296`); `tensor_count` (`:315-318`). Checked against C: state (`:268-270`); step, state, model, parameters, config claimed+recomputed, and vocabulary (`:421-457`). **Not** checked against C: `policy.output.sha256` (`:328`), per-tensor sha256s (`:308-314`), `corpus_sha256` (`:327`), the receipt bytes. Challenger path uses P (`:341-373`). | VB | `jope-policy-producer.rs:151-154`; `neural_search.rs:1411-1414`; `optimize.rs:506-511`, `:907-912`; `inference_cuda.rs:58-60` |
| E02 | `PolicyModel::load_bound` (`model.rs:215-251`) | `policy.safetensors`, one `fs::read` (`:216-217`) | File sha256 equals the receipt's value (**S**, `:218-222`). Names, shapes and F32 equal `expected_inventory(config)` (C via the config hash, `:226`, `:157-192`). Per-tensor entries equal the receipt's (S, `:227-233`). The recomputed parameter hash equals the receipt's (S, `:234-238`), which equals C after `require_accepted_identity` / `require_accepted_policy` (`lib.rs:198-225`). | VB | `seeds::propose` (`seeds.rs:467-488`), `propose_challenger` (`:495-520`); `Bindings::for_accepted_policy` (`ranking.rs:68-83`) |
| E03 | `PolicyModel::load_safetensors` (`model.rs:206-210`) | Policy file, read by path | Inventory only; the model is unbound | UN | Tests only (grep: `tests/jope_dime.rs:1278`; `challenger_tests.rs:169`, `:432`). An unbound model is refused by `propose` (`lib.rs:202-204`). |
| E04 | `bind_vocabulary` (`grammar.rs:918-963`); `require_accepted_vocabulary` (`lib.rs:231-249`) | Vocabulary JSON bytes | The hash of the parsed buffer equals the checkpoint label (C via E01) or `PRODUCT_RESTRICTIONS` (C). The restriction block is declared (S). The full token-compatibility proof `verify_restriction` (`grammar.rs:968-`) is exercised by tests, not at bind time; the restricted bytes are C-pinned. | VB | Decode masks; `score_pool` encoding (`ranking.rs:382-384`) |
| E05 | `jope-policy-producer` worker (`src/bin/jope-policy-producer.rs:37-199`) | Request (`:70-72`, no expected hash; the controller authenticates it at `neural_search.rs:1425-1431`). Source manifest (R, `:73-76`). Context of 8192 bytes (R `context_sha256`; finite, \|x\|≤32; `:77-89`). `--context-bundle` (`:90-115`, no expected hash; G-03) **or** the native `--context-receipt` (`:121-142`, no expected hash; fields checked against R and C). Transforms hashed against C (`:147-150`), then unused. Checkpoint receipt (`:151-154`, no expected hash). Vocabulary (R restricted to two C values, `producer.rs:121-128`). Weights (E02, `:159-160`). | `producer::produce` compares the loaded identities with R (`producer.rs:196-214`). `Request::validate` requires the accepted weights file to equal C (`producer.rs:100-120`, `:104`). | VB for every consumed file | Envelope `asm.private-policy-producer/v1` (`producer.rs:160-190`); controller reader `read_claims` (`:271-311`) |
| E06 | `CpuProducer::load_on_device` (`context_producer.rs:106-156`) | Hyperion (E07, E08), component bundle (E09, E10), transforms (E11) | `EncoderFiles` pins checked against C (`:107-112`). For the bundle root, only "absolute and physical" is checked (`:129-133`); its identity comes from E09. Target seed == 17 (C, `:135`). `EncoderIdentity` is built from the loaded manifest (`:136-143`; G-04). | see E07–E11 | `produce` (`:163-203`); `qualify_case` (`admission.rs:151-166`); classification route; `CudaProducer` (benchmark) |
| E07 | `verified_hyperion_settings` (`context_producer.rs:494-506`) → `hyperion::load_settings` (`hyperion/runtime.rs:225-276`) | Hyperion manifest JSON | Pin against C (`:495-499`); hashed before (`:499`) and after (`:503`) the path-based parse (`:502`); `emb_dim==128` | **CL** | `q_low/q_high/pad_u8/quant_scale/max_patch` (`context_producer.rs:235-251`); `layers/n_bands` (`hyperion_v7/backend.rs:166-188`) |
| E08 | `V7Model::load_cpu` / `load_cuda` (`backend.rs:51-72`, `:132-138`) → `load_tensor_map` (`hyperion_v7/weights.rs:22-29`) | Hyperion safetensors, opened by path (candle `safetensors::load`) | None inside the loader. The caller hashes the file before (`context_producer.rs:113`) and after (`:128`). Non-inference keys are silently dropped (`weights.rs:10-28`), and there is no inventory. | **CL** | `run_composite_embedding_sweep` (`context_producer.rs:237-252`) |
| E09 | `LoadedModelBundle::load` (`bundle/loading/loaded.rs:4-58`) | Root opened `O_DIRECTORY\|O_NOFOLLOW` with a dev/ino check (`loading.rs:16-41`). Manifest and every listed file and notice are read descriptor-relative (`loaded.rs:18-27`; `loading.rs:6-14`, `:43-123`). | Manifest equals C `5e7884ec…` (`loaded.rs:7-12`). Metadata (`manifest.rs:4-162`). Per file: bytes, sha256 and safetensors inventory against the manifest (`integrity.rs:3-32`, `:163-272`). Registry, JOPE and Target validators (`loaded.rs:28-51`; `validation/target.rs`). | VB, **then the bytes are discarded**; only inventories, JSON and parsed target metadata are kept | E10; E06 identity |
| E10 | `read_verified` (`loaded.rs:74-85`) → `StarlightFoundationEncoder::load_bundle` (`foundation.rs:317-329`); `FusedTargetEncoder::load_on_device` (`target_encoder.rs:123-142`) | Foundation and Target safetensors, re-read descriptor-relative | Re-verified against the manifest entry. Target also checks a tensor count of 171 (`target_encoder.rs:134-136`). | RV | `encode_standardized` (`context_producer.rs:271-274`); `encode_with_windows` (`:292-295`) |
| E11 | `FrozenTransforms::load` (`context_producer.rs:397-435`) | `context_transforms.safetensors` | Pin against C (`:398-401`); one read and hash (`:402`, `AssetFile::read` `:53-69`); parsed from that buffer (`:403`); exactly 8 F32 tensors of fixed shape, finite, with scale > 0 (`:406-433`) | VB | `assemble` (`:436-457`) |
| E12 | `worker::asset_inventory` / `verify_file` (`context_producer/worker.rs:102-151`, `:161-190`) | Every bundle file and notice, Hyperion, transforms: streaming hash by path | Manifest checked against C; files against the manifest | Pre-pass only; loading re-opens the files (E06–E11) | `worker::execute` (`:221`); `jope-pipeline/src/context_qualification.rs:36-48`, `:158-160`; `classification_bridge.rs:96-99` |
| E13 | `worker::execute` (`worker.rs:191-263`) | Own binary, input, assets JSON | Binary checked against invocation `producer_sha256` (P, `:210-217`); input and assets are `AssetFile` (P, `:218-219`) | VB | Writes output with `create_new` (`:265-283`) |
| E14 | Admission (`admission.rs`): `case_inputs` (`:219-229`), `aggregate_qualification` (`:234-265`), `verify_qualified` (`:297-387`), `verify_component_qualification` (`:411-458`), `read_manifest` (`:500-665`), `compare` (`:666-717`) | In-memory manifest, qualification, artifact and attestation bytes | Checked against `AdmissionPins` (P, `:305-323`, `:359-377`); verification policy, registry and index (C, `:509-524`); native receipt and component chains (`:528-659`); ≤3e-4 (C). `artifact.encoder == qualification.encoder`: both are producer outputs (`:346-349`); `qualification.encoder` is copied from case 1 (`:252-260`). | VB | `VerifiedCpuContext` (`:115-146`) |
| E15 | `CpuContextArtifact::validate` (`context_producer.rs:347-391`) | Artifact JSON | Transforms, assembly, Hyperion weights and manifest checked against C (`:376-382`); `component_bundle`, `foundation_weights`, `target_weights` and `recipe` are **not** (G-04) | — | E14 |
| E16 | `QualifiedContextBundle::verify` (`producer.rs:514-533`), called from the worker (`jope-policy-producer.rs:93-115`) | Bundle JSON, no expected hash | `verify_qualified` runs with the bundle's **own** `pins` (S). Then the context is compared with R fields and the actual vector (`:104-114`). | VB (in memory) | Policy input |
| E17 | `cpu_context_bridge::load_qualified` (`cpu_context_bridge.rs:270-330`) | Bundle, artifact, vector and transform `Artifact`s | `Artifact::verify` (`execution.rs:102-114`) followed by re-reads (`:294-295`, `:313`). Pins checked against `AdmissionExpectation` (P, "from the frozen caller input plan", `:270-271`, `:296-311`). Transforms against C (`:289-293`). | CL for the bundle JSON (content bound by inner pins); artifact rebound by E14 | `search_context::load_for_comparison` (`search_context.rs:90-158`) |
| E18 | `VerifiedRegressionContext::load` (`search_context.rs:171-201`); `validate_native_receipt` (`:232-275`) | Native receipt, vector and transforms; `Artifact`s pinned by the campaign manifest (P, `:40-42`) | Receipt fields checked against training (P) and against C (assembly hash and backend string). The vector is re-read (`:182`) after verify (`:181`), and the returned `sha256` is the pin, not the digest of the bytes read (`:197`). | **CL** | `neural_search.rs:1400-1403` |
| E19 | Policy-worker verification (`neural_search.rs:1400-1466`) | Checkpoint receipt, source manifest, request | Program pin (P, `:1405-1408`); receipt verify then re-read → `for_proposal` (C, `:1409-1414`); state and weights equal the request (`:1415-1419`); closure contains the weight and vocabulary digests (`:1450-1464`) | CL (content C-bound) | Schedules E05 |
| E20 | `encoder_receipt` (`jope-learning/src/encoder_receipt.rs:84-160`, `:177-207`, `:250-272`) | None (in memory) | Structural equality with `TeachingBinding` (P, `:237-245`). The module doc calls it a producer claim (`:5-7`). | — | `eligibility/context.rs:150-161`; `eligibility.rs:1263`. **No in-workspace producer** emits `jope.encoder_receipt/v1` (grep finds no reference outside `crates/jope-learning`). |
| E21 | `policy_export::export_on_cpu` / `load_checkpoint` (`jope-engine/src/policy_export.rs:786-920`, `:204-306`, `:122-145`) | `state.json` (one read), `model.ot`, `config.json` | State checked against the operator's `expect_state_sha256` (P, `:214-227`). The `model.ot` hash is checked against the state (S, `:243`, `:247-250`), then the file is loaded by path (`:273`). The parameter hash recomputed from the loaded tensors is checked against the state (`:289-293`). Architecture must match exactly (`:294`). The receipt's `accepted_identity` is set from `state == C` (`:32`, `:872`). | CL, mitigated by the parameter hash | Produces the E01/E02 inputs |
| E22 | Training warm-start and resume (`training.rs:1069-1180`) | `state.json`, `config.json`, `model.ot`, `optimizer.ot` | The `model.ot` hash is checked against the checkpoint's own `state.json` (S, `:1089-1092`, `:1157-1160`), then `vs.load(path)` (`:1093`, `:1172`). Optimizer: `:1112-1117`, `:1161-1166`, `:1173`. No C accepted-identity check exists in jope-engine outside `policy_export.rs:32`, `:872` (grep). | **CL** | `Policy::new` (`neural.rs:286-373`) |
| E23 | Training resident caches (`training.rs:42-44`, `:1220-1295`); corpus index (`:806-813`); `teaching_store::load_resident` (`teaching_store.rs:1034-1056`) | Contexts (on a miss: per-row read and hash, `:1229-1234`); corpus index | Keys are listed in G-11 | Miss: VB; hit: no re-read | Training forward |
| E24 | jope-bench `inference_cuda` `load`, `qualify`, `request` (`jope-bench/src/panel/inference_cuda.rs:40-82`, `:103-166`, `:167-178`, `:537-550`) | Configuration dependencies | `Artifact::verify` (`:56`) followed by re-reads (`:59`, `:62`, `:66`); `load_bound` (`:69`, `:108`); `CudaProducer` (`:68`) | CL (content mostly C-bound) | Benchmark only (`context_producer.rs:591`: "Cannot produce a CPU admission artifact") |
| E25 (off-path) | Legacy `jope_embed` (`jope-policy/src/jope_embed.rs:147-148`, `:315-317`, `:349-350`, `:396`) | Preprocessing, `standardization.json`, npz projection, `<cache>/<dataset>.npy` | None | UN | No in-workspace consumer (grep) |
| E26 (off-path) | gen31 `JopePolicyModel::load_bundle` (`jope-encoders/src/jope_policy.rs:30-41`) | gen31 policy and metadata | Via the bundle | RV | Not used by the lineage-B producer (`context_producer.rs:77`: "its policy is unused") |
| E27 (off-path) | Foundation Encoder V2 (`jope-campaigns/src/encoder_v2_identity.rs:607-619`) | — | The cache key binds the encoder and input identities | — | Not referenced outside jope-campaigns (grep) |
| E28 (off-path) | `jope-models` fetch and cache (`jope-models/src/fetch.rs:39-66`) | `models/bundles/<id>.json`: committed first, then the cache (keyed by `bundle_id`), then the store | — | — | CLI only (`jope-cli/src/main.rs:136`, `:230`); no loader reads `jope.model-bundle/v2` |

## 4. Active operations and the bytes they depend on

| Op | Operation (anchor) | Asset bytes, mode | Static parameters | Request inputs |
|---|---|---|---|---|
| O1 | Training-row preprocessing: median impute, standardize, target standardize (`input.rs:84-216`) | — | Code | Rows, hash-checked (`input.rs:90-114`) |
| O2 | Hyperion quantile bounds and u8 grid (`context_producer.rs:224-236`) | Manifest, **CL** (E07) | `q_low/q_high/pad_u8` from the manifest | Prepared rows |
| O3 | Hyperion sweep → 128 values per row (`:237-256`) | Weights **CL** (E08) and manifest **CL** | Seed (C, `:44`); plan (`:243-250`) | Grid |
| O4 | Hyperion pooling: mean, population std, p10, p90 → 512 (`:459-481`) | — | Code | — |
| O5 | Foundation min–max, PCG64 permutation, 768/256 split (`foundation.rs:212-289`) | — | Seed (C) | Rows |
| O6 | Foundation f64 context-label standardization (`context_producer.rs:264-270`) | — | Code | — |
| O7 | Foundation forward and pool → 2048 (`foundation.rs:351-449`) | Foundation safetensors, **RV** (E10) | Bundle (C) | — |
| O8 | Target label min–max (`context_producer.rs:280-287`) | — | Code | — |
| O9 | Target windows, classical and neural, seed 17 (`target_encoder.rs:80-92`; `context_producer.rs:288`) | Target metadata, VB at bundle load (E09) | Seed (C), cross-checked at `:135` | `n` |
| O10 | Target encode → 512 (`target_encoder.rs:153-199`) | Target safetensors, **RV** (E10) | — | — |
| O11 | Assembly (`context_producer.rs:436-457`): subtract Foundation mean, multiply by the 2048→1024 projection transposed, normalize the Hyperion, projected and Target blocks, concatenate 512+1024+512, clip ±32 | Transforms, **VB** (E11) | — | Components |
| O12 | Native parity, ≤3e-4 on each component and on the context (`admission.rs:666-717`) | Reference manifest, VB in memory (P and C) | C | — |
| O13 | Policy input scaling by `dataset_rms` / `target_rms` (`model.rs:367-381`) | Config (C via hash) | — | Context (R) |
| O14 | Decode: `encoder.*` → `ctx_proj` → `token_emb` + `pos_emb` → 6 layers → `out_norm` → tied projection narrowed to 401 (`model.rs:382-404`, `:436-540`; `model/cached.rs:85-`) | Policy, **VB** (E02) | — | Context, tokens |
| O15 | Grammar-masked sampling (`decode.rs:364-450`) | Vocabulary, **VB** (E04) | — | Seeds, rounds, batch, temperature (R) |
| O16 | Reward path: `reward_encoder` → masked mean of `reward_token_emb` → `reward_norm` → `z_gamma` / `z_proj` → phi → 4 `reward_heads` (`model.rs:544-584`) | Policy, **VB** | — | Context, programs |
| O17 | Runtime `softplus(runtime_head([rz,g]))` (`model.rs:586-603`); reported as `predicted_runtime_seconds`, not used for ordering (`ranking.rs:25`, `:416-423`) | Policy, **VB** | — | — |
| O18 | Ordering by the mean of the four heads, ties broken by program sha (`ranking.rs:22-25`, `:409-428`) | — | Code | Pool |

`DecodeCache` (`model/cached.rs:41-49`) is batch-local and borrows `&PolicyModel`. It carries no state across requests and needs no key.

## 5. Tensor inventory and roles

The step-60000 policy has 136 F32 tensors and 27,796,494 parameters.
- Roles follow `docs/internal/core-ip/tools/verify_inventory.py:28-33` at the source base (the verifier is now the Rust port `crates/jope-records/src/inventory.rs`, `accepted_roles`).
- Counts come from `docs/internal/core-ip/data/tensors.json` (`role_parameters`, asset `4b4b5a40…`); they were not re-verified here.
- Candle's forward must reference exactly the decode ∪ rank roots (`verify_inventory.py:121-124`, `:158-161` at the source base; now `inventory.rs`, `forward_roots` and `verify`).

| Root | Role | Tensors | Parameters | Inference consumer (Candle) | Note |
|---|---|---|---|---|---|
| `encoder.{0,2,4,6}` | decode | 8 | 1,313,792 | `model.rs:439`; `cached.rs:92` | |
| `ctx_proj` | decode | 2 | 262,656 | `model.rs:440` | |
| `token_emb` | decode | 1 | 336,384 | `:443` (embed), `:522` (tied output) | Rows 401–656 (131,072 parameters) are never indexed (token < `vocab_size`, `:417`), and their logits are discarded (`:523`). These are inactive rows inside an active tensor (derived from source, not measured). |
| `pos_emb` | decode | 1 | 66,048 | `:445-448` | |
| `decoder.layers.0-5` | decode | 72 | 18,914,304 | `:463-511` | |
| `out_norm` | decode | 2 | 1,024 | `:521` | |
| `reward_encoder` | rank | 8 | 1,313,792 | `:551-554` | |
| `reward_token_emb` | rank | 1 | 336,384 | `:560` | Rows 401–656 unreachable; PAD is excluded from the mean (`:555-562`) |
| `reward_norm` | rank | 2 | 1,024 | `:563-567` | |
| `z_gamma.0` | rank | 2 | 262,656 | `:568` | |
| `z_proj` | rank | 2 | 262,656 | `:569` | |
| `reward_heads.0-3` | rank | 16 | 3,149,828 | `:571-585` | Predicted quality → ranking order |
| `runtime_head` | rank | 4 | 525,313 | `:586-603` | Reported runtime only |
| `stability_head` | training_only | 4 | 787,457 | none | tch `neural.rs:458`. In current source its loss is multiplied by 0.0 (`training.rs:1365`) and it is listed in `inactive_objectives` (`:1529`). |
| `reward_head_ablation` | training_only | 4 | 263,169 | none | tch `neural.rs:459`. Current source uses it in `contextfree_bt` (`training.rs:1352`, weight 0.05 per `:1529`). |
| `loss_log_vars.*` | training_only | 7 | 7 | none | Zeros at init (`neural.rs:292-303`); listed as inactive (`training.rs:1529`) |
| **Totals** | | 86 / 35 / 15 = **136** | 20,894,208 / 5,851,653 / 1,050,633 = **27,796,494** | | |

**Existing structural evidence (present at HEAD, not run by this trace).** `crates/jope-policy/tests/scorer_dependencies.rs:14-17`, test `perturbing_each_stored_head_moves_only_its_declared_consumers` (`:353-354`), uses synthetic weights. It shows:
- the reward heads move quality and ranking;
- `runtime_head` moves the reported runtime only;
- the context and decoder tensors move logits only;
- `stability_head`, `reward_head_ablation` and `loss_log_vars` move nothing.

**Known doc inaccuracy.**
- `model.rs:3-6` lists `runtime_head` among the tensors "that inference does not use".
- `PolicyModel::reward_runtime` does consume it (`model.rs:586-603`), `score_pool` reports it (`ranking.rs:409`, `:422`), and the module's own line 16 describes that computation.
- `verify_inventory.py:30-31` (now `inventory.rs`, `accepted_roles`) assigns it the `rank` role.

## 6. Gaps W2-T02..T04 must close

**G-01 (T02). No release-controlled semantic manifest is consumed.**
- Receiving loaders trust compile-time constants spread across many files: `lib.rs:141-179`, `config.rs:121-122`, `context_producer.rs:38-44`, `worker.rs:10-11`, `bundle.rs:25-72`, `admission.rs:4-12`.
- Several are duplicated as literals: `search_context.rs:8-12`, `jope-policy-producer.rs:136`, `:149`, `producer.rs:104`.
- The published manifest `models/bundles/policy-step-60000.json:13-66` lists only `policy.safetensors` and `context_transforms.safetensors`. It omits the export receipt, the vocabulary, Hyperion, the component bundle and the assembly spec.
- No receiving loader reads the manifest; E28 is a fetch CLI only.

**G-02 (T02). The policy file identity is taken from the candidate's own receipt.**
- `load_bound` compares the file sha256 with `policy.output.sha256` from the receipt (`model.rs:218-222` ← `config.rs:328`).
- Neither `require_accepted_identity` (`config.rs:421-457`) nor `require_accepted_policy` (`lib.rs:198-225`) checks it against `ACCEPTED_WEIGHTS_SHA256`. That constant is used only by the challenger's negative check (`config.rs:161`).
- The per-tensor sha list and `tensor_count` also come from the receipt (`config.rs:308-318`; `model.rs:227-233`).
- `corpus_sha256` is carried but never checked (`config.rs:327`).
- The receipt bytes are never pinned (`jope-policy-producer.rs:151-153`: `None`).
- Mitigations in place: the C parameter hash binds tensor names and values; the C config binds shapes and F32; the worker enforces the file constant through `Request::validate` (`producer.rs:104`).
- Residual: the safetensors header, `__metadata__` and layout bytes are unbound on the library path. The exporter's self-attestations (`accepted_identity`, `verified_bindings`) are checked for consistency only (`config.rs:262-296`).

**G-03 (T02, overlaps W3). The policy worker verifies a context bundle against the bundle's own pins.**
- `jope-policy-producer.rs:93-103` reads the bundle with no expected hash, uses `bundle.pins.artifact_sha256` as its expectation, and then `verify()` runs with `&self.pins` (`producer.rs:523-531`).
- This contradicts `admission.rs:96-98`: "controller inputs, not fields trusted from artifact JSON".
- Independent pins exist only on the controller side (`cpu_context_bridge.rs:270-311`).

**G-04 (T02). The encoder identity in CPU context artifacts is producer-reported and unpinned at admission.**
- `EncoderIdentity` is built from the loaded manifest (`context_producer.rs:136-143`):
  - `component_bundle` is the sha of the *re-serialized* manifest, not the pinned file hash `5e7884ec…`;
  - `foundation_weights` and `target_weights` come from the manifest;
  - `recipe` is a code literal.
- `validate` pins only 4 of the 8 fields (`:376-382`).
- `AdmissionPins` has no encoder field (`admission.rs:101-114`).
- Admission only checks equality between two producer outputs (`:252-260`, `:346-349`).
- Indirect mitigations: the C-compiled producer binary is P-pinned; there is a numeric parity gate; `qualification_sha256` is P-pinned by the caller plan.

**G-05 (T02). Lineage-B Foundation and Target are anchored in the gen31 bundle manifest.**
- The bundle is `jope-prime-gen31-v2`: policy_v3.2 with the gen31 vocabulary, grammar and registry (`bundle.rs:27-59`; `manifest.rs:8-22`).
- The lineage-B producer does not use that bundle's policy (`context_producer.rs:77`).
- Yet every listed file, including the gen31 policy, preprocessing, chimera and tabicl artifacts (`encoder-assets.json:21-97`), is read and hashed on each load (`loaded.rs:18-26`).
- T02 must bind Foundation, Target, Hyperion, transforms and assembly to the lineage-B release without inheriting unrelated identities.

**G-06 (T02). Native-context lineage rests on a sidecar receipt.**
- A `parallel.encoding.v1` receipt is accepted on its fields, `encoder_spec_sha256 == ASSEMBLY_SHA256` and a backend string (`search_context.rs:232-275`).
- Inside the worker, the receipt is read with no expected hash (`jope-policy-producer.rs:121-142`).
- The native encoder assets are not identified.

**G-07 (T02). Static calculation parameters live outside any manifest.**
- The Hyperion seed and plan (`context_producer.rs:44`, `:243-250`), the Foundation seed and rows (`foundation.rs:18-20`), the label normalizations (`context_producer.rs:264-270`, `:280-287`) and the recipe string (`:142`, never checked by any receiver) are pinned only through the producer binary hash.

**G-08 (T02). Unknown-field rejection is incomplete.**
- The export receipt is parsed as an untyped `serde_json::Value` (`config.rs:241-318`), so unknown fields are accepted.
- The following are already refused: same-shape weight edits (parameter hash), vocabulary permutations (vocabulary hash), and rejected checkpoints (`config.rs:256-260`).
- Mismatched cache manifests cannot be rejected because no cache manifest exists (G-11).

**G-09 (T03). Semantic products are untyped `Vec<f32>` / `&[f32]`.**
- `Components` has public fields and width-only validation (`context_producer.rs:313-326`). Hyperion and Target are both 512 wide, so `assemble` accepts a swap. It is publicly reachable through `CpuProducer::assemble` (`:308-310`).
- Raw Foundation 2048 and the final context 2048 go through the same `vector(v, 2048)` check (`:323`, `:384`).
- `PolicyModel` checks only length and finiteness (`model.rs:346-364`); so do `score_pool` (`ranking.rs:342-362`) and `context_from_bytes` (`lib.rs:266-277`).
- `encoder_receipt::Product` has no raw-2048 or projected-1024 Foundation variant, and some variants share widths (`encoder_receipt.rs:18-34`).
- The single-hash `Encoder{artifact_sha256,...}` (`:57-64`) cannot express the multi-asset closure, and nothing produces it (E20).

**G-10 (T04). Check-then-load by path, and re-reads after verification.**

| Site | Anchor | Status |
|---|---|---|
| a. Hyperion weights | `context_producer.rs:113` → `:116` / `:120` (`weights.rs:23`) → `:128` | An A→B→A replacement is undetected; there is no inventory to catch it |
| b. Hyperion manifest | `:499` → `:502` (`runtime.rs:226-227`) → `:503` | Same A→B→A exposure |
| c. Training warm-start / resume | `training.rs:1089-1093`, `:1112-1117`, `:1157-1173` | The expected value comes from the checkpoint's own `state.json` |
| d. Policy export | `policy_export.rs:243` → `:273` | Mitigated by the parameter hash (`:289-293`) |
| e. Native context | `search_context.rs:173-174`, `:181-182` | The returned sha is the pin, not the bytes read (`:197`) |
| f. Controller and bench re-reads | `cpu_context_bridge.rs:285`→`:295`, `:313`; `neural_search.rs:1409`→`:1411-1414`; `inference_cuda.rs:56`→`:59-66` | Mitigated by inner pins or C |
| g. Corpus index | Hashed and re-read (`training.rs:806-813`), then re-hashed for the key (`teaching_store.rs:1041-1045`), then re-read by `load` (`:1052`) | Unknown whether `load_impl` binds its parsed index to that key (unverified) |

Validated-bytes patterns to keep: `model.rs:216-225`, `context_producer.rs:402-403`, `loaded.rs:74-85`, `jope-policy-producer.rs:25-36`, `grammar.rs:918-963`. `LoadedModelBundle::path()` (`loaded.rs:64-70`) returns a path that can be re-opened unverified; grep finds no production caller.

**G-11 (T04). Cache keys omit parts of the dependency closure.**

| Cache | Anchor | Key | What the key omits |
|---|---|---|---|
| a. `RESIDENT_CONTEXT` | `training.rs:1220-1249` | `index.context_index_sha256`, a field of the corpus index, compared at `pipeline.rs:436` with the contexts-index file hash | A hit skips the per-context re-read and hash. The key ignores the `index.contexts` row set and order, the device, and any encoder or qualification identity. Whether a collision with differing rows can occur on one thread was not verified. |
| b. `RESIDENT_TENSORS` | `:1250-1295` | `corpus_hash` | The device |
| c. `teaching_store` `RESIDENT` | `teaching_store.rs:1034-1056` | root + index hash | No re-verification on a hit |
| d. jope-bench journal reuse | `inference_cuda.rs:537-539`, `:550`, `:168-173` | Journal file / directory `{mode}-{dataset}-{seed}` | Reused with only a `passed==true` gate; no plan, binary or model identity check. Benchmark only; the journal path (`:449`) was not traced. |
| e. `context_cost.cached` | `search_context.rs:143-152` | — | Relaxes the arm-charge check. The bytes still pass `load_qualified`. Where the flag is set was not traced. |
| f. Legacy Foundation cache | `jope_embed.rs:379-396` | Dataset directory name | Everything else (off-path) |

A reference design for a complete key: Encoder V2 binds the encoder and input identities (`encoder_v2_identity.rs:607-619`).

**G-12 (T02/T06). Fix the `runtime_head` doc inaccuracy (§5).**

## 7. Unknown or unverified (not guessed)

- **Assembly spec.** What `ASSEMBLY_SHA256` (`6582df23…`) is a hash of.
- **Transforms provenance.** How the values in `context_transforms.safetensors` were derived. The only record is "jain-datasets/tools/ot-export" (`models/bundles/policy-step-60000.json:8`), which is outside this repository.
- **Hyperion weights.** The ancestry of `hyperion_v5_arm_b` (its manifest says v7).
- **Foundation licence.** Its provenance is marked "Provenance-Pending" (`manifest.rs:149-157`).
- **step-60000 training.** Which training source produced it. The reference cited is hosted jope-dime main `07304d6f` (`lib.rs:3-9`). It is unknown whether that run's objective weights match the current `training.rs` (for example, stability ×0.0), and whether every stored parameter has AdamW moments (the optional check `verify_inventory.py:127-131` was not run).
- **Native reference runs.** The runs behind the 8 native reference cases are bound only through the pinned registry and index constants.

---

**Could not verify** (nothing was built, tested or hashed): whether the resident-cache collision can happen, whether `teaching_store::load_impl` re-checks the index bytes, where `context_cost.cached` gets set, how the bench journal path is derived, and what `ASSEMBLY_SHA256` covers.

---

## 8. Appended 2026-09-26: W2-T03 / W2-T02 (G-03, G-04, G-08) / W2-T07 implementation record

**Appended section.** Sections 1–7 above are the unchanged W2-T01 trace. This section records
one implementation slice; it does not change any task state (the lead records states).

**Source.** Branch `rr/w2-semantics`, cut from `feat/regression-remediation` at
`40babe407e1843507ed1f459bd6c7f8a19e199c4`. Code commit `46e97ff89` ("context producer: opaque
products, pinned encoder, request-issued admission"). Evidence root:
`/home/ubuntu/jope-rescue-20260921/regression-remediation-20260926/claude-w0-w12/runs/`, one
directory per run (`head.txt`, `status.txt`, `diff.patch`, `argv.txt`, `log.txt`, `exit.txt`).

**Observation level.** Fixture-proven and CI-shaped local runs only. No accepted-weight
inference, native reference replay, CUDA run or live controller execution was performed.

### 8.1 Reproductions before the fix

Run `w2b-repro-prefix-20260926T065217Z` (exit 100, three failing tests, all expected):

| Gap | What the old code accepted |
|---|---|
| G-09 / T03 | An in-process Hyperion/Target swap (`Components` validated and assembled); a serialized Target/signature swap and a serialized Foundation/final-context swap (`CpuContextArtifact::validate` passed). |
| G-04 | An artifact whose `component_bundle`, `foundation_weights`, `target_weights` and `recipe` were arbitrary (only four of eight fields were pinned). |
| G-03 | A `QualifiedContextBundle` whose pins it chose itself passed the verification-policy check and all four byte-pin gates of `verify`; the first refusal came later, while parsing the artifact. |
| G-08 | An accepted export receipt with unknown top-level, `checkpoint` and `policy.output` fields parsed and bound. |

### 8.2 State after commit `46e97ff89`

| Item | Change | Proving tests |
|---|---|---|
| G-09 / W2-T03 | New `context_producer/products.rs`: `RowEmbeddings` (rows×128), `HyperionSignature` (512), `RawFoundation` (2048), `ProjectedFoundation` (1024), `TargetBlock` (512), `FinalContext` (2048, ±32). Private fields, crate-internal named constructors, no `Deserialize`; each records its input scope (sha256 of the prepared rows' JSON = the artifact's `prepared_rows_sha256`). `Components` holds the three typed products and refuses mixed scopes. Serialized blocks are `ComponentVectors` claims (JSON unchanged; widths, finiteness and unknown fields refused at parse). `FrozenTransforms::replay` is the only receiver route to a `FinalContext`: it replays the frozen assembly and allows `ASSEMBLY_REPLAY_MAX_ABS = 1e-4` (a new declared intermediate tolerance; the 3e-4 parity gate is unchanged). | Compile-fail doctests in `products.rs` (E0308 ×3, E0451, E0624, E0277); `serialized_swaps_padding_and_truncation_fail_at_the_receiver`; `products_are_scoped_to_one_encode_and_serialize_in_the_historical_layout`; the W2-T05 goldens pass with their original expected values. |
| G-04 | `EncoderIdentity::accepted()` issues all eight fields from reviewed constants. `component_bundle` keeps the value every earlier producer recorded, the `ModelBundleV1` re-serialization digest `985074846ccfe92c818e920e5268ea5269a9e35f1fa2057ddae662266c6c7f9f` of the pinned manifest `5e7884ec…`; foundation `f663ae49…`, target `be96adc6…`. The producer (`CpuProducer::load`), `CpuContextArtifact::validate`, qualification case verification (both admission routes) and `ClassificationArtifact::validate_for_input` refuse any other identity. | `every_encoder_identity_field_is_pinned_to_the_release_decision`; ignored evidence test `accepted_encoder_identity_rederives_from_the_pinned_component_bundle`, run against the local pinned manifest `/home/ubuntu/jain-split/jain-starforge/artifacts/model-bundle.v1.json` (first derivation: `w2b-probe-bundle-identity-20260926T065422Z`). |
| G-03 | `producer::Request.context_admission_sha256` (optional; omitted when absent, so native request bytes and identities are unchanged) carries `AdmissionPins::sha256()` issued by the controller. The worker admits a bundle only through `QualifiedContextBundle::admit_for_request`: pins must hash to the issued admission before verification, the final context must replay from the qualified components and equal the policy-input bytes, and the handoff must fit the request. Native mode refuses a request that names an admission. `verify_policy_inputs` requires the request to issue exactly the admission `load_qualified` verified; `Comparison::validate` attaches only encoding outputs to the preregistered plan. | `worker_admission_requires_the_request_issued_pins`; `context_admission_pin_is_optional_and_bound`; `comparison_attaches_only_encoding_outputs_including_the_issued_admission` (jope-pipeline). |
| W2-T07 | `VerifiedCpuContext::handoff` → `ContextHandoff` (schema `jope.cpu-context-handoff/v1`): release identity (encoder, source, producer, verification policy, reference manifest), the six-step typed transformation chain, input scope (training identity, input, preprocessing, prepared rows), artifact/qualification/admission digests, final-context digest and consumer. `deliver_to_policy` refuses another consumer, an admission the request did not issue, another cut/training/preparation/context, a stale encoder release, and unknown chains or protocols. No conversion from an unqualified artifact exists. | `handoff_reaches_only_the_intended_consumer_for_the_issued_scope`; compile-fail doctests on `VerifiedCpuContext`/`ContextHandoff`. |
| G-08 | **Not changed.** The accepted `jope.checkpoint_export.v1` receipt is written by jain-datasets `tools/ot-export`; its field set is not in this repository and bounded host searches did not find the receipt. The in-repo exporter (`jope-engine/src/policy_export.rs:875-908`) writes fields the test fixture lacks (`runtime`, `checkpoint.directory`, `state_bytes`, `refused_identities`, `refused_steps`, `policy.source`, `total_tensor_bytes`, `inventory_sha256`, `architecture`, `relative_change`). An allowlist from either would be a guess that could refuse the real accepted receipt. | Reproduction only (8.1). |

### 8.3 Wire and identity changes

- `CpuContextArtifact`, `ProducedCase` and `QualificationCaseResult` JSON is unchanged; parsing is
  stricter (padded, truncated or nonfinite blocks and unknown component fields are refused).
- `producer::Request` gains an optional field; absent, the bytes and `identity()` are unchanged.
  Present, it is part of the request identity.
- `CpuProducer::assemble`/`CudaProducer::assemble` return `FinalContext` (derefs to `[f32]`);
  `encode_prepared` returns typed `Components`.
- Receivers now refuse any encoder identity other than `EncoderIdentity::accepted()`. Historical
  artifacts produced from the pinned bundle carry exactly these values; an artifact from any other
  bundle or recipe is refused.

### 8.4 Runs

Baseline `w2b-baseline-policy-20260926T064735Z` (81 passed, 6 skipped) and
`w2b-baseline-pipeline-20260926T064951Z` (138 passed, 1 skipped). Post-change:
`w2b-t03-doctests-20260926T070724Z` (16 doctests incl. 9 new compile-fail),
`w2b-w2-post-pipeline-20260926T070736Z` (139 passed), `w2b-w2-post-check-workspace-*`
(`cargo check --workspace --all-targets`), `w2b-w2-post-check-full-default-*` (the full-default
feature selection `jope-cli/asm,jope-cli/qualified-training-context`), and the post-commit
`w2b-w2-commit-*` runs at `46e97ff89` (jope-policy 87 passed / 7 skipped; 16 doctests; the
ignored encoder-identity evidence test passed against the local pinned manifest; jope-pipeline
139 passed / 1 skipped). The compile-fail doctests live in the `context-producer` module. The
`full-doctests` lane (`cargo test --workspace --doc`) should collect them, because jope-cli's
default `qualified-training-context` feature enables `jope-policy/context-producer` in a
workspace build; that is read from the manifests, not verified by running the lane.

### 8.5 Follow-ups outside this slice's partition

- **jope-bench `panel/optimize.rs:930`** builds the qualified-context request with
  `RequestPlan::request`, so no admission is issued; the worker and `verify_policy_inputs` now
  refuse that bundle. It must call `RequestPlan::qualified_request(context_sha256,
  bundle.pins.sha256()?)` with the pins it verified (`:1161-1184`).
- **jope-bench `panel/inference_cuda.rs:115-119`** (feature `inference-cuda`, no CI lane) puts
  `&c.hyperion`, `&c.foundation`, `&c.target` in one array; with typed products it should use
  `c.hyperion.values()` etc.
- **Teacher admission (W3, jope-learning / jope-cli corpus)** should consume `ContextHandoff`
  (consumer `TeacherAdmission`) and record its digest; `load_qualified` already replays the
  assembly for the jope-cli corpus route.
- **`AdmissionPins` controller-issued encoder pin**: adding a field needs a coordinated change of
  the jope-bench struct literal (`optimize.rs:1165`); the release decision is C-issued meanwhile.
- **PolicyModel / score_pool / `context_from_bytes`** still take `&[f32]`; `FinalContext` derefs
  to it. Typing those APIs touches jope-bench and many tests.
- **Cell surfaces** (the Hyperion u8 grid) do not cross an API and were not typed.

## 9. Appended 2026-09-26: W2-T02 versioned semantic manifest (branch `rr2/integration-c`)

Evidence root as in section 8; runs `w2c-*`.

**G-01 (the missing release-controlled semantic manifest).** This is a new capability, so there
is no defect to reproduce first: before this change no receiver consumed a release document.
The accepted identity was compile-time constants in six jope-policy files, duplicated as
literals in `producer.rs` (weights), `jope-policy-producer.rs` (transforms, assembly) and
jope-pipeline `search_context.rs` (assembly, transforms).

**Manifest.** `crates/jope-policy/release/policy-step-60000.semantic-v1.json`, schema
`jope.policy-semantic-manifest/v1`, identity `60f61cd8b58a1eaa72220a03e91af9be074793b60cd1a05df9c6cac1ac225935`
(`semantic::ACCEPTED_SEMANTIC_MANIFEST_SHA256`). Canonical JSON (sorted keys, compact,
integers and strings only, trailing newline). It binds:
- policy: state, model, step, parameters, configuration and its layout, the released weights
  file, 136 tensors / 27,796,494 parameters, the decode/rank/training-only head roots and the
  ordering rule;
- the rejected step-69000 checkpoint;
- vocabulary: the checkpoint vocabulary, its 401 tokens and the `featmath-jain-v1` product
  restriction with its base;
- the context ABI: 2048 little-endian float32 values, 8192 bytes, clip 32, 1536/512 dataset and
  target dimensions, blocks 512 + 1024 + 512;
- the encoder closure: Hyperion weights, manifest and seed, the component bundle manifest and
  identity, Foundation weights and seed, Target weights and window seed, frozen transforms,
  assembly, recipe and label normalizations;
- the native qualification: registry, index, the `3e-4` bound and the verification policy.

**Receivers.** `semantic::accepted()` opens the embedded bytes and refuses any other identity
(`open_accepted`), an unknown field or schema version, a non-canonical encoding or an internally
inconsistent document. The accepted receivers now also compare with the manifest, after their
existing checks so their refusal wording is unchanged:
- `require_accepted_policy` (loaded weights);
- `require_accepted_vocabulary`;
- `CheckpointBinding::require_accepted_identity` (export receipt);
- `EncoderIdentity::require_accepted` (every CPU context artifact, qualification and component
  qualification).

`producer::Request::validate` and `jope-policy-producer` take the weights, frozen transforms and
assembly from the manifest instead of literals.

**Tests.**
- `semantic::tests::every_compiled_identity_is_the_manifests` checks every reviewed constant
  against the manifest. It also checks the committed `featmath.json` digest and token count, the
  `PolicyConfig::regression(401)` hash and dimensions, `expected_inventory`'s 136 tensors, and the
  recorded tensor inventory (`docs/internal/core-ip/data/tensors.json`: weights digest, tensor
  and parameter totals).
- `semantic::tests::the_encoder_release_identity_is_the_manifests` (`context-producer`):
  `EncoderIdentity::accepted()`, the transforms, assembly and component-bundle constants and the
  admission constants.
- `semantic::tests::only_the_accepted_manifest_bytes_are_the_release_decision`:
  - positive: the embedded manifest re-encodes to its exact bytes;
  - negative: a single substituted dependency (Foundation weights) is structurally valid but not
    the release decision; an unknown field, a v2 schema, a pretty-printed encoding, a rejected
    state named as accepted and an inconsistent ABI are refused.
- jope-pipeline `search_context::tests::duplicated_lineage_constants_are_the_accepted_semantic_manifests`.

**Runs.**
- Post `w2c-t02-post-20260926T201854Z`: jope-policy (`testing,context-producer`) 94/94 and
  jope-pipeline (`neural-search`) 119/119. The first attempt,
  `w2c-t02-post-20260926T201749Z`, stopped at a compile error (a `&String` comparison), which was
  fixed before this run.
- Pre-commit `w2c-t02-c-20260926T202015Z`: fence suites and `gp::fit_contract_tests` 82/82; workspace all-targets check, coverage (0 unassigned), verify-test-inventory, fmt and diff-check exit 0.

**Not done.**
- G-08 (unknown export-receipt fields) stays open: the external exporter's field set is still
  not available.
- The published `models/bundles/policy-step-60000.json` (jope-models fetch, `jope.model-bundle/v2`)
  is unchanged; the semantic manifest is a separate release record and names the weights digest
  it lists.
- The jope-bench literal (`optimize.rs:259-262`) and jope-engine `policy_export.rs` private copies
  are not changed; jope-engine does not depend on jope-policy.
