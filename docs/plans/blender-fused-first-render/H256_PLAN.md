# H256: actual five-fold inference → selected GP → complete working Rust

2026-11-03 · addendum to [MASTER_PLAN.md](MASTER_PLAN.md) · branch `codex/blender-fused-first-render` · draft PR [#2114](https://github.com/neverhuman/JopeDime/pull/2114). PR #2071 / `v1/fastlane` remains protected. H256 implementation follows base `8e37d6e7845a01fb4c53a71789039aeafdbaba1b`. [H256_RENDER_RECEIPT.json](H256_RENDER_RECEIPT.json) binds exact source-file/artifact hashes; external `h256/final-git-handoff.json` records the final committed HEAD and PR readback after packaging.

**Current state:** genuine saved Stage 2 inference, the optimized six-case native GP/source controller and enriched semantic validation **PASS**. Every new Rust/Python fitted prediction and emitted Rust training prediction matches native exactly across all 96 rows in all six cases. All 578 complete-source pages were produced. Three 4K H256 stills and the saved v2 scene are complete; nonsequential reopen reproduces decoded pixels exactly. Final source-only Git/PR packaging is recorded in the external handoff; hosted CI is **NOT_RUN**. Video remains behind user design approval.

## 1. Objective, dependencies and completion

Show one continuous angled 3D JAIN network taking actual tabular data through frozen DOPE, the genuine legacy base and trained H256 bypass, to actual G program selection and a complete native-generated Rust algorithm. Every weighted connection references a saved weight; changing strength comes from actual inference operands or contributions. Exact lowercase on-screen labels are **`problem encoder`** for all DOPE weights and **`solution encoder`** for all other neural weights. Repository/filesystem names remain JopeDime.

This is a bounded **local research route**: G selects from twelve actual programs in the recorded native admission slate, then the native DSL fitter/exporter fits the selected program on the exact 76 FIT rows and proves source on all 96 rows, including 20 held-out rows. It does not change production ranker admission, establish a registered Engine champion, replay an entire G-driven production search, or demonstrate improved model quality.

Completion requires each independent gate:

1. Saved five-fold model/source/input identity and genuine observed graph, without dropped records or parameter mutation.
2. Strict full-checkpoint/numeric validation and G selection bound to the program subsequently fitted.
3. **Release-native** fit → complete unchanged Rust/Python export → both 64-row self-checks → all-row prediction parity → emitted Rust `Model::fit` training parity on the same FIT rows.
4. Every Rust byte rendered top to bottom and reconstructable from code-page manifests, including training, fitted state, helpers and self-checks.
5. Revised 4K stills, saved `.blend` binding receipt and nonsequential reopen/replay checks. User design review precedes video.

Dependencies: read-only checkpoint/source custody; exact native FIT descriptors/program features; full saved critic and authorized golden fixture/proof; CPU PyTorch; the exact Rust compiler and release helper; Pillow/code-page font; pinned Blender build. Missing dependencies stop the dependent milestone, never become invented operands or outputs.

Non-goals: training edits/restarts/stops, injected checkpoint export/memory attachment, new campaigns/GPU/server launch, policy/signature bypass, serving promotion, video before design approval, or publication of private checkpoints/raw tensors/temporary GP debug. Neural decoding token-rate fields remain null. The displayed lexical rate is measured optimized-release warm repeated emission of the exact already-fitted Rust file, including allocation/loop overhead; it is not neural decoding or unique-code/end-to-end throughput. No new H256 quality, throughput or neural decoding claim follows from the preview.

## 2. Workspace, ordered reads and ownership

Worktree: `/Users/bentaylor/Documents/ChatGPT/JopeDime/blender-source/.agent/worktrees/blender-fused-first-render`.

External artifact root: `/Users/bentaylor/Documents/ChatGPT/JopeDime/blender-render-20261103/h256/`.

Read in order:

1. `AGENTS.md`, latest locked H256 claims/handoffs in `AGENT_CHAT.md`, root `MASTER_PLAN.md`, this addendum and [REVIEW_RECEIPT.json](REVIEW_RECEIPT.json). Earlier r882 proofs remain historical evidence for that model.
2. External `h256/custody.receipt.json`; `h256/stage2-custody-20261103T1911Z/{base-source-custody.receipt.json,critic-custody.receipt.json,golden-custody.receipt.json}`; then `h256/capture-stage2-six-final/{trace.json,capture.receipt.json,validation.receipt.json,validation-controls.receipt.json}`. Final capture asset hashes control; preliminary `stage2-custody/` and captures are superseded, not interchangeable.
3. In final capture `source/`, read `stage2_pre_condense_models.py`: `FoldScaler.transform_context`, `PreCondenseResidualBridge.encode_context_tokens`, `forward_prior_and_context`, `forward_causal_trajectory_residual`; then `stage2_run_h256_flywheel_ablation.py`: `evaluate_context_with_model` and checkpoint writer. Continue with `stage2_phase12_multimodal_model.py`, `stage2_flywheel_model.py`, `stage2_stage1_hurdle_sos_critic.py`, `stage2_common_h252.py`, `cpu_exact_fused.py` for base/query/critic/legacy projection. Research source was untracked at custody time; file hashes, not remote HEAD alone, identify it.
4. `tools/visualization/{h256_inference.py,h256_stage2_inputs.py,h256_stage2_inference.py}` for source-preserving CPU capture and bound critic inputs; then `{h256_validate.py,h256_stage2_validate.py,h256_functional.py}` for admission and captured functional projections.
5. `crates/jope-service/examples/visual_fit_dsl.rs`; `crates/jope-runtime/src/{math_pipeline.rs,model_object.rs}`; `crates/jope-service/src/discover/{finalize.rs,loop.rs}`; `crates/feat-math/src/model.rs`. Read the release refusal and unchanged-prefix training proof before invoking the helper.
6. `tools/visualization/{h256_gp_source.py,h256_gp_validate.py,code_pages.py}`, then `tools/visualization/blender/{h256_network.py,h256_scene.py}`. Actual current CLI supersedes the earlier Stage 1 draft flags.

| Packet / owner | Exact scope | Boundary |
|---|---|---|
| Algorithm owner | `h256_inference.py`, `h256_stage2_inputs.py`, `h256_stage2_inference.py` under `tools/visualization/` | Remote training/custody remains read-only |
| Native fit handed to root | `crates/jope-service/examples/visual_fit_dsl.rs`; root owns Cargo registration | No production ranker/HTTP discovery/admission edits |
| Root admission/controller | `h256_stage2_validate.py`, `h256_gp_source.py`, `h256_gp_validate.py`, integration/tests | Shared helpers need explicit ownership transfer before edits |
| Film owner | `blender/h256_network.py`, `blender/h256_scene.py`; functional helper per root claim | Preserve earlier scenes/evidence |
| Documentation owner | This `H256_PLAN.md` only | Other plan/status/readme/index/receipts remain root-owned |
| Root integration | Claims, index, Git/PR, exact-head freeze/checks | Protect #2071 and other active scopes |

Follow the latest exact-file claim/transfer. Only disjoint packets run concurrently. Subagents do not stage/commit/reset/rebase or append `AGENT_CHAT.md`; root serializes claims/integration. Models, exports, textures, traces, PNGs and `.blend` files remain outside Git. Use one reused Cargo target/build lane, never `/tmp`; retain failed/provisional receipts.

## 3. Current identities and observed gates

| Final artifact | Identity / evidence |
|---|---|
| Five-fold ensemble | `capture-stage2-six-final/assets/stage2-ensemble.pt`; 4,700,291 bytes; SHA `ee40e191e4ba8f2b0a3e0a34a44746984ea2aebafc277c588f71f73f7de323c8` |
| Full critic | `capture-stage2-six-final/assets/critic.pt`; SHA `ba5678e1097550d54ec5339cf42b341339af7d65e566cd3a2bcf28f16c596755` |
| Authorized golden fixture | `capture-stage2-six-final/assets/golden.pt`; SHA `de8f0ae10b0c38413cfd9171e6d627aff97764b38ffca8d9ccda48fdefe5a0bb` |
| Golden proof | `h256/stage2-golden-proof.receipt.json`, copied as `assets/golden-proof.receipt.json`; SHA `89b2691c43019acca10773a50c5f54af5231265ad0d5fcbdf6c2d2afeaa484d1`; **PASS**, exact pick **678**, exact reveals/top-four; separate real trajectory smoke |
| Frozen DOPE raw checkpoint | `assets/round-882.pt`; SHA `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64` |
| Frozen DOPE logical composite | `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`; unchanged before/after; different hash domain from file bytes |
| Final raw NN capture | `h256/capture-stage2-six-final/trace.json`; 234,686,862 bytes; SHA `9825d37e18786ca807c4a62e6ea7bf9c07c70e635ad26d77409d5ac1aadd6609` |
| Actual capture source | `h256_stage2_inference.py`; SHA `4c59ba65047b400067e6c6f1e5ea83a206acb97ab1747de83ea180c47b753f40` |
| Strict NN gate | `capture-stage2-six-final/validation.receipt.json`; all **590 parameters**, **38 saved buffer roots/case**, **30 saved scaler buffers**, native inputs/DAG/readouts/closure/critic/selection/golden **PASS** |
| Numeric coverage | **120 `nn.Linear` affines + 3 captured functional MHA projections = 123 matrix operations**, 1,014 operations and 1,102 tensors per case; maximum normalized Linear closure `8.714523648299934e-7`. Functional views preserve the same 590-parameter inventory |
| Native reference | Fresh original PyTorch query versus native backend maximum absolute difference `1.1324882507324219e-6`; exact raw412/proj16 roots and query→G operands separately bound. This backend comparison is not bitwise equality |
| Controls | `validation-controls.receipt.json`: changed readout, foreign parent, changed saved affine weight, changed critic cell and wrong selection rejected; receipt-backed six-case validation also passed with PyTorch imports actively prohibited |
| Release source controller | `h256/gp-source-six-release/controller.receipt.json`: **PASS**, raw NN SHA unchanged; enriched trace SHA `8847c6da62422c0757f845acf3a6b856090c4e97034c97223a76f2a29dd49b61` |
| Enriched semantic admission | `h256/gp-source-six-release/validation.receipt.json`: **PASS**, six per-case GP causal/source bindings **PASS**, bound to the enriched trace above |
| Complete source pages | `h256/full-rust-six-release/`: A/B/C/D/E/F = **219/60/90/90/84/35**, total **578** pages; saved complete inspection timeline **3,720 frames at 12 fps = 310 seconds**. This is replay dwell, not inference duration |
| Current render input | `h256/gp-source-six-release/trace-with-metrics.json`; SHA `22f3940018537ee6b5427529fb1014441b79a4c6879e88c7554ab3d39511af26`; `validation-with-metrics.receipt.json` **PASS**, including exact new-source metric bindings |
| Final scene and stills | `h256/review-fullcode-v2/scene.blend`; SHA `22afbf66fbd33372fcdc0f64d6dce271f09f6c636ea216477f7b98ce59e28ef7`; frames **37 / 2629 / 3583** at **3840×2160**, actual first-page / training / prediction cues |
| Saved-scene replay | `h256/reopen-fullcode-v2/receipt.json`: **PASS**, frames **3583 then 2629** reproduce original decoded RGBA exactly |

All six source helper receipts record both 64-row self-checks, 96-row Rust/Python fitted prediction and emitted Rust `Model::fit` prediction proof. Every reported maximum error is **exactly 0.0**; the unchanged tolerance remains `1e-12 + 1e-10*abs(native)`.

| Case | G selected slate index | New Rust SHA-256 |
|---|---|---|
| A | 9 | `998660b264645c6d6b46b72ca6026ccc8ccd4e7552a0fff2ee4041002747bbf8` |
| B | 4 | `6a657cdc219fc73e09afea4fe9cdb5af48e862dfce3ffac53c6bfda3ead7b712` |
| C | 7 | `659d316d98972f7ac7cc582f7bf37397478fcc666c3bcf8700faaf6572fa8931` |
| D | 7 | `161ca7ebf502b44e0e3336a277eb8324bad101eea66064387d43976024713604` |
| E | 11 | `7c8d6c769a57b5df955756883694e44f9864941261d88081fa89924a99839ea9` |
| F | 4 | `8725be24c51fa7e9586e0655310b1e5bfdd6973aa6f51b92a1ca90a17bd46e8e` |

The 590 parameters comprise 56 DOPE, 114 base-adapter and five complete G bridges, each with 84 tensors / 183,668 scalar parameters. Unique parameters, executed parameters, invocation counts and displayed scalar connections are distinct inventories. Saved trajectory weights exist; the displayed six cases execute **K_rev=0**, without fabricated reveal history.

Preserve Stage 1 history: the earlier 95,540-byte `stage1_encoder.pt`, SHA `7e5257741409575b237f0312ee8a7c5f6e1fc48d831056b6166ceea455528a44`, has 39 tensors / 15,005 parameters and lacks learned four-query pooling/full downstream heads. That initial snapshot could not support complete Stage 2 inference. The subsequently supplied genuine ensemble/critic/golden fixture supersede that blocker. No initialized fallback or old attempt001 weights are substituted. The attached report remains attributed research history, not an instruction to modify training or an independently verified quality result.

## 4. Actual computation and visual truth

The captured network has a parallel legacy base and a new G residual path:

```text
exact table/FIT descriptors → frozen DOPE (problem encoder)
  → fresh descriptor projection lm48 → original Stage1B 10-token base
  → Full629:20×32 + Hidden256:8×32 + columns:64×18→64×32
      → 92×32 context, 92-mask, 11D log norms→8D side channel
actual canonical native raw412/proj16 → base candidate-query encoder
  → five saved G candidate-conditioned attention/prior paths (solution encoder)
  → mean of five predictions → final K0 cost → minimum eligible proposal
  → native fitted GP → complete Rust
```

The twelve programs come from the actual recorded canonical native admission slate. Only indices **≥4** are eligible; stable first-index ties apply. The first four entries are references, while the present G invocation consumes no revealed outcomes/history. Gate/SP arrays are full-critic predictions bound to actual program feature artifact bytes. Complete selection cost is:

`fresh R882 K0 cost + 0.005 × saved-hurdle h_cost + alpha_prior × mean(five G delta_prior)`

Saved `alpha_prior = 0.032999999999999995`. Five FP32 fold predictions are promoted to FP64 for their mean and final addition; parameter-soup weights are unused. Saved trajectory coefficient is `0.01`, but no trajectory invocation is manufactured for these K0 cases. The separate golden smoke genuinely executes the full trajectory and checks exact pick/reveals; it does not select programs for the six displayed synthetic tables. No outcome reroll is allowed.

Explicit landmark paths:

1. `dope629[0:256]` → eight descriptor tokens in Full629.
2. Actual `64×17` column landmarks + actual `dope629[565:629]` gates → `64×18` → token slots 28..91.
3. Actual predicted landmark channels `dope629[526:549]` → semantic head token inputs.
4. `fresh_descriptor_projection_lm48`, `fresh_lm48_z`, `base_tokens_10x32` → the captured parallel legacy base. Use these actual readout names when deciding whether the fourth path is present.

Column-18 construction, saved canonical full-train scaler, `[-15,15]` clipping and actual feature mask must reproduce source. The eighteenth channel is a real gate. Full629 slices are 256+128+128+(53+11)+64; no pooled vector is inferred from the H256 name.

Scene rules:

- One connected angled 3D volume; fixed left table and readable right code. Section colors distinguish encoder roles; show real branches, functional routes and parallel reconvergence.
- Each weighted edge has a checkpoint coordinate/input; static width shows weight magnitude and dynamic light shows activation or `x*w`, with checked bias/sum closure. Weights remain fixed across datasets.
- Functional links are typed dependencies. Derive fused base MHA Q/K/V only from captured operations and saved joint-parameter row views through `h256_functional.py`; do not omit paths that bypass an `nn.Linear` hook.
- All-weight mode uses actual nonzero matrix entries/biases under the explicit 1,000,000-connection budget (850,956 actual connections in this saved scene). No silent truncation. Shared token/row weights use receipted signed maximum-absolute contribution aggregation with selected row indices, not a claim of simultaneous invocation.
- Every displayed weighted input/output node has an actual incident edge. Functional/mask/padding nodes retain meaning; zero contribution gets zero activity glow; calibration is fixed across cases.
- Replay timing, pulse travel, placement and code dwell are editorial; label recorded/time-expanded inference. Blender speed is not inference speed.
- Full opaque 21px neutral syntax pages retain every Rust byte at 48 columns / 24 rows. Verify offsets/order/page hashes/reconstruction; do not rejoin tokens or replace the file with an excerpt. Source inspection may exceed 90 seconds; final story length remains a design decision.

## 5. Ordered implementation packets

### H0 — Preserve and freeze

Root refreshes PR/Git/claims, protects #2071, pins source file hashes and later full commit, and preserves old failures. Algorithm owner uses read-only stable bytes, restricted `weights_only=True` CPU loading with explicit safe NumPy globals, exact fold keys/shapes and saved scalers. Stop on unknown origins, mutable checkpoints or initialized used weights. No remote training changes.

### H1 — Full Stage 2 inference/admission (passed)

Run original frozen DOPE, fresh legacy base K0 and five trained G prior forwards on six exact FIT descriptor/native-feature inputs. Capture module and functional producer/consumer IDs, verify observer on/off outputs and unchanged parameters. Gate all 590 saved parameter bytes, root buffers, 30 scaler hashes/arithmetic, native roots, critic bindings, actual prior input cells, fivefold arithmetic and selection. Schema is `jain.h256-stage2-capture/v1`.

API: `h256_stage2_validate.validate(path, saved_receipt=None, gp_root=None) -> (data, report)`. Fresh mode reads actual checkpoints. Blender receipt mode imports no PyTorch, requires exact trace/verifier/shared-helper/parameter/scaler/source identities, and reruns graph/numeric guards. Missing receipt never silently skips checkpoint verification. Actual PyTorch `NonDynamicallyQuantizableLinear` is treated as its Linear subclass without changing identities/operands.

### H2 — Native selected-DSL fit and complete source (release controller passed)

`h256_gp_source.py` validates the NN capture, takes the actual minimum `final_k0_cost` proposal and preserves dataset/FIT identity. It invokes `visual_fit_dsl` with exactly five paths: selected DSL, CSV, original same-case input-contract model, split proof and new output directory. The helper reuses only the validated prediction-v2 contract, recomputes FIT-only min/max/target scale, preserves selected-column/FIT order, rejects missing/nonfinite cells, calls `feat_math::model::fit(GenomeIr::parse(...), ...)`, creates a new content-addressed math-pipeline `ModelObject`, and uses native `export_source` conversion for Rust/Python.

**Use `--release`.** Development-native versus optimized emitted Rust training initially failed case D because of compiler trigonometric coalescing. The converter was unchanged. A matching release-native versus `rustc -O` diagnostic produced zero error on all 96 D predictions. Root added debug-build refusal and uses the native build's Rust compiler for source proofs. The subsequent complete six-case release controller passed with every fitted/trained prediction error exactly zero. Retain the failed evidence and original `1e-12 + 1e-10*abs(native)` tolerance; do not alter the selected DSL, loosen tolerance or reroll.

Preserve full exported bytes; execute both 64-row self-checks and Rust/Python predictions on all 96 rows; append only a proof driver after the unchanged Rust prefix; execute its `Model::fit` on the 76 FIT rows and compare all 96 trained predictions. Receipts bind compiler/profile, model/DSL/CSV/input/FIT/source hashes, command exits/stdout and vectors. `h256_gp_validate.validate_gp_sources` independently reopens proofs and returns per-case causal bindings. The enriched trace retains all raw neural data and adds only source bindings/coverage; raw NN capture stays immutable.

### H3 — Complete pages/semantic admission, 3D stills and saved-scene replay (passed)

Controller rasterized the entire Rust files via `code_pages.make_pages` and wrote the source-bearing trace/controller receipt. The enriched semantic gate passed all six causal/source bindings. The frozen scene loads verified complete-source manifests and builds the real network, all four measured landmark grids and actual fivefold attention/prior. Frames 37, 2629 and 3583 are rendered; reopening the saved scene at 3583 then 2629 passed decoded-image equality. Root has sent revised stills for design feedback. No animation before approval.

## 6. Commands, limits and expected artifacts

Run from the worktree; prefix shell commands with `rtk`. Root owns the single native build lane; other agents do not launch competing Cargo builds.

```sh
rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only CARGO_TARGET_DIR=/Users/bentaylor/Documents/ChatGPT/JopeDime/blender-source/.agent/worktrees/blender-fused-first-render/target RUSTC=/Users/bentaylor/.rustup/toolchains/1.97.1-aarch64-apple-darwin/bin/rustc /Users/bentaylor/.rustup/toolchains/1.97.1-aarch64-apple-darwin/bin/cargo build --locked --release -j4 -p jope-service --features visual-trace --example visual_fit_dsl
rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only CARGO_TARGET_DIR=/Users/bentaylor/Documents/ChatGPT/JopeDime/blender-source/.agent/worktrees/blender-fused-first-render/target RUSTC=/Users/bentaylor/.rustup/toolchains/1.97.1-aarch64-apple-darwin/bin/rustc /Users/bentaylor/.rustup/toolchains/1.97.1-aarch64-apple-darwin/bin/cargo test --locked -j4 -p jope-service --features visual-trace --example visual_fit_dsl
rtk proxy env OPENBLAS_NUM_THREADS=1 python3 tools/visualization/h256_stage2_validate.py /Users/bentaylor/Documents/ChatGPT/JopeDime/blender-render-20261103/h256/capture-stage2-six-final/trace.json --receipt /ABS/NEW/neural-validation.receipt.json
```

Final source checks pass: 41 portable Python controls, two native-helper controls, release helper build, Rust formatting and capability-matrix inventory. AI peers reviewed source and actual six-case receipts; this is not native human approval. Exact-head hash verification/PR readback is recorded in the external handoff. Hosted CI is NOT_RUN. Portable tests use synthetic controls, not private checkpoints.

Commands for the completed release controller and enriched semantic admission; existing outputs must not be overwritten. Reproduction uses new external output names:

```sh
rtk proxy env OPENBLAS_NUM_THREADS=1 python3 tools/visualization/h256_gp_source.py --trace /Users/bentaylor/Documents/ChatGPT/JopeDime/blender-render-20261103/h256/capture-stage2-six-final/trace.json --native-proof /Users/bentaylor/Documents/ChatGPT/JopeDime/blender-render-20261103/single-run-final --fit-binary /Users/bentaylor/Documents/ChatGPT/JopeDime/blender-source/.agent/worktrees/blender-fused-first-render/target/release/examples/visual_fit_dsl --output-dir /Users/bentaylor/Documents/ChatGPT/JopeDime/blender-render-20261103/h256/gp-source-six-release --code-pages-dir /Users/bentaylor/Documents/ChatGPT/JopeDime/blender-render-20261103/h256/full-rust-six-release
rtk proxy env OPENBLAS_NUM_THREADS=1 python3 tools/visualization/h256_stage2_validate.py /Users/bentaylor/Documents/ChatGPT/JopeDime/blender-render-20261103/h256/gp-source-six-release/trace.json --gp-root /Users/bentaylor/Documents/ChatGPT/JopeDime/blender-render-20261103/h256/gp-source-six-release --receipt /Users/bentaylor/Documents/ChatGPT/JopeDime/blender-render-20261103/h256/gp-source-six-release/validation.receipt.json
```

Use Cargo's actual reused-target executable if different; never infer success from an older binary. Controller bounds each selected-case subprocess to 180 seconds and stops on failure. Helper bounds CSV to 8 MiB, DSL to 64 KiB, rows to 2..4096, features to 256, model to 256 MiB, source to 16 MiB, and refuses output reuse. Old split JSON lacks a CSV digest, so adjacent original `proof/summary.json` must remain accessible, or split must carry `input_sha256`.

Completed final code-bearing scene command (use a new output directory for reproduction):

```sh
rtk proxy /Users/bentaylor/Documents/ChatGPT/JopeDime/blender-runtime/Blender.app/Contents/MacOS/Blender --background --factory-startup --python tools/visualization/blender/h256_scene.py -- --trace /Users/bentaylor/Documents/ChatGPT/JopeDime/blender-render-20261103/h256/gp-source-six-release/trace-with-metrics.json --saved-validation /Users/bentaylor/Documents/ChatGPT/JopeDime/blender-render-20261103/h256/gp-source-six-release/validation-with-metrics.receipt.json --gp-proof /Users/bentaylor/Documents/ChatGPT/JopeDime/blender-render-20261103/h256/gp-source-six-release --code-pages /Users/bentaylor/Documents/ChatGPT/JopeDime/blender-render-20261103/h256/full-rust-six-release --output /Users/bentaylor/Documents/ChatGPT/JopeDime/blender-render-20261103/h256/review-fullcode-v2/scene.blend --width 3840 --height 2160 --all-weights --max-connections 1000000 --show-gp-tree --render-frame 37 --render-frame 2629 --render-frame 3583
```

Each case expands for complete source dwell. Use the scene receipt's timeline for later-case/end-page stills, not an assumed fixed 72-frame total. Expected final artifacts: controller/enriched validation receipts, six model/source/proof directories, six complete-source manifests/textures, scene/binding/coverage receipt, selected PNG stills and saved-scene reopen receipt. Missing artifacts remain pending.

## 7. Stops, receipts and handoff

Stop the dependent gate on wrong model/source/critic/scaler identity; unbound operands; numeric/producer mismatch; dropped records; invented history; changed selected DSL; non-release native proof/compiler mismatch; conversion/training parity failure; incomplete source pages; silent connection truncation; missing landmark path; wrong labels; output reuse; or unproved code/rate attribution. Preserve valid NN/failure evidence while fixing the cause. Never change training or production admission to satisfy a render gate.

Receipts bind source commit/file hashes, command/cwd, compiler/runtime/profile, input/FIT/model/DSL/source/output identities, exits, numerical tolerances/errors, coverage, failure disposition and pending claims. Root owns claims/index/PR updates. Workers stop exact scopes before integration.

Final handoff names exact source commit/dirty state; raw/enriched trace identities; release binary/toolchain; six model/source/working-training proofs; complete-source page coverage; four landmark paths and connection coverage; stills/reopen evidence; exact-head/hosted-CI and PR disposition; user design/video disposition; correctly null/scoped performance fields; and protected PR state. Earlier r882 and Stage 1 evidence remains immutable history. Current disposition: **NN, release six-case source/export, enriched semantic admission, complete-source scene/stills and nonsequential saved-scene reopen passed**. User design feedback precedes video. Source-only commit/push/exact-head readback is recorded in external `h256/final-git-handoff.json`; hosted CI remains NOT_RUN, with no merge/release claim.
