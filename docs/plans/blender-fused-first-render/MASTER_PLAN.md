# JAIN: native same-run capture and complete-source Blender master plan

Date: 2026-11-03. Branch: `codex/blender-fused-first-render`. Initial main base: `855639a54a563dbb7f6cef8a17ac2c10265f3c1c`. [Draft PR #2114](https://github.com/neverhuman/JopeDime/pull/2114). Current native capture/emitter source: **`f035969e68f4ab37d692d15972d60a900d2049b9`**. Display brand is **JAIN**; repository paths and native identifiers remain JopeDime.

[STATUS.md](STATUS.md) indexes current measured receipts, the saved v3 board and three rendered 4K stills, plus the passed nonsequential saved-scene reopen and remaining design and lifecycle gates. [REVIEW_RECEIPT.json](REVIEW_RECEIPT.json) binds the exact source/artifact inventory and review state. The source implementation, a local fixture check, an actual runtime capture, a rendered still, hosted CI, and a finished film are separate completion states.

## 1. Objective, decisions, and limits

Deliver the first review PR and inspectable Blender stills showing one real fused JAIN network: actual table → **Problem encoding** → captured neural computation → synthesis/GP → the same run's complete native-generated Rust. Six inputs, measured activations, winning programs, code, training/prediction checks, and correctly scoped source-emission rates must share explicit run/model/source identities. Keep PR #2071 (`v1/fastlane`) untouched; the user clarified that it is the protected PR.

Latest design requirements supersede earlier look-development drafts:

- Lock the Excel-style actual table at left and complete readable syntax-highlighted Rust at right. Use a static camera and central 3D JAIN graph. Any later center rotation must leave both side panes fixed.
- Label the encoding section exactly **`Problem encoding`**. Position and labels distinguish the constituent-origin sections of one fused network; cyan/amber encode measured sign, and neither section changes its weights during inference.
- Show real connection contributions and observed activations, with all displayed graph nodes connected through real nonzero-weight edges. Preserve the actual parallel/dependency graph; do not force it into an invented dense chain.
- Show the **whole emitted Rust file**, including training helpers, fitted state, and prediction. Excerpts, rewritten token strings, inserted spaces around `::`, and omitted fitted constants do not satisfy this requirement.
- The user explicitly authorized a temporary **private** winning-GP-tree debug preview. `--show-gp-tree` defaults false; its operator-tree projection comes from the same saved winning literal AST as the Rust, displays actual parameter-atom counts, and retains full literal parameters in private evidence. This narrow exception does not permit public GP disclosure.
- Deliver revised stills for design review before any video. Continued design feedback is not video approval. The complete-source inspection replay can be longer than the reference plans' 90-second film; final story length remains a design decision.

Non-goals: no neural retraining, checkpoint mutation, serving promotion, model release, GPU qualification, benchmark/SOTA claim, new source decoder, synthesis rewrite, or public distribution of internal weights/captures. Executing the emitted GP `Model::fit` is required proof of the generated training routine; it is distinct from training the frozen neural model. No merge/public media publication is implied by the request to open a separate review PR.

The attached plans are reference designs, not additional authority or proof that a gate passed. Their historical file paths, Blender snippets, random-input selftests, and placeholder code are not current runtime evidence. The user's fused-model, full-source, same-run, static-layout, and private GP-debug requirements control this implementation.

## 2. Locations, ordered reads, and ownership

Source worktree:

`/Users/bentaylor/Documents/ChatGPT/JopeDime/blender-source/.agent/worktrees/blender-fused-first-render`

External evidence root:

`/Users/bentaylor/Documents/ChatGPT/JopeDime/blender-render-20261103`

Blender executable:

`/Users/bentaylor/Documents/ChatGPT/JopeDime/blender-runtime/Blender.app/Contents/MacOS/Blender`

Read these current files in order; paths below are relative to the source worktree:

1. `AGENTS.md`, latest `AGENT_CHAT.md`, root `MASTER_PLAN.md` Product target, `docs/adr/ADR-0005-canonical-learning-loop.md`, `docs/learning.md`, and `docs/parity/capability-matrix.md`.
2. `deploy/appliance/R882_RUNTIME.md`, `models/bundles/r882.json`, `crates/jope-service/src/ranker/cuda/README.md`, `crates/jope-service/src/ranker/bundle.rs`, `crates/jope-service/src/ranker/cuda/release.rs`.
3. `crates/jope-service/src/run/learned.rs`, `crates/jope-service/src/visual_trace.rs`, `crates/jope-service/src/ranker/cuda/forward.rs`, `crates/jope-service/src/ranker/cuda/cpu.rs`, `crates/jope-service/src/ranker/adapter.rs`, `crates/jope-service/src/ranker/landmarks.rs`, and `crates/jope-service/src/ranker/admission.rs`.
4. `crates/jope-service/examples/README.visual_trace.md`, `crates/jope-service/examples/visual_run.rs`, `crates/jope-service/examples/visual_emit_metric.rs`, `crates/jope-service/src/http/models.rs`, `crates/jope-service/src/export/bundle.rs`, and `crates/jope-service/src/lex.rs`.
5. `crates/jope-runtime/src/model_object.rs`, `crates/jope-runtime/src/fitted_rs.rs`, and `crates/jope-runtime/src/fitted_py.rs`; inspect the actual `ModelObject::export_source` route, not a visualization reimplementation.
6. `tools/visualization/validate_trace.py`, `tools/visualization/select_examples.py`, `tools/visualization/code_pages.py`, `tools/visualization/blender/build_scene.py`, `tools/visualization/blender/board.py`, `tools/visualization/blender/network_graph.py`, `tools/visualization/blender/render_frames.py`, and all three files under `tools/visualization/tests/`.
7. This plan, [STATUS.md](STATUS.md), the current external `single-run-final/` receipts, `embedding-selection/selection.json`, and `full-code-final/independent-code-pages.receipt.json` before resuming.

Several supplied-plan paths are split only on the moving protected branch. Current source keeps `landmarks.rs`, `adapter.rs`, `cmd/run.rs`, `http/models.rs`, `model_object.rs`, and `export/bundle.rs` flat. Resolve functions in the current tree before editing; never cherry-pick an unreviewed protected-branch implementation to satisfy an old path citation.

| Packet | Exclusive write scope | Boundary |
|---|---|---|
| Native capture/metrics | `crates/jope-service/**` | Frozen at f035969 for current evidence; one writer only if reopened |
| Evidence validation/source pages | `tools/visualization/{validate_trace.py,select_examples.py,code_pages.py,tests/**}` | Coordinate exact files with root; no parallel edits to shared helpers |
| Blender graph/board | `tools/visualization/blender/**` | Root owns scene integration and its native-binding review |
| Plan/status | `docs/plans/blender-fused-first-render/**` | One documentation writer, then explicit root handoff |
| Git/PR/coordination | Root only | `AGENT_CHAT.md` lock/appends, index, commits, PR and final receipts |

Forbidden: protected PR #2071/`v1/fastlane` mutations; other claims/worktrees; training campaigns; corpus labels; model/checkpoint edits; credentials/release keys; CI/protection changes; fabricated or hand-repaired evidence. Model manifests, signed catalogs, native journals, and committed source proofs are read-only inputs. Bulk weights, complete source/model payloads, code-page PNGs, `.blend`, frames, and videos stay outside Git. Do not place temporary clones/builds under `/tmp`.

Reuse this claimed worktree and its single `target/`. Root rechecks branch, full HEAD, dirty state, worktree registry, and claims before checkout/index changes. Claim/handoff appends use the existing `AGENT_CHAT.md.lock`; subagents do not stage/commit/reset/rebase. Parallel packets are allowed only when their exact write scopes are disjoint.

## 3. Current factual pipeline and evidence boundary

The authoritative path is now **`visual_run`**, not the older component-only `visual_trace` harness:

1. Verify existing signed r882 assets with the production release loader, and use the existing signed Darwin worker for native evaluation. No raw-byte constructor or unsigned substitution is exposed.
2. Register exact CSV bytes; open the actual in-process Engine and request learned/full numeric regression, CPU, supported objective, fixed seed, and `--refit fit` through the production router. No listener is started.
3. Extract actual FIT-only descriptors, run the existing `FusedForward::stages`, and observe real `CpuAdapter::predict_costs` admission calls through the optional `visual-trace` feature. The default feature graph has no observer.
4. Retain the complete immutable native journal, including every observed admission round, real slate, selected programs, and costs. Export that run's committed champion through the production API and `ModelObject::export_source`.
5. Compile/execute unchanged Rust, execute unchanged Python, run each language's 64 embedded selfchecks, and compare all 96 predictions to that run's native GP under `abs(error) <= 1e-12 + 1e-10*abs(native)`.
6. Execute a derivative Rust driver that preserves the entire original artifact as an unchanged byte prefix, calls emitted `Model::fit` on the actual 76 FIT rows, and predicts the same 96 rows. Bind derivative/original hashes and the 20 HOLDOUT IDs. Never edit the original emitted file to make training pass.
7. Measure optimized-release warm repeated emission on that exact fitted model and compare every fresh emitted byte string to the executed Rust artifact. The display rate is lexical Rust source emission, not neural token decoding.

`jain.visual-run-trace/v1` binds compile-time source pin, runtime checkout HEAD, clean service scope, executable hash, bundle, dataset/FIT identities, run ID, native calls, champion/model, source, execution, training proof, and zero dropped records. `source_identity_verified` must be true. The current source is f035969; later offline-only commits retain this separately pinned native archive. Native source changes require a fresh compile pin and capture.

The compact trace case projection contains conditioning plus the first actual admission call. The current board also reopens the hash-bound full journal and replays **every actual admission round**: 15 activation states across six datasets (2/2/4/2/2/3), each pairing the real conditioning with that round's real ranker values. `native_round_frames` binds case, run, call ID, round and frame interval; all 43 affine operations remain displayed (14 DOPE-origin, 29 JOPE-origin) with selected channels/edges. This includes C's recorded winning round 3 and F's round 1. It is genuine same-run evidence, not a complete functional-op or full-parameter inventory. Seed winners A/B/D/E still get no false neural-selection credit.

The former `visual_trace` manual slate and separate RC4 captures remain historical diagnostics. They are useful software controls but are superseded as authority for this film. Do not splice their source, dev metrics, or model IDs into the current same-run cases.

## 4. Dataset and visual contracts

### Diverse actual inputs

Six actual executed inputs are synthetic, finite numeric regression tables: 96 rows, four features, and target, with production split seed 20261103 / label `public-run/fit-v1` giving 76 FIT and 20 HOLDOUT rows. They are not claimed as a representative real-world benchmark.

Selection observed a predeclared 24-case pool and used greedy farthest-point cosine distance in the **learned DOPE task128 + distribution128 slice, `DOPE629[256:512]`**. It excludes the copied raw descriptor bypass. No holdout quality, GP winner, or code/performance score selected cases. Actual selected distance range is 0.0881873784–0.5074985193, and selected minimum exceeds the old six-case maximum 0.0804961532. This establishes diversity within that tested pool, not a global optimum or quality claim.

Preserve all pool/selection receipts and exact selected CSV hashes. `visual_trace`'s original two-case default and bounded 2–8 CSV diagnostic contract remain available; the actual `visual_run` accepts a file or bounded CSV directory. A refreshed catalog invalidates case/source/rate/page bindings and display normalizers.

### Fixed layout and real graph

- Left: Excel-style actual FIT row preview, deterministic sampled row IDs, complete finite values, and a measured v256 grid. Values are visibly rounded for display; original CSV remains byte-exact. v256 is an input descriptor grid, not the learned 256-dimensional embedding used for diversity selection.
- Center: one JAIN network, label exactly `Problem encoding`, labelled constituent-origin sections, actual parallel branches/operators, and the synthesis boundary. `network_graph.py` maps all 43 captured affine operations and typed dependency links; selected coordinate views are subsets, not invented full connectivity.
- Right: an opaque neutral code pane, containing complete original Rust pages. Every original byte must reconstruct from the page manifest. Keep neural-origin colors out of the code palette.
- Temporary private bottom preview: actual operator parent-child projection of the same winning stored `pipeline.dsl`, verified against content-addressed model manifest/payloads and champion hash. Operator labels and parameter-atom counts are shown; complete literal syntax and values remain in source evidence. This reduces clutter without inventing a tree or claiming every parameter is printed. The current six cases contain four literal GP identities and six distinct Rust artifacts; repeated trees with changed fitted constants remain repeated trees.

Each displayed affine input/output node has a selected incident **nonzero real weight**. Derive nodes from real edges, add a valid incident edge within budget, or omit isolated zero-only coordinates. Static width encodes weight magnitude; pulse brightness is the actual `abs(w*x)` for that captured call. Zero contribution does not pulse. Node light uses captured values. Scales and coordinate selection are fixed across the entire catalog.

Normalize/concat/reshape/residual/attention links have their actual typed meaning, not fictitious dense weights. Decorative camera/layout motion is separate from measured values. Reveal/travel time is editorial; caption recorded activations/time expansion. Preserve sign, masking, cancellation, and unavailable states. Store shown/eligible counts and selected-edge coordinates in the scene receipt.

### Complete-source inspection

`code_pages.make_pages(source_path, output_dir, font_path=None, font_size=21, columns=48, rows=24)` renders new PNGs directly from the full native-emitted UTF-8 file. It does not transcribe or edit code. It preserves `::`, indentation, tabs, Unicode, original line endings, every fitted constant, training helper, and prediction function. Soft wrapping adds no source spaces; gutters/continuation marks are separate UI. Output directories are create-once.

Default pages are 684×660 opaque RGBA, 21 px Menlo at 1080 review scale, 48 columns ×24 rows, background `#0B1119`, 1 px `#263140` border, and neutral syntax colors. Pygments Rust highlighting or the explicit fallback only colors original spans; it is not the TEL07 counter. The manifest carries font/source/page hashes and contiguous byte/character/line ranges. `reconstruct_source` must reproduce the executed artifact SHA exactly; Blender additionally checks PNG hashes and image dimensions. Caps: 4 MiB input and 4,096 pages.

Current complete page counts A–F are **74,35,860,74,73,65**, totaling 1,181. Case C's 963,831-byte file is included in full. Board timing is `36 + page_count*code_hold_frames + 6` frames per case; default hold is six frames. With these files at 12 fps, the inspection scene is 7,338 frames / 611.5 seconds. This is an inspection timeline, not measured generation speed and not an approved film duration. Do not discard pages or shrink type to force the reference 90-second target. Users can pause/scrub to inspect fit, fitted-state, prediction, first, and last pages; later editorial duration requires design review.

### Source rates and clocks

Use only optimized-release receipts bound to current same-run model/source IDs. Caption **Warm repeated source emission · Rust lexical tokens/s**; retain the full scope in methodology: `jope.source-lex/v1`, repeated same already-fitted artifact, allocation/loop overhead included, model load/warmup/calibration/tokenization/hashing/verification excluded, batch at least `max(100ms,1000*clock_resolution,100*p99_clock_pair)`, every output byte-identical. Release/debug-assertion status, CPU, threads, source, and binary hashes are mandatory.

Inference time, code-emission time, compiler/training time, full capture process wall, Blender frame time, and code-page reveal cadence are distinct. None is a neural source-decoder rate. Missing/unbound metric means unavailable, not zero or an old dev result.

## 5. Ordered remaining work and parallel packets

### P0 — Freeze and audit inputs

1. Read current claims/branch/HEAD and keep the protected PR untouched. Preserve f035969 source and current external archive.
2. Verify `single-run-final/invocation-receipt.json`, full trace hash, compile/runtime source identity, signed assets/worker, all six CSV hashes, complete disjoint splits, and journal drop counts. Existing manifests/journals are immutable.
3. Recompute selection distances/greedy choices from recorded embeddings and confirm selected CSV identities. Do not reroll using output quality.

Complete when identities/selection are verified. Stop on missing/wrong source, bundle, signed worker, dataset, FIT partition, or overwritten archive.

### P1 — Validate actual journals and complete source semantics

1. `validate_trace.validate` verifies every journal call's run/dataset/FIT/bundle, tensor finiteness, affine closure, and displayed-call FP32 bit identity, including signed zero. Accept different JSON decimal lengths only when they represent identical native f32 bits.
2. The actual-run CLI/render gate invokes `engine_evidence`/`verify_actual_execution`: parse proof semantics, compare full Rust/Python/native/trained prediction vectors, require exact row/tolerance scopes and zero command exits, verify original/derivative hashes and unchanged training prefix, require model content address and champion DSL/run/source bindings.
3. Keep portable controls in `tests/test_actual_trace.py`: synthetic two-input tensors only; no private CI assets. Cover missing/changed calls, metadata/FP32/±0, cross-run source, journal/proof corruption, later unshown call identity/closure/finiteness. Supplement with full current-artifact semantic validation, not a fixture-only claim.
4. Resolve/review any rejection-hardening findings at the current helper SHA and write a new validation receipt. Retain older review receipts as historical; do not erase them.

Complete when full actual-run CLI validation and focused controls pass. Stop on any altered observation, incomplete journal, failed source/training parity, or unsupported seed/selection credit.

### P2 — Bind release timing and whole-source raster assets

1. Build `visual_emit_metric` with `--release` at the current frozen source and one reused target. Require optimized profile and `debug_assertions=false` in each receipt; never reuse old dev numbers.
2. Load each actual same-run `ModelObject`, call native `export_source("rust")`, require exact executed-source bytes, then measure bounded repeated fresh calls. Keep original token counts, repetitions, total interval and clock calibration.
3. Generate all six complete-source page sets under external custody. Check `reconstruct_source`, page count/order/offsets, all source and PNG hashes, opacity/dimensions/font, and full last-page coverage. No excerpts or code-token rejoining.

Complete when every current case has a release metric and complete page manifest. Stop on mismatched model/source, below-threshold timing, missing page, bad byte range/image hash, or output-directory reuse.

### P3 — Build and inspect the static Blender board

1. Root owns `blender/board.py`, `network_graph.py`, and `build_scene.py` integration. Use `--layout board`, current same-run trace/proofs/metrics, complete pages, selection receipt, and private GP-debug flag only for internal review.
2. Recheck immutable source/data bindings before scene allocation. Build fixed table/code panes and static camera, measured descriptors, all 43 observed affine operations, typed parallel dependencies, selected real edges, the verified winning GP operator-tree projection with parameter counts and full code pages. Reopen each immutable native journal and bind every actual admission round to its own activation-state interval; do not reuse first-round values for later rounds. Use CONSTANT keys for data/page visibility; no stateful simulation.
3. Derive case spans and cue frames from complete page counts. Save `.blend` and receipt first. Render encoding, active neural, actual GP/source-change, Model::fit, fitted-state, prediction, and final-page stills from the receipt's cues; do not use stale 288-frame catalog positions.
4. Reopen saved `.blend`; render nonsequential cue frames and verify identical state, source/page coverage, no escaped paths, no unexpected datablock growth, correct origin/sign colors, readable 21 px source, exact labels, and all visible graph nodes' real-edge coverage. Keep frame receipts/hashes.
5. Deliver the revised stills for user design feedback. **Do not render animation** until that review is approved. A later full-source inspection video/story edit needs its own duration/readability agreement.

The current v3 board has rendered frames 271 (A emitted training), 1063 (C emitted training), and 7165 (F prediction) at 3840×2160; files and hashes are in [STATUS.md](STATUS.md). Saved-scene reopen now passes with exact decoded RGBA equality for nonsequential frames 7165 then 271. [REVIEW_RECEIPT.json](REVIEW_RECEIPT.json) records the completed local review and remaining design/publication scope.

Complete first-review artifacts when representative current stills are inspected and their exact source/trace/page/scene hashes are recorded. A saved scene without images, or images of an earlier separate diagnostic, do not pass.

### P4 — Source-only PR and handoff

1. Root reviews the full source diff and focused validation, freezes the offline tool/docs head, and records which checks ran at that head. Hosted CI is separate; missing/manual jobs are not passed. Do not merge from local subset evidence.
2. Open/update the separate PR with actual behavior, acceptance commands, source pins, private receipt paths/hashes, and remaining design/video gates. Do not upload raw trace/source/model/code-page payloads, `.blend`, or GP debug images to a public PR.
3. Append locked stopped-head handoff: full HEAD, dirty files, commands/exits, receipts, pending gates, PR URL, writer disposition. Preserve authorship and other writers' work. No force-push to shared/protected branches.
4. Preserve unique evidence and recovery route before lane cleanup after PR closure. No blanket `target/` deletion or `/tmp` cleanup shortcuts.

## 6. Validation and replay commands

Run from the source worktree. Shell commands use `rtk`; build settings remain `CARGO_INCREMENTAL=0`, `CARGO_PROFILE_DEV_DEBUG=line-tables-only`, one target directory, and no globally exported `CARGO_TARGET_DIR`.

```sh
rtk git status --short --branch
rtk git rev-parse HEAD
rtk git worktree list --porcelain
rtk git diff --check
rtk proxy python3 -m unittest discover -s tools/visualization/tests -v
rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-service --features visual-trace --lib visual_trace::tests -- --nocapture
rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo check --locked -p jope-service --lib
```

The recorded actual capture command below intentionally refuses an existing output directory. Preserve `single-run-final`; for a new capture choose a fresh external directory and update every dependent path. Compile pin must equal runtime checkout HEAD with clean service source; a stale build is refused.

```sh
rtk proxy env JOPE_BUILD_SOURCE_SHA=f035969e68f4ab37d692d15972d60a900d2049b9 CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 cargo run --locked -p jope-service --features visual-trace --example visual_run -- /Users/bentaylor/Documents/ChatGPT/JopeDime/blender-render-20261103/assets/r882-fused-cpu-signed /Users/bentaylor/Documents/ChatGPT/JopeDime/blender-render-20261103/engine/jopedime-rc4-darwin-arm64 /Users/bentaylor/Documents/ChatGPT/JopeDime/blender-render-20261103/engine-diverse /Users/bentaylor/Documents/ChatGPT/JopeDime/blender-render-20261103/single-run-final
rtk proxy python3 tools/visualization/validate_trace.py /Users/bentaylor/Documents/ChatGPT/JopeDime/blender-render-20261103/single-run-final/trace.json --engine-proof /Users/bentaylor/Documents/ChatGPT/JopeDime/blender-render-20261103/single-run-final --emit-metrics /Users/bentaylor/Documents/ChatGPT/JopeDime/blender-render-20261103/single-run-final/release-metrics --receipt /ABS/FRESH_VALIDATION_RECEIPT.json
```

Source-page CLI for each current `model-paths.json` source uses a new external per-case directory; current `full-code-final/` is already preserved:

```sh
rtk proxy python3 tools/visualization/code_pages.py --source /Users/bentaylor/Documents/ChatGPT/JopeDime/blender-render-20261103/single-run-final/case-a/export-rust-python/src/rust/predictor.rs --output /ABS/NEW_PAGE_ROOT/case-a --font-size 21 --columns 48 --rows 24
rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo build --release --locked -p jope-service --features visual-trace --example visual_emit_metric
```

Run the release binary with each current model directory, exact executed Rust path, and fresh metric receipt path. Copy exact successful argv/binary/model/source identities from `single-run-final/release-metrics.receipt.json`; do not substitute a process-wall timer.

Build the current board without animation. `--frames 288` satisfies the generic CLI's six-case minimum; board's **actual duration is derived from every full-source page**, not that argument. `--engine-proof` is a directory, not a JSON filename. Internal GP debug is on only for the requested private preview.

```sh
rtk proxy /Users/bentaylor/Documents/ChatGPT/JopeDime/blender-runtime/Blender.app/Contents/MacOS/Blender --background --factory-startup --python tools/visualization/blender/build_scene.py -- --layout board --trace /Users/bentaylor/Documents/ChatGPT/JopeDime/blender-render-20261103/single-run-final/trace.json --engine-proof /Users/bentaylor/Documents/ChatGPT/JopeDime/blender-render-20261103/single-run-final --emit-metrics /Users/bentaylor/Documents/ChatGPT/JopeDime/blender-render-20261103/single-run-final/release-metrics --code-pages /Users/bentaylor/Documents/ChatGPT/JopeDime/blender-render-20261103/full-code-final --selection /Users/bentaylor/Documents/ChatGPT/JopeDime/blender-render-20261103/embedding-selection/selection.json --output /ABS/NEW_BOARD_OUTPUT/preview.blend --width 3840 --height 2160 --frames 288 --fps 12 --code-hold-frames 6 --show-gp-tree
```

Read `preview.receipt.json` for `case_frames`, `cue_frames`, `actual_frames`, `native_round_frames`, GP projection, page/edge bindings and scope. The current saved file is external `review-final-board-v3/preview.blend`; it contains 7,338 frames at 12 fps, an inspection-only 611.5-second timeline. Three stills exist; no video has been rendered or approved. Render chosen cue frames from that saved file with `blender/render_frames.py --output /ABS/STILLS --frame FRAME` (repeat `--frame`); retain the exact invocation. Frame numbers/duration are editorial, never source-emission or neural latency measurements.

## 7. Stop conditions, evidence, and final completion

Stop promotion on source/bundle/run/FIT/model mismatch, journal loss, changed native FP32 bits, invalid affine closure, failed source/training execution, source-byte or page/image corruption, missing/unqualified rate, incorrect GP topology, invented/isolated neural nodes, or unreadable/clipped code. Preserve the failure and fix its owning source; never repair captured arrays, reroll inputs for favorable quality, or replace missing data with zero. Continue only disjoint work that does not imply the failed gate passed.

Expected external artifacts: selection catalog/traces/receipt, selected CSVs, exact same-run native journals, champion/model/source/proof archive, full semantic validation, optimized-release metric receipts, complete-source PNG manifests, saved board, cue/frame receipts, reviewed stills, and final source-bound handoff. Every receipt records full source identity, exact command/cwd, inputs/hashes, relevant environment/resources, exits, measured scope, output hashes, and skipped/pending work. [RECEIPT.template.json](RECEIPT.template.json) is a template, not evidence.

Root keeps concise progress updates during active work and sends schema/ownership/blocker decisions promptly. Final response supplies the separate PR/branch/worktree, master plan, actual revised stills and limitations. First review completion requires actual same-run fused/source/training proof, complete code inspection assets, representative current rendered stills and source-only PR. Final animation/story/public package remain pending user design approval and their own render/disclosure/CI gates. Neither a local check nor a signed internal candidate grants public-release or performance qualification.
