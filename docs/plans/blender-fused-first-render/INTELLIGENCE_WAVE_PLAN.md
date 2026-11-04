# JAIN Intelligence Wave short film

The [current revision](NEURAL_CELL_ATTRIBUTION.md) uses variable-size tables and measured cell sensitivity from the fused ranker. This document records the earlier cuts.

## Multilingual revision

The 42-second timeline uses C++, Python, Rust, TypeScript, Go, JavaScript, CUDA C++, Python, Rust, and C in dataset order. The displayed filename, syntax lexer, and source label follow each visible file. Each chapter's language icon and name sit below the network. TypeScript and Go show their bindings and Rust/Wasm kernels. C and CUDA C++ show their interfaces and Rust cores. The source archive contains the complete build files.

Use the exact selected GP and original model object for every chapter. The local C++, Python, Rust, JavaScript, and C ABI checks execute the displayed targets on the saved normalized FIT and probe rows. The TypeScript, Go, and CUDA C++ chapters bind their emitted sources and qualify the numerical Rust cores on the host. Their Wasm and CUDA execution paths have separate qualification gates. Source-span cues follow the actual operations in each file and the underlying core when the interface contains no selected operations.

The source pane shows the current line alone. The historical fitted export rate uses compact forms such as `2.4M tokens/sec` and `300k tokens/sec`. At 30 fps, round successive 0.28-second table boundaries to frames. This gives 15 orders per chapter and five times the previous cadence. Seed each permutation by case ID and record its frame and row IDs in the encode receipt.

Calculate feature importance percentiles once for each dataset across its 68 visible, measured FIT feature sensitivities. Average tied ranks and anchor exact zeros at the low end. Color cells from dark red through amber to green. The thin bars retain their relative sensitivity lengths. Move row IDs, values, the purple rightmost target, sensitivity marks, and percentile colors together at every shuffle.

Run the original source preparation below to reproduce the pinned Rust recipe catalog. Then emit and qualify the multilingual source and compose the earlier percentile shuffle cut.

```sh
rtk proxy python3 tools/visualization/multilingual_wave_source.py \
  --gp-source "$ANIMATION_TEN/gp-source" --rust-catalog "$WAVE_ROOT/recipe-source/catalog.json" \
  --output "$WAVE_ROOT/multilingual-source" --rustc "$PINNED_RUSTC" \
  --feat-math-rlib target/release/deps/libfeat_math-22a436f36cb6fece.rlib
rtk proxy python3 tools/visualization/intelligence_wave.py \
  --scene-receipt "$ANIMATION_TEN/scene-v2/scene.receipt.json" \
  --source-catalog "$WAVE_ROOT/multilingual-source/catalog.json" \
  --loops-root "$ANIMATION_TEN/loops" --table-panels "$ANIMATION_TEN/table-panels-v1" \
  --output-dir "$WAVE_ROOT/final-percentile-shuffle"
```

Validate all 1,260 decoded frames, ten chapters, source reconstruction for each displayed file, and the lexer selection. Check that all 150 table orders preserve every row and its colors. Verify icon and table pixels on the review stills. The [icon inventory](language-icons/README.md) binds the input marks and licenses.

## Original Rust cut

The original Rust cut is reproducible at commit `29928ffee2601aadafcfeb00a4b2b080497f56a2`. Its plan and evidence follow.

## Objective and scope

Deliver a 42.0 second 1920×1080, 30 fps cut on the existing [draft PR #2114](https://github.com/neverhuman/JopeDime/pull/2114). Play the ten existing, explicitly synthetic regression datasets in their original order. Each gets 126 frames. Keep the verified 17-row table at left and the real Blender neural display at center. Scroll the complete Rust recipe for the actual selected GP at right. A light wave across the existing neural display provides a visual cue for each data-to-recipe sequence.

Keep the previously delivered film, scene, source, models, trace and measurement receipts in their original custody. Each existing model object retains the fitted state for its selected recipe. The short film shows converter-generated recipe source, whose training and prediction routines are separate from that state. The same recipe can have identical source across several datasets. Cases A/G/J and C/D provide useful checks of this behavior.

The wave is editorial motion applied to verified render pixels. The plate's sampled weights and activations retain their original measured provenance. The overlay provides no measured per-operator neural attribution, elapsed inference time, weight update or topology claim. Keep the original data families, fits and selection. The narrow local refittable Rust profile in PR #2124 is a separate product change. Its current operation slice does not cover the ten prior selected recipes.

Assume the prior plates, tables, case identities and selected-model objects are available and hash-verifiable. Reuse them for this edit. Keep model training, candidate reselection, GPU inference, fresh GPU scene rendering, converter math changes, private asset publication, deployment and merge outside this packet.

## Read first and ownership

1. Read [AGENTS.md](../../../AGENTS.md), [AGENT_CHAT.md](../../../AGENT_CHAT.md), the root [product direction](../../../MASTER_PLAN.md#product-target), [ANIMATION_PLAN.md](ANIMATION_PLAN.md), [ANIMATION_RENDER_RECEIPT.json](ANIMATION_RENDER_RECEIPT.json) and [STATUS.md](STATUS.md). Check the locked claim, worktree, branch, full HEAD and open queue before writing.
2. Read `tools/visualization/animate_source.py` for the existing loop, scene, page and table binding checks. Read `table_panels.py`, `code_pages.py` and `cinematic_pages.py` for exact raster assets and byte reconstruction. Read `blender/jain_cinematic.py` for the old baked metric region, protected HUD and viewport.
3. Read [emitter documentation](../../emitters.md), `crates/jope-cli/src/kernel/emit.rs` and `crates/feat-math/src/emit_rs.rs`. Compare the existing selected DSL proofs with the recipe-only converter. Read `tools/visualization/h256_gp_validate.py` for the old trace-to-model binding. Distinguish the qualified native 0.4.1 epoch from later compile compatibility.

Use the existing `codex/blender-fused-first-render` worktree under `.agent/worktrees/`. The claim starts at `c16b5e634a55646c0012c202bd2c1eb669fea9b7`. The source worker owns new `tools/visualization/short_recipe_source.py` and its external source receipts. The effect worker owns new `tools/visualization/intelligence_wave_effect.py`. The integration owner controls new `tools/visualization/intelligence_wave.py`, focused tests, this plan after handoff, final receipts, documentation, Git and the PR. Review source/model binding and film composition in parallel when write paths are disjoint. Share one existing Cargo target if a converter build is needed.

The previous evidence root is `blender-render-20261103/h256/animation-ten-v1/` under local artifact custody. Read its `gp-source/`, `dataset-catalog.json`, `scene-v2/scene.receipt.json`, `loops/`, `table-panels-v1/`, `metrics/catalog.receipt.json` and prior review receipts in place. Use a fresh sibling `h256/intelligence-wave-v1/` for output. Keep render binaries, scene files, traces, CSVs, model objects, new code rasters, media and logs outside Git. Keep PRs #2071/#2115 and #2124 outside the write scope.

## Ordered work

1. **Bind the original inputs.** Read the ten case IDs and names from the qualified catalog. Verify each case's actual selected DSL and existing model receipt, original model directory and part hashes. Verify all 1,200 loop frames against their per-case render receipts and scene identity, and the exact 17-row panel PNGs against their existing catalog. Record old source, metric and plate identities before adding any derivative. Fail if a case is missing or ordered differently.
2. **Emit and verify recipe source.** Use the pinned `feat_math::transpile::emit` recipe path with `GenomeIr::parse` and `EmitOptions::default()`. Link the helper against the recorded release `feat-math` rlib and use the exact selected DSL. Save the complete Rust bytes and SHA-256 for each case. Verify deterministic re-emission and compile the source with the pinned toolchain in the existing qualified build scope. Confirm the selected operations and fit/predict routines appear in the generated recipe. Exercise the recipe on the 76 original normalized FIT rows and the 96 original probe rows under the saved input and target contract. Compare against the previous native predictions with the established tolerance. Check that no fitted-state constructor, frozen prediction fixture or retained model payload enters the source. Preserve the original immutable model path and ID as a separate binding. A recipe emitter refusal blocks that case. Keep the ten selected programs fixed.
3. **Build source view assets.** Syntax-highlight the exact converter bytes. Preserve whitespace and soft wrapping with reversible mappings. Use the old source viewport at `x=1319, y=133, w=563, h=845` unless the compositor verifies and records a new pane. Use a single vertical document coordinate across all raster segments. Hash the rasters and record the line/byte mapping. Prove full-source reconstruction with the manifest. Hold the initial and final scroll poses briefly when the coverage check permits it.
4. **Bind optional GP labels.** Resolve each label to an actual selected DSL operation and verified line or byte span in the displayed Rust, using the corresponding fit/predict call site. Distinguish repeated operations by occurrence and order. Display the label only while its bound span is in view, with enough room for readable code. Missing, ambiguous or out-of-view bindings omit the label. Labels identify recipe operations. Keep them independent of the central wave cue.
5. **Compose 10 chapters.** For output frame `i` within a chapter, use verified plate sample `round(i * 119 / 125)` for `0 <= i < 126`. This preserves one 120-sample loop in each chapter with six repeated samples. Keep original pixels outside the declared wave, code and KPI masks. Paste the existing fixed 17-row panel without resizing or changing its importance values. At a chapter cut, switch table, loop, recipe and label binding together. Keep the target purple and last. Confine the wave to existing pixels of the central network and its floor reflection. A 34-frame sweep travels left to right. Increase its gain within a few frames of a fit or predict call-site cue and mark the corresponding source rows while they are in view.
6. **Scroll and handle the KPI.** Reveal the opening viewport over seven frames. Scroll linearly from chapter frame 7 through 119 and hold the end through frame 125. Use pixel offsets within the source viewport and preserve line order between raster segments. Adjacent offsets must leave enough overlap to show every wrapped row fully in at least one frame. At 4.2 seconds, this is a complete traversal with a separate exact source file available for reading. Cover the old baked KPI area and use `FITTED EXPORT BENCHMARK` for the archived measured figures. A new throughput figure requires a fresh CPU measurement with explicit start/stop scope and the canonical count of the exact recipe bytes. Keep animation velocity independent of any measured rate.
7. **Validate a review render, then encode.** First generate snapshots at the start, moving wave and final scroll pose for representative short and long cases. Check focus, fixed bounds, table pixel identity, source readability, old KPI occlusion and an exact label binding. Encode H.264 with 30 fps constant timing and ten chapters in a new output directory. Decode all frames, inspect every chapter boundary and retain review stills across all ten cases. Record any remaining visual or qualification limitation.

## Evidence and validation

Use `rtk` for shell commands. Confirm the final compositor CLI with `--help` before publishing replay instructions. Retain exact argv, cwd, toolchain and exits. The source worker produces `recipe-source/catalog.json` with schema `jain.short-recipe-source/v1` and per-case original model bindings. Treat old `scene-build-v2.command.json` and `movie.command.json` as replay evidence for their own original artifacts.

```sh
rtk proxy python3 tools/visualization/short_recipe_source.py \
  --gp-source "$ANIMATION_TEN/gp-source" --output "$WAVE_ROOT/recipe-source" \
  --rustc "$PINNED_RUSTC" \
  --feat-math-rlib target/release/deps/libfeat_math-22a436f36cb6fece.rlib
rtk proxy python3 tools/visualization/intelligence_wave.py \
  --scene-receipt "$ANIMATION_TEN/scene-v2/scene.receipt.json" \
  --source-catalog "$WAVE_ROOT/recipe-source/catalog.json" \
  --loops-root "$ANIMATION_TEN/loops" --table-panels "$ANIMATION_TEN/table-panels-v1" \
  --output-dir "$WAVE_ROOT/final"
rtk proxy python3 -m unittest discover -s tools/visualization/tests -v
rtk git diff --check
rtk proxy ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate,avg_frame_rate,nb_read_frames:format=duration \
  -of json "$VIDEO"
rtk proxy ffmpeg -v error -xerror -i "$VIDEO" -map 0:v:0 -f null -
```

Require exactly 1,260 decoded frames, 1920×1080, 30/1 fps, 42.0 seconds and ten contiguous 126-frame chapters. The receipt binds full source HEAD, compositor and effect hashes, previous scene and all ten loop receipts, original table/catalog and model/DSL hashes, recipe converter identity, complete Rust/PNG manifests, per-case plate indices and scroll intervals, label spans, overlay masks, KPI provenance, MP4 hash, decode output and sampled visual review. A source change needs new source and evidence hashes.

Test the timeline and source-coordinate math on a compact fixture. Cover chapter transition, all 120 plate samples, first/last source lines, intermediate viewport coverage, duplicate operations, and refusal of a changed asset or unsupported label. Validate table equality and wave confinement on the uncompressed compositing frame. Review the decoded movie for codec changes, clipping, flicker and chapter timing. A changed source catalog must produce a new derivation receipt. Keep the archived scene and loop receipts valid for their original bytes.

## Stop and handoff

Stop dependent work on a changed input hash, missing case, unsafe output reuse, unsupported recipe, incorrect state binding, lost source bytes, scroll gap, out-of-bounds overlay, unbound KPI or label, or failed full decode. Retain the failed receipt and fix the cause without substituting different data or a different program. Document any missing toolchain as a qualification gap.

Post progress when source and model bindings pass, when the review frames are ready and when the final decode passes. Log the stopped full HEAD, dirty paths, command outcomes, source and media receipt locations, remaining gaps and ownership under the claim lock. Commit scoped source, documentation and bounded evidence only. Review the final full SHA, preserve shared branch history if an upstream update is needed, push normally, update the existing draft PR #2114 and verify remote readback. Report hosted CI separately from local proof.
