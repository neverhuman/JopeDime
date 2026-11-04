# JAIN inference animation

The [VEOX endcap](../../docs/plans/blender-fused-first-render/ENDCAP.md) adds a
six-second 1080p closing animation and a 45-second assembled film. Its supplied
vector logo pulses cyan to violet from V through X, with X glowing strongest and
longest. `blender/veox_endcap.py` builds/renders it; `endcap_film.py` preserves the
approved opening and all ten captured demonstration chapters while assembling
the ending. The [native UHD work](../../docs/plans/blender-fused-first-render/UHD.md)
is stopped and preserved at the owner's request, with no completed 4K delivery.

The current [45-second film and opener](../../docs/plans/blender-fused-first-render/INTRO.md) combines a 12-second opening sequence with the Dendrite demonstration. The opener's final second introduces the moving demo, which continues without restarting. `blender/jain_intro.py` builds the centered satin Go board, large algorithm tile, connected illustrative feedback loop, and transformation into the captured network. `blender/intro_render.py` renders GPU plates; `intro_film.py` adds centered typography, the actual demo's panels and wordmark, original synthesized sound, and the verified 45-second assembly. The search metaphor resolves into “One inference. One algorithm.” before the full demo takes over.

The equal-third 1080p30 demonstration shows ten varied synthetic tables, a rotating measured neural display, and the complete refittable recipe sources across eight languages. Each source section appears in the accelerated scroll; the full source packages provide the reading reference. Model parameters stay in separately saved fitted objects. CPU-only benchmark captures remain distinct from editorial typing speed, and GPUs are used only to render. The displayed fitted-export benchmark retains its original Rust counter and measured scope when other output languages appear.

All generated data/model/source/scene/video assets stay outside Git. The [original capture plan](../../docs/plans/blender-fused-first-render/ANIMATION_PLAN.md), [source wave plan](../../docs/plans/blender-fused-first-render/INTELLIGENCE_WAVE_PLAN.md), and [cell attribution method](../../docs/plans/blender-fused-first-render/NEURAL_CELL_ATTRIBUTION.md) retain the source qualification and measurement evidence. See the intro plan for rebuild commands, ownership, dependencies, and validation.

## JAIN spatial layout review

Add `--spatial-layout` to the cinematic scene build. The first set is `iris`, `twist`, `hourglass`, `prism`, and `manta`. The biological set is `dendrite` (three spreading fans), `synapse` (paired bulbs with a thin bridge), `fascicle` (two winding bundles), `ginkgo` (a tapered leaf), and `cortex` (corrugated folds). The default `layered` layout retains the original coordinates. Each option moves nodes across their existing causal plane. The plane sequence, actual parameter/channel identities, connection endpoints, and measured activations and contributions stay fixed. The table and source connectors follow the moved ports.

For a matched design review, build a fresh scene for each layout with the same capture and `--animation-loop`. Pass `--case-id case-a --frame-start 89 --frame-end 89` to `blender/render_animation.py` for each layout and composite them with the same source and table frame. Compare `network.operations` and `network.stage_planes` in the scene receipts for equality, then compare `network.ordered_endpoint_positions_f64_sha256` for distinct geometry.

The Dendrite cut uses `--spatial-layout dendrite --point-connectors --animation-loop --width 1920 --height 1080`. Each display lead fans into a short shared segment with one endpoint on its panel border. A faint independent display stroke keeps the lead visible at zero signal. The scene receipt records both camera-local endpoints and their projected pixels under `connectors.panel_contacts`. The existing per-strand animation and internal neural geometry use the same capture.

## Earlier still and baseline tooling

The latest [H256 checkpoint plan](../../docs/plans/blender-fused-first-render/H256_PLAN.md) covers the trained five-fold solution encoder, explicit landmark mappings, actual proposal selection, native selected-GP refits and complete Rust scrolling. Its entry point is `blender/h256_scene.py`; the earlier board below preserves the original authenticated r882 baseline.

Optional offline Blender tooling for one authenticated fused JAIN model. The
[master plan](../../docs/plans/blender-fused-first-render/MASTER_PLAN.md) gives
worker scopes, exact host paths, validation and remaining film gates;
[status](../../docs/plans/blender-fused-first-render/STATUS.md) binds current
source, datasets, models, code and measured receipts.

The authoritative `jope-service` example is **visual_run**, behind the opt-in
`visual-trace` feature. It runs the existing production Engine, observes its real
conditioning/admission calls, exports that run's GP champion through native
GP → Rust/Python conversion, and executes the unchanged artifacts. It also
executes emitted Rust training on the actual FIT rows. See the
[native example guide](../../crates/jope-service/examples/README.visual_trace.md).
The older `visual_trace` example remains a component diagnostic with a declared
slate; it is not the authority for the end-to-end review.

## Prepare private inspection assets

Use the repository's pinned Rust toolchain, a clean committed service scope,
and one reused target directory. On this host the system Cargo/Rust defaults and
global target override differ; copy the verified overrides from the capture
receipt instead of changing global settings. Keep all generated assets outside
Git. The signed release loader remains the authority; no unsigned constructor is
exposed. No neural training, serving promotion or GPU execution is involved.

```sh
rtk proxy python3 tools/visualization/validate_trace.py /ABS/CAPTURE/trace.json \
  --engine-proof /ABS/CAPTURE --emit-metrics /ABS/RELEASE_METRICS \
  --receipt /ABS/NEW_VALIDATION.json
rtk proxy python3 tools/visualization/code_pages.py \
  --source /ABS/CAPTURE/case-a/export-rust-python/src/rust/predictor.rs \
  --output /ABS/NEW_PAGES/case-a --font-size 21 --columns 48 --rows 24
rtk proxy python3 -m unittest discover -s tools/visualization/tests -v
```

Create pages for every captured case. `code_pages.py` needs Pillow; NumPy is
required for validation. Pygments supplies optional syntax coloring with a
byte-preserving fallback. Blender includes its own NumPy. Every original Rust
byte is preserved, including whitespace, `::`, training helpers, fitted state,
prediction and selfchecks. Neutral opaque PNG pages prevent 3D geometry from
crossing code. `reconstruct_source` and page hashes reject omission or mutation.

## Build review stills

```sh
rtk proxy /ABS/BLENDER --background --factory-startup --python \
  tools/visualization/blender/build_scene.py -- --layout board \
  --trace /ABS/CAPTURE/trace.json --engine-proof /ABS/CAPTURE \
  --emit-metrics /ABS/RELEASE_METRICS --code-pages /ABS/PAGES \
  --selection /ABS/EMBEDDING_SELECTION.json --output /ABS/NEW_REVIEW/preview.blend \
  --width 3840 --height 2160 --frames 288 --fps 12 --code-hold-frames 6 \
  --show-gp-tree
```

Pin Blender 5.2.3 LTS, build `e13b728c6880`, for the initial review. Read the saved
`preview.receipt.json` for actual case spans and encoding/training/fitted-state/
prediction/page cues. The board expands its duration to retain every source
page; the six current files require 7,338 inspection frames. Reopen the scene
and render selected cues without rebuilding:

```sh
rtk proxy /ABS/BLENDER --background /ABS/REVIEW/preview.blend --python \
  tools/visualization/blender/render_frames.py -- \
  --output /ABS/STILLS --frame FRAME --frame ANOTHER_FRAME
```

Table and full code remain fixed at the sides. The center shows the actual
parallel graph with selected real channels, all recorded admission rounds,
signed activations/contributions, and neutral unweighted functional routes.
Every displayed neuron has a real incident weight. Geometry and weights stay
fixed. FP32 numeric atlases use Non-Color, Closest and Separate Color → Red;
zero values produce no data light. Pulse travel and inspection time are
editorial and labelled. Attention is shown separately as `A − 1/8`.

The opt-in GP debug view shows actual operator parent/child structure and
parameter counts; full literal syntax stays in private evidence. Raw traces,
`.blend` files and receipts contain internal values and are not sanitized public
packages. Keep them and source-page assets outside the public PR.

Rates use optimized-release warm repeated emission of the same fitted artifact,
the `jope.source-lex/v1` Rust lexer, and a measured interval above 100 ms. They
include allocation/loop overhead and are distinct from neural decoding, unique
source throughput and end-to-end product timing. The renderer rechecks model,
source and rate identities. Seed winners receive no neural-selection credit.

The historical still-review CLI retains `--render-animation`. The approved
current film uses the Dendrite and intro pipeline described above.
