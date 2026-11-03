# JAIN inference review

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

Video rendering remains pending user design approval. The CLI retains
`--render-animation` for that later gate; do not invoke it during still review.
