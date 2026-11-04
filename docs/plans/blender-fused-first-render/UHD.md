# JAIN native UHD delivery

**Stopped at the owner's request, 2026-11-04.** The native render is incomplete;
no finished 4K movie is claimed. All owned processes on xbabe1, xbabe3, and the
local Metal device were stopped. The cancellation receipt records 120 sealed
production plates, eight sealed probes, and 48 unsealed partial plates retained
for recovery. Native rendering and composition code remains in this PR for a
later explicit restart. The active delivery is the [1080p VEOX ending](ENDCAP.md).

Custody: `h256/jain-final-4k-v1/cancellation.receipt.json`, original render plan,
frozen worker copies, per-chunk journals and receipts. Resume only after renewed
authorization; reuse sealed chunks and rerender interrupted chunks. Recheck
idle GPUs and disk reserve. The original UHD timeline predates the VEOX endcap;
a future full-film UHD request must explicitly choose its timeline before reuse.

Re-render the approved R2 opener and Dendrite demonstration at 3840 × 2160,
30 fps. Deliver the 12-second opener and seamless 45-second film as ProRes
4444 XQ/PCM masters and H.264/AAC playback copies. Preserve all ten datasets,
15 complete source files, 9,961 visual source rows, model identities, measured
benchmark scope, neural topology, input attribution, animation and source timing.

This is a native render, not an enlargement of the 1080p movie. Reuse the two
saved scenes and their original receipts; keep the approved images as read-only
references. Do not rerun inference, change model checkpoints, invent intermediate
activations, alter the approved spatial layout, or touch other PR branches.

## Read first and ownership

Read `INTRO.md`, `tools/visualization/blender/{intro_render,render_animation}.py`,
`tools/visualization/{intelligence_wave,code_pages,table_panels,intro_film}.py`,
and the canonical scene and R2 movie receipts. Work in the existing
`codex/blender-fused-first-render` lane and draft PR #2114.

The Blender worker owner handles UHD quality and sparse case-frame selection.
The demo compositor owner handles `uhd_demo.py`. The intro compositor owner
handles `intro_hq.py`, `uhd_pixels.py`, and the explicit `intro_film.py` UHD
dispatch. The root owns native UI scaling helpers, focused regression tests,
documentation and integration. Remote scheduling and generated media remain
outside Git. Do not overlap edits across these packets.

## Image and timing contract

- Cycles uses a 256-sample maximum, adaptive threshold 0.003, minimum 32
  samples, high-quality denoising, fixed seed, and native RGB16 PNG output.
  Probes may use 32/128 maximum samples; production receipts must record 256.
- `--scale 200 --quality uhd-master` renders the original 1920×1080 scene at
  native 3840×2160 without changing camera, geometry, textures or viewport ratios.
- Preserve 16-bit RGB through sparse PNG decoding, neural wave composition,
  the lossless FFV1 demo intermediate, and final ProRes encoding. Native text
  and flat UI colors use newly rasterized 4K fonts and RGBA overlays, composited
  onto the 16-bit background. Do not convert the 3D raster through Pillow RGB8.
- Keep the original 54-column logical source wrap. Scale source typography,
  line height and viewport by two; source chunks and GP span identities stay
  unchanged. Table values, row permutations, percentile colors and headers
  remain attached to their original measured records.
- The 1,020 selected moving demo frames use 960 distinct Blender plates.
  Render those exact sparse frames, plus all 330 opener plates. Every omitted
  raw frame is outside the approved final movie's source mapping.
- The UHD intermediate uses `jain.uhd-demo/v1`, logical `frames=1260`, actual
  `encoded_frames=1020`, and explicit `source_frame_indices` equal to the R2
  retime plan. Retain all 126 logical scroll offsets per case. The intro checks
  this mapping before decoding; it must never retime the sparse stream twice.
- Bind the intermediate to the canonical Dendrite scene receipt. The intro
  scene must bind to that same source receipt. A resolution change does not
  grant permission to substitute a different capture.

## Inputs and output location

On the delivery workstation, `JAIN_ROOT` is
`/Users/bentaylor/Documents/ChatGPT/JopeDime` and the external evidence root is
`$JAIN_ROOT/blender-render-20261103/h256`.

Read-only inputs:

1. `variable-attribution-v1/dendrite-final-v1/scene/scene.blend` and its receipt.
2. `jain-intro-v2/scene/intro.blend` and its receipt.
3. `variable-attribution-v1/dendrite-final-v1/final/encode.receipt.json`.
4. `variable-attribution-v1/capture/multilingual-source/catalog.json`.
5. `variable-attribution-v1/capture/table-panels/catalog.receipt.json`.
6. `variable-attribution-v1/capture/neural-influence.json`.
7. `variable-attribution-v1/dendrite-final-v1/branding/nvidia-cuda-toolkit.jpg`.

The similarly named `variable-neural-influence.json` and CUDA proof PNG are
different artifacts. Hash gates must reject them rather than silently using
their contents. Use original font files and full-resolution language artwork.

New output and receipts belong to `h256/jain-final-4k-v1/`. Its
`render-plan.receipt.json` binds every master frame, original source frame,
case, absolute Blender frame and historical plate hash. Keep the prior 1080p
files unchanged. Rendering code and composition code live in the repository;
bulk PNGs, scenes, fitted objects, movie files and lossless intermediates do not.

## Execution and validation

1. Verify the existing claim, branch, head, dirty state, sources and available
   disk. Pin the Blender runtime and worker hashes before dispatch.
2. Probe native 4K frames at 32, 128 and 256 samples, including a Go-board frame
   and dense neural frame. Inspect quality and measure actual render times.
3. Render disjoint sparse chunks on idle authorized GPUs. Guard each new job
   against foreign compute; do not terminate or compete with another owner's
   training. The local Metal GPU may supplement OptiX when measured useful.
4. Hash each transferred RGB16 PNG and verify its IHDR, dimensions, frame and
   source identity. Publish complete receipts only after every required frame
   has passed. Restart failed chunks into fresh directories, preserving evidence.
5. Regenerate native 4K tables and complete code pages. Verify all source bytes,
   GP spans, logical row counts and table permutations against the approved cut.
6. Compose the sparse lossless FFV1 intermediate, then the ProRes master and
   playback copy. The standalone master must retain the first 360 encoded
   video packets, with its own final audio fade.
7. Fully decode all four videos; verify 360/1350 frames, exact 12/45-second
   picture duration, 30 fps, UHD dimensions, color/profile/bit depth, audio,
   chapters, source coverage and the same-case handoff. Inspect native-size
   text, thin neural connections, dark gradients, and the source-panel border.
8. Run visualization tests, Python compilation and `git diff --check`. Bind
   review and delivery receipts to the exact final commit; update the existing
   draft PR and coordination log without claiming hosted CI or a merge.

Keep at least 25 GiB free locally. Stop dependent work on input hash drift,
invalid frame mappings, missing source rows, a busy foreign GPU, invalid bit
depth, disk reserve breach, or failed decode. A 1080p preview, an upscaled frame,
an incomplete shard, and an encoded but unverified file do not qualify as UHD
delivery. Report actual progress and failures without substituting weaker media.
