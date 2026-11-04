# VEOX closing shot

The owner stopped the native 4K render and requested a 1080p animated VEOX ending and an assembled film. Preserve the partial UHD render for a later explicit restart. The approved opening and measured demonstrations remain the source of truth.

The final line is **Turn what if into what’s next.** It turns the opening's search question into an invitation to build. The supplied VEOX vector mark resolves through cyan and violet illumination above a dark reflective floor, the line appears beneath it, and the whole photograph fades to black. The decorative closing filaments are brand animation, not a claim about measured neural pathways.

The owner subsequently requested a left-to-right letter pulse with the X glowing
strongest and longest. The final shot assigns a separate material to each letter
(both components of V share one material). Cyan V peaks at frame 24, blue E at
42, violet O at 60, and violet X at 82. X has the strongest crest followed by a
restrained after-pulse and a long glow through frame 155. The steady slogan is
unaffected. The complete photograph, including its real reflection, fades from
frame 158 to exact black by frame 179. Verify the actual rendered letter regions,
not only the keyed shader values, since tone mapping changes apparent intensity.

## Objective and limits

Produce a standalone six-second endcap and a seamless 45-second 1920 × 1080 film at 30 fps. Keep the approved standalone 12-second opener, ten captured datasets, eight supported language treatments, complete source files, measured timing, and original source identities. Do not run new inference, training, benchmarks, or the stopped UHD job. Do not change protected PR #2071, model/export implementations, or other worktrees. Use existing draft PR #2114.

The 330-frame opening is unchanged. The demonstrations occupy 84 frames each, followed by the 180-frame ending at frame 1170 / 39 seconds. The first 31 selected Case A frames are pinned to the approved cut, preserving its 12-second moving handoff. A constrained monotonic schedule keeps every previously displayed GP cue, every table shuffle, every fully visible source row, and source-frame skips of at most two. The first 14 ending frames blend from the last complete demonstration frame; its final source viewport is never obscured before the endcap begins. The standalone ending begins on black. Both versions end on black and silence.

This remains a fast visual overview. Source completeness in a viewport does not imply enough reading time for every program. The source artifacts remain the readable reference.

## Read first and ownership

1. `INTRO.md` and the approved `jain-intro-v2/final/intro-film.receipt.json`.
2. `tools/visualization/intro_film.py`, especially source custody, `plan_retime`, raster prelap, and encoding checks.
3. The original Dendrite `final/encode.receipt.json`, source viewport, complete file hashes, code cues, and table row orders.
4. `tools/visualization/blender/render_animation.py` for explicit GPU selection.
5. The exact owner-supplied SVG, copied byte-for-byte to `tools/visualization/assets/veox-logo.svg` (SHA-256 `2fb4e1c587044aad11977f9cbec3f5c5af772153850d6fa226fb4d91f65c1dfb`).

Root owns `blender/veox_endcap.py`, the supplied asset, this document, integration, and Git. The composition packet owns `endcap_film.py` and `tests/test_endcap_film.py`. Rendering owns only external scenes, logs, frame receipts, and media. The review packet is read-only. Freeze worker bytes before final rendering and freeze composer bytes before encoding; ownership transfers must be logged under the existing coordination lock.

All scenes, copied runtimes, fonts, images, sound, and videos remain outside Git under `/Users/bentaylor/Documents/ChatGPT/JopeDime/blender-render-20261103/h256/jain-veox-endcap-v1`. The supplied SVG is the only new brand asset in Git. Use the one existing worktree and its existing `target` for test scratch data.

## Implementation and reproduction

1. `veox_endcap.py build` parses the supplied five SVG paths, preserving both V components and the O's hole. It creates shallow relief, physically lit reflections, restrained transient filaments, packed typography, a locked camera, and keyed fades. Unsupported SVG commands fail rather than altering the logo. The saved scene embeds the same receipt as its external `.receipt.json`.
2. Inspect six 96-sample Cycles preview frames, including the full logo and final black. Render all 180 frames only after this visual review. Select an idle authorized OptiX or Metal GPU explicitly; never fall back or terminate foreign compute. Fresh output directories and per-frame SHA-256 receipts prevent accidental overwrite and stale frame reuse.
3. `endcap_film.py` validates the original film/source/audio/plate receipts, computes the bounded 840-frame demo mapping, and verifies complete row/file/cue/shuffle coverage. Recompose the original opening from its native plates, then append the resampled original demonstration and the new closing shot.
4. Preserve the original procedural PCM through 39 seconds. Add an original quiet resolving cue only in the six-second tail. Encode H.264 CRF 16 with the slower preset, AAC audio, and chapter metadata. Full decode both delivered MP4s and check exact frame counts, duration, picture profile, audio, and final fade.
5. Recheck source hashes after use. Record source/worker hashes, final output hashes, camera/geometry review, exact commit, decode logs, coverage counts, and stopped processes in external receipts and the existing coordination log.

Run from `/Users/bentaylor/Documents/ChatGPT/JopeDime/blender-source/.agent/worktrees/blender-fused-first-render`:

```bash
rtk proxy /Users/bentaylor/Documents/ChatGPT/JopeDime/blender-runtime/Blender.app/Contents/MacOS/Blender \
  --background --factory-startup --python-exit-code 1 \
  --python tools/visualization/blender/veox_endcap.py -- build \
  --logo tools/visualization/assets/veox-logo.svg \
  --font '/System/Library/Fonts/Avenir Next.ttc' \
  --output /absolute/fresh/external/scene/veox-endcap.blend

rtk proxy /Users/bentaylor/Documents/ChatGPT/JopeDime/blender-runtime/Blender.app/Contents/MacOS/Blender \
  --background --factory-startup --python-exit-code 1 \
  --python tools/visualization/blender/veox_endcap.py -- render \
  --scene /absolute/fresh/external/scene/veox-endcap.blend \
  --receipt /absolute/fresh/external/scene/veox-endcap.receipt.json \
  --output-dir /absolute/fresh/external/preview \
  --frame 1 --frame 28 --frame 55 --frame 95 --frame 170 --frame 180 \
  --samples 96 --device METAL

rtk proxy python3 tools/visualization/endcap_film.py --help
rtk proxy python3 -m unittest discover -s tools/visualization/tests
rtk git diff --check
```

For full rendering, pass each frame 1–180 once via repeated `--frame` arguments. Disjoint shards require a verified merged receipt covering every frame exactly once. The compositor requires the original 42-second demo and receipt, approved film receipt and PCM WAV, original opening plates and their directly bound render receipt, the opening scene receipt, and closing plates/scene receipts. The aggregate opening receipt has a different hash from the direct worker receipt; use the one named in the approved film receipt.

Stop on a missing or changed source hash, overlapping/missing frames, an unavailable GPU, altered logo shape, a lost code row or cue, a source jump larger than the approved cut, audio clipping, failed decode, or timing mismatch. Never weaken the checks to complete an encode. Preserve failed attempts and correction evidence. Completion means both actual videos pass and are delivered, rendering is stopped, and the source changes and review evidence are recorded in PR #2114.
