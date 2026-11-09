# JAIN — The next move

The opener turns a Go stone into an algorithm tile, shows an illustrative training feedback loop, and reveals the captured Dendrite network. Its final second introduces the moving demonstration. The standalone opener is 12 seconds. The assembled film is 45 seconds at 1920 × 1080 and 30 fps, following the user's decision to accelerate the demonstration. Revision 2 strengthens the object's size, framing and causal relationships in response to the supplied render reviews.

## Source concepts and decisions

Read the four supplied concepts in this order.

1. `spec1.md` supplies the stone-to-algorithm story.
2. `JAIN_Blender_Intro_Engineering_Spec.md` supplies the training/inference distinction and source-frame continuity contract.
3. `JAIN_Intro_Blender_Spec.md` supplies the reflective board, cyan reveal, and board-to-network transformation.
4. `jain_intro_blender_engineering_spec.md` supplies additional lighting and framing concepts.

These documents are creative references. The produced film uses the actual saved Dendrite scene and measured demonstration. The Go position and training beat are illustrative. The opener makes no historical match, comparative cost, optimality, or benchmark claims. The measured benchmark arrives with the demonstration.

The user subsequently supplied `JAIN_Intro_Render_Review_R2.md` and `JAIN_Intro_Render_Review_v1.md`, and the concept of innovation as search. Adopt R2's centered framing, substantial algorithm tile, board-free feedback circuit, larger headings, and full inference composition. Keep the explicit 12/45-second user timing instead of the reviews' 15-second alternative. Keep the training qualifier and the approved Dendrite geometry; the older rectangular network reference does not supersede that choice. Use the exact demo wordmark, panels and benchmark. The opening line becomes “Innovation is a search.” The sequence resolves toward one proposed algorithm without claiming an exhaustive or globally optimal search.

The prior scene already had 19 lines in each direction and nine star points. Revision 2 makes them visible through framing and material contrast. It preserves the v1 render evidence in `h256/jain-intro-v1`; all new scenes and media belong to `h256/jain-intro-v2`.

## Timeline

| Time | Picture | Copy |
| --- | --- | --- |
| 0–3.27 s | Full warm satin Go board; the central ivory stone lands and becomes the focal event | Innovation is a search. |
| 3.27–5.27 s | The same contour becomes a large face-on dark algorithm tile with braces | Every move. An algorithm. |
| 5.4–7.8 s | Board is gone; problem → candidate → test → problem circuit returns feedback and replaces the candidate | Test. Learn. Play again. / Algorithmic self-play · training |
| 7.8–10.53 s | Candidate grid unfolds into the captured network; real input arrives before the forward pulse, then the actual source panel | One inference. One algorithm. |
| 10.53–11 s | Full source composition settles, including its exact brand and benchmark | Headline clears; actual demo HUD |
| 11–12 s | Unmodified moving source footage continues | Actual demo HUD |
| 12–45 s | The same demonstration stream continues through all ten datasets | Actual demo HUD |

## Ownership and implementation

Use the existing `codex/blender-fused-first-render` worktree and draft PR #2114. All generated scenes, plates, audio, and videos live outside Git under `h256/jain-intro-v2` in the established render artifact root. Existing training, model, export, and other PR lanes are outside this task.

Read first `blender/jain_cinematic.py`, `blender/jain_display_network.py`, `blender/render_animation.py`, `intelligence_wave.py`, and the final Dendrite receipts.

The three disjoint implementation packets are `blender/jain_intro.py` for the scene, `blender/intro_render.py` for pinned GPU rendering, and `intro_film.py` for typography, original synthesized sound, source retiming, and final encoding. The root worker owns integration, this design record, and review. Avoid simultaneous writes to any packet.

The scene reads the saved Dendrite camera, floor, materials, and sampled neural geometry. The transformation ends at those captured coordinates. Rendering uses the verified Blender 5.2.3 runtime and explicit OptiX or Metal GPU selection. The first delivery uses the local Apple M4 Max through Metal because both remote GPUs acquired other compute jobs before launch. The worker does not fall back to processor rendering.

The 42-second source is mapped into 34 seconds with endpoint-inclusive nearest-frame sampling per chapter. Master frame 330 starts unmodified source playback. The preview uses that same first case's panels and source state before the moving handoff. Master frame 360 continues the same stream. Source mappings and maximum code viewport jumps are recorded so retiming can be checked for complete source coverage.

This is a fast visual overview. Every source row appears fully within the viewport, but 34 seconds is insufficient to read all ten programs. The original source packages remain the reading reference. Intro timing and the training illustration are editorial; the demo retains its captured measurements and source identities.

## Rebuild on the delivery workstation

Dependencies are Blender 5.2.3 with a supported GPU backend, Python 3.11 or newer with NumPy and Pillow, FFmpeg/FFprobe, the Avenir Next font on macOS, and the saved Dendrite scene/movie plus matching receipts. On another system, supply an installed font with `--font` and select `--device OPTIX`. Do not rerender on a GPU already occupied by another compute job.

Run from the claimed worktree. Use a new external output directory for every attempt; a previous run is evidence and must not be overwritten.

```bash
JAIN_ROOT=/Users/bentaylor/Documents/ChatGPT/JopeDime
JAIN_BASE="$JAIN_ROOT/blender-render-20261103/h256/variable-attribution-v1/dendrite-final-v1"
JAIN_OUT="$JAIN_ROOT/blender-render-20261103/h256/jain-intro-rebuild"
JAIN_BLENDER="$JAIN_ROOT/blender-runtime/Blender.app/Contents/MacOS/Blender"

rtk proxy "$JAIN_BLENDER" --background --factory-startup \
  --python tools/visualization/blender/jain_intro.py -- \
  --source-scene "$JAIN_BASE/scene/scene.blend" \
  --source-receipt "$JAIN_BASE/scene/scene.receipt.json" \
  --output "$JAIN_OUT/scene/intro.blend"

rtk proxy "$JAIN_BLENDER" --background --factory-startup \
  --python tools/visualization/blender/intro_render.py -- \
  --scene "$JAIN_OUT/scene/intro.blend" \
  --receipt "$JAIN_OUT/scene/intro.receipt.json" \
  --output-dir "$JAIN_OUT/frames" --frame 1:330 --samples 24 --device METAL

rtk proxy python3 tools/visualization/intro_film.py \
  --demo "$JAIN_BASE/final/JAIN-intelligence-wave-42s.mp4" \
  --demo-receipt "$JAIN_BASE/final/encode.receipt.json" \
  --plates "$JAIN_OUT/frames" --plates-receipt "$JAIN_OUT/frames/render.receipt.json" \
  --intro-scene-receipt "$JAIN_OUT/scene/intro.receipt.json" \
  --output-dir "$JAIN_OUT/final"
```

`--preflight` on the compositor validates the source and retime without rendering. The worker supports individual `--frame N` arguments for previews. The compositor supports zero-based `--preview N` arguments against those plates. The full build writes the standalone opener, assembled film, original procedural sound, chapters, full decode logs, and `intro-film.receipt.json`.

## Validation and handoff

Inspect the stone landing, algorithm contour, training loop, network emergence, and final handoff before the full render. Verify the scene's source hashes, final target geometry, packed assets, bounds and deterministic frame controls. Stop a render if a required source is missing, the saved scene identity changes, the requested GPU is unavailable, or another host job conflicts.

Run `python3 -m unittest discover -s tools/visualization/tests` and `git diff --check`. Validate every transferred frame against its render receipt. Fully decode both MP4s and check 360/1350 frames, 30 fps, 1080p, 12/45 seconds, original sound levels, all ten dataset chapters, monotonic source mapping, complete source viewport coverage, and no handoff restart. Record exact commands, source and output hashes, review findings, and the stopped commit in the existing coordination log. Deliver the opener, assembled film, representative stills, and draft PR link.
