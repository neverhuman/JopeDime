# Rendering controllers and production assembly recipes

These small source files preserve the orchestration that was previously kept only
beside the external render artifacts. They drive the existing repository Blender
workers; they do not contain scenes, neural weights, datasets, fonts or video.
`external-source-manifest.json` records the original and adapted SHA-256 for each
production script. The original files and historical receipts remain unchanged.

## Status and safety

The delivered film is the 1080p V7 endcap assembly. UHD rendering is **stopped**;
checking in the UHD controllers does not restart or authorize it. No render or
remote command was executed while importing these controllers into Git.

All CLI help and Python module imports are side-effect-free. Actual commands
require `JAIN_ARTIFACT_ROOT`, an existing absolute external directory corresponding
to the original `h256` custody tree. The product checkout, its common Git checkout,
and their worktrees cannot be output roots. Symlinks are resolved before this
check. The unrelated empty workspace-container Git directory is not the product
repository and does not invalidate the sanctioned external custody location.

Rendering is disabled by default. After a separately authorized run and a fresh
GPU availability check, explicitly set `JAIN_ENABLE_REMOTE_RENDER=1` and/or
`JAIN_ENABLE_LOCAL_RENDER=1`. Local rendering also requires `JAIN_BLENDER`.
The stopped UHD paths additionally require `JAIN_ENABLE_UHD_RENDER=1`. These are
admission guards, not substitutes for ownership or authorization. No script
installs software, changes drivers, terminates another user's job, or falls back
to processor rendering.

Remote defaults retain the two originally approved SSH aliases, `xbabe3,xbabe1`;
`JAIN_RENDER_HOSTS` may select exactly two distinct aliases. Strict host-key checks
remain enabled. Remote runtime/custody defaults and the SHA-256 pins
remain the original approved values in `run_intro_render.py` and
`gpu_case_worker.sh`. Provisioning is external: scripts fail if the pinned Blender
runtime or required worker is absent. Do not change a pin simply to pass a check.
The Dendrite worker is deliberately pinned to the historical render revision
`ca3c3f8e8866c01ffc54048fc1cee2e4321e213fef555b873b76579572f4dbf5`.
It predates optional UHD changes to the current worker; reproduce with that pinned
worker or qualify a new render revision rather than relabel historical evidence.

The source configuration is read at process import. Set environment variables
before launching the process, not after importing a controller.

## Source closure

| Entry point | Responsibility / dependencies |
| --- | --- |
| `run_intro_render.py` | Original R2 intro GPU shards; tracked `blender/intro_render.py` and `blender/render_animation.py`. Also supplies pinned remote transport helpers. |
| `run_veox_render.py` | VEOX previews/full shards and optional Metal tail; imports the intro transport helpers and tracked `blender/veox_endcap.py`. |
| `run_dendrite_render.py` | Ten-case 1080p loop staging, GPU admission, transfer and per-frame verification; stages `gpu_case_worker.sh`. |
| `compose_dendrite_film.sh` | Complete original 42-second demo assembly, source/scene validation, decode and pixel QA; calls tracked `intelligence_wave.py`, `verify_bindings.py`, and `verify_neural_cut.py`. Defaults to the checked-in NVIDIA artwork. |
| `build_render_plan.py` | Offline mapping from the approved R2 timeline to its sparse UHD plates. Writes a fresh plan only; no GPU use. |
| `render_uhd.py` | Stopped UHD probe/chunk scheduling, 16-bit plate validation, publication and recovery. Imports the plan and intro transport helpers. |
| `render_metal_tail.py` | Stopped bounded local UHD sidecar; leaves remotely claimed chunks alone and checks probe quality before rendering. |

Palette, typography, program-source scrolling, neural waves, procedural audio,
video assembly, Blender geometry, logo parsing and scene generation already live
in `tools/visualization/` and `tools/visualization/blender/`. Those delivered
implementations were not changed by this source-preservation work.

The external tree keeps the same families: `jain-intro-v2`,
`jain-veox-endcap-v1`, `jain-final-4k-v1`, and
`variable-attribution-v1/{capture,dendrite-final-v1}`. Existing input receipts,
model/capture identities, packed scenes, original fonts, render runtimes and
historical plates remain prerequisites. The scripts never synthesize missing
scientific inputs or silently overwrite completed output families.

## Commands

From the repository root, inspect help without any configuration or hardware:

```sh
rtk proxy python3 tools/visualization/orchestration/run_veox_render.py --help
rtk proxy python3 tools/visualization/orchestration/run_intro_render.py --help
rtk proxy python3 tools/visualization/orchestration/render_uhd.py --help
rtk proxy sh tools/visualization/orchestration/compose_dendrite_film.sh --help
```

Read-only receipt/hash validation against existing external inputs:

```sh
JAIN_ARTIFACT_ROOT=/absolute/external/h256 rtk proxy python3 \
  tools/visualization/orchestration/run_veox_render.py check \
  --scene-dir /absolute/external/h256/jain-veox-endcap-v1/scene-v7
JAIN_ARTIFACT_ROOT=/absolute/external/h256 rtk proxy python3 \
  tools/visualization/orchestration/run_dendrite_render.py check-inputs
```

For an independently authorized render, use the same external root and explicit
render admission flags, then select `preview` or `full`. Endcap V7 uses
`--scene-dir .../scene-v7 --full-output frames-v7 --samples 96`, optionally with
`--expected-scene-sha256` and `--expected-worker-sha256`. Existing output families
are intentionally refused. Use new custody when making a new revision.

The 1080p final stitch remains `tools/visualization/endcap_film.py`; its complete
inputs and verification contract are in `docs/plans/blender-fused-first-render/ENDCAP.md`.
The intro uses `intro_film.py`; native UHD composition uses `uhd_demo.py` and
`intro_hq.py`. No new rendering is needed merely to retain this source.

## Deliberate exclusions

Twelve unique historical design/still/batching helpers are preserved byte-for-byte
under `historical/` as non-entrypoint source snapshots. Its README gives the
current worker equivalents; the shared manifest retains every original source
path/hash. Superseded V1 intro controllers, before-fix duplicates, ad-hoc QA probes,
emitted model files and copied training sources remain historical custody. Their
reusable scene/layout/compositor behavior is already tracked. The discontinued
GPU-inference packet is not rendering code and is outside current inference scope.
Private captures, model weights, `.blend` scenes, PNG frame sequences, audio,
movies, copied Blender binaries and machine-local fonts remain external.

Validation: `python3 -m unittest discover -s tools/visualization/tests` includes
configuration, output-boundary, disabled-execution, dry-help and source-inventory
checks. No full rendering or remote qualification is claimed by these tests.
