# Historical render source snapshots

These 12 UTF-8 source snapshots preserve unique design/still/batching work from
the delivered film campaign. Their `.py.txt` / `.sh.txt` names deliberately make
them **non-entrypoints**. They are byte-exact historical code, not generated
model output or media; original paths and SHA-256 are in the parent manifest.
Read them as design/provenance references, not authorization to execute them.

The snapshots retain dated host paths, frame choices and source pins. Earlier
still launchers predate the stronger current GPU admission guards. Do not rename
and execute them unchanged. Their reusable worker logic is maintained in:

- `tools/visualization/blender/jain_cinematic.py --spatial-layout NAME`: all ten
  initial and biological layouts; `jain_spatial_layouts.py` preserves stage order.
- `tools/visualization/blender/render_animation.py --case-id case-a
  --frame-start 89 --frame-end 89`: one chosen layout's still with an explicitly
  selected GPU and fresh external output directory.
- `tools/visualization/intelligence_wave.py`: source/table/icon/wave composition.
- `tools/visualization/orchestration/compose_dendrite_film.sh`: final Dendrite
  assembly and binding/pixel verification under external custody.

`build_stills` snapshots batch scene generation, `render_stills` batch the
historical still renders, and `compose_review` creates labeled comparison sheets.
`verify_layouts` binds those sheets to stage/connection invariants. `prepare_scene`
and `finish_film` retain the original capture-to-scene orchestration; they are not
a new instruction to rerun inference. `compose_preview` and
`check_connector_geometry` retain the single-vertex panel-contact design checks.

Excluded duplicates: superseded V1 intro orchestration, before-fix copies of
tracked render workers, ad-hoc QA probes whose acceptance rules are already in
current tests/receipts, emitted model programs, copied training-source custody,
and the discontinued GPU-inference benchmark packet. None supplies an additional
current Blender scene, shader, palette, audio or assembly implementation.
