# Terminal preview notes

The [animated preview](terminal-preview.gif) is a designed example, not a
recording of an implemented UI. Every frame carries `DESIGN PREVIEW` and
`SYNTHETIC DATA`. All times, rates, scores, identifiers and progress values are
illustrative. It makes no model-quality, speed or compiler-qualification claim.

The animation has two separate scenes:

1. **Native CSV fitting:** `jopedime fit-predict` reads train/test inputs, reports
   row/feature and thread budgets, advances through training cross-validation,
   then refits, predicts and commits outputs. The final receipt/output phase
   precedes the success label. The proposed `--ui progress` flag is shown as
   part of the design. The native scene has no generated-source pane.
2. **generated-source viewer:** proposed `jope source watch --ui tui` attaches to the
   existing loopback WebSocket. A bounded pane switches between Rust, Python,
   C++17 and JavaScript, shows occurrence/sequence identity and compiler state,
   and updates source in place. New occurrence source appears before its
   compiler check completes. The footer explains follow, pause-display and
   detach; every occurrence is retained by the separate source log.

The selected visual language uses a dark neutral background, cyan activity,
green labeled success, amber pending state, red labeled errors and quiet gray
metadata. Borders and spacing separate phases and source from diagnostics.
No meaning depends on color alone. The workplan requires light/monochrome,
ASCII and narrow-terminal variants in the implementation; this preview shows
only one 1200×780 presentation.

The short source functions in the picture are handwritten illustrative snippets,
not emitted or compiled artifacts. Their compiler labels are fictional. The
generator performs no training, compiler execution or network access.

## Reproduction

From the repository root, with Python 3, Pillow and DejaVu Sans Mono available:

```sh
python3 docs/terminal/render_preview.py
```

The default output is the adjacent GIF. `--output`, `--still`, `--font` and
`--bold-font` override paths. The default fonts are the distro-installed DejaVu
Sans Mono regular and bold; fonts are not embedded or copied into the repo.
The DejaVu project distributes these under its font license; use the matching
fonts and Pillow version in the manifest for byte-for-byte regeneration.

The generator writes a sibling JSON manifest containing the base source SHA,
preview disclaimer, image properties, frame timing, Pillow version, font hashes,
generator hash and GIF hash. The GIF size budget is 1 MiB. It keeps a fixed
palette and fixed timing and has no timestamps, randomness or machine paths in
the committed manifest. The optional still is an inspection artifact, not a
second committed screenshot.

This documentation PR validates the renderer and source-backed commands; it
does not claim that the appliance build or the proposed UI acceptance tests
ran on the authoring host. The authoring host's workspace disk was full, so
no new Rust build was attempted. Follow the workplan's T16 gate to replace the
concept with a real, reproducible terminal recording after implementation.
