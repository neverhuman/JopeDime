# Causal difference ordered program and continuation routing

B08 remainder slice 1 adds lossless group-local cursor/history snapshots,
an explicit version-2 ordered program, and train-only fitting followed by causal
prediction through the existing numeric Ridge downstream model. Wire 59 uses program
variant 4 and required feature bit 64. The new program preserves width 1–512,
order 1–4 and fixed finite positive step. Legacy programs, models and emitters
retain their existing paths.

The initial fitted closure is an untransformed numeric Ridge pipeline with finite
nonnegative regularization. Other opaque legacy downstream programs preserve
their codec round trip and return `Unsupported` when asked to fit. Restored models
require the exact compatibility input layout, finite Ridge coefficients, matching
widths and positive finite scales. A fit that overflows its fitted coefficients
returns an error before exposing a model.

The fitter accepts exactly the training prefix and its finite labels. A declared
cut bounds future prediction; no held-out rows or labels enter fitting. Prediction
returns the stage's per-column fault words through `predict_with_faults`. A batch
commits its cursor/history only after all rows and downstream prediction succeed.
An empty prediction batch returns empty values/faults without consuming state.

Snapshots bind the ordered input columns, axis/causality/fit scope and complete
fold identity. They store raw floating-point words, including signed zero and
nonfinite history payloads, bounded group histories, timestamps, ordinal and
training counts. Decoders check canonical group order, counts, dimensions,
parameters, identity and complete input consumption. Model envelopes carry the
program identity, numerical policy v2, profile/emitter feature identity and output
schema; fitted floating-point values use integer words to preserve exact replay.
The embedded legacy downstream program retains its historical numerical policy.

Acceptance: `cargo test --locked -p feat-math --test finite_difference_v2
--test ordered_axis_contract`. The `routing::` module adds eleven cases to the
existing mapped B08 target. Tests preserve
refusal reproduction and cover snapshots, malformed/truncated codecs, IID and
identity rejection, transactional batches, independent downstream fitting,
future poison, boundary faults and all four orders on one ordered boundary file.
Evidence: `/home/ubuntu/jope-evidence/kernel2/bt/B08_REMAINDER/`.

The shared baseline at main `2e75f00cc8abe2323fa6e3c5e2d0965c35b68bc3`
executed PR 2108 (29 skipped) and main 4190 (66 skipped); its 38 evidence hashes
were verified before reuse. `just required` exits 101 on the owner-accepted
unchanged lint floor and is not recorded as passing. New owned lint must pass;
workspace findings must remain within the baseline.

The remaining per-operation definition of done stays explicit: three-language
training recipes, fitted C++ ordered export, profile/generation admission, EEC
creation/mutation/removal and canonical round trips, shortest-tree/token/byte
comparison, at least six training datasets with five seeds at equal evaluation
budget, and G-A…G-E utility/promotion evidence follow in the next two B08 PRs.
Wire 59 remains closed to generation in this slice. No Baseline promotion or
complete B08 claim follows from native routing and state replay alone. Existing
historical compatibility, IP scrub, dependency freeze and exact-head CI/review
remain required gates.
