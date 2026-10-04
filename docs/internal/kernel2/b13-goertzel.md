# Sparse Goertzel bank (B13 / wire 61)

`goertzel_bank` is a second v2 FeatureWithinRow row stage beside B12's savgol, in
the same opt-in `feat-math.feature-row.experimental.v2` program family (program
subtype 3, `feat-math.feature-row-model` envelope). A program that holds a bank
declares feature word `32|128`; savgol-only and stage-free programs keep B12's
feature 32, model identity fields, recipe bytes and fitted C++ bytes exactly
(pinned by digest against the B12 head). Bank models carry their own grammar
contract digest and the emitter identity
`feat-math.feature-row.recipes.v2+wire58+wire61`. The frozen registry, exact
scalar codec, default grammar and existing artifact bytes are unchanged.
Source additions to the shared planner/emitters mint an announced evaluator
source identity, recorded as an identity-only successor at publication.

A bank holds one to 32 normalized frequencies `f` in `[0, 0.5]` (cycles per
feature position), kept in the supplied order with duplicates; a signed zero is
stored as `+0`. For each bin the stage outputs the squared DFT magnitude
`|sum_n x_n exp(-2 pi i f n)|^2` of the row, so its output width is the bank
length and the next stage (or the ridge head) sees that width. Input width is
at most 512; the standalone kernel also accepts an empty row (one `+0` per
bin). SampleTime, timestamps, group state and cross-observation filtering are
refused; a SampleTime bank needs its own later admission.

Geometry-only preparation, before training or export, freezes per bin
`cosine = cos(2 pi f)` and `sine = sin(2 pi f)` from the in-crate portable
fdlibm kernel (`angle = TAU * f`, one multiplication), so the bits are the same
on every host and restore can recompute and compare them. The endpoints are
exact: `f = 0` gives `(+1, +0)` and `f = 0.5` gives `(-1, +0)`. This replaces
the platform-libm preparation of the B13 design notes, which would make recipe
bytes and restore depend on the host libm. All artifacts embed the bits and
call no trigonometric function.

Frozen arithmetic, per bin and per row: protect each input (non-finite to `+0`
with NONFINITE_INPUT, `|x| > 1e12` clamped with MAGNITUDE_CLAMPED);
`c = 2 * cosine`; with `q1 = q2 = +0`, for each input in increasing column order
`term = c*q1`, `sum = x + term`, `q0 = sum - q2`, `q2 = q1`, `q1 = q0`; then
`real = q1 - cosine*q2`, `imaginary = sine*q2`,
`power = real*real + imaginary*imaginary`. The component finalization cannot
cancel to a negative value and keeps near-DC power that the traditional
`q1^2 + q2^2 - c q1 q2` form loses (width-512 alternating row at `f = 1e-9`:
`2.587e-12` against an 80-digit DFT, where the traditional form returns 0). A
non-finite power is `+0` with NONFINITE_SANITIZED and a power above `1e12` is
clamped with MAGNITUDE_CLAMPED; a power is never `-0`. Every bin carries the
row's input faults and its own output fault; no new fault bit is allocated. C++
builds require `-ffp-contract=off -fno-fast-math`.

The bank has its own removable search switch, `GoertzelProfile`, separate from
B12's `ExperimentalProfile`: seeded Grow inserts one to four frequencies from
the exact grid `k/128`, Change inserts, removes or replaces one bin and always
alters the bank, and Drop removes a bank. Its prior gate is bounded at 0.25.
Disabling the switch prevents generation while parsing and removal stay
available. B12's savgol edits now pick only savgol stages; a savgol-only
program draws exactly as before.

Focused acceptance: `cargo nextest run -P pr -p feat-math --locked --test
kernel2_closure -E 'test(goertzel_v2::) | test(savgol_v2::)'` and the kernel
unit tests (`--lib -E 'test(goertzel)'`). They cover an independent DFT oracle,
analytic empty/one-element/endpoint cases, the near-DC references, protected
inputs and saturation, canonical restore, envelope and feature-declaration
refusals at every ingress, unknown wires, row independence, the shipped-text
lint, GP create/change/remove (including mixed savgol and bank programs) with
compiled Rust, Python and C++17 recipes and fitted C++17 bit-equal to native,
and the pinned boundary corpus `tests/fixtures/math_boundary/v2/goertzel.json`
(173 values) executed in all three languages. `JOPE_ROW_CPP_QUALIFY=1` adds
clang++, C++20 and ASan/UBSan.

The fixture utility lane (`goertzel_utility.json`) uses six synthetic spectrum
tasks x five seeds: rows of width 64 with a target tone at a fixed frequency
(random amplitude and phase), nuisance tones and noise, labelled by the target
amplitude squared. A one-bin bank and the existing integer-bin DFT magnitude
stage (`fft_mag`, 32 bins) each feed the same ridge head on the same 96 training
rows and are scored on the same 64 held-out rows. The bank's held-out MSE was
about 0.12 to 0.32 of the reference on five tasks and about 0.69 on the
three-nuisance task; the test asserts only that the bank is not worse on any
task and records the ratios. It is synthetic numerical utility evidence.
Real-campaign equal-budget G-A..G-E qualification, outer-fold generalization
and Baseline promotion remain held. Predictions keep the inherited ridge head
semantics, including its prediction clamp.

No feat-math dependency is added. JavaScript is explicitly unsupported for the
profile.
