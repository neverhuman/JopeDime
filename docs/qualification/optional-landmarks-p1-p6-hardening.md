# Published P1/P2 preparation and P6 verification hardening

Implementation source `64db44cc11f791c0e8442acaae4e9360d0628ad4` pins published Dope
`2535a36c0e7ced2a17a0b06818efaeb17f7a2d47`. Explicit Direct/JopeDL consumers use the
uncapped preparation API. The public legacy helper continues to use Cap32V1. Product preparation
checks the exact raw class vocabulary, every encoded target and physical row ID. It refuses
rounding changes, merged classes and binary threshold mappings that change class codes.

Product scope also binds every original target's bits. This closes a numeric binary case in which
`[0,1]` and `[-1,1]` could previously share the upstream preparation digest. Receipts and component
requests carry optional recipe `jope.dope-product-preparation/lossless-v1`; the marker participates
in admission scope and is absent from legacy scope serialization. Four schemas accept it.
The checkpoint, 512-coordinate equation and existing low-level consumer recipes are unchanged.

Eight new preparation tests pass, including K=3/32/33, exact fractional labels, row order/IDs,
numeric and categorical inputs, refusal cases and stable distinct source hashes. An ordinary
CPU product session emits finite Dope512 for K=33. This establishes functional execution,
not wider multiclass predictive quality.

The CPU library suite passes 79 tests with one artifact guard; ordinary integration passes 13
with five guards. Three selected learned-product tests and the selected projection test were
then executed explicitly and passed. CLI/pipeline propagation passes four tests; appliance
fit/fit-predict passes three; schemas and dependency fences pass six and three respectively.
Seven affected packages pass all-target CPU Clippy. These counts overlap and are not summed.
Historical whole-bundle artifact paths remain outside this selected-asset qualification.

The reserved RTX 3090 run passes all three tasks, resident raw/context/fusion execution, the
decoder bypass and exact terminal progressive/full equality. Its 21 CPU-oracle comparisons have
maximum absolute error `6.914138794e-6`, below `3e-5`. GPU 0 was released after the bounded run.
This is numerical/execution evidence; the 12.55-second test window is not a latency benchmark.
Exact commands, hashes and source identities are in the [execution record](optional-landmarks-p1-p6-hardening.json).

P6 source commits `5d449917d8bc7a0b28cbbe2733c0613e52a9c685` and
`80571c58baf53dba5de928cc6b6683b3722f1d27` fix two actual verifier defects:

- Legacy/v2 loaders previously ignored requested lineage, intent, writer-attempt and window
  identities. A runtime regression failed before the fix. All 11 checkpoint tests now pass;
  default old-format loads and unknown-format errors retain their behavior. New fixtures
  serialize initialized step-zero CPU state and perform no optimizer updates.
- Read-only native proof verification previously created missing evidence, staged files during
  exact replay and retained conflict files. All three regressions failed before the fix; all
  18 native-graph tests now pass. Read-only verification requires exact retained bytes and
  refuses missing, conflicting or unsafe files. The mutable retention path is unchanged.

An initial external checkpoint-test module path failed compilation and was corrected before
the behavioral regression was demonstrated. Its log is retained separately; it is not the
failure that establishes the loader defect.

The original [one-shot diagnostic run](optional-landmarks-p6.md) remains the sole P6 acquisition.
These changes install no keys, grants, adoption, service or production Snapshot root. Production
publication still requires a v3 committed-credit producer and authenticated source/command,
I27 input-root, lineage/witness, replay and compatibility-pin bindings. The numeric optimizer
epoch also needs an explicit mapping to the native epoch digest. A committed fit receipt or
hash-shaped JSON fields cannot substitute for these proofs.

ReleasedV1 remains the product checkpoint. Trained `ecded022` remains an unqualified SDK candidate.
The reported N=300 point criterion passes on its available paired cohort; binary uncertainty,
exclusions and denominator limits remain visible. The registration review packet records the
fixed artifact hashes and remaining owner export/preparation/rights fields. Stale sampled-arm
metadata in the published Dope candidate JSON was relayed to its sole owner for reconciliation.
Final registration, consumer predictive quality, a sealed trained package and matched latency
qualification remain open. No governed redistribution or serving promotion is established.

The owning identity-only regeneration and separate comparison pass 13 tests each; historical
verification passes all three. Only evaluator identity and lineage header fixtures changed.
A fourth append-only transition preserves every prior record. The DSL ABI, known identities,
numerical/emission fixtures and seeded outputs remain unchanged.
