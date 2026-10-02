# X1: fitted C++17 export from the pipeline

`jope pipeline run --emit cpp17-fitted` refits the selected genome on the verified
training partition through `fit_legacy_numeric_bound`, then uses the existing
`emit_fitted_cpp17`. Search and selection do not read sealed test labels. This is
a fitted inference artifact; the train-and-predict recipe remains separate.

Artifacts are `emit/cpp17-fitted/model.cpp`, `model.codec.json` and `standalone.cpp`.
The receipt records each file's bytes/hash and an executable strict C++17 build.
The shared runner accepts a header and finite numeric feature CSV, writes a
`prediction` column, and refuses malformed rows or input/output aliasing. Its
literal bytes are unchanged from the existing `jope emit` route. Recipe
`model.cpp` stays alongside the fitted directory.

The default and `all` remain the five legacy recipes. No profile/default change,
numerical function, model/codec format, old fixture, emitter helper or evaluator
identity input changed. Existing receipt fields suffice; no schema bump.

## Local acceptance

- Fresh full baseline at `601a0aacd7acbcfee100eabc201119125ee87a41`: PR1981/29,
  main4061/66, Python231/2 deselected, golden16/16, fitted sanitizer18/18 and C++20
  18/18, original O3 plus added strict O2 emitter pass. Named exclusions unchanged;
  known Clippy failures are a zero-new diagnostic ratchet.
- Before source edits, the clean private branch rebased onto freshly fetched
  `334fb9cb43834b1b278286526e13392bccfefaf2`. The intervening changes are only two
  pinned release-workflow actions; product/runtime/CI gate inputs are unchanged.
  The baseline is recorded at its actual SHA, not claimed as executed on334f.
- Tests-first CLI reproduces unknown-target refusal (1failed, exit101); library
  API compilerred executed0tests. Final focused8library+1CLI pass with12required
  g++13.3 C++17 O2 builds. Assertions include fitted source/codec bytes and exact
  prediction bits for symbolic/map/derive, target encoding, histgbm/histgbm2/gpr/
  ksmooth and a real E3 expression-growth child; malformed training/inference;
  legacy coexistence; selected training-partition refit; receipt tamper refusal.
- Acceptance commands:
  `cargo test -p jope-pipeline --locked --lib emit::fitted_tests` and
  `cargo test -p jope-cli --locked --test pipeline_cli pipeline_cli_emits_fitted_cpp_and_records_executable_build -- --exact`.
- Frozen71paths and38identity inputs must remain byte-identical. Full after gates,
  lane audit, exact-head automatic/manual CI and independent approval are pending
  at source commit; the external lifecycle receipt records their eventual result.

Evidence: `/home/ubuntu/jope-evidence/kernel2/bt/B03_PIPELINE_FITTED_CPP/receipt.md`.

## Remaining qualification

This closes the opt-in X1 pipeline export seam. Kernel2 shipping search/profile
selection, fault-policy CLI, v4 telemetry, fitted default for kernel2 runs,
EEC creation/change/removal and later ordered-state/full recipe closure remain
separate prerequisites. No full search, portability, release or default-flip
qualification is claimed here.
