# Rust appliance implementation checklist

This is the execution index for [SPECIFICATION.md](SPECIFICATION.md). The
specification owns the detailed contracts and acceptance criteria;
[AMENDMENTS.md](AMENDMENTS.md) records where the implementation corrects or decides them.
Ticked items are implemented with the named evidence at the source state of the change that
ticked them; exact-head CI and substantive agent review decide whether that change merges. No
stage is closed: P1 and P2 have their CPU parts implemented, and nothing here qualifies a product.

Each implementation PR must name its stage, changed paths, full source SHA,
acceptance commands and results, evidence locations, and remaining requirements.
Follow the repository's [contribution contract](../../AGENTS.md), including claims,
one implementation PR at a time, exact-head CI, and substantive head-bound agent
review. The implementing agent may perform that review. A passed fixture,
implemented command, qualified backend, and promoted release are separate states.

## P0 — Inventory and contracts

- [ ] Refresh the source and open-PR audit; map every promised native/neural method
  to its existing Rust entry point, task support, limits, and exact asset dependencies.
- [ ] Commit the deployment ADR, profiles, model lock, backend inventory, limits,
  qualification policy, and versioned schemas listed in
  [specification section 7](SPECIFICATION.md#7-repository-layout).
  *Partly:* profiles, `models.lock.json`, `limits.toml`, `size-budgets.toml`,
  `dependencies-deny.toml` and four `schemas/jope.appliance-*.v1` schemas exist; the ADR,
  CUDA backend inventory and qualification policy do not.
- [ ] Record concrete tensor/config/vocabulary/preprocessing identities and rights
  for the complete method dependency graph; reject placeholders and incompatible
  context ABIs. Keep the accepted policy's Hyperion v5 arm distinct from Hyperion v7.
- [x] Identify prohibited, internal, or unresolved redistribution records without
  rewriting them as permission. Separate fixture qualification from restricted
  full-model qualification. *Evidence:* `models.lock.json` blockers (A7);
  `crates/jope-dist/tests/closure.rs`; fixture tensors are generated, never real weights.

Exit evidence: a reviewed method/asset/rights inventory with explicit unresolved
items, concrete contracts, and a source-pinned implementation plan.

## P1 — Linkage and one-file feasibility

- [x] Introduce the minimal `jope-assets`, `jope-appliance`, and `jope-dist` package,
  verification, fixture prediction, `doctor`, and method-reporting paths.
  *Evidence:* `crates/jope-appliance/tests/single_file.rs`; `test-clean-host.sh`.
- [x] Implement the versioned envelope and signature contract in
  [section 10](SPECIFICATION.md#10-one-file-package-and-signature-format), including
  bounded parsing/decompression and an independently verifiable trust anchor.
  *As amended (A5, A6):* manifest-signed footer, per-asset authentication before
  decompression, `docs/release/signing.pub` as the anchor, `jope-dist verify`.
- [x] Cover tampering, invalid signatures, truncation, invalid offsets, wrong
  identities, and reproducible unsigned packing. Fixture keys must never authorize
  production model IDs or distribution profiles. *Evidence:*
  `crates/jope-assets/tests/{envelope,decode_bounds}.rs`,
  `crates/jope-dist/tests/reproducible_pack.rs`; the development key is trusted only by
  `dev-keys` builds and signs only `development`-scope profiles.
- [ ] Resolve Candle/cudarc linkage with a narrow pinned patch if required;
  additive Cargo features must not leave mandatory CUDA loader dependencies.
- [ ] Execute the **same tiny GPU-capable artifact** on both a host without CUDA
  and a driver-only GPU host. Inventory any bundled user-space libraries and their
  rights, and record exact file hashes and observed backends.
- [x] Route the new tests and receipts through the existing CI plan, inventories,
  capability matrix, and aggregate checks as soon as executable paths land.
  *Evidence (A4):* product test scope, `coverage.json`, the regression test inventory,
  capability-matrix rows, `ci-plan.py`/`ci_plan.rs` gates, and the `e2e-smoke` appliance step
  whose receipts the existing aggregate consumes.

Exit evidence: real package negative tests and both clean-host execution receipts.
An unavailable GPU runner leaves GPU feasibility pending. CPU scaffolding alone
is not a completed full-product deployment.

## P2 — Shared native runtime

- [x] Introduce `jope-runtime` table, method, budget, selection, receipt, and fitted
  state modules; extract the native roster from the benchmark-facing adapters.
  *Evidence:* A2; `crates/jope-bench` panel test `the_roster_is_the_appliance_registry`.
- [x] Preserve the frozen preprocessing and `feat_math::advanced_v3` numerical
  contracts, training-partition boundaries, seeds, and task-specific support.
  *Evidence:* moved code, unchanged pipeline (292) and panel (75) library suites.
- [x] Implement fit/save/reload/predict for every promised native recipe and bounded
  AutoML selection. Persist all transforms, schema, class order, and fitted state.
  *Evidence:* `crates/jope-runtime/tests/offline_fit_predict.rs` (bit-identical reload for
  all 26 task/recipe pairs; `jope.runtime-select/v1`, A10).
- [x] Enforce a target/profile-specific dependency fence excluding research/cloud
  services, Python, LibTorch, external training helpers, and runtime compilation.
  *Evidence:* `dependencies-deny.toml`, `crates/jope-dist/tests/dependency_fence.rs`, the
  `NEEDED` check in `build.sh`, and the `fitted-cpp` gate (A1).

Exit evidence: complete native roster coverage, reload parity, bounded execution,
dependency audit, and unchanged historical behavior tests.

## P3 — Accepted model dependency closure

The internal r882 successor has a production embedded-loader/Engine bridge:
[R882_RUNTIME.md](R882_RUNTIME.md). The older step-60000 generative-policy items
below remain open. Normal candidate packing is separate from qualification; this
bridge closes neither P3 as a whole nor any release gate.

- [ ] Add verified buffer/source loader seams to policy and encoders; retain strict
  original identities and load every required context projection and transform.
- [ ] Implement deterministic dependency resolution, safetensors inventory, and
  conversion receipt checks in `jope-dist`. Reuse accepted tensor bytes when possible.
- [ ] Load the accepted encoder → policy → native fit path from the embedded payload
  and compare preprocessing, assembled context, fixed-prefix logits, ranking heads,
  and predictions against independently retained references.
- [ ] Bind exports, assets, qualification, and fitted state to their exact model and
  semantic identities. Include any heads required by the selected ranking algorithm.

Exit evidence: a complete accepted CPU path with no runtime downloads or sidecar
model files, plus byte-identity and component/end-to-end parity receipts.

## P4 — Native neural alternatives

- [ ] Add the Starlight, TabICL, and Chimera facades and runtime adapters listed in
  [section 6](SPECIFICATION.md#6-native-alternatives-and-method-parity).
- [ ] Include each standalone expert and fusion dependency. Preserve task limits;
  the existing two-logit Chimera fusion does not establish multiclass support.
- [ ] Persist required in-context support rows/labels, preprocessing, class mapping,
  and all other fitted state; prove reload without another fit.
- [ ] Qualify every promised method/task/backend combination and fail a full-profile
  release when any required method is missing, unauthorized, or unqualified.

Exit evidence: per-task parity and complete executable method coverage for the
declared distribution profile, with rights decisions recorded separately.

## P5 — Packaging, performance, and GPU

- [ ] Implement lazy decode, bounded lossless compression, atomic verified caching,
  and corruption/concurrency handling without requiring a persistent cache.
- [ ] Implement explicit device selection, bounded automatic CPU fallback,
  same-binary worker isolation, and accurate device/fallback receipts.
- [ ] Qualify complete GPU execution and its CPU fallback, including missing or
  incompatible drivers, memory/resource limits, cancellation, and reload behavior.
- [ ] Measure final executable size, cold/empty-cache load time, populated-cache load
  time, warm throughput, p50/p95 latency, RSS, and VRAM on identified hardware.
- [ ] Apply LTO/stripping and per-component optimization using measured tradeoffs.
  Any lower-precision variant gets a new identity and separate quality qualification.

Exit evidence: exact-artifact backend qualification and a measured size/speed/memory
report against the limits in [section 17](SPECIFICATION.md#17-acceptance-and-qualification).

## P6 — CI and release integration

- [ ] Extend the existing CI workflow, applicability plan, test inventories, receipts,
  aggregate, trusted qualification lanes, and cache trust-domain separation.
- [ ] Reject missing, zero-test, cancelled, stale-source, wrong-attempt, and truncated
  qualification evidence. Report declared path exclusions as `skipped-by-path`.
- [ ] Extend `deploy/release/release.sh` and its tests while retaining the existing
  signing authority, source-export checks, host controls, and distribution gates.
- [ ] Build and sign a candidate once; qualify and publish those exact bytes.
  Produce SBOMs/notices, a signed asset inventory, and independent verification data.
- [ ] Replace the disabled release workflow only after the new release lane is
  provisioned and qualified; perform a dry run and download/readback drill.

Exit evidence: full exact-candidate release rehearsal, aggregate verification, and
downloaded artifact hash/signature equality. Publication remains a separate action.

## P7 — Documentation and promotion

- [ ] Update product quick start, supported targets, release verification, capability
  matrix, changelog, and version records to match the actual qualified behavior.
- [ ] Index every acceptance result by final full source SHA, model lock, signed
  artifact hash, command, hardware, run/attempt, and retained evidence location.
- [ ] Obtain required exact-head CI and substantive head-bound agent review; repeat the
  applicable checks/review after any source change.
- [ ] Qualify the merged candidate and publish only within its authorized scope.
  Verify the remote release and preserve rollback and verification instructions.

Exit evidence: [the complete definition of done](SPECIFICATION.md#22-definition-of-done)
is satisfied for one independently verifiable executable on every advertised target.
No stage may be closed solely because the proposed files or commands exist.
