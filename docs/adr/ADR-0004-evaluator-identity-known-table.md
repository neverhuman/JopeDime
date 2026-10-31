# ADR-0004 — Evaluator identity: accept known historical identities

**Status:** accepted 2026-09-18 · **Scope:** `crates/jope-campaigns/src/evaluation_v2.rs`

## Context
`JopeDimeEvaluatorIdentity.v1` hashes ~30 feat-math sources, the crate manifest and the workspace `Cargo.lock`
(`compute_implementation_sha256`). Its `validate()` compared a receipt's recorded identity with the *current
build* and rejected anything else, so every persisted `JopeDimeSemanticEvaluationReceipt.v2` became invalid the
moment feat-math was reformatted or the lockfile changed — which the single-workspace port does by construction
(one `Cargo.lock` for the whole repository).

## Decision
`validate()` still requires the DSL ABI to match, but accepts an implementation/lockfile pair that is either the
current build's **or** listed in the committed, append-only table
`crates/jope-campaigns/known-evaluator-identities.json`. Producing a receipt is unchanged: callers use
`EvaluatorIdentityV1::current()`. The table is excluded from the hashed inputs, so appending a row never changes
the current identity. A row is appended only together with the receipt path that proves the identity existed
(historical receipts in the learning store `runs/` prefix); the port's history scan seeds the first rows.

## Consequences
Old receipts remain verifiable and auditable under the identity that produced them; a new feat-math byte change
(WP2's `gpr.rs`, `registry.rs`) creates a new current identity without invalidating history. Reviewers see every
accepted historical identity in one file.

## Current identity transition: public stage names (2026-10-30)

PUB-02 adds the public-names module declaration to `crates/feat-math/src/lib.rs`, which is an ADR-0004
hashed input. After the merged DSC-01 identity successor (following API-01 and RNK-11), the implementation
hash changes from
`04d59d4d2bc54fc3448a35f8224c76fd0b0a8be99bd9f1218af57889b4cddb78` to
`cd46314f74ca3e6d56b5283a50c6b06c64a5560a39fc42c6af86285aecf102e2`; the identity digest changes from
`dbfe797208ac96bb67e5f5109c23e54fc2b9e8b76ac89638dea90189b9b48309` to
`d6612bb0854769ba9e5c869968191d8efa80ca827ca37ebf3e6a8f35d11ae7c8`. The Cargo.lock hash remains
`89d5812adf6dcfd0b549435ee38863753a21550206623cefad553a970dc80420` across this PUB-02 transition (API-01
is the preceding lockfile change). The DSL ABI and evaluator calculations are unchanged. The identity fixture
records the new current identity; the historical JSON table gains no row here because this PR has no shipped
receipt to support one.

## Current identity transition: PUB-05 phrase renderer (2026-10-30)

PUB-05 adds a public phrase-rendering module to `jope-service` and a test-only direct
`feat-math` dependency so its exhaustive test can iterate the reviewed heads, operations and
parameters catalogs. The added Cargo.lock edge changes the identity input even though it adds no
normal dependency and changes no evaluator behavior. After the post-TUI-07 identity tip
`d60ec583c5fd12b357722f5fe6d30e9f8dc9b25d`, implementation hash `de589d75e8de5de2cde4716532cb40a511ce499c5d52eab0c3d5a855cee5c8cf` advances to
`1ea51e44c33ad884306cb40a686e34d527f80fb6f57220730df484c36baa97de`; Cargo.lock hash
`0cbb2c7fc025679f52256137b1ed45e9ae1002ce265f9dc43e60b4b4276e9704` advances to
`cfbdd26fea16ec7637c12cbce081ecffe8a71f5426c755ddaa7b87fe0e5e794f`; evaluator identity
digest `ea72a331ffc5b0bae4dcfe48bf4ed2160f3eeb6348e8d6111ca04706e96db6f0` advances to
`c82443ae79473c7899f1abe76b214c3fa600858d70fb22468384c79353f428a8`. The lineage source
fixture's generation SHA is refreshed with the announced identity commit; its source hashes and
semantics are unchanged. There is no shipped receipt proving this identity existed, so the
append-only known-identity table gains no row.

## Current identity transition: EXP-09 deterministic export bundles (2026-10-30)

EXP-09 promotes `feat-math` to a normal `jope-service` dependency and adds the pinned `zip` dependency so the service can emit fitted language sources and publish deterministic ZIP bundles. These Cargo.lock edges advance implementation hash `a22a4aaa5d0a14da93376135f459d5bf041c683e248d1e4e811006e8b01412ce` to `bd94aff9d36165c43f9ddc91f835d5b8dd15b8714770e79d2106225cd8f5caf8`; Cargo.lock hash `13f796f7f7c025f5c759c8c1a93314e4c8ebc588c9eb748c1ebb575e3f6d6893` advances to `1add5155961c4df9751699019ec46d477f3fa4066f56e3b239c190c46e6455c8`; evaluator identity digest `37d8ae7cde8427da99c687fd34f12b8f99d3567f22ad6d7f2364dd14eb75d481` advances to `b74651b6f25d2276682af1a8c97636a44346c07a3ee3d46459282c41c4f36a2e`. Evaluator calculations and DSL ABI are unchanged. The identity transition is recorded in `exp09_bundle_lockfile_identity_rebaseline.v1.json`; no known-identity row is added because this change has no shipped historical receipt proving the new identity existed.


## Current identity transition: BAT-02 run commands (2026-10-30)

BAT-02 adds `jope-paths` as a direct `jope-appliance` dependency so run/status/events resolve
the owner-scoped daemon record through `jope_paths::jopedime_home()`. The normal
`jope-appliance -> jope-client` and `jope-service -> jope-paths` edges are already present;
BAT-02's only new lock edge is `jope-appliance -> jope-paths`. Chained after `svc09_engine`
(SVC-09), implementation hash
`d015e8339171917e4922c2d3116a608e79fb63d87846ce1e0891c19e4f0925e7` advances to
`0cbc80f8cfa08921b517dbd211321a92ab2606c33067ab72e9f3f5ef9e2c153e`; Cargo.lock hash
`dba7ae05208ccfd96594b162a3d95b2ca23c3fb8d3676f07e79a12d13083d86e` advances to
`ecfa4d1858fc0d4237d1356a34b6c5ce52a5408844229ea01ab7df5359a55a12`; and identity digest
`bd4f9b4d9708122c42e917dec0264811134f59873c0b74569a21800edd5265fc` advances to
`a5ec46b83045f5186016325ae166a250201f0d5bbf9f9e032875296ea935f892`. Evaluator calculations
and DSL ABI are unchanged. No known-identity row is added because this task supplies no shipped
receipt. The transition is registered in `bat02_jope_paths_lockfile_identity_rebaseline.v1.json`.

## Current identity transition: GOV-03 bounded run API (2026-10-31)

GOV-03 adds an `axum` dev-dependency to `jope-appliance` for its in-process run-API tests.
`axum` 0.8.9 is already locked through `jope-service`, `jope-client` and `jope-observatory`, so the only new
lock edge is `jope-appliance -> axum`, with no new package, version or feature. Chained after
`res09_candle` (RES-09), implementation hash
`408032c1357692348e5de8cd1ecb5fc349b0a2e1599329b2990b78fd961094d2` advances to
`d8cc9c3e1e7e7a426e3c3e5fdfb2af74f74af5d4b2174f4019ef84d83faf188d`; Cargo.lock hash
`3caec9c49b95c9b63024b24dc5df305bc6ab9ad54ea4265cd3373e984bdf6ba0` advances to
`da8e834b1b30bbdd93d692caa1a7b09b35c31bb0ac84096cb1fe3f3cb0cf9b93`; and identity digest
`464fb7086cbd37eaf1356c13335b41a14e1502fe955850d68c4db680452ba565` advances to
`5fb6602a96b16e5abb2cddf8a82f1c188a579d65be67abefe772f277833a0a7f`. Evaluator calculations
and DSL ABI are unchanged. No known-identity row is added because this task supplies no shipped
receipt. The transition is registered in `gov03_axum_dev_lockfile_identity_rebaseline.v1.json`.
