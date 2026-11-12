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

## Current identity transition: campaign one-shot SHA-256 helper (2026-11-12)

The remaining one-shot SHA-256 hex calls in `jope-dime-regressor` now use
`selfplay_runtime::sha256_bytes`. `campaign_evaluator.rs` and
`commands/single_fold_proof.rs` are ADR-0004 evaluator identity inputs, so this
announces an identity-only transition. `implementation_sha256` changes from
`014a6fbfaf0909fcc0edf3f1aadf9888fa646babd8a00ea283c0fe323e4810e8` to
`6367505c0d69f1a137321e8149d86cbad7f12beca37710d51ef22e26b0a3968f`, and
`identity_digest` changes from
`3174ad38add05467f69b1cf06b2ee9f57210d241144863d58c058831503ab3ee` to
`75f4f632406cad050c29cd51d5cd60acaeb733dc241888ab9694294e7f1fbf69`. The DSL ABI
is unchanged. The regeneration ran
`JOPE_COMPAT_FREEZE_REGENERATE=identity JOPE_COMPAT_FREEZE_SHA=b9f0998551d87a4c8b6564331ee2d5fc7c225a0a cargo nextest run --locked -P pr -p feat-math --test compat_freeze`
(13/13 passed). The append-only known-identity table gains no row because this
identity has no shipped receipt.

## Current identity transition: race test dedicated pool (2026-11-02)

P45 changes `crates/feat-math/src/evolve.rs` so `race_respects_tiny_time_budget` runs its one-chunk
race on a dedicated two-thread rayon pool; the unchanged 30s wall bound then measures the race and
not queueing behind sibling tests on the shared global pool (the test took 1.1s alone but 40s inside
the full `feat-math` suite). `evolve.rs` is an evaluator identity input, so this change announces an
identity-only transition. `implementation_sha256` changes from
`43c6ae985f652b408c94f1e5f60c93a4ecf99e801f5a11a189cda1a55d747c4f` to
`0750e049cdabf54a4af94b6673790bd71f3600541e40f80cbeb55f81f211d37d`, and `identity_digest`
changes from `9dab8f04a9bef7f36a5b432eb2123a8ff2fac9acba434ccc4331ba4cf25bf5c8` to
`77af8284de8e3e6e80177660f1d72c0790540be4e8556bca9382f6352a09bece`. The DSL ABI is unchanged.
The regeneration ran `JOPE_COMPAT_FREEZE_REGENERATE=identity JOPE_COMPAT_FREEZE_SHA=09653ef9a2a37e544b6ddff96d0a56e3d9f05847 cargo nextest run --locked -P pr -p feat-math --test compat_freeze`
(13/13 passed). The append-only historical chain records both fixture successors in
`crates/feat-math/tests/fixtures/compat_freeze/transitions/p45_race_dedicated_pool_identity_rebaseline.v1.json`.
The append-only known-identity table gains no row because this identity has no shipped receipt.

## Current identity transition: sealed corpus mapping (2026-11-02)

P35 changes `crates/jope-engine/src/corpus.rs` to snapshot unverified record files into sealed
storage and use safe `memmap2` slice access; it also adds the existing `memmap2` package as a
direct `jope-parallel` dependency. Both `corpus.rs` and `Cargo.lock` are evaluator identity
inputs, so this change announces an identity-only transition. `implementation_sha256` changes
from `e5b6edc767a192b3ce7b14b132b68cd29fa72407282ca3231562671ea236e3d5` to
`43c6ae985f652b408c94f1e5f60c93a4ecf99e801f5a11a189cda1a55d747c4f`; `cargo_lock_sha256`
changes from `e0f1d2f90337bf2df4930b051cf6b3115ee0b27bc5533c56de4b8a927678a401` to
`7d90c1d3f86960fcfbbf66e848d22e665c446debec38e386f512ef09c8a28042`; and `identity_digest`
changes from `2febbecb129d61930a625d5aa15f7f1a49ecf035c352e4999fc1719b88b20e86` to
`9dab8f04a9bef7f36a5b432eb2123a8ff2fac9acba434ccc4331ba4cf25bf5c8`. The DSL ABI is
unchanged. The identity-only regeneration and its receipt are documented in
[`P35 corpus identity transition`](../qualification/jankurai-p35-corpus-identity-20261102.md).
The append-only historical chain records both fixture successors in
`crates/feat-math/tests/fixtures/compat_freeze/transitions/p35_sealed_corpus_identity_rebaseline.v1.json`.
The append-only known-identity table gains no row because this identity has no shipped receipt.

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

## Current identity transition: portable PTY test dependency (2026-11-01)

The first stale-identity commit is PUB-06 #1249 (`4b51c5528129c3fc4c170f2c4074a1a176c25eb3`),
which adds the intentional `portable-pty = "=0.9.0"` `jope-service` test dependency for its PTY
coverage. The dependency adds a `jope-service -> portable-pty` edge to the workspace lockfile; it
does not change evaluator calculations. The identity regression test passes at the preceding
fixture baseline `3df2979abf861502f69f3fa70e7871952bbf0c06` and fails immediately at #1249. Later
intentional lockfile changes in #1488, #1555, #1559 and #1610 accumulated before this repair, so
this records one source-bound transition from that baseline to main `97ce7c95e9d573eb9b7c5518953b64bc9937e41c`.

Implementation hash `d8cc9c3e1e7e7a426e3c3e5fdfb2af74f74af5d4b2174f4019ef84d83faf188d` advances to
`e9540104996cee7b38ded1cbfec626eef45893a820e891619d4bfc2e92f4836d`; Cargo.lock hash
`da8e834b1b30bbdd93d692caa1a7b09b35c31bb0ac84096cb1fe3f3cb0cf9b93` advances to
`96a5734beed7151cfebd89651fdc4611b4b79b822010a0ce16af683a4a72236b`; and identity digest
`5fb6602a96b16e5abb2cddf8a82f1c188a579d65be67abefe772f277833a0a7f` advances to
`2d5c740d8cb55f65d76b4348bec3f71d0b7d8800d84de78a25088d9348853b18`. The DSL ABI and evaluator
calculations are unchanged. The identity and lineage source pins are regenerated at the
transition commit. The append-only known-identity table is unchanged because this work has no
shipped receipt proving that the new identity existed.

## Current identity transition: P36 safe CPU affinity sets (2026-11-02)

P36 replaces zero-initialized libc CPU sets with `nix::sched::CpuSet`, adding the existing
`nix` package as a direct `jope-campaigns` dependency and changing `Cargo.lock`. The
identity-only regeneration also closes the unrecorded identity drift introduced by #1659,
which was already red on `origin/main` before this lane's change. The transition advances
implementation hash `e9540104996cee7b38ded1cbfec626eef45893a820e891619d4bfc2e92f4836d` to
`e4dd3486d173c65209da3df7fdc6cc8aaccdbc5d7e0102eab512bd6f7f491716`, Cargo.lock hash
`96a5734beed7151cfebd89651fdc4611b4b79b822010a0ce16af683a4a72236b` to
`e0f1d2f90337bf2df4930b051cf6b3115ee0b27bc5533c56de4b8a927678a401`, and evaluator
identity digest `2d5c740d8cb55f65d76b4348bec3f71d0b7d8800d84de78a25088d9348853b18` to
`6ff84deb38b32b725960d9b18a1a7775ea7df6593fc4eb575f54a6611e22fd1d`. Evaluator
calculations, DSL ABI, behavior fixtures and lineage source hashes are unchanged; the lineage
fixture changes only its generation SHA. The transition is registered in
`rustmem_cpu_affinity_identity_rebaseline.v1.json`. No known-identity row is added because this
work has no shipped receipt proving the new identity existed.

## Current identity transition: rustunsafe authority lineage (2026-11-02)

The jankurai rustunsafe cap (#1782) replaced libc `getsockopt(SO_PEERCRED)` and `waitid` in
`crates/jope-engine/src/eval_registry/authority.rs` with `nix` safe wrappers. That file is a
lineage source hashed into the teaching-store `semantic_dependencies_sha256`, and #1782 merged
without recording it, so `compat_freeze` `lineage_sources_match_the_recorded_sources` was red on
`origin/main`. The identity-only regeneration at `1b5154d82787ea955f9e989a52eeb2f23c51627f`
records the new hash (`a4c02980707758e3067dc68cfc5e34f18f9544a8cb1f4a098ee3f883fd4b7a74` to
`5755ac38ea47c03427f2357b3e65c17c0822f7f780c2c29561094f1128e5de05`). The evaluator
implementation hash, Cargo.lock hash and identity digest are unchanged; the evaluator identity
fixture changes only its generation SHA. The transition is registered in
`rustunsafe_authority_lineage_rebaseline.v1.json`. No known-identity row is added because this
work has no shipped receipt proving a new identity existed.
## Current identity transition: browser wasm32 portability (2026-11-02)

The browser successor follows the already-merged CPU-affinity and authority-lineage
transitions above. It changes only the hashed stage-cache source so the default byte cap is
representable on 32-bit targets, retaining the native 64-bit 6 GiB default. Announced
identity-only regeneration will advance the current implementation
`e4dd3486d173c65209da3df7fdc6cc8aaccdbc5d7e0102eab512bd6f7f491716` and identity
`6ff84deb38b32b725960d9b18a1a7775ea7df6593fc4eb575f54a6611e22fd1d` from the merged
main fixture. Cargo.lock, numerical/model/emitter/generator/search fixtures, every historical
transition and the known-identity table remain unchanged. The two metadata fixture hashes,
old/new identities and committed generation source are recorded in
`browser_wasm_identity_rebaseline.v1.json`; no shipped receipt or behavior rebaseline is claimed.

## Current identity transition: fitted runtime moved to Rust (2026-11-02)

P23 ports the fitted-language runtime from `crates/feat-math/cpp/fitted_runtime.cpp` to Rust,
removing C++ from the product runtime surface while retaining the C++ implementation as a
test-only parity oracle. The change follows the merged browser wasm32, P35 corpus and P45
dedicated-pool identity transitions. The evaluator implementation hash advances from
`0750e049cdabf54a4af94b6673790bd71f3600541e40f80cbeb55f81f211d37d` to
`10fe3dee6daf50c3b54362df80cbcb60774c6169563fcba2c3c0036ab5e3dfda`, and the identity digest
from `77af8284de8e3e6e80177660f1d72c0790540be4e8556bca9382f6352a09bece` to
`8fe007189bca9fe7e0d92b85b509ac79c116616e41a16bb5a0c4bf93d1dda9d6`. Announced identity-only
regeneration records the fitted runtime source change; the DSL ABI, emitted bytes, Cargo.lock
and known-identity table remain unchanged. The transition is registered in
`prodlang_rust_runtime_identity_rebaseline.v1.json`. No shipped receipt or behavior rebaseline
is claimed.
## Current identity transition: P21 campaign family split contract (2026-11-02)

P21 moved the stable source-family train/validation assignment used by self-play
expansion into the reusable `jope-datasets` contract. Adding that crate as a
direct campaign dependency changes `Cargo.lock` and therefore the recorded
evaluator identity, even though evaluator calculations, DSL ABI and emitted
behavior fixtures are unchanged. The transition advances implementation hash
`10fe3dee6daf50c3b54362df80cbcb60774c6169563fcba2c3c0036ab5e3dfda` to
`9425373d24b3e6b2782ea50a17abcab8d84d20e4cb8ce12ff70600eb3fa9089e`, Cargo.lock
hash `7d90c1d3f86960fcfbbf66e848d22e665c446debec38e386f512ef09c8a28042` to
`2a6d41eac882df5122e2958cd4bc5845156b2d7b7d198bd9e52c59bac02b50ed`, and
identity digest `8fe007189bca9fe7e0d92b85b509ac79c116616e41a16bb5a0c4bf93d1dda9d6`
to `f5dff2d6cef17cf237eb11c2f45df318490b4ebba66b3b8c47cdc264b4e49b7b`. The
identity-only regeneration is bound to source commit
`e240a9f27e3bba296acf77d29dae10fd2529a0c1`. The candidate tree also includes
the already merged Rust enum naming change in `crates/feat-math/src/full_rust.rs`;
the recorded identity reflects the complete rebased source tree. No known-identity row is added
because this change has no shipped receipt proving that the new identity existed.

## Current identity transition: semantic campaign source names (2026-11-02)

P50 renames the active one-fold evaluator adapter from `commands/exact_cv_v2.rs` to
`commands/single_fold_proof.rs` and the evaluator implementation from `evaluation_v2.rs` to
`campaign_evaluator.rs`. These files remain in the evaluator's hashed closure. The semantic
renames therefore mint a new implementation identity; the identity fixture was regenerated
against the source commit with the established `JOPE_COMPAT_FREEZE_REGENERATE=identity` proof.

The implementation hash advances from
`e4dd3486d173c65209da3df7fdc6cc8aaccdbc5d7e0102eab512bd6f7f491716` to
`b1e0f654f710747379ecf7e852798b71150d5eec477cba16abdf7b4641874f6b`. The Cargo.lock hash
remains `e0f1d2f90337bf2df4930b051cf6b3115ee0b27bc5533c56de4b8a927678a401`; the identity
digest advances from `6ff84deb38b32b725960d9b18a1a7775ea7df6593fc4eb575f54a6611e22fd1d` to
`41befe853ef6d31972f3126a29d501b7d97c84c6025a4b9497bf1ca628651709`. The rebaseline commit
is `0a734502661a5e653960bb7623f4ee0bf3a14206`. No evaluator behavior is intentionally changed.
The known-identity table remains unchanged; this lane had no qualifying shipped-receipt path to
append for the prior current identity.

## Current identity transition: Engine & campaign SHA-256 helper deduplication (2026-11-12)

PRs `#2254` and `#2258` point one-shot SHA-256 hex formatting in `jope-engine` (`manifest.rs`, `eval_registry/authority.rs`, `eval_registry/service.rs`) and `jope-campaigns` (`commands/exact_cv.rs`) at shared `sha256_hex` / `sha256_bytes` helpers with identical hex digest output. `exact_cv.rs` is an ADR-0004 evaluator identity input (`8662b33e` -> `9c18ea09`), and the three `jope-engine` modules are recorded lineage sources. Chained after the Gaussian-process prediction clip identity, implementation hash `0c0c83700e3ecb5b1f1010bc70b26cb081313296513ca408051c6eaaebc9317a` advances to `014a6fbfaf0909fcc0edf3f1aadf9888fa646babd8a00ea283c0fe323e4810e8`; `Cargo.lock` hash stays `cbd61e7462acc04113899d15debd4cfc07b949598a54978600191c3acd5b19af`; evaluator identity digest advances from `db8158f5f39f54082374063b3332522deece09c9051cb3d7bc5cfa4f88911975` to `3174ad38add05467f69b1cf06b2ee9f57210d241144863d58c058831503ab3ee`. DSL ABI is unchanged. The identity-only regeneration is bound to source commit `2d97d841108f8eceabd4f0a4206248feacec2d39` and recorded in `sha256_dedup_identity_rebaseline.v1.json`. No known-identity row is added because this change has no shipped receipt proving the new identity existed.

## Current identity transition: Gaussian-process prediction clip (2026-11-11)

A plain `(pipe X (gpr))` fit and the composite-kernel twin clamped every prediction at +/-60. The stored native clip is now the union of that window and the training targets widened by 5%. Recipe Rust, Python, and fitted C++ keep the legacy +/-60 clamp, so the emitted-byte and generator-stream fixtures stay on the historical behavior pin. `gpr.rs` is an evaluator identity input, so this change announces an identity-only transition. Chained after the unused-direct-deps identity, implementation hash `0e01b9cd042273d2107cd60a8551968c494027e0e821814eaee92a8ffb8cdcc5` advances to `0c0c83700e3ecb5b1f1010bc70b26cb081313296513ca408051c6eaaebc9317a`. The Cargo.lock hash stays `cbd61e7462acc04113899d15debd4cfc07b949598a54978600191c3acd5b19af`. The identity digest advances from `d110fce29ea7d5f589feeec25b2ebaa2fffa8347743d16202b5c3b7cd5fdfb93` to `db8158f5f39f54082374063b3332522deece09c9051cb3d7bc5cfa4f88911975`. DSL ABI is unchanged. The identity-only regeneration is bound to source commit `668091be27e0138a819b2f9c86e17cbe7bd9382c` and recorded in `gpr_prediction_clip_identity_rebaseline.v1.json`. No known-identity row is added because this change has no shipped receipt proving the new identity existed. Lineage source hashes are unchanged; the lineage fixture header records the same source commit.

## Current identity transition: unused direct dependencies (2026-11-11)

Eight crates drop direct dependencies that their Rust sources do not import: `jope-prep` drops `hex`, `ruzstd`, and `thiserror`; `jope-telemetry` and `jope-carbon` drop `thiserror`; `jope-datasets` drops `walkdir`; `jope-dime-regressor` drops `memmap2`; `jainnode` drops `sha2`; `jain-smartcluster-daemon` drops `pidfd-util` and `rand`; `scq` drops `futures-util`. `pidfd-util` leaves `Cargo.lock`. Carbon's feature-gated crates stay. Chained after the CARBON PR-1 identity, implementation hash `4986c1d2033fc96ce0ea717ca13af65932f19e7dd362f1161ee9eb579da1eccd` advances to `0e01b9cd042273d2107cd60a8551968c494027e0e821814eaee92a8ffb8cdcc5`; `Cargo.lock` hash `11c133db9a82c346a96040852de9a836b62239fc6c19f64c7e2232f8a073e631` advances to `cbd61e7462acc04113899d15debd4cfc07b949598a54978600191c3acd5b19af`; evaluator identity digest `ac9f86accfaac507ed5f7588e302bbf8c22995fd8b22e01262c4a640f5d3296b` advances to `d110fce29ea7d5f589feeec25b2ebaa2fffa8347743d16202b5c3b7cd5fdfb93`. DSL ABI is unchanged. The identity-only regeneration is bound to source commit `3f5a25751601016f6ff90188e5e5684da0f0566e` and recorded in `unused_direct_deps_lockfile_identity_rebaseline.v1.json`. No known-identity row is added because this change has no shipped receipt proving the new identity existed.

## Current identity transition: CARBON PR-1 workspace registration (2026-11-10)

Carbon Phase C1 ([ADR-0007](ADR-0007-carbon-lambda-cells.md)) registers the first-party `jope-prep` and `jope-carbon` workspace crates and links `jope-service -> jope-prep` so client conditioning and service ranker conditioning share one pure-Rust implementation. All third-party crates, versions, features, and registry sources in `Cargo.lock` remain unchanged. Chained after the COMPARATORS PR-3 run-stage identity, implementation hash `f252680f72365d314c9839ed0a17f7a54b49a1be2009208d4a5a10b95de2118a` advances to `4986c1d2033fc96ce0ea717ca13af65932f19e7dd362f1161ee9eb579da1eccd`; `Cargo.lock` hash `82750afc35db964cd787e8998335be255dcde8cd5d4f5d7c36aae914df744630` advances to `11c133db9a82c346a96040852de9a836b62239fc6c19f64c7e2232f8a073e631`; evaluator identity digest `2b2b7c781c96b628e04c71ef0b001d77b49814be3dc3c8e8a0ecc4a067fecff6` advances to `ac9f86accfaac507ed5f7588e302bbf8c22995fd8b22e01262c4a640f5d3296b`. DSL ABI is unchanged. The identity-only regeneration is bound to source commit `45bb0b7f562ef92423dd954f92d1933bcf39f550` and recorded in `carbon_workspace_registration_lockfile_identity_rebaseline.v1.json`. No known-identity row is added because this change has no shipped receipt proving the new identity existed.

## Current identity transition: COMPARATORS PR-3 run stage (2026-11-10)

PR-3 adds `jope-comparators` as a normal `jope-service` dependency so the opt-in finalize stage can fit the roster. The only new Cargo.lock edge is `jope-service -> jope-comparators`. No package version, feature, or registry source changes. Chained after the PR-1 workspace-registration identity, implementation hash `0dd4ab4abea79300515e4a8a5029c455bedab5cf8e95a776faf59eab773de8cc` advances to `f252680f72365d314c9839ed0a17f7a54b49a1be2009208d4a5a10b95de2118a`; Cargo.lock hash `98d962b69e45d57ff381a4ab3208525d85deb958dbb1e18f7f757f88dbef16ae` advances to `82750afc35db964cd787e8998335be255dcde8cd5d4f5d7c36aae914df744630`; evaluator identity digest `51819084e01575f510b9c6fe4298861a7dc18ee4f3bb666529e8769cec80b868` advances to `2b2b7c781c96b628e04c71ef0b001d77b49814be3dc3c8e8a0ecc4a067fecff6`. DSL ABI is unchanged. The identity-only regeneration is bound to source commit `4383741dc4a5a4a317c4a96ea83003c626e002f7`. The transition is recorded in `comparators_run_stage_lockfile_identity_rebaseline.v1.json`. No known-identity row is added because this change has no shipped receipt proving the new identity existed.

## Current identity transition: P42 evaluator source paths and P66 typed error (2026-11-02)

The merged P42 (#1998) evaluator rename changed the labels and paths in the implementation
hash closure, but its compatibility fixture was not regenerated. The new
`reporot_evaluator_fixture_paths_identity_chain.v1.json` records the identity-only bridge from
the P21 fixture to a regeneration bound to P42 source commit
`d10a4a32dc324f6dd44d36c3aa9d07bba4fb2ad7`. Its implementation hash is
`42922a8abb695246512b1719a2edd6a52b29b8f3089fac988f90cc9869967e11`, with identity digest
`7e1d14fdbe2be226c56be91f3ed0cbc43f6b7b77cf4e6171c48e5b274fe864a1`.

P66 makes unsupported blend-kind failures expose a typed error. The source change advances the
implementation hash to `fc09526e5a60ea80968355957d43da364285c250aa21dc71dcb47087c0977163`
and identity digest to `647e126ba7a1f26b18ffe7c4b7a2116b478938a88d096b4fe65bbe65207fc052`.
The P66 rebaseline also updates the two recorded `jope-engine/src/manifest.rs` lineage pins to
`e9df1f454b5f2e303edf653f2b0d2b4cd600235c31fb800de976a96576c3510c`. Evaluator ABI, DSL ABI,
Cargo.lock, the known-identity table and behavioral compatibility fixtures are unchanged. Both
identity transitions are recorded in
`deadlang_unsupported_blend_identity_rebaseline.v1.json`; no shipped receipt or known-identity
row is claimed.
