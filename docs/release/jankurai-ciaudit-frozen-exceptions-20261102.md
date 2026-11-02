# Ciaudit frozen-copy dispositions

## Evidence-only snapshot class

The 1.7.6 evidence-snapshot policy now validates exact file paths against their
package-local receipt or manifest pins. Those entries are skipped only by HLT-043
(duplicate-copy detection) and HLT-045 (generated-zone hand-edit detection).
Every other Jankurai detector continues to audit the files. Missing reasons,
non-exact paths, absent pins, and hash mismatches fail the audit.

Twenty-two paths previously listed in `extra_excluded_paths` already had verified
`[[evidence_snapshot]]` entries. Their redundant global exclusions have been
removed. This includes the versioned trace archive copies, frozen `evergreen.py`,
`label_campaign.py`, `native_global_guard.py`, `neural_outbox.py`, `score_join.py`,
`training_snapshot_kumos.py`, and the h189/h192 copies. Their exact package pins
and reasons remain in `agent/audit-policy.toml`.

The t1955 deeper-fix pass adds the frozen revision2 and revision3
`generated_target_callbacks_v1/package/readonly_io.py` copies to their existing
snapshot entries. Both remain byte-identical at SHA-256
`49263f157b357809bd02d28be0ec1600dbf5362a68c5344e6f0a5859c11e4b09`; each
package's own `package/SOURCE_PINS.json` records that digest. This is the exact
HLT-043 class `sha256:671081d3da76cbc035fdef9c5c867ded3dab401437e7bc9bdb2266ee9cee8665`.
The pinned files remain unchanged.

The t2015 deeper-fix pass adds the frozen `kumo_transport.py` copies in
`encoding/fresh32` and `encoding/next128/packet0..packet3` to their existing
snapshot entries. All five remain byte-identical at SHA-256
`d5ceb1a91b6ddad7eb9c855651f6c57e19443cbdb9d2ecf0475abe43786c173f`; each
package's own `config.template.json` records the exact path and digest. This is
HLT-043 class `sha256:8b701c17f12848a77021422a5da6aba0697571f2c9c1842c01dec2749fc75f57`.
The pinned files remain unchanged.

The six canonical modules below also no longer have global exclusions. Their
frozen `hpc_verified_v5` copies remain bound by the package manifest and
`provenance/FINAL_VALIDATION.json`; the mutable canonical modules now receive all
Jankurai checks:

- `deploy/dl-flywheel/jope_flywheel/batch_stream.py`
- `deploy/dl-flywheel/jope_flywheel/evidence.py`
- `deploy/dl-flywheel/jope_flywheel/training_snapshot.py`
- `deploy/dl-flywheel/jope_flywheel/augmentation.py`
- `deploy/dl-flywheel/jope_flywheel/target_noise.py`
- `deploy/dl-flywheel/jope_flywheel/s3_library.py`

The frozen `neural_outbox.py` copies in revision2, revision3, and hpc_verified_v5
are covered by `[[evidence_snapshot]]`. The continuous and native copies remain
under their existing exact-path `extra_excluded_paths` dispositions: although
external pin records were identified, they are not package-local pins accepted by
the current evidence-snapshot validator. Their exclusions therefore remain
visible and are not represented as an evidence-only migration.

No frozen source bytes, manifests, receipts, or test inputs were changed.

## Finisher t2052 dispositions

### H230 HLT-033 frozen wave files

The H230 wave index records `INCOMPLETE` capture, `REMOTE_VERIFIED` archive status, and `GIT_COMMITTED` source; its exact source manifest is SHA-256 `d0aea2f630e93af3e066889327ffa3c6420bd5bfa494fbff945b345f8c1dc67a`. These three paths are exact `extra_excluded_paths` entries because their closed archived bytes are pinned by the wave manifest.

| Exact path | SHA-256 | HLT-033 fingerprint(s) |
|---|---|---|
| `research/flywheel_selfplay/h230_coswr_gated025_stack/analysis.py` | `0c648fde0a59d43ec70c781076753c086563adb71fa657a2882a941c1111fbc9` | sha256:0f7631a1fde8e8988d57ddf6c93c0ac8abb1a9b763cd8ae513c14b098ae4ebe6 |
| `research/flywheel_selfplay/h230_coswr_gated025_stack/gate.py` | `6943da4fef95e86b7d498b31d22e4ea078c0021f1a0bd05e779d3899359dc0f9` | sha256:af26b38952ce7a153b5576797285d515454e930db3a1cdf01a011b8794b5114f, sha256:871b86d5cf787e8d65ee71a31fffbd630f3edac551c311152cb2ae5f76c1e233 |
| `research/flywheel_selfplay/h230_coswr_gated025_stack/runner.py` | `c293080c8332b0a507e95545506f3f5e1dccb24903f8c8307bd71030c9b99acc` | sha256:dfcccc5693d4f332ac59a5b67a01025c23123d56135341067376b3bfc095a837 |

### P3 registry fixture dead-language allowance

The path-scoped `old` allowance is limited to this byte-pinned test fixture; the current file SHA-256 matches its `deploy/dl-trainonly/lane/MANIFEST.json` entry. The HLT-001 fingerprint is `sha256:d5d6b8453d8…`.

| Exact path | SHA-256 | Pin |
|---|---|---|
| `deploy/dl-trainonly/lane/src/dl_training_v1/tests_p3/test_p3_registry.py` | `2d315a3892070f2f72983009204f36909d9ee63e1269e03ed2d878673bbaa3e9` | `deploy/dl-trainonly/lane/MANIFEST.json` |

### HLT-043 receipt-pinned copies

These exact report instances were added to `[[evidence_snapshot]]` after their on-disk SHA-256 matched the path and digest in `deploy/dl-flywheel/SOURCE_MANIFEST.json`. The mutable `pipelines/programs/staged_inputs.py` and `generated_target_callbacks_v1/operations/*` instances remain outside these entries.

| Exact path | SHA-256 | HLT-043 fingerprint(s) |
|---|---|---|
| `deploy/dl-flywheel/pipelines/fresh_gp_v2/revision2/authority_lease.py` | `d16f87827b3efd5db2718c73b60b6e09c7acec1d55a1337cca9f6bb6d0c9773e` | sha256:0ac9f591624df497cfe75a783bd5d48e095ef5093b06455b3c093bf6a21964a0 |
| `deploy/dl-flywheel/pipelines/fresh_gp_v2/revision2/execution_boundary.py` | `9c338a6a5d22ed2073f5c14d62f40e73bac14ad3c162a2d9a81766ff720522ba` | sha256:a3f54623b344372682730a4cb0cccb90a9ac7b9291855c2d85406a477d7ebcfa |
| `deploy/dl-flywheel/pipelines/fresh_gp_v2/revision2/executor.py` | `69939b9932c232413ff52f8bc140530033d72c94d7062005d6fa49b5d26da981` | sha256:2f180b2797d64761cf58d33bce16e80ce4a60b1f9088fcc1f8b2371c229e2bf7 |
| `deploy/dl-flywheel/pipelines/fresh_gp_v2/revision2/global_attempts.py` | `d9159f5c72fb27f5b0b0ac4dc0a83c1d1f71501c1e4a4c903cda1e50e6e144dc` | sha256:9293e3f1c98dbcb56081a1e16a29f211025deade1465f90fd40c49eeee10f77f |
| `deploy/dl-flywheel/pipelines/fresh_gp_v2/revision2/prepare.py` | `38bfeb93ce2fc9fb0d1eca349dc2ff458bb777e5fb4cafde4bbfff1c8e5a7306` | sha256:0dbdc1bf695bd15949749507cdf157ea1548488d5d8d71cc52562619687d515d |
| `deploy/dl-flywheel/pipelines/fresh_gp_v2/revision2/staged_inputs.py` | `069a9c007db1e145476c2f3bffe13983daaf11f3ac5bc6d2db5763f43119971c` | sha256:1f1804583e512c586ff9d0c2ae527089498417804940b561d9aa465346b17e81 |
| `deploy/dl-flywheel/pipelines/fresh_gp_v2/revision2/transport.py` | `6f450bc96743da53242c7df670b2690bcb53393a136faab6db116dff28c8ae2f` | sha256:74dc8c76d58cfe213d982f612eff1ea709d5363c4d20e17b56bbc3f1bc55f52f |
| `deploy/dl-flywheel/pipelines/fresh_gp_v2/revision3/authority_lease.py` | `d16f87827b3efd5db2718c73b60b6e09c7acec1d55a1337cca9f6bb6d0c9773e` | sha256:0ac9f591624df497cfe75a783bd5d48e095ef5093b06455b3c093bf6a21964a0 |
| `deploy/dl-flywheel/pipelines/fresh_gp_v2/revision3/execution_boundary.py` | `9c338a6a5d22ed2073f5c14d62f40e73bac14ad3c162a2d9a81766ff720522ba` | sha256:a3f54623b344372682730a4cb0cccb90a9ac7b9291855c2d85406a477d7ebcfa |
| `deploy/dl-flywheel/pipelines/fresh_gp_v2/revision3/executor.py` | `3cbe2061fe5ce6147c2fd14a94d4c3fed1e0a32604ecb503c99de72bc9f9b36c` | sha256:2f180b2797d64761cf58d33bce16e80ce4a60b1f9088fcc1f8b2371c229e2bf7 |
| `deploy/dl-flywheel/pipelines/fresh_gp_v2/revision3/global_attempts.py` | `862569c5d71d34641d899bd44e7a897703c285e20f921448e65cd125da688087` | sha256:9293e3f1c98dbcb56081a1e16a29f211025deade1465f90fd40c49eeee10f77f |
| `deploy/dl-flywheel/pipelines/fresh_gp_v2/revision3/prepare.py` | `38bfeb93ce2fc9fb0d1eca349dc2ff458bb777e5fb4cafde4bbfff1c8e5a7306` | sha256:0dbdc1bf695bd15949749507cdf157ea1548488d5d8d71cc52562619687d515d |
| `deploy/dl-flywheel/pipelines/fresh_gp_v2/revision3/staged_inputs.py` | `069a9c007db1e145476c2f3bffe13983daaf11f3ac5bc6d2db5763f43119971c` | sha256:1f1804583e512c586ff9d0c2ae527089498417804940b561d9aa465346b17e81 |
| `deploy/dl-flywheel/pipelines/fresh_gp_v2/revision3/transport.py` | `6f450bc96743da53242c7df670b2690bcb53393a136faab6db116dff28c8ae2f` | sha256:74dc8c76d58cfe213d982f612eff1ea709d5363c4d20e17b56bbc3f1bc55f52f |
| `deploy/dl-flywheel/pipelines/fresh_gp_v2/v1/authority_lease.py` | `d16f87827b3efd5db2718c73b60b6e09c7acec1d55a1337cca9f6bb6d0c9773e` | sha256:0ac9f591624df497cfe75a783bd5d48e095ef5093b06455b3c093bf6a21964a0 |
| `deploy/dl-flywheel/pipelines/fresh_gp_v2/v1/provenance/staged_inputs.py` | `0796684b93d760504c0a1dac19a84bb9e05e1c9974dbb3064f66f5264a55b481` | sha256:0e27a19013031cba0b66acb12ed633769bc1500c09b941c9c983d467f14530ca |
| `deploy/dl-flywheel/pipelines/generated_target_callbacks_v1/revision1/package/callbacks.py` | `dfc0f289f51ed55c0a6b4a0d4b1ecd241d11ea31311403747ae6e2df035133d4` | sha256:2b72b142ce00aeb541ae5fb25de11f94f7e114d9b83f72c9a0a16a7decaa5bb3 |
| `deploy/dl-flywheel/pipelines/generated_target_callbacks_v1/revision1/package/tests_contracts.py` | `fccb0c587e7fdba0ebbfd9b04dbfb130a462c7a22bb2e2a1fc1a065ec944da78` | sha256:4e67950a1574f90a21a5ba1847a22128a742acb7de048ace2f223a1ad187d5ab |
| `deploy/dl-flywheel/pipelines/generated_target_callbacks_v1/revision2/package/callbacks.py` | `37935e0cba6ac645dddb3fc4158c14b2c418e379d6ed71eab7b7dea131ceae9f` | sha256:2b72b142ce00aeb541ae5fb25de11f94f7e114d9b83f72c9a0a16a7decaa5bb3 |
| `deploy/dl-flywheel/pipelines/generated_target_callbacks_v1/revision2/package/tests_contracts.py` | `fccb0c587e7fdba0ebbfd9b04dbfb130a462c7a22bb2e2a1fc1a065ec944da78` | sha256:4e67950a1574f90a21a5ba1847a22128a742acb7de048ace2f223a1ad187d5ab |
| `deploy/dl-flywheel/pipelines/generated_target_callbacks_v1/revision2/package/tests_custody.py` | `3563cc0b51748807be79108e8843452e7dca738825c94daceb68e3abc3416ab2` | sha256:b161e8d10ad3bf0b64c61998e79e516231424d5a66b8608003edcca69499b5c6 |
| `deploy/dl-flywheel/pipelines/generated_target_callbacks_v1/revision2/package/tests_semantics.py` | `f67a05e409efd6c1206f63d9a354ca7bfa71c61840997427c310b2a3c5ca2c1e` | sha256:b19bef1fa574585ae0b99dcac17ab08c5191428e432fd44d3b495094f3835967 |
| `deploy/dl-flywheel/pipelines/generated_target_callbacks_v1/revision3/package/callbacks.py` | `5e8b17daee58da4f66f1e4bf2bce4c62b5e5bc7889ae3150b4f52a159141e014` | sha256:2b72b142ce00aeb541ae5fb25de11f94f7e114d9b83f72c9a0a16a7decaa5bb3 |
| `deploy/dl-flywheel/pipelines/generated_target_callbacks_v1/revision3/package/tests_contracts.py` | `fccb0c587e7fdba0ebbfd9b04dbfb130a462c7a22bb2e2a1fc1a065ec944da78` | sha256:4e67950a1574f90a21a5ba1847a22128a742acb7de048ace2f223a1ad187d5ab |
| `deploy/dl-flywheel/pipelines/generated_target_callbacks_v1/revision3/package/tests_custody.py` | `3563cc0b51748807be79108e8843452e7dca738825c94daceb68e3abc3416ab2` | sha256:b161e8d10ad3bf0b64c61998e79e516231424d5a66b8608003edcca69499b5c6 |
| `deploy/dl-flywheel/pipelines/generated_target_callbacks_v1/revision3/package/tests_semantics.py` | `f67a05e409efd6c1206f63d9a354ca7bfa71c61840997427c310b2a3c5ca2c1e` | sha256:b19bef1fa574585ae0b99dcac17ab08c5191428e432fd44d3b495094f3835967 |
