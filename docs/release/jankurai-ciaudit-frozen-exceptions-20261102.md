# Ciaudit frozen-copy disposition

The full audit `rerun/j_ffba82986.json` reports HLT-043 fingerprint
`sha256:09d1f247ee49a47515af8d98f895f179687fd52c383bb0a4b00edf646d6b4c2d` at
`deploy/dl-flywheel/jope_flywheel/batch_stream.py`. The duplicate is the
manifest-pinned historical copy
`deploy/dl-flywheel/pipelines/hpc_verified_v5/jope_flywheel/batch_stream.py`.
Both files have SHA-256
`2c4be42aa2054d8a6041886af815c86537989abef923a5f0b4ce7cfa9908d0bb`; the
identity is recorded for both paths in `deploy/dl-flywheel/SOURCE_MANIFEST.json`
and for the frozen copy in
`deploy/dl-flywheel/pipelines/hpc_verified_v5/provenance/FINAL_VALIDATION.json`.
The preserved provenance also records the frozen source in `ORIGIN.json` and
`REVISION2_PACKET.json`.

Changing either copy would invalidate the reviewed package identity. The exact
canonical source path is therefore excluded from Jankurai scoring in
`agent/audit-policy.toml`, with no directory, glob, or rule-wide exclusion.
This disposition does not remove the source from Python tests or other security
scans. Revisit it if the pinned provenance is retired or either recorded hash
changes; any other finding requires separate review.

## `evidence.py`

The same full audit reports HLT-043 fingerprint
`sha256:fac8174f6c7182fe71fd357f91188a98935a8384bf1cbef838343e72ef27e626` at
`deploy/dl-flywheel/jope_flywheel/evidence.py`. The duplicate is the
manifest-pinned historical copy
`deploy/dl-flywheel/pipelines/hpc_verified_v5/jope_flywheel/evidence.py`.
Both files have SHA-256
`e4bd1d1e92455bd9b4a210d4e46cddf392cc565d7bcc7c1b6bea3e48a6a070a9`; the
identity is recorded for both paths in `deploy/dl-flywheel/SOURCE_MANIFEST.json`
and for the frozen copy in
`deploy/dl-flywheel/pipelines/hpc_verified_v5/provenance/FINAL_VALIDATION.json`.

Only the exact finding source path is excluded from Jankurai scoring. Tests and
other security scans remain in place. Revisit if the pinned provenance is retired
or either recorded hash changes; this does not cover other findings in the file.

## Frozen `evergreen.py` package copies

HLT-043 fingerprint
`sha256:ae6b84149c1e2e8b40e58f874b8d4b4b20d82beb3e59e1b550905af706177a71`
in `rerun/j_8df4fe341.json` includes four frozen copies, each SHA-256
`334f76b27e959230ae5f1b0a9b693d5c830734bb669d54c50fa1199759662dbe`:

- `deploy/dl-flywheel/pipelines/fresh_gp_v2/revision2/runtime/jope_flywheel/evergreen.py`
- `deploy/dl-flywheel/pipelines/fresh_gp_v2/revision3/runtime/jope_flywheel/evergreen.py`
- `deploy/dl-flywheel/pipelines/fresh_gp_v2/v1/runtime/jope_flywheel/evergreen.py`
- `deploy/dl-flywheel/pipelines/hpc_verified_v5/jope_flywheel/evergreen.py`

The hashes are bound by the revision 2 packaged-offline receipt, revision 3
packaged-tests receipt, v1 source manifest and package receipt, and hpc-v5
provenance. These exact paths alone are excluded from Jankurai scoring; the
mutable canonical `deploy/dl-flywheel/jope_flywheel/evergreen.py` remains in the
scan. No pinned bytes or other security scans change.

## Frozen `label_campaign.py` copies

HLT-043 fingerprint
`sha256:2881cc53a3b58139f0b00e15b81ad55df4ccb2fcb2202176976fe2f22fa52f5d`
in `rerun/j_8df4fe341.json` identifies these three preserved files:

- `deploy/dl-flywheel/pipelines/fresh_gp_v2/revision2/label_campaign.py`
- `deploy/dl-flywheel/pipelines/fresh_gp_v2/revision3/label_campaign.py`
- `deploy/dl-flywheel/pipelines/fresh_gp_v2/v1/provenance/label_campaign.py`

All three have SHA-256
`564f25581f8ce4e3e912e5b04a321235d887d050e5f20786c97fa8e52eb10b2f`, pinned
by the revision source packets and v1 source manifest. Only these exact paths
are excluded from Jankurai scoring. The frozen package bytes and other security
scans remain unchanged.

## Frozen `training_snapshot_kumos.py` instance

HLT-043 fingerprint
`sha256:562310a12c8e7b55e54dfe517f1424eb619c50231c32211bdc5df2d84ca0da53`
in `rerun/j_ffba82986.json` covers the copy class containing
`deploy/dl-flywheel/pipelines/hpc_verified_v5/jope_flywheel/training_snapshot_kumos.py`.
That exact historical file has SHA-256
`c571fb67d2b12739e15f0d1b79c5193a1e252178bfeea29725cb042b8ac65cb2`, pinned by
the package's `provenance/FINAL_VALIDATION.json` and `provenance/ORIGIN.json`.
The pinned instance is excluded by its exact path in `agent/audit-policy.toml`;
mutable copies remain eligible for behavior-preserving deduplication. No source
bytes, package provenance, tests, or other security scans are changed.

## Frozen `neural_outbox.py` copies

HLT-043 fingerprint
`sha256:99bfa2e00ed990007f3fbfbbc7eeb12d7f096ccb6d2e7dffdfda7420d4b375de`
in `rerun/j_ffba82986.json` includes five hash-pinned copies with SHA-256
`d1bc811f4ef1c298c0c6bcc680a5006cf987c81fda5d2b33b30ed2df16455f11`:

- `deploy/dl-flywheel/pipelines/continuous/neural_outbox.py`
- `deploy/dl-flywheel/pipelines/fresh_gp_v2/revision2/neural_outbox.py`
- `deploy/dl-flywheel/pipelines/fresh_gp_v2/revision3/neural_outbox.py`
- `deploy/dl-flywheel/pipelines/hpc_verified_v5/neural_outbox.py`
- `deploy/dl-flywheel/pipelines/native/neural_outbox.py`

Their identities are bound by the continuous source pin register, revision
source packets and packaged receipts, hpc-v5 `FINAL_VALIDATION.json`, and the
native runtime pin register. Only these five exact paths are excluded from
Jankurai scoring. The mutable `hpc_verified_v4/neural_outbox.py` stays in scope
for its owner’s deduplication. No pinned bytes or other security scans change.

## Frozen `native_global_guard.py` copies

HLT-043 fingerprint
`sha256:6c5aeec682e4127ffa1c9d63a7de8f4b5669044c3750a149e6519400c0b7945c`
in `rerun/j_ffba82986.json` identifies an exact-copy class of three frozen
files, each SHA-256
`22eee81339a46938c228bb6c17dc79147369a55ea684427819c92ee31294c63b`:

- `deploy/dl-flywheel/pipelines/fresh_gp_v2/revision2/native_global_guard.py`
- `deploy/dl-flywheel/pipelines/fresh_gp_v2/revision3/native_global_guard.py`
- `deploy/dl-flywheel/pipelines/hpc_verified_v5/native_global_guard.py`

The revision 2 and revision 3 source packets pin these hashes, and hpc-v5
`provenance/FINAL_VALIDATION.json` pins its copy. These exact paths alone are
excluded from Jankurai scoring. No source bytes or other security scans change.

## `s3_library.py`

The full audit `rerun/j_ffba82986.json` reports HLT-043 fingerprint
`sha256:7ec8a15a8c065dde9220d99090f7dbafb39b5ba80bfb4454ec8b0f743cbff839` at
`deploy/dl-flywheel/jope_flywheel/s3_library.py`. The duplicate includes the
manifest-pinned historical copy
`deploy/dl-flywheel/pipelines/hpc_verified_v5/jope_flywheel/s3_library.py`.
Both files have SHA-256
`28e1e0d41f322b69de4ee7a41ae8ba611b788934cac2963c279267260b1773b9`; the
identity is recorded for both paths in `deploy/dl-flywheel/SOURCE_MANIFEST.json`
and for the frozen copy in
`deploy/dl-flywheel/pipelines/hpc_verified_v5/provenance/FINAL_VALIDATION.json`.

Only the exact finding source path is excluded from Jankurai scoring. Tests and
other security scans remain in place. Revisit if the pinned provenance is retired
or either recorded hash changes; this does not cover other findings in the file.

## `target_noise.py`

The full audit `rerun/j_ffba82986.json` reports HLT-043 fingerprint
`sha256:095ee8f114b83fa9a95345c47f358e62e92184d595be2c307b928c9b5fcc48c9` at
`deploy/dl-flywheel/jope_flywheel/target_noise.py`. The duplicate is the
manifest-pinned historical copy
`deploy/dl-flywheel/pipelines/hpc_verified_v5/jope_flywheel/target_noise.py`.
Both files have SHA-256
`20864cfe1c0d7b639332964905f027575ad7b33e0b57e5ec7aac1fe0099be2b1`; the
identity is recorded for both paths in `deploy/dl-flywheel/SOURCE_MANIFEST.json`
and for the frozen copy in
`deploy/dl-flywheel/pipelines/hpc_verified_v5/provenance/FINAL_VALIDATION.json`.

Only the exact finding source path is excluded from Jankurai scoring. Tests and
other security scans remain in place. Revisit if the pinned provenance is retired
or either recorded hash changes; this does not cover other findings in the file.

## `augmentation.py`

The full audit `rerun/j_ffba82986.json` reports HLT-043 fingerprint
`sha256:804e402da408d7af58a4c8a5c5348b120712c8460316d52c93662b46d02af3bd` at
`deploy/dl-flywheel/jope_flywheel/augmentation.py`. The duplicate is the
manifest-pinned historical copy
`deploy/dl-flywheel/pipelines/hpc_verified_v5/jope_flywheel/augmentation.py`.
Both files have SHA-256
`cc0343abd3bcf709cdec933601efb7febd6258b56f395ccddc0ef7d88c2611bd`; the
identity is recorded for both paths in `deploy/dl-flywheel/SOURCE_MANIFEST.json`
and for the frozen copy in
`deploy/dl-flywheel/pipelines/hpc_verified_v5/provenance/FINAL_VALIDATION.json`.

Only the exact finding source path is excluded from Jankurai scoring. Tests and
other security scans remain in place. Revisit if the pinned provenance is retired
or either recorded hash changes; this does not cover other findings in the file.

## `training_snapshot.py`

The same full audit reports HLT-043 fingerprint
`sha256:e41dc90b284d1301fdf77d6e5d608798082813395422428cec2d948f8b35b477` at
`deploy/dl-flywheel/jope_flywheel/training_snapshot.py`. The duplicate is the
manifest-pinned historical copy
`deploy/dl-flywheel/pipelines/hpc_verified_v5/jope_flywheel/training_snapshot.py`.
Both files have SHA-256
`392348275c255f49fe2d774df4da678d8d23331235559e68e5885aafbe90ec09`; the
identity is recorded for both paths in `deploy/dl-flywheel/SOURCE_MANIFEST.json`
and for the frozen copy in
`deploy/dl-flywheel/pipelines/hpc_verified_v5/provenance/FINAL_VALIDATION.json`.

Only the exact finding source path is excluded from Jankurai scoring. Tests and
other security scans remain in place. Revisit if the pinned provenance is retired
or either recorded hash changes; this does not cover other findings in the file.

## Frozen `score_join.py` package copies

The full audit `rerun/j_8df4fe341.json` reports HLT-043 fingerprint
`sha256:757778574f5dde94ee0438f24257386f97f4170f6f038d9ab3cf6f9be0028172` at
`deploy/dl-flywheel/jope_flywheel/score_join.py`. Its frozen package copies are:

- `deploy/dl-flywheel/pipelines/fresh_gp_v2/revision2/runtime/jope_flywheel/score_join.py`
- `deploy/dl-flywheel/pipelines/fresh_gp_v2/revision3/runtime/jope_flywheel/score_join.py`
- `deploy/dl-flywheel/pipelines/fresh_gp_v2/v1/runtime/jope_flywheel/score_join.py`
- `deploy/dl-flywheel/pipelines/hpc_verified_v5/jope_flywheel/score_join.py`

Each frozen copy has SHA-256
`712ac8cdcf96ad33f11cc7a6051e4f5d079049f5e52b5da444223ff251dd1a06`. Their
identities are bound by the revision 2 and revision 3 `SOURCE_PACKET.json`
receipts and packaged provenance, the v1 `source-manifest.json` and packaged
receipt, hpc-v5 `provenance/FINAL_VALIDATION.json`, `ORIGIN.json`, and
`REVISION2_PACKET.json`, and the deploy-level `SOURCE_MANIFEST.json`.

Only these four exact paths are excluded from Jankurai scoring. The mutable
canonical `deploy/dl-flywheel/jope_flywheel/score_join.py` remains in scope.
No package bytes, tests, or other security scans change. Revisit the disposition
if these provenance pins are retired or a recorded hash changes.
