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
