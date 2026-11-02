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
