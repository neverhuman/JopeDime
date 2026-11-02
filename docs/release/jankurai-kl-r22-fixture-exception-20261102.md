# R22 pinned fixture disposition

The full audit `rerun/j_27137b5fb.json` reports HLT-019 fingerprint
`sha256:3cd49f1c3886e329566608707ba09946af5672ce041abb3ecba0a623c66e0273` at
`deploy/dl-flywheel/pipelines/continuous_v6/tests/fixtures/kl_ref_r22_harness_v2.py:142`.
That file is explicitly labeled a verbatim reference copy and says not to edit it.

The fixture's full SHA-256 is
`f4a424b360d5ba6306d65ef14874f307779664ecb9d2ac3ff9ab415778f1028f`, matching
`module_file_sha256` for `r22_harness_v2` in
`deploy/dl-flywheel/pipelines/continuous_v6/tests/fixtures/kl_pins.json`. The same
pin records each definition hash and binds the original lane file
`rlfw-takeover/lpi-scratch/lpi_scratch_train_v2.py` at SHA-256
`3300e591178cff47a1c2bb75a9ff8467625d0803c71fbcded49bab0d50931584`. The
registered pin tests verify these hashes before using the fixture.

The exact fixture path is excluded from Jankurai scoring in
`agent/audit-policy.toml` because changing this vendored reference would invalidate
the reproducibility pin. The fixture and pin remain in the repository and their
tests remain enabled; no other path or boundary finding is excluded. Revisit this
disposition only when the reference pin and its consumers are retired or replaced
through an evidence-preserving version change.
