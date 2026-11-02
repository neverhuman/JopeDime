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
