# HLT-001 disposition: generated admission prior state

- **Finding:** `sha256:f721d1bd0ddfef70ad394a991b174cb29a4ee513b2d69f7abd2ab7351bdf2e16`
- **Exact scope:** term `old` in `deploy/dl-flywheel/pipelines/generated_target_admission_v1/transition.py`
- **Source identity:** SHA-256 `fff9bd5ede97dc0d4cefc529a915a601ef65679772e460905f9b5b88c7e63b30`, 905 bytes. The digest is recorded in `deploy/dl-flywheel/pipelines/generated_target_admission_v1/SOURCE_PINS.json` and `deploy/dl-flywheel/SOURCE_MANIFEST.json`.

The validator compares a prior state (`old`) with a candidate state (`new`). It requires task count and task identity to remain stable, permits only `program_results` to extend, requires existing results to remain an ordered prefix, and rejects duplicate digests. In this contract, `old` is a precise state label, not a dead-language placeholder. The finding is therefore a false positive.

The source and its generated-target qualification records remain byte-for-byte unchanged. `agent/audit-policy.toml` records a term-and-path-specific allowance expiring 2027-02-01; the file remains in the scan and other findings remain visible. Reassess against fresh source and policy evidence if the source hash or transition semantics change.
