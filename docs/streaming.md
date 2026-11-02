# Streaming runtime boundary exception

## Pinned KL reference fixture unit false positive

- **Owner authorization:** Jepson's direct t1955 assignment and t205
  merge-first order (2026-11-02), recorded for this lane in `AGENT_CHAT.md` and
  the audit coordination log.
- **Owner:** `dl-flywheel` maintainers.
- **Scope:** the exact file
  `deploy/dl-flywheel/pipelines/continuous_v6/tests/fixtures/kl_ref_r22_harness_v4.py`,
  reported as HLT-019 fingerprint
  `sha256:cb962d49435f4b0ec40e4582a6c870fc1431a64925b8c56e763782c2cb6af9e0`.
  Its line 155 says “nats per program” to name the unit of natural-log KL and
  entropy values. The fixture is a verbatim reference copy, not a streaming
  client; its module bytes are pinned by `kl_pins.json` to SHA256
  `9740fc788ddf63e5630d876703fcd79495aa3246583b5279451d538c03e9cc38`.
- **Review expiry:** 2027-02-01. Reconfirm the exact finding and module pin
  before renewal.
- **Migration path:** remove this exact-file scoring exception when the
  detector distinguishes logarithmic-unit text from NATS client usage and a
  fresh full audit confirms this HLT-019 match is gone. Preserve the fixture
  bytes and `kl_pins.json` entry.
- **Evidence:** the fixture's pinned verbatim reference role, its line 155
  unit description, and the exact source report `rerun/j_cf49650df.json`.

## Telemetry entropy wire-name false positive

- **Owner authorization:** Jepson's HLT-019 ruling recorded in COORD at
  2026-11-02T14:59Z; P34 exact-file transfer recorded at 09:48Z.
- **Owner:** `jope-telemetry`.
- **Scope:** exact Jankurai HLT-019 fingerprint
  `sha256:e793ca0be77258177bd6fa1b7c935d86a4b33767463198b20b8ebadf82015948`
  in `crates/jope-telemetry/src/gp.rs`. The public Rust field is named
  `op_entropy_nat`; `serde(rename)` retains the established `op_entropy_nats`
  wire key. The value is Shannon entropy computed with natural logarithms.
  This source has no queue client or streaming runtime.
- **Review expiry:** 2027-02-01. Confirm the finding fingerprint, entropy
  calculation, and serialized key before renewal.
- **Migration path:** remove the exact-file score exclusion when the pinned
  scanner can distinguish serde wire names from Rust identifiers and a fresh
  full audit confirms the false positive is gone. Do not change the wire key
  before telemetry schema v1.0.0.
- **Evidence:** the unit test asserts serialized output keeps
  `op_entropy_nats` and does not emit `op_entropy_nat`; the full audit must
  confirm this exact HLT-019 finding is excluded while all other findings
  remain scored.

## Scoped false positive in sealed storage

- **Owner authorization:** Jepson, via the P34 packet direction on 2026-11-02.
- **Owner:** Rust maintainers (`rust` in `agent/owner-map.json`).
- **Scope:** the exact Jankurai scoring exception for
  `crates/jope-engine/src/sealed_storage.rs`. The flagged `xadd` marker at line
  332 is a substring of `UnixAddr`, imported by a test that transfers file
  descriptors over Unix sockets with `SCM_RIGHTS`. This module has no queue or
  streaming client and emits no stream records.
- **Other findings covered by this exact-file entry:** the three `mem::zeroed`
  markers at lines 25, 95, and 126 initialize C structs whose all-zero values
  are valid (`sockaddr_un` integer fields and `msghdr` pointer/integer fields);
  the `assume_init` marker at line 256 follows successful `getsockopt` and an
  exact `ucred` size check. Each operation has a local `SAFETY` invariant. The
  exception suppresses only Jankurai's scoring for this exact file; Rust tests,
  Clippy, Gitleaks, and other scanners still inspect it.
- **Review expiry:** 2027-02-01. Recheck all five findings covered by this
  exact-file entry; renew only while each remains a verified false positive
  and the source behavior is unchanged.
- **Migration path:** no runtime migration applies to the current code. If a
  queue or streaming client is introduced, put it behind the declared
  `crates/adapters/queues` boundary before use and remove this exception after
  the detector no longer matches the test import.
- **Evidence:** existing SCM_RIGHTS and sealed-storage regression tests in
  `sealed_storage.rs`, the per-file Jankurai advisory result, the local safety
  invariants cited above, and the review in
  `docs/reviews/hpc-audit-20261021.md`. Keep the descriptor-transfer controls;
  do not rename their identifiers to influence the detector.
