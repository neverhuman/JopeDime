# Streaming runtime boundary exception

## Telemetry entropy metric false positive

- **Owner authorization:** Jepson, via the P34 cap assignment on 2026-11-02.
- **Owner:** `jope-telemetry`.
- **Scope:** the exact Jankurai scoring exception for
  `crates/jope-telemetry/src/gp.rs`. HLT-019 matches `nats` inside
  `op_entropy_nats`, the generation entropy statistic computed with natural
  logarithms (`p * p.ln()`). The file implements telemetry statistics and has
  no queue or streaming client.
- **Review expiry:** 2027-02-01. Recheck the metric implementation and client
  inventory before renewal.
- **Migration path:** remove this exact-file exception after the pinned
  detector uses token-boundary matching and a fresh full audit confirms the
  metric field no longer matches.
- **Evidence and limits:** `Tracker::summarize` computes Shannon entropy from
  operation frequencies using `ln`; the per-file exception suppresses only
  Jankurai scoring for `gp.rs`. Rust tests, Clippy, Gitleaks and other scanners
  continue to inspect the file.

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
