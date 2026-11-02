# Streaming runtime boundary exception

## Scoped false positive in sealed storage

- **Owner:** Rust maintainers (`rust` in `agent/owner-map.json`).
- **Scope:** `crates/jope-engine/src/sealed_storage.rs`, line 332 in the current
  source. The `xadd` marker is a substring of `UnixAddr`, imported by a test
  that transfers file descriptors over Unix sockets with `SCM_RIGHTS`. This
  module has no queue or streaming client and emits no stream records.
- **Review expiry:** 2027-02-01. Recheck the source and the finding at that
  date; renew only if the same test-only false positive is still confirmed.
- **Migration path:** no runtime migration applies to the current code. If a
  queue or streaming client is introduced, put it behind the declared
  `crates/adapters/queues` boundary before use and remove this exception after
  the detector no longer matches the test import.
- **Evidence:** the existing SCM_RIGHTS regression tests in
  `sealed_storage.rs`, the per-file Jankurai advisory result, and the review in
  `docs/reviews/hpc-audit-20261021.md` confirm the match is lexical rather than
  a streaming runtime dependency. Keep the descriptor-transfer controls; do
  not rename their identifiers to influence the detector.
