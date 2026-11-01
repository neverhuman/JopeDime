# REL-07 R1 crash-consistency provider

R1 checks that a worker restart after a hard process death does not expose a
partial artifact or reuse a durable event sequence. The provider reuses the exact
system test `r1_sigkill_at_public_blob_and_journal_boundaries_recovers_atomic_records_and_sequences`.
That test injects actual supervisor-delivered SIGKILL at all 12 public blob and
journal boundaries, then verifies recovery to the last committed paused snapshot,
unchanged artifact roots and event prefix, absence of uncommitted records, and a
single contiguous continuation sequence. With the test-only `jope-service/failpoints`
feature and its explicit `INT02_INTERNAL_CAS_CUTS=1` switch, it also exercises all
15 internal CAS write, sync, hash, rename, and directory-sync cuts. Any absent hook,
non-SIGKILL termination, incomplete recovery, or assertion failure fails the test.

The command is `python3 -B deploy/qualification/v1.0/providers/r1.py`; it accepts
only the standard `JOPE_REL07_*` bindings for R1, a regular non-symlink candidate
whose full SHA-256 matches the runner, a non-development `full` manifest whose
source SHA matches the clean tested checkout, and a new evidence path outside that
checkout. It independently rechecks source cleanliness and candidate bytes around
verification and execution. The receipt count is one passing exact Cargo test; that
test contains the 27 separately supervised crash cuts above. The provider refuses
to pass if the exact test summary or any of the 15 internal-cut reports is absent.

The focused data-free controls are:

```sh
nice -n 10 python3 -B deploy/qualification/v1.0/test_provider_r1 -v
```

They use synthetic candidate bytes and mocked child processes only. They do not
emit a release-gate receipt or establish qualification. The selected system test
uses isolated synthetic store rows and model blobs; it reads no dataset, sealed or
holdout content and needs no GPU. This does not establish native Windows behavior,
other REL-07 gates, signer approval, or release qualification. R3 idempotency and
conflict behavior is outside this R1 provider.
