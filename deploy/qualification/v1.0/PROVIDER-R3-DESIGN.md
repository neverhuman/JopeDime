# REL-07 R3 idempotency and conflict provider

R3 runs seven exact `jope-service` library tests against the candidate-pinned
checkout. The cases cover replay of the original outcome after reopening the
store, principal separation and concurrent same-key retries, effect/receipt
rollback, malformed saved receipts, identical HTTP response replay, typed HTTP
409 conflicts without a second effect, and retry after crashes before commit and
after commit. The crash supervisor test enables the existing `failpoints`
feature; its child-process failures are expected internally, while the exact
supervisor test itself must pass.

The provider accepts only the standard R3 candidate/source/result bindings. It
requires a full, non-development candidate whose embedded source SHA matches a
clean checkout, rechecks the full candidate bytes and source before/after each
test, rejects symlink and in-checkout evidence paths, and creates a new receipt
exclusively only after all seven exact test selectors pass. Each selector must
report exactly one collected and executed passing test with no failures,
skips, or ignored cases.

The data-free controls are run with:

```sh
nice -n 10 python3 -B deploy/qualification/v1.0/test_provider_r3/test_r3 -v
```

Those controls use synthetic candidate bytes and mocked children; they do not
write a release-gate receipt or establish qualification. The provider's real
selectors use synthetic store fixtures in the existing Rust tests and no
datasets, sealed or holdout content, or GPU. This source slice does not establish
whole-system release qualification or behavior outside the tested service
component.
