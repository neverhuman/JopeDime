# REL-07 U1 candidate-bound editing-under-load provider

The provider runs the existing Linux PTY test for typing, bracketed paste,
cursor/menu preservation under 50 Hz incoming events; the composer history
privacy/permission test; and the Linux PTY fallback/resize test that narrows
the terminal below 40 columns while preserving the draft. Each exact selector
must pass once. Zero collection, a wrong selector, failure, ignore, measured
case, candidate drift, dirty source or source mismatch refuses a PASS.

Run through release-gates as:

    python3 -B deploy/qualification/v1.0/providers/u1.py

The standard five JOPE_REL07 bindings supply gate ID, candidate path and
SHA-256, source SHA and a new result path. The provider requires the U1 ID,
regular non-symlink candidate, full non-development verified envelope, exact
clean source checkout and result path outside that checkout. Candidate and
source identities are checked before and after each selected test. Its child
result follows jope.rel07.gate-execution/v1 and includes only the exact
selected cases; unrelated library tests filtered by the exact selectors are
outside this gate's case inventory.

The local controls are synthetic and data-free:

    nice -n 10 python3 -B -m pytest -v deploy/qualification/v1.0/test_provider_u1

These controls verify the command/result contract only. They do not use a
release candidate or qualify U1.

## Limits

The current exact PTY acceptance tests are Linux-gated; the provider refuses
other hosts rather than emit a zero-test PASS. The provider records Linux
candidate-bound evidence only. It does not claim cross-platform U1
qualification, invoke a signer, or access datasets, sealed inputs or a GPU.
The full product gate remains governed by the source criteria and platform
evidence in gate-status.md.
