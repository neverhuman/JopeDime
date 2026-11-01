# REL-07 U2 candidate-bound terminal-restoration provider

The provider runs the existing Linux PTY assertions for normal exit, error,
panic, SIGINT, SIGTERM and SIGHUP restoration; NO_COLOR, TERM=dumb and narrow
terminal fallback behavior; no-color fixture output; reset commands; color
detection; and control-sequence sanitization. Every exact selector must pass
once. Zero collection, a wrong selector, failure, ignore, measured case,
candidate drift, dirty source or source mismatch refuses a PASS.

Run through release-gates as:

    python3 -B deploy/qualification/v1.0/providers/u2.py

The standard five JOPE_REL07 bindings supply gate ID, candidate path and
SHA-256, source SHA and a new result path. The provider requires the U2 ID,
regular non-symlink candidate, full non-development verified envelope, exact
clean source checkout and result path outside that checkout. Candidate and
source identities are checked before and after each selected test. Its child
result follows jope.rel07.gate-execution/v1 and includes only the exact
selected cases; unrelated library tests filtered by the exact selectors are
outside this gate's case inventory.

The local controls are synthetic and data-free:

    nice -n 10 python3 -B -m pytest -v deploy/qualification/v1.0/test_provider_u2

These controls verify the command/result contract only. They do not use a
release candidate or qualify U2.

## Limits

The current restoration/session tests are Linux-gated; the provider refuses
other hosts rather than emit a zero-test PASS. It records Linux
candidate-bound evidence only. Spec §14 U2 also requires native macOS and
Windows ConPTY restoration evidence; this provider does not claim those
platforms or full U2 qualification. gate-status.md remains unchanged. No
signer, dataset, sealed input or GPU is used.
