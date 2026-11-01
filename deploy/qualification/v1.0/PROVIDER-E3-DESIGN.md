# REL-07 E3 export-safety provider

Spec §14 E3 requires atomic export failure, overwrite control, archive and
destination safety, and accurate training-data retention. This provider uses
the existing shared full-candidate verifier, requires its exact clean source,
and executes seven named Rust controls from `jope-service` and `jope-client`:

- Missing required language: writer is never called.
- Manifest path traversal and malformed content hashes: rejected.
- Training-row heads: retention true; ordinary Ridge: retention false.
- Explicit force: verified extraction and the existing output preserved as backup.
- No force, wrong digest, or interrupted archive: existing output unchanged.
- Archive traversal and symlink entries: rejected.
- Retry verification: tampered bytes and symlinked output refused.

Run `python3 -B deploy/qualification/v1.0/providers/e3.py` with standard
`JOPE_REL07_{GATE_ID,ARTIFACT,ARTIFACT_SHA256,SOURCE_SHA,RESULT}` bindings and
`GATE_ID=E3`. The result path must resolve outside the source checkout.

Every selected test must run exactly once under `--exact`, with one unambiguous
Rust summary, its exact test name, and no ignored, failed, measured or additional
cases. Other tests filtered out by that explicit selector are outside this
provider's seven assertions. Source and candidate hashes are checked around
each command. Failure, timeout, identity drift or ambiguous output produces no
PASS receipt. Exclusive creation preserves an existing or concurrent result.

A generic positive Cargo exit or broad test filter was rejected: it could pass
with zero relevant tests. Reimplementing export logic here was rejected in favor
of the product's existing regression controls. Shared provider helpers, runtime
export code, schemas, dependencies and signing are unchanged.

The data-free provider tests use mock child results to validate fail-closed
orchestration; they do not establish release qualification. A real E3 result
requires the full non-development candidate, matching clean source and all seven
actual Rust controls. No candidate was supplied for this implementation, so no
actual E3/28-gate release PASS, signing or overall REL-07 completion is claimed.
