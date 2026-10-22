# Audit execution and evidence repair — 2026-10-22

This batch repairs execution and evidence validation. It does not certify the
repository against the audit's 85-point gate.

The complete tracked-source audit at
c69ad5a6e49a8331d899288820a5723bf2a891c3 used Jankurai 1.7.5 and failed at
52/100: 354 findings, including 270 hard findings. Its tree is identical to
receiving main 7c3c7ed43ebb298d629656fa9cac3d5e807ca7b3. The raw report remains
unchanged; later source needs its own complete audit.

## Repaired boundaries

The dedicated hosted workflow is gated by an unavailable runner opt-in.
The independent audit job in full-main.yml uses the already-admitted
trusted-main workflow and jope-full-cpu runner. It is scheduled on main pushes
and main workflow dispatches, checks runner provenance, installs the existing
checksum-pinned 1.7.5 release and invokes the canonical immutable-input wrapper.
It preserves the eleven CPU families and their aggregate. A failed audit makes
the workflow fail independently of the CPU aggregate.

The old wrapper trusted the auditor's exit status alone. Controlled fake-auditor
tests reproduced zero-exit success with missing reports, a failed decision and a
lowered score floor. The wrapper now pins the input manifest before execution,
validates all required reports and checks the complete captured input afterward.
A valid failed audit retains its original nonzero exit; invalid or missing
evidence cannot become a passing audit.

The verifier binds the pinned auditor/schema/standard and default policy
fingerprint, full scope, unchanged 85 floor, hard/soft finding counts and policy
decision to the exit status. It checks the archive, every captured Git blob's
bytes and read-only mode, the complete path set, and the original manifest pin.
It emits audit-verification.json beside the original reports. These checks
establish artifact integrity and a consistent gate result; the pinned auditor
remains responsible for the analysis.

Evidence publication runs after failed audit gates too, through the existing
local-retention/cache transport with source/run/attempt-scoped names. Missing
tools, invalid source, missing reports and failed publication remain failures.
No installed runner, admission policy, storage service, model, generated
numerical fixture or scientific protocol is changed.

## Remaining findings

The broader report still requires substantive remediation and contextual review.
Examples already established from source include synthetic credential controls,
frozen historical test bodies, the UnixAddr substring reported as a streaming
client and a corpus verification option reported as a TLS override. Those
observations do not suppress findings or change the score. Python ownership,
duplication, code shape and unresolved runtime findings need separate focused
repairs. Frozen learner and history bytes remain preserved.

## Verification

Run the fault controls and existing scheduling controls with the repository's
pinned tool paths:

    python3 -B deploy/ci/test-audit-lane.py
    python3 -B deploy/ci/test-full-main.py
    actionlint .github/workflows/full-main.yml

The controls use isolated fixture repositories and a fake auditor. They are
execution/validation tests, not an actual Jankurai score. Use just audit for
a fresh full-source audit after committing source. Full manual CI, exact-head
review and receiving-main checks remain separate delivery requirements.
