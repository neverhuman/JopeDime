# Regression remediation findings

This is the initial W0 custody and routing inventory for the regression remediation
plan. It preserves **97 gaps and 98 improvements** from ten original audit shards.
It closes no product finding. Reproduction commands, receiving-code triage, and
operational evidence remain outstanding; W0 as a whole is not complete.

`register.json` preserves each original ID (including zero padding), title, source
and section SHA-256, line range, and original detection/hardening/acceptance clauses.
The immutable Markdown snapshots are in `sources/`. Their original metadata refers
to an excerpt-only audit, not an independent examination of the whole repository.
The snapshots' embedded acquisition hashes and the register's whole-file hashes
identify different objects and must not be interchanged.

Each entry names a primary work package. `packages` maps that package to a receiving
boundary, subsystem, source paths, and dependencies. These are proposed responsibility
assignments, not evidence of enforcement or staffing. Cross-cutting acceptance
clauses still apply even when an entry has only one primary package.

Every entry retains its own planned evidence location and separate attestation
slots for source implementation, fixture proof, hosted CI, live qualification, and
promotion. An absent reproduction command is an explicit unresolved task. V1 admits
only open findings: a reviewed schema extension with evidence validation is required
before this register can authorize closure. It must not turn a populated field into
a product capability claim.

Regression and shared infrastructure are in scope. Specialized binary, multiclass,
and time-series cases remain deferred within the original text; this does not defer
shared rejection, identity, isolation, or accounting controls. The original thirty
dataset panel remains historical; W1 must derive a distinct ten-regression release.

Run from the repository root:

```sh
python3 docs/internal/core-ip/tools/verify_findings.py
python3 -m unittest discover -s docs/internal/core-ip/tools -p test_verify_findings.py -v
git diff --check
```

These commands verify inventory integrity only. The test suite includes a valid
inventory and adversarial omissions, duplicate IDs, renumbering, changed hashes,
missing acceptance, evidence path traversal, unsupported closure, and promotion.
Original source whitespace is preserved for digest identity.

## Baseline and next gates

`baseline.json` retains counts, commands, exit statuses, failed test names, and
hash-addressed custody references from diagnostic run **36195898277 attempt 1** at
`418ef104c06a3a08b594cfcd6a5687dfeb882552`. The run's failed status was read back from
GitHub on 2026-09-26. It is not a new execution or a regression-only coverage total.
Default: 3,040 collected, 3,002 executed, 16 failed, 38 skipped. Scratch: 1,963
collected, 1,947 executed, 17 failed, 16 skipped, exit 124. Neural: exit 78 and
unknown counts. Python: 344 collected, 329 passed, 15 excluded. Classification
failures remain visible pending individual triage.

Next W0 work must reproduce the fit-authority/view-contract failures through the
legitimate issuer, preserve real-corpus fixture identities, diagnose scratch
timeouts, repair native SDK discovery, and collect a test inventory spanning
preparation, encoders, policy, GP, fitting, teaching, training, storage, and export.
Existing `deploy/ci/regression-suite.sh` is a separate frozen Golden42 campaign;
its 60-core requirement is not authorization to execute it under this plan.

The serial dependency order is W1 input sealing; W2 accepted semantic assembly;
W3 receiving provenance; W4 generator conformance; W5 fitting/selection ancestry;
W6 corpus/objective/resume; W7 three-host orchestration; W8 durable S3 publication;
W9 fitted export; W10 loop/transfer measurement; W11 performance; W12 claims/replay.
W7 and W8 are additional user requirements, not extra shard entries. Their acceptance
requires actual repository-managed jobs on all three hosts, real optimizer updates,
group drain and same-topology recovery; then conditional S3 conflict, dependency
verification, restoration, and resumed computation. Source-only controls cannot
close these requirements.

The plan's compute ceilings remain proposed until approved. Resumption never renews
a deadline. S3 qualification also requires a scoped publisher credential from its
owner; the available administrator profile is not a substitute. No IAM change,
live training, S3 write, or serving promotion follows from this inventory.
