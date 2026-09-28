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

Run from the repository root (the verifiers are the Rust crate `crates/jope-records`):

```sh
cargo run -q -p jope-records -- verify-findings
cargo run -q -p jope-records -- verify-dispositions
cargo run -q -p jope-records -- verify-test-inventory
cargo nextest run -p jope-records
git diff --check
```

## Dispositions and evidence (W12-T02)

`register.json` stays the immutable open inventory. Scoped dispositions are appended
to `dispositions.json`, which binds the exact register digest. `jope-records verify-dispositions`
accepts a closure only with stage-appropriate evidence under `evidence/<finding>/`.
Each evidence file must match its SHA-256 and name the same finding, stage and source commit.
A confirmed defect needs a pre-fix failure at a distinct commit plus negative and positive
tests observed passing, source trace and hosted CI. W7/W8/W10/W11 routes also need live
qualification, and promotion needs the scientific decision owner. Synthetic fixtures never
back hosted CI, live, measurement or promotion stages, and evidence whose bytes carry the
conformance `synthetic_identity` (W12-T07) may only be a source trace or a synthetic fixture,
whatever kind it declares. `historical-unknown`,
`missing-execution-evidence` and `deferred-non-regression` never close.
The committed record is empty, so every finding remains open.

## Maintained regression test inventory (W0)

`test-inventory.json` classifies every test-bearing item of `deploy/ci/coverage.json`.
The roles are regression, real-corpus regression, deferred classification, historical,
research and infrastructure. It maps regression items to the subsystems preparation,
encoders, policy, GP, fitting, teaching, training, storage, export, orchestration,
performance, evaluation and evidence, each owned by work packages.
`jope-records verify-test-inventory` requires complete classification and an executing regression
item for each of the nine W0 subsystems. It also requires exactly one triage record per
preserved baseline failure. Real-corpus, fence and deferred failures can never be marked
fixed; `fixed-locally` requires retained evidence. The map is static: it neither collects
nor executes tests.

Real-corpus fixtures read their inputs through the ADR-0003 roots
(`SOTARENA_ROOT`, `JOPE_STATE`, `JOPE_REGISTRY`). Those inputs exist only in host custody,
so every CI family declares the fixtures non-executable: each is `#[ignore]`d with a reason
starting `real-corpus:` (or `deferred-classification:` for binary42) and listed in
`deploy/ci/real-corpus-tests.txt`. They run in the host lane,
`just test-real-corpus <custody-root>` (`deploy/ci/real-corpus-lane.sh`, coverage lane
`host-real-corpus`). It refuses a writable or drifted custody copy, checks that the list
still names exactly the ignored tests, points the roots at the copy and runs only those
tests. An input the copy lacks still fails there with an explicit `unavailable:` message
instead of being skipped. A read-only custody copy of the regression fixture inputs, with
`MANIFEST.sha256` and `PROVENANCE.txt`, is kept outside the source tree. Host-lane outcomes
are host evidence, not CI verification, so these triage records are `moved` (passing in the
host lane) or stay `open`.

The neural build deliberately refuses the CPU fit-service binding. Its 80 CPU-only
service contracts remain executable in both default and scratch families and are
explicitly ignored only with `neural`. `deploy/ci/neural-cpu-only-tests.txt` names
every case; `check-test-routing.sh` checks their presence and ignore state against
the actual nextest collection before each family runs. It also checks the reverse, from
the source attributes: a test ignored for a `cpu-fit-service:` reason that is not listed,
or a listed test that no longer carries one, fails; the same holds for the `real-corpus:`
and `deferred-classification:` fixtures and `deploy/ci/real-corpus-tests.txt`. The neural refusal test
continues to execute. The campaign and ranked-proposal public entrypoints still
refuse unavailable GPU grants; separate private-core fixtures test post-admission
behavior without making a production grant or claiming GPU qualification.

These commands verify inventory integrity only. The test suite includes a valid
inventory and adversarial omissions, duplicate IDs, renumbering, changed hashes,
missing acceptance, evidence path traversal, unsupported closure, and promotion.
Original source whitespace is preserved for digest identity.

## Baseline and next gates

The [2026-09-27 health routing record](health-routing-20260927.md) records the later
reproduction, repair boundaries, explicit exclusions and bounded local results.

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
