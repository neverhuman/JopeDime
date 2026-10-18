# Research workflow status

This runbook describes the read-only status view and its evidence rules. It does not admit a
campaign or grant permission to train, publish, promote or access protected data. The view is a
thin adapter over producer receipts; each producer remains the authority for its own record.

## Status dimensions

Keep these evidence stages separate in the receipts and in operator reports:

| Stage | Evidence to inspect |
| --- | --- |
| Source | Full source commit and tree identity |
| Fixture | Exact command, inputs, collected and executed counts, and result |
| CI | Full tested source head, workflow run ID and attempt |
| Live | Runtime qualification receipt and its admitted limits |
| Science | Matching task, dataset, cut, evaluator, vocabulary and protocol identities |
| Serving | Explicit promotion receipt and serving pointer state |

An absent stage is unavailable. A successful fixture does not imply hosted CI, live qualification,
scientific improvement or serving promotion. A command string records configuration only. The view
reports execution only when an execution record includes the original argv, exit code, start/end
times, source hash and input hash.

## Receipt-backed view

The CLI adapter accepts one or more producer receipt JSON files. For example, after the integrator
wires the adapter into the umbrella `jope` dispatcher:

```sh
jope research status \
  --receipt target/task-evidence/I61/receipt.json \
  --stale-before-utc 2026-10-17T00:00:00Z \
  --json
```

The stale cutoff is chosen from the applicable evidence policy; the adapter does not invent a TTL.
The report includes the observer time and the producer's original timestamp. Stale evidence is
labeled `dated`; neither the source timestamp nor its source identity is rewritten. If the
timestamp is absent or not a normalized UTC timestamp, freshness is `unknown`.

`research verify` checks that each supplied file is readable JSON with an object at its root.
Receipt schemas and producer-specific validity checks remain with their owning libraries.
The worker adapter currently routes `research doctor`, `research status`, `research verify` and
`research brief` through the same read-only receipt evaluator; command-specific formatting remains
integrator work. None executes the commands named inside receipts.

At this worker revision the adapter is available as `jope_cli::research_view`; the umbrella CLI
dispatch and the `jope doctor` / `jope train` integration are serialized with the integrator. Do
not treat the example command as proof that the dispatcher has shipped. Until that wiring lands,
use the owning producer's receipt and report the unavailable CLI capability with stable reason
`capability_unavailable` and exit code 2.

## Reproduction and interpretation

Retain the receipt files, full source identities, original argv, input identities, toolchain,
resource limits, UTC start/end, exit status and test counts alongside any status report. Read the
receipt at its recorded path; do not infer state from a narrative board, a help listing, a built
binary or the presence of a configured command.

The view is read-only. Any future mutating verb needs an explicit intent/grant and a dry-run path;
it must call the owning library rather than reimplement policy in CLI code. Keep prior readers and
input identities available during migration. Roll back through the reviewed PR by disabling the
new version/profile, preserving receipts and evidence, and reverting through the same reviewed
lifecycle.
