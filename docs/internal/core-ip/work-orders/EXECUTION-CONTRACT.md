# Shared execution, evidence and delivery contract

Applies to every W1–W12 task. User direction and the repository's
[AGENTS.md](../../../../AGENTS.md) control authorization. This planning document
does not supersede stronger receiving code or permit bypassing a held authority.

## Roles and task state

The implementation owner owns code, reproductions and receipts. The independent
Codex reviewer did not write the candidate, uses the eligible review identity only,
and reviews the full final head. A controller owns runtime allocations and draining;
a credential owner supplies the publisher credential; a scientific decision owner
authorizes any separately defined serving promotion. These are required roles, not
claims that every role or credential is already provisioned.

For each task retain: task ID; owner; branch/PR; full source SHA; dependencies;
state; reproduction; acceptance command; input identities; evidence path; blocker;
review and CI identities. Allowed progress descriptions are `planned`, `reproduced`,
`implemented`, `fixture-proven`, `CI-verified`, `live-qualified`, `measurement-complete`,
and `blocked`. They are scoped observations, not an automatic ladder. Promotion is
its own decision. Record multiple observations when a fixture passes but live proof
is blocked. An unchecked task is not implicitly waived.

The current findings schema deliberately refuses closure attestations. Do not edit
it to say `closed` without first implementing and reviewing the evidence-validation
schema extension described in W12. Until then, retain task receipts externally and
keep the original findings open with their original hashes and predicates.

## Source and evidence custody

- Claim before edits; reread PR queue, worktrees and AGENT_CHAT under its append lock.
  Preserve predecessor work and do not infer release from inactivity.
- Use the canonical claimed branch or a registered worktree under `.agent/worktrees`.
  At most four secondary lanes; one writer per checkout; no new clone or build in `/tmp`.
- Use one reused `target/` per lane. Before Cargo set `CARGO_INCREMENTAL=0` and
  `CARGO_PROFILE_DEV_DEBUG=line-tables-only`; do not set a global `CARGO_TARGET_DIR`.
- Keep bulk inputs, checkpoints and receipts outside the source tree. A proposed
  custody layout is `regression-remediation/RUN/Wnn/TASK/ATTEMPT/`; resolve its actual
  root in the signed plan. Never overwrite a prior attempt's evidence.
- Preserve receipts and independent recovery before deleting a closed lane or any
  evidence-bearing build tree. This work plan authorizes no blanket cleanup.

Each execution receipt must bind the following, using existing receipt formats
where possible rather than inventing a parallel authority:

| Field group | Required contents |
|---|---|
| Source | Full commit, dirty-source inventory, executable/runtime digests, backend and feature selection |
| Invocation | Exact argv, working directory, start/end time, exit status, environment allowlist |
| Inputs | Release, family/split, row order, recipe/fitted subject, context/bundle, corpus and parent digests as applicable |
| Authority | Original approval/grant, receiving validator, permitted consumer and label role, resource ownership |
| Budget | Original deadline, CPU/RAM/GPU UUID limits, attempt counters, actual use, retries and remaining allowance |
| Outcomes | Collected/executed/passed/failed/skipped/excluded counts, reasons, terminal dispositions and unknowns |
| Integrity | Output sizes/hashes, committed-state identity, dependency closure, evidence location and verification result |
| Review | PR head, CI run/attempt/source and checkout SHAs, independent review ID, merged main and main-guard |

Do not write secrets, raw private labels, broad environment dumps or credentials to
receipts. Unknown counts remain unknown. Keep source/publication identity distinct
from the original execution source. A file hash authenticates bytes, not the truth
of a producer's assertion or permission to consume those bytes.

## Test design and command discipline

Every behavioral repair needs a failing pre-fix reproduction, corrected receiving-path
proof, and an authorized positive control. Negative tests must observe that no
authority-bearing state changed: no eligible teacher, committed HWM improvement,
corpus inclusion, optimizer update, winner publication or serving release.

For unverified safeguards first trace actual calls; close an excerpt-only concern by
demonstrating existing behavior and regression protection. For research, preregister
the comparison and report negative/inconclusive outcomes honestly. Historical gaps
cannot be repaired by generating new evidence and assigning it an old identity.

Command blocks in individual orders are **existing collection/discovery commands**,
not an assertion that the acceptance harness already exists or currently passes.
`-- --list` may compile tests and consumes a build budget. Record the collection
before execution and assign exact new tests to the maintained CI coverage map.
An owner must replace each proposed acceptance procedure with a real argv and
receipt before claiming execution. Do not paste an invented future CLI into a runbook
as if it were available. Feature `testing` is for fixtures only, never product builds.

## Proposed phase ceilings, awaiting phase admission

| Phase | Ceiling from original plan | Gate before launch |
|---|---|---|
| Local contracts | 4 CPUs, 16 GiB, 15-minute batches | Claimed source, fixtures and explicit bounded command |
| Three-host CPU lifecycle | 4 CPUs/8 GiB per host, 15 minutes | Managed allocations, host identities, no CI conflict |
| Individual GPU | One GPU, 8 CPUs/16 GiB per host, 15 minutes per host | Actual GPU UUID/free memory, runtime qualification and grant |
| Two-host synchronous | xbabe2 + xbabe3, same per-host caps, 20 minutes | Individual proofs, authenticated participants and group drain |
| Three-host synchronous | All three, same caps, 20 minutes | Two-host proof and lossless three-rank objective contract |
| Real loop smoke | Fixed admitted allocations, 60-minute global window | W1–W9 applicable gates and explicit complete work census |
| Initial live S3 | Run-scoped prefix, 20 GiB and 20,000 writes maximum | Scoped publisher credential and approved request/byte accounting |
| Transfer/research | Separately costed frozen manifests | Full planned coverage and sufficient budget; no simplification to fit |

These are not renewed campaign allowances. Each admitted phase records one deadline;
retries, restarts, topology transitions and recovery retain applicable original
limits. Include verification, failures, artifact movement and cleanup in the work
census. Define exactly which S3 API calls count toward the approved write allowance
before admission; include retries and failed conditional writes, not just successes.
If capacity or approval is missing, queue/refuse that phase and continue independent
source work. Never interrupt unrelated jobs or borrow CI reservations.

## Serial PR lifecycle and rollback

Use one focused implementation PR at a time. Each proposed slice has a concrete
behavior, acceptance command and durable evidence. Preserve attribution and shared
history. Review uses a separate Codex process under the owner's explicit override
of Claude-specific review; the reviewer must differ from the PR author and every
GitHub-associated commit author and committer. Record unassociated identities.

Freeze the full head, dispatch `ci.yml`, identify the new dispatch time/run/attempt,
and require all eight manual lanes plus aggregate at that head. Retain the PR's
synthetic checkout SHA separately. Read back an eligible undismissed native approval
on that exact head and resolve blockers. Any source change repeats CI and review.
Merge with `gh pr merge NUMBER --rebase --match-head-commit FULL_SHA` under the verified
writer identity; verify remote main, the resulting source tree and main-guard.
The established helper's warning about writer merging is not independent review;
the native eligible review remains mandatory.

Rollback preserves accepted weights, original manifests and all failed evidence.
Prefer a reviewed revert or a new explicit protocol/release identity; never erase
history, relabel a challenger accepted, reset a deadline, or force-update committed
remote pointers. If a prerequisite is revoked, quarantine descendants and pause
new affected work at a committed boundary while retaining reconstructable history.
