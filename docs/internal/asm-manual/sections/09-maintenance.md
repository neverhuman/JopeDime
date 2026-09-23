# Maintaining, reviewing and recovering the product

## Repository authority

The canonical product checkout is `/home/ubuntu/JopeDime`, upstream `veox-ai/JopeDime`. `/home/ubuntu/jain-split/jope-dime` is a different research repository; `/home/ubuntu/jope-dime` may be its symlink. Similar names are not interchangeable source, checkpoint or promotion authorities.

Before source changes, inspect the open PR queue, worktrees, branch/full HEAD, dirty state and latest `AGENT_CHAT.md`. Under `flock` on `AGENT_CHAT.md.lock`, reread overlapping claims and append checkout, branch, full HEAD, owner, paths and verification. The append lock does not permit two writers in one checkout. An idle session or old claim is not a stopped-head handoff.

Keep one writer per checkout and one implementation PR queue. Preserve pending work and authorship. A resumed campaign retains its resource limits and deadline.

## Worktrees and builds

Extra copies belong under `<repo>/.agent/worktrees/`, at most four lanes. `.claude/worktrees` stays a symlink into that shared root. Verify it before worktree operations and keep `.agent/` locally excluded from Git status.

Each lane reuses its own `target/`. Before Cargo:

```sh
export CARGO_INCREMENTAL=0
export CARGO_PROFILE_DEV_DEBUG=line-tables-only
export CARGO_BUILD_JOBS=4
```

Do not set `CARGO_TARGET_DIR` globally or create a new target per task. CI selects persistent PR and trusted-main storage separately. Resolve binaries from Cargo artifact messages: guessing a hashed test executable can select an old binary built with different features.

Never put source, worktrees or builds in `/tmp`. A target directory can contain evidence. Cargo markers do not prove that every descendant is disposable.

## Evidence checklist

Retain source revision and input manifest; exact command, directory, exit and timestamps; executable/runtime/features/environment; data/split/evaluator/vocabulary/protocol identities; original limits/start/deadline/cutoff; collected/executed/failed/skipped/excluded/unavailable inventories; artifact hashes and worker receipts; report replay; limitations and next owner.

Verify a source manifest against the committed blobs it claims. The current bytes at `target/release/jope` need not be the bytes used by an earlier result. Preserve immutable executable copies with their build receipts before another build replaces the usual path.

Current ASM custody is outside Git at `/home/ubuntu/jope-rescue-20260921/asm-integration-20260923/`. This host location is not a portable runtime default.

## Coverage and reviewed lifecycle

Read `deploy/ci/coverage.md`, mapping and generated inventory together. Discovery counts packages, features and families; execution receipts count what one invocation actually collected and ran. Invocations overlap. Path exclusions, ignored tests and unavailable dependencies do not become passing tests.

Fast CI has eight lanes plus aggregate. Full-main diagnostics cover additional families and have recorded failures/unavailable capabilities. A fast gate does not establish all-feature, GPU or live-S3 qualification. Regenerate inventories and scores through their owning proof scripts; never hand-edit generated evidence.

The established writer is `jepsontaylor`; the designated independent Claude reviewer uses `neverhuman`. Verify them for every PR. The reviewer must differ from the opener and every associated commit author and committer. Use the full paginated inventory and record unassociated commits.

```sh
/home/ubuntu/.local/bin/gh-role writer api user --jq .login
/home/ubuntu/.local/bin/gh-role eligible reviewer veox-ai/JopeDime PR_NUMBER
```

The helper prevents identity mistakes but cannot prove independence. Changing credentials does not turn a writer into a reviewer. Never print tokens or credential-bearing environments.

Freeze the full head; dispatch fresh manual CI; retain source SHA, run ID, attempt and dispatch time. Verify current-attempt jobs/aggregate and nonzero meaningful coverage. Supply the independent reviewer with diff, acceptance commands and evidence. A native approval must explicitly bind the final head and remain undismissed. Changed source repeats checks and review.

```sh
/home/ubuntu/.local/bin/gh-role writer pr merge PR_NUMBER \
  --rebase --match-head-commit FULL_REVIEWED_SHA
```

Immediately recheck head, eligibility, CI and approval before merging. Afterwards verify `merged_at`, fetch remote main, compare the reviewed tree and verify main-guard. Preserve any ledger tail or other dirty work before restoring clean verified main. Protection enforcement has been procedural; a protection API 403 does not prove a preventive branch lock. Main-guard detects unreviewed pushes after they occur.

Main-guard and the full-main diagnostic share the single xbabe2 `jope-full-cpu` slot and run serially. A main push queues both; full-main has a 120-minute timeout, so main-guard can wait behind it. Read back job/run states and retain the exact source/run/attempt before changing the queue. In the PR83 lifecycle the owner-authorized full-main diagnostic was cancelled while still queued, before any job started, and the owned full timer was restored to allow main-guard to run. That is a recorded unexecuted diagnostic, not passing coverage or general permission to cancel running work. Follow the current allocation owner and runner runbook; never stop another running job to accelerate a merge. Sources: `.github/workflows/main-guard.yml:17`, `.github/workflows/full-main.yml:24`, `docs/runbooks/runner.md:15`, `:131`; the PR83 lifecycle cancellation receipt.

## Shared compute

Fast CI runs on xbabe1. The canonical Ubuntu machine is local xbabe2 for full CPU work. The user's SSH alias `xbabe2` points elsewhere; use the runbook's local form for this machine.

The exclusive full runner and numerical campaigns share a host lock. Drain owned listeners and verify no active full job before numerical allocation. Do not seize the host by stopping another job. Record allocation ownership and restore the runner after numerical work releases it. Keep full CPU CI separate from the one-GPU-hour training campaign and the frozen panel.

Missing cgroups or dependencies cause explicit failures/unavailable outcomes; they do not authorize an unbounded fallback. Worker control readbacks, not prose, establish the limits actually enforced.

## Recovery phases

Identify whether failure occurred before admission, in setup, after the gate, during cleanup, in controller scoring or during report generation.

- A pre-admission dependency refusal may be retried only inside the original window.
- Setup can consume CPU without starting a fit. Unknown setup cost stays unknown.
- A started identity is durable. Controller death does not authorize silently repeating it.
- Completed folds inside a failed candidate still require artifact authentication.
- Original-cutoff inventory settles without extending the clock.
- A report can fail after successful fits. Retain both; do not call the whole operation successful.
- Saving a terminal outcome does not release cleanup authority before descendants drain.

Resume with the same plan and journal. A changed source, dataset, budget or protocol requires a separately identified experiment with the old evidence retained, not a relabelled continuation.

## Preservation and lane closure

Before deletion inspect `git status`, `git stash list`, `git log --all --not --remotes`, and local versus remote tags. A branch-only log misses commits elsewhere.

Preserve unique source/evidence for the tree being removed, restore independently into the reusable bare verification directory without alternates, and verify the exact commit, tree and connectivity. A valid bundle checksum does not prove the desired detached HEAD was included.

After preservation and a fresh consumer check, remove a closed lane with `git worktree remove`. Indefinite redundant copies are not an alternative to the required reclamation step. Unknown contents remain protected. Follow `deploy/maintenance/README.md`; its reporting timer proposes candidates and never authorizes deletion.

## Updating the book

Refresh inventories and receipts, then section Markdown, figures, checklist and PDF. Keep a rendering input manifest. A checkpoint does not update performance figures by implication; merged training source does not prove actual learning. Historical Elo remains historical after regeneration.

Before delivery check local links, source identities, totals, metric directions, captions, capability status and rendered pages. Extract PDF text and inspect dense tables, examples and figures. Preserve unavailable statistics and measurements explicitly rather than filling them with zero.
