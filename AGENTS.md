# JopeDime agent contract

The operating contract for contributors and automated agents working in this repository.

The 2026-09-21 audit found that `main` carried **no root `AGENTS.md`** — one existed only on an
unmerged branch, so an agent working from a fresh clone had no contract at all. That gap is a direct cause of the
disk incident recorded in [`MASTER_CLEAN_UP.md`](MASTER_CLEAN_UP.md).

---

## Direction and evidence

The product repository is `veox-ai/JopeDime`; its current goal is the bounded neural GP learning
loop in [MASTER_PLAN.md](MASTER_PLAN.md#product-target). Dataset labels stay intact; outer test
folds and sealed meta-test families stay outside the learning worker's input closure.
See [ADR-0005](docs/adr/ADR-0005-canonical-learning-loop.md) for the model lineage,
[learning status](docs/learning.md) and the [capability matrix](docs/parity/capability-matrix.md)
for recorded capabilities, and [MASTER_PLAN.md](MASTER_PLAN.md) for the delivery record.

Read those records as dated observations. Refresh the relevant Git, CI and runtime evidence before
acting; stale plans or historical PASS entries do not establish current capability. The latest
explicit user direction controls scope and budgets. A configured command or workflow, a merged
implementation, and a qualified live learning loop are separate milestones.

## Which checkout

**The canonical local checkout is `/home/ubuntu/JopeDime`**. Work on its claimed PR branch;
return it to verified remote `main` after the reviewed lifecycle.

Do not confuse this repository with `~/jain-split/jope-dime` (also reachable as `~/jope-dime`, a
symlink). That is a different repository on a different upstream — the local forge at
`git.neverhuman.org` — and it is the research monorepo, not this product repo.

## Claim before writing; hand off with evidence

Before the first write and after resuming a session, read the open PR queue, registered worktrees,
and the latest entries in [AGENT_CHAT.md](AGENT_CHAT.md). Hold `flock` on `AGENT_CHAT.md.lock`,
reread the claims, then append the checkout, branch, full HEAD, owner, paths and intended
verification before editing. An overlapping claim waits for the current owner's stopped-head
handoff; age or an idle session alone does not release it. An explicit later user assignment can
transfer ownership: preserve the predecessor's work and record the transfer and observed state
under the same lock before writing. Cleanup work also follows the
synchronized [MASTER_CLEAN_UP.md](MASTER_CLEAN_UP.md) record, whose second copy is in the research
repository; claim that repository separately before changing its copy. Recheck branch, HEAD,
dirty state and overlapping writers immediately before a checkout or index mutation. The append
lock serializes the record; it does not itself lock a checkout for the whole task.

Keep corrections on the claimed PR branch. At handoff, record the stopped HEAD, dirty files,
commands and outcomes, durable receipt locations, remaining blockers and ownership disposition.
A campaign claim links its pinned manifest and approved resource limits and deadline. A fresh
context resumes that state and its existing budget; it does not start a second lane or reset a
campaign deadline. JopeDime uses these existing records, not Bulletfarm's `bf board`/`bf claim`.

## Where work happens

**Never clone into `/tmp`.** `/tmp` is emptied at boot. On 2026-09-21 it held **442 GiB across 180
clones** of this repository, including commits that existed nowhere else and were one reboot from
being lost permanently. Use the reusable bare verification directory under custody for independent restoration.

**After its PR closes, preserve and restore-verify unique source and evidence, then remove the lane.**
**At most 4 concurrent lanes.**

**Worktrees live in exactly one place: `<repo>/.agent/worktrees/`** — owner ruling, 2026-09-21.
One shared directory for every agent (Claude, Codex, Grok) rather than a per-tool location each.
`.claude/worktrees` is a **symlink** to `.agent/worktrees`, so Claude Code's own worktree tool lands
in the shared directory without any configuration. **Do not replace that symlink with a real
directory**, and do not introduce `.codex/worktrees` or similar — that is the sprawl this replaces.

**Bootstrap before any worktree tool in a fresh checkout.** These ignored directories and
the symlink are not supplied by Git. From the repository root, create `.agent/worktrees`
and `.claude` with `rtk proxy mkdir -p .agent/worktrees .claude`. If there is no entry at
`.claude/worktrees` (a dangling symlink counts as existing), create it with
`rtk proxy ln -sT ../.agent/worktrees .claude/worktrees`. Before every worktree tool use,
verify it is a symlink and `rtk proxy readlink -f .claude/worktrees` resolves to this
repository's `.agent/worktrees`. A real directory or a different target requires
preservation and reconciliation before proceeding; never replace it with `ln -f`.
Its absence or wrong type is invisible to `git status` because `.claude/` is ignored.

- **Never `/tmp`** — not worktrees, not clones, not build output. It is emptied at boot.
- **At most 4 concurrent lanes per repository**, one build directory each, reused.
- **Remove a lane when its PR closes** with `git worktree remove`, never `rm -rf` (which orphans
  the admin entry).

This supersedes the previous blanket ban, which was phrased as a prohibition on `git worktree add`
and so was walked straight past by Claude Code's `EnterWorktree` **tool**. A rule stated as a
*location* is enforceable; one that names a single invocation is not.

## Build output

- **One build directory per lane, reused.** Never a fresh `--target-dir` per task. The alternative
  produced 119 separate Cargo build directories totalling 809 G.
- Verify or explicitly set `CARGO_INCREMENTAL=0` and
  `CARGO_PROFILE_DEV_DEBUG=line-tables-only` before Cargo; not every agent inherits them.
- **Keep `CARGO_TARGET_DIR` unset globally.** Issue #46 was fixed by PR #56 with normal and
  external-directory proofs: wrappers resolve executables from Cargo artifact messages and
  `jope-paths` preserves explicit root-selection precedence. CI now selects its persistent
  target per job through `.github/actions/cargo-target`; preserve one reused target per lane
  and the separation between PR and trusted-main storage domains.

## `target/` is partly an evidence store — never `rm -rf` it

`target/health/` holds generated score files that are **evidence**: regenerate them through their
owning proof lane, never hand-edit, and never blanket-delete the tree they live in.

The maintenance helper reports candidates only. Cargo markers do not certify contents;
source and evidence at every depth need preservation and independent verification before removal.

## Preserve before you delete — and then delete

Both halves are required. "Preserve everything" with no counterpart is how a disk reaches 91 %.

1. **Preserve first.** Bundle it, restore the bundle in a reusable independent `git clone --no-local --bare` under custody, and
   confirm **the specific commit is present** — a bundle can verify and still lack the commit you
   care about. This is not hypothetical: during the 2026-09-21 cleanup a detached worktree `HEAD`
   (`b2f72ec`) was absent from a `--all` bundle and would have been lost.
2. **Then delete.** A complete verified recovery route plus a fresh dependency check permits logged reclamation.

**Only trees you are going to delete need a bundle.** Bundling a tree you are keeping consumes the
disk you are trying to reclaim.

**Count unpushed commits with `git log --all --not --remotes`.** The branch-only form
(`git log --not --remotes`) examines only the checked-out branch and will report "zero unpushed
commits" for a repository that holds dozens on other local branches.

## Pull requests

Land work through the reviewed lifecycle: branch → PR → CI green at the exact head → review by a
**different identity** → merge → verify the remote state. `main-guard` enforces that every commit
on `main` belongs to a merged pull request.

**Review must be independent of the writer and bind the exact full head SHA.** Check the PR
author with `gh pr view <number> --json author,headRefOid` and every GitHub-associated commit
author with `gh api --paginate repos/veox-ai/JopeDime/pulls/<number>/commits --jq '.[].author.login'`.
The reviewing identity must differ from both the PR author and all those commit authors. Record
unassociated authors explicitly; local Git names and emails do not establish GitHub identity.
Changing a login does not make the writer an independent reviewer. Any changed head needs fresh
checks and review. Distinguish an independent review comment from a native GitHub approval, and
claim enforced protection only after reading it back from the API.
Use the established writer identity for new commits so the designated reviewer stays eligible;
preserve existing authorship. If that reviewer becomes ineligible, find another eligible independent
reviewer rather than rewriting attribution or self-approving.
If no eligible independent reviewer is available, keep the PR open with that blocker recorded.

Before starting new implementation work, check the open pull-request queue. If a PR is awaiting
review, checks, correction, or merge, resolve it before opening another lane. The current delivery
plan uses one implementation PR at a time; the four-lane ceiling is not a concurrency target.

Each focused PR states the behavior it changes, the acceptance command, and the evidence produced.
For behavioral fixes, preserve a reproducing failure and prove the corrected behavior; refusing
an operation and successfully executing its authorized path are different acceptance cases.
For documentation, `git diff --check` and link/import verification supplement the same exact-head
CI requirement; a declared path exclusion is recorded as `skipped-by-path`, not passed. Merge with
`gh pr merge <number> --rebase --match-head-commit <reviewed-full-sha>`, then verify the resulting
remote `main` and `main-guard`. GitHub's rebase merge does not authorize rewriting a shared branch.

### Approval procedure on this host

Verified on 2026-09-22: the established writer is **`jepsontaylor`**. The designated
review account is **`neverhuman`**, used by an independent Claude review process.
These are working defaults, not permanent eligibility: verify both identities for every PR.
The writer session never calls the approval API for its own change. Independence comes from
a separate reviewer process that did not write the change; selecting another account alone
does not provide it. An independent agent review is recorded as such, not as a human review.

1. Claim the checkout as above, preserve existing commit authorship, and use the writer for
   new commits and the PR. Check `gh api user --jq .login` before creating it. Current Git
   authorship is `Jepson Taylor <130782313+jepsontaylor@users.noreply.github.com>`; Git config
   and SSH key names alone do not prove the GitHub-associated author.
2. Freeze the final head. Run these read-only checks, replacing `PR_NUMBER` with the number:

   ```sh
   gh pr view PR_NUMBER --json author,headRefOid,headRefName
   gh api --paginate repos/veox-ai/JopeDime/pulls/PR_NUMBER/commits \
     --jq '.[] | {sha, login: (.author.login // "UNASSOCIATED"), git_author: .commit.author}'
   ```

   The reviewer must differ from the PR author and **every** associated commit author.
   Record unassociated authors and establish independence; do not silently omit them.
   If the reviewer opened the PR accidentally, preserve its discussion/evidence, close it
   as superseded and recreate it as the writer. Preserve the commits and their attribution.
3. Dispatch `gh workflow run ci.yml --ref BRANCH`, then identify and retain that run's
   **full source SHA, run ID and attempt**. Record the dispatch time, then use
   `gh run list --workflow ci.yml --branch BRANCH --event workflow_dispatch --json databaseId,headSha,createdAt,attempt`
   to select the new run created after that time at the frozen full SHA; inspect it with
   `gh run view RUN_ID --json headSha,attempt,status,conclusion,jobs`. Do not select an older
   success. All eight lanes plus `aggregate` must pass at
   that head for a full manual run. A PR run may have declared `skipped-by-path` lanes;
   inspect its aggregate and retain the synthetic checkout SHA separately. Cancelled,
   missing, stale or zero-test results cannot pass. Use **Re-run all jobs** for another
   attempt. Full-main CPU results are diagnostic until the later qualification phase.
4. Give a separate Claude reviewer the PR number, full head, diff, acceptance commands and
   evidence paths. Restrict it to reading/reviewing the candidate; it must not modify the
   source, deploy or merge. Supply the eligible review credential only to that child process
   (example below). It verifies its own `gh api user` identity and all authors before reviewing.
   Its prompt must prohibit credential-file reads, `gh auth` subcommands and using another
   identity; review credential separation on this shared Unix user is procedural, not a
   security boundary.
   It posts findings and, only if acceptable, submits a **native GitHub approval** with an
   explicit `commit_id: FULL_SHA` and `event: APPROVE` through the
   [review API](https://docs.github.com/en/rest/pulls/reviews#create-a-review-for-a-pull-request).
   A review comment, an approval of an ancestor, or a different login without independence
   is insufficient. Any changed head repeats CI and review. Keep the PR open if no eligible
   independent reviewer is available.
5. Read the native reviews back and verify an eligible, undismissed `APPROVED` review on
   exactly the current full SHA, with no unresolved blocking findings:

   ```sh
   gh api --paginate repos/veox-ai/JopeDime/pulls/PR_NUMBER/reviews \
     --jq '.[] | {id, reviewer: .user.login, state, commit_id, submitted_at}'
   ```

   Before executing the merge, recheck the head, author eligibility, current-attempt CI
   and review state. Preserve their API readbacks outside the checkout. Only then run:

   ```sh
   gh pr merge PR_NUMBER --rebase --match-head-commit FULL_SHA
   ```

   After merging,
   verify `merged_at`, fetch `origin/main`, verify the resulting main commit and its
   successful `main-guard`, and return the canonical checkout clean to that remote main.
   Preserve the lifecycle receipt before removing any closed worktree under the custody rules.

Protection readback returned HTTP 403 on 2026-09-22. This procedure is **procedural
enforcement**; `main-guard` detects unreviewed pushes after they happen and is not a preventive
branch lock. A paid-plan upgrade is not a prerequisite for the authorized reviewed merges.

### Credential and key locations

These are host-local paths, verified by file metadata on 2026-09-22. Keep secret bytes out of
Git, chat, review prompts, receipts and command output. Do not print these files or run token
retrieval as a standalone displayed command. Account access and eligibility must still be
checked at use time; `GH_TOKEN`/`GITHUB_TOKEN` can override the stored CLI account.

| Purpose | Location | How it is used |
| --- | --- | --- |
| GitHub CLI writer/reviewer authentication | `/home/ubuntu/.config/gh/hosts.yml` (0600) | Stored `jepsontaylor` and `neverhuman` accounts. Retrieve through `gh auth token --hostname github.com --user ACCOUNT` into a child environment; do not parse or copy the file. A configured `GH_CONFIG_DIR` changes the location. |
| GitHub SSH writer key | `/home/ubuntu/.ssh/id_ed25519_github` (0600) | Selected by `Host github.com` in `/home/ubuntu/.ssh/config`. Git transport only; it does not select the API approval identity. |
| GitHub SSH alternate key | `/home/ubuntu/.ssh/id_ed25519_neverhuman` (0600) | Selected by SSH aliases `github-neverhuman` and `github-mirror`. It does not authorize the writer to self-approve. |
| Claude CLI authentication | `/home/ubuntu/.claude/.credentials.json` (0600) | Authentication for the independent Claude process; separate from the GitHub credential it uses to submit review. |
| Runner registration PAT | `/etc/jope-runner/github-pat` (root:root, 0600 on runner hosts) | Host launcher only, for single-use JIT registration. Never mount into a job or use as the PR review credential. |

To select the review account without changing the shared writer's login, the review launcher
captures the token privately and passes it only to the independent process:

```python
env = os.environ.copy()
env.pop("GH_TOKEN", None)
env.pop("GITHUB_TOKEN", None)
token = subprocess.check_output(
    ["gh", "auth", "token", "--hostname", "github.com", "--user", "neverhuman"],
    env=env, text=True,
).strip()
env["GH_TOKEN"] = token
# Launch the separate reviewer with env=env; never print env or token.
# The reviewer first verifies: gh api user --jq .login
```

This is a launcher fragment (`os` and `subprocess` imports required), not a self-approval
command. Keep its prompt and redacted review receipt in custody outside the checkout.
Verify the parent still authenticates as the writer afterward. For runner host access,
use the [runner runbook](docs/runbooks/runner.md): this Ubuntu `xbabe2` is local (`sudo -n`),
whereas this user's SSH alias `xbabe2` points to `agent@192.168.68.89` using
`/home/ubuntu/.ssh/xbabe2_ed25519`. Do not confuse that alias with the local CPU runner host.

## What counts as verification

Bind evidence to the source commit, command and exit status, execution time, relevant environment,
input identities and resource budget. Record tests collected, executed, failed, skipped or excluded
and why. Missing runners, cancelled jobs, zero collected tests and unavailable required capabilities
are not passing coverage. Derive coverage totals from inventories and run receipts.

Status records distinguish source implementation, fixture/local proof, hosted CI, live operational
qualification and serving promotion. Preserve the remaining requirement for partial behavior.
Do not mark a whole capability complete from a passing subset, a checkpoint file, or an author's
summary. Training evidence includes actual optimizer updates and changed parameters; a target
improvement needs replay under the same HWM comparison identity (`task`, `dataset_hash`, `cut_id`,
evaluator, vocabulary and protocol). Resume preserves the
declared resource limits and deadline, and unknown outcomes remain unknown rather than zero.

## Change discipline

Use focused commits, preserve reproducibility, and do not rewrite shared branches. Update private feature branches onto current `origin/main` before handoff. For shared
PR branches, use a history-preserving update; never force-push or rewrite shared history. If a generated artifact is stale, delete or regenerate it
only through its owning proof lane.

## Large data

Bulk data does not live inside a git working tree — git does not protect untracked files, they can
never be committed at that size, and their presence makes every `git status` and every
untracked-file rescue walk them.

## Custody, maintenance and lane closure

One writer per checkout. Use the canonical checkout or a claimed, registered
worktree under `<repo>/.agent/worktrees/`; no sibling clones or tool-specific roots.
Preserve `.claude/worktrees` as the symlink to that shared root. At most four managed
lanes per repository, each with one reused build directory. No clones, worktrees
or build output under `/tmp`. Bulk data stays outside working trees.

When a lane's PR closes, preserve unique source and evidence first, verify recovery,
then remove the lane with `git worktree remove`. Reuse one independent **bare**
verification directory under custody outside working trees, populated with
`git clone --no-local --bare` or equivalent independent bundle restoration. Check
exact commits, trees and connectivity; no alternates or dependency on the original.
Remove verified generated restoration contents after each batch; keep immutable
archives and receipts. A valid checksum, a local commit and a merged change are
three different states.

The daily maintenance policy reports candidates older than 48 hours at **04:00 UTC**,
with `Persistent=true`; it never deletes. `cargo-build-dirs.sh` rejects `--delete`.
Cargo markers (`.rustc_info.json`, `debug/release/.fingerprint`) identify candidates,
not disposable contents. Neither `target` nor generic `CACHEDIR.TAG` identifies
Cargo output. Evidence may occur at any depth, including `debug`, `release`, and
`tmp`. Preserve and independently verify all relevant content and recheck active
consumers before an explicit logged removal. Unknown content is retained. See
`/home/ubuntu/JopeDime/deploy/maintenance/README.md` for installed reporting behavior.

Before running Cargo, verify or explicitly set `CARGO_INCREMENTAL=0` and
`CARGO_PROFILE_DEV_DEBUG=line-tables-only`. Claude's settings retain these values;
other agents may not inherit them. Keep `CARGO_TARGET_DIR` unset globally. Preserve
JopeDime CI's job-scoped target selection and separate PR/trusted-main storage domains.
