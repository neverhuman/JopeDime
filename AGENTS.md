# JopeDime agent contract

The operating contract for contributors and automated agents working in this repository.

Until 2026-09-21 `main` carried **no root `AGENTS.md`** — one existed only on an unmerged branch,
so an agent working from a fresh clone had no contract at all. That gap is a direct cause of the
disk incident recorded in [`MASTER_CLEAN_UP.md`](MASTER_CLEAN_UP.md).

---

## Which checkout

**The canonical local checkout is `/home/ubuntu/JopeDime`**, on `main`. There is one.

Do not confuse this repository with `~/jain-split/jope-dime` (also reachable as `~/jope-dime`, a
symlink). That is a different repository on a different upstream — the local forge at
`git.neverhuman.org` — and it is the research monorepo, not this product repo.

## Where work happens

**Never clone into `/tmp`.** `/tmp` is emptied at boot. On 2026-09-21 it held **442 GiB across 180
clones** of this repository, including commits that existed nowhere else and were one reboot from
being lost permanently. If you need a throwaway checkout, put it under `$HOME`.

**Delete a lane and its build directory when its pull request closes.** Not later, not never.
**At most 4 concurrent lanes.**

**Do not create a worktree — by any means.** This covers the `git worktree add` command, *and any
tool or harness feature that creates one*, including Claude Code's `EnterWorktree` tool and
`.claude/worktrees/`. A ban phrased as a prohibition on a shell command was walked straight past by
a tool. The rule is about the outcome: **there is one checkout.**

## Build output

- **One build directory per lane, reused.** Never a fresh `--target-dir` per task. The alternative
  produced 119 separate Cargo build directories totalling 809 G.
- `CARGO_INCREMENTAL=0` and `CARGO_PROFILE_DEV_DEBUG=line-tables-only` are set in the agent
  environment. Backtraces still work; do not unset them to "speed up" a build.
- **`CARGO_TARGET_DIR` must not be set globally** until #46 is fixed. `deploy/ci/emitters.sh`
  hard-codes `target/debug/examples/emit` and `crates/jope-paths` walks the running executable's
  ancestors for the `.jope-home` marker, so redirecting Cargo's output breaks both. `.github/
  workflows/ci.yml` pins `CARGO_TARGET_DIR` into the workspace for exactly this reason; that
  override is load-bearing, not tidy-up-able.

## `target/` is partly an evidence store — never `rm -rf` it

`target/health/` holds generated score files that are **evidence**: regenerate them through their
owning proof lane, never hand-edit, and never blanket-delete the tree they live in.

To reclaim build space use a tool that distinguishes build output from evidence — one that
identifies build directories by `.rustc_info.json` / `debug|release/.fingerprint` rather than by
name, and copies non-Cargo files out before deleting.

## Preserve before you delete — and then delete

Both halves are required. "Preserve everything" with no counterpart is how a disk reaches 91 %.

1. **Preserve first.** Bundle it, restore the bundle in a throwaway `git clone --no-local`, and
   confirm **the specific commit is present** — a bundle can verify and still lack the commit you
   care about. This is not hypothetical: during the 2026-09-21 cleanup a detached worktree `HEAD`
   (`b2f72ec`) was absent from a `--all` bundle and would have been lost.
2. **Then delete.** A rescued, verified copy is authority to reclaim the original.

**Only trees you are going to delete need a bundle.** Bundling a tree you are keeping consumes the
disk you are trying to reclaim.

**Count unpushed commits with `git log --all --not --remotes`.** The branch-only form
(`git log --not --remotes`) examines only the checked-out branch and will report "zero unpushed
commits" for a repository that holds dozens on other local branches.

## Pull requests

Land work through the reviewed lifecycle: branch → PR → CI green at the exact head → review by a
**different identity** → merge → verify the remote state. `main-guard` enforces that every commit
on `main` belongs to a merged pull request.

**The reviewing identity must not be the identity that authored the commits.** Check with
`gh api repos/<owner>/<repo>/commits --jq '.[].author.login'`; `git log` shows the committer name,
not the GitHub account, so a mismatch is invisible locally.

Before starting new implementation work, check the open pull-request queue. If a PR is awaiting
review, checks, correction, or merge, resolve it before opening another lane.

## Change discipline

Use focused commits, preserve reproducibility, and do not rewrite shared branches. Rebase a feature
branch onto `origin/main` before handoff. If a generated artifact is stale, delete or regenerate it
only through its owning proof lane.

## Large data

Bulk data does not live inside a git working tree — git does not protect untracked files, they can
never be committed at that size, and their presence makes every `git status` and every
untracked-file rescue walk them.
