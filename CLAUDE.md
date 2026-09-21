@AGENTS.md

# CLAUDE.md — JopeDime

**This file exists because Claude Code loads `CLAUDE.md`, not `AGENTS.md`.** Until 2026-09-21 this
repository had neither on `main` — the only `AGENTS.md` lived on an unmerged branch — so a Claude
session cloning this repo had **no contract at all**. The `@AGENTS.md` import on line 1 is what
makes the contract load. **Do not remove it.**

`AGENTS.md` is the contract; this file only adds Claude-specific reminders and never contradicts it.

## This is the canonical GitHub checkout

**`/home/ubuntu/JopeDime`** is the canonical checkout, on its claimed PR branch during work.
Additional lanes use registered worktrees under the shared root below, and do not confuse it with `~/jain-split/jope-dime` (also reachable as `~/jope-dime`,
a symlink) — that is a *different* repository on the local forge `git.neverhuman.org`.

## The rules that were missing when this repo filled a disk

See [`MASTER_CLEAN_UP.md`](MASTER_CLEAN_UP.md) for the full record.

1. **Never clone into `/tmp`.** It is emptied at boot. **442 GiB across 180 clones of this
   repository** lived there, holding commits that existed nowhere else.
2. **One build directory per lane, reused.** Never a fresh `--target-dir` per task. At most 4
   concurrent lanes. **Preserve and verify unique source/evidence before removing a closed lane.**
3. **Worktrees go in `<repo>/.agent/worktrees/` — nowhere else.** Owner ruling 2026-09-21: one
   shared directory for Claude, Codex and Grok, instead of a per-tool location each.
   `.claude/worktrees` is a **symlink** to it so the `EnterWorktree` tool lands there without
   configuration. Max 4 lanes; remove with `git worktree remove` when the PR closes. **Never
   `/tmp`.**
4. **`target/health/` is evidence.** Never `rm -rf target`. The helper reports Cargo candidates only;
   preserve and verify evidence at every depth before an explicit logged removal.
5. **`CARGO_TARGET_DIR` must not be set globally** until #46 lands — `deploy/ci/emitters.sh:29`
   hard-codes `target/debug/examples/emit` and `crates/jope-paths` walks the executable's
   ancestors, so redirecting Cargo's output breaks both. The `CARGO_TARGET_DIR` override in
   `.github/workflows/ci.yml` is load-bearing for the same reason.

## Two measurement traps

- **`git log --not --remotes` lies** — it examines only the checked-out branch. Use
  **`git log --all --not --remotes`**. The branch-only form reported "zero unpushed commits" for
  trees holding 39, 15 and 85.
- **A bundle can verify and still lack the commit you care about.** A detached worktree `HEAD`
  (`b2f72ec`) was absent from its `--all` bundle and would have been lost. Always confirm the
  specific SHA restores.

## Reviews

The reviewing identity must not be the identity that authored the commits. Check with
`gh api repos/veox-ai/JopeDime/commits --jq '.[].author.login'` — `git log` shows the committer
name, not the GitHub account, so a mismatch is invisible locally. Posting comments without
`GH_TOKEN` set falls through to the OAuth user, which is the *reviewing* account.

## Coordination

Several agents share this host. **`MASTER_CLEAN_UP.md` is the live record for cleanup work** —
read its tail before acting and append as you go. `AGENT_CHAT.md` is the general peer wire.
`MASTER_PLAN.md` is the board. None of them is release authority; git and the receipts are.

Verify or set `CARGO_INCREMENTAL=0` and `CARGO_PROFILE_DEV_DEBUG=line-tables-only`
before Cargo; not every agent inherits Claude settings. One writer per checkout.
The daily timer reports at 04:00 UTC with a 48-hour filter; it never deletes.
A checksum-valid archive, a local commit and a merged change are different states.
