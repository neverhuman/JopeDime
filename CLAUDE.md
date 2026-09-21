@AGENTS.md

# CLAUDE.md — JopeDime

**This file exists because Claude Code loads `CLAUDE.md`, not `AGENTS.md`.** Until 2026-09-21 this
repository had neither on `main` — the only `AGENTS.md` lived on an unmerged branch — so a Claude
session cloning this repo had **no contract at all**. The `@AGENTS.md` import on line 1 is what
makes the contract load. **Do not remove it.**

`AGENTS.md` is the contract; this file only adds Claude-specific reminders and never contradicts it.

## This is the canonical GitHub checkout

**`/home/ubuntu/JopeDime`, on `main`.** There is one. Do not create a second checkout of this
repository, and do not confuse it with `~/jain-split/jope-dime` (also reachable as `~/jope-dime`,
a symlink) — that is a *different* repository on the local forge `git.neverhuman.org`.

## The rules that were missing when this repo filled a disk

See [`MASTER_CLEAN_UP.md`](MASTER_CLEAN_UP.md) for the full record.

1. **Never clone into `/tmp`.** It is emptied at boot. **442 GiB across 180 clones of this
   repository** lived there, holding commits that existed nowhere else.
2. **One build directory per lane, reused.** Never a fresh `--target-dir` per task. At most 4
   concurrent lanes. **Delete a lane and its build when its PR closes.**
3. **Do not create a worktree by any means** — not `git worktree add`, not the `EnterWorktree`
   tool. Phrase the rule as an outcome: there is one checkout.
4. **`target/health/` is evidence.** Never `rm -rf target`. Reclaim with a tool that identifies
   build dirs by `.rustc_info.json` / `debug|release/.fingerprint` and copies non-Cargo files out
   first.
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
