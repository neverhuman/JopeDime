# MASTER_CLEAN_UP — collapsing the JopeDime sprawl

**Started** 2026-09-21 ~18:00 UTC · **Operator-approved plan** · **Status: IN PROGRESS**

This file is written *live as the work proceeds*, not afterwards. Every deletion is recorded with
what was preserved first and what it reclaimed. Mistakes are recorded too.

> **Where does this file live?** Both at `~/jope-dime/MASTER_CLEAN_UP.md` (which is a **symlink** to
> `~/jain-split/jope-dime/MASTER_CLEAN_UP.md` — the same file) and, committed, in
> `github.com/veox-ai/JopeDime`.

---

## The question that had to be answered first

**"Is `~/jain-split/jope-dime/` where everything should be?"** — No, not for GitHub work.

| Path | Upstream | Role |
| --- | --- | --- |
| `~/jope-dime` | *symlink* → `~/jain-split/jope-dime` | same tree, convenience path |
| `~/jain-split/jope-dime` | **`git.neverhuman.org`** (local forge, tunnelled on `127.0.0.1:8787`) | the research monorepo. 246 G |
| **`~/JopeDime`** | **`github.com/veox-ai/JopeDime`** | **the product repo — THE canonical GitHub tree.** 104 G, of which **102 G is `target/`** |

Two different repositories. Neither had a clean canonical working tree. GitHub `main` is
`6a85fa3985` and **no local checkout had that object**.

> ⚠️ Four git worktrees are registered through the **symlink path** `~/jope-dime/.claude/worktrees/*`.
> Replacing or moving that symlink breaks all four. Do not touch it.

## Why this happened — there was no rule, and two rules made it worse

The entire global agent-instruction budget is ~1.5 KB and 100 % of it concerns the `rtk` CLI wrapper.

1. **`~/.cargo/config.toml`** sets no `target-dir` **and** sets `auto-clean-frequency = "never"`.
   One private `target/` per clone, and Cargo's own garbage collection disabled. 119 build
   directories is precisely what that configuration produces.
2. **`MASTER_PLAN.md` line 3** tells agents the integrator clone lives "under the Claude session
   scratchpad" — i.e. `/tmp`, **which is emptied at boot**.
3. **`jope-dime/AGENTS.md`** says *"Preserve dirty and untracked work"* and mentions **nothing**
   about clone location, build output, disk, or cleanup. It does **not** import the parent
   `jain-split/AGENTS.md`, so that file's rules never reach an agent working here — the inheritance
   is prose, not a mechanism.
4. The parent rule that does exist — *"Never create a git worktree, anywhere, for any reason"* —
   bans a **shell command**. Claude Code's `EnterWorktree` **tool** walked straight past it:
   **8 worktrees exist in a repo whose contract permits none.**

**Nobody violated a rule. The rules did not cover the failure.**

## Measured scale

| | |
| --- | --- |
| Disk at start | **91 % — 697 G free of 7.3 T**, growing ~345 G/day |
| `/tmp/claude-1000` | **442 GiB** across **180 git repos**, 96 % of it build cache |
| Cargo build dirs under `$HOME` | **340** |
| The `jope-*` sprawl | **1,324 G**, of which **191 G is `target/`** |
| **Total irreplaceable content** | **491 MB** |

**1.3 TB of trees contained 491 MB of actual work.**

---

## A measurement error I made, and the correction

I first reported **"zero unpushed commits anywhere."** That was **wrong**. I used
`git log --not --remotes`, which examines only the *checked-out branch*; the unique commits live on
**other local branches**. The correct probe is `git log --all --not --remotes`, cross-checked
against `gh api` and `git ls-remote`. Corrected:

| Location | Unpushed | Note |
| --- | --- | --- |
| `~/jain-split/jope-dime` | **39** | 4 local-only branches + 2 stashes |
| `~/JopeDime` | **15** | + 1 stash + 41 non-build untracked |
| `~/jope-slime` | **85** | remote on a third host (xbabe3), unverified |
| `~/jope-prime` | **8** | **no remote configured at all** |
| `~/jope-health-validation-20260920` | **6** | in no other working tree |

**Every deletion below is gated on the `--all` probe, never the branch-only one.**

---

## PHASE 0 — PRESERVATION ✅ COMPLETE

**Custody root: `~/jope-rescue-20260921/` — 491 MB, 65 files.**
Manifest `SHA256SUMS` sha256 = `9449b989e6d28a18b21261fc2b1a75d747d9a93c334baa7e6229b83fb70d748c`
— all 65 files verify (`sha256sum -c SHA256SUMS`).

Built with the operator's existing `~/lane-tools/` scripts. **No new rescue tooling was written.**

### The most fragile object on the box — preserved first

`~/jope-recovery/20260921/restored-checkpoint.git` is a 12 M bare repo with **zero refs**
(`refs/` empty, `packed-refs` header-only). Seven unique commits are reachable **only through a
raw-sha `HEAD`**. A `git gc`/`prune` anywhere near it destroys them permanently.

Captured as a **whole-directory tar** rather than a bundle (`git bundle --all` on a ref-less repo
produces nothing). Verified by extracting and confirming all seven objects:
`9e0fd51be2`, `46c6736`, `e0cebe1`, `0867f2e`, `1e3c06f`, `4cb3df6`, `b4777dd` → all `commit`.

### Unique commits that existed only in `/tmp` — all preserved and verified

Every bundle was cloned into a throwaway `git clone --no-local --bare` and the target SHA confirmed
present. This is the check that matters: a bundle can verify and still lack the commit.

| Rescue | Unique SHA | Verified | Commits in bundle |
| --- | --- | --- | --- |
| `featmath-clippy` | `7b3063e` (8 commits, + `ARTIFACTS/`, `LOGS/`) | ✅ commit | 219 |
| `land-cli-split` | `00329bc` (34; the 55→2 binary CLI split) | ✅ commit | 217 |
| `r1-split-ops` | `b7b25f7` | ✅ commit | 433 |
| `r2-deploy` | `bc80699` | ✅ commit | 342 |
| `r2b-deploy` | unresolved **`UU` merge state** | ✅ commit | 355 |
| `wp-p1` | `ffd1175` | ✅ commit | 212 |
| `vfy-e2e-rebase` | `2ffb330` | ✅ commit | 216 |
| `wp1-redirect` | `8b4cf46` + **+1810/−248 uncommitted** | ✅ commit | 239 |
| `wp3` | `fb50490` (20) | ✅ commit | 144 |
| `wp6` | `b943bc2` (22) | ✅ commit | 146 |
| `wp7` | 1 commit | ✅ | 153 |
| `wp8`, `wp8b`, `wp9`, `wp10`, `rev-p2c`, `new`, `old`, `lfscfg-w` | 1–5 each | ✅ | — |

**`r2b-deploy`'s merge conflict needed special handling.** `git diff HEAD` does not represent `UU`
files faithfully, so it was captured three ways: `staged.patch`, `unmerged-index.txt`
(all three stages of each conflict), and `conflicted-worktree.tgz` (the files verbatim, conflict
markers intact). The state: 3 × `UU`, 1 × staged `M`, 1 × added `A`.

### Two audit claims I checked and found wrong

1. **The audit said `port/work.git` was "the only object store for `wp3` and `wp6`."** It is not.
   That bare hub holds 13 refs — `wp5`, `wp7`, `wp8`, `wp11`, `wp12` and 7 archives — and
   **no `wp3` or `wp6` ref at all**. The commits existed *only in the checkouts*. Had I trusted the
   audit and deleted the checkouts, `fb50490` and `b943bc2` would have been lost. Both are now
   rescued directly and verified.
2. **The audit flagged 6 HAS-WORK repos in the dead session; the operator's `lane-inventory.sh`
   found 39.** `wp8`, `wp8b`, `wp9`, `wp10`, `timing`, `new`, `old`, `rev-p2c` and `lfscfg/w` were
   all missed. All are now rescued. **The tool beat the agent**; running it was not optional.

### Home repositories preserved

`JopeDime-canonical` (bundle + 8.5 MB uncommitted patch + untracked), `jope-health-validation`,
`jope-prime-NOREMOTE` (the 8 commits that can never be pushed), `jope-recovery-candidate`,
`JopeDime-pr-review`, and the two **zero-object** repos whose source was never committed at all:
`jope-lime-ZEROOBJECTS.untracked.tgz` (88 MB) and `JopeDime-codex-stage-ZEROOBJECTS.untracked.tgz`.

`~/jain-split/jope-dime` gets **only** its 51 KB uncommitted patch — see the mistake below.

### Evidence that cannot be rebuilt — 137 MB copied off `/tmp`

PR-review inputs/verdicts/logs, `dryrun-quant{,2}`, `dryrun-mirror{,2}`, `port/evidence`,
`m-bin`/`m-reg` diagnostics, `wp5logs`, `hosted/` sampling runs, `nextest-timing.log`,
`wp4-final-receipts.log`, `trial-catalog.json`, and all three sessions' agent transcripts.

---

## Mistakes I made during this cleanup

Recorded because the whole point of this document is that the next agent inherits the truth.

1. **I started a rescue that would have made the emergency worse.** `lane-rescue.sh` tars
   *untracked* files, and `~/jain-split/jope-dime` has **117 G of untracked data**. The job reached
   a **4.5 GB** `untracked.tgz` and a **10.4 GB** bundle before I caught it, while the disk was
   91 % full. Stopped and both artifacts deleted.
   **The rule I should have applied from the start: only trees that will be DELETED need a bundle.**
   `jain-split/jope-dime` is being kept, so its `.git` and its untracked bulk needed no copy at all.
2. **I deleted a cargo cache before the inventory that covered it had finished.** I removed the dead
   session's `CARGO_HOME` (`ch/`, 957 MB) in a "zero-risk" batch; `lane-inventory.sh` then flagged
   `ch/git/checkouts/jankurai-4b9241362eced4c2/9695b55` as having 64 unsaved commits.
   **Checked rather than assumed:** no `jope-dime` or `JopeDime` crate depends on jankurai; the
   forge is live; the six `jankurai-tools-*` mirrors survive in the shared `~/.cargo/git/db/`.
   It was a derived cache, not authored work — **but the sequencing was wrong.** Inventory first,
   delete second, always.

---

## PHASE 1 — RECLAMATION ✅

The two deadlocked `cargo test -p jope-datasets` processes (PIDs `3730887`, `3856483` and parents
`3730238`, `3856327`) that had pinned session `c2314a57` since **04:19, 13.5 hours at 0 % CPU**
were terminated after `wp7` was rescued. All four confirmed gone.

**`target/` was never removed with `rm -rf`.** `jain-split/AGENTS.md` designates `target/` an
*evidence store* (`target/orchestrator-manual/`, `target/reviews/`, `target/coordination-receipts/`,
`target/ci.lock`) and `JopeDime/AGENTS.md` does the same for `target/health/`. Every sweep used
`~/lane-tools/cargo-build-dirs.sh`, which identifies build dirs by `.rustc_info.json` /
`debug|release/.fingerprint` (**not** by name) and copies non-Cargo files out **before** deleting.
It rescued **320 MB** this way, including `target/health`, `target/jankurai`, `target/nextest`,
`target/tools`, eight `e2e-fault-*`/`e2e-gpu`/`e2e-strace` evidence dirs, `emitters` and `nextest`.

| Reclaimed | |
| --- | --- |
| `~/JopeDime/target` | **100 G** |
| my session's 53 build dirs | **265 G** |
| 12 other home trees (health-validation 19 G, prc 12 G, rust-followup 8.1 G, scream 8.7 G, port-final 7.5 G, codex-stage 3.9 G, …) | **151 G** |
| dead session `c2314a57` (after full rescue) | ~23 G |
| venvs, LibTorch SDK, cargo caches, harness caches | ~14 G |
| orphaned worktree `scq-admission-recovery` | 427 M |
| duplicate clones (prc/prb/pra/pr-review/lane-rescue/leases) | ~290 M |

## PHASE 2 — WORKTREE COLLAPSE ✅ — 8 + 6 removed, **every repo now has exactly one**

**A correction to my own plan.** It said worktrees with unique commits must be "bundled and pushed
before removal." That is **unnecessary**: a worktree shares the parent's object store, so removing
it cannot lose a commit. Verified before acting — `worktree-agent-a15e…`, `worktree-agent-ae41…`,
`queue-hotfix` `1754756a36` and `singularity-dime` `8d971d0d21` all resolve to `commit` in the
parent. **Only the uncommitted files were at risk**, and those were captured as patches + tarballs
(`~/jope-rescue-20260921/worktrees/`, 11 files — `clean-main` 127 KB, `queue-hotfix` 128 KB,
`flywheel/jope-dime` 122 KB, `aspfm-multiclass-dl-v3` 163 MB untracked).

Removed with `git worktree remove --force` — never `rm -rf`, which orphans admin entries.

| Repo | Worktrees before | After |
| --- | --- | --- |
| `~/jain-split/jope-dime` | 9 (8 + main) | **1** |
| `~/JopeDime` | 2 | **1** |
| `~/jope-prime` | 3 | **1** |
| `~/JopeDime-pr-review` | 4 | deleted entirely |

**One commit was nearly lost here.** `b2f72ec` was a *detached worktree HEAD*, so `git bundle --all`
never saw it, and removing that worktree left it unreferenced and garbage-collectable. Caught by
checking the bundle actually contained each unpushed SHA; preserved via an explicit tag →
`JopeDime-pr5-synthetic-merge.bundle`, verified.

## PHASE 3 — CANONICAL TREE ✅

`~/JopeDime` was on a stale branch with **856 dirty files**. Rather than discard them, they were
**committed** to `codex/jankurai-health-20260919` as `dc06289`, so they remain reachable *in git*,
not only in a bundle. The stash survived. Then fast-forwarded to `origin/main`.

| | Before | After |
| --- | --- | --- |
| Branch | `codex/jankurai-health-20260919` | **`main` @ `6a85fa3`** |
| Dirty | 856 | **0** |
| Size | 104 G | **2.1 G** |

## PHASE 4 — BULK MOVED OUT OF THE REPO ✅

`parallel/recovery/xbabe2-migration-20260914` — **117 G, 0 tracked files, status `??`** — moved to
`~/jope-evidence/xbabe2-migration-20260914/`. Same filesystem, so a rename: instant, no copy.
`~/jope-evidence/INDEX.md` records provenance and a per-file manifest with its sha256.

`~/jain-split/jope-dime`: **246 G → 129 G**. Nothing deleted.

## PHASE 5 — PREVENTION ✅

1. **`~/.claude/settings.json`** gained the `env` block it never had:
   `CARGO_INCREMENTAL=0`, `CARGO_PROFILE_DEV_DEBUG=line-tables-only`. The existing
   `PreToolUse`/`Bash` → `rtk hook claude` hook was preserved (asserted before writing).
   **`CARGO_TARGET_DIR` was deliberately NOT set** — `deploy/ci/emitters.sh:29` hard-codes
   `target/debug/examples/emit` and `crates/jope-paths` walks the executable's ancestors, so a
   global redirect breaks both. That is issue #46; until it lands, discipline substitutes.
2. **`~/.cargo/config.toml` left alone.** Its `auto-clean-frequency = "never"` is tempting to flip,
   but the comment records a real reason — jankurai binaries read schemas from cargo git checkouts
   at runtime and auto-clean deleted them mid-CI. The scheduled sweep covers this instead.
3. **`AGENTS.md` rewritten** (old version kept at `~/jope-rescue-20260921/AGENTS.md.before`). It now
   `@import`s `jain-split/AGENTS.md` so the parent rules are **mechanically** inherited rather than
   asserted; states where to clone and that `/tmp` is never it; caps lanes at 4 and requires reuse;
   requires deletion when a PR closes; restates the worktree ban to cover **the tool as well as the
   command**; and pairs "preserve" with "then delete", including that only trees being deleted need
   a bundle and that unpushed commits must be counted with `--all`.
4. **`MASTER_PLAN.md` line 3 corrected** — it had named the Claude session scratchpad (`/tmp`) as
   the integrator clone location. That single sentence is the written instruction behind 442 GiB of
   doomed clones.
5. **`.gitignore`** now covers `.claude/` (previously untracked *and* un-ignored) and the moved bulk
   path.
6. **A sweep now exists.** `jope-build-sweep.timer` (user unit, lingering already enabled) runs
   `cargo-build-dirs.sh --stale 24 --delete` daily at 04:00 over `/tmp/claude-1000`, `~/JopeDime`
   and `~/jain-split/jope-dime`. Before today, `crontab -l` had no cleanup entry and no timer swept
   anything.

## PHASE 2b — `/tmp` SESSION CLONES COLLAPSED ✅

`lane-inventory.sh` reported **93 of 107** repos in my own session as `HAS-WORK`. That tool fails
safe: it cannot verify a commit through a local-path remote, so it counts every commit as unsaved.
The audit, conversely, claimed all 117 were fully present on GitHub.

Rather than trust either, I checked every unsaved SHA against the canonical clone's fetched origin
refs. **195 SHAs across ~25 clones could not be shown present.** Many of those branches were
squash-merged and deleted, so absence proves little — but proving it would have taken 195 API calls,
and **bundling was cheaper than the argument**: 44 artifacts, 713 MB. Then the clones went.

- 94 scratchpad directories removed.
- 442 loose inventory listings removed (already inside `scratchpad-loose-files.tgz`).
- My session's `/tmp` footprint: **293 GiB → 284 KB**.
- `/tmp/claude-1000/-home-ubuntu-jain-split-jope-dime`: **5.3 MB total**, 0 git repos.

## PHASE 1b — BUILD OUTPUT STRIPPED FROM THE EVIDENCE STORES ✅

The `~/jope-*` data stores are kept (decision C2), but Cargo output *inside* them is still build
output. Swept with the same evidence-preserving script: **+71 G**, with 5 non-Cargo directories
rescued out of `jope-migration-evidence` first.

`jope-migration-evidence` 33 G · `jope-recovery` 15 G · `jope-dime-recovery` 18 G ·
`jope-prime` 1.7 G · `jope-dime-mirror` 1.3 G · `jope-scratch` 486 M ·
`jope-dime-cache-evidence` 398 M · `flywheel` 309 M · `jope-slime` 230 M.

**Not one receipt was touched** — only directories carrying `.rustc_info.json` or
`debug|release/.fingerprint`.

## Result

| | Before | After |
| --- | --- | --- |
| **Free disk** | **697 G (91 %)** | **1.3 T (84 %)** — **+603 G** |
| `/tmp/claude-1000`, my session | 293 GiB | **284 KB** |
| `/tmp/claude-1000`, total | 442 GiB | 163 G (all of it the live `bullet` session) |
| `~/JopeDime` | 104 G, stale branch, 856 dirty | **2.1 G, `main` @ `6a85fa3`, clean** |
| `~/jain-split/jope-dime` | 246 G | 129 G |
| Worktrees across all repos | 14 | **3 — exactly one each** |
| Git repos in `/tmp` (jope-dime) | 180 | **0** |
| Custody preserved | — | **1.9 G, 1,424 files, `sha256sum -c` passes** |

Custody manifest sha256: `f903b6a00d2ee685…`

### Repo health after the work

`~/jain-split/jope-dime`: `git fsck --connectivity-only` reports **0 errors**; **2 stashes** and all
**39 unpushed commits** intact; 1 worktree.
`~/JopeDime`: on `main`, clean, with the rescued branch still reachable at `dc06289`.

## ⚠️ SCOPE CORRECTION — this cleanup did not end the disk emergency

The brief framed `/tmp/claude-1000` as the main cause. **It was ~6.6 % of used space.**

| Consumer | Size | Share of the 6.5 TiB used |
| --- | --- | --- |
| **`/home/ubuntu`** | **5,611 GiB** | **~84 %** |
| `/tmp` total | ~544 GiB | ~8 % |
| └ `/tmp/claude-1000` | 442 GiB | ~6.6 % |
| `/var` | 51 GiB | < 1 % |

A per-entry inventory of all 472 entries in `/home/ubuntu`
(`~/jope-rescue-20260921/logs/home-inventory.tsv`) shows the real ranking. **The three largest
trees on this host appear in no brief, no plan and no agent chat:**

| | GiB | In scope this round? |
| --- | --- | --- |
| **`remote_super`** | **1,597** | ❌ never mentioned anywhere |
| **`jain-split`** | **804** | partially — `jope-dime` (129 G) is inside it |
| **`jbaby`** | **423** | ❌ never mentioned |
| `jope-preservation` | 233 | kept per C2 |
| **`jmcp-split`** | **171** | ❌ never mentioned |
| **`gate-runner`** | **161** | ❌ never mentioned |
| `jope-flywheel` | 151 | kept per C2 |
| `preserved-archives` | 132 | kept per C2 |
| `jope-evidence` | 117 | created by this cleanup (the moved bulk) |
| `jankurai-split` | 85 | ❌ never mentioned |

**`remote_super` alone is more than twice everything this cleanup reclaimed.** Note also that
`remote_super/smart_cluster` — the parent of the orphaned worktree removed in Phase 2 — does
**not** exist, so that tree already contains at least one dead subtree.

### A measurement hazard that affected my own early numbers

The `rtk` `PreToolUse` hook rewrites `du`, `find`, `ls`, `grep` and `sort` and returns
**summarised or truncated** output. A globbed `du -sk /home/ubuntu/*` returned only entries
beginning `a`/`b` — the ~300 directories past `b` were silently missing, which is why
`remote_super`, `jbaby`, `jmcp-split` and `gate-runner` went unnoticed for so long. The inventory
above was re-derived with `/usr/bin/du` per entry.

**Treat any globbed `du` on this host as possibly partial.** Use `/usr/bin/…` absolute paths.

### Prior art nobody cited

`/home/ubuntu` already contains `disk-pressure-cleanup.sh` (dry-run by default; cleans
`*/target` trees oldest-first, `/var/tmp/buildah*`, journal vacuum, syslog truncate),
`big_files_fast.sh`, `fleet-disk-gc/`, and four dated `storage-{audit,cleanup,offload,reclamation}`
directories going back to 2026-07-24. **This host has had this exact problem at least four times.**
Read `disk-pressure-cleanup.sh` before writing anything new — it covers `/var` and `/var/tmp`,
which this cleanup did not touch.

### Still open

- **The `bullet` session's 163 G** — a live agent was writing in it. Needs the operator's consent.
- **`~/jope-node-mirror` → 193 G** (root-owned symlink) — the largest single reclaim candidate left
  on the box, and it has never had an owner decision.
- `~/jope` (64 G) has **no `.git`** — nothing in it is recoverable from any remote. Left alone.
- `jope-preservation` 234 G, `jope-flywheel` 152 G, `preserved-archives` 133 G — kept per C2.

---

## Explicitly NOT touched

- **The live `bullet` session** `/tmp/claude-1000/-home-ubuntu-bullet/ff4e2f6f…` — **132 GiB, and it
  was writing files two minutes before the scan.** Another agent is running in it. Its 126 GiB of
  build output needs the operator's consent; the default is hands off.
- `jope-preservation` (234 G), `jope-flywheel` (152 G), `preserved-archives` (133 G),
  `jope-migration-evidence` (66 G), `jope-dime-mirror`, `jope-golden42`.
- **`~/jope`** (64 G) — has a `.gitignore` but **no `.git`**. Nothing in it is recoverable from any
  remote. Treated as irreplaceable data.
- The two root-owned symlinks: `jope-node-mirror` → **193 G** and `jope-test-adaptive` → 4.1 G.
  The 193 G mirror is the largest single reclaim candidate left on the box and needs an owner
  decision it has never had.

---

## In-flight work parked (not cleanup, recorded so it is not lost)

- **#45 merged** → `main c572f177`; **#34 merged** → `main 6a85fa39`; `main-guard` green.
- **#44 open**, head `90df0ceb`, **APPROVED**, now `mergeStateStatus=DIRTY` — needs a rebase onto
  `main` after #34 landed.
- Issues filed: **#46** hard-coded build layout, **#47** gitleaks scan scope, **#48** the emit
  binding covering only `model.py`/`model.rs`.
