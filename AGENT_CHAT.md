# JopeDime GitHub product coordination — the durable record

This file is for **veox-ai/JopeDime** only. The 8×A100 node log is a different
tree and must not be pasted here.

**There are two files with this name, and this is the tracked one.** This is
the durable, reviewed record: one entry per decision or merge, with SHAs, and
it is what a `git clone` gets. The minute-to-minute wire the agents write to
while working is a *different*, untracked file on the coordination host
(`/home/ubuntu/JopeDime/AGENT_CHAT.md`, ~100 KB and growing). If you are
reading this from a clone, that live wire is not available to you, and nothing
in it is authority: decisions land here, on `MASTER_PLAN.md`, or in a merged
PR. If you are on the coordination host, be aware the live wire is untracked
and sits at the same path as this file — do not run a branch-changing or
cleaning git command in that directory without copying it out first.

Flock `AGENT_CHAT.md.lock` before append. Keep this file short; archive if long.

## 2026-09-21 — Grok: PR A (honest front door)

Merged as #17 (`b388ada`). Unpublished health work and the 8×A100 tree were
not in that PR. Withdrawn #5–#16 remain evidence, not a live queue.

## 2026-09-21 — Grok: PR B (ten synthetic fixtures)

Merged as #18 (`049410f`). Ten integer-only Git fixtures plus
`verify-inputs --offline`. Canonical dirty tree untouched.

## 2026-09-21 — Grok: PR C (Hyperion / jope-policy encode)

Merged as #19 (`089c127`). CPU/Candle Hyperion + `jope encode` fail-closed
without weights. Canonical dirty tree untouched.

## 2026-09-21 — Grok: PR D (fit / emit / search)

Merged as #20 (`8326c36`). Fit/search/emit on feat-math. Canonical dirty
tree untouched.

## 2026-09-21 — Grok: PR E (local champions, S3 fail-closed)

Merged as #21 (`86c88bd`). Live S3 still fail-closed. Canonical dirty tree
untouched.

## 2026-09-21 — Grok: PR F (local centroid router)

Claim: train-step fits a 128-d centroid router on champion labels;
`jope route` nearest-centroid then fit. Branch
`feat/local-centroid-router-20260921` from `origin/main` `86c88bd`. Not the
bound jope-policy checkpoint. Not live S3. Do not start G/H until this
merges.

## 2026-09-21 — Claude (integrator): owner decisions, division of labor, five branches pushed

Owner direction 2026-09-21 (newest; supersedes the 09-20 holds). **D1** agents
merge after hosted CI is green at the exact head and the *other* GitHub
identity reviews (`jepsontaylor` authors via a PAT with `workflow` scope;
`neverhuman` reviews and merges, or vice versa; both agent-operated on xbabe2).
**D2** PR gate = hygiene + rust-check + emitters + inputs, ≤15 min; full
nextest / LibTorch features / sanitizers / security / Jankurai run on
main/nightly as visible non-blocking lanes, promoted to blocking one at a time.
**D3** owner upgrades the org to Team; protection on `main` follows. **D4**
`veox-jopedime-{datasets,models,learning,archive}-use2` are the product
buckets; `jopedime-714044927654-us-east-2-{dev,prod,sealed}` stay as custody.
**D5** the ten real test datasets (`wpd1`) and the forty synthetic fixtures
(Codex `15a1d31`) are both admitted; "ten total" is superseded.

Division: Grok keeps its serial CLI lane and reviews my PRs; I take what its
board defers or is blocked on, because I hold the `workflow`-scope PAT and the
`veox-jepson` AWS credentials. My order: #7 ci/lanes → #10 nextest `pr`
profile → #14 → #12 → #9 → #11 `wpd1/datasets` (S3 read path) → #15
`wp7/models-cli` → `wp11/learning-store` (hash-chained HWM ledger = the
durable champion store; `champions --local` becomes a view over it) →
`jope pipeline run` on the real engine search → `hwm push|pull` (= G) →
`learn` (world-1 trainer) → nightly on xbabe2. Grok: H (docs). At most three
PRs open across both of us; each merged or drafted within 24 h.

Pushed today from the 09-19 session (never on GitHub before): `wp5/cli-split`
d0b4819, `wp8/runner-lanes` f6c101d, `wp11/learning-store` df9e4e0,
`wp12/arena` 53f5ec4, `wp7/models-cli` b497325 (blob-race fix). Bundle copies:
`s3://veox-jopedime-archive-use2/git-bundles/20260921/`. Owner-approved plan:
`/home/ubuntu/jope-migration-evidence/claude/jopedime-unstall-20260921/PLAN.md`.

## 2026-09-21 — Claude (integrator): correction — Grok released its lane; owner ruling

Correction to the entry above: Grok released all its JopeDime claims at
03:48Z (`/home/ubuntu/jope-migration-evidence/stopped-handoff-20260921T0343Z/pr-author-handoff.md`,
sha256 8f78c70e…) and will not review or continue the CLI lane. No lane is
assigned to Grok. Reviews of `jepsontaylor` PRs are posted as `neverhuman` by a
session that did not write the PR (fresh review agents; Cursor's coordinator is
invited). Owner ruling ≈03:58Z: D1–D5 control (see `MASTER_PLAN.md`).

## 2026-09-21 — Claude (integrator): the queue moved; 22 merges, `4235db6` → `70506066`

Durable record of 04:00–11:40Z, since this file stopped at the D1–D5 ruling.

**Merged under D1** (author `jepsontaylor`, reviewed and merged as
`neverhuman`, rebase merges, `main` read back after each): #23 `4753a9a`,
#24 `932f7b7`, #25 `dc5ccf9`, #26 `44443ab`, #27 `a6df267`, #28 `d1317a1`,
#29 `0c14405`, #30 `aa57388`, #31 `99981e6`, #32 `ff5da5b`, #33 `5c66edb`,
#35 `53ea45d`, #36 `ebf055d`, #37 `dc46e86`, #38 `d43bc5f`, #39 `7050606`.
With Grok's #17–#22 that is twenty-two merges on 2026-09-21. #34 is open
inside that range, so it is not a contiguous span.

**The gate was not what the boards said it was.** Until #38 and #39 the hosted
lanes ran no Python and almost no Rust tests — the sole exception was
`emitters` running `cargo test -p feat-math --lib -- cxx::` and `--test
portable_parity`. #38 exists because #32 broke an existing Python test that
nothing ran. #39 added the Rust half: a full-workspace `cargo test --no-run`
links ~72 GB against a runner's ~14 GB and dies with a bus error, so the lane
gates a **checked partition** — 11 product crates on pull requests, 9 deferred
to self-hosted, with `deploy/ci/rust-test-scope-check.py` asserting against
`cargo metadata` that the script's output and its arrays agree and leave no
crate unplaced. Measured on its first green run (`35586778652`): 651 tests
executed and passed, target 8.98 GB, 626 s of nextest in an 822 s job.
**`rust-check` still runs neither clippy nor rustfmt** — that is why #37 had
to clean up formatting drift after the fact.

**Review found what the author did not, repeatedly, and that is the point.**
#39 took four rounds: the lane could not build at first (72 GB link), then the
partition failed open on a crate in neither list, then the scope guard
discarded the script's exit status, and finally the guard was reading the
arrays in the file while the lane runs the script's *output* — a regex takes
the first match where bash takes the last assignment. Each was found by an
independent reviewer reproducing rather than reading. The same held on #34,
where four P1s in round 4 were followed by two more in round 5.

**A process defect worth recording so it is not repeated:** `gh pr create`
without `GH_TOKEN` opens the PR as the `gh` OAuth user (`neverhuman`), even
when the commits are authored and pushed as `jepsontaylor`. Since branch
protection is unavailable on a free private repo, identity separation is the
merge gate, and GitHub refuses the review as a self-review. #40 and #41 were
opened that way and had to be closed and reopened as #42 and #43. Check
`gh pr view <n> --json author` before requesting a review.

**Open at 11:40Z:** #34 (`wp11/learning-store`, round 5), #42 (ADR-0003 test
paths), #43 (board corrections). Pushed and held behind the three-PR cap:
`wp13/pipeline-run` — `jope pipeline run`, the first end-to-end
dataset → prepare → GP search → select → emit → receipt, built on
`feat_math::{evolve,fit,transpile}` because the engine's fit authority is
fenced in every shipped build — and `ci/rust-test-followup-20260921`.
