# JopeDime GitHub product coordination

This file is for **veox-ai/JopeDime** only. The 8×A100 node log is a different
tree and must not be pasted here.

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

## 2026-09-22T03:17:59.926108+00:00 — Codex — agent-contract follow-up

Claim canonical `/home/ubuntu/JopeDime` on `docs/master-cleanup-20260921` at
`5d251f50a5e4b80897e383fb02215e6eeded71f2` for `AGENTS.md`, `CLAUDE.md` and this coordination
entry. The owner requested studying Bulletfarm's agent setup for useful JopeDime lessons.
Scope: explicit ownership/handoff, concrete PR acceptance and evidence, and independent review
against PR and commit authors. Keep JopeDime's registered worktrees, reused targets, serial
queue and history-preserving shared-branch updates. No new board service, product behavior,
cleanup, campaign or Bulletfarm mutation. Evidence:
`/home/ubuntu/jope-rescue-20260921/implementation-20260922/bulletfarm-contract-study`.
Existing PR #49 still waits for the actual 04:00 UTC report; no new lane or PR.

## 2026-09-22T03:26:48.742285+00:00 — Codex — contract review corrections

Extend the same PR49 documentation claim to `MASTER_PLAN.md` to reconcile its
contradictory reviewer, queue and worktree rules with the current user direction.
Claude's read-only preliminary review found no blocking defect and requested
these corrections plus explicit claim locking, merge guards and budget records.
Verification: `git diff --check`, local link/import checks and independent Claude
review; final exact-head CI and full PR49 review follow the actual next-day report.
The earlier claim append did not hold this file's documented lock; no overlapping
writer or changed head was observed. This correction is written under the lock.

## 2026-09-22T03:37:40.396482+00:00 — Codex — current direction and external merge readback

The same four documentation paths remain claimed. MASTER_PLAN now records the
user's Sep22 neural-GP loop goal, current merge/CI observations and the corrected
workflow. Another session merged #44, #50 and #51; this session did not. The
#44 approval discloses author involvement in its last two CI commits. Codex owns
independent validation of the pipeline and those assertions with correction #52,
after #49 closes. This is not retroactive approval of the earlier merge.
Review/source evidence: `implementation-20260922/queue/external-merges-20260922.json`
and `pipeline-budget-issue.md` under `/home/ubuntu/jope-rescue-20260921/`.
The registered #44 lane is now absent; its removal and preservation receipt have
not been established by this session. No lane was removed by this task.

Current HEAD remains `5d251f50a5e4b80897e383fb02215e6eeded71f2`; dirty paths are the
four owned documentation files. The updated diff, hashes and link checks will be
sealed after these corrections and before committing. The actual 04:00 report,
final PR49 head checks and complete independent review remain outstanding.
