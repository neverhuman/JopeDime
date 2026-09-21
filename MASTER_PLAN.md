# JopeDime current direction

**2026-09-21.** Publication repo: private `veox-ai/JopeDime`. This note tracks
`origin/main`; the landing board below records each merge SHA. **Twenty-two**
PRs merged on 2026-09-21: Grok's #17–#22 (01:24–03:16Z, `4235db6` → `1f0fc5a`),
then #23–#33 and #35–#39 under D1 (03:58–11:11Z, `1f0fc5a` → `70506066`).
#34 is open, which is why that is two spans and not one. Every readback
line in this file is regenerated, not transcribed; the command that produced
it is next to it.

## Queue

PRs **#5–#16 were closed unmerged** on 2026-09-20 (administrative withdrawal,
not a merge or supersession). Source branches are retained. The full-history
bundle and per-PR repair checklists live in the builder's
`jope-pr-resolution/20260920` evidence tree (bundle SHA-256
`3bafaa52b9a2c6a6996a2dacc22b9c1bff9248b7833bf07d7cd9882fbcba978e`). See that
directory's `REPORT.md`.

PRs **A–F** ([#17](https://github.com/veox-ai/JopeDime/pull/17)–[#22](https://github.com/veox-ai/JopeDime/pull/22))
are merged, and so is everything through #39 except #34 — see the landing board. **Zero open PRs is not “done.”** Recover selected work **one PR at
a time**. Do not reopen the twelve-PR pile. Live S3 is not claimed. Bound
`jope-policy` checkpoint training is not claimed.

## Product target

```text
data → encoders (Hyperion, …) → DL policy → GP kernel →
  working Python/Rust/C++17/JS → S3 champion trees → train the router
```

Git holds tiny fixtures only (D5: the ten licensed test datasets from `wpd1`,
≈1 MB, and the forty synthetic fixtures from Codex `15a1d31`, ≈0.8 MB, are both
admitted). Every other dataset and all weights stay in S3.
The neural policy is a product router, not a held-out SOTA claim.

## Serial recovery (Grok's lane, one open PR at a time; now merged through F)

| PR | Goal | Status |
| --- | --- | --- |
| A | Honest README / layout / MASTER_PLAN matching that SHA | Merged #17 `b388ada` |
| B | Ten Git fixtures + offline verify | Merged #18 `049410f` |
| C | Encode path (recover `wp3/hyperion-policy`, CPU/Candle) | Merged #19 `089c127` |
| D | Search/fit/emit CLI on feat-math | Merged #20 `8326c36` |
| E | Local champion bank; live S3 fail-closed | Merged #21 `86c88bd` |
| F | Local 128-d centroid router train-step + `jope route` | Merged #22 `1f0fc5a` |
| G | Live S3 champion pull/push | Claude (integrator): `wp11/learning-store` is PR #34, in round 5; `hwm pull` and the S3 exercise are not written yet — every test so far is `file://` |
| H | Docs/ADRs after the above are true | Not started |

Grok's lane deferred: Dependabot #5, workflow copy (#7), 365-file Python dump
(#8), timings (#10), census regen (#12), path mega-move (#13), LibTorch (#14),
forty-fixture wholesale import. Those items are queued on the integrator's
landing board below (it holds the `workflow`-scope PAT and the AWS
credentials that blocked them here); none is a wholesale import — each lands
as its own reviewed PR.

## Blockers (do not fake)

- `neverhuman` gh OAuth has no `workflow` scope; `jepsontaylor`'s PAT does —
  workflow changes are authored from that identity (Claude) and reviewed by
  `neverhuman`.
- Private-repo rulesets API returns HTTP 403 until the org is upgraded (D3,
  owner action in progress). Until then the merge rule below is process-only.
- No `--admin` merge. No worktrees. No credentials in git.

Coordinate in [`AGENT_CHAT.md`](AGENT_CHAT.md).

## Owner decisions — 2026-09-21 (newest direction; supersede the 09-20 holds)

| # | Decision |
| --- | --- |
| D1 | Agents merge: hosted CI green at the exact head + review by the *other* GitHub identity than the author (`jepsontaylor` PAT / `neverhuman` gh; both agent-operated on xbabe2) + merge SHA and `main` readback recorded here. |
| D2 | PR gate at `70506066` = the seven jobs `ci.yml` defines: `changes`, `hygiene`, `rust-check`, `emitters`, `inputs`, `python` (#38), `rust-test` (#39) — all path-gated. **`rust-check` runs `cargo check --workspace --all-targets --locked` plus a systemd test and an MSRV check; it runs neither clippy nor rustfmt** — those exist only in the local `justfile`, which is why #37 had to land rustfmt drift that no lane would have caught. Full nextest, LibTorch `features`, sanitizers, security, artifacts, Jankurai = visible non-blocking main/nightly lanes, each promoted to blocking once it passes on main. Jankurai is a no-regression ratchet, never a merge gate. |
| D3 | Owner upgrades `veox-ai` to Team; branch protection (required checks + 1 approval + no force-push) is enabled the moment the API stops returning 403. |
| D4 | `veox-jopedime-{datasets,models,learning,archive}-use2` (+usw2 replicas) = product buckets the crates target. `jopedime-714044927654-us-east-2-{dev,prod,sealed}` = preservation custody. Nothing deleted; `catalogs/s3.json` records both. |
| D5 | Both the ten real test datasets (`wpd1`) and the forty synthetic fixtures (Codex `15a1d31`) are admitted to Git. |

## Merge rule (per PR)

1. Required contexts = every job that `.github/workflows/ci.yml` defines **at the reviewed head** (on `main` at `70506066`: `changes`, `hygiene`, `rust-check`, `emitters`, `inputs`, `python`, `rust-test`, where a path-gated lane that did not run because its paths are untouched is recorded as `skipped-by-path`, not as green). `gh pr checks <n>` must show every required context `pass` at the exact head SHA; absent, cancelled or failed = not green. Heavy lanes (full nextest, LibTorch `features`, sanitizers, security, Jankurai) are main/nightly lanes per D2 and are not required on PRs until each has passed on `main`.
2. Review posted by the other GitHub identity than the author (`neverhuman` for PRs authored by `jepsontaylor`, and vice versa), from a session that did not write the PR. **The identity that matters is the PR's author, not the commits'.** `gh pr create` without `GH_TOKEN` opens the PR as the `gh` OAuth user, and GitHub then refuses the review outright: `Can not request changes on your own pull request`. #40 and #41 were opened that way and had to be closed and reopened as #42 and #43. Check `gh pr view <n> --json author` before requesting a review (a fresh review agent or another lane — Cursor's coordinator is invited). Both logins are operated by agents on this host; an approval proves an independent read at the exact head, not a human sign-off. Trust-surface PRs (workflows, lockfiles, credentials, protection manifests) get a second, security lens.
3. `gh pr merge <n> --rebase` (merge commits are disabled in the repository settings; rebase keeps each original commit, squash is acceptable for single-commit docs PRs), then `git ls-remote origin main` recorded in the row below.
4. Never more than three PRs open across all agents (D1 supersedes Grok's one-open-PR rule for its lane only while that lane is active); a PR older than 24 h is merged, drafted, or closed with a reason.

## Landing board (Claude integrator; old PR → new PR → merged SHA)

Open rows are pinned to the head and review they describe, exactly as merged
rows are pinned to a merge SHA. A row about a moving PR is unfalsifiable and
goes stale within the hour; live state belongs on the PR, where GitHub is the
source of truth.

| Old PR | Branch | Head | New PR | Merged |
| --- | --- | --- | --- | --- |
| — | `coord/integrator-lane-20260921` | | #23 | `4753a9a` 04:06:47Z |
| #7 | `ci/lanes` | 38aeaa1 | #24 | `932f7b7` 04:06:52Z (rebase; + inputs lane; review found the strict-parity step ran 0 tests — fixed) |
| #10 | `wp1/test-timings` | 5f66e3f | #29 | `0c14405` 05:08:29Z (rebase; 3 nextest exclusions module-qualified after review) |
| #14 | `wp8/neural-receipt` | 44501a9 | #25 | `dc5ccf9` 2026-09-21T03:58:58Z (rebase; reviewer `neverhuman`) |
| #12 | `wp10/parity-census` | c483544 | #28 | `d1317a1` 04:16:19Z (rebase; + rows for the #19 Hyperion bins) |
| #9 | `wp7/model-manifests` | f62095c | #26 | `44443ab` 04:08:07Z (rebase; manifests byte-identical to S3) |
| #6 | `wp3/hyperion-policy` | a043f8b | — | recovered by Grok's #19 (`089c127`): `main` is a strict superset; no PR needed |
| #11 | `wpd1/datasets` | d672e76 | #27 | `a6df267` 04:44:49Z (rebase; object_store 0.14 for RUSTSEC-2026-0194/0195; scoped webpki-root-certs exception; lock additive) |
| #15 | `wp7/models-cli` | b497325 | #33 | `5c66edb` 04:54:50Z (rebase; `jope models`/`datasets` umbrella on Grok's verbs; blob-lock deadlock fixed) |
| #8 | `wp6/python` | d49662f | #31 | `99981e6` 04:42:33Z (rebase; ruff format + pidfd fallback) |
| #13 | `wp4/paths` | 96a61c8 | #32 | `ff5da5b` 05:04:22Z (rebase; abs-path hygiene step now blocking for crates/deploy/.github) |
| #16 | `wp9/docs-release` | 9bec536 | #30 | `aa57388` 04:33:58Z (rebase; README kept as Grok's honest front door, quickstart → docs/quickstart.md, status banners) |
| — | `wp12/arena` | 53f5ec4 | #35 | `53ea45d` 08:21:44Z (rebase; rustls provider guard fails closed, ureq 3 proxy/redirect deltas disclosed) |
| — | board | | #36 | `ebf055d` 06:36:46Z (landing rows through #33) |
| — | rustfmt drift | | #37 | `dc46e86` 06:47:00Z (two drifted hunks in `jope-engine eval_registry/query`) |
| — | python lane | | #38 | `d43bc5f` 07:49:28Z (fixes the systemd test #32 broke; **adds the hosted `python` lane that would have caught it**) |
| — | `wp11/learning-store` | df9e4e0 | **#34 open** | **at `5737aca7`, review 11:29:54Z**: all four P1s of round 4 fixed — three reproduced from the CLI, the fourth (non-finite entry refusal) read only, since no CLI source can carry a non-finite value in. Blocking then: `jope-learning` in neither test scope, so `check.sh` fails closed on the merged tree; the rebase; and two P2s. Live state is on the PR, not here |
| — | rust-test lane | | #39 | `7050606` 11:11:35Z (4 rounds; the hosted gate ran no Rust tests beyond two `feat-math` targets, and the scope guard itself failed open twice before the reviewer's negative probes closed it) |
| — | ADR-0003 test paths | | **#42 open** | **at `648c6c4f`**: three tests red since the ADR-0003 relocation, fixed; their quarantine entries dropped. #40 was the same branch opened from the wrong identity and is closed |
| — | `wp13/pipeline-run` | | queued | `jope pipeline run`: dataset → prepare → GP search on feat-math → select → emit → receipt |
| — | `wp5/cli-split` | d0b4819 | | needs a mechanical re-do on current main, not a replay (#32 rewrote 67 of the files it moves) |
| — | `wp8/runner-lanes` | f6c101d | | |
| #5 | dependabot cargo group | 1151493 | | re-trigger after main settles; lockfile = trust surface, two lenses |

Recovery of #5–#16: bundle `pr-5-through-16.bundle` (sha256 3bafaa52…) is also
at `s3://veox-jopedime-archive-use2/git-bundles/20260921/`.

## What the gate actually proves (2026-09-21, measured)

Read this before trusting a green check. Every number here names where it was
measured; a figure without a source is a defect in this section.

- Until #38 and #39 the hosted gate ran **almost** no tests: `hygiene`,
  `rust-check`, `emitters`, `inputs`. The one exception is real and worth
  naming rather than rounding away — `emitters` ran `cargo test -p feat-math
  --lib -- cxx::` and `--test portable_parity`. Nothing else. No other crate's
  tests and no Python ran at all, so a green PR meant "it compiles and the
  emitters agree". #38 added the `python` lane, and exists *because* #32 broke
  an existing Python test that nothing ran. #39 added the Rust half.
- **`rust-check` is `cargo check`, not a lint gate.** It runs `cargo check
  --workspace --all-targets --locked`, the `systemd_unit_paths` test and an
  MSRV check. Clippy and rustfmt live only in the local `justfile`, so
  formatting drift reaches `main` unchallenged — #37 is exactly that, landed
  after the fact. Adding `cargo fmt --all --check` to the gate is cheap and
  owed, and is already written on `ci/rust-test-followup-20260921`
  (`f1cbf820`) awaiting a slot. Clippy cannot join it: `cargo clippy
  --workspace --all-targets --locked -- -D warnings` at `70506066` on this
  host stops in `feat-math` — **31 errors in the lib, 37 in the lib test** —
  and aborts (`RC=101`) before compiling anything else. So the `~31 inside`
  half of the figure this board quoted at `9628a652` reproduces almost exactly,
  and the `143 outside feat-math` half is **unmeasured rather than refuted**:
  nothing outside `feat-math` was compiled, so no figure for the workspace is
  quoted here. `feat-math` has to be cleared before the rest can even be
  counted.
- **The hosted `rust-test` lane, measured on the first green run that retained
  an artifact** (`35586778652`, artifact `rust-test-evidence` = `10633028231`,
  independently audited by the qualification reviewer; the lane was already
  green at `35580809723` 09:00Z and `35583896833` 09:32Z, neither of which kept
  anything — which is the whole reason retention landed): **651 tests executed and passed**, target **8.98 GB**, free disk
  11.5 GB before and 3.9 GB after, nextest **626 s** inside an **822 s** job.
  Full product inventory 664 = 651 executed + 9 ignored + 4 filtered by the
  `pr` profile. The tree's own comments said 9.4 GB / 617 tests / "about a
  minute" — local numbers from before the lane ever ran hosted, corrected on
  `ci/rust-test-followup-20260921`.
- A full-workspace `cargo test --no-run` links ~72 GB against a runner's ~14 GB
  (documented identically in `deploy/ci/rust-test-scope.sh` and `ci.yml`) — it
  SIGBUSes rather than failing a test. #39 therefore gates a **checked
  partition**: 11 product crates on pull requests, 9 deferred to self-hosted,
  with `deploy/ci/rust-test-scope-check.py` asserting against `cargo metadata`
  that the two agree with the script's own output and leave nothing unplaced.
  A crate in neither list would be gated by nothing; `jope-arena` did that
  once, and the guard caught `jope-pipeline` doing it next — on the unmerged
  `wp13/pipeline-run` (`d13cc7e7`), so do not look for that crate on `main`.
- **A full-workspace run is red on `main` for reasons unrelated to any branch.**
  18 failures across `jope-parallel`, `jainnode` and `jain-smartcluster-daemon`,
  measured by the integrator session on this host with `cargo nextest run -P pr
  --locked --workspace` on 2026-09-21 and reproduced on `main` itself. **This
  has not been reproduced in CI and has no receipt in the tree** — treat it as
  an integrator measurement, not a verified fact, and re-measure before relying
  on it. Note the `pr` profile *as gated* is green: 651/651. Note also that
  `deploy/ci/known-broken-tests.txt` contains **zero** `jainnode` and
  `jain-smartcluster-daemon` entries: every entry in it is a `jope-parallel`
  binary, so the file is scoped to that crate and is not a census of what fails
  on `main`. It should say so.
- **The quarantine, counted rather than estimated** (15 entries at
  `70506066`): the `cause A` block holds 9, but the fit-authority fence
  explains **5** of them (2 "namespace capability inactive", 3 "fit authority
  fenced"). The other 4 are 3 × the ADR-0003 relocation (cause A0, fixed in the
  open #42) and 1 × a `queue.rs:133` panic that was misfiled under the fence.
  `cause B` (behavioural assertions) holds 5 and `cause C` (racy, passes
  serially) holds 1. #42 takes the file to 12 and unfiles the panic.

## Correction to the product plan: the engine cannot measure

`crates/jope-engine/src/fit_boundary.rs` requires `JOPE_FIT_SERVICE_SOCKET`
unconditionally — no socket, no fitting — and
`eval_registry/authority.rs::activation_block` returns a block for every
non-test scope ("production capabilities have no verifier yet"), so **no
shipped binary can measure through the engine**. That is not a bug to route
around quietly: it is a deliberate fence, and it explains **5** of the 15
quarantined tests — not the 9 the `cause A` block appears to hold, which is
why that block is being unfiled in #42.

Consequence for the pipeline: **WP-P1 (`jope pipeline run`) builds on
`feat_math::evolve` + `feat_math::fit` + `feat_math::transpile`, not on
`jope-engine`'s held entry points.** `crates/jope-cli/src/kernel.rs` is the
unfenced path today, but its `SEARCH_CANDIDATES` is a three-candidate loop, not
a genetic program — the GP is `feat_math::evolve`
(`random_canonical_pipeline` / `mutate_canonical_pipeline` /
`crossover_canonical_pipelines`) and it is complete. Lifting the fence is a
separate, reviewed change with its own verifier, never a side effect of a
pipeline PR.

## Recorded conflict — owner to settle (2026-09-21)

Two agents hold different owner directions dated 2026-09-21:

- **Integrator (Claude, this board's D1–D5):** rapid PRs; agents merge after CI +
  cross-identity review; fast PR gate; both fixture sets admitted; org upgrade
  for protection. Owner-approved plan copy:
  `/home/ubuntu/jope-migration-evidence/claude/jopedime-unstall-20260921/PLAN.md`.
- **Coordinator (`jdime_merger_cursor`, review on #23 at 03:50Z):** a
  "stopped-writer plan" requiring explicit owner releases from every writer,
  stopped preservation workers/supervisors and restoration first, one
  implementation/ledger owner, exactly ten synthetic scenarios, and
  protection/qualification before any merge. It reports that Grok has released
  its JopeDime lane and that the preservation owner has not.

**Owner ruling (2026-09-21 ≈03:58Z, given directly to the integrator session
with all three directions quoted verbatim): D1–D5 control.** Codex's foundation
work lands through the same reviewed-PR gate one small PR at a time with the
≥95 audit as a ratchet; Cursor keeps the reviewer/coordinator role; the
preservation lane and its live workers (PIDs 115122, 3692153, 3692154, 3777416
at 03:20Z) are not touched by anyone. The stopped-writer and audit-before-merge
holds do not apply to this repository from that time. Posted on the canonical
board, the canonical chat and the cross-tree wire at 03:58:38Z.

Merge history for the record: #1–#16 were closed unmerged (2026-09-18/20);
#17–#22 (Grok's A–F) are the first merges in this repository, 2026-09-21
01:24–03:16Z, `main` `4235db6` → `1f0fc5a`.

## Known stale — do not cite

Claims that were true once and are not now. They are listed rather than
silently deleted, because they have been quoted at each other across three
sessions.

| Claim, and where it still appears | Why it is wrong |
| --- | --- |
| "617 tests", "9.4 GB", "about a minute" — `deploy/ci/rust-test-scope.sh` and `.github/workflows/ci.yml` at `70506066` | Local numbers from before the lane ever ran hosted. Measured: 651 tests, 8.98 GB, 626 s. Corrected on `ci/rust-test-followup-20260921`; the tree carries the old ones until that lands. |
| "the hosted gate ran no Rust tests at all" — this board's earlier draft and the #39 discussion | `emitters` ran `cargo test -p feat-math --lib -- cxx::` and `--test portable_parity`. The rest of the claim holds. |
| "the fence explains 9 of the quarantined tests" — this board's earlier draft | It explains 5. The `cause A` block held 9 because 3 relocation failures and 1 panic were misfiled under it. |
| "`rust-check` runs clippy and fmt" — the D2 row before this revision | It runs `cargo check`, a systemd test and an MSRV check. Neither linter is in any hosted lane. |
| In the stale `jain-split` mirror: "Eight-A100 recovery" in the header of `/home/ubuntu/jain-split/jope-dime/AGENT_CHAT.md:12` (not this repository's file, whose header says the opposite), and `main` at `458ef58` in that tree's `FINAL_GITHUB_PORT_PLAN.md` and `CODEX_GITHUB_PORT_PLAN.md`. "13 open PRs" and "zero workflows" are in neither file and were themselves a misattribution by this appendix | Pre-2026-09-21 state of the stale `jain-split` mirror, which is now a pointer. |
| PR #40, PR #41 | Opened from the reviewer's identity and closed unmerged; the same branches are #42 and #43. Not withdrawals of the work. |
