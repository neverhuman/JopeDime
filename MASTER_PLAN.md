# JopeDime current direction

**2026-09-21.** Publication repo: private `veox-ai/JopeDime`. This note tracks
`origin/main` after PR F (#22, `1f0fc5a3eeaca5b536a902bad48d6cc09aa7b748`).

## Queue

PRs **#5–#16 were closed unmerged** on 2026-09-20 (administrative withdrawal,
not a merge or supersession). Source branches are retained. The full-history
bundle and per-PR repair checklists live in the builder's
`jope-pr-resolution/20260920` evidence tree (bundle SHA-256
`3bafaa52b9a2c6a6996a2dacc22b9c1bff9248b7833bf07d7cd9882fbcba978e`). See that
directory's `REPORT.md`.

PRs **A–E** ([#17](https://github.com/veox-ai/JopeDime/pull/17)–[#21](https://github.com/veox-ai/JopeDime/pull/21))
are merged. **Zero open PRs is not “done.”** Recover selected work **one PR at
a time**. Do not reopen the twelve-PR pile. Live S3 is not claimed. Bound
`jope-policy` checkpoint training is not claimed.

## Product target

```text
data → encoders (Hyperion, …) → DL policy → GP kernel →
  working Python/Rust/C++17/JS → S3 champion trees → train the router
```

Git holds tiny fixtures only (D5: the ten real test datasets from `wpd1` and the
forty synthetic fixtures from Codex `15a1d31` are both admitted). Real datasets
and weights stay in S3.
The neural policy is a product router, not a held-out SOTA claim.

## Serial recovery (one open PR)

| PR | Goal | Status |
| --- | --- | --- |
| A | Honest README / layout / MASTER_PLAN matching that SHA | Merged #17 `b388ada` |
| B | Ten Git fixtures + offline verify | Merged #18 `049410f` |
| C | Encode path (recover `wp3/hyperion-policy`, CPU/Candle) | Merged #19 `089c127` |
| D | Search/fit/emit CLI on feat-math | Merged #20 `8326c36` |
| E | Local champion bank; live S3 fail-closed | Merged #21 `86c88bd` |
| F | Local 128-d centroid router train-step + `jope route` | Merged #22 `1f0fc5a` |
| G | Live S3 champion pull/push | Claude (integrator): lands as `hwm push\|pull` over `wp11/learning-store` + `wpd1`/`wp7` S3 crates; credentials available |
| H | Docs/ADRs after the above are true | Not started |

Defer: Dependabot #5, workflow copy (#7, needs `workflow` scope), 365-file
Python dump (#8), timings (#10), census regen (#12), path mega-move (#13),
LibTorch (#14), forty-fixture wholesale import.

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
| D2 | PR gate = hygiene + rust-check + emitters + inputs (≤15 min, path-gated). Full nextest, LibTorch `features`, sanitizers, security, artifacts, Jankurai = visible non-blocking main/nightly lanes, each promoted to blocking once it passes on main. Jankurai is a no-regression ratchet, never a merge gate. |
| D3 | Owner upgrades `veox-ai` to Team; branch protection (required checks + 1 approval + no force-push) is enabled the moment the API stops returning 403. |
| D4 | `veox-jopedime-{datasets,models,learning,archive}-use2` (+usw2 replicas) = product buckets the crates target. `jopedime-714044927654-us-east-2-{dev,prod,sealed}` = preservation custody. Nothing deleted; `catalogs/s3.json` records both. |
| D5 | Both the ten real test datasets (`wpd1`) and the forty synthetic fixtures (Codex `15a1d31`) are admitted to Git. |

## Merge rule (per PR)

1. `gh pr checks <n>` all green at the head SHA in the PR.
2. Review posted by the other identity (`gh pr review <n> --approve` as `neverhuman`, or the PAT identity for Grok's PRs). Trust-surface PRs (workflows, lockfiles, credentials, protection manifests) get a second, security lens.
3. `gh pr merge <n> --merge` (original commits kept), then `git ls-remote origin main` recorded in the row below.
4. Never more than three PRs open across all agents; a PR older than 24 h is merged, drafted, or closed with a reason.

## Landing board (Claude integrator; old PR → new PR → merged SHA)

| Old PR | Branch | Head | New PR | Merged |
| --- | --- | --- | --- | --- |
| — | `coord/integrator-lane-20260921` (this) | | | |
| #7 | `ci/lanes` | 38aeaa1 | | |
| #10 | `wp1/test-timings` | 5f66e3f | | |
| #14 | `wp8/neural-receipt` | 44501a9 | | |
| #12 | `wp10/parity-census` | c483544 | | |
| #9 | `wp7/model-manifests` | f62095c | | |
| #11 | `wpd1/datasets` | d672e76 | | |
| #15 | `wp7/models-cli` | b497325 | | |
| #8 | `wp6/python` | d49662f | | |
| #13 | `wp4/paths` | 96a61c8 | | |
| #16 | `wp9/docs-release` | 9bec536 | | |
| — | `wp11/learning-store` | df9e4e0 | | |
| — | `wp12/arena` | 53f5ec4 | | |
| — | `wp5/cli-split` | d0b4819 | | |
| — | `wp8/runner-lanes` | f6c101d | | |
| #5 | dependabot cargo group | 1151493 | | |

Recovery of #5–#16: bundle `pr-5-through-16.bundle` (sha256 3bafaa52…) is also
at `s3://veox-jopedime-archive-use2/git-bundles/20260921/`.
