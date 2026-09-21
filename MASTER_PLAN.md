# JopeDime current direction

**2026-09-21.** Publication repo: private `veox-ai/JopeDime`. This note tracks
`origin/main` after PR E (`86c88bd7295b7430119e73272994df9ee2530ed4`) plus the
local centroid-router train-step in flight.

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

Git holds **at most ten** tiny fixtures. Real datasets and weights stay in S3.
The neural policy is a product router, not a held-out SOTA claim.

## Serial recovery (one open PR)

| PR | Goal | Status |
| --- | --- | --- |
| A | Honest README / layout / MASTER_PLAN matching that SHA | Merged #17 `b388ada` |
| B | Ten Git fixtures + offline verify | Merged #18 `049410f` |
| C | Encode path (recover `wp3/hyperion-policy`, CPU/Candle) | Merged #19 `089c127` |
| D | Search/fit/emit CLI on feat-math | Merged #20 `8326c36` |
| E | Local champion bank; live S3 fail-closed | Merged #21 `86c88bd` |
| F (this) | Local 128-d centroid router train-step + `jope route` | In flight |
| G | Live S3 champion pull/push in **dev** | Blocked (no AWS credentials) |
| H | Docs/ADRs after the above are true | Not started |

Defer: Dependabot #5, workflow copy (#7, needs `workflow` scope), 365-file
Python dump (#8), timings (#10), census regen (#12), path mega-move (#13),
LibTorch (#14), forty-fixture wholesale import.

## Blockers (do not fake)

- GitHub OAuth: `gist` / `read:org` / `repo`, no `workflow` scope. Do not
  replace `.github/workflows/ci.yml` from this token.
- Private-repo rulesets API returns HTTP 403 until GitHub Pro entitlement.
- No `--admin` merge. No worktrees. No credentials in git.

Coordinate in [`AGENT_CHAT.md`](AGENT_CHAT.md).
