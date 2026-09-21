# JopeDime current direction

**2026-09-21.** Publication repo: private `veox-ai/JopeDime`. This note tracks
`origin/main` after PR A (`b388ada472c70877a314c2f7e82508d4dcde9b2c`) plus PR B
(ten Git fixtures) in flight.

## Queue

PRs **#5–#16 were closed unmerged** on 2026-09-20 (administrative withdrawal,
not a merge or supersession). Source branches are retained. The full-history
bundle and per-PR repair checklists live in the builder's
`jope-pr-resolution/20260920` evidence tree (bundle SHA-256
`3bafaa52b9a2c6a6996a2dacc22b9c1bff9248b7833bf07d7cd9882fbcba978e`). See that
directory's `REPORT.md`.

PR **A merged** as [#17](https://github.com/veox-ai/JopeDime/pull/17). **Zero
open PRs is not “done.”** Recover selected work **one PR at a time** from
`origin/main`, rebase, CI, review, squash-merge, verify `main`, then the next.
Do not reopen the twelve-PR pile. Do not start C while B is open.

## Product target

```text
data → encoders (Hyperion, …) → DL policy → GP kernel →
  working Python/Rust/C++17/JS → S3 champion trees → train the router
```

Git holds **at most ten** tiny fixtures. Real datasets and weights stay in S3.
The neural policy is a product router, not a held-out SOTA claim.

## Serial recovery (do not start C while B is open)

| PR | Goal | Status |
| --- | --- | --- |
| A | Honest README / layout / MASTER_PLAN matching that SHA | Merged #17 `b388ada` |
| B (this) | Ten Git fixtures + offline verify | In flight |
| C | Encode path (recover `wp3/hyperion-policy`, CPU/Candle) | Not started |
| D | Search/fit/emit CLI on feat-math | Not started |
| E | S3 champion pull/push + one training step in **dev** | Not started |
| F | Docs/ADRs after A–E are true | Not started |

Defer: Dependabot #5, workflow copy (#7, needs `workflow` scope), 365-file
Python dump (#8), timings (#10), census regen (#12), path mega-move (#13),
LibTorch (#14), forty-fixture wholesale import.

## Blockers (do not fake)

- GitHub OAuth: `gist` / `read:org` / `repo`, no `workflow` scope. Do not
  replace `.github/workflows/ci.yml` from this token.
- Private-repo rulesets API returns HTTP 403 until GitHub Pro entitlement.
- No `--admin` merge. No worktrees. No credentials in git.

Coordinate in [`AGENT_CHAT.md`](AGENT_CHAT.md).
