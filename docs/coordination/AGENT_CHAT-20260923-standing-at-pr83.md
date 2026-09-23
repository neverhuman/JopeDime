# Historical standing summary at PR83 main

Verbatim excerpt from `AGENT_CHAT.md` at `a75e543d2166017cb0139e0a251d51d9a0c98e08`.
This is the dated summary replaced during the manual work, preserved for provenance; it is not current status.

## Standing state (summary; the archived entries are the record)

- Owner direction, two sessions: on 2026-09-22 (first archive, 23:14 entry) the owner told Claude:
  product name **AFM**, the strong review rule (reviewer differs from the PR opener and from every
  commit author/committer), parallel Track A / Track B. On 2026-09-23 08:02 Codex recorded a direct
  owner assignment in its own session: name **ASM** (Algorithm Synthesis Model), **one serial
  implementation PR queue with #78 first**, Codex keeps schemas/, the HWM train-side admission
  (its approved step 6) and the frozen-panel evaluator (actual prepared folds, never recut; raw
  panel inputs outside Git). Claude follows the serial queue from now on and opens no new
  implementation PR ahead of it. The direct current assignment to Codex uses ASM; historical AFM identifiers remain
  unchanged for reproducibility. Codex has not requested redundant naming clarification.
- Merged 2026-09-23 (UTC): #66 00:29 (main 5a61aa7ca), #68 00:41 (72e6e3db8), #69 00:57 (d3443add9),
  #70 03:03 (d61896ae8), #71 03:41 (5a935bc8a), #72 03:53 (cfb432f0b), #73 06:09 (241cba2e3),
  #74 06:34 (93b4022dc), #75 06:55 (533cac49e), #76 07:22 (935f6f2a4), #77 07:43 (4000be870), #78 08:25 (20122362b), #79 09:16 (cfa0a70bd; main-guard 35842015244 passed), #80 09:42 (d34ddc11d), #81 09:55 (1108284ef). Every
  merge: exact-head CI green on both runs plus a native neverhuman approval at that head.
- Open: none. Queue handed back to Codex at ~09:57 UTC after PR 80 (09:42, d34ddc11d) and PR 81 (09:55, 1108284ef) merged.
- Lanes: canonical (Codex, feat/asm-bounded-execution, worker/controller integration claim); .agent/worktrees/honesty (Claude, prep/b5-motifs at PR81 main1108284ef; B5 custody preparation, unpublished);
  .agent/worktrees/bench (Claude, idle stopped head c8d35c235 = merged PR 81 head, clean).
- On main now: the two-ladder README (benchmark-valid vs best-known); jope-bench same-cut benchmark
  for regression, binary and multiclass with maximize ladders and tier-2 subprocess adapters
  (external engines scored by the benchmark on the same cuts); EMITTER_VERSION 2.2.0 with the
  shipped-text lint in every emitter, masking only the caller's own names; the prose lint
  deploy/ci/ip-scrub.sh in the hygiene lane.
- full-main-cpu on main has failed on every push since 2026-09-22 19:41 UTC. Inventory of the
  533cac49e run (read-only, Claude custody track-a-20260922/b6/inventory.json): full-default 29 of
  2598 failed in 15 groups, all in jope-dime-regressor (crates/jope-campaigns) and jope-parallel
  (crates/jope-engine); full-scratch hit its 600 s budget (feat-math golden
  fast_profile_portfolio_head alone takes about 590 s); full-neural unavailable (xbabe2 lacks
  /usr/include/cuda_runtime.h). Root causes verified (Claude custody b6/b6-digest.md); MASTER_PLAN repair
  #3 merged as PR 80 (5 jope-parallel ids), #2 (3 jope-dime-regressor ids) follows; the rest need owner or
  integration decisions recorded in the 08:27 entry.
- Budgets and campaign limits: as in Codex's 08:02 entry (20 CPUs/workers, 64 GiB, zero swap, one
  RTX 3090; no overlap with exclusive full-CPU CI). No panel or training campaign started by Claude.
- Custody paths in Codex entries are relative to `/home/ubuntu/jope-rescue-20260921/`; Claude
  receipts are under `/home/ubuntu/jope-rescue-20260921/track-a-20260922/`.

<!-- End of verbatim standing summary. -->
