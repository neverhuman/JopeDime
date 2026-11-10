# H417 — APEX corpus and baseline (in progress)

**This is a status log, not a closeout.** No H417 AUDIT value has been read. Plan: [`PREREG.md`](PREREG.md),
[`PREREG_ADDENDUM_1.md`](PREREG_ADDENDUM_1.md).

| Time (UTC, 2026-11-10) | Item | State |
| --- | --- | --- |
| 07:01–09:53 | Search runs GS / GE / GL on the 221 `BASE` contexts (xbabe0) | complete, 221 records each; 2.9 h, 2.9 h and 3.9 h (my estimate was 1–1.5 h) |
| 09:54 | Deploy `APEX_BASE` started as one process on 96 cores | 25 of 221 contexts after 64 minutes |
| 10:58 | Deploy restarted as 8 context-disjoint processes (addendum 1) | — |
| **11:34** | **Deployed contexts** | **56 of 221**; 8 shard processes active; about 53 contexts per hour, so roughly 14:40 for the rest |
| 08:53–about 10:50 | Screener gate G0, 24 contexts (xbabe3) | finished, 1 status file; not pulled or read yet |

**Negatives.**
- My time estimates were wrong twice: the search runs and the deploy are bound by one Python thread per process,
  not by cores. The single-process deploy wasted about an hour of wall-clock with most of its cores idle.
- The CI pause rule on xbabe3 paused the sandbox 12 times since the watcher started (most during G0). Evaluations in
  flight during a pause can time out; timeouts per arm will be reported with the gate result.
- Only the core tier is deployed (rows ≥ 800): 115 `DEV` and 106 `FIT-SEL` families, about half of what the plan
  assumed, so only effects of about 2% RMSE or more will be detectable.
