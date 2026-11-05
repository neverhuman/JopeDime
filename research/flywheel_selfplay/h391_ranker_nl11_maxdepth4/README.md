# H391 — ranker NL11MD4 (num_leaves15→11 / max_depth3→4)

**INCOMPLETE_SCRATCH_GUARD — scientific UNKNOWN.**

- Hypothesis: shallower-leaf/deeper-tree ranker beats H352 NL15 on paired drop10 (TRAIN r882 slot0, family-grouped 3 roots × 36 tasks).
- Prereg `978a0b3e9c` before fits. One capacity probe; no replay/reprobe.
- Own guard stopped `h391-run.service` at 2026-11-05T21:02:03Z: scratch 2001883939 B > 2 GB cap, at 30/36 verified fit tasks, 0 outer scores. CPU ~34871 s.
- No partial scoring (per prereg). No Elo. The finisher only accepts a COMPLETE exit, so it failed as expected.
- The lane screen sol_v1_api01 ran out of Codex quota (until Nov 9 22:40Z), so the overseer pass closed it from evidence `/home/ubuntu/jope_plan/evidence/H391/api01-20261105-t1905`.
- Custody: ~1.9 GB own scratch in ~/jope-data/h391-api01 is **retained, not archived** (no deletion by the overseer).
- DOPE c51a56d0 unloaded; no dev91/sealed/GPU/writer action.
- Next: rerun only with a 3 GB scratch cap or per-task archive+reclaim; NL11MD4 is still untested.
