# H55 pre-outcome execution amendment — 2026-10-30

This amendment is fixed before any H55 outcome cell. The full rationale, timing receipt, resource readings, training-budget invariants, and terminal DL-Val order are recorded in the dated execution-amendment section of `PREREGISTRATION.md`.

Timing-only EMA-25 seed 12 passed its independent audit at source commit `d5bf0c7406e566065aab43440cb015fa6d7b46ed`. Coordinator cell wall was 230.237 seconds; the fixed training projection is 1,105.1 seconds. No fold-B or DL-Val data were read.

The total locked-cell GPU wall cap includes all six outcome cells and their setup/scoring, summed across cells. Lock waiting is tracked separately and is subject to the three-hour cumulative contention limit; it does not count as GPU cell wall. The coordinator enforces the remaining GPU budget after each cell and stops before launching a cell if the cap is exhausted. This is a hard stop; no efficacy-based extension or budget change is allowed.

total_gpu_wall_cap_seconds: 1800

All preregistered arm definitions, seeds 12–14, round/update counts, fold-A-only training, admission rule, CUDA lock and memory guards, DOPE pin, CPU/RSS/disk limits, and the at-most-one-terminal-read-per-seed rule remain as stated in `PREREGISTRATION.md`.

Clarification recorded before outcomes: the first outcome coordinator attempt stopped before cell start after 903.6 seconds of lock wait because its deadline arithmetic confused wait time with GPU execution time. Its incomplete infra-void receipt is preserved under `~/jope-data/h55/h55-wave/outcomes/wave.json`; it contains no completed or partial H55 cell. The corrected coordinator carries the 903.6-second prior wait into the three-hour lock-wait budget and applies the unchanged 1,800-second cap to summed locked-cell runtime only.

The second attempt acquired the lock for `swa8/seed12` but the runner exited immediately before `prepare_training` because it referenced an unregistered CLI attribute (`timing_only`). Its infra-void receipt and log are preserved in `~/jope-data/h55/h55-wave-attempt2/`; no training input, model state, fold B, or DL-Val was read. Add its 412.705 seconds of lock wait to the prior 903.6 seconds. Charge the full 5.330-second attempt wall conservatively to the fixed locked-cell cap. The next attempt therefore carries 1,316.305 seconds of prior lock wait and 5.330 seconds of prior locked-cell wall; remaining GPU-cell budget is 1,794.670 seconds. The focused test now exercises the runner's outcome-mode path with a pre-training sentinel.

## 2026-10-31 clarification before any outcome cell

Attempt 3 is preserved at `~/jope-data/h55/h55-wave-attempt3/outcomes/wave.json`. It acquired the qualification lock for `swa8/seed12`, then the post-lock `safe_preflight()` rejected filesystem `/` at 95.35%, before launching `runner.py`; no H55 inputs, model, fold B, or DL-Val were read. Its last waiter record reached a cumulative 2,279 seconds, rounded to whole seconds. To avoid undercounting the unreported post-lock check and receipt time, charge its full 973.830-second coordinator wall conservatively to lock wait. Carry 2,290.135 seconds total lock wait (attempts 1–3) and 5.330 seconds locked-cell wall (attempt 2 only); the 1,800-second locked-cell GPU cap remains unchanged, leaving 1,794.670 seconds. Do not start attempt 4 until an immediate disk preflight reports `/` below 95.0%. The coordinator hardening will recheck the full preflight at five-minute intervals while waiting and will avoid acquiring the lock while the disk guard is red; the post-lock check remains mandatory.

Owner follow-up `luna_ns_hbaseline_h55_attempt3_void_t0012.md` directs attempt 4 to carry the recorded status-log cumulative value, approximately **2,279 seconds**, as `--prior-lock-wait-seconds`. This replaces the conservative whole-coordinator-wall estimate above for the attempt-4 argument. Preserve 5.330 seconds as prior locked-cell wall; the fixed 1,800-second cap and 1,794.670-second remaining budget do not change.

Owner ruling `luna_ns_hbaseline_h55_capclock_t1735.md`: the cap counts GPU cell wall only; lock wait is excluded and bounded at 3 h. The coordinator starts each cell's timeout after successful lock acquisition, carries cumulative lock wait separately in `wave.json`, and infra-voids when the three-hour contention budget is exhausted. No arm, seed, split, admission, or terminal-read rule changes.
