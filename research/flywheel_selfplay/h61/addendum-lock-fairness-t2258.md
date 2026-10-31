# H61 Post-Launch Operational Addendum: `qualification.lock` Fairness Correction (`t2258` / `t2303`)

- **Date / Time**: `2026-10-31T05:05:00Z`
- **Wave**: `H61` (`research/flywheel_selfplay/h61/`)
- **Live H61 Execution Provenance Pin**: Commit `db8faaa158c7f7460186a6b92805309e0f0b2cbe` (`PR #816`, worktree `/home/ubuntu/JopeDime/.agent/worktrees/agy_h61`, coordinator `PID 3178624`, run root `/home/ubuntu/jope-data/h61/h61-wave`).
- **Frozen Pre-Outcome Amendment**: [amendment-2026-10-31.md](amendment-2026-10-31.md) (`SHA256: 3edfad892b40f60909af0c05c8a6c75c6c39c7e4291bc6451541b3f23ec6fe62`, preserved byte-identical to `db8faaa158c7f7460186a6b92805309e0f0b2cbe`).

---

## 1. Incident Timestamps & Provenance Statement

1. **`2026-10-31T04:51:56.839951Z`**: `H57` (`PID 3058000`) completed cell `null_sigma001/seed15` (`wall=555.56s`) and began waiting for `~/jope_plan/qualification.lock` at `04:51:57.873242Z` (polling every `30 s`).
2. **`2026-10-31T04:53:18.167222Z`**: Per `t2235` Step 5, the detached `H61` coordinator (`PID 3178624`) was started from `/home/ubuntu/JopeDime/.agent/worktrees/agy_h61` at commit `db8faaa158c7f7460186a6b92805309e0f0b2cbe` (`PR #816`, which had `immediate_first_retry=True` and `LOCK_POLL_SECONDS=5.0`).
3. **`2026-10-31T04:54:44Z`**: `H61` (`PID 3178624`) acquired `qualification.lock` `1.4 s` after `H59` (`PID 2250432`) released it, ahead of `H57` (`PID 3058000`).
4. **Provenance & Scientific Isolation**:
   - **H61 outcome run used PR #816 immediate retry at db8faaa; fairness fix applies to later runs only; lock policy does not touch any outcome computation.** (Full commit SHA: `db8faaa158c7f7460186a6b92805309e0f0b2cbe`.)
   - Neither `H61` (`PID 3178624`), `H57` (`PID 3058000`), nor `H59` (`PID 2250432`) was killed or paused mid-cell. Lock wait times for `H57` and `H59` are excluded from their GPU cell-wall caps and bounded by their separate `3 h` wait limit (scheduling delay only, not an infra-void).

---

## 2. FIFO Lock Fairness Fix for Subsequent Runs (`qualification_lock_fifo.py` & `h61/run_wave.py`)

For all runs launched after `H61` (`PID 3178624`):
- Immediate first retry is removed and lock polling is restored to `30.0 s` (`LOCK_POLL_SECONDS = 30.0`).
- Coordinators enqueue a FIFO ticket `~/jope_plan/qualification.queue/<epoch_ns>-<pid>-<wave>`, prune tickets belonging to dead PIDs, attempt `flock(LOCK_EX | LOCK_NB)` only when holding the oldest live ticket, delete their ticket upon acquiring the lock, and enqueue a new ticket at the tail after releasing each GPU cell.
- After releasing a cell lock, if any other `*_STATUS.log` under `~/jope-data/h*/` has a `"WAIT qualification.lock"` line within the last `90 s`, the coordinator backs off `>= 45 s` before re-trying.
