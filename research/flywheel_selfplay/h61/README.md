# H61: Adapter Checkpoint Selection & Strict-Margin Admission Ablation (`S0`–`S3`)

## Overview
`H61` (`research/flywheel_selfplay/h61/`) evaluates four round-end adapter checkpoint selection rules (`S0`, `S1`, `S2`, `S3`) starting from an isolated `r882` checkpoint copy (`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`) with frozen DOPE backbone (`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`) across seeds `12, 13, 14` on `DL-Train` (`seed 4900` family folds).

- **Preregistration**: [PREREG.md](PREREG.md)
- **Resource & Coordinator Amendment**: [amendment-2026-10-31.md](amendment-2026-10-31.md)
- **Runner**: [runner.py](runner.py)
- **Wave Coordinator**: [run_wave.py](run_wave.py) (backed by shared FIFO lock helper [qualification_lock_fifo.py](../qualification_lock_fifo.py))
- **Unit & Red-Team Tests**: [test_h61.py](test_h61.py)

## Operational Note: `2026-10-31T04:54:44Z` Unfair-Lock Incident & FIFO Correction (`t2258`)
- **Observed Incident**: Under PR #816 (`immediate_first_retry=True`, `LOCK_POLL_SECONDS=5`), the H61 coordinator (`PID 3178624`, launched after H57 `null_sigma001/seed15` completed at `04:51:56Z`) acquired `~/jope_plan/qualification.lock` at `04:54:44Z` (`1.4 s` after H59 `PID 2250432` released it), ahead of H57 (`PID 3058000`), which had been waiting since `04:51:57Z` on a `30 s` poll.
- **Disposition**: Per overseer directive `t2258`, neither H61 (`PID 3178624`), H57 (`PID 3058000`), nor H59 (`PID 2250432`) was killed or paused mid-wave (preregistered waves have no mid-cell resume, and lock wait time for H57/H59 is excluded from their GPU cell-wall caps and bounded by their separate `3 h` lock-wait limit, making this a scheduling delay rather than an infra-void).
- **FIFO Lock Fix for Subsequent Waves**:
  - Removed immediate first retry and restored `LOCK_POLL_SECONDS = 30.0`.
  - Added FIFO ticket queue in `~/jope_plan/qualification.queue/<epoch_ns>-<pid>-<wave>` via `research/flywheel_selfplay/qualification_lock_fifo.py`:
    - Prunes tickets whose PID is dead (`pid_is_alive`).
    - Attempts `flock(LOCK_EX | LOCK_NB)` only when holding the oldest live ticket.
    - Deletes the ticket immediately upon acquiring the lock and enqueues a new ticket at the tail after releasing each GPU cell.
  - Added legacy-waiter backoff: after releasing a cell lock, if any other `*_STATUS.log` under `~/jope-data/h*/` has a `"WAIT qualification.lock"` line within the last `90 s`, the coordinator backs off `>= 45 s` (longer than the `30 s` poll interval of legacy H57/H59 waiters) before attempting `flock`.
