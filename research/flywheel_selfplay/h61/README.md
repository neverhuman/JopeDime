# H61: Adapter Checkpoint Selection & Strict-Margin Admission Ablation (`S0`–`S3`)

## Overview
`H61` (`research/flywheel_selfplay/h61/`) evaluates four round-end adapter checkpoint selection rules (`S0`, `S1`, `S2`, `S3`) starting from an isolated `r882` checkpoint copy (`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`) with frozen DOPE backbone (`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`) across seeds `12, 13, 14` on `DL-Train` (`seed 4900` family folds).

- **Preregistration**: [PREREG.md](PREREG.md)
- **Frozen Pre-Outcome Resource Amendment**: [amendment-2026-10-31.md](amendment-2026-10-31.md) (`SHA256: 3edfad892b40f60909af0c05c8a6c75c6c39c7e4291bc6451541b3f23ec6fe62`, byte-identical to `db8faaa158c7f7460186a6b92805309e0f0b2cbe`)
- **Post-Launch Lock-Fairness Addendum (`t2258` / `t2303`)**: [addendum-lock-fairness-t2258.md](addendum-lock-fairness-t2258.md)
- **Runner**: [runner.py](runner.py)
- **Wave Coordinator**: [run_wave.py](run_wave.py) (backed by shared FIFO lock helper [qualification_lock_fifo.py](../qualification_lock_fifo.py))
- **Unit & Red-Team Tests**: [test_h61.py](test_h61.py)

## Provenance Pin for H61 Outcome Run (`db8faaa`)
- **Code Commit Producing H61 Results**: `db8faaa158c7f7460186a6b92805309e0f0b2cbe` (`db8faaa`, **PR #816**, executed from `/home/ubuntu/JopeDime/.agent/worktrees/agy_h61` by coordinator `PID 3178624` into `/home/ubuntu/jope-data/h61/h61-wave`).
- **Operational Note (`2026-10-31T04:54:44Z` Lock Incident)**:
  - **H61 outcome run used PR #816 immediate retry at db8faaa; fairness fix applies to later runs only; lock policy does not touch any outcome computation** (full SHA `db8faaa158c7f7460186a6b92805309e0f0b2cbe`).
  - Under PR #816 (`immediate_first_retry=True`, `LOCK_POLL_SECONDS=5`), the H61 coordinator (`PID 3178624`, launched at `04:53:18Z` after H57 `null_sigma001/seed15` completed at `04:51:56Z`) acquired `~/jope_plan/qualification.lock` at `04:54:44Z` (`1.4 s` after H59 `PID 2250432` released it), ahead of H57 (`PID 3058000`), which had been waiting since `04:51:57Z` on a `30 s` poll.
  - Per overseer directives `t2258` and `t2303`, neither H61 (`PID 3178624`), H57 (`PID 3058000`), nor H59 (`PID 2250432`) was killed or paused mid-wave (lock wait time for H57/H59 is excluded from their GPU cell-wall caps and bounded by their separate `3 h` lock-wait limit, making this a scheduling delay rather than an infra-void).
  - See [addendum-lock-fairness-t2258.md](addendum-lock-fairness-t2258.md) for the full FIFO ticket queue (`~/jope_plan/qualification.queue/<epoch_ns>-<pid>-<wave>`) and `>= 45 s` legacy-waiter backoff specification landed for subsequent runs.
