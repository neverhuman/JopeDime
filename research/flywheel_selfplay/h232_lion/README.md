# H232 — Lion optimizer for the r882 adapter continuation, xbabe1 CPU

**Status: preregistered; not yet run.** It runs only after this prereg merges. See [PREREG.md](PREREG.md).

- **Question.** At the incumbent's 40-update budget, does Lion (Chen et al. 2023; sign of interpolated momentum,
  decoupled weight decay) find adapter updates that lift DL-Val91 utility over r882 by more than +0.002?
- **Arms.**
  - CONTROL is the exact merged H170 CTRL AdamW recipe (lr 1e-4, wd 1e-4).
  - **LION** is the primary: lr 1e-5, betas (0.9, 0.99), decoupled wd 3e-4.
  - LION_LR3 is the secondary: lr 3.33e-5. Holm m = 2 over LION and LION_LR3.
- **Machinery.** Merged H170, unchanged except the optimizer: 205 families in 5 family folds (fold seed 232140),
  seeds 232101–232103, 15 cells, 1,800 updates. Adapter only, on a read-only r882 copy, with DOPE `c51a56d0` frozen.
- **Decision.** ADVANCE per arm iff, on DL-Val91, the median seed gain against r882 is > +0.002, the family LCB95 is
  > 0 and Holm p < 0.05. Ties are no advance. Elo against r882 and against CONTROL is reported, with no AUC and no
  speed verdict.
- **Host.** xbabe1 CPU, taskset 4 cores, nice 19. Load5 > 60 pauses the worker and it resumes at ≤ 55.
- **Hard stops.** RSS ≥ 4 GiB, MemAvailable < 20 GiB, free disk < 150 GB, or the compute stop at
  2026-11-03T04:00:00Z. Single attempt.
- **Code.** [runner.py](runner.py), [treatment.py](treatment.py) (Lion), [analysis.py](analysis.py),
  [monitor.py](monitor.py) and [test_h232.py](test_h232.py) (14 data-free tests).
