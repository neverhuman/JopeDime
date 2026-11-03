# H237 — AdaBelief (and RAdam) for the r882 adapter continuation, xbabe2 CPU (A1)

**Status: preregistered; not yet run.** It runs only after this prereg merges. See [PREREG.md](PREREG.md).

- **Question.** Is the plateau partly an adaptive-step-size artefact? AdaBelief (Zhuang et al. 2020) scales steps by
  the gradient's variance around its EMA instead of the raw second moment. At the same 40-update budget, does that
  lift DL-Val91 utility over r882 by more than +0.002?
- **Arms.** Only the optimizer differs; the CTRL loss, batches, lr 1e-4, betas (0.9, 0.999) and decoupled wd 1e-4
  are the same.
  - CONTROL uses AdamW (exact H170 CTRL).
  - **ABEL** is the primary: AdaBelief, inline, with ε = 1e-16 inside s, bias correction and no rectification.
  - RADAM is the secondary: `torch.optim.RAdam` with decoupled wd. Holm m = 2.
- **Machinery.** Merged H234/H232 with the A1/A2 fixes. Fresh per-seed family folds 237141–237143, seeds
  237101–237103, 15 cells, 1,800 updates. Read-only r882 copy, with DOPE `c51a56d0` frozen.
- **Decision.** ADVANCE per arm iff, on DL-Val91, the median seed gain against r882 is > +0.002, the family LCB95 is
  > 0 and Holm p < 0.05. Beating CONTROL alone is descriptive. Elo is reported against r882 and CONTROL, and each arm
  is also compared with the plateau best 1.1389572 (r3175).
- **Host (amendment A1, hw_t0036).** xbabe2 CPU, taskset cores 48–51, nice 19. It moved from xbabe1 because xbabe1
  MemAvailable (16–17 GiB) was below the 22 GiB gate.
- **Pauses** (own process group only). Load5 > 108 or MemAvailable < 20 GiB pauses the worker; it resumes at
  load5 ≤ 100 and MemAvailable ≥ 22 GiB.
- **Hard stops.** MemAvailable < 12 GiB, RSS ≥ 4 GiB, free disk < 300 GB, or the compute stop at 06:30Z. Single
  attempt.
- **Code.** [runner.py](runner.py), [treatment.py](treatment.py) (AdaBelief, RAdam), [analysis.py](analysis.py),
  [monitor.py](monitor.py) and [test_h237.py](test_h237.py) (15 data-free tests).
