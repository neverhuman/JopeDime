# H235 — heteroscedastic beta-NLL head for the r882 adapter continuation, xbabe2 CPU

**Status: preregistered; not yet run.** It runs only after this prereg merges. See [PREREG.md](PREREG.md).

- **Question.** If noisy contexts dominate the regression loss, does a learned per-candidate variance, trained with
  beta-NLL (Seitzer et al. 2022, β = 0.5, stop-gradient on σ^(2β)), lift DL-Val91 utility over r882 by more than
  +0.002 at the incumbent's 40-update budget?
- **Head.** One `Linear(32, 1)` log-variance head on the adapter's per-candidate query embedding, initialized at
  zero (σ² = 1) and clamped to [−6, 6]. Scoring uses the mean output only, so update-0 predictions are r882's.
- **Arms.** Identical H170 CTRL AdamW for all.
  - CONTROL uses the CTRL loss.
  - **BNLL_MIX** is the primary: CTRL's smooth-L1 regression term is replaced by beta-NLL and the KL term is kept.
    The fallback was not needed, because CTRL does have a (Huber) regression term.
  - BNLL_ADD is the secondary: CTRL + 0.5 · beta-NLL. Holm m = 2.
- **Machinery.** Merged H234/H232 with the A1/A2 fixes. Fresh per-seed family folds 235141–235143, seeds
  235101–235103, 15 cells, 1,800 updates. Read-only r882 copy, with DOPE `c51a56d0` frozen.
- **Decision.** ADVANCE per arm iff, on DL-Val91, the median seed gain against r882 is > +0.002, the family LCB95 is
  > 0 and Holm p < 0.05. Elo is reported against r882 and CONTROL, and each arm is also compared descriptively with
  the plateau best 1.1389572 (r3175).
- **Host.** xbabe2 CPU, taskset cores 40–43 (avoiding H231 and H233), nice 19. Load5 > 60 pauses the worker and it
  resumes at ≤ 55.
- **Hard stops** (own process group only). RSS ≥ 4 GiB, MemAvailable < 20 GiB, free disk < 300 GB, or the compute
  stop at 05:00Z. Single attempt.
- **Code.** [runner.py](runner.py), [treatment.py](treatment.py) (head and beta-NLL), [analysis.py](analysis.py),
  [monitor.py](monitor.py) and [test_h235.py](test_h235.py) (17 data-free tests).
