# H234 — LambdaLoss (NDCG-weighted pairwise) for the r882 adapter continuation, xbabe1 CPU

**Status: preregistered; not yet run.** It runs only after this prereg merges. See [PREREG.md](PREREG.md).

- **Why not ListNet.** The merged CTRL loss already contains a ListNet top-1 term: a within-context softmax KL at
  temperature 0.2, weight 0.1. Under hw_t2220's rule, both arms switch to LambdaLoss before any outcome.
- **Question.** At the incumbent's 40-update budget, does an NDCG-weighted pairwise loss, which concentrates the
  gradient on pairs that change the top of the ranking, lift DL-Val91 utility over r882 by more than +0.002?
- **Arms.** Identical AdamW for all (the exact merged H170 CTRL recipe); only the loss differs.
  - CONTROL uses the CTRL loss.
  - **LAMBDAMIX** is the primary: CTRL + 0.5 · Lambda.
  - LAMBDA is the secondary: Lambda only. Holm m = 2.
- **Machinery.** Merged H232 with its A1/A2 fixes. Fresh per-seed family folds (fold seeds 234141–234143), seeds
  234101–234103, 15 cells, 1,800 updates. Adapter only, on a read-only r882 copy, with DOPE `c51a56d0` frozen.
- **Decision.** ADVANCE per arm iff, on DL-Val91, the median seed gain against r882 is > +0.002, the family LCB95 is
  > 0 and Holm p < 0.05. Beating CONTROL alone is descriptive. Elo is reported against r882 and CONTROL.
- **Host.** xbabe1 CPU, taskset 4 cores, nice 19. Load5 > 60 pauses the worker and it resumes at ≤ 55.
- **Hard stops** (own process group only). RSS ≥ 4 GiB, MemAvailable < 20 GiB, free disk < 150 GB, or the compute
  stop at 04:30Z. Single attempt.
- **Code.** [runner.py](runner.py), [treatment.py](treatment.py) (LambdaLoss), [analysis.py](analysis.py),
  [monitor.py](monitor.py) and [test_h234.py](test_h234.py) (16 data-free tests).
