# H184 — which part of the adapter's structure carries the training-free lift?

**Status: preregistered; not yet run.** See [PREREG.md](PREREG.md).

- **Question.** H180 showed that a random, untrained adapter recovers the whole adapter benefit (+0.060 over the
  anchor alone). H184 measures how much of that each output term carries, all at 0 updates:
  - **PRIOR**: the multimodal prior correction;
  - **KERNEL**: the revealed-residual kernel regression;
  - **ATTN**: the trajectory cross-attention.
- **Arms.** R882, ZERO_ADAPTER, RAND0, and RAND0 with each term dropped. The terms are proved bit for bit to sum to
  the live adapter delta.
- **Harness.** The H180 harness with seeds 184101–184103 and fold seed 184140.
- **Reported.** Gains against R882 with a 10,000-draw paired family bootstrap (axis 0), Elo, −2.0 counts and
  choice agreement.
- **Decisions.** Attribution `CARRIES_LIFT`, `NEUTRAL` or `HURTS` with Holm m = 3, and the strict Holm m = 4
  ADVANCE gate (never adopted).
- **Code.**
  - [runner.py](runner.py) and [treatment.py](treatment.py), which contains the decomposition.
  - [analysis.py](analysis.py).
  - [monitor.py](monitor.py).
  - [test_h184.py](test_h184.py): 10 data-free tests.
