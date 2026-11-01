# H180 — adapter-contribution ablation and fresh-init update-budget dose

**Status: preregistered; not yet run.** See [PREREG.md](PREREG.md).

- **Question.** Does the trained adapter add utility beyond the frozen DOPE anchor path, and does a 4× larger
  update budget help a freshly initialized adapter?
- **Arms.**
  - R882: the reference.
  - ZERO_ADAPTER: the anchor-only scorer, proved to equal the live `total_cost − adapter_total_delta` bit for bit.
  - RAND0: a fresh init with no updates.
  - FRESH40 and FRESH160: one fresh trajectory, scored at 40 and 160 updates.
- **Harness.** H176's family-grouped harness (fit on 123 families, B unused, assess on 41) with seeds
  180101–180103. Results are pooled across the A folds.
- **Reported.** Median-seed gain against R882 with a 10,000-draw paired family bootstrap (axis 0), Elo against
  r882, −2.0 counts and choice agreement.
- **Verdicts.** `ADAPTER_INERT`, `ADAPTER_MATTERS` or `ADAPTER_HARMFUL`, and `BUDGET_HELPS`, `BUDGET_FLAT` or
  `BUDGET_HURTS`, with Holm m = 4 for any ADVANCE_CANDIDATE (never adopted).
- **Code.**
  - [runner.py](runner.py) and [treatment.py](treatment.py): the H176 engine plus the anchor-only scorer.
  - [analysis.py](analysis.py).
  - [monitor.py](monitor.py).
  - [test_h180.py](test_h180.py): 11 data-free tests.
