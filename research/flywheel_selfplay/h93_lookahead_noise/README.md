# H93: Lookahead, annealed gradient noise, and effective batch 64

**Closeout: INCOMPLETE; no arm admitted.** The registered panel did not produce enough paired outcomes to evaluate any candidate. This is an infrastructure stop, not evidence for or against Lookahead.

## Hypothesis and setup

H93 tests whether Lookahead synchronization, annealed gradient noise, or an effective batch of 64 improves candidate quality over the unchanged H82 CONTROL. The frozen [preregistration](PREREG.md) specifies fresh r882 copies, adapter-only updates, 40 optimizer updates per outcome cell, seeds 30–32, and family-grouped development readout. DOPE remains frozen at `c51a56d0`; the only permitted evaluation data are the preregistered DL-Train and DL-Val sets. No holdout, sealed, outer-test, or live-loop data are part of the study.

## Outcome

The EB4/seed30 timing-only run completed and was discarded as preregistered. CONTROL/seed30 produced one complete outcome. LA/seed30 attempt 1 stopped before scoring at the filesystem launch gate (94.50% used). The single authorized continuation, attempt 2, also stopped before scoring: its attempt reused the shared `scratch/r882_adapter_copy.pt`, which attempt 1 had left read-only. Both failures and their wall/wait charges remain in the external run ledger; no H93 scratch evidence was deleted or changed.

Only one of twelve paired outcome cells completed. The required family-bootstrap comparisons and Elo readouts are therefore **not estimable**. No arm is admitted, and the Lookahead hypothesis remains unresolved. See the compact [metrics record](metrics.json); it contains attempt accounting only, not fabricated outcome statistics.

## Repair

The CPU runner now stages every physical attempt in a unique directory under `scratch/`, with timing attempts separately named. A synthetic regression test keeps a pre-existing read-only legacy copy in place while staging a later attempt successfully. This repairs the collision for future registered use; it does not authorize another H93 attempt. Any Lookahead retest requires a new preregistration and FIFO authorization.
