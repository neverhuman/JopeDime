# H438 PREREG — why the 104M in-context model lags the 24M model: four rival explanations, settled from stored logs

**Status.** Published on `origin/v1/next` before any output of this wave exists. Claim:
`~/jope_plan/ns/receipts/H438_claim.txt` (2026-11-11T17:40Z); no other session holds H438.

**Authority.** Relayed by ENG:watcher on 2026-11-11, after the H437 closeout: back to the H419 in-context model;
why the 104M model lags the 24M model (residual readout 0.976 against 1.047 at hour 24, flat since hour 18), from
stored training logs, checkpoints and monitor reads only; no new GPU passes, the live runs are not touched;
3–4 pre-registered rival explanations, each with the number that would settle it; the 24M run at matched hours as
control; TRAIN / `FIT-SEL` only; CPU at most 4 threads, at most 45 minutes; success = one explanation clearly
favoured plus a concrete, bounded, pre-registered GPU intervention queued for a free GPU; failure = nothing
separates them, and then say what data is missing; negative findings recorded; fold in `agy_apex_w10`'s
three-snapshot averaging idea if it fits under (a) without GPU passes.

**Closed.** dev91-audit-v1, DL-Val, DL-Holdout and sealed data are never read. `CONFIRM` and `DEV` are not read.
No AUDIT array. No model is loaded and no pass is made, on GPU or CPU. The live runs are not touched: their log
files were copied once, read-only.

## 1. Inputs (pinned)
- **Log copies** taken at 17:42Z (`~/jope_plan/evidence/H438/pulled_logs/`, `SHA256SUMS` beside them): `ICL_M_r3`
  (24M, xbabe1) and the three workers of `ICL_L_g1` (104M: `w0` xbabe3, `w1` xbabe0, `w2` xbabe2), with their
  `config.json`. Each log holds a line a minute (step, loss, learning rate, tasks seen, task shares), an evaluation
  every 20 minutes on EMA weights (synthetic held-out tasks; the 106 `FIT-SEL` monitor tables, plain and residual
  readout) and, for the workers, one record per averaging round (distance to the group average, workers in the
  average, wait time).
- **Stored monitor reads** (`~/jope-data/apex-icl/diag/mon/`): 24M and 104M worker 2 at hours 8, 12, 18, 24.
- No checkpoint is opened: the averaging directory keeps only the latest round (live files), so a stored
  before / after / ensemble triple does not exist.

## 2. Design facts (from the configurations, not results)
| | 24M `ICL_M_r3` | 104M `ICL_L_g1` (each worker) |
| --- | --- | --- |
| Parameters | 23.66M | 104.36M |
| Peak learning rate; schedule length | 2.0e-4; 46 h | 1.5e-4; 72 h |
| Warm-up ends; decay starts (2% and 85% of the schedule) | 0.9 h; 39.1 h | 1.4 h; 61.2 h |
| Token budget per step; cap on batch × rows | 600,000; 40,000 | 550,000; 22,000 |
| Averaging | none | 3 workers, every 300 s |
| Switched to the current recipe (residual-target tasks, prior v2) at | step 138,685 (6.33 h) | `w2` step 40,692 (4.88 h); `w1` 45,598; `w0` 51,167 |

Clocks: **h** = training hours; **step** = optimizer steps of the process; **τ** = steps since the process took the
current recipe. The residual readout is only trained after that switch, so τ is its natural clock.
**Window W: 104M hours 12 to 24.5**, the stretch that leads to the hour-24 reading.

## 3. The four explanations and the number that settles each
| | Explanation | Settling number (worker 2; workers 0 and 1 as replicates) | It is favoured if |
| --- | --- | --- | --- |
| (d) | **Too few steps and tokens per parameter by hour 24.** The 104M model learns per step about as the 24M model does; it has simply taken fewer steps | **N1** = mean over the evaluations in W of (104M residual median on real tables − the 24M value interpolated at the same τ) | N1 ≥ −0.02 u on worker 2 and on both replicates |
| (a) | **Averaging interval or drift.** Loading the group average sets a worker back, or the workers drift apart | **N2** = training loss in the first minute after loading the average minus the last minute before it, mean over the rounds in W with a 95% interval (the 24M log cut into 5-minute blocks is the control); also the distance to the average over time and the share of wall time spent waiting | N1 < −0.02 **and** N2 ≥ +0.01 with its interval above 0 |
| (c) | **The training mix under-feeds the larger model.** It fits the synthetic prior at least as well but that does not reach real tables | **N4** = the same gap as N1 on the synthetic held-out residual u; the realised task shares of both runs | N1 < −0.02 **and** N4 ≥ +0.02 |
| (b) | **Learning-rate schedule or warm-up scaled wrong.** | **N3** = learning rate in W against each run's peak, and the schedule positions above | Stored data can only reject the schedule-position form (both runs at their constant peak in W). A wrong peak value shows only as "behind at matched steps with no other signature" |

## 4. Verdict
| Verdict | Condition |
| --- | --- |
| `STEPS_FAVORED` | N1 ≥ −0.02 on worker 2 and on both replicates |
| `AVERAGING_FAVORED` | N1 < −0.02, the N2 rule met, the N4 rule not met |
| `MIX_FAVORED` | N1 < −0.02, the N4 rule met, the N2 rule not met |
| `OPTIMISATION_UNSEPARATED` | N1 < −0.02, neither rule met: a per-step deficit that stored data cannot assign (peak learning rate, capacity, or averaging without a loss bump) |
| `NOT_SEPARATED` | anything else (replicates disagree; both rules met; N1 not computable) |

- Success in the watcher's sense: one of the first three. For the last two the README states what data is missing.
- `STEPS_FAVORED` does **not** say the three-worker group is efficient. It says the lag is a step deficit. Whether
  three averaged workers move faster per step than one cannot be read from these logs (no single-worker 104M run
  exists beyond its first 80 minutes); that is what the queued intervention measures.

## 5. Reported beside the verdict
- The same gaps at matched total steps and at matched hours; plain readout, win rates, synthetic plain u,
  evaluation loss.
- Step rate, tasks per step and per hour, tokens per parameter at hour 24 (one process and all workers), and the
  hour at which the 104M model would reach the 24M model's hour-24 step count.
- "Flat since hour 18": the slope of the residual median per 10,000 steps for the 104M model over hours 18–24 and
  for the 24M model over the same τ range.
- The loss profile minute by minute after a sync; distance to the average (start and end of W); wait time.
- The stored monitor reads at their step counts.

## 6. The queued GPU intervention (pre-registered here; not run in this wave; needs a free GPU and an order)
**`AVG-PROXY-24M`, on xbabe4 after the control arms end (two slots), at most 9 GPU-hours, no live run touched.**
All arms continue the stored 24M checkpoint `ICL_M_r3/ckpt_h024.pt` with its own recipe, under the memory supervisor.
- Arm P1: one worker, 3 training hours. Arm P2: two workers averaged every 300 s, 3 training hours each.
  Run P2 first (two slots), then P1.
- Reading: residual median on the monitor tables and training loss, averaged over each arm's last hour, compared
  **at matched worker steps**.
- If the verdict is `STEPS_FAVORED` or `OPTIMISATION_UNSEPARATED`: P2 − P1 ≥ +0.010 u → averaged workers add
  per-step progress, the group design is doing its job and the 104M model needs time; P2 − P1 ≤ 0 → the extra
  workers add nothing per step, and the recommendation to the owner is to release two of the three 3090s.
- If `AVERAGING_FAVORED`: the same two arms plus P3 = two workers averaged every 75 s (12 GPU-hours in all);
  P3 − P2 ≥ +0.010 u → shorten the interval of the 104M group at its next restart.
- If `MIX_FAVORED`: the intervention is prompt B of H427 (training mix), unchanged.
- Stop rules: an arm with no checkpoint for 40 minutes; the supervisor's memory limits.

## 7. The three-snapshot idea of `agy_apex_w10` (advisor note A-004)
It asks for the monitor score of one worker before a sync, of the group average, and of the three workers'
prediction ensemble. **It does not fit without passes:** the scores need three monitor passes of a 104M model
(about 8 core-hours each on CPU), and the before-sync weights of the three workers are not stored (the exchange
directory holds only the latest round and is live). What stored logs do contain is its training-loss counterpart:
the loss before a sync and after loading the average, for every round. That is N2.

## 8. Budget and gates
- One thread at the lowest priority; seconds. No GPU, no model library.
- Gates (`gates_h438.sh`, run by the driver **before** the analysis because the wave is one read): this PREREG
  published and equal to the staged copy; no result file exists; the log copies match their `SHA256SUMS` and were
  taken before this PREREG was published; the sandbox view still holds no `CONFIRM` table; published stage.

## 9. Expected, stated in advance
**`STEPS_FAVORED`.** Already known: at hour 24 the 104M worker has taken 186,324 steps and the 24M run about
427,000 (0.44×); the 24M model read about 0.98 on the residual median around hour 7 at a similar total step count.
I expect N1 between −0.02 and +0.01, N2 small (under 0.01), N4 near 0. I do not know the replicates, the loss
profile after a sync, or the slopes.

## 10. What was looked at before this PREREG
The H419 README's trajectories and monitor reads (hours 8 to 24 for both sizes); the advisor notes A-002 and A-004,
which quote those; the configurations and resume records of the four logs (the design facts above); the key names
of the log records; a self-test of the code on made-up curves. **No evaluation value, loss or sync record of the
copied logs has been read for this wave.**

## 11. Weaknesses
- **One run per size, not an experiment.** The two runs differ in size, learning rate, token and row budgets,
  hardware and averaging at once. Matching on steps removes throughput; it cannot separate the rest. A verdict
  says which explanation the stored numbers favour, not which is proved.
- The 24M model at a matched τ was earlier in its own training than the 104M model at that τ in total steps
  (138,685 against about 40,000 steps before the switch); both matchings are reported.
- In-training evaluations use EMA weights and are noisy from one to the next; the tolerance (0.02 u) and the
  window are my choices, fixed here. Evaluations 20 minutes apart are not independent.
- `FIT-SEL` monitor tables only; the stack value of the 104M model is not measured anywhere.
- N2 sees damage that shows in the training loss within a round. Slow damage from averaging (blurred features
  that never show as a bump) would be missed; it would appear as `OPTIMISATION_UNSEPARATED` or stay hidden.

## 12. Files
`lag_h438.py`, `gates_h438.sh`, `stage_h438.sh`.
