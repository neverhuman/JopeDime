# H438 — why the 104M in-context model lags the 24M model: `OPTIMISATION_UNSEPARATED`, on the threshold (closeout)

**Verdict: `OPTIMISATION_UNSEPARATED`. No explanation is clearly favoured; this is the "nothing separates them"
outcome, and a close one.** At matched steps since the recipe switch, the 104M model's residual median is
**0.020 u below** the 24M model's on worker 2 (0.019 and 0.016 on the other two workers); the pre-registered line
for "it is only a step deficit" was 0.020. At matched total steps the gap is zero (−0.001).
**What the stored logs do settle:**
- **Averaging does not set the workers back.** Loading the group average lowers a worker's loss; the workers move
  closer together over the window. Explanation (a), as damage, is rejected. Its cost is elsewhere: 17–19% of wall
  time is spent waiting for the average.
- **The two runs are fed the same mix** and are both at their constant peak learning rate. Explanation (c) shows
  no signature; (b) is rejected in its schedule-position form.
- **Most of the lag is a step deficit.** The three workers together see 1.08 times the 24M run's tasks per hour
  but take 0.47 times its steps.
- **The larger model is not taking smaller steps** (addendum 2, stored checkpoints): relative to their size its
  weights move 2.8 times as far per step as the 24M model's, in a steady direction.

**What is missing:** whether a real per-step deficit of about 0.02 u exists and is growing. The cheapest decisive
data is time: the same read at 104M hours 36 and 42 (no GPU). The pre-registered GPU intervention is queued below.
Stored logs and monitor reads only: no model pass on any device, the live runs were not touched; TRAIN / `FIT-SEL`
only.

Plan: [`PREREG.md`](PREREG.md), [addendum 1](PREREG_ADDENDUM_1.md). Numbers: [`metrics.json`](metrics.json).
Gates: [`gates_h438.sh`](gates_h438.sh), [`gates2_h438.sh`](gates2_h438.sh).

## Facts over the window (104M training hours 12 to 24.5; log copies of 17:42Z)
| | 24M `ICL_M_r3` | 104M, one worker (range of three) | 104M, three workers together |
| --- | --- | --- | --- |
| Optimizer steps per hour | 15,965 | 7,476–7,975 (0.47–0.50×) | the same as one worker |
| Tasks per step | 31.4 | 23.3 | 69.9 |
| Tasks per hour | 501,568 | 174,293–185,529 | 543,695 (1.08×) |
| Steps taken at hour 24 | 422,047 | 185,614–193,635 (0.44–0.46×) | — |
| Tokens per parameter at hour 24 | 10,703 | 978–1,020 | 2,935–3,061 (0.28×) |
| Learning rate in the window | 2.0e-4, constant (peak) | 1.5e-4, constant (peak) | |
| Realised shares: real tables / injected columns / residual tasks; row cap | 0.40 / 0.285 / 0.50; 1,024 | 0.40 / 0.285 / 0.50; 1,024 | |
| Waiting for the group average | — | 50–57 s per round: **17–19% of wall time** | |
| Distance of a worker to the group average | — | 0.026 at the start of the window, 0.021 at the end | |

Training hours exclude the waiting. The 104M group's hour 24 took about 28 hours of wall time.

## N1 — the gap at matched steps (104M minus 24M; in-training evaluations on the 106 monitor tables; mean over 46–47 evaluations)
| Residual median on real tables | Worker 2 | Worker 0 | Worker 1 |
| --- | --- | --- | --- |
| **At matched steps since the recipe switch (the pre-registered clock)** | **−0.020** | −0.019 | −0.016 |
| At matched total steps | −0.001 | −0.006 | +0.004 |
| At matched hours | −0.027 | −0.028 | −0.023 |

- Rule: `STEPS_FAVORED` needs ≥ −0.020 on worker 2 and both replicates. Worker 2 is at −0.0204 (naive standard
  error 0.004; evaluations 20 minutes apart are not independent). **The verdict turns on 0.0004 u.**
- **The matched-step gap widens through the window:** −0.006 in its first third, −0.030 in its last (worker 2;
  −0.009 to −0.030 and −0.006 to −0.020 on the others).
- **Slope over hours 18–24:** the 104M residual median gains 0.001 to 0.003 u per 10,000 steps; the 24M model
  gained 0.004 to 0.006 over the same step range after its switch. The plateau is the 104M model's own.
- **The stored monitor reads agree in size** (four reads per model, added after the readout): at matched steps
  since the switch −0.013 (hour 12), −0.001 (hour 18), −0.013 (hour 24); at matched total steps +0.021 (hour 24);
  at matched hours −0.027, −0.032, −0.070.

Other quantities, worker 2 (matched steps since the switch / matched total steps / matched hours):
| | | | |
| --- | --- | --- | --- |
| Residual win rate against ridge | +0.032 | +0.051 | +0.023 |
| Plain median on real tables | −0.041 | +0.040 | −0.067 |
| Synthetic held-out residual u (N4) | −0.005 | +0.001 | −0.012 |
| Synthetic held-out plain u | −0.028 | −0.012 | −0.040 |
| Held-out synthetic loss (lower is better) | −0.040 | −0.067 | −0.015 |

## N2 — loading the group average (training loss by minute after a sync, minus the last minute before it; 187–188 rounds)
| Minute after the sync | Worker 2 | Worker 0 | Worker 1 | 24M control (5-minute blocks) |
| --- | --- | --- | --- | --- |
| 1 | −0.012 [−0.032, +0.008] | −0.011 [−0.029, +0.007] | −0.004 [−0.024, +0.015] | +0.002 [−0.007, +0.010] |
| 2 | **−0.026 [−0.037, −0.014]** | **−0.029 [−0.039, −0.018]** | −0.010 [−0.021, +0.002] | +0.006 [−0.004, +0.015] |
| 3 | −0.010 [−0.022, +0.002] | −0.011 [−0.021, −0.001] | −0.002 [−0.013, +0.010] | 0.000 |
| 4 | −0.010 [−0.021, +0.001] | −0.010 [−0.021, +0.001] | +0.002 [−0.008, +0.012] | −0.002 |
| 5 | +0.003 [−0.012, +0.018] | +0.003 [−0.011, +0.017] | −0.001 [−0.016, +0.015] | 0.000 |

Rule for (a): a rise of at least 0.01 with its interval above 0. Found: no rise; a dip.

## What the result does and does not say
- **(a) Averaging as damage: rejected.** The average of three workers is a better point than a worker's own: the
  loss drops after loading it and climbs back to the worker's level within the five-minute round. The workers do
  not drift apart (0.026 to 0.021).
- **Averaging as cost: real.** A sixth to a fifth of wall time goes to waiting. And three GPUs buy three times the
  data per step, not more steps: the group moves at one worker's step rate.
- **(c) Mix: no signature.** Identical realised shares; on synthetic held-out residual tasks the two models are
  level (−0.005).
- **(b) Schedule position: rejected.** Both runs are past warm-up and far from decay. Whether 1.5e-4 is the right
  peak for 104M cannot be read from logs.
- **(d) Steps: most of it, not clearly all.** Matching on total steps removes the gap. Matching on steps since the
  switch leaves 0.02 u, and that remainder grows over the window.
- **A lead, not a finding:** the 104M model has the lower held-out synthetic loss at every matching, even at
  matched hours, and wins against ridge on more tables, yet its median utility is not ahead. It may be learning
  the training objective faster than that objective pays on real tables.
- **Not shown:** anything about the stack value of the 104M model; anything causal (one run per size, differing in
  size, learning rate, token and row budgets, hardware and averaging at once).

## Addendum 2 — how far the weights move (stored checkpoint copies, CPU, no forward pass; added after the closeout; the verdict is unchanged)
Relative displacement ‖θ_b − θ_a‖ ÷ ‖θ_a‖ of the raw weights (EMA weights within 1%):
| | 24M | 104M worker 2 | 104M ÷ 24M |
| --- | --- | --- | --- |
| Optimizer steps, hours 12→18 / 18→24 | 95,384 / 96,086 | 45,519 / 43,395 | |
| Displacement, 12→18 / 18→24 | 0.295 / 0.238 | 0.358 / 0.303 | |
| Per 10,000 steps, 18→24 | 0.0247 | 0.0698 | **2.8×** |
| Per square root of 10,000 steps, 18→24 | | | 1.9× |
| Per training hour, 18→24 | 0.040 | 0.051 | 1.3× |
| Cosine between the two successive displacements | 0.51 | 0.64 | |

- **Label by the addendum's fixed rule: `MOVES_MORE_PER_STEP`** (ratio 2.8, line 2).
- **The larger model is not taking smaller steps.** Relative to its size it moves almost three times as far per
  step as the 24M model, and further per hour, at a lower peak rate. The idea that its updates are too small or
  cancel in the average (advisor note A-006 expected 3–5× less movement) is not supported.
- **The movement is directed, not a random walk** (cosine 0.64, higher than the 24M model's 0.51), so the addendum's
  own gloss for this label ("would point to noise") is not supported either.
- **Read with the rest:** in hours 18–24 the 104M model moves a lot, in a steady direction, lowers its held-out
  synthetic loss, and its real-table residual median stays flat. That fits the lead above (it is learning its
  training objective; the gain is not showing on real tables in this stretch) better than it fits a learning-rate
  problem. It is still one pair of runs and a label, not a test.
- Limits: the two models are at different step counts (models move less later on); the displacement includes any
  change in weight norm, which was not separated; one worker of three.
- Gate (`gates3_h438.sh`, before the read): 4 of 4. Cost: 83 CPU-seconds, two threads. Addendum
  [2](PREREG_ADDENDUM_2.md), `6491869e01`. These files are not in the S3 bundle (made before the addendum).

## What data is missing
1. **Time (no GPU).** The 24M model rose from 1.006 to 1.047 between 192,000 and 288,000 steps after its switch.
   The 104M model reaches that range at its training hours 30 to 43 (about 16:00Z on Nov 12), and the 24M model's
   hour-24 total step count at hour 56 (about 07:00Z on Nov 13). If it repeats the rise, the lag is steps; if it
   stays flat, there is a per-step deficit. The 24M curve is already stored.
2. **A single-worker reference.** Nothing stored says whether three averaged workers move faster per step than one.
3. **A learning-rate contrast** for the larger model.

## Queued (none started; each needs an order)
- **A. Re-read, no GPU, minutes:** this wave's code on fresh log copies when the 104M model passes training hours
  36 and 43, over the new hours only, with this rule proposed for its PREREG: gap at matched steps since the switch
  ≥ −0.02 on all three workers → steps; ≤ −0.04 on all three → a per-step deficit; otherwise still open.
- **B. `AVG-PROXY-24M`, pre-registered in the PREREG (section 6), at most 9 GPU-hours on xbabe4 after its control
  arms end:** from the stored 24M hour-24 checkpoint, one worker for 3 hours against two workers averaged every
  300 s; compared at matched worker steps on the residual median and the training loss. Two averaged workers ahead
  by ≥ 0.010 u → the group design earns its GPUs and the 104M model needs time. Not ahead → the extra workers add
  nothing per step, and the recommendation to the owner is to release two of the three 3090s.
- **C. A peak learning-rate contrast** would need its own PREREG; not written. Addendum 2 lowers its priority: the
  larger model is not moving too little.
- Unchanged: the training mix for wide tables (prompt B of H427, held for a GPU) and the rule-against-gain read
  (needs AUDIT values and an order). With Jepson: the H433 read-only apply, `CONFIRM`, follow-up 1 of H422.

## The three-snapshot idea of `agy_apex_w10` (A-004)
It did not fit without passes, as the PREREG said: the before-sync weights of the three workers are not stored and
scoring them needs monitor passes of a 104M model. Its loss counterpart is N2, and it answers the question the
idea was after: the group average is the better point, so weight averaging is not what holds the model back.

## Integrity gates
- First attempt (`gates_h438.sh`, 4 of 4, run before the read): the analysis then **stopped with a code error in
  its first function, before computing or printing anything** (the distance field of a sync record is a list).
  No result file existed and the text output was empty.
- **Addendum 1** (`a89a72ed93`): the one-function fix, published before any result. `gates2_h438.sh`: 6 of 6 (the
  four gates again; the addendum published with no result in existence; 5 code lines changed).
- Gates cover: PREREG (`09f0e9d4c0`) published and equal to the staged copy; no result; the eight log files
  intact against their checksums and copied before the PREREG; the sandbox view still without a `CONFIRM` table;
  published stage.

## Deviations, negatives and corrections
1. **The watcher's success condition is not met:** no explanation is clearly favoured. The verdict sits on its
   threshold, and I report it as the rule gives it rather than as "steps".
2. **My expectation was `STEPS_FAVORED`** with a gap between −0.02 and +0.01. Found −0.020, at the edge, and
   growing. The loss dip after a sync and the 17–19% waiting were not expected.
3. **The code error and addendum above.** A made-up log with the real record layout was used to test the fix
   before the second run.
4. **Two clocks disagree** (−0.020 against −0.001). The PREREG named the first as the rule and the weakness of
   both; I do not choose between them after the fact.
5. The monitor cross-check was computed after the readout (`H438.monitor_crosscheck.posthoc.json`); descriptive.
6. Hour-matched, the in-training gap in this window (−0.027) is smaller than the hour-24 monitor read (−0.070):
   the 24M model's late rise shows more strongly in the monitor's protocol.
7. **Addendum 2 was added after the closeout** (stored checkpoints were in the order's scope and I had left them
   out; advisor note A-006 prompted it). Its tokens-per-parameter matching was not adopted: at that point the 24M
   run had not yet switched recipe.
8. Cost: one thread at the lowest priority, about one second. Log files of the live runs were copied once,
   read-only; nothing was written on their hosts.

## Timeline (UTC, 2026-11-11)
| Time | Event |
| --- | --- |
| 17:40 | H438 claimed (no other session holds it) |
| 17:42 | logs copied read-only from xbabe0, xbabe1, xbabe3 and locally; checksums written |
| 17:45 | PREREG and code published (`09f0e9d4c0`); gates 4 of 4; first run stops on a code error, no result |
| 17:48 | addendum 1 and fixed code published (`a89a72ed93`); gates 6 of 6; the read: `OPTIMISATION_UNSEPARATED` |
| 17:50 | closeout published (`cb1595161f`) and archived |
| 17:57 | addendum 2 published (`6491869e01`); gate 4 of 4; checkpoint displacement read: `MOVES_MORE_PER_STEP` |

Evidence (xbabe2): `~/jope_plan/evidence/H438/` (`results/H438.lag.{json,txt}`, both gate files,
`results/H438.monitor_crosscheck.posthoc.json`, `pulled_logs/` with `SHA256SUMS`, `logs/`).
