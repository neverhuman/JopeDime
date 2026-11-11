# H419 — APEX-ICL, our own in-context tabular regressor (in progress)

**This is a status log, not a closeout.** H419 itself has made no prediction on a test cohort (stored P201b
matrices, APEX-DEV, CONFIRM). The hour-20 and hour-24 checkpoints of `ICL_M_r3` were read on APEX-DEV by H424 under
its own PREREG (see the hour-24 block below). `CONFIRM` is untouched. Plan: [`PREREG.md`](PREREG.md) and addenda [1](PREREG_ADDENDUM_1.md), [2](PREREG_ADDENDUM_2.md),
[3](PREREG_ADDENDUM_3.md), [4](PREREG_ADDENDUM_4.md), [5](PREREG_ADDENDUM_5.md), [6](PREREG_ADDENDUM_6.md).

## Runs
| Run | Host | Size | State |
| --- | --- | --- | --- |
| `ICL_S_r2`, `ICL_M_r2` | xbabe3 3090, xbabe1 4090 | 7.56M, 23.66M | synthetic prior only; stopped at about 1.3 h; their checkpoints start r3 |
| `ICL_S_r3` | xbabe3 3090 | 7.56M | stopped at 0.77 h (11:38Z) for the 104M run (addendum 2); checkpoint kept |
| `ICL_M_r3` | xbabe1 4090 | 23.66M | 46 h run, started 2026-11-10T10:52Z; same data mix |
| `ICL_L_r4` | xbabe3 3090 | 104.36M | started 2026-11-10T11:40Z from scratch (addendum 2); stopped at its 13:00:43Z checkpoint for the cut-over; checkpoint kept |
| `ICL_L_g1_w0` | xbabe3 3090 | 104.36M | averaging worker since 13:01:55Z, cloned from the last `ICL_L_r4` checkpoint (addendum 3) |
| `ICL_L_g1_w1`, `ICL_L_g1_w2` | xbabe0 3090, xbabe2 3090 | 104.36M | started 12:29Z as averaging workers cloned from `ICL_L_r4` at step 5,531 (addendum 3) |
| `ICL_M_x4b_inj`, `ICL_M_x4b_noinj` | xbabe4 Apple GPU | 23.66M | column-injection ablation, two arms, stopped at 24 h each (addendum 4); **both finished 15:27Z Nov 11; reading `NO_EVIDENCE`** |
| `ICL_M_x4c_res` | xbabe4 Apple GPU | 23.66M | control for residual-target training: the `inj` recipe plus `--p-resid 0.5`, started 16:36Z (addendum 5); parked after the 20:25Z incident at 3.4 h; **resumed 15:41Z Nov 11** under the memory supervisor |
| `ICL_M_x4d_v2` | xbabe4 Apple GPU | 23.66M | control for prior version 2: the `res` recipe plus `--prior v2`, running since 18:08:44Z (addendum 6; the 17:07Z start failed, see below); parked at 2.0 h; **resumed 15:41Z Nov 11** under the memory supervisor |

Since addendum 6 the 24M run `ICL_M_r3` (from 17:15Z) and 104M workers 2 (17:14Z) and 1 (17:53Z) train with
`--prior v2 --p-resid 0.5 --row-floor 1024`; worker 0 followed at 18:25Z (gates passed 18:44Z).

## Tracked metrics
- **Real-table monitor:** 106 core FIT-SEL contexts, SEARCH rows only; query = fold 0 of `kfold_labels`, context =
  folds 1–4. u = −ln(RMSE / RMSE of the context-mean predictor). Ridge reference on the same rows: mean u 1.061,
  median u 0.630; ridge is near-exact (u > 3) on 7.5% of the contexts, which is why the mean is not the target.
- **Synthetic held-out tasks**, with and without injected columns, against ridge.

| Time (UTC) | Run, hours into r3 | Real win rate vs ridge | Real median u (ridge 0.630) | Real mean u (ridge 1.061) | Synthetic u (ridge) | Synthetic with injection (ridge) |
| --- | --- | --- | --- | --- | --- | --- |
| 10:50 | S, r2 checkpoint (smoke) | 0.462 | 0.585 | 0.784 | 0.247 (0.149) | 0.236 (0.109) |
| 11:13 | S, 0.33 h | 0.481 | 0.526 | 0.800 | 0.266 (0.149) | 0.241 (0.109) |
| 11:33 | S, 0.67 h | 0.491 | 0.597 | 0.818 | 0.277 (0.149) | 0.272 (0.109) |
| 11:13 | M, 0.33 h | 0.547 | 0.643 | 0.931 | 0.292 (0.132) | 0.265 (0.100) |
| 11:33 | M, 0.67 h | 0.575 | 0.645 | 0.917 | 0.297 (0.132) | 0.270 (0.100) |
| 12:13 | M, 1.33 h | 0.585 | 0.633 | 0.940 | 0.300 (0.132) | — |
| 12:01 | L (104M), 0.33 h | 0.019 | 0.013 | 0.033 | 0.007 | — |
| 12:53 | L, 0.98 h (worker 1) | 0.264 | 0.361 | 0.541 | 0.124 (0.149) | 0.123 (0.109) |
| 13:30 | M, 2.67 h | 0.585 | 0.663 | 0.899 | 0.365 (0.132) | 0.288 (0.100) |
| 14:30 | M, 3.66 h | 0.594 | 0.687 | 1.023 | 0.372 (0.132) | 0.299 (0.100) |
| 15:30 | M, 4.66 h | 0.613 | 0.675 | 1.013 | 0.373 (0.132) | 0.295 (0.100) |
| 13:35 | L group, 1.63 h (worker 0) | 0.406 | 0.474 | 0.703 | 0.161 (0.149) | 0.169 (0.109) |
| 14:30 | L group, 2.52 h | 0.472 | 0.580 | 0.831 | 0.207 (0.149) | 0.210 (0.109) |
| 15:42 | L group, 3.70 h | 0.538 | 0.608 | 0.884 | 0.249 (0.149) | 0.238 (0.109) |

Clock times of the 13:30–15:42 rows are approximate; the hours column is exact. Worker hours exclude the time a
worker waits for the group average.

**Worker 0 killed by a host out-of-memory event on xbabe3, 04:39:38Z Nov 11; resumed 04:47:44Z.**
- **Cause.** xbabe3 ran out of memory as a whole; the kernel killed several processes, worker 0 among them. The
  worker's own peak was 5.8 GiB of RAM and 1.5 GiB of swap on a 62 GiB host, so it was a victim. The owner's watcher
  names a feature-extraction job of another lane as the cause and has told it to cap its memory.
- **Loss.** One training step. The last checkpoint (04:38:15Z, step 116,058, 14.27 h) was complete.
- **Resume.** Same unit name, code stage (`r9`) and arguments, plus `--resume`; the resume record matches the one
  of 18:23Z apart from step and time. Worker 0 was down 8 minutes 10 seconds.
- **Effect on the group.** Rounds 5981240 and 5981241 averaged workers 1 and 2 only, and each waited the full 100 s
  collection window for the missing worker (about 110 s against 21–49 s before), so those two workers idled longer
  in two rounds. Round 5981242 (04:50:56Z) has three workers again; all three loaded it and train at 48–55 tasks
  per second.
- **Open risk.** Nothing protects the unit if the host runs out of memory again; the remedy is the same resume.

**Hour-12 rules (readings of 23:22Z Nov 10 and 02:04Z Nov 11): both passed.**
- **24M on-track rule (addendum 1):** pass. Training-log evaluations at 11.99 h and 12.33 h: win rate 0.660 and
  0.642 (needs 0.55), median 0.876 and 0.791 (needs 0.643).
- **104M size rule (addendum 5):** pass, the group continues. Worker 2's checkpoint at 12.24 h: residual readout
  win rate 0.925 (needs 0.848), median 0.946 (needs 0.899). Plain readout median 0.762.
- **Stack test on the 24M checkpoint of 12.33 h:** the residual column is used in 49% of contexts with weight
  0.113, for +0.0051 u. Small; no test cohort is involved.
- **Next size check:** hour 24, residual win rate ≥ 0.895 and median ≥ 0.953.

**Hour-18 and hour-24 readings (Nov 11): the 104M size rule passed at hour 24.**
Monitor: 106 `FIT-SEL` contexts, SEARCH rows only; median u (ridge 0.636, best GP program 1.200, GP blend 1.228).
| Checkpoint | Plain readout, median u | Residual readout, median u | Residual readout, win rate against ridge | Blend + model beats the blend |
| --- | --- | --- | --- | --- |
| 24M, 18.32 h | 0.832 | 1.006 | — | 0.179 |
| 24M, 24.32 h | 0.877 | 1.047 | 0.925 | 0.217 |
| 104M worker 2, 18.19 h | 0.795 | 0.973 | 0.972 | 0.170 |
| **104M worker 2, 24.10 h** | 0.811 | **0.976** | **0.934** | 0.170 |

- **104M size rule at hour 24 (addendum 5): pass.** Needs residual win rate ≥ 0.895 and median ≥ 0.953; has 0.934
  and 0.976 (reading of 16:57Z Nov 11, step 186,324). The group continues.
- **The 104M model is still behind the 24M model at the same hours** (0.976 against 1.047 on the residual readout,
  0.811 against 0.877 plain), and its residual median moved little between hours 18 and 24 (0.973 to 0.976). The
  rule compares it with the 24M model of hour 12, so a pass does not say the larger model is ahead.
- **Injection ablation (addendum 4), both arms at 24 h, rule of its section 3** (mean of each arm's last five
  evaluations): with injection, win rate 0.615 and median 0.709; without, 0.642 and 0.734. Difference −0.026 and
  −0.025: **`NO_EVIDENCE`** (the band is ±0.03; the difference leans against injection on real tables). One seed
  per arm. Synthetic held-out u is equal in both arms (0.344 plain, 0.420 with injected columns).
- **xbabe4 second pair:** `ICL_M_x4c_res` and `ICL_M_x4d_v2` resumed at 15:41Z Nov 11 from their own checkpoints
  (stages `r8` and `r9`, the hashes in their `config.json`; stored arguments plus `--resume`), two arms under the
  memory supervisor; both stepping at 15:42Z. Their 24-hour budgets end about 12:45Z and 14:10Z Nov 12.
- **Why the 104M model lags (H438, stored logs only, 17:48Z Nov 11): not clearly separated.** Most of the lag is a
  step deficit: the three workers together see 1.08 times the 24M run's tasks per hour but take 0.47 times its
  steps. At matched steps since the recipe switch a gap of 0.02 u remains and grows; at matched total steps it is
  zero. Averaging does not set the workers back (the loss dips after loading the group average) but 17–19% of wall
  time is spent waiting for it. Both runs are fed the same mix at their constant peak learning rate. A 24M
  averaging proxy (at most 9 GPU-hours) is pre-registered and queued.
- **What the 24M checkpoints did as a stack column is recorded in its own waves:** H424 (`ICL_CONFIRMED` on `DEV`
  with the hour-20 checkpoint; the hour-24 column is not better than the hour-20 one), H425–H427, H428–H429
  (wide tables), H432 (column only up to 30 features) and H434–H435 (a weight rule for screening columns).

**xbabe4 incident, 20:25Z: our four arms exhausted the Mac Studio's memory; it was reset at 20:30Z.**
- **What happened.** macOS logged a memory-exhaustion event at 20:25:08Z with 0.1 GiB free. The four arm processes
  held 178, 161, 138 and 108 GiB (586 GiB together; the machine has 512 GiB), another user's `llama-server` 84 GiB.
  The machine was reset at 20:30Z. Everything running on it stopped, including applications that are not ours.
- **Cause.** On the Apple GPU backend a trainer's memory grows with the number of distinct batch shapes it has
  seen: the buffer pool settles near 94 GiB without a cap (74 GiB with the cap the fourth arm had), and memory
  outside the pool grows by about 10 GiB an hour. Four arms cannot fit for long.
- **My mistakes.** No per-process memory limit; the memory guards of addenda 4–6 were not verified to be running
  (none was); at 20:17Z I read 36% free memory and arms slowed to 2–10 tasks per second and did not act.
- **Checkpoints.** All four arms have complete checkpoints from 20:05–20:17Z: `inj` 7.34 h, `noinj` 7.35 h, `res`
  about 3.5 h, `v2` about 2.1 h. At most 20 minutes of training per arm were lost.
- **Resumed 22:27Z (asked by the owner's watcher at 22:22Z): `ICL_M_x4b_inj` and `ICL_M_x4b_noinj` only**, from their
  own `ckpt_last.pt`, same code stage (`r6`, the hash in their `config.json`) and same arguments plus `--resume`; the
  24-hour budget is carried in the checkpoint. New and operational only: a buffer-pool cap
  (`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.14`, low 0.10) and a supervisor (`x4_supervise_arm.sh`) that restarts a
  trainer from its own checkpoint when its physical footprint reaches 75 GiB (right after a checkpoint) or 95 GiB
  (at once), and stops it if system free memory falls below 25%. Two aborted starts at 22:25Z (the first
  supervisor read a memory figure that does not count GPU buffers) left duplicate log lines for steps
  16,342–16,400 (`inj`) and 16,750–16,810 (`noinj`).
- **Not resumed: `ICL_M_x4c_res` and `ICL_M_x4d_v2`.** Reason: memory; two supervised arms stay under 190 GiB in
  the worst case, four would not leave room for the machine's other users. Their checkpoints are kept and they can
  resume with the same arguments when the first pair has finished. **Consequence for addenda 5 and 6:** the ladder
  arms will not have shared the GPU in the same way, so equal training hours no longer mean equal task counts;
  their readings will be reported at equal task counts as well.

**Prior version 2: cause, rollout and first readings, 17:05–18:12Z (addendum 6).**
- **Cause found:** the version-1 synthetic prior has almost no easy tasks (ridge u above 1 on at most 3% of its
  tasks; on 33% of the real monitor contexts). The 24M model at 5.0 h scored a median u of 0.63 on fresh linear
  tasks against ridge's 1.11. Prior version 2 gives 40% of synthetic tasks to linear, additive and
  additive-with-pairs families.
- **Negative before the change:** as a stack column beside the GP programs the 24M model's out-of-fold predictions
  changed the fold-0 u of the Caruana blend by +0.001 (used in 38% of contexts, mean weight 0.05).
- **Rollout** (`roll_driver2.sh`, each run from its own checkpoint, gates: unit active, arguments logged, an
  averaging round loaded, a new checkpoint, finite loss):

  | Run | Checkpoint it resumed from | Restarted | Gates |
  | --- | --- | --- | --- |
  | `ICL_M_r3` | 17:13:45Z, step 138,685 (6.33 h) | 17:15Z | passed 17:34Z |
  | `ICL_L_g1_w2` | 17:12:35Z, step 40,693 (4.88 h) | 17:14Z | passed 17:34Z |
  | `ICL_L_g1_w1` | 17:51:27Z, step 45,530 | 17:53Z | pending at 18:12Z |
  | `ICL_L_g1_w0` | — | after worker 1 passes | — |

- **First readings after the change** (training log; real = the 106-context monitor; simple = held-out tasks of the
  three new families; too few evaluations to call a trend, single readings move by up to 0.05):

  | Run, hours | Real plain: win vs ridge / median u | Real residual readout: win / median u | Simple tasks: model / ridge | Linear tasks: model / ridge |
  | --- | --- | --- | --- | --- |
  | 24M, 6.33 h (last before the change) | 0.632 / 0.727 | 0.868 / 0.919 (checkpoint diagnostic) | — | — |
  | 24M, 6.66 h | 0.660 / 0.732 | 0.887 / 0.945 | 1.086 / 0.996 | 1.183 / 1.841 |
  | 24M, 7.00 h | 0.679 / 0.782 | 0.868 / 0.980 | 1.149 / 0.996 | 1.255 / 1.841 |
  | 104M worker 2, 4.88 h (addendum-5 arguments) | 0.566 / 0.593 | 0.877 / 0.875 | — | — |
  | 104M worker 2, 5.17 h | 0.557 / 0.617 | 0.887 / 0.872 | 0.758 / 0.927 | 0.890 / 1.725 |
  | 104M worker 2, 5.47 h | 0.566 / 0.620 | 0.906 / 0.874 | 0.791 / 0.927 | 0.999 / 1.725 |

  - The 24M run's plain median (0.782) and residual median (0.980) are its highest so far.
  - **Still far from the references that matter:** the best GP program's median on the same rows is 1.200 and the
    Caruana blend's 1.228. On linear tasks both sizes are still far below ridge.
  - The 104M worker is behind the 24M run on every column, as before.
  - Throughput after the change: 24M about 140 tasks/s (275 before, at 384 rows); the 4090 shows 54–99% utilisation,
    so part of the time it waits for the trainer's main thread. Not yet profiled.
- **Mistake:** the xbabe4 arm `ICL_M_x4d_v2` started at 17:07Z died at once (`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.2`
  without a low-watermark setting is rejected by torch) and I recorded it as running without checking. Found at
  18:07Z; restarted 18:08:44Z with `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.15` added, and verified alive and logging. The
  arm is one hour later than recorded; the comparison is at equal training hours, so nothing else changes.

**Diagnostics and changes, 16:10–16:55Z (addendum 5).**
- **Recipes on the same checkpoints** (`icl_diag.py`; 24M at 5.0 h, 104M at 3.7 h; win rate vs ridge / median u):

  | Recipe | 24M | 104M |
  | --- | --- | --- |
  | Plain | 0.594 / 0.713 | 0.519 / 0.606 |
  | Residual of an in-context ridge (`icl_resid.py`) | 0.868 / 0.919 | 0.868 / 0.857 |
  | Ridge alone | — / 0.636 | |

  The plain models win 0 of 8 contexts above 100 features. 896 context rows are no better than four 320-row
  subsamples. The win rate under the residual readout moves by about ±0.04 with the fold draw.
- **Negative, and the more important number: against our own GP programs on the same rows the model is far behind.**
  The stored H417 matrices give, for the same folds: best cross-validated GP program median u 1.200, Caruana blend
  1.228. The 24M residual readout beats the best program on 0.10 of contexts and the blend on 0.04. Used as a
  corrector of the blend without training for it, blend + model beats the blend on 0.20 (24M) and 0.18 (104M) of
  contexts, mean u 1.765 against 1.774. **Ridge was too weak a reference; the tracked number is now the share of
  contexts where blend + model beats the blend.**
- **Change:** the 104M group moves to `--p-resid 0.5 --row-floor 1024` by rolling restart. Worker 2 restarted at
  16:52:08Z from its 16:51:44Z checkpoint (step 38,510, 4.60 h; a copy is kept). The others follow behind gates
  (`roll_driver.sh`). A 24M control arm with residual-target tasks started on xbabe4 at 16:36Z.
- **Averaging:** successive displacements of the group mean are orthogonal (cosine 0.008, 0.001); outer momentum is
  implemented and off.
- **Correction:** a 104M worker trains at about 97 tasks per second at 384 rows. Rates near 41–47 that I quoted
  elsewhere came from log intervals that include the averaging wait. With rows up to 1,024 worker 2 is at about 48.
- **Mistake:** I tracked the win rate against ridge for five hours before testing inference recipes or a stronger
  reference; both were ten-minute checks.

**Reading at 15:46Z.**
- **24M run (4.66 h):** ahead of ridge on real tables by win rate (0.613) and by median (0.675 against 0.630), and
  within 0.05 of ridge on the mean (1.013 against 1.061). The median has been above its start value (0.643) at 12
  of the 13 evaluations since the first. The on-track rule (hour 12) has not come due; the run is on course for it.
- **104M group (3.7 h per worker, three workers):** rising at every evaluation: win rate 0.26 → 0.54, median
  0.36 → 0.61, mean 0.54 → 0.88 in 2.7 hours. It is still behind the 24M run on every real-table measure and has
  not yet passed ridge's median. The size rule (first check at hour 12) has not come due.
- **Negatives:** on the linear-leaning `scm` synthetic tasks both sizes are still below ridge (24M: 0.267 against
  0.296; 104M: 0.242 against 0.254). The 24M run's win rate moves by up to 0.05 between consecutive evaluations, so
  single readings are noisy. On synthetic tasks the 24M run has gained little in its last two hours (0.365 → 0.373).
- **xbabe4 ablation at 2.67 h of 24:** with injection win rate 0.519, median 0.645, mean 0.926; without injection
  0.528, 0.585, 0.840. The arms cross between evaluations; nothing can be read from this yet.

**Cut-over of the 104M run into the averaging group (13:01:55Z, addendum 3).** `ICL_L_r4` on xbabe3 was stopped
right after its 13:00:43Z checkpoint, and `ICL_L_g1_w0` started from that checkpoint as the third worker. About one
minute of single-run progress was lost. Since then (to 15:56Z): 35 three-worker rounds, every round loaded by every worker,
mean wait 24–33 s of each 300 s round (8–11% of worker time), worker-to-mean distance steady at 0.032 (0.053 for
worker 0 in its first round).

Stop rule (c) (loss after the average more than 0.1 above the loss before it, three rounds in a row) has not
triggered. By the literal comparison 6 of 119 worker-rounds exceeded 0.1, at most two in a row (worker 0, rounds
5981072–5981073). In each of the six, the value before or the value after is an average
over only a few batches (the log window is cut at the average), and the loss after was at most 0.07 above the last
full log line except twice on worker 1 (+0.11 at round 5981061; +0.31 at round 5981086 on a few batches, back to
4.88 at the next full line).
**Weakness:** this comparison is noisy (consecutive full log lines differ by 0.07 on average), so the rule as written
is a coarse alarm, not a measurement.

**Reboots of 12:02Z (xbabe2) and 11:56Z / 12:05Z (xbabe0), not by this wave.** Neither training run was on a
rebooted host. `ICL_M_r3` (step 31,010 at 1.42 h, checkpoint 12:13:10Z) and `ICL_L_r4` (step 5,263 at 0.63 h,
checkpoint 12:00:27Z) kept running; **no training time was lost**. The session and its waiters were lost and
restarted. After the reboots the RTX 3090s of xbabe0 and xbabe2 are usable; addendum 3 puts them on the 104M run.

**Reading at 12:13Z.** The 24M run: win rate 0.585 (up from 0.575), median 0.633 (down from 0.645, now level with
ridge at 0.630), mean 0.940 (ridge 1.061). The 104M run at 0.33 h has not started to predict (win rate 0.02), as a
larger model is expected to at that point; this is neither good nor bad news yet.

**Reading at 11:34Z.**
- The larger model is ahead of ridge on real tables by win rate (0.575) and barely by median (0.645 against 0.630).
- **Negatives:** the smaller model is still behind ridge on real tables on every measure (win rate 0.491, median
  0.597, mean 0.818). Both models are behind ridge on the mean. The larger model's mean fell between the two readings
  (0.931 to 0.917) while its win rate rose. Forty minutes of the new data mix is too short to credit the column
  injection or the real-table tasks with anything; the real-table share was still ramping (0.25–0.29 of batches).
- The on-track rule (addendum 1: win rate ≥ 0.55 and median above its start value by hour 12) has not come due.

## Why xbabe4 runs an ablation and not a fourth 104M worker
- The Apple GPU trains the 104M model at about 0.36 times a 3090 per step (1.4–2.0 s against about 0.5 s). In an
  equal-weight average it would pull the group back each round, so it stays out of the group.
- The owner's direction for this wave is synthetic column injection, and nothing yet measured its contribution.
  xbabe4 therefore runs the published 24M recipe with and without injection, side by side on the same hardware
  (addendum 4). The arm with injection is also the only replicate of the 24M run, so it gives a first noise estimate.
- Limit: at 33–42 tasks per second per arm, 24 hours cover only the first 3 million tasks. A clear early effect
  can show; a small or late one cannot.

## Mistakes and corrections so far
- `ICL_*_r1` were stopped seconds after launch: the largest training shapes had not been tested. The stress test
  then found a CUDA attention limit at 65,536 sequences (fixed by chunking) and an out-of-memory shape for the M size
  on the 4090 (fixed by a row-token cap).
- No learning-rate sweep and no optimizer comparison were run, against the design review's plan.
