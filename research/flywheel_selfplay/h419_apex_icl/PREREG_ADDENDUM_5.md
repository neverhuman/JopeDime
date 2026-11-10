# H419 PREREG addendum 5 — ridge-residual tasks and longer rows for the 104M group; residual readout on the monitor

**Status.** Published on `origin/v1/next` before any training process is restarted with the new arguments. No
prediction on a test cohort (stored P201b matrices, APEX-DEV, CONFIRM) exists. Everything below uses the TRAIN-side
monitor (106 FIT-SEL contexts, SEARCH rows only) and synthetic tasks.

**Reason.** Owner instructions, 2026-11-10 about 16:10Z: get the win rates up; make the larger model work.

## 1. What a diagnostic on the current checkpoints found (16:10–16:35Z; no retraining)
`icl_diag.py` evaluates a checkpoint's EMA weights on the monitor under different inference recipes. Checkpoints:
24M `ICL_M_r3` at step 109,493 (5.0 h); 104M group worker 2 at step 31,019 (3.7 h).

| Recipe | 24M: win rate vs ridge / median u / mean u | 104M: win rate / median u / mean u |
| --- | --- | --- |
| Plain (what the training log reports) | 0.594 / 0.713 / 1.036 | 0.519 / 0.606 / 0.895 |
| 4 context subsamples of 320 rows, averaged | 0.585 / 0.729 / 1.027 | 0.547 / 0.614 / 0.864 |
| 4 feature permutations, averaged | 0.632 / 0.779 / 1.072 | 0.528 / 0.624 / 0.888 |
| Ridge prediction appended as a feature | 0.708 / 0.802 / 1.097 | 0.566 / 0.632 / 0.952 |
| **Residual readout** (first version, features standardised per fold) | 0.906 / 0.925 / 1.360 | 0.858 / 0.854 / 1.294 |
| **Residual readout** (`icl_resid.py`, the version used from here on) | 0.868 / 0.919 / 1.356 | 0.868 / 0.857 / 1.293 |
| Ridge alone | — / 0.636 / 1.061 | |

**Residual readout:** fit ridge (λ = 1) on the context; context rows get 5-fold out-of-fold ridge predictions, query
rows the full-context fit; the model is given the residuals as context targets and predicts the query residual; the
ridge prediction is added back. Only context rows are used for anything fitted.

- The models learn nonlinear structure well and the linear part badly. Above 100 features the plain models win
  0 of 8 contexts; with the residual readout 5 to 7 of 8.
- **The win rate against ridge depends on the fold draw by about ±0.04** (24M: 0.868 and 0.906 from two draws); the
  median and the mean do not move.
- **Ridge is now too weak a reference.** Under the residual readout any useful correction is a win. Stronger
  references on the same rows will be added from the stored H417 matrices (§3).
- **Long contexts are not exploited:** 896 context rows are no better than four 320-row subsamples. Training has not
  yet shown either model more than 320 context rows (the row curriculum is at 384 rows until 25% of the schedule).

## 2. Change to the 104M averaging group (`ICL_L_g1_w0`, `_w1`, `_w2`)
Two new arguments; everything else (schedule position, learning rate, data mix, averaging) is unchanged:

| Argument | Effect |
| --- | --- |
| `--p-resid 0.5` | half of the training batches become residual-target tasks: the target is replaced by what the in-context ridge leaves over (`icl_resid.residual_targets`, the code the readout uses). A task whose context residual has almost no spread keeps its raw target |
| `--row-floor 1024` | the row curriculum's maximum is at least 1,024 from now (it would reach 640 at hour 18 and 1,024 at hour 36). Rows per task: 1,024 with probability 2/3, 640 and 384 with 1/6 each; context sizes 96 to 896 |

- **Procedure.** Rolling restart, one worker at a time, each right after it has written its 20-minute checkpoint,
  with `--resume` from its own `ckpt_last.pt` and code stage `r8`. The next worker is restarted only after the
  previous one has trained, taken part in an averaging round and written a checkpoint. No checkpoint is overwritten
  by anything but the run's own next checkpoint; the named checkpoints stay.
- **If the first restarted worker fails** (does not start, non-finite loss, out of memory at the longer rows), it is
  restarted from the same checkpoint with the previous stage and arguments, and the change is abandoned and recorded.
- **Mixed group during the restart.** For up to about an hour the group averages workers on the old and the new task
  mix. They share weights and optimiser family; I expect no harm, and stop rule (c) of addendum 3 stays in force.
- **The 24M run `ICL_M_r3` is not changed.**

## 3. Monitor and rules
- Every evaluation of a restarted worker logs the plain readout as before and the residual readout beside it
  (`real_res_*`, `u_res`), with the per-context values of both.
- **Reference beyond ridge (reported, no rule):** for the same 106 contexts and the same fold-0 rows, the stored
  H417 matrices hold the out-of-fold predictions of each context's GP programs. The monitor report will add the win
  rate and median against (i) the program with the best cross-validated SEARCH utility and (ii) the KFB blend's
  out-of-fold prediction. These are stored values; nothing is recomputed by a worker.
- **Size rule (addendum 2), amended.** The 104M group now trains half on residual tasks, so its plain readout is no
  longer comparable. At the group's hours 12, 24 and 48 it continues only if, **on the residual readout**
  (`icl_diag.py`, same fold seeds), both its win rate against ridge and its median u are at least the 24M run's at
  half as many hours (6, 12, 24, counted as in addendum 2), minus 0.02. The 24M values come from its first stored
  checkpoint at or after those hours; for hour 6 that is the checkpoint of §1 (6.3 hours): win rate 0.868, median
  0.919, so the group needs at least 0.848 and 0.899 at its hour 12.
- The 24M run's hour-12 on-track rule (addendum 1) stays on the plain readout, as written.

## 4. A clean control for residual training: one more arm on xbabe4
`ICL_M_x4c_res`: the exact recipe of the running `ICL_M_x4b_inj` arm (24M, same initial checkpoint, same schedule
and stop at 24 training hours) plus `--p-resid 0.5`. It starts about 4 hours after the other arms and is compared
with `ICL_M_x4b_inj` at equal training hours, on the residual readout and on the plain readout.
- **Reading at hour 24** (mean of the last five evaluations): `RESIDUAL_TRAINING_HELPS` if the residual-readout
  median u is higher by at least 0.03; `HURTS` if lower by at least 0.03; `NO_EVIDENCE` otherwise. The plain readout
  is reported: residual training may cost something there.
- The three arms share one Apple GPU, so all slow down together; the injection ablation of addendum 4 stays matched
  in hours but will see fewer tasks than planned. The memory guard of addendum 4 covers the new arm.

## 5. Parameter averaging: a measurement, and no change
- The coordinator now records, each round, the displacement of the group mean and the cosine between successive
  displacements (`icl_avg.py --slowmo STATE --beta 0`; with β = 0 the returned weights are bit-identical to the
  plain mean). First values: displacement 0.026 of the weight norm per round, each worker 0.032 from the mean,
  **cosine between successive displacements 0.008**.
- Outer momentum (x_{t+1} = mean + β (x_t − x_{t−1})) is implemented but **not switched on**: with successive steps
  orthogonal it would add the previous step as noise. It may be switched on (β = 0.5) only if the cosine averages at
  least 0.2 over six consecutive rounds, and then with this revert rule: if the local worker's mean logged training
  loss over the three rounds after the switch exceeds the three rounds before by more than 0.10, β returns to 0.
- **Weakness recorded:** before the third worker joined, the two-worker group's loss at matched hours was only 0.017
  below the single run's. Plain averaging at an unchanged learning rate may be giving noise reduction and little
  speed. This addendum does not fix that; the learning rate is unchanged.

## 6. Expectations and risks
- **Expected:** the residual readout of the 104M group improves faster than under the old mix, because training now
  matches the readout. **Not expected:** a higher plain-readout win rate; it may fall.
- **Two changes at once.** Residual tasks and longer rows start together on the 104M group, so their effects cannot
  be separated there. §4 isolates residual training at 24M; the 24M run's own curriculum step to 640 rows (its hour
  11.5) is the only evidence on rows.
- **Throughput.** Longer rows cut tasks per second by roughly half; each task carries more rows.
- **Context / query asymmetry.** Context residuals are out-of-fold, query residuals come from the full fit and are
  smaller (ratio 0.87–0.95 on a test, 0.16 on an exactly linear target). Training and readout share the procedure,
  so the model can learn it; whether it does is visible in `u_res` on synthetic tasks.
- **Monitor reuse.** The readout recipe was chosen on the monitor, so monitor numbers are optimistic for it. The
  test cohorts are untouched; part B (stack columns on stored matrices, then APEX-DEV) is the test.

## 7. Code
`icl_resid.py` (new), `icl_train.py` (`--p-resid`, `--row-floor`, residual readout in `evaluate`, a log line on
resume with the arguments and code hash), `icl_diag.py` (new), `icl_avg.py` and `icl_sync.sh` (outer-step
measurement). Unit checks before publication: the batched out-of-fold ridge equals a per-task reference to 4e-14;
changing query labels changes neither context targets nor the ridge base; CUDA and CPU agree to 1e-14; a 35-step
run with both arguments on the Apple GPU logs both readouts.
