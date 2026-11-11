# H424 PREREG addendum 1 — the first read passed; the second read with the hour-24 checkpoint, and a duplicate screen

**Status.** Published on `origin/v1/next` before the second read's inference starts. The first read is complete and
is stated here; the second read's checkpoint does not exist yet (about 11:20Z).

## 1. First read (hour-20 checkpoint; gates 7 of 7 before the readout)
`I` vs `B` on the 115 `DEV` families: clipped family-mean gain **+0.0092 u** (0.9% lower RMSE), one-sided sign-flip
p 0.00006, leave-one-family-out minimum +0.0064, median +0.0001, 59 wins / 34 ties / 22 losses, ΔElo +115.9
[+64.2, +173.1]. **Verdict `ICL_CONFIRMED`.**
- Pre-declared secondary: no context is at the ±0.5 clip, so nothing is excluded: 59 / 34 / 22, win share of
  decided families 0.73, gain +0.0092 u.
- The column gets weight in 84 of 115 families (mean weight 0.11). The model alone is far below the stack (mean
  −0.56 u; ahead in 23 families).
- **The PREREG expected about +0.002 u and a null. That expectation was wrong.**
- Inference: 115 of 115 columns, 43,819 CPU-seconds, 78 minutes inside the 4-hour cap.

## 2. Second read (PREREG §4, consequence of `ICL_CONFIRMED`; ordered in the relayed note)
- **Checkpoint:** `ICL_M_r3/ckpt_h024.pt`, the named checkpoint of training hour 24. The driver waits for the file,
  copies it read-only, and writes its sha256 (copy and training host) to the log before inference starts.
- **Everything else is the first read's:** the same `DEV` bank file (no `DEV` table is opened again), the same
  published inference and stack code, 16 single-thread CPU processes, no GPU, the same 4-hour cap from the start
  of inference.
- **What it is:** a second read on the same 115 families. It is not independent of the first and carries **no new
  verdict**; the wave's verdict stays the first read's.
- **Reported (`readout2_h424.py`):** `I24` vs `B` with the first read's statistics and the pre-declared secondary;
  `I24` vs `I` (does four more training hours give a better column); the column's weight and its value alone.
- **Gates (`gates2_h424.sh`):** the first read's seven on the second read's outputs, plus: this addendum is older
  than the first column file of the second read.

## 3. A duplicate screen, run after the first readout (descriptive; `audit/dup_audit.py` in the evidence folder)
The model was trained on the `FIT-ICL` families. If a `DEV` table were a copy of a `FIT-ICL` table, the column
could have seen its targets.
- Manifest: 0 shared table hashes, dataset ids or family ids between `DEV` and `FIT-ICL`.
- **Target values:** for 105 `DEV` contexts with at least 20 distinct SEARCH target values, the largest share of
  those values that also occur in one `FIT-ICL` training table is at least 0.9 in 11 contexts and at least 0.5 in
  17. Some of these are the same public dataset under two names (for example `cpmp-2015-runtime-regression` in
  `DEV` and `cpmp-2015-regression` in `FIT-ICL`); others are coarse targets that many tables share (101 values
  from 0.00 to 1.00).
- **So the cohort split by source name does not remove every copy of a dataset.** This also touches earlier waves'
  assumption that `DEV` families are unrelated to the design cohorts, though those waves fitted at most a handful
  of parameters.
- **Sensitivity of the first read:** without the 17 flagged contexts the gain is +0.0087 u (p 0.0009) with 46 wins and 21
  losses (98 contexts); the three largest gains are on contexts with almost no overlap (shares 0.002 to 0.010).
  The result does not rest on the flagged contexts.
- **Limits of the screen:** it compares target values only. A copy with a transformed target is missed; a coarse
  target is flagged falsely. A feature-level comparison is not done.
- The second readout repeats `I` vs `B` and `I24` vs `B` without the flagged contexts (share ≥ 0.5).

## 4. Files added
`readout2_h424.py`, `gates2_h424.sh`; `stage_h424.sh` now stages them and this addendum.
