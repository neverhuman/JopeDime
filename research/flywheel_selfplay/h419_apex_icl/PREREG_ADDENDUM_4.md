# H419 PREREG addendum 4 — column-injection ablation on xbabe4 (two arms, 24 hours)

**Status.** Published on `origin/v1/next` before either arm starts. No prediction on a test cohort exists.

**Reason.** Owner order relayed by ENG:watcher (05:11 and 05:40 MST): use xbabe4's GPU for a separate, useful
pre-registered H419 job that does not slow the 104M group — a seed replicate or a data-mix ablation — with
checkpoints, at most 24 hours and its own stop rule.

## 1. Question
Addendum 1 added synthetic column injection on the owner's direction. Nothing yet measures what it contributes.
Does injection improve the model on real tables, at equal training?

## 2. Design
Two arms run side by side on xbabe4 (Mac Studio M3 Ultra, Apple GPU, torch 2.15, fp32), so they share hardware,
precision, seed, budgets and schedule and differ only in the data mix.

| Arm | Run | Data mix |
| --- | --- | --- |
| A (replicate of the published 24M recipe) | `ICL_M_x4b_inj` | addendum 1: injection in half of the synthetic tasks; real-table tasks filled with injected columns |
| B (ablation) | `ICL_M_x4b_noinj` | no injected column anywhere: `--inject-p 0`; real-table tasks take exactly F real columns from tables that have them |

- Both start from the 24M run's pre-injection checkpoint `ICL_M_r2/ckpt_last.pt` (sha256 `a603f767…`), exactly as
  `ICL_M_r3` did, with seed 1, token budget 600,000, row-token cap 40,000, peak LR 2e-4.
- **Schedule matched per task, not per hour.** The Apple GPU is slow: 33–42 tasks per second in the smoke, and the
  two arms barely slow each other (40–57 each in their first minutes), against 289 for `ICL_M_r3`. The schedule
  length is therefore 360 hours and the real-table ramp 28,100 seconds, so that warmup, ramp and row curriculum
  fall at the same task counts as in `ICL_M_r3`. Each arm is **stopped at 24 hours** (`--stop-h 24`), after about
  3 million tasks: the stretch `ICL_M_r3` covered in its first 3 hours.
- Arm A doubles as a cross-hardware replicate of `ICL_M_r3`: their gap at matched task counts is the only noise
  estimate this wave has.
- In arm B the real-table share is lower in wide batches (few tables have 33 or more feature columns); the realised
  share is logged for both arms.

## 3. Reading rule (fixed now)
Metric: the real-table monitor of addendum 1 (106 FIT-SEL contexts, SEARCH rows only), win rate against ridge and
median u, averaged over each arm's last five evaluations. Also reported: synthetic held-out u with and without
injected columns.
- `INJECTION_HELPS`: A − B win rate ≥ +0.03 and A's median u ≥ B's.
- `INJECTION_HURTS`: A − B win rate ≤ −0.03 and A's median u ≤ B's.
- `NO_EVIDENCE` otherwise. One seed per arm and 1.6 million tasks: this can show a clear early effect, not a small
  or late one. Whatever the reading, the test-cohort verdicts of part A are unaffected; a `HURTS` reading would be
  raised with the owner before the long runs' data mix is changed.

## 4. Stop rules and budget
- 24 hours per arm, both at once: 24 hours of wall-clock. Latest checkpoint every 20 minutes, named every 6 hours.
- Both arms stop if xbabe4's free memory falls below 15% (`x4_guard.sh`; the host also runs a VM and user
  applications, which are not touched), or if an arm averages fewer than 8 tasks per second over its first hour.
- Data: the FIT-ICL bank and the FIT-SEL SEARCH-row monitor of addendum 1 (TRAIN side). No dev91, DL-Val,
  DL-Holdout, sealed or CONFIRM byte.

## 4a. Correction before the scored arms (12:46Z)
The first launch (`ICL_M_x4_inj`, `ICL_M_x4_noinj`, schedule 700 hours, ramp 54,800 s) assumed the arms would halve
each other's speed. They do not. Both were stopped after 2 minutes, before any evaluation, and relaunched as
`ICL_M_x4b_inj` and `ICL_M_x4b_noinj` with the 360-hour schedule above. Nothing else changed.

## 5. Files
`icl_train.py` gains `--device`, `--stop-h`, `--no-inject-real` and CPU float64 for the reference maths (the Apple
backend has no float64); `icl_inject.RealBank.sample(fill=False)`; `x4_guard.sh`.
