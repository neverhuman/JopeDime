# H146 — Lookahead optimizer on the r882 adapter continuation

**Status: complete. Verdict: NO_ADVANCE** (rule preregistered in [PREREG.md](PREREG.md), PR #1276; CPU on xbabe1,
360 adapter updates).

**No arm beats r882 on the 91-context DL-Val.** Lookahead (k = 5, α = 0.5) lands at −0.0097
[−0.0230, +0.0002], with 0/3 seeds above r882. It does **reduce the harm of continuing to train**: against the
live-recipe CTRL (−0.0237 against r882) it gains **+0.0141 [+0.0014, +0.0331] on 3/3 seeds**, and it halves
adapter drift (0.22 vs 0.44 L2). LA_K10 gives the **same per-seed val utilities as LA**, so it is a pseudoreplicate
on this metric.

## Hypothesis

H93's Lookahead arm never reached scoring, and H122 prepared an executor but never ran it. The question: does
wrapping the live AdamW recipe in Lookahead let a continuation from r882 improve flywheel val utility?

## Setup

- **Arms** (fixed before any outcome):
  - CTRL: the live H140/H144 recipe (AdamW at learning rate 1e-4 and weight decay 1e-4, 40 steps, batch 16, slate
    192, gradient clip 1, SmoothL1 + 0.1 listwise KL).
  - LA: the same AdamW inside Lookahead with k = 5, α = 0.5.
  - LA_K10: k = 10, α = 0.5.
  - Slow weights are synced at step 40 and evaluated. The schedule was verified in every cell.
- **Seeds and folds.** Seeds 146101–146103 and fold seed 146140 (fold 0, 242 contexts, descriptive only).
- **Training scope.** Only the 29,211-parameter adapter trains. Every cell starts from the SHA-pinned r882 copy
  (`0084479a…`), copied read-only from the live run directory.
- **Code.** The executor is copied from merged H144 and the Lookahead state from H93/H122; neither study's files
  were edited.
- **Primary estimand.** 19-family equal-family native utility on the 91-context DL-Val, as the median over 3
  seeds, with a 10,000-draw shared family-bootstrap CI and Holm (m = 2).

## Outcome

**Regression: PASS.** r882 reproduces 91/91 contexts with zero error and mean 1.1377941898. The baseline
equal-family utility is 1.15362.

| Arm | Median gain vs r882 [95% CI] | Seeds > r882 | Elo vs r882 = 1500 [95% CI] | Gain vs same-seed CTRL [95% CI] | Invalid picks (per seed) | Adapter L2 drift |
|---|---|---:|---|---|---|---|
| CTRL | −0.0237 [−0.0550, −0.0027] | 0/3 | 1365.8 [1241.9, 1500.0] | — | 1 / 0 / 0 | 0.443 / 0.443 / 0.438 |
| **LA (k = 5)** | **−0.0097 [−0.0230, +0.0002]** | 0/3 | 1444.7 [1321.4, 1555.3] | **+0.0141 [+0.0014, +0.0331]** (3/3) | 0 / 0 / 0 | 0.219 / 0.219 / 0.217 |
| LA_K10 | −0.0097 [−0.0230, +0.0002] (same values as LA) | 0/3 | 1444.7 | +0.0141 [+0.0014, +0.0331] | 0 / 0 / 0 | 0.219 / 0.219 / 0.217 |

- **Gates.** Both Lookahead arms fail every ADVANCE gate against r882: margin, CI, seeds, concentration and Holm
  (adjusted p = 1.0). r882 itself has 0 invalid val picks.
- **Secondary (DL-Train fold 0, 242 contexts, descriptive).** All arms are below r882:
  - CTRL −0.039 [−0.084, −0.004];
  - LA and LA_K10 −0.028 [−0.077, +0.007].

  There are 3 invalid picks per seed in every arm, against 1 for r882.

## Reading

- **Continuing to train r882 hurts val utility, and Lookahead only limits the harm.** CTRL loses −0.024. Lookahead's
  α = 0.5 slow weights pull each update halfway back toward the start, so the adapter moves half as far and loses
  less (−0.010). Lookahead does not create a gain.
- **The LA-vs-CTRL advantage is concentrated.** 92% of the positive family gain is in the top 3 families. It reads
  as less drift, not a better direction.
- **Seed independence.** The seeds are not pseudoreplicates within an arm: utilities and adapters differ across
  seeds. **LA and LA_K10 are pseudoreplicates of each other** on DL-Val: their adapters differ slightly, but their
  argmin selections and utilities are identical. This is the quantized-plateau effect seen in H140 and H142.

## Next step

No adoption note and no live-loop change. With the H144 weight-decay sweep and H143's finding that the plateau's
"improvements" were all invalid-pick cliffs, the evidence points away from optimizer tweaks on the r882
continuation. The candidate generator, meaning the training signal itself, has to change.

## Execution, red-team and custody

- **Attempts.**
  - **Attempt 1** (02:15:39Z) stopped after 2 s, **before the baseline or any update**: the audit hook refused
    `launch.json` because the custody directory was under `/home/ubuntu/jope-data` (the data root). Its receipt
    shows 0 updates, 0 scored contexts and 0 protected reads.
  - **Attempt 2**, the single retry allowed by the dispatch, ran with identical merged source `8b025049a` and the
    custody directory outside the data root.
  - A constructor smoke had hit the same staging error before merge; it was re-staged with no code change.
  - All attempts are in custody.
- **Resources (supervisor, 3,736 samples).**
  - At most **3 OS threads** (limit 4).
  - Peak RSS **3.33 GB** (limit 6,000,000,000 B).
  - Free disk ≥ 157 GB, scratch ≤ 10 MB.
  - **0 pauses**, nice 19, 18 min wall.
  - No GPU (CUDA not initialized), no `qualification.lock`, and no contact with the live flywheel.
- **Integrity.**
  - 360 optimizer updates.
  - Every step checked the complete non-adapter state byte-for-byte and DOPE `c51a56d0`, with no drift.
  - **0 protected reads** and 0 audit refusals in attempt 2.
- **Custody.**
  - S3: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H146/opus-h146-20261101a/`, 45
    objects: both attempts, the smoke runs, nine cell receipts, nine adapter arrays, baselines, metrics, audits and
    resource logs. All are create-only with a version-pinned readback.
  - Rescue copy: `/home/ubuntu/jope-rescue-20260921/h146-rd_opus-20261101/`.
- **[metrics.json](metrics.json)** is 7.8 KB.
