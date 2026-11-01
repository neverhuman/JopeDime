# H152 — zero-update checkpoint soup of saved flywheel adapters

**Status: complete. Verdict: NO_ADVANCE.** The rule was preregistered in [PREREG.md](PREREG.md), PR #1298, which
`rd_astra` reviewed and passed at `cd64b61e` and which merged as `38897495`. The study ran on xbabe1 CPU with **0
optimizer updates**.

**No soup beats R882 on the 91-context DL-Val.**
- MID: −0.0303 [−0.0985, +0.0072].
- SOUP4: −0.0331 [−0.0994, +0.0008].

**The whole deficit comes from one context.** MID, SOUP4 and A075 all flip val context #27 (family `fam-9f67f144…`,
5 contexts) across the gate cliff, from R882's +1.1227 to an invalid pick (−2). Leaving that context out is a
descriptive check, not a preregistered test. On that basis the context-mean differences are:
- MID: +0.0027;
- SOUP4: −0.0002;
- A075: +0.0032.

Weight averaging changes only 5–11 of the 91 picks. It does not carry R782's TRAIN advantage over to val.

## Hypothesis

- **Background.** R782 beats R882 on the 1,219 TRAIN contexts (H139/H145), but R882 wins the 91-context DL-Val
  selection.
- **Question.** Can a zero-update weight-space blend of the saved rounds keep R882's val skill and add R782's
  robustness?
- **How this differs from earlier waves.** H55 averaged during training, and H131 trained a soup.

## Setup

- **Arms.** These were fixed before any score:

  | Arm | Weights | Role |
  |---|---|---|
  | R882 | — | regression gate |
  | **MID** | 0.5·R782 + 0.5·R882 | **primary** |
  | SOUP4 | the mean of R782/R785/R859/R882 | secondary |
  | A025 / A075 | 0.25 or 0.75 on R782, the rest on R882 | exploratory, never admitted |

  Only the 47 trainable adapter tensors (29,211 parameters) are averaged, in float64. Every other tensor comes
  from R882, and all non-trainable tensors were bitwise identical across the four inputs.
- **Inputs.** One-time sha-verified copies of xbabe1 `run-20261030` rounds 782, 785, 859 and 882.
- **Evaluator.** The merged H139 evaluator, byte-pinned (`7a37821d…`), with the native flywheel endpoint.
- **Primary.** The 19-family equal-family gain against R882 on DL-Val. Three bootstrap seeds (152101–152103) each
  draw 10,000 times. ADVANCE needs a gain > 0, a median CI lower bound > 0 and Holm over MID and SOUP4.
- **Binding and resources.**
  - **Source.** All 6 bound files were byte-identical at the reviewed head `cd64b61e` and the merged `38897495`.
  - **Supervision.** `monitor.py` supervised the run; peak usage was 2 threads, 1.02 GB RSS, 196 MB scratch, 89.3%
    disk and load 23.4.
  - **Wall time.** 1,616 s.
  - **DOPE.** `c51a56d0` unchanged across all five arms.
- **Scored work.**
  - **Smoke (owner-authorized exception, t2155).** The pre-merge smoke scored **R882 only**, on all 1,310 contexts,
    as an exact regression. It is not a zero-evaluation smoke, and no soup arm was scored before the merge.
  - **The run.** It scored all five arms on 91 DL-Val and 1,219 DL-Train contexts each.

## Outcome

**Regression: PASS.** R882 reproduces 91/91 contexts with zero error and a mean of exactly 1.1377941898164525.
It makes 0 invalid val picks and 17 invalid TRAIN picks.

| Arm | DL-Val gain vs R882 [median 95% CI] | Median p | Families W/T/L | Elo vs R882 = 1500 [95% CI] | Invalid val picks | TRAIN gain (descriptive) [CI] |
|---|---|---:|---|---|---:|---|
| **MID** | **−0.0303 [−0.0985, +0.0072]** | 0.682 | 4 / 12 / 3 | 1518 [1426, 1613] | 1 | +0.0058 [+0.0000, +0.0137] |
| SOUP4 | −0.0331 [−0.0994, +0.0008] | 0.854 | 3 / 12 / 4 | 1482 [1387, 1574] | 1 | +0.0036 [−0.0004, +0.0102] |
| A025 (expl.) | −0.0004 [−0.0017, +0.0007] | 0.729 | 2 / 14 / 3 | 1482 [1406, 1555] | 0 | +0.0009 [−0.0008, +0.0034] |
| A075 (expl.) | −0.0298 [−0.0981, +0.0077] | 0.641 | 5 / 12 / 2 | 1555 [1463, 1656] | 1 | +0.0131 [+0.0023, +0.0271] |

- **Gates.** MID and SOUP4 fail every gate: gain, CI lower bound and Holm (adjusted p = 1.0 for both).
- **Seeds.** The three bootstrap seeds agree to within 0.0005 on every CI bound. They are resampling repetitions,
  not independent runs.
- **TRAIN family-disjoint folds (descriptive, TRAIN-exposed).** There are 5 folds of 41 families per seed, and
  MID's per-fold gain ranges from −0.0012 to +0.0228 across seeds. The gain grows with the R782 weight (A025 < MID
  < A075), which reproduces the H139 TRAIN ordering. U4 is slightly negative for every soup (−0.0008 to −0.0020).

## Reading

- **The val loss is a single gate-cliff context, not a broad degradation.**
  - Context #27 drops by 3.12 in MID, SOUP4 and A075.
  - Its family has 5 contexts, so the equal-family effect is −3.12 / 5 / 19 ≈ −0.033, matching the observed gains.
  - Family records are mixed: MID wins 4 families and loses 3, with Elo 1518. The CI is wide because one family
    carries the whole loss.
- **This matches H139 and H143.** Val differences between these adapters come from validity cliffs, where a pick
  crosses the gate and costs −2, rather than from small utility shifts. A blend inherits R782's TRAIN gain roughly in
  proportion to its R782 weight. On val, every blend that moves at least half of its weight off R882 flips context
  #27 over the cliff: MID moves 0.5 off, A075 0.75 and SOUP4 0.75 (spread over R782, R785 and R859). A025 moves
  only 0.25 off and does not flip it.
- **A025 is essentially R882.** It changes 5 picks, its gain is −0.0004, and it has no invalid picks.

## Next step

**Do not adopt a soup.** The live flywheel and R882 stay as they are.

**What would decide whether soups help.** A val estimand that is not dominated by one gate-cliff context, such as
gate-aware, cliff-excluded or a larger validation set, would have to be preregistered. That is a GOV and protocol
question, not something to tune here.

**H152 closes with no follow-up wave proposed.** It confirms, for zero-update blends, the H139/H143 finding that
selection on this DL-Val is cliff-driven.

## Files and custody

- **Files here:** [metrics.json](metrics.json) is the full readout, at 12.8 KB.
- **Code:**
  - [runner.py](runner.py);
  - [monitor.py](monitor.py);
  - [test_h152.py](test_h152.py), 13 data-free tests;
  - [preflight.json](preflight.json), the supervised R882 smoke.
- **S3** (create-only; `manifest.json` lists the SHA-256 of all 23 files):
  `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H152/opus-h152-20261101a/`. It holds:
  - `metrics.json` and `per_context.json`;
  - the execution identity, `launch.json`, the monitor files and resource logs;
  - the smoke receipts;
  - the per-arm DOPE and checkpoint-load receipts.
- **Not uploaded.** The soup checkpoints (`*.pt`, about 73 MB) remain on xbabe1 at `~/jope-data/h152/run-run1`.
  Nothing was deleted, and no `.pt` file is in Git.
