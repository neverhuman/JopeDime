# H183: distill ten fresh adapters into one

**Outcome: NO_ADVANCE.** The primary DIST10 student gained `+0.009259` versus r882, but its paired-family 95% CI was `[-0.028009, +0.040530]`, only two seeds were positive, and Holm m=2 p was `0.454755`. The primary gate fails. All five folds completed after an explicitly authorized same-attempt continuation. No model was adopted.

The hypothesis was that one fresh 29,211-parameter student could retain the utility of a trained ten-member ensemble. RENS10 averages the exact same ten teacher initial states with zero updates, testing the contribution of training. See [PREREG.md](PREREG.md), the byte-identical [worker metrics](metrics.json), and [custody/replay receipt](custody.json).

## Fixed setup

Real S3 TRAIN: 1,219 contexts / 205 families; five family-grouped outer folds, each 123 FIT / 41 B reserved unused / 41 A. Teachers, target slates and students use FIT only. Replicate seeds are 183101–183103; teacher seeds are 183201–183210, 183301–183310 and 183401–183410; student seeds are 183501–183503. **Fold seed 183140 is shared across the three outer replicate seeds**: these are stochastic fits on one family partition, not three independent family cohorts. Bootstrap seed 183141 gives the same 10,000 paired family draws across arms and seeds, reducing the three-seed summary on axis=0.

DOPE remains frozen at c51a56d0; exactly 47 adapter tensors / 29,211 parameters train. Teachers receive 40 live-loss updates each. FRESH1 shares the student initialization and first 40 student minibatches. The teacher distribution is softmax of the ten mean float32 costs at T=.2. Distillation mixes .5 live loss and .5 detached KL(teacher || student), without T² scaling. DIST10 is the step-40 snapshot of the same 80-update DIST10_80 optimizer trajectory. The total is 7,800 actual updates / 8,400 nominal arm exposures. B never selects a model or threshold.

## Registered TRAIN outcomes

Utility averages contexts within family, then families equally, then takes the median of the three seed means. Contrast points are medians of paired per-seed differences; they need not equal differences of displayed arm medians. Intervals condition on the retained cohort and fixed trained fold models. Elo is canonical 400-iteration BT with .01 prior, 1e-4 utility ties, r882=1500 and the same paired family bootstrap.

| Arm | Median utility | Gain vs r882 [95% CI] | Elo [95% CI] | Exact −2 counts by seed |
|---|---:|---|---|---|
| R882 | 1.385355 | +0.000000 [+0.000000, +0.000000] | 1500.00 [1500.00, 1500.00] | 17 / 17 / 17 |
| FRESH1 | 1.390575 | +0.005220 [-0.030665, +0.037953] | 1506.14 [1492.40, 1521.57] | 22 / 23 / 18 |
| RENS10 | 1.402328 | +0.016972 [-0.014237, +0.050233] | 1507.08 [1491.36, 1520.66] | 18 / 17 / 17 |
| ENS10 | 1.398902 | +0.013547 [-0.016735, +0.048330] | 1508.91 [1492.63, 1522.47] | 18 / 17 / 17 |
| DIST10 | 1.394615 | +0.009259 [-0.028009, +0.040530] | 1507.27 [1493.89, 1523.52] | 21 / 22 / 18 |
| DIST10_80 | 1.387423 | +0.002068 [-0.033149, +0.034064] | 1507.23 [1493.88, 1522.44] | 21 / 26 / 20 |

Counts include every decision with original utility exactly −2.0; none is filtered out. RENS10 has 18/17/17 such decisions versus r882’s 17/17/17, so its positive point estimate is not a net reduction of that count. Per-seed utilities/Elo and absolute-utility intervals are in metrics.json.

| Arm | Gain seed 183101 | Gain seed 183102 | Gain seed 183103 |
|---|---:|---:|---:|
| R882 | +0.000000 | +0.000000 | +0.000000 |
| FRESH1 | +0.005220 | -0.009118 | +0.008378 |
| RENS10 | +0.008933 | +0.017878 | +0.016972 |
| ENS10 | +0.010626 | +0.016725 | +0.013547 |
| DIST10 | +0.009299 | -0.007158 | +0.009259 |
| DIST10_80 | +0.009153 | -0.017379 | +0.002068 |

DIST10 clears the +.001 point margin, but fails LCB>0, all-three-positive and Holm p<.05. DIST10_80 has gain +0.002068, CI [−0.033149, +0.034064], two positive seeds and the same adjusted p=0.454755. It cannot replace the primary. Holm includes only these two student arms; both ensembles are diagnostic.

## Prespecified diagnostics

| Paired contrast | Median gain | 95% CI |
|---|---:|---|
| ENS10 − RENS10 | -0.001153 | [-0.006673, +0.003551] |
| DIST10 − RENS10 | -0.007713 | [-0.025433, +0.004109] |
| DIST10_80 − RENS10 | -0.014904 | [-0.036693, -0.000841] |
| DIST10 − ENS10 | -0.004287 | [-0.022309, +0.006898] |
| DIST10_80 − ENS10 | -0.011479 | [-0.032655, +0.001732] |
| DIST10 − FRESH1 | +0.001960 | [+0.000164, +0.004408] |

These intervals are unadjusted diagnostics and do not alter admission. ENS10−RENS10 covers zero; there is no detected training advantage over the exact initial-state ensemble. DIST10−FRESH1 is positive in its prespecified diagnostic, but that does not establish primary lift versus r882. The negative DIST10_80−RENS10 interval is likewise diagnostic.

| Student | Equal-family teacher-choice agreement [95% CI] | Context-weighted agreement by seed | Point teacher-gain ratio |
|---|---|---|---:|
| DIST10 | 72.82% [69.47%, 75.42%] | 72.93% / 72.76% / 69.89% | 0.683511 |
| DIST10_80 | 72.01% [68.81%, 74.91%] | 72.68% / 72.03% / 69.73% | 0.152632 |

The ratios 0.683511 and 0.152632 are descriptive ratios of seed-median gains. ENS10’s gain CI includes zero, so the denominator is unstable: no ratio CI or reliable fraction-retained claim is made. The absolute paired student−teacher gaps above remain interpretable diagnostics.

## Interruption and authorized continuation

The original execution stopped at one-minute host load 138.596 above the registered 108.8 limit, while host CPU was 34.4%. It had persisted 12/15 complete assessment cells. The thirteenth cell had completed all 520 FIT updates and saved its teacher/control/student states and targets. The worker wrote a SIGTERM failure receipt with 6,760 started and completed calls; the supervisor finalized with −9 after its grace period. Those records remain in custody.

Owner ruling `hwave_t0838_h183_resume.md` explicitly authorized continuing the same attempt and changing the monitor to sustained overload. The continuation reused the partial thirteenth cell’s saved states with zero retraining, completed its missing A shards, and performed exactly 1,040 new updates for the two untouched cells. **All 385 original files, including completed FIT target locks, training receipts, adapter states and existing A shards, have identical pre/post-resume SHA-256 hashes.** All 15 cells and the original 7,800-update total are complete.

The resumed monitor requires three high-load readings five seconds apart; CPU, memory, disk, RSS and threads remain checked every .5s. It retained the original four-hour absolute deadline and passed. All work ran on xbabe2 CPU at nice19/math1; xbabe1 fallback was unused. Across both execution segments, peak RSS was 3.402 GB, maximum threads three, maximum host CPU 61.2%, minimum MemAvailable 52.51 GB and maximum disk use 93.724%, within the other caps. The original load stop is disclosed, not counted as an uninterrupted clean run.

The ruling’s shorthand about a B lock and no pre-existing A scores did not describe H183’s preregistration: B was and remains unused, and twelve A panels already existed. Those outcomes changed no arm, seed, loss, minibatch plan, cohort, margin or estimator. DL-Val remains descriptive: all 91 r882 utilities and native candidate-cost bytes, mean 1.1377941898164525, passed before both execution segments. No B selector or retrospective fold reduction was added.

## Verification and custody

A separate numerical implementation replayed all 390 A score shards / 95,082 choices, 230,400 FIT target probabilities, adapter hashes, FIT/A/B exclusions, shared-draw bootstrap, canonical Elo intervals, Holm m=2, agreement, ratios and the strict ruling. It performed zero optimizer updates; this is an independent numerical implementation, not an independent training rerun. A further deterministic replay matched all 7,800 recorded minibatches, full teacher plan hashes, and exact student target candidate indices against the fixed sampler, seeds and rehashed pool fitness. Source/assets/private-input end hashes and both native baselines passed. Both runtime audit receipts have zero protected reads and no refused paths.

Versioned S3 archive: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H183/gpu_fused_redteam_codex-20261101/09d1275ddb071d41/evidence.tar.gz`, version `W9KbeJhKqZpL4b4A.xWK35asdrAzWT88`, SHA-256 `09d1275ddb071d41526374296414e7e30d104f841ef002fc941c43395eb47ee7` (461,982,857 bytes). An actual versioned GET restored/hash-verified all 528 members; all 407 NPZ and 17 model files decoded. The scientific Git bundle from that restored archive was fetched into the independent bare verifier and contains commit `bf0db127d4a0311f8a48d015a96e614a6053f305`. Its incremental base `c859f11c…` is retained on main. Custody includes the original source, content-pinned operational resume worker/monitor and owner ruling, model states, all candidate scores/choices, FIT targets, interruption and preservation/replay receipts. This is not a new full restore of the inherited upstream dataset closure.

Preregistration is PR #1539. Scientific cross-review rd_argon #5728391631 is retained at its recorded head; the final pre-run signal/count operational delta was self-reviewed #5728482055 after the bounded follow-up window, as disclosed in that PR. The later explicit owner ruling authorizes continuation. The scientific recipe/estimands remain the original merged source; exact operational continuation code is content-pinned in S3 custody.

Local validation: 16 synthetic-only controls, Ruff, diff check and five jope-records gates, distinct from numerical replay and actual restore. No hosted full-CI or serving-qualification claim.

## Scope and next step

The inherited r882 and frozen features have historical TRAIN exposure. These are conditional cross-fit continuation results on one fixed family partition, with dependent fold models and a three-seed summary. They do not establish unseen-family generalization or a learning ceiling. H175’s measured 41-family oracle-bank MDE80 (+0.094678) and 205-family planning extrapolation (+0.042341) are scale references, not measured H183 power. The broad intervals and NO_ADVANCE do not prove that a smaller benefit is absent.

Keep r882 as reference. A future distillation test needs a new preregistration; any promotion still requires H177/H178 qualification.
