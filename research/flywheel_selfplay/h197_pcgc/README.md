# H197: per-context gradient-norm clipping

**Status:** complete — **NO_ADVANCE**. No serving promotion or live-loop change is authorized.

## Hypothesis and setup

H197 tested whether clipping each FIT context’s adapter gradient before averaging reduces outlier influence and improves held-family utility from the fixed r882 adapter. The loss, target, model/kernel, candidate sampling, optimizer recipe, and 40-update budget were unchanged. Only the 47 adapter tensors (29,211 parameters) trained; DOPE remained frozen.

The preregistered study used 1,219 real TRAIN contexts across 205 families, three training seeds (197101–197103), three independent family-fold seeds (197141–197143), five rotating outer folds, and five arms: R882, CTRL40, PCGC40 (primary), GCLIP40, and PCGC_SHUF40. FIT, B, and A families were disjoint within each cell. All 60 trained trajectories (2,400 optimizer updates) completed before B evaluation. The pooled B panels and all FIT states were bound by a fsynced fixed-step-40 lock before any A scoring.

## Primary result

Scores use equal-family averages, with contexts averaged equally inside each family. Each seed’s pooled A score covers all 205 families exactly once. The primary is the median of three seed-level PCGC40-minus-R882 gains, with a paired-family bootstrap (10,000 draws; shared family resamples across seeds; seed 197150).

| Seed | PCGC40 − R882 equal-family gain |
|---:|---:|
| 197101 | −0.03214 |
| 197102 | −0.02510 |
| 197103 | −0.02223 |

Median gain was **−0.02510** (95% paired-family bootstrap interval **[−0.05128, +0.00009]**; one-sided p = **0.9743**, Holm m=1; 0/3 positive seeds). The interval does not meet the preregistered strictly-positive lower-bound criterion. The family harm rate was **46.83%**, and PCGC40 did not strictly beat the shuffled-factor placebo. The preregistered verdict is **NO_ADVANCE**.

PCGC40 versus CTRL40 was near zero (median +0.00063; 95% interval [−0.00094, +0.00190]). PCGC40 versus PCGC_SHUF40 had median −0.00177 (95% interval [−0.00371, +0.00044]). These are descriptive comparisons.

H148 family-paired Elo versus R882 was 1485.63 for PCGC40 (lift −14.37; 95% family-bootstrap interval [−24.34, −4.38]). CTRL40, GCLIP40, and PCGC_SHUF40 were also below R882 (see `metrics.json`). Exact −2.0 selected-choice counts by seed and arm, clip rates/factor quantiles, and full unrounded family-level summaries are included in the compact metrics and raw evidence.

Native r882 DL-Val replay passed bit-exactly on 91/91 contexts before TRAIN outcome pools were opened. Subsequent DL-Val candidate results are descriptive only and do not affect selection. Corresponding-fold family Jaccard values across the three independent fold partitions ranged from 0.038 to 0.242 (all five fold pairs for each seed pair are retained in `metrics.json`). The audit recorded zero protected reads and zero refused accesses. The guarded CPU monitor passed without resource violations; the complete output remained below 300 MB.

## Conclusion and next step

Per-context clipping did not improve the preregistered family-level outcome and did not pass the placebo or harm-rate checks. Do not advance or promote PCGC40. Close H197 as NO_ADVANCE; any new experiment requires separate ENG authorization and a new preregistration.

## Reproducibility and custody

Preregistration: [`PREREG.md`](PREREG.md). Fold assignments: [`fold_manifest.json`](fold_manifest.json). Source is bound to merged commit `47c7a43cefdbb9ea360d335d829d9381aa55b15b`; writer fixture/fix merged in PR #1634. The full raw traces, fit states, B/A/DL-Val panels, audit, and resource receipts are kept outside Git in versioned S3 custody. See [`custody.json`](custody.json); the compact metrics include hashes for the principal run summaries.
