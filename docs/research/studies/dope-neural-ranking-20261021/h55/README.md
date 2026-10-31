# H55 r882 adapter averaging

## Hypothesis

Family-grouped SWA-8 or EMA-25 averaging of the r882 adapter would improve higher-is-better utility on one untouched seed-4900 fold-B cohort over the matched r882 control, while meeting the preregistered family-bootstrap and no-harm admission thresholds.

## Setup

The candidate adapter was trained on fold A only (four fifths of the family-grouped DL-Train split). The untouched fifth, fold B, was scored once after training. H55 used seeds 12–14, SWA-8 with eight members and one round of k=2 adapter-update steps per member, and EMA-25 with decay 0.9. DOPE `c51a56d0`, the r882 checkpoint, and the source manifest were pinned. No sealed holdout or outer rows were accessed.

All six GPU outcome cells completed in preregistered order under `qualification.lock`. The source-bound two-context dry run and its audit passed. The attempt4 wave receipt reports 611.887 seconds of summed GPU cell wall, including the 5.330 seconds conservatively carried from attempt2, against the fixed 1,800-second cap. Cumulative lock wait was 2,282.175 seconds, separately below the three-hour contention limit. Maximum observed process RSS was 4,878,388 KiB; CUDA free memory after each lock acquisition was 24,124 MiB. The independent outcome audit passed.

## Outcome

| Arm | Pooled fold-B mean gain vs r882 | One-sided family-bootstrap mean LCB95 | Pooled family median gain | Families below −0.01 | Admission |
|---|---:|---:|---:|---:|---|
| SWA-8 | +0.001807 | 0.000000 | 0.000000 | 0.0% | No |
| EMA-25 | −0.074640 | −0.155457 | 0.000000 | 39.0% | No |

Admission required mean LCB95 strictly greater than 0.001, pooled family median at least zero, and no more than 10% of families below −0.01; ties fail. SWA-8 failed the strict LCB threshold despite a small positive mean. EMA-25 failed the LCB and no-harm thresholds. Therefore no arm was admitted and **zero terminal DL-Val reads were allowed or performed**.

## Metrics

Elo versus r882 is diagnostic only and does not determine admission. SWA-8 seed deltas were +3.39, +3.39, and +1.69 Elo; EMA-25 seed deltas were −20.79, −17.39, and −20.22 Elo. Across the six cells, H55 recorded 100 wins, 1,086 ties, and 122 losses against r882 on the family-grouped diagnostic comparisons. The compact metrics artifact is [`metrics.json`](metrics.json) (25,234 bytes); full cell and audit receipts remain in `~/jope-data/h55/h55-wave-attempt4/`.

## Verdict

No H55 candidate passed the preregistered admission rule. The fold-B result is not a terminal validation result. The r882 control was trained on all of DL-Train (H56 audit §5), which biases this control against candidates and makes the comparison conservative; terminal DL-Val would have been the clean read, but the admission rule correctly allowed none.

SWA-8 tied r882 on 216 of 218 datasets for seeds 12 and 13, and 217 of 218 for seed 14; seeds 12 and 13 also had identical fold-B utility (1.285445). This repeated plateau suggests a structural limit in the candidate-pool/selection granularity rather than optimizer noise.

## Next

Do not promote either arm or extend its training budget from these results. H58 proceeds after H55 outcome-wave completion; H57 remains queued until H55 closeout and follows H58 as directed by the owner.
