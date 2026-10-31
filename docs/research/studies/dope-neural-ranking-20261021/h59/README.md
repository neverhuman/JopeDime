# H59 robust / context-normalized flywheel loss

## Hypothesis

Clipping per-candidate smooth-L1 at 1.0 (CLIP), dividing each context's loss by a robust MAD scale (NORM), or both (CLIP+NORM) would stop extreme (−2.0-clipped) contexts from dominating updates and raise family-level fold-B utility over the current loss (CONTROL).

## Setup

Four arms × seeds 12/13/14, each from a fresh r882 copy, with paired contexts, slates, and minibatches. Fixed 20 rounds × 2 AdamW steps (40 updates), batch 256, slate 192, lr 3e-4 cosine to 3e-5, weight decay 1e-4, grad clip 1.0, no early stopping. Training used fold A only. Fold B (41 families) was scored once per final candidate and paired with r882. Source `cce86a22`, run `~/jope-data/h59/h59-wave`. All 12 cells checked DOPE `c51a56d0` before and after (0 mismatches), trainable = 29,211, holdout/outer payloads opened = 0, DL-Val not opened.

## Outcome

| Arm | Seed-median gain [95% CI] | Family median gain [95% CI] | Mean LCB95 | Harm frac | Elo by seed (12/13/14) | Median Elo | Admitted |
|---|---:|---:|---:|---:|---|---:|---|
| CONTROL | −0.0986 [−0.1149, −0.0983] | −0.0122 [−0.0714, +0.0058] | −0.2073 | 53.7% | 1479.4 / 1472.5 / 1480.2 | 1479.4 | No |
| CLIP | −0.0326 [−0.0447, −0.0225] | −0.0021 [−0.0342, +0.0087] | −0.0984 | 46.3% | 1492.7 / 1487.4 / 1484.5 | 1487.4 | No |
| NORM | −0.0894 [−0.1103, −0.0888] | −0.0208 [−0.0735, +0.0078] | −0.1667 | 53.7% | 1475.5 / 1470.0 / 1477.7 | 1475.5 | No |
| CLIP+NORM | −0.0823 [−0.0943, −0.0741] | −0.0059 [−0.0735, +0.0028] | −0.1354 | 48.8% | 1475.5 / 1481.9 / 1473.8 | 1475.5 | No |

All gains are versus r882 (higher is better). Elo is family-grouped Bradley–Terry, with r882 = 1500.

## Verdict

No admission. Every arm is worse than r882 on all three seeds. CLIP consistently hurts least: about +0.066 seed-median utility and about +8 Elo over CONTROL on every seed. So clipping the loss does reduce the damage from extreme residuals, but no loss variant turns 40 updates from r882 into a gain. NORM alone does not help.

Red-team: seeds are paired and disjoint. The seed CIs are narrow, so the CLIP-vs-CONTROL ordering is not seed noise, but it is a less-harm result, not a win. There is no tie acceptance and no checkpoint pick on B. The update recipe here (lr 3e-4, batch 256) is more aggressive than the live loop (lr 1e-4, batch 16), so CONTROL harm partly reflects step size. The r882 exposure to fold B biases results against candidates.

## Next

Keep CLIP as the default loss for any later candidate-generation wave (H76/H77). Do not adopt it in the live loop on this evidence.
