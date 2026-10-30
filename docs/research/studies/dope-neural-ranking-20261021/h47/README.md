# H47: utility-aligned ranking losses

Status: **COMPLETE — NO WIN** (2026-10-30, xbabe2, `cuda:0`). H47 ran 12 seeds across R/Z/PW/RP on the approved round-785 checkpoint copy. All arm-level intervals below resample the 19 DL-Val families 10,000 times and summarize the median seed metric; they are not seed-noise intervals.

## Protocol and selection limits

Each arm used 64 AdamW steps with EMA 0.98, evaluated every 8 steps. Checkpoint selection was viewed on DL-Val; therefore the reported trajectories and best-seen values are exploratory. H47 did not use inner family-grouped CV over DL-Train for selection, so no arm can qualify as a selected win under the strict rule (median final Δ > 0, family-bootstrap lower 95% bound > 0, and selection using inner train-family CV).

All steps used the full 1,219-context training batch. Seeds only change the random portion of each slate and are near-pseudoreplicates (effective seed count approximately 1); Z and RP final utilities are bit-identical across all three seeds, and PW seeds 0 and 2 are identical. Do not interpret the three-seed spread as independent noise.

| Arm | Median final utility (family 95% CI) | Median final Δ (family 95% CI) | Elo Δ vs r785 | Minimum LOCO Δ | Verdict |
|---|---:|---:|---:|---:|---|
| H47-R | 1.067773 | -0.069822 [-0.147, -0.005] | -40.41 | -0.077643 | Significant loss |
| H47-Z | 1.063198 | -0.074397 [-0.150, -0.010] | -33.01 | -0.081508 | Significant loss |
| H47-PW | 1.109827 | -0.027768 [-0.066, +0.005] | -3.66 | -0.033506 | No win; CI includes 0 |
| H47-RP | 1.118971 | -0.018625 [-0.047, +0.001] | -21.97 | -0.019655 | No win; CI includes 0 |

The displayed delta intervals are the red-team cross-check's 10,000 family-bootstrap intervals over the 19 per-family deltas, around the median seed metric. The metrics artifact also retains the analyzer's full-precision bootstrap output for reproducibility.

## Per-seed final results

| Arm | Seed | DL-Val utility | Final Δ | Train utility | Train−val gap | Improved / worsened | LOCO min Δ | Best seen Δ (DL-Val selected) | Wall (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H47-R | 0 | 1.067773 | -0.069822 | 1.364936 | 0.297163 | 21 / 36 | -0.076881 | -0.002448 | 44.0 |
| H47-R | 1 | 1.068359 | -0.069236 | 1.362743 | 0.294384 | 22 / 36 | -0.076289 | -0.002448 | 43.1 |
| H47-R | 2 | 1.067020 | -0.070575 | 1.371924 | 0.304904 | 21 / 36 | -0.077643 | -0.002448 | 81.8 |
| H47-Z | 0 | 1.063198 | -0.074397 | 1.364380 | 0.301182 | 22 / 35 | -0.081508 | -0.003262 | 52.8 |
| H47-Z | 1 | 1.063198 | -0.074397 | 1.364248 | 0.301051 | 22 / 35 | -0.081508 | -0.003262 | 184.0 |
| H47-Z | 2 | 1.063198 | -0.074397 | 1.367274 | 0.304076 | 22 / 35 | -0.081508 | -0.003702 | 455.1 |
| H47-PW | 0 | 1.109827 | -0.027768 | 1.419460 | 0.309633 | 19 / 24 | -0.030561 | +0.000772 | 224.1 |
| H47-PW | 1 | 1.106915 | -0.030681 | 1.420598 | 0.313684 | 18 / 25 | -0.033506 | +0.000772 | 250.9 |
| H47-PW | 2 | 1.109827 | -0.027768 | 1.419238 | 0.309411 | 19 / 24 | -0.030561 | +0.003236 | 169.3 |
| H47-RP | 0 | 1.118971 | -0.018625 | 1.405630 | 0.286659 | 6 / 11 | -0.019655 | +0.000670 | 176.8 |
| H47-RP | 1 | 1.118971 | -0.018625 | 1.405610 | 0.286640 | 6 / 11 | -0.019655 | +0.000670 | 165.0 |
| H47-RP | 2 | 1.118971 | -0.018625 | 1.405585 | 0.286614 | 6 / 11 | -0.019655 | +0.000670 | 255.3 |

## Findings

RP is the best final arm (median Δ −0.018625, cross-check family CI [−0.047, +0.001]) but remains below r785 and fails the strict win rule. R and Z are significant losses by the cross-check intervals. PW is negative at the final checkpoint (median Δ −0.027768, cross-check family CI [−0.066, +0.005]). Its best-seen value was +0.003236 at PW seed2; that was one of 32 DL-Val checkpoint looks and is a winner's-curse result, not a win.

Train utility fell for every arm from r785’s 1.425042 (medians: R 1.364936, Z 1.364380, PW 1.419460, RP 1.405610). This continues H46’s evidence for surrogate/utility misalignment, not overfitting.

H47 has no working in-wave control because `arm == "0"` is dead code. Compare against H46-0’s last attempted candidate Δ −0.004087 (its accepted restore-on-reject incumbent remains r785 at Δ 0) and H46-A’s median final Δ −0.100828 under the same EMA protocol.

## Integrity and next waves

The cohort audit found 1219 train contexts / 205 families and 91 validation contexts / 19 families, with zero context or family overlap. Source-manifest exclusion counts: 93 DL-Holdout contexts excluded, 0 excluded payloads opened, and 0 outer rows delivered. All 12 runs used checkpoint SHA `ef6696f6369122a162d58b9b522fd83cae110a5576ef3760e3b43ba85d129e0a`, DOPE SHA `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`, and exactly 29,211 trainable adapter parameters. Per-seed family deltas, 10,000-resample CIs, all evaluation hashes, and full trajectories are in `metrics.json`.

H48 begins next: G diagnostic, 0 control, PF, and ES, with true family-balanced 256-context minibatch replicates, inner family-CV selection, and one terminal DL-Val score. H49 stays queued until H48 closes; its plan is in `~/jope_plan/ns/H48_STATUS.md`.
