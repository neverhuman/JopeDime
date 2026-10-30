# H46 — Gradient smoothing to break the DL-Val91 regression plateau (overnight wave 1)

Status: **RUNNING** (opened 2026-10-30 00:35 MDT; xbabe2 execution began 2026-10-30 00:34 MDT). Results are appended by a follow-up PR.
Owner ask: Jepson, 2026-10-30 00:02 MDT (overnight H46+ waves, 3 seeds, offline on a checkpoint COPY).

## Symptom (evidence from live summary.jsonl, rounds 775–844, xbabe1 run-20261030)
- DL-Val91 regression utility (higher is better): mean 1.1317, sd 0.0105, max 1.13760 (r785 = current best).
- Since r785 **every** candidate has been rejected and the loop takes action
  `restore_best_model_and_optimizer` (best_checkpoint = round-785.pt). Each round then does only
  `optimizer_steps=2` (AdamW lr 3e-4, wd 1e-4, batch 8192) from the SAME restored state.
- Consequence: candidate scores are a **deterministic cycle**, not noise: 1.13558 / 1.13544 / 1.13665 / 1.13443 /
  1.13700 / 1.13745 repeat exactly (identical per-context vectors). The "~4-round period" is the batch-order cycle
  replayed from one frozen starting point. The model can never move more than 2 steps from r785 → structural plateau.
- Update l2 ≈ 0.033/round; grad norm ≈ 0.45 < clip 1.0 (clipping inactive). Most of the 91 per-context scores are
  bit-identical across candidates: utility is an argmin-over-programs (discrete), so tiny steps rarely flip a selection.
- r785 accept was mainly one context recovering from a −2.0 failure score; one exact-tie accept was observed earlier.
- train−val gap (reg) ≈ 0.288.

## Hypotheses (ranked by expected value)
- **H46-A (multi-step + EMA)**: escaping restore-to-best requires accumulating many steps. Train N=64 steps continuously
  from r785 and evaluate a Polyak/EMA copy (decay 0.98) every 8 steps → EMA smooths the batch-cycle oscillation while
  letting the model actually travel.
- **H46-B (A + warmup/cosine + tighter clip)**: 4-step linear warmup to 3e-4 then cosine to 3e-5; clip 0.5.
- **H46-C (A + robust loss)**: Huber/clipped per-context loss (failure scores floored, per-context rank/z normalisation)
  so −2.0 failures don't dominate gradients.
- **H46-D (family-balanced robust loss)**: Huber/KL per-context loss weighted so each train family has equal total
  contribution.
- **H46-S (selection fix, applied post-hoc to every arm)**: accept only if the family-grouped paired bootstrap
  (2,000 resamples over the 91 DL-Val contexts, grouped by family) gives lower 95% CI of Δ > 0 AND Δ > 1e-4;
  exact ties are rejected.
- Control **H46-0**: live protocol replay (restore-on-reject, 2 steps/round) for the same wall budget.

## Setup
- Host xbabe2 (RTX 3090), worktree `~/JopeDime/.agent/worktrees/h46-gradsmooth`, branch `research/h46-gradient-smoothing`.
- Checkpoint COPY: `xbabe2:~/jope-data/h46/ckpt/round-785.pt` (sha256 ef6696f6…), plus round-844.pt, best_dev_checkpoint.json
  (evaluation_identity e79a7715…), input copy `~/jope-data/h46/input` (442 MB). The live loop on xbabe1 is NOT touched.
- Data: the approved regression 1,219 DL-Train / 91 DL-Val contexts only. Sealed meta-test and holdout never read.
  Binary deferred.
- DOPE frozen: `dope_freeze_guard.py` checksum must equal c51a56d0… at start and after every eval; only the 29,211
  adapter params train. Abort the arm on mismatch.
- Seeds: 0, 1, 2. The full 1,219-context batch is used every step, so seeds only change random slate details and
  observed results are near-pseudoreplicates (effective seed count approximately 1 per arm). Report the median final
  utility and a family-bootstrap interval over the 19 DL-Val families; never describe it as seed-noise uncertainty.
  Also report per-family deltas, improved/worsened contexts, regression Elo vs r785 (no AUC).
- GPU: hold `~/jope_plan/qualification.lock` (flock) per seed run; release between runs. CPU ≤85%, watch swap. Addendum: if that lock remains busy for more than 20 minutes, use CPU for every comparison with ≤16 threads, `nice -n 10`, host CPU ≤85%, and at least 20 GiB available RAM. Comparisons stay on one device.

## Execution record

- Runner: `research/flywheel_selfplay/h46/run.py`; serialized launcher: `research/flywheel_selfplay/h46/run_wave.sh`; result aggregation: `research/flywheel_selfplay/h46/analyze.py`.
- Uses the copied checkpoint and approved input packet, the pinned CPU-exact fused network, and the same `predict_costs` path as the full-cohort runner. Utility is the native reporter's fixed-cohort formula: `-fitness` for a passing gate, `-2` for a failed gate. No score-history sink is enabled.
- Device is `cuda:0`; the GPU lock is acquired per run and released between runs. DOPE guard is checked before training and before/after each validation. The 29,211 trainable-parameter guard passed.
- The control is 16 rounds × 2 steps, restoring the incumbent after every rejected validation. A/B/C each run 64 continuous steps, with EMA validation every 8 steps. All seed outputs are written under `~/jope-data/h46/runs/`; the wave status is maintained in `~/jope_plan/ns/H46_STATUS.md`.
- At 2026-10-30 00:46 MDT, H46-0 seeds 0 and 1 completed. Seed 2's first attempt completed training but failed in final utility batching because training pools have different sizes; no result was accepted. Evaluation now groups by pool size and the retry is queued behind another GPU lock holder. Seeds 0 and 1 both reached maximum candidate utility 1.135534 (Δ −0.002061). These are interim control observations, not an arm verdict. See the external status file for current lock/device and progress.

## Success criterion
An arm can qualify as a win only if its median **final** Δ vs r785 is > 0 and the family-bootstrap 95% CI lower bound
is > 0. Checkpoint selection must use inner family-grouped cross-validation over DL-Train families; DL-Val is scored
once at the end. H46 selected/checkpointed on DL-Val, so its trajectory comparisons are exploratory and cannot qualify
as selected wins. Do not use a seed-pooled interval: the seeds are near-pseudoreplicates, not independent replicates.

## Outcome / metrics / verdict / next step

H46 is complete on `cuda:0`; all 15 seed runs used the same device. No arm qualifies as a selected win. The required win rule is median **final** Δ > 0 with a family-bootstrap 95% lower bound > 0 and checkpoint selection by inner family-grouped CV on DL-Train. H46 checkpoint choices used DL-Val, so its trajectory results remain exploratory even where a per-checkpoint gate passed.

The intervals below resample the 19 DL-Val families and summarize the median seed metric. They are not seed-noise intervals: full-batch training made seed outputs near-pseudoreplicates (effective seed count approximately 1 per arm).

| Arm | Median final utility (family 95% CI) | Median final Δ (family 95% CI) | Regression Elo Δ vs r785 | Minimum LOCO Δ | Verdict |
|---|---:|---:|---:|---:|---|
| H46-0 | 1.137595 [0.720079, 1.653047] | +0.000000 [+0.000000, +0.000000] | +0.00 | +0.000000 | No gain; rejected proposals restore r785 |
| H46-A | 1.036767 [0.611815, 1.527791] | -0.100828 [-0.182550, -0.027974] | -51.58 | -0.108232 | Negative |
| H46-B | 1.053521 [0.615721, 1.567406] | -0.084074 [-0.159324, -0.022650] | -14.64 | -0.091292 | Negative |
| H46-C | 1.121032 [0.700803, 1.626509] | -0.016563 [-0.095663, +0.032493] | +33.01 | -0.020635 | No gain; final Δ negative |
| H46-D | 1.034436 [0.609825, 1.585219] | -0.103159 [-0.185021, -0.033009] | -40.41 | -0.110589 | Negative |

Per-seed final metrics (the corresponding 19-family intervals, per-family deltas, all evaluation hashes, and full H46-S trajectories are in `metrics.json`):

| Arm | Seed | DL-Val utility | Δ vs r785 | Train utility | Train−val gap | Improved / worsened | LOCO min Δ | Wall (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H46-0 | 0 | 1.137595 | +0.000000 | 1.425156 | 0.287561 | 0 / 0 | +0.000000 | 125.1 |
| H46-0 | 1 | 1.137595 | +0.000000 | 1.425156 | 0.287561 | 0 / 0 | +0.000000 | 76.2 |
| H46-0 | 2 | 1.137595 | +0.000000 | 1.425042 | 0.287447 | 0 / 0 | +0.000000 | 128.4 |
| H46-A | 0 | 1.036767 | -0.100828 | 1.354527 | 0.317760 | 17 / 35 | -0.108232 | 219.4 |
| H46-A | 1 | 1.037759 | -0.099836 | 1.352207 | 0.314448 | 18 / 35 | -0.107229 | 129.4 |
| H46-A | 2 | 1.036767 | -0.100828 | 1.355108 | 0.318341 | 17 / 35 | -0.108232 | 132.3 |
| H46-B | 0 | 1.053521 | -0.084074 | 1.358874 | 0.305353 | 18 / 26 | -0.091292 | 183.2 |
| H46-B | 1 | 1.076778 | -0.060817 | 1.363087 | 0.286310 | 18 / 26 | -0.067777 | 187.2 |
| H46-B | 2 | 1.053521 | -0.084074 | 1.360786 | 0.307264 | 18 / 26 | -0.091292 | 269.5 |
| H46-C | 0 | 1.121360 | -0.016235 | 1.453722 | 0.332362 | 20 / 11 | -0.020250 | 339.3 |
| H46-C | 1 | 1.120980 | -0.016615 | 1.454357 | 0.333377 | 19 / 11 | -0.020635 | 228.4 |
| H46-C | 2 | 1.121032 | -0.016563 | 1.453634 | 0.332602 | 20 / 11 | -0.020582 | 265.2 |
| H46-D | 0 | 1.034436 | -0.103159 | 1.354700 | 0.320263 | 19 / 34 | -0.110589 | 216.3 |
| H46-D | 1 | 1.034436 | -0.103159 | 1.351160 | 0.316724 | 19 / 34 | -0.110589 | 57.5 |
| H46-D | 2 | 1.040231 | -0.097364 | 1.354432 | 0.314201 | 22 / 31 | -0.104729 | 222.5 |

### Findings and limitations

A/B show objective mismatch, not ordinary overfitting: A seed0 training loss fell 3.69→3.36 while train utility fell 1.425156→1.354527 and DL-Val Δ was −0.100828. B likewise ended below the r785 train utility and validation utility. H46-D seed0 was also clearly negative (Δ −0.103159, family CI [−0.186141, −0.030640], LOCO range [−0.110589, −0.075359]); this after-seed0 hopeless cutoff was adopted after observing A and is not prespecified.

H46-C final Δ was −0.016235/−0.016615/−0.016563, hence **no gain**. Step48’s approximately +0.0173 was selected from five arm trajectories × eight standard EMA checkpoints using DL-Val (40 looks); it is a winner’s-curse result without multiplicity adjustment. The H46-S per-checkpoint pass is exploratory, not an arm-level result.

The independent red-team audit found 0 context overlap and 0 family overlap (1,219 train contexts across 205 families; 91 validation contexts across 19 families), 93 excluded DL-Holdout contexts, 0 excluded payloads opened, and 0 outer rows delivered. Every run used checkpoint SHA `ef6696f6…`, DOPE SHA `c51a56d0…`, and exactly 29,211 trainable adapter parameters. The full hashes and per-evaluation guards are recorded in `metrics.json`.

H46-0 accepted no candidate: its restore-on-reject incumbent is r785 at Δ 0. Its last attempted proposal was Δ −0.004087, the comparison point requested for H47. The per-arm Elo is pairwise regression Elo, not AUC; H46-C’s positive Elo reflects pairwise wins/ties and does not override its negative mean utility.

### H47 and H48 sequence
H47 remains queued for comparability with H46-A. Its full-batch default also makes seeds near-pseudoreplicates; report effective seed count approximately 1. H47 has no functioning in-wave control because the `arm == "0"` branch is dead code, so compare explicitly against H46-0's last attempted candidate Δ −0.004087 (its restore-on-reject accepted incumbent remains r785 at Δ 0) and H46-A under the same EMA protocol.

After H47, proposed H48 arms on an offline round-785 copy are: (0) H46-0 control; (P) persistent AdamW and EMA state across rounds with no reset to r785 on rejection; (PC) P plus per-context clipped/normalized loss, capping the −2.0 gate-failure contexts; and (PF) P plus family-balanced minibatches of 256 with seed-varied sampling. Select checkpoints by inner family-grouped CV over train families and score DL-Val91 once at the end. Keep the same lock/RAM guards and report median utility, family-bootstrap CI, Elo, train/validation gap, wall time, and checksums.
