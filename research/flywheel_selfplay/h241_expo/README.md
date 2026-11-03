# H241: zero-update adapter extrapolation along the accepted trajectory

**Status: complete. Verdict: NO_ADVANCE.** The rule was preregistered in [PREREG.md](PREREG.md) (PR #2082, reviewed
head `b30c9316`, merged as `df7ff4b7` at 03:02:55Z) before any study outcome was read. The single attempt ran on
xbabe2 CPU from 03:08:48Z to 03:36:56Z with **0 optimizer updates**.

**Extrapolating past r882 does not help.**
- **The selected-α rule.** It kept r882 in all 30 cell decisions (15 cells × 2 directions). Both directions therefore
  tie r882 exactly on DL-Val91: gain 0 [0, 0], 0/3 seeds positive, Holm p 1.0, Elo 1500.
- **The fixed grid.** Every fixed extrapolation point loses on DL-Val91, and the loss grows with α.
  - **r782 direction.** −0.0024 at α = 0.25, −0.0051 at α = 0.5 and −0.0079 at α = 1.0. At α = 1.0 the CI95 excludes 0
    and 7 of 19 families are harmed.
  - **r859 direction.** −0.0002 to −0.0032.

## Hypothesis

**The candidate.** θ(α) = r882 + α·(r882 − r_prev) moves past the last accepted adapter along the accepted flywheel
trajectory. This is ExPO-style extrapolation (Zheng et al. 2024), and it costs 0 training updates.

**Why it was new.** Earlier waves covered interpolation soups (H152), EMA/SWA, ensembles and distillation. None
extrapolated.

**The question.** Does an extrapolated adapter lift DL-Val91 regression utility over r882 by more than +0.002?

## Setup

- **Policies.** r_prev is **r782 (primary)** or r859 (secondary). α ∈ {0.25, 0.5, 1.0}, plus CONTROL α = 0, which is
  r882 bit for bit. That gives 7 policies.
  - **Inputs.** The r_prev adapters are SHA-verified copies of the exact rounds H152 used:
    - r782 `427b0d46…`;
    - r859 `5802245f…`;
    - r882 `0084479a…`.
  - **What moves.** Only the 47 adapter tensors (29,211 parameters) are extrapolated, in float64. The non-adapter
    tensors of r782 and r859 were verified bitwise identical to r882's.
- **Selection.** α was chosen per (seed, outer) cell on DL-Train FIT folds only. That is the 4 non-outer folds of
  fresh family folds: seeds 241101–241103, fold seeds 241141–241143, 15 cells.
  - **Selector.** The strict H237 selector: gain over α = 0 above .001 **and** family LCB95 above 0, else r882.
  - **Lock.** All decisions were fsync-locked (`selection.json`, `c9147d1e…`) before any extrapolated policy was
    scored on DL-Val91. Each policy was scored once.
- **Decision.** It is made on DL-Val91 (91 contexts, 19 families, policy mixture over each seed's 5 cells). A
  direction advances only with all of:
  - median gain > +0.002;
  - family LCB95 > 0;
  - Holm p < .05 over the 2 directions;
  - ≥ 2/3 seeds positive.
- **Engine.** The merged H237 engine with its cached frozen anchor. The Native91 regression passed first: 91/91
  contexts, mean exactly 1.1377941898164525, and the cached forward is bit-exact to native r882. CONTROL's DL-Val
  vector equalled the native baseline exactly.
- **Resources.**
  - xbabe2 cores 44–47, nice 19, one math thread.
  - Peak tree RSS 3.24 GB, 2 threads.
  - Elapsed 1,687 s.
  - DOPE `c51a56d0` was unchanged before and after.
- **Memory guard (H239 A1).** One cooperative pause at 03:12:23Z, when MemAvailable fell to 19.9 GiB. The run resumed
  at 03:28:45Z after MemAvailable stayed ≥ 25 GiB for 30 s, about 982 s paused in total. There was no hard stop and no
  resume.

## Outcome

### Decision panel (DL-Val91; Elo with r882 = 1500)

| Contrast | Per-seed gain | Median [CI95] | Seeds > 0 | Holm p | Elo | Advance |
|---|---|---|---:|---:|---:|---|
| **EXPO782 − r882 (primary)** | 0, 0, 0 | **0.00000 [0, 0]** | 0/3 | 1.0 | 1500 | no |
| EXPO859 − r882 | 0, 0, 0 | 0.00000 [0, 0] | 0/3 | 1.0 | 1500 | no |
| CONTROL − r882 | 0, 0, 0 | 0.00000 [0, 0] | 0/3 | — | 1500 | — |

- **Selected α.** It was 0 in 15/15 cells for both directions.
- **Harm.** 0 harmed families, and 0 invalid DL-Val picks for every policy.
- **Plateau.** Every arm sits at r882's level, 1.1377942: 0.00116 below the plateau best 1.1389572 (r3175).

### Fixed grid points (descriptive, never admitted)

| Point | L2 to r882 | DL-Val91 gain [CI95] | Families W/T/L | Elo | DL-Train gain [CI95] (TRAIN-exposed) |
|---|---:|---|---|---:|---|
| R782_A0.25 | 0.0369 | −0.0024 [−0.0092, +0.0021] | 1 / 16 / 2 | 1496 | −0.0021 [−0.0056, +0.0000] |
| **R782_A0.5** (designated primary point) | 0.0737 | **−0.0051 [−0.0141, +0.0013]** | 1 / 15 / 3 | 1489 | +0.0036 [−0.0050, +0.0172] |
| R782_A1 | 0.1474 | −0.0079 [−0.0175, −0.0003] | 1 / 11 / 7 | 1478 | −0.0079 [−0.0248, +0.0106] |
| R859_A0.25 | 0.0048 | −0.0002 [−0.0007, 0.0000] | 0 / 18 / 1 | 1496 | −0.0001 [−0.0003, +0.0000] |
| R859_A0.5 | 0.0096 | −0.0032 [−0.0094, 0.0000] | 0 / 17 / 2 | 1493 | +0.0003 [−0.0002, +0.0011] |
| R859_A1 | 0.0192 | −0.0032 [−0.0094, 0.0000] | 0 / 17 / 2 | 1493 | +0.0001 [−0.0005, +0.0010] |

- **Distances.** L2 = α·‖r882 − r_prev‖, with ‖r882 − r782‖ = 0.147 and ‖r882 − r859‖ = 0.019.
- **Picks.** On DL-Val91, no point made an invalid pick. On DL-Train, invalid picks ranged from 17 to 19, against
  r882's 17.

### Out-of-fold and unconstrained readouts (descriptive)

- **DL-Train out of fold.** All contrasts are exactly 0, because r882 was selected everywhere.
- **Without the selector's acceptance gate.** The FIT-best α was 0.5 in 12/15 cells for each direction. The mean FIT
  gain of R782_A0.5 was +0.004 to +0.0065 per cell, but its LCB95 was always below 0, so the selector correctly
  refused it.
- **What those refused choices would have scored** on DL-Val91:
  - EXPO782: −0.0040 [−0.0113, +0.0011];
  - EXPO859: −0.0026 [−0.0075, 0].

## Reading

- **The accepted trajectory is not a direction of further improvement.** Stepping past r882 along r882 − r782 lowers
  DL-Val91 utility monotonically in α. At α = 1 the loss is significant and spread across 7 families. The r859
  direction is short (the adapter moved only 0.019 between r859 and r882) and yields a smaller, still non-positive
  effect.
- **TRAIN and val disagree, as in H152.** On DL-Train the r782 direction gives a weak, noisy gain at α = 0.5 (+0.0036,
  CI spanning 0), echoing r782's known TRAIN advantage. The same point loses on val.
- **The strict selector did its job.** It never accepted a noisy FIT gain, so the selected rule exactly ties r882 and
  causes no harm.
- **Interpolation and extrapolation both fail.** Together with H152 (interpolation, α ∈ [0, 1] toward r782),
  weight-space movement along the accepted r782→r882 line fails on both sides of r882. r882 sits at a local optimum
  of DL-Val91 along that line.

## Verdict

**NO_ADVANCE.**
- Both directions tie r882 exactly: median 0, LCB 0, Holm p 1.0, 0/3 seeds.
- No fixed point beats r882 on DL-Val91.
- No serving or live-flywheel change. DOPE `c51a56d0` is unchanged.

## Next step

**Do not adopt extrapolation as a candidate generator for this adapter.** Zero-update weight-space moves have now
failed on this line in both directions:
- interpolation (H152);
- extrapolation (H241);
- EMA/SWA.

Six update-rule variants (H170–H237) also tie with or lose to r882.

**What could still move the plateau** is new signal, not new geometry: new data or families, or a better base model.
No follow-up extrapolation wave is proposed. Possible one-off checks would be extrapolating along a different
accepted pair, or a gate-aware val estimand. Neither is planned.

## Files and custody

- **In this directory:**
  - [metrics.json](metrics.json): the compact readout, under 100 KB, with `dope_before`/`dope_after`, the pause
    events and the full-metrics pointer;
  - [PREREG.md](PREREG.md);
  - [runner.py](runner.py), [expo.py](expo.py), [analysis.py](analysis.py) and [monitor.py](monitor.py);
  - [test_h241.py](test_h241.py): 17 data-free tests;
  - [manifest.json](manifest.json), [fold_manifest.json](fold_manifest.json) and [provenance.json](provenance.json).
- **S3, create-only with sha256 and VersionId readback:**
  - `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H241/attempt-001/` holds:
    - the full `metrics.json` (215,506 B, sha256 `d78db674…`);
    - `selection.json`;
    - the resource summary and the attempt claim;
    - `custody-attempt-001.tar.gz`, which contains every per-policy evaluation file, the panels, the stdout, the
      resource samples and both smokes;
    - `s3_manifest.json` (sha256 `14f8b61a…`, VersionId `0l3IPzLM…`).
  - The prereg sources and smokes are under `H241/prereg-b30c9316/` (14 objects).
- **Not in Git.** No checkpoint or large object. The extrapolated adapters are deterministic from the pinned inputs.
