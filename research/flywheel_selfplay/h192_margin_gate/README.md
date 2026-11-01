# H192 — strict-margin (confidence-gated) adapter selection, 0 updates

**Status: complete. Preregistered verdict: `NO_ADVANCE`.**
- The protocol is [PREREG.md](PREREG.md), from PR #1584, merged as `81df933f`.
- rd_argon's independent review passed at `b3967f24` (review 5728980376).
- The run was one supervised attempt on xbabe2 CPU with **0 updates**.
- The first launch, at 16:29Z, was refused by the supervisor before anything ran, because the disk was at
  93.808% against the 93.8% stop. No worker started and the attempt was not consumed. The same frozen launch was
  started once the disk fell to 93.59%.

**Gating the adapter never helps, and it gets steadily worse as the gate tightens.** Even r882's lowest-margin
overrides of the anchor-only pick are worth taking. Blocking them brings back invalid (−2.0) picks.

## Hypothesis

H180, H184 and H188 found that the adapter's value is its revealed-outcome kernel steering away from −2.0 picks.
H192 asked whether trusting the adapter's re-ranking **only when it is confident** would remove noisy low-margin
overrides. Concretely: keep the anchor-only base pick unless r882's full pick wins by a normalized cost margin
greater than τ.

## Setup

- **Definitions** (exact, from the code):
  - **Base pick** = `argmin anchor_total`, the H180 anchor-only scorer. The live decomposition was proved on all
    1,310 contexts.
  - **Full pick** = r882's argmin.
  - **Margin** = `(full[base] − full[top1]) / std(full)`.
  - **τ = 0 is R882**, bit-exact on every TRAIN context and on DL-Val against the pinned reference.
- **Disagreement.** The base and full picks disagree on **697 of the 1,219 TRAIN contexts (57%)**.
- **τ grid.** {0, q25, q50, q75, q90} of the disagreement margins on each cell's FIT families. For example, seed
  192101 fold 0 had {0, 0.037, 0.097, 0.157, 0.217}.
- **Selection.** τ\* = argmax of the equal-family utility on B, with ties going to τ = 0. **The lock
  (`3a4c4466…`) was fsynced before any A or DL-Val readout.**
- **Folds.** Seeds 192101–192103 used distinct fold seeds, 192141–192143. Each family was assessed once per seed.
- **Primary.** MGATE against R882 with m = 1: the H177 pooled rule, the median gain with a 10,000-draw paired
  family bootstrap (axis 0), and the H178 receipt gate.
- **Regression.** **r882 reproduced 91/91 DL-Val contexts exactly** (mean 1.1377941898164525).
- **Integrity.** DOPE `c51a56d0` and the adapter hash were unchanged.

## Outcome (205 families, pooled A)

### Primary: MGATE against R882

| Statistic | Value |
|---|---|
| H177 pooled rule | mean gain −0.0048 [−0.0153, +0.0064], p = 0.81, **not admitted** |
| Median of seeds [95% CI] | **−0.0042 [−0.0177, +0.0066]**; per seed −0.0042, −0.0133, +0.0030 |
| Elo against R882 [95% CI] | 1501.6 [1496.9, 1505.6] |
| −2.0 picks per seed | 20, 20, 16 (R882: 17) |
| τ\* chosen, across the 15 cells | τ = 0 in **8**, q25 in 5, q50 in 1, q75 in 1, q90 in 0 |
| H178 receipt gate | passes |

Verdict gates: the selector does not admit MGATE, the median gain is below +0.001, and the lower bound is below 0.
So the verdict is **`NO_ADVANCE`**.

### Descriptive: each fixed τ applied in every cell

| τ | Gain vs R882, median of seeds [95% CI] | −2.0 picks per seed | Share of disagreements that use the full pick |
|---|---|---|---|
| 0 (R882) | 0 | 17 / 17 / 17 | 100% |
| q25 | −0.0003 [−0.0227, +0.0268] | 21 / 20 / 20 | 75% |
| q50 | −0.0120 [−0.0409, +0.0195] | 24 / 24 / 23 | 50% |
| q75 | −0.0280 [−0.0645, +0.0103] | 28 / 28 / 28 | 25% |
| q90 | −0.0395 [−0.0775, +0.0014] | 31 / 30 / 30 | 11% |

**DL-Val (91 contexts, descriptive),** using the full-TRAIN τ grid: q25 −0.040, q50 −0.038, q75 −0.096 and
q90 −0.096. The number of −2.0 picks rises from 0 to 3.

**Seeds.** At 0 updates, per-context utilities are shared across seeds, and the seeds only reshuffle folds. The
family bootstrap CI is the inference, as disclosed.

## Reading

- **Low-margin overrides are signal, not noise.** As τ rises, the gate blocks more of r882's overrides, utility
  falls monotonically, and invalid picks rise monotonically (17 → 31). Many of the adapter's small-margin re-ranks
  are the ones that avoid a −2.0 outcome, so suppressing them gives back exactly the invalid picks the kernel
  exists to prevent.
- **B selection learned to prefer little or no gating.** It chose τ = 0 or q25 in 13 of 15 cells. The pooled MGATE
  result, about −0.004 with a CI spanning 0, is the cost of occasionally picking a stricter τ on a noisy 41-family
  B fold.
- **Together with H180, H184 and H188:** the adapter's revealed-outcome kernel should be trusted fully. Neither
  removing it, changing its inputs, nor gating its use beats r882.
- **Scope.** These are TRAIN families r882 has historically seen, at 0 updates; this is conditional evidence.

## Next step

**No adoption, and nothing to flag** to the `flywheel_gp_fused` owner. r882 and the live flywheel are unchanged.

**Confidence gating of the adapter is closed as a lever.** The levers that remain are those that add information:
- more revealed outcomes per context, if the protocol allows;
- new candidate features.

Each would need its own preregistration.

## Files and custody

- **Files here.**
  - [metrics.json](metrics.json): the run's metrics, 12.8 KB, including the lock hash, τ grids, τ\* per cell and
    override rates.
  - Code: [runner.py](runner.py), [treatment.py](treatment.py), [analysis.py](analysis.py) (imports the merged
    H188 helpers), [monitor.py](monitor.py) and [test_h192.py](test_h192.py), 8 data-free tests.
  - [preflight.json](preflight.json).
- **S3** (create-only; `manifest.json` lists the SHA-256 of each file):
  `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H192/opus-h192-20261101a/`. It holds:
  - the run outputs: panels, gates, the lock, the audit and resource samples;
  - the smoke evidence;
  - `launch.json` and the attempt claim.
- **Resources.**
  - 1,323 s of wall time.
  - Peak: 2 threads, 3.28 GB RSS, host CPU 42%, load 30, disk 93.62% (the stop is at 93.8%).
  - 0 refused or protected reads, no GPU.
