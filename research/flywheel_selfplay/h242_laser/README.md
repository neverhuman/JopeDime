# H242: zero-update LASER-style spectral rank reduction of the r882 adapter

**Status: complete. Verdict: NO_ADVANCE.** The rule was preregistered in [PREREG.md](PREREG.md) (PR #2085, reviewed
head `db62164f`, merged as `86590e8c` at 04:05:27Z) before any study outcome was read. The single attempt ran on
xbabe2 CPU from 04:06:08Z to 04:22:12Z with **0 optimizer updates**.

**Rank truncation does not denoise r882.**
- **The decision arms.** Neither advances. SVD_SELECT and LASER_SINGLE − r882 both have median 0 [0, 0], 0/3 seeds
  positive, Holm p 1.0 and Elo 1500.
  - **SVD_SELECT** accepted a truncation (ρ = .90) in only 1 of 15 cells, and that one cell cost its seed −0.0055 on
    DL-Val91.
  - **LASER_SINGLE** kept r882 in all 15 cells.
- **Truncation is no better than the placebo.** On DL-Train the truncations' apparent gains (+0.012 to +0.013) are
  matched by the RANDPROJ placebo at the same Frobenius change (+0.012 to +0.013). On DL-Val91, every large loss is
  the same H152 gate-cliff context, and the placebo flips it too.

## Hypothesis

**The candidate.** Following LASER (Sharma et al. 2023), remove the small-singular-value components of r882's own
2-D adapter matrices. That is zero-training denoising.

**The question.** Does the rank-k reconstruction keeping the energy fraction ρ of Σσ² lift DL-Val91 regression
utility over r882 by more than +0.002?

**Why it was new.** It is the first spectral or low-rank wave. H160 projected onto a trajectory subspace and never
truncated rank.

## Setup

- **The reduction.**
  - **Which matrices.** 16 two-dimensional adapter matrices, with shapes, full ranks and kept ranks frozen in
    [spectral_inventory.json](spectral_inventory.json). Three have rank 1 and never change.
  - **Untouched.** Biases, 1-D and 3-D tensors, the frozen anchor and DOPE are untouched.
  - **Census check.** The runner recomputed the census from the r882 copy and found it equal to the frozen list.
  - **Kept ranks.** At ρ = .99 / .95 / .90 the kept ranks are, for example, 31 / 27 / 23 of 32 for the attention
    `in_proj` and 46 / 39 / 34 of 48 for `parent_delta_net.0`.
- **The policies** (20 in all):
  - r882, which is ρ = 1 and bit-exact;
  - RHO99 / RHO95 / RHO90, which change every 2-D matrix;
  - the RAND99 / RAND95 / RAND90 placebos: a seeded random orthogonal direction set with the same per-matrix
    Frobenius change, verified to 1e-5;
  - 13 S90 single-matrix truncations at ρ = .90.
- **The arms:**
  - **SVD_SELECT (primary):** ρ chosen per FIT cell.
  - **LASER_SINGLE (exploratory):** the best single matrix on FIT.
  - **RANDPROJ (report-only placebo):** the placebo at SVD_SELECT's chosen ρ.
  - **CONTROL:** r882.
- **Selection.** Strict selection on DL-Train FIT folds: fold seeds 242141–242143, seeds 242101–242103, 15 cells.
  The gain over r882 must be above .001 **and** the family LCB95 above 0, else r882. Every decision was
  fsync-locked (`selection.json`, sha256 `c090ba78…`) before any truncated policy was scored on DL-Val91. Each policy was
  scored once.
- **Decision.** It is made on DL-Val91 (91 contexts, 19 families, policy mixture over each seed's 5 cells). A
  decision arm advances only with all of:
  - median gain > +0.002;
  - family LCB95 > 0;
  - Holm p < .05 over SVD_SELECT and LASER_SINGLE;
  - ≥ 2/3 seeds positive.
- **Engine.** The merged H241 zero-update harness. The Native91 regression passed first: 91/91 contexts, mean exactly
  1.1377941898164525, and the cached forward is bit-exact to native r882. CONTROL's DL-Val vector equalled the native
  baseline.
- **Resources.**
  - xbabe2 cores 44–47, nice 19, one math thread.
  - Peak tree RSS 3.34 GB, 2 threads.
  - Elapsed 964 s, with no memory or load pause.
  - DOPE `c51a56d0` was unchanged before and after.

## Outcome

### Decision panel (DL-Val91; Elo with r882 = 1500)

| Contrast | Per-seed gain | Median [CI95] | Seeds > 0 | Holm p | Elo | Advance |
|---|---|---|---:|---:|---:|---|
| **SVD_SELECT − r882 (primary)** | 0, 0, −0.0055 | **0.00000 [0, 0]** | 0/3 | 1.0 | 1500 | no |
| LASER_SINGLE − r882 | 0, 0, 0 | 0.00000 [0, 0] | 0/3 | 1.0 | 1500 | no |
| RANDPROJ − r882 (placebo) | 0, 0, −0.0047 | 0.00000 [0, 0] | 0/3 | — | 1500 | — |
| RANDPROJ − SVD_SELECT (placebo) | 0, 0, +0.0008 | 0.00000 [0, 0] | 1/3 | — | — | — |

- **SVD_SELECT.** It selected r882 in 14 cells and RHO90 in 1 cell: seed 242103, outer 2, FIT gain +0.0187 with LCB
  +0.0008. That made seed 242103's mixture −0.0055, with 4 harmed families.
- **LASER_SINGLE.** It selected r882 in 15/15 cells. Its FIT-best matrix was `prog_412_bottleneck.0` in 8 cells and
  `cand_query_proj.0` in 7, but the LCB95 was always below 0.
- **Plateau.** The context means are 1.137794, 1.137794 and 1.132073 for SVD_SELECT across seeds, against the plateau
  best 1.1389572 (r3175).

### Fixed policies (descriptive, never admitted)

| Policy | L2 to r882 | DL-Val91 gain [CI95] | W/T/L | Elo | DL-Train gain [CI95] (TRAIN-exposed) |
|---|---:|---|---|---:|---|
| RHO99 | 1.142 | −0.0026 [−0.0094, +0.0019] | 1/14/4 | 1488 | −0.0022 [−0.0079, +0.0015] |
| RHO95 | 2.962 | −0.0263 [−0.0966, +0.0124] | 5/11/3 | 1499 | +0.0132 [−0.0013, +0.0354] |
| RHO90 | 4.408 | −0.0274 [−0.0974, +0.0111] | 5/10/4 | 1497 | +0.0120 [−0.0030, +0.0343] |
| RAND99 (placebo) | 1.142 | −0.0339 [−0.1006, +0.0001] | 3/12/4 | 1491 | +0.0013 [−0.0106, +0.0149] |
| RAND95 (placebo) | 2.962 | −0.0260 [−0.0977, +0.0141] | 6/8/5 | 1496 | +0.0123 [−0.0202, +0.0457] |
| RAND90 (placebo) | 4.408 | −0.0235 [−0.0971, +0.0192] | 7/6/6 | 1497 | +0.0127 [−0.0118, +0.0413] |
| S90 `prog_412_bottleneck.0` | 0.710 | −0.0301 [−0.0991, +0.0067] | 3/12/4 | 1490 | +0.0047 [−0.0040, +0.0149] |
| S90 `cand_query_proj.0` | 0.976 | −0.0318 [−0.1009, +0.0059] | 2/13/4 | 1490 | +0.0053 [−0.0125, +0.0285] |
| S90 `traj_cross_attn.in_proj` | 2.212 | −0.0342 [−0.1023, +0.0012] | 1/14/4 | 1489 | +0.0012 [−0.0007, +0.0044] |
| S90 `mm_cross_attn.in_proj`, `hist_token_proj.0`, `traj_delta_head.0` | 0.97–2.27 | −0.0002 to −0.0001 | 0/18/1 | 1496 | ≈ 0 |
| S90 other 6 matrices* | 0.14–1.22 | 0 (identical picks) | 0/19/0 | 1500 | 0 |

\* The 6 matrices are `modality_type_emb`, `mm_cross_attn.out_proj`, `ctx_bilinear_u`, `prog_bilinear_v`,
`mm_prior_out.0` and `parent_delta_net.0`. Truncating any of them at ρ = .90 changes no pick on either split.
`mm_prior_out` and `parent_delta_net` are the branches H239 identified as unused.

- **Invalid picks.** On DL-Val91, every policy that flips the cliff context makes 1 invalid pick, against 0 for r882.
  On DL-Train, invalid picks drop slightly under truncation: 12–15, against r882's 17.

### Where the DL-Val losses come from (post hoc, not preregistered)

- **One context.** Every fixed policy that loses more than 0.02 does so on the same val context: #27, family
  `fam-9f67f144…`. This is the gate-cliff context identified in H152. It drops from +1.1227 to an invalid pick, a
  change of −3.123.
- **The placebo too.** RAND99, RAND90 and three S90 policies flip it as well.
- **Without context #27,** the other 90 contexts move by a summed +0.52 (RHO90) and +0.62 (RHO95), but also by +0.89
  for the RAND90 placebo. That is a descriptive check and no advantage for truncation over a random perturbation of
  the same size.

## Reading

- **No evidence of spectral noise in r882's adapter.**
  - Removing the small-singular-value tail does not beat the matched random-direction placebo on DL-Train (+0.012 to
    +0.013 for both) or on DL-Val91. LASER's premise, that the tail holds harmful noise, does not show up here.
  - Truncation and placebo behave alike: both perturb r882 by the same Frobenius amount and get the same apparent
    TRAIN lift.
- **Val remains cliff-driven** (H139/H143/H152). Any perturbation of about 1 or more in adapter L2 that touches the
  candidate/program pathway flips context #27 across the gate cliff, and that one context dominates the 19-family
  estimate.
  - **The strict selector held.** It protected the decision in 29/30 cell decisions.
  - **The one cell it accepted** passed with a FIT LCB of +0.0008, and it lost on val.
- **Several matrices are inert at this scale.** Six of them change no picks at all when truncated to ρ = .90.

## Verdict

**NO_ADVANCE.**
- SVD_SELECT and LASER_SINGLE tie r882: median 0, LCB 0, Holm p 1.0, 0/3 seeds.
- No fixed policy beats r882 on DL-Val91.
- The placebo matches the truncations.
- No serving or live-flywheel change. DOPE `c51a56d0` is unchanged.

## Next step

**Do not adopt rank reduction.** Zero-update weight-space candidate generators now include:
- interpolation (H152);
- extrapolation (H241);
- EMA/SWA;
- spectral truncation (H242).

All of them tie with or lose to r882, and on val they are dominated by the same gate-cliff context. **Further
weight-space geometry is unlikely to pay** on this panel.

What could move the decision:
- **Validation.** A preregistered gate-aware or cliff-robust val estimand, which is a GOV/protocol decision.
- **New signal:** new families or data, or a better base model.

No follow-up truncation wave is proposed.

## Files and custody

- **In this directory:**
  - [metrics.json](metrics.json): the compact readout, under 100 KB, with `dope_before`/`dope_after`, the full
    spectral census, the Frobenius change per ρ and the full-metrics pointer;
  - [PREREG.md](PREREG.md) and [spectral_inventory.json](spectral_inventory.json);
  - [runner.py](runner.py), [spectral.py](spectral.py), [analysis.py](analysis.py) and [monitor.py](monitor.py);
  - [test_h242.py](test_h242.py): 15 data-free tests;
  - [manifest.json](manifest.json), [fold_manifest.json](fold_manifest.json) and [provenance.json](provenance.json).
- **S3, create-only with sha256 and VersionId readback:**
  - `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H242/attempt-001/` holds:
    - the full `metrics.json` (386,670 B, sha256 `44cb8f31…`);
    - `selection.json`;
    - the resource summary and the attempt claim;
    - `custody-attempt-001.tar.gz`, which contains all 40 evaluation files, the panels, the stdout, the resource
      samples and both smokes;
    - `s3_manifest.json` (sha256 `35f6cdeb…`, VersionId `TNRlWBBy…`).
  - The prereg sources and smokes are under `H242/prereg-db62164f/` (15 objects).
- **Not in Git.** No checkpoint or large object. Every truncated or placebo adapter is deterministic from the pinned
  r882 copy.
