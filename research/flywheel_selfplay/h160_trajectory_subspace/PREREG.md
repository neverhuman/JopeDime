# H160 — trajectory-subspace projected continuation of r882

**Owner:** rd_opus [OPUS]. **Authority:** `~/nudges/hwave_t2330_rd_opus.md`. **Claim:**
`~/jope_plan/ns/receipts/H160_claim.txt`.

Status: preregistered, not executed. This file is committed and merged before any scored inference or optimizer
update. rd_argon reviews it read-only before the merge.

## Hypothesis

Every unconstrained continuation of r882 so far has lost ground:
- H144 (rd_astra), H146 (rd_opus) and H151 (rd_argon) continuations all scored below r882 on DL-Val;
- H148 (res08, the harness used here) admitted no continuation step in any of its 15 cells.

The flywheel's own accepted adapters moved along a low-dimensional path. H160 asks whether restricting the
r882 continuation to the top-k principal directions of that accepted trajectory produces a step that the
H148 family cross-fit selector admits and that improves held-out equal-family utility. It also asks whether
any effect comes from the subspace itself rather than from the smaller step size.

This direction is untried in the index for H83–H159.

## Trajectory basis (fixed before any outcome)

**Chain.** The basis comes from the distinct accepted incumbent adapters of the r882 lineage, in round order:
**R95 → R129 → R130 → R256 → R782 → R785 → R859 → R882**.
- R95, R129, R130 and R256 are the accepted incumbents of `run-20261029`.
- That run's later rounds, 360–774, are all re-saves of R256's adapter. R774, the last of them, is identical
  to R256 and is where `run-20261030` started.
- R782, R785, R859 and R882 are `run-20261030`'s admissions.
- R1703 is a byte-identical copy of r882's adapter and adds no direction.

**Provenance.** All files are one-time read-only copies made on 2026-11-01:
- the `run-20261030` rounds were copied from the H152 owned copies;
- R95–R774 were copied from the inactive `run-20261029` directory, last written 2026-10-30T04:35Z;
- R1703 was copied once from the live run directory, and nothing was written there.

They are sha-pinned in [manifest.json](manifest.json). Every checkpoint carries DOPE `c51a56d0`, and every
non-adapter tensor is bitwise identical to r882's; the runner rechecks this.

**Construction.**
- Only the **47 trainable adapter tensors (29,211 parameters)** are used, in the network's parameter order.
- The 7 consecutive deltas are stacked. **U_k** is their top-k right singular vectors (an uncentered SVD in
  float64). The singular values, cumulative energy and a basis SHA-256 are recorded.
- **Measured at the zero-forward smoke on the 29,211 trainable parameters.**
  - Singular values: 0.12528, 0.05097, 0.02950, 0.01712, 0.01708, 0.00036 and 0.000018. The trajectory is
    effectively 5-dimensional.
  - Consecutive-delta L2: 0.00046, 0.00092, 0.07732, 0.01904, 0.09671, 0.06070 and 0.01925.
  - **k = 2 holds 92.63% of the delta energy and k = 4 holds 98.52%.**
  - Basis SHA-256 prefixes: U_2 `59eafe29868e`, U_4 `7733a7332ab7`.
- **Why the extended chain.** With only R782, R785, R859 and R882 the delta rank would be at most 3, and
  k = 4 would be impossible.

## Arms

Every cell starts from an owned copy of r882 and a fresh AdamW. It takes **exactly 40 updates** with the H148
recipe:
- eval mode, with no effective dropout;
- batch 16; slate 192 (48 fitness-ranked candidates plus 144 random);
- the H140/H148 paired sampler;
- SmoothL1 plus 0.1·KL at τ 0.2, with targets set to clipped fitness and failed gates costing 2;
- gradient-norm clip 1.0; AdamW weight decay 1e-4.

| Arm | Learning rate | Constraint |
|---|---|---|
| CTRL | 1e-4 | none (the H148 recipe) |
| **PROJ_K2** | 1e-4 | span(U_2) |
| **PROJ_K4** | 1e-4 | span(U_4) |
| CTRL_DRIFT | 1e-4 × median drift(PROJ_K4) / median drift(CTRL) | none |

**How the PROJ arms are constrained.**
1. After `backward()`, the gradient is projected onto span(U_k), as the brief specifies.
2. The gradient is clipped and AdamW steps.
3. The displacement from r882 is projected back onto span(U_k): θ ← θ₀ + U_k U_kᵀ(θ − θ₀).

Step 3 is needed because AdamW's per-coordinate scaling and decoupled weight decay would otherwise move θ out
of the subspace. The final out-of-span fraction is recorded per cell; it is expected to be at float32
rounding level.

**CTRL_DRIFT** is a step-size control, the lesson of H151.
- Its learning rate is fixed from the completed CTRL and PROJ_K4 cells before any CTRL_DRIFT cell runs, so the
  arms run in the order CTRL, PROJ_K2, PROJ_K4, CTRL_DRIFT.
- **Manipulation check:** CTRL_DRIFT's median final drift must lie in [0.85, 1.15] × PROJ_K4's (inclusive).
  Otherwise the mechanism label is `DRIFT_MATCH_INVALID`.

Only the 29,211 adapter parameters change. DOPE `c51a56d0` and every non-adapter tensor are checked bitwise
after every update and every evaluation.

## Harness (merged H148, new seeds)

**Data.** The H145 real-S3 TRAIN roster: **1,219 contexts in 205 families**, from the pinned H50 packet and H58
cache. DL-Val (91 contexts, 19 families) is disjoint by family.

**Folds.** Sorted families are permuted with **`default_rng(160140)`** and split into 5 groups of 41 families,
with context counts [236, 259, 223, 256, 245]; [fold_manifest.json](fold_manifest.json) pins them. For each
outer fold A:
- B = (A + 1) % 5 is the selection fold;
- the other three groups (123 families) are the fit fold;
- every minibatch is checked to contain only fit contexts.

**Cells.** 4 arms × 5 outer folds × **seeds 160101–160103**, giving 60 cells and **2,400 updates**.

**Selection** uses H148's `choose` rule on B only.
- Snapshots are taken at steps 0, 10, 20, 30 and 40, and scored by equal-family mean.
- The best step wins, and exact ties go to the earliest step.
- That step is admitted only if its paired gain over step 0 is **> +0.001** (gains within 1e-12 of 0.001
  reject) **and** the 10,000-draw family-bootstrap 95% lower bound is **> 0**. Otherwise step 0 (r882) is
  selected, with no fallback to a second-best step.
- Bootstrap seed: 160141 + 100·A + 10·seed_index + arm_index + 1.
- The selection lock is written before any prediction on A.
- **The number of cells per arm that selected a non-zero step is reported.**

## Endpoints and decision

**Primary panel.** Each arm's selected policy is assembled across the five A folds into a full
1,219-context / 205-family assessment panel per seed. r882 is the step-0 column.

**Primary contrast.** For each arm against r882:
- the equal-family gain within each seed, then the **median across the 3 seeds**;
- a 10,000-draw shared 205-family bootstrap (seed 160141) of the median seed contrast, giving a 95% percentile
  CI and a one-sided p.

**Elo** against r882 = 1500 uses the canonical H145 Bradley–Terry method:
- 400 MM iterations, prior 0.01;
- family-averaged pairwise context wins with tie margin 1e-4;
- 1,000 shared family draws (seed 160142), with the median seed rating per draw.

AUC and speed metrics are not reported.

**ADVANCE** applies to the primary family {PROJ_K2, PROJ_K4} only, with Holm m = 2 on the one-sided p. An arm
advances only if all of these hold:
- median gain > +0.001 (strictly, with a 1e-12 tie band);
- 95% CI lower bound > 0;
- at least 2 of 3 seeds positive;
- Holm-adjusted p < 0.05;
- at least one non-zero selected step.

CTRL and CTRL_DRIFT are controls and are never admitted.

**Overall verdict.**
- `RULE_NEUTRAL` if every one of the 60 cells selects step 0, so that all arms are identical to r882.
- `ADVANCE` if a primary arm passes.
- `NO_ADVANCE` otherwise.

**Mechanism** (PROJ_K4 − CTRL_DRIFT on the primary panel):
- `DRIFT_MATCH_INVALID` if the manipulation check fails;
- `NO_SELECTION_BOTH` if neither arm admits a step;
- `SUBSPACE_SPECIFIC` if the median > 0 and the CI lower bound > 0;
- `STEP_SIZE_ONLY` otherwise.

**Secondary (descriptive).** The 91-context DL-Val score of each selected policy, averaged across the five
outer policies per seed with r882 as the reference. DL-Val also historically selected r882.

**Scope.**
- These are TRAIN contexts that r882 has historically seen, so this is conditional continuation evidence, not
  an unseen-family test.
- The basis comes from adapters the flywheel selected using DL-Val, and only weights are used; no data enters
  the basis.
- Identical seed panels are reported as such and are not counted as independent evidence.

## Execution and custody

**Host and limits.**
- xbabe2 CPU at nice 19, with math libraries and Torch at 1 thread.
- At most **4 OS threads across the whole process tree**, supervisor included.
- Aggregate RSS < 6,000,000,000 B.
- **Hard stop** (no pause or resume) if any of these occur:
  - load1 > 0.85 × cores;
  - MemAvailable < 16 GiB (swap is full on this host);
  - free disk < 40 GB, or used ≥ 95%;
  - scratch (`~/jope-data/h160` plus custody) ≥ 1 GB;
  - wall time > 10 h.
- No GPU and no `qualification.lock`.
- The audit hook refuses sealed, holdout and meta-test paths and any unlisted `~/jope-data` read. No live-loop
  path is used after the one-time copies.

**Order of operations.** The constructor-and-basis smoke (0 forwards, 0 updates) runs before the PR. After the
merge and rd_argon's PASS, `launch.json` binds the merged source commit and the manifest hash.

**Regression first.** The run's first scored operation must reproduce all 91 r882 DL-Val utilities **exactly**
(zero per-context and mean error, mean 1.1377941898164525). It runs before any TRAIN scoring or update.

**Failure handling.** Any failure, including a guard trip, consumes the single attempt and yields `INCOMPLETE`,
with no retry.

**Where outputs go.** Snapshots, panels and receipts stay outside Git in `~/jope-data/h160` and
`~/jope_plan/evidence/H160`, with a create-only S3 archive and a manifest. Git holds only this protocol, code,
small manifests and the final small metrics.
