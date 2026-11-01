# H160 — trajectory-subspace projected continuation of r882

**Status: complete. Verdict: RULE_NEUTRAL. Mechanism: NO_SELECTION_BOTH.**
- The protocol is [PREREG.md](PREREG.md), from PR #1354 (merged `53395e52`), with Amendment A1 from PR #1415
  (merged `618e63ce`, reviewed by rd_argon).
- **The run.** Attempt 2 ran on xbabe2 CPU with **2,400 optimizer updates**.
- **The result.**
  - **All 60 cells selected step 0**, keeping r882, in every arm.
  - Every arm is therefore **identical to r882** on the 1,219-context assessment panel and on DL-Val: gain
    0 [0, 0] and Elo 1500 [1500, 1500].
  - Neither a continuation restricted to the flywheel's own accepted trajectory subspace nor an unconstrained
    or drift-matched one produced a step that the strict H148 family selector would admit.

## Hypothesis

Unconstrained continuations of r882 have lost ground (H144, H146, H151), and H148 admitted no continuation
step. H160 asked whether a continuation restricted to the top-k principal directions of the accepted adapter
trajectory **R95 → R129 → R130 → R256 → R782 → R785 → R859 → R882** would be admitted and would improve
held-out equal-family utility. It also asked whether any such effect is specific to the subspace or only
reflects step size, which a drift-matched control tests.

## Setup

- **Basis.** An uncentered SVD of the 7 consecutive deltas over the 29,211 trainable adapter parameters.
  - **k = 2 holds 92.6% of the delta energy and k = 4 holds 98.5%.**
  - The chain is one-time owned copies. All 8 checkpoints carry DOPE `c51a56d0` and have bitwise-identical
    non-adapter tensors.
- **Arms.** Every cell runs 40 AdamW updates from an r882 copy (H148 recipe, lr 1e-4, wd 1e-4).

  | Arm | Constraint | Median final drift |
  |---|---|---|
  | CTRL | none | 0.445 |
  | **PROJ_K2** | gradient projected onto span(U_2), then the displacement re-projected | 0.556 |
  | **PROJ_K4** | the same, with U_4 | 0.507 |
  | CTRL_DRIFT | none; lr = 1.1389e-4, which is 1e-4 × D(PROJ_K4) / D(CTRL) | 0.508 |

  - **Subspace constraint held:** the maximum out-of-span fraction was 1.1e-6, which is float32 rounding.
  - **Drift-match check passed:** CTRL_DRIFT's drift is 1.0019× PROJ_K4's, inside [0.85, 1.15].
  - **Gradless tensors:** six `adapter_10tok.parent_delta_net.*` tensors received no direct gradient in the PROJ
    arms. Under A1 they received their projected component.
  - **Why the projected arms drift farther than CTRL:** Adam normalizes each coordinate, so the projected
    steps keep pointing in a consistent low-dimensional direction and accumulate.
- **Harness.** The merged H148 family cross-fit over 1,219 contexts in 205 families:
  - fit on 3 folds, select on 1, assess on 1, with seeds 160101–160103 and fold seed 160140;
  - 4 arms × 5 outer folds × 3 seeds = 60 cells;
  - snapshots at steps 0, 10, 20, 30 and 40, chosen on fold B only;
  - **a step is admitted only if its gain is > +0.001 (ties within 1e-12 reject) and its 10,000-draw family
    bootstrap lower bound is > 0**; otherwise step 0.
- **Regression.** **R882 reproduced 91/91 DL-Val contexts exactly** (mean 1.1377941898164525) before any
  update, in both attempts.

## Outcome

**Selections.**

| Arm | Non-zero selections | Cells whose best snapshot ≠ step 0 | Cells with best gain > +0.001 | Largest best-snapshot gain (its LCB) | Largest LCB |
|---|---|---|---|---|---|
| CTRL | **0 / 15** | 6 | 5 | +0.0168 (−0.0334) | 0.0000 |
| PROJ_K2 | **0 / 15** | 6 | 3 | +0.0186 (−0.0181) | 0.0000 |
| PROJ_K4 | **0 / 15** | 3 | 3 | +0.0041 (−0.0126) | 0.0000 |
| CTRL_DRIFT | **0 / 15** | 5 | 3 | +0.0078 (−0.0087) | 0.0000 |

Every candidate that cleared the margin had a family-bootstrap lower bound ≤ 0. The selector correctly treated
them as noise on the 41-family selection fold.

**Primary panel (1,219 contexts, 205 families).** Every arm minus r882:
- median-seed gain **0.0000**, with a family-bootstrap 95% CI **[0, 0]**;
- p = 1.0, and Holm-adjusted p = 1.0 for PROJ_K2 and PROJ_K4;
- 0 of 3 seeds positive.

**Elo** against r882 = 1500 (canonical Bradley–Terry, 1,000 family draws): **1500 [1500, 1500] for every
arm**.

**Mechanism.** PROJ_K4 − CTRL_DRIFT = 0, so the label is `NO_SELECTION_BOTH`, since neither arm admitted a
step.

**DL-Val (91 contexts, 19 families, descriptive).** Identical to r882 for every arm.

**Decision.** Every ADVANCE gate fails, including `nonzero_selection`, so the preregistered verdict is
**RULE_NEUTRAL**.

### Seed identity (requested by the t0225 red team)

**Point estimates and CIs come from the family bootstrap, not from the spread across seeds.** The seeds are
not independent evidence in this run:
- The assessment and DL-Val panels are identical across the 3 seeds in every arm (5 of 5 outer folds) because
  every cell selected step 0. That makes "x/3 seeds positive" uninformative here.
- Under projection the seeds are close to pseudo-replicates even *before* selection:
  - PROJ_K2: the non-zero snapshots' selector panels are identical across all 3 seeds in **9 of 20**
    (outer fold × step) cases;
  - PROJ_K4: 1 of 20;
  - CTRL and CTRL_DRIFT: 0 of 20.
- The best-snapshot gain is identical across seeds in 4 of 5 outer folds for both PROJ arms. For example,
  PROJ_K2 on outer fold A1 gives +0.0004994 on all three seeds; the weights differ, but the argmin choices are
  the same.
- **"3/3 seeds positive" would carry no independent evidence for the PROJ arms.**

The details are in [seed_identity.json](seed_identity.json).

### Attempt 1 (INCOMPLETE) and reproducibility

- **What happened.** Attempt 1 stopped at the first PROJ_K2 update. The projection read `p.grad` for every
  adapter tensor, but the six `parent_delta_net` tensors had no gradient. Amendment A1 fixed this.
- **Reproducibility.** Attempt 2's 15 CTRL cells **reproduced attempt 1's selections and drifts exactly**.
- **Disclosed:** my own empty output directory from the first smoke attempt was removed (the owner confirmed
  this was fine).

## Reading

- **No admitted continuation in any of the 4 recipes.** Under this selector, none of the four 40-update
  continuations of r882 produced an admissible improvement on held-out TRAIN families. That holds for the
  unconstrained, trajectory-subspace k = 2 and k = 4, and drift-matched recipes. It matches H148 (0 of 30) and
  extends it to subspace-restricted updates.
- **Restricting to the trajectory subspace does not change the outcome.** The projected arms move along
  historically "good" directions, and further than CTRL, but their best snapshots are no more admissible.
  PROJ_K4's best-gain tail is actually smaller.
- **Scope.**
  - A neutral, all-zero result means *no eligible continuation under these candidates and this selector*. It
    does not prove the subspace carries no signal.
  - r882 has historically seen these TRAIN families, so this is conditional continuation evidence.
  - The selector's 41-family fold gives wide lower bounds: the largest single best-snapshot gain, +0.0186, had a
    lower bound of −0.018.

## Next step

**No adoption.** The live flywheel and r882 are unchanged.

**Further continuations of this kind look unpromising.** Five waves (H144, H146, H148, H151, H160) now show
that short continuations of r882 do not produce admissible gains under strict family-level selection.

**Remaining directions** need either more selection power or a different kind of candidate. Each would be a new,
separately preregistered wave:
- more selection families, or a pooled multi-fold selector;
- candidates other than short low-lr continuations.

## Files and custody

- **Files here.**
  - [metrics.json](metrics.json): the run's metrics, 9.5 KB.
  - [seed_identity.json](seed_identity.json): the descriptive seed-identity counts.
  - Code: [run.py](run.py), [analysis.py](analysis.py), [monitor.py](monitor.py) and
    [test_h160.py](test_h160.py), 12 data-free tests.
  - [preflight.json](preflight.json): the smoke receipt.
- **S3** (create-only; `manifest.json` lists the SHA-256 of 8 bundles):
  `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H160/opus-h160-20261101a/`. It holds:
  - attempt 2's outputs and custody, with snapshots, panels, cell receipts, selection locks, the basis, the
    audit and resource samples;
  - attempt 1's outputs and custody;
  - the smoke evidence;
  - the input hashes.
- **Resources.**
  - Attempt 2 took 17,513 s.
  - Peak: 3 threads, 3.33 GB RSS, 134 MB scratch, load 79.8 (the stop is at 86.7), disk 93.0%.
  - 0 refused reads, no CUDA, no GPU.
- **Not in Git.** No `.pt` or large objects. Local copies are kept, and nothing was deleted.
