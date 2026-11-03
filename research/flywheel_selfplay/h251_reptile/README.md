# H251: first-order Reptile family meta-continuation of the r882 adapter

**Status: complete. Verdict: NO_ADVANCE.** The rule was preregistered in [PREREG.md](PREREG.md) (PR #2105, reviewed
head `124b3122`, merged as `40f69bc5` at 12:08:15Z) before anything ran. The single attempt ran on xbabe2 CPU from
12:09:02Z to 12:51:56Z: **1,800 gradient evaluations** over 15 cells × 3 arms × 40, with 600 / 120 / 600 outer
updates for CONTROL / REPTILE / REPTILE_K1.

**Reptile ties r882.**
- **The decision.** Under the strict selector both Reptile arms tie r882: REPTILE and REPTILE_K1 − r882 = 0 [0, 0] on
  DL-Val91, 0/3 seeds positive, Holm p 1.0, Elo 1500.
- **The unconstrained step-40 diagnostic** shows why:
  - Both Reptile arms move the adapter only 0.008 in L2, against 0.37 for AdamW.
  - At that distance they lose a little on DL-Val91: REPTILE −0.0018 [−0.0074, +0.0014], REPTILE_K1 −0.0024. They are
    flat on TRAIN out of fold.
  - That is +0.031 [+0.009, +0.060] better than step-40 AdamW (3/3 seeds), whose step-40 loss is −0.033.
- **The mechanism contrast is null.** REPTILE − REPTILE_K1 = +0.0006 [0, +0.0018] on DL-Val91 at step 40, and
  −0.0006 [−0.0014, +0.0004] on TRAIN out of fold. Inner adaptation on one family adds nothing measurable over
  per-family SGD with the same step size.

## Hypothesis

**The background.** Every gradient continuation of r882 tied with or lost to r882: AdamW, one-cycle, SAM, Lion,
gradient noise, clipping, GroupDRO/ANDMASK, AdaBelief/RAdam, Muon, Grokfast and the low-rank side branch.

**The question.** Does first-order Reptile (Nichol et al. 2018) across TRAIN families lift DL-Val91 by more than
+0.002? Reptile adapts on one family per outer step and moves the weights toward the adapted weights, which implicitly
rewards gradients that agree within a family.

## Setup

- **Arms.** All use the same CTRL loss, 8 × 24 batch shape, clip 1.0 and 40 gradient evaluations per cell from a
  read-only r882 copy. Hyperparameters were frozen before any outcome:
  - **CONTROL:** exact H170 CTRL AdamW (lr 1e-4, betas .9/.999, weight decay 1e-4).
  - **REPTILE (primary):** 8 outer steps × 5 inner plain-SGD steps (lr 1e-3 = H170 lr × 10, no momentum or weight
    decay) on ONE FIT family per outer step, then θ ← θ + 0.5 (φ − θ). It visited 7–8 families per cell.
  - **REPTILE_K1 (mechanism control):** 40 outer steps × 1 inner step, with the same lr, ε and family stream. It
    visited 30–38 families per cell.
  - **Hypothesis family.** REPTILE and REPTILE_K1 against r882, with Holm m = 2.
- **Design.**
  - **Machinery.** The merged H247/H237/H170 machinery.
  - **Folds and cells.** One fresh fold seed 251140, seeds 251101–251103, 15 cells per arm.
  - **Selector.** The strict inner selector on SELECT (gain > .001 and family LCB95 > 0, else r882), locked before
    assessment.
  - **Decision.** On DL-Val91 (91 contexts, 19 families, each seed's 5-cell policy mixture), with the four gates:
    median > +0.002, LCB95 > 0, Holm p < .05, ≥ 2/3 seeds positive.
- **Checks.**
  - The Native91 regression passed first: 91/91 contexts, cached forward bit-exact to native r882.
  - Each arm's optimizer covered exactly the 47 adapter parameters.
  - The runner verified the applied lr, outer/inner position and outer-update flag at every step, and every Reptile
    minibatch came from one FIT family.
  - DOPE `c51a56d0` was unchanged before and after.
- **Resources.**
  - xbabe2 cores 44–47, nice 19.
  - Peak RSS 3.37 GB, 3 threads.
  - 2,574 s, with no memory or load pause and no resume.

## Outcome

### Decision panel (DL-Val91; Elo with r882 = 1500; plateau best 1.1389572)

| Contrast | Per-seed gain | Median [CI95] | Seeds > 0 | Holm p | Elo | Harm families per seed |
|---|---|---|---:|---:|---:|---|
| **REPTILE − r882** | 0, −0.0005, 0 | **0.0000 [0, 0]** | 0/3 | 1.0 | 1500 | 0, 2, 0 |
| **REPTILE_K1 − r882** | 0, 0, 0 | **0.0000 [0, 0]** | 0/3 | 1.0 | 1500 | 0, 0, 0 |
| CONTROL − r882 | 0, 0, 0 | 0.0000 [0, 0] | 0/3 | — | 1500 | 0, 0, 0 |
| REPTILE − REPTILE_K1 | 0, −0.0005, 0 | 0.0000 [0, 0] | 0/3 | — | — | — |

- **Selections.** The strict selector kept r882 in 44 of 45 decisions.
  - **REPTILE** accepted step 40 in 1 cell (A2-251102; SELECT gain +0.0025, LCB +0.0001). It cost that seed −0.0005 on
    DL-Val91, through the two families described below.
  - **Selector signal.** REPTILE's best SELECT gain was at most +0.0028. REPTILE_K1 and CONTROL never cleared the rule.
- **Strict out-of-fold panel on TRAIN** (1,219 contexts): every contrast is 0 [0, 0]; REPTILE's per-seed values are
  0, +0.00006, 0.
- **Plateau.** r882 sits at −0.00116 from the plateau best. The strict mixtures sit there too, except REPTILE seed
  251102 at −0.00167.

### Unconstrained diagnostics (descriptive; before any strict fallback)

| Readout | CONTROL − r882 | REPTILE − r882 | REPTILE_K1 − r882 | REPTILE − CONTROL | REPTILE − REPTILE_K1 |
|---|---|---|---|---|---|
| **Step 40, DL-Val91** | −0.0328 [−0.0642, −0.0086] | **−0.0018 [−0.0074, +0.0014]** | −0.0024 [−0.0092, +0.0019] | **+0.0314 [+0.0088, +0.0602], 3/3** | +0.0006 [0, +0.0018], 3/3 |
| Step 40, TRAIN out of fold | −0.0228 [−0.0474, +0.0026] | −0.00001 [−0.0004, +0.0004] | +0.00007 [−0.0003, +0.0005] | +0.0228 [−0.0025, +0.0458], 3/3 | −0.0006 [−0.0014, +0.0004], 1/3 |
| Best SELECT step, DL-Val91 | −0.0033 [−0.0085, +0.0004] | −0.0017 [−0.0056, +0.0003] | −0.0024 [−0.0074, +0.0006] | +0.0016 [−0.0005, +0.0054] | +0.0010 [−0.0005, +0.0036] |
| Step-40 Elo, DL-Val91 [CI95] | 1489 [1445, 1542] | 1497 [1487, 1506] | 1496 [1482, 1507] | — | — |
| Step-40 harm families (DL-Val, per seed) | 10, 9, 10 | 2, 2, 2 | 2, 2, 2 | — | — |
| Step-40 L2 drift from r882 (median [min, max]) | 0.373 [0.348, 0.386] | **0.0077 [0.0071, 0.0111]** | 0.0082 [0.0071, 0.0090] | — | — |

- **Where the small loss comes from.** Both Reptile arms harm the same two DL-Val families in every seed
  (`fam-60b608db…` and `fam-94f2455b…`). Those families flip their pick under a very small adapter move, whichever
  Reptile variant makes it.

### The H152 gate-cliff context #27 (`fam-9f67f144…`), reported separately and not excluded

r882 scores +1.1227 there.

| Mixture | Value per seed | Context #27 |
|---|---|---|
| Strict, all arms | +1.1227 in all seeds | kept |
| Step 40, REPTILE and REPTILE_K1 | +1.1227 in all seeds | **never flips** |
| Step 40, CONTROL | +0.4982 and −0.7509 in seeds 251101 and 251102 | flips |

## Reading

- **Reptile behaves like a very small step.** With plain SGD at lr 1e-3, the inherited clip 1.0 and ε = 0.5, the
  frozen recipe can move the adapter at most 0.02 per cell. It moved 0.008, which is about 1/48 of AdamW's drift. At that
  size it keeps the gate cliff, avoids AdamW's damage (+0.031 better at step 40) and lands within noise of r882.
- **No sign of the Reptile mechanism.** REPTILE and REPTILE_K1 are indistinguishable, at about +0.0006 on DL-Val91 and
  −0.0006 on TRAIN out of fold. Five inner steps on one family before interpolating did not produce more transferable
  updates than one step per family at the same total budget.
- **It fits the pattern from H232–H249.** Smaller moves from r882 cost less, but no continuation direction found so far
  improves it. The step-40 OOF gain of Reptile is 0 to the fourth decimal.
- **The strict selector protected the decision again.** It kept r882 in 44 of 45 decisions; the one accepted cell was
  slightly harmful on DL-Val91.

## Verdict

**NO_ADVANCE.**
- REPTILE and REPTILE_K1 tie r882 on DL-Val91: median 0 [0, 0], 0/3 seeds, Holm p 1.0, Elo 1500.
- No serving or live-flywheel change. DOPE `c51a56d0` is unchanged.

## Next step

**Do not tune Reptile's step size up.** A larger inner lr or ε would only move Reptile toward the AdamW regime that
H232–H249 already mapped as harmful. The mechanism contrast, which is the scientific question, was null at this
budget.

**The continuation family looks exhausted.** Every update rule tried so far, now including family meta-learning, ties with or
loses to r882. Further progress likely needs new signal, such as new families or data. It does not need another
optimizer.

No follow-up Reptile wave is proposed.

## Files and custody

- **In this directory:**
  - [metrics.json](metrics.json): the compact readout, 50 KB, with `dope_before`/`dope_after`, the strict,
    unconstrained and step-40 panels, drift, families visited, the cliff context and the full-metrics pointer. The
    TRAIN OOF step-40 and unconstrained panels carry harm-family counts; the full lists are in S3;
  - [PREREG.md](PREREG.md);
  - [runner.py](runner.py), [treatment.py](treatment.py), [analysis.py](analysis.py) and [monitor.py](monitor.py);
  - [test_h251.py](test_h251.py): 14 data-free tests;
  - [schedules.json](schedules.json), [manifest.json](manifest.json), [fold_manifest.json](fold_manifest.json) and
    [provenance.json](provenance.json).
- **S3, create-only with sha256 and VersionId readback:**
  - `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H251/attempt-001/` holds:
    - the full `metrics.json` (354,163 B, sha256 `a98aa973…`, VersionId `hIkDp3nZ…`);
    - the resource summary and the attempt claim;
    - `custody-attempt-001.tar.gz` (26.5 MB), which contains every cell JSON, the per-cell selection locks, the
      adapter snapshots (`.npz`), the panels, the stdout, the resource samples and both smokes;
    - `s3_manifest.json` (sha256 `8edba266…`, VersionId `fetHLzBg…`).
  - The prereg sources and smokes are under `H251/prereg-124b3122/` (18 objects).
- **Not in Git.** No checkpoint or snapshot.
