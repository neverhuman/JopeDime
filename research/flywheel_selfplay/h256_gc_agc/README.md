# H256: Gradient Centralization and Adaptive Gradient Clipping for the r882 adapter continuation

**Status: complete. Verdict: NO_ADVANCE.** The rule was preregistered in [PREREG.md](PREREG.md) and published on
`origin/v1/next` as `4e94607b` at 13:38:29Z (PR #2071 split; same-agent review receipt sha256 `da280013…`) before
anything ran. The single attempt ran on xbabe2 CPU from 13:39:07Z to 14:38:50Z: **1,800 optimizer updates** over
15 cells × 3 arms × 40.

**Neither filter beats r882, and AGC does nothing measurable under AdamW.**
- **The decision.** Under the strict selector both arms tie r882: GC_AGC and AGC_ONLY − r882 = 0 [0, 0] on DL-Val91,
  0/3 seeds, Holm p 1.0, Elo 1500. r882 was kept in all 45 decisions.
- **AGC is inert under AdamW.** AGC clipped 16% of unit gradients and shrank the gradient norm about tenfold. AdamW
  normalizes each coordinate by its own second moment, so the step barely changed: step-40 drift was 0.374 against 0.373
  for CONTROL. AGC_ONLY − CONTROL is −0.0001 [−0.0006, +0.0005] on DL-Val91 at step 40.
- **GC slightly reduces the damage, without significance.** At step 40, GC_AGC loses −0.023 against −0.032 for CONTROL.
  GC_AGC − CONTROL is +0.0058 [−0.0012, +0.0155] (3/3 seeds) on DL-Val91 and +0.0029 [−0.0017, +0.0088] (3/3) on
  TRAIN out of fold.

## Hypothesis

**The background.** Loss clipping (H125), gradient surgery (H197), Muon/Grokfast (H247), Reptile (H251) and many other
continuation rules tied with or lost to r882. None filtered each unit's gradient relative to its own weights.

**The question.** Do Gradient Centralization (GC; Yong et al. 2020) and unit-wise Adaptive Gradient Clipping (AGC;
Brock et al. 2021) stabilize the 40-update continuation enough to lift DL-Val91 over r882 by more than +0.002?

## Setup

- **Arms.** All use the exact H170 CTRL AdamW (lr 1e-4, betas .9/.999, weight decay 1e-4), the same CTRL loss, batches,
  data order, global clip 1.0 and 40 updates from a read-only r882 copy. Constants were frozen before any outcome:
  - **CONTROL:** no filter.
  - **GC_AGC (primary):** after backward, GC (each output row's mean removed, every ≥ 2-D adapter gradient), then AGC
    (λ 0.01, ε 1e-3; units are output rows of ≥ 2-D tensors and whole ≤ 1-D tensors), then the global clip.
  - **AGC_ONLY (mechanism control):** AGC only.
  - **Hypothesis family.** GC_AGC and AGC_ONLY against r882, with Holm m = 2.
- **Design.**
  - **Machinery.** The merged H247/H237/H170 machinery.
  - **Folds and cells.** One fresh fold seed 256140, seeds 256101–256103, 15 cells per arm.
  - **Selector.** The strict inner selector on SELECT (gain > .001 and family LCB95 > 0, else r882), locked before
    assessment.
  - **Decision.** On DL-Val91 (91 contexts, 19 families, each seed's 5-cell policy mixture): median > +0.002,
    LCB95 > 0, Holm p < .05 and **3/3 seeds positive**.
- **Checks.**
  - The Native91 regression passed first: 91/91 contexts, cached forward bit-exact to native r882.
  - Each arm's optimizer covered exactly the 47 adapter parameters.
  - DOPE `c51a56d0` was unchanged before and after.
  - The parent-source pins are the `v1/next` files that executed. Two H156 lineage files differ from main; H256 does
    not import them, and the first pre-publication smoke failed on exactly that pin before re-pinning.
- **Resources.**
  - xbabe2 cores 44–47, nice 19.
  - Peak RSS 3.35 GB, 3 threads.
  - 3,593 s, including one cooperative load pause of 274 s (13:51:38–13:56:12Z, load5 110.7). There was no memory pause
    and no resume.

## Outcome

### Decision panel (DL-Val91; Elo with r882 = 1500; plateau best 1.1389572)

| Contrast | Per-seed gain | Median [CI95] | Seeds > 0 | Holm p | Elo |
|---|---|---|---:|---:|---:|
| **GC_AGC − r882** | 0, 0, 0 | **0.0000 [0, 0]** | 0/3 | 1.0 | 1500 |
| **AGC_ONLY − r882** | 0, 0, 0 | **0.0000 [0, 0]** | 0/3 | 1.0 | 1500 |
| CONTROL − r882 | 0, 0, 0 | 0.0000 [0, 0] | 0/3 | — | 1500 |

- **Selections.** The strict selector kept r882 in all 45 decisions. The best SELECT step beat r882 in 4–5 cells per
  arm, by at most +0.029, but never with LCB95 > 0.
- **Strict out-of-fold panel on TRAIN** (1,219 contexts): every contrast is 0 [0, 0].
- **Plateau.** Every strict mixture sits at r882, −0.00116 from the plateau best.

### Unconstrained diagnostics (descriptive; before any strict fallback)

| Readout | CONTROL − r882 | GC_AGC − r882 | AGC_ONLY − r882 | GC_AGC − CONTROL | GC_AGC − AGC_ONLY | AGC_ONLY − CONTROL |
|---|---|---|---|---|---|---|
| **Step 40, DL-Val91** | −0.0323 [−0.0636, −0.0032] | −0.0227 [−0.0531, −0.0029] | −0.0286 [−0.0611, −0.0026] | +0.0058 [−0.0012, +0.0155], 3/3 | +0.0035 [−0.0031, +0.0130], 2/3 | −0.0001 [−0.0006, +0.0005], 1/3 |
| Step 40, TRAIN out of fold | −0.0292 [−0.0556, −0.0023] | −0.0263 [−0.0509, −0.0004] | −0.0281 [−0.0558, −0.0022] | +0.0029 [−0.0017, +0.0088], 3/3 | +0.0037 [−0.0030, +0.0083], 2/3 | +0.0002 [−0.0006, +0.0011], 2/3 |
| Best SELECT step, DL-Val91 | −0.0069 [−0.0135, −0.0002] | −0.0069 [−0.0137, −0.0008] | −0.0037 [−0.0108, +0.0002] | +0.0014 [0, +0.0030], 3/3 | −0.00004 [−0.0002, +0.0003] | +0.0003 [0, +0.0008], 3/3 |
| Step-40 Elo, DL-Val91 [CI95] | 1489 [1448, 1539] | 1486 [1450, 1520] | 1500 [1451, 1552] | — | — | — |
| Step-40 harm families (DL-Val, per seed) | 9, 10, 10 | 9, 9, 10 | 9, 9, 10 | — | — | — |
| Step-40 L2 drift from r882 (median [min, max]) | 0.373 [0.355, 0.396] | 0.378 [0.362, 0.402] | 0.374 [0.359, 0.396] | — | — | — |
| AGC-clipped units / unit evaluations | — | 15.9% (43,496 / 273,000) | 16.1% (43,923 / 273,000) | — | — | — |

### The H152 gate-cliff context #27 (`fam-9f67f144…`), reported separately and not excluded

r882 scores +1.1227 there.

| Mixture | Value per seed | Context #27 |
|---|---|---|
| Strict, all arms | +1.1227 in all seeds | kept |
| Step 40, CONTROL | +0.4982 in seed 256102 | flips |
| Step 40, GC_AGC | +1.1237 in seeds 256102 and 256103, +1.1227 in 256101 | never flips |
| Step 40, AGC_ONLY | +1.1237 in seed 256102, +1.1227 otherwise | never flips |

## Reading

- **AGC cannot bite under AdamW.** AGC rescales whole units of the gradient, but AdamW divides each coordinate by a
  running estimate of its own gradient scale. A unit that stays clipped by a steady factor produces almost the same
  update. AGC_ONLY therefore tracks CONTROL: nearly the same drift (0.374 vs 0.373), harm-family counts and a contrast of −0.0001. AGC was
  designed for SGD-type optimizers in NFNets.
- **GC is the only active ingredient, and it is weak.** Removing each row's mean gradient changes the update direction,
  not just its scale. It trimmed the step-40 loss by about 0.006 on DL-Val91 and 0.003 on TRAIN out of fold, in 3/3
  seeds, and kept the gate cliff in every seed. Neither CI excludes zero, and the GC_AGC − AGC_ONLY contrast is 2/3
  seeds.
- **It fits the H232–H251 pattern.** The adapter still moves about 0.37 from r882 in 40 updates and loses about 0.02–0.03.
  Filtering the gradient does not find a better adapter than r882.
- **The strict selector protected the decision.** It kept r882 in all 45 decisions.

## Verdict

**NO_ADVANCE.**
- GC_AGC and AGC_ONLY tie r882 on DL-Val91: median 0 [0, 0], 0/3 seeds, Holm p 1.0, Elo 1500.
- No serving or live-flywheel change. DOPE `c51a56d0` is unchanged.

## Next step

**Drop AGC for AdamW continuations.** It is a no-op by construction under per-coordinate adaptive optimizers.

**GC is a minor descriptive lead, not a candidate.** Its +0.006 step-40 reduction in damage is smaller than Muon's in
H247 (+0.023 against AdamW). Like every other update rule tried so far, it does not beat r882. Progress likely needs new
signal, such as new families or data, rather than another gradient filter.

No follow-up GC/AGC wave is proposed.

## Files and custody

- **In this directory:**
  - [metrics.json](metrics.json): the compact readout, 53 KB, with `dope_before`/`dope_after`, the strict,
    unconstrained and step-40 panels, drift, AGC clip counts, the cliff context and the full-metrics pointer. The TRAIN
    OOF step-40 and unconstrained panels carry harm-family counts; the full lists are in S3;
  - [PREREG.md](PREREG.md);
  - [runner.py](runner.py), [treatment.py](treatment.py), [analysis.py](analysis.py) and [monitor.py](monitor.py);
  - [test_h256.py](test_h256.py): 13 data-free tests. Its coverage, test-inventory and source-classification
    registrations are deferred until `v1/next` is rebased on main, because the freeze forbids CI-config edits;
  - [schedules.json](schedules.json), [manifest.json](manifest.json), [fold_manifest.json](fold_manifest.json) and
    [provenance.json](provenance.json).
- **S3, create-only with sha256 and VersionId readback:**
  - `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H256/attempt-001/` holds:
    - the full `metrics.json` (VersionId `b8tZq3Jv…`);
    - the resource summary and the attempt claim;
    - `custody-attempt-001.tar.gz`, which contains every cell JSON, the per-cell selection locks, the adapter
      snapshots (`.npz`), the panels, the stdout, the resource samples and all smokes;
    - `s3_manifest.json` (sha256 `14bb6762…`, VersionId `ADymijir…`).
  - The prereg sources, smokes and the review receipt are under `H256/prereg-4e94607b/` (22 objects).
- **Not in Git.** No checkpoint or snapshot.
