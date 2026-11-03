# H247: Muon and Grokfast-EMA for the r882 adapter continuation

**Status: complete. Verdict: NO_ADVANCE.** The rule was preregistered in [PREREG.md](PREREG.md) (PR #2094, reviewed
head `6ac693bb`, merged as `d42ebcc2` at 05:35:23Z) before anything ran. The single attempt ran on xbabe2 CPU from
05:36:04Z to 06:21:27Z: **1,800 optimizer updates** over 15 cells × 3 arms × 40.

**Neither new geometry beats r882.**
- **The decision.** Under the strict selector both tie r882: MUON and GROKFAST − r882 = 0 [0, 0] on DL-Val91, 0/3
  seeds, Holm p 1.0, Elo 1500.
- **The unconstrained step-40 diagnostic** shows they move utility very differently:
  - **Muon** barely hurts: −0.0027 [−0.0080, +0.0015], with L2 drift 0.12. It is **+0.023 [+0.003, +0.051] better
    than AdamW** at step 40 (3/3 seeds) and never flips the cliff context.
  - **AdamW CTRL** loses −0.026 [−0.058, −0.003], with drift 0.37.
  - **Grokfast** loses the most: −0.032 [−0.066, −0.008], with drift 0.49.
- **Bottom line.** Orthogonalized updates contain the drift damage, but they still do not find a better adapter than
  r882.

## Hypothesis

**The background.** Every earlier update-rule variant of the 40-update r882 continuation tied with or lost to r882:
AdamW CTRL, one-cycle, SAM, Lion, gradient noise, clipping, GroupDRO and AdaBelief/RAdam.

**The question.** Does changing the update **direction** lift DL-Val91 by more than +0.002? The candidates are
orthogonalized momentum (Muon, Jordan et al. 2024) and slow-gradient amplification (Grokfast-EMA, Lee et al. 2024).

## Setup

- **Arms.** All use the same CTRL loss, H156 batches, data order, clip 1.0 and 40 updates from a read-only r882 copy.
  Hyperparameters were frozen before any outcome:
  - **CONTROL:** exact H170 CTRL AdamW (lr 1e-4, betas .9/.999, weight decay 1e-4).
  - **MUON:** Nesterov momentum .95 with quintic Newton–Schulz-5 orthogonalization, lr 2e-4 × √max(1, rows/cols),
    decoupled weight decay 1e-4, on all 16 two-dimensional adapter matrices. All other tensors use CTRL AdamW.
  - **GROKFAST:** CTRL AdamW with g ← g + 2.0·EMA_.98(g), applied before the clip.
  - **Hypothesis family.** MUON and GROKFAST are co-primary, with Holm m = 2.
- **Design.**
  - **Machinery.** The merged H237/H232/H170 machinery.
  - **Folds and cells.** One fresh fold seed 247140, seeds 247101–247103, 15 cells per arm.
  - **Selector.** The strict inner selector on SELECT (gain > .001 and family LCB95 > 0, else r882), locked before
    assessment.
  - **Decision.** On DL-Val91 (91 contexts, 19 families, each seed's 5-cell policy mixture), with the four gates:
    median > +0.002, LCB95 > 0, Holm p < .05, ≥ 2/3 seeds positive.
- **Checks.**
  - The Native91 regression passed first: 91/91 contexts, cached forward bit-exact to native r882.
  - Each arm's optimizer covered exactly the 47 adapter parameters, with Muon on the 16 matrices.
  - DOPE `c51a56d0` was unchanged before and after.
- **Resources.**
  - xbabe2 cores 44–47, nice 19.
  - Peak RSS 3.35 GB, 3 threads.
  - 2,722 s, with no memory or load pause and no resume.

## Outcome

### Decision panel (DL-Val91; Elo with r882 = 1500; plateau best 1.1389572)

| Contrast | Per-seed gain | Median [CI95] | Seeds > 0 | Holm p | Elo | Harm families per seed |
|---|---|---|---:|---:|---:|---|
| **MUON − r882** | 0, 0, −0.0005 | **0.0000 [0, 0]** | 0/3 | 1.0 | 1500 | 0, 0, 2 |
| **GROKFAST − r882** | −0.0070, 0, 0 | **0.0000 [0, 0]** | 0/3 | 1.0 | 1500 | 4, 0, 0 |
| CONTROL − r882 | 0, 0, 0 | 0.0000 [0, 0] | 0/3 | — | 1500 | 0, 0, 0 |

- **Selections.** The strict selector kept r882 in 14 of 15 cells for each arm.
  - **MUON** accepted step 20 in 1 cell (A2-247103). It cost that seed −0.0005.
  - **GROKFAST** accepted step 10 in 1 cell (A2-247101). That policy flips the H152 cliff context, which cost seed
    247101 −0.0070.
- **Muon's selector signal.** Muon's best SELECT step beat r882 in most cells, but only by +0.0001 to +0.0026, with an
  LCB ≤ 0.
- **Strict out-of-fold panel on TRAIN** (1,219 contexts): MUON and GROKFAST − r882 = 0 [0, 0]. The per-seed values
  are 0, 0, −0.00014 for MUON and −0.00033, 0, 0 for GROKFAST.
- **Plateau.** r882 sits at −0.00116 from the plateau best. The strict mixtures sit at −0.00116, except GROKFAST seed
  247101 at −0.0084 and MUON seed 247103 at −0.0017.

### Unconstrained diagnostics (descriptive; before any strict fallback)

| Readout | CONTROL − r882 | MUON − r882 | GROKFAST − r882 | MUON − CONTROL | GROKFAST − CONTROL |
|---|---|---|---|---|---|
| **Step 40, DL-Val91** | −0.0263 [−0.0583, −0.0026] | **−0.0027 [−0.0080, +0.0015]** | −0.0324 [−0.0664, −0.0077] | **+0.0231 [+0.0035, +0.0506], 3/3** | −0.0109 [−0.0241, +0.0016] |
| Step 40, TRAIN out of fold | −0.0340 [−0.0609, −0.0055] | −0.0043 [−0.0113, +0.0008] | −0.0308 [−0.0544, −0.0070] | +0.0297 [+0.0035, +0.0530], 3/3 | −0.0029 [−0.0137, +0.0135] |
| Best SELECT step, DL-Val91 | −0.0044 [−0.0109, +0.0007] | −0.0024 [−0.0084, +0.0008] | −0.0077 [−0.0217, −0.0003] | +0.0020 [−0.0026, +0.0087] | +0.0005 [−0.0036, +0.0036] |
| Step-40 Elo, DL-Val91 | 1496 | 1500 | 1466 | — | — |
| Step-40 harm families (DL-Val, per seed) | 7, 10, 7 | 4, 4, 4 | 10, 10, 11 | — | — |
| Step-40 L2 drift from r882 (median [min, max]) | 0.372 [0.362, 0.392] | **0.122 [0.120, 0.125]** | 0.486 [0.465, 0.514] | — | — |

### The H152 gate-cliff context #27 (`fam-9f67f144…`), reported separately and not excluded

r882 scores +1.1227 there.

| Mixture | Value per seed | Context #27 |
|---|---|---|
| Strict, CONTROL | +1.1227 in all seeds | kept |
| Strict, MUON | +1.1227 in all seeds | kept |
| Strict, GROKFAST | +0.4982 in seed 247101 | flipped in 1 of that seed's 5 cells |
| Step 40, MUON | +1.1227 in all seeds | **never flips** |
| Step 40, CONTROL | 0.4982 in seed 247102 | flips |
| Step 40, GROKFAST | 0.4992 and 0.4944 in seeds 247101 and 247102 | flips, and erodes in seed 247103 (1.1199) |

## Reading

- **Muon's orthogonalized steps move the adapter a third as far.** Its drift is 0.12, against 0.37 for AdamW at twice
  the lr. That spares most of the utility AdamW loses: step-40 Muon is statistically better than step-40 AdamW on
  both DL-Val91 and TRAIN out of fold (3/3 seeds), and it never trips the gate cliff.
- **But Muon only stays close to r882.** Its best case is a tie; it does not find a better adapter. That fits the
  H232–H243 pattern, where every move away from r882 tends to cost utility, and smaller moves cost less.
- **Grokfast-EMA amplifies the harmful direction.** Grokfast tripled the gradient's slow component, which raised drift
  to 0.49. It is the worst arm at step 40 and the only arm whose strict selection flipped the cliff.
- **The strict selector protected the decision again.** It kept r882 in 28 of 30 decisions, and the 2 accepted cells
  were small or harmful on val.

## Verdict

**NO_ADVANCE.**
- MUON and GROKFAST tie r882 on DL-Val91: median 0 [0, 0], 0/3 seeds, Holm p 1.0, Elo 1500.
- No serving or live-flywheel change. DOPE `c51a56d0` is unchanged.

## Next step

**Stop varying the update rule.** Every update-rule family tried on this 40-update continuation (AdamW, one-cycle,
SAM, Lion, gradient noise, clipping, GroupDRO, AdaBelief, RAdam, Muon and Grokfast) ties with or loses to r882. The
update rule is not the bottleneck.

**Keep one useful finding.** Muon is the safest continuation optimizer found so far, with a third of AdamW's drift
and no cliff flips. If a future wave adds **new signal**, such as new families or data, Muon is the preferred
continuation optimizer to test it with.

No follow-up geometry wave is proposed.

## Files and custody

- **In this directory:**
  - [metrics.json](metrics.json): the compact readout, 45 KB, with `dope_before`/`dope_after`, the strict,
    unconstrained and step-40 panels, drift, the cliff context and the full-metrics pointer;
  - [PREREG.md](PREREG.md);
  - [runner.py](runner.py), [treatment.py](treatment.py), [analysis.py](analysis.py) and [monitor.py](monitor.py);
  - [test_h247.py](test_h247.py): 14 data-free tests;
  - [schedules.json](schedules.json), [manifest.json](manifest.json), [fold_manifest.json](fold_manifest.json) and
    [provenance.json](provenance.json).
- **S3, create-only with sha256 and VersionId readback:**
  - `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H247/attempt-001/` holds:
    - the full `metrics.json` (382,862 B, sha256 `6c52c27d…`, VersionId `7QFb14Ga…`);
    - the resource summary and the attempt claim;
    - `custody-attempt-001.tar.gz`, which contains every cell JSON, the selection locks, the adapter snapshots
      (`.npz`), the panels, the stdout, the resource samples and both smokes;
    - `s3_manifest.json` (sha256 `38bf97b8…`, VersionId `HEn0gldg…`).
  - The prereg sources and smokes are under `H247/prereg-6ac693bb/` (15 objects).
- **Not in Git.** No checkpoint or snapshot.
