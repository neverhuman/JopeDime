# H241: zero-update adapter extrapolation along the accepted trajectory, xbabe2 CPU

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hw_t0300_opus_h241_expo.md` (watcher, 2026-11-03 03:00Z,
decided). **Claim:** `~/jope_plan/ns/receipts/H241_claim.txt`, taken under `Hwave-admission.lock` at 02:53:02Z.

This is a diagnostic only. It cannot advance a model, a serving release or a live flywheel checkpoint. DOPE stays
`c51a56d0`: it is checked when r882 loads, after every evaluation pass, and before and after the study.

The study runs offline on read-only, SHA-verified COPIES of saved accepted rounds. The live flywheel (xbabe1
`run-20261030`) is never read, written, paused or preempted.

**This prereg merges before any study outcome is read.** The pre-merge smoke scores 0 contexts.

## Hypothesis

The flywheel's accepted trajectory r782 → r859 → r882 improved DL-Val selection. Existing evidence covers:
- weight **interpolation** (H152 MID/SOUP4, with α in [0, 1]);
- EMA/SWA, ensembles (H163/H179/H182) and distillation.

No H-wave has tried **extrapolation**.

ExPO (Zheng et al. 2024, "Weak-to-Strong Extrapolation Expedites Alignment") observes the following. When a later
checkpoint improves on an earlier one, moving *past* the later checkpoint along their difference can improve
further, at zero training cost.

**H241 asks:** does θ(α) = r882 + α·(r882 − r_prev) lift DL-Val91 regression utility over r882 by more than +0.002?
Candidate generation is the bottleneck (H61). A positive answer would be a free candidate generator.

## Arms (zero optimizer updates)

**The extrapolation.** Only the 47 trainable adapter tensors (29,211 parameters) are extrapolated, in float64, then
cast back to float32. Every other tensor and buffer comes from r882. The runner verifies that all non-adapter
tensors and buffers of r782 and r859 are bitwise identical to r882's, and refuses to run otherwise. A probe found
that only the 47 adapter parameters differ.

**The policies.**

| Policy | r_prev | α | Role |
|---|---|---:|---|
| r882 / CONTROL | none | 0 | reference (Elo 1500). α = 0 returns r882's bits exactly. |
| R782_A0.25 / R782_A0.5 / R782_A1 | r782 | 0.25 / 0.5 / 1.0 | **primary direction**. `R782_A0.5` is the designated primary grid point. |
| R859_A0.25 / R859_A0.5 / R859_A1 | r859 | 0.25 / 0.5 / 1.0 | secondary direction |

**The r_prev adapters.** These are the exact saved accepted adapters that H152 used, as SHA-verified copies:

| Round | sha256 | Copied from |
|---|---|---|
| r782 | `427b0d46…f4af0c` | xbabe2 `jope-data/h160/inputs` |
| r859 | `5802245f…4fa715` | xbabe2 `jope-data/h160/inputs` |
| r882 | `0084479a…6cdb64` | the H237 input copy |

Each copy is identical by sha to H152's pinned `SOURCES`. No substitution was needed.

**The arms.** These are the two selected-α rules, one per direction:
- **EXPO782 (primary)** and **EXPO859 (secondary)**;
- **CONTROL** is α = 0, which is r882 exactly.

**The six fixed grid points** are scored on DL-Val91 and reported descriptively. That includes `R782_A0.5`. A
fixed point can never by itself produce ADVANCE. The decision belongs to the selected-α rule, as the assignment
requires: "α chosen per seed on a FIT split of DL-Train only… then scored once on DL-Val91".

## Machinery (merged H237 engine)

- **Inherited unchanged:**
  - the H146 runtime, with the merged H232 A2 namespace release;
  - the H237 `adapter_forward`, loaded from merged `h237_adabelief/treatment.py` and pinned;
  - the cached frozen anchor, whose native/cached identity is proven bit-exact on r882;
  - the H50 cohort, with `has_meta_test` rows excluded;
  - the real-S3 provenance closure (`provenance.json`, the same as H237);
  - the audit hook.
- **Native91 first.** r882 must reproduce 91/91 contexts and the exact mean 1.1377941898164525 before any TRAIN
  outcome opens.
- **Phase 1 (DL-Train only).** All 7 policies are scored once on the 1,219 DL-Train contexts.
  - **Folds.** Fresh family-grouped folds use seeds 241101–241103 with fold seeds 241141–241143 (205 families → 5
    folds of 41). They are pinned in [fold_manifest.json](fold_manifest.json); the within-wave best-match Jaccard
    is ≤ 0.162.
  - **Cells.** Each of the 15 (seed, outer) cells has a FIT split, the 4 non-outer folds, used only for choosing α.
    The outer fold is ASSESS: descriptive out-of-fold only.
  - **Selector (inherited from H237).** The best α by equal-family FIT mean is accepted only if its gain over α = 0
    is above .001 **and** its family-bootstrap LCB95 is above 0. Otherwise α = 0 (r882). Numerical ties pick the
    smallest α, including 0.
  - **Lock.** All 15 × 2 decisions are written and fsynced to `selection.json` before Phase 2.
- **Phase 2.** Every policy is scored **once** on DL-Val91. Each seed's DL-Val policy for an arm is the equal-weight
  mixture of its 5 cells' selected policies, as in H232–H237. CONTROL's cached DL-Val vector must equal the native
  baseline exactly.
- **Budget.** 7 policies × 1,310 contexts, plus the Native91 regression, at **0 optimizer updates**.

## Statistics and decision

- **Panel.** DL-Val91: 91 contexts, 19 families. Higher utility is better.
- **Statistic.** The median over 3 seeds of the per-seed equal-family gain against r882. The CI95 is the
  family-bootstrap median (10,000 draws, boot seed 241150), with a one-sided bootstrap p and Holm over the 2 r_prev
  directions (m = 2).
- **ADVANCE** for a direction requires all four of:
  - median gain > **+0.002**;
  - family LCB95 > 0;
  - Holm p < 0.05;
  - **≥ 2/3 seeds positive.**

  Ties mean no advance. Beating only CONTROL is descriptive, never an advance. The wave verdict is ADVANCE if
  either direction advances, else NO_ADVANCE.
- **Also reported:**
  - Elo against r882 = 1500 (canonical H148 Bradley–Terry with a 1e-4 tie band) with a CI;
  - the gap to the plateau best 1.1389572 (r3175);
  - the adapter L2 distance to r882 per policy (which equals α·‖r882 − r_prev‖);
  - harm families (equal-family loss > 1e-4) per seed and per fixed point;
  - invalid picks;
  - selected and best α counts;
  - the descriptive OOF and unconstrained readouts;
  - all six fixed points on DL-Val91 and, descriptively, on DL-Train.
- **Not used.** No AUC, and no speed verdicts.

## Resources and the A1 memory guard (H239 A1)

- **Compute.**
  - xbabe2, CPU only: `taskset` cores 44–47, nice 19, one math thread.
  - Aggregate tree RSS < 4 GiB, ≤ 4 threads, disk free ≥ 300 GB.
  - No GPU, xbabe0, xbabe3 or live flywheel. No foreign process control. PR #2071 is not touched.
- **Admission.** load5 ≤ 100 and MemAvailable ≥ 30 GiB.
- **Cooperative pause.** Triggered by load5 > 108 or MemAvailable < 20 GiB. It uses a pause file, not SIGSTOP. The
  worker blocks between evaluation batches and pool loads.
  - **Resume** at load5 ≤ 100 and MemAvailable ≥ 25 GiB, sustained for 30 s.
- **Hard stop.** MemAvailable < 12 GiB sustained for 30 s, or < 6 GiB on any sample, plus the RSS, disk, thread and
  deadline stops. The supervisor signals only its own worker process group.
- **Checkpoints.** Every completed evaluation file is hashed into `progress.json`.
  - **Up to two resumes** are allowed within the window. Only hash-verified evaluation files of the same manifest
    are reused.
  - No re-roll and no deadline reset.
- **Deadlines.**
  - Prereg merge by **04:00Z**.
  - Compute hard stop **07:00Z**.
  - Closeout merge by **07:30Z**.

  If the study is incomplete, it closes as CLOSED INCOMPLETE with honest receipts and null outcomes.

## Custody

- **Repo.** Targeted pytest, Ruff, JSON parse, `git diff --check` and the records gates precede the head-bound
  same-agent review and the owner-authorized admin squash merge. Remote CI is not a gate.
- **S3.** Full outputs go to create-only S3 `…/rd_breakthrough_20261030/H241/` with sha/VersionId readback.
- **Closeout.** It commits only the README and a compact metrics JSON. No checkpoints or large objects go in Git.
