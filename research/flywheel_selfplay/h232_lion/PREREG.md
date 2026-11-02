# H232 — Lion optimizer for the r882 adapter continuation, xbabe1 CPU

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hw_t2120_opus_h232_lion.md` (watcher, 2026-11-02 21:25Z).
**Claim:** `~/jope_plan/ns/receipts/H232_claim.txt`. This is a diagnostic only. It cannot advance a model, a serving
release or a live flywheel checkpoint. DOPE stays `c51a56d0`. Training runs offline on a read-only COPY of r882; the
live flywheel is never read, written or preempted. **This prereg merges before anything runs.**

## Hypothesis

No earlier H-wave (H40–H231) tried Lion or another sign-momentum optimizer for the adapter; H228's ANDMASK is a
gradient mask, and H197/H59 were clipping. Lion (Chen et al. 2023) updates with the **sign** of an interpolated
momentum, so every coordinate moves by the same step size. **H232 asks: does Lion, at the same 40-update budget as the
incumbent AdamW continuation, find adapter updates that lift DL-Val91 utility over r882 by more than +0.002?**

## Machinery (merged H170, unchanged except the optimizer)

- **Inherited from merged H170** (`research/flywheel_selfplay/h170_onecycle`, pinned as parent sources):
  - the H146 runtime;
  - the frozen-anchor cache;
  - H156's paired batches of 8 contexts and 24 candidates, with its loss;
  - the 47 adapter tensors (29,211 parameters) that train;
  - global grad-norm clipping at 1.0;
  - snapshots at steps 0/10/20/30/40;
  - the strict inner selector: earliest best step, gain > .001 with 1e-12 tie rejection and a 95% bootstrap LCB > 0,
    else r882;
  - per-cell selection locks fsynced before any trained-policy assessment;
  - canonical Elo.
- **Cross-fit.** 1,219 DL-Train contexts in 205 families, 5 family folds (fold seed 232140, 41 families each;
  best-match Jaccard 0.159 against H170's partition). Each cell fits on 123 families, selects on 41 and assesses on
  41. Seeds 232101–232103: 15 cells.
- **Budget.** 15 cells × 3 arms × 40 updates = **1,800 optimizer updates**, at the same per-trajectory budget as
  CONTROL.

## Arms

| Arm | Optimizer | lr | betas | decoupled weight decay | Role |
|---|---|---:|---|---:|---|
| r882 | none (frozen incumbent) | — | — | — | reference (Elo 1500) |
| CONTROL | AdamW (exact H170 CTRL recipe) | 1e-4 | (0.9, 0.999) | 1e-4 | incumbent recipe |
| **LION** | Lion | 1e-5 (CONTROL/10) | (0.9, 0.99) | 3e-4 (3 × CONTROL) | **primary** |
| LION_LR3 | Lion | 3.33e-5 (CONTROL/3) | (0.9, 0.99) | 3e-4 | secondary (Holm m = 2) |

- **Lion step** (Algorithm 2, decoupled decay), per parameter with a gradient:
  `p ← p·(1 − lr·wd)`; `c = β1·m + (1 − β1)·g`; `p ← p − lr·sign(c)`; `m ← β2·m + (1 − β2)·g`.
  Parameters with a None gradient are skipped entirely, as in native AdamW.
- **Schedule.** All settings are constant over the 40 updates, matching CONTROL's constant schedule. They are pinned
  in [schedules.json](schedules.json), and the runner refuses any difference.
- **Reference test.** A test checks the implementation against the reference formula.

## Statistics and decision

- **Primary panel: DL-Val91** (91 contexts, 19 families), per hw_t2120. Each seed averages the context utilities of
  its five selected policies (a uniform policy mixture, as in H170), then infers over 19 families.
- **Direction check first.** Higher is better. The run first requires exact native r882 on all 91 contexts:
  mean 1.1377941898164525, bit-exact.
- **Statistic.** The median over the 3 seeds of the per-seed equal-family gain against r882. The CI95 comes from
  10,000 shared family draws (seed 232141), taking the median seed effect in every draw. The one-sided p is Holm-
  adjusted over LION and LION_LR3 (m = 2).
- **ADVANCE (per arm)** iff all of:
  - median gain against r882 **> +0.002**;
  - family LCB95 **> 0**;
  - Holm p **< 0.05**.

  Anything else is NO_ADVANCE. **Ties are no advance:** a gain at or below +0.002 fails. CONTROL is reported, not a
  decision arm.
- **Also reported, not gated.**
  - Every contrast against CONTROL (LION − CONTROL, LION_LR3 − CONTROL).
  - Elo against r882 and CONTROL (canonical Bradley–Terry, 400 iterations, tie margin 1e-4, prior .01, 1,000 draws).
  - The 1,219-context family out-of-fold panel.
  - The descriptive unconstrained best step.
  - Selection counts, step-40 drift, gradient-norm traces and pseudoreplication groups.
- **Elo and utility only.** No AUC and no speed verdict.

## Data, reads and frozen state

- **Data.** Real S3-derived data only. All 276 dataset IDs join `origin=real` in the pinned catalog
  ([provenance.json](provenance.json), inherited unchanged from H170 because the packet is identical). Inputs are
  the H50 packet and the H58 feature cache, with every input and asset pinned by SHA-256 in
  [manifest.json](manifest.json).
- **Checkpoint.** A read-only copy of r882 at `~/jope-data/h232/inputs/round-882.pt` (`0084479a…`) and its
  reference `best_dev_checkpoint.json` (`30853cad…`), staged on xbabe1 from the H170 inputs with hashes verified.
- **Never read.** Sealed, holdout or meta-test payloads, and the live loop. An audit hook allows only the pinned
  paths.
- **Frozen state.** DOPE `c51a56d0` and every non-adapter byte are checked after every update and evaluation. Only the
  adapter trains.

## Host, guards and process

- **Host.** xbabe1 CPU (load about 8 of 102), nice 19, `taskset` on 4 cores, one math thread. No GPU, so no
  qualification lock is needed. If a GPU turned out to be needed, H232 stops and would queue through
  `~/jope_plan/qualification.lock`.
- **Load pauses, never stops.** The monitor sends SIGSTOP to its own worker's process group when load5 > 60, and
  SIGCONT when load5 ≤ 55.
- **Hard stops:**
  - process-tree RSS ≥ 4 GiB (H170's peak was 3.45 GB);
  - MemAvailable < 20 GiB (swap is full on both hosts);
  - disk free < 150 GB;
  - more than 4 tree threads;
  - the compute stop at **2026-11-03T04:00:00Z**.
- **Admission.** load5 ≤ 55 and every hard limit satisfied.
- **Order.**
  1. Prereg merged by 22:45Z.
  2. Constructor smoke: inventory, census and frozen checks, 0 updates.
  3. The single full attempt. It runs the exact Native91 regression first and opens TRAIN outcomes only after it
     passes.

  There is a **single attempt and no retries**. H170 took 40 min for 4 arms; H232 is expected to take about 30 min.
- **Closeout** by 04:30Z: README, a small `metrics.json`, the INDEX.jsonl row and the overnight index row. Otherwise
  H232 closes INCOMPLETE with honest receipts.

## Custody

The run root and logs go create-only to
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H232/<run_id>/`, with a manifest. No large
objects go into git.
