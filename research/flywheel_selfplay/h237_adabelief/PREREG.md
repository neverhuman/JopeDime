# H237 — AdaBelief (and RAdam) for the r882 adapter continuation, xbabe1 CPU

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hw_t0030_opus_h237_adabelief.md` (watcher, 2026-11-03 00:30Z).
**Claim:** `~/jope_plan/ns/receipts/H237_claim.txt`, taken under `Hwave-admission.lock`. This is a diagnostic only. It
cannot advance a model, a serving release or a live flywheel checkpoint. DOPE stays `c51a56d0`. Training runs offline
on a read-only COPY of r882; the live flywheel (xbabe1 `run-20261030`) is never read, written, paused or preempted.
**This prereg merges before anything runs.**

## Hypothesis

No earlier H-wave (H40–H236) used AdaBelief, RAdam or gradient centralization. Lion (H232), schedule-free AdamW
(H236), lookahead, SAM, SWA/EMA, clipping and accumulation are different. **H237 asks: is the plateau partly an
adaptive-step-size artefact?** AdaBelief scales each step by the variance of the gradient around its EMA (the
"belief") instead of the raw second moment. It takes larger steps where gradients are consistent and smaller ones
where they are noisy across families. At the same 40-update budget, does that lift DL-Val91 utility over r882 by more
than +0.002?

## Arms (only the optimizer differs)

| Arm | Optimizer | lr | betas | weight decay | Role |
|---|---|---:|---|---:|---|
| r882 | none (frozen incumbent) | — | — | — | reference (Elo 1500) |
| CONTROL | AdamW (exact H170 CTRL recipe) | 1e-4 | (0.9, 0.999) | 1e-4 decoupled | incumbent recipe |
| **ABEL** | AdaBelief (inline, no new dependency) | 1e-4 | (0.9, 0.999) | 1e-4 decoupled | **primary** |
| RADAM | `torch.optim.RAdam` (Liu et al. 2020), `decoupled_weight_decay=True` | 1e-4 | (0.9, 0.999) | 1e-4 decoupled | secondary (Holm m = 2) |

- **AdaBelief step** (Zhuang et al. 2020, with decoupled decay, bias correction and no rectification). Per
  parameter with a gradient, at step t:

      p ← p·(1 − lr·wd)
      m ← β1·m + (1 − β1)·g
      s ← β2·s + (1 − β2)·(g − m)² + ε      (ε = 1e-16)
      p ← p − lr · (m / (1 − β1^t)) / ( sqrt(s / (1 − β2^t)) + ε )

  Parameters with a None gradient are skipped entirely, as in native AdamW.
- **Tests.** The step is checked against the formula above over 30 steps (to 1e-12). A second test checks the
  defining property: on a consistent gradient it takes larger steps than Adam.
- **Settings.** All three arms use the inherited H146 CTRL loss, H156 batches, clipping at 1.0 and constant settings
  for 40 updates.
- **Starting point.** Update 0 is r882 bit for bit: the exact Native91 gate passes before any TRAIN outcome opens.

## Machinery (merged H234 / H232, with the H232 A1/A2 fixes)

- **Inherited unchanged:**
  - the H146 runtime (re-pinned after #2006) and the early-`research`-namespace release;
  - snapshots at steps 0/10/20/30/40;
  - the strict inner selector (as in H235): earliest best step, ties keep the earliest step (r882), gain > .001 with
    a 95% bootstrap LCB > 0, else r882;
  - locks fsynced before assessment;
  - canonical Elo and the audit hook.
- **Fresh per-seed family folds.** Seeds 237101–237103 use fold seeds 237141–237143. Best-match Jaccard is ≤ 0.168
  within H237 and ≤ 0.186 against H170, H232, H234 and H235. They are pinned in [fold_manifest.json](fold_manifest.json).
- **Budget.** 15 cells × 3 arms × 40 updates = **1,800 updates**. Only the adapter trains: 47 tensors, 29,211
  parameters.

## Statistics and decision

- **Decision panel: DL-Val91** (91 contexts, 19 families). Each seed averages its five selected policies.
- **Direction check first.** Higher is better. r882 Native91 must be exact (1.1377941898164525) before TRAIN opens.
- **Statistic.** The median over the 3 seeds of the per-seed equal-family gain against r882. The CI95 comes from
  10,000 shared family draws (seed 237150). The one-sided p is Holm-adjusted over ABEL and RADAM.
- **ADVANCE (per arm)** iff all of:
  - median gain against r882 **> +0.002**;
  - family LCB95 **> 0**;
  - Holm p **< 0.05**.

  Ties are no advance. The primary contrast is against **r882**. Beating only CONTROL is descriptive (hw_t2200).
- **Also reported, not gated.**
  - Contrasts and Elo against CONTROL.
  - The TRAIN out-of-fold panel (1,219 contexts, 205 families) and the unconstrained best step.
  - Harm: families whose three-seed mean delta is < −0.01.
  - Positive seeds, selections and drift.
  - Each arm's DL-Val91 context mean against the plateau best **1.1389572** (r3175).
- **Elo and utility only.** No AUC and no speed verdict.

## Data, reads, frozen state, host and guards

- **Data.** Real S3-derived data only, with the 276 dataset IDs joined to `origin=real` ([provenance.json](provenance.json)).
  The H50 packet and H58 cache are pinned by SHA-256 in [manifest.json](manifest.json).
- **Rows refused.** The cohort check refuses any `has_meta_test` row, any non-train/inner source split and any
  family overlap.
- **Never read.** Sealed, holdout or meta-test data, and the live loop.
- **Checkpoint.** A read-only r882 copy at `~/jope-data/h237/inputs/round-882.pt` (`0084479a…`).
- **Frozen state.** DOPE and every non-adapter byte are checked after every update.
- **Host.** xbabe1 CPU, `taskset` cores 44–47, nice 19, one math thread. No GPU and no qualification lock. Never
  xbabe0 or xbabe3.
- **Memory guard (changed from H234, which a hard 20 GiB stop ended at 19.92 GiB).** MemAvailable < 20 GiB now
  **pauses** the worker's own process group (SIGSTOP), and it resumes once MemAvailable ≥ 22 GiB and load5 ≤ 55.
  Load5 > 60 also pauses.
- **Hard stops** (own process group only): MemAvailable < **12** GiB, RSS ≥ 4 GiB, disk free < 150 GB, more than 4
  threads, and the compute stop at **2026-11-03T06:30:00Z**.
- **Admission.** load5 ≤ 55 and MemAvailable ≥ 22 GiB.
- **Order.**
  1. Prereg merged by 01:45Z.
  2. Constructor smoke: 0 updates.
  3. The single full attempt: Native91 first, then TRAIN.

  There are no retries.
- **Closeout** merged by 07:00Z: README, a compact `metrics.json` (with `dope_before`/`dope_after`; the full metrics
  go to S3), the INDEX.jsonl row and the overnight index row. Otherwise H237 closes INCOMPLETE with honest receipts.

## Custody

The run root and logs go create-only to
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H237/<run_id>/`, with a manifest. No large
objects go into git.
