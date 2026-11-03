# H247: Muon and Grokfast-EMA for the r882 adapter continuation, xbabe2 CPU

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hw_t0530_opus_h247.md` (watcher, 2026-11-03 05:30Z, decided).
**Claim:** `~/jope_plan/ns/receipts/H247_claim.txt`, taken under `Hwave-admission.lock`.

This is a diagnostic only. It cannot advance a model, a serving release or a live flywheel checkpoint. DOPE stays
`c51a56d0`: it is checked at load, after every optimizer step and evaluation, and before and after the study.

Training runs offline on a read-only COPY of r882 (`0084479a…`). The live flywheel is never read, written, paused or
preempted.

**This prereg merges before anything runs.** The pre-merge smoke makes 0 updates and scores 0 contexts.

## Hypothesis

**What has been tried.** Update-geometry variants of the 40-update r882 continuation, all of which tied with or lost
to r882:
- AdamW CTRL and one-cycle (H170);
- SAM/ASAM;
- Lion (H232);
- gradient noise (H165);
- per-context clipping (H197);
- GroupDRO/ANDMASK (H228);
- AdaBelief/RAdam (H237).

**What has not.** No wave has orthogonalized the update (Muon) or amplified the slow gradient component (Grokfast).
Both change the update **direction** rather than the loss, which is where H46/H47 located the loss-down/utility-down
mismatch.

**H247 asks:** at the same 40-update budget, data and loss, does either geometry lift DL-Val91 utility over r882 by
more than +0.002?

## Arms (only the update rule differs; hyperparameters frozen here, no tuning after)

**All arms share:**
- the exact inherited H146/H170 CTRL loss;
- the H156 batches (8 contexts × 24 candidates) and the same data order per (seed, outer);
- the global gradient clip at 1.0;
- 40 updates per cell;
- start from r882, the 47 adapter tensors (29,211 parameters).

**The arms:**
- **CONTROL.** Exact H170 CTRL AdamW: lr 1e-4, betas (0.9, 0.999), decoupled weight decay 1e-4.
- **MUON (co-primary)** (Jordan et al. 2024). Applied to every **2-D adapter matrix**: 16 tensors, including the
  modality embedding and the three 1×n output rows.
  - **The update:**
    1. Momentum: buf ← 0.95·buf + g.
    2. Nesterov direction: d = g + 0.95·buf.
    3. Orthogonalize: O = NS5(d). That is 5 quintic Newton–Schulz iterations with coefficients (3.4445, −4.7750,
       2.0315), in float32, on X = d / (‖d‖_F + 1e-7), with tall matrices processed transposed.
    4. Step: p ← p·(1 − lr·1e-4) − lr·√max(1, rows/cols)·O, with **lr 2e-4**.
  - **Other tensors.** All 1-D and 3-D tensors (biases, norms, scalars, `null_hist_token`) use the exact CTRL AdamW.
  - **None gradients.** Parameters with a None gradient are skipped.
- **GROKFAST (co-primary)** (Lee et al. 2024, `gradfilter_ema`). The CTRL AdamW recipe, with the gradient filtered
  **after backward and before the global clip**:
  - the EMA starts as the first gradient;
  - then ema ← 0.98·ema + 0.02·g;
  - and g ← g + 2.0·ema.

[schedules.json](schedules.json) records the per-update settings, and the runner verifies them. Each arm's optimizer
must cover exactly the 47 adapter parameters.

## Machinery (merged H237 / H232 / H170)

- **Inherited unchanged:**
  - the H146 runtime, with the H232 A1/A2 fixes;
  - the cached frozen anchor, whose update-0 identity is bit-exact to native r882;
  - snapshots at steps 0/10/20/30/40;
  - the strict inner selector: earliest best step on the SELECT fold, accepted only if the gain is above .001
    **and** the family-bootstrap LCB95 is above 0, else r882;
  - fsync-locked selections before any assessment of a trained policy;
  - each policy is assessed once;
  - canonical Elo and the audit hook.
- **Folds.** One fresh fold seed, **247140**, gives the family partition (205 families → 5 × 41), shared by seeds
  247101–247103. The seeds drive batch plans, optimizer streams and bootstraps. The partition is pinned in
  [fold_manifest.json](fold_manifest.json).
- **Cells.** 15 per arm, each with FIT, SELECT and ASSESS folds as in H237.
- **Budget.** 15 × 3 × 40 = **1,800 updates**.
- **Native91 first.** Exact Native91 (1.1377941898164525) before any TRAIN outcome opens.
- **H239 A1 additions:**
  - cooperative pauses between optimizer steps, evaluation batches and pool loads;
  - per-cell fsynced checkpoints hashed into `progress.json`;
  - at most 2 hash-verified resumes, which skip completed cells. No re-roll and no deadline reset.

## Statistics and decision

- **Panels.**
  - **Primary:** DL-Val91, each seed's equal mixture of its 5 cells' selected policies.
  - **Secondary:** the 1,219-context DL-Train strict out-of-fold panel.
- **Statistic.** The median over 3 seeds of the per-seed equal-family gain against r882. The CI95 is the
  family-bootstrap median (10,000 draws, boot seed 247150), with a one-sided p and **Holm over MUON and GROKFAST**
  (m = 2).
- **ADVANCE** requires all four of:
  - median gain > **+0.002**;
  - family LCB95 > 0;
  - Holm p < .05;
  - **≥ 2/3 seeds positive.**

  Ties mean no advance. Beating only CONTROL is descriptive.
- **Also reported:**
  - Elo against r882 = 1500 with a CI;
  - the gap to the plateau best 1.1389572 (r3175);
  - the step-40 L2 drift from r882;
  - harm families (loss > 1e-4);
  - the unconstrained best-step readout;
  - **the UNCONSTRAINED step-40 diagnostic**: every arm's step-40 adapter scored on ASSESS and DL-Val91 before any
    strict fallback, so we learn whether either geometry moves utility at all.
- **The H152 gate-cliff context.** Context #27 (`fam-9f67f144…`) is reported separately for the strict and step-40
  mixtures. It is never excluded.
- **Not used.** No AUC, and no speed verdicts.

## Resources and the A1 memory guard (H239 A1)

- **Compute.**
  - xbabe2, CPU only: `taskset` 44–47, nice 19, one math thread.
  - Tree RSS < 4 GiB, ≤ 4 threads, disk free ≥ 300 GB.
  - Cores 40–43, 48–51 and 52–55 are not used.
  - No GPU, xbabe0, xbabe3 or live flywheel. No foreign process control. PR #2071 is not touched.
- **Admission.** load5 ≤ 100 and MemAvailable ≥ 30 GiB.
- **Cooperative pause.** Triggered by load5 > 108 or MemAvailable < 20 GiB.
  - **Resume** at load5 ≤ 100 and MemAvailable ≥ 25 GiB, sustained for 30 s.
- **Hard stop.** MemAvailable < 12 GiB sustained for 30 s, or < 6 GiB on any sample, plus the RSS, disk, thread and
  deadline stops. Only its own worker group is signalled.
- **Deadlines.**
  - Prereg merge by **06:30Z**.
  - Compute hard stop **09:30Z**.
  - Closeout by **10:15Z**.

  If the study is incomplete, it closes as CLOSED INCOMPLETE with honest receipts.

## Custody

- **Repo.** pytest, Ruff, JSON parse, `git diff --check` and the records gates run on a private, sha-verified
  `jope-records`. They precede the head-bound same-agent review and the admin squash merge. Remote CI is not a gate.
- **S3.** Full outputs go to create-only S3 `…/rd_breakthrough_20261030/H247/<run_id>/` with VersionId readback.
- **Closeout.** It commits only the README and a compact metrics JSON. No checkpoints go in Git.
