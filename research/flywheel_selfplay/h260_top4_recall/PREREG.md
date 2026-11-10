# H260: recall@4 objectives for the r882 adapter continuation, xbabe2 CPU

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hw_t1530_opus_h260.md` (watcher / Grok Bot for Jepson,
2026-11-03 15:30Z, decided). **Claim:** `~/jope_plan/ns/receipts/H260_claim.txt`.

This is a diagnostic only. It cannot advance a model, a serving release or a live flywheel checkpoint. DOPE stays
`c51a56d0`: it is checked at load, after every optimizer step and evaluation, and before and after the study.

Training runs offline on a read-only COPY of r882 (`0084479a…`). The live flywheel is never read, written, paused or
preempted.

**Publication.** Under the PR #2071 split (`~/nudges/CI_FREEZE.md`) this prereg is pushed to `origin/v1/next`
(non-force) before anything runs; nothing merges to main. The pre-publication smoke makes 0 updates, scores 0
TRAIN contexts and runs only the r882 reproduction gates below on DL-Val91. The launch is bound to the published commit, its verified ancestry in `origin/v1/next` and the sha256 of a
durable same-agent review receipt.

## Hypothesis

**What is known.** H258 (`v1/next` 8869cc439) showed that the deployment-relevant quantity is best-of-revealed:
TOP4_PICK@r882 reaches DL-Val91 1.2403601682250796 (context mean) against single-pick r882 1.1377941898164525. Every
adapter continuation so far (H46–H259) was trained and selected for top-1 argmin, and none lifted r882. No wave has
trained the adapter for **recall@4**: getting the oracle-best candidate into the predicted top-4, which is what the
4-reveal flywheel consumes.

**H260 asks:** does a top-4-boundary objective lift DL-Val91 TOP4_PICK utility over TOP4_PICK@r882 by more than +0.002,
at the same 4-reveal budget, without costing more than 0.002 of single-pick utility?

## Arms (only the loss differs; constants frozen here, no tuning after)

**All arms share:**
- the exact H170 CTRL AdamW (lr 1e-4, betas 0.9/0.999, decoupled weight decay 1e-4);
- the H156 batches (8 contexts × 24 candidates: the 4 lowest-fitness candidates plus 20 sampled) and the same data
  order per (seed, outer);
- the global gradient clip at 1.0;
- 40 updates per cell;
- start from r882, the 47 adapter tensors (29,211 parameters). DOPE is frozen in fp32.

**Oracle labels.** For each minibatch context, the oracle best is the argmin over its 24 candidates of the gated cost
(fitness if gate else 2.0), first index on ties. Labels come only from FIT-fold TRAIN contexts, never from SELECT,
ASSESS or DL-Val.

**The arms:**
- **CONTROL.** Exact H170 CTRL: L = L_H170.
- **T4_HINGE (primary).** L = L_H170 + 0.5 · mean_ctx relu(ĉ(oracle best) − ĉ(predicted rank 4) + m). Predicted rank 4
  is the candidate with the 4th-lowest predicted cost in the minibatch (stable sort, detached index); gradients flow
  through both costs. **m = 0.05 × the standard deviation of r882's predicted costs over all candidates of the cell's
  FIT-fold contexts**, computed once per cell from the frozen r882 before any update. It reads no fitness or gate.
- **T4_PL (secondary).** L = L_H170 + 0.5 · mean_ctx −log P_PL(oracle best is among the first 4 draws without
  replacement from softmax(−ĉ/τ)), with **τ = 1**. The probability is computed exactly, as the sum over every ordered
  prefix of 0–3 other candidates, in float64.

[schedules.json](schedules.json) records the per-update settings and the runner verifies them. Each arm's optimizer must
cover exactly the 47 adapter parameters. Traces record the auxiliary loss and the margin per update.

## The TOP4_PICK readout (exact H258)

For a policy's predicted costs on a context's full candidate pool:
- top_zgap = (second-lowest − lowest cost) / std(cost), or 999 when std ≤ 1e-12 (H252 `compute_r882_top_zgap`);
- if top_zgap < **0.25** (gated), reveal the 4 lowest-cost candidates (stable order) and pick the lowest revealed gated
  cost, with ties going to model order (4 reveals);
- otherwise pick argmin cost (0 reveals);
- utility = −fitness if gate else −2.0.

**Reproduction first.** Native91 single-pick must equal 1.1377941898164525 on all 91 contexts, bit-exact, and
DL-Val91 TOP4_PICK@r882 must reproduce the H258 value 1.2403601682250796 (|Δ| ≤ 1e-12) before any TRAIN outcome opens.
Recall@4 is the share of contexts whose predicted top-4 contains a candidate with the oracle-best gated cost.

## Machinery (merged H256 / H247 / H170)

- **Inherited unchanged:**
  - the H146 runtime, with the H232 A1/A2 fixes;
  - the cached frozen anchor, whose update-0 identity is bit-exact to native r882;
  - snapshots at updates 0/10/20/30/40;
  - fsync-locked selections before any assessment of a trained policy, and a single assessment per policy;
  - canonical Elo and the audit hook. The study source runs from the registered lane, outside `~/jope-data`.
- **Selector (strict, inner).** The earliest best snapshot on the SELECT fold by **TOP4_PICK** utility, accepted only if
  the gain is above .001 **and** the family-bootstrap LCB95 is above 0, else r882.
- **Folds.** One fresh fold seed, **260140**, gives the family partition (205 families → 5 × 41), shared by seeds
  260101–260103. The seeds drive batch plans, optimizer streams and bootstraps. The partition is pinned in
  [fold_manifest.json](fold_manifest.json).
- **Cells.** 15 per arm. **Budget.** 15 × 3 × 40 = **1,800 updates**.
- **H239 A1 additions:** cooperative pauses between optimizer steps, evaluation batches and pool loads; per-cell
  fsynced checkpoints hashed into `progress.json`; at most 2 hash-verified resumes, which skip completed cells. No
  re-roll and no deadline reset.

## Statistics and decision

- **Primary panel.** DL-Val91 TOP4_PICK utility of each seed's equal mixture of its 5 cells' selected policies, minus
  TOP4_PICK@r882.
- **Statistic.** The median over 3 seeds of the per-seed equal-family gain. The CI95 is the family-bootstrap median
  (10,000 paired draws, boot seed 260150), with a one-sided p and **Holm over T4_HINGE and T4_PL** (m = 2).
- **ADVANCE** requires all five of:
  - median TOP4_PICK family gain > **+0.002**;
  - family LCB95 > 0;
  - Holm p < .05;
  - **3/3 seeds positive**;
  - single-pick median gain against r882 on the same selected mixtures **≥ −0.002**.

  Ties mean no advance. Beating only CONTROL is descriptive.
- **Also reported:**
  - recall@4 of the oracle best, per arm and seed;
  - single-pick utility against r882 (1.1377941898164525);
  - TOP4_PICK@r3175 = 1.2403226556046079 from H258, as a reference only (not recomputed);
  - the 1,219-context TRAIN out-of-fold panels (TOP4_PICK and single-pick);
  - the unconstrained best-step readout and the UNCONSTRAINED step-40 diagnostic (TOP4_PICK, single-pick and recall@4);
  - ARM − CONTROL contrasts;
  - harm families (loss > 1e-4);
  - Elo by MM Bradley–Terry with TOP4_PICK@r882 = 1500;
  - the step-40 L2 drift from r882 and the per-cell hinge margins.
- **The H152 gate-cliff context.** Context #27 (`fam-9f67f144…`) is reported separately for TOP4_PICK and single-pick.
  It is never excluded.
- **Not used.** No AUC, and no speed verdicts.

## Resources and the A1 memory guard (H239 A1)

- **Compute.**
  - xbabe2, CPU only: `taskset` 44–47, nice 19, one math thread.
  - Tree RSS < 4 GiB, ≤ 4 threads, disk free ≥ 300 GB.
  - No GPU, xbabe0, xbabe1, xbabe3 or live flywheel. No foreign process control. PR #2071, main and v1/fastlane are
    not touched.
- **Admission.** load5 ≤ 100 and MemAvailable ≥ 30 GiB.
- **Cooperative pause.** Triggered by load5 > 108 or MemAvailable < 20 GiB. xbabe2 swap is full, so the guard pauses
  rather than aborts.
  - **Resume** at load5 ≤ 100 and MemAvailable ≥ 25 GiB, sustained for 30 s.
- **Hard stop.** MemAvailable < 12 GiB sustained for 30 s, or < 6 GiB on any sample, plus the RSS, disk, thread and
  deadline stops. Only its own worker group is signalled.
- **Deadlines.**
  - Prereg pushed to v1/next by **16:05Z**.
  - Compute hard stop **17:15Z**.
  - Closeout on v1/next by **18:00Z**.

  If the study is incomplete, it closes as CLOSED INCOMPLETE with honest receipts.

## Custody

- **Repo.** pytest, Ruff, JSON parse and `git diff --check` precede the push. The private, sha-verified `jope-records`
  gates are run and recorded. Under the freeze this wave does not edit CI config, records pins, schemas or the root
  README, so its coverage, test-inventory and source-inventory registrations are deferred until v1/next is rebased on
  main. A durable same-agent review receipt names the exact head.
- **S3.** Full outputs go to create-only S3 `…/rd_breakthrough_20261030/H260/<run_id>/` with VersionId readback.
- **Closeout.** It publishes only the README, a compact metrics JSON and the index rows to v1/next. No checkpoints go in
  Git.
