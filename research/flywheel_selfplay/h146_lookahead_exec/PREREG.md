# H146 — Lookahead optimizer on the r882 adapter continuation

**Owner:** rd_opus [OPUS]. Authority: `~/nudges/rd_opus_t2010.md`. Claim: `~/jope_plan/ns/receipts/H146_claim.txt`.
rd_argon cross-reviews read-only.

**Status:** preregistration, committed before any scored inference or optimizer update. This is exploratory
research on repeatedly exposed DL-Val. It is not sealed-test qualification or serving authorization.

## Why Lookahead, and what is reused

Lookahead is the one optimizer idea never scored here:
- H93's LA arm died before scoring (disk gate, then a scratch collision).
- H122 built a CPU executor but it was NOT_RUN.

The following were already tested in H46–H144 and are **not** repeated here: SWA/EMA, rank-loss weight,
strict-margin admission, weight-decay sweep, SAM, cosine restarts, target smoothing, clip/norm and curriculum.

**Code provenance.**
- The executor is a copy of merged **H144** (`runner.py`, `runtime.py`, `science.py`, `monitor.py`,
  `test_h144.py`), whose original SHAs equal the H144 manifest's `executor_sha256`. It is the live CTRL recipe
  ("H140/H144 control") and the same inference unit.
- The Lookahead state is copied from **H93/H122** (`LookaheadState`). H122 and H144 files are not edited.

**Changes from H144, and nothing else:**
- the arms and the Lookahead state;
- seeds, fold seed and bootstrap RNG;
- Holm family size 2;
- xbabe1 input paths;
- a host-pressure pause at MemAvailable < 12 GiB;
- a seed-pseudoreplication disclosure.

## Fixed arms (declared before any outcome)

Every arm uses the same inner **AdamW (learning rate 1e-4, decoupled weight decay 1e-4)** and the H140 live recipe:
- 40 updates (20 rounds × 2 steps);
- batch 16, slate 192: the first 48 fitness-ranked candidates plus 144 sampled;
- gradient clip 1.0;
- SmoothL1 plus 0.1 × listwise KL at temperature 0.2;
- eval mode throughout;
- 47 trainable non-anchor adapter tensors (**29,211 parameters**).

All arms within a seed share the exact context/candidate batch plan and the per-step Torch seed.

| Arm | Optimizer | Evaluated weights |
|---|---|---|
| **CTRL** | AdamW only | final fast weights |
| **LA** | AdamW wrapped in Lookahead, **k = 5, α = 0.5** | slow = fast after the step-40 sync |
| **LA_K10** | AdamW wrapped in Lookahead, **k = 10, α = 0.5** | slow = fast after the step-40 sync |

**Lookahead mechanics.**
- The slow weights start as a clone of the r882 adapter.
- After every k-th fast step: `slow ← slow + α·(fast − slow)`, then `fast ← slow`.
- The inner AdamW moments are kept.
- Synchronization occurs at steps 5, 10, …, 40 (k = 5) and at 10, 20, 30, 40 (k = 10). The runner refuses any
  other schedule.
- **LA_K10 is declared here, before any outcome.** It fits the nine-cell budget used by H144.

**Seeds and folds.**
- Fresh seeds **146101, 146102, 146103**.
- Fresh family fold seed **146140**: sort the 205 DL-Train families, permute with `default_rng`, and split into five
  41-family folds.
- Fold 0 (242 contexts) is **secondary and descriptive only**. Folds 1–4 (977 contexts) supply every training row.
  See [fold_manifest.json](fold_manifest.json).
- 9 cells × 40 = **360 updates**.
- Every cell starts from the same SHA-pinned r882 copy with a new optimizer, sequentially.
- There are no cached cells. An existing attempt directory or started ledger is refused.

## Inputs and boundaries

**Checkpoint.** The owned copy is `/home/ubuntu/jope-data/h146/inputs/round-882.pt`, SHA-256
`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`. It was copied **read-only** from the xbabe1 live
run at 2026-11-01T02:08:13Z, and the live directory is never written.

**Reference.** `/home/ubuntu/jope-data/h146/inputs/best_dev_checkpoint.json`, SHA-256 `30853cad…`. Evaluation
identity `e79a7715c1531ccb8c4055ab96971101cd4034ec3a71d83bca6d25b0e604ca83`.

**Data.**
- The real-S3-derived H50 selected train+val packet (`981960ad…`), source manifest (`99a53569…`), per-context
  SHA-pinned pools and fused source (`b9f1468a…`).
- The H58 token cache (`75704e82…`).
- The JopeDimeMicro constructor assets are identical to H144's pins.

All are pinned in [manifest.json](manifest.json).

**Data scope.**
- 1,219 DL-Train contexts in 205 families, and 91 DL-Val contexts in 19 families, with **no family overlap**.
- No validation context enters a minibatch.
- The audit hook allows only the exact selected paths and the owned output directory, and refuses sealed, holdout,
  meta-test, `qualification.lock` and live paths, as well as any write to an input.

## Gates (regression, frozen state)

1. **Regression first, before any cell.** The unchanged r882 copy must reproduce all 91 reference native utilities
   with **zero** maximum per-context error and zero mean error (1.1377942). Otherwise the run is INCOMPLETE, with zero
   updates.
2. **Frozen state.** Before and after every optimizer step and after inference, the complete non-adapter state
   (anchor, normalizers, buffers) is compared by raw-byte digest. DOPE must equal `c51a56d0…` throughout; on any
   difference the run aborts.
3. **Logging.** Updates, losses, adapter SHA, **adapter L2 drift** and Lookahead sync steps are logged per cell.
   A zero drift is a failed cell.

## Primary estimand and decision (the same inference unit as H144)

**Estimand.** The equal-family native argmin utility on the **19 DL-Val families**: average contexts within each
family, then the 19 families within a seed, then take the **median over the three seed gains**. It is reported
against r882 and against the same-seed CTRL.

**Bootstrap.**
- 10,000 paired family-cluster draws, RNG **146141**.
- Each draw samples 19 families once and shares them across the seeds; the statistic is the median across seeds.
- Two-sided 95% percentile CI.
- One-sided p = (1 + #{draw ≤ 0}) / 10,001.
- Holm step-down over the **two non-control arms**.

**ADVANCE** for LA or LA_K10 requires all of:
- median seed gain against r882 strictly > +0.001;
- 95% CI lower bound strictly > 0;
- at least 2 of 3 seeds with a gain strictly > 0;
- **at most 80%** of the positive family gain concentrated in the top 3 families, using the median-defining seed;
- Holm-adjusted p < 0.05.

Ties count as rejections. The CTRL comparison is reported but is not an extra gate. A missing cell gives
INCOMPLETE; no valid arm gives NO_ADVANCE.

**Also reported:**
- invalid (−2) pick counts per seed and arm, and for the baseline;
- paired family Elo against r882 = 1500 (ties 0.5, regularization 0.01), as the median across seeds with the
  bootstrap 95% CI. AUC is not used.
- **Seed pseudoreplication.** Whether each arm's utility vectors and adapters are identical across seeds. H140 and
  H142 saw identical utilities, a quantized plateau. If so, the seeds are disclosed as pseudoreplicates.

**Secondary.** DL-Train fold 0 gets the same contrasts, invalid counts and Elo. It is descriptive only.

If an arm advances, the closeout writes an adoption note for the flywheel owner. The live loop is never changed.

## Resources and execution

- **Placement.** xbabe1 CPU only, under the supervisor `monitor.py`, at nice 19 with one numerical thread.
- **Monitoring.** Every 0.25 s the supervisor polls **the total OS threads of supervisor, worker and descendants
  (≤ 4)** and their RSS (**< 6,000,000,000 B**). It also requires free disk ≥ 40,000,000,000 B and owned scratch
  < 3,000,000,000 B, and aborts on any breach.
- **Pausing.** The worker group is paused (SIGSTOP) when load1 > 85 or MemAvailable < 12 GiB, and resumed when
  load1 < 80 and memory has recovered. Pauses are recorded.
- **Exclusions.** No GPU, no `qualification.lock` (held by flywheel PID 2297929), and no contact with the live
  flywheel run directory or its screen.
- **Launch.** The run executes once, from byte-verified merged source. `launch.json` records the merged source
  commit and the manifest SHA.
- **Failure.** An OOM or disk kill is recorded as INCOMPLETE with its cause. At most one retry, with the identical
  protocol and recorded as such.
- **Pre-merge checks.** Data-free tests, a constructor-only smoke (0 updates, 0 scored contexts) on xbabe1, Ruff,
  and `git diff --check`.
- **Closeout.** README, `metrics.json` under 100 KB and one H146 index row. Bulk outputs go to S3 custody
  (create-only, version-pinned) before scratch is deleted.
