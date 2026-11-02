# H235 — heteroscedastic beta-NLL head for the r882 adapter continuation, xbabe2 CPU

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hw_t2300_opus_h235_betanll.md` (watcher, 2026-11-02 23:05Z).
**Claim:** `~/jope_plan/ns/receipts/H235_claim.txt`, taken under `Hwave-admission.lock`. This is a diagnostic only. It
cannot advance a model, a serving release or a live flywheel checkpoint. DOPE stays `c51a56d0`. Training runs offline
on a read-only COPY of r882; the live flywheel is never read, written, paused or preempted. **This prereg merges
before anything runs.**

## Hypothesis

No earlier H-wave (H40–H234) used a heteroscedastic, Gaussian-NLL or variance-head loss. Huber and clipping
(H125/H161/H187), target smoothing (H121/H137), focal loss (H173) and SWAG (H174) are different. **H235 asks: if
noisy contexts dominate the regression loss, does a learned per-candidate variance, trained with beta-NLL (Seitzer et
al. 2022, β = 0.5), down-weight them and improve DL-Val91 utility over r882 by more than +0.002, at the incumbent's
40-update budget?**

## The variance head

- **Architecture.** **One** linear layer, `Linear(32, 1)` (33 parameters), on the adapter's per-candidate query
  embedding (`cand_q`, shape `[B, N, 32]`). It gives a log-variance clamped to [−6, 6].
- **Initialization.** Weight 0 and bias 0, so the log-variance is 0 (σ² = 1) at update 0.
- **Scoring uses the mean output only.** Every evaluation (selector panels, A, DL-Val, Native91) goes through the
  unchanged mean path, so update-0 predictions are r882's, bit for bit.
  - The exact Native91 gate proves this before TRAIN opens.
  - A pre-merge probe on xbabe2 confirmed it on the real model: the mean prediction is bit-identical with and
    without the hidden output, and the zero-initialized head gives a log-variance of exactly 0. The probe made no
    update and computed no utility.
- **Trainables.** The 47 adapter tensors (29,211 parameters) plus, for the beta-NLL arms only, the head. The head
  starts from zero in every trajectory. DOPE is untouched.
- **Custody.** Head weights are saved alongside each snapshot (`*-varhead.npz`).

## Loss arms

**beta-NLL**, per candidate, mean over candidates:

    stopgrad(σ^(2β)) · (½ log σ² + (y − μ)² / (2σ²)),   β = 0.5

Here μ is the adapter's cost prediction and y is the CTRL loss's own cost target: fitness clamped to [−4, 2] where
the gate passes, else 2.

**Reading of the CTRL loss (recorded before any outcome).** The inherited H146 CTRL loss is

    smooth_l1(pred, target) + 0.1 · KL( softmax(−target/0.2) ‖ softmax(−pred/0.2) )

Its regression term is smooth-L1: Huber with δ = 1, squared for |error| < 1. The assignment's fallback ("if the CTRL
loss has NO squared-error regression term, BNLL_MIX = CTRL + 1.0 · beta-NLL") is therefore **not** taken. The
assignment's primary design applies: **BNLL_MIX replaces the regression term with beta-NLL** and keeps the KL term
unchanged. With σ² = 1, beta-NLL is half the squared error, so BNLL_MIX starts as a squared-error variant of CTRL; the
head then learns to down-weight noisy candidates. Tests pin this reading of the CTRL loss, and check that its split
into regression and KL terms reproduces the loss exactly.

| Arm | Loss | Trainables | Role |
|---|---|---|---|
| r882 | none (frozen incumbent) | — | reference (Elo 1500) |
| CONTROL | CTRL loss (unchanged) | adapter | incumbent recipe |
| **BNLL_MIX** | beta-NLL + CTRL's 0.1 · KL term | adapter + head | **primary** |
| BNLL_ADD | CTRL loss + 0.5 · beta-NLL | adapter + head | secondary (Holm m = 2) |

Every arm uses the identical exact H170 CTRL AdamW recipe: lr 1e-4, wd 1e-4, betas (0.9, 0.999), constant. Global
grad-norm clipping at 1.0 covers each arm's trainables.

## Machinery (merged H234 / H232, with the H232 A1/A2 fixes)

- **Inherited unchanged:**
  - the H146 runtime (re-pinned after #2006) and the early-`research`-namespace release;
  - H156's 8-context, 24-candidate batches;
  - snapshots at steps 0/10/20/30/40;
  - the strict inner selector: earliest best step, gain > .001 with a 95% bootstrap LCB > 0, else r882;
  - locks fsynced before assessment;
  - canonical Elo, the exact Native91 gate and the audit hook.
- **Fresh per-seed family folds.** Seeds 235101–235103 use fold seeds 235141–235143. Best-match Jaccard is ≤ 0.176
  within H235 and ≤ 0.182 against H170, H232 and H234. They are pinned in [fold_manifest.json](fold_manifest.json).
- **Budget.** 15 cells × 3 arms × 40 updates = **1,800 updates**.

## Statistics and decision

- **Decision panel: DL-Val91** (91 contexts, 19 families). Each seed averages its five selected policies.
- **Direction check first.** Higher is better. r882 Native91 must be exact (1.1377941898164525) before TRAIN opens.
- **Statistic.** The median over the 3 seeds of the per-seed equal-family gain against r882. The CI95 comes from
  10,000 shared family draws (seed 235150). The one-sided p is Holm-adjusted over BNLL_MIX and BNLL_ADD.
- **ADVANCE (per arm)** iff all of:
  - median gain against r882 **> +0.002**;
  - family LCB95 **> 0**;
  - Holm p **< 0.05**.

  Ties are no advance. The primary contrast is against **r882**. Beating only CONTROL is descriptive (hw_t2200).
- **Also reported, not gated.**
  - Contrasts and Elo against CONTROL.
  - The family out-of-fold panel and the unconstrained best step.
  - Harm: families whose three-seed mean delta is < −0.01.
  - Positive seeds, selections and drift.
  - The log-variance traces.
  - Each arm's DL-Val91 context mean against the **plateau best 1.1389572** (r3175), descriptively.
- **Elo and utility only.** No AUC and no speed verdict.

## Data, reads, frozen state, host and guards

- **Data.** Real S3-derived data only, with the 276 dataset IDs joined to `origin=real` ([provenance.json](provenance.json)).
  The H50 packet and H58 cache are pinned by SHA-256 in [manifest.json](manifest.json), with host paths re-pinned
  for xbabe2.
- **Checkpoint.** A read-only r882 copy at `~/jope-data/h235/inputs/round-882.pt` (`0084479a…`).
- **Never read.** Sealed, holdout or meta-test data, and the live loop.
- **Frozen state.** DOPE and every non-adapter byte are checked after every update.
- **Host.** xbabe2 CPU (MemAvailable about 70 GiB). `taskset` cores **40–43**, which avoid H231 (2, 22, 67, 104) and
  H233 (0–7). Nice 19, one math thread. No GPU; if one were ever needed, H235 stops and would queue through
  `~/jope_plan/qualification.lock`.
- **Load.** Load5 > 60 pauses the worker's own process group, which resumes at ≤ 55.
- **Hard stops** (own process group only): RSS ≥ 4 GiB, MemAvailable < 20 GiB, disk free < 300 GB, more than 4
  threads, and the compute stop at **2026-11-03T05:00:00Z**. Admission needs load5 ≤ 55.
- **Order.**
  1. Prereg merged by 00:30Z.
  2. Constructor smoke: 0 updates.
  3. The single full attempt: Native91 first, then TRAIN.

  There are no retries.
- **Closeout** by 05:30Z: README, a compact `metrics.json` (with `dope_before`/`dope_after`; the full metrics go to
  S3), the INDEX.jsonl row and the overnight index row. Otherwise H235 closes INCOMPLETE with honest receipts.

## Custody

The run root and logs go create-only to
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H235/<run_id>/`, with a manifest. No large
objects go into git.
