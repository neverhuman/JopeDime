# H234 — LambdaLoss (NDCG-weighted pairwise) for the r882 adapter continuation, xbabe1 CPU

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hw_t2220_opus_h234_listnet.md` (watcher, 2026-11-02 22:20Z).
**Claim:** `~/jope_plan/ns/receipts/H234_claim.txt`, taken under `Hwave-admission.lock`. This is a diagnostic only. It
cannot advance a model, a serving release or a live flywheel checkpoint. DOPE stays `c51a56d0`. Training runs offline
on a read-only COPY of r882; the live flywheel is never read, written, paused or preempted. **This prereg merges
before anything runs.**

## Which loss, and why not ListNet (decided before any outcome)

The assignment proposes a ListNet top-1 cross-entropy, with one rule: "If you find the CTRL loss is ALREADY a
listwise softmax-CE, replace both arms by LambdaLoss NDCG-weighted pairwise (LAMBDAMIX = CTRL + 0.5·Lambda, LAMBDA =
Lambda only)."

The merged CTRL loss (H146 runtime `loss`, inherited by H170/H232) is already of that form:

    smooth_l1(pred, target) + 0.1 · KL( softmax(−target/0.2) ‖ softmax(−pred/0.2) )

The KL term over within-context softmaxes is ListNet top-1 cross-entropy up to a target-only constant, at temperature
0.2. Adding ListNet would re-weight a term CONTROL already has, so **H234 tests LambdaLoss**. A data-free test pins
this reading of the CTRL loss.

## Hypothesis

No earlier H-wave (H40–H233) used a listwise or NDCG-weighted loss beyond CTRL's KL term. H47, H155, H190, H198 and
H202 were plain pairwise, and H173 was focal. The flywheel picks the argmin-cost candidate, so the top of each
context's ranking matters most. **H234 asks: does an NDCG-weighted pairwise loss, which concentrates the gradient on
pairs that change the top of the ranking, find adapter updates that lift DL-Val91 utility over r882 by more than
+0.002, at the incumbent's 40-update budget?**

## LambdaLoss (as implemented in `treatment.lambda_loss`)

Per context (24 candidates per training context), with the CTRL loss's own cost target
`t = clamp(fitness, −4, 2)` where the gate passes, else 2.0:
- **Utility and score.** Utility is `u = −t` and score is `s = −pred`; lower cost is a higher score.
- **Graded relevance.** `rel = (u − min u) / (max u − min u)`, with gain `2^rel − 1`.
- **Pair weights.** For each pair with `u_i > u_j`, the weight is `|ΔNDCG_ij| = |gain_i − gain_j| ·
  |1/log2(1+r_i) − 1/log2(1+r_j)| / IDCG`, with ranks `r` from the current scores (detached). This is LambdaRank's
  weight (Burges 2010), the NDCG case of the LambdaLoss framework (Wang et al. 2018).
- **Pair loss.** `softplus(−σ·(s_i − s_j))` with σ = 1.
- **Reduction.** A context's loss is the weight-normalized mean over its pairs. A context with no strictly ordered
  pair contributes 0. The batch loss is the mean over contexts.

A test checks the implementation against an explicit pairwise loop to 1e-9. Further tests check that a perfectly
ordered prediction scores below a reversed one, that constant targets give 0, and that the gradient lowers the best
candidate's cost.

## Arms (identical optimizer; only the loss differs)

| Arm | Loss | Optimizer | Role |
|---|---|---|---|
| r882 | none (frozen incumbent) | — | reference (Elo 1500) |
| CONTROL | CTRL loss (unchanged) | AdamW, lr 1e-4, wd 1e-4, betas (0.9, 0.999), constant (exact H170 CTRL) | incumbent recipe |
| **LAMBDAMIX** | CTRL + 0.5 · Lambda | same | **primary** |
| LAMBDA | Lambda only | same | secondary (Holm m = 2) |

## Machinery (merged H232, with its A1/A2 fixes)

- **Inherited from merged H232 / H170:**
  - the H146 runtime (re-pinned after #2006) and the early-`research`-namespace release (H232 A2);
  - H156's 8-context, 24-candidate batches;
  - clipping at 1.0;
  - snapshots at steps 0/10/20/30/40;
  - the strict inner selector: earliest best step, gain > .001 with a 95% bootstrap LCB > 0, else r882;
  - selection locks fsynced before any trained-policy assessment;
  - canonical Elo;
  - the exact Native91 gate before TRAIN outcomes;
  - the audit hook.
- **New: fresh per-seed family folds.** Seeds 234101–234103 use fold seeds 234141–234143, one partition per seed, so
  the cell split is computed per (outer, seed). Best-match Jaccard is 0.152–0.158 within H234, and at most 0.169
  against H170's and H232's partitions. They are pinned in [fold_manifest.json](fold_manifest.json).
- **Budget.** 15 cells × 3 arms × 40 updates = **1,800 updates**, at the same per-trajectory budget as CONTROL.

## Statistics and decision

- **Decision panel: DL-Val91** (91 contexts, 19 families). Each seed averages its five selected policies (the H170
  uniform policy mixture).
- **Direction check first.** Higher is better. r882 Native91 must be exact (1.1377941898164525) before TRAIN opens.
- **Statistic.** The median over the 3 seeds of the per-seed equal-family gain against r882. The CI95 comes from
  10,000 shared family draws (seed 234150). The one-sided p is Holm-adjusted over LAMBDAMIX and LAMBDA.
- **ADVANCE (per arm)** iff all of:
  - median gain against r882 **> +0.002**;
  - family LCB95 **> 0**;
  - Holm p **< 0.05**.

  Ties are no advance. The primary contrast is against **r882**. A result where an arm beats only CONTROL is
  descriptive and never an advance (hw_t2200): the AdamW continuation is known to degrade r882.
- **Also reported, not gated.**
  - Contrasts against CONTROL.
  - Elo against r882 and CONTROL.
  - The family out-of-fold panel and the unconstrained best step.
  - Harm: families whose three-seed mean delta is < −0.01.
  - Positive seeds, selections, drift and pseudoreplication.
- **Elo and utility only.** No AUC and no speed verdict.

## Data, reads, frozen state, host and guards (as H232)

- **Data.** Real S3-derived data only, with the 276 dataset IDs joined to `origin=real` ([provenance.json](provenance.json)).
  The H50 packet and H58 cache are pinned by SHA-256 in [manifest.json](manifest.json).
- **Checkpoint.** A read-only r882 copy at `~/jope-data/h234/inputs/round-882.pt` (`0084479a…`).
- **Never read.** Sealed, holdout or meta-test data, and the live loop.
- **Frozen state.** DOPE and every non-adapter byte are checked after every update. Only the adapter trains.
- **Host.** xbabe1 CPU, `taskset` 4 cores, nice 19, one math thread. No GPU; if one were ever needed, H234 stops
  and would queue through `~/jope_plan/qualification.lock`.
- **Load.** Load5 > 60 pauses the worker's own process group, which resumes at ≤ 55.
- **Hard stops** (own process group only): RSS ≥ 4 GiB, MemAvailable < 20 GiB, disk free < 150 GB, more than 4
  threads, and the compute stop at **2026-11-03T04:30:00Z**.
- **Order.**
  1. Prereg merged by 23:30Z.
  2. Constructor smoke: 0 updates.
  3. The single full attempt: Native91 first, then TRAIN.

  There are no retries. H232 took 29 min; H234 should take about 30 min.
- **Closeout** by 05:00Z: README, a compact `metrics.json` (with `dope_before`/`dope_after`; the full metrics go to
  S3), the INDEX.jsonl row and the overnight index row. Otherwise H234 closes INCOMPLETE with honest receipts.

## Custody

The run root and logs go create-only to
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H234/<run_id>/`, with a manifest. No large
objects go into git.
