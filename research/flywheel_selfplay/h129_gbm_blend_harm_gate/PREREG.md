# H129 — GBM-prior blend: replication, smaller lambda and a fit-fold harm gate vs r882

**Owner:** `rd_opus` (claim `~/jope_plan/ns/receipts/H129_claim.txt`; hwave overseer `rd_opus_t1140`). CPU only.
This file is committed and published **before** H129 opens any DL-Train pool, checkpoint or feature-cache payload
for scoring.

## Motivation (no reuse for selection)

[H124](../h124_gbm_shrink_blend/README.md) blended a fresh `GBM_PROG412_CTX` utility score halfway toward r882.
- **Gain.** It passed the gain gates: median seed gain +0.032, 95% CI [+0.015, +0.054].
- **Harm.** It failed the harm cap: 22 of 205 families (10.7%) were harmed, against a cap of 10%.

H124 is motivation only. None of its fold (12400), seeds (64–66), predictions, utilities, thresholds or outcomes
enters H129. The arms and the gate rule below are fixed from first principles here, not tuned on H124.

## Fixed protocol

The data, model recipe, endpoint and reference are H124's. The fitting, loading, r882 reference pass, read boundary
and RSS watchdog are the merged H124 module (`h124_gbm_shrink_blend/runner.py`, sha256 `9ebce965…`) run unchanged,
with H129's fold, sampling seed and model seeds.

**Data.**
- The pinned real-S3 H50 DL-Train packet: 1,219 regression contexts in 205 families.
- The pinned H58 features. Only train-row byte ranges are read (H124 Amendments 1–2), and 0 DL-Val rows are decoded.
- A byte-verified copy of r882.
- DOPE stays frozen at `c51a56d0…`.
- No DL-Val, holdout, sealed, meta-test, outer-test or live-loop data.
- r882 had earlier exposure to these families, so this is **conditional development evidence**.

**Split and fits.**
- A **new** family fivefold split: sort the family ids, permute them with `default_rng(129029)`, and cut five
  41-family folds.
- 192 unrevealed training programs per context, sampling RNG `129029`.
- **New, previously unused model seeds 91, 92, 93.**
- Per seed: one outer HistGBM per held-out fold, plus ten inner-pair models. That is **45 fits** in total, with the
  H119/H85 recipe (`max_iter=300`, `learning_rate=0.08`, `max_leaf_nodes=63`, `l2_regularization=1.0`).
- Positive-slope fit-only calibration from the inner predictions, exactly as in H124.

**Arms.** Costs are negated to utility scores, so blending in cost space equals the registered utility interpolation.

| Arm | Score per candidate | Role |
|---|---|---|
| `CONTROL` (r882) | r882 | reference (Elo 1500) |
| `BLEND_050` | r882 + 0.5·(GBM − r882) | primary: H124 replication |
| `BLEND_025` | r882 + 0.25·(GBM − r882) | primary |
| `HARM_GATED_050` | `BLEND_050` if the context's disagreement D ≤ τ, else r882 | primary |
| `PURE_GBM` | GBM | report-only, no gate |

**Disagreement.** D = 1 − Spearman rank correlation (average ranks for ties) between the calibrated GBM cost and the
r882 cost over unrevealed slots 4..P−1. D is outcome-free, ranges from 0 to 2, and is unaffected by the
positive-slope calibration. A constant vector gives D = +∞, which closes the gate.

**Gate threshold τ, frozen from fit folds only.** For each seed and outer fold k:
1. Use the inner out-of-fold calibrated GBM predictions of the four fit folds; context c in fold j comes from the
   model that excludes {k, j}.
2. For each fit-fold context, compute D and the `BLEND_050` − r882 U4 delta.
3. The candidate τ values are the fit-fold D quantiles {0.25, 0.5, 0.75, 1.0}.
4. A candidate is feasible if the gated fit-fold family harm fraction is ≤ 5%. Harm uses the same < −0.01 family
   rule, here with a single seed.
5. Pick the feasible candidate with the largest fit-fold equal-family mean gain. On a tie, pick the smaller
   quantile. If none is feasible, τ = −∞ and the arm equals r882 for that fold.

The held-out fold's outcomes are never used to choose τ.

**Endpoint and admission.**
- **Endpoint.** H119 Track-2 U4: the maximum utility among revealed slots 0..3 and the arm's four best-scored
  unrevealed candidates.
- **Aggregation.** Contexts are averaged within family, and the 205 families are weighted equally. Each arm is
  paired with r882 within seed and family.
- **Admission.** Each primary arm passes only if all three hold:
  - the median of its three seed-mean family gains is **> 0**;
  - it is **rejected by Holm (m = 3, α = .05)**, using a one-sided family-cluster bootstrap p-value against the
    margin +0.001. The p-value is (1 + #{bootstrap median ≤ 0.001}) / (10,000 + 1), from 10,000 draws with RNG
    129000, keeping all three seeds of each sampled family. The unadjusted LCB95 and 95% CI are reported too.
  - **harm ≤ 10%**, where harm is the fraction of families whose three-seed mean delta is < −0.01.
- **Thresholds are as written.** No rounding, re-thresholding or harm redefinition is allowed.
- **Elo.** Report the registered median-of-seed Bradley–Terry Elo against r882 = 1500 with a 95% family-paired
  bootstrap interval (1,000 draws, RNG 129001), as the merged H124 helper computes it. The pooled-score Elo is
  labelled additional. Elo is report-only, and AUC is not used.
- **Verdict.** If any primary passes every gate, the verdict is `CANDIDATE: <arms>` and it is flagged as a
  **CANDIDATE BREAKTHROUGH** for overseer red-team before any further action. Otherwise it is `NO_ADVANCE`.

**Completeness.** No outcome summary is written unless all 45 fits complete and every arm covers 1,219 contexts × 3
seeds. The size of `metrics.json` is checked **before** it is written (< 100 KB). Per-family arrays go to an evidence
file whose SHA-256 is recorded in the metrics.

## Resource limits

- **Placement.** CPU only, `nice 15`, OMP/MKL/OpenBLAS at most 4 threads. No GPU, CUDA, qualification lock or
  ticket.
- **Memory.**
  - The worker RSS ceiling is 6 GiB, enforced by the H124 watchdog.
  - MemAvailable must be ≥ 30 GiB at launch and before every fit.
  - The H124 checks between fits stay as well: load ≤ 85/102 and MemAvailable ≥ 16 GiB.
- **Disk.** Filesystem use must stay below 94%.
- **Scratch.** Scratch stays under 2 GB under `~/jope-data/h129/run-*`. Evidence custody is S3 (create-only) plus
  `/home/ubuntu/jope-rescue-20260921/`. Scratch is removed only after the custody copies are verified.
- **Abort.** A breached limit aborts the attempt, which is then recorded as incomplete. A rerun needs a new
  preregistered decision.

The live flywheel is never touched.
