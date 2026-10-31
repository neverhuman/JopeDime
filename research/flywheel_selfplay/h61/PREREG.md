# H61 Preregistration: Adapter Checkpoint Selection & Strict-Margin Admission Ablation (`S0`–`S3`)

- **Wave ID**: `H61` (`research/flywheel_selfplay/h61/`)
- **Status**: Preregistered (`CPU and docs only` — no runner executed, no `qualification.lock` acquired, no GPU used, zero `DL-Val` / holdout / outer reads)
- **Author / Lane**: `agy_fw_audit` (`Antigravity`, Claude Opus 4.6)
- **Date**: `2026-10-31 UTC` (`2026-10-30 21:05 MDT`, per `~/nudges/agy_fw_audit_h61_prereg_t2105.md`)
- **Execution Queue**: Gated behind `H57 -> H59 -> H60` under `/home/ubuntu/jope_plan/qualification.lock` and root-filesystem `< 95%` `statvfs` disk guard.

---

## 1. Motivating Evidence & Problem Statement

### 1.1 Live `xbabe1` Adapter Admission Audit (`rounds 775–1740`)
Read-only audit of the live `xbabe1` closed-loop log `/home/ubuntu/jope-data/flywheel-full-20261030/run-20261030/adapter_admissions.jsonl` (`966` total rows covering rounds `775` through `1740`) and the live source tree (`research/flywheel_selfplay/run_fused_closed_loop.py` and `crates/jope-telemetry/src/hooks/flywheel.rs`, audited in PR #754) established four empirical facts:

1. **Only 4 adapter promotions occurred across 966 rounds (`0.41%` acceptance rate), and 2 of the 4 were exact `0.0` ties**:
   - `r782`: `candidate_utility = 1.107428`, `incumbent_utility = 1.107428`, `gain = 0.0` (exact tie accepted under `candidate >= incumbent`), `updates = 17`.
   - `r785`: `candidate_utility = 1.137630`, `incumbent_utility = 1.107428`, `gain = +0.030202`, `updates = 14`.
   - `r859`: `candidate_utility = 1.137786`, `incumbent_utility = 1.137630`, `gain = +0.000156` (`+0.0002` rounded), `updates = 14`.
   - `r882`: `candidate_utility = 1.137786`, `incumbent_utility = 1.137786`, `gain = 0.0` (exact tie accepted under `candidate >= incumbent`), `updates = 14`, checkpoint SHA256 `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`.
   - Rounds `883–1740` (`858` consecutive rounds) produced **zero** accepted updates (`incumbent_utility` frozen at `1.137786`).
2. **Extreme quantization of the top-1 selection metric**:
   - Across the last `200` rounds of `adapter_admissions.jsonl` (`rounds 1541–1740`), `candidate_utility` takes **only `13` distinct floating-point values** across 200 training attempts, because hard `argmin` top-1 utility only changes when a discrete argmin index switches on one of the validation contexts (`H55` similarly observed `216/218` dataset ties between `swa8` and `r882`).
3. **Revealed-anchor leakage (`0..3`) in `native` `val_reg` scoring (PR #754 audit)**:
   - In `research/flywheel_selfplay/run_fused_closed_loop.py:469-490` (`score_contexts`) and `crates/jope-telemetry/src/hooks/flywheel.rs:207-258` (`ValidationHarness::utility`), `pred = _predict_context(...)` is evaluated on all `0..P-1` pool candidates and `best = int(np.argmin(pred))` is taken over the **entire** pool `0..P-1` without masking the revealed anchor indices `0..3` whose ground-truth outcomes were just fed into `context_tokens` (`native` mode).
4. **Permissive tie acceptance (`candidate >= incumbent` with `min_gain = 0.0`)**:
   - By default (`strict_admission=False`, `min_gain=0.0`), `ValidationHarness::admit` accepts `0.0`-gain ties (`r782`, `r882`) and sub-noise micro-fluctuations (`r859`, `+0.000156`). PR #744 (`2fc407ce7958b3c3ca330f22f30f92b3a2d4c236`) added `--strict-adapter-admission` (`strict_margin_admission` in `research/flywheel_selfplay/dev_incumbent.py:372-415`), requiring `candidate_utility > incumbent_utility + margin` and `LCB95 > 0.0`.

### 1.2 Hypothesis
The post-`r882` plateau is substantially a **selection-metric and admission-gate problem**: the live loop gates round-end adapter updates on a coarse, heavily quantized, tie-accepting, `native`-leaky (`0..3` unmasked) hard top-1 `argmin` rule. Replacing that selector with a leak-free (`unrev` `4..P-1` masked), strict-margin (`m` calibrated from on-disk null replicates), smoothed (`softmin` expected utility), or family-cluster bootstrap lower-confidence-bound (`LCB95 > m`) selection rule — while holding the starting `r882` checkpoint copy, frozen DOPE backbone, data stream, and per-round gradient updates identical — allows genuine adapter improvements to be retained while rejecting ties and null noise.

---

## 2. Frozen Invariants & Data Hygiene

1. **Starting Incumbent Checkpoint Copy (`r882`)**:
   - Every cell initializes from an isolated, read-only copy of the `r882` incumbent checkpoint:
     - Path: `/home/ubuntu/jope-data/flywheel-20261024/checkpoints/incumbent_checkpoint.pt` (copied to cell-local scratch before use; live run dir is never touched).
     - `R882_SHA`: `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`.
2. **Frozen DOPE Backbone & Adapter-Only Training**:
   - DOPE weights remain strictly frozen at:
     - `DOPE_SHA`: `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`.
   - Verified before and after every cell via `dope_freeze_guard.py`.
   - Only the 6 `adapter.*` tensors (`131,840` trainable parameters) have `requires_grad=True`.
3. **Real S3 Input Manifest Only & Zero Holdout/Outer Exposure**:
   - Canonical manifest path: `/home/ubuntu/jope-data/flywheel-20261024/s3_input_manifest.json`
   - `MANIFEST_FILE_SHA256`: `99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4`
   - `MANIFEST_IDENTITY_SHA256`: `3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3`
   - Partition counts: `1,219` `DL-Train` contexts across `205` dataset families; `97` `DL-Val` contexts across `19` dataset families.
   - **Zero** `Holdout` (`170` rows) or `Outer` (`480` rows) payloads are ever opened, loaded, or scored (`holdout_payloads_opened == 0`, `outer_rows_delivered == 0`).
4. **Split & Seed Protocol**:
   - All development, inner cross-validation, hyperparameter selection (`tau`, margin verification), and round-end selection decisions use **`DL-Train` (`1,219` contexts, `205` families) only**, partitioned into `5` family-grouped disjoint folds using `seed 4900` (with the `H55` `4:1` `fold_a` / `fold_b` family split `164` train families / `41` selection families as the paired within-round selection split).
   - **Seeds**: `3` seeds (`12, 13, 14`).

---

## 3. Experimental Arms (`S0`, `S1`, `S2`, `S3`)

All four arms start from the exact same `r882` adapter copy (`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`) and execute the **exact same sequence of training batches and candidate gradient updates** per `(seed, round)` on `DL-Train` `fold_a` (`164` families, `1,001` contexts). When a candidate update is proposed at the end of a round, the four arms differ **only** in the selection / admission rule evaluated on the family-disjoint `DL-Train` selection split (`fold_b`, `41` families, `218` contexts):

| Arm | Selection Mode | Candidate Pool Mask | Selection Signal $\mathcal{S}(\theta; \mathcal{D}_{\text{sel}})$ | Round-End Admission Rule (vs current incumbent $\theta_{\text{inc}}$) |
| :--- | :--- | :--- | :--- | :--- |
| **`S0`** | Live baseline (`native` top-1, ties accepted) | Unmasked `0..P-1` (`native`, includes revealed anchors `0..3`) | Mean context utility $U_{\text{native}}(\theta) = \frac{1}{\lvert\mathcal{D}_{\text{sel}}\rvert}\sum_{c} u_{c, \arg\min_{j \in 0..P_c-1} \hat{y}_{c,j}(\theta)}$ | Accept iff $U_{\text{native}}(\theta_{\text{cand}}) \ge U_{\text{native}}(\theta_{\text{inc}})$ (`min_gain = 0.0`, ties accepted) |
| **`S1`** | Strict top-1 `unrev`-masked (`#744` `--strict-adapter-admission`) | Masked `4..P-1` (`unrev`, excludes revealed anchors `0..3`) | Family-grouped mean of unrevealed top-1 utility $U_{\text{unrev}}(\theta) = \frac{1}{\lvert\mathcal{F}_{\text{sel}}\rvert}\sum_{f \in \mathcal{F}_{\text{sel}}} \frac{1}{\lvert f\rvert}\sum_{c \in f} u_{c, \arg\min_{j \in 4..P_c-1} \hat{y}_{c,j}(\theta)}$ | Accept iff $\Delta U_{\text{unrev}} = U_{\text{unrev}}(\theta_{\text{cand}}) - U_{\text{unrev}}(\theta_{\text{inc}}) > m$ **and** paired family bootstrap $\text{LCB}_{95}(\Delta U_{\text{unrev}}) > 0$ (ties fail) |
| **`S2`** | Smoothed `softmin` expected utility (`unrev`-masked `4..P-1`) | Masked `4..P-1` (`unrev`, excludes revealed anchors `0..3`) | Family-grouped mean of temperature-smoothed expected utility $U_{\text{soft}}(\theta; \tau) = \frac{1}{\lvert\mathcal{F}_{\text{sel}}\rvert}\sum_{f \in \mathcal{F}_{\text{sel}}} \frac{1}{\lvert f\rvert}\sum_{c \in f} \sum_{j=4}^{P_c-1} w_{c,j}(\theta;\tau)\, u_{c,j}$, where $w_{c,j}(\theta;\tau) = \frac{\exp(-\hat{y}_{c,j}(\theta)/\tau)}{\sum_{k=4}^{P_c-1}\exp(-\hat{y}_{c,k}(\theta)/\tau)}$ | Accept iff $U_{\text{soft}}(\theta_{\text{cand}}; \tau^{\star}) - U_{\text{soft}}(\theta_{\text{inc}}; \tau^{\star}) > m_{S2}^{\star}$ (ties fail), where $(\tau^{\star}, m_{S2}^{\star})$ are chosen **solely** by `DL-Train` 5-fold family-grouped inner CV (`seed 4900`) |
| **`S3`** | Family-cluster bootstrap LCB selection (`unrev`-masked `4..P-1`) | Masked `4..P-1` (`unrev`, excludes revealed anchors `0..3`) | Paired per-family unrevealed utility gain $\Delta u_f(\theta_{\text{cand}}, \theta_{\text{inc}}) = U_{\text{unrev},f}(\theta_{\text{cand}}) - U_{\text{unrev},f}(\theta_{\text{inc}})$ across families $f \in \mathcal{F}_{\text{sel}}$ | Accept iff paired family-cluster bootstrap (`10,000` draws over families) satisfies $\text{LCB}_{95}(\Delta u_f) > m$ **and** $\text{median}_f(\Delta u_f) \ge 0$ (ties fail) |

### 3.1 Controlled Training Trajectory Pairing
To guarantee that `S0`, `S1`, `S2`, and `S3` receive the **exact same training updates** and isolate only the selection rule:
- Each round $t = 1, \dots, R$ (`R = 20` rounds, `k = 2` optimizer steps/round, `batch_size = 16`, `lr = 1e-4`, `weight_decay = 1e-4`, identical to the live `r882` / `H55` update step) uses a deterministic round RNG seeded by `(seed, round_idx)` so that the sampled training contexts and candidate slates are identical across `S0..S3` at round $t$.
- Furthermore, we record both:
  1. **Single-step paired selection (`paired_step`)**: From the exact same pre-update checkpoint $\theta_{t-1}$ and the exact same candidate update $\tilde{\theta}_t$ produced by step $t$, each rule `S0..S3` makes its accept/reject decision, isolating the pure 1-step selection contrast on identical candidate weights.
  2. **Closed-loop multi-round trajectory (`closed_loop`)**: Each arm `S0..S3` carries forward its own accepted incumbent $\theta_t^{(S_k)}$ across rounds $t = 1, \dots, R$ while sharing the exact same per-round training batches and gradient step schedule.

### 3.2 Inner-CV Hyperparameter Selection for `S2` (`DL-Train` Only, `5` Folds, `seed 4900`)
- For `S2`, the smoothing temperature $\tau \in \{0.02, 0.05, 0.10, 0.20, 0.50\}$ and selection margin $m_{S2} \in \{0.000357, 0.001000, 0.001546\}$ are selected **exclusively** using 5-fold family-grouped inner cross-validation on `DL-Train` (`seed 4900`, `205` families partitioned into 5 folds of `41` families each) prior to any terminal evaluation.
- The inner-CV objective is the out-of-fold family-grouped `unrev` (`4..P-1`) mean utility gain vs `r882` (with tie-breaker: highest family-cluster `LCB95` on `DL-Train`). `DL-Val` is **never** consulted during $(\tau^{\star}, m_{S2}^{\star})$ selection.

---

## 4. Calibration of Null Margin `m` from On-Disk `H55` and `H52` Receipts

Per `~/nudges/agy_fw_audit_h61_prereg_t2105.md`, the strict selection margin `m` is calibrated before running any H61 cell as the **95th percentile of null `|gain|`** (`m = P95(|gain_null|)`) from the null and tie replicates already on disk in `H55` and `H52`.

### 4.1 Exact Source Receipts & SHA256 Checksums

| Study | Receipt File Path | SHA256 |
| :--- | :--- | :--- |
| **H55** | `docs/research/studies/dope-neural-ranking-20261021/h55/metrics.json` | `37cf4dcc671aa9517a67e150862699379e12cc1cfe1ebdefb624f90640e6ff71` |
| **H55** | `/home/ubuntu/jope-data/h55/h55-wave-attempt4/outcomes/wave.json` | `3182820707bcb471c5b78483636ff19a15f4658c6341d7cb4c9c27f7afbac6a9` |
| **H55** | `/home/ubuntu/jope-data/h55/h55-wave-attempt4/outcomes/admission.json` | `cc932232d589283820bf5854e20b830271bc3e4cc30ce52f3a85c62a9fca13d1` |
| **H55** | `/home/ubuntu/jope-data/h55/h55-wave-attempt4/outcomes/cells/swa8/seed12/summary.json` | `40e36bab6d4e3f22def366b1e35d8183bdb8018a217d38d1725153d99dcb7819` |
| **H55** | `/home/ubuntu/jope-data/h55/h55-wave-attempt4/outcomes/cells/swa8/seed13/summary.json` | `09f3e6825120e8a4d385f4249f6bb86e471d480d054632469175e0d9d787425a` |
| **H55** | `/home/ubuntu/jope-data/h55/h55-wave-attempt4/outcomes/cells/swa8/seed14/summary.json` | `e581d8504998828096e98429d6ff98c41957a3f5564f5ea49a33f58e5b22e875` |
| **H52** | `docs/research/studies/dope-neural-ranking-20261021/h52/metrics.json` | `b1385a0bc02633587df16269e3878925349f0f2f5fb14f840b61ae88d89ee557` |
| **H52** | `/home/ubuntu/jope-data/h52/runs/audit.json` | `4859a750fbfbac81c1ccb137907408650aa86bcde1ecd43b18729b7a625c4863` |
| **H52** | `/home/ubuntu/jope-data/h52/runs/CONTROL-k2-lr1e-4/seed12/CONTROL-k2-lr1e-4-seed12.json` | `94c9ff8a6232ca26ab2ba04882efdece428562b0b34131c8750c12277d24e76e` |
| **H52** | `/home/ubuntu/jope-data/h52/runs/CONTROL-k2-lr1e-4/seed13/CONTROL-k2-lr1e-4-seed13.json` | `b496574c77264be2dc6968ae5d2859a282c81ecbfeb13f6e494ef775a8976359` |
| **H52** | `/home/ubuntu/jope-data/h52/runs/CONTROL-k2-lr1e-4/seed14/CONTROL-k2-lr1e-4-seed14.json` | `2047df85bf9c92a2e82de6958e34ec4ef33b188fd8b836a3052b02504c24a84c` |
| **H52** | `/home/ubuntu/jope-data/h52/runs/NULL-tau0.1/seed12/NULL-tau0.1-seed12.json` | `80225b2b0f0f79bd61d0feb5bdd2f814c383f7351fb0b9a207949190b2428f20` |
| **H52** | `/home/ubuntu/jope-data/h52/runs/NULL-tau0.1/seed13/NULL-tau0.1-seed13.json` | `23593a769901bbad485461bcabbb4a33b4e946fdb9519101c80321a9608b9e86` |
| **H52** | `/home/ubuntu/jope-data/h52/runs/NULL-tau0.1/seed14/NULL-tau0.1-seed14.json` | `6213a94c325a03b37b4c420dc94bc7bcde28a2a8ce8ef48e7b3d3640d8e5ff5f` |
| **H52** | `/home/ubuntu/jope-data/h52/runs/NULL-tau0.1/seed12/NULL-tau0.1-seed12-family-gains.json` | `8d523c5efb6b5258b4225d6ae0d84d5725882b5517b4fa4dd6a813a9a8ff2f1e` |
| **H52** | `/home/ubuntu/jope-data/h52/runs/NULL-tau0.1/seed13/NULL-tau0.1-seed13-family-gains.json` | `c7b9def7aaa6d4a928cd254c769beb7b45dcf500fdf53791c3d734f6824f6d8b` |
| **H52** | `/home/ubuntu/jope-data/h52/runs/NULL-tau0.1/seed14/NULL-tau0.1-seed14-family-gains.json` | `a43b092d2450dae1f4f10bb1024a7f0f79bd79521cc530a99003ddb69438de21` |

### 4.2 Empirical Null `|gain|` Distribution & Calibrated Margin `m`

From the receipts above, we extract the null `|gain|` replicates across `H55` and `H52`:

| Null Replicate Stratum | Replicate Count ($n$) | Min `\|gain\|` | Median `\|gain\|` | **95th Percentile (`P95`)** | Max `\|gain\|` | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`H55` `r882`-vs-`r882` self-ties** (seeds `12, 13, 14` on `fold_b`, `41` families) | `3` | `0.000000` | `0.000000` | **`0.000000`** | `0.000000` | Deterministic `1.2836196786634644` across all 3 seeds |
| **`H55` `swa8` seed-vs-seed ties** (`seed12 vs 13 = 0.0`, `seed14 vs 12/13 = 5.581e-5`) | `3` | `0.000000` | `0.000056` | **`0.000056`** | `0.000056` | `216/218` dataset ties (`seed12/13`); `217/218` (`seed14`) |
| **`H55` `swa8` intra-seed member-vs-member pairs** ($\binom{8}{2} \times 3$ seeds on `fold_b`) | `84` | `0.000000` | `0.000056` | **`0.000265`** | `0.000401` | 24 `k=2` checkpoints around `r882` (`6` distinct utilities) |
| **`H52` `CONTROL-k2-lr1e-4` seed-vs-seed mean ties** (seeds `12, 13, 14`, `205` families) | `3` | `0.000027` | `0.000053` | **`0.000078`** | `0.000081` | `204/205` families tied (`0.0`) between seeds `12` and `13` |
| **`H52` `CONTROL-k2-lr1e-4` seed-vs-seed fold ties** (`5` folds $\times$ `3` seed pairs, `41` fam/fold) | `15` | `0.000000` | `0.000000` | **`0.000323`** | `0.000323` | `10/15` fold pairs are exact `0.0` ties |
| **`H52` `NULL-tau0.1` placebo seed-vs-seed mean ties** (label-permuted null across seeds `12, 13, 14`) | `3` | `0.000112` | `0.000307` | **`0.000408`** | `0.000419` | Seed means `-0.008040, -0.007928, -0.007621` vs `r882` |
| **`H52` `NULL-tau0.1` placebo seed-vs-seed fold ties** (`5` folds $\times$ `3` seed pairs, `41` fam/fold) | `15` | `0.000000` | `0.000882` | **`0.002411`** | `0.002411` | Single-fold (`41`-family) variance of label-permuted null |
| **Pooled Seed-Level Null/Tie Replicates (`Set A`)** (`H55 r882` `[3]` + `H55 swa8` `[3]` + `H52 k2` `[3]` + `H52 NULL` `[3]`) | `12` | `0.000000` | `0.000055` | **`0.000357`** | `0.000419` | 205-family / full-seed null tie noise floor (`m_seed = 0.000357`) |
| **Pooled Single-Fold (`41`-Family) Null/Tie Replicates (`Set B`, Primary Calibration)** (`H55 r882` `[3]` + `H55 swa8` `[3]` + `H52 k2 folds` `[15]` + `H52 NULL folds` `[15]`) | `36` | `0.000000` | `0.000056` | **`0.001546`** | `0.002411` | **Matches 41-family `fold_b` selection granularity (`m = 0.001546`)** |

**Pinned Margin Specification**:
- **Primary calibrated margin for `S1` and `S3` (`41`-family `fold_b` selection pool)**:
  $$m = \text{P95}_{\text{Set B}}(\lvert\text{gain}_{\text{null}}\rvert) = \mathbf{0.001546}$$
  This is the exact 95th percentile across the $n = 36$ single-fold (`41`-family) null/tie replicates on disk (`H55` `r882` self-ties, `H55` `swa8` seed ties, `H52` `CONTROL-k2` fold seed-ties, and `H52` `NULL-tau0.1` placebo fold seed-ties). Notice that $m = 0.001546$ cleanly rejects both `0.0` ties (`r782`, `r882`), sub-noise micro-accepts (`r859` at `+0.000156`), and `H55` `swa8`intra-seed member jitter (`P95 = 0.000265`), while sitting just above the `+0.001` wave admission threshold.
- **Secondary reference margins recorded for `S2` inner-CV grid**:
  - $m_{\text{seed}} = \mathbf{0.000357}$ (`P95` of `Set A`, the $n = 12$ seed-level null/tie replicates), and
  - $m_{\text{wave}} = \mathbf{0.001000}$ (the preregistered wave admission threshold).
  As specified in Section 3.2, `S2` selects $(\tau^{\star}, m_{S2}^{\star}) \in \{0.02, 0.05, 0.10, 0.20, 0.50\} \times \{0.000357, 0.001000, 0.001546\}$ solely via 5-fold family-grouped inner CV on `DL-Train` (`seed 4900`).

---

## 5. Evaluation Metrics & Reporting Contract

Across the `3` seeds (`12, 13, 14`), every cell is evaluated on the untouched `DL-Train` selection fold (`fold_b`, `41` families, `218` contexts, `1` scoring pass at round end) and in 5-fold family-grouped CV (`205` families, `seed 4900`). We report:

1. **Median and 95% CI across seeds (`12, 13, 14`)** and family-grouped median gain vs `r882` (`10,000` family-cluster bootstrap resamples, `seed 20261031`).
2. **Mean gain and paired family-cluster bootstrap `mean LCB95`** vs `r882` (`unrev`-masked `4..P-1`, where each of the `41` `fold_b` / `205` `DL-Train` families is weighted equally as a cluster unit).
3. **Harm tail (`%families < -0.01`)**: exact fraction and count of families whose per-family utility delta vs `r882` is strictly below `-0.01`, plus the 5 worst families by ID.
4. **Family-grouped Bradley-Terry Elo vs `r882` (`elo_tracker.py`, never AUC)**:
   - Computed via `/home/ubuntu/jope_plan/ns/elo_tracker.py` (`compute_family_grouped_bradley_terry_elo`, `r882` anchored at `1500.0`, `tie_margin = 1e-4`, `prior = 0.01`, `400` iterations, `2,000` family bootstrap resamples) with Win / Tie / Loss (`W-T-L`) family counts.
   - **Never report binary classification AUC.**
5. **Selection Diagnostics**:
   - Number of accepted vs rejected rounds per arm/seed, distinct candidate utilities observed, and agreement rate between `paired_step` decisions and full 5-fold out-of-fold family gains.

---

## 6. Admission Gate & Terminal `DL-Val` Protocol

### 6.1 Strict Pre-Validation Admission Gate (on `DL-Train` Family-Grouped Evaluation)
An arm `S_k` (`k in {0, 1, 2, 3}`) is admitted to a terminal `DL-Val` read if and only if on `DL-Train` held-out family evaluation vs `r882`:
1. **Strict lower confidence bound**: `mean LCB95 > +0.001` (paired family-cluster bootstrap 95% lower bound on mean family utility gain vs `r882`).
2. **Non-negative median**: `median family delta >= 0.0` (and median across seeds `12, 13, 14` `>= 0.0`).
3. **Bounded family harm tail**: `<= 10%` of families have gain `< -0.01` (`%families < -0.01 <= 10.0%`).
4. **Ties fail**: Exact ties with `r882` (`0` non-zero family deltas or `mean gain == 0.0`) are rejected (`admitted = False`).

### 6.2 Single Terminal `DL-Val` Read Rule
- If an arm fails any criterion in Section 6.1 on `DL-Train`, **zero** `DL-Val` payloads are opened or scored (`validation_read_count == 0`).
- If an arm passes all criteria in Section 6.1, **at most one (`1`) terminal `DL-Val` read per admitted arm/seed** (`97` contexts across `19` families) is executed on the frozen final checkpoint of that arm/seed (`validation_read_count <= 1`). `DL-Val` is never used to tune `tau`, `m`, round count, or checkpoint choice.

---

## 7. Resource Budget & Operational Guards

1. **Wall-Clock Budget**: `<= 60` minutes GPU cell wall time total (`<= 5` minutes expected per seed cell for `3` seeds).
2. **GPU Lock Discipline**:
   - Acquire `/home/ubuntu/jope_plan/qualification.lock` **per cell** (releasing between cells if another higher-priority job is queued) and **only after** `H57 -> H59 -> H60` have completed their queued GPU turns.
3. **Root Disk Guard (`< 95%` `statvfs`)**:
   - Before acquiring the lock and immediately after acquiring the lock before launching any cell, check `os.statvfs('/')` available-block usage (`1 - f_bavail / f_blocks < 0.95`). If `/` usage is `>= 95%`, release the lock immediately and wait/abort without writing cell artifacts.
4. **Host CPU & Memory Guards (`xbabe2` swap is full)**:
   - Run all Python processes under `nice -n 10` with `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4`.
   - Enforce host CPU utilization `<= 85%` (`102` logical cores) and `/proc/meminfo` `MemAvailable >= 16 GiB` (`16,777,216 KiB`) floor before and during every cell so `xbabe2` never swaps.
5. **Live Loop Non-Interference**:
   - Never touch `xbabe0` or the live `xbabe1` loop (`/home/ubuntu/jope-data/flywheel-full-20261030/run-20261030/`).

---

## 8. Pre-Outcome Timing & Red-Team Amendment — `2026-10-31` (`t2132`)

Status: Recorded and committed **before** any H61 GPU cell, `fold_b` outcome read, or `DL-Val` read (`outcome_cells_started = 0`, `validation_read_count = 0`).

### 8.1 Projected GPU Cell-Wall Budget from `H57` Timing Receipt
Using the audited `H57` timing receipt (`~/jope-data/h57/h57-wave/timing/timing.json`):
- `measured_generation_seconds`: `60.1135667 s` (`~60.1 s/gen`)
- `timing_cell_wall_seconds` (cell setup + baseline overhead): `128.6147321 s` (`~128.6 s` cell overhead)
- In `H61`, all four selection arms (`S0..S3`) share the same `r882` copy initialization, `fold_a` memory load, and paired `20`-round (`40`-step) update stream per seed (`12, 13, 14`), corresponding to `8` H57-equivalent timing units per seed cell (`3` seed cells total):
  - **Per-seed cell wall**: `128.615 s + 8 * 60.114 s = 609.523 s` (`10.16 min`)
  - **`3` seed outcome cells (`12, 13, 14`)**: `3 * 609.523 s = 1,828.570 s` (`30.48 min`)
  - **`fold_b` / inner-CV scoring reserve**: `128.615 s` (`1` full H57 timing cell wall)
  - **Maximum terminal `DL-Val` reserve (`3` seeds for an admitted arm)**: `3 * 128.615 s = 385.844 s`
  - **Subtotal**: `2,343.029 s`
  - **`+10%` contingency**: `234.303 s`
  - **Projected Total GPU Cell Wall**: **`2,577.332 s` (`42.96 min`)** `<= 3,600.0 s` (`60.0 min` cap; slack `1,022.668 s` / `17.04 min`).

### 8.2 Pre-Outcome Red-Team Safeguards Enforced in `runner.py` & `test_h61.py`
1. **Unified leak-free evaluation mode (`unrev 4..P-1`) across all arms (`S0..S3`) and `r882`**:
   - `S0` selects candidates during round updates using the legacy `native` (`0..P-1`, ties accepted) rule, but all reported utilities, `fold_b` admission contrasts, and family-grouped Bradley-Terry Elo scores for `S0, S1, S2, S3`, and `r882` are evaluated strictly in the **same leak-free `unrev` (`4..P-1`) mode** (`score_fold_b_arms(..., eval_mode="unrev")`).
2. **Ties fail everywhere (`gain > m = 0.001546` AND `LCB95 > +0.001`)**:
   - Both round-end selection for `S1..S3` and wave-level admission require `mean_gain > 0.001546` (`NULL_MARGIN_M`) **and** `mean_lcb95_one_sided > +0.001` (`ADMISSION_LCB`), with exact ties or identical utilities rejected.
3. **Identical training updates per `(seed, round)` across arms**:
   - `build_paired_batches` and `assert_identical_round_streams` verify that `S0..S3` share the exact same `round_rng_seed`, `step_stream_seed`, `context_ids`, `candidate_indices`, and `candidate_pool_sha256` at every round and step.
4. **Strict `DL-Train` `fold_a` closure for `S2` inner CV**:
   - `select_s2_hyperparameters_inner_cv` selects `(tau, m_S2)` solely on `DL-Train` `fold_a` family folds and fails closed (`RuntimeError`) if any `fold_b` or `DL-Val` context or family enters the inner CV.
5. **H59-consistent per-context loss and utility clipping**:
   - Training loss uses `preregistered_targets` (`u` clipped to `[-2.0, 4.0]`, gate failure `-2.0`) and `clipped_loss` (`ROBUST_CAP = 1.0` + detached slate MAD scale `SCALE_FLOOR = 0.05`). Evaluation and bootstrap clip per-context utility deltas to `[-ROBUST_CAP, +ROBUST_CAP]` (`[-1.0, +1.0]`) so `-2.0` gate-failure contexts cannot dominate the family-cluster bootstrap.
6. **Hardened `< 94.5%` `statvfs` disk guard & retryable write-time `InfraVoid`**:
   - Cells start only when `statvfs('/')` used percent is `< 94.5%` (`DISK_START_MAX_USED_PERCENT = 94.5`). Any cell-start or write-time disk guard violation raises `RetryableDiskInfraVoid(retryable=True, same_cell=True, stop_wave=False)`, which retries the **same cell** after disk clearance and never halts the wave.
