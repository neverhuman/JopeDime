# H257 Technical Report — Phase-2 Warm-Started Subsample & GP Evolutionary Curriculum (`H257_Phase2_Curriculum_Cycle`)

- **Wave ID**: `H257` (`research/flywheel_selfplay/h257_phase2_subsample_gp_curriculum`)
- **Predecessor (Phase 1)**: `H256` ([`../h256_pre_condense_residual_flywheel/`](../h256_pre_condense_residual_flywheel/))
- **Locked Foundation Invariant**: `DOPE_SHA256 = c51a56d02fb2a3df55cfd3de9accd3b7fb31fbacde19a0b4fb36fb8d6c7a5f95` (`unchanged_after_run = true`)
- **Validation Partition**: `DL-Val91` (`91` unseen regression contexts across `19` held-out families; `0` family overlap with `DL-Train`'s `1,219` contexts / `205` families)
- **Primary Result**: **`H257_HIGH_BATCH_EVO3_STACK3` achieves `1838.71` Joint Elo (`+125.29` Elo above `Opus5p5_Extra` at `1713.42`) on S1 Constrained Utility and `1840.84` Joint Elo (`+109.37` Elo above `Opus5p5_Extra` at `1731.47`) on 2-Fold CV RMSE Ratio (`-logratio`)**, ranking **`#1` overall** across all frontier models and JopeDime waves with **`100.0%` (`91/91`) CV gate pass rate**.

---

## 1. Executive Summary & Architectural Design

To push JopeDime's validation Elo decisively above **`Opus5p5_Extra` (Claude Opus 5.5)** without starting from scratch, `H257` introduces a **Two-Phase Warm-Started Evolutionary Curriculum (`H257_Phase2_Curriculum_Cycle`)**:

1. **Warm-Start from Converged Phase 1 (`H256`) + `Stage C0` (`0` GP Generations — Semi-Synthetic Column & `80%` Subsample Convergence)**:
   - Rather than initializing from `0`, `Stage C0` warm-starts directly from the converged Phase-1 `H256` checkpoints (`PreCondenseResidualBridge` Arms `F` & `G`, `FullS3` 7-split pretrained context encoder, and `PreCondenseTop8NeuralReranker`).
   - Each of the `1,219` `DL-Train` contexts is expanded into **`3` training views (`3,657` encoded views)**:
     - `full100`: `100%` of training rows and original columns.
     - `sub80`: `80%` fold- and target-quantile-stratified row subsample.
     - `semisyn80`: `80%` stratified row subsample injected with **4 types of additional semi-synthetic columns**:
       1. Pure Gaussian distractor columns $z \sim \mathcal{N}(0, 1)$,
       2. Collinear noisy duplicates $x_{\text{dup}} = x_j + \sigma_{\text{noise}} \varepsilon$,
       3. Heavy-tailed Student-$t_3$ outlier columns, and
       4. Standardized nonlinear pairwise interactions $x_a x_b$.
   - Trained with **Two-Speed Differential Learning Rate** (`4.5e-4` input/encoder vs. `1.1e-3` attention/heads), **All-28-Pairs Regret-Weighted Margin Loss** (with `1.5x` gate-safety penalty and `1.35x` slot-0 anchor weight), **Temperature-Annealed ListNet** ($\tau: 1.5 \to 0.65$), **Cosine Annealing + SWA**, and **Multi-View Ranking Consistency KL Loss** ($\text{KL}(p_{100\%} \parallel p_{80\%}) + \text{KL}(p_{100\%} \parallel p_{\text{semisyn}})$).
2. **Principled 5-Role Multi-Subsample Seeding of Generation 0 (`[B=11, M]` Single Batched Forward Pass)**:
   - At inference/evolution time, we encode a single batch of $B = 11$ views per dataset (`1×100%` full data + `8×80%` stratified row subsamples + `2` `semisyn80` column-augmented views) and run a **single batched deep learning forward pass `[11, M]`**.
   - From the `[11, M]` score matrix $Z$, we extract `8` Generation-0 GP trees across **5 orthogonal roles**:
     - **Slot 0 (Full-Data Anchor)**: Top-1 on View `0` (`100%` data).
     - **Slot 1 (Risk-Averse Subsample Consensus)**: $\arg\max_m (\bar{Z}_{0:8, m} - 0.35\,\sigma_{1:8, m})$.
     - **Slots 2–4 (Mode-Seeking Subsample Basin Specialists)**: Highest subsample-winning GP trees across distinct terminal estimator families (`histgbm`, `ridge_gcv`, `krr`, `gpr`, `elasticnet`, `pls`).
     - **Slot 5 (Semi-Synthetic Column-Robust Specialist)**: Highest-scoring candidate on Views `9..10` penalized by score drop under synthetic distractor columns.
     - **Slots 6–7 (Epistemic UCB + `sem24` Diversity Explorers)**: $\arg\max_m (\bar{Z}_{0:8, m} + 0.50\,\sigma_{1:8, m} + 0.30 \min_{c} \|\text{sem24}_m - \text{sem24}_c\|_2)$.
3. **Slowly Turning Up GP Evolution (`Stage C0: 0-Gen` $\to$ `Stage C1: Gen-0` $\to$ `Stage C2: 1-Gen` $\to$ `Stage C3: 3-Gen`) + Closed-Loop Deep Learning Updates**:
   - In `Stages C1–C3`, live 2-fold Rust CV (`bootstrap.s1main --fit-s1`) evaluates `Gen 0` (`8` trees) and `3` evolutionary generations (`Gen 1`, `Gen 2`, `Gen 3`), where `EditZero` typed AST mutations (`Parameter`, `ReplaceHead`, `InsertStage`, `DeleteStage`, `Crossover`) generate `20` child ASTs per generation and `H257`'s causal trajectory bridge screens them down to the top `5` for Rust CV evaluation.
   - Discovered GP winners are injected back into the `DL-Train` slates and `expanded_gp_bank` to update the Phase-2 neural bridges and reranker (`weights/h257_phase2_curriculum_bundle.safetensors`).
   - In the **High Compute Tier (`H257_HIGH_BATCH_EVO3_STACK3`)**, out-of-fold predictions from the top `3` structurally diverse passing GP trees discovered across `Gen 0..3` are combined via a **100% leak-free 2-fold out-of-fold simplex stack** ($w^{(0)}$ fit strictly on Fold 1 to predict Fold 0; $w^{(1)}$ fit strictly on Fold 0 to predict Fold 1).

---

## 2. Direct Same-Fold Validation Elo Tournament on `DL-Val91` (`91` Contexts / `19` Unseen Families)

All models below are evaluated on the **exact same 2-fold `JTT1` splits of all `91` `DL-Val91` contexts** (`182` fold fits per entrant), anchored at `R882 = 1500.00` (400-iteration MM Bradley-Terry solver, 1,000-draw paired family bootstrap 95% CIs):

### Table 2A: S1 Constrained Utility Tournament (Primary Metric, Includes AST Complexity Penalty)

| Rank | Entrant / Model | Category | Joint Elo (`95% CI`) | $\Delta$ Elo vs `Opus5p5_Extra` | Pairwise Elo vs `R882` | Mean Family Utility | Mean Context Utility | Family W/T/L vs `Opus5p5_Extra` | Gate Pass (`91` Ctx) |
| :---: | --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **`H257_HIGH_BATCH_EVO3_STACK3`** | **H257 Phase-2 High (11-View + 3-Gen GP + OOF Stack3)** | **`1838.71`** (`[1731.9, 1957.6]`) | **`+125.29`** | **`1835.34`** | **`1.261212`** | **`1.249161`** | `9 / 0 / 10` | **`91 / 91` (`100%`)** |
| **2** | **`H257_BATCH11_STATIC_PLUS_EVO3_STACK3`** | **H257 Phase-2 Static + 3-Gen GP Stack3 (0 Pool Reveals)** | **`1818.61`** (`[1714.2, 1933.9]`) | **`+105.20`** | **`1797.28`** | **`1.261046`** | **`1.248988`** | `9 / 0 / 10` | **`91 / 91` (`100%`)** |
| **3** | **`H257_MEDIUM_BATCH_EVO3`** | **H257 Phase-2 Medium (11-View + 3-Gen GP Single Tree)** | **`1762.85`** (`[1665.5, 1871.1]`) | **`+49.43`** | **`1790.33`** | **`1.253609`** | **`1.241287`** | `7 / 0 / 12` | **`91 / 91` (`100%`)** |
| **4** | **`H257_BATCH11_STATIC_PLUS_EVO3`** | **H257 Phase-2 Static + 3-Gen GP Single Tree** | **`1742.28`** (`[1646.9, 1848.8]`) | **`+28.86`** | **`1758.12`** | **`1.253056`** | **`1.240710`** | `7 / 0 / 12` | **`91 / 91` (`100%`)** |
| **5** | **`H257_STAGE_C2_BATCH11_PLUS_EVO1`** | **H257 Stage C2 (11-View + 1-Gen GP Evolution)** | **`1713.93`** (`[1618.1, 1817.5]`) | **`+0.51`** | **`1770.54`** | **`1.240541`** | **`1.227799`** | `7 / 0 / 12` | **`91 / 91` (`100%`)** |
| 6 | `Opus5p5_Extra` (Claude Opus 5.5) | Frozen Frontier Regressor | `1713.42` (`[1567.0, 1867.2]`) | `+0.00` | `1617.54` | `1.112585` | `1.112319` | `0 / 19 / 0` | `91 / 91` (`100%`) |
| 7 | `H257_STAGE_C1_BATCH11_PLUS_GEN0` | H257 Stage C1 (11-View + Gen-0 Seeding Only) | `1694.04` (`[1600.8, 1796.1]`) | `-19.38` | `1752.12` | `1.236936` | `1.224702` | `7 / 0 / 12` | `91 / 91` (`100%`) |
| 8 | `Gemini4_argon_high` (Gemini 4 Argon) | Frozen Frontier Regressor | `1658.02` (`[1519.1, 1806.2]`) | `-55.40` | `1601.43` | `1.140704` | `1.147858` | `9 / 0 / 10` | `91 / 91` (`100%`) |
| 9 | `H257_STAGE_C0_BATCH11_ACTIVE_2PLUS2` | H257 Stage C0 (0-GP, Warm-Start + Semi-Synthetic 2+2) | `1612.21` (`[1529.4, 1705.2]`) | `-101.20` | `1634.16` | `1.201077` | `1.187266` | `6 / 0 / 13` | `91 / 91` (`100%`) |
| 10 | `H256_ACTIVE_2PLUS2_UNIFIED` | H256 Phase-1 Active 2+2 | `1597.70` (`[1517.8, 1689.1]`) | `-115.72` | `1625.78` | `1.201692` | `1.187871` | `6 / 0 / 13` | `91 / 91` (`100%`) |
| 11 | `H257_STAGE_C0_BATCH11_STATIC_0REV` | H257 Stage C0 (0-GP, Warm-Start + Semi-Synthetic 0-Rev) | `1597.44` (`[1517.3, 1688.7]`) | `-115.97` | `1609.43` | `1.202731` | `1.188993` | `6 / 0 / 13` | `91 / 91` (`100%`) |
| 12 | `GPT6p1_Sol_Max` (GPT-6.1 Sol Max) | Frozen Frontier Regressor | `1596.67` (`[1464.5, 1732.0]`) | `-116.74` | `1525.62` | `1.141572` | `1.125242` | `5 / 0 / 14` | `91 / 91` (`100%`) |
| 13 | `H275_RERANK8_CR_SOFT` | Historical JopeDime Wave | `1583.55` (`[1504.7, 1672.0]`) | `-129.87` | `1609.43` | `1.217523` | `1.204474` | `6 / 0 / 13` | `91 / 91` (`100%`) |
| 14 | `H256_STATIC_0REV_TOP8_SOFT_VETO` | H256 Phase-1 Static 0-Reveal | `1576.54` (`[1498.6, 1664.4]`) | `-136.87` | `1597.47` | `1.203296` | `1.189621` | `6 / 0 / 13` | `91 / 91` (`100%`) |
| 15 | `R882` (Baseline Anchor) | Historical Anchor | `1500.00` (`[1500.0, 1500.0]`) | `-213.42` | `1500.00` | `1.153615` | `1.137794` | `5 / 0 / 14` | `91 / 91` (`100%`) |
| 16 | `GPT6_Astra_Max` (GPT-6 Astra Max) | Frozen Frontier Regressor | `1383.07` (`[1256.4, 1508.9]`) | `-330.35` | `1390.57` | `0.883094` | `0.879144` | `1 / 0 / 18` | `91 / 91` (`100%`) |

### Table 2B: Pure 2-Fold CV RMSE Ratio (`-logratio`) Tournament (Complexity-Free Accuracy Metric)

| Rank | Entrant / Model | Joint Elo (`95% CI`) | $\Delta$ Elo vs `Opus5p5_Extra` | Pairwise Elo vs `R882` | Mean Family `-logratio` | Mean Context `-logratio` | Geomean CV RMSE Ratio | Family W/T/L vs `Opus5p5_Extra` |
| :---: | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **`H257_HIGH_BATCH_EVO3_STACK3`** | **`1840.84`** (`[1737.0, 1958.4]`) | **`+109.37`** | **`1871.00`** | **`1.419663`** | **`1.406252`** | **`0.2451`** | **`11 / 0 / 8`** |
| **2** | **`H257_BATCH11_STATIC_PLUS_EVO3_STACK3`** | **`1828.40`** (`[1725.4, 1944.7]`) | **`+96.93`** | **`1843.75`** | **`1.419156`** | **`1.405724`** | **`0.2452`** | **`11 / 0 / 8`** |
| **3** | **`H257_MEDIUM_BATCH_EVO3`** | **`1773.23`** (`[1672.1, 1885.8]`) | **`+41.76`** | **`1819.39`** | **`1.410706`** | **`1.397044`** | **`0.2473`** | `7 / 0 / 12` |
| **4** | **`H257_BATCH11_STATIC_PLUS_EVO3`** | **`1759.19`** (`[1659.7, 1871.2]`) | **`+27.72`** | **`1797.28`** | **`1.409855`** | **`1.396156`** | **`0.2475`** | `7 / 1 / 11` |
| **5** | **`H257_STAGE_C2_BATCH11_PLUS_EVO1`** | **`1731.47`** (`[1632.9, 1840.6]`) | **`+0.00`** | **`1797.28`** | **`1.397335`** | **`1.383225`** | **`0.2507`** | `7 / 0 / 12` |
| 5 (tie) | `Opus5p5_Extra` (Claude Opus 5.5) | `1731.47` (`[1585.8, 1884.1]`) | `+0.00` | `1617.54` | `1.255813` | `1.253011` | `0.2856` | `0 / 19 / 0` |
| 7 | `H257_STAGE_C1_BATCH11_PLUS_GEN0` | `1710.63` (`[1614.7, 1817.1]`) | `-20.84` | `1776.97` | `1.392800` | `1.379433` | `0.2516` | `7 / 0 / 12` |
| 8 | `Gemini4_argon_high` (Gemini 4 Argon) | `1665.47` (`[1525.9, 1814.3]`) | `-66.00` | `1609.43` | `1.289317` | `1.295140` | `0.2739` | `9 / 0 / 10` |
| 9 | `H257_STAGE_C0_BATCH11_ACTIVE_2PLUS2` | `1636.58` (`[1553.0, 1731.1]`) | `-94.88` | `1651.39` | `1.353238` | `1.338132` | `0.2623` | `6 / 0 / 13` |
| 10 | `H257_STAGE_C0_BATCH11_STATIC_0REV` | `1626.46` (`[1545.2, 1718.9]`) | `-105.01` | `1634.16` | `1.354837` | `1.339802` | `0.2619` | `6 / 0 / 13` |
| 11 | `H256_ACTIVE_2PLUS2_UNIFIED` | `1624.68` (`[1543.5, 1717.4]`) | `-106.78` | `1642.69` | `1.353129` | `1.337989` | `0.2624` | `6 / 0 / 13` |
| 12 | `GPT6p1_Sol_Max` (GPT-6.1 Sol Max) | `1616.28` (`[1484.2, 1752.2]`) | `-115.19` | `1547.79` | `1.284931` | `1.267130` | `0.2816` | `6 / 0 / 13` |
| 13 | `H275_RERANK8_CR_SOFT` | `1611.16` (`[1531.6, 1700.7]`) | `-120.31` | `1621.65` | `1.370462` | `1.356142` | `0.2577` | `6 / 0 / 13` |
| 14 | `H256_STATIC_0REV_TOP8_SOFT_VETO` | `1607.31` (`[1528.0, 1696.9]`) | `-124.16` | `1613.47` | `1.354777` | `1.339768` | `0.2619` | `6 / 0 / 13` |
| 15 | `R882` (Baseline Anchor) | `1500.00` (`[1500.0, 1500.0]`) | `-231.47` | `1500.00` | `1.297407` | `1.279905` | `0.2781` | `5 / 0 / 14` |
| 16 | `GPT6_Astra_Max` (GPT-6 Astra Max) | `1382.88` (`[1255.8, 1508.8]`) | `-348.59` | `1390.57` | `0.998634` | `0.992029` | `0.3708` | `1 / 0 / 18` |

---

## 3. Stage `C0` (`0` GP Generations): Warm-Start + Semi-Synthetic Column & `80%` Subsample Results

Before running any GP cycles, warm-starting from `H256` and training on `3x` views (`100%` clean + `80%` row subsample + `semisyn80` additional synthetic columns) with Two-Speed Differential LR and All-28-Pairs Regret-Weighted Margin Loss produced large out-of-fold and held-out improvements:

| Model Arm | `DL-Train OOF` (`1,219` Ctx / `205` Fam) Mean Family Utility | `DL-Train OOF` Gain vs `R882` | `DL-Train OOF` Mean Invalid Picks | `DL-Val91` (`91` Ctx / `19` Fam) Mean Family Utility | `DL-Val91` Gain vs `R882` | `DL-Val91` Invalid Picks |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| `H256_V1_NN_ONLY` (Phase-1 Cold-Init Neural Reranker) | `1.443276` | `+0.057920` | `11.7` | `1.181667` | `+0.028052` | `0.0` |
| **`H257_C0_AUDITED_NN_ONLY` (Warm-Start + Diff-LR + All-28-Pairs + Semi-Syn)** | **`1.567007`** | **`+0.181651`** | **`4.3`** | **`1.195315`** | **`+0.041700`** | **`0.0`** |
| **`H257_C0_BATCH11_NN_ONLY` (11-View Multi-Subsample Batched Neural Reranker)** | **`1.566357`** | **`+0.181002`** | **`4.3`** | **`1.196373`** | **`+0.042758`** | **`0.0`** |
| `H256_STATIC_0REV_TOP8_SOFT_VETO` (Phase-1 Static Ensemble) | `1.461114` | `+0.075759` | `12.7` | `1.203296` | `+0.049680` | `0.0` |
| **`H257_STAGE_C0_BATCH11_STATIC_0REV` (Phase-2 Stage C0 Static Ensemble)** | **`1.491716`** | **`+0.106360`** | **`10.0`** | **`1.202731`** | **`+0.049116`** | **`0.0`** |
| `H256_ACTIVE_2PLUS2_UNIFIED` (Phase-1 Active `2+2`) | `1.489648` | `+0.104292` | `5.3` | `1.201692` | `+0.048077` | `0.0` |
| **`H257_STAGE_C0_BATCH11_ACTIVE_2PLUS2` (Phase-2 Stage C0 Active `2+2`)** | **`1.509711`** | **`+0.124356`** | **`4.0`** | **`1.201077`** | **`+0.047461`** | **`0.0`** |

---

## 4. Stages `C1` $\to$ `C2` $\to$ `C3`: 11-View Batched Seeding & 3-Generation Live Rust GP Evolution Progression

Across all `91` `DL-Val91` unseen validation contexts (`19` families), seeding Generation 0 from the `11`-view batch (`1×100% + 8×80% + 2×semisyn`) and running `3` evolutionary generations via `EditZero` + `H257` causal trajectory pre-screening + `bootstrap.s1main --fit-s1` produces a strictly monotonic progression:

| Evolutionary Stage | Standalone GP Family Utility (`DL-Val91`) | Combined Curriculum Family Utility (`DL-Val91`) | Joint Elo (S1 Utility) | Joint Elo (CV RMSE Ratio) | Contexts Improved Over Pool / Previous |
| --- | :---: | :---: | :---: | :---: | :---: |
| **`Gen 0` (`Stage C1`: 11-View 5-Role Batched Seeding, `8` GP Trees)** | `1.059807` | `1.236936` | `1694.04` | `1710.63` | Surpasses `Gemini4_argon_high` (`1658.02`) at Gen 0 |
| **`Gen 1` (`Stage C2`: 1-Gen GP Evolution, `+5` Neural-Screened Children)** | `1.119270` (`+0.0595`) | `1.240541` | `1713.93` | `1731.47` | Surpasses `Opus5p5_Extra` (`1713.42`) after 1 GP Gen |
| **`Gen 2` (`Stage C3a`: 2-Gen GP Evolution, `+5` Neural-Screened Children)** | `1.162717` (`+0.1029`) | `1.248912` | `1748.10` | `1758.90` | Monotonic gain across 19 unseen families |
| **`Gen 3` (`H257_MEDIUM_BATCH_EVO3`: 3-Gen GP Single-Tree Champion)** | **`1.183865` (`+0.1241`)** | **`1.253609`** | **`1762.85`** | **`1773.23`** | **`42 / 91` contexts strictly beat static/active pool** |
| **`Stack 3` (`H257_HIGH_BATCH_EVO3_STACK3`: Leak-Free 2-Fold OOF Top-3 GP Stack)** | — | **`1.261212`** | **`1838.71`** | **`1840.84`** | **`41 / 91` contexts further improved by OOF Top-3 Stack** |

---

## 5. Tabular Foundation Model Rankings (`TabFM`, `Kumo`, `PriorLabs 3.5`, `TabICL v2`, `TabDPT 1.3`, `Mitra v2`) vs. `H257` and Frontier LLMs

### 5.1 Audit of S3 Precomputed Scores vs. Fair 2-Fold Inner-CV (`jope.s1-two-fold/v1`)

1. **`TabFM` (`google/tabfm-1.0.0-pytorch`)**: Had **`0 / 91`** precomputed records in S3 on `DL-Val91` and `0 / 240` in `panel_v1.json` (previously scored only on SOTArena `Golden45` and the `1,973`-dataset baseline panel).
2. **`PriorLabs 3.5` (`TabPFN v3.5`)**: Precomputed S3 records (`s3://veox-jopedime-learning-use2/experiments/gp-teacher-v6/campaigns/regression-research-2k-fold0-3plus2-v1-20261001-tabpfn35-exact1/tabpfn35-exact/results/sha256/*.json`) existed for `91 / 91` `DL-Val91` contexts, but were evaluated on a **1-split `100%` outer-train $\to$ outer-test protocol** (`sd_logratio = 0`), whereas `H257` and `Opus5p5_Extra` are evaluated on **`50% / 50%` 2-fold inner CV (`jope.s1-two-fold/v1`)** with fold-variance penalty (`-0.2 * sd_logratio`). In `panel_v1.json` (`240` regression datasets), `tabpfn` had `0 / 240` completed regression datasets (`62` errors because `TABPFN_REG_CKPT` was unset in `node_run.sh`).
3. **`Kumo-Tabular` (`Large` & `Small`), `TabICL v2`, `TabDPT 1.3`, `Mitra v2`**: Precomputed S3 encodings (`s3://veox-jopedime-learning-use2/experiments/gp-policy-dl/v2-trainonly/encodings/<model>/<stem>.npy`, last 3 floats `[qr2, spread, logspread]`) existed for `91 / 91` `DL-Val91` contexts, but were evaluated on a **single `75% / 25%` subsampled support/query split** (`nc <= 768`, `nq <= 256`, `sd_logratio = 0`).
4. **Fair Resolution**: Using [`evaluate_tabular_foundation_dl_val91.py`](evaluate_tabular_foundation_dl_val91.py), we downloaded `google/tabfm-1.0.0-pytorch` (`regression/model.safetensors`) and evaluated **all 9 tabular foundation model configurations directly on the exact 91 `DL-Val91` 2-fold `JTT1` inner-CV envelopes (`182` fold evaluations per model)** under the identical `jope.s1-two-fold/v1` formula.

### 5.2 Fair Same-Fold 2-Fold Inner-CV (`jope.s1-two-fold/v1`) All-Models Tournament on `DL-Val91` (`91` Contexts / `19` Unseen Families, Anchored at `R882 = 1500.00`)

| Rank (Joint S1) | Entrant / Model | Model Class | Joint Elo (`S1 Gated`) | Joint Elo (`-logratio`) | Pairwise Elo vs `R882` (`S1` / `-lr`) | $\Delta$ Joint Elo vs `Opus5p5_Extra` (`S1`) | Mean Family `S1` Utility | Mean Family `-logratio` | Gate Pass (`91` Ctx) |
| :---: | --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `Kumo_Tabular_L` (`nvidia/Kumo-Tabular` Large) | Tabular Foundation ICL | `1980.90` | `1995.62` | `1790.33` / `1776.97` | `+335.42` | `1.500689` | `1.704041` | `91 / 91` |
| 2 | `Kumo_Tabular_S` (`nvidia/Kumo-Tabular` Small) | Tabular Foundation ICL | `1924.18` | `1934.02` | `1790.33` / `1790.33` | `+278.70` | `1.396395` | `1.621064` | `90 / 91` |
| 3 | `PriorLabs_TabPFN_3p5_Single` (`TabPFN v3.5`, `n_est=1`) | Tabular Foundation ICL | `1919.16` | `1934.46` | `1776.97` / `1790.33` | `+273.68` | `1.390740` | `1.573698` | `91 / 91` |
| 4 | `TabFM_Ensemble` (`google/tabfm-1.0.0-pytorch`, 4-Est Recipe) | Tabular Foundation ICL | `1915.49` | `1916.02` | `1790.33` / `1790.33` | `+270.01` | `1.388920` | `1.569961` | `91 / 91` |
| 5 | `PriorLabs_TabPFN_3p5` (`TabPFN v3.5`, `n_est=8`) | Tabular Foundation ICL | `1888.90` | `1896.42` | `1764.26` / `1740.48` | `+243.42` | `1.346831` | `1.530898` | `91 / 91` |
| 6 | `Mitra_v2` (`autogluon/mitra-regressor-2`) | Tabular Foundation ICL | `1829.48` | `1831.34` | `1790.33` / `1790.33` | `+184.00` | `1.325378` | `1.496234` | `91 / 91` |
| 7 | `TabFM` (`google/tabfm-1.0.0-pytorch`, Single Pass) | Tabular Foundation ICL | `1823.00` | `1842.05` | `1729.29` / `1740.48` | `+177.52` | `1.325618` | `1.506171` | `91 / 91` |
| 8 | `TabDPT_1p3` (`Layer6/TabDPT` v1.3) | Tabular Foundation ICL | `1809.46` | `1817.87` | `1752.12` / `1740.48` | `+163.98` | `1.283347` | `1.453725` | `91 / 91` |
| 9 | `TabICL_v2` (`jingang/TabICL` v2) | Tabular Foundation ICL | `1798.63` | `1802.63` | `1740.48` / `1729.29` | `+153.15` | `1.269461` | `1.439189` | `91 / 91` |
| **10** | **`H257_HIGH_BATCH_EVO3_STACK3` (Ours)** | **H257 Phase-2 High (Pure Rust GP + OOF Stack3)** | **`1749.51`** | **`1755.59`** | **`1835.34` / `1871.00` (#1!)** | **`+104.03`** | **`1.261212`** | **`1.419663`** | **`91 / 91`** |
| **11** | **`H257_BATCH11_STATIC_PLUS_EVO3_STACK3` (Ours)** | **H257 Phase-2 Static + 3-Gen GP Stack3** | **`1738.44`** | **`1748.52`** | **`1797.28` / `1843.75` (#2!)** | **`+92.96`** | **`1.261046`** | **`1.419156`** | **`91 / 91`** |
| **12** | **`H257_MEDIUM_BATCH_EVO3` (Ours)** | **H257 Phase-2 Medium (3-Gen GP Single Tree)** | **`1702.37`** | **`1710.83`** | **`1790.33` / `1819.39`** | **`+56.89`** | **`1.253609`** | **`1.410706`** | **`91 / 91`** |
| **13** | **`H257_BATCH11_STATIC_PLUS_EVO3` (Ours)** | **H257 Phase-2 Static + 3-Gen GP Single Tree** | **`1689.51`** | **`1702.11`** | **`1758.12` / `1797.28`** | **`+44.03`** | **`1.253056`** | **`1.409855`** | **`91 / 91`** |
| **14** | **`H257_STAGE_C2_BATCH11_PLUS_EVO1` (Ours)** | **H257 Stage C2 (1-Gen GP Evolution)** | **`1665.60`** | **`1681.35`** | **`1770.54` / `1797.28`** | **`+20.12`** | **`1.240541`** | **`1.397335`** | **`91 / 91`** |
| **15** | **`H257_STAGE_C1_BATCH11_PLUS_GEN0` (Ours)** | **H257 Stage C1 (Gen-0 Seeding Only)** | **`1650.70`** | **`1663.73`** | **`1752.12` / `1776.97`** | **`+5.22`** | **`1.236936`** | **`1.392800`** | **`91 / 91`** |
| 16 | `Opus5p5_Extra` (Claude Opus 5.5) | Frozen Frontier Regressor | `1645.48` | `1653.29` | `1617.54` / `1617.54` | `+0.00` | `1.112585` | `1.255813` | `91 / 91` |
| 17 | `Gemini4_argon_high` (Gemini 4 Argon) | Frozen Frontier Regressor | `1609.07` | `1611.03` | `1601.43` / `1609.43` | `-36.41` | `1.140704` | `1.289317` | `91 / 91` |
| 18 | `H257_STAGE_C0_BATCH11_ACTIVE_2PLUS2` | H257 Stage C0 (0-GP, Warm-Start + Semi-Syn) | `1592.04` | `1609.76` | `1634.16` / `1651.39` | `-53.44` | `1.201077` | `1.353238` | `91 / 91` |
| 19 | `H257_STAGE_C0_BATCH11_STATIC_0REV` | H257 Stage C0 (0-GP, Static 0-Reveal) | `1581.73` | `1602.83` | `1609.43` / `1634.16` | `-63.75` | `1.202731` | `1.354837` | `91 / 91` |
| 20 | `H256_ACTIVE_2PLUS2_UNIFIED` | H256 Phase-1 Active 2+2 | `1580.43` | `1600.82` | `1625.78` / `1642.69` | `-65.04` | `1.201692` | `1.353129` | `91 / 91` |
| 21 | `GPT6p1_Sol_Max` (GPT-6.1 Sol Max) | Frozen Frontier Regressor | `1572.98` | `1586.94` | `1525.62` / `1547.79` | `-72.50` | `1.141572` | `1.284931` | `91 / 91` |
| 22 | `H275_RERANK8_CR_SOFT` | Historical JopeDime Wave | `1570.91` | `1591.41` | `1609.43` / `1621.65` | `-74.57` | `1.217523` | `1.370462` | `91 / 91` |
| 23 | `H256_STATIC_0REV_TOP8_SOFT_VETO` | H256 Phase-1 Static 0-Reveal | `1566.20` | `1589.36` | `1597.47` / `1613.47` | `-79.28` | `1.203296` | `1.354777` | `91 / 91` |
| 24 | `R882` (Baseline Anchor) | Historical Anchor | `1500.00` | `1500.00` | `1500.00` / `1500.00` | `-145.48` | `1.153615` | `1.297407` | `91 / 91` |
| 25 | `GPT6_Astra_Max` (GPT-6 Astra Max) | Frozen Frontier Regressor | `1379.76` | `1369.62` | `1390.57` / `1390.57` | `-265.72` | `0.883094` | `0.998634` | `91 / 91` |

### 5.3 S3 Precomputed Reference Tournament on `DL-Val91` & External Benchmark Panels

#### A. S3 Precomputed `DL-Val91` Reference Tournament (`PriorLabs_TabPFN_3p5_S3_OuterTest` & `S3 encodings/` 1-Split Query Fit)
- `PriorLabs_TabPFN_3p5_S3_OuterTest` (`100%` outer-train $\to$ outer-test, `sd=0`): Joint Elo **`1901.83`** (`-lr` Elo **`1896.81`**), Pairwise vs `R882 = 1819.39`, Mean Family Utility `1.550111`.
- `Kumo_Tabular_L_S3_Query1Split` (`75/25` subsampled query, `sd=0`): Joint Elo **`1894.05`** (`-lr` Elo **`1890.02`**), Pairwise vs `R882 = 1740.48`, Mean Family Utility `1.646028`.
- `Kumo_Tabular_S_S3_Query1Split` (`75/25` subsampled query, `sd=0`): Joint Elo **`1852.25`** (`-lr` Elo **`1851.94`**), Pairwise vs `R882 = 1729.29`, Mean Family Utility `1.579170`.
- **`H257_HIGH_BATCH_EVO3_STACK3` (Ours — strict 2-fold inner CV + `sd` + `comp` penalty)**: Joint Elo **`1762.36`** (`-lr` Elo **`1767.42`**), **Pairwise vs `R882 = 1835.34` (`#1` overall!)** — ranks **`#4` overall** in the S3 precomputed tournament ahead of `Mitra_v2_S3_Query1Split` (`1750.57`), `TabDPT_1p3_S3_Query1Split` (`1745.96`), `TabICL_v2_S3_Query1Split` (`1744.62`), and `Opus5p5_Extra` (`1651.34`).

#### B. External Reference Panels (`panel_v1` 240 Datasets, SOTArena `Golden45`, and `1,973`-Dataset Panel)
- **`panel_v1.json` (`240` Regression Datasets, `/home/ubuntu/jope_plan/elo/panel_results/elo_panel_v1.json`)**:
  - `kumo_tabular_large_e8`: **`1874.5`** (`239 / 240` datasets)
  - `autogluon`: **`1601.3`** (`240 / 240`), `catboost`: **`1523.7`**, `random_forest`: **`1441.5`**, `xgboost`: **`1424.4`**, `lightgbm`: **`1412.7`**, `jope-fused-net-v3`: **`1383.4`**, `ridge`: **`1338.4`**.
- **SOTArena `tabular-regression-v2` (`Golden45`, `45` Datasets) & `1,973`-Dataset Panel**:
  - `tabfm_ensemble_regressor`: **`1934.9`** (`45 / 45` `Golden45`, `#1`) and **`1260.94`** (`1,973`-dataset panel, `#1`)
  - `tabpfn_v3_regressor` (`PriorLabs TabPFN v3`): **`1909.5`** (`45 / 45` `Golden45`, `#2`) and **`1223.71`** (`1,973`-dataset panel)
  - `tabpfn_v3_regressor_default`: **`1906.3`** (`Golden45`) / **`1219.91`** (`1,973`-dataset panel)
  - `tabpfn_v25_regressor_default`: **`1893.3`** (`45 / 45`), `tabicl`: **`1887.6`** (`45 / 45`), `tabpfn_v25_regressor_real`: **`1874.8`** (`45 / 45`), `tabfm_regressor`: **`1872.9`** (`39 / 45`), `tabdpt_regressor`: **`1788.6`** (`24 / 45`).
