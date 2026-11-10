# H255 Status: Causal 2+2 Neural Trajectory Critic & Live Rust GP Closed-Loop Flywheel

- **Wave**: `H255` (`h255_causal_2plus2_live_gp`)
- **Pre-Execution Code SHA (`origin/v1/next` & `origin/research/h255-causal-2plus2-live-gp`)**: `ff8ec7ed771dfba57e69976da136a387687f864f`
- **Host / Resource Profile**: `xbabe2` CPU (`taskset -c 48-51 nice -n 19`, `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=4`)
- **Wall Time**: Part A `491.52s`, Part B `296.16s`
- **Sealed Meta-Test Read**: `false` (`0` bytes of sealed meta-test or outer test folds accessed)
- **DOPE Checkpoint SHA-256**: `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` (`unchanged = true`)
- **R882 Native91 Reproduction**: `1.1377941898164525` (`diff = 0.0`)
- **Frozen H252 Hashes Verified**:
  - `common_h252.py`: `abe974ed10a0cd0d4b08440015a72a7b0adcebb22118c234732b812e9f0b0f92`
  - `stage1_hurdle_sos_critic.py`: `fc0dd0e2fa543f0540d677609989c1785ef33b07f6cf7e54f4f8bedbd0e5e797`
- **Decision**: **`ADVANCE`** (`CAUSAL_2PLUS2_NEURAL` and `LIVE_GP_CAUSAL_2PLUS2_FLY`)

---

## 1. Root-Cause Audit: Why `FUSED_CLOSED_LOOP` Regressed in H252

In [`run_fused_closed_loop.py`](../h252_hurdle_trajectory_flywheel/run_fused_closed_loop.py) (lines 693–694), `FUSED_CLOSED_LOOP` hardcoded `revealed_indices=[0, 1, 2, 3]` at evaluation time, whereas training slates sorted candidates by true oracle fitness descending so slots `0..3` were always the top-4 oracle winners:

| Audit Metric | Value |
| --- | --- |
| `training_slate_slot03_oracle_ranks` | `[0, 1, 2, 3]` |
| `training_self_distance_d2_joint_on_slots_0_to_3` | `0.0` (exact self-leakage on slots `0..3`) |
| `dl_train_raw_slot03_median_best_rank` | `178.0` |
| `dl_val_raw_slot03_median_best_rank` | `181.0` |
| `dl_val_raw_slot03_mean_best_rank` | `223.07` |
| `dl_val_raw_slot03_contains_winner_frac` | `1.10%` (`1 / 91`) |
| `dl_val_r882_top2_in_slot03_frac` | `17.03%` |

Because `TrajectorySelfAttentionAdapter` in H252 was trained with unmasked slots `0..3` equal to the oracle winners (`d2_joint == 0.0` to themselves) and evaluated on arbitrary raw pool slots `0..3` (median best rank `181.0`) while charging `4` reveal slots up front (`16` total vs `4` for single-shot arms), it suffered both training self-leakage and a `4x` reveal-budget mismatch.

`H255` fixes all three defects:
1. **Strict $B = 4 = 2 + 2$ Budget Matching**: Every arm (`R882`, `GATED_025`, `ACTIVE_2PLUS2_HURDLE`, `CAUSAL_2PLUS2_R882_ZERO_SHOT`, `CAUSAL_2PLUS2_NEURAL`, `NEGATIVE_CONTROL_PERMUTED_REVEAL`) reveals **exactly 4 candidates per context** (`reveal_count_histograms[arm] = {"4": 3930}`).
2. **Causal Pre-Outcome Batch-1 Selection**: Batch-1 (`k1 = 2`) is selected * strictly prior to any reveal* (`t_steps = 0`) using zero-shot `R882` + `HurdleSOSCritic` scores.
3. **Strict Self-Exclusion on Batch-2 (`N - 2` Remaining Candidates)**: After observing the true inner-CV outcomes of the 2 Batch-1 candidates, `Causal2Plus2Adapter` cross-attends from the $N - 2$ unrevealed candidates (`cand_idx != idx1` and `cand_idx != idx2`, so $d_{\text{joint}} > 0$ strictly) to the 2 revealed `(embedding, centered_outcome, gate_pass)` tokens to select Batch-2 (`k2 = 2`).

---

## 2. Part A: Offline DL-Train (1,219 Contexts, 205 Families) & DL-Val (91 Contexts, 19 Families)

Evaluated across 3 independent random seeds (`255101`, `255102`, `255103`) with 10,000 paired family-clustered bootstrap resamples and Holm-Bonferroni correction across all 5 non-baseline arms.

### DL-Train (1,219 Contexts, 205 Families) vs `GATED_025` (`1.385465`) & `R882` (`1.385355`)

| Arm |Reveals/Ctx| Median Context Utility | Median Family Utility | Median Family $\Delta$ vs `GATED_025` | 95% Family Bootstrap CI | `LCB90` | `p_one_sided` | `holm_p` | Family W/T/L (Seed Med) | Harm Frac $>0.01$ |Median Elo| Decision |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `R882` | 4 | `1.423073` | `1.385355` | `-0.000109` | `[-0.000920, +0.000046]` | `-0.000712` | `0.8504` | `0.8504` | `2/199/4` | `0.49%` | `1499.38` | BASELINE |
| `GATED_025` | 4 | `1.423165` | `1.385465` | `0.000000` | — | — | — | — | — | — | `1500.00` | BASELINE |
| `ACTIVE_2PLUS2_HURDLE` | 4 | `1.495432` | `1.467077` | **`+0.081342`** | `[+0.041346, +0.140055]` | `+0.045454` | `1.0e-4` | `5.0e-4` | `77/118/10` | `1.95%` | `1525.08` | **ADVANCE** |
| `CAUSAL_2PLUS2_R882_ZERO_SHOT` | 4 | `1.496568` | `1.467975` | **`+0.081363`** | `[+0.041884, +0.140195]` | `+0.045770` | `1.0e-4` | `5.0e-4` | `82/110/13` | `2.44%` | `1525.56` | **ADVANCE** |
| **`CAUSAL_2PLUS2_NEURAL`** | **4** | **`1.496949`** | **`1.468249`** | **`+0.082033`** | **`[+0.042161, +0.140386]`** | **`+0.046069`** | **`1.0e-4`** | **`5.0e-4`** | **`84/109/12`** | **`2.44%`** | **`1525.65`** | **ADVANCE** |
| `NEGATIVE_CONTROL_PERMUTED_REVEAL` | 4 | `1.425507` | `1.387938` | `+0.002473` | `[-0.000542, +0.007917]` | `-0.000422` | `0.2865` | `0.5729` | `4/196/5` | `0.98%` | `1499.63` | REJECT |

- On DL-Train, **`CAUSAL_2PLUS2_NEURAL` achieves the highest utility (`1.468249` family, `1.496949` context) and highest Elo (`1525.65`) of all arms**, improving over `GATED_025` by **`+0.082033`** median family utility (`+0.082894` vs `R882`, `LCB90 = +0.046751`, `3/3` seeds positive) and outperforming both `ACTIVE_2PLUS2_HURDLE` (`1.467077`) and `CAUSAL_2PLUS2_R882_ZERO_SHOT` (`1.467975`).
- PCGrad multi-task training diagnostics across the 5 family-grouped folds show `pcgrad_mean_cosine` in `[0.9227, 0.9850]` with `pcgrad_neg_cosine_frac = 0.0`, confirming zero destructive gradient conflict between the ranking hinge loss and continuous utility Huber loss when self-leakage is removed.

### DL-Val (91 Contexts, 19 Families) vs `R882` (`1.153615` Family / `1.137794` Context)

| Arm |Reveals/Ctx| Median Context Utility | Median Family Utility | Median Family $\Delta$ vs `R882` | 95% Family Bootstrap CI | `LCB90` | `p_one_sided` | `holm_p` | Family W/T/L (Seed Med) | Harm Frac $>0.01$ |Median Elo| Decision |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `R882` | 4 | `1.137794` | `1.153615` | `0.000000` | — | — | — | — | — | — | `1500.00` | BASELINE |
| `GATED_025` | 4 | `1.137192` | `1.153038` | `-0.000577` | `[-0.001273, 0.000000]` | `-0.001154` | `1.0000` | `1.0000` | `0/18/1` | `0.00%` | `1496.33` | REJECT |
| `ACTIVE_2PLUS2_HURDLE` | 4 | `1.153636` | `1.168790` | **`+0.015174`** | `[+0.005419, +0.026475]` | `+0.006747` | `1.0e-4` | `5.0e-4` | `9/8/2` | `0.00%` | `1529.31` | **ADVANCE** |
| `CAUSAL_2PLUS2_R882_ZERO_SHOT` | 4 | `1.153636` | `1.168790` | **`+0.015174`** | `[+0.005439, +0.026480]` | `+0.006773` | `1.0e-4` | `5.0e-4` | `9/9/1` | `0.00%` | `1529.31` | **ADVANCE** |
| **`CAUSAL_2PLUS2_NEURAL`** | **4** | **`1.153636`** | **`1.168790`** | **`+0.015174`** | **`[+0.005419, +0.026475]`** | **`+0.006747`** | **`1.0e-4`** | **`5.0e-4`** | **`9/8/2`** | **`0.00%`** | **`1529.31`** | **ADVANCE** |
| `NEGATIVE_CONTROL_PERMUTED_REVEAL` | 4 | `1.137192` | `1.153038` | `-0.000577` | `[-0.001380, 0.000000]` | `-0.001154` | `1.0000` | `1.0000` | `0/18/1` | `0.00%` | `1496.33` | REJECT |

- On DL-Val, all three causal 2+2 arms (`CAUSAL_2PLUS2_NEURAL`, `CAUSAL_2PLUS2_R882_ZERO_SHOT`, `ACTIVE_2PLUS2_HURDLE`) achieve **`1.168790` median family utility** (`+0.015174` vs `R882`, `LCB90 = +0.006747`, `p = 1.0e-4`, `holm_p = 5.0e-4`, **`0.0%` harm > 0.01**, `3/3` seeds positive), while `NEGATIVE_CONTROL_PERMUTED_REVEAL` fails to advance (`1.153038`, `holm_p = 1.0`).

---

## 3. Part B: Live Rust GP 2+2 Closed-Loop Flywheel (`32` Stratified DL-Train Contexts, `2` Rounds $\times$ `16` Proposals, $B=4$ Rust Evaluations/Context/Round)

Evaluated using the live Rust `evaluator_cli` binary (`SHA-256: e3be938644722d14a1c1a0e82168518838c4d174cfe031076a339745c48870a8`) on 32 stratified DL-Train contexts across 32 distinct families (`2` rounds $\times$ `16` grammar-valid mutations/crossovers per context per round, strict budget of $B=4$ Rust evaluations per context per round = `256` total Rust evaluations per arm).

| Live GP Arm | Rust Evals / Arm | Passing | Total Promotions | Batch-2 Causal Promotions | Initial Mean Utility | Round 1 Mean Utility | Final (Round 2) Mean Utility | Mean Family $\Delta$ vs Initial | 95% Family Bootstrap CI vs Initial | `LCB90` vs Initial | `holm_p` vs Initial | Family W/T/L vs Initial | Elo vs Initial |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `INITIAL_INCUMBENT` | 0 | — | 0 | 0 | `1.809228` | `1.809228` | `1.809228` | `0.000000` | — | — | — | — | `1500.00` |
| `LIVE_GP_LEGACY_STATIC_4` | 256 | 251 | 9 | 0 | `1.809228` | `1.810553` | `1.811185` | `+0.001956` | `[+0.000210, +0.004678]` | `+0.000286` | `4.0e-4` | `8/24/0` | `1577.50` |
| `LIVE_GP_CHAMPION_COND_STATIC_4` | 256 | 252 | 8 | 0 | `1.809228` | `1.810858` | `1.811035` | `+0.001807` | `[+0.000146, +0.004435]` | `+0.000276` | `4.1e-3` | `5/27/0` | `1561.02` |
| **`LIVE_GP_CAUSAL_2PLUS2_FLY`** | **256** | **251** | **12** | **8** | **`1.809228`** | **`1.810887`** | **`1.812076`** | **`+0.002848`** | **`[+0.000721, +0.005652]`** | **`+0.000960`** | **`3.0e-4`** | **`9/23/0`** | **`1605.17`** |

### Direct Pairwise Contrast: `LIVE_GP_CAUSAL_2PLUS2_FLY` vs `LIVE_GP_LEGACY_STATIC_4`
- **Utility Gain**: **`+0.000892`** (`1.812076` vs `1.811185`, `45.6%` more total utility gain over initial incumbents under the exact same $B=4$ Rust evaluation budget).
- **Family W/T/L**: **`3 / 29 / 0`** (`0` harmed families, `harm_frac_gt_001 = 0.0%`).
- **Bootstrap Confidence**: `LCB90 = +0.000116`, `p_one_sided = 0.0424`, Elo `+27.67` (`1527.67` vs `1500.00`).
- **Mechanism Proof**: **8 of the 12 incumbent promotions (`66.7%`) in `LIVE_GP_CAUSAL_2PLUS2_FLY` were discovered in Batch-2 (`from_batch2_reveal = true`)**, where the causal similarity kernel updated candidate scores after observing the Batch-1 Rust inner-CV outcomes (`2` evaluated programs).

---

## 4. Artifact Locations

- **Part A Script**: [`part_a_causal_2plus2_neural.py`](part_a_causal_2plus2_neural.py)
- **Part B Script**: [`part_b_live_rust_gp_2plus2.py`](part_b_live_rust_gp_2plus2.py)
- **Committed Results**:
  - [`results/part_a_metrics.json`](results/part_a_metrics.json)
  - [`results/part_b_metrics.json`](results/part_b_metrics.json)
- **Host Data Receipts (`xbabe2`)**:
  - `/home/ubuntu/jope-data/h255/part_a/part_a_metrics.json`
  - `/home/ubuntu/jope-data/h255/part_b/part_b_metrics.json`
