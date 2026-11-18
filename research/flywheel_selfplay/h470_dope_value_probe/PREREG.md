# H470 Pre-Registration: DOPE Value Probe (`WP0`–`WP4`) & Zero-Retraining Accuracy Harvester (`WP3` & `B03`)

- **Wave ID**: `H470`
- **Branch**: `research/h470-dope-value-probe` (base `7655570db6`, `origin/main`)
- **Scratch & Bulk Cache Root**: `/home/ubuntu/jope-data/hwave-h470/` (no `.pt`, `.npz`, or raw logs committed to Git)
- **Frozen DOPE Pins**:
  - Standalone weights SHA-256: `d0b6960328d1955386632ff78b512e3add53a9b0256db7bbf067e2e0be5f0c6c`
  - Fused serving composite SHA-256 (`dope_freeze_guard`): `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`
  - CPU-exact `FusedJopeNetwork` source SHA-256: `b9f1468a61aa4dc888c8f2931f9c759f7d4f057793a77ccb05eed6300fb4f9ab`
  - Incumbent checkpoint: `round-3175.pt` (`1.420532014` train utility / `19` gate fails; `1.138957240` val91 utility / `0` gate fails)
- **Product Isolation**: `crates/jope-encoders`, `crates/jope-service`, and `python/src/jope/north_star/dope_freeze_guard.py` remain 100% untouched.

---

## 1. Scientific Questions & Hypotheses

Adapter-only optimizer waves `H451`–`H468` plateaued near `+4.7` to `+5.7` OOF Elo (`NO_LIFT`) with ~58% non-floor win rate on kept decisions. `H470` executes a pre-registered, CPU-only factorized diagnostic and zero-retraining accuracy harvest across three parts:

1. **Part A — Instrument & Bottleneck Diagnostics (`DL-Train` 1,219 contexts / 205 families only)**:
   - **A1 (Stage-1A Conditioning Gap)**: Quantify the cosine similarity, held-out-family Ridge $R^2$, and `FusedJopeNetwork` ranking/utility gap between fast GPU-sketch `Stage1ADistilledBridge` (`landmark_bridge` and `nolm_bridge`), the H58/H469 `FusedLandmarkWorkspace` `high` bundle, and true 23-landmark 5-fold FIT-only CV conditioning (`landmarks-native-v1` + `build_v3_conditioning_bundle`).
   - **A2 (`L7` Audit — `lm48_z` vs `lm_smart_80`)**: Verify in source and empirically whether `lm48` is an SVD coordinate vector (`256 -> 48`) distorted when `lm_smart_80` reshapes it into `16 x 3` `(mean, |std|, max)` triples.
   - **A3 (0-Reveal vs Wave-0 Pseudo-Reveal)**: Compare true masked 0-reveal (`revealed_indices=None`) against Wave-0 fake `fitness=100.0, gate=0.0` pseudo-reveals (`k=1` and `k=4`) on `FusedJopeNetwork` (`round-3175`).
   - **A4 (`H410` Sample-Size & Split-Half Stability + `DVP-S4` MDE)**: Measure split-half row-disjoint representation stability and fitness/regret sensitivity across `n_fit` strata, plus the 80%-power Minimum Detectable Effect (`MDE`, two-sided $\alpha=0.05$) on the 205-family `DL-Train` panel before scoring upgrades.

2. **Part B — Full CPU Value Probe Matrix (`WP0`–`WP2`, `WP4`, Contrasts `C0`–`C7`)**:
   - **H-Ceiling (`C0` / Gate `G0` & `O-S2`)**: Measured program outcomes (`profile_oracle` on split-pool `A -> B` and `R_oracle64` `K=64` pivoted-QR common-program basket evaluated on basket-excluded slates) beat `R_none` and `R_dope` (`ci_low > 0`), proving ranking headroom is learnable from table representations on this panel.
   - **H-Pretrain (`C1`, `C2`, `C3`, `C5`)**: Separate raw descriptor signal (`C3: R_desc - R_shape`), router nonlinear accessibility (`C1: R_dope - R_desc`), pretrained weights vs random router initialization (`C2: R_dope - R_randrouter`, 5 random router seeds), and Stage-1B `629 -> 96` SVD condensation loss (`C5: R_dope - R_tok`).
   - **H-Placebo (`C4`, `DVP-S1`)**: Cross-family donor-permuted `R_shuffled` (`family[donor] != family[i]`) and width-matched Gaussian `R_noise` both include `0` in their 95% paired family bootstrap CIs against `R_none` across `T1`–`T3`.
   - **H-Ultra (`C6`, `WP1`)**: Unmasking the 5 fine FIT-CV landmarks (`gbdt`, `knn1`, `knn16`, `net`, `gbdt_quarter`) by conditioning frozen DOPE at `level="ultra"` on true `ultra` descriptors (`R_dope_ultra`, never passing `high`-masked `vec256` to `ultra`) improves `T1`/`T2`/`T3` over `high`, while `T4` quantifies how much fine-landmark variance is unpredictable from `high`.
   - **H-Packet (`C7`, `WP2`)**: Extracting DOPE's fitted tabular generative program on FIT rows into the `196D` scale-invariant, column-permutation-invariant, extrema-free vector `R_packet_v1` improves program selection and operator/anchor prediction (`R_dope_plus_packet_v1 - R_dope` and `R_packet_v1 - R_packet_notarget`).

3. **Part C — Zero-Retraining Accuracy Upgrades (`WP3` & `B03`, Contrasts `C5` & `C8`)**:
   - **H-WP3 (`C8a` — Variance-Shrunk Predicted Reveals at 0 Real Evaluations)**: Cross-fitted out-of-family outcome + uncertainty heads `mu_hat(R, prog), sigma_hat(R, prog)` shrunk via Empirical Bayes `y_hat = lam * mu_hat + (1 - lam) * prior` (`lam = sigma_prior^2 / (sigma_prior^2 + sigma_hat^2)`) activate H184's revealed-history kernel at **0 candidate evaluations**, matching 0-reveal bit-for-bit when `lam = 0` (`V-S1`) and outperforming `incumbent_0rev`, `wave0_fake_reveal`, and cross-family `WP3_shuffled_ctrl` (`V-S2`, `V-S3`).
   - **H-B03 (`C8b` — Folded Stage-1B SVD `629 -> 96` + Zero-Init `Ultra+Packet` Residual Adapter)**: Folding `svd_slot{2,4,5}` `(mu, sd_safe, active_mask, effective_proj_mat)` into trainable `nn.Linear(629, 96)` (`B03_svd96`, compared against `pairnf_z40` and parameter-matched `96 -> 96` `B03_ctrl96`) and enriching slots `2, 4, 5` with `R_dope_ultra + R_packet_v1` via a zero-initialized residual adapter (`r_out = 0`, `alpha = 1.0`, bit-identical to `round-3175` at step 0) tests whether unlocking pre-SVD and packet/ultra features breaks the `pairnf_z40` plateau.

---

## 2. Exact Cohort, Partitions, and Zero-`val91`-Peeking Protocol

1. **Cohort**: `full-DL-Train1219-DL-Val91/v1` custody packet (`1,219` `DL-Train` contexts across `205` families; `91` `DL-Val` contexts across `19` families; `0` family overlap; `0` outer/query rows in `.jtt1` tables, `nq == 0`).
2. **Partitions**: Exact `H452`–`H469` floor-stratified family-grouped 5-fold CV partitions (`salts = ["h452a/", "h452b/"]`, fold sizes `[244, 244, 244, 244, 243]`, incumbent gate failures `[4, 4, 4, 3, 4]` per fold), evaluated across `3` seeds (`1, 2, 3`).
3. **Fold-Safety & Cache Authentication (`WP01`)**:
   - Every feature standardizer, PCA basis, anchor basket, ridge hyperparameter, LightGBM ranker, `WP3` outcome/variance head, and `B03` adapter step is fitted **strictly on the 4 training folds** of each `(partition, fold)` cell.
   - All cached representations in `/home/ubuntu/jope-data/hwave-h470/cache/` are guarded by `cache_auth.py` (`SHA-256` over `(dope_weights_sha256, stage1a_mode_and_bridge_sha256, prep_recipe, context_sha256, fit_rows_sha256, conditioning_level, stage1b_sha256)`), refusing any mismatched digest.
4. **Strict Zero `val91` Peeking**:
   - `DL-Val91` (`91` contexts, `19` families) is **never** read during Part A diagnostics, Part B probe matrix development, basket selection, or Part C model selection.
   - Pre-registered `REGCHECK` rule: after all `DL-Train` OOF results are frozen, `val91` is evaluated **at most once** on (i) the incumbent parity check, (ii) the CV-selected Part C neural arm per seed (using H452–H469's exact selection rule: max mean partition non-floor OOF Elo), and (iii) the CV-selected `WP3` 0-reveal arm vs 0-reveal incumbent.

---

## 3. Pre-Registered Arms, Readouts (`T1`–`T4`), Contrasts (`C0`–`C8`), and Statistics

### 3.1 Representation Arms
| Arm | Width | Definition (FIT-rows only) |
| --- | ---: | --- |
| `R_none` | 0 | Candidate program features only (zero dataset features) |
| `R_shape` | 3 | `[log10(n_fit)/6, log2(p)/12, is_clf]` |
| `lm48_z` | 48 | Standardized Stage-1A/1B `lm48_z` |
| `R_lm80` | 80 | Stage-1B `lm_smart_80` (`16x3` triple reshape of `lm48`) |
| `R_desc` | 290 | `vec256` (`256D`) + active-column `(mean, max)` of `col_lm` (`34D`) at `high` |
| `R_learned` | 373 | `dope629[256:629]` (learned router + column-gate block only) |
| `R_dope` | 663 | `dope629` (`629D`) + pooled `col_lm` (`34D`) at `high` |
| `R_hidden` | 256 | Frozen DOPE router pre-head hidden state `h` at `high` |
| `R_tok` | 320 | Flattened Stage-1B `10 x 32` tokens |
| `R_randrouter` | 663 | `R_dope` with `DopeEncoder629` weights re-initialized across 5 seeds (`20261101..20261105`) |
| `R_shuffled` | 663 | Cross-family donor-permuted `R_dope` (`family[donor] != family[i]`) |
| `R_noise` | 663 | Width-matched i.i.d. standard Gaussian `N(0, 1)` |
| `R_desc_ultra` | 290 | Native 23-landmark FIT-CV `vec256_ultra` + pooled `col_lm_ultra` (paired with `R_desc_native_high`) |
| `R_dope_ultra` | 663 | Frozen DOPE router at `level="ultra"` on `(vec256_ultra, col_lm_ultra)` + pooled `col_lm_ultra` (paired with `R_dope_native_high`) |
| `R_packet_v1` | 196 | Scale-invariant, column-equivariant, extrema-free DOPE fitted packet (`14+14+16+96+32+24`) |
| `R_packet_notarget` | 196 | `R_packet_v1` with all target-bundle (`P(Y|X)`) slots zeroed (`P-S2` ablation) |
| `R_dope_plus_packet_v1` | 859 | Concatenation `R_dope ⊕ R_packet_v1` (`P-S1`) |
| `profile_oracle` | ceiling | Split-pool `A -> B`: standardized pool-half `A` outcome profile scored strictly on disjoint half `B` |
| `R_oracle64` | ceiling | `K=64` pivoted-QR TRAIN-selected common-program outcomes + masks, scored on basket-excluded slates |

### 3.2 Probe Tasks (`T1`–`T4`)
- **T1 (Fixed LightGBM `num_leaves=15` Slate Ranking)**: Fixed `LGBMRanker` (`objective="lambdarank"`, `num_leaves=15`, `n_estimators=100`, `learning_rate=0.08`, `min_child_samples=25`, `colsample_bytree=0.8`, identical across all arms, zero per-arm tuning) trained on 96-candidate training-fold slates and scored over the **entire ~1,152-candidate pool** of every held-out context at both **0-reveal** and **4-reveal**. Reports family-mean utility (`-fitness` if `gate` else `-2.0`), pool regret, `NDCG@1`, `NDCG@8`, and gate-failure rate.
- **T2 (Winning Head & Nonlinearity Readout)**: Fold-safe L2-regularized multinomial logistic/ridge regression predicting the best passing program's 5-way terminal head family (`elasticnet`, `histgbm`, `ksmooth`, `ridge`, `ridge_gcv`) and 5-way stage nonlinearity regime. Reports held-out log-loss and macro-F1.
- **T3 (Held-Out Family Anchor Outcome $R^2$)**: Fold-safe Ridge regression predicting context-standardized passing fitness across the `32` highest-coverage `DL-Train` anchor programs; reports mean and median held-out $R^2$.
- **T4 (`Ultra` Fine-Landmark Redundancy)**: Fold-safe Ridge regression predicting the 5 real FIT-CV `ultra` fine landmark squashed gains (`gbdt`, `knn1`, `knn16`, `net`, `gbdt_quarter`) from `high` representations.

### 3.3 Contrasts & Statistical Protocol
- **Bootstrap**: `paired_family_bootstrap_ci(..., n_boot=10000, seed=20261118)` over the `205` `DL-Train` families (unweighted mean of per-family mean paired deltas, reporting `[ci_low, ci_high]`, two-sided `p_value`, family W/T/L, and family harm rate).
- **Multiple Testing**: Step-down `holm_bonferroni` across the primary Part B contrasts (`C0`–`C7`) and Part C contrasts (`C8`).

---

## 4. Stop / Go Decision Rules

- **`DVP-S1` (Placebo Cleanliness)**: Both `C4` placebos (`R_shuffled - R_none`, `R_noise - R_none`) include `0` in their 95% paired family bootstrap CIs on `T1`.
- **`DVP-S2` (Freeze & Cache Bit-Parity)**: Recomputed `hpc_high` `dope629` and `tokens_10x32` match H58/H469 (`max_abs_diff < 1e-6`), and `dope_before == dope_after == c51a56d0...`.
- **`G0` / `O-S2` (Ceiling Gate)**: `profile_oracle - R_none` and `R_oracle64 - R_dope` achieve `ci_low > 0` on `T1` 0-reveal utility; if null, halt outcome-supervised pretraining (`H2`) on this panel.
- **`U-S2` (`WP1` Ultra Gate)**: Productize `dope629-ultra/v1` only if `R_dope_ultra - R_dope` (and vs `R_dope_native_high`) has `ci_low > 0` on `T1` OOF and is not worse on `val91`.
- **`P-S1` / `P-S2` / `P-S3` (`WP2` Packet Gate)**: Advance `R_packet_v1` if `R_dope_plus_packet_v1 - R_dope` has `ci_low > 0` on `T1` or `T2` OOF, `P-S3` column-permutation invariance holds (`< 1e-6`), and `P-S2` (`R_packet_v1 - R_packet_notarget`) isolates target-bundle lift.
- **`V-S1` / `V-S2` / `V-S3` (`WP3` Predicted-Reveal Gate)**: `lam = 0` matches 0-reveal bit-for-bit (`V-S1`); promote `WP3` if it achieves `ci_low > 0` over `incumbent_0rev` (`>= 30%` of real 4-reveal gain for full `V-S2` target, or positive paired lift with family harm rate `<= 10%` for `V-S3`) and `not_worse` on `val91`.
- **`B03` Strict Flywheel Gate**: Matches `H452`–`H469` strict criteria (`pooled OOF gain > 0.002`, `gain CI low > 0`, `pooled Elo > 0`, `Elo CI low > 0`, `non-floor Elo CI low > 0`, `Elo > 0` on both partitions, paired gain `CI low > 0` vs `pairnf_z40` and `B03_ctrl96` across all 3 seeds, plus `val91` `not_worse`).
