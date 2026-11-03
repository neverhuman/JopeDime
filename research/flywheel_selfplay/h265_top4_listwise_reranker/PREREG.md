# H265 Preregistration (`h265_top4_listwise_reranker`)

## 1. Scientific Question & Hypothesis

**Motivation**:
- `H258` demonstrated that `r882`'s own `top-4` candidates (`order_rcost[:4]`) already contain high-utility programs (`TOP4_PICK@r882` reaches `1.2403601682250796` mean context utility / `1.2519997168684853` mean family utility on `DL-Val91`, vs `1.1377941898164525` for `r882` rank-1 single-pick).
- Thus, `+0.1026` context / `+0.0984` family utility of headroom sits purely in the **ORDER inside `r882`'s `top-4`**.
- While `H263` (`rd_astra`) attacks this ordering problem via gradient updates to the adapter, **`H265` (`h265_top4_listwise_reranker`) is the complementary zero-update test**: a small separate `FIT`-only listwise LambdaMART re-ranker (`lightgbm.LGBMRanker(objective="lambdarank")`) that re-orders **ONLY `r882`'s `top-4`** using strictly frozen outcome-free features and **NO reveals at test time (`0` test-time reveals)**.
- Unlike `GATED_025` / `H223` / `H224` (which blend a HistGBM over all `1,152` pool candidates or gate a full-pool blend), `H265` restricts action space strictly to permuting `r882`'s `top-4` and is evaluated under **SINGLE-PICK** (`0` test-time reveals = fixed artifact `r882` + reranker).

## 2. Locked Inputs & Provenance

| Artifact | Path | SHA-256 |
| --- | --- | --- |
| Read-Only Primary Checkpoint (`r882`) | `/home/ubuntu/jope_plan/evidence/H265/argon-t1658/inputs/r882/round-882.pt` | `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64` |
| Frozen DOPE Backbone Metadata | Embedded in `round-882.pt` (`dope_sha256`) | `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` |
| Canonical Cohort Selection (`selected.json`) | `/home/ubuntu/jope-data/h50/input/selected.json` (`DL-Train`: `1,219` ctx / `205` fam; `DL-Val`: `91` ctx / `19` fam) | `981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1` |
| Source Closure Manifest (`source_manifest.json`) | `/home/ubuntu/jope-data/h50/input/source_manifest.json` | `99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4` |
| Context Features Cache (`context_features.npz`) | `/home/ubuntu/jope-data/h58/cache/context_features.npz` | `75704e8279c19fee8a7843ccbea801e3cd4b67e0ee08d282b070ddae7ab82faf` |
| Pool Priors Cache (`pool_priors.npz`) | `/home/ubuntu/jope-data/h58/cache/pool_priors.npz` | `6db567ef2c10d2c67c375f7500305ec693a700c4e71eb23003b4cd312ee51cbe` |
| 5-Fold Family Partition (`fold_seed = 265140`) | `research/flywheel_selfplay/h265_top4_listwise_reranker/fold_manifest.json` | Computed before any outcome read (`outcomes_read = 0`) |

## 3. Per-Context Top-4 Feature Construction, Labels, and Arms

- **Per-context candidate set**: Fixed to `top4 = np.argsort(rcost, kind="mergesort")[:4]` (`4` candidates with lowest `r882` predicted cost, ordered $j \in \{0, 1, 2, 3\}$ where $j = 0$ is `r882` rank-1).
- **Outcome-free feature vector per candidate ($560$ float32 features, $0$ revealed costs, $0$ `DL-Val` labels)**:
  1. `r882` score, rank, gap to rank-1, exact `H217` `zgap` (`(rcost[j] - rcost[0]) / std(rcost)`), context top-2 zgap & top-4 spread, softmax probabilities over `top4` ($\tau \in \{1.0, 0.2\}$), `base_prior_cost` z-score, and raw + standardized pairwise score differences to the other $3$ candidates in `top4` (`20` features).
  2. Within-`top4` and pool semantic/projection cosine geometry (`sem24` and `proj16` cosines to rank-1, mean/max cosine to other $3$, cosine to pool top-8 centroid, operator count total & diff to rank-1, learner family agreement with rank-1; `12` features).
  3. Frozen candidate-feature columns already used by `H217`/`GATED_025` and `H252` (`program` `412` cols + pool-standardized `sem24` `24` cols + pool-standardized `proj16` `16` cols; `452` features).
  4. Family-free context features (`lm48_z` `48` cols + `dope629` `28` cols/summary; `76` features).
- **Training label (`DL-Train` `FIT` folds only)**:
  - True revealed cost rank within the $4$ candidates on `DL-Train` `FIT` contexts, mapped to integer listwise relevance `rel[j] in {0, 1, 2, 3}` (higher = better): any invalid gate candidate (`utility = -2.0`) receives relevance `0` (worst); valid candidates (`gate == True`) are ordered by ascending true `fitness` (ties broken by `r882` order) receiving relevance `3, 2, 1, 0`.
- **Arms**:
  1. **`CONTROL` (`r882`)**: Always picks `top4[0]` (`r882` rank-1). Must reproduce `Native91` `1.1377941898164525` (`91/91` bit-exact) FIRST; also reproduce `TOP4_PICK@r882` `1.2403601682250796` ctx / `1.2519997168684853` fam as the descriptive 4-reveal oracle ceiling.
  2. **`RERANK4` (PRIMARY)**:
     - `lightgbm.LGBMRanker(objective="lambdarank", max_depth=3, num_leaves=7, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0)`.
     - Across `15` cells (`5` folds $\times$ `3` seeds `265101, 265102, 265103` with `fold_seed = 265140`), each cell holds out `ASSESS` fold $k$ (`41` families) and uses the remaining `4` folds (`164` families) as `FIT`.
     - Shrinkage (`learning_rate` $\in \{0.03, 0.05, 0.10\}$), tree count (`n_estimators` $\in \{50, 100, 200\}$), and fallback margin $m \in \{0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, \infty\}$ are chosen by 4-fold inner cross-validation across the `164` `FIT` families.
     - Fallback rule: pick `j* = argmax_{j in 0..3} score[j]` iff `score[j*] - score[0] > m`, else keep `r882` rank-1 (`j = 0`).
     - **Strict FIT Selector (`as H257`)**: A cell's inner-FIT best `(lr*, n_est*, m*_unc)` is admitted if and only if inner-FIT equal-family utility gain $> +0.001 + 10^{-12}$ **AND** inner-FIT paired family bootstrap $\text{LCB95} > 0$; otherwise $m^* = \infty$ (fallback to `r882` rank-1).
     - All `15` cells' selections are globally `os.fsync`-locked to `fit_selections_lock.json` before any `DL-Val` feature or pool read for model scoring.
  3. **`RERANK4_SHUF` (SECONDARY placebo)**:
     - Candidate relevance labels permuted within each `FIT` family (`shuffle_labels_within_family`), using the exact same model class, hyperparameter grid, and strict FIT selector.
  4. **Descriptive Unconstrained Diagnostics (`RERANK4_UNC`, `RERANK4_SHUF_UNC`, `RERANK4_M0`, `RERANK4_SHUF_M0`, `TOP4_PICK_ORACLE`)**:
     - Reported alongside the strict selector (as in `H257`) to show raw re-ranking accuracy, changed-choice counts, and utility without the `m = inf` fallback gate.

## 4. Statistical Protocol & Decision Criteria (`Holm m = 2`)

- **Primary Contrast (`Holm m = 2`)**: `DL-Val91` single-pick equal-family utility of `RERANK4 - CONTROL` (`r882`), where each seed (`265101–265103`) is the 5-cell utility average across the `5` outer folds (`10,000` paired family bootstrap draws, `boot_seed = 265150`).
- **Secondary Contrast (`Holm m = 2`)**: `RERANK4_SHUF - CONTROL`, plus direct comparison `RERANK4 - RERANK4_SHUF`.
- **Required Diagnostics**:
  - Changed-choice count (`j* != 0`), top-1-within-top-4 accuracy (vs `TOP4_PICK_ORACLE`), harm families (`delta < -1e-4` and `delta < -0.01`), Context `#27` (`43022ae6a8bc83e9e25ea4a2882389152856226f5b2bdd4668db07f2f510ae91`), `-2.0` invalid gate pick counts, `TRAIN OOF` (`1,219` contexts / `205` families), and `400`-iteration MM Bradley-Terry Elo (`r882 = 1500.0`). No AUC or speed verdicts.
- **ADVANCE Criteria**:
  - `ADVANCE` iff on `DL-Val91`:
    1. Median family gain `> +0.002 + 1e-12`,
    2. `LCB95 > 0` and Holm-adjusted 95% LCB `> 0`,
    3. Holm `p < 0.05` (`m = 2`),
    4. `3 / 3` seeds positive,
    5. `RERANK4` beats `RERANK4_SHUF` (`median > 0` and `LCB95 > 0`).
  - Ties or fallback identity (`m* = inf` across cells) = **`NO_ADVANCE` ("fallback identity, not equivalence")**.
