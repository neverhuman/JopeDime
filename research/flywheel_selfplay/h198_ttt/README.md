# H198 — Per-Context Test-Time Training (TTT) of the `r882` Adapter on Revealed Candidates Only

**Verdict: `NO_ADVANCE (INERT TREATMENT)`** (`reject treatment (inert at k<=5, lr<=1e-4); keep r882`; `serving_promotion = false`).

The preregistered `k in {1, 3, 5} x lr in {1e-5, 3e-5, 1e-4}` grid updated the `r882` adapter parameters and changed candidate scores on **`100%` of contexts** (`1,219 / 1,219` `TRAIN` and `91 / 91` `DL-Val`), but moved candidate scores too little relative to the `r882` top-1 vs top-2 cost margin (`median max |delta score| = 1.55e-5` vs `median top-2 gap = 2.09e-2` at Stage-1 locked `k* = 1, lr* = 1e-5`, and `7.82e-4` at `K5_LR1e-04`) to flip any first-argmin pick at `k* = 1, lr* = 1e-5` (`0 / 1,219` `TRAIN`, `0 / 91` `DL-Val`) and flipped only `1–4 / 1,219` `TRAIN` contexts at the largest grid budgets. **This is an inert treatment at `k <= 5, lr <= 1e-4`, not evidence against test-time training at larger step sizes or step counts.**

Single guarded CPU attempt on `xbabe2` (`nice -n 19`, `1` torch math thread, `3 <= 4` peak aggregate threads, `3.44 GB < 6.0 GB` peak RSS, `2,052.47 s / 10,800 s` wall, `5.93 MB < 300 MB` local output root) executed from merged commit `208acd63772f4c07926027ee9d765cde80ef1198` ([PR #1632](https://github.com/neverhuman/JopeDime/pull/1632), exact-head review `5729825315`). Full outcomes and custody bindings are recorded in [`metrics.json`](metrics.json) and [`custody.json`](custody.json).

---

## 1. Hypothesis, Setup & Leakage Verification

- **Hypothesis**: Before scoring the unrevealed candidate pool of context $c$, taking $k \in \{1, 3, 5\}$ plain-SGD steps (`momentum=0.0, weight_decay=0.0`, $\text{lr} \in \{10^{-5}, 3\times 10^{-5}, 10^{-4}\}$) on a fresh copy of the `round-882.pt` adapter (`47` tensors / `29,211` parameters) using **only** the `4` revealed candidates (`revealed_indices = [0, 1, 2, 3]`) under the live `H146`/`H140` Hubered z-normalized listwise loss, then resetting the adapter bit-exact to `r882` after scoring context $c$, improves equal-family outer-fold `A` utility over `R882` (`TTT0`) and beats both a revealed-label derangement placebo (`TTT_SHUF`) and L2-anchored TTT (`TTT_L2`, $\beta = 10.0$).
- **Pinned baseline & model lineage**: `round-882.pt` (`initial_adapter_sha256 == final_adapter_sha256 == 9c3ba940d9eb60aa2a39dd57c229c0952927ea1c2f22c372ae9764f43baa8669`), frozen `DOPE` backbone (`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`), frozen non-adapter hash (`8d1070faf574ee1cd2a4e630887e1662b43189b3f89b04982143ef2704818aae`), structural kernel active as live.
- **Exact `91/91` `DL-Val` baseline gate**: `PASS` (`mean = 1.1377941898164525`, per-context error `0.0`, single-context `TTT0` forward bit-exact with batched native `r882`).
- **Leakage red lines**:
  - `Engine.context_inputs` slices `[:4]` only and never passes full-pool `fitness` or `gate`; `ttt_context_trajectory` rejects any input dict containing `"fitness"` or `"gate"`; `realized_utility(cid, pick)` reads ground-truth outcomes only after post-TTT `argmin(costs)`.
  - Runtime poisoned-unrevealed-label self-check (`verify_leakage_mask_invariance`, `fitness[4:] = np.nan, gate[4:] = False`) passed bit-for-bit across `TTT_K`, `TTT_SHUF`, and `TTT_L2` on `8/8` `TRAIN` contexts before Stage 1.
  - Across the entire run (`24,340` plain-SGD steps), `9,424 / 9,424` per-context adapter resets verified `rt.tensor_hash == initial_hash` bit-exactly; runtime `sys.addaudithook` recorded `0` protected or refused file reads.

---

## 2. Proof That TTT-Adapted Weights Were Used for Scoring (Not the Reset Copy)

### 2.1 Code Execution Order (`treatment.py`, `ttt_context_trajectory`, lines 160–214)

Inside the `try:` block of `ttt_context_trajectory`, each gradient step updates `pdict[n]` in-place (`optimizer.step()`, line `175`) and **immediately** scores all $M$ candidates using those adapted weights (`post = adapter_forward(...)[0]`, lines `178–190`), capturing `costs = post.detach().cpu().numpy().copy()` (line `191`), `pick = int(np.argmin(costs))` (line `194`), and `drift = adapter_drift(initial, {n: pdict[n] for n in names})` with a hard runtime assertion `drift > 0.0` (lines `195–197`). Only **after** the `for step in range(1, max_k + 1)` loop exits does the `finally:` block (lines `205–212`) restore `pdict[n].copy_(initial[n])` (`r882`) and verify `rt.tensor_hash(...) == initial_hash`.

### 2.2 Single-Context & Single-Cell Score-Delta Proof (`cell = (seed=198101, outer=0)`)

For cell `(seed=198101, outer=0)`, first outer-test `A_0` context `cid = 004a7438dc815b62c821976ac64379b7bf9a1a138d3ca441a9fb5a5e0e5dfb24` (`dataset_id = e745feb6c192f73b`, `family_id = fam-3682425120361f20ec8ece6c872e95f6c83c828e30205adb5206473540ca5247`, $M = 1{,}152$ candidates), `R882` has `argmin = 573`, best cost $s_{(1)} = -4.69346666$, second-best cost $s_{(2)} = -4.47974730$, and top-1 vs top-2 margin $s_{(2)} - s_{(1)} = 2.13719368 \times 10^{-1}$:

| Scope / Arm on `cid = 004a7438dc815b62...` (`M = 1,152`) | `(k, lr)` | `max \|score_ttt - score_r882\|` | `mean \|score_ttt - score_r882\|` | `\|\|score_ttt - score_r882\|\|_2` | Candidates Changed (`\|delta\| > 0`) | `argmin` Pick (`R882 = 573`) |
|---|---|---:|---:|---:|---:|---:|
| Stage-2 `TTT_K` (cell `198101, outer=0`) | `(1, 1e-5)` | `7.62939453e-05` | `2.32763123e-05` | `9.26294888e-04` | **`1,112 / 1,152`** | `573` |
| Stage-2 `TTT_SHUF` (cell `198101, outer=0`) | `(1, 1e-5)` | `3.05175781e-05` | `5.90424861e-06` | `2.71950150e-04` | **`923 / 1,152`** | `573` |
| Stage-2 `TTT_L2` (cell `198101, outer=0`) | `(1, 1e-5)` | `7.62939453e-05` | `2.32763123e-05` | `9.26294888e-04` | **`1,112 / 1,152`** | `573` |
| Stage-1 `K1_LR3e-05` | `(1, 3e-5)` | `2.21252441e-04` | `6.99290031e-05` | `2.77930088e-03` | **`1,139 / 1,152`** | `573` |
| Stage-1 `K1_LR1e-04` | `(1, 1e-4)` | `7.24792480e-04` | `2.32954815e-04` | `9.26273689e-03` | **`1,148 / 1,152`** | `573` |
| Stage-1 `K3_LR1e-05` | `(3, 1e-5)` | `2.21252441e-04` | `6.99043449e-05` | `2.77859624e-03` | **`1,139 / 1,152`** | `573` |
| Stage-1 `K3_LR3e-05` | `(3, 3e-5)` | `6.56127930e-04` | `2.09589227e-04` | `8.33273586e-03` | **`1,148 / 1,152`** | `573` |
| Stage-1 `K3_LR1e-04` | `(3, 1e-4)` | `2.17437744e-03` | `6.98018353e-04` | `2.77469102e-02` | **`1,151 / 1,152`** | `573` |
| Stage-1 `K5_LR1e-05` | `(5, 1e-5)` | `3.66210938e-04` | `1.16555901e-04` | `4.63218940e-03` | **`1,146 / 1,152`** | `573` |
| Stage-1 `K5_LR3e-05` | `(5, 3e-5)` | `1.09100342e-03` | `3.49342183e-04` | `1.38834678e-02` | **`1,149 / 1,152`** | `573` |
| Stage-1 `K5_LR1e-04` | `(5, 1e-4)` | `3.62396240e-03` | `1.16204203e-03` | `4.61889952e-02` | **`1,152 / 1,152`** | `573` |

Likewise, on the first `B_0` context of cell `(198101, outer=0)` (`cid = 004aa15dce5500aa...`, $M = 1{,}152$, `top2_gap = 3.94914150e-02`), `TTT_K` has `max |delta score| = 1.54972076e-05` (`1,149 / 1,152` candidates changed, `pick = 674`); and on the first `DL-Val` context (`cid = 016648dc8e6f474c...`, $M = 1{,}152$, `top2_gap = 1.48671746e-01`), `TTT_K` has `max |delta score| = 1.29699707e-04` (`1,109 / 1,152` candidates changed, `pick = 678`).

### 2.3 Count of Contexts Where Any Score Changed vs. `argmin` Flipped

| Evaluation Scope | Arm / Grid Point | Contexts With Any Score Changed (`max \|delta\| > 0`) | Contexts With `argmin` Flip (`pick != r882_pick`) | Contexts With Utility Change | `max \|delta score\|` `min / median / mean / max` | Median `R882` Top-2 Margin $s_{(2)} - s_{(1)}$ |
|---|---|---:|---:|---:|---|---:|
| Cell `(198101, outer=0)` `A_0` (`233` ctx) | `TTT_K` (`1, 1e-5`) | **`233 / 233` (`100%`)** | `0 / 233` | `0 / 233` | `1.88e-06 / 1.69e-05 / 4.93e-05 / 1.00e-03` | `2.56e-02` |
| Cell `(198101, outer=0)` `A_0` (`233` ctx) | `TTT_SHUF` (`1, 1e-5`) | **`233 / 233` (`100%`)** | `0 / 233` | `0 / 233` | `1.52e-06 / 2.48e-05 / 1.22e-04 / 5.31e-03` | `2.56e-02` |
| Cell `(198101, outer=0)` `A_0` (`233` ctx) | `TTT_L2` (`1, 1e-5`) | **`233 / 233` (`100%`)** | `0 / 233` | `0 / 233` | `1.88e-06 / 1.69e-05 / 4.93e-05 / 1.00e-03` | `2.56e-02` |
| Cell `(198101, outer=0)` `B_0` (`254` ctx) | All `9` grid points | **`254 / 254` (`100%`)** | `0 / 254` | `0 / 254` | `1.53e-05` (`K1_LR1e-05` med) $\to$ `7.36e-04` (`K5_LR1e-04` med) | `2.09e-02` |
| Pooled `TRAIN` `A` (`1,219` ctx, all 3 seeds) | `TTT_K` (`1, 1e-5`) | **`1,219 / 1,219` (`100%`)** | `0 / 1,219` | `0 / 1,219` | `9.54e-07 / 1.55e-05 / 3.48e-05 / 1.44e-03` | `2.09e-02` |
| Pooled `TRAIN` `A` (`1,219` ctx, all 3 seeds) | `TTT_SHUF` (`1, 1e-5`) | **`1,219 / 1,219` (`100%`)** | `0 / 1,219` | `0 / 1,219` | `5.96e-07 / 2.22e-05 / 6.05e-05 / 5.31e-03` | `2.09e-02` |
| Pooled `TRAIN` `A` (`1,219` ctx, all 3 seeds) | `TTT_L2` (`1, 1e-5`) | **`1,219 / 1,219` (`100%`)** | `0 / 1,219` | `0 / 1,219` | `9.54e-07 / 1.55e-05 / 3.48e-05 / 1.44e-03` | `2.09e-02` |
| Held-out `DL-Val` (`91` ctx, all 3 seeds) | `TTT_K` / `TTT_L2` (`1, 1e-5`) | **`91 / 91` (`100%`)** | `0 / 91` | `0 / 91` | `2.62e-06 / 1.53e-05 / 2.66e-05 / 1.68e-04` | `2.22e-02` |
| Held-out `DL-Val` (`91` ctx, all 3 seeds) | `TTT_SHUF` (`1, 1e-5`) | **`91 / 91` (`100%`)** | `0 / 91` | `0 / 91` | `5.84e-06 / 2.10e-05 / 3.93e-05 / 6.56e-04` | `2.22e-02` |
| Stage-1 Grid on `TRAIN` (`1,219` ctx) | `K1_LR1e-05`, `K1_LR3e-05`, `K3_LR1e-05`, `K5_LR1e-05` | **`1,219 / 1,219` (`100%`)** | `0 / 1,219` | `0 / 1,219` | med `1.55e-05`, `4.67e-05`, `4.67e-05`, `7.82e-05` | `2.09e-02` |
| Stage-1 Grid on `TRAIN` (`1,219` ctx) | `K1_LR1e-04`, `K3_LR3e-05`, `K5_LR3e-05` | **`1,219 / 1,219` (`100%`)** | **`1 / 1,219`** (`e47eb2a306c2`, `729 -> 756`, tied fitness) | `0 / 1,219` | med `1.56e-04`, `1.41e-04`, `2.35e-04` | `2.09e-02` |
| Stage-1 Grid on `TRAIN` (`1,219` ctx) | `K3_LR1e-04` (`3, 1e-4`) | **`1,219 / 1,219` (`100%`)** | **`3 / 1,219`** (`4a9c8c621585`, `68efbff713c8`, `e47eb2a306c2`) | **`2 / 1,219`** (`2/205` fam) | `2.67e-05 / 4.69e-04 / 1.04e-03 / 4.21e-02` | `2.09e-02` |
| Stage-1 Grid on `TRAIN` (`1,219` ctx) | `K5_LR1e-04` (`5, 1e-4`) | **`1,219 / 1,219` (`100%`)** | **`4 / 1,219`** (`2b2bbf534dca`, `4a9c8c621585`, `68efbff713c8`, `e47eb2a306c2`) | **`3 / 1,219`** (`3/205` fam) | `4.48e-05 / 7.82e-04 / 1.73e-03 / 6.81e-02` | `2.09e-02` |

---

## 3. Stage-1 `B`-Only Grid Selection (`ttt_selection_lock.json`)

Seeds `198101, 198102, 198103` use independent `5`-fold family partitions (`fold_seeds = {198101: 198141, 198102: 198142, 198103: 198143}`; pairwise diagonal fold-Jaccard means `0.0965` (`198101–198102`), `0.1149` (`198101–198103`), and `0.0993` (`198102–198103`)).

On each of the `15` `(seed, outer)` `B` cells (`41` families each), the `3 x 3` grid `k in {1, 3, 5} x lr in {1e-5, 3e-5, 1e-4}` was scored using `B` equal-family utility only and locked (`fsync`ed to `ttt_selection_lock.json`, SHA-256 `f931a9c1c863d38fff9003fe3068f3ce7eed2da21cb0a35d440484e0f5829b1e`) prior to Stage-2 `A` readout:

| Grid Label | `(k, lr)` | Cells Selected (`15` total) | `TRAIN` Score Changed / `argmin` Flips (`1,219` ctx) | Pooled `TRAIN` (`205` fam) Median $\Delta$ vs `R882` `[95% CI]` | Harm Rate ($\Delta_f < 0$ / $\Delta_f < -0.01$) | `-2.0` Picks/Seed |
|---|---|---:|---:|---|---:|---:|
| `K1_LR1e-05` | `(1, 1e-5)` | **`15 / 15`** | `1,219` / `0` | `0.000000 [0.000000, 0.000000]` | `0.00%` / `0.00%` | `[17, 17, 17]` |
| `K1_LR3e-05` | `(1, 3e-5)` | `0 / 15` | `1,219` / `0` | `0.000000 [0.000000, 0.000000]` | `0.00%` / `0.00%` | `[17, 17, 17]` |
| `K1_LR1e-04` | `(1, 1e-4)` | `0 / 15` | `1,219` / `1` | `0.000000 [0.000000, 0.000000]` | `0.00%` / `0.00%` | `[17, 17, 17]` |
| `K3_LR1e-05` | `(3, 1e-5)` | `0 / 15` | `1,219` / `0` | `0.000000 [0.000000, 0.000000]` | `0.00%` / `0.00%` | `[17, 17, 17]` |
| `K3_LR3e-05` | `(3, 3e-5)` | `0 / 15` | `1,219` / `1` | `0.000000 [0.000000, 0.000000]` | `0.00%` / `0.00%` | `[17, 17, 17]` |
| `K3_LR1e-04` | `(3, 1e-4)` | `0 / 15` | `1,219` / `3` | `-0.000094 [-0.000275, 0.000000]` | `0.98%` / `0.49%` | `[17, 17, 17]` |
| `K5_LR1e-05` | `(5, 1e-5)` | `0 / 15` | `1,219` / `0` | `0.000000 [0.000000, 0.000000]` | `0.00%` / `0.00%` | `[17, 17, 17]` |
| `K5_LR3e-05` | `(5, 3e-5)` | `0 / 15` | `1,219` / `1` | `0.000000 [0.000000, 0.000000]` | `0.00%` / `0.00%` | `[17, 17, 17]` |
| `K5_LR1e-04` | `(5, 1e-4)` | `0 / 15` | `1,219` / `4` | `-0.000128 [-0.000347, 0.000000]` | `1.46%` / `0.49%` | `[17, 17, 17]` |

No `(k, lr)` point improved `B` equal-family utility over `K1_LR1e-05` in any of the `15` cells. Deterministic tie-breaking toward smallest `k`, then smallest `lr` therefore locked `K1_LR1e-05` (`k* = 1, lr* = 1e-5`) in all `15/15` cells (and as the modal `B` policy for descriptive `DL-Val` across all `3` seeds).

---

## 4. Stage-2 Pooled `A`, Family-Grouped `DL-Val` & Paired BT Elo Outcomes (3 Seeds)

| Arm | Role | Pooled `A` (`205` fam) Equal-Family Utility | Pooled `A` Median $\Delta$ vs `R882` `[95% CI]` (3 Seeds) | Holm $p$ ($m=1$) | Harm ($\Delta_f < 0$ / $<-0.01$) | Paired BT Elo vs `R882=1500` `[95% CI]` | Mean Adapter L2 Drift / Context (Max) | `DL-Val` (`19` fam) Equal-Family Utility ($\Delta$ `[95% CI]`) |
|---|---|---:|---|---:|---:|---|---:|---|
| `R882` (`TTT0`) | Baseline (`0` steps) | `1.385355` | — | — | — | `1500.00` | `0.0` | `1.153615` |
| `TTT_K` | **Primary** ($m=1$) | `1.385355` | `0.000000 [0.000000, 0.000000]` | `1.0000` | `0.00%` / `0.00%` | `1500.00 [1500.00, 1500.00]` | `1.046e-5` (`8.775e-5`) | `1.153615` (`0.000000 [0.000000, 0.000000]`) |
| `TTT_SHUF` | Derangement placebo | `1.385355` | `0.000000 [0.000000, 0.000000]` | `1.0000` | `0.00%` / `0.00%` | `1500.00 [1500.00, 1500.00]` | `1.519e-5` (`1.464e-4`) | `1.153615` (`0.000000 [0.000000, 0.000000]`) |
| `TTT_L2` | L2-anchored ($\beta=10$) | `1.385355` | `0.000000 [0.000000, 0.000000]` | `1.0000` | `0.00%` / `0.00%` | `1500.00 [1500.00, 1500.00]` | `1.046e-5` (`8.775e-5`) | `1.153615` (`0.000000 [0.000000, 0.000000]`) |

- **Placebo & anchor contrasts**: `TTT_K - TTT_SHUF = 0.000000 [0.000000, 0.000000]` ($p = 1.0$); `TTT_K - TTT_L2 = 0.000000 [0.000000, 0.000000]` ($p = 1.0$).
- **`-2.0` invalid picks**: `[17, 17, 17]` on `TRAIN` (`1,219` contexts) and `[0, 0, 0]` on `DL-Val` (`91` contexts) across all arms.

---

## 5. Verdict & Next Step

- **Verdict**: **`NO_ADVANCE (INERT TREATMENT)`** (`reject treatment; keep r882`; `serving_promotion = false`).
  - The preregistered `k <= 5, lr <= 1e-4` grid moved the `29,211` adapter parameters and altered candidate scores on `1,219/1,219` `TRAIN` contexts and `91/91` `DL-Val` contexts, but at `k* = 1, lr* = 1e-5` the resulting score shift (`median max |delta score| = 1.55e-5`, `max = 1.44e-3`) was two orders of magnitude smaller than the `r882` top-1 vs top-2 cost margin (`median = 2.09e-2`), producing `0 / 1,219` first-argmin flips (and only `1–4 / 1,219` flips at the upper corner `k in {3, 5}, lr = 1e-4`).
  - Consequently, `H198` is an **inert treatment** at `k <= 5, lr <= 1e-4`, not evidence against per-context test-time training at larger step sizes. Per governance rules, `H198` is not rerun with a larger grid.
- **Next step (new `H` wave if pursued)**:
  - If per-context revealed-only TTT is revisited under a new `H` number, preregister a step-size / step-budget grid calibrated directly to the `r882` top-2 cost margin ($\text{median } s_{(2)} - s_{(1)} = 2.09\times 10^{-2}$), such as $\text{lr} \in \{3\times 10^{-4}, 10^{-3}, 3\times 10^{-3}\}$ or output-head-only closed-form ridge/temperature adaptation on the `4` revealed candidates, paired with an explicit KL/L2 trust region around `r882` so that candidate score shifts reach $O(10^{-2})$ without overfitting the `4`-candidate anchor subset.

---

## 6. Custody & Independent Numerical Replay

All candidate cost arrays and panel archives were streamed to versioned S3 under `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H198/rd_argon-t1205/` (`native/VAL.npz`, `native/TRAIN.npz`, `stage1/B_grid_costs.npz`, `stage2/A_and_VAL_costs.npz`, `stage2/panels.npz`, and `manifest.json`) and verified via exact-version (`VersionId`) full-payload and per-member SHA-256 readback (`20,434` verified array members; [`custody.json`](custody.json)). Independent model-free numerical replay from the restored S3 arrays reproduced the `91/91` `DL-Val` baseline (`1.1377941898164525`), all `10,971` Stage-1 `B`-grid cost vectors and `15` `B`-lock selections, and all Stage-2 contrasts bit-for-bit with `0` model forwards.
