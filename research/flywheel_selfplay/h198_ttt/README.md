# H198 — Per-Context Test-Time Training (TTT) of the `r882` Adapter on Revealed Candidates Only

**Verdict: `NO_ADVANCE`** (`reject treatment; keep r882`; `serving_promotion = false`).

Single guarded CPU attempt on `xbabe2` (`nice -n 19`, `1` torch math thread, `3 <= 4` peak aggregate threads, `3.44 GB < 6.0 GB` peak RSS, `2,052.47 s / 10,800 s` wall, `5.93 MB < 300 MB` local output root) executed from merged commit `208acd63772f4c07926027ee9d765cde80ef1198` ([PR #1632](https://github.com/neverhuman/JopeDime/pull/1632), exact-head review `5729825315`). Full outcomes and custody bindings are recorded in [`metrics.json`](metrics.json) and [`custody.json`](custody.json).

---

## 1. Intervention & Leakage Verification

Starting from a fresh copy of `round-882.pt` (`47` adapter tensors / `29,211` trainable parameters, initial and final SHA-256 `9c3ba940d9eb60aa2a39dd57c229c0952927ea1c2f22c372ae9764f43baa8669`, frozen `DOPE` `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`, frozen non-adapter hash `8d1070faf574ee1cd2a4e630887e1662b43189b3f89b04982143ef2704818aae`, structural kernel active as live), `H198` tests per-context plain-SGD test-time adaptation (`momentum=0.0, weight_decay=0.0`) strictly on the `4` revealed candidate rows (`revealed_indices = [0, 1, 2, 3]`) of each scored context, followed by an immediate bit-exact reset to `r882` after every context.

- **Exact `91/91` `DL-Val` baseline gate**: `PASS` (`mean = 1.1377941898164525`, per-context error `0.0`, single-context `TTT0` forward bit-exact with batched native `r882`).
- **Leakage red lines**:
  - `Engine.context_inputs` slices `[:4]` only and never passes full-pool `fitness` or `gate`; `ttt_context_trajectory` rejects any input dict containing `"fitness"` or `"gate"`; `realized_utility(cid, pick)` reads ground-truth outcomes only after post-TTT `argmin(costs)`.
  - Runtime poisoned-unrevealed-label self-check (`verify_leakage_mask_invariance`, `fitness[4:] = np.nan, gate[4:] = False`) passed bit-for-bit across `TTT_K`, `TTT_SHUF`, and `TTT_L2` on `8/8` `TRAIN` contexts before Stage 1.
  - Across the entire run (`24,340` plain-SGD steps), `9,424 / 9,424` per-context adapter resets verified `rt.tensor_hash == initial_hash` bit-exactly; runtime `sys.addaudithook` recorded `0` protected or refused file reads.

---

## 2. Stage-1 `B`-Only Grid Selection (`ttt_selection_lock.json`)

Seeds `198101, 198102, 198103` use independent `5`-fold family partitions (`fold_seeds = {198101: 198141, 198102: 198142, 198103: 198143}`; pairwise diagonal fold-Jaccard means `0.0965` (`198101–198102`), `0.1149` (`198101–198103`), and `0.0993` (`198102–198103`)).

On each of the `15` `(seed, outer)` `B` cells (`41` families each), the `3 x 3` grid `k in {1, 3, 5} x lr in {1e-5, 3e-5, 1e-4}` was scored using `B` equal-family utility only and locked (`fsync`ed to `ttt_selection_lock.json`, SHA-256 `f931a9c1c863d38fff9003fe3068f3ce7eed2da21cb0a35d440484e0f5829b1e`) prior to Stage-2 `A` readout:

| Grid Label | `(k, lr)` | Cells Selected (`15` total) | Pooled `TRAIN` (`205` fam) Median $\Delta$ vs `R882` `[95% CI]` | Harm Rate ($\Delta_f < 0$ / $\Delta_f < -0.01$) | `-2.0` Picks/Seed |
|---|---|---:|---|---:|---:|
| `K1_LR1e-05` | `(1, 1e-5)` | **`15 / 15`** | `0.000000 [0.000000, 0.000000]` | `0.00%` / `0.00%` | `[17, 17, 17]` |
| `K1_LR3e-05` | `(1, 3e-5)` | `0 / 15` | `0.000000 [0.000000, 0.000000]` | `0.00%` / `0.00%` | `[17, 17, 17]` |
| `K1_LR1e-04` | `(1, 1e-4)` | `0 / 15` | `0.000000 [0.000000, 0.000000]` | `0.00%` / `0.00%` | `[17, 17, 17]` |
| `K3_LR1e-05` | `(3, 1e-5)` | `0 / 15` | `0.000000 [0.000000, 0.000000]` | `0.00%` / `0.00%` | `[17, 17, 17]` |
| `K3_LR3e-05` | `(3, 3e-5)` | `0 / 15` | `0.000000 [0.000000, 0.000000]` | `0.00%` / `0.00%` | `[17, 17, 17]` |
| `K3_LR1e-04` | `(3, 1e-4)` | `0 / 15` | `-0.000094 [-0.000275, 0.000000]` | `0.98%` / `0.49%` | `[17, 17, 17]` |
| `K5_LR1e-05` | `(5, 1e-5)` | `0 / 15` | `0.000000 [0.000000, 0.000000]` | `0.00%` / `0.00%` | `[17, 17, 17]` |
| `K5_LR3e-05` | `(5, 3e-5)` | `0 / 15` | `0.000000 [0.000000, 0.000000]` | `0.00%` / `0.00%` | `[17, 17, 17]` |
| `K5_LR1e-04` | `(5, 1e-4)` | `0 / 15` | `-0.000128 [-0.000347, 0.000000]` | `1.46%` / `0.49%` | `[17, 17, 17]` |

No `(k, lr)` point improved `B` equal-family utility over `K1_LR1e-05` in any of the `15` cells (`7` grid points tied `R882` with `0` discrete argmin flips, while the two largest step budgets `K3_LR1e-04` and `K5_LR1e-04` degraded utility on `2/205` and `3/205` families). Deterministic tie-breaking toward smallest `k`, then smallest `lr` therefore locked `K1_LR1e-05` (`k* = 1, lr* = 1e-5`) in all `15/15` cells (and as the modal `B` policy for descriptive `DL-Val` across all `3` seeds).

---

## 3. Stage-2 Pooled `A` & Descriptive `DL-Val` Outcomes

| Arm | Role | Pooled `A` (`205` fam) Equal-Family Utility | Pooled `A` Median $\Delta$ vs `R882` `[95% CI]` | Holm $p$ ($m=1$) | Harm ($\Delta_f < 0$ / $<-0.01$) | Paired BT Elo vs `R882=1500` `[95% CI]` | Mean Adapter L2 Drift / Context (Max) | `DL-Val` (`19` fam) Equal-Family Utility ($\Delta$) |
|---|---|---:|---|---:|---:|---|---:|---:|
| `R882` (`TTT0`) | Baseline (`0` steps) | `1.385355` | — | — | — | `1500.00` | `0.0` | `1.153615` (`0.000000`) |
| `TTT_K` | **Primary** ($m=1$) | `1.385355` | `0.000000 [0.000000, 0.000000]` | `1.0000` | `0.00%` / `0.00%` | `1500.00 [1500.00, 1500.00]` | `1.046e-5` (`8.775e-5`) | `1.153615` (`0.000000`) |
| `TTT_SHUF` | Derangement placebo | `1.385355` | `0.000000 [0.000000, 0.000000]` | `1.0000` | `0.00%` / `0.00%` | `1500.00 [1500.00, 1500.00]` | `1.519e-5` (`1.464e-4`) | `1.153615` (`0.000000`) |
| `TTT_L2` | L2-anchored ($\beta=10$) | `1.385355` | `0.000000 [0.000000, 0.000000]` | `1.0000` | `0.00%` / `0.00%` | `1500.00 [1500.00, 1500.00]` | `1.046e-5` (`8.775e-5`) | `1.153615` (`0.000000`) |

- **Placebo & anchor contrasts**: `TTT_K - TTT_SHUF = 0.000000 [0.000000, 0.000000]` ($p = 1.0$); `TTT_K - TTT_L2 = 0.000000 [0.000000, 0.000000]` ($p = 1.0$).
- **`-2.0` invalid picks**: `[17, 17, 17]` on `TRAIN` (`1,219` contexts) and `[0, 0, 0]` on `DL-Val` (`91` contexts) across all arms.
- **Mechanistic interpretation**: Every context step updates the `29,211` adapter parameters with finite nonzero gradients (`mean_adapter_drift_per_context = 1.046e-5` for `TTT_K`/`TTT_L2` and `1.519e-5` for `TTT_SHUF`), but at `k=1, lr=1e-5` the score perturbation is strictly smaller than the top-1 argmin margin on all `1,219` pools (`0` pick changes). When `k * lr` is increased enough to cross argmin boundaries (`K3_LR1e-04`, `K5_LR1e-04`), fitting on only the `4` revealed candidates overfits the `4`-candidate anchor subset and degrades full-pool selection (`-0.000094` and `-0.000128`). Because the `r882` adapter already incorporates the `4` revealed candidates in closed form via its revealed-history structural kernel (`H184`), per-context gradient descent on the `4` revealed rows adds no positive signal.

---

## 4. Custody & Independent Numerical Replay

All candidate cost arrays and panel archives were streamed to versioned S3 under `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H198/rd_argon-t1205/` (`native/VAL.npz`, `native/TRAIN.npz`, `stage1/B_grid_costs.npz`, `stage2/A_and_VAL_costs.npz`, `stage2/panels.npz`, and `manifest.json`) and verified via exact-version (`VersionId`) full-payload and per-member SHA-256 readback (`20,434` verified array members; [`custody.json`](custody.json)). Independent model-free numerical replay from the restored S3 arrays reproduced the `91/91` `DL-Val` baseline (`1.1377941898164525`), all `10,971` Stage-1 `B`-grid cost vectors and `15` `B`-lock selections, and all Stage-2 contrasts bit-for-bit with `0` model forwards.
