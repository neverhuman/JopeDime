# H147 — Pre-Execution Launch Binding and Amendment A1 (`PREREG_AMENDMENT_A1.md`)

**Owner:** `gpu_fused_agy` `[FUSED-AGY]`
**Authority:** `~/nudges/hwave_t0000_h147exec_gpu_fused_agy.md` (explicit H-wave loop t0000 launch assignment under Jepson standing order).
**Base Preregistration:** [PREREG.md](PREREG.md) (`sha256 = f694b4386939ca18bee165de92aeb18a00d61080f415f096ee79b1396ae9ea89`, unchanged) and [fold_manifest.json](fold_manifest.json) (`sha256 = a03f5aa9d9d00d40ea2f990009e64e0844d72729e3ff2ca020ae9cb9be7f0954`, unchanged).
**One-Use Ledger:** `/home/ubuntu/jope_plan/ns/receipts/H147_exec_claim.txt`
**Absolute Deadline:** `2026-11-01T14:00:00Z`

## 1. Launch Authorization and Host Placement (`xbabe2` CPU)

[PREREG.md](PREREG.md) was filed by `rd_astra` as a preregistration-only proposal (`PROPOSED_NOT_EXECUTED`) requiring a separate launch assignment, a pinned `TRAIN-only` feature packet, an implemented executor, an atomic one-use ledger, and an absolute deadline before any scored execution. `~/nudges/hwave_t0000_h147exec_gpu_fused_agy.md` provides that explicit launch assignment to `gpu_fused_agy` and directs execution on **`xbabe2` CPU** (`CUDA_VISIBLE_DEVICES=""`, `nice 19`, `OMP/MKL/OPENBLAS/NUMEXPR/RAYON=1`, total process-tree OS threads $\le 4$, summed RSS $< 6,000,000,000$ B, free disk $\ge 40,000,000,000$ B, disk used $< 0.95$, `load1 <= 85.0`, `MemAvailable >= 16 GiB`, owned scratch $< 3,000,000,000$ B, hard-stop supervisor and runtime guards) while `qualification.lock` is held by `H157`.

## 2. Pinned `TRAIN-Only` Feature Packet and Harmonized `91`-Context `r882` Baseline Gate / Descriptive `DL-Val` Readout

1. **Owned `r882` Checkpoint Copy & `TRAIN-Only` Feature Packet (`/home/ubuntu/jope-data/h147/inputs/`):**
   - `round-882.pt`: `/home/ubuntu/jope-data/h147/inputs/round-882.pt` (`sha256 = 0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`).
   - `train_only_selected_1219.json`: `/home/ubuntu/jope-data/h147/inputs/train_only_selected_1219.json` (`sha256 = 110307b8f1fb19de1b33c4005004e7b1512eee1e5d2042db49d6282407a3ab33`), containing exclusively the `1,219` `DL-Train` contexts across `205` families (`0` `DL-Val` rows).
   - `train_only_context_features_1219.npz`: `/home/ubuntu/jope-data/h147/inputs/train_only_context_features_1219.npz` (`sha256 = 5194c29a916cbde22d5fe312a595c6d3e0d213dcbede048bea3c731a4bdf8f10`), containing exclusively the `1,219` `DL-Train` context token/mask arrays (`0` `DL-Val` entries).
   - Before any continuation step, two independent loads of `round-882.pt` are evaluated on the `1,219`-context `DL-Train` cohort and must reproduce identical full native utility vectors (`max_abs_diff == 0.0`) and hashes (`PREREG.md` Section "Proposed execution envelope and launch requirements").
2. **Harmonization of `hwave_t0000_h147exec_gpu_fused_agy.md` Required `91`-Context `r882` Baseline Gate (`1.1377941898164525`) and Descriptive `91`-Context `DL-Val` Readout:**
   - `~/nudges/hwave_t0000_h147exec_gpu_fused_agy.md` explicitly requires:
     - `exact 91-ctx r882 baseline PASS first (1.1377941898164525)`
     - `Report median-seed equal-family out-of-fold gain vs r882 + family-bootstrap 95% CI, Elo vs r882=1500, -2 counts, drift per arm, 91-ctx DL-Val descriptive.`
   - To satisfy both [PREREG.md](PREREG.md) (`TRAIN-only` primary closure without loading the mixed 1,310-context `H58` cache) and `~/nudges/hwave_t0000_h147exec_gpu_fused_agy.md`, the `91` `DL-Val` features are isolated in a separate read-only file `/home/ubuntu/jope-data/h147/inputs/val_only_context_features_91.npz` (`sha256 = c480fee5f4e26a3fb0efb0a294ad73ddbec72f7c1fc7d9344889d3a520efaa9e`) paired with `/home/ubuntu/jope-data/h147/inputs/best_dev_checkpoint.json` (`sha256 = 30853cad78d860870338316dfb5a4e869e46dc5c22d42fe4ab9d5f24f31c373a`).
   - Step 1 verifies the exact `91/91` `r882` baseline gate (`max_context_error == 0.0`, `mean == 1.1377941898164525`) AND the two-load `1,219/1,219` `DL-Train` `r882` parity gate (`max_context_error == 0.0`, context mean `1.423073434552299`, 205-family equal-family mean `1.3853554329931828`).
   - All `45` training/out-of-fold assessment cells (`3 arms x 3 seeds x 5 folds`, `1,800` updates) train exclusively on the `4` in-fold `DL-Train` groups (`164` families) and assess on the `1` held-out `DL-Train` group (`41` families) per [fold_manifest.json](fold_manifest.json). After each cell's out-of-fold `DL-Train` assessment is locked, a descriptive-only `91`-context `DL-Val` evaluation is recorded (never used for training, hyperparameter/step/arm selection, or the primary `ADVANCE` / `NO_ADVANCE` decision).
3. **Frozen Non-Adapter Anchor & `sem24` Evaluation/Step Cache (`runtime.py`):**
   - As verified in `H154` (`PR #1331`), `net.adapter_10tok.anchor` is asserted to be a frozen non-adapter module (`not anchor.training`, `requires_grad == False` on all parameters, zero overlap with the `47` trainable `names`, and `tensor_hash(anchor.state_dict())` verified before and after every step and evaluation).
   - Frozen `anchor.predict_costs` and `compute_semantic_struct_sig_24d_torch` outputs are cached keyed by `(split_name, tuple(context_ids), candidate_set_sha256)` (`eval_batch_cache_key`), producing `0.0` bitwise difference vs uncached execution.

## 3. Unchanged Scientific Contract

All arms (`SUPPORT_CTRL`, `STORAGE_REVEAL`, `STORAGE_MASK`), seeds (`147101, 147102, 147103`), fold seed (`147140`), bootstrap seed (`147141`, `10,000` shared 205-family draws), effective fold stream seed (`seed + 1000000 * fold`), common-support slate construction ($S = A \cup T_{48} \cup \text{draw}_{192 - |A \cup T_{48}|}$), 40-step AdamW continuation (`lr = 1e-4`, `weight_decay = 1e-4`, batch `16`, slate `192`, grad clip `1.0`, SmoothL1 $\beta=1.0$ + `0.1` KL $\tau=0.2$, unrevealed-188 masking in `STORAGE_MASK`), intersection Holm $m=2$ gate vs both `r882` and `SUPPORT_CTRL` (`median_gain > 0.001`, `ci95[0] > 0`, `>= 2/3` positive seeds, `<= 10%` negative 3-seed families vs `r882`, `top3_positive_share <= 0.80`), and pinned Bradley–Terry MM Elo (`r882 = 1500.0`, `tie_margin = 1e-4`, `prior = 0.01`, `400` iterations) remain **100% unchanged** from [PREREG.md](PREREG.md).
