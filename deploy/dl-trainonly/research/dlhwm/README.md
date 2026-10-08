# dlhwm — clean-license deep models for GP high-water-mark (HWM) prediction

Owner: ENG:watcher (DL_FINAL, 2026-10-08). Design: `DL_FINAL_DESIGN.md` (ofm-20261008/dl_final).

* **Inputs (precomputed only):** LV4 label-free train-row statistics (stats_v2.1 seeded, v3, v4 + log landmarkers + encoder
  query-fit scalars) and the clean encoder embeddings `kumo_tabular_l/s, mitra_v2, tabdpt13, tabicl2, a0, dope, hyperion_v6all`.
  `foundation, tabpfn35, target, context2048, hyperion_v7` (held pending lineage) and TA v8 blocks are refused at load time
  (`common.assert_clean`). TabDPT features are marked missing on TabDPT-pretraining-overlap datasets; the `--no-tabdpt` twin drops them.
* **Targets:** one-holdout GP-HWM `fitness`, `gap` (log GP/cheap-ladder ratio) and `beat10` (wave-10 v2.1 definitions).
  Sealed / heldout / unknown-role contexts are purged before the split; folds are lineage-grouped (`group_folds`, 5 x 3 repeats).
* **Wave 0 (PURE end-to-end DL):** `blockformer` (block-token transformer), `resmlp` (residual MLP on concatenated features),
  `gatedset` (gated set / Perceiver over blocks). No LightGBM anchor, no distillation, no GBM prediction as input.
  LightGBM (`lgb_baseline.py`, wave-5 `head_lgb`) is only the external baseline for the lift deltas.
* **What PURE means here:** block L contains the LV4 landmarkers `v4_s_hgb`, `v4_s_et`, `v4_hgb_minus_ridge`: HistGB/ExtraTrees
  cross-validated scores computed on each context's *own* outer-train rows. They carry no GP label and are not GBM predictions of the HWM
  target, so they are allowed in PURE arms (`gbm_input: false` refers to HWM-target GBM predictions).
* **Encoder mask (`--mask encoder_mask_final.json`):** `{"encoders": {enc: {"block_weight", "mode": "pcs"|"dims", "n_pcs": 32, "kept": [...], "raw_dims": [...]}}}`
  (or the short form `{"blocks": {enc: {"weight", "keep"}}}`). PCs are fit per fold on outer-train rows only (exact SVD, variance order, deterministic
  sign) after the robust scaling; `--mask-mode` forces the PC or raw-dim lists. Fails closed on unknown encoders/keys/modes; the mask sha256 is recorded
  and mask arms are tagged `optimistic` (the mask is selected on outer-fold OOS scores).
* **Wave 1:** three new proposals, including a matched GBM-anchored twin of the best wave-0 architecture (`--anchor`: output = LightGBM cross-fitted prediction + learned residual, GBM predictions also as inputs; no distillation term is implemented). The anchor fails closed unless its te/ti rows match the bundle's folds.
* **Checkpoints:** create-only to `s3://veox-jopedime-learning-use2/experiments/gp-policy-dl/v2-trainonly/models/hwm-dlwave/<run_id>/`
  with sha256 read-back (`common.put_create_only`), never a fake store.

| file | role |
|---|---|
| `common.py` | bucket/prefix, clean-license guard, create-only S3 put with sha256 read-back |
| `build_data.py` | bundle assembly: evidence (sha-verified), encodings, LV4 stats, targets, lineage graph, folds, MANIFEST |
| `flatstats/stats_backfill.py` | label-free LV4 stats replay for views not covered by wave 10 (pinned module md5s) |
| `models.py` | wave-0 architectures |
| `train.py` | per-arm outer-fold trainer: beta-NLL + BCE + RankNet, AdamW/warmup/cosine, EMA, early stop, block ablation, S3; wave-1 `--anchor` (matched GBM-anchored twin), `--pca`, `--mask` |
| `lgb_baseline.py` | external LV4 LightGBM baseline on the same folds (wave-5 `head_lgb` vendored; sees L only, so lift = DL(L+enc) vs LGB(L)) |
| `tests/` | guard refusal, create-only S3 stub, lineage-pure folds, fail-closed mask parser |
| `report.py` | per-fold loss curves, dashboard, paired lineage-cluster bootstrap lift vs LightGBM |

## Feature-subset wave (`dlsub-*`)

PURE resmlp, same lineage-grouped folds (5x3, seeds 10100+r) and 3 seeds per arm as wave 0. Only the inputs change:

| arm | flags |
|---|---|
| S0 all clean | `--enc all` |
| S1 LV4 only | `--enc none` |
| S2 LV4 minus Z/ZL | `--enc none --lgroups V21,TL,V3,V4` |
| S3-<enc> LV4 + one encoder | `--enc <enc>` (kumo_tabular_l, kumo_tabular_s, tabicl2, mitra_v2, tabdpt13) |
| S4 all minus TabDPT | `--enc <all clean except tabdpt13>` |
| S5-K50 / S5-K150 | `--topk K`: candidates = L cols + 32 fold-fit PCs per encoder, ranked by LightGBM TreeSHAP on inner-train rows only |
| S5-auto | built by `subsets.py`: K picked per outer fold by inner-validation `best_val_crit` only |
| S6 mask | `--mask encoder_mask_final.json` (optimistic: mask search saw all folds) |

`subsets.py` groups runs by `summary.subset`, makes 3-seed ensembles and writes `subsets_report.json` and `dashboard_subsets.png` with paired lineage-bootstrap CIs vs S0 and vs the LightGBM yardstick on the same rows. Run dirs and metrics are additive only. Existing wave-0 and wave-1 paths and schemas do not change.
