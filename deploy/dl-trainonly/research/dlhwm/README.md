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
* **Wave 1:** three new proposals, including a matched GBM-anchored/distilled twin of the best wave-0 architecture.
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
