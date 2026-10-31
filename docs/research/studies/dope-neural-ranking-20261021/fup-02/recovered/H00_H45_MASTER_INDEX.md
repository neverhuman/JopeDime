# Waves H00–H45 Master Index (`v4.7` Reconciled)

- **Whitepaper**: [`H35_H45_PERPETUAL_CLUSTER_FLYWHEEL_REPORT.md`](H35_H45_PERPETUAL_CLUSTER_FLYWHEEL_REPORT.md)
- **Visual Time-Series Plot**: `h40_h45_self_improving_flywheel_timeseries_20261026.png` (`SHA-256: 31ccd01f5ee41cd399a411a6596c7e745b520d3851b496695ac6f927fb091c1e`) / `h40_h45_self_improving_flywheel_timeseries_20261026.pdf` (external payloads; see the [S3 custody receipt](h40_h45_s3_custody_receipt_v4_7_20261026.json) for object identity).
- **Global Additive Accounting & Qualification Receipt (`v3.0`)**: `h40_h45_global_additive_accounting_reconciliation_v3.json` (external payload; see the [S3 custody receipt](h40_h45_s3_custody_receipt_v4_7_20261026.json) for object identity).
- **Full-S3 7-Split & All-1,298-Splits Receipt**: `h40_h45_full_s3_7split_and_all_1298_splits_receipt.json` (`SHA-256: 9675ef12af46f2cb...`; external payload; see the [S3 custody receipt](h40_h45_s3_custody_receipt_v4_7_20261026.json) for object identity).
- **S3 Custody Verification Receipt (`v4.7`)**: [`h40_h45_s3_custody_receipt_v4_7_20261026.json`](h40_h45_s3_custody_receipt_v4_7_20261026.json) (`s3://veox-jopedime-archive-use2/evidence/flywheel-h40-h45-20261026/`)
- **Crash-Safe Synthetic Fixture Suite (`v15`, `32/32 PASS`)**: `flywheel_dope_high_only_fixture_verification_receipt_v15_20261026T141435Z.json` (`SHA-256: c29c4e85c255ef37eed9ba1dcd44cc5822ae5abfa63145dee2e84557658965fa`)

## Wave Summary Table (`H35`–`H45`)

| Wave | Scope & Protocol | Key Verified Metrics & Matched Controls | Qualification Disposition |
| :--- | :--- | :--- | :---: |
| **`H35`–`H37`** | Legacy 9-Encoder Ablation & Initial `2+1+1` Search | Identified `G0` nomination over-selection (`-0.42274` mean utility) and quantile parity requirements. | Diagnostic Baseline |
| **`H38`–`H39`** | First-Party `DOPE` High (`629D` $\to$ `256D`) + `LM_SMART_80D` + Block-RMS SVD (`alpha=0.65`) | Beat external model swath (`Kumo`, `TabICL v2`, `Mitra v2`) on `st_val_gate_32` (`-0.980614` vs. `-0.9781`) and `stage2_20` Cycle 2 (`-0.37466`, `8W/8T/4L`). | Qualified Stage-A / Stage-2 (`H39`) |
| **`H40-Fast` & `H40B`** | `112` Guarded Vectorized Adapter Updates (`115.22s`) + `48` Post-Guard `10TOK` Updates (`98.0s`), `0` Native Fits (Cached-Pool Replay) | `dl_holdout_91` (reused holdout) $u_w$: `COMPLETE` `-1.133758` $\to$ `-1.134014`; `N0` `-1.134141`. `stage2_20` `G1C` replay: `COMPLETE` `+0.010238` $\to$ `+0.007644` (`2W/14T/4L`); `N0` `+0.048538` (`2W/13T/5L`, `std=0.2178`). | Cached-Pool Replay (`0` Native Fits) |
| **`Track 2` Full-S3** | `26,679` S3 Manifest Cells (`23,641` 4-Role Ordinary + `3,038` Excluded) Across 7 Splits (`120.25s` Guarded) | Policy A (`-161` $\to$ `15,228` / `23,480`), **Policy B Gate-Disjoint (`-238` $\to$ `15,151` / `23,403`, `0` `st_val_gate_32` overlap)**, **Policy C Maximal-Disjoint (`-273` $\to$ `15,116` / `23,368`, `0` `eval_phys_set` overlap)**; `64D` SVD `94.75%` train / `88.78%` OOS variance. | Verified Gate-Disjoint & Maximal-Disjoint |
| **`H41`–`H44` Live** | 4-Arm `2+1+1` (`0` Optimizer Steps; `SPQ-1` Tilt + `93D` `QHat` `train_spearman_rho=+0.3902`) | `G1C` `holdout30_verify`: `COMPLETE` `+0.028847` (`2W/15T/3L`), `ANALYTIC_ZERO_CKPT` `+0.028847` (`2W/15T/3L`), `FROZEN` `-0.000851` (`1W/16T/3L`), **`UNGUIDED` `+0.035839` (`3W/14T/3L`)**; `max_conditional_base_kl = 1.148685 > 0.35`. | **UNQUALIFIED** (`KL > 0.35` & Below `UNGUIDED`) |
| **`H45` Cluster** | 4-Host (`xbabe1`, `xbabe2` paused for CI lock, `xbabe3`, `xbabe4` = `338` CPUs) Actor-Learner Ratchet | `2` generations (`Gen 1: 7/8/5`, `Gen 2: 12/4/4`); both generations returned **`REJECTED_RETAIN_INCUMBENT_GENERATION`** on Criterion 3 (`KL > 0.35`) and non-superiority vs. `UNGUIDED`. | **UNQUALIFIED / RATCHET REJECTED** |
