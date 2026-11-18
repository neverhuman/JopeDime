# DOPE value-probe contract

Status: contract active; scored execution in [`../h470_dope_value_probe/`](../h470_dope_value_probe/).
Sealed and holdout cohorts remain strictly **NOT_RUN**.

This directory records the released 629 layout, the 196D `R_packet_v1` layout,
the arm widths, contrasts `C0`–`C8`, and pure-Python deterministic paired
family bootstrap and Holm–Bonferroni statistical helpers. This contract module
keeps zero external imports so it remains completely off the product path.
`crates/jope-encoders`, `crates/jope-service`, and `dope_freeze_guard.py` are
unchanged.

## What is fixed here

- Released `dope629` blocks, and the Dope512 prefix of the first 512 coordinates.
  Dope512 remains the separate product recipe
  `dope-mlcv3-high-f32-descriptor256-task128-dist128/v1`.
- Scale-invariant, extrema-free `R_packet_v1` (`196D`): `14` router gates (with
  `noise_scale` once) + `14` normalized task/count/residual globals (zero
  `y_min`/`y_max`) + `16` copula/regime/`w_basis` norm shares + `96` `(mean,
  std, max)` symmetric column-packet pool + `32` activity-weighted column pool +
  `24` tree/pair/latent stats, alongside `R_packet_notarget` (`196D`) zeroing
  target-bundle slots (`P-S2`).
- Fine landmarks masked at level `high`: `gbdt`, `knn1`, `knn16`, `net`,
  `gbdt_quarter`.
- Arm widths for `R_none` (0), `R_shape` (3), `lm48_z` (48), `R_lm80` (80),
  `R_desc` (290), `R_learned` (`dope629[256:]`, 373), `R_dope` (663),
  `R_hidden` (256), `R_tok` (320), `R_randrouter` (663), `R_shuffled` (663),
  `R_noise` (663), `R_desc_ultra` (290), `R_dope_ultra` (663), `R_packet` (24),
  `R_packet_v1` (196), `R_packet_notarget` (196), `R_dope_plus_packet_v1` (859),
  and `B03_svd96` (96).
- Ceiling and zero-retraining upgrade arms: `profile_oracle`, `R_oracle64`,
  `WP3_pred_reveal`, `B03_svd96`, `B03_ultra_packet`.
- Contrasts `C0`–`C8`, deterministic 64-bit SplitMix64
  `paired_family_bootstrap_ci` (`n_boot=10000`, `seed=20261118`), and step-down
  `holm_bonferroni`.

Run the contract with:

```sh
python3 -B research/flywheel_selfplay/dope_value_probe/test_dope_value_probe.py
```
