# H137: Target Smoothing (`TS-0.1` $\epsilon=0.1$, `TS-0.3` $\epsilon=0.3$) vs `CONTROL` and `r882`

- **Wave ID**: `H137`
- **Owner**: `rd_argon` (`[ARGON]`)
- **Status**: `PREREGISTERED_NOT_EXECUTED` (`optimizer_updates = 0`, `dl_val_reads = 0`, `gpu_seconds = 0.0`)
- **Lineage**: Scored CPU execution of the merged `H121` preregistration (`research/flywheel_selfplay/h121_target_smoothing/PREREG.md`, `sha256 = 127e0481f06c0d9b42690b3091138b5a927af9fa7b90f10fbf88a80735e6b923`; `runner.py`, `sha256 = 36ef51730ebe9db21a59455c99fcb15580252c92bf296f2618911c1dc1624aec`) with prospective fresh `fold_seed = 137171` and fresh seeds `171, 172, 173` ([PREREG_AMENDMENT_A1.md](PREREG_AMENDMENT_A1.md)).
- **Documents**:
  - [PREREG.md](PREREG.md)
  - [PREREG_AMENDMENT_A1.md](PREREG_AMENDMENT_A1.md)
  - [fold_manifest_137171.json](fold_manifest_137171.json) (`sha256 = a5ef0a687c26ab0da527bd9bfd54ce4d831b81049493abc77a924b6ea8d97af9`: `fold 0` = `41` assessment families / `212` contexts; `folds 1..4` = `164` fit families / `1,007` contexts)
  - [manifest.json](manifest.json)
  - [metrics.json](metrics.json)
- **Locked Custody Runner**: `/home/ubuntu/jope-rescue-20260921/h137-target-smoothing-run/run_h137_cpu.py` (`chmod 0444`, `sha256 = 23f221180129940a800ec5a0c275d7bcb03948a364d9f7783b3655cd0274ae70`, `--dry-run` `PASS`)
