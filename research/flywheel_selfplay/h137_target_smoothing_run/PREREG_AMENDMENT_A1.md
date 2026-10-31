# H137 Prospective Preregistration Amendment (`Amendment A1` over `H121`)

- **Wave**: `H137` (`research/flywheel_selfplay/h137_target_smoothing_run/`)
- **Parent Preregistration**: `H121` (`research/flywheel_selfplay/h121_target_smoothing/PREREG.md`, `sha256 = 127e0481f06c0d9b42690b3091138b5a927af9fa7b90f10fbf88a80735e6b923`; `runner.py`, `sha256 = 36ef51730ebe9db21a59455c99fcb15580252c92bf296f2618911c1dc1624aec`)
- **Timestamp (UTC)**: `2026-10-31T22:40:00Z` (committed and merged **before** any `H137` scoring or training execution)

## Prospective Changes from `H121` Preparation to `H137` Scored CPU Execution

1. **Fresh Fold Seed & Paired Seeds**:
   - `H121` specified `fold_seed = 12100` and paired seeds `45, 46, 47`.
   - `H137` prospectively assigns unused `fold_seed = 137171` (`fold_manifest_137171.json`, `sha256 = a5ef0a687c26ab0da527bd9bfd54ce4d831b81049493abc77a924b6ea8d97af9`: `fold 0` = `41` assessment families / `212` contexts; `folds 1..4` = `164` fit families / `1,007` contexts) and unused paired seeds `171, 172, 173` (`bootstrap_seed = 137171`), audited before launch to have zero prior occurrences in git or `/home/ubuntu/jope_plan/ns/receipts/`.
2. **CPU-Only Execution & Pause-on-Load Guard**:
   - Executes on CPU only (`CUDA_VISIBLE_DEVICES=""`, `nice 19`, `<= 4` threads, `0.0` GPU seconds, no `qualification.lock` / FIFO GPU ticket) using the locked custody runner `/home/ubuntu/jope-rescue-20260921/h137-target-smoothing-run/run_h137_cpu.py` (`chmod 0444`, `sha256 = 23f221180129940a800ec5a0c275d7bcb03948a364d9f7783b3655cd0274ae70`), which directly imports and calls `h121.smooth_targets` and `h121.mean_regret` from `research/flywheel_selfplay/h121_target_smoothing/runner.py`.
   - Replaces host-load abort with a non-aborting pause guard (`while load1 > 86.7: time.sleep(15.0)`).
3. **Unchanged Scientific Contract**:
   - Arms (`CONTROL` $\epsilon=0.0$, `TS-0.1` $\epsilon=0.1$, `TS-0.3` $\epsilon=0.3$), `40` AdamW updates per cell (`360` total updates), adapter-only trainable inventory (`29,211` params / `47` tensors), frozen `DOPE` (`c51a56d0...`), and admission gates (`median_seed_gain > +0.001`, `Holm LCB95 > +0.001`, `harm <= 10%`, `3/3` seeds `> 0`, `seeds_not_identical`) are unchanged.
