# H136 Execution & Post-Processing Disclosure (`Amendment A1`)

- **Wave**: `H136` (`research/flywheel_selfplay/h136_fambal_cosine_run/`)
- **Owner**: `rd_argon` (`[ARGON]`)
- **Timestamp (UTC)**: `2026-10-31T22:23:00Z`
- **Immutable Preregistration Document**: `PREREG.md` (`sha256 = a9bbce83d944ab1fab9d88a96b82ea01207fc093f85a08e7cfef117dfda1aa74`, untouched since PR [#1199](https://github.com/neverhuman/JopeDime/pull/1199))
- **Immutable Fold Manifest**: `fold_manifest_136161.json` (`sha256 = 6ff1a5c48564a2b13b83f3b50403bfa113f8e1d0361bd5f1c3cb3c83ce872496`, untouched since PR [#1199](https://github.com/neverhuman/JopeDime/pull/1199))

## 1. Pre-Forward-Pass Loader Fix (`Attempt 1` -> `Attempt 2`, Zero Forward Passes or Optimizer Updates in `Attempt 1`)

1. **Attempt 1 (`run_h136_cpu.py` `sha256 = a8fa2908812964cfd5f33207a5bef2e3a2b5a9c2e23558888cd0e45e5b13e001`)**:
   - Launched at `2026-10-31T21:50:37Z` after PR [#1199](https://github.com/neverhuman/JopeDime/pull/1199) merged (`7b368755462c0a2caac99b8779f5c26e2a65b295`).
   - Exited in Step 1 (`unchanged-r882 Regression-First Gate`) at `h55.build_model(INPUT_DIR, ckpt_a, device, h48)` tuple unpacking (`(net, params, guard, state)` vs `(net, guard, dope, adapter)`) **before any model forward pass, fold-0 prediction, utility evaluation, or optimizer update occurred** (`0` forward passes, `0` assessment reads, `0` cells, `0` optimizer updates).
   - Full log preserved in local custody at `/home/ubuntu/jope-rescue-20260921/h136-fambal-cosine-run/attempt1_execute_wave.log`.
2. **Pre-Forward-Pass Loader Fix (`run_h136_cpu.py` `sha256 = b0eda24a960f31c52fc308f5b667f1b76c9a6a55dd0ce031898beeda4fc21072`, locked `chmod 0444`)**:
   - Aligned Step-1 model/token loader calls (`h55.build_model` 4-tuple unpack + `h48.h46.checkpoint_guard` + `h48.make_predictor` + `torch.Tensor` token/mask loading) with `run_h132_cpu.py`, verified `--dry-run` (`PASS`), re-locked `run_h136_cpu.py` read-only (`chmod 0444`), and launched `Attempt 2` (`--execute-wave`) at `2026-10-31T21:55:31Z` (log preserved at `/home/ubuntu/jope-rescue-20260921/h136-fambal-cosine-run/attempt2_execute_wave.log`).
   - Per the directive (`Two PRs only: Prereg READY -> execute -> Closeout READY`), no intermediate 3rd PR was opened; this disclosure is included in the Closeout PR while keeping `PREREG.md` and `fold_manifest_136161.json` byte-for-byte unchanged.

## 2. Read-Only Step-3 Summary Finalizer (`finalize_h136_summary.py`, Zero Cell Re-Executions)

- During `Attempt 2`, Step 1 (`regression_receipt.json`, `sha256 = e30990a5e8b5e887b6b96d433152f2d43ad14f358b066644ed4964de7f5bec73`) and **all 9 training cells** (`360` AdamW updates across `CONTROL`, `FAMBAL_ACC4`, `COSWR` $\times$ seeds `161, 162, 163`) completed and wrote their immutable per-cell receipts (`cells/{arm}_seed{seed}.json`).
- At the end of Step 3 (post-cell summary aggregation), `run_h136_cpu.py` called `elo_mod.fit_bradley_terry(...)` instead of `elo_mod.compute_family_grouped_bradley_terry_elo(...)`.
- Rather than editing `run_h136_cpu.py` after the 9 cells ran or re-running any cell, `run_h136_cpu.py` (`b0eda24a960f31c52fc308f5b667f1b76c9a6a55dd0ce031898beeda4fc21072`, `chmod 0444`) and all 9 cell receipts were left untouched, and a read-only Step-3 summary script (`/home/ubuntu/jope-rescue-20260921/h136-fambal-cosine-run/finalize_h136_summary.py`, `chmod 0444`, `sha256 = 92e0b42540426c4cdeb8bcf6e0407c057040f74126f94b74e22510e4a16b1e1f`, `0` optimizer updates) computed the preregistered 10,000-draw paired family-cluster bootstrap, 2-arm Holm step-down, continuous mean-regret co-readout, and `compute_family_grouped_bradley_terry_elo` directly from the 9 immutable cell receipts to produce `h136_summary.json` (`sha256 = 7a8d7d56751b7ef6f3879313e40bdc84c33df47e92ab9399511862708ae33bb5`) and `metrics.json`.
