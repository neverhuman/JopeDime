# H450 Fleet Extensions (`research/flywheel_selfplay/h450_fleet_agy/`)

Additive, non-overlapping H450 fleet components coordinated with `cl_fwriter` via
`/home/ubuntu/jope_plan/H450_CLAUDE_GEMINI_COORD.md` and `/home/ubuntu/jope_plan/H450_FLEET_COORDINATION.md`
(staged read-only under `/home/ubuntu/h450-stage/fleet-r1`):

1. `mmap_bank_h450.py` — Zero-copy `np.memmap` drop-in replacement (`MmapF64Bank`) for `CombinedF64Bank`
   (`< 0.5 s` init, `< 80 MiB` private RSS), plus `--cmd convert` and `--cmd verify-parity` (verified bit-identical
   across all `3,525` tables / `164,651,987` IEEE-754 `uint64` cells against `everything_combined_f64_h450_bank.npz`
   `9dfc60ed...`).
2. `icl_model_mps_h450.py` — Exact `state_dict`-compatible `ICLH450MPS` supporting Apple Silicon `mps` (`xbabe4`
   M3 Ultra 80-core GPU) with exact CPU `float64` context normalization and bin interpolation + Metal `bfloat16`/`float32`
   transformer execution.
3. `train_mps_104m_h450.py` — `104M` (`ICL_L`) trainer for `xbabe4` MPS (and `cuda`/`cpu`) using `mp.get_context("spawn")`,
   `MmapF64Bank`, `HostCpuGovernor` (`<= 82%` hard cap -> `<= 85%` host CPU average), bidirectional synthetic feature
   injection (to & from features even on `F < 10`), and pre-forward `128-bit` `blake2b` deduplication (`0` repeats).
4. `fleet_telemetry_and_xbabe1_eval.py` — 5-host live telemetry & unique-dataset union tracker + `xbabe1` CPU
   every-`25,000`-dataset frozen-split `dev91-audit-v1` Elo evaluator (`<= 75%` CPU on `xbabe1`, `ZImage` GPU untouched).
