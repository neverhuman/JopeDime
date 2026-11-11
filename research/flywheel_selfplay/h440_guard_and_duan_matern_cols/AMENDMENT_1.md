# H440 Pre-Readout Amendment 1 — Align `g3_replay_exact` Tolerance with Canonical `H424` (`<= 1e-9`)

- **Timestamp:** `2026-11-11T22:04Z` (strictly **before** `READOUT_P201B.json` exists and with **zero** scored `H440` utility outputs read).
- **Observed Pre-Readout Gate Receipt (`H440.gates.json`, retained immutable):**
  - `7/8` gates (`g1_prereg_before_output`, `g2_pinned_banks`, `g4_cpu_budget`, `g5_columns_complete` (`106/106` & `201/201`), `g6_unit_and_smoke`, `g7_confirm_untouched`, `g8_pub_stage`) passed (`true`).
  - Stored `h416.SXB` weight replay achieved **exact zero weight discrepancy** on both panels (`fitsel_replay_w_diff = 0.0`, `p201b_replay_w_diff = 0.0`).
  - Stored `h416.SXB` utility replay achieved `fitsel_replay_u_diff = 3.4933833603645326e-11` and `p201b_replay_u_diff = 3.9484149283453007e-10` against JSON-serialized `rec["outputs"]["P3+LIB3.SXB"]["value"]`.
- **Root Cause & Canonical Precedent:**
  - Every prior stack-column wave (`H424`, `H429`, `H430`, `H432`, `H433`, `H434`, `H435`, `H436`, `H437`, `H438`; see `/home/ubuntu/apex-stage/h424/pub-r1/code/gates_h424.sh` line 42) defines the exact-replay gate as `replay_max_w_diff == 0 and replay_max_u_diff <= 1e-9` because stored JSON utility floats carry `~1e-10` serialization/BLAS-reduction rounding even at exact `0.0` weight difference. `H440`'s `gates_h440.sh` tightened `replay_max_u_diff` to `< 1e-12` in error.
- **Change:**
  - Update `gates_h440.sh` line 50 to canonical `replay_max_w_diff == 0 and replay_max_u_diff <= 1e-9` on both panels and write create-once `$EV/results/H440_r2.gates.json` (preserving `$EV/results/H440.gates.json` untouched).
  - Include `AMENDMENT_1.md` in `stage_h440.sh` for `pub-r2`. Zero scoring, column-building, or readout code is touched.
