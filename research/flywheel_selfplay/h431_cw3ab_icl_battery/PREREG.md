# H431 — CW-3 Parts A & B: ICL Capability Battery, FIT-SEL Contamination Screen, and Structure-Mined `v3s` Prior (`PREREG`)

| | |
| --- | --- |
| **Wave** | `h431_cw3ab_icl_battery` (implements `CW3_CURRICULUM.md` Parts A & B and §5 `FIT-SEL` screen) |
| **Owner** | `agy_apex_w10` (Gemini), coordinated with `cl_fwriter` (`H430`) and `cl_apex` |
| **Compute** | CPU-only on `xbabe2` (`<= 16` single-thread processes at `nice 19`, `CUDA_VISIBLE_DEVICES=""`), hard cap `4` hours. Zero GPU |
| **Cohorts touched** | Synthetic `ICB_v1` (`340` tasks) + `FIT-ICL` / `FIT-SEL` (`screen_h425.py` + `fitsel_search_val.npz` monitor). **Zero `DEV`, `CONFIRM`, `P201b`, `P201e`, `dev91`, `DL-Val`, or `DL-Holdout` access** |

---

## 1. Question and Hypotheses

H424's second read (`h024` vs `h020`: `+0.0008 u`, `p = 0.23`) and H429 (`WIDE_NULL`: `-0.0003 u`, `p = 0.79`, ICL column still `-1.43 u` behind stored candidates out of fold) show that training volume and wide-table subset bagging alone do not close the in-context transformer's gap to tabular structure. Meanwhile, `cl_fwriter`'s TRAIN `P201b` probe (`EVIDENCE_structure_probe.md`) found that the largest foundation-model gaps come from exact-key lookup tables, two-column raw arithmetic, and unordered factor/multiplicative structure — none of which exist in `icl_prior.py` (`v1`/`v2`).

- **H1 (Missing-structure hypothesis, Part A):** Evaluated on the 12-family synthetic ICL Capability Battery (`ICB_v1`, `340` tasks with known structure-aware reference predictors), our stored `24M` (`h020`, `h024`) and `104M` (`L_w2_h018`) checkpoints have capability `< 0.30` under the `full` readout on `B3` (single key), `B4` (pair key), `B5` (key + slope), `B9` (unordered factors), and `B10` (multiplicative), and `< 0.70` on noise-free `B1` (dense linear) under `full` readout.
- **H2 (Contamination inflation on `FIT-SEL`, §5):** Running H425's published `screen_h425.py` on `FIT-SEL` against `FIT-ICL` identifies exact/row copies (`tier == "copy"`) whose monitor `u` inflates unrestricted `FIT-SEL` medians relative to clean (`tier != "copy"` and `tier == "clear"`) contexts.
- **H3 (Part B readiness):** Adding the five missing task families (`key`, `keylin`, `cat`, `arith`, `mult` under `--prior v3s`) plus default-off flags for Factors B–E (`--aug-cell-drop`, `--aug-jitter`, `--dropout`, `--replay`, `--p-resid-strong`, `--consist`, and `strong_base.py`) generates finite, high-signal tasks while preserving **byte-for-byte identity** with `h419/r9` when all new flags are off.

---

## 2. Pinned Inputs (SHA-256)

1. **`M_h020` (`24M`, hour 20):** `/home/ubuntu/jope-data/h424/ICL_M_r3_ckpt_h020.pt`
   `e3c8fd9972ce0c1684630ff06b159e950f6f498c4461d0454e0c6a6f44d38dd7`
2. **`M_h024` (`24M`, hour 24):** `/home/ubuntu/jope-data/h424/ICL_M_r3_ckpt_h024.pt`
   `7fbca0e702bee00b4df0a58fac4062f550ceb71e9319fbbed5356a9b326ef176`
3. **`L_h018` (`104M`, latest monitor checkpoint):** `/home/ubuntu/jope-data/apex-icl/diag/mon/L_w2_h018.pt`
   `153117c5fd76dde38fc3d4d7de0997dd9f4bb92ba4eef42bf741ff7530124121`
4. **`FIT-SEL` SEARCH-row monitor bank (`106` contexts):** `/home/ubuntu/jope-data/apex-icl/bank/fitsel_search_val.npz`
   `8e517921b7d5bb46c223a12262e7e0c64531db3fad7dede601d002a09840bba0`
5. **Base stage:** `/home/ubuntu/apex-stage/h427/pub-r1` (`screen_h425.py`, `sandbox_h335_docker.sh`) + `/home/ubuntu/apex-stage/h419/r9` (`icl_diag.py`, `icl_resid.py`, `icl_inject.py` verified byte-identical to `origin/v1/next:research/flywheel_selfplay/h419_apex_icl/`).

---

## 3. Method

### 3.1 Step 1: `FIT-SEL` Contamination Screen (`SCREEN_FITSEL.json` + `EXCLUSION_FITSEL.json`)
Run H425's published `screen_h425.py --queries FIT-SEL` inside `sandbox_h335_docker.sh` (`661` tables in `apex/tables`: `FIT-ICL` `440`, `FIT-SEL` `106`, `DEV` `115`; `0` `CONFIRM` tables) and cut `EXCLUSION_FITSEL.json` using H425's exact tier rules (`copy`, `possible`, `target_only`, `clear`). Clean `FIT-SEL` for `CW-3` is `tier != "copy"` (with `tier == "clear"` reported alongside as strict sensitivity).

### 3.2 Step 2: 12-Family ICL Capability Battery (`ICB_v1.npz`, `340` Tasks)
Fixed seed `431_001`. Context sizes `n_ctx in {256, 1024}`, query rows `n_q = 256`. Noise levels `sigma in {0.0, 0.05, 0.20}` of signal standard deviation for `B1`–`B11` (`5` tasks per cell = `30` tasks/family) and `0.20` `t(2)` noise for `B12` (`5` tasks per `n_ctx` = `10` tasks); `340` tasks total. Reference predictors are fitted strictly on context rows `[:n_ctx]` (except `B12`, whose reference is the noise-free true linear signal `X_q beta` per `CW3_CURRICULUM.md` §3):

| Family | Name | Features | Generator | Context-Only Reference Predictor |
| --- | --- | --- | --- | --- |
| `B1` | Linear | 16 | Dense `y = X beta` | Ordinary least squares (ridge `1e-8` tie-break) |
| `B2` | Sparse linear | 64 | `5` active linear features + `59` distractors | OLS on the true `5` columns |
| `B3` | Single key | 16 | `y = e_g`, `16` or `128` groups, random integer codes | Per-group context mean (global mean backoff) |
| `B4` | Pair key | 16 | `y = e_{g,h}`, two columns (`8` or `12` levels), random codes | Pair-group context mean (additive/global backoff) |
| `B5` | Key + slope | 16 | `y = a_g + b_g x_s`, `16` groups | Per-group OLS `[1, x_s]` on context rows |
| `B6` | Arithmetic | 16 | `y = h(x_a op x_b)`, `op in {-, *, /}`, `h in {id, tanh, sq}` | 64-knot penalized spline smoother on true `x_a op x_b` |
| `B7` | Smooth 1D | 16 | Smooth random Fourier function `f(x_j)` + `15` distractors | 64-knot penalized spline smoother on `x_j` |
| `B8` | Additive smooth | 16 | Sum of `4` smooth univariate functions + `12` distractors | Joint 4-feature penalized spline basis on true `4` |
| `B9` | Unordered factors | 16 | `6` factor columns (`3..12` levels, random integer codes) | One-hot penalized least squares on true `6` factors |
| `B10` | Multiplicative | 16 | `y = exp(4 factor effects + 0.5 x_s)` (log-normal noise) | One-hot + `x_s` OLS on `ln y` with Duan smearing |
| `B11` | Thresholds | 16 | Depth-3 tree (`8` leaves) on `3` features | Context mean per true leaf |
| `B12` | Heavy tails | 16 | `B1` with `t(2)` noise at `0.20 * sd(signal)` | True noise-free signal `X_q beta` |

Both `full` and `ridgeres` readouts (`icl_diag.run_variant`, EMA weights, query targets masked to zero) are scored on `M_h020`, `M_h024`, and `L_h018`:
- `u8 = clip(-ln(RMSE / RMSE_mean), -2.0, 8.0)` (`CW-3` §3 cap at `8`).
- `capability = mean(clip(u8_model / max(u8_ref, 1e-6), 0.0, 1.2))` per family and `(n_ctx, noise)` cell.
- `precision_ceiling = max_{noise == 0}(u8_model)` per family and overall.
- `delta_vs_ridge = mean(u8_model - u8_ridge)` per family.
- Monitor diagnostic: `full` and `ridgeres` scored on `fitsel_search_val.npz` (`106` contexts) for all three checkpoints, reporting `all (106)`, `clean non-copy`, `clear`, and `copy` subsets.

### 3.3 Step 3: Part B (`--prior v3s` + Factor B–E Trainer Extensions)
Staged and verified on CPU (`icl_prior_v3s.py`, `icl_model_cw3.py`, `strong_base.py`, `icl_train_cw3.py`, `test_cw3.py`):
- Five new synthetic families (`key`, `keylin`, `cat`, `arith`, `mult`) with `PROBS_V3S = 0.75 * PROBS_V2 + (0.07, 0.03, 0.06, 0.06, 0.03)` and optional census-share override.
- Default-off trainer flags for `CW-3` Part C: `--prior v3s`, `--aug-cell-drop`, `--aug-jitter`, `--dropout`, `--replay` (seeded deterministic batch regeneration), `--p-resid-strong` (`strong_base.py` 16-bin boosted stumps), and `--consist` (symmetric KL across two random slot assignments).

---

## 4. Pre-Registered Verdicts and Integrity Gates

1. **Primary Part A Verdict (`M_h024`, primary checkpoint of the `24M` series; `M_h020` and `L_h018` reported alongside):**
   - **`BATTERY_AS_EXPECTED`** if under `full` readout:
     (a) mean capability `< 0.30` on each of `B3`, `B4`, `B5`, `B9`, `B10`, **and**
     (b) mean capability `< 0.70` on noise-free `B1` (`noise == 0`).
   - **`BATTERY_PARTIAL_SURPRISE`** if at least 4 of the 6 conditions hold and mean capability across `{B3, B4, B5, B9, B10}` is `< 0.30` (recording the exact family/cell exception).
   - **`BATTERY_SURPRISE`** otherwise (triggering Part B redesign before any Part C GPU run).
2. **Part B Readiness Gate:** **`PARTB_READY`** iff all 6 synthetic, zero-leakage, finite-draw, and `r9` byte-for-byte equivalence checks in `test_cw3.py` pass (`PARTB_FAIL` otherwise).
3. **Automated Integrity Gates (`gates_h431.sh` -> `H431.gates.json`):**
   - `g1_prereg_before_output`: `PREREG.md` + code committed to `origin/v1/next` strictly before `SCREEN_FITSEL.json`, `ICB_v1.npz`, and any battery/monitor score file, and before `readout_h431.py` runs.
   - `g2_pinned_inputs`: all 3 checkpoint SHA-256s and `fitsel_search_val.npz` SHA-256 match §2.
   - `g3_cpu_only_16_threads`: `device == "cpu"`, `cuda_visible == false`, total concurrent worker threads `<= 16`.
   - `g4_four_hour_cap`: total wall time `<= 14,400` s.
   - `g5_battery_and_screen_complete`: `340/340` battery tasks and `106/106` `FIT-SEL` contexts scored on all 3 checkpoints, `SCREEN_FITSEL.json` covers `106` `FIT-SEL` contexts, `0` failed tasks.
   - `g6_partb_equivalence`: `test_cw3.py` passes all 6 checks (`r9` byte-for-byte match verified).
   - `g7_confirm_untouched`: `/home/ubuntu/jope-data/h335-cl/apex/tables` count equals `661` (`0` `CONFIRM` tables).
   - `g8_pub_stage`: executed from read-only `pub` stage `/home/ubuntu/apex-stage/h431/pub-r1` matching `CODE.sha256`.
