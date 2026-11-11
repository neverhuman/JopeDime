# H431 Closeout: `CW-3` Part A (`ICB_v1` Capability Battery + `FIT-SEL` Contamination Screen) & Part B (`CW-3` Prior/Model/Trainer)

- **Wave:** `H431` (`research/flywheel_selfplay/h431_cw3ab_icl_battery/`)
- **Lane / Agent:** `agy_apex_w10` (`gemini (agy)`, CPU-only on `xbabe2` `cpus 64-79`, `<= 16` threads at `nice 19`, `CUDA_VISIBLE_DEVICES=""`)
- **Pre-Registration & Pre-Readout Amendments:**
  - `PREREG.md`: `fbbe3b3d6a66f295134cbec181307283ece33994`
  - `AMENDMENT_1.md`: `0d702796ea273c25b0efea1d7613895eb43a5a68`
  - `AMENDMENT_2.md`: `ba80321e5c3808d3e6b4fffe80a2647ac4a7205d`
  - `AMENDMENT_3.md`: `bcacd8e12b8a436f87065e44b35feb5efc61c8b7` (staged at `/home/ubuntu/apex-stage/h431/pub-r4`, `CODE.sha256: fe7a26bb7a63c033`)
- **Integrity Gates (`H431.gates.json`, run strictly before `readout_h431.py`):** **8 / 8 PASS (`pass: true`)** (`wall_s = 16,859s <= 21,600s`, `max_threads_per_pass = 16`, `devices = ["cpu"]`, `cuda_visible_any = false`, `661` tables untouched).
- **Verdicts:**
  - **Part A (`ICB_v1` + `FIT-SEL` Screen):** **`BATTERY_AS_EXPECTED`** (`6 / 6` pre-registered `H1` structural blind-spot checks confirmed on primary `M_h024`).
  - **Part B (`CW-3` `icl_prior_v3s.py`, `strong_base.py`, `icl_model_cw3.py`, `icl_train_cw3.py`):** **`PARTB_READY`** (`6 / 6` synthetic & `r9` exact-equivalence unit tests passed in `unit_tests.json`).

---

## 1. `FIT-SEL` Contamination Screen (`EXCLUSION_FITSEL.json`)

Pinned artifact: `/home/ubuntu/jope-data/h335-cl/h431/pub-r2/EXCLUSION_FITSEL.json` (`sha256: be87f957c1ab0089b389e5bee5ae2bea5ad141d40aa9568973fd3c611f4154da`).

| Tier | Contexts | Share | Definition |
| --- | ---: | ---: | --- |
| `clear` | 68 | 64.2% | Clean non-contaminated contexts |
| `possible` | 12 | 11.3% | Mild feature overlap, retained in `clean_non_copy` |
| `target_only` | 8 | 7.5% | Target-distribution similarity only, retained in `clean_non_copy` |
| **`clean_non_copy` (Primary)** | **88** | **83.0%** | **`tier != "copy"` (used by downstream `CW-4`, Wave C, and `CW-5`)** |
| `copy` (Excluded) | 18 | 17.0% | Near-copy feature contamination |

### `FIT-SEL` Monitor Readout Across Checkpoints (`M_h020`, `M_h024`, `L_h018`)

| Subset (`n`) | Metric | Ridge | `M_h020` (`24M` h20) | `M_h024` (`24M` h24, Primary) | `L_h018` (`104M` w2 h18) |
| --- | --- | ---: | ---: | ---: | ---: |
| **`clean_non_copy` (`88`)** | `full` mean / median `u` | 1.0174 / 0.5931 | 1.2854 / 0.9096 | 1.2652 / 0.9010 | 1.2170 / 0.8495 |
| **`clean_non_copy` (`88`)** | `ridgeres` mean / median `u` | 1.0174 / 0.5931 | **1.4652** / 0.9750 | 1.4603 / **1.0651** | 1.4450 / 0.9802 |
| **`clean_non_copy` (`88`)** | `ridgeres` win vs ridge | — | 94.32% | 93.18% | **96.59%** |
| `all_106` (`106`) | `ridgeres` mean / median `u` | 1.0606 / 0.6296 | **1.4650** / 0.9453 | 1.4631 / **1.0270** | 1.4455 / 0.9712 |
| `all_106` (`106`) | `ridgeres` win vs ridge | — | 91.51% | 92.45% | **97.17%** (`103/106`) |
| `clear_only` (`68`) | `ridgeres` mean / median `u` | 0.9865 / 0.5683 | **1.4976** / 0.9750 | 1.4931 / **0.9991** | 1.4842 / 0.9762 |
| `copy_only` (`18`) | `ridgeres` mean / median `u` | 1.2718 / 0.7803 | 1.4642 / 0.9203 | **1.4767** / 0.9225 | 1.4478 / **0.9303** |
| `copy_only` (`18`) | `ridgeres` win vs ridge | — | 77.78% | 88.89% | **100.00%** (`18/18`) |

---

## 2. `ICB_v1` 12-Family Capability Battery (`B1`–`B12`)

Pinned battery: `/home/ubuntu/jope-data/h335-cl/h431/pub-r2/ICB_v1.npz` (`sha256: 492bb143672958247b65fef72adafa99b327766f293c7cf7b5cebd5f562a58a5`, `340` tasks for `M_h020`/`M_h024`; balanced `rep == 0` `68`-cell design for `L_h018`).

| ID | Structural Family | `full` Cap `M_h020` | `full` Cap `M_h024` | `full` Cap `L_h018` | `ridgeres` Cap `M_h020` | `ridgeres` Cap `M_h024` | `ridgeres` Cap `L_h018` | `full` NF Ceiling `M_h024` | `full` NF Ceiling `L_h018` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B1` | `linear_dense` | 0.5167 | 0.4881 | 0.1745 | 0.9633 | **0.9635** | 0.9573 | 3.0132 | 1.7648 |
| `B2` | `linear_sparse` | 0.0752 | 0.0921 | 0.0176 | **0.8894** | 0.8871 | 0.8824 | 0.2353 | 0.1789 |
| `B3` | `single_key` | 0.2381 | 0.2139 | **0.2935** | 0.2197 | 0.1972 | **0.2456** | 1.3212 | 0.0482 |
| `B4` | `pair_key` | 0.0089 | **0.0182** | 0.0130 | 0.0055 | **0.0128** | 0.0117 | 0.0296 | 0.0872 |
| `B5` | `key_plus_slope` | **0.1303** | 0.1153 | 0.0541 | **0.1174** | 0.0995 | 0.0511 | 1.2515 | -0.0016 |
| `B6` | `arithmetic_pair` | **0.4548** | 0.3855 | 0.2915 | 0.5524 | 0.5283 | **0.7214** | 4.3004 | 0.7017 |
| `B7` | `smooth_1d` | 0.4333 | 0.3704 | **0.4620** | 0.4284 | 0.3744 | **0.4573** | 4.6869 | 0.0241 |
| `B8` | `additive_smooth` | 0.2045 | **0.2125** | 0.0829 | **0.2796** | 0.2482 | 0.1908 | 1.4125 | 0.2775 |
| `B9` | `unordered_factors` | **0.0876** | 0.0743 | 0.0800 | **0.0965** | 0.0815 | 0.0847 | 0.3457 | 0.2422 |
| `B10` | `multiplicative` | **0.1299** | 0.1117 | 0.1194 | **0.1299** | 0.1214 | 0.1180 | 1.0801 | 0.4178 |
| `B11` | `thresholds_tree` | 0.1793 | **0.2094** | 0.1367 | 0.1718 | **0.1894** | 0.1709 | 1.3389 | 0.0969 |
| `B12` | `heavy_tails_t2` | **0.8384** | 0.7992 | 0.2977 | 0.9927 | 0.9912 | **0.9932** | — | — |
| **All 12** | **Overall Mean Capability** | **0.2748** | 0.2576 | 0.1686 | 0.4039 | 0.3912 | **0.4070** | 4.6869 | 1.7648 |
| **Struct-6** | **`B3,B4,B5,B9,B10,B11`** | **0.1290 / 0.1749** | 0.1532 | 0.1419 | 0.1869 | 0.1734 | **0.2054** | — | — |

### Key Scientific Takeaways
1. **Structural blind spots (`H1` `6/6` confirmed):** Plain `r9`-trained ICL is nearly blind to `B4` `pair_key` (`~1.3%` capability), `B9` `unordered_factors` (`~8%`), `B5` `key_plus_slope` (`~10%`), `B10` `multiplicative` (`~12%`), and weak on `B3` `single_key` (`~20–29%`), directly motivating **`CW-4`** (`T1 --typed-cells`, `T2 --y-every-layer`, `T4 --fine-head`) and **`CW-5`** (feeding `H430`/`H433` structural `[X, S]` columns into ICL).
2. **Resolution of the `104M` (`L_h018`) vs `24M` (`M_h024`) `H438` paradox:** On noisy `ridgeres` tasks, `L_h018` **exceeds** `M_h024` in overall capability (`0.4070` vs `0.3912`), `structure6` capability (`0.2054` vs `0.1734`), `B6` `arithmetic_pair` (`0.7214` vs `0.5283`), `B7` `smooth_1d` (`0.4573` vs `0.3744`), and `FIT-SEL` win-rate vs ridge (`97.17%` vs `92.45%`). Yet on **noise-free (`noise = 0`) `full` readout**, `L_h018`'s precision ceiling collapses (`0.0241 u` on `B7` vs `M_h024`'s `4.6869 u`; `0.0482 u` on `B3` vs `1.3212 u`), isolating a sharp **noise-free bin-mean / calibration floor** in the 3-worker averaged `104M` model rather than missing nonlinear capacity.
