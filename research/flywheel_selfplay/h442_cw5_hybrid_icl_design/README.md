# H442 (`CW-5`) — Hybrid `[X, S] -> 24M ICL` Design Wave Closeout

- **Wave:** `H442` (`h442_cw5_hybrid_icl_design`, `CW-5`)
- **Pre-Registration Commit:** `eda6dff070ad737a991e2eddc5b98d78ed2a6638` on `origin/v1/next`
- **Published Stage:** `/home/ubuntu/apex-stage/h442/pub-r1` (`CODE.sha256` prefix `94985c6c0253387f`, `mode = pub`)
- **Pre-Readout Peer Review:** Reviewed by `@cl_fwriter` at commit `eda6dff070ad737a991e2eddc5b98d78ed2a6638` with **`NO BLOCKING FINDING`** (`/home/ubuntu/jope_plan/evidence/CW5_review/REVIEW_H442_eda6dff070_20261111T2150Z.md`)
- **Design Verdict (`FIT-SEL` Design Panel Only, Zero Confirmation Claim):** **`CW5_HYBRID_READY_FOR_JOINT_DEV`** (`FROZEN.json`: `selected_recipe = "R4"`, `promote_to_joint_dev = true`, `clean-88` clipped gain `+0.0431 u`, unclipped `+0.0941 u`, `p = 0.01491`)
- **Holdout Hygiene:** `APEX-DEV`, `APEX-CONFIRM`, `dev91-audit-v1`, `DL-Val`, `DL-Holdout`, and `sealed` untouched (`0` opens; `g6_confirm_untouched = true`, `661` sandbox tables)

---

## 1. Pre-Readout Integrity Gates (`H442.gates.json`: `8/8` PASS)

`gates_h442.sh eda6dff070ad737a991e2eddc5b98d78ed2a6638` ran strictly prior to `eval_cw5_fitsel.py readout`:

| Gate | Status | Verified Check |
| --- | --- | --- |
| `g1_prereg_before_output` | `true` | `prereg_commit_time = 1794433276 < first_output_time = 1794435981`, `READOUT_CW5.json` & `FROZEN.json` absent |
| `g2_pinned_inputs` | `true` | `ICL_M_r3_ckpt_h020.pt` (`e3c8fd99...`), `fitsel_search_val.npz` (`8e517921...`) across all `16/16` shards |
| `g3_cpu_only_16_threads` | `true` | `device == ["cpu"]`, `cuda_visible == false`, `total_threads == 16`, `wall_s = 12,524 <= 14,400` (`total_cpu_s = 159,511.16`) |
| `g4_complete_106` | `true` | `106 / 106` `FIT-SEL` contexts `ok == true` |
| `g5_unit_tests` | `true` | `test_cw5.py` `6/6` unit tests passed (`all_passed == true`, including query-invariance & leakage checks) |
| `g6_confirm_untouched` | `true` | `/home/ubuntu/jope-data/h335-cl/apex/tables` holds `661` files (`0` `APEX-CONFIRM` / `dev91` opens) |
| `g7_pub_stage` | `true` | `/home/ubuntu/apex-stage/h442/pub-r1/CODE.sha256` verified (`mode == "pub"`) |
| `g8_r0_replays_h431_m20` | `true` | Across all `73` narrow (`F <= 30`) `FIT-SEL` contexts, `R0` matches `H431` `mon_M_h020` `ridgeres` with `max_rel_diff == 0.0` |

---

## 2. Design Readout (`READOUT_CW5.json`)

> **Governance note (`PREREG` §2 & `REVIEW_H442_eda6dff070` note 2):** `FIT-SEL` is a design/selection panel already used in `H430`, `H431`, `H433`, and `H440`. Accordingly, `H442` makes **no confirmation claim**; `FROZEN.json` fixes at most **one** recipe (`R4`) prior to the single pre-registered joint `APEX-DEV` (`110`-cluster) evaluation.

### 2.1 Linear-Base & Residual Attribution Controls

Across the `88` clean non-copy `FIT-SEL` families (`70` with `d_S > 0`: `69` with exact-key columns, `28` with physical-formula columns):

| Contrast | `clean-88` Raw Gain (`u`) | `clean-88` Clipped (`u`) | `clean-88` `p` (`W/T/L`) | `clean_with_struct` (`n=70`) Raw (`u`) | `all-106` Raw (`u`) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ridge_XS` vs `ridge_X` | `+0.2343` (`1.2517` vs `1.0174`) | `+0.0983` | `0.0` (`56/19/13`) | `+0.2945` (`1.3662` vs `1.0716`) | `+0.2256` (`71/19/16`) |
| `R0` (`ridge(X) + ICL(X)`) vs `ridge_X` | `+0.4518` (`1.4692` vs `1.0174`) | `+0.2199` | `0.0` (`84/1/3`) | `+0.4455` (`67/0/3`) | `+0.4131` (`101/1/4`) |
| `R2` (`ridge([X,S]) + ICL([X,S])`) vs `ridge_XS` | **`+0.3008`** (`1.5525` vs `1.2517`) | **`+0.1755`** | **`0.0`** (`78/2/8`) | **`+0.2557`** (`61/1/8`) | **`+0.2777`** (`94/2/10`) |
| `R2` vs `ridge_X` | `+0.5351` (`1.5525` vs `1.0174`) | `+0.2301` | `0.0` (`78/1/9`) | `+0.5503` (`61/0/9`) | `+0.5032` (`94/1/11`) |

### 2.2 Hybrid Recipes `R1..R5` vs Baseline `R0` (`ridge(X) + ICL(X)`)

| Recipe | Description | `clean-88` Mean `u` | `clean-88` Raw Gain | `clean-88` Clipped Gain | `clean-88` `signflip_p` | `clean-88` `lofo_min` | `clean-88` `W/T/L` | `clean_with_struct` (`n=70`) Clipped (`p`) | `all-106` Clipped (`p`) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `R0` | `ridge(X) + ICL(X)` baseline | `1.4692` | — | — | — | — | — | `1.5171` (mean `u`) | `1.4737` (mean `u`) |
| `R1` | `ridge(X) + ICL([X, S])` | `1.4615` | `-0.0077` | `-0.0106` | `0.789` | `-0.0165` | `33/19/36` | `-0.0134` (`0.789`) | `-0.0035` (`0.618`) |
| `R2` | `ridge([X, S]) + ICL([X, S])` | `1.5525` | `+0.0833` | `+0.0335` | `0.0411` | `+0.0281` | `34/18/36` | `+0.0421` (`0.0408`) | `+0.0430` (`0.00634`) |
| `R3` | 2-stage residualized `ICL` on `[X, S]` | `1.5493` | `+0.0801` | `+0.0345` | `0.0373` | `+0.0291` | `34/19/35` | `+0.0433` (`0.0371`) | `+0.0505` (`0.00227`) |
| **`R4`** | **`ridge([X, S]) + 4-perm-bag ICL([X, S])` (`F+d_S <= 30`)** | **`1.5632`** | **`+0.0941`** | **`+0.0431`** | **`0.01491`** | **`+0.0378`** | **`41/18/29`** | **`+0.0542` (`0.0150`)** | **`+0.0517` (`0.00183`)** |
| `R5` | Context OOF ridge gate (`<= 0.995x`) `R2` vs `R0` | `1.5470` | `+0.0778` | `+0.0279` | `0.0713` | `+0.0225` | `20/42/26` | `+0.0351` (`0.0706`) | `+0.0340` (`0.0194`) |

### 2.3 Diagnostic Mechanism

1. **Why `R1` fails while `R2`/`R4` win:** Subtracting only `ridge(X)` (`R1`) leaves exact-key group jumps and raw-scale physical power/ratio trends in the target residual; rank-normalized `24M` ICL does not extrapolate raw scale jumps on its own (`-0.0106 u` clipped). Removing `ridge([X, S])` first (`R2`) centers the residual around the exact-key + physical-formula surface so `24M` `M_h020` ICL models nonlinear feature-key-formula interactions (`+0.3008 u` over `ridge_XS`, `+0.0833 u` unclipped / `+0.0335 u` clipped over `R0`).
2. **Why `R4` is the clean winner:** When `F + d_S <= 30` (where no wide feature subsampling occurs), averaging 4 deterministic feature-column permutations in `ICL([X, S])` reduces column-order variance in the transformer's attention layers, lifting `clean-88` to `+0.0431 u` clipped (`p = 0.01491`, `W/T/L = 41/18/29`, `lofo_min = +0.0378 u`) and `clean_with_struct` (`n = 70`) to `+0.0542 u` clipped (`+0.1182 u` raw = `11.2%` RMSE reduction over `R0`, `W/T/L = 41/0/29`).

---

## 3. Frozen Handoff (`FROZEN.json`)

Per the pre-registered Freeze Rule (`PREREG.md` §2: `clipped_mean >= +0.01 u` and one-sided `signflip_p < 0.05` on `clean-88`), `eval_cw5_fitsel.py readout` emitted `FROZEN.json`:
- `selected_recipe`: `"R4"`
- `promote_to_joint_dev`: `true`
- `ckpt_sha256`: `"e3c8fd9972ce0c1684630ff06b159e950f6f498c4461d0454e0c6a6f44d38dd7"` (`ICL_M_r3_ckpt_h020.pt`)

## 4. Artifact Inventory (`xbabe2`)

- Pre-readout gates: `/home/ubuntu/jope_plan/evidence/H442/results/H442.gates.json`
- Single-shot readout: `/home/ubuntu/jope_plan/evidence/H442/results/READOUT_CW5.json`
- Frozen handoff: `/home/ubuntu/jope_plan/evidence/H442/results/FROZEN.json` (copied verbatim to `FROZEN.json` in this directory)
- Per-context outputs (`106` contexts + `16` shard status receipts): `/home/ubuntu/jope-data/h335-cl/h442/pub-r1/ctxs/`
