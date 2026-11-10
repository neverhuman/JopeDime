# H287 — `BAG_SWITCH` Seed-Ensemble & Clipped-Utility Freeze/Export on `DL-Train` (`h287_bagswitch_seed_ensemble_export`)

## 1. Hypothesis & Context
- **Wave**: `H287` (`h287_bagswitch_seed_ensemble_export`)
- **Lane / Agent**: `rd_argon` (`[ARGON]`)
- **Authority**: `/home/ubuntu/nudges/hw_t0005_argon_h287.md`
- **Diagnosis from `H284` (`98d5b9f5`) and `H286` (`9a79996f`)**:
  - `BAG_SWITCH` ($q = 6$, $B = 8$) is the strongest and safest post-hoc adapter arm (`H286` median gain `+0.0630 [+0.019, +0.117]`, harm `3.5–10.5%`, Elo `1555/1672/1550`), but every single-seed arm fails the `3/3` per-seed `LCB95 > 0` gate because one seed misses (`2/3` seeds `LCB95 > 0`).
  - Furthermore, in `H284`, `seed 279101` selected `strict_margin = 0.02` (`lr = 0.03, n_estimators = 50`) instead of `0.10` (`279102` and `279103`), causing `36` harmed families on `DL-Train` inner CV vs `12` and `26` for the other two seeds.
- **Hypothesis (`H287`)**: A seed-ensemble of the three `H284` frozen `BAG_SWITCH` policies (`seeds 279101–279103`) is at least as good as the median single seed under 4-fold family-grouped inner CV on `DL-Train` (`205` families, `1,219` contexts), with lower or equal harm, and can be frozen as ONE deterministic policy for out-of-sample fresh-pool evaluation in `H288`.

## 2. Experimental Setup & Isolation Contract (`fresh_pools_read = 0`)
1. **Step 1 — Regressions & Isolation**:
   - Reproduced `Native91 = 1.1377941898164525` (`91/91` exact context matches) and `TOP4_PICK@r882` (`1.2403601682250796` context mean, `1.2519997168684853` family mean).
   - Reproduced `H279` cell `KM276140-S279101-C0` bit-exact (`receipts`, `assess_picks`, `outer_votes`, `shuf_votes`, `invalid`, `changed`).
   - **`fresh_pools_read = 0` enforced for the entire wave**: `sys.addaudithook` blocked any open of `/sealed`, `/holdout`, `meta-test`, `qualification.lock`, `fresh_pools`, `fresh_cohort`, `fresh_top8`, `fresh-top8`, `fresh-build`.
   - Zero neural updates (`optimizer_updates = 0`); `DOPE` composite `sha256` (`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`) and `round-882.pt` (`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`) verified before and after.
2. **Step 2 — Arms & 4-Fold Family-Grouped Inner CV (`DL-Train`, `205` families, `1,219` contexts)**:
   - Evaluated across the 3 `H284` 4-fold family-grouped partitions (`fold_seeds = 279101–279103`; `refit_seeds = 287101–287103`), where on each held-out fold `va` (`1/4` of `DL-Train`) all 3 ensemble members are fit strictly on the remaining `3/4` training folds `tr`:
     - `BAG_SWITCH` (single-seed reference, $q = 6$, $B = 8$; bit-exact to `H284` inner CV on each replicate).
     - `ENS_MEAN` (deterministic 1-policy ensemble): mean veto probability $\bar{p}_{\text{inv}}[k] = \frac{1}{3}\sum_{m=1}^3 p_{\text{inv},m}[k] < \tau = 0.10$ for $k \ge 1$; pick $j^* = \arg\max_k \text{masked}(\bar{s})[k]$ where $\bar{s}[k] = \frac{1}{3}\sum_{m=1}^3 s_m[k]$; switch from `slot 0` to $j^*$ iff mean ranker margin $\frac{1}{3}\sum_{m=1}^3 (s_m[j^*] - s_m[0]) > \bar{\mu} = 0.073333$ AND total bag votes across the 3 seeds' $24$ bagged rankers $\ge 18$ (`mean bag vote >= 6.0`); else `slot 0`. Also reported without the 24-bag vote gate as `ENS_MEAN_PROPOSER`.
     - `ENS_VOTE2` (deterministic 1-policy majority vote): switch away from `slot 0` to candidate $j \ge 1$ iff $\ge 2$ of $3$ member `BAG_SWITCH` ($q = 6$) policies switch AND agree on the exact same candidate $j$; else `slot 0`.
     - `ENS_VOTE3` (deterministic 1-policy unanimous vote): switch away from `slot 0` to candidate $j \ge 1$ iff all $3$ of $3$ member `BAG_SWITCH` ($q = 6$) policies switch AND agree on the exact same candidate $j$; else `slot 0`.
     - `CLIP` (pre-declared exploratory): `BAG_SWITCH` ($q = 6$) refit on `DL-Train` with per-context training utility clipped at `-1.0` (`np.maximum(utility_k, -1.0)`) so `-2.0` invalid picks do not dominate the inner hyperparameter selection loss (`3` seeds; evaluated on true unclipped utility), plus its deterministic majority-vote ensemble `CLIP_ENS_VOTE2`.
   - Note on `CLIP` mechanism: on `DL-Train`, 0 valid candidates have `utility < -1.0` (`min valid utility = +0.0805`), while `231` invalid candidates have `utility = -2.0`. Clipping training utility at `-1.0` specifically halves the inner-CV hyperparameter-selection penalty of a single invalid pick, which flips `seed 279101` from the noisy `(lr = 0.03, n_estimators = 50, strict_margin = 0.02)` to `(lr = 0.03, n_estimators = 200, strict_margin = 0.10)` so all 3 seeds select `strict_margin = 0.10`.

## 3. Primary Inner-CV Results on `DL-Train` (`205` families, `1,219` contexts; `10,000` bootstrap draws, seed `287150`)

| Arm | Median Gain vs `slot 0` `[CI95]` | Per-Seed Gains (`287101 / 287102 / 287103`) | Per-Seed `CI95` | `LCB95 > 0` | Median `W / T / L` (`\|d\| <= 1e-4` tie) | Harmed (`loss > 0.01`) by Seed (Med / Mean) | Invalid `-2.0` vs `slot 0` (`24`) | Changed Picks by Seed | Median Elo (`slot 0 = 1500`) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `RERANK8_VETO` | `+0.0827 [+0.0390, +0.1552]` | `+0.0804 / +0.0827 / +0.0962` | `[+0.0266,+0.1489] / [+0.0386,+0.1525] / [+0.0399,+0.1761]` | `3/3` | `136 / 3 / 55` | `[57, 32, 38]` (`38.0 / 42.33`) | `[12, 14, 13]` | `[840, 587, 722]` | `1710.8` |
| `BAG_SWITCH` (1-seed ref) | `+0.0693 [+0.0362, +0.1216]` | `+0.0693 / +0.0617 / +0.0874` | `[+0.0300,+0.1199] / [+0.0316,+0.1073] / [+0.0451,+0.1474]` | `3/3` | `134 / 30 / 41` | `[36, 12, 26]` (`26.0 / 24.67`) | `[16, 13, 10]` | `[526, 248, 401]` | `1677.6` |
| **`ENS_MEAN`** | **`+0.0751 [+0.0349, +0.1216]`** | **`+0.0589 / +0.0751 / +0.0809`** | **`[+0.0263,+0.1059] / [+0.0428,+0.1224] / [+0.0332,+0.1550]`** | **`3/3`** | **`131 / 42 / 39`** | **`[26, 19, 27]` (`26.0 / 24.00`)** | **`[17, 12, 13]`** | **`[353, 342, 347]`** | **`1676.0`** |
| `ENS_MEAN_PROPOSER` | `+0.0802 [+0.0356, +0.1462]` | `+0.0802 / +0.0776 / +0.0843` | `[+0.0403,+0.1309] / [+0.0305,+0.1465] / [+0.0318,+0.1609]` | `3/3` | `146 / 9 / 52` | `[38, 40, 36]` (`38.0 / 38.00`) | `[11, 14, 12]` | `[636, 662, 676]` | `1730.7` |
| `ENS_VOTE2` | `+0.0599 [+0.0308, +0.1039]` | `+0.0539 / +0.0599 / +0.0622` | `[+0.0311,+0.0797] / [+0.0289,+0.1053] / [+0.0232,+0.1199]` | `3/3` | `111 / 61 / 33` | `[24, 17, 22]` (`22.0 / 21.00`) | `[14, 13, 14]` | `[296, 268, 265]` | `1645.4` |
| `ENS_VOTE3` | `+0.0217 [+0.0088, +0.0317]` | `+0.0217 / +0.0144 / +0.0236` | `[+0.0079,+0.0377] / [+0.0051,+0.0257] / [+0.0087,+0.0421]` | `3/3` | `52 / 139 / 14` | `[9, 7, 7]` (`7.0 / 7.67`) | `[18, 16, 14]` | `[96, 92, 78]` | `1546.0` |
| **`CLIP`** (exploratory) | **`+0.0874 [+0.0433, +0.1463]`** | **`+0.0936 / +0.0617 / +0.0874`** | **`[+0.0467,+0.1660] / [+0.0312,+0.1071] / [+0.0450,+0.1476]`** | **`3/3`** | **`134 / 30 / 39`** | **`[31, 12, 26]` (`26.0 / 23.00`)** | **`[13, 13, 10]`** | **`[472, 248, 401]`** | **`1703.9`** |
| `CLIP_ENS_VOTE2` (exploratory) | `+0.0649 [+0.0327, +0.1128]` | `+0.0637 / +0.0649 / +0.0666` | `[+0.0314,+0.1107] / [+0.0329,+0.1094] / [+0.0279,+0.1237]` | `3/3` | `117 / 53 / 35` | `[24, 16, 21]` (`21.0 / 20.33`) | `[15, 11, 13]` | `[290, 274, 282]` | `1635.9` |

## 4. Reused `DL-Val91` Descriptive Check (`91` contexts, `44` families; explicitly labeled `"reused": true`)

| Split (`reused = true`) | Arm | Gain vs `slot 0` `[CI95]` | `LCB95 > 0` | `W / T / L` | Harmed (`loss > 0.01`) | Invalid `-2.0` vs `slot 0` (`0`) | Changed Picks | Elo (`slot 0 = 1500`) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `reused_r882_dlval91` | `BAG_SWITCH` (3-seed med) | `+0.0212 [+0.0088, +0.0342]` | `3/3` | `12 / 4 / 2` | `[1, 0, 0]` | `[0, 0, 0]` | `[36, 19, 31]` | `1739.6` |
| `reused_r882_dlval91` | `ENS_MEAN` (1-policy) | `+0.0163 [+0.0046, +0.0295]` | `3/3` | `11 / 5 / 3` | `[2, 2, 2]` | `[0, 0, 0]` | `27` | `1664.9` |
| `reused_r882_dlval91` | `ENS_MEAN_PROPOSER` (1-policy) | `+0.0419 [+0.0216, +0.0657]` | `3/3` | `17 / 1 / 1` | `[0, 0, 0]` | `[0, 0, 0]` | `46` | `1929.1` |
| `reused_r882_dlval91` | `ENS_VOTE2` (1-policy) | `+0.0156 [+0.0040, +0.0300]` | `3/3` | `10 / 6 / 3` | `[1, 1, 1]` | `[0, 0, 0]` | `23` | `1701.0` |
| `reused_r882_dlval91` | `ENS_VOTE3` (1-policy) | `+0.0104 [+0.0023, +0.0208]` | `3/3` | `7 / 11 / 1` | `[0, 0, 0]` | `[0, 0, 0]` | `11` | `1608.5` |
| `reused_r882_dlval91` | `CLIP` (3-seed med) | `+0.0282 [+0.0126, +0.0447]` | `3/3` | `13 / 4 / 2` | `[0, 0, 0]` | `[0, 0, 0]` | `[35, 19, 31]` | `1776.3` |
| `reused_r882_dlval91` | `CLIP_ENS_VOTE2` (1-policy) | `+0.0212 [+0.0076, +0.0380]` | `3/3` | `11 / 6 / 2` | `[0, 0, 0]` | `[0, 0, 0]` | `25` | `1719.2` |
| `reused_r782_r785_distinct_mean_dlval91` | `BAG_SWITCH` (3-seed med) | `+0.0326 [+0.0091, +0.0687]` | `3/3` | `12 / 4 / 1` | `[0, 0, 0]` | `[0, 0, 0]` | `[71, 38, 58]` | `1739.2` |
| `reused_r782_r785_distinct_mean_dlval91` | `ENS_MEAN` (1-policy) | `+0.0283 [+0.0054, +0.0645]` | `3/3` | `10 / 6 / 3` | `[1, 1, 1]` | `[0, 0, 0]` | `53` | `1675.2` |
| `reused_r782_r785_distinct_mean_dlval91` | `ENS_MEAN_PROPOSER` (1-policy) | `+0.0652 [+0.0178, +0.1316]` | `3/3` | `16 / 1 / 2` | `[0, 0, 0]` | `[0, 0, 0]` | `93` | `1888.8` |
| `reused_r782_r785_distinct_mean_dlval91` | `ENS_VOTE2` (1-policy) | `+0.0310 [+0.0072, +0.0679]` | `3/3` | `12 / 7 / 0` | `[0, 0, 0]` | `[0, 0, 0]` | `43` | `1757.4` |
| `reused_r782_r785_distinct_mean_dlval91` | `ENS_VOTE3` (1-policy) | `+0.0279 [+0.0055, +0.0642]` | `3/3` | `10 / 9 / 0` | `[0, 0, 0]` | `[0, 0, 0]` | `20` | `1672.1` |
| `reused_r782_r785_distinct_mean_dlval91` | `CLIP` (3-seed med) | `+0.0411 [+0.0097, +0.0897]` | `3/3` | `14 / 4 / 1` | `[1, 0, 0]` | `[0, 0, 0]` | `[61, 38, 58]` | `1816.1` |
| `reused_r782_r785_distinct_mean_dlval91` | `CLIP_ENS_VOTE2` (1-policy) | `+0.0381 [+0.0082, +0.0886]` | `3/3` | `12 / 7 / 0` | `[0, 0, 0]` | `[0, 0, 0]` | `47` | `1758.0` |

## 5. Frozen S3 Export & 64-Context Reload Verification (Step 4)
- **S3 Prefix**: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H287/export/` (`60` LightGBM model `.txt` files + `ensemble_spec.json` + `fit_lock.json` + `export_manifest.json`):
  - `export_manifest.json`: `VersionId = rpqNaWcC23tUA06TThSW3103x3DHoinR`, `sha256 = 874693a78d52ebeee2e2c0c38d0f028cd3887a01abec8d0c383128e0b4cf06d7`
  - `ensemble_spec.json`: `VersionId = Wg0jPfW1mYwuaEawsCU2gVn5PkGWGCCX`, `sha256 = 32ca19f30f427ae135f5f6c2aacc0fe28097d956f4e2ca0e7fd5f41ffec49995`
  - `fit_lock.json`: `VersionId = 7FgS1SS0xdHQvxfSXUZtCxIcw.GeyEHs`, `sha256 = 8a20c58bdb4bfcbadfa5ba927aa0577b630e3174e6dc4a7bfd579175ca08f35f`
- **Scorer Entrypoints**:
  - Module: `research/flywheel_selfplay/h287_bagswitch_seed_ensemble_export/ensemble.py`
  - Single-arm function: `score_h287_arm(export_dir, arm, X, n_ctx, k=8, spec=None, seed_index=0)`
  - Batch function: `score_all_h287_arms(export_dir, X, n_ctx, k=8, spec=None)`
- **64-Context `DL-Train` Reload Bit-Exact Check**: `PASS` (`64/64` contexts bit-exact across all 60 serialized LightGBM models and all deterministic ensemble policies).

## 6. Verdict, Deviations & Next Step
- **Verdict**: **`EXPORTED_ADVANCE`**
  - `ENS_MEAN` passes all 3 primary `H287` inner-CV advancement criteria on `DL-Train`:
    1. Inner-CV median gain `+0.0751 >= +0.0693` (single-seed `BAG_SWITCH` median),
    2. `per-seed LCB95 > 0` on `3/3` (`[+0.0263, +0.1059]`, `[+0.0428, +0.1224]`, `[+0.0332, +0.1550]`),
    3. Harmed families (`loss > 0.01`) median `26.0 <= 26.0` (and mean `24.00 < 24.67`, cutting worst-seed harm from `36` to `27`).
  - Exploratory `CLIP` (`-1.0` training-utility clipping) also passes all 3 criteria (`median gain +0.0874 >= +0.0693`, `3/3` `LCB95 > 0`, harmed median `26.0 <= 26.0`, mean `23.00 < 24.67`), while `ENS_VOTE2` (`harmed = [24, 17, 22]`, median `22.0`), `CLIP_ENS_VOTE2` (`harmed = [24, 16, 21]`, median `21.0`), and `ENS_VOTE3` (`harmed = [9, 7, 7]`, median `7.0`) provide strictly lower-harm deterministic policies.
- **Deviations**: None (`fresh_pools_read = 0`; single official execution `attempt-001`).
- **Next Step**: Out-of-sample evaluation of the frozen `H287` policies (`ENS_MEAN`, `ENS_MEAN_PROPOSER`, `ENS_VOTE2`, `ENS_VOTE3`, `CLIP`, `CLIP_ENS_VOTE2`) by `rd_opus` in `H288` on the new fresh ODD cohort v3.
