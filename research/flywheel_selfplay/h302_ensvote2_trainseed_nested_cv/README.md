# H302 — `ENS_VOTE2` True Training-Seed Confirmation via 4-Fold Nested Family-Grouped CV inside `DL-Train` (`h302_ensvote2_trainseed_nested_cv`)

- **Wave ID**: `H302` (`h302_ensvote2_trainseed_nested_cv`)
- **Owner**: `rd_argon`
- **Status / Verdict**: **`COMPLETE` / `INNERCV_CONFIRMED_TRAINING_SEEDS`**
- **Preregistration commit (pushed to `origin/v1/next` BEFORE any outer-test score was computed)**: `2a77787c3df2a0ac9b5ae6bf5684bd862bfca7fc`
- **Pinned `H287` export commit**: `213b30ad52e1ce48927bba960ed609bd98ab4dfe`
- **Pinned `selected.json` (`DL-Train`) SHA256**: `981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1` (`1,219` contexts across `205` coarse families)
- **Frozen `DOPE` (`r3175`) SHA256**: `7eb6adcd9ad9bcdffcebe4d5fa3ba8a5ca4fbefbd7ff0ba814df21db2524c58f` (`0` optimizer updates, `0` GPU use, `0` fresh cohorts touched)

---

## 1. Executive Summary

`H301` (`ed8956e2c1`) proved `SELECTION_NOISE_PLATEAU` across adapter-optimizer waves (`H296` SWA/Polyak also `NOT_ADVANCED`, `b915eaf1`). The selection layer (`ENS_VOTE2`) is the productive path (`H294` `CONFIRMED_FRESH`, `+0.0329` family-equal gain, Elo `1652.14` vs `1500` baseline). However, `H294`'s 3 reported "seeds" (`294150..294152`) were **evaluation bootstrap seeds on a single deterministic policy** (identical pick hash `6f595a93...` across all 3 rows), and `H300` (true training-seed confirmation on fresh `v6A`) is currently `BLOCKED_NO_COHORT` (`H299`, `3b8253f68f`).

`H302` answers the exact `H300` training-seed confirmation question **without fresh data** via **4-fold nested family-grouped cross-validation inside `DL-Train`** (`205` coarse families, `1,219` contexts):
1. **Step 1 (`H287` Serialization/Scorer Regression)**: Passed bit-for-bit (`True`) on all `60` models and all ensemble/member arms (`ENS_VOTE2` `19` changed picks on the `64` check contexts, `picks_sha256 = 8ef2837ae43e22721b7f7ffdf30371c29ba18e1ad3d9f0f788462d7c350d6d01`; see [`regression_receipt.json`](regression_receipt.json)).
2. **Step 2 (4-Fold Outer Family Split × 3 Independent Root Training Seeds `30001 / 30002 / 30003`)**:
   - Deterministic outer fold assignment `SHA256("H302-outer-v1\x00" + family_id)[-1] % 4` produced `46 / 54 / 52 / 53` outer-test families (`308 / 270 / 331 / 310` outer-test contexts, `205` families / `1,219` contexts total).
   - In every outer fold $k \in \{0,1,2,3\}$ and for each root training seed `30001`, `30002`, `30003`, all `3` `BAG_SWITCH` members (`36` members / `360` LightGBM models total across the 4 folds × 3 root seeds) were refit from scratch using **only `outer_train` families** (inner 4-fold family CV inside `outer_train` for LR/trees/strict-margin selection, `8` family-bootstrap rankers + `1` full ranker + `1` validity classifier, invalid threshold `0.10`, vote `6/8`), frozen to disk in `fold_{k}_fit_lock.json`, verified bit-exact on `64` `outer_train` contexts, and then used to score `outer_test` families **once**.
3. **Primary Verdict — `INNERCV_CONFIRMED_TRAINING_SEEDS`**:
   - **Condition 1 (`median_s gain > +0.001`)**: **PASS** — median family-equal mean gain across the 3 root training seeds is **`+0.058892`** (`pooled 95% CI = [+0.013275, +0.126473]`; also passes `H294`'s stricter `+0.010` threshold by `5.89x`).
   - **Condition 2 (`per-seed LCB95 > 0` on `3/3` training seeds)**: **PASS** — one-sided `95%` lower confidence bounds (`0.05` quantile on shared `10,000`-draw whole-family bootstrap `boot_seed = 302150`) are **`+0.014776`** (`30001`), **`+0.017449`** (`30002`), and **`+0.016561`** (`30003`), and two-sided `95%` lower bounds (`0.025` quantile) are **`+0.012795`**, **`+0.011802`**, and **`+0.010329`** (`3/3 > 0`).
   - **Condition 3 (`harmed-family share <= 35%` each seed)**: **PASS** — harmed-family shares are **`4.39%` (`9/205`)**, **`10.73%` (`22/205`)**, and **`4.88%` (`10/205`)** (also well below `H294`'s stricter `25%` ceiling).
   - **Condition 4 (`invalid excess <= 3` and McNemar `p >= 0.05` each seed)**: **PASS** — invalid picks are **`15`**, **`14`**, and **`14`** vs slot-0's **`17`** (`excess = -2, -3, -3 <= 3`; exact one-sided McNemar `p = 0.8906, 0.9375, 0.9688 >= 0.05` — `ENS_VOTE2` **reduces** invalid selections on every training seed).

---

## 2. Primary `ENS_VOTE2` Results by Root Training Seed (`205` `DL-Train` Families, `1,219` Out-of-Fold Contexts)

Baseline `r3175` slot `0`: family-equal mean utility = `1.385355`, context mean utility = `1.423073`, invalid count = `17 / 1219` (`1.39%`), Bradley-Terry Elo = `1500.00`.

| Root Training Seed | Member Seeds (`m=0, 1, 2`) | Family-Equal Mean Gain (`[2-Sided 95% CI]`) | 1-Sided `LCB95` (`q05`) | Secondary Family-Median Gain (`[95% CI]`) | Context Mean Gain (`[95% CI]`) | Family `W / T / L` (`\|d\| <= 1e-4` tie) | Exact Sign Test $p$ | Harmed Families (`<= 35%`) | Invalid Picks vs Slot 0 (`excess`, McNemar $p$) | Changed Picks (`/ 1219`) | MM Bradley-Terry Elo (`[95% CI]`) | Picks SHA256 (prefix) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **`30001`** | `685781424, 1430007429, 1330439319` | **`+0.026823`** (`[+0.012795, +0.042500]`) | **`+0.014776`** | `0.000000` (`[0.000000, 0.000000]`) | `+0.023735` (mean `1.446809`) | `77 / 111 / 17` | `1.35e-10` | `9 / 205` (**`4.39%`**) | `15` vs `17` (`-2`, $p=0.8906$) | `154` (`12.63%`) | **`1762.34`** (`[1679.30, 1868.82]`) | `b47088d981af...` |
| **`30002`** | `1869595753, 786733481, 1272701215` | **`+0.060358`** (`[+0.011802, +0.129037]`) | **`+0.017449`** | `+0.001588` (`[0.000000, +0.010223]`) | `+0.049968` (mean `1.473041`) | `108 / 61 / 36` | `7.48e-10` | `22 / 205` (**`10.73%`**) | `14` vs `17` (`-3`, $p=0.9375$) | `301` (`24.69%`) | **`1690.82`** (`[1628.11, 1762.52]`) | `0c5761b312cd...` |
| **`30003`** | `1510579523, 89540748, 1415857569` | **`+0.058892`** (`[+0.010329, +0.127669]`) | **`+0.016561`** | `0.000000` (`[0.000000, +0.004783]`) | `+0.045482` (mean `1.468556`) | `96 / 87 / 22` | `1.63e-12` | `10 / 205` (**`4.88%`**) | `14` vs `17` (`-3`, $p=0.9688$) | `211` (`17.31%`) | **`1755.88`** (`[1682.91, 1846.58]`) | `a4ff743c9b97...` |
| **Across Seeds (`median_s` / pooled)** | — | **`+0.058892`** (`[+0.013275, +0.126473]`) | **`+0.016561`** (min `+0.014776`) | `0.000000` (`[0.000000, +0.004759]`) | `+0.045482` | — | — | max **`10.73%`** | max excess **`-2`** | `211` (range `154..301`) | **`1755.88`** (`[1674.25, 1814.75]`) | **3 distinct hashes** |

---

## 3. Seed-Noise Red-Team: `H302` True Training Seeds vs `H294` Evaluation Bootstrap Seeds

| Metric | `H294` (`294150..294152` Bootstrap Seeds on 1 Policy) | `H302` (`30001..30003` True Root Training Seeds, Shared Bootstrap `302150`) | Interpretation |
| --- | --- | --- | --- |
| **Policies make identical picks?** | **`True`** (`1` unique pick hash `6f595a93...`) | **`False`** (`3` distinct pick hashes: `b47088d9...`, `0c5761b3...`, `a4ff743c...`) | `H302` genuinely varies the full training pipeline (inner CV, bootstrap rankers, LightGBM feature/row subsampling). |
| **Changed-picks count across seeds** | `297, 297, 297` (range **`0`**) | `154, 301, 211` (range **`147`**) | In Fold 2, Root Seed `30001` Member 0 hit the strict infinite-margin fallback (`strict_admitted=False`), making Seed `30001` more conservative (`154` switches), yet `ENS_VOTE2` still achieves `+0.026823` (`LCB95 = +0.014776`, Elo `1762.34`). |
| **Pairwise context pick agreement (`N = 1,219`)** | `100.0%` (`all 3 equal = 100.0%`) | `78.67%` (`30001 vs 30002`), `84.41%` (`30001 vs 30003`), `83.76%` (`30002 vs 30003`); all 3 equal on `912 / 1219` (**`74.82%`**) | Among the `372` contexts where at least one policy switched away from slot 0, `65` were switched to the exact same nonzero slot by all 3 training seeds. |
| **Point family-equal mean gain range (`SD`)** | `0.000000` (`SD = 0.000000`) | **`0.033536`** (`+0.026823` to `+0.060358`, `SD = 0.018953`) | True training-seed variation induces a `0.0335` spread in point family-equal mean gain (`0.0` under `H294` bootstrap seeds). |
| **One-sided `LCB95` range (`SD`)** | `0.000377` (`+0.016042` to `+0.016419`, `SD = 0.000206`) | **`0.002673`** (`+0.014776` to `+0.017449`, `SD = 0.001361`, **`7.10x` wider**) | Despite `7.10x` wider `LCB95` variation across true training seeds, **every root training seed's `LCB95` remains strictly above `+0.0147`** (and two-sided `CI95_lo >= +0.0103`). |

---

## 4. Secondary Ensemble Arms, Single-Member Controls, and Nulls

Why `>=2 of 3` voting (`ENS_VOTE2`) is essential: single `BAG_SWITCH` and `RERANK8_VETO` members suffer from training-seed variance (`MEMBER_1_BAG_SWITCH` drops to `LCB95 = -0.005665` on seed `30002`; `MEMBER_0_RERANK8_VETO` has `LCB95 < 0` on seeds `30002` and `30003`), whereas `ENS_VOTE2` (`>=2 of 3` agreement) filters out single-member false-positive switches and passes all 4 confirmation criteria on `3/3` root training seeds.

| Arm | Median Gain Across Seeds (`[Pooled 95% CI]`) | Per-Seed Gains (`30001, 30002, 30003`) | Per-Seed 1-Sided `LCB95` | Harmed Share by Seed (`<= 35%`) | Invalid Picks by Seed (vs `17`) | Changed Picks by Seed | Median MM Elo | Passes All 4 `H302` Gates? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **`ENS_VOTE2` (Primary)** | **`+0.058892`** (`[+0.013275, +0.126473]`) | `+0.026823, +0.060358, +0.058892` | **`+0.014776, +0.017449, +0.016561`** | `4.39%, 10.73%, 4.88%` | `15, 14, 14` | `154, 301, 211` | **`1755.88`** | **`PASS` (`INNERCV_CONFIRMED_TRAINING_SEEDS`)** |
| `ENS_MEAN` (Secondary) | `+0.054993` (`[+0.017711, +0.124155]`) | `+0.054993, +0.067252, +0.052854` | `+0.019589, +0.027406, +0.012264` | `4.88%, 10.24%, 7.80%` | `16, 14, 14` | `185, 224, 198` | `1745.51` | `PASS` |
| `ENS_VOTE3` (Secondary) | `+0.018087` (`[+0.003854, +0.040773]`) | `+0.008090, +0.018432, +0.018087` | `+0.001459, +0.004363, +0.003913` | `1.46%, 3.90%, 3.41%` | `18, 16, 17` | `41, 86, 75` | `1698.85` | `PASS` |
| `MEMBER_0_BAG_SWITCH` | `+0.051042` (`[+0.012191, +0.122320]`) | `+0.020963, +0.058325, +0.051042` | `+0.007978, +0.013075, +0.006149` | `6.83%, 10.73%, 11.22%` | `15, 14, 14` | `176, 307, 297` | `1698.37` | `PASS` |
| `MEMBER_1_BAG_SWITCH` | `+0.053960` (`[-0.003770, +0.119061]`) | `+0.056603, +0.036937, +0.053960` | `+0.014830, -0.005665, +0.011003` | `9.27%, 14.15%, 9.76%` | `15, 16, 15` | `265, 315, 232` | `1696.32` | `FAIL` (seed `30002` `LCB95 < 0`) |
| `MEMBER_2_BAG_SWITCH` | `+0.052267` (`[+0.008777, +0.118054]`) | `+0.048779, +0.056065, +0.052267` | `+0.007764, +0.010817, +0.008167` | `8.78%, 11.71%, 6.34%` | `14, 15, 14` | `223, 305, 224` | `1699.13` | `PASS` |
| `MEMBER_0_RERANK8_VETO` | `+0.038929` (`[-0.004422, +0.107339]`) | `+0.056979, +0.038929, +0.013792` | `+0.016164, -0.001279, -0.002976` | `7.32%, 14.15%, 3.90%` | `16, 15, 15` | `191, 293, 98` | `1685.64` | `FAIL` (seeds `30002, 30003` `LCB95 < 0`) |
| `MEMBER_1_RERANK8_VETO` | `+0.049302` (`[+0.005669, +0.116250]`) | `+0.049302, +0.056176, +0.025812` | `+0.012470, +0.012224, +0.008568` | `6.83%, 12.20%, 6.83%` | `15, 15, 16` | `158, 250, 149` | `1698.37` | `PASS` |
| `MEMBER_2_RERANK8_VETO` | `+0.052291` (`[+0.005801, +0.119066]`) | `+0.022674, +0.059887, +0.052291` | `+0.008770, +0.014830, +0.004091` | `4.88%, 11.71%, 9.76%` | `15, 15, 14` | `124, 286, 200` | `1719.24` | `PASS` |
| `NULL_RANDOM_VALID_SLOT` | `+0.012351` (`[-0.042417, +0.069460]`) | `+0.015644, +0.012351, +0.024189` | `-0.037881, -0.042685, -0.030001` | `47.32%, 44.88%, 42.93%` | `18, 17, 15` | `1202, 1202, 1202` | `1506.92` | `FAIL` (`LCB95 < 0`, harmed `> 35%`) |
| `NULL_RANDOM_SLOT_NO_LABELS` | `-0.079692` (`[-0.128961, -0.023673]`) | `-0.080115, -0.052126, -0.079692` | `-0.136573, -0.107873, -0.136767` | `43.41%, 51.71%, 46.34%` | `30, 28, 28` | `1053, 1084, 1061` | `1462.66` | `FAIL` (all 4 gates fail) |

---

## 5. Artifacts & S3 Archival

- [`PREREG.md`](PREREG.md) — Preregistration pushed in commit `2a77787c3df2a0ac9b5ae6bf5684bd862bfca7fc` before any outer-test evaluation.
- [`regression_receipt.json`](regression_receipt.json) — Step 1 `H287` serialization/scorer regression receipt (`PASS`, bit-exact on `64` `DL-Train` contexts).
- [`metrics.json`](metrics.json) — Compact metrics (`< 50 KB`, `allow_nan=False`).
- [`s3_manifest.json`](s3_manifest.json) — S3 archive receipt for all `360` LightGBM models (`4` outer folds × `3` root training seeds × `3` members × `10` models), `4` `fold_{k}_fit_lock.json` files, and `metrics_full_h302.json` uploaded to `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H302/`.
- [`analysis_h302.py`](analysis_h302.py), [`run_h302.py`](run_h302.py), [`test_h302.py`](test_h302.py) — Deterministic nested-CV runner, analysis module, and unit tests (`5/5` passing).
