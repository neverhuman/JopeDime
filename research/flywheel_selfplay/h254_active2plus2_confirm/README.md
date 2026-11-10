# Wave H254 — Independent Frozen-Constant Confirmation of `ACTIVE_2PLUS2_HURDLE` vs `GATED_025` and `R882` (`h254-active2plus2-confirm-20261103`)

- **Status**: `COMPLETE` (`attempt-001` on `xbabe2` CPU cores `57-60`, `nice 19`, `283.67s`, peak RSS `3.57 GiB`, `0` pauses)
- **Verdict**: **`ADVANCE`** (all 4 preregistered gates passed: `DL-Val` primary median family gain `+0.015519 > 0`, Holm-adjusted `97.5% CI` `[+0.004015, +0.029097] > 0`, `Holm p = 0.0002 < 0.05`, `3/3` positive seeds, and Negative Control `95% CI` `[-0.008987, +0.010203]` covers `0`)
- **Preregistration**: [PREREG.md](PREREG.md) (PR `#2111` at `e77ab29e2debcc74850aa60d7ea79a6dc764ac0f`, S3 prereg `VersionId=fyfEwHKOy2bFdiO.VRpiyPWFfs2QbRTf`)
- **Fold Manifest**: [fold_manifest.json](fold_manifest.json) (`fold_seed = 254140`, seeds `254101-254103`)
- **Manifest & Metrics**: [manifest.json](manifest.json), [metrics.json](metrics.json)
- **Code & Tests**: [runner.py](runner.py), [monitor.py](monitor.py), [test_h254.py](test_h254.py)

---

## 1. Hypothesis

Wave H254 tests whether the Stage-1 `ACTIVE_2PLUS2_HURDLE` sequential `2+2` active reveal policy from Wave H252 (`agy_research_wave`) independently replicates its positive utility and Elo gains over both passive 4-reveal `GATED_025` (at an identical 4-reveal per-context budget) and frozen `R882` on `DL-Val` (`Native91`, `91` contexts / `19` families) and `DL-Train` OOF (`1,219` contexts / `205` families) when evaluated with:
1. Byte-exact frozen arms and constants from the H252 Stage-1 snapshot (`0` constant changes),
2. A fresh 5-fold family-grouped cross-fitting partition (`fold_seed = 254140`) on `DL-Train` only (`DL-Val` priors fit on `DL-Train` only),
3. Fresh evaluation seeds (`254101`, `254102`, `254103`) and bootstrap seed (`254150`), and
4. Four explicit falsification/hygiene audits: `(a)` zero `DL-Val` leakage, `(b)` per-context reveal count `== 4`, `(c)` within-context permuted revealed-fitness negative control (`95% CI` must cover `0`), and `(d)` descriptive `±50%` constant perturbation sensitivity.

## 2. Setup

- **Snapshot Custody** (`/home/ubuntu/jope_plan/evidence/H254/argon-t1305/snapshot/`, read-only):
  - `common_h252.py`: `abe974ed10a0cd0d4b08440015a72a7b0adcebb22118c234732b812e9f0b0f92`
  - `stage1_hurdle_sos_critic.py`: `fc0dd0e2fa543f0540d677609989c1785ef33b07f6cf7e54f4f8bedbd0e5e797`
- **Read-Only Inputs** (`/home/ubuntu/jope_plan/evidence/H254/argon-t1305/inputs/`):
  - `r882/round-882.pt`: `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64` (`DOPE` composite `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`, verified unchanged before and after execution)
  - `r882/best_dev_checkpoint.json`: `30853cad78d860870338316dfb5a4e869e46dc5c22d42fe4ab9d5f24f31c373a`
  - `r882-train-costs.npz`: `57004a91a315b372cde62b126be20edd4eb0f337942e08b19bd85438b520f69f`
  - `r882-secondary-costs.npz`: `0e812aa7ea4ce46a85cbe17cb7eba3a58678cf13ec3b215d48a334dd5626f11f`
  - `promotions.jsonl`: `c9afe08ed4d76f59aa807102423b502b57f8c6bddcfedb6a3ae89f8b85821550` (`3,449` `DL-Train`-only GP promotions)
- **Baseline Regression Check (`Native91`)**: Executed first before any treatment arm; reproduced `1.1377941898164525` (`91/91` bit-exact context matches).
- **Compute Envelope**: `xbabe2` CPU only, `taskset -c 57-60`, `nice -n 19`, `1` math thread, peak `2` threads (`<= 4`), peak RSS `3,836,780,544 B` (`3.57 GiB <= 4.0 GiB`), `0` H239 A1 memory/load pauses (`MemAvailable` `62.5–74.8 GiB >= 30 GiB`), wall time `283.67s`.

## 3. Outcome and Key Metrics

### Table 1 — `DL-Val` (`Native91`, `91` Contexts / `19` Families, Seeds `254101–254103`, `10,000` Bootstrap Draws)

| Contrast / Arm | Median Family Utility | Median Context Utility | Median Family Gain | Per-Seed Family Gain (`254101`, `254102`, `254103`) | Family `95% CI` | Holm `97.5% CI` (`m=2`) | Raw `p` / `Holm p` | Pos Seeds | Family W/T/L (by seed) | `400`-Iter MM Elo vs `R882` (`95% CI`) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **`R882` (frozen anchor)** | `1.153615` | `1.137794` | `0.000000` | `[0.0, 0.0, 0.0]` | `[0.0, 0.0]` | `[0.0, 0.0]` | `—` | `—` | `—` | `1500.00` (`[1500.00, 1500.00]`) |
| **`GATED_025` (passive 4-reveal)** | `1.153270` | `1.137434` | `-0.000345` vs `R882` | `[-0.002987, -0.000345, -0.000232]` | `[-0.001035, 0.000000]` | `[-0.001035, 0.000000]` | `1.0000` | `0/3` | `0/18/1` (all 3) | `1496.34` (`[1492.64, 1500.00]`) |
| **`ACTIVE_2PLUS2_HURDLE` — PRIMARY vs `GATED_025`** | **`1.168790`** | **`1.153636`** | **`+0.015519`** | **`[+0.018161, +0.015519, +0.015407]`** | **`[+0.005326, +0.027345]`** | **`[+0.004015, +0.029097]`** | `0.0001` / **`0.0002`** | **`3/3`** | `9/8/2, 9/8/2, 10/7/2` | **`1529.31`** (`[1507.22, 1551.58]`) |
| **`ACTIVE_2PLUS2_HURDLE` — SECONDARY vs `R882`** | **`1.168790`** | **`1.153636`** | **`+0.015174`** | **`[+0.015174, +0.015174, +0.015174]`** | **`[+0.005196, +0.026915]`** | **`[+0.003785, +0.028457]`** | `0.0001` / **`0.0002`** | **`3/3`** | `9/8/2` (all 3) | **`1529.31`** (`+29.31` Elo vs `R882`) |

### Table 2 — `DL-Train` 5-Fold OOF (`1,219` Contexts / `205` Families, `fold_seed = 254140`)

| Contrast / Arm | Median Family Utility | Median Context Utility | Median Family Gain | Per-Seed Family Gain (`254101`, `254102`, `254103`) | Family `95% CI` | Holm `97.5% CI` (`m=2`) | Raw `p` / `Holm p` | Pos Seeds | Family W/T/L (by seed) | `400`-Iter MM Elo vs `R882` (`95% CI`) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **`R882` (frozen anchor)** | `1.385355` | `1.423073` | `0.000000` | `[0.0, 0.0, 0.0]` | `[0.0, 0.0]` | `[0.0, 0.0]` | `—` | `—` | `—` | `1500.00` (`[1500.00, 1500.00]`) |
| **`GATED_025` (passive 4-reveal)** | `1.385359` | `1.423077` | `+0.000004` vs `R882` | `[-0.005995, +0.000338, +0.000004]` | `[-0.000050, +0.000994]` | `[-0.000066, +0.001062]` | `0.4273` | `2/3` | `3/201/1, 3/201/1, 1/203/1` | `1500.57` (`[1499.77, 1501.70]`) |
| **`ACTIVE_2PLUS2_HURDLE` vs `GATED_025`** | **`1.467114`** | **`1.495450`** | **`+0.081421`** | **`[+0.088299, +0.081421, +0.080065]`** | **`[+0.041601, +0.138682]`** | **`[+0.038235, +0.148780]`** | `0.0001` / **`0.0002`** | **`3/3`** | `81/111/13, 81/113/11, 82/112/11` | **`1525.73`** (`[1519.52, 1532.48]`) |
| **`ACTIVE_2PLUS2_HURDLE` vs `R882`** | **`1.467114`** | **`1.495450`** | **`+0.081759`** | **`[+0.082304, +0.081759, +0.080069]`** | **`[+0.041666, +0.138944]`** | **`[+0.038024, +0.149158]`** | `0.0001` / **`0.0002`** | **`3/3`** | `81/112/12, 83/112/10, 82/113/10` | **`1525.73`** (`+25.73` Elo vs `R882`) |

---

## 4. Required Audits (`(a)-(d)` in [metrics.json](metrics.json))

1. **Audit (a) — Zero `DL-Val` / Holdout Leakage (`PASS`)**:
   - `val_contexts_in_prior_fit = 0`, `val_families_in_prior_fit = 0`, `val_contexts_in_promotion_fit = 0`, `val_families_in_promotion_fit = 0`.
   - `per_fold_train_oof_cid_overlap = [0, 0, 0, 0, 0]`, `per_fold_train_oof_family_overlap = [0, 0, 0, 0, 0]`, `meta_test_contexts_loaded = 0`.
   - Decomposed Hurdle OOF correlations (`fold_seed = 254140`): `DL-Train` OOF continuous `logratio` correlation `0.4971`, gate-failure correlation `0.2286`; `DL-Val` continuous `logratio` correlation `0.4750`, gate-failure correlation `0.2395`.
2. **Audit (b) — Equal 4-Reveal Budget (`PASS`)**:
   - Across all `273` `DL-Val` (`91 x 3`) and `3,657` `DL-Train` (`1,219 x 3`) context-seed evaluations (`3,930` total), both `ACTIVE_2PLUS2_HURDLE` (`2` Stage-A + `2` Stage-B disjoint reveals) and `GATED_025` used **exactly `4` unique reveals per context** (`min = 4`, `max = 4`, `stage_a_plus_stage_b_all_2plus2_disjoint = true`).
   - All `3,930/3,930` context-seed evaluations matched the unmodified snapshot `stage1_hurdle_sos_critic.score_arms_for_context` bit-for-bit.
3. **Audit (c) — Negative Control: Within-Context Permuted Revealed `fitness` (`PASS`)**:
   - Permuting revealed `fitness` within each context (`stream_seed("H254_NEGCTRL", seed, cid)`) while scoring the chosen candidate on true pool `utility` collapses the `DL-Val` gain of `ACTIVE_2PLUS2_HURDLE` to null:
     - `ACTIVE_2PLUS2_HURDLE (perm) - GATED_025`: median family gain `+0.001481`, **`95% CI [-0.008987, +0.010203]`** (**covers `0`**, `p = 0.4537`), per-seed `[+0.010405, -0.017981, +0.001481]`.
     - `ACTIVE_2PLUS2_HURDLE (perm) - GATED_025 (perm)`: median family gain `+0.004007`, **`95% CI [-0.008632, +0.016140]`** (**covers `0`**, `p = 0.3395`).
     - `ACTIVE_2PLUS2_HURDLE (perm) - R882`: median family gain `+0.001249`, **`95% CI [-0.009784, +0.009452]`** (**covers `0`**, `p = 0.5056`).
   - This confirms that the `+0.015519` `DL-Val` gain is driven by genuine informative feedback from the active `2+2` reveals on the top contenders (`b1, b2`), not by static prior shift or holdout leakage.
4. **Audit (d) — Descriptive `±50%` Constant Sensitivity (`36/36` Positive)**:
   - Varying each of the `18` `ACTIVE_2PLUS2_HURDLE` constants one at a time by `-50%` (`0.5x`) and `+50%` (`1.5x`) produces **`36/36` positive `DL-Val` median family gains and positive `95% LCB`s** vs both `GATED_025` (median gain range `[+0.009755, +0.016097]`) and `R882` (median gain range `[+0.009410, +0.015752]`). The largest sensitivity is on `direct_shrink_w` (`0.08 -> 0.04` reduces `DL-Val` gain from `+0.015519` to `+0.009755`, whereas `0.08 -> 0.12` yields `+0.015563`).

---

## 5. Verdict and Next Step

- **Verdict**: **`ADVANCE`**.
- **Next Step**: Promote Stage-1 `ACTIVE_2PLUS2_HURDLE` (sequential `2+2` active contender reveal + cross-fitted Decomposed Continuous Hurdle prior + direct revealed-slot shrinkage under the 4-reveal budget) as a confirmed Stage-1 inference/reveal component (`+0.015519` family utility / `+32.97` Elo over passive 4-reveal `GATED_025` and `+0.015174` family utility / `+29.31` Elo over `R882` on `DL-Val`), while keeping `round-882.pt` and `DOPE` (`c51a56d0...`) frozen until the `MAIN HOLD` lifts and downstream integration is reviewed.

## 6. S3 Custody

- **Preregistration S3 Manifest** (uploaded before any outcome):
  - `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H254/prereg-e77ab29e2debcc74850aa60d7ea79a6dc764ac0f/prereg_s3_manifest.json` (`VersionId = fyfEwHKOy2bFdiO.VRpiyPWFfs2QbRTf`, `sha256 = 3082e112ae4fe6e96c1c7d203f8b064e8b6db579505aa477d7ce48baad793c7a`)
- **Closeout Run (`attempt-001`) S3 Manifest**:
  - `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H254/attempt-001/s3_manifest.json` (`VersionId = 2HmNPWjZLGk4IZX_RHgkU_p3mkGkKRaT`, `sha256 = db7c9ca26a882b969e05e5778ed0d2fc256ae4a048f2607faaae5d1c2bae78fc`)

## Red-team caveats (watcher 14:05Z)

- Revealed `fitness` IS the scored quantity (`utility = -fitness`, `common_h252.py:137`). `ACTIVE_2PLUS2_HURDLE` is a test-time 4-true-evaluation search policy, NOT adapter/checkpoint lift; the vs-`R882` secondary compares 4 oracle reveals vs 0 reveals (unequal budget).
- `GATED_025` reveals 4 RANDOM candidates, a weak equal-budget comparator (Elo `1496.34` < `R882` `1500.00`). The obvious strong equal-budget baseline "reveal `r882` top-4 by cost, pick best revealed" (`TOP4_PICK`) was never run. Verdict stands as preregistered (`ADVANCE` vs `GATED_025`) but adoption is **BLOCKED pending H258**.
- Seeds only vary the reveal RNG on one fixed model; `3/3` seeds is weak evidence of robustness. `19` `DL-Val` families; W/T/L `9/8/2`.
