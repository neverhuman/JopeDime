# H151 — Lookahead vs drift-matched AdamW step-size controls on the r882 continuation

**Status: complete. Verdict: `STEP_SIZE_ONLY` (`mechanism_verdict: STEP_SIZE_ONLY`, `r882_verdict: NO_ADVANCE`)**
(preregistered in [PREREG.md](PREREG.md), PR [#1295](https://github.com/neverhuman/JopeDime/pull/1295), merged source `36ec1f60e41f3ed31fe4c018eda0571ed5596c07`; single 480-update CPU run on `xbabe1`).

## Summary

- **Mechanism test (`LA` vs drift-matched AdamW step-size control): `STEP_SIZE_ONLY`.**
  - Halving the AdamW learning rate (`CTRL_HALF`, `lr = 5e-5`) matches Lookahead's (`LA`, `lr = 1e-4, k = 5, alpha = 0.5`) median adapter L2 drift to within **0.00782%** (`0.218146` vs `0.218129`, relative gap `7.82e-5 <= 0.15`, inside the preregistered inclusive `[0.85 * LA, 1.15 * LA] = [0.185409, 0.250848]` band; `DRIFT_MATCH_VALID`).
  - Although `LA` and `CTRL_HALF` produce distinct final adapter weights (`adapter_sha256` differs on every seed), they make **identical argmin candidate selections on all 91/91 `DL-Val` contexts across all 3 seeds** (`choice_diffs_out_of_91_by_seed = [0, 0, 0]`).
  - Consequently, the primary `LA` minus drift-matched `CTRL_HALF` contrast on 91-context `DL-Val` (19 families, equal-family mean within seed, median over 3 seeds) is **exact zero**: **`0.0000` (`95% CI [0.0000, 0.0000]`, `p_one_sided = 1.0`, `0/3` seeds beating `CTRL_HALF`, Elo `1500.0`)**.
  - This confirms that `H146`'s `+0.0141` Lookahead advantage over full-step `CTRL` (`lr = 1e-4`) was **100% due to 50% step-size / L2-drift damping (`alpha = 0.5`)**, not slow-weight trajectory averaging.
- **Advancement vs `r882` (`Holm m = 4` across all 4 executed arms): `NO_ADVANCE`.**
  - Every arm degrades below frozen `r882` (`91/91` exact baseline `1.1377941898`, equal-family mean `1.1536153`, `0` invalid picks) on all `3/3` seeds (`holm_adjusted_p = 1.0` for all 4 arms).
  - Quarter-step `CTRL_QTR` (`lr = 2.5e-5`, median drift `0.108550`) reduces continuation harm further (`-0.0070` vs `r882`, `+0.0288` vs `CTRL`) and induces cross-seed argmin pseudoreplication on `DL-Val`, confirming the quantized argmin plateau around `r882`.

## Preregistered Design

- **Arms** (40 steps, batch 16, slate 192, grad clip `1.0`, SmoothL1 $\beta = 0.05$ + $0.1 \times$ listwise KL $\tau = 0.10$, `weight_decay = 1e-4` on the 47 trainable adapter tensors / 29,211 parameters from `round-882.pt`):
  - `CTRL`: live AdamW (`lr = 1e-4`, `lookahead_k = null`).
  - `CTRL_HALF`: half-step AdamW (`lr = 5e-5`, `lookahead_k = null`).
  - `CTRL_QTR`: quarter-step AdamW (`lr = 2.5e-5`, `lookahead_k = null`).
  - `LA`: Lookahead (`lr = 1e-4`, `lookahead_k = 5`, `alpha = 0.5`, slow-weight pullbacks at steps `5, 10, ..., 40`).
- **Seeds & Folds**: paired batch plans across all 4 arms for seeds `151101, 151102, 151103`; 5-fold family split of 1,219 `DL-Train` contexts (`fold_seed = 151140`), fitting on folds 1–4 (`974` contexts / `164` families) and holding out fold 0 (`245` contexts / `41` families) as descriptive secondary only.
- **Primary Evaluation**: 91-context `DL-Val` (`19` families), equal-family mean within seed, median across 3 seeds, 10,000-draw shared family-cluster bootstrap (`boot_seed = 151141`, strict `ties = "reject"`), inclusive 15% drift-matching gate (`0.85 * LA <= CTRL_* <= 1.15 * LA`), and `Holm(m = 4)` across all 4 executed arms vs `r882`.

## Primary Results (`DL-Val`, 91 contexts / 19 families)

**Baseline `r882` gate:** `PASS` (`91/91` exact context matches, `max_context_error = 0.0`, context mean `1.1377941898`, equal-family mean `1.1536153`, `0` invalid picks).

| Arm | Median L2 Drift (by seed) | Median Gain vs `r882` [95% CI] | Seeds > `r882` | Holm $p$ vs `r882` | Gain vs `CTRL` (`1e-4`) [95% CI] | Gain vs Matched `CTRL_HALF` (`5e-5`) [95% CI] | Elo vs `r882 = 1500` [95% CI] | Invalid Picks |
|---|---|---|---|---|---|---|---|---|
| `CTRL` (`1e-4`) | `0.4405` (`0.458 / 0.439 / 0.441`) | `-0.0358` `[-0.0969, -0.0008]` | `0/3` | `1.000` | `0.0000` `[0.0000, 0.0000]` | `-0.0155` `[-0.0302, +0.0003]` (`0/3`) | `1406.5` `[1270.7, 1536.7]` | `1 / 0 / 1` |
| `CTRL_HALF` (`5e-5`) | **`0.2181`** (`0.226 / 0.218 / 0.218`) | `-0.0097` `[-0.0228, +0.0006]` | `0/3` | `1.000` | `+0.0155` `[-0.0003, +0.0302]` (`3/3`) | `0.0000` `[0.0000, 0.0000]` | `1463.3` `[1365.8, 1555.3]` | `1 / 0 / 0` |
| `CTRL_QTR` (`2.5e-5`) | `0.1086` (`0.112 / 0.109 / 0.108`) | `-0.0070` `[-0.0169, +0.0009]` | `0/3` | `1.000` | `+0.0288` `[-0.0003, +0.0814]` (`3/3`) | `+0.0027` `[-0.00003, +0.0079]` (`3/3`) | `1463.3` `[1365.8, 1555.3]` | `0 / 0 / 0` |
| **`LA` (`k=5, α=0.5`)** | **`0.2181`** (`0.226 / 0.218 / 0.218`) | **`-0.0097` `[-0.0228, +0.0006]`** | **`0/3`** | **`1.000`** | **`+0.0155` `[-0.0003, +0.0302]` (`3/3`)** | **`0.0000` `[0.0000, 0.0000]` (`0/3`)** | **`1463.3` `[1365.8, 1555.3]`** | **`1 / 0 / 0`** |

## Secondary Descriptive Readout (`DL-Train` Fold 0, 245 contexts / 41 families)

- **vs `r882` (`4` invalid picks at baseline)**:
  - `CTRL`: `-0.0195` `[-0.0663, +0.0161]` (`0/3` seeds, invalid `4 / 5 / 5`)
  - `CTRL_HALF`: `-0.0177` `[-0.0608, +0.0131]` (`0/3` seeds, invalid `5 / 4 / 5`)
  - `LA`: `-0.0177` `[-0.0608, +0.0132]` (`0/3` seeds, invalid `5 / 4 / 5`)
  - `CTRL_QTR`: `+0.0073` `[-0.0008, +0.0221]` (`3/3` seeds, `p = 0.1074`, invalid `4 / 4 / 4`)
- **`LA` minus `CTRL_HALF` on Fold 0**: per-seed gains `[+0.000270, -0.000130, 0.000000]`, median `0.0000` `[0.0000, 0.0000]`, confirming that `LA` and `CTRL_HALF` have distinct weights (`adapter_sha256` `b0f96260...` vs `f01e2191...`, `dccc482b...` vs `ab280873...`, `8149990a...` vs `17976307...`) whose trajectories remain virtually indistinguishable in ranking choices.

## Execution, Resource Bounds & Custody

- **Single Execution Attempt (`xbabe1` CPU)**:
  - Merged source commit: `36ec1f60e41f3ed31fe4c018eda0571ed5596c07` (`manifest_sha256 = 5d8daf3248ae068a2ea71b71a2afd5af4d619e1ec65e484846849b20e649eaec`).
  - Total optimizer updates: `480` (`12` cells $\times$ `40` steps, `0` retries, `0` pauses).
  - Wall clock: `1,449.67 s` (`24.16 min`) at `nice 19`, `CUDA_VISIBLE_DEVICES=""` (`torch.cuda.is_available() == False`).
  - Supervisor resource samples (`4,894` samples): max OS threads **`3`** (cap `<= 4`), peak RSS **`3,332,915,200 B`** (`3.33 GB`, cap `< 6.0 GB`), min free disk `112.9 GB` (floor `>= 40 GB`), max scratch `10.56 MB` (cap `< 3.0 GB`).
  - Frozen backbone & DOPE weight byte-identity (`frozen_sha256 = 8d1070faf574ee1cd2a4e630887e1662b43189b3f89b04982143ef2704818aae`, `dope_sha256 = c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`) verified before and after every cell; `0` protected/outer-test reads (`audit.json`).
- **Custody & Cleanup**:
  - S3 archive (41 files restore-verified byte-for-byte): `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H151/argon-h151-20261101a/`
  - Local custody: `/home/ubuntu/jope-rescue-20260921/h151-rd_argon-20261101/` and `/home/ubuntu/jope_plan/evidence/H151/`
  - Remote scratch `/home/ubuntu/h151-exec` and `/home/ubuntu/jope-data/h151` on `xbabe1` deleted after S3 restore verification.
