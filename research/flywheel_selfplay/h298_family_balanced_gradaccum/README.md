# H298 (`h298_family_balanced_gradaccum`) — Family-Balanced Sampling + 4x Gradient Accumulation

- **Wave ID**: `H298`
- **Slug**: `h298_family_balanced_gradaccum`
- **Owner**: `rd_argon` (`[ARGON]`)
- **Verdict**: **`NOT_ADVANCED`** (`0/3` seeds pass the strict `DEV_ADVANCE` gate)
- **Preregistration Commit**: `cddee6c8dacea9890c10e1f9dcf73b3975507d12` (pushed to `origin/v1/next` at `2026-11-04T02:52:56Z`, before any H298 arm was scored)
- **Execution Host**: `xbabe1` (`CUDA_VISIBLE_DEVICES="" taskset -c 64-67 nice -n 19`, `OMP_NUM_THREADS=4`)

## 1. Step 1 Regression Verification (`PASS`)

Executed on an isolated copy at `xbabe1:/home/ubuntu/jope-data/H298-rd_argon/` before any H298 arm ([`regression_receipt.json`](regression_receipt.json)):

- `round-3175.pt` SHA256: `1d6ea900ebed8d2a05c3b3f37576adc15aadd2b53c585bfd5cb71a103e9ff97b`
- `selected.json` SHA256: `981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1` (`1,219` `TRAIN` contexts across `205` families; `91` `dev` contexts across `19` families)
- Frozen DOPE SHA256: `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` (`29,211` adapter trainable parameters only; frozen state SHA256 `d4dde909145d9da75ec74fd61b9827d8fc2ec9236505b7cdf72b4161994551b7`)
- `dev91` evaluation identity: `e79a7715c1531ccb8c4055ab96971101cd4034ec3a71d83bca6d25b0e604ca83`
- **`r3175` replay**: **`1.1389572403748425`** (`91/91` program choices exact match)
- **`t1845` control seed `184501` 2-update replay**: **`1.1354387884248622`** (`round(u, 10) == 1.1354387884`, `91/91` choices exact match, `adapter_sha256 == fd690ea216af7e93fe246e849e9f66a876d23306492cca3ad6866c57e28d56f1`)

## 2. Terminal `dev91` Results (`seeds 29801–29803`, `8` `TRAIN`-Only Updates/Seed)

All arms started from identical `r3175` adapter/optimizer copies with paired v2 warm-restart perturbations per seed (`29801`: $\sigma=8.133\times 10^{-5}, \text{lr}=2.195\times 10^{-4}$; `29802`: $\sigma=1.500\times 10^{-4}, \text{lr}=3.551\times 10^{-5}$; `29803`: $\sigma=1.925\times 10^{-5}, \text{lr}=2.867\times 10^{-4}$), trained for `8` `TRAIN`-only updates (`SmoothL1 + 0.1 KL`, microbatch `32`, `clip_grad_norm_(1.0)`, `dope_freeze_guard` verified every step), and froze each terminal checkpoint before touching `dev91` once.

| Arm | Seed | `dev91` Mean | Lift vs `r3175` (`1.138957`) | 19-Fam `10k` Boot `95%` CI vs `r3175` | Lift vs `PAIRED_CONTROL` | `W/T/L` vs `r3175` (`chg`) | `W/T/L` vs `CTRL` (`chg`) | Top-4 Pos Share (`n_pos`) | MM BT Elo (`r3175=1500`) `[95% CI]` | Seed Gate (`3/3` req) |
| --- | ---: | ---: | ---: | --- | ---: | --- | --- | ---: | --- | --- |
| **`FAMBAL_ACC4`** (Primary, `32,768` pres/step) | `29801` | `1.1054287815` | `-0.0335284589` | `[-0.098139, +0.003503]` | **`+0.0033018428`** | `6/75/10` (`16`) | `2/88/1` (`3`) | `98.96%` (`6`) | `1484.72` `[1455.89, 1511.46]` | `FAIL` |
| **`FAMBAL_ACC4`** (Primary, `32,768` pres/step) | `29802` | `1.1273007477` | `-0.0116564927` | `[-0.025467, -0.000458]` | `+0.0000000000` | `3/80/8` (`11`) | `0/91/0` (`0`) | `100.00%` (`3`) | `1480.90` `[1459.60, 1500.00]` | `FAIL` |
| **`FAMBAL_ACC4`** (Primary, `32,768` pres/step) | `29803` | `1.0766830748` | `-0.0622741656` | `[-0.132187, -0.011068]` | **`+0.0002791675`** | `7/65/19` (`26`) | `1/90/0` (`1`) | `96.11%` (`7`) | `1453.93` `[1410.30, 1492.36]` | `FAIL` |
| `FAMBAL_ACC1` (Secondary, `8,192` pres/step) | `29801` | `1.1054287815` | `-0.0335284589` | `[-0.098139, +0.003503]` | `+0.0033018428` | `6/75/10` (`16`) | `2/88/1` (`3`) | `98.96%` (`6`) | `1484.72` `[1455.89, 1511.46]` | `FAIL` |
| `FAMBAL_ACC1` (Secondary, `8,192` pres/step) | `29802` | `1.1273007477` | `-0.0116564927` | `[-0.025467, -0.000458]` | `+0.0000000000` | `3/80/8` (`11`) | `0/91/0` (`0`) | `100.00%` (`3`) | `1480.90` `[1459.60, 1500.00]` | `FAIL` |
| `FAMBAL_ACC1` (Secondary, `8,192` pres/step) | `29803` | `1.0766830748` | `-0.0622741656` | `[-0.132187, -0.011068]` | `+0.0002791675` | `7/65/19` (`26`) | `1/90/0` (`1`) | `96.11%` (`7`) | `1453.93` `[1410.30, 1492.36]` | `FAIL` |
| `PAIRED_CONTROL` (`reporter.training_order`) | `29801` | `1.1021269386` | `-0.0368303017` | `[-0.102607, +0.002992]` | `0.0000000000` | `7/73/11` (`18`) | `0/91/0` (`0`) | `98.01%` (`7`) | `1484.72` `[1455.89, 1511.46]` | `FAIL` |
| `PAIRED_CONTROL` (`reporter.training_order`) | `29802` | `1.1273007477` | `-0.0116564927` | `[-0.025467, -0.000458]` | `0.0000000000` | `3/80/8` (`11`) | `0/91/0` (`0`) | `100.00%` (`3`) | `1480.90` `[1459.60, 1500.00]` | `FAIL` |
| `PAIRED_CONTROL` (`reporter.training_order`) | `29803` | `1.0764039073` | `-0.0625533331` | `[-0.132808, -0.011119]` | `0.0000000000` | `7/64/20` (`27`) | `0/91/0` (`0`) | `96.11%` (`7`) | `1450.04` `[1406.17, 1492.01]` | `FAIL` |

### Summary Across Seeds (`FAMBAL_ACC4`)

- **Median-of-seeds `dev91` mean**: `1.1054287815` (mean across seeds `1.1031375347`)
- **Median-of-seeds lift vs `r3175`**: **`-0.0335284589`** (median seed 19-family `10k` bootstrap `95%` CI `[-0.0981389124, +0.0035028817]`)
- **Median-of-seeds lift vs `PAIRED_CONTROL`**: **`+0.0002791675`** (`+0.0033018428` on `29801`, `0.0000000000` on `29802`, `+0.0002791675` on `29803`)
- **Strict Gate Verdict**: **`NOT_ADVANCED`** (`0/3` seeds beat `r3175 + 0.001 = 1.1399572404`, `0/3` seeds have `LCB95 > 0` vs `r3175`, `0/3` seeds have top-4 positive gain share `< 80%`).

## 3. Diagnostic Findings

1. **Family-balanced sampling strictly dominates or ties `t1845` baseline sampling (`PAIRED_CONTROL`) on `3/3` seeds**: Removing the `42x` presentation skew across the `205` `TRAIN` families (`252` vs `6` presentations/step in `reporter.training_order`) prevents `1–2` context regressions on seeds with larger learning rates (`29801`: `1.105429` vs `1.102127`, `+0.003302`, `2W/88T/1L`; `29803`: `1.076683` vs `1.076404`, `+0.000279`, `1W/90T/0L`).
2. **`1x` (`8,192` presentations) already saturates the family-balanced batch gradient**: `FAMBAL_ACC1` (`8,192` presentations/step, `~40` draws per `TRAIN` family) and `FAMBAL_ACC4` (`32,768` presentations/step, `~160` draws per `TRAIN` family) produce distinct adapter parameter SHA256s (e.g. `c0184a7d...` vs `8baeccff...` on `29803`) and slightly lower gradient norms (`0.4932` vs `0.4989`), yet induce **zero argmin selection differences (`91/91` identical `dev91` choices on all 3 seeds)**.
3. **Why `8` unconstrained `TRAIN` updates regress vs `r3175`**: Even with zero family-size bias and `4x` gradient accumulation, `8` unconstrained `SmoothL1 + 0.1 KL` updates from `r3175` flip `11–26` `dev91` argmin choices (`6W/10L`, `3W/8L`, `7W/19L`), and positive gains remain concentrated in `3–7` contexts (`top-4 share 96.1%–100.0%`).

## 4. S3 Custody & Cleanup Receipt

All 9 terminal `.pt` checkpoints, per-arm JSONs, `events_h298.jsonl`, `preregistered_batches.json`, and `regression_receipt.json` are archived under `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H298/` ([`s3_manifest.json`](s3_manifest.json)), verified via `head-object` + independent `get-object` SHA256 readback, and `/home/ubuntu/jope-data/H298-rd_argon/` on `xbabe1` has been deleted:

- `h298_checkpoints_and_results.tar.gz`: `VersionId = lG1UYi3tUz3vNfkFllycDAWzTvOLIbOE` (`41,324,918` bytes, `sha256 = 95494ce2efc99eead40e5f40d503aa750f8783e1b6f5cb472bfcdb8d5f72865a`)
- `metrics_h298.json`: `VersionId = hn.QqQDNDME9X1KtQ5xwUckbS26dOlGM` (`27,768` bytes, `sha256 = 0af56f66fdfa7fd7a0b94441efe324fbdb75f0f2576c42354f5428d1a097ec90`)
- `s3_manifest.json`: `VersionId = 6qFHIFeiLNUQNWqxCbcVNkKYJiAeJyvx` (`4,737` bytes, `sha256 = b9e0a80bc733146110f2cc33d272ab48de3ccf91f716584334987e62308d44cc`)
