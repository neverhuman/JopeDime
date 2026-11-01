# H172 — Anchored Convex Last-Layer Refit (`L-BFGS`) + Family Bagging on Frozen `r882` Penultimate Features

- **Wave**: `H172`
- **Owner**: `gpu_tokbench_agy`
- **Authority**: [`/home/ubuntu/nudges/hwave_t0225_h172_tokbench_agy.md`](/home/ubuntu/nudges/hwave_t0225_h172_tokbench_agy.md)
- **Claim Receipt**: [`/home/ubuntu/jope_plan/ns/receipts/H172_claim.txt`](/home/ubuntu/jope_plan/ns/receipts/H172_claim.txt)
- **Preregistration Specification**: [`PREREG.md`](PREREG.md) (merged in PR [#1470](https://github.com/neverhuman/JopeDime/pull/1470); zero-update preflight fixes in PR [#1476](https://github.com/neverhuman/JopeDime/pull/1476) and PR [#1482](https://github.com/neverhuman/JopeDime/pull/1482) at source commit `dd42251e509f0ff6ee5041fa4c83db75881fa9bf`)
- **Status**: **INCOMPLETE — NO_ADVANCE (RSS GUARDRAIL STOP AT `2/15` CELLS / `144` CONVERGED `L-BFGS` SOLVES; `0/2` FINITE $\lambda$ SELECTIONS; NO RERUN AFTER OPTIMIZER UPDATES)**
- **S3 Custody**: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H172/gpu_tokbench_agy-h172-20261101/` (`35` objects: `23` run artifacts + `12` custody artifacts; local artifacts also preserved in `/home/ubuntu/jope-data/h172/run/` and `/home/ubuntu/jope_plan/evidence/H172/`)

---

## 1. Execution & Guardrail Stop Summary

1. **91/91 `DL-Val` Baseline Gate & Native Raw-Cost Bit-Identity**:
   - Before any `L-BFGS` solves, `run.py` verified exact `91/91` `DL-Val` utility parity (`max_context_error = 0.0`, `mean_error = 0.0`, `mean = 1.1377941898164525`) and `torch.equal` native step-0 raw-cost bit-identity across all `91` `DL-Val` contexts (`raw_cost_bit_exact_91_91 = True`), plus exact `07:26Z` `None`-grad handling on `adapter_10tok.parent_delta_net.3.{weight,bias}`.
2. **Completed Cells (`2 / 15`: `A0-172101` and `A0-172102`, `144` Converged `float64` `L-BFGS` Solves)**:
   - Outer fold `A0` (`fit = 970` contexts across `164` families; `assess = 249` contexts across `41` families) completed both `seed = 172101` (`75` cumulative `L-BFGS` solves, `54.74 s`, `max_icv_grad_norm = 2.68e-7 <= 1e-6`, `max_shuf_grad_norm = 3.52e-7 <= 1e-6`) and `seed = 172102` (`144` cumulative `L-BFGS` solves, `55.22 s`, `max_icv_grad_norm = 2.65e-7 <= 1e-6`, `max_shuf_grad_norm = 3.53e-7 <= 1e-6`).
3. **Host Guardrail Stop During Cell 3 (`A0-172103`) & No-Rerun Enforcement**:
   - During cell `A0-172103`, `monitor.py` recorded `aggregate_rss_bytes = 6,466,510,848 >= 6,442,450,944` (`6.022 GiB >= 6.000 GiB` ceiling; `peak_total_threads = 3 <= 4`, `gpu_lock_held = False`) and cleanly terminated the child with `SIGTERM` (`exit_code = -15`, `status = INCOMPLETE`).
   - Per the hard preregistered rule (*"A preflight failure before optimizer updates may be fixed once with an explicit receipt; do not rerun after optimizer updates. ... Incomplete: preserve artifacts, open a partial-status PR, update AGENT_CHAT.md, and stop"*), no rerun was performed after optimizer updates (`lbfgs_solves = 144 > 0`). All partial artifacts (`baseline.json`, `A0-172101.json`, `A0-172101-selection.json`, `A0-172102.json`, `A0-172102-selection.json`, and 18 `.npz` head snapshots) were preserved locally and synced to S3.

---

## 2. Inner-CV $\lambda$-Path Results on Completed Cells (`A0-172101` & `A0-172102`)

Across both completed cells on the `164`-family `fit` split (`970` contexts), every finite anchor strength $\lambda \in \{10^4, 10^3, 10^2, 10^1, 1.0, 10^{-1}\}$ produced **non-positive** (`<= 0.0`) 5-fold inner family CV equal-family utility gain vs `r882`, degrading monotonically as $\lambda$ decreased from $10^4$ to $10^{-1}$ and tracking the within-family target-permuted placebo (`REFIT_SHUF`):

| $\lambda$ | `A0-172101` `REFIT_CV` Inner-CV $\Delta U_{\text{fam}}$ `[95% CI]` ($p$) | `A0-172102` `REFIT_CV` Inner-CV $\Delta U_{\text{fam}}$ `[95% CI]` ($p$) | `A0-172101` `REFIT_SHUF` Inner-CV $\Delta U_{\text{fam}}$ `[95% CI]` ($p$) | `A0-172102` `REFIT_SHUF` Inner-CV $\Delta U_{\text{fam}}$ `[95% CI]` ($p$) |
| --- | --- | --- | --- | --- |
| `inf` (`r882`) | `+0.000000` `[+0.000000, +0.000000]` (`1.0000`) | `+0.000000` `[+0.000000, +0.000000]` (`1.0000`) | `+0.000000` `[+0.000000, +0.000000]` (`1.0000`) | `+0.000000` `[+0.000000, +0.000000]` (`1.0000`) |
| `1e4` | `+0.000000` `[+0.000000, +0.000000]` (`1.0000`) | `+0.000000` `[+0.000000, +0.000000]` (`1.0000`) | `+0.000000` `[+0.000000, +0.000000]` (`1.0000`) | `+0.000000` `[+0.000000, +0.000000]` (`1.0000`) |
| `1e3` | `+0.000000` `[+0.000000, +0.000000]` (`1.0000`) | `+0.000000` `[+0.000000, +0.000000]` (`1.0000`) | `+0.000000` `[+0.000000, +0.000000]` (`1.0000`) | `+0.000000` `[+0.000000, +0.000000]` (`1.0000`) |
| `1e2` | `-0.000212` `[-0.000561, +0.000028]` (`0.9221`) | `-0.000212` `[-0.000565, +0.000028]` (`0.9201`) | `-0.000221` `[-0.000577, +0.000019]` (`0.9331`) | `-0.000221` `[-0.000570, +0.000019]` (`0.9321`) |
| `1e1` | `-0.002998` `[-0.009374, +0.000412]` (`0.7912`) | `-0.002997` `[-0.009366, +0.000428]` (`0.7872`) | `-0.003055` `[-0.010267, +0.001265]` (`0.8342`) | `-0.003055` `[-0.009972, +0.001142]` (`0.8322`) |
| `1` | `-0.008348` `[-0.031716, +0.015406]` (`0.7453`) | `-0.007574` `[-0.030540, +0.014921]` (`0.7323`) | `-0.007985` `[-0.031587, +0.015736]` (`0.7393`) | `-0.007656` `[-0.030276, +0.014118]` (`0.7443`) |
| `1e-1` | `-0.035830` `[-0.077441, +0.001104]` (`0.9710`) | `-0.037387` `[-0.078797, +0.000276]` (`0.9730`) | `-0.051799` `[-0.097284, -0.012449]` (`0.9930`) | `-0.051623` `[-0.092711, -0.013362]` (`0.9960`) |

- Both `strict` and `unconstrained` inner-CV rules selected $\lambda = \infty$ (`0.0` head $L_2$ drift) for `REFIT_CV`, `REFIT_BAG`, and `REFIT_SHUF` in both `A0-172101` and `A0-172102`.
- Under `best_finite_lambda` ($\lambda = 10^4$), the refit head $L_2$ drift was `8.28e-6` (`REFIT_CV`), `8.19e-6` / `8.36e-6` (`REFIT_BAG` 8-bag mean head), and `1.16e-5` (`REFIT_SHUF`), with zero argmin shift on outer fold `A0`.

---

## 3. Disclosures

- **Historical exposure**: `r882` was trained on `H145` `TRAIN` (`historical_r882_train_exposure = True`).
- **No speed claims, no AUC, no serving promotion**: `serving_promotion = False`, `gpu_lock_held = False`.
