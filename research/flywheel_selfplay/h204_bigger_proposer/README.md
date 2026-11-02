# H204: Bigger Proposer (`128 -> 256 -> 512`) with Research-Only `DOPE` Unfreeze — Closeout (`NO_ADVANCE`)

- **Wave**: `H204`
- **Verdict**: **`NO_ADVANCE`** (Stage-1 `W128` gate did not pass; Stage-2 widths `W256` and `W512` were **not** launched per the preregistered gate rule; `0%` selector adoption -> `100%` safe fallback to `r882`)
- **Owner**: `gpu_tokbench_agy` (`[TOKBENCH-AGY]`)
- **Host / GPU**: `xbabe2` (`NVIDIA GeForce RTX 3090`, `cuda:0` — **confirmed released**: `0%` util, `1 MiB / 24576 MiB`, `0` compute processes)
- **PR**: `#1729` (`rd/h204-bigger-proposer`)
- **Authority**: `"Jepson via Grok Bot, card answer 8:22 PM MT" (2026-11-01, t11s3): "Start H205 and H204 now with those unblocks; H206 after xbabe3 disk is cleared."`
- **Evaluator Contract**: `fixed-dev-cost-argmin-negative-fitness/v1` (pinned `round-882.pt` SHA-256 `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`)
- **S3 Evidence Prefix**: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261101/H204/`
- **Primary Repaired Stage-1 GPU Run**: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261101/H204/run-20261102T0511Z/` (local: `/home/ubuntu/jope-data/h204/run-20261102T0511Z`, PID `2843546`, `528.1s` wall time, `36` resource monitor snapshots with `0` violations)

---

## 1. Stage-0 Baseline Verification vs. `r882`

Before evaluating any Stage-1 arm, native `r882` (`round-882.pt`) was verified on the 91 `DL-Val` contexts (`19` families):

| Metric | Expected (`r882` Reference) | Measured (`Stage-0` on `xbabe2`) | Status |
|---|---:|---:|---|
| `DL-Val` Contexts / Families | `91 / 19` | `91 / 19` | **PASS** |
| Exact Top-1 Pick Matches | `91 / 91` | `91 / 91` | **PASS** |
| Pick Vector SHA-256 | `460560e44b8d...0c3624d96` | `460560e44b8d...0c3624d96` | **PASS** |
| Utility Vector SHA-256 | `77fd7f596390...0ebfec8e9` | `77fd7f596390...0ebfec8e9` | **PASS** |
| Native `r882` Mean Utility (`-fitness`) | `1.1377941898164525` | `1.1377941898164525` (`abs_diff = 0.0`) | **PASS** |
| Step-0 Live Token Bridge Discrepancy | `<= 1e-5` | `0.000000e+00` (`CPU` & `cuda:0`) | **PASS** |

---

## 2. Stage-1 Key Numbers (`DL-Val` 91 Contexts / 19 Families & `OOF Dev` 1,219 Contexts / 205 Families)

All three Stage-1 arms were trained across `3` preregistered seeds (`204101`, `204102`, `204103`) $\times$ `5` family-stratified CV folds (`15` folds/arm, `15` epochs/fold, batch size `32`, `K=16` exact two-fold Rust CV labeled candidates per context, `run_outer_test=False`):

### Pooled Stage-1 Summary Table (vs. Native `r882` = `1.1377941898`)

| Arm | Width | `DOPE` Policy (`#1726` Guard) | OOF Dev Mean $\Delta$ (`205` fam) | Raw `DL-Val` Mean Utility | Raw `DL-Val` Pooled $\Delta$ vs `r882` | Raw `DL-Val` 95% Family Bootstrap CI | Pooled Elo Lift | Raw One-Sided $p$ | Holm Adjusted $p$ | Positive Seeds | H178 Selector Adoption | Deployed Mean Utility (with `r882` Fallback) |
|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| **`r882` (Baseline)** | `64` | Frozen (`eval` mode) | `+0.000000` | `1.137794` | `+0.000000` | `[+0.000000, +0.000000]` | `+0.00` | — | — | — | — | `1.137794` |
| **`R882_CONT` (Control 1)** | `64` | Frozen (`0` trainable params, `0/15` folds changed) | `-0.250659` | `0.832602` | `-0.305192` | `[-0.408008, -0.175471]` | `-51.31` | `1.0000` | `1.0000` | `0 / 3` | **`0.0%` (`0/15` folds)** | **`1.137794`** (`100%` fallback) |
| **`H180_RAND` (Control 2)** | `128` | Frozen `DOPE` + frozen trunk (`0` trainable `DOPE` params) | `-0.067422` | `1.086400` | `-0.051395` | `[-0.090652, -0.013148]` | `-38.38` | `0.9985` | `1.0000` | `0 / 3` | **`0.0%` (`0/15` folds)** | **`1.137794`** (`100%` fallback) |
| **`PROPOSER_W128` (Treatment)** | `128` | Research unfreeze (`router` + `col_gate_head`, `444,168` params, `eval` mode, `15/15` folds updated) | `-0.201733` | `0.876864` | `-0.260930` | `[-0.455367, -0.079643]` | `-59.14` | `1.0000` | `1.0000` | `0 / 3` | **`0.0%` (`0/15` folds)** | **`1.137794`** (`100%` fallback) |

### Per-Seed Breakdown (`204101`, `204102`, `204103`)

| Arm | Seed | OOF Dev Mean $\Delta$ (`1,219` ctx) | OOF Dev 95% CI | Raw `DL-Val` Mean $\Delta$ (`91` ctx) | Raw `DL-Val` 95% CI | `DL-Val` Elo Lift | Worst Regression | `max_dope_data_grad_l2` | H178 Admitted / Fallback |
|---|---:|---:|---|---:|---|---:|---:|---:|---|
| `R882_CONT` | `204101` | `-0.234343` | `[-0.269091, -0.151299]` | `-0.324394` | `[-0.438596, -0.179218]` | `-57.80` | `3.2244` | `0.0000e+00` | `False` (`100%` fallback) |
| `R882_CONT` | `204102` | `-0.259550` | `[-0.330967, -0.187766]` | `-0.347374` | `[-0.475041, -0.192750]` | `-57.80` | `3.2699` | `0.0000e+00` | `False` (`100%` fallback) |
| `R882_CONT` | `204103` | `-0.258085` | `[-0.326510, -0.175666]` | `-0.243808` | `[-0.367927, -0.107759]` | `-38.33` | `3.1174` | `0.0000e+00` | `False` (`100%` fallback) |
| `H180_RAND` | `204101` | `-0.065583` | `[-0.109783, -0.034782]` | `-0.087706` | `[-0.167230, -0.016667]` | `-34.47` | `2.6051` | `0.0000e+00` | `False` (`100%` fallback) |
| `H180_RAND` | `204102` | `-0.075263` | `[-0.127426, -0.048687]` | `-0.050199` | `[-0.084884, -0.016643]` | `-53.88` | `1.0247` | `0.0000e+00` | `False` (`100%` fallback) |
| `H180_RAND` | `204103` | `-0.061421` | `[-0.096927, -0.026094]` | `-0.016278` | `[-0.039449, +0.008887]` | `-26.78` | `0.6816` | `0.0000e+00` | `False` (`100%` fallback) |
| `PROPOSER_W128` | `204101` | `-0.199343` | `[-0.270419, -0.153907]` | `-0.276408` | `[-0.482390, -0.087986]` | `-69.64` | `4.4694` | `2.0104e+01` | `False` (`100%` fallback) |
| `PROPOSER_W128` | `204102` | `-0.204708` | `[-0.272063, -0.147736]` | `-0.251890` | `[-0.434846, -0.074417]` | `-57.80` | `4.4694` | `1.9881e+01` | `False` (`100%` fallback) |
| `PROPOSER_W128` | `204103` | `-0.201149` | `[-0.277051, -0.142413]` | `-0.254492` | `[-0.466770, -0.061091]` | `-49.98` | `4.4694` | `2.5353e+01` | `False` (`100%` fallback) |

---

## 3. Diagnostic Findings & What to Try Next

1. **Positive Signal Inside the `K=16` Training Regime (`PROPOSER_W128` Beats `R882_CONT`)**:
   - With the live differentiable `DOPE` token bridge (`compute_live_tokens_10x32` wiring `router` `gates_14` `512:526` and `target_net.col_gate_head` `565:629` into `tok2/tok4/tok5/tok9` with exact `0.0` step-0 discrepancy and nonzero gradient norms `max_dope_data_grad_l2` of `19.88–25.35`), `PROPOSER_W128` outperformed width-64 `R882_CONT` on **both** 205-family OOF Dev (`-0.201733` vs. `-0.250659`, **`+0.048926`**) and 91-context `DL-Val` (`-0.260930` vs. `-0.305192`, **`+0.044262`**) across all 3 seeds.
2. **Why Raw `DL-Val` Regressed Against Native `r882` (`1.1377941898`)**:
   - **Subsampled training pool (`16 / 1,152 = 1.4%`) vs. full-pool argmin inference (`1,152` candidates)**: Training pairwise/Huber ranking on `4` revealed anchors + `12` stratified random non-anchors leaves `1,136` candidates per context unseen during supervision. When the learned residual delta is applied across all `1,152` candidates at test time, out-of-distribution candidates receive optimistic negative cost residuals that displace `r882`'s true top-1 pick. That also explains why `H180_RAND` (where the trunk is frozen and only a lightweight random-subset linear head moves slightly) suffers much smaller out-of-sample displacement (`-0.051395`) than full trunk/adapter training (`R882_CONT` `-0.305192`, `PROPOSER_W128` `-0.260930`).
   - **Pre-`JRB-01` sample size (`1,219` contexts / `205` families)**: Unfreezing `444,168` `DOPE` parameters (`router` + `col_gate_head`) on `205` families overfits fold-local context features when `JRB-01` (`1,967` D4 `TRAIN` datasets) is not yet admitted (`jrb01_ready = False`).
   - **H178 205-Family OOF Selector Worked as Designed**: Across all `45` trained models (`3` arms $\times$ `3` seeds $\times$ `5` folds), the `10,000`-draw family-cluster bootstrap OOF selector (`evaluate_h178_selector`) rejected `100%` of regressing checkpoints (`adoption = 0.0%`), preserving native `r882` (`1.1377941898`) with zero deployed harm.
3. **What to Try Next**:
   - **Top-$K$ hard-negative candidate mining under `r882`**: Replace uniform stratified sampling over slots `4..1151` with `r882`'s top-$K$ (`K=16..32`) lowest-cost candidates per context so pairwise ranking directly calibrates the decision boundary where top-1 argmin flips occur.
   - **Trust-region / L2-to-zero residual penalty on unseen slots**: Add a bounded residual clip ($\tanh(\Delta/\tau)\cdot\tau$ with $\tau \le 0.05$) or an explicit margin constraint that penalizes any candidate beating `r882`'s top-1 pick unless verified superior by two-fold Rust CV.
   - **Scoped `col_gate_head`-only (`18,496` params) or LoRA rank-4 unfreeze on `JRB-01` D4 (`1,967` datasets)**: Defer full `444,168`-parameter `router` unfreezing until `JRB-01` `TRAIN` stream admission is live, and test `col_gate_head`-only unfreeze first to avoid context-router overfitting.

---

## 4. Files & Verification

- [`PREREG.md`](PREREG.md): Preregistered scientific hypothesis, controls (`R882_CONT`, `H180_RAND`), Stage-1/Stage-2 gates, and resource rules.
- [`manifest.json`](manifest.json): Closeout manifest pointing to S3 evidence (`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261101/H204/run-20261102T0511Z/`) and pinned SHA-256 digests.
- [`metrics.json`](metrics.json): Final Stage-0 and Stage-1 metrics (`NO_ADVANCE`, per-arm pooled & per-seed numbers, CIs, Holm $p$-values, `0%` adoption / `100%` fallback).
- [`analysis.py`](analysis.py), [`labeler.py`](labeler.py), [`trainer.py`](trainer.py), [`monitor.py`](monitor.py), [`test_h204.py`](test_h204.py): Implementation and deterministic unit test suite (`7 passed`).

```bash
PYTHONPATH=python/src:. pytest -q research/flywheel_selfplay/h204_bigger_proposer/test_h204.py
```
