# H204: Bigger Proposer (`128 -> 256 -> 512`) with Research-Only `DOPE` Unfreeze — Archival Diagnostic Closeout (`NO_ADVANCE` / `SCIENTIFIC HOLD`)

- **Wave**: `H204`
- **Verdict**: **`NO_ADVANCE`** (Stage-1 `W128` gate failed; Stage-2 widths `W256` and `W512` were **not** launched; `0/3` seed-level selector admissions per arm [`0/9` total across the `3` arms] -> `100%` offline fallback to `r882`)
- **Qualification & Governance Status**: **`UNQUALIFIED_HISTORICAL_DIAGNOSTIC_NO_ADVANCE` (`SCIENTIFIC HOLD` preserved; NOT cleared for rerun, reuse, or promotion)**
- **Owner**: `gpu_tokbench_agy` (`[TOKBENCH-AGY]`)
- **Host / GPU**: `xbabe2` (`NVIDIA GeForce RTX 3090`, `cuda:0` — owner-reported released after run completion)
- **PR**: `#1729` (`rd/h204-bigger-proposer`)
- **Authority**: `"Jepson via Grok Bot, card answer 8:22 PM MT" (2026-11-01, t11s3): "Start H205 and H204 now with those unblocks; H206 after xbabe3 disk is cleared."`
- **Evaluator Contract**: `fixed-dev-cost-argmin-negative-fitness/v1` (pinned `round-882.pt` SHA-256 `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`)
- **S3 Evidence Prefix**: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261101/H204/`
- **Archived Historical Diagnostic Run**: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261101/H204/run-20261102T0511Z/` (local: `/home/ubuntu/jope-data/h204/run-20261102T0511Z`, PID `2843546`, `528.1s` wall time)
- **Create-Only S3 Custody Audit Envelope (Exact-Version Replay at `2d5e15fa347ca2b95b0c7dc6dacc0507dedd0b9f`)**:
  - **Audit Envelope URI**: `s3://veox-jopedime-learning-use2/experiments/research_restore_20261101/20261102T0642Z/H204-evidence-replay-2d5e15fa.json`
  - **Audit Envelope `VersionId` / Bytes / SHA-256**: `v4GSvIQ4B.kcyDG5JBwhkFtrBm7knQyi` / `14,532` bytes / `6e9fd82be2509b5258d3ae433450ee1ff949c95cada4b01339f095b0176f5841` (create-only audit envelope + exact-version readback verified; local receipts: `/home/ubuntu/jope_plan/evidence/research_restore_20261101/check-20261102T0642Z/H204-S3-CUSTODY.json` and `H204-S3-REPLAY.{json,md}`).
  - **Bounded Replay Summary**: `11` listed small H204 S3 objects downloaded by exact `VersionId` (`303,964` bytes total): `10` match direct committed SHA-256 pins at `2d5e15fa347ca2b95b0c7dc6dacc0507dedd0b9f` and `1` (`root_report_md`, `VersionId = un7NNz3cHt8rCf0fJ4xYyXG9ZDnbpnqA`, `4,438` bytes) has no tracked root blob in Git but matches the committed run `REPORT.md` pin `a3d4c734ef470c8885631f00066ee0f3feded867732e86b13925108d6d321106`; `0` mismatches.
  - **Provenance & Exclusion Limits**: **Original H204 upload create-only provenance is `UNKNOWN`** (the audit envelope's create-only provenance and exact-version replay do not retroactively prove create-only semantics for the original H204 uploads). The training-label cache (`run-20261102T0511Z/cache/h204_labels_seed204101_k16.json`), models, datasets, NPZ/tensor payloads, and earlier/aborted historical attempts (`run-20261102T0415Z`, `run-20261102T0455Z`, `run-20261102T0458Z`, `run-20261102T0501Z`, `astra-t0305`) were **excluded and are not qualified**. No S3 objects were re-uploaded or overwritten after `2d5e15fa347ca2b95b0c7dc6dacc0507dedd0b9f` (so `root_manifest_json` `lTVIENDygf20ENs.cnYzzgbrgEQ_.CbJ` and `root_readme_md` `vB_CQp1Re97Z4iVu4Ccw0zBR0ay1M0TT` remain bound to the `2d5e15fa347ca2b95b0c7dc6dacc0507dedd0b9f` blobs).

---

## 0. Quarantine & Known Source / Execution Limitations (`SCIENTIFIC HOLD`)

This directory is archived strictly as a **negative historical diagnostic record (`NO_ADVANCE`)** on the static `205`-family pool. The historical source files (`PREREG.md`, `analysis.py`, `labeler.py`, `trainer.py`, `monitor.py`) are preserved byte-for-byte at the executed commit (`d8d241a3d6dd2af597b3e033141dfe79711ca758`) to maintain provenance for `run-20261102T0511Z`, and **remain under scientific `HOLD`** (see `/home/ubuntu/jope_plan/evidence/sm-n2022-codex/review1729/re-review-d8d241a3d.md` and `/home/ubuntu/jope_plan/evidence/sm-gbw2308-codex/review1729/review-951f51422.md`). **The runnable code is NOT cleared for rerun, reuse, or promotion without repairing the defects below and obtaining a new review:**

1. **Static-205 Historical Label Scope (No Live Rust Labeling Executed)**:
   - `run_h204_stage1` (`trainer.py:1258–1264`) loaded historical two-fold Rust CV pool fitnesses from `pinned_205_family_two_fold_rust_cv_pools` (`labeler.py:536–556`).
   - Although `evaluate_context_candidates_rust_cv` (`labeler.py:349–419`) matches the native `evaluate_gp_batch_rust` signature, it has **zero production call sites** in `trainer.py`, `labeler.py`, `monitor.py`, or `analysis.py` (exercised only by a mocked unit test). No fresh live Rust CV labeling and no `JRB-01` D4 (`~3,300` dataset) stream was executed.
2. **Validation Family Count Deviation (`19` Observed vs. `39` Preregistered)**:
   - `PREREG.md:79` specified `91` validation contexts across `39` families for `DL-Val`, whereas the executed `selected.json` `DL-Val` split (`91` contexts) spans **`19` unique `family_id` prefixes** (`n_val_families = 19` vs. `preregistered_val_families = 39`).
3. **Column-Gate Head Bias Cancellation & Partial Input Slice (`trainer.py:338–349`)**:
   - In `compute_live_tokens_10x32`, the residual column-gate update computes `delta_raw_g = task_lat @ W - task0.detach() @ W.detach() + (bias - bias.detach())` using only `W[:, :128]` (omitting the 17 column-feature weight coordinates of the 145-input `col_gate_head`).
   - Because `bias - bias.detach()` evaluates to `0.0` in every forward pass while still propagating a nonzero derivative to `col_gate_head.bias`, optimizer updates to `col_gate_head.bias` never change `tail64`, downstream tokens, or ranking loss. Consequently, nonzero `DOPE` gradient norms and changed parameter hashes do **not** qualify the stated full `col_gate_head` joint fine-tuning experiment.
4. **Executable Gate, Pooled-$p$-to-Holm Wiring, Equal-Step (Non-Compute-Matched) Controls, and Resource Preflight Limits**:
   - **Stage-0 Utility Hash Gate**: `verify_r882_native_parity` (`analysis.py:236–251`) computes `utilities_sha256` (which matched the pinned literal `77fd7f5963905a1aa0fd321f4ae04fe24328961357aa745efb18bbe0ebfec8e9` in `run-20261102T0511Z`), but its `passed` predicate checks only count, finiteness, mean, and picks without enforcing `utilities_sha256 == R882_EXPECTED_DL_VAL_UTILITIES_SHA256` as an executable veto.
   - **Holm Input Statistic**: `trainer.py:1359–1365, 1396–1402` feeds the mean of per-seed `p_one_sided` values (rather than `evaluate_arm_gate`'s `pooled_p_one_sided`) into `holm_bonferroni`.
   - **Equal-Step Controls (Not FLOP / GPU-Time Compute-Matched)**: All three Stage-1 arms ran equal epochs/batches (`15` epochs, batch size `32`) across unequal architectures (`width=64` continuation vs. `width=128` frozen-trunk readout vs. `width=128` + `444,168` unfrozen `DOPE` parameters) without FLOP or GPU-time dose matching.
   - **Resource Monitor / Preflight**: `labeler.py:123` divides `usage.used / usage.total` (diluting reserved root blocks relative to `used / (used + free)`), `monitor.py:203–206` checks `min(load1, load5, load15) <= max_load1` instead of current `load1 <= max_load1`, and `monitor.py:415–435` writes `launch.json` without aborting on an initial `!snap["ok"]` before starting `run_h204_stage1`. Thus `"36 snapshots / 0 violations"` is the monitor's self-report, not independent certification of the approved resource policy.

---

## 1. Stage-0 Baseline Diagnostic Observation vs. `r882`

In `run-20261102T0511Z`, native `r882` (`round-882.pt`) produced the following observed values on the 91 `DL-Val` contexts (`19` observed families; note that `utilities_sha256` equality was verified in the recorded output rather than enforced as an executable veto in `verify_r882_native_parity`):

| Metric | Reference Literal (`r882`) | Observed in `run-20261102T0511Z` | Observation Note |
|---|---:|---:|---|
| `DL-Val` Contexts / Families | `91 / 39` (in `PREREG.md:79`) | `91 / 19` (in `selected.json`) | Population deviation (`19` vs. `39` families) |
| Exact Top-1 Pick Matches | `91 / 91` | `91 / 91` | Pick hash enforced & matched |
| Pick Vector SHA-256 | `460560e44b8d...0c3624d96` | `460560e44b8d...0c3624d96` | Exact match |
| Utility Vector SHA-256 | `77fd7f596390...0ebfec8e9` | `77fd7f596390...0ebfec8e9` | Recorded hash matches literal (not gated in code) |
| Native `r882` Mean Utility (`-fitness`) | `1.1377941898164525` | `1.1377941898164525` (`abs_diff = 0.0`) | Exact match |
| Step-0 Residual Bridge Discrepancy | `<= 1e-5` | `0.000000e+00` (`CPU` & `cuda:0`) | Observed diagnostic (`col_gate_head.bias` cancels) |

---

## 2. Stage-1 Key Numbers (`DL-Val` 91 Contexts / 19 Observed Families & `OOF Dev` 1,219 Contexts / 205 Families)

All three Stage-1 arms were run across `3` seeds (`204101`, `204102`, `204103`) $\times$ `5` family-stratified CV folds (`15` fitted fold models per arm, `45` fitted fold models total across the `3` arms) on the static `205`-family pool (`K=16` historical pool candidates per context). `trainer.py:1100–1168` invokes `evaluate_h178_selector` **once per seed per arm** on the concatenated 5-fold OOF predictions, yielding **`3` seed-level selector decisions per arm (`9` total seed/arm selector decisions across the `3` arms)**.

### Pooled Stage-1 Summary Table (vs. Native `r882` = `1.1377941898`)

| Arm | Width | `DOPE` Setting | OOF Dev Mean $\Delta$ (`205` fam) | Raw `DL-Val` Mean Utility | Raw `DL-Val` Pooled $\Delta$ vs `r882` | Raw `DL-Val` 95% Family Bootstrap CI | Pooled Elo Lift | Raw One-Sided $p$ (Seed Avg / Pooled) | Holm Adjusted $p$ | Positive Seeds (`> r882`) | Seed-Level Selector Decisions (`Fitted Fold Models`) | Offline Fallback `DL-Val` Mean Utility (`r882`) |
|---|---:|---|---:|---:|---:|---|---:|---|---:|---:|---|---|
| **`r882` (Baseline)** | `64` | Frozen (`eval` mode) | `+0.000000` | `1.137794` | `+0.000000` | `[+0.000000, +0.000000]` | `+0.00` | — | — | — | — | `1.137794` |
| **`R882_CONT` (Control 1)** | `64` | Frozen (`0` trainable `DOPE` params) | `-0.250659` | `0.832602` | `-0.305192` | `[-0.408008, -0.175471]` | `-51.31` | `1.0000 / 1.0000` | `1.0000` | `0 / 3` | **`0 / 3` seeds admitted** (`15` fold models fit) | `1.137794` (`100%` offline fallback) |
| **`H180_RAND` (Control 2)** | `128` | Frozen `DOPE` + frozen trunk | `-0.067422` | `1.086400` | `-0.051395` | `[-0.090652, -0.013148]` | `-38.38` | `0.9649 / 0.9985` | `1.0000` | `0 / 3` | **`0 / 3` seeds admitted** (`15` fold models fit) | `1.137794` (`100%` offline fallback) |
| **`PROPOSER_W128` (Treatment)** | `128` | Research unfreeze (`router` + partial `col_gate_head`, `444,168` params in optimizer) | `-0.201733` | `0.876864` | `-0.260930` | `[-0.455367, -0.079643]` | `-59.14` | `0.9997 / 1.0000` | `1.0000` | `0 / 3` | **`0 / 3` seeds admitted** (`15` fold models fit) | `1.137794` (`100%` offline fallback) |

### Per-Seed Breakdown (`204101`, `204102`, `204103`) & `PROPOSER_W128` vs. `R882_CONT` Seed Comparison

- **OOF Dev (`1,219` contexts / `205` families)**: `PROPOSER_W128` had higher OOF Dev utility than `R882_CONT` on **`3 / 3` seeds** (pooled descriptive difference `+0.048926`: seed `204101` `+0.035000`, seed `204102` `+0.054842`, seed `204103` `+0.056936`).
- **Raw `DL-Val` (`91` contexts / `19` observed families)**: `PROPOSER_W128` had higher `DL-Val` utility than `R882_CONT` on **only `2 / 3` seeds** (pooled descriptive difference `+0.044262`):
  - Seed `204101`: `PROPOSER_W128` `-0.276408` vs. `R882_CONT` `-0.324394` (`+0.047986`).
  - Seed `204102`: `PROPOSER_W128` `-0.251890` vs. `R882_CONT` `-0.347374` (`+0.095484`).
  - Seed `204103`: `PROPOSER_W128` `-0.254492` vs. `R882_CONT` `-0.243808` (**`PROPOSER_W128` worse by `-0.010684`**).

| Arm | Seed | OOF Dev Mean $\Delta$ (`1,219` ctx) | OOF Dev 95% CI | Raw `DL-Val` Mean $\Delta$ (`91` ctx) | Raw `DL-Val` 95% CI | `DL-Val` Elo Lift | Worst Regression | `max_dope_data_grad_l2` | Seed Selector Decision (`1` per seed; `5` fold models fit) |
|---|---:|---:|---|---:|---|---:|---:|---:|---|
| `R882_CONT` | `204101` | `-0.234343` | `[-0.269091, -0.151299]` | `-0.324394` | `[-0.438596, -0.179218]` | `-57.80` | `3.2244` | `0.0000e+00` | `selector_admitted=False` (offline fallback to `r882`) |
| `R882_CONT` | `204102` | `-0.259550` | `[-0.330967, -0.187766]` | `-0.347374` | `[-0.475041, -0.192750]` | `-57.80` | `3.2699` | `0.0000e+00` | `selector_admitted=False` (offline fallback to `r882`) |
| `R882_CONT` | `204103` | `-0.258085` | `[-0.326510, -0.175666]` | `-0.243808` | `[-0.367927, -0.107759]` | `-38.33` | `3.1174` | `0.0000e+00` | `selector_admitted=False` (offline fallback to `r882`) |
| `H180_RAND` | `204101` | `-0.065583` | `[-0.109783, -0.034782]` | `-0.087706` | `[-0.167230, -0.016667]` | `-34.47` | `2.6051` | `0.0000e+00` | `selector_admitted=False` (offline fallback to `r882`) |
| `H180_RAND` | `204102` | `-0.075263` | `[-0.127426, -0.048687]` | `-0.050199` | `[-0.084884, -0.016643]` | `-53.88` | `1.0247` | `0.0000e+00` | `selector_admitted=False` (offline fallback to `r882`) |
| `H180_RAND` | `204103` | `-0.061421` | `[-0.096927, -0.026094]` | `-0.016278` | `[-0.039449, +0.008887]` | `-26.78` | `0.6816` | `0.0000e+00` | `selector_admitted=False` (offline fallback to `r882`) |
| `PROPOSER_W128` | `204101` | `-0.199343` | `[-0.270419, -0.153907]` | `-0.276408` | `[-0.482390, -0.087986]` | `-69.64` | `4.4694` | `2.0104e+01` | `selector_admitted=False` (offline fallback to `r882`) |
| `PROPOSER_W128` | `204102` | `-0.204708` | `[-0.272063, -0.147736]` | `-0.251890` | `[-0.434846, -0.074417]` | `-57.80` | `4.4694` | `1.9881e+01` | `selector_admitted=False` (offline fallback to `r882`) |
| `PROPOSER_W128` | `204103` | `-0.201149` | `[-0.277051, -0.142413]` | `-0.254492` | `[-0.466770, -0.061091]` | `-49.98` | `4.4694` | `2.5353e+01` | `selector_admitted=False` (offline fallback to `r882`) |

---

## 3. Speculative Hypotheses & Requirements Before Any Future Follow-Up

1. **Observed Descriptive Comparison vs. Unresolved Confounders**:
   - While `PROPOSER_W128` had a less negative pooled mean than `R882_CONT` (`+0.044262` on `DL-Val` with `2/3` seed wins; `+0.048926` on OOF Dev with `3/3` seed wins), both arms remained far below native `r882` (`1.1377941898`) and below frozen-trunk `H180_RAND` (`-0.051395`). Because of the `2/3` seed split on `DL-Val`, unequal parameter/compute budgets, and the `col_gate_head.bias` forward cancellation defect (`trainer.py:344`), this difference is strictly descriptive and **does not establish causal benefit from `DOPE` unfreezing**.
   - In the offline evaluation calculation, all `9` seed-level selector calls (`3` seeds $\times$ `3` arms, each aggregating `5` fold models) returned `selector_admitted = False`, which maps to `r882`'s baseline utility (`1.1377941898`) in the offline fallback formula. This is an offline arithmetic property of the negative OOF Dev gains, not a production deployment qualification.
2. **Speculative / Untested Hypotheses for Raw `DL-Val` Regression**:
   - *Hypothesis A (Subsampled candidate pool displacement)*: Training ranking loss on `16` candidates (`4` anchors + `12` stratified random slots) out of `1,152` may leave `1,136` candidates per context uncalibrated at inference time, allowing out-of-subset candidates to receive optimistic negative cost residuals that displace `r882`'s top-1 pick.
   - *Hypothesis B (Over-parameterization on static `205` families)*: Optimizing `444,168` `DOPE` parameters (`router` + partial `col_gate_head` weights) plus a width-128 trunk on `1,219` static contexts (`205` families) may overfit fold-local context features.
   - *Confounder Note*: Neither hypothesis is isolated by this run from the known source defects documented in Section 0 (e.g., `col_gate_head.bias` forward cancellation, missing 17 column-feature weight coordinates, and static pool reuse).
3. **What to Try Next (Requires Fresh Preregistration & Source Review)**:
   - Repair all Section 0 `HOLD` items first: wire live native `evaluate_gp_batch_rust` labeling into the training loop, remove the `bias - bias.detach()` cancellation in `compute_live_tokens_10x32`, enforce `utilities_sha256` in `verify_r882_native_parity`, feed `pooled_p_one_sided` into `holm_bonferroni`, and enforce strict preflight resource admission (`load1 <= 85%`, `used / (used + free) <= 93.8%`).
   - Test top-$K$ (`K=16..32`) hard-negative candidate mining under `r882` and bounded trust-region residual clipping (`|\Delta cost| <= \tau`) on the admitted `JRB-01` D4 `TRAIN` stream (`1,967` datasets).

---

## 4. Files & Unit Test Command

- [`PREREG.md`](PREREG.md): Original preregistered document (note Section 0 deviations/limitations above).
- [`manifest.json`](manifest.json): Archival closeout manifest (`UNQUALIFIED_HISTORICAL_DIAGNOSTIC_NO_ADVANCE`, `SCIENTIFIC HOLD` disclosure, S3 URIs, and SHA-256 digests).
- [`metrics.json`](metrics.json): Final Stage-0 and Stage-1 diagnostic metrics (`NO_ADVANCE`, `2/3` `DL-Val` seed wins vs. `3/3` OOF Dev seed wins, `9` seed-level selector decisions vs. `45` fitted fold models, `19` observed vs. `39` preregistered `DL-Val` families, and explicit quarantine metadata).
- [`analysis.py`](analysis.py), [`labeler.py`](labeler.py), [`trainer.py`](trainer.py), [`monitor.py`](monitor.py), [`test_h204.py`](test_h204.py): Preserved historical source and unit tests (`7 passed`; not cleared for production/experiment reuse without resolving Section 0).

```bash
PYTHONPATH=python/src:. pytest -q research/flywheel_selfplay/h204_bigger_proposer/test_h204.py
```
