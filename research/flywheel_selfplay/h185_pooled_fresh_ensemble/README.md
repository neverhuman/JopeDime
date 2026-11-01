# H185: Power-Aware Pooled Confirmatory Test of Fresh-Init and Ensemble Candidates (`FRESH160`, `ENS5`, and `RENS5` vs `R882`)

- **Wave**: `H185`
- **Owner**: `gpu_tokbench_agy` (`[TOKBENCH-AGY]`)
- **Reserved by**: `ENG:hwave-loop`
- **Authority**: `/home/ubuntu/nudges/hwave_t0700_h185_tokagy.md` (`DECISIONS_20261029.md`)
- **Preregistration**: [PREREG.md](PREREG.md), [fold_manifest.json](fold_manifest.json), [manifest.json](manifest.json), [provenance.json](provenance.json)
- **Imported Selector & Calibration Anchor**:
  - Imported directly from [`research/flywheel_selfplay/h177_pooled_selector/analysis.py`](../h177_pooled_selector/analysis.py) (`H177_DECISION_RULE`, `pooled_b_select`, `StageOrderGuard`, `verify_pooled_b_coverage`, `resolve_h177_verdict`).
  - Calibrated by `H178` (`CALIBRATED_PASS_POWER_OK`, [`research/flywheel_selfplay/h178_pooled_calibration/h178_receipt.json`](../h178_pooled_calibration/h178_receipt.json), commit `6b70d803b3c51ca8cd46720244dbdfa872c08868`: `0/20` null/harmful admissions, `24`-slate family-clustered `MDE80 = +0.0258` vs single-fold `+0.0640`).
- **Execution Host**: `xbabe2` (`CPU-only`, `CUDA_VISIBLE_DEVICES=""`, `nice -n 19`, `<= 4` threads)
- **Status**: **`PREREGISTERED`** (awaiting preregistration review and single guarded CPU execution on `xbabe2`)

---

## 1. Design Summary & Disclosures

| Parameter | Specification |
| :--- | :--- |
| **Reference Arm** | `R882` (`round-882.pt`, `0` updates; `DL-Val` context utility `1.1377941898164525`, `1500.0` Elo) |
| **Candidate Arm 1 (`FRESH160`)** | `1` fresh-initialized `adapter_10tok` (`29,211` params) trained for `160` updates on `FIT_k` (`H180` recipe; deterministic fallback to `80` updates only if 4-update `FIT_0` timing pilot projects `> 3.0 h`) |
| **Candidate Arm 2 (`ENS5`)** | `5` independently seeded fresh-init adapters trained for `40` updates each on `FIT_k` (`H179` recipe); member candidate cost predictions averaged before `argmin` |
| **Candidate Arm 3 (`RENS5`)** | Untrained random-structure control: the **exact same `5` fresh-initialized member states** as `ENS5` at `0` updates, averaged before `argmin` |
| **Outer Seeds (`3`)** | `185101, 185102, 185103` |
| **Member Seeds (`5` per outer seed)** | `185201–185205` (`185101`), `185301–185305` (`185102`), `185401–185405` (`185103`) |
| **Cross-Fit Fold Partition (`5` folds)** | `fold_seed = 185140` (`123 FIT / 41 B / 41 A` families per fold; `205` `DL-Train` families, `1,219` contexts) |
| **Stage-1 Selector (`Pooled B`)** | Imported `H177_DECISION_RULE` (`pooled_b_select` over `205` families, `10,000` paired family bootstrap draws with `boot_seed = 185141`, within-family 3-seed mean, $m = 3$ Holm over `FRESH160`, `ENS5`, `RENS5`: `gain > +0.001 + 1e-12`, `LCB95 > 0`, `holm_p < 0.05`, locked in `pooled_b_selection_lock.json` via `StageOrderGuard` before Stage 2) |
| **Stage-2 Readout (`Pooled A` + `DL-Val`)** | Correlated cross-fold `A` readout (`205` families, `1,219` contexts) + descriptive `DL-Val` (`19` families, `91` contexts) + active contrasts (`ENS5 - RENS5`, `FRESH160 - RENS5`, `ENS5 - FRESH160`) + `-2.0` failure pick counts |
| **Confirmatory Verdict Rule** | `ADVANCE` iff an arm is admitted on Stage-1 pooled `B` (with `H178` receipt passing) **and** achieves Stage-2 pooled `A` gain $> 0$ with `3/3` positive outer seeds (ties reject); else `NO_ADVANCE` |

### Selection, Shared Fold-Seed, & Cross-Fold Dependency Caveats
`FRESH160`, `ENS5`, and `RENS5` were selected for `H185` after observing their unconditional `A` gains in `H179` and `H180` on the same `205` `DL-Train` families. All three outer seeds (`185101–185103`) share the single five-fold family partition generated from `fold_seed 185140` (`123 FIT / 41 B / 41 A` families per fold, like `H182`), so across-seed variation is fresh-adapter / ensemble-member initialization and `FIT` minibatch-stream order only (leaving `R882` utility identical across seeds). Furthermore, because `A_k` and `B_k` cycle across the same `205` `DL-Train` families and fold training splits share `82/123` fit families, Stage-2 pooled `A` is a **correlated cross-fold candidate readout**, not an independent sample split. `DL-Val` (`19` disjoint families, `91` contexts) is reported descriptively.

---

## 2. Verification Commands

```bash
PYTHONPATH=research/flywheel_selfplay/h185_pooled_fresh_ensemble pytest -q research/flywheel_selfplay/h185_pooled_fresh_ensemble/test_h185.py
```
