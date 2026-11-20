# H473 — Ultra+Packet Two-Tower Retrieval, Zero-Shot Archetypes, & `DV-01..03` Audits

Scored CPU-only research evaluation (`h473_ultra_packet_retrieval_and_audits`) on the frozen `full-DL-Train1219-DL-Val91/v1` cohort (`1,219` `DL-Train` contexts across `205` families; `91` `DL-Val` contexts across `19` families; `0` family overlap; `0` query rows).

Frozen DOPE (`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`) is verified bit-identical before and after execution via `dope_freeze_guard`. No raw source dataset values, headers, or extrema are read or emitted.

## Scope & Module Inventory

| File | Role |
| --- | --- |
| `PREREG.md` | Frozen pre-registration recording exact SHA-256 code digests, hypotheses, arms, `h452a/` + `h452b/` 5-fold protocol, seeds `(1, 2, 3)`, and decision rules prior to scored execution. |
| `token_layout.json` | Authoritative `jope.h256-token-layout/v1` 92-token layout contract (`[0..92]`, `token_dim = 32`) plus width-8 non-token `scale_side_features` for `PreCondenseResidualBridge` Arms F and G. |
| `h256_token_layout_audit.py` | Part A1 (`DV-01`): Contiguous `[0..92]` layout validator, active-column-masked L2 norm telemetry across `PreCondenseResidualBridge` checkpoints, and exact error quantification of the historical 93-token slice indices. |
| `prog412_collision_audit.py` | Part A2 (`DV-03`): Full-pool exact and near (`cosine > 0.9999`) `prog_412` feature collision and within-context utility/gate disagreement audit, permutation-invariant 4-archetype selector, and Part C `I10` zero-shot `op_desc` canonical template selector. |
| `two_tower_retrieval.py` | Part B (`I3`/`I8`): FP32 `TwoTowerRetriever` with Matryoshka `[32, 64]` InfoNCE + Huber utility calibration across cross-context `K = 96` champion-bank retrieval, within-pool `K = 16, 32` shortlists, and inner-fold-calibrated `base_prior_cost` integration. |
| `run_h473_study.py` | End-to-end CLI runner executing Parts A1–A3 (`DV-01..03`), Part B (`I3`/`I8`), Part C (`I10` & `I9` dynamic `pair_noise` confidence threshold `z_ctx`), and the one-time frozen `DL-Val91` `REGCHECK`, emitting `token_layout.json`, `metrics.json`, `verdict.json`, and `SUMMARY.md`. |
| `test_h473_study.py` | Offline deterministic unit and contract test suite (`unittest`). |

## Execution Commands

```bash
# 1. Offline deterministic unit & contract tests
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 nice -n 19 \
  python3 -B -m unittest discover -s . -p 'test_*.py' -v

# 2. Fast smoke verification on DL-Train subset (never opens DL-Val91; writes only *_smoke.* to scratch)
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 nice -n 19 \
  python3 -B run_h473_study.py --smoke --threads 8

# 3. Full pre-registered H473 scored study (2 partitions x 5 folds x 3 seeds + 1x DL-Val91 REGCHECK)
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 nice -n 19 \
  python3 -B run_h473_study.py --threads 8
```
