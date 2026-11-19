# H470-DVP — DOPE Value Probe (`WP0`–`WP4`) & Zero-Retraining Accuracy Harvester (`WP3` & `B03`)

Scored CPU-only research evaluation (`H470-DVP`) turning the `dope_value_probe/` layout stub into the full pre-registered diagnostic, probe matrix, and zero-retraining upgrade harness on the `full-DL-Train1219-DL-Val91/v1` cohort (`1,219` `DL-Train` contexts across `205` families; `91` `DL-Val` contexts across `19` families; `0` family overlap; `0` outer/query rows).

Product crates (`crates/jope-encoders`, `crates/jope-service`) and `python/src/jope/north_star/dope_freeze_guard.py` remain 100% untouched, and frozen DOPE (`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`) is verified bit-identical before and after execution.

## Resolution of Audit Question `L7` (`lm48` vs `lm_smart_80`)

Direct inspection of `cpu_exact_fused.py` (`_run_dope_mlc_v3_and_lm48` and `_compute_lm_smart_80d_gpu`) and empirical verification in `A2_l7_lm48_vs_lm80_audit` resolve `L7`:
- `lm48 = clamp((vec256 - step2_lm_mean_1x256) / step2_lm_std_1x256, -6, 6) @ step2_svd_basis_256x48` is an orthonormal **rank-48 SVD coordinate projection** (`||B^T B - I_48||_max < 1e-6`), **not** a 16-probe `(mean, std, max)` basket.
- `_compute_lm_smart_80d_gpu` reshapes `lm48` into `(16, 3)` `(pm, |ps|, px)` triples and computes quantiles, IQRs, and tail-gaps across arbitrary signed singular coordinates (`ps < 0` before `abs()` on ~50% of entries; `px < pm` on ~50% of triples). `H470` therefore benchmarks `lm48_z` (`48D`) directly alongside `R_lm80` (`80D`) across retrieval and `T1`–`T4`.

## Module Inventory

| File | Role |
| --- | --- |
| `PREREG.md` | Pre-registered hypotheses, arms, `C0`–`C8` contrasts, `h452a/`+`h452b/` 5-fold protocol, zero-`val91`-peeking rule, and stop/go criteria (committed prior to any scored run). |
| `cache_auth.py` | `WP01` 7-field SHA-256 content-addressed `.npz` cache authenticator (`dope_weights_sha256`, `stage1a_mode_and_bridge_sha256`, `prep_recipe`, `context_sha256`, `fit_rows_sha256`, `conditioning_level`, `stage1b_sha256`), strictly refusing `query_rows_used != 0` or mismatched digests. |
| `packet_v1_extract.py` | FIT-row-only `.jtt1` reader (`nq == 0`), 5-fold FIT-CV 23-landmark panel (`landmarks-native-v1`, including fine probes `gbdt, knn1, knn16, net, gbdt_quarter` and `U-S3` latency), `high`/`ultra` research view (`conditioning_level == level` enforced), and `196D` scale-invariant, column-permutation-invariant, extrema-free `R_packet_v1` (`dope.packet-features/v1`) + `R_packet_notarget` (`P-S2`). |
| `anchor_fast_cache.py` | Shared frozen-anchor `FlywheelGameNet` precomputation bank (`< 1e-6` verified parity, >100x CPU speedup). |
| `h470_probe_tasks.py` | Reusable `dvp_score(R, labels, folds)` selector (`DVP_CONFIG_SHA256`), cross-family donor permutation (`family[donor] != family[i]`), `profile_oracle` (`A -> B`) and `R_oracle64` (pivoted-QR basket-excluded ceiling), Part A (`A1`–`A4`), and Part B (`T1`–`T4`, `C0`–`C7` + Holm). |
| `h470_part_c.py` | Part C upgrades: `WP3` Empirical-Bayes variance-shrunk predicted reveals at 0 real candidate evaluations (`V-S1..S3`) and `B03` folded Stage-1B SVD `nn.Linear(629, 96)` + zero-init `Ultra+Packet` residual adapter under `pairnf_z40`, plus the one-time frozen `val91` `REGCHECK`. |
| `run_h470_probe.py` | End-to-end CLI entrypoint writing reproducible `metrics.json`, `verdict.json`, and `SUMMARY.md` (with bulk caches/logs confined to `$JOPE_DATA_ROOT/hwave-h470/`). |
| `test_h470_probe.py` | Offline deterministic contract & unit test suite (`"disposition": "research"` in `deploy/ci/coverage.json`). |

## Commands

```bash
# 1. Offline contract & unit tests (no local jope-data packet required)
python3 -B research/flywheel_selfplay/dope_value_probe/test_dope_value_probe.py -v
python3 -B research/flywheel_selfplay/h470_dope_value_probe/test_h470_probe.py -v

# 2. Fast fold-0 smoke check (never opens val91, writes only *_smoke.* to scratch)
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 nice -n 19 \
  python3 -B research/flywheel_selfplay/h470_dope_value_probe/run_h470_probe.py --smoke --threads 8

# 3. Full pre-registered H470 scored evaluation (2 partitions x 5 folds x 3 seeds + 1x val91 REGCHECK)
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 nice -n 19 \
  python3 -B research/flywheel_selfplay/h470_dope_value_probe/run_h470_probe.py --threads 8
```
