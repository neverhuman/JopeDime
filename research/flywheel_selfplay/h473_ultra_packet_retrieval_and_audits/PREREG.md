# H473 Pre-Registration (`h473_ultra_packet_retrieval_and_audits`)

- **Wave ID**: `H473`
- **Status**: Frozen prior to any scored execution on `DL-Train1219` or `DL-Val91`
- **Cohort Protocol**: `full-DL-Train1219-DL-Val91/v1` (`1,219` `DL-Train` contexts across `205` families; `91` `DL-Val` contexts across `19` families; `0` family overlap; `query_rows_used == 0`)
- **Cross-Validation Partitions**: `["h452a/", "h452b/"]` (`2` floor-stratified family-grouped `5`-fold partitions)
- **Random Seeds**: `(1, 2, 3)` (`2 x 5 x 3 = 30` fold-seed cells per CV study)
- **Inference Statistical Protocol**: `10,000`-draw paired family bootstrap (`paired_family_bootstrap_ci`, `seed = 20261120`, `alpha = 0.05`) + step-down `holm_bonferroni` across primary contrasts

## 1. Frozen Code SHA-256 Digests

| File | Frozen SHA-256 |
| --- | --- |
| `h256_token_layout_audit.py` | `13a67b2e36f5e6579d67fa25d88ddb81711d4f57c433027420ae2ff1316d88e8` |
| `prog412_collision_audit.py` | `45a886276d57068cdbf403926137e0971dc53a16da9f7ff333b1226935368ea0` |
| `two_tower_retrieval.py` | `1064ff859c4dfb3aefbd8b56fb9f02110e4d61c3c76ef55359290c65c16c0600` |
| `run_h473_study.py` | `289b281331d7ea30afd60cfe94d644d81eab5d5d19c0e50824009c3d6fb3ed7f` |
| `test_h473_study.py` | `c2e49d9188143e73952bb238302974b2ac31453db40d09602a7578ad8195a467` |

## 2. Immutable Read-Only Input Pins

| Artifact | SHA-256 |
| --- | --- |
| Frozen DOPE Composite (`before == after`) | `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` |
| `cpu_exact_fused.py` | `b9f1468a61aa4dc888c8f2931f9c759f7d4f057793a77ccb05eed6300fb4f9ab` |
| `round-3175.pt` | `1d6ea900ebed8d2a05c3b3f37576adc15aadd2b53c585bfd5cb71a103e9ff97b` |
| `promotions.jsonl` | `d7796fdd7c0dde19e1a23f34c28afc1b9aafa27b0792df746e9f9db01f8cd6c5` |
| `selected.json` | `981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1` |
| `champions.json` | `a444047ffb937dc8d69012cbdbf0891faf3e7d743b127e7cac3baed114cf141d` |
| `h470_features_dl_train1219` manifest | `5c70c45d76e0b00262c88cbc1c732aa8183170d43f02e925f914b02d2aba1422` |
| `h470_features_dl_val91` manifest | `f482f8546f93a2aaa87fcca3fdbfe686dfe730f0f8e18119b880bb06f76d804d` |
| `pre_condense_context_features.npz` | `5821dc6b5931d6b415f9bea3308d94e54a7b60c1cd9ebbb94fcebb9155fa4980` |

## 3. Pre-Registered Audits, Hypotheses, & Arms

### Part A1 (`DV-01` Repair — Authoritative 92-Token Layout)
- Emit `token_layout.json` (`schema: "jope.h256-token-layout/v1"`): `desc256: [0, 8]`, `task128: [8, 12]`, `dist128: [12, 16]`, `heads53_lognorms11: [16, 18]`, `col_gates64: [18, 20]`, `hidden256: [20, 28]`, `columns64: [28, 92]` (`total_tokens = 92`, `token_dim = 32`), plus non-token `scale_side_features` (`width = 8`).
- Assert contiguous non-overlapping `[0..92]` coverage, inactive column zero-masking, and quantify the exact error produced by the historical 93-token slice indices (`[0:32]`, `[32:44]`, `[44]`, `[45:77]`, `[77:93]`).

### Part A2 (`DV-03` Audit — `prog_412` Structural Sufficiency)
- Scan all `1,219` `DL-Train` candidate pools (`1,404,288` `(context, candidate)` rows) plus `champions.json` (`DL-Train`).
- Report distinct `program_sha256` hashes, distinct exact and `1e-4`-quantized `prog_412` vectors, colliding program fractions, and within-pool CV utility (`|u(p) - u(q)|`) and gate-pass disagreement for both exact collisions and `cosine > 0.9999` near-collisions.

### Part A3 (`DV-02` Reveal-Policy & `MaskedPresenceV1` Empty-History Benchmark)
- Compare 0-reveal `MaskedPresenceV1` (`[false; 4]`) vs legacy `wave0_fake_k4` (`fitness=100, gate=0`) and `wave0_fake_k1` on `round-3175.pt` across both `DL-Train1219` and `DL-Val91`.
- Compare 4-reveal policies (`serving_storage_prefix4` `[0,1,2,3]` vs permutation-invariant `deterministic_diverse_anchor4` vs `training_oracle_best4`).

### Part B (`I3` / `I8` — Matryoshka `[32, 64]` Two-Tower Retrieval & `base_prior_cost`)
- Arms: `ret_pop`, `ret_lm48_knn`, `ret_desc_2t` (`290-D`), `ret_dope_2t` (`663-D`), `ret_ultra_packet_2t` (`859-D` `DopeUltraPacket859`, `m = 64` and `m = 32` prefix), and `ret_ultra_packet_shuffled` (cross-family donor-deranged `859-D` placebo).
- Tasks:
  1. Cross-context `TRAIN`-fold champion-bank `K = 96` slate construction (`Spec 2 §6.4`): `Recall@96`, hit rate, and measured champion top-1 family utility.
  2. Within-pool `K = 16` / `K = 32` shortlist (`Spec 2 §6.3`): shortlist oracle utility, top-8 pool recall, and `predict_costs(..., base_prior_cost=...)` at inner-fold-calibrated `alpha_2t` vs `alpha_2t = 0` (`incumbent_0rev`).

### Part C (`I10` Zero-Shot `op_desc` & `I9` Context-Scaled `pair_noise` Gate + One-Time `DL-Val91` `REGCHECK`)
- **`I10` Zero-Shot Archetype Selection**: Predict 24-D winner `semantic_signature` / 5 terminal heads from `859-D` `DopeUltraPacket859` vs `R_dope` (`663-D`) vs `R_desc` (`290-D`) vs `R_none` (`0-D`) vs `shuffled`, selecting the single closest canonical `TRAIN` template (`1` inference, `0` ranker evaluations).
- **`I9` `pair_noise` Confidence Gate**: Predict context `pair_noise` and `noise_over_headroom` from `859-D` `DopeUltraPacket859`, evaluating `z_ctx = 0.40 * clip(sigma_hat / sigma_med, 0.5, 2.0)` on 16-step `pairnf` OOF predictions vs fixed `z = 0.40`.
- **One-Time `DL-Val91` `REGCHECK`**: `DL-Val91` remains completely unopened during CV and is scored once (`val91_evaluations_count == 1`) after all `DL-Train1219` CV selections are frozen.

## 4. Decision Rules

1. **DV-01**: `PASS` iff `[0..92]` contiguity and inactive-column zero masking (`max_abs == 0.0`) hold and slice-bug errors are quantified.
2. **DV-02**: `PASS` iff `masked_presence_0rev - wave0_fake_k4` 95% family bootstrap CI lower bound `> 0.0` on `DL-Train1219` and `DL-Val91`.
3. **DV-03**: Classify `LOSSY_ALIASING_DETECTED` if distinct `program_sha256` hashes collide or near-collide (`cos > 0.9999`) with non-zero utility/gate disagreement; otherwise `EXACTLY_BIJECTIVE`.
4. **Part B (`I3`/`I8`)**: `PASS` iff `ret_ultra_packet_2t` achieves 95% family bootstrap `ci_low > 0.0` over `ret_lm48_knn` on champion retrieval OR over `incumbent_0rev` on 0-reveal `base_prior_cost`, AND beats `ret_ultra_packet_shuffled`; otherwise report `NO_LIFT` transparently.
5. **Part C (`I10`, `I9`)**: `PASS` iff `i10_ultra_packet_859 - i10_none_0d` has 95% family bootstrap `ci_low > 0.0` (`I10`) and `pairnf_i9_ultra_ratio - pairnf_fixed_z40` has `ci_low > 0.0` (`I9`); otherwise report `NO_LIFT` transparently.

## 5. Pre-Score Runtime Bug Addendum

- **Addendum 1 (`2026-11-20`, prior to any pool load or scored evaluation)**: During `--smoke` startup cache authentication (`authenticate_feature_cache`), `jope.h470.cache-auth/v1`'s `le_sha256` format includes the canonical `shape={shape}|dtype={dtype}|` header prefix prior to `tobytes(order="C")`. Updated line 154 of `run_h473_study.py` (`289b281331d7ea30afd60cfe94d644d81eab5d5d19c0e50824009c3d6fb3ed7f` -> `c6f4bdb930e010f26d13e29118e4c414f0dbe79d0cb65895a96d29ddf526bb4f`); all other 4 module SHA-256 digests, hypotheses, hyperparameters, seeds, partitions, and decision rules are unchanged.
- **Addendum 2 (`2026-11-20`, `--smoke` runtime AttributeError fix prior to any full scored run or `DL-Val91` access)**: Replaced private-attribute access in `_adapter_4rev_batch` with direct calls to `FlywheelBlockFormerAdapterNet`'s public methods (`encode_multimodal_context_and_candidates` + `compute_adapter_delta_from_static_terms`). Updated `run_h473_study.py` SHA-256 (`c6f4bdb930e010f26d13e29118e4c414f0dbe79d0cb65895a96d29ddf526bb4f` -> `bdcdcc8fa24bd85f5d541c89c92d5a11836d20391486a46edebdfe835984f777`); all other 4 module SHA-256 digests, hypotheses, arms, seeds, partitions, and decision rules remain unchanged.
