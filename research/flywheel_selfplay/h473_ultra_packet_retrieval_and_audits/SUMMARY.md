# H473 Scored Study Summary: Ultra+Packet Retrieval, Zero-Shot Archetypes, & DV-01..03 Audits

- **Wave**: `H473` (`overall_verdict = AUDITS_CLOSED_WITH_ULTRA_PACKET_RETRIEVAL_SIGNAL`)
- **Frozen DOPE Composite SHA-256 (`before == after`)**: `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`
- **Cohort**: `1219` `DL-Train` contexts (`205` families) / `91` `DL-Val` contexts (`19` families), `0` family overlap, `0` query rows
- **Runtime & Memory**: `5466.48s` wall, `11.186 GiB` peak RSS, `0.0 MB` scratch disk

## 1. Criterion Verdicts

| Criterion | Status | Summary Evidence |
| --- | --- | --- |
| `DV01_h256_92tok_layout` | **PASS** | 92-token layout [0..92] + 8D non-token scale_side_features verified; historical 93-token slice distortion quantified. |
| `DV02_masked_presence_and_4rev_policy` | **PASS** | masked_0rev - wave0_fake_k4 Δ=+0.219228 CI=[+0.112787, +0.330410]; oracle4 - prefix4 Δ=+0.342796. |
| `DV03_prog412_collision_audit` | **LOSSY_ALIASING_DETECTED** | 1316986 distinct hashes -> 1313440 exact prog_412 vectors (275 within-pool exact pairs, 1381118 near pairs cos>0.9999). |
| `I3_I8_two_tower_ultra_packet` | **PASS** | Champ Top-1 ultra-lm48_knn Δ=+0.002592 CI=[-0.064172, +0.072930]; Prior 0-rev ultra-incumbent Δ=+0.539384 CI=[+0.393708, +0.703244]. |
| `I10_zero_shot_op_desc` | **NO_LIFT** | i10_ultra_packet_859 - i10_none_0d Δ=+0.035251 CI=[-0.005940, +0.081963]; vs dope663 Δ=+0.032205. |
| `I9_pair_noise_dynamic_z_gate` | **NO_LIFT** | log_pair_noise R²=0.0570; i9_ultra_ratio - fixed_z40 Δ=+0.005139 CI=[-0.000935, +0.012310]. |

## 2. Part A1 (`DV-01`): 92-Token `PreCondenseResidualBridge` Layout & Historical 93-Token Slice Error
- Contiguous `[0..92]` coverage verified (`token_dim = 32`, separate `scale_side_features` width `8`, inactive column max abs `0.00e+00`).
- `hidden256` `[20:28]` authoritative L2 `3.2935` vs buggy `[45:77]` `0.2649` (relative error `-92.0%`).
- `columns64` active-masked `[28:92]` L2 `2.8713` vs buggy `[77:93]` (truncated to `15` tokens) `0.0929` (`-96.8%`).

## 3. Part A2 (`DV-03`): `prog_412` Collision & Sufficiency Audit (`DL-Train1219`)
- Scanned `1404287` evaluations across `1316986` distinct `program_sha256` hashes (`deterministic = True`).
- Exact `prog_412` distinct vectors: `1313440` (`4969` colliding programs, `0.3773%`).
- Within-pool exact `prog_412` disagreement: `275` pairs, mean `|Δu| = 0.00178`, gate disagreement `0.00%`.
- Within-pool near-collisions (`cos > 0.9999`): `1381118` pairs across `1219` contexts, mean `|Δu| = 0.18791`, p90 `0.36996`.

## 4. Part A3 (`DV-02`): Reveal-Policy & `MaskedPresenceV1` Benchmark

| Mode (`DL-Train1219`) | Fam Mean Utility | Ctx Mean Utility | Gate Fails | Top-1 Agree vs `masked_0rev` |
| --- | ---: | ---: | ---: | ---: |
| `masked_presence_0rev` | `0.986389` | `1.038181` | `72` | `1.0000` |
| `wave0_fake_k4` | `0.767161` | `0.805814` | `116` | `0.0804` |
| `wave0_fake_k1` | `0.726729` | `0.780149` | `104` | `0.1034` |
| `serving_storage_prefix4` | `1.382530` | `1.420532` | `19` | `0.0246` |
| `deterministic_diverse_anchor4` | `1.106508` | `1.131933` | `32` | `0.0295` |
| `training_oracle_best4` | `1.725326` | `1.792664` | `23` | `0.0533` |

## 5. Part B (`I3`/`I8`): Two-Tower Retrieval & `base_prior_cost` (`2` Partitions x `5` Folds x `3` Seeds)

| Arm | Champ Recall@96 | Champ Top-1 Fam Util | Shortlist-16 Oracle | Shortlist-32 Oracle | Top-8 Recall@32 | Prior 0-Rev Fam Util |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ret_pop` | `0.5932` | `0.961782` | `1.533824` | `1.650302` | `0.0310` | `0.986389` |
| `ret_lm48_knn` | `0.1763` | `0.982622` | `—` | `—` | `—` | `—` |
| `ret_desc_2t` | `0.1195` | `0.985946` | `1.883882` | `1.896473` | `0.4089` | `1.487276` |
| `ret_dope_2t` | `0.1286` | `0.979287` | `1.888475` | `1.900819` | `0.4106` | `1.510453` |
| `ret_ultra_packet_2t` | `0.1352` | `0.985214` | `1.884561` | `1.895176` | `0.4129` | `1.525772` |
| `ret_ultra_packet_2t_m32` | `0.1367` | `0.989977` | `1.875792` | `1.888958` | `0.4057` | `1.485439` |
| `ret_ultra_packet_shuffled` | `0.1343` | `0.976294` | `1.863684` | `1.882942` | `0.3735` | `1.418129` |

## 6. Part C (`I10` Zero-Shot `op_desc` & `I9` `pair_noise` Gate) + `DL-Val91` `REGCHECK`
- `I10` `op_desc` OOF family utility (`DL-Train1219`): `i10_ultra_packet_859` `1.129687` (`97.55%` gate-pass, `head5_accuracy = 0.2013`) vs `i10_dope_663` `1.097482` (`Δ = +0.032205`, 95% CI `[-0.031525, +0.126233]`, `106W/4T/95L`), `i10_none_0d` `1.094436` (`Δ = +0.035251`, 95% CI `[-0.005940, +0.081963]`, `116W/2T/87L`), and `i10_ultra_packet_shuffled` `1.094739` (`Δ = +0.034948`, 95% CI `[+0.000278, +0.080180]`, `126W/0T/79L`).
- `I9` predictability (`DopeUltraPacket859` OOF): `log_pair_noise` $R^2 = 0.0570$ ($\rho = 0.3936$), `log_incumbent_headroom` $R^2 = 0.1371$ ($\rho = 0.4201$), `log_noise_over_headroom` $R^2 = 0.0189$ ($\rho = 0.3234$).
- `I9` `pairnf` OOF utility (`DL-Train1219`): `pairnf_i9_ultra_ratio` `1.395984` (`1.09%` gate-fail) vs `pairnf_fixed_z40` `1.390845` (`Δ = +0.005139`, 95% CI `[-0.000935, +0.012310]`, `54W/105T/46L`), `pairnf_i9_dope663_ratio` `1.388717` (`Δ = +0.007267`, 95% CI `[+0.001445, +0.014971]`), and `incumbent_4rev` `1.382530` (`Δ = +0.013453`, 95% CI `[+0.003274, +0.026397]`).
- **Step-down Holm-Bonferroni adjusted $p$-values (`DL-Train1219` primary contrasts)**:
  - `DV02_masked0rev_vs_wave0_fake_k4`: $p_{\text{Holm}} = 0.0012$
  - `B_prior0rev_ultra_vs_incumbent0rev`: $p_{\text{Holm}} = 0.0012$
  - `B_shortlist32_ultra_vs_dope2t`: $p_{\text{Holm}} = 0.2120$
  - `C_i9_ultra_ratio_vs_fixed_z40`: $p_{\text{Holm}} = 0.3168$
  - `C_i10_ultra_vs_dope663`: $p_{\text{Holm}} = 0.8675$
  - `B_champ_ultra_vs_lm48_knn`: $p_{\text{Holm}} = 0.9837$

## 7. One-Time Frozen `DL-Val91` `REGCHECK` (`91` Contexts / `19` Families)

| Contrast (`DL-Val91`) | Left Mean | Right Mean | `Δ` Family Utility | 95% Family Bootstrap CI | W / T / L |
| --- | ---: | ---: | ---: | --- | :---: |
| `DV-02`: `masked_0rev - wave0_fake_k4` | `0.769448` | `0.417480` | `+0.351968` | `[+0.116953, +0.599881]` | `16 / 1 / 2` |
| `DV-02`: `serving_prefix4 - masked_0rev` | `1.154729` | `0.769448` | `+0.385281` | `[+0.186473, +0.620635]` | `15 / 0 / 4` |
| `Part B`: `champ_top1_ultra - lm48_knn` | `0.538014` | `0.591738` | `-0.053724` | `[-0.284949, +0.134917]` | `11 / 1 / 7` |
| `Part B`: `shortlist32_ultra - dope2t` | `1.331125` | `1.318574` | `+0.012552` | `[-0.010254, +0.047396]` | `12 / 0 / 7` |
| `Part B`: `prior0rev_ultra - incumbent0rev` | `0.678303` | `0.769448` | `-0.091145` | `[-0.324307, +0.109050]` | `9 / 0 / 10` |
| `Part C I10`: `ultra_859 - dope_663` | `0.814883` | `0.797960` | `+0.016923` | `[-0.043195, +0.099924]` | `8 / 0 / 11` |
| `Part C I10`: `ultra_859 - none_0d` | `0.814883` | `0.819730` | `-0.004847` | `[-0.125654, +0.128701]` | `7 / 0 / 12` |
| `Part C I9`: `i9_ultra_ratio - fixed_z40` | `1.159160` | `1.155695` | `+0.003466` | `[+0.000430, +0.007299]` | `5 / 13 / 1` |

- **Regime Interpretation**: `DL-Train1219` pools include `3,711` replayed cross-context promotion rows (`promotions.jsonl`), creating shared high-performing program templates across contexts that `TwoTowerRetriever` exploits in OOF within-pool scoring (`top8_pool_recall_at_32 = 0.4129`, `prior_0rev` `Δ = +0.539384`), whereas `DL-Val91` pools contain only unpromoted local candidates (`top8_pool_recall_at_32 = 0.0957`, `prior_0rev` `Δ = -0.091145` with CI straddling zero). Conversely, `DV-02` (`MaskedPresenceV1` `[false; 4]` vs legacy `wave0_fake_k4`, `Δ = +0.351968`) and `I9` (`pairnf_i9_ultra_ratio` vs `pairnf_fixed_z40`, `Δ = +0.003466`, `5W/13T/1L`) replicate cleanly on `DL-Val91`.
