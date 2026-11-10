# H252 (`W-FLYWHEEL-NEXT`): Decomposed Hurdle Critic, Active `2+2` Reveal, Hurdle-RL + `PCGrad-EMA` Continuation, Sibling-DPO Proposer & Retrospective Pool-Subset Simulation

## 1. Hypothesis

Across waves `H140`–`H250`, full-adapter and subspace neural continuations of `R882` (`round-882.pt`) on the 1,219 `DL-Train` contexts (`205` families) and 91 `DL-Val` contexts (`19` families) repeatedly failed to beat frozen `R882`, while the live self-play runner on `xbabe1` (`flywheel-full-20261030`) plateaued at `round 3175` (`1,156` consecutive rejected rounds through `round 4331`).

`H252` tested four distinct hypotheses across `xbabe1` (RTX 4090 GPU), `xbabe2` (128-core CPU), and `xbabe3` (RTX 3090 GPU):
1. **Stage 1 (`stage1_hurdle_sos_critic.py`, `W3` + `W4` on `xbabe2` CPU)**: Can a 5-fold family-grouped **Decomposed Continuous Hurdle Critic** (separating continuous `logratio` regression from CV `gate` failure risk, enriched with a train-only self-play promotion semantic direction from `3,449` harvested GP promotions) combined with **Sequential `2+2` Uncertainty-Guided Active Reveal (`ACTIVE_2PLUS2_HURDLE`)** improve 4-reveal selection utility over `R882` and `GATED_025` (`H217`)?
2. **Stage 2 (`stage2_rl_gradsmooth_gpu.py`, `xbabe1` RTX 4090 GPU)**: Does feeding the cross-fitted 2-channel Hurdle prior into the previously zeroed `sos_prior` (`[B, K, 2]`) columns of `FlywheelBlockFormerAdapterNet`, replacing the `+2.0` gate-cliff target with continuous `logratio` + Top-16 Plackett-Luce rank-advantage + KL trust region, and applying `PCGrad-EMA` gradient projection + `AGC` + Polyak-EMA shadow weights + Trust-Region Contender Gating improve neural continuation over `R882` and `CTRL_H170`?
3. **Stage 3 (`stage3_proposer_dpo_gpu.py`, `W6` on `xbabe3` RTX 3090 GPU)**: Can a contextual Parent$\to$Child transition network trained with Counterfactual Sibling-DPO + AWR (`PROPOSER_DPO_DELTA`) and submodular semantic diversity re-rank top-64 candidate children better than `R882_RANK_PROPOSER` and `UNIFORM_MUTATION`?
4. **Stage 4 (`stage4_closed_loop_2x2_flywheel.py`, `W9` Retrospective Pool-Subset Simulation on `xbabe2` CPU)**: How do guidance selection and candidate pool expansion behave when partitioning each static 1,152-candidate pool into an initial 384-candidate chronological subset (`M0` by `birth_seq_min`) and two 16-candidate waves (`M1, M2`) from the remaining 768 candidates?

---

## 2. Setup

- **Cohorts & Data Closure**:
  - `DL-Train`: `1,219` regression contexts across `205` dataset families (`/home/ubuntu/jope-data/h50/input/selected.json`).
  - `DL-Val` (`Native91`): `91` regression contexts across `19` disjoint dataset families (`0` family overlap with `DL-Train`).
  - `Sealed / Meta-Test / Holdout`: Never accessed (`has_meta_test == False`, `source_split in ("train", "inner")`, `run_outer_test == False`).
  - `Harvested Self-Play Promotions`: `3,449` inner-CV GP promotions harvested from `xbabe1:/home/ubuntu/jope-data/flywheel-full-20261030/run-20261030/promotions.jsonl` (snapshot preserved at `xbabe1:/home/ubuntu/jope-data/flywheel-full-20261030/stop-snapshot-20261103T1222Z/`), restricted strictly to `DL-Train` (`977` contexts across `205` families; `0` `DL-Val` rows).
- **Anchor & Frozen Backbone Verification**:
  - `R882` (`round-882.pt`, SHA-256 `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`) verified bit-exact on `Native91`: context-mean utility = `1.1377941898164525` (`diff = 0.00e+00`), equal-family mean utility = `1.153615`.
  - `DOPE` SHA-256 (`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`) verified bit-identical before and after all GPU updates.
- **Seeds & Statistical Protocol**:
  - 3 seeds (`252101, 252102, 252103`), 5 family-grouped outer folds per seed (`15` cells/arm in Stages 2–3), `10,000`-draw paired family-cluster bootstrap (`boot_seed = 252150`), Holm-Bonferroni step-down multiplicity correction, and `400`-iteration MM Bradley-Terry Elo (`1500` anchor).

---

## 3. Outcome & Key Metrics

### 3.1 Stage 1 (`stage1_hurdle_sos_critic.py` — Decomposed Hurdle Critic & Active `2+2` Reveal)

| Arm | `DL-Train` Fam Utility (`205` fam) | `DL-Train` $\Delta_{\text{R882}}$ (`95% CI`, $p_{\text{Holm}}$) | `DL-Val` Fam Utility (`19` fam) | `DL-Val` $\Delta_{\text{R882}}$ (`95% CI`, $p_{\text{Holm}}$) | `DL-Val` $\Delta_{\text{G025}}$ | `DL-Val` W/T/L | `DL-Val` Elo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`R882` (Full Pool 1,152)** | `1.385355` | `0.000000` (anchor) | `1.153615` | `0.000000` (anchor) | `0.000000` | `0/19/0` | `1500.0` |
| **`GATED_025` (`H217` Passive 4-Reveal)** | `1.385416` | `+0.000061` (`[-0.00007, +0.00014]`, `p=0.296`) | `1.153615` | `+0.000000` (`[0.0, 0.0]`, `p=1.00`) | `0.000000` | `0/19/0` | `1500.0` |
| **`HURDLE_SOS_PRIOR` (0 Extra Reveals)** | `1.399253` | `+0.013897` (`[+0.00310, +0.02783]`, `p=0.0042`) | `1.156383` | `+0.002768` (`[-0.00080, +0.00821]`, `p=0.368`) | `+0.002768` | `3/12/4` | `1493.4` |
| **`GATED_HURDLE_SP` (Passive 4-Reveal)** | `1.399121` | `+0.013765` (`[+0.00310, +0.02761]`, `p=0.0039`) | `1.156191` | `+0.002576` (`[-0.00099, +0.00802]`, `p=0.368`) | `+0.002576` | `2/14/3` | `1493.4` |
| **`ACTIVE_2PLUS2_HURDLE` (`W4` Active 4-Reveal)** | `1.469325` | `+0.083969` (`[+0.04263, +0.13862]`, `p=0.0004`) | `1.168790` | `+0.015174` (`[+0.00508, +0.02638]`, `p=0.0004`) | `+0.015519` | `9/8/2` | `1527.9` |

### 3.2 Stage 2 (`stage2_rl_gradsmooth_gpu.py` — Neural Adapter Continuation on `xbabe1` RTX 4090)

| Arm (`15` cells $\times$ `40` updates) | `DL-Train` Fam Utility | `DL-Train` $\Delta_{\text{R882}}$ (`95% CI`, $p$) | `DL-Train` $\Delta_{\text{CTRL}}$ | `DL-Val` Fam Utility | `DL-Val` $\Delta_{\text{R882}}$ (`95% CI`, $p$) | `DL-Val` $\Delta_{\text{CTRL}}$ | `DL-Val` W/T/L |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`R882` (Frozen Anchor)** | `1.385355` | `0.000000` | `+0.031216` | `1.153615` | `0.000000` | `+0.031266` | `0/19/0` |
| **`CTRL_H170` (`+2.0` Cliff + Zero `sos_prior`)** | `1.354139` | `-0.031216` (`[-0.05098, -0.01409]`, `p=0.9997`) | `0.000000` | `1.122349` | `-0.031266` (`[-0.06300, -0.00815]`, `p=0.9996`) | `0.000000` | `3/7/9` |
| **`RL_HURDLE_ADV` (Hurdle `sos_prior` + PL Adv + KL)** | `1.399725` | `+0.014370` (`[+0.00359, +0.02825]`, `p=0.0012`) | `+0.045589` | `1.156383` | `+0.002768` (`[-0.00080, +0.00821]`, `p=0.1228`) | `+0.034034` | `3/12/4` |
| **`RL_PCGRAD_EMA_SHADOW` (`PCGrad-EMA` + `AGC` + Shadow)** | `1.399448` | `+0.014093` (`[+0.00328, +0.02805]`, `p=0.0017`) | `+0.045309` | `1.156383` | `+0.002768` (`[-0.00080, +0.00821]`, `p=0.1228`) | `+0.034034` | `3/12/4` |

### 3.3 Stage 3 (`stage3_proposer_dpo_gpu.py` — Offline Parent$\to$Child Pool Re-Ranking on `xbabe3` RTX 3090)

| Proposer Arm | Split | Pairwise Sibling Acc (`pair_acc`) | Top-4 Promotion Yield (`q_beat@4`) | Expected Tries ($\mathbb{E}[\text{tries}]$) | Best-of-4 Utility `prom_u@4` ($\Delta_{\text{R882}}$, $p$) | Elo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`R882_RANK_PROPOSER`** | `DL-Train` | `51.47%` | `62.10%` | `8.09` | `1.609212` (anchor) | `1500.0` |
| **`UNIFORM_MUTATION`** | `DL-Train` | `49.86%` | `65.00%` | `7.03` | `1.628789` (`+0.01958`, `p=0.0014`) | `1536.5` |
| **`AWR_DELTA_CRITIC`** | `DL-Train` | `63.94%` | `63.36%` | `7.44` | `1.610115` (`+0.00090`, `p=0.3452`) | `1520.8` |
| **`PROPOSER_DPO_DELTA`** | `DL-Train` | **`66.83%`** | **`68.14%`** | **`6.79`** | `1.627798` (`+0.01859`, `p=0.0059`) | **`1536.9`** |
| **`R882_RANK_PROPOSER`** | `DL-Val` | `51.50%` | `61.54%` | `7.43` | `1.277656` (anchor) | `1500.0` |
| **`UNIFORM_MUTATION`** | `DL-Val` | `50.92%` | `67.03%` | `6.83` | **`1.311304`** (`+0.03365`, `p=0.0016`) | **`1582.3`** |
| **`AWR_DELTA_CRITIC`** | `DL-Val` | `66.23%` | `71.79%` | `6.15` | `1.300496` (`+0.02284`, `p=0.0064`) | `1533.4` |
| **`PROPOSER_DPO_DELTA`** | `DL-Val` | **`69.39%`** | **`76.56%`** | **`5.63`** | `1.301270` (`+0.02361`, `p=0.0010`) | `1558.3` |

### 3.4 Stage 4 (`stage4_closed_loop_2x2_flywheel.py` — Retrospective Pool-Subset Simulation)

| Pool-Subset Simulation Cell | `DL-Train` Fam Utility | `DL-Val` Fam Utility | Relation to Full-Pool `R882` (`1.153615` `DL-Val`) |
| :--- | :---: | :---: | :--- |
| **`CELL_G0_M0` (384-cand `birth_seq_min` subset)** | `1.330884` | `1.073732` | `-0.079883` below full-pool `R882` (subset restriction) |
| **`CELL_G1_M0` (Active `2+2` on 384-cand subset)** | `1.430760` | `1.132886` | `-0.020729` below full-pool `R882` |
| **`CELL_G0_M1` (384 + 16 uniform pool cands)** | `1.342378` | `1.085999` | `-0.067616` below full-pool `R882` |
| **`CELL_G1_M1` (384 + 16 DPO-ranked pool cands)** | `1.459836` | `1.169142` | Matches full-pool `ACTIVE_2PLUS2_HURDLE` (`1.168790`) |
| **`CELL_G0_M2` (384 + 32 uniform pool cands)** | `1.345828` | `1.087813` | `-0.065802` below full-pool `R882` |
| **`CELL_G2_M2` (384 + 32 DPO-ranked pool cands)** | `1.467601` | `1.169142` | Identical to `CELL_G1_M1` on `DL-Val` (`Cycle2 - Cycle1 = 0.0`) |

---

## 4. Red-Team Verdicts & Guardrails (`~/nudges/hw_t1305_agy_h252_redteam.md`)

1. **Stage 1 Verdict — `EXPLORATORY_UNPREREGISTERED_CANDIDATE` (`PENDING_H254_REPLICATION`)**:
   - `ACTIVE_2PLUS2_HURDLE` (`+0.01552` vs. `GATED_025` / `+0.01517` vs. `R882` on `DL-Val`, `95% CI [+0.00508, +0.02638]`, $p_{\text{Holm}} = 0.0004$) is **not** claimed as a confirmed breakthrough because Stage 1 was unpreregistered, its constants were hand-set shortly before the run, and `Native91` (`DL-Val`) is a heavily reused dev set.
   - `common_h252.py` and `stage1_hurdle_sos_critic.py` are **frozen byte-exact** for `rd_argon`'s independent, preregistered frozen-constant replication in **`H254`**.
2. **Stage 2 Verdict — `NO_LIFT_VS_R882` (`DAMAGE_CONTROL_ONLY_VS_CTRL_H170`)**:
   - On `DL-Val`, `RL_HURDLE_ADV` and `RL_PCGRAD_EMA_SHADOW` tie `R882` (`+0.002768`, `95% CI [-0.000798, +0.008208]`, covers `0`, $p = 0.1228$). Their `+0.03403` (`DL-Val`) / `+0.04559` (`DL-Train`) improvement is relative to `CTRL_H170` (which loses `-0.03127` to `R882`), acting as damage control rather than net validation lift over `R882`.
   - Any future GPU stage on `xbabe1` must acquire `flock` on `~/qualification.lock`.
3. **Stage 3 Verdict — `OFFLINE_POOL_RERANKING_DIAGNOSTIC` (`MIXED_VS_UNIFORM`)**:
   - `PROPOSER_DPO_DELTA` improves pairwise sibling accuracy (`69.39%` vs. `51.50%`) and `q_beat@4` (`76.56%` vs. `61.54%`) over `R882_RANK_PROPOSER`, but trails `UNIFORM_MUTATION` on `DL-Val` `prom_u@4` (`1.3013` vs. `1.3113`).
4. **Stage 4 Verdict — `RETROSPECTIVE_POOL_SIMULATION_ONLY` (`NO_FLYWHEEL_SYNERGY_CLAIM`)**:
   - Stage 4 is a retrospective pool-subset simulation (`CELL_G0_M0` is a 384-candidate subset starting at `1.073732` vs. full-pool `R882 = 1.153615`; `Cycle2 - Cycle1 = 0.0` on `DL-Val`; `rust_gp_s3_jtt1_verification` was a `1e-6s` import check with `run_outer_test=False`). No flywheel synergy is claimed.

---

## 5. Next Step

- **`H254` (`rd_argon`)**: Independent, preregistered frozen-constant replication of `ACTIVE_2PLUS2_HURDLE` vs. `GATED_025` and `R882` with fresh fold seed `254140`, seeds `254101–254103`, within-context permuted-fitness negative control, and $\pm 50\%$ constant sensitivity audits.
