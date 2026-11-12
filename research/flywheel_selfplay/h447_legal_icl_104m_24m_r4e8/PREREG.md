# H447 — Pre-Registration: Legal `104M + 24M` Plain-`ICL` Warm-Continuation & `R4-e8` `[X, S]` Champion Freeze

- **Wave:** `H447` (`h447_legal_icl_104m_24m_r4e8`)
- **Owner:** `@agy_apex_w10`
- **Cohorts:**
  - **Pretraining Banks (`H446` @ `c6cec7a3f5`, `0` holdout or `AUDIT` rows):**
    - `all_legal_excl_p201b_bank.npz` (**Primary `WC` training bank**, `1,368` distinct datasets / `1,319` families / `35,137,085` cells, `P201b` family-disjoint so `WC` vs `BASE` on `P201b excl10` is strictly out-of-sample per peer review `B1`, `sha256 = bc5c7d6cd1f42096d0f3b31ac95686e81e7bfc439d07f42f1b6992f20b35763e`)
    - `all_legal_bank.npz` (`1,609` distinct datasets / `1,509` families / `39,621,927` cells, `MAX_FEATS = 400`, `85` wide `F > 100`, `sha256 = 7d4164f9a9b5e618a2c47b8437c7e06ef87b1de11f3fd36ede2a48d7bfc2dc04`)
  - **Base Checkpoint Bank Overlap Audit (`R2`):** `fit_icl_bank.npz` (`419` tables, `sha256 = f6e283cebf9c7e40cd62b7cedf8f1283d010f7dc5feff0ab5045d0b2fd8e8ce7`) has **`0 / 106` context or family overlap with `FIT-SEL`** and **`0 / 201` context or family overlap with `P201b`**.
  - **Design Sanity Panel:** `FIT-SEL` (`106` core contexts / `88` `H431` `clean_non_copy` contexts)
  - **Primary Out-of-Sample Promotion Panel:** `P201b excl10` (`191` unprobed families out of `201` `P201b` families; `PROBED_P201B.txt` `10` probed prefixes excluded; `0` `P201b` tables in `all_legal_excl_p201b_bank.npz` or `fit_icl_bank.npz`)
  - **Untouched Holdouts (`0` opens):** `APEX-CONFIRM` (`142` families), `dev91-audit-v1` (`91` contexts / `19` families), and `DL-Holdout` / `sealed`

---

## 1. Scientific Motivation & Peer-Review Alignment (`A-024`, `A-025`, `REVIEW_H447_40af8f1a99`)

1. **Plain `ICL` Architecture (`Zero` Untested `CW-4` Switches):** Per `@cl_fwriter`'s peer guidance (`A-024`), `H447` uses the exact unmodified plain `ICL` architecture (`icl_model.ICL`, sizes `"L"` = `104,369,792` params and `"M"` = `23,668,608` params; combined `128,038,400` params vs. `kumo_tabular_large_e8.v1`'s `215,683,543` params).
2. **Cross-Scale `R4-e8` (`0.60 * 104M + 0.40 * 24M`) `[X, S]` Structural Residual Ensemble (`N2`):**
   - Base checkpoints trained on `fit_icl_bank.npz` (`419` tables, `0` overlap with `FIT-SEL` or `P201b`):
     - `104M` `ICL_L_g1_w2_ckpt_h030.pt` (`sha256 = 637bc123db18ca027fb46f5113a8f778f3aafb703fecb8c0edc91da7faeebffc`)
     - `24M` `ICL_M_r3_ckpt_h040.pt` (`sha256 = eb3a0903a7da136e2cdedd1fa3f1863fd351121d49f50a4b49e5b0e6d57e9ae4`)
   - The `0.60 / 0.40` cross-scale member weights were fixed a priori before any evaluation output (~log-parameter / capacity prior weighting `4.4x` larger `104M` at `0.60` and `24M` at `0.40`).
   - **`R4-e8`** matches `kumo_tabular_large_e8.v1`'s `8`-member inference budget (`4` `[X, S]` residual members from `104M` under `tag = "R4E8_L"` + `4` `[X, S]` residual members from `24M` under `tag = "R4E8_M"`, retaining all `d_S` structural columns `S`).
3. **Out-of-Sample GPU Warm-Continuation (`B1`) + Exact Token Accounting:**
   - Per `@cl_fwriter` `B1`, `warm_continue_icl.py` warm-continues `104M` (`L`, on `xbabe2` `RTX 3090`) and `24M` (`M`, on `xbabe1` `RTX 4090`) EMA weights strictly on **`all_legal_excl_p201b_bank.npz` (`1,368` `P201b`-disjoint non-`dev91` tables, `3.26x` more real tables and `3.77x` more real cells than `fit_icl_bank.npz`)** to `--target-cell-tokens 1000000000` (`1.0B` cell-tokens per model = **`+2.0B` total GPU adaptation cell-tokens**), mixing `35%` pre-cached direct `[X, S]` `R4` residual episodes (`0` query target leakage) + `65%` multi-table `LegalRealBank` & `PROBS_V3S` (`--real-frac 0.60 --p-resid 0.50`, cosine LR `3e-5 -> 3e-6`, `EMA = 0.998`, `bfloat16` AMP).
   - Every training step logs exact integer `cum_cell_tokens`, `cum_row_tokens`, `cum_real_cell_tokens`, and `approx_train_flops = 6 * params * cum_cell_tokens` into `wc_manifest.json`.
4. **H445 Linkage (`R1`) & Contamination Reporting Rule for Wave `H448` (`dev91-audit-v1`, Zero `dev91` Files Opened in `H447`):**
   - `H448` unlocks only if `H445/results/READOUT_DEV.json` is not `HARM` (recording `"following_null_h445_dev": true` if `NULL`), or if the owner's Stage-V receipt explicitly carries `OWNER_DIRECT_D91` (`H448/PREREG.md` §4.1).
   - Inside Wave `H448` Stage-V (`views-search` `X_search` features ONLY — `0` `dev91` `AUDIT` target reads), `H448` runs `screen_h425.screen` against the frozen champion's `training_bank` (`fit_icl_bank.npz` or `all_legal_excl_p201b_bank.npz`) and reports **both** the **Headline Strict Clean-Holdout Fairness Comparison** (`H425` `without_feature` clear slice Elo vs `kumo_tabular_large_e8.v1`) and the **Canonical 91-Context / 19-Family `dev91-audit-v1` Board Score** (`scores/h448.APEX_FAIR_CHAMPION.jsonl`).

---

## 2. Candidate Arms & `FROZEN_FAIR_CHAMPION.json` Promotion Rule (`P201b excl10`, `n = 191`)

Inside the frozen 5-block guarded 2-layer stack (`guard_and_stack_2l` + `M7` + `K3` + `F4` + `ICL`):

1. **`GM_CW12` (`NO_ICL`):** Stored GP pool + `guard_and_stack_2l` + `M7` + `K3` + `F4` (`0` neural ICL columns).
2. **`STACK_R4_M20`:** `GM_CW12` + `CW-5` `R4` (`24M` `ICL_M_r3_ckpt_h020.pt`, `e3c8fd99...`).
3. **`STACK_R4_E8_BASE`:** `GM_CW12` + `R4-e8` (`0.60 * 104M ckpt_h030.pt + 0.40 * 24M ckpt_h040.pt`).
4. **`STACK_R4_E8_WC`:** `GM_CW12` + `R4-e8` (`0.60 * 104M_wc + 0.40 * 24M_wc` warm-continued on `all_legal_excl_p201b_bank.npz`, `0` `P201b` tables).

**Deterministic Freeze Rule (`eval_h447.py` -> create-once `FROZEN_FAIR_CHAMPION.json`):**
- Among evaluated `ICL` arms (`STACK_R4_E8_WC` if run, `STACK_R4_E8_BASE`, `STACK_R4_M20`), select the highest-priority arm (`WC > BASE > M20`) that achieves:
  1. Non-harm on `FIT-SEL clean-88` (`clipped_mean_gain >= -0.002` vs `GM_CW12`), and
  2. Out-of-sample improvement on `P201b excl10` (`n = 191` unprobed families, `0` training overlap): `STACK_R4_E8_WC` is promoted over `STACK_R4_E8_BASE` iff `clipped_mean(WC - BASE) > 0` and `clipped_mean(WC - GM_CW12) >= 0` on `P201b excl10`; otherwise `STACK_R4_E8_BASE` is promoted over `STACK_R4_M20` iff `clipped_mean(BASE - M20) >= 0` and `clipped_mean(BASE - GM_CW12) >= 0` on `P201b excl10`. If no `ICL` arm improves over `GM_CW12` on `P201b excl10`, freeze `GM_CW12`.

---

## 3. Integrity Gates (`gates_h447.sh`, `6/6` Required)

1. **`g1_prereg_before_output`:** `PREREG` commit on `origin/v1/next` predates every output file in `$R`.
2. **`g2_pinned_stage_and_inputs`:** Published stage `CODE.sha256` verifies, `mode == "pub"`, and all input banks & checkpoints match pinned SHA-256 digests.
3. **`g3_selftests`:** `warm_continue_icl.py selftest`, `cols_r4_e8.py selftest`, and `eval_h447.py selftest` pass (`0` query target leakage, `FORBIDDEN_COHORTS` refused, token counters verified).
4. **`g4_exact_replays_and_completeness`:** `FIT-SEL` (`106/106`) and `P201b` (`201/201`) `SXB` baseline exact replay (`max_w_diff == 0`, `max_u_diff <= 1e-9`) and `>= 95%` column completeness (`>= 101/106` on `FIT-SEL`, `>= 191/201` on `P201b`).
5. **`g5_zero_holdout_opens`:** `0` `APEX-CONFIRM` opens, `0` `dev91` opens (`no_d91_lock == true`), and `sandbox/tables == 661`.
6. **`g6_pre_readout_ready`:** Pre-readout gate passes before `eval_h447.py readout` writes create-once `FROZEN_FAIR_CHAMPION.json`.
