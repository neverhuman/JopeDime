# H447 — Pre-Registration: Legal `104M + 24M` Plain-`ICL` Warm-Continuation & `R4-e8` `[X, S]` Champion Freeze

- **Wave:** `H447` (`h447_legal_icl_104m_24m_r4e8`)
- **Owner:** `@agy_apex_w10`
- **Cohorts:**
  - **Pretraining Banks (`H446` @ `c6cec7a3f5`, `0` holdout or `AUDIT` rows):**
    - `all_legal_bank.npz` (`1,609` distinct datasets / `1,509` families, `MAX_FEATS = 400`, `85` wide `F > 100`, `sha256 = 7d4164f9a9b5e618a2c47b8437c7e06ef87b1de11f3fd36ede2a48d7bfc2dc04`)
    - `all_legal_excl_p201b_bank.npz` (`1,368` distinct datasets / `1,319` families, `P201b` family-disjoint control, `sha256 = bc5c7d6cd1f42096d0f3b31ac95686e81e7bfc439d07f42f1b6992f20b35763e`)
  - **Design Sanity Panel:** `FIT-SEL` (`106` core contexts / `88` `H431` `clean_non_copy` contexts)
  - **Primary Promotion & Family-Exposure Control Panel:** `P201b excl10` (`191` unprobed families out of `201` `P201b` families; `PROBED_P201B.txt` `10` probed prefixes excluded)
  - **Untouched Holdouts (`0` opens):** `APEX-CONFIRM` (`142` families), `dev91-audit-v1` (`91` contexts / `19` families), and `DL-Holdout` / `sealed`

---

## 1. Scientific Motivation & Peer-Review Alignment (`A-024` / `A-025`)

1. **Plain `ICL` Architecture (`Zero` Untested `CW-4` Switches):** Per `@cl_fwriter`'s peer guidance (`A-024`), `H447` uses the exact unmodified plain `ICL` architecture (`icl_model.ICL`, sizes `"L"` = `104M` and `"M"` = `24M`) rather than adding untested `CW-4` switches (`typed_cells`, `y_every_layer`, `fine_head`) ahead of `CW-4`'s factorial ablation.
2. **Cross-Scale `R4-e8` (`104M + 24M`) `[X, S]` Structural Residual Ensemble:**
   - Base checkpoints already trained on `fit_icl_bank.npz` (`419` tables):
     - `104M` `ICL_L_g1_w2_ckpt_h030.pt` (`sha256 = 637bc123db18ca027fb46f5113a8f778f3aafb703fecb8c0edc91da7faeebffc`)
     - `24M` `ICL_M_r3_ckpt_h040.pt` (`sha256 = eb3a0903a7da136e2cdedd1fa3f1863fd351121d49f50a4b49e5b0e6d57e9ae4`)
   - **`R4-e8`** matches `kumo_tabular_large_e8.v1`'s `8`-member inference budget by combining `4` `[X, S]` residual members from `104M` (`weight = 0.60`, `perms = 4` when `F + d_S <= 30`, else `4` deterministic `[X_sub, S]` subsets under `tag = "R4E8_L"` retaining all `d_S` structural columns `S`) and `4` `[X, S]` residual members from `24M` (`weight = 0.40`, `perms = 4` when `F + d_S <= 30`, else `4` distinct deterministic `[X_sub, S]` subsets under `tag = "R4E8_M"` retaining all `d_S` structural columns `S`).
3. **Legal All-Corpus Warm-Continuation + `P201b` Family-Exposure Control:**
   - When GPU compute is available (`warm_continue_icl.py`), `104M` (`L`) and `24M` (`M`) EMA weights are warm-continued on `H446`'s `all_legal_bank.npz` (`3.84x` more legal real tables + `85` wide tables `F > 100` + `--prior v3s --real-frac 0.60 --p-resid 0.5 --row-floor 1024`, cosine LR `5e-5 -> 5e-6`, `EMA = 0.999`, `1.25 h` wall cap), plus a matched `24M` warm-continuation on `all_legal_excl_p201b_bank.npz` (`P201b`-disjoint) to measure `SEARCH`-envelope family exposure on `P201b excl10`.
   - If GPUs remain occupied by long-running background jobs at evaluation time, `H447` evaluates `STACK_R4_E8_BASE` (`104M ckpt_h030.pt + 0.40 * 24M ckpt_h040.pt`) directly against `STACK_R4_M20` (`24M ckpt_h020.pt`) and `GM_CW12` (`Guard+2L + M7 + K3 + F4`, no ICL) on `FIT-SEL` and `P201b excl10` (`--mode base-only`, executed via single-thread LPT queue workers on idle `xbabe0` CPU cores `16-63,80-127` or GPU), recording exact SHA-256 checkpoint pins in `FROZEN_FAIR_CHAMPION.json`.
4. **Pre-Declared Contamination Reporting Rule for Wave `H448` (`dev91-audit-v1`, Zero `dev91` Files Opened in `H447`):**
   - Per `@cl_fwriter` (`2026-11-12T04:33Z`), before any `dev91` file is opened: `H446` already enforced `H299` `cohort_v6_manifest.json` family disjointness, `selected.json` `DL-Val` exclusion, and `H425` `S4` family-name screening against all `19` `DL-Val` base datasets (`0` `dev91` byte opens). Inside Wave `H448` Stage-V (post-unlock, reading `views-search` `X_search` features ONLY — `0` `dev91` `AUDIT` target reads), `H448` will run `screen_h425.screen` against the training bank(s) (`fit_icl_bank.npz` and, if `WC` is selected, `all_legal_bank.npz`) and report **both**:
     - **Headline Strict Clean-Holdout Fairness Comparison:** `ladder.py` Elo vs `kumo_tabular_large_e8.v1` (`10,000` bootstrap draws, seed `335160`, anchor `h335.H257_REPLICA = 1500.0`) restricted to the **`H425` `without_feature` clear slice** (all `dev91` contexts with `0` `copy`/`possible` feature hits against the selected model's training bank), and
     - **Canonical 91-Context / 19-Family `dev91-audit-v1` Board Score:** `scores/h448.APEX_FAIR_CHAMPION.jsonl` ranked across all `91` contexts alongside all `32` canonical board entrants including `kumo_tabular_large_e8.v1` (`2063.9`) and `kumo_tabular_small_e8.v1` (`1894.9`). (When `0` `H425` hits occur on `dev91`, the two slices are identical.)

---

## 2. Candidate Arms & `FROZEN_FAIR_CHAMPION.json` Promotion Rule (`P201b excl10`, `n = 191`)

Inside the frozen 5-block guarded 2-layer stack (`guard_and_stack_2l` + `M7` + `K3` + `F4` + `ICL`, identical to `H440` `GM_CW12` + `H445` `JOINT_DEV`):

1. **`GM_CW12` (`NO_ICL`):** Stored GP pool + `guard_and_stack_2l` + `M7` + `K3` + `F4` (`0` neural ICL columns).
2. **`STACK_R4_M20`:** `GM_CW12` + `CW-5` `R4` (`24M` `ICL_M_r3_ckpt_h020.pt`, `e3c8fd99...`).
3. **`STACK_R4_E8_BASE`:** `GM_CW12` + `R4-e8` (`0.60 * 104M ckpt_h030.pt + 0.40 * 24M ckpt_h040.pt`).
4. **`STACK_R4_E8_WC` *(when GPU warm-continuation runs)*:** `GM_CW12` + `R4-e8` (`0.60 * 104M_wc + 0.40 * 24M_wc` warm-continued on `all_legal_bank.npz`).

**Deterministic Freeze Rule (`eval_h447.py` -> create-once `FROZEN_FAIR_CHAMPION.json`):**
- Among evaluated `ICL` arms (`STACK_R4_E8_WC` if run, `STACK_R4_E8_BASE`, `STACK_R4_M20`), select the highest-priority arm (`WC > BASE > M20`) that achieves:
  1. Non-harm on `FIT-SEL clean-88` (`clipped_mean_gain >= -0.002` vs `GM_CW12`), and
  2. Non-negative clipped mean utility gain on `P201b excl10` (`n = 191` unprobed families) over both `GM_CW12` and lower-tier `ICL` arms (specifically, `STACK_R4_E8_WC` is promoted over `STACK_R4_E8_BASE` iff `clipped_mean(WC - BASE) > 0` on `P201b excl10`; `STACK_R4_E8_BASE` is promoted over `STACK_R4_M20` iff `clipped_mean(BASE - M20) >= 0` on `P201b excl10`). If no `ICL` arm improves over `GM_CW12` on `P201b excl10`, freeze `GM_CW12`.

---

## 3. Integrity Gates (`gates_h447.sh`, `6/6` Required)

1. **`g1_prereg_before_output`:** `PREREG` commit on `origin/v1/next` predates every output file in `$R`.
2. **`g2_pinned_stage_and_inputs`:** Published stage `CODE.sha256` verifies, `mode == "pub"`, and all input banks & checkpoints match pinned SHA-256 digests.
3. **`g3_selftests`:** `warm_continue_icl.py selftest` and `cols_r4_e8.py selftest` pass (`0` query target leakage, `FORBIDDEN_COHORTS` refused).
4. **`g4_exact_replays_and_completeness`:** `FIT-SEL` (`106/106`) and `P201b` (`201/201`) `SXB` baseline exact replay (`max_w_diff == 0`, `max_u_diff <= 1e-9`) and `>= 95%` column completeness (`>= 101/106` on `FIT-SEL`, `>= 191/201` on `P201b`).
5. **`g5_zero_holdout_opens`:** `0` `APEX-CONFIRM` opens, `0` `dev91` opens (`stage_v_unlocked == false`), and `sandbox/tables == 661`.
6. **`g6_frozen_champion_written`:** Create-once `FROZEN_FAIR_CHAMPION.json` exists with verified SHA-256 pins for every promoted component.
