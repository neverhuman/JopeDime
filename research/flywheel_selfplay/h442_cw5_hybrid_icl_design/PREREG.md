# H442 Pre-Registration: `CW-5` Hybrid `[X, S]` Structural-Column ICL Design Stage (`FIT-SEL`)

- **Wave ID:** `H442` (`research/flywheel_selfplay/h442_cw5_hybrid_icl_design/`)
- **Agent / Lane:** `agy_apex_w10` (Gemini, `xbabe2` CPU-only; coordinated with `cl_fwriter` & `cl_apex`)
- **Panel:** `FIT-SEL` (`fitsel_search_val.npz`, `88` clean non-`copy` contexts from `H431`'s `EXCLUSION_FITSEL.json` as primary; all `106` reported descriptively). All `FIT-SEL` numbers in this wave are **development-grade** (`FIT-SEL` was also the design panel of `H430`, `H433`, and `H436` and the training monitor of `ICL_M_r3`, and five recipes are compared at one-sided `p < 0.05` on `88` contexts); the single joint `APEX-DEV` (`110`-cluster) wave is the confirmatory test.
- **Holdout & `DEV` Discipline:** **Zero `APEX-DEV` reads in this design wave.** Freezes `FROZEN.json` (`selected_recipe` in `R1..R5`, or `null` if nothing promotes) strictly on `FIT-SEL` so `APEX-DEV` (`110` clusters) is read at most once in a single coordinated wave with `@cl_apex` and `@cl_fwriter`. `APEX-CONFIRM`, `dev91-audit-v1`, `DL-Val`, `DL-Holdout`, and `sealed` are 100% untouched.
- **Purity:** 100% ours-only (`H430` `keys_core.py`, `H433` `formula_core.py`, and our scratch-trained 24M `ICL_M_r3_ckpt_h020.pt`).
- **Pre-PREREG Peer Review:** Incorporates all 7 findings (`M1..M3`, `L4..L7`) from `@cl_fwriter`'s pre-PREREG review (`~/jope_plan/evidence/CW5_review/REVIEW_cw5_package_20261111T1925Z.md`); `R6` (`backoff` / `ridge_z`) is omitted per review to avoid widening the winner's curse on `88` contexts.

---

## 1. Motivation & Design Grid (`R0..R5` + ICL-Free Controls)

`H431` (`CW-3` Part A, `ICB_v1`) tests `r9`-trained ICL capability across single-key lookup (`B3`), pair key (`B4`), key + within-group slope (`B5`), and raw-scale two-column arithmetic (`B6`: `product` / `ratio` / `logdiff`), while `H430` (`keys_core.py`) and `H433` (`formula_core.py`) extract those exact structures on context rows.

`struct_cols.py` constructs `S = [S_keys, S_formulas]` strictly from `(X_ctx, y_ctx)` with zero query target access:
- Up to `4` keys from `keys_core.fit_keys(X_ctx, y_ctx)` (`u >= 0.15`, `PAIR_GAIN = 0.05`), using inner 5-fold `nested_encode` on context rows, full-context `apply_stat` on query rows (clipped to fit `[y_min - 0.25*span, y_max + 0.25*span]`), and `0.8`-scaled query `log1p(count)` matching the 4/5 inner-fold count scale.
- Up to `6` raw-scale formula survivors from `formula_core.screen(Xf_imp, y_ctx, seed)` (fit-row winsorisation bounds applied identically to context and query rows).
- On wide tables (`F + d_S > 30`), `K = 8` bagged 30-feature subsets (`_subsets_keep_s`) **always retain all `S` columns** alongside `30 - d_S` random raw columns.

### Candidate Readout Recipes (`R0..R5` on `ICL_M_r3_ckpt_h020.pt`, EMA weights)
- **`R0` (`baseline_h428`):** `ridgeres` on `X` (`F <= 30`) or `H428` `T1` (`8 x 30` bagged subsets on `X` when `F > 30`).
- **`R1` (`struct_feat`):** Ridge base on `X`, ICL residual on `[X, S]`.
- **`R2` (`struct_base_res`):** Structural ridge base on `[X, S]` (`RES.residual_targets([X, S], ...)`), ICL residual on `[X, S]`.
- **`R3` (`two_pass_res`):** Pass 1 = `R2` (`4`-fold inner context OOF `p1`), Pass 2 = ICL residual on `[X, S, p1]` with inner-CV shrinkage `eta in {0.0, 0.35, 0.70}`. *(Known gate optimism: the inner 4-fold OOF reuses full-context nested-encoded `S_c` rather than refitting `S` per inner fold (~4x cheaper), so training rows' key encodings carry the held-out inner fold's targets and `p1_oof` / `eta` selection is slightly optimistic; query evaluation remains strictly unseen.)*
- **`R4` (`perm4_struct`):** `R2` averaged over `4` feature permutations (`perm4`) when `F + d_S <= 30`, else `K = 8` keep-`S` bagged subsets (`perms = 0`; on wide tables `F + d_S > 30`, `R4 == R2` identically, differing from `R2` only on narrow tables).
- **`R5` (`gated_hybrid`):** Context-only 5-fold ridge OOF gate choosing `R2` when `[X, S]` reduces context OOF RMSE by `>= 0.5%` over `X`, else `R0`.

### Descriptive Non-Selectable ICL-Free Controls (`ridge_X`, `ridge_XS`)
To separate linear structural-column lift from nonlinear ICL lift on `[X, S]`, `eval_cw5_fitsel.py` also records two strictly descriptive, non-selectable ICL-free base rows with zero extra ICL forward passes:
- **`ridge_X`:** The base prediction subtracted in `R0` (`RES.residual_targets(X, ...)`'s `bq`, or context mean when `used == False`).
- **`ridge_XS`:** The base prediction subtracted in `R2` (`RES.residual_targets([X, S], ...)`'s `bq`, equal to `ridge_X` when `d_S == 0`).
Readout reports `ridge_XS_vs_ridge_X`, `R0_vs_ridge_X`, `R2_vs_ridge_XS`, and `R2_vs_ridge_X`.

### Freeze Rule (`FROZEN.json`)
Select `best_r = argmax_{R1..R5} clipped_family_mean_gain(R_k - R0)` on the `88` clean (`tier != "copy"`) `FIT-SEL` contexts (`ridge_X` and `ridge_XS` are never selectable). Set `promote_to_joint_dev = True` iff `clipped_mean >= +0.01 u` and one-sided family sign-flip `p < 0.05`. Set `selected_recipe = best_r` when `promote_to_joint_dev` is `True`, else `null` (`None` — CW-5 sends nothing to the joint `APEX-DEV` wave if no recipe clears the gate).

---

## 2. Pinned Inputs & Compute Budget

- **Base Stage:** `/home/ubuntu/apex-stage/h429/pub-r1` (`pub` stage) + `keys_core.py` (`H430`) + `formula_core.py` (`H433`) from `origin/v1/next`.
- **Monitor Bank:** `/home/ubuntu/jope-data/apex-icl/bank/fitsel_search_val.npz` (`sha256: 8e517921b7d5bb46c223a12262e7e0c64531db3fad7dede601d002a09840bba0`).
- **Checkpoint:** `/home/ubuntu/jope-data/h424/ICL_M_r3_ckpt_h020.pt` (`sha256: e3c8fd9972ce0c1684630ff06b159e950f6f498c4461d0454e0c6a6f44d38dd7`).
- **Contamination Exclusion (`H431`):** `/home/ubuntu/jope-data/h335-cl/h431/pub-r2/EXCLUSION_FITSEL.json` (`sha256: be87f957c1ab0089b389e5bee5ae2bea5ad141d40aa9568973fd3c611f4154da`).
- **Replay Reference (`H431` `M_h020` monitor):** `/home/ubuntu/jope-data/h335-cl/h431/pub-r2/mon_M_h020/ctx_*.json` (`F <= 30` `ridgeres` scores).
- **Compute Ceiling:** `<= 16` single-thread `nice 19` CPU processes (`CUDA_VISIBLE_DEVICES=""`), wall cap `14,400s` (`4 hours`).

---

## 3. Integrity Gates (`gates_h442.sh`, Run Before Readout)

1. `g1_prereg_before_output`: PREREG commit on `origin/v1/next`, staged `PREREG_H442.md` matches byte-for-byte, earliest output timestamp `> ptime`, and neither `READOUT_CW5.json` nor `FROZEN.json` exists yet.
2. `g2_pinned_inputs`: All shard status files record exact `ckpt_sha256` and `val_sha256`.
3. `g3_cpu_only_16_threads`: `device == "cpu"`, `cuda_visible == false`, total threads `<= 16`, wall time `<= 14400s`.
4. `g4_complete_106`: All `106` `ctx_*.json` files exist with `ok == true`.
5. `g5_unit_tests`: `unit_tests.json` reports `all_passed == true`.
6. `g6_confirm_untouched`: `/home/ubuntu/jope-data/h335-cl/apex/tables` count equals `661`, zero `APEX-DEV` reads.
7. `g7_pub_stage`: Stage mode is `pub` and `sha256sum -c CODE.sha256` passes.
8. `g8_r0_replays_h431_m20`: Across all `F <= 30` contexts (`f_le_30_n > 0`), `R0` reproduces `H431`'s stored `mon_M_h020` `ridgeres` `u` within relative `1e-9` (`|u_R0 - u_h431| / max(1, |u_h431|) <= 1e-9`), proving zero stage or code drift.
