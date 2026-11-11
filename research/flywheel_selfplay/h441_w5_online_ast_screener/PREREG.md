# H441 Pre-Registration: Grammar-Complete Online AST Screener vs `H0.1-A` (`W5`, Clean `FIT-SEL`)

- **Wave ID:** `H441` (`research/flywheel_selfplay/h441_w5_online_ast_screener/`)
- **Agent / Lane:** `agy_apex_w10` (Gemini, `xbabe2` CPU-only)
- **Panel:** `FIT-SEL` (`88` clean non-`copy` contexts from `H431`'s `EXCLUSION_FITSEL.json` as primary; full `106` reported descriptively without `copy` contexts ever entering training).
- **Holdout & `DEV` Discipline:** **Zero `APEX-DEV` reads.** `APEX-CONFIRM`, `dev91-audit-v1`, `DL-Val`, `DL-Holdout`, and `sealed` are 100% untouched.
- **Purity:** 100% ours-only (`NumPy`, `scikit-learn` `HistGradientBoostingRegressor`, and stock `H0.1-A` neural screener baseline). Foundation models (`Kumo L`, `TabPFN 3.5`, `TabFM`) are never loaded, queried, or distilled.

---

## 1. Motivation & Hypotheses

In `H417` Gate G0 (`APEX_G0.g0.txt`), the stock `H0.1-A` neural screener (`SCR`, `feat412`) scored `2.3438` `AUDIT u` vs uniform random `2.4052` `AUDIT u` (`-0.0614 u` worse than random) while oracle `ORC` reached `2.4505` (`build_surrogate_wave = true`), because `feat412` is blind to `gpr`, `krr`, `ksmooth`, `histgbm2`, `oblivious`, `xtrees`, `rforest`, and `ensemble` heads and strips all `H420` extended stages (`lang_h420.py:297`).

`W5` (`w5_screener_core.py`) replaces `feat412` with:
1. **48-dim Grammar-Complete AST Encoder (`encode_program_ast`)** covering every stock and `H420` stage, head family, kernel (`rq`, `linear`, `periodic`), symbolic `map`/`derive` operator, capacity hyperparameter, and cost with zero stripped stages.
2. **16-dim SEARCH-Only Dataset Meta-Profile (`compute_dataset_meta_profile`)** including 2-way split linear vs tree probes.
3. **5-Member Bootstrap-by-Family `HistGradientBoostingRegressor` Cross-Family Prior** fit strictly out-of-family via 5-fold cross-family CV on clean (`tier != "copy"`) `FIT-SEL` families.
4. **Gen-0 Online AST Ridge Residual Adapter + Epistemic UCB Diversity Pick (`W5_ONLINE_UCB`)** adapting the cross-family prior to table `D`'s observed seed programs (`24` Gen-0 memo seeds; `12` `TBOTH` seeds).

### Pre-Registered Primary Hypotheses (on `clean_non_copy_88` `FIT-SEL`, Holm-Bonferroni FWER = 0.025)
- **Primary P1 (Gen-1+ Memo Pool Top-8 SEARCH Screening):** Given `24` observed Gen-0 seed evaluations per table, `W5_ONLINE_UCB` achieves higher family-mean Top-8 SEARCH utility than stock `H01A_ONLINE` (`H0.1-A` fold-0 neural screener + head-family online residual corrector) on the `88` clean `FIT-SEL` contexts (`T16.pair_stats["pass"] == True`).
- **Primary P2 (Downstream Held-Out `AUDIT` Stack Utility on `TBOTH`):** Given `12` seed candidates + `16` screened candidates (`28` total out of `m0` stored `H420_FIT-SEL` `TBOTH` candidates) stacked via `CA.bcar`, `W5_ONLINE_UCB` achieves higher held-out `AUDIT` stack utility than `H01A_ONLINE` on the `88` clean `FIT-SEL` contexts (`T16.pair_stats["pass"] == True`).
- **Holm-Bonferroni Step-Down (`alpha = 0.025`):** Sort `(p_P1, p_P2)` ascending; `p_(1) < 0.0125` and `p_(2) < 0.025`.

---

## 2. Pinned Inputs & Compute Budget

- **Base Stage:** `/home/ubuntu/apex-stage/h424/pub-r1` (`pub` stage, verified against `CODE.sha256`).
- **Contamination Exclusion File (`H431`):** `/home/ubuntu/jope-data/h335-cl/h431/pub-r2/EXCLUSION_FITSEL.json` (`sha256: be87f957c1ab0089b389e5bee5ae2bea5ad141d40aa9568973fd3c611f4154da`, `88` non-`copy` contexts).
- **`FIT-SEL` Bank:** `/home/ubuntu/jope-data/h335-cl/h424/dev/bank_FITSEL.npz` (`sha256: f8849ccb7343cfec1c0fd6688aaafef7580b85001046f2707fab58c5ecebe069`).
- **Stored `H420` Memo & `TBOTH` Matrices:** `/home/ubuntu/jope-data/h335-cl/runs/H420_FIT-SEL/gproot_AGG/memo` and `/home/ubuntu/jope_plan/evidence/H420/pulled/H420_FIT-SEL/H420_FIT-SEL`.
- **Compute Budget:** Single-thread CPU-only (`CUDA_VISIBLE_DEVICES=""`, `OMP/MKL/OPENBLAS=1`, `nice 19`), wall cap `3,600s` (`1 hour`; expected `< 5 minutes`).

---

## 3. Integrity Gates (`gates_h441.sh`, Run Before Readout)

1. `g1_prereg_before_output`: PREREG commit is on `origin/v1/next`, staged `PREREG_H441.md` matches byte-for-byte, earliest output timestamp strictly exceeds PREREG commit timestamp, and `READOUT_W5.json` does not yet exist.
2. `g2_pinned_inputs`: `APPLY_W5.json` records exact `bank_sha256` and `excl_sha256` matches and `n_total == 106`, `n_clean == 88`.
3. `g3_cpu_only`: `h01a_scores_FITSEL.meta.json` records `cuda_visible == ""` and wall time `<= 3600s`.
4. `g4_unit_tests`: `unit_tests.json` reports `all_passed == true`.
5. `g5_confirm_untouched`: `/home/ubuntu/jope-data/h335-cl/apex/tables` count equals `661`, zero `APEX-DEV` reads.
6. `g6_pub_stage`: Stage mode is `pub` and `sha256sum -c CODE.sha256` passes.

---

## 4. Amendment Policy

Code changes after `PREREG.md` is published on `origin/v1/next` are restricted strictly to fixing an unhandled crash before `READOUT_W5.json` is generated, documented in `AMENDMENT_k.md` on `origin/v1/next` before staging `pub-r<k+1>`.
