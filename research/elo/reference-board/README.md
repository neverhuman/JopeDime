# Reference board: stored scores of every reference model

**Never re-run a reference model.** TabFM, Kumo-Tabular, TabPFN 3.5 / v3, TabICL, TabDPT, Mitra, AutoGluon, CatBoost,
XGBoost, LightGBM, random forest, ridge and the 5 LLM regressors have already been scored on DL-Val91. The per-context
scores are in git (this directory). The full results, including raw predictions and the original producer files, are
in S3. Read them; do not fit, infer or download these models again.

Machine-readable form of this page: [`SCORE_LOCATIONS.json`](SCORE_LOCATIONS.json).

## 1. Where every reference-model score is

DL-Val91 = the 91 dev contexts / 19 families. They were scored under **three different row splits**. A number from
one split is never comparable with a number from another (section 2).

| Model | dev91 **two-fold** (H257, 2026-11-04) | dev91 **AUDIT** (H335 split, 2026-11-05) | dev91 **S3 single-split** (precomputed, 2026-10) | panel_v1, 240 datasets (2026-10-29) |
| --- | --- | --- | --- | --- |
| TabFM 1.0 | `h257.TabFM` 91/91 | `tabfm.v1` 91/91 | none (0/91 in S3) | not run |
| TabFM 1.0 ensemble | `h257.TabFM_Ensemble` 91/91 | `tabfm_ensemble.v1` 91/91 | none | not run |
| Kumo-Tabular large | `h257.Kumo_Tabular_L` 91/91 | `kumo_tabular_large_e8.v1` 91/91 | `s3.Kumo_Tabular_L_S3_Query1Split` 91/91 | `kumo_tabular_large_e8` 239/240 |
| Kumo-Tabular small | `h257.Kumo_Tabular_S` 91/91 | — | `s3.Kumo_Tabular_S_S3_Query1Split` 91/91 | — |
| TabPFN 3.5 | `h257.PriorLabs_TabPFN_3p5` (8 estimators) 91/91 | `tabpfn_v3p5.v1` (tabpfn 9.2.0 default, `n_estimators="auto"`) 91/91 | `s3.PriorLabs_TabPFN_3p5_S3_OuterTest` 91/91 | — |
| TabPFN 3.5 (1 estimator) | `h257.PriorLabs_TabPFN_3p5_Single` 91/91 | — | — | — |
| TabPFN v3 | — | `tabpfn_v3.v1` 91/91 | — | — |
| TabICL v2 | `h257.TabICL_v2` 91/91 | — | `s3.TabICL_v2_S3_Query1Split` 91/91 | — |
| TabDPT 1.3 | `h257.TabDPT_1p3` 91/91 | — | `s3.TabDPT_1p3_S3_Query1Split` 91/91 | — |
| Mitra v2 | `h257.Mitra_v2` 91/91 | — | `s3.Mitra_v2_S3_Query1Split` 91/91 | — |
| AutoGluon (300 s) | — | `autogluon_300s.v1` 91/91 | — | `autogluon` 240/240 |
| CatBoost / XGBoost / LightGBM / random forest / ridge | — | `catboost.v1`, `xgboost.v1`, `lightgbm.v1`, `random_forest.v1`, `ridge.v1` 91/91 each | — | 240/240 each |
| LLM regressors (Opus5p5_Extra, Gemini4_argon_high, GPT6p1_Sol_Max, GPT6_Astra_Max) | `h257.<name>` 91/91 | `h335.<name>` 91/91 | — | — |
| LLM regressor grok4p7_xhigh (`crates/jope-automl/src/grok4p7_xhigh.rs` at `1177c4f9c1`, frozen) | — | `h412.grok4p7_xhigh` 91/91 (scored once by H412) | — | — |
| JopeDime two-level GP stack (H364 XSTACK-2L on the H335 AGG output) | — | `h364.XSTACK_2L` 91/91 (scored once by H369) | — | — |

**Where each column is stored:**

| Column | In git (per-context scores) | In S3 (create-only; exact-version GET verified) |
| --- | --- | --- |
| dev91 two-fold | [`boards/dev91-s1-two-fold-v1/`](boards/dev91-s1-two-fold-v1/LADDER.md) `scores/h257.*.jsonl` | `s3://veox-jopedime-learning-use2/experiments/reference-board/dev91-s1-two-fold-v1/results-20261105T070830Z.tar.zst` (version `kT5xci1wsaPSYO8nz3mNSnolvdY6bSJU`). Holds the 5 original H257 files verbatim, including the LLM fold predictions |
| dev91 AUDIT | [`boards/dev91-audit-v1/`](boards/dev91-audit-v1/LADDER.md) `scores/*.jsonl` | `s3://veox-jopedime-learning-use2/experiments/reference-board/dev91-audit-v1/results-20261107T120005Z.tar.zst` (version `wSLUByLp0bS8RozTkAMOyBshyuaXPhwL`, 1,029 members). Holds all 25 score files and the 1,001 prediction files `preds/<model>/<cid>.f64` (11 reference models × 91). It supersedes the pre-H412 archive `results-20261105T140116Z.tar.zst` (version `0Ax4wTWMm5FEtd0UqGKtQZae9pHK2AcQ`), the pre-H369 archive `results-20261105T085053Z.tar.zst` (version `Hz8d5S8acwppcnoqjT.hssbC_pQgCxWc`) and the pre-resume archive `results-20261105T070823Z.tar.zst` (version `L0zjqB_SDwriDnJc_4yLthrNOPyQnkHj`). All three stay in S3; `ARCHIVE.json` records all four |
| dev91 S3 single-split | [`external/dev91-s3-single-split-v1.jsonl`](external/dev91-s3-single-split-v1.jsonl): 546 lines, each with the exact object URI and S3 version id | The original objects, left where the campaigns wrote them: `s3://veox-jopedime-learning-use2/experiments/gp-policy-dl/v2-trainonly/encodings/{kumo_tabular_l,kumo_tabular_s,tabicl2,tabdpt13,mitra_v2}/…/<stem>.npy` (last 3 floats = `[qr2, spread, logspread]`), and TabPFN 3.5 under **two** campaigns: `experiments/gp-teacher-v6/campaigns/regression-research-2k-fold0-3plus2-v1-20261001-tabpfn35-exact1/tabpfn35-exact/results/sha256/*.json` (50 contexts) and `…/regression-research-2k-per-fold-hwm-v1-20261001-incoming-v1-tabpfn35-exact1/…` (41 contexts). All 546 confirmed present by HEAD on 2026-11-05 |
| panel_v1 | [`external/panel_v1/`](external/panel_v1/): `all_results.jsonl` (3,834 method × dataset results, regression + binary), `elo_panel_v1.json`, `panel_v1.json` (dataset list) and the producer scripts | `s3://veox-jopedime-learning-use2/experiments/reference-board/external/panel_v1/results-20261105T070833Z.tar.zst` (version `SFPCcT4Rmeig0yMQjQrxwNVoYeaHnl9C`) |
| TRAIN board (P201b) | [`boards/p201b-train-audit-v1/`](boards/p201b-train-audit-v1/LADDER.md) | `s3://veox-jopedime-learning-use2/experiments/reference-board/p201b-train-audit-v1/results-20261105T070826Z.tar.zst` (version `20ND6QS_IFxlJTrYtNo_EhLNU53AkdIB`) |
| SOTArena Golden45 (45 public datasets) | [`../../../assets/regression/`](../../../assets/regression/SOURCES.md) (`sotarena-tabular-regression-v2/results/…/scores.csv.zst`, leaderboards) | — (in git) |

Each board directory has an `ARCHIVE.json` receipt (URI, version id, sha256, member count). The archives never contain
envelopes: TRAIN and DL-Val rows stay on the hosts.

**Who produced what:**
- **Two-fold:** H257 phase 2, `evaluate_tabular_foundation_dl_val91.py` and `evaluate_frontier_vs_h257_elo.py`. The
  code is on `origin/main` only (commit `571873a38c`, PR #2120), in
  `research/flywheel_selfplay/h257_phase2_subsample_gp_curriculum/`. That directory also has `REPORT.md` and
  `frontier_elo_comparison.json` (aggregate tables only, no per-context scores). Those files are not on `v1/next`.
  Imported unchanged by [`tools/import_h257_two_fold.sh`](tools/import_h257_two_fold.sh). Check: the family means of
  the 13 imported entrants equal H257's published means to 6 decimals.
- **AUDIT:** this directory's harness (`tools/score_reference.py`, `models.json`), plus the 12 H335 entrants imported
  by `tools/import_h335.py`, plus `h364.XSTACK_2L`, scored once by H369
  (`research/flywheel_selfplay/h369_dev91_stack_entrants/`; stored board AUDIT predictions, no entrant rescored),
  plus `h412.grok4p7_xhigh`, scored once by H412 (`research/flywheel_selfplay/h412_final_frontier_elo/`). H412 used
  the H335 frontier AUDIT harness with a fifth entrant compiled in. The four H335 entrants in that binary reproduce
  their 804 stored TRAIN records exactly, so the five frontier rows are comparable.
- **S3 single-split:** the gp-policy-dl v2 encoding campaign and the gp-teacher-v6 TabPFN 3.5 campaigns (2026-10).
  Indexed by [`tools/index_s3_single_split.sh`](tools/index_s3_single_split.sh).
- **panel_v1:** `~/jope_plan/elo/` on xbabe2, 2026-10-29. Copied unchanged.

**The three partial AUDIT runs were completed once, on owner instruction (2026-11-05).**
- **Before the stop:** `tabfm_ensemble.v1` (26/91), `kumo_tabular_large_e8.v1` (1/91) and `autogluon_300s.v1` (0/91)
  were stopped after the owner ruled "no recomputing".
- **The owner's instruction (just before the 07:29Z launch):** finish only these three, with the resumable driver.
- **Runs:** `tools/run_board.sh` was launched by the systemd wrappers `tools/run_dev91_resume_cpu.sh` (xbabe2, Kumo +
  AutoGluon, 07:29:54Z–08:40:55Z) and `tools/run_dev91_resume_gpu.sh` (xbabe3 RTX 3090, TabFM ensemble,
  07:29:54Z–08:17:32Z).
- **Nothing was run twice:** `run_board.sh` skips every (model, context) that already has a record, and the 8 complete
  reference models were not passed to it.
- **Checks after the run:**
  - All three are 91/91 `completed`, with no failure or timeout.
  - Each has one spec and harness identity across its 91 records.
  - The records from before the stop are an unchanged byte prefix of each file.
  - Every `preds_sha256` matches its prediction file (1,001 of 1,001).
  - Against sha256 snapshots taken before the launch, these are byte-unchanged: the 8 complete reference models'
    score files and their 728 prediction files (8 × 91), and the 12 `h335.*` files.
- **Not mixed in:** the H257 two-fold scores were not copied onto this board; they are a different split (section 2).

## 2. The three dev91 splits: why their numbers differ

All three cover the same 91 contexts, but they fit and score different rows. So a model's u, and every Elo built from
it, differ between them.

| Split | Rows fitted → rows scored | u per context | Failure |
| --- | --- | --- | --- |
| **two-fold** (`jope.s1-two-fold/v1`, H257) | half → other half, both directions | −(0.9 · mean log RMSE ratio + 0.2 · SD of the two) | −2.0 (also when the CV gate fails) |
| **AUDIT** ([`PROTOCOL.md`](PROTOCOL.md), H335) | SEARCH ≈70% → AUDIT ≈30% | −ln(RMSE_audit / RMSE_base) | 0 |
| **S3 single-split** | outer-train → outer-test (TabPFN 3.5), or a 75/25 subsampled support/query (≤768 / ≤256 rows; the encodings) | from one ratio; SD = 0 | — |

The S3 single-split numbers use different rows and give no fold-variance penalty. They are a record of what exists;
do not put them in a ladder with the other two.

## 3. Ladders (generated from the stored scores; `tools/ladder.py`)

| Board | Anchor = 1500 | Top of the table (Elo; family-mean u) |
| --- | --- | --- |
| [`dev91-audit-v1`](boards/dev91-audit-v1/LADDER.md) | `h335.H257_REPLICA` | `kumo_tabular_large_e8.v1` 2045 (1.936); `tabfm_ensemble.v1` 1899 (1.768); `tabfm.v1` 1873 (1.720); `tabpfn_v3p5.v1` 1820 (1.641); `tabpfn_v3.v1` 1786 (1.643); `h364.XSTACK_2L` 1725 (1.561); `h335.H335_AGG` 1706 (1.559); … `autogluon_300s.v1` 1526 (1.252); … `catboost.v1` 1480 (1.189); … `h412.grok4p7_xhigh` 1127 (0.864, last). 25 entrants, all 91/91 |
| [`dev91-s1-two-fold-v1`](boards/dev91-s1-two-fold-v1/LADDER.md) | `h257.Opus5p5_Extra` | `h257.Kumo_Tabular_L` 1976 (1.501); `h257.Kumo_Tabular_S` 1870; `h257.PriorLabs_TabPFN_3p5_Single` 1854; `h257.TabFM_Ensemble` 1840; … |
| [`p201b-train-audit-v1`](boards/p201b-train-audit-v1/LADDER.md) | `h335.H257_REPLICA` | TRAIN, development only: `h335.H335_AGG` 1675; … `catboost.v1` 1414 |

- Elo depends on the field and the anchor. Compare entrants within one table; use the head-to-head tables for effect
  sizes.
- The published H335 AUDIT-Elo (dev91 `H335_AGG` 1742.6) is the 12-entrant field.
  `tools/ladder.py boards/dev91-audit-v1 --only h335.…` reproduces it exactly. With the 11 reference models added,
  the field changes and `H335_AGG` reads 1708.0. With `h364.XSTACK_2L` added (H369, 24 entrants), it reads 1707.8.
  With `h412.grok4p7_xhigh` added (H412, 25 entrants), it reads 1705.7. The family-mean u of every stored entrant is
  unchanged.
- H257's own published two-fold table (`frontier_elo_comparison.json`: TabFM_Ensemble 1915.49, Kumo L 1980.90, H257
  1749.51, anchor R882) has a 25-entrant field. Only its 13 reference entrants are stored per context, so this
  board's two-fold ladder has 13 entrants and different numbers.
- **H335 vs Kumo is now measured on one split (AUDIT).** `h335.H335_AGG` vs `kumo_tabular_large_e8.v1`:
  - family-mean gain −0.376, so Kumo's AUDIT RMSE is about 31% lower;
  - Kumo is better in 18 of 19 families.
  - Head-to-head table: [`LADDER.md`](boards/dev91-audit-v1/LADDER.md).
  - No H335 entrant has a two-fold score.

## 4. Rules
1. **Read, never re-run.** Compare against a reference model by reading its stored scores (section 1) and running
   `tools/ladder.py`. Do not re-fit, re-infer or re-download it.
2. **Once per id.** A model id is scored on a board exactly once.
   - Failures are final.
   - A changed library version, weights file, config, seed, device or harness is a **new id**. `h257.TabFM` and
     `tabfm.v1` are separate ids because their harnesses differ.
   - [`models.json`](models.json) pins the versions and weights sha256 of this directory's harness. The driver
     refuses drift.
3. **One split per comparison.** Never mix columns of section 1.
4. **dev91 is a test cohort.** Our models are read on it only as locked candidates, under a create-once receipt in
   `~/jope_plan/ns/receipts/`. It is a reuse cohort (H257 tuned on it).
   - **Exposure log.** H381 (2026-11-05, LOCK `b9375201`): a 5-context direction check of two TRAIN arms (not entrants; nothing added to any board) read `ac07466e…`, `8eeca816…`, `016648dc…`, `6ecbb7d5…` and `b1a5c1a1…` ([H381 README](../../flywheel_selfplay/h381_rapid_lf_grow/README.md#dev91-5-dataset-direction-check-stored-board-rows-nothing-added-to-the-board)).
   - H410 (2026-11-06, LOCK `c828dab5`): micro-tests of deployment builders, a strong-learner library and a retrained scale-L screener on the **same 5 contexts**, adaptively tuned on under an owner overfitting waiver (not entrants; nothing added to any board). Treat these 5 as tuned-on for any later comparison ([H410 README](../../flywheel_selfplay/h410_deploy_population/README.md#dev91-the-5-exposed-contexts-stored-board-rows-nothing-added-to-the-board)).
   - H412 (2026-11-07, LOCK `8b368f93`): the frozen frontier entrant `grok4p7_xhigh` was scored once on **all 91** contexts as `h412.grok4p7_xhigh` (owner instruction 2026-11-06; nothing was chosen on dev91). No model of ours was read or rerun. The FINAL frontier study reads the stored rows only ([H412 FINAL_STUDY](../../flywheel_selfplay/h412_final_frontier_elo/FINAL_STUDY.md)).

## 5. Comparing a new model of ours
1. Choose the board:
   - `dev91-s1-two-fold-v1` only: Kumo small, TabICL, TabDPT, Mitra and TabPFN 3.5 with 1 estimator;
   - `dev91-audit-v1` only: the H335 entrants, TabPFN v3, AutoGluon and the classical libraries;
   - on both: TabFM, the TabFM ensemble, Kumo large, TabPFN 3.5 and the LLM regressors.
2. Produce one u per board context with that board's split, using the envelopes named in its `BOARD.json`. Write
   `boards/<board>/scores/<id>.jsonl` with `board`, `cid`, `envelope_sha256`, `model_id`, `status` and `utility`.
   - Python models: add a `kind` to `tools/score_reference.py` and an id to `models.json`, then run
     `tools/run_board.sh` (AUDIT boards).
   - GP or Rust pipelines: write the same fields.
3. Run `python3 tools/ladder.py <board root> --anchor <anchor> --focus <id> --write`. Commit the score file and the
   regenerated ladder.
4. Archive the board root with `tools/archive_board.sh <root> <board> --upload`, and commit the new `ARCHIVE.json`.

## 6. Host paths (board roots hold the envelopes and predictions; not in git)

| What | Where |
| --- | --- |
| dev91 AUDIT root | xbabe2 `~/jope-data/h335-stagev/refboard/dev91-audit-v1/` (copy on xbabe3, same path, used for the `tabfm.v1` and `tabfm_ensemble.v1` GPU runs; their scores and predictions were pulled back to xbabe2) |
| dev91 two-fold root | xbabe2 `~/jope-data/h257/refboard/dev91-s1-two-fold-v1/`; envelopes `~/jope-data/h257/cache/val91_jtt1_envelopes/` |
| P201b root | xbabe2 `~/jope-data/h335-cl/refboard/p201b-train-audit-v1/` (copy on xbabe3) |
| panel_v1 set | xbabe2 `~/jope-data/refboard/external/panel_v1/` (copied from `~/jope_plan/elo/`) |
| Predictions (AUDIT boards) | `<root>/preds/<model>/<cid>.f64`: little-endian float64, AUDIT rows in envelope order; sha256 in each score line |
| CPU environment (xbabe2) | system Python + `~/.local` (sklearn, xgboost, lightgbm, catboost, autogluon); `~/jope-data/refboard/venv` (TabPFN, Kumo). Freezes in `envs/` |
| GPU environment (xbabe3, RTX 3090) | `~/jope-data/refboard/venv-gpu` (TabFM) |
| Weights | xbabe2 `~/jope-data/refboard/weights/` (TabPFN v3, 3.5); `~/jope-data/refboard/hf/` (TabFM, Kumo; Kumo pinned to revision v1.0.1) |
| dev91 scoring authorization | `~/jope_plan/ns/receipts/REFBOARD_DEV91_SCORING_AUTH.txt` (0444) |

## 7. Tools

| Tool | Does |
| --- | --- |
| `tools/ladder.py` | Elo + head-to-head from stored scores only; verifies the envelope sha and each prediction file's sha |
| `tools/archive_board.sh` | Create-only S3 archive of a board root or result set; exact-version GET; member verify; `ARCHIVE.json` |
| `tools/import_h257_two_fold.sh` | Builds `dev91-s1-two-fold-v1` from the H257 records (no model runs) |
| `tools/index_s3_single_split.sh` | Indexes the precomputed S3 single-split objects (HEAD only) |
| `tools/import_h335.py` | Imports the 12 H335 entrants' stored values |
| `tools/make_board.py` | Writes an AUDIT board's envelopes + `BOARD.json` (runs in the H335 sandboxes) |
| `tools/score_reference.py`, `tools/run_board.sh`, `tools/sandbox_refboard.sh` | AUDIT harness: one fit per (model, context), resumable, preflight version/weights check, bwrap sandbox |
| `tools/run_cpu_boards.sh`, `tools/run_gpu_boards.sh`, `tools/run_cpu_train_parallel.sh`, `tools/setup_env*.sh` | The exact launches and environment setup used on 2026-11-05 |
| `tools/run_dev91_resume_cpu.sh`, `tools/run_dev91_resume_gpu.sh` | The systemd wrappers of the one owner-authorized completion of the three partial dev91 AUDIT runs (section 1); kept as run |

## 8. Reproducibility notes
- **Isolation.**
  - xbabe2 AUDIT fits ran in `sandbox_refboard.sh` (bwrap: own network and PID namespaces).
  - xbabe3 cannot create unprivileged user namespaces. TabFM there ran with `REFBOARD_ALLOW_NOSANDBOX=1`, Hugging
    Face offline.
- **Kumo is not bit-identical across thread counts:** u 0.920 at 16 threads vs 0.930 at 8 on one context. One more
  reason to store, not re-run.
- **TabFM on CPU is impractical:** more than 10 minutes for one 1,120-row context. On the xbabe3 GPU it takes about
  26 s per context.
- **Licenses:** TabPFN 3.5 and TabFM weights are non-commercial / non-production. These numbers are for internal
  benchmarking only.
