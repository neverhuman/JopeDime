# H414 PREREG: the frontier regressor `kimik3p1_max` on every stored board

**Status:** registered before the LOCK, the unlock and any DL-Val91 read by H414.
**Owner:** `cl_fwriter` (Claude Opus 5.5).
**Request:** Jepson, 2026-11-07. "We have a new model we added: xbabe2:~/JopeDime/kimik3p1_max.rs please add it to
the right location in our code base, and please score it on ALL data and include the elo ranking with this new item.
Don't rush this, double check it and make sure you are really doing it right, in the right location, etc.."

## 1. The entrant
- **Source:** the owner's file `~/JopeDime/kimik3p1_max.rs`: 47,156 bytes, mtime 2026-11-07T16:43:30Z, sha256
  `0c211694ec66605b994595fd8f4cf2b8894df6f156caa3b80f1fcc75f5d6ceb3`. It is used byte-for-byte and never edited.
- **Code location:** `crates/jope-automl/src/kimik3p1_max.rs`, registered in `jope_automl::REGRESSORS` next to
  `GPT6p1_Sol_Max` and `Gemini4_argon_high`, with `crates/jope-automl/submissions/kimik3p1_max/`. This goes through a
  separate pull request to `main`.
- **Authorship:** the file came without an authoring manifest, so its authoring conditions are recorded as unknown.
- **Interface:** the same challenge API as the other frontier regressors:
  - `Regressor::new(seed)`, `fit(train, n_rows, n_cols)` with the target in the last column, then
    `predict(features, n_rows, n_features)`;
  - std only, no `unsafe`.

## 2. "ALL data": the four panels
| Panel | Split | Id | Status |
| --- | --- | --- | --- |
| AutoML Elo v3 (64 TRAIN families × 3 folds × 3 seeds, 576 cells) | its own cut | `kimik3p1_max` | done before this PREREG: TRAIN only, in the code PR (`assets/automl-elo/v3/`) |
| `p201b-train-audit-v1` (201 TRAIN contexts) | AUDIT | `h414.kimik3p1_max` | the pre-LOCK TRAIN run below, boarded after the LOCK |
| `dev91-audit-v1` (DL-Val91, 91 contexts / 19 families) | AUDIT | `h414.kimik3p1_max` | **new dev91 read**, once, under the LOCK |
| `dev91-s1-two-fold-v1` (DL-Val91) | H257 two-fold | `h414.kimik3p1_max` | **new dev91 read**, once, under the same LOCK |

- No other split exists for a frontier LLM regressor.
- The dev91 S3 single-split column holds precomputed foundation-model encodings only, and `panel_v1` has no frontier
  entrant. Kimi is not added to either.
- **Optional, same commit:** `h412.grok4p7_xhigh` on `p201b-train-audit-v1`. These are grok's existing 201 TRAIN
  records from the H412 pre-LOCK run (binary `085c1acb…`, the binary that produced its dev91 row; two runs identical).
  They are imported, not rerun. With them, the TRAIN board carries all 6 frontier entrants.

## 3. Harnesses (both frozen before the LOCK)
### 3a. AUDIT boards: `frontier_audit_h414`
- **What it is:** the H412 harness (`frontier_audit_h412.rs`) with a sixth extern crate, entrant name and match arm.
  Nothing else differs: same seed (42), parser, fold assembly and arithmetic.
- **Run:** fit on SEARCH (fold 0) → predict AUDIT (fold 1); u = −ln(RMSE_audit / RMSE of mean(y_SEARCH)).
- **Limits:** 300 s CPU / 900 s wall per context; a failure or timeout counts as u = 0.
- **Build:** rustc 1.97.1, one rlib per entrant, fat LTO, `-C opt-level=3 -C codegen-units=1 -C panic=unwind`.
  - Binary sha256 `c34025df88cc3ec681ac186e5a0ad911b0caaa84a41f0ee68c2f31476a627162`.
  - Two builds are identical.
- **Wrapper:** `frontier_audit_h414.py` = the H412 wrapper with `kimik3p1_max` appended to `ENTRANTS`.

### 3b. Two-fold board: `twofold_eval_h414`
- **What it is:** the H257 two-fold evaluator source (`eval_frontier_dl_val91.rs`, H257 main) with a fifth extern
  crate and match arm, plus an optional comma-separated entrant filter as argv[5].
- **Name:** the wave copy is named `twofold_eval_h414.rs`, because the H317 guard refuses `dl_val` in a path before
  Stage V. Its bytes equal the build tree's `harness/eval_frontier_dl_val91_h414.rs`.
- **Unchanged:** the two-fold split, the CV gate, `s1_utility_zero_comp`, the failure value and the output records.
  - Fit on one fold, predict the other, both directions.
  - u = −(0.9·logratio + 0.2·sd_logratio) when the gate passes (exp(logratio) ≤ 0.90 and both folds beat the
    training-fold mean); otherwise −2.0, also on a fit failure.
- **Build:** same flags as 3a. Binary sha256 `97bf86530bef4c61c837883721d570e78cdd7aabecb6713b740cd02cad6cd871`.
  - Two builds and their rlibs are identical.
  - The binary is installed at `/home/ubuntu/jope-data/h414/bin/twofold_eval_h414` (same bytes).
- **Origin of the stored H257 binary:** its build flags are unrecorded, because its h256 source copy no longer exists.
  Equivalence is therefore shown empirically (check 3 below).

### Pre-LOCK checks (done; TRAIN and synthetic only, no DL-Val91 byte read)
1. **Builds:** both binaries reproduce bit-for-bit from a clean rebuild.
2. **AUDIT parity:** on the 201 TRAIN P201b envelopes, the four H335 frontier entrants in `frontier_audit_h414`
   reproduce all 804 stored H335 TRAIN records (`h335-cl/frontier/T/frontier_results.jsonl`). Status,
   `audit_preds_sha256`, utility, RMSE and base RMSE are all identical.
   - grok4p7_xhigh reproduces its 201 H412 TRAIN records exactly.
   - So the stored dev91 rows of all five earlier frontier entrants stay comparable with the new one.
3. **Two-fold parity:** on the 16 TRAIN two-fold envelopes (`h257/cache/train_curr16_jtt1_envelopes`), the four H257
   frontier entrants in `twofold_eval_h414` give records identical to the stored H257 binary (`bd78de47…`) in every
   field except `wall_ms`: 64 of 64.
4. **Kimi on the AUDIT harness:** 201/201 TRAIN envelopes complete, and a second run is identical (predictions and
   utilities).
5. **Kimi on the two-fold harness:** 16/16 TRAIN envelopes complete, all gates pass, and a second run is identical.

## 4. Runs after the LOCK
### Gates
- **LOCK:** `DATA/stage_v/LOCK.json`, with `DATA` = `/home/ubuntu/jope-data/h414/d91`.
  - `code_sha256`: the staged wave files, both binaries, every rlib and entrant source, `h335_core` and the H317 pins.
  - `inputs_sha256`: the dev91 and P201b `BOARD.json` files and the two TRAIN parity record files.
  - `twofold_sha256`: the two-fold `BOARD.json` and the two-fold build tree. These are checked by bash, because the
    H317 guard refuses `~/jope-data/h257/` at every stage.
- **Receipt:** create-once `ns/receipts/H414_LOCK_sha256.txt`.
- **Unlock:** one only (`STAGE_V_STARTED`).
- **Scope:** nothing runs on DL-Val91 before the receipt exists and the unlock marker names this LOCK.

### The three runs
1. **dev91 AUDIT:** `kimik3p1_max` on the 91 board envelopes, each sha-checked against `BOARD.json`. xbabe2, 16 jobs.
   Writes create-once `scores/h414.kimik3p1_max.jsonl`.
2. **dev91 two-fold:** `kimik3p1_max` on the 91 `val91_jtt1_envelopes`, sha-checked against the two-fold `BOARD.json`,
   one run (8 threads). A context with no record counts as failed (u = −2.0).
   Writes create-once `scores/h414.kimik3p1_max.jsonl` on that board.
3. **P201b rows:** from the pre-LOCK TRAIN run records (check 4, same binary sha), bound to the LOCK sha.

### Board rules
One score per id; failures are final; no stored entrant is rerun. Rule 4 applies: a locked candidate, a create-once
receipt and an exposure-log entry.

## 5. Readout (fixed now)
- **Ladders:** `tools/ladder.py` on each board root, with its own anchor: `h335.H257_REPLICA` on the AUDIT boards and
  `h257.Opus5p5_Extra` on two-fold.
  - Each is a family-grouped MM Bradley–Terry Elo with a 10,000-draw family-bootstrap CI.
  - Each committed `LADDER.md` is regenerated, and must regenerate identically from the git copy. The other entrants'
    stored score files stay byte-unchanged.
- **Head-to-head for `h414.kimik3p1_max`** (`--focus`) on each board:
  - the family-equal mean gain in u and the % RMSE change;
  - families won / tied / lost;
  - vs each frontier entrant, our best model and every reference model.
- **Frontier-only view:** the order of the 6 LLM regressors on each panel, side by side.
  - AutoML v3 and P201b are TRAIN panels; the dev91 boards are the test-side read.
  - Panels are not comparable with each other (PROTOCOL §4), so they are compared for rank agreement only.
  - grok has no two-fold score, so that column holds 5 entrants.
- **Board limits:** dev91 is a reuse cohort (H257 tuned on it), and the 5 H381 / H410 contexts are adaptively exposed.
  Nothing is chosen on dev91; the boards are reported as stored.

## 6. Not done
- No model of ours is rerun, refit or rescored, and neither is any reference model or earlier frontier entrant.
- grok4p7_xhigh gets no two-fold score here; that would be a new DL-Val91 read for an entrant the owner did not name.
- Sealed / holdout / DL-Test stay closed. DOPE `c51a56d0` is not touched. The Grok lane (PR #2154) is not touched.
