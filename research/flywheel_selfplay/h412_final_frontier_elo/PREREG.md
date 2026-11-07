# H412 PREREG: the FINAL frontier Elo study on dev91-audit-v1

**Status:** registered before the LOCK, the unlock and any dev91 read by H412.
**Owner:** `cl_fwriter` (Claude Opus 5.5).
**Request:** Jepson, 2026-11-06. "produce ELO scores on our strongest model and compare to ALL of our frontiers we
have... we just added grok4p7 etc... we want all of them. Please work to get the FINAL study."

## 1. What is compared
- **Our strongest model:** the stored `h364.XSTACK_2L`. This was the owner's choice (AskUserQuestion, 2026-11-06). It
  is the best JopeDime entrant on the board (AUDIT-Elo 1726.8, scored once by H369). Nothing of ours is rerun.
  - `h335.H335_AGG` is shown as a second row.
- **Every frontier LLM regressor:**
  - the four stored H335 rows: `h335.Opus5p5_Extra`, `h335.Gemini4_argon_high`, `h335.GPT6p1_Sol_Max` and
    `h335.GPT6_Astra_Max`;
  - plus **`grok4p7_xhigh`**, which is new.
  - A rescan of `origin/main`, `origin/v1/next` and the open PR branches on 2026-11-07 found no other frontier
    submission.
- **Every other board entrant:** the foundation models (Kumo, TabFM, TabPFN), AutoGluon, the classic learners and the
  other JopeDime rows. All are stored rows.

## 2. The one new score: `h412.grok4p7_xhigh`
- **Source:** `crates/jope-automl/src/grok4p7_xhigh.rs`, sha256 `8e648cef93186aca9d10868ed1ed2a62407950169082aa1f4fab87af379664f8`,
  at commit `1177c4f9c1` (`feat/grok4p7-xhigh-automl-elo`, PR #2154, owned by the Grok lane). It is used exactly as
  frozen.
  - The authorship record says `fresh_session: false`. It is noted here and does not affect scoring.
- **Harness:** `frontier_audit_h412` = the H335 frontier AUDIT harness (`frontier_audit_h335.rs`, sha `a2ecafd1…`)
  with a fifth extern crate, entrant name and match arm. Everything else is byte-for-byte H335:
  - rustc 1.97.1, rlib per entrant, fat LTO, `-C opt-level=3 -C codegen-units=1 -C panic=unwind`;
  - seed 42, one fit per context;
  - fit on SEARCH (fold 0) → predict AUDIT (fold 1);
  - u = −ln(RMSE_audit / RMSE of mean(y_SEARCH));
  - 300 s CPU / 900 s wall per context; a failure counts as u = 0.
- **Pre-LOCK checks (done on TRAIN and synthetic envelopes only; no dev91 byte read):**
  1. Two builds give the same binary sha `085c1acb…` and identical rlibs.
  2. On the 201 TRAIN P201b board envelopes, the four H335 entrants reproduce all 804 stored records
     (`frontier/T/frontier_results.jsonl`). Status, `audit_preds_sha256`, utility, RMSE and base RMSE are identical. So
     the stored dev91 rows of the four stay comparable with the new score.
  3. Old and new binaries give identical output on the 40 H335 synthetic pairs, including the degenerate envelopes.
     An `nq != 0` envelope is refused for grok too.
  4. grok completes 201/201 TRAIN envelopes and is deterministic: two runs give identical predictions.
- **dev91 run:**
  - **Gates:** LOCK `stage_v/LOCK.json` (code, binary, rlibs, sources; `BOARD.json` in inputs) with create-once receipt
    `ns/receipts/H412_LOCK_sha256.txt`; one unlock.
  - **Run:** the 91 board envelopes, sha-checked against `BOARD.json`, on xbabe2 with 16 jobs.
  - **Rows:** create-once `scores/h412.grok4p7_xhigh.jsonl`.
  - **Board rules:** one score per id; failures are final; rule 4 (a locked candidate, a create-once receipt, and an
    exposure-log entry).

## 3. Readout (fixed now)
- **Ladder:**
  - `tools/ladder.py` on the board root (anchor `h335.H257_REPLICA` = 1500): the family-grouped MM Bradley–Terry
    AUDIT-Elo with a 10,000-draw family-bootstrap CI.
  - The committed `LADDER.md` is regenerated with the new row. The other entrants' stored scores are byte-unchanged.
- **Head-to-head for `h364.XSTACK_2L`** vs each of the 5 frontier LLMs and every other entrant:
  - the family-equal mean gain in u and the % RMSE cut;
  - families won / tied / lost;
  - the H335 primary test (clip ±0.5, one-sided sign-flip, LOFO min, pass = p < 0.025 and LOFO > 0).
  - The same table for grok4p7_xhigh as focus.
- **Frontier-only view:** the dev91 order of the 5 LLMs side by side with AutoML Elo v1 / v2 (TRAIN panel, 64
  families). That panel is shown for rank agreement only; it is a different panel and not comparable (PROTOCOL §4).
  Our GP is never placed on it, because the GP was selected on TRAIN.
- **Board limits:** dev91 is a reuse cohort (H257 tuned on it), and the 5 H381 / H410 contexts are adaptively exposed.
  The study reports the board as stored; nothing is chosen on dev91.

## 4. Not done
- No new candidate of ours; this was the owner's choice. The H410 TRAIN lead (pooled runs + library, +8.1% RMSE on 120
  TRAIN families) is listed as the next candidate and is not scored here.
- No reference model, frontier entrant or GP entrant is refit, rerun or rescored.
- Sealed / holdout / DL-Test stay closed. DOPE `c51a56d0` is not touched.
