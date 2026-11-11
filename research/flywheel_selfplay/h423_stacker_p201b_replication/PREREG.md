# H423 PREREG — the frozen H422 combiner on the stored P201b matrices (replication), and deploy-path code

**Status.** Published on `origin/v1/next` before any stacker value on P201b is computed. What was looked at is in
§8. Claim: `~/jope_plan/ns/receipts/H423_claim.txt` (2026-11-11T07:17Z, `cl_apex`).

**Authority.** Owner order of 2026-11-10 (APEX campaign). Relayed by ENG:watcher on 2026-11-11 07:18Z (12:18 AM
MST): run follow-up 2 of H422 as the next free H number with a PREREG that fixes one primary arm and the one-sided
0.025 rule; confirm P201b has no dev91-audit-v1 rows and stop if it does; do follow-up 3 (both levers in the
offline deploy path, code only, behind a flag that defaults off, with tests); follow-up 1 is held.

**Closed.** dev91-audit-v1, DL-Val, DL-Holdout and sealed data are never read. `CONFIRM` and `DEV` are not read by
this wave. No worker evaluation is run. Nothing is fitted.

## 1. Question
H422's combiner alone gained +0.0197 u in the TRAIN-side cross-fit (291 design families) and +0.0045 u on `DEV` (115
families, not significant). Which is closer to the truth? P201b offers 201 more TRAIN-side contexts with stored
matrices, at no worker cost.

## 2. Data
- **P201b** (`p201b-train-audit-v1`): the second seeded TRAIN context of every TRAIN family with at least two
  contexts; 201 contexts in 201 DL-Train families. Stored by H416's confirm run (`H416_CONFIRM`, tag `P201B`, on
  xbabe0): per context the 64 admitted candidates' out-of-fold SEARCH predictions, AUDIT predictions (fold-model bag
  and refit), targets, and the record of the stored `h416.SXB` output with its weights and AUDIT value.
- **dev91 check (done before this PREREG, identifiers only).** The published board definitions were compared:
  P201b shares **0 context ids, 0 envelope hashes and 0 family ids** with `dev91-audit-v1` and with
  `dev91-s1-two-fold-v1` (`results/dev91_disjoint.json`). The apply step repeats this comparison and refuses to run
  on any overlap.
  - Limit: this compares identifiers. A comparison of table rows would need the dev91 tables, which stay closed.
- **Not independent of H422's design.** 185 of the 201 P201b families are also P201e design families of H422
  (different contexts; no context id is shared). The combiner's form was chosen on those families. 16 families are
  outside the design.

## 3. Method (nothing is fitted)
- **`B`** = the stored `h416.SXB` value of each P201b record; not recomputed.
- **`S`** = H422's frozen model (`FROZEN.json`, sha256 `a43739b6469e2964…`, commit `3b30a9051e`): the winsorised
  combiner with constant k = 6 on the same stored candidates and weights, computed by H422's published code
  (`sel_h422.context_stats`). No AUDIT target enters a prediction (checked again by zeroing the targets).
- **Wider admission cannot be replicated here.** P201b has one stored variant per context (the H416 recipe), so
  there is no second stored candidate set to pool. The arm "wider admission + stacker" is **not run**; this is
  recorded as a limit of the replication, not as a result.

## 4. Test and verdicts (one primary arm)
`S` vs `B` on the 201 P201b contexts, families as groups, value = AUDIT u, `h335_stats.primary_test`.

| Verdict | Condition |
| --- | --- |
| `REP_CONFIRMED` | the test passes: clipped family-mean gain above 0, **one-sided sign-flip p < 0.025**, leave-one-family-out minimum above 0, median not negative |
| `REP_HARM` | clipped family-mean gain below 0 with mirror sign-flip p < 0.025 |
| `REP_NULL` | anything else |

- Reported, no verdict: whether the clipped gain reaches +0.010 u; ΔElo with its 95% family-bootstrap interval
  (10,000 draws, seed 423150); the same test on the 16 families outside H422's design; how many contexts move by
  more than 0.05 u.
- **What a verdict changes.** Nothing in the champion recipe: H422's `DEV` verdict stands, and P201b shares
  families with the design. The result says how much weight the combiner-alone effect deserves when the held
  follow-up 1 is decided.

## 5. Deploy-path code (follow-up 3; not scored)
`deploy_h423.py`, engine-free, both levers off unless an environment variable is exactly `1`:
- `JOPE_H423_WINSOR_DESIGNS`: `design_pred` accepts `<META>.mixw<k>` (for example `BCAR.mixw6`); every stored
  design name goes to `meta_h416.design_pred` unchanged, with or without the flag.
- `JOPE_H423_POOLED_ADMISSION`: `hf_candidates(pool, hf, sub_pools)` returns the ordered union of the stored
  admission rule over the given sub-pools; without the flag, or without sub-pools, it returns the stored rule's
  result.
- Tests (`test_h423.py`, `test_admission_h423.py`; 12 tests; two stored `FIT-SEL` contexts and synthetic data):
  flags default off; new names refused when off; stored designs byte-equal and the stored record value replayed
  to 1e-9; `mixw<k>` equals H422's offline code for every finite k of its grid; a huge k equals the stored design;
  fallbacks equal the stored `mix`; the pooled rule equals `deploy_h415.hf_candidates` when off and is the ordered
  union when on.
- **Limits:** no stored deploy unit calls these functions; wiring them into a unit needs the fourth search run and
  a scored smoke, which this wave does not do.

## 6. Integrity gates (`gates_h423.sh`, after the apply step, before the readout)
1. The input copy matches its manifest and matched the source host at the pull.
2. The cache holds exactly the 201 P201b contexts; no overlap with a dev91 board definition.
3. Replay: recomputed stack weights equal the stored ones (difference 0) and the recomputed baseline value is
   within 1e-9 u of the record on all 201; k = ∞ reproduces it within 1e-6 u.
4. The model applied is H422's published `FROZEN.json`.
5. This PREREG's commit is older than the first P201b cache file; the staged PREREG equals the published one.
6. No prediction changes when the AUDIT targets are set to zero.
7. Published stage, code hash verified.
8. The 12 deploy-code tests pass from the published stage.
9. Budget kept.

## 7. Budget
xbabe2 CPUs only (cpus 16–63), no GPU, no worker evaluation; cap 6 core-hours (21,600 CPU-seconds), expected
under 0.2. The stored files were copied read-only from xbabe0 (605 files, 168 MB). The training runs are not
touched. Checkpointing: one create-once cache file per context; every other step is under a minute.

## 8. What was looked at before this PREREG
- The directory listing of `H416_CONFIRM` on xbabe0, its copy and checksum comparison, the config, and the format
  of **one** stored record, which printed that record's stored baseline value (0.890). No other value was read.
- Identifier comparisons: P201b against the dev91 board definitions (§2) and against P201e's families.
- **No stacker value on P201b has been computed.**
- The deploy-code tests ran on a development stage (12 of 12 pass); they read two `FIT-SEL` contexts.

## 9. Weaknesses
- Family overlap with the design (§2): a pass is weaker evidence than a pass on fresh families.
- One arm only; the arm that looked best in H422 (wider admission + stacker) is not testable on P201b.
- P201b is the campaign's TRAIN board; it has been read by H416 (as the test of `h416.SXB` itself) and by the
  reference models, never for a combiner.

## 10. Files
`rep_h423.py` (`apply`, `readout`), `gates_h423.sh`, `stage_h423.sh`, `deploy_h423.py`, `test_h423.py`,
`test_admission_h423.py`, `run_tests_h423.sh`. Reused at published bytes: `cache_h422.py`, `sel_h422.py`.
