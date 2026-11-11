# H429 PREREG — the stack column for wide tables, rebuilt with the readout H428 confirmed (a DEV development read)

**Status.** Published on `origin/v1/next` before any output of this wave exists. Launch waits for the H424 hour-24
gates and readout (ordered). Claim: `~/jope_plan/ns/receipts/H429_claim.txt` (2026-11-11T12:02Z).

**Authority.** Relayed by ENG:watcher on 2026-11-11 (stamped 5:04 AM MST): run the stack-column test written in
H428's README with the passing variant as the readout for tables wider than 30 features; H424's stack-column result
as baseline; cluster-level units per H426; a pre-declared primary with success and failure criteria; CPU only, at
most 16 single-thread processes on free xbabe2 CPUs, at most 4 hours; no GPU, no `CONFIRM`, never dev91; training
runs left alone; **a `DEV` development read, not a champion read.**

**Closed.** dev91-audit-v1, DL-Val, DL-Holdout and sealed data are never read. `CONFIRM` is not staged, opened or
read. No `DEV` table is reopened: the inputs are H424's stored inference bank and stored columns.

## 1. Question
H424's column earned nothing on the 35 `DEV` tables with more than 30 features (H427). H428 found a readout that is
better there on the `FIT-SEL` monitor (+0.023 u). Does rebuilding the column with that readout make the stack better
on those 35 tables?

## 2. What is held fixed
The checkpoint (`ICL_M_r3/ckpt_h020.pt`, sha256 `e3c8fd9972ce…`, the one H424 and H428 used), the stored stack
matrices, the weight builder and the combiner, H424's bank, and H424's stored column for the 80 tables with at most
30 features (copied unchanged; the stack step must reproduce H424's values on them exactly).

## 3. The one change (35 wide tables)
In every pass that builds the column (five fold-context passes and the all-SEARCH pass, as in H424):
- the ridge is fitted on all features with H424's seed of that pass, so the ridge part is H424's;
- the model predicts the ridge's residual from each of 8 seeded random subsets of 30 features, and the 8
  predictions are averaged and added to the ridge (H428's variant T1).
- Subsets: `numpy.default_rng(sha256("H429-subset" ‖ NUL ‖ context ‖ NUL ‖ j))`; the same 8 subsets in every pass of
  a context. Not tuned.

## 4. Test, success and failure
Arms per context (AUDIT u): `B` = stored stack; `I` = H424's stored arm (stack + H424's column); `IT` = stack + the
rebuilt column.
- **Primary: `IT` vs `I` on the 35 wide tables, clusters as units** (H426's `feature` clustering from the frozen H425
  screen; 30 units), `h335_stats.primary_test`: clipped unit-mean gain above 0, one-sided sign-flip p < 0.025,
  leave-one-unit-out minimum above 0, median not negative.

| Verdict | Condition | Consequence |
| --- | --- | --- |
| `WIDE_CONFIRMED` | the primary passes | the rebuilt readout becomes the development default for wide tables in the column recipe |
| `WIDE_HARM` | clipped gain below 0 with mirror sign-flip p < 0.025 | recorded; H424's readout stays for wide tables |
| `WIDE_NULL` | anything else | recorded as a negative result; wide tables wait for the training-mix prompt (held) |

- Reported, no verdict: the same statistic with single contexts as units; `IT` vs `B` and `I` vs `B` on the wide
  tables; `IT` vs `B` on all 115 (the H424 comparison with the rebuilt column); the column's stack weight, its
  AUDIT value alone and its out-of-fold distance to the best stored candidate, before and after; 31–100 against
  more than 100 features.
- A wide table whose column is not rebuilt before the budget ends keeps H424's column (`IT` = `I`, gain 0) and is
  counted.
- **No champion consequence.** `DEV` has been read many times; this read steers development only.

## 5. Budget
- CPU inference of 35 tables: 6 passes × 8 subsets = 48 model passes of 30 features per table, about 2,700
  seconds each on one thread; **about 26 core-hours; about 2.3 hours with 16 single-thread processes** (three
  tables per process at most). xbabe2 cpus 40–55, `CUDA_VISIBLE_DEVICES` empty.
- Hard cap 4 hours from start to the end of the stack step: no new table after 3 h 10 min, stopped at 3 h 40 min.
- Checkpointing: one create-once file per table; a restart continues.

## 6. Gates (`gates_h429.sh`, after the stack step, before the readout)
This PREREG is older than the first rebuilt column and no readout exists; the first rebuilt column is younger than
the H424 hour-24 readout; pinned checkpoint and bank; CPU only, at most 16 threads, no CUDA device visible; the
4-hour cap; the stack step's alignment and replay checks on all 115 tables; 35 wide tables rebuilt or counted as
fallback, and the 80 narrow column files byte-identical to H424's; the sandbox view still holds no `CONFIRM` table;
published stage.

## 7. Expected, stated in advance
**A small or null effect.** On these 35 tables the column's out-of-fold fit is 0.7 to 1.7 u behind the best stored
candidate; H428's readout is about 0.02 to 0.04 u better. The column should get a little more weight in a few tables
and the stack should barely move. A clear gain would be a surprise.

## 8. What was looked at before this PREREG
H424's stored values on the wide tables (H427: weight 0.03, gain 0, 11 wins and 11 losses); H428's result on
`FIT-SEL`; a code smoke on a synthetic bank of random numbers; a self-test of the readout that feeds it H424's
values twice (all gains 0). **No rebuilt column exists for any `DEV` table.**

## 9. Weaknesses
- 35 tables in 30 clusters: only a clear effect can pass.
- `DEV` is a reused development cohort here; H424's values on these tables were already studied in H427.
- One checkpoint; the hour-24 read (H424 second read) is not combined with this change.

## 10. Files
`cols_h429.py` (CPU inference), `readout_h429.py`, `gates_h429.sh`, `stage_h429.sh`. Reused at published bytes:
`stack_h424.py`, `cache_h422.py`, `restate_h426.py`, and H419's `icl_model.py`, `icl_resid.py`, `icl_diag.py`,
`icl_inject.py`.
