# H418 analyze result: INCOMPLETE (harness gate `ge_replay_335102`); no SXB value read

- **When:** `run_h418.sh analyze`, 2026-11-10 21:49:43Z–21:55:15Z, on xbabe2, from `pulled/` (pull 21:49Z: 1,107 files
  sha-compared with xbabe0). Exit 3.
- **Output:** `results/H418_DEV91.json` (status `INCOMPLETE`, gate names + harness numbers only) and
  `results/H418_DEV91.txt`. `results/h416.SXB.jsonl` was **not** written. No SXB value was computed into any output,
  printed or read.
- **Run facts:**
  - D1: 91/91 units, 0 errors, 13,301.7 s.
  - D2: 91/91 units, 0 errors, 13,756.0 s.
  - GE: 182/182, 4,364.6 s. GL: 182/182, 10,928.8 s.
  - xbabe0 booted at 12:05Z, before the run, so no step was resumed.

## Gates
Every gate passed except `ge_replay_335102`:

| Gate | Result |
| --- | --- |
| Board | 33 files match the manifest; 31 complete entrants; ladder reproduced to 0.0 |
| Contexts, LOCK, configs | pass |
| AUDIT envelope | 182/182 |
| Records | 182 confirm records; 0 bad; 0 with no usable H |
| GS replays (both seeds) | 91/91 same members; max delta 0.0 vs stored and vs board; 0 amendment-1 exemptions |
| GL replays (both seeds) | 91/91 same members; max delta 0.0 |
| GE replay, seed 335101 | 91/91 same members; max delta 0.0 |
| **GE replay, seed 335102** | **90/91 same members; max delta 3.43e-5 (tolerance 1e-9): FAIL** |
| Re-derivations | SXB offline max 2.7e-15; KCAR100.bag = KFB 0.0; reference predictions max 8.9e-16 (tolerance 1e-6) |

## Diagnosis (harness fields only: program ids, memo line positions, status, wall clock; no utility value printed)
1. **The context:** `43022ae6a8bc83e9e25ea4a2882389152856226f5b2bdd4668db07f2f510ae91`, seed 335102.
   - In D2's replay of GE, the stock `AGG.g50.CV3@12` members differ from the GE arm record in one slot.
     - Deploy: `35f192f28974…` is member 11 and `5ea11843…` is member 12.
     - Arm: `5ea11843…` is member 11 and `261e8073…` is member 12.
   - The other 10 members are identical and in the same order.
2. **`35f192f2` was never evaluated by the GE arm.**
   - It has 0 lines in the GE memo for this context (633 lines) and is not in the GE `result_shas` of either seed.
   - It has 1 SEARCH line in the GL memo (GL line 129).
   - The D2 memo's first 1,916 lines are byte-identical to `cat GS(643) GE(633) GL(640)`.
     - `35f192f2`'s SEARCH entry there is D2 line 1405, which is GL's line 129: a memo hit.
     - Its AUDIT entry is D2 line 1917, the first line the deploy appended, from the GE.stock AUDIT evaluation.
   - So D2's replay of the GE search went live with a program that the GE arm did not pick.
3. **Not a memo conflict, timeout, retry or JSON round-trip.**
   - Memo-conflict scan: 448 shared keys, 0 GS or arm conflicts.
   - The context's GE and GL memos are all `completed`, and GE `retries.jsonl` has 0 entries for it.
   - The memo round trip is exact: `_result_to_json` writes Python floats and float64 lists, and `_result_from_json`
     reads `np.asarray(float64)`.
   - The pool sizes are consistent with a one-program swap in GE's population:
     - D2: P2 = 505, P3 = 754. These equal |GS∪GE| and |GS∪GE∪GL| from the stored seed-335102 records.
     - So the GE replay's population differs from the arm's by a swap in which both programs are in GL.
4. **Likely mechanism: the screener score cache is shared across seeds in the arm process.** This is not a proof; no
   code was run.
   - `screener_h317.Screener.score` caches each candidate's score per (cid, views, program sha) when the model is
     candidate-isolated (`cache_scores`). The model runs float32 torch on CPU, and the batch that first scores a
     program sets its cached value; CPU float32 GEMM output can depend on the batch shape at rounding level.
   - In the GE arm, both seeds of a context run in one process and share that cache. For this context, seed 335101
     ran first (45 s, record written 15:48:45). Seed 335102 started after it (271 s, written 15:53:16), so it read
     scores cached from 335101's batches.
   - D2 is one process per seed (PREREG §2), so seed 335102 started with an empty cache and scored every program in
     its own batches.
   - A rounding-level difference in a screened score is enough to flip a near-tie in `pick_exploit_ucb`. One flip in
     generation g changes every later pick, and here the replay's new pick was a program GL had evaluated.
   - Consistent with this:
     - only 1 of 364 arm replays diverged (it needs a near-tie);
     - seed 335101, which ran first and so used no other seed's cache entries for this context, replays exactly;
     - H416 never met this case, because P201b is single-seed.
5. **Status of the rest:** every other (context, seed, source) replay is verified identical by the gate (members
   equal and value delta 0.0), so no other cell is affected.

## What the PREREG allows (§4 "On a failure", §4 host-reboot block)
- "Diagnosis reads harness fields only. A repair of the readout is a published amendment, made before any SXB value
  is read. No completed unit is rerun."
- "If a gate cannot be met, the result is `INCOMPLETE` and is reported; there is no second score."
- This is not a readout defect: the gate correctly found a real replay divergence in one deploy cell. Any amendment
  changes the protocol after a harness failure, though still before any value is read. It is the owner's decision.
  Nothing was amended, re-analyzed or published.

## Options for the owner (nothing done)
- **A. Close H418 as INCOMPLETE.** No dev91 value of `h416.SXB` is recorded; there is no second score.
- **B. Amendment 2, by analogy with H416 amendment 1 (recommended).** A (context, seed) whose GE or GL replay fails
  scores X0b (the stored `h335.H335_AGG` per-seed value). That is 1 of 182 cells: (`43022ae6`, 335102).
  - The deploy's own value there is reported descriptively only, as for amendment-1 exemptions.
  - The readout change is published on v1/next before analyze runs, with a new create-once output path; the
    INCOMPLETE output stays. Synthetic selftests cover the new case.
  - Then there is exactly one analyze. Nothing is rerun.
- **C. Amendment 2'.** Accept the deploy value for that cell (the recipe as executed). This is weaker: GE in that pool
  is not the stored GE arm.
