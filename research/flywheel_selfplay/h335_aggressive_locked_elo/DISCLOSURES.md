# H335 disclosures

Everything that departs from a clean reading of the result. PREREG §8 and §9 (A1) hold the pre-outcome items in full.

## Data and cohort
- **dev91 is a reuse cohort, not a clean holdout.**
  - H257 tuned on it, and the legacy 1749.51 Elo comes from it (a best-of-arms number on its scored folds).
  - H335 is this lineage's first read. Its AUDIT-Elo is on a different scale (held-out 30% rows, single locked output,
    anchor H257_REPLICA = 1500). It is never comparable to 1749.51.
- **Read accounting.**
  - The dev91 rows were read exactly once, through the pinned unlock (00:56:16Z, LOCK `4d58dd37…`).
  - Every record was sealed until stats created `FIRST_OUTCOME_READ` (02:24Z).
  - No other sealed / holdout / meta-test / DL-Test path was opened.
  - Guard: 0 DL-Val-path refusals; 0 sealed-path refusals.
- **Entrant provenance.** The 4 LLM Rust regressors are the origin/main blobs. GPT6p1_Sol_Max was refactored on 11-02
  (PREREG §8); whether any entrant's history was tuned on dev91 is disclosed there, not re-audited here.
- **H257_REPLICA is the GP part of H257 only** (A 8 → 3 gens × 5 live → CV1 Top-3). H257's pool-champion branch has no
  DSL and cannot be refit SEARCH → AUDIT. P1 therefore compares against the incumbent GP *recipe*, not the full H257
  submission.

## Selection on TRAIN
- G\* (50) and the stack (CV3@12) were chosen on P201b. The TRAIN contrasts and TRAIN Elo in README §2 are therefore
  mildly optimistic: 5 checkpoints × 2 output kinds.
- dev91 was not used for any choice.

## Execution
- **Guard log baseline.** The production refusal log holds 17 lines, all from deliberate test probes before Phase T:
  16 from `test_h335.py` before its scratch-log fix, and 1 from a Stage V builder test whose script name contained a
  guard token. No new line was added during Phase T, the LOCK, prep, the audit or stats.
- **Seed F short counts.** 7 of 201 TRAIN AGG units and 6 of 182 dev91 AGG units used 17–21 Seed-F evals instead of 22.
  - The count is never above 22, so AGG never received extra evals.
  - The AGG vs OLD_MATCHED live-eval match held at every unit (max |Δ| = 2, 0 violations of the ±2 tolerance).
  - The cause (duplicate candidate shas or a short cross-family bank) was not investigated further.
- **Eval CPU timeouts** (60 s RLIMIT_CPU, failed evals inside the search, never invalid outputs): 36 on TRAIN AGG and
  45 on dev91. 0 invalid outputs and 0 failed units or segments in either phase.
- **Prep cores.** Prep / stats moved from xbabe2 cores 54-63,118-127 to 38-53,102-117 because `h344-run` held 54-63 at
  00:49Z (A1). This affects wall time only.
- **Stage V agent test runs** were pinned to xbabe2 cores 44-45, inside the Phase T xbabe2 set, for about 2.5 min each
  at nice 19 during Phase T. Worker limits are CPU-time, so results are unaffected.
- **Stray file.** `/home/ubuntu/.cache_never` on xbabe0 (366 B, a list of 20 staged code filenames) came from a reviewer's
  stray redirect. It is left in place (no-delete rule).
- **Exit statuses.** `systemctl show` on the finished transient units prints defaults (`LoadState=not-found`), so it is
  not used as proof of exit. Completion evidence:
  - Phase T: the 16 shard status files and 402/402 unit records.
  - dev91: the audit unit's own summary line (`audit: done`, 0 failed shards, no integrity stop), 12/12 `AUDIT_DONE`
    records and 364/364 unit records.
- **Host load.** xbabe0 peaked at a load of about 100 / 128 (78%) during the dev91 audit, under the 85% cap; the engine
  recorded 0 pauses.
- **Timing.** Phase T ended at 00:52Z (stop 01:30Z). The LOCK was written at 00:55Z (deadline 02:00Z). The dev91 audit
  ran 00:58Z → 02:24Z (stop 07:30Z).

## Statistics
- The primary is a clipped (±0.5) family contrast. The unclipped headline gains are larger than the clipped T values
  because a few families gain a lot. Both are reported.
- **Raw adoption gate vs H257_REPLICA fails only the ≤10% harm rule:** 2 of 19 families lose more than 0.01, which is
  10.5% (one family over the line). Against OLD_MATCHED the raw gate passes every check.
- Elo intervals are a family bootstrap (descriptive). Inference rests on the exact sign-flip tests.
