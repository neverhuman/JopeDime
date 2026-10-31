# H106 preregistration — proposal-distribution scorer and W9 memory-source replay (non-neural learner)

Fixed on 2026-10-31, before any H106 prospective outcome. `run_h106.py` implements this design. Its frozen sha256,
recorded in every receipt, is authoritative; any difference from this text is a defect.

## Why

- [H94](../h94_w6_prospective_hedge/README.md) and [H103](../h103_w4_two_plus_two_scored/README.md) showed that
  W1-recipe priors rank admitted pool programs well but **do not beat random4 among the live proposer's new
  programs**. On a fresh H103 stream, prog_top4 − random4 was −0.051. So any W4/W6 learner first needs a scorer trained
  on the proposal distribution.
- W9 ([PROPOSALS.md](../rd_breakthrough/PROPOSALS.md#w9--causal-flywheel-closeout),
  [H91](../h91_w9_closeout_readiness/README.md)) asks whether better guidance acquires better data: data that improve a
  common learner at matched cost. H91 counts 5 memory sources × 2 learner classes. H75 covers only two neural cells,
  and it is parked with no GPU slot.
- H106 runs the **non-neural learner class** of that replay on CPU, using the H94 and H103 reveal logs as memories
  acquired by different guides. The neural cells stay with H75; the GPU queue (H57 → Fused D → W1) is unchanged.

## Design

| Item | Fixed value |
|---|---|
| Memories | Native reveals logged by H94 (`rows.jsonl`, sha256 `18564293…`) and H103 (`rows_w4.jsonl`, sha256 `2e075757…`) on the frozen H67 roster (50 DL-Train contexts, 44 families). DSLs are recovered by replaying each run's deterministic proposer stream; a single unrecovered program stops the build. |
| Memory sources | **UNIFORM** = H103 random4. **PRIOR** = H103 prog_top4 + H94 PRIOR_top4. **ADAPTIVE** = H103 prog_2plus2 + prog_2plus2_swap (its first pair is the prior's top 2, so it overlaps PRIOR by design). **CONTEXT** = H94 CTX_top4 + HEDGE_CTX. **ALL** = the union. Rows are de-duplicated per (context, program). |
| Common learner | Features: `[prog412(p) − prog412(context champion), prog412(p)]`. Target: the within-context utility percentile among that memory's rows, centred at 0, with ties sharing a rank. `HistGradientBoostingRegressor(max_iter 200, lr .05, leaves 15, l2 1, random_state 1060+k)`. One model per W1 outer fold k, trained only on contexts outside fold k. |
| Matched rows | For each fold, UNIFORM, PRIOR, ADAPTIVE and CONTEXT are each cut, in sha256-keyed order, to the smallest of their four row counts. ALL uses every row. |
| Prospective programs | Roster contexts × **fresh seeds 801–803**, 16 new proposals each (stream `h106-stream`, admitted pool excluded) |
| Labels | **All 16 proposals per context-seed are evaluated** by native original-cut training CV, 2,400 fits, so every arm is scored on a fully labelled batch. Picks are fixed from scores before any outcome is read. |
| Arms | `S_ALL`, `S_UNIFORM`, `S_PRIOR`, `S_ADAPTIVE`, `S_CONTEXT` (each scorer's top 4); `prog_top4` (the H94 program-only refit plus the pool mean); `random4` (`h106-random`); `ORACLE4` (the best possible 4; a descriptive ceiling, never a contender) |
| Endpoint | Best utility among an arm's 4 picks (zero history). Seeds are averaged within context, then contexts within family. |
| **Primary A** | `S_ALL − random4`: does a proposal-distribution scorer beat uniform selection on new programs? |
| **Primary B (W9)** | `S_PRIOR − S_UNIFORM`: does guide-acquired memory make a better common learner than uniform memory at matched rows? |
| Gate | Common gate per primary with Bonferroni m = 2 (one-sided .025 LCB > +.001), family median ≥ 0 and ≤ 10% of families losing > .01; a 10,000-draw paired family bootstrap, seed 20261031 |
| Secondary | `S_ADAPTIVE − S_UNIFORM`, `S_CONTEXT − S_UNIFORM`, `S_ALL − prog_top4`, and `prog_top4 − random4` (a replication of H103 on a fresh stream); median 95% CIs; W/T/L at tie 1e-4 |
| Elo | Canonical family-grouped Bradley–Terry, `random4` = 1500, tie 1e-4, prior .01, 400 iterations. No AUC. |
| Completeness | A verdict only on the full roster × arm × seed panel with no shortfalls (as H96). Otherwise INCOMPLETE and no verdict. |
| Budget | 2,400 native fits, deadline 2026-11-01T12:00Z, xbabe2 (x86-64 evaluator), nice 15, load-gated at 85%, 0 GPU |

## Decision

- **A PASS:** a proposal-distribution scorer is the first learner shown to beat random selection on new programs.
  Recommend it as the base for any W4/W6 continuation; a live change needs a separate owner decision.
- **B PASS:** guided memory improves the common non-neural learner at matched rows. This is the W9 claim's
  non-neural half. **B FAIL** with A PASS means the data volume, not the guide, carries the gain.
- **Neither:** the logged memories (about 1,400 rows) are too small or uninformative. Record that, and keep W9's
  causal claim open.

## Disclosure and leakage

- **Adaptive design.** The design follows H94/H103 outcomes on the same roster. Primary A is new, though: no scorer
  of this kind has been evaluated. Every training row is a native outcome that existed before the preregistration.
- **Out of fold.** The prospective proposals use fresh seeds, and each context is scored by the fold model that
  excluded its whole family, together with all of that family's logged rows.
- **Data scope.** DL-Train only. No DL-Val, outer-test, sealed or holdout row is read, and DOPE is not loaded.
