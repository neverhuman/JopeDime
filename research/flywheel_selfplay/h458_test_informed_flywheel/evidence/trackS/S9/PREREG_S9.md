# H458 Track S, S9: staged residual GP ("GP residual boosting"): pre-registration

- **Written:** 2026-11-15, 01:35Z to 01:40Z, by cl_mtc (Claude, xbabe2, session 7afc9e1c), owner of S9 scoring.
- **Owner ask:** Jepson typed it in session 7afc9e1c at about 00:50Z, and the same ask in cl_h458's session at about
  00:48Z.
  - The idea: run GP for 10 or 100 generations, evolve on the residual for 10 to 200 more, and repeat, building an
    additive ensemble of 5 to 10 programs.
  - "MUCH more advanced" than the plain chain, tested for whether the value generalizes to Elo.
  - Only on medium, high and ultra runs.
- **Split (shared_CHAT.md, 00:54Z to 01:31Z):**
  - cl_h458: driver `gp_chain_h458.py` and pilot.
  - agy_apex_w10: arm A5, under its own PREREG.
  - cl_fwriter: review.
  - cl_mtc: this PREREG, the budget table, the P201B stack-level gate and all scoring.
- **State when written:** the high-tier pilot is running (10 contexts; single_a 8/10, single_b 7/10, chain 1/10,
  chain_sc 1/10, out-of-fold arms 0/10). No S9 column has been scored against any label. The pilot's three probe arms
  were registered by cl_h458 at 01:18Z (`PROBE_LEDGER.jsonl` lines 6–8) and are adopted here unchanged.
- **Immutability:** this file is made 0444 once its sha is posted in the chat. Changes go into numbered
  `AMENDMENT_<n>.md` files beside it, each posted before the result it affects exists.

## 1. Questions

**Q1, product.** At a fixed budget per tier, does an additive chain of GP programs, each evolved on the residual of the
earlier ones, predict held-out rows better than today's single GP run? This is the GP-alone question. A product run
returns one model, not our stack.

**Q2, ladder.** Does the chained column raise the legacy stack, on P201B TRAIN and then on dev91? This is the Elo question.

**Q3, mechanism.** Is any gain due to fitting residuals, or to having several programs? The A6 controls answer this.

**Prior evidence that the answer may be no:**
- H453: deeper GP search raised SEARCH CV (1.7975 to 1.8372) while the stack fell (1861.17 to 1851.68), and Ultra-GP
  alone went from 1834.7 at g50 to 1814.6 at g1000 cons.
- H454: the UGP g1000 stacks equal the no-UGP stack exactly.
- H410: an own-run Caruana bag of programs gained −0.009 u, while a run-bag of independent runs gained +0.066 u.

Expect a null. A positive result has to clear the controls below.

## 2. Budget table (cl_fwriter 01:10Z, point 1)

The primary unit is **unique candidates evaluated**, the `effort.rs` unit. Worker calls, wall seconds, slot-seconds and
time-outs are recorded beside it for every arm.

| Tier | effort.rs evaluations / wall | G, total generations | Notes |
|---|---|---|---|
| fast | 24 / 10 s | — | not run; the ask excludes it |
| medium | 128 / 60 s | 40 | |
| high | 512 / 300 s | 160 | pilot tier |
| ultra | 4,096 / 1,800 s | 1,250 | |

- **Conversion.** G = evaluations / 3.25 unique candidates per generation, cl_h458's 01:05Z timing (172 candidates in
  40 generations, 48 of them gen-0 seeds).
- **Seed overhead.** Every engine call seeds 48 fresh candidates, so multi-call arms evaluate more candidates than one
  run of G. At high: single about 544, chain 4 × 40 about 688 (+26%), oof_chain 4 × 2 × 20 about 880 (+62%). Measured
  counts replace these estimates in every table.
- **Budget-dominating control.** A chain or out-of-fold arm counts as a win only if it also beats a single run whose
  measured unique candidates are at least the arm's own.
  - In the pilot that control is `single_240` (1.5 G, about 792 candidates, seed 2458). It dominates chain and
    chain_sc but **not** oof_chain (about 880), so the pilot's oof_chain read is flagged "budget not dominated".
  - In the P201B stage the control is single at 2 G (about 1,040 candidates), which dominates every A1 to A4 arm.
- **Wall time.** Out-of-fold arms run K folds side by side, at K times the slots. Product tiers cap wall time with 1
  worker, so wall time and slot-seconds per arm are reported and never hidden behind a parallel run.

## 3. Arms

Settings for A0 to A4 are those of `gp_chain_h458.py`, reviewed at f7fb09e9 (REVIEW_S9_driver_f7fb09e9, sha
01ba8dab…): nu 0.7, committee 3, min_gain 0.002, stages 4, outer folds 4, inner_slots 3.

| Arm | Driver tag | What it is |
|---|---|---|
| A0a / A0b | single, seeds 458 / 1458 | One run of G (today's recipe, no anchor). A0b − A0a is the noise unit: the two share fold partitions (D5), so it measures GP stochasticity only. |
| A0x | single, 1.5 G (pilot) or 2 G (P201B) | The budget-dominating control (§2). |
| A1 | chain | 4 stages of G/4. Stage j+1 is evolved on y minus the running cross-fitted (`*_cons`) column. This is the owner's plain chain. |
| A2 | chain_sc | A1 plus a 3-program NNLS committee per stage, shrinkage nu = 0.7 and the keep rule. The keep rule is "decided on inner cross-fitted error" and is never called honest (D2). |
| A3 | oof_single | 4 outer folds, each one run of G/4 on the other folds. The held-out fold is predicted by a pipeline that never saw it. |
| A4 | oof_chain | 4 outer folds, each a chain_sc of 2 stages of G/8. |
| A5 | agy's OC-FGB-GP | Under PREREG_S9_ARM_A5_OC_FGB_GP.md (sha 3c4d72be…), scored against its own A0_iso. It enters the A0 to A4 tables only once it reports unique candidates (§2). Its prior qhat_v4_heldout.pt (73c1594e…) holds 8,190 dev91 Ultra-GP observations and is flagged canonical_promotion_forbidden, so any A5 dev91 or probe number carries the in-corpus label unless A5 switches to qhat_v2.pt. |
| A6a | committee, no residual | NNLS blend of A0a's own top exported programs: the Q_cons, AKF_cons and PA_cons columns already in its engine snapshot, 4 programs per context in the pilot. No GP compute. Weights are fitted on the Q the programs were selected on, so this arm is in-sample like A2's committee. H410 predicts no gain. |
| A6b | run-bag | Equal-weight mean of A0a and A0b, about 2 G candidates. No extra compute. It is compared with A0x. H410 predicts a gain. |
| A7 | stack residual (conditional) | A chain whose first stage is evolved on y minus the cross-fitted legacy stack (driver `--base-block`). cl_mtc produces the base column: stack weights are refit on the other outer folds' q_ only, so it is approximately honest, with the stacking leak E1. It runs only if §5 rule R1 passes. |

**Mechanism reading (Q3), stated now.**
- A6b ≥ the best of A1 to A4: the value is in ensembling independent runs, not in residuals, and the product
  recommendation is a run-bag.
- A2 > A1 and A6a ≈ A0: the committee helps only within a residual chain.
- A3 ≈ A4 > A0: the gain is honesty and fold-bagging, not staging.

## 4. Boards and order

1. **Pilot (running).** High tier, the first context of each probe family (10 contexts), probe X in the audit slot,
   prior qhat_v2_all.
   - Reads, registered by cl_h458 at 01:18Z, one scoring each:
     - `h458s9.PILOT_HIGH_GP_ALONE`: every arm against single_a;
     - `h458s9.PILOT_HIGH_STAGE_CURVES`: the column after each stage;
     - `h458s9.PILOT_HIGH_STACK_WITH_ICL`: an NNLS blend of the three ICL columns of the M_x2 probe init, with and
       without the arm's GP column.
   - cl_mtc registered `h458s9.PILOT_HIGH_ENSEMBLE_CONTROLS` (A6a and A6b against single_a; `PROBE_LEDGER.jsonl`
     line 10, 01:38:08Z) before any pilot column was scored.
   - The pilot has 10 families, so the exact two-sided sign-flip p has a floor of 0.002. It is a pilot read, not a
     gate.
   - Labels: "fresh-row probe; prior dev91 in-corpus (qhat_v2_all)". STACK_WITH_ICL is also in-corpus, through the
     M_x2 block.
2. **P201B TRAIN (selection), prior qhat_v2.pt (dev91-free, bf05e040…).**
   - Contexts: the 80 P201B contexts with the smallest sha256 of the cid string (64 hex characters, no newline).
     Each is its own family on this board.
   - Tiers: medium and high.
   - Arms: A0a, A0b, A0x (2 G), A2, A4, plus A6a and A6b at no cost. A1 and A3 are added if, in
     `h458s9.PILOT_HIGH_GP_ALONE`, either one beats single_a on at least 7 of the 10 contexts.
   - Ultra (G = 1,250) runs on the first 20 of the 80, for A0a, A0b and the high-tier winner only. It is costed in §6
     and is not started unless the high tier passes R1.
   - Bank: /home/ubuntu/jope-data/h430/bank_P201B.npz (sha da29e8ec…), whose audit slot holds the P201B AUDIT X.
3. **FITSEL (frozen confirm), 106 contexts.** Only the arm selected on P201B, against A0a and A0x, at the selected
   tier, with no retuning.
4. **S7 fresh-row probe (dev91-freshprobe-v1).** The selected arm and its controls, named on `PROBE_LEDGER.jsonl` first.
   Priced in family-games against the externals: five of the nine close families are on the board.
5. **dev91, one official look** (`LOOK_LEDGER.jsonl`, named first).
   - It scores the selected arm against A0a, both added to the same base.
   - Two bases are scored in the same receipt:
     - in-corpus: the e44 legacy base;
     - fair closure (post hoc): the g00_init base, H449-clean, with no dev91 row.
   - Columns use prior qhat_v2.pt.
   - It counts as one of Track S's two looks, and only the pre-registered pair is read.

## 5. Statistics and decision rules

- **Utility.** u = −ln(RMSE / RMSE of the SEARCH mean) on AUDIT (P201B, FITSEL, dev91) or on probe rows (S7).
- **Test.** `h335_stats.primary_test` on per-context differences clipped to ±0.5 and averaged per family:
  T = family mean, one-sided sign-flip p (exact up to 24 families, otherwise 2^20 Monte Carlo draws, seed 335160),
  lofo_min and median. W/T/L uses ties within 1e-4. The probe scorer reports the exact two-sided p.
- **Noise.** A0b − A0a is reported beside every contrast as a null contrast. If its own test passes, the board is
  declared miscalibrated for that tier and no arm passes there.
- **R1, GP-alone pass (Q1) on P201B at a tier.** Arm − A0a and arm − A0b both pass `primary_test` (T > 0, p < 0.025,
  lofo_min > 0, median ≥ 0), **and** arm − A0x has T > 0.
- **R2, stack pass (Q2) on P201B.**
  - The legacy stack (V0: m7, k3, f4, icl and the H416_CONFIRM GP pool) plus the arm column is compared with V0 plus
    the A0a column, and separately with V0 plus A0b. Both pass `primary_test`.
  - The P201B Elo (21 entrants plus the arm) is reported for V0, V0 + A0a and V0 + arm.
  - A tier can pass R1 without R2. That means product value without ladder value, and it is reported that way.
- **Selection.** The arm passing R1 at the highest tier with the largest T over A0a is selected. If none passes R1,
  S9 stops after P201B and reports a null per tier, with no FITSEL, probe or dev91 stage.
- **FITSEL confirm.** The selected arm against A0a: T > 0 and W > L, at GP-alone level and stack level.
- **dev91 look.** Reported in full whatever its sign:
  - Elo of base + arm and of base + A0a, ranks among the 34, and the externals-only rank in family-games of 418
    (base 298.4);
  - the paired family test;
  - w_extra, the stack's weight on the arm column.
  - No further dev91 look follows from it.

## 6. Compute (ESTIMATES from cl_h458's 01:05Z timing, 3.5 s per generation at 3 slots)

| Stage | Slot-hours |
|---|---|
| P201B high, 80 contexts × (A0a, A0b, A0x 2 G, A2, A4) | ≈ 6.9 × 1,680 slot-s × 80 ≈ 256 |
| P201B medium | ≈ 64 |
| P201B ultra (conditional), 20 contexts × (A0a, A0b, winner) | ≈ 13 per context, ≈ 260 |
| FITSEL | ≈ 1/4 of the P201B high stage for one arm and two controls |

- Hosts and cores are agreed in the chat before launch. Track S's own allowance on xbabe2 is 800% CPU, MemoryMax 16G,
  nice 19.
- Every job stops when MemAvailable < 30 GiB or host PSI full > 10%.
- GP stages run from a reviewed driver SHA, with SETTINGS.json folder claims (D1).
- No result is computed that a rule above does not read (Jepson 11-10).

## 7. Closures, priors and code owned by cl_mtc

- **Closure follows the prior (cl_fwriter F1).** A column is "fair-gp-search-only" only if every arm feeding it used
  qhat_v2.pt; otherwise it is "in-corpus-prior". The pilot is in-corpus-prior.
- **score_h458s.py.** Before any dev91 S9 score, CLOSURE_RANK learns these two labels, mapping them to fair and
  in-corpus. The score label stays the most restrictive closure of every input: the e44 base is in-corpus, and the
  g00_init base takes the fair-closure (post hoc) label.
- **New Track S code, each selftested before use, in ~/h458s-stage/code/:**
  - `s9_p201b_stack.py`: the P201B and FITSEL legacy stack with `--extra` blocks and paired tests, built from
    s2_tune_board / gm_ts pieces with a `--check` bit-identity replay of V0;
  - `s9_ensemble_controls.py`: the A6a and A6b blocks from packed arm blocks, reading no label;
  - `s9_stack_oof.py`: the A7 base column.
- **E2.** Appending probe X to the audit slot leaves dev91 predictions byte-identical. Tested on 27 champion programs ×
  3 environments (cl_h458, 01:29Z); the limit is the operators those 27 programs use. Any dev91 column reused from a
  combined run carries that limit.
- **Not done here.** No engine change, no GPU, no tuning on dev91, no max-of-N across tiers or seeds, and no reuse of
  Tier-C's dev91 vault (agy, 01:09Z).
