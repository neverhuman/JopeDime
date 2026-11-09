# H335 (`h335_aggressive_locked_elo`): did the H317 cycle produce a better model? — closeout

- **Verdict: `BETTER_MODEL`.** On the one-shot DL-Val91 read, the locked pipeline beats both preregistered controls in
  the fixed sequence:
  - **P1**, the incumbent GP recipe H257_REPLICA;
  - **P2**, H317's recipe given the same number of evaluations (OLD_MATCHED).
- **Owner:** `cl_fwriter` (Claude Opus 5.5, xbabe2 session).
- **Authority:** Jepson's direct instruction (2026-11-04): "continue working forward to help us conclude if the new
  research wave produced a better model, and please take your more aggressive inference". There was no power gate
  (owner: "You should just be training and testing").
- **Commits** (all on `origin/v1/next`, each with a same-agent review receipt in `~/jope_plan/evidence/H335/review_*.md`):
  - prereg `e64f945d13e4` (before any fit);
  - A1 `8b70068544e2` (Stage V driver and the Phase T decision record, before the LOCK);
  - this closeout.
- **LOCK** `4d58dd37d023db91…`, receipt `ns/receipts/H335_LOCK_sha256.txt` (0444), written 2026-11-05 00:55Z.
  - Unlocked once at 00:56:16Z.
  - Stats created `FIRST_OUTCOME_READ` at 02:24:43Z, after all 364 unit records and 364 LLM records had passed their
    identity checks.
- **Archive:** `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H335/closeout-20261105/h335_closeout_bundle.tar.zst`,
  VersionId `hOD8sV_aiBkSa0nlqpSpthqNvYpIclFM`, sha256 `cc62ccae…` (create-only; exact-version GET verified; all 1,308
  member shas checked; `archive_manifest.json`). DL-Val tables, pools, views and LLM envelopes stay local.

## 1. Question and design
**What H317 trained** (per context × 2 search seeds; the owner asked "batches? generations?"):
- **GP search.** Gen-0 seeds, then **10 generations**. Each generation, the neural ranker **screens 48 proposals** and
  the best **6 are evaluated live**: 68 live evals per context in total.
- **Neural screener.** **3,000 optimizer steps × batch 16 contexts × slate 128 candidates**, 5-fold family-grouped
  OOF, 2 seeds. H335 reuses these screener models unchanged; nothing neural was retrained.
- **Outcome.** H317 closed `NO_ADOPTION_STAGE_V_SKIPPED`: dev91 was never read, so there was no Elo.

**H335's "more aggressive inference":** spend the compute on **generations, not batch size**. The red team showed that
K = 16 live evals per generation would cut exploration from 33% to 12.5% and buy no throughput.
1. **Seed F** (22 evals): 6 new-head presets from the superset worker (extra-trees / random forest / oblivious trees),
   8 registry presets and 8 cross-family bank seeds.
2. **Evo A, panmictic.** The same K = 6 live / P = 48 screened per generation, run up to **107 generations** (664 evals,
   ≈ 10× H317), with checkpoints at g10 / 25 / 50 / 75 / 107.
3. **Output.** A **CV3@12** S1-direct stack, chosen on SEARCH rows only.

**Selection-bias control.** Every output is scored exactly once on a 30% held-out **AUDIT** slice the search never sees:
u = −ln(RMSE_audit / RMSE of the SEARCH-mean predictor).

**Primary.**
- d = clip(u_arm − u_ctrl, ±0.5) per context, averaged within family (19 families).
- Exact one-sided sign-flip over all 2^19 sign vectors.
- Pass ⇔ p < 0.025, leave-one-family-out (LOFO) min > 0, family median ≥ 0, T > 0.
- Fixed sequence: P2 counts only when P1 passes.

## 2. Phase T (TRAIN, P201b: 201 families × 1 context × seed 335101) — fixes the settings
- **Run.** 2026-11-04 22:04Z → 2026-11-05 00:52Z. 402/402 units, 284,087 worker runs, 0 tracebacks, 0 invalid outputs.
- **Decision (PREREG §3.3).**

  | Checkpoint (AGG, CV3@12) | AUDIT family mean |
  | --- | --- |
  | g10 | 1.9030 |
  | g25 | 1.9285 |
  | **g50** | **1.9340** |
  | g75 | 1.9391 |
  | g107 | 1.9448 |

  - **G\* = 50:** the smallest checkpoint within 1 paired SE (0.0147) of the best, g107.
  - **Stack:** CV3@12, which is +0.0445 over single at G\*.
  - The projected dev91 wall time was 2.6 h, so the time cap did not bind. Actual: 1 h 26 min.
- **TRAIN AUDIT-Elo** (descriptive; anchor H257_REPLICA = 1500; settings were picked on this panel, so it is mildly
  optimistic):

  | Entrant | Elo | 95% CI |
  | --- | --- | --- |
  | H335_AGG | 1680 | 1640–1721 |
  | OLD_MATCHED_CV3 | 1632 | 1598–1668 |
  | Opus5p5_Extra | 1565 | 1522–1610 |
  | H257_REPLICA | 1500 | anchor |
  | OLD_MATCHED | 1492 | 1458–1528 |
  | H317_BASE | 1458 | 1426–1490 |

- **TRAIN contrasts** of the locked output (201 families, Monte Carlo p):

  | Control | Gain | Clipped T | Families W/T/L | Primary | Raw adoption gate |
  | --- | --- | --- | --- | --- | --- |
  | H257_REPLICA | +0.165 | +0.104 | 156/0/45 | pass | fails harm ≤ 10% |
  | OLD_MATCHED | +0.038 | +0.080 | 153/1/47 | pass | fails |
  | H317_BASE | +0.085 | +0.098 | 163/3/35 | pass | fails |

## 3. Phase V — the one-shot DL-Val91 read (91 contexts / 19 families × seeds 335101 / 335102, xbabe0)
- **Harness.**
  - R882 Native91 regression: 91/91 bit-exact (mean 1.1377941898164525, |Δ| = 0).
  - Dedupe: 0 dev91 tables overlap any TRAIN table, so all 91 are used.
  - Encoder parity: exact.
  - LLM regressors: 364/364 completed.
- **Run.** 00:58Z → 02:24Z. 364/364 units, 124,147 worker runs. 0 failed units, outputs or segments; 0 invalid outputs;
  0 resumes; 0 culprit attempt lines.
  - Live-eval match AGG vs OLD_MATCHED: max |Δ| = 2, 0 violations.
  - Live evals per context × seed: AGG 322, OLD_MATCHED 320, H317_BASE 68, H257_REPLICA 23.

**Primary tests (both pass → `BETTER_MODEL`):**

| Test | Control | Clipped T | Exact sign-flip p | LOFO min | Family median | Families W/L |
| --- | --- | --- | --- | --- | --- | --- |
| **P1** | H257_REPLICA | +0.111 | **0.0011** | +0.089 | +0.068 | 17/2 |
| **P2** | OLD_MATCHED | +0.078 | **0.0030** | +0.054 | +0.044 | 18/1 |

**Contrasts of the locked output** (unclipped family-mean AUDIT gain, and the matching RMSE reduction 1 − e^−gain):

| Control | Gain (RMSE cut) | Clipped T | p | Families W/T/L | Contexts W/T/L |
| --- | --- | --- | --- | --- | --- |
| H257_REPLICA (incumbent GP recipe) | +0.189 (17.2%) | +0.111 | 0.0011 | 17/0/2 | 76/0/15 |
| H317_BASE (H317 as shipped) | +0.152 (14.1%) | +0.090 | 0.0030 | 18/0/1 | 71/0/20 |
| OLD_MATCHED (H317 recipe, same evals) | +0.113 (10.7%) | +0.078 | 0.0030 | 18/0/1 | 74/0/17 |
| OLD_MATCHED_CV3 (same + the stack) | +0.099 (9.4%) | +0.064 | 0.0045 | 15/0/4 | 65/0/26 |
| BANK_NN8_PORTFOLIO | +0.288 (25.0%) | +0.143 | < 0.0001 | 17/0/2 | 74/0/17 |
| BANK_NN1 | +0.656 (48.1%) | +0.258 | < 0.0001 | 19/0/0 | 85/0/6 |

**Adoption-grade secondary** (the raw H317 contract gate: unclipped, LCB95 > 0.001, median ≥ 0, ≤ 10% of families
harmed > 0.01, sign-flip, LOFO):
- **vs OLD_MATCHED: PASS** on every check (LCB95 +0.033; 1/19 harmed).
- **vs H257_REPLICA: fails only the harm rule**, at 2/19 families = 10.5%. It passes LCB95 (+0.058), median, sign-flip
  and LOFO.

**dev91 AUDIT-Elo** (12 pre-declared entrants; family-grouped Bradley–Terry; anchor H257_REPLICA = 1500; family
bootstrap 95% CI, descriptive):

| Entrant | Elo | 95% CI | AUDIT family mean |
| --- | --- | --- | --- |
| **H335_AGG** (`AGG.g50.CV3@12`) | **1742.6** | 1640–1869 | 1.5592 |
| OLD_MATCHED_CV3 | 1654.7 | 1575–1744 | 1.4607 |
| Opus5p5_Extra (LLM regressor) | 1597.0 | 1463–1728 | 1.2943 |
| OLD_MATCHED | 1538.0 | 1472–1613 | 1.4465 |
| H317_BASE | 1527.8 | 1483–1577 | 1.4073 |
| Gemini4_argon_high | 1526.0 | 1394–1658 | 1.3587 |
| H257_REPLICA | 1500 | anchor | 1.3703 |
| GPT6p1_Sol_Max | 1491.6 | 1387–1595 | 1.3543 |
| BANK_NN8_PORTFOLIO | 1456.2 | 1372–1542 | 1.2715 |
| BANK_NN8_PICK | 1413.8 | 1341–1485 | 1.2636 |
| GPT6_Astra_Max | 1274.7 | 1107–1420 | 1.0215 |
| BANK_NN1 | 1161.2 | 1027–1262 | 0.9031 |

- Pairwise H335_AGG vs H257_REPLICA: 1790 (95% CI 1669–1967).
- This AUDIT-Elo scale is **not** the legacy dev91 Elo scale (see DISCLOSURES).

## 4. Plain answer
**Yes: the H317 cycle, pushed with more aggressive inference, produced a better model.**
- On dev91, which this lineage read once, after the LOCK, the locked pipeline:
  - **beats the incumbent GP recipe** in 17 of 19 families (17% lower held-out RMSE);
  - **beats what H317 shipped** in 18 of 19 families (14% lower);
  - **beats H317's recipe given the same 322 live evals** in 18 of 19 families (11% lower).
- So the gain is a better recipe, not only more compute.
- It is the top entrant on the AUDIT-Elo ladder (1743), ahead of every LLM regressor (best 1597) and of H317 as shipped
  (1528, +215).
- Seed F's new heads and the CV3 stack both contribute: the same stack on H317's trees (OLD_MATCHED_CV3) gets only part
  of the way, +0.014 over OLD_MATCHED vs +0.113 for the full pipeline.
- **What it costs.** About 14× the live evals of H257_REPLICA and 4.7× H317's. The dev91 run took 1 h 26 min on
  xbabe0's 96 workers.
- **Caveats.**
  - dev91 is a reuse cohort (H257 tuned on it), so this is the first read by this lineage, not a pristine holdout.
  - H257_REPLICA is H257's GP recipe, not its pool-champion branch.
  - Against H257_REPLICA the strict production gate misses by one family on the harm rule.
- **Adoption** of `AGG.g50.CV3@12` as the production GP recipe is an owner decision and was not made here. The flywheel
  writer stays stopped.

## 5. Integrity
- Guard v2 plus the deny-by-default sandboxes (bwrap on xbabe2, Docker on xbabe0): 0 refusals beyond the 17-line test
  baseline. 0 DL-Val-path refusals. DOPE `c51a56d0` unchanged.
- The frozen H317 modules were imported unmodified (18 pins). The LOCK verified 47 runtime files on every host, at the
  unlock and at every resume.
- See `DISCLOSURES.md` for the cohort, the selection on TRAIN, Seed-F short counts, timeouts and host notes.

## Files
- `PREREG.md`: the design, rules and amendment A1. `metrics.json`: Phase T decision, TRAIN Elo and the full dev91 readout.
- Code: `h335_core.py`, `h335_lab.py`, `h335_stats.py`, `stage_v_h335.py`, `h335_frontier.py`,
  `frontier_audit_h335.{rs,py}`, `gp_grammar_h317_learners_v1.py`, plus the sandboxes and runbook
  `stage_v_mounts_h335.md`.
- `h335_train_elo.py`: the descriptive TRAIN Elo.
