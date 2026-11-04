# H317 (`h317_island10_multiscale_curriculum`): Gate H0 rapid-hypothesis lab (TRAIN-only) — closeout

- **Verdict:** `NO_ADOPTION_STAGE_V_SKIPPED`. None of the six Gate H0 experiments adopted a challenger under the
  preregistered contract gate. Per PREREG §2.6, the one-shot DL-Val (dev91) stage was therefore skipped:
  **0 DL-Val rows were read**.
- **Owner:** `cl_fwriter` (Claude Opus 5.5, xbabe2 session).
- **Authority:** Jepson's direct instruction to execute the Gemini "H258" master plan (renumbered H317), with the owner
  answers recorded in PREREG.
- **Commits:**
  - prereg `b560c139a2bc` (2026-11-04 13:11:37Z, before any true-target fit);
  - A1 `4668076b3d29` (compute waiver; P205 × 2 seeds);
  - A2 `774b22f8e9a5` (GP execution layer on xbabe0 + code red-team fixes);
  - A3+A4 `b91f5b44bd19` (Stage V driver; Evo B duplicate-pick fix).

  Every commit is on `origin/v1/next` with a same-agent review receipt (`~/jope_plan/evidence/H317/review_*.md`).
- **Archive:** `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H317/closeout-20261104/h317_closeout_bundle.tar.zst`,
  VersionId `QCaZLShDtGBbI6ftRWYsGPYq6kxYGCIz`, sha256 `525ac163…`. An exact-version GET was verified and all
  2,199 member shas were checked.

## 1. Hypothesis
The source plan claimed large lifts from six levers:
1. a larger corpus + B = 32 perturbed table views;
2. a bigger axial ranker;
3. SAM/Muon/Lion + WSD/SWA;
4. multi-regime Gen-0 seeding;
5. 10-generation 4-island MAP-Elites + ε-lexicase GP;
6. Top-N S1-direct stacking.

H317 tested each lever on TRAIN only. Each challenger had to beat its baseline under the repository's contract gate:
- Holm-adjusted LCB95 > +0.001;
- family median ≥ 0;
- ≤ 10% of families losing > 0.01;
- Holm-adjusted one-sided sign-flip p < 0.025;
- leave-one-family-out minimum > 0.

## 2. Setup (what is different from the plan, and why)
- **Data.** 1,219 DL-Train contexts / 205 families only. res-01's "~1,100 new contexts" are exactly the 1,219 contexts
  H256 already used.
- **Read boundary.** Guard v2 (audit hook) plus a deny-by-default sandbox: bubblewrap on xbabe2, Docker on xbabe0.
  0 production guard refusals.
- **Selection-bias control (the central design change).** H258 showed that the trivial `TOP4_PICK` already scores
  1.252 on DL-Val, and an audit of Gemini's H257 code found its DL-Val headline is best-of-9…29 on the same 2-fold CV
  it reports. So every search and stack measurement uses a held-out **AUDIT** slice:
  - 30% of each context's rows;
  - search sees only the remaining 70% SEARCH rows;
  - AUDIT utility = −log RMSE ratio vs the SEARCH-target mean;
  - the worker was verified not to transduce AUDIT rows.
- **Neural block (H0.1–H0.3).** 0-reveal top-1 pick over the 1,152-candidate pool, 5-fold family-grouped OOF, 2 training
  seeds, equal optimizer steps (3,000). The R882 TRAIN regression was bit-exact on every run (context 1.423073434552299,
  family 1.3853554329931828).
- **GP block (H0.4–H0.6).**
  - All 205 families, one context each (P205), × 2 search seeds, on xbabe0 (11 shards × 8 workers, host CPU ≤ 85%).
  - Screener = the H0.1-winning neural config's OOF fold models.
  - The cross-family champion bank excludes the target's family.
  - Islands are defined by the worker's real heads: tree / linear / kernel / portfolio.
- **Red teams before any fit:** statistics, leakage, method brainstorm, prior-evidence miner, Gemini-H257 audit, ops, and
  three code red teams (neural, GP, Stage V). Findings and dispositions are in PREREG §6–§10; DISCLOSURES.md lists every
  out-of-boundary read.

## 3. Outcome

### Neural (TRAIN OOF, family-equal mean native utility; R882 reference 1.3854 has seen all TRAIN families)
| Exp | Baseline (fm) | Challenger: gain vs baseline [Holm LCB], W/T/L | Decision |
| --- | --- | --- | --- |
| H0.1 data/views | A: 525 ctx × 4 views (1.1708) | B 1,219×16: −0.0027 [−0.054]; C 1,219×32+mixup: −0.0033 [−0.088] | **A kept** |
| H0.2 scale | S: d=32, L=1 | M: +0.033 [−0.020], 113/0/92; **L (4.2M params): +0.072 [−0.006], 116/0/89, LOFO +0.050, harm 34%** | **S kept** |
| H0.3 optimizer | opt1 AdamW + cosine | opt1+ scaffold +0.009; SAM +0.011; Muon +0.002; Lion+SAM +0.045 [−0.027] | **opt1 kept** |

### GP (P205 × 2 seeds, family-equal mean AUDIT utility)
| Exp | Baseline | Challengers | Decision |
| --- | --- | --- | --- |
| H0.4 Gen-0 seeding (best-by-SEARCH of the first 8 evals) | A (H257 bank seeds) 1.572 | B 1.679, C 1.632, D 1.700 (Holm LCB +0.010), **E registry presets + bank 1.773: +0.201, Holm LCB +0.043, sign-flip p 0.0004, but harm 23%** | **A kept** (E and D fail only the harm rule) |
| H0.5 evolution (10 gens, 6 live evals/gen) | Evo A panmictic: g0 1.572 → g3 1.753 → g6 1.812 → **g10 1.832** | Evo B 4-island MAP-Elites + ε-lexicase: g10 1.816 (−0.015 [−0.086]) | **Evo A kept** |
| H0.6 stacking (LOFO-CV-selected config, one test) | single best tree 1.832 | **CV3@12 (chosen in 204/205 folds): +0.020, median +0.022, 170/1/34, sign-flip p 0.007; LCB −0.0007, harm 11.2%** (narrow miss); CV5 in-worker NNLS portfolio −0.013; BAG_FINAL −0.019 | **single kept** |

Invalid outputs: 0 across all GP arms. BANK_NN1 (one nearest cross-family champion): 1.259.

## 4. What we learned
1. **Evolution generalizes.** Under a held-out AUDIT slice that search never sees, 10 generations of panmictic GP from
   8 bank seeds lift utility from 1.572 to 1.832. This is real out-of-sample gain, not winner's curse. The 4-island
   machinery adds nothing at equal budget.
2. **The strongest levers were candidate diversity, not selection or model scale.**
   - Seeding with the worker's never-used registry presets (gpr/krr/rff/bagged histgbm2/gam) gave +0.20 at gen 0
     (Holm LCB > 0).
   - A 12-member S1-direct stack gave +0.020 over the gen-10 single tree (170 of 205 families improved).
   - Both failed only the contract's ≤10%-harm rule: the effect is broad in mean but concentrated.
3. **Neural levers were null or small**, as prior waves predicted: data size, views, mixup, optimizers. Scale L is the
   one promising signal (+0.072), but it is not robust enough to pass.
4. **The plan's headline claims do not survive.** H257's DL-Val lift is largely best-of-N selection on the reported
   metric (audit in PREREG §0). With selection controlled, its specific mechanisms (islands, the H257 stack rule,
   active reveals) did not reproduce as gains here.

## 5. Next steps (each needs its own preregistered wave; none was run here)
- **E-seeded GP with a CV3 stack as one locked pipeline**, confirmed TRAIN-only on a fresh family partition and then
  in a one-shot dev91 shot, against budget-matched Evo A and screen-top-K controls. This is the configuration most
  likely to pass.
- **New learner heads** (`xtrees`, `rforest`, `oblivious`; superset worker `2a848c8f…`, parity-verified). They were
  delivered after the GP launch, so they were not admitted (PREREG §2.5); the exploratory report-only run is described
  in §6. The upstream patch for `crates/feat-math` (identity-changing: needs compat_freeze/ADR-0004 handling) is in
  `~/jope_plan/evidence/H317/learners/patch/`.
- **Scale-L ranker** with 3+ seeds and a harm-aware objective.

## 6. Exploratory (report-only, not LOCK-eligible)
- **What ran.** The superset worker added three heads (`xtrees`, `rforest`, `oblivious`). It was parity-verified on
  2,010 program × context pairs, but delivered after the GP launch, so it was not admitted.
  - It ran on the same P205 × 2 seeds, under the same sandbox and screener (the screener cannot featurize the new
    heads).
  - Seed F = Seed E + 6 new-head presets, i.e. 22 Gen-0 evals vs E's 16. Then Evo A for 10 gens with the new heads
    allowed in island I0, then the single best tree and the CV5 portfolio.
  - Three shards first failed on a harness race (concurrent `config.json` temp write); they were rerun sequentially
    under the same shard partition. 410/410 records.
- **Results** (AUDIT, family-equal; paired whole-family bootstrap; descriptive only):
  - F full − E full: **+0.124** [LCB +0.023], sign-flip p 1e-4, harm 7.8%, W/T/L 65/123/17. Not budget-matched
    (22 vs 16 evals).
  - F − E, budget-matched first 8: −0.073. F − A, first 8: +0.128.
  - End to end, exploratory single after 10 gens vs the main run's single: 1.884 vs 1.832, +0.052 [−0.084, +0.192],
    p 0.24, W/T/L 121/1/83.
  - CV5 portfolio within the exploratory run vs its single: +0.010 (p 0.10).
- **New-head usage.** A new head is the chosen output in **37%** of contexts after 10 generations (40% at Gen 0).
- **Reading.** The new heads are selected often and point the right way, but at this budget and power the end-to-end
  gain is not established. A preregistered, budget-matched test (F vs E at equal Gen-0 evals; ideally with the
  screener retrained to featurize the new heads) is the follow-up.
- **Archive.** `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H317/closeout-20261104/h317_explore_learners.tar.zst`,
  VersionId `DpLUJyNKYSm_71ekAW9vtxRJBv4XBtBM`, sha256 `b7ae80b1…`.

## 7. Incidents (all disclosed in PREREG §8–§10 / DISCLOSURES.md)
- **H0.2 relaunch.** The xbabe3 unit refused at its in-process GPU-free check (a foreign Blender job) before training,
  and was relaunched once with the shared-GPU allowance.
- **H0.5 crash.** One H0.5 job crashed on an Evo B duplicate migration splice. It was fixed in A4; 30/30 completed jobs
  replayed byte-identical; resumed.
- **Pre-prereg reads beyond the TRAIN boundary.** Several agents made such reads (file names only; header-only reads of
  29 DL-Val tables by the learner agent before the rules applied). None fed any H317 artifact.
