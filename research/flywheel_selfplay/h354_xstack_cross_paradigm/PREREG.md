# H354 (`h354_xstack_cross_paradigm`): cross-paradigm SEARCH-only super-learner on the locked GP search — PREREG

- **Owner:** `cl_fwriter` (Claude Opus 5.5, xbabe2). Claim: `ns/receipts/H354_claim.txt` (0444, sha256 `18625467…`).
- **Authority:** Jepson, 2026-11-05 ~09:05Z. The instruction: start the next flywheel research wave; first summarize
  recent wins and losses; then propose and run a smarter training concept that maximizes Elo, using TRAIN/P201b ideas
  only; never fit, train, tune or leak labels from the fixed holdout `dev91-audit-v1`; work on `v1/next`.
- **dev91 is not read in this wave.** No DL-Val91 / dev91 / DL-Test / sealed path is opened or mounted. A dev91
  one-shot of the locked candidate is a separate step that needs a new owner decision. That decision is not part of
  this PREREG.
- This PREREG is published to `origin/v1/next` **before any library fit**.
- **Run before publication:** only two harness checks (`phase_d_h354.py align` and `x0`). They fit nothing, score no
  new output, and read only stored H335 Phase T artifacts (§3.2).

## 1. What happened in the recent waves (why a new concept)
- **The one win: H335 `BETTER_MODEL`.**
  - Recipe: Seed F → Evo A, 50 generations × 6 live evals from 48 screened proposals → CV3@12 S1-direct stack.
  - It won by changing the **generator** (new learner heads, 5× more generations) and the **aggregator**. On TRAIN
    P201b the stack added +0.0445 family-mean AUDIT u over the single best tree, the largest single gain in the recipe.
  - **Published dev91 AUDIT-Elo** (12 entrants): 1742.6.
  - **On the 23-entrant reference board** it reads 1708.0. It is behind every tabular foundation model there.
- **The losses: H315–H353, except H317 and H335.** About thirty waves were single-knob changes to the H303 NESTED
  pool-selection ranker: a LightGBM LambdaRank that picks among r882's top-8 slots. Examples:
  - H331: cosine shrinkage;
  - H332: leaf-step clipping;
  - H338: column subsampling;
  - H344: LambdaRank sigmoid 0.5;
  - H347: L1;
  - H350R: L2 = 5;
  - H351: min_child_samples 40.

  Every one ended `*_NOT_ADVANCED` or `INCOMPLETE_*` (resource guard or runtime forecast). The reasons recur:
  1. **The base effect is small.** The BASE gain is +0.048 family-mean, but the literal family-median gain is only
     +0.0036: most families gain about nothing.
  2. **Treatment effects are smaller still.** They are ±0.01, below the +0.002 paired drop-top-10 bar with a
     positive LCB95, and they flip sign across roots. Examples:
     - H331: per-root medians −0.005 / −0.009 / −0.011;
     - H344: −0.0006 / −0.0024 / −0.011, LCB −0.0076.
  3. **The selection is config-sensitive.** About 7 of 36 fold/root config picks agree.
  4. **Bigger mean gains were not robust.** H328 +0.071, H329 +0.059, H338 +0.058 and H347 +0.056 came from a few
     families. They failed the drop-top-10, harm or seed-consistency gates.
- **H317** closed `NO_ADOPTION_STAGE_V_SKIPPED`: every lever missed the ≤10% harm rule.
- **Lesson.** Tuning one hyperparameter of a saturated selector cannot move Elo. The binding constraint is the
  **diversity and quality of the candidate set** that the aggregator combines.
- **TRAIN evidence for diversity across paradigms.** On the TRAIN board `p201b-train-audit-v1`, H335_AGG is first.
  Yet two learners from other paradigms beat it on a large minority of families:
  - CatBoost beats it on 42/201 families;
  - the Rust LLM regressor `Opus5p5_Extra` beats it on 68/201.

  Both lose on average (−0.50 and −0.36 u), so their errors differ from the GP stack's. A super-learner that weights
  them **on SEARCH only** can take those families without paying on the others.
- **Disclosure.** That TRAIN head-to-head table was read before this PREREG. P201b is therefore a **development**
  panel for H354. The confirmatory claim is made only on the fresh panel P201c (§4).

## 2. The concept: XSTACK
For each context:
1. **GP search (unchanged).** Run the locked H335 recipe `AGG.g50.CV3@12` (LOCK `4d58dd37`). Its K ≤ 12 stack members
   (`cv3_members`) are the GP candidates, unchanged and in stored order.
2. **Library (fixed; every license permits commercial use).** Each member gets out-of-fold predictions on **H335's own
   SEARCH two-fold split**: p0 = predictions for fold-0 rows from a fit on fold 1; p1 the mirror.
   - **Classical.** `catboost.v1`, `lightgbm.v1`, `xgboost.v1`, `random_forest.v1`.
     - These are the reference-board registry ids and configs, used through its own `score_reference.fit_predict`.
     - Harness sha256: `score_reference.py` `169cb04e…`, `models.json` `20d100c4…`.
   - **LLM.** The four Rust LLM regressors `Opus5p5_Extra`, `Gemini4_argon_high`, `GPT6p1_Sol_Max`, `GPT6_Astra_Max`.
     - They run through the pinned H335 frontier binary `b359807c…`, which fits fold 0 and predicts fold 1; the
       folds are relabelled for p0 / p1.
3. **Stacker (frozen, unchanged).** `stacking_h317.weights_s1_direct`, the CV3 S1-direct Frank-Wolfe on 5
   target-stratified subfolds per fitting fold, with λ = 0.02 toward softmax(fitness / 0.15).
   - Candidates: the GP members plus every library member that **passes the S1 gate on SEARCH** (both fold log-ratios
     ≤ 0 and exp(mean) ≤ 0.90).
   - A library member's S1 fitness uses complexity `C_LIB = 0`; GP members keep their DSL complexity.
   - Cross-fitted SEARCH weights: w0 / w1; deployed weights: wa (pooled OOF).
4. **AUDIT (once per arm per context, after the weights are fixed).**
   - Prediction = Σ wa_i · (member i fit on all SEARCH rows → AUDIT rows).
   - Scored by the frozen `stack_audit_utility`: u = −ln(RMSE_audit / RMSE of the SEARCH-mean predictor).
   - A member whose AUDIT output failed contributes the SEARCH mean (H317 rule). Weights never change with AUDIT
     outcomes.
5. **No AUDIT label reaches the stacker.** `stack_arm` receives SEARCH arrays only, and `audit_arm` is called once per
   arm after the weights are fixed.

**Arms** (`h354_core.ARMS`):
- **X0** = GP members only. This is the harness check: it must equal the stored H335 output.
- **X_CL** = GP + 4 classical.
- **X_LLM** = GP + 4 LLM.
- **X_ALL** = GP + all 8.

**Compute.** XSTACK adds 3 fits per library member per context to the GP search: 2 OOF fits plus 1 SEARCH→AUDIT fit.
The GP part is identical (same pools, same evals).

## 3. Phase D — development on TRAIN P201b (201 families × 1 context, H335 Phase T seed 335101)
### 3.1 Inputs (stored; nothing in the GP search is re-run)
- `~/jope-data/h335-cl/runs/T_xbabe{0,2}/`:
  - the stage records (`outputs["AGG.g50.CV3@12"]`);
  - the memo (each member's SEARCH OOF predictions p0 / p1 and its AUDIT predictions pa);
  - the SEARCH and AUDIT worker envelopes.
- The frontier AUDIT envelopes `frontier/T/envs/<cid>.jtt1`; these are the reference-board `p201b-train-audit-v1`
  envelopes.
- **Classical AUDIT predictions** are the **stored** reference-board predictions (`preds/<model>/<cid>.f64`, each
  sha-checked against its record). They are not re-run.
- **LLM AUDIT predictions** are not stored, only their sha256. They are re-run by the pinned binary, and each must
  reproduce the stored sha256 or the run stops.

### 3.2 Harness checks (run before this PREREG; results in `~/jope-data/h354-xs/D/`)
- **`align`:** 201/201 contexts aligned. The maximum deviations:
  - member log-ratio recomputed from p0 / p1: 8e-15;
  - AUDIT baseline: 5e-15;
  - member AUDIT RMSE: 2e-15;
  - SEARCH rows of the AUDIT envelope vs the SEARCH envelope: 0;
  - GP AUDIT envelope vs the reference-board envelope: byte-identical rows and folds (0).

  6 of the stored stacks are single-member fallbacks.
- **`x0`:** the frozen stacker on the stored members reproduces the stored `w_fold0`, `w_fold1` and `w_audit` exactly
  (max |Δ| = 0.0). The AUDIT utility matches the stored value and the reference-board `h335.H335_AGG` value to 9.6e-12
  (tolerance 1e-9), on 201/201 contexts.

### 3.3 Runs
0. **Code root.** Every run uses a frozen read-only copy, `~/h354-stage/code-<TAG>`, made by `stage_h354.sh`
   (create-once, every file listed in `CODE.sha256`). The H317 guard, imported for the frozen stacker, refuses any
   other wave's `~/jope_plan/evidence/` tree; H335 ran from `~/h335-stage/code` the same way. The staged files must
   be byte-identical to the files in this commit, and each run receipt records the `CODE.sha256` digest.
1. `phase_d_h354.py prepare`: copy SEARCH envelopes, AUDIT envelopes, stored classical AUDIT predictions and the
   frontier sha256 into `D/inputs` (create-once, `MANIFEST.sha256`).
2. `sandbox_h354.sh D python3 lib_oof_h354.py --audit stored`, on xbabe2:
   - cores 38-53 / 102-117, at most 4 concurrent 8-thread classical fits;
   - nice 19, MemoryMax 48 GiB, no swap;
   - bwrap with no network, and only `D/inputs` (read-only) and `D/lib` (read-write) visible.
3. `phase_d_h354.py arms`: all four arms on 201 contexts. X0 inside `arms` must equal the `x0` check.
4. `phase_d_h354.py decide`.

### 3.4 Decision (LOCK rule; fixed now)
- Statistics are the frozen H335 functions (`h335_stats.py` `ee3b4543…`), over 201 families:
  - clipped family-mean contrast, clip ±0.5;
  - one-sided sign-flip with 2^20 Monte Carlo draws (F = 201 > 24);
  - LOFO minimum and family median.
- **Choosing the arm.** Find the candidate arm (X_CL / X_LLM / X_ALL) with the best family-mean AUDIT u.
  - An arm is **eligible** if its paired family-mean gap to the best is ≤ one paired family SE.
  - Among eligible arms, pick the one with the fewest library members; tie → lower library CPU (X_LLM before X_CL).
- **LOCK** that arm only if its H335 primary test vs X0 passes: p < 0.025, LOFO min > 0, family median ≥ 0, T > 0.
  Otherwise the wave closes `XSTACK_NOT_ADVANCED`, and Phase C does not run.
- **Descriptive output.** A Phase D AUDIT-Elo over the arms and the 8 library learners alone (anchor `h354.X0` = 1500,
  family bootstrap). It is labelled development: P201b was H335's settings panel and was read for this design.
- **LOCK record.** `LOCK.json` holds the arm, the library ids, every pin and this PREREG's commit. Its sha256 goes into
  a create-once 0444 receipt `ns/receipts/H354_LOCK_sha256.txt` before Phase C starts.

## 4. Phase C — confirmation on a fresh TRAIN panel P201c (never read by any GP run, by me, or at AUDIT)
- **Panel.**
  - P201c = the **third** seeded context of every TRAIN family with ≥ 3 contexts.
  - It uses the same seeded order as P205 (first context) and P201b (second); disjoint from both, asserted.
  - It is TRAIN only (`selected.json["train"]`).
- **GP.**
  - The H335 AGG recipe, frozen: the H335 code at `origin/v1/next` and the superset worker `2a848c8f…`.
  - The learners-v1 grammar; the unchanged H317 H0.1-A screener fold models (each context's out-of-family fold).
  - Seed F, then Evo A to **g50**; output `AGG.g50.CV3@12`.
  - Search seed **354101**, one seed per context, as in Phase T.
  - Views are encoded with the frozen H317 recipe (`h335_encode.py` path). DOPE `c51a56d0` is read-only and its sha is
    checked before and after.
- **Library.** The LOCKED arm's members, through the same runner, with `--audit fit`: SEARCH→AUDIT fits, since none
  are stored.
- **Hosts.**
  - GP on xbabe0 (idle at claim time): the H335 Stage V docker / sandbox path, 96 workers.
  - Library on xbabe2 (same caps as §3.3).
  - CPU ≤ 85% per host.
- **Implementation.** The Phase C driver code may be added in an amendment commit before Phase C. That commit may
  only implement this section and may change no rule here.
- **Primary (one test).** XLOCK vs X0 = `AGG.g50.CV3@12`, on the same P201c pools, using the H335 primary procedure.
  - Pass ⇔ T > 0 ∧ sign-flip p < 0.025 ∧ LOFO min > 0 ∧ family median ≥ 0.
- **Adoption-grade secondary:** the raw H317 contract gate (unclipped LCB95 > 0.001, median ≥ 0, ≤ 10% of families
  harmed by more than 0.01, sign-flip, LOFO).
- **Verdicts.**

  | Verdict | When |
  | --- | --- |
  | `XSTACK_CONFIRMED` | the primary passes |
  | `XSTACK_CONFIRMED_ADOPTION_GRADE` | the primary and the secondary both pass |
  | `XSTACK_INFERIOR` | the mirrored primary passes |
  | `XSTACK_NOT_CONFIRMED` | otherwise |
  | `INCOMPLETE` | an infrastructure failure or a missed deadline |

- **Headline.**
  - Family-mean AUDIT gain and % RMSE cut; families won / tied / lost.
  - AUDIT-Elo on P201c over XLOCK, X0 and the library learners alone (anchor X0 = 1500), with a family-bootstrap CI.
  - Library weight mass: the share of the deployed weight given to non-GP members.
- **Deadline.** Phase C compute must end by **2026-11-05 22:00Z**; otherwise the verdict is `INCOMPLETE`. No partial
  scoring, and no cut cells.

## 5. Not in scope; risks
- **No dev91, no Elo on the fixed test board.** The reference-board ladder stays untouched by this wave.
- **No foundation models.** TabPFN 3.5 / v3 and TabFM have non-commercial weights; Kumo-Tabular's model card is
  unverified. A stack with them is an owner option, not part of H354.
- **No change to** the H335 recipe, the H317 stacker, the screener, the grammar or any library config. No
  hyperparameter search.
- **Phase D risk.** In Phase D, the classical OOF fits and the stored AUDIT fit use the same registry config, but the
  thread scheduling may differ from the stored run. Only the stored AUDIT predictions are used for AUDIT, so the
  AUDIT side is bit-identical to the reference board.
- **Ops.**
  - The flywheel writer stays STOPPED; DOPE `c51a56d0` is not modified.
  - No deletes; all work files are kept.
  - Create-only S3 archive at closeout, with exact-version GET and member sha verification.
  - Commits go to `v1/next` via a private index, non-force, with a same-agent review receipt.
  - The AGENT_CHAT claim and release are made under flock.
