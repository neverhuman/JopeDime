# H364 (`h364_xstack_two_level`): XSTACK-2L, a two-level stack on the frozen H335 GP, on fresh TRAIN P201d — PREREG

- **Owner:** `cl_fwriter` (Claude Opus 5.5, xbabe2). Claim: `ns/receipts/H364_claim.txt` (0444, sha256 `460fe3bc…`).
- **Authority:** Jepson, 2026-11-05 ~11:31Z: pre-register and run the proposed two-level stack on a fresh TRAIN-only
  panel P201d. Freeze the H335 AGG GP weights. Fit the classical library as a second level on the GP output, with
  shrinkage toward GP-only and a weight cap, especially for near-perfect families. Set the guard from the H354/H357
  per-family loss analysis before seeing P201d results. Report Elo vs H335 AGG. The owner named this wave "H363";
  H363 was already claimed by `sol_v1_jrb01` (11:28:48Z), so this is H364.
- **dev91 is not read.** No DL-Val91 / dev91 / DL-Test / sealed / holdout path is opened or mounted, and no holdout
  model is rescored.
- **Two-part publication, both before any P201d outcome is read.**
  - **Part A** (this text, §1–§6) is published to `origin/v1/next` before any P201d view, GP run, fit or score.
  - **Part B** is `DESIGN.json` + §7 + the scoring code. It holds the second-level constants, chosen by the §4.3 rule on
    P201b + P201c only. It is published before any P201d GP or library output is pulled to xbabe2 or read.
  - `run_c_h364.sh` refuses `pull`, `prep`, `lib` and `score` unless the staged `DESIGN.json` is byte-identical to the
    one on `origin/v1/next`.

## 1. Why this wave
H357 closed `XSTACK_CL_NOT_CONFIRMED` (`590699aac3`). Adding the 4 classical learners to the GP stack's candidate set
gave +0.0040 (p = 0.074) on P201c, half of P201b's +0.0093.
- **The cause of its large losses:** in the three worst contexts (−0.695, −0.252, −0.153) the library weight was about
  zero.
  - The losses came from the stacker **re-weighting the GP members** once library columns joined the candidate set.
  - Two of those contexts are near-perfect fits (X0 AUDIT u = 8.37, 7.29), where small weight shifts are large
    log-RMSE changes.
- **H354's worst context** (`351a6d94`, X0 u = 4.59) lost through library members that extrapolated badly on the
  SEARCH → AUDIT refit.
- **Hence:** leave the GP stack exactly as locked, and let the library move the prediction only through one small,
  capped, shrunk second level. Hold it at zero where the GP is already near-perfect, or where the second level does not
  help out-of-fold on SEARCH.

## 2. Hypothesis and arms
- **Hypothesis.** On fresh TRAIN contexts, the two-level stack 2L beats X0 in family-mean AUDIT utility.
- **X0** = the locked H335 recipe output `AGG.g50.CV3@12`, LOCK `4d58dd37` (H335 BETTER_MODEL, dev91 AUDIT-Elo 1742.6).
  It is rerun from scratch on P201d with search seed 364101.
  - Its **stored** weights (`w_fold0`, `w_fold1`, `w_audit`) are used as stored. The stacker is not re-solved.
  - **X0 harness check:** on every context, the stored weights must reproduce the stored cross-fitted SEARCH log-ratios
    and the stored AUDIT utility within 1e-9.
- **2L** = X0 + one second level over the classical library `catboost.v1`, `lightgbm.v1`, `xgboost.v1` and
  `random_forest.v1`. These use the reference-board configs through `score_reference.fit_predict`, as in H357.
  - **Library columns.** Each learner's SEARCH OOF predictions p0 (fit fold 1) and p1 (fit fold 0), plus its AUDIT
    prediction (one fit on all SEARCH rows). A learner is a column only if it completed with finite predictions and
    passes the S1 gate on SEARCH (as in H357).
  - **Model.** ŷ = x + Σⱼ βⱼ (Lⱼ − x), where x is the X0 prediction. Constraints: βⱼ ≥ 0 and Σⱼ βⱼ ≤ c(context). The X0
    weight is 1 − Σβ, so the GP members' relative weights never change.
  - **Objective** on a fitting set: mean((x + Dβ − y)²) / mean((x − y)²) + λ‖β‖², with D = [Lⱼ − x].
    - The objective is scaled by X0's own error, so λ is a scale-free shrinkage toward GP-only.
    - On two folds, the objective is the mean of the two folds' scaled terms.
    - It is solved exactly: a convex QP in ≤ 4 variables, by KKT enumeration of the supports with and without the sum
      constraint active.
  - **Cross-fitting.** At level 1, X0's fold-k SEARCH prediction is the stored `P_k @ w_fold_k`; each w_fold_k was fit on
    the other fold.
    - β⁽⁰⁾ is fit on fold-1 rows and applied to fold 0; β⁽¹⁾ the reverse.
    - On a fitting fold, x is the X0 prediction made with the weights fit on that fold. It is in-sample for X0, so the
      fitted β leans toward GP-only and never uses the evaluation fold's targets.
    - β⁽ᵃ⁾ for AUDIT is fit on both folds, with x = `P_k @ w_audit`.
  - **Acceptance gate.** β⁽ᵃ⁾ is used only if the cross-fitted 2L SEARCH log-RMSE ratio beats X0's cross-fitted ratio by
    at least δ on **both** folds. Otherwise β⁽ᵃ⁾ = 0 and 2L = X0 (a tie).
  - **Near-perfect guard.** If X0's cross-fitted SEARCH utility −½(lr₀ + lr₁) is at least τ, then c = 0 and 2L = X0.
  - **AUDIT prediction:** x_AUDIT + Σⱼ β⁽ᵃ⁾ⱼ (Lⱼ,AUDIT − x_AUDIT).
    - x_AUDIT is the stored X0 AUDIT prediction: stored `w_audit` × member AUDIT predictions, with the H317
      SEARCH-mean substitution for a failed member.
    - A declared library column whose AUDIT fit failed takes the SEARCH target mean (the H317 rule).
- **Descriptive arms (no test):**
  - `2L_noguard`: the chosen c, λ, δ with τ = ∞;
  - each library learner alone;
  - X_CL (H357's one-level stack) is **not** run, because it would need an audit-extra pass.

## 3. Panel and procedure
- **P201d.**
  - Definition: the fourth seeded context (`h335_core._ctx_order`) of every TRAIN family with ≥ 4 contexts.
  - Size: 190 contexts in 190 families; cids sha256 `9a1ace6cfd9abdf6c436d980b61d98b59825e56ea3719cd67911a0f2aa25de98`.
  - Disjoint from P205 (first), P201b (second, H335 Phase T, H354) and P201c (third, H357), which is asserted.
  - What was computed before this PREREG: the panel rule alone, which read only the TRAIN row index. No view, GP, fit
    or score exists for P201d.
  - **Disclosure:** P201d's families are the same families as P201b/P201c, and 172 of its 190 datasets also appear in
    P201c as a different context. P201d is fresh at the context level, not the dataset level. The design (§4.3) is
    chosen on P201b + P201c, so dataset-level dependence between the dev and confirmation panels is possible.
- **GP: `gp_c_h364.py` on xbabe0.** It is identical to H357's `gp_c_h357.py` except for the panel, seed, wave labels
  and the removed audit-extra command.
  - It runs the frozen H335 code (byte-identical to origin/v1/next, staged by `stage_gpc_h364.sh`) in the H335 docker
    sandbox (no network), as unit `h364-C-gp` with 12 AGG shards × 8 workers.
  - Recipe: Seed F (22 evals), then Evo A K = 6 / P = 48 for **50 generations**, giving `AGG.g50.single` and
    `AGG.g50.CV3@12`.
  - Inputs: superset worker `2a848c8f…`, learners-v1 grammar, H317 H0.1-A screener fold models, and the cross-family bank
    (same-family hits asserted to be 0).
  - SEARCH views are encoded on xbabe2 with the frozen H317 recipe; DOPE `c51a56d0` is read-only, sha checked before and
    after.
- **Library: `lib_oof_h354.py --audit fit`** on xbabe2 (bwrap sandbox, no network), exactly as H357, unit `h364-lib`.
- **Scoring: `phase_c_h364.py score` + `decide`** on xbabe2, with the H317 guard installed and 1 BLAS thread.
  - No stacker is re-solved, so the H357 cross-host Frank-Wolfe issue does not arise.
  - A failed X0 harness check gives `INCOMPLETE`.
- **Hosts and caps.**
  - xbabe0: GP, MemoryMax 100 GiB, docker 96 GiB, 96 cores.
  - xbabe2: library and scoring on cores 38-53 / 102-117, nice 19, library MemoryMax 48 GiB, no swap.
  - Host CPU ≤ 85% and MemAvailable ≥ 20 GiB.

## 4. Second-level design (space fixed here; constants chosen by §4.3 and published as part B)
### 4.1 Grid (144 configurations)
| Constant | Values | Meaning |
| --- | --- | --- |
| c | 0.10, 0.20, 0.35, 0.50 | cap on the total library weight Σβ |
| λ | 0, 0.02, 0.10 | ridge shrinkage toward β = 0 (scale-free, §2) |
| δ | 0, 0.005, 0.02 | acceptance margin in SEARCH log-RMSE ratio, required on both folds |
| τ | ∞, 6, 4, 3 | near-perfect guard: X0 cross-fitted SEARCH utility ≥ τ ⇒ c = 0 |

### 4.2 Development data
The development data is exactly the stored outputs of the two **used** panels:
- **P201b:** H335 Phase T `T_AGG`, seed 335101, `AGG.g50.CV3@12`, with the H354 Phase D classical-library outputs.
- **P201c:** H357 `C_AGG`, seed 357101, with the H357 library outputs.

That is 395 contexts in 201 families, all TRAIN, already AUDIT-scored and reported. The H354/H357 per-family loss
analysis (§1) motivates τ. Its values come from the X0 utilities of the worst contexts (8.37, 7.29, 4.59).

### 4.3 Selection rule (fixed now)
For each configuration:
1. Compute the per-context AUDIT gain u(2L) − u(X0).
2. Average to one value per **family** over that family's dev contexts.
3. Score LCB = mean − 1.96 · sd / √F over the F = 201 families (unclipped).

**Pick the configuration with the largest LCB.** Ties within 1e-6 go to the more conservative configuration: smaller
c, then larger λ, then larger δ, then smaller τ. Part B also records:
- the full grid results;
- the chosen configuration's dev gain, wins/ties/losses and contexts losing more than 0.05;
- the same quantities for τ = ∞.

**No constant may be changed after part B is published.**

## 5. Decision (fixed now)
- **Primary (one test).** The H335 `primary_test(2L, X0)` (`h335_stats.py` `ee3b4543…`) over the 190 P201d families:
  - clipped (±0.5) family-mean contrast T;
  - one-sided sign-flip p with 2^20 Monte Carlo draws;
  - LOFO minimum;
  - family median.
  - **Pass** ⇔ T > 0 ∧ p < 0.025 ∧ LOFO min > 0 ∧ median ≥ 0.
- **Adoption-grade secondary.** The raw H317 contract gate: `robustness_summary` + `contract_advance_gate`, with boot
  seed 335150. Unclipped LCB95 > 0.001, median ≥ 0, ≤ 10% of families harmed by more than 0.01, sign-flip p < 0.025,
  LOFO > 0.
- **Verdicts.**

  | Verdict | When |
  | --- | --- |
  | `XSTACK2L_CONFIRMED_ADOPTION_GRADE` | the primary and the secondary both pass |
  | `XSTACK2L_CONFIRMED` | the primary passes |
  | `XSTACK2L_INFERIOR` | the mirrored primary passes |
  | `XSTACK2L_NOT_CONFIRMED` | otherwise |
  | `INCOMPLETE` | any of the conditions below |

  `INCOMPLETE` conditions: a missing GP record or library output, a failed X0 harness check, an infrastructure failure,
  part B not published before `pull`, or compute not done by **2026-11-05 20:00Z**.
  - A library learner failing on a context is an outcome, not an infrastructure failure. It is excluded on SEARCH; at
    AUDIT, the SEARCH mean replaces it.
  - There is no retry and no partial scoring.
- **Headline (descriptive):**
  - family-mean AUDIT gain and % RMSE cut;
  - families won / tied / lost;
  - contexts losing more than 0.05;
  - share of contexts where the second level is active, and the mean Σβ;
  - **AUDIT-Elo on P201d** over 2L, X0 (= H335 AGG), `2L_noguard` and each classical learner alone. Anchor X0 = 1500,
    family bootstrap, `h335_stats.elo_panel`. This is a TRAIN scale, not the dev91 scale; the dev91 1742.6 is never
    placed beside it.

## 6. Scope
- **The claim is TRAIN-only.**
  - `dev91-audit-v1` and the reference board are not touched.
  - Placing 2L on the fixed board would be a separate owner decision.
- **No change to:** the H335 recipe, the H317 stacker, the screener, the grammar, the library configs, or any
  H335/H354/H357 code.
- **Ops.**
  - The flywheel writer stays STOPPED.
  - No deletes.
  - Closeout: a create-only S3 archive with exact-version GET and member verification.
  - Commits go to `v1/next` via a private index, non-force, with same-agent review receipts.
  - The AGENT_CHAT claim and release are made under flock.

## 7. Part B: frozen second-level constants
Added by part B, before any P201d output is pulled. See `DESIGN.json`.
