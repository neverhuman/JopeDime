# H357 (`h357_xstack_cl_confirm`): XSTACK-CL confirmation on the fresh TRAIN panel P201c — PREREG

- **Owner:** `cl_fwriter` (Claude Opus 5.5, xbabe2). Claim: `ns/receipts/H357_claim.txt` (0444, sha256 `562075dc…`).
- **Authority:** Jepson, 2026-11-05 ~09:05Z: start the next flywheel research wave and run a smarter training concept
  that maximizes Elo, using TRAIN/P201b ideas only. Never fit, train, tune or leak labels from `dev91-audit-v1`. Work
  on `v1/next`. H357 is the confirmation step of the concept chosen for that instruction, XSTACK (H354).
- **dev91 is not read.** No DL-Val91 / dev91 / DL-Test / sealed / holdout path is opened or mounted.
- This PREREG and its code are published to `origin/v1/next` **before** any P201c view, GP run, fit or score.
  `run_c_h357.sh` refuses every step unless the staged PREREG is byte-identical to the one on `origin/v1/next`.

## 1. Why this wave
H354 closed `XSTACK_NOT_ADVANCED` (`f7530131f0`), after Phase D on TRAIN P201b (201 families).
- Its pre-registered 1-SE arm rule picked X_LLM, and X_LLM's primary missed α (p = 0.040).
- **X_CL** (GP + the 4 classical learners) was descriptively the best arm:
  - +0.0093 family-mean AUDIT u, p = 1e-4;
  - wins 132 of 201 families;
  - 3 contexts lose more than 0.05;
  - no sign of the LLM members' AUDIT-refit blow-ups.
- **X_CL is chosen here after seeing those P201b results.** P201b is therefore exploratory for this hypothesis. This
  wave's only claim comes from P201c, which nothing has searched, fit or scored.

## 2. Hypothesis and arms
- **Hypothesis.** On fresh TRAIN contexts, X_CL beats X0 in family-mean AUDIT utility.
- **X0** = the locked H335 recipe output `AGG.g50.CV3@12`, LOCK `4d58dd37` (H335 BETTER_MODEL, dev91 AUDIT-Elo 1742.6).
- **X_CL** = X0's stack members plus `catboost.v1`, `lightgbm.v1`, `xgboost.v1` and `random_forest.v1`.
  - The classical learners use the reference-board registry configs, through `score_reference.fit_predict` (pins
    `169cb04e…` / `20d100c4…`).
  - Stacker: the frozen H317 `weights_s1_direct`, SEARCH-only. Library members join only if they pass the S1 gate on
    SEARCH, with `C_LIB = 0`.
  - Scoring: AUDIT once per arm, after the weights are fixed.
  - Code: `h354_core.stack_arm` / `audit_arm` at `f7530131f0`, unchanged.
- **Fixed.** Nothing about the design may be tuned. There are no other arms and no hyperparameter search.

## 3. Panel and procedure
- **P201c.**
  - Definition: the third seeded context (`h335_core._ctx_order`) of every TRAIN family with ≥ 3 contexts.
  - Size: 194 contexts in 194 families; cids sha256 `36fbf6a92204487a31f2533905a57393e9c317fa7484ab7cbde5026e34468f6d`.
  - Disjoint from P205 and P201b, which is asserted.
  - What was computed before this PREREG: the panel rule alone, which read only the TRAIN row index (context id,
    family id, dataset id). No table content, view, GP, fit or score exists for P201c.
- **GP: `gp_c_h357.py` on xbabe0.** The frozen H335 code, byte-identical to origin/v1/next, is staged with this
  driver by `stage_gpc_h357.sh`. It runs in the H335 docker sandbox (no network) on 96 cores, 12 AGG shards × 8 workers.
  - It calls only `h335_lab.build_rt` (the Phase T TRAIN runtime) and `run_units` → `unit_agg`:
    - Seed F (22 evals);
    - then Evo A, K = 6 live / P = 48 screened, for **50 generations**;
    - outputs `AGG.g50.single` and `AGG.g50.CV3@12`.
  - Inputs:
    - the superset worker `2a848c8f…`;
    - the learners-v1 grammar;
    - the H317 H0.1-A screener fold models (each context uses the fold without its family);
    - the cross-family bank, with same-family hits asserted to be 0.
  - Search seed **357101**, one per context.
  - SEARCH views are encoded first on xbabe2 with the frozen H317 recipe. DOPE `c51a56d0` is loaded read-only, and its
    sha is checked before and after.
- **Library: `lib_oof_h354.py --audit fit`** on xbabe2, in the bwrap sandbox (no network; only the inputs and the
  output dir visible).
  - SEARCH OOF on the GP's own SEARCH split: p0 is fit on fold 1, p1 on fold 0.
  - AUDIT: one fit on all SEARCH rows → AUDIT rows, using the GP's own AUDIT envelope in train-only form.
- **Scoring: `phase_c_h357.py`.**
  - `arms` runs the **X0 harness check**: the frozen stacker must reproduce the stored weights and AUDIT utility of
    `AGG.g50.CV3@12` (|Δ| ≤ 1e-9) on every context. Otherwise the wave is `INCOMPLETE`.
  - It then scores X_CL.
- **Hosts and caps.**
  - xbabe0: GP `h357-C-gp`, MemoryMax 100 GiB, docker 96 GiB.
  - xbabe2: library `h357-lib` on cores 38-53 / 102-117, nice 19, MemoryMax 48 GiB, no swap.
  - Host CPU ≤ 85% and MemAvailable ≥ 20 GiB.

### 3.1 Amendment 1 (implementation only; no rule changed) — AUDIT outputs for every X_CL member
**What happened.** The first `run_c_h357.sh score` (11:01Z) stopped in `arms` with a `KeyError` inside
`stacking_h317.stack_audit_utility`, before anything was written or printed. No AUDIT value or utility was seen.

**The cause.**
- The GP AUDIT-evaluates only an output's *declared* members (`StackSpec.declared()`, w_audit > 1e-9).
- P201c ran a single checkpoint (g50), so the only AUDIT records are those of X0's non-zero-weight members and of the
  single-best output.
- X_CL's SEARCH-only weights can give weight to a GP member that X0 left at zero, and that member has no AUDIT
  prediction.
- In H354 Phase D this did not arise: the Phase T memo held AUDIT records from 5 checkpoints × 2 outputs, and Phase D
  ran without this error.

**The fix.** §2 requires that "AUDIT [be scored] once per arm, after the weights are fixed", and X_CL's AUDIT
prediction needs every declared member's AUDIT output.
- **`phase_c_h357.py audit-request`** (xbabe2) lists each context's declared X_CL GP members that have no AUDIT record.
  The list depends only on SEARCH-derived weights; no AUDIT value is read.
- **`gp_c_h357.py audit-extra`** (xbabe0, `h357-C-auditx`, the same docker sandbox) evaluates exactly those members.
  It uses:
  - the frozen GP engine, superset worker, CPU-limited runner and 60 s timeout of the GP run;
  - one `AuditLedger` request per context.

  It writes into its own run dir (`runs/H357_C_xbabe0_auditx`), so the GP run dir is never modified.
- **`arms`** merges those records. Each must be on the GP's own AUDIT envelope (sha256 equality). `arms` refuses
  (harness failure) if any declared X_CL member still lacks an AUDIT record.
- **Unchanged:** a member whose extra AUDIT evaluation fails takes the SEARCH mean (the H317 rule, as for any member).
  The weights, arms, panel, statistics and verdicts are as above.

### 3.2 Amendment 2 (implementation only; no rule or tolerance changed) — score on the GP's host
**What happened.** After Amendment 1, `arms` ran on xbabe2 (11:14Z). Its X0 harness check failed on 1 of 194 contexts:
`53e44260`, a 2-member stack, |Δw| = 6.1e-5 and |Δu| = 3.0e-7 against the 1e-9 tolerance. The step then stopped.
That attempt's outputs are kept under `attempt1_xbabe2_unread/`, and **its X_CL values were never read**. Only the
`x0_check` fields were inspected.

**The diagnosis.** It used X0 only, with no library and no X_CL.
- **xbabe2:** the same recompute gives the same deviation with 1 BLAS thread as with the default.
- **xbabe0** (the host and docker sandbox that computed the stored stacks): it is **exact**, |Δw| = |Δu| = 0.0.
- **Conclusion.** The frozen Frank-Wolfe stacker is not bit-reproducible across CPUs on near-degenerate stacks. In
  H354 Phase D the stored Phase T stacks happened to reproduce exactly on xbabe2.

**The fix.** `arms` and `decide` run on xbabe0, in the same H335 docker sandbox. Unit `h357-C-score`, started by
`run_c_h357.sh stagescore` / `score0` / `pullscore`.
- The inputs are the sha-checked H354/H357 scoring code and the 776 library output files.
- The stacker, the 1e-9 X0 tolerance, the arms, the statistics and the verdict map are unchanged.
- An X0 check failure there still gives `INCOMPLETE`.

### 3.3 Fix to Amendment 2 (implementation only; no rule or tolerance changed) — the statistics module path
**What happened.** The first `h357-C-score` run (11:19–11:20Z) completed `arms`. Its X0 harness check was exact on
all 194 contexts (worst |Δw| = |Δu| = 0.0). `decide` then stopped before computing anything: the H317 guard refused to
open `h335_stats.py` at the H335 evidence-tree path, where the sandbox binds the staged code. That run's
`arms.jsonl` and `x0_check.json` stay in `h357/score` on xbabe0. **No X_CL value, contrast or verdict was printed or
read**; the log holds only the start line, the `arms` X0-check line and the guard traceback.

**The fix.** `run_score_xbabe0.sh` scores into a new create-once dir, `h357/score2`, rerunning `arms` and `decide` in
full. It loads `h335_stats.py` from a copy in that dir, the same file the sha pin in `h354_core` (`ee3b4543…`)
already verifies. `pullscore` also pulls the first run's two files and reports whether they are byte-identical to
`score2`'s, as a determinism check; their values are not read. Everything else in §3.2 and §4 is unchanged.

## 4. Decision (fixed now)
- **Primary (one test).** The H335 `primary_test(X_CL, X0)` (`h335_stats.py` `ee3b4543…`) over the 194 families:
  - clipped (±0.5) family-mean contrast T;
  - one-sided sign-flip p with 2^20 Monte Carlo draws (F > 24);
  - LOFO minimum;
  - family median.
  - **Pass** ⇔ T > 0 ∧ p < 0.025 ∧ LOFO min > 0 ∧ median ≥ 0.
- **Adoption-grade secondary.** The raw H317 contract gate: `robustness_summary` + `contract_advance_gate`, with
  boot seed 335150 as in H335. Unclipped LCB95 > 0.001, median ≥ 0, ≤ 10% of families harmed by more than 0.01,
  sign-flip p < 0.025, LOFO > 0.
- **Verdicts.**

  | Verdict | When |
  | --- | --- |
  | `XSTACK_CL_CONFIRMED_ADOPTION_GRADE` | the primary and the secondary both pass |
  | `XSTACK_CL_CONFIRMED` | the primary passes |
  | `XSTACK_CL_INFERIOR` | the mirrored primary passes |
  | `XSTACK_CL_NOT_CONFIRMED` | otherwise |
  | `INCOMPLETE` | a missing GP record, a failed X0 harness check, an infrastructure failure, or compute not done by **2026-11-05 20:00Z** |

  A library learner that fails on a context is an outcome, not an infrastructure failure. It is excluded on SEARCH; at
  AUDIT, the SEARCH mean replaces it (the H317 rule). There is no retry, no partial scoring and no cut cell.
- **Headline (descriptive):**
  - family-mean AUDIT gain and % RMSE cut;
  - families won / tied / lost;
  - contexts losing more than 0.05;
  - library weight mass;
  - AUDIT-Elo on P201c over X_CL, X0 and each classical learner alone (anchor X0 = 1500, family bootstrap). This is
    TRAIN, not the dev91 scale.

## 5. Scope
- **The claim is TRAIN-only:** whether the classical library adds to the H335 GP stack on fresh TRAIN contexts.
  - Placing XSTACK-CL on the fixed `dev91-audit-v1` board would be one new once-scored entrant. That is a separate
    owner decision and is not part of H357.
  - The reference board is not changed.
- **No change to:** the H335 recipe, the H317 stacker, the screener, the grammar, the library configs or any H354
  code.
- **Ops.**
  - The flywheel writer stays STOPPED.
  - No deletes; all work files are kept.
  - Closeout: a create-only S3 archive with exact-version GET and member verification.
  - Commits go to `v1/next` via a private index, non-force, with same-agent review receipts.
  - The AGENT_CHAT claim and release are made under flock.
