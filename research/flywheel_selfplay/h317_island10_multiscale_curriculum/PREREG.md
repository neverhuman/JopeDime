# H317 (`h317_island10_multiscale_curriculum`) — PREREGISTRATION v2

Gate H0 rapid-hypothesis lab (TRAIN-only, held-out AUDIT protocol), LOCK, then ONE locked DL-Val (dev91) shot.

- **Owner:** `cl_fwriter` (Claude Opus 5.5, interactive xbabe2 session; also the flywheel writer since 2026-11-04T11:18Z).
- **Authority:** Jepson's direct instruction in this session, 2026-11-04 ~11:00Z: "execute this plan", meaning
  `~/.gemini/antigravity-cli/brain/4f6d4a8e-3b18-4d7c-8f93-dfc0cc42a86b/implementation_plan.md`
  (sha256 `425f995d27b891520dd626e388a488da59160cb1efcfbbecb3bdefb7c877c6f3`; it calls the wave "H258"). Owner answers in
  the same session: (1) Gate H0 runs TRAIN-only, then one preregistered DL-Val91 evaluation of the locked winner;
  (2) the xbabe3 RTX 3090 is allowed. Then: stop the xbabe1 flywheel writer and become the writer (done 11:18Z; the
  writer stays stopped during H317, so the 4090 belongs to H317); "be very aggressive … brainstorm, red team"; "if you
  identify key learners needed in the grammar … task an agent with building them".
- **Claim:** `~/jope_plan/ns/receipts/H317_claim.txt`; cores `~/jope_plan/ns/receipts/H317_cores.txt`; comms + AGENT_CHAT.
- **Publication:** non-force push to `origin/v1/next` (CI_FREEZE). This PREREG is pushed **before any fit on true targets**.
- **Red-team inputs (before any fit):**
  - statistics/design;
  - leakage/guard (17 bypasses; the patch adopted as guard v2 plus sandbox);
  - method brainstorm (worker grammar inventory; synthetic winner's-curse/stack/bag simulations);
  - prior-wave evidence miner;
  - audit of Gemini's H256/H257 (main, PR #2120);
  - ops/compute.

  Each finding and its disposition is in §6.

## 0. Departures from the source plan, and why

| Plan item | H317 |
| --- | --- |
| Wave id `H258` | `H258` is rd_argon's closed wave (`h258_top4_reveal_baseline`, NO_ADVANCE). Renumbered H317. |
| "~1,100+ new res-01 contexts (>2× data)" | `res-01` = exactly 1,219 DL-Train + 91 DL-Val. TRAIN is the same 1,219 contexts H256 used. The expansion is real only against H257's 525-context packet set. Prior: data-size waves were null (H176 DATA_SATURATED). |
| H257 as the frontier to beat | The H257 audit (main `fa0c58c2b`, read-only) found the following. **H317 does not treat H257 numbers as a target; `H257_REPLICA` is descriptive only.** <br>• DL-Val headline arms are best-of-9…29 on the very 2-fold CV they report. <br>• C1/C2 lose to the trivial 4-eval `TOP4_PICK` (1.252000, H258). <br>• The standalone 23-fit GP output (1.184) sits below H256's 0-eval static pick (1.203). <br>• The +0.124 OOF lift is a warm-start leak. <br>• DL-Val was the development set. |
| Per-arm DL-Val ladder | Owner: Gate H0 TRAIN-only. DL-Val is read once, in Stage V, for locked pipelines only. |
| Search/stack scored on the CV that selects them | **Selection bias.** H317 adds the held-out **AUDIT** protocol (§2.3). Every search/stack decision is made on AUDIT; in-sample SEARCH utility is reported alongside. |
| "Top-32 S3 pool" seeds | **Dropped.** res-01 pools carry no DSL. Each pool is an 18-generation GP history selected on all rows (AUDIT rows included), so own-pool seeds would reintroduce exactly the bias AUDIT removes. No sha→DSL catalog is built. Pool `fitness/gate/logratio` never enter a GP-block decision. |
| Islands by catboost/rf/extra_trees/svr/mlp/blend2/stack2 | Those learners do not exist in the pinned worker's grammar. Islands are defined by the heads the worker has: **I0 tree** (histgbm/histgbm2), **I1 linear** (ridge/ridge_gcv/elasticnet plus the transform grammar), **I2 kernel/local** (ksmooth/gpr/krr/rff/knn), **I3 meta** (in-worker `portfolio` NNLS blends of I0–I2 elites). Missing learners are being built as a separate superset worker (§2.5). |
| Active 2+2 reveals | H258 rejected `ACTIVE_2PLUS2_HURDLE` vs `TOP4_PICK` (0/19 families). Seed D keeps a reveal-conditioned variant only at a matched budget. |
| 384 cores, xbabe0/2/3 | xbabe2 cores 24–27, ≤ 4 numeric threads (PREREG_TEMPLATE cap). xbabe3 3090 + xbabe1 4090 for torch. xbabe0 unused. |
| Frontier-LLM / TabFM Elo ladder | Not recomputed (no AUDIT outputs exist for them; the audit found the FM numbers unverifiable). |
| Rust `.safetensors` export / distilled head | Out of scope; follow-up only after ADVANCE. "Save the gene, not the model" (PREREG_TEMPLATE) applies. |

## 1. Data boundary

- **TRAIN** = `selected.json["train"]` (1,219 contexts / 205 families). Gate H0 opens only:
  - TRAIN tables and pools;
  - `champions.json`;
  - `winners.json` (90 records, all TRAIN);
  - H317's own derived files;
  - the TRAIN prefix of the mixed `h58/cache/{pool_priors,context_features}.npz`, through the H308 `pread` loader only
    (byte ranges asserted equal to the exact TRAIN prefix).
- **Enforcement, two layers.**
  - **(1) Guard v2** (`common_h317.py`), installed at import:
    - realpath + inode refusal of every DL-Val table/pool, the mixed npz and `full-packet.tar`;
    - dev cid 12-prefix / dataset_id / family-hash matching on any hex run;
    - a denylist of DL-Val/holdout-bearing trees with no dev id in their names (JopeDimeMicro
      `assets/benchmark_data`, `~/jope-evidence`, gov-03 custody, flywheel-redteam runs, `~/flywheel/data`,
      `~/scratch`, git `research/flywheel_selfplay/custody/files/**`);
    - a scan of exec argv;
    - a frame check inside `mixed_prefix_read()` (no `np.load` of mixed files);
    - a cross-process refusal log;
    - a pinned one-shot Stage V unlock.
  - **(2) Deny-by-default sandbox** (`sandbox_h317.sh`, bubblewrap) for every Gate H0 process on xbabe2:
    - `/home/ubuntu`, `/tmp`, `/mnt`, `/media`, `/srv`, `/app`, `/review`, `/Users` and `/opt/work` are empty;
    - only the TRAIN view (hard links to the 1,219 TRAIN tables/pools, champions/winners, and `selected.json` with dev
      rows reduced to identifiers, sha `7d64774a…`), the H317 source (ro), `$H317_DATA` (rw, redteam hidden),
      JopeDimeMicro (ro, `benchmark_data` hidden) and `~/.local` are visible;
    - no network.

  Tests: `test_h317_guard.py`, 28/28 outside and 27/27 (+1 skip) inside the sandbox.
- GPU-host processes (xbabe1/xbabe3) run from a TRAIN-only staged copy (sha-verified) under guard v2. The h58 cache
  there is read only via the prefix loader.
- **Always forbidden:** `~/jope-data/h256/**` and `~/jope-data/h257/**` (mixed caches; models selected while reading
  DL-Val; one bundle, `h256_production_full_1310`, is fit and tuned on DL-Val), sealed/holdout/meta_test/fresh-cohort
  and DL-Test.
- Frozen DOPE `c51a56d0…` is never updated (hash checked before and after every neural run).
- **Disclosed pre-prereg reads:**
  - (a) One champion-DSL probe on one non-pilot TRAIN context (native/SEARCH/AUDIT envelopes) to verify worker parity
    and envelope ordering.
  - (b) The leakage red team ran one tree-wide `git grep -l` on `origin/v1/next`. It regex-scanned every blob,
    including the 91 dev table/pool blobs committed under `research/flywheel_selfplay/custody/files/`, and printed only
    file names; no DL-Val content was decoded or shown. A provenance sub-agent of that red team may have done the same.
    The neural builder, before the warning, likewise ran tree-wide `git grep -l 'context-feature-cache'` and
    `git log --all -S…` on `origin/main` and `origin/v1/next` (paths and commit subjects printed only).
  - (c) The GP builder's catalog search (stopped; the files it opened are listed in `DISCLOSURES.md`).
  - (d) Neural builds used permuted targets for timing.
- **DL-Val is a reuse cohort, not fresh confirmation.** r882 is the argmax of ~1,700 dev91 evaluations; ~110 waves have
  read DL-Val; H257 tuned on it; DOPE's pretraining corpus is undocumented. Before the first DL-Val outcome read,
  Stage V runs a content-fingerprint dedupe of DL-Val vs TRAIN tables (row-hash Jaccard of standardized feature rows),
  and reports it.

## 2. Design

### 2.1 Splits and seeds
- Family-grouped 5-fold over TRAIN, seed 317140 (`fold_manifest.json` v2 sha `d682024c…`; fold sizes
  244/244/244/244/243; cid→fold map sha `3b22149d…`).
- **Pilot P96:** 96 contexts from 96 distinct families (20/19/19/19/19 per outer fold), seed 317141. Fallback P48 (its
  nested subset, sha in the manifest), used only if the measured GP cost exceeds the window. The choice is made from
  timing before any pilot outcome and recorded in `LAUNCH.json`.
- Seeds:
  - training: 317101, 317102 (2 per arm);
  - search: **317101 only**, for both the GP block and Stage V (fixed from timing before any pilot outcome, §4);
  - bootstrap/sign-flip: 10,000 draws, RNG 317150 (Gate H0) and 317160 (Stage V).

### 2.2 Neural block (H0.1–H0.3): 0-reveal top-1 pick from the 1,152-candidate pool, 5-fold OOF on all 1,219 TRAIN contexts
- **Metric:** family-equal mean native utility of the top-1 pick per held-out context (−fitness if the 2-fold gate
  passes, else −2.0), averaged over training seeds first. Invalid-pick counts are reported.
- **Regression gate first:** the true per-context TRAIN r882 pick (the r882 adapter's predicted-cost argmin, from a
  pinned prior per-cid result file or a recomputation with the pinned adapter `0084479a…` and the H162 forward; source
  and sha in `LAUNCH.json`) must reproduce the H258 TRAIN anchors: family 1.385355, context 1.423073 (1e-6). This is
  asserted before any training. (Code red team: `argmin base_prior_cost` is the adapter's INPUT prior, not r882. It
  gives 0.779950 / 0.794549 and is not used as R882.)
- Candidate inputs are computable from any DSL (feat412 + proj16 + sem24); r882 priors are not inputs. Context inputs
  are DOPE encodings of view 0 or of the B = 32 four-regime views (plan §3.1). The fold→GPU mapping is fixed for every
  arm and seed (all arms see identical GPU types per fold; mapping in `LAUNCH.json`). All arms within an experiment get
  **equal optimizer steps**.
- **H0.1** (Scale S, Opt1): A = seeded 525-context family-stratified subset × 4 views; B = 1,219 × 16 views;
  C = 1,219 × 32 views + cross-view latent mixup Beta(0.4, 0.4). Baseline A. (C − B confounds view count with mixup;
  this is reported as such.)
- **H0.2** (H0.1 winner, Opt1): S (d=32, L=1) / M (d=96, L=3, SwiGLU + RMSNorm + QK-norm) / L (d=160, L=4, axial,
  DropPath 0.1). Baseline S.
- **H0.3** (H0.2 winner): Opt1 AdamW + cosine (baseline); **Opt1+** = AdamW + WSD + EMA/SWA + curriculum annealing (the
  control that isolates the optimizer); Opt2 = Opt1+ with SAM ρ=0.05; Opt3 = Opt1+ with Muon on 2-D hidden matrices;
  Opt4 = Lion (lr 3–10× below AdamW) + SAM + WSD + SWA. Per-optimizer lr is set from permuted-target stability checks
  only. Parameter drift is logged.
- **Decision rule** (each experiment, sequential):
  - a challenger is eligible iff, vs the experiment's baseline, it passes the **contract gate** with Holm adjustment
    over that experiment's m challengers;
  - the eligible challenger with the highest family-equal mean wins; otherwise the baseline stays;
  - ties never advance.

  Contract gate: Holm-adjusted percentile LCB > +0.001; family median ≥ 0; ≤ 10% of families losing > 0.01;
  one-sided sign-flip p < 0.025 (Holm-adjusted); leave-one-family-out minimum gain > 0.

  Prior evidence predicts nulls (H176, H220, H118, H180–H190; optimizer waves within ±0.002). The baseline is meant to
  be sticky.
- **MDE** (205 families, per-family SD of a near-null contrast ≈ 0.15–0.18 from H258 OOF): ≈ 0.029–0.035 at 80% power.

### 2.3 AUDIT protocol (every search/stack measurement)
- **Split.** Per context, AUDIT = 30% of training rows, target-quantile stratified (blocks of 10, 3 seeded picks, seed
  stream 317200 ⊕ cid). SEARCH = the other 70%, keeping their native 2-fold labels.
- **SEARCH-only selection.** Within a context, every pipeline chooses its single output (tree or stack) from SEARCH
  2-fold results only. AUDIT is evaluated exactly once per declared output, after it is fixed, and is never compared
  within a context. Code asserts AUDIT requests per (pipeline, cid, seed) == declared outputs. AUDIT drives
  **arm-level** decisions across contexts only.
- **AUDIT utility** = −log(RMSE_AUDIT(program fit on all SEARCH rows) ÷ RMSE_AUDIT(SEARCH-target mean)). It comes from
  the pinned worker on [fold 0 = SEARCH, fold 1 = AUDIT], record 1. Verified 480/480 rows, baseline equal to 1e-14. The
  worker does not transduce AUDIT rows: record-1 predictions were bit-identical under replaced AUDIT targets and
  rescaled AUDIT features, 41 DSLs.
- **Stack AUDIT predictions** = Σ wᵢ·(member i's AUDIT predictions); weights are fit on SEARCH out-of-fold only.
- **Failure policy (AUDIT-outcome-independent).** An output counts as completed iff the worker exits 0 with finite
  predictions; its utility stays continuous (no −2.0 cliff on the AUDIT scale). A crashed or non-finite output, or a
  stack member, gets AUDIT predictions := the SEARCH-target mean (a single output scores 0.0). Members are never
  dropped or reweighted on AUDIT. Invalid counts are a reported secondary. ADVANCE requires
  invalid(H317_LOCKED) ≤ invalid(comparator) + 1.
- **Encodings.** The screener's context encoding for pilot and DL-Val contexts is built from SEARCH rows only (view 0 =
  SEARCH table; perturbed views from SEARCH rows).
- **Reliability diagnostic.** Split each AUDIT slice into two y-stratified halves from the same fits; report the
  across-half correlation of paired pipeline differences.
- **Scale.** AUDIT and native utilities are on different scales; only paired same-row differences are interpreted.
  Levels are never compared with H257 numbers.

### 2.4 GP block (H0.4–H0.6) on P96 × 1 search seed (317101), xbabe2 cores 24–27 (3 Rust workers + 1 screener thread), in the sandbox
- **Screener.** The **H0.1-winning** neural config's OOF model for the pilot context's outer fold. The training cid set
  sha is stored, and loading asserts the target context and family are absent. Every GP arm shares it. If H0.1 is not
  complete by 15:30Z, the H0.1-A (Scale S) OOF screener is used (disclosed in `LAUNCH.json`).
- **Bank.** The 1,219 TRAIN champions + TRAIN `winners.json`, excluding every record with
  `family_id == family(target)` (this includes the target's own champion, found on all rows). Same-dataset_id records
  are excluded too. Asserted; `same_family_bank_hits` must be 0.
- **Shared parameters.** Bank, islands and stack configs are frozen before the run. No state crosses contexts.
- **H0.4 seeding.** A = H257 baseline (8 bank seeds); B = 16, 4 per view regime; C = 16 across 4 islands from one
  [32 × ~160] pass; D = 4 live reveals → screener residual correction → 20 seeds; E = 8 worker registry presets (`gpr`,
  `krr`, `rff_ridge`, `knn_loo`, bagged `histgbm2`, `gam_tbe`, …; never used by any bank champion) interleaved with 8 bank
  seeds; F = E plus the new learners (only if §2.5 delivers before launch).
  - **Primary:** AUDIT utility of the best-by-SEARCH seed among the first 8 live evals in each scheme's declared order
    (D: its 4 reveals + first 4 seeds; E/F: preset/bank interleaved).
  - **Decision:** contract gate vs A with Holm over m = number of available challengers (4, or 5 with F), fixed in
    `LAUNCH.json` before any pilot outcome.
- **H0.5 evolution** (from the adopted seeds): Evo A panmictic vs Evo B 4-island MAP-Elites + ε-lexicase (cases:
  fold-0 lr, fold-1 lr, tail lr, complexity).
  - Migration at gens 3/6/9 builds an I3 portfolio from the I0–I2 elites.
  - Coordinate line-search at gens 8–10 is **inside** the budget.
  - Both arms: 48 screened proposals and exactly 6 live SEARCH evals per generation (asserted), **10 generations**
    (locked). Gens 0/3/6/10 AUDIT curves are descriptive.
  - Adopt Evo B iff it passes the contract gate vs Evo A at gen 10.
- **H0.6 stacking** over the adopted arm's evaluated trees. Configs:
  - CV1: H257 Top-3 cross-fitted MSE simplex.
  - CV2: Top-N Frank–Wolfe with L2 prior shrinkage.
  - CV3: 5-subfold S1-direct Frank–Wolfe with ρ < 0.985 pruning and island quota.
  - CV4: hierarchical island meta-stack.
  - CV5: `PORTFOLIO_NNLS`, the in-worker `(portfolio (member …)… (blend kind=nnls folds=5 repeats=auto guard=clip))`
    over the best-by-SEARCH program of each head class; one deployable program, 1 SEARCH + 1 AUDIT eval.
  - N ∈ {3, 5, 7, 10, 12} where applicable.

  **Selection is leave-one-family-out cross-validated:** each pilot context gets the config that maximizes family-equal
  AUDIT mean over the other 95 families. **One test:** CV-selected stack − single best tree, contract gate. The config
  locked for Stage V maximizes over all 96.
- **Secondaries (report-only):**
  - `PORTFOLIO_NNLS` − single best (direct);
  - `BAG_FINAL` (histgbm → bagged histgbm2 refit);
  - `SEL_MEANLR` (finalists by mean logratio);
  - `NOISE_PROBE` (re-evaluate the 8 Gen-0 seeds under 2 extra random 2-fold splits of SEARCH: candidate-specific
    split SD vs shared fold offset);
  - per-head win counts;
  - shrunk estimates ĝ − E[max_K]·SE for every adopted step.
- **MDE (P96 × 1):** ≈ 0.044 (context SD 0.08, seed SD 0.13; stats red team: contexts buy more power than seeds, and
  P96×1 beats P48×2's 0.049). H257's stack effect (+0.0076) is below it.

### 2.5 Conditional arm: superset worker with new learners
A learner-build agent is implementing the missing regression learners in Rust (gap analysis first; candidates include
randomized deep-tree ensembles and an early-stopped MLP) as a **separate** worker binary. The pinned `bootstrap.s1main`
(sha `e3be9386…`) is never modified. The new worker is admitted to H317 only if both hold before the GP block launches,
and no later than 16:30Z:
- (a) ≥ 300 existing programs × ≥ 5 non-pilot TRAIN contexts give bit-identical `--fit-s1` outputs vs the pinned worker
  (superset parity);
- (b) its unit tests pass.

If admitted, its sha is recorded in `LAUNCH.json` and seed scheme F and the new heads (in their island) are evaluated
with it; every other arm uses the pinned worker. Parity makes shared programs identical. Otherwise the learners are a
follow-up deliverable (patch + receipts) and F is absent.

### 2.6 LOCK and Stage V (one shot on DL-Val91)
- **Run condition.** Stage V runs only if at least one Gate H0 adoption occurred (otherwise DL-Val stays unread and the
  verdict is `NO_ADOPTION_STAGE_V_SKIPPED`).
- **LOCK.** `LOCK.json` (fsynced) records:
  - the neural winners, retrained on all 1,219 TRAIN contexts with the locked budget;
  - the GP config;
  - every source/worker/model sha (`code_sha256`).

  Its sha256 goes to the create-once receipt `~/jope_plan/ns/receipts/H317_LOCK_sha256.txt` and to comms **before** the
  pinned one-shot unlock.
- **One shot.** Stage V runs once on 91 contexts / 19 families with search seed 317101. Two seeds × 7 pipelines would
  not fit the window on 3 workers (≈ 6 h). The seed component therefore stays in the per-family variance, and the MDE
  rises from ≈ 0.029 to ≈ 0.04.
  No retry after any DL-Val outcome is read. An infrastructure failure before the first outcome may be relaunched once,
  disclosed. No state crosses contexts (contexts within a family share 60–75% of rows): no bank growth, no shared memo,
  no cross-context corrector.
- **Order.** Dedupe check (§1); then the R882 Native91 regression.
  - R882 per-context reference = H258's pinned `inputs/r882/best_dev_checkpoint.json` (sha `30853cad78d8…`; opened
    only after unlock; before the prereg only its keys and context count were read).
  - Check: recompute each context's utility from its DL-Val pool and the reference's selected program; it must match
    the reference per-context values 91/91 bit-exactly, with mean `1.1377941898164525`. A mismatch is
    REGRESSION_FAIL and stops the stage.
- **V-native (secondary).** Locked neural 0-reveal top-1 pick − R882 (both 0 reveals), contract gate. `TOP4_PICK`
  (recomputed, H258 reference 1.252000) is descriptive.
- **V-AUDIT pipelines** (AUDIT protocol on every DL-Val context; live evals recorded per pipeline):
  - `H317_LOCKED`;
  - `EVOA_MATCHED`: adopted seeds → panmictic Evo A for the same generations and live evals as H317_LOCKED → CV1
    Top-3; |Δevals| ≤ 2 asserted;
  - `SCREEN_TOPK_PICK`: screen bank + grammar proposals once, live-evaluate the top-K by screener, K = H317_LOCKED's
    live evals, output best-by-SEARCH;
  - `BANK_NN8_PICK` and `BANK_NN8_PORTFOLIO`;
  - `BANK_NN1`;
  - `H257_REPLICA` (Seed A 8 → 3 gens × 5 children → CV1 Top-3, as specified by the audit; descriptive).
- **Primary (intersection–union).** `H317_LOCKED` passes the contract gate vs **both** budget-matched controls,
  `EVOA_MATCHED` and `SCREEN_TOPK_PICK`, with paired whole-family bootstrap and sign-flip over 19 families.
- **ADVANCE** iff the primary holds AND invalid(H317_LOCKED) ≤ invalid(comparator) + 1 AND the point gain vs
  `BANK_NN8_PORTFOLIO` > 0. Otherwise:
  - **NO_ADVANCE-INFERIOR** (UCB95 < 0 vs either control);
  - **NO_ADVANCE-UNDETECTED** (CI ⊂ (−0.03, +0.03));
  - **INCONCLUSIVE**.

  Ties never advance. **MDE ≈ 0.029** (19 families, per-family SD 0.045): smaller true gains are *undetected*, not
  *absent*.
- **Also reported:** W/T/L (tol 1e-4), family median, drop-top-5/10, harmed share, LOFO, in-sample SEARCH utilities,
  invalid counts, eval counts, wall time, and family-grouped MM Bradley–Terry Elo within the fixed Stage V panel
  ("reference = R882 slot0 (dev91)" for V-native; "reference = EVOA_MATCHED (dev91 AUDIT)" for V-AUDIT).

## 3. Resources, retention, deadlines
- **xbabe2.** Cores 24–27, 4 single-threaded workers, nice 19. `systemd-run --user` with MemoryMax 8G, MemorySwapMax 0,
  TasksMax 64, CPUQuota 400%, RuntimeMaxSec to the stage deadline. All `*_NUM_THREADS=1`. Inside `sandbox_h317.sh`.
  Pause if MemAvailable < 20 GiB or host CPU > 85%. H316/H318 cores are untouched.
- **xbabe1 4090 / xbabe3 3090.** One H317 process each, nice 10, numeric threads ≤ 4 (CUDA exempt). xbabe3 ≤ 600% CPU
  (it hosts CI runners) and **≤ 3 GB H317 data** (≈ 14 GiB below its 93.8% disk guard). RSS ≤ 16 GB on xbabe1.
- **Retention.** Save the gene, not the model: compact per-context predictions/scores (npz/jsonl). Fit objects are kept
  only for the H0.1-winner OOF screeners and the locked Stage V model (sha recorded). Archive to S3
  `experiments/rd_breakthrough_20261030/H317/` (exact-version GET + sha verify). Local scratch deletion at close only
  under the PREREG_TEMPLATE retention receipt, with owner confirmation. No models in git.
- **Deadlines (UTC, 2026-11-04):**
  - PREREG push before the first true-target fit;
  - Gate H0 compute stop 20:00Z;
  - LOCK by 20:30Z;
  - Stage V compute stop 23:00Z;
  - closeout 23:59Z.

## 4. Budgets (pinned from permuted-target timing; no true-target fit preceded this section)
- **Neural (H0.1–H0.3), every arm.** Training:
  - **3,000 optimizer steps** of 16 contexts each (≈ 49 epoch-equivalents of a 975-context training fold; arm A's
    ≈ 421-context subset ≈ 114);
  - **slate 128** per context per step (the training context's 48 best candidates + 80 uniform others; prediction on
    a held-out context always scores all 1,152 candidates);
  - **2 training seeds** (317101, 317102);
  - **fold→GPU: folds 0–2 → xbabe1 RTX 4090, folds 3–4 → xbabe3 RTX 3090**, identical for every arm and seed (`run_arm`
    refuses other folds on a host);
  - 2 numeric threads per process, ≤ 4 per host;
  - fold models kept only for H0.1 arms (`--keep-models`, sha recorded) and the LOCK model.

  Measured wall per fold-run: S ≤ 1 min; M opt3 2.4 min; L opt1p 2.6 min; L opt4 6.1 min (worst). Stage wall: H0.1
  ≈ 10 min, H0.2 ≈ 21 min, H0.3 ≤ 116 min (if L wins H0.2). Parameter counts: S 178,520 / M 1,422,340 / L 4,220,164.
- **Pinned implementation choices** (from the build; plan ambiguities resolved before outcomes):
  - Arm A subset seed = `stream_seed("H317","armA525",317140)`.
  - Synthetic-column count = the plan's per-view fraction inside the clamp max(4, min(20, ·)).
  - View 15 masks the column with the largest |corr(x, y)|.
  - Opt3 = Moonlight-scaled Muon on 2-D hidden matrices (AdamW lr 8e-4 / wd 2e-3 carried over; drift 0.19 vs AdamW
    0.13 on permuted targets). The unscaled Muon (`opt3k`) is not an arm.
  - Opt1+ / Opt2 / Opt3 / Opt4 share SWA as the averaging scheme, so their contrasts isolate one component each.
  - Lion lr 1.5e-4 / wd 1e-2.
  - sem24 = the canonical JopeDimeMicro sem24 (Gemini's `compute_sem24_numpy` differs, max diff 0.9996, so it is not
    used).
- **H0.3 decision.** Challengers {Opt1+, Opt2, Opt3, Opt4} vs **Opt1** with Holm m = 4 (the governing rule). The
  isolating contrasts Opt2/3/4 − Opt1+ are reported descriptively.
- **Not built (plan items dropped before outcomes):** the 2+2 trajectory head, the distilled d=32 export, LLRD,
  cross-family V-REx, family-batched SAM, column-landmark tokens.
- **Featurization facts.**
  - The champion row was found in 1,208 of 1,219 pools, and `featurize_dsl_412` is bit-exact with it on all 1,208.
  - 11 TRAIN champions are absent from their own pool.
  - View-0 DOPE re-encoding equals the h58 cache on 1,219 / 1,219 contexts (max abs diff 0.0); all views were
    CPU-encoded (provenance in `views_provenance.json`).
- **GP (H0.4–H0.6).**
  - xbabe2 cores 24–27: **3 Rust workers + 1 parent screener thread** (4 numeric threads in total; the screener is the
    H0.1-winning Scale-S fold model, 0.10 s per [32 × 160] call).
  - **Measured on 3 non-pilot contexts** (6, 20 and 60 columns, sandbox, test-only screener):
    - 0 failures and 0 timeouts in 1,257 worker runs; 0 operator failures in 209 children (0 of 41 on a 365-column
      context);
    - mean eval 1.41 s, p95 5.6 s;
    - one H0.4+H0.5+H0.6 job (context × seed) costs 128–530 s of eval CPU, mean 272 s.

    Projected on 4 workers: P96×2 4.0–4.8 h; P96×1 and P48×2 2.0–2.4 h each; on 3 workers, × 4/3.
  - **Decision (before any pilot outcome):** **P96 × 1 seed**, about 2.7–3.2 h on 3 workers. It fits the Gate H0
    window in parallel with H0.2/H0.3, and its MDE beats P48×2.
  - **Cost rules applied to every arm alike:**
    - no `stack` head in generated proposals;
    - portfolio members ≤ 3, cheap heads only;
    - heavy heads only when the post-transform width is ≤ 256;
    - gpr m = 300 on tables wider than 100 columns;
    - histgbm2 rounds×bags capped;
    - `--max-expanded-width 1000`;
    - timeouts: 60 s, ×4 for portfolios, ×2 for AUDIT evals.
  - **Grammar facts:**
    - the worker demands canonical DSL text (sorted params, Rust-style floats) and at most 9 stages;
    - the only combinator is a top-level `portfolio` (1–12 members, NNLS blend, no nesting);
    - there is no MLP, forest, SVR, PLS, catboost or kNN head (`knn_loo` = `ksmooth k=32`).
  - **Screener coverage:** feat412 covers 23 ops, so it cannot see histgbm2/gpr/krr/stack/portfolio/target or the
    newer transforms. Seed E's presets are 4/8 featurized, so E uses its declared fixed interleaved order.
  - **Implementation deviations from the plan text (pinned now):**
    - Deployed stack weights are fit on the pooled out-of-fold predictions of both SEARCH folds. Cross-fitted weights
      are used only for the in-sample SEARCH S1.
    - CV2–CV4 normalize on the fitting fold only. CV1 is an exact H257 port, including its one-scalar dependence.
    - CV3 uses no bootstrap resamples, and Frank–Wolfe with grid line search instead of L-BFGS-B.
    - A config that cannot be built (fewer than 2 passing members) scores as the single best tree.
    - Pool-candidate seed roles are replaced by island consensus.
    - NOISE_PROBE uses Seed A's 8 seeds.
    - Seed A neighbours use lm48_z[:12] of view 0; BANK_NN1/NN8 use z-scored dope629. Bank contexts are encoded from
      their full-table view 0, targets from their SEARCH view 0.
    - The H257 operators are the EditZero kinds plus portfolio member exchange.
  - Every pilot-size, seed and worker choice is written to `LAUNCH_GP.json` before the first pilot eval.

## 5. Outputs
`metrics.json` (< 50 KB, `jq -c`), README (hypothesis, setup, outcome, verdict, next step), compact per-context panels on
S3, one row each in `docs/research/waves/INDEX.jsonl` and `overnight-hwave-index.md`, `DISCLOSURES.md`.

## 6. Red-team findings and disposition
Statistics: BLOCKER-1 budgets (→ §4 before fit), BLOCKER-2 bank leak (→ same-family exclusion, asserted), BLOCKER-3
Stage V R882 path (→ pinned `best_dev_checkpoint.json` reference, post-unlock; the earlier plan to use
`base_prior_cost` was wrong, see the neural code red team below), HIGH-1 budget confound (→ EVOA_MATCHED +
SCREEN_TOPK_PICK primary), HIGH-2 power (→ MDE stated, three-way readout, both seeds), HIGH-3 gate-level winner's curse
(→ LOFO-CV stack selection, Holm, shrunk estimates), HIGH-4 within-context AUDIT use (→ SEARCH-only selection, asserted),
HIGH-5 −2.0 dominance (→ continuous AUDIT utility, crash → SEARCH mean), MEDIUM-1 pilot (→ P96), MEDIUM-2 g-rule (→ g = 10),
MEDIUM-3 budgets (→ inside 48/6; D/E/F first-8 rules), MEDIUM-4 encodings (→ SEARCH-row views), MEDIUM-5 robustness
(→ sign-flip + LOFO in the gate), MEDIUM-6 screener mismatch (→ disclosed; a post-LOCK pilot consistency rerun of the
adopted GP config with the locked screener is descriptive), LOW-1…4 (→ reliability diagnostic, scale note, BANK_NN1
metric = screener view-0 distance, fallback schedule). Leakage: BLOCKER-1 (→ guard v2 + sandbox), HIGH-2…5 (→ auto
install, pinned unlock, envelope-keyed memo, no own-pool seeds), MEDIUM-6…10 (→ outcome-independent failure policy,
cross-process log, loader frame check, no cross-context state, reuse-cohort disclosure + dedupe), LOW-11…16 (→ P96
manifest, redacted selected.json, sandbox, screener training-set sha, staging re-verify, single SEARCH-view recipe).
Evidence miner: H0.3 confound (→ Opt1+), contract gate (→ adopted), thread cap (→ 4), TRAIN regression anchors
(→ §2.2). Brainstorm: registry seeds (→ E), head islands (→ §0/§2.4), PORTFOLIO_NNLS (→ CV5), BAG_FINAL/SEL_MEANLR/
NOISE_PROBE (→ secondaries), BANK_NN8 references (→ Stage V). Gemini audit: H257 claims not used as targets;
`H257_REPLICA` spec adopted as descriptive.
Neural code red team (pre-launch, NO-GO until patched): BLOCKER-1 `decide` used plain LCB>0 (→ contract gate + Holm
implemented and tested), BLOCKER-2 "R882" was argmin of the adapter's input prior (→ true r882 pick, anchors asserted),
HIGH-3 coverage (→ 1,219/205/5-fold/2-seed assertions), MEDIUM-4 equal steps (→ config budgets asserted identical),
MEDIUM-5 SAM passes on different random draws (→ shared draws per step), LOW-6…8 (→ float64 anchors, float16-matched
screener features, arm-independent (seed, fold) seeds for pairing). Launch only after the patched tests pass.
