# H335 (`h335_aggressive_locked_elo`) — PREREGISTRATION

A locked, aggressive-inference GP pipeline is trained and calibrated on TRAIN, LOCKed, then evaluated ONCE on DL-Val91
under the selection-controlled AUDIT protocol, against the incumbent recipe and a budget-matched control. A fair
AUDIT-Elo ladder is reported.

- **Owner:** `cl_fwriter` (Claude Opus 5.5, xbabe2 interactive session; also the xbabe1 flywheel writer, which stays
  STOPPED).
- **Authority:** Jepson, this session, 2026-11-04:
  - ~19:2xZ: "Please continue working forward to help us conclude if the new research wave produced a better model,
    and please take your more aggressive inference (what were we doing in the train? Batches? generations?)";
  - ~21:1xZ: approval of the plan `~/.claude/plans/validated-crunching-truffle.md` (train → lock → one dev91 shot →
    report; no owner touchpoint between phases);
  - on the power-gate question: "why would you do anything to predict? You should just be training and testing and
    doing research with the highest value", so there is no power gate.
  - The plan calls the wave H331, but rd_astra claimed H331 at 19:52Z (H332–H334 are also taken), so it is renumbered
    **H335**.
- **Claim:** `~/jope_plan/ns/receipts/H335_claim.txt` + AGENT_CHAT.
- **Publication:** a non-force push to `origin/v1/next` (CI_FREEZE) **before any fit in this wave**.
- **Predecessor:** H317, closed `NO_ADOPTION_STAGE_V_SKIPPED` at origin/v1/next 97a1ffce. Its code is imported
  **unmodified** and sha-pinned (`h335_core.H317_PINS` == the 97a1ffce blobs == the xbabe0 stage).

## 0. Question, and what changed since H317
- **The question.** Did the H317 cycle produce a better model? In H317 no lever passed the contract gate (mostly the
  ≤10%-families-harmed rule), so DL-Val91 was never read and no Elo exists.
- **TRAIN evidence that motivates the locked pipeline** (H317, AUDIT, family-equal, P205 × 2 seeds; descriptive):
  - Seed F (the superset worker's new heads xtrees/rforest/oblivious + registry presets + bank, 22 evals) alone:
    1.917.
  - Evo A gen 10 from Seed A (68 evals): 1.832, still rising (1.572 / 1.753 / 1.812 / 1.832 at g0/3/6/10).
  - CV3@12 stack over single: +0.020 (170/205 families better).
  - F → Evo from F → CV3 was never run end to end.
- **"More aggressive inference" = more generations, not bigger batches.**
  - H317 ran 10 generations × 6 live SEARCH evals (from 48 screened proposals) = 68 evals per context.
  - H335 runs up to **107 generations** (22 + 6·107 = **664 evals**, ≈10×).
  - The per-generation batch stays K = 6 / P = 48. K = 16 would cut Evo A's UCB exploration share from 33% to 12.5%
    (`pick_exploit_ucb` keeps 2 UCB slots), and all TRAIN evidence is at K = 6.
  - The neural screener is unchanged: the H317 H0.1-A OOF models, 3,000 steps × 16 contexts × slate 128. No
    retraining.
- **The legacy ladder is not a target.**
  - H257's 1749.51 Joint Elo is family-grouped MM Bradley–Terry on native 2-fold S1.
  - Its top arms are best-of-~25 on the very folds the ladder ranks, and that number is never shown beside H335's.
  - H335's AUDIT-Elo puts every entrant on the same held-out footing.
- **dev91 is a reuse cohort.** Many earlier waves read it and H257 was tuned on it. H335 is the first read **by this
  lineage** (H317 → H335), not a pristine test.

## 1. Data boundary (unchanged from H317 §1, enforced by the same guard v2 + sandboxes)
- **Before the LOCK + pinned unlock:**
  - only TRAIN (`selected.json["train"]`, 1,219 contexts / 205 families);
  - H317-derived TRAIN artifacts (screener OOF models, merged TRAIN views, bank = TRAIN champions + winners);
  - H335's own files.
- **`h335_core` extends the H317 guard:**
  - the H335 evidence dir is allowed;
  - `/home/ubuntu/jope-data/h335-stagev/` and `h335-cl/redteam/` are denied pre-unlock;
  - the LOCK receipt is `~/jope_plan/ns/receipts/H335_LOCK_sha256.txt`.
- **Sandboxes:** `sandbox_h335.sh` (bwrap, xbabe2) and `sandbox_h335_docker.sh` (xbabe0). Their deny-by-default
  layout is H317's.
- **Never opened:** sealed / holdout / meta_test / fresh-cohort / DL-Test, `~/jope-data/h256|h257`, `~/jope-evidence`,
  `~/scratch`, `~/flywheel/data`, JopeDimeMicro `assets/benchmark_data`. DOPE `c51a56d0` stays frozen.

## 2. Pipelines (per context × search seed; one AUDIT request per output, `gp_engine.AuditLedger`)
- **Worker.** Every GP arm runs on the superset worker `2a848c8f…`. It is parity-verified against the pinned
  `bootstrap.s1main` `e3be9386…` on 2,010 program × context pairs, so shared programs give identical results.
- **The runner** is H317's with one change: the per-kind H317 timeout (60 s; ×4 portfolios; ×2 AUDIT) is enforced as
  a **CPU-time limit** (RLIMIT_CPU), plus a 3× wall backstop. A limit kill counts as a timeout (H317 retry-once
  policy).
- **Process groups.** The grammar is the only intended difference between the two groups. They never share a
  process:
  - `AGG`: the LEARNERS-V1 port, sha `3d6ec975…` (the H317 exploratory run's);
  - `BASE`: the pinned H317 grammar.
- **`H335_AGG`** (group AGG)
  - Seed F: 6 new-head presets interleaved with Seed E (8 registry presets + Seed A's 8 bank seeds) = 22 evals.
  - Then `evo_panmictic` (H317 Evo A) from F's 22 results, **K = 6 live / P = 48 screened per generation**.
  - Outputs at each checkpoint gc: the **single** best-by-SEARCH tree and the **CV3@12** S1-direct Frank–Wolfe stack
    (deployed weights from pooled SEARCH OOF predictions).
- **`OLD_MATCHED`** (group BASE)
  - Seed A (8) → the same Evo A.
  - Outputs at gen gc + 2: realized evals 8 + 6(gc+2) = 6gc + 20 vs AGG's 6gc + 22 (|Δ| = 2).
  - Single output = H317's adopted GP recipe at AGG's budget. `OLD_MATCHED_CV3` (same trees, CV3@12) is a
    decomposition only. Eval-seconds are logged per shard and group (engine stats); per-unit live-eval
    matching is checked (max |Δ|, count > 2) and reported with any dedupe budget shortfall.
- **`H317_BASE`** (BASE): OLD_MATCHED's gen-10 single (68 evals) = H317's adopted pipeline at H317's budget.
- **`H257_REPLICA`** (BASE): Seed A → 3 generations × 20 proposals / 5 live → CV1 Top-3 (the H317 port of H257's GP
  recipe). H257's pool-champion branch picks from res-01 pools that carry no DSL and were evolved on all rows, so it
  cannot be AUDIT-refit. This is disclosed.
- **`BANK_NN8_PICK`, `BANK_NN8_PORTFOLIO` (CV5), `BANK_NN1`** (BASE): cross-family bank references, as in H317.
- **LLM regressors.** The 4 frozen Rust regressors from the H257 ladder, single fit, seed 42: Opus5p5_Extra,
  Gemini4_argon_high, GPT6p1_Sol_Max, GPT6_Astra_Max. They are run with `frontier_audit_h335` (binary `b359807c…`,
  rustc 1.97.1, entrant sources == origin/main blobs):
  - fit on SEARCH rows only, predict AUDIT rows;
  - u = −ln(RMSE_audit / RMSE(SEARCH mean));
  - failure or timeout → u = 0;
  - RLIMIT_CPU 300 s, wall 900 s.
  - Parity with the unmodified H257 harness was checked on synthetic envelopes (identical ratios).
- **Prefix property** (makes checkpoints exact stand-ins for shorter runs): `evo_panmictic` draws an independent RNG
  stream per generation (`g.rng(arm, "gen", gen)`), and every proposal/pick depends only on the population so far.
  A G-generation run's first g generations therefore equal a g-generation run. Unit-tested with the fake worker.

## 3. Phase T — TRAIN run that fixes the settings
### 3.1 Panel and seed
- **P201b** = the second seeded context of each of the 201 TRAIN families with ≥ 2 contexts, in the same
  `stream_seed(PILOT_SEED, "ctx", ·)` order H317 used. P205 took the first, so P201b is disjoint from P205.
- Search seed **335101**.
- Within-family row overlap with P205 (60–75%) makes Phase T a calibration, not an independent confirmation. dev91 is
  the test.

### 3.2 Hosts, screener, views
- **Hosts.**
  - xbabe0: Docker sandbox, 12 shards × 8 single-threaded workers, cpus 16-63,80-127.
  - xbabe2: bwrap sandbox, 4 shards × 8 workers on cpus 38-53,102-117 (16 physical cores, both SMT siblings,
    idle-sampled at 21:38Z and disjoint from the H332/H334 reservations and their siblings), nice 19.
  - Contexts are split `index % 4` over the sorted P201b list (0 → xbabe2, else xbabe0). Within each host, AGG and
    BASE get half of the shards each. All arms of a context run on one host, and the host is recorded.
- **Screener.** The target's outer-fold H0.1-A OOF model pair (mean of its 2 training seeds), as in H317.
- **Views.** P201b SEARCH-row view sets are encoded with the frozen H317 recipe (`h335_encode.py`) on xbabe2 and
  copied to xbabe0 with a sha manifest.
- **Phase T arms:**
  - AGG at checkpoints {10, 25, 50, 75, 107}, single + CV3@12;
  - OLD_MATCHED at the matching gens {12, 27, 52, 77, 109}, single + CV3@12;
  - H317_BASE, H257_REPLICA and the BANK_* references;
  - the LLM regressors (descriptive).

### 3.3 Decision rules (fixed now; `h335_stats.phase_t_decision`)
- **G\*.** The smallest checkpoint whose family-mean AUDIT (AGG CV3@12) is ≥ the best checkpoint's minus one paired
  family-level SE (SD of the per-family difference to the best checkpoint / √F).
- **Stack.** Lock CV3@12 unless its family-mean gain over single at G\* is < 0 (then single).
- **Time cap.** dev91 wall projection = mean eval-CPU per physical run (Phase T shard engine stats) × projected dev91
  evals (both groups, 91 × 2 seeds) × 1.5 / (96 workers × 0.6).
  - The projection is the larger of that throughput bound and a critical-path bound: the slowest Phase T unit's wall
    × (G\* + 2) / 109 × 1.5. It refuses to project from missing shard status files.
  - If it exceeds 5 h, G\* drops to the largest checkpoint that fits, and the stack rule is re-applied there.
  - The two hosts' run configs must agree on code / worker / budget identity, and together they must partition
    P201b, or the decision refuses.
- **Hard stop (no dev91 read):** AGG invalid outputs > 2% at the locked output, or an unrecoverable infrastructure
  failure. Otherwise LOCK and Phase V follow without an owner touchpoint.
- **Reported, not gating:**
  - the full TRAIN table (every arm, family means, invalid counts, live evals);
  - the primary procedure of §5 applied to the TRAIN contrasts (locked output vs H257_REPLICA / OLD_MATCHED /
    OLD_MATCHED_CV3 / H317_BASE / BANK_NN8_PORTFOLIO / BANK_NN1);
  - the adoption-grade gate.
- **Engineering steps before Phase T (outcome-free):**
  - a memo-replay regression of the frozen H317 code path;
  - a 3-context smoke on non-panel TRAIN contexts (counts, failures, timing; utilities not looked at for any
    decision).
  - A code fix found there is a disclosed pre-outcome amendment.

## 4. LOCK and Phase V — one shot on DL-Val91
1. **LOCK.** `LOCK.json` (create-once, fsynced, `stage_v_h335.py write-lock`) records:
   - G\*, the stack, K / P, OLD_OFFSET, seeds **335101 / 335102**, checkpoints {10, G\*};
   - the pre-declared entrant list, the §5 rules, the Phase T decision file sha;
   - `code_sha256`, verified at the unlock and at every `resume_stage_v` on every host:
     - the runtime code only (h335_core, h335_lab, h335_stats, h335_frontier, frontier_audit_h335, stage_v_h335, the
       learners-v1 grammar, the three sandboxes);
     - the 18 frozen H317 modules;
     - the JopeDimeMicro featurizer files (`rust_evaluator.py`, `rank_prepare.py`, `program-feature-names.json`);
     - the superset worker, the frontier binary, the 10 screener models, the merged views, `champions.json`,
       `winners.json` and the TRAIN-inputs manifest.
   - Docs, tests and helper scripts are recorded in a non-verified `docs_sha256`.
   - write-lock refuses unless the Phase T run configs' code hashes equal the current runtime files.
   - Every Stage V step runs from a frozen read-only copy of the code.

   Its sha goes to the create-once receipt `~/jope_plan/ns/receipts/H335_LOCK_sha256.txt` and to AGENT_CHAT
   **before** the pinned one-shot unlock (`common_h317.unlock_stage_v`, receipt-checked, create-once
   `STAGE_V_STARTED`).
2. **Prep** (xbabe2, Stage V sandbox):
   - input sha verification;
   - the dedupe report (DL-Val vs the 1,219 TRAIN tables), recorded before any outcome;
   - the DL-Val SEARCH-row view sets;
   - the AUDIT envelopes for the LLM regressors;
   - **the R882 Native91 regression:** 91/91 bit-exact vs H258's pinned reference, mean 1.1377941898164525. A
     mismatch is `REGRESSION_FAIL` and stops the stage.
3. **Units** (xbabe0 only, single-host screener numerics, Docker Stage V mode):
   - (cid, seed, group) for 91 contexts × 2 seeds × {AGG, BASE}; the largest tables first;
   - AGG runs to G\* only, with outputs at the distinct checkpoints {10, G\*}; BASE runs to G\* + 2;
   - **screener = the fold-0 H0.1-A OOF pair** (same 2-model form as Phase T; no model saw a DL-Val family);
   - no state crosses contexts (the per-context memo, keyed by envelope, replays exact evals).
4. **Sealing.** Unit records are written but never aggregated or printed until every unit and every LLM record
   exists. Only then does `stats` create `FIRST_OUTCOME_READ` (create-once) and compute the readout.
5. **Failures.**
   - Before `FIRST_OUTCOME_READ`, an infrastructure crash is resumed: missing units only, byte-identical code and
     seeds, memo replay. Each resume is disclosed.
   - A deterministic pipeline exception fails **only the pipeline segment it occurs in**: its outputs get u = 0
     (the SEARCH-mean predictor), invalid. The other pipelines of the unit are unaffected, so a failing
     reference arm cannot hand AGG a free win. Segments: AGG seeds+evolution (all AGG outputs); each AGG output;
     BASE evolution (OLD_MATCHED*, H317_BASE); each of those outputs; H257_REPLICA; BANK_NN8 pick / portfolio;
     BANK_NN1. A failure is never re-raised or retried.
   - Integrity violations stop the stage instead (guard, AUDIT ledger, bank leak, any assertion, screener leak). They
     write a create-once `STOPPED_INTEGRITY` marker; audit, prep and stats then refuse.
   - **Unit-attempt cap** (H317's `MAX_UNIT_ATTEMPTS` = 2): a unit that raised a non-integrity error in-process
     twice (durable culprit lines) is recorded failed on its third start. Units that were merely in flight when a
     process was killed are never capped.
   - **Early read.** `stats --final-incomplete` (verdict `INCONCLUSIVE`) is accepted only when every planned audit
     shard has finished and the LLM step is done, or after the Stage V compute stop (2026-11-05 07:30Z).
   - Prep may be resumed any number of times before `FIRST_OUTCOME_READ`. A recorded R882 regression result is
     never recomputed.
   - After `FIRST_OUTCOME_READ` nothing is recomputed. An incomplete run reads `INCONCLUSIVE`.
6. **Dedupe** is descriptive plus a sensitivity readout. The primary is recomputed without DL-Val contexts that the
   dedupe flags as duplicating a TRAIN table (flag rule = H317 `DEDUPE_FLAG` 0.5, interpretive mode `raw_sorted`).
   The primary verdict uses all 91 contexts.

## 5. Statistics (h335_stats.py; per-context value = mean over the two search seeds)
- **Primary procedure** for a contrast arm vs ctrl over the 19 DL-Val families:
  - d_c = clip(u_arm − u_ctrl, −0.5, +0.5); D_f = the family mean of d_c; T = mean_f D_f.
  - p = the **exact** one-sided sign-flip p over all 2^19 sign vectors (identity included).
  - **pass** ⇔ T > 0 AND p < 0.025 AND the leave-one-family-out min of T > 0 AND median_f D_f ≥ 0. Ties never pass.
  - **inferior** ⇔ the mirrored test passes.
  - Why clipping: H317 recipe contrasts have heavy family tails (per-family SD ≈ 1; E−A mean +0.20 but drop-top-10
    only +0.013). The clip at ±0.5 log-RMSE (≈ ±40% RMSE) is fixed here, before any H335 outcome.
- **Fixed sequence**, each test at full α = 0.025:
  - **P1:** H335_AGG vs H257_REPLICA ("better than the incumbent GP recipe?").
  - **P2:** H335_AGG vs OLD_MATCHED ("better recipe, not just more compute?"). P2 counts only if P1 passes.
- **Verdicts:**
  - P1 ✓, P2 ✓ → **`BETTER_MODEL`**;
  - P1 ✓, P2 ✗ → **`BETTER_WITH_COMPUTE`**;
  - P1 inferior → **`INFERIOR_TO_INCUMBENT`**;
  - otherwise **`INCONCLUSIVE`**.
- **Adoption-grade secondary** (reported, not the verdict): the H317 contract gate on unclipped values.
  - Gates: paired whole-family bootstrap LCB95 > +0.001, median ≥ 0, ≤ 10% of families losing > 0.01, sign-flip
    p < 0.025, LOFO > 0. Bootstrap RNG 335150.
- **Headline:**
  - the unclipped family-mean AUDIT gain and its % RMSE reduction 1 − e^−Δ;
  - families won / tied / lost out of 19 (tie 1e-4), the context W/T/L, drop-top-k;
  - OLD_MATCHED_CV3 and H317_BASE contrasts (decomposition: how much is stack vs seeds/heads vs compute).
- **AUDIT-Elo ladder** (descriptive; inference rests on the sign-flip):
  - Family-grouped MM Bradley–Terry (H252/H258 code, copied into `h335_stats` and asserted equal): 400 iterations,
    prior 0.01, tie 1e-4. Family-bootstrap CI95, 10,000 draws, RNG 335160.
  - **Anchor H257_REPLICA = 1500.**
  - **Entrants (pre-declared; all select on SEARCH only or fit once):** H335_AGG, OLD_MATCHED, OLD_MATCHED_CV3,
    H317_BASE, H257_REPLICA, BANK_NN8_PICK, BANK_NN8_PORTFOLIO, BANK_NN1, Opus5p5_Extra, Gemini4_argon_high,
    GPT6p1_Sol_Max, GPT6_Astra_Max.
  - Also the pairwise Elo of H335_AGG vs H257_REPLICA.
  - It is labelled "AUDIT-Elo", never printed beside the legacy 1749.51, and noted to saturate when one entrant wins
    every family.
- **Also reported:** invalid counts (ADVANCE-style rule `invalid(AGG) ≤ invalid(ctrl) + 1` as a flag), live-eval and
  CPU counts, wall, guard counters, and the in-sample SEARCH utilities.

## 6. Resources, retention, deadlines (UTC)
- **CPU.** ≤ 85% per host, including other load. The engine pauses new evals above 85% CPU or below 20 GiB
  MemAvailable. Work runs as `systemd-run --user` units with CPU/memory caps, with logic in script files.
  - xbabe0: ≤ 96 workers.
  - xbabe2: 32 workers (Phase T only), nice 19, disjoint from other lanes' reserved cores (rechecked at launch).
- **Not used:** GPUs; xbabe1, xbabe3 and xbabe4.
- **Retention.** "Save the gene, not the model": compact unit records + the memo. S3
  `experiments/rd_breakthrough_20261030/H335/` (create-only, exact-version GET + sha verify). No deletes without owner
  approval.
- **Deadlines:**
  - prereg push before any fit;
  - Phase T compute stop 2026-11-05 01:30Z;
  - LOCK by 02:00Z;
  - Stage V compute stop 07:30Z;
  - closeout 08:30Z.

## 7. Red-team input (design red team, before this prereg) and disposition
- **BLOCKER** power at 19 families (heavy tails; harm rule ≈ unpassable) → clipped family primary + exact sign-flip;
  the raw contract gate stays as the adoption-grade secondary. Owner rejected a pre-test power gate → no power gate.
- **BLOCKER** R882/neural pool picks cannot be AUDIT entrants (no DSL; selected on all rows) → R882 only in the
  Native91 regression; anchor H257_REPLICA.
- **HIGH** OLD_MATCHED underspecified → superset worker for all arms, base grammar, matched within 2 evals at G\*;
  OLD_MATCHED_CV3 decomposition; CPU logged.
- **HIGH** code (F fell through to seed_D; import-time grammar flag) → explicit `h335_core.seed_F`; one process per
  group; grammar/worker sha in every record.
- **HIGH** K = 16 changes Evo dynamics → K = 6, more generations.
- **HIGH** Phase T rules (noise-level tie band; Phase T is calibration) → 1-SE rule; stated.
- **HIGH** wall / no-retry → largest-first, CPU-time limits, memo resume, sealed outputs, `INCONCLUSIVE` if
  incomplete.
- **MEDIUM:**
  - the LLM harness trained on AUDIT in its f=0 pass → fold-0 → fold-1 only, u = 0 on failure, CPU timeout;
  - Elo saturation → headline = gain + families won; "AUDIT-Elo" label;
  - legacy check invalid → dropped;
  - P1 comparator is a GP-only port → disclosed, plus the H317_BASE contrast;
  - screener form → same 2-model form in both phases.
- **LOW:** dev91 reuse cohort → stated; dedupe → sensitivity; bootstrap coverage at F = 19 → inference on the exact
  sign-flip.

### Code red team (before this prereg) and disposition
- **HIGH** G\* = 10 made the dev91 checkpoints (10, 10), and the second AUDIT request would abort Phase V → distinct
  sorted checkpoints, tested.
- **HIGH** one exception zeroed every output of a unit (could bias the primary toward AGG) → per-segment isolation
  (§4.5), tested.
- **MEDIUM** bank-leak / assertion errors were recorded as failed units → integrity errors stop the stage.
- **MEDIUM** the ±2 budget match was not checked → per-unit check in the decision.
- **MEDIUM** no cross-host identity check; staging checks were not fatal → run-config identity asserted in
  `decide`; xbabe2 runs from a frozen staged copy; fatal staging checks.
- **MEDIUM** the time projection ignored the critical path and could silently return 0 → critical-path bound;
  refuses without status files.
- **LOW** the width cap now applies in every phase; a worker exec failure (rc 126/127) is infrastructure (raises)
  rather than a program failure; the run-config check covers every identity field and shards use the config's
  timeout; the TRAIN Monte Carlo sign-flip is chunked.
- **Runner** uses `prlimit` (no `preexec_fn` in the threaded engine; core dumps off).

## 8. Disclosures (before any fit)
- **GPT6p1_Sol_Max** was split into modules and given static-analysis fixes on 2026-11-02, after its registration.
  H257 used this same version, and H335 uses origin/main's bytes. No entrant's commit history mentions DL-Val or
  tuning on a validation set.
- **Pre-prereg reads.** A read-only exploration agent mapping the legacy Elo ladder read H257 code
  (`/home/ubuntu/audit/JopeDime_main/research/flywheel_selfplay/h257_*`), README/REPORT text, and the JSON key
  names / schemas (no values) of H257 result caches. It reports opening no per-context values. Nothing from it feeds
  an H335 decision except the ladder's algorithm and the entrant list.
- **Plan vs PREREG.** The plan said "P201b × seed 331101" and "H331". The wave number changed (§0); seeds are
  renumbered 335101 / 335102 with identical roles.

## 9. Pre-outcome amendment A1 (2026-11-04 ~22:10Z; no Phase T or dev91 outcome read)
Engineering changes after the prereg commit e64f945d, before any Phase T outcome was read:
- **`sandbox_h335.sh` TRAIN view.**
  - The P201b view encoder (frozen H317 recipe) needs `cpu_exact_fused.py`, which is code: sha `b9f1468a…`, pinned
    by `crop_shuffle_synthetic_views.FUSED_SOURCE_SHA256`. H317's TRAIN-only view lacks it.
  - The sandbox therefore mounts `/home/ubuntu/jope-data/h335-view/res01-train-view`: hard links of H317's TRAIN-only
    view (1,219 tables / pools, redacted `selected.json`) plus a hard link of that one file. No DL-Val byte is in it.
  - xbabe0's Docker sandbox is unchanged (the GP needs no encoder).
- **Encoding.** 201/201 P201b SEARCH view sets were encoded on xbabe2 (69 s, DOPE `c51a56d0` unchanged, 0 guard
  refusals).
  - Cross-host check: 4 P205 contexts re-encoded on xbabe2 are bit-identical to H317's xbabe3 encodings.
- **Test guard log.** `test_h335.py` now sends its deliberate guard-probe refusals to a scratch log
  (`H317_GUARD_LOG`).
  - Before this fix, 16 probe lines (8 runs × 2 deliberate probes: an `h335-stagev` path and the H258 R882
    checkpoint path) went into the production `h335-cl/guard_refusals.log`.
  - The production refusal count is therefore reported as lines beyond this baseline of 16.
- **`stage_h335_xbabe0.sh`.** xbabe0's staged H317 screener dir lacks the 10 OOF `preds.npz` files, so the neural
  group is now an `rsync --ignore-existing` copy of xbabe2's `h335-cl/neural`, sha-checked against the full
  41-file manifest.
  - The failed first attempt had hard-linked 31 present files. They are byte-identical, verified by the same
    manifest.
- **Replay regression (PASS, PREREG §3.3).** On 4 P205 contexts × seed 317101, against copies of H317 memos:
  - AGG (`unit_agg`, 10 gens, single) == the exploratory run's `single`;
  - BASE (Seed A → Evo A 10 → single) == the main run's H0.6 `single`;
  - 4/4 identical each, 0 physical runs (428 / 276 memo hits).
  - Evidence: `xbabe0:/home/ubuntu/jope-data/h335-cl/replay_out_*/replay_{AGG,BASE}.json`.
- **Live smoke (PASS).** 1 P205 context, fresh memo, real superset worker via the `prlimit` runner in the Docker
  sandbox.
  - AGG: 3 gens, 52 physical runs. BASE: 12 gens + references, 118 physical runs.
  - 0 failures, 0 timeouts, 0 invalid outputs; ledger consistent. Utilities were not inspected.
- **Phase T launch.** 2026-11-04 22:04:52Z (xbabe0 `h335-T-xbabe0`) and 22:04:54Z (xbabe2 `h335-T-xbabe2`).
  - Both run from the staged manifest `manifest_T20261104T220258Z`; the Phase T files are sha-identical on both hosts.
- **`run_frontier_T.sh`** (descriptive LLM-regressor Phase T run on xbabe2 cpus 54-57,118-121) was added.
- **Stage V driver** (`stage_v_h335.py`, `sandbox_h335_stagev.sh`, the Docker `--stage-v` mode, the runbook
  `stage_v_mounts_h335.md`): built by a sub-agent and red-teamed independently. The statistics conform to the spec.
  Freeze / staging / resume findings were fixed before the LOCK:
  - the LOCK hashes runtime code only (docs and tests go in a non-verified field), and every step runs from a frozen
    copy;
  - the featurizer files (`rust_evaluator.py`, `rank_prepare.py`, `program-feature-names.json`) are in the LOCK;
  - durable integrity stop; unit-attempt cap; identity checks before `FIRST_OUTCOME_READ`;
  - guarded `--final-incomplete`; write-lock checks the Phase T code identity; staging scripts.
- **Phase V runtime smoke (PASS; red team HIGH-2; outcome-free).** The real `h335_lab.build_rt(phase="V")` +
  `run_units` were run on 2 NON-panel fold-0 TRAIN contexts treated as targets (views encoded into a scratch dir).
  - Setup: the Stage V flag faked in that process only, no DL-Val byte present; real superset worker, fold-0 screener
    pair; 8 gens; checkpoints (8, 8) dedupe path; 2 seeds.
  - AGG 4 units / 276 physical runs; BASE 4 units / 380 physical runs.
  - 0 failures, 0 timeouts, 0 invalid; ledger consistent. Utilities were not printed.
  - Script and output: `h335-cl/checks/smoke_v.py`, `checks/smoke_v_out_*`.
- **Encoder parity before the LOCK (PASS; MEDIUM-6).** The frozen `SV.encoder_parity` on xbabe2 for 4 contexts: the
  SEARCH npz and the full-table view 0 vs the merged TRAIN views (encoded on xbabe1 / xbabe3) are 4/4 bit-exact.
  Output: `checks/encoder_parity_pre_lock.json`.
- **Stage V agent test runs** were pinned to xbabe2 cores 44-45 (inside the Phase T xbabe2 set) for about 2.5 min
  each at nice 19. This affects wall time only; the worker limits are CPU-time.
- **`/home/ubuntu/.cache_never`** on xbabe0 (366 B, a list of 20 staged code filenames) was created by a reviewer's
  stray redirect. It is left in place (no-delete rule).
- **Production guard log baseline 17.** Line 17 is a Stage V builder test that tried to run a script whose name
  contained a guard token ("dlval"). The guard refused it by filename; no DL-Val data was involved. The script was
  renamed `build_stagev_view.sh`, and a test now asserts that no code-dir filename contains a guard token.
- **Stage V re-review (GO) fixes:**
  - the unit-attempt cap counts only culprit lines; bystanders of a process kill are never capped;
  - `run_audit_v.sh` is staged;
  - `--final-incomplete` also requires `LLM_DONE`;
  - a docker `--stage-v` dry run (`verify-lock-files`) runs before the audit;
  - S3 is a script. Stage V tests: 43 / 43.
- **Prep's LLM step is authoritative** for the dev91 LLM entrants (`run/llm_audit.jsonl`). `h335_frontier.py --phase V`
  is not used in Phase V.
- **Stage V prep cores (xbabe2).** At 00:49Z, `h344-run` held cores 54-63. Prep and stats therefore moved to
  38-53,102-117, the Phase T xbabe2 set, which was free after `h335-T-xbabe2` ended (`run_prep_v.sh`, runbook, and
  the matching test assertion). Wall time only; no result depends on it.
- **Phase T completed** (TRAIN, P201b, 201 contexts × seed 335101; 2026-11-04 22:04Z → 2026-11-05 00:52Z, within the
  01:30Z stop).
  - All 402 units completed (xbabe0 300, xbabe2 102); 0 tracebacks; 16/16 shard status files.
  - 284,087 physical worker runs. AGG had 36 eval CPU timeouts (60 s; failed evals, not invalid outputs); BASE had 0.
  - Guard: 0 refusals beyond the baseline of 17.
  - The xbabe0 run dir was pulled back to xbabe2 and every file sha-checked (1,419 files).
- **Phase T decision** (`h335_lab.py decide` from the frozen copy; `h335-cl/decision_T.json` sha `c36591c36cff6005…`).
  - **G\* = 50.** The best checkpoint was 107 (CV3 family mean 1.9448). g50 = 1.9340 is within 1 paired SE (0.0147);
    g25 = 1.9285 is not.
  - **Stack CV3@12.** Its gain over single at G\* is +0.0445.
  - Time cap not binding: projected dev91 wall 2.62 h (throughput bound; critical path 1.55 h).
  - **No hard stop:** AGG invalid share at the locked output is 0.0.
  - Live-match: max |Δ live evals| = 2 at every checkpoint, 0 over tolerance. Mean live evals: 321.9 locked, 319.9 OLD.
  - 0 budget shortfalls.
  - Reported: 7 of 201 AGG units used ≠ 22 Seed-F evals.
  - **Locked output:** `AGG.g50.CV3@12`.
- **Phase V shard layout kept at 6 AGG + 6 BASE × 8 workers** (`run_audit_v.sh`, unchanged). Phase T cost per run was
  3.10 s (AGG) and 2.30 s (BASE). At G\* = 50 that predicts a balanced makespan (6:6 ≈ 7:5 within 1%).
