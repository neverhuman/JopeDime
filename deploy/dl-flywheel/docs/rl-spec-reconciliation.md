# JopeDime RL flywheel: reconciled design and execution handoff

Evidence date: 2026-10-09 UTC. This is a code-informed reconciliation of the user-supplied recommendation, not a claim of measured RL lift or generalization. The original attachment has SHA `c0103ef73437f45e50b3d49b9644478a780a9d8a677eba031351f798039f9217` (52,796 bytes). It explicitly describes an unexecuted recommendation based on an older review packet, not a source audit.

## Decision

Use outcome-verified expert iteration first: generate a diverse finite-library slate, measure programs through native Lambda and emitted Rust, build comparable measured targets, distil those targets into a separate policy challenger, and qualify it against frozen equal-budget opponents. Add genuinely on-policy RLOO plus a marginal best-of-B objective only after that loop passes executed games. Break the finite-library ceiling through a separately versioned typed AST search arm. A larger replay pool, falling imitation loss, rising historical maxima, and busy GPUs do not establish a better generator.

This fits the separation of search and learned policy in [Expert Iteration](https://arxiv.org/abs/1705.08439), measured weighted likelihood in [AWR](https://arxiv.org/abs/1910.00177), and the simpler sequence-level policy-gradient route studied in [RLOO](https://arxiv.org/abs/2402.14740). These papers motivate mechanisms; their results do not predict JopeDime's effect size.

The immediate missing edge is **verified outcome → policy probability update → measured better next search**. The current live PolicyLM uses supervised imitation and quality/feasibility/runtime heads. New generated outcomes are published to the library; they have not yet been admitted as new actor targets. Calling this live model RL would be wrong.

## Current code and live evidence

PR277 head at this audit is `61e712cfc1ee8341776161eb6886ad73d3c094b3`. Runtime source snapshots remain immutable; commits do not hotpatch active modules. Existing PR277 is conflicting with newer main; final delivery must resolve that conflict without dropping foreign changes or rewriting shared history. Current-head CI and independent native approval are separate from local fixtures. No merge, tag or release is certified here.

The active same-model executor is `jopedime-neural-continuous-recovery5-20261009`, logical campaign `flywheel_neural_continuous_20261009_v1`, config `fec184217e7c50a3fc3bc1cc7e33edafdbd218d1fafa1bd0d68debf626dcd4ed`. Its global deadline remains `1791592146.380285`; the maximum is 50,000 optimizer updates from checkpoint429. Checkpoint, AdamW, CPU/CUDA RNG, normalizer, roles, batch membership fingerprints and fixed loss panels remain in custody. Physical checkpoint9981 was read, hashed and joined to its manifest, committed batch and optimizer control; it was not inferred from utilization. Rolling files can retire; this is not an S3 archival claim.

The previously held128 extension **has been admitted**: 7,248 contexts, 6,982 training and266 validation. All6,982 admitted training contexts were actually consumed. A dated dashboard read at14:10 showed9,986 committed batches,22 completed epochs and159,594 presentations; refresh those numbers before reporting a live total. A context is a dataset version plus its exact split/protocol, not an independent parent family. Reused contexts raise presentations while unique coverage remains flat.

A further160 fresh Kumo vectors were produced and exact-version published during this turn. Raw variants rose4,832→4,992 and verified required encodings736→896. Those160 are **encoded, not GP-scored, admitted or consumed**. The next128 comprise118 train and10 validation descendants of128 distinct parents. Raw/encoded/scored/published/admitted/first-consumed must remain separate counters.

Python3.10's broken NumPy failed before dispatcher main/GP execution. A new immutable packet uses the physically verified Python3.12 binary `e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f`. At14:39 four native scorers were active:177 claims,172 successful parity-qualified native/Rust executions and172 exact-version publications, two known nonnumeric failure custodies (the same two outcomes also counted as known rejections), andzero unknown outcomes. These counters are not a count of every started native command. These are dated counts. Their fixed deadline is14:54:12UTC; neither failure nor completion resets it. They all use the same permanent context/program registry. More workers must share that registry and pass current memory/foreign-job guards.

Loss plots are restored with independent Linear/Log X and Y. Current fixed validation is approximately0.8732 versus best0.8712135; the unchanged controller reached LR1e-6. This is a plateau, not evidence that more identical supervised updates will create an RL flywheel. Preserve that live control; a challenger gets an explicitly new optimizer/config, not a silent LR reset.

## Reuse the foreign RLFW work instead of duplicating it

A read-only snapshot at private head `6de9120f821f84b345ae292d8d486c5b1fa92759` in the foreign `ci-determinism` lane contains the following. It is not a merged-package or live-qualification claim. Public arena PR280 is separate from PR277.

| Interface | Existing source under deploy/dl-flywheel/pipelines/rlfw | Status and required connection |
|---|---|---|
| Comparable HWM | research_hwm.py, hwm_adapter.py | Append-only SQLite CAS, first-label/improvement distinction; remote-reference shape checks rely on the caller's actual exact-version closure verifier. Never equate a reference with verified bytes. |
| Actual probability | trie_logprob.py, policy_binding.py, decision_trace.py | Inference readout includes EOS and actual trie children. Read-only docs record a1,000-draw CPU proof for an older sealed checkpoint; a different checkpoint/device/batch geometry needs its own qualification. |
| Equal-budget games | arena/game_spec.py, oracle.py, scheduler.py, kpi.py | Score-free specs, charged disclosure slots and family statistics exist. Their documented unknown retries conflict with this pipeline's permanent no-retry rule: connecting adapter must refuse physical replay and quarantine unknowns. Do not modify the foreign STOP or package silently. |
| Measured distillation | exit_targets.py, exit_objective.py, exit_offline_train.py | Differentiable trie likelihood, measured-set/raw-iid target forms, frozen anchor and default-zero L6 parity. Trainer is a separate shadow optimizer, not a resume of the live service. Its current qualified train corpus/runtime admission remains unavailable and the foreign parity STOP is active. |
| Search/RLOO/AST | Not established by those files | Treat as later code and executed gates, not already running capabilities. |

The latest direct human request says: “Please make sure you are ALSO cordinating Codex sessions to help with Grok and ensure they are working together well.” This authorizes the recorded cross-chat evidence request and coordination with Grok. Replies from another task alone would not authorize new messaging. The shared handoff and locked AGENT_CHAT are the authority/ownership channel. Codex implementation agents own disjoint local pool, fresh-variant guard, and independent-review scopes. Foreign RLFW paths, protected STOPs, Grok's native/emitter code, canonical main/index, active runtime modules and historical outputs are read-only unless their owner supplies an explicit stopped-head transfer. The append lock serializes claims, not checkout ownership.

## Corrections required to the recommendation

### Policy probability and computation

Every actor term must use the **summed EOS-inclusive probability under the actual sampling support, temperature and execution geometry**. Grammar-masked mean-token CE is a different distribution. Keep CE as supervised regularization; do not use it as a likelihood ratio.

Define B as **logical score-disclosure slots**, including duplicate/invalid/failed/cache-hit disclosures. The attachment also calls B new executions elsewhere; that is inconsistent. Physical cold executions and CPU cost are a separate ledger. Warm libraries must not grant free information to one arm.

Current batched decoder casts supported logits to CPU float64, uses sorted trie children and per-row RNG. Batched versus scalar float32 network kernels can differ. A scalar differentiable actor API that matches scalar log-softmax is not alone proof that it matches the deployed batched sampler. Pin/qualify the complete protocol. Full-trie KL requires logits at every branching prefix; prefix target aggregation reduces duplicated work but is not one ordinary Transformer forward over a context. Measure time/memory before selecting exact versus sampled KL. A sampled KL can have an unbiased gradient while its numerical estimate is negative; do not label it exact.

### Tilt target and its guarantee

For a **deduplicated measured set** M use normalized weights proportional to `pi_anchor(p|c)^alpha * exp(g_p)`, with `g_p=clip((q_p-b_c)/tau,-A,A)`. For raw iid draws from that anchor, retain multiplicity and omit the extra anchor factor. Otherwise its probability is counted twice. All samples require exact comparable measured outcomes; no q-head prediction enters a target. Raw-iid weights with no anchor factor implement alpha=1 only. For generalized alpha, the corrected weights require pi_anchor^(alpha-1); foreign exit_targets.py deliberately ignores alpha in raw_iid mode, so bind it to alpha=1 rather than claiming arbitrary-alpha equivalence.

The unclipped `alpha=1` tilt is the optimizer of raw-q expected reward minus tau·KL on M. The specified clipped transform is instead a regularized optimizer for a transformed utility. The derivative `dE[q]/deta=Var(q)` holds for the unclipped raw-q tilt; do not claim that identity for saturated clipping. Monotonic ordering on the measured set does not guarantee a shared network learns it, preserves outside-set quality or generalizes. Logged certified anchor mass reveals how little of total support may be measured. The current foreign target implementation uses raw q in the exponent with a q_opt mean baseline; record that choice exactly rather than calling it clipped-q advantage.

First shadow A/B: current L6 baseline versus L6+L_PI+anchor KL, using equal updates, immutable train-only targets, fixed source/normalizer/trie and declared optimizer control. Starting suggestions lambda_PI=.25, tau=.2 and lambda_KL=.02 are experiments, not measurements. They must not overwrite the live LR1e-6 controller. Preserve lambda=0 bitwise parity. The current foreign shadow trainer checks unique context IDs, which alone does not enforce unique parents: its runtime next_batch adapter must prove parent uniqueness and parent-macro weighting. Record every shared-trunk gradient norm/cosine, target receipt and first exposure.

### Best-of-B gradient normalization

For iid current-policy draws of sizeG=B, a fixed comparable incumbent h and the same utility u everywhere:

`A_i=max(0,u_i-max(h,max_{j!=i}u_j))`

`E[sum_i A_i * grad(log pi(a_i))] = grad E[max(h,u_1,...,u_G)-h]`.

The recommended loss uses an additional1/G. Its gradient is therefore1/G of that objective gradient. Either retain the normalization and version that scale in lambda_HWM, or use the sum. Do not call the written average an exactly normalized gradient of J_B. G must equal the reportedB for that identity.

The proposed raw-q HWM objective and q_opt sample utilities must not be mixed. A raw incumbent above log1000 with capped samples yields zero marginal advantages even when raw-q improvements exist. Choose one of: raw comparable-q J_B with appropriate numerical/gradient controls; or a **separately named clipped-utility best-of-B** with identically clipped incumbent. The public research HWM remains raw q_grid. Record both objectives separately.

### Dense RLOO, staleness and unknowns

Dense averaged RLOO is unbiased for **unconditional registered utility** when samples are iid and its leave-one-out baseline is independent of each sampled action. It is not automatically the gradient of q conditional on successful completion. Program-failure utility is separate from q; parity/infra unknowns receive no invented utility.

A policy changed by64 optimizer updates is not on-policy merely because it is younger than180 seconds. For the first RLOO experiment freeze the actor/learner parameter identity through collection and use one update per fully resolved group. For delayed data, use L_PI by default. Per-item importance ratios can correct the dense mean-reward estimator because its baseline remains independent of the selected action; the **group maximum** generally requires joint `product_i pi_theta(a_i)/mu(a_i)` weighting to target current-policy J_G. Importance sampling requires known behavior probabilities and positive behavior support wherever the current policy has positive mass. Support/version drift or unavailable behavior log-probabilities refuses importance sampling. All importance weights are detached in the score-function log-loss; differentiating the weights as well changes the estimator. Joint ratios have high variance. Rejecting groups based on ratios/outcomes introduces selection effects; label gated/clipped estimators as approximations, not unbiased guarantees.

Unknown groups are quarantined with their full raw draw inventory. Never replay an unknown GP, shrink to successful samples, invent failure labels or reroll until a complete favorable group appears. Duplicate draws remain in iid groups; an exact successful cached outcome can supply each duplicate's measured value without another physical execution. Report censored-group fraction and whether worst-case bounds could reverse lift. Completion is action-dependent, so quarantine alone does not make the complete-case population an unbiased unconditional estimate. The foreign arena has win-score censoring bounds; these do not also certify an unbounded raw-q improvement interval. For promotion require registered per-context q bounds derived from exact baseline/epsilon, or declare the lift inconclusive.

The attached fresh-rerun teacher integrity step is not permission to execute completed or unknown keys again. Re-verify retained artifact bytes and receipts; a new robustness split is a new exact context/protocol, with its own one-use custody. Native promotion remains a separate owner gate.

### Verified numerical audit

An exhaustive finite three-action audit enumerated every group for B=1,2,4. The marginal gradient matched central finite differences with maximum error below1e-9; averaged loss produced precisely the1/B scale. Dense averaged RLOO matched the analytic mean gradient forG=2,4. A stale per-item HWM correction had maximum bias0.481645, while joint weighting error was6.94e-18. A raw-incumbent/clipped-sample counterexample produced all-zero actor gradients despite a nonzero raw objective gradient. A completion-conditioned constant-q example had zero conditional-quality gradient and a nonzero failure-inclusive utility gradient. These are data-free mathematical checks, zero optimizer updates/GP/cloud requests; not live convergence evidence.

## Encoder and judge boundary

Use only frozen licensed KumoSmall2051. No PriorLabs, TabPFN3.5 or unnecessary encoder blocks. Context vectors contain pooled pre-final-normalization hidden states0:2048, target-aware query skill2048, predicted quantile spread2049 andlogspread2050. Technical-review source documents all-task-label XGBoost feature selection and all-task min/max; Kumo's context targets also condition hidden states. Masking2048–2050 alone does not remove that upstream label dependence.

Current inner-fold search is **task-label-aware inner-CV search**. Build a distinct judge-clean lane: quarantine judge rows before every selection/scaler/encoder operation, record fit-row IDs and family/source crosswalk, freeze selection/model, choose one program per arm afterB optimization-score disclosures, then unlock judge once. Poisoned judge labels must leave encodings, proposals, optimization scores and weights unchanged. A2 query-label diagnostic, A3 fit-only feature selection, A4 judge poison and A5 source/pretraining-overlap checks have separate receipts. Do not claim unseen families while Kumo overlap is unknown.

Adding siblings toward100k variants can improve robustness but does not multiply independent families. Preserve source lineage/transform/RNG/target/split metadata in exact-version S3; all descendants inherit their parent's role. Parent-first sampling, one context per parent per batch and a rolling sibling-exposure cap prevent prolific families dominating. Needed encodings and actual measured GP labels precede admission. New outcomes reach the learner only through a reviewed immutable epoch-boundary target transfer, with exact optimizer/control/planner/outbox custody.

Keep mean-baseline q canonical. With identical folds/epsilon, switching to CatBoost q adds a context-only constant, so raw within-context ranks and centered advantages do not improve merely by changing denominator. This equivalence is before clipping; applying q_opt saturation after different context shifts can change clipped advantages. CatBoost is valuable as an external competition yardstick and difficulty report; pin and measure it separately. A changing baseline must not silently redefine rewards/HWM keys.

## Worker-ready implementation sequence

### Read-first and completion conditions

Read `/home/ubuntu/JopeDime/AGENTS.md`, MASTER_PLAN.md, docs/adr/ADR-0005-canonical-learning-loop.md, current lockedAGENT_CHAT and the shared dashboard handoff. Then read `deploy/dl-flywheel/docs/{technical-review,generated-variant-execution,prepared-admission-and-transfer,native-build-plan-revision2}.md` at the actual PR head. Read the foreign RLFW pinned docs/interfaces above and its current INTEGRATION_DELTA/STOP from the owner. Do not use this audit's private head as an official merged-package dependency.

Completion of the first learned flywheel requires: (a) qualified measured train targets first-exposed in a real shadow optimizer update; (b) its next proposals actually executed/verified; (c) paired equal-budget family-macro games improve over both the frozen policy and uniform comparator, with bounds robust to unknowns; (d) separate judge evidence before generalization claims. A receipt, code merge, lower loss or HWM increase alone is insufficient.

Stop on source/role/support/protocol drift, unknown previous execution/update, missing exact artifact closure, foreign STOP, conflicting writer, unsupported evaluator semantics, failed gradient-sign fixture, resource/deadline violation, or a gate whose result is unavailable. Preserve all attempts/deadlines; no automatic replay.

### P1: close measured-outcome and target custody

Reuse RegisteredInputs and S3Library exact-version/hash/bytes checks and existing emitted Rust closure. Create an owned read-only export adapter, proposed `deploy/dl-flywheel/pipelines/rlfw_bridge/{verified_rows.py,custody.py}`; do not modify the foreign rlfw package before transfer. Map only qualified complete same-context outcomes to rawq/q_opt/q_grid and fullHWMKey. Record explicitly `fitness_only`, absent, incompatible, degenerate baseline, refused and unknown. Audit original pending/no-winner eligibility using current physical manifests, not the dated5,461 count.

Connect the exported rows to the owner's `ResearchHWM.publish` and `exit_targets.build_targets` via a pinned actual closure verifier. Prove that foreign evaluator/protocol/parity metric match before accepting our original or prepared-variant results. Equal tolerance numbers alone do not make protocols identical. For remote closures require the versioned wrapper plus all seven artifacts. Tests: corrupted/stale/missingVersionId, role flips, objective mismatch, parent aliases, program/token mismatch, outside-trie zero actor support, duplicate publication and null incumbent. Exports are immutable, train-only for targets; validation stays evaluation-only.

### P2: qualify game and distillation integration

Reuse the owner's committed score-free arena. Add the no-replay adapter and global exact execution registry; prove cache hits charge one disclosure and unknown attempts remain censored without native invocation. Capture frozen comparison support/cache/roster/seeds/B/receipts before any scores. Measure current q-head Spearman/top8 recall by parent and program length; use simple measured ranking until calibrated uncertainty exists. K32/B8 and5exploit/2explore/1uniformaudit are starting arms, with K16/64 comparisons at equalB. Uniform means whole-program uniform, not per-node uniform. Retrieval uses train-only measured programs and records migrations as unscored on the destination context.

Build exactly pinned train targets and the owner's shadow runtime binding. Run the default-zero L6 exactness gate, then a bounded real CPU challenger; CUDA requires its own passed same-pin device/geometry qualification. The real reserved optimizer step logs include source/config/target/checkpoint/RNG/parameter hashes and first-exposure contexts. Preserve unknown optimizer calls, never resume them automatically. Track target starvation separately from model training. Every terminal program sampled by the shadow actor, including KL-only sampling, needs its own durable observed-proposal outbox before it can be lost. The current foreign trainer's internal KL draws do not inherently establish this custody. The new runtime adapter must record exact context/program/checkpoint/role identities, use exact cache and the permanent no-retry registry, and enforce bounded scoring demand. Never write the active service outbox or mint fresh execution identities to escape old holds. Teacher-forced target likelihoods are not new proposals. Conduct policy-alone and search-plus-policy arena comparisons to separate better search from learning.

### P3: true on-policy and novelty experiments

Only afterP2 passes equal-budget executed games: add proposed `rlfw_bridge/rloo_groups.py` and `rloo_objective.py` with the corrected objective identities/scales. Required fixtures enumerate finite action groups B1/2/4, check finite differences, dense baseline independence, duplicate multiplicity, mixedclipping refusal, stale jointIS and unknown censorship. No group-standard-deviation normalization under an unbiased claim. Compare RLOO+HWM against L_PI-only at equal evaluator budget, not equal learner steps alone. Failed/unknown groups and lag distributions are published.

Perform a train-only library-ceiling audit on30 distinct families before committing most compute to trie versus AST. A new typed AST module has a pinned operator/type/arity/emitter manifest, <=64nodes/depth8/tokenbounds and new grammar-support identity. It must produce at least one outside-trie canonical program with qualified native/Rust results. Mutation/crossover/retrieval have unknown behavior propensity unless explicitly modelled; use measured distillation rather than laundering their likelihood as on-policy probability. Constants fitted on fold fitting rows define a new parametric evaluator/HWM namespace.

### P4: throughput and delivery

Scale real evaluators using current resources and one permanent execution registry. New xbabe2 pool retains unchanged revision2 evaluator bytes and distinct one-use workers; fresh xbabe3 variants have whole-context exclusivity, historical alias holds, exact inherited roles and a final live lease/authority check immediately before nativefit after expensive replay. Do not copy SQLite between hosts as a supposed global lock. Preserve foreign jobs and the encoder/learner residual memory reserves. Measure executedworkers separately from configuredworkers, actual core seconds versus quotas, and disjoint phase spans; do not add overlapping parent/leaf times.

Prioritize verified input/baseline caches, native/emit/compile/Rust/parity/publication timings, bounded readback concurrency and producer-lock contention. Key compile caches on exact generatedsource/toolchain/flags/target/emitter. No unknown execution becomes a cache hit. KV decoding, persistent resident Kumo and actor/learner split are later reviewed protocols; CPU/GPU numerical traces, RNG and exact artifacts must remain qualified. Reuse the current producer's actual audit fraction; first measure it from the pinned worker instead of assuming the recommendation's1% applies to every new packet.

All code goes under the xbabe2 project, with SHA manifests/CI classifications/runner registration and coherent docs. Preserve the PR277 history and source packet hashes even if a clean conflict-resolved replacement PR becomes necessary. Final acceptance uses exact final-head CI plus an independent eligible GitHub reviewer; main/index/foreign checkouts stay intact. A changing head invalidates old approvals. Do not force-push, self-approve or use an admin merge. Update the live dashboard's pipeline-state counters, target feedback latency and separately labelled supervised/actor losses only from real receipts.

## Logging and dashboard contract

For each proposal/group/program/target/update/game log exact UTCnanoseconds and monotonic boundaries, parent/context/role, source/config/evaluator/protocol/checkpoint/trie/normalizer/RNG pins, raw draws and per-token summedlogprobs, rawq/q_opt/q_grid, each outcome class, logical disclosure versus physical execution, S3 VersionIds/hash/bytes, queue/planner/lease/globalattempt custody, CPUuser/system/child+cgroupseconds, CUDAevent time, cache bytes/hits/evictions, and target first-exposure latency. Retain rollback and failed packets. GPU utilization is separate from CUDA execution time.

Dashboard claims should be: unique contexts actuallyconsumed, totalpresentations, batches/completedepochs, verifiedrequiredencodings, measuredprograms, admittedpending/refused contexts, researchfirst-labels versus improvements, equal-budget wins with lineage intervals/censoringbounds, and feedback latency. No global GP Elo across unrelated contexts; if added, version ratings on actual comparable measured games. Absent games remain unavailable. Keep fixed original panels separate from expanded validation and judge scores. Keep independent log axes with explicit nonpositive omissions and honest unique-coverage plateaus.

## Near-term concrete deliverables and remaining blockers

1. Push the cardinality-aware native pool and fresh128 guarded scorer, after affected fixtures and independent exact-packet review; launch only with complete physical resource/history/input qualification.
2. Export current successful native/variant artifact closures for RLFW owner's adapter qualification. Reconcile its parity STOP and retry contract without bypassing them.
3. Build first train-only measured targets and one bounded shadow challenger with the existing owner package. This is the first reward-directed policy update, and remains pending until exact closure/runtime qualification is complete.
4. Resolve final source integration conflicts, complete exact-head CI and independent approval, then merge and read back main. Preparation is not merger evidence.
5. Run equal-budget baseline/challenger games before claiming flywheel takeoff. Build the sealed judge lane before claiming generalization. All starting hyperparameters remain registered experiments.
