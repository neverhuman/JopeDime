# GP language, search and evaluation audit

The retained production CPU release implements bounded, neural-seeded evolutionary program search. The evidence does not establish AlphaGo Zero, complete language coverage, universal fit custody, or independently measured generalization from its adaptive teaching scores. Several useful safeguards exist in current source, but that source is distinct from the binary that collected the retained observations.

## Evidence boundary

The direct target capture in [remote-source.json](authority-gp/remote-source.json) was made at 2026-09-11 12:27:19 UTC. It independently hashes the native-v5-cpu executable, checks that hash against target-build.json, hashes five retained source/configuration files against build.json, and records that fit_boundary.rs is absent from that frozen source. This establishes retained release identity and source-manifest agreement. It does not reconstruct every compiler input, dynamic mapping, historic process interval, or numerical result.

The earlier root capture, parallel/runs/recovery-execution-20260911/live-before/stdout.json, records GP-next active with 104 configured workers. The later direct target capture records both GP units with MainPID=0 after root containment. The historic live state must not be presented as current. Root owns the complete containment and resident-pool evidence.

All source claims below distinguish the retained native-v5 release, current dirty source, and proposed work. No fitting, protected target access, new search, compile, deployment, or numerical test was performed for this audit. The code edits interrupted by the switch to planning remain unqualified; their exact status appears below.

## 1. Language availability has six separate boundaries

The language must be audited through parser acceptance, semantic validity, interpreter support, emitter support, policy encoding, and search reachability. Passing one boundary does not establish the others.

| Surface | Direct source evidence | Consequence |
|---|---|---|
| General GenomeIr parser | vendor/feat-math/src/genome.rs:19 and :75 accepts pipelines, symbolic expressions and nested portfolios | General parser breadth exceeds the selected evolutionary generator |
| FullRust grammar | vendor/feat-math/src/full_rust.rs:15-80: six transforms maximum, expression depth 3/nodes 15, 31 base plus five expansion transforms, 11 canonical heads | A separately bounded, opt-in pipeline grammar; not arbitrary Rust and not all parser forms |
| Selected policy vocabulary | parallel/assets/grammar/featmath.json:2518, :2538, :2868; hashed metadata census | 401 tokens; 53 op records (31 transforms/22 heads); 51 generation-enabled; portfolio and ridge_gcv generation disabled |
| Policy geometry | parallel/src/grammar.rs:103-150; selected vocabulary | Six transforms, three expression stages, depth 4/nodes 15; eight direct column-variable tokens |
| Generic GP generation | parallel/assets/source/jope-dime/vendor/feat-math/src/evolve.rs:28-88 and :1494-1542 | Legacy draws use 18 transform names and nine weighted head slots, corresponding to seven distinct head names; expansion needs separate opt-in entry points |
| Deployed search | retained native-v5 source/parallel/src/gp.rs:821-832, captured remotely | Calls legacy random, mutation and crossover helpers; broad FullRust availability is not evidence that this loop samples it |

The eight explicit column-variable symbols constrain that expression sublanguage, not the dataset to eight features: vector transforms and model heads still consume the full permitted matrix. Turing completeness is not an acceptance goal; finite, validated programs with useful operators and predictable cost are the relevant target.

The current parallel search has the same relevant legacy calls at parallel/src/gp.rs:1080-1098. Neural proposals and retained seeds can still introduce operators absent from generic generation. Thus the accurate conclusion is an unmeasured reachability difference, not that all such operators are impossible. A retained seed may survive or be crossed, while random insertion and head replacement remain narrower. Repeated projections can also collapse distinct native proposals to the same effective candidate.

Proposal projection is explicitly lossy: parallel/src/grammar.rs:347-402 removes unsupported parameters, snaps numeric values to bins, and can substitute the first bin for unsupported categorical values. Current GP records pre-projection syntax when changed (parallel/src/gp.rs:1343). This is defensible only when the projected program is the actual evaluated, deduplicated and trained identity. A count of raw proposals overstates distinct legal programs.

The general symbolic parser accepts any mode other than `affine` as raw and tolerates a missing final closing parenthesis through `unwrap_or(rest)` (vendor/feat-math/src/genome.rs:87-101). This is a source-confirmed parser leniency; no exploit was executed. Canonicalization may collapse such inputs safely for fit reuse, but invalid spellings should not silently become a different declared mode in a claimed strict language.

**Required plan:** publish an immutable operator/parameter capability matrix with one row per canonical operation and separate columns for parse, canonicalize, execute, emit, encode, sample, mutate, crossover, and deployed support. Treat grammar changes as new lineages; do not alter legacy RNG arrays. Add parser rejection fixtures for unsupported symbolic modes and malformed delimiters. Quantify proposal rejection and projection collision rates by operation and family before broadening the search.

**Acceptance:** all declared reachable operations have deterministic generation/mutation witnesses, exact canonical round-trips, original synthetic execution and emission parity artifacts, source/runtime bindings, and explicit resource caps. Unsupported cells remain unavailable. A larger grammar is an experimental arm only after those obligations pass.

## 2. The selected algorithm is evolutionary search with a learned prior

The current search retains up to 80 distinct prediction signatures, sorts by adaptive validation RMSE, selects parents from the top 10, uses a grammar-random attempt every eighth proposal attempt, and alternates mutation/crossover otherwise (parallel/src/gp.rs:1028-1098, :1310-1320). It starts from 80 seeds and enforces 1,000 evaluations/600 seconds (parallel/src/gp.rs:819-870). Duplicate projected programs are rejected before another local fit attempt (:1047-1113). These are concrete search mechanics.

AlphaGo Zero trains a policy/value network from search-generated move choices and eventual game winners, improving tree search across iterations. That result used self-play game trajectories and tree search, not merely a population of independently scored candidate programs. The author manuscript is available from UCL.[1] The inspected Jope loop contains no MCTS selection, visit-count policy target, PUCT action selection, or terminal two-player game return. Its neural objectives instead include weighted sequence likelihood, pairwise/listwise reward ranking, runtime loss and an ablation ranking term (parallel/src/training.rs:674-705).

This does not invalidate learned evolutionary search. It means AlphaGo Zero is an inspiration, not a demonstrated implementation equivalence. The useful causal claim is whether a new trained policy produces better admitted programs or better quality per reserved compute on independent tasks, and whether its subsequent discoveries improve the next policy. Neither busy GPUs nor an autoregressive grammar alone proves that loop.

**Required plan:** retain evolution as the explicit baseline. Define the state/action/reward and search-improvement operator before proposing an MCTS variant. If actions are grammar tokens, state must include a valid prefix, terminal fitting must obey authority, and a value estimate must predict future program quality under a specified completion policy and budget. Compare this separately against current evolution with matched reservations; do not silently replace frozen qualification protocols.

**Acceptance:** two original successor discoveries must be attributable to distinct accepted policy checkpoints, with independent training lineages, matched protected-reference decisions, complete compute charges and no history-rejoin counted as fresh discovery.

## 3. The retained CPU release predates the authority boundary

The directly read native-v5-cpu source calls feat_math::fit at gp.rs:35, :70 and :440. The frozen source lacks src/fit_boundary.rs. Its executable sha256 is 974e93830b51413187ccb0ac19d1dc878b84a6246a2d0981419feb12d38b7dc2, independently matching target-build.json. The source gp.rs hash is 29d3452f775f91851b1cc906bf2dd188be76c85b052309f46abc583523892f02.

The retained build receipt records successful compilation and selected CPU tests for that older release. Those results do not verify current authority/witness changes. Current parallel/src/gp.rs:643 now delegates fitting to fit_boundary::fit, which requires the service socket and provenance (parallel/src/fit_boundary.rs:58-86). This is implementation progress, not deployed custody for older observations.

The general engine is a Rust interpreter over DSL data, followed by retained-model prediction (vendor/feat-math/src/model.rs:35-79, :177-204). Calling the engine from a compiled Rust binary does not mean each invented candidate is compiled Rust. FullRust canonicalization emits source (:723-737), but emitted source length is not compile success or numerical equivalence. The V3 route fits a BoundModel, encodes it, loads a fitted C++ runtime, and predicts (src/golden42_v3.rs:1299-1338), which is yet another implementation route.

**Required plan:** freeze a complete release closure and trace every deployed process to it before activation. Keep parser/interpreter, emitted Rust, fitted C++ codec/runtime, and any CUDA prototype as separately qualified execution contracts. Preserve exact fixtures and original artifacts; do not reuse older passing tests as a release-wide stamp.

**Acceptance:** the original final compilation/test results bind the complete source closure and runtime, plus all fit launch routes are authority-mediated or reject before numerical execution. Historical known/ambiguous fits must never be rerun to fill verification gaps.

## 4. Root fit routes remain outside the parallel boundary

Root src/evaluation_v2.rs:1188 directly calls fit after explicit fold and identity validation. Root src/golden42_v3.rs:2256-2268 calls fit_table, and its confirmation route refits at :1351. These routes have meaningful local contract checks, but the inspected functions do not require the parallel authority token, independent event witness, or its global historical exclusion set.

The shared vendor model::fit is publicly exported and directly performs fitting (vendor/feat-math/src/model.rs:35-79). Closing only parallel fit/fit-view/evaluate commands therefore cannot certify a full-project boundary. A service fence also cannot prevent a separate retained executable or library client from calling these routes. This is a source-level bypass inventory, not a claim that each route is currently active.

**Required plan:** build a command-to-fit call graph across both workspaces, all 56 root binaries, Python launchers, imported/vendor copies, examples and integration tests. Assign every route a dispatch rejection or authority adapter. Pure parser/emitter tests can remain separate from numerical execution, but test entry points using original real data require the same historical exclusions.

**Acceptance:** adversarial invocations cover original and renamed executables, alias forms, root and embedded source copies, direct workers and library clients. Every denied invocation must leave numerical-start counters and original receipt counts unchanged; authorized synthetic invocations require a single original completion and durable witness acknowledgment.

## 5. Controls answer different questions

| Control | Source construction | What it can isolate | What it cannot establish alone |
|---|---|---|---|
| Generic grammar | parallel/src/ranked_proposals.rs:227-240, :1020 | Handwritten grammar/population baseline | A randomly initialized neural network baseline |
| Random ordering | parallel/src/ranked_proposals.rs:1016-1018 | Ranking benefit on the same candidate set | Generation benefit or equal candidate diversity |
| Shuffled context | parallel/src/proposals.rs:75-84, :113; ranked_proposals.rs:1005 | Dependence on matching dataset conditioning | Absence of globally learned program priors |
| Full GP with neural seeds | parallel/src/gp.rs:835-870, :1077-1098 | Entire seeded-search pipeline benefit | Direct policy quality separated from teacher slots and later mutation |
| Pure predicted winner | parallel/tools/continuous_quality_report.py:95-113 | Policy selection among its frozen pure pool when actually evaluated | Missing predictions or failed fits counted as successful evidence |

The shuffled donor is selected from a distinct training family, while the network, grammar, population assembly and target-side retained teachers remain informative. A tie with correctly conditioned proposals can therefore mean weak conditioning, dominant shared priors, weak statistical power, or search/teacher masking. It is not proof that the network learned nothing. Conversely, an advantage over generic alone cannot isolate learned ranking when candidate generation differs.

Current source separates pure pools from retained teachers and later GP in reporting, and preserves unevaluated candidates as missing. That is useful infrastructure. Its availability does not certify that every historical result was generated with the same protocol. Keep each arm's checkpoint, candidate set, seed, actual exposure, time, failures, reuse, and selection rule bound to original receipts.

**Acceptance:** compare same-pool neural/random/shuffled ranking and independently compare neural/generic generation under matched budgets. Report direct predicted winner, best evaluated pure proposal, retained teacher and final evolved winner separately. Bootstrap/inference units must preserve source-family dependence; repeated cuts or generations of the same family are not independent discoveries. Do not fund perpetual controls without an explicit unresolved hypothesis and reservation.

## 6. Adaptive teaching scores and cross-dataset units cannot be promoted into win claims

Current observation construction correctly labels repeated cut-heldout optimization as adaptive teaching and independent_evaluation=false (parallel/src/gp.rs:1343). Current evaluator reports normalized cut-target RMSE (:647-651), then search rescales by the training preprocessing target scale (:1310). The retained deployed source does the same rescaling (:938). Thus best_validation_rmse is original-target RMSE, and comparing its absolute value across datasets is not a common quality scale.

Optimizing a noisy finite-sample selection score can produce substantial selection overfitting even when the score itself is a familiar validation criterion.[2] A thousand candidate comparisons on one heldout cut can create useful training observations, but that cut is no longer an untouched test for the resulting champion.

For a heldout cut, define B as RMSE of its actual training-target mean on heldout targets. Report E=original-unit RMSE, E/B, and 1-(E/B)^2 when B>0. Ordinary R² uses the heldout-target mean in its denominator and must be separately named. When B=0, the ratio/skill is unavailable; E=0 is still valid perfect-error evidence. These are definitions, not measured results from this audit.

A reporting gap remains: continuous_quality_report.py:117-136 attaches baselines only from an observation baseline field or embedded train_targets/validation_targets arrays. It also passes raw metrics.rmse to cut_baseline.annotate_observation, whose default unit is cut_normalized_target; the report separately declares overall original-target units (:87-92). The inspected native observation constructor does not supply baseline_rmse. Therefore baseline availability and units cannot be assumed complete merely because the helper exists. No target arrays were opened to infer a missing baseline.

Root task-rating wins compare defined policy utility on common contexts (src/task_rating.rs:455-485). Such a win is a ranking of measured task utilities, not a literal self-play game winner and not proof of an independent policy improvement. Preserve the exact utility definition, missing-slot policy and family grouping in any Elo-like headline.

**Acceptance:** native producer emits actual-cut baseline and explicit units from already-authorized training/validation execution; independent verifier recomputes arithmetic from retained permitted artifacts. Synthetic fixtures cover zero error, zero baseline, shifted heldout mean, target rescaling, failed fits, wrong dimensions, missing rows, and equal/near-equal score ties. Protected evaluation access remains separately authorized and immutable.

## 7. Strict finite outputs do not imply unmodified mathematical semantics

The engine sanitizes the feature matrix before fitting and prediction, and symbolic evaluation replaces expression errors/nonfinite values with zero and clamps final output (vendor/feat-math/src/model.rs:58, :83-88, :195-204). Math expressions also use bounded power/exp arithmetic (vendor/feat-math/src/math_expr.rs:700, :740). The engine has MAX_WIDTH=512 (vendor/feat-math/src/lib.rs:80).

These choices can be legitimate frozen semantics. They must be part of the language contract and parity qualification, not silently interpreted as exact unrestricted mathematical operators. A final finite RMSE does not prove every intermediate was finite. Changing sanitization, clipping, width handling or target transforms is a semantic change that requires new fit identities and objective/corpus comparisons.

**Acceptance:** a source-bound semantic table records each exceptional-value rule and capacity limit, with parity fixtures spanning singular matrices, extreme values, all-constant columns, target transforms and malformed models. Existing known fit artifacts remain the comparison source; no historical fit is repeated merely to regenerate a fixture.

## 8. Current weighted parent selection does not randomize adjacent proposal seeds

A concrete local implementation defect appears in parallel/src/gp.rs:532: the selection quantile is `proposal_seed as f64 / u64::MAX as f64`. The caller constructs proposal_seed by adding the proposal counter to rng_seed (:1073), without passing it through a random generator or hash. Adjacent proposal counts therefore barely change the quantile across one search. With small seeds the quantile stays near zero and selects the first positive cumulative weight; with large seeds f64 conversion can map many adjacent integers to exactly the same value. This is deterministic behavior, but determinism does not make it a valid pseudorandom categorical draw.

The metadata example in root's paired diagnostic has rng_seed=8129661450841087504. A direct arithmetic audit of the first 1,000 seed increments yields one distinct f64 quantile, 0.4407098303286783; the total span is zero. This arithmetic is not a fit or a search result. The exact output is retained in findings.json. Population weights can change while this near-fixed quantile remains, so the observation does not claim one unchanging genome is always selected.

A second mismatch appears at :527: ranked offspring without a stored log probability receive zero, hence weight one. Original normalized log-probability weights are at most one and typically much smaller. The comment promising historical uniform fallback does not describe that mixed case. The current test called unmatched_seeds_keep_uniform_parent covers only missing scores/alignment, not ranked genomes absent from a partially scored map (:566). The retained native-v5 source contains no policy_weighted_parent function, so this finding is about current unqualified source, not an explanation established for generation 66.

**Required plan:** use a deterministic RNG/hash stream with domain separation to turn the request seed and proposal index into a uniform variate. Define the unscored-offspring policy explicitly: re-score them, use a qualified mixture, or fall back uniformly over the whole survivor set. Do not silently treat log probability zero as uncertainty. Keep original search lineages unchanged.

**Acceptance:** pure, no-fit fixtures exercise small and large search seeds, sequential proposal indices, skewed weights, reordered survivors, partial score maps and absent scores. A bounded frequency check must match the specified categorical distribution across many deterministic draws; source inspection and deterministic replay must confirm identical requests reproduce identical draws. Existing three handpicked parent tests are insufficient.

## 9. Generation-66 ties support a narrower conclusion

Root's [paired-diagnostic.json](evidence/paired-diagnostic.json) selects the latest complete report at capture without nominating a candidate. It joins 140 context groups/420 arm rows. It records 120/124 training ties against generic and 120/124 against shuffled; all 16 development groups tie in both comparisons. It also records zero available baselines, 420 pure predictions marked protocol_unavailable, and unequal completed evaluations in 75/124 training pairs against generic, 76/124 against shuffled, and 15/16 development pairs for each opponent. The diagnostic explicitly says original source receipts were not independently reverified, independent_evaluation=false and promotion_evidence=false.

A shared strong retained teacher can saturate final-best GP comparisons while direct ranking or proposal quality differs. Current source intentionally places incumbent/retained teachers first for new searches (parallel/src/gp.rs:486-497, :957-961). However, equality of final RMSE does not itself prove the same teacher won; original winning-program hashes, origins and complete attempt identities must be joined to establish that explanation. Unequal evaluation counts can be a valid consequence of equal wall-time budgets, but they prevent describing the observations as equal realized numerical work without further receipts.

The supported conclusion is low observed separation of this adaptive final-best endpoint in this selected generation. It does not establish global absence of neural signal or invalidate all learning. Distinguish the endpoint's sensitivity from population diversity, pure winner quality, probability of improving the retained teacher, quality-versus-time, and genuinely new admitted discoveries. Require the missing pure and baseline producer evidence before extending a win-rate interpretation.

## Interrupted implementation status

Before the switch to planning, an owned query-prediction child was added in current dirty source, with sealed descriptor/response FDs, process/executable/runtime checks and process-group cleanup. Main command wiring was performed by root. This change is incomplete and must not deploy.

The original narrow unit run completed with 8 passed and 1 failed, 100 filtered out. The failing test expected `refitting is forbidden` in the missing-model-reader error; the new in-memory test adapter omitted that context. The new child module compiled in test configuration. The service child path, hard-deadline behavior, production build, process-loss behavior, and final integrated matrix have not been verified. No successful final test result is implied by the partial compile.

Production query access and namespace activation remain explicitly unavailable in current source (parallel/src/eval_registry/query.rs:82; authority.rs:2193-2212). That closure is appropriate until the original access, release, synchronization and funding verifiers are qualified. A disabled operation is not a completed scientific producer.

## Proposed priority and stop conditions

1. Keep root containment and resident custody; finish original history and unknown-outcome reconciliation before any numerical resumption.
2. Freeze the language/route/deployed-source capability matrix, including unsupported entries and the permanent no-rerun fixture exclusions.
3. Complete native witness/authority/query custody and reject every other fit route. Qualify the interrupted child path with bounded synthetic work only.
4. Repair producer-level baseline/status evidence and distinguish adaptive teaching from independent query quality.
5. Run explicitly funded causal controls that distinguish generation, ranking, conditioning, retained-teacher benefit and evolution.
6. Demonstrate policy-to-discovery-to-policy improvement with independently accepted checkpoints and original receipts. Consider grammar or MCTS extensions only as separately reserved experiments.

Stop new dispatch when original fit identity, witness, funding, deadline, loaded runtime, output custody or historical disposition is unavailable. Do not create work to fill utilization. Do not repair scientific evidence by replaying known or ambiguous original fits.

## Sources

[1] Silver, D. et al. (2017). [Mastering the game of Go without human knowledge](https://discovery.ucl.ac.uk/id/eprint/10045895/). Nature 550:354-359. Author manuscript and abstract hosted by UCL. Used only for the policy/value, self-play and search-improvement mechanism comparison.

[2] Cawley, G. C. and Talbot, N. L. C. (2010). [On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation](https://www.jmlr.org/papers/v11/cawley10a.html). JMLR 11:2079-2107. Used for the adaptive selection-bias distinction.

Local/retained-source identities and individual finding confidence are retained in [findings.json](authority-gp/findings.json). Remote paths and line excerpts refer to the frozen source root shown in [remote-source.json](authority-gp/remote-source.json). Current local line references bind the recorded hashes; concurrent integration edits can move line numbers.
