# CLOVER vNext — Qualified Program Discovery and Learned Search Allocation



Final revised concept | October 12, 2026
Status: Research proposal, not an approved implementation or demonstrated improvement
Basis: Supplied engineering snapshot, five candidate proposals, initial synthesis, and critical review. No independent verification has been performed.



## 1. Executive proposal



CLOVER vNext is a budget-constrained system for discovering, verifying, selecting, and delivering executable machine-learning programs for tabular regression.



The system consists of two independently testable capabilities:



1. CLOVER-Discover: A qualified program-discovery engine that uses conventional genetic programming (GP), reproducible execution, independent semantic checks, and controlled evaluation to identify useful programs beyond the existing catalog.
2. CLOVER-Learn: A learned evaluation-allocation controller that attempts to improve which programs are evaluated, and eventually which candidate programs GP generates, using qualified historical and fresh feedback.



The central architectural decision is to separate discovery from learning. CLOVER-Discover must provide a useful baseline without a neural controller. CLOVER-Learn is accepted only when controlled experiments establish incremental performance or efficiency benefits over that baseline.



This directly addresses the critique's strongest objection: the original synthesis treated expanding program support, learning from measured outcomes, and feedback-guided search as though they were already one coherent improvement mechanism.



They are instead three distinct hypotheses:



- H1 — Discovery: Qualified GP can repeatedly find useful programs outside the original catalog.
- H2 — Selection: A learned selector can choose more useful candidates from identical candidate pools than conventional selection methods.
- H3 — Feedback: Fresh verified outcomes improve selection or subsequent discovery beyond frozen models, offline replay, and optimizer restarts.



A fourth, conditional hypothesis is that learned guidance can improve GP candidate generation itself.



The system remains valuable if H1 succeeds but H2–H3 fail. In that case, CLOVER should remain a verified GP discovery engine rather than claim to be a self-improving reinforcement-learning flywheel.



## 2. Why this direction was selected



The supplied snapshot reports a system with a 6,640-program fixed trie, a small PolicyLM, expensive native/Rust verification, sparse incumbent-beating observations, and no established improvement from the reported learner experiments.



In particular:





| Reported observation | Implication for the proposal |
| --- | --- |
| Original policy restricted to 6,640 catalog programs | Improving the loss alone cannot discover programs outside the current support |
| q-head pairwise accuracy approximately 0.506 | Candidate ranking usefulness has not been established |
| Positive PKPO surplus in 47 of 2,267 training contexts | Strong-incumbent policy-gradient signal is sparse |
| Frozen v1 validation L6 of 0.897327 versus 0.899279 and 0.900328 for two challengers | The reported challengers do not establish better validation loss |
| DPO β=0.1 reportedly violates the 0.05 KL guard | More preference optimization is not justified as the immediate priority |
| Quantile-binning parity defects and inconsistent protocol identities | Evaluation semantics must be corrected before results are pooled |
| Significant executable-storage and custody overhead | Full-cost search efficiency matters alongside model quality |





These are input-reported observations, not independently authenticated results. The previous candidate scores are likewise model judgments, not measurements; their close rankings provide no defensible basis for declaring a precise winner.



The revised proposal combines the strongest elements of the five candidates while changing their integration.





| Candidate | Selected contribution | Modification |
| --- | --- | --- |
| 5 — Evidence-qualified expert iteration | Verification, provenance, authority separation, promotion discipline | Convert governance principles into enforceable experiment invariants |
| 4 — Budget-first iteration | Capped pilots, measured throughput, backpressure, stop rules | Require complete resource accounting, not nominal attempt limits alone |
| 3 — Expanding-support iteration | Bounded GP discovery beyond the trie | Separate generator, support registry, and selector |
| 1 — Engineering workbench | Reproducible executable delivery and incumbent comparisons | Treat practical utility as independent from learning success |
| 2 — CLOVER-DELTA | Parent-child edit-gain modeling | Retain as a conditional experiment after ranking learnability is established |





Selection rationale: This architecture maximizes scientific identifiability while retaining a practical fallback. It does not assume that a neural model is necessary for successful program search.







## 3. Proposed architecture







Task + versioned campaign contract



Dataset family · permissible data · incumbent · evaluator · support · budgets · independent judge















1. Discovery



Catalog retrieval and bounded typed GP create candidate programs











2. Selection



Simple baseline or learned controller decides evaluation priority















3. Verification and evidence



Semantic checks · Native/Rust parity · complete attempt accounting · immutable receipts















4. Learning



Authorized training feedback only; versioned models and support











5. Independent judging



Locked tasks, resource-matched comparisons, selected-program quality















Qualified executable handoff + decision



Select challenger · Retain incumbent · Unresolved



Logical architecture only. No current deployment or integration is implied.



### 3.1 Discovery authority



The discovery engine produces candidate programs through existing qualified catalog methods and bounded typed GP operations.



Its responsibilities are candidate construction, type validation, canonicalization, identity assignment, semantic-duplicate detection, and search provenance.



It must be useful independently of the PolicyLM.



Two separate identities are maintained:



- A program identity based on canonical code and evaluator-relevant dependencies.
- A semantic equivalence record based on validated program behavior or proven canonical equivalence.



Different program hashes alone do not establish meaningfully different programs. Approximate behavioral fingerprints are useful for screening, but must not be treated as mathematical proof of equivalence.



### 3.2 Support registry



The registry contains programs eligible for selection under a particular campaign.



Every support version binds the grammar, tokenizer, canonicalizer, evaluator compatibility, program identities, admission criteria, and qualifying evidence.



An admitted program may be eligible for sampling without having an estimated reward. Conversely, a historically measured program is not automatically eligible for a new evaluator protocol.



Support expansion must never be silently interpreted as policy improvement.



The existing catalog, expanded candidate pools, and new learned selectors receive separate version identities.



### 3.3 Verification authority



Every disclosed candidate evaluation creates a durable attempt record with its selection rule, inclusion probability where applicable, execution status, consumed budget, evaluator identity, and provenance.



Results distinguish qualified completion, known failure, timeout, parity-unqualified execution, and infrastructure unknown.



Native/Rust parity remains a required qualification check where applicable, but implementation agreement alone is insufficient. Independent semantic oracles and numerical edge-case tests must also be used to detect shared errors.



An infrastructure unknown cannot be turned into an artificial regression reward. Its attempt budget remains charged, and incomplete-outcome uncertainty remains visible in analysis.



### 3.4 Learning authority



The first learning task is candidate ranking, not autoregressive generation or unrestricted reinforcement learning.



Given a dataset context cc, candidate program aa, and relevant incumbent information, an outcome model estimates the probability and magnitude of useful improvement, together with execution feasibility and resource cost.



An initial predictor may estimate:



q^(c,a)\\widehat q(c,a)



but it should be assessed against simpler alternatives, including retrieval, regularized regression, tree-based predictors, existing GP ranking, and the reported PolicyLM q-head.



A second-stage model may estimate:



Δ^(c,a,hc)\\widehat\{\\Delta\}(c,a,h\_c)



the candidate's predicted improvement over a fixed comparable incumbent.



The eventual controller may choose the next program based on expected incremental discovery value, uncertainty, execution cost, already observed results, and remaining budget.



No specific neural architecture is assumed to be superior in advance.



### 3.5 Judging authority



The judging process remains independent of generator and learner updates.



It uses predeclared task populations, lineage exclusions, evaluator versions, frozen incumbents, resource limits, decision rules, and permitted statistical looks.



Judging results must not flow back into model selection or training unless the associated judge is permanently retired from confirmatory use.







## 4. Exact learning and selection contract



This section addresses the critique that the initial synthesis used technically suggestive phrases without specifying an actual learning mechanism.



### 4.1 Frozen candidate pools



For the first learned-selection experiment, construct a candidate pool PcP\_c independently of the selectors being evaluated.



All tested selectors face the same eligible pool, evaluator, incumbent, and attempt budget.



For each proposed evaluation, record:



(c, a, Pc, πb(a∣c,H), H, t, status, cost)(c,\\ a,\\ P\_c,\\ \\pi\_b(a\\mid c,H),\\ H,\\ t,\\ \\text\{status\},\\ \\text\{cost\})



where πb\\pi\_b is the behavior selection probability when randomized, HH is the available observation history, and tt identifies the decision position.



Deterministic selectors record their exact rule and version. They are not assigned fictitious nonzero propensities.



### 4.2 Exploration and observation bias



Historical GP outcomes are preferentially observed, so simple regression on historical successes can reproduce archive selection bias.



The initial experimental data-collection policy therefore includes a randomized allocation that gives the intended candidate strata measurable coverage.



The allocation is selected from an explicit tradeoff among evaluation cost, candidate-family coverage, effective sample size, and intended deployment support.



It is not assigned an arbitrary universal percentage.



Observed propensities support weighting or off-policy diagnostics only where positivity and adequate overlap hold. Where they do not hold, conclusions are restricted to covered support or require fresh randomized observations.



### 4.3 Outcome modeling



The initial learner fits three conceptually separate targets:





| Prediction | Required treatment |
| --- | --- |
| Qualified predictive outcome | Train only on comparable qualified rewards |
| Completion, failure, timeout, and parity outcome | Model execution status separately |
| Resource cost | Model and evaluate runtime and custody expenditure, including censoring and timeouts |





Context-dependent reward normalization must be fitted using authorized training observations and frozen before validation.



Calibration is evaluated across task families, incumbent-strength strata, and candidate types, not solely across pooled observations.



### 4.4 Initial selector



The simplest learned selector ranks eligible programs by predicted reward or incumbent-relative improvement, subject to eligibility, completion, and budget constraints.



It is compared against selectors trained on the same measured data, including inexpensive non-neural outcome predictors.



More sophisticated acquisition scores can be introduced after the ranking baseline is established. An illustrative score is:



A(a∣c,H,b)=E\[(Qa−hcurrent)+∣c,H\]−λ(b) E\[Ca∣c,H\]A(a\\mid c,H,b)
=
\\mathbb E\[(Q\_a-h\_\{\\mathrm\{current\}\})\_+ \\mid c,H\]
-
\\lambda(b)\\,\\mathbb E\[C\_a\\mid c,H\]



where QaQ\_a is a predictive outcome distribution, CaC\_a is execution cost, and bb represents remaining budget.



This is a proposed acquisition heuristic, not a proven unbiased optimizer.



### 4.5 Policy improvement and distillation



Measured-outcome distillation remains optional.



If used, it must declare the frozen support, behavior distribution, eligibility of measured outcomes, reward normalization, duplicate treatment, reference distribution, and weighting or clipping behavior.



An AWR-style target fitted to a selectively measured archive is not automatically an unbiased policy-improvement estimator.



The system should first demonstrate that outcome prediction produces useful selection decisions. Only then should policy distillation, RLOO, PKPO, or preference optimization compete against that simpler approach.



When support expands, introduce an approved reference distribution with positive probability on the new support, and establish a new versioned stability protocol. The old trie-only KL threshold cannot automatically be reused.







## 5. Objectives and evaluation metrics



The critique correctly identifies that best-of-eight discovery reward, independent predictive quality, and resource efficiency are different objectives.



CLOVER will report all three rather than collapsing them into a single score.



### 5.1 Fixed-attempt discovery



For a context with a comparable, frozen incumbent hch\_c, the established discovery statistic is:



S8(c)=max⁡\{hc,q(c,a1),…,q(c,a8)\}−hcS\_8(c)=\\max\\\{h\_c,q(c,a\_1),\\ldots,q(c,a\_8)\\\}-h\_c



The eight slots are charged disclosed attempts, not necessarily eight unique programs or eight successful executions.



The qualifying sampling protocol must specify replacement, duplicate handling, cache hits, failures, and stopping behavior.



For an initial protocol, duplicates consume attempt slots even when their executions can be served from a compatible cache; actual compute and custody work are accounted separately.



Where infrastructure unknowns prevent exact scoring, do not fabricate qq. Report evaluable results, unknown incidence, and predeclared sensitivity bounds. If defensible bounds cannot be established, label the affected comparison inconclusive.



The incumbent is frozen per context and experiment. When no qualified incumbent exists, the context uses a separately defined no-incumbent protocol instead of substituting zero.



### 5.2 Independent selected-program performance



The principal scientific benefit is measured on the program actually selected by the search procedure.



For example:



D(c)=Lind(c,ainc)−Lind(c,aselected)D(c)=
L\_\{\\mathrm\{ind\}\}(c,a\_\{\\mathrm\{inc\}\})
-
L\_\{\\mathrm\{ind\}\}(c,a\_\{\\mathrm\{selected\}\})



where LindL\_\{\\mathrm\{ind\}\} is a predeclared loss computed on independent assessment data.



Positive DD indicates better predictive performance when lower loss is preferable.



This assessment must exclude data used to construct the candidate, fit its preprocessing, tune its parameters, select it, or choose the learning method.



This prevents optimistic inner-cross-validation winners from being mistaken for independently improved programs.



### 5.3 Resource-adjusted performance



Measure selected-program performance and discovery success at both:



- The fixed eight-attempt budget.
- A fixed elapsed-time or fully accounted resource budget.



Resource accounting includes program construction, evaluator CPU/GPU time, compilation, cache behavior, queue delay, S3 requests and bytes, retries if authorized, failed attempts, checkpointing, and evidence retrieval.



Where relevant, report human intervention separately.



A system cannot claim greater efficiency solely because it runs fewer neural forward passes while consuming more total evaluation resources.



### 5.4 Practical acceptance criteria



Each confirmatory study must specify a minimum practically important effect, relevant noninferiority margin if efficiency is the intended benefit, maximum acceptable failure behavior, and resource ceilings.



Statistical positivity alone is insufficient.



An improvement can be real but too small to justify engineering or operating cost.







## 6. Corrected experimental design



The original six-arm design is replaced by two nested experiments, followed by a separately controlled feedback study.



### Experiment A — Can learning select better programs?



Question: Given the identical candidate pool, does learned selection improve results compared with inexpensive conventional selectors?



Common candidate pools are constructed and frozen before selector evaluation.





| Selector | Candidate pool | What it tests |
| --- | --- | --- |
| Uniform | Identical frozen pool | Random-choice baseline |
| Retrieval | Identical frozen pool | Similar-task reuse |
| Native GP ranking | Identical frozen pool | Conventional search prioritization |
| Frozen PolicyLM | Identical frozen pool | Value of existing learned representation |
| Simple supervised predictor | Identical frozen pool | Outcome learnability without elaborate RL |
| Proposed learned selector | Identical frozen pool | Incremental selection benefit |





Where ground-truth candidate outcomes are incomplete, evaluation uses newly qualified attempts or a predeclared coverage-aware design; it must not treat unmeasured candidate values as known.



Pass condition: A learned selector demonstrates practically meaningful independent selection or discovery improvement against the strongest relevant simple baseline, with acceptable cost, uncertainty, and failure behavior.



A neural selector does not pass merely by beating uniform choice.



### Experiment B — Does neural guidance improve discovery?



Question: Does learned guidance improve GP search itself, rather than merely select from an already-generated pool?



Compare ordinary GP with policy-guided GP under the same grammar, candidate-construction rules, permitted data, evaluator, and resource accounting.



The comparison must disclose whether the learned controller affects parent selection, mutation choice, candidate prioritization, or multiple components. These are different interventions.



At minimum, the ordinary-GP comparison receives equivalent opportunities to generate and evaluate programs. Also compare under equal wall-clock or full-cost budgets.



Pass condition: Neural guidance shows independent improvement in selected-program quality, or comparable quality at a predeclared meaningful cost reduction.



### Experiment C — Does fresh feedback cause improvement?



This experiment begins only after ranking learnability has been established.



Use identical candidate support, qualified training data available before randomization, initialization parameters, optimizer state, and applicable training schedules.





| Arm | Intervention |
| --- | --- |
| Frozen | No new learning |
| Restart-only | Same scheduled optimization without new feedback |
| Offline-data control | Continue using preexisting qualified observations |
| Fresh-feedback treatment | Add specifically authorized newly collected training outcomes |
| Optional matched-data control | Equalize observation quantity or exposure to distinguish freshness from data volume |





The design must distinguish a learning-rate restart, additional optimization, additional observations, their recency, and changed program support.



If fresh data collection is policy-dependent, a simple common batch-order requirement cannot fully equalize arms. Instead, the collection protocol and treatment estimand must explicitly account for the changed behavior distribution.



Pass condition: Feedback produces an incremental, independently assessed improvement over matched controls. If it does not, retain the simplest useful offline selector.







## 7. Staged development and decision gates













Phase 0 — Establish comparability



Reconcile source hashes, checkpoint initializations, objective definitions, evaluator semantics, task roles, label boundaries, support identities, and effective resource caps. Validate persistent HOLD and recoverable evidence.



Exit: One fully qualified, reproducible baseline and a consistent accounting contract.



















Phase 1 — Prove useful support expansion



Run bounded ordinary GP on authorized discovery families. Measure semantically distinct new programs, repeated incumbent improvements, parity qualification, diversity, and complete discovery costs.



Exit: Repeated useful expansion, not just one new hash or one favorable inner-fold score.



















Phase 2 — Establish ranking learnability



Generate frozen candidate pools, collect coverage-aware measurements, and evaluate simple predictors versus PolicyLM-based selection.



Exit: Evidence that candidate ranking improves actual selection, not merely predictive training metrics.



















Phase 3 — Isolate fresh-feedback value



Compare frozen, restart-only, offline-data, and fresh-feedback learners under compatible support, collection, and evaluation contracts.



Exit: A measurable feedback contribution distinguishable from support changes and ordinary additional optimization.



















Phase 4 — Test guided discovery



Permit the controller to influence GP evaluation allocation or mutation selection. Compare with ordinary GP using identical permitted operators and matched resources.



Exit: Independent evidence that guidance improves program discovery or lowers its total cost.



















Phase 5 — Independent qualification and handoff



Evaluate frozen contenders on genuinely unexposed families, assess the program actually selected, examine uncertainty and heterogeneity, and verify complete artifacts and resource accounting.



Exit: Approval-quality evidence for a narrowly specified benefit—or a documented inconclusive or no-practical-lift result.









The stages are gates, not fixed-duration commitments. There is no requirement to advance through all of them.



## 8. Statistical and evidence safeguards



The original synthesis treated a reported 881-context planning threshold too prominently. The revised approach discards any universal sample-size threshold.



Confirmatory power depends on the number of independent dataset families, the parent/descendant dependence structure, variation in paired effects, expected practically important effect, number of contender comparisons, and planned interim looks.



The experiment registry must distinguish discovery data, training feedback, development validation, locked confirmation, and final delivery assessment.



Previously inspected families cannot become genuinely untouched merely through reshuffling or new random splits.



When sufficient independent families are unavailable, results must remain exploratory. Nested validation may improve estimation discipline but does not manufacture unseen source-family diversity.



Family-clustered inference is preferred when the sampling design supports it. Report the family count, context count, seed variability, paired effect distribution, and sensitivity to family weighting.



Promotion also requires controlled multiplicity and predeclared statistical looks. A lower confidence bound above zero is not enough unless it clears the study's practical and safety requirements.



Where evidence is too imprecise to distinguish useful benefit from negligible benefit, state inconclusive rather than claiming success or proof of no effect.







## 9. Operational integrity and storage



The verification system is a necessary foundation, but the critique correctly identifies failure modes that can contaminate both resource and scientific comparisons.



### Non-negotiable controls



- Evaluator correctness: Repair the reported quantile semantics through authorized versioning and independent oracles. Preserve old evidence under its original evaluator identity.
- Data-boundary safety: Audit the reported target-supervised encoder and every cached embedding for role, split, fold, label, and normalizer compatibility.
- Persistent HOLD: A guard violation blocks adoption and promotion, including after restart or partial recovery. HOLD cannot be only a dashboard warning.
- Attempt completeness: Preserve failures, timeouts, unknowns, duplicate attempts, cache events, and full consumed budgets.
- Immutable evidence: Use versioned content-addressed blobs and complete manifests with verified read-back and manifest-last publication.
- Protocol-aware caching: Reuse scores only when evaluator, prepared input, program identity, and dependency identities are compatible.
- Controlled resources: Isolate experiments when feasible; otherwise log contention, queueing, and cache conditions so apparent speedups are not attributed incorrectly.



The supplied evidence reports a conflict between specification limits of 25 GB/day and 1.5 million PUTs/day and implementation limits of 200 GB/day and 2 million PUTs/day.



Until resolved by the authorized owner, the stricter specification limits should be treated as proposed planning constraints, not as evidence that current code enforces them.



Immutable evidence packing should be investigated for qualified throughput and request savings, but no deletion, expiry, lifecycle transitions, or custody changes are authorized by this proposal.



Projected request reductions and twofold throughput improvement are engineering targets, not established benefits.







## 10. Tradeoffs and alternatives considered





| Choice | Benefit | Cost or sacrifice | Decision |
| --- | --- | --- | --- |
| Conventional GP first | Strong baseline and immediate discovery utility | Does not inherently learn from feedback | Select |
| Learned selector before learned generator | Identifiable ranking benefit; simpler experimental design | Does not initially improve program construction | Select |
| Full expanding-support PolicyLM generation | Potentially more expressive | Support, likelihood, and transfer behavior unproven | Defer |
| Learned edit-gain model | Local paired improvement structure | Requires valid edit histories and transfer evidence | Conditional |
| Cost-aware sequential acquisition | Closest to actual evaluation-allocation decision | Adaptive evaluation complicates causal and statistical analysis | Research extension |
| AWR or measured-outcome distillation | Reuses qualified observations | Vulnerable to coverage and archive-selection bias | Conditional |
| RLOO/PKPO policy gradients | Targets incumbent-relative performance | Reportedly sparse signal; estimator assumptions require testing | Defer |
| DPO on the existing trie | Relatively direct preference training | Reported guard violations and no established winning result | Reject as immediate priority |
| MCTS, edit-MDPs, league self-play | Richer planning or adversarial search | No demonstrated need or suitable current game structure | Defer |
| Engineering workbench without learned improvement | Practical, reproducible output | Less algorithmically novel | Preserve as fallback |





The main tradeoff is slower progression toward an ambitious self-improving system in exchange for substantially stronger causal attribution and lower risk of investing in unsupported learning mechanisms.



That tradeoff is deliberate.



## 11. Unresolved assumptions and remaining disagreements



The critique does not eliminate uncertainty about the best ultimate learning algorithm. The final proposal preserves the following disagreements explicitly.



### A. Should the model predict absolute reward, edit gain, or marginal evaluation value?



Resolution: Absolute quality or incumbent-relative gain prediction is the lowest-complexity initial test. A cost-aware marginal-value controller is the more operationally aligned long-term target.



Remaining disagreement: An edit-gain model may prove more sample-efficient than either, particularly for local GP search. There is not yet sufficient evidence to choose it as the default.



### B. Should PolicyLM remain the primary learned representation?



Resolution: Treat it as one candidate model, not the assumed winner.



Remaining disagreement: Its existing task representation may transfer effectively once sufficient measured outcomes exist—or may be inadequate for program-level generalization. The reported q-head result cannot distinguish these explanations.



### C. Is catalog expansion the binding problem?



Resolution: Test ordinary GP expansion directly.



Remaining disagreement: The current catalog may be restrictive, but strong incumbents, low reward variance, biased measurement, or weak search operators may explain much of the apparent saturation.



### D. Should incumbent-relative best-of-eight remain primary?



Resolution: Retain it as the fixed-attempt mechanistic discovery measure, while independently evaluated selected-program performance becomes the principal scientific benefit.



Remaining disagreement: Other deployment objectives may reasonably prioritize noninferior accuracy at substantially lower cost. These should be separately preregistered, not mixed opportunistically.



### E. How much randomized exploration is justified?



Resolution: Choose it through coverage, variance, expected value, and cost analysis.



Remaining disagreement: Exploration may be too expensive to obtain reliable overlap for broad off-policy claims. In that case, restrict inference to smaller eligible supports.



### F. Is an independent judge feasible?



Resolution: Require genuinely unexposed family-level confirmation for generalization claims.



Remaining disagreement: The supplied material does not establish that enough unexposed, legally usable families exist or that the effective family-level power will be adequate.



### G. Can custody economics support the research?



Resolution: Instrument marginal qualified costs before authorizing large campaigns.



Remaining disagreement: Caching, evidence packing, execution batching, and reproducible builds may improve economics, but the achievable savings and implementation cost remain unknown.



### H. How novel is CLOVER?



Resolution: Make no blanket algorithmic novelty claim. Versioned evidence, verification boundaries, expanding-support selection, and controlled feedback experiments form a coherent systems approach.



Remaining disagreement: Whether a particular acquisition policy or feedback mechanism represents publishable algorithmic novelty requires separate literature review and empirical comparison.







## 12. Concrete next steps



The next actions are proposed work items, not authorization to execute them.





| Priority | Work item | Required deliverable or acceptance criterion |
| --- | --- | --- |
| P0 | Create a single experiment identity contract | Full source, initial state, evaluator, support, objective, data roster, seed, budget, and guard identities; reject conflicting lane labels |
| P0 | Audit evaluation semantics and data boundaries | Independent quantile fixtures, versioned parity contract, encoder-label visibility audit |
| P0 | Validate fail-closed operation | Fault-tested HOLD, recovery, unknown terminalization, and immutable evidence publication |
| P1 | Measure the actual qualified baseline | Complete attempt-level throughput, queue, compilation, custody, cache, and cost metrics |
| P1 | Design bounded GP expansion study | Frozen train-family roster, declared grammar and operator set, novelty test, ordinary-GP baseline, stop conditions |
| P1 | Design ranking-learnability experiment | Identical frozen candidate pools, randomized observation coverage, simple selector baselines, independent ranking and selection metrics |
| P2 | Implement matched feedback controls | Exact intervention definitions separating restart, offline learning, new-data volume, and fresh feedback |
| P2 | Establish statistical analysis plan | Independent-family count, effect threshold, power simulation, multiplicity, sensitivity, and repeated-look rules |
| P2 | Test guided GP only after learning evidence | Equal-operator, fixed-attempt and equal-resource comparisons with conventional GP |
| P3 | Qualify executable delivery | Reproducible selected-program artifacts, independent performance report, complete provenance, and explicit selection decision |





The earliest substantive decision is whether the ordinary-GP baseline can generate useful qualified improvements at an affordable cost.



If it cannot, further policy-learning complexity has no established discovery advantage to exploit.



If it can, the next decision is whether learning improves selection on the same programs.



If that succeeds, the final causal question is whether fresh feedback improves the controller beyond ordinary supervised learning and restart effects.



## 13. Explicit falsification rules





| Hypothesis | Result that would undermine it | Appropriate response |
| --- | --- | --- |
| GP support expansion is useful | Repeated qualified discovery yields negligible practical improvement at substantial cost | Investigate grammar, search, evaluator, and incumbent limitations |
| Candidate ranking is learnable | Learned selectors consistently fail to outperform simple selectors with adequately precise results | Stop neural-selection escalation |
| Fresh feedback matters | Adequately powered, matched studies exclude practically meaningful feedback gain | Retain offline or frozen selection |
| Neural guidance improves GP | Conventional GP meets or exceeds guided GP on quality and cost | Keep ordinary GP |
| Outcome verification is economically sustainable | Qualified evidence and execution remain above authorized budgets | Simplify custody or execution before scaling |
| Generalization holds | Selection gains disappear on genuinely unexposed families | Limit claims to exploratory or narrower in-distribution settings |





An inconclusive comparison triggers reconsideration of measurement, power, and cost; it does not automatically justify more expansive modeling.







## 14. Final concept statement



CLOVER vNext: Qualified Discovery with Learned Evaluation Allocation



CLOVER is a reproducible, budget-constrained system for finding and delivering executable regression programs. It combines conventional program search, semantic and cross-implementation verification, versioned evidence custody, independent evaluation, and an optional learned controller that chooses which candidates deserve expensive verification.



The system first establishes that ordinary GP can discover useful programs beyond its original fixed catalog. It then tests whether learned candidate selection improves outcomes on identical candidate pools. Only after those hypotheses are supported does it test whether fresh feedback or learned guidance improves discovery itself.



Search quality, independent predictive performance, resource efficiency, and reproducibility are measured separately. Every claimed learning benefit must survive comparisons against simpler selectors, frozen controls, restart-only controls, and conventional GP using compatible evaluation protocols and meaningful resource budgets.



The architecture explicitly allows the simplest successful mechanism to remain the production design.



Final decision: Adopt CLOVER-Discover as the proposed foundational architecture and treat CLOVER-Learn as a sequence of falsifiable research extensions. Retain the original PolicyLM, outcome distillation, edit-gain modeling, and policy-gradient methods as candidates rather than commitments.



Current evidence status: No incremental learning improvement, independent generalization advantage, or economic advantage is established by the supplied evidence.



The decisive standard: CLOVER becomes a demonstrated learning flywheel only when qualified fresh feedback causally improves independently confirmed program discovery or selection beyond the strongest relevant non-learning baseline, at an acceptable complete cost. Until then, its defensible identity is a qualified program-discovery system with an experimental learned search controller.