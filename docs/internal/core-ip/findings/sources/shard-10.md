<!-- from-tar tab=10 url=https://chatgpt.com/c/6ab703a8-c570-83ea-878c-050dcf90bf52 sha256=5cbb6f569da488946474579f907a5dfc49f13816b9b5fdc61ceaf23c37ef7a2f chars=41523 -->

# Moonshot IP gaps and evidence-gated roadmap

**Review identity:** shard 10 of 10; batch tab 10. **Primary scope:** §§10–12 only, with dependency cross-links rather than separate reviews of other chapters.  
**Evidence boundary:** supplied structured extract, cutoff `2026-09-25T19:31:50Z`; publication source reported as `3d71fce041d1dd1173ed98160f7d986bb426ba6f`. No product source tree, retained measurement journal, raw outputs, assets, or executable was inspected or run.  
**Reading key:** DOCUMENTED means explicitly stated in the extract, not independently verified in code. UNVERIFIED means the supplied evidence cannot establish the claim. HYPOTHESIS identifies a falsifiable risk, not an observed exploit. Proposed controls, owners, experiments, and acceptance criteria are recommendations, not existing implementation facts.  
**Decision:** prioritize enforceable provenance and replayable measurement evidence before promoting GPU-performance or self-improvement claims. Continue bounded development; the extract is insufficient to approve new headline latency, acceleration, transfer, or universal-export claims (§§10–12).

**Study anchor, as reported (§10.1):** one local RTX3090; accepted step-60,000 policy; pinned encoders; FP32; ten regression datasets using cut0 fitting rows; two warmups, ten target measured warm requests, and one fresh-process observation per dataset. Full GP covers `230_machine_cpu`, `auto_insurance_price`, and `197_cpu_act` in the seed0 campaign, with at most 120 seconds and 96 candidate attempts per arm. None of these planned quantities is a completed-outcome count.

## Executive hits

- **Gap — critical admission boundary:** §11.1 explicitly says live `eligibility.rs` does not invoke the encoder-receipt validation. An opt-in validation spike cannot establish mandatory provenance enforcement for the learning loop; actual bypass exploitation remains untested.
- **Improvement — make evidence compulsory:** connect fit-scope authentication to the live admission/HWM/corpus path and exercise real entrypoints with deliberately invalid receipts. A syntactically valid record must not reach learning merely because one caller skipped validation (§11.1; §12.1).
- **Gap — staging qualification is unresolved:** §10.1 discloses an actual staging deviation, but its description is trimmed. Neither a clean held-label boundary nor actual leakage can be inferred; obtain the exception record and affected-artifact inventory before upgrading the study's status.
- **Improvement — create a replayable evidence gate:** reconstruct the frozen request inventory, qualification outcomes, deviations, and source/binary/input identities from retained records. Preserve failures and historical identities; require an explicitly new study for a corrected protocol (§10; §12.1).
- **Gap — numerical closeness is not end-to-end equivalence:** the eight encoder cases and `3e-4` maximum-absolute-error qualification in §10.2 do not alone establish stable proposal decisions, fitted-state parity, or robust deployment behavior (§11.1; §12.1).
- **Improvement — optimize a measured bottleneck:** first replay identical work, then test safe residency, transfers, decoding, fitting reuse, or compilation changes according to the observed critical path. Re-evaluate changed search policies separately under matched budgets (§10.1.1; §11).
- **Gap — scope cannot support broad moonshot conclusions:** ten regression datasets, three full-GP datasets, one seed campaign, and one RTX3090 are the stated bounded protocol, not evidence of hardware portability, independent transfer, or GPU-versus-CPU speedup (§10.1; §12.1).
- **Improvement — build a defensible integration asset:** an independently checkable package linking permitted inputs, search work, selected fitted state, runtime qualification, and claim scope would strengthen the project-specific combination. Its value must be demonstrated through successful replay and rejection of tampering, not asserted novelty (§11; §12.1; cross-link §1.1).

## Ranked IP gaps (CRITICAL/HIGH/MED/LOW)

### G1 — CRITICAL: optional provenance validation leaves a live enforcement hole
**Refs / status:** §11.1 “Reassessment against current source”; §12.1 “Claim-to-evidence map.” DOCUMENTED integration gap; exploit outcome UNVERIFIED.  
**Attack/gap story:** the audit states receipt types and binding validation exist, but live eligibility does not call them. A teacher could therefore appear eligible without this validator authenticating its upstream context. Other defenses might still reject it; the extract does not establish their coverage.  
**Detection:** exercise the live submission-to-corpus route with a valid control and separately missing, stale, wrong-fit-scope, wrong-consumer, and altered-context receipts. Inspect persisted admission, HWM, and corpus outcomes, not only validation-helper return values.  
**Harden:** require validated context provenance at the mandatory admission boundary and retain its identity through subsequent state changes; explicitly isolate historical/unproven records. Proposed owner: learning-integrity lead; kickoff K1.  
**Closure:** every invalid fixture is rejected before an eligibility-dependent mutation, the valid control proceeds, and each result has an attributable receipt. A unit test of the opt-in helper alone does not close G1.

### G2 — HIGH: the disclosed staging deviation has an unknown blast radius
**Refs / status:** §10.1 “Frozen protocol”; §10 “Observations” and §12 retained evidence, both incomplete here. Deviation DOCUMENTED; contents and consequences UNVERIFIED.  
**Attack/gap story:** a deviation might concern harmless extra staging, forbidden data accessibility, actual reads, or downstream use; these are different findings. The omitted disclosure prevents classification and tracing of potentially affected contexts, selected models, or reports. Confirmed forbidden influence would escalate this to CRITICAL.  
**Detection:** inspect the actual deviation, stage inventory, access permissions and read logs, cache lineage, and downstream artifact links. In a separate controlled replay, change only forbidden labels and compare permitted outputs against a repeatability baseline; distinguish unread accessibility from actual consumption.  
**Harden:** stage an allowlisted fitting-only closure; remove access to prohibited objects, credentials, and inherited caches. Preserve the original exception record and rerun affected work under a new identity rather than retroactively labeling it clean. Proposed owner: evaluation-security lead; K2.  
**Closure:** the deviation is classified with evidence, affected claims are explicitly restricted, and the clean replay demonstrates both denied forbidden access and unchanged permitted outputs under the counterfactual test. Missing historical logs leave historical exposure unknown.

### G3 — HIGH: qualification and headline evidence cannot be reconstructed from this extract
**Refs / status:** §10 opening, §10.2 “Qualification,” §12.1. Documentary contract present; actual benchmark outcomes UNVERIFIED.  
**Attack/gap story:** “failed qualification produces no headline latency” is a useful rule, but the retained plan, pass/fail records, run identities, and observations are absent. A downstream report could accidentally publish a complete-looking number from incomplete or differently qualified evidence; no such publication defect is demonstrated here.  
**Detection:** reconstruct qualification-to-run-to-report dependencies from retained artifacts; substitute a mismatched binary identity, missing qualification result, or failed prerequisite into a test evidence package. Verify that headline generation refuses each invalid package.  
**Harden:** make publication depend on explicit qualification records and exact measured source/binary/assets/input/protocol identities, distinct from the later publication commit. Hash agreement must not automatically mean qualified or promoted. Proposed owner: benchmark/release lead; K3.  
**Closure:** a clean verifier rebuilds permitted tables from the journal and suppresses headlines for invalid prerequisites. The current review supplies no latency, success fraction, speedup, or qualification-pass count.

### G4 — HIGH: censoring, retries, and cache ambiguity can distort reported latency
**Refs / status:** §10.1 “Frozen protocol”; §10.1.2 “Window and resume”; §12.1. Failure/resume details UNVERIFIED because the relevant span is trimmed.  
**Attack/gap story:** dropping timeouts, replacing a failed attempt with a successful retry, or omitting late backends can make the surviving requests look faster. A fresh process also does not, by itself, specify the state of filesystem, compiler, or other external caches. These are reporting risks, not observed manipulations.  
**Detection:** reconcile every planned warm request against terminal status and elapsed budget, link retries to their original identities, and inspect interrupted/resumed runs. Compare explicitly controlled process/cache states; do not assume that two warmups either guarantee or fail to provide stability.  
**Harden:** retain the stated 100-request warm denominator, report conditional successful-request latency alongside coverage and failure/censoring information, and label fresh-process observations by actual cache state. Required-backend failure semantics must be fixed before execution. Proposed owner: benchmark lead; K3–K4.  
**Closure:** injected timeout, unsupported export, interrupted write, and resume cases remain visible without double counting or silent replacement. One fresh-process observation per dataset stays an observation, not a cold-start tail estimate.

### G5 — HIGH: timing boundaries are strong on paper but not proven in execution
**Refs / status:** §10.1.1 “Timing boundaries”; §10.2. Contract DOCUMENTED; synchronization and subprocess completion traces UNVERIFIED.  
**Attack/gap story:** the stated whole-request boundary includes data reading, transfers, neural stages, fitting, all attempted backends, compilation, and first prediction. Reporting a source-ready milestone instead, summing cumulative language milestones, or ending before dependent GPU/subprocess work completes would measure a different subject.  
**Detection:** replay a known request with controlled delayed GPU work and delayed compiler/prediction subprocesses. Verify inclusion at the appropriate boundaries; use profiling only in separate diagnostic runs. Official timing guidance supports explicit synchronization and representative warmup/thread settings [E1–E2].  
**Harden:** retain one host wall-clock request timeline plus diagnostic device/stage spans and explicit dependency completion. Distinguish the study executable's off-clock build/staging from on-clock emitted-artifact compilation. Proposed owner: performance lead; K4.  
**Closure:** required work cannot escape the headline boundary; Python/Rust/C++ milestones remain labeled cumulative. Standalone backend experiments receive separate protocol identities and are not reconstructed by treating cumulative milestones as independent timings.

### G6 — HIGH: tensor-level tolerance leaves decision and artifact behavior unqualified
**Refs / status:** §10.2; §11.1; §12.1. Stated numerical gate DOCUMENTED; behavioral coverage UNVERIFIED.  
**Attack/gap story:** near a ranking or sampling boundary, small numeric differences can select different instructions even while satisfying the stated absolute tolerance. Encoder/policy checks also cannot substitute for checking the selected fitted predictor and its emitted artifacts. A comparator's handling of unexpected nonfinite values is not shown.  
**Detection:** supplement the existing cases with boundary-sensitive ranking fixtures, declared mask semantics, extreme/missing inputs, and malformed/nonfinite output tests. Follow legal-token decisions, candidate identities, selected fitted-state identity, and native/export predictions through the same receipt chain.  
**Harden:** retain the existing `3e-4` gate for its stated subject; add distinct behavioral and per-artifact qualification gates. Document legitimate stochastic variation rather than claiming bitwise cross-device equivalence. Proposed owner: numerical/runtime qualification lead; K5.  
**Closure:** deliberate decoder-mask, ordering, invalid-number, and fitted-state substitutions are caught at the intended gate. Valid numeric agreement is reported only for the tested subject and cases, not as universal semantic equivalence.

### G7 — HIGH: the bounded study cannot identify broad acceleration or search advantage
**Refs / status:** §10.1; §11 improvement opportunities; §12.1. Narrow protocol DOCUMENTED; generalization and causal advantage UNVERIFIED.  
**Attack/gap story:** the direct request uses eight sampling rounds of 128 candidates; full GP uses its separate pinned RequestPlan. Paired neural/generic arms are not a CPU/CUDA acceleration comparison. Warm repeats of a task do not create additional independent datasets, and one seed campaign cannot characterize seed robustness.  
**Detection:** demand separate task, process/session, seed, and request identities; inspect executed attempts and resource limits per arm. Require a matched CPU/CUDA comparator for acceleration claims and neural/generic or parent/challenger comparisons for their respective search claims.  
**Harden:** preserve the historical fixed protocol; register prospective counterbalanced arm-order and multi-seed experiments separately. Report quality at the declared wall-time budget and work performed, without silently treating both capped arms as having completed equal work. Proposed owner: experimental-methods lead; K7–K9.  
**Closure:** a claim names its comparator, workload, budget, analysis unit, uncertainty treatment, and scope. Repeated prediction on fitting features is never used as an accuracy or transfer result; a null result remains publishable.

### G8 — HIGH: optimization could change the work rather than accelerate it
**Refs / status:** §10.1.1; §11 ranked opportunities, trimmed; §12.1. HYPOTHESIS; the extract provides no stage profile establishing a present bottleneck.  
**Attack/gap story:** caching, batching, feasibility filters, or reordered execution can reduce observed time by reusing forbidden state, skipping difficult candidates, changing random-stream consumption, or changing which fits finish before a budget expires. A faster kernel also need not dominate a request that includes fitting and compilation.  
**Detection:** first replay an immutable candidate/work inventory and compare stage traces, outputs, logical evaluations, physical work, refusals, and random-stream identities. Then run a separately labeled adaptive-search comparison if scheduling or search semantics change.  
**Harden:** bind caches to fit scope and artifact versions; preserve refusal and incomplete-attempt accounting; select optimizations from measured critical-path evidence. Reject “speedups” attributable solely to a cheaper workload or a moved timing boundary. Proposed owner: performance/search integration lead; K6–K7.  
**Closure:** report both semantic-preserving replay performance and matched-budget live-search outcomes, with all required work charged to the relevant boundary. No kernel bottleneck or performance gain is asserted by this review.

### G9 — HIGH: a closed real-data learning-and-transfer demonstration is still missing
**Refs / status:** §11.1; §12.1; necessary dependencies §6.1.1.1 and §7.5. Transfer measurement explicitly absent in the supplied account; real-GPU learning closure UNVERIFIED.  
**Attack/gap story:** synthetic CPU update/reload tests, lower training loss, teacher eligibility, and a successful GPU request each validate different links. Combining them rhetorically does not show that authentic teachers changed a challenger and improved a subsequent independent matched search.  
**Detection:** trace one real lineage from admitted teachers and frozen corpus through actual optimizer updates and reload, then matched parent/challenger search. Separately inspect the nested transfer comparison and its inaccessible outer evaluation scope.  
**Harden:** make an operational loop demonstration and a positive transfer result two different milestones. Keep parent deployment available; require a preregistered improvement decision and guardrails before promoting the challenger. Proposed owner: learning/evaluation leads jointly; K8.  
**Closure:** real-GPU receipts can establish plumbing even when gains are null; promotion requires measured success under the separate transfer contract. Synthetic tests or changed parameter bytes alone cannot satisfy the improvement milestone.

### G10 — MED: the evidence map is not yet a self-checking, reviewer-usable integration asset
**Refs / status:** §11 “Audit findings and improvement opportunities”; §12 “Reproduction and evidence index,” especially §12.1. Sufficiency of omitted commands, access paths, and retained packages UNVERIFIED.  
**Attack/gap story:** the monograph correctly says hashes identify bytes, not evidence status. Without an executable dependency check, a source-level fix may nevertheless be mistaken for an operationally requalified result, or an unavailable private receipt may make a claim impossible for an authorized reviewer to reproduce.  
**Detection:** give a separately provisioned reviewer the declared evidence package and commands; test missing files, inaccessible dependencies, a historical result paired with a new build, and a revoked prerequisite. Record what can actually be reproduced without author intervention.  
**Harden:** maintain claim-specific evidence requirements, executable verification, access policy, and explicit impact/requalification rules. Preserve historical evidence; do not inherit a historical qualification onto a different runtime identity. Proposed owner: evidence/release lead; K10.  
**Closure:** each promoted claim has a runnable witness or a clearly stated access limitation, and invalid prerequisites block the relevant current claim. This closes an assurance gap; it does not establish patent novelty or freedom to operate.

## Ranked improvements / kickoffs (P0/P1/P2)

**Priority rule:** P0 closes integrity and measurement prerequisites; P1 tests useful mechanisms without broadening claims; P2 expands scope or creates a reusable assurance asset. Owners below are proposed functions, not inferred personnel. Every acceptance test remains unexecuted in this review.

### K1 — P0: enforce a provenance firewall on the actual learning path
**Refs / addresses:** §11.1; §12.1; G1. **Owner:** learning-integrity lead. **Why:** mandatory integration turns the documented receipt spike into an enforceable property rather than optional caller discipline.  
**Kickoff:** define a validated-evidence object that binds fitting membership, upstream context/encoder identities, allowed label use, and the consumer; require it at live eligibility. Audit alternate import/replay paths and serialize the bound identity into the accepted teaching record.  
**Experiment:** submit valid evidence and one-field-at-a-time mutations through the production-shaped route. Include a stale cached context with otherwise plausible metadata, duplicate submission, and interruption between admission and downstream state changes.  
**Falsifiable acceptance:** the valid control succeeds; invalid scope, identity, and missing evidence never produce an eligible corpus record or unauthorized HWM mutation. Retries are idempotent and interrupted transitions recover to a consistent state.  
**Stop / dependency:** do not claim protection from tests that only call the helper. Historical records without adequate evidence remain quarantined or explicitly unproven; signing a record cannot cure unsupported provenance.

### K2 — P0: resolve staging, then establish a closed-input replay
**Refs / addresses:** §10.1; §12; G2. **Owner:** evaluation-security lead. **Why:** performance and learning claims depend on knowing what the execution could access and what it actually used.  
**Kickoff:** retrieve and classify the disclosed exception; inventory affected datasets, stages, caches, credentials, reads, and artifacts. Build a staged fitting-only package and run it without a path to outer labels or campaign held-look results.  
**Experiment:** attempt denied file/object access; alter forbidden labels only from an independent test controller; repeat with identical permitted inputs and controlled randomness. Establish ordinary repeatability before interpreting counterfactual differences.  
**Falsifiable acceptance:** forbidden access attempts fail; permitted artifacts satisfy the declared repeatability comparison; the study receipt records the closure and exception status. All contaminated or unresolved descendants remain excluded from clean-qualification claims.  
**Stop / dependency:** absence of historical access logs is recorded as unknown, not proof of no read. A clean rerun gets a new identity and does not repair the historical claim by renaming it.

### K3 — P0: build the study evidence verifier and complete outcome ledger
**Refs / addresses:** §10 opening and §10.1.2; §12.1; G3–G4. **Owner:** benchmark/release lead. **Why:** machine-recomputed evidence prevents failures, retries, or build identities from disappearing between execution and publication.  
**Kickoff:** freeze request membership, qualifying prerequisites, terminal statuses, retry links, execution identities, and headline eligibility. Keep warm requests, warmups, fresh-process observations, and full-GP attempts in separate typed populations.  
**Experiment:** rebuild tables from the journal, then inject a missing result, duplicate success, timeout, stale binary hash, failed qualification, and interrupted resume. Preserve elapsed work and distinguish unscheduled/unexecuted work from successful completion.  
**Falsifiable acceptance:** all 100 planned warm requests reconcile without denominator changes; invalid evidence blocks the appropriate headline; retries cannot erase earlier outcomes. Per-arm GP attempt accounting reconciles independently against its pinned plan.  
**Stop / dependency:** incomplete evidence produces an incomplete-study report, never a fabricated aggregate. K1–K2 establish the integrity prerequisites for new clean learning or boundary-sensitive claims; K3 can still inventory historical limitations.

### K4 — P0: instrument a faithful multi-boundary latency harness
**Refs / addresses:** §10.1.1–§10.2; G4–G5. **Owner:** performance lead. **Why:** the existing whole-request definition is valuable only when every timing consumer measures that same subject.  
**Kickoff:** capture host monotonic start/end, stage milestones, dependent subprocess completion, device completion, attempted-backend status, and environment/cache settings. Keep profiled diagnostics separate from unprofiled headline runs; record per-stage device placement, effective math mode, driver/toolchain, CPU threading, GPU power/clock/thermal state, and competing workloads [E1–E2].  
**Experiment:** introduce controlled delays before data reading completes, during GPU work, and during emitted-artifact compilation/prediction. Repeat in declared warm and fresh-process states; calibrate instrumentation overhead without silently subtracting unexplained costs.  
**Falsifiable acceptance:** injected required work appears at the correct boundary, no backend is silently excluded, and timeline reconciliation explains overlap rather than summing overlapping/cumulative spans. The existing synchronized whole-request definition remains the headline.  
**Stop / dependency:** isolated-language or different-cache experiments are additional named protocols. A fresh-process case is not relabeled a fully cold system, and profiling-derived timing does not replace unprofiled measurement.

### K5 — P1: extend qualification from tensors to selected behavior
**Refs / addresses:** §10.2; §11.1; §12.1; G6. **Owner:** numerical/runtime qualification lead. **Why:** the system's deployable subject is the selected fitted predictor, not merely an encoder output array.  
**Kickoff:** preserve the stated numeric qualification unchanged and add subject-specific fixtures covering allowed mask values, finite-value requirements, ranking ties, fixed proposal replay, selected fitted-state identity, and emitted prediction parity on supported artifacts.  
**Experiment:** perturb close-ranking outputs within the existing tensor tolerance; substitute a wrong fitted artifact; inject unexpected NaNs and invalid mask behavior. Include valid supported-subset controls and declare which outputs permit exact versus tolerance-based comparison.  
**Falsifiable acceptance:** every deliberate semantic/state corruption is rejected or explicitly classified, while valid fixtures pass their own gate. Near-tie differences cannot silently inherit a behavior-preserving label merely from numeric closeness.  
**Stop / dependency:** do not weaken `3e-4` to make a faster mode pass. New precision modes, hardware, or unsupported model heads require their own declared qualification; passing tests does not prove all possible inputs.

### K6 — P1: pursue bottleneck-first, semantics-preserving GPU optimization
**Refs / addresses:** §10.1.1; §11; §12.1; G5 and G8. **Owner:** performance lead. **Why:** optimization effort should follow the measured limiting stage, with transfer and host work included rather than assumed negligible [E2].  
**Kickoff:** use K4 to choose one measured candidate: persistent accepted models, scope-safe context reuse, fewer transfers, incremental decoding, batched compatible work, or content-bound compilation reuse. These are hypotheses to evaluate, not reported repository features or guaranteed wins.  
**Experiment:** freeze proposals, candidate order, inputs, precision, support coverage, and backend set; compare baseline versus one change at a time. For reuse, mutate each relevant data/fold/model/preprocessor identity and verify invalidation; record hit and miss paths separately.  
**Falsifiable acceptance:** the preregistered paired latency decision favors the change beyond its uncertainty criterion, K5 qualification still passes, and failure coverage and required work do not degrade. Include lookup, validation, transfer, and compilation costs wherever the protocol charges them.  
**Stop / dependency:** reject a change whose apparent benefit comes only from skipping encoders/candidates without an explicitly new reuse protocol. Delay reduced precision or broad kernel rewrites until measured stage evidence justifies them.

### K7 — P1: prototype a scope-safe search execution DAG
**Refs / addresses:** §10.1; §11; §12.1; G7–G8. **Owner:** search/performance integration lead. **Why:** shared deterministic, fold-local work across recipes could reduce redundant execution while preserving the project's original-versus-executable and fitted-state identities.  
**Kickoff:** represent a frozen candidate inventory as dependencies keyed by exact inputs, fitting scope, transform parameters, backend, and precision. Reuse only qualified identical intermediates; preserve each candidate's logical ancestry and evaluation identity even when physical work is shared.  
**Experiment:** compare uncached sequential replay with DAG execution on the same work; inject fit-scope/key mutations, cancellation, resource refusal, and partial failure. Separately evaluate live bounded GP, where faster scheduling may change the completed candidate set.  
**Falsifiable acceptance:** replay produces the same declared candidate results and selected fitted-state identity under the applicable numeric contract; every scheduled logical attempt reconciles. Live-search benefits are demonstrated under the same declared budget and inclusion rules, with completed work disclosed.  
**Stop / dependency:** no outer-fold or unrelated-family reuse, no uncharged preprocessing, no hidden refusal filtering. Stop when coordination or memory costs erase benefits; ordinary DAG/caching techniques are not asserted as inventions.

### K8 — P1: execute the smallest real learning-closure and causal-gain experiment
**Refs / addresses:** §11.1; §12.1; dependencies §§6–7; G9. **Owner:** learning/evaluation leads. **Why:** this is the decisive bridge from a working GPU pipeline to evidence that teaching changes later search usefully.  
**Kickoff:** freeze the accepted parent, admissible real teacher corpus, optimizer/run identity, comparison protocol, and inaccessible transfer-evaluation boundary. Record actual exposure, updates, changed challenger state, reload, matched search, and qualified fitted output.  
**Experiment:** compare parent and challenger with the same search budget, task identities, evaluator, encoders, and paired randomness where applicable. Keep a no-update control and separate development comparisons from the nested transfer audit; no outer-evaluation result may feed tuning.  
**Falsifiable acceptance:** first, the full real-GPU chain reproduces with authenticated receipts. Separately, the challenger meets the preregistered transfer-improvement and guardrail criteria; a plumbing pass with null/negative lift is recorded honestly and does not trigger promotion.  
**Stop / dependency:** depends on K1–K5. A new retained checkpoint, lower imitation loss, or additional accepted teachers cannot substitute for the independent matched-search outcome.

### K9 — P2: broaden the falsification portfolio without rewriting the original study
**Refs / addresses:** §10.1–§10.2; §11; §12; G6–G7. **Owner:** experimental-methods lead. **Why:** broader stress and portability evidence would support a wider product claim, while preserving the value of the bounded regression study.  
**Kickoff:** preregister workload strata relevant to the proposed claim, such as fitting size, feature width, missingness, category expansion, numeric scale, and resource pressure. Add independent sessions/seeds and new hardware only under explicit new qualification identities.  
**Experiment:** use counterbalanced prospective arm order and controlled cache states; retain all outcomes. Analyze dataset/session/seed structure rather than treating repeated requests as independent tasks; establish standalone backend or service-load protocols only where needed.  
**Falsifiable acceptance:** each advertised domain/hardware/backend scope has its own complete qualification and measurement records, including failures and uncertainty. Unsupported tasks or classification remain outside the original regression qualification until separately tested.  
**Stop / dependency:** do not expand a scope after observing only favorable subgroups. Declare power/precision goals and material-effect criteria before additional sampling; an inconclusive experiment cannot be relabeled a broad win.

### K10 — P2: create proof-carrying search and deployable-claim packages
**Refs / addresses:** §11; §12.1; G3 and G10. **Owner:** evidence/release lead. **Why:** a reusable verifier connecting the project's mechanisms could strengthen technical defensibility more than a narrative catalogue of individually familiar methods.  
**Kickoff:** package the claim, permitted input closure, context provenance, original/executable recipe links, search-work ledger, fitted-state identity, supported export qualification, environment, and evidence status. Extend K3 with dependency invalidation and an independent replay interface.  
**Experiment:** give an authorized reviewer a clean environment and declared access; ask them to verify one valid chain, then test missing prerequisites, tampered fitted state, replayed old qualification, and revoked upstream evidence. Keep sensitive search internals access-controlled without hiding qualification limitations.  
**Falsifiable acceptance:** the valid package reproduces the scoped outcome without undocumented intervention; every tampered prerequisite blocks affected claims; an upstream correction triggers the specified downstream requalification. Serving promotion remains a separate explicit authorization.  
**Stop / dependency:** “proof-carrying” means checkable evidence within declared verifier trust assumptions, not a formal proof of generalization or truth. Document these assumptions and distinguish integration contribution from generic hashing, signing, caching, or code generation.

**Dependency order:** start K1–K2 and evidence recovery; complete K3–K4 for valid measurements; qualify K5 before accepting K6–K7 gains; run K8 only on the closed evidence path. Expand with K9 and K10 after their prerequisites. Null results, refusals, and unresolved history remain valid outputs, not reasons to weaken gates.

## Evidence gaps / UNVERIFIED load-bearing claims

| Needed claim or decision | What is missing here | Section/evidence that would close it |
|---|---|---|
| Mandatory encoder provenance enforcement | Live call-path integration and negative-entrypoint receipts; the supplied reassessment instead describes non-enforcement | §11.1 and §11 “Merged encoder-receipt spike”; §12.1 live evidence binding |
| Exact meaning and effects of the staging deviation | Full exception text, staged/read object inventories, scope, and affected descendants | §10.1 disclosure plus §10 “Observations” and its retained journal |
| CPU/CUDA qualification actually passed | Complete case identities, raw differences, nonfinite handling, device records, and pass/fail receipts | §10.2 and corresponding §12 evidence artifacts |
| Actual warm latency and coverage | Per-request outputs, terminal states, and report derivation for the planned population | §10 “Observations,” §10.1.2, and §12 manifest/journal |
| Resume, retry, and execution-window integrity | Clock rules, interruption outcomes, attempt linkage, and deduplication behavior | Full §10.1.2 “Window and resume” |
| Prediction and subprocess timing completeness | Omitted prediction-path details, input shapes/work, completion events, and emitted-backend logs | §10.1.1.1 “Prediction path”; §12 retained commands/results |
| Current measurement build identity | Study source/executable/environment identity and mapping to the later publication | §10 retained study plan; §12 manifest, not publication hash alone |
| Full-GP fairness and realized work | Paired RequestPlans, three random-stream identities, arm order, attempt outcomes, actual stopping, and elapsed budgets | §10.1 paired campaign evidence; §12 operational records |
| A particular GPU stage is the bottleneck | Per-stage profile and unprofiled whole-request comparison on the same work | §10 measurements; full §11 ranked opportunities with supporting traces |
| Supported deployment breadth | Per-selected-artifact qualification and actual runtime/host dependencies | §11.1; §12.1 “Fitted source and parity”; cross-link §8 |
| Real teaching causes transferable improvement | Authenticated real-GPU update lineage and independent matched transfer outcomes | §12 learning/transfer evidence; closing measurements required by §7.5 |
| Prioritization and reproduction are independently actionable | Omitted opportunity details, commands, artifact accessibility, owners, and executable claim checks | Complete §§11–12, especially §12.1 |

Every missing item above remains **UNVERIFIED in this review**. Missing extract content is not proof that an artifact or implementation does not exist. The §10 observations heading is visible but its subsection number is not supplied; no number is invented.
**External methodology references, not product evidence:** [E1] PyTorch, “PyTorch Benchmark,” accessed 2026-09-25, `https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html`; supports synchronization, warmup, thread-setting, and repeated-measurement controls. [E2] NVIDIA, “CUDA C++ Best Practices Guide,” accessed 2026-09-25, `https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html`; sections on assessment/profiling, host/device transfer, and timing support bottleneck-first diagnosis and complete timing. These current guidance pages do not identify the study's installed software versions.

## Fail-open / silent-lie paths (if any in shard)

| Path | Evidence posture | Required barrier |
|---|---|---|
| Optional receipt validator → live eligibility → apparently proven teacher | Integration omission DOCUMENTED (§11.1); successful bypass untested | Mandatory validated evidence at admission and persistent downstream identity; K1 |
| Unclassified staging exception → “clean” benchmark or transfer claim | Deviation DOCUMENTED; contamination unknown (§10.1) | Classify reachability/read/use separately; restrict affected claims; clean new-identity replay; K2 |
| Missing/failed qualification → report generated from remaining rows | HYPOTHESIS; no report generator inspected (§10; §12) | Dependency-aware publication refusal and complete status ledger; K3 |
| Timeout/retry/unsupported export disappears → apparently faster successful population | HYPOTHESIS; resume and observations omitted (§10.1.2) | Preserve original inventory, failures, retry links, and backend requirements; K3–K4 |
| Source-ready or cumulative backend milestone → “E2E per language” | Explicitly forbidden by the documented contract; execution unverified (§10.1.1) | Typed milestone/report definitions and completion tests; K4 |
| Numerically close tensors → “same program and fitted artifact” | HYPOTHESIS; qualification scope insufficient for that inference (§10.2) | Separate numeric, decision, fitted-state, and per-artifact checks; K5 |
| New source fix or matching hash → historical result becomes current qualification | Contrary to the stated status rules (§12) | Preserve original measured identity; require change-specific requalification; K3/K10 |
| Synthetic update or development gain → “transfer proven” | Unsupported by supplied evidence (§11; §12; §7.5) | Separate real-loop and independent transfer milestones; K8 |

## Cross-links to other sections the owner must not ignore

**§§2.1, 2.5–2.6 → K1/K2/K8:** a clean row split does not establish clean inherited contexts, teachers, caches, or family-level learning closure. Boundary ownership must include permitted upstream inputs; this report does not re-audit the panel or corpus.  
**§§3.1–3.6 and §4.2 → K4/K5/K6:** receipt and cache identities must preserve encoder-product semantics and accepted asset identity, not just a matching tensor width. This is an integration dependency, not a new architecture finding.  
**§§5.5–5.6, 5.8–5.10 → K7:** preserve paired budgets, refusal visibility, selected-candidate ancestry, original/executable recipe distinction, and fitted state when changing execution scheduling. A feasibility rejection remains an accounted attempt under the declared protocol.  
**§§6.2–6.8 and §9.1 → K1/K8:** corpus availability, observed exposure, actual optimizer updates, and resumed state are distinct evidence objects. An admission pass is not automatically a comparable HWM improvement or a reusable challenger gain.  
**§§7.5–7.6 → K3/K8/K9:** keep denominator and comparison identities stable; do not convert repeated request timing into independent task evidence or use development tuning as transfer validation.  
**§8.1–§8.2 → K4/K5/K10:** export the selected fitted subject, retain unsupported-model status, and disclose host dependencies. Faster compilation or successful C++ execution does not establish universal zero-Python deployment.  
**§1.1 and §12 → K10:** articulate the project-specific integration and its tested evidence boundary without claiming ownership of established constituent methods or treating a byte hash as operational proof.

## Concerns / residual risk

**Evidence honesty:** the strongest direct finding is the explicitly described live receipt-integration gap. Most performance and replay concerns are hypotheses or unavailable evidence, not verified product defects. No latency, speedup, success rate, new model identity, or invented numerical acceptance target is supplied.  
**Limited guarantees:** denied access and counterfactual tests strengthen a boundary but do not reconstruct unknown historical behavior or prove all semantic family separation. Finite fixtures cannot guarantee every numerical decision, grammar path, or exported predictor.  
**Verifier trust:** receipts are only as trustworthy as their producers, keys, isolation, and validators. A correctly signed false assertion can still be false; independent observation and replay remain necessary. Verifier overhead belongs in the relevant measured boundary.  
**Performance/quality tradeoffs:** increased throughput may change which candidates finish under a time cap. Report that changed search behavior honestly; do not collapse an identical-work acceleration experiment into a quality-at-budget experiment or a learning-transfer claim.  
**Commercial/IP restraint:** this is a technical gap and roadmap review, not a novelty, patentability, or freedom-to-operate determination. The proposed integration asset is an engineering opportunity whose value must be established empirically and whose legal status is not assessed.  
**Release recommendation:** fund P0 enforcement and measurement closure now; advance P1/P2 on their falsifiable gates. Retain bounded, dated claims supported by actual receipts, and withhold broader promotion when evidence is missing, contradictory, or inconclusive.
