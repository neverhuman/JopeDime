<!-- from-tar tab=5 url=https://chatgpt.com/c/6ab702a9-3518-83ea-9996-ec0670f23f68 sha256=96758df6ef21555ccbdbb5101ac6575b3735d8eb0c9a4323ee6b0591fb8bbc5e chars=38106 -->

# GP representation and evolution — evidence-bounded technical review

Batch tab: 5 of 10. Scope: §5, with adjacent sections cited only where they constrain GP search, selection, refit, or ancestry.
Evidence cutoff supplied by the monograph: **2026-09-25T19:31:50Z**. Publication source named in the extract: `3d71fce041d1dd1173ed98160f7d986bb426ba6f`.
Review basis: the pasted structured extract and its section headings. No product source, retained RequestPlan, execution journal, fitted artifact, or ancestry trace was inspected; no product tests were run or repository changes made.
**Evidence vocabulary:** DOCUMENTED means explicitly stated in the supplied extract, not independently verified. UNVERIFIED means the required detail or execution evidence is missing. A conditional failure story is not a finding that the implementation actually fails.
**Ranking:** severity reflects the consequence if the identified boundary is unenforced; priority reflects the order of proposed engineering work. CRITICAL is an integrity/promotion blocker, HIGH a search-validity or containment blocker, and MED a claim-strength limitation. None is a patent-novelty determination.
All new receipt fields, lifecycle states, tests, and gates below are **proposals**, not claims about existing APIs. External method references support evaluation principles only; they do not establish repository behavior.

## Executive hits

- **CRITICAL — close the selection/refit boundary before claiming clean GP evidence.** Five training-side fold losses are documented, but the full refit procedure and transitive fit scopes are absent. Test outer-label noninterference and fold-local fitting separately; ordinary inner-validation feedback is allowed to guide search. (§§5.8–5.9; §2.1.1.1)
- **Preserve a real strength: original and executable recipes are explicitly distinguished.** Extend that distinction into immutable, linked records for variation, adaptation, fold fits, selected recipe, and final refit. Never treat a changed fit scope as the same fitted predictor. (§§5.1, 5.10; §8.1)
- **HIGH — paired arms are a protocol, not merely a shared seed.** Generation-zero construction, replacement rules, random-stream consumption, arm order, and proposal overhead need a frozen comparison contract before a neural-seeding advantage is attributable. (§§5.4–5.5, 5.8; §10.1)
- **HIGH — “bounded GP” needs enforced stopping and complete accounting.** The study states at most 120 seconds and 96 candidate attempts per arm; it does not expose deadline enforcement, in-flight work handling, or attempt semantics. Build an auditable scheduler, not another timer field. (§§5.2, 5.8; §10.1)
- **P1 — turn dense-width refusal into a conservative feasibility contract.** Keep refusal as a safety outcome, then preflight supported operations using fitting-side metadata and checked resource bounds; record unknown cases and rejection bias rather than silently simplifying recipes. (§5.6; §8.2)
- **P0/P1 — make measured selection and ancestry independently checkable.** Require complete fold evidence, explicit invalid-result handling, deterministic declared tie rules, and replayable parent-to-child transitions before a winner is handed off. (§§5.2, 5.4, 5.10)
- **MED — a warm fitted example cannot establish evolutionary benefit.** Preserve the demonstration as integration evidence, then compare generation-zero selection, fresh-proposal search, and evolution under declared matched budgets. This report assigns no unobserved win rate or speedup. (§§5.3, 5.5, 5.7; §7.5)
- **Decision:** first fund four integrity kickoffs—fit-scope isolation, candidate/fitted identity, paired accounting, and enforced bounds—then resource preflight, selection/ancestry qualification, and a controlled benefit study. Missing chapter bodies must remain UNVERIFIED until their named evidence arrives. (§§5.1–5.10)

## Ranked IP gaps (CRITICAL/HIGH/MED/LOW)

### G1 — CRITICAL: selection, refit, and held-fold identities are not closed by the extract
**§ref / status:** §§5.8–5.9; §§2.1, 2.1.1.1; §11.1. Full GP boundary enforcement is UNVERIFIED. The extract explicitly reports that the encoder-receipt validator is not called by live eligibility; this does not prove the GP fitter itself reads forbidden data.
**Attack/gap story:** a fold fitter can exclude validation labels while consuming a transform, target-conditioned context, or ancestor that carries information forbidden by the declared protocol. A later refit can also expand membership without changing the displayed recipe name. Either can misstate what the score or fitted artifact actually represents.
**Important distinction:** inner-validation scores legitimately influence GP selection. A selected-on-OOF score is not an independent estimate of the entire adaptive search. Outer scoring labels must remain unavailable to search; fold-local preprocessing must not fit on inner validation data under a train-fitted contract. Method background: [^preprocessing] [^selection].
**Detection:** freeze a candidate and perturb only its inner validation labels: its fold-fitted state must not change, although its score may. Separately perturb outer labels under a controlled deterministic search schedule: proposal/selection outputs must not change. Instrument reads and invalidate caches; a label-read boolean alone is insufficient.
**Harden:** separate fitter, inner scorer, and outer auditor capabilities. Bind every state to permitted rows, label access, transforms, context, and ancestors. Freeze the selected recipe before a declared final refit; link the new fitted state to its actual membership rather than presenting it as an OOF fold model.
**Close with:** complete §5.9 procedure, §5.8 request construction, fold/final-fit receipts, negative access tests, and the actual §10 staging-deviation details. Do not infer a clean run from the stated intended protocol.

### G2 — HIGH: typed syntax and preserved bytes do not establish fitted-state integrity
**§ref / status:** §§5.1, 5.8–5.10; §§8.1–8.2. `GenomeIr`, original bytes, and explicit raw-input adaptation are DOCUMENTED; their full variation and state-lifecycle invariants are UNVERIFIED.
**Attack/gap story:** a child may inherit stale fitted dictionaries or coefficients after its recipe changes; a score cache may alias different fit scopes; repeated adaptation may insert the input stage twice. A syntactically typed recipe could then carry evidence belonging to another executable or state.
**Detection:** use controlled fixtures for adaptation repeated twice, a changed transform with unchanged display name, a changed fitting membership, and a mutated child sharing a parent object. Check that parent state stays unchanged and every fit/score cache hit has the correct subject.
**Harden:** keep original proposal bytes, typed recipe, adaptation record, executable recipe, fold-fitted states, and final-refit state separately identified. Revalidate supported variation outputs. Distinguish fit-cache identity from score-cache identity; scoring on a new partition must not reuse an old score merely because fitted state matches.
**Semantic guard:** recipe order is part of §5.1's contract. Any canonicalization must preserve declared execution semantics; it must not reorder transforms or conceal an adaptation. A changed executable requires a new candidate identity and evaluation, not a repaired winner label.
**Close with:** §5.1 type/lifecycle rules, §5.8 variation validation, and §5.10 serialized examples tracing one unchanged parent and its changed child through fitting and selection.

### G3 — HIGH: paired-arm attribution can fail through generation-zero or scheduling asymmetry
**§ref / status:** §§5.4–5.5, 5.8; §10.1. Paired neural/generic order and a pinned RequestPlan containing sampling settings and three random streams are DOCUMENTED; the plan itself and stream-consumption rules are UNVERIFIED.
**Attack/gap story:** one arm may receive uncharged resampling, different duplicate handling, cached fits, different initial-population admission, or favorable warm execution order. A shared seed does not prevent an extra draw in one path from perturbing later shared randomness.
**Detection:** replay a small controlled plan while changing only origin labels and arm execution order. Inject an extra rejected proposal; verify that unrelated random streams and budget charges remain as specified. Inspect generation-zero proposed/admitted/evaluated inventories rather than comparing final winner labels alone.
**Harden:** freeze the comparison's estimand: whole neural-seeded optimizer versus whole generic optimizer, or initialization quality specifically. Bind common folds, evaluator, downstream operators, caps, cache policy, and tie rules. Record source-dependent overhead and all permitted arm differences explicitly.
**Attribution guard:** do not require identical populations after legitimate evolutionary divergence. Require reproducible boundaries and common evaluation conditions. Keep end-to-end time-matched and attempt-matched conclusions separate; equal caps need not yield equal completed evaluations.
**Close with:** §5.5 generation-zero specification and paired manifests, §5.8 random-stream/attempt semantics, and order-balanced run receipts. Do not copy the direct request's sampling configuration into full GP; §10.1 distinguishes them.

### G4 — HIGH: stated search limits do not prove containment of expensive or unfinished work
**§ref / status:** §§5.2, 5.8; §10.1. The 120-second/96-attempt per-arm limits apply to the retained study, not an established universal product contract. Exact clock, admission, cancellation, and accounting semantics are UNVERIFIED.
**Attack/gap story:** a cooperative timer checked between candidates cannot by itself bound a blocking fitter or allocation. Concurrent dispatch can over-admit work; regenerated or refused candidates can disappear from counts; a timed-out fold can become a misleading partial score.
**Detection:** inject a blocked worker, allocation failure, delayed completion, exhausted grammar, repeated invalid proposals, and deadline/dispatch races in an isolated harness. Verify ledger reconciliation, stop reason, absence of unauthorized late selection, and cleanup before worker reuse.
**Harden:** define all charged work and separately bounded preparation. Reserve attempt capacity before dispatch, enforce resource isolation, and freeze the eligible incumbent set at the declared deadline. Preserve all retry/refusal events even where the protocol does not count them as fitting attempts.
**Reporting guard:** distinguish stopping new work, returning a result, and completing cleanup. Disclose actual overshoot; do not call a delayed cleanup an “at-most” wall-clock success. With no complete valid incumbent, return an explicit failure rather than a reward-head guess or an undeclared baseline.
**Close with:** §5.2 stop policy, §5.8 scheduler contract and failure receipts, and §10's actual admission/termination timestamps. No runtime containment or compliance measurement is available here.

### G5 — HIGH: feasibility preflight lacks an exposed soundness boundary
**§ref / status:** §5.6; §§5.1, 5.8; §8.2. The heading explicitly names dense-width refusals and a proposed preflight; its algorithm, integration status, and refusal counts are UNVERIFIED.
**Attack/gap story:** individually acceptable stages can produce an unacceptable intermediate or terminal workload. The input-stage setting `max_onehot=256` does not, by itself, establish a bound for every complete recipe. A permissive estimate risks over-allocation; an opaque conservative filter changes the searched candidate distribution.
**Detection:** exercise supported operators with training-side high-cardinality data, missing-value expansion, empty/degenerate inputs, chained widening, and arithmetic-overflow boundaries. Compare predicted bounds with instrumented intermediate widths and covered allocations, not only terminal output width.
**Harden:** use operator-specific conservative checks with explicit covered resources and outcomes: proved within contract, proved outside contract, or unknown. Unknown must not silently mean feasible. Preserve runtime containment for uncovered backend workspaces and reject or sandbox unresolved cases under a stated policy.
**Fairness guard:** apply the same versioned policy to both arms; retain refusal reasons and original recipes. Report source-specific attrition. Preflight must not use forbidden-partition category statistics or replace a selected recipe with a narrower alternative.
**Close with:** full §5.6 proposal and implementation status, operator coverage, cap scope, overflow tests, estimate-versus-observation records, and paired refusal inventories. A safe refusal is not itself a bug or a completed candidate.

### G6 — HIGH: proposal rank, measured fitness, and stopping may be conflated
**§ref / status:** §§5.2–5.4, 5.8; §§2.1.1.1, 6.1, 7.1. The benchmark's mean of five training-side fold losses is DOCUMENTED; the omitted selection details do not establish metric direction, invalid handling, tie rules, or stopping behavior for every GP path.
**Attack/gap story:** a high proposal rank could survive a failed fit; missing fold losses could be averaged away; a default numerical sentinel could outrank a complete candidate. An undocumented improvement tolerance or tie-break could make faster/smaller recipes the effective objective without naming that change.
**Detection:** feed a selector complete strict-winner fixtures, exact ties, invalid/nonfinite results, incomplete fold vectors, and results from incompatible evaluator identities. Deliberately reverse predicted-rank and measured-fitness order; verify the declared measured selection rule controls the winner.
**Harden:** represent validity separately from the numerical objective. Require the complete declared evidence before ranking; freeze objective direction, aggregation, ties, any secondary criterion, and stopping policy in the RequestPlan. Proposal rank may control evaluation priority, but must not impersonate measured fitness.
**Stopping guard:** distinguish budget exhaustion, explicit early stopping, candidate-space exhaustion, and failure. Log the incumbent and evidence available at each transition. Do not assume tournament size, elitism, crossover rate, or stagnation rules that the extract never supplies.
**Close with:** full §§5.2–5.4 and §5.8 selection tables, executable reference fixtures, and a real ordered candidate ledger. §6.1 explicitly distinguishes reward scores, percentiles, and fitted RMSE.

### G7 — HIGH: ancestry labels do not yet prove the winner's causal or fitted provenance
**§ref / status:** §§5.5, 5.8, 5.10; §12.1. The evidence map calls population, ancestry, and limits implemented; the actual ancestry trace is absent and operational reconstruction is UNVERIFIED.
**Attack/gap story:** an evolved child can be described as a direct neural proposal; a cached winner can be attached to the wrong parent or fold scope; an input adaptation can masquerade as the only change despite intervening variation. A plausible ancestry diagram would not expose these substitutions.
**Detection:** reconstruct a selected lineage from original proposals through every recorded variation, adaptation, evaluation, and final refit. Tamper independently with a parent identity, operator record, executable recipe, and fitted-state reference; each inconsistency must prevent an authenticated handoff.
**Harden:** retain an append-only event trace with parent references, exact changed recipe, operation identity, relevant random-stream state, cache provenance, fit scopes, scores, and selection decisions. Distinguish direct proposal, evolved descendant, copied incumbent, and reused evaluation without inventing unrecorded parentage.
**Replay guard:** exact event replay and fresh wall-clock re-execution are different tests. Reconstruct recorded decisions first; separately qualify numerical reproduction and timing-dependent stopping under declared tolerances. Hashes prove byte identity, not that an event actually executed.
**Close with:** the real §5.10 trace, linked fitted artifact, resolver for all ancestors/cache entries, and an independently replayed selected path with tamper-test results.

### G8 — MED: integration success is insufficient evidence for evolutionary contribution
**§ref / status:** §§5.3, 5.5, 5.7; §§7.5, 10.1. The warm demonstration's body is absent. The retained full-GP protocol names three datasets and a seed0 campaign; the extract provides no GP benefit measurements.
**Attack/gap story:** a valid warmed fitted predictor may be presented as evidence that evolution improves neural seeds, or a final winner may be compared with a weak generic result after failed attempts were excluded. Neither attribution follows from the available text.
**Detection:** request generation-zero best valid results, later incumbents, source/ancestry labels, all attempted and failed work, proposal overhead, and matched paired outcomes. Check whether the winner is an evolved descendant, an unchanged initial candidate, or a declared reuse.
**Harden:** label integration, feasibility, search-quality, and independent performance evidence separately. Compare evolution against additional fresh proposals from the same seed source; retain the generic arm as a separate source comparison rather than assuming it isolates evolution alone.
**Claim guard:** adaptive improvement on selection folds is descriptive, not independent confirmation of the entire search. Freeze the comparison protocol before outer evaluation and preserve negative/inconclusive results. The transfer reporting contract in §7.5 is explicitly not a measured lift result.
**Close with:** §§5.3, 5.5, 5.7 worked examples and paired trajectory receipts. Do not infer a general win rate, population diversity, or cross-dataset benefit from missing operational tables.

## Ranked improvements / kickoffs (P0/P1/P2)

### I1 — P0: capability-scoped fit/select/refit transaction
**§ref / gaps:** §§5.8–5.9; G1. **Suggested owner:** GP orchestration with the evaluation-boundary owner.
**Why it matters:** makes permitted information access and the final fit subject explicit instead of relying on shared dataset handles or a statement that held labels were not read.
**Build:** distinct fold-fitting, inner-scoring, selected-recipe, and final-refit records; narrow data capabilities; provenance-aware context/transform caches; a final-refit membership declaration. Keep outer auditing outside the search worker and preserve declared campaign held-look policy.
**Falsifiable acceptance:** in a controlled fixed-work schedule, changing outer labels cannot change search outputs. For a frozen candidate, changing inner validation labels cannot change its fitted state. Forbidden reads fail, cache mismatches reject, and final refit cannot silently expand the permitted membership.
**Legitimate variation:** inner-validation scores may change later search choices; a declared refit may change coefficients. The tests must allow both rather than mistakenly demanding that every artifact remain identical.
**Completion evidence:** retained negative tests, actual fold/final-fit receipts, and one full bounded run showing the selected recipe frozen before refit. Reusing held labels later requires a distinct artifact/protocol, not relabeling earlier accuracy evidence.

### I2 — P0: immutable candidate and fitted-state envelope
**§ref / gaps:** §§5.1, 5.8–5.10; G2. **Suggested owner:** GP representation/fitter interfaces.
**Why it matters:** converts the documented original-versus-executable distinction into a checkable boundary across variation, caching, selection, and handoff.
**Build:** linked identities for original bytes, grammar/IR version, executable recipe and adaptation, parent events, fitting membership/data, fitter configuration/randomness, fitted state, and evaluator-specific scores. Keep fit-cache and score-cache bindings separate; use immutable parent state or equivalent tested isolation.
**Falsifiable acceptance:** every supported variation either validates as a fresh candidate or fails explicitly; changing executable semantics, fit scope, or evaluator scope cannot produce a stale eligible score. Repeat adaptation is rejected or demonstrably follows a declared idempotent rule without corrupting original provenance.
**State acceptance:** a child mutation cannot change its parent's predictions; a final refit cannot inherit a fold score as though that score measured its exact final state. Unsupported export retains the same selected candidate and its explicit export-failure status.
**Completion evidence:** round-trip/corruption fixtures and a serialized candidate-to-final-state record consumed by the actual handoff, not only a standalone receipt validator.

### I3 — P0: frozen paired RequestPlan plus complete generation-zero ledger
**§ref / gaps:** §§5.4–5.5, 5.8; G3. **Suggested owner:** paired-search harness.
**Why it matters:** separates genuine proposal-source advantage from extra attempts, different initialization rules, cache benefits, and execution-order effects.
**Build:** expose initial population construction, rank/admission rules, common operators, fit/evaluator scopes, duplicate/retry policy, random-stream boundaries, budget charging, and the controlled arm difference. Record all generation-zero proposals and whether each became an evaluated candidate.
**Falsifiable acceptance:** with identical injected candidates, swapping only origin labels cannot alter evaluator results, admissibility, or budget policy. An inserted rejected proposal cannot perturb unrelated random streams. Reversing arm order follows the same declared cache/warmup policy and produces an auditable record of any observed differences.
**Operational acceptance:** paired reports reconcile proposed, scheduled, completed, rejected, cached, and failed events under non-overlapping declared categories. Missing/failed arms remain visible; direct-path sampling settings are never silently substituted for the paired plan.
**Completion evidence:** a small controlled comparison plus retained real paired manifests and order-balanced observations. Exact equality of noisy production timing is not an acceptance requirement.

### I4 — P0: enforced bounds and explicit terminal outcomes
**§ref / gaps:** §§5.2, 5.8; G4. **Suggested owner:** search scheduler/runtime.
**Why it matters:** prevents nominally bounded evolution from becoming an unbounded retry loop, a resource incident, or a success report built from partial fits.
**Build:** pre-dispatch quota reservation; a monotonic deadline; isolation for risky fits; an explicit policy for in-flight work; stop/cleanup records; and a complete-incumbent snapshot. Separately bound proposal regeneration so rejected candidates cannot create unlimited work outside the fitting-attempt counter.
**Falsifiable acceptance:** blocking, OOM, repeated-rejection, cancellation, and dispatch-race fixtures cannot admit work beyond the declared quotas or promote incomplete evidence. Late results obey the frozen policy, and contaminated workers cannot service another request without verified cleanup or replacement.
**Timing acceptance:** each run reports admission, deadline, stop, result, and cleanup boundaries. An observed overrun fails any strict elapsed-time claim; excluding cleanup from a metric requires a distinct, explicitly named metric.
**Completion evidence:** failure-injection receipts and a real bounded run whose complete event inventory reconciles to its counters. No valid incumbent must produce a typed no-result outcome, not a substitute winner.

### I5 — P1: typed feasibility analysis with refusal diagnostics
**§ref / gaps:** §§5.1, 5.6, 5.8; G5. **Suggested owner:** GP/operator authors with runtime containment.
**Why it matters:** avoids spending scarce search budget on provably impossible candidates while preserving honest failure accounting and the searched recipe identity.
**Build:** operator-specific shape/resource transfer rules for the supported subset, trained-dictionary metadata, checked arithmetic, intermediate-width tracking, and an explicit record of covered allocations/workspaces. Treat unresolved operations as unknown; keep runtime caps as the backstop.
**Falsifiable acceptance:** for covered controlled cases, measured intermediate widths and allocations never exceed the stated conservative bounds. Unsupported operators, arithmetic overflow, and insufficient metadata cannot receive a proved-feasible result. Held-only categories cannot influence fitting-side estimates.
**Quality acceptance:** compare rejection decisions against bounded actual evaluations on an adjudication set; disclose conservative false refusals and unknowns by arm. A speedup claim must include preflight overhead and refusal inventory; changing search feasibility policy requires a new protocol identity.
**Completion evidence:** per-operator soundness tests, boundary/overflow fixtures, an estimate-versus-observation report, and paired attrition tables. This does not assert a new dense cap or promise an unmeasured speed gain.

### I6 — P1: independently executable selector and stop-policy specification
**§ref / gaps:** §§5.2–5.4, 5.8; G6. **Suggested owner:** GP selection/evaluator interfaces.
**Why it matters:** prevents cheap proposal heuristics, numerical failure, or partial evidence from becoming silently authoritative fitness.
**Build:** a small reference adjudicator for the declared objective, fold completeness, aggregation, identity compatibility, tie/secondary rules, and stop transitions. Test the production path against that reference; keep predicted ranks in a separate field.
**Falsifiable acceptance:** hand-checkable complete-score fixtures select the expected candidate; nonfinite/incomplete/incomparable records never become an eligible winner. When proposal ordering conflicts with measured ordering, the declared measured rule wins. Stopping uses only permitted evidence and its recorded policy.
**Extension gate:** any optional resource penalty, complexity preference, diversity rule, early-pruning scheme, or budget-adaptive stopping must be declared and evaluated as a changed search policy, not quietly inserted into the existing objective.
**Completion evidence:** a passing adversarial selector suite and replay of a retained real candidate ledger. Move this work to P0 if production currently lacks an explicit completeness/validity gate; that absence is not established by the extract.

### I7 — P1: ancestry replay certificate for each selected candidate
**§ref / gaps:** §§5.5, 5.8, 5.10; G7. **Suggested owner:** GP event trace/handoff.
**Why it matters:** strengthens the technical case for measured evolution by proving which candidate was generated, changed, fitted, selected, and handed off—not merely displaying a lineage label.
**Build:** append-only events resolving each parent, supported variation, executable change, adaptation, cache reuse, fit result, and selection decision. Produce a compact selected-path certificate plus access to the full attempt ledger needed to validate selection.
**Falsifiable acceptance:** a separate reader reconstructs the selected recipe and decision sequence without trusting the final summary. Broken parent references, omitted changes, wrong fit scope, substituted executable bytes, or incompatible cached scores prevent authentication.
**Replay acceptance:** recorded-event replay reproduces the decisions; fresh numerical execution is checked under an explicitly qualified reproducibility contract. Run actual bounded searches for timing claims rather than substituting offline trace replay.
**Completion evidence:** one genuine selected lineage with final fitted-state linkage, full candidate inventory, and tamper-test results. Keep private discovery information separate from public deployable state as §8.1.1 requires.

### I8 — P2: isolate and measure the incremental value of evolution
**§ref / gaps:** §§5.3, 5.5, 5.7–5.8; G8. **Suggested owner:** GP experiment lead with paired-protocol review.
**Why it matters:** makes the evolutionary contribution distinguishable from neural initialization, successful preprocessing, or simply evaluating more candidates.
**Build:** predeclare comparisons across neural versus generic seeding and evolution versus additional fresh proposals from the same source. Capture generation-zero and later incumbents. Hold common evaluation rules fixed; analyze attempt-matched and inclusive-time-matched budgets separately.
**Falsifiable acceptance:** every planned outcome, including refusal and no-result, is retained. The primary comparison, data/seed allocation, and uncertainty decision rule are frozen before independent evaluation. Claim evolutionary benefit only if the prespecified comparison supports it; otherwise report no demonstrated benefit.
**Safeguard:** proposal overhead, failed attempts, and unchanged initial winners stay visible. Use protected evaluation for the entire adaptive search; do not repeatedly inspect outer results to choose the winning operator policy.
**Completion evidence:** paired trajectories, actual ancestry attribution, complete outcome denominators, and the preregistered comparison report. The existing seed0/three-dataset study does not supply these missing conclusions in the extract.

## Evidence gaps / UNVERIFIED load-bearing claims

A heading identifies where to ask for evidence; it is not evidence that its promised mechanism was implemented or executed. “Implemented” in §12.1 is the monograph's status assignment, not independent verification by this review.

| Missing or incomplete claim | Section needed to close it | Minimum closure evidence |
|---|---|---|
| Full type system, adaptation invariants, expression-reference rules, and ownership of fitted state | §5.1 | IR/operator specification; accepted/rejected examples; state-isolation and adaptation tests. |
| Exact selection, secondary criteria, ties, invalid results, and stopping | §§5.2–5.3, 5.8 | Frozen policy plus controlled expected-result fixtures and real stop records. |
| Proposal generation versus ranking versus measured selection | §5.4 | Candidate/rank/fit ledger; proof that unmeasured ranks cannot become measured winners. |
| Generation-zero membership, duplicated seeds, retries, backfill, and arm comparability | §5.5 | Paired RequestPlans, full initial inventories, stream boundaries, and charge rules. |
| Dense-width refusals and operational status of the proposed preflight | §5.6 | Exact caps and scope, operator coverage, typed refusal records, overflow tests, measured estimates. |
| Warm fitted demonstration after integration | §5.7 | Input/recipe/state identities and executed command/result; this closes only that demonstration. |
| Candidate/attempt semantics, variation validation, budget boundaries, and containment | §5.8 | Full algorithm, ledger reconciliation, watchdog/cancellation results, real bounded execution. |
| Historical versus current refit membership and access to the held fold | §5.9 | Versioned protocol, permitted-read scopes, fold/final-fit receipts, and negative access tests. |
| Real candidate ancestry and final fitted-state trace | §5.10 | Resolvable event history, complete competing-candidate ledger, selected-state link, independent replay. |
| Actual protocol compliance, including the acknowledged staging deviation | §10.1 and omitted observations/staging discussion | Execution journal and deviation details; intention cannot stand in for actual access behavior. |
| Runtime substantiation of the “implemented” search and selection map | §12.1 with §§5.5, 5.8, 5.10 | Exact source/build binding and retained real-input receipts; source-path names alone are insufficient. |

## Fail-open / silent-lie paths (conditional unless explicitly documented)

| Path | Silent failure | Proposed fail-closed behavior / detection | §ref |
|---|---|---|---|
| Context, transform, or ancestor from a different fit scope enters a fold fit | “OOF” masks forbidden information or misstates the evaluated subject | Validate transitive provenance; isolate fitting capabilities; perturb forbidden inputs with caches invalidated | §§5.8–5.9; §2.1 |
| Original recipe label survives a changed executable or mutable inherited state | Score/state belongs to another candidate | Separate immutable identities and invalidate stale fit/score evidence; test parent-child isolation | §§5.1, 5.10 |
| Duplicate, refused, or regenerated proposals vanish from paired accounting | One arm receives unreported search opportunity | Preserve all events and apply frozen quota/denominator rules; bound regeneration separately | §§5.5–5.6, 5.8 |
| Timeout or fit error leaves partial folds or a favorable default score | Incomplete work becomes the winner | Explicit invalid/incomplete terminal outcomes; only eligible completed evidence can select | §§5.2, 5.8 |
| Preflight unknown is treated as feasible, or refusal triggers hidden simplification | Resource safety fails or an easier recipe becomes the apparent original winner | Bounded unknown handling; refusal keeps original identity; any changed recipe is a new candidate | §§5.1, 5.6 |
| Final refit silently changes membership; export performs another fit | Reported score, final state, and deployed state are conflated | Freeze selected recipe; declare final-refit scope; emit that frozen fitted state without reselection | §5.9; §8.1 |
| Missing ancestry/cache events are reconstructed from a plausible narrative | A descendant or reused model is attributed to an unproven proposal | Authentication fails on unresolved lineage; preserve typed origin and reuse events | §5.10 |
| Warm integration or selection-fold improvement is called general evolutionary lift | Descriptive success becomes an unsupported effectiveness claim | Keep claim classes separate; require the independently evaluated comparison | §§5.3, 5.7; §7.5 |

## Cross-links to other sections the owner must not ignore

- **§2.1 and §2.1.1.1 → §§5.8–5.9:** GP needs the split owner to identify permitted rows, transforms, labels, and ancestors. Five fitting-side fold losses do not by themselves certify the whole search against adaptive-selection bias.
- **§§3.3–3.6 → §§5.4–5.5, 5.9:** request context provenance must match its allowed consumption scope. This is a GP input contract; this report does not audit encoder architecture or dimensions.
- **§4.4 → §§5.1, 5.4–5.5:** grammar/proposal changes require a declared decoder-to-IR boundary and traceable generation-zero identity. Direct sampling settings cannot define omitted full-GP settings by implication.
- **§§6.2, 6.4 and §11.1 → §§5.8–5.10:** a selected GP winner is not automatically an authenticated teacher. The reported non-enforcement of encoder receipts downstream must not be hidden by a “pure search” label.
- **§§7.1, 7.5–7.6 → §§5.2, 5.5, 5.8:** bind score direction/scope and preserve planned-attempt outcomes; warm success, completed candidates, teacher eligibility, and transfer are different consumers.
- **§§8.1–8.2 → §§5.1, 5.9–5.10:** explicit input adaptation is allowed; replacing the winner after an unsupported export is not. Final-refit identity must arrive intact at fitted export.
- **§10.1 and §12.1 → §§5.5–5.8, 5.10:** obtain the exact measurement plan, actual staging deviation, and execution ledger. Publication hashes and implementation labels cannot fill in missing operational evidence.

## Concerns / residual risk

**Decision boundary:** the supplied material supports the existence of a documented representation/selection story, not an independent qualification of its operational guarantees. The largest gaps may already be answered in the omitted bodies; obtain those sections before treating this review as proof of a defect.
**First release gate:** require I1–I4 evidence before upgrading claims about clean selection, preserved fitted identity, paired attribution, or enforced search bounds. This is a proposed gate, not a claim that an existing release is unsafe or that these mechanisms are absent.
**Scope restraint:** the study's regression inputs, seed0 campaign, and stated per-arm limits do not establish behavior for every grammar, task, backend, or workload. Unknown coverage remains unknown; no new numerical cap, architecture dimension, KPI, or checkpoint identity is inferred.
**Residual technical risk:** sound preflight is limited to its explicitly covered operations/resources; replay does not prove honest execution; isolation does not remove selection bias; and controlled noninterference tests cannot prove every forbidden read is impossible. Combine enforcement, independent reconstruction, and retained real-input evidence.
**Residual experimental risk:** more rigorous feasibility filters or budget rules can change the searched distribution. Re-baseline paired comparisons when those policies change. Do not add a hidden complexity penalty or interpret a safety improvement as accuracy improvement.
**Evidence ownership:** the GP owner should deliver §§5.1–5.10 closure artifacts; evaluation, runtime, and export owners must jointly sign off only their interfaces. Preserve missing and negative outcomes rather than granting a blanket “verified” status.

[^preprocessing]: External method reference, not product evidence: scikit-learn official documentation, “Common pitfalls and recommended practices,” §12.2, explains fitting preprocessing only on training subsets and separating fitting from transformation. Consulted 2026-09-25. Source: `https://scikit-learn.org/stable/common_pitfalls.html`.
[^selection]: External primary research, not product evidence: Gavin C. Cawley and Nicola L. C. Talbot, “On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation,” Journal of Machine Learning Research 11 (2010), 2079–2107. Its abstract identifies overfitting of selection criteria and resulting evaluation bias. Source: `https://www.jmlr.org/papers/v11/cawley10a.html`.
