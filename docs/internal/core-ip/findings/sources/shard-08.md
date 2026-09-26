<!-- from-tar tab=8 url=https://chatgpt.com/c/6ab7032f-c978-83ea-9fcc-05afbd25ea2d sha256=fbae0fbaf3eba5dee53e009539e96425ac1f5b605873fa8a9a223fa7e6b5e3d1 chars=41713 -->

# Scores, KPIs and gates — core IP technical review

**Review notes:** Shard 8 of 10; batch tab 8. Scope: §7, with other sections referenced only where they determine score validity or a KPI consumer's authority.
**Publication evidence cutoff:** 2026-09-25T19:31:50Z. **Publication source, as supplied:** `3d71fce041d1dd1173ed98160f7d986bb426ba6f`.
**Evidence basis:** The pasted structured extract, not the complete monograph. Most of §§7.2–7.6 is absent; their TOC headings establish topics, not implementations or measured results. No product source, evaluator, receipt bundle or benchmark was inspected or executed.
**Reading key:** DOCUMENTED means explicitly stated in the extract; DERIVED means an implication of its equations or contracts; UNVERIFIED means the implementation, policy detail or outcome cannot be established here. Suggested types, states and tests below are proposals, not existing APIs.
**Severity:** Ranked by consequence at the named consumer. CRITICAL denotes possible corruption of a learning decision or a load-bearing capability claim; it is not a claim that an exploit ran. Implementation findings require the specified negative tests before closure.
**Decision:** Preserve the separation between development scores, teaching eligibility, HWM movement, transfer evidence and historical ranking. Do not authorize a stronger claim merely because a weaker quantity is present. The extract supports neither a numerical transfer estimate nor an Elo reconstruction.

## Executive hits

- **Gap — transfer has a contract, not a measured result.** §7.5's supplied title explicitly says “reporting contract, not yet measured”; §7.1.1 requires separate training-side and outer identities. A development win cannot establish transferable learning. Enforcement is UNVERIFIED. [G01]
- **Improvement — make evidence status executable.** Bind each displayed claim to its permitted consumer, comparison identity and validated receipts. A complete negative transfer result must remain publishable as measured; missing evidence must remain unmeasured, never zero or success. [K01; §§7.1.1, 7.2, 7.5]
- **Gap — authenticated OOF is not automatically authorized evidence.** §11.1 reports that live eligibility does not call the added encoder-receipt validator. This limits what §7.1.2 admission and HWM KPIs can certify about fit scope, regardless of their numerical values. [G02]
- **Improvement — enforce provenance where the decision is made.** Reject absent, mismatched or unauthorized fit-scope evidence before creating an eligible-teacher or comparable-HWM event; require an authenticated evaluator identity and retained rejection reasons. [K02; §7.1.2]
- **Gap — stated denominators still permit implementation mistakes.** “Complete paired audit tasks” is conditional on completion; successful warm requests, admitted teachers and comparable HWM events are different populations. Selective omissions or replayed events could flatter several KPIs without changing model quality. [G03; §§7.1.2, 7.6]
- **Improvement — rebuild every KPI from a failure-inclusive journal.** Reconcile planned work, actual scheduling, submissions, completion, rejection and comparison outcomes; retain the exact declared denominator and publish coverage beside conditional results. [K03; §7.6]
- **Gap — equations do not finish the scoring contract.** Class order, undefined metrics, fold aggregation, training-only baselines, lift units, tie rules and Elo construction are not recoverable from the supplied §7 slices. Four-of-five admission is explicitly not a worst-fold guarantee. [G04–G09]
- **Improvement — prioritize executable score specifications over another composite score.** Pair evaluator conformance tests, atomic HWM receipts, a matched transfer audit and an isolated historical-Elo replay. These create falsifiable evidence without asserting new performance. [K04–K10; §§7.1–7.6]

## Ranked IP gaps (CRITICAL/HIGH/MED/LOW)

### G01 — CRITICAL: a transfer label could outrun its evidence
**§ref / status:** §§7.1.1 “Primary-split improvement and transfer,” 7.2 “Preserved metric definitions and transfer contract,” 7.5 “Observed transferable lift: reporting contract, not yet measured.” The distinction is DOCUMENTED; live claim gates and the full transfer rule are UNVERIFIED.
**Attack / gap story:** A consumer could accept a positive primary-split delta, a checkpoint update, or a report-shaped object as proof of transfer. The extract requires both nested comparison identities and execution scopes, but does not expose the rule computing lift, uncertainty or a successful transfer claim.
**Detection:** Present a primary-only result, an outer-only result, mismatched task identities, an unauthenticated result and an incomplete pair to every transfer-report/promotion entry point. Separately present a complete, valid audit with adverse results. Check that completeness and benefit are not the same boolean.
**Harden:** Require typed, authenticated evidence for both sides before classifying an audit as measured; evaluate any benefit claim under the recovered §7.2 rule. Reject unsupported promotion while preserving descriptive scores and complete negative findings. A prose heading must never be the authority for a capability badge.

### G02 — CRITICAL: eligibility arithmetic may certify evidence with an unclosed fit scope
**§ref / status:** §7.1.2 “Admission and HWM,” with §11.1's explicit statement that `eligibility.rs` does not call the encoder-receipt validator. That call-path limitation is DOCUMENTED by the monograph, not independently verified; equivalent enforcement elsewhere is UNVERIFIED.
**Attack / gap story:** A candidate could have authentic fold scores yet derive its context or ancestry from a forbidden partition. Counting that candidate as an eligible teacher or comparable HWM event would make the KPI stronger than its evidence. Authentication identifies a producer; it does not by itself establish permitted inputs.
**Detection:** Hold candidate scores and fold wins fixed while substituting a mismatched context receipt, a missing ancestor scope or a receipt from another cut. Exercise the actual eligibility consumer, not just the opt-in validator. Observe both the rejection and the absence of downstream learning-authorizing events.
**Harden:** Bind the authorized fit-scope closure and evaluator authority to the score receipt consumed by §7.1.2. Missing or incompatible evidence must block admission/comparison, while the submitted record remains visible in the admission denominator. Reconcile historical counts separately rather than retroactively treating new checks as past qualification.

### G03 — HIGH: conditional denominators can hide attrition or count retries as progress
**§ref / status:** The KPI table in §7.1.2 and §7.6 “A KPI ledger with defined denominators.” Several denominators are DOCUMENTED; their event predicates, retry rules and journal reconciliation are UNVERIFIED.
**Attack / gap story:** A dashboard could divide by successful requests instead of all planned requests, discard malformed teacher submissions, include only improving HWM events, or report transfer only on tasks where both arms survived. Repeated delivery of one result could inflate wins. A search budget ceiling is also not automatically a scheduled-attempt inventory.
**Detection:** Inject a timeout, preflight refusal, parse rejection, transport replay, explicit new resubmission and a missing paired arm. Rebuild every KPI from the original plan plus append-only events. Check that failures cannot disappear and that genuine new submissions are distinguishable from duplicate delivery.
**Harden:** Preserve the specified conditional denominators, but display their population counts, completion coverage and exclusion reasons. Record scheduling before execution and receipt of submissions before validation. Freeze a numerator predicate for each KPI; do not infer “completed” from a file's existence or “win” from the presence of a score.

### G04 — HIGH: an apparently valid scalar can have the wrong metric semantics
**§ref / status:** §7.1 equations (7.1)–(7.4); §7.3 “Metrics and class order,” whose body is missing. Equations are DOCUMENTED; degenerate-case, dtype, class-mapping and task-extension policies are UNVERIFIED.
**Attack / gap story:** Empty inputs, constant-target R², single-class AUROC, non-finite residuals, invalid probabilities or a reordered class column can yield undefined or misleading values. The displayed AUROC equation is binary; a multiclass/multilabel extension cannot be inferred. Reference implementations illustrate why defaults matter: R² can replace non-finite values, while log loss and AUROC impose class-order conventions.[^r2][^logloss][^auc]
**Detection:** Test empty and constant folds, absent classes, score ties, NaN/Inf, extreme residuals, invalid probability sums, and declared versus undeclared class permutations. Compare the metric producer, gate and reporter under the same explicit configuration. An unavailable metric must not become an ordinary finite win or an omitted fold.
**Harden:** Version metric identity with direction, input meaning, class/positive-label mapping, weights, reduction, numeric policy and supported task domain. Preserve raw value and validity status separately. Probability clipping, renormalization and undefined-value replacement require declared policy; library defaults are not evidence that JopeDime implements that policy.
**Direction check (DERIVED from §7.1):** RMSE and log loss are minimized; R² and AUROC are maximized. A comparator that applies one raw sign convention to all four can invert wins. Test native-direction decisions before applying any separately specified lift or ranking transformation.

### G05 — HIGH: fold aggregation, baseline choice and lift normalization can change the winner
**§ref / status:** §§7.1, 7.1.2, 7.2; §2.1.1.1 explicitly describes a mean of five training-side fold losses. That selection aggregation is DOCUMENTED; its binding across other consumers and the exact lift definition are UNVERIFIED.
**Attack / gap story:** From the supplied equations, averaging fold RMSE is generally different from pooled OOF RMSE, even with equal fold sizes. Pooled AUROC also adds cross-fold score comparisons. R²'s evaluation-set mean in its denominator is not the separately fitted training-mean predictor used by an admission guard; these baseline comparisons are not interchangeable.
**Detection:** Retain one prediction bundle with unequal fold sizes and heterogeneous residuals, then evaluate every declared reduction without relabeling them. Use folds whose training and scoring means differ. Check that the admission baseline is fitted without that fold's held labels, and inspect treatment of a zero baseline or signed metric in any relative-lift calculation.
**Harden:** Make fold reduction, target scale, baseline fit scope and lift units part of the comparison identity. Recover, rather than invent, §7.2's delta definition and cross-task aggregation. A regression training-mean guard does not specify classification baselines; keep unsupported tasks explicit. Never pool raw metrics across incompatible target units under one “lift” label.

### G06 — HIGH: an HWM improvement needs a committed transition, not merely eligibility
**§ref / status:** §7.1.2 binds HWM to task, dataset hash, cut, evaluator, vocabulary and protocol, and separates it from admission. Those requirements are DOCUMENTED; serialization, bootstrap treatment, replay handling and full identity coverage are UNVERIFIED.
**Attack / gap story:** A stale challenger could be compared with an outdated HWM, a newly initialized identity could count as an improvement, or a replay could move the numerator twice. The documented identity may be sufficient if its evaluator/protocol hashes cover scoring details; the extract does not demonstrate that binding.
**Detection:** Change each bound field in turn, replay an accepted event, initialize an empty identity, and race challengers against the same previous state. Compare the final journal with the actual committed transitions. Include ties and eligible-but-worse challengers, which must not be counted as improvements by mere admission.
**Harden:** Commit comparison evidence, expected prior-HWM version, decision and new state atomically. Count comparable non-improving events in the denominator, record bootstrap separately, and retain a canonical transition order. Improvement-event counts can depend on legitimate arrival order; report that history rather than presenting the count as an order-independent model-quality measure.

### G07 — HIGH: matched-looking transfer can still be adaptive or confounded
**§ref / status:** §§7.1.1, 7.2, 7.5. Reusing a primary development cut does not create independent evidence, as explicitly DOCUMENTED. The audit's full matching, look accounting, uncertainty method and target population are UNVERIFIED.
**Attack / gap story:** Selecting the best checkpoint or seed after viewing outer results, granting one arm extra search effort, or counting correlated folds as independent tasks could manufacture an impressive transfer summary. Nested evaluation separates selection from evaluation, but repeated adaptive access to its outer results requires its own control.[^nested][^adaptive]
**Detection:** Compare parent/challenger receipts for task membership, actual model identity, budgets, random-stream definitions, stopping, evaluator and preprocessing. Inspect the history of outer-result access and all challengers considered, not just the published pair. Distinguish new-row, new-dataset and new-family claims.
**Harden:** Freeze the audit estimand and matched protocol before outer access; preserve each arm's full outcomes and the declared look policy. Analyze paired effects at the stated sampling unit, retain task/family heterogeneity, and expose uncertainty. Repeated warm timing requests and overlapping folds do not become new independent transfer tasks by receiving new run IDs.

### G08 — HIGH: “win” is underspecified, and a fold majority does not bound harm
**§ref / status:** §7.1.2's declared fold wins; §11.1 explicitly notes that four-of-five admission is not a worst-fold guarantee. That limitation is DOCUMENTED. Exact tie, margin, missing-fold and cross-KPI win rules are UNVERIFIED.
**Attack / gap story:** Small favorable changes on the passing folds can coexist with a severe adverse fold. Counting near-ties as wins, treating an undefined fold as excluded, or mixing fold wins, task wins, HWM updates and Elo matches can exaggerate evidence without violating an informal “wins” label.
**Detection:** Construct qualitative fixtures with slight improvements on the passing folds and a severe remaining regression; add exact ties, differences around the declared tie boundary, an invalid fold and duplicate fold IDs. Inspect the legacy admission outcome separately from aggregate effect and worst-fold diagnostics.
**Harden:** Keep win/loss/tie/invalid outcomes distinct at each level. Preserve the existing four-of-five policy under its historical identity, but add effect-size and tail-risk diagnostics. Any new veto or minimum-benefit threshold must be frozen and versioned before use; do not assert it is already part of eligibility.

### G09 — MED: historical Elo cannot be reconstructed from a heading and an anchor label
**§ref / status:** §7.4 “What Elo means here” and the §7.1.2 ledger restriction to historical ranking. The historical-only consumer is DOCUMENTED; the match graph, score-to-outcome mapping, rating procedure and numerical settings are UNVERIFIED.
**Attack / gap story:** Different opponent schedules, repeated matches, model aliases or disconnected graph components can make a rating summary misleading. An anchor alone does not identify the missing construction, nor does a relative historical ranking establish current accuracy or transfer. No conventional Elo formula or parameter should be substituted for the missing §7.4.
**Detection:** Request the original match events with exact competitor/task/protocol identities, outcome conversion, anchors, initialization and update/reduction order where applicable. Replay the actual algorithm. Inspect disconnected components, duplicated events and whether cross-component comparisons are data-supported or imposed by conventions.
**Harden:** Publish a versioned historical replay with graph coverage and sensitivity diagnostics, retaining source/date identities. Keep Elo outside teaching, HWM and transfer gates. Escalate severity if a consumer currently uses it for those decisions; absent that evidence, this remains a historical reproducibility and interpretation risk.

### G10 — MED: generic “score” and “success” fields can silently change the subject
**§ref / status:** §7's opening consumer separation and §7.6, cross-linked to §§6.1 and 10.1.1–10.2. The distinction between objective units, timing coverage and model quality is DOCUMENTED; reporting-schema enforcement is UNVERIFIED.
**Attack / gap story:** A lower log-domain runtime loss could be labeled faster inference; a parity tolerance could be labeled predictive accuracy; a source-ready event could count as a completed request. Sequential backend milestones could be shown as independent per-language end-to-end timings. A later publication hash could obscure an older measurement identity.
**Detection:** Feed a loss-only training receipt, a failed qualification, a source-only export and cumulative backend milestones to the reporting layer. Verify permitted labels, timing populations and original measurement identities. Check that a task's accuracy evidence is not multiplied by its warm timing repetitions.
**Harden:** Separate metric kind, units, subject, lifecycle milestone, validity and evidence status. Generate consumer-specific views from the same underlying receipts; block unsupported conversions. Preserve qualified-latency coverage beside conditional latency summaries, and never manufacture a headline latency for a failed qualification.

## Ranked improvements / kickoffs (P0/P1/P2)

### K01 — P0: receipt-backed claim authorization [closes G01]
**§ref / why:** §§7.1.1, 7.2, 7.5. The highest-value improvement is an enforceable distinction between measured evidence and a beneficial result, not a larger headline score. Suggested owner: evaluation lead with the reporting/serving decision owner.
**Kickoff:** Define separate evidence states and consumer permissions for development observations, complete transfer audits and benefit claims. Bind both nested comparison/execution identities, metric specification, task inventory, model subjects and retained receipts. Recover missing §7.2 semantics before implementing a benefit predicate.
**Falsifiable acceptance:** Every incomplete, substituted or mismatched audit fixture is barred from a transfer-capability claim; a complete adverse audit is accepted as measured and displayed with its adverse outcome. Report and gate make the same decision from the same receipt bundle, including after serialization/reload.
**Deliverable / value:** A validator and generated claim view with retained negative tests. Document the rejection path for each consumer and export the exact evidence references behind every claim; this turns §7's prose distinction into a reproducible product contract.

### K02 — P0: provenance-bound admission and comparison [closes G02]
**§ref / why:** §7.1.2, dependent on §11.1. Validity must be enforced at the live consumer, not inferred from the presence of receipt types. Suggested owner: learning-gate owner, with the fit-scope owner supplying the closure evidence.
**Kickoff:** Require validated evaluator authority and fit-scope evidence before issuance of eligible-teacher and comparable-HWM events. Bind the accepted evidence to the candidate, task/cut, preprocessing/context dependencies and protocol. Route explicit legacy compatibility through a non-qualified status rather than a permissive default.
**Falsifiable acceptance:** Missing, altered, expired-by-policy or cross-cut evidence cannot authorize either event; valid in-scope evidence can. A rejected submission remains in the submitted-record inventory. An integration test reaches the real consumer with the opt-in path disabled and still enforces the same predicate.
**Deliverable / value:** A trace from received submission to admission/comparison outcome, carrying the validator version and reason codes. Produce a separately labeled historical re-audit where raw evidence exists; leave unavailable historical cases unknown rather than inferring compliance from modern code.

### K03 — P0: failure-inclusive KPI journal and denominator reconciliation [closes G03]
**§ref / why:** §§7.1.2, 7.6. A well-defined fraction is useful only when its complete population can be reconstructed. Suggested owner: benchmark accounting owner jointly with the learning-event owner.
**Kickoff:** Persist plan identity, scheduled attempts, submission IDs, parent/child retry identities, stage outcomes and cutoff status before aggregation. Distinguish duplicate event delivery from genuinely new submissions. Store unknown/unresolved work explicitly; preserve both conditional results and their planned/submitted population coverage.
**Falsifiable acceptance:** Independently rebuilding the ledger from events reproduces every numerator, denominator and exclusion; injected failures cannot improve coverage by vanishing. Transport replay has no extra effect, while an explicitly new submitted record remains counted. Empty denominators yield unavailable fractions, not a fabricated rate.
**Deliverable / value:** One replayable accounting reducer and a rejection/attrition report, including interrupted campaigns. An attempt ceiling must never be substituted for actual scheduling; a complete report must explain every relevant inventory member under the declared cutoff policy.

| Ledger subject | Preserve the supplied denominator / baseline | Additional reconciliation required; no invented outcome |
|---|---|---|
| Warm success | All 100 planned measured requests; no baseline | Separate warmups and fresh-process observations; retain failed and unattempted planned requests. This is not evidence that 100 succeeded. |
| Completed GP candidates | Scheduled attempt inventory; no baseline | Declare the completion stage; distinguish scheduled, attempted, refused and completed work. A maximum budget is not this inventory. |
| Teacher admission | Every submitted candidate record; training-mean guard | Include rejected submissions; distinguish true resubmission from event-delivery replay; attach task-specific baseline policy. |
| HWM improvement | Comparable admitted events; current bound HWM | Retain comparable losses/ties and explicit incomparable/bootstrap cases; count only the declared committed improvement events. |
| Primary-split lift | Matched parent/challenger tasks; parent search | Bind task membership and actual matching; preserve missing arms and failed tasks beside descriptive complete-pair results. |
| Transfer lift | Complete paired audit tasks; parent matched protocol | Publish planned audit inventory, attrition and exclusion reasons alongside the conditional estimate; do not silently redefine the existing denominator. |
| Historical Elo | Historical match graph; declared rating anchor | Preserve the full graph, event identity and exact historical construction; no transfer or current-quality interpretation. |

### K04 — P0: versioned metric specifications and evaluator conformance [closes G04]
**§ref / why:** §§7.1, 7.3. A scalar without class order or validity semantics is not a safe gate input. Suggested owner: evaluator maintainer; use official library behavior only as a test reference where its configured semantics match the product contract.
**Kickoff:** Implement a registry covering direction, input domain, row alignment, class vocabulary/order, positive label, sample/target weights, reduction, target scale, log convention, clipping and non-finite policy. Derive documentation and consumer configuration from the same versioned specification; mark unsupported combinations explicitly.
**Falsifiable acceptance:** The degenerate fixtures in G04 have explicit, stable validity outcomes. Jointly permuting probability columns and their declared labels preserves the result; inconsistent metadata is rejected. Undeclared raw-byte column corruption is not assumed detectable: bind predictions to their producer/schema and test that provenance path too.
**Deliverable / value:** Retained adversarial fixtures and reference comparisons at owner-declared numerical tolerances, with no tolerance inferred from encoder parity. No default library conversion can silently turn an invalid metric into an eligibility pass or remove it from the inventory.

### K05 — P0: fold-and-baseline score receipts [closes G05]
**§ref / why:** §§7.1, 7.1.2, 7.2. This closes a route by which every individual calculation is plausible but the reported winner answers a different question. Suggested owner: selection evaluator owner with the teaching-baseline owner.
**Kickoff:** Retain fold membership, prediction identities, valid-row counts/weights, target scale, per-fold baseline-fit identity, per-fold scores and the explicit aggregation procedure. Keep selection RMSE, any pooled diagnostic and admission-baseline comparison distinct. Restore §7.2's lift definition instead of choosing an attractive substitute.
**Falsifiable acceptance:** The heterogeneous-fold fixture yields separately named reductions, with only the declared reduction reaching its consumer. A training-only mean baseline can be rebuilt without held-fold labels. Changing aggregation, baseline, class mapping or lift units changes the comparison identity or is rejected as incompatible.
**Deliverable / value:** An independent replay of selection, admission and reporting from the same prediction bundle, including undefined baselines. Proposed cross-task summaries must declare weighting and comparability before execution; otherwise retain per-task native metrics without manufacturing an overall normalized score.

### K06 — P0: atomic HWM decisions with idempotent receipts [closes G06]
**§ref / why:** §7.1.2. Curriculum updates require a committed comparison against the actual current state, not a best-effort metric increment. Suggested owner: HWM state owner.
**Kickoff:** Extend the comparison receipt to identify its admitted event, metric/protocol identity, prior-HWM version, comparison result and committed transition. Separate bootstrap, incomparable, non-improving and improving outcomes. Ensure evaluator/protocol hashes actually bind all score semantics rather than merely naming a scorer.
**Falsifiable acceptance:** Competing updates follow the declared serializable rule; stale state cannot authorize an unchecked overwrite. Replaying a receipt does not create another transition or numerator increment. A new dataset/evaluator identity cannot inherit the old HWM, and admission alone never counts as improvement.
**Deliverable / value:** A canonical transition log whose replay reproduces the final state and KPI ledger. Include concurrency and crash/recovery fixtures; publish order-dependent event counts as historical progress accounting, not as an intrinsic capability rating.

### K07 — P1: first complete, matched transfer evidence package [closes G07; depends on K01–K05]
**§ref / why:** §§7.1.1, 7.2, 7.5. This is the constructive path from “not yet measured” to an observed result of either sign. Suggested owner: independent evaluation owner, with dataset-family scope supplied by the data owner.
**Kickoff:** Restore the nested contract, then predeclare task/family population, parent/challenger identities, allowed adaptation, matched search effort, random streams, stopping, exclusions, aggregation and uncertainty method. Name the estimand: new rows, datasets or families. Freeze numerical decision settings before examining outer outcomes.
**Falsifiable acceptance:** The completed package accounts for every scheduled audit task and includes both nested identities, both arms' receipts, original outcomes and all deviations. An independent replay reproduces the declared aggregates. Missing closure, an undeclared outer look or a changed comparison protocol blocks the corresponding stronger claim.
**Deliverable / value:** Per-task paired results and an uncertainty/heterogeneity report at the declared sampling unit, plus the complete attrition ledger. A valid adverse or inconclusive result completes the measurement milestone; one completed audit is not automatically proof of broad transfer or training-caused benefit.

### K08 — P1: explicit win rules and a tail-risk companion view [closes G08]
**§ref / why:** §7.1.2 and §11.1's fold-guard limitation. Keep curriculum exploration and release-level robustness as separate decisions. Suggested owner: eligibility policy owner with the deployment risk owner.
**Kickoff:** Specify the metric direction, comparator, tie/margin treatment and invalid-fold handling for fold wins, task wins and HWM events separately. Add per-fold magnitudes, baseline-relative deterioration where defined and worst-fold visibility. Any risk veto is a new policy version, not a retroactive reinterpretation of four-of-five admission.
**Falsifiable acceptance:** The majority-win/adverse-tail fixture retains the legacy policy result while prominently surfacing harm; missing folds cannot shrink the required evidence set. Under a newly approved veto, the same case is rejected exactly as specified. Repeated fold IDs cannot satisfy a distinct-fold requirement.
**Deliverable / value:** Separate outputs for eligibility, aggregate effect and risk qualification, with no generic “wins” field accepted across levels. Calibrate future thresholds on permitted development evidence only; a nicer pass rate is not the acceptance criterion for the new policy.

### K09 — P1: reproducible historical Elo replay [closes G09]
**§ref / why:** §7.4. Preserve the historical ranking asset without letting missing construction or modern conventions rewrite it. Suggested owner: historical benchmark maintainer.
**Kickoff:** Recover the original method, event graph, outcome mapping, anchors, initialization and any order-sensitive settings. Bind each competitor to its historical subject and each match to its task, metric and protocol. Report disconnected components, coverage and the treatment of ties/failures rather than hiding them in one rank.
**Falsifiable acceptance:** The recovered implementation reconstructs the retained ratings/rankings within an explicitly declared historical numerical policy; incomplete inputs produce “unreconstructable,” not substitute Elo. Duplicate-delivery fixtures cannot add matches. Every exported rating remains marked historical and rejected by transfer/HWM consumers.
**Deliverable / value:** A deterministic replay and graph-sensitivity appendix distinguishing measured comparisons from anchor-imposed conventions. Where original evidence is irrecoverable, publish only attributable historical observations; do not claim a successfully reconstructed rating system.

### K10 — P2: generated score cards and consumer-contract regression tests [closes G10]
**§ref / why:** §7's consumer contract and §7.6, dependent on §§6.1 and 10.1.1–10.2. This prevents correct underlying evidence from being overstated downstream. Suggested owner: reporting maintainer with each decision owner signing its view.
**Kickoff:** Generate each score card from validated receipts with subject, direction, units, denominator, baseline, reduction, scope, evidence status and measurement identity. Separate source-ready, qualified execution, latency and predictive accuracy. Keep publication revision and original measurement build as distinct provenance fields.
**Falsifiable acceptance:** Consumer-contract fixtures reject a runtime-loss-as-seconds label, parity-as-accuracy label, source-ready-as-success event and cumulative-milestone-as-independent-E2E claim. Missing evidence remains visible. Rebuilding a publication reproduces the cards without upgrading historical or synthetic evidence status.
**Deliverable / value:** A release/report check with machine-readable rejection reasons and traceable source receipts. Public cards may redact sensitive contents but must not imply that a public hash proves the truth of its private measurements; retain controlled auditor access to the needed evidence.

## Evidence gaps / UNVERIFIED load-bearing claims

The following items are closure requests, not claims that the complete monograph lacks them. The supplied excerpt cannot establish current transfer magnitude, Elo values, win totals, admission rates, HWM rates or numerical gate thresholds. None is estimated here.

| Item / status | § that would close the documentary gap | Operational evidence needed to close the stronger claim |
|---|---|---|
| Exact lift definition, units, reduction and benefit rule — UNVERIFIED | Full §7.2 | Executed evaluator plus matched raw outcomes; zero/undefined-baseline and metric-direction tests. |
| Transfer observed at the cutoff — NOT ESTABLISHED; supplied §7.5 title says not yet measured | Full §7.5 and §7.2 | Complete paired nested audit receipts and execution scopes, with failures and result-access history. |
| Classification mapping and unsupported-domain policy — UNVERIFIED | Full §7.3 | Class vocabulary/order receipts and adversarial producer-to-consumer conformance tests. |
| Historical Elo construction and reproduced ratings — UNVERIFIED | Full §7.4 | Original graph, competitors, outcome conversion, rating configuration/order and retained replay. |
| Realized numerators, exclusions and replay semantics — UNVERIFIED | Full §7.6 | Plan/submission inventory plus complete event journal and independently rebuilt ledger. |
| Fold-win strictness, tie policy and adverse-tail treatment — PARTLY DOCUMENTED | Full §7.1.2, with §11.1 | Fold-level evidence and consumer tests; distinguish existing majority admission from any proposed risk veto. |
| HWM key completeness and transactional enforcement — UNVERIFIED | Full §7.1.2, with §§6.2–6.5 | Identity-mutation, bootstrap, replay, concurrency and crash/recovery tests against the live state owner. |
| Authorized fit scope enforced before KPI admission — OPEN PER §11.1 | §7.1.2 plus full §11.1 and §12.1 | Actual eligibility call-path evidence and negative receipt substitution tests; a validator unit test is insufficient. |
| Consumer-specific units, timing success and measurement identity — UNVERIFIED | Full §7.1/§7.6, with §§10.1.1–10.2 | Qualified request journals and generated report outputs retaining original source/date subjects. |

## Fail-open / silent-lie paths (if any in shard)

Only the opt-in validator limitation and documented policy/evidence boundaries are established by the extract. The paths below are targeted failure hypotheses; this review did not execute them. A passing rejection test would falsify the corresponding implementation concern.

| Path to test | False or overstrong conclusion | Required fail-closed behavior |
|---|---|---|
| Primary-cut result substituted for transfer evidence [G01] | “Learning transferred” despite an incomplete audit | Preserve development observation; refuse measured-transfer/benefit authorization lacking its receipts. |
| Valid fold scores with missing or foreign fit-scope evidence [G02] | “Eligible” or “HWM-comparable” means scope-clean | Reject the authority-bearing event; retain the rejected submission and reason. |
| Failed arm/request omitted before aggregation [G03] | Strong performance on an apparently complete workload | Keep failure/attrition inventory; label complete-pair and qualified-only summaries as conditional. |
| Undefined score coerced to a favorable scalar or removed [G04] | A mathematical edge case becomes a win | Preserve invalidity and block that gate outcome under the declared metric policy. |
| Pooled score, R² baseline or new lift units substituted [G05] | A different estimand appears to beat the bound baseline | Refuse incompatible metric identity; keep each legitimate diagnostic separately named. |
| Stale, bootstrap or duplicate HWM event counted as progress [G06] | Artificial curriculum advancement | Require the committed transition and exact comparator identity; separate initialization and replay. |
| Best outer seed/checkpoint selected after repeated looks [G07] | Adaptive selection presented as independent transfer | Preserve look history and restrict the claim to the authorized evaluation protocol. |
| Missing/duplicate fold IDs or near-ties treated as fold wins [G08] | Admission satisfies an apparent fold majority | Validate distinct membership and the declared tie/invalid rules before counting. |
| Unreconstructed Elo, loss units or timing milestones reused [G09–G10] | Historical rank, optimization proxy or partial execution implies capability | Enforce historical-only and unit/milestone-specific consumers; reject unsupported conversions. |

## Cross-links to other sections the owner must not ignore

**Data and context — §§2.1, 2.2, 2.5–2.6; §§3.3–3.6:** KPI identity needs authorized membership/context evidence and a declared transfer population. A frozen row panel alone does not certify new-family transfer; these owners supply evidence, not alternative score definitions.
**Matched search — §§5.2, 5.5, 5.8–5.9:** Recover actual effort, random-stream pairing, stopping, selection and refit subjects before describing parent/challenger tasks as comparable. Selection OOF and outer evaluation must remain distinct consumers.
**Learning and HWM — §§6.1–6.6:** Authenticate teacher and update receipts, but do not promote lower training loss or changed parameters into measured transfer. Preserve eligibility, HWM comparison and later audit outcomes as different events.
**Inference and timing — §§8.1–8.2; §§10.1–10.2:** Fitted-artifact parity, successful export and predictive quality are different facts. The KPI owner must bind the correct fitted subject and respect cumulative backend timing and qualification boundaries.
**Historical identity — §§4.2, 4.5–4.6; §9.1:** Parent/challenger and Elo competitors need exact subject identities; historical reset/continuation labels or corpus availability are not additional independent evaluation evidence. Inventory the score subjects without reopening the model-inventory review.
**Enforcement and evidence status — §§11.1, 12.1:** A source path or validator type does not establish a live consumer gate. Preserve the distinction between implemented, locally tested, operationally measured and promoted when closing every §7 finding.

## Concerns / residual risk

**No invented results:** No product metric was recalculated, no code-path rejection was verified, and no positive/negative transfer result is asserted. All kickoff acceptance tests are proposed. External references below motivate edge-case tests only; they cannot fill the missing product-specific §7 contracts.
**Statistical limits:** Matched identities and honest denominators prevent some misleading claims, not all selection bias or uncertainty. A small or family-clustered audit can remain inconclusive; a valid negative result is useful evidence, not a failed measurement process. (§§7.1.1, 7.2, 7.5.)
**Policy tradeoff:** A tail veto may improve robustness while reducing teacher diversity; the tradeoff requires explicit development-side validation. Keeping historical four-of-five admission is defensible only with truthful claims about what it does and does not guarantee. (§7.1.2; §11.1.)
**Historical and operational limits:** Missing receipts may prevent retrospective qualification. Hashes authenticate bytes, not experimental truth; equivalent validation elsewhere could narrow G02 after review. Report residual unknowns rather than declaring the entire system unsafe or qualified. (§12; §11.1.)
**Priority decision:** Treat K01–K06 as blockers for stronger KPI-authorized learning/transfer claims, not as a demand to stop honestly labeled development work. K07–K09 build the missing measurement/replay evidence; K10 protects downstream presentation. No patent novelty or legal conclusion is implied by these engineering recommendations.

### External primary references used only for test rationale

[^r2]: scikit-learn developers, `r2_score` documentation, checked 2026-09-25: constant-target non-finite handling and explicit multioutput reduction. `https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html`
[^logloss]: scikit-learn developers, `log_loss` documentation, checked 2026-09-25: class-column interpretation, dtype-dependent clipping, reduction and logarithm convention. `https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html`
[^auc]: scikit-learn developers, `roc_auc_score` documentation, checked 2026-09-25: positive-class semantics, class order and explicitly selected multiclass treatment. `https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html`
[^nested]: scikit-learn developers, “Nested versus non-nested cross-validation,” checked 2026-09-25: separation of model selection and outer evaluation. `https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html`
[^adaptive]: Dwork et al., “Generalization in Adaptive Data Analysis and Holdout Reuse,” 2015, arXiv:1506.02629v2, abstract checked 2026-09-25: adaptive holdout reuse can overfit the holdout. No claim is made that the product implements the paper's method. `https://arxiv.org/abs/1506.02629v2`
