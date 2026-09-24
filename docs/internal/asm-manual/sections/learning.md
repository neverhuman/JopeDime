# Measuring repeated learning and deciding improvement

## What “better and better” must mean

A genuine flywheel joins four measured improvements. GP finds a better executable teacher for an unchanged comparison. Training actually consumes that new information. The policy becomes better at proposing or ranking programs for the relevant dataset-and-split context. Finally, the next bounded GP search improves because of that policy. Each arrow needs its own evidence; a growing corpus, a saved checkpoint or a completed search cannot stand in for the whole chain.

Four claims therefore stay distinct, and each needs its own evidence: an **eligible teacher** (a target admitted under its full comparison identity), a **new discovery** (a program absent from the authenticated initial population), **actually consumed information** (the optimizer sampled it in committed updates) and an **improved policy** (a measurably better next fitted search). None implies the next. Initial populations must be authenticated, and a discovery made in one arm must not silently seed another arm.

The comparison identity is essential. A target belongs to a task, dataset content, exact split and groups, evaluator, vocabulary and numerical protocol. If any changes, the score is a new experiment rather than a better version of the old target. Context identity also binds the fitted preprocessing, encoder assets and allowed fitting rows. Two vectors with the same width need not represent the same problem.

An incumbent can improve monotonically under a fixed development objective while real generalization gets worse through repeated selection. We therefore need both a target-quality ratchet on development data and independent evidence that the learned search transfers to other families. The latter is the north star (proposed; owner ratification pending; defined in [§](status.md#the-proposed-north-star-and-matched-comparisons)). Repeatedly attacking one dataset is useful specialization; it is not evidence of broad transfer.

## What the historical training and target evidence establishes

ASM means **Algorithm Synthesis Model: learns to generate and optimize executable algorithms from data**. The repository has real historical neural training and real retained GP teaching corpora. It also has substantial uncompleted work before a newly bounded ASM learning campaign can establish useful improvement. These facts must be stated together. “Zero freshly admitted teachers under the new ASM admission guard” describes the new campaign's inspected readiness; it does **not** mean that JopeDime has never trained on real data, that the accepted policy is untrained, or that its historical ten-million-record corpora are synthetic.

This chapter reconstructs progress from original retained metrics and checkpoint metadata without loading model tensors, reopening sealed targets, or starting training. It separates three kinds of evidence: loss measurements, changed training state, and improved executable algorithms. Losses measure how the model fits its training objectives. Checkpoint lineage records which parameters and optimizer state were produced. Only a matched downstream fit/search comparison can establish that conditioning on a dataset and split produces better algorithms. A completed optimizer run does not imply the third result.

The retained evidence shows both substantial historical training corpora and some improving historical GP targets. It does **not** show a newly qualified ASM teaching loop, a current thirty-dataset result, or a causal advantage from neural training. Those are different claims with different evidence requirements.

## LOOP-01: the central regression milestone

LOOP-01 is the central regression product milestone. One LOOP-01 generation is the complete chain:

**new eligible target → actual exposure → changed parameters → compatible reload → improved fitted search → another new generation**

Each arrow is a separate claim with its own evidence. A new eligible target passes the admission guard under its full comparison identity and is absent from the authenticated initial population. Actual exposure is measured in committed optimizer updates, not inferred from corpus membership. Changed parameters are shown by parameter digests before and after training. A compatible reload loads the exported checkpoint in the consumer with its architecture, vocabulary, encoder and source identities intact. An improved fitted search is judged by the preregistered rule in [§](#the-proposed-preregistered-regression-decision). Another new generation means that search yields new eligible targets for the next round. Two passes through one corpus are continuation, not two generations.

Status of each link as of 2026-09-23 at about 23:55 UTC:

| Link | Mechanism | Status | Evidence now |
|---|---|---|---|
| New eligible target | Five-fold admission guard and strict HWM identity in the pending eligibility source ([§](teachers.md#the-admission-guard-label-exposure-and-hwm-identity)) | pending integration | Zero teachers admitted under the new contract |
| Actual exposure | Per-update fresh-exposure record and initial-population/discovery identity | proposed | Not measured; the stopped training campaign records neither |
| Changed parameters | Campaign verification of positive updates and changed parameter digests ([§](training.md#plan-run-resume-verify-and-export)) | pending integration | Not measured for a new lineage. The retained step 60,100 run did genuine optimizer work on the unchanged corpus with zero fresh program presentations: continuation, not a generation |
| Compatible reload | Worker reload of `policy.pt` and `resume.pt` with equal parameter proof, then the fail-closed `experimental.regression` consumer ([§](training.md#compatibility-checks-and-fail-closed-rules)) | pending integration | Not measured |
| Improved fitted search | Consumption of the export in a new operational proof, judged by the proposed decision rule | proposed | Not measured |
| Another new generation | A further teaching campaign over searches conditioned on the reloaded checkpoint | proposed | Not measured |

The current one-dataset origin proof, which closes context → proposal → fit → evolution for the accepted model, is available evidence; it is not a LOOP-01 link. No LOOP-01 generation has run.

### Lineages, continuation names and generations

The accepted **512×6 step-60,000** model of lineage B (width 512, six layers) remains the serving model and a report baseline. The pending campaign starts a separate line, **experimental regression lineage v1 (256×2, planned new initialization)**: the width-256, two-layer Python model described in [§](training.md#what-the-pending-campaign-trains). Its newly initialized model, `M_init`, is called `M0` only after it has actually trained on the first admitted corpus, `C0`. Step 60,100 is historical hardware continuation of step 60,000, and step 69,000 remains unaccepted ([§](architecture.md#checkpoint-binding-and-configuration)); neither is a LOOP-01 generation. The two lines are compared only through matched downstream fits, never through loss values.

Every training attempt must declare one of four continuation names (proposed naming): **cold initialization**; **weight warm start**; **explicit new-corpus transition**, which declares whether optimizer state is carried and how the sampler is initialized; and **exact interrupted resume**, which keeps the original state, inputs and deadline. Exact resume must reject an optimizer fallback before any update and prove that interrupted and uninterrupted runs reach the same state. Both requirements are proposed: the stopped campaign checks optimizer resumption only after training and has no interrupted-versus-uninterrupted parity test. A new generation needs a new corpus containing new eligible targets; a second pass or an exact resume on the same corpus continues the current generation.

### Two controlled rounds within the existing GPU cap

LOOP-01 calls for two linked controlled rounds. Each round (proposed) trains two sibling branches from one immutable parent model and optimizer state, under the round's original deadline:

- **F (fresh):** samples 25% fresh and 75% replay;
- **R (replay-only):** samples replay only.

Both branches commit the same number of updates and batch slots. For each branch, record the measured fresh exposure, duplicate and unique families, non-padding tokens and GPU cost; keep the unchanged parent as a comparison artifact. Neither branch is promoted to serving. The stopped training campaign has no sibling branches, immutable sibling parent or 25/75 sampler allocation.

The existing allocation gives the regression category at most 20 minutes of GPU time inside the one-GPU-hour campaign ([§](scheduling.md#resource-policy-and-the-training-clock)), and LOOP-01 grants no further allocation. The two controlled rounds may therefore remain unfunded, or inconclusive within that cap. Either outcome is reported as mechanical proof and descriptive results, with the causal question marked inconclusive.

## The closure ledger

| Link in the loop | Existing evidence | Remaining proof | Roadmap connection |
| --- | --- | --- | --- |
| Dataset and split → valid context | Current regression producer checks original rows, preprocessing, assets and eight reference cases | Classifier context/checkpoint consumption and newly trained context compatibility | NEURAL / TRAIN items |
| Context → real initial GP candidates | Current 80-slot path (64-slot policy pool plus 16 exploration, with explicit rejection fallback) has authenticated genuine `neural_policy` fits | Show the newly trained checkpoint produced the measured candidates | NEURAL / TRAIN items |
| GP evaluation → improved target | Current bounded search fits and evolves candidates; historical champion and corpus records exist | New admission needs complete finite OOF evidence, baseline improvement on at least 4/5 folds and the full identity | TEACH items |
| Improved target → frozen teaching corpus | Historical binary teaching stores and corpora survive; pending exporter authenticates inputs | Execute the new eligible-state export with complete admitted-family coverage; record fresh versus repeated examples | TEACH / TRAIN items |
| Corpus → parameter update | Historical optimizer logs and accepted step-60000 exist | New bounded run must prove positive updates, changed parameters, reload and true optimizer continuation | TRAIN items |
| Parameter update → better policy | Historical imitation/ranking diagnostics exist | Compare parent and challenger on unchanged diagnostic inputs; then measure fitted-program benefit under matched budgets | NEURAL / TRAIN items |
| Better policy → better next search | Current accepted model reaches actual evolution on one dataset | Run the exported challenger in a subsequent search and retain a causally interpretable comparison | NEURAL / PANEL items |
| Improved search → next useful training generation | Historical controller and target chains exist | Complete at least two linked new cycles (flywheel turns, that is LOOP-01 generations as defined in [§](#loop-01-the-central-regression-milestone), not training blocks or passes) with eligible new targets, actual exposure and measured downstream outcomes | End-to-end flywheel qualification |
| Qualified checkpoint → serving | Product is bound to recovered accepted step-60000 | Design and qualify the missing experimental-to-serving promotion contract, then require the new checkpoint to pass it; training-pointer movement alone does not move the serving pointer | FUTURE-01: deferred; no executable acceptance command yet |

The prefixes above refer to the detailed roadmap's existing workstreams; they are not new executable command names. The current one-dataset proof closes context→proposal→fit→evolution. It does not close the new teaching→training→improved-search cycle. The historical training chapter separately reports what earlier controllers actually achieved, including unavailable or inconclusive promotion evidence.

## A minimal experiment that can establish learning

### Freezing the experiment and choosing families

Freeze the experiment before opening the run window: source and binaries, exact parent checkpoint, corpus membership, encoder and vocabulary, objective weights, optimizer carry policy, sampler, random streams, fit/search budgets and comparison metrics. Keep the accepted parent available as a control. A new objective, new encoding or incompatible optimizer state starts a new lineage; it must not inherit a misleading uninterrupted-step counter.

Use eligible teachers from independently identified training families. Keep family-disjoint development diagnostics for the model. When a dataset was used to manufacture teachers, label subsequent evaluations on that dataset as exposed/adaptive. The pending path's two prospective teaching families, `230_machine_cpu` and `712_chscase_geyser1`, are both frozen-panel regression datasets ([§](teachers.md#present-teacher-readiness-and-the-smallest-real-path)); if they supply teachers, later panel reports must label those two datasets exposed/adaptive. Unknown historical checkpoint exposure remains unknown. A split name alone cannot prove an unseen family. Assign families, at family level, to three domains: teacher/train, adaptive development, and evaluation outside training and selection. The frozen thirty-dataset panel remains adaptive development, and the pretrained encoders' exposure to any family remains unknown. A hash authenticates bytes; it does not make an input eligible. The input closure covers records, contexts, transforms and all assets, and unused forbidden descriptors must be rejected before loading. These are proposed requirements: the stopped training packet validates only the contexts that its retained rows reference, does not reject extra context-index entries, and partitions families only into train and development domains.

### What to retain for each cycle

For each cycle, retain all of the following:

1. **Target creation:** incumbent and challenger recipe hashes, all five candidate and baseline OOF losses, complete fit outcomes and exact comparison identity. Authenticate the complete initial seed population. Distinguish a newly discovered program absent from that seed set from a carried incumbent, a neural seed and its evolved descendants. Retain discovering arm, program hash, generation, context, parent/ancestry, first measurement and later shared-seed exposure. Count each distinct dataset/cut/program discovery once before reporting per-arm reuse. Record rejected, tied, failed and unavailable candidates too.
2. **Corpus delta:** previous/new corpus hashes; added, improved, removed and unchanged comparison/program identities; distinct contexts and families; rejection reasons. Distinguish logical rows from unique programs and unique contexts.
3. **Actual optimizer exposure:** how often newly admitted programs and contexts appeared in committed steps, their weights, sample repetition, padding fraction and fresh/broad sampling fractions. New rows in a manifest are insufficient if the sampler never reaches them.
4. **Training continuity:** initialization parent, parameter hashes before/after, optimizer initialization versus restored moments, local and cumulative committed steps, effective batch and actual clock/resource use. A failed attempt's compute stays in accounting.
5. **Policy diagnostics:** component losses on a fixed, identified monitor; valid-program rate, uniqueness, conditional ranking and proposal quality. Report corpus/monitor changes as boundaries rather than drawing a continuous evaluation curve through them.
6. **Next-search outcome:** execute both parent and challenger proposals as actual fits under the same candidate/compute allowance, plus the generic control. Retain neural-origin and ancestry counts, failure rates, selected program identities and the downstream losses.
7. **Decision:** accept, reject or inconclusive with the exact gate and original evidence; for regression rounds, apply the proposed rule in [§](#the-proposed-preregistered-regression-decision). Preserve the old serving pointer unless promotion succeeds.

Two linked successful cycles are a practical demonstration of repeatability. They do not prove unlimited improvement. Plateaus, search-space ceilings, noisy estimates, sample reuse and model capacity remain real constraints. Report the improvement per cycle and cost per eligible improvement; do not promise a monotonic performance curve on unseen data.

### The proposed preregistered regression decision

This is the owner's proposed decision rule for LOOP-01 regression rounds. It is a proposal, not an achievement: no round has been judged by it. Preregister it, together with the monitor families, seeds, folds and trivial predictor, before any challenger is compared, and do not change it after seeing results.

#### Family-normalized gain

For each monitor family, the gain is `g = (L_control − L_challenger) / L_trivial`.

`L_control` is the comparison model's loss: the parent for the main test, and the replay-only branch for the fresh-versus-replay test. `L_challenger` is the challenger's loss, and `L_trivial` is the loss of the preregistered trivial predictor on the same family. Positive `g` means the challenger is better. Average over the declared seeds and folds within each family, so that each family contributes one value. The preregistration states explicitly how a family with zero or nonfinite `L_trivial` is treated; no hidden epsilon is added and no such family is dropped silently.

#### Round thresholds

A round shows a gain only if all of the following hold:

- at least 10 independent monitor families, each with complete comparisons;
- at least 9/10 of the families gain more than 1% against the parent;
- the mean gain across families exceeds 1%;
- no family is worse than −5%;
- the fresh branch beats the replay-only branch in at least 9/10 of the families, with a positive mean gain.

Round 2 uses monitor families separate from those of round 1.

#### Baselines, stopping and reporting

Generic search and the accepted step-60,000 model remain report baselines. A gain over the parent is a different claim from an advantage over generic search or over the product's accepted model. Stop the loop when there are no eligible discoveries, on an incompatibility, when there is no material gain, when the evidence is insufficient, or when the budget is exhausted. Preserve the best checkpoint and never promote to serving automatically. An unfunded round, or one with insufficient data, yields mechanical proof and descriptive results; its causal conclusion is "inconclusive", never an invented success.

## Distinguish routing, ranking and search recovery

The learned route is a conditional distribution over executable program tokens given the dataset-and-target context. It is richer than selecting a fixed named estimator. The same model also supplies learned ranking signals. Its reward representation mean-pools independent token embeddings, a bag of actions, with no positional term: at `a75e543d2` both the Python model (`python/src/jope/policy/model_v3.py:399-404`) and the product's Rust port (`crates/jope-policy/src/model.rs:555-567`) compute a padding-masked mean. It has no learned representation of token order: for the same context, programs with the same token multiset have the same pooled representation in exact arithmetic. This limits ranking distinctions between programs whose execution differs only through token order; it does not remove order from the autoregressive proposal decoder. See [§](architecture.md#decoder-and-reward-architecture); [§](#conditioning-diagnostics-before-the-first-causal-experiment) proposes measuring its collisions before any architecture change.

Measure the policy before GP recovery as well as after it. A long search can find the same winner despite poor neural proposals, while a short search may reveal useful initialization that disappears at a larger budget. Shared incumbents can create many ties. Equal time caps do not imply equal completed fit counts, especially when candidate runtimes differ.

The admitted historical evidence in [§](targets.md#did-gp-targets-get-better) illustrates this limit: at 1,000 completed evaluations, 3,268 paired neural/generic final-search observations contain 24 lower neural scores, 17 higher scores and 3,227 exact ties. Their geometric mean neural/generic RMSE is 1.0001869094. They repeat families and generations, consume unequal reported wall time and measure adaptive final winners. The historical arms also share incumbent, retained-teacher and exploration seed pools; a winner can seed every arm in the next generation. In the separate first-to-last table, 64 lower arm groups collapse to 23 distinct dataset/cut pairs out of 55, with identical available-arm final scores in 22 of those 23. Those are dependent carried-forward outcomes, not 64 independent discoveries or wins. They neither establish initial routing quality nor show that later training caused better search. A larger external extract is kept outside this claim where its rows have not joined the admitted inventory.

Useful controls answer different questions: generic grammar sampling tests end-to-end benefit over the non-neural search; the previous accepted checkpoint tests whether training helped; shuffled contexts test conditional information; and random ranking on the same candidate pool tests ranking utility. These additional controls are proposed experiments, not extra runs hidden inside the approved panel. Declare their budget and scope before execution.

## Conditioning diagnostics before the first causal experiment

These diagnostics ask whether the policy actually uses its dataset-and-split context. All are proposed and none has been measured. Run them before the first causal LOOP-01 experiment, and keep every diagnostic substitution outside teacher admission: a program proposed or ranked under a substituted context never becomes a teacher.

| Diagnostic | Question | Comparison | Status |
|---|---|---|---|
| Boundary correctness | Is the context built only from permitted fitting information, within an input closure that covers the exact records, contexts, transforms and assets? | Unused forbidden descriptors are rejected before loading | proposed |
| Functional sensitivity | Do proposals and ranking scores change when the context changes? | The true context against substituted contexts | proposed |
| Useful ranking | Does the learned ranking order candidates better than simpler signals? | Donor-context, program-only and random baselines on one fixed, already-fitted candidate bank | proposed |

The current regression producer already checks original rows, preprocessing, assets and eight reference cases ([§](#the-closure-ledger)). The stopped training campaign, however, does not reject extra context-index entries or unlisted transitive artifacts, so boundary correctness is not yet established for training inputs.

The reward path's order blindness is measured before it is changed. Both implementations mean-pool token embeddings without a positional term ([§](#distinguish-routing-ranking-and-search-recovery)). First measure how often distinct programs in one context collide on the same pooled representation, and the ranking regret those collisions cause. Defer architecture changes, so that the first causal experiment keeps architecture, objective, corpus and sampler changes distinguishable.

Dense-width feasibility checks must use only permitted fitting information, keep the original proposal limits and clocks, and report a preflight rejection separately from a launched fit failure; they grant no unlimited free retries. A naive preflight must not rely on the retryable `Unavailable` outcome that a job-preparation error receives at `a75e543d2` (`crates/jope-pipeline/src/neural_search.rs:1608-1623`); it needs a terminal outcome with its own accounting. The source investigation is custody `plan-execution-20260923/feasibility-phase-accounting.md`, SHA256 `0154e6b0074d6b3c7100abf28568abb41a998b6ed21730237d58649f31cc9f18`. This check is proposed source work.

## Specialization without starving the general learner

One difficult dataset can produce many near-duplicate teachers. Count unique canonical programs, meaningful improvements and distinct contexts, not only serialized rows. Retain broader replay and family balance so a successful focused attack does not make the policy forget other problems. [§](data-estate.md#packing-semantics-legal-sets-and-label-policies) describes how to share repeated contexts and immutable shards without losing original provenance.

Record the fresh-program exposure actually sampled. A nominal 50% fresh-comparison schedule need not yield 50% fresh program slots when each comparison also contains an incumbent and old comparators. Prefer measured exposure, a frozen baseline checkpoint, component-loss diagnostics and actual downstream fitting over assumptions based on the sampler's name.

If the target ratchet moves but fixed-monitor policy quality deteriorates, investigate the objective, label consistency, sampler and comparator support before simply training longer. If policy metrics improve but fitted searches do not, inspect proposal diversity, candidate runtimes, grammar coverage and GP recovery. If neither improves, report the plateau and preserve the best verified incumbent. The system earns a flywheel claim through this evidence, not through an endlessly running process.

## A KPI ledger with defined denominators

### KPI definitions

These are proposed reporting definitions for acceptance, not a claim that every current report already emits every field. Preserve the fixed protocol where it already specifies a quantity. Additions must be preregistered before comparing challengers; do not change a definition after seeing results.

| KPI | Definition and denominator | Evidence and present gap |
|---|---|---|
| Task loss delta | For a valid matched pair, `loss_reference − loss_challenger`; positive means better. Regression loss is RMSE; binary loss is `1−AUROC`; multiclass is ordered probability-matrix log loss. Report both generic and previous-checkpoint references separately. | Exact predictions, rows/classes, evaluator, numerical policy and resource receipts. Never average raw RMSE with log loss across tasks. |
| Strict wins / planned comparisons | Count valid matched improvements over all preregistered comparisons, including failed/unavailable comparisons in the denominator. Report ties, losses, one-sided failures and two-sided failures separately. | Complete planned inventory. Failure is not a fabricated numerical loss; paired mean delta is separately labeled as complete-pair-only with coverage. |
| PANEL-WINS (proposed definition; owner ratification pending) | Per task, the number of the 10 frozen-panel datasets (asm-private-30-v1, protocol asm-panel-v1) on which `asm.neural`'s confirmed five-fold mean development loss is strictly below the minimum over all eligible frozen non-ASM comparators (every `rust.*` and `external.*` recipe). `asm.generic` is an ASM control reported in its own column. An incomplete, failed or unconfirmed dataset is not won. Reported as (regression, binary, multiclass); the owner's goal of ten per task, (10, 10, 10), is a 30/30 sweep. | Not computed: no win-counting function exists in `crates/jope-bench/src` at `a75e543d2`, and the panel has not run. The statistic is adaptive and favours `asm.*`: the ASM recipe is selected on cut-0 fitting rows that the five-fold mean also scores, while `rust.*` and `external.*` recipes are fixed before the dataset screen (`crates/jope-bench/src/panel/selection_exposure.rs:91`; [§](benchmark.md#adaptive-development-exposure)). At source `68f56404` binary and multiclass have no neural arm ([§](architecture.md#experimental-classification-context)). |
| Family generalization | Per-task mean of family means under the preregistered split, with cluster uncertainty at family level when sample size permits | Frozen family provenance and prior-exposure evidence. No current frozen-panel (asm-private-30-v1) unseen-family claim; historical unknown exposure stays unknown. |
| Relative improvement | `(Lref−Lnew)/Lref` only for finite, strictly positive reference loss; report absolute delta when zero | No hidden epsilon convention or cross-task aggregate introduced after results. |
| Family-normalized gain (LOOP-01; proposed) | Per monitor family, `(L_control − L_challenger)/L_trivial`, averaged over the declared seeds and folds within that family; the denominator is the preregistered monitor families, each with complete comparisons | Not measured. A zero or nonfinite `L_trivial` needs an explicitly preregistered disposition ([§](#the-proposed-preregistered-regression-decision)). |
| Teaching admission | Eligible comparisons / all requested comparisons, and eligible independent families / all requested families | Preregistered five-fold train-side OOF evidence, finite baseline improvement and improvement on four of five folds. Newly demonstrated ASM-qualified count currently zero; historical corpus counts are separate. |
| Genuine neural consumption | Completed `neural_policy` generation-zero candidates / actual `neural_policy` slots; also report completed genuine neural candidates /64 policy-pool slots, fallback slots and completions, and exploration completions /16. Completed descendants require verified neural ancestry. | Authenticated proposal→request→started→finished→artifact chains. A recipe file, neural tag, generic fit or started-only attempt is insufficient. |
| Validity and fit coverage | Canonical task-compatible proposals / all proposed draws; completed fits / all scheduled eligible fits | Distinguish grammar refusal, dependency unavailability, fit error, timeout, cancellation, never-started and unknown. State whether duplicate candidates are deduplicated. |
| Optimizer progress | Updates this run, cumulative optimizer step, start/end parameter digests, committed passes, and exact resumed-state identity | Batch/epoch metrics and checkpoint proofs. Positive epoch count or a checkpoint filename is insufficient; training loss denominator and objective must match before plotting together. |
| Latency | Cold build/install/preparation separately; warm end-to-end wall time and declared target `<60s`; actual per-arm planned/available/executed time | Original monotonic/absolute windows and launch receipts. A completed >60s demo can be execution-complete with target missed. |
| Resource and cost efficiency | Total known creation CPU/GPU/wall and monetary/token costs divided by authenticated distinct complete fitted artifacts; also cost per distinct reusable recipe | Distinct means declared hash identity, not proven functional uniqueness. Whole-unit counters overlap worker sums; do not add them. Unknown overhead, price or tokens remain null with reason. |
| Fitted portability | Qualified exact selected fitted predictor/backend pairs / all requested selected predictor/backend pairs | Actual emitted code execution and prediction parity, in order Python/Rust then C++17/C++20 then JS. Unsupported pairs stay in the inventory and never replace the winner. Current five-language toy parity is recipe refitting only. |
| Reproducibility | Successfully authenticated independent report/reload replays / all requested replays | Same source, model, input, plan, protocol and artifact identities. Stale outputs, missing rows or changed class order are failures. |
| Promotion | Accepted serving-pointer advance with independent matched evaluation, exact checkpoint identity and controlled pointer update | Separate from training-pointer advance and experimental export. No new promotion is claimed. |

### Win counting and a proposed acceptance record

Define a dataset-level comparison using the protocol's frozen fold summary; retain fold-level results separately. Keep separate win columns for generic search, the previous checkpoint, and the best eligible frozen non-ASM comparator; the last column is PANEL-WINS as proposed in [§](#kpi-definitions). A win against one reference is not automatically a win against the others. For the owner's goal of ten wins per task on the 30-dataset panel, each reference's denominator remains ten per task and all thirty outcomes are listed, including unavailable ones; meeting it in every task is a 30/30 sweep. It is a declared goal, not a promise. As of 2026-09-23 there is no completed full-panel result establishing those counts.

A minimal additional acceptance record could use the following explicit nulls. This is a **proposed reporting shape**, not an existing on-disk schema:

```json
{
  "definition_status": "proposed_operator_reporting",
  "source_commit": null,
  "checkpoint_state_sha256": null,
  "reference_checkpoint_state_sha256": null,
  "corpus_sha256": null,
  "encoder_identity_sha256": null,
  "vocabulary_sha256": null,
  "panel_manifest_sha256": null,
  "protocol": null,
  "original_deadline_unix_ms": null,
  "planned_comparisons": 30,
  "valid_matched_pairs": null,
  "strict_wins": null,
  "eligible_teacher_families": null,
  "optimizer_updates_this_run": null,
  "exact_fitted_backend_pairs_qualified": null,
  "creation_cost": {"usd": null, "tokens": null, "unknown_reason": "not measured"},
  "serving_promoted": false
}
```

Retain the receipt path and content hash for each populated quantity, plus its unit, scope, numerator, denominator, exclusions and measurement time. Compare the supplied one-hour/ten-million-token scenario with actual token and cost records if those become available; a token budget is not evidence of tokens consumed, nor does it determine GPU time or dollars. Historical totals, current implementation counts and newly executed acceptance must remain separate series.

## What is still needed to demonstrate the learning flywheel

The desired loop is concrete: a dataset and split produce verified encodings; conditioned policy proposals become real generation-zero fits; bounded GP/evolution produces eligible improved programs; the full comparison identity admits those programs as teaching targets; optimizer updates produce a new checkpoint; matched downstream fits show whether that new checkpoint proposes and searches better algorithms. Only then can a promotion change the serving pointer. [§](teachers.md#a-dataset-is-not-one-training-example) and [§](data-estate.md#binary-format-the-exact-packed-teaching-record) trace the historical input-to-target representation, and [§](targets.md#did-gp-targets-get-better) separately reports observed adaptive target improvements in repeated searches with matching recorded comparison fields, including improvements in generic and shuffled arms. Those observations are largely carried-forward development scores from seed pools the arms share; they do not establish neural causality. [§](#what-better-and-better-must-mean) defines the remaining current acceptance work.

For every cycle, preserve both sides of the feedback: target improvement under the same task/dataset/split/evaluator/vocabulary/protocol identity, and policy improvement in fitted performance, failure rate and cost under a frozen comparison. Report cold/warm preparation and unique fitted predictors separately from recipe counts. Keep a context-free control, generic proposal control and unchanged accepted policy. A lower teacher-forced NLL can accompany memorization, degraded ranking, or no better executable search, as the historical measurements demonstrate.

The missing historical evidence is specific: full intermediate trajectories for the older 108-block capture (distinct from the complete later 63,000-step series); an exact run-source binding and event timestamp for gen31; per-term gradient diagnostics; a common independent evaluation series across changed corpora; and a fully joined sequence of qualified target improvements and serving promotions. For the new bounded ASM campaign, corpus admission, actual optimizer update/reload/resume proof, trained-export consumption and matched downstream improvement remain acceptance work. Keep these gaps visible rather than replacing them with a smooth synthetic curve or treating old training as if it had never happened. This chapter contains no performance plot. Every LOOP-01 quantity, including per-round family-normalized gain, the fresh-versus-replay difference and cycle-over-cycle improvement, is not measured. A future figure of any of them is drawn only from receipts, and every unmeasured series says "not measured" rather than showing an illustrative curve.

To establish a closed improvement loop rather than these separate observations, join: original search/fit completion → exact program and comparison identity → admission event and immutable target record → corpus membership → actual sampler exposure and optimizer continuation → later policy proposal/fit under the same comparison protocol. The historical logs in [§](training-history.md#dated-lineage-and-evidence-coverage) show training activity; the target receipts here show some improving, largely carried-forward development scores. That complete causal/eligibility join is still missing.
