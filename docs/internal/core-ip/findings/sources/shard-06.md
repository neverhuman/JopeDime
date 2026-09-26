<!-- from-tar tab=6 url=https://chatgpt.com/c/6ab702d5-ee40-83e9-a510-cc719c750d06 sha256=72069aa01cff66c9fca9be7f2e1a0a4b10f983a7395ff15fff27ce9ef32207ca chars=41068 -->

# Fitness, objectives, and learning-loop closure

Review notes: **shard 6 of 10; batch tab 6**. Scope: §6, with adjacent sections used only to establish training-loop boundaries.
Evidence basis: the supplied structured extract. Author-stated publication source: `3d71fce041d1dd1173ed98160f7d986bb426ba6f`; evidence cutoff **2026-09-25T19:31:50Z**. These metadata were not independently verified.
No product source, full chapter bodies, authenticated receipts, or executable training artifacts were supplied or inspected. No product tests, fits, optimizer runs, or repository writes were performed.

**Evidence vocabulary.** **DOCUMENTED** means explicitly stated in the extract, not independently verified in code. **UNVERIFIED** means the extract cannot establish the claim. **PROPOSED** means a new control or experiment, not an existing implementation. Severity ranks consequence if the risk materializes, not confidence that exploitation occurred; missing text alone is not a demonstrated defect.

**Boundary rule.** Training-side labels and scores are legitimate inputs where the declared protocol authorizes them. Proposed perturbation tests target forbidden access, not all label exposure; authorized scoring and later selection may legitimately change when their permitted labels change (§§2.1, 6.2, 6.4, 7.1.1).

**Decision.** Close production teacher provenance and the auditable learning transaction before expanding the self-improvement claim. Preserve the distinction between teacher eligibility, HWM advancement, corpus exposure, an actual policy update, and improved matched search. The extract explicitly makes these separate obligations (§6 opening; §§6.1.1.1, 7.1.2, 9.1).

## Executive hits

- **Gap — CRITICAL:** the extract says live `eligibility.rs` does not call the encoder-receipt validators. A valid-looking score packet therefore does not establish the required context provenance; actual contaminated admission is not demonstrated (§§6.2, 6.4, 11.1).
- **Improvement — P0:** make authenticated score evidence and consumer-specific context provenance mandatory inputs to production teacher admission; quarantine missing evidence instead of silently grandfathering it (§§6.2, 6.4; K01–K02).
- **Gap — HIGH:** HWM identity is documented, but atomic advancement, replay handling, and its binding to the frozen corpus are not visible. Concurrent or mismatched events could turn “new record” into a bookkeeping artifact (§§6.2, 6.5, 6.7, 7.1.2).
- **Improvement — P0:** join admission, comparable HWM events, immutable corpus manifests, loader exposure, and optimizer receipts into one auditable chain with explicit recovery semantics (§§6.2–6.8; K03–K07).
- **Gap — HIGH:** fixed objective coefficients do not establish reduction semantics, relative gradient influence, preference orientation, or runtime-label validity. Lower aggregate loss is not proof of better fitted predictions or search (§6.1; §§6.5, 7.1).
- **Improvement — P0/P1:** freeze an objective contract, validate loss masks and grouping, and measure each auxiliary term's contribution against matched search outcomes rather than optimizing the training-loss headline (§§6.1, 6.5; K05, K09–K11).
- **Gap — HIGH:** the reported 97 synthetic CPU tests and changed-parameter checks do not establish a real-corpus/GPU learning loop; transferable lift is explicitly not yet measured (§6.1.1.1; §7.5 heading).
- **Improvement — P0/P1:** demonstrate the smallest real admission-to-update-to-search cycle, then separately test reusable gains; add explicit fold-tail policy rather than calling the four-of-five guard a worst-fold guarantee (§§6.4–6.6, 11.1; K07–K08).

## Ranked IP gaps (CRITICAL/HIGH/MED/LOW)

### G01 — CRITICAL — Context provenance is not a mandatory live admission dependency
- **§refs / evidence:** §§6.2 “Teacher admission and corpus semantics,” 6.4 “How a fitted recipe becomes an eligible teacher”; §11.1 explicitly documents the missing live call to `encoder_receipt.rs`. This integration gap is DOCUMENTED; an actual leak or successful exploit is UNVERIFIED.
- **Attack / consequence:** forbidden labels can influence a target-conditioned context, fitted transform, or selected ancestor before the final fitter runs. An authenticated OOF score does not authenticate that upstream access history. A contaminated teacher can then propagate through corpus reuse and later updates (§2.1).
- **Detection:** exercise the real admission entrypoint with absent, wrong-scope, and mismatched context receipts. Separately perturb outer labels forbidden to the entire learner; proposal, admission, corpus, and update artifacts must remain unchanged under a reproducible setup. For a fixed candidate and authorized search history, perturb held-fold labels forbidden to its fit/context producers but allowed to its scorer; test those restricted producers, not downstream scores or authorized selection.
- **Harden:** require validated, consumer-bound provenance before eligibility returns success. Resolve transitive ancestors and cached contexts as well as direct inputs. Missing historical proof means quarantine or a clearly restricted evidence class, not an assertion that historical data were contaminated.

### G02 — HIGH — “Authenticated OOF” is a contract, not a demonstrated evidence verifier
- **§refs / evidence:** §§6.2, 6.4 and 7.1.2 require complete authenticated OOF evidence and declared fold wins. The authentication mechanism, field completeness, duplicate handling, and rejection behavior are UNVERIFIED because the relevant §6 bodies are trimmed.
- **Attack / consequence:** replayed folds, swapped fitted subjects, a score from another recipe, wrong class order, or nonfinite values could survive superficial completeness checks. Authentic bytes can still encode the wrong comparison or an incorrectly fitted baseline. None of these failures is established by the extract.
- **Detection:** mutate one bound field at a time; duplicate a fold while preserving record count; remove predictions or a required fold; inject nonfinite scores; swap original/executable recipe identities. Recompute the declared metrics and fold-local regression training-mean baseline from trusted fitting/evaluation memberships.
- **Harden:** verify the expected fold inventory and trusted producer, not only file hashes. Bind each prediction/score to dataset, cut, evaluator, metric, recipe adaptation, fitted state, context provenance, and applicable class order. Task-specific baselines must be declared rather than borrowing regression semantics.

### G03 — HIGH — HWM state can diverge from eligibility and corpus truth
- **§refs / evidence:** §7.1.2 binds HWM to task, dataset hash, cut, evaluator, vocabulary, and protocol; §6 opening requires comparison under the same identity. Transactionality, ties, initialization, and crash recovery in §§6.2, 6.5, 6.7 are UNVERIFIED.
- **Attack / consequence:** two workers can read the same predecessor and each claim advancement; a weaker late write can replace a stronger state. A changed evaluator can be incorrectly counted as improvement over an old namespace, or a crash can publish an HWM event whose teacher never reaches a usable corpus.
- **Detection:** race comparable candidates in opposite arrival orders, replay an admitted event, change each comparison field, and interrupt between validation, HWM persistence, and corpus publication. Verify that ties, initialization, and noncomparable candidates cannot masquerade as fresh improvements.
- **Harden:** use an idempotent, atomic HWM transition with a bound predecessor and explicit outcome. Preserve eligibility separately from improvement. Publish a corpus only when its admitted records and provenance are durable; asynchronous publication is acceptable only with a recoverable, visible intermediate state.

### G04 — HIGH — Fold wins and repeated OOF selection do not establish robust teachers
- **§refs / evidence:** §11.1 explicitly says the four-of-five admission guard is not a worst-fold guarantee. §§6.4 and 7.1.1 connect teaching to selection and warn about adaptive reuse. Additional aggregate guards, margins, and stability requirements are UNVERIFIED.
- **Attack / consequence:** a teacher may improve the required folds yet harm the remaining fold. Separately, repeated selection against the same OOF surface can favor search-specific winners without providing independent evidence of reusable improvement. These are limitations of the claimed inference, not proof that the declared guard is implemented incorrectly.
- **Detection:** construct a four-win/one-loss case satisfying every actual aggregate guard and inspect its disposition. Compare repeated adaptive selection with a preregistered independent audit, keeping that audit outside teacher creation, pair construction, HWM, and selection of the frozen contenders.
- **Harden:** make tail tolerance, ties, and minimum improvement policy explicit; retain fold vectors rather than a win count alone. Separate “eligible under the declared training policy” from “qualified for a robustness or transfer claim”; stronger qualification need not redefine all historical teachers.

### G05 — HIGH — A frozen manifest need not imply frozen optimizer inputs
- **§refs / evidence:** §§6.2, 6.3 “A dataset is not one training example,” and 6.7 “Corpus updates: reuse, device caches and loading”; §9.1 distinguishes availability, sampling, and optimizer exposure. Cache keys, publication atomicity, and loaded-byte receipts are UNVERIFIED.
- **Attack / consequence:** a manifest can stay unchanged while a loader resolves mutable paths, stale device tensors, a replaced context, or a different tokenizer. A long-running worker may consume a mixture of corpus versions. Unique-record counts would then describe neither the effective objective nor the actual training input.
- **Detection:** prepare a frozen corpus, then mutate an underlying record or encoder asset; reuse a cache across different permitted fit scopes; resume after a corpus replacement. Cold and cached loading must resolve identical authorized records, or explicitly fail, rather than silently combine versions.
- **Harden:** address records and derived caches by content and semantic dependencies, verify resolved bytes, and pin a complete immutable snapshot per run. Log sampled record identities and effective weights separately from manifest counts; a corpus switch must be an explicit training event.

### G06 — HIGH — Effective coefficients leave the implemented optimization target underdetermined
- **§refs / evidence:** §6.1 documents `L = Lseq + 0.25 LBT + 0.10 Llist + 0.10 Lruntime + 0.05 Lablation`, legal-token cross-entropy, recorded smoothing, and fixed effective coefficients. Exact reductions, masks, missing-term behavior, and accepted-checkpoint training-history bindings are UNVERIFIED.
- **Attack / consequence:** different sequence lengths, pair counts, or list sizes can alter influence despite unchanged coefficients. Illegal/PAD target mass or mismatched teacher weights can corrupt gradients. Seven stored log-variance tensors do not prove learned loss weighting, and this source's objective does not by itself establish an accepted checkpoint's historical objective.
- **Detection:** test legal support, padding, empty groups, singleton lists, missing auxiliary labels, and controlled duplication under the declared reductions. Record raw and weighted terms plus gradients reaching active policy parameters; trace the exact objective/configuration through checkpoint-specific training receipts.
- **Harden:** version the full objective contract, including normalization, masks, smoothing, teacher weights, and active parameters. Reject nonfinite or malformed supervision explicitly. Report coefficients alongside realized gradient influence without representing a monitoring statistic as proof of beneficial learning.

### G07 — HIGH — Preference and listwise labels can optimize the wrong ordering
- **§refs / evidence:** §6.1 says program weights and pair construction depend on comparison identity, while reward scores, objective percentiles, and fitted RMSE are distinct. Detailed pair/list construction in §§6.1–6.2 is UNVERIFIED; the extract does not justify calling this objective PPO/GRPO or assuming a DPO implementation.
- **Attack / consequence:** combining percentiles from different populations, reversing a lower-is-better metric, or treating a failed evaluation as a poor but valid score can manufacture training preferences. Repeated programs can dominate lists; a token-length convention can reward syntax rather than fitted quality.
- **Detection:** reverse a comparable pair and inspect the intended gradient direction; permute a list; inject a cross-identity pair, tie, duplicate, and invalid evaluation. Verify percentile reference populations, program sequence-scoring conventions, and that training pairs point to the actual authenticated fitted subjects.
- **Harden:** form preferences only within declared comparable groups, with explicit direction, tie policy, sequence normalization, and evidence status. Invalid evaluations require a separately typed supervision path, not a fabricated objective rank. Keep learned reward outputs distinct from measured admission evidence.

### G08 — HIGH — Runtime supervision can reward missing, censored, or incomparable execution
- **§refs / evidence:** §6.1 documents runtime Huber loss in a `log1p` representation and explicitly forbids reporting that loss as seconds. Runtime-label boundaries, units, hardware conditioning, failure treatment, and target availability are UNVERIFIED; §10.1.1 defines a different, explicit measurement boundary.
- **Attack / consequence:** zero-filled missing times can make failed candidates look cheap; treating a timeout as an exact duration can prefer candidates whose actual cost is unknown. Mixing fit-only, compile-inclusive, cold, and warm measurements can produce a low loss for a target with no stable operational meaning.
- **Detection:** trace representative labels back to timing receipts; inject missing, timed-out, unsupported, and completed outcomes; vary timing scope or hardware while retaining the same recipe. Check both the regression target and any ranking consumer for silent status-to-number coercion.
- **Harden:** type the runtime subject, scope, unit, environment, and censoring status. Exclude or explicitly model censored observations under a declared policy; gate feasibility separately. Report log-domain training loss separately from calibrated duration predictions and measured search/runtime tradeoffs.

### G09 — HIGH — Continuation names can hide a different optimization process
- **§refs / evidence:** §6.1.1 states optimizer state is part of resume; §6.8 promises four continuation names and commands, but their bodies are absent. Their actual names, guarantees, and handling of corpus transitions are UNVERIFIED and are not reconstructed here.
- **Attack / consequence:** reloading weights while resetting optimizer moments, scheduler, sampler, or random streams can be advertised as uninterrupted training. Mid-update restarts can duplicate exposure or drop an update. Reset step labels can conceal a fork or import a rejected state into accepted-lineage accounting (§§4.2, 9.1).
- **Detection:** compare uninterrupted and interrupted runs on a declared reproducible setup, including sampled identities, optimizer state, active parameter updates, and counters. Remove state components individually; change vocabulary, corpus, or topology; attempt to present a weights-only load as an exact continuation.
- **Harden:** bind checkpoints to objective, corpus, optimizer, scheduling, sampler, RNG, precision, and accumulation state where applicable. Publish the real four-mode behavior matrix; missing state must reject the exact-resume claim or require an explicit fork/reset mode, never an automatic downgrade.

### G10 — HIGH — The complete real learning loop is not established by local update tests
- **§refs / evidence:** §6.1.1.1 reports 97 passing synthetic CPU tests and changed-parameter/update/reload checks, explicitly not real-corpus/GPU qualification. §§6.5–6.6 are trimmed; §7.5 is explicitly a reporting contract, not measured transferable lift.
- **Attack / consequence:** loss reduction, a newer checkpoint file, or changes confined to inactive parameters can be narrated as acquired reusable knowledge. Even a valid active-policy update can worsen legal proposal quality, waste the search budget, or fail to improve fitted outcomes.
- **Detection:** follow a real admitted teacher through a frozen corpus and logged minibatch into objective-driven active-policy changes, reload, and matched parent/challenger search. Require authentic identities throughout; independently evaluate the gain claim rather than treating a successful update as its own evaluation.
- **Harden:** retain two outcomes: operational loop closure and measured learning benefit. The smallest real path must establish the former; a preregistered independent comparison must establish the latter. A null or harmful result is evidence, not a reason to relabel a development score as transfer.

### G11 — MED — Program-only ablation may not establish useful conditioning
- **§refs / evidence:** §6.1 includes `0.05 Lablation` and describes a program-only ablation. Its exact target, sign, gradient routing, and interaction with context-conditioned sequence training are UNVERIFIED; §6.5's reusable-knowledge argument therefore remains incomplete.
- **Attack / consequence:** the model may lower loss by learning frequent programs or dataset fingerprints while making little useful use of the permitted context. Depending on its actual definition, the ablation term could encourage or diagnose that behavior; the extract does not resolve which.
- **Detection:** inspect gradient routing and perform compute/exposure-matched training comparisons using the actual ablation definition. Measure conditional benefit on permitted, isolated evaluation groups. Context shuffling or zeroing is a sensitivity probe, not by itself causal evidence because it can create distribution shift.
- **Harden:** specify the intended role of the term and the downstream outcome that justifies it. Evaluate context utility and program memorization separately; do not infer useful conditioning merely from the context tensor being consumed or from the total loss decreasing.

### G12 — MED — Curriculum selection can concentrate exposure or erase useful contrasts
- **§refs / evidence:** §§6.2–6.3 distinguish corpus semantics from dataset count; §§6.4–6.5 connect admission/HWM to learning. Deduplication, sampling weights, repeated ancestry, and the handling of eligible non-HWM candidates are UNVERIFIED.
- **Attack / consequence:** repeated relatives of one winning program can dominate optimizer exposure without adding comparable diversity. Conversely, HWM-only ingestion, if used, can discard valid non-record teachers and informative inferior alternatives. The extract does not establish that either sampling policy is actually present.
- **Detection:** reconcile available records, unique program identities, comparable groups, family/ancestor concentrations, sampled positions, and effective loss mass. Follow eligible-but-non-improving and valid-but-inferior candidates through the loader; distinguish intentional repetition from accidental duplication.
- **Harden:** preserve separate statuses for eligibility, HWM outcome, exposure, and contrastive suitability. Use an explicit, audited sampling policy; compare a diversity-aware curriculum with the existing policy at matched exposure and compute before asserting a learning advantage.

## Ranked improvements / kickoffs (P0/P1/P2)

### K01 — P0 — Enforce proof-carrying teacher admission
- **§refs / addresses:** §§6.2, 6.4; boundary obligations in §§2.1, 11.1. Addresses G01. Proposed owner: learning/admission maintainer, with encoder provenance owner.
- **Why / implementation:** a mandatory trust boundary prevents bad evidence from becoming persistent training knowledge. Wire scope and identity validation into the actual eligibility path; require transitive context/ancestor closure and expose a typed rejection or quarantine reason rather than an optional validator call.
- **Falsifiable acceptance:** missing, mismatched, revoked, or unauthorized context evidence cannot produce an eligible teacher through any production ingestion route. With allowed inputs and RNG fixed, forbidden outer-label perturbations leave the learning chain unchanged. Score-only fold-label tests require invariance only from explicitly restricted fit/context producers, not from later authorized scoring or selection.
- **Retained proof:** production-route negative tests, permitted-access traces, and a valid real teacher receipt. Passing a validator in isolation is insufficient; do not claim historical contamination or historical cleanliness from the new gate alone.

### K02 — P0 — Make teaching evidence a verifiable typed record
- **§refs / addresses:** §§6.2, 6.4 and 7.1.2. Addresses G02 and the supervision-subject portion of G07. Proposed owner: learning evidence/evaluator maintainer.
- **Why / implementation:** make the recipe being taught inseparable from the fitted subject that earned its label. Define required fold inventory, prediction/metric evidence, baseline fit scope, original-to-executable adaptation, trusted producer identity, and context bindings; use a versioned schema with explicit unknown/unsupported outcomes.
- **Falsifiable acceptance:** every required-field deletion, fold replay, identity swap, invalid metric, and wrong-subject fixture is rejected. Trusted metric recomputation matches the declared protocol; historical records lacking proof remain distinguishable from qualified records and never acquire invented default values.
- **Retained proof:** schema, producer/verifier trust assumptions, mutation-test results, and an acceptance/rejection decision trace. Cryptographic authentication establishes origin/integrity under those assumptions; it does not certify methodological correctness without semantic checks.

### K03 — P0 — Transactional HWM events with durable corpus linkage
- **§refs / addresses:** §§6.2, 6.5, 6.7; §7.1.2. Addresses G03. Proposed owner: learning-state/storage maintainer.
- **Why / implementation:** make “improvement” an auditable event rather than an overwritten scalar. Persist comparison identity, predecessor, candidate evidence, comparator result, and disposition atomically; use idempotent replay and link committed eligible evidence to a recoverable corpus-publication state.
- **Falsifiable acceptance:** racing candidates cannot regress the HWM; replay cannot create another improvement; a new comparison identity initializes independently. Crash/restart testing leaves neither a falsely finalized corpus nor an unrecoverable claimed teacher. Eligible non-improvements retain their distinct status.
- **Retained proof:** concurrent event histories and recovery transcripts that reconstruct the final HWM and each corpus transition. The evaluator's metric direction, tie policy, and initialization behavior must be pinned rather than inferred by a reporting consumer.

### K04 — P0 — Freeze resolved inputs, not just manifest filenames
- **§refs / addresses:** §§6.2, 6.3, 6.7; §9.1. Addresses G05 and supports G12. Proposed owner: corpus/loader maintainer.
- **Why / implementation:** content-address teaching records and all derived context dependencies; publish immutable snapshots; bind device-cache entries to record, encoder, fit-scope, vocabulary, and objective-relevant preprocessing identities. Emit the actual sampled record sequence and effective supervision weights.
- **Falsifiable acceptance:** changing a backing file cannot silently alter a pinned run; cross-scope cache reuse is rejected; cold and cached loading produce the same authorized training tensors. Corpus replacement is either a new explicit snapshot transition or an error, never a mixed run.
- **Retained proof:** resolved-input manifests, cache-validation tests, sampling/exposure logs, and a restart trace. Establish separate counts for availability and exposure without inventing missing historical totals or treating repeated records as newly discovered programs.

### K05 — P0 — Publish and test the complete objective contract
- **§refs / addresses:** §6.1, including §6.1.1.1; checkpoint-history boundary in §§4.5–4.6, 9.1. Addresses G06. Proposed owner: native trainer maintainer.
- **Why / implementation:** preserve the documented coefficients while specifying reductions, legal-token smoothing support, masks, term availability, program weights, active parameters, and every term's label source. Log term values and active-policy gradient contributions; bind the contract to each new training checkpoint.
- **Falsifiable acceptance:** PAD-only extension cannot change non-PAD supervision; illegal targets fail explicitly; empty/missing supervision follows the declared policy; group duplication changes influence exactly as specified. Stored log-variance tensors are never reported as active adaptive weighting without an executed dependency.
- **Retained proof:** synthetic loss/gradient fixtures and real-batch traces, plus exact objective/configuration receipts. Reproducing the scalar loss is insufficient when masks or active-policy gradients differ; historical checkpoint claims require their own evidence rather than backfilling today's configuration.

### K06 — P0 — Define and battletest the actual four continuation modes
- **§refs / addresses:** §§6.1.1, 6.8; §§4.2, 9.1. Addresses G09. Proposed owner: training CLI/checkpoint maintainer.
- **Why / implementation:** recover the full §6.8 and publish its real names/commands with state-preservation guarantees. Specify optimizer, scheduler, corpus, sampler, RNG, accumulation, step/exposure counters, model topology, and objective handling for each mode; distinguish exact continuation from explicitly authorized resets or forks.
- **Falsifiable acceptance:** interruption/restart reproduces the next consumed records and updates in a declared deterministic fixture. Unsupported exact replay is reported honestly for other backends. Missing required state or incompatible identities cannot silently select a different mode; reset/fork receipts remain distinguishable after subsequent saves.
- **Retained proof:** mode matrix, missing-state tests, mid-update recovery tests, and lineage/exposure comparisons. Reject a rejected checkpoint's silent substitution for accepted lineage, while permitting a deliberately named experimental fork that makes no accepted-continuation claim.

### K07 — P0 — Execute the smallest real evidence-to-search learning cycle
- **§refs / addresses:** §§6.4–6.6 and 6.1.1.1; §§7.1.1, 7.5. Addresses G10. Proposed owner: learning-loop integration maintainer, with independent evaluation owner.
- **Why / implementation:** use a real authenticated eligible teacher, an explicit comparable HWM disposition, a frozen corpus, logged optimizer consumption, objective-driven active-policy changes, and reload. Then run matched parent/challenger search under the declared budgets; keep independent evaluation labels inaccessible to the learner and its selection decisions.
- **Falsifiable acceptance:** operational closure passes only when every transition joins to exact evidence and the reloaded updated policy is actually used. A learning/transfer claim passes only when the separately preregistered audit meets its declared criteria. Null or harmful outcomes do not invalidate plumbing proof but block a positive-gain claim.
- **Retained proof:** a joined real-run receipt bundle and an outcome report including failed attempts. A fresh file timestamp, changed inactive head, lower imitation loss, or GPU inference parity cannot replace the specific update/search evidence.

### K08 — P1 — Separate training eligibility from tail-safe promotion
- **§refs / addresses:** §6.4; §§7.1.1–7.1.2, 11.1. Addresses G04. Proposed owner: learning policy/evaluation maintainer.
- **Why / implementation:** retain the existing declared admission policy as a versioned baseline, then define whether worst-fold degradation, minimum gains, uncertainty, and repeated-look history constrain promotion or teaching weights. Thresholds and treatment of ties must be chosen prospectively by the owner, not invented in this review.
- **Falsifiable acceptance:** a four-win/one-loss fixture satisfying current aggregate guards receives the exact declared current-policy result, never an unsupported worst-fold-safe label. A stricter prospective policy applies its explicit tail rule; independent audit data cannot feed admission, weighting, or threshold tuning.
- **Retained proof:** policy comparison showing both accepted-teacher yield and fold-tail outcomes, including rejected candidates and uncertainty. A safer tail policy is not automatically a better learning curriculum; compare the tradeoff at matched search/training resources.

### K09 — P1 — Build comparable preference groups and a contrast audit
- **§refs / addresses:** §§6.1–6.2 and 6.4. Addresses G07. Proposed owner: preference/listwise data and trainer maintainer.
- **Why / implementation:** derive pair/list labels from authenticated comparable evaluations; bind metric direction and percentile population; specify ties, duplicates, invalid outcomes, and sequence normalization. Preserve valid inferior programs as contrasts when authorized, without promoting them to eligible positive teachers.
- **Falsifiable acceptance:** cross-identity pairs and invented numeric failure scores cannot reach ranking losses. Pair reversal and list permutation behave according to the declared objective, and duplicate handling matches its weighting policy. Measured reward-ranking agreement alone does not authorize HWM movement.
- **Retained proof:** hand-auditable comparison groups, gradient-direction tests, and an equal-budget downstream search comparison against the unmodified objective. Claim improvement only for a demonstrated endpoint; additional pair count is not itself increased information.

### K10 — P1 — Turn runtime labels into an operationally meaningful target
- **§refs / addresses:** §6.1; measurement-boundary cross-link §10.1.1. Addresses G08. Proposed owner: runtime-label producer and trainer maintainer.
- **Why / implementation:** define the runtime subject and boundary, retain device/backend/warm-state provenance, and represent censoring or failure explicitly. Keep the documented log-domain Huber loss separate from observed durations; audit any use of the learned runtime head in candidate ranking.
- **Falsifiable acceptance:** unsupported, missing, and timed-out outcomes are never silently labeled as fast completed runs. Heterogeneous timing scopes are partitioned or modeled under an explicit contract; duration calibration and ranking utility are evaluated on authorized held-out comparisons, not inferred from lower training loss.
- **Retained proof:** raw timing receipts, target-construction traces, status-injection tests, and a matched search comparison reporting fitted-quality/cost tradeoffs. Do not assume the headline end-to-end benchmark is the native training label without matching evidence.

### K11 — P1 — Test whether the ablation term supports conditional learning
- **§refs / addresses:** §§6.1, 6.5. Addresses G11. Proposed owner: training-objective researcher.
- **Why / implementation:** recover the actual ablation computation, target, and gradient routing before proposing coefficient changes. Run matched-exposure and compute-controlled variants to separate useful conditioning, generic program priors, and memorization; preserve the original objective as the comparison baseline.
- **Falsifiable acceptance:** useful-conditioning claims require improvement on the prespecified isolated task groups and downstream endpoint, not merely sensitivity to context corruption. Any shuffled-context test respects permitted grouping and is labeled diagnostic; a null result narrows the claim instead of being hidden by aggregate loss.
- **Retained proof:** variant configurations, exposure receipts, component-loss and search outcomes, and uncertainty. A program-only prior may still be operationally valuable; failure to show conditional benefit does not prove the entire search system ineffective.

### K12 — P2 — Develop an evidence-aware, diversity-preserving curriculum
- **§refs / addresses:** §§6.2–6.5 and 6.7; exposure cross-link §9.1. Addresses G12. Proposed owner: curriculum/learning research maintainer.
- **Why / implementation:** retain distinct roles for eligible positives, comparable non-HWM results, valid preference contrasts, and invalid/censored outcomes. Audit ancestor and dataset-family concentration, then test diversity-aware sampling without altering the evidence requirements or laundering failed evaluations into fitness labels.
- **Falsifiable acceptance:** reported unique programs and actual sampled exposure reconcile; replays do not create fictitious diversity. Compare the proposed curriculum with the existing sampler at matched compute/exposure and isolated evaluation access; adopt only for a demonstrated benefit without unacceptable tail or invalid-proposal regression.
- **Retained proof:** corpus-role schema, concentration/exposure reports, sampler receipts, and a controlled outcome comparison. This is a research kickoff, not an established breakthrough or a reason to weaken admission before K01–K07 close the evidence chain.

## Evidence gaps / UNVERIFIED load-bearing claims

The following are closure requests, not claims that the missing implementations are absent. The supplied section titles alone cannot establish the trimmed bodies.

| Claim needing closure | Current extract boundary | Exact evidence / section needed |
|---|---|---|
| All production teachers satisfy context provenance | §11.1 explicitly says the live validator call is absent | Full §§6.2, 6.4; live admission dispatch and rejection traces; reassessed §11.1 |
| OOF evidence is complete and authentically bound | Requirement stated in §7.1.2; verifier details absent | Full §§6.2, 6.4; trusted-producer model, fold/prediction bindings, negative tests |
| HWM advancement is comparable and recoverable | Identity fields stated; transition semantics absent | Full §§6.2, 6.5, 6.7; `hwm.rs` behavior and concurrent/crash receipts |
| Loss terms implement one stable objective | Coefficients and selected semantics visible, reductions omitted | Full §6.1; masking, smoothing, weighting, term formulas, gradient and optimizer traces |
| Accepted weights were trained under that exact objective | Present source and historical checkpoint are distinct evidence subjects | §§6.1, 6.8 plus §§4.5–4.6, 9.1; checkpoint-specific objective/corpus/optimizer receipts |
| Preferences and runtime labels have valid semantics | Comparison dependence and log-domain representation stated | Full §§6.1–6.2; pair/list label construction, runtime scope and censoring records |
| Corpus freeze matches actual optimizer exposure | Conceptual distinction stated; loader/cache behavior trimmed | Full §§6.3, 6.7; resolved records, cache dependencies, sampling and update logs |
| Historical teacher readiness was sufficient for a real cycle | §6.6 supplies a dated heading, not the readiness evidence | Full §6.6 “Historical teacher readiness (2026-09-23 snapshot) and the smallest real path”; actual admitted-record/run receipts |
| Four continuation modes preserve their declared state | §6.8 heading only; §6.1.1 body trimmed | Full §§6.1.1, 6.8; actual names/commands, serialization and restart traces |
| Local tests qualify real-corpus or GPU training | §6.1.1.1 explicitly limits its synthetic CPU result | Full §§6.5–6.6; real-corpus/GPU update-and-reload evidence at a bound source |
| Training improves reusable search behavior | §6 opening rejects loss-only inference; §7.5 says lift is not yet measured | Full §6.5 plus §7.5; independent matched parent/challenger audit with learning-access closure |

## Fail-open / silent-lie paths (if any in shard)

| Path | Status in this review | Required closed behavior |
|---|---|---|
| Missing context receipt → eligible teacher | Documented live integration gap (§11.1); contaminated admission UNVERIFIED | No production eligibility without validated provenance; quarantine explicitly |
| Authenticated bytes → “methodologically valid OOF” | Conditional risk; verifier scope UNVERIFIED (§§6.2, 6.4) | Validate fold, fit-scope, baseline, prediction, and subject semantics |
| Eligible teacher → automatic HWM improvement | Contrary to the stated distinction; occurrence UNVERIFIED (§7.1.2) | Record comparable improvement separately from eligibility and initialization |
| Frozen manifest → stale or mixed cached training tensors | Conditional risk (§6.7 trimmed) | Bind and verify resolved input/cached tensor dependencies |
| Invalid runtime outcome → cheap successful program | Conditional coercion risk (§6.1) | Explicit status/censoring policy; no fabricated duration target |
| Missing optimizer state → successful “exact resume” | Conditional risk; stated resume obligation (§§6.1.1, 6.8) | Reject the claim or require a visibly different authorized mode |
| Lower loss/new checkpoint → “learned transferable knowledge” | Unsupported inference, not an observed false statement (§§6.5, 7.5) | Separate active update, operational closure, development gain, and independent transfer |
| Four fold wins → “robust on every fold” | Explicitly unsupported implication (§11.1) | Retain fold vector and report only the guarantee actually enforced |

## Cross-links to other sections the owner must not ignore

- **§§2.1, 2.6 → §§6.2, 6.4, 6.7:** the learning boundary covers labels, transforms, cached contexts, ancestors, and permitted family-level access, not merely final fitting rows. Define each consumer's permitted scope before designing perturbation tests.
- **§§3.3–3.6 → §§6.2, 6.4:** target-conditioned context and fitted encoders require consumer-specific provenance. Matching tensor width is not an admission proof; encoder architecture redesign is outside this review.
- **§§5.1, 5.9–5.10 → §6.4:** teacher targets must identify the original proposal, executable adaptation, and fitted/scored subject. A refit or rewritten recipe cannot silently inherit another subject's score.
- **§§7.1.1–7.1.2, 7.5–7.6 → §§6.4–6.5:** eligibility, HWM, development lift, and independent transfer are distinct consumers with different evidence. Reused audit outcomes enter adaptive learning/selection history through teachers, pairs, hyperparameters, or later checkpoint choice; the final comparison must remain outside creation and selection of its frozen contenders.
- **§§4.2, 4.5–4.6, 9.1 and Appendix A → §§6.1, 6.8:** exact parameter identity supports, but does not replace, objective/exposure/resume lineage. Stored inactive tensors are not evidence of active loss terms or useful policy updates.
- **§§10.1.1–10.2, 11.1, 12.1 → §§6.1, 6.5–6.6:** timing/inference qualification, source implementation, synthetic testing, real training, and promotion cannot substitute for one another. Update the evidence map only to the status actually reached.

## Concerns / residual risk

Evidence enforcement can establish what data and objective were used without proving useful generalization. HWM comparisons can be perfectly authentic yet overfit an adaptively reused development surface; independent audits also become adaptive once repeatedly used for design or checkpoint decisions (§§6.5, 7.1.1, 7.5).

Cryptographic receipts prove only the properties covered by their trust model. Compromised producers, incorrect evaluators, incomplete dependency inventories, and falsely declared family boundaries remain possible. Counterfactual invariance tests help localize leaks but do not prove universal noninterference (§§2.1, 6.2, 6.4).

Strict provenance and tail policies may shrink the usable historical corpus or discard exploratory teachers. Quarantine and separately typed research pools preserve information without upgrading its evidence status. Any recovery requires legitimate requalification, not synthesized historical receipts (§§6.2–6.7).

Exact GPU restart equality may depend on deterministic kernels and execution setup. Declare the supported reproducibility contract prospectively; do not demand unjustified bitwise equivalence everywhere, or weaken identity and exposure checks because numerical replay is imperfect (§§6.1.1, 6.8).

**Release boundary:** K01–K07 target the defensible operational foundation; K08–K12 test robustness and learning value. Neither package alone establishes patent novelty, freedom to operate, a measured breakthrough, or production promotion. Until the missing evidence is retained, report the specific implemented/tested/measured link—not “the loop learns” as an undifferentiated conclusion (§§1, 1.1, 6, 12.1).
