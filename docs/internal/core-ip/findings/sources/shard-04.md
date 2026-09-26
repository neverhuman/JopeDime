<!-- from-tar tab=4 url=https://chatgpt.com/c/6ab7027b-e6c4-83ea-b060-48faa33f0b84 sha256=a419fd1d3aa3258474f4d9b272bdf08e6a5d6155c2ddfa71d9279cb103bc3b14 chars=37932 -->

# Neural program generator — evidence-bounded IP review

Artifact notes: shard 4 of 10; batch tab 4. Scope: §4, with boundary-only cross-links.
Monograph evidence cutoff: **2026-09-25T19:31:50Z**. Publication source: `3d71fce041d1dd1173ed98160f7d986bb426ba6f`.
Accepted weights, as reported: SHA256 `4b4b5a40f03c9226b4b15856fd35b90127c8b6a4d1435b815484cdd050744f1e`.
**Evidence boundary:** this review inspected the supplied extract, not the product repository, checkpoint, complete chapters, or retained execution records. No product tests were run. Section references identify the supplied source or the omitted section needed for closure; external references support general test mechanisms only.
**Status vocabulary:** “documented” means asserted in the extract; **UNVERIFIED** means unavailable for independent inspection. Severity ranks the consequence of a conditional failure, not the probability that an unseen implementation contains it. No confirmed CRITICAL implementation defect is established.
**Documented baseline:** lineage B, step 60,000; accepted 512×6 topology; input context `B×2048`; token input `B×T`; four attention heads of width 128; a context-generated prefix; causal and PAD-key masks. The reported inventory is 136 tensors and 27,796,494 parameters, with F32 storage separately asserted. These are not independently recounted here (§§4–4.2; Appendix A).

## Executive hits

- **Preserve the accepted-checkpoint firewall, then close semantic identity.** Name/shape/dtype checks and separate accepted/challenger constructors are explicitly documented. Whether vocabulary meanings, grammar, forward configuration, and ranker routing are equally bound remains UNVERIFIED (§§4.2, 4.4–4.5).
- **Test causality at the actual next-token boundary.** Correct-looking attention equations do not establish correct target shifting, prefix indexing, padding behavior, or inference-mode execution. Metamorphic tests can resolve these without changing the accepted weights (§§4.1, 4.3–4.4).
- **Investigate the causal-prefix scorer trap.** Under the stated causal arrangement, the first prefix representation cannot see later program tokens. A scorer reading only that representation would be program-blind; the omitted readout architecture must establish that this is not the live ranking path (§§4.1, 4.3–4.4).
- **Make decoding terminate honestly.** Empty legal support, non-finite sampling weights, premature EOS, and budget exhaustion must produce typed outcomes rather than disguised successful proposals. A grammar/parser differential test is a concrete P0 kickoff (§4.4; §5.1).
- **Publish the complete forward contract, not just a tensor total.** The visible `ctx_proj.weight` is `[512,512]`; the unshown context MLP must explain the route from the 2,048-value input. This is an unresolved bridge, not evidence of a shape contradiction (§§4.1, 4.3; Appendix A).
- **Prove ranking value before trusting its gate.** Evaluate a diagnostic sample from before rank rejection, distinguish reward estimates from fitted objectives, and measure which genuinely good feasible candidates the gate loses. Stored heads and lower training losses do not establish useful ranking (§§4.3–4.4; §6.1).
- **Separate conditional intelligence from a generic recipe prior.** Matched context interventions, a properly defined program-only control, and generation-zero measurements can isolate the generator’s contribution before GP rescues or amplifies it (§§4.3–4.4; §§5.5, 7.5).
- **Turn proposal history into reproducible evidence.** Bind logical random streams, numerical qualification, deduplication, feasibility refusals, adaptations, and explicit fallback origin to each proposal. Then test diversity or richer conditioning as challengers, not silent changes to accepted lineage B (§§4.2–4.6; §10.1).

## Ranked IP gaps (CRITICAL/HIGH/MED/LOW)

### G01 — HIGH: a valid tensor inventory may still describe the wrong executable policy
**§ref / evidence:** §4.2 documents name, shape, dtype, and receipt-identity checks; §§4.4–4.5 are omitted. Strict loading is a documented strength; complete semantic-bundle enforcement is **UNVERIFIED**.
**Attack/gap story:** a same-sized token-ID permutation, changed special-token meaning, altered grammar, different positional convention, or ranker-route change can preserve tensor compatibility while changing proposals. A receipt identifier is insufficient when the consumer does not validate every interpretation-changing dependency.
**Detection:** run negative fixtures through every accepted-serving entrypoint: unchanged weights with permuted vocabulary meanings, altered grammar/configuration, missing receipt, rejected step-69,000 identity, and experimental 256×2 identity. Instrument whether inference begins before rejection; distinguish rejection from an explicitly authorized challenger load.
**Harden:** authenticate the opened weight bytes together with vocabulary, grammar, context-interface version, forward configuration, active-head routing, and serving-selection receipt. Record the selected bundle on proposals; prohibit a “latest checkpoint” fallback. Close with §4.2 loader tests and the §4.5 acceptance record.

### G02 — HIGH: causal and padding claims do not establish autoregressive correctness
**§ref / evidence:** §4.1 gives attention scaling and says future positions and PAD keys are hidden; target shifting, prefix offsets, and live mask construction are **UNVERIFIED** in §§4.3–4.4.
**Attack/gap story:** an off-by-one training target can expose the token being predicted despite a causal mask; prefix misalignment can hide context; padded queries can contaminate an unmasked reward pool. A backend port can also invert a boolean mask. PyTorch’s documented mask conventions illustrate that last hazard, without establishing the product uses those APIs.[^attention]
**Detection:** compare teacher-forced and incremental next-token outputs at identical logical histories; alter future suffixes; append supported padding; permute batch membership while preserving each example; test empty/short/maximal legal sequences. Inspect evaluation/dropout settings and any cache path actually implemented.
**Harden:** freeze logical position and target-shift conventions, use one explicit mask contract across consumers, exclude padding from readout, and qualify supported batching/padding modes. Reject unsupported layouts instead of broadcasting or truncating silently; close with §§4.1, 4.3–4.4 forward traces.

### G03 — HIGH: reward readout could be program-blind or routed through an inactive head
**§ref / evidence:** §4.1 places context first under causal attention; §4.2 acknowledges stored inactive heads. Reward architecture, pooling, and live head selection are **UNVERIFIED** in §§4.3–4.4 and the omitted Appendix A inventory.
**Attack/gap story:** under the stated causal layout, a readout using only the leading prefix cannot depend on following program tokens unless a separately documented path supplies them. Mean pooling that includes PAD, a last-position readout that lands on PAD, or accidental use of an untrained auxiliary head creates different but equally misleading candidate scores.
**Detection:** trace program-token dependencies into every serving score. With context fixed, apply controlled program edits and inspect intermediate/readout changes; combine this with graph inspection because equal scores on one pair do not prove blindness. Perturb each head in a disposable checkpoint copy and verify the declared consumers respond, while declared inactive heads have no serving effect.
**Harden:** publish a head-to-input-to-consumer map, explicit pooling/length rules, and per-head training/serving status. Qualify the existing route first; any replacement pooling or scorer is a challenger requiring training/evaluation, not a transparent repair of accepted weights (§§4.2–4.5).

### G04 — HIGH: legal-token filtering is not yet a closed decoding contract
**§ref / evidence:** §4.4 is omitted; §12.1 labels `grammar.rs`, `decode.rs`, and `seeds.rs` implemented. Empty-support, completion, and cutoff behavior remain **UNVERIFIED**.
**Attack/gap story:** when a truncation filter precedes grammar restriction, it can remove every legal option; a locally legal prefix may also have no completion within remaining length. All-masked or non-finite distributions, forced EOS, automatic parenthesis repair, or an unmasked fallback can turn failure into a falsely valid neural proposal. As a reference sampling contract, PyTorch requires finite, nonnegative weights with positive total mass.[^sampling]
**Detection:** exercise accepting/nonaccepting parser states, exhausted length, unknown token IDs, invalid temperature settings, extreme logits, and empty support after every enabled filter. Differentially compare accepted token sequences with the production typed parser; record whether retries and repairs consume the declared budget.
**Harden:** validate support after the actual filter composition; distinguish explicit greedy mode from invalid temperature; permit EOS only under the declared completion rules. Return typed failure on exhaustion, or an explicitly identified repair with new provenance. Never convert an incomplete sequence into a successful neural draw (§4.4; §5.1).

### G05 — HIGH: the architecture cannot be independently reconstructed from the extract
**§ref / evidence:** §4.1 supplies the attention/FFN equations and top-level widths; Appendix A shows representative projection and decoder tensors. The full §4.3 forward definition and complete inventory are **UNVERIFIED**.
**Attack/gap story:** the visible `[512,512]` context projection cannot alone account for the stated `B×2048` input, but an omitted upstream MLP may do so correctly. Separately, pre/post normalization, residual order, positional encoding, GELU variant, biases, dropout, embedding/output tying, and output readout can change behavior while preserving many or all tensor shapes.
**Detection:** map every active tensor to an ordered forward operation and every operation to its configuration. Trace the complete context-to-prefix route, decoder blocks, token logits, and ranking outputs. Reconcile stored-versus-active inventories; do not infer cross-attention, vocabulary size, sequence capacity, or head widths from names alone.
**Harden:** provide a versioned operation-level contract and golden intermediate/output fixtures from the accepted path. Keep the documented widths unchanged; fill omitted details from §4.3, `model.rs`, `config.rs`, and the complete Appendix A rather than constructing a plausible substitute architecture.

### G06 — HIGH: ranking may optimize a proxy while excluding the best usable programs
**§ref / evidence:** §4.4 ranking is omitted. §6.1 explicitly distinguishes reward scores, objective percentiles, fitted RMSE, and log-domain runtime loss. The serving combination, direction, normalization, and calibration are **UNVERIFIED**.
**Attack/gap story:** an undocumented combination of sequence likelihood, learned reward, length, or runtime can favor familiar, short, or cheap programs rather than the permitted fitted objective. Evaluating only the gate’s survivors conceals false rejections; apparent accuracy on those survivors is not evidence the gate improves selection.
**Detection:** freeze a pre-ranking candidate pool and evaluate a preregistered diagnostic sample spanning both accepted and rejected ranks using permitted fitting-side evaluation. Compare the live ranker with random ordering and likelihood-only ordering on the same pool; inspect length/operator/feasibility slices and non-finite/tied-score handling.
**Harden:** expose named score fields with units and directions, the exact selection rule, tie-breaking, and a gate version. Report false rejection and selection regret relative to the best actually evaluated feasible candidates—not an unobserved global optimum. Close with §4.4 traces and the relevant §6.1 target/readout definitions.

### G07 — HIGH: accepted architecture and training progress do not establish conditional proposal value
**§ref / evidence:** §§4.1–4.3 describe a conditional generator; §6 says lower imitation loss closes only one loop link; §7.5 labels transferable lift not yet measured. Generator-specific causal benefit is **UNVERIFIED**.
**Attack/gap story:** the network may learn a strong unconditional recipe prior while ignoring some or all context. Downstream GP, generic seeds, filtering, or fallback can produce good final predictors while masking weak neural proposals. Conversely, useful generation-zero proposals may be obscured by an unfavorable downstream budget.
**Detection:** hold accepted weights and decoding settings fixed for context permutations and individual-block interventions; separately evaluate a properly trained program-only control under matched resources and its own identity. Measure pre-ranking, pre-GP, and downstream outcomes. Where both decoder and ranker consume context, intervene on them independently so ranker-only gains cannot be credited to proposal generation.
**Harden:** separate the claims “context affects outputs,” “context improves feasible proposal quality,” and “improvement transfers.” Zeroed or shuffled contexts are diagnostic interventions, not automatically distribution-matched controls. Close with §§4.3–4.4 ablations and the independently enforced evaluation boundary in §7.5.

### G08 — MED: proposal scores may not remain attached to the same executable subject
**§ref / evidence:** §5.1 explicitly preserves original proposal bytes and inserted raw-input stage; §§4.4 and 5.4 details are omitted. The generator-to-parser/ranker/adapter handoff is **UNVERIFIED**.
**Attack/gap story:** a reward computed for original tokens can be displayed beside a repaired or adapted recipe without saying which subject it estimates. Grammar-valid proposals can also be infeasible after production preprocessing. The documented input-stage adaptation is not itself a defect; hiding its consequences or substituting another candidate would be.
**Detection:** follow selected proposals from original tokens through typed parsing and the explicit input-stage adaptation. Recompute identities after every real modification; inspect whether refusals, unsupported operators, and fallback seeds retain their original score, origin, and failure status accurately.
**Harden:** bind each score to its precise subject and record the original-to-executable mapping. Propagate static capability/feasibility requirements where available; do not claim a fit succeeded from grammar acceptance. Leave evolutionary mechanics and fitted-state enforcement to the §5 owner.

### G09 — MED: numerical parity and named seeds may overstate reproducible proposal behavior
**§ref / evidence:** §10.2 states a `3e-4` maximum-absolute-error tolerance for CPU/CUDA policy outputs; §10.1 distinguishes direct sampling from the full-GP RequestPlan and its random streams. Token/ranking replay guarantees in §4.4 are **UNVERIFIED**.
**Attack/gap story:** sufficiently close logits can still reverse a near-tied decision or cross a sampling threshold. A shared RNG stream can change after rejection, early EOS, batching, retries, or scheduling. Equal seeds therefore need not identify equal proposals; general backend reproducibility limitations are also documented by PyTorch.[^reproducibility]
**Detection:** retain fixed-logit sampler fixtures, logical random draws, token decisions, grammar states, and rank ties. Repeat the same plan across permitted batching/resume paths, then separately test CPU/CUDA numerical agreement and decision divergence. Examine cache reuse across contexts only where caching is implemented.
**Harden:** specify whether the guarantee is pinned-runtime replay, numerical agreement, or statistical comparability. Use candidate-local logical RNG streams or bind the exact schedule; retain the published numerical tolerance without claiming it implies identical sampled programs (§§4.4, 10.1–10.2).

### G10 — MED: raw sampling volume may conceal duplicate concentration and attrition
**§ref / evidence:** §10.1 reports eight direct sampling rounds of 128 candidates, distinct from the full-GP plan. Unique, complete, feasible neural-proposal counts and deduplication rules in §4.4 are **UNVERIFIED**.
**Attack/gap story:** many token draws can collapse to repeated recipes, superficial aliases, invalid completions, or the same unsupported family. Silent replenishment can exceed the intended generator budget; deduplicating with unsafe semantic assumptions can discard genuinely different useful programs.
**Detection:** maintain separate counts for scheduled draws, completed sequences, parser-valid programs, canonical identities, executable subjects, feasibility refusals, rank survivors, and handed-off candidates. Break down duplicates and failures by grammar/operator family, length, and sampler setting; reconcile every attempt.
**Harden:** preserve raw and conservatively canonicalized identities, count rejected/duplicate attempts, cap replenishment, and label fallback origin. Use behavioral fingerprints diagnostically, not as proof of semantic equivalence. Close with §4.4 sampling/dedup traces and boundary receipts from §5.4.

## Ranked improvements / kickoffs (P0/P1/P2)

### I01 — P0: accepted semantic-bundle firewall [closes G01]
**§ref / why:** §§4.2, 4.4–4.5. This extends the documented loader discipline to the dependencies that give accepted tensors their meaning, without changing model capacity.
**Kickoff / proposed owner:** policy-runtime owner publishes the required bundle fields, routes accepted-serving entrypoints through validation, and binds receipts to opened bytes rather than directory names. Challenger construction remains explicitly separate.
**Falsifiable acceptance:** every negative fixture in G01 is refused before accepted inference; the exact permitted bundle loads and produces a matching proposal receipt. A vocabulary permutation with unchanged cardinality must fail. Rejected step-69,000 and experimental 256×2 states cannot acquire accepted lineage-B status through any entrypoint.
**Stop condition:** any compatibility fallback that runs before identity validation blocks qualification of that entrypoint; an explicitly identified challenger result is not an accepted-model result.

### I02 — P0: autoregressive conformance suite [closes G02]
**§ref / why:** §§4.1, 4.3–4.4; §10.2. Masks and shifts are inexpensive to test relative to the cost of accepting falsely conditioned proposals.
**Kickoff / proposed owner:** inference and training owners agree on logical token positions and target shifts; build suffix-perturbation, supported-padding, batch-permutation, teacher-forced/incremental, and inference-mode fixtures. Test implemented cache paths against the uncached reference.
**Falsifiable acceptance:** earlier next-token outputs are invariant to unavailable suffixes; padding does not alter valid-token outputs or sequence scores; equivalent logical histories agree under the declared numerical policy. Unsupported lengths/layouts reject deterministically. The published CPU/CUDA output check retains `3e-4`; additional intermediate tolerances are explicitly declared, not retroactively loosened.
**Stop condition:** any target leakage, context omission, unsupported silent broadcast, or unexplained mode dependence blocks a conformance claim, even when final parsing succeeds.

### I03 — P0: program-sensitive scorer certificate [closes G03]
**§ref / why:** §§4.1, 4.3–4.4; Appendix A. Establishing a real program-to-score path is more valuable than adding another nominal reward head.
**Kickoff / proposed owner:** model/ranking owner publishes readout positions, masks, head inputs, target semantics, active consumers, and training status. Retain dependency traces and disposable head-perturbation results with accepted model identity.
**Falsifiable acceptance:** every claimed program-dependent serving score has a demonstrated computational path from program tokens and responds on controlled nonsaturated fixtures; no scorer relies solely on a program-inaccessible causal prefix. Padding is excluded according to the declared rule. Each stored head is explicitly active for a named consumer or inactive, with corroborating tests.
**Stop condition:** a structurally context-only score may be retained as a context prior, but cannot be represented as per-program quality. Readout redesign changes the model contract and must be evaluated as a challenger.

### I04 — P0: total, bounded grammar/sampling state machine [closes G04]
**§ref / why:** §4.4; §5.1. Converting exceptional decoder states into explicit outcomes prevents validity inflation and makes failure reproducible.
**Kickoff / proposed owner:** grammar/decoder owner specifies the order of enabled filters, EOS acceptance, length accounting, retry caps, and numeric checks; differentially tests the decoder against the production typed parser over a declared bounded state domain plus adversarial fixtures.
**Falsifiable acceptance:** every emitted token belongs to the legal support actually used to sample it; successful sequences reach an accepting parser state; invalid numerics, no support, and exhaustion yield bounded typed outcomes. No test may recover through unmasked sampling or relabel an unauthorized repair as an original draw. Exact retries appear in the attempt ledger.
**Stop condition:** one accepted invalid completion fails the safety gate. Any additional completion-feasibility pruning must be versioned and evaluated as a changed proposal policy, not silently imposed on the accepted benchmark plan.

### I05 — P0: executable architecture specification and golden trace [closes G05]
**§ref / why:** §§4.1–4.3; Appendix A. A reviewer should be able to identify the accepted computation without guessing missing layers or treating parameter count as behavior.
**Kickoff / proposed owner:** architecture/publication owner supplies the full context MLP, prefix placement, positional rules, decoder operation order, embedding/logit heads, reward/runtime paths, normalization settings, and serving mode; links each active operation to the actual tensor inventory.
**Falsifiable acceptance:** an independent implementation using that contract reproduces accepted golden outputs under declared tolerances; every stored tensor is accounted for as active or inactive; no undocumented operation or inferred intermediate width is needed. The visible `[512,512]` projection has an explicitly documented upstream input producer.
**Stop condition:** matching the reported 136 tensors or 27,796,494 parameters alone does not pass. Golden traces must carry accepted-weight and executable-bundle identities, not an unrelated reference initialization.

### I06 — P1: pre-gate shadow evaluation and ranker qualification [closes G06]
**§ref / why:** §4.4; §§6.1, 7.1. Ranking saves evaluations only when its excluded candidates are sufficiently worse, not merely unseen.
**Kickoff / proposed owner:** ranking/evaluation owners preregister diagnostic candidate pools, permitted fitting-side scoring, diagnostic budgets, and comparisons against random and likelihood-only ordering. Keep this study separate from the accepted timing protocol; log actual score components and unit conversions.
**Falsifiable acceptance:** at equal evaluation budget on matched pools, the live ranker improves the best permitted fitting-side OOF objective reached over both random and likelihood-only ordering, with paired uncertainty excluding no benefit; invalid and refused proposals remain in accounting. Report false rejection, coverage, and selection regret against the evaluated feasible subset. Survivor-only agreement cannot pass.
**Stop condition:** an unqualified reward gate cannot be advertised as beneficial. Any fallback ranker is separately identified and evaluated; no automatic transfer claim follows from a successful fitting-side ranking study.

### I07 — P1: conditional-generator contribution experiment [closes G07]
**§ref / why:** §§4.3–4.4; §§5.5, 7.5. This distinguishes useful conditioning from imitation, generic fallback, and downstream search effort.
**Kickoff / proposed owner:** policy/evaluation owners preregister correct-context, mismatched-context, block-intervention, and properly trained program-only comparisons. Separate decoder-context interventions from ranker-context interventions, using pre-gate samples and fixed-pool reranking. Freeze budgets, task restrictions, and downstream protocols; use the other owners’ authenticated partition boundaries rather than inventing a new split here.
**Falsifiable acceptance:** correct conditioning improves a preregistered feasible-proposal objective over the matched control with paired uncertainty excluding no benefit; report generation-zero and downstream outcomes separately. Include all planned tasks/attempts and declared resource costs. Output sensitivity alone or GP-only improvement does not establish the stronger proposal-quality claim.
**Stop condition:** no positive result means “conditional benefit unestablished,” not necessarily “context unused.” A later P2 multi-context-token or per-layer-conditioning experiment requires a new explicit lineage, matched training/resources, and new qualification; no undisclosed architectural change to lineage B.

### I08 — P1: proposal-to-executable identity receipt [closes G08]
**§ref / why:** §4.4; §§5.1, 5.4. This preserves the generator’s contribution across an explicitly adapted executable representation.
**Kickoff / proposed owner:** decoder/pipeline interface owners retain original tokens/bytes, bundle identity, raw score subjects, parse result, explicit adaptation, executable identity, feasibility status, and origin. Use the existing original-versus-executable distinction rather than replacing it with a single ambiguous recipe field.
**Falsifiable acceptance:** every ranked and handed-off candidate is traceable to its exact original proposal; every score identifies whether it estimates original or adapted semantics. Modification, refusal, repair, or fallback cannot silently inherit another subject’s identity. A successful decode is never counted as a successful fit.
**Stop condition:** missing identity edges quarantine that record from generator-quality attribution; the pipeline may continue only with an explicitly authorized, separately labeled alternative.

### I09 — P1: replay capsule with decision-aware numerical qualification [closes G09]
**§ref / why:** §4.4; §§10.1–10.2. Repeatable token choices and comparable numerical outputs are related but different properties.
**Kickoff / proposed owner:** inference owner records request-plan identity, logical RNG streams/draw counters, sampler settings, grammar states, retry history, deterministic tie rules, backend/software identity, and score subjects. Bind direct requests and full-GP plans separately; investigate candidate-local streams to reduce schedule coupling.
**Falsifiable acceptance:** pinned-runtime replay reconstructs tokens, failures, ranks, and identities; fixed-logit fixtures reproduce sampler decisions; supported schedule/resume variations either preserve declared behavior or are explicitly excluded. Cross-backend outputs within §10.2’s actual qualification scope meet its documented check; any additional logits/readouts need explicit criteria. Near-tie decision divergence is measured and reported rather than hidden.
**Stop condition:** a numerical-tolerance pass cannot substitute for a failed replay guarantee. Changing stream allocation or tie-breaking creates a newly qualified decoding protocol, even with identical weights.

### I10 — P2: diversity- and feasibility-aware proposal budget [closes G10]
**§ref / why:** §4.4; §§5.4, 5.6. More distinct usable candidates can be more valuable than more raw draws, but feasibility pruning must not eliminate useful programs by assumption.
**Kickoff / proposed owner:** generator owner first builds the reconciled stage ledger and conservative identity scheme; then compares a versioned duplicate-aware allocator and sound static capability filters against the frozen sampler. Preserve grammar-family exploration and report any extra decode, filtering, and encoder costs.
**Falsifiable acceptance:** at matched total proposal resources, the challenger increases distinct feasible yield with paired uncertainty excluding no gain and satisfies a preregistered objective noninferiority rule. Duplicate/refusal/retry costs stay in the denominator; behavioral fingerprint collisions cannot trigger irreversible deduplication. False feasibility refusals fail the filter’s soundness qualification.
**Stop condition:** no promotion on throughput or uniqueness alone. A sampler that becomes diverse by proposing worse programs, silently spending more, or discarding unproven-equivalent recipes fails this kickoff.

## Evidence gaps / UNVERIFIED load-bearing claims

The following are closure requests, not assertions that the full monograph or repository lacks the evidence. A source path listed in §12.1 is not a substitute for inspecting its pinned content.

| Claim requiring closure | Status in this review | Exact closure target |
|---|---|---|
| Accepted lineage B step 60,000 and exact byte identity | Documented; asset and selection receipt UNVERIFIED | §4.2 loader/receipt; §4.5 accepted-model record |
| Rejected step 69,000 cannot become accepted through another caller | Rule documented; enforcement coverage UNVERIFIED | §4.2 entrypoint matrix and negative load tests |
| Full context-to-prefix computation | Partly documented; MLP bridge UNVERIFIED | §§4.1, 4.3; complete Appendix A; `model.rs`/`config.rs` identified by §12.1 |
| Exact decoder operation order and accepted inference mode | Widths/equations partly visible; remaining configuration UNVERIFIED | §§4.1, 4.3; identity-bound golden trace |
| Vocabulary, special tokens, positions, and length accounting | UNVERIFIED; no sizes or limits inferred | §§4.3–4.4; vocabulary and decoder configuration |
| Autoregressive training/inference equivalence | Causal/PAD intent documented; behavior UNVERIFIED | §§4.1, 4.3–4.4; target-shift and mask tests |
| Reward sees the complete permitted program | UNVERIFIED; no pooling or scorer topology visible | §§4.3–4.4; readout dependency graph |
| Stored versus active heads and serving score units | Inactive-head warning documented; exact routes UNVERIFIED | §4.3; §6.1 consumer mapping; complete Appendix A |
| Grammar legality, EOS, cutoff ordering, and numeric failure behavior | Implementation status asserted; operational contract UNVERIFIED | §4.4; pinned `grammar.rs`, `decode.rs`, and tests |
| Sampling replay and direct/full-GP plan separation | Distinction documented; replay traces UNVERIFIED | §4.4; §10.1 retained RequestPlan and stream records |
| Ranker usefulness and conditional proposal lift | UNVERIFIED; §7.5 explicitly does not supply measured transfer lift | §4.4 ablations; §§5.5, 7.5 matched evaluations |
| Unique-feasible proposal yield and exact adaptation attribution | Original/executable distinction documented; counts/traces UNVERIFIED | §4.4; §§5.1, 5.4 proposal handoff receipts |

## Fail-open / silent-lie paths (if any in shard)

These are **conditional failure paths to test**, not observed product behaviors. A labeled, authorized fallback may be operationally legitimate; hiding its origin is not.

| Trigger | Silent-lie path to rule out | Required behavior |
|---|---|---|
| Same tensor shapes, changed token meanings or grammar | “Accepted policy” executes a different language | Reject incompatible semantic bundle before accepted inference |
| Empty legal support or non-finite sampling distribution | Retry without grammar and count a valid neural completion | Typed bounded failure with every attempt retained |
| Length limit reached outside an accepting grammar state | Force EOS/repair and report the original draw as complete | Reject or separately identify authorized repair and its new subject |
| Missing, inactive, or program-blind reward route | Substitute a default score and advertise learned program ranking | Explicit unavailable/context-prior status or a separately qualified ranker |
| Mis-shifted targets, padding pollution, or stale implemented cache | Outputs appear conditional but depend on unavailable/wrong tokens | Fail conformance; invalidate the affected proposal receipt |
| Parser adaptation or replacement loses original identity | Original neural score is attached to a different executable candidate | Preserve both subjects and score semantics; reject missing mappings |
| Neural proposal exhausted; generic seed used | Final success is credited to the neural generator | Retain explicit origin and separate contribution accounting |
| CPU/CUDA output tolerance passes near a decision boundary | Report identical proposals without checking token/rank decisions | Report numerical and decision qualification as separate results |

## Cross-links to other sections the owner must not ignore

| Other section | Generator owner’s required boundary, without taking over that shard |
|---|---|
| §§3.1, 3.5–3.6 | Consume the declared normalized context blocks with semantic identity; reject malformed/incompatible inputs. Encoder construction and fit-scope provenance remain the encoder owner’s responsibility. |
| §§5.1, 5.4–5.6 | Preserve original/executable subjects, proposal origin, and refusal/budget records. Request necessary feasibility information; do not change GP selection/evolution rules within this kickoff. |
| §6.1 | Match head consumers to training targets, legal-token conventions, and runtime units. Stored heads and optimizer activity are not proof of serving utility; broader objective/admission review remains separate. |
| §§7.1, 7.5–7.6 | Use declared permitted objectives and attempt denominators; separate generation-zero, downstream, and transfer claims. Do not upgrade a diagnostic fitting-side ablation into transfer evidence. |
| §8.1.1 | Preserve discovery-side proposal identities for downstream private receipts; do not conflate generated recipe bytes with a fitted export or audit emitter internals here. |
| §§4.5–4.6, 9.1 | Accept only the authorized regression bundle; historical exposure totals, recovery history, and inventory reconciliation belong to the lineage owner. No broader task support follows from a shared topology. |
| §§10.1–10.2 | Preserve direct/full-GP protocol separation and the published numerical check. Additional profiling, sampler changes, or replay tests do not retroactively qualify historical measurements. |
| §12.1; Appendix A | Obtain complete pinned model/configuration/grammar material and tensor roles. Hashes identify bytes; they do not establish successful behavioral tests. |

## Concerns / residual risk

**Immediate decision:** prioritize I01–I05 as qualification/evidence closure, then run I06–I09 to establish useful and attributable proposal behavior. I10 and richer-conditioning research are challengers, not prerequisites justified by an observed deficiency. These are proposed ownership roles and gates, not claims that work has been assigned or completed.
**Uncertainty:** the heavily trimmed §§4.3–4.6 and Appendix A may already resolve several questions. No absence-of-code conclusion, parameter recount, benchmark lift, scorer dimension, vocabulary size, or maximum sequence length is established by this review.
**Residual technical risk:** passing finite fixtures does not prove all grammar states or data distributions safe; qualification must name its tested domain. Program-sensitive scores may still be poor predictors. Identical outputs on a fixture do not prove a head inactive; dependency inspection and interventions must agree.
**Interpretation risk:** “accepted” records a selection decision, not universal task superiority. The supplied accepted-model heading is regression-specific (§4.5); classification or other task support requires its own contract and evidence. No attention cache or optional sampling filter is presumed implemented merely because a test discusses it.
**Change control:** stricter loaders and additional evidence can preserve accepted semantics; altered masks, pooling, feasibility pruning, random-stream allocation, or conditioning may change behavior. Identify and requalify such changes rather than silently relabeling them as the same accepted generator.
**IP boundary:** the constructive contribution is a testable, identity-bound conditional proposal/ranking system, not an assertion that attention, grammar restriction, calibration, or genetic programming are novel. No patentability or freedom-to-operate determination is made (§1.1).

### External mechanism references — background, not product evidence

These official references were checked during review. They describe their own APIs, not the product’s dependencies or implementation version; the product must bind its actual backend separately.
[^attention]: PyTorch, `scaled_dot_product_attention` and `MultiheadAttention` documentation. Boolean mask conventions differ; the functional attention API also requires explicit dropout handling for evaluation. `https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html` and `https://docs.pytorch.org/docs/2.14/generated/torch.nn.MultiheadAttention.html`.
[^sampling]: PyTorch, `torch.multinomial` documentation: sampling weights must be nonnegative, finite, and have nonzero total mass. This is a reference contract, not evidence that JopeDime uses this function. `https://docs.pytorch.org/docs/2.14/generated/torch.multinomial.html`.
[^reproducibility]: PyTorch, reproducibility notes: matching seeds alone does not guarantee reproducibility across releases/platforms or CPU/GPU execution. `https://docs.pytorch.org/docs/2.14/notes/randomness.html`.
