<!-- from-tar tab=1 url=https://chatgpt.com/c/6ab701f1-03a4-83ea-98e9-33a18dc26575 sha256=27745ff94492115f5b4e5fca00fd6732d60aa284a55c95348bc33623299a0ee7 chars=33728 -->

# System overview and core-contribution contract audit

**Artifact identity:** Jailgun farm; shard 1 of 10; batch tab 1. Scope: §1, especially the responsibility map and rows → programs → teachers → changed parameters → subsequent search / emission handoffs.
**Source boundary:** User-supplied structured extract only; evidence cutoff `2026-09-25T19:31:50Z`; publication source `3d71fce041d1dd1173ed98160f7d986bb426ba6f`. No product source tree, retained execution receipts, weights, or complete monograph were inspected.
**Reading rule:** §1.2–§1.7 are named in the contents but their bodies are trimmed. Their implementation and narrative completeness are **UNVERIFIED**, not established omissions in the full monograph. Cross-chapter excerpts support interface findings only.

## Executive hits

- **Critical documented boundary gap:** The monograph says encoder provenance validation is not called by live teacher eligibility. The extract therefore does not establish an enforced context-to-teacher provenance contract; actual contaminated admission is not demonstrated. Make that consumer gate mandatory before claiming closure. (§1.4–§1.5; §11.1)
- **Highest-leverage construction:** Turn the overview into a consumer-validated artifact graph with two branches: selected fitted predictor → emission, and candidate evidence → teaching → policy updates → another search. Teachers and policy weights are not the objects directly lowered into predictor code. (§1 opening; §1.5–§1.7; §6 opening; §8.1)
- **Ownership remains unverified:** The supplied §1.2 responsibility map is absent, so the extract cannot establish who rejects a bad handoff, owns retries, or authorizes promotion. Require named accountable roles, receiving validators, rejection outcomes, and retained receipts rather than a component-name diagram. (§1.2)
- **Preserve the strongest existing correction:** Original proposal, adapted executable recipe, fitted state, and emitted backend are distinct subjects. The documented fitted-export path already rejects the conceptual shortcut of refitting a recipe and calling it the original export; join that protection back to the selected candidate. (§1.7; §5.1; §8.1–§8.2)
- **Do not narrate a measured learning victory:** The extract describes local synthetic update/reload checks, not a real-corpus closed-loop qualification. §7.5 is explicitly titled a reporting contract with transfer not yet measured. A saved or changed checkpoint cannot fill that missing outcome. (§1.6; §6.1.1.1; §7.5)
- **Make evidence scope executable:** Preserve §1’s distinction between implemented, locally tested, CI verified, operationally measured, and promoted. Attach these as separate, scoped attestations and decisions; never treat them as an automatically ascending status ladder. (§1 opening; §1.3; §12)
- **Make product claims capability-specific:** Separate deterministic emission, qualified prediction, runtime dependencies, and supported model/backend combinations. The excerpt expressly limits universal Rust/C++ and zero-Python claims; the overview must retain those limits. (§1.7; §8.1; §11.1)
- **Strengthen the contribution story with witnesses:** Present the project-specific combination as an integration hypothesis supported by boundary tests and scoped outcome evidence. Keep established primitives outside the invention claim, and identify unmeasured benefits explicitly. (§1.1; §1.3; §12.1)

## Ranked IP gaps (CRITICAL/HIGH/MED/LOW)

Severity ranks the consequence of a failed handoff or misleading claim, not the certainty of an observed exploit. **DOCUMENTED** means stated in the supplied monograph, not independently verified in code. **UNVERIFIED** means the required evidence is unavailable; **HYPOTHESIS** marks a testable failure scenario.

### G1 — CRITICAL: Context provenance is not established as a live admission invariant
**§ref / evidence:** §1.4–§1.5; §2.1; §11.1. DOCUMENTED: receipt types and binding validation exist, while live `eligibility.rs` does not call them. The exact §1 account is UNVERIFIED.
**Attack / gap story:** Otherwise convincing candidate evidence could arrive with an omitted or mismatched context provenance record. Training-side fitting alone would not establish that the upstream context respected the declared scope. The missing call is documented; equivalent mandatory enforcement elsewhere is not demonstrated. Successful admission of a contaminated teacher is a HYPOTHESIS, not an observed result.

**Detection:** Exercise the actual admission entrypoint with a valid teacher fixture, then omit the upstream receipt or substitute one bound to a different context/scope. Observe the decision and every downstream corpus/HWM side effect; testing the receipt helper alone does not close the gap.

**Harden:** Require the receiving admission boundary to verify context binding and its trusted provenance before accepting the teacher. Quarantine missing/legacy provenance with an explicit migration policy; do not translate unknown provenance into a pass.

**Closure criterion:** The live-path negative cases are rejected before teacher acceptance or associated HWM/corpus mutation, while an authorized matching case succeeds. This closes a handoff invariant, not a general proof of leakage freedom.

### G2 — HIGH: No recoverable owner-and-validator map for the end-to-end contract
**§ref / evidence:** §1.2 and §1.4–§1.7, UNVERIFIED because their bodies are trimmed. The opening distinguishes artifacts but does not supply the omitted responsibilities.
**Attack / gap story:** A producer can report success while the consumer assumes someone else checked identity, scope, or completeness. Without a recoverable acceptance owner, a technically valid artifact can be used for the wrong purpose. This is a responsibility risk, not proof that the repository lacks owners.

**Detection:** Trace a real request across each boundary. For every transition, identify its accountable role, receiving validator, accepted subject, durable output, and rejection path; mark any answer relying only on producer assertions as unresolved.

**Harden:** Add a responsibility table whose unit is the transition, not the subsystem. Assign a system integration owner to maintain the table and an individual accountable role for each receiving gate; proposed roles must not be represented as existing assignments.

**Closure criterion:** The full §1.2 map resolves every transition in K2, and a retained negative integration trace demonstrates each receiving rejection. A documentation-only responsibility assignment is insufficient to establish enforcement.

### G3 — HIGH: Local identity safeguards do not establish a composed selected-subject chain
**§ref / evidence:** §1.4–§1.7 are UNVERIFIED; §5.1 and §8.1–§8.2 DOCUMENT original/executable distinction and the corrected fitted-export path.
**Attack / gap story:** A summary can accidentally join the score of one evaluated candidate, the teacher bytes of another representation, and the export of a different fitted instance. Authorized input-stage insertion and any declared final fitting produce related but nonidentical artifacts; requiring identical hashes everywhere would also be wrong.

**Detection:** Replay a selected candidate through its documented transformations. Swap a recipe, fitted artifact, or parent receipt while leaving the rest valid; require rejection. Separately force an unsupported export and check that the selected subject remains unchanged.

**Harden:** Use explicit parent-child identity edges for original proposal, executable adaptation, evaluation evidence, selected subject, fitted state, and backend artifact. Represent authorized transformations as transformations, not equality or silent replacement.

**Closure criterion:** An auditor can follow the selected subject through every authorized change and distinguish evaluation-fit evidence from any declared final fitted instance. The documented exporter correction is credited; its complete upstream composition still needs receipts.

### G4 — HIGH: Learning-loop execution and learning benefit can be conflated
**§ref / evidence:** §1.6 is UNVERIFIED; §6 opening defines the bounded loop; §6.1.1.1 describes local synthetic changed-parameter/update/reload checks; §7.5’s supplied heading says transfer is not yet measured.
**Attack / gap story:** An eligible teacher, saved file, reduced imitation loss, or changed policy could be presented as proof that future search improved. Those observations establish different links. They also do not show that previously exported predictors changed or that frozen context components learned.

**Detection:** Request one joined real-data witness from authenticated candidate evidence through the corpus consumed, actual optimizer event, reloaded changed policy, and matched subsequent search. Identify the exact link at which evidence becomes synthetic, descriptive, or absent.

**Harden:** Separate “loop executed,” “policy parameters changed,” “matched search completed,” and “improvement observed.” Record the component allowed to change and keep the resulting fitted predictor and policy checkpoint identities distinct.

**Closure criterion:** The full execution witness closes the operational chain regardless of outcome; a separate outcome report determines improvement. No gain, adverse results, and incomplete comparisons remain reportable rather than being converted into a successful-learning label.

### G5 — HIGH: Evidence identities can be silently upgraded at narrative joins
**§ref / evidence:** §1 opening and §1.3; §6.1.1.1; §10 opening; §12 opening. DOCUMENTED safeguards separate dated evidence, publication identity, measurements, and promotion; enforcement in the overview is UNVERIFIED.
**Attack / gap story:** A newer publication source can be mistaken for the source that produced an older measurement, or a local synthetic check can become “qualified on real inputs” after a report refresh. A matching file subset does not identify an entire execution or its input conditions.

**Detection:** Mutate an evidence fixture’s execution build, input scope, or status class while retaining its artifact digest. The claim renderer must reject unsupported joins or display the unresolved identity rather than importing the stronger claim.

**Harden:** Bind each claim to its subject, original execution source, input scope, command/outcome receipt, date, and applicable consumer. Record publication source separately. Promotion is a decision with an authorizing role, not a synonym for tested or measured.

**Closure criterion:** Historical and current evidence coexist without relabeling; changing presentation metadata cannot create a stronger attestation. Explicit equivalence evidence, where applicable, remains a scoped relationship rather than identity substitution.

### G6 — HIGH: Capability and dependency limits may disappear from the top-level promise
**§ref / evidence:** §1.7 is UNVERIFIED; §8.1–§8.2 document supported fitted export; §11.1 rejects broad all-head/zero-Python assertions and notes C++ host dependencies.
**Attack / gap story:** “Executable code without an LLM” can be heard as “every selected model becomes a standalone dependency-free artifact.” Deterministic lowering, compiler success, numerical qualification, supported model coverage, and runtime independence are separate promises. The excerpt does not establish that broader promise.

**Detection:** Compare the overview’s claims with retained capability receipts for requested modes and backends. Include unsupported selection, compiler failure, missing host dependencies, and failed parity; inspect whether any partial outcome is promoted to unconditional system success.

**Harden:** Freeze the requested capability and completion rule before execution, including required versus optional backends and runtime dependencies. Return explicit unsupported/failed/qualified outcomes without silently replacing the selected predictor.

**Closure criterion:** Every externally visible success resolves to the requested capability and qualified selected artifact. Required failures remain visible; a different runtime or substituted model cannot inherit the original success claim.

### G7 — MED: Terminal outcomes, retries, and publication side effects lack visible ownership
**§ref / evidence:** §1.2 and §1.5–§1.7 are UNVERIFIED; §6 opening, §7.1.2, and §9.1 establish distinct admission, HWM, corpus, and exposure concepts.
**Attack / gap story:** An interruption can leave an accepted teacher unbound to a frozen corpus, an HWM record attached to the wrong comparison, or an export receipt published before qualification. A retry might count a single logical admission twice or conceal repeated optimizer exposure. No such incident is demonstrated.

**Detection:** Interrupt and retry at durable acceptance/publication boundaries. Reconcile the logical request, teacher decision, HWM event, corpus snapshot, update evidence, and emitted-artifact status; inspect for accepted-but-unverifiable outputs and contradictory terminal states.

**Harden:** Specify idempotency and consistency per boundary, with durable transition identifiers and explicit incomplete states. Do not demand one giant transaction across the entire pipeline; define recovery ownership for each externally visible side effect.

**Closure criterion:** Recovery cannot publish conflicting logical admissions or qualification outcomes. Intentional training reuse remains allowed and separately counted as exposure; an interrupted or failed stage cannot silently disappear from the trace.

### G8 — MED: The contribution combination lacks a visible claim-to-witness boundary in §1
**§ref / evidence:** §1.1 DOCUMENTS the combination and excludes established methods from claimed inventions; §1.3 is UNVERIFIED; §12.1 provides component evidence categories, not a complete causal benefit proof.
**Attack / gap story:** Evidence that ingredients exist can be read as evidence that their combination causes improved search, transferable learning, or deployment advantage. Conversely, a generic “neural + GP” summary hides the specific identity and enforcement mechanisms that the project actually describes.

**Detection:** Attach a witness to every contribution sentence. Distinguish mechanism existence, tested invariant, measured outcome, and hypothesis; flag sentences supported only by a source path but phrased as operational or causal results.

**Harden:** State each contribution as mechanism → promised invariant or outcome → evidence scope → known limit. Retain the six named combination elements and separate integration specificity from any legal novelty claim.

**Closure criterion:** Each material benefit is either supported by its scoped comparison/receipt or explicitly unmeasured. A contract test can prove rejection behavior, but cannot be used as an accuracy, speed, or transfer result.

## Ranked improvements / kickoffs (P0/P1/P2)

These are proposed work packages, not claims about existing interfaces or assigned personnel. K1–K8 directly pair with G1–G8 and are ordered below by priority; the acceptance tests are proposed predicates, not invented measured KPIs.

### K1 — P0: Put the provenance contract on the actual receiving boundary
**§ref / why it matters:** §1.4–§1.5; §11.1. This turns a documented optional safeguard into a required context-to-teacher invariant rather than adding another disconnected receipt format.
**Kickoff / owner:** The learning-admission owner and context-producing owner trace the shipping call path together. Reuse the existing receipt work where suitable, but require receiver-side scope and subject binding before an acceptance result becomes visible.

**Deliverable:** An integration-level rejection suite and a compatibility policy for historical records with absent provenance. Preserve “unknown,” “rejected,” and “accepted” as distinct outcomes; migration must not silently authenticate old metadata.

**Falsifiable acceptance:** Omitted, mismatched, and replayed-for-another-subject provenance fails through the same entrypoint used by normal admission, with no accepted teacher or associated HWM/corpus commit. The corresponding authorized matching record passes. Supply the retained command and source identity.

**Scope limit:** The underlying encoder/fit-scope semantics belong to their section owners. This package proves that their verdict is required and correctly bound, not that every provenance issuer tells the truth.

### K2 — P0: Publish an executable responsibility map and a forked artifact story
**§ref / why it matters:** §1.2 and §1.4–§1.7. A receiving gate can be assigned and tested; a subsystem diagram alone cannot settle acceptance authority.
**Kickoff / owner:** A designated system integration owner resolves the actual owners and validators behind the following proposed transition map. No personal or repository ownership is inferred from the extract.

| Transition | Proposed accountable receiving role | Acceptance subject and required handoff |
|---|---|---|
| Rows / declared fit scope → context | Context integration owner | Prepared-input identity and upstream fit-scope/encoder provenance bound to the produced context |
| Context → conditional proposal | Proposal runtime owner | Context identity, accepted policy/configuration identity, and the actual proposed program |
| Proposal → executable recipe / search | Search integration owner | Original bytes, explicit adaptation, executable subject, and planned evaluation scope |
| Candidate evidence → teacher / HWM decisions | Learning-admission owner | Authenticated candidate evidence and distinct eligibility/comparison outcomes |
| Accepted records → frozen corpus | Corpus ingestion owner | Approved inclusion policy, record identities, and immutable snapshot consumed by training |
| Corpus → changed/reloaded policy | Training/checkpoint owner | Consumed snapshot, sampled-record exposure references, update evidence, declared trainable components, and reloaded artifact |
| Reloaded policy → matched search / selection decision | Qualification and serving-selection owner | Bound parent/challenger subjects, matching protocol, retained outcome, and explicit decision |
| Selected fitted predictor → qualified emission | Export integration owner | Selected-subject ancestry, fitted state, requested capability, and backend qualification |

**Narrative replacement:** Rows and their declared scope condition proposals; executable recipes are evaluated. Candidate evidence may enter teaching under explicit admission and corpus rules, while a selected predictor has a separate fitted-export path. Policy updates affect subsequent proposal/search runs, not an already emitted predictor. (§1 opening; §5.1; §6 opening; §8.1)

**Falsifiable acceptance:** The complete §1.2 names an accountable owner and real receiving validator for every applicable row. A traced request resolves every parent edge; an invalid boundary input reaches that receiver’s documented rejection rather than an undocumented fallback.

**Decision separation:** Eligibility is not automatically HWM improvement, corpus inclusion, serving promotion, or export qualification. The actual corpus inclusion policy and final-fitting boundary must be supplied, not inferred. (§7.1.2; §1.5–§1.7)

### K3 — P0: Build one selected-subject witness with explicit transformation edges
**§ref / why it matters:** §1.4–§1.7; §5.1; §8.1–§8.2. This composes existing safeguards without erasing legitimate differences between proposals, executable recipes, evaluation fits, and deployment fits.
**Kickoff / owner:** Search and export integration owners produce a retained witness for an actual supported request. Link original proposal, inserted stage, executable recipe, evaluation evidence, selected candidate, any declared final fitting, final fitted state, and backend outputs.

**Deliverable:** A private authoritative ancestry record and an appropriately scoped public receipt. Every transformation states its inputs, outputs, and authority; a hash identifies bytes but does not by itself prove authorization or correctness.

**Falsifiable acceptance:** Swapping any parent subject causes rejection or an explicitly new subject requiring new qualification. Unsupported export retains the selected predictor and reports the limitation; it cannot borrow the receipt of an easier replacement.

**Scope limit:** Numerical algorithms, support-matrix completeness, and backend internals stay with their owners. This witness demonstrates which subject those checks qualified and where the evidence is retained.

### K5 — P0: Generate the dated overview from a scoped claim/evidence ledger
**§ref / why it matters:** §1 opening and §1.3; §12. This preserves the monograph’s unusually useful evidence vocabulary across updates, presentations, and serving decisions.
**Kickoff / owner:** The publication/evidence owner defines proposed records for claim subject, execution identity, input scope, evidence class, receipt reference, observation date, limitations, and any authorized promotion decision. Keep publication identity in a separate field.

**Deliverable:** A generated §1.3 summary with independent attestations, not one scalar maturity label. Source implementation, synthetic/local tests, CI, operational execution, and promotion may each have different scopes and dates.

**Falsifiable acceptance:** A local synthetic receipt cannot satisfy a real-input operational claim; a publication rebuild cannot relabel a historical execution; a promotion decision cannot manufacture missing qualification. Contradictions block the stronger claim while preserving valid weaker evidence.

**Scope limit:** Receipt authenticity and issuer authority require a defined trust boundary. Content hashes and an attractive status display do not supply either automatically.

### K4 — P1: Produce the smallest real closed-loop execution witness
**§ref / why it matters:** §1.6; §6 opening and §6.1.1.1; §7.5. A real joined trace would strengthen the learning story without pretending that synthetic update checks establish useful transferable learning.
**Kickoff / owner:** Training and qualification owners select a permitted task and retain candidate evidence, admission/HWM outcomes, the frozen corpus snapshot, actual sampled-record identities, optimizer/update evidence, saved/reloaded policy identity, and the subsequent matched search.

**Deliverable:** A report with separate verdicts for admission, HWM comparison, corpus inclusion, parameter update, reload, matched-search execution, and observed outcome. State which components are frozen and which are authorized to change; available manifest records must not be presumed to have been sampled.

**Falsifiable acceptance:** A checkpoint save with no demonstrated parameter update cannot pass the changed-policy claim. A changed policy with no matched rerun cannot pass the completed-loop claim. A completed loop with no demonstrated benefit cannot pass the improvement claim.

**Scope limit:** A development witness is not a transferable-lift measurement. Any such assertion additionally requires the independent reporting/evaluation contract owned by §7.5; an unfavorable outcome does not invalidate honest execution evidence.

### K6 — P1: Make a capability-scoped success contract visible at request and publication time
**§ref / why it matters:** §1.7; §8.1; §11.1. A precise runtime promise is stronger than an expansive claim that silently excludes unsupported selections or depends on an unmentioned host.
**Kickoff / owner:** Product integration and export owners declare requested execution mode, model/backend support, runtime dependencies, required/optional outputs, and the success rule before execution. Preserve direct-request versus full-search identity at the overview boundary. (§10.1)

**Deliverable:** A receipt-backed capability statement distinguishing source generated, compiled, numerically qualified, and deployable under the declared runtime. “No LLM in emission” does not imply “no Python host” or universal model coverage.

**Falsifiable acceptance:** A failed required backend or missing runtime dependency cannot produce unconditional success. Partial results remain labeled partial; unsupported emission never changes the selected model without a new explicit selection and qualification subject.

**Scope limit:** Backend owners define their numerical qualification protocol. This work ensures the overview neither strips its limits nor upgrades finite-scope parity into universal equivalence.

### K7 — P1: Add failure-and-retry traces to the system contract
**§ref / why it matters:** §1.2 and §1.5–§1.7. Explicit failure ownership prevents a chain of individually successful log messages from masquerading as one completed request.
**Kickoff / owner:** Each receiving owner identifies its acceptance/publication side effect and recovery rule. Exercise interruption before and after teacher admission, HWM recording, corpus freezing, checkpoint publication/reload, and export qualification where those transitions occur.

**Deliverable:** A retained transition journal with proposed stable logical event identifiers, parent subjects, terminal outcomes, and reconciliation rules. Distinguish logical admission from physical retries and intentional optimizer reuse.

**Falsifiable acceptance:** Recovery produces no contradictory logical admissions, no unqualified artifact labeled qualified, and no HWM mutation under a different comparison identity. Repeated training consumption remains visible as exposure rather than being erased by idempotency.

**Scope limit:** Do not prescribe a repository-wide transaction protocol without source inspection. The work package requires observable invariants at existing durable boundaries and leaves the implementation to their owners.

### K8 — P2: Rewrite core contributions as contract-backed, testable claims
**§ref / why it matters:** §1.1 and §1.3. The distinctive story is the documented combination and its disciplined transitions, not an unsupported claim to have invented attention, optimization, regression, or GP.
**Kickoff / owner:** The system author and evidence owner create a claim card for each named combination element: frozen three-part context, grammar restriction, original/executable identity, paired bounded search, fitted-export receipts, and teaching/HWM closure.

**Deliverable:** Each card records the mechanism, intended invariant or benefit, scoped evidence, current limitation, and the result that would falsify the claim. Mark composition evidence separately from evidence for the ingredients.

**Falsifiable acceptance:** Every material §1 contribution resolves to a tested invariant, a retained scoped outcome, or an explicit unmeasured hypothesis. Unsupported causal wording is removed; contract tests are not relabeled as accuracy or performance gains.

**Scope limit:** Demonstrating a project-specific integration does not establish patent novelty or freedom to operate; the supplied §1.1 expressly makes no such determination.

## Evidence gaps / UNVERIFIED load-bearing claims

The following specifies what would close the missing overview evidence. Cross-links identify the appropriate supplying section; they do not upgrade trimmed material into reviewed evidence.

| Missing or limited claim | Present evidence and status | Evidence needed to close it |
|---|---|---|
| Current accountable responsibility map | §1.2 body trimmed: UNVERIFIED | Full §1.2 with actual acceptance owners, consumer validators, failure ownership, and retained integration traces |
| Dated end-to-end system explanation | §1.3 body trimmed: UNVERIFIED | Full §1.3 joined to §12’s ledger, distinguishing publication identity from original execution identities |
| Original rows become an authorized conditional proposal | §1.4 body trimmed; §2.1 describes upstream leakage channels | Full §1.4 with consumed-input/context lineage and live boundary evidence; encoder details supplied by §3.5–§3.6 |
| Candidate evidence becomes an eligible teacher under all required gates | §1.5 body trimmed; §11.1 documents non-wired encoder validation | Full §1.5 and §6.4 plus actual admission-path negative tests, corpus-inclusion rules, and §7.1.2 decision bindings |
| Teaching changes the intended policy and closes a real subsequent search loop | §1.6 body trimmed; §6.1.1.1 only describes dated local synthetic checks | Full §1.6 with a joined real-corpus/update/reload/rerun witness from §6.5–§6.8; separate §7.5 evidence for any transfer assertion |
| Selected fitted predictor is the subject actually emitted | §1.7 body trimmed; §8.1–§8.2 document the corrected supported path | Full §1.7 with selected/refit/fitted/export ancestry, capability scope, and per-artifact qualification; clarify §5.9’s historical versus current boundary |
| Context provenance is enforced in normal eligibility | §11.1 explicitly says live eligibility does not call the validator | A later identified source/command receipt proving mandatory integration; present extract does not supply this closure |
| Project-specific combination produces an operational benefit | §1.1 states mechanisms; §12.1 identifies component evidence | Full §1.1/§1.3 claim-to-witness mapping, with scoped comparative evidence for any benefit or an explicit unmeasured label |

## Fail-open / silent-lie paths (if any in shard)

Only the first row has an explicit source-described wiring gap. The remaining rows are falsifiable failure hypotheses or expressly forbidden behaviors, not findings of observed production incidents.

| Path | Evidence classification / §ref | Required receiving behavior |
|---|---|---|
| Provenance exists in a helper but is not required by live teacher admission | DOCUMENTED wiring limitation; §11.1, affecting §1.4–§1.5 | Require and bind the provenance verdict before admission side effects |
| Well-formed receipt is replayed for a different context, selected recipe, or fit | HYPOTHESIS; §1.4–§1.7; §5.1; §8.1.1 | Reject the subject mismatch even when bytes/hash format are valid |
| Eligible teacher is labeled an HWM improvement or automatic corpus inclusion | HYPOTHESIS; §1.5; §7.1.2 | Preserve independent decision outcomes and the declared inclusion policy |
| Checkpoint saved/changed is labeled a completed learning benefit | HYPOTHESIS; §1.6; §6 opening; §7.5 | Require the distinct update, reload, matched-search, and outcome evidence |
| Unsupported export silently selects a simpler candidate | Explicitly forbidden, not observed; §1.7; §8.1.1 | Preserve selection and report unsupported; any reselection creates a new subject |
| Historical or synthetic evidence receives a current operational badge | HYPOTHESIS; §1.3; §6.1.1.1; §12 | Retain original scope/date/source; block unsupported evidence-class upgrades |
| Retry, partial backend success, or direct request becomes “full pipeline complete” | HYPOTHESIS; §1.2; §1.7; §10.1 | Reconcile terminal outcomes against the predeclared request mode and required outputs |

## Cross-links to other sections the owner must not ignore

These are interface dependencies of §1, not additional audits of those chapters.

| Owning section | Contract that §1 must consume rather than reinvent |
|---|---|
| §2.1; §3.1 and §3.4–§3.6 | Prepared-input/fit-scope and context-product provenance; “frozen” or shape-compatible is not proof of permitted provenance |
| §4.2; §5.1 and §5.9–§5.10 | Accepted policy identity, original/executable recipe identity, candidate ancestry, and the actual declared final-fitting boundary |
| §6.4–§6.8; §7.1.2 | Teacher authenticity, separate eligibility/HWM outcomes, corpus inclusion, updates, and continuation semantics |
| §7.5; §9.1 | The difference between a completed operational loop, recorded optimizer exposure, and measured transferable improvement |
| §8.1–§8.2 | Selected fitted subject, public/private receipt boundary, supported capability, and per-artifact qualification |
| §10.1–§10.2 | Exact measured execution mode, resource/timing scope, and qualification identity; no direct/full-GP substitution |
| §11.1; §12.1 | Known enforcement exceptions and the source/receipt evidence behind each overview claim |

## Concerns / residual risk

**Assessment limit:** The source cutoff is the document’s declared boundary, not an independent verification date. Missing §1 bodies may already contain parts of these proposals. The source-described G1 limitation is stronger evidence than a general absence of detail, but even it has not been reproduced here. No current production exploit is asserted.

**Proof limit:** A fully bound artifact graph can still carry incorrect inputs, mistaken evaluator results, or trusted-but-wrong receipts. Boundary tests prove rejection and linkage behavior; their success does not establish general leakage freedom, transfer, numerical correctness, or useful search. The relevant chapter owners must supply those qualifications.

**Operational limit:** Fine-grained receipts can add storage, implementation, and privacy burdens. Keep the authoritative private discovery evidence separate from public fitted artifacts as §8.1.1 requires; publish scoped claims without pretending that redacted evidence is independently reproducible in full.

**Decision recommendation:** Close K1, K2, K3, and K5 before presenting the complete chain as enforced and provenance-qualified. Preserve the documented implementation strengths now, describe real-loop benefit as unverified until K4 and the relevant outcome contract close it, and state export capabilities narrowly under K6. This is a gate on claims about the chain, not a claim that all individual components are unusable.
