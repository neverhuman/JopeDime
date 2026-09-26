<!-- from-tar tab=9 url=https://chatgpt.com/c/6ab70383-18a0-83ea-9995-26fe4c8add25 sha256=b8de3b9fa7873684b6a473ff10f439e2cd4a47ac7a60a1c4aae07d3bf88a00e5 chars=37062 -->

# Model inventory, weights lineage and training-scale audit

**Review identity:** Jailgun shard 9 of 10; batch tab 9.  
**Primary scope:** §§4.5–4.6, §9 and Appendix A; other sections appear only where they bind model identity or training attribution.  
**Monograph evidence cutoff:** `2026-09-25T19:31:50Z`.  
**Publication source, as supplied:** `3d71fce041d1dd1173ed98160f7d986bb426ba6f`.  
**Accepted weights SHA256, as supplied:** `4b4b5a40f03c9226b4b15856fd35b90127c8b6a4d1435b815484cdd050744f1e`.  
**Declared accepted inventory:** 136 tensors; 27,796,494 scalar parameters; F32 storage (§4 opening; Appendix A). These are document claims, not an independently authenticated asset inventory.  
**Evidence boundary:** Only the supplied structured extract was available for product review. No product code, weights, complete tensor manifest, training journals, promotion receipts or omitted chapter bodies were inspected. External primary references inform proposed methods, not product verification.  
**Status convention:** DOCUMENTED means stated in the extract; CHECKED HERE means arithmetic on visible rows only; UNVERIFIED means the necessary product evidence is unavailable. Proposed field names and tests are recommendations, not existing APIs or commands.  
**Ranking convention:** CRITICAL denotes the consequence if a hypothesized identity bypass exists, not a demonstrated exploit. HIGH denotes material attribution/reproduction risk; MED denotes consequential scope or reporting ambiguity. No implemented bypass is established here.  

## Executive hits

- **CRITICAL — A matching checkpoint label or tensor topology is not accepted-model authorization.** §4.2 documents exact-loading checks, but their trusted receipt source, coverage and fail-closed behavior remain UNVERIFIED. Keep the stated accepted digest as a release-controlled expectation, not a value learned from the candidate itself (§§4.2, 4.5).
- **P0 — Turn the existing exact-loading contract into a mandatory acceptance firewall.** Exercise missing receipts, payload swaps, rejected-state aliases and wrong consumer roles through every accepted-model entry point; preserve explicit challenger loading rather than forbidding research (§§4.2, 4.5–4.6).
- **HIGH — The gen31 / native 108-block / A100 recovery genealogy cannot be reconstructed from this packet.** These focus labels do not supply parent hashes, reset semantics, ordering or update totals. Neither the accepted 60,000 label nor the rejected 69,000 label establishes a continuous training history (§§4.5–4.6, 9.1).
- **P0 — Build a receipt-backed lineage graph, not a timeline inferred from filenames.** Separate weight inheritance, optimizer restoration, corpus transitions, conversions and promotion decisions; retain unresolved edges as unknown (§4.6; §9.1).
- **HIGH — Corpus availability, sampled exposure and effective optimizer updates remain separate denominators.** A catalog count cannot establish how much the accepted branch learned, and rejected-branch work must not be attributed to accepted ancestry without a recorded inheritance path (§9.1).
- **P0 — Add a reconciled exposure ledger.** Distinguish unique program identities, repeated sampled records, supervised token positions, minibatches and successful updates; publish accepted-ancestry totals separately from all experimentation, with explicit coverage (§9.1).
- **HIGH — Appendix A is only partially auditable here.** CHECKED HERE: its 34 complete visible rows have consistent shape products and sum to 8,669,184 scalar entries. This does not verify the full claimed inventory, F32 payload bytes or trainable/active parameter counts (§4 opening; Appendix A).
- **P1 — Bind publication, training, execution and promotion evidence without merging them.** A regenerated inventory or later publication edit must not silently relabel historical measurements or upgrade synthetic CPU evidence into A100 recovery evidence (§4.6; §9; §§6.1.1.1, 12.1).

## Ranked IP gaps (CRITICAL/HIGH/MED/LOW)

### G1 — CRITICAL: accepted bytes versus a trusted acceptance decision
**§ref / status:** §§4.2, 4.5; publication preface. Exact name/shape/dtype and receipt identity checks are DOCUMENTED; the trust anchor and all-path enforcement are UNVERIFIED.  
**Attack/gap story:** A loader could compare an artifact against its own replaceable sidecar, validate one pathname and load another object, or reach a constructor without the accepted check. Each would let compatible but unauthorized bytes inherit the accepted label. None is proven present.  
**Detection:** Trace every accepted load boundary and mutate the payload while preserving its schema; replace both the artifact and its self-consistent receipt; remove the receipt; redirect a cache/path after validation. Require refusal before accepted inference and an explicit failed identity record.  
**Harden:** Anchor the expected digest and approval to release-controlled policy; consume the same immutable object that was validated; isolate experimental constructors and cache identities. A challenger may run as a challenger, never inherit accepted status.  
**Closure:** Provide the complete §4.2 implementation references, §4.5 acceptance receipt and retained negative-test results tied to an exact executable identity. A successful baseline load alone does not close this gap.  

### G2 — CRITICAL: correct policy weights, wrong semantic bundle
**§ref / status:** §§4.5–4.6; identity dependencies in §§3.4–3.6, 4.2 and 7.1.2. The complete bundle binding is UNVERIFIED in this extract.  
**Attack/gap story:** The accepted weight digest could accompany a different token-to-operation mapping, grammar, encoder/projection asset or context-normalization convention. Tensor compatibility would then coexist with a different effective proposer; its outputs could still be attributed to the accepted model.  
**Detection:** Hold policy bytes constant and substitute each semantic dependency independently. Test cold loading and warm caches. Accepted execution must reject an unapproved bundle; an authorized comparison must acquire a distinct execution identity.  
**Harden:** Separate a release bundle identity from a request/protocol identity. Bind weights, model configuration, vocabulary semantics, grammar and frozen context assets in the former; bind sampling settings, input scope, executable and device configuration in the latter.  
**Closure:** Obtain the full accepted assembly in §4.5, component identities from §3.4, and receipts showing that the runtime—not merely report generation—checks their relationships. This is an identity handoff finding, not a redesign of the encoders.  

### G3 — HIGH: an unresolved genealogy can become a fictional continuous run
**§ref / status:** §§4.5–4.6 and §9.1. The headings and accounting warning are visible; the detailed lineage bodies and historical series are omitted.  
**Attack/gap story:** gen31, the native 108-block history and A100 recovery could be flattened into one monotonically improving run. A reset, rejected continuation or independent initialization could then inherit the accepted model's ancestry, exposure or accomplishments without a valid edge.  
**Detection:** Attempt to resolve every claimed ancestor of the accepted digest to a parent artifact and transition receipt. Find reused local step labels, unresolved parents, cycles, conflicting parent claims and events attributed to more than one branch.  
**Harden:** Maintain a directed lineage graph with separate weight-parent and optimizer-state relationships. Record accepted, rejected and unresolved states without inferring chronology from numeric labels or hardware names.  
**Closure:** Full §§4.5–4.6 and §9 historical evidence must identify what “gen31” and “108-block” mean, their actual relation to recovery, and which path reaches the accepted bytes. Do not interpret 108 as decoder depth or multiply it by an assumed block length.  

### G4 — HIGH: recovery can restore weights without restoring the training trajectory
**§ref / status:** §4.6 and §9.1; dependencies in §§6.1.1 and 6.8. Resume-state details and the four actual continuation commands are omitted and UNVERIFIED.  
**Attack/gap story:** An A100 recovery could reload weights but reset optimizer moments, scheduler position, sampler cursor or random state while retaining a continuation label. A torn multi-file save could also combine weights and optimizer state from different update boundaries.  
**Detection:** Inspect the saved-state membership and transaction boundary; interrupt saving and recovery; compare the next sampled records, effective settings and controlled next update against an uninterrupted branch. Missing state must produce refusal or an explicitly different continuation classification.  
**Harden:** Use an atomic committed training-state manifest, stable parameter-to-optimizer-slot mapping and explicit reset declarations. PyTorch's general-checkpoint guidance also distinguishes weights from optimizer state; it does not verify this native implementation.[^resume]  
**Closure:** Retrieve the full §6.8 semantics and the §4.6 recovery receipt. A changed checkpoint hash alone cannot distinguish an optimizer update from conversion, metadata rewriting or manual mutation.  

### G5 — HIGH: step and corpus counters do not establish accepted-branch exposure
**§ref / status:** §9.1; §§4.5–4.6. The separation of accounting units is DOCUMENTED; the actual historical sampling/update ledgers are not supplied.  
**Attack/gap story:** Catalog entries may be reported as consumed examples; repeated records as new programs; decoded/evaluated candidates as training exposures; microbatches as optimizer steps. Restarted counters or rejected branches can further inflate the accepted lineage's apparent scale.  
**Detection:** Reconcile available-record manifests to sampler events and update receipts under run/block identities. Test duplicate receipt ingestion, repeated programs, partial accumulation, skipped updates, replay after a crash and branch forks.  
**Harden:** Preserve each unit, its scope and completeness. Define program identity before deduplication; log actual supervised positions after masking. Keep all-experiment work, accepted-ancestor work and discarded/replayed work separately reportable rather than deleting inconvenient work.  
**Closure:** §9.1 must expose auditable denominators and known coverage per lineage. Missing historical intervals remain unknown; neither zero-filling nor back-solving totals from “60,000” is defensible.  

### G6 — HIGH: the full tensor inventory and payload binding are not independently checkable
**§ref / status:** §4 opening, §§4.2, 4.5 and Appendix A. The full totals and reread-from-header process are DOCUMENTED; the asset, header and `data/tensors.json` are unavailable.  
**Attack/gap story:** A stale generated inventory could describe one checkpoint while deployment loads another. A header-only check could establish declared structure without establishing that the approved payload was loaded; a total count could also hide missing or unexpected named tensors.  
**Detection:** Recreate the inventory from the exact authenticated asset; compare names, shapes, dtypes and payload ranges individually, then reconcile aggregate counts. The visible 34-row arithmetic check is a useful partial check, not a substitute for this join.  
**Harden:** Generate Appendix A and the machine-readable inventory from one verified input, record that input's digest, and validate full-file integrity separately. Safetensors metadata can be obtained without downloading the entire weight payload, which is precisely why metadata inspection and payload authentication must be distinct.[^format]  
**Closure:** Supply full Appendix A, `data/tensors.json`, extraction-tool identity and the accepted artifact or an independently reproducible authenticated inventory receipt. Do not reconstruct omitted tensors by assuming repeated layers.  

### G7 — HIGH: a hardware-labelled recovery can inherit unrelated verification
**§ref / status:** §4.6 and §9 historical evidence; §§6.1.1.1 and 10.1–10.2 are limiting cross-references. Actual A100 recovery/update evidence is UNVERIFIED.  
**Attack/gap story:** The reported synthetic CPU changed-parameter/reload checks, an A100 name, and an RTX3090 inference study could be stitched together as proof of one GPU training lineage. Those pieces establish different subjects and execution scopes.  
**Detection:** Require each claimed update interval to join parent weights, effective training configuration, real sampled corpus, runner/device evidence, optimizer event and saved child. Confirm that the child is on accepted ancestry rather than merely in the same directory.  
**Harden:** Keep training, recovery, inference qualification and publication observations as separate event classes. Store actual device/backend and execution identities; record casts or conversions as transformations rather than optimizer progress.  
**Closure:** The omitted §9 historical series and §4.6 recovery artifacts must bind the hardware claim to actual training events. §6.1.1.1's reported 97 passing synthetic CPU tests do not establish that historical A100 chain.  

### G8 — HIGH: publication provenance can silently replace measurement provenance
**§ref / status:** Publication preface; §§4.5–4.6, 9 and 12.1. Preserving original historical source/date identities is an explicit DOCUMENTED strength; enforcement remains UNVERIFIED.  
**Attack/gap story:** A refreshed report could stamp the publication commit or currently accepted model onto an older throughput point, converted asset or rejected run. A file hash proves the present bytes match an expectation, not which execution originally generated a measurement.  
**Detection:** Trace each lineage/scale claim and plotted point to its original source data, measurement build, subject weights, run identity, units and timestamp. Regenerate the publication after a text-only edit and compare the measurement identity records.  
**Harden:** Make historical evidence immutable and append corrections. Bind source, build, artifact, runtime and publication as related but distinct subjects; a claim's evidence status must survive regeneration unchanged.  
**Closure:** Obtain the retained publication manifest and §9 figure/data inputs identified by §12.1. A current inventory refresh is not a new historical training measurement, and source equality in selected files is not complete binary/environment identity.  

### G9 — MED: “accepted,” “rejected” and “recovered” need scoped decisions
**§ref / status:** §§4.2 and 4.5–4.6; §1 status definitions and §7.5. Accepted lineage B at step 60,000 and rejection of step 69,000 are DOCUMENTED; decision records are absent.  
**Attack/gap story:** An alias such as “latest” could outrank an approved digest; a recovered artifact could be promoted merely because it loads; regression acceptance could be broadened into an all-task or transferable-improvement claim.  
**Detection:** Resolve release aliases against decision receipts; test a numerically later rejected state, renamed files and a valid rollback. Require a task/bundle-scoped approval rather than a filename ordering rule or generic “verified” flag.  
**Harden:** Version decisions independently of training steps, retain rejection reasons and scope, and require explicit authorization for rollback or later requalification. Historical rejection need not permanently bar a separately evaluated future decision.  
**Closure:** §4.5 must supply the actual approval criteria, evaluator identity, decision authority and serving-selection receipt; §4.6 must identify rejected artifacts without inventing their hashes from labels.  

### G10 — MED: stored scalar count can be mistaken for trained or whole-system capacity
**§ref / status:** §§4.2, 4.5 and Appendix A; §§3.4 and 6.1. Stored but inactive heads and seven log-variance tensors not implying seven active losses are DOCUMENTED; the complete role map is absent.  
**Attack/gap story:** The accepted file's inventory might be marketed as the number of parameters updated by a specific objective, active during inference, or resident across all encoders and policy components. These interpretations do not follow from the stored count or F32 header.  
**Detection:** Join each tensor to loader registration, forward/loss consumers, freeze policy, optimizer membership and observed update evidence. Inventory separately loaded context assets without assuming their totals or treating all stored entries as trainable.  
**Harden:** Publish separate stored, trainable, active-by-path and separately loaded component views; mark missing categories unknown. Keep storage dtype distinct from observed compute dtype and avoid inferring hardware memory consumption from one checkpoint file.  
**Closure:** Full Appendix A plus the source-consumer and optimizer maps in §§4.5–4.6/6.1 must support each capacity claim. Removing an “inactive” tensor creates a new artifact identity, even when a scoped parity check passes.  

## Ranked improvements / kickoffs (P0/P1/P2)

### I1 — P0: ship a release-controlled accepted-load gate — closes G1
**§ref / owner:** §§4.2, 4.5–4.6; proposed owner: model-runtime maintainer with release authority.  
**Why it matters:** Protects the already documented accepted/challenger distinction at the actual point of consumption instead of relying on publication language.  
**Kickoff:** Enumerate entry points, constructors and caches; bind a trusted release expectation to the opened immutable object and attach a validation result to the loaded instance. Preserve explicit experimental operation under its own identity.  
**Falsifiable acceptance:** The approved fixture loads; missing/forged receipts, a same-schema payload edit, an unauthorized self-consistent artifact/receipt pair and a post-check path swap fail before accepted inference. No tested failure silently selects another model or reports success.  
**Evidence to retain:** Fixture identities, expected/observed decisions, exact build, invoked path and failure reason. Tests using unavailable historical artifacts stay UNVERIFIED rather than being replaced by invented hashes.  

### I2 — P0: create a semantic model-bundle manifest — closes G2
**§ref / owner:** §§4.5–4.6, with §§3.4 and 4.2; proposed owner: model assembly maintainer.  
**Why it matters:** Makes “the accepted model” reproducible when behavior depends on assets outside the policy weight file.  
**Kickoff:** Define required identities for policy configuration, token semantics, grammar and frozen context components; bind these to the accepted artifact. Link, but do not conflate, the request plan, input scope and executable identity.  
**Falsifiable acceptance:** Swapping any declared semantic dependency invalidates accepted qualification even when dimensions still match; swapping request settings changes the execution identity without falsely claiming new weight bytes. Warm caches obey the same rules as cold loads.  
**Evidence to retain:** One complete approved bundle, mutation matrix, cache-key tests and explicit unknown/missing-field rejection. An input manifest is not evidence that forbidden historical inputs were absent.  

### I3 — P0: reconstruct the smallest defensible lineage graph — closes G3
**§ref / owner:** §§4.5–4.6 and 9.1; proposed owner: training-history custodian.  
**Why it matters:** Preserves the value of genuine prior work while preventing undocumented resets or rejected branches from becoming accepted-model history.  
**Kickoff:** Start at the accepted digest and work backward through available receipts; inventory gen31, native 108-block and A100 recovery records without assuming their order. Record actual weight inheritance, optimizer resets/restoration, corpus transitions and conversion operations.  
**Falsifiable acceptance:** Every claimed accepted-ancestry edge resolves to retained parent/child evidence; unresolved edges are visibly unknown and contribute no asserted exact exposure. Contradictory parents and cycles fail validation; shared ancestors are not counted twice in a branch union.  
**Evidence to retain:** Per-edge evidence locations, artifact digests, local counter meaning and decision status. Publish separate views for accepted ancestry and all experiments; do not erase rejected work from the latter.  

### I4 — P0: make recoverable state an atomic, typed artifact — closes G4
**§ref / owner:** §4.6 and §9.1, using the actual semantics of §§6.1.1 and 6.8 when supplied; proposed owner: training-runtime maintainer.  
**Why it matters:** Distinguishes an exact resumable state from a legitimate weights-only restart and makes recovery claims testable.  
**Kickoff:** Bind weights, optimizer slots, effective configuration, applicable scheduler/random state, sampler cursor, corpus identity and update boundary in a committed manifest. Include accumulation or precision-scaler state only when the actual execution uses it; otherwise declare it inapplicable.  
**Falsifiable acceptance:** Crash injection never admits a partial or cross-generation state. A controlled interrupted/uninterrupted fixture resumes the same data order and next update under declared determinism; missing state fails exact-resume classification. Cross-device equivalence uses a separately justified criterion, not an invented tolerance borrowed from inference.  
**Evidence to retain:** State-membership checks, parameter-slot bindings, recovered event identity and observed continuation outcome. This proposed contract does not substitute invented names for the four omitted §6.8 commands.  

### I5 — P0: reconcile exposure from availability through committed updates — closes G5
**§ref / owner:** §9.1; proposed owner: training telemetry maintainer, reviewed by the lineage custodian.  
**Why it matters:** Converts training-scale statements into reproducible accounting without requiring a fabricated complete history.  
**Kickoff:** Link corpus snapshots to sampled record/program identities, effective token masks, minibatches and optimizer outcomes. Namespace events by run and reset block, record commit/replay status, and version deduplication rules; preserve exact unknown intervals.  
**Falsifiable acceptance:** Synthetic ledger fixtures distinguish repeated exposure from unique programs, attempted from committed updates, and accumulation from steps. Duplicate event ingestion leaves totals unchanged; incomplete journals yield explicit partial/unknown totals; rejected-branch exposure never enters accepted ancestry without a valid inheritance path.  
**Evidence to retain:** Reconciliation output and contributing event identities for each reported unit. A materialized corpus establishes availability only; telemetry overhead should be measured before assigning it an operational budget.  

### I6 — P1: generate and verify the entire tensor inventory from one subject — closes G6
**§ref / owner:** §§4.2, 4.5 and Appendix A; proposed owner: artifact tooling maintainer.  
**Why it matters:** Gives an independent reviewer a direct way to reproduce the inventory instead of trusting synchronized-looking prose and JSON.  
**Kickoff:** Authenticate the approved asset, parse its complete inventory with a format-aware validator, reject malformed/duplicate declarations, validate payload bounds, and emit the table plus machine-readable rows from the same run. Compare every row to the model's required schema.  
**Falsifiable acceptance:** The exact accepted artifact reproduces the declared 136 tensors and 27,796,494 stored scalar count with declared F32 storage, or produces a blocking discrepancy. Missing/extra tensors, wrong dtype/shape, truncated payload and malformed ranges fail; rank-zero scalar and zero-element test tensors are covered without inventing accepted rows.  
**Evidence to retain:** Full header/inventory, input digest, extraction-tool identity and row-level diff. Treat a total-count match as necessary but insufficient; keep optional functional equivalence separate from exact artifact identity.  

### I7 — P1: retain a real training-update witness chain — closes G7
**§ref / owner:** §4.6 and §9 historical evidence; proposed owner: training execution maintainer.  
**Why it matters:** Connects a real optimizer action to the saved descendant, rather than treating a new filename or a generic GPU check as learning evidence.  
**Kickoff:** For a bounded, explicitly authorized real-corpus run, record the parent bundle, sampled records, active optimizer configuration, actual runner/device, tensor-level update summary and reloaded child. Where original A100 recovery receipts exist, attach them rather than retroactively substituting a new experiment.  
**Falsifiable acceptance:** A claimed updated child joins to committed optimizer events and reloads as the intended descendant; a reserialization-only control does not count as a parameter update. A modern replay qualifies only its own hardware/run and cannot close an undocumented historical A100 event.  
**Evidence to retain:** Raw update/reload results and execution identity. Changed parameters establish mutation under that run, not transferable improvement or the causal value of every changed tensor.  

### I8 — P1: make publication claims immutable joins over evidence — closes G8
**§ref / owner:** §9 and §12.1, with §§4.5–4.6; proposed owner: publication/evidence maintainer.  
**Why it matters:** Lets the document evolve without rewriting which model or build produced an older result.  
**Kickoff:** Bind each claim or figure point to its original measurement artifact, subject weights, build, run, units and evidence status. Generate publication-facing summaries from these joins. SLSA's distinction between produced subjects, resolved dependencies and builder identity is a useful design reference, not proof of existing compliance.[^provenance]  
**Falsifiable acceptance:** Text-only publication changes leave historical measurement identities unchanged; missing evidence, subject mismatches and unauthorized builder attestations block the affected qualified claim. New corrections append an explicit superseding record instead of overwriting history.  
**Evidence to retain:** Regeneration diff, claim-to-subject table and verifier output. Builder/signature checks authenticate the claimed origin only within the declared trust model; they do not make a reported experiment scientifically sufficient.  

### I9 — P1: version promotion decisions independently of step labels — closes G9
**§ref / owner:** §§4.5–4.6; proposed owner: release decision authority, separate from the checkpoint writer.  
**Why it matters:** Preserves accepted service behavior during recovery, later training and rollback without assuming that larger step numbers are better.  
**Kickoff:** Record the subject digest/bundle, task scope, evaluation protocol, decision basis, authority and activation/revocation history. Resolve human-friendly names through this registry and record the resolved subject in each consuming run.  
**Falsifiable acceptance:** A later-numbered rejected or unapproved artifact cannot become accepted via an alias or rename; an explicitly approved rollback can. Regression-scoped approval cannot authorize an unrelated task, and successful loading cannot create promotion or transfer-lift evidence.  
**Evidence to retain:** Approved/rejected decision fixtures, alias resolution traces and rollback tests. Where the historical rejection rationale is unavailable, preserve that limitation rather than inventing a failed metric.  

### I10 — P2: add role-aware, component-scoped capacity accounting — closes G10
**§ref / owner:** §§4.5–4.6 and Appendix A; dependency ownership in §§3.4 and 6.1.  
**Why it matters:** Makes the inventory useful for architecture, IP attribution and operational planning without confusing stored state with learned or executed capacity.  
**Kickoff:** Add per-tensor role, component, declared consumers, freeze state, optimizer membership and measured activity under named fixtures. Keep separately loaded encoder assets outside the policy-file subtotal and expose unknown component totals explicitly.  
**Falsifiable acceptance:** Stored totals reconcile to I6; training membership reconciles to I4/I7; claimed inactive tensors show no effect under declared perturbation fixtures, with coverage limits stated. A gradient or parameter delta alone is not reported as proof of useful contribution.  
**Evidence to retain:** Role map, fixture coverage and unresolved consumers. Any stripped or converted deployment artifact receives a new identity, explicit parent transformation and appropriate qualification; no memory or speed gain is assumed in advance.  

## Evidence gaps / UNVERIFIED load-bearing claims

| Claim or subject | What is available here | What would close it / required § |
|---|---|---|
| Accepted lineage B, step 60,000 | DOCUMENTED in §4 and §4.2; detailed §4.5 body absent | Full §4.5 approval record, accepted bytes and release-controlled digest binding. |
| Rejected step 69,000 cannot replace acceptance | DOCUMENTED prohibition, not exercised | Exact rejected subject, decision basis and negative load/alias tests in §§4.2, 4.6. |
| gen31 genealogy | Name appears in the assigned focus, not an expanded evidence record | Full §4.6 and §9 historical series identifying the term, artifacts and actual ancestry. |
| Native 108-block history | Focus label only; block meaning, membership and update totals absent | §4.6/§9 block ledger, reset boundaries and parent/checkpoint mapping; no implied architecture depth. |
| A100 recovery lineage | Requested focus plus §9.1's generic recovery warning | §4.6 recovery/state manifests and §9 device-bound update receipts; no assumed relation to gen31. |
| 136 tensors; 27,796,494 parameters; F32 | DOCUMENTED totals; truncated Appendix A | Complete Appendix A and `data/tensors.json` reproduced from the authenticated accepted payload. |
| Visible inventory arithmetic | CHECKED HERE: 34 complete rows, totaling 8,669,184 scalar entries; every visible shape product matches its printed count | This closes only transcription arithmetic for those rows, not asset identity, dtype bytes or the omitted remainder. |
| 256×2 versus accepted 512×6 | DOCUMENTED in §4.2 as distinct lineages | §4.6 complete configuration and per-artifact inventory/transition records; no partial-load equivalence assumed. |
| Exact continuation and corpus transitions | §9.1 warns against collapsing them; §§6.1.1/6.8 details omitted | Full continuation contracts plus atomic state manifests and sampling/update records. |
| Historical exposure totals | Unit distinctions are DOCUMENTED; event totals absent | §9.1 reconciled ledgers with coverage; irrecoverable history remains unknown. |
| Stored versus active/trained capacity | Inactive heads and non-active loss-state warning are DOCUMENTED | Appendix A consumer/optimizer role map, with scoped tests from §§4.5–4.6 and 6.1. |
| Publication preserves original evidence identity | Explicit promise in preface and §12 | Retained manifest, original §9 figure/data artifacts and regeneration checks under §12.1. |

## Fail-open / silent-lie paths (if any in shard)

These are falsifiable paths to test, not observed incidents. The documented safeguards may already block some; their enforcement evidence was not supplied.

| Path | Silent false conclusion | Required closed behavior / link |
|---|---|---|
| Candidate controls both weights and its “expected” receipt | “Digest matches, therefore accepted.” | Trusted release expectation independent of the candidate; reject mismatches before use — G1/I1 (§4.5). |
| Valid weights paired with changed vocabulary/context assets or stale cache | “Same checkpoint, same proposer.” | Check the approved semantic bundle and resolved cache identity — G2/I2 (§§4.5–4.6). |
| Local step labels stitched across resets/forks | “One continuous accepted training run.” | Receipt-backed ancestry and explicit discontinuities — G3/I3 (§9.1). |
| Weights-only or torn restore labelled exact continuation | “The original training trajectory resumed.” | Atomic state membership plus explicit reset classification — G4/I4 (§4.6). |
| Catalog size, replayed records or rejected work counted as accepted exposure | “This many unique programs trained the accepted model.” | Unit-preserving, branch-aware reconciled ledger — G5/I5 (§9.1). |
| Header/table or serialized-file changes treated as learned tensor changes | “Full model verified; training progressed.” | Payload-bound inventory and optimizer/update witnesses — G6–G7/I6–I7 (Appendix A; §9). |
| Current publication commit or inference device retroactively labels history | “These training results belong to the current accepted build.” | Immutable subject/build/run identities and status classes — G7–G8/I7–I8 (§4.6; §9). |
| Highest step, generic “verified” flag or policy-only parameter count drives claims | “Newest is accepted; all tasks qualified; whole system this size.” | Scoped decisions and component/role-specific capacity — G9–G10/I9–I10 (§§4.5–4.6; Appendix A). |

## Cross-links to other sections the owner must not ignore

| Dependency | Required identity handoff, not a separate audit |
|---|---|
| §§2.1, 2.5–2.6 | Each corpus/ancestor identity must retain input scope and family exclusions; a perfect checkpoint hash cannot prove uncontaminated training. |
| §§3.4–3.6 | Frozen encoder/projection assets and context conventions belong to the bundle boundary; their internal architecture remains outside this review. |
| §§4.2, 4.4 | Exact loading and token/grammar semantics must agree with the accepted bundle; proposal quality is not inferred from inventory equality. |
| §§6.1.1, 6.3, 6.7–6.8 | Obtain optimizer/reset semantics, loader/cache behavior and actual continuation commands before declaring ancestry exposure or exact resume. |
| §§7.1.2, 7.5–7.6 | Comparison identities and decision records must refer to the same subjects; promotion or parameter mutation cannot substitute for measured transfer. |
| §§8.1.1–8.2 | Preserve the policy/bundle identity in private discovery lineage; do not confuse generator weights with the independently fitted inference artifact. |
| §§10.1–10.2, 12.1 | Bind qualification to its exact model, build, device and evidence scope; do not transfer inference tolerances or CPU evidence to historical GPU training. |

## Concerns / residual risk

**Decision recommendation:** Prioritize I1–I5 before expanding qualified claims about accepted identity, continuous recovery or training scale. I6–I9 make those claims inspectable; I10 strengthens capacity attribution. This is a proposed assurance gate, not a finding that the existing model is unusable.  
**Historical limits:** Prospective instrumentation cannot recreate missing historical sampler, optimizer or promotion records. Reproduce what can be reproduced, preserve documentary evidence as documentary evidence, and publish unknowns where the record stops (§4.6; §9.1).  
**Scientific limit:** Integrity, correct resumption and changed weights do not establish useful learning, superiority or transferable improvement. Those require the separately bound comparisons in §7; this shard supplies their subject identity, not their outcome.  
**Trust and scope:** Signed receipts can preserve false producer assertions; negative tests cover specified paths, not all possible behavior. Protect trust roots, separate checkpoint production from approval, and retain exact consumer coverage. Inventory counts also do not settle ownership, novelty or freedom to operate (§1.1).  
**Privacy and feasibility:** Exposure receipts should use controlled identifiers and restricted underlying evidence rather than publishing private rows. Exact replay requirements must declare hardware/software assumptions; weaker equivalence claims need an explicit tested scope, not a convenient tolerance.  
**Existing strengths:** The pinned digest, explicit lineage distinction, inventory publication and accounting caveats provide concrete starting points; missing extract content is not proof those mechanisms are absent from the product (§§4.2, 4.5–4.6, 9.1; Appendix A).  

### External methodological references — not product evidence

[^format]: Hugging Face, *Safetensors — Metadata Parsing*, metadata/header retrieval and tensor dtype/shape/offset example; consulted September 25, 2026. `https://huggingface.co/docs/safetensors/metadata_parsing`
[^resume]: PyTorch, *Saving and Loading Models*, “Saving & Loading a General Checkpoint for Inference and/or Resuming Training”; page updated June 26, 2025, consulted September 25, 2026. `https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html`
[^provenance]: SLSA v1.2, *Build: Provenance*, Model, RunDetails and Builder sections; consulted September 25, 2026. Used as an attestation design reference, not a claim of SLSA certification. `https://slsa.dev/spec/v1.2/build-provenance`
