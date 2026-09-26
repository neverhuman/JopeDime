<!-- from-tar tab=7 url=https://chatgpt.com/c/6ab70302-eebc-83ea-9056-ddd8287d97f2 sha256=ce56dd3a82f3ac040878d0bd3e796abd15ce57e73d25201a2b28cc569eecab6b chars=41452 -->

# Fitted export and deployment: adversarial review and hardening plan

**Artifact notes:** Batch tab 7; shard 7 of 10. Scope: §8, “Executable code without an LLM”; other sections appear only where they constrain export/deployment.  
**Evidence boundary:** User-supplied structured extract; monograph cutoff `2026-09-25T19:31:50Z`; stated publication source `3d71fce041d1dd1173ed98160f7d986bb426ba6f`.  
**Review method:** No product source tree, fitted artifact, support registry, private receipt, binary, or runtime journal was inspected. No product test, compilation, or deployment was executed. Section references identify supplied evidence, not independent source verification.  
**Labels:** DOCUMENTED means stated in the extract; UNVERIFIED means the implementation or execution evidence needed to close the claim is unavailable here. Attack stories are testable hypotheses, not discovered exploits. Severity ranks consequences if realized, not exploit probability.  
**Decision:** Preserve the documented fitted-export correction. Require identity, numerical, dependency, and release-boundary evidence for each advertised deployment profile before expanding claims. All proposed schemas, states, tests, and ownership assignments below are recommendations, not existing product APIs.  

## Executive hits

- **Preserve a real architectural improvement:** §8.1 says the current study uses `export_synthesized` on the original selected fitted candidate, not the legacy recipe-refit emitter. Do not reopen this as an already-proven current refit defect; make the corrected boundary a regression invariant.
- **Highest-impact evidence gap:** A private receipt that describes the right predictor does not itself establish that the shipping bytes are that predictor. Selected-state substitution and post-qualification artifact replacement need separate negative tests and deployment enforcement (§§8.1.1, 8.2; G01–G02).
- **Freeze the complete inference subject:** Turn the explicit input stage, learned preprocessing, predictor state, and output semantics into a versioned, immutable inference capsule; keep original proposal identity separate from executable adaptation identity (§8.2; K01, K03).
- **Parity remains bounded evidence:** The export comparator, adversarial inputs, expected output cardinality, and independent native oracle are not supplied. The `3e-4` encoder/policy tolerance in §10.2 is not an established export tolerance (§§8.1, 12.1; G04).
- **Capability claims need narrowing:** §11.1 explicitly rejects broad all-head/zero-Python conclusions. The trimmed C++17 host boundary does not establish whether Python is needed for qualification, inference, or both (§8.1.1.1; G05–G06).
- **Make support executable:** Generate the Python/Rust/C++17 support matrix from composition-aware qualification records; preserve the selected predictor on unsupported export and report the actual runtime dependency profile (§§8.1.1, 8 support-matrix heading; K06–K07).
- **Harden the delivery boundary:** Build untrusted generated artifacts without secrets, authenticate the resulting package and qualification subjects outside that sandbox, and verify them at deployment. A stored command or checksum alone is not the proposed authorization policy (§8.1.1; K02, K05).
- **Strengthen the core IP with reusable evidence:** Pair private fitted-state replays with public synthetic conformance fixtures, then add bounded translation validation for supported lowerings. This makes faithful export independently challengeable without publishing private search history (§§8.2, 8 worked-examples heading; K09–K10).

## Ranked IP gaps (CRITICAL/HIGH/MED/LOW)

### G01 — CRITICAL — Same recipe, different fitted predictor
**§ref / status:** §§8.1, 8.1.1, 8.2; corrected fitted export and winner preservation are DOCUMENTED; complete subject binding and no-refit enforcement are UNVERIFIED.  

**Attack/gap story:** A loader, serializer, compatibility branch, or fallback could retain recipe identity while changing medians, category indices, coefficients, or other learned state. Alternatively, export could refit after selection. Either yields a different inference subject even when familiar test rows happen to produce similar outputs. The accepted proposal-policy weight hash cannot substitute for the selected predictor's fitted-state identity.  

**Detection:** Construct two permitted fitted subjects with the same executable recipe but different learned state. Swap their state or receipt while retaining the recipe identifier, and remove a required state field; require rejection independently of parity, never default initialization. Deny access to training labels and instrument fitting entry points during emission and prediction; any fitting call fails the test.  
**Harden:** Seal the original proposal, inserted input stage, executable recipe, schema, and complete fitted inference state as distinct but linked objects. Export only a read-only fitted subject; retain the selected native subject on unsupported export. Qualify against native predictions frozen before lowering, not a reference reconstructed from emitted state. See K01.  

### G02 — CRITICAL — Qualified artifact is not necessarily the deployed artifact
**§ref / status:** §§8.1.1, 12.1; backend files, compilation commands, and native parity are DOCUMENTED receipt contents; mandatory release/deployment verification is UNVERIFIED.  

**Attack/gap story:** A successful source emission or compilation could be promoted without numerical qualification. A stale executable, cached prediction file, changed shared library, or replacement package could inherit an earlier pass. Python success could also mask Rust/C++ failure if the request collapses all backend states into one success flag.  

**Detection:** Inject compile failure, timeout, skipped qualification, stale cache entries, and changes to executable/state/wrapper bytes after qualification. Replay a valid receipt against a different backend or runtime profile. Every required backend must block release on missing or mismatched evidence; optional failures must remain visible as partial results.  
**Harden:** Bind qualification to the complete shipping subject and runtime profile, then authenticate and verify that binding at installation/load. Make release a separate explicit state from source-ready, runtime-ready, and numerically qualified. SLSA's verification model likewise checks trusted identity, signature, and actual artifact digest; it does not replace numerical validation [E1]. See K02.  

### G03 — HIGH — Preprocessing and output semantics are an underspecified ABI
**§ref / status:** §8.2; the input recipe and fitted dictionaries/medians are DOCUMENTED; the complete raw-input/output contract is UNVERIFIED.  

**Attack/gap story:** The same width can conceal swapped columns, different category indices, omitted missing indicators, changed type coercion, or inconsistent handling of unseen categories. An emitted predictor can faithfully apply coefficients to the wrong feature representation. Where supported, inverse target transforms and class ordering also belong to the inference subject; this is not evidence that classification export is currently supported.  

**Detection:** Exercise reordered/duplicate/missing/extra columns, numeric-versus-string categories, empty strings versus missing values, all-missing numeric columns, unseen categories, and cases at/beyond the stated `max_onehot=256` boundary. Compare preprocessing outputs before final predictions. Test output units/order and native buffer shape/stride rejection wherever those interfaces exist.  
**Harden:** Specify named and positional input modes, type/missing-value conventions, learned category-to-index mappings, feature order, output semantics, and explicit refusal behavior. Invalid inputs must not silently truncate, pad, infer a new vocabulary, or recompute medians. Preserve semantics when canonicalizing dictionaries. See K03.  

### G04 — HIGH — Numerical qualification can pass the wrong comparison
**§ref / status:** §§8.1, 8.1.1, 12.1; numerical qualification is DOCUMENTED; comparator logic, export tolerances, fixture coverage, and oracle independence are UNVERIFIED.  

**Attack/gap story:** Comparing an empty or truncated prediction intersection can miss failures; unchecked nonfinite values can defeat a simplistic threshold test. Comparing only preprocessed inputs skips raw-input lowering. A native reference reconstructed through the same faulty exporter can agree with a wrong backend. Familiar feature rows need not exercise unseen-category or numerical boundary behavior.  

**Detection:** Mutate output length/order, insert nonfinite outputs, alter a fitted constant, remove a preprocessing stage, and poison expected-output caches. Require the checker to reject these cases. Exercise small/large-scale inputs, missingness patterns, and supported decision boundaries; use independent hand-checkable fixtures as well as the selected native subject.  
**Harden:** Validate successful execution, exact output schema/cardinality, nonempty qualification coverage, row correspondence, and explicit finite/exceptional-value policy before error comparison. Freeze per-profile numerical criteria before results; never borrow §10.2's encoder tolerance. Bind compiler floating-point settings because they can change semantics [E3]. See K04.  

### G05 — HIGH — An operator allowlist is not a composition support proof
**§ref / status:** §§8.1, 8 “Dated support matrix and reproducible worked examples,” 11.1; allowlisted exports are DOCUMENTED; the dated matrix and supported compositions are trimmed/UNVERIFIED.  

**Attack/gap story:** A backend might support a transformation and a terminal separately but not their state layout, chaining, dtype, or resource requirements together. A permissive default may silently drop an unsupported stage. Treating an external comparison adapter as an emitter or substituting a simpler winner would overstate export coverage; the latter violates the stated contract.  

**Detection:** Enumerate the claimed compositions, fitted-state versions, targets, and relevant limits. Insert an unsupported node/state version inside an otherwise valid chain. Require explicit refusal with the unchanged selected-subject identity; ensure a comparison-only external adapter never produces an export-supported status.  
**Harden:** Make support a versioned predicate over the entire fitted graph and deployment profile, not merely a language label or terminal name. Unknown means unqualified, not supported. If exportability is imposed before search, disclose it as a different selection protocol rather than a hidden post-selection replacement. See K06.  

### G06 — HIGH — “Without an LLM” can silently become “standalone native”
**§ref / status:** §8.1.1.1 is trimmed; §11.1 DOCUMENTS C++ gates and a Python host and says broad zero-Python/all-head claims remain unsupported.  

**Attack/gap story:** C++17 prediction code may still rely on Python for input preparation, model loading, orchestration, or qualification; the extract does not identify which. Conversely, a Python qualification harness does not establish that serving needs Python. No-LLM generation, no-LLM inference, no training assets, no Python, and no external services are different claims.  

**Detection:** Trace the actual entry point in a minimal image for each declared profile. Deny undeclared network access and remove training/search assets; remove Python only when testing a Python-free claim. Record child processes, imports, loaded libraries, and file reads during generation, qualification, and serving separately.  
**Harden:** Publish phase-specific dependency manifests and scoped claims. For a native-only profile, qualify raw-input-to-output behavior through the actual native serving entry point, not only a Python-preprocessed matrix. Do not count XGB/LGBM/CatBoost/sklearn subprocess comparisons as Rust/C++ emission evidence (§11.1). See K07.  

### G07 — HIGH — Fitted data crosses a code-building trust boundary
**§ref / status:** §§8.1.1, 8.2; fitted dictionaries and compilation commands are DOCUMENTED; safe literal lowering, sandboxing, trusted build identity, and dependency closure are UNVERIFIED.  

**Attack/gap story:** Dataset-derived category names or identifiers may reach generated literals, filenames, or command arguments despite a grammar-restricted recipe. Unsafe interpolation could turn data into code or shell syntax. Separately, a changed compiler, build hook, dependency, or local environment could produce a different executable while preserving the recipe and source narrative.  

**Detection:** Fuzz permitted strings, delimiters, Unicode, long values, and malformed state lengths; assert preservation as data or explicit rejection. Probe path containment, ambient compiler substitution, dependency drift, and attempted secret/network access in disposable builders. Retain the exact failing input and build subject.  
**Harden:** Use backend-specific literal encoders and structured command arguments; confine output paths and bound allocations. Build without credentials or signing keys. Authenticate provenance outside the untrusted compile/run sandbox, and require deployment verification. Record resolved inputs, but do not infer complete closure from provenance field presence: SLSA itself treats dependency completeness as best effort [E2]. See K05.  

### G08 — HIGH — Private discovery separation does not establish safe public disclosure
**§ref / status:** §8.1.1 DOCUMENTS exclusion of the private search envelope from public source/input dictionaries; field-level disclosure controls and package/log scans are UNVERIFIED.  

**Attack/gap story:** Even with the search envelope removed, fitted categorical dictionaries may contain sensitive tokens. Packaging could also include private receipts, training paths, diagnostic rows, debug symbols, or environment data. Arbitrarily redacting a dictionary to solve disclosure risk can itself change inference semantics; absence of an LLM does not settle either issue.  

**Detection:** Add synthetic canaries to private-only discovery metadata and builder environment; inspect public source, state, wrappers, symbols, logs, and archives for leakage. Separately classify inference-essential dictionary content rather than assuming it is safe because it is public-state material. Test that disclosure rejection cannot trigger silent state modification.  
**Harden:** Use field-level public/private schemas and an explicit disclosure decision for inference-essential state. Release an authenticated minimal public attestation while retaining detailed private replay evidence under controlled access. Omit nonessential private material; refuse an unsafe export rather than silently changing the fitted predictor. See K09.  

### G09 — MED — Deterministic trace lacks a declared reproducibility scope
**§ref / status:** §8.2, “Deterministic trace”; most trace details are trimmed/UNVERIFIED.  

**Attack/gap story:** Identical fitted subjects could emit different source through map iteration, float formatting, paths, or timestamps. Identical source can also yield different binaries under a different toolchain/profile. Collapsing semantic equivalence, source-byte determinism, and reproducible binaries into one “deterministic” badge obscures which property was demonstrated.  

**Detection:** Re-emit a frozen subject in clean processes while varying irrelevant directory, locale, and insertion-order details. Compare semantic identities and normalized source bytes separately. Rebuild within a pinned target/toolchain environment for any claimed binary reproducibility; do not require cross-architecture binaries to match.  
**Harden:** Version serialization and rendering, preserve category-index associations, use lossless fitted-constant encoding, and separate event metadata from deterministic content. State exactly which dependencies and environment attributes define each reproducibility claim; requalify when relevant attributes change. See K08.  

### G10 — MED — Worked-example evidence can be mistaken for present coverage
**§ref / status:** §8.1 explicitly distinguishes historical pending-integration wording from current source; §8's dated support/worked-example content is trimmed, and §12.1 limits parity to per-artifact qualification.  

**Attack/gap story:** A historical example, current source-path existence, and an unrelated backend pass can be assembled into an apparent current operational proof. A private example also cannot independently support unseen language/task combinations. The supplied extract is insufficient to establish which selected fitted artifacts were actually replayed.  

**Detection:** Reconcile every advertised backend/example with exact selection, fitted subject, emitter source, build, runtime profile, fixtures, command outcome, and qualification identities. Treat missing links as UNVERIFIED even when filenames look plausible. Require a fresh-process replay by an authorized reviewer.  
**Harden:** Maintain an evidence index with separate dated source and execution statuses. Publish synthetic examples for external scrutiny; retain private fitted-state traces for authorized replay. Keep historical observations attached to their original identities rather than relabeling them with the publication source. See K09–K10.  

## Ranked improvements / kickoffs (P0/P1/P2)

### K01 — P0 — Establish the immutable inference capsule
**§ref / closes:** §§8.1.1, 8.2; G01 and the state-binding portion of G03. **Why:** It turns “the selected fitted predictor” into a concrete export input that cannot be replaced by an equal-looking recipe.  

**Build:** Define a versioned fitted-state contract covering executable structure, every inference-used transform/terminal state, raw-input schema, output semantics, and validation rules. Preserve original proposal bytes and inserted-stage identity in the private linkage; do not require private search history in the deployable package.  
**Boundary:** A proposed export entry point accepts only this read-only capsule and a backend/profile. Training labels, search workers, fit functions, and the proposal policy are unavailable capabilities. Existing recipe-refit functionality, if retained, has a different product operation and identity.  

**Falsifiable acceptance:** Same-recipe/different-state swaps, missing required state, and unsupported state versions fail verification; serialize/reload preserves state semantics; emission and prediction succeed with all fit entry points disabled. Every successful backend points to the original selected capsule; unsupported export leaves that capsule unchanged.  
**Owner / sequence:** Fitted-export maintainer with the fitted-state owner; establish the subject contract before broadening backend coverage. Review the monograph-named `fitted_export.rs` boundary without assuming its present schema.  

### K02 — P0 — Enforce backend-specific qualification and release states
**§ref / closes:** §§8.1, 8.1.1, 12.1; G02. **Why:** It closes the handoff from a documented parity result to the actual package allowed into service.  

**Build:** Proposed states distinguish not-requested, unsupported, source-ready, runtime-ready, numerically-qualified, and released, plus explicit failures. Record requested versus optional backends before execution. Python need not have a native compilation phase, but must still have a defined executable runtime subject.  
**Boundary:** Release verifies selected capsule, emitted files, runtime wrapper, binary/state bundle, declared dependency profile, and qualification record. Authenticate both build and qualification subjects; verify installed content through an immutable package or equivalent check-to-use protection. Follow trusted-identity/digest verification principles without claiming a SLSA level [E1].  

**Falsifiable acceptance:** Missing, skipped, failed, expired-by-policy, wrong-target, tampered, or stale evidence cannot produce release. Altering any inference-relevant package file invalidates its prior release binding. A failed required backend blocks the corresponding advertised result; an optional failure remains explicitly partial.  
**Owner / sequence:** Export runner plus release owner; inspect enforcement around the named `fitted_export_run.rs` and `inference.rs` paths. The first deployable claim should carry an exact verified subject, not merely an emission log.  

### K03 — P0 — Compile an explicit input/output contract into every backend
**§ref / closes:** §8.2; G03. **Why:** Cross-language agreement is useful only when all languages consume the same raw features and produce the same semantic outputs.  

**Build:** Treat `(input numeric=median categorical=onehot missing=indicator min_count=1 max_onehot=256)` as the documented recipe, not a complete ABI. Specify its fitted medians, category/index ordering, unknown/missing behavior, column identity, dtype conversions, output units, and applicable target decoding.  
**Boundary:** Validate the schema before computation; version any allowed coercions and positional interfaces. Native interfaces must reject incompatible buffer lengths/layouts rather than accepting same-width but semantically different input. No runtime vocabulary expansion or implicit refit is permitted.  

**Falsifiable acceptance:** Each schema mutation either follows the declared canonicalization rule and matches native predictions or returns the documented error. Every supported backend agrees at preprocessing and final-output boundaries on the adversarial cases in G03, including the one-hot limit boundary.  
**Owner / sequence:** Preprocessing/ABI owner and all backend owners jointly; maintain public synthetic conformance fixtures that do not require private training data. Make ABI changes trigger capsule/profile version changes and renewed qualification.  

### K04 — P0 — Add independent, mutation-tested translation qualification
**§ref / closes:** §§8.1, 8.1.1, 8.2, 12.1; G04. **Why:** A strong checker must demonstrably reject wrong exports rather than only confirm convenient examples.  

**Build:** Capture native outputs from the selected immutable subject before emission. Add independently specified tiny fixtures for supported operations and end-to-end raw-input probes. Compare stage outputs where meaningful so preprocessing, constants, terminal computation, and output decoding can be localized.  
**Boundary:** Pin feature fixture membership, reference outputs, comparator version, backend/profile, and predeclared numerical criteria. Check cardinality/order/nonfinite policy first. Test repeated/interleaved predictions for subject isolation and immutable state; test batching and row-permutation invariance only for operators whose declared semantics guarantee them; parity does not require outer scoring labels or establish accuracy.  

**Falsifiable acceptance:** Every deliberately corrupted required fixture is rejected, including empty/truncated output, wrong ordering, altered constants, lost missing indicators, and poisoned reference caches. Valid emitted subjects pass the predeclared per-profile contract; failures cannot be fixed by loosening tolerances after observing results.  
**Owner / sequence:** Qualification owner independent of backend implementation review. Wire retained pass/fail evidence into K02; keep compiler flags and arithmetic behavior in scope because LLVM documents semantics-changing fast-math options [E3].  

### K05 — P0 — Isolate generated-code builds and authenticate the result
**§ref / closes:** §§8.1.1, 8.2; G07 and the supply-chain portion of G02. **Why:** The exporter handles fitted data and creates runnable code; the build and prediction tests must not gain ambient authority.  

**Build:** Use fixed backend templates or typed lowering, validated literal encoders, structured argument lists, confined paths, pinned dependencies/toolchain, and bounded resources. Provision declared build inputs before executing in an isolated, credential-free, network-denied workspace; audit any necessary exception explicitly.  
**Boundary:** Keep signing/authorization outside the generated-code sandbox. Bind build input/output identities and qualification records to the shipping subject, and verify them at deployment. Inventory compilers, templates, lockfiles, runtime libraries, and build hooks; actual dependency closure needs evidence beyond a populated provenance record [E2].  

**Falsifiable acceptance:** Hostile-but-valid data remains data; malformed state rejects without out-of-bounds behavior. Build attempts cannot read seeded credentials or write outside permitted paths. Compiler/dependency substitutions invalidate the profile or require new qualification; a copied receipt cannot authorize altered output.  
**Owner / sequence:** Build/release security owner plus emitter owners. Exercise these controls before accepting less-trusted fitted artifacts or allowing third-party backend extensions; do not give generated programs access to the attestation signer.  

### K06 — P1 — Replace narrative support claims with a tested capability registry
**§ref / closes:** §8 support-matrix heading, §§8.1, 8.1.1, 11.1; G05. **Why:** A precise supported subset can grow safely and is more credible than a universal language badge.  

**Build:** Registry entries describe whole supported compositions and constraints: task/output kind, input ABI, operators, terminal state version, numeric representation, backend, target, and host/runtime profile. Generate the dated human-readable matrix from this registry and linked qualification evidence.  
**Boundary:** Distinguish implemented, fixture-qualified, selected-artifact-qualified, unsupported, and unverified. A comparison-only external adapter is not an emitter. Support checks may reject the selected subject, but cannot choose another winner or erase the refusal from the request record.  

**Falsifiable acceptance:** Every published supported combination has retained runnable conformance evidence and a declared per-artifact gate. Unknown combinations and state versions refuse explicitly. A test that disables a backend or removes required evidence removes its qualification claim without changing the selected model.  
**Owner / sequence:** Export capability owner; prioritize expansion from actual refused selected subjects, not invented coverage metrics. Any pre-search deployment constraint must become part of a separately declared search protocol.  

### K07 — P1 — Publish and test phase-specific dependency profiles
**§ref / closes:** §8.1.1.1, §11.1; G06. **Why:** It makes “without an LLM” testable while avoiding an unsupported leap to “zero Python everywhere.”  

**Build:** Describe emission, compilation, qualification, and serving independently. For each profile, identify the entry point, required files/libraries, interpreters, child processes, and network policy. Include raw-input preprocessing and postprocessing, not just the terminal predictor.  
**Boundary:** Qualify Python exports in their declared Python runtime. Claim native-only Rust/C++17 serving only after a native raw-input entry point passes without Python; a Python-based test harness can be disclosed separately. No-LLM tests deny model-service access and omit generation/training assets not needed by inference.  

**Falsifiable acceptance:** A fresh instance with only declared serving inputs performs the required predictions. Removing undeclared tooling causes no hidden fallback or download. Unexpected file/process/network dependencies fail the profile; retained traces distinguish a qualification harness from a production dependency.  
**Owner / sequence:** Deployment/packaging owner with the C++ host owner. Publish the narrowest demonstrated profile immediately; expand wording only when the corresponding isolation test passes.  

### K08 — P1 — Make deterministic emission and reproducible builds separate contracts
**§ref / closes:** §8.2; G09. **Why:** Stable, well-scoped identities simplify diagnosis, release verification, and independent replay without demanding impossible cross-target byte equality.  

**Build:** Canonically serialize the fitted subject and version each renderer. Preserve learned mapping indices while stabilizing map traversal. Round-trip fitted constants without precision loss; put timestamps, temporary paths, and journal metadata outside deterministic generated content.  
**Boundary:** Record source-byte determinism separately from prediction equivalence. Bind any binary-reproducibility claim to an explicit target/toolchain/dependency environment, including arithmetic flags. Do not treat a different binary hash alone as a numerical failure, or matching source hashes as proof of runtime parity.  

**Falsifiable acceptance:** Re-emission in clean directories/processes with irrelevant environment changes yields the same declared source content and semantic subject. Rebuilds within any advertised reproducible-build profile match the promised binary identity; otherwise the claim stays at source determinism with separate numerical qualification.  
**Owner / sequence:** Serialization and build owners; add replay fixtures alongside K04. Preserve historical source/execution identities when replay uses a newer publisher or report generator.  

### K09 — P1 — Deliver safe public fixtures and complete private worked replays
**§ref / closes:** §8.1.1 and §8 “Dated support matrix and reproducible worked examples”; G08 and G10. **Why:** This makes external scrutiny possible without confusing public disclosure with private audit completeness.  

**Build:** Public synthetic examples cover supported preprocessing/terminal compositions, expected predictions, deliberate failure cases, and dependency profiles. Private examples retain proposal, adaptation, selected fitted state, native reference, emitted artifacts, build commands/results, and exact qualification identities for authorized replay.  
**Boundary:** Link public attestations and private evidence to the same relevant artifact identities without exporting the search envelope. Apply field-level release rules and scan the entire shipping package/log set. Review inference-essential sensitive dictionaries explicitly; hashing identifiers is not a substitute for a disclosure decision.  

**Falsifiable acceptance:** An authorized reviewer replays each claimed worked example from retained bytes without refitting, reading outer scoring labels, or consulting an LLM. Public fixtures reproduce their scoped claims. Private-only canaries are absent from released material; essential-state disclosure refusals cannot silently alter model semantics.  
**Owner / sequence:** Evidence custodian with deployment/privacy reviewers. Treat unavailable private materials as UNVERIFIED rather than substituting screenshots, stale examples, or a source-existence assertion.  

### K10 — P2 — Add bounded translation validation for the supported fitted language
**§ref / closes:** §§8.1, 8.2; strengthens G01, G03–G05, G09, G10. **Why:** A small, explicit inference language with checked lowerings can make faithful executable export a stronger technical differentiator than language-template breadth alone.  

**Build:** Prototype a declarative fitted-operation representation with input/output semantics, typed state, and versioned lowering rules. Attach structural checks for state preservation, feature order, dimensions, and output decoding; use stage-local equivalence obligations and independent reference fixtures for each admitted lowering.  
**Boundary:** Start with a genuinely supported subset established by K06; no new operator coverage is presumed. A validation certificate identifies exactly which structural or semantic obligations were checked. Floating-point tolerance, unproven library behavior, and unsupported operations remain explicit limits, not hidden assumptions.  

**Falsifiable acceptance:** The validator rejects intentionally miswired feature indices, changed fitted constants, omitted operations, and unsupported semantics. An authorized reviewer can regenerate the validation record and K04 parity evidence from the capsule. A new lowering cannot acquire the existing support claim merely by emitting compilable code.  
**Owner / sequence:** Compiler/IR owner after the P0 contracts stabilize. Promote only measured verification coverage; do not market finite tests or structural certificates as a proof of universal numerical equivalence or patent novelty.  

## Evidence gaps / UNVERIFIED load-bearing claims

| Claim needing closure | Present evidence and limitation | Exact evidence needed / closing section |
|---|---|---|
| Current path exports, rather than refits, the selected subject | §8.1 states the corrected call path; no implementation or execution inspected | Complete §8.1 with retained no-refit call trace, frozen subject identity, and refusal tests for the named export paths |
| Receipt binds every inference-used value | §8.1.1 names major objects; field schema and verification consumers are absent | Full §8.1.1 and §8.2: receipt/capsule schema, state inventory, loader checks, subject-swap tests |
| Exact shipping artifact is qualified | §12.1 explicitly requires per-artifact qualification; deployment gating is not shown | §8.1: release/load enforcement, complete artifact hashes, authenticated build/qualification evidence, tamper replay |
| C++17 host is or is not required during serving | §8.1.1.1 body is trimmed; §11.1 mentions a Python host | Full §8.1.1.1: phase-specific process graph, native entry point, dependency manifest, isolated serving run |
| Specific Py/Rust/C++17 compositions are supported | §8 names an allowlist; the dated support matrix is unavailable | Full §8 “Dated support matrix and reproducible worked examples”: registry version, constraints, positive/negative qualification records |
| Native-versus-emitted numerical parity is adequately checked | Qualification is stated; comparator, fixture scope, and errors are not supplied | §§8.1–8.2: exact comparator, frozen export criteria, input/output counts, per-case results, oracle lineage, mutation tests |
| Emission is deterministic at a stated level | §8.2 heading and a small state description survive trimming | Full §8.2: canonicalization rules, repeated source generation, build profiles, differences and their interpretation |
| Private discovery evidence is excluded safely | §8.1.1 states envelope exclusion; state/log disclosure rules are absent | Full §8.1.1: public/private field schemas, package/log scans, canary results, essential-state disclosure decisions |
| Worked examples establish current runtime coverage | Example bodies and private receipts were not supplied | Full §8 worked-example heading plus §12.1: complete dated replay packets tied to specific fitted and deployed subjects |

## Fail-open / silent-lie paths (if any in shard)

**Status:** The following are conditional paths to test, not observed product vulnerabilities. The extract explicitly prohibits recipe refitting as fitted export and replacing an unsupported winner (§§8.1.1–8.2).  

| Potential path | Silent assertion it would create | Required fail-closed result |
|---|---|---|
| Refit/reconstruct at export; bind recipe but not state | “Exported the selected predictor” | Reject changed subject or fit call; preserve original capsule (G01) |
| Unsupported node silently omitted or simpler candidate selected | “This winner is supported” | Explicit unsupported result with unchanged winner (G05) |
| Source-ready/compile success used as release approval | “Numerically qualified deployment” | Separate qualification and release authorization (G02) |
| Empty/truncated/nonfinite comparison accepted; row order lost | “Native parity passed” | Comparator rejects before numeric acceptance (G04) |
| Python backend passes while required Rust/C++ fails | “All requested targets passed” | Required-target failure blocks its aggregate claim (G02) |
| Cached binary/output or changed wrapper inherits old receipt | “These shipped bytes were tested” | Subject/profile mismatch invalidates qualification (G02, G07) |
| Python preprocessing or service call hidden by a native kernel | “Standalone native/no-LLM deployment” | Phase-specific dependency violation; no fallback (G06) |
| Public package includes private receipt or unapproved dictionary | “Private discovery is excluded; disclosure is safe” | Disclosure refusal without modifying fitted semantics (G08) |

## Cross-links to other sections the owner must not ignore

**§§1.1, 1.7:** Keep deterministic lowering distinct from neural proposal generation, fitting, and promotion. The supplied §1 evidence vocabulary prevents “implemented” from becoming “operationally qualified.” §1.7 itself is trimmed; it supplies no extra deployment proof here.  
**§2.1 and §5.9:** Fitted-export parity can preserve a contaminated or improperly refitted predictor perfectly. Export must inherit fit-scope/refit identity; it cannot certify upstream leakage freedom. §5.9 is trimmed and needs the original refit/held-fold trace.  
**§§3.1, 3.5 and §4.2:** Encoders and accepted proposal weights identify upstream generation components, not the deployable fitted subject. A standalone inference claim needs a dependency test, not an inference that these assets disappear automatically.  
**§§5.1, 5.10:** Preserve original versus executable recipe identity and candidate ancestry at the export handoff. §5.10's trace is trimmed; obtain it to connect a worked export to its real selected candidate, without exposing it publicly by default.  
**§§7.3, 7.6:** Where a task is supported, output/class order is part of the export contract. Keep target-specific requested, unsupported, failed, and qualified outcomes visible; do not derive export reliability only from successful builds. This is an export denominator requirement, not a re-audit of KPI scoring.  
**§§10.1.1, 10.2:** Source-ready, compilation, first prediction, and qualified deployment are different milestones. Python/Rust/C++ milestones are cumulative in the stated sequential study; do not relabel them as independent per-language end-to-end timings. Encoder/policy tolerance is not export parity tolerance.  
**§§11.1, 12.1:** Retain the documented improvement in current fitted export and the equally explicit supported-subset limitation. Reproduction manifests identify evidence bytes; they do not turn missing per-artifact runtime evidence into a pass.  

## Concerns / residual risk

**Release posture:** Block unsupported or unqualified deployment profiles, not the existence of a native selected predictor. The report finds critical unclosed evidence obligations, not confirmed critical implementation bugs; missing trimmed text may already contain some controls.  
**Oracle trust:** Native/emitted agreement is not predictive validity, leakage freedom, or a guarantee that both implementations are correct. Independent synthetic semantics checks reduce common-mode risk; finite tests do not prove behavior for every input (§§2.1, 8.1, 12.1).  
**Platform scope:** A qualified target/runtime profile does not establish other compilers, architectures, library versions, arithmetic flags, or concurrency behavior. Extend support through explicit profile qualification rather than inherited language labels (§8 support-matrix heading; [E3]).  
**Security scope:** Provenance verification still relies on trusted builders/verifiers and their roots of trust [E1]. Generated-code isolation and dependency checks require their own tests; adding a signature cannot validate fitted semantics or repair unsafe disclosure.  
**Change control:** New serializers, ABI coercions, support predicates, or compiler settings may change the inference subject or qualification profile. Do not migrate silently; retain the old subject and require explicit compatibility/requalification decisions (§§8.1.1–8.2).  
**IP scope:** This review identifies technical differentiation and evidence improvements, not patent novelty or freedom to operate, consistent with §1.1. Public emitted state can expose implementation detail even when private discovery history remains excluded; settle that disclosure boundary deliberately (§8.1.1).  
**Immediate owner decision:** Fund K01–K05 as the release boundary, then K06–K09 for honest coverage expansion and replayability. Keep K10 a bounded strengthening experiment whose claims are limited to its demonstrated validation scope.  

### External primary-source method notes

These sources inform proposed compiler/supply-chain checks only; they do not verify JopeDime, alter the monograph cutoff, or establish product compliance. Product findings remain grounded in the supplied section references.
**[E1]** SLSA v1.2, “Build: Verifying artifacts,” especially artifact-subject, signer/builder, and expectation checks: `https://slsa.dev/spec/v1.2/verifying-artifacts`.  
**[E2]** SLSA v1.2, “Build: Provenance,” especially build inputs, output subjects, and best-effort resolved-dependency completeness: `https://slsa.dev/spec/v1.2/build-provenance`.  
**[E3]** LLVM Language Reference, “Floating-Point Semantics” and “Fast-Math Flags,” documenting changes permitted by arithmetic flags, including assumptions about NaNs/infinities: `https://llvm.org/docs/LangRef.html#fast-math-flags`.
