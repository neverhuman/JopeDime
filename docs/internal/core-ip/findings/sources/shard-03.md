<!-- from-tar tab=3 url=https://chatgpt.com/c/6ab70250-145c-83e9-9c66-5fcbaeb984f6 sha256=61fdfcdeedf56528a9fe023f6f3ea556d4a37d05ab1a92caa47e086a2e580141 chars=37536 -->

# Encoders and Hyperion — contract, provenance, and qualification audit

**Artifact notes:** Shard 3 of 10; batch tab 3. Primary scope: §§3.1–3.6. Other sections appear only where they constrain or consume encoder products.
**Evidence basis:** User-supplied structured extract; stated cutoff `2026-09-25T19:31:50Z`; stated publication source `3d71fce041d1dd1173ed98160f7d986bb426ba6f`.
**Limits:** No product source, weights, full chapter bodies, or retained execution artifacts were inspected or run. Section references identify supplied documentary evidence, not independent code verification.
**Reading rule:** DOCUMENTED means stated in the extract; UNVERIFIED means omitted or unsupported by inspected runtime evidence. Proposed names, tests, owners, and gates below are recommendations, not existing implementation claims.
**Decision:** Preserve the accepted encoder semantics, but require enforceable provenance and a versioned semantic context contract before strengthening purity, interchangeability, or encoder-benefit claims.

## Executive hits

- **CRITICAL / documented enforcement gap:** §11.1 says encoder-receipt validation exists but live `eligibility.rs` does not call it. For context-bearing teachers, a receipt type is not an admission barrier; wire the validator into real consumers and prove rejection with otherwise-valid teacher evidence (§§3.3, 3.5).
- **HIGH / unverified fit boundary:** Foundation is target-conditioned and Target encodes target windows; frozen weights do not establish which current labels or rows conditioned an output. Restore the TrainFitted-versus-joint-processing contract and test forbidden-input noninterference (§§3.1, 3.3).
- **HIGH / semantic type collision:** The dataset signature and Target context are both 512 values; raw Foundation and the final proposal context are both 2,048. Length checks cannot distinguish these products, while Hyperion cell surfaces are explicitly not a context ABI (§3.1).
- **P0 / preserve useful architecture separation:** Implement distinct product types, validated deserialization, and one authorized assembly path. Reject cell/logit-surface substitutions, swapped 512-value blocks, and raw-Foundation bypasses instead of adapting them by padding or reshaping (§§3.1, 3.5–3.6; §11.1).
- **P0 / bind the complete assembly:** Pin encoder architectures, active readouts, assets or deterministic construction recipes, preprocessing state, projection, pooling, normalization, and consumer compatibility together. The accepted policy identity alone does not close these dependencies (§§3.2, 3.4; §4.2).
- **P1 / turn examples into a conformance suite:** Extend the retained encoder cases with stage-level fixtures, malformed-input rejection, cache-scope mutations, and backend checks. Preserve the documented `3e-4` limit for the outputs it already covers; do not infer universal equivalence (§§3.2–3.6; §10.2).
- **P2 / prove the context earns its complexity:** Evaluate whether pooling, projection, and each context block improve matched downstream search. Separate diagnostic block perturbations from properly qualified challengers; the extract establishes neither causal encoder value nor an optimal compression choice (§§3.5–3.6; §7.5).
- **Evidence boundary:** The widths in §3.1 are available, but substantive §§3.2–3.6 are trimmed. Exact summary functions, block order, normalization, target-window rules, encoder loading, and runtime enforcement remain UNVERIFIED; absent text is not proof that implementation is absent.

## Ranked IP gaps (CRITICAL/HIGH/MED/LOW)

Severity ranks potential consequence and urgency; evidence status ranks confidence separately. Only G01 is an explicit live-enforcement limitation in the supplied reassessment. Other entries identify missing proof or conditional failure paths, not reproduced defects.

### G01 — CRITICAL: Context provenance can remain advisory at a live consumer
**§refs / status:** §§3.3, 3.5; §11.1, “Reassessment against current source”; §12.1. DOCUMENTED: receipt types/binding validation exist but live eligibility does not invoke them. Actual unauthorized admission is UNVERIFIED.

**Attack/gap story:** A context-bearing teacher could satisfy ordinary score evidence while its context was built from the wrong fit scope, wrong component assembly, or unauthenticated bytes. If the provenance check remains opt-in, evidence at the encoder boundary need not constrain the learning consumer.

**Detection:** Submit otherwise-valid context-bearing records with missing receipts, mismatched scope, altered context bytes, and a valid receipt replayed onto a teacher with a different context or scope identity. Exercise the production admission entrypoint, not just the receipt helper; observe rejection and absence of downstream state mutation.

**Harden:** Make successful validation a mandatory input to each context-bearing consumer, including proposal construction and teacher ingestion. Bind scope, assembly, context bytes, and subject identity; quarantine unknown historical provenance rather than manufacturing a passing receipt. Coordinate HWM/corpus side-effect assertions with their owners.

### G02 — HIGH: TrainFitted, joint processing, and inner-fold conditioning lack a closed contract
**§refs / status:** §§3.1, 3.3; §2.1. Foundation target conditioning and Target-window encoding are DOCUMENTED; actual scope rules and historical-mode handling are UNVERIFIED because §3.3 is trimmed.

**Attack/gap story:** A frozen encoder can still consume forbidden current labels or fit preprocessing jointly on fitting and held rows. A context using all outer-fitting labels can be legitimate for training-side search, yet does not establish fold-pure proposal conditioning for every inner validation fold.

**Detection:** Hold authorized inputs, component identities, candidate ancestry, and randomness fixed; alter only labels forbidden to the tested context consumer. Compare stage outputs and instrument reads. Separately perturb excluded features where the declared mode forbids joint feature conditioning; do not confuse changed scoring labels with changed context outputs.

**Harden:** Specify allowed feature/label membership separately for task-level search, inner-fold conditioning, and final fitting. Isolate historical joint/transductive modes from TrainFitted qualification; use fold-scoped contexts wherever fold-purity is claimed. Frozen pretraining ancestry is a separate provenance requirement, not a substitute for these current-input rules. [R1]

### G03 — HIGH: Width-compatible products can cross semantic boundaries
**§refs / status:** §3.1, “Product and shape contracts”; §§3.5–3.6; §11.1. Product distinctions are DOCUMENTED; a full typed ABI firewall is explicitly not proven.

**Attack/gap story:** Both the pooled dataset signature and Target context have 512 values; swapping their roles preserves length. Raw Foundation has 2,048 values, like the final teaching/proposal context, enabling a projection/assembly bypass if a consumer accepts only an untyped vector. Cell/logit surfaces must not masquerade as V7 row embeddings.

**Detection:** Attempt these substitutions through internal constructors, serialized records, and caches. Include flattening, truncation, and padding attempts involving cell surfaces; no universal cell-surface width is asserted. Check rejection before decoding or teacher acceptance, with a control showing the accepted assembly still works.

**Harden:** Give each semantic product a distinct opaque type and version tag, validate external payloads, and permit only named transformations between types. Labels on arbitrary bytes are insufficient: require producer provenance and the declared transformation chain. Do not use type aliases as the firewall.

### G04 — HIGH: Policy acceptance does not establish accepted encoder assembly
**§refs / status:** §§3.2, 3.4; §4.2; Appendix A. Encoder architecture and component-assembly bodies are trimmed; exact encoder asset identities, active heads, and compatibility checks are UNVERIFIED.

**Attack/gap story:** A different encoder readout, projection construction, or preprocessing configuration could preserve every advertised width while changing the policy's input meaning. The policy loader's documented checks do not, by themselves, establish analogous checks or compatibility binding for these upstream products.

**Detection:** Substitute one same-output-width component or configuration at a time, including an inactive-versus-active readout where the actual inventory permits that test. Confirm whether loading rejects the mismatch before encoding; compare the declared assembly with actual loaded artifacts and construction settings.

**Harden:** Define one accepted assembly manifest covering all producers and transforms, including deterministic construction recipes where no learned asset exists. Validate actual bytes/configuration and bind the manifest to the consumer policy. Treat missing ancestry as unknown, and encoder substitutions as challengers requiring requalification.

### G05 — HIGH: The 2,048-float calculation is not yet a reproducible semantic layout
**§refs / status:** §§3.1, 3.5, 3.6. The widths reconcile as `512 + 1,024 + 512 = 2,048`; ordering, offsets, summary definitions, normalization rules, and edge handling are UNVERIFIED in the supplied extract.

**Attack/gap story:** Wrong block order, per-coordinate versus per-summary flattening, different normalization axes, or projection at the wrong stage could yield a plausible finite vector of the accepted length. Normalizing twice or handling an empty/constant block differently could also change meaning without a shape error.

**Detection:** Use fixtures that make blocks and summary positions distinguishable; retain intermediate products and reconstruct the final context using an independent implementation of the recovered specification. Exercise empty, constant, missing, and non-finite inputs according to the declared domain; verify values and rejection behavior, not only width.

**Harden:** Restore §3.6 as a normative calculation with exact transformation order, layout, precision, masks, summary functions, and numerical boundary rules. Version semantic changes. Do not infer unnamed pooling statistics, an L2 normalizer, or a linear projection from the shape table.

### G06 — HIGH: Cache reuse can detach an encoding from its producing scope
**§refs / status:** §§3.3–3.5; §2.1 identifies cached embeddings as a leakage carrier. Cache keys, storage validation, and device-buffer lifetime rules are UNVERIFIED; §6.7 is an integration dependency only.

**Attack/gap story:** A composite context cached by dataset name, dimensions, or policy identity could survive changes to labels, fitting membership, projection, or feature order. A valid receipt could also be checked against one buffer and later paired with another if ownership or mutation is uncontrolled.

**Detection:** Independently change fitting membership, allowed labels, schema/order, target-window selection, and component identity while keeping superficial dimensions stable. Compare cache hits with clean recomputation; test concurrent requests and serialized cache replay. Verify the consumer uses the exact validated subject.

**Harden:** Key each cached product by its actual dependency closure and bind its content to that key. Hyperion feature-only products need not inherit irrelevant label dependencies, but the assembled target-conditioned context does. Make validated products immutable through use; reject cross-scope replay and stale compatibility identities.

### G07 — HIGH: Row, schema, mask, and target-window alignment are unspecified
**§refs / status:** §§3.1–3.3, 3.6. The accepted row embedding is 128 per row and Target is a frozen target-window encoding; row alignment, window selection, padding, ordering, and task restrictions are UNVERIFIED.

**Attack/gap story:** Feature rows and labels can become misaligned through independent sorting, subsampling, missing-value removal, or window selection while all output widths remain correct. An unspecified feature order or truncated input could change the representation silently; permitted temporal ordering must not be mistaken for an invariance defect.

**Detection:** Build row-identity fixtures, independently reorder labels, exercise missing rows and window boundaries, and vary batch partitioning. Pair joint row permutations with the actual declared ordering contract: demand invariance only for set-semantic operations, and explicit rejection or documented behavior for order-sensitive ones.

**Harden:** Carry aligned row identity, schema identity, masks, and sample/window selection receipts into each producer. Publish supported target/task forms and oversize-input policy. Reject unsupported inputs or represent their handling explicitly; do not infer a window length, sorting convention, or classifier-specific encoding.

### G08 — HIGH: Numerical parity does not certify semantic or execution coverage
**§refs / status:** §3 opening; §§3.2–3.6; §10.2. The CUDA-device refusal and eight retained native encoder cases are DOCUMENTED; their complete fixtures, stage coverage, and execution witnesses are UNVERIFIED here.

**Attack/gap story:** A numerically close result can still be the wrong semantic product or use an unintended execution path. Limited fixtures do not close non-finite behavior, unusual masks, same-width asset substitutions, or backend divergence near downstream ranking boundaries.

**Detection:** Inspect each retained case's actual compared outputs, reference identity, and finite-value checks; rerun without changing its `3e-4` maximum-absolute-error limit. Add mixed-device, missing-asset, non-finite, and stage-level probes; observe real execution placement rather than only constructor device names.

**Harden:** Preserve retained qualification, then add separately declared coverage and consumer-impact tests. Test the documented CPU rejection before asset access. Fail unsupported fallback explicitly; distinguish permitted CPU host work from operations claimed to execute on CUDA. Numerical closeness alone must not confer scope or type validity.

### G09 — MED: Pooling and projection sufficiency is an untested IP-performance claim
**§refs / status:** §§3.1, 3.5–3.6; §7.5. Four summaries per embedding coordinate and the Foundation width reduction are DOCUMENTED; their information retention and causal usefulness are UNVERIFIED.

**Attack/gap story:** The unspecified summaries may discard distinctions that matter for selecting programs, while the other context blocks may or may not compensate. Neither the width calculation nor a well-formed proposal establishes that this combination outperforms simpler conditioning.

**Detection:** First identify the actual summaries. Construct controlled embedding-level collisions, then seek realizable table/task fixtures with similar signatures but different useful programs. Measure the complete context and matched search outcomes; a synthetic aggregator collision alone is not an end-to-end failure.

**Harden:** Run a staged diagnostic and challenger study for pooling, projection, and block contributions. Preserve the accepted assembly as baseline, record compute as well as task outcomes, and retain null results. Make encoder-benefit claims only where the measured comparison supports them.

### G10 — MED: A context can be correct yet ignored, mismatched, or mislabeled at handoff
**§refs / status:** §§3.5–3.6; §§4.1, 6.5. Context-conditioned policy input is DOCUMENTED, but matched teaching/proposal context semantics and effective block use are UNVERIFIED in the trimmed material.

**Attack/gap story:** An older serialized teaching context might differ semantically from the current producer while retaining width. Alternatively, a policy might be insensitive to a block or receive a stale context for another request; successful decoding cannot distinguish these cases.

**Detection:** Trace one authorized request and one context-bearing teacher from producing scope through serialized context to the actual consumer tensor. Perform controlled block perturbations and measure intermediate policy responses as diagnostics, without assuming every perturbation must change the selected program.

**Harden:** Require producer/consumer ABI compatibility at ingestion and inference, bind request identity, and expose block-level diagnostics. Separate dead conditioning, weak incremental value, saturation, and stale-input routing; they require different fixes and none is established by a width check.

## Ranked improvements / kickoffs (P0/P1/P2)

Work below has equal decision weight to the gaps. Owners are proposed responsibility roles, not identified incumbents. Acceptance conditions are proposed tests, not reported passes; follow the existing evidence ladder from source implementation to retained execution and promotion (§1).

### K01 — P0: Make a validated context an admission capability
**§refs / closes:** §§3.3–3.5; §11.1. Closes G01 and supplies the enforcement endpoint for G02/G06. **Owner:** context-boundary maintainer with learning-admission maintainer.

**Why it matters:** This converts provenance from documentation into a restriction on what the actual consumer can accept, without redesigning the encoders or claiming new model capability.

**Kickoff:** Inventory context-bearing entrypoints; require a validated object bound to producer scope, assembly, context content, and consuming subject. At deserialization, reconstruct validity by checking evidence, not by trusting a stored success flag. Keep unknown historical contexts outside qualified ingestion. Permit deliberate context sharing only when dependencies match and each consumer association is explicit.

**Falsifiable acceptance:** Every enumerated missing/mismatched/replayed-receipt case is rejected through real entrypoints before decoding or admission side effects, as applicable; an authorized control succeeds. Retain the route inventory, validator witness, rejection reason, and state-before/state-after evidence.

### K02 — P0: Build a typed product firewall, including serialized boundaries
**§refs / closes:** §§3.1, 3.5–3.6; §11.1. Closes G03/G05. **Owner:** encoder/context API maintainer.

**Why it matters:** Semantic substitutions become explicit forbidden operations rather than errors discovered after poor search results. Distinct Rust newtypes, unlike aliases, provide a language-level distinction; runtime payload validation remains necessary. [R2]

**Kickoff:** Introduce separate opaque products for the cell surface, V7 row embedding, dataset signature, raw/projected Foundation, Target, and assembled context; names are design proposals. Centralize legitimate transformations and require validated constructors rather than arbitrary vector wrapping.

**Falsifiable acceptance:** Compile-fail tests reject cross-product calls; runtime tests reject mislabeled serialized payloads, swapped 512-value products, raw-Foundation-as-final-context, and cell-surface adaptations. Known accepted production fixtures remain compatible and retain the same semantics.

### K03 — P0: Publish and enforce the complete accepted assembly manifest
**§refs / closes:** §§3.2, 3.4–3.6. Closes G04 and the identity part of G06/G10. **Owner:** encoder-release maintainer with policy-loading maintainer.

**Why it matters:** It creates a reproducible subject for acceptance: not merely a policy checkpoint plus whichever width-compatible encoders happen to load.

**Kickoff:** Bind module architectures, active readouts, actual component assets, deterministic construction settings, preprocessing fit state, pooling/projection/normalization versions, precision policy, and supported consumer identity. Separate static assembly identity from request-specific scope and data identity; bind both in the resulting context receipt.

**Falsifiable acceptance:** Mutating any bound component causes rejection or an explicitly new challenger identity before proposal generation; restoring the accepted manifest reproduces the documented fixture results. An independent loader audit reconciles declared components with those actually used, with no invented legacy hashes.

### K04 — P0: Run a scoped counterfactual and denied-read encoder harness
**§refs / closes:** §§3.3, 3.5; §2.1. Closes G02 and tests scope binding from G01/G06. **Owner:** encoder-evaluation boundary maintainer.

**Why it matters:** Output comparison challenges the claim that forbidden data cannot influence context, while denied-read instrumentation addresses paths that output comparison alone could miss.

**Kickoff:** Specify allowed membership per consumer. In isolated runs, vary forbidden labels; separately vary excluded features for inductive/TrainFitted contexts. Freeze authorized inputs, selection, ancestry, and randomness; block access to forbidden stores. Include an allowed-input positive control designed to exercise target conditioning.

**Falsifiable acceptance:** No forbidden access occurs; context-stage outputs satisfy a predeclared, justified repeatability bound under forbidden-only perturbations. Controls exercise the intended dependencies. Test outer-held and fold-local boundaries separately; where joint conditioning is permitted, label it explicitly rather than asserting invariance it does not promise.

### K05 — P1: Make §3.6 an executable context calculation
**§refs / closes:** §§3.1, 3.5–3.6. Closes G05 and localizes G07/G08. **Owner:** context producer maintainer with an independent test author.

**Why it matters:** A third party can reproduce each transformation and identify the first divergence, instead of comparing two opaque 2,048-value outputs produced by the same potentially wrong helper.

**Kickoff:** Recover actual formulas and layouts; publish fixtures for row embeddings, each named summary, raw Foundation, projected Foundation, Target, normalization outputs, and final assembly. Include schema, masks, precision, and edge-input expectations; do not choose missing formulas during the audit.

**Falsifiable acceptance:** An independently implemented calculation matches the accepted fixtures under declared stage-specific criteria, identifies deliberate layout/projection/normalization mutations at their first divergence, and handles or rejects every declared edge case. New semantics receive a new ABI and compatibility qualification.

### K06 — P1: Make caching dependency-aware and proof-preserving
**§refs / closes:** §§3.3–3.5. Closes G06 and request-routing aspects of G10. **Owner:** context/cache maintainer.

**Why it matters:** Correctly scoped reuse can preserve encoder efficiency without turning historical or cross-request outputs into apparently current evidence.

**Kickoff:** Derive cache keys per product from actual dependencies, including relevant row/label identities, schema, fit state, selection, assembly, and normalization. Use immutable validated payloads, explicit versioning, and atomic publication; propagate dependency identities into the composed context.

**Falsifiable acceptance:** Each dependency mutation misses or rejects the stale entry, while unchanged authorized inputs reuse it. Cache-on and cache-off paths satisfy identical qualification criteria. Concurrent requests and restart/replay tests cannot substitute an incompatible request context or misbind its receipt.

### K07 — P1: Qualify row alignment and Target-window semantics
**§refs / closes:** §§3.1–3.3, 3.6. Closes G07 and strengthens G02. **Owner:** input-preparation and Target-encoder maintainers.

**Why it matters:** Alignment is part of the meaning of a supervised encoding, not a minor loader detail. Explicit selection also makes deliberate sampling distinguishable from silent truncation.

**Kickoff:** Define schema/order contracts, feature-label joins, missingness masks, input-domain restrictions, and window selection/padding rules from actual implementation. Add identity-tagged row fixtures and separate set-semantic behavior from chronological or other order-sensitive behavior.

**Falsifiable acceptance:** Label-only reordering and mismatched row membership fail validation; authorized joint reordering follows the declared contract. Boundary, missing-row, oversized-input, and batch-partition tests either reproduce the specified output or return the specified refusal, with selection receipts retained.

### K08 — P1: Extend encoder qualification without moving its goalposts
**§refs / closes:** §§3.2–3.6; §10.2. Closes G08. **Owner:** encoder-backend qualification maintainer.

**Why it matters:** This preserves the meaning of the existing evidence while adding coverage for execution placement, numerical edge cases, and semantic handoff behavior.

**Kickoff:** Recover the retained cases and exact comparison surfaces; keep their `3e-4` maximum-absolute-error limit unchanged. Add independently identified stage fixtures, explicit finite-value checks, device/asset rejection probes, repeated-run checks, and downstream handoff observations under declared randomness.

**Falsifiable acceptance:** Existing cases continue to pass unchanged; non-finite results cannot be marked as numerical passes; documented CPU refusal precedes asset access. Every operation claimed as CUDA has execution evidence, and undeclared fallback is rejected. Any new consumer-level drift threshold is frozen before evaluation, not inferred from encoder tolerance.

### K09 — P2: Establish the marginal value of the three-block context
**§refs / closes:** §§3.5–3.6; §7.5. Investigates G09/G10 without claiming they are observed failures. **Owner:** encoder-research maintainer with search-evaluation owner.

**Why it matters:** This distinguishes a useful engineering composition from attractive dimensional bookkeeping, and identifies whether redundancy, pooling loss, or ignored conditioning is the highest-value research target.

**Kickoff:** First measure controlled block perturbations with the accepted assembly as diagnostics only. Then create separately trained/qualified compatible controls or challengers for selected pooling/projection/block hypotheses. Match data scope, search budget, and comparison protocol; preregister task metrics and compute-accounting rules.

**Falsifiable acceptance:** A block-value claim requires a retained matched comparison meeting a preregistered improvement criterion and showing uncertainty, failures, and resource cost. Null results must be reportable. Zeroing a block in the accepted policy is not alone a fair causal comparison because it changes the input distribution.

### K10 — P2: Turn context receipts into an independent replay and migration tool
**§refs / closes:** §§3.2–3.6; §12.1. Consolidates G04/G05/G10 closures. **Owner:** context-contract maintainer with an independent reviewer.

**Why it matters:** A minimal reproducible evidence bundle strengthens the system-specific context composition: owners can distinguish source availability, faithful reconstruction, live enforcement, and actual consumer use rather than presenting them as one claim (§1.1).

**Kickoff:** Build an authorized replay bundle containing the normative specification, assembly identity, scope/selection identity, representative stage outputs, and consumer binding. Replay without trusting producer-side success flags; generate a compatibility diff for proposed encoder changes. Keep sensitive rows and targets access-controlled.

**Falsifiable acceptance:** An independent consumer reconstructs the declared fixture and rejects deliberate scope, component, layout, and handoff mutations. An ABI-changing migration cannot overwrite the accepted identity or relabel old contexts. Publish an explicit list of remaining unverified historical inputs rather than retroactively certifying them.

**Execution order:** Start K01/K02/K03 as the P0 contract boundary and K04 as its adversarial proof; develop K05–K08 against that boundary. K09/K10 follow stable identities. Hardening acceptance is mandatory for stronger integrity claims; positive encoder-performance claims require separate measured evidence.

## Evidence gaps / UNVERIFIED load-bearing claims

| Claim or missing detail | What the extract establishes | Evidence needed to close it |
|---|---|---|
| Hyperion product identity | §3.1: V7 row embeddings are 128 per row; cell surfaces vary by configuration and are not a context ABI. | Full §§3.1–3.2: accepted active readout, architecture, input semantics, and rejection of other surfaces; runtime negative tests. |
| Dataset signature | §3.1: four summaries per embedding coordinate, yielding 512 values. | Full §§3.5–3.6: actual summary functions, axes/order, masking, empty-input behavior, and independent intermediate fixtures. |
| Foundation transformation | §3.1: target-conditioned raw width 2,048, frozen transformed width 1,024. | Full §§3.2–3.4 and §3.6: architecture, projection mechanism/identity, fitting or construction lineage, label scope, and numerical calculation. |
| Target context | §3.1: frozen target-window encoding of width 512. | Full §§3.2–3.3 and §3.6: supported targets, selection/order, window/padding/mask rules, asset lineage, and scoped input/output fixtures. |
| TrainFitted versus historical joint processing | §3.3 exists in the contents, but its substantive body is omitted; §2.1 explains the general leakage concern. | Full §3.3: mode definitions, exact producer memberships, fitted transform scope, ancestor closure, and counterfactual plus denied-read evidence. |
| Accepted component assembly | §3.4 is named but omitted; §4.2 describes policy-loading checks, not a complete inspected encoder manifest. | Full §3.4 and component load receipts: names, shapes, dtypes, hashes/configurations, active paths, and policy/encoder compatibility checks. |
| Final normalized 2,048-value context | §3.1 gives constituent widths; it does not supply a complete layout or formula. | Full §§3.5–3.6: block order/offsets, normalization sequence and boundary rules, typed assembly entrypoint, and consumer-bound replay. |
| Live provenance enforcement | §11.1 explicitly says live eligibility does not call encoder-receipt validation. | §§3.3, 3.5 plus expanded §11 receipt discussion: exact consumer wiring and production-path rejection/no-side-effect receipts. |
| CUDA-only loading and retained parity | §3 opening states early CPU refusal; §10.2 describes eight retained native encoder cases and the unchanged tolerance. | §§3.2–3.4 and §10.2 artifacts: actual case inventory, comparison surfaces, references, finite checks, device execution, and load-order probes. |
| Cache and teaching/proposal compatibility | §2.1 names cached embeddings as a possible carrier; §§3.5–3.6 are trimmed. | Full §§3.3–3.6: dependency keys, immutable ownership, serialization versions, teaching/runtime compatibility, and consumer-use traces. |

The abbreviated policy tensor inventory is not a substitute for an encoder inventory. In particular, Appendix A's `ctx_proj.weight [512, 512]` must not be identified as the Foundation transformation or used to invent a missing 2,048-to-1,024 operator (§§3.2, 3.4; §4.2; Appendix A).

## Fail-open / silent-lie paths (if any in shard)

| Path | Misleading outcome | Boundary that must refuse it | Evidence status |
|---|---|---|---|
| Context-bearing teacher bypasses the receipt helper | Ordinary eligibility appears to imply authenticated context scope. | Live admission validates subject-bound provenance before downstream mutation (§11.1; G01/K01). | Bypass limitation DOCUMENTED; actual contaminated admission UNVERIFIED. |
| Width-only conversion swaps products or bypasses assembly | A correctly sized vector is called the accepted context. | Typed constructor plus serialized producer/transform validation (§3.1; G03/K02). | Hypothesis; full typed firewall not proven in §11.1. |
| Joint/fold-inappropriate target conditioning is called TrainFitted | Final fitter isolation is presented as encoder isolation. | Consumer-specific scope capability and denied-read tests (§3.3; G02/K04). | Scope details UNVERIFIED; no observed leakage claimed. |
| Stale cache or receipt replay is accepted | Current dataset/cut/component identity labels historical bytes. | Dependency-bound immutable cache and consuming-subject validation (§§3.3–3.5; G06/K06). | UNVERIFIED cache behavior. |
| Missing block or encoder failure is silently replaced | Zeros, a generic embedding, padding, or fallback receives the accepted identity. | Explicit unsupported/failure result, or separately named qualified mode (§§3.1, 3.4–3.6; G03/G08). | Candidate negative tests, not documented existing fallbacks. |
| Frozen or numerically qualified is reported as leakage-free | An asset/parity property is promoted into a claim about information access. | Separate identity, fit-scope, semantic, and numerical gates (§§3.2–3.4; §10.2). | Invalid inference identified; actual reporting instance not established. |
| Context is produced but a stale/mismatched tensor is consumed | Trace describes the correct producer while decoding/teaching uses another subject. | Immutable content and request binding at the consumer boundary (§§3.5–3.6; G10/K10). | UNVERIFIED handoff behavior. |

## Cross-links to other sections the owner must not ignore

| Dependency owner | Required encoder-boundary handshake; no independent audit of that owner is implied |
|---|---|
| §2.1 and §§2.5–2.6 — data and permitted input closure | Provide membership, labels, transform ancestry, and family-domain permission identities. Encoders consume these restrictions; they cannot establish historical family isolation merely by excluding current held rows. |
| §§4.1–4.2 — policy consumer | Bind accepted context semantics to the actual policy input and loader identity. A policy tensor inventory or successful forward pass cannot authenticate upstream encoder meaning. |
| §§5.5, 5.8 and §2.1.1 — search/evaluation scopes | Declare whether contexts belong to outer-training search or a particular inner-fit computation; carry that identity through candidate ancestry. Do not silently recast adaptive training-side conditioning as fold-independent evidence. |
| §§6.2, 6.4, 6.7 and §11.1 — teacher/corpus consumers | Enforce context provenance on context-bearing records, preserve it through storage and caches, and refuse unknown legacy contexts under qualified ingestion. Learning owners retain responsibility for their other admission rules. |
| §§7.5–7.6 — evidence consumers | Distinguish conformance and scope integrity from causal encoder utility or transfer lift. A new assembly changes the comparison subject and must not inherit an old measured-benefit claim automatically. |
| §§10.1–10.2 and §12.1 — qualification/publication | Bind each encoder claim to retained fixtures, assembly, source, execution, and input scope. Keep the eight-case qualification narrower than universal product equivalence; publication hashes identify evidence rather than upgrading it. |

## Concerns / residual risk

**Document completeness:** Several HIGH entries may be resolved by untrimmed §§3.2–3.6 or existing code. First recover those sections, then test the claimed contracts; do not mistake audit uncertainty for proof of a vulnerability. G01 remains a specifically documented integration limitation at the supplied source, not a claim about a later build.
**Trust boundary:** A self-asserted type tag or content hash does not establish lawful input access or faithful execution. Mandatory validators still need trusted producer identity, bounded read access, and subject binding; tests provide bounded evidence rather than a universal proof of noninterference (§2.1; G01/G02).
**Compatibility:** Changing normalization, pooling, projection, ordering, or target-window rules can change the policy's input meaning even when widths are unchanged. Preserve accepted behavior during hardening; promote semantic changes as explicit challengers with new identities and qualification (§§3.1–3.6).
**Value versus integrity:** A fully valid context may contribute little predictive/search value, and a useful context may violate an evaluation boundary. K01–K08 establish enforceable integrity and conformance; K09 addresses incremental utility. Neither automatically establishes transfer or legal novelty (§1.1; §7.5).
**Privacy and historical limits:** Keep target-bearing fixtures and scope membership access-controlled; digests are not a de-identification guarantee. New controls cannot reconstruct missing historical preprocessing or encoder-training ancestry. Preserve unknown status and avoid retroactively certifying old contexts (§§2.1, 3.3–3.4).
**Bottom line:** The strongest immediate encoder-side improvement is a mandatory, typed, scope-bound, replayable context contract. Complete its rejection tests before expanding purity or interchangeable-encoder claims; use a separate matched experiment to establish whether the accepted three-block design earns its complexity.

External engineering references support general design choices only, not product findings: **[R1]** scikit-learn, *Common pitfalls and recommended practices*, §12.2, on training-only fitting of preprocessing (`https://scikit-learn.org/stable/common_pitfalls.html`); **[R2]** *The Rust Programming Language*, *Advanced Types*, on distinct newtypes versus aliases (`https://doc.rust-lang.org/book/ch20-03-advanced-types.html`). All product-specific evidence above is anchored to the supplied monograph sections.
