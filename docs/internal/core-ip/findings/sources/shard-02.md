<!-- from-tar tab=2 url=https://chatgpt.com/c/6ab70221-df50-83ea-84b5-30af4ad0b80a sha256=2f40431e46cbfc8128af04271f8ca6a0899599fdf4cc0733bfb29fc684c03903 chars=40584 -->

# Data and evaluation boundaries — adversarial review and strengthening plan

**Artifact notes:** shard 2 of 10; batch tab 2. Scope: §§2.1–2.7, with other chapters cited only where they determine a data-boundary claim.  
**Evidence cutoff:** 2026-09-25T19:31:50Z. Publication source stated in the extract: `3d71fce041d1dd1173ed98160f7d986bb426ba6f`.  
**Historical boundary:** §2 separately dates observations to 2026-09-23T16:41:48.371336+00:00 at `68f56404c3e199e06413bfd645d3fe0803d7b96f`; publication time does not refresh those observations.  
**Review basis:** the supplied structured extract, not a source-tree audit. No product files, manifests, AWS account, models, or runtime receipts were inspected; no product tests were executed.  
**Evidence labels:** DOCUMENTED means explicitly stated in the extract, not independently verified; INFERRED means a falsifiable failure scenario; UNVERIFIED means the supplied evidence cannot settle the claim.  
**Missing material:** §§2.2–2.7 are represented by their contents headings, not substantive bodies. Their absence is an evidence gap in this review, not proof that the underlying implementation is absent.  
**Ranking:** severity measures damage to the claimed evaluation boundary, not demonstrated exploitation. P0 blocks the affected independence/qualification claim; it need not block clearly labelled development experiments.  
**Decision:** withhold unconditional “unseen-family,” “historically uncontaminated,” and “fully fold-pure discovery” claims until their specific closure evidence exists. Preserve narrower, accurately labelled results.  

## Executive hits

- **Gap — the clearest documented enforcement gap is upstream provenance.** §11.1 says live `eligibility.rs` does not call the fit-scope validation in `encoder_receipt.rs`; a clean final fit therefore cannot establish a clean context or ancestor chain (§2.1).
- **Improvement — make data scope an admission prerequisite, not an optional receipt.** Require a validated dependency closure at proposal consumption, teacher admission, and corpus loading, including cached contexts and selected ancestors; reject missing provenance before the affected artifact is reused (§§2.1, 2.5–2.6).
- **Gap — a thirty-dataset name list is not an immutable benchmark release.** The exact members, totals, group mapping, preparation recipe, object versions, and mutation enforcement are unavailable here. All corresponding claims remain UNVERIFIED (§§2.2–2.4).
- **Improvement — release one sealed panel identity.** Bind source bytes, prepared-table semantics, row/target correspondence, group/family inventories, splits, and permitted consumers; corrections create a new release rather than silently repairing an old one (§§2.2–2.3, 2.7).
- **Gap — row-disjoint and group-disjoint do not establish family-disjoint or history-clean.** Related corpus variants, inherited encoders, policy ancestors, and cached teachers may cross the intended boundary; the supplied catalog and historical-closure bodies are missing (§§2.1.1, 2.4–2.6).
- **Improvement — combine family exclusion with a hermetic input snapshot.** Resolve dataset variants to evidence-backed families, quarantine unresolved identities for final audit use, and deny the learning worker access to nonallowlisted objects and derivative artifacts (§§2.4–2.6).
- **Gap — inner OOF, outer evaluation, and timing are different evidence.** §2.1.1.1 describes five training-side fold losses and timing on repeated fitting features; §10.1 acknowledges an omitted staging deviation. Neither establishes unseen-family accuracy or a clean staging boundary.
- **Improvement — attack the boundary with counterfactual replays and score-release controls.** Perturb forbidden labels, swap caches, rename aliases, mutate staged inputs, and replay resumed campaigns; require visible rejection or invariant pre-score artifacts under the declared protocol (§§2.1–2.3, 2.5–2.6).

## Ranked IP gaps (CRITICAL/HIGH/MED/LOW)

### G1 — CRITICAL: mandatory provenance enforcement remains open at admission
**§ref / evidence:** §2.1, “Leakage paths that a row split alone does not close”; §11.1, first finding. The missing live eligibility call is DOCUMENTED; successful exploitation is not demonstrated.  
**Attack/gap story:** a target-conditioned context, fitted transform, or selected ancestor carrying forbidden information could accompany a candidate whose final fitter obeyed its row split. A caller-supplied `heldout_labels_read=false` cannot authenticate those upstream dependencies.  
**Detection:** submit otherwise eligible fixtures with an absent receipt, wrong consumer, wrong fold, mismatched label digest, and a forbidden grandparent dependency; trace whether the live admission and corpus-loading paths reject them before use.  
**Harden:** require one boundary validator over the entire dependency closure, not just the terminal fit. Bind producer, consumer, membership, labels, transforms, cache identity, ancestry, and protocol; missing or unresolved dependencies cannot imply permission.  
**Decision consequence:** block “boundary-qualified teacher” status where closure is missing. Do not assert that existing accepted weights are contaminated without tracing their actual inputs.  

### G2 — CRITICAL for an unseen-family claim: historical input exclusion is not established
**§ref / evidence:** §§2.5–2.6, “The historical input closure and the meta-test question” / “Family-level domains and the permitted input closure”; §9.1 distinguishes availability from exposure. Historical closure is UNVERIFIED.  
**Attack/gap story:** a nominal meta-test family may have contributed to earlier policy training, frozen encoder training, teacher selection, or inherited caches. Excluding it from today's loader cannot undo information already embodied in an inherited artifact.  
**Detection:** resolve proposed meta-test families against every reachable learning root and derivative record; distinguish catalog availability, actual reads, encoder use, teacher use, and optimizer exposure. An untraceable root must produce “historical exposure unknown,” not “never seen.”  
**Harden:** publish a component-specific exposure classification and claim scope. Establish family independence from an auditable clean root, or narrow the claim to a prospective no-new-exposure experiment with inherited history explicitly unresolved.  
**Decision consequence:** do not promote row-level generalization, a fresh split, or an unchanged checkpoint into evidence of historical family disjointness. Missing history is a qualification blocker, not proof of contamination.  

### G3 — HIGH: fold-local fitting may coexist with discovery-level leakage
**§ref / evidence:** §§2.1–2.1.1.1; §3.1 describes target-conditioned context; §3.3 names TrainFitted and historical joint processing. The actual per-inner-fold context/preparation schedule is UNVERIFIED.  
**Attack/gap story:** computing one target-conditioned context over all outer-training rows and reusing it inside inner folds would let inner-held labels affect proposals. Globally fitted scalers or imputers could similarly cross a fold boundary; the extract does not establish that either occurs.  
**Detection:** before any inner score is revealed, perturb only that fold's forbidden labels and inspect context, transform state, proposal inputs, and ancestor provenance. Test forbidden features separately under an inductive protocol; preprocessing learned from held data is a general leakage risk [M1].  
**Harden:** distinguish adaptive inner-search evidence from fold-pure evaluation of the complete discovery procedure. For the latter, rebuild all label-dependent discovery inputs within each permitted training scope and assess the selected procedure on an untouched outer scope.  
**Decision consequence:** using inner validation scores for selection is not itself an implementation defect. The defect would be representing that adaptively selected, possibly context-conditioned evidence as independent evaluation of the whole learner.  

### G4 — HIGH: “frozen panel” does not yet establish immutable bytes and interpretation
**§ref / evidence:** §§2.2–2.3, “The frozen thirty-dataset panel…” / “Local preparation and panel storage.” Exact membership, totals, immutable storage, and validation behavior are UNVERIFIED.  
**Attack/gap story:** the manifest stays unchanged while a local file, S3 key, parsing rule, target column, group assignment, or row order changes. Reusing positional split indices after such a change can silently evaluate a different subject.  
**Detection:** mutate source bytes, swap target order, change parser settings, redirect a staged path, remove a dataset, and update an object under its existing key. An unchanged release identity must not authorize any changed input interpretation.  
**Harden:** seal raw and prepared identities, row-to-target mapping, split memberships, grouping logic, and preparation implementation/configuration. Verify the bytes actually consumed, not merely an earlier path lookup; missing members cannot trigger substitutions.  
**Decision consequence:** the advertised panel cardinality is not independently reconstructed here. A valid correction should create a new panel release and explicitly preserve the previous release's reports and comparison identities.  

### G5 — HIGH: duplicate groups and dataset names are insufficient family boundaries
**§ref / evidence:** §2.1.1 explicitly distinguishes duplicate/group checks from semantic family exclusion; §§2.4 and 2.6 omit the catalog and family inventory. Operational family disjointness is UNVERIFIED.  
**Attack/gap story:** alternate targets, resampled releases, renamed tables, or transformed copies of one underlying source population can occupy different dataset identifiers. Conversely, treating an entire broad application domain as one family could overexclude unrelated studies.  
**Detection:** combine source lineage, collection/entity identifiers, row overlap, schema fingerprints, and declared derivations; review ambiguous matches independently of performance. Similar dataset names alone neither prove nor disprove shared family membership.  
**Harden:** version a hierarchy separating content duplicates, shared entities/source families, and broader domains. Resolve task variants to the relevant exclusion unit; leave uncertain memberships unresolved rather than treating them as novel families.  
**Decision consequence:** no actual alias collision is established by this extract. The actionable gap is the unavailable family registry and enforcement evidence, not an accusation about any named dataset.  

### G6 — HIGH: a catalog allowlist may not close the worker's reachable inputs
**§ref / evidence:** §§2.3–2.6 cover local storage, S3 catalog, history, and permitted closure. Actual credentials, mounts, object versions, network restrictions, and cache handling are UNVERIFIED.  
**Attack/gap story:** an allowlisted loader could coexist with broad S3 credentials, inherited local caches, alternate object versions, logs containing forbidden summaries, or a “latest” fallback. These paths could expose data excluded from the stated input set.  
**Detection:** attempt unauthorized reads through supported storage paths and inherited cache locations; force an allowlisted-object miss and watch fallback behavior. Also test a new version under the same key: S3's default GET retrieves the current version, not a pinned historical one [M2].  
**Harden:** build an immutable authorized snapshot before worker startup, constrain mounts/credentials/network, and validate every opened artifact against it. Record S3 version IDs when usable and independently verify content; an ETag is not universally a full-object MD5 [M3].  
**Decision consequence:** current closure requires both artifact lineage and enforced access policy. Access logs alone are not enough if uncontrolled input paths exist; no particular AWS configuration is claimed to be present here.  

### G7 — HIGH: repeated feedback and an undisclosed staging detail can defeat a nominal holdout
**§ref / evidence:** §§2.1.1.1, 2.5–2.6; §7.1.1 warns about adaptive reuse; §10.1 acknowledges a staging deviation whose details are omitted. Deviation existence is DOCUMENTED; its scope is UNVERIFIED.  
**Attack/gap story:** outer scores, rejection reasons, or dataset-specific diagnostics can inform later candidate or checkpoint choices even if the fitter never reads outer labels. Merely freezing panel bytes does not freeze the analyst's feedback channel; adaptive holdout reuse can overfit [M4].  
**Detection:** recover the staging journal and classify what was present, readable, actually read, and used for decisions. Audit scorer invocations, returned messages, aborted runs, retries, and subsequent model choices, including human-visible feedback.  
**Harden:** separate learning and scoring privileges; freeze the selected subject before releasing outer scores; bind every score release to an append-only campaign history that survives retries. Explain the omitted deviation before issuing claims that depend on its boundary.  
**Decision consequence:** no actual outer-label read is established here. Retain narrow historical timing observations where justified; withhold only the stronger claims for which the missing staging or feedback evidence is load-bearing.  

### G8 — HIGH: local preparation may preserve bytes while changing the prediction problem
**§ref / evidence:** §§2.1, 2.3, 2.7; §8's fitted input stage makes preprocessing part of the inference subject. Preparation rules and catalog-to-table mappings are UNVERIFIED.  
**Attack/gap story:** target aliases, post-outcome columns, ambiguous joins, target-derived grouping, implicit missing-value coercions, or reordering labels independently of rows can survive ordinary hash and row-disjointness checks if the wrong representation was sealed.  
**Detection:** inspect feature availability at prediction time, transformation fit scopes, join cardinalities, and stable row/target identifiers. Use fixtures with target aliases, repeated entities, shuffled targets, unseen categories, and held-side-only missingness; these are proposed tests, not observed anomalies.  
**Harden:** declare task/target semantics, feature availability, parsing and missing-value policy, join rules, entity/group provenance, and row alignment. Treat learned preparation as fold-scoped fitted state rather than an undocumented catalog-cleaning step.  
**Decision consequence:** an accurately hashed but semantically wrong table is reproducible, not valid. Feature-only preparation is not automatically permissible: the declared inductive or transductive protocol must decide which features may be consumed.  

### G9 — MED: refreshed catalog metadata can silently rewrite historical evidence
**§ref / evidence:** §§2.4, 2.7; §2's dated observation paragraph; §9.1; §12.1. Actual catalog entries, counts, anomalies, and correction records are UNVERIFIED.  
**Attack/gap story:** a catalog refresh changes dataset definitions, family assignments, task labels, or prepared row counts, while a report continues to use the old experiment identity. Catalog size may also be mistaken for a clean training set or actual optimizer exposure.  
**Detection:** diff dated snapshots and reconcile raw objects, parsed tables, valid tasks, excluded items, and family assignments under explicit counting rules. Compare every report's input identity and observation date with the particular snapshot it actually used.  
**Harden:** retain historical snapshots and append correction records; publish deterministic change classes and reasons. State separately what was available, permitted, observed, and historically unknown; keep refreshed metadata out of old experimental identities.  
**Decision consequence:** no numeric catalog anomaly can responsibly be reported from the extract. File hashes establish byte identity, not semantic correctness, complete exposure history, or current operational qualification (§12).  

## Ranked improvements / kickoffs (P0/P1/P2)

All interfaces and acceptance tests below are proposals, not claims about existing APIs. Each kickoff must retain negative and positive cases; an always-reject implementation is not an acceptable solution.

### I1 — P0: require a transitive data-boundary receipt at every reuse boundary
**§ref / closes:** §§2.1, 2.5–2.6; G1 and the derivative-artifact portions of G2/G6. **Suggested owner:** learning/integration, with evaluation sign-off.  
**Why it matters:** the core reusable unit becomes a program/context with authenticated permitted dependencies, rather than a score plus a boolean. This directly addresses the documented separation between receipt types and live eligibility enforcement.  
**Kickoff:** extend the existing receipt path into a mandatory validator used by proposal consumption, teacher admission, and corpus loading. Resolve raw inputs, transforms, contexts, selected ancestors, and consumer identities recursively; treat unresolved roots as unqualified.  
**Falsifiable acceptance:** the live paths reject absent, tampered, wrong-fold, wrong-consumer, and forbidden-grandparent receipts before qualified use. A complete authorized fixture passes; caller flags cannot override the result; validator invocation is visible in the retained trace.  
**Retained evidence:** dependency closure, exact input/protocol identities, rejection reason, consumer trace, and successful control receipt. Enforce again on reload/resume rather than trusting the artifact's former admission flag.  
**Claim boundary:** validation authenticates the declared evidence chain; it does not prove historical pretraining exclusion unless that history is itself supported. No standalone receipt can establish statistical generalization.  

### I2 — P0: publish a sealed, replayable thirty-dataset panel release
**§ref / closes:** §§2.2–2.3 and 2.7; G4. **Suggested owner:** data platform and benchmark owner.  
**Why it matters:** one release identity would identify the evaluation subject independently of local staging paths and publication edits, making changes discoverable before they corrupt longitudinal comparisons.  
**Kickoff:** seal advertised membership, raw content digests, prepared content/schema, row/target alignment, target definition, source/group/family inventory, split membership, preparation implementation/configuration, and exclusions. Preserve ordered row mapping separately from order-insensitive duplicate fingerprints.  
**Falsifiable acceptance:** independently reconstruct the advertised thirty distinct dataset-version entries and reconcile each declared total to its stated counting rule. Mutation of any bound input or semantic field fails release validation; missing entries cannot be replaced or quietly skipped.  
**Retained evidence:** manifest, preparation replay, membership reconciliation, mutation fixtures, and exact objects consumed. A path replacement between validation and consumption must be rejected or rendered irrelevant by consuming an already verified immutable object.  
**Claim boundary:** when order has semantic significance, reordered data are a new prepared identity; when canonicalization is declared, preserve the explicit mapping. Corrections receive a new release ID and cannot overwrite historical reports.  

### I3 — P0: run learning from a deny-by-default input snapshot
**§ref / closes:** §§2.3–2.6; G6. **Suggested owner:** runtime/data infrastructure.  
**Why it matters:** a permitted-input list becomes an enforceable boundary only when the worker cannot retrieve excluded raw data or a derivative substitute through another route.  
**Kickoff:** prepare only authorized objects and derivative caches, verify them, then start the worker with restricted read-only mounts and no unnecessary storage/network privileges. Pin non-null object versions where available; otherwise create a separately sealed content-addressed snapshot rather than treating an unversioned key as immutable.  
**Falsifiable acceptance:** reads outside the snapshot, alternate-version requests, stale caches, symlink escapes, and network fallback attempts fail before data consumption. A missing authorized object stops the request; a valid offline replay succeeds without broad credentials.  
**Retained evidence:** snapshot identity, effective permissions/mounts, observed reads, cache validation decisions, denied-access fixtures, and a complete authorized run. Compute an independent whole-content digest rather than assuming S3 ETag semantics [M2], [M3].  
**Claim boundary:** authorized scoring infrastructure may access its own separate heldout snapshot. Its permissions, logs, and result channel must not be inherited by the learning worker.  

### I4 — P0: build a counterfactual, fold-aware boundary battletest
**§ref / closes:** §§2.1–2.3 and 2.6; G1/G3/G4/G8. **Suggested owner:** evaluation engineering.  
**Why it matters:** perturbation tests challenge whether forbidden information can influence an artifact, while access-denial tests challenge whether it can be reached. Both are needed; neither is replaced by a self-reported label-read flag.  
**Kickoff:** fix the permitted data, split identities, and random streams; independently perturb forbidden labels, forbidden features where applicable, target aliases, ancestors, and caches. Observe artifacts before any legally permitted score feedback, and include a deliberately instrumented forbidden-read fixture.  
**Falsifiable acceptance:** outer-label perturbations leave upstream learned state and proposal outputs invariant, apart from identifiers explicitly describing the changed audit input. Fold-pure mode similarly isolates inner-held labels before scoring. Reuse of label-dependent caches across changed labels, or scope-dependent caches across changed fit membership, is rejected or recomputed; the planted violation is detected.  
**Retained evidence:** paired inputs, allowed/forbidden scope, access traces, dependency receipts, artifact differences, and an authorized training-data perturbation that exercises the fixture's intended signal path. Demonstrate both cold-cache and warm-cache behavior.  
**Claim boundary:** compare deterministic identities exactly and declare any necessary numerical nondeterminism policy in advance, using unchanged-input controls. Adaptive inner selection may change after its permitted scores arrive; do not misclassify that as a failed isolation test.  

### I5 — P0: make family identity and historical uncertainty first-class
**§ref / closes:** §§2.4–2.6; G2/G5. **Suggested owner:** data stewardship, jointly with learning provenance.  
**Why it matters:** a family-heldout result is defensible only when the exclusion unit and inherited exposure status are known. A renamed dataset or new target must not acquire a false “unseen” label.  
**Kickoff:** build a versioned registry of source populations, entities, dataset derivations, alternate targets, and domains; attach evidence and unresolved relationships. Map every permitted input and learned component to this registry without choosing assignments from model performance.  
**Falsifiable acceptance:** documented aliases and derived-target variants cannot straddle a family exclusion boundary; unrelated controls are not collapsed merely by a broad domain label. Unresolved relationships or untraceable pretrained roots block the corresponding never-seen-family claim.  
**Retained evidence:** registry snapshot, adjudication reasons, cross-partition relationship checks, inherited-root status, and a deliberate alias-insertion test. Materializing a newly discovered forbidden relationship must invalidate the affected qualification, not rewrite its past status.  
**Claim boundary:** “prospectively withheld after freeze” and “never present anywhere in historical learning” are separate assertions. Permit the former only with its actual scope; do not promote it automatically into the latter.  

### I6 — P1: evaluate the entire learning procedure on sealed families
**§ref / closes:** §§2.1.1 and 2.5–2.6; boundary handoff to §7.5; G2/G3/G7. **Suggested owner:** independent evaluation.  
**Why it matters:** the desired evidence is about transfer of the learned proposal/search procedure, not simply predictions on rows withheld from the last fitter. Family-level nesting can test that stronger proposition once the input history supports it.  
**Kickoff:** precommit family assignments, allowed inputs, parent/challenger identities, search protocol, and result-release rules. Exclude audit families from cross-task learning. A separate evaluator may use each audit task's designated fitting partition for context construction and fitted search; keep its outer scoring labels sealed and prevent its resulting teachers/artifacts from returning to learning.  
**Falsifiable acceptance:** the run completes with validated family/input closure and frozen predictions before outer audit-label scoring; any route from outer audit feedback into the compared subjects invalidates that audit instance. Within-task fitting-side adaptation remains allowed only as preregistered. Report matched outcomes even when negative, inconclusive, or operationally failed.  
**Retained evidence:** complete boundary receipts, family membership, subject-freeze record, scorer access history, paired protocol identities, and all scheduled audit dispositions. A successful integrity check is distinct from a positive transfer result.  
**Claim boundary:** do not invent a performance threshold, sample count, or measured lift here. The evaluation owner must preregister those choices under §7.5; this kickoff supplies the data-boundary conditions required to interpret them.  

### I7 — P1: control score release and resolve the staging deviation
**§ref / closes:** §§2.1.1.1, 2.5–2.6; cross-links §§7.1.1 and 10.1; G7. **Suggested owner:** benchmark operations and audit.  
**Why it matters:** feedback can contaminate future decisions without any label-read event in the fitter. The omitted staging disclosure also prevents distinguishing a protocol breach from actual downstream influence.  
**Kickoff:** retrieve the retained staging evidence and separate materialized, accessible, read, and decision-used states. Require an immutable prediction/subject commitment before scorer release, with an append-only campaign record that reserves and records every permitted look.  
**Falsifiable acceptance:** retry, crash, resume, and campaign-fork fixtures cannot obtain an unrecorded extra score release. An aborted request after release retains its exposure record; scorer diagnostics do not leak per-row targets or unapproved summaries to learning.  
**Retained evidence:** staging classification with timestamps, effective access policy, scorer-call history, release receipts, decision history, and a signed disposition for each claim affected by the deviation. Unknown influence remains unknown rather than becoming “no effect.”  
**Claim boundary:** a look ledger exposes and limits feedback; it does not itself supply a statistical correction for adaptive reuse. Any repeated-look analysis needs its own prespecified validity argument [M4].  

### I8 — P1: validate prepared-table semantics before split-dependent work
**§ref / closes:** §§2.1, 2.3, 2.7; G8/G4. **Suggested owner:** ingestion/preparation.  
**Why it matters:** a frozen table can still encode the wrong task, misaligned targets, or information unavailable at deployment. Semantic validation complements, rather than replaces, byte-integrity checks.  
**Kickoff:** require stable row/target correspondence, explicit task/target definitions, predictor availability declarations, join cardinality checks, group provenance, and declared missing/category/parse policies. Keep learned statistics in fit-scoped state; preserve source-to-prepared mappings.  
**Falsifiable acceptance:** target-alias, post-outcome-feature, duplicate-join, and target-shuffle fixtures are rejected or explicitly quarantined. Held-side-only category or missingness changes cannot alter training-fitted preparation under the declared inductive mode.  
**Retained evidence:** preparation receipt, feature availability review, row/join reconciliation, fold-specific fitted-state identities, and positive fixtures with legitimate missing values and new categories. Rejecting all imperfect real-world tables is not a passing implementation.  
**Claim boundary:** transformations allowed by a declared transductive protocol must be named and scored separately from inductive results. Dataset-wide grouping for split construction also needs an explicit allowed-data rule, not an accidental exception.  

### I9 — P2: reconcile catalog snapshots and generate bounded claims from evidence
**§ref / closes:** §§2.4–2.7; G9. **Suggested owner:** data catalog and publication tooling.  
**Why it matters:** consistent accounting protects the core narrative from both accidental scale inflation and loss of comparability when metadata are refreshed. It also makes real catalog anomalies actionable instead of anecdotal.  
**Kickoff:** diff immutable snapshots into added, removed, content-changed, schema/target-changed, family-reclassified, and status-changed records. Reconcile physical objects, logical tasks, preparation outcomes, and exclusions; render only the counts supported by each dated snapshot.  
**Falsifiable acceptance:** stale counts, duplicate task IDs, missing content, unresolved target mappings, and changed family assignments trigger explicit findings. An intentionally refreshed catalog cannot change the input identity or exposure claims of a historical report.  
**Retained evidence:** machine-readable reconciliation, exception disposition, snapshot lineage, and a generated evidence table linking every data claim to its observation date, source version, and status. Preserve unknown historical exposure rather than filling it from catalog totals.  
**Claim boundary:** these are proposed anomaly classes, not discovered product anomalies. Exact totals and object names must come from the missing §§2.4/2.7 evidence; no catalog numbers are inferred here.  

### I10 — P2: add dependency-based quarantine and clean replay after boundary discoveries
**§ref / closes:** §§2.1 and 2.5–2.7; G1/G2/G5/G9. **Suggested owner:** provenance/data platform.  
**Why it matters:** a late alias discovery or invalid preparation receipt should have a traceable effect on contexts, teachers, evaluation claims, and their descendants, without silently erasing history or indiscriminately invalidating unrelated work.  
**Kickoff:** reuse the dependency closure from I1 to compute affected descendants when input/family status changes. Append qualification revocations and issue a clean-replay plan rooted before the first affected dependency; keep the historical record intact.  
**Falsifiable acceptance:** inserting a forbidden-family edge quarantines every reachable qualified descendant in a test graph and no unrelated control. Warm caches and resumed runs cannot retain revoked status; resolving an appeal creates a new reviewed status rather than deleting the finding.  
**Retained evidence:** impact path, affected claims/artifacts, revocation history, authorized replay inputs, and new qualification receipts. Measure operational repair cost only after implementation; no speed or accuracy benefit is assumed.  
**Claim boundary:** removing a record from a catalog or masking a current input does not remove its influence from learned parameters. Unknown ancestry requires conservative quarantine of the affected claim or a genuinely clean-root replay, not cosmetic relabelling.  

## Evidence gaps / UNVERIFIED load-bearing claims

The following evidence requests are specific closure conditions. A section heading, hash, source-link label, or successful unit test cannot substitute for the requested operational evidence.

| Claim needing closure | Status from supplied extract | Minimum evidence / section that would close it |
|---|---|---|
| Exact frozen panel members, totals, and grouping | UNVERIFIED; title advertises thirty datasets | Full §2.2; pinned member manifest, explicit counting rules, row/group inventories, independent reconciliation. |
| Panel immutability from storage through consumption | UNVERIFIED | §§2.2–2.3; versioned raw/prepared identities, runtime verification, mutation/refusal receipts, correction policy. |
| Local preparation respects every relevant split | UNVERIFIED | §2.3 plus §3.3; preparation implementation/configuration and per-scope transform/context receipts. |
| S3 catalog identity, contents, and relation to panel | UNVERIFIED | Full §2.4; exact dated catalog bytes, object/version inventory, aliases, and panel-to-corpus joins. |
| Historical meta-test exclusion | UNVERIFIED | Full §2.5; component-specific permitted/observed histories and provenance for inherited learning roots; otherwise retain unknown. |
| Family/domain membership and worker exclusion | UNVERIFIED | Full §2.6; evidence-backed registry, frozen assignments, dependency closure, effective permissions, and denied-access tests. |
| Refreshed catalog counts or named anomalies | UNVERIFIED; no actual entries supplied | Full §2.7; old/new snapshots, reconciliation, anomaly rows, and dispositions. No invented anomaly count. |
| Row/group disjointness in production | DOCUMENTED at §2.1.1; not independently tested | Retained `SelectionPlan` tests and real membership receipts at the execution source; historical statements alone do not prove current operation. |
| Provenance gate closes teacher admission | Documented limitation: §11.1 says live eligibility omits receipt validation | §11.1/§12.1 plus §2.1; production consumer trace and refusal tests, including nested ancestry and reload. |
| Staging stayed within the intended boundary | Deviation is DOCUMENTED; actual access/influence UNVERIFIED | Complete §10.1 deviation disclosure and retained staging/access journal, interpreted against §§2.3 and 2.6. |
| Timing demonstrates heldout accuracy or transfer | Not established; §2.1.1.1 expressly separates timing from accuracy | A separate, complete boundary-qualified audit under §§2.5–2.6 and §7.5, not more repetitions of fitting-feature prediction. |

## Fail-open / silent-lie paths (if any in shard)

Only the receipt enforcement limitation below is explicitly documented. The other rows are attack paths to test; their operational existence is UNVERIFIED.

| Path | Resulting false impression | Required refusal or containment |
|---|---|---|
| Receipt types exist, but live eligibility never invokes them (§11.1) | “Eligible” implies clean upstream context (§2.1) | Mandatory closure validation at each qualified-use boundary. |
| Unknown family or missing historical root defaults to clean (§§2.5–2.6) | “Unseen family” despite unresolved history | Unknown blocks that claim; retain a narrower, explicit status. |
| Missing object triggers latest-version/local-cache fallback (§§2.3–2.4) | Same panel or corpus identity despite different consumed inputs | Exact-snapshot read or hard failure; no implicit substitute. |
| Catalog refresh overwrites panel/task meaning (§§2.2, 2.7) | Historical results apply to refreshed data | New release and explicit lineage; old results retain old identities. |
| Inner-held target influences proposal context (§2.1; §3.1) | Fold-local final fit becomes “independent discovery OOF” | Declare adaptive-search scope or cross-fit the complete evaluated procedure. |
| Related dataset variant bypasses a name-based denylist (§§2.1.1, 2.6) | Dataset-disjoint means family-disjoint | Versioned source-family registry plus closure checks. |
| Retry erases a previously released outer result (§2.5; §7.1.1) | Fresh audit while choices already used feedback | Append-only score-release history inherited by retries and forks. |
| Global preparation or row/label misalignment is sealed (§2.3) | A content hash proves task correctness | Semantic and fit-scope validation before release qualification. |
| Timing on repeated fitting features is labelled generalization (§2.1.1.1) | Throughput becomes accuracy evidence | Separate result types and evidence requirements; retain the original scope. |

## Cross-links to other sections the owner must not ignore

**§3.3 and §§3.5–3.6 — encoder boundary:** identify exactly which rows, features, labels, and learned preparation entered each context and cached product. Do not assume a correct width proves a correct fit scope.  
**§§5.5, 5.8–5.10 — search boundary:** carry permitted-data ancestry through candidate selection and refit; identify the stage at which each score may legitimately influence a decision. This is provenance, not a redesign of GP mechanics.  
**§§6.2, 6.4, 6.7–6.8 — reuse boundary:** prevent teacher admission, cache reload, corpus reuse, or continuation from discarding scope restrictions. A newly correct loader does not certify older inherited inputs.  
**§§7.1.1–7.1.2 and §7.5 — interpretation boundary:** dataset/cut/evaluator identity changes require new comparisons; adaptive development evidence and family-heldout transfer evidence need distinct claims.  
**§8.1 and the fitted input stage — deployment boundary:** retain preparation identity through fitted artifacts and exported state so the evaluated task is the deployed task. Export parity is not evidence that training inputs were permitted.  
**§9.1 — history boundary:** catalog availability and actual exposure are different; the data owner supplies exact identities and unknowns without manufacturing training totals.  
**§10.1 and §§11.1–12.1 — evidence boundary:** recover the staging deviation, close the documented live-receipt omission, and bind claims to execution-specific receipts rather than the publication revision alone.  

## Concerns / residual risk

**Scope of conclusion:** this is an extract-level audit. It establishes a documented enforcement limitation and multiple precisely testable evidence gaps, not a demonstrated contaminated benchmark, discovered catalog anomaly, or measured effect size.  
**Trust roots:** validated hashes and signed receipts authenticate recorded bytes and assertions; they cannot certify an upstream collector's undocumented history. External pretrained components may require narrower claims even after all local controls pass.  
**Counterfactual limits:** invariance on a finite perturbation suite does not prove universal noninterference, and fixed inherited contamination may not react to a current label permutation. Pair tests with access constraints, historical evidence, and positive controls.  
**Family uncertainty:** source families can be ambiguous, and near-duplicate detection may miss transformed relatives or overmerge unrelated data. Preserve review reasons and sensitivity analyses; do not choose the taxonomy to improve the result.  
**Statistical limits:** immutable data can still be repeatedly overused. Integrity controls and score-release accounting make the experiment auditable but do not by themselves establish transfer, unbiased post-selection estimates, or broad domain coverage.  
**Practical tradeoff:** fold-local context rebuilding and clean-root replay may increase cost. Introduce safe caching only after dependency identity is complete; measure overhead rather than weakening the boundary or guessing its price.  
**Recommended sequence:** land I1–I5 before issuing the affected clean-boundary claims; use I6–I8 to establish stronger evaluation evidence; add I9–I10 for durable catalog reconciliation and contamination recovery. Continue clearly labelled development work throughout.  

### External methodological references
These primary sources support only the general methodological/storage statements marked above; they do not verify this product. Accessed for this review on 2026-09-25; the monograph's historical evidence dates remain unchanged.
[M1] scikit-learn documentation, “Common pitfalls and recommended practices,” data leakage: https://scikit-learn.org/stable/common_pitfalls.html#data-leakage  
[M2] AWS documentation, “How S3 Versioning works,” version-specific retrieval: https://docs.aws.amazon.com/AmazonS3/latest/userguide/versioning-workflows.html  
[M3] AWS API documentation, “Object,” ETag semantics: https://docs.aws.amazon.com/AmazonS3/latest/API/API_Object.html  
[M4] Dwork et al. (2015), “Generalization in Adaptive Data Analysis and Holdout Reuse,” arXiv abstract: https://arxiv.org/abs/1506.02629  
