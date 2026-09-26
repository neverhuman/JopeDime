# W12 — Generate defensible claims and close the evidence chain

Status: planned. Owner: evidence/publication. Consumes qualified W3/W8/W9 decisions
and W10/W11 measurements. Source scaffolding can precede live results, but publication
claims cannot. Follow [the execution contract](EXECUTION-CONTRACT.md).

## Source anchors and claim boundaries

Inspect `docs/internal/core-ip/tools/build.py`, `verify.py`, `crates/jope-records` (the
findings verifier, formerly `verify_findings.py`),
`docs/internal/core-ip/findings/register.json`, `docs/parity/capability-matrix.md`,
`docs/learning.md`, `MASTER_PLAN.md`, `crates/jope-learning/src/ledger.rs`, and the
original scoring/replay paths behind each proposed KPI.

The register's v1 validator deliberately accepts only open inventory states. A reviewed
evidence-validation extension is required before closing any finding. Publication
source, original execution source, fixture proof, CI, live qualification, measurement
and promotion are distinct identities/states. No refreshed prose upgrades them.

## Ordered tasks

### W12-T01 — Define the claim-to-evidence schema

- [ ] Inventory every capability, KPI and contribution sentence to be generated.
  Bind subject, metric, units, denominator, comparison identity, baseline, input scope,
  original execution source, time, uncertainty and limitations.
- [ ] Represent implementation, fixture/local proof, hosted CI, live qualification,
  measurement completion and promotion separately. Require trusted authority for decisions,
  not a publisher-authored `passed` flag.
- [ ] Acceptance: altered scope/source/subject/denominator cannot inherit an existing
  claim merely because artifact bytes or a publication date match.

### W12-T02 — Extend findings closure with validated evidence

Depends on T01 and the original 195-entry inventory.

- [ ] Preserve original IDs, hashes, source clauses and primary routes. Add explicit
  confirmed-defect, verified-safeguard, missing-execution-evidence, historical-unknown,
  research-result and deferred-non-regression dispositions with scoped attestations.
- [ ] For defects require pre-fix failure, corrected negative/positive receiving tests
  and applicable live proof. For safeguards require current code trace plus regression
  protection. For research require the agreed complete experiment, not positive benefit.
- [ ] Acceptance: missing, substituted, stale, wrong-subject or insufficient-scope
  evidence prevents closure. Original unknown history cannot be changed into a successful
  reconstruction by a new run or an unchecked manually entered evidence path.

### W12-T03 — Generate status and KPI views from committed records

Depends on T01/T02 and W8 publication verification.

- [ ] Generate capability matrices, status tables, score cards and monograph observations
  from validated committed records. Display actual coverage/freshness, failures, exclusions
  and pending gates rather than conditional-success denominators alone.
- [ ] Separate HWM improvement, unique eligible teachers, corpus availability, exposure,
  committed updates, matched search and transfer. Include fold-tail/absolute-skill diagnostics.
- [ ] Acceptance: regeneration is deterministic from the same verified inputs; changing
  presentation metadata cannot strengthen evidence scope or silently change a metric subject.

### W12-T04 — Reconcile historical Elo and genealogy honestly

Depends on T01; original records may remain unavailable.

- [ ] Locate original matches, participants, score transformation, algorithm version,
  ordering and update settings. Attempt independent reconstruction only from retained inputs.
- [ ] Distinguish current scorer parity from historical match recovery. Record missing
  lineage/exposure edges and irrecoverable histories explicitly; exclude them from exact totals.
- [ ] Acceptance: reproducible histories have original input/algorithm receipts;
  unreconstructable ones retain that disposition and block unsupported rankings/continuity claims.

### W12-T05 — Implement dependency-based revocation and quarantine

Depends on T01/T02 and W1/W3 dependency identities.

- [ ] Index descendants of family/input/receipt relationships through contexts,
  candidates, teachers, corpus snapshots, checkpoints, exports and published claims.
  Append revocation decisions with issuer, cause, scope and effective time.
- [ ] Reject newly invalid descendants at live consumers; mark prior publication
  claims revoked/superseded without deleting history. Requalification requires new
  valid evidence and identity rather than clearing a flag.
- [ ] Acceptance: revoking one parent invalidates exactly the affected descendants;
  unrelated evidence stays usable. Offline views show revocation freshness/unknown status.

### W12-T06 — Assemble an independently replayable private evidence package

Depends on W8–W11 relevant gates and T05.

- [ ] Package the regression release references, semantic bundle, candidate/fit lineage,
  admission/HWM decisions, corpus/exposure, checkpoint, matched searches, export and
  measured performance/transfer records with a complete dependency manifest.
- [ ] Restore into fresh custody; invoke receiver/verifier/replay code without relying
  on producer-local caches, mutable `latest` pointers or broad worker credentials.
- [ ] Acceptance: an independent reviewer reconstructs valid decisions and rejects
  missing/substituted/stale/revoked prerequisites. Separate record replay from numerical
  recomputation and account for the latter's explicit input/compute permissions.

### W12-T07 — Publish synthetic conformance fixtures safely

Depends on T02/T06.

- [ ] Derive public synthetic fixtures covering each contract boundary and major
  refusal class without private labels, fitted private constants, credentials or canaries.
  Retain explicit synthetic identity and supported-language/runtime limitations.
- [ ] Exercise them through the same receiving validators as private cases; document
  where they cannot establish real-corpus, hardware, historical or operational claims.
- [ ] Acceptance: private canaries are absent, public replay is reproducible, and a
  synthetic fixture cannot acquire a real-data or live-qualification attestation.

### W12-T08 — Reconcile all findings and produce the release decision

Depends on T02–T07 and completed applicable packages.

- [ ] Account for all 97 gaps and 98 improvements individually, including duplicates,
  W0 dependencies, deferred binary/multiclass/time-series subcases and residual regression
  blockers. Preserve every original acceptance clause, including stop/dependency conditions.
- [ ] Publish completed source/fixture/CI/live/measurement/promotion states separately.
  The initial release requires regression controls, managed three-node execution,
  recoverable S3 progress and the real loop witness at their declared gates.
- [ ] Acceptance: independent audit can explain every closure and every still-open
  limitation. No blanket “all fixed” claim while required operational/historical claims
  remain unsupported; unknown history may remain unknown with the affected claim blocked.

## Verification and delivery

```sh
cargo run -q -p jope-records -- verify-findings
cargo nextest run -p jope-records
git diff --check
```

These commands validate the current open inventory only. New claim/revocation validators
and tests must be implemented and explicitly mapped to CI before their acceptance is
claimed. Existing publication rendering requires its recorded inputs and owning proof
lane; do not hand-edit generated outputs. Proposed slices: (A) claim/closure contracts
T01–T02; (B) generation/history T03–T04; (C) revocation T05; (D) private/public replay
T06–T07; (E) final 195-entry reconciliation T08. Deliver reproducible claim records,
revocation graph, replay package, public fixtures and an honest release/blocker decision.

## Original finding routes

Primary register entries: `S01-G5`, `S01-G8`, `S01-K5`, `S01-K8`, `S02-G9`, `S02-I9`, `S02-I10`, `S03-K10`, `S08-G01`, `S08-G09`, `S08-G10`, `S08-K01`, `S08-K09`, `S08-K10`, `S09-G8`, `S09-G9`, `S09-I8`, `S09-I9`, `S10-G10`, `S10-K10`.

See [the complete crosswalk](FINDINGS-COVERAGE.md) and original clauses. This primary
routing is not a waiver of cross-cutting acceptance or later live integration.
