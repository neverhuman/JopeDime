# CLOVER Phase 0 foundations

This implements the three P0 rows of §12 of the [CLOVER vNext direction in
PR #295](https://github.com/neverhuman/JopeDime/pull/295). The acceptance lane is offline CPU;
it does not qualify a live campaign, authorize compute, or establish H1–H3 improvement.

| P0 deliverable | Implementation | Acceptance evidence |
| --- | --- | --- |
| Single experiment identity | `jope_learning::clover::ExperimentIdentity`, canonical JSON SHA-256, persistent lane bindings | Missing/unknown fields and malformed pins rejected; every comparison dimension changes the hash; conflicting labels rejected after reopening |
| Evaluation semantics and data boundaries | Versioned independent quantile fixtures; `project_encoder_input`; production preprocessing/target-copy guard audit | Runtime and emitted Python/Rust checked against hand-derived edges, ties and bins; label/future perturbation invariance and role/task/family/split/fold refusal tests |
| Fail-closed operation | `RunStore`, `EvidenceStore` | Persistent HOLD, interrupted UNKNOWN terminalization, charged slots, failed SQLite transaction, stale recovery, missing/corrupt objects, faults at every publication boundary, immutable retries |

Run from the repository root:

```sh
env -u CARGO_TARGET_DIR CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only \
  cargo test --locked -p jope-learning --test clover_phase0 \
  -p feat-math --test clover_quantile
just required
```

Both packages already belong to the blocking `rust-test` product scope; no GPU, model, corpus,
network, or missing-tool skip is used by these new tests. The quantile test requires `rustc`
and `python3`, which CI supplies. Scratch stays in each lane's existing Cargo target.

## Identity and trust boundaries

The experiment record covers the full Git commit and a complete source/build closure digest;
model-free GP population/controller and RNG state or checkpoint parameters, optimizer and RNG;
evaluator and parity contract;
support; objective; frozen data roster; seed; all resource budgets; and guard identities.
All contract references are SHA-256 content identities. Support must bind the program set,
grammar, tokenizer, canonicalizer and admission evidence. The data roster must bind task,
family, role, split, fold, ordered rows, normalizer and time limits. The issuer retains and
verifies these referenced contracts; syntactically valid hashes alone do not authenticate them.

Lane labels are aliases outside the scientific hash. Once a SQLite registry associates a label
with an identity, it rejects reassignment even across restarts. Keep one registry for the
comparison cohort; independent registries cannot detect each other's conflicting labels.

## Quantile and encoder audit scope

The [v1 fixture](../../crates/feat-math/tests/fixtures/clover_quantiles.v1.json) defines linear
interpolation at `(n - 1) * q`, bins from strictly-less edge counts, normalized indices and
one-hot output. Values are hand-derived, including unsorted training rows, exact-edge ties,
duplicate edges, singleton/constant columns, negative values, signed zero and subnormals.
The audit calls the actual `feat_math::ops` transforms and the actual generated Python/Rust
helpers, checking each against the oracle rather than treating backend agreement as correctness.
It deliberately proves that a shared right-tie implementation would fail the fixtures.
This helper contract admits finite rectangular inputs; pipeline sanitization, GPU and fitted
C++ qualification remain their existing owning lanes. Historical evidence is not reidentified.

Encoder projection looks up only authorized training rows. It returns features separately from
labels; a feature-only encoder receives no labels. A target-supervised encoder may receive the
declared training labels, which remain intact. Validation, confirmation, delivery, other tasks,
families, splits, folds and future observations cannot enter that projection. Tests perturb or
remove forbidden payloads and run the production preprocessing implementation on the projected
cut, checking invariance. The existing production guard refuses exact label-copy features.
Trusted row/role/time metadata and feature lineage remain the split issuer's responsibility;
this audit does not certify arbitrary external encoder code or undisclosed derived features.
Cached inputs must bind the resulting visibility digest and projected bytes, as well as encoder
and normalizer identities; this PR does not silently bless historical embedding caches.

## HOLD, recovery and evidence

Use one controller per run registry. `create` explicitly initializes a new registry and refuses
an existing file; `open` refuses a missing registry or experiment state and never recreates it.
`start_attempt` records and charges an ID transactionally
before execution. `finish_attempt` permits finite rewards only for qualified outcomes; known
failure, timeout, parity-unqualified and UNKNOWN carry no reward. Terminal outcomes cannot be
overwritten. `require_ready` fences adoption/promotion on HOLD or unfinished attempts.

Reopening with unfinished work latches HOLD before new admission. `recover` terminalizes that
work as UNKNOWN and preserves its budget charge. It never clears HOLD. Explicit release requires
no running attempts, the current HOLD generation, experiment and guard identities, and a verified
content-addressed recovery manifest. The caller must independently evaluate the pinned guard
before issuing recovery evidence; matching receipt fields are not proof of guard execution.
Any new violation invalidates earlier recovery evidence.

Evidence uses local immutable SHA-256 objects, same-filesystem staged writes, fsync and create-only
hard-link publication, with read-back verification. The complete identity and all payloads are
verified before the content-addressed manifest is written last. Partial publication leaves only
uncommitted objects; a lost response is retried with identical bytes. No mutable serving pointer
or remote storage adapter is introduced.

`RunStore::publish_evidence` persists HOLD on any unresolved publication, including a lost
response; a restarted controller cannot admit new work or publish result evidence through it.

Budgets must plan at or below **25,000,000,000 bytes/day** and **1,500,000 PUTs/day**. The local
publisher refuses batches, including identity and manifest bytes, above that allowance. These
are planning limits, not a claim of aggregate daily metering: campaign admission must account for previous
publications, retries and all other custody work before any operational rollout. This change
performs no S3 operations or deletion/lifecycle changes.
