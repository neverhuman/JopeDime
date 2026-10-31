# Encoder receipt and teaching binding: P0 spike

Status: executable, opt-in contract spike. **No production admission enforcement** is
added. Existing teaching/HWM paths and historical receipt identities are unchanged.
The typed receipt and rejection tests are the first stage of the capability-bound
encoder item, not completion of the Jailgun P0.

## Source and problem

Source: Core IP Pipeline — Jailgun Battletest,
host path `/home/ubuntu/xbabe2_kicker/docs/CORE_IP_PIPELINE_JAILGUN_BATTLETEST.md`.
This external audit is not shipped with a JopeDime clone. Its SHA-256 when consulted was
`d54645e62597c7446b88d72522b8949e19961e924905af858fc3ff3c97060463`;
run `run-6206c0d6-52ea-4031-94b8-f3794c1f3573`, dated 2026-09-24.
Executive Brief §1.1 identifies encoder/context leakage; §2.1 proposes
`jope.encoder_receipt/v1` and counterfactual noninterference; §3.1 prioritizes
encoder receipts and the teaching-admission contract as P0.

A `heldout_labels_read=false` claim does not prove that a scaler, target encoder,
vocabulary or context was fitted without forbidden rows. A matching vector width
also does not establish semantic identity. The consumer needs a separately trusted
scope and dependency closure, then must bind the exact transform and output to the
candidate's teaching identity.

## Implemented contract

[encoder_receipt.rs](../crates/jope-learning/src/encoder_receipt.rs) defines:

- `Scope`: full `HwmIdentity`, source family, outer/optional inner fold, ordered
  fitting row/group membership, and explicit forbidden row/group membership.
  Empty or duplicate fitting rows, duplicate forbidden rows, cross-boundary rows
  or groups, malformed required digest pins, unsupported tasks, absent comparison
  context, sealed or
  benchmark claims and non-training label roles are refused. The issuer supplies
  authenticated membership; this module cannot discover an omitted forbidden row.
- `EncoderCapability`: a private-field in-process object, with **no `Deserialize`**.
  The trusted caller constructs it using `bind`; it is not a signed token or a
  sandbox. The constructor is public, so an untrusted caller that controls the
  issuer inputs can invent a capability. Production consumers must never construct
  their expected capability from the receipt they are checking.
- `Encoder`: implementation/bundle digest, configuration digest, seed and semantic
  product. The spike supports one finite little-endian FP32 vector per receipt:
  `row_embedding128`, `dataset_signature512`, `target_embedding512` or
  `teaching_context2048`. Equal-width dataset and target products are distinct.
  Batch outputs and Hyperion cell surfaces require a later explicit ABI.
- `Inputs`: projected feature bytes, optional projected label bytes, and ordered
  parent-receipt pins. Parent pins must be unique SHA-256 values; an empty list
  declares a root encoder. Parent authenticity, completeness and transitive scope
  checks belong to the future issuer/verifier, not this structural checker.
- `EncoderReceipt` (`jope.encoder_receipt/v1`): capability, scope, comparison,
  ordered fitting-row and permitted-label-row hashes; encoder and input closure;
  fitted-transform bytes and output bytes. Serde rejects unknown fields, including
  caller-invented `pure` flags. A label-free receipt binds an empty label-row list.
- `TeachingBinding`: independently retained comparison, candidate program, expected
  receipt digest and product. `check_teaching_binding` rejects missing provenance,
  substitutions and unknown schemas, even if the supplied receipt is rehashed.
  It verifies the retained transform/output bytes, vector length and finiteness.
  Success means **these bindings match**, never that teaching is qualified.

Hashes use existing [canonical JSON](../crates/jope-learning/src/canon.rs), including
array order. The scope hash includes row/group assignments and both fold levels;
the comparison hash includes task, dataset, cut, evaluator, vocabulary, protocol
and comparison context. Source dataset identity and training payload identity are
separate: changing forbidden labels can legitimately change a whole-dataset hash,
while the authorized training payload/transform must remain unchanged.

`project_fitting` only looks up fitting row IDs, preserves their frozen order and
fails when a fitting row is absent. This is a **row projection helper**, not an
encoder sandbox or a feature/label column-capability implementation. Its generic
payload can contain labels; the issuer must supply the appropriate payload type.
The receipt's optional label digest alone cannot prevent undeclared label reads.

## Production teaching-admission contract (not yet wired)

A production adapter must perform all of these checks before admitting a record:

1. Authenticate the prepared source and split, rederive the inner partition via
   [derive_inner_partition](../crates/jope-learning/src/eligibility/inner.rs), and
   construct the capability from that trusted partition, not producer assertions.
   Every inner fold must exclude its own validation rows as well as outer held-out
   and sealed rows. Capabilities for supervised transforms may expose labels only
   for the authorized fitting rows. Use separate feature/label views and confined
   worker inputs; generic row projection alone is insufficient.
2. Verify the selected encoder artifact/configuration and all parent receipts,
   including scalers, target transforms, vocabulary and contexts. Enforce an
   explicit root-encoder policy, reject missing parents/cycles, and independently
   derive projected input hashes from authenticated rows. Unknown provenance is
   diagnostic-only; do not backfill invented receipts for historical corpora.
3. Rehash actual retained state/output and call `check_teaching_binding` with pins
   from the frozen candidate/consumer record. The expected pins cannot be supplied
   by the producer alongside its receipt. Record these pins in the immutable
   admission closure; do not add optional fields that a legacy consumer ignores.
4. Call existing [revalidate_qualification](../crates/jope-learning/src/eligibility.rs)
   on the retained `HwmRecord`. Do not trust a deserialized `Qualification` or a
   producer's eligible Boolean. Match its comparison/program and authenticated
   outer-fold/fitting-row scope to the capability. Keep the current measured
   numerical guard and label-burn checks; this spike does not relax or redesign
   the four-of-five-fold rule or address catastrophic-fold policy.
5. Apply the same gate to every ingest, cached reuse, corpus export and resume
   path. A receipt alone does not establish pure, sealed, benchmark-valid,
   promotion or successful namespace activation. Those authorities remain separate.

Binding equality is necessary but insufficient. Hashes cannot prove that an
external process did not read undeclared data. The deployment gate still needs
input confinement and real-encoder counterfactual tests before claiming inductivity.
No `HwmRecord`, schema, existing admission threshold or namespace activation rule
is changed by this spike.

## Acceptance

```sh
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only \
  cargo test --locked -p jope-learning --lib encoder_receipt::tests
```

The tests cover a matching receipt; missing/unknown provenance; recomputed forged
receipts; dataset/task/vocabulary/fold/order substitution; same-width product swaps;
row and group leakage; retained state/output/program/pin substitution; nonfinite
outputs; and label-free membership. The in-memory reference encoder compares
projected-input and fitted-transform digests after forbidden feature/label changes
and removal, then proves sensitivity to an allowed-row change. This establishes
the helper's limited row-isolation behavior, **not** production encoder invariance.

The existing jope-learning library target runs in PR `rust-test` and `full-default`
according to [coverage.json](../deploy/ci/coverage.json). No new test target or
feature-gated exclusion is introduced. Exact-head CI and independent review are
still required before landing.
