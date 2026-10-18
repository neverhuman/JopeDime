# Replay admission v2 (I04 lane contract)

The additive API is `jope_learning::replay::admission`. Historical HWM qualification,
`effective_label_role`, and v1 teaching/seed/coverage records retain their original bytes and
meaning. None can be deserialized into `ExposureClosure` or `ReplayAdmission`. A current HWM
decision is necessary for the teaching companion, but never sufficient for global admission.

## Trust and identity

The controller installs a canonical `jope.replay-policy-adoption/v2` `PolicyPin`
(byte SHA-256, issuer, epoch) at the fixed `State` path `replay/adopted-policy.json`.
`AdmissionVerifier::from_adopted_policy` reads only that privileged configuration; no public
constructor takes a candidate's pin or policy pathname. Candidates supply only subject/proof
references and requested purpose. Policy installation and write access to the controller State
are trusted administrative operations, outside candidate authority. A changed adoption pin
invalidates existing verifiers and requires reconstruction. This lane provides no policy
adoption endpoint, policy installer,
signature verification claim, or live unsealing. The integrator must connect the controller's
admitted verifier output to this configuration before enabling a live consumer.

The immutable `jope.replay-admission-policy/v2` payload enumerates every exact admitted
exposure receipt, purpose and effective interval. References use the existing State byte SHA
namespace. Policy and exposure bytes must also pass I02 `CanonicalPayload::from_canonical_bytes`
and carry the domain-separated identity given by
`sha256(schema || 0x00 || u64_be(length) || canonical_payload)`. Strict JSON and version checking
reject duplicate keys, extra fields, raw identity floats and signature-shaped additions. A
digest establishes identity relative to the controller pin, not issuer adoption by itself.
I02's public `MeasuredEvidenceProof.independent_verifier_passed` Boolean is never consulted.

Each `jope.replay-exposure/v2` receipt binds the exact subject bytes, original role, family,
family partition, within-task view, fidelity, producer, observation time, policy epoch,
permitted purposes, known exposure list, complete material parent roster and observed labels.
An empty known exposure list differs from missing exposure. A summary, embedding, distilled
selection or context must list every material parent; the admitted producer/verifier is
responsible for the completeness of that roster. This code cannot discover a concealed input.

The verifier checks retained subject/proof bytes, admitted receipt membership, every parent,
policy epoch/time, purpose, and current local revocation closure. It refuses missing, altered,
cyclic, duplicate-parent or excessive ancestry (128 nodes, depth 32). An imported ancestor
revocation invalidates descendants on the next admission/revalidation; unrelated evidence
remains usable. Revocations cover byte and domain identities and policy target artifacts.
An absent local revocation ledger means no revocations imported here, not global freshness.

## Purpose and label matrix

| Purpose | Required ancestry | Result |
| --- | --- | --- |
| Global policy training, representation learning, retrieval/index update, distillation | Every ancestor: `train_search`, original role `train`/`train_fit`, `training`/`inner_validation` view, known train-only prior exposure, explicit purpose | Independent label masks |
| Episode fitting | Every ancestor: same nonempty episode, explicit fitting purpose, permitted training/inner view; no judge, sealed, meta-test or adaptive-test source | Local capability only |
| Audit | Authenticated retained closure and explicit audit purpose | Audit capability only |

Family partitions use the S5 vocabulary `train_search`, `meta_dev`, `qualification`,
`sealed_final`. Original roles remain historical strings; this is not a new role registry.
Unknown roles/views never pass a global request. Fidelity is orthogonal and cannot upgrade a
role. D16 forbids historical `test_adaptive` representation reuse. D30 applies the same global
restriction to summaries, contexts, embeddings and selections as to their inputs.

| Observation | Quality | Runtime | Failure | Policy/action |
| --- | --- | --- | --- | --- |
| Completed, admitted measured receipt | Optional exact finite binary64 metric with unit/denominator | Optional exact milliseconds | Observed false, one attempt | Only an explicit retained measured target |
| Program deadline timeout | Missing | Optional positive right-censored lower bound in ms | Observed true, one attempt | Missing |
| Infrastructure/unknown timeout | Missing | Only an explicitly observed lower bound in ms | Missing | Missing |
| Failed attempt | Missing | Optional exact attempt duration in ms | Observed true, one attempt | Missing |
| Unevaluated proposal | Missing | Missing | Missing | Missing |

Missing fields carry I02 `MissingReason` values plus diagnostic detail and never become zero
scores, success labels, completed
durations or policy rewards. A quality metric's denominator is independent of attempt count.
An explicit target is supervised imitation evidence, not an on-policy estimator or behavior-law
claim. Timeout has no quality/policy fields that a caller could accidentally populate.

A successor policy has a new immutable identity and effective interval. A receipt must have
that epoch and an observation time within the interval. Old receipts and their original roles
and exposures remain unchanged; a new epoch never upgrades an old claim. A new train receipt
still fails when its known prior exposure contains sealed/judge/adaptive-test information.
No actual declassification is performed.

## Ingress census and integration gates

| Existing route | V2 companion and binding | Remaining integration work |
| --- | --- | --- |
| `eligibility::revalidate_replay` / `freeze_teaching` | `qualify_for` / `revalidate_replay_for` / `freeze_teaching_for`: recheck historical measured guards, exact Qualification bytes, every context receipt/state/output/handoff and parent | Switch external consumers to the companion and per-label masks |
| `bank::export` / `bank::list` | `bank::export_for`: authenticate every exact entry before any ranking; require admitted RMSE to equal the bank score | Keep legacy seed output local; global drivers must consume this companion |
| `pools::source::coverage_rows` | `coverage_rows_for`: bind the entire legacy document; project descriptors without quality/incumbent/outcome labels; permit only representation/retrieval; refuse v1 aggregate global label training | Per-attempt supervised ingestion requires individual v2 observations |

Known external callers at the lane base are `jope-cli/src/train.rs`,
`jope-cli/src/train/corpus.rs`, `jope-motifs/src/learning.rs`, and
`jope-learning/src/bin/jope_pools.rs`; legacy bank exports also have CLI callers in
`jope-learning/src/main.rs`. They are outside I04's worker allocation. They have not been
migrated or certified as using the new gate. Existing fixtures are compatibility evidence,
not evidence of global consumer migration. The integrator must resolve this census before
declaring I04 complete or enabling the v2 live loop.

I02 currently supplies canonical/outcome primitives, but no adopted Rust role/purpose registry.
These purpose names and the local v2 payloads are draft interfaces derived from S5. I03 is a
draft dependency, absent from this lane base; no selector/model changes are included here.
The lane's minimal `lib.rs`/`replay/mod.rs` declarations need reconciliation with I27/I36.
The integrator owns canonical schema, coverage, inventory, README counts, full required gate,
exact-head CI and independent review. Lane evidence does not establish those milestones.

Rollback disables the v2 consumer route or reverts its reviewed change, retaining old readers,
original roles, all accepted/rejected/held proof bytes and revocation history.
