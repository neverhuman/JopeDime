# Proposed qualified provider contract for research recording

Status: **proposal for review; CLI activation remains unavailable**. This document
specifies the missing production closure for QLT-03. It introduces no constructor,
registry, command option, schema, dataset admission or execution authorization.
Source inspection: `c989ac5cdcb6769380027a700d2822a54d32b3da`, 2026-10-31.
The durable capture component merged in [PR #1039](https://github.com/neverhuman/JopeDime/pull/1039).

The [current CLI](../../crates/jope-appliance/src/cmd/research.rs) authenticates
its package, then returns `BackendUnavailable`. The
[opaque context and private provider](../../crates/jope-service/src/research/capture.rs)
have no production factory. A valid digest, successful model load or synthetic
control cannot substitute for the qualification evidence below.

## Resolution and admission

Proposed follow-up syntax is `research record --job JOB_ID --out PATH
[--evaluate-all-proposals]`. **`--job` is not implemented today.** It would select
one opaque identifier in an authenticated, explicitly provisioned admission
registry. It would not accept a JSON import, filesystem path or S3 URI as a job.
No directory discovery, latest-checkpoint lookup or implicit dataset selection is
permitted. Registry provisioning and this CLI extension need a separately scoped
implementation approval after contract review.

The registry must resolve a reviewed qualification receipt and an explicit run
admission. Their immutable locators, versions where applicable, hashes and decision
identities must be retained outside the public round stream. The receipt is
trusted through the registry's authorization, not because a caller supplied its
hash. An admission binds these checks to one Plan, RunId, source commit and
recording mode; it supplies finite evaluation, wall, storage and round bounds.
Qualification alone does not authorize compute. Missing or stale evidence refuses
before opening dataset payloads, reserving attempts or creating a record output.

The factory must compare every existing `Binding`/`RecordIdentity` pin with the
accepted receipt and actual loaded bytes. An implementation must specify how the
receipt is retained and revalidated on resume. Any new private CAS reference must
be an explicit run GC root; hiding a reference in opaque JSON is insufficient.
This proposal does not add fields to closed round v1 or checkpoint3.

## Required input closure

| Input | Required evidence and refusal |
| --- | --- |
| Corpus and role | Version/hash-pinned real-S3 catalog, dataset/family identity, rights and explicit train-only admission, authoritative family exclusions, exact allowed object/row closure. Reject unresolved family roles, heldout/sealed families and test/outer-query payloads before opening them. A catalog entry or `partition=train` string alone is insufficient. |
| Prepared matrix | Exact source rows, target/feature order, preprocessing source/parameters, JMAT bytes/hash and immutable path. Preserve labels. Prove that conditioning, evaluator and Plan refer to the same admitted training rows. |
| Model/source | Actual authenticated BundleDigest, tensor/source closure and frozen DOPE `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`. No checkpoint substitution or learned-guidance claim from shape validation. |
| Conditioning | Actual finite 10x32 tokens and ten masks from the qualified full landmark/DOPE/token path, with source/input/parameter pins. No fabricated tokens or silent extraction-mode fallback. |
| Programs | Canonical DSL hash, actual operator-family provenance, feat412 in frozen name order, and original f64 proj16 before adapter conversion. Reject unsupported or unproven features; do not infer missing provenance from a hash or insert zero vectors. |
| Rung0 | Qualified evaluator executable/source/runtime, exact full-training-table CV/fold identities and membership, preprocessing/baseline/metric definitions, measured values and gates, failure semantics and supported input bounds. Current M1 evaluation is insufficient. |
| Reveals/policy | Explicit OD13 decision, ordered selection/tie rule and measured admitted-pool identities. Learned costs require the qualified forward actually used for selection; heuristic costs remain null. |
| Seeds/request mapping | Accepted, hash-pinned deterministic protocol with concrete vectors for admitted and label-only requests, fold/row identities and all randomness that affects evaluation. Unknown historical seed scalars stay unknown; no invented zero or catalog-seed substitution. |

Dataset metadata is already available in the host's
`jope_plan/master/source/S3_DATASET_CATALOG.{md,json}`. Its family-key and sealed
exclusion caveats must be resolved for the admitted job. Do not ask for a dataset
choice or open protected payloads to work around an eligibility gap.

## Provider and executor obligations

The existing `CaptureProvider` methods are the integration boundary:

- `validate_binding(binding, plan)` must resolve the accepted closure, verify
  capabilities, identities, bounds and protocol compatibility before dispatch.
  Digest syntax and array shape checks alone do not satisfy this method.
- `observe(round, dsls, measured)` returns aligned actual program inputs and
  indices into the supplied measured admitted pool. Round0 has no reveals;
  subsequent rounds require four distinct actual surrogate/gate pairs under the
  explicit OD13 rule. Label-only results cannot enter this pool. Preserve raw
  source ordinals and canonical eligible order through the same proposer call.
- `measurement(request, response)` consumes the qualified evaluator's actual
  request-bound result. A successful response has surrogate in `scores[0]` and
  minimization objective in `scores[1]`; `scores[2]` is user value, **not a gate**.
  The separate gate must have an auditable evaluator origin. Do not run a second
  evaluation or reconstruct missing gates from user metrics.
- `label_seed(round, proposal_index)` implements the accepted extra-request
  protocol. Current capture reserves disjoint rung0 slots32..47 with a private
  counter; admitted request allocation and exploration RNG stay unchanged. The
  receipt must establish whether native fitting consumes those seeds and bind
  the actual fold bytes. Slot separation alone does not establish seed parity.

The worker `Response` has no separate gate field. The qualified executor/provider
must therefore share a trusted, immutable receipt keyed by the exact Attempt and
request payload hash, available before outcome journaling. It must bind the actual
evaluator invocation, input/protocol pins, full-precision F/objective and gate;
`measurement` checks agreement with the response. Missing or mismatched receipts
leave the result unknown. The follow-up must approve that transport and any durable
receipt roots; this contract does not extend the worker wire format or repurpose
`scores[2]`.

Admission's `admit_recorded` must return the decision and trace from the same
invocation and RNG stream. An unavailable observed policy refuses; do not call a
second forward to obtain costs. Learned rounds require the real12/20-entry slate
and cost vector. Full-slate and input/cost agreement must be qualified against
actual reference captures. Heuristic recording must honestly identify that policy;
it still requires the actual corpus, conditioning, features and rung0 closure.

## Rung0 and failure boundary

The authoritative [RES-05 recipe](../../research/flywheel_selfplay/custody/SURROGATE.md)
is the dimensionless constrained log-RMSE-ratio cost F and separate CV gate from
`jope.s1-two-fold/v1`, `evaluation_scope=full_train_cv`. Its pinned evaluator source
is `53422343699172b61ed69ed707015558f7cf1a344e5449f30e1eb9d708aef900`.
Use its original DSL complexity, baselines, two folds and full-precision arithmetic;
do not replace F with raw RMSE, a user objective, capped history channels or a
smaller screen. Shape limits and observed733–1959-row replay coverage do not
qualify arbitrary larger product inputs.

A completed gate failure is a known success measurement with the real penalized F,
real objective and `gate=false`. An actual worker/native failure is a failed
observation with its code; the historical numeric failure sentinel must not be
promoted into a successful measurement. Missing, stale, aborted or unjournaled
replies remain unknown and charged. Any difference between historical sentinel
handling and the new executor's failure mapping needs explicit review and fixture
coverage. No unknown is converted to a failed label to make a stream complete.

A qualified executor must produce the real F/objective pair and gate receipt.
Wrapping M1's `[RMSE, objective, user value]` callback cannot qualify it. The
follow-up must demonstrate that the existing Plan/request/fold mapping expresses
the accepted S1 protocol; otherwise obtain an exact source grant before changing
that mapping. Keep regression and binary qualification separate; the current
research context refuses binary tasks.

## Activation evidence and export semantics

Before registering a production factory or replacing the CLI's unavailable
boundary, retain a head-bound acceptance receipt with actual input identities,
commands/exits, runtime/resources, counts and independent review:

1. Positive real-S3, explicitly admitted train-only Rust-engine to standalone
   Python readback in both dispatch modes. Count rounds, proposals, admissions,
   dispatches, successes, known failures and incomplete rounds separately.
   `--require-evaluate-all` must accept only complete all-proposal output.
2. Actual RES-07 reference comparisons for the consumed input/measurement closure;
   no synthetic oracle or successful asset loading is real numeric parity. Learned
   activation additionally needs qualified RNK-08/RNK-10 same-forward costs and
   selection, including ties, masks, ordering and actual reveal composition.
3. At fixed Plan/R/seed,1/8-worker runs preserve planned/admitted context and normal
   GP behavior. Extra labels cannot change parents/promotion. Prove known recovery,
   retained unknown charges, CAS/GC roots, atomic round/outbox installation and
   private/public separation with the actual provider attached.
4. Refusal controls cover missing/changed pins, unsupported task/shape, wrong
   corpus/family/row role, absent OD13/seed protocol, stale result, partial output
   and destination overwrite. Perform protected-role refusals with metadata
   controls; never read protected payloads to create a negative test.

Current materialization requires a completed run with every captured round known,
committed and complete. Convergence/time termination that discards a pending
suffix taints export; a known prefix is retained privately and is not a successful
training-ready export. Do not silently drop/renumber that suffix. Prefix export,
if desired, needs its own reviewed semantics. Store and filesystem commits remain
separate; staging evidence is retained and retry uses a fresh output path.

## Decisions and deliveries still blocking activation

These are responsibilities recorded by the source/status owners, not new task or
compute assignments. Their availability/acceptance must be refreshed before work.

| Accountable owner | Missing decision or delivery at the inspected source |
| --- | --- |
| Jepson/OWNER, recorded through ENG:watcher | Explicit OD13 reveal rule and tie disposition. RES-05 and `RevealedRule` still require a choice; the recommendation is not approval. Accept this provider contract and any subsequent source/registry grants separately from execution budgets. |
| RES-01 source owner `sol_v1_gov03`, catalog maintainer ENG:watcher | Accepted job-specific real-corpus/family/row and source closure; resolve the catalog's unreviewed family/exclusion gaps for that job. No need to choose a new dataset in this PR. |
| RES-05 measurement owner `sol_v1_hwv`, DSC-08 executor owner `sol_v1_dsc08` | Qualified native full-train-CV F/objective/gate mapping, exact fold/request/seed protocol, supported bounds and failure vectors. Arithmetic replay is available; a qualified research executor is not registered. |
| RES-07 source owner `sol_v1_dsc08` and relevant RNK component owners | Actual admitted conditioning/features/projection and measurement capture fixtures. The tracked golden directory contains source/checkers and no qualification fixture set. Respect the existing qualification FIFO for any separately approved generation. |
| RNK-08 implementation owner `sol_v1_gov03`; RNK-10 source owner `sol_v1_hwv` | Qualified production forward and connected same-call observation/admission when learned recording is requested. The inspected main still has the shape-only adapter; no fake costs or implicit reveal choice can bridge it. |
| QLT-03 integration owner `sol_v1_api01`, after approved prerequisites | Implement the trusted job resolver/private factory and conditional CLI wiring under an exact source grant; retain real-engine acceptance evidence above. No such factory or job option is supplied by this documentation. |
| `super_merger_codex` and required COMPLEX reviewers | Review contract, source grants and eventual exact-head implementation/evidence. Merging this proposal establishes neither scientific choices nor runtime qualification. |
