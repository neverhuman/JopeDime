# GB-06: bounded GP expansion and PolicyLM seeds

This is the P1 discovery study and G14/G15 interface. It does not establish H1,
policy improvement, qualified reward, or a serving candidate. Phase 0 (#296) is
already merged. This work uses its distinction between source, fixture proof,
independent qualification and promotion.

`clover::expansion` freezes an ordered training-context roster, verifies its
original parent-role manifest, hashes transformed rows with their labels intact,
and refuses validation/sealed families or changed rows. The role manifest is
issued by the data owner; a caller cannot substitute a new role assignment under
the same hash. Roles and encoder provenance remain trusted issuer boundaries.
The GP worker receives only fitting rows and an inner holdout within those
training families. No outer-test or sealed rows are supplied.

## Actual GP configuration

`expansion::config::audit` reads `gp.rs`, `grammar.rs`, `tournament.rs`, their
selection/activation sources, grammar vocabulary, and the functional pipeline
search sources. The machine receipt records the full source commit and every
audited file's SHA256. It refuses dirty audited inputs or changed extraction
fragments. Changes to the source require a new review of the extraction; the
receipt is a source audit, not a Rust AST proof or operational qualification.

The paths differ:

| Setting | Engine `gp.rs` | Pipeline scratch comparator |
|---|---|---|
| Initial population | 80 | 12 |
| Surviving parents | Top 10 | Top 4 elites |
| Operators after initialization | Draw modulo 16: random 2/16, mutation 7/16, crossover 7/16 when survivors exist | Even offspring slots cross when two parents exist; other slots mutate; no measured parents means random |
| Maximum transforms | 6 | 4 |
| Seed ingestion | Exactly 80 distinct programs | Bootstrap plus at most 4 canonical distinct offered seeds, then random fill |
| Duplicate policy | Canonical identity once per search; 10,000 proposals per evaluation slot | Once per cut including failures; 8 child draws; stop after 8 barren generations |
| Islands in this study | One, no migration | One, one worker, no migration |
| Measured execution | Shipped namespace activation currently refuses fits | Real in-process `feat-math` fits on training rows and scoring on inner holdout |

The engine's existing tournament replaces 64 neural slots with 64 generic slots
and retains 10 teacher and 6 exploration slots. It uses the engine grammar's
projection and task/generation filters. The source receipt records the grammar
limits separately from the native random generator's depth limit. Actual
per-operator counts come from observations; an unexecuted path has null counts.

No engine code or activation rule is changed. The scratch result explicitly names
the functional pipeline engine. It cannot be used as evidence that the fenced
80-program engine measured these seeds.

[gp-config.post300.json](gp-config.post300.json) is the owning test's source audit
at `e9f07496c2d0b2f864a84f58099115edd0447454`, after owner PR #300 restored
the evaluator's lockfile identity. It binds all twelve audited files.
[gp-config.main944.json](gp-config.main944.json) remains a dated pre-revert audit;
its lockfile pin does not describe the current main source.

## Current PolicyLM bridge

[policy_gp_seeds.py](policy_gp_seeds.py) uses the existing `PinnedPolicy`, actual
`PolicyLM`, tokenizer and trie decoder on CPU. It verifies immutable code and
checkpoint manifests, the physical parameter digest, Kumo's 2,051-dimensional
context, the original float64 vector digest, frozen train-only roster and parent
roles. An existing STOP file refuses inference. There are no optimizer updates.

The output binds model/checkpoint/code, tokenizer/support/normalizer, context,
batch protocol, temperature and each original draw seed. Likelihood is the sum
of actual sampled trie-child log probabilities including EOS. It is not a
grammar-normalized training likelihood, q-head score, or descendant propensity.
Duplicate draws remain in the source batch. The Rust parser validates every
program before deduplicating/capping the fitted seeds.

`PolicySeeds::pipeline_seeds` produces the actual `jope.pipeline-seeds/v1` input.
The pipeline fits admitted seeds on its own cut and records consumed programs.
`engine_population` produces `parallel.seed_population.v1`, with the separate
native preparation and training-CSV hashes that engine ingestion requires.
It requires independent seeds and exactly 80 distinct canonical programs.
Engine parent weights remain null to isolate population seeding; behavior
likelihoods remain in the referenced policy batch. Caller-supplied independent
programs still require the engine's actual grammar/fit admission before execution.

Retrieval ancestry must refer to a context in the same frozen train roster and
carry its original measurement receipt pin. Receipt pins do not qualify scores.

## Frozen study design

The checked fixture packet freezes one synthetic training parent, its role
manifest, ordered fit/inner-holdout rows and encoder identity. Its weights are
fixture weights. It is a CI proof of the bridge and stopping/accounting behavior.

The empirical context roster is frozen from CX-KV's train-only packet
`7dd9955418e0b2e27986b1c9ec247f09321c231e6bad4c6d860f294b8f32d7b7`:
24 contexts from 24 parents, selected from 2,267 contexts and 430 parents.
The source cohort omits contexts without measured native detail. It inherits a
research split and does not establish an untouched holdout. Select up to 24
distinct authorized training parents in sorted parent-ID
order, at most one context per parent, using declared dataset/cut/preprocessing
and original encoder references. Fewer available parents are reported explicitly;
neither validation263 nor sealed families may fill gaps. Preserve the original
training labels. Freeze the initial 6,640-program catalog and permitted cache
snapshot; no later winners or scores enter that snapshot.

Use frozen v1 parameter identity
`ea078456690b0665592652b2ed5b04a0deb39b9e604de97a513dc885e467f9fb`
for the policy-seeded arm. Freeze three search RNG seeds, 20261013–20261015,
and rotate arm execution order by seed to expose timing effects. The primary
control is ordinary GP; a third arm uses training-only retrieval seeds. All arms
use identical rows, operators, worker/island counts, fit precision and per-cut
limits: 200 attempted fits and 180 active seconds. These are design limits,
not optimal search settings or authorization for a live campaign. Proposal
inference, compilation, failed fits, cache activity and custody costs are retained
separately and included in any later resource comparison.

Each arm records attempted/measured/failed candidates, seed consumption,
per-operator counts, source/row/policy pins, stop reason, best inner-cut score and
observation hashes. Equal attempt caps do not establish equal resource use.
Pipeline fits are nonpreemptive and generation-0 bootstrap/seeds are
seconds-exempt. A host supervisor must enforce CPU/RAM/wall limits for the real
study. An overrun, unequal completed attempt counts, missing terminal resource
receipt, or zero measured candidates invalidates an equal-resource comparison.
The scratch receipt therefore leaves CPU totals null and resource comparability
false until an independent host receipt exists.
`policy_optimizer_updates` counts updates to PolicyLM only. GP candidates fit
their own models; archived early scratch receipts used `optimizer_updates` for
this same PolicyLM-only count.

No extension or silent retry follows timeout, failed publication, missing input,
HOLD, or unknown execution. Freeze identities through the Phase 0 experiment/run
store before a real study; check its HOLD and attempt ledger before dispatch.
This scratch API does not replace that admission service or mint qualified rewards.

Novelty has three stages: a canonical program outside the frozen catalog; distinct
prediction bits on the same legal training probe; and independent parity/protocol
qualification. Different strings with identical probe predictions collapse at
stage two. Missing probe measurements stay unknown, and matching a finite probe
does not establish global semantic equivalence. The current novelty API reports
syntactic/probe novelty with catalog/probe hashes and keeps qualification absent.

H1 requires repeated useful, independently qualified non-catalog improvements
across training families at a declared practical effect/cost threshold. Adaptive
inner-holdout scores and a new SHA alone cannot close it. Statistical thresholds
and family-cluster inference come from GB-07; no posthoc threshold is chosen here.
Ordinary-GP failure or negligible useful expansion stops further guided-GP claims.
This seed bridge may run scratch comparisons without claiming learned guidance
helps; the causal guided-GP study remains gated by ranking/learning evidence.

## Verification and remaining evidence

Run the existing library target; no new Cargo target or foreign coverage entry is
needed:

```sh
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only \
  cargo test --locked -p jope-learning --lib clover::expansion
```

For the actual-class CPU fixture, export the fixture packet with
`CLOVER_GB06_FIXTURE_PACKET`, then run the producer's `--fixture-smoke PACKET`
with a fresh `--out BATCH`. Set `CLOVER_GB06_POLICY_BATCH=BATCH` and
`CLOVER_GB06_TEST_RECEIPT=RECEIPT` on the focused Rust command to feed its real
draws into the GP comparator. This uses actual PolicyLM and actual GP fits, with
initialized fixture weights and zero optimizer updates. It is separate from the
frozen-v1 empirical study. `CLOVER_GB06_CONFIG_RECEIPT` exports the audited source
configuration through its owning test.

Actual frozen-v1 CPU inference has produced four original draws for the first
selected training context and a byte-identical replay. Kernel readbacks bind one
CPU, 4 GiB RAM, zero swap, 64 tasks and a finite supervisor deadline. Parameters
remain `ea078456…`, with zero optimizer updates. No real-context program was
fitted by this inference proof.

CX-KV subsequently supplied the selected context's original outer-training prefix:
889 rows and three features, unchanged labels, complementary ordered inner folds,
preprocessing identity and training-only retrieval ancestry. No outer-test suffix
was read or exported. The dated machine record is
[gp-evidence.20261013.json](gp-evidence.20261013.json). Raw tables, encoder vectors,
weights, full observations and resource logs remain in the outside-checkout
`clover-gb06` custody archive.

Real pilot A declared sixteen fits per arm, 45 active seconds per arm and a
55-second supervised cut deadline. Its ordinary arm completed five measured fits
in 49.62 active seconds; one nonpreemptive fit crossed the admission-time limit.
The supervisor terminated the worker at the deadline. Remaining arm execution is
unknown without a terminal comparison, and the second cut was not started.
The evolutionary comparison is inconclusive. No retry or extension followed.

A separately declared minimal smoke B used two fits per arm on the first cut:
bootstrap plus one random, PolicyLM or retrieval program. All six fits were
measured, and both original offered seeds were consumed. Its supervised wall time
was 0.503 seconds, with one CPU, 4 GiB RAM, zero swap and a 55-second deadline.
Pilot A and smoke B stay within the original aggregate 110-second/96-fit
allocation; unknown work from A is retained in that accounting. This proves real
seed ingestion at equal attempt caps. It contains no offspring or evidence of
learned guidance improving evolutionary search. Resource comparability remains
false because the study's complete inference and per-arm costs are not closed.

The production novelty API compared seven unique measured programs with the
frozen 6,640-program catalog. Five are outside that catalog; this includes a
previously measured retrieval seed. Training-probe behavior and independent
parity qualification remain unknown. Catalog absence alone does not establish
a newly discovered or useful program.

### Closed real-budget H1 pilot, 2026-10-13

The owner authorized a separate frozen-roster CPU pilot on xbabe3 using the
catalogue's original `train.csv` and incumbent remeasurement on matching pipeline
cuts. This projection differs from the original S1 flywheel views. Source
`bd716765c401e02210cb6add87e31e1d6efc5c35`, roster `42f5caa7…`, export
`eb081147…`, and the frozen-v1 actor parameters were pinned before execution.
Each context had ordinary, policy-seeded and retrieval-seeded arms, two cuts,
one island and two threads. The admission cap was 512 fit attempts per arm,
including eight reserved additional fits; each cut had a 252-attempt/300-active-
second search budget. Nice-15 scopes capped each arm at two CPUs, 2 GiB and
900 seconds, within a 32-CPU/32-GiB slice with zero swap. The original campaign
deadline was 06:16:21Z; its closed event was recorded at 06:16:23Z.

The coordinator recorded 53 completed arm receipts and 18 interrupted arms.
All three arms for context indices 08, 12, 15, 16, 17 and 22 are incomplete;
their terminal fit counts and outcomes remain unknown. The first five contexts
hit the 900-second scope limit; context 22 hit the remaining campaign deadline.
Context 00 had two coordinator arms plus a completed ordinary arm launched under
a separate supervisor. That separate receipt is retained, giving 54 physical
receipts, but context 00 is excluded from the requested coordinator-only readout.
There was no retry, deadline extension or replacement context.

The resulting paired summary covers 17 contexts. Lower loss is better; counts
and median log ratios describe selected champions on the same TRAIN inner folds:

| Numerator / denominator | Lower loss | Equal | Higher loss | Median log loss ratio |
| --- | ---: | ---: | ---: | ---: |
| Policy-seeded / ordinary | 10 | 2 | 5 | -0.000349264 |
| Retrieval-seeded / ordinary | 16 | 1 | 0 | -0.035361252 |
| Policy-seeded / retrieval-seeded | 2 | 0 | 15 | +0.017235560 |

Ordinary, policy-seeded and retrieval-seeded champions had lower loss than the
remeasured incumbent in 8, 10 and 11 of these 17 contexts, respectively. Their
completed search-attempt totals were 1,913, 1,714 and 2,082. All arms used the
same configured caps and matched prepared data, but actual attempts and CPU
costs were unequal. The frozen-v1 producer completed all 24 contexts and 96
original draws in 369.972 CPU seconds, with unchanged parameters and zero
PolicyLM optimizer updates.

Forty of the 51 selected champions are syntactically outside the frozen catalog.
Full-catalog behavioral novelty remains unknown. Native/emitted Rust predictions
passed bitwise comparison on both TRAIN probes for 48 of the 51 champions.
Parity failed for context 14's retrieval arm and context 18's ordinary and policy
arms, with maximum absolute deltas of 0.0021133743, 0.0321004556 and 0.1367213160.
These failures remain visible in the descriptive summary and cannot support
qualified discovery claims.

The [machine-readable readout](gp-h1-readout.20261013.json) and
[17-context table](gp-h1-paired17.20261013.tsv) record exact context identities,
coverage and failures. Raw receipts, interrupted observations, executor sources,
input pins and resource logs remain in the outside-checkout
`clover-gb06/h1-20261013T0510Z` archive. This is NON-QUAL: selected inner-fold
losses and a complete-case subset do not establish useful support expansion,
native S1/HWM qualification or paired frozen-v1 L6 improvement. L6 is a validation
loss; lower is better. Qualified rewards remain zero and H1 remains unestablished.

Pending evidence includes complete terminal resource accounting, full-catalog
behavioral novelty, independent parity qualification, repeated useful discoveries,
and GB-07's preregistered effect/cost threshold. Refresh the private branch after
each main merge and wait for the owner's open queue and main CI before publishing.
The experiment freeze remains in effect; any next arm requires the owner's
separate sign-off on a preregistered proposal.
