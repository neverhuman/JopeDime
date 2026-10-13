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

Pending evidence: legal real-context fitting/inner-holdout tables, preprocessing
and retrieval ancestry from the data owner; real-cut search resource receipts;
repeated independent parity-qualified discoveries; GB-07's analysis threshold.
Those gaps remain open. Main identity
repair belongs to owner PR #300; rebase after it merges before publishing GB-06.
No S3 calls/settings, training runs, other screens, main checkout source changes,
seal, adoption or promotion are part of this implementation.
