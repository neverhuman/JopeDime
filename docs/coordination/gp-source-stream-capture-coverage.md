# GP complete-source capture implementation coverage

Developer implementation and independent-review record for PR388. This describes internal
construction paths; the operator access/protocol guide is `docs/program-source-stream.md`.
Source freeze: `3d4fd8823f20a5e3b002ecab135df88462a74f43`. No live/model/benchmark qualification
is implied by source/parser fixtures.

## Engine and language coverage

Coverage is complete for the materialized full-program capture points below when the process
starts with capture enabled. Parser fixtures check the actual captured strings; backend
refusals remain in the four-target denominator. This is not a claim that every vocabulary
operation has a standalone emitter in every language.

| Producer | Capture point | Evidence / limit |
| --- | --- | --- |
| Surrogate and lite-race GP | Every population slot before filtering; private random/mutation/crossover construction also captures children never reached before stopping | `feat-math --test source_capture`; frozen seeded search/generator corpus remains unchanged. Population/operator/elite repeats are separate emitted occurrences. |
| Legacy canonical mutation APIs | Shared random construction, dedicated mutation/crossover observation seams, and explicit-arm output construction | Same typed `DslExpr`/canonical program path; source fixtures and generalized offspring exercise it. |
| FullRust GP | Shared `From<CanonicalGenome>` for random/mutation/expanded mutation/crossover outputs | Actual random-construction parser fixture plus the shared constructor; no refit solely to emit source. |
| Advanced V3 GP | Validated `make_candidate` for seeds/new candidates, including descendant construction | Actual initial population strings parse; all callers share the same canonicalizing constructor. Existing V3 controls retain their own scope. |
| Legacy product pipeline | Every draw that actually produced program text, before score/budget/dedup filtering and independently of `record_events` | Actual CLI comparison and pipeline fixture with search-event recording disabled. Failed operators producing no program retain their native draw errors and are not invented empty-program emissions. |
| Kernel2 and shared strategy | Operator proposals, population slots, evaluation attempts and shared-strategy proposals | Actual kernel2/shared-strategy capture child; every captured full program parses. Overlapping capture points remain labeled. |
| Kernel2 rewrite and linearization moves | Every complete rewritten/linearized program before child admission, including refused or unevaluated proposals | Direct synthetic rewrite and linearization fixture parses actual captured rows. Re-deriving evidence invokes the same construction and is a distinct capture. |
| Engine and policy grammars | Active `Vocab::decode` and `project_proposal` complete output, including GP projection and model-decoded/neural proposals | Actual engine and policy grammar fixtures exercise lossy projection plus decode and parse every output. Decode/projection/replay invocations are separate labeled captures. Historical byte-frozen `jope-engine/src/grammar.rs` is not the active module and is unchanged. No model inference/training is needed for this fixture. |
| Kernel2 complete refinement trials | Coordinate/guarded and LM whole-pipeline objective construction, before admission and numerical catch boundaries; final child evaluation also captures | Coordinate/guarded synthetic training fixtures parse captured programs. LM uses the same canonical full-program construction; existing LM controls retain numerical scope. Bare solver AST steps enter only when a complete pipeline materializes. |
| Generalized GP | Generation zero and successor populations, including preserved elites | Actual generation-zero plus two successor constructions: 240 materialized rows parse. This is a synthetic source fixture, not a benchmark campaign. |
| External/custom or already-running producers | Explicit `Journal::append`; updated binaries/settings required for automatic hooks | No automatic historical archive backfill, unknown-engine discovery or mutation of frozen/running jobs is claimed. |


The six legacy operator bodies and their existing capability-manifest hash expectations remain
byte-for-byte main. All mutation/crossover callers enter observation seams around those bodies;
the core capture fixture invokes each public variant and parses its actual recorded output.
The existing immutable-body pin also runs early in the emitter lane, alongside the 21 new
source contracts (nine selected checks, 22 cases including that pre-existing guard).
