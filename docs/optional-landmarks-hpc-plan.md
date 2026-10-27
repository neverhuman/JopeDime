# Optional landmarks and HPC: coordinated completion plan

Execution record: 27 October 2026. The [engineering specification](specs/optional-landmarks/engineering-spec-optional-landmarks.md)
and [mission](specs/optional-landmarks/mission-optional-landmarks.md) govern correctness. This plan includes
the user's subsequent requests for GPU Dope features, Micro HPC reuse, and coordinated upgrades
in both repositories. The attached source documents preserve their wording; trailing Markdown
spaces are normalized for repository hygiene. Implementation and qualification are separate milestones.

## Ownership and coordination

Codex owns cross-repository acceptance and the JopeDime optional-landmarks checkout. Claude owns
the Dope integration checkout. Claude-Code owns native quality/performance qualification. Gemini
owns proposed preparation/kernel optimizations in its own claimed checkout. The separate Codex
Micro-import owner retains the prepared JopeDime import packet; the optional-landmarks writer
will not independently apply that overlapping packet. Read-only reviewers inspect shared
contracts and lifecycle behavior. Claude-Code acknowledged Q1/Q2 ownership in the shared log at
14:32Z; the full quality study follows a validated three-table smoke run. Other proposed owners
remain pending acknowledgement.

Use append-only `jope_plan/comms.md` for contract decisions, claims, exact commits, benchmark
reservations and stopped-head handoffs. JopeDime claims additionally use `AGENT_CHAT.md` under
its lock. A proposed assignment becomes active only when its owner acknowledges it. Each
checkout has one writer. Reuse permitted lanes, respect their limits, and preserve existing
work. Share immutable commits/patch hashes, not mutable local dependencies. No new standalone
Micro experiments are required: integration and new qualification happen in Dope/JopeDime.

The dependency sequence is:

1. Close contract/lifecycle defects and freeze shared fixtures.
2. Publish Dope draft, pin its published immutable head in JopeDime, and qualify the combined path.
3. Publish the linked JopeDime draft with actual package, compatibility and quality evidence.
4. Integrate focused HPC changes through their existing owners, rerun the same acceptance matrix,
   and update the drafts or use linked focused follow-ups. No merge or release is authorized by
   this plan.

## Fixed behavior

- One bundle contract: 23 squashed f32 gains, 23 mask bits, derived status, sample size, scoped
  source digest and optional provenance. NaN and both infinities become positive zero/missing;
  measured finite zero stays present. Raw and squashed constructors are distinct.
- None/fast perform no native CV, spectrum or building-block work. They keep complete train-only
  target, shape and table descriptors. Partial conditioning uses observed evidence only.
- Progressive defaults to N=300. Full remains unsampled unless requested. Actual groups are
  macro 1–6,14,15,22; larger 7–13,16,17; fine 18–21,23. Native folds and completed work are reused.
- Preserve original full native reports and their conditioning path. Terminal progressive and
  synchronous full must agree for identical sampling/options. Completion can still have partial
  measured coverage; unsupported and over-budget cells must remain visible.
- Dope512 is the first 512 actual Dope629 coordinates, with a separate recipe identity. Resident
  GPU weights, intermediates and supplied fusion components remain on the selected GPU until
  explicit serialization. CPU is an explicit reference option, never an automatic fallback. Serial and batched GEMM
  may differ numerically; qualify same-shape equality separately and retain the serial default
  unless a batched path is identical or explicitly versioned and qualified.
- Preserve accepted Hyperion products, normalizers, search candidate ordering/seeds and frozen
  pipeline receipt v3. Publish Dope/fusion updates through the session/event API. Changing how
  search acts on an updated vector is a separate behavior decision.

## Gap closure and acceptance

| ID / owner | Concrete work | Required evidence | Current state |
| --- | --- | --- | --- |
| C1 — Claude + Codex | Freeze bundle JSON, sanitation, sentinel, merge scopes and shared fixtures; bind source to task, targets, schema, labels and physical training IDs | Same fixture parsed in Python, Dope Rust and JopeDime; NaN/Inf, measured zero, single-squash, scope rejection and stable-hash tests | Implemented; final combined rerun pending |
| C2 — Claude | Expose named MLC-v3 training options with work budget 2e12; leave legacy Default unchanged; use named options in both product paths | Options digest and budget in receipts; real/wide tables demonstrate genuine native coverage; legacy full bytes unchanged | Correction requested; Jope currently sets the matching value explicitly |
| C3 — Codex | One admitted native CPU shared across pipeline cuts; cancellation-aware sessions; bounded updates; join all workers on output failure | Initial event precedes worker completion, duplicate idempotency, sibling-join failure test, CPU-ceiling admission test, cancellation/panic preserving last snapshot | Implemented; final rerun pending |
| C4 — Codex | Finish CLI/appliance output protection, error mapping, receipt reference and actual generated schema validation | Weights/input/output alias refusals preserve bytes; event-output errors retain output failure class; generated run and encoding receipts validate | Reviewer defects fixed; positive package proof pending |
| C5 — Claude + Codex | Fix Dope raw-table CUDA CLI to call resident tensor forward and bound its explicit CPU reference pool; published immutable Dope pin; reviewed Candle source reconciliation; released whole checkpoint and license packaged | Actual RTX 3090 numerical parity and residency, CPU-only and CUDA feature builds, dependency fences/security provenance, sealed appliance fit/fit-predict with no runtime download | Combined GPU tests passed on development candidate; final published pin/package pending |
| C6 — packaging owner, pending ACK | Qualify GPU runtime portability separately from weight packaging; retain documented host-library requirement until supported CUDA runtime libraries are packaged with reviewed licenses/loader paths | Clean driver-only host/container, offline startup and GPU fit/fit-predict, tamper refusal, library/driver compatibility matrix and SBOM; existing native-only package remains portable | Open; current GPU profile requires host CUDA libraries |
| Q1 — Claude-Code | Real native retention and feature-importance comparison on the available 112 development datasets | Zero, macro, macro+larger, full, N=100/200/300/500/1000; fixed paired seeds; dataset exclusions/errors and per-dataset metrics; zero relative retention loss, progressive changes, N300/full ratio, smallest tested N meeting 95% | Outstanding; gain-vector cosine/ranking proxies do not close it |
| Q2 — Claude-Code + Codex | Matched latency matrix on real fixtures and synthetic 1k/10k/50k tables | Real fixtures plus synthetic 1k/10k/50k × p10/64/200; ≥15 repeats; p50/p95 public encode time, native evidence separately, full preparation included; CPU and CUDA weight-cold/resident timings, thread/device/commit/checkpoint identities and resource sampling | Partial CPU/device studies exist; full product gate outstanding |
| Q3 — both integration owners | Full required suites and compatibility audit on final heads | Dope engine/fast/check gates; Jope `just required`, relevant feature combinations, historical receipts, checkpoint loading, frozen Hyperion assembly and Python/Rust tolerances | Focused suites pass; full gates have explicit unresolved failures |

Do not silently change the N=300 default in response to an ablation. Report the smallest tested
N meeting the target and distinguish it from this version's agreed default. Do not train an
adapter without a separate decision. Singleton-class sampling currently fails explicitly;
preserve the last snapshot and document that restriction rather than silently switching modes.

## HPC integration waves

Prioritize elapsed public-call cost and CPU work. Port as many applicable improvements as pass
parity, resource and end-to-end benefit checks; component speed alone does not qualify a port.
Micro's correlation-based landmark reports cannot substitute for genuine native CV evidence.
Dope PR [#339](https://github.com/neverhuman/dope/pull/339) is a separately owned native ridge/basis
optimization draft; it is distinct from the optional-landmarks integration draft. Its measured
component gains require a combined-head rerun before product attribution.

| Wave / proposed owner | Work in Dope / JopeDime | Admission and completion test |
| --- | --- | --- |
| H1 — Gemini, integrated by Claude | Rust preparation candidate `7a5cffa9c1ef17ba47c91bf3b3dd340762be435e`: reuse sorted columns/ranks, inverse permutations, scratch buffers, fused exact descriptor/building-block reductions; profile parse, scaler and descriptor spans separately | Existing full/parity goldens and real-table conditioning digests identical; measured 1k/10k/50k preparation improvement; deterministic 1/multiple-thread reductions |
| H2 — Codex + GPU specialist | Preserve resident whole-model CUDA inference; add bounded batched encoding where consumers can batch, reuse input/output buffers, measure allocation/transfer bytes and owned stream lifetimes | Actual GPU numerical parity, bounded concurrent sessions, retained snapshot tensor immutability, B=1/8/64/256 throughput and p95; no per-update weight reload/hash; host copies only at requested export |
| H3 — Claude-Code, integrated by Claude | Reuse full-panel preparation/folds/bases and exact Rust kernel optimizations (PR #339 ridge/basis, then claimed `eval/gbdt.rs` allocation reuse and separate net/spectrum work); evaluate applicable C++17/20 kernels behind explicit backend selection | Same native reports/coverage within a backend, FFI ownership/error tests, sanitizer and package/dependency checks, paired full-path speedup; preserve legacy portable backend |
| H4 — Micro-import owner + Gemini | Focused JopeDime feat-math ports: contiguous Jacobi, width-512 no-op/pre-capping, sort cache, exact HistGBM constant/duplicate filtering and reusable partitions, ksmooth selection | Frozen source/state/prediction parity, tie-order and edge cases, unchanged default emitter identity; opt-in recipe identity for changed generated code; Rust plus C++17/20 emitted-program compile/run tests |
| H5 — Micro-import owner | Persistent bounded workers, binary/slab transport and direct buffered CSV writing; retained fit state; source/toolchain-keyed split runtime/glue compilation and atomic CAS publication | Real selected-model reload, independent reference execution, cold/miss/hit/mixed traffic, no leaked/detached children, input lifetime/CSV-expiry stress and source rebuild custody |
| H6 — Codex + GPU/kernel owners | Evaluate CUDA graph capture and fused routing only after stable buffer ownership and measured B=1 launch costs; evaluate exact GPU native kernels only where genuine CV semantics and costs are preserved | Graph cache keys bind checkpoint/source/Candle/CUDA/device compute capability/dtype/shape/level; lazy active-shape capture and measured break-even; graph slot generations and leased output storage; A/B/A inputs, delayed consumers/backpressure, cancellation/lane recreation and stale-handle tests; same outputs/tie decisions and explicit recipe/device identity; include cold capture/JIT/transfer costs; show end-to-end benefit before activation |

H4/H5 overlap the existing 149-file Micro import packet. The import owner will split and
register applicable source, workspace, tests, assets and CI changes; this checkout will not
apply the separate feat-math patch independently. Coordinate the final Dope dependency and
shared runtime/feature changes before merging any integration branches.

The policy integration currently publishes compatible-size Dope/fusion tensors; the pretrained
policy has not been qualified on their changed distribution. Call this plumbing in PRs. Actual
policy use and any ranking changes require quality evidence and a separate behavior decision;
current search remains unchanged.

A C++ native kernel backend and C++17/20 generated-program execution are separate surfaces.
Pin C++ compiler/libm/ISA and strict floating-point flags (`-ffp-contract=off`, no fast-math
or reassociated reductions), row/tie order and bounded scratch. No C++ exceptions cross FFI.
Any new native FFI requires its own opt-in feature, ABI/ownership contract, sanitizer results,
package/dependency policy and SBOM. Do not feature-unify `fitted-cpp` into the native appliance.
Preserve its existing pure-Rust fitting closure.

Keep full checkpoint SHA verification on every load. Existing measured hashing cost is small
for this checkpoint; resident sessions already amortize it. Do not introduce a stat-only trust
cache. Preserve FP32 inference and exact CPU arithmetic for compatibility-preserving waves;
FP16/TF32/fast-math, alternate solvers, synthesized evidence and changed search policies need
separate accuracy/quality decisions.

## Measurement and gate protocol

Freeze source commit, checkpoint SHA, fixture IDs/physical rows, options digest, compiler,
CPU affinity/thread limits and GPU identity. Use matched baseline/candidate order and repeated
runs; report cold startup, parsing, complete preparation, native groups, conditioning, forward,
serialization and queueing without adding overlapping spans. Include failed and slow cases.
Report time to initial snapshot, time to terminal receipt and resident throughput separately,
including queue/batch wait and measured native coverage/states. For concurrent paths record peak
runnable tasks, CPU-ms, memory and throughput as well as p95. OpenMP, Rayon, C++ workers, JIT and
linker threads all count toward admission. Reserve a common CPU/GPU benchmark window in the
shared log and hold `jope_plan/qualification.lock` during timed latency runs; coordinate builds
and benchmarks through the same reservation. Host load near 240 is unsuitable for promotion
claims: reschedule unstable measurements rather than treating load logging as isolation.
Consider an upstream Dope512 tensor API that shares the existing hidden-state computation and
skips only output heads unused by the first512 coordinates. Retain the complete checkpoint and
legacy629 APIs. Require bit-identical first512 equality at B1/8/64/256 on CPU/CUDA and measured
benefit before adoption; this candidate avoids graph lifetime complexity.

Persistent/captured buffers must not mutate an already published snapshot: retain its tensor
bits, source/conditioning digests and version until the consumer releases it. Use explicit
slot/generation ownership and leased storage before allowing replay reuse.

Freeze the 112-table development roster and its hash before Q1. Record requested N, actual
selected row count and physical-ID digest for every arm (N300 can equal full on small tables).
Keep singleton/wide-table failures in the roster with per-arm availability, not silent deletion. Use the existing six-auditor
`v3_ledger_replay` retention R and RF feature-importance Spearman protocol, with paired generation
seeds 20261024 and 20261025. Average paired seeds per table before equal-weight dataset
aggregation. Publish numerator/denominator, valid/failed counts, mean/median and paired confidence
intervals. For nonpositive full-R baselines report absolute changes and mark relative loss
undefined, without silently dropping the table. The primary relative result is the ratio of
paired mean R over the same valid roster; include per-table ratios and regressions. Zero-mode
loss must be at most 8% (also report the 5% threshold); N300/full must reach 95% before either
claim passes. Every progressive transition, including macro+larger, needs a paired comparison.

Fast-mode success requires at least 5× versus genuine full evidence. Define each required
1k–50k GPU cold cell as p50(full public call)/p50(fast public call); a blanket suite pass requires
all those supported cells to meet 5×. Publish per-cell p95 and every failure, plus a descriptive
geometric mean; an aggregate mean must not hide failing cells. Small real fixtures and CPU
reference results remain separately labeled. Resident speedups are reported separately. Retention success
requires the actual existing transfer metrics, not landmark cosine, latent cosine or training
scores. Report uncertainty and paired changes; a progressive quality regression beyond noise
is a gap even if the implementation is deterministic.

Run focused suites first, then the required gates. Current JopeDime qualification records four
existing RustSec advisories also present on unchanged main and unchanged smartcluster Clippy
failures with the locked Tokio version. Preserve those exact failures; do not suppress them or
claim the aggregate gate passes. Coordinate their remediation with the owning maintenance
agent before promotion. Dope's earlier aggregate fast run likewise has a retained pilot-fixture
failure despite an isolated passing rerun. Neither result is an aggregate pass.

Expand the package harness beyond its initial none-mode fit: execute fit-predict, progressive
and sampled full terminal equality, mixed/missing/binary/multiclass input, unavailable GPU,
tampered package/weights, output alias refusals, and actual run/encoding receipt schema
validation. Dope also needs an actual CUDA raw-table CLI invocation checked against its resident
tensor reference; a default CPU engine gate cannot establish that CUDA behavior. These are
required acceptance cases even while the harness is incomplete.

Useful commands, run from each claimed lane with its normal reused target directory:

```bash
# JopeDime: focused actual GPU model/session coverage.
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only \
  cargo test --locked -p starforge --features dope-cuda --lib --test dope_encoding
# Consumer feature checks and actual packaged behavior.
cargo check --locked -p jope-cli -p jope-pipeline -p jope-appliance \
  --features jope-cli/dope-cuda,jope-pipeline/dope-cuda,jope-appliance/dope-cuda
bash deploy/appliance/build.sh --profile dope-gpu --out target/dope-appliance
bash deploy/appliance/test-dope-gpu.sh target/dope-appliance/jopedime target/dope-package-proof
just required

# Dope: contract/preparation plus Python reference, then repository aggregate gates.
cargo +1.88.0 test --manifest-path engine/Cargo.toml --locked \
  --test landmark_bundle --test prepare_parity_a --test prepare_parity_b
python -m pytest research/levels/tests/test_optional_landmarks.py -q
just engine
just fast
just check
```

The final linked draft PRs must contain concrete behavior, exact acceptance commands, measured
results, published dependency identity and remaining gaps. Canonical checkouts retain foreign
dirty work; integration proceeds through reviewed PR branches, without an automatic merge,
release, service rollout or promotion of unqualified HPC results.
