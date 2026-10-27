# Optional landmarks and HPC: coordinated completion plan

Execution record: 27 October 2026. The [engineering specification](specs/optional-landmarks/engineering-spec-optional-landmarks.md)
and [mission](specs/optional-landmarks/mission-optional-landmarks.md) govern correctness. This plan includes
the user's subsequent requests for GPU Dope features, Micro HPC reuse, and coordinated upgrades
in both repositories. The attached source documents preserve their wording; trailing Markdown
spaces are normalized for repository hygiene. Implementation and qualification are separate milestones.

**Binding update:** Jepson’s ADDENDUM in the mission supersedes this plan’s earlier no-training
assumption. The current old-checkpoint implementation is a compatibility baseline. Completion
now additionally requires trained mask robustness, both optional Jope consumption paths,
flywheel regression/liveness, per-task quality, and entirely Rust inference.

## Ownership and coordination

Codex owns cross-repository acceptance and the JopeDime optional-landmarks checkout. Claude-Code is the sole writer/pusher of Dope PR #340; Claude/AGY owns the separate training source checkout. Claude-Code also owns native quality/performance qualification. Gemini
owns proposed preparation/kernel optimizations in its own claimed checkout. The separate Codex
Micro-import owner retains the prepared JopeDime import packet; the optional-landmarks writer
will not independently apply that overlapping packet. Read-only reviewers inspect shared
contracts and lifecycle behavior. Claude-Code acknowledged Q1/Q2 ownership in the shared log at
14:32Z; the full quality study follows a validated three-table smoke run. Gemini acknowledged H1/G1 preparation work at 14:36Z; Claude has committed the named training
options and resident CLI fix at `82cea8c48f9194322a6c4f71ae7cea4d2202b986`. The Micro import
owner retains its checkout handoff dependency.

Use append-only `jope_plan/comms.md` for contract decisions, claims, exact commits, benchmark
reservations and stopped-head handoffs. JopeDime claims additionally use `AGENT_CHAT.md` under
its lock. A proposed assignment becomes active only when its owner acknowledges it. Each
checkout has one writer. Reuse permitted lanes, respect their limits, and preserve existing
work. Share immutable commits/patch hashes, not mutable local dependencies. No new standalone
Micro experiments are required: integration and new qualification happen in Dope/JopeDime.

The dependency sequence is:

1. Close contract/lifecycle defects and freeze shared fixtures.
2. Train and qualify one mask-aware Dope checkpoint using training-only data; preserve the old
   checkpoint as a separately identified compatibility baseline.
3. Export safetensors, verify Python/Rust CPU/CUDA parity, evaluate both optional Jope consumer
   paths, and prove the flywheel remains live and passes its regression test.
4. Publish Dope draft, pin its published immutable code/checkpoint identity in JopeDime, then
   publish the linked JopeDime draft with package, compatibility and per-task quality evidence.
5. Integrate focused HPC changes through their existing owners, rerun the same acceptance matrix,
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
| C2 — Claude | Expose named MLC-v3 training options with work budget 2e12; leave legacy Default unchanged; use named options in both product paths | Options digest and budget in receipts; real/wide tables demonstrate genuine native coverage; legacy full bytes unchanged | Dope helper/CLI verified at 82cea8c4; Jope adopts helper on published `972685cd` |
| C3 — Codex | One admitted native CPU shared across pipeline cuts; cancellation-aware sessions; bounded updates; join all workers on output failure | Initial event precedes worker completion, duplicate idempotency, sibling-join failure test, CPU-ceiling admission test, cancellation/panic preserving last snapshot | Implemented; final rerun pending |
| C4 — Codex | Finish CLI/appliance output protection, error mapping, receipt reference and actual generated schema validation | Weights/input/output alias refusals preserve bytes; event-output errors retain output failure class; generated run and encoding receipts validate | Expanded sealed baseline package proof passed; trained-candidate package rerun pending |
| C5 — Claude + Codex | Fix Dope raw-table CUDA CLI to call resident tensor forward and bound its explicit CPU reference pool; published immutable Dope pin; reviewed Candle source reconciliation; released whole checkpoint and license packaged | Actual RTX 3090 numerical parity and residency, CPU-only and CUDA feature builds, dependency fences/security provenance, sealed appliance fit/fit-predict with no runtime download | Jope pins published `972685cd`; baseline package passed; trained candidate/new final package pending |
| C6 — Codex | Qualify GPU runtime portability separately from weight packaging; retain documented host-library requirement until supported CUDA runtime libraries are packaged with reviewed licenses/loader paths | Clean driver-only host/container, offline startup and GPU fit/fit-predict, tamper refusal, library/driver compatibility matrix and SBOM; existing native-only package remains portable | Open; current GPU profile requires host CUDA libraries |
| Q1 — Claude-Code | Real native retention and feature-importance comparison on the available 112 development datasets | Zero, macro, macro+larger, full, N=100/200/300/500/1000; fixed paired seeds; dataset exclusions/errors and per-dataset metrics; zero relative retention loss, progressive changes, N300/full ratio, smallest tested N meeting 95% | Trained candidate paired stage:102complete/112requested,10wideMC unavailable; none/full point ratios .9638 regression/1.0528 binary/.9950MC, with unresolved uncertainty. Parallel all-arm wide-table metrics invalid; owner repair pending. Existing policy unqualified |
| Q2 — Claude-Code + Codex | Matched latency matrix on real fixtures and synthetic 1k/10k/50k tables | Real fixtures plus synthetic 1k/10k/50k × p10/64/200; ≥15 repeats; p50/p95 public encode time, native evidence separately, full preparation included; CPU and CUDA weight-cold/resident timings, thread/device/commit/checkpoint identities and resource sampling | Old baseline CUDA cold 7/7 measured synthetic cells FAIL 5× (2.0–4.9×); eight CUDA cells unmeasured/quarantined; post-H1/training rerun required |
| Q3 — both integration owners | Full required suites and compatibility audit on final heads | Dope engine/fast/check gates; Jope `just required`, relevant feature combinations, historical receipts, checkpoint loading, frozen Hyperion assembly and Python/Rust tolerances | Focused suites pass; full gates have explicit unresolved failures |

Do not silently change the N=300 default in response to an ablation. Report the smallest tested
N meeting the target and distinguish it from this version's agreed default. The binding ADDENDUM authorizes and requires mask-aware training; this supersedes the earlier
no-training restriction. Train against the actual optional-conditioning recipe, not a leaked
full-report feature vector with only its first46 entries masked. Singleton-class sampling currently fails explicitly;
preserve the last snapshot and document that restriction rather than silently switching modes.

## Binding addendum phase: trained optional landmarks and live consumers

| ID / owner | Implementation | Acceptance evidence |
| --- | --- | --- |
| T1 — Claude, Dope training writer (ACK 16:10Z) | Fine-tune the released MLC-v3 encoder on an authenticated training-only roster with deterministic task-balanced none, progressive-group, random-partial and full masks. Rebuild gains/masks/ranks/contrasts/interactions/confidence/column/spectrum/building-block availability exactly as product conditioning permits. Keep the full input dimension and output layout fixed. | Frozen roster/exposure hash, seed/schedule/configuration, source/checkpoint identity, real optimizer steps/losses; tests proving withheld evidence cannot leak through derived inputs. The112 development roster and outer/sealed families are evaluation-only. |
| T2 — Claude + Codex | Export one mask-aware safetensors checkpoint and a checksummed model/recipe manifest. Every mask mode uses that same pin. Preserve the historical d0 checkpoint and its legacy recipe/tests instead of overwriting them. | Strict Rust load of expected tensors/dtypes/shapes; finite CPU/CUDA none/partial/full output; Python/Rust tolerances for the trained weights; fixed512/629 dimensions; no Python process, LibTorch or runtime weight download in product inference. New trained weights have a new identity; historical full outputs remain frozen under the historical identity. |
| J1 — Codex, Jope product writer | Implement and keep optional both consumers: Dope encoding directly through the product encoding/fusion surface, and the same encoding into JopeDime’s own DL model. Bind input recipe/checkpoint/mask scope; keep accepted legacy defaults available. | Run and document matched ablations: legacy baseline, Dope-only, Dope plus Jope DL; same splits/seeds/budgets and per-task downstream metrics. Actual model invocation/learning receipts are required; tensor assembly alone does not close this row. |
| F1 — Codex + current flywheel owner | Establish current live flywheel health without interrupting its services. Add a regression test exercising the production self-play/data path with the optional encoder, updates and pinned identity, and run it for every PR head. | Existing rounds continue, checkpoint/receipt provenance remains valid, no poisoned/held-out rows enter training, none/partial/full paths stay finite and dimensionally compatible, checkpoint reload/resume works. Record live observations separately from unit-test success. A green historical service receipt does not establish current health. |
| Q4 — Claude-Code | Repeat quality and latency qualification with the trained candidate and both optional consumers; report each mask level and regression/binary/multiclass separately. | Existing task-appropriate metrics/baselines, paired seeds/CIs and explicit regressions, per-task fast latency test, N ablation and feature importance. Old-checkpoint Q1/Q2 remain baseline evidence. “World-class” is not claimed from plumbing or pooled averages. |

Training/research may use Python and CUDA; **product inference is100% Rust**. This includes
conditioning/preparation, Dope forward, progressive updates and optional Jope DL inference.
The new training mandate does not authorize a Python production fallback, a live service
restart, training on development/outer-test families, or silently promoting a checkpoint.
Coordinate GPU0 reservations before any training run. Start with a small reproducible gradient
smoke, then run the declared balanced schedule and retain the full learning/evaluation receipts.

The training admission roster has 2,199 gradient datasets (1,623 regression, 378 binary, 198
multiclass). Fresh manifests contain 2,001 reg/bin and 197 valid multiclass partitions; one
multiclass table has only one FIT class and is explicitly excluded. Filter the pinned roster
before opening any blob. Never deserialize the mixed-role tensor cache or full report archive
in the trainer. Preserve full FIT separately from capped optimizer episodes; static descriptors
and class vocabulary use complete FIT, and native/BB evidence uses the declared sample. Bind
report hashes to exact physical FIT/query IDs. Cache column conditioning per level; keep full
measured auxiliary targets separate from masked inputs. A functional loader/one-update smoke
must prove forbidden-open sentinels, finite gradients, positive real updates, active encoder
parameter changes and accurate task×view counters. Syntax checks cannot establish these facts.

J1's Rust SDK execution proof now passes both consumers on all three task inputs with released and trained-candidate weights. The candidate also executes native progressive sessions and exact terminal/full equality on RTX3090. Its real CUDA context kernels differ from the scalar CPU oracle by at most 7.40e-6 at tolerance 3e-5. Resident product fields are opaque, public exports copy
device storage, and mutation-isolation tests pass. Caller-supplied pins are explicitly
unqualified. Predictive ablation and CLI/appliance consumer selection still remain.

For the consumer ablation, reuse outcome-free frozen program views and actual policy reward
inference for a shadow TRAIN-pool diagnostic. The existing external-score harness records
provenance but does not authenticate it; require the context-to-FIT mapping, native/checkpoint
ancestry, evaluator identity and exact 1103-token vocabulary. Do not relabel development data
to satisfy the TRAIN-only pool guard. Actual development prediction needs a fixed program bank
and same-cut evaluator where each arm selects its program from FIT-only inputs, then refits and
scores untouched rows. A and B compare complete consumer recipes; B additionally uses
foundation/target components, so add a component-matched control before attributing a difference
solely to its MLP. The governed Gen31 policy supports binary/regression only. Multiclass needs
a genuinely supported downstream readout; finite context execution cannot qualify that task.
Do not replace the flywheel's target-free ID features with global Dope vectors: any learned
provider must fit inside the declared outer fold and bind its training provenance.

F1 has 26 passing admission/snapshot regression tests, including nine frozen real Rust outputs
(three tasks × none/macro/terminal), protected-ancestor refusal, missing labels, duplicate/pin
rules and unchanged authority/remote state on incompatible updates. These are contract proofs.
`LocalSnapshotAuthority` and `SnapshotEvidenceVerifier` currently have only test implementations;
a production evidence verifier/store binding and fresh deployed PID/runtime/heartbeat/progress
observations are still required. Gemini confirmed zero deployed Rust flywheel/appliance processes on this host at17:19UTC. Retired Python units and stale healthy receipts cannot satisfy the live-green requirement. Keep existing services untouched while the owner supplies the deployment boundary.

The first candidate has completed a safe paired development stage:112requested,102complete,10wideMC explicitly unavailable. None/full point retention is .9638 regression,1.0528 binary and .9950 multiclass, but confidence intervals do not establish the target across tasks. The independent102cohort exactly matches1,224 common rows from the parallel study; that study's extra wideMC values need quarantine because synthesis/evaluation widths differ and failures were converted to chance scores. Training source custody and lossless class-preparation lineage also remain open. See [the trained-candidate evidence and ordered owner handoff](qualification/optional-landmarks-dropout-candidate.md). Do not promote from pooled retention or tensor parity.

## HPC integration waves

Prioritize elapsed public-call cost and CPU work. Port as many applicable improvements as pass
parity, resource and end-to-end benefit checks; component speed alone does not qualify a port.
Micro's correlation-based landmark reports cannot substitute for genuine native CV evidence.
Dope PR [#339](https://github.com/neverhuman/dope/pull/339) is closed and its native ridge/basis work has been folded into [#340](https://github.com/neverhuman/dope/pull/340). Claude-Code remains that PR's sole writer. Gemini's phase2 port `3793ba31a3dc93a138ccee4234b3a4f9ec755ed5` is available on the split baseline25d1785c with75 focused tests/Clippy and exact conditioning goldens passing; it is not yet the published Jope dependency. Component gains still require a combined-head timing rerun before product attribution.

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
shared runtime/feature changes before merging any integration branches. The packet currently includes a copied Python `MultiLevelLandmarkModel`
and stat-keyed hash helpers. Preserve them as historical/reproduction evidence if needed, but
route new product execution through the immutable Dope dependency and shared bundle contract.
Do not introduce a second live model implementation or replace strict checkpoint verification
with the historical stat cache.

The current baseline publishes compatible-size Dope/fusion tensors; the pretrained policy has
not been qualified on their changed distribution. Call that baseline plumbing in PRs. The
binding addendum now requires the optional actual DL-consumer ablation (J1); do not call it
complete until the model executes and its task-specific quality is measured.

A C++ native kernel backend and C++17/20 generated-program execution are separate surfaces.
Pin C++ compiler/libm/ISA and strict floating-point flags (`-ffp-contract=off`, no fast-math
or reassociated reductions), row/tie order and bounded scratch. No C++ exceptions cross FFI.
Any new native FFI requires its own opt-in feature, ABI/ownership contract, sanitizer results,
package/dependency policy and SBOM. Do not feature-unify `fitted-cpp` into the native appliance.
Preserve its existing pure-Rust fitting closure.

Keep full checkpoint SHA verification on every load. Existing measured hashing cost is small
for this checkpoint; resident sessions already amortize it. Do not introduce a stat-only trust
cache. Preserve FP32 inference and exact CPU arithmetic for compatibility-preserving waves;
FP16/TF32/fast-math, alternate solvers and synthesized evidence need separate accuracy/quality
decisions. The addendum requires evaluating optional neural consumers; retain legacy defaults
and use separate identities/configuration for those experiments.

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
skips only output heads unused by the first 512 coordinates. Retain the complete checkpoint and
legacy629 APIs. Require bit-identical first 512 equality at B1/8/64/256 on CPU/CUDA and measured
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
intervals, separately for regression, binary and multiclass as well as pooled. For nonpositive full-R baselines report absolute changes and mark relative loss
undefined, without silently dropping the table. The primary relative result is the ratio of
paired mean R over the same valid roster; a pooled pass must not hide a failing task; include per-table ratios and regressions. Zero-mode
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

Run focused suites first, then the required gates. Current JopeDime qualification retains four existing RustSec advisories also present on unchanged main and unchanged smartcluster Clippy failures with the locked Tokio version. A separately reviewable repair in Codex's own lane upgrades only the packer zstd graph and migrates two encapsulated ownedFile registrations to Tokio's sound API. Keep global unsafe-code denial with narrowly documented registration exceptions. Preserve prior failures and require fresh security/fence/I/O-safety review and aggregate gates; source edits alone do not establish a pass. Dope's earlier aggregate fast run likewise has a retained pilot-fixture
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

## Candidate and package handoff, 27 October 2026

The first dropout candidate is `ecded0225fca7c6e13c413b0e39b1040bf95f7694940b2809f1ce2ce2abf7bd9`.
Its receipt records eight epochs, 360 optimizer steps and 714 training cells (319 regression,
199 binary, 196 multiclass), with 495 sampled native arms. Independent raw-file comparison
confirms all 40 encoder-active tensors changed, among 56 finite FP32 tensors/444,168 parameters.
This is an initial capped training run, not exposure to every eligible training table.

Rust CPU and actual RTX 3090 full 629/first 512 numerical comparisons both pass against the Python
CPU reference at maximum absolute error 1.91e-6 (tolerance 3e-5). Both Rust consumers also pass
none/partial/terminal execution and terminal/full equality for the candidate on three tasks.
These are numerical and execution results; the synthetic operator fixture does not establish
native coverage or retained quality. The candidate remains explicitly unqualified pending Q4.

The currently published Dope972 preparation retains its 32-class cap. Preserve that released
behavior and its old goldens, and introduce a separately identified lossless FIT-class policy
for the new product rather than silently changing historical full outputs. It must pass both
backends for K3/32/33, fractional/nearby labels, label permutations and unknown labels. The
training source archive also needs exact byte custody: the integration commit includes later
trainer/test fixes and differs from two pre-run file hashes; pin imported conditioning helpers
as well. Do not relabel an integration commit as the exact original training source.

Sealed package source `1c05c9e1de83b792c162bd95001bb36ef2eea819` passes the expanded actual GPU
smoke and packaging checks with the released checkpoint. A fresh-process CPU probe found and
fixed global Rayon initialization in asset-frame decoding before the Dope pool existed. With
thread environment overrides removed and CPU affinity 16–23, fixed fast-mode peaks are 4/3/4
threads across regression/binary/multiclass; the old package reached 10 under that affinity and
104–105 with ordinary host affinity. Paired regression encodings remain bit-identical. This is
resource evidence, separate from the unmet 5× cold latency gate.
