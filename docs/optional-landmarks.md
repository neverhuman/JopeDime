# Optional and progressive Dope encodings

Explicit landmark options select a separate Dope512 product. With no landmark options, the
existing encoder, native fitting and search paths retain their behavior. Dope512 selects the
first 512 actual Dope629 coordinates: descriptor256, task latent128 and distribution latent128.
It uses its own recipe and never receives Hyperion's normalizer or accepted product identity.
Supplied, scoped foundation1024 and target512 components can form separate dataset1536 and
policy2048 products. Publishing these products does not reseed or rerank search.

Build the Rust GPU path with `--features dope-cuda`. Dope loads the whole checksummed released
SafeTensors model, keeps weights and neural intermediates on the selected device, and returns
device tensors. Host copies occur when an event is serialized. The API follows Micro HPC's
resident Stage 1A design; CUDA graph capture is not claimed for this implementation. CPU is an
explicit reference/debug option. CUDA selection never falls back to CPU.

`--landmark-mode none` and `fast` are equivalent. Both keep train-only target, shape and table
descriptors and skip native CV, spectrum and building-block probes. Progressive starts with
this immediate snapshot, then adds macro (1–6,14,15,22), larger (7–13,16,17) and fine
(18–21,23) evidence. Its default sample is 300 rows. Full uses all training rows unless
`--landmark-subsample N` is supplied. Sampling changes landmark evidence, not base descriptors.
Native landmark probes execute on CPU; their timings are separate from GPU inference.
Product evidence uses the checkpoint training work budget (2×10¹²), retaining legacy native
defaults for other consumers. Pipeline cuts share one native evidence worker, with one CPU
reserved in addition to the unchanged search worker count. Classification subsampling currently
requires at least two rows per class; a singleton class produces an explicit failed outcome
and preserves the initial snapshot.

The shared `LandmarkBundle` comes directly from the immutable Dope dependency. Values are
squashed once; masks distinguish measured zeros from missing observations. Nonfinite values
become positive zero and missing. Status derives from the sanitized mask. Scoped updates merge
valid observations and reject different sources/options. The explicit zero sentinel can start
a session. Full conditioning retains the original native report: a 23-value bundle cannot
reconstruct its confidence, spectrum and column metadata.

Sessions expose an initial snapshot and an ordered bounded channel. Version zero is the initial
encoding. Versions change only with the effective conditioning digest, including all 256
features and 64×17 column inputs. Duplicate observations are idempotent. New coverage that leaves effective conditioning
unchanged can emit a receipt update at the same embedding version. Cancellation or worker
failure retains the last valid snapshot and records the outcome. A terminal native computation
may still have partial coverage; coverage and completion describe different facts.
JSONL streams and collection callbacks also finish with that snapshot and its explicit terminal
outcome. This final metadata event reuses the last embedding version and does not rerun inference.

The CLI flags are shared by `jope encode`, pipeline search commands, and appliance fit and
fit-predict. `--landmark-events PATH` writes progressive JSON lines; `--landmark-receipt PATH`
selects the final companion receipt. Existing final stdout formats and pipeline receipt v3 are
preserved. The appliance run receipt can reference the companion by file hash. The snapshot
schema is [jope.landmark-encoding.v1](../schemas/jope.landmark-encoding.v1.schema.json).

```bash
cargo run -p jope-cli --features dope-cuda -- encode --input dataset.json \
  --landmark-mode progressive --landmark-subsample 300 --device cuda:0 \
  --dope-weights model.safetensors --landmark-events events.jsonl \
  --landmark-receipt encoding.json --json

bash deploy/appliance/build.sh --profile dope-gpu --out target/dope-appliance
bash deploy/appliance/test-dope-gpu.sh target/dope-appliance/jopedime target/dope-package-proof
```

The GPU appliance profile packages weights and their MIT license. It requires compatible host
CUDA libraries and an NVIDIA driver for GPU execution; it does not yet bundle CUDA toolkit
libraries for a driver-only host. See [amendment A16](../deploy/appliance/AMENDMENTS.md#a16--opt-in-dope-gpu-encoding-and-a-separate-package-profile).

Acceptance includes focused Dope API/session tests, CUDA reference parity, package smoke,
feature/dependency/schema checks and `just required`. Performance and retention qualification
must use genuine native measurements, fixed paired seeds and matched thread/device settings.
Micro's synthesized correlation reports and approximate GPU landmark extractor do not establish
native coverage or retention. No latency gate, retention threshold or sample-size target is
passed merely by implementing these APIs. Draft PR evidence records the executed measurements
and any remaining gaps.

The [coordinated completion and HPC plan](optional-landmarks-hpc-plan.md) names remaining
gaps, owners, integration order and acceptance evidence across both repositories.

Jepson’s binding mission addendum requires a newly trained mask-aware checkpoint and both
optional Jope consumer ablations. The original released-checkpoint path described above is
a compatibility baseline. Training may use Python; product inference remains entirely Rust.
The trained checkpoint must have its own pin/recipe, pass per-task mask-level evaluation, and
keep the live flywheel and its new regression test green. See the completion plan’s T1/T2/J1/F1/Q4 rows.

Explicit `--landmark-consumer direct` selects raw Dope512. Explicit
`--landmark-consumer jope-dl` selects a learned Jope context512 and requires
`--landmark-components requests.json` plus `--jope-policy-assets selected-policy/` (or a
reviewed sealed projection). Both selectors activate encoding; omitting the selector preserves
the existing landmark-only path. None/fast, progressive and full apply to either consumer.
The selectors propagate through raw encode, pipeline and appliance fit/fit-predict. The learned
output already passed through the context MLP. Its opaque `JopeContext512` can bridge to the
current Rust CPU decoder without another MLP pass; it cannot enter Dope fusion. Decoder and
reward execution remain CPU-only. Dope and the context MLP share the selected owned CUDA device.

The request file contains only FIT-scope, component subject, exposure-receipt and episode
references. Authority comes from the separately configured `JOPE_STATE` controller store and
its adopted replay policy; a request cannot supply its own issuer, policy pin, authority root or
admission Boolean. `ComponentInputs::load_with_controller` is a trusted host SDK adapter, not a
candidate-controlled selector. A controller owner must provide the existing authority snapshot;
`jope_learning::replay::admission::component_authority_snapshot` produces a read-only review
artifact and never installs it, adopts policy, repairs a ledger or signs a new grant.
No production controller installation or live flywheel authority is established by fixture tests.

`FusionComponentArtifact` binds original physical FIT rows, prepared source, fold, task,
preparation source, supported component recipe and producer ancestry. Foundation1024/target512
must already have their producer's projection/normalization; fusion applies no Hyperion
normalizer. The supported assembly recipe is
`foundation1024-target512-supplied-normalized/v1`. Component values use exact finite little-endian
f32 hex bytes. Every ancestor must be admitted for this episode's FIT-only training purpose;
protected metadata is refused before subject/target payloads open. Known prohibited model
payloads are also refused before opening. Unknown, capped or rounded multiclass vocabularies
are refused by explicit products until the new lossless preparation lineage is pinned. This
keeps the P1 requirement open; it does not establish large-class support.

Direct companion receipts expose `fit_scope` for controller producers. That scope remains
static across masks/subsamples of the same FIT preparation, while the landmark conditioning
scope changes with options/evidence. Replaying an artifact against another fold or physical row
roster fails. Admission, policy time, retained bytes and revocations are rechecked during
updates and final collection, including none/fast. Failure retains the last valid snapshot and
records a failed outcome. These guards add no retrieval, policy-training or global-use grant.

Product sessions load `VerifiedJopePolicyAssets`, then
`JopePolicyModel::load_policy_assets_on_device(assets, encoder.device().clone())`. The projection
opens only the pinned original manifest, four selected policy verification files and required
notice; the full-bundle loader remains strict. Content verification does not approve
redistribution. The [concrete asset review](qualification/optional-landmarks-policy-assets.md)
contains layout, exact checksums, draft SBOM and the release/model owner's unresolved decision.
No governed projection is added to an appliance release profile in this phase.

Consumer receipts bind the actual loaded Jope weight bytes and verified projection, Dope
checkpoint, component admission, physical FIT scope, effective conditioning and component float
bytes. CPU and CUDA context arithmetic have distinct recipes. Events contain both raw encoding
and explicitly selected consumer output; final stdout and pipeline receipt v3 stay unchanged.
See the [consumer receipt schema](../schemas/jope.dope-consumer.v1.schema.json),
[event schema](../schemas/jope.landmark-event.v1.schema.json),
[component subject schema](../schemas/jope.dope-fusion-components.v1.schema.json) and
[request schema](../schemas/jope.fusion-component-requests.v1.schema.json).
Public tensor exports use deep device copies because Candle tensors permit in-place writes through
shared references. Event/receipt serialization is an explicit host-copy boundary.

Caller-supplied checkpoint pins remain `UnqualifiedCandidateV1`. A checksum or a supplied training
receipt digest proves neither dropout training nor admitted ancestry. Register the trained
product only after reviewing the real optimizer/export manifest. The released checkpoint and
its fixtures remain separate. The current immutable Dope code pin is `25d1785c823e0a401b8775306666924e407ccd9f`,
published in [Dope draft #340](https://github.com/neverhuman/dope/pull/340).

Execution proof covers both consumers on regression, binary and multiclass inputs, sampled
progressive/full equality, CPU oracle compatibility and actual CUDA context/copy isolation.
These tests do not establish predictive quality. The governed Gen31 policy vocabulary supports
binary and regression; its multiclass context execution is an experimental surface requiring a
supported downstream readout and its own quality benchmark. Product selection is implemented; predictive consumer ablations, production component authority and governed package approval remain open. A trained-candidate development stage and real CUDA session proof are recorded in the [candidate qualification](qualification/optional-landmarks-dropout-candidate.md); the candidate remains unqualified. See the [qualification record](qualification/optional-landmarks.json).

The sealed loader also bounds asset-frame decompression before the encoder pool exists. Its
regression runs fresh Rust processes with thread environment overrides removed, across all
three task types:

```bash
python3 deploy/appliance/test-dope-cpu-threads.py \
  target/dope-appliance-published/jopedime target/dope-cpu-pool-proof
```

This is a process resource check, separate from the latency gate. The earlier sealed baseline
reproduced an unbounded global Rayon pool during asset loading; the qualification record retains
that failure and the fixed-package result separately.
