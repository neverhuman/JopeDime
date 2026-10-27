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
