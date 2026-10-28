# JopeDime adoption of released Dope resident512

Measured 2026-10-28 against base JopeDime
`176bfe7ab7819eddf1121a2362d2d9222207edcf`, using the existing pinned
Dope `15e12aa3dc33851655580f6aa7c9c66353143f23`. This is a consumer optimization
of the released checkpoint; landmark training, model registration, retention,
and the mission's cold-start latency gate remain separate qualifications.

## Behavior

`DopeEncoder::encode_tensors` uses Dope's `encode_features_512_tensor` instead
of evaluating 629 coordinates and copying the first 512. It retains both-input
shape, dtype, batch and model-device validation. `encode_conditioning` uses
`encode_descriptor_512_tensor`, avoiding the unused 64×17 column allocation and
upload. The old host column padding/truncation behavior remains accepted.

The output is still descriptor256 + task128 + distribution128, on the selected
CPU or CUDA device. The full checksummed checkpoint is loaded, and CPU forwards
use the existing one-thread pool. Source and conditioning digests still include
the same evidence, including column metadata. Recipes, checkpoint identities,
preparation, frozen receipts, Hyperion assembly, sessions and search are unchanged.
No identity fixture was regenerated: the 38 frozen evaluator inputs exclude this
adapter and its test support, and their bytes are unchanged.

## Compatibility evidence

The independently invoked full629 oracle and product use the same released D0
weights, SHA256
`d0b6960328d1955386632ff78b512e3add53a9b0256db7bbf067e2e0be5f0c6c`.
Tests compare `f32::to_bits` on CPU and actual CUDA for batches 1, 8, 64 and 256.
They cover malformed shape/dtype/batch/device rejection, positive-zero sanitation
of NaN and both infinities, signed zero, unused dirty columns, retained A/B/A
outputs and mutation independence. Host tests exercise actual none-mode
conditioning for regression, binary and multiclass, with 0/3/64/70 column rows,
unchanged digest padding/truncation and column-sensitive identities.

Both CPU tests and both strict CUDA tests passed. The retained Python/CUDA fixture
passed with maximum absolute error `6.482005119e-7`, below its `3e-5` tolerance.
The existing feature suite also exercises actual native progressive/full terminal
equality and session behavior; synthesized Micro reports do not establish native
coverage. Governed-policy tests remain explicitly excluded without their assets.

## Matched resident measurements

Hardware: AMD Ryzen Threadripper PRO 3995WX and NVIDIA RTX 3090 (24 GiB).
Rust 1.97.1, existing `dist-ci` profile (numerical code at optimization level 3,
thin LTO), same compiled `dope-cuda` test binary for both device selections.
CPU measurements pin core 16, with one forward-pool/math thread and nice 15.
CUDA uses cores 16–21, host math one thread and no CPU forward pool. GPU work was
serialized by the shared qualification lock after an empty compute-PID check;
the completed window was below 90 seconds, with no remaining compute processes.
The highest sampled device memory was 350 MiB; sampling does not establish the
true allocation peak. No persistent cache or mutable scratch state was added.

Each case has three warmup pairs and 21 alternating paired trials. The timer
excludes pre-synchronization and includes the call and completion synchronization.
Weights, model loading, input preparation and correctness readbacks are excluded.
The baseline reproduces the old full629 + crop implementation; the host case
also includes the old packing/upload and identical metadata/digest binding.
This is a retained source oracle, not a separately built historical executable.
The old-adapter diagnostic measured ratios 0.991–1.004 in the unoptimized test
profile; it only checks workload comparability and is not performance evidence.

CPU, optimized and GPU hidden:

| Call | Batch | Full629 p50 / p95 (µs) | Resident512 p50 / p95 (µs) | p50 ratio |
| --- | ---: | ---: | ---: | ---: |
| resident | 1 | 148.652 / 171.595 | 88.819 / 106.823 | 1.67× |
| resident | 8 | 641.234 / 655.923 | 332.990 / 344.051 | 1.93× |
| resident | 64 | 7656.470 / 7904.431 | 1882.646 / 2102.613 | 4.07× |
| resident | 256 | 41777.674 / 48064.129 | 6654.654 / 7770.297 | 6.28× |
| host-conditioning-public-call | 1 | 153.291 / 161.386 | 90.271 / 101.953 | 1.70× |

CUDA, optimized and synchronized:

| Call | Batch | Full629 p50 / p95 (µs) | Resident512 p50 / p95 (µs) | p50 ratio |
| --- | ---: | ---: | ---: | ---: |
| resident | 1 | 941.423 / 974.285 | 516.578 / 538.089 | 1.82× |
| resident | 8 | 1152.453 / 1192.157 | 603.583 / 635.864 | 1.91× |
| resident | 64 | 1211.104 / 1326.873 | 660.521 / 741.955 | 1.83× |
| resident | 256 | 1192.338 / 1272.410 | 610.676 / 642.517 | 1.95× |
| host-conditioning-public-call | 1 | 950.690 / 1009.802 | 520.486 / 582.122 | 1.83× |

These resident/public-call measurements do not include native landmarks, raw-table
preparation, process startup or weight loading. They establish neither the 5×
none/full cold-start mission gate nor retention/quality improvement. GPU dispatch
cost also means a single-row CPU call is faster here; this does not change an
explicit caller's device choice. Larger real-table cold/resident comparisons and
mask/subsample retention ablations remain open. No serving promotion, training,
release publication, candidate registration or E03 work is part of this change.

## Reproduction and provenance

CPU feature and frozen comparison gates:

```sh
cargo test --locked -p starforge --features dope --lib --test dope_encoding
cargo test --locked -p feat-math --test compat_freeze --test historical_v1
cargo clippy --locked -p starforge --features dope-cuda --all-targets -- -D warnings
cargo test --locked --profile dist-ci -p jope-appliance --features dope-cuda --test single_file
just required
```

Build the optimized SDK test executable before reserving GPU hardware:

```sh
cargo test --locked --profile dist-ci -p starforge --features dope-cuda \
  --test dope_encoding --no-run --message-format=json > target/resident512-build.jsonl
sdk_test=$(jq -r 'select(.reason == "compiler-artifact" and .target.name == "dope_encoding") | .executable // empty' target/resident512-build.jsonl)
CUDA_VISIBLE_DEVICES='' "$sdk_test" resident512::released_cpu --test-threads=1
CUDA_VISIBLE_DEVICES='' JOPE_DOPE_BENCH_DEVICE=cpu taskset -c 16 "$sdk_test" \
  --exact resident512::benchmark_released_resident512_matched --ignored --nocapture
```

Under the host's existing GPU reservation/budget, select device 0 explicitly and
run the already compiled executable with `resident512::released_cuda`, the exact
`actual_cuda_projection_stays_on_device_and_matches_whole_model_reference` test,
and the same ignored benchmark with `JOPE_DOPE_BENCH_DEVICE=cuda`. CUDA tests
fail when the requested hardware is unavailable; there is no CPU fallback.
CUDA CI remains disabled. Set the repository's required build/math/thread limits;
reuse the lane's target directory and keep temporary output outside `/tmp`.

Raw sample arrays and command receipts were retained under the qualification
owner's `target/resident512-20261028`. The PR handoff must include a durable copy
of these records. Test binary SHA256:
`8bd95f3dc5f4c201675797564ba98da1716676a455b60e0c7bab283141b1a576`.
CPU sample-log SHA256:
`dbe4662831dc7703f5d3e8082cfc6480490f211fc635c75ca33d6eb4620f6517`.
CUDA sample-log SHA256:
`00ed9fb77395af89ed2bb89435afe66cfb22318926d969e537832afd630967d6`.
CUDA window receipt SHA256:
`ad21b64fc8612f1438aae1736da076015dd2b00d96e7bfffb9808f0335ed4897`.
Adapter source SHA256:
`dbeaf2f3538b3f7adbdfe6d98474e6c121804f26b2f8f2b8ea7af92e0a3bfc91`.
Final source-bound gate results, sealed-package execution, reviews and hosted CI
are recorded with the PR; these component results do not substitute for them.
