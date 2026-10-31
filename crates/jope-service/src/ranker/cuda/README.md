# Frozen fused forward and manual RNK-12 M2

`FusedForward::from_bundle` accepts only the existing authenticated bundle type.
It loads the frozen high-level DOPE router/column gates, lm48/Stage-1B projections
and the existing adapter on one explicit Candle device. `placement()` reports
those instantiated stages and CPU evaluation/workers. `auto` chooses CPU;
explicit CUDA requires `--features fused-cuda` and actual device initialization,
otherwise it fails with `BACKEND_UNAVAILABLE`. This does not enable native
evaluation on CUDA or activate learned admission/serving/Q1 defaults.

Dense, attention and SVD operations use the selected device. Small scalar recipe
operations (including lm80 order statistics, adapter semantic kernels and final
admission) remain CPU work and incur transfers. This is a parity implementation,
not a claim of a sync-free CUDA graph or latency qualification.

The two real tests are ignored by default. Missing fixture inputs are errors.
Use the admitted RES-07 DL-Train packet (50 contexts, both12/20 slates), its
explicit trusted index SHA, and the original RES-06 bundle. No recapture,
surrogate fit, model mutation, checkpoint unpickling or optimizer is performed.
The test-only byte constructor hashes the original weights and the shared strict
fixture helper verifies every consumed payload/schema. Runtime production bytes
remain behind the existing authenticated loader.

CPU replay, with `JOPE_SOURCE_SHA` set to the complete tested source commit:

```sh
JOPE_GOLDEN_DIR=/path/to/res07/generation-1 \
JOPE_GOLDEN_INDEX_SHA256=27905f29703431bedacac90eed1bd54257fd0c59bdf24b56605d0b9416ae1393 \
JOPE_PARITY_BUNDLE_DIR=/path/to/res06 \
JOPE_PARITY_REPORT_DIR=/path/to/external/cpu-report \
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
nice -n 10 cargo test --locked -p jope-service --lib -j8 \
  ranker::cuda::fixture::real_full_forward_cpu_fixture_replay -- --ignored --exact --nocapture
```

The GPU command is authorized only after actual source merge, H99 seed38 release
and the existing queue/resource/driver checks. Set the same fixture/source
variables and a fresh external report directory. The inherited stdin descriptor
holds the existing exclusive flock throughout Cargo and the complete GPU test;
the test checks its inode and exclusive ownership without reacquiring a new open
file description. No outer second lock is needed or permitted inside the test.

```sh
(
  exec 0<> "$HOME/jope_plan/qualification.lock"
  flock --exclusive --nonblock 0 || exit 1
  export JOPE_GPU_LEASE_STDIN=1
  export JOPE_QUALIFICATION_LOCK="$HOME/jope_plan/qualification.lock"
  timeout --signal=TERM --kill-after=5s 300s \
    nice -n 10 cargo test --locked -p jope-service --features fused-cuda \
      --lib -j8 ranker::cuda::fixture::real_cuda_m2_fixture_replay \
      -- --ignored --exact --nocapture
)
```

Preserve `RNK-12.json` and the command/exit/device UUID/driver/source/fixture
readbacks outside the checkout, and append PID/SHA to `gpu_chain_t1015.log`.
The test compares every required stage and all100cost slates, retaining tie counts
separately from admission agreement. M2 is open until this actual CUDA invocation
passes on the declared GPU and driver. A CPU pass or compiled feature is not M2.

CPU replay uses the frozen reference's deterministic f32 reduction boundaries:
eight-lane Welford/cascade moments with explicit FMA, ordered dense/attention
products, reference exp for SiLU/softmax, and f64 tanh rounded once to f32.
Attention scales by the f32 square-root divisor. Keeping these boundaries matters
because frozen SVD scalers amplify one-ULP upstream differences; promoting the
whole graph or only SVD does not reproduce the frozen model. The small scalar
exp port is attributed to SLEEF 3.6.1 with its Boost license alongside the code.
CUDA tensors continue through device-native Candle kernels. The CPU arithmetic
regressions and real CPU replay do not establish GPU M2 or serving promotion.
