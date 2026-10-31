# RES-07 CPU pre-stage + model_prior subset

`generate.py` validates a RES-06 bundle and stages deterministic reference fixtures.
This directory currently contains generator source, documentation and tests, **no
qualification fixture set**. M2, M3 and M5 remain open. Owner passes `gb1841` and
`gb1915` admit only the `model_prior`-compatible, non-proposer subset. Actual fixture
generation/commit requires merged RES-06 and the exclusive
`~/jope_plan/qualification.lock`. H58/H57 have priority; never preempt their owner.

Validation uses exported SafeTensors and hash-verified RES-02 source definitions.
It does not import the historical runner, call the file-loading fused-network
constructor, unpickle checkpoints, read original host assets, fit a model, open
S3/held-out/sealed data, or initialize CUDA. Source definitions include the frozen
constructor's owned attention wrappers. Model layers are constructed on CPU and
strictly loaded from the already-exported tensors. No tensor/weight is changed.

Merged RES-06-F02 (#721) tags all 881 bank records `role=model_prior` and
`coordinate_space=standardized_lm48_l2_unit`. Validation requires both tags and
refuses proposer/raw-coordinate or unlabelled records. It never calls the exporter
to load legacy assets. The 881 rows/885 neural prior entries and normalized
summaries are distinct from the actual 1,219-context proposer bank's first twelve
**raw SVD lm48** coordinates. Its pinned raw snapshot/promotion prefix is absent
from admitted custody; RNK-09 retrieval parity stays parked. No inversion or
replacement bank is admitted by this generator.

## CPU validation

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  nice -n 10 python3 research/flywheel_selfplay/golden/generate.py \
  --bundle "$RES06_BUNDLE" --expected-bundle-digest "$RES06_BUNDLE_DIGEST"
```

The expected digest is an explicit trusted input, not an inferred authentication
claim. The report checks the existing schema, all four payload hashes/BundleDigest,
338 tensor contracts/finiteness/frozen composite/anchor aliases, source hashes and
all prior-bank `feat412` records. It reports prior-bank versus raw-FP64 projection
differences without replacing either representation. Stored FP32 bank vectors are
inputs; they are **not** the raw program-projection oracle. The validation report
sets qualification false and writes no fixture files.

The #721 tagged bundle used for this pre-stage is
`sha256:94fd9587675c7806bd0a0582b6e97540cd2fdc7ebbc51958411b905185cad85c`.
Its admitted subset is exactly 881 prior records / 872 unique DSLs. This does not
establish completeness of the historical training-program corpus. The validation
report explicitly sets `historical_corpus_complete` and
`proposer_retrieval_qualified` false.
Runtime/provenance receipts belong outside the Git tree. Use the same pinned
Python/PyTorch/NumPy/runtime and thread configuration for regeneration.

## Consumers and layout

A generated root has `index.json` binding the input capture manifest, BundleDigest,
reference hashes, counts, per-file byte length/SHA256 and exact tensor names,
dtypes and dimensions. `total_bytes` includes the index itself. Paths below are
relative to that root; `<context>` is the original 64-hex training-context ID.

| Consumer | Required fixture | Contract |
|---|---|---|
| RNK-01 | Original RES-06 bundle | Existing loader/schema/weight checks, not a new bundle format |
| RNK-02 | `programs.safetensors:proj16` | `[872,16]` F64, raw program projection for unique model-prior DSLs, absolute tolerance `1e-12` |
| RNK-03 / RNK-04 | `programs.safetensors:feat412`, `programs.jsonl` | `[872,412]` F32 in frozen name order; model-prior program subset only, count/flag exact, other values `1e-6` |
| RNK-05 | `landmarks/<context>.safetensors` | Actual training `input.X`, normalized `input.y`, source `input.row_ids`, native `col_full`, `v256`, caller-padded `col`; full trained HPC recipe |
| RNK-06 | `contexts/<context>.safetensors:lm48,lm48_z` | `[1,48]` F32 raw and standardized SVD coordinates, separately named |
| RNK-07 | `contexts/<context>.safetensors:tokens,mask` | `[1,10,32]`, `[1,10]` F32; exact source projection/masking |
| RNK-08 | `slates/<context>-12.safetensors`, `slates/<context>-20.safetensors` | All named `input.*` and `output.*` tensors, including total costs; four actual measured reveals plus 8/16 proposals |
| RNK-09 | PARKED; no retrieval fixture in this subset | Requires the separately pinned raw proposer snapshot/promotion epoch; `model-prior.jsonl` is not a substitute |
| DSC-07 | `landmark-pools.json` | Context-keyed ordered stage/head multisets from raw `lm48`; duplicates retained |
| RES-09 | All SafeTensors + index schemas | Use `JOPE_GOLDEN_DIR`; `JOPE_PARITY_STRICT=1` makes missing fixtures fail |
| M3 evaluator | `surrogate/<context>.json` | Original full-train-CV measurement identity, values, gates and failed/unknown outcomes; compare relative tolerance `1e-9` |

Each context tensor file also includes `v256:[1,256]`, `col:[1,64,17]` and
`dope629:[1,629]` F32. `model-prior.jsonl` preserves all 881 original prior records
and their order, including explicit role/coordinate tags; `programs.jsonl` indexes
their 872 unique exact DSL byte strings in first-occurrence order. This is the
only program population admitted for the subset. Program projections remain F64; slate inputs are the
frozen caller's FP32 tensors. RES-09's `values_f32` convenience map must not be used
for RNK-02's `1e-12` comparison; retain/read the F64 Candle tensor instead.

The frozen closed loop passes **zero base priors and zero SOS priors** to h40b;
those inputs are reproduced from its source. The separate fused GPU fitter's
calibrated-prior route is not substituted. Costs do not select new contexts,
change revealed slots, or define exploration admission. Rust admitted-set tests
apply the qualified policy outside the spec tie band
`2 * (1e-5 + 1e-4 * abs(cost))`; ties remain reported separately.

## Required source-capture packet

No training table is discovered automatically. The generation owner supplies a
reviewed, immutable `jope.golden-source-captures/v1` JSON manifest with a separately
pinned SHA256. Every referenced file uses a canonical relative path and SHA256;
symlinks/escapes, unknown contexts and identity mismatches refuse. No numeric
example/placeholder fixture is supplied here.

Required fields are:

- `bundle_digest`: the explicitly selected RES-06 BundleDigest.
- `selected`: hash-pinned row-free selection metadata, matching RES-01's original
  selection SHA256. Only its `train` context/family identifiers are used; dataset
  paths, development and outer-query payloads are never followed.
- `contexts`: at least 50 distinct pinned DL-Train contexts, each with `family_id`,
  `landmarks`, `surrogate` and `slates` keys `12` and `20`.

The program corpus comes directly from the hash-verified tagged model-prior bank.
`corpus`, `historical_inventory` and `proposer_bank` packet keys refuse in this
subset. Every slate program must belong to that prior population. Full historical
corpus coverage and raw retrieval goldens require a separate future source closure;
their absence does not block validation of this admitted subset.

A landmark capture retains the **actual function inputs**: F32 table input,
already-normalized target, distinct nonnegative source row IDs, at most 1,024 rows,
`col_full:[P,24]` native output, `v256` and `col:[64,17]` caller padding/truncation.
Raw missing input values may remain in the captured input; computed intermediates
must be finite. Its metadata pins the trained native workspace source, native
library and frozen caller hashes, and `outer_query_rows:0`. These are pre-existing
reference oracle captures, not values inferred from a Rust port or an approximate
PyTorch landmark extractor. The generator reuses their exact bytes and computes
the downstream neural reference stages from the pinned source/weights.

Surrogate captures contain `evaluator_sha256`, `context_sha256`, exact `recipe`,
and `measurements` keyed by program SHA with original `raw_f`, `gate` and execution
outcomes. Each slate supplies its ordered `program_sha256`, four `revealed_raw_f`
and four `revealed_gate` values; all four must match those measurement records.
Do not turn unknown/failed outcomes into zero, change selection from the frozen
caller, or replace full-training CV with a newly chosen surrogate. Capturing these
landmark/native-evaluator inputs remains a qualification prerequisite; CPU bundle
validation does not supply them. A subset index retains the false full-qualification
flags even when its particular outputs have been generated and checked.

## Gated generation and regeneration

Only the generation owner with the real free qualification lock may run:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  nice -n 10 python3 research/flywheel_selfplay/golden/generate.py \
  --bundle "$RES06_BUNDLE" --expected-bundle-digest "$RES06_BUNDLE_DIGEST" \
  --captures "$REVIEWED_CAPTURES" --expected-captures-sha256 "$CAPTURE_SHA256" \
  --output "$NEW_EXTERNAL_OUTPUT"
```

Generation refuses a busy lock, missing merged RES-06 ancestry, incomplete input
coverage and existing output. It publishes the non-proposer subset atomically into
a new directory. Check regeneration using two fresh external outputs, each holding
the required lease and using the same runtime; compare every file byte for byte and
read `index.total_bytes`.
No clock, execution-host path, random staging name or input dictionary insertion order enters
the output bytes. Validation remains CPU-only; CUDA parity is separately qualified.

At or below 50,000,000 total bytes, the qualification owner may stage verified
fixtures into Git while holding the required lease. Above that limit, retain the
payloads in immutable content-hash custody and commit only the reviewed index and
retrieval record. Materialize files from that exact approved location, verify every
index length/SHA256, then set `JOPE_GOLDEN_DIR` to the resulting root. No bucket
listing, credentials, upload permission or serving promotion is implied.
