# Train-only preparation controls

These standard-library Python tools implement the owner's October 5 code-only
packet. They perform local planning and verification. They do not invoke models,
launch jobs, control pools, call AWS, or grant execution. The existing DL pilot
and source-construction code remain owned by `JopeDime_DLTrain`.

Run the CPU conformance tests:

```sh
python3 deploy/dl-trainonly/test_contracts.py
```

The existing `just required` and hosted hygiene lane also run this command.
The synthetic test objects prove control behavior; they are not corpus, encoder,
GPU, distributed-training, or live S3 qualification.

## GPU budget gate

`gpu_gate.py` reserves worst-case GPU seconds in one persistent SQLite ledger.
The immutable policy allows xbabe1, xbabe2, xbabe3; it excludes xbabe0. The
planned total is 57,600 seconds (16 GPU-hours), the hard maximum 86,400 seconds
(24 GPU-hours), and the original deadline is `2026-10-10T00:00:00Z`.
Each host has one GPU, so a job's GPU seconds also bound its wall duration.

A job JSON contains exactly `job_id`, `host`, and positive integer
`gpu_seconds`. The requesting code must obtain a worst-case bound, including
startup, encoding/training, warmup, qualification and cleanup. All GPU work
shares the same ledger. Concurrent reservation is atomic; a changed retry is
refused. Repeated unchanged delivery is idempotent. Failed and cancelled jobs
retain their reserved charge; restarting never resets it. Actual overruns are
retained by `Budget.finish` and stop new reservations. Retain unknown attempts
and reconcile them before new work. The future runtime supervisor must enforce
the reserved wall limit and record actual elapsed GPU seconds.

```sh
python3 deploy/dl-trainonly/gpu_gate.py --ledger /custody/run/gpu.sqlite --job /custody/run/job.json
```

`--contingency` permits calculating reservations between 16 and 24 hours. It
requires separate execution admission to spend that contingency; it is not an
approval flag. Every result explicitly reports `execution_admitted=false`.
The caller still needs trusted grants, current allocation/device checks,
participant admission and the repository supervisor. Do not put this ledger
under disposable scratch, recreate it on recovery, or use one ledger per host.

## Output expectations and completeness

[Encoder pins](../../docs/internal/core-ip/dl-trainonly/encoder-pins.json) bind
six components: Hyperion v5_armB, Foundation, Target, context2048, TabPFN3.5,
and Dope attention-small. TabPFN3.5 is always research-only and denied to
production loaders; its embeddings are not cleared for competitive training.
Known historical exposure remains explicit. Pins alone do not qualify v2 ABI,
input scope, rights, numerical behavior or holdout independence.

The generator reads the hash-pinned source-receipt JSONL produced by Claude,
requires exactly 4,873 unique dataset/fold contexts, and emits **29,238** encoder
expectations (24,365 GPU and 4,873 CPU Dope outputs). A0 is additional CPU
metafeature evidence, not one of the six encoders, and is excluded here.
The prefix is fixed to
`s3://veox-jopedime-learning-use2/experiments/gp-policy-dl/v2-trainonly/`.
Each manifest requires create-only `If-None-Match: *`, with no unconditional
fallback. The local output also uses exclusive creation and refuses existing
files and symlinks. There is no storage client in these tools.

```sh
python3 deploy/dl-trainonly/outputs.py generate \
  --sources /custody/source-receipts.jsonl --sources-sha256 FULL_SOURCE_SHA256 \
  --pins docs/internal/core-ip/dl-trainonly/encoder-pins.json --pins-sha256 FULL_PINS_FILE_SHA256 \
  --out /custody/expected.json
```

Without resolved input bindings this is a draft with null final keys; the
checker cannot call it complete. `context_sha256` identifies the historical
source context, not a claim of v2 preprocessing qualification. To bind new
outputs, supply `--bindings /custody/bindings.json --bindings-sha256 FULL_SHA256`.
Bindings map each `view_id` to four full SHA256 values: `train_input_sha256`,
`scaler_sha256`, `code_sha256`, `recipe_sha256`. They must come from the admitted
train-only input producer, not arbitrary caller tags. The future admission path
must authenticate that producer and verify its input closure. This tool verifies
identity and custody only. Changed input, scaler, recipe or model bytes create
new output identities. Full hashes in keys avoid prefix collision reuse.

The completion file is a separately hash-pinned JSON array. Each terminal item
has `output_id` and `outcome` (`complete`, `failed`, `cancelled`, `unknown`).
Complete items additionally contain `object` and `receipt` references with
`bucket`, `key`, explicit `version_id`, `sha256`, and `bytes`. A separate trusted
publisher ledger must authenticate these references and supply their pinned
file hash. Giving a sidecar its own hash is not that authentication.

The receipt bytes are canonical JSON with exactly these fields:

```json
{
  "schema": "jope.dl-output-receipt/v1",
  "manifest_sha256": "SHA256_OF_CANONICAL_MANIFEST_CONTENT",
  "expectation": "THE_EXACT_EXPECTATION_OBJECT_FROM_THE_MANIFEST",
  "object": "THE_EXACT_OUTPUT_OBJECT_REFERENCE_FROM_THE_TRUSTED_LEDGER",
  "if_none_match": "*"
}
```

The strings standing in for objects above are documentation placeholders.
Store restored object and receipt bytes in a flat custody directory, with each
filename its full SHA256. The checker consumes the same opened bytes it hashes,
refuses blob symlinks, verifies sizes/digests, checks exact context/input/pin/key
bindings and immutable versions, rebuilds the entire expectation matrix, and
reports failed/unknown/missing/invalid coverage. No fallback or latest version
is accepted. Duplicate terminal IDs and unexpected outputs are errors; dedup
transport deliveries in the trusted ledger before producing the terminal view.

```sh
python3 deploy/dl-trainonly/outputs.py check \
  --manifest /custody/expected.json --manifest-sha256 FULL_MANIFEST_FILE_SHA256 \
  --completions /custody/completions.json --completions-sha256 FULL_COMPLETIONS_FILE_SHA256 \
  --blobs /custody/sha256
```

Exit 0 means complete offline pinned custody; 1 means incomplete; 2 means
refused/malformed inputs. The generator reports both the file digest (including
its trailing newline, used by `--manifest-sha256`) and canonical content digest
(used inside receipts). Exact-version remote GET, S3 create-only behavior,
authenticated production, array/ABI qualification and compute admission remain
separate live gates.

See the [C6 packet](../../docs/internal/core-ip/dl-trainonly/teacher-packet.md)
and [Dope CPU build receipt](../../docs/internal/core-ip/dl-trainonly/dope-cpu-build.json).
