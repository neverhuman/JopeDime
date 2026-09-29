# Lambda GP experiment storage contract

This is an isolated diagnostic archive, not the canonical JopeDime learning store.
The files remain in the ignored experiment area until a reviewed product integration.
All source lives under /home/ubuntu/JopeDime/.agent/experiments/lambda-csv/gp.

## Identity and layout

Archive prefix:
s3://veox-jopedime-learning-use2/experiments/lambda-gp/v1/campaigns/<campaign-id>/

- objects/successes/sha256/<sha256>.jsonl.gz: complete successful fit records, bounded champion model, program, predictions, selected original feature indices, fold row identities, metrics and profiling.
- objects/failures/sha256/<sha256>.jsonl.gz: explicit job failures; no fabricated metrics.
- objects/attempts/sha256/<sha256>.jsonl.gz: invocation admission, request ID, actual or reserved cost, timeout and resource observations.
- objects/coverage/sha256/<sha256>.jsonl.gz: one terminal disposition per frozen census entry, including sealed, structurally unsupported and projection-blocked entries.
- objects/dataset-index/sha256/<sha256>.jsonl.gz: dataset/job/attempt to shard path, row index and canonical row SHA256. This avoids one S3 object per dataset.
- objects/source/sha256/... and objects/protocol/sha256/...: source snapshots and frozen protocol/input manifests when supplied by the controller.
- manifests/<sha256>.json: exact dependency closure, counts, schema and campaign protocol.
- COMMIT.json: create-only publication marker, names the manifest version/hash and aggregate object count/bytes.

The local mirror has the same object paths plus local upload receipts keyed by SHA256 of the full S3 key, and manifest-ref.json. Mirror receipts are transport evidence, not additional scientific results. A campaign ID must be unique. A resumed identical campaign reuses the exact ID and manifest; a later scientific run receives a new campaign ID. A changed event is never written over the earlier event.

Keep dataset identity, evaluation identity and attempt identity separate. Runtime integer job IDs index the compiled frozen manifest; input_identity_sha256 names the pinned dataset/CSV version/task/manifest. The controller's semantic job identity should additionally bind program/search configuration, evaluator/source digest, fold recipe, preprocessing recipe and seed. Attempt IDs bind actual invocations/retries without making duplicate attempts appear as independent data.

## API and commands

Python controller API:

    archive = Archive(mirror_path, campaign_id)
    refs, index = archive.stage_records("successes", rows, shard_rows=100)
    index_refs, _ = archive.stage_records("dataset-index", index)
    manifest_ref = archive.manifest(frozen_campaign_metadata, refs + index_refs)
    commit = archive.publish(s3_client, manifest_ref)

Rows are ordinary dictionaries; preserve the complete original runtime record. Cost is attached once per invocation in attempts; any per-job cost allocation must be labelled allocated, not measured. Stage failures separately, including all retries. Do not drop failed or excluded denominator rows from coverage.

CLI preparation performs local writes only:

    python3 store.py stage --campaign-id ID --mirror MIRROR --metadata campaign.json --records successes=successes.jsonl --records failures=failures.jsonl --records attempts=attempts.jsonl --records coverage=coverage.jsonl

The controller alone executes the authorized AWS writes:

    python3 store.py publish --campaign-id ID --mirror MIRROR --profile veox-jepson-prod
    python3 store.py restore --campaign-id ID --mirror FRESH_RESTORE --profile veox-jepson-prod --expect-commit-sha256 EXPECTED_SHA

Select credentials by private required .env at the controller layer. store.py takes the selected profile explicitly, uses boto3's provider chain, verifies account 714044927654, never embeds or prints keys, and rejects publication outside the learning bucket. The function IAM role need not receive write authority: the coordinator can batch results and publish them with a prefix-restricted writer.

## Durability and validation

Local staging writes a same-directory temporary file, fsyncs it, links it atomically without replacement, then fsyncs the directory. Existing different content is a collision. Paths reject traversal and symlink escape.

Gzip uses a fixed timestamp, canonical sorted JSON per line and deterministic boundaries. Each shard records compressed and decompressed hashes and byte/row counts. S3 writes set If-None-Match: *, SHA256 checksum, sha256 metadata, AES256 encryption and explicit content type. Every created or existing object is read back in full and compared. A 409/412 response is never accepted on its own. Dependencies publish before the manifest; COMMIT is last. A crash before COMMIT leaves recoverable uncommitted objects and can resume without overwrites. A crash after COMMIT has a complete verified manifest closure.

Restore starts at a separately retained COMMIT hash, verifies the manifest, every compressed object, decompressed hash and row count, aggregate object count and bytes, then materializes a fresh local mirror. A self-consistent replaced whole campaign cannot be detected without that external anchor; retain the hash in the user-facing report and independent delivery bundle.

Validation:

    cd /home/ubuntu/JopeDime/.agent/experiments/lambda-csv/gp
    python3 -m unittest -v test_store

Thirteen tests cover deterministic shards, resume, exact restore, duplicate content under different kinds, immutable collisions, interrupted publication, missing/corrupt dependencies, post-write readback corruption, remote tampering, external anchor mismatch, aggregate mismatch, path escape and wrong-bucket refusal. No AWS mutation is made by the tests.

## Cost and query policy

Use 100 complete event records per shard initially; bound by payload bytes as well when models grow. One PUT per event would cost $5 per million events at the measured Ohio Standard request rate; one per 100 records costs $0.05 per million per shard family, before manifest/index/source/verification/storage costs. This is storage request amortization, independent of Lambda batch size.

The dataset index supports later per-dataset review without listing thousands of one-event objects. For large history, build a derived Parquet table (task, family, evaluator, fold, seed, event time, quality, duration, peak RSS, charged cost, shard reference) with a separate versioned manifest. Keep raw JSONL archives authoritative. S3 lifecycle changes, Glue/Athena tables and canonical learning admission are not created here.

## Scientific limits and the useful flywheel

Each cheap GP search creates two different assets: a replayable candidate and a measured resource profile. Candidate comparison must match evaluator, task, family, fold, preprocessing and budget; a smaller cost alone does not prove equal search quality. Fold selection is an adaptive objective. Inside-train audit rows support diagnostics; GP workers never receive original outer test rows or labels. The separately isolated BeyondArena curator may decode raw mixed-split Arrow batches transiently, then projects only declared official training rows before dtype conversion, feature encoding or label mapping; no outer values or statistics are persisted.

Current CSV diagnostics lack a complete frozen family registry and many original CSVs contain preprocessing fitted before the stored split. Every result therefore remains canonical_learning_admission=false. Protected meta-test matches are excluded. Unknown family protection is explicit, not silently promoted to safe. Historical meta-validation does not become teacher training. Legacy multi_label becomes multiclass only when metadata proves exactly one classification target; timeseries stays identified and uses an ordered inner split whose causal/group adequacy must be qualified separately.

The productive repeat cycle is: inventory and hash-pin inputs -> freeze comparable search budgets/seeds -> safe concurrent GP evaluation -> independently replay programs/predictions -> retain successes, failures and exact costs -> derive family-balanced candidate comparisons -> admit only qualified evidence through jope-learning -> export corpus with intact ledgers -> train proposals -> compare fresh versus replay under a preregistered protocol. These archives feed that process only after admission; they do not mutate HWM, bank, accepted policies, serving pointers or teacher corpora.

Train a routing policy on byte count, rows, columns, selected feature count, task, candidate grammar, number of fits, and observed memory/time. Split validation by dataset family and time. Use conservative upper quantiles for memory and timeout and optimize cost per successful admitted event, including retries, transfers and logging. Historical cold/warm/cache status and AWS Report memory are essential because Rust process RSS is not billed memory. Route only among measured tiers; use a documented fallback and keep cost reservation before concurrent invocation.


## Timestamps and the separate BeyondArena screening wave

Keep microsecond UTC ISO8601 and integer epoch_us pairs for dispatch, handler start/end, individual jobs and candidates, plus monotonic elapsed_ns for durations. The dataset index preserves available timestamps; the complete event record remains authoritative for all timing stages. Plot exact points and quantiles against memory, workload, cold/warm state and completion outcome. Never derive start times by subtracting rounded billed duration from a client timestamp.

The bounded BeyondArena v2 curator freezes one official repeat0/fold0 per source, at most4096 training rows and512 original features. Source-family metadata seeds target-independent row selection; feature-name hashes select the bounded column set. Official outer rows are excluded from GP input. Numeric features use fixed missing-zero plus indicators; text/categorical features use a stateless48-bit token hash; dates use epoch seconds. This is an explicit screening representation, not an official full-feature benchmark or a text-understanding claim. Regression targets retain exact float64 values. Classification labels use independent per-label SHA256 first52-bit integers with collision detection and a retained typed-label inverse mapping; no fit label code depends on the presence or ordering of audit labels.

The curator never writes cloud objects. The root coordinator conditionally uploads training CSVs, plans and preprocessing recipes under its experiment prefix and pins returned S3 VersionIds in a separate runtime manifest. It preserves source Parquet hashes/versions, original and selected training-index hashes, source family, declared task, split regime and precise timestamps. Later GP replay needs the recipe plus the chosen original columns and the native model. Thirty-nine grouped/temporal sources remain diagnostically labelled; the generic inner GP split is not a qualified group-disjoint or causal evaluation. Every output retains canonical_learning_admission=false.
