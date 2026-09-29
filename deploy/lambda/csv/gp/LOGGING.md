# Precise Lambda GP telemetry

Run after each completed experiment wave:

    python3 /home/ubuntu/JopeDime/.agent/experiments/lambda-csv/gp/analyze.py --root /home/ubuntu/scratch/jopedime-lambda-gp-20260928 --out /home/ubuntu/scratch/jopedime-lambda-gp-20260928/analysis/final --phase corpus

The analyzer performs no AWS calls. It reads immutable attempt intent/result/receipt files plus a read-only SQLite admission snapshot, tolerates missing receipts as unknown admissions, writes JSONL and Zstandard Parquet tables, then generates a self-contained offline Plotly dashboard and publication-sized PNG/SVG figures. Rerunning rebuilds derived files atomically; authoritative attempt receipts are untouched. Do not publish an analysis directory while it is being rebuilt. Its final analysis-manifest.json binds every exported file and analyzer source hash.

For calibration, choose --phase canary. Raw tables always retain all phases; summaries, throughput and plots use only the selected phase. Never merge calibration invocations into corpus throughput without labelling that choice.

## Three measured units

- Invocation: one Lambda API attempt, including an unknown or failed response. Cost and AWS billed duration occur once on this row.
- Dataset GP event: one requested manifest-bound dataset entry inside an invocation. A batch contains up to five events. An event qualifies as a GP success only when status is trained and at least one mutation/crossover candidate was measured. A bootstrap-only model is not counted as a qualified GP search.
- Candidate: one observed actual fit attempt within a GP search. Measured candidates and evolved measured candidates are separate counts. Fit failures are counted when their trace survives. A job that fails before returning its trace can contain unobserved work; plots therefore say observed candidate fits.

Repeated attempts never create new dataset identities. Jobs join by manifest_sha256 + integer job_id, then attempt_id. A wave can reuse integer job IDs under a different manifest. The source census is retained separately: 4,001 source entries, 3,987 required diagnostics and 14 explicit canonical exclusions at this inventory snapshot. Source entries are not proven independent families.

## Files

- invocations.jsonl/.parquet: client request, handler span, billing, memory tier, cold state, source/code/manifest identity, observed or unknown outcome.
- jobs.jsonl/.parquet: task, shape, input bytes, fold identity, stage durations, search stopping/counts, champion identities, audit metric JSON, errors and an invocation cost join key. No duplicated invocation dollars.
- candidates.jsonl/.parquet: generation, origin, parents, unsigned seed as decimal text, draw, program/model identities, selection metric JSON, fit/end timestamps, duration and RSS.
- spans.jsonl/.parquet: queue/tier wait, client request, handler, download, prepare, train, total candidate, fit and selection-prediction intervals.
- events.jsonl/.parquet: exact observed event timestamps and clock domains for starts and completions.
- throughput.jsonl/.parquet: one-second starts/completions for calls, dataset events and observed candidate fits.
- integrity_issues.jsonl/.parquet: missing or mismatched intent/result identities.
- summary.json: counts, unknowns, p50/p95/p99 durations, cold/warm, task and memory breakdowns, costs and census coverage.
- burst-series.json: exact UTC sweep-line concurrency and one-second count series.
- dashboard.html: offline interactive figures, exact UTC hovers and audit notes.
- burst-overview.png/.svg: three aligned throughput/concurrency panels.
- efficiency.png/.svg: latency CDF and memory-tier compute-cost scatter.

## Clock contract

Coordinator stamps have UTC microseconds, readable UTC ISO8601, nanosecond epoch text and local monotonic nanosecond text. Lambda handler stamps have UTC microseconds and monotonic duration nanoseconds. Jobs and candidates retain UTC boundary stamps plus monotonic relative offsets. The analyzer derives durations from monotonic differences whenever available; otherwise it explicitly labels a same-clock UTC difference.

Epoch microseconds are signed int64 in Parquet and integers in JSON. ISO8601 has six fractional digits. Unsigned seeds and nanosecond coordinates are decimal strings to avoid JavaScript integer rounding. Plot positions use elapsed seconds; hover text and exported tables preserve the original UTC microseconds.

Precision is not accuracy. Different Lambda environments and the coordinator do not have a measured microsecond synchronization bound. Do not subtract coordinator start time from a Lambda wall timestamp to claim a precise network or scheduler delay. Same-process monotonic queue and request durations are valid; cross-environment concurrency is an observed wall-clock estimate.

Active concurrency uses half-open intervals [start,end), merges simultaneous changes, and excludes missing or reversed spans. Zero-length spans do not create concurrency. Binned counts are per one-second interval from the first observed Lambda timestamp; they are counts, not a smoothed instantaneous rate. Missing spans remain in invocation/job denominators and are reported as unknown.

AWS max-memory reports can reflect the reusable environment lifetime. Rust process high-water memory also persists through warm reuse. Neither field is a clean per-candidate allocation measurement. Configured memory, process RSS, and billable memory are different quantities and remain separately named.

## Cost and success rules

The invocation is the accounting unit. Total observed compute cost sums lambda_usd once per receipt; actual S3 GET estimates join once per invocation. Job rows carry only invocation_cost_join_key. If a downstream report allocates cost to jobs, its allocation rule must be named and must sum back to the original invocation total.

Unknown bills are not zero-cost successes. They are counted separately and retain controller reservations. Observed totals include input GETs plus raw-result archive PUT and coordinator readback GET costs, each once per invocation. Other campaign publication requests, storage, CloudWatch and host curation remain outside this total; the final campaign report must add or state those separately. A per-million equivalent-call scatter is not a per-million dataset-event price when batch sizes differ.

Do not pool task quality metrics. Regression, binary and multiclass objectives differ; adaptive selection scores and inner audit scores also differ. Retain exact metric JSON and comparison identities for later matched analyses.

## Validation

    python3 /home/ubuntu/JopeDime/.agent/experiments/lambda-csv/gp/analyze.py --self-test

Synthetic cases test tied interval boundaries, one-second buckets, manifest-scoped IDs, batched costs counted once and unknown attempts retained in denominators. A live five-call canary validated real timestamps, 60 observed fits, 40 measured evolved candidates, five qualified GP events and the joined observed cost. Figures were rendered and visually inspected. The root coordinator must rebuild after both corpus waves finish, inspect summary and integrity_issues, verify Parquet/JSONL row counts, then package the final manifest and charts.


Admissions and dispatch are separate: a reservation without an intent file is recovered from admission.sqlite3 and stays in liabilities and planned-job denominators. No request-start record means it is not counted as an observed Lambda call. The analyzer never refunds or edits the ledger. Full source-completion coverage joins successful events across all phases because corpus resume intentionally skips successful canaries; corpus-phase throughput and cost still exclude those canaries.


## Durable-result timing and cost extension

New handlers archive the raw result before API return. `handler_end_utc_us` with `handler_end_scope=before_archive_put` and its monotonic duration describe work before the archive. `response_ready_utc_us` is the observed post-archive boundary; `handler_including_archive_s` is the full monotonic duration. The analyzer retains both spans and uses the latter boundary for Lambda starts/completions/concurrency whenever available. It never invents a post-archive wall timestamp from AWS billed milliseconds. Recovered raw S3 bodies can lack this boundary; their shorter observed span stays explicitly labelled before_archive_put.

`archive_receipt.put_start_utc_us`, `put_end_utc_us` and `put_s` preserve Lambda-side archive latency. The coordinator archive-readback span is separate and uses its own monotonic clock. Receipt `archive` binds the verified bucket/key/VersionId/SHA256/bytes. AWS `report_duration_ms` and `init_duration_ms` remain separate billing/runtime observations.

Invocation rows contain `archive_put_count`, `archive_get_count`, `archive_put_usd`, `archive_get_usd`, and combined `observed_compute_and_s3_usd`; dataset rows retain only the invocation join key. Summary `observed_lambda_plus_s3_usd` adds Lambda, input GET, archive PUT and archive readback GET once. The old `observed_lambda_plus_get_usd` remains the Lambda+input-GET subtotal for compatibility. Legacy attempts without result_object/archive fields have zero archive costs. Expected archive operations without a receipt remain unknown and are counted in archive_cost_unknown_invocations. Other campaign publication operations are outside this subtotal.

Confirmed `not_dispatched` reconciliation remains in admission counts and the conservative non-refunded budget reservation total. Its planned jobs carry status not_dispatched; it contributes neither an observed Lambda call nor an unknown-work liability. Pending/unknown attempts with no bill remain separate. `reserved_usd` is an intentionally conservative admission-budget total, not measured spending.

## Final immutable campaign archive

After dispatch stops and receipts are reconciled, rebuild `analysis/final`, then run:

    python3 /home/ubuntu/JopeDime/.agent/experiments/lambda-csv/gp/final_archive.py stage
    python3 /home/ubuntu/JopeDime/.agent/experiments/lambda-csv/gp/final_archive.py publish

Only publish performs cloud writes. It uses the existing scoped writer role and `.env` configuration through `load_env` and `writer_session`. All object writes are conditional and read back; the manifest follows its payloads and COMMIT is last. Preserve the printed `commit_sha256` outside S3 as the restore trust anchor. After publication, perform a fresh download and content verification:

    python3 /home/ubuntu/JopeDime/.agent/experiments/lambda-csv/gp/final_archive.py restore --mirror /home/ubuntu/scratch/jopedime-lambda-gp-20260928/fresh-restore-mirror --restore-dir /home/ubuntu/scratch/jopedime-lambda-gp-20260928/restored --expect-commit-sha256 REPLACE_WITH_PRINTED_SHA256

The final manifest includes all 426 existing curated input/preprocessing/receipt objects. Large raw source Parquet and extra uncompressed training CSV copies are excluded. Exact attempt files are compressed into separate success, failure and confirmed non-dispatch bundles. Source, evidence, final analysis and a read-only admission-ledger export have deterministic tar/gzip bundles. The file catalog binds every member path, SHA256 and byte size; the dataset index joins manifest-scoped job IDs to attempts and champion hashes. Restore verifies the COMMIT anchor, manifest, objects, bundle hashes, member catalog, path containment and every reconstructed file. Canonical training S3 inputs remain pinned external dependencies; the derived BeyondArena training projections are included in this closure.

The builder refuses credential files, detected keys and signed capability URLs. One fabricated Rust URL-validator unit-test fixture is narrowly excepted by both exact full-source SHA256 and exact matched-query SHA256, recorded in manifest metadata; the runtime owner confirmed its impossible TEST credential identifier. This preserves exact deployed source without treating a fabricated test string as a live capability.

`stage` makes no AWS calls. It requires a current analyzer source hash, verifies all final analysis hashes, refuses pending admissions, and checks the ledger stayed unchanged while staging. Unknown reconciled invocations may remain archived as unknown. `--allow-incomplete` is only for a clearly labelled local diagnostic snapshot. Staged manifest references are immutable: finalize sources and analysis first, then stage once in the publication mirror. Self-tests use fake S3 only:

    python3 /home/ubuntu/JopeDime/.agent/experiments/lambda-csv/gp/final_archive.py self-test

The final corpus and synchronized-probe views share all-phase master raw tables. Their analysis manifests and summaries explicitly set raw_table_scope=all_phases; only throughput is a phase-filtered table. Use the phase column to query canaries, corpus, cost trials and burst trials. The separate CloudWatch ingest snapshot counts unique exported events across all phases, UTF-8 message bytes plus 26 bytes/event, with an explicit assumed rate of $0.50 per 1,073,741,824 bytes. This estimate is not added to individual phase invocation costs.

When including the separate synchronized-probe figures in the final archive, pass `--analysis-extra /home/ubuntu/scratch/jopedime-lambda-gp-20260928/analysis/burst-probe` to stage. Cost-study charts under the evidence root cost-study directory are included automatically.
