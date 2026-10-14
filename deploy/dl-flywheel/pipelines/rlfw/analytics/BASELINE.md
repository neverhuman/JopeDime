# GB-08 baseline export contract

This reader computes observations from immutable, owner-exported receipts. It
does not run training, evaluation, SQL against live databases, S3 requests or
qualification. All generated reports remain **NON-QUALIFYING**, including
synthetic fixtures. Missing values are NULL. L6 is a loss; lower is better.
Existing historical D9 reports and their definitions remain separate.

Run from the pipelines directory with its Python environment, inside a capped
scope with at most 200% CPU and 4 GiB memory:

```sh
python -m rlfw.analytics.build_baseline --manifest /absolute/manifest.json --output /absolute/new-run
```

The manifest schema is `jope.rlfw-baseline-manifest/v1`, with `date`, `window`
(two UTC epoch seconds), `window_kind` (`utc_day` or `whole_run`), and `exports`.
UTC days must be midnight to midnight; whole-run counters are never prorated
into days. Each export is an absolute local file reference with `path`, `sha256`
and `bytes`. Export domains are `throughput`, `arrivals`, `feedback`, `phases`,
`cache` and `cost`. Omitted domains remain unmeasured. The small example below
is deliberately an availability report, with no measured production inputs:

```json
{"schema":"jope.rlfw-baseline-manifest/v1","date":"2026-10-13","window":[1791849600,1791936000],"window_kind":"utc_day","exports":{}}
```

Each copied JSON export has schema `jope.rlfw-baseline-export/v1`, its `domain`,
the exact report `window`, `scope`, nullable `producer_commit`,
`evidence_status`, `cohort_sha256`, boolean `complete`, and `data`. Completeness
is a producer assertion backed by the pinned export, not inferred from rows.
The reader retains producer/consumer commits and the export's file:line and
SHA256. Arbitrary unexported live sources are not a supported input contract.

`baseline.py` validates the following data contracts. Synthetic accepted inputs
and refusals are executable in `tests/test_baseline.py`.

| Domain | Required evidence and behavior |
| --- | --- |
| throughput | `allocations`: span_id, allocation_id, interval, positive cores, source_kind and owner_receipt_sha256. Reservation spans integrate cores times clipped seconds, including reserved idle time. Overlap within one reservation refuses; distinct reservations may overlap. Quota capacity is a separate denominator. `allocation_complete` is required for allocated hours. |
| active CPU | `cpu_counters`: counter_id, scope_id, explicit ancestors, invocation_id, receipt_sha256, exact interval, start_seconds, end_seconds, counter_kind (`evaluator_cgroup` or `process_rusage`). Aliases deduplicate, resets refuse, and parent/child or mixed counter sums refuse. `cpu_complete` is required. Process counters never fill north-star G's evaluator-cgroup definition. |
| throughput counters | `counts_complete` plus `counts` with exact interval and separate nullable nonnegative integer programs, folds, predictions, compiles, cached_disclosures, physical_attempts, missing_outcomes. Whole-run counts crossing the window remain NULL. `report_window_seconds` is the stated interval, not an inferred runtime. Optional thread_configuration has workers, inner_threads and declared_cores; its product is configured demand, never measured concurrency. |
| arrivals | Complete ledger plus lifetime_novelty_complete. Every record binds outcome_id, context_id, parent_id, role, canonical_program_sha256, comparison_identity (task, dataset_sha256, cut_id, evaluator_sha256, vocabulary_sha256, canonicalizer_sha256, protocol, role_map_sha256), qualified, qualification_receipt_sha256, remote_version_readback_sha256, first_qualified_proof_sha256 and first_qualified_at. Count first-qualified context/program/comparison tuples in the half-open window. These are readbacks of owner qualification, not fresh qualification by this reader. |
| noncatalog novelty | Additionally require frozen_catalog_sha256, catalog_membership_proof_sha256, boolean in_frozen_catalog, first_program_seen_at and first_program_seen_proof_sha256. This is the first global qualified arrival, not a proposal/draw/export clock. Deduplicate canonical programs globally. An old program arriving on another context is a new context/program outcome but not a new program. Missing novelty proof keeps the full novelty count NULL. |
| feedback | episodes bind episode_id, context_id, parent_id, outcome_id, target_packet_sha256 and support_sha256. `clocks` preserves proposed, compute_started, locally_verified, remotely_qualified, adopted, first_trained and first_evaluated; first_exposure is separate. A clock has at, domain and nullable uncertainty_seconds. First-trained additionally requires optimizer_update > 0, consumption_receipt_sha256, parameter_change_receipt_sha256, input_snapshot_sha256, learner_invocation and the full comparison_identity. Exposure needs exposure_receipt_sha256. observed_until is fixed at the report end. |
| latency | Same-domain clocks can subtract directly; cross-domain clocks require uncertainty bounds. Missing intermediate endpoints stay NULL. Proposal-to-update, remote-qualification-to-update and proposal-to-exposure remain separate. Completed-only mean/p95 and their sample count are separate from censored ages/counts; the completed p95 is not a full-population p95. |
| phases | records bind phase_id, name, start/end clock records; queue requires enqueued/dequeued. Custody completion additionally requires remote_version_id and readback_sha256; PUT return is not completion. Pending phases are censored at observed_until. Nested compile/execution/custody durations are never summed. |
| queue | queue_events are ordered right-continuous at/depth changes, with an initial value at/before the report start and queue_event_coverage_complete. An incomplete trace or a late first sample cannot establish time-weighted depth. Missing samples are not interpolated into completeness. |
| cache | cache_kind and identity_sha256 bind source/compiler/target/flags/dependencies/ABI/embedded data and comparison/fit identity at the producer. lookups have lookup_id, same kind/identity and boolean hit. Complete hit/miss coverage is required; zero lookups yields NULL ratio. Aliases do not add hits or new qualified arrivals. |
| cost | Owner ledger records bind ledger_event_id, bucket, prefix, interval, window_kind, bytes_measure_kind, cost_category, currency, cost_kind, pricing_identity, nullable usd/bytes. A cost JSON export may instead pin `ledger` Parquet with those columns; the reader copies it before opening. No S3 lister or pricing fallback exists. |
| cost coverage | `expected_categories` is the producer's exhaustive category set. Only exact report_window rows contribute to total USD; missing categories or NULL components keep the total NULL, with a separately named known subtotal. Overlapping bucket/prefix rollups, conflicting aliases, mixed currencies/pricing or billed/estimated totals refuse. Stock bytes and monthly estimates do not fill daily additions or bills. Cost per qualified outcome requires positive measured outcomes and matching scope/cohort/window. |

The output preserves the existing daily Parquet columns and records the strict
contract in versioned `details`. HTML and METRICS.md are generated exclusively
from Parquet readback. `COMPLETED.json` is written last and pins the produced
files, copied inputs, source identity, resource scope and fixture status. A
partial failed run has no completion marker; prior runs are never overwritten.

Read-only real baselines require retained owner exports. Historical G-S3GP's
952.9329 programs/active CPU-hour and 52.1980 programs/allocated core-hour are
whole-run, parity-unqualified observations, not this day's qualified rates.
GB-01's 5,461-run backlog remains G34-blocked and unexecuted. Source review,
local fixtures, hosted CI, live qualification and promotion remain separate.
