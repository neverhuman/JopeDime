# Complete observation archive, version 2

The additive `retention::distilled::import_gp_results_index_complete` API retains every
occurrence of a supplied `jopedime.gp-results-index/v1` runs array. It does not survey an estate,
issue replay admission, start work, access a store, or change the v1 importer. Its local archive
version is 2; the integration owner allocates the canonical replay schema separately.

The archive retains the exact original UTF-8 index, its SHA-256, the supplied producer manifest,
all parsed rows (including null/scalar rows and unknown fields), and one primary disposition per
array occurrence. Every observation binds the original input hash, `/runs/N` locator and compact
parsed-row hash. The occurrence identifier uses I02's canonical codec and domain separation with
local domain `jope.observation-occurrence/v2`; raw measurement floats never enter this identity.
The complete input hash retains original whitespace and numeric spelling. Malformed whole JSON,
duplicate object keys at any depth, wrong schema or a missing runs array is an explicit import error, never a complete empty archive.

`produce_manifest` is a deterministic census of supplied bytes. It records source commit, code
version, argv, input URI and optional VersionId, additional immutable input roots, cut, time,
ordering and every row identity. Values come from explicit producer inputs; missing fields are
never filled from the importer's checkout, environment or clock. `verify_producer` checks the
manifest against an **independently authenticated controller pin**, required provenance fields
and a fresh census of these exact input bytes. The pin must come from admitted retained custody,
not the candidate manifest or caller's claim of authenticity. The importer has no issuer or
credential. An absent/invalid pin, missing source/command/roots, tampering or census mismatch
returns HELD while retaining every supplied row. Bound means this particular supplied-input
closure; it does not authenticate an individual measurement, establish event closure, prove the
truth of a role claim or establish completeness of a historical/current remote estate.

## Independent axes and denominator

| Axis | Archive behavior |
| --- | --- |
| Outcome | Explicit ok/completed/success, failed/failure, timeout/timed_out and partial spellings are recognized; missing/unrecognized status remains unknown, including on a scored row. Both original status fields survive; conflicting claims are held. |
| Measurement | Missing/null is missing. Finite binary64 retains exact `f64_bits` and optional score identity. Missing/unknown identity holds validation without erasing the observed value. Malformed/nonfinite strings retain their original evidence and get rejected measurement status. |
| Failure reason | Missing reason remains null; neither a score nor its absence constructs a failure reason. Original failure/retry/event/cost references survive. |
| Runtime | Explicit elapsed_ms on a known completed/failed/partial outcome is an observed duration. Timeout lower_bound_ms is right-censored; absent/invalid telemetry remains unknown. It is never a quality reward or exact completed runtime. |
| Provenance | Missing/invalid program, dataset, context and object references are recorded independently of outcome/measurement. Supplied roles are retained without a train default. |
| Eligibility | Per-label observed flags describe availability only. Quality/runtime/failure/policy masks remain HELD pending I04 per-purpose capability and I05 event closure integration. No replay capability or training export is produced here. |

Primary classification precedence is conflicting delivery, subsequent identical delivery,
rejected measurement, explicit failed/timeout/partial, finite score, explicit ok without score,
then unknown. Each original occurrence has exactly one primary disposition; counts sum to the
original array length. Missing provenance/role reasons are orthogonal and do not increment that
sum again. Even scalar rows and malformed score evidence survive. A finite score on a failed or
timeout row stays observed with its original explicit outcome; its primary count follows that
outcome. A rejected score on a failed row retains both the rejection and explicit failure axis.

Deliveries sharing bucket, key and VersionId are grouped conservatively. Identical compact parsed
rows retain multiplicity, with later occurrences marked duplicate_delivery; distinct content
marks **all** participants conflict. Different physical versions remain separate. Original row
locators and cross references remain intact. These are supplied index occurrence counts, never
physical-attempt counts, measured-success rates or inferred I05 logical-request totals. All
unscored and empty arrays remain archives; zero rows supplies no rate or completeness claim.

## Compatibility and dependency handoff

`distill_gp_results_index_with_stats` and its existing v1 enums, role defaults, grouping, ranking,
score direction, validation, refusals and skip statistics remain unchanged. The explicit
`legacy_scored_projection` calls that exact implementation over the retained original index.
Its caller-supplied role is v1 diagnostic metadata, never a replay grant. Its all-unscored,
duplicate-key and invalid-score refusals also remain. Legacy's inferred `Outcome::Ok` is confined
to that compatibility projection; the complete archive does not infer it from a score.

This lane was admitted under Addendum 2 before I04/I05 acceptance. `DependencyEvidence::from_records`
retains serialized I04 admission and I05 event records without inventing another role registry
or treating JSON/Boolean claims as authority. It accepts their actual serializable draft types
at the caller boundary. The integrator must bind the final `ReplayAdmission`/`LabelEligibility`
capabilities and reconciled draw/request/physical-attempt/cost closures, revalidate per purpose,
and lease the consumption ingress before enabling replay. The lane's always-held masks are an
explicit integration boundary, not implemented authority verification. No missing event is
inferred terminal; imported rows do not replace I05 denominators. Shared replay module exports,
canonical schemas, consumer census, coverage mapping, inventory and README totals remain owned
by the integrator. Local module exports included here are only compilation hooks.

## Columnar companion

`ObservationArchive::columnar` uses the existing Parquet writer's row groups, pages, dictionaries
and codecs with versioned additive statistics. Occurrence ID and observation JSON are required;
score is nullable and exact. The full archive, original input and producer bytes must be retained
alongside the derived table: the table is not a standalone replacement for input closure.
Neither this function nor its statistics are an authorized shared/global export. The caller
must apply I04 disclosure/role rules before any publication, including aggregate statistics.

`parquet::write_with_statistics` adds ColumnMetaData Statistics with exact null count and modern
min_value/max_value, explicit TYPE_ORDER column orders and bound exactness flags. Value count includes null occurrences. Booleans and signed integers compare
by value; double bounds follow Parquet TYPE_ORDER with outward -0/+0 zero bounds and exactness flags (page values retain their bits); UTF-8/binary values use unsigned
lexicographic bytes with no length prefix. Nulls never enter bounds. All-null/empty chunks have
no bounds. Chunks containing any nonfinite double retain those values in pages but omit both
bounds so pruning cannot erase them; they remain nonnull. Missing legacy/external statistics or
null counts remain unknown. Double chunks record their NaN count; no distinct count is invented. `FileReader::statistics` returns
typed per-group readback; many-group/page readback uses the existing streaming FileReader.
Legacy write/write_with bytes and the legacy one-group reader are preserved. No dependency or
Cargo.lock change is needed. Footer fields follow the [Apache Parquet format](https://github.com/apache/parquet-format/blob/master/src/main/thrift/parquet.thrift).

## Historical and verification limits

The 18,380-row number is a dated task-card target. `HistoricalStatus` stays explicitly UNKNOWN/
HELD unless the original dated input hash, authenticated producer hash, expected class counts
and matching 18,380 original occurrences are supplied. The tiny fixture is ten invented public
test observations and **does not** stand in for historical data. No historical input or current
estate census was supplied to this lane. No remote listing, recovery or synthetic 18,380-row
construction is authorized. Even fixture verification never establishes today's estate size.

Run observation_import, retention_pack and retention with default CPU features, listing each
first. The fixtures exercise exact score parity, mixed/missing outcomes, producer refusal,
unknown roles/reasons, all-unscored archives, duplicates/conflicts and multi-group/page Zstd
readback with typed/null statistics. Integration still needs dependency acceptance, shared hook
and inventory changes, the full required gate, independent exact-head review and CI. No local
fixture establishes operational durability, scientific benefit, release or serving promotion.

## Custody Provenance Note (`F1`)

`ProducerVerificationStatus::Bound` attests that the manifest digest matches an externally supplied `authenticated_manifest_sha256` pin and re-derives deterministically from the supplied input bytes; the integrator or caller must source `authenticated_manifest_sha256` from independent custody rather than computing it inline from the untrusted manifest itself.
