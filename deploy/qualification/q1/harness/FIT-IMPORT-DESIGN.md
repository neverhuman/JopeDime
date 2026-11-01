# Exact native FIT import for Q1

Q1's engine previously sent every bound CSV to the header-requiring parser.
All153 verified native roster manifests specify headerless TRAIN, so the first
data row became a header and admission failed. This change supports that exact
format inside the pinned evaluator, after existing request/Plan/binding and blob
verification. Q1 cut, request, protocol-manifest and worker schemas stay unchanged.

The existing headered path returns the same table without native metadata.
Otherwise import requires the exact full `catalog_record_sha256` bytes at
`$JOPEDIME_HOME/q1/catalog/<catalog_record_sha256>.json`. This is the native dataset
manifest or raw v246 catalog record, not the normalized development-summary row.
That record must bind the same dataset identity, real regression task, raw TRAIN
hash/size/rows/width, `has_header:false`, and final target name/index. Its pinned
`files.meta.sha256` and bytes resolve another bounded regular JSON file in the same
directory. Meta must bind the same dataset, TRAIN rows, ordered current feature
columns/count, final target and removed-header flags. SHA filenames are fixed
lookup keys; traversal, links, oversized data, changed content and stale fields
refuse. Producer metadata is cached as exact bytes; never rewrite it to fake a pin.

Only then prepend one correctly escaped CSV header, leaving every raw byte and
label untouched. Parser limits include the actual header bytes. Check the same
bound row/column/name invariants again; existing JMAT and numerical admission
continue unchanged. Current `feature_columns` plus the explicitly final target
control order: two native meta files retain stale pre-ingest `output_headers`
and nested counts, so those older summaries cannot supply the header.

The metadata helper/source tests reproduce the old direct-parser refusal, retain
all rows/cells with quoted names, preserve headered imports without metadata,
and refuse wrong roles/digests/targets/order/rows and linked metadata. The external
t0545 packet separately exercises154 data-free real-parser/JMAT fixtures.
No protected table is read by authoring/tests; real import occurs only inside the
admitted pinned evaluator. Missing native metadata remains a refusal, never a
caller-selected flag or guessed target/header fallback.

Rejected alternatives: changing raw blob hashes to invented prepared hashes,
reading protected rows in the authoring harness, using stale output_headers,
selecting another dataset to avoid import, or changing frozen worker/request
formats. This closes one concrete import blocker; it does not admit a fifth wide
family, adopt newly proposed FIT folds, sign a plan, change h40b/r782/DOPE, or
establish Q1/quality/release PASS. API01 retains final manifest/plan/signing/run
ownership; ENG/RES03 retain authoritative population/family disposition.
