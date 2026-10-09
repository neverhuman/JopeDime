# Generated-target callbacks: parent encoder custody correction (revision3)

This separate, source-only revision fixes the actual selected qualification failure `closed local pin`. Frozen original packet e1ea1963 and revision2 packet 3ee87876, their completed fixture receipts, the failed live attempt, its config and its clock remain unchanged. `provenance/preserved-attempt-and-packets.json` binds their exact hashes and verifies every prior packet file unchanged.

## Exact correction

The captured selected parent encoder reference has exactly four fields: `path`, `sha256`, `bytes`, `raw_sha256`. Other 35 references keep the strict two/three-field local pin contract. `callbacks.configure` now retains this full encoder reference in a separate immutable role map derived from the pinned RegisteredInputs parent registry. `_physical_checks` derives the exact replayed lineage set and delegates its checks to `parent_encoder.recheck`.

The helper requires the map to match that set, checks the parent lineage and exact registry encoder role, rejects all unknown fields, verifies compressed bytes/SHA, joins the raw annotation to `parent.raw_payload_sha256`, and checks the bounded decompressed SHA. Compressed and decompressed sizes are each bounded at 16 MiB; the latter matches the unchanged evergreen source limit. Generic `read_bound` accepts no new annotation. No unknown annotation is silently dropped. The original source, native verifier, role authority, Kumo weight/license, objective, normalization, timing and durable event paths remain unchanged.

## Qualification and custody

The portable runner was qualified once for exactly 11 new CPU fixtures. They cover both valid hashes, wrong compressed hash/length, wrong raw hash, parent registry mismatch, corrupt gzip, decompression overflow, unknown fields, wrong/unreplayed role, unchanged generic schemas, and actual `_physical_checks` routing. Tests use synthetic gzip payloads with the actual captured metadata shape; they do not copy or execute the live parent data. No original 37 or revision2 seven new fixtures were rerun. The two prior affected configuration cases also were not rerun. Exact roster/source/import guards and rejection of skips/expected failures remain.

`qualification.json` is the author test result; `provenance/independent-source-review.json` is a separate narrow read-only source review, not a repeated test run. The failed live attempt is retained as failure: five verified GETs, 788,233 bytes, 32 durable events, zero GP, PUT, outbox or optimizer mutation. This source correction is not evidence that a new live attempt passed.

## Worker instructions and limits

Read the retained revision2 README for the unchanged durable output/config and selected-result procedure. Stage this entire new package with `config-template.json` source pins into a new owned immutable runtime. Supply a new reviewed config/output root/window; never overwrite or resume the failed attempt. The CLI stays `python -B -m <installed_package>.cli --config /absolute/new/config.json --config-sha <exactsha>` inside the explicitly admitted service. Selected ONE train-result qualification remains bounded by 16 GETs, 64 MiB, 180 seconds, CPU1/512 MiB/swap0/noGPU, subject to owner resource admission and the unchanged global deadline.

This packet does not execute GP, PUT, full cutoff export, numeric target construction, optimizer admission, boundary adoption or source publication. It does not qualify all 128 parent schemas. The separate full-export publication-reference compatibility and whole-cutoff memory/time limits remain outstanding. Package source pins change because this is a new revision; native/binary/data/license pins and scientific semantics do not.
