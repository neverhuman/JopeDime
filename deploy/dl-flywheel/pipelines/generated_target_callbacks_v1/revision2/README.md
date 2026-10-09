# Generated-target production callbacks: durable custody correction (revision2)

This is a source-only correction of frozen packet e1ea1963fa88f9f862058d6556ed080add7c8745ee7eade07d1dbff90c34d223. The entire prior packet and its37-fixture qualification stay unchanged. The numerical verifier, registered replay, role/partition, reference annotations, Kumo license/weight, objective and original-normalizer contracts are unchanged. Read provenance/predecessor-README.md for their full description and explicit full-export HOLD.

## Why this revision exists

The first packet retained Window counters only in memory and assigned the callback runtime after setup. A configure failure or OOM during a GET could lose attempt custody. It was therefore held before live use. No operation was retried or performed remotely during this correction.

## New own-output contract

`cli.py` reads the exact bounded config and physical cgroup first, then calls `callbacks.start_evidence` **before configure**. Config has a new explicit `output_root`: a nonexistent `output` directory below a new `/home/ubuntu/JopeDime/.agent/proposals/dl-flywheel-generated-target-*` proposal. Its parent must already exist from staging. Source/config/data paths remain read-only. The directory and append-only events.jsonl are created exclusively, fsynced, and cannot be reused or resumed. No SQLite, S3, model, outbox or parent artifact is written. Receipts now explicitly distinguish `own_output_writes=true` from zero database/S3/model changes.

`durable.EventSink` writes a bounded hash-linked JSONL event and fsyncs it before each GET/native dispatch and each registered-input replay. Events bind configSHA, monotonic+UTC ns, counters and source/reference identities; raw programs, data and credential material are not logged. Known failures retain complete available Window timing/attempt/byte counters in a create-only result.json. Hard process death leaves a durable attempt without a terminal result; this means incomplete/unknown, never completion. Bytes read after the last fsynced event are not invented after a hard death. Every normal GET failure closes its stream and writes timing/known byte counts. Canonicalizer failures retain mode, nativeSHA and stdinSHA only; no evaluation mode exists.

Events are capped at8192/4MiB, each16KiB, with two terminal slots reserved. Final local result≤8MiB. All original GET caps, native stdout/stderr caps, no-retry rules, exact versions, fixed work/global clocks and CPU1/512MiB/swap0/noGPU guard remain. Owner resource admission is still required; a sampled free-memory value is not a reservation. An honest OOM is failure with retained incomplete custody.

## Qualification

Portable `package/run_offline.py` here executes exactly9 affected cases: seven new durable-custody fixtures plus the two existing configuration fixtures changed only to include the new required output_root field. It rejects skips, expected/unexpected failures, wrong roster, altered source hashes or foreign package origins. It does not rerun the35 unaffected previous fixtures or any older project suite. `qualification.json` records9PASS, and `provenance/predecessor-37-qualification.json` retains earlier evidence without describing it as a rerun on these bytes. The same core math/replay vendor files remain byte-identical. Additional integration qualification should wire original and new-version fixture identities honestly, not count the two affected repeated identities as new coverage.

The hard-exit fixture uses an owned fork exiting137 after fsync to test interruption custody. It does not claim an actual kernel OOM experiment. The native fixture tests canonicalizer process I/O boundaries using a temporary Python script; real Linux native serde_json qualification remains pending.

## Actual selected-result instructions

Use revision2/package and revision2/config-template.json, not the held predecessor CLI. Fill the reviewed actual runtime/source input refs, whole package .py hash roster, import-file inventory and physical licenses before freezing the clock. Root has exact local Kumo audit0b5a bytes available for create-only staging. Choose ONE dated successful TRAIN observation initially,16GET/64MiB/180seconds, CPU1/512MiB/swap0/noGPU, plus the new owned output_root. All full128 metadata roles remain checked; only selected contexts are replayed and their exact-version data/result/encoder receipts read.

Command inside the reviewed service: `python -B -m <installed_package>.cli --config /absolute/new/config.json --config-sha <exactsha>`. Capture stdout in a separate new owned receipt if desired; events.jsonl/result.json are now the durable primary operational custody. Do not change/restart an active runtime. No GP, PUT, full live cutoff, target export, numeric target construction or optimizer admission command is provided.

The current actual predecessor's seven-field encoder-publication refs still require a separately reviewed exporter compatibility successor. This packet never strips those prior references in place. Exact full cutoff/event/ACK verification and repeated epoch-boundary adoption remain independent work. A passed selected result proves only the explicitly named artifact/semantic/role checks and zero mutation claims, not model improvement.
