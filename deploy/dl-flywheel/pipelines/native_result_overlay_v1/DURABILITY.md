# W03 caller completion

This is a BUILD ONLY repair for CL to adopt from a stacked branch. It does not authorize live qualification,
learning, publication, retries or merges. Existing acceptance and failed-root bytes remain unchanged.

Both `completed_packet` and `validate_chain` use `accepted_publication`. The gate now requires the original
accepted envelope **and** `<root>.acceptance-complete.json`, bound to that exact envelope, root and original paired
deadline. Legacy positive envelopes without this completion remain inadmissible; they are never upgraded by a
reader or silently retried.

The caller writes and file-fsyncs the envelope, then fsyncs its parent directory. Only after that succeeds does it
create a private `.acceptance-complete.json.pending` file, file-fsync it and fsync its parent. It rechecks the same
original deadline and atomically hard-links the private file to the public completion path. This last operation
is the visibility/commit boundary; the caller performs no fallible I/O after it. Readers cannot observe the private
file as completion while its write or fsync is unfinished. The public path is create-only and the private file
remains as custody. A power loss that loses the last directory link produces missing completion and refusal;
any surviving link names already-fsynced marker bytes for an already-durable envelope. This deliberately favors
refusal over automatic recovery. Process death after the atomic link may leave an admissible committed packet
without a returned acknowledgement; duplicate delivery still refuses without changing bytes.

Any earlier storage error, deadline crossing or process death preserves the packet, receipt, builder marker,
envelope and any private marker already written. The create-only names prevent a later call from republishing the
same failed root. Admission does not depend on best-effort negative records. No evaluator or optimizer cost is
reset or recast as zero; the original builder receipt remains custody even when admission fails.

`tests/test_durability.py` adds actual private process-death injections and I/O failures, including the retained
CX-REV final-parent-fsync failure, concurrent reads while marker fsync is blocked, partial packet publication,
checkpoint/manifest/claim boundaries, interrupted SQLite CAS adoption, and first-exposure proof deduplication.
All prior 213 registered identities and fixture bodies remain. These CPU worlds include toy optimizer/checkpoint
operations and are NON-QUALIFYING scratch evidence, not a hardware power-loss test, native evaluation, parity
qualification, live feedback closure, promotion or retry permit. Checkpoint/CAS/exposure fixtures exercise their
existing contracts; the repair does not broaden those contracts or resolve separate F2/F3 findings.
