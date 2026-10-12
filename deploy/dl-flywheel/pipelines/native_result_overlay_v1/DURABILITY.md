# W03 caller completion

This is a BUILD ONLY repair for CL to adopt from a stacked branch. It does not authorize live qualification,
learning, retries or merges. Existing acceptance, failed-root and original cost receipt bytes remain custody.

Both `completed_packet` and `validate_chain` use `accepted_publication`. The gate requires the accepted envelope,
the publication candidate, and an operator return witness at `<root>.acceptance-qualified/completion.json`.
Legacy positive envelopes and publication candidates alone are never admissible and are never upgraded by readers.

The bounded publisher writes and file-fsyncs the envelope and fsyncs its parent. It then writes/file-fsyncs a private
publication candidate, fsyncs its parent, atomically links the candidate to `<root>.acceptance-complete.json`,
fsyncs that parent again, checks the UNCHANGED original wall/monotonic/boot bound AFTER these final operations,
and returns the exact acceptance/publication references. Thus a link or final directory fsync crossing either
original deadline cannot return a qualified success. A reader inside that operation still has no return witness.

The operator samples FIRST after the bounded publisher actually returns and checks the same original paired
bound. Only then does it record the return witness, binding the exact publication, acceptance, root and original
bound. This is terminal evidence of a measured completed operation; evidence materialization/delivery is separate
from the bounded operation it witnesses, as with the builder's terminal acceptance. It does not extend the bound,
replace the post-return sample with a pre-I/O sample, or certify the publication from its own readable bytes.

The witness is written/file-fsynced in a private directory; that directory and its parent are fsynced before
create-only atomic directory delivery (`renameat2(RENAME_NOREPLACE)` on Linux). Both consumers require its exact
canonical bytes. Any error/death before delivery leaves only custody and refuses. A concurrent reader cannot use
a witness whose file fsync is unfinished. The final delivery has no subsequent fallible I/O; losing its directory
entry after power loss fails closed, while surviving entries name already-fsynced evidence for an already-durable,
measured in-time publication. An existing file or directory, including an empty directory, is never overwritten.

All partial envelope, candidate, witness and failed-root bytes remain in place. The create-only envelope claim
prevents silent republication/recovery of the same root, including a bounded publication whose evidence delivery
failed. A negative record is not needed. The original builder receipt/cost remains custody even without admission.
Death after successful evidence delivery may leave a committed packet without a returned acknowledgement;
duplicate delivery refuses without changing bytes. The protocol does not silently reuse an uncertain optimizer
batch or mint another evaluation identity.

`tests/test_durability.py` adds actual private process-death and I/O faults: the retained CX-REV parent-fsync case,
concurrent reads during marker fsync, partial packs, final-link wall/monotonic crossings plus in-time control,
final publication parent-fsync, witness failures, claim/checkpoint/manifest boundaries, interrupted SQLite CAS,
and pre/post first exposure with duplicate proof delivery. All original213 fixture IDs/bodies remain. These CPU
worlds include toy optimizer/checkpoint operations and are NON-QUALIFYING, not hardware power-loss qualification,
native evaluation/parity, live feedback closure, promotion or retry permission. Separate F2/F3/CLI findings remain.
