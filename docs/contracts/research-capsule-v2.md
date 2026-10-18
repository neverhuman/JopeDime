# Research capsule v2

Capsule v2 is a deterministic projection of one research round and its immutable evidence. It is not a new experiment authority or a promotion record. The source commit remains unknown when the source record does not identify it.

The records crate compiles a caller supplied `RoundInput` through an injected `EvidenceResolver`. The learning crate can implement that interface from an immutable read view; records has no dependency on learning. Each locator binds URI, content hash and version. Revocation is checked before and after lookup. Missing, revoked, mismatched, undisclosed, or invalid evidence cannot support a numeric assertion and is retained as a limitation.

Requested, attempted, computed, applied, committed, exported, reloaded and next-actor counts remain separate. Failed, aborted and inconclusive rounds are representable. An interval containing zero is explicitly inconclusive; this format does not establish equivalence. A derived inference keeps its source stage, including shadow or adaptive, and cannot confer promotion authority.

Rendering sorts set-like arrays and map keys and emits stable JSON bytes without timestamps, machine paths or credentials. Disclosure must be authorized before compilation. Callers must not render protected source material without that authorization. This interface is a draft pending integration of the versioned experiment family and shared schema by the coordinator.
