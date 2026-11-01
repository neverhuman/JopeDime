# Trusted research job resolver design

**Approach.** Add an in-memory, empty-by-default registry keyed by a strict opaque job ID. A trusted service-side loader must verify the accepted qualification receipt before it can construct `AcceptedJobAdmission`; construction is crate-private and rejects synthetic control contexts. Each admission pins the SHA-256 of the complete serialized `Plan` and the accepted receipt digest. Lookup rejects unknown IDs, changed plans, invalid IDs and stale capture contexts before returning the opaque provider capability.

**Alternatives rejected.** Caller-selected paths, S3 URIs, JSON imports and automatic discovery would let untrusted input choose a job or receipt. A default/latest job would make dataset and budget selection implicit. Fabricating a backend or provider before its receipts exist would turn a resolver into false qualification evidence.

**Risks and limits.** This is the resolver boundary, not a production admission source or provider implementation. The registry cannot be populated from outside the service crate, and it remains empty until the accepted job/corpus, conditioning, evaluator, gate and seed receipts are available. No CLI is wired, no run is started, and no Q1 or sealed-data claim follows. Plan serialization and receipt digests bind identity but do not establish trust; only the authorized loader may construct admissions.

**Spec clauses.** Spec §13.2 requires private round records bound to their run inputs; §14 Q1 remains separately gated. The design preserves the existing record/checkpoint/worker contracts and adds no schema or wire fields.
