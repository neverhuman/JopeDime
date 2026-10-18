# Pipeline seed set v2

`jope.pipeline-seeds/v1` remains the legacy wire format and retains its four-seed admission
cap and existing origins. V2 is an additive, immutable input format. Its producer freezes and
hashes the archive snapshot, eligibility roster, context, ancestry, and retrieval policy before
selection; the pipeline receives materialized seeds and never reads the archive.

Each draw keeps its draw identity, canonical program digest, source (`own_incumbent`,
`other_lineage_retrieval`, or `neural_proposal`), and available flattened ancestry. Missing
ancestry is unknown and cannot qualify as transfer. Retrieval excludes the consuming incumbent's
parent families, dataset families, and canonical program identities transitively. A renamed bank
key does not establish independence. Roles must be TRAIN and candidates must belong to the frozen
eligible-support roster. Retrieval requests pin the digest of the caller-verified role manifest,
and candidate role proofs must cite that same digest.

V2 requires an explicit positive cap and requested source mixture. Admission records offered and
realized counts by source, rejects, and surplus after canonical deduplication. Deduplication does
not erase draw effort or turn source mixture shares into accepted-program propensities. Seeds are
starting points only; every pipeline remeasures them on the consuming run's own cuts.

Cold retrieval, warm retrieval, and shuffled-context controls use the same pinned eligible
support, incumbent, cap, and resource clock. Receipts report retrieval and cache warmup costs;
warm caches change the measured intervention cost and must remain disclosed. This contract
establishes engineering controls only. It does not establish prospective discovery benefit or
authorize archive loading in the pipeline.
