# Lake views v1 contract

Lake views are disposable analytical projections over immutable, versioned evidence. The source
manifest remains authoritative. A view carries its source-manifest digest, evaluator/profile,
exact source key and version IDs, content hashes, exposure ancestry hashes, and requested/observed/
excluded counts. Outcome bytes are fetched only after role authorization, then hash checked and
parsed. Missing, corrupt, or wrong-version objects are excluded with a reason and make
the view incomplete; consumers must not silently treat the remaining rows as complete evidence.

## Disclosure

The I04 principal verifier owns authentication and purpose capabilities. The lake accepts only an
adapter implementing `RoleAuthorizer`; it does not interpret a caller's role string as an
identity or capability. The role filter enforces the purpose/row-role match before source bytes
are fetched. Denial errors are generic and contain no row identifiers, labels, or feedback.
Production wiring must use the I04 verified-principal adapter; a permissive test adapter is only
for synthetic fixtures.

## Publication and recovery

Projection rows and commit manifests are content-addressed, create-only objects. Existing objects
must be read back and byte-verified. The head pointer is the only publication commit point and
uses the observed version from the existing store's conditional update. The writer re-observes the
head before that conditional update; a changed head aborts. Listings are discovery-only and never
choose a transaction parent. Rebuild starts from the committed head and follows its parent chain,
not the newest-looking object name. A crash before head publication leaves unreachable immutable
objects that a later rebuild may safely ignore.

This lane defines a draft `RoleAuthorizer`, `VersionedReader`, and `ProjectionStore` boundary.
I04 must provide the authenticated capability adapter; I26/I27 must bind the accepted delivery,
fence, observation, and failure schemas before integration. The worker does not allocate a live
namespace, credentials, IAM principal, canonical schema, or CLI surface. No fixture establishes
live durability, scientific benefit, or serving eligibility.
