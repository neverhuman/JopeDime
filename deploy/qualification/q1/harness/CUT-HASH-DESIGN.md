# Q1 canonical cut hash design

## Decision

Use `jope.q1-cut/v1`: SHA-256 over UTF-8 canonical JSON containing the full
dataset digest, train-object digest, FIT row count, target, ordered feature
names, and ordered repeat/train/validation fold membership. Canonical JSON
sorts object keys recursively, uses compact separators and literal UTF-8.
Arrays retain their supplied order. The hash excludes its own digest and the
outer manifest-entry digest.

## Control and alternatives

The shared golden control is
`0c87e715ccb1a01fe05d4bc4e8b81f62f978d8aed0126248d97b866686acf249`.
Tests show object insertion order does not affect the hash, while changing
target, feature order, fold order, or row order does. We reject hashing only a
row/column count or a caller-provided opaque cut label: those alternatives do
not bind the exact training bytes and fold recipe. The production evaluator
recomputes the digest from the pinned manifest and loaded FIT object before
charging work.

## Risks and boundary

Python and Rust must preserve the stated integer, string, and array semantics;
the golden fixture is the cross-language control. Any schema or canonical
encoding change requires a new schema version. This protocol addition does
not establish an authoritative cohort, evaluator qualification, authorization,
or permission to launch.
