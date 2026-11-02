# Reporot 3 narrator renderer version disposition

Status: justified HLT-040 false positive. Owner: `jope-narrator` maintainers.

The HLT-040 `repo-rot.path.fake-versioned-source` matches for `render/v2.rs` and
`render/v3.rs` treat active versioned implementations as parked copies. Both modules are
required runtime code. `crates/jope-narrator/src/render.rs` declares both modules, publishes
v3 as the writer, and dispatches persisted renderer identifiers to v1, v2, or v3 during
re-derivation. `crates/jope-narrator/src/journal.rs` writes the active renderer identifier into
each record. Removing either module would make retained records using that renderer
unreadable or unreplayable.

Source identity at review:

| File | SHA-256 |
| --- | --- |
| `crates/jope-narrator/src/render/v2.rs` | `ff5e946783d3d094a1435244b8c3122bc4f36f99d1dd4e329f03406b962c584d` |
| `crates/jope-narrator/src/render/v3.rs` | `87e44b1f1901d789aa2b2061d1c047f00a66bdf29ca684ec9ab8c1fa5d09308d` |

The hashes pin the audited source versions for this disposition; they do not freeze future
renderer development. If either file changes, re-check the active dispatch and update this
record's hashes in the same change.

Proof lane: `cargo test --locked -p jope-narrator` including the retained v1 golden journal and
renderer replay coverage, plus `cargo clippy --locked -p jope-narrator --all-targets -- -D
warnings`. Review by 2027-02-02, or earlier when a renderer revision is introduced. Retire this
disposition only after a versioned journal migration removes the corresponding replay arm and
proves retained records remain readable under the replacement.
