# Historical engine manifests

These versioned files are immutable inputs and provenance records for earlier
engine runs. Their version suffixes identify the pinned run or review set; they
are not alternate current implementations. Keep their bytes stable because
the adjacent SHA-256 lists and downstream receipts bind to them.

| Files | Purpose and proof | Owner and retirement |
| --- | --- | --- |
| `champion-import.sha256` | Binds the v44 champion import executable used by the recorded import run. Verify with `sha256sum -c` against the named artifact. | JopeDime engine maintainers. No expiry while the import receipt is retained; retire only after a consumer audit and archival of the receipt and its artifacts. |
| `hyperion-concurrent-inputs.sha256` | Binds the v30 concurrent executable, specification, and generated job manifest. Verify with `sha256sum -c` against the named artifacts. | JopeDime engine maintainers. No expiry while the Hyperion run record is retained; a future format migration must preserve the old digest mapping. |
| `opportunistic-proposals/qualification-cycle-0000.*`, `opportunistic-proposals/qualification-cycle-0001.*` | Pin proposal inputs, executable, service, and specification for qualification cycles 0000 and 0001. The JSON records retain each cycle identity; adjacent `.sha256` files are their integrity proofs. | JopeDime engine maintainers. No expiry while either cycle is used for replay or comparison; migrate only by adding a new version, never by editing these pinned bytes. |
| `dataset-provenance/origin-approval.json`, `dataset-provenance/preparation-controls.json` | Preserve the origin-approval and preparation-control decisions for the respective dataset review snapshots. Their evidence paths retain the original review IDs; file bytes are unchanged. | JopeDime dataset and engine maintainers. No expiry while a dataset admission or audit cites the snapshot; replace by adding a new review record after a consumer audit. |

These records remain in place for reproducibility. New active runs should use
their current manifests and write a new versioned receipt rather than treating
these historical paths as defaults.
