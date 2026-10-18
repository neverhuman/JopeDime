# Lineage checkpoint closure v1

This document defines the additive `parallel.checkpoint_commit.v3` reader
contract. Legacy checkpoints and `parallel.checkpoint_commit.v2` retain their
existing bytes and version-dispatched readers.

## Bound identities

The v3 `state.json` keeps every v2 complete field and adds:

- `lineage` and `intent`: `{ "schema": "jope.…", "value": <object>, "sha256": <digest> }`.
  The digest is SHA-256 of the compact serialized `value`; its `schema` member
  must match the envelope. I11 and I12 own the payload schemas and acceptance.
- `attempt_id`, `window_id`, and `applied_boundary`: the active admitted writer,
  immutable training window, and common optimizer step this checkpoint closes.
- `journal_prefixes`: exactly one entry per rank, in rank order. Each entry
  carries the attempt/window IDs, applied boundary, root relative journal path,
  and the byte count plus SHA-256 binding of the exact prefix.
- `closure_objects`: nonempty root relative `{path, sha256}` objects that are
  verified and synced before checkpoint rename.

The lineage and intent values are inputs to the checkpoint digest. The
consumption acknowledgement that names the published state hash is appended
after publication and therefore cannot feed back into the state hash. This
keeps the digest graph acyclic.

## Publication and recovery

The existing six phases remain ordered: artifacts, manifest, synced staging,
checkpoint rename, staged latest pointer, published latest pointer. A fresh
lease callback is checked before the checkpoint rename and again before the
latest pointer publication. A refusal after rename leaves a complete but
unpublished directory for recovery to quarantine; it cannot advance `latest`.
The training caller must supply a live attempt/lease check. A captured token is
not sufficient.

Recovery verifies every rank prefix against its journal, validates the common
applied boundary, and keeps journal-applied work beyond the checkpoint visible
as uncommitted. If the checkpoint is published but its journal acknowledgement
is missing, `recover_published_ack` appends one commit event through the
checkpoint boundary when that boundary is at or below journal-applied work.
Later applied updates remain visible as uncommitted work. Repeating recovery
recognizes the existing acknowledgement and does not add another logical
commit. A torn tail or a conflicting acknowledgement is refused.

## Compatibility and limits

The format adds no authorization. It does not establish lineage provenance,
genuine intent admission, a GPU grant, custody, numerical qualification, or
serving eligibility. I11/I12 must supply accepted payloads; the production
training caller must wire the v3 manifest fields and live publication fence.
Until that integration lands, worker tests establish only the local format and
recovery behavior.
