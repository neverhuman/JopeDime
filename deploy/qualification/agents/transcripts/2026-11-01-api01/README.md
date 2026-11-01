# MCP-05 named-client transcript evidence

These records combine the actual V6 Claude full flow and the actual Codex full flow plus its separately captured zero-fit export-retry continuation. The client-tool JSONL files retain tool requests/results and omit model thinking/internal analysis. `wire.jsonl` is the exact paired MCP transport transcript. Original captures, command receipts, raw streams, export bundles and predictions remain in the external custody paths recorded in each result; no raw TRAIN rows, prediction CSV, bundle payload or auth material is included here.

## Exact runtime and limits

- Runtime source: `72ed48163b5deb911e3b8e49ba9f860b793cf600`, based on main `17118d28a9a0c12ab36a93be7193a09887566f7c`.
- Binary SHA-256: `ff0aad38bdc953fc905f7826609d5459a0bda9e8937d93a901acaf7ca88184f7`; its version output embeds the same source.
- Both full flows used the approved real TRAIN-only 25-row, 13-column catalog input (`fixture_sha256` is recorded per result), one CPU evaluation per client, isolated stores, actual named clients and a local exact-source engine.
- Each full flow completed same-key run retry (one persisted run), status/events, bundle export to a private `out_dir`, identical export retry, and run-based prediction. The Codex full attempt's strict observer timestamp is honestly false; its separately linked zero-fit continuation took a pre-launch tree snapshot, made two exact export retries on the same completed run/store, and verified unchanged files and inventory 1→1.
- `parity_status=unavailable` and `qualified_set_id=not-qualified` are preserved. Gemini is `WATCHER_PROVISIONAL_WAIVER_NOT_INSTALLED`; no Gemini session occurred. This package does not claim three-client A5 qualification.
- V6 native source controls are summarized in `TEST-INVENTORY-V6.json`; full hygiene/lint/audit disposition belongs to the final PR receipt and is not inferred from transcript success.

## Capture integrity

For each full session, `result.json` binds tool counts, run IDs, IDs/counts, output digests and original raw-stream sizes/hashes. The external capture verifier independently checked all raw stream hashes and JSON-RPC pairing. The supplemental Codex `result.json` binds its four exact replay calls, 13 frames, inventory 1→1, zero additional fits, before/after file identities and clean exits.
