# native_result_overlay_v1 — revision 2 (W3 / X3)

This package turns **verified native results** into **train-only overlay records** that the frozen learner sees at each closed epoch. About 98% of proposals are on native contexts. Before this package, their verified scores had no path back into training, because the X1 exporter accepts only `flywheel_variant` rows.

The loss is unchanged (L6): CE applies only to the frozen native HWM winners, and every overlay record is CE-false.

**Status: SOURCE + PRIVATE FIXTURES ONLY**, for independent review by the live owner (Codex /root).
- Nothing here has run against live data, S3, a GPU, a GP or program executor, a planner or an outbox.
- Root owns all live integration, including the boundary inbox.
- Revision 1 (`w3/candidate/`, DELIVERY `597cfd14…`) is preserved byte-exact.

## What revision 2 changes

| # | Change |
|---|---|
| 1 | **Delta packet chain for unbounded volume.** Packets are now `jope.native-result-overlay-packet/v2` (see the first open question):<br>- each holds at most 2,048 **new** rows (never earlier rows) and touches at most 1,024 contexts;<br>- it declares the anchors only of contexts first seen in it, and records per-context `label_unions`;<br>- the head formula is unchanged;<br>- a chain holds at most 4,096 packets and 500,000 rows;<br>- `validate_chain` returns the cumulative view;<br>- `build_packet(..., predecessor_chain=[refs], predecessor_head=H)`;<br>- `overlay_snapshot(base, chain_view, ...)`. |
| 2 | An `ALREADY_IN_SHARD` row with any per-fold RMSE outside isclose(rel 1e-10, abs 1e-12) of the shard row is refused: `SHARD_RMSE_DISAGREEMENT`. |
| 3 | The emission body is consumed from W4's `emission_body_b64` (base64 of the exact stored bytes) plus `emission_body_sha256`, and the frozen-Outbox emission id is re-derived from it. |
| 4 | The boundary envelope-component proposal is updated for the chain (see below). |

## Interface

```python
build_packet(rows_iter, *, native, store, authority, inputs, canonicalize, native_spec_bytes, preprocessing_bytes,
             preprocessing_sha256, registry_bytes, objective_sha256, source_pins, r6_roster_bytes, r6_roster_sha256,
             campaigns, sources, max_padded_tokens, root, deadline, predecessor_chain=None, predecessor_head=None,
             max_read_bytes=8 GiB) -> receipt
validate_chain(packet_refs_in_order, *, expected_head) -> ChainView      # cumulative view; strict, read-only
validate_packet(packet_ref: {path, sha256, bytes}, *, predecessor_head) -> packet dict   # ONE packet, no chain props
head_sha256(packet_raw: bytes, predecessor_head: str | None) -> str
overlay_snapshot(base_snapshot, chain_view, *, max_padded_tokens) -> snapshot
```

### `build_packet`

- Before any row is read, it re-validates `predecessor_chain` in full against `predecessor_head`; both are `None` for genesis.
- It then builds delta packet k and writes a **new** root:
  - `packet.json`, then `receipt.json`, then `overlay-complete.json` (the success marker, written last; it carries `head_sha256`, `predecessor_overlay_head_sha256` and `seq`);
  - every file is created with O_EXCL|O_NOFOLLOW, mode 0444, and fsynced together with its directory.
- Nothing is written on refusal.

**Unbounded volume.** The build stops pulling rows from the iterator once it holds 2,048 distinct new keys or 1,024 touched contexts (also once 32,768 rows are consumed). It records the reason in `stopped` (`NEW_ROW_CAP` / `CONTEXT_CAP` / `SOURCE_ROW_CAP`, or `None` when the iterator is exhausted). The caller passes the **same iterator** to the next build with the extended chain, so no row is lost or consumed twice:

```python
it = verified_results(source, since_head)
while True:
    r = build_packet(it, ..., predecessor_chain=refs or None, predecessor_head=head, root=new_root(), deadline=...)
    refs.append(r['packet']); head = r['head_sha256']
    if r['stopped'] is None: break
```

### `validate_chain`

It strictly validates every packet in order (exact bytes, canonical JSON, closed schema) and recomputes the head chain to `expected_head`. Across the **whole chain** it refuses:
- a broken predecessor link, sequence or chain position (`chain.packets_before` / `rows_before`), and changed pins;
- a duplicate (context, program), evaluation key or result ref (a conflicting or reused ref);
- an anchor declared twice, or any row whose anchored baseline, shard ref or lineage differs bit-for-bit from the context's **first** anchor;
- a `label_unions` entry that is not `shard_rows + every earlier chain row + own rows` for that context. The counts are checked, as is the digest of the sorted overlay program SHAs used;
- more than 4,096 packets or 500,000 rows.

**Memory:** raw packet bytes and parsed packets are released after each packet is accumulated. The view keeps one slotted `ChainRow` per overlay row, holding:
- the record as a field tuple with tokens packed as uint16;
- the result ref as a 5-tuple;
- shared context, lineage and bucket strings.

It also keeps the key index, holding each row's ref signature.

Measured with `w3/fault-r2/chain_memory_probe.py` (synthetic rows: 48-token records, 8 rows per context; peak RSS 0.25 GB for the whole process):
- **2.14 KB of RSS per accumulated row** at 100,352 rows;
- that extrapolates to **about 1.07 GB at the 500,000-row cap**;
- the expected drain volume (about 16k rows) is about 34 MB.

The build's dry-run snapshot covers only the contexts its packet touched. The learner's `overlay_snapshot` materializes one `NativeRecord` per overlay row; this is root's runner memory, sized by the chain.

### `overlay_snapshot`

It takes a `ChainView` and uses the **cumulative union**. The loader's per-context rotation runs over all overlay rows of the context, whichever packets they came from (k_c = min(|O_c|, 8), digest rotation; unchanged). The returned snapshot carries:
- `overlay_records`, `overlay_head_sha256`, `overlay_packet_seq`;
- `overlay_packets` (`[{path, sha256, bytes, seq}]`);
- `overlay_base_snapshot`.

Identity, `batch_manifest` and membership are unchanged.

## Input: uniform verified-result rows (W4 `verified_results(source, since_head)`)

Each row has exactly these keys:

```
outbox_kind, observation, emission_body_b64, emission_body_sha256, state, record_ref, artifact_refs,
evaluation_key, context_id, phase, campaign, program
```

**Emission body checks:**
- `emission_body_b64` must be strict, canonical base64 of the **exact stored Outbox body bytes**.
- The decoded bytes must hash to `emission_body_sha256`, which must also equal `observation.body_sha256`.
- They must parse strictly and be exactly the frozen Outbox canonical JSON.
- The emission id is then re-derived as `digest({origin, batch_id, ordinal})` and must equal `observation.emission_id`.
- Any mismatch is census `REFUSED/EMISSION_IDENTITY`. A row in the revision-1 shape (`emission_body` as a parsed dict) breaks the contract and stops the build.

**Observation and sources:**
- The required `observation` keys are `emission_id, body_sha256, context_id, phase, campaign_sha256, canonical_program_sha256, evaluation_key`. Extra W4 keys are allowed; when it has `backup_sha256`, that must equal the declared source.
- `sources = [{outbox_kind, backup_sha256, event_head: {events, head_sha256}}]`.
- Rows must arrive in emission order.

## Per-row verification (unchanged from revision 1; the authority proof is at most 60 s old)

1. **Emission identity** (from the exact stored bytes, as above).
2. **Inputs.** `describe(descriptor, 'training')` (frozen `NativeInputs`); outer-test values are never decoded, and the train-prefix-only decoder cross-checks y and folds.
3. **Request identity.** `request_identity == evaluation_key` under `WORKER_SHA e3be9386…`.
4. **Measurement.** Exact-version GETs, then `verified_measurement` (7 bodies) and Rust parity at the fixed 1e-9.
5. **Baseline.** The opposite-fold baselines are rebuilt from the training rows; they must agree with the evaluation's (rel 1e-10, abs 1e-12).
6. **Shard join** (frozen `_shard`, y-target join, frozen `_baseline`). Then anchoring:
   - with a shard pair, the row is anchored to the shard's EXACT pair; otherwise `BASELINE_DISAGREEMENT`;
   - without a shard, the first anchor is the reconstruction, carried in packet k and enforced bit-equal through the chain.
7. **Tokens.** No token refusal, and a detokenize round trip back to the program SHA.

## Census (five classes per packet; counts sum exactly to that build's consumed rows)

| Class | Reasons |
|---|---|
| `TRAIN_VERIFIED` | `NEW_OVERLAY_ROW`, `ALIAS`, `ALREADY_IN_SHARD` (only when the RMSE agrees), `ALREADY_IN_OVERLAY` (`row = [seq, index]` of the earlier chain row) |
| `EVALUATION_ONLY` | `VALIDATION_PHASE`, `VALIDATION_PARTITION`, `R6_ROSTER` |
| `ROUTED_TO_X1` | `VARIANT_SOURCE` |
| `REFUSED` | `PARITY_UNQUALIFIED` (R5), `CAMPAIGN_NOT_ALLOWED`, `UNSUPPORTED_SOURCE_KIND`, `EMISSION_IDENTITY`, `OUTSIDE_SNAPSHOT`, `PROTECTED_CONTEXT`, `OUTSIDE_FIT_ROSTER`, `VERIFICATION_FAILED`, `BASELINE_DISAGREEMENT`, `TOKEN_REFUSAL`, `CONFLICTING_REFS`, `OLD_LABELS_WOULD_CHANGE`, `LABEL_ERROR`, **`SHARD_RMSE_DISAGREEMENT`** |
| `PENDING` | `UNRESOLVED_STATE` (recorded / leased / unknown_attempt / unknown / pending_unsupported / blocked), `NO_INPUT_LOCATOR`; never zero-filled |

## Labels

The frozen `label_context` runs on shard rows + **all earlier chain rows** + this packet's rows of each touched context, and the packet records this in `label_unions[cid] = {shard_rows, earlier_rows, packet_rows, overlay_programs_sha256}`. Every overlay row carries its context's first anchor.

The build also requires:
- shard-row union labels equal to the shard-only labels (else `OLD_LABELS_WOULD_CHANGE`);
- every earlier chain record to recompute byte-identically;
- every new record to equal its direct recomputation (exact ints and grids, floats ≤1e-12 relative).

`NativeRecord` fields are exact, with `source_kind='native_overlay_verified_s1'` and `ce_eligible=False`.

## OverlayLoader (unchanged)

- The base window is kept unchanged.
- Up to min(|O_c|, 8) overlay rows are appended in `digest{native-overlay-rotation/v1, context, program}` order, starting at `(epoch*k_c) % |O_c|`; programs already present are skipped.
- Selection stops before `(len(window)+1)*max_len > max_padded_tokens`.
- The winner merge keeps ONE CE record, so the overlay never creates a CE teacher.
- The receipt is `jope.native-overlay-window/v1`; each selected record carries `result_ref_sha256` and `packet_seq`.

The build's dry run covers every context **touched by the new packet**, at every rotation offset of its cumulative overlay set, through the real `prepare_microbatches` + `collate`. Untouched contexts' overlay sets are unchanged, so their earlier dry runs still apply.

## Packet `jope.native-result-overlay-packet/v2` (canonical JSON, ≤32 MiB)

Fields:
- `seq`, `predecessor_overlay_head_sha256`, `chain{packets_before, rows_before}`;
- `pins` (identical across the chain), `campaigns`, `sources[{…, rows}]`, `authority_proofs{first, final}`;
- `rows` (new rows only; each has `packet_seq == seq`);
- `contexts` (anchors of first-seen contexts, including `shard_rows` and `first_seq == seq`);
- `label_unions`, `census`, `limits`, `stopped`;
- the flags `new_ce_teachers: 0`, `optimizer_updates: 0`, `admission_committed: false`.

```
head_sha256(k) = digest({'schema': 'jope.native-overlay-head/v1', 'predecessor': H(k-1) or None,
                         'packet_sha256': sha256(packet.json)})
```

## Boundary integration (Codex/root) — envelope component proposal

```
components.native_overlay = {packets: [{path, sha256, bytes, seq}, ...],   # >= 1 NEW packets appended by this candidate
                             head_sha256,                                  # head after the last appended packet
                             predecessor_head_sha256,                      # active overlay head before (None at genesis)
                             receipt_sha256}                               # build receipt of the last appended packet
```

- The candidate's chain is the active chain plus `packets`. The boundary re-runs `validate_chain(active refs + packets, expected_head=head_sha256)` at publication, adoption and restart.
- The boundary sets `result_snapshot_sha256 = composite_identity(E, head_sha256)` (C10 r4 `boundary_admission_v2`). It applies `overlay_snapshot(base, view, max_padded_tokens=8192)` and rewrites only `batch_manifest.snapshot_sha256`.
- `first_exposures` joins overlay rows through `proof['selection']['selection']['members'][*]['program_ids']` and `overlay_ref_digest`, which has r4's `_ref_in` digest form.

Other inputs root supplies, as in revision 1:
- the live `N.load_snapshot` base and its exact spec bytes;
- `NativeInputs` over the spec `input_lists` with the real `bound` reader and `native_corpus_authority`;
- an exact-version store with `nativecanonical_json`, plus the pinned `canonical_program`;
- the authority callback;
- the campaign contract's `normalizer_sha256` / `objective_sha256`;
- the real R6 roster (`d228d413…`) and the campaign allow-list;
- `max_padded_tokens` equal to the live runner (8192);
- fixed panels from `snapshot.overlay_base_snapshot`.

## Fixtures and fault injection

30 private CPU fixtures run on xbabe3 with the pinned runner (`run_offline.py`, the same environment as revision 1):
- 20 are carried from revision 1, adapted to the chain API.
- 10 are new targeted regressions (`test_r2_*`):
  - the 3-packet chain with a context in packets 1 and 3 and the first anchor carried;
  - chain-vs-one-pass replay equality;
  - a tampered middle packet, reordered packets, a duplicate across packets, anchor drift / redeclaration, and label-union forgery;
  - `SHARD_RMSE_DISAGREEMENT`;
  - the exact-bytes emission body;
  - row-cap deferral across packets with the chain bounds.

`w3/fault-r2/fault_inject_r2.py` disables each new guard in a scratch copy (G9 chain prefix/link, G10 anchor drift, G11 RMSE refusal, G12 cross-chain duplicates, G13 label union, G14 exact emission bytes, G15 row cap). It also re-runs the carried revision-1 guards.

## Not claimed / limits

- The fixture world, its versioned store, its canonicalizer and the Rust/parity bodies are synthetic; nothing here is live qualification.
- Chain memory at 500k rows is **extrapolated** from a 100k-row synthetic probe (2.14 KB/row, so about 1.07 GB), not measured at the cap. The fixtures exercise the bounds by patching the limits.
- Order dependence of cross-packet conflicts: within one build, every row of a key with two ref sets is refused. If the conflicting refs arrive in different packets, the earlier admitted row stays and only the later one is refused (`CONFLICTING_REFS`).
- The census is per packet (a delta). The cumulative census is the union over the chain.
