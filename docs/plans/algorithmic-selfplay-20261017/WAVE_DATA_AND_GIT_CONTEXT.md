# Detailed wave data and compact Git context

This is a required design and acceptance contract added by the human on 2026-10-17. It applies to every model version, research wave, aborted attempt, rejected child and inconclusive result. It is proposed implementation work (I66), not a claim that all historical runs already satisfy it.

## One record, three layers

1. **Detailed immutable evidence:** exact training intent, input/replay closure, action and attempt records, numerical training traces, checkpoint/optimizer lineage, raw evaluation outcomes, resource charges, failures and terminal decisions. Keep these in the existing content-addressed local state/mirror abstraction. Proposed additive logical root: `$JOPE_STATE/research/v2/`; resolve `$JOPE_STATE` through `jope-paths`, never invent a competing home directory. Remote publication uses the currently admitted store and principal only. Until verified remote publication, report `LOCAL_COMMITTED_REMOTE_PENDING`.
2. **Packed review data:** versioned Parquet/Zstd trace and outcome shards plus exact content-addressed manifests. Start with the in-repository writer/pack APIs. The 32–128 MiB shard target is a tuning proposal; small completed waves need not wait to fill a shard. A flush deadline and completion reserve prevent lost terminal records. Preserve original evidence member identity and exact S3 VersionId when remote versions exist.
3. **Compact Git review record:** schemas, tiny deterministic fixtures, wave manifest, metrics summary, evidence-linked research card and semantically compressed context. Git stores what a reviewer needs to understand the result and find/reconstruct the detailed evidence; it does not store bulk datasets, tensors, full batch payloads, private row data, credentials or uncontrolled logs.

## Exact proposed Git layout

```text
schemas/jope.wave-evidence.v1.schema.json
docs/contracts/wave-evidence-and-git-context.v1.md
docs/research/waves/INDEX.jsonl
docs/research/waves/<safe-wave-id>/wave.manifest.json
docs/research/waves/<safe-wave-id>/card.json
docs/research/waves/<safe-wave-id>/context.md
docs/research/waves/<safe-wave-id>/metrics.json
```

The schema and contract are shared once. The index contains only identifiers, status, immutable root digests and relative Git paths; do not repeat the full metrics or narrative there. A version record links its parent version and contributing wave roots. Many versions can reference the same content-addressed evidence without copying it.

Proposed canonical byte ceilings: manifest 12 KiB, card 4 KiB, context 4 KiB, metrics 8 KiB, index row 1 KiB, and 32 KiB total new review bytes per ordinary wave. A necessary larger record is an explicit exception with reason and measured bytes. Never truncate uncertainty, denominators, failures or missing-data reasons to meet a cap. Move detailed tables to verified evidence and retain their references. Git already compresses and deduplicates text; keep `context.md` as a compact diffable text projection instead of adding a redundant zipped copy to Git.

## Detailed training trace requirements

Every trace shard identifies schema/source commit, run/attempt/wave/version, parent checkpoint, training intent, objective and optimizer recipe, architecture/context/support/registry versions, numerical backend/device/precision and actual execution grant. Do not mix native401-token and research282-token lineages.

Per actual update/logging unit, preserve:

- Optimizer epoch and local/global step; computed, applied and committed status; rank/world/partition identity; consumption-journal prefix and sampler/RNG cursor needed by the declared recovery promise.
- Input replay shard and occurrence/draw IDs, fresh/replay/source-role counts, label masks, token/example/supervised-unit counts, multiplicity and selection-law provenance. Detailed eligible inputs remain reconstructible through the frozen snapshot rather than copied into each metric row.
- Actual total and per-head losses, reductions/weights and denominators, active versus disabled heads, finite-gradient checks, gradient norm/clip result, learning rate/schedule position, decay groups and numerical overflows/skips. Report missing telemetry as unknown; a disabled zero-weight head is not trained uncertainty.
- KL/entropy/support coverage and the exact distribution/reference used, guard state, calibration and failure/censoring status where the current recipe actually computes them. Do not fabricate fields merely to populate a dashboard.
- Physical CPU/GPU/wall/compile/custody/retry charges, peak memory and retained bytes. Telemetry sampling cadence is declared; optimizer/consumption commit accounting cannot be sampled away.
- Checkpoint commit phase, required object digests, model/optimizer/RNG/scheduler lineage, exported child digest, numerical capability status and subsequent actor-use receipt. Full tensors follow the declared checkpoint cadence, not every metric event.

The data must distinguish a completed optimizer update from a parameter hash changed only by decay, and in-memory applied work from durable committed exposure. Failed/retried work consumes resource while committed logical exposure advances once. A research wave with zero updates is still archived truthfully with its actual result and reason.

## Outcomes and compressed context

Outcome tables retain all proposed/drawn/requested/attempted/evaluated states, exact evaluator/fidelity/fit/split identities, raw measured vectors, selected incumbent, censored timeouts, parity status and reasons for unavailable or provenance-incomplete rows. Preserve both parents and decision laws for edits; an evolved child's probability is not its seed ancestor's log-probability.

The compiled context uses fixed slots:

```text
Question and preregistered hypothesis
Exact version/wave/parent/source/input identities
Treatment, controls, family/role boundary and resource envelope
What actually ran: units, denominators, failures, stops and missing data
Measured effect and uncertainty, with exact evidence locator
Engineering/fixture/CI/development/qualification/serving stage
What changed in the model or search; what was used next
Counterevidence, limits and non-claims
Next falsifier; repeat only if stated conditions change
Detailed evidence root, schema and reconstruction command reference
```

Every numeric statement resolves to a typed source field or an explicit derived expression. An interval containing zero is inconclusive unless an adopted equivalence test supports a null claim. A short card does not erase selection, missingness, exposure or resource caveats. LLM explanation is an optional derived annotation; the deterministic compiler owns numerical and status truth.

## Publication order and Git discipline

1. Commit immutable local training/outcome/checkpoint evidence with its source/input identities.
2. Verify all promised dependencies; publish admitted remote blobs/shards first, manifest last, then exact hash/version readback and conditional authority movement if permitted.
3. Compile the small Git records from verified closure. If remote publication is pending, say so explicitly and prohibit any claim that requires remote durability.
4. Have the sole claimed writer stage only the wave records and approved generated index, run the existing claim/schema/link checks, and commit on the one active reviewed integration branch. Never push directly to `main`, open stacked wave PRs, or sweep unrelated local files into a commit. After this integration PR lands, subsequent wave-record publication still follows the repository's one-PR reviewed lifecycle.
5. Record Git publication as an external receipt naming source-training SHA, wave evidence root, exact publication commit and PR. The wave manifest cannot include its own future Git commit hash; this prevents a circular identity. `GIT_PUBLICATION_PENDING` is separate from a scientific or serving decision.

A failed Git publication retries the compiler/publication step; it never reruns training. A missing evidence body blocks the relevant reconstruction/qualification promise instead of being replaced by a plausible summary.

## Information per byte and acceptance

Measure uncompressed versus packed detailed bytes, bytes per attempted evaluation/committed update, unique evidence bytes versus duplicated bytes, Git bytes per wave, context bytes, source-resolution rate and time to answer fixed review questions. Store the measured denominators and codec/version. High compression alone is not information density: 100% of required facts, failures, uncertainty and declared reconstruction obligations must remain resolvable.

Acceptance cases: a success, rejection, abort, timeout-only wave, interrupted-resume wave and remote/Git outage all produce schema-valid compact records plus honest detailed closure states. Recompiling identical evidence is byte-identical. Removing/changing a referenced shard, exact version, consumption prefix or denominator is detected. A clean independent reconstruction verifies the declared review/resume promise without the original host's mutable indexes. No archive, thinning or cleanup task can remove accepted/active/recovery evidence merely because its compact Git context exists.

## Historical partial runs and disclosure boundary

Backfill only from admitted retained original evidence. Use HISTORICAL_PARTIAL for missing consumption prefixes, unverified admission, truncated runs, inconsistent stop labels or unavailable remote closure. Preserve original label and append correction, never silently normalize it. Historical EI/LRDOSE names are locators to review later under custody permission, not authority to scan protected transcripts/assets or rerun experiments. Scratch symlink paths do not identify stored bytes; bind available custody object hashes/versions, and preserve links themselves.

Git publication is a disclosure boundary even for short summaries. The exporter must enforce an allowlist of fields/roles and governed public aggregate policy. Sealed family identities, targets, restricted outcome feedback, private row data and secrets stay out of Git/context retrieval unless separately authorized for disclosure. A compact aggregate can leak a protected look. Keep a role-appropriate private capsule in evidence and publish only an explicitly permitted Git projection. Add negative fixtures for summary-based leakage and context retrieval from forbidden roles.
