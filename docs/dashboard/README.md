# JopeDime research observatory

A read-only Rust API and Vite/TypeScript/React browser app for real JopeDime training,
GP library snapshots, fleet resources, S3 metadata and durable research waves.
See [implementation plan](WORKPLAN.md) and [KPI catalog](KPI_CATALOG.md).

## Build and run

From your claimed JopeDime lane (use `rtk` on hosts requiring it):

```sh
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo build --locked -p jope-observatory --bins -j4
cd web/training-dashboard
npm ci
npm run typecheck
npm test
npm run build
```

Node supported versions: 24.15+ in the 24 LTS line, or 26+. Use the repository fleetcap
wrapper and nice10 for remote Cargo/npm work; reuse the lane target. No GPU required.

Create an external operator configuration using [config.example.json](config.example.json).
All files and their ancestors must be real paths, without symlinks or parent components.
Replace paths with the current owning captain's evidence paths. Paths are never accepted
from an HTTP request. Keep credentials, private raw logs, weights and corpora outside Git.

```sh
target/debug/jope-observatory --config /home/ubuntu/jope-evidence/observatory/config.json
```

Default URL: `http://127.0.0.1:18894`. This is separate from Admiralty's port18890.
From the user's computer, with a free local18894:

```sh
ssh -N -L 18894:127.0.0.1:18894 -p 6969 ubuntu@162.218.217.123
```

Open `http://127.0.0.1:18894`. The HTTP Host must match the configured numeric listen
authority; use that URL rather than a different hostname. No direct public listener is
supported. Coordinate an authenticated reverse-proxy link through the existing Admiralty
owner rather than starting a replacement dashboard deploy. For frontend development run
`npm run dev` with the Rust server on18894; Vite rewrites proxy Host appropriately.

`--once` emits a snapshot and exits for diagnostics. A healthy `/healthz` reports only
observer availability, with `authority:false`; it proves no autonomous engineering/training.

## Feeds and interpretation

| View | Input | Refresh / limits |
| --- | --- | --- |
| Training | Explicit campaign root `config.json`, `training/progress.json`, loaded receipt | 5s; known schema and config-byte SHA256 must match; missing/mismatch is unknown |
| Loss | `loss_history` fixed panels with dropout=false | Panels/objectives get separate lines; no smoothing, synthetic points or panel/full-validation conflation |
| Checkpoints | `training/checkpoints/<six-digit-step>/manifest.json` | Latest24 shown; bounded directory census; hashes are producer declarations, not repeated weight reads |
| GP library | Explicit SHA256-pinned `{programs:[{program_sha256,dsl,cid}]}` snapshots | Deduplicate declared identities, family/operator/search/page/DSL drill-down; no global completeness or evaluator score inferred |
| S3 | Optional `jope.storage-observation/v1`; otherwise dated `catalogs/s3.json` | Cached metadata only. Metric observation date distinct from fetch time. Unknown buckets stay null |
| Datasets | Versioned dataset census in `catalogs/s3.json` | Catalog date prominently shown; contexts/records/objects are not dataset IDs |
| Results | `docs/research/waves/*/{wave.manifest,card,metrics}.json` | Compact assertions and hashes; producer archive claims do not prove restoration or Git publication |
| Code | Optional external `jope.code-identity-observation/v1` | Actual Git/build and training source proof supplied by deployment owner, never guessed from a version string |
| Fleet | Up to4 `jope.fleet-node-observation/v1` snapshots | Age shown; absent nodes are unknown, not idle; host resource use not attributed to learning |

The optional `collect-fleet` executable performs read-only resource probes of exactly
xbabe1/2/3 and xbabe4, never xbabe0. It reads only fixed native resource commands (no argv or
process environment). On xbabe2 it samples local counters directly; xbabe1 uses its existing
trusted SSH host-key alias. Host verification remains enabled. Example: `target/debug/collect-fleet
--output-dir /home/ubuntu/jope-evidence/observatory/fleet --cycles 120 --interval-seconds 30`.
The default is finite. An explicitly owned dashboard deployment may pass `--continuous` for a
persistent native telemetry feed under fleetcap/nice15; this has no training/model-provider authority.
Run exactly one collector. Its failure or expiry becomes visible in the UI. No agent timer,
provider retry, GPU job or new screen is installed.

An optional `collect-datasets` command creates a compact census from a locally hash-verified,
operator-recovered exact S3 VersionId of `catalog/v1/catalog.jsonl` in the ordinary datasets bucket.
It accepts `--catalog PATH --sha256 HASH --source-s3-uri URI --source-version-id ID
--source-modified-at ISO --output PATH`. Dataset rows are metadata, not training payloads; duplicate
IDs, malformed/unknown schemas and empty input fail. Configure `dataset_snapshot` to this compact
output; the UI shows capture and original source identities. Remote HEAD/GET/latest-key evidence
belongs in the operator receipt; importing a file alone is not a fresh S3 verification.

`GET /api/v1/snapshot` excludes GP program rows to keep repeated polls compact.
`GET /api/v1/gp` supplies library detail only when requested; browser reloads it on source
identity change. POST and unknown API routes are refused. Static assets have no external
fonts/scripts. Source read failures are summarized without dumping raw errors/credentials.

## Optional S3 refresh

This is an explicit finite metadata collection, **never a browser-triggered AWS scan**:

```sh
target/debug/collect-storage --catalog catalogs/s3.json \
  --profile veox-jepson-prod --expected-account 714044927654 \
  --aws /home/ubuntu/.local/bin/aws \
  --output /home/ubuntu/jope-evidence/observatory/storage.json
```

It checks STS account, configured bucket/regions, bounded AWS/S3 CloudWatch daily metrics,
and writes atomically. No object payloads, sealed data, credentials, configuration changes,
or alternate-profile retries. CloudWatch publication is normally daily: this cannot promise
second-by-second S3 bytes. Physical storage may include versions, storage overhead and replicas;
it differs from the dated catalog's current-key payload-byte inventory. Never add those totals.
CloudWatch object counts include versions, delete markers and incomplete-upload parts; see [AWS metric definitions](https://docs.aws.amazon.com/AmazonS3/latest/userguide/metrics-dimensions.html).
Missing common timestamps/incomplete metrics are unavailable, not zero. A partial estate total
is labeled partial. Coordinate refresh scheduling with the existing fleet owner; no timer is
installed by this package. Do not run an unbounded object inventory every UI poll.

## Source and fleet producer contracts

Code identity example (external operator receipt, not automatically trusted scientific proof):

```json
{"schema":"jope.code-identity-observation/v1","observed_at":0,"dashboard_commit":"FULL_SHA","dashboard_dirty":false,"scope":"actual source/build observation","training_runs":{"CAMPAIGN_DIRECTORY":{"source_commit":"TRAINING_SOURCE_SHA","verified_files":21,"receipt_sha256":"SOURCE_CLOSURE_PROOF_SHA256","scope":"actual copied runtime sources checked against this Git commit"}}}
```

Fleet node example:

```json
{"schema":"jope.fleet-node-observation/v1","host":"xbabe2","observed_at":0,"cpu_count":16,"load":[0,0,0],"memory_total_bytes":0,"memory_available_bytes":0,"disk_available_bytes":0,"gpus":[{"name":"GPU model","utilization_percent":0,"memory_used_bytes":0,"memory_total_bytes":0}],"workload_scope":"Host-wide counters only; no learning attribution"}
```

Those zeroes illustrate **measured zero** only; producers emit null for unavailable values.
No argv, process environment, tokens, user prompts or sealed records belong in these feeds.
Original failures and dated source identities remain in the owning wave. Compact Git wave
manifests/cards/metrics/context/index plus immutable compressed bulk archives, S3 VersionIds,
decoded/member hashes and independent restoration evidence remain the preservation contract.

## Operational handoff

The current captain owns run admission, result publication and finite budgets. The dashboard
writer owns this crate/UI until review/merge; existing Admiralty owners retain integration and
deploy authority. Keep config/runtime/evidence outside the working tree; record source head,
dirty state, binary/UI/config hashes, listener PID/URL, sample identity/counts and actual tests.
Never delete training checkpoints, symlinks, worktrees or old results to make the dashboard green.
