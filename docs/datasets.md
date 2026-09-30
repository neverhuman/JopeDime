# Datasets

> **Status on `main`:** this page describes the target design from the port plan. Run `jope --help` on `main` for the verbs that exist today. There is no `jope learn`, `jope hwm` or `jope scores`; the learning store is the separate `jope-learning` binary, with `hwm` and `scores` subcommands. See the README's "What this SHA actually does".

Plan section: "Datasets: S3 organisation, de-duplication, the 10 test datasets, downloader".
Implementation: crate `jope-datasets` (landed by PR [#27](https://github.com/veox-ai/JopeDime/pull/27),
WP-D1/D5). This page is the contract a contributor needs; the crate's rustdoc has the code-level
detail.

## Identity

A dataset is `<dataset_hash>` = the first 16 hex characters of the canonical sha256
(`meta.json.hash_algorithm == "sha256:first16"`, true for 42 of 45 SOTArena registry entries), so
identity and content-addressing coincide. Its payload is three headerless files
`train.csv`, `test.csv`, `meta.json`; `columns == features + 1`, the last column is the target.
Everything else found next to a dataset in the historical trees (`*.vs`, `*.csvx`,
`extra_features/`, `predictions/`, `reports/`, `signature_manifest.json`) is regenerable from the
payload plus a model bundle and is **not** a dataset.

## Catalog schema (`jope.dataset-catalog/v1`)

One row per dataset (`datasets/<hash>/manifest.json` in S3; `metadata.json` in a committed test
dataset); JSON Schema 2020-12, extending SOTArena's `dataset.schema.json` required set
`id, task, target, files, split`:

| Field | Content |
|---|---|
| `dataset_hash` / `canonical_sha256` | `^[0-9a-f]{16}$`; CI asserts `canonical_sha256[:16] == dataset_hash` |
| `task` | `regression | binary | multiclass | multi_label | timeseries` |
| `origin` | `real | synthetic` |
| `shape` | `{rows, features, categorical_features, columns}` |
| `target` | `{name, index: -1, type}` |
| `files.{train,test,meta}` | `{sha256, bytes, rows, columns, has_header: false}` — `sha256` is the blob key |
| `split` | `{method, seed, train_rows, test_rows}` |
| `provenance` | `{platform, dataset_id, url, retrieved_at, dedupe_fingerprint, citation, license {spdx, status, evidence_url, text_sha256}}` |
| `generator` | `{name, version, seed, attempt_seed, recipe_sha256, script_sha256}` — required iff synthetic |
| `collections[]`, `aliases[{path, sha256}]` | memberships; other locations of byte-identical copies |
| `ingest` | `{source, relpath, ingested_at, tool_version}` |

A collection (`jope.dataset-collection/v1`, `datasets/manifest.json` for `test-10`) is
`{name, version, visibility (public|private), task, members[], policy {license_required, allowed_spdx[]}, members_sha256}`,
write-once per version in S3 (`collections/<name>/v<N>.json` + `LATEST`).

## Tiers and buckets

`s3://veox-jopedime-datasets-use2` (us-east-2; versioning, Object Lock, SSE-S3, Block Public
Access; public access later only through CloudFront OAC): `blobs/sha256/<64hex>` (every file once;
`x-amz-meta-sha256` + `ChecksumSHA256`; tags `tier`, `kind`), `datasets/<hash>/manifest.json`,
`collections/`, `catalog/v<N>/catalog.jsonl`, `prepared/<view>/<hash>/`, `generators/<name>/<version>/`,
`ingest/reports/<ts>/{duplicates,rejects}.json` + `ledger.jsonl`.

| Tier | Content | Storage class |
|---|---|---|
| T1 (≈ 10 GiB) | the mirror `dataset` class (12,919 distinct sha), the real regression registry, `prepared` views, raw corpus | Standard |
| T2 (≤ 2.3 GiB + ≈ 20 GiB derived) | non-synthetic `quant/binary` remainder; packed/derived training inputs go to the archive bucket | Standard / Deep Archive |
| T3 (≈ 127 + 20 + 119 GiB) | synthetic binary (102,005 dirs), regression `synth_*` (25,205 dirs), synthetic sources | Glacier Instant Retrieval; sources Deep Archive |

**What the bucket holds (metadata inventory, 2026-09-23).** Only the real part of T1 has been
uploaded: 3,859 `origin: real` datasets (regression 2,611, binary 843, multi_label 322,
timeseries 83) as 11,577 `blobs/` objects (3,224,104,137 B, about 3.22 GB), 3,859
`datasets/<hash>/manifest.json`, `catalog/` and `collections/`, for 15,442 objects and
3,234,922,515 B in total. The inventory did not identify the mirror `dataset` and `prepared`
classes, T2 or T3 in this bucket. The `prepared/`, `generators/` and `ingest/reports/` prefixes are planned and
absent. The bucket has versioning and Object Lock enabled with no default retention, and no
configured replication destination. A HeadBucket probe for `veox-jopedime-datasets-usw2`
returned HTTP 404 on 2026-09-23; this inventory does not rule out copies elsewhere. Per-prefix figures are in
[`catalogs/s3.json`](../catalogs/s3.json).

## Ingest and de-duplication (`jope datasets ingest --source mirror|quant [--dry-run] [--resume]`)

1. Enumerate directories `^(synth_)?[0-9a-f]{16}$` holding all three files; stream-sha256 them.
2. **Reject unless `meta.sha256[:16] == dir hash`** (reason `dir_name_pattern`, `meta_sha256_missing`,
   `meta_sha256_mismatch`, …, all written to `rejects.json`).
3. De-duplicate by file sha256 (one blob) and by `dataset_hash` (one catalog row; every other
   location becomes an `aliases[]` entry and a `duplicates.json` line). The root-level
   `quant/{<16hex>,synth_*}` copies are considered by content hash and metadata, since a
   synthetic path can contain a real dataset. Confirmed synthetic rows are excluded only with
   `--real-only`; unknown origin remains in the private catalog. Non-dataset research files are
   handled by the custody inventory below.
4. Merge provenance from `meta.json` (`source_platform/source_dataset_id/source_url/source_license`),
   the mirror index (its sha must equal the computed sha) and, for real datasets, the `hyper`
   manifests as a cross-reference.
5. Upload: stream source files in bounded multipart parts, hash source bytes during transfer,
   and stream the entire stored object back to verify SHA-256. Existing keys are read and hashed,
   not trusted by size alone. `--resume` rechecks ledger entries before skipping them. Publishing
   merges earlier catalog rows and verifies their referenced blobs before a conditional `LATEST`
   update. A failed or changed source does not advance the pointer.

### Complete quant custody

`jope-ci custody-inventory --root /home/ubuntu/remote_super --out CUSTODY/remote-super.jsonl`
records every reachable file, directory and symlink with exact path bytes, type, size, metadata,
SHA-256 for files, classification and proposed USB shard. Incomplete dataset directories receive
explicit records. Read errors or files that change while scanned leave a partial inventory and
fail the census. Unknown origin is treated as unresolved research data.

`jope-datasets custody-upload --inventory CUSTODY/remote-super.jsonl --root
/home/ubuntu/remote_super --out CUSTODY/quant-catalog.jsonl` uploads every real or unresolved
quant file, including models, scores and files in incomplete datasets, into the private canonical
datasets bucket. Existing `blobs/sha256/` content is fully hashed and reused, so cataloged
train/test files are not stored a second time. Only metadata-confirmed synthetic train/test
payloads are omitted. Each catalog
row binds the exact source path to its SHA-256, byte count, object key and S3 version. A durable
partial ledger resumes after interruption by rehashing both source and stored object. The tool
publishes an immutable inventory and a full catalog superset, then conditionally advances its own
`custody/quant/catalog/LATEST`. This path catalog supplements the dataset catalog and does not
make unknown-license data public. No source may be removed on the strength of an incomplete census
or an unverified upload.

### Virtual evaluation splits

`jope-datasets split-publish` takes a source JSON binding the train/test blob SHA-256s,
schema SHA-256 and row counts, plus the corresponding headerless CSVs and one JSON annotation
per row. Annotations supply task labels, known groups and chronological times; the publisher
derives exact duplicate identities from parsed CSV rows. It verifies the local files and both
canonical S3 blobs before publishing immutable row-index bundles. The virtual table is original
train rows followed by original test rows. Each fold stores sorted test indices; training rows
are their complement. No per-fold CSV copies are created.

The historical original train/test boundary is one `original` track. New five-repeat folds are
the `historical_test_exposed_exploratory` track, with up to five folds depending on class,
group, duplicate and time constraints. An impossible case is published as an explicit exception.
Classification and multilabel folds balance labels while keeping known groups and duplicate
rows together. Time-series folds use chronological forward chaining. Scores carry source,
split, track, algorithm/version, metric, seed and prediction hashes; the reader rejects a score
for a different bundle or track. `split-scores-verify` also streams each named prediction object,
checks its SHA-256 and requires the recorded S3 version to match. Callers must preserve historical
scores in the original track.

The PR #11 dry run over the real tree reproduced the plan's numbers exactly: 132,537 candidates →
105,482 rows, 316,446 distinct blobs (138.98 GB), 27,519 + 23,292 duplicates, 27,429 rejects.

**Open decision — `meta_sha256_missing` (owner):** 27,055 of those rejects are directories whose
`meta.json` carries no `sha256` — 25,197 `regression/synth_*` (the `synthetic_regression_mixed`
1.0/2.0 generator, whose source exists nowhere) and 1,856 others. Rule 2 cannot admit them without
a content identity. Options on the table: (a) compute `sha256(train.csv ‖ test.csv ‖ meta.json)`
at ingest and record `hash_algorithm: "sha256:first16-ingest"` with the directory name as an alias
(admits them, but their `dataset_hash` would no longer be self-describing); (b) keep them rejected
and let the `synth_*` families live only in the archive bucket's `derived/quant-packed/` copy
(that prefix was absent from the 2026-09-23 current-object inventory);
(c) regenerate under a committed generator (impossible for regression until a generator source is
found; WP-D7 covers the binary generator). Until the owner decides, they stay in `rejects.json` and
no collection references them.

## Licence rule (enforced by `jope datasets verify --catalog`)

`provenance.license.status ∈ {recorded, unknown, restricted}`. `recorded` needs an SPDX expression
(or `LicenseRef-*`, e.g. `LicenseRef-US-Gov-Public-Domain`) and an `evidence_url`; `unknown` is
legal only in private collections; `restricted` never in a public collection; a public collection
with any non-`recorded` member fails CI. Planned collections: `test-10` (public once the switch is
flipped), `golden42`/`golden46` (private until their 24 blank licences are backfilled),
`registry-real-v1` (private), `synthetic-binary-0.2.0`,
`synthetic-regression-2.0.0` (own-generated, Apache-2.0). Only `test-10` and `registry-real-v1`
were found in the 2026-09-23 bucket inventory. `registry-real-v1` has 3,859 members: 3,755 with licence status `unknown`
and 104 `recorded`. These figures were counted on 2026-09-23 from `catalog/v1/catalog.jsonl`
(sha256 `ab9fda8d2dea46067b70a42812e9d3c1d9dc6ba025780100df34377e81aa1120`). The older figure of
1,416 unknown covered the 2,613 real regression datasets only.

## The ten committed test datasets (`datasets/test/`, collection `test-10`)

Chosen using the historical SOTArena `tabular-regression-v2` published snapshot, which listed
jope-dime at Elo rank 1 (2259.8) with 11 strict wins. That view includes development/test-selected
rows; it is not a held-out result or a current ASM qualification. The retained corrected
benchmark-valid snapshot lists `jope-dime-regressor` at rank 31 (1687.0); these different
eligibility views must not be combined. See the [historical plot data and provenance](internal/asm-manual/data/plot-data-manifest.json).
`datasets/manifest.json` is the collection; each `metadata.json` is the catalog row;
each `LICENSE` is the upstream text. Total 1,096,413 B (largest 540,717 B); no LFS.

| Slug | `dataset_hash` | Platform | Licence (recorded) |
|---|---|---|---|
| `pmlb-229-pwlinear` | `1cb97b1d7314d43d` | PMLB | MIT |
| `pmlb-first-principles-supernovae-zg` | `693890aae9dc860b` | PMLB | MIT |
| `pmlb-695-chatfield-4` | `047e62b52a3f4ede` | PMLB | MIT |
| `rdatasets-agridat-bancroft-peanut-uniformity` | `79e600bc5fcd5da7` | Rdatasets (CRAN agridat) | MIT — the registry's "GPL" was wrong; CRAN `DESCRIPTION` is the evidence |
| `rdatasets-stevedata-chile88` | `c7d815714ca52bce` | Rdatasets (stevedata) | GPL-2.0-only |
| `rdatasets-wooldridge-barium` | `30ef3eaa6d2971a9` | Rdatasets (wooldridge) | GPL-3.0-only |
| `uci-parkinsons-telemonitoring` | `21fbfe8b53dbcd60` | UCI | CC-BY-4.0 (+ citation) |
| `openml-29-credit-approval` | `b5beff4a53fa12c0` | OpenML (UCI 27) | CC-BY-4.0 |
| `openml-23-cmc` | `c32fa09b48fb2f27` | OpenML (UCI 30) | CC-BY-4.0 |
| `socrata-cdc-food-safety-culture` | `43b4cc3ab0ebbd97` | Socrata (CDC) | LicenseRef-US-Gov-Public-Domain |

Excluded on purpose: hash-only datasets without provenance, `ZTS_small`, both NOMAD releases
(restricted encrypted bundle). CI (`jope datasets verify --repo datasets/test`, a hygiene step
with PR #11): each file's sha256 equals `metadata.json.files.*.sha256`, the SOTArena registry and
(on main) the S3 blob key; schema validation; headerless and `columns == features + 1`; caps
≤ 10 datasets, ≤ 600 KiB each, ≤ 1.5 MiB total; the public-collection licence rule is a warning
while private and the gate for going public; `JOPE_OFFLINE=1 jope datasets fetch test-10`
succeeds with the repo directory as the seed cache.

## Downloader (`jope datasets fetch <collection|task|id>`)

One transport, the `object_store` crate (pinned 0.13.2 — 0.14 pulls a licence `deny.toml`
rejects): `s3://` with env/OIDC credentials now, anonymous or an `HttpStore` over CloudFront later,
`file://` in tests; `JOPE_DATASETS_URL` defaults to `s3://veox-jopedime-datasets-use2`. Cache under
`$JOPE_HOME` (default `${XDG_CACHE_HOME:-~/.cache}/jope`): `datasets/blobs/sha256/<64hex>`
(download to `.partial`, hash, rename only on match; a mismatch is deleted and reported),
`datasets/by-id/<hash>/{train.csv,test.csv,meta.json,manifest.json}` (hardlinks to blobs),
`datasets/catalog/`, `datasets/collections/`. 16 parallel gets, ranged-GET resume above 8 MiB,
`--offline`/`JOPE_OFFLINE=1` serves only the cache. `jope datasets verify [--quick]` re-hashes;
`jope datasets link --from <dir>` seeds the cache from a local tree by hashing (xbabe2 never
downloads its own data). The engine reads `JOPE_REGISTRY` (default `$JOPE_HOME/datasets/by-id`),
the same `<hash>/{train.csv,test.csv,meta.json}` layout as the historical
`quant/regression/<hash>/`, so no reader changes; `fetch --print-registry` prints the path for CI.

Adding a dataset: [runbooks/add-a-dataset.md](runbooks/add-a-dataset.md).

### AWS credential selection (PR63 correction)

Signed S3 stores retain static environment keys, web identity (OIDC), ECS and EKS
credential providers. A complete static key pair wins; otherwise `JOPE_AWS_PROFILE`
then `AWS_PROFILE` selects a profile through `aws configure export-credentials`.
A selected profile replaces the complete key/token tuple, including any stale ambient
session token. Native OIDC/container providers remain available when no profile is
selected. `JOPE_AWS_ALLOW_IMDS=1` explicitly enables instance metadata if no other
source is configured; incomplete sources otherwise fail before network access.
`JOPE_DATASETS_ANON=1` bypasses credential resolution for unsigned requests.

Profile resolution requires the AWS CLI. It does not mutate the caller's environment
and does not echo either output stream from a failing helper. The helper currently
has no process deadline; prepare and verify credentials before a bounded campaign.
Exported temporary credentials are a snapshot and are not refreshed during that process.
