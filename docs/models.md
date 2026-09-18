# Models and weights

Weights never enter git. The repository commits **manifests** under `models/`; the bytes live in
one private, content-addressed bucket and are fetched on demand into a local cache by
`jope models` (crate `crates/jope-models`, binary `jope-models`, mounted as `jope models …`).

## Bucket layout and identity

`s3://veox-jopedime-models-use2` (us-east-2; SSE-S3, versioning, Object Lock governance):

| Key | Content |
|---|---|
| `blobs/sha256/<64 hex>` | every file once, keyed by its sha256 — immutable, never rewritten |
| `bundles/<bundle_id>/manifest.json` | the bundle manifest (`jope.model-bundle/v2`), written last on publish |

A **bundle** is a named, immutable set of files (`bundle_id` matches `^[a-z0-9][a-z0-9.-]{1,63}$`).
Corrections are published under a new id; an existing blob or manifest is never overwritten
(`If-None-Match: *` — `object_store::PutMode::Create`).

### `jope.model-bundle/v2` (schema: `schemas/jope.model-bundle.v2.schema.json`)

```json
{
  "schema": "jope.model-bundle/v2",
  "bundle_id": "policy-step-60000",
  "bucket": "veox-jopedime-models-use2",
  "published_at": "2026-09-18T20:17:48Z",
  "files": [
    { "path": "policy.safetensors", "sha256": "…64 hex…", "bytes": 111199088,
      "blob": "blobs/sha256/…64 hex…", "kind": "policy",
      "license": { "schema": "jain.model-license/v1", "spdx_expression": "LicenseRef-Jain-Internal-Model",
                   "license_class": "eligible_non_open", "redistribution": "internal",
                   "notice_paths": ["docs/notices/jain-governed-models.txt"], "…": "…" } }
  ],
  "notices": [ { "path": "docs/notices/jain-governed-models.txt", "text": "…" } ],
  "…": "free-form provenance (policy_version, source_checkpoint, encoders, consumers, …)"
}
```

* `files[].path` is the bundle-relative path the consumer reads (`$JOPE_HOME/models/bundles/<id>/<path>`);
  `sha256` is the identity and the blob key (`blob` must equal `blobs/sha256/<sha256>`).
* `files[].license` is a `jain.model-license/v1` block or `null` (metadata files). `redistribution`
  is one of `permitted | internal | prohibited | conditional`; `prohibited` files are only fetched
  with `--accept-restricted`. `notice_paths` name entries of `notices[]`.
* `notices[]` embed the notice texts; `fetch` prints them and writes them at
  `bundles/<id>/<notice path>`.

### Committed under `models/`

| File | Purpose |
|---|---|
| `models/bundles/<id>.json` | one bundle manifest per published bundle (byte-identical to the bucket copy) |
| `models/manifest.json` | `jope.model-index/v1`: bucket, layout, and per bundle the manifest path, its sha256, `s3_key`, file count, byte total and licence set |
| `models/PINNED` | the bundle ids the `main`/`nightly` lanes fetch (one per line, `#` comments) |

`deploy/ci/manifests.sh` (Python, dependency-free) and `jope models verify --manifests-only`
(Rust, embedded schema) both check that the three agree; neither touches the network.

Current bundles: `policy-step-60000` (step-60000 policy + context transforms, 119.6 MB),
`jope-prime-gen31-v2` (the starforge encoder bundle: foundation/target/Hyperion v5 encoders, gen31
policy, chimera experts, upstream `model-bundle.v1.json`; 1.10 GB; three files are
`redistribution: prohibited`), `hyperion-v7` (246.5 MB).

## Local cache (`$JOPE_HOME/models`)

`$JOPE_HOME` defaults to `${XDG_CACHE_HOME:-~/.cache}/jope` (shared with `jope datasets`).

```
$JOPE_HOME/models/
  blobs/sha256/<sha256>           downloaded to <sha256>.partial, hashed while streaming,
                                  renamed only on a match (a mismatch deletes the partial)
  bundles/<id>/<path>             hardlinks to the blobs (copies across devices)
  bundles/<id>/docs/notices/*.txt the embedded notices
  manifests/<id>.json             the bundle manifest (outside the bundle directory, whose
                                  paths belong to the bundle)
```

Nothing is linked under `bundles/<id>/` until every selected blob has verified.

## Commands

```
jope models fetch <bundle_id> [--component <path|dir/>]… [--offline] [--accept-restricted]
jope models fetch --pinned [--accept-restricted]        # every id in <repo>/models/PINNED
jope models verify <bundle_id> | --all [--quick]        # re-hash a populated cache
jope models verify --manifests-only                     # committed manifests only, no network
jope models publish <manifest.json> --from <dir> [--store <url>]
jope models path <bundle_id> [<path>]                   # cache path; exit 1 when not fetched
```

`fetch` resolves the manifest from the checkout (`<repo>/models/bundles/<id>.json`, `--repo`
defaults to `.`), then the cache, then the store; prints every notice and one
`<path>  <spdx_expression> / <redistribution>` line per selected file **before** downloading;
refuses the whole fetch when a selected file is `redistribution: prohibited` unless
`--accept-restricted`; downloads up to 16 blobs concurrently with ranged-GET resume above 8 MiB;
prints the bundle directory on success. The last line of stdout is the bundle directory.

`publish` hashes every `<dir>/<path>` against the manifest first (nothing is uploaded if any file
differs), uploads missing blobs create-only, then writes `bundles/<id>/manifest.json` last; an
existing manifest must be byte-identical (otherwise: new `bundle_id`).

Wiring consumers:

```sh
export HYPERION_V7_WEIGHTS=$(jope models path hyperion-v7 model.safetensors)
export JOPE_MODELS_ROOT=$(jope models path jope-prime-gen31-v2)   # starforge parity tests
```

## Environment

| Variable | Meaning |
|---|---|
| `JOPE_MODELS_URL` | store URL; default `s3://veox-jopedime-models-use2`; `file:///dir` for tests and mirrors |
| `JOPE_MODELS_ANON` | `1` sends unsigned S3 requests (public-read or CloudFront later) |
| `JOPE_HOME` | cache root (default `${XDG_CACHE_HOME:-~/.cache}/jope`) |
| `JOPE_OFFLINE` | `1` serves only the cache (same as `--offline`) |
| `JOPE_MODELS_ROOT` | read by the starforge weight-bound tests: the fetched `jope-prime-gen31-v2` bundle directory (or a jain-starforge-style checkout with `artifacts/model-bundle.v1.json`). Unset → each such test prints `SKIPPED: <test>: <reason>` and passes |
| `AWS_*` | credentials/region for `s3://` (env, profile or OIDC role); the region defaults to `us-east-2` |

The starforge tests accept the fetched layout directly: the upstream `model-bundle.v1.json` names
files as `artifacts/<path>` while the v2 bundle stores them at `<path>`, so the test helper
(`crates/jope-encoders/tests/support/models_root.rs`) builds a hardlinked sibling view
`<bundle dir>.starforge/` in the crate's layout the first time it runs.

## CI

* **PR lanes fetch nothing.** `deploy/ci/check.sh` runs `deploy/ci/manifests.sh` and
  `jope-models verify --manifests-only`; the starforge weight-bound tests print `SKIPPED:` and pass.
* **main / nightly** (self-hosted, OIDC read role): `jope models fetch --pinned --accept-restricted`
  then `jope models verify --all`, and export `JOPE_MODELS_ROOT=$(jope models path
  jope-prime-gen31-v2)` before `cargo test -p starforge`; the lane fails on any `SKIPPED:` line.

## Publishing a new bundle (maintainers)

1. Stage the files in a directory laid out exactly as the bundle (`<dir>/<path>`).
2. Write `models/bundles/<new-id>.json`: `schema`, a new `bundle_id`, `bucket`, `published_at`,
   `files[]` (path, sha256, bytes, blob, kind, licence block or `null`), `notices[]`, and the
   provenance fields (`source_checkpoint(_sha256)`, `policy_version`, `consumers`, …).
3. `jope models verify --manifests-only` after adding the row to `models/manifest.json`
   (`manifest_sha256` = sha256 of the file, `files`, `bytes`, `licenses` = the set of
   `spdx_expression`s with `none` for unlicensed files) — or run `deploy/ci/manifests.sh`.
4. With the publish role: `jope models publish models/bundles/<new-id>.json --from <dir>`.
5. Read it back: `jope models fetch <new-id> [--accept-restricted] && jope models verify <new-id>`.
6. Add the id to `models/PINNED` if the main lane must have it; commit the manifests through the
   reviewed lifecycle. Never edit a published manifest — publish `<id>-vN+1`.
