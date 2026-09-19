# Runbook: add a dataset

Two different things are "adding a dataset": (A) ingesting one into the S3 catalog so campaigns
and the learning loop can fetch it, and (B) committing one of the ≤ 10 test datasets to
`datasets/test/`. Both go through `jope datasets` (crate `jope-datasets`, PR #11) and both are
subject to the licence rule in [docs/datasets.md](../datasets.md).

## A. Ingest into the catalog (maintainer, `JopeDimeMaintainer` credentials)

1. Put the payload in a directory named by its hash: `<dataset_hash>/{train.csv,test.csv,meta.json}`,
   headerless, last column = target, `meta.json` carrying `sha256` (the canonical sha256 whose
   first 16 hex are the directory name), `hash_algorithm: "sha256:first16"`, `task`,
   `source_platform`, `source_dataset_id`, `source_url`, `source_license`, `sampling{method,seed}`.
   A directory whose `meta.json` has no `sha256` is rejected (`meta_sha256_missing`) — that is a
   rule, not a bug; see the open decision in [docs/datasets.md](../datasets.md).
2. Dry run, then ingest:
   ```bash
   cargo run -p jope-datasets --locked -- ingest --source quant --root <parent-dir> --dry-run
   cargo run -p jope-datasets --locked -- ingest --source quant --root <parent-dir> --resume
   ```
   The dry run prints rows/blobs/duplicates/rejects and writes `duplicates.json` / `rejects.json`;
   read them before uploading. Ingest uploads blobs with `If-None-Match: *` (existing keys are
   never rewritten), verifies each `HeadObject` checksum, appends `ingest/ledger.jsonl`, and is
   safe to kill and rerun.
3. Record the licence: `provenance.license.{spdx, status: recorded, evidence_url, text_sha256}`.
   `unknown` keeps the dataset out of every public collection; `restricted` keeps it out of public
   collections forever.
4. Add it to a collection: edit (or create) `collections/<name>/v<N+1>.json` — collections are
   write-once per version, so a new member means a new version and `LATEST` moves. A public
   collection with any non-`recorded` member fails `jope datasets verify --catalog`.
5. Verify from a clean cache: `JOPE_HOME=$(mktemp -d) cargo run -p jope-datasets -- fetch <name> && … verify`.

## B. Commit a test dataset (`datasets/test/`, collection `test-10`)

Only when a dataset earns it (rank-1 on SOTArena or a coverage gap in the emitter matrix) and the
caps hold: **≤ 10 datasets, ≤ 600 KiB each, ≤ 1.5 MiB total, plain files, no LFS**. To add one you
almost certainly remove one.

1. Choose a slug `<platform>-<id>-<name>` (e.g. `pmlb-229-pwlinear`); the emitter matrix and
   docs refer to datasets by slug, the catalog by `dataset_hash`.
2. Create `datasets/test/<slug>/` with `train.csv`, `test.csv` (headerless; `columns == features + 1`),
   `metadata.json` (= the catalog row of step A, including `files.*.sha256`) and `LICENSE` (the
   upstream licence text; for `LicenseRef-US-Gov-Public-Domain` a short statement with the source URL).
3. Update `datasets/manifest.json`: append the `dataset_hash` to `members`, bump `version`,
   recompute `members_sha256`, and make sure the licence's SPDX id is in `policy.allowed_spdx`.
4. Run the verifier the hygiene lane runs:
   ```bash
   cargo run -q -p jope-datasets --locked -- verify --repo datasets/test
   bash deploy/ci/check.sh
   ```
   It checks every file's sha256 against `metadata.json`, the SOTArena registry and (on main) the
   S3 blob key, the schema, headerlessness, the caps, and — as a warning while private, a failure
   once the switch is flipped — the public licence rule.
5. If the emitter matrix should exercise it, add a `matrix` line in `deploy/ci/emitters.sh` with
   a genome, run `UPDATE=1 bash deploy/ci/emitters.sh` once to create the pins under
   `crates/feat-math/tests/fixtures/emitted/`, and commit the pins with the data.
6. PR description: platform, upstream URL, licence and evidence URL, why this dataset (the
   SOTArena row), which dataset it replaces and why. The reviewer re-runs step 4.

## Removing one

Delete the directory, drop the hash from `datasets/manifest.json` (new version), keep the catalog
row in S3 (the blobs are immutable; a removed test dataset simply stops being committed), update
any `emitters.sh` line and its pins, and say in the PR why it goes.
