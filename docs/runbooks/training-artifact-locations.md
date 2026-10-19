# Training artifact locations and recovery gates

Verified 2026-10-18 UTC against source `a2759e9d375d0697ed40a32c23ac297b5de42347`. Read `catalogs/training-artifacts.json` first on every new research wave. It is the location index; it is **not** a native admission policy. The assistant named Root is not a signing or evaluator authority.

## Current answer

A historical teaching-corpus index, matching vocabulary, native authority manifest and original local witness were located. **A complete current production-admitted training package remains unverified.** The historical index is `parallel.teaching_shards.v1`, `qualification: true`, `label_policy: reference_v1`; current production packing requires `strict_midrank_v4` and an original `parallel.label_admission.v2` manifest. Renaming files or changing those fields cannot admit it.

The historical index declares 10,214,332 records, 8,894 contexts and 7,877 shards. These are index counts, not verified current shard availability, independent dataset-family counts, optimizer exposures or new training progress. All 7,877 indexed shard **directories** have all three expected current S3 members: 23,631 keys total. Every `records.bin` object length matches its declared record count × 532 bytes (5,434,024,624 record bytes; 8,691,019,699 bytes across all members). Exact-version hashes passed for a fixed first meta-train shard and root index/vocabulary/groups/provenance. Full payload hashes for all shards remain unverified, so no complete corpus closure is claimed. The earlier Root zero-match census compared directory paths against file/object names; it was a lookup error, not missing-data evidence, and is retained with an explicit correction.

## Exact locations

Use AWS on xbabe2 through `/home/ubuntu/.local/bin/aws --profile veox-jepson-prod --region us-east-2`. Verify STS identity at use; the verified account was `714044927654`. Credentials remain outside Git. An S3 resource-policy deny stays a blocker; do not switch identities to get around it.

The historical corpus directory is:

`s3://veox-jopedime-learning-use2/staging/lineage-b/accepted60000-restored/parallel/runs/eight-a100-recovery-20260912/learning/snapshot-000097/corpus/`

- `index.json`: 60,589,288 B; SHA-256 `1c840597554f76598b7c173a0de6f5d13e1b29bfa971f4fa49777d9d1da47798`; VersionId `mH0mMu2Q3xrAk.LjmJC_S.0hNOjt2AJB`. Exact-version GET hash matches the historical accepted-parent decision. The private recovered gzip is `/home/ubuntu/jope-evidence/fleet-rnd-20261018T1140Z/s3-training-discovery-20261018T1550Z/metadata/HISTORICAL_ACCEPTED60000_CORPUS_INDEX.json.gz` (5,626,110 B). It has **not** been installed as a current corpus.
- `vocabulary.json`: 74,785 B; SHA-256 `8d55fee4366acd2c175cf55bfd769cebc904770f5227a6fd50a613723dc85787`; VersionId `ybemOCwSNfRJHczDRzC8PEY5exfC4IDB`. Its exact-version bytes match the index binding. `groups.bin`, `provenance.json` and `packing-timing.json` also exist in the directory; existence alone is not qualification.
- Shards are referenced relative to the historical **parallel workspace root**, for example `runs/flywheel-20260909/packing-qualification-v1/store/shards/82e5f6db3d09d1a56611bac72ed15adda85569eec3da3755163d9f940a2d41c6`. Each directory contains `shard.json`, `records.bin`, and `labels.jsonl`. For that example, `shard.json` has raw SHA `fbc8b3cb4582485ef071c78ca090bfee2bda2e892218827ef78095bb4e6639ac`; the separate `records_sha256` and `labels_sha256` fields bind their respective files. The parent `shard.sha256` binds `shard.json`, whose own SHA field is blank. Source `teaching_store.rs:100-125` defines this exact contract. A group/directory key is not any of those payload hashes. All three fixed-first-shard exact-version hashes and the manifest-to-parent metadata comparison passed.

The historical authority manifest is:

`s3://veox-jopedime-learning-use2/staging/lineage-b/accepted60000-restored/parallel/runs/xbabe2-native-20260914-60000/authority/authority.json`

It is 501 B, schema `parallel.fit_authority.v6`, SHA-256 `f75060912598c8e223cecf65f9f161f02bce20a09a54c7b4b8e7a6f73c6ab0d9`, VersionId `yXyQaad53jcbE0BxB.G0TMjHwMHhJex9`. The directory also contains `registry.sqlite`, `dispatch.jsonl`, original objects and custody files. Their complete committed-terminal, runtime/evaluator and witness closure has not been verified.

Its original witness still exists on xbabe2 at `/home/ubuntu/jain-split/jope-dime/parallel/runs/native-fit-qualification-20260914/authority-witness-v5`: 178 B, SHA-256 `d011cac3c1e9bb74ec3778f2e8e21e87f3914ba961d61276273f778336ed7dae`, observed sequence 95. It matches the manifest's authority instance `e7610d3b24763cc94de40d73ae96c1425e2e9de9112d91188329ed278a1c2ed4`. A private unchanged byte copy was preserved. Matching instance does not prove terminal membership, freshness, independent high-water custody, current policy approval or a GPU grant. Do not move, refresh, rebuild or install authority/witness state from these notes.

The historical scientific decision is at the same `xbabe2-native-20260914-60000/` prefix under `scientific-evidence/670cf4aeee66192ff4bfcecb734578619e07e3631575b43df79d40f1bdc20d6e-initial-accepted-decision.json`. It explicitly states `training_dispatch_authorized: false` and `requires_separate_signed_run_approval: true`. The partition freeze also records original sealed partitions and further production gates. Historical accepted-parent status is not a live run approval.

## Other stores and misleading references

`JOPE_DATASETS_URL` defaults to `s3://veox-jopedime-datasets-use2`: raw datasets/catalogs, not admitted teaching labels. `JOPE_LEARNING_URL`/`--store` controls the learning mirror; it has no default. The `learning-use2/staging/` copies are preserved source data, not canonical `blobs/`, HWM or score-ledger promotion. `JOPE_MODELS_URL` points to model bundles, which do not establish corpus admission.

The 2026-09-23 bucket counts in `AGENTS.md` and `docs/learning.md` are historical. Current metadata includes experiment checkpoints, plans, schema registries and newer preservation copies; this audit does not replace the old dated inventory with a claim of exhaustive payload recovery.

Git commit `f5b013648f8e17d997950e9b676d88711351d1dd` and current `deploy/dl-flywheel/docs/native-runtime-pins-20261009.json` point to `/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/flywheel_native_full_20261009/spec.json`, SHA `05dc9558e36461068832b449cedff46123376e3567b9c011e7d74f9b8a8018ee`. That exact local file is missing. The source pin describes a historical bounded research optimizer, explicitly not production admission. Do not fabricate a replacement snapshot from the counts.

Git history's `docs/legacy/root-docs/legacy/parallel/ADMISSION_COMPLETION_INTEGRATION_20260910.md` is readable at `69bd961fbcb4e6ed969bc2d17dd144bf0f846c1a^`. It describes strict admission and remaining gates, explicitly calls itself source implementation rather than deployment/admission proof, and is useful recovery context. Reachable-ref searches did not locate an approved JSON policy or original clean-completion package. Archive commits `9d43f4e37c44a1d99bb2aeeb7b60a7ccbfeadc00` and `0964c4c730da4d4e90db57497ad0857b38d15ccc` were inspected by path; their subjects alone do not prove retained data closure. Source-only tests and synthetic fixtures are not production packages. The original Claude audit's phrase “owner or Root” is not an issuer rule: Root here is an assistant name and has no signing authority.

## Where recovered and future artifacts should live

Keep this Git registry and this runbook as the durable front door. For every actual artifact, record exact S3 URI, VersionId, raw length, raw SHA-256, schema, label policy, roles, source/evaluator/runtime identity, custodian, verification command, receipt and last verification time. Use null plus a concrete blocker for anything unknown. Never put raw corpus shards, model weights, secrets, credentials or authority private keys in ordinary Git.

For a **verified corpus recovery**, use a private cache under `/home/ubuntu/jope-evidence/corpus-cache/sha256/<original-manifest-sha256>/` on xbabe2, or `/mnt/xbabe3-external/jope-evidence/corpus-cache/sha256/<original-manifest-sha256>/` on xbabe3. These are proposed conventions, not claims that a complete corpus is already present. Record the actual cache path when populated. Check every ancestor with `lstat`, refuse overwrite and never replace or remove a symlink. Preserve relative paths and original bytes; do not rewrite an original index to hide missing shards. Signed/bound authority roots must retain their own verified semantics and cannot be relocated by a corpus-cache convention.

Keep bulk evidence in private content-addressed S3 objects and a verified second host copy, with compact Git cards, metric provenance, context, hashes and exact-version locators. Custom tar/zstd indexes must identify themselves as custom; they are not native Parquet manifests. Preserve incomplete, timed-out and failed runs as well as successful ones.

## Worker-ready next steps and stop conditions

Objective: recover and verify the original historical package for study, then obtain a genuinely current native-admitted package and finite run approval before GPU learning. Non-goals: relabel qualification data as production, rewrite historical evidence, open sealed partitions for fitting, create more model workers/checkouts, or change serving models.

Read first: `AGENTS.md`; this registry and runbook; `crates/jope-engine/src/main.rs` (`BuildCorpus`); `src/teaching_store.rs` (`publish_admitted`, `read_index`); `src/label_admission.rs` (`approved_authority`, `validate_completion`); `src/eval_registry/native_scq/policy.rs`; `src/task_contract.rs`; then the exact-version discovery receipts. Source paths after `src/` above are under `crates/jope-engine/`.

1. Recovery owner: use the 7,877 already located shard directories and their 23,631 members. Complete exact-version payload hash verification for every shard manifest, record and label file, plus original observation/source closure; all directory members and expected record lengths are already located. Restore only unchanged original bytes into the private recovery cache, preserving the relative directory layout. Native loader checks, roles and semantic dependencies must still pass before calling this a verified corpus. Preserve failures and explicit unknown counts. Seek original alternate copies only if a payload fails. Do not open or fit sealed partition rows during recovery validation.
2. Admission custodian: locate the **original** `parallel.label_admission.v2` policy and original `parallel.clean_completion.v3`/native completion, result/model, registry, consumed-token and independent-witness evidence for eligible observations. Current source checks evaluator, runtime and numerical-contract hashes and authority instance before labels. The historical `reference_v1` index cannot supply this by field changes. If the required original evidence is unavailable, record the precise gap and plan newly authorized native measurements; do not reconstruct it.
3. Integration owner: separately fix the fail-closed public GPU bridge through the reviewed design, positive native funding/grant verification and per-process opaque permit custody. The current unconditional `require_original_gpu_grant` hold, current hardware/RAM checks and original resource grants remain required gates. Source tests passing do not issue authority. Reuse the sole claimed writer checkout, one integration PR, eligible independent exact-head review, required local CI and receiving-main validation.
4. Research owner: only after data, policy, source, resource and current finite run-admission gates qualify, freeze a new experiment's hashes, roles, baselines, seeds, metrics, resource limits and absolute deadline before launch. Preserve optimizer exposure/checkpoints, original labels and failure/censor data, plus concise per-wave conclusions and exact reconstruction locators in Git.

The actual production packing CLI exists: `jope-parallel --root <verified-root> build-corpus --teaching-store <new-lineage-store> --label-policy strict_midrank_v4 --admission-manifest <original-approved-policy> --contexts <approved-contexts> --vocabulary <bound-vocabulary> --observations <original-observations> --output <new-output>`. `--observation-manifest` is the supported alternative to observation paths. This is a gate-dependent command template, not authorization to execute it against this historical qualification index. `--qualification` is not a production workaround. No unsupported environment variable is introduced here.

Allowed writes are new private recovery evidence and the sole publisher's claimed registry/runbook/research paths. Canonical dirty checkout, paper checkout, serving state, original corpus/authority/witness data, sealed partitions and other owners' jobs are outside this work's write scope. On any hash, schema, role, ownership, deadline, grant, disk or resource mismatch, stop that dependent lane, retain the exact receipt and continue independent admitted work. No resets, cleanup, symlink deletion, gate bypass or automatic chat rerun.

## Search limits and evidence

The audit inventoried current object names across nine known JopeDime product/preservation buckets, completed eight whole-bucket current-key inventories, examined 1,567,000 experiment keys before its finite scan limit, and separately completed all 645,326 current learning `staging/` keys. The rest of the very large experiment namespace, all historical S3 versions and opaque archive members are not exhaustive. Selected artifact payloads have exact-version GET/hash receipts. Sealed-bucket payloads were not opened. A prod custody catalog metadata pass completed 220/225 eligible GETs before explicit resource-policy denial; its partial per-GET versions were not persisted, so no exact-version claim is made for that pass and no alternate credentials were tried.

Receipts live in `/home/ubuntu/jope-evidence/fleet-rnd-20261018T1140Z/`: `ROOT_HISTORICAL_ACCEPTED60000_CORPUS_INDEX_LOCATOR.json`, `ROOT_ORIGINAL_NATIVE_WITNESS_BYTE_CUSTODY.json`, `ROOT_HISTORICAL_SHARD_DIRECTORY_CLOSURE_CENSUS.json` (authoritative directory-member census), `ROOT_SHARD_DIRECTORY_LOOKUP_CORRECTION.json`, `ROOT_HISTORICAL_CORPUS_SHARD_LOCATION_CENSUS.json` (superseded erroneous lookup retained), `ROOT_HISTORICAL_SHARD_ALTERNATE_LOCATOR_SEARCH.json`, `ROOT_GIT_DISCOVERY_COMMAND_RECEIPT.json`, `ROOT_GIT_ARCHIVE_CANDIDATE_PATH_AUDIT.json`, `GIT_TRAINING_ARTIFACT_LOCATIONS_AUDIT.json`, and `s3-training-discovery-20261018T1550Z/` inventories/fetch receipts. Never turn “not located in this scope” into “does not exist anywhere.”

## Compressed discovery context for future review

A private custom tar/zstd archive preserves 41 curated source/metadata/correction payloads (70,724,504 B) in 5,655,469 B, including the original historical corpus index, vocabulary, authority manifest and original witness byte copy. Full shard records/labels are referenced at their original S3 locations and are not copied into this archive. Bulk current-key inventories remain on xbabe2 with hashes in the archive index.

- URI: `s3://veox-jopedime-learning-use2/staging/fleet-rnd-20261018T1140Z/evidence/blobs/sha256/b7d9e669b7be6259aa5e63d7ec1a54b6341e2bc7d5667b283a9903d2c4b82175`
- VersionId: `H_UgBwgQ1oWvYPK6GI_W2Nk0duKr9R7q`
- Compressed SHA-256: `b7d9e669b7be6259aa5e63d7ec1a54b6341e2bc7d5667b283a9903d2c4b82175`
- Decoded tar SHA-256: `97d7c68bd2e226aba828ccce6a15c3995fcebac35574e215926135b592becd91`
- `DETAIL_INDEX.json` SHA-256: `7ea683678c41f7eb299f49fecd840f3213d2def76a2c53da71bd5929a425f428`

xbabe2 verified exact-version S3 GET bytes; the Mac controller independently decoded the archive and checked every indexed file hash and length. This is preserved context, not an installable authority or currently admitted production corpus.

## Track C S3 GPU parity and latency study archive (`track-c-gpr-cuda-sberbank-s3-20261019T0034Z`)

A private custom tar/zstd archive preserves all 124 member files (15,678,234 B across 35 repetitions per GPU host on `xbabe1`, `xbabe2`, and `xbabe3`, plus S3 provenance, `jope-parallel` audit/prepare manifests, and both original and final 35/35 receipts) in 223,684 B for wave `track-c-gpr-cuda-sberbank-s3-20261019T0034Z` on `fd1c0f065138bb68f6786aa77494890f05dbe5d2`.

- URI: `s3://veox-jopedime-learning-use2/staging/fleet-rnd-20261018T1140Z/evidence/blobs/sha256/b0f8af43c338ec6cfbb19be6f3ad0409e8d8ca6b0d8ac01745ff222e27e08d66`
- VersionId: `Lmaq0HCXEPM7hKWne0JIJqsYh3NEtWjI`
- Compressed SHA-256: `b0f8af43c338ec6cfbb19be6f3ad0409e8d8ca6b0d8ac01745ff222e27e08d66`
- Decoded tar SHA-256: `e62bba3e0cdb9d62ebe5b6352a10b8651dbc11f6d812920de1bd35be90206e0d`
- `index.json` SHA-256: `eeb063d97f0cffd306c6d08b834c21180bcbf46d5a8ebf8b8accbc68a0814704`
- Input S3 Parquet: `s3://veox-jopedime-datasets-use2/raw/BeyondArena/2ecfe882ccfb814fc27c4de10a64ceefd5d7655c/sberbank_housing_market_forecasting/019d9279-a91a-73ac-a221-72a6aab17df0/dataset.parquet` (`VersionId: 700u27HTJgxZwidGhVh3k98Q4bBEW35D`, `sha256: 95d4d4de028b996669a520a0585257a03987ce3a4face3d65a750434c87859fa`)
- Admission & scope limits: `HONEST_GPU_PARITY_LATENCY_BENCHMARK` (`NO_SELFPLAY_PROMOTION`). Source-build attestation and strict uncontended timing qualification remain unverified; this wave establishes numerical kernel envelope parity (`945/945` case rows `parity_pass`) and descriptive latency distributions only, not neural checkpoint or self-play superiority.
