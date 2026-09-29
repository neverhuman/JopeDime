# AWS Lambda experiments, 2026-09-28

On 2026-09-28 Codex did six pieces of AWS Lambda work, each under its own claim in the
coordination ledger. This page collects what the frozen records say about all six; their source
is in [`deploy/lambda/`](../../../deploy/lambda/README.md). Every figure below comes from the
frozen results files under `deploy/lambda/` or from Codex's closing ledger entries, for the
regression workers its stopped-state handoff of 2026-09-29T05:04:31Z, and no figure was
re-measured for this page. The costs are list-price estimates from AWS billing logs, not
invoices.

| Experiment | Claimed (UTC) | Frozen or closed (UTC) | Source now |
| --- | --- | --- | --- |
| Single-fit feasibility probe | 15:10 | 16:04 (with the cost study) | `deploy/lambda/csv/` (`rust/src/main.rs`, `versions/`) |
| Cost study | 15:36 | 16:04 | `deploy/lambda/csv/cost/`, `rust/src/batch.rs` |
| Corpus run `gp-corpus-20260928-v1` | 16:57 | 18:40:38, CLOSE published | `deploy/lambda/csv/gp/`, `rust/src/` |
| Planning discovery (read-only) | 19:18 | 20:03:26, CLOSE published | Python only: custody |
| Terminal monitor | 20:17 | 20:53:47 | `deploy/lambda/watch/` |
| Regression workers (v3) | 21:09 | 2026-09-29T05:03:06Z, closeout published | `deploy/lambda/regression/`, `deploy/lambda/regression-ops/` |

## Single-fit feasibility probe

- One statically linked `bootstrap` in a ZIP, with no container registry: 1,442,120 bytes zipped,
  2,904,912 bytes unpacked, no shared-library dependencies.
- It fetched one version-pinned training CSV (1,000 rows by 21 features) from S3 and fitted
  `feat-math.histgbm2` (500 trees, 31 leaves). Five native runs produced the same model and
  prediction hashes as the local build; RMSE and R² were replayed independently.
- 23 MB AWS-reported peak memory on every successful call. Fit seconds: 24.658 cold at 128 MB,
  12.555 cold at 256 MB, 5.999 cold at 512 MB.
- Six native calls: five fits and one retained failure (S3 signing without the payload-checksum
  header, fixed with `PayloadChecksumKind::XAmzSha256`). With the earlier 12-call Python and
  XGBoost container proof, 18 calls in all.
- Estimated Lambda cost: $0.00026418 for the native calls, $0.00145890 for all 18.

## Cost study

- 60 Lambda calls, 175 training events, zero training failures. All 175 model and prediction
  hashes matched the local baselines, on x86_64 and ARM64.
- ARM64 at 512 MB was the cheapest of the screened settings. Median cost per million training
  events on the same five-job cohort:

  | Jobs per call | Inputs | USD per million events |
  | --- | --- | ---: |
  | 1 | fresh S3 reads | 8.83 |
  | 2 | fresh S3 reads | 8.72 |
  | 3 | fresh S3 reads | 8.52 |
  | 5 | fresh S3 reads | 7.99 |
  | 5 | warm cached inputs | 7.37 |

- The cheapest single fit was $0.70 per million equivalent events (a small dataset with a cached
  input). The ARM ZIP was 1,294,149 bytes, sha256 `402ea725b37652816c96ec285b6c8ac758c889deaaf91767cca1d5f00df6a444`.
- 155 S3 GETs, 20 cache hits, no S3 PUT. Estimated cost for the whole study: $0.00167329.
- Both functions, their role, policy and log groups were deleted, and their absence read back.

## Corpus run

- Claimed with a $100 Lambda cap. The frozen job manifest (`train-jobs.json`, sha256
  `687f22c82397af9606455473e70e825c612c469ad917378c9cfb578d3590ee97`) admits 3,845 of 3,859
  canonical inputs; the extra wave's manifest (`extra-train-jobs.json`, sha256
  `3a838ea7f00dfa2afd05728ae69e86f156feb275d9129ed9162768f45af89341`) holds 142 distinct
  source families.
- Census of 4,001 sources: 3,987 eligible and examined, 3,981 complete bounded searches, six
  class-support refusals, 14 explicit exclusions, none eligible and unattempted.
- 2,005 durable raw results, each read back at its exact S3 version. CloudWatch: 2,012 executions,
  a Lambda list-price estimate of $0.030055 and a log-ingestion estimate of $0.026454. The ledger's
  worst-case reservations were $13.6535432 of the $100 cap.
- Cost sub-study: 208 invocations and 360 events; the cheapest small-cohort condition was
  128 MiB with batches of five, $0.0000028536 per event including S3 requests.
- Burst probe: 896 successful calls, a peak of 338 observed handlers and 231 completed jobs in
  one second.
- Archive under `s3://veox-jopedime-learning-use2/experiments/lambda-gp/v1/campaigns/gp-corpus-20260928-v1/`:
  - COMMIT sha256 `68ad89bab3a1bf10dcf8330d6e21bb985b27d826f68c256d900ba0680dbcbab1`, version
    `qFcvVRAdz6qjKumxv9z3Hf.7_1jijpWB`. A fresh download re-hashed 611 objects, 18,864 files and
    2,102,327,874 member bytes.
  - Source bundle sha256 `d359ff3e44f01eaaeeb664bdcd41e6924a6317f4049926fd84fd6ff95f616a0a`, 133
    files. Its 37 files from this source directory equal the imported ones.
  - CLOSE sha256 `a9d9bbd77da43cc2af9e4f36d46bb677001db4c52419be6b473e410e8f4953ff`, version
    `BzdbbUeVDiYFGTajIMWuwp_PvI_9SASj`. Independent live reads confirmed all four functions, their
    log groups, the runtime role and the archive-writer role absent.
- The results are experimental engineering evidence, not qualified teaching data.
- The qualified v2 corpus binary (Codex's clarification, 2026-09-29T05:04:31Z) is
  `bootstrap-gp-extra`, built from `deploy/lambda/csv/rust/src/gp_v2.rs`. It embeds
  `deploy/lambda/csv/gp/corpus/extra-train-jobs.json` through `JOPE_GP_MANIFEST_PATH` (manifest
  sha256 `3a838ea7f00dfa2afd05728ae69e86f156feb275d9129ed9162768f45af89341`). `gp.rs` is the
  preserved v1 compatibility runtime, and `gp_diagnose.rs` is diagnostic. The regression worker
  is a separate binary, `regression-bootstrap`, and does not use that manifest variable.

## Planning discovery

- A read-only S3 scan and a worker-ready plan for a longer run. No Lambda was deployed or
  invoked, and no budget was reset.
- Census: 7,581 declared views, 44,636 distinct native programs and 3,690 history rows.
- Published under `s3://veox-jopedime-learning-use2/experiments/` (key prefix
  `gp-evolution/v1/discoveries/scan-20260928T1918Z/`):
  - COMMIT sha256 `feb6e38cd829227d93b12dfbf898f71ccfeb5071128d18f8493d0702b9ea9039`;
  - archive sha256 `040501251ed410691037f9d7dc25fce8be78e6ba708ac99e73d0ec64d6f8ed62`,
    345,639,144 compressed bytes, 38,762 members, 1,484,792,947 bytes uncompressed;
  - CLOSE sha256 `83a85ba85014412bd4c7fbf5bf268f2133e1376f7000fb8a3627d6ea9f9d83ea`, version
    `d.8oUDdDEIMlJs2y7afVxVv55hoRVXSG`.
- Its tools are Python, so all 14 files are in custody and none is tracked.

## Terminal monitor

- A Ratatui and crossterm dashboard over local snapshot files. It makes no AWS calls.
- 14 Rust tests pass here (12 Python tests of its feed adapter are in custody). Codex's frozen
  proof replayed all 896 historical burst calls (10,752 fits) through the installed launcher.
- 1,000 frames of a synthetic 7,581-view fixture at 954.6 µs per frame, 16,384 KiB peak RSS.
  Release binary 891,808 bytes, sha256 `8c0b860a476cadb9230afc44b388f5c5b72a09a54df42b49bfed307e27ea7e01`.
- Source archive sha256 `d2fda4e7d582a633d4b364ceb96801112e89bbf25139b2328a57826dd9115c86`
  (17 members, each equal to the imported files and to the frozen tree); delivery manifest sha256
  `9ce88f6b61d6f65457ed0cf3dca19a5360681169a110d1efa88e7b463003b314`.

## Regression workers (v3)

- A Rust Lambda worker (`regression-bootstrap`) fitted candidate programs on three inner folds of
  each eligible regression view, in rounds, from a Python controller. An offline Rust tool
  (`regression-assess`) replays a stored fold model and measures how much its validation error grows
  when one feature column is shuffled, with no new fit. The run's fixed deadline was 2026-09-29T05:00Z.
- Terminal state: published. The closeout supervisor wrote `closeout-complete.json` at
  2026-09-29T05:03:06Z (sha256 `9ed8cdf377536c56867467f9f228cbd9072357a8fdb3ed15320fd14cf63a5002`).
  No run process remains, and the reserved concurrency of the four owned functions read back
  zero.
- Terminal S3 authority:
  `s3://veox-jopedime-learning-use2/experiments/regression-evolution/v1/campaigns/regression-20260928-v3/COMMIT.json`,
  VersionId `u6VROvAgGDHlJE220X6Nl._UutWYKA_o`, sha256
  `69643bbf38d9b3becf14df8e52ad100f57651c22a6897ac331f817c2ae1657a3`. The manifest has sha256
  `4a022fccc11f9c7c53bec7966b07a837bc571f63a23d667ca0ed9a0caa9205f4` (VersionId
  `0UxtrH95iVNNfR3CIiAX2hy4pM3W2f5o`, 112,341,749 bytes), and the restore proof sha256
  `42ded421b44e91fd8c28ff6c632ade34779775ad98b43fba5550e086759a650a` (VersionId
  `5LF1jUSqdIcjHPevOEJ3Kr9KlQUhJ6pA`).
- Coverage: 3,533 source views, of which 3,502 were eligible and 31 excluded; 3,476 views have a
  committed round. Zero views reached round 100: the summary records
  `all_views_reached100=false`, with `outer_labels_generated=false` and
  `canonical_teacher_promotion=false`.
- Accounting: 55,109 admitted attempts and 54,852 verified invocation archives (the query closure
  proves 54,852 accepted and 54,852 indexed, as equal sets); 54,908 AWS REPORT records were
  observed; 58 attempts remain unknown and reserved. The observed Lambda list-price estimate is
  about $5.96 ($5.96248166068485). The conservative reservation, earlier runs included, is $63.37
  ($63.365159459), under the $1,000 controlled-Lambda ceiling. S3 and logging costs are separate
  and not counted as zero.
- Accounted experiment storage: 105.8 GB (105,833,397,478 bytes).
- Import: the sources were frozen at 2026-09-29T00:54:49Z. A read-only freeze proof
  (`FREEZE-PROOF.json`, sha256 `c4a740be1544e7a92758c32fbe9c04c74f244400cf9d382b2379d223c879e63b`)
  found no process using them and two identical sha256 passes 620 s apart, equal to the
  pre-freeze snapshot. #125 imported them from a custody copy of that freeze.
- The results are experimental engineering evidence: no outer labels were generated and nothing
  was promoted to teaching data.

## Custody and what stays out of git

- **Import record:** [`deploy/lambda/SNAPSHOT.json`](../../../deploy/lambda/SNAPSHOT.json) lists
  all 148 files of the five bodies (csv, planning, regression, regression-ops and monitor): 71
  tracked, 1 gzipped, 1 renamed, 73 in custody and 2 legacy schemas that equal the product
  schemas, plus 52 excluded entries (the `.env` files by existence only, or by existence and
  sha256 for the regression bodies, build output, bytecode caches and two links to bulk
  evidence).
- **Python:** 73 files in five deterministic tarballs on xbabe2 under
  `~/jope-evidence/lambda-20260929/python/`: the #124 three with `SHA256SUMS` and `FILES.sha256`,
  and the two regression ones with `SHA256SUMS-regression` and `FILES-regression.sha256`.
  [`docs/archive/lambda-python-manifest.json`](../../archive/lambda-python-manifest.json) records
  each file, its tarball and its anchor. A Rust port is a follow-up.
- **Evidence:** receipts, logs, packages, results and restored mirrors stay in the runs' scratch
  custody on xbabe2 (`~/scratch/jopedime-lambda-{csv,cost,gp,watch}-20260928/`,
  `~/scratch/jopedime-gp-evolution-100-20260928/` and
  `~/scratch/jopedime-regression-lambda-v3-20260928/`). Codex's local deliverables (reports,
  dashboards and tables under `outputs/`) are on the owner's workstation, per its 16:04Z entry.
  The regression freeze proof and its custody copy are in `~/jope-evidence/closeout-20260929/`
  (`20-v3-freeze-proof/` and `21-v3-frozen/`), mirrored under
  `/mnt/jope-archive/jopedime-closeout-20260929/`.
- **Regression workers (v3):** the owner decided on 2026-09-29 at about 00:15Z that v0.2.0 waits
  for their freeze, and at about 04:20Z (O1) that #124 merges first and #125 imports them;
  v0.2.0 is tagged from #125's merge commit.
