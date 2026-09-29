# Lowest measured Rust Lambda training costs

The cheapest **individual measured fit** cost **$0.0000007000**: about **$0.70 per million equivalent events**. This was the small Chatfield dataset with an already cached input, ARM64, 512 MB, a 75 ms billed call, and 30 trees retained by the unchanged early-stopping rules. It is not a generic cost for every dataset or GP program.

For the complete five-dataset mixture, lowest repeated median was **$0.0000073734 per event**, about **$7.37 per million**, using **ARM64, 512 MB, batches of five and warm version-pinned input caching**. Every event still refit its model. Fitted models were never cached or reused.

Without input caching—every dataset read from S3 on every call—five-job batches cost **$7.99 per million events**. All estimates below include Lambda duration, its request charge and actual S3 GETs. They exclude logs, storage, output writes, coordinator charges, network egress, tax and discounts/free-tier effects. These are extrapolations from the measured calls, not million-event load tests or a final AWS invoice.

## Matched batching experiment

Every row completes the same five distinct dataset/fold/model jobs on ARM64 at 512 MB. Sizes 2 and 3 include a final smaller batch. Three repeated cohorts per row; medians and observed ranges, not confidence intervals.

| Jobs per call | Input access | Calls per five-job cohort | USD per training event | USD per million events | Observed range per million |
|---|---|---:|---:|---:|---:|
| 1 | Fresh S3 reads | 5 | $0.0000088320 | $8.83 | $8.75–$8.91 |
| 2 | Fresh S3 reads | 3 | $0.0000087187 | $8.72 | $8.70–$8.76 |
| 3 | Fresh S3 reads | 2 | $0.0000085214 | $8.52 | $8.46–$8.63 |
| 5 | Fresh S3 reads | 1 | $0.0000079934 | $7.99 | $7.96–$8.01 |
| 5 | Warm cached inputs | 1 | $0.0000073734 | $7.37 | $7.35–$7.39 |

Batching alone reduced matched-cohort cost **9.5%**. Warm caching reduced it another **7.8%**; combined reduction versus fresh-read ARM singletons was **16.5%**. Input caching matters only when the same immutable inputs reach a reused execution environment; Lambda does not guarantee that reuse. Initial cache population cost $8.12 per million-equivalent events for its measured five-job call and is retained in the total experiment cost.

## Architecture and memory screening

Same five-job batch, fresh S3 reads, one cold plus one warm sample per setting. These are screening observations, not stable tail estimates. ARM was faster and had a lower GB-second price for this workload. Minimum memory and minimum cost are different objectives; memory also controls CPU allocation.

| Architecture | Memory | Cold billed seconds | Warm billed seconds | Warm USD per million training events |
|---|---:|---:|---:|---:|
| x86_64 | 512 MB | 7.195 | 7.137 | $12.34 |
| x86_64 | 1024 MB | 3.670 | 3.493 | $12.08 |
| x86_64 | 1769 MB | 2.149 | 2.079 | $12.41 |
| x86_64 | 2048 MB | 2.151 | 2.091 | $14.38 |
| arm64 | 512 MB | 5.581 | 5.519 | $7.80 |
| arm64 | 1024 MB | 2.980 | 2.849 | $8.04 |
| arm64 | 1769 MB | 1.948 | 1.914 | $9.26 |
| arm64 | 2048 MB | 1.755 | 1.677 | $9.38 |

Screening selected ARM 512 MB at $7.80/million. Repeated fresh-read five-job cohorts later measured $7.99/million median; report that repeated result rather than presenting the best isolated sample as a guaranteed price. ARM 128/256 MB, CPU-specific code generation, different algorithms, and parallel fits within an invocation were not tested in this campaign. No claim of a global minimum is made. Additional low-memory ARM and held-out workload tests are the next evidence needed before production sizing.

## Dataset-level costs explain the range

All jobs used the same parameter cap: one bag, 500 rounds, 31 leaves, existing native defaults and early stopping. Actual trees varied by dataset and stayed identical across every comparison. Fresh singleton costs are medians of three calls; cached singleton costs are single samples. Cached data saves I/O, never training work.

| Dataset | Train-file rows × features | Retained trees | Fresh-read singleton USD/million | Cached singleton USD/million |
|---|---:|---:|---:|---:|
| Parkinsons Telemonitoring | 1000 × 21 | 500 | $33.85 | $30.05 |
| Voting Intentions in the 1988 Chilean Plebiscite | 1000 × 7 | 31 | $2.13 | $1.42 |
| 695_chatfield_4 | 195 × 12 | 30 | $1.40 | $0.70 |
| 229_pwLinear | 166 × 10 | 196 | $1.94 | $1.17 |
| barium | 109 × 30 | 393 | $4.87 | $3.97 |

The Parkinsons job dominated cohort CPU. Replacing it with easier datasets would reduce the average but would not be a batching optimization; this experiment kept the five-job mixture fixed. These are `feat-math.histgbm2` fits, not XGBoost and not arbitrary GP-program or GP-search qualification. The earlier Python/XGBoost test used different defaults/arithmetic and must not be treated as a language-speed comparison.

## Package, correctness and history

- **60 Lambda calls, 175 successful training events, zero model-training failures.** One controller reporting failure after a successful first batch was preserved; corrected reporting did not reset the budget.
- **175/175 exact model and prediction hashes matched local Rust baselines**, including across architectures. Maximum prediction difference: zero. Independent RMSE/R² replay and identical cohort coverage passed.
- Three native tests passed: fold separation/completeness, corrupt-input refusal, and validation-label poisoning. Original env-file refusal tests remain part of the harness; configuration is parsed as data, never sourced/evaluated.
- **23 MB maximum AWS-reported memory** across all calls. Process HWM is labeled as environment-lifetime memory, not falsely treated as an independent per-job peak. Broad GP work still needs isolated child rusage plus runtime overhead.
- ARM ZIP: **1,294,149 bytes (1.234 MiB)**; executable 2,320,024 bytes. No shared-library dependencies, no ECR, no Python in Lambda. Both uploaded ZIP hashes matched AWS CodeSha256; archives contain only `bootstrap`.
- Actual observed S3 GETs: **155**; cache hits: **20**. Estimated total for this entire campaign: **$0.00167329** for Lambda + GETs, before the other charges excluded above.

History is saved as two joined append-only exports: [`history-invocations.jsonl`](results/history-invocations.jsonl) (60 rows) and [`history-training.jsonl`](results/history-training.jsonl) (175 rows). Join by request ID. Cost lives at invocation level so a five-job batch is not counted five times. Training history records input/version/fold/model/parameter identities, shapes, phase times, cache status and memory. Wall-clock metadata is explicitly the host response-file timestamp, not an invented AWS execution timestamp.

This is a useful seed for sizing history, not enough data to fit a reliable global predictor. Use algorithm/code/architecture-specific conservative envelopes initially; add held-out dataset families and GP structures before learning upper-bound memory/runtime models. Choose the cheapest tier with runtime and memory headroom, then pack 2–5 jobs by predicted total duration and peak memory. Sequential batch peak is approximately the largest live fit plus cached-input/runtime overhead, not blindly the sum of all independent job peaks.

## S3 writing remains disabled

No S3 PUT was made or granted. Both temporary functions, their role/policy and log groups were deleted; AWS absence readbacks are in [`cleanup-readback.json`](results/cleanup-readback.json). Existing services were not changed. No background job, event trigger, or provisioned capacity remains.

When the implementation is vetted, a single immutable output object per batch can limit write cost:

| Future output layout | S3 PUT request cost per million training events |
|---|---:|
| One object per event | $5.00 |
| One object per batch of five | $1.00 |

Those amounts exclude storage, encryption, readbacks and model-serialization/upload CPU. Five separate model objects plus a manifest are six PUTs, not one. At these low training costs, output-object count is material. Bundle per-job identities/metrics/checksums and bounded model artifacts; use deterministic job/batch keys and conditional creation to handle retries without overwriting or double-admission. Keep initial destination limited to an experiment prefix and require checksum/version readback before canonical admission.

For a same-region design, S3-to-Lambda data transfer is free, but S3 operations and stored bytes still cost money. Returning full predictions to an external coordinator, as this audit does, can incur transfer charges; a vetted production path should retain only necessary result bytes and measure its complete I/O cost. [AWS Lambda pricing](https://aws.amazon.com/lambda/pricing/) describes these charges; region-specific GET/PUT/compute rate responses are preserved with the receipts.

## Code and next execution contract

Code lives on xbabe2 under `/home/ubuntu/JopeDime/.agent/experiments/lambda-csv/`, with batch runtime in `rust/src/batch.rs` and deployment/config in `cost/`. `cost/.env` is mode 0600, ignored, and references an existing AWS profile; account keys are not stored in Git, artifacts or the ZIP. Runtime credentials come from the temporary scoped Lambda execution role. The source remains ignored experiment staging, not a merged product change; tracked import into `deploy/lambda-csv/` follows existing lane ownership and independent review.

Read the [full worker-ready test and sizing plan](../TEST-AND-SIZING-PLAN.md) and [cost optimization protocol](COST-OPTIMIZATION-PROTOCOL.md) for exact files, commands, resource caps, source/package identities, validation, failure interpretation, S3-write vetting and handoff. This campaign's 60-call cap is exhausted; `cost/run.sh` refuses another run before cloud mutations. Future tests need a distinct frozen protocol/budget, not deletion of receipts to reset the counter.

Evidence: [`verification.json`](results/verification.json), [`verified-cohorts.json`](results/verified-cohorts.json), [`verified-summaries.json`](results/verified-summaries.json), [`source-repair-manifest.json`](results/source-repair-manifest.json), [`AWS receipts archive`](results/aws-receipts.tar.gz). Tested [ARM deployment ZIP](packages/arm64/bootstrap.zip) and [x86 deployment ZIP](packages/x86_64/bootstrap.zip) are retained for inspection; deploying either is a new compute action, not performed by opening the file.
