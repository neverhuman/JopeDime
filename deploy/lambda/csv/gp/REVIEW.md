# Independent review: Lambda GP campaign

Reviewer role: source/correctness review only; no product source changes, AWS mutations, dataset reads or canonical-state writes. Source repository examined: main 4d9a27e64d31c57b6573a84e35f22c69ebb2b1b6. This is an interim review; runtime and campaign controller remain under active development. It is not launch approval for a later unreviewed source.

## Passed at initial source

control.py SHA256 0725e06a8f7f0e3111ce8a812de90e82e62545fcdde8ff875c0ef6cf9fbe96d5:
- Durable SQLite BEGIN IMMEDIATE serializes concurrent spend admission.
- Reservations survive process reconstruction; success does not refund worst-case liability.
- Expired deadline, STOP, unknown function, immutable manifest changes and repeated completion are rejected.
- Required mode-0600 config parser rejects keys, unknown fields, duplicates and executable syntax.
- Six tests independently executed, six passed, zero failures/errors. Temporary test files were confined to review custody.

Receipt: /home/ubuntu/scratch/jopedime-lambda-gp-20260928/review/independent-initial-tests.json.

The GP source inspected uses actual task-bound fit_table and deterministic existing mutation/crossover operators, not the evolve default ridge-surrogate ranking. The top-64 feature variance calculation reads fitting rows only and stores original selected column indices. Winner model is fixed before audit scoring; encoded audit labels do not enter candidate fit or selection. Binary/multiclass are explicitly experimental and cannot be claimed as the product's qualified parallel/teaching path.

## Findings sent to owners

1. Storage receipt collision, reproduced. store.py SHA256 083db2a465d0a266509db8415c0d039cbf84fc38eeae5c39f21f2548c86790e9 keys receipts only by payload SHA. Identical bytes written under two distinct object keys cause the second write to fail with "receipt identity changed". Receipt identity must include bucket+key, not just bytes. Fix and rerun independent reproducer.

2. Metadata/protection mismatch. Initial GP Job drops inventory catalog_task, sealed, historical_partitions and family identity; its default partition is empty. Initial sealed check also occurs only after GET. Runtime must preserve protection metadata and refuse sealed entries before download, with frozen manifest already excluding them.

3. Failure request accounting. Initial runtime records GET count only after a successful training result. Parsing/fit failures after successful retrieval still incur requests and bytes. Record attempted/completed GETs for every path; unknown transport byte count remains unknown.

4. Batch timeout containment. Initial invoke has no remaining-Lambda-time check between jobs. A late job can discard successful earlier siblings on handler timeout. Bound batch packing and refuse to start later jobs without an adequate remaining deadline. A single candidate still needs a hard function limit and honest timeout outcome.

5. Runtime metrics/projection tests. Constant-target R2 must be marked undefined rather than inherited feat_math zero. Initial projection_uses_fit_only test only asserts selected width; replace with actual audit/selection feature-poison invariance under unchanged membership.

6. Budget/config binding. Intent memory and timeout must match exact read-back published Lambda version/configuration. A ledger based on caller-declared resources alone cannot bound actual charge. Root plans exact version/CodeSha256/config verification.

7. Existing AWS control objects. Initial budget setup accepts any existing action and existing role/policy. Readback must verify expected trust/action/deny policy and scoped target, not just existence. Root plans explicit validation.

8. Restore anchor. Verify total restored byte count as well as per-object hashes and count; bind the expected COMMIT hash from an external/local campaign anchor to detect replacement of an internally consistent closure.

## Scope conditions

The all-catalog sweep is an experimental restricted-grammar screen. It is not neural proposal execution, product teacher admission, an unrestricted GP search or a serving promotion. Current task_contract::require_task permits only regression in the qualified product path. Independently tested advanced_v3 classification is a separate experimental protocol.

Historical family information is incomplete. Unknown-family datasets may be measured diagnostically under this explicit campaign but must remain excluded from canonical learning. Known sealed families are not read or trained. Historical normalized CSVs, unverified time-series chronology and row-hash rather than entity-group splits all limit generalization claims.

A $100 campaign admission cap and a $100 monthly AWS budget are different controls. AWS Budgets is delayed; automatic dispatcher denial does not cap unrelated account functions or principals. Report that limitation plainly.

No final launch or completeness verdict until final runtime/controller/source hashes, boundary tests, pilot raw receipts, S3 immutable-write/readback/restore evidence and full catalog reconciliation are inspected.

## Follow-up verification (2026-09-28)

Storage findings 1 and 8 are resolved at store.py SHA256 be0205d50678883db03abd6eb1c8c0467bc421e6626fa4885ec6b452973b473f. Thirteen tests independently pass, including the original same-bytes/different-key reproducer, interruption/resume, immutable collision, corruption, external commit anchor and aggregate byte count. Receipt: review/independent-storage-fixed-tests.json.

Runtime findings 2-5 are resolved at gp.rs SHA256 3ef647fe8177369170f79cf6977382682a8c43dd59e28da3d4296a33913282fb. Seven tests independently pass, zero skipped/filtered, including raw audit-feature/target poisoning under fixed dataset-ID membership, three task model/selection invariance, sealed metadata refusal, invalid CSV/hash refusal and bounded grammar. Test executable SHA256 b3c2683758fdb2b4f8cd8dee93ba1014800c166a604321414ecdfaa1b4683dda. Receipt: review/independent-runtime-tests.json. Runtime removes rank/blom fitted-row arrays, caps model/result bytes and checks remaining handler time before another batch job. Per-candidate timeout remains a soft search gate; the Lambda timeout is the hard outer bound.

Independent catalog reconciliation passes: 3859 unique catalog entries = 3845 runnable jobs + 14 explicit exclusions (7 known sealed meta-test, 7 zero-feature). Runnable task counts: 2597 regression, 843 binary, 322 mapped single-target multiclass, 83 CSV-order timeseries. Of runnable entries, 3784 have unknown historical family authority; all remain diagnostic and ineligible for canonical learning. Compiled train-jobs manifest SHA256 687f22c82397af9606455473e70e825c612c469ad917378c9cfb578d3590ee97. Receipt: review/independent-inventory-check.json. Review read metadata only, no dataset payload bodies.

No remaining runtime/storage source blocker for this bounded experimental scope at those identities. Controller/function binding and actual pilot/publication/recovery evidence still require review. Later source changes require fresh checks.

Reporting must distinguish trained partial screens with no mutation/crossover from actual evolutionary events, because soft-deadline stopping can occur during initial candidates. Top-level task_original initially uses effective task; original catalog task and mapping survive in nested dataset metadata. Use catalog_task for historical identity until that label is corrected.

## Superseding native and controller review

The earlier seven/eight-test runtime identities above are historical evidence, not deployable qualification. Actual CSV canaries exposed a serde_json f64 deserialization round-trip defect that synthetic fixtures missed. The runtime enabled serde_json float_roundtrip and added a 10,000 finite-IEEE754 bit-preservation test. Original failed canaries and pre-fix binaries remain in experiment custody.

Current V1 gp.rs SHA256 is 42775359f369ee359b7d1b8ec0a5015151c92e2e1aec59501298346c816f9a96. Nine tests independently pass, zero skipped/filtered, using test executable SHA256 5ae9c2602408b1645c76a64f2f6da60fcd2b7711932323b6a06ed22c8baaefa1. This supersedes the prior native gate. Receipt: review/independent-runtime-float-roundtrip-tests.json. Native x86 release SHA256 is 9a1100909789f3c9f3ec8ceb55f79c229765df04392898aec6e1382cd7f4baf4; ARM release SHA256 is 15e9b4c83234b57deee50097fccdb3235cb38ca0e3eb4b360a467b42141ff7e8.

Five actual native canaries (IDs 0, 11, 37, 543, 1417; including 138 MB / 20,532 columns) each complete 12 measured candidates, eight measured mutation/crossover descendants and exact serialized-model replay. An independent result-only audit verifies result file hashes, complete/disjoint row membership, program hashes, parent indices, winner selection by selection loss then executed DSL, IEEE754 audit-prediction hashes and independently recomputed RMSE/logloss/macro-F1. Maximum metric difference is 1.11e-16. Receipt: review/independent-canary-replay.json. These results establish bounded native execution; actual ARM Lambda equivalence remains a separate gate.

The runtime now records UTC microseconds plus monotonic relative nanoseconds as decimal strings for job phases and candidate spans. Failure paths retain stage stamps. It preserves original catalog_task and reports evolved_measured_candidates separately; an event with only bootstrap/random candidates is not reported as completed evolutionary search.

Two controller regressions were independently reproduced and corrected: ten benign batch deferrals previously created STOP, and an unreceipted result.json could suppress work on resume. The current offline harness executes all 12 benign partial batches without STOP and executes an unreceipted apparently-trained job again. Current tested campaign.py SHA256: 1b1cab99146c7b8ea1d5f491589a1f5628c2b5c45f91f72f9d474377a58fd826. Receipt: review/independent-controller-latest-tests.json. The fixture mocks every AWS session and stays inside review custody.

Published-version configuration readback pins the ARM ZIP CodeSha256, memory, timeout and architecture before admission. The reviewed deployment versions all use CodeSize 3,388,769 bytes, version 1, provided.al2023, ARM64 and 180 seconds. Reserved concurrency is 768/64/32/32 for 512/1024/2048/3072 MB, totaling 896; the coordinator retains per-tier semaphores and a campaign max-concurrency bound. Reserved concurrency does not itself bill when idle.

The actual budget readback is $100 USD monthly AWS Lambda COST with an automatic ACTUAL $80 IAM action against only JopeDimeLambdaDispatcher. A live test first receives Invoke DryRun 204, temporarily attaches the exact stop policy, observes explicit denial after propagation, then removes only the test attachment and restores dispatch. It executes zero training invocations; the AWS budget action remains STANDBY. Receipt: budget/live-dispatcher-cutoff-test.json. This demonstrates the scoped deny works; it does not make delayed AWS budget accounting instantaneous or cap other account principals.

The separate scoped S3 writer live proof succeeds on conditional create and idempotent readback and receives denial for unconditional overwrite and delete. Receipt: writer/live-proof.json. Archive completeness and a restore using the externally anchored final COMMIT still require the finished campaign.

## BeyondArena train-view curator review

The additional 142 raw BeyondArena sources are a separately pinned diagnostic view, not silently merged into the old compiled manifest. The isolated trusted curator verifies raw object identity, selects declared repeat-0/fold-0 training indices before feature/label conversion, caps to 4,096 source-family-hashed rows and 512 source-feature-name-hashed original features, and writes numeric train-only CSVs. Raw Arrow batches in curator memory may contain outer rows; the Lambda receives only derived official-train rows. This distinction must remain explicit.

Review found and fixed an initial boundary issue: row sampling originally depended on whole-Parquet SHA, so an outer-row edit could change which training rows were sampled. V2 uses stable source_family_id instead. Independent synthetic outer-feature/label poison leaves encoded training arrays, encoding recipes and label maps unchanged. Path-containment tests reject absolute and relative escapes. Stateless per-label SHA52 codes are exactly representable f64 values, checked for collisions; the GP then fits its vocabulary only on its own fitting rows. Feature hashing is stateless and target-independent. Numeric missing indicators can make the derived feature count exceed 512 before the GP's fit-only top-64 projection.

Four curator self-tests independently pass at source 4b6c2bad23dd3f4c0da0e7d320d3edeae0f3add5eb9079ef78cd0661b50f34d8; boundary evidence is review/independent-curation-boundaries.json. One real period[Y-DEC] metadata failure required exactly one conversion change: table.to_pandas(ignore_metadata=True), still after row selection and followed by the explicit dtype recipe. Four self-tests pass again at final source 15c5460604cc2a2050db124116d21e225b2831843567db474d7f16f2994c215a. Receipt: review/independent-curation-period-fix-tests.json. Earlier loaded curator processes used the earlier source; preserve per-source execution provenance and do not relabel all artifacts as produced by the later source. The original failure and original source remain preserved.

These views are unsuitable as official grouped/time-series benchmark claims or automatic canonical teaching input. Their purpose is experimental task-bound GP feasibility, cost and memory measurement. The fixed encoder and sample/projection recipes must accompany each input.

## Pending final evidence

V2 runtime adds a compile-time manifest path, exact input-bucket/prefix allowance and a bootstrap baseline evaluated on audit only after champion selection. Its baseline predictions and hashes allow independent gain recomputation. Provisional tests with a V1 manifest do not qualify the final V2 input identities: require final manifest/version pins, final rebuilt source/binary hashes, tests and native/AWS canaries.

Before a full burst, inspect actual ARM Lambda canaries against native evidence. Before the final user report, reconcile every source to a completed evolutionary event or explicit disposition, distinguish retries from unique events, inspect billed-duration/memory receipts and campaign reservations, and verify the final immutable S3 closure with an externally anchored COMMIT restore. No canonical teaching admission or serving promotion follows automatically from this campaign.

## Actual ARM Lambda canary gate

All five deployed V1 ARM canaries (0, 11, 37, 543, 1417) complete all 12 measured candidates and eight evolutionary descendants. Champion fitted models and their hashes, audit predictions and their hashes, audit/selection metrics, fold/projection/task metadata and all candidate program/parent lineages exactly match the native results. Every runtime phase/candidate and coordinator queue/admission/request/return timestamp chain is ordered; monotonic queue/tier differences recompute exactly. One nonwinning sigmoid multiclass candidate on wide input 1417 has a different fitted-model hash across platforms while its metrics match exactly. Therefore all-champion replay/parity is established; bytewise identity of every discarded intermediate model is not claimed.

Receipt: review/independent-aws-canary-comparison.json, binding exact intent/result/receipt/native file hashes per attempt. Estimated Lambda cost of these five calls totals $0.0001361673425 before S3 requests. Four small inputs use 63-64 MB peak at 512 MB allocation and 672-964 ms billed; the 138 MB wide input uses 199 MB peak at 3,072 MB allocation and 2,856 ms billed. This is a live feasibility gate, not an optimized cost frontier or corpus-wide quality claim. No V1 burst blocker remains from independent review.

## Final V2 native gate

Final V2 gp_v2.rs source SHA256 3895c3d2d54524ec3ecff43e737f27e7a1f44ddad81aa27c00278f96821d05d1 independently passes all ten tests, zero ignored/filtered. Test executable SHA256 af410ed9456369754dca3175c2f7afafffa8646d6ac65ae26c4943832ac39d81. Receipt: review/independent-runtime-v2-tests.json.

The final compiled manifest SHA256 3a838ea7f00dfa2afd05728ae69e86f156feb275d9129ed9162768f45af89341 has exactly 142 distinct source families: 44 regression, 73 binary and 25 multiclass. Independent metadata reconciliation verifies job ordinals, exact S3 input namespace/hash/version pins, all curation/preprocessing receipt hashes, row-cap/official-train-count constraints and false canonical admission/outer-test availability flags. Provenance correctly binds 141 curations to source 4b6c2bad... and the single period-metadata retry to source 15c54606....

Final native binary SHA256 b7032dbf608acab246701b8c5f3a98792b4e2c60f1cd279dfdeb5f56233bd9d1 completes regression/binary/multiclass canaries 2, 4 and 20 with 12 measured candidates and eight evolved candidates each. An independent result-only audit verifies membership, selection, serialized replay, both champion/baseline IEEE754 prediction hashes, audit metric recomputation and baseline-minus-champion improvement. Largest floating recomputation difference is 1.42e-14. Audit loss gains are respectively 0, 0 and -0.004796152066355974; the search protocol does not guarantee improvement and these outcomes must be retained. Receipt: review/independent-v2-manifest-canaries.json. No final V2 native gate blocker remains; final ARM/AWS gate and publication closure remain outstanding.

## Coordinator burst failure and recovery gate

The first 896-worker corpus burst exhausted the coordinator's RLIMIT_NOFILE=1024. SQLite connection context exit commits/rolls back but does not close descriptors; concurrent pending connections and network work reached the process limit. Even STOP creation failed, so the original traceback is preserved and ordinary completed-canary evidence must not be mistaken for a successful corpus burst. The root retained all 511 admission reservations, including 506 pending liabilities, without refund or ledger reset; precise classification remains a reconciliation task.

Independent read-only CloudWatch collection across the four owned log groups observed seven START and seven REPORT records at snapshot time: five attributable canaries and two otherwise unattributed burst invocations. Their request IDs are 7f404763-e097-48e9-927a-c370f75f95a1 and 3b57b161-4ab1-436e-99bc-e4fef1689339, billed 741/850 ms at 512 MB with peaks 67/66 MB. Estimated Lambda charge for the two is $0.0000110067197; all seven reports total $0.0001471740622. No per-job attribution is fabricated, and log-delivery latency limits any absence claim. Raw read-only evidence and parsed receipts: review/cloudwatch-old-burst.json.

The corrected control.py SHA256 96db072d7409c3b6a42cafd8833c02f16d415a2ce2299c847dfc1bfe1271e39e uses a process-wide 16-slot SQLite semaphore and explicitly closes connections after rollback/commit. Eight tests independently pass, including 800 concurrent independent Ledger admissions and 800 success plus 800 exceptional connection lifetimes; process descriptors remain exactly four before/after. Test source SHA256 d85ba3cedf949393b68970ddd668c67bcdbfb33c690cfbee9bdc2aaee3126960. Receipt: review/independent-control-fd-tests.json. Controller raises the per-process soft descriptor limit up to 65,536 and rejects insufficient limits before work.

This failure motivates exact coordinator attempt-ID logging and an immutable S3 result capability archived before API return. Those runtime/protocol changes require a fresh pinned build and canary gate. Preserve original campaign liability/deadline and version histories; no new $100 budget ledger.

Analyzer review requires all-phase source completion distinct from phase-only throughput/cost, because successful canaries are intentionally skipped on corpus resume. Ledger-only reservations must remain visible even when failure prevented an intent directory from being created. Invocation peak memory remains an environment/batch-level measurement, not an uncensored independent per-job memory label.

## Full coordinator pressure proof

The final close hardening adds nested cleanup so connection closure still runs if rollback raises. At control.py SHA256 c5054a4ccca0be53107e89ff6865c18456b61cc06c428f61f0200250d922a353, a separate offline test creates 896 worker threads and forces a barrier while all 896 live socketpairs remain open: 1,792 socket descriptors and 1,796 total descriptors. Every worker then reserves and finishes one attempt through an isolated fake $100 ledger. All 896 succeed, zero remain pending, and process descriptors return exactly 4 to 4 after cleanup. The test raises RLIMIT_NOFILE from 1,024 to 65,536 and restores it on exit; elapsed time 23.21 seconds. No AWS calls or real campaign ledger access occur. Script: review/socket-stress.py. Receipt: review/independent-socket-stress.json.

Analyzer source SHA256 9bfbaf9f2693cabfaf29bdfc9ed99a7c22416ba54910666e67912d60ad9a161e independently passes eight tests after adding all-phase coverage and read-only ledger joins. On actual failure evidence it reports five canary calls/successes plus 506 corpus admissions, six request-start records, 500 admissions without request start, $2.1126512 unreleased corpus reservations and zero corpus successes; all-phase census preserves the five canary successes. Integrity issues: none. Receipts: review/independent-analyzer-tests.json and review/independent-analyzer-current-ledger.json.

## Durable protocol and deployed gates

The durable runtime sources are V1 gp.rs SHA256 7e5ee064d0eb051ac991dcf007c5dbd79b93cab06e8813e05e1e0aa0fe310328 and V2 gp_v2.rs SHA256 5c9d614afada84702796b0e5839056a17d647bf3c4b731765776d1a1cf83f09d. V1 independently passes 11 tests and its five native canaries retain every prior learning output exactly. V2 independently passes 12 final tests (test executable SHA256 3d674c39ade6c02f4f4b51761802383cf111a0132b95be2026e3649f002060d2) and its three native canaries retain prior champion/baseline/fold/projection results. Receipts: review/independent-archive-runtime-tests.json, review/independent-archive-native-canaries.json, review/independent-archive-v2-final-native.json. The first intermediate V2 observation still saw an old ten-test executable and is explicitly marked unqualified; the later twelve-test receipt supersedes it.

The runtime requires a 32-lowercase-hex attempt ID and a presigned PUT capability for exactly that campaign raw-attempt key. It checks exact HTTPS bucket/region/path, bounded expiry, SigV4 fields, mandatory signed content-type/If-None-Match/encryption headers, disables redirects, bounds PUT to 15 seconds and 5.5 MB, and requires AES256 plus a non-null object-version acknowledgement. It archives the complete result before returning the API response. Archived body includes published function name/version and exact attempt/request ID; oversize fallback retains timestamps. The API separately returns the archive version/raw-byte hash and response-ready timestamp, avoiding a recursive hash. Failures redact capabilities and point recovery to deterministic keys. Lambda runtime role still has no general S3 grant.

An independent five-call V1 version-2 gate fetched every exact S3 object version again, verifying raw bytes/hash/size against both controller and API receipts, AES256, function/version/attempt/request identity and ordered PUT/response/readback timestamps. Champions and all program/parent/metric traces match native. Each invocation has all 29 expected CloudWatch events, including 12 candidate starts/ends and an archive hash linked to the same request/attempt. Receipt: review/independent-durable-aws-canaries.json.

The equivalent V2 version-3 gate passes exact archived-byte/binding and CloudWatch checks for all three tasks. Regression2 and multiclass20 match native exactly. Binary4 has cross-platform near-tie sensitivity: its initial bootstrap loss differs by 5.55e-17 from native, reversing the rank of two effectively tied candidates and changing later offspring/champion. Audit accuracy remains identical, logloss differs by 1.80e-7 and maximum predicted probability by 8.01e-6. Independent selection/metric/prediction-hash checks and within-artifact serialized replay pass. Do not claim cross-architecture bit identity for this case. Receipt: review/independent-extra-durable-aws-canaries.json. A same-ARM repeat is a separate planned determinism check.

Durable controller source ca413c58... passed offline tests for wave manifests, archive corruption/version rejection, no unreceipted resume and benign partial continuation. Review found and corrected a new-wave bug that initially selected the extra manifest for canonical-durable. Readbacks now pin exact S3 VersionId and separately count the one archive PUT plus one readback GET. Receipt: review/independent-controller-durable-tests.json; the same suite reran at later source 3d56f989... alongside synchronized-start tests.

## Small-memory and synchronized-start controls

Small-memory deploy source ca93dbb228ce92d2c6c0558323be56c582cb22e529d9ff1bc704ac280ba4a57c passes independent fake-AWS success and injected publish-failure restoration tests. It publishes immutable 128/256 MB versions of the frozen canonical package on the existing owned function and restores its latest configuration to 512 MB plus the prior extra package in finally. Original functions/cap/deadline and shared ledger remain unchanged. Per-180-second-call reservations are $0.00105145/$0.0020927/$0.0041752 for 128/256/512 MB. Proof: review/independent-smallmem-tests.json. The small-memory versions share one function's reserved concurrency; tier semaphores are not additive account capacity.

Synchronized-start controller source 3d56f989d432397cd3e02ed1c6f8a317d0de03d79add1bc6519780b7f671ac02 passes a 16-worker offline gate. Every fake invoke sees all 16 ready records and reservations. An injected third-presign failure aborts with zero invokes and zero request-start records, retaining four conservative liabilities. Insufficient workers or tier slots refuse before reservation. Proof: review/independent-barrier-tests.json. The earlier 896-socket pressure proof establishes coordinator FD capacity. Configured worker count is not observed Lambda concurrency; post-gate request-record fsync and AWS launch timing can reduce overlap.

## Independent whole-corpus verifier

review/verify_corpus.py reads only saved results, manifests and controller receipts. It checks exact input/metadata bindings, all program hashes and parent indices, complete/disjoint fold membership and fold hash, fit-only projection indices, selected champion from frozen selection loss plus DSL tie-break, prediction IEEE754 hashes and independent audit metrics, plus raw archived byte hashes/version/function/attempt bindings for every trusted durable result. It cannot reconstruct typed BoundModel serialization bytes through generic Python JSON; native runtime replay and model-to-trace binding remain the appropriate proof for that hash.

The first completed canonical snapshot has 3,839/3,845 full twelve-fit/eight-offspring events, no partial-search successes and six explicit binary failures. The extra 142 sources were pending that snapshot. All 848 durable result archives then present pass integrity checks. Six failures (1847,2389,2495,2606,3176,3404) require label/fit diagnosis, not blind memory retries. The verifier retains failures and duplicates instead of shrinking the denominator. Results are saved as timestamped review/corpus-verification-*.json plus corpus-verification-latest.json and per-job JSONL.

All recomputed metrics pass scaled tolerance; largest scaled difference is 5.35e-14. Regression1660 has very poor audit R2 near -1.0194e47, so its absolute floating difference is large despite relative error near 4e-16. Preserve that outcome and avoid presenting mean corpus R2 as a useful quality result. Experimental multiclass macro-F1 averages classes present in targets, matching the current implementation; it is not canonical teacher qualification.

## Frozen final campaign gate (2026-09-28 18:11 UTC)

The final independent coverage check supersedes the earlier partial snapshots. Inventory contains 4,001 source entries: 3,987 eligible sources plus 14 explicit exclusions (seven historically sealed meta-test families and seven zero-feature inputs). All 3,987 eligible sources were examined. Exactly 3,981 have a durable, fully replayable experimental GP event with all twelve actual successful model fits, including eight mutation/crossover candidates. Six have diagnosed terminal class-support refusals. Zero eligible sources are unattempted or unresolved, and zero successful events have partial search. Canonical coverage is 3,839/3,845; the separately curated BeyondArena wave is 142/142. These are diagnostic screening events, not canonical teaching admissions or qualified outer-fold benchmark evaluations.

Across all phases there are 5,249 returned job results: 5,243 trained plus six refused. All 2,005 durable local archive bodies pass raw SHA256/size/version/function/attempt/input binding checks; the controller had already fetched every exact S3 version. Independent canary reads additionally checked actual S3 versions. No integrity error appears. Successful returned traces contain 62,916 measured candidates and 41,944 evolutionary candidates. Six absent failure traces remain explicitly null rather than falsely counted as zero work. Receipt: review/corpus-verification-20260928T180818Z.json, also corpus-verification-latest.json and corpus-verified-job-results.jsonl.

The six failures have only one or two minority examples in the entire 1,000-row training input. Four fixed fit partitions contain one minority row, violating the head's minimum of two examples per class; two partitions contain only one class. Independent diagnosis matches the frozen manifest/object/version and archived error for every case. Final CloudWatch now confirms 48 actual failed candidate fits in the four cases and zero candidate fits in the two preparation refusals, agreeing with the separate native reproduction. This is not a memory/timeout failure. Do not retry resource tiers blindly, borrow outer/audit rows, or choose a favorable split after outcomes. A future independently versioned train-only support policy needs larger official training input or an explicit unsupported disposition. Diagnosis SHA256: 99ee7260a6906cfa20be97e7bcd48167ef82a80540280561cfa57029974a6a6d.

The same-ARM repeat of extra binary job 4 matches champion, serialized model, predictions, baseline metrics and every candidate's learning trace exactly. Its earlier x86-to-ARM near-tie divergence remains an explicit portability limitation. Removing only baseline audit clock fields is necessary for a fair repeat comparison; no learned fields are excluded.

## Final observed concurrency, logs and Lambda spend

The frozen log export contains 2,012 unique REPORT records: all 2,010 known completed result calls plus exactly the two original unattributed calls from the descriptor incident. No known result lacks a REPORT. Independent billed-duration/memory arithmetic gives $0.030055233263425 in paid-rate Lambda execution/request charges, including $0.0000110067197 for the two unattributed calls. This is an estimate from measured AWS billing units, not a settled invoice; unrelated account usage and non-Lambda charges are outside it.

All 2,005 durable invocations have complete handler start/archive start/handler complete and job start/end logs. Structured events total 142,311, including 62,904 candidate starts and 62,904 candidate ends; these omit the old prelogging five canaries and include the 48 failed fits. Final evidence is review/independent-final-log-reconciliation.json. High-resolution timing remains auditable in the raw export; monotonic timestamps are never compared across processes.

The synchronized throughput probe completed 896 separate calls successfully in 24.557005 seconds from first handler start to last response-ready timestamp, about 36.49 completed calls per second. Maximum observed runtime handler overlap was 338. Configured 896 dispatch workers and reserved slots are not evidence of 896 simultaneous Lambda handlers. UTC overlap across environments is an observed estimate; each runtime's monotonic elapsed span is authoritative within that invocation.

The original $100 campaign reservation ledger remains intact, with no refunds or resets. All initial unknown and confirmed-not-dispatched reservations are retained conservatively. Budget enforcement is a campaign admission bound plus a delayed AWS Budget action scoped to the campaign dispatcher; it must never be described as an instantaneous hard cap on unrelated account-wide Lambda spend.

## Independent cost and batching result

A separate result-only cost verifier recomputed 24 memory/batch/replicate conditions from 208 unique receipts and checked all 360 events against their canonical learning fingerprints. Every event completed twelve successful fits including eight evolutionary candidates: 4,320 fits. No output-quality change, partial search or duplicated invocation charge was accepted. Evidence: review/independent-cost-results.json and review/final_cost_review.py.

The cheapest observed cohort was 128 MB with five datasets per invocation: $0.0000013735622233 Lambda-only per training event, or $0.0000028535622233 including input GET, one batch archive PUT and one batch verification GET. This corresponds to $1.37 or $2.85 per million respectively. The 128 MB batch-five peak reported memory was 66 MB and median call elapsed time about 3.74 seconds; 512 MB batch-five cost about $0.000002978674 per event and took about 1.42 seconds. Thus the smallest tier saved only about 4.2% versus 512 MB after request costs, while taking about 2.63 times as long. Batching five saved about 70.8% versus 128 MB singleton calls in this cohort.

These are measured estimates for fifteen small qualified inputs and two counterbalanced replicates under naturally occurring cold/warm conditions. They exclude data transfer, CloudWatch, storage, publication, host and curation costs. In particular the external coordinator's location is not assumed to be co-regional. They are not a universal corpus price, million-event invoice prediction or latency SLA. Keep the wide/large-input profile separate and use observed safe memory headroom plus conservative upper-bound runtime predictions before routing future jobs to 128 MB.

## Publication closure status

The final archive implementation has separately passed path traversal, object closure, external COMMIT anchor, raw-byte reconstruction, source hash, fixture-secret rejection and ledger stability checks. Final campaign publication and a fresh externally anchored restore are a separate remaining gate. Successful local staging alone is not proof that S3 closure or disk restoration completed. The parent must bind the final report to the published COMMIT SHA256 and fresh restore receipt before declaring this gate complete.
