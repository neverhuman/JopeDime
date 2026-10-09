# JopeDime deep-learning evidence and white-paper protocol

Version 1, 2026-10-08. Owner: Codex experiment/dashboard lane, coordinated with Grok in `DL_TRAINING_DASHBOARD_HANDOFF.md`. This document defines what is measured, where the original evidence lives, and the claims that evidence supports. Status fields and immutable receipts take precedence over prose.

## Evergreen and S3 evidence update — 2026-10-09 00:40 UTC

The corrected 45-fit comparison is complete: equal-lineage fitness R² is 0.690235 for the DL ensemble and 0.784889 for LightGBM, with paired DL-minus-LGB 95% interval [-0.132907, -0.056273]. Selected validation loss reduction does not establish superiority over this comparator. Retain the full comparison method and development-corpus caveat.

All three paper archives now have independent S3 custody: 233 events, 131,532,800 archive bytes, six exact-VersionId/SHA-verified objects. Receipt: `DL_PAPER_S3_CUSTODY.json`, SHA256 `8b37c9a8f6d6ac32707942e32d8b88bc25ffbb5e5a67cf0121919490688ceb32`.

The first 100 variants now have all eight encoders in S3: 800 vectors and 1,200 total objects across the two publication stages, 33,202,223 bytes. Metadata was preserved; additional immutable completion receipts bind later blocks. These 100 variants remain one parent lineage. New vector availability does not prove GP labels or training admission.

Evergreen generation has produced a separate 128 variants from 128 parent lineages. They contain 125 train and three validation lineages in this first deterministic tranche; the full qualified registry contains 1,014 train and 54 validation. Seventy-eight first-tranche variants use 1% target-noise standard deviation and 50 retain target values. 3% and 5% recipes occur at later per-parent ordinals. All have fewer than 2,000 rows. Parent lineage, original source hashes, complete transform state, new target identity and research caveats are retained. Target noise is Gaussian standard deviation relative to designated inner-fit target scale, not a maximum per-observation change.

Two GP programs have been executed on a transformed dataset, independently emitted/compiled as Rust and checked against all 384 native predictions with exact equality. Their two measured program-result entries and four canonical `jope.score/v1` inner-fold rows have now been published to the main S3 tree, with exact-version artifact closure. This qualifies the local evaluation/publication path, not a learned policy's win rate or a new-data generalization claim. Future generator hooks must call the shared gate before treating a missing score as available.

The live dashboard reports generated, published, encoded, executed and admitted counts separately. The 100,000 number is a target, never an observed count. Intake pauses behind 128 unadmitted variants; publishing alone does not release capacity. Historical developer/pretraining exposure remains explicit in the new research validation partition.

## Verified completion and live custody — 2026-10-08 23:18 UTC

Both original and corrected campaigns completed 45 fits. For the corrected study, the direct audit verified all 3,796 epochs, 79,716 optimizer updates and 19,459,594 sample presentations, with no omitted or duplicated training rows. All selected validation losses improved; 37/45 final epochs exceeded their best validation MSE by more than 5%. Complete curves and selected checkpoints are retained, rather than selecting a favorable excerpt.

Three host-local archives are live under the `dl_final` root: `paper_archive_full7120_xbabe3`, `paper_archive_original7087_xbabe3`, and `paper_archive_original7087_xbabe2`. Each uses a manifest-bound read-only allowlist, immutable content-addressed blobs, hash-linked imported observation events, and JSONL/CSV exports. Initial capture reported zero missing paths and zero read errors. Observer source SHA256 is `addfba0176a77720f0156f65f61777617e6f7741434491926215a8ca769eec0d`; 17 archive tests and independent review passed, with 70 tests across the foundation package. Each observer has CPU quota 50%, memory 512 MiB and the original campaign service deadline plus 900 seconds. The 512 MiB archive storage cap is distinct from its cooperative read budget; it is not a strict physical-I/O guarantee.

The [evidence observatory](http://127.0.0.1:8792/paper/) shows live archive counts, retained chain heads, missing/error counts, and host GPU/CPU/memory. [Full training CSV](http://127.0.0.1:8792/export-full.csv) contains actual logged epochs with source/corpus/run identities and checkpoint-selection metadata. Local copies of observed archive heads provide a second-host anchor; off-host S3 archival is a separate receipt and must not be assumed from the local archive state.

## Current experiments and their identities

| Campaign | Corpus | Study | Interpretation |
|---|---|---|---|
| Wave4 `d910a26e9c` | 7,087 views, 1,571 dataset IDs | Frozen G model, seeds 3/4/5, 3 repeats × 5 grouped folds each | Development replication; the original corpus omitted 33 eligible contexts |
| Corrected Wave4 `full7120-ea0da768ff` | 7,120 views, 1,579 dataset IDs, 1,068 lineage components | Same G model and 45-fit design; new immutable bundle | Full currently published eligible selection corpus, subject to the documented publication/role exclusions |
| Flywheel successor | Separately versioned corpus and model | Program-conditioned quality prediction, actual program evaluation, then actor tests | Must not be described as an established capability or improvement until qualified |

The corrected bundle SHA256 is `bb6b706cd7db29c9feb2e83afae9743655fb8a8a5ee39b00be45eb239075f2f4`. Its campaign manifest is `c942197bf28e292f407abd7acae61395b8cf88b6ada8cf3a35ffa5cec5d6f584`. Trainer source is `19e3a3e58aaf8a535c573ab8760bc1e6f3f3662d`. Retain full hashes in machine-readable receipts; shortened hashes in a figure caption are labels only.

The first corrected build stopped because its changed fingerprint threshold split one established lineage (18 rows across four dataset IDs). The failed candidate remains preserved as `data_v1_failed_lineage_20261008`. Recovery takes the least transitive union of all old and new equivalences, then recomputes grouped folds with the unchanged algorithm. The successful build proof records exact equality of every old feature/target value, exact old-plus-33 membership, no split of either old or newly inferred groups, and all 16 payload hashes. This is a recorded protocol correction, not an erased failed attempt.

## Original files to retain

Root on both compute hosts: `/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final`.

- Original campaign: `wave4_codex/manifest.json`, `status.json`, `events.jsonl`, `preflight.json`, run identity sidecars, launch/review/authorization receipts; original runs under `runs/dlw4-v1-s*-d910a26e9c/`.
- Corrected campaign on xbabe3: `wave4_full_20261008/runtime/manifest.json`, `status.json`, `events.jsonl`, `preflight.json`, `gpu_qualification.json`, sidecars and logs; `data_v1/MANIFEST.json`; `build_runtime_recovery1/` proofs and full lineage mapping; runs under `wave4_full_20261008/runs/dlw4full-v1-s*-full7120-ea0da768ff/`.
- Each run: `batch_metrics.jsonl`, `exposure.jsonl`, all `hist_r*f*.json`, `summary.json`, standardized and native `oof_r*.npy` / `oofraw_r*.npy`, selected model checkpoints and S3 exact-version readback receipts in the summary.
- Frozen source, feature-selection/preprocessor state, target normalizers, arm specification and environment versions. A model state dictionary alone is insufficient to reproduce inference.
- Authoritative data/role inventories and exact S3 key, VersionId, byte length and SHA256 for referenced inputs. Preserve exclusions and failure reasons as well as admitted examples.

The web dashboards are projections. They do not replace raw evidence. Existing source files and old results are read-only; new archival objects are create-only. Do not rewrite an old run to add a field that was never observed.

## What each loss means

The displayed comparable fitness curve is mean squared error against the GP fitness target after normalization derived only from the inner training rows. Train and validation comparisons use EMA weights, evaluation mode, and dropout disabled. The fixed train evaluation panel is measured every five epochs; validation is measured each epoch. Missing train-panel points are missing observations, not interpolation-based measurements.

The optimization objective additionally contains gap regression and Beat10 classification. Its batch value is not directly interchangeable with the displayed fitness MSE. A figure must state target, transform, weight reduction, model weights (raw/EMA), training/evaluation mode, support size and units.

The selected checkpoint minimizes **equal-lineage validation fitness MSE**, with the registered early-stopping rule. Show the selected epoch and its criterion separately from row-average validation MSE and the final epoch. The test fold does not select the checkpoint.

In the first 30 original Wave4 fits, selected validation MSE improved in every fit; median reduction from the first epoch was about 67–70% across seeds. However, 22/30 fits later rebounded by more than 5% while train error continued falling. Report both observations. Continuing every fit to its 250-epoch cap would not constitute better or more complete training.

## Exposure and randomness

Existing Wave4 emits batch metrics and epoch exposure receipts: number of rows, unique rows, minimum/maximum observation count, omitted/duplicated rows, fit/validation/test membership hashes and optimizer step. Short final batches are retained. The frozen source applies a fresh deterministic permutation each epoch from the run/fold seed.

**Historical limitation:** the current trainer logs membership hashes, not an independent ordered-permutation hash. Its random ordering is source/RNG reproducible, but do not invent historical per-batch membership or claim a recorded ordered trace.

For new training code, add native events for epoch-start RNG-state hash, ordered row-ID hash, each batch's ordered row-ID hash, weighted/unweighted support, component losses, learning rate, finite gradient norm before/after clipping, optimizer step, scaler state, parameter-change proof and elapsed time. Record validation at its actual optimizer step. Named RNG streams must separate sampling, dropout, augmentation, GP evaluation and diagnostics; diagnostics must not consume the training RNG.

## Flywheel evidence schema

Every run registers an immutable manifest: objective and selection rule; source and dependency hashes; dataset/fold/encoder/program/baseline/evaluator/vocabulary identities; seeds; data roles; CPU/GPU/storage/time caps; and planned comparisons. Append events with a unique idempotency key, run/manifest/source identity, event type, canonical finite JSON payload, observation UTC, monotonic timestamp and clock-origin identifier. Clock values from different origins cannot be subtracted.

The evidence ledger uses SQLite transactions, a SHA256 event chain, collision refusal and create-only exports. Hash chains detect changes relative to a retained head digest; they do not provide independent signing or prove that a writer's observations are true. Export the chain head/count and artifact manifest to independent custody with exact-version readback. Missing/deleted tails require the externally retained head/count to detect them.

Distinguish `native` events from `imported` observations. Imported records bind their original receipt hash and observation time. Never backdate an imported observation as a native training event. Sealing a completed run binds each explicitly admitted artifact's bytes and hash; it must not silently follow arbitrary paths supplied in an event payload.

The job queue also records every state transition. Expired work enters `needs_reconcile`; it cannot be blindly retried because the external process or S3 write may have succeeded. An unresolved lease occupies capacity until readback establishes its outcome and release. Dataset generation, encoding, baseline scoring, proposal, execution, publication, admission and training have separate completion counters.

## Dataset growth and validation

The new flywheel assigns each authoritative parent lineage using a fixed, versioned SHA256 threshold with 5% validation probability. Every alias, target, original fold and synthetic descendant inherits that assignment. This gives approximately 5% of independent families, not exactly 5% of rows. Report the actual parent and row fractions. Do not move old parents when new parents arrive; a newly discovered cross-role lineage link blocks admission for reconciliation.

The corrected Wave4 preserves its original grouped CV design, including 10% inner validation within each outer training fold. It is not silently converted to the new 95/5 flywheel protocol. Existing sealed data remains outside learning. A newly designated validation family previously seen by the checkpoint is exposed development data, not a fresh generalization test.

Count independent parent families, dataset IDs, task folds, variants, encodings, unique programs, evaluated program-context pairs, training rows and row presentations separately. One hundred thousand synthetic variants are not one hundred thousand independent datasets. Transform parameters and fitted scales use training rows only; changed features require new encoder, baseline and GP evaluation identities. Never inherit a parent's GP score for a modified dataset.

## Program improvement measurements

Wave4 predicts an outcome from a dataset encoding; it does not generate a GP program or distinguish candidate programs for one dataset. Program-conditioned PolicyLM and native experimental V3 are separate models with separate checkpoint formats and qualification paths.

For each real proposal record canonical DSL/program hash, proposing policy/checkpoint, proposal seed, dataset/fold identity, grammar validation, emitted Rust/source hash, emitter/compiler/runtime identity, execution resources, actual score and durable artifact references. Missing code or score is a queued dependency, not a successful library entry. Generated programs execute through the bounded existing engine with no storage credentials; the trusted publisher handles S3 separately.

Freeze the known opponent pool before evaluating the candidate. Draw a reproducible uniform random opponent and identify the current eligible winner under the exact same task, dataset hash, cut, evaluator, vocabulary, protocol, split and metric direction. Store the pool hash and selection seed. A different fold or evaluator is a different comparison. Predictions never count as executed wins.

Report wins, ties, losses, attempted failures, timeouts and pending evaluations. Failed attempted programs count as losses in the attempt-level metric; pending work has an explicit denominator and is not silently dropped. Native score improvement and budget-normalized utility accompany win rate. Elo is descriptive and order-dependent; retain match order, starting ratings and K. It is not a replacement for paired, lineage-clustered confidence intervals.

## Paper analysis and claims

- Predeclare primary metric, candidate model(s), data snapshot, seeds, selection rule, comparator, compute budget, bootstrap unit/count and stopping conditions before inspecting the relevant comparison.
- Report every registered arm and failed run, with exclusion reasons. Keep a trial ledger for post-result changes and distinguish exploratory from confirmatory analysis.
- Compare on matched rows, folds, seeds, evaluator and data bundle. The old 7,087-row LightGBM baseline cannot establish lift on the corrected 7,120-row bundle.
- Use lineage-level paired resampling; preserve paired predictions and multiplicities across repeats. Report support, confidence intervals, per-seed results and aggregate policy, not the best fold alone.
- Keep fixed-validation repeated tuning and unknown pretrained-encoder exposure explicit. Dope/Hyperion exposure was not established by this audit.
- Export full-resolution CSV/JSONL plus plotting code, immutable input manifest and a generated figure manifest. Downsampling is a rendering choice only and must not alter stored measurements or statistics.
- Separate five milestones: source implemented, unit/fixture verified, hosted CI/review passed, live execution qualified, measured improvement established. Serving promotion is a sixth, separately controlled milestone.

## Handoff checklist

Archive the run manifest, reviews, test counts/failures/skips, exact source head, resource envelope, actual commands and exit status, raw metrics, selected checkpoints and preprocessors, OOF predictions, coverage and role checks, exclusions, S3 readbacks, analysis code and figure data. Publish a compact claim-to-receipt index. Update the shared handoff under its append lock with completed work, remaining blockers, owned services and stopped/active states. Never report a queue launch as a completed training experiment.

## Evergreen first cohort — 2026-10-09 00:49 UTC

All 128 new datasets have create-only S3 publication with exact VersionId and SHA256 readback: 512 objects, 12,241,675 bytes. They span 128 parent lineages (125 training, three validation); 78 use 1% target noise scaled from designated inner-fit labels, and 50 retain their source targets. Rows range from 80 to 1,634; features from one to 210. No outer-test rows are included. The publication receipt is `DL_FLYWHEEL_EVERGREEN_PUBLICATION.json` (SHA256 `7f3d945a66a57945662735eb172d0eddc103284a5dc15c3949f94bef22588734`).

The original pilot and this cohort total 228 distinct variant identities in S3. Publication does not imply fresh labels or training admission: the evergreen backlog remains 128 published and zero admitted. Report generated, published, fully encoded, freshly scored, admitted and actually presented-to-training counts separately.

## 2026-10-09 01:14 UTC: observed-program bridge measurement

The generic observed-program bridge produced one new measured exact-context result on a 102-row, six-feature task with 1% target noise. Both inner folds used 51 scoring rows. Native evaluation, emitted Rust compilation and execution matched all 102 predictions bit-for-bit. RMSE was 0.006271888006852582 and 0.010831365897680825 in prepared target units. Seven artifacts plus the lookup record were created in S3 and verified by exact VersionId and SHA256. Two subsequent phase observations returned the same measured result without duplicate execution.

The two initial program qualifications plus this cold-miss execution total three measured program/context results, six actual fits and 486 bit-exact Rust predictions. These are infrastructure measurements, not DL-generated proposals, selection win rates or evidence that augmentation improves generalization. The third task differs from the original pilot; its raw RMSE must not be ranked against that pilot's errors.

Receipt: `DL_FLYWHEEL_PROGRAM_BRIDGE_VERIFICATION.json`, SHA256 `8ebf8112183712e7473703475c760ee291e634bf7b500db14bbd91138f1c97bd`. The source journal retains four origin observations and eight chain events. Training and validation are origin labels in this functional proof; the deliberately synthetic checkpoint provenance is recorded. The program input's lineage partition remains independently enforced. A live model-emission hook remains unqualified.

## 2026-10-09 01:25 UTC: diverse evergreen encoder closure

Sixteen new tasks from sixteen parent lineages completed all eight encoder blocks, totaling 128 locally verified vectors. The subset includes seven changed targets, three validation lineages and a one-feature task. Final receipt: `DL_FLYWHEEL_EVERGREEN_ENCODE16_FINAL.json`, SHA256 `e128d43855ed38ebf239261b91eaf3ac6b3654513bca65f3b95011e18bccccb4`. Do not count the original failed xbabe2 attempt and its successful recovery as additional tasks or encodings. The original failure, unchanged absolute deadline, exact recovery source and final phase/coordinator ledger heads remain part of the evidence. Separate S3 publication receipts establish custody; this encoder receipt alone establishes neither fresh GP labels nor optimizer exposure.


## 2026-10-09 UTC — first real continual PolicyLM update

The prior Wave4 critic and its multi-encoder curves remain historical. A fresh Kumo Small-only2,051-input PolicyLM has now produced actual program proposals and changed its parameters. Its live stream is /paper/, with raw committed batches, completed epochs, unique data_runs seen and separate measured program outcomes. The full design and limitations are in DL_CONTINUAL_MODEL_V1.md.

Frozen neural config: ce55dfe5f887d78e50937545d294850625fc036da5305cb3b1070205a3699cf5.
Actual accepted snapshot:0f828765b6ec3a2e2947a7d1696923021cc2ac917e3b6861cfe4b6b911578cc7.
Native TrainData:dfd6136ec7b38704275d283fb3fbfd182792c67c92407fdfa9f131522aa004d6.
First physical-update proof:DL_FLYWHEEL_NEURAL_FIRST_UPDATE_PROOF.json, SHAa694d3c719e28a990fd9bc2de7c6622f84dac54b30040a80de9f05347904233e.

First batch: train fixed-panel loss1.5667484998703003 →1.4988421201705933; validation1.4469008445739746 →1.3929407596588135. The initial parameter hashcd435e85… changed toec6868257f288f73d4b1045c818a99f65945ccc29f2fa60f98387c963d43462d. The physical checkpoint contains144named AdamW state tensors and CPU/CUDA RNG state. All three generated proposals for that batch had ready measured cache/execution closures before the optimizer update. This establishes one actual update, not a generalization result or production promotion.

Source/runtime failures remain evidence: the first loader stopped at its256MiB read cap; recovery1 expired during prefetch after a delayed dispatch; recovery2 found a real relative-vs-absolute encoder output-path mismatch; recovery3 preserved all content/source checks while resolving the phase-relative path and successfully verified567,179,944artifact bytes. None of those failed attempts counted as optimizer exposure.

The current growing corpus has128published variants and128required Kumo vectors in S3. The initial panel now has128exact-context GP results (original125 plus separately measured3fill), with256inner-fold RMSE rows. The first immutable neural snapshot intentionally retains125; later outcomes require a subsequent explicit snapshot. There are13train and3validation contexts; native supervision supports7train groups and all3validation groups, with6train contexts restricted to feasibility supervision.

Original-corpus audit: all7,120previous Kumo rows are finite and already exist; no need to regenerate these encodings. Their full program-policy label/custody bridge remains pending, and the new model must not be described as trained on all7,120.

New paper archive source has independently reviewed byte custody for physical checkpoints, preprocessing, source/config, actual order/journal, loss components/support, proposal measurements and game records. Its actual S3 completion will be recorded separately after the first closed run. Source integration remains proposed to DLTrain after PR267 and OWNER queue approval; source custody under ~/JopeDime/.agent/proposals is not a merged PR.


## Actual first neural segment and capacity release — 2026-10-09 03:00 UTC

Eight actual updates, one complete epoch, 13 unique train contexts, 15 presentations and three validation contexts. Fixed train loss 1.5667485→1.291574; fixed validation loss1.4469008→1.224733.23actualdecodedproposals:14newexecuted+published,9initialcachehits,0failures. Validation score(tieshalf)62.5% vsrandom and6.25%vsbestpanel, eightgames/threecontexts. This is early qualification; loss movement and executed-policy strength are distinct.

Physical final verification: DL_FLYWHEEL_NEURAL_FINAL_VERIFICATION.json SHA35b38c2265fee43c4302bd8032da5a8ecee2a2f64ad8fccada14585a5904d1ed. Paperarchive130files,87,562,240bytes, packagec3180ce93eae0ec2937f316107fcf678e7c17320af9fade069a0f19ceb3ec087; completion82ac7c91d7a49baaf11cee81cc27ee59bfc4caaabb8e10819aa1c561ddcaa594 includes independent exact-version receipt readback.

Actual exposure admission committed exactly10supportedrows(7train+3validation) againstplan199f0cbd90c631cdae88beabbe8583ed4a95efcedc880a4abaddcd34926e3303, with sixNOT_GROUP_ELIGIBLErowsleftpending. Commitreceipt d90d3c784402c52161b4e2ec1e9e76a702f0f6c2bccaad92d134d5454c1384de; atomicDBproof745e76c9f3edd9cccda3d4cf29258da7f60b108d7ec243e6af319fa02dcd69e3. Generator thencreatedsequences128–137:138generated/128S3published atthisreceipt. Those10newrowsare notyetcountedasencoded/scored/trained.

GP112initialexpansionattempt passedsource/resource/input/compilerpreflight but nativeworkerhardlimit128rejectedits448queue beforeanyGPexecution. Zero execution is retained, not counted as896successes; a separately versioned bounded worker/preflight fix is required. The initialfailedattempt and its deadline/custody remain preserved.
