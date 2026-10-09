# JopeDime continuous GP policy — v1.0.0 research candidate

Updated 2026-10-09. This document describes the active full-corpus PolicyLM continuation, not the older Wave4 outcome critic or the 13-context qualification. It is a research candidate; measured development loss is not proof of improved GP win rate on independent unseen datasets.

## Current run and the 6,857 counter

The first full pass completed 429 optimizer updates across 6,857 distinct training data runs and scored 263 reserved validation data runs. The prior trainer stopped after that configured pass. The continuous version restores the physical checkpoint at update 429, all AdamW moments, CPU/CUDA random state and the frozen input normalization. It repeatedly reshuffles complete epochs, retaining earlier batch-membership fingerprints. It does not reset the model at an epoch boundary.

`unique training data runs` counts distinct dataset-version × split contexts consumed by a committed optimizer update. `total training data-run exposures` counts every committed presentation, including reuse in later epochs. Batches and exposures should increase when training revisits data; unique coverage should increase only when the optimizer actually consumes a new context. Two inner scoring folds do not create two input identities.

Actual continuation verification found physical update457, 7,305 total exposures and changed model parameters. At update493 the same fixed validation panel measured1.284191, versus1.381380 at update429; fixed training loss measured1.340202 versus1.427419. Read the live dashboard for current values. The original first-pass fixed losses were4.453104 training and4.394558 validation. The separate full263-context validation pass measured1.343791 at update429; do not compare this directly with the smaller fixed panel.

Active root on xbabe2: `/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/flywheel_neural_continuous_20261009_v1`. Service: `jopedime-neural-continuous-20261009-v1`. Config SHA256: `fec184217e7c50a3fc3bc1cc7e33edafdbd218d1fafa1bd0d68debf626dcd4ed`. Original full-run config: `3725119c62edb81f7e1be7a92b307a10a71812900e4f80d428c37fc60cb0fda4`. Resumed parameter SHA256: `b888b5d66fda926c3609fa068a42a022a079e2d4bbc8db3b5badec8f51acaf20`.

## Data coverage and admission

The reconciled original inventory contains14,987 data runs from3,003 dataset versions. Current role/exclusion rules admit12,581 contexts. A complete teacher/input join exists for7,120 of those:6,857 training and263 validation. The other5,461 eligible contexts are an explicit input/label-join backlog; they have not all trained. Some have terminal records without winner labels. Kumo availability alone does not make a context training-ready.

The initial continuous snapshot contains the original7,120 plus three verified augmented contexts:6,859 training and264 validation. The two additional training contexts only increase the seen counter after their actual optimizer updates. Further measured128/138-context extensions are being validated separately; their existence in S3 does not imply admission.

New snapshots are adopted only at closed epoch boundaries via an atomic inbox. They must preserve all old contexts, exact roles, original normalization and previous batch fingerprints. Raw, unencoded, unscored, unknown or failed contexts remain pending. Changed dataset features, targets or split require a fresh required encoding and exact-context GP measurements; parent labels are not copied. Every snapshot records exact S3 bucket/key/VersionId/hash/bytes closure and source/role identities.

About5% of original parent lineages are reserved for validation, and descendants inherit their parent's role. The qualified augmentation registry contains1,068 parent lineages:1,014 training and54 validation. Context proportions need not equal5% because parents have different numbers of folds. Fuzzed variants do not create independent source families. Excluded and sealed families remain outside the optimizer closure.

## Required encoder and preprocessing

Only NVIDIA Kumo-Tabular Small enters this model:2,051 values, comprising2,048 pooled hidden values plus three query diagnostics. Revision:`3c3e10bbdb590ace29e7026847f92db3c603096d`; small regressor weight SHA256:`fc672f536501d64356036050ac2acea7e46927f7b9ad38366abe048f88e7d423`.

The loader does not consume PriorLabs/TabPFN3.5, Kumo Large, A0, Hyperion, Mitra, TabDPT, TabICL, DOPE, Lion statistics or other-model prediction blocks. Historical extra vectors remain historical artifacts. New encoding jobs compute only Kumo Small.

Training-only vector means/scales are frozen from the original training corpus. Zero scales become one; standardized inputs are clipped at±10. Validation never fits preprocessing. New variants use the same statistics without refitting. No PSO/CMA-ES feature mask or validation-selected binary gate is active in this version; that remains a separately specified experiment.

## Model and program output

The native PolicyLM has893,957 parameters: a context projection and causal prefix Transformer, width128, two layers, four attention heads, feed-forward width512, dropout0.1, vocabulary282 and maximum sequence length640. AdamW starts at0.0003 learning rate with0.01 weight decay and the native bias/norm exclusions; gradient norm is clipped at1. Plateau control reduces the learning rate by0.5 down to1e-6 while retaining the best checkpoints and continuing shuffled epochs.

The decoder samples actual model logits through a fixed trie containing6,640 programs derived only from original training teachers. It does not read validation teacher programs into that trie. This is restricted program selection/generation: a decoded program may be new for a particular dataset/split even when the same string has appeared elsewhere. Arbitrary unseen GP topology generation is not claimed.

A macro-batch contains up to16 training contexts with at most one context from any parent lineage. Micro-batches respect8,192 padded tokens and do not silently truncate long sequences. Teacher windows rotate across epochs; the loader uses up to eight measured candidates plus an eligible winner per context. It does not ingest every historical GP program into every update.

Each update durably emits two proposals: one rotating training context and one rotating validation context. This is not yet one proposal for every context in the macro-batch. Proposal records include exact context/program identity, checkpoint, decode seed, token evidence and phase. Evaluation is asynchronous so GP execution latency does not block every gradient update. Shared backlog limits apply before additional proposals can overwhelm consumers.

## Loss and score normalization

The support-masked objective is:

`L = CE + 0.25 quality_Huber + 0.25 pairwise_Bradley_Terry + 0.10 listwise + 0.10 feasibility_CE + 0.05 runtime_Huber`.

CE teaches eligible program sequences; pairwise/listwise terms teach ordering among measured programs on the same context. Quality, feasibility and runtime are separate auxiliary targets. Terms without real support are masked, not filled with zero scores. New augmented programs currently contribute measured quality/feasibility/runtime and supported ranking terms; self-generated variant programs do not automatically become CE imitation teachers.

Retain both raw inner-fold RMSEs and normalize quality against the complementary-fold mean predictor:

`q = -0.5 * sum(log(max(GP_RMSE, epsilon) / max(mean_baseline_RMSE, epsilon)))`.

Optimization clips q to±log(1000). The context epsilon uses target scale with MAD/IQR/standard-deviation fallbacks and a numerical floor; source code and snapshot retain the exact formula. Positive q means lower error than the mean predictor. The baseline uses only the appropriate fitting fold.

CatBoost-relative quality is not active. A stronger reference can normalize difficulty more meaningfully, but near-perfect baselines make ratios unstable, and it adds fitting cost and another protocol dependency. Keep this run's normalization fixed; evaluate a separately versioned CatBoost-relative objective with raw errors, denominator support and training-only fit before changing it. Never mix objectives under one label identity.

Fixed training/validation panel readouts both disable dropout and retain the same members/teacher windows across the continuation. Full original validation runs at completed epoch boundaries; expanded validation is reported separately. Lower validation loss supports this development trajectory but is not an executed-program win rate or an unbiased generalization estimate. The reserved development panel also controls learning rate, so it is not an untouched final test set.

## Epochs, checkpoints and failure recovery

The planner exhausts the current training snapshot before closing an epoch. It shuffles order and groups distinct parents into new batch memberships, preserving order-independent membership hashes across the original and continuation journals. A context may appear in many epochs; a whole exact batch membership is not reused. If no legal new batch exists, it waits explicitly.

Exposure is recorded only after optimizer work, physical checkpoint creation and a committed batch proof. Failed or ambiguous updates are preserved for reconciliation. Checkpoints include model, AdamW, CPU/CUDA RNG and separate optimizer-control state. Recovery retention keeps the last two, best two and each completed epoch checkpoint; historical batch proofs and loss records remain.

The current service is a bounded continuous campaign, with50,000 additional-update limit, inherited deadline1791592146.380285,24GiB output cap and32,768 shared unresolved proposals. It continues across normal epoch endings. An existing15-minute maintenance heartbeat verifies progression and prepares reconciled bounded successors; an integrity/resource failure must not be bypassed or replayed merely to keep a counter moving.

## GP execution, parity and S3

Every generated proposal is durably queued before resolution. Exact-context cache identity includes dataset, target, split, canonical program, evaluator and protocol. Confirmed misses execute the native local evaluation route and generated Rust on both inner folds. Rust/native predictions must agree within the fixed tolerance `1e-9 * max(1, abs(reference))`. Successful artifacts and the final lookup record are create-only and verified by exact-VersionId GET plus SHA256. The result record is published last.

Native/Rust parity failures are explicit implementation/evaluation failures, not fabricated bad RMSEs. Six such failures were retained during the first cached native expansion; numeric publication and gradients are withheld. Consumers preserve unknown attempts and immutable local/S3 custody. Never loosen the tolerance to inflate throughput.

Newly measured results become training targets only through a later admitted snapshot. Win rate and Elo require matched same-context evaluation against frozen random/winner opponents. Those comparisons have not yet been calculated for the current full model, so the dashboard leaves them unavailable. Historical13-context comparisons must not be relabeled as current-model results.

S3 bucket:`veox-jopedime-learning-use2`; research prefix:`experiments/gp-policy-dl/v2-trainonly/`. Derived datasets live under`datasets/flywheel/v1/<identity>/`; source/transform JSON and exact splits accompany each. Encoder receipts live under`receipts/flywheel-datasets/v1/`. Program-result and attempt namespaces are schema-versioned and distinguish native-original from augmented contexts. These are research records, not silent promotion of the production GP high-water mark.

## Growing the corpus and compute allocation

The augmentation generator rotates parents toward100,000 distinct variants under2,000rows. Recipes preserve source hashes, lineage, split, seed, column correspondence, target units and transformation parameters. Approved changes include bounded column drops, bounded synthetic Gaussian nuisance columns and0/1/3/5% target noise scaled from designated fit labels. Generic feature noise remains disabled until continuous/categorical semantics are qualified. No-op content is rejected. Current variants retain the source inner splits; a general new-fold factory is not yet implemented.

Raw generation, required encoding, GP scoring and training admission have independent bounded queues. Raw generation no longer waits for optimizer exposure. Its current unencoded buffer is4,096 contexts; a plateau at that bound means encoder throughput is limiting generation. Historical encoded receipts are reconciled separately from new stage commits.

xbabe1 runs the dedicated required-encoder pipeline; xbabe2 runs the continuous model, generation and bounded native GP consumers; xbabe3 supports GP evaluation and other coordinated owned work. Resource checks account for foreign workloads. GPU memory is used for actual tensors/model work; filling memory or maximizing utilization alone is not the throughput objective.

## Timing and white-paper evidence

All three hosts record UTC nanoseconds, monotonic timestamps, source identities, CPU counters, cgroup user/system/throttling, memory/swap/pressure, I/O, network and GPU utilization/memory/power. New instrumented workers record stage spans, process/child CPU time and CUDA event time. Wall duration, summed worker time, utilization samples and actual GPU kernel time are different measurements and must not be added indiscriminately.

Hash-chained host segments are sealed, published create-only and exact-version verified in S3 under`paper-evidence/20261009/hpc-trace-v1/`. Active unsealed segments remain local until sealed. Checkpoint, program and stage evidence have their own archives; the host archive alone does not prove every model artifact is already in S3.

The completed844.3-second first pass spent309.4seconds(36.6%) on input preparation,252.6seconds(29.9%) on decoding and29.0seconds(3.4%) in the optimizer stage. Remaining time includes validation, checkpoint and other overhead pending finer attribution. Exact repeated native verification measured2.644seconds/18downloads versus0.149seconds/zero downloads after cache warm-up, with identical measured scores. Cold-cache work was slower; this17.7x warm verification gain is not an end-to-end speedup.

See `DL_HPC_BOTTLENECKS_AND_TIMING.md`, the live dashboard and exact source receipts for current throughput findings. Source is being integrated under`~/JopeDime/deploy/dl-flywheel` on xbabe2 in PR277. No release tag, production serving claim or completed100k-corpus claim is implied.
