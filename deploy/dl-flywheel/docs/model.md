# JopeDime continual GP policy — research candidate v1.0.0

## Status and scientific scope

Updated 2026-10-09 03:00 UTC. This is the corrective continual-policy candidate, separate from the historical DL_wave4 outcome critic. It is not a production release or a demonstrated best-generalizing model. The first GPU qualification completed eight actual optimizer updates, one full epoch, 13 unique training data runs, 15 training presentations and readouts on all three validation contexts. Fixed-support training loss fell from 1.5667485 to 1.291574; validation loss fell from 1.4469008 to 1.224733. All 23 model-decoded GP proposals resolved before their updates: 14 new native executions with Rust parity and S3 publication, nine exact-context initial cache hits, zero failed proposals.

Validation score rate (ties count one half) was 62.5% versus a random panel GP and 6.25% versus the best panel GP, across eight proposals on only three contexts. The loss improvement has not established superiority over the strongest measured opponent. The physical nine-checkpoint chain, AdamW state, CPU/CUDA RNG and exact result references are independently verified in DL_FLYWHEEL_NEURAL_FINAL_VERIFICATION.json. The 130-file, 87,562,240-byte archive has exact-version S3 custody documented by DL_FLYWHEEL_NEURAL_PAPER_ARCHIVE_COMPLETE.json.

There are 128 S3-published new data runs with required Kumo Small vectors, and the first 16 have a complete eight-program measured panel (128 fixed GP results). The first frozen loader retains the original 125 results; the three later completions belong to a subsequent snapshot. Verified exposure admitted ten supported contexts through an atomic evidence-bound transition, causing the generator to create ten more variants (138 generated; the ten new variants still await publication/encoding at this update). Six auxiliary-only training contexts remain pending rather than releasing main-objective capacity. Follow the live dashboard for subsequent observations.

First neural config: ce55dfe5f887d78e50937545d294850625fc036da5305cb3b1070205a3699cf5. Exact admission plan: 199f0cbd90c631cdae88beabbe8583ed4a95efcedc880a4abaddcd34926e3303. Actual commit receipt: d90d3c784402c52161b4e2ec1e9e76a702f0f6c2bccaad92d134d5454c1384de. The next segment must continue the same weights, optimizer, RNG and global batch journal; it must not reset the model or splice a new run into this curve.

A data_run means one dataset version with one exact split definition. Two inner cross-validation scoring passes on that data_run are two scores, not two new inputs. A genuinely different split receives a distinct identity. The new growing corpus, the historical 7,120-view corpus, and the earlier 100-variant one-parent pilot are separate inventories. The new policy is not yet consuming all 7,120 original views.

## Required input and explicit exclusions

The sole input block is NVIDIA Kumo-Tabular Small: 2,051 values, comprising 2,048 pooled hidden values and three query diagnostics. Model repository revision: 3c3e10bbdb590ace29e7026847f92db3c603096d. Small regressor weight SHA256: fc672f536501d64356036050ac2acea7e46927f7b9ad38366abe048f88e7d423. The roster/license audit is DL_FLYWHEEL_KUMOS_ROSTER_AUDIT.md/.json.

No PriorLabs/TabPFN 3.5, A0, Hyperion, Kumo Large, Mitra, TabDPT, TabICL, DOPE, Lion statistics, or other-model prediction diagnostics enter this candidate. Historical additional vectors remain preserved, but are not read by its loader. Historical mixed-input model performance cannot establish Kumo-only performance.

Input vectors are standardized using means and scales fit only on training contexts. A zero scale becomes one; standardized values are clipped at ±10. The raw vector and preprocessing statistics remain reproducible. There is no learned feature selection or validation-selected encoder filtering in this version.

Each input is joined to its exact target identity, split, source lineage, raw content hash and current role authority. Changed features, target noise, or split require fresh encoding and exact-context scores. Parent scores are not copied to descendants. Outer test rows and excluded/sealed families are outside the optimizer closure. The current authority still excludes all 1,565 G3 views pending an explicit rights ruling.

## Model

The native PolicyLM has 828,421 parameters: a Kumo context projection followed by a causal prefix Transformer, d_model=128, two layers, four attention heads, feed-forward width512, dropout0.1, vocabulary282 and maximum128 tokens. The context conditions the program token sequence. Training uses native AdamW parameter groups, learning rate0.0003, weight decay0.01 with the native bias/norm exclusions, and gradient-norm clipping at1.

The first live qualification uses actual model logits for autoregressive sampling through a fixed trie of16 preregistered native-valid programs. This deliberately limits the action space while proving the complete execution loop. It does not claim unrestricted GP topology generation. The initial eight controls include ridge, standardized ridge, robust-scaled ridge, ranked ridge, RFF+ridge, random-projection+ridge, row-statistics+ridge and elastic net. The additional allowed programs change registered numeric parameters.

Every emitted proposal includes model/checkpoint identity, batch index, data_run/context identity, exact program string/hash, decode seed and token evidence. Sampling seeds bind fixed run seed, batch index, data_run ID and position, rather than a wall-clock-dependent record identity.

## Loss and what a falling curve means

This candidate retains the native successor-v1 objective:

L = CE + 0.25 × Bradley–Terry pairwise loss + 0.10 × listwise loss
    + 0.25 × quality Huber loss + 0.10 × feasibility CE
    + 0.05 × runtime Huber loss.

CE teaches measured high-quality program sequences. Pairwise and listwise terms teach ordering among measured programs for the same context. Quality regression estimates normalized measured quality; feasibility and runtime are auxiliary tasks. Unsupported terms remain explicitly unsupported rather than receiving invented targets.

The native teacher/comparator rules are unchanged, including distinct program-skeleton support. The run requires nonzero real teacher and ranking support in both training and validation before any optimizer update. Feasibility-only data cannot masquerade as successful program learning.

The plotted train and validation losses use fixed context/member panels, with dropout disabled during both readouts. This makes their changes interpretable across updates. A lower validation objective is useful evidence, but not equivalent to a higher executed GP win rate or generalization to new independent datasets. The dashboard therefore reports measured GP comparisons separately.

Initial optimization uses one frozen measured-label snapshot. Newly executed proposal outcomes are stored immediately but become optimizer targets only in a subsequent versioned snapshot. This is supervised policy/ranking qualification, not an online policy-gradient or reinforced-learning performance claim.

## Normalization decision

Retain every raw inner-fold RMSE and target unit. The current native objective uses its original same-split mean-baseline normalization:
q = -0.5 × sum over two folds of log(max(GP_RMSE,epsilon)/max(mean_baseline_RMSE,epsilon)).
q is clipped to ±log(1000) for optimization. The context-specific epsilon is max(1e-12 × max(1,abs(mean(y))), 1e-6 × robust_target_scale), with MAD, then IQR, then standard-deviation fallbacks. A positive q means lower error than the mean baseline. Baseline disagreement or missing support does not become an invented target. A separately versioned CatBoost-relative objective is planned; it will not silently replace or mix with the current objective.

A useful candidate target is log((baseline_RMSE + epsilon)/(GP_RMSE + epsilon)), with a documented scale-aware epsilon and fixed robust clipping. A positive value means lift over the baseline. Advantages: less dominance from target units and a stronger difficulty-adjusted comparator. Risks: near-perfect baselines make ratios unstable; baseline weaknesses become part of the target; train/validation protocol mismatch causes leakage; normalizing does not create independent data. Fit the baseline only on the complementary inner fold, use a fixed configuration without validation tuning, and preserve raw errors plus denominator support. The CatBoost job and its measured comparisons are not claimed as completed here.

## Shuffling, exposure and validation

The planner persistently records membership fingerprints as sorted data_run IDs, while separately recording shuffled order. It permits at most one source lineage in a batch. Epoch membership is exhausted before starting a new pass; exact membership repetition is rejected. When a finite pool cannot produce a new permitted batch, the run waits explicitly instead of recycling one invisibly.

A reservation is not exposure. A batch counts only after actual backward/optimizer work, changed model parameters, physical checkpoint creation, optimizer evidence and a committed planner record. Reserved-but-uncommitted work remains visible for reconciliation. New frozen corpus adoption and any restart must preserve this journal; creating a fresh journal must not be presented as global deduplication.

Approximately5% of source lineages are reserved for validation; all their descendants retain that role. The current qualified source registry has1,068 lineages:1,014 training and54 validation. The first16-context qualification has13 train and3 validation contexts for development support, so its ratio is intentionally not the full-registry ratio. Fuzzed descendants do not count as new independent source families.

## Mandatory GP execution and S3

An emission is durably recorded before resolving its outcome. Cache identity includes canonical program, dataset bytes, target identity, split, evaluator, vocabulary and protocol. Exact matching is required; a program scored on a different split or target is not a hit.

Uncached programs use the local native evaluation route and GP-to-Rust emitter. Both inner folds are scored; generated Rust predictions must match the native evaluator within1e-9 × max(1,abs(reference)). Seven artifacts plus the final lookup record are published create-only, then checked through exact VersionId GET and SHA256. The lookup record is written last. Failures and ambiguous outcomes remain explicit and have no numeric reward. A batch cannot update until all its emitted programs have verified results.

Bucket: veox-jopedime-learning-use2.
Research prefix: experiments/gp-policy-dl/v2-trainonly/.
Dataset artifacts: datasets/flywheel/v1/<full-identity>/.
Encoder/dataset completion receipts: receipts/flywheel-datasets/v1/.
Program result library: program-results/flywheel/v1/.
Attempt custody: program-attempts/flywheel/v1/.
Paper archives: paper-evidence/20261008/neural/.

These are versioned research flywheel records, not an assertion that the native production HWM or original dataset catalog has been promoted.

## Fuzzing and growth

The source registry keeps train-only prepared views under2,000 rows. Every derived dataset preserves exact source hashes, source lineage, split definitions, recipe version, random seed, added/dropped columns, feature correspondence, target-noise level, target units and caveats in JSON.

The current recipe can drop0/5/10% of columns while retaining at least one, append bounded Gaussian nuisance columns, and add0/1/3/5% target noise relative to training-label scale. The first128 happened to exercise unchanged and1% target noise. Feature noise is disabled until categorical/prepared-column semantics can be established. No-op content is rejected; labels/scores are never inherited for changed inputs. Existing inner split definitions are preserved; manufacturing new independent outer folds is not yet implemented.

Generation targets100,000 variants, rotates source lineages and limits descendants per lineage. It is subject to a bounded pending queue. Published, encoded, scored and actually trained are separate states. Only verified train/validation exposure may release pending capacity; changing a count or ticking a publication flag is insufficient.

## Comparisons

Each measured model proposal is compared against a uniform random member and the best measured member of the frozen eight-program panel on the exact same context and protocol. The comparison metric is the geometric mean of the two measured inner-fold RMSEs. This panel winner is not the global S3 HWM.

The dashboard's score rates use validation proposals only: win=1, tie=0.5, loss=0. Diagnostic Elo starts at1,500 against a fixed1,500 opponent with K=16 in chronological order. Training and validation games remain separate; distinct-context count is shown alongside repeated games. With only three initial validation contexts, these numbers are pipeline qualification evidence, not a white-paper generalization result.

## Compute, logging and limits

The first policy qualification is on xbabe2: one GPU, at most8CPU and8GiBRAM, no swap, maximum8 batches of2 training contexts plus one rotating validation proposal per batch. The new1200-second run reserves180seconds for final attempt custody. xbabe1 is left for its owner. xbabe3 handles separately bounded corpus preparation after resource checks.

Every run retains immutable source/config/role/snapshot hashes; actual batch order and membership; raw and weighted loss terms and support; gradients and checkpoint hashes; named AdamW moments; CPU and CUDA RNG tensors; preprocessing statistics; generated token/program records; actual measurements and exact S3 versions; failures; and measured comparison games. A separate closed-run paper archive preserves checkpoints and evidence with exact-version readback.

Completed gates: actual neural optimizer qualification, full first-run paper custody, and verified exposure-based generator admission with observed new generation. Remaining completion gates: persistent model continuation; publishing/encoding the next ten variants; scheduling/labeling the remaining112 encoded variants; original7,120-view admission audit; evaluated CatBoost normalization; broader program generation; long-run continuous scheduling and resource arbitration; source PR integration after the existing owner queue clears. These are explicit remaining tasks, not completed capabilities.
