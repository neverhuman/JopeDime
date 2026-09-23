# Real training history, loss trajectories and the limits of progress

## What this history establishes

ASM means **Algorithm Synthesis Model: learns to generate and optimize executable algorithms from data**. The repository has real historical neural training and real retained GP teaching corpora. It also has substantial uncompleted work before a newly bounded ASM learning campaign can establish useful improvement. These facts must be stated together. “Zero freshly admitted teachers under the new ASM admission guard” describes the new campaign's inspected readiness; it does **not** mean that JopeDime has never trained on real data, that the accepted policy is untrained, or that its historical ten-million-record corpora are synthetic.

This chapter reconstructs progress from original retained metrics and checkpoint metadata without loading model tensors, reopening sealed targets, or starting training. It separates three kinds of evidence: loss measurements, changed training state, and improved executable algorithms. Losses measure how the model fits its training objectives. Checkpoint lineage records which parameters and optimizer state were produced. Only a matched downstream fit/search comparison can establish that conditioning on a dataset and split produces better algorithms. A completed optimizer run does not imply the third result.

The supplied CSVs retain original values, source file SHA-256, source path and line, and JSON Pointer where appropriate. There is no interpolation. Seven figures show different populations separately; joining them into one curve would erase changes in architecture, objective, corpus, sampler and validation protocol.

## Dated lineage and evidence coverage

| Observation or historical interval | Retained state | What is established; what is not |
|---|---|---|
| Jope Prime gen31, preserved by the provenance inventory dated 2026-07-12 | Fourteen complete epoch-summary rows, explicitly `policy_v3.2`; immutable warm-start checkpoint identified as `6a204434…`; 125,970 training and 2,169 validation rows | Real logged training and validation measurements. The log itself has no wall-clock timestamp or exact trainer-source hash; its event date is not inferred from file modification time. The provenance inventory's date is a historical declaration, not a newly verified execution date. |
| Lineage A, historical xbabe2 Python control plane | Four surviving frozen corpus snapshots and original warm-start state; substantial live control-plane state and later checkpoint descendants declared lost | Historical training and teaching data existed. The lost accepted-pointer and sealed-use history cannot be reconstructed merely from aggregate scores. |
| September 9–11 native training audit | 108 completed 3,000-step blocks, 108 different corpus identities and 62 starting checkpoint identities | Original initialization records explicitly reset AdamW. The capture retains each block's first and last metric row, not its complete per-step log. |
| September 12 recovery lineage, retained A100 checkpoint manifests | 78 checkpoints at recorded steps 4,000–69,000, 22 corpus identities; all record carried FP32 AdamW | This later lineage really carries optimizer ancestry. The accepted serving checkpoint is 60,000, not the maximum retained step. Validation remains adaptive and its corpus/panel changes. |
| September 14, 22:19:52–22:27:47 UTC | Complete 100-row RTX qualification trajectory, steps 60,001–60,100, interrupted and resumed after 60,006 | Original metadata records carried optimizer state and changed model/parameter identities. It explicitly describes unchanged-corpus hardware qualification, with zero fresh programs; it is not a production improvement or promotion run. |
| September 18, 20:17:48 UTC publication manifest | Product bundle `policy-step-60000`, source checkpoint `8ac80c269cc47c253b6f53fa3bc1eb75bb2fcda0bdace3ff8554be0175c400d6` | The current product's accepted source checkpoint is pinned. Publication time is distinct from training time. |
| Current bounded ASM work | Verified regression neural-origin search evidence; stopped training64 follow-on and admission readiness | New category training and subsequent consumption of its learned export remain pending. No new GPU-hour budget is inferred from historical training. |

Sources: `docs/adr/ADR-0005-canonical-learning-loop.md:13`, `docs/learning.md`; historical native audit `learning-hpc.md:37`; original checkpoint manifests and metrics pinned in the CSVs; `models/bundles/policy-step-60000.json:7,73`; `/home/ubuntu/jope-prime/provenance.json:3,13,129`. The retained legacy source declares `policy_v3.2` at `/home/ubuntu/jope/src/jope/policy/model_v3.py:40`; it is useful for interpreting the old log, but the log does not attest that exact source build.

## Gen31: fitting the objectives versus using the context

The gen31 log contains epochs 1–14, with phase A at epochs 1–6 and phase B at 7–14. Both phases train the ranking branch. The retained v3.2 trainer lowers the context-encoder learning rate at the phase boundary, while phase B adds reward-weighted sequence imitation. It logs two GPUs, requested batch size 2,048 and learning rate 0.00015. Its final elapsed timer is 1,135.158 seconds; that timer is a run-local measurement, not a cold-install or end-to-end campaign duration.

![Fourteen original gen31 epoch summaries, including training and validation components](../figures/training-gen31-loss.png)

Every point is an original epoch summary. Sequence training loss fell from 1.563002 to 0.918725. The full training composite fell from 2.926419 to 2.022394, while validation composite changed from 7.039105 to 6.926823 and reached its minimum of 6.671162 at epoch 3. The final epoch therefore was not the best by every monitored objective. The train/validation composite gap grew from 4.112686 to 4.904429. These measurements show fitting with uneven validation progress; they do not show that a newly generated algorithm outperformed its ancestor.

Routing is a particularly visible warning: training routing cross-entropy fell from 0.642189 to 0.005816, while validation routing cross-entropy remained 3.498959 at the end. The retained labels and grouping still need to be held constant before assigning causality, but a near-zero training classifier loss alone is plainly insufficient evidence of useful routing.

![Gen31 recorded context advantage and ranking diagnostics](../figures/training-gen31-quality.png)

Pairwise validation ordering accuracy rose from 0.533419 to 0.538499. The genome-only control rose from 0.501411 to 0.503346, so recorded conditional advantage changed from 0.032008 to 0.035153: approximately 3.20 to 3.52 percentage points. This is a modest ranking diagnostic on retained candidates, not a 53.8% downstream tournament win rate. Historical top-1 regret decreased from 0.017565 to 0.016676 in the retained target's units. Predicted-reward correlation remained small, 0.036390 to 0.025889. The uncertainty/error statistic was negative, −0.196674 to −0.228137; it does not establish calibrated uncertainty.

The following definitions explain why similarly named numbers cannot be substituted for one another:

| Logged metric | Source meaning and limitation |
|---|---|
| `train_loss`, `val_loss` | Combined objectives, including learned uncertainty weights and regularizing log-variance terms. They are not sums of the plotted raw component values. Training aggregation averages executed batch losses across ranks. |
| `train_loss_components.seq_nll` / `val_loss_components.seq_nll` | Legal-token, 0.05-smoothed sequence cross-entropy with applicable provenance/phase row weights. |
| `val_seq_nll` | A separate validation evaluator uses zero label smoothing and averages batch NLL. It is deliberately distinct from `val_loss_components.seq_nll`; the first epoch logs 2.055572 versus 2.175046. |
| Pairwise ordering accuracy | Agreement on unequal-value pairs within a dataset. This implementation does not give explicit half-credit to tied predictions; its Boolean comparison has an order-dependent tie convention. |
| Conditional advantage | Conditional reward ranking accuracy minus the genome-only control's accuracy on the same retained population. It probes context use; it does not fit new programs. |
| Historical top-1 regret | Per dataset, best retained target value minus the target value of the retained program selected by predicted reward, averaged over datasets. It is not regret relative to an unknown optimum. |
| Reward CV correlation | Pearson correlation with retained `robust_delta` labels. The name does not make it a new downstream cross-validation score. |
| Head accuracy | Match at the first head-role grammar token under teacher-forced decoding, grouped by engine. It is not fitted-prediction accuracy. |
| Uncertainty/error Spearman | Correlation between sorted positional ranks of ensemble dispersion and absolute reward error. The retained implementation does not average tied ranks, so preserve the implementation identity when comparing it. |

The old v3.2 objective has seven components: sequence NLL, reward Bradley–Terry, reward listwise, runtime Huber, stability BCE, ablation Bradley–Terry and routing cross-entropy. Its profile scales are respectively 1, 1, 0.5, 0.2, 0.5, 1 and 0.5. Reward/ranking/ablation/routing terms use fixed profile scales; sequence, runtime and stability use the retained Kendall–Gal-style learned weighting. The log's `loss_weights` values are `exp(-log_var)`, not complete effective coefficients. This differs materially from the five-term native objective below. Source interpretation is grounded in retained v3.2 code; absence of an exact run-source binding remains explicit.

Sources: original `train_metrics.jsonl` SHA `d57de1b28a8678327c984a82732cd774520db2cb0994d893a45f8a5ea428e312`, lines 1–14; `/home/ubuntu/jope/src/jope/policy/train_v3.py:360,364,438,444,474,580,613`; `policy/model_v3.py:40,286,300,305,372,376`; `policy/losses.py:23,35,48,86,100`; `model.py:981,997,1015`. Full hashes and exact definitions are retained in this chapter's metadata.

## The September 11 native objective and 108 reset blocks

The audited native decoder consumes a 1,536-dimensional dataset vector and a 512-dimensional target vector. The configuration uses width 512, six decoder layers, four attention heads, FFN width 2,048, dropout 0.1, 401 vocabulary tokens, 657 embedding rows and maximum sequence length 128. The reward path has separate context processing and four reward heads. These are the observed historical dimensions, not a claim that every historical model version used identical architecture.

The active native objective was:

`L = sequence_nll + 0.25 × ranking_bt + 0.10 × ranking_listwise + 0.10 × runtime_huber + 0.05 × contextfree_bt`.

Sequence loss is legal-token cross-entropy with 0.05 smoothing and normalized program weights `0.05 + percentile^4`. Bradley–Terry uses weighted within-comparison winner/loser pairs. Listwise uses a `softmax(percentile / 0.25)` target; its nonzero entropy floor matters when interpreting an apparent plateau. Runtime loss compares `log1p` seconds. The program-only ablation term checks whether ranking can be achieved without dataset context. Stability and the seven stored log-variance parameters were inactive in this native build. Seven parameter names do not imply seven optimized losses.

![First and last sampled minibatch measurements for each of 108 reset training blocks](../figures/training-historical-endpoints.png)

The audit's original capture includes exactly two metric records per block: first and last, plus the original full log's path, SHA and reported 3,000-row count. The intermediate 2,998 rows per block are not reconstructed. These endpoints are sampled minibatches, not averages over the whole block. All 108 blocks reset AdamW, so their 324,000 summed steps are **block work**, not one optimizer's age. At global batch 2,048, the corresponding 663,552,000 nominal program-example presentations are not unique programs or discoveries. Replayed or uncommitted work outside this completed block census is not silently included.

The starting checkpoint changes in three regimes. Cycle 0 starts from the earlier campaign checkpoint `20dca670…`. Cycles 1–60 warm-start their model from the preceding block's checkpoint, while still resetting AdamW. Cycles 61–73 repeatedly start from cycle 59 (`61ba6043…`), and cycles 74–107 repeatedly start from the recovery checkpoint `c2f229c9…`. These are model-parent relationships, not optimizer-state continuation.

All 34 first-to-last reductions in composite, sequence NLL, Bradley–Terry and listwise loss occur exactly in cycles 74–107, the recovery-parent regime. Every block in cycles 0–73 has a higher final sampled value for those four losses than its first value. Runtime Huber decreases exactly in cycles 78–107. The counts therefore align with the parent regime; they are not scattered improvements across 108 comparable starts. The visible jumps at cycle 74 coincide with the rollback to the recovery parent and must not be interpreted as a continuous learning event. Endpoints still sample different minibatches on different corpora, so this is descriptive evidence, not a paired significance test or causal isolation of the parent change. The last block's training NLL was 0.561710, while its adaptive validation NLL was 1.410741; that gap is not a sealed generalization estimate.

![Adaptive validation observations from changing corpus and panel identities](../figures/training-adaptive-validation.png)

The plotted validation points are deliberately not a continuous generalization curve. Dashed boundaries identify the repeated-cycle-59 regime from cycle 61 and the recovery-parent regime from cycle 74. The validation Bradley–Terry jump near cycle 74 coincides with that rollback; it is not evidence of smooth accumulated improvement. Each source identity travels with its value. Old native telemetry exposed total loss and four component losses but omitted the active ablation component; its contribution cannot be inferred exactly by subtracting rounded or differently aggregated values. Per-term gradient norms were also unavailable. The later RTX log does expose `contextfree_bt` and weighted terms, so that historical limitation must not be presented as universal.

Sources: `docs/legacy/root-docs/audits/worldclass-redteam-20260911/learning-hpc.md:9,21,33,37`; retained `evidence/learning-native-v3/neural.rs:447,512` and `training.rs:692,712,720,863`; research capture `/home/ubuntu/jain-split/jope-dime/docs/audits/worldclass-redteam-20260911/learning-remote-metadata.json` SHA `1d2ab6ad9a8df4749bd1da000c6b1ea04d00395a777199907f3b599ad9af528b`.

## Fixed monitoring and actual promotion evidence

![Fixed-corpus and fixed-panel historical checkpoint comparison](../figures/training-fixed-monitor.png)

The fixed monitor is the stronger loss comparison in this audit: all records bind corpus `11e60a43…`, panel `fbff7a048…` and 2,048 sampled examples. Recovery was evaluated by both relevant executable builds with identical recorded outputs. Cycle 25 can be paired with recovery under executable `8ca21024…`; cycle 107 can be paired with recovery under `3382cdcb…`. The CSV retains both recovery rows even where the visual combines their identical values.

| Checkpoint | Sequence NLL | Bradley–Terry | Listwise | Runtime Huber |
|---|---:|---:|---:|---:|
| Recovery | 1.394498 | 0.533676 | 2.262484 | 0.076231 |
| Cycle 25 | 1.396643 | 0.659752 | 2.485061 | 0.065244 |
| Cycle 107 | 1.408606 | 0.542086 | 2.277948 | 0.073749 |

Sequence and reward-ranking losses did not improve over recovery in these fixed-monitor observations; runtime error was lower. This is an objective tradeoff, not an overall win. The monitor explicitly marks itself adaptive teaching validation, not independent evaluation; `training_selection_used` is false, but that does not turn previously used teaching labels into sealed data.

All 48 promotion decisions indexed in the September 11 capture were **inconclusive**, with qualification evidence not supplied. This is not a measured 0/48 challenger win rate. It establishes missing evidence at that historical observation. The later accepted 60,000 checkpoint belongs to a subsequent recovery lineage; the September 11 result must not be extended to claim that no later checkpoint was ever accepted. A full joined chain of matched executable comparisons and successive qualified target improvements is still required for a quantitative “the flywheel kept improving” claim.

## Carried state, accepted step 60,000 and RTX step 60,100

![Original A100 checkpoint validation measurements, with accepted step 60000 marked](../figures/training-a100-checkpoints.png)

The 78 retained A100 state manifests live under `/home/ubuntu/jope-node-mirror/weights/runs/eight-a100-recovery-20260912/learning/cycle-*/train-2/checkpoints/step-*/state.json`. The retained metadata covers 78 checkpoints, all with “carried FP32 AdamW with bias-correction step.” This is materially different from the 108 reset blocks. It spans 22 corpus identities and recorded steps 4,000–69,000. The graph keeps these as adaptive validation observations. A higher retained step is not automatically the serving model. The accepted 60,000 state's exact hash is `8ac80c26…`, corpus `1c840597…`, model `053df4f5…`, and parameter digest `9c7057e7…`; it records ancestry from step 57,000 and requests fresh-balanced sampling. Its request for 50% fresh comparisons does not mean 50% fresh program slots or measured downstream improvement.

![All 100 original RTX qualification step measurements, with the recorded resume boundary](../figures/training-rtx60100-steps.png)

The RTX run retains every metric record at steps 60,001–60,100. The exact 100-row source is `/home/ubuntu/jope-evidence/xbabe2-migration-20260914/accepted60000-restored/parallel/runs/xbabe2-hardware-qualification-20260914/final-runtime-v2-eof/train/metrics.jsonl`, distinct from the earlier recovery attempts. RTX 3090 attribution comes from ADR-0005; the retained run metadata itself records world size 1. Its first attempt starts from 60,000; later rows record initial optimizer step 60,006 after recovery. Therefore the graph uses optimizer step, not the run-local elapsed timer, which resets at resume. The timestamp span includes interruption/recovery and differs from the final attempt's 73.561-second elapsed value; neither should replace the other without labeling its scope.

The final state records **world 1 × microbatch 256 × accumulation 8 = global batch 2,048**. This corrects the historical ADR's “32 × 8” description for this retained qualification run. Model digest changes from `053df4f5…` to `1f202cce…`; parameter digest changes from `9c7057e7…` to `762094a0…`; optimizer identity also changes. The final state is `745be4b5e998bce85615349a5b03b901cd8ad2cf605e6dce3b86fbf777cc7388`. These are retained training-update and continuation evidence, not a fresh numerical replay performed for this manual.

The receipt commits 204,800 program presentations and zero fresh programs. Its purpose is explicitly “isolated unchanged-corpus hardware qualification, not production continuation outcomes.” Final adaptive validation NLL is 1.409792 and Bradley–Terry 0.475421, on 256 examples. Step 60,000's retained validation used a different panel and 2,048 examples; comparing those numbers as a paired improvement would be invalid. The checkpoint is not promoted by being 100 steps newer.

## What is still needed to demonstrate the learning flywheel

The desired loop is concrete: a dataset and split produce verified encodings; conditioned policy proposals become real generation-zero fits; bounded GP/evolution produces eligible improved programs; the full comparison identity admits those programs as teaching targets; optimizer updates produce a new checkpoint; matched downstream fits show whether that new checkpoint proposes and searches better algorithms. Only then can a promotion change the serving pointer. The corpus and packing chapter traces the historical input-to-target representation and separately reports observed adaptive target improvements in repeated searches with matching recorded comparison fields, including improvements in generic and shuffled arms. Those observations are evidence of target progress, without establishing neural causality. The flywheel chapter defines the remaining current acceptance work.

For every cycle, preserve both sides of the feedback: target improvement under the same task/dataset/split/evaluator/vocabulary/protocol identity, and policy improvement in fitted performance, failure rate and cost under a frozen comparison. Report cold/warm preparation and unique fitted predictors separately from recipe counts. Keep a context-free control, generic proposal control and unchanged accepted policy. A lower teacher-forced NLL can accompany memorization, degraded ranking, or no better executable search, as the historical measurements demonstrate.

The missing historical evidence is specific: full intermediate trajectories for the 108-block capture; an exact run-source binding and event timestamp for gen31; per-term gradient diagnostics; a common independent evaluation series across changed corpora; and a fully joined sequence of qualified target improvements and serving promotions. For the new bounded ASM campaign, corpus admission, actual optimizer update/reload/resume proof, trained-export consumption and matched downstream improvement remain acceptance work. Keep these gaps visible rather than replacing them with a smooth synthetic curve or treating old training as if it had never happened.

## Reusing the evidence

The adjacent `data/training-history` CSVs are long-form original measurements. `block-endpoint-metrics.csv` has 1,296 rows; `block-adaptive-validation.csv` 432; `fixed-monitor-metrics.csv` 16; `gen31-epoch-metrics.csv` 462; `a100-checkpoint-lineage.csv` 78; `a100-checkpoint-validation.csv` 312; and `rtx60100-step-metrics.csv` 700. The two original small logs are retained alongside the normalized data. Source hashes and line/JSON Pointer anchors make a plotted point independently traceable. The initial extraction audit covers four CSVs. The shipped `verify_history.py` checks all 3,218 metric `value` cells across the measurement CSVs, plus seven fields (step, model/parameter/optimizer/corpus hashes, world and global batch) and the state-file hash for each of the 78 lineage rows. It does not check every auxiliary numeric column, such as elapsed time, parent/local step, requested fresh fraction or row counts. Running that helper writes `verification.json` beside the CSVs; use a custody copy when preserving a stopped evidence bundle. Model bodies and sealed target values were not read for this reconstruction.

The plot builder's figure receipt binds the exact CSV hashes. Regeneration may change PDF structure identifiers, but it must not change these measurements, silently drop recovery records, smooth missing steps, or relabel adaptive validation as independent generalization.
