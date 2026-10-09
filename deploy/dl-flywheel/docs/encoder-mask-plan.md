# JopeDime: encoder gates and the next defensible experiment

Decision date: 2026-10-09 UTC. Status: research-informed design, not a launched mask-search campaign or a measured improvement. This extends the full-corpus flywheel work; it does not replace the source, weights, input roster or metrics of the running qualification model.

## Decision

Build a **mask-conditioned, fixed-shape encoder front end**, train it with structured masks, and evaluate a small, nested binary search against the actual downstream network. Keep LV4 learnability features as the anchor. Prioritize Kumo-L and Kumo-S; make provenance-qualified TabICLv2 and Mitra available as separate gated branches for an incremental test. Reuse existing encodings. New augmented datasets continue receiving only the selected deployed roster until an expanded roster passes its experiment.

Use binary PSO as the primary discrete challenger and an equal-budget random-mask search as its control. Use CMA-ES only for a small continuous group-weight problem, or its explicit mixed-integer margin variant for binary variables. Do not repeat a million-dimensional-mask-evaluation search merely because CPUs are available. The scarce resource is independent validation evidence.

The strongest measured GP-HWM outcome predictor remains LV4 + LightGBM. It is the benchmark and fallback. A GP program generator solves a different problem: its primary outcome is the score of the program it produces on the exact data run. Better HWM prediction does not by itself establish better program generation, and the encoder-null HWM studies do not prove that encoders have no value for choosing a program.

## Evidence read

Read the entire `/home/ubuntu/jopedime_probe_exp/ofm-20261008/ENCODER_RESEARCH_MASTER_SUMMARY.md`, 46,780 bytes, SHA256 `5bbd743e62658aca21a9968f84bb44a25a4b05b83481cc617c1bfbdf3cf168de`. Also read the actual xbabe2 `encsearch/results/summary_table.txt`, `encsearch_results.json`, `figs/dl_waves/confirm2_report.json`, `DL_FINAL_DESIGN.md` and the ledger. Their source snapshots are saved in `work/flywheel-20261008/encoder-mask-research-sources.json` in the task workspace. The summary is dated research evidence; current role authority, S3 inventory and live processes are refreshed separately.

| Observation | Consequence |
|---|---|
| Clean-pool LV4 + LightGBM: fitness R² 0.785, beat10 AUC 0.911 | Keep a strong same-corpus benchmark; do not subtract results from different pools or weighting schemes. |
| Encoder-only Kumo-L + Z: R² 0.491 on the probe; within DL, adding Kumo-L to LV4 improved the weak all-clean baseline | Test Kumo-L explicitly; concatenating every encoder is not the evidence-supported default. |
| Removing LV4 cost wave-0 blockformer R² 0.584; individual encoder ablations changed R² by at most 0.005 | Do not mask away the main learnability signal just to find a sparse encoder mask. |
| ENCSEARCH: 1,090,324 evaluations, 369.7 main core-hours; fitness nested lift −0.011…+0.013 with every CI including zero | A stronger optimizer alone does not solve selection overfitting. |
| Several nested LightGBM beat10 mask arms lost approximately 0.009–0.011 AUC, with CIs below zero | Search must include matched random masks and an ungated model, with an explicit no-promotion outcome. |
| Earlier dimension-mask inner–outer gaps were 0.28–0.40 | Search coarse groups first; fit preprocessing inside each training split; reserve outer evaluations. |
| Fitness-favored and beat10-favored DL arms differ | Use separate supervised heads and report both; do not hide a tradeoff in an arbitrary combined score. |
| Blend confidence intervals include zero, but fitness point estimates remain negative | “No significant difference” is not proof of equivalence or a win. |
| More independent lineages improved learning curves; acquisition heuristics did not beat random acquisition | Prioritize genuinely new lineages. Fuzzed descendants are useful augmentation, not independent evidence. |

The optimizer recorded with a B is **binary PSO**, named `bpso` in configurations such as `R_lgb_fit_bpso`. The source reviewed does not establish that BOHB or BIPOP was the prior optimizer.

## What binary gates can and cannot preserve

Let `e_b` be the complete vector from encoder block `b`, with a stable feature order. Fit `(mu_b, scale_b)` on training lineages only and freeze them. Retain complete raw encodings and their provenance. Never overwrite S3 encodings with selected subsets.

For a binary feature/group mask `m_b` and a block gate `g_b`:

```text
z_b = clip((e_b - mu_b) / scale_b, -10, 10)
masked_b = where(m_b, z_b, 0)
h_b = g_b * BlockNorm(Project_b(masked_b))
context = Fuse(LV4_branch, h_KL, h_KS, h_ICL, h_Mitra,
               mask_descriptor, availability_descriptor)
```

Use `where`, not `0 * NaN`; reject malformed available encodings before masking. An intentionally absent block is explicitly unavailable and zero-filled. Gate **again after projection and normalization** so biases cannot leak a supposedly disabled branch. Separate unavailable inputs from intentionally suppressed inputs in the descriptor.

All block dimensions, feature IDs, projection slots and output dimensions remain fixed. Mask replacement changes a versioned configuration and buffers, not the parameter tensor shapes or parameter identities. Existing optimizer states stay attached to the same parameters. Do not physically slice columns or reconstruct the first linear layer when a mask changes.

That guarantees structural compatibility, **not unchanged predictions**. A network trained only with all inputs present can fail when useful inputs disappear. It must be trained on the proposed mask distribution. New encoder slots cannot be added invisibly to the current 2,051-input Kumo-S-only model: that is a new architecture/checkpoint lineage, with separate exposure counters and evaluation. Loading an old checkpoint with `strict=False` and calling it the same run is not acceptable.

Use fixed training normalization before gates, then per-branch LayerNorm/RMSNorm with the post-normalization gate. Avoid changing BatchNorm population statistics as the search changes masks. Do not apply inverted-dropout scaling by `1/p` to arbitrary deterministic deployment masks. Train the exact 0/1 convention used at inference.

## Inputs and roster

1. **Always-present anchor:** LV4 statistics and learnability scores, all computed strictly from the registered inner-training/CV closure. No outer-test targets, winner fitness, mask-search outcomes or validation role IDs enter these features. Z query-fit scalars require their encoder/input provenance and completed leakage canaries.
2. **Primary encoders:** Kumo-L and Kumo-S. Use the complete cached vectors, not only their previously selected PCs or top dimensions. Verify dimensions and feature order from manifests, not names.
3. **Optional challenger branches:** HF-pinned TabICLv2 and Mitra-v2, evaluated incrementally. Availability is not an instruction to encode every new variant with them.
4. **Default off:** TabDPT until its pretraining-overlap exclusions and no-TabDPT twin are enforced; Dope and Hyperion-v6all are ablations rather than default inputs given their observed lack of lift and exposure caveats.
5. **Excluded:** foundation, PriorLabs/tabpfn35, target, context2048, TA-contaminated blocks and unresolved Hyperion-v7 provenance. A binary mask does not make a disallowed source admissible: loaders reject it before fitting normalization, search or training.

License labels in the master summary are the current project's recorded disposition, not a new legal determination. Product admission still binds exact code/weight/provenance pins.

## Search space and training recipe

Start with global gates; defer a dataset-conditioned routing network until a global gate has demonstrated a benefit. Conditional gates multiply search capacity and could simply memorize dataset families.

Use a hierarchy:

- Level 1: whole encoder blocks, keeping LV4 enabled.
- Level 2: eight fixed groups within each primary encoder, then optional branch groups. Group definitions and order are frozen with the encoder version. Use a training-only unsupervised correlation grouping if qualified; contiguous groups are only an engineering baseline, not semantic coordinates.
- Level 3: individual-coordinate refinement only if Level 2 beats matched random masks out of sample. Correlated embedding coordinates and rotation dependence make raw coordinate selection unstable.

A raw gate precedes the projector, satisfying the proposed switch layer. Optional learned projected-channel gates are a separate ablation and must not be described as raw-feature selection.

Warm up the new model with a preregistered mixture: 25% full approved roster, 25% anchor + one primary encoder, and 50% sampled legal group masks. Balance encoder participation across an update window and log it. Include the deployment candidates in training support. Validation inference disables stochastic sampling and evaluates a pinned deterministic mask. Full and masked predictions can be consistency-regularized on **training inputs only**, with the full-input prediction detached; keep this coefficient fixed initially so it does not become another large search.

Checkpoint the model/AdamW or Lion state, preprocessing, mask sampler RNG, mask-schema SHA, active-mask SHA, per-group usage, corpus/split authority and global optimizer cursor. Choose the optimizer from a matched baseline; the historical Lion result is evidence for that specific HWM head, not a reason to replace the GP policy's AdamW without comparison.

Search works on an immutable model snapshot. The asynchronous CPU search must never patch the active GPU trainer's tensors. Publish candidate mask files with `model_checkpoint_sha256`, `normalization_sha256`, `feature_schema_sha256`, `corpus_snapshot_sha256`, `search_split_sha256`, budget and all evaluation receipts. After evaluation, the trainer may adopt a compatible mask **only at a logged checkpoint boundary**, with rollback to the previous mask. A masked-input checksum or a changed model checkpoint invalidates cached search scores.

## Small final experiment matrix

Run on one common eligible corpus and the same lineage folds, candidate budgets and scoring identities. These are proposed tests, not already running results.

| Arm | Purpose |
|---|---|
| A: LV4 LightGBM | Frozen outcome-predictor benchmark; also a cheap GP-value router benchmark. |
| B: matched LV4 + Kumo-S network, no searched mask | Neural control based on the strongest current DL input family. |
| C: LV4 + full Kumo-L + Kumo-S, no searched mask | Whether adding the strongest encoder helps this exact downstream network. |
| D: same C, mask-conditioned training + equal-budget random search | Is mask robustness/search itself useful? |
| E: same mask-trained weights as D + binary PSO | Primary test: does informed discrete search beat random search and C? |
| F: same input schema, training-only hard-concrete gates | Secondary alternative: differentiable sparse gating. No separate million-trial search. |

Keep parameter budgets matched where possible; report exact parameter counts and extra encoder/runtime cost when not. TabICL/Mitra become a second-stage expansion only if their cheap incremental test survives a frozen outer comparison. Do not choose all expansion variants after observing the test set.

For each outer training partition, use three inner lineage folds. Start with 512 distinct masks per search method, with a population of 32 for binary PSO and at most 16 evaluated generations including initialization. Deduplicate actual bitsets. Use the same eligible training-family subsets, random seeds and stopping budget for D and E. Count failed or duplicate proposals explicitly; they do not buy extra unseen validation queries.

Use three evaluation levels:

1. Cheap inner-CV ridge/LightGBM screen over cached features. Preserve some randomly chosen candidates rather than screening exclusively by the proxy; encoder interactions can change under the neural head.
2. Batched forward evaluation on the frozen, mask-trained **actual network**. Compare proxy ranking with this ranking. Stop trusting the cheap proxy if it fails to rank candidates usefully.
3. Refit the selected mask and the ungated control from the same initialization/budget on each outer-training partition. Evaluate once on its outer partition. Shared-weight search estimates do not replace this refit.

Only the top fixed number (initially 8) advances from level 1; keep 4 independently sampled masks as proxy-bias controls. Select the final candidate using inner evidence only. Primary E-vs-C and E-vs-D comparisons are preregistered; other results are secondary. Use paired lineage-bootstrap intervals and report all tested arms, not just the largest winning delta. A current 5% development set can monitor learning, but repeated mask search makes it selection data. It is not a fresh lockbox. The spent historical lockbox stays spent; final independent confirmation needs genuinely new source lineages.

Promotion requires a positive lower confidence bound on the preregistered executed-program utility improvement versus the ungated policy, with no meaningful increase in failure/timeout rate or evaluation cost at the matched budget. For outcome prediction, report R² and beat10 AUC separately and define noninferiority tolerances before opening outer results. A CI covering zero is inconclusive. If no mask passes, keep the ungated model and stop the search.

## Loss and score normalization

For the program generator, use measured same-context comparisons as the central signal. Preserve token CE on qualified teacher programs and within-context pairwise ranking; mask unsupported objectives per record. A context with only one measured program supports teacher CE if tokenization/identity qualify, but cannot manufacture a pairwise comparison. Failed attempts support feasibility/cost targets, not fabricated numeric quality.

For a new baseline-normalized objective, the candidate is:

```text
q(context, program) = mean_over_inner_folds(
    log((baseline_rmse + epsilon_context) /
        (program_rmse + epsilon_context)))
```

Higher is better; zero is parity. Use the same rows, folds, target units and evaluator. Fit the baseline separately inside each inner split. A fixed CPU CatBoost baseline is worth testing because it controls for dataset difficulty better than target scale alone. Keep a fast mean/ridge baseline as a separately identified reference; do not silently substitute denominators between contexts. Determine CatBoost's bounded recipe from training-only timing/accuracy evidence before freezing it.

Use a training-target-scale-dependent epsilon, cap extreme log ratios under a recorded objective version, and retain uncapped ratios and raw RMSE. Near-perfect baselines need explicit floor handling. Pairwise differences within a context already cancel a shared log baseline, so CatBoost mainly changes the quality-regression scale and cross-context weighting; it is not magic extra ranking information. First compare whether its improvement justifies baseline cost.

For the outcome predictor, keep separate fitness and beat10 heads because the empirical best input/regularization differs. For the GP policy, judge masks by **executed program utility at a fixed proposal budget**, with frozen same-context random and best-known opponents. An improved predictor R², teacher loss or mask-search score is secondary evidence.

## HPC and data allocation

The measured flywheel bottleneck is serial orchestration/publication, not currently GPU arithmetic: first-run decode/measurement waits totaled about 202 seconds versus 3.53 seconds for the eight final-measurement-to-update intervals. These are wall-stage measurements, not a CUDA profiler. Native GP evaluation was about 1.17 seconds; publication plus exact-version readback took roughly 7.3–8.2 seconds. Increase bounded concurrent independent queues and batch cached GPU inference before adding more trainer replicas.

- xbabe1: Grok's two-head experiment already occupies its GPU. Coordinate its terminal handoff; do not interrupt it or count its steps as this policy's steps.
- xbabe2: one model coordinator/GPU trainer, publication/admission control and parallel native GP workers as measured memory permits.
- xbabe3: required encodings and independent GP workers; evaluate frozen-mask batches when the GPU is free.
- CPU mask evaluation uses one BLAS/OpenMP thread per worker initially, shared read-only feature matrices, disjoint leases and a bounded queue. Increase workers by measured throughput/RSS, with live reserves for foreign jobs and active GPU input pipelines. “128 CPUs” is a ceiling, not a reason to allocate 128 memory-heavy Python workers.

Batch cached masks and contexts on the GPU. Avoid repeated S3 downloads and duplicate compilations; compile caches bind source/emitter/compiler/flags/sysroot, while every new context still gets native execution and Rust parity. Exact publication checks remain mandatory and are overlapped across independent jobs. Every proposed GP gets a durable outbox entry before the optimizer advances; completed scores enter the next frozen training snapshot, not an unmeasured reward.

The fresh metadata audit found 14,987 original data runs across 3,003 dataset hashes; 12,581 are currently eligible. Of the eligible published finals, 7,120 have a winner and 4,581 are terminals; 880 final records await publication. Terminal does not prove that its detailed attempt archive has no usable measured programs. The all-program join is being audited. The 9,899,594 completed-attempt counter is **not** a count of distinct GP programs or admitted training examples.

Fuzzing toward 100,000 data runs should preserve approximately 5% validation assignment at the **source-lineage** level. New splits and siblings inherit their parent's role. Weight original lineages equally and sample at most one member of a lineage per batch. Report independent lineages, dataset versions, unique splits, unique examples seen and repeated presentations separately. More folds/variants cannot be presented as 100,000 independent datasets.

## Worker-ready implementation and ownership

Read first, in order:

1. `/home/ubuntu/JopeDime/AGENTS.md`, current locked `AGENT_CHAT.md`, open PRs/worktrees and the shared `DL_TRAINING_DASHBOARD_HANDOFF.md`.
2. This plan, the pinned master summary and actual ENCSEARCH nested result file.
3. Current role/exclusion/target receipts listed in `DL_FLYWHEEL_FULL_CORPUS_INVENTORY.json`.
4. Existing `deploy/dl-trainonly/lane/src/dl_training_v1/dltrain/{model,objective,sampler,labels,encodings}.py` and source-pinned current flywheel code.
5. The existing predictor implementation and fold/preprocessing contract in `deploy/dl-trainonly/research/dlhwm/`; locate via `git ls-tree` on the claimed source head rather than guessing a trainer filename.

Proposed new source scope is `deploy/dl-flywheel/jope_flywheel/{encoder_schema,encoder_gates,mask_search,mask_evaluation}.py`, tests alongside the package, and `deploy/dl-flywheel/docs/encoder-mask-plan.md`. Existing model, trainer and corpus source belongs to concurrent owners: use explicit integration interfaces and a stopped-head handoff for overlapping writes. The live neural qualification, its checkpoints, native worker binaries, S3 versions, frozen corpus receipts and foreign worktrees are read-only. Runtime output stays outside Git.

Ordered implementation:

1. `encoder_schema.py`: immutable ordered block schema, source/weight/license pins, dimensions, role closure, availability and training-only normalizer receipts. Reject unknown/restricted blocks before fitting.
2. `encoder_gates.py`: fixed-shape binary gate module, post-projection gate, explicit mask/availability conditioning, stable parameter IDs and checkpointable mask RNG. Unit tests: disabled values cannot influence output; unavailable is distinct; all-zero optional blocks valid; LV4 cannot be disabled; malformed masks/schema mismatch rejected; validation changes cannot change normalization; optimizer state survives a mask swap.
3. `mask_search.py`: mask IDs, deterministic BPSO and matched random schedule, per-mask cost/failure records, exact deduplication, full checkpoint/corpus/split/evaluator cache identity, fixed search budget and persistent RNG. No active-model mutation.
4. `mask_evaluation.py`: lineage-nested folds, proxy/actual-network stages, fresh final refit, equal-context/lineage weighting and a complete exclusion ledger. Tests must catch same-lineage train/validation overlap, search access to outer labels, wrong snapshot cache reuse and silent omission of failed masks.
5. Separate worker manifests/cgroups per host, benchmark a representative packet, then increase concurrency. Record GP/s, masks/s, GPU active time, CPU utilization, peak RSS, queue depth, publication latency and failure counts.
6. Dashboard: current model/corpus/mask version; available vs active encoder blocks; mask sparsity/group usage; proxy score vs nested actual score; search budget; committed mask changes; measured GP win-rate and uncertainty. Never plot search-best as validation performance.
7. Independent review, source tests, exact-head CI and a new PR. Live launch has its own source/config/input/resource receipt. No automatic production promotion from passing unit tests.

Validation commands after integration, using the qualified environment and repository root:

```sh
rtk proxy env PYTHONPATH=deploy/dl-flywheel python3 -m unittest discover -s deploy/dl-flywheel/tests -v
rtk proxy python3 -m compileall -q deploy/dl-flywheel
rtk git diff --check
```

For the existing native PolicyLM integration, set `JOPE_POLICYLM_SOURCE` to the explicit checked-in `deploy/dl-trainonly/lane/src/dl_training_v1` parent and include it on `PYTHONPATH`. Missing Torch/native sources are a failed required capability, not a passing skip. Test exact actual GPU shapes and a small real-data train/validation packet before launch; synthetic tests do not establish encoder provenance or generalization.

Completion requires the exact full-corpus join, a reproducible mask-compatible model, every preregistered arm's results, independent review and an honest promotion/no-promotion decision. Stop the affected operation on source/role drift, leakage, unexpected encoder, missing objective support, memory admission failure, unresolved external outcome, deadline expiry or checkpoint incompatibility. Preserve receipts; do not repair counters by hand.

Parallel packets are disjoint: corpus/admission adapter; neural/outbox worker; HPC GP dispatcher; encoder-gate/search design and dashboard. Changes to common `jope_flywheel` modules are integrated by one claimed source owner. Every handoff records source/config hashes, owned paths/units, commands with exit status, actual measured output, failures and the next exact step. Update the shared handoff under its append lock.

## Research references

- [Louizos et al., Learning Sparse Neural Networks through L0 Regularization](https://arxiv.org/abs/1712.01312): hard-concrete stochastic gates provide a differentiable sparsity alternative. This motivates arm F, not a claim that it wins here.
- [Hamano et al., CMA-ES with Margin](https://arxiv.org/abs/2205.13482): ordinary continuous CMA-ES can stagnate under naive discretization; the margin variant addresses discrete-variable sampling. Prefer BPSO for this initial pure-binary experiment.
- [Chu et al., FairNAS](https://openaccess.thecvf.com/content/ICCV2021/papers/Chu_FairNAS_Rethinking_Evaluation_Fairness_of_Weight_Sharing_Neural_Architecture_Search_ICCV_2021_paper.pdf): unequal training of shared submodels can bias their rankings. Balanced mask exposure and final independent refits are therefore part of this design.
- [Li et al., Hyperband](https://www.jmlr.org/beta/papers/v18/16-558.html): allocate computation across candidates at multiple budgets. Our three evaluation levels are an engineering adaptation; they are not a demonstrated Hyperband result on JopeDime.
- [PyTorch Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html): use measured bottlenecks, batching and careful thread allocation. Changes must remain compatible with the installed, pinned runtime.
