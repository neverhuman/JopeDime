# The policy model, proposals and bounded search

## Accepted regression model

### Checkpoint binding and configuration

The accepted lineage is the recovered lineage-B step-60000 checkpoint. The binding verifies the state, original model, exported tensors, configuration, parameter digest and vocabulary. The rejected step-69000 identity is explicitly refused. A self-consistent receipt around unrelated weights cannot impersonate the accepted model. The model loader rejects missing, extra, incorrectly shaped or non-FP32 tensors.

| Accepted-model property | Value or behavior |
|---|---|
| Dataset context (historical name of `dataset_dim`) | 1536 FP32 values: Hyperion 512 from features only plus Foundation 1024 conditioned on the training targets, so not label-free |
| Target context | 512 FP32 values; all 512 retained by the configuration (`target_keep_dim` 512), although most of them may be constant zero (`python/src/jope/policy/train_v3.py:919-920`) |
| Combined context | 2048 FP32 values; 8192 bytes in little-endian vector form |
| Decoder width | 512 |
| Decoder depth | 6 layers |
| Attention heads | 4 |
| Feed-forward width | 2048 |
| Token vocabulary | 401 output tokens |
| Embedding rows | 657, including the configuration's 256 extra rows |
| Maximum configured sequence | 128 tokens; positional table includes the context position |
| Training dropout field | 0.1; disabled for inference |
| Reward ensemble | 4 heads |
| Tensor inventory | 136 tensors for this configuration, including unused inference heads and scalar loss weights |
| Current Rust arithmetic | Candle CPU FP32 |
| Native CUDA BF16 policy parity | Not established by the current CPU inference implementation |

Sources: `crates/jope-policy/src/config.rs:21`, `config.rs:39`, `crates/jope-policy/src/model.rs:1`, `model.rs:41`, `crates/jope-policy/src/lib.rs:133`.

### Decoder and reward architecture

The decoder and reward paths each encode context with a two-layer MLP: 2048→512, GELU, LayerNorm, 512→512, GELU, LayerNorm. The decoder prepends a projected context token to the token embeddings, adds positional embeddings, applies causal self-attention with padding-key masking through pre-normalized layers, normalizes the result and projects through the tied token embedding table narrowed to the output vocabulary.

The reward path pools program-token embeddings, combines them with a separately encoded context, and evaluates four reward MLPs. The combined representation contains the program embedding, its elementwise interaction with a context gate, and an absolute-difference block. A separate runtime head emits a positive runtime estimate through softplus. Learned reward is a proposal-ranking signal; it is not a measured dataset loss.

### Accepted identities and the product restriction

The source records these accepted identities:

| Identity | SHA256 |
|---|---|
| State | `8ac80c269cc47c253b6f53fa3bc1eb75bb2fcda0bdace3ff8554be0175c400d6` |
| Original model | `053df4f53b84d2ebe7c14fbf907a149760af3d11de4a57b5ff89be8c2d3e8129` |
| Canonical parameter values | `9c7057e7f6a9e348f030da73f03522977b8c08915d3a221d7d5ab2e4213fcd92` |
| Configuration | `2508fc8b8268ea000141c9b119ba3da4e777c506c366168912a88d03482939bf` |
| Original checkpoint vocabulary | `8d55fee4366acd2c175cf55bfd769cebc904770f5227a6fd50a613723dc85787` |
| Executable product restriction | `570b2e5e8d859a73b888f5d0bee27f59da0e2376536babce24c241604f818e81` |

The product restriction preserves the 401-token table while limiting executable heads. It removes ten head operators (`extra_trees`, `gam_boost`, `logit_ridge`, the four multiclass heads, `portfolio`, `ridge_gcv` and `tab_boost`), and the vocabulary has no operator token for `pls` or `hurdle2`. The regression proof bound both arms to this restricted vocabulary, and mutation replaces a head only with `histgbm`, `ridge`, `ridge_gcv`, `elasticnet`, `histgbm2`, `gpr` or `ksmooth`; crossover keeps a parent's head. Six of the seventeen registered regression recipes (`tab_boost`, `extra_trees`, `gam_boost`, `pls`, `hurdle2` and `portfolio`) are therefore unreachable by any ASM operator at this source, and `ridge_gcv` is reachable only through mutation. Sources: `crates/jope-engine/assets/grammar/featmath-jain-v1.json` (`restriction.removed_ops`); `crates/feat-math/src/evolve.rs:29-39`, `:1480-1492`; `crates/jope-bench/src/panel/optimize.rs:254`; `crates/jope-bench/src/panel/methods.rs:60-89`. A token-compatible restriction is different from inventing a new vocabulary or silently snapping parameters to nearby bins.

## Policy producer contract

`jope-policy-producer` is a separate pinned bounded job. Its `asm.private-policy-producer/v1` request binds worker source, prepared/training/context hashes, cut, accepted checkpoint state, policy/vocabulary hashes, three RNG seeds and sampling shape. It accepts only the pinned step-60000 weights (`4b4b5a40…44f1e`) and state, rounds 1–8, batch 1–128 and a finite positive temperature. Context input is exactly 8,192 bytes, finite and clipped within [-32,32]. It requires either a qualified CPU context bundle or an authenticated original `parallel.encoding.v1` receipt with the original LibTorch lineage, train-side partition and `final_test_targets_used:false`. Frozen transforms are pinned to `1b779c6f…6e22`.

Its output must be an absolute physical-parent path outside `/tmp`, created new with mode 0600. The final stdout JSON declares `numerical_backend:candle_cpu_float32` and `native_cuda_parity_qualified:false`. Reading the envelope proves consistency, not execution: consumption also requires the completed pinned worker/output receipts. Replacing accepted weights, transforms or vocabulary is not a free configuration substitution.

The producer shares the 30-second worker cap and original 120-second neural allowance; the accepted proof used a 3 GiB worker allocation. The pre-cache proof at source `55e9ed688faf` was killed at 30.004805335 seconds, before neural fits. With cached decoding, later measured producer wall times were 20.832133058 seconds at source `3fd7d5b2ebb6`, 20.651920435 at `c5d11ffca92c` and 20.486968041 at `68f56404c3e1`. These leave about 9.5 seconds of per-job headroom in the last measurement, not a portability guarantee. Sampling-shape or host changes need another bounded proof, and the fixture-only decoder-equivalence limits of [§](06-encodings-and-context.md#component-dimensions-and-assembly) still apply.

Sources: `crates/jope-policy/src/context_producer/worker.rs:28`, `:188`, `:255`; `crates/jope-policy/src/producer.rs:16`, `:91`; `crates/jope-policy/src/bin/jope-policy-producer.rs:13`, `:78`, `:121`, `:172`; `crates/jope-bench/src/panel/optimize.rs:454`, `:705`; retained `proposal-execution.json` receipts.

## Proposal generation and ranking

Accepted proposals are generated under grammar constraints, and the neural pool is the first 64 distinct valid programs in draw order; any shortfall after the bounded sampling rounds is filled with distinct grammar-uniform programs labelled `neural_rejection_fallback`. Ranking orders that pool by the mean of four reward heads, descending; exact ties use program SHA256 then text. Runtime estimates are reported without subtracting an undocumented runtime penalty. Ranking binds the checkpoint, context and vocabulary and records that scoring labels and fitted GP scores were not read. It does not select or reject programs: all 64 ranked programs and the 16 exploration programs enter generation zero. Sources: `crates/jope-policy/src/ranking.rs:22`; `crates/jope-policy/src/decode.rs:9-11`, `:212-225`, `:359`; `crates/jope-policy/src/seeds.rs:257-270`.

Each ranked policy-pool program carries `rank_log_weight`, the log-softmax of its four-head mean predicted reward over that pool. Historical files called this `policy_logprob`. It is a within-context ranking weight, not a sampling/selection probability or propensity; do not use it for importance weighting or off-policy correction. Exploration candidates carry no rank weight.

## Generation zero and bounded evolution on twenty workers

### Paired arms, generation zero and evolution

ASM compares a 64-slot policy pool plus 16 exploration generation-zero slots with 80 generic slots. The pool retains `neural_policy` versus `neural_rejection_fallback` origins explicitly. Both arms receive the same declared search and fit allowances, but encoding and proposal work consume the neural arm's own clock; fitting time can differ. A producer refusal, duplicate or failed candidate is an outcome to record. It cannot silently become another neural success or a free replacement arm. Actual inner-fold fits determine selection and offspring ancestry. A model's predicted reward never substitutes for measured loss.

Each arm starts with 80 generation-zero slots. In the neural arm, slots 0–63 form the ranked policy pool: distinct valid checkpoint samples are labelled `neural_policy`; any shortfall after the bounded sampling rounds is filled by distinct grammar-uniform programs labelled `neural_rejection_fallback`. Slots 64–79 are `random_exploration`. Only `neural_policy` roots and completed descendants with verified neural ancestry count as genuine neural evidence; fallback and exploration roots do not. In the origin proof at source `68f56404` all 64 pool slots were `neural_policy`, 48 completed all five inner folds, and no fallback slots were needed. Receipts retain initial-slot origin and descendant ancestry. Actual completed inner OOF losses select the best four parents. New generations alternate mutation and crossover attempts, canonicalize recipes, reject duplicate executable identities, and retain variation attempts in the journal. Source: `crates/jope-pipeline/src/neural_search.rs:27`, `:1380`.

### Arm order and clocks

The arm order is fixed: neural first, then generic. Fresh context encoding and policy production run inside the neural arm clock, and each arm is clipped to the original execution cutoff. In the origin proof at source `68f56404` the producer gate opened 21.094 seconds after the campaign start and its exit was observed at 41.581 seconds; its measured wall time was 20.486968041 seconds. The neural cutoff then lay 78.613 seconds ahead, before allowing for post-exit verification/setup, so this is remaining arm time rather than guaranteed fit execution time. The neural arm stopped at `original_deadline`. The generic arm began 0.553 seconds after that cutoff, retained its full declared 120-second allowance and stopped at `evaluation_limit`. Equal declared allowances are not equal fit-only time. Sources: `crates/jope-pipeline/src/neural_search.rs:654`, `:701`, `:729`, `:772`; `crates/jope-pipeline/src/execution.rs:592`; the accepted proposal execution and recovered report.

### Twenty bounded workers: resource limits and supervision

| Constraint | Official panel policy |
|---|---|
| Worker/CPU count | Twenty isolated workers shared by search and benchmarking; at most twenty CPUs. |
| Thread count | One thread per worker, including external libraries. |
| Memory and swap | 64 GiB total; zero swap. The shared policy reserves 4 GiB for the controller and caps each fit at 3 GiB. |
| GPU capacity | At most one GPU task; the approved training hardware is one RTX 3090. GPU training has its own separate total hour. |
| Fit and search time | At most 30 seconds per fit, normally 120 seconds per search arm. |
| Panel wall time | 7,200 seconds total, including 300 seconds reserved for reporting. No new fit/qualification starts after the original execution cutoff at 6,900 seconds. |
| Coexistence | Do not overlap the experiment with exclusive full-CPU CI. |

The CLI uses the shared Linux systemd/cgroup-v2 supervisor and process-group cleanup. Missing host capability is unavailable execution. Do not put an additional outer allocation lock around a command that already acquires the shared allocation. A child crash, timeout, cancellation, launch failure and unavailable dependency have different recorded states; unknown work/cost is not zero.

The default admitted panel allocation is twenty CPUs/workers, 64 GiB memory, zero swap, one RTX 3090 when GPU work is explicitly allocated, thirty seconds per fit and 120 seconds per search arm. The full panel receives two hours total, including a five-minute reporting reserve. Training receives one GPU-hour total and twenty minutes per category. Do not overlap numerical campaigns with exclusive full-CPU CI. Resume continues the original campaign deadline and resource envelope.

### Counting proposals, recipes, predictors and commands

A recipe may be evaluated on multiple folds or re-used in later fits. Therefore proposal count, recipe count, unique fitted predictors, launched commands and physical fit starts are different denominators. A cache hit does not become another new predictor; a partial/unknown attempt does not become a zero-cost fit. [§](18-measured-performance.md#cpu-wall-time-and-per-predictor-accounting) applies this accounting to the current proof.

Report elapsed wall time, summed worker wall time, worker CPU time, aggregate supervisor CPU, GPU time if measured, launched commands, successful fits, failures, unique fitted predictors and unique reusable recipes separately. Aggregate supervisor CPU contains worker CPU; adding them double-counts. Parallel worker durations do not equal elapsed wall time. Deduplicated/cache reuse is useful but is not another physical fitted predictor.
