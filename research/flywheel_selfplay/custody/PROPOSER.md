# Frozen h40b training proposer

This describes the proposer on the lineage of frozen r782, not the current in-repository implementation. [MANIFEST.json](MANIFEST.json) and [CHECKPOINT.json](CHECKPOINT.json) pin the custody and checkpoint. No proposer or weight behavior is changed here.

## Source and actual conditioning input

The frozen runner inserts `JopeDimeMicro/python` before importing `rust_evaluator`. The actual proposer is [Micro rust_evaluator.py](files/home/ubuntu/JopeDimeMicro/python/rust_evaluator.py), SHA256 `53422343699172b61ed69ed707015558f7cf1a344e5449f30e1eb9d708aef900`, functions `propose_flywheel_candidates`, `select_landmark_preferred_pools` and their string/parameter helpers. The upstream Micro revision was not recorded; the exact bytes are the source identity. Run manifests 662, 775 and 776 pin that same hash.

Caller semantics come from [frozen run_fused_closed_loop.py](files/home/ubuntu/jope_plan/evidence/RES-01/frozen-source/research/flywheel_selfplay/run_fused_closed_loop.py) and [frozen dev_incumbent.py](files/home/ubuntu/jope_plan/evidence/RES-01/frozen-source/research/flywheel_selfplay/dev_incumbent.py), revision `07e03984de9a1a7be5b0c94bba166423aa21a549`. Exact hashes and earlier manifest identities are in [proposer_frequencies.json](proposer_frequencies.json).

Despite the helper's `landmark_v2_48` name and gap-feature comments, the caller passes `lm[0]` returned by [cpu_exact_fused.py](files/home/ubuntu/jope-data/res-01/input/cpu_exact_fused.py) `_run_dope_mlc_v3_and_lm48(use_landmarks=True)`: `clamp((vec256 - step2_lm_mean_1x256) / step2_lm_std_1x256, -6, 6) @ step2_svd_basis_256x48`. It is the SVD projection, before the stage-1B lm48 normalization. The branch thresholds below act on those actual coordinates. Their comment names do not establish that they are raw landmark gap measurements. A port must preserve the actual input and thresholds.

## Bank, parents and generation order

The loop filters `champions.json` to training context IDs. Its 1,219 initial records contain context/family/split, DSL and program hash; the caller fills `feat412`, `proj16` and `landmark_summary = lm48[:12]` when `feat412` is absent. Existing populated records retain their supplied summaries. Replayed promotions are appended once per `(context_sha256, program_sha256)` with the current context summary; later promotions are appended in evaluation order. This proposer bank is separate from the model's fixed 885-entry prior bank.

For each GP context, parents are bank entries with that exact context. They rotate by `(round + context_loop_index) % parent_count`; no parents means offset 0. The full-cohort caller obtains 16 GP contexts through `training_order(round_seed, 16, True)`; the legacy path uses its training list. These context-order differences affect both rotation and RNG seeds.

A fresh `random.Random(seed)` is created per call. `seen_dsl` starts with parent DSL strings. `add_proposal` accepts only a previously unseen exact string beginning `(pipe X `, hashes its UTF-8 bytes, and records operator/parent IDs. It does not canonicalize whitespace or prove grammar legality. Within-call dedup is by exact DSL; the caller later removes hashes already in that context's pool. A parseable prefix is not a successful native evaluation.

Generation is ordered: family-excluded retrieval first, cyclic parent edits second, synthesis to fill remaining slots last. Default function argument is 16; the actual loop requests 8 normally or 16 during exploration.

### Retrieval

Only bank entries whose `dl_split == "DL-Train"` and whose `family_id` differs from the target family are eligible. The code uses `.get`; absent family metadata is not separately rejected by this helper. Actual initial and replayed caller records carry family IDs.

Distance is squared Euclidean distance between the first 12 target coordinates and the supplied 12-value `landmark_summary`, in float64, without an additional retrieval normalization. Entries sort by `(distance, program_sha256)`, retaining input order for exact ties. The first `min(4, n_proposals // 3)` entries are offered: at most 2 for 8, 4 for 16. Dedup can reduce the accepted count. Each candidate is `champion_retrieval` with the retrieved hash as its parent. If the mutation parent list was empty, the first retrieved entry becomes its sole parent. There is no random retrieval probability or per-family quota beyond exclusion.

### Typed-edit attempt schedule

If parents exist, attempt at most `3 * n_proposals` edits, stopping as soon as the accepted proposal count reaches the requested size. Parent index is `step_idx % len(parents)`; edit family is `step_idx % 5`:

| Residue | Operator | Exact branch |
|---|---|---|
| 0 | `edit_parameter` | Mutate numeric arguments of every stage and the head. |
| 1 | `edit_replace_head` | Choose uniformly from the preferred head list. |
| 2 | `edit_insert_stage` | Choose a preferred stage uniformly and an insertion position uniformly from 0 through current stage count; accept only if at most 6 stages result. |
| 3 | `edit_delete_stage` | Only with at least 2 stages: delete one uniformly chosen stage; otherwise this attempt produces nothing. |
| 4 | `edit_crossover` | Only with at least 2 parents: use the next cyclic parent; concatenate the first parent's first `max(1, stage_count // 2)` stages with the second parent's stages from `other_stage_count // 2` onward, truncate to 5, and take the second head. Both parent hashes are recorded. Otherwise this attempt produces nothing. |

This is a deterministic cycle, not five independent 20% choices. Complete five-attempt cycles allocate one attempt to each residue. At the full bounds 24 and 48, attempt counts would be `[5,5,5,5,4]` and `[10,10,10,9,9]`; reaching the proposal quota stops earlier. Branch ineligibility and exact-string dedup change the output mix. Missing parent hashes are recomputed from DSL.

`parse_pipe_dsl` strips whitespace, returns `([], "(ridge_gcv)")` for an invalid prefix or no complete top-level clauses, otherwise separates balanced parenthesized clauses into stages plus final head. It is a string parser, not full grammar validation. `assemble_pipe_dsl` also permits an empty stage list.

### Numeric mutation

The regex matches lowercase/underscore `key=numeric-value` arguments. `seed` is unchanged. Values containing a dot, or keys `q`, `lr`, `gamma`, `lambda_frac`, use multiplication by `exp(uniform(-0.30, 0.30))`, then clamp those four keys respectively to `[.01,.25]`, `[.015,.30]`, `[.05,8]`, `[.0001,1]`. Other matched floats are not clamped. Floats round to 8 decimal places, adding `.0` when the formatted value has neither a dot nor exponent.

Integer depth, degree and knots add a uniformly chosen ±1 and clamp respectively to 2–6, 2–7, 2–8. Rounds multiply by `uniform(.75,1.30)`, round to integer and clamp 40–280. Bins/topk/k/n multiply by `uniform(.75,1.35)`, round and clamp 2–128. Other integers remain unchanged. Parse/conversion exceptions retain the original argument. A parameter attempt may therefore reproduce an existing DSL and be rejected by dedup.

### Synthesis fallback

After edits, repeatedly draw 1–3 stages uniformly, each from the preferred stage list with replacement, mutate their arguments, choose/mutate one preferred head, and offer the assembled string as `landmark_synthesis` with no parents. Continue until the requested number of unique accepted strings exists, then return the first requested count. This path also handles no usable parents/retrieval. There is no explicit retry ceiling on this final loop.

## Landmark-preferred lists and conditional probabilities

Stage lists append in this order, using the helper's coordinate names:

| Condition | Appended group and exact templates |
|---|---|
| `[39] < -.01` (symlog) | heavy_tail: `(winsor_clip append=true q=0.05)`; `(winsor_clip append=true q=0.1)`; `(winsor_clip q=0.08)`; `(blom)`; `(sigmoid append=true)` |
| `[38] < -.01` (rank) | monotone_rank: `(rank)`; `(blom)`; `(quantile_bins append=true bins=16)`; `(quantile_bins append=true bins=32)`; `(target_bin_encode append=true bins=12)` |
| `[40] < -.01` or `[36] < -.02` (poly/nonlin) | nonlinear_poly: `(cheb append=true degree=3)`; `(cheb append=true degree=4)`; `(cheb degree=5)`; `(poly_interact append=true)`; `(hinge append=true knots=3)`; `(hinge append=true knots=5)` |
| `[37] < -.005` (interaction) | interaction: `(pair_prod append=true topk=8)`; `(pair_prod append=true topk=16)`; `(poly_interact append=true)`; `(oblique_hinge gamma=0.25 n=32 seed=1337)`; `(rff_sin append=true gamma=0.2 n=32 seed=2026)` |
| Always, last | projection_scale: `(standardize append=true)`; `(standardize append=false)`; `(robust_scale)`; `(row_stats append=true)`; `(rand_proj append=true k=32 seed=701)` |

For heads, if `[36] < -.03`, append tree_deep then tree_shallow when `[37] < -.01`, otherwise shallow then deep. If the nonlinearity test fails, append linear_regularized then tree_shallow. Append local_kernel when `[43] < -.02`; always append linear_regularized last.

| Head group | Exact templates |
|---|---|
| tree_deep | `(histgbm depth=6 lr=0.08 rounds=160)`; `(histgbm depth=5 lr=0.1 rounds=180)`; `(histgbm depth=6 lr=0.06 rounds=220)` |
| tree_shallow | `(histgbm depth=3 lr=0.06 rounds=150)`; `(histgbm depth=4 lr=0.08 rounds=140)`; `(histgbm depth=2 lr=0.05 rounds=180)` |
| linear_regularized | `(ridge_gcv)`; `(ridge lambda_frac=0.01)`; `(ridge lambda_frac=0.1)` |
| local_kernel | `(ksmooth gamma=4.0 k=32)`; `(ksmooth gamma=2.0 k=64)`; `(ksmooth gamma=4.0 k=16)` |

These are ordered multisets: duplicates remain and receive repeated probability mass. A template's conditional probability is its multiplicity divided by list length. For example, in the non-strong-nonlinearity branch without local kernels, the duplicate linear group has probability 2/3 and the shallow tree group 1/3; with local kernels these become1/2,1/4,1/4. With strong nonlinearity, deep/shallow/linear each receive 1/3, or each group plus local kernels 1/4. Tree ordering still changes deterministic seed-to-output behavior. Stage-group lengths are5/5/6/5/5; overlapping blom/poly_interact templates retain their multiplicities. Positions 41/42 and 44–47 are not threshold inputs here.

## Caller dedup, ranking and exploration

After generating proposals, remove program hashes already present in the target pool. The ranker conditions on that pool's first four stored programs and ranks `4 + novel_proposals`; it does not replace those four with a newly selected best four here. Ordinary admission takes the first 4 candidates sorted by `(float(cost), original_index)`.

`proposal_plan` seeds round `r` with `seed + r * 104729`; context call and admission use `round_seed + context_loop_index`. Exploration requests 16 instead of 8 proposals, still at most 4 native evaluations: take the best 2, then uniformly sample up to 2 distinct indices from the remainder. With at most 2 candidates, return the ranked prefix. This is 2 ranked + 2 random admission, not 16 evaluations.

Exploration activates after 3 consecutive rejected adapter rounds (the helper also tracks identical candidate-dev signatures), or a GP promotion drought. A drought requires 18 adjacent completed prior rounds with `promotions == 0` and `GP.evaluated > 0`; missing/unknown rounds break the streak. Its installed-policy clock advances only for completed rounds carrying a non-null phase of `promotion-drought-burst/v1`: 3 burst rounds then 3 cooldown rounds. Adapter rejection can independently keep exploration active during a GP cooldown. Earlier historical rounds without that policy do not advance its burst clock. The helper also bounds rejected-adapter retries (level at most 3, learning-rate factor `.25**level`, at most 1 optimizer step after rejection); these do not change the maximum 4 GP evaluations.

Selected proposals, including native failures, are written to `round-NN-scores.jsonl`; successful promotions also enter `promotions.jsonl` and the bank. The latter is a selected subset and was not used as the frequency denominator.

## Measured evaluated-proposal frequencies

The measurement includes every score-log record from rounds 767–774 in the legacy run and 775–782 in the full-cohort run: 16 complete rounds, 64 evaluated proposals each, 1,024 attempts total. Every row was joined in order to the round's selected program IDs, its DSL hash checked, and its context matched to the frozen training roster. All 1,024 recorded statuses are completed; no gate/promotion/success filter or global-program dedup was applied. This uses metadata/DSL hashes only, no held-out rows or new evaluations.

Those rounds recorded 3,712 generated proposals, of which 1,024 were selected for evaluation. Full operator labels for the other 2,688 are not retained in these score journals. The following rates are conditional on evaluation/admission, not the unconditional generator mix. They measure the frozen lineage before or at r782; later rounds are excluded.

| Operator family | Count /1,024 | Frequency | Marginal Wilson 95% interval |
|---|---:|---:|---:|
| `champion_retrieval` | 252 | 24.609% | 22.069–27.339% |
| `edit_parameter` | 238 | 23.242% | 20.758–25.926% |
| `edit_replace_head` | 175 | 17.090% | 14.908–19.517% |
| `edit_insert_stage` | 187 | 18.262% | 16.015–20.745% |
| `edit_delete_stage` | 130 | 12.695% | 10.795–14.875% |
| `edit_crossover` | 42 | 4.102% | 3.049–5.498% |
| `landmark_synthesis` | 0 | 0.000% | 0.000–0.374% |

The JSON retains all 16 source journal/round-receipt hashes, per-round counts, separate 512-record protocol strata, run-manifest source pins and the interval formula. Repeated contexts/programs and adaptive round selection violate independent binomial-trial assumptions: these Wilson intervals are descriptive, not calibrated bounds on the unobserved generated mix. Zero observed synthesis is not proof of an impossible branch. M3 comparisons must use a matched recorded population and protocol; this report does not close Rust proposer parity or the learned-guidance quality gate.
