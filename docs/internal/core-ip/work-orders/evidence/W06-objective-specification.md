# W6-T02 objective specification by term

**Scope.** This covers the specification half of W6-T02 for the native lineage-B trainer. It is read-only source and evidence tracing. No fixture was run, no Cargo command was run, and no model or corpus payload was loaded. Status under EXECUTION-CONTRACT: **planned / specification only**. It is not `fixture-proven`. The W6-T02 acceptance still requires independently computed tiny fixtures that reproduce each scalar loss and its intended gradients.

**Source observed.** JopeDime worktree `.agent/worktrees/bench` at `a748a496982ef907aa060444e6462e73ea7dbdb9`. None of the files cited below differ from `0e2ca312b31afd1221b9948ab277160ea2d23929`. Anchors are `path:line` at that head; unqualified `training.rs`/`neural.rs`/`corpus.rs` are under `crates/jope-engine/src/`.

**Evidence labels**
- **SRC**: read from the current source.
- **EVID**: read from a retained evidence file in this repository.
- **SNAP**: read from the 2026-09-11 native-v3 source snapshot, `docs/legacy/root-docs/audits/worldclass-redteam-20260911/evidence/learning-native-v3/{neural.rs,training.rs}`, SHA-256 `0ad270f6…` and `4bd74f36…`.
- **UNVERIFIED**: not established.

---

## 1. The objective actually executed (lineage B)

For each rank-local microbatch `M` of `B` rows, `training.rs:1298-1380` computes:

```
L_mb = L_seq + 0.25·L_BT + 0.10·L_list + 0.10·L_rt + 0.05·L_abt
       + 0.0·Σ_i stability_i                      (training.rs:1365)
       + Σ_{n∈7 names} 0.0·loss_log_vars.n        (training.rs:1366-1369)
```

Each committed optimizer update uses `W` ranks and `A = global_batch/(W·micro)` accumulation microbatches (`training.rs:1381`):

```
g   = (1/W) Σ_rank Σ_{mb=1..A} ∇(L_mb / A)          (training.rs:1441,1447; neural.rs:1063-1072)
g'  = g / max(1, ‖g‖₂)                               (training.rs:1480-1483)
θ  ← FusedAdamW(θ, g', η=1e-4, λ=0.01, β=(0.9,0.999), ε=1e-8)   (training.rs:1062,1491; native/runtime.cpp:65)
```

- **Coefficients are fixed literals.** No learned or uncertainty weighting runs in lineage B. No native training path computes `exp(−s)·L + s` or any other function of `loss_log_vars` except `×0.0`. The one other use is the non-training probe (§3.9).
- **The normalization is per microbatch, then averaged.** Its identity string is `"microbatch_weighted_means_rank_mean_v1"` (`task_contract.rs:22-40`). The objective is therefore partition-dependent: the same global batch split differently is a different objective (`task_contract.rs:22-23`; `docs/legacy/.../learning-hpc.md:33`).

### Mapping between the seven `loss_log_vars` names, code and telemetry

| `loss_log_vars` name (`crates/jope-policy/src/model.rs:30-38`; `neural.rs:293-303`) | Term in code | Telemetry key (`training.rs:1525,1529`) | Coefficient | Active |
|---|---|---|---|---|
| `seq_nll` | `sequence_nll` | `sequence_nll` | 1.0 | yes |
| `reward_bt` | `ranking(...).0` | `ranking_bt` | 0.25 | yes |
| `reward_listwise` | `ranking(...).1` | `ranking_listwise` | 0.10 | yes |
| `runtime_huber` | `smooth_l1_loss` | `runtime_huber` | 0.10 | yes |
| `ablation_bt` | `ranking_pairwise(ablation)` | `contextfree_bt` | 0.05 | yes |
| `stability_bce` | `stability.sum()*0.0` | listed under `"inactive_objectives":["stability",…]` | 0.0 | no: no labels, no BCE |
| `routing_ce` | none | none | none | no: no routing module exists in `Policy` (`neural.rs:266-284`) |

---

## 2. Shared batch construction, masks and labels

### 2.1 Sampling and comparison groups
- **Sampler** (`training.rs:55-86`). Every 8 consecutive rows form one comparison drawn from one corpus group, which is one context:
  - Slot 0 is `group.records[0]`, the lowest cut-normalized RMSE (the champion or teacher).
  - Slots 1–7 draw one record from each of 7 disjoint quantile bands over `records[1..]`.
  - Families get equal mass per cycle. The sampler asserts at least 8 records per group (`training.rs:71-74`), and the corpus drops groups with fewer than 8 (`corpus.rs:880-883`).
  - The 8 rows are distinct records.
- **Replay variant** (`training.rs:88-144`). In `fresh_balanced` mode, even-numbered comparisons draw from fresh groups. If no fresh program was drawn, the band slot with the nearest record id is replaced with a fresh id from the same group; the champion is never replaced (`training.rs:131-141`).
  - Id distance stands in for rank distance. That relies on group record ids being contiguous and sorted, which is **UNVERIFIED for the sharded corpus**.
- **Microbatch alignment.** Allowed microbatches are 64, 128 or 256 and the global batch is 2048 (`training.rs:778-791`). The sample `serial` is always a multiple of 8, so a comparison never spans microbatches or ranks.
- **Group key.** `format!("{context}:{i/8}")`, where `i` is the row index in the microbatch (`training.rs:1346-1350`). Two comparisons of the same context in one microbatch stay separate groups. There are no cross-comparison pairs.

### 2.2 Tokens, shift, PAD, BOS/EOS, legality
- **Encoding.** `[BOS=1, engine, task, program tokens…, EOS=2]` (`grammar.rs:420-424, 476`). `record.length` counts BOS and EOS. Token 0 is PAD; record arrays are zero-filled after `length` (`corpus.rs:899-911`). Reserved ids 0–2 are also implied by `neural.rs:1248-1252`.
- **Microbatch width.** `T = min(8·⌈max_i length_i / 8⌉, 128)` (`training.rs:1300-1301`). A row's padded width depends on the other rows in its microbatch.
- **Present mask.** `pad = tokens.ne(0)` (`training.rs:1309`). Despite the name, it is true for non-PAD tokens.
- **Decoder input.** `[ctx_proj(z), token_emb(x_0..x_{T−1})] + pos_emb(0..T)` (`neural.rs:427-436`).
  - Attention is causal AND key-present. The context slot is always present, so no query row is fully masked (`neural.rs:437-448`).
  - Logits use the tied `token_emb` projection narrowed to `vocab_size` (`neural.rs:452-456`).
- **Target shift.** `logits.narrow(1,1,T−1)` is paired with `tokens.narrow(1,1,T−1)` and `legal_ids.narrow(1,1,T−1)` (`training.rs:1318-1328, 1336-1342`). Decoder position `j`, which has seen the context and `x_0..x_{j−1}`, predicts `x_j` for `j = 1..T−1`.
  - **BOS (`x_0`) is never a target.** The context-slot output that would predict it is dropped, and so is the final position.
  - **EOS is a supervised target.**
- **Supervised positions.** `j ∈ [1, length_i − 1]`. The mask is `m_ij = pad[i, j]`.
- **Legal sets** (`corpus.rs:908-926`; `teaching_store.rs:206-217`; `continuation_corpus.rs:449-468`).
  - `legal[j]` is the grammar state's legal set before consuming `x_j`. The state never consumes `x_0` (`j==0 → continue`).
  - Legal set id 0 is `{0}`. It fills BOS and PAD positions, so every PAD target (0) is legal.
  - Sets are built with `state.sampling = false`, which includes operators with `generate=false` that sampling-time legality excludes (`grammar.rs:142, 150`).
  - Singleton legal sets include at least `Phase::Eos → [2]` and `Phase::ExprEnd` (`grammar.rs:170-171`).

### 2.3 Per-record labels (`corpus::Record`, `corpus.rs:138-146`)
- **`value`**: `v = 1/(1 + RMSE/median_group RMSE)`, using the upper median `rmses[n/2]` clamped to ≥1e-8 (`corpus.rs:887, 904`). This is identical in `teaching_store.rs:163-177` and `continuation_corpus.rs:431, 445`. Higher is better; `v` is in (0, 1].
- **`percentile`**: depends on the corpus label policy.
  - `reference_v1`: `p = 1 − rank/(n−1)`, with RMSE ties broken by program string (`corpus.rs:886, 898`; `teaching_store.rs:174-182`).
  - Midrank policies: tied RMSEs share `p` (`teaching_store.rs:84-98, 167-173`; `continuation_corpus.rs:430`).
  - `frozen_z_v1`: previously admitted programs keep their old `v, p` (`teaching_store.rs:59-82`).
  - `p` is a rank over the whole corpus group, not within the 8-row comparison.
- **`runtime`**: `y = ln(1 + seconds)` stored as f32 (`corpus.rs:906`; `teaching_store.rs:198`).
  - Seconds come from `numerical_runtime_seconds`, falling back to `runtime_seconds` (`corpus.rs:561-566`), via `scan_ledger`, which the shard store also uses (`teaching_store.rs:695`).
  - The continuation corpus uses `fit_runtime_seconds` (`continuation_corpus.rs:447`).
  - Observations with `failure` set are skipped before labeling (`corpus.rs:551-556`). Values must be finite and ≥0 (`corpus.rs:566, 421-424`).
- **Accepted corpus.** Corpus `1c840597…` has schema `parallel.teaching_shards.v1` and label policy **`reference_v1`**, with 10,214,332 records and 7,877 groups (EVID: `docs/internal/asm-manual/data/target-history/corpus-snapshots.csv:2`; `docs/internal/asm-manual/sections/data.md:64`).

---

## 3. Per-term specification

Notation: rows `i ∈ M` (`|M| = B`). `π_ij` is the softmax of the position-`j` logits restricted to legal set `A_ij`. `s_i = ¼ Σ_{k=0..3} reward_heads.k(φ_i)`, the mean over the four heads (`training.rs:1343`). `a_i` is the ablation head output. `ŷ_i` is the runtime head output.

### 3.1 `seq_nll`: grammar-masked, smoothed, percentile-weighted imitation
- **Code:** `neural.rs:534-577`, called at `training.rs:1329-1342`.
- **Inputs and target source.** Decoder logits against the corpus token sequence of every one of the 8 sampled programs, not just the champion. The weight on each program comes from its percentile.
- **Masks.**
  - Illegal tokens get logit −∞ before `log_softmax` (`neural.rs:555-558`).
  - The error `"empty legal token set"` is raised if any position's legal set is empty, including PAD positions (`neural.rs:541-544`).
  - The error `"illegal supervision token"` is raised if a supervised target is outside its legal set (`neural.rs:545-554`).
  - PAD positions are zeroed (`neural.rs:570`).
- **Smoothing.** ε = 0.05, uniform over the legal set only:
  `ℓ_ij = 0.95·(−log π_ij(x_ij)) + 0.05·(1/|A_ij|)·Σ_{a∈A_ij} −log π_ij(a)` (`neural.rs:559-569`).
  - A singleton legal set gives `ℓ = 0` exactly.
- **Row reduction.** `r_i = Σ_{j=1}^{T−1} m_ij ℓ_ij / max(1, Σ_j m_ij)` (`neural.rs:571-574`). This is the mean over supervised tokens, including singleton-legal forced positions such as EOS, which add 0 to the numerator and 1 to the denominator.
- **Batch reduction.** `w_i = max(0, 0.05 + p_i⁴)` (`training.rs:1332`; `neural.rs:575`) and `L_seq = Σ_i w_i r_i / max(1e−8, Σ_i w_i)` (`neural.rs:576`).
  - Numerator: weighted row means. Denominator: sum of row weights within the microbatch.
  - The champion has weight 1.05 and the worst program 0.05 (a 21× ratio).
- **Empty support.**
  - A row with no supervised token gives `r_i = 0` but keeps `w_i` in the denominator (dilution). Encoding yields at least 5 tokens, but `loss_for` does not assert this.
  - The denominator cannot reach 1e-8 because every `w_i ≥ 0.05`.
- **Units.** Nats per supervised token.
- **Gradient consumers.** `encoder.{0,2,4,6}`, `ctx_proj`, `token_emb` (input rows used, plus the tied output rows below `vocab_size`), `pos_emb` (rows `0..T`), `decoder.layers.{0..5}.*`, `out_norm`. It is the only term that trains the decoder side. No other term reaches these tensors (§4).
- **Accepted step-60000.** Present: `validation.sequence_nll = 1.3876872` (EVID `accepted60000-state.json:21315-21329`). Coefficient 1.0 per SNAP `training.rs:720`.

### 3.2 `reward_bt`: pairwise Bradley–Terry on mean reward
- **Code:** `neural.rs:700-707, 720-807`, called at `training.rs:1351`. Backward identity `dense_pairwise_v2` (`neural.rs:13`).
- **Inputs.** `s_i` (mean of 4 heads, each with dropout 0.05: `neural.rs:323-332, 514-521`), the labels `v_i` and `p_i`, and the group key.
- **Pair construction** (`neural.rs:739-759`).
  - For each group, all unordered pairs of rows are formed.
  - A pair is skipped if either `v` is non-finite or `v_a == v_b` (exact equality after f32→f64). Ties are skipped.
  - The winner is the larger `v`. The pair set depends only on membership and labels, not row order. Row order changes only the dense-matrix layout and floating-point summation order.
  - A group of 8 has at most 28 pairs.
- **Pair weight.** `ω_wl = (v_w − v_l) · (2 if max(p_w, p_l) ≥ 0.9 else 1)` (`neural.rs:750-757`). Every comparison contains the champion (`p = 1.0`), so all 7 champion pairs are doubled.
- **Per-element loss.** `softplus(s_l − s_w) = −log σ(s_w − s_l)`. The dense matrix has `M[w,l] = s_l − s_w` (`neural.rs:787-797`).
- **Reduction.** `L_BT = (1/|P|) Σ_P (ω_wl/ω̄) · softplus(s_l − s_w)` with `ω̄ = mean_P ω` (`neural.rs:782, 790`), which equals `Σ_P ω_wl·softplus / Σ_P ω_wl`.
  - The weights `ω/ω̄` are cast to f32.
  - Normalization is over the whole microbatch, so comparisons with a larger value spread carry more weight. Groups are not equally weighted.
- **Empty support.** If `P = ∅`, `L_BT = reward.sum()·0` (`neural.rs:778-781`). The term contributes exactly 0 loss and 0 gradient for that microbatch, with no renormalization across microbatches.
- **Units.** Dimensionless logistic loss, in nats, weighted by relative Δvalue.
- **Gradient consumers.** `reward_heads.{0..3}.{0,3}`, `z_gamma.0`, `z_proj`, `reward_norm`, `reward_token_emb` (rows in use), `reward_encoder.{0,2,4,6}` (`neural.rs:489-521`).
  - `reward_heads.k.3.bias` gets **zero gradient in exact arithmetic** because BT is shift-invariant (`neural.rs:783-786`). The realized numerical gradient is **UNVERIFIED**.
- **Accepted step-60000.** Present: `validation.ranking_bt = 0.5075364` and `ranking_backward = "dense_pairwise_v2"` (EVID `accepted60000-state.json:21301`). Coefficient 0.25 per SNAP `training.rs:721`.

### 3.3 `reward_listwise`: softmax-percentile listwise cross-entropy
- **Code:** `neural.rs:760-776, 799-806`.
- **Group eligibility.** At least 3 members and all `p` finite (`neural.rs:760-763`). In training every group has 8 members, so every comparison is eligible.
- **Target.** `q_g = softmax(p_g / 0.25)` over the group (`neural.rs:770`). The temperature is 0.25. The target has non-zero entropy, so the loss has a floor above zero.
- **Per-group loss.** `L_g = −Σ_{i∈g} q_{g,i} · log_softmax(s_g)_i` (`neural.rs:771-775`).
- **Reduction.** `L_list = mean over eligible groups` (`neural.rs:804`). Each group counts equally regardless of size or spread.
- **Empty support.** No eligible group gives a connected zero (`neural.rs:801-803`).
- **Order and membership.** Permutation-invariant (a set function). A duplicated member would count twice.
  - Under `reference_v1`, RMSE-tied programs get different `p` by program-string order, so listwise distinguishes ties that BT skips.
- **Units.** Nats.
- **Gradient consumers.** Same as §3.2. `reward_heads.k.3.bias` is also exact-zero here because of shift invariance.
- **Accepted step-60000.** Present: `validation.ranking_listwise = 2.2188580`. Coefficient 0.10 per SNAP `training.rs:722`.

### 3.4 `runtime_huber`: Huber on log1p seconds
- **Code:** `training.rs:1353-1359`; head at `neural.rs:333, 522-529`.
- **Prediction.** `ŷ_i = softplus(runtime_head([rz_i, g_i]))`, on a seconds scale and ≥0. The runtime head has no dropout.
- **Target.** `y_i = ln(1 + seconds_i)` (§2.3).
- **Per-element loss.** `d_i = log1p(ŷ_i) − y_i`. SmoothL1 with β = 1: `h(d) = 0.5d²` if `|d| < 1`, else `|d| − 0.5`.
- **Reduction.** `L_rt = (1/B) Σ_i h(d_i)` (`tch::Reduction::Mean`). Every row is supervised: there is no mask and no missing, censored or timeout representation. Failed observations never reach the corpus.
- **Empty support.** Not possible (B ≥ 64).
- **Units.** Log1p-seconds error. The runtime objective identity is `huber_log1p_seconds_v2` (`training.rs:270`).
- **Gradient consumers.** `runtime_head.{0,2}`, `reward_encoder.*`, `reward_norm`, `reward_token_emb`.
- **Accepted step-60000.** Present: `validation.runtime_huber = 0.0695442` and `runtime_objective = "huber_log1p_seconds_v2"` (EVID `:21304`). Coefficient 0.10 per SNAP `training.rs:723`.

### 3.5 `ablation_bt`: context-free program-only Bradley–Terry
- **Code:** `neural.rs:709-718`, called at `training.rs:1352`. The head is `reward_head_ablation(g)` with dropout 0.05 (`neural.rs:335, 459`).
- **Construction.** Identical pairs, weights, reduction and empty-support rule as §3.2, applied to `a_i` instead of `s_i`. No listwise term.
- **Units.** Dimensionless.
- **Gradient consumers.** `reward_head_ablation.{0,3}`, `reward_norm`, `reward_token_emb`. It does not reach `reward_encoder`. `reward_head_ablation.3.bias` is exact-zero by shift invariance.
- **Accepted step-60000: source-inferred only, not in telemetry.**
  - SNAP `training.rs:712, 724` includes `+ abt * 0.05`, but SNAP telemetry carries only 4 components (`training.rs:732-737, 770, 859`).
  - The accepted state's `validation` has exactly those 4 metrics and no `contextfree_bt` (EVID `accepted60000-state.json:21315-21329`).
  - The manual records the gap: "Absent from Sep11 endpoint telemetry; present in 60100 original logs" (EVID `training-history/metric-definitions.json:20-23`).
  - The later step-60100 hardware qualification logs it: EVID `rtx60100-state.json:83, 97` and 100 `contextfree_bt` rows in `rtx60100-step-metrics.csv`.

### 3.6 `stability_bce`: inactive
- **Code.** `stability = stability_head(φ)` is computed (`neural.rs:334, 458`); the loss adds `stability.sum()·0.0` (`training.rs:1365`).
- **No stability label exists** in `Record`, so no BCE is ever computed.
- **Purpose of the zero term.** It keeps every parameter's gradient defined so the packed all-reduce layout is identical (`training.rs:1366, 1460-1465`).
- **Gradient.** `stability_head.{0,2}.*` get exact zeros. The tensors are nonetheless updated by AdamW decoupled weight decay (§5).
- **Accepted run.** Inactive per SNAP `training.rs:725`. The head is exported with role `training_only` (EVID `docs/internal/core-ip/data/tensors.json`).

### 3.7 `routing_ce`: absent
There is no routing head, label or computation anywhere in `Policy` or `loss_for`. Only the scalar `loss_log_vars.routing_ce` exists (§3.8).

### 3.8 `loss_log_vars.*`: stored, inert scalars
- **Initialization.** Seven zero-initialized f32 scalars (`neural.rs:292-303`).
- **Training use.** Only `x·0.0` (`training.rs:1367-1369`). This is fixed weighting, not learned uncertainty weighting.
- **Accepted value.** All seven are **exactly +0.0 at step 60000**. Each tensor's `payload_sha256 = df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119` (EVID `tensors.json`), which is the SHA-256 of four zero bytes (−0.0 hashes to `6d586926…`).
- **Why they stay zero.** Zero gradient and zero moments leave only decay applied to 0.
- **Origin of the names.** They are inherited from the Python `policy_v3.2` lineage, whose `train_loss` "includes learned uncertainty weights for sequence/runtime/stability and fixed-scale ranking/routing terms" (EVID `metric-definitions.json:48-50`). That formula lives outside this repository, is **UNVERIFIED**, and is not lineage B.

### 3.9 Regularizers and non-objective code paths
- **Dropout.** 0.1 in both context encoders and the decoder attention, residual and FFN (`neural.rs:45, 104, 160, 168-176`); 0.05 in the reward heads and ablation head (`neural.rs:329, 335`). Off during validation (`train=false`, `training.rs:1520`).
- **Label smoothing.** 0.05 (§3.1).
- **Weight decay.** Decoupled, λ = 0.01, applied to every parameter including biases, LayerNorm, embeddings and the log-vars (§5).
- **Defined but unreachable from any training path** (grep of `crates`). These must not be cited as training terms:
  - `sequence_logprob`, `preference_pairs`, `preference_dpo` (`neural.rs:578-698`).
  - `OrderAwareRewardEncoder` (`neural.rs:180-264`; used only in `tests/reward_sequence.rs`).
- **`training_probe` objective** (`neural.rs:1265-1276`). Unmasked cross-entropy, plus 0.01 × the mean of squares of each head, plus 0.001·Σ `loss_log_vars²`.
  - It is the only path where `loss_log_vars` get a non-zero gradient.
  - It is a gated BF16 architecture probe on random fixtures and writes `eligible_checkpoint:false` (`neural.rs:1223, 1342`). It is not lineage-B training.
- **Checkpoint selection, not a loss.** `auxiliary_training.rs` drives the same `train_controlled` path and selects checkpoints by fixed-panel validation NLL (`auxiliary_training.rs:226-375`). It adds no loss term.

---

## 4. Which tensors each term reaches (by root name)

Shared context: dropout, BF16 autocast. **G** = gradient; **0\*** = zero in exact arithmetic (shift-invariant); **—** = none.

| Tensor root | seq_nll | reward_bt | reward_listwise | runtime_huber | ablation_bt | stability | log-vars |
|---|---|---|---|---|---|---|---|
| `encoder.*`, `ctx_proj.*`, `decoder.layers.*`, `out_norm.*`, `pos_emb` | G | — | — | — | — | — | — |
| `token_emb` (rows < 401) | G (input and tied output) | — | — | — | — | — | — |
| `reward_encoder.*` | — | G | G | G | — | 0 | — |
| `z_gamma.0.*`, `z_proj.*` | — | G | G | — | — | 0 | — |
| `reward_heads.k.{0.*,3.weight}` | — | G | G | — | — | — | — |
| `reward_heads.k.3.bias` | — | 0\* | 0\* | — | — | — | — |
| `runtime_head.*` | — | — | — | G | — | — | — |
| `reward_norm.*`, `reward_token_emb` (rows < 401) | — | G | G | G | G | 0 | — |
| `reward_head_ablation.{0.*,3.weight}` | — | — | — | — | G | — | — |
| `reward_head_ablation.3.bias` | — | — | — | — | 0\* | — | — |
| `stability_head.*` | — | — | — | — | — | exact 0 | — |
| `loss_log_vars.*` (7) | — | — | — | — | — | — | exact 0 |

The decoder side and the reward side share no parameters. Adam is almost invariant to a constant gradient scale, so a coefficient mostly matters for parameters fed by more than one term:
- `reward_token_emb` and `reward_norm`: BT, listwise, runtime and ablation.
- `reward_encoder`: BT, listwise and runtime.
- All parameters, through global clipping.

This is an analytical observation. The realized per-term gradient influence is **UNVERIFIED** because no per-term gradient norms are logged (`learning-hpc.md:33`).

### 4.1 Tensors that receive no learning signal
These receive decay-only or zero updates and must not be cited as learning:
- `stability_head.{0,2}.{weight,bias}`: exact-zero gradient.
- `loss_log_vars.*`: exact-zero gradient; verified +0.0 at step 60000.
- `token_emb` and `reward_token_emb` rows 401..656. `embedding_rows = vocab + 256` (`neural.rs:43`); the step-60000 config has 657 rows (`crates/jope-policy/tests/jope_dime.rs:133, 165`). These rows are never indexed, and the output projection is narrowed to `vocab_size` (`neural.rs:456`).
- Row 0 of both embeddings: `padding_idx=0`, zero-initialized (`neural.rs:339-342`). The PAD logit is legal only at masked positions.
  - That no grammar state lists token 0 or 1 as legal is **UNVERIFIED** against the vocabulary file.
- `reward_heads.k.3.bias` and `reward_head_ablation.3.bias`: exact-arithmetic zero.

All of these still receive decoupled decay. Every gradient is defined (`training.rs:1462-1465`), so fused AdamW touches every tensor (`neural.rs:909-926`). With zero moments the update is `θ ← θ(1 − 1e-6)` per step, a factor of 0.94176 over 60,000 steps. Whether the moments of the inactive tensors were zero throughout the accepted ancestry is **UNVERIFIED**.

---

## 5. Optimizer, clipping, precision, accumulation

- **AdamW.** FP32 master weights and moments, fused CUDA kernel `at::_fused_adamw_(…, lr, 0.9, 0.999, decay, 1e-8, amsgrad=false, maximize=false)`, then gradients are zeroed (`native/runtime.cpp:53-69`; `neural.rs:900-960`).
  - Reference CPU path: `neural.rs:868-899`. Parity is asserted by `tests/training.rs:184` (<1e-6).
  - η = 1e-4 and λ = 0.01 are hard-coded (`training.rs:1062`).
  - Bias correction uses the carried global step `t` (`neural.rs:941`; the accepted state has `"optimizer":"carried FP32 AdamW with bias-correction step"`).
  - Scheduler: constant, no warmup (`training.rs:244`; EVID `accepted60000-state.json:21307-21313`, lr 1e-4, `epoch_steps` 4582).
  - A loaded optimizer takes lr and decay from `optimizer.json` (`neural.rs:1000-1002`). These are checked to equal 1e-4 and 0.01 only on the warm-start carry path (`neural.rs:825-831`; `training.rs:1122`), not on exact resume.
- **Clipping.** Global L2 norm of the all-reduced packed gradient; `if norm > 1: g /= norm` (`training.rs:1480-1483`). Max norm 1.0, no epsilon, applied after averaging over ranks.
- **Accumulation.** Each microbatch loss is divided by `A` and backpropagated into `.grad` (`training.rs:1441, 1447`). Gradients are then packed in `BTreeMap` name order, checked finite, NCCL-summed and divided by `W` (`neural.rs:1063-1072`).
- **Precision.** Forward runs under scoped CUDA BF16 autocast, with the cache cleared at scope exit (`native/runtime.cpp:37-51`; `training.rs:1439, 1446`). Loss arithmetic is cast to FP32 (`neural.rs:556, 795`; `training.rs:1355`). There is no loss scaling.
- **RNG (dropout).** `manual_seed(mix(seed ^ local_step·65537 ^ rank·257 ^ mb))` before each microbatch (`training.rs:1422-1425`).
- **Accepted partition.** World 8, microbatch 256, accumulation 1, global batch 2048, giving 32 comparisons per microbatch and 256 per update.
  - World 8 and global batch 2048 are in EVID `accepted60000-state.json:6, 21330`.
  - Microbatch 256 and accumulation 1 are only in the manual's reading of external `metrics.jsonl` (EVID `sections/training-history.md:141`). The state has no `execution_partition`. The external logs were **not re-read here**.

---

## 6. Nonfinite and malformed-supervision handling

| Condition | Behavior | Anchor |
|---|---|---|
| Nonfinite context vector | error, step aborted | `neural.rs:382-386` |
| Token out of `[0, vocab)`, `T ≤ 1`, pad/token shape mismatch | error | `neural.rs:416-423` |
| Empty legal set, or target outside legal set at a supervised position | error (not skipped) | `neural.rs:541-554` |
| Nonfinite value or percentile in a pair or list | pair skipped; listwise group skipped | `neural.rs:744, 762` |
| Nonfinite or negative RMSE/runtime at corpus build | refused before training | `corpus.rs:566, 421-426` |
| Nonfinite packed gradient on any rank (pre-reduce) | error: `rank_impl` fails and writes a failed rank receipt; no update on that rank; the other ranks' collective cannot complete; the controller terminates the group | `training.rs:1473-1476, 1677-1697`; `native/runtime.cpp:11-12` |
| Nonfinite gradient at the fused optimizer | error before the kernel; step not incremented | `neural.rs:928-931, 957` |
| Nonfinite scalar loss | **not checked.** An overflowing inactive head times 0.0 gives a NaN loss value with zero gradients; it is logged and not failed | `training.rs:1365-1369, 1442` |
| Rank parameter divergence | the checkpoint is refused | `training.rs:231-236` |

There is no skip-and-continue, retry, or loss-scaler step skipping in lineage B.

---

## 7. What a "committed update" is

1. **Applied optimizer update (in memory).** One fused AdamW call on the all-reduced, clipped gradient. `optimizer.step` increments only after the kernel succeeds (`neural.rs:941-958`).
   - The CPU reference path increments `step` before its per-parameter loop and could partially update if it hit a nonfinite gradient mid-loop (`neural.rs:874-896`). It is not used by `rank_impl`, which is CUDA-only (`training.rs:1058`).
2. **Durable committed global update.** When the checkpoint vote passes (`training.rs:1503-1514`):
   - Triggers: every `checkpoint_steps`, each epoch boundary, the final step, rank 0 at ≥540 s wall-clock, or native stop.
   - Each rank writes a receipt with its parameter hash and exposure. After a barrier, rank 0 verifies identical hashes.
   - Rank 0 stages `model.ot`, `optimizer.ot`, `optimizer.json`, `config.json` and `state.json`, fsyncs them, renames the directory to `checkpoints/step-N`, and updates `latest.json` (`training.rs:194-306`).
   - Committed exposure advances only here (`training.rs:1580`).
   - Updates after the last committed checkpoint are lost on failure. Recovery re-executes them from `latest.json` with the same counter-based sampler and RNG (`training.rs:1422-1438`).
   - The committed-update count is `state.json.step == optimizer.json.step` (`task_contract.rs:54-58`).

---

## 8. Accepted step-60000: presence of each term

The accepted state is `accepted60000-state.json` (state SHA `8ac80c26…`): a warm start from step 57000 with a carried optimizer, 3000 local steps, and corpus `1c840597…`.

| Term | Coefficient (current SRC) | SNAP 2026-09-11 | Accepted state (EVID) | Classification |
|---|---|---|---|---|
| seq_nll | 1.0 | 1.0 | `sequence_nll` 1.3877 logged | present |
| reward_bt | 0.25 | 0.25 | `ranking_bt` 0.5075; `ranking_backward=dense_pairwise_v2` | present |
| reward_listwise | 0.10 | 0.10 | `ranking_listwise` 2.2189 | present |
| runtime_huber | 0.10 | 0.10 | `runtime_huber` 0.0695; `runtime_objective=huber_log1p_seconds_v2` | present |
| ablation_bt | 0.05 | in loss, not logged | absent from telemetry | source-inferred only |
| stability_bce | 0.0 | 0.0 | head tensors exist (`training_only`) | inactive |
| routing_ce | — | — | log-var scalar only | absent |
| loss_log_vars | ×0.0 | ×0.0 | all seven = +0.0 | inactive; must never be reported as adaptive |

- **Consistency with the snapshot.** The accepted state's 12-key `validation` object matches the SNAP JSON (`training.rs:859`) exactly.
- **Source identity is UNVERIFIED.** The state carries no source or executable digest and no coefficients, so byte identity with the snapshot cannot be established from it. The snapshot hashes match a Sep-11 build binding (`learning-hpc.md:11`), not the Sep-12/13 cycles 1–21.

---

## 9. Existing fixture coverage relevant to the T02 acceptance

Test names are from source; none were run.
- `tests/training.rs:104` `actual_policy_learns_and_optimizer_continues_after_reload`: `sequence_nll` only, with an all-true legal mask; CUDA-only and returns early without CUDA.
- `tests/training.rs:184` fused vs reference AdamW parity; `:247` dense BT values and gradients vs reference; `:326` reward-policy continuation reproduces parameters exactly.
- `tests/ablation_ranking.rs:128-153`: `ranking_pairwise` bit-identity against a frozen reference, ties, missing values, irregular groups, label-shape validation.
- `neural.rs:1153-1210` `loss_optimizer_parity` (CLI `main.rs:1964`): uses an external `fixtures.safetensors` whose location and provenance are **UNVERIFIED**.
- **Not covered by any test found:**
  - The composite `loss_for` and its coefficients.
  - Runtime Huber.
  - Listwise alone.
  - Smoothing over non-trivial legal sets.
  - Empty-support and dilution behavior.
  - Exact-zero gradients of the inactive terms and the decay-only drift.
  - PAD-extension invariance.
  - Per-term gradient reach (the §4 matrix).

---

## 10. Gaps for W6-T02 / W6-T03

1. **G1 — The objective is unversioned beyond two IDs.**
   - Coefficients (1/0.25/0.10/0.10/0.05/0), smoothing (0.95/0.05), row weights (0.05 + p⁴), listwise temperature 0.25, the ×2 boost at p ≥ 0.9, ablation inclusion and dropout are literals: `training.rs:1332, 1360-1365`; `neural.rs:569, 750-757, 770`.
   - The coefficients are duplicated in telemetry (`training.rs:1525, 1529`).
   - Checkpoints record only `ranking_backward` and `runtime_objective` (`training.rs:269-270`), and the resume and carry guards compare only those plus the partition (`training.rs:991-999, 1105-1110, 1140-1144`).
   - Any other change passes as "exact resume" or "compatible carry". This is S06-G06 and S06-K05.
2. **G2 — The accepted run's objective is not bound to its source.** There are no coefficients and no source or executable digest in the step-60000 state. `ablation_bt` is evidenced only by the snapshot source and later logs (§8). The microbatch comes from the manual's reading of external logs only.
3. **G3 — The partition-dependent objective is absent from the accepted state.** There is no `execution_partition`, so the current exact-resume guard (`training.rs:1140-1141`) cannot accept this checkpoint; only warm start works (as the 60100 run did).
4. **G4 — The inert log-variance names risk misreporting.** Seven zero scalars are multiplied by 0.0, and `routing_ce` has no module. Any report must say "fixed coefficients; log-vars inactive" (S06-K05). The Python `policy_v3.2` uncertainty formula is outside the repository and **UNVERIFIED**.
5. **G5 — Inactive tensors are decayed, not frozen.** `stability_head`, the unused embedding rows, row 0 and the log-vars all receive AdamW decay (§4.1). There is no frozen-parameter mechanism: every gradient must be defined (`training.rs:1462`). The realized drift is **UNVERIFIED**.
6. **G6 — Shift-invariant output biases.** The four `reward_heads.k.3.bias` tensors and `reward_head_ablation.3.bias` should receive zero gradient. Their realized numerical updates are **UNVERIFIED** (`neural.rs:783-786`).
7. **G7 — Forced tokens dilute `seq_nll`.** Singleton-legal positions (EOS, ExprEnd, possibly engine and task) add 0 to the numerator and 1 to the denominator. This is undeclared normalization. Engine/task cardinality in the vocabulary is **UNVERIFIED**.
8. **G8 — The empty-row policy is not enforced.** A row with no supervised token keeps its weight (`neural.rs:571-576`). This is unreachable through `encode`, but `loss_for` does not assert it.
9. **G9 — Tie policy depends on the corpus label policy and is not recorded in state.**
   - BT skips exact ties.
   - `reference_v1` (the accepted corpus) breaks ties by program string in `p`, which affects the listwise target and the sequence weights. Midrank corpora share `p`. `frozen_z_v1` keeps stale `v, p`.
   - The boost threshold uses the corpus-group rank, not the rank within the comparison. This is S06-G07 and S06-K09.
10. **G10 — The group weighting policy is undeclared.** BT weights are normalized over the whole microbatch by |Δv| and the boost, so high-spread comparisons dominate. Listwise weights comparisons equally.
11. **G11 — Runtime label typing** (S06-G08, S06-K10).
    - Seconds come from `numerical_runtime_seconds` falling back to `runtime_seconds` (`corpus.rs:561-563`), or `fit_runtime_seconds` for continuation, so timing scopes can mix.
    - No device, profile or censoring fields exist.
    - Whether timed-out observations can carry `failure = null` and reach labels is **UNVERIFIED**. `teaching_store.rs:764-775` records timeout outcomes and deadlines, but the label path was not traced.
    - Predictions use `log1p(softplus)` in seconds, not a calibrated duration.
12. **G12 — Nonfinite gaps.** The scalar loss is unchecked. A `×0.0` inactive term can make the logged loss NaN without failing. The exact-resume path does not re-validate lr and decay. A gradient-norm overflow to inf (giving `g/inf = 0`) is **UNVERIFIED**.
13. **G13 — Train/serve legality mismatch.** Legal sets are built with `sampling=false` (`corpus.rs:909`), which differs from generation-time legality (`grammar.rs:142, 150`). This is undeclared in any objective contract.
14. **G14 — PAD-only extension.** `T` depends on the longest row in the microbatch. In train mode, dropout draws depend on tensor shape, so the realized stochastic supervision changes. Exact invariance is expected only with dropout off and in exact arithmetic; BF16 kernel effects are **UNVERIFIED**. The K05 fixture must fix dropout at 0.
15. **G15 — Sequence order vs list order** (for T03). The reward, runtime and ablation heads are permutation-invariant over the program's token multiset (a masked mean, `neural.rs:496-505`). Only `seq_nll` sees token order. BT and listwise depend on membership, not row order.
16. **G16 — Telemetry naming and completeness.**
    - Logged names differ from the `loss_log_vars` names.
    - Validation hard-codes `sampler_id: SAMPLER_ID` even under replay (`training.rs:1525`).
    - No per-term gradient norms are logged, so coefficient influence cannot be measured from logs.
17. **G17 — Unreachable objective code.** The DPO and preference helpers and `OrderAwareRewardEncoder` exist but are not called by any trainer. Only `training_probe` gives the log-vars a gradient. The objective contract must list these as excluded.

---

**Could not verify:** the exact source and executable that produced step 60000 (and so `ablation_bt` in that run), the external per-step logs behind "microbatch 256, accumulation 1", the ancestry of the inactive tensors' moments, the realized numerical bias gradients, the vocabulary's reserved and singleton legal sets, the timeout-to-label path, and the `loss_optimizer_parity` fixture file.
