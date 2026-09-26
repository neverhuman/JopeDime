# W4-T01 operation-level forward contract

## 0. Status, scope and source identity

| Item | Value |
|---|---|
| Task | W4-T01, `docs/internal/core-ip/work-orders/W04-generator-ranking.tasks.md:23-31` |
| Evidence type | Read-only source trace. Nothing was built, tested or run, and no weights were loaded. This document does not by itself move W4-T01 to any progress state in `EXECUTION-CONTRACT.md`. |
| Source | Worktree `/home/ubuntu/JopeDime/.agent/worktrees/bench`, branch `feat/regression-remediation`, HEAD `0ea10dd48aafce95a88dab115963ab448cfe6097`. HEAD moved from `0e2ca312b31afd1221b9948ab277160ea2d23929` while I was reading. `git diff --stat` between those two commits is empty for every traced path, and no traced path is dirty. |
| Dependencies read | `candle-core`/`candle-nn` 0.11.0, `tch` 0.26.0 and `torch-sys` 0.26.0 from the Cargo registry. `torch-sys-0.26.0/build.rs:13` expects LibTorch `2.13.0`. The LibTorch actually linked at runtime was **not verified**. |
| Accepted tensors (W2) | `docs/internal/core-ip/data/tensors.json`. It covers `policy.safetensors` sha256 `4b4b5a40f03c9226b4b15856fd35b90127c8b6a4d1435b815484cdd050744f1e`: 136 tensors, 27,796,494 parameters, all F32. Roles are `decode` 86, `rank` 35 and `training_only` 15. |
| Accepted configuration | `PolicyConfig::regression(401)`, per `crates/jope-policy/src/lib.rs:154-159` (`ACCEPTED_CONFIG_SHA256`). This matches every shape in `tensors.json`. I did not re-hash `config.json` myself. |

**Concurrent change note.** Another agent's branch `rr/w4-decoding` (demo worktree, HEAD `b476f79aa747636d6136b0dfc7ae4f86af1c5286`) changes `crates/jope-policy/src/{decode.rs,grammar.rs,producer.rs}` relative to bench.

- `model.rs` and `model/cached.rs` are byte-identical on both branches, so **this forward contract is unaffected**.
- **Decode bookkeeping may shift.** The demo branch adds typed terminal outcomes and budget checks (rounds ≤ 8, batch ≤ 128, temperature, NaN/+Inf logits, fallback exhaustion). It also adds `State::phase_name`.
- The demo sampling loop still calls `next_token_logits`/`DecodeCache` with `limit = min(MAX_PROGRAM_TOKENS, max_seq)` (demo `decode.rs:253, 384-424`).
- Decode references below are to the **bench** version.

### Path legend

| Tag | Path (repo-relative) |
|---|---|
| C:model | `crates/jope-policy/src/model.rs` |
| C:cached | `crates/jope-policy/src/model/cached.rs` |
| C:config | `crates/jope-policy/src/config.rs` |
| C:grammar | `crates/jope-policy/src/grammar.rs` |
| C:decode | `crates/jope-policy/src/decode.rs` |
| C:ranking | `crates/jope-policy/src/ranking.rs` |
| C:lib | `crates/jope-policy/src/lib.rs` |
| N:neural | `crates/jope-engine/src/neural.rs` |
| N:proposals | `crates/jope-engine/src/proposals.rs` |
| N:pranking | `crates/jope-engine/src/proposal_ranking.rs` |
| N:ranked | `crates/jope-engine/src/ranked_proposals.rs` |
| N:training | `crates/jope-engine/src/training.rs` |
| N:runtime | `crates/jope-engine/native/runtime.cpp` |
| T:jd | `crates/jope-policy/tests/jope_dime.rs` |

**Native forward location.** `runtime.cpp` implements **no** forward pass and no decoding. It provides NCCL, fused AdamW and the BF16 autocast scope switch `jp_bf16` (N:runtime:37-51). The native forward is entirely `Policy` in N:neural, executed through tch/LibTorch.

### Notation

- B = rows, T = tokens per row, S = T + 1 decoder positions.
- Accepted values: W = 512, H = 4, D = W/H = 128, F = 2048, V = `vocab_size` = 401, E = `embedding_rows` = 657, `max_seq` = 128.
- Context C = `dataset_dim + target_dim` = 1536 + 512 = 2048. `target_keep_dim` = 512.

---

## 1. Entry points, device, dtype and mode

| Consumer path | Backend and device | Dtype regime | Train/dropout | Anchors |
|---|---|---|---|---|
| Candle production proposal (`seeds::propose` → `decode::pure_pool_checked` → `sample_batch_cached`, then `ranking::score_pool`) | Candle, **CPU** (`crates/jope-policy/src/bin/jope-policy-producer.rs:160`) | F32 end to end. Every tensor must be F32 at load (C:model:187-191) | No dropout op exists; `config.dropout` is validated but never used | C:decode:413-476, C:ranking:339-451 |
| Candle full-prefix reference (`sample_batch`, `logits`, `next_token_logits`) | Candle CPU; CUDA only in the `inference-cuda` bench panel (`crates/jope-bench/src/panel/inference_cuda.rs:69`) | F32 | none | C:model:515-540, C:decode:92-110 |
| Native pure-pool decode (`pure_pool_impl`) | tch; `rank_impl` always uses `Device::Cuda(rank)` (N:proposals:147) | **BF16 autocast**. `Bf16::enter()` is unconditional (N:proposals:807) but only enables CUDA autocast (N:runtime:44-45) | `train=false`, inside `tch::no_grad` | N:proposals:748-875 |
| Native `score_pool` | CUDA → BF16 autocast; CPU → FP32 (N:pranking:186) | Backend label `libtorch_cuda_bf16` / `libtorch_cpu_float32` (N:pranking:228) | `reward_runtime` hard-codes `false` (N:neural:484-485) | N:pranking:88-233 |
| Native `ranked_proposals` sample / `scoring_parity` | CUDA by default; CPU if `JOPE_RANKED_DEVICE=cpu` (N:ranked:801-805); BF16 only on CUDA (N:ranked:299, 426) | as left | `false` | N:ranked:251-353, 395-463 |
| Native training loss | CUDA, BF16 autocast (N:training:1439-1440) | BF16 autocast | **`train=true`** | N:training:1298-1380 |
| Native validation and monitoring | BF16 autocast | BF16 autocast | `false` | N:training:1518-1521, `crates/jope-engine/src/monitoring.rs:166-175` |
| Native parity harness `neural::parity` | tch CPU | FP32 | `false`, `no_grad` | N:neural:1116-1151 (atol 3e-4, compared against externally supplied fixtures) |

**Weight artifacts differ.**

- Native loads `model.ot` into a `VarStore`. Identity is the `model.ot` file hash compared with `state.json` (N:proposals:133-150; N:ranked:369-389). The parameter hash is **not** recomputed at native load.
- Candle loads the exported `policy.safetensors`. It checks the file sha256, per-tensor name/shape/dtype/sha256 against the export receipt, and a recomputed parameter hash (sorted names plus little-endian f32 bytes) that must equal `ACCEPTED_PARAMETER_SHA256` (C:model:215-251, 332-344).
- **The two paths are linked only through that parameter hash.**

---

## 2. Accepted configuration and tensor map

**Configuration.** `regression(401)` gives (C:config:39-55; N:neural:33-49):

- `dataset_dim` 1536, `target_dim` 512, `target_keep_dim` 512
- `width` 512, `layers` 6, `heads` 4, `ffn` 2048
- `vocab_size` 401, `embedding_rows` 657, `max_seq` 128
- `dropout` 0.1, `dataset_rms` 1.0, `target_rms` 1.0

**Tensor map.** Names and shapes below are from `tensors.json`. Operation IDs refer to §3–§6.

| Tensor(s) | Shape | Role | Used by op | Candle | Native |
|---|---|---|---|---|---|
| `encoder.0.{weight,bias}` | [512,2048], [512] | decode | E2 | C:model:382-386 | N:neural:96-101 |
| `encoder.2.{weight,bias}` | [512], [512] | decode | E4 (LayerNorm) | C:model:388-392 | N:neural:103 |
| `encoder.4.{weight,bias}` | [512,512], [512] | decode | E6 | C:model:393-397 | N:neural:105 |
| `encoder.6.{weight,bias}` | [512], [512] | decode | E8 (LayerNorm) | C:model:399-403 | N:neural:107 |
| `ctx_proj.{weight,bias}` | [512,512], [512] | decode | P1 | C:model:440 | N:neural:307, 429 |
| `token_emb.weight` | [657,512] | decode | P2 (input) **and** O2 (tied output) | C:model:443, 522 | N:neural:308, 430, 455 |
| `pos_emb.weight` | [129,512] | decode | P4 | C:model:445-448 | N:neural:309, 433-436 |
| `decoder.layers.{0..5}.norm1.{weight,bias}` | [512] ×2 | decode | L1 | C:model:472 | N:neural:152 |
| `decoder.layers.{0..5}.self_attn.in_proj_{weight,bias}` | [1536,512], [1536] | decode | L2–L3 | C:model:490-502 | N:neural:153-154 |
| `decoder.layers.{0..5}.self_attn.out_proj.{weight,bias}` | [512,512], [512] | decode | L6 | C:model:474-478 | N:neural:168 |
| `decoder.layers.{0..5}.norm2.{weight,bias}` | [512] ×2 | decode | L8 | C:model:480 | N:neural:172 |
| `decoder.layers.{0..5}.linear1.{weight,bias}` | [2048,512], [2048] | decode | L8 | C:model:481 | N:neural:171-172 |
| `decoder.layers.{0..5}.linear2.{weight,bias}` | [512,2048], [512] | decode | L8 | C:model:482 | N:neural:169 |
| `out_norm.{weight,bias}` | [512] ×2 | decode | O1 | C:model:521 | N:neural:453-454 |
| `reward_encoder.{0,2,4,6}.*` | same shapes as `encoder.*` | rank | R1 | C:model:551-552 | N:neural:306, 426 / 484 |
| `reward_token_emb.weight` | [657,512] | rank | R3 | C:model:560 | N:neural:314-319, 498 |
| `reward_norm.{weight,bias}` | [512] ×2 | rank | R4 | C:model:563-567 | N:neural:497 |
| `z_gamma.0.{weight,bias}` | [512,512], [512] | rank | R5 | C:model:568 | N:neural:321, 509 |
| `z_proj.{weight,bias}` | [512,512], [512] | rank | R5 | C:model:569 | N:neural:322, 510 |
| `reward_heads.{0..3}.0.*` / `.3.*` | [512,1536], [512] / [1,512], [1] | rank | R7 | C:model:571-585 | N:neural:323-332, 514-521 |
| `runtime_head.0.*` / `.2.*` | [512,1024], [512] / [1,512], [1] | rank | R8 | C:model:586-596 | N:neural:333, 522-529 |
| `stability_head.0.*` / `.2.*` | [512,1536] / [1,512] | training_only | N1 (native only) | loaded, never read | N:neural:334, 458 |
| `reward_head_ablation.0.*` / `.3.*` | [512,512] / [1,512] | training_only | N2 (native only) | loaded, never read | N:neural:335, 459 |
| `loss_log_vars.*` (7 scalars) | [] | training_only | none in any forward | loaded, never read | N:neural:293-303 |

**Role counts check out.** `decode` 86 = 8 + 2 + 1 + 1 + 6×12 + 2. `rank` 35 = 8 + 1 + 2 + 2 + 2 + 16 + 4. `training_only` 15 = 7 + 4 + 4.

- `token_emb` rows 401..656 and `reward_token_emb` rows 401..656 never affect any output in either backend. Input ids must be < 401 (C:model:417; N:neural:420, 478), and the tied projection is narrowed to 401 (O3).
- Candle refuses a missing, extra, reshaped or non-F32 tensor (C:model:157-203). Native `vs.load` fills a `VarStore` whose init values (padding-row zeroing, layer copying, N:neural:339-371) are overwritten by the loaded checkpoint.

---

## 3. Decoder (logits) path, operation by operation

### 3.1 Input validation and context input (E0–E1)

**E0 tokens.**

- Rows must be rectangular, with 2 ≤ T ≤ `max_seq` and every id in [0, V). Candle: C:model:406-426. Native: N:neural:415-422.
- Native also takes a caller-supplied boolean `pad` of shape [B,T]: **true means present / non-PAD** (N:neural:423).
- Candle takes no mask. Presence is derived from `id != PAD(0)` (M1).
- Every production native caller passes `tokens.ne(0)` (N:proposals:794; N:pranking:182; N:ranked:304, 429; N:training:1309; monitoring.rs:168).

**E0 context.**

- One 2048-value f32 vector, which must be finite. If `target_keep_dim < target_dim`, the dropped target entries must be exactly 0.
- Candle: C:model:346-364. Native: N:neural:374-397.
- Candle accepts **one** context per call. Native accepts [B,1536] + [B,512], one context per row; callers repeat the single context (N:proposals:760-763; N:pranking:183).
- The context bytes are 8192-byte little-endian f32 files (N:proposals:253-261; C:lib:266-279).
- Composition of the 2048 values (Hyperion/Foundation/Target per `W02-model-encoder-semantics.tasks.md:17-22`) is upstream and **not traced here**.

**E1 input scaling.**

- x = [dataset / `dataset_rms` ‖ target[:, :`target_keep_dim`] / `target_rms`], shape [1 or B, 2048].
- Candle divides in f64 and rounds to f32 once (C:model:370-381). Native divides an f32 tensor by an f64 scalar in ATen (N:neural:398-404).
- With the accepted rms = 1.0 this is an identity in both.

### 3.2 Context encoder `encoder.*` (E2–E8)

| Op | Computation | Output shape |
|---|---|---|
| E2 | Linear 2048→512: y = x·W₀ᵀ + b₀ | [·,512] |
| E3 | GELU, exact erf form. Candle `gelu_erf` (C:model:387); native `gelu("none")` (N:neural:102) | [·,512] |
| E4 | LayerNorm `encoder.2`, affine, eps 1e-5 | [·,512] |
| E5 | **Native train only:** dropout p = `config.dropout` (0.1) (N:neural:104). Absent in Candle. Identity when `train=false`. | [·,512] |
| E6 | Linear 512→512 `encoder.4` | [·,512] |
| E7 | GELU (erf) | [·,512] |
| E8 | LayerNorm `encoder.6` → z | [·,512] |

- There is no dropout after E8.
- Candle computes z once as [1,512] (C:model:439). Native computes z per row as [B,512] (N:neural:425).
- Native `forward` also computes the reward encoder rz (N:neural:426) during every decoder call. The result is discarded in decode.

### 3.3 Prefix construction (P1–P4)

| Op | Computation | Shape | Candle | Native |
|---|---|---|---|---|
| P1 | c = ctx_proj(z) = z·Wᵀ + b; placed as **position 0** | [B,1,512] | C:model:440-442 (broadcast from [1,1,512]) | N:neural:429 (`unsqueeze(1)`) |
| P2 | tok = `token_emb.weight`[ids]; plain row gather, **no √W scaling**, no embedding dropout | [B,T,512] | C:model:428-433, 443 | N:neural:430. tch `Embedding` forward is `Tensor::embedding(ws, xs, padding_idx=0, …)` (`tch-0.26.0/src/nn/sparse.rs:45-53`); `padding_idx` affects gradients only (PyTorch semantics, not re-read in ATen) |
| P3 | h = concat_seq[c, tok] | [B,S,512] | C:model:449 | N:neural:427-432 |
| P4 | h = h + `pos_emb.weight`[0..S) | [B,S,512] | C:model:444-449 (`narrow(0,0,S)`) | N:neural:433-436 (`arange(T+1)`) |

**Position indexing.** Position 0 is the context. Token index t (0-based) sits at position t+1 and receives `pos_emb[t+1]`. Because S = T+1 ≤ 129, rows 0..T of the 129-row table are read.

### 3.4 Attention mask (M1–M2)

**M1 presence.** present[b,0] = true (the context). present[b,k] = token[b,k−1] ≠ PAD for k ≥ 1.

- Candle: C:model:455 (full path), C:cached:103-110 (prefill).
- Native: `cat([ones(B,1), pad])` (N:neural:437-443).

**M2 allowed.** allowed[b,q,k] = (k ≤ q) ∧ present[b,k], shape [B,1,S,S], broadcast over the 4 heads.

- Candle: additive mask, 0 where allowed and −∞ otherwise (C:model:450-462).
- Native: boolean mask, true = attend: `ones(S,S).tril(0)` AND present (N:neural:444-448). It is passed to SDPA as `attn_mask` with `is_causal=false` (N:neural:155-164).
- Key 0 is allowed for every query, so no query row is ever fully masked in either backend. The −∞/NaN edge case cannot arise.
- A PAD **query** position is still computed. Its own key is hidden, so it attends to the context and to earlier present tokens only.

### 3.5 Decoder layers ×6, pre-norm (L1–L9, i = 0..5)

| Op | Computation | Shape |
|---|---|---|
| L1 | a = LayerNorm(`norm1`)(h), eps 1e-5 | [B,S,512] |
| L2 | qkv = a·`in_proj_weight`ᵀ + `in_proj_bias` | [B,S,1536] |
| L3 | q, k, v = columns [0,512), [512,1024), [1024,1536). Each is reshaped to [B,S,4,128] and transposed to [B,4,S,128]. Head j uses columns j·128..(j+1)·128 within each block | [B,4,S,128] each |
| L4 | scores = (q·kᵀ)·(1/√128). Masked by M2, softmax over keys. **Native train only:** attention-weight dropout p = 0.1 inside SDPA | [B,4,S,S] |
| L5 | o = weights·v, transposed/reshaped to [B,S,512] (heads concatenated in head order) | [B,S,512] |
| L6 | o = `out_proj`(o). **Native train only:** dropout 0.1 | [B,S,512] |
| L7 | h = h + o | [B,S,512] |
| L8 | f = `linear2`(GELU_erf(`linear1`(LayerNorm(`norm2`)(h)))). **Native train only:** dropout 0.1 after GELU and after `linear2` | [B,S,2048] → [B,S,512] |
| L9 | h = h + f | [B,S,512] |

- Candle anchors: C:model:469-484 (layer) and 486-511 (attention: split 495-502, scale 503, mask 504, softmax 505, merge 506-510).
- Native anchors: N:neural:150-177. The call is `scaled_dot_product_attention(q, k, v, Some(allowed), train?dropout:0.0, is_causal=false, scale=None, enable_gqa=false)`. The signature is at `tch-0.26.0/src/wrappers/tensor_generated.rs:16269-16283`, and `scale=None` means 1/√128 by PyTorch default.
- There is no per-layer final norm. All 6 layers share this structure, each with its own tensors.

### 3.6 Output (O1–O4)

| Op | Computation | Shape | Candle | Native |
|---|---|---|---|---|
| O1 | LayerNorm(`out_norm`) | [B,S,512] | C:model:521 | N:neural:453-454 |
| O2 | Tied projection: logits_full = h·`token_emb.weight`ᵀ, **no bias** | [B,S,657] | C:model:522 | N:neural:455 |
| O3 | Narrow to columns [0,401) | [B,S,401] | C:model:523 | N:neural:456 |
| O4 | Next-token logits = position S−1 (= T), i.e. the last column | [B,401] | C:model:527-540 (`h.dim(1)-1`) | N:proposals:816, N:ranked:308 (`select(1, length)`) |

**Position semantics.** Logits at position p predict token index p (T:jd:635, 1957).

- Position 0 (context only) predicts token 0, which is always BOS. It is never used.
- Position T predicts the token after the row. That is the decode distribution.
- O4 is only meaningful when rows have **no trailing PAD**:
  - Candle guarantees this by removing finished rows (C:decode:148-150).
  - Native keeps finished rows, but their O4 row is only ever read under the legal set {PAD} (N:proposals:797).

---

## 4. Candle cached incremental decode (production Candle route)

Production Candle sampling uses the cache: `pure_pool_checked` → `sample_batch_cached` (C:decode:432). The full-prefix `sample_batch` is the in-crate reference. Native has **no cache**; it recomputes the full prefix for all 128 rows at every step (N:proposals:781-831).

**Prefill** (`DecodeCache::new`, C:cached:85-143):

- Runs the same P1–P4, M1–M2 and L1–L9 over the prefix.
- Before each layer it captures K and V ([B,4,S,128]) with `projected_qkv` (C:cached:51-79, 123-127), which repeats that layer's L1–L3.
- Returns O1–O3 at position T (C:cached:128-132) and sets `next_position = S` (C:cached:139).

**Advance** (C:cached:148-266), one token per surviving row:

- Refusals: `next_position ≤ max_seq` (C:cached:172-175) and token < V (C:cached:176-179).
- Rows may be dropped, but never reordered or revived. K/V and presence are gathered with `index_select` (C:cached:149-165, 185-189, 227-233).
- The new token's input is `token_emb[id] + pos_emb[next_position]` (C:cached:195-201).
- The mask is [B,1,1,next_position+1]. It hides only PAD keys, since no future keys exist (C:cached:202-221).
- Per layer: L1–L3 via `step_linear`; concat past K/V; L4–L5; L6 and L8 via `step_linear` (C:cached:223-256). `step_project` gives O1–O3 (C:cached:32-39).
- `step_matmul` flattens to [B,512]·Wᵀ, which is a different GEMM shape from the full path (C:cached:12-26).
- State commits only after all tensor work succeeds (C:cached:261-264).

**Evidence.** Unit tests compare cached and full-prefix outputs within 1e-4, **using random tiny weights only**:

- step linear and projection: C:cached:300-338
- positions to `max_seq` with interior PAD: C:cached:340-376
- row gather and refusals: C:cached:378-411
- PAD keys stay hidden: C:cached:413-448

C:decode:484-546 checks identical tokens and RNG tail between cached and full sampling (random weights, temperature 25).

---

## 5. Reward and runtime readout (R0–R9)

Candle: `PolicyModel::reward_runtime` (C:model:544-608). Native: `Policy::reward_runtime` (N:neural:466-487) and `reward_runtime_encoded` (N:neural:489-531). The native full `forward` computes the same branch (N:neural:457). The decoder is never used.

| Op | Computation | Shape | Candle | Native |
|---|---|---|---|---|
| R0 | Same token checks as E0. Native also requires a context batch equal to the token batch | — | C:model:550 | N:neural:473-482 |
| R1 | rz = `reward_encoder` applied to E1–E8 (E5 dropout only if train; always false here) | [B,512] | C:model:551-554 (one [1,512] broadcast to B) | N:neural:483-484 |
| R2 | m[b,t] = 1 if token ≠ PAD else 0 | [B,T,1] | C:model:555-559 | N:neural:496 (from `pad`) |
| R3 | mean = Σₜ `reward_token_emb`[id]·m ÷ max(Σₜ m, 1). **No positional term: order-blind** | [B,512] | C:model:560-562, 564 | N:neural:497-505 |
| R4 | g = LayerNorm(`reward_norm`)(mean), eps 1e-5 | [B,512] | C:model:563-567 | N:neural:497 |
| R5 | γ = tanh(`z_gamma.0`(rz)); π = `z_proj`(rz) | [B,512] each | C:model:568-569 | N:neural:509-510 |
| R6 | φ = [g ‖ g⊙γ ‖ \|g − π\|] | [B,1536] | C:model:570 | N:neural:506-513 |
| R7 | rᵢ = `reward_heads.i.3`(GELU_erf(`reward_heads.i.0`(φ))), i = 0..3. **Native train only:** dropout **0.05** (hard-coded, not `config.dropout`) between GELU and `.3` | [B] ×4 → [B,4] | C:model:571-585 | N:neural:81-92, 323-332, 514-521 |
| R8 | u = `runtime_head.2`(GELU_erf(`runtime_head.0`([rz ‖ g]))); runtime = softplus(u), β = 1, threshold 20 | [B] | C:model:586-603, softplus on host C:model:107-114 | N:neural:333, 522-529 |
| R9 | quality = mean(r₀..r₃) (ranking consumer) | [B] | C:ranking:416 | N:pranking:199 |

- The reward branch depends on the context (through rz) and on the program's **token multiset** (through R3). Permuting token positions leaves g unchanged. This is input for W4-T05 and is not asserted here beyond the formula.
- Scoring inputs are encoded programs with BOS first, EOS last and length 2..128, right-padded with PAD to the batch maximum. Candle: C:ranking:382-409. Native: N:pranking:148-183. PAD positions contribute nothing (R2/R3).

### 5.1 Native-only heads (not computed by Candle)

- **N1 stability** = `stability_head.2`(GELU(`stability_head.0`(φ))), shape [B]. N:neural:334, 458.
- **N2 ablation** = `reward_head_ablation.3`(GELU(`reward_head_ablation.0`(g))), shape [B], with dropout 0.05 in train. N:neural:335, 459.
- Both are computed on every native `forward`, including every decode step, where the result is discarded (N:proposals:808).
- Neither is computed by native `reward_runtime` or by any Candle path. `tensors.json` labels them `training_only`.

---

## 6. Sequence semantics

### BOS, EOS and PAD ids

- PAD = 0, BOS = 1, EOS = 2 (C:grammar:19-21). Candle validates these against the vocabulary strings "PAD", "BOS" and "EOS" (C:grammar:321-328).
- Native uses the literals 1 and 2 (N:proposals:704, 772, 841).

### Decode prefix and termination

- Every decode row starts `[BOS, featmath engine id, regression task id]`. Candle: C:grammar:399-401; the automaton is advanced past it (C:grammar:404-409). Native: N:proposals:772-780.
- The prefix is 3 tokens, so the first forward call already has T ≥ 2. Neither backend accepts a 1-token row.
- A row finishes when the automaton accepts EOS. Afterwards:
  - Candle writes PAD without sampling and drops the row from the batch (C:decode:148-150, 178).
  - Native keeps the row, gives it legal set {PAD}, and draws PAD with probability 1 (N:proposals:797, 826-831).
- Program extraction truncates after the first EOS. Candle: C:decode:189-202. Native: N:proposals:834-844.

### Target shifting (training; native only)

- `logits[:, 1..T)` are paired with targets `tokens[:, 1..T)`. The legal masks are the sets offered before each target, and the loss is masked by `pad[:, 1..T)` (N:training:1336-1342).
- Position 0 (would predict BOS) and position T (after the last token) are excluded.
- `sequence_nll` blends 0.95·NLL with 0.05 uniform smoothing over the legal set, then takes a per-row mean over present targets and a weighted row mean (N:neural:534-577).
- The training window T = max row length rounded up to a multiple of 8, capped at `SEQUENCE` = 128 (N:training:1300-1301; `crates/jope-engine/src/corpus.rs:18`).
- Candle has no training path. It follows the same prediction convention in its log-probability diagnostic (T:jd:1938-1971).

### PAD participation

| Where | Behaviour | Both backends? |
|---|---|---|
| Decoder keys | Masked (M2), except key 0 | yes |
| Decoder queries | Computed; the output is never sampled or trained on | yes |
| Input embedding | PAD looks up `token_emb` row 0 plus its position row. That only affects PAD-query outputs (Candle evidence: T:jd:606-651) | yes (native PAD-row independence **not traced by a test**) |
| Output logits | Column 0 (the PAD logit) is h·`token_emb[0]`. It is never legal for unfinished rows | yes |
| Reward | Excluded from the mean (R2–R3); the count is clamped ≥ 1 | yes |
| Training loss | PAD targets excluded | native only |
| Interior PAD | Supported, masked by id | yes (never produced in production) |

### Sequence limits

- **Forward:** 2 ≤ T ≤ 128, so S ≤ 129 and `pos_emb` has 129 rows. Candle: C:model:410-413. Native: N:neural:416-422, 475-479.
- **Candle config:** `validate` also requires `max_seq > 1` (C:config:75). **Native `Config::validate` has no `max_seq` check** (N:neural:50-69).
- **Cache:** `next_position ≤ 128` (C:cached:172-175).
- **Decode budget:**
  - Candle: `limit = min(MAX_PROGRAM_TOKENS = 128, max_seq)`, looping from the prefix length up to `limit` (C:decode:146-147).
  - Native: hard-coded `for _ in 3..128`, with batch 128 and 8 rounds (N:proposals:759, 770, 781; N:ranked:274).
  - Either way at most 125 sampled tokens and a final length ≤ 128. The **largest decode forward has T = 127**, so `pos_emb` row 128 is never read while decoding.
- **Scoring:** program length ≤ `max_seq` with BOS and EOS present (C:ranking:385-391; N:pranking:151-157). Encoding is itself bounded at 128 tokens (C:grammar:627).
- **Row 128:** `pos_emb` row 128 is read only for a 128-token input. By causality it then influences only position 128, which is excluded from the training loss and not used by any consumer (read-only reasoning).

---

## 7. Dtype and inference mode

**Candle.**

- F32 is enforced for all 136 tensors (C:model:187-191). Every op runs in F32.
- Softplus and the quality mean run on the host in f32 (C:model:597-603; C:ranking:416).
- Nothing implements dropout. There is no autograd.

**Native.**

- Dropout is controlled per call by `train`. Every inference and eval caller passes `false`, and `reward_runtime` hard-codes it. SDPA `dropout_p` is 0.0 when `!train` (N:neural:160).
- Inference runs inside `tch::no_grad`.
- On CUDA every inference path is inside `Bf16::enter()`. That calls `at::autocast::set_autocast_dtype(kCUDA, kBFloat16)` and enables CUDA autocast, clearing the cast cache on scope exit (N:runtime:37-51; N:neural:1081-1096).
- Per-op dtypes under that scope are **inferred from PyTorch's documented CUDA autocast lists, not verified against the linked LibTorch**:

| Op | Expected dtype under CUDA autocast |
|---|---|
| All `linear` (E2, E6, P1, L2, L6, L8, R5, R7, R8, N1, N2) and the tied `matmul` (O2) | BF16 output |
| SDPA (L4–L5) | BF16 (q/k/v are views of the BF16 `in_proj` output) |
| `layer_norm` (E4, E8, L1, L8, O1, R4), `softmax`, `softplus`, `sum` | FP32 |
| GELU, `tanh` | follow the input (BF16 after a linear) |
| Embeddings | FP32 (FP32 weights) |
| `cat`/`stack` and residual adds | promote to the widest input. Residual stream FP32; `reward` stack of BF16 heads BF16 |

- Consequence: native CUDA logits are BF16-rounded before `.to_kind(Float)`, masking and sampling (N:proposals:815-821). Native CUDA reward heads are BF16 values (N:pranking:199-203).
- T:jd:1798-1801 and C:lib:99-101 independently state that jope-dime samples and ranks under CUDA BF16 autocast.

---

## 8. Backend differences that block parity claims until qualified

Each item below blocks a Candle↔native numeric-parity claim, or a claim of identical sampling decisions, until it is qualified. "Semantic" means the math differs; "numeric" means the math agrees but the arithmetic differs.

| ID | Operation | Candle | Native | Kind | Evidence |
|---|---|---|---|---|---|
| BD-01 | Whole forward precision on production devices | F32 on CPU | BF16 autocast on CUDA (decode is always CUDA in `rank_impl`) | numeric, large | C:model:187-191; producer bin :160; N:proposals:147, 807; N:pranking:186; N:runtime:37-51; C:lib:99-101 |
| BD-02 | Per-op dtype map under autocast | not applicable | inferred table in §7; not probed | numeric (unverified) | §7 |
| BD-03 | Attention kernel (L4–L5) | explicit q·kᵀ, then multiply by f32(1/√128) (`Tensor / f64` → `affine(1/v)`, `candle-core-0.11.0/src/tensor.rs:3044`), additive −∞ mask, own softmax (`candle-nn-0.11.0/src/ops.rs:287-330`), then ·v | fused SDPA with boolean mask; backend (math / flash / mem-efficient / cuDNN) chosen at runtime and **unverified**; internal scaling order unverified | numeric | C:model:503-505; N:neural:155-164 |
| BD-04 | LayerNorm arithmetic (all norms) | fused CPU kernel: single-pass f32 sums, var = E[x²] − μ², eps cast to f32 (`candle-nn-0.11.0/src/ops.rs:742-757`, `layer_norm.rs:116-122`) | `at::layer_norm` via tch (`tch-0.26.0/src/nn/layer_norm.rs:54-61`); kernel algorithm not re-read. eps 1e-5 in both (`layer_norm.rs:15-25`; C:model:28) | numeric; possible cancellation when \|μ\| ≫ σ | as cited |
| BD-05 | GELU (E3, E7, L8, R7, R8) | own `erf_f32` (`candle-core-0.11.0/src/op.rs:1012`) | ATen `gelu("none")` | numeric (same variant) | C:model:387, 481; N:neural:84, 102, 173 |
| BD-06 | Softplus (R8) | host scalar f32 `x > 20 ? x : ln_1p(exp x)` | ATen `softplus`; under autocast, FP32 of a BF16 input | numeric (same formula) | C:model:107-114; N:neural:529 |
| BD-07 | Quality reduction (R9) | sequential f32 ((h₀+h₁)+h₂)+h₃, then /4 | `mean_dim(1, Float)` over BF16 heads on CUDA | numeric; BF16 heads can create exact ties that change the hash tie-break order | C:ranking:416; N:pranking:199; tie rule C:ranking:301-307 |
| BD-08 | Decode route | KV cache in production, full prefix as reference; only active rows computed | full-prefix recomputation for all 128 rows every step, finished rows fed PAD | numeric (GEMM shapes and batch composition); batch invariance unverified in both | C:decode:148-177, 432; C:cached:12-30; N:proposals:781-831 |
| BD-09 | Cached vs full agreement at accepted weights | tested only with random tiny weights at 1e-4 | not applicable | unqualified | C:cached:340-376; see OBS-1 |
| BD-10 | Presence-mask source | derived from id == 0 | caller-supplied bool `pad` | semantic only if a caller passes `pad ≠ tokens.ne(0)`; **no production caller does** | C:model:455, 557; N:neural:410-411, 437-448, 496; callers in §3.1 |
| BD-11 | Context batching | one context broadcast across rows | one context per row (callers repeat) | numeric (batch-dependent kernels) / API scope | C:model:439-442, 551-554; N:proposals:760-763 |
| BD-12 | Heads computed per call | logits and reward via separate calls; never stability or ablation | `forward` computes logits, reward, runtime, stability and ablation together | none on shared outputs in exact math; a coverage gap for N1/N2 | C:model:515-608; N:neural:406-461 |
| BD-13 | Input rms scaling (E1) | f64 division, single rounding | ATen f32 tensor ÷ f64 scalar (CUDA kernel may multiply by the reciprocal; not verified) | numeric; **identity for the accepted rms = 1.0** | C:model:370-380; N:neural:398-404 |
| BD-14 | Config validation | requires `max_seq > 1` | no `max_seq` check | none for the accepted config | C:config:75; N:neural:50-69 |
| BD-15 | Decode length and batch source | `min(128, config.max_seq)`, options-driven | literals `3..128`, batch 128, 8 rounds | none for the accepted config; diverges for a challenger with `max_seq < 127` | C:decode:146; N:proposals:759, 770, 781 |
| BD-16 | Weight artifact and identity check at load | safetensors, parameter hash recomputed | `model.ot`, file hash only | identity linkage; parameter hash is the only bridge | §1 |
| BD-17 | Sampler (outside the forward, but it blocks decision parity) | ChaCha20 inverse-CDF in f64 over the legal set; finished rows draw nothing | `torch.multinomial` after `manual_seed`, CUDA generator; finished rows consume draws | semantic for token decisions | C:decode:57-89; N:proposals:758, 819-821; C:lib:102-104 |
| BD-18 | Threads and determinism | rayon `par_chunks` and Candle gemm; thread count not pinned in the traced code | 2 intra-op / 1 inter-op threads (N:neural:1346-1353) | numeric run-to-run; unverified in both | as cited |

### No difference found (read-only)

- Context-encoder structure (E2–E8, with dropout only in native train).
- `ctx_proj` placement at position 0 (P1).
- Token embedding without scaling (P2).
- Positions 0..T from `pos_emb` (P4).
- Causal and PAD-key mask semantics, key 0 always visible (M1–M2).
- Pre-norm layer order and residuals (L1, L7, L9).
- Packed QKV split order and head layout (L2–L3, L5).
- Softmax axis = keys.
- FFN shape and activation (L8).
- `out_norm` then tied, bias-free projection narrowed to 401 (O1–O3).
- Next-token position = T (O4).
- LayerNorm eps 1e-5 and affine parameters.
- GELU variant (exact erf).
- Dropout inactive in eval (identity).
- Token range check (< 401).
- Reward mean pooling with clamp ≥ 1 (R2–R3).
- `reward_norm` (R4).
- φ order [g, g⊙tanh(z_gamma(rz)), |g − z_proj(rz)|] (R5–R6).
- Reward head layer names `.0`/`.3` (R7).
- Runtime input order [rz, g] and head names `.0`/`.2` (R8).
- Softplus β/threshold (R8).

---

## 9. Observations for later W4 tasks (read-only, not executed)

- **OBS-1: `cache_max_error` cannot complete.**
  - It builds `rows = [[tokens[0]]]` and calls `DecodeCache::new` (C:model:616-617). That calls `token_tensor`, which requires `length > 1` (C:cached:90 → C:model:410-413).
  - The helper therefore returns "invalid policy token batch shape" before comparing anything.
  - Its only caller, the `inference-cuda` panel, propagates the error with `?` (`crates/jope-bench/src/panel/inference_cuda.rs:155-158`).
  - Cached-vs-full agreement at accepted weights is therefore unmeasured (BD-09).
- **OBS-2: `next_token_logits` reads the last column** (C:model:533). Callers must keep rows free of trailing PAD, which both decoders do (§3.6).
- **OBS-3: the reward branch is order-blind** (R3) in both backends, and program-dependent through the token multiset. This is for W4-T05.
- **OBS-4: stale doc claim?** C:lib:99-101 says the Candle `cuda` feature has "never been compiled", yet the bench `inference-cuda` panel loads the Candle policy on CUDA (`inference_cuda.rs:67-70`). Which statement is current was **not verified**.
- **OBS-5: existing comparison surfaces and what they cover.** None establishes Candle↔native numeric parity.

| Surface | What it compares | Tolerance |
|---|---|---|
| T:jd:1798-1936 (ignored) | Candle CPU F32 vs recorded native BF16 rewards, cross-precision diagnostic | `max_quality < 0.25` |
| N:neural:1116-1151 | tch CPU vs external fixtures | 3e-4 (no run receipt read) |
| N:ranked:395-463 and `crates/jope-engine/tests/proposal_ranking.rs:50-139, 205-248` | native `reward_runtime` vs native full `forward` | 2e-6 |
| `inference_cuda.rs:139-154` | Candle CUDA vs Candle CPU | `TOLERANCE` 3e-4, `crates/jope-bench/src/panel/inference_benchmark.rs:19` |
| C:lib:90-94 | explicitly states no numeric parity with tch is claimed | — |

---

## 10. Unverified items

1. The LibTorch build linked at runtime (`torch-sys` expects 2.13.0), and whether any `LIBTORCH` override is used.
2. SDPA backend selection on CPU and CUDA for a boolean mask, BF16/F32 and head dim 128, and its internal scale and softmax arithmetic.
3. The per-op autocast dtypes in §7. These should be confirmed by dtype probes.
4. ATen kernel arithmetic for LayerNorm, GELU, softplus and `mean_dim`. Candle CUDA kernels and GEMM reduced-precision/TF32 settings (Candle CUDA is not on the production producer path).
5. Values of `token_emb`/`reward_token_emb` row 0 and `pos_emb` row 128 in the accepted weights (not read).
6. Batch-composition invariance of per-row outputs in both backends.
7. Whether `neural::parity` or any Candle↔native fixture comparison has ever run on the accepted checkpoint (no receipts read).
8. The upstream composition of the 2048-value context (W2 prose only).
9. The Python module `python/src/jope/policy/model_v3.py`, which the bundle lists as a "reference" consumer. It was not traced, and it visibly differs (task lanes, input-block normalization). Equality with either Rust path must not be assumed.
10. Decode bookkeeping on the unmerged demo branch (`rr/w4-decoding` `b476f79aa`) may change terminal-outcome and budget behaviour. The forward (`model.rs`, `cached.rs`) is unchanged there.

---

Could not verify: the linked LibTorch version, SDPA kernel selection, per-op autocast dtypes, ATen and Candle-CUDA kernel arithmetic, the accepted PAD-row and `pos_emb` row-128 values, or any actual numeric agreement, because nothing was built or run.
