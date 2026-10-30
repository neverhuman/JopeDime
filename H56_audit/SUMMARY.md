# H56_audit (`H_AUDIT`): Whitepaper-vs-Code Audit, H00–H55 Wave Post-Mortem, and Live Flywheel Diagnosis

- **Date**: 2026-10-30
- **Branch / Worktree**: `agy/fw_audit_t544` (`/home/ubuntu/JopeDime/.agent/worktrees/agy_fw_audit`)
- **Base Commit**: `761b08e125aa34486f89c855c7975015b6373631` (`origin/main`)
- **Audit Scope**:
  1. Whitepaper (`docs/whitepaper/sections/*.tex`, `~/whitepaper_tmp_t541/`) vs. Lineage-B / CLOVER v1 (`crates/jope-policy`, `crates/jope-engine`) vs. Live Fused Self-Play Flywheel (`research/flywheel_selfplay/run_fused_closed_loop.py`, `/home/ubuntu/JopeDimeMicro/python/phase12_multimodal_model.py`, and `/home/ubuntu/jope-data/flywheel-full-20261030/` on `xbabe1`). Directly answers the `\TBD{Document the adapter's architecture, score normalization and evaluation population...}` in `docs/whitepaper/sections/06_method/flywheel_status.tex`.
  2. Past research waves `DL03–DL06` and `H00–H55` (`~/jope_plan/ns/H*_STATUS.md`, `~/jope_plan/ns/receipts/H00_H45_MASTER_INDEX.md`, `docs/research/studies/dope-neural-ranking-20261021/`, `research/flywheel_selfplay/h53/`).
  3. Ranked never-tried / under-tested interventions with preregistered test specifications.
  4. Exact `file:line` bugs in the live flywheel loop.
- **Compliance**: CPU-only (`nice -n 19`), zero GPU usage, zero access to sealed/holdout data, DOPE weight file untouched (`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`), `xbabe1` inspected strictly read-only.

---

## Executive Diagnosis: Why the Live Flywheel Plateaued at `r882`

Direct read-only inspection of the live self-play run on `xbabe1` (`/home/ubuntu/jope-data/flywheel-full-20261030/run-20261030`, rounds `775–1485`, following `/home/ubuntu/jope-data/flywheel-gp-fused/run-20261029`, rounds `1–774`) resolves all three empirical puzzles noted in the `t544` brief:

### 1. Why `DL-Val` Plateaued at `r882 = 1.1377942` with 603+ Consecutive Rejections (`r883..r1485+`)
- **Two-phase cohort history**:
  - Rounds `1–774` (`run-20261029`) trained on only **16 `DL-Train` contexts** and gated adapter admission on **8 `DL-Train` dev contexts**, accepting 133 updates up to Round 256 (`step=471`, dev utility `2.267816`) followed by 518 consecutive rejections (`r257–r774`).
  - At Round 775 (`run-20261030`), the loop warm-started from `round-774.pt` (which held `round-256` weights at `step=471`) and switched to the full cohort (`1,219 DL-Train / 91 DL-Val`). Across **711 rounds (`r775–r1485`)**, only **4 adapter updates were ever accepted**:
    1. `r782`: `accept_tie` (`1.1073773 → 1.1073773`, `step 471 → 473`)
    2. `r785`: `accept_improvement` (`1.1073773 → 1.1375953`, `step 473 → 475`, `+0.0302180`)
    3. `r859`: `accept_improvement` (`1.1375953 → 1.1377942`, `step 475 → 477`, `+0.0001989` from flipping a single context)
    4. `r882`: `accept_tie` (`1.1377942 → 1.1377942`, `step 477 → 479`)
- **2-step rollback trap**: On every rejected round (`value < old` in `DevIncumbent.decide`, `research/flywheel_selfplay/dev_incumbent.py:490-500`), `incumbent.restore(net, opt)` restores **both** the adapter weights and the AdamW optimizer state back to `r882` (`step=479`). Then `warm_restart_adapter` (`research/flywheel_selfplay/run_fused_closed_loop.py:927-929`) injects isotropic Gaussian weight noise $\mathcal{N}(0, \sigma^2 I)$ ($\sigma \in [10^{-5}, 3\times 10^{-4}]$) into all 29,211 trainable adapter parameters, overrides the learning rate (`lr` cycling in `[2.5e-5, 3e-4]`), and runs **only 2 AdamW steps** (`args.steps` hard-capped at `2` when `full=True` at `run_fused_closed_loop.py:220`). Consequently, for 603 consecutive rounds (`r883–r1485+`), the model has never moved more than 2 gradient steps away from `r882` (`step 479 → 481` every single round).
- **Loss vs. evaluation mismatch**: The training loss (`run_fused_closed_loop.py:978-982`) is dominated `10:1` by pointwise scalar regression (`F.smooth_l1_loss(pred, target)` with weight `1.0` vs. `0.1 * F.kl_div(log_softmax(-pred/0.2), softmax(-target/0.2))`), whereas `utility()` (`run_fused_closed_loop.py:460-481`) is a shift- and scale-invariant **hard `argmin`** over discrete candidate slates.

### 2. Why the Last 100 Rounds Produce Only 12 Distinct `DL-Val` Scores (and 603 Rounds Produce Only 18)
- `utility()` averages $u_c(\arg\min_i \hat{c}_{c,i})$ over the 91 fixed `DL-Val` contexts. Because `argmin` over a finite pool is piecewise constant, small parameter steps only change the score when a context's top-1 margin crosses zero.
- Across all 711 full-cohort rounds (`r775–r1485`) on `xbabe1`:
  - **72 of the 91 `DL-Val` contexts (79.1%) never change their selected `argmin` program in any round.**
  - Within the 2-step neighborhood of `r882`, almost all variation comes from **4 knife-edge contexts** in `DL-Val`:
    - `val[10]`: flips in **686 / 711** rounds ($\Delta u_c = -0.1753$ when flipped away from `r882` choice; margin $= -0.001925$ per 91-context average)
    - `val[21]`: flips in **666 / 711** rounds ($\Delta u_c = -0.1305$ when flipped; margin $= -0.001434$ per 91-context average)
    - `val[22]`: flips in **457 / 711** rounds ($\Delta u_c = +0.0742$ when flipped; margin $= +0.000815$ per 91-context average)
    - `val[18]`: flips in **233 / 711** rounds ($\Delta u_c = -0.2070$ when flipped; margin $= -0.002275$ per 91-context average)
- `r882` (`1.1377942`) was latched by winner's curse when a random perturbation happened to keep both `val[10]` and `val[21]` on their higher-utility branch while flipping `val[62]` (`+0.0001989` at `r859`). Any 2-step gradient update on `DL-Train` immediately flips `val[10]` and `val[21]`, producing a finite truth table of discrete sums:
  - Flip `[10, 21]`: `1.1344341` (**32 of the last 100 rounds**)
  - Flip `[10, 21, 22]`: `1.1352490` (**23 of the last 100 rounds**)
  - Flip `[10, 21, 22, 62]`: `1.1354388` (**13 of the last 100 rounds**)
  - Flip `[10, 18, 21, 22]`: `1.1329746` (**10 of the last 100 rounds**)
  - Flip `[10, 18, 21, 22, 62]`: `1.1331644` (**8 of the last 100 rounds**)
  - Flip `[10, 21, 62]`: `1.1346238` (**6 of the last 100 rounds**)
  - Those **6 binary flip combinations across 5 contexts account for 92 of the last 100 rounds**.

### 3. Why There Is a Persistent `~0.29` Train-vs-Val Utility Gap (`train ~1.422` vs `val ~1.134`)
- **Primary cause — Intrinsic candidate-pool distribution shift between `DL-Train` and `DL-Val` (`+0.186` to `+0.507` gap)**:
  Direct audit of the 1,219 `DL-Train` pools vs. 91 `DL-Val` pools on `xbabe1` shows that `DL-Train` has a substantially higher utility ceiling and baseline before any neural ranking:

  | Pool Statistic | `DL-Train` (1,219 contexts) | `DL-Val` (91 contexts) | Intrinsic `Train - Val` Gap |
  | --- | ---: | ---: | ---: |
  | Base pool oracle (`max(u[:1152])`) | `1.872969` | `1.399017` | **`+0.473952`** |
  | Current pool oracle (`max(u)` at `r1485`) | `1.905853` | `1.399017` | **`+0.506836`** |
  | Top-4 revealed seed oracle (`max(u[:4])`) | `1.483107` | `1.177432` | **`+0.305675`** |
  | Uniform random candidate (`mean(u)`) | `0.827791` | `0.642202` | **`+0.185589`** |
  | `r882` achieved `utility()` | `1.422548` (`75.9%` of base oracle) | `1.137794` (`81.3%` of base oracle) | **`+0.284754`** |

  Normalized by each split's base pool oracle, `r882` actually captures a **higher** fraction of the available oracle on `DL-Val` (`81.3%`) than on `DL-Train` (`75.9%`), and a nearly identical fraction of the `oracle - mean` headroom (`57.0%` on `DL-Train` vs. `65.5%` on `DL-Val`).
- **Secondary cause — In-sample exposure of `DL-Train`**:
  The frozen `gamenet_anchor` (`flywheel_winner_pool_v3a_q25_train_only_430.pt`), the initial `h40b` adapter (`h40b_round2_dope_self_contained_10tok.pt`), and the `r775–r882` updates were trained directly on `DL-Train`.
- **Crucial discovery — The live GP self-play loop is effectively OPEN-LOOP (`<1%` of GP promotions are ever selected by `utility()`)**:
  By Round 1485, GP search had logged **1,478 strict high-water-mark (HWM) promotions across 710 of the 1,219 `DL-Train` contexts**, lifting the `DL-Train` pool oracle from `1.872969` to `1.905853` (`+0.032884`). Yet at Round 1485, `utility(train, "train")` selects a promoted GP program (`index >= 1152`) on **only 7 out of 1,219 `DL-Train` contexts (`0.57%`)**! Why? Because of **Bug 1** (`training_indices` sorts the pool by ground-truth fitness so the promoted HWM champion is always placed at index `0` inside `revealed_indices = [0, 1, 2, 3]`, where Nadaraya-Watson kernel self-distance `d(0,0) = 0` leaks its ground-truth fitness during training; at `utility()` eval time, pools are unsorted, so `0..3` are historical seeds and promoted GP champions sit unrevealed at indices `>= 1152`) and **Bug 2** (all non-HWM GP evaluations, including every false positive and gate failure, are discarded by `promote()` and never added to `pools[c]`).

---

## 1. Paper-vs-Implementation Mismatch Table

The repository and whitepaper currently conflate **three fundamentally different systems** under the single name "EurekaZero / JopeDime flywheel":
1. **System A — Whitepaper Conceptual Architecture**: Sections 1–5 (`docs/whitepaper/sections/01_introduction.tex` through `05_price.tex`) and `~/whitepaper_tmp_t541/` (`writer-07_results.tex`, `writer-08_meaning.tex`).
2. **System B — Lineage-B Step 60,000 & CLOVER v1**: Section 6 (`docs/whitepaper/sections/06_method.tex`), `~/whitepaper_tmp_t541/src/01M674E8XA0S4ZS5BT6FVH5QKX.md`, `01M674E8YNJ677EGYWZ5ZM63BA.md`, `crates/jope-policy`, and `crates/jope-engine/src/neural.rs`.
3. **System C — Live Fused Self-Play Flywheel**: Section 6.5 (`docs/whitepaper/sections/06_method/flywheel_status.tex`), `research/flywheel_selfplay/run_fused_closed_loop.py`, `/home/ubuntu/JopeDimeMicro/python/phase12_multimodal_model.py`, and `/home/ubuntu/jope-data/flywheel-full-20261030/input/cpu_exact_fused.py` on `xbabe1`.

| Dimension | System A: Whitepaper Narrative (`01_introduction.tex` – `05_price.tex`, `writer-07/08.tex`) | System B: Lineage-B Step 60,000 / CLOVER v1 (`06_method.tex`, `crates/jope-policy`) | System C: Live Fused Self-Play Flywheel (`run_fused_closed_loop.py`, `phase12_multimodal_model.py`) | Impact of Mismatch |
| --- | --- | --- | --- | --- |
| **1. Neural Writer ($p_\theta(w \mid p)$)** | 8B–1.5T (or 32B) autoregressive LLM emitting `1,200–5,000` tokens per program with chain-of-thought reasoning tokens $t_{\text{think}}$ (`02_anatomy.tex:16-30`, `05_price.tex:54-69`). | **Lineage-B**: 27.8M-param decoder transformer (6 layers, $d=512$, 401-token DSL vocab, max 128 tokens, 0 reasoning tokens). **CLOVER v1**: 893,957-param 2-layer transformer constrained to a fixed trie of 6,640 teacher programs (cannot compose new programs). | **None (0 neural writer parameters).** All candidate programs are generated by non-neural heuristic AST mutations (`EditZero`), 12D Landmark-V2 nearest-neighbor lookup, and random grammar sampling (`rust_evaluator.py:804-899`). | The live flywheel (`fig_fw_*` in Section 6.5) does not train or run a neural program writer at all; it trains a 29K-param cost-ranking adapter over a fixed/GP pool. |
| **2. Compiler-in-the-Loop Repair** | Iterative compiler diagnostic self-repair loop costing $\mathbb{E}[r]\,C_{\text{repair}}$ per attempt (`05_price.tex:20-33`). | **None.** Prefix-automaton grammar mask prevents syntax errors during decoding (`crates/jope-policy/src/grammar.rs`); 0 repair turns. | **None.** AST mutations operate directly on parsed DSL trees; invalid fits fail the Rust CV gate (`fitness=100.0`) with 0 repair turns. | The repair cost term $\mathbb{E}[r]\,C_{\text{repair}}$ in Section 5 (`eq:attempt-cost`) has no counterpart in either codebase. |
| **3. Problem Context Encoder ($f_\psi(p)$)** | Learned multimodal encoder mapping raw problem/dataset to a shared latent manifold (`02_anatomy.tex:31-45`, `04_game.tex:55-68`). | **Lineage-B**: Frozen 2,048-D concatenation (`512` Hyperion + `1,024` TabICL-v2 + `512` raw target windows). **CLOVER v1**: Frozen 2,051-D Kumo-Tabular Small vector. Neither uses DOPE (`512-D`, optional/unqualified in `06_method.tex:32-34`). | Frozen DOPE model (`444,168` params, SHA `c51a56d0...`) + frozen `gamenet_anchor` (`520,785` params) + frozen bridges (`161,824` params). **However, `predict()` passes `ctx_feat = torch.zeros((b, 584))` (`run_fused_closed_loop.py:445`)**, zeroing out the 584D context vector to the anchor! | Section 6.1 describes a 2,048-D Hyperion+TabICL encoder, while Section 6.5 plots a loop using frozen DOPE + zeroed 584D `ctx_feat`. |
| **4. Map / Value Network ($v_\psi$) & Steering** | Contrastive InfoNCE embedding $v_\psi(z_p, z_w) = \text{sim}(f_\psi(p), g_\psi(w))$ that screens out stillborn/losing programs and steers the writer before execution (`02_anatomy.tex:35-48`, `04_game.tex:55-82`). | Order-blind mean token embeddings (`06_method.tex:72-85`) trained only on lists of $\ge 8$ valid scored programs (stillborn programs never enter $G_p$). **Shares no weights with the writer and never screens or steers writer draws** (`06_method.tex:83-85`). | `FlywheelBlockFormerAdapterNet` (`adapter_10tok`, `29,211` trainable params, of which `5,569` in `parent_delta_net` are disconnected) predicts scalar cost on candidate slates. Ranks 8 (or 16) heuristic GP proposals to pick 2 for Rust evaluation (`run_fused_closed_loop.py:1201-1215`). | In System B, the Map is completely disconnected from the Writer; in System C, there is no neural Writer, and the adapter only ranks 2 of 4 GP proposals per context. |
| **5. Training Objective & Reward** | 3-tier RL/distillation reward (`04_game/sources/01M674MYAMRY2QAFM8E798AWH3.txt:67-93`): $+1 + \alpha(\text{incumbent}-\text{score}) - \lambda_{\text{comp}}C(w)$ for new record; $-1 + \sigma(\kappa\Delta) + \lambda_{\text{div}}D(w) - \lambda_{\text{comp}}C(w)$ for valid also-ran; $-R_{\text{fail}}$ for stillborn. | **Writer**: Supervised cross-entropy on tournament winners ($w^\star$) + $10^{-3}$ entropy bonus. **Value head**: $L_{\text{list}}(\tau=0.25) + 0.5 L_{\text{BT}} + 0.1 L_{\text{rt}}$ (`06_method.tex:56-82`). | `F.smooth_l1_loss(pred, target) + 0.1 * F.kl_div(log_softmax(-pred/0.2), softmax(-target/0.2))` (`run_fused_closed_loop.py:978-982`), where `target = -clip(u, -2, 4)`. No complexity penalty, no diversity bonus, no RL policy gradient. | Pointwise `smooth_l1_loss` dominates `10:1` over listwise KL, causing gradient steps to fit global offsets and `-2.0` failure cliffs at the expense of top-1 ranking. |
| **6. Revealed Trajectory / In-Context Conditioning** | Not present in paper equations; inference is described as zero-shot conditioning on problem $p$ (`02_anatomy.tex:49-58`). | Zero-shot conditioning on the 2,048-D problem context vector (`crates/jope-policy/src/model.rs`). | `predict()` **always passes `revealed_indices = [0, 1, 2, 3]` with ground-truth `fitness[:, :4]` and `gate[:, :4]`** into Nadaraya-Watson trajectory kernels (`run_fused_closed_loop.py:432-455`), and `utility()` allows `argmin` to select indices `0..3` without masking (`line 468`). | Leaks ground-truth labels of indices `0..3` into `total_cost[0..3]` via self-distance $d(i,i)=0$, causing `utility()` to pick indices `0..3` on **28.8%** of contexts (`82×` over-representation) and creating a fatal train/eval ordering mismatch. |
| **7. Referee Evaluation Protocol** | $K=5$-fold stratified/grouped cross-validation plus latency/memory/leakage gates (`04_game.tex:24-41`). | 5-fold CV in offline tournament (`crates/jope-engine`). | **2-fold inner CV** (`bootstrap.s1main --fit-s1` on `JTT1` with 10 dummy zero query rows via `training_cv_envelope`, `rust_evaluator.py:466-525`) evaluating 64 candidates/round (16 contexts $\times$ 4 candidates). | Live flywheel GP candidates are admitted on 2-fold CV (not 5-fold CV), using `1.0 / (1.0 + max(loss, 0))` converted to log-loss utility. |
| **8. Pool Admission of Evaluated Candidates** | Every valid attempt (record, also-ran, and stillborn failure) enters the training ledger so the Map learns hazard boundaries and gradients (`04_game.tex:42-82`). | Offline tournament stores all $\ge 8$ valid scored programs per problem in $G_p$; stillborn programs are excluded (`06_method.tex:73-75`). | `promote()` (`automatic_promotion.py:224-261`) discards **100% of GP evaluations that do not set a strict new all-time context HWM** (`score < incumbent - 1e-6`). All false positives and gate failures are thrown away (`run_fused_closed_loop.py:1291-1293`). | The live neural ranker receives **zero negative feedback** when it ranks a bad or gate-failing GP proposal in its top 2. |
| **9. Incumbent Ratchet & Optimizer Continuity** | Strict improvement margin $\delta > 0$ plus independent re-verification (`04_game/sources/01M674MYAMRY2QAFM8E798AWH3.txt:48-55`); neural weights advance continuously across generations. | Offline supervised training for 60,000 steps without per-round rollback (`06_method.tex:50-65`). | `DevIncumbent.decide` (`dev_incumbent.py:490-500`) gates the neural adapter on 91 `DL-Val` contexts with **zero margin (`value >= old`, accepting ties)** and **rolls back both `net` and `opt` to `r882` (`step=479`) on every rejection**, capping progress at 2 AdamW steps per attempt. | Traps the adapter in a 2-step random walk around `r882` for 603+ rounds while winner's curse on 2 knife-edge contexts (`val[10]`, `val[21]`) blocks every trajectory. |
| **10. Diversity / MAP-Elites Archive** | Behavioral niche archive (MAP-Elites) prevents mode collapse and maintains diverse stepping stones (`04_game.tex:83-105`). | None in `jope-policy`; decoding samples 64 policy drafts + 16 uniform random grammar programs (`crates/jope-policy/src/decode.rs`). | Per-context flat array `pools[c]` (1,152 initial programs + appended HWM winners). Submodular diversity acquisition exists in `phase12_multimodal_model.py:2105-2128` but is **never called** by `run_fused_closed_loop.py`. | Live GP selection (`dev_incumbent.py:607-616`) picks top-2 by cost + 2 uniform random proposals, often evaluating duplicate head tweaks. |
| **11. Deployment Budget ($B=1$ vs Search)** | Trained writer deployed at $B=1$ (single forward pass, zero test-time search) beats $B=1,000$ unguided search (`01_introduction.tex:60-78`, `05_price.tex:95-135`). | Production engine (`crates/jope-policy/src/decode.rs`) samples $8\times 128$ tokens to form **64 policy drafts + 16 random exploration programs ($B=80$)** evaluated by CV. Tied generic baseline on `3,227 / 3,268` paired searches; `98.84%` of winners were already in seeds (`src/01M674E8XA0S4ZS5BT6FVH5QKX.md`). | `utility()` (`run_fused_closed_loop.py:460-481`) scores all **1,152+ pre-existing candidates** in `pools[c]` (with indices `0..3` having their ground-truth fitness revealed!) and takes `argmin`. | Neither System B nor System C is a zero-search $B=1$ generative writer; System C is a pool re-ranker over 1,152+ cached candidates with 4 revealed ground-truth anchors. |
| **12. Section 6.5 Telemetry Table vs Actual Live Run** | Section 6.5 (`06_method/flywheel_status.tex` & `figures/data/rounds.csv`) presents Rounds `1–1332` as a single continuous self-play trajectory (`val_u` rising from `0.8129` at Round 1 to `1.1378` at Round 882). | N/A | Rounds `1–774` (`run-20261029`) and Rounds `775–1485` (`run-20261030`) are **two different runs stitched together by `extract_flywheel.py`**: Rounds `1–774` ran on 16 train / 8 dev contexts (where actual dev utility was `2.2678`), and `extract_flywheel.py` substituted `train_loss_after` or retrospective rescoring while omitting that `r775` switched cohorts and `r883–r1485` had 0 accepted adapter steps. | Readers of Section 6.5 are shown a continuous 1,332-round learning curve without disclosure of the Round 775 cohort switch (`16/8 → 1219/91`) or the 603-round rollback freeze after `r882`. |

---

## 2. Mistakes & Confounds Across Research Waves (`H00–H55`)

Across waves `DL03–DL06`, `H01–H28`, `H35–H45`, and `H46–H55`, ten structural and statistical confounds repeatedly invalidated positive claims or made valid improvements appear negative.

### 2.1 Thematic Post-Mortem of the 10 Major Confounds

1. **Contaminated Baseline (`r882` / `r785`) Trained on All 5/5 of `DL-Train` Used as the Inner-CV Reference (`H48`, `H49`, `H50`, `H51`, `H52`)**:
   - In `run-20261030`, `r785` and `r882` were trained on **all 1,219 `DL-Train` contexts** (all 205 dataset families).
   - When waves `H48–H52` constructed a 5-fold family-grouped inner CV on `DL-Train` (training on 4/5 of families and evaluating on the held-out 1/5 fold), they measured $\Delta u = u(\text{candidate}) - u(r882)$ on the held-out 1/5 fold.
   - Because `r882` had already been trained on that "held-out" 1/5 fold, any gradient training on only 4/5 of the folds progressively washed out `r882`'s in-sample adaptation on the 5th fold.
   - **Direct consequence**: In `H51 STEPS` (`docs/research/studies/dope-neural-ranking-20261021/h51/README.md`), taking more gradient steps monotonically **lowered** held-out inner-CV utility relative to `r882`:
     - `k=2` (`CONTROL`): `-0.004422`
     - `k=8`: `-0.011703`
     - `k=16`: `-0.026243`
     - `k=32`: `-0.056440`
     - `k=64`: `-0.105089`
   - This was interpreted as "gradient training degrades the adapter," when in reality training on 4/5 of `DL-Train` simply moves weights away from a checkpoint (`r882`) that had been fit to the 5th fold.

2. **Overtrained Regression Control (`k=16`) + Target-Permutation Placebo Parity (`H52`)**:
   - In `H52` (`docs/research/studies/dope-neural-ranking-20261021/h52/SUMMARY.md`, `~/jope_plan/ns/H52_STATUS.md`), new ranking objectives (`L1_RANK`, `L2_GATE`, `L3_COMB`, `FOCUS`, `GATE`) were run at `k=16` steps (`lr=1e-4`) and compared against a paired `k=16` regression control (`C16`).
   - Because `C16` had already dropped by **~`-0.030`** below `r882` (and `-0.025` below `C2`, the `k=2` control) due to Confound #1 above, any objective with smaller effective gradient norm moved less far from `r882` and therefore "beat" `C16`!
   - Proof: `FOCUS` (`+0.0206` vs `C16`) and `GATE` (`+0.0251` vs `C16`) were **still worse than `r882` (`-0.0092` and `-0.0048`)**, and the **`NULL` target-permutation placebo** (trained on randomly shuffled fitness targets!) achieved virtually the **same paired gain over `C16` (`+0.0220`, LCB `+0.0042`)** simply because permuting targets canceled gradient direction and kept weights closer to `r882`.

3. **Algebraically Unpassable Family Median Gate (`median > 0.001` and `LB95 > 0.001` in `H49-SEL`, `H50`, `H51`, `H52`)**:
   - Preregistrations in `H49-SEL`, `H50`, `H51`, and `H52` required the **family-level median utility delta across all 205 `DL-Train` families** (and its 95% bootstrap lower bound) to exceed `+0.001`.
   - Because `utility()` is a hard `argmin` over discrete candidates, small or moderate weight updates only flip the `argmin` candidate on **4% to 20% of families** (e.g., `38–41 / 205` families in `H51 CONTROL`; `4–12 / 205` families in `H51 DATA100`; `20.3%` of contexts at radius $r=0.01$ in `H50-S`; `37.6%` of families in `H52 GATE`).
   - **Mathematical identity**: Whenever fewer than `103 / 205` (`50.2%`) of families change their `argmin`, at least `50%` of family deltas are **identically `0.000000`**, so the sample median across 205 families is **identically `0.000000`** (with bootstrap CI `[0.000000, 0.000000]`) regardless of how positive the mean gain is on the families that do flip! This gate defect was finally diagnosed in `H52` (`t537` replaced it with `mean_LCB95 > 0.001` and `median >= 0`).

4. **Harness-vs-Live Evaluation Mismatch (`revealed_indices`, Zeroed `ctx_feat`/`prior`, and Unmasked `0..3`)**:
   - All offline waves (`H46–H55`) imported `predict()` and `utility()` directly from `research/flywheel_selfplay/run_fused_closed_loop.py:419-481`.
   - Consequently, every offline wave inherited `predict()`'s hardcoded `revealed_indices = [0, 1, 2, 3]` leakage, zeroed `ctx_feat`/`base_prior_cost`/`sos_prior`, and the train/eval slate-sorting mismatch (`training_indices` sorting best-48 into `0..47` vs. `utility()` leaving pools in unsorted storage order).

5. **80-Look Inner-CV Selection Bias / Winner's Curse (`H46`, `H48`, `H49`)**:
   - In `H46`, checkpoint selection took the `max` across 5 arms $\times$ 8 `DL-Val` checkpoints (40 looks at `DL-Val`), producing `H46-C` step 48's `+0.000626` false positive that vanished under honest inner CV in `H48-C` (`-0.021349`).
   - In `H48` and `H49`, moving selection to inner CV still took the `max` over `5 folds × (baseline + 16 steps) = 85 candidates`, **without averaging across folds**. Every seed in `H48` picked Fold 3 (where baseline was lowest, `1.3948`) and every seed in `H49` picked Fold 1, guaranteeing a positive selected inner-CV delta (`+0.0309` in `H48-0`, `+0.0553` in `H48-PF`) that immediately lost on `DL-Val` (`-0.0063`, `-0.0213`).

6. **Pseudoreplicate Seeds (`n_eff ≈ 1` in `H46`, `H47`, `H49-0`, `H50-ESEMA`, `H51`)**:
   - In `H46` and `H47`, full-batch training over all 1,219 contexts with deterministic top-48 slate selection (`automatic_promotion.py:328`) meant different seeds only varied the random tail of the 192-slate, producing **identical `DL-Val` utilities to 6 decimal places** across seeds (`1.036767` in `H46-A` seeds 0 and 2; `1.053521` in `H46-B` seeds 0 and 2; identical across all 3 seeds in `H49-0`).
   - In `H50`, `H50-ESEMA` was **byte-identical** to `H50-ES3` (`~/jope_plan/ns/H50_STATUS.md:38`) because EMA was never applied to the evaluated candidate.
   - In `H51`, `training_indices()` used unseeded global `np.random.choice`, breaking matched-seed pairing between `CONTROL` and `DATA100`/`STEPS` (`~/jope_plan/ns/H51_STATUS.md:34, 39`).

7. **Silent Zeroed Multimodal & DOPE Features in the 584D Context Vector (`DL03–DL06`, `H01–H28`)**:
   - Documented in `/home/ubuntu/JopeDimeMicro/python/phase12_multimodal_model.py:732-752`:
     - `mm_proj` (`ctx_raw_584[:, 200:520]`, 320 dimensions — **54.8% of the 584D context vector**) was **100% zeros across all 1,780 contexts** because `dataset_builder.py:649` used a 2-level glob (`BUILD_V2_ROOT.glob('*/*')`) instead of 3-level (`*/*/*`).
     - `dope112` (`ctx_raw_584[:, 88:200]`) took `dope_full[:, :112]`, silently dropping **100% of `dope_model` (24D) and `dope_neural` (128D)**.

8. **Cached-Pool Replay Misreported as Closed-Loop Self-Play (`H39`, `H40-Fast`, `H40B`)**:
   - `H39` terminated early at `FAILED_EVAL_CAP_EXCEEDED` before its final Cycle 2 7-fold evaluation; `H40` Launch 1 hit `FAILED_WALL_CLOCK_CAP_EXCEEDED` after 8 steps and **0 native Rust fits**, and v1 of the report extrapolated a `+45.0s` breach using an uncalled `[32, 40, 40]` schedule instead of the actual `[48, 32, 32]` code path.
   - `H40-Fast` and `H40B` executed **0 native Rust fits** (pure cached-pool replay), repeatedly inspected `dl_val_71` and `dl_holdout_91` across rounds, ran 48 updates post-guard (`guard=None`), and on `stage2_20` the non-neural `N0_AST_PRIOR_ONLY` baseline (`+0.0485`) beat `SMART_SVD_FIRST_PARTY_COMPLETE` (`+0.0076`).

9. **SPQ-1 Base-Support KL Violation & Zero Optimizer Steps in Online Neural Waves (`H41–H45`)**:
   - Waves `H41–H45` (`docs/research/studies/dope-neural-ranking-20261021/fup-02/recovered/H35_H45_PERPETUAL_CLUSTER_FLYWHEEL_REPORT.md`) executed **0 adapter/encoder optimizer steps**.
   - A mask bug (`ns_mask`) misclassified standard `"crossover"` and `"transfer"` operators (`87.5%` of base prior mass) as new support, causing `max_conditional_base_kl = 1.148685` (violating the `0.35` gate by `3.28×`).
   - Both online neural arms (`H42_ONLINE_Q_HAT` and `H44_DUAL_ONLINE_COMBINED`, `+0.028847`) **underperformed the unguided control (`H41_UNGUIDED_CONTROL`, `+0.035839`)**, and the reported `QHat` Spearman $\rho = +0.3902$ was in-sample training correlation on 160 points (`p = 0.286` on 12 holdout points).

10. **Horizon Starvation in Learn-From-Zero Pilots (`H53`, `H54`)**:
    - `H53` (`research/flywheel_selfplay/h53/SUMMARY.md`) tested cold-start learning (`G0`/`R0`/`W0`) using **only 3 rounds $\times$ 1 optimizer step = 3 total gradient steps**, compared against `r882` (`479` steps), while evaluating `DL-Val` at every intermediate round (10 looks per seed). `3` gradient steps from random initialization is orders of magnitude too short to reach `r882` (`0.9317` vs `1.1378`).

### 2.2 Wave-by-Wave Summary Table (`DL03–H55`)

| Wave / Study | Stated Hypothesis / Goal | Actual Outcome | Key Mistake / Confound / Bug |
| --- | --- | --- | --- |
| **DL03–DL06** | Multimodal context + DOPE neural ranking over candidate pools. | Superseded / invalidated (`H00_H45_MASTER_INDEX.md`). | Post-hoc ensemble selection (`DL03`); missing controls (`DL04`); `mm_proj` (`320/584` dims) was 100% zeros due to `glob('*/*')` depth bug; `dope112` slice dropped all 152 dims of `dope_model` + `dope_neural`. |
| **H01–H25** | Early neural ranker & feature ablations on 850-context cohort. | Exploratory only; no sealed/holdout promotion. | Repeated reuse of exposed 850-context `TRAIN` cohort without fold-safe out-of-family isolation. |
| **H26** | Audit of historical case matrix. | Found historical high-water leakage. | Case matrix reported exposed outer-test high-water scores rather than single preregistered terminal evaluations. |
| **H27–H28** | 5-fold landmarking + neural ranker ablation (`H27`) and gate calibration (`H28`). | Inconclusive (`95% CI [-0.789, +0.287]`). | Dominated by rare catastrophic gate failures (`fitness=100.0`) on a previously exposed 850-context cohort. |
| **H35–H37** | Survivor-gated self-play candidate selection. | **Failed** (`-0.410` split-half gain, worst `-6.573`). | `G0_LEGACY_BUGGY_RAW_FITNESS` survivor gate bug selected high-variance failing candidates. |
| **H38** | Gate-repaired survivor ranking. | Local recovery on split-half, not promoted. | Still coupled to cached candidate pools without native Rust closed-loop execution. |
| **H39** | Two-cycle native Rust self-play flywheel. | **Aborted (`FAILED_EVAL_CAP_EXCEEDED`)**. | Exhausted evaluation cap before completing final Cycle 2 7-fold verification. |
| **H40 (Launch 1)** | Bounded wall-clock closed-loop flywheel. | **Aborted (`FAILED_WALL_CLOCK_CAP_EXCEEDED`, `10.86s > 10.0s`)**. | 8 steps, 0 native Rust fits; v1 report cited an uncalled `[32, 40, 40]` schedule to claim a `+45.0s` extrapolation. |
| **H40-Fast / H40B** | `FlywheelBlockFormerAdapterNet` (`h40b`, 10-token adapter) refinement. | Produced `h40b_round2_dope_self_contained_10tok.pt` seed for live run. | **0 native Rust fits** (cached-pool replay); repeated looks at `dl_val_71` and `dl_holdout_91`; 48 updates ran with `guard=None`; on `stage2_20`, non-neural `N0_AST_PRIOR_ONLY` (`+0.0485`) beat `SMART_SVD` (`+0.0076`). Track 2 Policy A leaked 11 `st_val_gate_32` datasets (77 cells) into `train_gradient`. |
| **H41–H44** | Online `QHat` (`H42`), operator bandit (`H43`), and combined (`H44`) vs unguided (`H41`). | **Failed** (`+0.028847` neural vs `+0.035839` unguided control). | **0 neural adapter/encoder optimizer steps**; `ns_mask` bug treated `"crossover"`/`"transfer"` (`87.5%` base mass) as new support (`KL = 1.149 > 0.35`); `QHat` $\rho=+0.390$ was in-sample training correlation. |
| **H45** | Multi-generation cluster flywheel ratchet. | **0 promotions** (all generations rejected by ratchet). | Inherited `H41–H44` zero-step adapter and `ns_mask` KL bug. |
| **H46** | Offline adapter tuning from `r785` (`H46-0, A, B, C, D`). | **No promotion** (`H46-C` step 48 `+0.000626` was selection noise). | Full-batch training made seeds pseudoreplicates (`H46-A` seeds 0 & 2 identical `1.036767`); `H46-C` chosen from 40 looks at `DL-Val`; `H46-A` stopped post hoc after viewing seed 0. |
| **H47** | Margin / temperature / freeze variations (`H47-A..D`). | **All negative** (`-0.0297` to `-0.0472` on `DL-Val`). | In-wave control `arm == "0"` was dead code (never run); pseudoreplicate seeds; checkpoint selection peeked at `DL-Val`. |
| **H48** | 5-fold family-grouped inner CV for `H48-0, C, LR, PF`. | **All lost on `DL-Val`** (`-0.0063` to `-0.0213`). | 80-look inner-CV selection bias (picked best `(fold, step)` pair without averaging folds; all 12 runs picked Fold 3); `r785` baseline had been trained on the held-out inner folds; 3 code bugs (`requires_grad` detached tensor, `run` shadowing, `ord("PF")`). |
| **H49** | Fold-averaged inner CV (`H49-0, WD, SWA, DO, SEL`). | **0 promotions** (`H49-SEL` selected 0 updates; `H49-SWA` `-0.0241`). | `H49-0` seeds were 100% identical (`n_eff = 1`); `H49-SWA` bug selected SWA even when worse than baseline; `H49-SEL` introduced the unpassable family median gate (`median > 0.001`). |
| **H50** | Landscape sensitivity (`H50-S`) + gradient-free ES/Coordinate search (`H50-ES1, ES3, ESEMA, COORD`). | **0/12 runs passed gate** (`median = 0.0000` on all 12). | Unpassable family median gate (`only 20.3%` of contexts flip at $r=0.01$, so median is identically `0.0`); `H50-ESEMA` was byte-identical to `H50-ES3`; `elapsed + 150 < 282` truncated search to 2 generations; `-inf` JSON crash. |
| **H51** | Matched-seed paired ablations (`CONTROL`, `DATA100`, `STEPS`, `TEMP`, `EMA`, ` SUBSET`). | **0/18 runs passed gate**; `STEPS` degraded monotonically (`k=64: -0.105`). | `r882` baseline was trained on held-out inner folds (so training on 4/5 folds washed out 5th-fold memorization); unseeded global `np.random.choice` in `training_indices` broke matched seeds; unpassable median gate. |
| **H52** | Ranking-aligned objectives (`L1_RANK`, `L2_GATE`, `L3_COMB`, `FOCUS`, `GATE`, `NULL`). | **Rejected** (`NULL` placebo `+0.0220` matched `FOCUS` `+0.0206` and `GATE` `+0.0251`). | Paired against an overtrained `k=16` control (`C16`) that had lost `-0.030` vs `r882`; `NULL` shuffled-target placebo beat `C16` by taking smaller effective steps away from `r882`; all arms still negative vs `r882`. |
| **H53** | Learn-from-zero provenance pilot (`G0`, `R0`, `W0`, `I_R882`). | `W0` reached `0.9465` (`+0.0148` over `G0`), far below `r882` (`1.1378`). | Only **3 rounds $\times$ 1 step = 3 gradient steps** total; per-round `DL-Val` curve evaluation (10 looks/seed); manifest SHA typo (`3788b...` vs `378bb...`). |
| **H54–H55** | Longer-horizon learn-from-zero (`H54`) & fold-clean objective follow-up (`H55`). | Superseded / queued behind GPU lock (`~/jope_plan/ns/H54_STATUS.md`, `H55_STATUS.md`). | Pending fold-clean initialization and removal of the `predict()` `revealed_indices` / `ctx_feat=0` bugs identified below. |

---

## 3. Never-Tried / Under-Tested Ideas Likely to Make the Flywheel Improve (Ranked by EV & Cost)

Every past wave (`H46–H53`) modified loss weights, step counts, temperatures, or ES perturbations **while calling the buggy `predict()` and `training_indices()` functions unchanged**. None of the waves ever tested fixing the train/eval `revealed_indices` mismatch, un-zeroing `ctx_feat`/`base_prior_cost`/`sos_prior`, admitting false-positive GP evaluations into `pools[c]`, or training the existing `parent_delta_net` transition head.

Below are the highest-leverage interventions, ranked by **Expected Value (EV) / Compute Cost**, each with a self-contained preregistered test specification using a **fold-clean initialization** (never comparing 4/5-fold updates against a 5/5-contaminated `r882` checkpoint) and the **`t537` gate** (`paired family bootstrap mean_LCB95 > +0.001` and `family median >= 0.0` on held-out families, with a single terminal `DL-Val` evaluation).

### Rank 1 (Highest EV, Lowest Cost — ~15 min CPU/GPU): Eliminate `revealed_indices=[0,1,2,3]` Label Leakage & Train/Eval Slate-Order Mismatch + Restore Non-Zero `ctx_feat`, `base_prior_cost`, `sos_prior`, and Per-Context `scale`
- **Mechanistic Rationale**:
  1. In `run_fused_closed_loop.py:956`, training calls `training_indices()`, which sorts the pool by ground-truth `fitness` (`automatic_promotion.py:326`) so indices `0..3` are **always the 4 best programs in the pool** (including any newly promoted GP champion at index `0`). `predict()` then passes `revealed_indices = [0, 1, 2, 3]` with their ground-truth fitness into `predict_costs()`, where Nadaraya-Watson kernel self-distance `d(i, i) = 0` leaks their exact fitness into `pred[:, 0:4]`.
  2. At `utility()` evaluation time (`run_fused_closed_loop.py:460-481`), the pool is passed in **unsorted storage order**, where `0..3` are historical seeds and promoted GP champions sit at the end of the array (`>= 1152`). Yet `utility()` still reveals `0..3` and allows `np.argmin(pred)` to pick `0..3` (`28.8%` of all selections!), while promoted GP champions at `>= 1152` are unrevealed and selected `< 1%` of the time (`7 / 710` promoted contexts).
  3. Simultaneously, `predict()` (`lines 445-450`) passes all-zero `ctx_feat (b, 584)`, all-zero `base_prior_cost (b, m)`, all-zero `sos_prior (b, m, 2)`, and constant `scale = 0.4208`.
- **Preregistered Test Spec (`H56-P1`)**:
  - **Code change**: In `predict()`, (a) either pass `revealed_indices=None` (pure prior + multimodal context ranking) OR fix `revealed_indices` to the 4 initial `G0` baseline seeds in **both** training and evaluation while **masking `revealed_indices` out of the loss and out of `np.argmin(pred)`**, and randomly permuting the remaining non-revealed candidate slots during training so `order[0]` is not deterministically at slot 0; (b) pass actual fold-safe `base_prior_cost`, `sos_prior`, non-zero `ctx_feat`, and per-context `scale` from the pool/anchor.
  - **Protocol & Budget**: 5-fold family-grouped inner CV on `DL-Train` (205 families, seed `4900`), initialized from a **fold-clean** checkpoint (`W0` warm-started only on the 4 training folds, 3 seeds $\times$ 32 steps, batch size 256). Budget: $\le 15$ min.
  - **Acceptance Gate**: Paired family bootstrap `mean_LCB95 > +0.005` and `median >= 0.0` vs. current `predict()` under the identical fold-clean protocol, plus $\ge 5\times$ increase in selection rate of promoted GP champions (`index >= 1152`) on `DL-Train`.

### Rank 2 (High EV, Zero Extra Compute Cost — Live Loop Fix): Admit All Evaluated GP Candidates (Especially False Positives & Gate Failures) into `pools[c]` as Hard Negatives
- **Mechanistic Rationale**:
  - Each live round runs 64 native Rust 2-fold CV evaluations (`run_fused_closed_loop.py:1253-1255`), including 32 proposals chosen specifically because the neural adapter predicted they had the lowest cost (`gp_selection_indices`, `dev_incumbent.py:607-616`).
  - When those neural top-picks fail the CV gate (`fitness=100.0`) or perform poorly (`score >= incumbent - 1e-6`), `promote()` (`automatic_promotion.py:224-261`) returns `None` and **discards them completely**. They are never appended to `pools[c]`. Thus the ranker never learns from its own false positives, and in rounds with 0 HWM promotions (`>50%` of recent rounds), `pools[c]` is completely static.
- **Preregistered Test Spec (`H56-P2`)**:
  - **Code change**: Decouple **pool memory admission** (`append_evaluated_candidate(pool, result)` for all 64 completed Rust evaluations per round, tagging neural top-picks that scored poorly as hard negatives) from **HWM incumbent promotion** (`promote()`). Update `training_indices()` (`automatic_promotion.py:323-333`) to reserve 32 of the 192 slate slots for recently evaluated neural false positives (`predicted_rank <= 2` and `actual_u < median`).
  - **Protocol & Budget**: 20 closed-loop rounds on 5 fold-clean inner splits (3 seeds), replaying historical + live GP evaluations. Budget: $\le 20$ min CPU.
  - **Acceptance Gate**: `> 30%` reduction in repeated top-2 selection of gate-failing AST templates across rounds and paired family `mean_LCB95 > +0.002` on held-out inner folds.

### Rank 3 (High EV, Low Cost — ~15 min): Replace `smooth_l1_loss`-Dominated Objective with Shift-Invariant Top-Focused Plackett-Luce / ListNet + Explicit Gate-Failure Margin on Fold-Clean Initialization
- **Mechanistic Rationale**:
  - In `run_fused_closed_loop.py:978-982`, `F.smooth_l1_loss(pred, target)` has weight `1.0` while `F.kl_div(log_softmax(-pred/0.2), softmax(-target/0.2))` has weight `0.1`. Because gate-failing candidates have `u = -2.0` (`target = +2.0`) while passing candidates cluster near `u ∈ [0.8, 1.8]`, `smooth_l1_loss` forces the 29K-param adapter to spend its capacity fitting global per-context offsets and `-2.0` failure cliffs. In `H46-A/B`, this caused training loss to drop (`3.69 → 3.36`) while **both** `DL-Train` and `DL-Val` `argmin` utilities dropped by `-0.07` to `-0.10`.
  - `H52` attempted to test ranking losses (`FOCUS`, `GATE`), but was confounded because it initialized from 5/5-contaminated `r882` and compared against an overtrained `k=16` control (`C16`).
- **Preregistered Test Spec (`H56-P3`)**:
  - **Code change**: Train from a **fold-clean** initialization (`G0` or fold-clean `W0`, never `r882`) with `H56-P1` applied, comparing (A) current `1.0 * smooth_l1 + 0.1 * kl(τ=0.2)` vs. (B) shift-invariant dual-temperature ListNet on passing candidates (`0.7 * kl(τ=0.05) + 0.3 * kl(τ=0.25)`) + pairwise hinge barrier $\max(0, 0.5 + \hat{c}_{\text{pass}} - \hat{c}_{\text{fail}})$ on gate failures + tiny anchor regularization $10^{-3} \|\hat{c}\|_2^2$.
  - **Protocol & Budget**: 5-fold family-grouped inner CV on `DL-Train` (3 seeds, `k=32` steps, evaluated at a single preregistered terminal step with EMA $\beta=0.95$). Budget: $\le 15$ min.
  - **Acceptance Gate**: Positive movement on **both** `DL-Train` ($\Delta u_{\text{train}} > +0.015$) and held-out inner folds (`mean_LCB95 > +0.003`, `median >= 0.0`), AND beating a target-permutation `NULL` placebo by `mean_LCB95 > +0.005`.

### Rank 4 (High EV, Low Cost — ~15 min): Replace Single-Round 91-Context Hard-Argmin Ratchet (`value >= old` + Restore-on-Reject + 2-Step Random Walk) with Persistent Multi-Step EMA Trajectory + Split-Dev LCB Gate
- **Mechanistic Rationale**:
  - Restoring `net` and `opt` to `r882` after every 2-step rejected round (`dev_incumbent.py:490-500`) and injecting random Gaussian noise (`warm_restart_adapter`) makes multi-step representation learning impossible. Meanwhile, accepting ties (`value >= old`) on raw 91-context `argmin` latched a 2-context winner's-curse spike (`val[10]` and `val[21]`) at `r882`, blocking 603+ subsequent rounds.
- **Preregistered Test Spec (`H56-P4`)**:
  - **Code change**: (1) Freeze or remove dead `parent_delta_net` (`5,569` params receiving weight noise with zero gradient); (2) remove `warm_restart_adapter` Gaussian weight noise; (3) maintain a persistent training state across rounds ($k=8$ steps/round, AdamW with cosine LR, and Polyak/EMA shadow weights $\theta_{\text{EMA}} \leftarrow 0.98\,\theta_{\text{EMA}} + 0.02\,\theta$); (4) gate promotion of $\theta_{\text{EMA}}$ to serving incumbent only every $M=8$ rounds using a paired family-bootstrap LCB gate (`mean_LCB95 > +0.0005` on a smoothed top-$K$ soft-argmin surrogate + hard-argmin non-inferiority `median >= 0.0`), without resetting the training worker's optimizer on rejection.
  - **Protocol & Budget**: 5-fold inner-CV simulation over 32 rounds (3 seeds). Budget: $\le 15$ min.
  - **Acceptance Gate**: Monotonic improvement of EMA shadow weights across rounds (`R32 > R16 > R0`) and `mean_LCB95 > +0.002` on held-out inner folds.

### Rank 5 (Medium-High EV, Medium Cost — ~30 min): Train and Wire the Existing `parent_delta_net` (`predict_parent_child_delta`) to Steer `EditZero` GP Mutations
- **Mechanistic Rationale**:
  - `FlywheelBlockFormerAdapterNet` (`/home/ubuntu/JopeDimeMicro/python/phase12_multimodal_model.py:1198-1207, 1597-1640`) already contains a 5,569-parameter directional mutation head (`parent_delta_net` / `predict_parent_child_delta`), and `research/flywheel_selfplay/model.py:417-514` already implements `DirectionalEvoLoss`.
  - Meanwhile, `xbabe1` has already accumulated thousands of exact `(parent_ast, operator, child_ast, parent_fitness, child_fitness)` transitions in `round-*-scores.jsonl` and `tree-shards/trees-round-*.jsonl.zst`.
  - Yet `run_fused_closed_loop.py` **never trains or calls `predict_parent_child_delta`**, and `propose_flywheel_candidates` (`rust_evaluator.py:804-899`) generates mutations completely blind to neural gradients!
- **Preregistered Test Spec (`H56-P5`)**:
  - **Code change**: Train `parent_delta_net` using `DirectionalEvoLoss` on `DL-Train` parent $\to$ child transitions logged in `round-*-scores.jsonl` (grouped by 5 inner family folds). During GP proposal generation (`run_fused_closed_loop.py:1162-1215`), sample 32 `EditZero` mutations of the current context HWM parent and rank them by `parent_cost - predict_parent_child_delta(parent, child, op_id)` before selecting the top 4 for Rust CV.
  - **Protocol & Budget**: Evaluate on 64 `DL-Train` contexts held out by family fold (4 Rust CV fits/context $\times$ 5 rounds, `nice -n 19` CPU). Budget: $\le 30$ min CPU.
  - **Acceptance Gate**: Strict increase in GP HWM promotion rate per 64 Rust evaluations and paired context `mean_LCB95 > 0` vs. unguided `propose_flywheel_candidates`.

### Rank 6 (Medium EV, Low Cost — ~20 min): Activate Submodular Semantic + Operator Diversity Acquisition (`phase12_multimodal_model.py:2105-2128`) in GP Proposal Selection
- **Mechanistic Rationale**:
  - `gp_selection_indices` (`dev_incumbent.py:607-616`) currently picks the top 2 proposals by predicted cost + 2 uniform random proposals. When 8 proposals contain several minor parameter perturbations of the same estimator head, the top 2 picks are nearly identical (`sem24` cosine similarity $> 0.99$), wasting 50% of the exploited Rust CV budget.
  - `/home/ubuntu/JopeDimeMicro/python/phase12_multimodal_model.py:2105-2128` already implements a greedy submodular acquisition rule combining predicted cost, epistemic ensemble spread, `sem24` cosine distance (`diversity_weight=0.08`), and novel operator bonus (`operator_bonus=0.04`), but it is dead code in `run_fused_closed_loop.py`.
- **Preregistered Test Spec (`H56-P6`)**:
  - **Code change**: Replace `gp_selection_indices` in `run_fused_closed_loop.py:1212` with the submodular diversity-penalized selector over `proposals` (using `sem24` features and `edit_op` tags).
  - **Protocol & Budget**: 5-fold inner-CV replay + 32-context CPU Rust evaluation (`nice -n 19`). Budget: $\le 20$ min.
  - **Acceptance Gate**: Mean distinct terminal heads evaluated per context per round increases from $\sim 1.3$ to $\ge 2.5$, with non-inferior or higher HWM discovery rate per round.

---

## 4. Bugs in the Live Loop Logic (`file:line`)

Below are the **9 verified implementation bugs** in the live flywheel codebase, with exact file paths, line numbers, mechanistic traces, and fixes.

### Bug 1 — Ground-Truth Label Leakage in `revealed_indices = [0, 1, 2, 3]` and Train-vs-Eval Slate Ordering Mismatch
- **Locations**:
  - `research/flywheel_selfplay/run_fused_closed_loop.py:419-457` (`predict`)
  - `research/flywheel_selfplay/run_fused_closed_loop.py:460-481` (`utility`)
  - `research/flywheel_selfplay/run_fused_closed_loop.py:956, 974-982` (training loop)
  - `research/flywheel_selfplay/automatic_promotion.py:323-333` (`training_indices`)
  - `/home/ubuntu/JopeDimeMicro/python/phase12_multimodal_model.py:1383-1400` (`FlywheelBlockFormerAdapterNet.predict_costs`)
  - `research/flywheel_selfplay/model.py:290-335` (`FlywheelGameNet.predict_costs`)
- **Mechanism**:
  1. `predict()` (`run_fused_closed_loop.py:432-455`) unconditionally sets `revealed = np.asarray([0, 1, 2, 3])` and passes ground-truth `revealed_raw_f = fitness[:, :4]` and `revealed_gate = gate[:, :4]` into `net.adapter_10tok.predict_costs(...)`.
  2. Inside `predict_costs()`, the Nadaraya-Watson kernel computes distances between all candidates $i \in \{0,\dots,m-1\}$ and the revealed candidates $j \in \{0,1,2,3\}$. For $i \in \{0,1,2,3\}$, the self-distance $d(i,i) = 0$ gives maximal kernel weight to candidate $i$'s own ground-truth residual $r_i = (f_i/\text{scale}) - \text{eff\_prior}_i$ and ground-truth gate failure, leaking the target directly into `pred[:, 0:4]`.
  3. In `utility()` (`run_fused_closed_loop.py:467-468`), `np.argmin(pred)` is taken over **all** indices `0..len(pids)-1` without masking out `0..3` (unlike GP proposal ranking at `line 1207`, which slices `[0, 4:]`). Empirical check on `xbabe1` (`round-1485.json`): `utility()` selects an index in `0..3` on **26 / 91 (`28.6%`) of `DL-Val` contexts** and **351 / 1,219 (`28.8%`) of `DL-Train` contexts** — an **`82×` over-representation** relative to `4 / 1,152` (`0.35%`).
  4. Worse, during training (`run_fused_closed_loop.py:956`), `training_indices()` (`automatic_promotion.py:326-332`) sorts the pool by ground-truth `fitness` (`order = np.argsort(pool["fitness"])`) and puts `order[:48]` first! Thus during training, indices `0..3` are **always the 4 best programs in the pool** (including any newly promoted GP champion at `order[0]`), and their ground-truth fitness is revealed to the kernel while the loss (`lines 978-982`) is computed across all 192 slots including `0..3`. At `utility()` evaluation time, however, `predict()` is called with `np.arange(len(p["pids"]))` (unsorted storage order), where `0..3` are initial seed programs and promoted GP champions sit at the end of the array (`>= 1152`, unrevealed). As a result, the model selects promoted GP champions on only **7 out of 710** promoted `DL-Train` contexts (`< 1%`)!

### Bug 2 — `promote()` Discards All Non-HWM GP Evaluations (Zero Negative Feedback on False Positives & Gate Failures)
- **Locations**:
  - `research/flywheel_selfplay/run_fused_closed_loop.py:1291-1306`
  - `research/flywheel_selfplay/automatic_promotion.py:224-261` (`promote`) and `264-289` (`apply_event`)
- **Mechanism**:
  - After evaluating 64 new GP candidates in Rust (`run_fused_closed_loop.py:1253-1255`), the loop calls `event = promote(p, r, journal, rnd)` (`line 1291`) and only updates `pools[c]` via `apply_event(p, event)` when `event is not None`.
  - In `automatic_promotion.py:239-243`, `promote()` returns `None` whenever `cv_gate_passed is not True` OR `score >= incumbent - 1e-6`.
  - Consequently, every GP candidate that the neural network ranked in its top 2 (`gp_selection_indices`) which subsequently fails the Rust CV gate (`fitness=100.0`) or fails to beat the all-time context HWM is **never added to `pools[c]`**. The neural ranker never trains on its own mistakes, and in rounds with 0 HWM promotions, the training pools `pools[c]` are 100% unchanged from the previous round.

### Bug 3 — All-Zero `ctx_feat`, `base_prior_cost`, and `sos_prior` + Constant `scale=0.4208` Passed to `predict_costs`
- **Locations**:
  - `research/flywheel_selfplay/run_fused_closed_loop.py:444-450` (`predict`)
  - `/home/ubuntu/JopeDimeMicro/python/phase12_multimodal_model.py:1244-1315` (`FlywheelBlockFormerAdapterNet.encode_candidates`)
- **Mechanism**:
  - `predict()` calls `net.adapter_10tok.predict_costs(...)` with:
    - `ctx_feat=torch.zeros((b, 584), dtype=torch.float32, device=device)` (`line 445`)
    - `base_prior_cost=torch.zeros((b, m), dtype=torch.float32, device=device)` (`line 448`)
    - `sos_prior=torch.zeros((b, m, 2), dtype=torch.float32, device=device)` (`line 449`)
    - `scale=torch.full((b,), 0.4208, dtype=torch.float32, device=device)` (`line 450`)
  - Zeroing `ctx_feat`, `base_prior_cost`, and `sos_prior` blinds the frozen `FlywheelGameNet` anchor (`self.anchor.encode_candidates`) to the 584D dataset context features, the calibrated AST structural prior (`FoldSafeCalibratedPriorEstimator`), and the Leave-One-Family-Out Empirical-Bayes SoS prior, while feeding zeros into `cand_query_proj` (`torch.cat([sem24_rel, sos_prior, base_prior_cost], dim=-1)`) and `mm_prior_out` (`phase12_multimodal_model.py:1267, 1286`).

### Bug 4 — Dead Trainable Submodule (`parent_delta_net`, 5,569 Params / 19.1% of Adapter) Receives Warm-Restart Noise and Weight Decay with Zero Loss Gradient
- **Locations**:
  - `research/flywheel_selfplay/run_fused_closed_loop.py:316-318, 444-458, 927-929, 994-1001, 1029, 1043`
  - `/home/ubuntu/JopeDimeMicro/python/phase12_multimodal_model.py:1200-1207, 1597-1640`
- **Mechanism**:
  - `run_fused_closed_loop.py:316-318` sets `requires_grad_(True)` on every parameter in `net.adapter_10tok` whose name does not start with `"anchor."` (`29,211` parameters total).
  - `5,569` of those `29,211` parameters (`19.1%`) belong to `parent_delta_net` (`phase12_multimodal_model.py:1200-1207`), which is only used by `predict_parent_child_delta()` (`line 1597`).
  - `predict()` (`run_fused_closed_loop.py:444-458`) only calls `predict_costs()`, so `parent_delta_net.grad` is always `None` / `0`.
  - Yet every round, `warm_restart_adapter()` (`line 927`, `dev_incumbent.py:681-728`) injects Gaussian noise `torch.randn_like(param) * std` into `parent_delta_net`, and `opt.step()` (`line 997`) applies AdamW weight decay to it, corrupting `parameter_update_l2` (`line 1029`) and `candidate_parameter_sha256` (`line 1043`).

### Bug 5 — `full` Mode Overwrites `proposal_plan` Schedule, Clobbers LR Twice per Round, and Traps Optimizer in a 2-Step Rollback Loop
- **Locations**:
  - `research/flywheel_selfplay/run_fused_closed_loop.py:220-223, 874-888, 926-930`
  - `research/flywheel_selfplay/dev_incumbent.py:488-500, 620-666, 681-728`
- **Mechanism**:
  1. `run_fused_closed_loop.py:220` forces `if full and args.steps > 2: args.steps = 2`.
  2. At `lines 875-878`, `if full:` overwrites the adaptive `proposal_plan` schedule (`dev_incumbent.py:620-666`, which was designed to vary `steps` in `[2, 4, 8]` and decay `lr_factor`) with `plan["steps"] = args.steps` (`2`) and `plan["lr_factor"] = 1.0`.
  3. At `lines 886-887`, the loop updates `group["lr"] = max(plan["minimum_lr"], group["lr"] * plan["lr_factor"])`, and then **40 lines later (`line 927`) immediately overwrites `group["lr"]` again** inside `warm_restart_adapter(net, opt, plan["round_seed"], rnd)`!
  4. When the 2-step update fails to beat or tie `r882` on the 91 `DL-Val` contexts, `incumbent.decide()` (`dev_incumbent.py:490-500`) calls `self.restore(net, opt)`, resetting both weights and AdamW momentum/variance buffers back to `step=479`. And when a random perturbation ties or nudges 1 context (`value >= old` at `line 490`, with zero margin $\delta=0$), it permanently ratchets the threshold upward via winner's curse.

### Bug 6 — Unaligned `temporary` Pool Dict & Unsorted Seed Anchors in GP Proposal Ranking
- **Locations**:
  - `research/flywheel_selfplay/run_fused_closed_loop.py:1201-1211`
- **Mechanism**:
  - To rank the `8` (or `16`) new GP proposals for context `c`, `lines 1201-1204` construct `temporary = dict(p)` and overwrite `temporary["program"]` and `temporary["proj16"]` with arrays of length `4 + len(proposals)` (`12` or `20`), while leaving `temporary["pids"]`, `temporary["fitness"]`, and `temporary["gate"]` at the full pool length (`1,152+`).
  - Furthermore, `p["program"][:4]` and `p["fitness"][:4]` are the **first 4 unsorted entries in `pools[c]`**, NOT the 4 best programs in `pools[c]` (unlike training, where `training_indices` sorts the pool so `0..3` are the 4 best programs, including the current HWM champion).
  - Finally, `encode_candidates` (`phase12_multimodal_model.py:1260, 1289`) mean-centers `sem24` and `mm_prior_adj` across the slate dimension `m`; centering across a `12`-candidate miniature slate produces a different distribution than centering across the `192`-candidate training slate or `1,152`-candidate eval slate.

### Bug 7 — Unseeded Global `np.random.choice` in `training_indices()` + Fixed `round_seed` RNG Clobbering
- **Locations**:
  - `research/flywheel_selfplay/automatic_promotion.py:323-333` (`training_indices`)
  - `research/flywheel_selfplay/run_fused_closed_loop.py:883-885, 938-956`
- **Mechanism**:
  - `training_indices()` (`automatic_promotion.py:330`) calls global `np.random.choice(order[head:], size=limit - head, replace=False)` without accepting a local `np.random.Generator` or `step_seed`.
  - In `run_fused_closed_loop.py:938-956`, `step_seed = batch_seed(args.seed, rnd, step)` seeds a local `np.random.default_rng(step_seed)` for `cids`, **but never seeds global `np.random` per step**, so `training_indices()` draws from the global NumPy RNG state.
  - Furthermore, at `line 879`, `if full and inc_val.accepted_rounds == 0: plan["round_seed"] = int(args.seed)` resets `round_seed` to a constant whenever `accepted_rounds == 0`, clobbering `random`, `np.random`, and `torch.manual_seed` (`lines 883-885`) with the identical seed every round until the first acceptance.

### Bug 8 — In-Place Log File Truncation (`open("wb")`) in `retain_checkpoint_payloads` Without Writer FD Coordination
- **Locations**:
  - `research/flywheel_selfplay/dev_incumbent.py:228-234, 376-380` (`retain_checkpoint_payloads`)
- **Mechanism**:
  - When a `*.log` or `*.jsonl` file in `run_dir` exceeds `rotate_bytes` (`8 MiB`), `retain_checkpoint_payloads()` copies its tail into an archive and then truncates the live file by opening a brand-new file descriptor with `with (root / rel).open("wb") as stream: stream.flush()` (`dev_incumbent.py:377-379`).
  - If the parent process or redirect shell holds an open stdout/stderr file descriptor to `*.log` opened without `O_APPEND` (e.g., standard `>` redirection), the kernel write offset on that existing file descriptor remains at the old byte offset (`> 8 MiB`), so subsequent writes create a sparse hole of megabytes of `\x00` null bytes at the start of the truncated log file.

### Bug 9 — Broken Mathematical Transpiler in `transpile_dsl_to_rust_code` (`selfplay_loop.py` / `rust_evaluator.py`)
- **Locations**:
  - `research/flywheel_selfplay/rust_evaluator.py:1101-1182` (`transpile_dsl_to_rust_code`)
  - `research/flywheel_selfplay/selfplay_loop.py:1665-1671`
- **Mechanism**:
  - Although `run_fused_closed_loop.py` bypasses `transpile_dsl_to_rust_code` by invoking the compiled `bootstrap.s1main --fit-s1` binary (`rust_evaluator.py:466-525`), `selfplay_loop.py` and `RustEvaluator.evaluate_batch` call `transpile_dsl_to_rust_code`, which replaces almost every DSL primitive with a bogus mathematical formula:
    - `blom` and `rank` (order-statistic rank transforms across rows) are transpiled to a pointwise logistic sigmoid `1.0 / (1.0 + (-v).exp()) - 0.5` (`lines 1119-1121`).
    - `poly_interact` and `pair_prod` (all pairwise feature products) are transpiled to 1D adjacent window products `feats.windows(2).map(|w| w[0] * w[1])` (`lines 1132-1135`).
    - All other transforms (`robust_z`, `winsor`, ` Yeo-Johnson`, etc.) are transpiled to `v / (1.0 + v.abs())` (`line 1148`).
    - The terminal estimator head (`histgbm`, `ksmooth`, `ridge_gcv`, etc.) is completely ignored (`let _ = head_op;` at `line 1152`) and replaced by a raw linear dot product (`lines 1176-1180`).

---

## 5. Summary Action Matrix for Next Wave (`H56` Execution Order)

| Priority | Action | Files to Modify | Expected Effect |
| --- | --- | --- | --- |
| **P0 (Immediate)** | Fix Bug 1 (`revealed_indices=[0,1,2,3]` leakage & train/eval slate order mismatch), Bug 3 (pass real `ctx_feat`, `base_prior_cost`, `sos_prior`, `scale`), and Bug 4 (freeze or wire `parent_delta_net`). | `research/flywheel_selfplay/run_fused_closed_loop.py:316-318, 419-481, 956`; `research/flywheel_selfplay/automatic_promotion.py:323-333` | Aligns train and eval slate distributions; stops `28.8%` `0..3` leakage; enables the ranker to actually select the `710` promoted GP champions (`>= 1152`). |
| **P1 (Immediate)** | Fix Bug 2: append all evaluated GP candidates (especially neural top-picks that fail the gate or lose) into `pools[c]` and sample hard negatives in `training_indices`. | `research/flywheel_selfplay/run_fused_closed_loop.py:1291-1306`; `research/flywheel_selfplay/automatic_promotion.py:224-333` | Closes the self-play feedback loop so the ranker learns to stop re-selecting failing ASTs. |
| **P2 (Short-Term)** | Fix Bug 5: replace 2-step restore-on-reject + Gaussian weight noise + tie-accepting 91-context `argmin` ratchet with persistent multi-step AdamW + EMA shadow weights and a fold-clean `t537` paired family LCB gate (`H56-P3` & `H56-P4`). | `research/flywheel_selfplay/run_fused_closed_loop.py:220, 874-930, 974-982`; `research/flywheel_selfplay/dev_incumbent.py:488-514, 681-728` | Breaks out of the 603-round 2-step random walk around `r882` and eliminates the 4-context (`[10, 18, 21, 22]`) winner's-curse trap. |
| **P3 (Medium-Term)** | Train `parent_delta_net` on historical `round-*-scores.jsonl` parent $\to$ child transitions (`H56-P5`) and enable submodular diversity acquisition (`H56-P6`). | `/home/ubuntu/JopeDimeMicro/python/phase12_multimodal_model.py:1597-1640, 2105-2128`; `research/flywheel_selfplay/run_fused_closed_loop.py:1162-1215` | Turns `EditZero` GP proposal generation from unguided heuristic mutation into neurally steered, diversity-aware search. |
| **P4 (Whitepaper)** | Reconcile Section 6 (`06_method.tex` and `06_method/flywheel_status.tex`) with the mismatch table in Section 1 of this audit (explicitly distinguishing System A conceptual LLM writer, System B Lineage-B/CLOVER v1, and System C Fused GP + 29K-param adapter loop, including the Round 775 cohort switch). | `docs/whitepaper/sections/06_method.tex`; `docs/whitepaper/sections/06_method/flywheel_status.tex` | Eliminates all remaining paper-vs-code discrepancies before external review. |
