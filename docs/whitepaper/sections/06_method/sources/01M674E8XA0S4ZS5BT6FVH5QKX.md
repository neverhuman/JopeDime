# Section 6 — editor note

Section: **6. EurekaZero: Learning Taste** (`sections/06_method.tex`, label `sec:method`).
Brief: *EurekaZero — Final Paper Plan* (the Oct 30 version with "What changed from the last plan"),
form per `EurekaZero_Writing_Guide.md`. Facts were checked against `neverhuman/jopedime` at
`ff127e4d61cb` (30 Oct 2026). The section describes the **accepted lineage-B checkpoint, step
60,000**, because it is the only accepted policy and the one `crates/jope-policy` loads.

## 1. Placeholders (`\TBD{}`)

| Where | Placeholder | What closes it |
| --- | --- | --- |
| 6.2 | `name, version, licence` | Identity of the third-party regressor behind the 1,024-value foundation block. The repo's bundle (`crates/jope-encoders/artifacts/model-bundle.v1.json`) lists TabICL-v2 artifacts as "provenance pending, redistribution prohibited, policy-ineligible". Confirm which model, then state its licence. |
| 6.5 | `acceptance record` | On what evidence step 60,000 was accepted for serving. `docs/internal/asm-manual/sections/training-history.md` says all 48 indexed F10 decisions of 11 Sep were inconclusive and does not cite an F10 pass for step 60,000. Either cite the F10 receipt or say it was accepted by owner decision. |

No `\citeneeded{}` was used.

## 2. Core points advanced

- **3, search = 1**: 6.1, 6.7 and Algorithm 2 define exactly what one inference includes and excludes.
- **4, referee not imitation**: 6.4–6.5. The writer imitates the *referee's* verdicts (rank-weighted), not human history; 6.6 protects that referee.
- Supports **1** through the board, move, referee and incumbent vocabulary.

## 3. Brief items the code does not support (read this first)

The brief describes a design that differs from the implemented system in five places. I wrote what the code does, as the brief's "avoid describing mechanisms that are not in the code" requires.

1. **Encoder is not DOPE.** Step 60,000 reads a 2,048-value context: pooled row-encoder statistics (512), a projected third-party regressor state (1,024) and a target encoding (512) (`crates/jope-policy/src/context_producer.rs`, `context.rs:53-57`). The DOPE/Dope512 landmark path is optional and unqualified (`docs/optional-landmarks.md`), and H27 found no established lift (`docs/research/studies/dope-neural-ranking-20261021/h27/nn-fixed400-study.md`). There is no measured-versus-missing mask: non-finite contexts are refused. To keep the main text inside its page budget, DOPE is not mentioned there. Add one sentence only if a later checkpoint uses it.
2. **Map loss is listwise, not CLIP-style InfoNCE.** The trained loss is ListNet with targets `softmax(percentile/0.25)`, plus a weighted Bradley–Terry term (`crates/jope-engine/src/neural.rs:15-25,745-830`). Equation 3 is written exactly as that ListNet term. The text shows that it becomes InfoNCE as τ→0, which keeps the CLIP framing honest. Notation: τ here is the target temperature (0.25), not a logit temperature, and **γ is unused**, because stillborn programs never enter a list (`crates/jope-engine/src/corpus.rs`, groups of ≥8 scored programs). Either drop γ from the notation table or add a stillborn term to training.
3. **The map does not raise q in this checkpoint.** The writer and map are separate branches with separate context networks and token embeddings (`docs/internal/asm-manual/sections/deep-stack.md`, parameter table). The map orders the 64 drafts but drops none and chooses no search parents (`crates/jope-policy/src/ranking.rs`). None of the brief's three mechanisms is in the accepted checkpoint:
   - **Conditioning on nearby winners** (kNN-CE) is scratch arm R25, which raised a regression alert (`deploy/dl-flywheel/docs/model.md`).
   - **Screening** (the surrogate funnel) is offline and shadow-only (`crates/jope-learning/src/clover/expansion.rs`).
   - **Steering** is not found.

   **Consequence for the abstract and Section 7:** "problem–solution embeddings cut wasted moves from X% to Y%", and a Table 7 row "without the map", cannot show a map effect on the writer for this checkpoint. Either ship one mechanism before Section 7 or reword the abstract.
4. **Program embedding is order-blind.** The map reads `z_w` as a mean of token embeddings with no position term. Behavior on probe data is only a hash, used for novelty, not a learned embedding (`clover/expansion.rs:577-608`).
5. **Self-play evidence.** The loop (Algorithm 1) is implemented: the HWM board ratchet (`crates/jope-learning/src/hwm.rs`), the F10 player ratchet (`docs/adr/ADR-0005-canonical-learning-loop.md` §6) and lineage-B DL→search→DL cycles. The repo's own status line is "the optimizer fitted its objective; better proposals have not been demonstrated" (`docs/internal/asm-manual/sections/status.md:56`). Other relevant records:
   - In 3,227 of 3,268 paired final searches, the neural arm tied the generic arm exactly.
   - 98.84% of historical winners were already in their search's seeds.
   - The quality head ranks fresh programs at chance, 0.506 [0.441, 0.568]. That figure is from v1, not step 60,000.

   Section 6 makes no performance claim. Section 7 must report these honestly.

## 4. Protocol commitments Section 7 must honor

These are stated as method in 6.6–6.7, so the results have to match them:

- **One draw per problem.** Take the first sample at T=1 with a recorded seed. Use no fallback programs, no exploration programs, no ranking by the reward heads and no redraw.
  - The production decoder samples 8×128 rows and keeps 64 plus 16 (`crates/jope-policy/src/decode.rs`); the search = 1 harness must bypass that.
  - A draft that does not reach EOS within 128 tokens, crashes or times out scores as a loss.
- **Re-run every reported winner from emitted code.** Native and emitted predictions must agree within 1e-9·max(1,|ŷ|), the v1 execution rule in `model.md`.
- **Run the arena through the sandboxed referee.** That is seccomp and Landlock (`crates/jope-engine/src/eval_registry/service/sandbox.rs`) plus a cgroup memory limit with swap 0 (`crates/jope-pipeline/src/execution.rs:611-613`). Lineage-B historical fits used a different executor.
- **Disjoint families.** Arena families must be disjoint from the 80 training datasets of snapshot-000097 and from lineage-B meta-validation.
- **Game-day runtime parity.** If game day runs in the Rust/Candle port, its parity with the trained tch forward pass is "not established" (`crates/jope-policy/src/lib.rs:82-103`). Run game day on the reference runtime or establish parity first.

## 5. Facts used, with sources

| Fact in text | Source |
| --- | --- |
| Context 2,048 = 512 + 1,024 + 512; frozen; non-finite refused; `final_test_targets_used:false`; 30 s cap | `crates/jope-policy/src/context_producer.rs:273-312,416`; `context.rs:83-88`; `docs/internal/asm-manual/sections/architecture.md:219` |
| Measured 11.7–20.8 s per job; moved to Appendix E | `architecture.md:219` |
| 401-token vocabulary; 31 transforms; 22 heads; ≤6 transforms; grammar automaton | `crates/jope-engine/assets/grammar/featmath.json`; `deep-stack.md` Stage 4; `crates/jope-policy/src/decode.rs` |
| Canonical form → SHA-256 identity; behavior hash on training probe | `crates/jope-pipeline/src/seeds.rs:124`; `crates/jope-learning/src/clover/expansion.rs:577-608` |
| Python/Rust emitters | `docs/emitters.md:7` |
| Map: mean token embedding; φ = [g, g⊙tanh(·), \|g−proj\|]; 4 heads, mean | `deep-stack.md` "Reward branch"; `crates/jope-engine/src/neural.rs:489-531` |
| Objective `seq_nll + 0.25 BT + 0.1 ListNet + 0.1 runtime + 0.05 ablation BT`; row weight 0.05+ρ⁴; smoothing 0.05 | `deep-stack.md` Stage 7; `neural.rs:15-25` |
| Lists need ≥8 scored programs; ρ = 1 − rank/(n−1) | `crates/jope-engine/src/corpus.rs` (groups_below_eight) |
| 64 drafts at T=1; 16 exploration programs; grammar-uniform fill-ins | `decode.rs:26-46`; `deep-stack.md` Stage 4 |
| Five inner folds | `crates/jope-pipeline/src/prepare.rs:35` |
| HWM strict improvement; ties keep older | `docs/learning.md`; `crates/jope-learning/src/hwm.rs` |
| F10: 16 cuts × 3 seeds, mean ln ratio < 0, wins ≥ losses, failures ≤ serving | `docs/adr/ADR-0005-canonical-learning-loop.md` §6 |
| 6 blocks, width 512, 4 heads, prefix token, max 128 tokens, 27,796,494 parameters | `deep-stack.md` Stage 3; `crates/jope-policy/src/config.rs` |
| 8×A100, September 2026; 10,214,332 scored programs; 80 datasets; 7,877 lists; 8,894 contexts | ADR-0005; `docs/internal/asm-manual/sections/data-estate.md` |
| Sealed meta-test outside the input closure; `final_tests_opened` false | ADR-0005; `AGENTS.md` |
| Inner-fold poison test; sealed canary; train-only red team | `crates/feat-math/tests/inner_fold_leakage.rs`; `crates/jope-engine/examples/research_loop_fixture.rs:26,239-241`; `crates/jope-campaigns/src/commands/train_only_redteam.rs` |
| Golden45 test-selected entry (best of 63,108 test-scored evaluations, 42 datasets) | `README.md` arena block |

## 6. Bibliography

All keys are in the delivered `refs.bib`. Each was checked on 30 Oct 2026 against the DOI, arXiv, PMLR or NeurIPS page.

- **Keys named in the brief:** `silver2016mastering`, `silver2017mastering`, `anthony2017expert`, `radford2021clip`, `oord2018cpc`, `kusner2017grammar`, `dai2018sdvae`, `sakana2025walkback`, `metr2025rewardhacking`, `zhong2026hackerfixer`.
  - `sakana2025walkback` is the TechCrunch report of 21 Feb 2025. The page names no author, so the entry uses `{TechCrunch}`. Sakana's own page carries no correction note.
  - `zhong2026hackerfixer` is arXiv:2606.08960, "Hardening Agent Benchmarks with Adversarial Hacker-Fixer Loops".
- **New keys:**
  - `cao2007listnet` (ICML 2007, doi 10.1145/1273496.1273513), the source of the map's listwise loss.
  - `bradley1952rank` (Biometrika 39(3/4):324–345). It is already planned for Section 7, so it may already be in the shared file.

## 7. Labels, symbols and shared-file issues

- **Labels defined:** `sec:method`, `sub:method-picture`, `sub:method-board`, `sub:method-move`, `sub:method-map`, `sub:method-selfplay`, `sub:method-referee`, `sub:method-gameday`, `fig:network`, `eq:mapLoss`, `alg:selfplay`, `alg:gameday`.
- **Labels used from other sections:** `sec:game`, `sec:results`, `eq:taste`, plus two appendix labels I assumed: `app:training` (Appendix E) and `app:incidents` (Appendix I). Please confirm those two names with the appendix authors.
- **New symbols**, all defined in the text: `G_p` (scored list), `ρ_w` (percentile in the list), `L_writer`, `L_BT`, `L_runtime`, `L_prog`.
- **Bug in the shared `macros.tex`.** `\newcommand{\Game}` fails because `amssymb` already defines `\Game`. Use `\renewcommand{\Game}` or rename it. The preview works around it with `\let\Game\relax`.
- **IP vocabulary.** The text passes the repo's `deploy/ci/ip-scrub.sh` pattern: no GP, genome, evolution, population, mutation, crossover or fitness. The teacher appears only as "verified search", by its interface, as `docs/paper/eurekazero/appendices/J-disclosure.tex` requires.
  - Sections 3 and 4 of the plan name genetic programming, Koza and "the GP language". Whoever owns those sections should decide whether that history is allowed, since the same scrub blocks those words.

## 8. Length and material for Appendix E

- **Length.** The standalone preview runs from the top of page 1 to six lines into page 4. That is about 2.85 pages of material, including roughly eight lines lost to float placement. The budget is 2.5 pages, with 2.75 as the 10% limit.
  - If the editor needs room, cut first the Golden45 sentence in 6.6, then the Zhong clause.
- **Moved to Appendix E** to fit the budget:
  - per-block encoder details: Hyperion 128-d pooled by mean, std, p10 and p90; foundation capped at 768 context rows and 256 query rows; target windows with seed 17;
  - encoder timings of 11.7–20.8 s on eight reference jobs;
  - Bradley–Terry weighting: score gap, doubled when either program is at or above the 0.9 percentile;
  - optimizer: AdamW, learning rate 1e-4, weight decay 0.01, clip 1.0, global batch 2,048;
  - training history: retained steps 6,001–60,000 in 18 blocks of 3,000;
  - the 1.05 M parameters used only for training-time controls;
  - the sandbox details.
