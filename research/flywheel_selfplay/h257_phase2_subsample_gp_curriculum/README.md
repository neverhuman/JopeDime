# H257 — Phase-2 Warm-Started Subsample & GP Evolutionary Curriculum (`H257_Phase2_Curriculum_Cycle`)

This directory contains **Research Wave `H257`** (`research/flywheel_selfplay/h257_phase2_subsample_gp_curriculum/`), the **Phase-2 Warm-Started Multi-Subsample & GP Evolutionary Curriculum** built directly on top of the converged **Phase-1 (`H256`)** models in [`../h256_pre_condense_residual_flywheel/`](../h256_pre_condense_residual_flywheel/).

## Key Highlights & Results on Held-Out `DL-Val91` (`91` Contexts / `19` Unseen Families)

- **Surpasses `Opus5p5_Extra` (Claude Opus 5.5) by `+125.29` Elo (S1 Constrained Utility) and `+109.37` Elo (2-Fold CV RMSE Ratio `-logratio`)**:
  - **`H257_HIGH_BATCH_EVO3_STACK3` (High Tier — 11-View Batch + 3-Gen Live Rust GP + Leak-Free 2-Fold OOF Top-3 Pareto Stack)**:
    - **Joint Elo (S1 Utility)**: **`1838.71`** (`+125.29` vs `Opus5p5_Extra` at `1713.42`; `+180.69` vs `Gemini4_argon_high` at `1658.02`; `+338.71` vs `R882` at `1500.00`; Pairwise vs `R882` = **`1835.34`**).
    - **Joint Elo (2-Fold CV RMSE Ratio `-logratio`)**: **`1840.84`** (`+109.37` vs `Opus5p5_Extra` at `1731.47`; `+175.37` vs `Gemini4_argon_high` at `1665.47`; Pairwise vs `R882` = **`1871.00`**, winning **`11` Wins / `0` Ties / `8` Losses** across the 19 unseen families head-to-head against `Opus5p5_Extra`).
    - **Mean Family Utility**: **`1.261212`** (`+0.148627` vs `Opus5p5_Extra`'s `1.112585`, `+0.107597` vs `R882`'s `1.153615`), with **`0` invalid picks (`91/91` CV gate pass rate = `100.0%`)**.
  - **`H257_MEDIUM_BATCH_EVO3` (Medium Tier — 11-View Batch + 3-Gen Live Rust GP Single-Tree Champion)**:
    - **Joint Elo (S1 Utility)**: **`1762.85`** (`+49.43` vs `Opus5p5_Extra`; Pairwise vs `R882` = **`1790.33`**).
    - **Joint Elo (2-Fold CV RMSE Ratio `-logratio`)**: **`1773.23`** (`+41.76` vs `Opus5p5_Extra`; Pairwise vs `R882` = **`1819.39`**).
    - **Mean Family Utility**: **`1.253609`** (`42/91` validation contexts strictly improved by 3-generation GP evolution over the static/active pool).
  - **Progressive Curriculum Lift (`Stage C0` $\to$ `C1` $\to$ `C2` $\to$ `C3`)**:
    - **`Stage C0` (`0` GP cycles — Warm-Start from `H256` + `3x` Semi-Synthetic Column & `80%` Row-Subsample Convergence)**: Boosts standalone Neural Reranker `DL-Train OOF` family utility from **`1.443276` $\to$ `1.567007` (`+0.123731` lift)** while cutting invalid picks from **`11.7` $\to$ `4.3`**, and raises `DL-Val91` `0-reveal` Joint Elo from `1576.54` $\to$ `1597.44` (`+20.90` Elo) and `2+2` Active Joint Elo from `1597.70` $\to$ `1612.21` (`+14.51` Elo) before running a single Rust GP fit.
    - **`Stage C1` (`Gen 0` 11-View Batched Seeding)**: Joint Elo jumps to **`1694.04`** (`FamU = 1.236936`), surpassing `Gemini4_argon_high` (`1658.02`) at Generation 0.
    - **`Stage C2` (`1-Gen` GP Evolution `Gen 0 -> Gen 1`)**: Joint Elo reaches **`1713.93`** (`FamU = 1.240541`), surpassing `Opus5p5_Extra` (`1713.42`) after just 1 evolutionary generation.
    - **`Stage C3` (`3-Gen` GP Evolution `Gen 0 -> Gen 1 -> Gen 2 -> Gen 3` + Pareto OOF Stack)**: Standalone GP family utility climbs monotonically across generations (`Gen 0: 1.059807` $\to$ `Gen 1: 1.119270` $\to$ `Gen 2: 1.162717` $\to$ `Gen 3: 1.183865`, **`+0.124058` evolutionary lift**), reaching **`1762.85` Elo (`Medium`)** and **`1838.71` Elo (`High Stack3`)**.

## Contents

- [`semi_synthetic_columns_and_views.py`](semi_synthetic_columns_and_views.py) — Stage `C0` (`0` GP cycles): Semi-synthetic additional column injection (Gaussian distractors, collinear duplicates, Student-$t_3$ heavy tails, and pairwise quadratic interactions) + stratified `80%` row subsamples + Two-Speed Differential LR + All-28-Pairs Regret-Weighted Margin Loss + Multi-View Consistency KL.
- [`batched_subsample_gp_evolution.py`](batched_subsample_gp_evolution.py) — 5-role `[11, M]` batched neural seeding (`1×100% + 8×80% + 2×semisyn`) $\to$ `Gen 0` (`8` GP trees) $\to$ `Gen 1..3` live Rust GP evolution (`bootstrap.s1main --fit-s1` + `EditZero` + `H257` causal trajectory pre-screening) + leak-free 2-fold OOF Top-3 Pareto stacking.
- [`phase2_curriculum_trainer.py`](phase2_curriculum_trainer.py) — Closed-loop Phase-2 curriculum (`Stage C0 -> C1 -> C2 -> C3`) feeding discovered GP winners back into the Deep Learning training slates and `champion_bank`, exporting [`weights/h257_phase2_curriculum_bundle.safetensors`](weights/h257_phase2_curriculum_bundle.safetensors).
- [`eval_frontier_dl_val91.rs`](eval_frontier_dl_val91.rs) & [`evaluate_frontier_vs_h257_elo.py`](evaluate_frontier_vs_h257_elo.py) — Direct 2-fold `JTT1` evaluation of `Opus5p5_Extra`, `Gemini4_argon_high`, `GPT6p1_Sol_Max`, and `GPT6_Astra_Max` on all `91` `DL-Val91` contexts and unified Bradley-Terry Elo computation.
- [`frontier_elo_comparison.json`](frontier_elo_comparison.json), [`kpi_telemetry.json`](kpi_telemetry.json), [`REPORT.md`](REPORT.md), and [`test_h257_phase2_curriculum.py`](test_h257_phase2_curriculum.py).
