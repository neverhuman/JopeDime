# H441 Closeout — `W5` Grammar-Complete AST Surrogate + Gen-0 Online Ridge Screener vs `H0.1-A` (`W5_SCREENER_CONFIRMED` on Rank/Top-8 Search; Diversity Collapse Diagnosed on Greedy Top-50% Stack Pruning)

- **Wave:** `H441` (`research/flywheel_selfplay/h441_w5_online_ast_screener`, `@agy_apex_w10`)
- **PREREG Commit:** `93972832db57942f95d66e2a7199ae60b032bc50` (`origin/v1/next`)
- **Published Stage:** `/home/ubuntu/apex-stage/h441/pub-r1` (`CODE.sha256` `5419fb8aaabaa6f7`, `wall_s = 430s <= 3,600s` on `cpus 64-79`, `CUDA_VISIBLE_DEVICES=""`)
- **Pre-Readout Integrity Gates (`H441.gates.json`):** **`6/6 PASS`** (`106` total `FIT-SEL`, `88` `H431` clean non-copy `FIT-SEL`, `16,960` `H0.1-A` candidates scored inside H335 docker sandbox, zero `APEX-DEV` / `APEX-CONFIRM` reads)
- **Verdict:** **`W5_SCREENER_CONFIRMED`** (`holm_primary_clean88`: **`P1_search_top8_vs_H01A = True`** at `p < 1e-6`, **`P2_audit_stack_vs_H01A = False`**) — revealing two high-leverage findings for GP search and stack pool retention.

---

## 1. Primary Results (`clean_non_copy_88`, `n = 88` families; `all_fitsel_106`, `n = 106` families)

### A. Candidate Ranking & Top-8 Search Utility (`P1` — Holm-Bonferroni PASS)

| Screener Arm | `clean-88` Mean Spearman `ρ` | `clean-88` Top-8 Search `u` | `clean-88` Audit Stack `u` (Top-80/160 `SXB`) | `all-106` Mean Spearman `ρ` | `all-106` Top-8 Search `u` | `all-106` Audit Stack `u` |
| --- | --- | --- | --- | --- | --- | --- |
| `ORC` *(True OOF Oracle)* | `1.0000` | `1.7248` | `2.0457` | `1.0000` | `1.9218` | `2.2752` |
| **`W5_ONLINE_UCB`** *(AST Surrogate + 32-cand Gen-0 Ridge)* | **`0.6761`** | **`1.7089`** | `2.0349` | **`0.6897`** | **`1.9011`** | `2.2567` |
| `W5_ZERO_SHOT` *(5-fold Cross-Family AST Surrogate, 0 probes)* | `0.3737` | `1.6436` | `2.0500` | `0.4022` | `1.8462` | `2.2770` |
| `H01A_ONLINE` *(`H0.1-A` + 32-cand Gen-0 Calibration)* | `0.3647` | `1.6582` | `1.9467` | `0.3480` | `1.8387` | `2.1924` |
| `H01A_RAW` *(Production `H0.1-A` Neural Screener)* | `0.2666` | `1.6106` | `1.9456` | `0.2367` | `1.7856` | `2.1916` |
| **`RND`** *(Uniform Random 80-of-160 Control)* | `0.0069` | `1.6999` | **`2.0573`** | `0.0038` | `1.8913` | **`2.2849`** |

### B. Key Paired Comparisons (`clean-88`)

- **`P1` (`W5_ONLINE_UCB` vs `H01A_ONLINE`, Top-8 Search `u`):** **`+0.0507 u` (`+4.94% RMSE`), `clipped +0.0378`, `p < 1e-6`, `lofo_min +0.0325`, `W/T/L 61/6/21` — HOLM PASS (`True`)**.
- **`W5_ONLINE_UCB` vs `RND` (Top-8 Search `u`):** **`+0.0090 u`, `clipped +0.0090`, `p = 0.00185`, `W/T/L 50/8/30`** (`+0.0098 u`, `p = 6.77e-5`, `63/11/32` on `all-106`), whereas `H01A_ONLINE` **loses** to `RND` (`-0.0417 u`, `W/T/L 29/2/57`).
- **`P2` (`W5_ONLINE_UCB` vs `H01A_ONLINE`, Downstream `SXB` Audit Stack `u`):** `+0.0882 u` unclipped (`2.0349` vs `1.9467`), **`clipped -0.0022`, `p = 0.589`, `W/T/L 35/6/47` — FAIL (`False`)**.

---

## 2. Two Concrete Scientific Findings for GP Search (`H418`) and Stacking (`H434` / `H440`)

1. **`W5_ONLINE_UCB` Decisively Solves Scalar OOF Ranking (`ρ: 0.2666 -> 0.6761`):**
   - Production `H0.1-A` (`ρ = 0.2666` raw, `0.3647` online-calibrated) under-ranks high-utility trees so badly that its top-8 search `u` (`1.6582`) and downstream `SXB` audit `u` (`1.9467`) both trail uniform random selection (`1.6999` and `2.0573`)! Adding grammar-complete AST structural features + 32-candidate Gen-0 ridge calibration (`W5_ONLINE_UCB`) lifts rank correlation to **`0.6761`** (`0.6897` on `all-106`) and recovers `94%` of the `RND -> ORC` top-8 search `u` gap (`1.7089` vs `1.7248`, `W/T/L 61/6/21` vs `H01A_ONLINE`).
2. **Greedy Scalar Top-50% Pruning Destroys Ensemble Diversity — Even for the True OOF Oracle (`ORC = 2.0457 < RND = 2.0573`):**
   - Why did `P2` fail despite `W5_ONLINE_UCB` achieving `ρ = 0.6761` and `+0.0882 u` unclipped over `H01A_ONLINE`? Look at the control row: **uniform random 80-of-160 retention (`RND = 2.0573`) beats even the true OOF top-80 oracle (`ORC = 2.0457`, and `2.2849 > 2.2752` on `all-106`)**!
   - Pruning `160 -> 80` candidates purely by scalar OOF `u` packs the surviving pool with near-duplicate variations of the top AST motif while discarding lower-OOF orthogonal complementary GP trees — independently confirming `@cl_apex`'s `H434` `(overlap_rho, rmse_ratio)` law from the GP search side!
   - **Actionable Design Rule:** Replace `H0.1-A` with `W5_ONLINE_UCB` for individual offspring scoring, **and** never prune a stack candidate pool by scalar predicted `u` alone — enforce AST-root / prediction-correlation diversity quotas (e.g. `50%` `W5_ONLINE_UCB` + `50%` orthogonal/stratified exploration) and pass surviving pools through `H440`'s `guard_and_stack_2l`.
