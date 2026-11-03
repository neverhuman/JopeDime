# H255 (`h255_causal_2plus2_live_gp`) — Causal 2+2 Neural Trajectory Critic & Live Rust GP Closed-Loop Flywheel

- **Wave ID**: `h255-causal-2plus2-live-gp-20261103`
- **Status**: `COMPLETE`
- **Verdict**: **`ADVANCE as preregistered vs GATED_025; ADOPTION REJECTED (red-team 15:30Z)`** (superseded by [`H258`](../h258_top4_reveal_baseline/README.md))
- **Pre-Execution Preregistration Commit (`origin/v1/next`)**: `ff8ec7ed771dfba57e69976da136a387687f864f` (preregistered at `6c9227e5b40aae77ce53bb2f3ecb25071bb9bf29` with `0` pools loaded)
- **Host / Compute**: `xbabe2` CPU (`taskset -c 48-51 nice -n 19`, `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=4`; Part A `491.52s`, Part B `296.16s`)

---

## Red-team caveats & adoption rejection (watcher 15:30Z)

- **Part A is test-time 4-true-evaluation search, not checkpoint lift** (`utility = -fitness` is the revealed quantity, `common_h252.py:137`). On `DL-Val` (`Native91`, `91` contexts / `19` families), `CAUSAL_2PLUS2_NEURAL = CAUSAL_2PLUS2_R882_ZERO_SHOT = ACTIVE_2PLUS2_HURDLE = 1.168790` median family utility (`1.153636` median context utility): **the neural critic adds exactly `0` on `DL-Val`**. Furthermore, the comparators `R882` (`0` effective reveals used for selection) and `GATED_025` (random reveals from `idx4`, Elo `1496.33`) are weak baselines; [`H258`](../h258_top4_reveal_baseline/README.md) (`8869cc439c5f218bdef8b5276254b7acd595dff3`) proves that the trivial equal-budget baseline **`TOP4_PICK`** (`1.252000` family / `1.240360` context utility, Elo `1846.46`) beats `ACTIVE_2PLUS2_HURDLE` (`= CAUSAL_2PLUS2_NEURAL` on `DL-Val`) by **`-0.083210`** median family utility (`95% CI [-0.133392, -0.046824]`, `Holm p = 1.0`, `0/3` seeds, family W/T/L `0/0/19`).
- **Part B (`LIVE_GP_CAUSAL_2PLUS2_FLY`) vs the proper equal-budget comparator `LIVE_GP_LEGACY_STATIC_4` is `NO_ADVANCE`**: `LIVE_GP_CAUSAL_2PLUS2_FLY` minus `LIVE_GP_LEGACY_STATIC_4 = +0.000892` mean family utility (`95% CI [0.000000, +0.002125]`, `Holm p = 0.127`, `1` seed, family W/T/L `3/29/0` across `32` `DL-Train` contexts $\times$ `2` rounds). Gains and Elo reported against `INITIAL_INCUMBENT` (`+0.002848`, Elo `1605.17`) are monotone by construction because promotions only accept strict improvements over the incumbent, and therefore do not establish policy lift.
- **Hygiene OK**: `DOPE` checkpoint `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` verified unchanged (`dope_unchanged = true`), `R882` `Native91` anchor `1.1377941898164525` reproduced bit-exact (`diff = 0.0`), and `sealed_meta_test_read = false`.

---

## 1. Hypothesis & Root-Cause Audit of H252 `FUSED_CLOSED_LOOP`

1. **Root-Cause Audit of H252 `FUSED_CLOSED_LOOP`**:
   - In [`run_fused_closed_loop.py`](../h252_hurdle_trajectory_flywheel/run_fused_closed_loop.py), `FUSED_CLOSED_LOOP` hardcoded `revealed_indices=[0, 1, 2, 3]` at evaluation time (raw pool slots `0..3`, median best oracle rank `181.0` on `DL-Val`), whereas training slates sorted candidates by true oracle fitness descending so slots `0..3` were always the top-4 oracle winners (`training_self_distance_d2_joint_on_slots_0_to_3 = 0.0`).
2. **H255 Hypothesis**:
   - **Part A**: Fixing the reveal budget to $B = 4 = 2 + 2$ (`k1 = 2` pre-outcome Batch-1 reveals from `R882` + `HurdleSOSCritic`, followed by `k2 = 2` Batch-2 reveals over the $N - 2$ remaining unrevealed candidates with strict self-exclusion $d_{\text{joint}} > 0$) will eliminate self-leakage and match or exceed `ACTIVE_2PLUS2_HURDLE` vs `GATED_025`.
   - **Part B**: Embedding the 2+2 sequential reveal policy inside the live Rust `evaluator_cli` GP loop (`32` stratified `DL-Train` contexts $\times$ `2` rounds $\times$ `16` grammar-valid proposals, $B = 4$ Rust evaluations/context/round) will discover more incumbent promotions than static 4-reveal selection (`LIVE_GP_LEGACY_STATIC_4`).

---

## 2. Setup

- **Frozen H252 Snapshot**:
  - `common_h252.py`: `abe974ed10a0cd0d4b08440015a72a7b0adcebb22118c234732b812e9f0b0f92`
  - `stage1_hurdle_sos_critic.py`: `fc0dd0e2fa543f0540d677609989c1785ef33b07f6cf7e54f4f8bedbd0e5e797`
  - `DOPE` checkpoint: `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` (`unchanged = true`)
  - Live Rust `evaluator_cli`: `e3be938644722d14a1c1a0e82168518838c4d174cfe031076a339745c48870a8`
- **Part A Evaluation**: `1,219` `DL-Train` contexts (`205` families, 5-fold family-grouped OOF) and `91` `DL-Val` (`Native91`, `19` families) across 3 seeds (`255101`, `255102`, `255103`), `10,000` paired family bootstrap resamples, Holm-Bonferroni correction across 5 arms (`m = 5`).
- **Part B Evaluation**: `32` stratified `DL-Train` contexts (`32` distinct families), `2` rounds $\times$ `16` proposals/round, $B = 4$ Rust evaluations/context/round (`256` total Rust evaluations/arm), comparing `LIVE_GP_CAUSAL_2PLUS2_FLY`, `LIVE_GP_CHAMPION_COND_STATIC_4`, and `LIVE_GP_LEGACY_STATIC_4`.

---

## 3. Outcome & Key Metrics

### 3.1 Part A — `DL-Val` (`Native91`: `91` Contexts, `19` Families)

| Arm | Effective Reveals | Median Context Utility | Median Family Utility | Median Family $\Delta$ vs `R882` | 95% Family Bootstrap CI | `holm_p` | Family W/T/L | Harm $>0.01$ | Median Elo | Status vs `GATED_025` / `TOP4_PICK` |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | --- |
| `R882` | `0` (4 logged) | `1.137794` | `1.153615` | `0.000000` | — | — | — | — | `1500.00` | Zero-reveal baseline |
| `GATED_025` | `4` | `1.137192` | `1.153038` | `-0.000577` | `[-0.001273, 0.000000]` | `1.0000` | `0/18/1` | `0.00%` | `1496.33` | Weak random-4 comparator |
| `ACTIVE_2PLUS2_HURDLE` | `4` | `1.153636` | `1.168790` | `+0.015174` | `[+0.005419, +0.026475]` | `0.0005` | `9/8/2` | `0.00%` | `1529.31` | Superseded by `TOP4_PICK` (`1.252000`) |
| `CAUSAL_2PLUS2_R882_ZERO_SHOT` | `4` | `1.153636` | `1.168790` | `+0.015174` | `[+0.005439, +0.026480]` | `0.0005` | `9/9/1` | `0.00%` | `1529.31` | Identical to `ACTIVE_2PLUS2_HURDLE` |
| `CAUSAL_2PLUS2_NEURAL` | `4` | `1.153636` | `1.168790` | `+0.015174` | `[+0.005419, +0.026475]` | `0.0005` | `9/8/2` | `0.00%` | `1529.31` | `+0.000000` over `ACTIVE_2PLUS2_HURDLE` |
| `NEGATIVE_CONTROL_PERMUTED_REVEAL` | `4` | `1.137192` | `1.153038` | `-0.000577` | `[-0.001380, 0.000000]` | `1.0000` | `0/18/1` | `0.00%` | `1496.33` | Negative control (`PASS`) |

### 3.2 Part A — `DL-Train` OOF (`1,219` Contexts, `205` Families)

| Arm | Median Context Utility | Median Family Utility | Median Family $\Delta$ vs `GATED_025` | 95% Family Bootstrap CI | `holm_p` | Family W/T/L | Median Elo |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `R882` | `1.423073` | `1.385355` | `-0.000109` | `[-0.000920, +0.000046]` | `0.8504` | `2/199/4` | `1499.38` |
| `GATED_025` | `1.423165` | `1.385465` | `0.000000` | — | — | — | `1500.00` |
| `ACTIVE_2PLUS2_HURDLE` | `1.495432` | `1.467077` | `+0.081342` | `[+0.041346, +0.140055]` | `0.0005` | `77/118/10` | `1525.08` |
| `CAUSAL_2PLUS2_R882_ZERO_SHOT` | `1.496568` | `1.467975` | `+0.081363` | `[+0.041884, +0.140195]` | `0.0005` | `82/110/13` | `1525.56` |
| `CAUSAL_2PLUS2_NEURAL` | `1.496949` | `1.468249` | `+0.082033` | `[+0.042161, +0.140386]` | `0.0005` | `84/109/12` | `1525.65` |
| `NEGATIVE_CONTROL_PERMUTED_REVEAL` | `1.425507` | `1.387938` | `+0.002473` | `[-0.000542, +0.007917]` | `0.5729` | `4/196/5` | `1499.63` |

### 3.3 Part B — Live Rust GP (`32` Stratified `DL-Train` Contexts $\times$ `2` Rounds, `256` Rust Evaluations/Arm)

| Contrast / Arm | Promotions (Batch-2) | Final Mean Utility | Mean Family $\Delta$ vs `LIVE_GP_LEGACY_STATIC_4` | 95% Family Bootstrap CI vs `LEGACY` | `holm_p` vs `LEGACY` (`m=3`) | Family W/T/L vs `LEGACY` | Verdict vs `LEGACY` |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | --- |
| `INITIAL_INCUMBENT` (`0` evals) | `0 (0)` | `1.809228` | `-0.001956` | — | — | `0/24/8` | Monotone baseline only |
| `LIVE_GP_LEGACY_STATIC_4` (`256` evals) | `9 (0)` | `1.811185` | `0.000000` | — | — | — | Equal-budget static-4 comparator |
| `LIVE_GP_CHAMPION_COND_STATIC_4` (`256` evals) | `8 (0)` | `1.811035` | `-0.000150` | `[-0.001305, +0.000720]` | `1.0000` | `1/29/2` | `NO_ADVANCE` |
| `LIVE_GP_CAUSAL_2PLUS2_FLY` (`256` evals) | `12 (8)` | `1.812076` | **`+0.000892`** | **`[0.000000, +0.002125]`** | **`0.1272`** | **`3/29/0`** | **`NO_ADVANCE` (`1` seed, `CI95` touches `0`)** |

---

## 4. Verdict & Next Step

- **Verdict**: **`ADVANCE as preregistered vs GATED_025; ADOPTION REJECTED (red-team 15:30Z)`**.
  1. In Part A, `CAUSAL_2PLUS2_NEURAL` adds `0.000000` on `DL-Val` over `ACTIVE_2PLUS2_HURDLE`, and both are beaten by `-0.083210` median family utility (`0/3` seeds, W/T/L `0/0/19`) by `TOP4_PICK` in [`H258`](../h258_top4_reveal_baseline/README.md).
  2. In Part B, `LIVE_GP_CAUSAL_2PLUS2_FLY - LIVE_GP_LEGACY_STATIC_4 = +0.000892` (`95% CI [0.000000, +0.002125]`, `Holm p = 0.1272`, `1` seed, W/T/L `3/29/0`) does not pass Holm significance (`p < 0.05`) or strict LCB (`> 0`).
- **Next Step**: Proceed to **`H261` (`h261_diverse_reveal_set`)** to test whether diversity-aware 4-reveal candidate selection (`MMR4_CV` and `STRUCT4`) with direct best-of-revealed selection (`argmin` revealed gated cost) improves over `TOP4_PICK` at matched 4-reveal budget.

---

## 5. Artifacts & Code Links

- [`STATUS.md`](STATUS.md) — Detailed root-cause audit and per-context tables
- [`PREREG.md`](PREREG.md) — Preregistered protocol
- [`part_a_causal_2plus2_neural.py`](part_a_causal_2plus2_neural.py) — Part A evaluation script
- [`part_b_live_rust_gp_2plus2.py`](part_b_live_rust_gp_2plus2.py) — Part B live Rust GP script
- [`results/part_a_metrics.json`](results/part_a_metrics.json) — Part A metrics JSON
- [`results/part_b_metrics.json`](results/part_b_metrics.json) — Part B metrics JSON
