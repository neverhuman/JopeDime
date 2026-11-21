# H464--H465 Tabular Black-Hole Master Study & White-Paper Guide

**Updated:** 2026-11-16;
**Scope:** Canonical technical synthesis of the `H458`--`H465` tabular black-hole research waves, problem and kernel-embedding formulation, receipt-pinned `Dev91` results (`2408.5` per-table absorbed, `2343.8` family absorbed, `1859.0` in-corpus, `1821.5` pre-registered unseen / `1842.0` adaptive unseen), training and bank-disjoint validation loss curves, `TabArena v0.1` + `BeyondArena 2026` benchmarking, and the 5-host `Bank r5` $\to$ `Bank r6` foundation training loop.

---

## 1. Executive Summary & Regime-Separated Headline Results

A core principle of this research program is **strict data-exposure regime separation**. Absorbing a benchmark corpus under random non-official data partitions answers a fundamentally different scientific question from zero-shot inference on unseen tables, and merging those regimes is prohibited across all manuscripts and scoreboards:

| System Variant | Exposure Regime | Fit ID | `Dev91` Elo | 95% Bootstrap CI | Key Paired / Head-to-Head Evidence |
| --- | --- | --- | ---: | --- | --- |
| `Kumo-Tabular-Large (e8)` | `external` (0% test in run) | Fit C (`board_34`) | 2057.6 | `[1882.5, 2322.2]` | Rank `#1/34` on unabsorbed generalization board (`FITS_RECEIPT.json` `4abed343`) |
| **`LimiX-2 (406.2M, 32 pipelines)`** | **`external` (`diagnostic_only`, 0% test in run)** | **Fit C (`board_34`)** | **2035.1** | `[1823.1, 2362.3]` | **Rank `#2/34`** (`91/91` contexts, `0` caps, `be943010`; *release checkpoint ed01f7a0 (HF de07b679); upstream found it contaminated with benchmark data (autogluon/tabarena#648; makers' audit 2026-11-11); TabArena dropped it 2026-11-16 for LimiX-2-1016*); tied with `Kumo-Large` (`12-0-7` families, `52-1-38` contexts, $\Delta u = -0.0087$, $p = 0.574$); beats `TabFM-1.0 Ensemble` (`17-0-2`, `14.7%` geom-mean RMSE cut, $p=0.0062$), `Kumo-Small` (`16-0-3`), `TabFM-1.0 Single` (`16-0-3`), `TabPFN-3.5` (`15-0-4`), and our unabsorbed `fair` `h448` (`1813.6`, `18-0-1` families). S3: `campaigns/h465-limix-dev91-20261116/` (`ce03e115` / `3ba5acda`) and `results-20261116T160110Z.tar.zst` (`dd62cb20`) |
| **EurekaZero Stack + 19 Family + 91 Table Dossiers (`GEN2A`)** | **`recall` (absorbed\*)** | **Fit D (`paper_field_35`)** | **2396.1** | `[2218.8, 2754.2]` | *Absorbed: not a generalization comparison.* Joint 35-entrant fit alongside `Kumo-Large` (`2059.1`, `#2`) and `LimiX-2` (`2036.4` `[1825.3, 2355.7]`, `#3`); `GEN2A` vs `LimiX-2`: **`16-0-3` families, `79-0-12` contexts** (`+1.171 u`, `69.0%` lower RMSE) |
| **EurekaZero Stack + 19 Family + 91 Table Dossiers (`GEN2A`)** | **`recall` (absorbed\*)** | **Fit B** | **2408.5** | `[2237.3, 2726.3]` | *Absorbed: not a generalization comparison.* `+336.8` Elo over `Kumo-Tabular-Large` (`2071.7` in Fit B); **19/19** families gain over Look-2 (`+0.382 u`, `32%` lower RMSE, one-sided $p = 1.9\times 10^{-6}`); `77-1-13` tables vs `Kumo-Large`, `86-0-5` vs `Kumo-Small`, `83-0-8` vs `TabFM-1.0 Single`, `88-0-3` vs `TabPFN-3.5` |
| `Kumo-Tabular-Large (e8)` | `external` (0% test) | Fit B | 2071.7 | `[1892.7, 2337.7]` | Strongest external foundation baseline in Fit B (`SCORE.json` `e7ffd5c0002d08a5`) |
| **EurekaZero Stack + 19 Family Dossiers (Look 2)** | **`recall` (absorbed\*)** | **Fit A** | **2343.8** | `[2195.4, 2612.5]` | *Absorbed: not a generalization comparison.* Standalone receipt CI `[2181.2, 2619.2]`; `16-0-3` families / `71-0-20` contexts vs `LimiX-2` (`+0.789 u`) |
| `Kumo-Tabular-Large (e8)` | `external` (0% test) | Fit A | 2070.2 | `[1892.2, 2335.6]` | Rank 2 on the 35-entrant Look-2 `rev20` ladder |
| `TabFM-1.0 Ensemble` | `external` (0% test) | Fit A | 1915.6 | `[1747.0, 2152.0]` | Multi-checkpoint ensemble reference (`1902.0` in Fit C) |
| `Kumo-Tabular-Small (e8)` | `external` (0% test) | Fit A | 1900.2 | `[1735.0, 2135.0]` | Local offline weights (`1890.9` in Fit C) |
| `TabFM-1.0 Single` | `external` (0% test) | Fit A | 1888.1 | `[1726.0, 2117.0]` | Single-checkpoint baseline (`1877.2` in Fit C) |
| **EurekaZero Generalist Stack (no dossiers)** | **`in-corpus`** | **Fit A** | **1859.0** | `[1707.0, 2078.0]` | `+7.4` Elo above `TabPFN-3.5` on Look-2 ladder (`4W-0T-15L` family head-to-head); isolates the `+484.8` Elo within-Fit-A separation to `2343.8` (*absorbed: not a generalization comparison*, `ea4a1470`) |
| `TabPFN-3.5 (e8)` | `external` (0% test) | Fit A | 1851.6 | `[1681.0, 2097.0]` | Local v3.5 weights (`1847.5` in Fit C; `0` paid API calls) |
| **EurekaZero Pre-Registered Champion (`h448`)** | **`fair` (0% `Dev91`)** | **Fit A** | **1821.5** | `[1697.0, 1990.0]` | Single pre-registered unseen read (`1813.6` in Fit C); a separate 7-read adaptive unseen series reached **1842.0** Elo in its own fit |

> **Fit comparability rule:** Elo ratings from Fit B (`GEN2A` per-table look) and Fit A (`rev20` Look-2 ladder) come from separate maximum-likelihood Bradley--Terry fits and are compared strictly within their respective blocks. Bootstrap 95% confidence intervals span approximately `300` to `500` Elo (`489` Elo width on the `2408.5` row), whereas paired family sign-flip tests (`19/19` positive, $p = 2^{-19} = 1.9\times 10^{-6}$) establish within-board superiority with high statistical power.

---

## 2. Two-Space Mathematical Framing: Problem Encoding & Kernel Solution Embedding

All public manuscripts (`docs/paper/blackhole/` and `docs/whitepaper/`) present the discovery and consolidation architecture through two continuous geometric objects:

1. **Problem Encoding $z_{\mathrm{p}} = \Phi_{\mathcal{P}}(X_{\mathrm{ctx}}, y_{\mathrm{ctx}}) \in \mathbb{R}^{2048}$:**
   - Maps variable-row, variable-column heterogeneous tables into a row-order-invariant descriptor summarizing marginal quantile geometry, rank-copula target associations, missingness structure, and baseline residual moments.
2. **Kernel Solution Embedding $z_{\mathrm{s}} = \Psi_{\mathcal{K}}(s) \in \mathbb{R}^{d_s}$ with Kernel $K_{\mathcal{K}}(s, s') = \langle \Psi_{\mathcal{K}}(s), \Psi_{\mathcal{K}}(s') \rangle_{\mathcal{H}}$:**
   - Represents candidate feature and residual transformations as unit-norm vectors in a reproducing kernel Hilbert space where functionally equivalent transformations on standardized probe distributions cluster together regardless of surface parameterization, and orthogonal residual directions satisfy $K_{\mathcal{K}}(s, s') \approx 0$.
   - Enables (a) fast learned compatibility ranking $v_{\psi}(z_{\mathrm{p}}, z_{\mathrm{s}})$ before inner cross-validation, (b) Gram-matrix diversity selection, and (c) **strict every-fold inner-CV referee gating (`A-14(f)`)**, which raises held-out win precision across `3,639` corpus tables from `56.22%` (unconditional) to **`99.28%`** (`137/138` admitted tables, mean `+2.3827 u` gain over a fixed nonlinear dictionary + ridge baseline).

---

## 3. Random-Split Audit, Training Loss Curves, and Disjoint Validation

### 3.1 Cryptographic Non-Official Split Audit
In the `recall` (absorbed) regime, per-family and per-table neural dossiers train across all rows of a table using pseudorandom support/query partitions, while test-time scoring conditions solely on official training-side rows (`X_train, y_train`) with a runtime `sys.addaudithook` blocking `y_test`. Deterministic PRNG replay across committed ledgers established:
- **Look-2 19-family dossiers (`2343.8` Elo):** `61,814` replayed training episodes $\to$ **`0` official-split hits** across all `91` contexts (`PAPER_NON_OFFICIAL_SPLIT_AUDIT_RECEIPT_v2.json`).
- **`GEN2A` 91-table dossiers (`2408.5` Elo):** `155,937` replayed training episodes (`147,856` logged steps, `dup_splits = 0` in all `91` final records) $\to$ **`0` exact official-split partition hits on `91/91` contexts** (`67` contexts verified by index-set inequality at equal row count; `24` contexts with `1`--`178` filtered/deduplicated rows verified by content-hash overlap to have a maximum observed training-side `SEARCH` row share of **`82.33%`** on `k2_f8fd51f3`; receipt `ee7e4be4`).

### 3.2 Dossier Training Loss vs. Fresh-Row Generalization
Across all `91` committed `GEN2A` `log.jsonl` files (`23.7M`-parameter `M` reader, three `1,024`-bin distributional heads: robust residual `rr`, Yeo--Johnson `yj`, and Rank-Gauss `rg`):
- **100 exposures/row:** 3-head median cross-entropy `4.5145 nats` (IQR `[3.3687, 5.1566]`), median `7,174.9` query rows/s.
- **500 exposures/row:** 3-head median cross-entropy `3.7189 nats` (IQR `[2.7478, 4.5713]`); **20% held-out fresh-row cross-fitted utility peaks at `1.185 u`** (up `+0.333 u` from `0.852 u` generalist init across 10 families).
- **1,000 exposures/row:** 3-head median cross-entropy `2.4631 nats`, fresh-row utility `1.170 u`, median `7,366.6` query rows/s.
- **2,000 exposures/row:** 3-head median cross-entropy **`1.9840 nats`**, fresh-row utility `1.043 u` (`+0.191 u` above init), driving the `2343.8`--`2408.5` Elo absorbed board performance (`2.1` to `15.8` min/table on one 24 GB GPU; `3.22 h` wall-clock across 4 GPU slots for all `81` new tables).

### 3.3 Five-Host Foundation Training (`Bank r5` $\to$ `Bank r6`) & 32-Table Bank-Disjoint Validation
- **Live Fleet Scale:** Five concurrent trainers (`3` `L` at `104.4M` params + `2` `M` at `23.7M` params across `xbabe1`--`xbabe4` plus dedicated CPU harvester) have processed **`129.9M` cumulative dataset episodes** (`22.0M` post-ruling all-data episodes on `Bank r5`, `4,826` tables / `2,736` families, `bank_sha256=41c8aafb`) at `212`--`350` datasets/s, while preparing the `4,999`-table `Bank r6` 72-hour run with `59,849+` multi-split zero-loser teacher realizations and zero sealed-cohort overlap.
- **32-Table Bank-Disjoint Validation (`30ccaa2a`):** On `32` real tables verified disjoint from every historical bank (`fit_icl` through `r5`), `M`-scale validation cross-entropy improves monotonically across unseen-board checkpoints (`3.9569 -> 3.9282 -> 3.9087 nats`; median NRMSE `0.8216 -> 0.8103`) and across expanded all-data banks `r3 -> r4 -> r5` (`3.9605 -> 3.9539 -> 3.9480 nats`).
- **Context Scaling & Inference Cost:** Across `371` regression tables (`5ad51fc7`), scaling context length from `256` to `2,048` rows (`v=8` views) improves median NRMSE from `0.5886` to **`0.4864`** (`327/371` wins). Producing one three-head neural reader block (`2x7`-fold CV + full refit) takes `28.1 s` mean wall time across `18` tables, of which **`0.56 s` (`2.0%`) is GPU compute (`3.88M` table cells/GPU-s)**.

---

## 4. Public Benchmark Matrix (`TabArena v0.1` & `BeyondArena 2026`)

All public evaluations enforce pre-registered per-regime ladders (`RF (default) = 1000` Elo anchor, `1,000` bootstrap rounds, `PrivateNetwork=yes`, zero paid API credits) comparing **EurekaZero** against **`TabPFN-3.5`**, **`TabFM-1.0 Single`**, **`Kumo-Tabular-Large`**, and **`Kumo-Tabular-Small`**:

1. **Wave A (`44` Public Regression Tasks = `36` `BeyondArena` + `8` `TabArena`, `xbabe1` RTX 4090):**
   - Headlines `20` full-TEST `unseen` `BeyondArena` tasks (`+9` `in-corpus` full-TEST + `7` `unseen` `sub8192` head-to-head tasks) and `7` `in-corpus` `TabArena` tasks (`LOOK_LEDGER.jsonl` lines 43--48).
   - **Official Upstream R2 Refresh Reference (`LOOK_LEDGER` line 60, `UPSTREAM_R2_REFRESH_SCOREBOARD.json` `f9af8e90`; *published methods only (official data.tabarena.ai R2 refresh through 2026-11-12); field Elo computed by us on fold 0 over the 44 PUBLIC regression tasks of TASKS_44 (36 BeyondArena + 8 TabArena); this is NOT a Wave A read (Ours and local external runs not included)*):** On `BeyondArena/regular_29_full_coverage` (`29` full-TEST tasks, `40` methods; `TA-LIMIX-2-1016` and `TA-TABFM` dropped as partial `28/29` missing `sf_permit_time`): `#1` `TA-KUMO-TABULAR (default)` `1723.48` `[1562.31, 1999.91]`, `#2` `TA-KUMO-TABULAR-MEDIUM (default)` `1636.50` `[1475.42, 1879.22]`, `#3` `TA-KUMO-TABULAR-SMALL (default)` `1603.51` `[1465.61, 1822.73]`, `#4` `TA-CAUSILO (default)` `1555.60` `[1439.87, 1757.10]`, `#5` `TA-TABPFN-3.5 (default)` `1552.59` `[1401.10, 1774.63]`; on `BeyondArena/common_28_all_4_fm_families_clean_no648` (`28` tasks, `42` methods): `#1` `TA-KUMO-TABULAR (default)` `1740.44` `[1574.90, 2012.81]`, `#2` `TA-LIMIX-2-1016 (default)` `1729.84` `[1545.49, 2051.69]`, `#3` `TA-KUMO-TABULAR-MEDIUM (default)` `1654.24` `[1492.81, 1911.89]`, `#4` `TA-KUMO-TABULAR-SMALL (default)` `1625.43` `[1492.92, 1824.42]`, `#5` `TA-TABPFN-3.5 (default)` `1589.13` `[1433.60, 1809.71]`, `#11` `TA-TABFM (default)` `1483.83` `[1365.25, 1672.84]`; on `TabArena/all_8_clean_no648` (`8` tasks, `75` methods): `#1` `TA-LIMIX-2-1016 (default)` `2684.87` `[2353.01, 5800.77]` and `#2` `TA-KUMO-TABULAR (default)` `2684.87` `[2460.56, 3622.15]`, `#3` `TA-KUMO-TABULAR-MEDIUM (default)` `2478.84` `[2291.38, 3299.61]`, `#6` `TA-TABPFN-3.5 (default)` `2347.72` `[2165.11, 3011.79]`, `#7` `TA-KUMO-TABULAR-SMALL (default)` `2299.53` `[2111.10, 3062.79]`, `#8` `TA-TABFM (default)` `2255.74` `[2070.46, 2919.07]` (`TA-KUMO-TABULAR` L/M/S published on all `44` tasks: `verified=false` on `2026-10-06`, `verified=true` at `tabarena@9f985c0f`; no number from line 60 is compared against EurekaZero).
   - **Pre-Refresh Wave A External Kumo Read (`R_K` `f63ff1ce`, `88/88` completed, `SCORES` `1254231b`, `FIELD_ELO` `b2728e4c`; provisional under line 48 — external only, Ours not yet scored, TabFM not yet scored; field Elo computed on fold 0 anchored at `RF (default) = 1000`):**
     * **`BeyondArena` (`29` full-TEST regression tasks, 31 methods):** `Kumo-Tabular-Large` (local) ranks **`#1/31` at `1702.7` Elo** `[1539.8, 1965.6]` (`22-0-7` vs published `TA-TABPFN-3.5`, `20-0-9` vs `#3` `TA-CAUSILO` `1599.0`; `1862.9` `#1` minus the 8 context-capped tasks with width-capped tasks kept); `Kumo-Tabular-Small` (local) ranks **`#2/31` at `1624.1` Elo** `[1489.7, 1843.7]` (`16-0-13` vs `TA-CAUSILO`, `13-0-16` vs `#4` `TA-TABPFN-3.5` `1588.4`). On the `7` `8,192`-row subsampled unseen tasks (head-to-head only), `Kumo-Large` leads `Kumo-Small` `5-0-2`.
     * **`TabArena v0.1` (`8` public regression tasks, 71 methods):** `#1` `TA-LIMIX-2 (default)` (`2453.9`, `diagnostic_only`: *release checkpoint ed01f7a0 (HF de07b679); upstream found it contaminated with benchmark data (autogluon/tabarena#648; makers' audit 2026-11-11); TabArena dropped it 2026-11-16 for LimiX-2-1016*), `#2` `TA-TABPFN-3.5` (`2301.6`), **`#3/71` `Kumo-Tabular-Large` (`2192.2` Elo `[2050.6, 2850.5]`, `4-0-4` vs `TA-TABPFN-3.5`; rises to `#2` at `2704.0` minus the 3 context-capped tasks with width-capped `QSAR-TID-11` kept)**, and **`#8/71` `Kumo-Tabular-Small` (`1964.2` Elo, `3-0-5` vs `TA-TABPFN-3.5`)**.
2. **Wave B (`129` Public Classification Tasks = `94` `BeyondArena` + `35` `TabArena`, `xbabe2` RTX 3090):**
   - Headlines `32` `unseen` `BeyondArena` tasks (`+62` `in-corpus`) and `32` `in-corpus` `TabArena` tasks (`+3` `unseen` W--T--L), backed by `676`-table binary (`0.7568` vs `0.7516` `CatBoost` ROC-AUC) and `199`-table multiclass (`81.4%` win rate vs `CatBoost`, `95.5%` vs `LightGBM` on LogLoss) parity verification (`WP5`).
3. **Post-Pre-Read Absorption (`T9-A` & `T9-B`):**
   - Strictly after Phase 1 `unseen`/`in-corpus` pre-reads lock, random-split per-task dossiers (`T9-A` on `44` regression tasks and `T9-B` on `129` classification tasks) execute and score a single pre-registered `recall` (absorbed) look per suite.

---

## 5. Manuscript & Evidence Map

- **Standalone Modular Black-Hole Manuscript:** [`docs/paper/blackhole/main.tex`](../../paper/blackhole/main.tex) (`14` modular sections in `docs/paper/blackhole/sections/`, `4` `pgfplots`/TikZ figures in `docs/paper/blackhole/figures/`, `14` measured CSVs in `docs/paper/blackhole/data/measured/`, and `11` receipt-bound claims in `docs/paper/blackhole/evidence/claims.json`).
- **EurekaZero Main White-Paper Additions:** [`docs/whitepaper/sections/07a_blackhole_dev91.tex`](../../whitepaper/sections/07a_blackhole_dev91.tex), [`docs/whitepaper/sections/07b_public_benchmarks_tabarena_beyondarena.tex`](../../whitepaper/sections/07b_public_benchmarks_tabarena_beyondarena.tex), and [`docs/whitepaper/sections/07c_blackhole_training_dynamics.tex`](../../whitepaper/sections/07c_blackhole_training_dynamics.tex).
- **Wave Context & Committed `GEN2A` Dossier Logs:** [`docs/research/waves/h458-h465-black-hole-20261116/context.md`](../waves/h458-h465-black-hole-20261116/context.md) and `research/flywheel_selfplay/h458_test_informed_flywheel/evidence/h465_gen2a/`.
