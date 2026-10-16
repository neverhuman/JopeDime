# JopeDime / CLOVER Experiment Catalog (`docs/experiments/`)

This directory is the canonical, compressed, LLM-ready index of every quantitative experiment, pilot, ablation, and qualification protocol executed in **JopeDime / CLOVER**.

It is designed so an LLM or researcher can ingest the entire experimental history, exact dataset provenance, metrics, negative results, and next-experiment implications in **under 4,000 tokens**.

---

## Directory Layout

| File | Format | Purpose |
| :--- | :--- | :--- |
| [`README.md`](README.md) | Markdown | Executive summary matrix, compressed experiment cards (`EXP-001`..`EXP-012`), and update protocol. |
| [`experiments.jsonl`](experiments.jsonl) | JSON Lines (`1` record/line) | Machine- and LLM-consumable ledger of all experiments conforming to [`schema.v1.json`](schema.v1.json). |
| [`schema.v1.json`](schema.v1.json) | JSON Schema (2020-12) | Strict schema (`jope.experiment-record/v1`) requiring `summary`, `data`, `protocol`, `metrics`, `outcome`, `artifacts`, and `next_experiment_implications`. |
| [`SYNTHESIS_AND_NEXT_EXPERIMENTS.md`](SYNTHESIS_AND_NEXT_EXPERIMENTS.md) | Markdown | Cross-experiment empirical laws (what worked, what failed, and why) plus the prioritized next-experiment portfolio (`EXP-NEXT-01`..`EXP-NEXT-04`). |

---

## Qualification Status Taxonomy

Every experiment carries an explicit `qualification_status` so diagnostic runs, pre-seal pilots, and negative findings are never conflated with qualified claims:

- **`ESTABLISHED`**: Verified on frozen receipts with full provenance and parity checks within its declared scope.
- **`PRELIMINARY`**: Directional signal on real data, but predates a full contamination/split seal (e.g., pre-`R6` `T1/D5b` matches).
- **`NON_QUALIFYING`**: Executed experiment that produced an inconclusive result, a negative result, or triggered a regression/incomplete-coverage gate.
- **`DIAGNOSTIC_ONLY`**: Verified offline contract/verifier or preregistered protocol awaiting live compute execution.
- **`FAILED_BEFORE_EXECUTION`**: Run aborted at preflight/controller guard before collecting task measurements.

---

## Master Experiment Matrix (`EXP-001` – `EXP-012`)

| ID | Phase / Hypothesis | Title | Data & Split Scope | Primary Quantitative Outcome | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`EXP-001`** | `H3` Distillation | Generation-1 PolicyLM (`v1` / `R21`) Continuous Training | `6,982` train / `263` val contexts (`291` clusters), `6,640` AST catalog | Val `L6`: `1.0055 -> 0.8968` (`-10.8%`, ep 28); `CE/BT/listwise/runtime` improved; **`q`-head flat (`0.1804 -> 0.1809`)**; 31-ep plateau after step `5,677` | `ESTABLISHED` |
| **`EXP-002`** | `H2` Selection | Outcome Head (`q-hat`) Fresh Ranking & GBDT Comparison | `32` dev contexts (`D5b`) + equal-lineage `DL_wave4` splits | Fresh pairwise acc `0.506 [0.441, 0.568]` (chance); `32->8` rerank `dScore +0.031 [-0.0625, +0.125]` (null); **neural $R^2$ `0.690` vs LightGBM `0.785` (`-0.095`)** | `ESTABLISHED` |
| **`EXP-003`** | `H4` Task Win | Best-of-8 Dev Panel Matches (`T1` / `D5b`) | `32` dev contexts $\times$ `8` program evaluations/arm | Mean score: `teacher 2.134`, `policy+q 2.013`, `policy 1.994`, `uniform 1.542` (**4.2x lower regret vs uniform**, pre-`R6` split seal) | `PRELIMINARY` |
| **`EXP-004`** | `Phase 0` Parity | `C7` Native vs. Emitted Rust Evaluator Parity Census | `6,640` catalog programs (`1,073` quantile); `1,554` proposals | `41/1,554` (`2.64%`) failed parity; **`41/342` (`11.99%`) in `quantile_bin`**, **`0/1,212` (`0.0%`) in non-quantile** | `ESTABLISHED` |
| **`EXP-005`** | `Phase 0` Parity | `v2` (`continuous_v6` / `R23`) Bitwise Build Qualification | `step-17344` checkpoint + `R22` batch + `230` CPU fixtures | `8/8` GPU runs bitwise identical (`loss = 0.9540845`, `0` tensor mismatches across `893,957` params); `230/230` CPU fixtures pass | `ESTABLISHED` |
| **`EXP-006`** | `H3` Distillation | `R22` Scratch Sweeps (`3, 4a, 4b, 5, 6`): `KL` & `L_PI` Trade-Off | `488` active train, `263` val, `885` held-out train (`1,636` total) | `0.2*KL` capped 8-ep KL drift at `<= 0.014` (vs `0.052+` ep 2 unanchored); **`L_PI` improved `7.080 -> 6.59..6.87` but worsened `L6` `0.8610 -> 0.8621..0.8660`** | `NON_QUALIFYING` |
| **`EXP-007`** | `H3` Distillation | `R25/R26/R27` kNN Sweeps & `1,148`-Context Paired Judge | Trained on `488` contexts; judged on `1,148` (`263` val + `885` train) | **`REGRESSION ALERT` on all 3 judged arms vs frozen `v1`**: `r25control +0.00310`, `R26 +0.00333`, `R25 +0.00511`, driven by `+0.0042..+0.0063` forgetting on `885` unvisited lineages | `NON_QUALIFYING` |
| **`EXP-008`** | `H1` GP Expansion | `GB-06` / `H1` Bounded GP Vocabulary Expansion Pilot (`xbabe3`) | `24` frozen train contexts $\times$ `3` arms (`512` fits/arm cap) | `17/24` complete: **`retrieval_seeded` vs `ordinary` `16/1/0` (`-0.0354` median log-loss ratio)**; `policy_seeded` vs `ordinary` `10/2/5` (`-0.000349`); **`40/51` (`78.4%`) novel out-of-catalog ASTs**; `48/51` parity | `NON_QUALIFYING` |
| **`EXP-009`** | `H2` / `H3` | `J9` Frozen-Pool Selector Audit & 5-Round Offline Distill | `1,222` frozen pools (`21,199` rows) + `8,908` S3 GP index rows | **`strongest_baseline: ridge_program`**, `ranking_learnability: NOT_ESTABLISHED`, `flywheel_established: false`; `--release` cut runtime `17.4x` (`767s -> 44s`) and `14.6x` (`8125s -> 558s`) | `NON_QUALIFYING` |
| **`EXP-010`** | `Benchmarks` | Core-IP Bounded 1-Hour GPU Latency Window (`step-60000`) | `10` tabular datasets planned on RTX 3090 | Aborted at `428ms` (`0/10` datasets) due to pre-existing `window.json` receipt guard and staging path mismatch; fixed in `v3` window contract | `FAILED_BEFORE_EXECUTION` |
| **`EXP-011`** | `Benchmarks` | Same-Cut 10-Dataset Benchmark & SOTArena Elo Replay | `10` same-cut datasets + `52,329` SOTArena registry entries | `10/10` same-cut datasets completed in `273s` CPU; `52,329`-entry Elo ladder replayed byte-identically (`jope-arena elo --check`) | `ESTABLISHED` |
| **`EXP-012`** | `H1` GP Expansion | `SPQ-1` Self-Play Round Qualification Verifier & Round-1 Prereg | Preregistered for `>= 24` parent clusters, `B_total = 64` (`4x16` vs `1x64`) | Merged in PR #343 (`c402b2118`): enforces cluster bootstrap (`margin <= -0.01`, `alpha = 0.025`), chain-rule KL decomposition (`base_kl <= 0.20`, `new_mass <= 0.15`), and derived admissibility | `DIAGNOSTIC_ONLY` |

---

## Compressed Experiment Cards (`EXP-001` – `EXP-012`)

### `EXP-001-v1-flywheel-baseline` — Generation-1 PolicyLM Continuous Training (`v1` / `R21`)
- **Summary**: Trained the `893,957`-parameter PolicyLM (`d_model=128`, 3-layer Transformer) on the `6,640`-program catalog under unanchored `L6 = CE + 0.5*BT + 0.5*listwise + 0.25*q-MSE + 0.1*runtime` for `17,773` steps (`40` continuation epochs).
- **Data**: `6,982` train contexts, `263` held-out validation contexts (`291` total parent clusters, `R6` lineage split seal + `R5` inner-fold target fix).
- **Outcome (`ESTABLISHED`)**:
  - Validation `L6` dropped `-10.81%` from `1.005528` (ep 1) to `0.896815` (ep 28 best) and `0.897510` (ep 40), plateauing across epochs `10..40` in `[0.896815, 0.897791]`.
  - Policy heads improved cleanly: `CE` `1.0644 -> 0.8950`, `BT` `0.5625 -> 0.5255`, `listwise` `0.7420 -> 0.7037`, `runtime` `0.1197 -> 0.1066`.
  - **Negative finding**: Outcome regression head (`q-hat`) did not learn (`0.1804 -> 0.1809`), and `ReduceLROnPlateau` halved LR `9` times to the `1e-6` floor by step `5,677`.

### `EXP-002-qhat-fresh-program-ranking` — Outcome Head (`q-hat`) Ranking & LightGBM Comparison
- **Summary**: Evaluated `q-hat` pairwise ranking and `32 -> 8` candidate reranking on fresh PolicyLM draws across `32` held-out contexts (`D5b`), and compared neural outcome prediction against LightGBM on equal-lineage splits (`DL_wave4`).
- **Data**: `32` kept development contexts (`D5b`) + `DL_wave4` equal-lineage cross-validation splits.
- **Outcome (`ESTABLISHED`)**:
  - Pairwise ranking accuracy on fresh programs is `0.506` (`95% CI [0.441, 0.568]`) — statistically indistinguishable from a `0.50` coin flip.
  - Reranking top-8 from 32 policy samples gave `dScore = +0.03125 [-0.0625, +0.125]` and `dBestQ = +0.0187 [-0.0320, +0.0811]` (`INCONCLUSIVE`).
  - On `DL_wave4`, the neural outcome predictor achieved $R^2 = 0.690$ vs. LightGBM $R^2 = 0.785$ ($\Delta R^2 = -0.095$, `95% CI [-0.133, -0.056]`).

### `EXP-003-dev-panel-matches-t1-d5b` — Best-of-8 Development Panel Matches (`T1` / `D5b`)
- **Summary**: Benchmarked best-of-8 candidate draws from `teacher`, `policy`, `policy + q-rerank`, and `uniform` catalog sampling across `32` development contexts.
- **Data**: `32` development contexts $\times$ `8` evaluated candidate programs per arm (`6,640` catalog).
- **Outcome (`PRELIMINARY`)**:
  - Mean scores: `teacher = 2.134`, `policy + q-rerank = 2.013`, `policy = 1.994`, `uniform = 1.542`.
  - `policy` reduced regret vs. `teacher` from `0.591` (`uniform`) to `0.140` (`4.22x` regret reduction), but remains `PRELIMINARY` because `T1/D5b` predated the `R6` lineage split seal.

### `EXP-004-evaluator-quantile-parity-census` — `C7` Native vs. Emitted Rust Evaluator Parity Census
- **Summary**: Audited prediction equivalence between the in-memory AST evaluator and emitted standalone Rust code across `1,554` harvested proposals from the `6,640`-program catalog.
- **Data**: `6,640` catalog programs (`1,073` = `16.16%` containing `quantile_bin`); `1,554` tested proposals (`342` with `quantile_bin`, `1,212` without).
- **Outcome (`ESTABLISHED`)**:
  - `41/1,554` (`2.64%`) proposals failed exact parity: **`41/342` (`11.99%`) among `quantile_bin` programs** and **`0/1,212` (`0.0%`) among non-quantile programs**.
  - Root cause isolated 100% to boundary tie-breaking in `quantile_bin`.

### `EXP-005-v2-r23-bitwise-build-qualification` — `v2` (`continuous_v6` / `R23`) Bitwise Build Qualification
- **Summary**: Verified bitwise reproducibility of the refactored `continuous_v6` (`v2`) trainer under `R23` (`L6 + 0.2*KL`) across `8` GPU receipts and `230` offline CPU contract fixtures.
- **Data**: Frozen `step-17344` checkpoint + deterministic `R22` batch + `230` offline CPU fixtures.
- **Outcome (`ESTABLISHED`)**:
  - All `8` GPU runs produced exact IEEE-754 next-update loss `0.9540845` with `0` parameter tensor mismatches across `893,957` parameters; `230/230` CPU fixtures passed.

### `EXP-006-r22-scratch-sweeps-3-to-6` — `R22` Scratch Sweeps (`3, 4a, 4b, 5, 6`): `KL` & `L_PI` Trade-Off
- **Summary**: Ran five 8-epoch GPU scratch sweeps from frozen `v1` on the `488`-context `R22` active train subset (with `885` train contexts held out) testing `0.2*KL` anchoring and $\lambda_{\text{PI}} \in \{0.01, 0.03, 0.10\}$.
- **Data**: `1,636` kept `R22` contexts split into `488` active train, `263` val, and `885` held-out train contexts (`221` parent clusters).
- **Outcome (`NON_QUALIFYING`)**:
  - `0.2*KL` bounded 8-epoch probe KL drift to `<= 0.014` (vs. `>= 0.052` by epoch 2 in unanchored Sweep 3).
  - Adding `L_PI` improved held-out `L_PI` (`7.080 -> 6.59..6.87`) but degraded held-out `L6` (`0.8610 -> 0.8621..0.8660`).

### `EXP-007-r25-r26-r27-and-1148-context-judge` — `R25/R26/R27` kNN Sweeps & `1,148`-Context Judge
- **Summary**: Tested kNN neighborhood cross-entropy regularization (`R25` $\lambda=0.25$, `R26` replicate, `R27` $\lambda=0.05$) against `r25control` (`L6 + 0.2*KL`) and evaluated on the combined `1,148`-context paired judge (`263` val + `885` held-out train).
- **Data**: Trained on `488` active train contexts; judged on `1,148` held-out contexts (`291` total clusters).
- **Outcome (`NON_QUALIFYING`)**:
  - `R25` ($\lambda=0.25$) spiked probe KL to `0.1079` at epoch 1 and was automatically frozen by the KL guard.
  - **Crucial discovery**: On the `1,148`-context judge, **all 3 judged runs triggered `REGRESSION ALERT` vs. frozen `v1`**: `r25control ep4` `+0.00310 [+0.00195, +0.00434]`, `R26 ep4` `+0.00333 [+0.00214, +0.00461]`, `R25` `+0.00511 [+0.00413, +0.00611]`.
  - Sub-split decomposition revealed that the `263` val contexts were flat (`-0.0001` to `-0.0005`), whereas the `885` unvisited train lineages degraded by `+0.0042` to `+0.0063` — proving that fine-tuning on a `488`-context subset causes cross-lineage forgetting.

### `EXP-008-gb06-h1-bounded-gp-expansion-pilot` — `GB-06` / `H1` Bounded GP Vocabulary Expansion Pilot
- **Summary**: Executed a 3-arm real-budget GP pilot (`ordinary`, `policy_seeded` from frozen `v1`, `retrieval_seeded` from nearest-neighbor training dataset) across `24` frozen training contexts on `xbabe3` (`512` fits/arm cap, `900s`/arm timeout).
- **Data**: `24` frozen training contexts (`19` classification, `5` regression); `53/72` arms completed, `18/72` interrupted at timeout across `6` wider datasets (`17/24` complete-case contexts across all 3 arms).
- **Outcome (`NON_QUALIFYING`)**:
  - **`retrieval_seeded` dominated `ordinary` GP**: `16` wins, `1` tie, `0` losses (`median log_loss_ratio = -0.03542`, `mean = -0.08823`).
  - **`policy_seeded` (frozen `v1`) gave negligible median gain over `ordinary` GP**: `10` wins, `2` ties, `5` losses (`median log_loss_ratio = -0.000349`, `mean = +0.06281` skewed by outlier `ctx-001279` where policy seeding stuck at `3.081` vs. `0.675`).
  - **Vocabulary expansion proven**: `40/51` (`78.43%`) valid GP champions were novel ASTs outside the `6,640`-program `v1` catalog.
  - **Parity**: `48/51` valid champions matched emitted Rust bitwise; `3/51` failed parity on `div` / `log1p` domain edge cases.

### `EXP-009-j9-frozen-pool-selector-and-flywheel-distill` — `J9` Frozen-Pool Selector & 5-Round Distill
- **Summary**: Ran the offline CLOVER Phase 2/3 host pipeline (`jope_pools` + `jope_retention`) across `1,222` frozen candidate pools (`21,199` rows) and `8,908` S3 GP index rows over 5 rounds (`eval_budgets = [1, 4, 8]`).
- **Data**: `1,382` kept `R22` contexts $\to$ `1,222` frozen pools (`864` train, `180` val, `178` test; `21,199` rows) + `8,908` S3 GP index rows (`426` programs across `10` datasets).
- **Outcome (`NON_QUALIFYING`)**:
  - On frozen pools, simple **`ridge_program` remained the `strongest_baseline`** (`ranking_learnability: NOT_ESTABLISHED`).
  - On static S3 GP index rows, 5-round offline distillation returned `flywheel_established: false`.
  - Compiling with `--release` reduced `evaluate` runtime by `17.4x` (`767s -> 44s`) and `flywheel-distill` runtime by `14.6x` (`8,125s -> 558s`).

### `EXP-010-core-ip-gpu-latency-study` — Core-IP Bounded 1-Hour GPU Latency Window
- **Summary**: Dispatched a bounded 1-hour RTX 3090 latency benchmark (`step-60000` policy across `10` datasets) that aborted at `428ms` due to an existing `window.json` receipt guard and staging path mismatch.
- **Data**: `10` tabular benchmark datasets (`0/10` timed).
- **Outcome (`FAILED_BEFORE_EXECUTION`)**:
  - Fail-closed receipt check prevented overwriting prior artifacts; fixed in the `v3` timing-window contract.

### `EXP-011-same-cut-and-sotarena-elo-benchmarks` — Same-Cut 10-Dataset Benchmark & SOTArena Elo Replay
- **Summary**: Verified deterministic execution of the 10-dataset same-cut CPU benchmark and byte-identical replay of the `52,329`-entry SOTArena Elo ladder (`jope-arena elo --check`).
- **Data**: `10` same-cut tabular datasets + `52,329` historical SOTArena registry entries across `15` model families.
- **Outcome (`ESTABLISHED`)**:
  - `10/10` same-cut datasets completed in `273s` CPU wall-clock; `52,329`-entry Elo replay verified byte-identical.

### `EXP-012-spq1-selfplay-round-qualification-prereg` — `SPQ-1` Self-Play Round Qualification & Prereg
- **Summary**: Implemented and merged (`PR #343`, commit `c402b2118`) the fail-closed `SPQ-1` round qualification verifier (`jope.selfplay-round-qualification/v1`) and froze the Round-1 equal-budget preregistration (`B_total = 64`).
- **Data**: Preregistered for `>= 24` held-out parent clusters comparing Arm A (`1 x 64` single-round ordinary GP) vs. Arm B (`K = 4 x 16` multi-round self-play ratchet with roundwise distillation).
- **Outcome (`DIAGNOSTIC_ONLY`)**:
  - Solves the expanding-vocabulary KL divergence problem via chain-rule decomposition (`conditional_base_kl <= 0.20` on shared base vocabulary + `new_support_mass <= 0.15` on novel GP programs) and derived admissibility verification.
  - Ready to gate `EXP-NEXT-02` (live Round-1 self-play campaign).

---

## How to Add Future Experiments (`EXP-013+`)

1. Append a single-line JSON record to [`experiments.jsonl`](experiments.jsonl) with the next sequential ID (`EXP-013-...`) conforming to [`schema.v1.json`](schema.v1.json).
2. Add a corresponding row to the **Master Experiment Matrix** and a compressed card to [`README.md`](README.md), and update [`SYNTHESIS_AND_NEXT_EXPERIMENTS.md`](SYNTHESIS_AND_NEXT_EXPERIMENTS.md) if priorities shift.
3. Validate before committing:
   ```sh
   jq -c . docs/experiments/schema.v1.json >/dev/null
   jq -c . docs/experiments/experiments.jsonl >/dev/null
   git diff --check
   bash deploy/ci/check.sh
   bash deploy/ci/ip-scrub.sh
   ```
