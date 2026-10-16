# JopeDime / CLOVER Curated Experiment Digest (`docs/experiments/`)

This directory provides a **curated, non-exhaustive 12-record digest (`EXP-001`–`EXP-012`) as of 2026-10-16** (compiled at `main` commit `c402b21183670d3e4df73b835292a385a04db30f`) in a structured JSONL + Markdown format for LLM and script consumption.

> **Authoritative Ledger & Scope Note:** [`docs/paper/claims.json`](../paper/claims.json) (`177` claims: `118` established, `20` preliminary, `16` non-qualifying, `23` pending, traced in [`docs/paper/NUMBERS_TRACE.md`](../paper/NUMBERS_TRACE.md)) remains the repository's authoritative claim ledger. Every numerical value in this folder is taken directly from a cited `Cxx` entry in `claims.json` or from a named tracked receipt (`gp-h1-readout.20261013.json`, `gpu-closeout.json`, `selfplay-round1-prereg.v1.json`, `NUMBERS_TRACE.md#fresh-clone-run`).

---

## Directory Layout

| File | Format | Purpose |
| :--- | :--- | :--- |
| [`README.md`](README.md) | Markdown | Scope and known omissions, status taxonomy, master matrix (`EXP-001`..`EXP-012`), compressed experiment cards, and validation commands. |
| [`experiments.jsonl`](experiments.jsonl) | JSON Lines (`1` record/line) | Machine- and LLM-consumable digest of the 12 curated records conforming to [`schema.v1.json`](schema.v1.json). |
| [`schema.v1.json`](schema.v1.json) | JSON Schema (Draft 2020-12) | Strict schema (`jope.experiment-record/v1`) enforcing 40-hex or `"unknown"` `source_commit`, `catalog_snapshot_commit`, `claim_ids`, and typed `metrics`. |
| [`SYNTHESIS_AND_NEXT_EXPERIMENTS.md`](SYNTHESIS_AND_NEXT_EXPERIMENTS.md) | Markdown | Five scope-bounded working hypotheses (non-qualifying) and a prioritized roadmap of next-step proposals (`EXP-NEXT-01`..`EXP-NEXT-04`) with explicit admission gates. |

---

## Known Omissions (`Not Yet Indexed as Standalone Rows in experiments.jsonl`)

Because this folder is a 12-entry curated digest rather than an exhaustive dump of every historical run, the following executed studies and sub-panels remain in [`docs/paper/claims.json`](../paper/claims.json) or host evidence custody and are not yet expanded into separate `EXP-xxx` rows:

1. **2026-10-14 / 2026-10-15 `EI-0`, `EI-1`, `EI-2`, `EI-2X`, `EI-3`, and `EI-4` `1,148`-context paired studies:** Host-local receipts reside under `/home/ubuntu/jope-evidence/{jopedime-flywheel-ei0,jopedime-flywheel-ei1,jopedime-flywheel-ei2,ei3-study-20261015,ei4-study-20261015}` (note that `C175`'s `"EI-1 planned only"` statement in the paper write-up is scope-dated `2026-10-13` per [`NUMBERS_TRACE.md`](../paper/NUMBERS_TRACE.md#findings-and-corrections) correction #14).
2. **Individual rows for `C182`'s four post-sweep scratch runs and five judge records:** Summarized jointly under `EXP-007` (`r25control`, `R25`, `R26`, `R27`; judge records `sweep6-verdict-8b7aaead.json`, `r23e4-v1-verdict-de48c164.json`, `r25-e1-verdict-de66cb6a.json`, `j21-paired-verdict-98fa563d.json`, and `l6-1148-verdict.json`).
3. **Sweep 3 own-control paired verdicts at epochs 1–3:** Recorded in [`NUMBERS_TRACE.md`](../paper/NUMBERS_TRACE.md#findings-and-corrections) correction #8 (`C163` notes no paired verdict against frozen `v1`, whereas Sweep 3's two `L_PI` arms were judged against their own `L6` control at epochs 1–3).
4. **`J21` (`C170`) and `J22` judge panels and Wave 1–4 encoder studies (`C178`, `C179`):** Summarized only where referenced by `EXP-002` and `EXP-007`.

---

## Qualification Status Taxonomy

Every record carries an explicit `qualification_status` matching [`schema.v1.json`](schema.v1.json):

- **`ESTABLISHED`**: Matches an `established` entry `Cxx` in [`docs/paper/claims.json`](../paper/claims.json) or a verified fresh-clone reproduction row in [`docs/paper/NUMBERS_TRACE.md`](../paper/NUMBERS_TRACE.md#fresh-clone-run) as of 2026-10-16 (means the value was verified at the cited locator within its stated scope; does not imply live closed-loop generalization).
- **`PRELIMINARY`**: Directional or historical evaluation whose split or roster predates a full contamination/split seal or has censored contexts (e.g., pre-`R6` `D5b` 32-context panel in `EXP-002` and `EXP-003`).
- **`NON_QUALIFYING`**: Executed scratch run, complete-case inner-fold pilot, or command smoke run that is explicitly non-qualifying (`R22` scratch sweeps, `L6-1148` regression alerts, `GB-06` `NON-QUAL` readout, or `R22` scratch host-step smoke outputs).
- **`DIAGNOSTIC_ONLY`**: Merged offline diagnostic verifier or unpinned pre-registration packet whose outputs unconditionally set `qualified_claim: false` (`EXP-012`).
- **`PENDING`**: Pre-registered or planned evaluation that has not yet been executed (`C158`, `C160`).
- **`FAILED_BEFORE_EXECUTION`**: Admitted run that halted at a preflight/controller guard before collecting task measurements (`EXP-010`).

---

## Master Experiment Matrix (`EXP-001` – `EXP-012`)

| ID | Date (UTC) | Source Commit | Claims / Receipt | Title & Data Scope | Primary Quantitative Readout | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`EXP-001`** | `2026-10-10` | `unknown` | `C01-C02, C05-C06, C08, C14-C15, C19-C23, C25-C28, C31-C32, C141, C180-C181` | **`v1` PolicyLM Continuous Run**: `6,857->6,982` train, `263->266` val (`1,068` lineages: `1,014` train, `54` val), `6,640` catalog | Val `L6` (`263` ctx): `1.0055212` (ep 2) $\to$ `0.8968201` (ep 28) $\to$ `0.8974925` (ep 40, step `17,416`; `0.8973272` at terminal step `17,773`); 31-ep plateau `[0.8968201, 0.8978393]` (ep 10..40); **`q` flat `0.1804 -> 0.1809`** | `ESTABLISHED` |
| **`EXP-002`** | `2026-10-08` | `unknown` | `C75-C78, C178, C179` | **`q`-Head Ranking (`D5b`) & `DL_wave4` vs. LightGBM**: `32` kept `D5b` contexts + `7,120`-view pool (`1,068` lineages) | `q`-head fresh pairwise acc `0.506 [0.441, 0.568]` (chance `0.5`); `32->8` rerank `dScore 0.031 [-0.0625, 0.125]` (`INCONCLUSIVE`); separate DL vs. LightGBM equal-lineage $R^2$ `0.690235` vs. `0.784889` (`[-0.133, -0.056]`) | `PRELIMINARY` |
| **`EXP-003`** | `2026-10-08` | `unknown` | `C72-C75, C77` | **Best-of-8 Dev Matches (`T1/D5b`)**: `32` kept contexts (of `53` roster) $\times$ `8` programs/arm | Expected best $q$ of 8: `teacher 2.134`, `rerank 2.013`, `policy 1.994`, `uniform 1.542`; mean regret vs. teacher `0.140` (`policy`), `0.121` (`rerank`), `0.591` (`uniform`) | `PRELIMINARY` |
| **`EXP-004`** | `2026-10-08` | `unknown` | `C41-C45, C47, C49, C149` | **Verification Parity Census**: `6,640` catalog (`1,636` = `24.64%` with binning); `1,554` evaluated pairs | `41/1,554` (`2.6%`, `40` programs) failed parity: **`41/342` (`12.0% [8.7%, 15.9%]`) with binning**, **`0/1,212` (`<= 0.25%`) without binning** (`s_lo*(1-f) + s_hi*f` one-ulp plateau edges + stdlib partition vs. midpoint search) | `ESTABLISHED` |
| **`EXP-005`** | `2026-10-11` | `unknown` | `C09, C118, C131, C160-C162` | **`v2` Build Qualification (`r2`–`r3`)**: `v1` next batch (step `17,774`) + `230` offline CPU fixtures | `8/8` CUDA receipts (`r2`–`r3`, incl. `r3` at KL `0`) bitwise equal on `48` model, `144` AdamW, `2` RNG tensors (`loss = 0.9540845`); `230/230` CPU fixtures pass (`0` skips) | `ESTABLISHED` |
| **`EXP-006`** | `2026-10-13` | `unknown` | `C136-C139, C163, C173, C176-C177` | **`R22` Scratch Sweeps (`3, 4a, 4b, 5, 6`)**: `885` train contexts (`167` lineages) held out in `4b–6` only (`3, 4a` had no holdout) | Pure `L6 + 0.2*KL` probe KL `<= 0.014` (`L6`-only `0.052–0.053` by ep 2 at `3.75e-5`; `L_PI + 0.2*KL` up to `0.020–0.057` and `0.102`); in `4b–6`, held-out `L_PI` `7.080 -> 6.591..6.865` while held-out `L6` `0.8610 -> 0.8621..0.8660` at last eval epoch | `NON_QUALIFYING` |
| **`EXP-007`** | `2026-10-13` | `unknown` | `C165, C169-C174, C177, C182, C183` | **Post-Sweep Runs (`r25control, R25, R26, R27`) & `L6-1148` Judge**: `1,148` contexts (`221` clusters = `54` val + `167` train) | **`REGRESSION ALERT` on all 3 judged arms vs. `v1`** (`C174`): `R25 ep1 +0.00511`, `r25control ep4 +0.00310`, `R26 ep4 +0.00333`; directional strata (`C177`, not decisive): `263` val `+0.00119 / -0.00070 / -0.00108`, `885` held-out train `+0.00628 / +0.00423 / +0.00463` | `NON_QUALIFYING` |
| **`EXP-008`** | `2026-10-13` | `bd716765c4...` (local custody ref) | `C166-C168`, `gp-h1-readout.20261013.json` | **Closed `H1` GP Pilot (`GB-06`)**: `24` planned train contexts (`53` completed, `18` interrupted across `6` ctx, `17` complete-case) | On `17` contexts (inner folds): `retrieval/ordinary` `16/1/0` (`median -0.035361`), `policy/ordinary` `10/2/5` (`-0.000349`), `policy/retrieval` `2/0/15` (`+0.017236`); `40/51` syntactically non-catalog (behavioral novelty unknown); `48/51` parity pass (`3` fail: ctx `14` retrieval, ctx `18` ordinary & policy) | `NON_QUALIFYING` |
| **`EXP-009`** | `2026-10-16` | `9fdff904cb...` | `NUMBERS_TRACE.md#fresh-clone-run` | **CLOVER Host Steps 3–5 Smoke Run**: `R22` scratch inputs (`1,382` ctx $\to$ `1,222` pools / `21,199` rows; `8,908` GP index rows) | All 7 commands exit `0`; `evaluate` `767s` (debug) vs. `44s` (release bin at `9fdff904c`) / `51s` (`cargo run --release` at `73f9327fc`); `flywheel-distill` `8,125s` vs. `570s` / `558s`; **R22 scratch outputs show only that commands run, not results** | `NON_QUALIFYING` |
| **`EXP-010`** | `2026-09-25` | `802fb464c8...` | `gpu-closeout.json`, `core-ip/sections/10.md` | **Core-IP 1-Hour GPU Latency Window**: `10`-dataset panel admitted `2026-09-25T19:11:34.939Z` | Stopped in `428ms` (`0` requests/arms/batches) due to immutable `window.json` receipt collision; separate staging protocol deviation parsed outer `cut0` labels via `load_development` | `FAILED_BEFORE_EXECUTION` |
| **`EXP-011`** | `2026-10-16` | `9fdff904cb...` | `NUMBERS_TRACE.md#fresh-clone-run` | **Fresh-Clone Arena & Same-Cut Reproduction**: `2` arena ladders + `10`-dataset same-cut + `feat-math` strict suite | `2 x jope-arena elo --check` byte-identical; 10-dataset same-cut run `273s` (`elo --check` byte-identical to `leaderboard-jope-same-cut-v1-deep.csv`); `feat-math` strict `1,168` passed, `12` ignored, `0` failed | `ESTABLISHED` |
| **`EXP-012`** | `2026-10-16` | `c402b21183...` | `C158`, `selfplay-round1-prereg.v1.json` | **`SPQ-1` Round-Qualification Verifier & Unpinned Round-1 Prereg**: `rounds = 4`, `seed_cap_b = 8` (`PRE_REGISTERED_UNPINNED_NO_EXECUTION_GRANT`) | Merged at `c402b2118` (`11` dedicated tests: `9` integration + `2` unit); checks `ln(ref) - ln(cand)` test RMSE (`alpha <= 0.025`), sequential registry looks, and caller KL (`base_kl <= 0.20`, `new_mass <= 0.15`); **`qualified_claim/promotion/l6_claim` always `false`** | `DIAGNOSTIC_ONLY` |

---

## Compressed Experiment Cards (`EXP-001` – `EXP-012`)

### `EXP-001-v1-flywheel-baseline` — Generation-1 PolicyLM Continuous Training (`v1` / `R17` Capture)
- **Date & Provenance**: `2026-10-10` | `source_commit: "unknown"` | Claims: `C01, C02, C05, C06, C08, C14, C15, C19, C20, C21, C22, C23, C25, C26, C27, C28, C31, C32, C141, C180, C181`.
- **Summary**: Trained the `893,957`-parameter PolicyLM (`d=128, L=2, H=4, d_ff=512, V=282, T_max=640`, `C01–C02`) over the `6,640`-program catalog (`C05`) for `429` first-pass updates (`844.3 s`, `C20–C21`) plus `17,344` continuation updates (`17,773` total updates across `40` closed lineage epochs + `357` updates in epoch 41, mean `3.550 s/update`, `C22–C23, C181`).
- **Data**: `1,068` parent lineages (`1,014` training, `54` validation, `C15`); `6,857` training and `263` validation contexts in first pass (`C14, C20`), growing from `6,859 + 264` to `6,982 + 266` during continuation (`C19`).
- **Outcome (`ESTABLISHED` in `claims.json`)**:
  - Full original validation `L6` (`263` contexts, dropout off, `C26`): `1.0055212` at **epoch 2**, best `0.8968201` at **epoch 28**, `0.8974925` at **epoch 40** (step `17,416`; terminal step `17,773` re-evaluates to `0.8973272` per `C141, C181`).
  - Plateau (`C27`): validation loss stayed within `[0.8968201, 0.8978393]` across all `31` closed epochs from epoch `10` to epoch `40`.
  - Per-term full validation losses from epoch `2` to epoch `40` (`C28`): `CE 0.6376 -> 0.5477`, `q regression 0.1804 -> 0.1809`, `Bradley-Terry 0.4719 -> 0.4136`, `listwise 1.9985 -> 1.9736`, `runtime 0.1001 -> 0.0758`.
  - Learning rate halved `9` times, reaching the `1e-6` floor at step `5,677` and remaining at the floor for the last `12,096` updates (`C25`). Read-only terminal capture walked `52,329` entries (`3,272,631,982` B) with `0` writes (`C32`).

### `EXP-002-qhat-fresh-program-ranking` — Outcome Head (`q-hat`) Ranking (`D5b`) & Separate Outcome Predictor vs. LightGBM
- **Date & Provenance**: `2026-10-08` | `source_commit: "unknown"` | Claims: `C75, C76, C77, C78, C178, C179`.
- **Summary**: Evaluated the `v1` outcome head (`q-hat`) on freshly executed programs and `32 -> 8` reranking across `32` kept contexts of the historical `D5b` panel (`C75–C77`), and compared a separate earlier DL outcome-prediction ensemble against LightGBM on raw LV4 features across the `7,120`-view pool (`C78, C178`).
- **Data**: `32` kept contexts (from `53` roster contexts) on `D5b`; `7,120`-view pool (`1,579` datasets, `1,068` lineages, `3` repeats $\times$ `5` lineage-grouped folds) for `C178`.
- **Outcome (`PRELIMINARY` — values match `established` `C75–C78, C178`, but `D5b` panel predates the `R6` split seal)**:
  - `q`-head pairwise ranking accuracy on fresh programs (`C76`): `0.5056656` (`0.506 [0.441, 0.568]` over `32` lineages; chance is `0.5`).
  - `q`-head rerank (`top 8 of 32`) minus plain policy sampling (`C75`): `dScore = 0.03125 [-0.0625, 0.125]`, `dBestQ = 0.01873 [-0.03204, 0.08107]`, recorded decision `INCONCLUSIVE` (censoring sensitivity bounds over all `53` roster contexts `[-0.353, 0.392]`).
  - Separate DL outcome ensemble vs. LightGBM (`C78, C178`): equal-lineage $R^2$ `0.690235` vs. `0.784889` (`difference [-0.132907, -0.056273]`), row-weighted $R^2$ `0.651858` vs. `0.751845`, `beat10` AUC `0.900874` vs. `0.903092` (`difference -0.002217 [-0.010492, +0.006826]`). (`C179` notes a separate `7,087`-view `data_v1` LightGBM yardstick with $R^2$ `0.7176 / 0.7603 / 0.7450`.)

### `EXP-003-dev-panel-matches-t1-d5b` — Best-of-8 Historical Development-Panel Matches (`T1` / `D5b`)
- **Date & Provenance**: `2026-10-08` | `source_commit: "unknown"` | Claims: `C72, C73, C74, C75, C77`.
- **Summary**: Compared best-of-8 executed programs from `policy_sampling`, `q_rerank_of_32`, and `uniform_catalog` against the verified `teacher` across the `32` kept contexts of `D5b`.
- **Data**: `32` kept contexts (of `53` roster contexts) on the pre-`R6` `D5b` development match panel (`2,000` percentile bootstrap draws over parent lineages, seed `7`, `C77`).
- **Outcome (`PRELIMINARY` — historical, directional)**:
  - Tie-adjusted score vs. teacher (`C72`): `policy 0.109 [0.031, 0.203]`, `uniform 0.078 [0.016, 0.156]`, `rerank 0.141 [0.0625, 0.234]`.
  - Expected best $q$ of 8 (`C73`): `teacher 2.134 [1.186, 3.438]`, `policy 1.994 [1.050, 3.268]`, `rerank 2.013 [1.078, 3.284]`, `uniform 1.542 [0.900, 2.526]`.
  - Mean regret vs. teacher (`C74`): `policy 0.140`, `rerank 0.121`, `uniform 0.591`.

### `EXP-004-evaluator-quantile-parity-census` — Verification Parity Census & One-ULP Binary-Search Disagreement
- **Date & Provenance**: `2026-10-08` | `source_commit: "unknown"` | Claims: `C41, C42, C43, C44, C45, C47, C49, C149`.
- **Summary**: Audited prediction parity between the native evaluator (`crates/feat-math/src/gpr.rs`) and emitted code (`crates/feat-math/src/emit_rs.rs`, `emit_py.rs`, `emit_cpp.rs`, `emit_js.rs`) across `1,554` evaluated `(context, program)` pairs (`C41–C45, C47, C49`).
- **Data**: `6,640` catalog programs, of which **`1,636` (`24.64%`)** contain a quantile-binning operator (`C42`); `1,554` evaluated pairs (`342` = `22.0%` with binning, `1,212` without binning).
- **Outcome (`ESTABLISHED` in `claims.json`)**:
  - `41 / 1,554` (`2.6%`) evaluated pairs covering `40` distinct programs failed parity deterministically across `114` observations on two hosts (`C41, C44`).
  - **Conditional failure rates (`C42, C43`)**: `41 / 342 = 12.0%` (`95% CI [8.7%, 15.9%]`) with quantile binning; `0 / 1,212` without binning (`one-sided 95% upper bound <= 0.25%`). Clean evaluations are bit-exact on `253/253` screen rows and `2,557/2,558` arena fold checks (`C45`).
  - **Mechanism ([`TECHNICAL_WRITEUP.md` §5.5](../paper/TECHNICAL_WRITEUP.md#55-case-study-a-one-ulp-disagreement-between-two-binary-searches), `crates/feat-math/src/gpr.rs:68`, `crates/feat-math/src/emit_rs.rs:502`)**: Quantile edges are interpolated as `s_lo*(1-f) + s_hi*f` (`sorted[lo] * (1.0 - frac) + sorted[hi] * frac`). On a tie plateau (`s_lo == s_hi == a`), this two-product sum can land one ulp below or above `a`, making the edge vector non-monotone (`C47`: three of nine plateau edges one ulp off). The native code uses the standard-library partition search whose probe sequence changed in Rust 1.82.0, while all four emitters keep the classic midpoint loop (`C49`).

### `EXP-005-v2-r23-bitwise-build-qualification` — `v2` Learner CUDA Next-Update Parity (`r2`–`r3`) & `230` CPU Fixtures
- **Date & Provenance**: `2026-10-11` | `source_commit: "unknown"` | Claims: `C09, C118, C131, C160, C161, C162`.
- **Summary**: Verified update-only CUDA bitwise parity between frozen `v1` step primitives and the `v2` learner across `8` receipts from `r2` to `r3` on `v1`'s next batch (step `17,774`), alongside the `230`-identity offline CPU fixture suite (`run_v2_offline.py`).
- **Data**: One never-committed update from the `v1` terminal checkpoint on batch `9a70e4f5...` (epoch 40, index 357, step `17,774`, `C131`); `230` offline CPU fixture identities (`83` learner, `55` native-result overlay, `36` target preparation, `32` terminal capture, `24` adoption inbox, `C118`).
- **Outcome (`ESTABLISHED` in `claims.json`, update-only / build-only scope)**:
  - All `8` GPU receipts (`abec3214` `r2`, `13ada28b` `r2.1`, `c19fa1d2` `r2.2`, `c15aea30` `r2.3`, `7c3da3a5` `r2.4`, `d03aac03` `r2.6`, `364ebfa4` `r2.7`, and `10153efb` `r3` anchored step at KL `0`) are bitwise equal on `48` model, `144` AdamW, and `2` RNG tensors (`C09, C131`) with loss `0.9540845`.
  - `230/230` offline CPU fixtures pass with `0` skips (`C118`; `213/213` at `r3.1` across two torch versions, `C161`). Next-update parity on CPU was not run (`C162`), and live GPU step time/memory of the anchored step remains pending (`C160`).

### `EXP-006-r22-scratch-sweeps-3-to-6` — `R22` GPU Scratch Sweeps (`3, 4a, 4b, 5, 6`)
- **Date & Provenance**: `2026-10-13` | `source_commit: "unknown"` | Claims: `C136, C137, C138, C139, C163, C173, C176, C177`.
- **Summary**: Executed five non-qualifying `R22` GPU scratch sweeps (`3, 4a, 4b, 5, 6`) on RTX 3090 comparing `L6` alone, `L6 + 0.2*KL`, and `L_PI + 0.2*KL` ($\lambda_{\text{PI}} \in \{0.01, 0.03, 0.10\}$).
- **Data**: Sweeps `3` and `4a` had **no `885`-context holdout** and trained on those contexts; sweeps `4b`, `5`, and `6` held out the `885` training contexts (`167` parent lineages) and scored full validation on `263` contexts (`54` validation lineages) (`C138, C173, C177`, [`NUMBERS_TRACE.md`](../paper/NUMBERS_TRACE.md#findings-and-corrections) correction #5).
- **Outcome (`NON_QUALIFYING`)**:
  - **Probe KL drift (`64` train probes, `C136–C137`)**: At LR `3.75e-5`, `L6` alone reached `0.053` (sweep `4a`) and `0.052` (sweep `5`) within two epochs (stopped); at LR `1e-5`, `L6` alone reached `0.044` (sweep `3`) and `0.041` (sweep `4b`). Every pure `L6 + 0.2*KL` arm stayed at or below `0.014` (`0.004–0.011` in `4a`, `0.004–0.008` in `5`, `0.006–0.014` in `6`), whereas `L_PI + 0.2*KL` arms reached `0.004–0.036` (`L_PI 0.01`), `0.020–0.057` (`L_PI 0.03`), and `0.102` at epoch 1 (`L_PI 0.10` in sweep `3`, stopped).
  - **Held-out `885`-context losses in sweeps `4b–6` (`C138–C139`)**: Measured-tilt loss started at `7.080` and ended lower in `L_PI` arms (`6.591–6.865`) than comparators (`7.066–7.211`), while held-out `L6` (`0.8610` at start) did not improve at any arm's last evaluated epoch (`0.8621–0.8660`, though two sweep `4b` arms were marginally below `0.8610` at epoch 1).
  - **Paired verdicts (`C163, C176`)**: Sweep `6`'s two arms were `INCONCLUSIVE` against frozen `v1` on the `263` validation contexts (`n < 881`): `L6 + 0.2*KL` `-0.000978 [-0.00446, +0.00250]` and `L6 + 0.2*KL + 0.01*L_PI` `-0.000315 [-0.00365, +0.00302]`.

### `EXP-007-r25-r26-r27-and-1148-context-judge` — Post-Sweep Scratch Runs (`r25control, R25, R26, R27`) & `L6-1148` Judge
- **Date & Provenance**: `2026-10-13` | `source_commit: "unknown"` | Claims: `C165, C169, C170, C171, C172, C173, C174, C177, C182, C183`.
- **Summary**: Executed four further scratch training runs (`r25control` with `1,408` updates, `R25` with `352`, `R26` with `1,408`, `R27` with `1,408`) excluding the `885` held-out training contexts and recorded five judge evaluations on `2026-10-13`, including `L6-1148` (`C173, C174, C182`).
- **Data**: `L6-1148` combines `1,148` contexts across **`221` parent clusters** (`263` validation contexts in `54` clusters + `885` held-out-lineage training contexts in `167` clusters, `C173, C177`).
- **Outcome (`NON_QUALIFYING`)**:
  - `R25` (`+ 0.25*CE_kNN`) was frozen after epoch 1 (`352` updates) at mean probe KL `0.1079` (`C169`).
  - **`L6-1148` verdicts against frozen `v1` (`C174`, `K = 3` Holm, `2,000` cluster-bootstrap draws)**: `R25` epoch 1 `+0.00511 [+0.00413, +0.00611]`, `L6 + 0.2*KL` epoch 4 (`r25control`) `+0.00310 [+0.00195, +0.00434]`, and `R26` epoch 4 `+0.00333 [+0.00214, +0.00461]`. All three raised a non-qualifying **`REGRESSION ALERT`** (`p = 1/2000` each).
  - **Directional strata (`C177` — explicitly marked reported and not decisive)**: On the `263` validation contexts (`54` clusters), `R25` ep 1 was `+0.00119 [-0.00073, +0.00294]`, `L6 + 0.2*KL` ep 4 was `-0.00070 [-0.00214, +0.00060]`, and `R26` ep 4 was `-0.00108 [-0.00277, +0.00044]` (all intervals including zero); on the `885` held-out-lineage training contexts (`167` clusters) that `v1` trained on and the fine-tunes excluded, differences were `+0.00628 [+0.00524, +0.00737]`, `+0.00423 [+0.00286, +0.00563]`, and `+0.00463 [+0.00329, +0.00609]`.

### `EXP-008-gb06-h1-bounded-gp-expansion-pilot` — Closed `H1` Offline GP Search Pilot (`PR #306` / `GB-06`)
- **Date & Provenance**: `2026-10-13` | `source_commit: "bd716765c401e02210cb6add87e31e1d6efc5c35"` (resides on host-local custody ref `refs/clover-evidence/gb06/20261013-bd716`, not pushed to GitHub) | Claims: `C166, C167, C168` & [`gp-h1-readout.20261013.json`](../internal/clover/gp-h1-readout.20261013.json).
- **Summary**: Executed a bounded 3-arm offline GP search pilot (`ordinary`, `policy_seeded` from frozen `v1`, `retrieval_seeded` from previously verified training programs) across `24` planned training contexts.
- **Data**: `53` completed arm receipts and `18` interrupted arms (`~900 s` wall-clock limit) across `6` incomplete contexts (indices `8, 12, 15, 16, 17, 22`); context `00` excluded from the coordinator-only primary readout because `ordinary` was separately launched, leaving **`17` complete-case training contexts** (`C168`).
- **Outcome (`NON_QUALIFYING` — `NON-QUAL` selection-biased training inner-fold complete-case summary)**:
  - **Paired comparisons on `17` contexts (same training inner folds, lower loss is better, `C166`, `gp-h1-readout.20261013.json`)**:
    - `policy_seeded / ordinary`: `10` lower / `2` equal / `5` higher (`median_log_loss_ratio = -0.000349264`).
    - `retrieval_seeded / ordinary`: `16` lower / `1` equal / `0` higher (`median_log_loss_ratio = -0.035361252`).
    - `policy_seeded / retrieval_seeded`: `2` lower / `0` equal / `15` higher (`median_log_loss_ratio = +0.017235560`).
  - `ordinary`, `policy_seeded`, and `retrieval_seeded` beat the remeasured incumbent in `8`, `10`, and `11` of `17` contexts, with unequal total search attempts (`1,913`, `1,714`, and `2,082`; `actual_equal_attempt_counts: false`).
  - **Syntactic novelty & parity (`C167`)**: `40/51` selected champions are **syntactically** outside the `6,640` catalog (`behavioral_novelty_vs_full_catalog: "unknown; no full6640 finite-probe replay"`). `48/51` passed bitwise native/emitted Rust parity on both training probes; `3/51` failed parity (**context `14` `retrieval_seeded`** cut 1 max abs delta `0.0021133743`; **context `18` `ordinary`** cut 0 max abs delta `0.0321004556` and **`policy_seeded`** cut 0 max abs delta `0.1367213160`).

### `EXP-009-j9-frozen-pool-selector-and-flywheel-distill` — CLOVER Host Steps 3–5 Fresh-Clone Smoke Run
- **Date & Provenance**: `2026-10-16` | `source_commit: "9fdff904cb224e24f28c166427fbdacd50fb97f8"` (and `73f9327fc4d5466fbbc9d260c3c8ca2a67d4d1ca`) | Source: [`NUMBERS_TRACE.md#fresh-clone-run`](../paper/NUMBERS_TRACE.md#fresh-clone-run), [`docs/clover-host-steps.md`](../clover-host-steps.md).
- **Summary**: Verified in the `2026-10-16` fresh-clone audit that the seven offline CLOVER host commands (`ingest-coverage`, `freeze`, `ingest-roster-features`, `evaluate`, `distill-gp-index`, `flywheel-distill`, `inspect-parquet`) exit `0` when given host-local `R22` scratch inputs.
- **Data**: Host-local `R22` scratch inputs (`2,267` train contexts minus `885` `heldout.ids` $\to$ `1,382` emitted $\to$ `1,222` frozen pools from `21,199` rows; `8,908` `GP_RESULTS_INDEX.json` rows).
- **Outcome (`NON_QUALIFYING` — command execution verification only; per `NUMBERS_TRACE.md`, *"The outputs below show only that the commands run; they are not results"*)**:
  - `evaluate` printed `strongest_baseline: ridge_program`, `ranking_learnability: NOT_ESTABLISHED` (receipt `b9f6dd48...` identical in debug and release).
  - `flywheel-distill` printed `5` rounds, `flywheel_established: false`, `ranking_learnability: NOT_ESTABLISHED` (receipt `bed5ad79...` identical across runs).
  - **Timings (`NUMBERS_TRACE.md`)**: `evaluate` took `767 s` (debug at `9fdff904c`), `44 s` (release binary at `9fdff904c`), and `51 s` (`cargo run --release` at `73f9327fc`). `flywheel-distill` took `8,125 s` (debug at `9fdff904c`), `570 s` (release binary at `9fdff904c`), and `558 s` (`cargo run --release` at `73f9327fc`; `820 s` total for all 7 commands build-included).

### `EXP-010-core-ip-gpu-latency-study` — Core-IP `2026-09-25` Bounded 1-Hour GPU Latency Window
- **Date & Provenance**: `2026-09-25` | `source_commit: "802fb464c8a30eb7aeb05296313300e737a96044"` | Source: [`gpu-closeout.json`](../internal/core-ip/data/gpu-closeout.json), [`docs/internal/core-ip/sections/10.md`](../internal/core-ip/sections/10.md).
- **Summary**: Admitted a single `3,600 s` GPU latency window at `2026-09-25T19:11:34.939Z` that halted after `428 ms` before CUDA qualification began.
- **Data**: `10`-dataset benchmark panel (`0` direct requests, `0` full GP arms, `0` prediction batches executed).
- **Outcome (`FAILED_BEFORE_EXECUTION`)**:
  - **Receipt collision**: `ExecutionPool::admit` had already written `window.json`, and the timing adapter attempted to write a different schema to the same immutable path (`428 ms` service runtime, `294 ms` CPU, `131.9M` peak memory are controller failure diagnostics only).
  - **Staging protocol deviation (`sections/10.md`)**: The original planner called `load_development`, which parsed development labels including outer `cut0` before selecting fitting rows, and constructed an outer hold partition. Although no sealed target table was opened and no outer score was computed, parsing outer `cut0` labels violated the strict do-not-open constraint, so the original study **must not be described as fully protocol-compliant**. Repaired in contract `v3`.

### `EXP-011-same-cut-and-sotarena-elo-benchmarks` — Fresh-Clone Arena Ladders & Same-Cut Reproduction
- **Date & Provenance**: `2026-10-16` | `source_commit: "9fdff904cb224e24f28c166427fbdacd50fb97f8"` (and `827881ef7a5333779ddf4df5cc83d519274905e5`) | Source: [`NUMBERS_TRACE.md#fresh-clone-run`](../paper/NUMBERS_TRACE.md#fresh-clone-run).
- **Summary**: Verified the README's arena ladder, same-cut reproduction, and strict emitter parity shell blocks from fresh clones on `xbabe2`.
- **Data**: `2` committed arena Elo ladders, `10`-dataset same-cut benchmark (`datasets/test`), `40` synthetic datasets, and `feat-math` strict parity tests (`1,168` passed, `12` ignored, `0` failed).
- **Outcome (`ESTABLISHED` in `NUMBERS_TRACE.md#fresh-clone-run`)**:
  - Both `jope-arena elo --check` commands exited `0` with ladders byte-identical to the committed files.
  - All `3` same-cut reproduction commands exited `0` (`273 s` for the 10-dataset run), reproducing `assets/same-cut/v2/leaderboard-jope-same-cut-v1-deep.csv` byte-for-byte.
  - `JOPE_PARITY_STRICT=1 cargo test --locked -p feat-math` exited `0` (`1,168` passed, `12` ignored, `0` failed) in `1,854 s` (`827881ef7`) / `1,720 s` (`9fdff904c`).

### `EXP-012-spq1-selfplay-round-qualification-prereg` — `SPQ-1` Offline Diagnostic Verifier & Unpinned Round-1 Prereg
- **Date & Provenance**: `2026-10-16` | `source_commit: "c402b21183670d3e4df73b835292a385a04db30f"` (`source_commit_at_draft: "525be65b38540eb64b84c9bf36f7da6812812f2f"`) | Claim: `C158` & [`selfplay-round1-prereg.v1.json`](../internal/clover/selfplay-round1-prereg.v1.json).
- **Summary**: Merged the offline diagnostic verifier (`jope-pools qualify-rounds`, schema `jope.selfplay-round-qualification/v1`) and the unpinned Round-1 confirmatory pre-registration packet (`selfplay-round1-prereg.v1.json`, status `PRE_REGISTERED_UNPINNED_NO_EXECUTION_GRANT`) in PR #343.
- **Data & Contract ([`SELFPLAY_ROUND_QUALIFICATION.md`](../internal/clover/SELFPLAY_ROUND_QUALIFICATION.md), [`SELFPLAY_ROUND1_PREREG.md`](../internal/clover/SELFPLAY_ROUND1_PREREG.md))**:
  - `arms_and_rounds`: `rounds = 4`, `seed_cap_b = 8`, `equal_attempts_per_round_required = true`, `equal_resource_comparison_required = true`.
  - `endpoints`: Primary is `final_vs_round0_incumbent` (final-round self-play ratchet champion vs. Round-0 incumbent on once-only held-out test RMSE, `jope.pipeline-test-score/v1`). Secondary (`ratchet_vs_ordinary_final_round`) is `UNSUPPORTED_PENDING_PRODUCER_ORDINARY_CHAMPION_GENOME` and refused by `qualify-rounds`.
  - Paired statistic: per-context log-RMSE reduction `paired_log_reduction = ln(test_rmse_reference) - ln(test_rmse_candidate)` (`|ln| <= 1e-12` treated as `0.0` per F10 tie tolerance), resampled via two-sided parent-cluster percentile bootstrap (`SplitMix64`, `bootstrap_resamples = 2000`, `bootstrap_seed = 20261016`, code minimum `>= 2` parent clusters, conservative single-endpoint `alpha <= 0.025`), with `margin_cleared = paired_log_reduction.cluster_bootstrap.low > plan.practical_margin_log_ratio`.
- **Outcome (`DIAGNOSTIC_ONLY` — `PRE_REGISTERED_UNPINNED_NO_EXECUTION_GRANT`)**:
  - Validates caller-supplied `decomposed_support_kl` records (computed by `support::decomposed_support_kl` from PRs #333/#334) against `max_conditional_base_kl = 0.20` and `max_new_support_mass = 0.15`, probability mass conservation, chain-rule `total_kl` identity, derived admissibility, sequential same-registry test-split look ordering, and R5 parity failure charging (`11` dedicated tests: `9` integration in `selfplay_round_qual.rs` + `2` unit in `round_qual.rs`).
  - **Explicit non-claims**: Inputs are caller-asserted (`provenance_binding_status: "unverified_caller_asserted_inputs"`); `qualified_claim`, `promotion`, and `l6_claim` are **unconditionally `false`** in `RoundQualReport`; and `experiment_identity_sha256`, `sealed_roster_sha256`, `min_parent_clusters`, `practical_margin_log_ratio`, `pinned_test_score_registry`, and `owner_admission_locator` are all `UNPINNED` (`execution_grant.authorized: false`).

---

## Validation Before Committing (`EXP-013+`)

Run the real Draft 2020-12 JSON Schema validator (`jsonschema`), commit/claim/path/anchor verifier (including prose `Cxx` inclusion, `data.sources`, and Markdown heading anchors; note that `bd716765c401e02210cb6add87e31e1d6efc5c35` is a host-local custody ref `refs/clover-evidence/gb06/20261013-bd716` not pushed to GitHub), and repository hygiene suite before committing any change to `docs/experiments/`:

```sh
python3 - << 'PY'
import importlib.metadata, json, os, re, subprocess
from jsonschema import Draft202012Validator

print(f"Using jsonschema version: {importlib.metadata.version('jsonschema')}")
LOCAL_CUSTODY_ONLY_COMMITS = {"bd716765c401e02210cb6add87e31e1d6efc5c35"}

with open("docs/experiments/schema.v1.json") as f:
    schema = json.load(f)
Draft202012Validator.check_schema(schema)
validator = Draft202012Validator(schema)

with open("docs/paper/claims.json") as f:
    valid_claims = {c["id"] for c in json.load(f)["claims"]}

def slugify_heading(heading_text):
    s = heading_text.strip().lower()
    s = re.sub(r'[^\w\s-]', '', s)
    s = re.sub(r'[\s]+', '-', s)
    return s

def get_file_anchors(md_path):
    anchors = set()
    with open(md_path, encoding="utf-8") as f:
        for line in f:
            m = re.match(r'^#{1,6}\s+(.*)$', line)
            if m:
                anchors.add(slugify_heading(m.group(1)))
    return anchors

def expand_claim_ranges(text):
    cited = set(re.findall(r'\bC[0-9]{2,3}\b', text))
    for lo, hi in re.findall(r'\bC([0-9]{2,3})-C([0-9]{2,3})\b', text):
        width = len(lo)
        for n in range(int(lo), int(hi) + 1):
            cid = f"C{n:0{width}d}" if width == 2 and n < 10 else f"C{n}"
            if cid in valid_claims:
                cited.add(cid)
    return cited

with open("docs/experiments/experiments.jsonl") as f:
    rows = [json.loads(line) for line in f if line.strip()]

for row in rows:
    validator.validate(row)
    subprocess.check_call(["git", "cat-file", "-e", f"{row['catalog_snapshot_commit']}^{{commit}}"])
    sc = row["source_commit"]
    if sc != "unknown":
        rc = subprocess.call(["git", "cat-file", "-e", f"{sc}^{{commit}}"], stderr=subprocess.DEVNULL)
        if rc != 0:
            assert sc in LOCAL_CUSTODY_ONLY_COMMITS, f"Commit {sc} missing and not in LOCAL_CUSTODY_ONLY_COMMITS"
    declared_claims = set(row["claim_ids"])
    for cid in declared_claims:
        assert cid in valid_claims, f"Unknown claim_id {cid} in {row['id']}"
    prose_claims = expand_claim_ranges(json.dumps(row))
    missing_in_claim_ids = prose_claims - declared_claims
    assert not missing_in_claim_ids, f"{row['id']} cites {sorted(missing_in_claim_ids)} in prose but omits them from claim_ids"
    for path in row["artifacts"]:
        assert os.path.exists(path), f"Missing artifact path {path} in {row['id']}"
    for src in row["data"]["sources"]:
        file_part, _, anchor_part = src.partition("#")
        assert os.path.exists(file_part), f"Missing source file {file_part} in {row['id']}"
        if anchor_part:
            assert anchor_part in get_file_anchors(file_part), f"Missing anchor #{anchor_part} in {file_part}"

for md_file in ["docs/experiments/README.md", "docs/experiments/SYNTHESIS_AND_NEXT_EXPERIMENTS.md"]:
    base_dir = os.path.dirname(md_file)
    with open(md_file, encoding="utf-8") as f:
        content = f.read()
    for cid in expand_claim_ranges(content):
        assert cid in valid_claims, f"Unknown claim {cid} in {md_file}"
    for m in re.finditer(r'\]\(([^)]+)\)', content):
        target = m.group(1)
        if target.startswith(("http://", "https://")):
            continue
        file_part, _, anchor_part = target.partition("#")
        resolved = os.path.normpath(os.path.join(base_dir, file_part)) if file_part else md_file
        assert os.path.exists(resolved), f"Broken link {target} in {md_file}"
        if anchor_part:
            assert anchor_part in get_file_anchors(resolved), f"Broken anchor #{anchor_part} in {md_file}"

print(f"Validated {len(rows)} experiment records, prose Cxx citations, data.sources, paths, and markdown anchors.")
PY
git diff --check
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only bash deploy/ci/check.sh
bash deploy/ci/ip-scrub.sh
```
