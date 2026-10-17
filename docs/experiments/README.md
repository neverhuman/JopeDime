# JopeDime / CLOVER Curated Experiment Digest (`docs/experiments/`)

This directory provides a **curated, non-exhaustive 18-record digest (`EXP-001`–`EXP-018`) as of 2026-10-16** (compiled at `main` commit `873fb406a8ef4985565617bc812fe497bd4584f1`) in a structured JSONL + Markdown format for LLM and script consumption.

> **Authoritative Ledger & Scope Note:** [`docs/paper/claims.json`](../paper/claims.json) (`177` claims: `118` established, `20` preliminary, `16` non-qualifying, `23` pending, traced in [`docs/paper/NUMBERS_TRACE.md`](../paper/NUMBERS_TRACE.md)) remains the repository's authoritative claim ledger. Every numerical value in this folder is taken directly from a cited `Cxx` entry in `claims.json`, from a named tracked receipt (`gp-h1-readout.20261013.json`, `gpu-closeout.json`, `selfplay-round1-prereg.v1.json`, `NUMBERS_TRACE.md#fresh-clone-run`), or from a SHA-256-pinned host-local receipt under `/home/ubuntu/jope-evidence/` (`jopedime-flywheel-ei0`, `rlfw-g-abl`, `jopedime-flywheel-ei1`, `jopedime-flywheel-ei2`, `ei3-study-20261015`, `ei4-study-20261015`, `selfplay-s3-sos-debiased-flywheel-20261016`).

---

## Directory Layout

| File | Format | Purpose |
| :--- | :--- | :--- |
| [`README.md`](README.md) | Markdown | Scope and known omissions, status taxonomy, master matrix (`EXP-001`..`EXP-018`), compressed experiment cards, and validation commands. |
| [`experiments.jsonl`](experiments.jsonl) | JSON Lines (`1` record/line) | Machine- and LLM-consumable digest of the 18 curated records conforming to [`schema.v1.json`](schema.v1.json). |
| [`schema.v1.json`](schema.v1.json) | JSON Schema (Draft 2020-12) | Strict schema (`jope.experiment-record/v1`) enforcing 40-hex or `"unknown"` `source_commit`, `catalog_snapshot_commit`, `claim_ids`, and typed `metrics`. |
| [`SYNTHESIS_AND_NEXT_EXPERIMENTS.md`](SYNTHESIS_AND_NEXT_EXPERIMENTS.md) | Markdown | Five scope-bounded working hypotheses (non-qualifying) and a prioritized roadmap (`EXP-NEXT-01` done, `EXP-NEXT-04` opt-in recipe/bound-Python `partition_point` emitter mode + synthetic 4-language/bound-model regression landed while default CLI/pipeline/campaign emissions retain the legacy search and `R5` parity charging and original `EXP-008` context replay remains outstanding, and `EXP-NEXT-02`..`EXP-NEXT-03` proposals with explicit admission gates). |

---

## Known Omissions (`Not Yet Indexed as Standalone Rows in experiments.jsonl`)

Because this folder is an 18-entry curated digest rather than an exhaustive dump of every historical run, the following sub-runs and historical judge panels remain summarized within existing rows or in [`docs/paper/claims.json`](../paper/claims.json):

1. **Individual rows for `C182`'s four post-sweep scratch runs and five judge records:** Summarized jointly under `EXP-007` (`r25control`, `R25`, `R26`, `R27`; judge records `sweep6-verdict-8b7aaead.json`, `r23e4-v1-verdict-de48c164.json`, `r25-e1-verdict-de66cb6a.json`, `j21-paired-verdict-98fa563d.json`, and `l6-1148-verdict.json`). Similarly, individual seed- and host-level sub-run receipts for `EI-0` (`24` runs / `96` checkpoint slots), `EI-2`/`EI-2X` (`3` hosts), and `EI-3` (`3` hosts) are consolidated into `EXP-013`, `EXP-015`, and `EXP-016`.
2. **Sweep 3 own-control paired verdicts at epochs 1–3:** Recorded in [`NUMBERS_TRACE.md`](../paper/NUMBERS_TRACE.md#findings-and-corrections) correction #8 (`C163` notes no paired verdict against frozen `v1`, whereas Sweep 3's two `L_PI` arms were judged against their own `L6` control at epochs 1–3).
3. **`J21` (`C170`) and `J22` judge panels and Wave 1–4 encoder studies (`C178`, `C179`):** Summarized only where referenced by `EXP-002` and `EXP-007`.

---

## Qualification Status Taxonomy

Every record carries an explicit `qualification_status` matching [`schema.v1.json`](schema.v1.json):

- **`ESTABLISHED`**: Matches an `established` entry `Cxx` in [`docs/paper/claims.json`](../paper/claims.json) or a verified fresh-clone reproduction row in [`docs/paper/NUMBERS_TRACE.md`](../paper/NUMBERS_TRACE.md#fresh-clone-run) as of 2026-10-16 (means the value was verified at the cited locator within its stated scope; does not imply live closed-loop generalization).
- **`PRELIMINARY`**: Directional or historical evaluation whose split or roster predates a full contamination/split seal or has censored contexts (e.g., pre-`R6` `D5b` 32-context panel in `EXP-002` and `EXP-003`).
- **`NON_QUALIFYING`**: Executed scratch run, complete-case inner-fold pilot, host-local `EI-0..EI-4` continuation study, offline shadow-pool flywheel evaluation, or command smoke run that is explicitly non-qualifying (`R22` scratch sweeps, `L6-1148` regression alerts, `GB-06` `NON-QUAL` readout, `EI-0..EI-4` paired continuation probes, `PR #339` offline shadow-pool flywheel, or `R22` scratch host-step smoke outputs).
- **`DIAGNOSTIC_ONLY`**: Merged offline diagnostic verifier or unpinned pre-registration packet whose outputs unconditionally set `qualified_claim: false` (`EXP-012`).
- **`PENDING`**: Pre-registered or planned evaluation that has not yet been executed (`C158`, `C160`).
- **`FAILED_BEFORE_EXECUTION`**: Admitted run that halted at a preflight/controller guard before collecting task measurements (`EXP-010`).

---

## Master Experiment Matrix (`EXP-001` – `EXP-018`)

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
| **`EXP-013`** | `2026-10-14` | `unknown` | `C163-C164, C173, C176-C177`, `ei0-fate-96-20261014.json` | **`EI-0` Paired `L_PI` Dose & KL-Weight Study**: `24` runs / `96` checkpoint slots (`0` unresolved); `L6-1148` (`221` clusters) | `P_EI0` ($\lambda_{\text{PI}}=0.01, \lambda_{\text{KL}}=3.0$, seeds `26–27` E4): **`INCONCLUSIVE`** (Stratum H `-0.000096`, Stratum A `+0.000343`, pooled `+0.000004`); `P_EI0r` (seeds `28–29` E4): Stratum H `+0.000596 [+0.000115, +0.001087]` (`p=0.0105`), pooled `+0.000269`; $\lambda_{\text{PI}}=0.03, 0.10$ pooled `-0.000416`, `-0.000377` | `NON_QUALIFYING` |
| **`EXP-014`** | `2026-10-14` | `unknown` | `C173-C175, C177`, `EI1_RUN_RECEIPT.json` | **`EI-1` Archive-CE Continuation (`v8`, `R69` vs `R70`)**: `42` archive ctx (`52` programs), $\lambda_{\text{KL}}=3.0, \lambda_{\text{ar}}=0.10, m=4$ | `ei1_arm` (`R69`) hit KL alarm at ep 2 (`0.038850`) & froze at ep 3 (`1,056` updates, `kl_mean = 0.079634 > 0.05`, `kl_stop`); **`PRIMARY_NOT_COMPUTED`** at ep 4; descriptive ep-3 `L6-1148` delta `+0.003854` | `NON_QUALIFYING` |
| **`EXP-015`** | `2026-10-15` | `unknown` | `C173, C175, C177`, `EI2_EI2X_3HOST_SUMMARY.json` | **`EI-2` / `EI-2X` 3-Host Paired Study (`xbabe1..3`)**: $\lambda_{\text{KL}}=3.0, \lambda_{\text{ar}}=0.03, m=4$, `4` complete epochs (`1,408` updates/arm) | Zero KL stops; `L6-1148` pooled delta: `xbabe1` `+0.000754 [+0.000312, +0.001190]`, `xbabe2` `+0.000375 [-0.000114, +0.000866]`, `xbabe3` `+0.000006 [-0.000418, +0.000420]`; **3-seed panel mean `+0.000378 [+0.000021, +0.000744]`** | `NON_QUALIFYING` |
| **`EXP-016`** | `2026-10-15` | `unknown` | `C173, C175, C177`, `EI3_3HOST_SUMMARY.json` | **`EI-3` 3-Host Joint Tilt + Archive-CE Study (`xbabe1..3`)**: $\lambda_{\text{KL}}=3.0, \lambda_{\pi}=0.05, \lambda_{\text{ar}}=0.10, m=8$ | All 3 hosts' `ei3_arm` froze at ep 3 (`1,056` updates, `xbabe2` KL `0.080342`); equal-exposure `paired_k_star_e3` pooled deltas: `xbabe1` `+0.003861`, `xbabe2` `+0.004222`, `xbabe3` `+0.003803`; **3-seed panel mean `+0.003962 [+0.003222, +0.004744]`** | `NON_QUALIFYING` |
| **`EXP-017`** | `2026-10-15` | `unknown` | `C173, C175, C177, C189`, `ei4-study-20261015/.../receipt.json` | **`EI-4` Stronger-KL (`5.0`) Archive-CE Paired Run (`xbabe2`)**: $\lambda_{\text{KL}}=5.0, \lambda_{\text{ar}}=0.025, \lambda_{\pi}=0.0, m=8$, seed `20264102` | Both arms completed all `4` epochs (`1,408` updates, `0` KL alarms; ep 4 probe KL `0.008758` vs `0.010309`); in-run `receipt.json` L6 delta **`+0.000378` H / `+0.001870` A** (weighted `+0.000720`) at ep 3 and **`+0.000051` H / `+0.000187` A** (weighted `+0.000082`) at ep 4 (standalone >512 KiB row-file cost deltas `+0.000518` ep 3, `+0.000106` ep 4) | `NON_QUALIFYING` |
| **`EXP-018`** | `2026-10-16` | `d222807a9f...` | `C190`, `PR #339`, `gp-results-index-s3-receipt.json`, `r3` extracts | **S3 `8,908`-Row SoS-Debiased Surrogate Flywheel**: `shadow-pool-v2` (`1,136` ctx, `255` clusters) & `v1` (`1,222` ctx, `258` clusters) | On `v2` (`all_rounds_kl_admissible: true`): `R0->R4` `B=1` gain **`+0.134751 [+0.067497, +0.215110]` (`GREATER`)**, pairwise **`+0.046025 [+0.039928, +0.051692]` (`GREATER`)**, active discovery **`+0.035792` (`GREATER`)**, Active vs. Passive `B=1` **`+0.034743` (`GREATER`)**; `R4` **`ESTABLISHED_SHADOW`** vs `program_prior` & `ridge_program`; `flywheel_established: false` | `NON_QUALIFYING` |

---

## Compressed Experiment Cards (`EXP-001` – `EXP-018`)

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

### `EXP-013-ei0-lpi-dose-paired-probes` — `EI-0` Paired `L_PI` Dose & KL-Weight Continuation Probes (`24` Runs / `96` Checkpoint Slots)
- **Date & Provenance**: `2026-10-14` | `source_commit: "unknown"` | Claims: `C163, C164, C173, C176, C177` | Host receipts: `/home/ubuntu/jope-evidence/rlfw-g-abl/ei0-fate-96-20261014.json` (SHA-256 `e3a8e26183ddff61194126fb64964f73eb6b71f3db718d370108bb68e4d7e1e7`), `/home/ubuntu/jope-evidence/jopedime-flywheel-ei0/l6-1148-pei0.json` (SHA-256 `c831da441fa5aa79ddcfbaec4107b42897f9f7b4f78d608ff2b3f7fc19c3496c`), `/home/ubuntu/jope-evidence/jopedime-flywheel-ei0/l6-1148-pei0r.json` (SHA-256 `d3e44ebbc393179619b0b123baac7c82fb6a564db3c8826b9bd4e131274df39e`), `/home/ubuntu/jope-evidence/jopedime-flywheel-ei0/l6-1148-ei0-dose-descriptive.json`.
- **Summary**: Executed `24` paired continuation runs (`R28`–`R53`, `96` planned checkpoint slots across `4` epochs each) from `v1` step `17,773` testing `L_PI` dose ($\lambda_{\text{PI}} \in \{0.01, 0.03, 0.10\}$) paired against identical-seed `L6 + \lambda_{\text{KL}}\text{KL}` controls ($\lambda_{\text{KL}} = 3.0$ across seeds `26–29` and $\lambda_{\text{KL}} \in \{1.0, 3.0, 5.0\}$ across follow-up seeds `30–31`).
- **Data**: `1,148` `L6-1148` contexts (`221` parent clusters: `54` validation clusters / `263` contexts in Stratum A; `167` held-out training clusters / `885` contexts in Stratum H). All `96` planned checkpoint slots accounted (`64` in seeds `26–29`, `32` in seeds `30–31`, `0` unresolved, `0` imputed; `10` partial attempts `R40–R49` preserved unused).
- **Outcome (`NON_QUALIFYING` — paired `L6-1148` differences $\Delta = L_6(\text{arm}) - L_6(\text{ctrl})$, lower is better)**:
  - **Primary `P_EI0` ($\lambda_{\text{PI}}=0.01, \lambda_{\text{KL}}=3.0$, seeds `26–27` at Epoch 4, `K=1`)**: **`INCONCLUSIVE`** — Stratum H (`885` ctx, `167` clusters) `-0.000096 [-0.000501, +0.000293]` (`p=0.692`); Stratum A (`263` ctx, `54` clusters) `+0.000343 [-0.000475, +0.001368]`; pooled `1,148` `+0.000004 [-0.000362, +0.000389]` (`p=0.985`).
  - **Replication `P_EI0r` ($\lambda_{\text{PI}}=0.01, \lambda_{\text{KL}}=3.0$, seeds `28–29` at Epoch 4, `K=1`)**: Stratum H **`+0.000596 [+0.000115, +0.001087]`** (`p=0.0105`, `INTERVAL_LOW_ABOVE_0` degradation vs. control); Stratum A `-0.000832 [-0.002155, +0.000248]`; pooled `+0.000269 [-0.000191, +0.000720]`.
  - **Descriptive higher doses (seeds `28–29` Epoch 4)**: $\lambda_{\text{PI}}=0.03$ had Stratum H `-0.000353 [-0.000732, -0.000022]`, Stratum A `-0.000629 [-0.001577, +0.000201]`, pooled `-0.000416 [-0.000780, -0.000082]`; $\lambda_{\text{PI}}=0.10$ had Stratum H `-0.000400 [-0.000792, -0.000016]`, Stratum A `-0.000300 [-0.001151, +0.000462]`, pooled `-0.000377 [-0.000723, -0.000024]`.

### `EXP-014-ei1-v8-archive-ce-kl-stop` — `EI-1` Archive-CE Continuation (`v8`, `R69` vs. `R70`) & Epoch-3 `kl_stop`
- **Date & Provenance**: `2026-10-14` | `source_commit: "unknown"` | Claims: `C173, C174, C175, C177` | Host receipts: `/home/ubuntu/jope-evidence/jopedime-flywheel-ei1/EI1_RUN_RECEIPT.json` (SHA-256 `fd836804b698ad1efbd45d2a7c87382820477145ed4f64df3b1457d26f0164a4`), `/home/ubuntu/jope-evidence/jopedime-flywheel-ei1/claude-ei1-v8-armctrl-20261014T191054Z/receipt.json` (SHA-256 `1d163d66b66f89f7644f8c73f0f8a1ca72be199ed86b8681ddab1151d6bc5c4e`), `/home/ubuntu/jope-evidence/jopedime-flywheel-ei1/l6-1148-ei1-klstop-endpoint.json` (SHA-256 `c3ca4ac612be365fb9f0864207c36a760f8f7be7d42989ab2ba045a93f94a53d`).
- **Summary**: Executed the `EI-1` (`v8`) paired continuation experiment (`claude-ei1-v8-armctrl-20261014T191054Z`, seed `20261026`, LR `1e-5`, $\lambda_{\text{KL}}=3.0, \lambda_{\text{PI}}=0.0$) comparing `ei1_arm` (`R69`, $\lambda_{\text{ar}}=0.10, m=4$ archive-CE oversampling over `42` archive contexts / `52` archive programs) against `ei1_ctrl` (`R70`, $\lambda_{\text{ar}}=0.0, m=0$). (`C175` in `claims.json` was scope-dated `2026-10-13` before this run.)
- **Data**: `1,382` training contexts (`42` with archive targets, `52` archive programs; `0` validation or `885`-holdout leakage); evaluated on `L6-1148` (`1,148` contexts, `221` parent clusters).
- **Outcome (`NON_QUALIFYING` — `PRIMARY_NOT_COMPUTED` due to epoch-3 KL stop)**:
  - `ei1_arm` (`R69`) triggered a KL alarm at epoch 2 (`kl_mean = 0.038850 > 0.025`, halving LR to `5e-6`) and **froze at epoch 3** (`1,056` updates, `kl_mean = 0.079634 > 0.05`, `frozen_reason = "kl_stop"`), while `ei1_ctrl` (`R70`) completed all `4` epochs (`1,408` updates, `0` alarms, epoch 3 KL `0.005259`, epoch 4 KL `0.005638`).
  - **G-JUDGE disposition (`l6-1148-ei1-klstop-endpoint.json`)**: Pre-registered epoch-4 primary comparison is **`PRIMARY_NOT_COMPUTED`** (`primary_pass = false`). Descriptive equal-step (`1,056` updates, epoch 3) `L6-1148` difference $\Delta = L_6(\text{R69 ep3}) - L_6(\text{R70 ep3})$ was **`+0.003854`** (`0.8240560` vs. `0.8202022`; Stratum A `+0.004825`, Stratum H `+0.003565`).

### `EXP-015-ei2-ei2x-three-host-paired-study` — `EI-2` / `EI-2X` 3-Host Paired Reduced-Weight Archive-CE Study (`xbabe1..3`)
- **Date & Provenance**: `2026-10-15` | `source_commit: "unknown"` | Claims: `C173, C175, C177` | Host receipts: `/home/ubuntu/jope-evidence/ei3-study-20261015/EI2_EI2X_3HOST_SUMMARY.json` (SHA-256 `8ecea57ad8a7e0a80dd9fa988f131582680bd4c521cf20f0dd6f3e3b46461556`), `/home/ubuntu/jope-evidence/jopedime-flywheel-ei2/EI2_RUN_RECEIPT.json` (SHA-256 `82d7d25b3782c619f5b7265c12ba37d75b900bb49697b5f35854b9064e0b1a60`).
- **Summary**: Executed a 3-host paired continuation study across `xbabe1` (`seed 20262101`), `xbabe2` (`seed 20261026`), and `xbabe3` (`seed 20262102`) with reduced archive-CE weight $\lambda_{\text{ar}}=0.03$ ($\lambda_{\text{KL}}=3.0, m=4$, LR `1e-5`, `4` epochs = `1,408` updates/arm).
- **Data**: `1,148` `L6-1148` contexts (`221` parent clusters: `54` Stratum A, `167` Stratum H) across `3` independent host seeds (`2,000` cluster-bootstrap draws, seed `1729`).
- **Outcome (`NON_QUALIFYING` — all `3` hosts completed `4` epochs with `0` KL stops, but pooled `L6-1148` degraded slightly vs. control)**:
  - `xbabe1` (`seed 20262101`, ep 4): pooled **`+0.000754 [+0.000312, +0.001190]`** (Stratum A `+0.001053 [+0.000100, +0.002011]`, Stratum H `+0.000665 [+0.000181, +0.001151]`).
  - `xbabe2` (`seed 20261026`, ep 4, arm KL `0.011068` vs. ctrl `0.005638`): pooled **`+0.000375 [-0.000114, +0.000866]`** (Stratum A `-0.000124 [-0.001178, +0.000908]`, Stratum H `+0.000523 [-0.000016, +0.001081]`).
  - `xbabe3` (`seed 20262102`, ep 4): pooled **`+0.000006 [-0.000418, +0.000420]`** (Stratum A `-0.000087 [-0.000977, +0.000772]`, Stratum H `+0.000034 [-0.000429, +0.000496]`).
  - **3-seed panel mean (ep 4)**: pooled **`+0.000378 [+0.000021, +0.000744]`** (Stratum A `+0.000281 [-0.000484, +0.001057]`, Stratum H `+0.000407 [-0.000001, +0.000814]`).

### `EXP-016-ei3-three-host-paired-study` — `EI-3` 3-Host Paired Joint Tilt + Archive-CE Study (`xbabe1..3`)
- **Date & Provenance**: `2026-10-15` | `source_commit: "unknown"` | Claims: `C173, C175, C177` | Host receipts: `/home/ubuntu/jope-evidence/ei3-study-20261015/EI3_3HOST_SUMMARY.json` (SHA-256 `26deddc2e92ab866d03d7abcf44e19bc71cb25b07c74171d8a391b7c9543d8fb`), `/home/ubuntu/jope-evidence/ei3-study-20261015/xbabe2/ei3-xbabe2-armctrl-20261015/receipt.json` (SHA-256 `3af8bdaa61a5bd20079a5330a3204be2594127e7060fc5c4df6b21a5b0c73d68`).
- **Summary**: Executed a 3-host paired study (`xbabe1` seed `20263101`, `xbabe2` seed `20263102`, `xbabe3` seed `20263103`) testing joint measured-tilt ($\lambda_{\pi}=0.05$) and archive-CE ($\lambda_{\text{ar}}=0.10, m=8$) with $\lambda_{\text{KL}}=3.0$ against matched controls ($\lambda_{\pi}=0.0, \lambda_{\text{ar}}=0.0, m=0$).
- **Data**: `1,148` `L6-1148` contexts (`221` parent clusters) across `3` hosts (`2,000` cluster-bootstrap draws, seed `1729`).
- **Outcome (`NON_QUALIFYING` — all `3` hosts' `ei3_arm` froze at epoch 3 under `kl_stop`, with large positive `L6-1148` degradation)**:
  - On every host, `ei3_arm` triggered a KL alarm at epoch 2 (`xbabe2` `kl_mean = 0.039816`) and froze at epoch 3 (`1,056` updates, `xbabe2` `kl_mean = 0.080342 > 0.05`, `frozen_reason = "kl_stop"`), while `ei3_ctrl` completed all `4` epochs (`1,408` updates, `xbabe2` ep 3 KL `0.005471`, ep 4 KL `0.005688`).
  - **Equal-exposure epoch-3 (`paired_k_star_e3`, `1,056` updates) `L6-1148` pooled deltas**: `xbabe1` **`+0.003861 [+0.003121, +0.004620]`**, `xbabe2` **`+0.004222 [+0.003424, +0.005044]`**, `xbabe3` **`+0.003803 [+0.003048, +0.004584]`**; **3-seed panel mean `+0.003962 [+0.003222, +0.004744]`** (Stratum A `+0.003940 [+0.002373, +0.005494]`, Stratum H `+0.003969 [+0.003107, +0.004877]`).

### `EXP-017-ei4-kl5-archive-ce-paired-study` — `EI-4` Stronger-KL ($\lambda_{\text{KL}}=5.0$) Archive-CE Paired Study (`xbabe2`)
- **Date & Provenance**: `2026-10-15` | `source_commit: "unknown"` | Claims: `C173, C175, C177, C189` | Host receipts: `/home/ubuntu/jope-evidence/ei4-study-20261015/xbabe2-run/ei4-xbabe2-armctrl-20261015/receipt.json` (outer file SHA-256 `26fe86ee9d38be57e2399a2dab8cfc03ac1e10d17e1849c5ab0cf27765b60cd2`) and per-epoch row receipts `xbabe2-run/rows/*.receipt.json` (`ei4_arm-e4-b40f66a1.receipt.json` SHA-256 `bc4bff46051a7d0a9f0d105e3ffa62fcd8822299fdf1ea9f38b9530cb8e575fe`, `ei4_ctrl-e4-27c6190a.receipt.json` SHA-256 `4e4c62a1c0fa5cb92d092b0345c4ff3f3a7152a6fba461dcddf739c6493d0b20`).
- **Summary**: Tested stronger KL anchoring (`config.body.arms` positional tuples give each arm KL `5.0`) with reduced archive-CE weight ($\lambda_{\text{ar}}=0.025, \lambda_{\pi}=0.0, m=8$) on `xbabe2` (`config.body.seed = 20264102`, LR `1e-5`, `4` epochs = `1,408` updates/arm; `config.body` retains inherited EI-3 template fields `label`, `training_seed = 20263102`, and top-level `lambda_kl = 3.0`) against an identical-seed $\lambda_{\text{KL}}=5.0$ control (`ei4_ctrl`).
- **Data**: `1,382` training contexts (`42` with archive targets, `52` archive programs); evaluated on `L6-1148` (`1,148` contexts, `221` parent clusters; `19,332` in-run paired evaluation rows).
- **Outcome (`NON_QUALIFYING` — `0` KL alarms across all `4` epochs, with positive `L6-1148` deltas)**:
  - Both `ei4_arm` and `ei4_ctrl` completed all `4` epochs (`1,408` updates) with `0` KL alarms (`receipt.json#.epochs[0..3].eval[].probe.kl_mean`: `ei4_arm` ep 1 `0.014909`, ep 2 `0.006857`, ep 3 `0.008521`, ep 4 `0.008758`; `ei4_ctrl` ep 1 `0.006007`, ep 2 `0.005840`, ep 3 `0.005937`, ep 4 `0.010309`).
  - **In-run `receipt.json` L6 evaluation losses (`ei4_arm` minus `ei4_ctrl`, rounded to 6 decimal places)**: **`+0.000378` on Stratum H `heldout_l6`** (`0.862749` vs. `0.862371` over `885` contexts) and **`+0.001870` on Stratum A `a_l6`** (`0.897056` vs. `0.895186` over `263` contexts; `1,148`-context weighted mean **`+0.000720`**, `0.870609` vs. `0.869889`) at epoch 3, and **`+0.000051` on Stratum H `heldout_l6`** (`0.861867` vs. `0.861816`) and **`+0.000187` on Stratum A `a_l6`** (`0.896747` vs. `0.896560`; `1,148`-context weighted mean **`+0.000082`**, `0.869858` vs. `0.869776`) at epoch 4. (Standalone CPU row-file cost deltas in the `~637 KiB` `.jsonl` files outside the `512 KiB` source-inspection cap were `+0.000518` at epoch 3, `0.8205880` vs. `0.8200703`, Stratum A `+0.000992`, Stratum H `+0.000378`; and `+0.000106` at epoch 4, `0.8200851` vs. `0.8199788`, Stratum A `+0.000291`, Stratum H `+0.000051`.)

### `EXP-018-s3-sos-debiased-surrogate-flywheel` — S3 `8,908`-Row SoS-Debiased Surrogate Flywheel & Selector Evaluation (`PR #339`)
- **Date & Provenance**: `2026-10-16` | `source_commit: "d222807a9f9a6461ba16c5def6b5cb737839f220"` (`PR #339` on `main`; pre-rebase receipt commit `0b9bc150a99ee1921cdce6908cc60c4346d3bac4`) | Claim: `C190` | Host receipts & bounded `< 21 KiB` review extracts: `/home/ubuntu/jope-evidence/selfplay-s3-sos-debiased-flywheel-20261016/{gp-results-index-s3-receipt.json,pr339-body.md,inner-cv-tuning-log.md}`, `/home/ubuntu/jope-evidence/review-pr339-0b9bc150a99ee1921cdce6908cc60c4346d3bac4-r3/{scratch-r3-extract.txt,scratch-r3-receipts-bins.txt,VERDICT-source.md,scripts/extract.sh}`, and `/home/ubuntu/jope-evidence/review-pr339-0b9bc150a99ee1921cdce6908cc60c4346d3bac4-r2/scratch-extract-rounds.txt`.
- **Summary**: Distilled all `8,908` scored S3 `GP_RESULTS_INDEX.json` rows (from `18,380` total index runs; `9,383` unscored and `89` incomplete-provenance skipped; `13` program hashes across `1,579` contexts) into `gp-results-index-s3.parquet` (`3,093,919` B, SHA-256 `005b90dc1f2bde401e454966db8dd43f2007b0d2fd6c35b23cad5e1547ce6539`, verified in `gp-results-index-s3-receipt.json` outer SHA-256 `f9415faf7072064b3cb2f9e94f7194d18c19a844fc723626abed99c205bed1f6`, inner `.receipt_sha256` `367a859463f3076363499bf26c264c9b187301c13a43b2913282f5fa669751ab`), and evaluated `ContextualSurrogateQHat` (`crates/jope-learning/src/pools/selectors.rs:581-729`) with unclamped leave-one-parent-cluster-out (`parent_id != view.parent_id`) Strength-of-Schedule (`SoS`) target shift (`bt + ctx_offset`) and 5-fold parent-grouped inner-CV ridge regularization over `93` features (`13` summary features + `8` linear context + `8` linear program + `64` bilinear cross-terms).
- **Data**: Evaluated across `5` distillation rounds (`R0` to `R4`, `9,088` to `45,064` cumulative distilled rows) on frozen shadow candidate pools `shadow-pool-v2` (`1,136` teacher-excluded contexts, `255` parent clusters, `773` distilled join contexts = `68.05%`) and `shadow-pool-v1` (`1,222` teacher-included contexts, `258` parent clusters, `825` distilled join contexts = `67.51%`). All intervals are two-sided 95% parent-cluster percentile bootstrap `cluster_bootstrap.{low,high}`.
- **Outcome (`NON_QUALIFYING` — offline shadow-pool surrogate evaluation traced via bounded `< 21 KiB` `r3` review extracts; `flywheel_established = false` on both pools)**:
  - **On `shadow-pool-v2` (`flywheel-v2-receipt.json`, inner `.receipt_sha256` `2108664030edb26226b29c4fb7a348839e8262e9cfedc23c52fa218c82c0f0e7` verified in `scratch-r3-extract.txt`)**:
    - `all_rounds_kl_admissible = true` (`0` frozen contexts; `R4` max conditional base KL `0.008568 <= 0.20`, max new support mass `0.045455 <= 0.15`).
    - **`R0 -> R4` `B=1` top-1 gain lift (`.receipt.round_lifts[0].realized_gain_lift`)**: **`+0.134751 [+0.067497, +0.215110]` (`GREATER`)** (`0.785168 -> 0.919919`; `B=4` `+0.048424 [+0.020387, +0.076841]` `GREATER`; `B=8` `+0.022078 [+0.011148, +0.035404]` `GREATER`).
    - **`R0 -> R4` pairwise accuracy lift (`.receipt.round_lifts[0].pairwise_accuracy_lift`)**: **`+0.046025 [+0.039928, +0.051692]` (`GREATER`)** (`0.634318 -> 0.680342`).
    - **Active vs. passive discovery HWM lift (`.receipt.active_vs_passive_discovery_lift`)**: **`+0.035792 [+0.018341, +0.057848]` (`GREATER`)**.
    - **`R4` Active vs. Passive retraining (`.receipt.active_vs_passive_lifts`)**: `B=1` gain **`+0.034743 [+0.000512, +0.075291]` (`GREATER`)** (`0.919919` vs. `0.885176`), while `B=4` `-0.001835 [-0.018085, +0.009953]`, `B=8` `-0.001355 [-0.003403, +0.000407]`, and pairwise accuracy `+0.001880 [-0.000696, +0.004405]` remained `INCONCLUSIVE`, keeping `flywheel_established = false`.
    - **`R4` vs. baselines (`.receipt.final_round_receipt.comparisons`, `ranking_learnability = ESTABLISHED_SHADOW` vs. strongest baseline `program_prior`)**: vs. `program_prior` (`B=1` gain `+0.043596 [+0.010236, +0.082032]` `GREATER`, pairwise `+0.007620 [+0.003907, +0.011430]` `GREATER`) and vs. `ridge_program` (`B=1` gain `+0.066919 [+0.030060, +0.105549]` `GREATER`, pairwise `+0.029398 [+0.023834, +0.035189]` `GREATER`).
  - **On `shadow-pool-v1` (`flywheel-v1-receipt.json`, inner `.receipt_sha256` `bdd5605dec41c25610b9730fc23e89ebc72cd8b08e06178a9f056393d6aa1a39` verified in `scratch-r3-extract.txt`)**: `all_rounds_kl_admissible = true`, `R0 -> R4` `B=1` gain **`+0.118753 [+0.068081, +0.175087]` (`GREATER`)** (`1.344010 -> 1.462763`), `R0 -> R4` pairwise **`+0.033363 [+0.028599, +0.038003]` (`GREATER`)** (`0.688796 -> 0.722159`), active vs. passive discovery lift **`+0.138373 [+0.084957, +0.209026]` (`GREATER`)**, `R4` Active vs. Passive `B=1` gain **`+0.016280 [+0.000282, +0.036196]` (`GREATER`)** and pairwise `+0.001164 [-0.000891, +0.003141]` (`INCONCLUSIVE`, keeping `flywheel_established = false`), and `R4` vs. strongest baseline `ridge_program` **`ESTABLISHED_SHADOW`** (`B=1` gain `+0.034577 [+0.000923, +0.080958]` `GREATER`, pairwise `+0.045204 [+0.038397, +0.051677]` `GREATER`).
  - **Standalone full-pool `evaluate` (`scratch-r3-extract.txt`, `pr339-body.md`, `scratch-r3-receipts-bins.txt`)**: On `shadow-pool-v2` at `B=1` (`evaluate-v2-b1-receipt.json`, inner `.receipt_sha256` `c7b2455d80b5e5cbda9f5d0c064556a868f3f877217d940b816519fb0c852d77`), `ranking_learnability` reached **`ESTABLISHED_SHADOW`** against strongest baseline `program_prior` (`B=1` gain `+0.033773 [+0.003919, +0.068287]` `GREATER`, pairwise `+0.009851 [+0.004909, +0.014899]` `GREATER`; vs. `ridge_program` gain `+0.057258 [+0.026068, +0.089689]` `GREATER`, pairwise `+0.029015 [+0.023765, +0.034371]` `GREATER`), while remaining `NOT_ESTABLISHED` at `B=8` (`evaluate-v2-b8-receipt.json`, inner `.receipt_sha256` `2ea09e4c48f0c0b225e8739ba362aaf3689f20887cb4bac06813a8e8691ef1a2`) and on `shadow-pool-v1` (`f4b3b873...`, `cc28ddc7...`).

---

## Validation Before Committing (`EXP-019+`)

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
