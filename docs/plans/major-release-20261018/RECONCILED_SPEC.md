# JopeDime Major Release (2026-10-18) — Unified Reconciled Engineering Specification & Gated Delivery Plan

- **Repository:** `neverhuman/JopeDime`
- **Pinned Baseline Commit (`origin/main`):** `a2759e9d375d0697ed40a32c23ac297b5de42347` (proposal review baseline `296d07e1cacc7188b240985e1a52c88676b061e7`)
- **Reconciled Source Specifications:**
  1. `SPEC_A` (`SPEC_A_final_engineering_spec_gated_plan.md` — selects Candidate F/C06 `93.8/100`, 20 work packets `P00..P19`, Releases `R0..R4`, Gates `G0..G10`)
  2. `SPEC_B` (`SPEC_B_language_for_learning_algorithms.md` — selects Candidate P4/C03 `95.2/100`, 26 work packets `GP-00..GP-25`, Milestones `M0..M6`, Gates `G0..G9`)
  3. `SPEC_C` (`SPEC_C_proposal_review_eng_spec_gated_plan.md` — selects Candidate C04 `93.44/100`, 21 work packets `WP00..WP20`, Releases `R0..R7`, Gates `G00..G20`)
- **Additional Mandatory Scope (per `ORDER_from_jepson_1014.md`):**
  - Unfinished/weak/prototype items from the prior 66-item packet (`I53`, `I62` doc-only receipts; `I07/I08/I09/I47/I54/I55/I56` prototype integration seams; `I65` scientific qualification)
  - Root GPU & corpus authority gates (`AUTH-GAP-01` native GPU grant verification & caller bridge; `AUTH-GAP-02` `parallel.label_admission.v2` teaching-store publication gate)
  - Multi-host GPU `E1`/`E2` execution on `xbabe1` (RTX 4090), `xbabe2` (RTX 3090), and `xbabe3` (RTX 3090) (`gpr_cuda_bench` `cases.json` + end-to-end GPU solver/challenger qualification; never `xbabe0`)
  - Flywheel closure spec items (`FLYWHEEL_AUTH_GAP_CLOSURE_SPEC_20261018.json`, `FLYWHEEL_GIT_HISTORY_AUDIT_20261018T1548Z.json`, and 3-arm prospective CPU cohort + artifact registry publication)

---

## 1. Executive Reconciliation & Conflict Resolution Ledger

All three specifications (`SPEC_A`, `SPEC_B`, `SPEC_C`) independently audited the same 7 candidate proposals (6 unique documents; the two `HyperGP` files `63ef13ae2accd55547af819c1eb22f175b94c85c2a50e3cf85b10c7f896cbf16` are byte-identical and unanimously rejected as an implementation baseline, retained only as a long-term research inventory `M10..M13`). All three specifications converge on:
- The provisional successor wire/IR schema **`jope.learning-program/v1`** hosted in `crates/feat-math/src/kernel/ir/` and shared across `jope-pipeline`, `jope-policy`/`edit_policy`, `jope-motifs`, and `jope-records`.
- Preserving the existing 7 execution routes (`Routes 1–7`) and keeping the 401-token native actor (`TOTAL_ACTIONS = 401`, `MAX_SLOTS = 8`) frozen until challenger shadow and prospective confirmation gates pass.
- Fixing the verified codebase defects (`A01..A15`) before expanding language expressiveness.

Where `SPEC_A`, `SPEC_B`, and `SPEC_C` differ, this unified plan resolves each conflict explicitly as follows:

| Conflict Area | `SPEC_A` Position | `SPEC_B` Position | `SPEC_C` Position | Unified Resolution & Rationale |
|---|---|---|---|---|
| **1. Critical-Path Release Ordering** | `R0` Baseline -> `R1` Defects + Pure/Lazy DAG -> `R2` Nested OOF + VarPro + PSD Kernels -> `R3` Programmable Learner + AD + Policy + Library -> `R4` Promotion (`P00..P19`) | `M0` Repairs -> `M1` Pure DAG + Tiny Analytical Learner (`GP-07`) -> `M2` AD + Recipe (`GP-12/13`) -> `M3` Nested OOF + VarPro + Kernels -> `M4` Policy/Library -> `M5/M6` Promotion (`GP-00..GP-25`) | `R0` Baseline -> `R1` Repairs/Bridge -> `R2` Lazy/Pure DAG -> `R3` Nested OOF -> `R4` VarPro + PSD Kernels -> `R5` Learner + AD + Recipe -> `R6` Policy + Library -> `R7` Promotion (`WP00..WP20`) | **Follow `SPEC_C` (`R0..R6`) + `SPEC_A` ordering for the release critical path**, while preserving **`SPEC_B`'s fine-grained packet boundaries (`GP-00..GP-25`)**. Rationale: `NestedFitScope` (`R2`), multi-feature VarPro (`R3`), and PSD kernels (`R3`) extend existing `jope-pipeline` OOF and `feat-math` ridge/kernel machinery with high immediate predictive leverage and lower implementation risk than general bounded-scan tensor AD (`R4`). However, `GP-07` (`MR28`) only depends on `MR22` (`NestedFitScope`), so analytical-gradient learner work can begin in parallel once `MR22` lands. |
| **2. Resource Profile Caps (`compose-small` vs `learner-tiny` vs `learner-small`)** | Single initial profile (`max_expanded_nodes=64`, `max_region_depth=4`, `max_rows=4096`, `max_feature_width=512`, `max_trainable_scalars=4096`, `max_state_bytes=8 MiB`, `max_live_buffer_bytes=128 MiB`, `5s` fixture deadline) | `learner-tiny` (`max_base_nodes=128`, `max_ad_transformed_nodes=512`, `max_region_depth=4`, `max_state_bytes=1 MiB`, `max_temp_buffer_bytes=32 MiB`, `max_ad_tape_bytes=32 MiB`, `max_feature_width=64`) | 3-tier profile table (`compose-small`, `learner-tiny`, `learner-small`) scaling up to `rows=65536`, `width=512`, `trainable_scalars=8192`, `state=8 MiB`, `temp=128 MiB` | **Follow `SPEC_C` §16.1's 3-tier profile hierarchy (`compose-small`, `learner-tiny`, `learner-small`)** augmented with **`SPEC_B` §14.1's explicit `max_base_nodes` (128) vs `max_ad_transformed_nodes` (512) and `max_ad_tape_bytes` (32 MiB) separation** and **`SPEC_A` §10.3's pre-allocation checked-arithmetic rules and 5s fixture deadline**. Rationale: A single profile either starves tabular feature width (`64` in `SPEC_B`) or over-allocates for tiny analytical learners (`8 MiB` in `SPEC_A`); 3 explicit profiles give tight bounds per workload class. |
| **3. Statistical Default-Promotion Confidence Bound** | One-sided **97.5% UCB (`alpha = 0.025`)** on paired log-loss ratio $\Delta_d = \log((\text{RMSE}_{\text{new}}+\epsilon)/(\text{RMSE}_{\text{base}}+\epsilon))$, stratified cluster bootstrap $B=10{,}000$ | One-sided 95% UCB on quality ($\le \log(0.99)$) or efficiency ($\le 0$ with cost $\le 0.90$), plus failure-rate $\Delta \le 0.005$ | One-sided 95% UCB on quality ($< \log(0.99)$) or efficiency ($< \log(1.00)$ with cost $< 0.90$) | **Follow `SPEC_A` §11 Gate G9's stricter one-sided 97.5% UCB (`alpha = 0.025`)** combined with **`SPEC_B` §16.3's zero-quality-deterioration floor (`UCB <= 0.0`) on the efficiency path and `<= 0.005` failure-rate guardrail**. Rationale: Matches JopeDime's existing prospective cohort statistical standard (`97.5%` CI in `SC-DEC-072`) and prevents promoting a noisier search space on marginal 95% fluctuations. |
| **4. Rollout Switches & Budget Accounting Modes** | 3 budget modes: `Shadow`, `AdditiveSideBudget`, `FixedTotalBudget` | 3 rollout switches: `Available`, `ExperimentalSearch`, `Default` | 3 budget modes: `Legacy`, `ProtectedExtension`, `FixedBudgetChallenger` | **Unify both orthogonal axes**: `ActivationMode in {Available, ExperimentalSearch, Default}` (`SPEC_B`) crossed with `BudgetMode in {LegacyShadow, ProtectedExtension, FixedBudgetChallenger}` (`SPEC_A`/`SPEC_C`). Rationale: Format availability/default routing and search-budget deduction are distinct operational controls during canary and rollback. |
| **5. Canonical Program Identity Digests** | 3 digests (`structural_id`, `semantic_id`, `fit_contract_id`) | 3 digests (`structural_hash`, `semantic_hash`, `fitted_artifact_hash`) | 4 digests (`program_semantic_id`, `program_structural_id`, `fit_contract_id`, `execution_plan_id`) + `fitted_artifact_id` | **Follow `SPEC_C` §9.1's 4-digest separation (`program_semantic_id`, `program_structural_id`, `fit_contract_id`, `execution_plan_id`) plus `SPEC_B`'s `fitted_artifact_id`**. Rationale: Separating `execution_plan_id` from `program_semantic_id` allows CPU/GPU backend scheduling changes without invalidating semantic deduplication caches, while `fit_contract_id` prevents cross-fold/cross-label OOF cache collisions. |

---

## 2. Architecture & Language Specification (`jope.learning-program/v1`)

### 2.1 Four-Stratum Typed IR (`crates/feat-math/src/kernel/ir/`)
1. **Stratum A — Pure Value & Feature DAG (`Purity::PureRowwise`):**
   - `RawColumn(SchemaColId)`, `DerivedFeature(FeatureSlotId)`, `LiteralScalar { quantized_bin: Option<u16>, exact_f64: f64 }`, `UnaryOp`, `BinaryOp`, `TernaryOp`, `Let { var, bound, body }`, `LazyIf { cond, then_region, else_region }`, `RouteMask`, `ConcatFeatures`, `SelectFeature`.
   - Explicitly separates `ValueTensor { f64, ValueMask }` so genuine `0.0` and missing (`NaN`/masked) never conflate (`A02`/`A12`).
2. **Stratum B — Fit-State & Cross-Fit Procedure DAG (`Purity::FitTransform` / `NestedFit`):**
   - `FitTransform { op, fit_hyperparams, input }`, `CrossFitMap { partition_plan, inner_learner, input }`, `FitPredict { learner, features, target }`, `Calibrate { method, input }`.
   - Enforces `NestedFitScope`: inner fit sees only `DataView { rows: fold_train_rows }` (`RowClosure::narrow` preserved); outer validation rows and labels are structurally unreachable (`G04`).
3. **Stratum C — Structured Solver & PSD Kernel Layer (`Purity::StructuredSolve`):**
   - **Multi-Feature Variable Projection (`VarPro`):** Jointly fits basis parameters $\theta$ and linear weights $w(\theta) = \arg\min_w \|\Phi_\theta(X)w - y\|_2^2 + \lambda\|w\|_2^2$ via augmented QR solve $\begin{bmatrix}\Phi_\theta(X)\\ \sqrt{\lambda}I\end{bmatrix} w \approx \begin{bmatrix}y\\ 0\end{bmatrix}$ with column norm stabilization and explicit choice of `EnvelopeTheorem` vs `ValidationThroughFit` hypergradient mode.
   - **Constructive PSD Kernel Grammar (`KernelExpr`):** `Rbf`, `Matern`, `Linear`, `Polynomial`, `RationalQuadratic`, closed under `Sum`, `Product`, `PositiveScale`, `FeaturePullback(phi, k)`, `GatedKernel(g, k)` ($g(x)g(x')k(x,x')$), `LinearMapDot(B)`, plus explicit `NystromApprox` and `RffApprox` nodes with distinct `program_semantic_id`s.
4. **Stratum D — Bounded Iterative Learner & First-Order AD (`Purity::BoundedLearner`):**
   - Explicit `TrainState` (parameters with `TieId`, optimizer state, immutable support sets, monotone thresholds), `initialize`, `update` (`Scan` with compile-time `max_steps` and `BatchSchedule`), `finalize`, and `predict`.
   - Elementary losses (`Squared`, `Huber`, `Logistic`, `Pinball`, `Margins`) and proximal/projection operators (`SoftThreshold`, `L2Shrink`, `ProjectBox`, `ProjectL2Ball`, `ProjectSimplex`, SPD Mahalanobis metric $M = \operatorname{diag}(\operatorname{softplus}(a)+\varepsilon) + UU^\top$).
   - First-order forward (`JVP`) and reverse (`VJP`) AD over the primitive whitelist, accumulating gradients by `TieId` and solving implicit linear adjoints $A^\top \bar{b} = \bar{x}$ without unrolling solver iterations.

### 2.2 Dual Artifact Emission Contract (`G08`)
Every admitted program supports two distinct, deterministic exports:
1. **Standalone Fitted Predictor Artifact (`.rs` + `.py`):** Contains frozen fitted parameters/stats/support sets, executes in a fresh process with zero training data and zero `JopeDime` runtime dependencies, and matches in-process predictions within $\le 10^{-12}$ (CPU `f64`).
2. **Standalone Training Recipe Artifact (`.rs`):** Contains the full `initialize`/`update`/`finalize`/`predict` training algorithm, retrains from scratch on fresh synthetic/held-out data in a fresh process, and matches in-process training loss and predictions within $\le 10^{-10}$.

---

## 3. Release Gates (`G00..G20` + Authority & GPU Gates)

| Gate ID | Name | Blocking Release / Tasks | Mandatory Acceptance Criterion |
|---|---|---|---|
| `G00` | Baseline, Capability Join & Legacy Receipt Hardening | Wave 0 / `R0` (`MR01..MR03`, `MR11..MR12`) | 64-fixture golden bank frozen; `I53`/`I62` upgraded to executable `cargo test --exact` receipts; generated capability join passes with 0 drift across `registry_v2`, profiles, policy masks, evaluators, and emitters. |
| `G-AUTH-01` | Native GPU Grant Verification & Bridge (`AUTH-GAP-01`) | Wave 0 (`MR04..MR05`) | Production `controller.rs` and per-rank `participant.rs` call `verify_gpu_task_grant` against `jope.smartcluster_fit_policy.v1` + `SignedNativeGrant`; `require_original_gpu_grant` fails closed on any schema/signature/epoch/slot/uid mismatch. |
| `G-AUTH-02` | Strict Teaching-Store Admission (`AUTH-GAP-02`) | Wave 0 (`MR06..MR07`) | `parallel.label_admission.v2` preflight validator & 7,877-shard payload hash verifier pass; dataset license & owner-approved evaluator witness bound before production teaching-store publication. |
| `G-GPU-E1` | Multi-Host CUDA Kernel Benchmark Suite (`GPU-E1`) | Wave 0 (`MR08..MR09`) | `gpr_cuda_bench` `cases.json` runs on `xbabe1` (RTX 4090), `xbabe2` (RTX 3090), and `xbabe3` (RTX 3090) with `residual_l2_rel <= 1e-6`, `pred_rmse_rel_diff <= 1e-5`, and hardware receipts committed. |
| `G-GPU-E2` | Multi-Host GPU Solver/Challenger Qualification (`GPU-E2`) | Wave 0 (`MR10`) | End-to-end GPU solver/challenger qualification across `xbabe1-3` GPUs and `I65` scientific qualification receipt verified. |
| `G01` | Historical Compatibility & Defect Repair Gate | `R0` (`MR13`, `MR15`) | 100% of historical `legacy_program_v1`/`LegacyFeatProgram` fixtures translate identically (`<= 1e-12`) or wrap in explicit `OpaqueLegacyNode` (`0` silent drops); `NaN != 0.0` mask separation and removal of accidental $\pm 60$ linear clamp verified. |
| `G02` | Wire Determinism & Parameter Effect Gate | `R0` (`MR14`, `MR16`) | `jope.learning-program/v1` canonical round-trip byte-identical across 10,000 proptest cases; `l1_ratio in [0,1]` domain bug (`A01`) fixed; every parameterized op has a passing existential parameter-effect witness test. |
| `G03` | Schema Pointer & Off-Bin Numeric Gate | `R1` (`MR17`) | Columns `0, 7, 8, 17, 63, 511` Round-trip and evaluate identically under column permutation when accessed by `SchemaColId`; exact off-bin `f64` literals (`0.137`, `1.7320508075688772`) survive serialization without bin quantization loss. |
| `G04` | Zero-Leakage Nested OOF Gate | `R2` (`MR22..MR24`) | Adversarial label-poisoning (`1e9`, `-1e9`, `NaN`) on validation fold produces `0` bit change in inner fitted states; distinct selection/tuning histories produce distinct `fit_contract_id` cache keys (`A09`/`A10`). |
| `G05` | Lazy Evaluation, DAG Sharing & State Discipline Gate | `R1`/`R4` (`MR18..MR19`, `MR28`) | `LazyIf` never evaluates unselected branch on poison traps (`W03`); shared DAG node with $k \ge 3$ consumers executes exactly once (`W04`); bounded `Scan` rejects unbounded/mutating-predict programs at validation time. |
| `G06` | Numerical Stability & PSD Kernel Gate | `R3`/`R4` (`MR26`, `MR29`) | Constructive `KernelExpr` grammar rejects indefinite forms (`K1 - K2`) at typecheck; empirical Gram matrices satisfy $\lambda_{\min}(G + \varepsilon I) \ge -10^{-10}$; proximal/projection operators satisfy Moreau & idempotence identities within $\le 10^{-8}$. |
| `G07` | First-Order AD & Implicit/VarPro Derivative Gate | `R3`/`R4`/`R6` (`MR25`, `MR30`, `MR39`) | Analytic JVP/VJP, tied-parameter gradients, VarPro envelope/validation-through-fit gradients, and implicit linear solve adjoints match central finite differences (`rel_err <= 1e-5`) and satisfy $\langle Jv, u\rangle = \langle v, J^\top u\rangle$ within $\le 10^{-10}$. |
| `G08` | Dual Artifact Export & Fresh-Process Parity Gate | `R1`..`R4` (`MR21`, `MR24`, `MR27`, `MR31..MR32`) | Exported Rust & Python predictors match in-process predictions within $\le 10^{-12}$ in a fresh process; exported Rust training recipes retrain from scratch in a fresh process and match in-process loss/predictions within $\le 10^{-10}$. |
| `G09` | Constructive GP Edit Soundness Gate | `R1` (`MR20`) | 100,000 random mutations and two-parent crossovers over `jope.learning-program/v1` produce `0` panics, `0` unbound variables/pointers, and `0` invalid survivors entering evaluation. |
| `G10` | Factored Edit Policy & Corpus Accounting Gate | `R5` (`MR33`) | Grammar-masked edit policy emits `< 0.1%` invalid actions across 10,000 rollouts; trace log-prob and multi-trace lower-bound sums are logged separately; all-attempt trajectory logging records 100% of failed/pruned candidates. |
| `G14` | Challenger Policy Equal-Compute Gate | `R5` (`MR34`) | Trained successor typed-edit policy outperforms random valid edits and matches/exceeds baseline proposal distribution under identical wall-clock/eval budget (`E07`), while frozen 401-token native actor remains bit-compatible. |
| `G15` | Hygienic Typed Library Learning Gate | `R5` (`MR35`) | Extracted library abstractions pass capture-avoiding hygiene checks, charge expanded complexity cost during search, and improve held-out task discovery under the 4-arm ablation (`E06`). |
| `G17` | Resource Enforcement & Pruned-Candidate Fidelity Gate | `R6` (`MR36`) | Pre-allocation checked arithmetic catches 100% of synthetic memory/step bombs in `< 50 ms`; default-off wall-clock/RSS overhead UCB $\le 1.02$; 10% randomized high-fidelity audit of pruned candidates satisfies rank correlation $\ge 0.70$ and regret bound (`E08`). |
| `G19` | Prospective Paired Confirmation Gate | `R6` (`MR37`) | Family-weighted paired log-loss ratio $\Delta_d$ satisfies one-sided **97.5% UCB `< log(0.99)`** (quality path) or **`<= 0.0` with cost ratio `< 0.90`** (efficiency path) across held-out datasets (`D >= 24`, $B=10{,}000$ cluster bootstrap), with failure-rate delta $\le 0.005$. |
| `G20` | Atomic Bundle Activation & Rehearsed Rollback Gate | `R6` (`MR38`, `MR40`) | Atomic compare-and-swap bundle switch preserves in-flight jobs on their pinned manifest and completes a rehearsed rollback to legacy default in `< 60 s` with `0` failed jobs. |

---

## 4. Complete Cross-Reference Matrix (`MR01..MR40` <-> `SPEC_A` / `SPEC_B` / `SPEC_C` / Legacy & Root Gates)

| Task ID | Wave / Release | Owner Screen | `SPEC_A` | `SPEC_B` | `SPEC_C` | Legacy / Root / Flywheel Scope | Gate |
|---|---|---|---|---|---|---|---|
| `MR01` | Wave 0 | `2830191.jopedime_agy` | — | — | — | Harden `I53` & `I62` doc-only receipts to executable `cargo test --exact` | `G00` |
| `MR02` | Wave 0 | `487851.jope_review_claude` | — | — | §3.2 | Audit & wire `I07/I08/I09/I47/I54/I55/I56` prototype seams into regression suite | `G00` |
| `MR03` | Wave 0 | `2724936.captain_jopedime` | — | — | — | Publish 3-arm CPU cohort compact roots + `ROOT_TRAINING_ARTIFACT_LOCATION_*` (`PR #362`) | `G00` |
| `MR04` | Wave 0 | `2830191.jopedime_agy` | — | — | — | `AUTH-GAP-01`: Native GPU grant verifier (`jope.smartcluster_fit_policy.v1` + `SignedNativeGrant`) & caller wiring | `G-AUTH-01` |
| `MR05` | Wave 0 | `2724936.captain_jopedime` | — | — | — | `AUTH-GAP-01`: Owner/Root policy (`uid 0`) & signed grant issuance binding for `xbabe1-3` | `G-AUTH-01` |
| `MR06` | Wave 0 | `1061448.jopedime_flywheel` | — | — | — | Flywheel `parallel.label_admission.v2` preflight validator & 7,877-shard payload hash verifier | `G-AUTH-02` |
| `MR07` | Wave 0 | `1061448.jopedime_flywheel` | — | — | — | `AUTH-GAP-02`: Dataset license, evaluator witness & `clean_completion.v3` teaching-store closure | `G-AUTH-02` |
| `MR08` | Wave 0 | `2830191.jopedime_agy` | — | — | — | `GPU-E1`: Author canonical `cases.json` & harness for `gpr_cuda_bench` | `G-GPU-E1` |
| `MR09` | Wave 0 | `2724936.captain_jopedime` | — | — | — | `GPU-E1`: Execute `gpr_cuda_bench` across `xbabe1` (4090), `xbabe2` (3090), `xbabe3` (3090) | `G-GPU-E1` |
| `MR10` | Wave 0 | `2724936.captain_jopedime` | — | — | — | `GPU-E2`: Multi-host GPU solver/challenger qualification on `xbabe1-3` & `I65` scientific qualification | `G-GPU-E2` |
| `MR11` | `R0` | `487851.jope_review_claude` | `P00` | `GP-00` | `WP00` | Baseline commit/env manifest, route inventory (`Routes 1–7`) & 64-fixture golden bank | `G00` |
| `MR12` | `R0` | `1061448.jopedime_flywheel` | `P00` | `GP-01` | `WP01` | Generated capability join across `registry_v2`, profiles, policy, evaluators & emitters | `G00` |
| `MR13` | `R0` | `487851.jope_review_claude` | `P01` | `GP-04` | `WP04` | Historical compatibility harness & `LegacyBridge` (`OpaqueLegacyNode`, 0 silent drops) | `G01` |
| `MR14` | `R0` | `1061448.jopedime_flywheel` | `P02` | `GP-02` | `WP02` | Fix `l1_ratio` domain bug (`A01`/`A13`) & existential parameter-effect witness suite | `G02` |
| `MR15` | `R0` | `2830191.jopedime_agy` | `P03` | `GP-03` | `WP02` | Fix `NaN -> 0.0` conflation (`A02`/`A12`), remove silent $\pm 60$ clamp (`A04`/`A14`) & exact width check | `G01` |
| `MR16` | `R0` | `2830191.jopedime_agy` | `P02` | `GP-04` | `WP03` | `jope.learning-program/v1` wire envelope, strict reader (`A11`/`A15`) & 4-digest identity codec | `G02` |
| `MR17` | `R1` | `2830191.jopedime_agy` | `P04` | `GP-05` | `WP05` | `SchemaId` vs `LayoutId` separation, >8-slot/`0..511` column pointers & exact off-bin `f64` payload (`W02`) | `G03` |
| `MR18` | `R1` | `2830191.jopedime_agy` | `P05` | `GP-05` | `WP06` | Region-based lazy scalar conditional (`LazyIf`) & partitioned batch routing (`W03`) | `G05` |
| `MR19` | `R1` | `1061448.jopedime_flywheel` | `P05` | `GP-05` | `WP06` | Pure shared feature DAG (`Let`, branch/join CSE, `ConcatFeatures`, single-eval counter `W04`) | `G05` |
| `MR20` | `R1` | `1061448.jopedime_flywheel` | `P06` | `GP-09` | `WP10` | Constructive GP generation, typed mutation & two-parent crossover (100k edit soundness) | `G09` |
| `MR21` | `R1` | `487851.jope_review_claude` | `P07` | `GP-10` | `WP06` | Standalone Rust & portable Python fitted predictor emission and fresh-process parity (`R1`) | `G08` |
| `MR22` | `R2` | `2830191.jopedime_agy` | `P08` | `GP-06` | `WP07` | `NestedFitScope` capability (`RowClosure::narrow`), projected `DataView` & ancestry cache keys (`A09`/`A10`) | `G04` |
| `MR23` | `R2` | `1061448.jopedime_flywheel` | `P08` | `GP-06` | `WP08` | Two-level OOF stacking, cross-fitted residual correction vs boosting (`W04`/`W05`) & target encoding | `G04` |
| `MR24` | `R2` | `487851.jope_review_claude` | `P08` | `GP-06` | `WP08` | Standalone Rust/Python fitted export for nested OOF stacks + label-poisoning adversarial suite | `G08` |
| `MR25` | `R3` | `2830191.jopedime_agy` | `P10` | `GP-14` | `WP09` | Multi-feature variable projection (augmented QR solve, scale normalization, envelope/val derivatives `W06`/`E02`) | `G07` |
| `MR26` | `R3` | `1061448.jopedime_flywheel` | `P09` | `GP-15` | `WP10` | Constructive PSD kernel AST (`Sum`, `Product`, `Pullback`, `Gated`, `LinearMapDot`) & `LinearOperator` KRR (`W05`/`E03`) | `G06` |
| `MR27` | `R3` | `487851.jope_review_claude` | `P09` | `GP-15` | `WP10` | Explicit whitened Nyström & RFF kernel approximations + Rust/Python fitted export | `G08` |
| `MR28` | `R4` | `2830191.jopedime_agy` | `P11` | `GP-07` | `WP11` | Explicit `TrainState`, `ParameterSpec` (`TieId`), bounded `Scan` lifecycle & analytical sparse linear learner (`W07`) | `G05` |
| `MR29` | `R4` | `1061448.jopedime_flywheel` | `P12` | `GP-16` | `WP14` | Elementary loss & proximal library (`Huber`, `Logistic`, `SoftThreshold`, `Simplex`, SPD metric $M$, `W08`/`E04`) | `G06` |
| `MR30` | `R4` | `2830191.jopedime_agy` | `P13` | `GP-12` | `WP12` | First-order tensor JVP/VJP & bounded-scan AD with `TieId` accumulation & implicit solve adjoint $A^\top\bar{b}=\bar{x}$ | `G07` |
| `MR31` | `R4` | `487851.jope_review_claude` | `P13` | `GP-13` | `WP13` | Tiny MLP & SGD/Momentum/Adam learners + standalone Rust training recipe export (`W08`/`W10`/`W11`) | `G08` |
| `MR32` | `R4` | `1061448.jopedime_flywheel` | `P11` | `GP-11` | `WP15` | Non-AD discrete & support learners (`W12` tree stump split, `W13` kNN/support, weighted PAVA isotonic) | `G08` |
| `MR33` | `R5` | `2830191.jopedime_agy` | `P14` | `GP-17` | `WP16` | Factored grammar-masked typed-edit action space, trace vs program probability accounting & all-attempt logging | `G10` |
| `MR34` | `R5` | `1061448.jopedime_flywheel` | `P14` | `GP-18` | `WP16` | Successor typed-edit policy training/export/reload/proposal wiring & equal-compute ablation (`E07`) | `G14` |
| `MR35` | `R5` | `487851.jope_review_claude` | `P15` | `GP-19` | `WP17` | Hygienic typed parameterized library learning (`jope-motifs`, $\Delta\text{MDL}>0$, 4-arm discovery study `E06`) | `G15` |
| `MR36` | `R6` | `2830191.jopedime_agy` | `P16` | `GP-20,21` | `WP18` | 3-tier resource profiles, pre-allocation checked bounds, fuel ledger & 10% pruned-candidate fidelity audit (`E08`) | `G17` |
| `MR37` | `R6` | `2724936.captain_jopedime` | `P17` | `GP-22` | `WP19` | Prospective family-weighted paired confirmation study (one-sided 97.5% UCB, 6 ablation arms) | `G19` |
| `MR38` | `R6` | `2724936.captain_jopedime` | `P18` | `GP-23` | `WP19` | Atomic compare-and-swap bundle activation, pinned in-flight jobs & rehearsed `<60s` rollback drill | `G20` |
| `MR39` | `R6` | `1061448.jopedime_flywheel` | `P19` | `GP-24` | `WP20` | Convex output potential layer ($\Omega_\alpha(p)=\alpha p^2/2$) with learned $\alpha>0$ derivative witness (`W17`/`E05`) | `G07` |
| `MR40` | `R6` | `487851.jope_review_claude` | `P19` | `GP-25` | `WP20` | Task-gated domain packs (`M10` implicit inference, `M11` calibration, `M12` semirings, `M13` symmetry/OT) | `G20` |
