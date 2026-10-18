# JopeDime Major Release (2026-10-18) — Unified Reconciled Engineering Specification & Gated Delivery Plan

- **Repository:** `neverhuman/JopeDime`
- **Pinned Baseline Commit (`origin/main`):** `e0e14ddb8646deff34e3c9d45b46cc236c2fdcc2` (proposal review baseline `296d07e1cacc7188b240985e1a52c88676b061e7`)
- **Reconciled Source Specifications:**
  1. `SPEC_A` (`SPEC_A_final_engineering_spec_gated_plan.md` — selects Candidate F/C06 `93.8/100`, 20 work packets `P00..P19`, Releases `R0..R4`, Gates `G0..G10`)
  2. `SPEC_B` (`SPEC_B_language_for_learning_algorithms.md` — selects Candidate P4/C03 `95.2/100`, 26 work packets `GP-00..GP-25`, Milestones `M0..M6`, Gates `G00..G20`)
  3. `SPEC_C` (`SPEC_C_proposal_review_eng_spec_gated_plan.md` — selects Candidate C04 `93.44/100`, 23 work packets `WP00..WP12`, `WP13A`, `WP13B`, `WP14..WP21`, Releases `R0..R7`, Gates `G00..G14`)
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

## 4. Source-accurate GP scope and cross-reference matrix

The original MR40 synthesis omitted SPEC_B `GP-08`, substituted discrete learners for `GP-11`, misnumbered several SPEC_C packets, and did not separately own SPEC_C `WP13B` or SPEC_A `P03` route closure. This revision retains all 40 tasks and adds four explicit obligations: `MR41` (small tensor/solve reference), `MR42` (full real-consumer tiny-learner slice), `MR43` (state/update genetic edits), and `MR44` (numeric-v2 actual caller/target closure).

The exact source identities, packet text and all mappings are in [GP_SPEC_COVERAGE.json](GP_SPEC_COVERAGE.json). SPEC_C has **23**, not 21, packet IDs: `WP00..WP12`, `WP13A`, `WP13B`, `WP14..WP21`. Source gate numbers are namespaced; SPEC_C `G14` means release/rollback, whereas SPEC_B `G14` means real neural wiring. Unified gates in section 3 retain their own names; equal numbers are never silently equated.

`GP-08`'s complete analytical ridge reference follows the bounded learner state in `MR28`. Earlier `MR25`/`MR26` numerical work continues to use qualified existing solver references; it cannot claim GP-08 closed. `MR42` initially needs lawful feature edits and crossover on the analytical tiny learner, without depending on general AD or transferring fitted state. `MR43` later adds full state/update/scan edits after AD. `MR33` cannot enable state actions before that gate. `MR44` closes real numeric-v2 consumers before `MR21` claims portable export of that supported route. Preliminary historical replay under `MR13` does not close the full successor legacy bridge until `MR16`'s strict reader also passes.

No source work package disappears because its candidate lost a ranking. Optional packs remain named, task-gated research obligations; they are not compulsory default generation. Existing 401-token actor, accepted evaluators, scientific family boundaries, authority gates and active jobs retain their identities. Default-search adoption still requires prospective matched-resource evidence; a negative or inconclusive experiment remains a valid recorded outcome.

Read [GP_UPGRADE_WORKPLAN.md](GP_UPGRADE_WORKPLAN.md) for exact new file/symbol scopes, edge cases, stop conditions, validation and handoff rules. All proposed new tests must be implemented and collect real cases before any receipt can pass.

| Source packet | Required MR tasks (all) | Source lines |
|---|---|---|
| `spec_a:P00` | `MR11,MR12` | `1160..1160` |
| `spec_a:P01` | `MR13` | `1161..1161` |
| `spec_a:P02` | `MR13,MR16` | `1162..1162` |
| `spec_a:P03` | `MR15,MR44` | `1163..1163` |
| `spec_a:P04` | `MR17` | `1164..1164` |
| `spec_a:P05` | `MR18,MR19` | `1165..1165` |
| `spec_a:P06` | `MR20,MR43` | `1166..1166` |
| `spec_a:P07` | `MR21,MR42` | `1167..1167` |
| `spec_a:P08` | `MR22,MR23,MR24` | `1168..1168` |
| `spec_a:P09` | `MR26,MR27,MR41` | `1169..1169` |
| `spec_a:P10` | `MR25` | `1170..1170` |
| `spec_a:P11` | `MR28,MR32,MR41,MR42,MR43` | `1171..1171` |
| `spec_a:P12` | `MR29` | `1172..1172` |
| `spec_a:P13` | `MR30,MR31,MR43` | `1173..1173` |
| `spec_a:P14` | `MR33,MR34` | `1174..1174` |
| `spec_a:P15` | `MR35` | `1175..1175` |
| `spec_a:P16` | `MR36` | `1176..1176` |
| `spec_a:P17` | `MR37` | `1177..1177` |
| `spec_a:P18` | `MR38` | `1178..1178` |
| `spec_a:P19` | `MR39,MR40` | `1179..1179` |
| `spec_b:GP-00` | `MR11` | `1034..1034` |
| `spec_b:GP-01` | `MR12` | `1035..1035` |
| `spec_b:GP-02` | `MR14,MR16` | `1036..1036` |
| `spec_b:GP-03` | `MR15,MR17,MR44` | `1037..1037` |
| `spec_b:GP-04` | `MR13,MR16` | `1038..1038` |
| `spec_b:GP-05` | `MR17,MR18,MR19` | `1039..1039` |
| `spec_b:GP-06` | `MR22,MR23,MR24` | `1040..1040` |
| `spec_b:GP-07` | `MR28` | `1041..1041` |
| `spec_b:GP-08` | `MR41` | `1042..1042` |
| `spec_b:GP-09` | `MR20,MR43` | `1043..1043` |
| `spec_b:GP-10` | `MR21,MR42` | `1044..1044` |
| `spec_b:GP-11` | `MR42` | `1045..1045` |
| `spec_b:GP-12` | `MR30` | `1046..1046` |
| `spec_b:GP-13` | `MR31` | `1047..1047` |
| `spec_b:GP-14` | `MR25` | `1048..1048` |
| `spec_b:GP-15` | `MR26,MR27` | `1049..1049` |
| `spec_b:GP-16` | `MR29` | `1050..1050` |
| `spec_b:GP-17` | `MR33` | `1051..1051` |
| `spec_b:GP-18` | `MR34` | `1052..1052` |
| `spec_b:GP-19` | `MR35` | `1053..1053` |
| `spec_b:GP-20` | `MR36,MR37` | `1054..1054` |
| `spec_b:GP-21` | `MR36` | `1055..1055` |
| `spec_b:GP-22` | `MR37` | `1056..1056` |
| `spec_b:GP-23` | `MR38` | `1057..1057` |
| `spec_b:GP-24` | `MR29,MR39` | `1058..1058` |
| `spec_b:GP-25` | `MR40` | `1059..1059` |
| `spec_c:WP00` | `MR11` | `1430..1437` |
| `spec_c:WP01` | `MR12` | `1438..1445` |
| `spec_c:WP02` | `MR14,MR15,MR44` | `1446..1453` |
| `spec_c:WP03` | `MR16` | `1454..1461` |
| `spec_c:WP04` | `MR13,MR16` | `1462..1469` |
| `spec_c:WP05` | `MR17` | `1470..1477` |
| `spec_c:WP06` | `MR18,MR19,MR21` | `1478..1485` |
| `spec_c:WP07` | `MR22` | `1486..1493` |
| `spec_c:WP08` | `MR23,MR24` | `1494..1501` |
| `spec_c:WP09` | `MR25` | `1502..1509` |
| `spec_c:WP10` | `MR26,MR27,MR41` | `1510..1517` |
| `spec_c:WP11` | `MR28,MR32,MR41,MR42` | `1518..1525` |
| `spec_c:WP12` | `MR30` | `1526..1533` |
| `spec_c:WP13A` | `MR20` | `1534..1541` |
| `spec_c:WP13B` | `MR43` | `1542..1548` |
| `spec_c:WP14` | `MR33,MR34` | `1549..1555` |
| `spec_c:WP15` | `MR31,MR32,MR42` | `1556..1562` |
| `spec_c:WP16` | `MR29,MR39` | `1563..1569` |
| `spec_c:WP17` | `MR35` | `1570..1576` |
| `spec_c:WP18` | `MR36` | `1577..1583` |
| `spec_c:WP19` | `MR37` | `1584..1590` |
| `spec_c:WP20` | `MR38` | `1591..1597` |
| `spec_c:WP21` | `MR40` | `1598..1604` |
