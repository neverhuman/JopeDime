# Algorithm Synthesis Model (ASM)

ASM **learns to generate and optimize executable algorithms from data**. This is
its product direction, not a claim that every part of the loop is qualified.
Discovery programs, optimization histories, training inputs and provenance remain private.
Readable generated source reveals the computation it implements.

## Current capability inventory

This inventory was refreshed from the source at the first ASM contract PR, based on
main `95e92af34fc95854f102965b3f4fbf01d08beeb7`. `jope doctor` reports these boundaries.
A declared operator, a runnable adapter, a passing fixture and a qualified live loop
are separate milestones.

| Component | Implemented source | Current product boundary |
| --- | --- | --- |
| Regression search | [pipeline search](../crates/jope-pipeline/src/search.rs), existing `feat-math` fit/evolution | Numeric finite inputs; generic generation-zero seeds; in-process fit can overrun its allowance. |
| Native method metadata | [governed registry](../crates/feat-math/src/registry.rs) | Algorithms, aliases and lifecycle states exist; metadata is not an executable benchmark roster. |
| Classification | [advanced-v3](../crates/feat-math/src/advanced_v3.rs) and its fitted heads | Binary logistic/tree/boosting and multiclass ridge, OvR logistic, diagonal LDA and nearest-centroid implementations exist. Product regression search now refuses other tasks before creating output. |
| Task metrics | [panel evaluator](../crates/jope-bench/src/panel/score.rs) | RMSE, binary AUROC and multiclass probability-matrix log loss use original labels and explicit class ordering. The multiclass fitting objective remains macro-F1; confirmation uses log loss. |
| External methods | [typed adapters](../crates/jope-bench/adapters/_panel.py) | XGBoost, LightGBM, CatBoost and scikit-learn RF accept nullable raw cells, fit preprocessing on training rows, and retain complete private fitted artifacts. Dependencies and campaign execution still require qualification. |
| Neural proposals | [bounded proposal evaluation](../crates/jope-pipeline/src/neural_search.rs) | Source connects verified proposals to fitted generation-zero candidates and training-side selection. Actual accepted-model consumption and CPU context parity still need qualification. |
| Training | [existing trainer](../crates/jope-engine/src/training.rs), Python trainers | Execution/category/provenance qualification remains. `jope train-step` is a centroid router. Existing sequence, quality and runtime objectives remain unchanged. |
| Export | Legacy transpiler; advanced bound emitters and fitted C++ runtime | Coverage differs; emitted training recipes do not establish parity of a fitted predictor. Multiclass matrices need one shared output contract. |
| Reuse | Existing champions and learning/HWM ledgers | General typed motif mining and governed operator promotion are pending. |

The legacy fixture `fit`, `search` and centroid demonstrations report numeric R²
diagnostics, including for non-regression fixtures; they are not task-qualified
classification evidence and are excluded from this panel execution path.

The existing production fit-authority fences and accepted regression-only loader
remain intact. Classification training adapters need explicit experimental lineages;
serving pointers require their existing qualification and promotion procedure.

## Frozen private panel

[panel.json](../benchmarks/panel.json) is frozen at SHA256
`ba7dace4cd535687002cc95538dbc7f7aaf77ec973177c7159b3c6d22471eee0`.
It uses [PMLB revision 7c1f4bd](https://github.com/EpistasisLab/pmlb/tree/7c1f4bdc00136dc2e55c87fa6b8ba6e8af6d1a68).
[Selection criteria](../benchmarks/panel-selection.json) were applied to metadata
before fitting. This panel excludes the historical favorable `test-10` and preserves
original PMLB task definitions and labels. PMLB's upstream encodings are described in
its pinned per-dataset metadata; they are not newly relabeled by ASM.

| Regression (RMSE) | Binary (AUROC) | Multiclass (log loss) |
| --- | --- | --- |
| 197_cpu_act | breast_cancer_wisconsin_diagnostic | iris |
| 225_puma8NH | credit_approval_germany | wine_recognition |
| 230_machine_cpu | congressional_voting_records | car_evaluation |
| 522_pm10 | haberman | balance_scale |
| 529_pollen | ionosphere | dermatology |
| 579_fri_c0_250_5 | sonar | segmentation |
| auto_insurance_price | spambase | vehicle |
| 690_visualizing_galaxy | phoneme | optdigits |
| 712_chscase_geyser1 | tic_tac_toe | penguins |
| 503_wind | horse_colic_surgery | tae |

Puma, pollen and Friedman provide simulated regression data; the panel is not a
claim of thirty independent real-world families. Checkpoint exposure is unknown
unless separately established. Dataset licenses are recorded individually, including
unknown original licenses; PMLB's repository MIT license does not establish every
upstream license. Data is not redistributed in inference bundles.

Preparation uses pinned scikit-learn 1.7.2 grouped split semantics and seed 20260922.
Identical feature rows are joined irrespective of label; explicitly declared date
groups (`pm10`: day, `wind`: year/month) are joined transitively. One fifth is sealed
first; the remaining rows form five development folds. Classification is stratified
and every fitting/scoring partition must contain every declared class. The actual
row assignments and prepared bytes are hashed. Grouping cannot recover absent subject
or time identifiers; this is tabular development evidence, not temporal forecasting
or an independence guarantee. Ordinal inputs use categorical encoding. No imputation,
scaling, feature selection or outcome measurement occurs in preparation.

Use an external data directory. Install the optional benchmark dependencies through
`uv sync --project python --extra benchmark` (Python 3.12 is the tested environment):

```sh
cargo build --locked -p jope-cli
./target/debug/jope doctor
./target/debug/jope benchmark verify-manifest --manifest benchmarks/panel.json
uv run --project python --extra benchmark python benchmarks/prepare.py \
  --manifest benchmarks/panel.json --output "$HOME/.cache/jope/asm-private-30-v1"
./target/debug/jope benchmark verify-inputs --manifest benchmarks/panel.json \
  --datasets "$HOME/.cache/jope/asm-private-30-v1"
./target/debug/jope benchmark methods
```

The manifest command validates the contract, not local data bytes. Preparation checks
all pinned downloads and reproduces exact development, sealed and split identities.
It refuses changed existing files. `--freeze` is a maintainer operation that refuses
an existing manifest; it is not a way to replace datasets after measuring outcomes.
Preparation receipts explicitly record zero fits and an unstarted execution budget.

The benchmark loader checks the approved manifest digest and every development file before
using its recorded folds. It opens no sealed or split files. Row order, original labels,
group membership, categorical strings and nulls survive the boundary. A lossless categorical
namespace distinguishes literal strings such as `NA` or `null` from a missing cell.

The Rust registry contains 17 regression recipes in 16 algorithm families, five binary
families and four multiclass families. Aliases do not add screening slots. The additional
regression GPR configuration uses its distinct RBF-plus-linear fitted implementation.
Default hyperparameters remain defaults, including the multiclass feature budget. Each
recipe fits median imputation, capped one-hot categories and missing indicators on training
rows only. External adapters retain their own engine precision; the controller uses f64
scores, a probability row-sum tolerance of 1e-6, and a log-loss floor of 1e-15, without
renormalizing predictions. These constants are pinned before campaign scoring.
Install the external engines with the existing locked `foundry` extra in addition to
`benchmark`: `uv sync --project python --locked --extra cpu --extra benchmark --extra foundry`.
The required Python CI lane installs both extras and executes all twelve synthetic
engine/task fits; missing engines fail that qualification.

`jope benchmark plan --manifest benchmarks/panel.json --datasets PATH --source-sha FULL_SHA
--output PLAN.json` writes a private, immutable Rust screening plan after verification.
It starts no execution clock. External versions and ASM checkpoint identities still need
to be added by campaign admission; this preliminary plan is not a complete comparison.
The low-level fit worker emits complete fitted state and proves reload prediction parity.
The Linux shared execution controller now supplies its process resource limits. Its
source implementation and live host qualification are separate evidence milestones.

### Bounded comparator execution

`deploy/prepare-panel-runtime.py` pins a checksum-verified binary/build receipt, the
Python interpreter, adapter source, installed package files and engine versions without
fitting. It requires the provisioned shared CPU lock and refuses a busy allocation.
Missing engines remain explicit entries in the comparison roster. The native command
prefix is `benchmark` for `jope`, or `panel` for `jope-bench`.

Supply that private runtime inventory to the existing plan command:

```sh
jope benchmark plan --manifest benchmarks/panel.json --datasets DATA_ROOT \
  --source-sha FULL_SOURCE_SHA --runtimes RUNTIMES.json --output PLAN.json
jope benchmark run --manifest benchmarks/panel.json --datasets DATA_ROOT \
  --plan PLAN.json --plan-sha256 PLAN_ID --journal PRIVATE_JOURNAL
jope benchmark report --manifest benchmarks/panel.json --datasets DATA_ROOT \
  --plan PLAN.json --plan-sha256 PLAN_ID --journal PRIVATE_JOURNAL
```

`PLAN_ID` is the semantic identity returned by planning. The journal is outside any
checkout and `/tmp`. `run` launches the pinned controller as the MainPID of a bounded
systemd service; it never stops CI services. It requires a delegated cgroup-v2 host,
the reviewed runner deployment and an idle exclusive CPU allocation. There is no
process-only fallback. Controller preparation counts inside the original two-hour
window; the last five minutes are reserved for reporting. Repeating `run` resumes that
same window, verifies retained inputs and outputs, and does not repeat interrupted fits.
Jobs cancelled before launch remain eligible to start within the original deadline.

Twenty one-thread fit processes share twenty CPU slots, 64 GiB and zero swap. Dataset
admission rotates fairly. An immutable allocation policy reserves 4 GiB for controller
state and gives each of the twenty fit slots a fixed 3 GiB cap. A slot does not borrow
memory when other slots are idle. Independent watchers enforce each 30-second fit deadline,
including when controller work stalls, and drain forked or detached descendants before
reusing a slot. Logs are bounded. Resource receipts distinguish measured CPU/wall time,
cgroup memory, sampled RSS and unknown exact peak RSS. Reusing fold zero during
confirmation adds no fit cost. Reporting authenticates original commands and rescoring
inputs, preserves failed/unavailable outcomes, and starts no execution clock.

This plan is explicitly a **comparator qualification**, not the complete ASM panel.
Neural/generic paired optimization and the final unchanged thirty-dataset demonstration need
their additional pinned inputs and measured receipts. The eight synthetic host cases in
`deploy/ci/qualify-execution.py` qualify isolation separately from real panel outcomes.

On 2026-09-23, all six cases passed on xbabe2: timeout with detached descendants,
successful and timed-out fits during a stalled controller, controller SIGKILL cleanup,
21 jobs with exactly twenty concurrently active workers, and a kernel-observed memory
limit failure. Both pool and runner refused the occupied shared lock. These are
synthetic host proofs, not panel results. The first two attempts exposed missing
delegated-controller initialization and an incorrect cgroup path comparison; both
failures and the corrected run are retained in private custody under
`asm-integration-20260923/workers-integration/host-proof-{1,2,3}-cases`.
The successful test executable SHA256 is
`bf1fe4d600564122cab48093969f4934a3d804751ed7b36b183964870fdeadd1`.

All eight cases passed in the follow-up run, including graceful SIGTERM, original
cutoff/resume and per-worker memory readback. The 16.38-second run used test executable
SHA256 `fbad494413fe22514f20503e44f4233400c8a5d8dd6396bf246d041f70ea01b7`;
its commands, original journals, source hashes and receipts are retained under
`asm-integration-20260923/neural-integration/host-proof-eight-cases` and
`neural-integration/host-proof-eight.json`. These remain synthetic host proofs.
The outer service
ends within the original total deadline; numerical work ends at the earlier cutoff,
leaving the declared reporting interval available to the controller.

### Neural execution and context qualification

The optional pipeline `qualified-context` feature includes a separate CPU context
worker and controller. It preserves the accepted component assets and transforms.
Admission requires eight retained native reference cases from two preselected training
families, followed by the fresh dataset encoding. All nine original bounded-worker
receipts and pinned inputs are checked; a supplied vector or success flag is insufficient.
The component and final-context error limit is fixed at 3e-4 before measurement.
CPU numerical parity requires a successful receipt for the exact source and worker
executable; implementation and fixture tests do not supply that receipt. Existing
native GPU entry-point holds and accepted regression checkpoint restrictions remain in force.

Bundle validation pins decimal parsing with `serde_json/float_roundtrip`. The
unchanged component manifest retains its historical grammar identifier, computed
with the original lower-precision parser; validation separately checks the grammar's
full-precision content digest. Both identities and the original metadata file digest
are required, so dependency feature unification cannot change admission.

The paired evaluator preregisters 64 neural slots plus 16 exploration slots against
80 generic slots under equal 120-second arm allowances. Encoding and proposal work
count inside the neural allowance. Canonical duplicates retain their origins and
zero additional fit cost. Five inner training folds select a recipe before its outer
development fold is scored. Every completed generated fit saves the complete fitted
model, reloads it and checks prediction parity. Incomplete or unavailable arms remain
explicit outcomes. Classification contexts use a separate experimental lineage with
label-renaming invariance; they cannot be supplied to the accepted regression policy.

Build the optional command with `cargo build --locked -p jope-cli --features asm`.
`jope optimize capabilities` reports compiled support without starting work. The
`plan`, `run` and `report` commands bind paired arms to the verified comparator plan;
reporting rechecks original training cells, selection, configuration, fitted outputs
and execution deadlines. One supervisor shares twenty fit workers across bounded
dataset waves. Classification uses a task-specific generic sampler while the accepted
neural producer remains regression-only.

`jope optimize proof-plan` prepares a separately budgeted single-regression operational
proof using an explicit resource file. Its report records genuine neural-origin and
offspring fits separately from fallback proposals. This proof does not replace the
unchanged thirty-dataset panel. Worker installation evidence binds the actual Cargo
artifact messages, source manifest, enabled features and immutable executable copies;
the CLI installation receipt alone cannot authenticate these dedicated workers.

## Experiment and evidence contracts

[experiment.rs](../crates/jope-pipeline/src/experiment.rs) owns the shared manifest and
evaluation identities. The committed schemas are generated from those Rust types;
semantic validation also checks task/metric compatibility, finite results, distinct
partitions and resource limits. Existing typed programs, fitted state, codecs and
checkpoint vocabularies remain authoritative.

An experiment binds the full source SHA, fixed panel, complete method configuration
registry, checkpoint hashes, seed, numerical policy, protocol, expected artifacts and
resource limits. An evaluation additionally binds ordered fitting/scoring row hashes,
split, recipe, evaluator, checkpoint, task, metric, class order, fold and stage. Cache
reuse requires the entire identity. Completed results require finite primary scores,
nonzero rows and hashed prediction/fitted evidence. Failure, timeout, cancellation,
unavailability and exhausted budgets remain explicit results without fabricated scores.
Unknown GPU time, RSS or currency cost remains unknown, not zero.

Installation and immutable plan preparation precede admission. The execution window
is created after host reservation, before controller input verification and numerical work.
It binds the original start and deadline to the experiment; resume cannot extend it.
The initial panel permits twenty isolated workers, at most twenty CPUs, 64 GiB RAM,
zero swap, one numerical thread per worker, and at most one RTX 3090 task. Its total
execution limit is two hours including a reserved reporting window. No paid cloud
services or overlap with the host's exclusive full-CPU CI job is permitted. These
contracts are enforced by the shared Linux worker implementation; positive host receipts
must separately establish the deployed isolation behavior.

Screen every applicable implementation on shared development fold zero. Select the
true top five frozen method recipes, then confirm them across all five development
folds, counting any additional diagnostic ASM challenger. Ranking uses smaller primary
loss (1 - AUROC for binary), ties break by canonical method identifier, and confirmation
reports the arithmetic mean of all five fold losses. An incomplete five-fold result has
no mean score. Selection binds the full screen outcome inventory and recipe identities;
stale results, missing methods and changed confirmation recipes are rejected. Repeated development use
is adaptive evidence. Sealed final partitions stay outside proposal, search, training
and motif inputs. Model-selection bias remains a limitation of adaptive evidence;
see [Cawley and Talbot (2010)](https://www.jmlr.org/papers/v11/cawley10a.html).
Report the best completed comparator under the protocol and budget, including missing
strong comparators and untuned defaults. Ten wins per category is an empirical target.

Eligible HWM winners additionally need finite complete evidence, train-side OOF
improvement over a trivial baseline, declared robustness/runtime/expanded-complexity
selection and consistent comparison identities. Raw exploratory HWM stays separate.
Suspicious candidates remain diagnostic; they do not automatically enter teaching data.

Creation accounting separates unique validated fitted predictors and reusable recipes.
Include loading, encoding, proposals, all failed and successful fits, confirmation,
final fitting, export, compilation and parity. Report wall/CPU/GPU use, memory, time to
quality and training amortization assumptions. One hour/ten million tokens is the
supplied reference scenario, not a measured LLM baseline. Currency cost is unknown
without rates; emitted code length is not billable model tokens.

## Serial acceptance still required

1. Task-correct fitting/prediction, executable Rust roster and real external adapters.
2. Twenty isolated workers, cancellation, fair admission, resume and cost receipts.
3. Verified regression neural candidates evaluated in generation zero with matched generic arms.
4. Bounded training frontend and experimental category adapters: actual optimizer updates,
   changed hashes, reload, resume, exported proposals and task-correct evaluation. One GPU-hour
   total, at most twenty minutes per category; encoder frozen; no serving promotion implied.
5. Shared fitted inference schema and Python/Rust parity, then C++17/20 and JavaScript
   coverage for measured winning families. Preserve strict numerical guarantees.
6. Pinned private Linux installation, minimal CLI source build, cold preparation and warm
   under-60-second demo receipts, bounded thirty-dataset report and allowlisted shareable exports.
7. After the measured demo: eligible typed subtree motifs, expanded complexity/runtime,
   cross-family usefulness and replay. New operators require ablation, semantic/backend
   coverage and versioned vocabulary compatibility. Objective experiments follow a working loop.

The panel and training budgets have not started. This first change establishes
contracts and verified inputs; it does not claim a completed demonstration.
