# Algorithm Synthesis Model (ASM)

ASM **learns to generate and optimize executable algorithms from data**. This is
its product direction, not a claim that every part of the loop is qualified.
Discovery programs, search histories, training inputs and provenance remain private.
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
| Neural proposals | [policy seed binding](../crates/jope-policy/src/seeds.rs) | Accepted regression checkpoint and verified proposals exist; generation-zero product consumption is not connected. |
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

`jope benchmark plan --manifest benchmarks/panel.json --datasets PATH --source-sha FULL_SHA
--output PLAN.json` writes a private, immutable Rust screening plan after verification.
It starts no execution clock. External versions and ASM checkpoint identities still need
to be added by campaign admission; this preliminary plan is not a complete comparison.
The low-level fit worker emits complete fitted state and proves reload prediction parity.
Its process resource limits are supplied by the shared execution controller, still pending.

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

The execution window is created only after verified inputs and dependencies are ready.
It binds the original start and deadline to the experiment; resume cannot extend it.
The initial panel permits twenty isolated workers, at most twenty CPUs, 64 GiB RAM,
zero swap, one numerical thread per worker, and at most one RTX 3090 task. Its total
execution limit is two hours including a reserved reporting window. No paid cloud
services or overlap with the host's exclusive full-CPU CI job is permitted. These
contracts do not yet enforce process isolation; the worker implementation must prove it.

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
