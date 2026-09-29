# JopeDime Lambda GP flywheel: source review and next steps

Reviewed 2026-09-28 at /home/ubuntu/JopeDime HEAD 4d9a27e64d31c57b6573a84e35f22c69ebb2b1b6. This design is not a claim of canonical teaching admission. The campaign's actual manifest and receipts govern what executed.

## Highest-value output

The valuable result is a reproducible measurement dataset: which programs were proposed for each exact training cut, what each actually did, which won under a frozen objective, how much it cost, and why other attempts failed or stopped. Cheap invocations are engineering evidence; qualified program discoveries that improve the next policy are the product objective.

MASTER_PLAN.md:336 defines the chain:

verified training inputs -> encoders -> neural GP proposals plus GP exploration -> measured programs and bound emitted code -> durable S3 targets and comparisons -> immutable teaching corpus -> policy training -> export and qualification -> next bounded GP generation.

ADR-0005 selects lineage B as canonical. Accepted serving policy, experimental challengers and lineage-A observations remain distinct. Old unbound results cannot become qualified by being copied into a new prefix. Preserve evidence now; future qualification must be a newly preregistered computation with complete provenance.

## Read-first implementation map

1. AGENTS.md, MASTER_PLAN.md Product target: input boundaries, claims and proof.
2. docs/adr/ADR-0005-canonical-learning-loop.md: lineage.
3. docs/learning.md: immutable blobs, ledger seals, mirror ordering, pointer CAS and recovery.
4. docs/experimental-learning-preparation.md: training closure and fresh/replay sibling comparison.
5. crates/jope-engine/src/task_contract.rs::require_task and score_predictions.
6. crates/jope-engine/src/prepare.rs::folds_with_seed, Preprocessing, run_with_admission and load_bound_view_in_root.
7. crates/feat-math/src/evolve.rs, model.rs and feasibility.rs.
8. crates/jope-engine/src/gp.rs and gp/selection_replay.rs.
9. crates/jope-policy/src/seeds.rs and proposal_ledger.rs.
10. crates/jope-learning/src/eligibility.rs, eligibility/inner.rs, hwm.rs and record.rs.
11. crates/jope-engine/src/training/{inputs,continuation,objective,consumption,checkpoint_commit}.rs.

This campaign writes only the ignored experiment path and a new S3 experiment prefix. Do not modify product files, canonical HWM/current pointers, serving state, historical prefixes, datasets or other agents' worktrees. Eventual product integration needs a separate reviewed adapter.

## What can be qualified today

A GP event generates/mutates programs, fits their actual pipelines, selects using measured inner-validation performance and records lineage. A fixed histgbm2 fit is not GP search. A tiny bounded evolution with a restricted grammar is an honest experimental GP event when its budget and limits are explicit.

The lightweight runtime may use GenomeIr, feat_math::fit_table, BoundModel::predict_table, random_canonical_pipeline, mutate_canonical_pipeline and crossover_canonical_pipelines. This is a new experimental evaluator/protocol, not an invocation of the full product authority path.

Product gp::search_seeded_with_rng_in_root requires immutable preparation, frozen family roles, explicit evaluator binding, fit authority, candidate intent/completion, budgets and selector replay. Teacher admission additionally requires preregistered comparison, qualified context, train-side OOF predictions, fitted/runtime identities and prescribed baseline improvement. One 60/20/20 event cannot satisfy that five-fold admission.

Current task_contract.rs:388 recognizes four task names but permits only regression for parallel execution and teaching. The campaign's separately tested advanced_v3 classification path can be experimental; preserve original binary/multiclass labels and class order and do not call it a qualified product task. Never turn classification into regression merely because labels are numeric.

Results should declare benchmark_valid=false, canonical_admission=false and neural_proposals=false unless those capabilities actually ran. Exact local/Lambda replay proves the specified experimental protocol only.

## Coverage and leakage boundaries

"All datasets" means a frozen exhaustive catalog and an explicit disposition per entry. It does not authorize reading every object. Bind dataset ID, original task, family and meta partition, train object/version/hash, fold membership, preprocessing and protocol. Meta-test families are forbidden to search even under product qualification. Meta-validation observations remain development-only. Unknown family authority cannot enter automatic teaching.

Only exact training blobs belong in the Lambda read grant. Do not grant outer-test or sealed object access. Distinguish consumed normalized CSV bytes from independently reconstructed raw-training bytes; historical globally normalized data cannot support a clean held-out claim.

Per entry record success, sealed exclusion, unsupported task/protocol, missing/inconsistent identity, malformed input, insufficient groups/classes/history, resource refusal, timeout, numerical failure or unknown outcome. No silent omissions. Preserve dataset IDs separately from duplicate content hashes so coverage and actual fit counts stay meaningful.

For the declared 60/20/20 experiment, feature selection, imputation and model fitting must use the 60% fitting subset only. Inner validation may select the winner. Audit labels are read for scoring only after the winner is fixed; no audit-driven tuning, retry selection or early stop. Save the original feature indices/names and fitted statistics for top-64 variance projection; projected models need this artifact to consume raw rows again.

Group feature-identical rows and known entity IDs before iid splits, including duplicates with conflicting labels. Time series needs chronological origins and a train-derived scale, never random shuffling. Classification needs fitting class coverage; unknown classes fail explicitly. A simple hash-by-row split without group checks is an engineering protocol, not the product's group-qualified protocol.

## Actual GP execution and replay

Freeze candidate count, vocabulary, seed streams, parent selection, mutation distribution, objective, tie order and stop rule. Include a train-only trivial baseline. Fit the actual head named by each candidate; retain canonical DSL and parent hashes.

Important trap: EvolveConfig defaults to FitnessMode::Surrogate, replacing heads with ridge. Either label these as surrogate screens and record a separate actual-head evaluation, or fit actual heads throughout. A surrogate score is not the declared candidate's true-head measurement.

Retain every candidate/refusal/failure, not only winners:
- order, generation, parents, origin, mutation kind and RNG seeds;
- DSL/program hash, feasibility, requested and completed fidelity;
- actual fit/predict time, finite metrics and prediction hash;
- winner model plus preprocessing/feature selection, row membership, final audit predictions;
- baseline metrics, selector contract, completed/requested evaluations and stop reason.

Resolve ties by preregistered order, not completion order. Constant-target R2 is undefined, not silently zero; RMSE remains valid. Reject nonfinite values before JSON encoding. Independently recompute winner selection and metrics from durable candidate/prediction rows.

## Smart durable storage

Keep a new experiments/lambda-gp/v1 campaign namespace in the learning bucket. Do not touch canonical pointers. Recommended logical layout:

campaigns/<campaign-id>/manifest.json
campaigns/<campaign-id>/inventory.jsonl.gz
campaigns/<campaign-id>/jobs.jsonl.gz
campaigns/<campaign-id>/source/source-manifest.json
campaigns/<campaign-id>/source/source-bundle.tar.gz
campaigns/<campaign-id>/attempts/<shard>/<invocation-id>/result.json.gz
campaigns/<campaign-id>/receipts/<shard>/<invocation-id>.json
campaigns/<campaign-id>/summaries/<sha256>.json
campaigns/<campaign-id>/exports/<sha256>.jsonl.gz
campaigns/<campaign-id>/commits/<sha256>.json

Adapt actual names to the frozen storage schema. Index task/dataset/cut in compact JSONL initially, later Parquet, rather than millions of mutable per-candidate pointers. Reference input CSVs and weights by immutable hash/version; avoid repeated copies.

A batch of 2-5 jobs can use one compressed result object containing candidate rows and bounded winner artifacts. Large models become separate content-addressed objects. Count every actual PUT, including manifests/models/receipts; "one write per batch" must reflect actual calls. Do not remove failure/lineage information for compression.

Use If-None-Match: * on immutable writes. Identical existing bytes mean idempotent replay; different bytes mean collision/refusal. Freeze serialized bytes before retries so timestamps/ordering cannot change. Separate logical job identity from attempt identity; a retried fit is not another unique teaching example.

Publish payloads -> verify readbacks -> publish summary/commit referencing the complete closure. Never label a missing result complete. Commit an honest partial campaign when necessary. Retain the commit digest outside the S3 prefix as recovery anchor. Versioning/Object Lock configuration alone does not demonstrate payload recovery.

Mirror exact relative paths to external scratch/custody, with SHA256SUMS and a restore verifier. Download to a fresh directory and independently validate the full closure. Local SQLite/DuckDB indexes are regenerable views, never the only surviving learning state again.

## Identity schema

Campaign: source SHA and source-bundle digest, runtime ZIP digest, architecture/region, pricing snapshot, catalog/jobs/protocol hashes, deadline, spend/call/concurrency caps, function version, role policy and completion/restore digests.

Job: dataset/family/task and original role; training object/hash/version; cut membership; preprocessing/vocabulary/evaluator/protocol; seed and search fidelity; expected runtime; resource tier.

Attempt/invocation: stable job IDs plus unique attempt/request IDs; function/config; cold/warm/cache facts; phase timings; CPU when measured; memory; billed duration; GET/PUT counts and bytes; worst-case reservation/settled cost; errors and durable result reference.

Candidate: parents, DSL/program identity, original proposal origin, actual-head versus surrogate, fidelity, feasibility, metrics and prediction signature, timings and failure.

Winner: exact comparison identity, selected candidate, selection contract, row/prediction/model hashes, replay receipt, baseline delta and admission flags.

Never compare only by dataset name. HWM identity includes task, dataset_hash, cut_id, evaluator_sha256, vocabulary_sha256 and protocol. Changed caps, folds, preprocessing, grammar or fidelity require explicit identities.

## Cost and concurrency controls

AWS Budgets is delayed accounting, not a synchronous hard stop. The campaign needs durable synchronized worst-case reservations before every invocation, including in-flight and retry liability. Unknown/disconnected outcomes retain reservations. Root's conservative never-refund reservation design is safe even when actual charges are much lower.

Keep original deadline, immutable maximum calls, frozen memory/timeout and reserved concurrency. Resume the same ledger after crashes. Synchronous invocations with explicit attempts are easiest to account for; uncontrolled async retries/triggers are not. A local campaign cap cannot cap unrelated functions/principals/regions in the entire AWS account. Report account budget versus campaign admission separately.

Concurrency improves throughput, not necessarily cost per CPU-second. Pack by predicted time/size, not count alone. Large tasks should not erase four completed siblings at timeout. Freeze available quota and other-function demand rather than consuming all account capacity blindly. At low-memory tiers, uncontrolled Rayon/BLAS threads can increase cost.

## Historical sizing

Pre-execution features: bytes, rows, raw width, task, permitted dtype/cardinality/missingness sketch, grammar/operators, predicted expanded width, head family, parameters, split sizes, candidate budget and cache state. Bytes alone cannot predict cost of linear algebra versus tree/kernel fits.

feat_math::feasibility::classify bounds live f64 cells, not all allocator/runtime memory or CPU time. Unknown does not authorize unbounded execution. Keep headroom and record OOM, timeout/refusal and numerical failures, including censored duration observations.

Measure phase wall/CPU time, per-job process memory where possible and invocation AWS high-water memory separately. Warm-container Max Memory Used is not an exact per-job allocation. Capture init, download, parse, search, fit, serialization and PUT.

Start with conservative shape tiers and a fixed calibration set. Later fit upper-quantile memory/runtime predictors with censored failures, evaluate on held-out jobs/families, and choose the cheapest tier likely to finish with headroom. Log predictor version, inputs, estimate and chosen tier before execution. Never use outcome measurements to pretend a prediction was correct.

Track valid GP events per dollar, candidates per second, quality gain per dollar, time to first verified winner and exact replay rate. Repeatedly fitting the easiest tiny task is not a useful learning flywheel.

## Next product stages

1. Finish the engineering sweep: exact coverage, durable S3 closure, cost ledger and fresh restore.
2. Create a reviewed product adapter with qualified training-only contexts and preregistered comparison/admission.
3. Compare neural proposals to grammar/random proposals on identical cuts/budgets; retain fallback origins and all draw denominators.
4. Replay winners and required OOF measurements; admit only records passing jope-learning eligibility/HWM rules.
5. Freeze the teaching corpus; perform actual native optimizer updates with exact continuation state.
6. Compare fresh-plus-replay and replay-only siblings from the same parent/resources. Current contract specifies 25% fresh/75% replay.
7. Export and qualify through the bound promotion protocol; only then change serving and run the next bounded generation.

This campaign supplies the measurement and execution substrate. It does not claim later learning/promotion from fast Lambda calls.

## Review gates and handoff

Before sweep: frozen identities, no secrets in ZIP/Git, actual evolution replay, train/audit boundaries, task preservation, resource ceilings, finite metric/selector checks, concurrent/crash budget tests, immutable duplicate/collision tests, scoped IAM readback and successful S3 pilot restore.

After sweep: reconcile every catalog row, preserve raw request/REPORT receipts and source, account for all unknown/in-flight costs, read back/hash every result, restore complete closure, confirm no canonical/dataset mutation, disable/delete temporary execution resources, and report GP searches versus candidate fits, skips, failures, costs and remaining qualification separately.

## Evidence-driven next routing policy from this campaign

The completed experiment provides 3,981 full twelve-fit/eight-offspring diagnostic GP outcomes and six explicit class-support refusals across 3,987 eligible inputs; fourteen excluded sources remain documented. It does not change canonical admission policy. Source-level success, audit quality, teacher eligibility and execution cost are separate fields.

For the fifteen-input small-job cost cohort, 128 MB with five independent datasets per invocation achieved about $0.000001374 Lambda-only or $0.000002854 including measured S3 request costs per event, with exact frozen learning outputs. The 512 MB batch-five cohort was about 4.2% dearer and 2.63 times faster. These are exploratory naturally warm/cold measurements and exclude transfer, logs, storage and host/curation/publication. Use both cost and latency rather than hard-coding the smallest memory globally.

A future router should first apply admissibility/support checks, then predict upper memory and duration bounds from versioned task/algorithm/input/projection features, and finally pack at most five compatible jobs under the batch deadline. Treat unknown families and unseen dimensional regimes as higher-uncertainty cases. Store prediction, selected tier, uncertainty, actual process/batch HWM, elapsed time, cold state, failures and eventual billing units together. Warm-environment HWM is a cumulative or censored label. Refused one-class/minority-support events should train an admissibility classifier, not a memory estimator. Preserve fixed learning seeds and output fingerprints in routing qualification, so apparent savings cannot result from silently reduced search work.
