# Combined S3 learning, targeted improvement and data-to-code demonstration

Planning snapshot: 2026-09-24. This is an execution design, not a result or a new compute authorization. “Targeted trading” is provisionally interpreted as targeted **training**. If financial trading was intended, temporal evaluation and transaction-cost-aware objectives need a separately versioned protocol before execution.

## Latest scope: regression, one primary split

The owner's 2026-09-24 correction supersedes the earlier multi-fold plan: **regression only, one fixed primary fitting/scoring split per dataset, no other split runs, no nested audit folds or inner cross-validation.** Use one matched search seed (0) for the first demonstration. Repeated focused learning cycles reuse this split; they do not create additional splits.

For the existing regression panel, primary split means cut 0: pool development folds 1–4 into one fitting partition and use development fold 0 as the scoring partition. Do not run cuts 1–4 separately. For other S3 regression datasets, freeze one existing appropriate primary split or one deterministic group-preserving split if absent; retain its exact row IDs and hash. Exclude non-regression task categories from execution.

The first result uses the existing ten regression-panel datasets, giving **20 parent/challenger searches** (10 datasets × 1 split × 1 seed × 2 arms), rather than the earlier 1,800-search proposal. Report broad corpus training coverage separately against the full eligible regression inventory; ten evaluated datasets do not establish downstream performance on all 2,611 catalog entries. Do not add a new untouched-family panel as a prerequisite for this first descriptive demonstration.

## Outcome and scope

Produce one evidence-backed publication with three linked experiments:

1. Broad regression learning: restore the latest compatible canonical training head, consume real eligible regression teaching data derived from S3 datasets, perform actual GPU optimizer updates, export, reload and continue.
2. Focused improvement: repeatedly search one preregistered dataset, admit eligible new programs, train a challenger on those targets, and measure whether the challenger improves the next bounded search relative to the frozen parent.
3. Inference latency: run the already specified raw-fitting-data → encoders → generation/ranking → original selected program → Python/Rust benchmark. Preserve its accepted step-60,000 baseline and original 100-warm/10-cold protocol. A successor checkpoint comparison is an additional experiment, never a silent replacement of the baseline.

The demonstration KPI comes from matched downstream regression performance on the primary scoring partition. Targeted experiments supply a specialization KPI and the teachers for a potential broader improvement. Label the aggregate **primary-split regression lift**, with its evaluated dataset count and exposure history; it is not the existing nested-transfer KPI. Improvements are outcomes to test, not promised acceptance results.

## Fresh observations and remaining gaps

Read-only checks on 2026-09-24 established:

- AWS profile `veox-jepson-prod` resolves to account `714044927654`.
- The current `s3://veox-jopedime-datasets-use2/catalog/v1/catalog.jsonl` has SHA-256 `ab9fda8d2dea46067b70a42812e9d3c1d9dc6ba025780100df34377e81aa1120`: 3,859 entries, including **2,611 regression**, 843 binary, 322 multi-label and 83 time-series. Entries are not a verified count of independent families, eligible datasets or successful training exposures.
- A current root-prefix listing of `veox-jopedime-learning-use2` exposes only `staging/`, with no root objects. A canonical “latest training head” is therefore not demonstrated by that bucket layout.
- Model bundle metadata lists `policy-step-60000`, `hyperion-v7` and `jope-prime-gen31-v2`. Object modification time is not checkpoint eligibility or lineage authority.
- The retained origin-proof configuration's checkpoint receipt, accepted policy weights, vocabulary, transforms and component assets matched their hashes on readback.
- Current `jope train` routes to `python/src/jope/policy/training_campaign.py`; real-data settings are width 256, two layers. This is a distinct experimental lineage, not the accepted native width-512, six-layer policy.
- Native `train_with_policy` exists, but `train_controlled` calls the original GPU grant verifier. A documented command is not evidence that current operational admission succeeds. A new bounded canonical route must be implemented and qualified without bypassing the existing hold.
- PR90 provides a separate nested-transfer reporting contract. That contract is deferred for this demonstration. Reuse the existing panel’s ten regression datasets for primary-split evaluation, clearly labelled adaptive development rather than untouched transfer.
- PR91 merged at `f03b1eb8575d5bcc7ff509e3613ceb7ede5abdae`; exact-head CI, independent approval and merged `main-guard` passed. CUDA inference work is in progress on `feat/gpu-data-to-code-benchmark`; compilation checks have passed, but no GPU qualification or measurement window has started.

Refresh these observations at each numerical freeze. Keep source implementation, local proof, exact-head CI, live qualification and serving promotion as separate status columns.

## 1. Freeze the experiment and the denominators

Create a durable campaign directory outside checkouts and `/tmp`, with a content-addressed manifest recording source and executable hashes, dataset and family inventory, object versions and hashes, row partitions, encoder/checkpoint identities, seeds, metrics, candidate/evaluation limits, resource caps and deadlines.

Define “all regression data” as **all eligible training regression entries in the frozen S3 inventory**. Account for every one of the 2,611 catalog entries: included, duplicate/alias, unsupported, inaccessible, invalid, reserved evaluation family, or other explicitly justified exclusion. No silently disappearing failures. Publish both the full inventory denominator and the smaller admitted-training denominator. A successful subset is not “all regression data trained.”

Freeze each primary split before new search or training. Preserve original labels, values, missing cells and categories. Keep the primary scoring rows and all sealed meta-test inputs outside the learning worker closure. Record historical exposure; previously used development data remains exposed. No new evaluation-family acquisition is required for this initial demonstration.

Record separate coverage counts for: inventoried, fetched, payload-verified, partitioned, context-encoded, teacher-admitted, presented to the optimizer, included in committed updates, and downstream-evaluated. Membership in a corpus alone is not exposure. Prefer an auditable coverage pass before family-balanced replay; do not claim every admitted dataset was consumed until the committed-update exposure ledger establishes that fact.

## 2. Resolve and restore the correct checkpoint

Implement an explicit canonical checkpoint resolver with separate `training-head` and `serving/current` references. Freeze the resolved checkpoint at campaign start; it must not change during a run merely because a newer object appears.

The resolver must verify lineage, architecture, tensor inventory, parameter digests, optimizer moments and step, scheduler, RNG, sampler cursor, corpus identity, objective, vocabulary, context transforms, encoder identity and execution partition. It must distinguish:

| Operation | Required behavior |
|---|---|
| Exact interrupted resume | Same run, corpus, optimizer, sampler, counters and original deadline; reject incompatibility before any update |
| New-corpus continuation | Explicit parent plus a verified optimizer carry/reset and sampler-transition policy; new corpus identity is recorded |
| Weight warm start | Weights only, fresh optimizer, honestly named and recorded |
| New initialization | Separate experimental lineage; never presented as continuation of step-60,000 |

Start from the latest **verified compatible training head**, not the largest step or latest S3 timestamp. Inspect the retained step-60,100 state as a candidate, but do not assume it is accepted serving or resumable under today's execution partition. Step-60,000 remains the serving/inference baseline. Its missing historical resume fields cannot be fabricated.

Restore named, hash-pinned staging assets to local custody without relabeling staging as canonical. Establish a canonical candidate layout and pointer only through a reviewed migration with full object closure, readback and conditional pointer updates. Initial runs can use the frozen restored manifest while that migration is completed; the PDF must then say so. Do not claim automated latest-head S3 operation until a clean restore through that interface succeeds.

Acceptance: missing/corrupt object, wrong architecture, incompatible optimizer, changed corpus on exact resume, stale pointer and competing pointer update are explicit negative tests. Interrupted and uninterrupted short runs must agree under the declared deterministic policy in model/optimizer/scheduler/RNG/sampler state. A fallback to a fresh optimizer fails an exact-resume test.

## 3. Build real teaching inputs across the eligible corpus

A raw dataset is not itself a GP teaching example. Join each admitted target to its dataset/cut context, executable program, comparison identity, scores and provenance.

For every eligible training dataset:

1. Fetch exact objects through the dataset API and verify content hashes; retain cache/download observations separately.
2. Materialize only the primary fitting partition, construct frozen context encodings and record actual sampling sizes and devices; perform no additional cut or fold runs.
3. Resolve existing teachers and authenticate their evaluation receipts under the current admission contract. Historical corpus membership does not retroactively establish eligibility.
4. Where no compatible teacher exists, schedule bounded real proposal/search/fit work. Preserve failures and unsupported cases; do not synthesize scores or skip them from coverage.
5. Admit only valid comparisons under a separately versioned primary-split demonstration contract; freeze immutable teaching shards and an index of every dataset/family and target identity. Existing five-fold-qualified teacher gates must not be satisfied with fabricated or missing folds. Add an explicit experimental admission path for fitting-only teachers, with no serving promotion authority, rather than silently weakening those gates.
6. Record new discoveries separately from inherited initial-population programs. A better carried incumbent is not a new discovery.

Acceptance: mutation of primary scoring labels cannot influence generated programs, selection or training before evaluation; learning/search workers receive no scoring-label or sealed-data paths. Candidate search uses fitting-partition objectives only. Such objectives are labelled training/development fitness; no hidden inner-fold search is introduced. Rejected teachers stay in an audit ledger with reasons. Freeze the corpus before the corresponding training branch starts.

## 4. Demonstrate broad canonical learning

Use one RTX 3090, with an explicitly admitted CPU/memory budget and no overlapping inference or training GPU consumer. Preserve the frozen encoders for the initial experiment. Train the compatible canonical policy; the experimental 256×2 route may remain a separately labelled auxiliary experiment but cannot satisfy this requirement.

Before a long run, qualify model loading, a short positive-update run, changed parameters, finite losses/gradients, checkpoint serialization, exact reload and interrupted resume. Record per-update loss components, batch and family composition, committed dataset exposure, parameter/optimizer step counts, throughput, GPU memory and utilization, CPU/memory use and elapsed time.

The broad run must produce:

- A named parent and a real successor with positive optimizer updates and different parameter hashes.
- An exposure ledger proving the scope actually consumed, including fresh versus replayed programs.
- A verified exported checkpoint loadable by the intended proposal/ranking consumer.
- A resumed continuation from the exported training state without resetting its original deadline.
- Durable checkpoint closure and an independent restore/readback check from the intended storage route.

This establishes an operational training loop. Useful improvement additionally requires the downstream evaluation in section 6. Flat or worse results are publishable findings, not operational passes rebranded as KPI wins.

## 5. Demonstrate focused dataset improvement

Preregister the dataset, primary split, seed, comparison identities and compute limits before looking at new outcomes. `230_machine_cpu` is a practical already-exposed regression candidate with retained origin-proof evidence, but is not an unseen test and must not be picked because of a favorable new result. A user-selected dataset can replace it before freezing.

Target three short discovery/training/search cycles with one fixed seed (0), subject to the bounded pilot-derived budget. Report actual completed cycles if time expires. Each cycle:

1. Run matched bounded searches using the frozen parent and a fresh-teacher challenger on the same primary fitting partition. Replay-only and generic controls are deferred from the first time-bounded demonstration; do not claim isolation of fresh-data causality without them.
2. Authenticate each arm's initial population and maintain separate arm state so one arm's discovery does not seed another.
3. Preserve every candidate evaluation, incumbent replacement, failure, resource observation and elapsed time.
4. Admit genuinely new valid targets from the teaching arm, record their full comparison identities and create a new immutable corpus version.
5. Initialize the challenger from the verified parent model and declared optimizer state. Freeze its training/replay mixing policy, update limit and exposure accounting in advance. The unchanged parent provides the comparison baseline; equal search budgets apply to both.
6. Export/reload successors, then run the next matched downstream searches.

Report two separate focused metrics:

- **Target ratchet:** raw incumbent loss/score under the same `task`, dataset hash, cut, evaluator, vocabulary and protocol; for loss, also `(initial loss − current loss) / initial loss` when the denominator is positive. Plot actual discovery times and plateaus. An HWM is monotone by acceptance rule; that does not imply generalization is monotone.
- **Learned search benefit:** parent-versus-challenger primary-split performance under equal search budgets and the same seed. Include all failures and negative effects. This first comparison measures observed benefit; it does not isolate training-data effects from other changes through extra control arms.

Only fitting-partition objectives guide repeated targeting. Freeze the final checkpoint and selected programs before scoring the primary scoring partition. Record per-cycle fitting-side GP curves; score frozen intermediate checkpoints only after the final cycle if a scoring trajectory is needed, and do not feed those results back into this campaign. This does not erase historical exposure of the existing panel. Other task categories, including time-series, are outside this phase.

Acceptance for a closed loop: a traceable chain of new discovery → admitted teacher → sampled committed updates → changed compatible checkpoint → subsequent matched fitted search. A claimed benefit additionally requires measured favorable downstream results. Two passes through the same corpus count as continuation, not two discovery generations.

## 6. Primary-split regression KPI

Implement a separately named receipt/report contract, `jope.regression-primary-split/v1`. The current nested `transfer-plan`/`transfer-report` requires other evidence and must not be used to stamp a single-split result as that KPI. The old nested protocol is deferred, not partially executed or silently redefined.

For each dataset:

1. Freeze one fitting/scoring split, seed 0, the parent and final challenger, and identical per-arm candidate/evaluation/time limits.
2. Run one full search per arm using only the fitting partition, including preprocessing, context construction, proposal/ranking and fitting-based selection. Do not run nested folds, audit searches or other cuts.
3. Freeze selected program identities and scoring predictions before reading scoring labels. Compute RMSE on the same primary scoring rows for both arms. The baseline is a constant fitted from the fitting labels only.
4. Retain predictions, absolute RMSE, baseline RMSE, improvement, source/checkpoint identities, budget consumption and failures. Publish the observed denominator and all planned outcomes.

```text
normalized_gain_d = (RMSE_parent_d − RMSE_challenger_d) / RMSE_fit_mean_baseline_d
primary_split_regression_lift = 100 × mean_datasets(normalized_gain_d)
```

Use equal dataset weights. Also report per-dataset RMSE reduction percentages when parent RMSE is positive, parent/challenger absolute skill relative to the baseline, win/tie/loss counts, worst-dataset degradation and coverage. Zero/nonfinite denominators or missing pairs prevent a complete aggregate; report their reasons and any clearly labelled partial summaries. Preserve negative gains. A one-split/one-seed result is descriptive; do not attach multi-fold uncertainty or unseen-family/causal claims to it.

The focused dataset's primary-split result supplies its specialization result. Across the first ten regression datasets, the same matched comparisons supply the demonstration aggregate. This connects targeted training to actual downstream performance without making the deferred nested-transfer claim.

Work estimate: `datasets × 2 arms × 1 split × 1 seed` complete searches. The initial ten-dataset comparison therefore requires **20 searches**. If K candidates are fitted once per search, the base fitting work is `20 × K`, plus explicitly capped additional evolutionary evaluations/refits; there is no five-fold multiplier. Set the candidate cap, maximum evaluations and per-arm time before execution, using pilot costs. A future full 2,611-entry comparison would require 5,222 searches before exclusions and is not the first-panel runtime claim.

## 7. Preserve the inference benchmark as a separate measurement

Use only the primary split for the ten regression datasets: cut 0 fitting rows, accepted step-60,000 policy, CUDA component requirements, standard proposal settings, two warmups, ten seeds per dataset, rotating order, ten fresh-process observations, separate stage profiling and original emitter selection rules.

The ten timing seeds are repeated requests on that same primary fitting partition, not additional dataset splits.

The one-hour window starts with GPU qualification after compilation and preparation; compute ends at 55 minutes. Resume preserves that deadline. No training is charged to this window. Qualification requires eight native context cases and CPU/CUDA policy/reward/cache comparisons within the frozen 3e-4 tolerance. No silent CPU fallback.

Only 100/100 successful measured warm requests plus qualification allow the complete-path headline. An unsupported or fallback-origin top selection remains failed. Emit and validate the same original selected program. Source compilation is outside latency timing. The assumed 3,600-second LLM workflow is labelled as an assumption, with no matched-quality or equivalent-token claim.

A later successor latency comparison must name its own checkpoint, qualification and coverage. Faster inference cannot substitute for better GP performance, and better GP performance cannot substitute for this latency measurement.

## 8. PDF structure and figures

Integrate into the existing manual through its owning publication tools, preserving dated historical chapters, anchors and provenance. Give readers a short outcome table before the detailed sections.

| PDF section | Real figures/data | Required commands and evidence |
|---|---|---|
| S3 corpus and checkpoint restoration | Coverage waterfall from 2,611 catalog entries to committed exposure; checkpoint lineage diagram | Inventory, restore, hashes, eligibility, pointer readback |
| Broad regression training | Unsmoothed per-update loss plus clearly labelled smoothing; dataset/family exposure; throughput/memory; resume boundary | Exact invocation, source/binary/env, parent/successor hashes, optimizer/update proofs |
| Focused dataset learning | Stepwise incumbent fitting loss versus elapsed time/evaluations; per-cycle checkpoint markers; fixed-seed parent/challenger comparison | Frozen dataset/primary split/seed, matched resources, discovery/admission/exposure joins |
| Primary-split regression improvement | Parent-versus-challenger RMSE; per-dataset gains; win/tie/loss and coverage; worst-dataset result | One split per dataset, prediction freeze, receipt-only descriptive report |
| Data-to-code latency | Zero-based linear 3,600-second reference comparison, direct labels/inset; separate stage bars; per-dataset warm and individual cold observations | Inference plan/run/report and CUDA qualification |
| Reproduction and limitations | Full command transcript, artifact hashes, resource/deadline table and status matrix | Clean restore/replay instructions, expected outputs and failure behavior |

Every numerical table and figure is generated from retained receipts by a checked-in script. Export SVG/PDF/PNG and sanitized CSV/JSON. No invented interpolation through gaps, no omitted failures, and no illustrative curve presented as experimental data. Label historical and newly executed evidence separately. No performance number is filled in before the required measurements exist.

## 9. Command examples and interface status

The following read-only inventory commands are available now:

```sh
aws --profile veox-jepson-prod sts get-caller-identity
aws --profile veox-jepson-prod s3api list-objects-v2 \
  --bucket veox-jopedime-learning-use2 --delimiter /
aws --profile veox-jepson-prod s3api list-objects-v2 \
  --bucket veox-jopedime-models-use2 --prefix bundles/
```

The existing experimental trainer interface is real but **does not establish canonical 512×6 continuation**:

```sh
jope train plan --state "$RUN" --python "$PYTHON" --inputs "$INPUTS" \
  --task regression --device cuda:0 --source-sha "$SOURCE_SHA" \
  --learning-state "$LEARNING_STATE" --preparation-receipt "$PREPARATION" \
  --seed 7 --passes 2
jope train run --state "$RUN"
jope train resume --state "$RUN"
jope train verify --state "$RUN"
jope train export --state "$RUN" --output "$EXPORT"
```

These examples require authenticated real preparation inputs and an admitted compute window. The canonical continuation front door remains implementation/qualification work; its final command will be documented from the reviewed binary's help and an actual successful invocation, rather than presenting a proposed command as runnable today.

Existing nested-transfer reporting commands are deferred and are **not commands for this primary-split demonstration**:

```sh
jope benchmark transfer-plan --input "$SPEC" --output "$PLAN"
jope benchmark transfer-report --plan "$PLAN" --freeze "$FREEZE" \
  --release "$RELEASE" --release-sha256 "$TRUSTED_RELEASE_SHA" --output "$REPORT"
```

The primary-split search/teacher/report interface is implementation work; document its actual reviewed CLI and a successful invocation before presenting it as runnable. Do not route this demonstration through an executor that automatically expands to five folds.

The inference commands are being implemented in the current branch and are **not yet reviewed or GPU-qualified**:

```sh
jope benchmark inference-plan --manifest "$PANEL" --datasets "$DATASETS" \
  --configuration "$CONFIG" --configuration-sha256 "$CONFIG_SHA" \
  --references "$REFERENCES" --references-sha256 "$REFERENCES_SHA" \
  --source-sha "$SOURCE_SHA" --output "$PLAN"
jope benchmark inference-run --plan "$PLAN" --plan-sha256 "$PLAN_SHA" --journal "$JOURNAL"
jope benchmark inference-report --plan "$PLAN" --plan-sha256 "$PLAN_SHA" \
  --journal "$JOURNAL" --output "$FIGURES"
```

The final PDF must replace placeholders with a real sanitized transcript and point to each concrete plan, command exit, run receipt and output digest. Planning commands and metadata checks never stand in for numerical execution.

## 10. Delivery order, budgets and acceptance gates

Use one implementation PR at a time, with full exact-head CI and a genuinely independent native approval, then guarded merge and remote-main verification. Keep active checkout ownership synchronized.

1. Finish the current inference implementation, timer/device/asset/cache/failure/deadline/report tests, and its existing authorized measurement or explicit failure report.
2. Deliver canonical S3 restore/head-resolution and strict compatible native continuation under bounded supervision; qualify short real-data update/reload/resume behavior.
3. Deliver the regression-only corpus inventory/admission/exposure pipeline and focused discovery→teaching→training controller for one primary split and one seed.
4. Deliver and audit the primary-split parent/challenger runner and descriptive reporter. Exclude nested-transfer execution and other task categories from this phase.
5. Freeze numerical manifests, approve measured resource estimates, execute the broad/focused/evaluation stages serially on the GPU, and preserve every partial outcome.
6. Generate and verify the PDF and figures from receipts; perform source/data fact checks, link/anchor checks, visual page review and independent publication review.

Proposed additional allowances, **not authorized by this document**:

| Stage | Initial proposal | Stop/extension rule |
|---|---|---|
| Restore and corpus preparation | Read-only metadata first; estimate bytes and context/teacher work before bulk execution | No blanket mirror of hundreds of GB; explicitly pin required closure |
| Canonical GPU operational qualification | Up to 45 minutes, one RTX3090 | Must prove updates/reload/resume before a longer campaign |
| Real cost pilot | Up to 15 minutes, representative preregistered sizes | Publish throughput/fit/context costs and projection; cannot qualify all-data coverage |
| Broad real training | Propose an initial two-hour tranche after the pilot | Full exposure gate determines whether “all eligible data” is achieved; retain partial counts otherwise |
| Focused learning | Pilot-bounded short cycles, one primary split and seed 0; replace the earlier multi-seed two-hour proposal with a measured cap | Retain completed cycles and failures; no seed sweep or automatic extension |
| Primary-split evaluation | First ten regression datasets, 20 matched parent/challenger searches; pilot determines the cap | One split and one seed; all failures retained, no nested-search expansion |
| Inference | Existing separate one-hour window, compute cutoff 55 minutes | No reset or reuse for training |

Use a separately recorded CPU/RAM cap for each added phase; initial recommendation is four CPU cores and 16 GiB for qualification, with pilot evidence determining whether corpus/search stages need a different explicit grant. No automatic extension, overlapping GPU campaigns, training-worker AWS writer credentials, or serving promotion is implied.

Completion is a matrix: operational continuation, corpus coverage, focused target improvement, learned downstream benefit, primary-split regression lift and latency each receive their own passed/failed/incomplete result. The combined PDF is complete when it reports the actual evidence honestly; the requested improvement claim is achieved only when its corresponding preregistered metric passes.
