# ASM — Maintainer and User Manual

Edition 3 · 2026-09-24 · implementation 7f2b2bf860e4

# Chapter 1 — Current status, versions, north star and measured limitations

An **Algorithm Synthesis Model (ASM) learns to generate and optimize executable algorithms from data.** JopeDime turns that objective into a bounded, inspectable workflow: validate original data and row assignments, encode a fitting-only context, propose executable recipes, fit them, measure them under a fixed protocol, evolve measured recipes, retain fitted artifacts, and eventually admit qualified examples into another round of model training. This chapter states the position on 23 September 2026 at about 22:00 UTC, at merged main `a75e543d2166` (reviewed head `68f56404c3e1`, same source tree).

**Later implementation observation — 2026-09-24.** PR88 integrated the training64, product28, warm-demonstration and winner-export packets, the mandatory regression-consumption gate, and preparation, input-closure, strict-resume and controlled-comparison changes. Reviewed source `8fcf5470cf90f8facb4fac598b67725660448864` merged as main `0fa1841834d1fc67535f021a91a1911a321e6363` at 05:11:50 UTC; exact-head CI 35957350158 attempt 1, independent native approval 5300062934 and main-guard 35958877242 passed. A source-bound runtime build and small CPU fixtures passed. No new real-teacher campaign, GPU window or serving promotion was admitted. The operational learning outcome remains inconclusive. The chapter and roadmap status labels below describe their named PR83/stopped-overlay snapshot, not later main. Use the [later implementation runbook](https://github.com/veox-ai/JopeDime/blob/8fcf5470cf90f8facb4fac598b67725660448864/docs/experimental-learning-preparation.md) for the integrated interfaces; historical commands and evidence are retained here.

<a id="c-status--asm-at-a-glance"></a>

## 1.1 ASM at a glance

This table is the bottom line on 23 September 2026 at about 22:00 UTC. Each row links to the sections that hold its evidence. Development scores are adaptive, and historical results do not establish current capability.

| Area | Current state | Evidence |
| --- | --- | --- |
| **Works** | One bounded regression proof on one frozen-panel dataset, `230_machine_cpu`: verified context → 64 policy proposals and 16 exploration proposals → five-fold fits on 20 bounded workers → 16 offspring (8 mutation, 8 crossover) → an independently recovered report. Genuine neural-origin fits and fitted descendants are verified. | [§](#c-status--summary-of-the-one-dataset-proof), [§](#c-status--canonical-proof-record-and-per-arm-facts) |
| **Not better yet** | On that dataset the neural arm (policy-seeded search) had 1.071 times the development-hold RMSE of the generic arm (grammar-uniform search): 166.3 against 155.3 on 34 held rows, under equal declared allowances but unequal fit time. In retained historical final searches, whose arms shared their seed pools, the two arms tied exactly on 3,227 of 3,268 paired comparisons (98.7%). Parity of the product's CPU policy port with the trained model is not established. | [§](#c-scheduling--arm-order-and-clocks), [§](#c-targets--paired-final-search-outcomes-within-the-admitted-development-inventory), [§](#c-status--what-works-now) |
| **Not run** | The frozen thirty-dataset panel (0 of 30 datasets); teacher admission under the new contract (0 teachers); new training (no newly trained checkpoint); consumption of a newly trained checkpoint; exact fitted-winner export; installer and warm demonstration. | [§](#c-status--pending-acceptance), [§](#c-training--what-can-be-claimed-now) |
| **Historical standing** | On the corrected, benchmark-valid historical SOTArena ladder the fixed-seed `jope-dime-regressor` entrant ranks 31 of 304 methods (Elo 1687.0; rank 1 is TabFM Ensemble at 1934.9). The published ladder's rank 1 for `jope-dime` rests on test-selected rows and is not a held-out claim. | [§](#c-targets--why-the-two-historical-ladders-differ) |
| **Next** | The critical order, in short: integrate the guarded follow-on source; run the actual tests, host qualification and packaging; produce fresh qualified contexts and admit preregistered real teachers; build the all-eligible family-disjoint corpus; one GPU-hour of training with continuation; consume the new checkpoint; run the real fitted warm demonstration; then run the unchanged thirty-dataset campaign. Under the current coordination rule, source integration waits until the manual's pull-request lifecycle closes. | [§](#c-plan--critical-order), [§](#c-plan--proposed-owner-decisions-and-external-dependencies) |
| **Risks** | Cross-region protection of the current learning objects is not established: the surveyed learning replica is empty and the `staging/` prefix is not replicated. Several large data classes have no established remote copy, and the disk was 87% used at the dated readback. | [§](#c-data-estate--recorded-copy-and-retention-gaps) |

<a id="c-status--reading-this-manual-audience-evidence-levels-and-figures"></a>

## 1.2 Reading this manual: audience, evidence levels and figures

This manual serves operators, scientists, implementation maintainers and reviewers. Operators need installation, preparation, execution, reporting and recovery procedures. Scientists need dataset identities, split boundaries, metrics, model lineage, comparison budgets and limits of the evidence. Maintainers need the workspace map, command contracts, fitted-state formats, feature boundaries and tests. Reviewers need to distinguish code that exists, behavior that has been demonstrated, changes that have merged, and claims still requiring evidence.

| Evidence level | Establishes | Does not establish |
| --- | --- | --- |
| Source implemented | The named source and command path exist at that revision | Successful execution on this host |
| Static or fixture verification | Parsing, linting, schema checks or named synthetic tests passed | Real corpus behavior, GPU execution or competitive performance |
| Bounded local qualification | Named real processes, artifacts, limits and receipts were observed | Other datasets, backends, checkpoints or protocols |
| Exact-head CI | Required current-attempt jobs and aggregate passed | Skipped tests, all optional features, GPU qualification or promotion |
| Reviewed merge | Eligible independent native review and guarded merge were verified | Installation qualification or serving promotion |
| Measured campaign | Admitted and unadmitted outcomes are accounted for under one window | Promised win rate or unbiased sealed-test performance |
| Release qualification | Exact installation, artifact and backend passed acceptance | Other runtimes, artifacts or numerical policies |

A checkpoint file, a nonempty journal, a successful exit and a genuine optimizer update are different observations. A skipped test is not a pass. A missing measurement is unknown, not zero. Cancelled, failed and unavailable fits remain outcomes rather than scores.

The corpus-size figures are metadata/development statistics, not performance. The language coverage figure counts admitted source recipes under additional state restrictions; it is not a backend success rate. The architecture diagram is explanatory, and the completion matrix is an evidence inventory; neither is an experimental result.

<a id="c-status--north-star-better-algorithms-from-accumulated-experience"></a>

## 1.3 North star: better algorithms from accumulated experience

<a id="c-status--the-proposed-north-star-and-matched-comparisons"></a>

### 1.3.1 The proposed north star and matched comparisons

The product goal is a repeatable learning cycle: verified dataset-and-split inputs become contexts; the policy proposes executable programs; bounded genetic-programming (GP) search fits and improves them; eligible measured programs become teachers; training produces a new checkpoint; and that checkpoint improves a subsequent bounded search. The north star (*proposed; owner ratification pending*) is **better predictive algorithms on previously unseen dataset families at the same declared compute budget**, with reproducible fitted artifacts; [§](#c-status--the-proposed-north-star-and-matched-comparisons) states it in full. More optimizer steps, a larger corpus, lower training loss or a higher historical Elo are supporting observations; each answers a different question.

Measure progress against both the generic search control and the previous accepted checkpoint, using the same data identities, folds, method set and resource envelope. Report task-specific loss, wins/ties/losses, failure and unavailable rates, and cost per unique fitted predictor. A trained checkpoint earns a promotion claim only through its bound qualification procedure. The frozen thirty-dataset panel is a development comparison; it cannot establish unseen-family performance when prior checkpoint exposure is unknown. Sealed evaluation remains untouched.

**ASM learns to generate and optimize executable algorithms from data.** Its north star (proposed; owner ratification pending) is better downstream predictive loss from generated, actually fitted algorithms on previously unseen dataset families, under the same compute allowance, with reproducible predictions and a usable fitted artifact. This is an operational definition for the next evaluation, not a ratified objective or a claim that the current model has already met it. The frozen-panel win count PANEL-WINS (proposed definition; owner ratification pending; [§](#c-learning--kpi-definitions)) is a separate development KPI, not the north star: the panel is adaptive development data whose prior checkpoint exposure is unknown, so it cannot establish unseen-family performance.

Compare each challenger against both a matched generic search and the previous checkpoint, with the same grammar, task, rows, class ordering, preparation, seeds, resources, scoring policy and exposure rules. When the challenger belongs to a different model family, compare it with the previous checkpoint only through such matched downstream fits, never through loss values: the pending bounded campaign trains a width-256, two-layer Python `JopePolicyModelV3` ([§](#c-training--what-the-pending-campaign-trains)), not a continuation of the accepted width-512, six-layer step-60,000 model ([§](#c-architecture--the-accepted-regression-model-lineage-b-step-60000)). A lower imitation loss alone is insufficient: the generated programs must compile or interpret, fit, survive the resource limits, and improve predictions. Conversely, an honestly measured optimizer update is training progress even when the downstream candidate does not win.

The frozen thirty-dataset panel (asm-private-30-v1; not yet executed) defines an adaptive **development** comparison whose protocol records selection and confirmation exposure. It cannot establish the previously-unseen-family part of the north star merely because a dataset appears in that panel. That requires independently frozen family/exposure evidence and a separately authorized evaluation; this manual grants no access to sealed labels. Historical exposure that cannot be reconstructed stays unknown. See [§](#c-learning--what-the-historical-training-and-target-evidence-establishes), [§](#c-targets--did-gp-targets-get-better) and [§](#c-learning--the-closure-ledger).

<a id="c-status--where-the-current-evidence-stands"></a>

### 1.3.2 Where the current evidence stands

The latest retained origin proof at source `68f56404` is a useful concrete boundary: it executed actual neural proposals, fitted generation zero, and fitted neural descendants, with an authenticated recovered report. Its neural development-hold RMSE (n = 34) was 166.3181717454848 versus generic 155.25909998168862 on one regression dataset, under equal declared arm allowances but unequal fit time; the neural arm's selected recipe descends only from random-exploration candidates, not from a `neural_policy` proposal. This demonstrates consumption, not a quality win, a quality loss or a thirty-dataset result. The fourteen private language-example commands establish recipe-refit prediction parity across five language configurations; they run no GP evolution and are not exact-state fitted-winner export. Historical trained corpora also must not be confused with the zero newly demonstrated teachers under the pending ASM admission contract. The measured origin report SHA is `6e0c20720256b0735a6969c45032f44082c92abacadf2c4eb21ef5c390f249dd`; see [§](#c-status--canonical-proof-record-and-per-arm-facts) and [§](#c-inference--measured-execution-of-the-private-examples).

There **has been real historical policy training**. The accepted serving model is the recovered lineage-B step-60000 checkpoint. Retained logs cover 63,000 optimizer steps of that lineage (steps 6,001–69,000, in 21 blocks, each on a different corpus). Loss falls inside every block and jumps upward at every corpus change. The descriptive linear fit over steps 6,001–60,000 changes by −0.018284 in total, while the tabulated block losses lie between 0.785951 and 0.855045 ([§](#c-training-history--within-block-and-boundary-patterns)). The 78 retained checkpoint observations of that lineage use changing evaluation populations. The retained fixed-monitor comparison covers three earlier checkpoints (recovery, cycle 25 and cycle 107), and there sequence and reward-ranking losses did not improve over recovery ([§](#c-training-history--fixed-monitoring-and-actual-promotion-evidence)). In retained paired final searches, whose arms shared their seed pools, the neural arm tied the generic arm exactly on 3,227 of 3,268 comparisons ([§](#c-targets--paired-final-search-outcomes-within-the-admitted-development-inventory)). Across all 26,357 historical search receipts, an external seed audit reports that 98.84% of reported winners were already in their search's seed population ([§](#c-targets--first-to-last-comparison-at-1000-evaluations)). The optimizer fitted its objective; better proposals have not been demonstrated.

Historical optimizer evidence for the serving lineage: 21 original A100-recovery logs retain all 63,000 steps from 6,001 through 69,000, with 378,000 selected loss/component/gradient values preserved exactly. All 21 blocks show declining within-block composite loss, while each uses a different corpus. This is historical fitting evidence only, separate from better downstream GP targets or proposal quality, and it implies no new training or promotion. Accepted serving remains step 60,000; later A100 step 69,000 is explicitly refused, and the separately retained world-1 step 60,100 is hardware qualification. The source-bound CSV, corpus boundaries and explicit averaging are in [§](#c-training-history--the-complete-a100-loss-series-steps-600169000).

The new ASM training path has not completed a real bounded acceptance. No eligible real corpus exists for it yet: zero teachers have been admitted under the new contract, and its two prospective teaching families, `230_machine_cpu` and `712_chscase_geyser1`, are both frozen-panel regression datasets ([§](#c-teachers--present-teacher-readiness-and-the-smallest-real-path)). Its pending campaign would train a new, smaller Python model (width 256, two layers), not a continuation of step 60,000 ([§](#c-training--what-the-pending-campaign-trains)). The native command that could continue step 60,000 is held: its GPU-grant check always refuses (`crates/jope-engine/src/task_contract.rs:88-92`; [§](#c-training--the-dispatch-hold-resource-incompatibility-and-valid-topologies)). Zero newly admitted teachers must not be read as an empty historical corpus or as evidence that JopeDime never trained: the accepted checkpoint's retained closure corpus, `snapshot-000097`, holds 10,214,332 target records labelled under the historical `reference_v1` policy ([§](#c-data-estate--six-retained-corpora)).

<a id="c-status--what-works-now"></a>

## 1.4 What works now

The frozen benchmark foundation consumes actual row assignments, original labels, declared categorical columns, missing values, groups and class order. It uses regression RMSE, binary AUROC and multiclass log loss from a complete probability matrix. [§](#c-inference--complete-registered-recipe-matrix) inventories the executable native recipes, and [§](#c-installation--native-roster-and-external-comparator-runtimes) the four external adapter families, which run at their default configurations.

The shared execution layer has twenty bounded fit slots, resource accounting, independent deadline enforcement, process-tree cleanup, durable journals and original-window resume. Its ten real Linux process/cgroup tests use synthetic workloads to exercise completion, stalled controllers, cancellation, forced process death, memory failure, concurrency and setup failure.

The regression neural path has a qualified native context bridge and a measured proposal-to-fit-to-evolution proof. Policy-origin metadata is checked against pinned producer output, completed inner-fold measurements and measured ancestry. A candidate labelled neural without those links is insufficient. Canonical duplicates retain proposal slots and origins but do not justify fitting the same executable twice.

Numerical parity of the product's policy implementation with the trained model is not established. The product runs a Candle port of the policy on the CPU in float32. The crate states that no numeric parity with the trained tch forward pass is claimed, that its logits and rewards are unverified until reference fixtures exist, and that its `cuda` feature is declared but has never been compiled (`crates/jope-policy/src/lib.rs:84-95`; `crates/jope-policy/src/model.rs:18`). Until such fixtures establish parity, a neural-versus-generic result measures this port, not necessarily the trained model.

Observed generations survive cancellation at the arm cutoff and interruption before the final receipt. Resume reauthenticates completed folds, including completed work inside a partially failed candidate. Only jobs positively proven never admitted can acquire a different expiry status. That exception cannot create a new fit, replace fitted state or extend the clock.

Legacy fit, emit and pipeline surfaces also exist. Their recipe emitters write a program that trains the recipe again; they do not export a fitted predictor; that is the separate, pending portable fitted-state export work. At `a75e543d2` they also cannot emit any registered frozen-panel recipe or executable synthesized ASM recipe: every such recipe carries the raw-table `(input …)` stage, and the legacy planner has no case for it (`crates/jope-bench/src/panel/methods.rs:20-21`, `:50`; `crates/jope-pipeline/src/synthesized_recipe.rs:34-46`; `crates/feat-math/src/spec.rs:383`, `:443`, `:592`). [§](#c-inference--language-interfaces-at-the-frozen-source) gives the detail and [§](#c-inference--what-passed-and-what-was-measured) the worked-example result. In short:

| Language | Recipe emit and refit (legacy emitters) | Exact fitted-state export |
| --- | --- | --- |
| Python, Rust | The emitter's legacy program set, not panel or ASM recipes; in the private worked example the emitted programs refitted `(pipe X (standardize) (ridge))` with bit-identical predictions | Pending (INFER-02); the public fitted Python/Rust packet is unmerged |
| C++17, C++20 | One C++ emitter compiled under both standards; same worked-example result | Pending (INFER-02); the pending public bundle still needs Python `ctypes` at inference |
| JavaScript | ES2020 emitter; same worked-example result | Pending (INFER-02) |
| CUDA | None | Deferred (FUTURE-01) |

<a id="c-status--latest-measured-operational-proof"></a>

## 1.5 Latest measured operational proof

The current source (reviewed `68f56404`, merged as main `a75e543d2` with the same tree) has a successful operational proof on `230_machine_cpu`: the verified encoder produced context; the neural producer supplied actual generation-zero fits; a fitted descendant followed; two fitted refits each returned 34 development-hold predictions. The neural refit did not beat the generic arm, which had the same declared allowance but more fit time; this is one budget-mismatched sample, not a matched quality comparison. This proof uses the accepted checkpoint, not a newly trained experimental export. As of 2026-09-23, new training, final installed warm demonstrations and the frozen thirty-dataset panel (asm-private-30-v1) had not started, and no current thirty-dataset ranking or Elo exists.

<a id="c-status--summary-of-the-one-dataset-proof"></a>

### 1.5.1 Summary of the one-dataset proof

This is **one preselected regression-dataset operational qualification**, not the unchanged thirty-dataset campaign, which has not run. Its scores are development-hold scores: each final refit is scored on the 34 held rows of development fold 0. This is a short summary; the canonical proof record is [§](#c-status--canonical-proof-record-and-per-arm-facts).

| Item | Observation |
| --- | --- |
| Dataset | `230_machine_cpu` |
| Initial neural arm | 64 actual `neural_policy` slots, zero fallback slots, plus 16 `random_exploration` slots in this proof |
| Initial generic arm | 80 generic slots |
| Declared arm / fit allowance | 120 seconds and at most 96 candidate slots per arm; 30 seconds per worker fit. Equal declared allowances are not equal fit time: context and policy production ran inside the neural arm's clock, and the policy producer's exit was observed 41.581 seconds after the campaign start |
| Arm stop reasons | Neural `original_deadline`; generic `evaluation_limit` |
| Global proof window | 600 seconds, including 30 seconds for reporting |
| Host limits | 20 CPUs/workers, 64 GiB, zero swap, no GPU tasks |
| Started and finished worker commands | 964 |
| Completed / failed commands | 834 / 130 |
| Unique fitted predictors / reusable recipes | 832 / 164 |
| Final refits / predictions per refit | 2 / 34 |
| Run-command wall time | 239.1 seconds |
| Independent report replay | 18.7 seconds |
| Neural development-hold RMSE (n = 34) | 166.3 |
| Generic development-hold RMSE (n = 34) | 155.3 |
| Neural ÷ generic RMSE | 1.071 (neural worse) |
| Score ordering | Generic lower RMSE in one budget-mismatched, one-dataset sample; not a matched quality comparison |
| Interval, baseline and winner origin | Not computed: no confidence interval and no mean-predictor RMSE on the same 34 rows. The neural arm's selected recipe (`ad665e5b…`) is a generation-1 crossover of two generation-0 `random_exploration` candidates, with no `neural_policy` ancestor ([§](#c-status--canonical-proof-record-and-per-arm-facts)) |
| Genuine neural initial fits and fitted descendants | Verified |
| Teachers admitted from this proof | None |
| Full frozen-panel conclusion | Not established |

Table values are rounded; the exact values are in [§](#c-status--canonical-proof-record-and-per-arm-facts), and the exact losses and run wall time are also in `data/performance-evidence.json.gz`. The recovered report SHA256 is `6e0c20720256b0735a6969c45032f44082c92abacadf2c4eb21ef5c390f249dd`. The experiment identity is `646095badab6706168c7e869118d62fddd399faa2a9d6bc7f484c95a021a3944`; its original window ran from 2026-09-23T16:16:50.613Z to 16:26:50.613Z (Unix milliseconds `1790180210613` and `1790180810613`). Plan identity: `a7befb386344bd0cbffa866ad8793077dead185f64e2ae0fb4035a4ed3dcfcc2`.

This proves execution and evidence continuity. It does not prove ASM superiority, ten wins per task, learned improvement after training, an under-sixty-second warm demo, installation speed or portable inference for every winner. The arm clocks are explained in [§](#c-scheduling--arm-order-and-clocks); the dense-width refusals (16 of 64 genuine neural-policy candidates against 4 of 80 generic candidates) are in [§](#c-status--failures-dense-width-refusals-and-the-earlier-failed-reproducer).

<a id="c-status--canonical-proof-record-and-per-arm-facts"></a>

### 1.5.2 Canonical proof record and per-arm facts

| Quantity | Measured result |
|---|---|
| Source | `68f56404c3e199e06413bfd645d3fe0803d7b96f` |
| Dataset | `230_machine_cpu`, regression |
| Immutable build wall seconds (preparation; not a cold-install measurement) |170.4488760890672|
| Eight-case native context qualification wall seconds (separate preparation) |29.248627813067287|
| Run wall seconds |239.14312197000254|
| Commands started / finished |964 / 964|
| Completed / failed command outcomes |834 / 130|
| Train OOF / diagnostic / refit-screen commands |960 / 2 / 2|
| Unique complete fitted predictor artifacts |832|
| Unique reusable recipes |164|
| Successful final refits |2, each 34 predictions|
| Neural development-hold RMSE (n = 34) |166.3181717454848|
| Generic development-hold RMSE (n = 34) |155.25909998168862|
| Generic minus neural loss |−11.059071763796169; neural did not win|
| Neural ÷ generic RMSE |1.071 (neural 7.1% higher; `neural_win=false`)|
| Known worker CPU seconds |181.220773|
| Sum of worker wall seconds |202.266164543|
| Exact supervisor aggregate CPU seconds |270.889403|
| Unknown worker CPU / wall counters |0 / 0|
| Teachers admitted from this proof |0|

The two arms had the same declared allowance but not the same work. Per arm, from the recovered report and its audit:

| Quantity | Neural arm | Generic arm |
|---|---|---|
| Declared arm clock | 120 s | 120 s |
| Work inside the arm clock before any fit | Context encoding and policy production; producer exit observed 41.581 s after campaign start, leaving at most 78.613 s to the neural cutoff ([§](#c-scheduling--arm-order-and-clocks)) | No context or proposal production (`proposal: null`) |
| Evaluation cap | 96 slots: 80 generation zero, 8 mutation, 8 crossover (one offspring wave) | 96 slots, same split |
| Stop reason | `original_deadline` | `evaluation_limit` |
| Completed five-fold unique recipes | 77 of 96 | 89 of 96 |
| Generation-zero completions | 62 of 80: `neural_policy` 48 of 64, `random_exploration` 14 of 16 | 76 of 80 `generic` |
| Generation-zero dense-width failures | 16 of 64 `neural_policy` | 4 of 80 `generic` |
| Selected recipe | `ad665e5b…`: generation-1 crossover of two generation-0 `random_exploration` candidates; no `neural_policy` ancestor | `c806c3be…`: generation-1 mutation of a generation-0 `generic` candidate |
| Selection score of the selected recipe (mean of five inner-fold RMSEs on the same 135 cut-0 fitting rows and inner split) | 61.776 | 65.362 |
| Refit and scored rows | Refit on the 135 cut-0 fitting rows; scored on the 34 held rows of development fold 0 | Same rows |
| Development-hold RMSE | 166.3181717454848 | 155.25909998168862 |
| Neural ÷ generic RMSE | 1.071 | — |

Generic has the lower development-hold RMSE in one budget-mismatched, one-dataset sample; this is not a matched quality comparison. Because the neural arm's selected recipe has no `neural_policy` ancestor, its refit score measures the arm as a whole, not a program the policy proposed. Not computed or not recorded for this proof: a mean-predictor (trivial-baseline) RMSE on the same 34 rows, a paired bootstrap interval for the RMSE difference, and a comparison at equal completed evaluations. Numeric parity of the Candle CPU policy port that produced the proposals with the trained tch forward pass is not established, and the port's `cuda` feature has never been compiled (`crates/jope-policy/src/lib.rs:84-95` at `a75e543d2`); the report records `ranking_backend: candle_cpu_float32` and `native_cuda_parity_qualified: false`.

<a id="c-status--the-outcome-figure-and-its-limits"></a>

### 1.5.3 The outcome figure and its limits

This is the current bounded operational proof, with one preselected regression dataset and equal declared search-arm allowances but unequal fit time. It demonstrates authenticated neural-origin fits and fitted descendants. The complete thirty-dataset ASM panel has not run, so this figure cannot establish its wins, ranking or Elo.

The current operational figure contains only the two development-hold RMSE values (n = 34) from the preselected `230_machine_cpu` proof: neural 166.3181717454848 and generic 155.25909998168862. Lower is better. Generic has the lower RMSE in this one budget-mismatched, one-dataset sample; it is not a matched quality comparison ([§](#c-status--canonical-proof-record-and-per-arm-facts)). It would be misleading to compute a new thirty-dataset Elo from these two values or describe the 832 fitted predictors as 832 independent datasets.

![Current operational proof: neural and generic development-hold RMSE (n = 34)](figures/current-origin-loss.png)

The two underlying records are in `data/plot-current-origin.csv`; exact evidence and denominators are in `data/performance-evidence.json.gz`. There are 34 predictions per final refit. The proof's run wall time is 239.143 seconds; its independent recovered report takes another 18.719 seconds. CPU, wall-time and per-predictor accounting follows in [§](#c-scheduling--worked-example-cpu-wall-time-and-per-predictor-cost-of-the-one-dataset-proof); the status summary is [§](#c-status--latest-measured-operational-proof).

<a id="c-status--failures-dense-width-refusals-and-the-earlier-failed-reproducer"></a>

### 1.5.4 Failures: dense-width refusals and the earlier failed reproducer

Every one of those 130 failed commands exited 1 with a verified stderr message `generated dense width N exceeds 512`. They represent 26 distinct candidate recipes with five failed folds each. Generation zero included 16 of 64 genuine neural-policy candidates failing this way (25%), compared with 4 of 80 generic candidates (5%); the remaining failures came from exploration or offspring. These jobs reached their process gates, but the failed candidates lack complete physical-fit counters, so the corresponding per-origin total remains unknown rather than inferred as zero or five. The 512-column check occurs inside fit execution; a verified prelaunch width rejection remains an optimization opportunity. Sources: `crates/feat-math/src/advanced_v3/feature_block.rs:109`, `crates/feat-math/src/advanced_v3/grammar.rs:2299-2300`; retained 130 finished/stderr hashes and report origin counts.

The earlier failed reproducer at source `d0f5748b7db1` failed after 194.789382994 seconds at scorer replay. All 964 commands had terminal receipts, with 834 completed and 130 failed. Its exact supervisor CPU was 252.614611 seconds and worker CPU 181.863951 seconds. It remains a failed reproducer; the old journal and deadline were not repaired or reopened.

<a id="c-status--failure-records-are-retained"></a>

## 1.6 Failure records are retained

Two failed cancellation/crash reproductions preceded the resume correction. A partial-fold fixture initially failed on a noncanonical path, then passed after fixing the fixture without weakening verification. A live run at `d0f5748b7db1ac419110289f912c7eface0022bc` completed its worker inventory but failed report verification because live and recovered scoring populated different fold-evidence fields. Its journal remains unchanged. The shared scorer correction and the successful proof at the corrected PR head commit `68f56404c3e1` are separate evidence.

Final local suites at the snapshot passed **53 benchmark and 106 pipeline tests**. One privileged test was ignored in that invocation and qualified separately through **ten actual host cases**. The numerical encoder proof passed **eight fixed cases** under unchanged tolerances and thirty-second per-job limits. [§](#c-catalogue--the-pr83-lifecycle-record) records CI and merge state independently.

<a id="c-status--pending-acceptance"></a>

## 1.7 Pending acceptance

| Workstream | Current boundary | Remaining acceptance |
| --- | --- | --- |
| Regression neural consumption | Qualified contexts, proposals, actual fits and descendants on one dataset; parity of the CPU policy port with the trained model not established | Reference-fixture parity with the trained model, full panel and newly trained checkpoint consumption |
| Classification neural consumption | No classification neural arm at the current source: the accepted policy is regression-only, and classification `asm.generic` searches ridge heads only ([§](#c-architecture--experimental-classification-context)). Experimental context/invariance support; follow-on producer prepared | Task-correct live consumption and clearly labelled fixture scope |
| High-water mark (HWM) and teaching eligibility | Follow-on source prepared; zero teachers admitted under the new contract | Full identity, finite out-of-fold (OOF) improvement, four-of-five-fold gate and no-signal refusal |
| New ASM training path | Historical trained model exists; new guarded source and test plans prepared. The pending campaign trains a new, smaller Python model, not a continuation of step 60,000 | Positive updates, changed parameters, reload, continuation, resume compatibility and bounded GPU proof for this new path |
| Trained regression export use | Explicit experimental consumer prepared | Compile and execute exported-checkpoint-driven fits |
| Portable fitted inference | Native fitted artifacts; export packet prepared; legacy emitters cannot emit panel or ASM recipes | Full state parity and actual selected-winner backend qualification |
| Installation and warm demo | Source packets prepared | Checksum install, CLI-only build, preparation, fitted artifact/predictions and separate cold/warm timing |
| Thirty-dataset comparison | Frozen panel and protocol; not yet run (0 of 30 datasets) | Unchanged two-hour campaign, every loss/refusal/failure and gap |
| Motifs | Separate typed library branch | Eligible-state reader, mine command, umbrella wiring and validated reuse |
| Broad full-main tests | Historical failures retained; full-main runs 35929010741 and 35936673628 failed, each with eight families passed, two failed and one unavailable. The earlier PR83 record remains historical ([§](#c-catalogue--the-pr83-lifecycle-record)) | Reproduce and repair; fast CI is not full qualification |
| New serving promotion | Recovered accepted step-60000 remains the model binding; the native command that could continue it is held | Exact release/backend/model acceptance and reviewed promotion for a new checkpoint |
| Manual publication | Edition 1 landed through PR85 at `bdfc95c63`; DOCFIX-01 landed through PR86 at `2b375ef11`. PR84 was superseded because its intermediate history exceeded the 5 MiB file cap. Edition 2 landed through PR87 at `02c0bef6c` after exact-head CI and independent native approval; main-guard passed | Each later revision needs fresh exact-head CI, independent native approval and reviewed merge |

[§](#c-plan--compact-task-index) and the acceptance cards in [§](#c-catalogue--primary-acceptance-items-by-stage) expand these rows into actionable items; the property list carries the same items. A prepared patch or planned command is not an installed or executed capability.

<a id="c-status--what-version-are-we-running"></a>

## 1.8 What version are we running?

<a id="c-status--implementation-snapshot-and-this-manuals-own-identity"></a>

### 1.8.1 Implementation snapshot and this manual's own identity

The implementation snapshot is **68f56404c3e199e06413bfd645d3fe0803d7b96f**, dated 23 September 2026. [§](#c-catalogue--the-pr83-lifecycle-record) records the separately refreshed reviewed-main state. A rebase merge can produce a different commit with the same reviewed tree. Older plans and historical PASS entries do not establish current capability.

This edition describes the implementation reviewed at `68f56404c3e199e06413bfd645d3fe0803d7b96f`. PR83 merged on 23 September 2026 at 16:34:47 UTC. The resulting remote main is `a75e543d2166017cb0139e0a251d51d9a0c98e08`. Both revisions have Git tree `176b75bc36d76c7ce4eaeb7234677ec013468889`. Rebase merge changed commit identity; it did not change the reviewed source tree.

The manual itself is a later documentation change. Its publication identity must be read from its own Git history and CI receipt, not confused with the implementation snapshot. Follow-on patches described as pending are outside the implementation snapshot even if they are preserved on this host.

Publication history, refreshed after the PR87 merge; the application snapshot above stays pinned to PR83:

- **PR84 was closed as superseded** (coordination record of 22:22:38 UTC). Its head is `e06c0551e3ec5ee5260debb3d953c17f34d15093`. The independent non-contributor review 5295763455 (state COMMENTED, 19:30:12 UTC, at that head) withheld approval for two reasons: a blocking finding that three intermediate commits each contain two files larger than the repository's 5 MiB cap (`deploy/ci/check.sh:46`), so a rebase merge would put those commits on `main`; and exact-head CI that was not yet fully green.
- **PR85, the clean-history successor, merged at 22:33:36 UTC.** Codex, the manual owner, published the single commit `2022017a8c45cc25967ae99ae781ca26276140b8` on `a75e543d2` (branch `docs/asm-maintainer-manual-final`, pushed at 22:21:30 UTC) with a 203-page PDF (SHA-256 `69418c48…`). Exact-head CI run 35927921851 attempt 1 succeeded, and the independent non-contributor review 5297559032 approved that head at 22:32:39 UTC. The merge produced `main` `bdfc95c63d905c647e41175e79dc30cdc8600287`, whose Git tree `fbbed08a…` is identical to that of `2022017a8`.
- **Edition 2** restructures and corrects `2022017a8`. It was prepared in Claude's lane and integrated by the manual owner, preserving the authoring commits. Edition 2 merged through PR87 at 2026-09-24T02:58:00Z as main `02c0bef6c81ab17e034daea308c1a0b5a7940467`, after exact-head CI 35948653955 attempt 1 and independent native approval 5299139743; main-guard 35949456287 passed. Further corrections repeat the reviewed lifecycle.

<a id="c-status--the-identity-tuple"></a>

### 1.8.2 The identity tuple

There is no single version number that describes all of the following. Record the entire identity tuple whenever comparing results or continuing training.

| Dimension | Current evidenced identity | What changing it means |
|---|---|---|
| Product/package | Rust workspace and Python package `0.1.0`; immutable CLI reports `jope 0.1.0` | A packaging/API release number; not a checkpoint generation or proof of installed capability. |
| Implementation | Reviewed source `68f56404c3e199e06413bfd645d3fe0803d7b96f`, merged as `a75e543d2166017cb0139e0a251d51d9a0c98e08` with the same source tree | A later documentation commit does not rebuild the binary. Preserve binary/build/source-manifest hashes. |
| Accepted regression policy | Lineage B, step 60000, `parallel.checkpoint.v1`; state hash `8ac80c269cc47c253b6f53fa3bc1eb75bb2fcda0bdace3ff8554be0175c400d6` | The current accepted consumer hard-pins this identity and its model/configuration/parameters. Pinning does not establish numerical parity: the Candle CPU port claims no parity with the trained tch forward pass, its logits and rewards are unverified, and its `cuda` feature has never been compiled (`crates/jope-policy/src/lib.rs:84-95`). |
| Hardware-qualification continuation | Step 60100 is retained hardware-qualification continuation evidence, state SHA `745be4b5e998bce85615349a5b03b901cd8ad2cf605e6dce3b86fbf777cc7388`; verify the actual training pointer (the `training-head` key) independently | This is not the accepted serving checkpoint. Do not advance a pointer from a filename or step number. |
| Rejected policy | Step 69000 is explicitly refused by state or model hash | A numerically higher training step does not imply promotion. |
| Python model family | Current Python implementation declares `policy_v3.4-split-context`, with legacy `policy_v3.3-fused-heads`; trainer resume format `PrimePolicyTrainerResume.v1` | These identify a different trainer/model serialization surface from the accepted native `model.ot` lineage. |
| Experimental ASM policy (pending) | The stopped training64 patch would train a new Python `JopePolicyModelV3` (width 256, 2 layers, 4 heads, feed-forward 1024), initialized at most from a Python checkpoint; no real campaign has run with it | Experimental export only: not serving, not lineage B and not a continuation of step 60000, which the Python path cannot load. Compare the two only through matched downstream fits, never through loss values. The native route that would continue step 60000, `jope-parallel`, is held ([§](#c-training--the-dispatch-hold-resource-incompatibility-and-valid-topologies)). |
| Encoder | Current CPU (Rust/Candle FP32) context schema `jope.cpu-context/v1`, separate from the held native GPU path; produced artifacts require pinned native-fixture replay before qualification. Dataset 1536 (Hyperion 512 from the features only + projected foundation 1024 from the features and training targets) + target 512 = 2048 floats, so “dataset 1536” is a historical label that includes target-conditioned values; legacy frozen encoder descriptor `parallel.frozen_encoder.v1` | Every component weight, transform, source/backend and assembly hash participates in identity. Dimensions alone do not establish compatibility. |
| Hyperion assets | Current CPU producer pins the v5 arm-B weight hash `88b9e8345e44d3ea609a8286215618a762f14e607fcdada6d27447b1f74668be`; a separately stored bundle is called `hyperion-v7` | A bundle name or Rust `V7Model` type is not permission to substitute newer-looking weights. “v5” here is asset lineage, not a global product schema version. |
| Action vocabulary | Accepted 401-token vocabulary SHA `8d55fee4366acd2c175cf55bfd769cebc904770f5227a6fd50a613723dc85787`; documented product restriction SHA `570b2e5e8d859a73b888f5d0bee27f59da0e2376536babce24c241604f818e81` | The restriction preserves the token table. Adding tokens or changing parameter bins requires a new compatibility and training decision. |
| Corpus | Native `parallel.teaching_shards.v1` / `parallel.teaching_corpus.v1`, plus immutable index, vocabulary, contexts, records, label policy and exposure identities | A corpus schema does not certify teacher eligibility, data freshness, or unseen-family status. |
| Dataset/catalog | `jope.dataset-catalog/v1`, `jope.dataset-collection/v1`; individual content and split hashes remain separate | Keeping the schema while changing rows, labels, categories, groups or splits produces a different comparison. |
| Model packaging | `jope.model-bundle/v2` | Package/license/inventory metadata is separate from accepted model identity and successful inference. |
| Frozen benchmark | `asm-panel-v1:screen-fold=0:top=5:confirm=all-5:mean-fold-loss`; evaluator `jope-bench.frozen-panel-f64/v1`; manifest SHA `ba7dace4cd535687002cc95538dbc7f7aaf77ec973177c7159b3c6d22471eee0` | Altering a method, split, deadline, evaluator or selection rule requires a new protocol identity. Do not relabel an altered run as the frozen panel. |

Source: [workspace version](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/Cargo.toml#L30), [Python package](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/python/pyproject.toml#L5), [accepted checkpoint and vocabulary pins](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-policy/src/lib.rs#L132), [Python model version](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/python/src/jope/policy/model_v3.py#L43), [resume schema](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/python/src/jope/policy/train_v3.py#L57), [context assets and identity](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-policy/src/context_producer.rs#L35), [legacy encoder descriptor](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-engine/src/encoding.rs#L47), [catalog schemas](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-datasets/src/catalog.rs#L9), [bundle schema](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-models/src/manifest.rs#L9), [panel identity](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-bench/src/panel/mod.rs#L36).


---

# Chapter 2 — Architecture, encodings, GP representation and model lineages

<a id="c-architecture--explain-it-like-im-5-learning-to-suggest-better-recipes"></a>

## 2.1 Explain it like I’m 5: learning to suggest better recipes

Imagine a child learning to choose recipes for unfamiliar ingredients. The table of data is
the ingredients. A small description says what those ingredients look like. The neural
model reads that description and suggests recipes: perhaps clean the numbers, combine some
columns, then fit a predictor. It suggests instructions; it does not already know the fitted
answer for this table.

We cook the recipes by actually fitting them. We test them on examples whose answers were
withheld from that fit. Genetic programming (GP) changes promising recipes, or combines
parts of them, and cooks the new versions too. A learned reward score helps choose what to
try; only the real fitting and scoring tells us how a recipe performed.

Some recipes become teaching examples, but only after the training-side admission rules
check them. Neural training changes the model’s numerical weights so it can imitate useful
recipes and rank candidates. We then ask the new model to suggest recipes for another
measured search. A lower loss on the same objective and examples means a better fit to that lesson; it does not
by itself mean it will cook better on a new table.

Think of withheld answers as an exam. Practice answers may guide study. The exam answer key
must stay outside every decision that chooses the recipe being tested. If we look at an exam
score and change the recipe, that exam has become practice. The nested transfer audit gives
each practice-side audit its own full search, then separately tests a full-training search
on the outer exam. All predictions must be committed before anyone sees the round’s audit or outer scores.

![Data, search, teaching and measurement boundaries](figures/ip-pipeline.svg)

The diagram describes executable source responsibilities and the required measurement
boundary. It does not claim that a new live learning loop has been qualified. Red paths are
prohibited dependencies: outer answers must not reach context construction, search,
teaching or optimizer updates; released results must not adapt unfinished folds.

<a id="c-architecture--current-executable-pipeline-and-the-historical-snapshot"></a>

## 2.2 Current executable pipeline and the historical snapshot

This addition is pinned to merged measurement source `7f2b2bf860e4b1ac8815c8da55e507143570a905` (PR90).
It refreshes the executable path while preserving the older sections and their anchors.
The PR83 source narrative below is a **historical snapshot**: its “pending overlay” language
is dated to that snapshot. Several of those source components have since merged, as the
following map records. Implemented source, CPU fixtures, historical runs, live qualification
and serving promotion remain separate claims.

<a id="c-architecture--from-original-rows-to-a-conditional-program"></a>

### 2.2.1 From original rows to a conditional program

The native regression context producer uses only permitted fitting rows. Hyperion embeds
feature rows into 128 values per row, then pools mean, population standard deviation and
10th/90th percentiles into 512 values. Foundation consumes features **and permitted training
labels**, producing 2048 values projected to 1024. The target encoder contributes 512.
Frozen normalizations and clipping assemble a 2048-vector. Thus the configuration name
`dataset_dim=1536` does not mean label-free: its Foundation block is target-conditioned.
The source and numerical qualification details remain in [§](#c-architecture--component-dimensions-and-assembly).

The accepted native model has width 512, six causal decoder layers, four attention heads
and feed-forward width 2048. Its vocabulary projects to 401 output tokens from 657 embedding
rows. Separate context MLPs map 2048→512→512 for decoding and reward. Decoding prepends a
context token to program embeddings and adds positions; causal and padding-key masks prevent
future-token/padding attention. A grammar mask limits each sampled token to legal continuations.
A valid token sequence still needs executable-schema validation and an actual bounded fit.

The reward path has a different representation: it averages non-padding program-token
embeddings, without positions. For one context, reordering the same token multiset cannot
create a different semantic reward representation. It combines that pooled vector with a
context-gated product and an absolute-difference block, giving 3×width features to four
reward heads. This learned ranking signal is not a measured RMSE. The runtime head applies
softplus to predict a positive cost. The autoregressive decoder remains order-sensitive.

![Accepted and experimental neural dimensions, attention and objectives](figures/policy-network.svg)

<a id="c-architecture--from-program-candidates-to-eligible-teachers"></a>

### 2.2.2 From program candidates to eligible teachers

`crates/jope-policy/src/decode.rs` and `ranking.rs` produce and rank a grammar-constrained candidate
pool. `crates/jope-pipeline/src/neural_search.rs` and the `jope-bench::panel` orchestration retain
origin, ancestry, selection and resource identities. `feat-math` supplies typed variation
and fitted preprocessing/predictors. Mutation and crossover operate on executable recipes;
scoring evaluates retained predictions against the appropriate labels. A neural proposal,
a mutated descendant and a generic candidate are distinct origins.

`crates/jope-bench/src/panel/teaching.rs` now implements preregistration, admission and verification;
`crates/jope-learning/src/eligibility.rs` supplies the guard. Admission uses the fixed candidate’s
complete train-side OOF comparison, including the trivial baseline and immutable comparison
identity. Failed, incomplete or ineligible observations are not quietly relabeled teachers.
That fixed-program evidence serves teaching. The new transfer procedure audit instead
repeats selection independently inside each fold; these receipts are not interchangeable.

<a id="c-architecture--from-teachers-to-changed-parameters-and-another-measured-search"></a>

### 2.2.3 From teachers to changed parameters and another measured search

`crates/jope-cli/src/train.rs` now exposes preparation/materialization, planning, run, resume,
comparison and export. `python/src/jope/policy/training_campaign.py` supervises the bounded
Python path. Its normal experimental configuration is width 256, two layers, four heads and
feed-forward width 1024. Its input width is corpus/checkpoint-bound `dataset_dim + target_keep_dim`,
including any permitted zero-tail pruning; it is not universally the native 2048. This is a newly initialized lineage, not a continuation of native
512×6 step 60,000. Tiny 8×1 CPU smoke fixtures are a third configuration, not historical
training evidence.

`model_v3.py::loss_components` and `losses.py::kg_combine` implement supervised sequence
negative log-likelihood, reward pairwise Bradley–Terry and listwise ranking terms, runtime
Huber, stability BCE, a program-only ablation term and routing cross-entropy, subject to the
selected profile and available supervision. Diagnostic listwise entropy is a baseline for
interpreting cross-entropy, not a new measure of generalization. These are imitation and
reward/ranking objectives; the implemented training path is not PPO or GRPO.

`train_v3.py` runs backward passes and AdamW updates, retaining model/optimizer state,
sampler/exposure information and proofs of actual update counts and parameter changes.
`training_transition.py` binds corpus, optimizer and schedule invariants. The comparison
path implements declared fresh/replay siblings and a replay-only control within their
original windows; actual sampled exposure must be verified, not inferred from the nominal
mixture. Resume is not permission to reset a deadline or silently replace replay supervision.

A checkpoint file alone proves none of this. Verify committed updates, changed parameters,
lineage and reload first; then compare the challenger with its parent by actual downstream
fitting under one frozen protocol. `training_campaign.py` export verifies the campaign and
marks the experimental bundle `serving_eligible: false`. It does not promote the accepted
serving pointer. Native Candle-versus-tch policy numerical parity remains limited as recorded
in [§](#c-architecture--decoder-and-reward-architecture); the source map is not a new parity qualification.

<a id="c-architecture--from-a-fitted-predictor-to-deployable-code"></a>

### 2.2.4 From a fitted predictor to deployable code

The private recipe, fitted native state, emitted refitting recipe and fitted inference
bundle remain four different artifacts ([§](#c-architecture--gp-representation-recipes-fitted-predictors-and-the-four-artifact-types)).
`fitted_export.rs` prepares an allowlisted bundle and `fitted_export_run.rs` qualifies emitted
predictions under the original limits. Match the raw input boundary, saved preprocessing and
native predictions across supported backends. Successful source generation is not numerical
qualification, and a qualified export is not evidence that training improved the policy.


<a id="c-architecture--preserved-historical-chapter-snapshot"></a>

## 2.3 Preserved historical chapter snapshot

The following sections retain the original dated source narrative and evidence. The current-source additions above supersede implementation-status descriptions where explicitly stated.

**Algorithm Synthesis Model (ASM) learns to generate and optimize executable algorithms from data.** The implementation combines a learned proposal policy with actual fitted evaluation and genetic variation. A generated program is a candidate recipe; a fitted predictor is that recipe plus the state learned from a specific training partition. A checkpoint is the proposal model. Those three artifacts have different identities and different evidence requirements.

This chapter describes source commit `68f56404c3e199e06413bfd645d3fe0803d7b96f`, the reviewed PR83 head. PR83 merged on 2026-09-23 at 16:34:47Z as `main` commit `a75e543d2166017cb0139e0a251d51d9a0c98e08`, which has the same tree (`176b75bc36d76c7ce4eaeb7234677ec013468889`), so the current-source references below also describe `main`. As of 2026-09-23 about 23:55 UTC, the only later merge to `main` was PR85, edition 1 of this manual, which produced `main` `bdfc95c63d905c647e41175e79dc30cdc8600287` at 22:33:36 UTC. PR85 changed only files under `docs/` plus `README.md` and `AGENT_CHAT.md`, and no file under `crates/` or `python/`, so the source references below still describe `main`. Changes described as **pending training overlay** refer to the stopped 64-file custody patch `595824ef6eb8e15872be058668503cedad1dd9a5ff98ac4700c6f713251d384c`. That patch includes experimental regression consumption; it is not in this source commit. Current-source references below are repository paths with source line numbers. Overlay references are explicitly marked and use the stopped overlay's line numbers.

<a id="c-architecture--system-at-a-glance-flow-diagram-and-component-map"></a>

## 2.4 System at a glance: flow diagram and component map

![ASM implementation and pending learning loop](figures/architecture-flow.svg)

The solid path has bounded regression execution evidence. Dashed stages require the named runtime acceptance; this diagram does not claim an already qualified self-improving training loop.

| Component | Main responsibility | Durable boundary |
|---|---|---|
| `jope-datasets` | Frozen dataset metadata, source and split identity | Content-addressed manifests, source hashes and exact row assignments |
| `jope-bench::panel` | Development-data preparation, methods, metrics, screen/freeze/confirm protocol, ASM orchestration | Campaign plan, prepared data, fits, frozen selections and reports |
| `jope-pipeline` | Search identity, bounded execution, original deadlines, process and output authentication | Started/finished records, arm windows, observations and receipts |
| `jope-policy` | Accepted regression checkpoint binding, CPU context producer, grammar-constrained proposals and learned ranking | Checkpoint receipt, vocabulary, qualified context and proposal evidence |
| `feat-math` | Executable recipe representation, variation and fitting | Canonical recipe identity, fitted state and prediction output |
| `jope-learning` | HWM and score ledgers; pending stricter teaching admission | Comparison identity, immutable blobs, hash-chained events and frozen teaching manifests |
| Python policy package | Existing trainable policy model; pending bounded training campaign and experimental proposal consumer | Model/optimizer checkpoints, architecture, task/vocabulary binding and training proof |
| Native CLI | Current planning/execution front doors; pending `train` orchestration | Pinned requests, source/runtime inventories and supervised execution |

The ordinary CLI and `asm` feature graphs in the stopped follow-on were inspected as manifest text: they have no edge to the engine, Torch, CUDA bindings or a local dependency cycle. That is a source graph check, not a compilation result. Python training runs as a supervised external process; native orchestration does not require linking Torch into the CLI.

<a id="c-architecture--the-learning-cycle-what-is-proven-and-what-remains"></a>

## 2.5 The learning cycle: what is proven and what remains

The most recent bounded origin proof at this source completed its run command in 239.143 seconds; independent report replay took 18.719 seconds. It demonstrated neural-origin fitting and evolution on one regression dataset, `230_machine_cpu`. On the same 34 held development rows the neural arm's refit scored an RMSE of 166.32 against 155.26 for the generic arm (`neural_win: false`). The arms had equal declared allowances, not equal fit time: context encoding and policy production ran inside the neural arm's clock, the neural arm stopped at its original deadline and the generic arm at its evaluation limit ([§](#c-scheduling--arm-order-and-clocks)). This one-dataset result is therefore not a matched quality comparison. It did not execute the complete thirty-dataset demonstration, create current eligible teachers, train a new policy, or qualify a newly trained policy for serving. [§](#c-status--canonical-proof-record-and-per-arm-facts) gives its exact receipts and denominators.

1. Preserve the frozen data schema, original target strings, category strings, missingness and row/group assignments.
2. Derive the proposal context using only the allowed fitting rows. Record preprocessing and encoder identities, then verify actual context production and the numerical qualification required by that context lineage.
3. Use the chosen checkpoint to sample in a bounded number of rounds, stopping once 64 distinct valid executable recipes exist; the first 64 in draw order form the neural pool, and any shortfall is filled as described in step 4. Score exactly the 64 pool programs with the mean of the four learned reward heads. That score orders the pool's slots and records `rank_log_weight`; it does not select or reject programs. This stage has no access to held-out scoring labels or fitted GP scores.
4. Place a 64-slot policy pool plus 16 exploration candidates into the neural arm's generation zero; preserve any policy-pool shortfall as `neural_rejection_fallback`, never as a genuine neural sample. Compare with an 80-candidate generic arm under the same declared budgets and executable grammar policy. Context encoding and policy production run inside the neural arm's clock, so equal declared budgets are not equal fit time.
5. Actually fit candidates on inner training folds and score their out-of-fold predictions. The measured outcomes, not model-predicted quality, determine the search's parents and winner.
6. Generate mutation and crossover descendants with explicit parent identities. Preserve duplicate/cache handling, rejected variations, unavailable methods and every fit failure.
7. Freeze a recipe and evaluate the unchanged development protocol. Its fitted fold artifacts are separate from its reusable recipe.
8. In the pending teaching path, admit a recipe only after its independently preregistered train-side OOF evidence beats a trivial baseline under the frozen numerical rule.
9. Freeze the eligible corpus, train an explicitly experimental policy, prove real updates and continuation, and export a pinned checkpoint. The pending campaign trains a new, smaller Python `JopePolicyModelV3` (width 256, two layers, four heads); it is a different architecture from the accepted native step-60000 model (width 512, six layers), not a continuation of it ([§](#c-training--what-the-pending-campaign-trains)). This manual names that line **experimental regression lineage v1 (256×2, new initialization)**; [§](#c-architecture--model-lineages) keeps it separate from the accepted 512×6 lineage-B model.
10. Consume that exact new checkpoint in a subsequent bounded experiment. A successful export alone is insufficient evidence for this final step.

Steps 1–7 have current bounded regression evidence at the scope described above. The stricter teaching/training/experimental consumption integration is stopped source work awaiting the next reviewed lifecycle and actual runtime acceptance.

Steps 8–10 support four distinct claims, and evidence for one never establishes another (owner rule):

| Claim | What must establish it | Status |
|---|---|---|
| Eligible teacher | Admission under the preregistered train-side guard of step 8, with the full comparison identity; a hash, a stored score or a corpus row is not eligibility | Pending integration |
| New discovery | A program absent from the authenticated initial population of the search that found it, retained with its discovering arm and ancestry; a discovery cannot silently seed another arm | Proposed |
| Actually consumed information | Committed optimizer updates that measurably sampled the new rows, with that exposure recorded | Proposed |
| Improved policy | A matched downstream fitted comparison of the trained checkpoint with its parent under the same budgets, as in step 10 | Proposed |

The preregistered train-side guard exists only in the stopped overlay, and the demonstrated teacher count under it remains zero in the inspected evidence ([§](#c-training--what-can-be-claimed-now)). A source investigation of that overlay found no initial-population or discovery identity and no per-update fresh-exposure record in the pending campaign (`plan-execution-20260923/boundary-resume-siblings.md` under the ASM custody root, [§](#c-installation--physical-layout-and-identities)).

The owner's rules, also proposed, fix three family-level domains: teacher/train families, adaptive development families, and evaluation families kept outside training and selection. The frozen thirty-dataset panel remains adaptive development ([§](#c-benchmark--adaptive-development-exposure)), and the exposure of the pretrained encoders to any family remains unknown. The permitted input closure covers records, contexts, transforms and all assets; an unused forbidden descriptor must be rejected before loading, not merely left unsampled ([§](#c-data--audit-the-entire-input-closure-before-continuation)). The same investigation found that the stopped overlay's native family partitioning has only train and development domains, and that its Python campaign does not reject extra context-index entries.

Remaining architectural acceptance includes real eligible-teacher collection, actual new-parameter training/reload/continuation, subsequent fitted use of that newly trained checkpoint, classification operational qualification, complete fitted-state parity across supported deployment backends, the unchanged full30 panel, and measured warm/cold installation demonstrations. Reusable motif mining must retain full comparison identity and eligibility; new operators need ablation and compatibility tests. GPU inference portability remains future work. Numerical parity of the Candle CPU policy port with the trained tch forward pass is not established: the source states that logits and rewards from the port are unverified until reference fixtures exist, and its `cuda` feature has never been compiled (`crates/jope-policy/src/lib.rs:84-95`).

<a id="c-architecture--gp-representation-recipes-fitted-predictors-and-the-four-artifact-types"></a>

## 2.6 GP representation: recipes, fitted predictors and the four artifact types

The private program `(pipe X (standardize) (ridge))` is a small executable algorithm description. It takes a feature table, fits standardization from training rows, then fits a ridge-regression head. It is an ordered typed recipe, rather than an already fitted set of coefficients. A different transform or head changes the algorithm. GP search works within its admitted grammar and task constraints; the presence of a code emitter does not grant arbitrary-language or arbitrary-operator coverage.

Keep these artifact types distinct:

1. **Private recipe:** the lossless canonical program and its configuration. It can be fitted again on new training data and can carry private discovery provenance.
2. **Native fitted predictor:** preprocessing state, category dictionaries, imputation statistics, transforms, calibration and head parameters learned from one exact fit. It predicts without training again.
3. **Emitted recipe:** a program in another language with `fit` and `predict` behavior. Running it on training data fits a new instance of the algorithm. Agreement between these refits tests recipe/runtime equivalence.
4. **Fitted inference bundle:** an allowlisted implementation of one exact already fitted predictor. It contains the selected inference state and an input boundary; qualification compares its predictions against the original saved native artifact.

<a id="c-architecture--encodings-and-the-proposal-context"></a>

## 2.7 Encodings and the proposal context

<a id="c-architecture--regression-context-production"></a>

### 2.7.1 Regression context production

<a id="c-architecture--original-data-and-preprocessing"></a>

#### 2.7.1.1 Original data and preprocessing

The context input binds prepared-data hash, exact training-content hash, ordered fitting-row hash, cut identity and preprocessing hash. It requires at least eight training rows and a nonempty declared feature schema. Numeric versus categorical kinds come from the schema rather than guesses based on strings. Numeric nulls are imputed using training-only medians, with a zero fallback for an entirely missing column; the recorded training transform contains its fitted location/scale. For categories, every present string is a value, including empty text or strings spelling `NA` or `null`. Only an absent value is missing. Original target strings remain separately authenticated before numerical regression parsing.

This context preprocessing is for the proposal encoder. It does not authorize discarding the original schema or replacing a fitted predictor's own training-only preprocessing. Source: `crates/jope-policy/src/context_producer/input.rs:84`.

<a id="c-architecture--component-dimensions-and-assembly"></a>

#### 2.7.1.2 Component dimensions and assembly

The current CPU producer uses three components with fixed pretrained assets and fixed transforms:

| Component | Production | Raw output | Contribution to policy context |
|---|---|---:|---:|
| Hyperion | Training feature columns only (no label), quantized from fitted quantile bounds; three embedding sweeps, scales 4/7/11/15, batch 1024, internal setting `target_chunks_per_sweep=1` (a chunking setting, not the label) | 128 values per row | Mean, population standard deviation, 10th and 90th percentile per coordinate: 512 |
| Foundation | Feature columns plus the training targets as in-context labels, prepared with the native deterministic recipe; target normalization computed in f64 after the native f32 cast | 2048 | Frozen mean-centering, learned 2048→1024 projection, then frozen normalization: 1024 |
| Target | Training targets, already standardized in preparation, then min–max rescaled to [0,1] by the training minimum and maximum (span 1 when constant); two frozen native target windows with seed 17 | 512 | Frozen normalization: 512 |

Assembly concatenates normalized Hyperion512 + projected Foundation1024 + Target512 and clips the full 2048-vector to [-32,32]. Only the Hyperion block is label-free: Foundation is conditioned on the training targets, so the first 1,536 values, which the policy configuration still calls `dataset_dim`, include target-conditioned values. It checks all component dimensions and finite values. The source explicitly checks that the Hyperion row width is 128; confusing row width with pooled512 changes the model input meaning. Sources: `crates/jope-policy/src/context_producer.rs:188`, `:278`, `:356`, `:399`, `:422`.

The deterministic foundation preparation selects at most 768 context rows and 256 query rows before `encode_standardized`; that encoder adds no further cap. Hyperion receives every supplied training row and feature column, with no row/column subsampling in this producer. The serialized input is capped at 128 MiB and every context job remains subject to the fixed 30-second worker deadline. On the eight final reference jobs at source `68f56404`, Hyperion-stage elapsed time ranged from 6.143 to 14.287 seconds; whole-job wall time ranged from 11.731 to 20.755 seconds. These small-case measurements do not establish coverage for wider or larger cuts. A timeout leaves the context unavailable; it does not permit a new deadline or silent subsampling.

The frozen numerical qualification contract is unchanged, but the implementation includes both execution and numerical corrections. Hyperion uses an embeddings-only sweep that skips the unused confidence head. Foundation uses the separate `encode_standardized` path: train-side feature moments accumulate in f64 before the retained f32 transform, and p10/p90 use Torch 2.7 FP32 rank and stable-lerp arithmetic. Ordinary `encode` retains its legacy feature-moment and percentile arithmetic. The standardized route consumes all rows supplied by the preceding deterministic foundation preparation without additional caps. Cached-decoder fixtures separately check small random-model operations/full-prefix agreement within 1e-4, row/padding behavior and sampling-token/RNG agreement. They do not establish that the Candle CPU policy port matches the trained tch forward pass: the source states that logits and rewards from the port are unverified until reference fixtures exist (`crates/jope-policy/src/lib.rs:84-92`). Native CUDA policy parity is also not established; the `cuda` feature has never been compiled (`lib.rs:93-95`). Sources: `crates/jope-policy/src/context_producer.rs:238`; `crates/jope-encoders/src/foundation.rs:345`; `crates/jope-encoders/src/foundation/support.rs:123`; `crates/jope-encoders/src/model/tensor_ops.rs:114`; `crates/jope-policy/src/model/cached.rs:270`; `crates/jope-policy/src/decode.rs:485`.

<a id="c-architecture--qualification-and-admission"></a>

#### 2.7.1.3 Qualification and admission

The CPU regression encoder has a stronger boundary than “the vector has 2048 values.” Eight fixed native-reference cases span two families and cuts 0–3. Qualification checks finite FP32 component and final-context values with maximum absolute difference at most 0.0003, exact asset/source/program identities, and actual bounded executions. Consumption binds those eight jobs plus the actual dataset context job: nine independently authenticated process proofs. The reference registry/index and transform hashes are pinned in source.

The eight retained reference outputs at source `68f56404` have final-context maximum absolute differences from 5.3644180e-5 to 2.8765202e-4. The worst case is `cardata-blackmore`, dataset `55e9dfc7a86361d7`, cut 0: 95.884% of the unchanged 3e-4 tolerance. Across all eight cases, maximum component differences are 4.127622e-6 for Hyperion, 2.908707e-5 for foundation and 3.911555e-7 for target. The frozen projection and projected-scale division can amplify small component differences in the assembled vector. Coverage is limited to two numeric meta-train families: blackmore, 629–630 training rows with 4 features, and `cardata-ces11`, 800 rows with 8 features, each at cuts 0–3. Changes to arithmetic, libraries, build/ISA flags or summation order require requalification; this gate is not evidence for wide or category-heavy contexts. Never relax its tolerance to make a build pass.

`VerifiedCpuContext` has private fields and can only be constructed by the verification path outside explicit test support. A syntactically valid JSON vector cannot create admission authority. Sources: `crates/jope-policy/src/context_producer/admission.rs:4`, `:115`, `:234`, `:297`; `crates/jope-pipeline/src/cpu_context_bridge.rs`.

<a id="c-architecture--experimental-classification-context"></a>

### 2.7.2 Experimental classification context

Current source contains the pure class-partition context representation. The pending training overlay adds the bounded feature producer, process admission and operational proposal wiring. Classification does not parse class names as numbers or reuse the regression target encoder by ordinal coding.

The feature blocks remain Hyperion512 and projected Foundation1024. The pending producer explicitly conditions the foundation path on neutral zero targets and labels this an experimental feature-only probe. Each class gets a centroid over the 128-dimensional training row encodings. The target block contains, per coordinate, the mean, standard deviation, minimum and maximum across class centroids, totaling 512. The centroids are unweighted, so class frequencies never enter the block. Classes are visited in first-row order, so a bijective renaming preserves even floating-point summation order. Original labels still appear in authenticated input identity, even though their spelling cannot influence numerical pooling.

The resulting lineage is `asm.experimental-class-partition-pool/v1`; `policy_compatible` is false. The false marker means it cannot enter the accepted regression policy loader. A declared binary input must have exactly two classes; multiclass requires at least three, and the fitting data must contain the declared classes. Missing classes are an explicit refusal. Sources: `crates/jope-pipeline/src/search_context.rs:278`; pending `crates/jope-policy/src/context_producer/classification.rs:1`.

At source `68f56404`, classification has no neural arm: planning records the accepted regression-only policy as unavailable for binary/multiclass data. The available `asm.generic` classification sampler uses `logit_ridge` for binary or `multiclass_ridge` for multiclass, with l2 in [1e-4, 10], preceded by no transform, `standardize`, `winsor_clip`, or both. Advanced-v3 mutation/crossover preserve that head family and allow at most 4 transforms. A classification ASM result at this source would therefore be a ridge-head transform search; the pending experimental neural consumer is a separate capability.

<a id="c-architecture--the-accepted-regression-model-lineage-b-step-60000"></a>

## 2.8 The accepted regression model: lineage B, step 60,000

<a id="c-architecture--checkpoint-binding-and-configuration"></a>

### 2.8.1 Checkpoint binding and configuration

The accepted lineage is the recovered lineage-B step-60000 checkpoint. The binding verifies the state, original model, exported tensors, configuration, parameter digest and vocabulary. The rejected step-69000 identity is explicitly refused. A self-consistent receipt around unrelated weights cannot impersonate the accepted model. The model loader rejects missing, extra, incorrectly shaped or non-FP32 tensors.

| Accepted-model property | Value or behavior |
|---|---|
| Dataset context (historical name of `dataset_dim`) | 1536 FP32 values: Hyperion 512 from features only plus Foundation 1024 conditioned on the training targets, so not label-free |
| Target context | 512 FP32 values; all 512 retained by the configuration (`target_keep_dim` 512), although most of them may be constant zero (`python/src/jope/policy/train_v3.py:919-920`) |
| Combined context | 2048 FP32 values; 8192 bytes in little-endian vector form |
| Decoder width | 512 |
| Decoder depth | 6 layers |
| Attention heads | 4 |
| Feed-forward width | 2048 |
| Token vocabulary | 401 output tokens |
| Embedding rows | 657, including the configuration's 256 extra rows |
| Maximum configured sequence | 128 tokens; positional table includes the context position |
| Training dropout field | 0.1; disabled for inference |
| Reward ensemble | 4 heads |
| Tensor inventory | 136 tensors for this configuration, including unused inference heads and scalar loss weights |
| Current Rust arithmetic | Candle CPU FP32 |
| Native CUDA BF16 policy parity | Not established by the current CPU inference implementation |

Sources: `crates/jope-policy/src/config.rs:21`, `config.rs:39`, `crates/jope-policy/src/model.rs:1`, `model.rs:41`, `crates/jope-policy/src/lib.rs:133`.

<a id="c-architecture--decoder-and-reward-architecture"></a>

### 2.8.2 Decoder and reward architecture

The decoder and reward paths each encode context with a two-layer MLP: 2048→512, GELU, LayerNorm, 512→512, GELU, LayerNorm. The decoder prepends a projected context token to the token embeddings, adds positional embeddings, applies causal self-attention with padding-key masking through pre-normalized layers, normalizes the result and projects through the tied token embedding table narrowed to the output vocabulary.

The reward path averages the program-token embeddings over non-padding positions, with no positional embedding, combines that mean with a separately encoded context, and evaluates four reward MLPs. The combined representation contains the program embedding, its elementwise interaction with a context gate, and an absolute-difference block. A separate runtime head emits a positive runtime estimate through softplus. Learned reward is a proposal-ranking signal; it is not a measured dataset loss.

**Reward pooling is order-blind.** The mean has no positional input. For one context, two programs whose tokens form the same multiset, for example the same operators or parameter values in a different order, receive the same pooled representation and therefore the same four reward scores, up to floating-point summation order. The decoder still conditions on token order; only the learned ranking signal is affected. At this source that signal orders the neural pool's slots and records `rank_log_weight`; it does not select programs (step 3 of [§](#c-architecture--the-learning-cycle-what-is-proven-and-what-remains)). The pooling is the same in the Candle port (`crates/jope-policy/src/model.rs:14`, `:555-567`), in the native tch model it mirrors (`crates/jope-engine/src/neural.rs:496-505`), and in the Python `JopePolicyModelV3` that the pending campaign trains (`python/src/jope/policy/model_v3.py:391-404`; the stopped overlay's copy keeps it). The Python module docstring (`model_v3.py:7-9`) still describes a bidirectional reward encoder with attention pooling; the code discards that encoder, recording that it collapsed (`:258`), and mean-pools instead.

Native source also contains an isolated experimental `OrderAwareRewardEncoder` (`parallel.reward_program_attention_2x256x4.v1`), which adds positional embeddings and two attention layers. Its source comment states that production `Policy` does not construct it and that a separately qualified reranker must opt in with a new artifact; at this source only its unit tests use it (`crates/jope-engine/src/neural.rs:180-263`; `crates/jope-engine/tests/reward_sequence.rs:7-24`). Status: implemented but unqualified.

**Owner rule (proposed): measure before changing the architecture.** First measure how often distinct candidate programs collide on one token multiset, and the ranking regret those collisions cause. Defer architecture changes, including adopting an order-aware reward input, so that the first causal learning experiment keeps changes to architecture, objective, corpus and sampler distinguishable.

<a id="c-architecture--accepted-identities-and-the-product-restriction"></a>

### 2.8.3 Accepted identities and the product restriction

The source records these accepted identities:

| Identity | SHA256 |
|---|---|
| State | `8ac80c269cc47c253b6f53fa3bc1eb75bb2fcda0bdace3ff8554be0175c400d6` |
| Original model | `053df4f53b84d2ebe7c14fbf907a149760af3d11de4a57b5ff89be8c2d3e8129` |
| Canonical parameter values | `9c7057e7f6a9e348f030da73f03522977b8c08915d3a221d7d5ab2e4213fcd92` |
| Configuration | `2508fc8b8268ea000141c9b119ba3da4e777c506c366168912a88d03482939bf` |
| Original checkpoint vocabulary | `8d55fee4366acd2c175cf55bfd769cebc904770f5227a6fd50a613723dc85787` |
| Executable product restriction | `570b2e5e8d859a73b888f5d0bee27f59da0e2376536babce24c241604f818e81` |

The product restriction preserves the 401-token table while limiting executable heads. It removes ten head operators (`extra_trees`, `gam_boost`, `logit_ridge`, the four multiclass heads, `portfolio`, `ridge_gcv` and `tab_boost`), and the vocabulary has no operator token for `pls` or `hurdle2`. The regression proof bound both arms to this restricted vocabulary, and mutation replaces a head only with `histgbm`, `ridge`, `ridge_gcv`, `elasticnet`, `histgbm2`, `gpr` or `ksmooth`; crossover keeps a parent's head. Six of the seventeen registered regression recipes (`tab_boost`, `extra_trees`, `gam_boost`, `pls`, `hurdle2` and `portfolio`) are therefore unreachable by any ASM operator at this source, and `ridge_gcv` is reachable only through mutation. Sources: `crates/jope-engine/assets/grammar/featmath-jain-v1.json` (`restriction.removed_ops`); `crates/feat-math/src/evolve.rs:29-39`, `:1480-1492`; `crates/jope-bench/src/panel/optimize.rs:254`; `crates/jope-bench/src/panel/methods.rs:60-89`. A token-compatible restriction is different from inventing a new vocabulary or silently snapping parameters to nearby bins.

<a id="c-architecture--model-lineages"></a>

## 2.9 Model lineages

A lineage is a chain of checkpoints sharing one architecture and parameter ancestry. A higher step number, a newer file or a shared vocabulary does not place a checkpoint in the accepted line. The table covers three lineage-B checkpoints and the planned experimental line, as of 2026-09-23 about 23:55 UTC.

| Checkpoint or line | Architecture | Role and status | Identity and sources |
|---|---|---|---|
| Lineage B, step 60,000 | Native tch `model.ot`: width 512, six layers, four heads, feed-forward 2048 | The accepted regression model and the only identity the current consumer admits. Available. | State `8ac80c26…`, pinned with its model, parameter, configuration and vocabulary hashes (`crates/jope-policy/src/lib.rs:133-150`, `:161`; `crates/jope-policy/src/config.rs:289-325`); [§](#c-architecture--checkpoint-binding-and-configuration) |
| Lineage B, step 60,100 | Same as step 60,000 | Historical hardware continuation: the September 14 qualification run carried AdamW for 100 updates from step 60,000 on the unchanged corpus, with zero fresh programs, at world 1 × microbatch 256 × accumulation 8. It is neither a serving model nor a production continuation, and the accepted binding refuses any step other than 60,000. | Final state `745be4b5…`; [§](#c-training-history--the-rtx-step-60100-qualification-run) |
| Lineage B, step 69,000 | Same as step 60,000 | Endpoint of the last retained A100 block, steps 66,001–69,000. Unaccepted: an export receipt naming its state or model hash is refused before any other check. | State `58c4a1f6…`, model `f3694654…` (`crates/jope-policy/src/lib.rs:152-159`, `:249-253`; `crates/jope-policy/src/config.rs:177-182`); [§](#c-training-history--a100-checkpoints-and-the-accepted-step-60000) |
| Experimental regression lineage v1 (256×2, new initialization) | Python `JopePolicyModelV3`: width 256, two layers, four heads, feed-forward 1024 | The line the stopped training64 campaign would train, starting from a new initialization. No real campaign has run with it. Pending integration. | Stopped patch `595824ef…`; [§](#c-training--what-the-pending-campaign-trains) |

The two architectures are not interchangeable. The Python path cannot load step 60,000, and the accepted loader rejects the smaller model's tensor shapes, so lineage v1 is a new line rather than a continuation of lineage B. Compare the two only through matched downstream fits under one protocol, never through loss values ([§](#c-training--what-the-pending-campaign-trains)). Training or exporting a lineage-v1 checkpoint does not move the accepted serving pointer ([§](#c-training--plan-run-resume-verify-and-export)).

Historical lineage A and the legacy Python gen31 and round-1 runs are recorded in [§](#c-training-history--dated-lineage-and-evidence-coverage); the current accepted consumer admits none of their checkpoints.

Owner naming rules for lineage v1 (proposed):

- The new initialization is `M_init`. It is named `M0` only after it has actually trained on its first frozen corpus, `C0`, and that run has passed the pending campaign's verification, which includes positive optimizer updates, changed parameter digests, exact reload, an actual resumed optimizer state, increasing step counters and at least two committed passes ([§](#c-training--plan-run-resume-verify-and-export)).
- Passes through one corpus are continuation, not generations: the pending plan's two passes through `C0` (`--passes 2`) do not make two generations. A new generation needs the LOOP-01 chain: a new eligible target, actual exposure, changed parameters, compatible reload and improved fitted search.
- The pending corpus request can name an optional Python initialization checkpoint ([§](#c-training--build-the-real-corpus-from-admitted-evidence)); the campaign then uses a weight warm start, and otherwise a cold initialization. This manual retains cold initialization as the plan for lineage v1. Warm-start support is an unselected interface option, not a qualified lineage start; the owner has not ruled whether lineage v1 admits a warm start, and no prohibition is inferred. Before any admitted run, record which form `M_init` actually took.
- Step 60,100 remains historical hardware continuation and step 69,000 remains unaccepted; neither seeds lineage v1.


---

# Chapter 3 — Installation, preparation, assets, workers and doctor

<a id="c-installation--scope-and-command-surface"></a>

## 3.1 Scope and command surface

<a id="c-installation--product-checkout-and-cli-evidence"></a>

### 3.1.1 Product, checkout and CLI evidence

Algorithm Synthesis Model (ASM) means **“learns to generate and optimize executable algorithms from data.”** The product is the private `veox-ai/JopeDime` repository. On this host its canonical checkout is `/home/ubuntu/JopeDime`; the similarly named research repository at `~/jain-split/jope-dime` is a different upstream.

This chapter is pinned to source `68f56404c3e199e06413bfd645d3fe0803d7b96f`, the reviewed PR83 head. PR83 merged it on 2026-09-23 at 16:34:47 UTC as main `a75e543d2166…` with the same Git tree (`176b75bc36d7…`); as of 2026-09-23 22:00 UTC no later change had merged to main. Sixteen actual version/help invocations against its immutable CLI completed successfully. The executable reports `jope 0.1.0`; its SHA256 is `eab2d5b1b05629020aab3d540d51eb24ba0db18ff0917da188f89a34b6dc5e2b`. The machine-readable evidence is `agent-maintainer-manual/cli-help-68.json` under the ASM custody root (`C` in [§](#c-installation--physical-layout-and-identities)); the `68` in its name abbreviates source `68f56404`. A version string alone cannot distinguish later builds; use the full source, executable digest and build receipt.

The source inventory, a help page, an executed fixture, exact-head CI and a live operational qualification establish different things. This chapter records interfaces and operational requirements. It does not turn an unexecuted command example into measured capability. The maintainer-private examples were executed at the same source: all fourteen commands passed. Their separate receipt and scope limits appear in [§](#c-inference--measured-execution-of-the-private-examples) and `data/example-acceptance/`.

<a id="c-installation--command-surfaces-and-the-stopped-integration-order"></a>

### 3.1.2 Command surfaces and the stopped integration order

| Surface | Present in the immutable 68f56404 CLI | Remaining distinction |
|---|---|---|
| `doctor` | Yes | Static capability inventory; it does not inspect every runtime dependency or qualify a live campaign. |
| `benchmark verify-manifest`, `verify-inputs`, `methods`, `plan`, `run`, `report` | Yes | Planning performs zero fits. Run/report require exact admitted resources, inputs and journal identities. |
| `optimize capabilities`, `proof-plan`, `plan`, `run`, `report` | Yes, execution feature-gated | A compiled ASM feature is separate from qualified encoding, genuine neural-origin fits and successful evolution. |
| `fit`, `emit`, `search`, `pipeline run` | Yes | Legacy numerical regression/recipe paths; their outputs are not complete frozen-panel fitted inference exports. |
| `encode`, `champions`, `train-step`, `route` | Yes | `train-step` trains a local centroid router, not the bounded neural policy campaign. |
| `datasets`, `models` | Yes | Existing catalog/model tools; a fetchable file is not automatically admitted as an ASM input. |
| `train prepare-inputs`, `plan`, `run`, `resume`, `export`, `verify` | Pending combined training overlay | Includes native admission, actual updates/reload/continuation checks and experimental exports. See [§](#c-training--the-bounded-training-campaign-chain). |
| `demo` and private CLI installer | Pending product overlay | Includes generic fitted demos and authenticated paired ASM demos; actual speed and parity must be measured. |
| `benchmark export-winner`, `qualify-export` | Pending winner add-on | Preserves the selected native state and ranking; backend qualification needs actual supervised execution. |
| `motifs mine`, `validate` under the umbrella CLI | Not present at 68f56404 | Separate preparation exists; production eligible-State reading, mine command and umbrella integration were identified as remaining work. |

The stopped integration order is combined **training64** (including regression consumption, accounting and coverage), then **product28**, then unchanged **warm-target** and **winner-export** add-ons. A later two-file regression proof-gate patch (`0e634858…` under `agent-training-proof-gate`, stopped on 2026-09-23 at 19:30 UTC) comes after them: generate the training, product and add-on replays before applying it, because the downstream guards pin every training postimage. The whole queue waits until the manual's own publication lifecycle closes ([§](#c-maintenance--updating-the-book)). The add-on refresh is pinned under `agent-delivery-refresh-68f56404`; two strict replays produced identical nine-file postimages, with 95 canonical guards. These are custody patches, not installed behavior. Never apply the training coverage/accounting constituents a second time or copy an older loose installer postimage over its corrected form.

<a id="c-installation--physical-layout-and-identities"></a>

## 3.2 Physical layout and identities

Keep source, bulk inputs and evidence in separate locations. The examples below use these names; resolve them from actual receipts before execution:

```sh
R=/home/ubuntu/JopeDime
C=/home/ubuntu/jope-rescue-20260921/asm-integration-20260923
D=/home/ubuntu/.cache/jope/asm-private-30-v1
M="$R/benchmarks/panel.json"
# S: full reviewed source SHA; J: physical immutable CLI path.
# A: new durable private acceptance directory, outside Git and /tmp.
```

`JOPE_HOME` chooses the source/assets root; `JOPE_STATE` chooses mutable local state; `JOPE_REGISTRY` chooses the dataset registry. Those roots are resolved at call time. Pass explicit paths when producing reproducible evidence. Do not repurpose `HOME` or globally override `CARGO_TARGET_DIR`. A managed installation symlink is convenient for interactive use; a campaign pins the physical executable path and checksum so changing the symlink cannot silently replace its worker.

There are two different hashes in the CLI contracts. An artifact SHA256 hashes exact file bytes. A campaign's `plan_sha256` is its validated semantic identity, printed by planning. Do not substitute `sha256sum plan.json` for that semantic ID. Worker/build/source inventories have their own schemas and pins; preserve each rather than renaming one receipt into another type.

Campaign journals require absolute durable paths outside Git working trees and `/tmp`. Private directories are restricted by the front doors. Do not publish entire run directories: they can include raw development features, labels, recipes, private context, model state and provenance.

<a id="c-installation--build-and-install"></a>

## 3.3 Build and install

<a id="c-installation--development-build-of-the-cli"></a>

### 3.3.1 Development build of the CLI

The current development build path builds the Rust CLI from the existing checkout:

```sh
cd "$R"
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only \
  cargo build --locked --release -p jope-cli --features asm --bin jope
```

Allocate build/test resources through the repository's existing coordination before executing it. Use one reused `target/` per claimed checkout. The CLI's `asm` feature exposes the private Linux execution path; a default-feature build may show the command while reporting ASM execution unavailable. The ordinary CLI should not require the monolithic engine, Torch or CUDA. Separate context and policy workers and their authenticated model inputs are still necessary for a live neural run.

<a id="c-installation--private-installer-after-integration"></a>

### 3.3.2 Private installer after integration

After the pending installer is integrated, prefer its measured source-build and checksum-install front doors:

```sh
python3 "$R/deploy/install-private.py" source \
  --checkout "$R" --source-sha "$S" --output "$A/cli-build" --jobs 4
python3 "$R/deploy/install-private.py" binary \
  --binary "$PRIVATE_BINARY" --sha256 "$BINARY_SHA256" \
  --source-sha "$S" --prefix "$A/install"
```

Read the binary path and digest from the actual source-build receipt. The source command checks the checkout before and after compilation, uses the same `asm` selection for dependency inventory and build, rejects unwanted engine/Torch/CUDA dependencies, and selects the executable from Cargo artifact messages. Its receipt includes the toolchain, dependency graph, features, source identity, compiler output, executable digest and measured cold duration. Compilation failures retain failure receipts.

The binary installer requires Linux x86_64, Python 3.10 or later and `readelf`. It validates the supplied checksum before executing the binary, checks architecture/glibc/dynamic dependencies, probes version and command help, and activates a digest-addressed private binary through a managed symlink. It does not download models or datasets. A caller-supplied checksum proves byte identity; retain independent source/review/build provenance as well. Cold build time, cold installation time, data preparation and warm demonstration time are separate measurements.

The installer's `jope.private-cli-source-build/v1` receipt does not replace a `jope.asm-cargo-build/v1` worker receipt or `jope.asm-worker-build-inventory/v1`. Build and retain the context/policy workers, source manifest, accepted assets and native context qualification through their owning workflow. The native canonicalizer used by experimental proposal admission must match the final source/build and physical executable bytes.

<a id="c-installation--prepare-the-panel-and-external-runtimes"></a>

## 3.4 Prepare the panel and external runtimes

<a id="c-installation--prepare-and-verify-the-panel"></a>

### 3.4.1 Prepare and verify the panel

The approved panel contains ten regression, ten binary and ten multiclass datasets. Its exact manifest SHA256 is `ba7dace4cd535687002cc95538dbc7f7aaf77ec973177c7159b3c6d22471eee0`. Reformatting the JSON changes its bytes and is refused even when its apparent meaning is unchanged. Do not replace a failed dataset with an easier fixture.

```sh
uv run --project "$R/python" --extra benchmark python "$R/benchmarks/prepare.py" \
  --manifest "$M" --output "$D"
"$J" benchmark verify-manifest --manifest "$M"
"$J" benchmark verify-inputs --manifest "$M" --datasets "$D"
"$J" benchmark methods
```

Preparation must preserve actual frozen row assignments, original labels, category strings, nulls and groups. `verify-inputs` verifies development bytes and schema, not just file existence. The benchmark does not open the sealed evaluation partition. An operational proof of one regression family retains the original thirty-dataset manifest but selects a separately declared proof scope; it does not modify the official panel.

<a id="c-installation--native-roster-and-external-comparator-runtimes"></a>

### 3.4.2 Native roster and external comparator runtimes

The native roster has 17 regression recipes, five binary recipes and four multiclass recipes. Aliases do not create extra screening slots. The second GPR configuration is distinct because it has a distinct fitted implementation. Every task also includes four external adapters, each run at library defaults with seed 0 and one thread: XGBoost, LightGBM, CatBoost and a scikit-learn random forest. Native recipes are fixed before the dataset screen and the external adapters keep those defaults, so the campaign performs no per-dataset hyperparameter search for any non-ASM comparator (a head's own internal fitting, such as `ridge_gcv`'s generalized cross-validation, is part of its fixed recipe; [§](#c-benchmark--minimum-complete-future-campaign-report)), while each ASM arm searches within its declared allowance. The highest-rated methods on the historical benchmark-valid ladder, such as TabFM Ensemble, are not panel comparators ([§](#c-targets--sotarena-published-and-corrected-ladders)); panel wins against this roster would therefore not show that ASM beats those methods. Sources: `crates/jope-bench/src/panel/campaign_cli.rs:67`; `crates/jope-bench/adapters/sklearn_rf_adapter.py:2`; `crates/jope-bench/src/panel/selection_exposure.rs:91`; `crates/feat-math/src/heads.rs:65-67`. External library versions, entry points, environment and package bytes must be authenticated before admission.

```sh
python3 "$R/deploy/prepare-panel-runtime.py" \
  --binary "$J" --native-command benchmark \
  --build-receipt "$CLI_BUILD_RECEIPT" --source-sha "$S" \
  --python "$EXTERNAL_PYTHON" --site-packages "$EXTERNAL_SITE_PACKAGES" \
  --adapters "$R/crates/jope-bench/adapters" --output "$A/runtimes.json"
"$J" benchmark plan --manifest "$M" --datasets "$D" --source-sha "$S" \
  --runtimes "$A/runtimes.json" --output "$A/base.json"
```

The preparation helper checks actual dependencies/constructors and pins the runtime. It is not a completed fit comparison. Retain unavailable-dependency reasons rather than dropping methods. Prior inventories observed XGBoost 3.4.1, LightGBM 4.7.0, CatBoost 1.2.10 and scikit-learn 1.7.2 on this host; these historical metadata observations do not authenticate a later environment. A new package installation invalidates old package-byte inventory. Keep the external CPU environment separate from the dedicated CUDA training environment.

Omitting `--runtimes` from `benchmark plan` produces a verified Rust screening inventory with zero fits; it is not the complete admitted ASM/comparator campaign. The runtime-aware plan is the base for `optimize plan`.

<a id="c-installation--stored-model-assets-and-the-active-encoder-distinction"></a>

## 3.5 Stored model assets and the active encoder distinction

<a id="c-installation--bundle-manifests-and-declared-asset-sizes"></a>

### 3.5.1 Bundle manifests and declared asset sizes

The metadata-only S3 audit on 2026-09-23 fetched the three bundle manifests below and pinned their returned bytes, ETags and lengths. No model bodies were downloaded for this chapter. These are declared object sizes from the retrieved manifests; they do not measure resident memory, compute cost or a new numerical qualification.

| Bundle asset | Bytes | Role in this chapter |
|---|---:|---|
| `policy-step-60000/policy.safetensors` | 111,199,088 | Accepted policy export identity, consumed only with its checkpoint binding |
| `policy-step-60000/context_transforms.safetensors` | 8,413,808 | Accepted fixed context assembly transforms |
| `jope-prime-gen31-v2/model-bundle/hyperion_v5_arm_b.safetensors` | 6,650,692 | Exact Hyperion weights pinned by the current CPU context producer |
| `jope-prime-gen31-v2/starforge/starlight_core_regressor.safetensors` | 233,183,532 | Foundation component; manifest describes512 hidden features pooled into2048 |
| `jope-prime-gen31-v2/model-bundle/target_encoder_fused_v1…safetensors` | 78,615,176 | Target component; manifest describes normalized64+64+384 fused representation |
| `hyperion-v7/model.safetensors` | 246,456,988 | Separate stored bundle, not the accepted context producer's pinned weights |
| `jope-prime-gen31-v2/model-bundle/jope_policy.safetensors` | 108,846,812 | Separate gen31 policy artifact; not step-60000 |

The three current model bundle manifests are `hyperion-v7`, `jope-prime-gen31-v2`, and `policy-step-60000`. Their existence and object sizes do not establish new training, inference parity or serving promotion. [§](#c-status--the-identity-tuple) gives their roles among the product identities, and [§](#c-maintenance--release-and-installation-are-independent-from-model-promotion) gives the acceptance boundaries. Historical lineage-A material in learning storage cannot replace its missing canonical training-state chain.

<a id="c-installation--encoder-identity-pitfalls-and-provenance-declarations"></a>

### 3.5.2 Encoder identity pitfalls and provenance declarations

The storage name `hyperion-v7` must not be confused with the CPU implementation's `V7Model` Rust type. The actual current producer pins weight hash `88b9e834…`, which matches the v5 arm-B asset, with row width 128. The separate stored v7 manifest describes dimension512, eight heads,12 layers and six bands. Replacing the pinned asset based on its newer-looking name changes the encoder identity and invalidates compatibility.

Two identity pitfalls matter when maintaining this boundary. Hyperion `load_settings` reports a legacy 16-hex identifier, not a SHA-256 digest. The producer therefore authenticates manifest bytes before and after the path-based settings load, authenticates weights before and after `V7Model::load_cpu`, and requires embedding width 128. A legacy identifier must never be compared with or substituted for a cryptographic admission pin. Separately, ordinary `jope encode` resolves `HYPERION_V7_WEIGHTS` as a **bundle directory**, else `$JOPE_HOME/models/bundles/hyperion-v7` (relative to the working directory when unset). The default stored bundle is 512-dimensional and differs from the pinned v5-arm-B producer; both manifests use the loader string `version: v7`, so that string cannot establish checkpoint identity. `jope encode` is not the qualified 2048-value ASM context interface. Current documentation drift remains at `docs/models.md:105`, which supplies a file instead of the expected directory, and the default-v5 comment in `crates/hyperion/src/hyperion.rs:28`. Use exact asset hashes and the qualified context-worker flow.

The retrieved manifests also contain internal/redistribution/provenance declarations that vary by artifact. They are retained evidence about the bundle metadata, not proof that every object is approved for every distribution. The active private run's exact release/input inventory remains the authority for which files it used; S3 presence alone does not establish serving eligibility. Sources: `crates/jope-policy/src/context_producer.rs:41`; custody `metadata-receipts/veox-jopedime-models-use2--bundles--policy-step-60000--manifest.json`, `…--hyperion-v7--manifest.json`, and `…--jope-prime-gen31-v2--manifest.json`.

<a id="c-installation--bounded-workers-the-context-worker-and-the-policy-producer"></a>

## 3.6 Bounded workers: the context worker and the policy producer

<a id="c-installation--context-worker-contract"></a>

### 3.6.1 Context worker contract

In the bounded ASM flow, encoder weights are loaded inside `jope-context-worker` (`jope-policy`, feature `context-producer`). `--version` prints `jope-context-worker/1`. Its exact eight named arguments are:

```text
jope-context-worker --mode produce|qualify-case --input ABSOLUTE_PATH --input-sha256 HEX64 --assets ABSOLUTE_ENCODER_FILES_JSON --assets-sha256 HEX64 --source-sha40 HEX40 --producer-sha256 HEX64 --output ABSOLUTE_PATH
```

Extra arguments such as `--dataset-root` are refused. The child requires OMP, OPENBLAS, MKL, BLIS, NUMEXPR and RAYON thread limits of 1, `VECLIB_MAXIMUM_THREADS=1` and `CUDA_VISIBLE_DEVICES=-1`; it authenticates its own executable and the component closure before model loading. The pinned bundle manifest begins `5e7884ec…`; each inventoried file/notice is verified with a 512 MiB per-file limit. Outputs are create-new, mode 0600 and fsynced. Stdout records `jope-context-worker/1 <output_sha256>`. A failed `qualify-case` comparison retains `<stem>.unqualified.json`, exits nonzero and does not create the requested successful output.

For current `jope optimize`, input derives from `dataset.cut(0)` fitting rows and is retained at `<journal>/asm-context-inputs/<input_sha256>.json`. Admitted output is `<journal>/asm-contexts/<neural-arm-id>/context.f32` plus `bundle.json`; the vector contains 2,048 little-endian float32 values. Invoke this through the supervised planner/controller contract, not as an unsupervised shortcut.

<a id="c-installation--policy-producer-contract"></a>

### 3.6.2 Policy producer contract

`jope-policy-producer` is a separate pinned bounded job. Its `asm.private-policy-producer/v1` request binds worker source, prepared/training/context hashes, cut, accepted checkpoint state, policy/vocabulary hashes, three RNG seeds and sampling shape. It accepts only the pinned step-60000 weights (`4b4b5a40…44f1e`) and state, rounds 1–8, batch 1–128 and a finite positive temperature. Context input is exactly 8,192 bytes, finite and clipped within [-32,32]. It requires either a qualified CPU context bundle or an authenticated original `parallel.encoding.v1` receipt with the original LibTorch lineage, train-side partition and `final_test_targets_used:false`. Frozen transforms are pinned to `1b779c6f…6e22`.

Its output must be an absolute physical-parent path outside `/tmp`, created new with mode 0600. The final stdout JSON declares `numerical_backend:candle_cpu_float32` and `native_cuda_parity_qualified:false`. Reading the envelope proves consistency, not execution: consumption also requires the completed pinned worker/output receipts. Replacing accepted weights, transforms or vocabulary is not a free configuration substitution.

The producer shares the 30-second worker cap and original 120-second neural allowance; the accepted proof used a 3 GiB worker allocation. The pre-cache proof at source `55e9ed688faf` was killed at 30.004805335 seconds, before neural fits. With cached decoding, later measured producer wall times were 20.832133058 seconds at source `3fd7d5b2ebb6`, 20.651920435 at `c5d11ffca92c` and 20.486968041 at `68f56404c3e1`. These leave about 9.5 seconds of per-job headroom in the last measurement, not a portability guarantee. Sampling-shape or host changes need another bounded proof, and the fixture-only decoder-equivalence limits of [§](#c-architecture--component-dimensions-and-assembly) still apply.

Sources: `crates/jope-policy/src/context_producer/worker.rs:28`, `:188`, `:255`; `crates/jope-policy/src/producer.rs:16`, `:91`; `crates/jope-policy/src/bin/jope-policy-producer.rs:13`, `:78`, `:121`, `:172`; `crates/jope-bench/src/panel/optimize.rs:454`, `:705`; retained `proposal-execution.json` receipts.

<a id="c-installation--doctor-checks-and-troubleshooting"></a>

## 3.7 Doctor checks and troubleshooting

<a id="c-installation--reading-identities-from-the-immutable-cli"></a>

### 3.7.1 Reading identities from the immutable CLI

These metadata commands do not train or load a checkpoint:

```sh
R=/home/ubuntu/JopeDime
C=/home/ubuntu/jope-rescue-20260921/asm-integration-20260923
J="$C/neural-integration/live-build-68f56404c3e199e06413bfd645d3fe0803d7b96f/bin/jope"
"$J" --version
"$J" doctor
"$J" optimize capabilities
git -C "$R" rev-parse HEAD
sha256sum "$J" "$R/Cargo.toml" "$R/python/pyproject.toml" "$R/benchmarks/panel.json"
```

Do not interpret `qualified: false` in capability metadata as a missing command, or compile-time availability as completed runtime qualification. The measured executable SHA is `eab2d5b1b05629020aab3d540d51eb24ba0db18ff0917da188f89a34b6dc5e2b`; a different executable needs its own receipt.

<a id="c-installation--troubleshooting-and-safe-recovery"></a>

### 3.7.2 Troubleshooting and safe recovery

| Symptom | Meaning and next action |
|---|---|
| `compiled: false` or ASM feature unavailable | Use the reviewed Linux CLI built with `asm`; then separately qualify its model inputs and workers. |
| Running binary differs from pinned native runtime | Resolve the installed physical binary, verify its hash and prepare a new pre-execution inventory/plan; do not alter a running plan. |
| Frozen manifest or development checksum differs | Stop admission, preserve both files and find the preparation/source mismatch. Do not regenerate folds or substitute datasets to make a check pass. |
| Missing external dependency/compiler | Retain unavailable status and inventory. Install/qualify before opening a new authorized experiment; never drop the method or replace a winner. |
| Context/reference/worker identity mismatch | Rebuild or requalify through the owning source-bound workflow; a copied success flag is insufficient. |
| Deadline expired | Recover/report the original window. A repeated command cannot renew it. Any new experiment needs its own explicit scope and resource declaration. |
| Started job lacks a finished receipt | Treat execution cost/status as unknown; let the shared pool perform terminal recovery/drainage. Do not blindly retry the child or delete its artifacts. |
| No complete five-fold winner | Keep incomplete/failed confirmations visible. Do not select a different recipe merely because it has an emitter. |
| Warm result exceeds 60 seconds | Preserve fitted artifacts, comparisons and the measured miss of the 60-second goal. Do not relabel elapsed time or omit preparation performed inside the warm invocation. |
| S3 access stalls at instance metadata | On the host, select the configured credential-process profile such as `AWS_PROFILE=veox-jepson-prod`; never source the bare-token credential note files. Runners have no AWS credentials. |
| No eligible training families/no-signal admission | Preserve refusal; obtain real eligible data under the declared protocol before GPU training. Do not use development/test gain as a substitute for the train-side OOF admission rule. |


---

# Chapter 4 — Data, splits, groups, families and permitted input boundaries

Observation date: 2026-09-23T16:41:48.371336+00:00. Frozen source: `68f56404c3e199e06413bfd645d3fe0803d7b96f`, the reviewed PR83 head; `main` commit `a75e543d2166` has the same tree. These are observations at that source and time. They do not turn stored metadata, a source packet, or historical results into a current operational qualification.

Every number in this chapter comes from a pinned manifest, prepared development table, immutable report, committed historical snapshot, filesystem metadata, or authenticated AWS metadata request. The audit did not run a fit, load a model, open sealed target bodies, or execute a report command.

<a id="c-data--the-frozen-thirty-dataset-panel-identity-totals-groups-and-immutability"></a>

## 4.1 The frozen thirty-dataset panel: identity, totals, groups and immutability

Panel `asm-private-30-v1` uses ten regression, ten binary, and ten multiclass datasets from PMLB revision `7c1f4bdc00136dc2e55c87fa6b8ba6e8af6d1a68`. Manifest SHA-256: `ba7dace4cd535687002cc95538dbc7f7aaf77ec973177c7159b3c6d22471eee0`. It was selected before measuring ASM outcomes, excludes the historical favorable test-10 collection, and cannot replace a dataset because a method failed or lost.

There are **55,029 total rows**, **44,048 development rows**, and **10,981 sealed rows** according to the frozen assignments. Feature counts sum to 522 across datasets; this is not a union of unique semantic variables. Development data contains 1,612 missing feature cells in 302 rows. All thirty development, split-assignment and metadata file SHA-256 values match the frozen manifest. Every row ID is unique within its dataset, and every group stays wholly within one original development fold or sealed partition.

The panel is immutable once comparison begins. Its identity covers dataset metadata, original labels and types, row IDs, group membership, original development folds, sealed membership and prepared data hashes. Do not replace difficult datasets, reshuffle rows, map categories to arbitrary numeric order, drop missing rows or substitute a different target to make a task executable. Missing source subject or time identifiers remain limitations even when duplicate-feature groups are protected.

Known date columns and identical-feature groups are retained by the prepared split. Missing original subject/time identifiers remain a limitation. A grouped split cannot establish independence when the source never supplied those identifiers. Checkpoint exposure is unknown for these datasets; an unseen-family generalization claim is not supported. Pretrained-encoder exposure is also unknown, and the panel remains an adaptive development domain ([§](#c-data--three-family-level-domains)).

<a id="c-data--local-preparation-and-panel-storage"></a>

## 4.2 Local preparation and panel storage

Local preparation: `/home/ubuntu/.cache/jope/asm-private-30-v1/`. `development/` contains the permitted development tables; `splits/` contains original row/group/fold assignments; `metadata/` contains pinned upstream YAML; `downloads/` contains the original compressed archives; `sealed/` contains withheld partitions. The audit opened development, assignment, and metadata files only. It inspected sizes and expected hashes of raw and sealed files without opening their bodies.

| File class | Logical bytes | Content read in this audit |
|---|---:|---|
| development | 17,923,164 | Yes; metadata/development statistics and SHA-256 |
| metadata | 82,681 | Yes; metadata/development statistics and SHA-256 |
| sealed | 4,518,613 | No; size and expected identity only |
| source_archive | 2,075,982 | No; size and expected identity only |
| split_assignments | 9,812,165 | Yes; metadata/development statistics and SHA-256 |

File sizes are logical bytes, not compressed-memory use or total disk allocation. The raw archive sizes sum to 2,075,982 B, matching the frozen declared download byte counts; content hashes of those raw archives were not revalidated here because the archives also contain sealed targets. The prepared sealed file hashes are declared frozen identities, not freshly verified content checks in this audit.

<a id="c-data--the-broader-s3-corpus-catalog"></a>

## 4.3 The broader S3 corpus catalog

<a id="c-data--catalog-composition-by-task"></a>

### 4.3.1 Catalog composition by task

The pinned current catalog is `ab9fda8d2dea46067b70a42812e9d3c1d9dc6ba025780100df34377e81aa1120`: 3,859 distinct dataset hashes, all marked real, 6,208,051 declared rows. Shape metadata sums to 265,105 features and 4,628 categorical features; sums across datasets are not deduplicated semantic quantities. The catalog contains no task explicitly named multiclass; 322 entries are multi-label, which is a different task and must not be silently relabeled.

| Catalog task | Datasets | Total declared rows | Training rows | Test rows metadata only | Feature-count range | Row-count range |
|---|---:|---:|---:|---:|---|---|
| binary | 843 | 1,490,965 | 892,307 | 598,658 | [1, 2000] | [10, 69624] |
| multi_label | 322 | 786,436 | 555,756 | 230,680 | [1, 20531] | [23, 50000] |
| regression | 2,611 | 3,025,922 | 1,898,974 | 1,126,948 | [0, 1688] | [10, 50000] |
| timeseries | 83 | 904,728 | 775,519 | 129,209 | [2, 86] | [81, 49999] |

<a id="c-data--licence-metadata-anomalies-and-the-corpus-figure"></a>

### 4.3.2 Licence metadata, anomalies and the corpus figure

**Catalog discrepancy:** the stored registry collection description says 1,416 unknown licenses. Counting the current catalog’s actual license-status fields gives **3,755 unknown and 104 recorded**. This chapter uses the counted fields and retains the stale description as provenance. All 2 collections remain private: `registry-real-v1` has 3,859 members and `test-10` has 10; every listed member occurs in the catalog. No payload or training eligibility validation was performed simply by counting them.

Seven regression catalog entries declare zero features. These are retained anomalies, not automatically usable supervised tasks: `5ffbe1543f683e4d`, `66fea7ec7dffa02a`, `72235646c52be632`, `7d1497edfe70495d`, `d12486ba35015a09`, `e1c1ff8cdf4ee36f`, `f773853db5b3dc9c`. Corpus metadata lacks feature distributions, missingness, actual class balance and row-level dependence evidence for many entries; those statistics remain unknown here.

The broader catalog has 3,859 real datasets and 6,208,051 metadata-reported rows. It is a different collection from the frozen panel. Its `multi_label` task is not interchangeable with admitted multiclass classification. Per-payload statistics for the entire corpus were not recomputed.

![Corpus task counts and license metadata completeness](figures/inventory-corpus.png)

The catalog has 3,755 entries with unknown license metadata and 104 with a recorded license. The collection's older prose count of 1,416 unknown licenses is inconsistent with this current pinned catalog. This is a metadata quality gap; neither the chart nor a repository-level license resolves each dataset's provenance. See [§](#c-data-estate) for exact storage bytes, prefixes and observation scope.

<a id="c-data--the-historical-input-closure-and-the-meta-test-question"></a>

## 4.4 The historical input closure and the meta-test question

<a id="c-data--record-split-and-the-meta-test-closure-question"></a>

### 4.4.1 Record split and the meta-test closure question

The accepted corpus's record split is precise: 9,382,874 `meta_train` records in 6,836 shards and 831,458 `meta_validation` records in 1,041 shards. There are no `meta_test` record shards. The context index nevertheless contains 6,962 training, 1,054 validation and **878 meta-test context descriptors**. Current legacy `training.rs` stages every context in that index before sampling records. Consequently, withholding meta-test record labels is not a proof that sealed families were excluded from the entire historical encoder/trainer input closure. This is a concrete remaining closure audit, not a claim that this documentation audit opened sealed vectors or labels.

A separate external historical-inventory report also discloses parsing manifests for eight datasets marked `meta_test` **and hashing their source CSVs**; its shorter coordination entry mentioned manifests only. That report says all eight have lineage-A `test_exact` high-water marks. This audit did not reopen those paths or inspect their labels. The disclosure does not establish whether labels reached lineage-B training. Before reuse or continuation under the current policy, audit the complete encoder, context-staging, target-admission and sampler input closure, and resolve the old exposed-HWM lineage. Neither zero meta-test record shards nor an old qualification flag closes that question.

The accepted index SHA `1c840597…`, vocabulary `8d55fee4…` and context index `d77e856b…` bind the preserved checkpoint lineage. Its label policy is explicitly `reference_v1`. An old `qualification: true` flag means qualification under that historical implementation and policy. It does not retroactively satisfy current finite train-side OOF improvement over a trivial baseline, improvement on four of five folds, exact source closure and non-exposure requirements.

<a id="c-data--audit-the-entire-input-closure-before-continuation"></a>

### 4.4.2 Audit the entire input closure before continuation

Before any future continuation, audit the **entire** input closure, including every context descriptor and staged tensor, not only training records. The retained snapshot 97 index lists 8,894 contexts, including 878 `meta_test` descriptors, while record rows are train/validation only. Current legacy training stages all indexed contexts before sampling. Absence of sealed record rows therefore does not prove strict sealed-family exclusion. Do not blindly continue that corpus under the new closure policy; preserve the metadata observation and resolve it without opening sealed payloads. See [§](#c-data--record-split-and-the-meta-test-closure-question) and [context staging source](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-engine/src/training.rs#L1179).

[§](#c-data--record-split-and-the-meta-test-closure-question) records unresolved historical input closure: meta-test context descriptors were staged by the legacy trainer, while the external author disclosed meta-test manifest inspection and source-CSV hashing alongside old exposed high-water marks. Do not treat these findings as proof of training-label leakage or dismiss them because the packed record split excludes meta-test rows. Resolve the complete allowed-input closure and current teacher eligibility before continuing the old corpus under the new policy.

<a id="c-data--family-level-domains-and-the-permitted-input-closure"></a>

## 4.5 Family-level domains and the permitted input closure

This section states the owner's edition-2 rules for which data may reach teaching, training, selection and evaluation. They are requirements, not results. Each carries one status as of 2026-09-23 about 23:55 UTC: available, implemented but unqualified, pending integration or proposed.

<a id="c-data--three-family-level-domains"></a>

### 4.5.1 Three family-level domains

Domains are assigned to dataset families, not to rows, files or split names. A split name alone cannot prove an unseen family ([§](#c-learning--freezing-the-experiment-and-choosing-families)).

| Domain | Permitted use | Status |
|---|---|---|
| Teacher/train | Train-side evidence may produce eligible teachers and policy-training rows | Pending integration: the stopped `training64` packet partitions admitted families into `train` and `development` sides only (pending `crates/jope-cli/src/train/corpus.rs:196-264`) |
| Adaptive development | Labels may guide screening, selection, confirmation and model monitoring; results are adaptive, not independent | Implemented but unqualified: every panel report at `a75e543d2` records `confirmation_role: "adaptive_development"` (`crates/jope-bench/src/panel/campaign.rs:894`); no full panel has run |
| Evaluation outside training and selection | Families whose labels informed no teaching, training, selection or monitoring | Proposed for this ASM path. The legacy engine assigns a sealed meta-test family partition and withholds its observations from corpus records, but the legacy trainer stages all indexed descriptors, including the 878 meta-test descriptors; that does not establish their exclusion from the worker input closure |

The frozen panel `asm-private-30-v1` remains an adaptive development domain. Its development labels drive the screen, freeze and confirm procedure ([§](#c-benchmark--adaptive-development-exposure)), and a panel family that supplies teachers must additionally be reported as exposed ([§](#c-learning--freezing-the-experiment-and-choosing-families)). Because domains are family-level, withholding the panel's sealed rows does not place its families in the third domain. Prior exposure is unknown in two separate places: the accepted policy checkpoint, which `optimize plan` records as `checkpoint_training_exposure: "unknown"` (`crates/jope-bench/src/panel/optimize_cli.rs:128`), and the fixed pretrained encoder assets that produce the proposal context ([§](#c-architecture--component-dimensions-and-assembly)). Pretrained-encoder exposure remains unknown; neither unknown may be reported as zero.

<a id="c-data--hashes-eligibility-and-the-complete-input-closure"></a>

### 4.5.2 Hashes, eligibility and the complete input closure

A hash is not eligibility. A digest authenticates bytes; it does not decide whether those bytes may be used. The legacy trainer at `a75e543d2` checks every indexed context's bytes against its recorded `sha256` digest and expected size, then stages all of them on the device (`crates/jope-engine/src/training.rs:1187-1203`), so the 878 `meta_test` descriptors in the retained index would pass that integrity check ([§](#c-data--record-split-and-the-meta-test-closure-question)). Eligibility comes from each input's admission record and family domain ([§](#c-teachers--the-admission-guard-label-exposure-and-hwm-identity)), never from a matching digest.

The permitted input closure covers everything a teaching or training worker can read, directly or transitively: records, context descriptors and tensors, fitted and frozen transforms, and all assets, including encoder and policy weights, vocabulary, admission manifests and checkpoints. Its parts have different status:

| Closure requirement | Status as of 2026-09-23 about 23:55 UTC |
|---|---|
| Native projection of admitted contexts | Pending integration: the stopped `training64` preparation requires the requested contexts to cover exactly the admitted comparisons and writes only the selected row contexts (pending `crates/jope-cli/src/train/corpus.rs:572-575`, `:600-635`) |
| One authenticated inventory of every permitted worker read and its role | Proposed: the same packet's Python campaign opens the complete context index, encoder, vocabulary and admission file, and validates only the contexts that retained rows reference (pending `python/src/jope/policy/training_campaign.py:298-363`) |
| Rejection of unused forbidden descriptors before loading | Proposed: neither current nor pending source rejects extra context-index entries, and the legacy native route still stages every indexed context |

An unused forbidden descriptor, such as a `meta_test` or evaluation-domain context that no admitted row references, must be rejected before any model, tensor or corpus is loaded; leaving it unsampled is not enough. The tests this requires do not exist yet: an injected unused forbidden context, an unlisted reachable artifact, invariance under mutation of an excluded artifact, and invariance of preprocessing and context under changed holdout labels. Source: the read-only boundary investigation `plan-execution-20260923/boundary-resume-siblings.md` in the ASM custody root.

<a id="c-data--datasets-outside-the-frozen-panel"></a>

### 4.5.3 Datasets outside the frozen panel

Arbitrary ingestion is separate from the frozen panel. Status: proposed. No arbitrary-dataset ingestion command exists at `a75e543d2`: the one-dataset ASM front door takes a dataset already present in the frozen panel ([§](#c-search--front-door-and-safe-metadata-checks)), and a fetchable catalog file is not automatically an admitted ASM input ([§](#c-installation--command-surfaces-and-the-stopped-integration-order)). A separate authenticated ingestion request must preserve the new dataset's labels, categories, missingness and groups and assign its family to a domain; it never edits the frozen manifest ([§](#c-search--what-is-missing-for-a-sustained-incumbent-seeded-dataset-attack)). Results on an ingested dataset are not frozen-panel results, and frozen-panel results do not qualify an ingested dataset.


---

# Chapter 5 — Regression search, fitting, prediction and focused optimization

<a id="c-search--proposal-generation-and-ranking"></a>

## 5.1 Proposal generation and ranking

Accepted proposals are generated under grammar constraints, and the neural pool is the first 64 distinct valid programs in draw order; any shortfall after the bounded sampling rounds is filled with distinct grammar-uniform programs labelled `neural_rejection_fallback`. Ranking orders that pool by the mean of four reward heads, descending; exact ties use program SHA256 then text. Runtime estimates are reported without subtracting an undocumented runtime penalty. Ranking binds the checkpoint, context and vocabulary and records that scoring labels and fitted GP scores were not read. It does not select or reject programs: all 64 ranked programs and the 16 exploration programs enter generation zero. Sources: `crates/jope-policy/src/ranking.rs:22`; `crates/jope-policy/src/decode.rs:9-11`, `:212-225`, `:359`; `crates/jope-policy/src/seeds.rs:257-270`.

Each ranked policy-pool program carries `rank_log_weight`, the log-softmax of its four-head mean predicted reward over that pool. Historical files called this `policy_logprob`. It is a within-context ranking weight, not a sampling/selection probability or propensity; do not use it for importance weighting or off-policy correction. Exploration candidates carry no rank weight.

The four reward heads read a masked mean of the program's token embeddings, with no position information (`crates/jope-policy/src/model.rs:555-567`). Within one context, two programs with the same token multiset therefore receive the same rewards in exact arithmetic, so their relative order carries no learned signal; an exact tie falls to the program-hash tie-break. This does not remove token order from the autoregressive proposal decoder ([§](#c-learning--distinguish-routing-ranking-and-search-recovery)). The owner's rule for this limitation (status: proposed): first measure how often such collisions occur and the ranking regret they cause, and defer architecture changes, including adopting an order-aware reward input, so that the first causal training experiment keeps architecture, objective, corpus and sampler changes distinguishable.

<a id="c-search--paired-arms-generation-zero-and-evolution"></a>

## 5.2 Paired arms, generation zero and evolution

ASM compares a 64-slot policy pool plus 16 exploration generation-zero slots with 80 generic slots. The pool retains `neural_policy` versus `neural_rejection_fallback` origins explicitly. Both arms receive the same declared search and fit allowances, but encoding and proposal work consume the neural arm's own clock; fitting time can differ. A producer refusal, duplicate or failed candidate is an outcome to record. It cannot silently become another neural success or a free replacement arm. Actual inner-fold fits determine selection and offspring ancestry. A model's predicted reward never substitutes for measured loss.

Each arm starts with 80 generation-zero slots. In the neural arm, slots 0–63 form the ranked policy pool: distinct valid checkpoint samples are labelled `neural_policy`; any shortfall after the bounded sampling rounds is filled by distinct grammar-uniform programs labelled `neural_rejection_fallback`. Slots 64–79 are `random_exploration`. Only `neural_policy` roots and completed descendants with verified neural ancestry count as genuine neural evidence; fallback and exploration roots do not. In the origin proof at source `68f56404` all 64 pool slots were `neural_policy`, 48 completed all five inner folds, and no fallback slots were needed. Receipts retain initial-slot origin and descendant ancestry. Actual completed inner OOF losses select the best four parents. New generations alternate mutation and crossover attempts, canonicalize recipes, reject duplicate executable identities, and retain variation attempts in the journal. Source: `crates/jope-pipeline/src/neural_search.rs:27`, `:1380`.

Neither arm is seeded from the other, but at `a75e543d2` only the neural arm's initial population is authenticated against its producer. The neural arm's 80 generation-zero slots come only from the verified policy-producer artifact: the controller verifies the artifact against the worker request, requires exactly 80 slots and refuses any slot whose declared origin does not match its position (`crates/jope-pipeline/src/neural_search.rs:1066-1102`), and report recomputation replays that producer output against every generation-zero genome, origin and rank weight (`crates/jope-bench/src/panel/asm_campaign.rs:516-597`). The generic arm draws its 80 distinct candidates in process from its own grammar stream, seeded by the plan's `random_seed` (`neural_search.rs:1147-1176`). Resume and report replay check that population's slot count, `generic` origins and empty parents, but do not re-derive it from that stream: a resumed generic arm reloads its saved initial candidates (`neural_search.rs:1136-1146`), and the population check, by its own contract, does not authenticate proposal origins (`neural_search.rs:422-424`). Generation zero reads no candidate discovered by the other arm. Historical final searches differed: their arms shared incumbent, retained-teacher and exploration seed pools, and shared incumbents can create many ties ([§](#c-learning--distinguish-routing-ranking-and-search-recovery)). The owner's rule for future seeding (status: proposed; not yet met for the generic arm): initial populations must be authenticated, and a discovery cannot silently seed another arm. This manual proposes declaring any incumbent, retained teacher or another arm's discovery as an authenticated origin in the plan before execution.

<a id="c-search--dense-width-refusals-and-the-proposed-feasibility-preflight"></a>

## 5.3 Dense-width refusals and the proposed feasibility preflight

At `a75e543d2` the 512-column cap applies to generated dense width, not to total schema or raw input width (`MAX_GENERATED_DENSE_WIDTH`, `crates/feat-math/src/advanced_v3/feature_block.rs:6`, `:107-110`). No check runs before launch: the recipe validation that precedes fitting checks task, schema, syntax and stage admissibility but does not predict generated widths, so an over-wide recipe fails inside a launched fit. In the origin proof those refusals were 130 failed commands from 26 recipes, including 16 of 64 `neural_policy` generation-zero candidates ([§](#c-status--failures-dense-width-refusals-and-the-earlier-failed-reproducer)).

A pre-launch dense-width feasibility check is **proposed**; nothing below is implemented. The owner's requirements (relayed 2026-09-23T22:27:11Z) and Codex's read-only source investigation (`plan-execution-20260923/feasibility-phase-accounting.md`, SHA-256 `0154e6b0…`) bound it:

- **Permitted fitting information only.** The check may read each inner fold's task, exact schema, class order and training-side fitting rows, never prediction rows or validation labels. One-hot and hybrid categorical widths depend on the fitting-side category layout, which the compiler deliberately refuses to guess; an undecidable case stays unknown and goes to the actual worker rather than being rejected.
- **Unchanged limits and clocks.** A rejected candidate keeps its slot, origin and ancestry, counts against the arm's evaluation allowance and earns no replacement draw. The check runs inside the existing arm allowance.
- **Separate reporting.** A preflight rejection is reported separately from a launched fit failure, with its own receipt and zero physical fit starts.
- **No unlimited free retries.** Resume neither retries a proven rejection nor changes the deadline.

The accounting hazard is specific. `PoolBackend::evaluate` records every job-preparation error as `Unavailable`, with zero physical fit starts and `retryable_inventory: true` (`crates/jope-pipeline/src/neural_search.rs:1608-1624`). A naive preflight that returned an error from job preparation would therefore mark a deterministically infeasible recipe as retryable and could repeat preflight work on resume. The investigation instead proposes a typed terminal outcome with accounting: all five folds are checked before any fold of the candidate launches, and the candidate keeps one observation with zero physical fit starts, no worker receipt, no out-of-fold loss and `retryable_inventory: false`. Recovery then verifies the retained preflight receipt without fitting. The closest existing precedent is the zero-fit terminal record for canonical duplicates (`evaluate_unique`, `neural_search.rs:884-925`).

<a id="c-search--warm-fitted-demonstration-after-integration"></a>

## 5.4 Warm fitted demonstration after integration

Runbook summary (status as of 2026-09-23 about 23:55 UTC: **pending integration**):

| Runbook part | Warm fitted demonstration |
| --- | --- |
| Purpose and status | Fit, save, reload and predict in one timed invocation: a generic mode with one registered recipe, and an authenticated ASM mode with paired arms on one frozen regression dataset. **Pending integration:** `demo` belongs to the stopped product overlay and is absent from the immutable `68f56404` CLI ([§](#c-installation--command-surfaces-and-the-stopped-integration-order)) |
| Prerequisites and resources | Generic mode: the allowance stated in [§](#c-search--generic-and-asm-demonstration-commands). ASM mode: the reviewed CLI built with `asm`, the worker build inventory, qualified context and policy workers, the runtime inventory for the installed executable, and a resource declaration; the product guide's example declares 20 workers and CPUs, 64 GiB, zero swap, no GPU tasks, 60 seconds in total with 5 for reporting, and 30 seconds per fit |
| Manifest-creation command | Generic mode writes its own private plan inside its output directory. ASM mode uses `benchmark plan` and then `optimize proof-plan`, both present at `a75e543d2`. No `jope` command writes the warm resource declaration or the warm-only ASM configuration |
| Execution | The `demo` commands in [§](#c-search--generic-and-asm-demonstration-commands) |
| Artifacts and success | Generic: the private plan, worker execution journal, `fitted.json`, `predictions.json`, `baseline.json`, `demo-result.json` and `invocation-receipt.json`. ASM: `asm-fitted.json`, `generic-fitted.json`, their predictions, `baseline.json`, the complete `origin-proof.json`, `asm-demo-result.json` and `invocation-receipt.json`. Success requires saved complete fitted state, unchanged predictions after reload and finite comparisons; ASM mode also requires two full equal arm allowances, genuine neural-origin fits and evolution, and both selected refits. `under_60_seconds` (and, with the warm-target add-on, `target_met`) is true only for a successful full invocation measured below 60 seconds |
| Failure and resume | Failed attempts keep their failure and execution evidence. The pending guide describes no resume of an interrupted demonstration: a new attempt uses a new output directory outside Git checkouts and `/tmp`, and cannot reuse an expired origin-proof journal. Missing or clipped arm time sets `matched_comparison: false` and fails the ASM demonstration |
| Limitations | One dataset: not a thirty-dataset result, a held-out quality claim or training evidence. Synthetic mode is labelled. Optional exports stay `emitted-unqualified` or explicitly unsupported. No speed claim exists until a run is measured |

Source: pending `docs/private-install-demo.md:90-203` and `crates/jope-bench/src/panel/demo_asm.rs:373-459` in the product overlay replay (`agent-product-final-68f56404/replay-001/source-text` under the ASM custody root).

<a id="c-search--generic-and-asm-demonstration-commands"></a>

### 5.4.1 Generic and ASM demonstration commands

The pending generic demo fits one task-compatible registered recipe, saves complete private state, reloads it, verifies unchanged predictions and compares against a train-mean or train-prior baseline. It uses one CPU/worker, 2 GiB total, zero swap/GPU, 30 seconds per fit and 60 seconds total with five seconds reserved for reporting. Its synthetic mode is explicitly labelled; it is not evidence from a real panel dataset.

```sh
"$J" demo --dataset "$DATASET_ID" --manifest "$M" --datasets "$D" \
  --source-sha "$S" --output "$A/demo-real"
"$J" demo --synthetic multiclass --source-sha "$S" \
  --output "$A/demo-synthetic" --export python
```

The authenticated ASM route consumes a separately prepared single-regression proof plan:

```sh
"$J" demo --asm-plan "$WARM_PLAN" --asm-plan-sha256 "$WARM_PLAN_ID" \
  --manifest "$M" --datasets "$D" --output "$A/demo-asm" \
  --export python --export rust
```

<a id="c-search--the-under-sixty-second-target-and-historical-timing"></a>

### 5.4.2 The under-sixty-second target and historical timing

The warm-target add-on treats under 60 seconds as a declared, measured goal, not a promise and not a cap that prevents a required real fitted demonstration. Both complete equal arm allowances must fit within the declared original execution window. Unequal, missing or globally clipped arms fail the matched-comparison condition. An explicitly admitted longer allowance can produce a successful complete execution with `target_met: false`; it cannot claim under-60 acceptance. It does not change the official panel's 120-second arm defaults.

The retained proof at source `3fd7d5b2` started the policy about 22.1 seconds into its original window, spent about 20.8 seconds in policy production and started the first fit around 50.4 seconds. Those are historical measurements, not a forecast or a proof that equal 20-second warm arms work. A suggested longer configuration is a sizing proposal until explicitly admitted. Neither a generic substitute nor clipping the second arm solves this requirement.

<a id="c-search--focused-single-dataset-search"></a>

## 5.5 Focused single-dataset search

Status as of 2026-09-23 about 23:55 UTC: the bounded one-dataset proof front door below is **available** at `a75e543d2` for a regression dataset already in the frozen panel; the sustained, incumbent-aware attack is **proposed** ([§](#c-search--what-is-missing-for-a-sustained-incumbent-seeded-dataset-attack)).

<a id="c-search--commands-that-exist-today-for-one-dataset"></a>

### 5.5.1 Commands that exist today for one dataset

Runbook summary:

| Runbook part | One-dataset focused proof |
| --- | --- |
| Purpose and status | Run one bounded paired neural and generic search on one frozen-panel regression dataset and recompute an authenticated development report. **Available** at `a75e543d2`; measured in the cited `68f56404` proof on `230_machine_cpu`; this is not a count of all earlier origin runs ([§](#c-status--canonical-proof-record-and-per-arm-facts)) |
| Prerequisites and resources | The inputs in [§](#c-search--required-inputs); no exclusive full-CPU CI running; a configuration and resource declaration within the implementation caps (an arm of at most 120 seconds, 80–1000 evaluations, at most 30 seconds per fit). The measured proof declared 20 workers and CPUs, 64 GiB, zero swap, no GPU tasks and a 600-second window including 30 seconds for reporting |
| Manifest-creation command | `benchmark plan` writes the base (comparator) plan ([§](#c-installation--native-roster-and-external-comparator-runtimes)); `optimize proof-plan` writes the focused plan and prints its semantic `plan_sha256`. No `jope optimize` subcommand writes the complete `jope.asm-configuration/v1` configuration or the resource declaration: the origin proof's configuration came from the custody helper `agent-neural/prepare-origin-config.py` (receipt `configuration-preparation.json` beside that proof), which is not a product command. The frozen panel manifest itself is fixed, not generated |
| Execution | `optimize run` ([§](#c-search--plan-run-and-report-a-focused-proof)); it refuses a running binary that differs from the pinned native runtime |
| Artifacts and success | `plan.json`, `plan.selection.json` and `plan-receipt.json`; the journal (window, campaign and per-arm records, started and finished worker receipts); the report returned by `run` and recomputed by `optimize report` without fitting. Success is an authenticated report of both arms inside the original window, not a neural win |
| Failure and resume | The same `run` resumes the journal within its original window; an expired window cannot be renewed. Failed and refused fits, such as dense-width refusals ([§](#c-search--dense-width-refusals-and-the-proposed-feasibility-preflight)), remain recorded outcomes |
| Limitations | One regression dataset already in the frozen panel; no `--incumbent` input; no arbitrary-dataset ingestion; development-hold scores only; the dense-width cap is checked only inside launched fits |

Sources: `crates/jope-bench/src/panel/optimize_cli.rs:48-89` and `crates/jope-bench/src/panel/campaign_cli.rs:130-208` at `a75e543d2`; custody `neural-integration/origin-proof-68f56404…/configuration-preparation.json`.

<a id="c-search--front-door-and-safe-metadata-checks"></a>

#### 5.5.1.1 Front door and safe metadata checks

The current private Linux `asm` build supports a **separately bounded operational proof on one frozen regression dataset**. This is the closest current neural search front door to focused optimization. It is not an arbitrary-dataset ingestion command, a daemon, or an incumbent-seeded high-water-mark attack.

The following metadata commands are safe preparation and were checked against the immutable reviewed 68 CLI:

```sh
"$JOPE" optimize capabilities
"$JOPE" optimize proof-plan --help
"$JOPE" optimize run --help
"$JOPE" optimize report --help
```

`compiled:true` means support was compiled, not that assets, context qualification or training have passed. Use a binary whose build inventory, source manifest and executable hashes match the configuration. The implementation at `a75e543d2` caps an arm at 120 seconds and its candidate allowance at 80–1000 evaluations (`crates/jope-bench/src/panel/optimize.rs:132-133`). Per-fit time remains at most 30 seconds (`crates/jope-bench/src/panel/origin_proof.rs:86`). Editing a JSON number beyond those caps does not extend the implementation.

<a id="c-search--required-inputs"></a>

#### 5.5.1.2 Required inputs

| Variable | Meaning |
| --- | --- |
| `JOPE` | Authenticated Linux CLI built with `asm`; not an unrelated binary found on `PATH` |
| `BASE_PLAN` / `BASE_PLAN_ID` | Verified comparator plan and its emitted semantic identity |
| `ASM_CONFIG` / `ASM_CONFIG_SHA` | Complete qualified ASM configuration and its file SHA256 |
| `FOCUS_RESOURCES` / `FOCUS_RESOURCES_SHA` | Explicit resource-window declaration and its file SHA256 |
| `PANEL` / `PREPARED` | Unchanged frozen panel manifest and verified prepared development data |
| `FOCUS_DIR` | New durable private output directory given as an absolute path, outside Git working trees and `/tmp`; `optimize run` refuses a relative journal path |
| `DATASET` | One regression dataset already present in the frozen panel, such as `230_machine_cpu` |

The base plan and complete configuration are prepared as described in [§](#c-installation--native-roster-and-external-comparator-runtimes) and [§](#c-architecture--qualification-and-admission). They include real runtime/worker inventories, checkpoint/vocabulary/transform assets and context qualification. Hand-writing a short configuration with invented hashes cannot replace those artifacts. The resource declaration includes workers, CPUs, memory, zero swap, GPU task allowance, threads per worker, total/reporting seconds and the per-fit cap.

<a id="c-search--plan-run-and-report-a-focused-proof"></a>

#### 5.5.1.3 Plan, run and report a focused proof

Prepare a plan before admitting numerical work:

```sh
mkdir -m 700 -- "$FOCUS_DIR"
"$JOPE" optimize proof-plan \
  --base-plan "$BASE_PLAN" --base-plan-sha256 "$BASE_PLAN_ID" \
  --configuration "$ASM_CONFIG" --configuration-sha256 "$ASM_CONFIG_SHA" \
  --dataset "$DATASET" \
  --resources "$FOCUS_RESOURCES" --resources-sha256 "$FOCUS_RESOURCES_SHA" \
  --output "$FOCUS_DIR/plan.json" > "$FOCUS_DIR/plan-receipt.json"

FOCUS_PLAN_ID=$(jq -er '.plan_sha256' "$FOCUS_DIR/plan-receipt.json")
```

Planning reports `execution_budget_started:false`. After the declared allocation is available and exclusive full-CPU CI has finished, the execution command is:

```sh
"$JOPE" optimize run \
  --plan "$FOCUS_DIR/plan.json" --plan-sha256 "$FOCUS_PLAN_ID" \
  --manifest "$PANEL" --datasets "$PREPARED" \
  --journal "$FOCUS_DIR/journal"
```

The same `run` command resumes that journal within its original window. Do not change directories, timestamps or identities to disguise an expired run as a resume. Recompute the report without fitting:

```sh
"$JOPE" optimize report \
  --plan "$FOCUS_DIR/plan.json" --plan-sha256 "$FOCUS_PLAN_ID" \
  --manifest "$PANEL" --datasets "$PREPARED" \
  --journal "$FOCUS_DIR/journal"
```

The command audit for this chapter executed only metadata/help operations. These example execution commands were not run as a new campaign while writing the manual. The separately measured origin proof at source `68f56404` ([§](#c-status--canonical-proof-record-and-per-arm-facts)) remains the actual numerical example.

<a id="c-search--other-current-search-surfaces"></a>

#### 5.5.1.4 Other current search surfaces

`jope search --input "$FIXTURE" --json` scores a tiny ridge-family candidate bank. It is useful for its documented fixture scope; it is not sustained neural GP exploration.

`jope pipeline run` accepts a catalog dataset or dataset directory, cuts, seeds and a budget, and can emit several languages. Its `--policy auto` and `skip` report a skip, while `require` fails closed. Its legacy budget description explicitly allows an in-flight fit to overrun. Therefore neither increasing its attempt count nor choosing `--policy require` supplies the current bounded neural learning path. See [§](#c-inference--example-b-actual-legacy-pipelinesearch-shape) for the actual legacy pipeline example and its receipt.

<a id="c-search--what-is-missing-for-a-sustained-incumbent-seeded-dataset-attack"></a>

### 5.5.2 What is missing for a sustained, incumbent-seeded dataset attack

There is no current `--incumbent` input on `optimize proof-plan`, and no shipped `jope optimize attack` command. A finite, wave-by-wave attack interface is a proposed follow-on. Documenting a plausible command string would not make it available.

The owner's rules for that follow-on (relayed 2026-09-23T22:27:11Z; status: proposed): focused optimization is finite and incumbent-aware and runs inside one total window declared before searching. Before the first wave it freezes the exact rows, groups, evaluator, vocabulary and numerical identity, the authenticated initial population, the exploration and wave rules, and the atomic update rule. A tie keeps the incumbent, and a race with a concurrent incumbent change forces a re-decision rather than an overwrite. Ingesting an arbitrary dataset is a separate, authenticated request, never an edit of the frozen thirty-dataset panel.

An implementation should reuse the current worker pool, selection plan and journals, and integrate the pending eligibility overlay (TEACH-01/TEACH-02) only after it is qualified, with these specific additions:

| Addition | Required behavior |
| --- | --- |
| Frozen focus request | Authenticate one dataset, task, exact fold/group/row identity, evaluator, vocabulary, numerical policy and declared total window before searching |
| Eligible incumbent reader | Read the incumbent under the complete comparison identity; refuse a stale, incompatible or unauthenticated record |
| Incumbent-aware initialization | Authenticate the complete initial population; include the incumbent and bounded mutations/macros with explicit, declared origins; preserve an exploration fraction and canonical duplicate accounting |
| Wave accounting | Reuse the original global deadline and journal; each bounded wave consumes remaining time rather than resetting it |
| Re-evaluation and admission | Use complete finite train-side OOF evidence; meet the required trivial-baseline/4-of-5-fold gate, then compare against the incumbent under the same identity |
| Atomic improvement | Strict improvement moves the pointer only after durable evidence; ties keep the incumbent; concurrent pointer changes trigger re-decision |
| Useful teaching delta | Export newly eligible programs and justified comparison records with original context/labels/exposure, rather than copying a winner's reported score |
| Honest completion | Report best retained incumbent, all waves and costs, improvement/stagnation, failures and unmet backend support even when no new winner appears |

The original acceptance requires four-of-five-fold improvement over a trivial baseline. A proposed extra four-of-five-fold comparison against the incumbent is a separate robustness policy, not a shipped requirement or retrospective reason to reject old evidence. Freeze that policy before running a focused extension. The current HWM strict-improvement rule also does not by itself prove the new teaching eligibility contract.

For a dataset outside the frozen panel, first implement or use a separate authenticated ingestion/preparation request with the same label, category, missingness and group guarantees. Do not edit the frozen panel's manifest (asm-private-30-v1) to add it. Repeated comparisons use development labels; sealed evaluation remains outside search, teaching and motif mining.


---

# Chapter 6 — Teacher admission, HWM identities and corpus updates

<a id="c-teachers--a-dataset-is-not-one-training-example"></a>

## 6.1 A dataset is not one training example

<a id="c-teachers--accounting-units"></a>

### 6.1.1 Accounting units

A training context describes a particular dataset, split and encoding. Many executable programs can share that context. Each program supplies a separate target sequence and measured supervision. The useful accounting units are therefore:

| Unit | Meaning | Why the distinction matters |
| --- | --- | --- |
| Dataset/family | A problem and its related datasets | Family partitions control generalization; row counts alone do not identify independence. |
| Cut/context | A dataset split plus prepared-data and encoder identities | The same dataset can have many contexts. Reusing a context must preserve its actual split and fitting rows. |
| Program observation | A measured evaluation of a concrete executable recipe | Failed, duplicated, incompatible or exposed observations need not become training examples. |
| Admitted target record | A representable program and its supervision for one context. Under the pending contract, a **teacher** is such a record that passed the `jope.teaching-guard/v1` admission ([§](#c-teachers--the-admission-guard-label-exposure-and-hwm-identity)) and entered a frozen corpus | Repeated observations of the same program/cut do not automatically earn repeated votes. |
| Corpus snapshot | An immutable collection of target records and identities | Its record count is not the number of optimizer updates or unique datasets. |
| Presentation | A sampled training example used in an update | Sampling revisits records. Millions of presentations do not mean millions of newly discovered targets. |

<a id="c-teachers--context-vectors-and-program-targets"></a>

### 6.1.2 Context vectors and program targets

The legacy Rust context is 2,048 little-endian float32 values, or 8,192 bytes. The model receives a 1,536-value dataset lane and a separate 512-value target-statistics lane. The historical encoder recipe described the dataset lane as Hyperion 512 plus Foundation/PCA 1,024. Despite its name, the dataset lane is not label-free: in the current context producer the 1,024 Foundation values are computed from the features together with the standardized training targets, and only the 512 Hyperion values use the features alone ([§](#c-architecture--component-dimensions-and-assembly)). These vectors are shared by context identity; they are not repeated inside every 532-byte target record. The program target is a grammar-token sequence, with the legal-token set for each step, context ID, score-derived value, percentile and log-runtime supervision.

A concrete metadata binding in the accepted step-60000 corpus connects dataset `031201c6895f724b`, family `rdatasets-agridat-kalamkar-potato-uniformity`, cut 0 and `meta_train` to a prepared view and context vector. Its context SHA begins `6f5fdd79…`, prepared-view SHA `811feabd…`, and content-addressed shard contains 24,231 records. The full paths and hashes are in `accepted-corpus-binding.json`. This audit verified that metadata relation without opening the encoded vector or target records.

<a id="c-teachers--the-learning-chain"></a>

### 6.1.3 The learning chain

The learning chain is:

1. Pin the raw dataset, family partition and actual fitting/development row assignments.
2. Fit preprocessing and encoding only within the permitted input closure; pin their bytes and versions. The closure covers records, contexts, transforms and every asset, and a forbidden descriptor must be rejected before loading even when no admitted row uses it. At the retained PR83/stopped-overlay snapshot, that rejection is **proposed**: neither source rejects extra context-index entries, and the legacy trainer at `a75e543d2` stages every indexed context after only a digest and size check ([§](#c-data--hashes-eligibility-and-the-complete-input-closure)).
3. Generate a concrete executable program, fit it and retain its outcome, measurement, resource use and evaluator identity.
4. Apply the declared admission policy; reject failed, incompatible, exposed or unsupported targets.
5. Tokenize the exact program and retain original observations beside the training representation.
6. Sample the admitted record, update parameters, and prove checkpoint reload/continuation.
7. Evaluate the new policy through actual proposals and fits under the comparison protocol.

A file at step 5 does not prove steps 6–7. Conversely, a lower optimizer loss does not establish better executable proposals.

The chain yields four distinct claims, and the owner requires them to stay separate (relayed 2026-09-23T22:27:11Z). Each needs its own evidence; none implies the next:

| Claim | Established by | Not established by |
| --- | --- | --- |
| Eligible teacher | A record that passed the preregistered `jope.teaching-guard/v1` admission under its complete comparison identity (step 4) | A content hash, a saved recipe, a completed run or a historical HWM entry |
| New discovery | A program absent from the authenticated initial seed population of the search that found it, with its discovering arm, generation and first measurement retained ([§](#c-learning--what-to-retain-for-each-cycle)) | A carried incumbent, a re-measured seed or a duplicate of an existing record |
| Actually consumed information | Sampler and optimizer receipts showing the record sampled in committed optimizer updates (step 6); changed parameters are a separate LOOP-01 link ([§](#c-learning--loop-01-the-central-regression-milestone)) | Corpus membership or a manifest row |
| Improved policy | A matched comparison of parent and challenger checkpoints with actually fitted proposals under the same protocol and budget (step 7; [§](#c-learning--the-closure-ledger)) | A lower training loss, a new checkpoint file or the teacher's own development score |

For the new ASM path none of the four is established yet: zero teachers have been admitted under the new contract ([§](#c-teachers--present-teacher-readiness-and-the-smallest-real-path)), and the pending campaign records no initial-population or discovery identity. Reporting the four separately is a **proposed** rule.

<a id="c-teachers--how-a-fitted-recipe-becomes-an-eligible-teacher"></a>

## 6.2 How a fitted recipe becomes an eligible teacher

<a id="c-teachers--the-preregistered-freeze-and-train-side-inner-split"></a>

### 6.2.1 The preregistered freeze and train-side inner split

The pending eligibility path freezes independent facts before optimization: original dataset and prepared-data identity, ordered rows and groups, target role, exact inner-fold assignment, evaluator and vocabulary identities, numerical policy, baseline and worker/resource pins. This freeze cannot be created retrospectively from a winning candidate's own output. A historical successful fit is not equivalent to preregistered teaching evidence.

The train-side inner split uses five whole-group folds. Groups are assigned deterministically by largest size first, with lexical tie-breaking and balancing. No group's rows are split simply to fill a fold. The cut's held-out development fold remains outside the search's inner fitting/selection closure.

<a id="c-teachers--the-admission-guard-label-exposure-and-hwm-identity"></a>

### 6.2.2 The admission guard, label exposure and HWM identity

Each candidate must produce complete authenticated OOF predictions for all five inner folds. Scores are recomputed from retained predictions, exact targets and ordered class probabilities. The trivial baseline is fitted from each fold's training rows: target mean for regression, class frequencies for classification. The guard uses mean fold loss and admits only if:

- all five fold losses and baseline losses are finite;
- the mean candidate loss is below the mean baseline loss by more than `1e-12 + 1e-6 * abs(mean_baseline_loss)`;
- at least four of five folds each improve by more than `1e-12 + 1e-6 * abs(that fold's baseline_loss)`;
- actual worker requests, process outcomes, output hashes, fitted artifacts, seeds, source, resources and deadlines match the original freeze.

The criteria are intentionally stronger than “a run completed” or “the score field is positive.” Constant/no-signal predictors, missing folds, reordered rows, changed class order, stale scores, unauthenticated prediction blobs and unknown execution accounting cannot provide teaching authority. A failed admission is retained as evidence rather than discarded. A hash is not eligibility: a content digest proves that bytes are unchanged, not that the record passed this guard under the right comparison identity with permitted labels. In the pending overlay, verifying a retained admission report re-derives the qualification from the learning state (`eligibility::revalidate_qualification`) and refuses any mismatch, rather than trusting a stored `qualified` flag (pending `crates/jope-bench/src/panel/teaching.rs:730-736`).

Optimization-visible labels initially called `test` become development labels, with a durable record of that exposure. `sealed` and `meta_test` inputs are refused for optimization/teaching. The HWM identity includes the full comparison context, so an improvement under a different evaluator, split, vocabulary, preprocessing or numerical protocol cannot displace an unrelated incumbent. These details belong to the pending eligibility/HWM correction, not a claim that every historical HWM record already meets this rule.

Sources: pending `crates/jope-learning/src/eligibility.rs:170`, `:253`, `:277`, `:427`, `:700`, `:1023`; `crates/jope-learning/src/eligibility/inner.rs`; `crates/jope-learning/src/record.rs`; `crates/jope-pipeline/src/search_plan.rs`.

<a id="c-teachers--from-optimization-to-reusable-knowledge"></a>

## 6.3 From optimization to reusable knowledge

The pending strict HWM key includes task, dataset hash, cut identity, evaluator, vocabulary and protocol. Scores from different identities cannot overwrite each other or supply training feedback. A recipe must show finite train-side out-of-fold improvement over a trivial baseline and improve on four of five folds before teaching admission. No-signal data should be refused by this gate. An optimization-visible test label is development evidence even if its filename says otherwise.

Typed motifs can reuse eligible program fragments and macros while retaining provenance. The prepared motif library does not yet establish the production eligible-state reader, `motifs mine` integration or operational reuse. A new operator requires compatibility tests and an ablation under the unchanged comparison protocol. Repeatedly selecting the best development result can improve development performance without proving improved sealed generalization.

<a id="c-teachers--present-teacher-readiness-and-the-smallest-real-path"></a>

## 6.4 Present teacher readiness and the smallest real path

The metadata readiness audit verified 28 source-family evidence files and the retained original process/context receipts. The inspected historical origin runs at `55e9ed688faf` and `3fd7d5b2ebb6` had no preregistered teaching freeze or vocabulary pins; they cannot be upgraded after the fact. The operational origin proof at `68f56404` (the reviewed PR83 head, merged on 2026-09-23 as main `a75e543d2` with the identical tree) also was not a teaching campaign. There is therefore no currently demonstrated eligible real corpus for the new ASM admission-and-training proof. This does not negate the historical training corpus or the optimizer work that produced the retained model.

The prospective small pair selected from frozen provenance/size, before observing losses, is:

| Dataset | Evidenced family | Cut 0 fitting rows (development fold 0 held out) |
|---|---|---:|
| `230_machine_cpu` | UCI computer-hardware relative-performance source | 135 |
| `712_chscase_geyser1` | Chatterjee casebook geyser source | 140 |

Both must independently produce eligible recipes. The minimum of ten completed inner fits is only the five-fold×two-family floor for one successful recipe per family. It is not a replacement for the specified matched search and does not promise that either family will improve.

If exactly these two families qualify, seed 7 assigns `230_machine_cpu` to the policy-validation families and `712_chscase_geyser1` to the policy-training families under the current partition rule, leaving a single training family. If more families qualify, use the entire admitted set and let the native family partition determine membership. Do not omit inconvenient eligible comparisons, force the two-family allocation, or reuse the frozen panel's (`asm-private-30-v1`) deadline for a separate trial. Historical checkpoint exposure remains a separately reported unknown; family provenance alone does not prove zero prior exposure. Keep three family-level domains distinct: teacher and training families; adaptive development families, where the frozen thirty-dataset panel remains; and evaluation families outside both training and selection. Exposure of the pretrained encoder components to any family also remains unknown.

Both prospective families are small regression datasets of the frozen panel itself (169 and 172 development rows, [§](#c-datasets--per-dataset-summary)); no real binary or multiclass family is prospective. With exactly these two families, the first training campaign is a **mechanics qualification**: it can show optimizer updates, reload, continuation and consumption of a new checkpoint, but it cannot test whether training improves proposals, and its export must not be compared with step 60,000 as a flywheel challenger. That campaign trains experimental regression lineage v1 (256×2, new initialization), a width-256, two-layer model family separate from the accepted 512×6 step-60000 lineage ([§](#c-training--what-the-pending-campaign-trains)). If either dataset is used to manufacture teachers, later panel evaluations on it are labelled exposed/adaptive ([§](#c-learning--freezing-the-experiment-and-choosing-families)); see also OD-8 in [§](#c-plan--proposed-owner-decisions-and-external-dependencies).

Source evidence: custody `agent-training-readiness/readiness-v3.json`, SHA256 `9197900d4b696811909c4ba4e779e6c5d488569dd9c360165cc55fd43bf8c735`. Its metadata inspection does not itself admit training or re-run numerical qualification.

<a id="c-teachers--prepare-a-teaching-campaign-before-it-starts"></a>

## 6.5 Prepare a teaching campaign before it starts

Runbook summary (status as of 2026-09-23 about 23:55 UTC: **pending integration**):

| Runbook part | Teaching campaign |
| --- | --- |
| Purpose and status | Preregister train-side teaching freezes, run a bounded search, and admit each arm's selected recipe into the learning state only if it passes `jope.teaching-guard/v1`. **Pending integration:** the preregistration and admission code belongs to the stopped training64 overlay, and the configuration at main `a75e543d2` has no teaching member. Zero teachers have been admitted |
| Prerequisites and resources | The final reviewed build and its receipts ([§](#c-teachers--final-build-and-the-teaching-configuration)); a `teaching` member whose `state_root` and `prepared_root` are absolute, whose state root lies outside `/tmp` and outside Git checkouts, whose panel file is the approved frozen manifest and whose label role maps to development; a separately declared, approved resource allocation; no exclusive CPU CI running |
| Manifest-creation command | `benchmark plan` writes the comparator plan; `optimize plan` or `optimize proof-plan` writes the ASM plan and, in the pending overlay, preregisters the `jope.inner-teaching-registry/v1` freezes before the final experiment identity. No `jope optimize` subcommand, at `a75e543d2` or in the pending overlay, writes the complete configuration or its `teaching` member |
| Execution | `optimize run` ([§](#c-teachers--plan-run-and-recover-a-teaching-campaign)). Admission runs inside the controller before outer-refit scheduling and consumes completed receipts only; it starts no fit and no clock |
| Artifacts and success | In the journal, a `jope.inner-teaching-admission/v1` report at `teaching-admission/<key>.json` when every selected arm's entry qualified or the registration was unavailable, otherwise a `teaching-inventory/<key>-<hash>.json` record. Success is at least one qualified record per family, re-verified from the retained evidence |
| Failure and resume | `optimize run` resumes the original journal within its window. A non-terminal admission failure is inventory that revalidation may retry within the original experiment; a failed admission is retained, never deleted |
| Limitations | The two prospective families are frozen-panel regression datasets, so the first campaign is a mechanics qualification ([§](#c-teachers--present-teacher-readiness-and-the-smallest-real-path)); admission considers only each arm's selected recipe; the real admission path has not run |

Sources: pending `crates/jope-bench/src/panel/teaching.rs:48-67`, `:123-125`, `:537-623`, `:754-756` and `crates/jope-bench/src/panel/optimize_cli.rs:13-53` in the training overlay replay (`agent-training-integration/refresh-68f56404/replay-001/source-text` under the ASM custody root).

<a id="c-teachers--final-build-and-the-teaching-configuration"></a>

### 6.5.1 Final build and the teaching configuration

The final reviewed build must supply its CLI, worker inventory, source manifest, runtime manifests, component assets and qualification receipts. Record the actual built source SHA; do not substitute the latest checkout hash for the binary's source. Recheck that no exclusive CPU CI is running before opening an authorized CPU campaign.

In the pending overlay, the `teaching` member of the complete ASM configuration has this shape (the configuration at main `a75e543d2` has no teaching member):

```json
{
  "state_root": "/durable/private/learning-state",
  "prepared_root": "/durable/private/prepared-panel",
  "panel": {"path": "/source/benchmarks/panel.json", "sha256": "EXACT_PANEL_FILE_SHA256"},
  "original_label_role": "development"
}
```

The capitalized value is a placeholder, not a valid digest. The complete configuration also requires exact source/build/runtime/checkpoint/vocabulary/context fields; the small object above cannot be passed as the whole configuration. Planning preregisters the teaching freezes before final experiment identity, so worker evidence can bind them.

<a id="c-teachers--plan-run-and-recover-a-teaching-campaign"></a>

### 6.5.2 Plan, run and recover a teaching campaign

The following shell variables must already name authenticated artifacts. Commands are interface examples and allocate no budget merely by appearing here:

```sh
"$JOPE_FINAL_CLI" benchmark plan \
  --manifest "$JOPE_PANEL" --datasets "$JOPE_PREPARED" \
  --source-sha "$JOPE_FINAL_SOURCE" --runtimes "$JOPE_RUNTIMES" \
  --output "$JOPE_CAMPAIGN/comparator-plan.json"

"$JOPE_FINAL_CLI" optimize plan \
  --base-plan "$JOPE_CAMPAIGN/comparator-plan.json" \
  --base-plan-sha256 "$JOPE_BASE_PLAN_ID" \
  --configuration "$JOPE_TEACHING_CONFIGURATION" \
  --configuration-sha256 "$JOPE_CONFIGURATION_FILE_SHA" \
  --output "$JOPE_CAMPAIGN/asm-plan.json"
```

Use the semantic plan identity emitted by native planning for `--base-plan-sha256` and `--plan-sha256`. A JSON file's byte hash and its semantic identity need not match. Configuration/resource arguments explicitly refer to their required file hashes.

For a separately admitted small proof, substitute this planning command. `proof-plan` selects one regression dataset from the verified frozen base panel, replaces the resources with the explicitly pinned proof resource manifest, and preregisters teaching only for that selected dataset:

```sh
"$JOPE_FINAL_CLI" optimize proof-plan \
  --base-plan "$JOPE_CAMPAIGN/comparator-plan.json" \
  --base-plan-sha256 "$JOPE_BASE_PLAN_ID" \
  --configuration "$JOPE_TEACHING_CONFIGURATION" \
  --configuration-sha256 "$JOPE_CONFIGURATION_FILE_SHA" \
  --dataset "$JOPE_SELECTED_DATASET" \
  --resources "$JOPE_PROOF_RESOURCES" \
  --resources-sha256 "$JOPE_PROOF_RESOURCES_SHA" \
  --output "$JOPE_CAMPAIGN/asm-plan.json"
```

A second family requires its own actual preregistered proof plan and journal, with its separately declared approved resource allocation. Both may point at the same learning state and frozen30 metadata. Neither proof changes or spends an unstarted official panel's immutable window. Proposed example durations in earlier queue documents were not resource admissions.

Run and recover using the same immutable plan and original journal:

```sh
"$JOPE_FINAL_CLI" optimize run \
  --plan "$JOPE_CAMPAIGN/asm-plan.json" --plan-sha256 "$JOPE_ASM_PLAN_ID" \
  --manifest "$JOPE_PANEL" --datasets "$JOPE_PREPARED" \
  --journal "$JOPE_CAMPAIGN/journal"

"$JOPE_FINAL_CLI" optimize report \
  --plan "$JOPE_CAMPAIGN/asm-plan.json" --plan-sha256 "$JOPE_ASM_PLAN_ID" \
  --manifest "$JOPE_PANEL" --datasets "$JOPE_PREPARED" \
  --journal "$JOPE_CAMPAIGN/journal"
```

Sources: `crates/jope-bench/src/panel/optimize_cli.rs:13` and `crates/jope-bench/src/panel/origin_proof.rs` (the `optimize plan`, `proof-plan`, `run` and `report` front door is present at main `a75e543d2`, where it makes no teaching call; the pending overlay adds the `teaching::preregister` call to `optimize plan` and `proof-plan` and the new `crates/jope-bench/src/panel/teaching.rs` that defines it); current `crates/jope-bench/src/panel/campaign_cli.rs:18`.

<a id="c-teachers--corpus-updates-reuse-device-caches-and-loading"></a>

## 6.6 Corpus updates: reuse, device caches and loading

<a id="c-teachers--on-disk-shard-reuse"></a>

### 6.6.1 On-disk shard reuse

The on-disk teaching store can reuse unchanged verified shards **within an unchanged store binding**. In reviewed source `68f56404` (merged as main `a75e543d2` with the identical tree), a group's key includes binding, context, programs and lineage. The binding includes twelve semantic source files and the SHA256 of `/proc/self/exe`. A changed packer executable digest therefore rejects the old store binding and changes every group key, requiring a new store/full disk repack even if the logical records are unchanged. A byte-identical reproducible rebuild does not change that digest. This disk invalidation is separate from GPU cache invalidation. New ledger contributions mark affected groups dirty; changed or reordered historical source inputs require reconciliation or a new store rather than silently editing history. Publication uses immutable artifacts and atomic replacement of the published index. Tokenization caches are bounded and exact; invalid exact parameters remain rejected.

Snapshot 97's historical `packing-timing.json` records **7,859 reused groups and 18 rebuilt groups**, from 51 new ledgers. Attribute these figures to its legacy writer: the retained receipt has the four-field legacy format, whereas that source writes seven fields and has the stronger executable/semantic/lineage binding. The old reuse observation is not a qualification of the current writer. That is 99.7715% reuse by group count, not a measured 99.7715% reduction in bytes, CPU or latency. The September 11 snapshot 421 audit recorded 7,863 reused and 14 rebuilt groups. Neither receipt by itself measures end-to-end pack time or memory. The accepted snapshot's 7,877 record files all exist at the declared lengths in this metadata audit, totaling 5,434,024,624 bytes; their label sidecars add 3,227,289,461 bytes. Sidecars preserve auditability but are not free storage.

<a id="c-teachers--the-two-device-caches-in-the-native-trainer"></a>

### 6.6.2 The two device caches in the native trainer

The native trainer implementation maintains two different device caches. Its current direct dispatch remains held behind the original-GPU-grant requirement described in [§](#c-training--the-dispatch-hold-resource-incompatibility-and-valid-topologies); the following source behavior is not permission to launch it:

| Cache | Reuse identity | Behavior when identity changes |
| --- | --- | --- |
| Context vectors | Context-index SHA | Release prior tensor; read/hash every indexed 2,048-float context and stage the new table |
| Token/legal/context-ID tensors | Whole corpus SHA | Release prior tensors; traverse **every** target record and rebuild the full tables |

Tokens and legal IDs are each stored on the device as 16-bit signed tensors with all unsigned bits preserved; after gathering a minibatch, the code converts IDs to int64 and masks with `0xffff`. Context IDs are int64. The legal mask table is shared and boolean. For each minibatch, the sequence length is trimmed to the longest selected record, rounded up to a multiple of eight and capped at 128. Thus the implementation already avoids recomputing grammar masks per optimizer step and reduces unnecessary sequence work within a batch.

<a id="c-teachers--identifier-memory-and-whole-corpus-cache-invalidation"></a>

### 6.6.3 Identifier memory and whole-corpus cache invalidation

For 10,214,332 records, token-plus-legal identifiers alone are `N × 128 × 4 = 5,229,737,984` bytes. Context IDs add `N × 8 = 81,714,656` bytes. The 8,894 shared float32 contexts would occupy 72,859,648 bytes. These are layout calculations, not measured GPU peaks; they exclude the legal mask table, model, optimizer, activations, batch conversions, allocator overhead and host staging. The emitted `resident_identifier_bytes` counter covers only the token/legal identifier term, not complete device use.

A newly published corpus SHA invalidates the entire token/legal/context-ID cache even when most disk groups were reused under an unchanged binding. The historical 7,859-of-7,877 receipt illustrates disk reuse under the legacy writer; a changed current packer executable additionally forces full disk repacking as described above. There is no qualified incremental device-shard cache in this path. Disk-level incremental packing therefore does not guarantee low-latency continual training. Repeated trainer processes also cannot rely on process-local tensor residency. This is a concrete optimization opportunity to measure before changing the format.

<a id="c-teachers--python-training-input-and-the-pending-bounded-campaign"></a>

### 6.6.4 Python training input and the pending bounded campaign

The current Python `train_v3.py` path accepts JSONL `TrainingRow` objects or an in-memory list. It materializes the input rows, partitions them before grammar/reward preparation, and caches each train/validation row's token IDs and sparse per-position legal-ID lists once before the epoch loop. It still rebuilds a dense boolean legal mask of shape `[batch, sequence − 1, vocabulary size]` on the host for every batch before moving it to the device. It also supports a vocabulary-bound persistent row-artifact cache. It is inaccurate to say that Python reparses or retokenizes every example on every optimizer step.

However, the pending `training64` campaign's `admitted_rows` path still reads JSONL into memory and writes a per-attempt `corpus.jsonl`, then invokes `train_policy_v3_from_jsonl`. That stopped source packet does not add a reader for the Rust `teaching_shards.v1` packed store. Its stricter context/task/admission checks must be integrated and qualified before any real category training. The existence of an efficient historical Rust corpus does not prove that the pending Python route can ingest ten million examples within the current twenty-minute category window or 64 GiB host limit. No such performance run was authorized or performed for this chapter.

<a id="c-teachers--concrete-corpus-improvements-with-acceptance-criteria"></a>

## 6.7 Concrete corpus improvements, with acceptance criteria

The useful next change is evidence-led rather than another format by name. These are proposals (status: **proposed**), not implemented delivery claims. The owner's order (relayed 2026-09-23T22:27:11Z) is: profile the phases; then a verified Python packed-reader bridge; then immutable shard reuse with bounded staging; then compression; and a narrower record layout only if measurements justify it. Every step preserves float32 supervision, the legal-set mappings and the 128-token grammar limit, and proves record, batch, loss, gradient, update and resume parity ([§](#c-data-estate--compression-the-proposed-180-byte-layout-and-label-precision)).

1. **Measure the actual preparation path.** Retain separate wall time and peak host/device memory for metadata parsing, admission validation, tokenization, hashing/sealing, shard loading, host construction and device transfer. Measure cold and reused cases on the same admitted corpus identity. Include controller/setup cost; do not equate one identifier-array counter with total memory or CPU.
2. **Give Python a verified packed-reader bridge if those measurements justify it.** Preserve exact tokens, legal sets, float semantics, task/class routing, contexts, splits, original observations and vocabulary/admission pins. Prove JSONL/packed equivalence on admitted fixtures, corruption and changed-input rejection, exact recipe round trips, reload and continuation. A converter that copies old `qualification: true` fields without rechecking the current teacher contract is not sufficient.
3. **Then reuse immutable shards with bounded staging.** Key reusable material by content digest and explicit vocabulary/grammar/context identities; measure only changed shards transferred, and stage them through a bounded, measured host buffer rather than rebuilding the whole table. Global legal-set remapping, context IDs, balanced sampling, frozen labels and optimizer/sampler resume must remain reproducible. Appending raw old IDs into a new global table is unsafe without that mapping contract.
4. **Then evaluate compression.** Current u16 IDs and shared contexts already save space. The accepted-corpus audit ([§](#c-data-estate--what-the-accepted-corpus-packing-audit-measured)) gives externally measured compressed byte counts. Compressed archive size is not a loader-memory result; preserve bounded random access, decompressed and stored-byte hashes, corruption rejection and deterministic sampling before adopting it.
5. **Adopt a narrower layout only if measured.** The same audit gives a measured padding opportunity. Length bucketing or a narrower layout still requires parity and measured GPU throughput. Keep the 180-byte record and binary-index estimates labelled as unqualified proposals.
6. **Close the evidence chain before promoting the result.** Prove an eligible new target entered the corpus, was actually sampled, changed parameters, survived reload and improved later bounded executable-search behavior. Keep resource deadlines and sealed boundaries unchanged.

Keep exactly one learning-store writer until an atomic server-side pointer update is qualified; this manual records no such qualification. The on-disk packer already takes an exclusive, non-blocking `writer.lock` on its store directory and refuses a second concurrent packer (`crates/jope-engine/src/teaching_store.rs:377-385` at `a75e543d2`), but that local lock does not coordinate writers of a remote learning store. Backup, replication and retention operations on learning storage remain separately authorized; a corpus update must not start them.

Relevant source contracts are `crates/jope-engine/src/corpus.rs`, `teaching_store.rs`, `sealed_storage.rs` and `training.rs`; Python uses `python/src/jope/schema.py` and `policy/train_v3.py`. Existing source tests cover byte-identical reuse, changed inputs, frozen labels, original-precision midranks, interruption and immutable verified records. This documentation audit inspected those tests but did not run them. `source-pins.json` binds the inspected source and pending overlay; `HANDOFF.json` binds the chapter and its data.


---

<a id="c-training"></a>

# Chapter 7 — Training, continuation, export and compatibility

<a id="c-training--what-can-be-claimed-now"></a>

## 7.1 What can be claimed now

At source `68f56404c3e199e06413bfd645d3fe0803d7b96f`, the latest bounded regression origin proof has actual neural proposals, fitted generation-zero candidates, fitted descendants and a successfully authenticated recovered report. The newly authored training path is still a stopped follow-on. Its guarded 64-file patch is `595824ef6eb8e15872be058668503cedad1dd9a5ff98ac4700c6f713251d384c`; manifest `b86b550d7ee39fd93d5202accd5f77c975b7bcb4fd8f8e7706c69c5f279294a2` binds every original packet, overlap resolution and current source/API guard.

The follow-on passed custody replay guards, source syntax, pure metadata checks and feature-graph inspection. That evidence does not establish Rust type checking, a GPU optimizer run, a real newly trained checkpoint, or a successful live trained-checkpoint consumption loop. The demonstrated teacher count under this pending ASM admission contract remains zero in the inspected evidence. Historical lineage-B teaching corpora and its accepted trained checkpoint are separate retained assets, described in [§](#c-data-estate--retained-corpus-sizes) and [§](#c-training-history--carried-state-accepted-step-60000-and-rtx-step-60100); they have not been retroactively certified under the new admission contract. Source support, fixture tests, exact-head CI, live qualification and serving promotion must remain separate status fields.

All commands below involving `jope train` or experimental consumer configuration describe the **pending overlay's implemented interface**. They are a maintainer runbook for integration and authorized acceptance, not claims that these commands were executed at main `a75e543d2` (PR83). Preserve this status notice when extracting examples.

**Historical interface status as of 2026-09-23 about 23:55 UTC.** The implementation snapshot is unchanged: reviewed `68f56404` merged as main `a75e543d2`. At that observation, main had advanced to `bdfc95c63` (PR85), whose changes since `a75e543d2` touch only documentation, `README.md` and the coordination ledger. Training64 remains a stopped patch outside main; no training GPU window has been admitted and no training clock has started. Each runbook in this chapter carries exactly one status label per route: **available**, **implemented but unqualified**, **pending integration** or **proposed**. The owner's edition-2 design requirements appear here as rules or proposals, not as achieved results.

<a id="c-training--the-four-continuation-names-and-their-commands"></a>

## 7.2 The four continuation names and their commands

<a id="c-training--names-runbooks-and-status"></a>

### 7.2.1 Names, runbooks and status

Every way this chapter starts or continues training has exactly one of four names. Report a run by its name rather than by a command flag, and never describe one kind of run with another's name. [§](#c-plan--the-continuation-interfaces-that-exist-today) lists the same interfaces at a glance; this chapter remains the operational runbook.

| Name | Definition | Runbooks in this chapter, with status |
|---|---|---|
| Cold initialization | New parameters from the declared architecture and seed, with a fresh optimizer; no parent weights or optimizer state | Native `jope-parallel train` with neither `--initialize` nor `--resume`, which records initialization kind `fresh`: implemented but unqualified (dispatch held). `jope-py train-policy` without `--init-from`: implemented but unqualified. The pending campaign's first pass when its inputs name no initialization checkpoint: pending integration. |
| Weight warm start | Parent weights only; the optimizer starts fresh | Native `--initialize` with no replay policy (recorded as fresh FP32 AdamW, with the new corpus starting at step zero) or with a policy whose `optimizer` is `reset`: implemented but unqualified (held). `jope-py train-policy --init-from` in [§](#c-training--weight-warm-start-with-the-existing-cli): implemented but unqualified. The pending campaign's first pass from an initialization checkpoint: pending integration. |
| Explicit new-corpus transition | Parent weights **and** optimizer state carried onto a different corpus, with the optimizer carry and the sampler initialization both declared | Native `--initialize` with a verified `parallel.replay_policy.v2` policy whose `optimizer` is `carry`; the run records "carried FP32 AdamW with bias-correction step" and a lineage-bound sampler counter whose local block step starts at zero: implemented but unqualified (held). No authenticated Python new-corpus transition exists in this snapshot: the pending trainer's strict input provenance rejects a changed corpus on resume, while the existing Python API can restore optimizer state without binding corpus identity. Sibling branches built on this transition are proposed ([§](#c-training--proposed-sibling-fresh-and-replay-branches)). |
| Exact interrupted resume | The original run continues from its own saved state, with its original inputs and original deadline; no allowance is renewed | Native `--resume` in [§](#c-training--the-implemented-command-shape): implemented but unqualified (held). The Python API and its companion helper in [§](#c-training--exact-optimizer-continuation-through-the-python-api): implemented but unqualified. Pending `train resume` in [§](#c-training--plan-run-resume-verify-and-export): pending integration. |

Two passes through one corpus are continuation, not two generations: the pending campaign's second pass continues the same run on the same corpus inside the same window ([§](#c-training--plan-run-resume-verify-and-export)).

**Historical instances.** Lineage B's A100 recovery blocks each trained on a distinct corpus while carrying FP32 AdamW, the historical form of a new-corpus transition. Their accepted 512x6 step 60,000 remains the serving model and cannot be exactly resumed, because its state records no `execution_partition`. RTX step 60,100 is a historical hardware continuation: carried AdamW on the unchanged corpus with zero fresh programs, not a production continuation or a promotion. Step 69,000 remains unaccepted. See [§](#c-training-history--carried-state-accepted-step-60000-and-rtx-step-60100).

<a id="c-training--rules-for-exact-interrupted-resume"></a>

### 7.2.2 Rules for exact interrupted resume

These are owner requirements, **proposed**: neither PR83 main nor the stopped training64 overlay implements both of them.

1. **Reject optimizer fallback before any update.** An exact resume whose optimizer state is missing or marked incompatible must stop before its first update. It must never continue with a fresh optimizer under the resume name.
2. **Prove state parity.** A deterministic run that is interrupted and resumed must reach the same parameters, optimizer moments, scheduler, RNG, sampler position and counters as the same run left uninterrupted.

Present behavior against these rules:

- The Python API starts a fresh optimizer, scheduler and scaler when a resume artifact marks its optimizer state incompatible (main `train_v3.py:1919–1930`; pending `train_v3.py:2095–2111`). The companion helper refuses such an artifact before it calls the trainer, so it meets rule 1 for its own narrow case; it provides no parity proof.
- The pending campaign checks `optimizer_state_resumed` only after a pass has trained (pending `training_campaign.py:771`), so a fallback can consume updates and window time before it is rejected.
- Resume artifacts are written at epoch end without a minibatch cursor, and the campaign commits whole passes: an interrupted pass restarts from the last committed head inside the original window (pending `training_campaign.py:799`, `:1065`). That re-runs the pass; it is not an update-level continuation.
- Native `--resume` checks the execution partition, sampler, objectives, corpus, world, global batch and seed against the checkpoint's recorded state and optimizer metadata before CUDA (`crates/jope-engine/src/training.rs:931`, `:968–977`), and repeats the partition, sampler and objective checks after CUDA initialization (`:1023–1027`, `:1106–1110`). This manual has no parity evidence for it, and its dispatch is held.
- No Python test in that snapshot compares interrupted and uninterrupted trajectories: the pending `test_policy_training_contract.py:126–196` checks only changed parameters, increasing optimizer counters, compatible reload and contract rejection. Two native checks at `a75e543d2` do compare restored against uninterrupted state: the CPU unit test `cpu_optimizer_restore_completes_original_window_identically` (`crates/jope-engine/src/neural.rs:1359`) shows that a saved and reloaded AdamW finishes a three-step window with parameters and moments identical to the uninterrupted optimizer, and the driver `python/src/jope/tools/qualify_training_recovery.py` compares the final parameter hash of an eight-rank run with an injected rank failure against an uninterrupted run. Neither compares scheduler, RNG or sampler state, neither exercises the Python route, and this manual cites no receipt from that driver.

The proposed follow-up is an explicit strict-resume mode that rejects before updates; update-boundary checkpoints that carry the sampler permutation and cursor, counters and all RNG state; and deterministic equality tests of interrupted against uninterrupted runs. Warm-start behavior keeps its own name. Source: Codex's read-only analysis `boundary-resume-siblings.md` in custody `plan-execution-20260923`.

<a id="c-training--variables-used-in-the-continuation-examples"></a>

### 7.2.3 Variables used in the continuation examples

The paths below are operator-supplied placeholders, not newly admitted inputs. Resolve them from the compatible original run inventory; use absolute paths and preserve its hashes, resource limits and original time window. The native command remains held. The pending commands require reviewed integration and the admission checks described below.

| Variables | Required meaning |
| --- | --- |
| `JOPE_PARALLEL_NEURAL`, `NATIVE_ROOT` | Source-bound native executable built with `neural`, and its authenticated native data/run root; these paths do not release the dispatch hold. |
| `QUALIFIED_NATIVE_CORPUS`, `VERIFIED_NATIVE_PARENT`, `VERIFIED_CARRY_POLICY` | Compatible corpus directory with verified allowed input closure, original native checkpoint directory, and validated optimizer-carry policy JSON. Do not substitute a Python checkpoint or blindly reuse the legacy corpus. |
| `NATIVE_RUN` | Output directory for the declared native run; an interrupted resume retains that same directory, inputs and original allowance. |
| `PYTHON_ENV`, `TRAINING_PYTHON` | Verified Python environment prefix and the matching Python executable, respectively. |
| `PYTHON_CORPUS`, `TRUSTED_PYTHON_POLICY` | Compatible admitted Python JSONL corpus and trusted weight artifact for the warm-start example; warm-start weights alone do not restore optimizer state. |
| `PYTHON_RESUME`, `ORIGINAL_SETTINGS` | Trusted `PrimePolicyTrainerResume.v1` artifact file and companion JSON containing its original compatible settings and audited input inventory. These are separate from the native checkpoint directory. |
| `PYTHON_CORPUS_SHA256`, `PYTHON_RESUME_SHA256`, `ORIGINAL_SETTINGS_SHA256` | Expected file SHA256 values from the original authenticated input inventory, checked before the helper imports Torch. |
| `NEW_RUN` | New private output directory for the Python example; the continuation helper requires it not to exist and refuses it under `/tmp` or inside a Git working tree. This new output location does not renew a resource or time allowance. |
| `FINAL_JOPE`, `FINAL_SOURCE_SHA` | Future source-bound CLI and full source commit after the pending training integration and its required checks; not an executable built from the current main `a75e543d2`. |
| `LEARNING_STATE`, `CORPUS_REQUEST`, `CORPUS_REQUEST_SHA256`, `CORPUS_OUTPUT` | Admitted learning state, exact corpus-preparation request and its file SHA256, and private prepared-input output directory, as specified in [§](#c-training--build-the-real-corpus-from-admitted-evidence). |
| `TRAINING_STATE`, `TRAINING_EXPORT` | Original supervised training state directory and private experimental export output directory; resume reuses `TRAINING_STATE`. |

<a id="c-training--native-lineage-b-the-held-command-and-its-blockers"></a>

### 7.2.4 Native lineage B: the held command and its blockers

| Runbook part | Native lineage-B continuation |
|---|---|
| Purpose and status | Continue lineage B natively: a weight warm start or an explicit new-corpus transition with `--initialize`, an exact interrupted resume with `--resume`. **Implemented but unqualified**: every direct dispatch stops at the dispatch hold. |
| Prerequisites and resources | A source-bound `neural` build; a compatible corpus with verified allowed input closure; a verified parent checkpoint and carry policy. The controller's fresh-telemetry check requires at least 128 GiB available RAM and 50 GiB free disk, more than the approved 64 GiB limit. |
| Manifest-creation command | None is qualified in this manual. Main contains a corpus publisher (`jope-parallel build-corpus`) and code that writes `parallel.replay_policy.v2` policies, but none has been qualified for this runbook. Supply an already verified corpus and policy, never a fabricated one. |
| Execution | The commands in [§](#c-training--the-implemented-command-shape), which are not to be dispatched under the present allocation. |
| Artifacts and success | Checkpoint directories whose `state.json` binds model, optimizer, configuration, corpus and vocabulary hashes, plus the run's `initialization.json`. No success receipt exists for this held route. |
| Failure and resume | `--resume` applies only to an interrupted original run with the identical window and inputs. A completed run refuses resume, and accepted step 60,000 cannot be resumed ([§](#c-training--rules-for-exact-interrupted-resume)). |
| Limitations | The dispatch hold, the resource gate and the world-8 partition of the accepted lineage ([§](#c-training--the-dispatch-hold-resource-incompatibility-and-valid-topologies)). |

<a id="c-training--the-implemented-command-shape"></a>

#### 7.2.4.1 The implemented command shape

This is the exact implemented command shape, **not an instruction to dispatch it under the present allocation**:

```sh
"$JOPE_PARALLEL_NEURAL" --root "$NATIVE_ROOT" train \
  --corpus "$QUALIFIED_NATIVE_CORPUS" --output "$NATIVE_RUN" \
  --world 1 --microbatch 64 --steps 100 --checkpoint-steps 50 \
  --initialize "$VERIFIED_NATIVE_PARENT" --replay-policy "$VERIFIED_CARRY_POLICY"

# Only an interrupted original run, with identical original window and inputs:
"$JOPE_PARALLEL_NEURAL" --root "$NATIVE_ROOT" train \
  --corpus "$QUALIFIED_NATIVE_CORPUS" --output "$NATIVE_RUN" \
  --world 1 --microbatch 64 --steps 100 --checkpoint-steps 50 \
  --resume --replay-policy "$VERIFIED_CARRY_POLICY"
```

Here `--steps 100` declares the same increment for the whole original run; resume does not authorize another hundred steps. A completed run explicitly refuses resume. `--initialize` alone starts fresh FP32 AdamW and resets the new-corpus sampler; preserving optimizer ancestry requires the verified carry policy and compatible objective/partition evidence. Do not fabricate that policy JSON or relabel an incompatible recovered checkpoint. Corpus, vocabulary, model, optimizer, config, sampler, partition and deadline checks occur before CUDA. In this chapter's names, the first command is a weight warm start when the policy's `optimizer` is `reset`, and an explicit new-corpus transition when it is `carry` and the corpus differs from the parent's; the second command is an exact interrupted resume.

<a id="c-training--the-dispatch-hold-resource-incompatibility-and-valid-topologies"></a>

#### 7.2.4.2 The dispatch hold, resource incompatibility and valid topologies

The current direct controller calls `require_original_gpu_grant`, which **always returns an error**: “original native GPU grant verifier unavailable; numerical dispatch is held.” Environment variables cannot release this gate. A separate typed native funding-admission API exists, but this is not a working direct-trainer bypass. Even after a future proper dispatch integration, the controller's fresh-telemetry check requires at least 128 GiB available RAM and 50 GiB free disk, incompatible with the current approved 64 GiB limit. Valid normal microbatches are 64, 128 or 256, and global batch is 2048. With world 1/microbatch 64, accumulation is 32. The retained step 60100 hardware qualification actually used world 1, microbatch 256 and accumulation 8, carrying AdamW from 60000 to 60100 on the unchanged corpus with zero fresh program presentations. Its 100-step update is genuine optimizer work; it is not production continuation or promotion. The ADR's microbatch 32 example is inaccurate for that retained run and is not a currently accepted CLI topology.

Sources: [native command parser](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-engine/src/main.rs#L106), [controller admission and step window](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-engine/src/controller.rs#L479), [dispatch hold](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-engine/src/task_contract.rs#L88), [checkpoint checks and warm start](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-engine/src/training.rs#L923).

<a id="c-training--current-python-training-and-a-concrete-continuation-call"></a>

### 7.2.5 Current Python training and a concrete continuation call

<a id="c-training--weight-warm-start-with-the-existing-cli"></a>

#### 7.2.5.1 Weight warm start with the existing CLI

For a compatible, already admitted Python corpus, the existing CLI can launch a weight warm start:

```sh
"$PYTHON_ENV/bin/jope-py" train-policy \
  --corpus "$PYTHON_CORPUS" --checkpoint "$NEW_RUN/policy.pt" \
  --init-from "$TRUSTED_PYTHON_POLICY" --epochs 1 --batch-size 256 \
  --lr 0.0005 --device cuda:0 --seed 7 \
  --d-model 256 --n-layers 2 --ffn-dim 1024 --n-heads 4 \
  --metrics-jsonl "$NEW_RUN/metrics.jsonl" \
  --progress-jsonl "$NEW_RUN/progress.json"
```

This example uses the stated 256-width architecture; it is suitable only when the intended experiment explicitly uses that architecture. Warm starts may shape-filter or prefix-copy weights. The script's default model, architecture and vocabulary must not be assumed to reproduce the accepted native step 60000 model. The CLI command is source-checked but was not executed for this chapter.

| Runbook part | Python weight warm start |
|---|---|
| Purpose and status | Start a new Python optimization run from compatible weights: a weight warm start. **Implemented but unqualified**: the CLI exists at `a75e543d2` and was source-checked, not executed, for this chapter. |
| Prerequisites and resources | A verified Python environment, an admitted compatible JSONL corpus, a trusted weight artifact of the stated architecture, and one CUDA device inside a separately admitted resource/time supervisor. |
| Manifest-creation command | No manifest-creation command exists at `a75e543d2` for an admitted corpus: the admission contract and its exporter, `train prepare-inputs`, are pending integration ([§](#c-training--build-the-real-corpus-from-admitted-evidence)). |
| Execution | The command above. |
| Artifacts and success | `policy.pt`, `policy.imitation.pt`, `metrics.jsonl` and `progress.json` under `$NEW_RUN`. With `--epochs 1` and the default `--phase-a-fraction 0.4`, the single epoch is phase A, so the trainer also writes the phase-A imitation checkpoint `policy.imitation.pt` beside `policy.pt`. A completed run shows an optimization run only, not eligibility, optimizer continuation or improvement. |
| Failure and resume | The CLI exposes no `--resume-from`. A failed warm start is a failed run; repeating it is a new run in a new directory, without a renewed allowance. |
| Limitations | Shape-filtered or prefix-copied weights, defaults that do not reproduce the accepted model, and the device fallback in [§](#c-training--limits-of-the-current-python-route). |

<a id="c-training--exact-optimizer-continuation-through-the-python-api"></a>

#### 7.2.5.2 Exact optimizer continuation through the Python API

For **actual optimizer continuation**, call the existing API from a private maintainer script. The following is the essential code; `settings.json` must contain the original compatible `kwargs`, and all referenced inputs must be hash-pinned in the operator's original run inventory. The complete companion `continue-python-policy.py` includes those byte-hash checks, exclusive output creation, explicit GPU preflight, refusal of distributed or non-current resume modes, and refusal of an `--output` under `/tmp` or inside a Git working tree (the path itself or any parent directory containing a `.git` entry). It is an illustrative API caller, not a new installed product command or a qualified supervisor.

```python
from jope.policy.train_v3 import train_policy_v3_from_jsonl

result = train_policy_v3_from_jsonl(
    corpus=verified_corpus_path,
    checkpoint=new_output / "policy.pt",
    resume_from=verified_resume_path,
    resume_checkpoint=new_output / "resume.pt",
    metrics_jsonl=new_output / "metrics.jsonl",
    progress_jsonl=new_output / "progress.json",
    epochs=1,                        # one additional epoch, not target epoch 1
    checkpoint_state="current",     # keep export and optimizer state aligned
    allow_train_val_fallback=False,
    amp=False,
    autotune_batch_size=False,
    device="cuda:0",
    **original_compatible_settings,
)
```

<a id="c-training--the-companion-continuation-helper-and-the-resume-artifact"></a>

#### 7.2.5.3 The companion continuation helper and the resume artifact

A concrete invocation of the companion is below. Without the final switch it verifies the supplied byte inventories and prints the call without importing Torch. Fill settings from the original run; do not copy arbitrary architecture, routing or vocabulary choices into a resume. An `audited_input_files` list does not itself prove closure completeness or teacher eligibility.

```sh
MANUAL_EXAMPLES="$R/docs/internal/asm-manual/examples"
"$TRAINING_PYTHON" "$MANUAL_EXAMPLES/continue-python-policy.py" \
  --settings "$ORIGINAL_SETTINGS" --settings-sha256 "$ORIGINAL_SETTINGS_SHA256" \
  --corpus "$PYTHON_CORPUS" --corpus-sha256 "$PYTHON_CORPUS_SHA256" \
  --resume "$PYTHON_RESUME" --resume-sha256 "$PYTHON_RESUME_SHA256" \
  --output "$NEW_RUN"
# Numerical execution, only inside the separately admitted supervisor:
# append --execute-diagnostic to that same command.
```

The settings object uses the illustrative schema `manual.python-resume-settings/v1`, an `audited_input_files` array of `{path, sha256}`, and a `kwargs` object containing exactly: `batch_size`, `lr`, `seed`, `phase_a_fraction`, `checkpoint_metric`, `greedy_eval_datasets`, `d_model`, `n_layers`, `ffn_dim`, `n_heads`, `prime_vocab`, and `routing_classes`. These values must come from the compatible original configuration. This helper is deliberately limited to ordinary current-state resumes: staged schedules, anti-overfit experiments and distributed training require their own original configuration and proof. Only syntax/API and metadata-path checks were performed for this new example; no model was loaded or trained.

Use the trusted `PrimePolicyTrainerResume.v1` artifact, not `model.ot`, a safetensors export, or a selected `policy.pt` without optimizer state. Existing current-state resume loads model state strictly, verifies action tokens and routing classes, restores optimizer/scheduler/RNG, and appends epochs after `global_epoch`. A best/EMA selected checkpoint may carry an explicit optimizer-incompatible marker and cause a fresh optimizer; that is not accepted as exact continuation by the companion example. Sources: [restore and corpus partition](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/python/src/jope/policy/train_v3.py#L1453), [strict model restore](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/python/src/jope/policy/train_v3.py#L1692), [optimizer compatibility](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/python/src/jope/policy/train_v3.py#L1918), [resume artifact contents](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/python/src/jope/policy/train_v3.py#L2498).

| Runbook part | Python exact interrupted resume (companion helper) |
|---|---|
| Purpose and status | Continue an ordinary current-state Python run through the existing API from its epoch-end resume artifact. This is an exact interrupted resume only when the settings, corpus and resume artifact are the original run's and its original window is still open; the helper cannot check the window. **Implemented but unqualified**: only syntax/API and metadata-path checks were performed. |
| Prerequisites and resources | The trusted resume artifact, original settings, corpus and audited input closure, each with its expected SHA256; one visible RTX 3090; a separately admitted resource/time supervisor. |
| Manifest-creation command | No manifest-creation command exists: the operator writes `settings.json` in the illustrative `manual.python-resume-settings/v1` schema from the original run's configuration. |
| Execution | The helper without `--execute-diagnostic` (metadata only, no Torch import), then the same command with it inside the supervisor. |
| Artifacts and success | `policy.pt`, `resume.pt`, `metrics.jsonl`, `progress.json` and `api-result.json` in the new output directory. The result is scoped as a diagnostic API call, not product acceptance. |
| Failure and resume | Refuses changed input digests, missing or unknown settings, a resume artifact that is not an ordinary current-state resume with compatible optimizer state, more than one process, anything other than one visible RTX 3090, an existing output directory, and an output under `/tmp` or inside a Git working tree. A retry needs a new output directory and keeps the original window. |
| Limitations | No interrupted-versus-uninterrupted parity proof and no in-epoch cursor ([§](#c-training--rules-for-exact-interrupted-resume)). It fixes `amp=False` and compares only the four architecture fields with the artifact, so it does not detect a changed AMP or batch-size regime. |

<a id="c-training--limits-of-the-current-python-route"></a>

#### 7.2.5.4 Limits of the current Python route

This current Python route has material limits. CUDA requests can silently fall back to CPU in the underlying code, so the companion refuses absent CUDA before calling it. Current task-routing/zero-update and resume-proof fixes belong to the pending training overlay. The loader can discard minority embedding widths with a warning; a qualified input inventory should reject such a mixed corpus earlier. Existing Python training alone does not provide the shared cgroup resource contract, original campaign deadline, teaching admission, parameter-change proof or serving promotion. Execute any diagnostic caller only inside a separately admitted resource/time supervisor, not as a way to bypass the pending product gates. See [current device fallback](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/python/src/jope/policy/ddp.py#L25) and [mixed-width handling](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/python/src/jope/policy/train_v3.py#L1497).

<a id="c-training--evidence-and-identity-before-a-bounded-campaign"></a>

## 7.3 Evidence and identity before a bounded campaign

A training run has at least four separate identities:

| Identity | What it binds | Why it is separate |
|---|---|---|
| Learning state | Eligible HWM records and teaching evidence | Accumulated data authority is distinct from a single training attempt |
| Training plan | Task/category inputs, source, runtime, numerics, seed, passes and resource policy | Changing it invalidates an existing run's compatibility |
| Execution window | Actual first admission and immutable deadline | Resume must use elapsed budget rather than restart the clock |
| Checkpoint/export | Parameters, architecture, vocabulary, task, optimizer ancestry and completed proof | A file's existence is not evidence of an update or eligibility |

Use durable directories outside Git working trees and outside `/tmp`. Keep corpus and model assets out of Git. The native front door inventories the actual Python interpreter, preserves its virtual-environment invocation path, captures environment and installed runtime identity, and supervises workers rather than treating a bare Python subprocess as resource isolation.

The training runtime inventory hashes installed dependency bytes, excluding generated bytecode, and records all installed distributions rather than only a short library allowlist. Version strings alone do not protect against modified installed files. The full source commit and source-package hashes are part of compatibility. An incompatible environment should be rebuilt/replanned before admission, not silently accepted during resume.

Sources: pending `crates/jope-cli/src/train.rs:1`; `python/src/jope/policy/training_campaign.py:120`.

<a id="c-training--the-bounded-training-campaign-chain"></a>

## 7.4 The bounded training campaign chain

| Runbook part | Bounded training campaign chain |
|---|---|
| Purpose and status | Train, verify and export bounded experimental checkpoints, one per planned category: experimental regression lineage v1 (256×2) from admitted real regression evidence, plus explicitly synthetic classifier fixtures, with a smaller architecture, only where real classification admission fails. **Pending integration**: the training64 patch is outside main and has never run on a GPU. |
| Prerequisites and resources | Reviewed integration with required checks at the final head, a source-bound `jope` build, fresh real teacher admission covering at least two independent families, and an admitted training window. Resources: one RTX 3090 and one GPU-hour in total, at most 20 minutes per category, 20 CPUs, 64 GiB and zero swap. |
| Manifest-creation command | `train prepare-inputs` writes the inputs manifest (`inputs.json`) from a `jope.real-training-corpus-request/v1` request, and `train plan` freezes the plan without starting a clock. No command, at `a75e543d2` or in the pending overlay, writes the request itself; its comparison proofs must come from actual admission ([§](#c-training--build-the-real-corpus-from-admitted-evidence)). |
| Execution | `train run`, or `train resume` after an interruption inside the original windows ([§](#c-training--plan-run-resume-verify-and-export)). |
| Artifacts and success | Per-category attempts, receipts and heads in the state directory. `train verify` succeeds only with positive updates, changed parameters, exact reload, resumed optimizer state and at least two committed passes; `train export` then writes a private experimental bundle with a file inventory. |
| Failure and resume | An exhausted window is recorded as `budget_exhausted`. Resume keeps the plan, the state and the original global and category deadlines; an interrupted pass restarts from the last committed head ([§](#c-training--rules-for-exact-interrupted-resume)). |
| Limitations | Real corpus preparation supports regression only. With two prospective families it is a mechanics qualification, not a flywheel test. Input is JSONL without a packed reader, there are no sibling fresh/replay branches, and export never moves the serving pointer. |

<a id="c-training--what-the-pending-campaign-trains"></a>

### 7.4.1 What the pending campaign trains

The pending training campaign uses the existing Python `JopePolicyModelV3` with task-routed reward/stability behavior. Real admitted training uses width 256, two layers, four heads and feed-forward 1024; synthetic qualification fixtures use width 8, one layer, two heads and feed-forward 16. Both consume the explicit context schema. Training uses one-epoch passes, learning rate 5e-4, batch 256 for admitted rows or 2 for fixtures, AMP disabled and automatic batch-size tuning disabled. These are the bounded campaign's settings, not the accepted step-60000 model's architecture (width 512, six layers, feed-forward 2048; [§](#c-architecture--checkpoint-binding-and-configuration)). The campaign therefore produces a **new, smaller model family**, not a continuation of step 60,000: its optional initialization is a Python policy payload, and its checkpoint cannot be loaded in place of the accepted model, whose loader rejects incorrectly shaped tensors. Compare it with step 60,000 only through matched downstream fits under the same protocol, never through loss values.

**Naming.** The regression line that this campaign would start is called **experimental regression lineage v1 (256×2, new initialization)**. It is pending integration and has not been trained. The planned initial model, M_init, uses new parameters from the declared architecture and seed, with no parent weights. The campaign also exposes an optional weight-warm-start interface, but this plan has not selected it; its shape-filtered loading is not proof of full checkpoint compatibility. M_init is named M0 only after it has actually trained on the lineage's first admitted corpus, C0, and that run has passed verification; until then no M0 exists. The line stays separate from the accepted 512×6 step 60,000 of lineage B: it is neither its successor nor a replacement for it.

<a id="c-training--build-the-real-corpus-from-admitted-evidence"></a>

### 7.4.2 Build the real corpus from admitted evidence

`train prepare-inputs` takes a request, not hand-written training examples. The request schema binds requested tasks, seed, family registry, vocabulary, encoder and optional initialization checkpoint. Its `contexts` map has one independently pinned proof for every eligible comparison. Each proof includes producer/source, original input, qualification/reference manifest, vector/artifact, frozen transforms, component bundle, actual context execution and eight case executions.

The exporter reopens the original exact fitting input, verifies feature schema/values/labels and row order, verifies complete context admission, and checks each recipe against exact executable vocabulary bins. It refuses parameter snapping, wrong-task heads and fabricated vectors. It then writes corpus rows with 1536 dataset plus 512 target dimensions, family-disjoint train/development metadata and retained source/context identities.

```sh
"$JOPE_FINAL_CLI" train prepare-inputs \
  --learning-state "$JOPE_LEARNING_STATE" \
  --request "$JOPE_CORPUS_REQUEST" --request-sha256 "$JOPE_CORPUS_REQUEST_SHA" \
  --output "$JOPE_CORPUS_OUTPUT"
```

Do not author a request using invented eligible comparison IDs: these must come from actual successful admission. The exported request requires all eligible requested comparisons. At least two independent families are necessary for a real train/development partition. Source: pending `crates/jope-cli/src/train/corpus.rs:43`, `:195`, `:362`, `:485`.

<a id="c-training--plan-run-resume-verify-and-export"></a>

### 7.4.3 Plan, run, resume, verify and export

The intended real all-category path preserves real admitted regression while allowing explicitly labeled synthetic fallback only for classification categories whose real admission fails:

```sh
"$JOPE_FINAL_CLI" train plan \
  --state "$JOPE_TRAINING_STATE" --python "$JOPE_TRAINING_PYTHON" \
  --inputs "$JOPE_CORPUS_OUTPUT/inputs.json" \
  --task all-admitted --device cuda:0 --source-sha "$JOPE_FINAL_SOURCE" \
  --learning-state "$JOPE_LEARNING_STATE" --seed 7 --passes 2

"$JOPE_FINAL_CLI" train run --state "$JOPE_TRAINING_STATE"
"$JOPE_FINAL_CLI" train verify --state "$JOPE_TRAINING_STATE"
"$JOPE_FINAL_CLI" train export \
  --state "$JOPE_TRAINING_STATE" --output "$JOPE_TRAINING_EXPORT"
```

If interrupted before the original deadlines and while all plan/runtime/input pins still agree, continue with:

```sh
"$JOPE_FINAL_CLI" train resume --state "$JOPE_TRAINING_STATE"
```

Do not use the top-level `--synthetic-fixture` switch in place of missing real regression evidence. For deliberately small mechanical CPU qualification only, the explicit interface is `train plan --synthetic-fixture --device cpu`, with the same source/runtime/state requirements. Such a result stays synthetic and does not satisfy the real-data task.

The planner records requested, admitted, unavailable and fallback categories. With `all-admitted`, a missing real regression category can remain explicitly unavailable while classifier fixtures proceed; that is partial coverage and cannot be reported as completed real-regression training. Current real-classification corpus admission is not qualified. Binary/multiclass fixture fallback retains its own synthetic provenance and task labels.

Verification must establish positive optimizer updates, changed parameter digests, exact reload, an actual resumed optimizer state and increasing step counters, at least two committed passes, complete planned category coverage where claimed, and compatible device/source/runtime/vocabulary/input identities. The worker reloads both `policy.pt` and `resume.pt` and requires equal parameter proof. Each pass belongs to the original category/global windows and contributes an immutable execution/receipt chain. A zero-update checkpoint or interrupted partial output cannot become a successful pass. Both planned passes read one corpus inside one category window, so the second pass is continuation within one generation, not a second LOOP-01 generation. It supplies the optimizer-continuation proof: verification refuses a category head with fewer than two committed passes (pending `training_campaign.py:1210`). Verification establishes training mechanics only. An eligible teacher, a new discovery, information actually consumed by the optimizer and an improved policy remain four distinct claims, each needing its own evidence.

Export first performs verification, then writes the exact plan, verification and category artifacts into a private output bundle with a file inventory. It creates experimental training artifacts; it does not update the accepted serving pointer. Sources: pending `python/src/jope/policy/training_campaign.py:744`, `:780`, `:1043`, `:1203`, `:1303`.

<a id="c-training--the-pending-bounded-extension-path-sequence-and-limits"></a>

### 7.4.4 The pending bounded extension path: sequence and limits

The stopped training64 patch is `595824ef6eb8e15872be058668503cedad1dd9a5ff98ac4700c6f713251d384c`, with manifest `b86b550d7ee39fd93d5202accd5f77c975b7bcb4fd8f8e7706c69c5f279294a2`. It has not been integrated into main (`a75e543d2`, PR83) or qualified by a fresh real GPU run. Its exact command contract is described in [§](#c-training--plan-run-resume-verify-and-export); pending files cannot honestly be linked as existing GitHub files at main.

After reviewed integration, source-bound build/tests, fresh real two-family teacher admission, and verified corpus preparation, the operational sequence is:

```sh
"$FINAL_JOPE" train prepare-inputs \
  --learning-state "$LEARNING_STATE" \
  --request "$CORPUS_REQUEST" --request-sha256 "$CORPUS_REQUEST_SHA256" \
  --output "$CORPUS_OUTPUT"
"$FINAL_JOPE" train plan --state "$TRAINING_STATE" \
  --python "$TRAINING_PYTHON" --inputs "$CORPUS_OUTPUT/inputs.json" \
  --task all-admitted --device cuda:0 --source-sha "$FINAL_SOURCE_SHA" \
  --learning-state "$LEARNING_STATE" --seed 7 --passes 2
"$FINAL_JOPE" train run --state "$TRAINING_STATE"
# If interrupted within the original allowance, retain the same plan/state:
"$FINAL_JOPE" train resume --state "$TRAINING_STATE"
"$FINAL_JOPE" train verify --state "$TRAINING_STATE"
"$FINAL_JOPE" train export --state "$TRAINING_STATE" --output "$TRAINING_EXPORT"
```

Run and resume are alternatives according to state; do not blindly execute resume after a completed run. The whole campaign permits one RTX 3090 and one GPU-hour, at most 20 minutes per category, 20 CPUs, 64 GiB and zero swap. Two passes share each category window; they do not receive fresh time allowances. At least two real admitted families are necessary before claiming the real regression training path; with only the two prospective panel families of [§](#c-teachers--present-teacher-readiness-and-the-smallest-real-path), the campaign is a mechanics qualification, not a flywheel test. Classification fixtures may demonstrate labeled mechanics when real admission fails, but remain synthetic. Unknown or unavailable categories are reported rather than filled with invented success.

Verification must establish positive updates, changed parameters, reload equality, actual optimizer continuation, full input/runtime/source/device/vocabulary compatibility and persisted accounting. Export is experimental. The pending `experimental.regression`, `experimental.binary` and `experimental.multiclass` consumer entries bind task-specific exported checkpoints; a later real fit must prove the new checkpoint generated and fitted candidates. None of these commands updates the accepted serving pointer. Product28, warm-target and winner-export follow training64; they do not fix teacher eligibility by themselves.

<a id="c-training--proposed-sibling-fresh-and-replay-branches"></a>

### 7.4.5 Proposed sibling fresh and replay branches

**Status: proposed.** No implementation exists: the pending campaign advances one mutable head per category over one pinned corpus. It has no immutable sibling parent, branch identifier, committed-update target, 25/75 sampler allocation or per-update fresh-exposure record (pending `training_campaign.py:799`, `:1065`). The owner's design for one controlled round is:

- Two sibling branches, F (fresh) and R (replay only), start from one immutable parent model **and** optimizer state and share the original deadline; neither receives a new window.
- Both commit the same number of updates and batch slots. F samples 25% fresh and 75% replay examples; R samples replay only.
- Each records measured fresh exposure, duplicate and unique families, non-padding tokens and GPU cost.
- Neither branch is promoted to serving.

Codex's read-only analysis (`boundary-resume-siblings.md` in custody `plan-execution-20260923`) proposes three further details: each branch also records its consumed example identities; the unchanged parent is kept as a comparison artifact; and moving onto the new corpus is an explicit new-corpus transition that declares optimizer carry and sampler initialization, not a resume.

The first causal experiment keeps architecture, objective, corpus and sampler changes distinguishable, so it does not change the architecture or the reward representation alongside these branches. Within the existing 20-minute regression GPU cap, the two controlled LOOP-01 rounds may remain unfunded or inconclusive. An unfunded round, or one with insufficient data, yields mechanical proof and descriptive results, and its causal outcome is reported as inconclusive, not as success. [§](#c-learning--loop-01-the-central-regression-milestone) defines LOOP-01.

<a id="c-training--training-input-packing-and-the-learning-store"></a>

### 7.4.6 Training input packing and the learning store

**Status: proposed.** The pending campaign still reads JSONL into memory and writes a per-attempt `corpus.jsonl`; no packed-reader scalability qualification exists ([§](#c-teachers--python-training-input-and-the-pending-bounded-campaign)). Packing work follows phase profiling, in this order:

1. Profile the preparation phases, as in [§](#c-teachers--concrete-corpus-improvements-with-acceptance-criteria).
2. Add a verified Python packed-reader bridge.
3. Reuse immutable shards, with bounded staging.
4. Compress.
5. Adopt a narrower record layout only if measurement justifies it.

Every step preserves float32 supervision, the legal-set mappings and the 128-token grammar, and must show record, batch, loss, gradient, update and resume parity ([§](#c-data-estate--compression-the-proposed-180-byte-layout-and-label-precision)). Keep one learning-store writer until an atomic server-side pointer update is qualified. Backup, replication and retention operations remain separately authorized; nothing in this chapter authorizes them ([§](#c-data-estate--recorded-custody-risks-and-disk-capacity)).

<a id="c-training--consuming-a-newly-trained-checkpoint"></a>

## 7.5 Consuming a newly trained checkpoint

<a id="c-training--the-experimentalregression-configuration-entry"></a>

### 7.5.1 The experimental.regression configuration entry

The pending final regression consumer is selected by the `experimental.regression` entry of the complete `jope.asm-configuration/v1` object. This entry is an object with:

| Field | Required meaning |
|---|---|
| `schema` | `jope.experimental-python-configuration/v1` |
| `task` | `regression`, `binary`, or `multiclass`, matching its configuration key |
| `runtime` | Full pinned PythonRuntime object, including program, manifest, source/package files and runtime digest |
| `export_manifest` | Exact exported training manifest artifact |
| `checkpoint` | Exact task's exported `policy.pt` artifact |
| `vocabulary` | Exact frozen action vocabulary artifact |
| `canonicalizer` / `canonicalizer_build` | Native executable and actual successful build evidence |
| `canonicalizer_features` | Exact feature set from Cargo artifact messages |
| `sampling` | Bounded count, temperature, top-p and maximum sequence length |
| `generic_draws` | Declared bounded generic bank allowance |

Artifacts use `{path, sha256}` with absolute paths and real file hashes. Runtime is not merely a Python executable/version: source files and installed package files must be inventoried. The native canonicalizer must have a successful build receipt whose source and source-manifest pins agree with the consumer. Do not copy an accepted checkpoint receipt into this export field.

<a id="c-training--compatibility-checks-and-fail-closed-rules"></a>

### 7.5.2 Compatibility checks and fail-closed rules

Regression additionally requires the export to identify real admitted regression training, the same source commit, and the exact native encoder identity. The consumer admits the complete native context and independently verifies architecture, task, vocabulary, checkpoint and proof. Classification uses its separate context lineage. Missing architecture fields, missing/extra model state, altered vocabulary, source changes, changed encoder assets or a different training task fail closed. No explicit experimental selection silently falls back to the accepted model.

The export records the complete architecture, vocabulary, task, source, input identities, model state and training proof. The pending experimental consumer requires all architecture fields explicitly; the general-purpose Python loader's backward-compatible defaults cannot reinterpret a saved experimental checkpoint. It compares the reconstructed model's architecture and full state to the export. Sources: pending `python/src/jope/policy/training_campaign.py:780`, `python/src/jope/policy/experimental_proposal.py`, `crates/jope-pipeline/src/experimental_producer.rs`.

The final regression follow-on closes a specific gap: `Configuration.experimental.regression` can explicitly select a newly trained regression export. Omission keeps the accepted baseline. This operational regression route requires **real admitted** training, the same source commit and the exact native encoder identity used by training. It first admits the qualified native2048 regression context, then wraps those exact values in the separate `asm.experimental-regression-context/v1` lineage with zero classes and `policy_compatible:false`. It does not promote the new checkpoint to accepted serving or weaken the accepted hard pins.

Binary and multiclass use their separate experimental context path; a synthetic classifier fixture can qualify mechanical training/consumption behavior only when explicitly reported as synthetic. Synthetic regression training is permitted as a CPU test but cannot be substituted into the operational real-regression consumer. There is no silent accepted-policy fallback after an explicitly chosen experimental checkpoint fails compatibility.

<a id="c-training--proving-consumption-in-a-new-operational-proof"></a>

### 7.5.3 Proving consumption in a new operational proof

After building a new configuration around the verified export, preregister a separately authorized operational proof using the same `optimize proof-plan` front door. Its report must bind the new checkpoint hash to actual generation-zero neural fits, retain fitted artifacts/predictions and, when evolution is claimed, show completed descendants with neural ancestry. Recover the report from the same original journal and deadline. Only after those facts are demonstrated should the final official panel configuration be frozen around that checkpoint.

Source: pending `crates/jope-bench/src/panel/experimental_proposal.rs:36`, `crates/jope-pipeline/src/experimental_producer.rs:36`, `python/src/jope/policy/experimental_proposal.py`; overlay documentation `docs/experimental-regression-consumption.md`.

| Runbook part | Consumption of a newly trained checkpoint |
|---|---|
| Purpose and status | Show that a verified experimental export generated and fitted real candidates. **Pending integration**: the `experimental.regression` consumer exists only in the stopped overlay. |
| Prerequisites and resources | A verified export, a complete configuration whose `experimental.regression` entry pins it ([§](#c-training--the-experimentalregression-configuration-entry)), and a separately authorized operational proof with its own resources. |
| Manifest-creation command | `optimize proof-plan` preregisters the proof; it exists at `a75e543d2`, without the experimental consumer. No command writes the configuration entry: in the pending source the schema string occurs only as the consumer's expected value, and the overlay documentation has the operator bind it. |
| Execution | The same front door's run and report steps, recovered from the original journal and deadline. |
| Artifacts and success | A report that binds the checkpoint hash to actual generation-zero neural fits, with retained fitted artifacts and predictions, and completed descendants with neural ancestry when evolution is claimed. |
| Failure and resume | Compatibility failures fail closed, with no fallback to the accepted model; recovery uses the original journal and deadline. |
| Limitations | Consumption is not improvement, and nothing here moves the serving pointer. |

<a id="c-training--remaining-acceptance-queue-and-maintenance-risks"></a>

## 7.6 Remaining acceptance queue and maintenance risks

The required next integration/acceptance work is sequential: apply guarded training64 after PR83 lifecycle completion; compile and run all required tests at the final head; admit fresh genuine teachers; materialize and verify the corpus; execute the one approved GPU hour; verify/export; consume the new checkpoint in actual fits; qualify fitted portability and warm/cold delivery; execute the unchanged30 panel and report all outcomes. The product chapters cover the remaining deployment stages in detail.

LOOP-01 is the central regression product milestone ([§](#c-learning--loop-01-the-central-regression-milestone)): a new eligible target, actual exposure, changed parameters, compatible reload, improved fitted search, then another new generation. No LOOP-01 generation has run. Once integrated and qualified, the pending training chain in this chapter can supply changed parameters and compatible reload; both links are pending integration in that snapshot. The stopped version cannot supply actual exposure, because it keeps no per-update fresh-exposure record ([§](#c-training--proposed-sibling-fresh-and-replay-branches)), so that link remains proposed. Improved fitted search and a further generation also need the consumption proof above and a preregistered decision. Under the owner's proposed rule, the loop stops on no eligible discoveries, incompatibility, no material gain, insufficient evidence or an exhausted budget; the best checkpoint is preserved, and serving is never promoted automatically.

Maintain explicit checks for changed dependency bytes, an unsupported GPU/device, absent families, zero updates, wrong task heads, changed vocabulary bins, missing continuation, missing native component receipts, stale code hashes, counterfeit fitted outputs, incomplete process-group cleanup and partially written journals. The experimental route is intentionally separate from accepted serving. Any future promotion mechanism must add its own independent evaluation and pointer discipline rather than rename an experimental export.

The current source-only regression follow-on has an independent review with no confirmed production/API blocker. Seven pure Python checks passed; the three actual-model regression/binary/multiclass cases were explicitly deselected because custody work was barred from model execution. The next required CI/runtime acceptance must execute those cases and verify their collected inventory. This chapter records the gap rather than claiming it closed.


---

<a id="c-learning"></a>

# Chapter 8 — Measuring repeated learning and deciding improvement

<a id="c-learning--observed-transferable-lift-reporting-contract-not-yet-measured"></a>

## 8.1 Observed transferable lift: reporting contract, not yet measured

**Current global KPI: not yet measured.** Merged PR90 adds private zero-fit planning and
retained-prediction reporting. This publication contains no new training campaign or
untouched nested evaluation. Existing acceptance thresholds, LOOP-01 definitions and serving
decisions remain unchanged. The older proposal sections below retain their dated scope.

<a id="c-learning--what-the-two-measurements-mean"></a>

### 8.1.1 What the two measurements mean

For each dataset, outer fold and seed, compare the challenger with its parent. The audit
component averages five gains, each measured by an independently repeated complete search
using only that audit fold’s training complement, with inner selection confined there, before
scoring its held-out audit rows. The outer component measures a separate complete search
selected and fitted using the full outer training partition. Each gain is
`(parent RMSE − challenger RMSE) / training-mean baseline RMSE`; the baseline is fitted only
on that score’s permitted training rows. Average the five separately normalized audit gains,
not their raw RMSEs divided by one pooled denominator.

Take `q = min(audit gain, outer gain)` **before aggregation**. Average folds/seeds within each
dataset, datasets within each family, then families equally. Display `100 × result`, in
percentage points of baseline RMSE. A holdout-only gain receives no positive contribution;
audit-only improvement also receives none. For units `(audit, outer) = (0.3, −0.1)` and
`(−0.1, 0.3)`, the contribution remains −0.1, rather than becoming +0.1 by taking the minimum
after pooling. These numbers illustrate the definition and are not experimental results.

The headline is accompanied by both gains, parent/challenger absolute skill against the
trivial predictor, worst-family degradation, three-way sign agreement, planned/valid
coverage, family distributions and an audit-versus-outer scatter plot. If relative improvement
exists without positive mean absolute skill on both components, it says **no absolute skill
demonstrated**. Generic-search and fresh-versus-replay controls remain separate contrasts.

<a id="c-learning--why-this-needs-a-new-frozen-evaluation"></a>

### 8.1.2 Why this needs a new frozen evaluation

Five outer folds and five training-side audit folds require **30 complete searches per
arm/dataset/seed**, each with its own inner selection and final refit. Different folds may
select different recipes. Re-scoring a previously selected winner, or selecting from an
adaptively generated pool after the fact, does not satisfy this contract.

The planner freezes checkpoints, procedure, controls, families, ordered rows/groups, folds,
seeds, assets, source and resource estimates. Insufficient groups/resources remain explicit
failures. Every planned prediction or terminal failure is committed before any audit or outer evaluation score
is released. Inner candidate-selection scores remain available within their permitted search. Missing comparisons, contaminated exposure, invalid identities and zero/nonfinite
baselines block a complete global score; they never disappear from the denominator.

The existing thirty-dataset panel is adaptive development. It cannot become untouched by
changing its label in a manifest. Checkpoint and encoder exposure are tracked separately;
unknown historical exposure prevents an unseen-family claim. A trusted external release
receipt must attest the admitted executor’s isolation and chronology. Hash verification
checks bytes and declared dependencies; this reporting implementation does not establish
the truth of an executor’s isolation claims or admit that future executor.

<a id="c-learning--what-the-score-does-and-does-not-establish"></a>

### 8.1.3 What the score does and does not establish

This is a conservative descriptive score, not a confidence bound or proof of causality.
Folds and seeds share data and are not independent statistical samples. Released outcomes
become exposed development evidence when used to design subsequent work. The selection-bias
problem is described by [Cawley and Talbot](https://www.jmlr.org/papers/v11/cawley10a.html),
fold-dependence by [Bengio and Grandvalet](https://www.jmlr.org/papers/v5/grandvalet04a.html),
and adaptive reuse by [Dwork et al.](https://arxiv.org/abs/1506.02629). These references motivate
the precautions; they do not prove a special statistical guarantee for this minimum score.

The CPU tests exercise arithmetic, evidence validation, coverage and a deterministic toy
selection boundary. Changing its own scoring labels leaves the fixture’s frozen recipe and
predictions unchanged. That test is not production executor qualification. See the private
contract `docs/internal/transfer-measurement.md` and merged source `7f2b2bf860e4b1ac8815c8da55e507143570a905`.

<a id="c-learning--reading-the-real-historical-plots-alongside-this-kpi"></a>

### 8.1.4 Reading the real historical plots alongside this KPI

The figures in [§](#c-training-history) retain the original observations and their hashes:
63,000 A100 steps (6001–69000, not invented steps 1–6000), the separate 100-step RTX
continuation, fixed-monitor comparisons and available conditional ranking/regret diagnostics.
Training corpus, objective, lineage and validation-population boundaries stay visible.
The fixed monitor is adaptive teaching evidence; decreasing training loss or ranking regret
cannot substitute for the unmeasured nested transfer KPI. Raw histories remain intact; plotted
bins are declared within-corpus summaries, with no interpolation across missing history.


<a id="c-learning--preserved-historical-chapter-snapshot"></a>

## 8.2 Preserved historical chapter snapshot

The following sections retain the original dated source narrative and evidence. The current-source additions above supersede implementation-status descriptions where explicitly stated.

<a id="c-learning--what-better-and-better-must-mean"></a>

## 8.3 What “better and better” must mean

A genuine flywheel joins four measured improvements. GP finds a better executable teacher for an unchanged comparison. Training actually consumes that new information. The policy becomes better at proposing or ranking programs for the relevant dataset-and-split context. Finally, the next bounded GP search improves because of that policy. Each arrow needs its own evidence; a growing corpus, a saved checkpoint or a completed search cannot stand in for the whole chain.

Four claims therefore stay distinct, and each needs its own evidence: an **eligible teacher** (a target admitted under its full comparison identity), a **new discovery** (a program absent from the authenticated initial population), **actually consumed information** (the optimizer sampled it in committed updates) and an **improved policy** (the trained checkpoint beats its parent in a matched downstream fitted comparison under the same budgets). None implies the next. Initial populations must be authenticated, and a discovery made in one arm must not silently seed another arm.

The comparison identity is essential. A target belongs to a task, dataset content, exact split and groups, evaluator, vocabulary and numerical protocol. If any changes, the score is a new experiment rather than a better version of the old target. Context identity also binds the fitted preprocessing, encoder assets and allowed fitting rows. Two vectors with the same width need not represent the same problem.

An incumbent can improve monotonically under a fixed development objective while real generalization gets worse through repeated selection. We therefore need both a target-quality ratchet on development data and independent evidence that the learned search transfers to other families. The latter is the north star (proposed; owner ratification pending; defined in [§](#c-status--the-proposed-north-star-and-matched-comparisons)). Repeatedly attacking one dataset is useful specialization; it is not evidence of broad transfer.

<a id="c-learning--what-the-historical-training-and-target-evidence-establishes"></a>

## 8.4 What the historical training and target evidence establishes

ASM means **Algorithm Synthesis Model: learns to generate and optimize executable algorithms from data**. The repository has real historical neural training and real retained GP teaching corpora. It also has substantial uncompleted work before a newly bounded ASM learning campaign can establish useful improvement. These facts must be stated together. “Zero freshly admitted teachers under the new ASM admission guard” describes the new campaign's inspected readiness; it does **not** mean that JopeDime has never trained on real data, that the accepted policy is untrained, or that its historical ten-million-record corpora are synthetic.

This chapter reconstructs progress from original retained metrics and checkpoint metadata without loading model tensors, reopening sealed targets, or starting training. It separates three kinds of evidence: loss measurements, changed training state, and improved executable algorithms. Losses measure how the model fits its training objectives. Checkpoint lineage records which parameters and optimizer state were produced. Only a matched downstream fit/search comparison can establish that conditioning on a dataset and split produces better algorithms. A completed optimizer run does not imply the third result.

The retained evidence shows both substantial historical training corpora and some improving historical GP targets. It does **not** show a newly qualified ASM teaching loop, a current thirty-dataset result, or a causal advantage from neural training. Those are different claims with different evidence requirements.

<a id="c-learning--loop-01-the-central-regression-milestone"></a>

## 8.5 LOOP-01: the central regression milestone

LOOP-01 is the central regression product milestone. One LOOP-01 generation is the complete chain:

**new eligible target → actual exposure → changed parameters → compatible reload → improved fitted search → another new generation**

Each arrow is a separate claim with its own evidence. A new eligible target passes the admission guard under its full comparison identity and is absent from the authenticated initial population. Actual exposure is measured in committed optimizer updates, not inferred from corpus membership. Changed parameters are shown by parameter digests before and after training. A compatible reload loads the exported checkpoint in the consumer with its architecture, vocabulary, encoder and source identities intact. An improved fitted search is judged by the preregistered rule in [§](#c-learning--the-proposed-preregistered-regression-decision). Another new generation means that search yields new eligible targets for the next round. Two passes through one corpus are continuation, not two generations.

Status of each link as of 2026-09-23 at about 23:55 UTC:

| Link | Mechanism | Status | Evidence now |
|---|---|---|---|
| New eligible target | Five-fold admission guard and strict HWM identity in the pending eligibility source ([§](#c-teachers--the-admission-guard-label-exposure-and-hwm-identity)) | pending integration | Zero teachers admitted under the new contract |
| Actual exposure | Per-update fresh-exposure record and initial-population/discovery identity | proposed | Not measured; the stopped training campaign records neither |
| Changed parameters | Campaign verification of positive updates and changed parameter digests ([§](#c-training--plan-run-resume-verify-and-export)) | pending integration | Not measured for a new lineage. The retained step 60,100 run did genuine optimizer work on the unchanged corpus with zero fresh program presentations: continuation, not a generation |
| Compatible reload | Worker reload of `policy.pt` and `resume.pt` with equal parameter proof, then the fail-closed `experimental.regression` consumer ([§](#c-training--compatibility-checks-and-fail-closed-rules)) | pending integration | Not measured |
| Improved fitted search | Consumption of the export in a new operational proof, judged by the proposed decision rule | proposed | Not measured |
| Another new generation | A further teaching campaign over searches conditioned on the reloaded checkpoint | proposed | Not measured |

The current one-dataset origin proof, which closes context → proposal → fit → evolution for the accepted model, is available evidence; it is not a LOOP-01 link. No LOOP-01 generation has run.

<a id="c-learning--lineages-continuation-names-and-generations"></a>

### 8.5.1 Lineages, continuation names and generations

The accepted **512×6 step-60,000** model of lineage B (width 512, six layers) remains the serving model and a report baseline. The pending campaign would start a separate line, **experimental regression lineage v1 (256×2, new initialization)**, planned and not yet trained: the width-256, two-layer Python model described in [§](#c-training--what-the-pending-campaign-trains). Its newly initialized model, `M_init`, is called `M0` only after it has actually trained on the first admitted corpus, `C0`. Step 60,100 is historical hardware continuation of step 60,000, and step 69,000 remains unaccepted ([§](#c-architecture--checkpoint-binding-and-configuration)); neither is a LOOP-01 generation. The two lines are compared only through matched downstream fits, never through loss values.

Every training attempt must declare one of four continuation names (proposed naming): **cold initialization**; **weight warm start**; **explicit new-corpus transition**, which declares whether optimizer state is carried and how the sampler is initialized; and **exact interrupted resume**, which keeps the original state, inputs and deadline. Exact resume must reject an optimizer fallback before any update and prove that interrupted and uninterrupted runs reach the same state. Both requirements are proposed: the stopped campaign checks optimizer resumption only after training and has no interrupted-versus-uninterrupted parity test. A new generation needs a new corpus containing new eligible targets; a second pass or an exact resume on the same corpus continues the current generation.

<a id="c-learning--two-controlled-rounds-within-the-existing-gpu-cap"></a>

### 8.5.2 Two controlled rounds within the existing GPU cap

LOOP-01 calls for two linked controlled rounds. Each round (proposed) trains two sibling branches from one immutable parent model and optimizer state, under the round's original deadline:

- **F (fresh):** samples 25% fresh and 75% replay;
- **R (replay-only):** samples replay only.

Both branches commit the same number of updates and batch slots. For each branch, record the measured fresh exposure, duplicate and unique families, non-padding tokens and GPU cost; keep the unchanged parent as a comparison artifact. Neither branch is promoted to serving. The stopped training campaign has no sibling branches, immutable sibling parent or 25/75 sampler allocation.

The existing allocation gives the regression category at most 20 minutes of GPU time inside the one-GPU-hour campaign ([§](#c-scheduling--resource-policy-and-the-training-clock)), and LOOP-01 grants no further allocation. The two controlled rounds may therefore remain unfunded, or inconclusive within that cap. Either outcome is reported as mechanical proof and descriptive results, with the causal question marked inconclusive.

<a id="c-learning--the-closure-ledger"></a>

## 8.6 The closure ledger

| Link in the loop | Existing evidence | Remaining proof | Roadmap connection |
| --- | --- | --- | --- |
| Dataset and split → valid context | Current regression producer checks original rows, preprocessing, assets and eight reference cases | Classifier context/checkpoint consumption and newly trained context compatibility | NEURAL / TRAIN items |
| Context → real initial GP candidates | Current 80-slot path (64-slot policy pool plus 16 exploration, with explicit rejection fallback) has authenticated genuine `neural_policy` fits | Show the newly trained checkpoint produced the measured candidates | NEURAL / TRAIN items |
| GP evaluation → improved target | Current bounded search fits and evolves candidates; historical champion and corpus records exist | New admission needs complete finite OOF evidence, baseline improvement on at least 4/5 folds and the full identity | TEACH items |
| Improved target → frozen teaching corpus | Historical binary teaching stores and corpora survive; pending exporter authenticates inputs | Execute the new eligible-state export with complete admitted-family coverage; record fresh versus repeated examples | TEACH / TRAIN items |
| Corpus → parameter update | Historical optimizer logs and accepted step-60000 exist | New bounded run must prove positive updates, changed parameters, reload and true optimizer continuation | TRAIN items |
| Parameter update → better policy | Historical imitation/ranking diagnostics exist | Compare parent and challenger on unchanged diagnostic inputs; then measure fitted-program benefit under matched budgets | NEURAL / TRAIN items |
| Better policy → better next search | Current accepted model reaches actual evolution on one dataset | Run the exported challenger in a subsequent search and retain a causally interpretable comparison | NEURAL / PANEL items |
| Improved search → next useful training generation | Historical controller and target chains exist | Complete at least two linked new cycles (flywheel turns, that is LOOP-01 generations as defined in [§](#c-learning--loop-01-the-central-regression-milestone), not training blocks or passes) with eligible new targets, actual exposure and measured downstream outcomes | End-to-end flywheel qualification |
| Qualified checkpoint → serving | Product is bound to recovered accepted step-60000 | Design and qualify the missing experimental-to-serving promotion contract, then require the new checkpoint to pass it; training-pointer movement alone does not move the serving pointer | FUTURE-01: deferred; no executable acceptance command yet |

The prefixes above refer to the detailed roadmap's existing workstreams; they are not new executable command names. The current one-dataset proof closes context→proposal→fit→evolution. It does not close the new teaching→training→improved-search cycle. The historical training chapter separately reports what earlier controllers actually achieved, including unavailable or inconclusive promotion evidence.

<a id="c-learning--a-minimal-experiment-that-can-establish-learning"></a>

## 8.7 A minimal experiment that can establish learning

<a id="c-learning--freezing-the-experiment-and-choosing-families"></a>

### 8.7.1 Freezing the experiment and choosing families

Freeze the experiment before opening the run window: source and binaries, exact parent checkpoint, corpus membership, encoder and vocabulary, objective weights, optimizer carry policy, sampler, random streams, fit/search budgets and comparison metrics. Keep the accepted parent available as a control. A new objective, new encoding or incompatible optimizer state starts a new lineage; it must not inherit a misleading uninterrupted-step counter.

Use eligible teachers from independently identified training families. Keep family-disjoint development diagnostics for the model. When a dataset was used to manufacture teachers, label subsequent evaluations on that dataset as exposed/adaptive. The pending path's two prospective teaching families, `230_machine_cpu` and `712_chscase_geyser1`, are both frozen-panel regression datasets ([§](#c-teachers--present-teacher-readiness-and-the-smallest-real-path)); if they supply teachers, later panel reports must label those two datasets exposed/adaptive. Unknown historical checkpoint exposure remains unknown. A split name alone cannot prove an unseen family. Assign families, at family level, to three domains: teacher/train, adaptive development, and evaluation outside training and selection. The frozen thirty-dataset panel remains adaptive development, and the pretrained encoders' exposure to any family remains unknown. A hash authenticates bytes; it does not make an input eligible. The input closure covers records, contexts, transforms and all assets, and unused forbidden descriptors must be rejected before loading. These are proposed requirements: the stopped training packet validates only the contexts that its retained rows reference, does not reject extra context-index entries, and partitions families only into train and development domains.

<a id="c-learning--what-to-retain-for-each-cycle"></a>

### 8.7.2 What to retain for each cycle

For each cycle, retain all of the following:

1. **Target creation:** incumbent and challenger recipe hashes, all five candidate and baseline OOF losses, complete fit outcomes and exact comparison identity. Authenticate the complete initial seed population. Distinguish a newly discovered program absent from that seed set from a carried incumbent, a neural seed and its evolved descendants. Retain discovering arm, program hash, generation, context, parent/ancestry, first measurement and later shared-seed exposure. Count each distinct dataset/cut/program discovery once before reporting per-arm reuse. Record rejected, tied, failed and unavailable candidates too.
2. **Corpus delta:** previous/new corpus hashes; added, improved, removed and unchanged comparison/program identities; distinct contexts and families; rejection reasons. Distinguish logical rows from unique programs and unique contexts.
3. **Actual optimizer exposure:** how often newly admitted programs and contexts appeared in committed steps, their weights, sample repetition, padding fraction and fresh/broad sampling fractions. New rows in a manifest are insufficient if the sampler never reaches them.
4. **Training continuity:** initialization parent, parameter hashes before/after, optimizer initialization versus restored moments, local and cumulative committed steps, effective batch and actual clock/resource use. A failed attempt's compute stays in accounting.
5. **Policy diagnostics:** component losses on a fixed, identified monitor; valid-program rate, uniqueness, conditional ranking and proposal quality. Report corpus/monitor changes as boundaries rather than drawing a continuous evaluation curve through them.
6. **Next-search outcome:** execute both parent and challenger proposals as actual fits under the same candidate/compute allowance, plus the generic control. Retain neural-origin and ancestry counts, failure rates, selected program identities and the downstream losses.
7. **Decision:** accept, reject or inconclusive with the exact gate and original evidence; for regression rounds, apply the proposed rule in [§](#c-learning--the-proposed-preregistered-regression-decision). Preserve the old serving pointer unless promotion succeeds.

Two linked successful cycles are a practical demonstration of repeatability. They do not prove unlimited improvement. Plateaus, search-space ceilings, noisy estimates, sample reuse and model capacity remain real constraints. Report the improvement per cycle and cost per eligible improvement; do not promise a monotonic performance curve on unseen data.

<a id="c-learning--the-proposed-preregistered-regression-decision"></a>

### 8.7.3 The proposed preregistered regression decision

This is the owner's proposed decision rule for LOOP-01 regression rounds. It is a proposal, not an achievement: no round has been judged by it. Preregister it, together with the monitor families, seeds, folds and trivial predictor, before any challenger is compared, and do not change it after seeing results.

<a id="c-learning--family-normalized-gain"></a>

#### 8.7.3.1 Family-normalized gain

For each monitor family, the gain is `g = (L_control − L_challenger) / L_trivial`.

`L_control` is the comparison model's loss: the parent for the main test, and the replay-only branch for the fresh-versus-replay test. `L_challenger` is the challenger's loss, and `L_trivial` is the loss of the preregistered trivial predictor on the same family. Positive `g` means the challenger is better. Average over the declared seeds and folds within each family, so that each family contributes one value. The preregistration states explicitly how a family with zero or nonfinite `L_trivial` is treated; no hidden epsilon is added and no such family is dropped silently.

<a id="c-learning--round-thresholds"></a>

#### 8.7.3.2 Round thresholds

A round shows a gain only if all of the following hold:

- at least 10 independent monitor families, each with complete comparisons;
- at least 9/10 of the families gain more than 1% against the parent;
- the mean gain across families exceeds 1%;
- no family is worse than −5%;
- the fresh branch beats the replay-only branch in at least 9/10 of the families, with a positive mean gain.

Round 2 uses monitor families separate from those of round 1.

<a id="c-learning--baselines-stopping-and-reporting"></a>

#### 8.7.3.3 Baselines, stopping and reporting

Generic search and the accepted step-60,000 model remain report baselines. A gain over the parent is a different claim from an advantage over generic search or over the product's accepted model. Stop the loop when there are no eligible discoveries, on an incompatibility, when there is no material gain, when the evidence is insufficient, or when the budget is exhausted. Preserve the best checkpoint and never promote to serving automatically. An unfunded round, or one with insufficient data, yields mechanical proof and descriptive results; its causal conclusion is "inconclusive", never an invented success.

<a id="c-learning--distinguish-routing-ranking-and-search-recovery"></a>

## 8.8 Distinguish routing, ranking and search recovery

The learned route is a conditional distribution over executable program tokens given the dataset-and-target context. It is richer than selecting a fixed named estimator. The same model also supplies learned ranking signals. Its reward representation mean-pools independent token embeddings, a bag of actions, with no positional term: at `a75e543d2` both the Python model (`python/src/jope/policy/model_v3.py:399-404`) and the product's Rust port (`crates/jope-policy/src/model.rs:555-567`) compute a padding-masked mean. It has no learned representation of token order: for the same context, programs with the same token multiset have the same pooled representation in exact arithmetic. This limits ranking distinctions between programs whose execution differs only through token order; it does not remove order from the autoregressive proposal decoder. See [§](#c-architecture--decoder-and-reward-architecture); [§](#c-learning--conditioning-diagnostics-before-the-first-causal-experiment) proposes measuring its collisions before any architecture change.

Measure the policy before GP recovery as well as after it. A long search can find the same winner despite poor neural proposals, while a short search may reveal useful initialization that disappears at a larger budget. Shared incumbents can create many ties. Equal time caps do not imply equal completed fit counts, especially when candidate runtimes differ.

The admitted historical evidence in [§](#c-targets--did-gp-targets-get-better) illustrates this limit: at 1,000 completed evaluations, 3,268 paired neural/generic final-search observations contain 24 lower neural scores, 17 higher scores and 3,227 exact ties. Their geometric mean neural/generic RMSE is 1.0001869094. They repeat families and generations, consume unequal reported wall time and measure adaptive final winners. The historical arms also share incumbent, retained-teacher and exploration seed pools; a winner can seed every arm in the next generation. In the separate first-to-last table, 64 lower arm groups collapse to 23 distinct dataset/cut pairs out of 55, with identical available-arm final scores in 22 of those 23. Those are dependent carried-forward outcomes, not 64 independent discoveries or wins. They neither establish initial routing quality nor show that later training caused better search. A larger external extract is kept outside this claim where its rows have not joined the admitted inventory.

Useful controls answer different questions: generic grammar sampling tests end-to-end benefit over the non-neural search; the previous accepted checkpoint tests whether training helped; shuffled contexts test conditional information; and random ranking on the same candidate pool tests ranking utility. These additional controls are proposed experiments, not extra runs hidden inside the approved panel. Declare their budget and scope before execution.

<a id="c-learning--conditioning-diagnostics-before-the-first-causal-experiment"></a>

## 8.9 Conditioning diagnostics before the first causal experiment

These diagnostics ask whether the policy actually uses its dataset-and-split context. All are proposed and none has been measured. This manual proposes running them before the first causal LOOP-01 experiment, and keep every diagnostic substitution outside teacher admission: a program proposed or ranked under a substituted context never becomes a teacher.

| Diagnostic | Question | Comparison | Status |
|---|---|---|---|
| Boundary correctness | Is the context built only from permitted fitting information, within an input closure that covers the exact records, contexts, transforms and assets? | Unused forbidden descriptors are rejected before loading | proposed |
| Functional sensitivity | Do proposals and ranking scores change when the context changes? | The true context against substituted contexts | proposed |
| Useful ranking | Does the learned ranking order candidates better than simpler signals? | Donor-context, program-only and random baselines on one fixed, already-fitted candidate bank | proposed |

The current regression producer already checks original rows, preprocessing, assets and eight reference cases ([§](#c-learning--the-closure-ledger)). The stopped training campaign, however, does not reject extra context-index entries or unlisted transitive artifacts, so boundary correctness is not yet established for training inputs.

The reward path's order blindness is measured before it is changed. Both implementations mean-pool token embeddings without a positional term ([§](#c-learning--distinguish-routing-ranking-and-search-recovery)). First measure how often distinct programs in one context collide on the same pooled representation, and the ranking regret those collisions cause. Defer architecture changes, so that the first causal experiment keeps architecture, objective, corpus and sampler changes distinguishable.

Dense-width feasibility checks must use only permitted fitting information, keep the original proposal limits and clocks, and report a preflight rejection separately from a launched fit failure; they grant no unlimited free retries. A naive preflight must not rely on the retryable `Unavailable` outcome that a job-preparation error receives at `a75e543d2` (`crates/jope-pipeline/src/neural_search.rs:1608-1623`); it needs a terminal outcome with its own accounting. The source investigation is custody `plan-execution-20260923/feasibility-phase-accounting.md`, SHA256 `0154e6b0074d6b3c7100abf28568abb41a998b6ed21730237d58649f31cc9f18`. This check is proposed source work.

<a id="c-learning--specialization-without-starving-the-general-learner"></a>

## 8.10 Specialization without starving the general learner

One difficult dataset can produce many near-duplicate teachers. Count unique canonical programs, meaningful improvements and distinct contexts, not only serialized rows. Retain broader replay and family balance so a successful focused attack does not make the policy forget other problems. [§](#c-data-estate--packing-semantics-legal-sets-and-label-policies) describes how to share repeated contexts and immutable shards without losing original provenance.

Record the fresh-program exposure actually sampled. A nominal 50% fresh-comparison schedule need not yield 50% fresh program slots when each comparison also contains an incumbent and old comparators. Prefer measured exposure, a frozen baseline checkpoint, component-loss diagnostics and actual downstream fitting over assumptions based on the sampler's name.

If the target ratchet moves but fixed-monitor policy quality deteriorates, investigate the objective, label consistency, sampler and comparator support before simply training longer. If policy metrics improve but fitted searches do not, inspect proposal diversity, candidate runtimes, grammar coverage and GP recovery. If neither improves, report the plateau and preserve the best verified incumbent. The system earns a flywheel claim through this evidence, not through an endlessly running process.

<a id="c-learning--a-kpi-ledger-with-defined-denominators"></a>

## 8.11 A KPI ledger with defined denominators

<a id="c-learning--kpi-definitions"></a>

### 8.11.1 KPI definitions

These are proposed reporting definitions for acceptance, not a claim that every current report already emits every field. Preserve the fixed protocol where it already specifies a quantity. Additions must be preregistered before comparing challengers; do not change a definition after seeing results.

| KPI | Definition and denominator | Evidence and present gap |
|---|---|---|
| Task loss delta | For a valid matched pair, `loss_reference − loss_challenger`; positive means better. Regression loss is RMSE; binary loss is `1−AUROC`; multiclass is ordered probability-matrix log loss. Report both generic and previous-checkpoint references separately. | Exact predictions, rows/classes, evaluator, numerical policy and resource receipts. Never average raw RMSE with log loss across tasks. |
| Strict wins / planned comparisons | Count valid matched improvements over all preregistered comparisons, including failed/unavailable comparisons in the denominator. Report ties, losses, one-sided failures and two-sided failures separately. | Complete planned inventory. Failure is not a fabricated numerical loss; paired mean delta is separately labeled as complete-pair-only with coverage. |
| PANEL-WINS (proposed definition; owner ratification pending) | Per task, the number of the 10 frozen-panel datasets (asm-private-30-v1, protocol asm-panel-v1) on which `asm.neural`'s confirmed five-fold mean development loss is strictly below the minimum over all eligible frozen non-ASM comparators (every `rust.*` and `external.*` recipe). `asm.generic` is an ASM control reported in its own column. An incomplete, failed or unconfirmed dataset is not won. Reported as (regression, binary, multiclass); the owner's goal of ten per task, (10, 10, 10), is a 30/30 sweep. | Not computed: no win-counting function exists in `crates/jope-bench/src` at `a75e543d2`, and the panel has not run. The statistic is adaptive and favours `asm.*`: the ASM recipe is selected on cut-0 fitting rows that the five-fold mean also scores, while `rust.*` and `external.*` recipes are fixed before the dataset screen (`crates/jope-bench/src/panel/selection_exposure.rs:91`; [§](#c-benchmark--adaptive-development-exposure)). At source `68f56404` binary and multiclass have no neural arm ([§](#c-architecture--experimental-classification-context)). |
| Family generalization | Per-task mean of family means under the preregistered split, with cluster uncertainty at family level when sample size permits | Frozen family provenance and prior-exposure evidence. No current frozen-panel (asm-private-30-v1) unseen-family claim; historical unknown exposure stays unknown. |
| Relative improvement | `(Lref−Lnew)/Lref` only for finite, strictly positive reference loss; report absolute delta when zero | No hidden epsilon convention or cross-task aggregate introduced after results. |
| Family-normalized gain (LOOP-01; proposed) | Per monitor family, `(L_control − L_challenger)/L_trivial`, averaged over the declared seeds and folds within that family; the denominator is the preregistered monitor families, each with complete comparisons | Not measured. A zero or nonfinite `L_trivial` needs an explicitly preregistered disposition ([§](#c-learning--the-proposed-preregistered-regression-decision)). |
| Teaching admission | Eligible comparisons / all requested comparisons, and eligible independent families / all requested families | Preregistered five-fold train-side OOF evidence, finite baseline improvement and improvement on four of five folds. Newly demonstrated ASM-qualified count currently zero; historical corpus counts are separate. |
| Genuine neural consumption | Completed `neural_policy` generation-zero candidates / actual `neural_policy` slots; also report completed genuine neural candidates /64 policy-pool slots, fallback slots and completions, and exploration completions /16. Completed descendants require verified neural ancestry. | Authenticated proposal→request→started→finished→artifact chains. A recipe file, neural tag, generic fit or started-only attempt is insufficient. |
| Validity and fit coverage | Canonical task-compatible proposals / all proposed draws; completed fits / all scheduled eligible fits | Distinguish grammar refusal, dependency unavailability, fit error, timeout, cancellation, never-started and unknown. State whether duplicate candidates are deduplicated. |
| Optimizer progress | Updates this run, cumulative optimizer step, start/end parameter digests, committed passes, and exact resumed-state identity | Batch/epoch metrics and checkpoint proofs. Positive epoch count or a checkpoint filename is insufficient; training loss denominator and objective must match before plotting together. |
| Latency | Cold build/install/preparation separately; warm end-to-end wall time and declared target `<60s`; actual per-arm planned/available/executed time | Original monotonic/absolute windows and launch receipts. A completed >60s demo can be execution-complete with target missed. |
| Resource and cost efficiency | Total known creation CPU/GPU/wall and monetary/token costs divided by authenticated distinct complete fitted artifacts; also cost per distinct reusable recipe | Distinct means declared hash identity, not proven functional uniqueness. Whole-unit counters overlap worker sums; do not add them. Unknown overhead, price or tokens remain null with reason. |
| Fitted portability | Qualified exact selected fitted predictor/backend pairs / all requested selected predictor/backend pairs | Actual emitted code execution and prediction parity, in order Python/Rust then C++17/C++20 then JS. Unsupported pairs stay in the inventory and never replace the winner. Current five-language toy parity is recipe refitting only. |
| Reproducibility | Successfully authenticated independent report/reload replays / all requested replays | Same source, model, input, plan, protocol and artifact identities. Stale outputs, missing rows or changed class order are failures. |
| Promotion | Accepted serving-pointer advance with independent matched evaluation, exact checkpoint identity and controlled pointer update | Separate from training-pointer advance and experimental export. No new promotion is claimed. |

<a id="c-learning--win-counting-and-a-proposed-acceptance-record"></a>

### 8.11.2 Win counting and a proposed acceptance record

Define a dataset-level comparison using the protocol's frozen fold summary; retain fold-level results separately. Keep separate win columns for generic search, the previous checkpoint, and the best eligible frozen non-ASM comparator; the last column is PANEL-WINS as proposed in [§](#c-learning--kpi-definitions). A win against one reference is not automatically a win against the others. For the owner's goal of ten wins per task on the 30-dataset panel, each reference's denominator remains ten per task and all thirty outcomes are listed, including unavailable ones; meeting it in every task is a 30/30 sweep. It is a declared goal, not a promise. As of 2026-09-23 there is no completed full-panel result establishing those counts.

A minimal additional acceptance record could use the following explicit nulls. This is a **proposed reporting shape**, not an existing on-disk schema:

```json
{
  "definition_status": "proposed_operator_reporting",
  "source_commit": null,
  "checkpoint_state_sha256": null,
  "reference_checkpoint_state_sha256": null,
  "corpus_sha256": null,
  "encoder_identity_sha256": null,
  "vocabulary_sha256": null,
  "panel_manifest_sha256": null,
  "protocol": null,
  "original_deadline_unix_ms": null,
  "planned_comparisons": 30,
  "valid_matched_pairs": null,
  "strict_wins": null,
  "eligible_teacher_families": null,
  "optimizer_updates_this_run": null,
  "exact_fitted_backend_pairs_qualified": null,
  "creation_cost": {"usd": null, "tokens": null, "unknown_reason": "not measured"},
  "serving_promoted": false
}
```

Retain the receipt path and content hash for each populated quantity, plus its unit, scope, numerator, denominator, exclusions and measurement time. Compare the supplied one-hour/ten-million-token scenario with actual token and cost records if those become available; a token budget is not evidence of tokens consumed, nor does it determine GPU time or dollars. Historical totals, current implementation counts and newly executed acceptance must remain separate series.

<a id="c-learning--what-is-still-needed-to-demonstrate-the-learning-flywheel"></a>

## 8.12 What is still needed to demonstrate the learning flywheel

The desired loop is concrete: a dataset and split produce verified encodings; conditioned policy proposals become real generation-zero fits; bounded GP/evolution produces eligible improved programs; the full comparison identity admits those programs as teaching targets; optimizer updates produce a new checkpoint; matched downstream fits show whether that new checkpoint proposes and searches better algorithms. Only then can a promotion change the serving pointer. [§](#c-teachers--a-dataset-is-not-one-training-example) and [§](#c-data-estate--binary-format-the-exact-packed-teaching-record) trace the historical input-to-target representation, and [§](#c-targets--did-gp-targets-get-better) separately reports observed adaptive target improvements in repeated searches with matching recorded comparison fields, including improvements in generic and shuffled arms. Those observations are largely carried-forward development scores from seed pools the arms share; they do not establish neural causality. [§](#c-learning--what-better-and-better-must-mean) defines the remaining current acceptance work.

For every cycle, preserve both sides of the feedback: target improvement under the same task/dataset/split/evaluator/vocabulary/protocol identity, and policy improvement in fitted performance, failure rate and cost under a frozen comparison. Report cold/warm preparation and unique fitted predictors separately from recipe counts. Keep a context-free control, generic proposal control and unchanged accepted policy. A lower teacher-forced NLL can accompany memorization, degraded ranking, or no better executable search, as the historical measurements demonstrate.

The missing historical evidence is specific: full intermediate trajectories for the older 108-block capture (distinct from the complete later 63,000-step series); an exact run-source binding and event timestamp for gen31; per-term gradient diagnostics; a common independent evaluation series across changed corpora; and a fully joined sequence of qualified target improvements and serving promotions. For the new bounded ASM campaign, corpus admission, actual optimizer update/reload/resume proof, trained-export consumption and matched downstream improvement remain acceptance work. Keep these gaps visible rather than replacing them with a smooth synthetic curve or treating old training as if it had never happened. This chapter contains no performance plot. Every LOOP-01 quantity, including per-round family-normalized gain, the fresh-versus-replay difference and cycle-over-cycle improvement, is not measured. A future figure of any of them is drawn only from receipts, and every unmeasured series says "not measured" rather than showing an illustrative curve.

To establish a closed improvement loop rather than these separate observations, join: original search/fit completion → exact program and comparison identity → admission event and immutable target record → corpus membership → actual sampler exposure and optimizer continuation → later policy proposal/fit under the same comparison protocol. The historical logs in [§](#c-training-history--dated-lineage-and-evidence-coverage) show training activity; the target receipts here show some improving, largely carried-forward development scores. That complete causal/eligibility join is still missing.


---

# Chapter 9 — Complete fitted inference and language support

At source `68f56404c3e199e06413bfd645d3fe0803d7b96f`, merged unchanged as main `a75e543d2` (same source tree), the legacy `jope emit` and `jope pipeline run --emit` paths produce type 3. `jope fit` computes a fit and reports scores but does not provide a saved fitted-artifact output flag. The pending product/winner packets, which are not part of merged main, expose type 4 from complete benchmark/demo fitted state. Do not advertise a legacy recipe file as the exact fitted winner.

This chapter and the supplied tiny fixture are **maintainer-private**. The DSL, private recipe manifest, discovery metadata and qualification references do not belong in a public inference bundle. The public allowlist described below is deliberately narrower than the manual's working directory.

<a id="c-inference--deployment-support-definition-priority-and-status"></a>

## 9.1 Deployment support: definition, priority and status

The owner's rule for this chapter, relayed on 2026-09-23 at 22:27:11 UTC, sets what may be called deployment support. It is a requirement for export work, not a description of the current source:

- **A complete fitted pipeline.** A deployable artifact carries the whole fitted pipeline of the selected model: its ordered transforms, fitted dictionaries, target mappings, classes and their order, and its numerical policy.
- **The actual winner, without refit.** The backend executes the selected winner from its saved fitted state. Refitting a recipe, even the same recipe on the same rows, is not deployment support.
- **Unsupported stays visible.** When a backend cannot represent the winner, that backend's result is "winner + unsupported". The winner is never replaced by another recipe that happens to export.
- **Priority.** Python and Rust first, then C++17 and C++20, then JavaScript.
- **Inside the original window.** Export work uses the original campaign window and stops before its reporting reserve ([§](#c-inference--backend-numerical-policies-public-metadata-and-the-export-clock)).

By this rule, no route at merged main `a75e543d2` provides deployment support yet. Status as of 2026-09-23, about 23:55 UTC:

| Route | What it executes | Status |
|---|---|---|
| Legacy recipe emitters (`jope emit`, `jope pipeline run --emit`) | Generated source that refits a recipe from a training CSV. None of the 26 registered benchmark recipes can be emitted ([§](#c-inference--coverage-by-recipe-and-backend)). | **available** for recipe refit only, which the fourteen-command private example run qualified on its fixtures ([§](#c-inference--measured-execution-of-the-private-examples)). This is not deployment support. |
| Private fitted C++17 library (`emit_fitted_cpp17`) | Scalar-state C++ generated from a saved `BoundModel`, without refit | **implemented but unqualified** as a deployment route. At `a75e543d2` only the research crate `crates/jope-campaigns` calls it; the `jope` CLI does not. Its admission record, `crates/jope-campaigns/manifests/advanced-v3/fitted-cpp-v3-admission.v1.json`, is marked `test_adaptive: true` and `benchmark_valid: false`. |
| Public fitted bundles for Python/Rust, C++17/C++20 and JavaScript, driven by `benchmark export-winner` and `benchmark qualify-export` | The selected winner's saved fitted state, lowered for the supported subset, without refit | **pending integration**. These exist only as stopped custody patch views, absent from merged main ([§](#c-inference--example-c-exact-fitted-winner-export-after-pending-integration)). No winner has been exported or qualified. |
| CUDA inference | Nothing yet | **proposed**, and deferred until the backends above qualify. |

<a id="c-inference--language-interfaces-at-the-frozen-source"></a>

## 9.2 Language interfaces at the frozen source

| Interface | Input and output | Coverage and operational meaning |
|---|---|---|
| Native Rust `feat_math`/advanced-v3 | Typed recipe or authenticated `BoundModel`; numeric or raw-table prediction boundary | The benchmark uses the raw-table boundary and preserves complete private fitted state. The native roster is task-specific. |
| Legacy Python emitter | Generated single-file Python with `Model` fit/predict and CSV CLI | Numerical recipe execution; host `libm`. Supports the emitter's legacy program set, not every advanced-v3 state. |
| Legacy Rust emitter | Generated standard-library Rust with fit/predict and CSV CLI | Numerical recipe execution; source emitted with Python as a pair. Compiler/runtime and fixture qualification remain required. |
| Legacy C++ emitter | One C++ source, C ABI, numeric input, fit/predict; external CSV driver for examples | Ridge/ElasticNet heads and the emitter's admitted transform subset. C++17 and C++20 use this same source with different standard flags. |
| Legacy JavaScript emitter | ES2020 numeric fit/predict API; external CSV driver | Ridge/ElasticNet heads and its admitted transform subset. `Math` may differ from native platform libm. |
| Existing private fitted C++ library | `supports_fitted_cpp17`, `emit_fitted_cpp17` over `BoundModel` | Broader scalar-state emitter exists at `68f56404` and merged main `a75e543d2`. Its source gate is not the pending public bundle gate or proof of winner parity. Multiclass probability matrices are refused by its scalar ABI. |
| Pending public fitted Python/Rust | `predict_raw` over ordered raw rows with string/null cells | Complete selected-state lowering for the supported subset below; no model refit. |
| Pending public fitted C++17/C++20 | Same generated C++ inference core plus `codec.json` and Python `ctypes` raw-table boundary | Public gate currently restricts this broader emitter to regression ridge and binary logit ridge. Inference therefore needs a Python 3 `ctypes` host beside the compiled library (the pending packet names a Python 3.10 `ctypes` typed-table host); it is not a self-contained C++ raw-table parser. |
| Pending public fitted JavaScript | `predictRaw(rows)` over raw rows; Node/CommonJS qualification harness | Linear and multiclass subset below; a declared ECMAScript numerical policy and tolerance are used. |
| CUDA | No qualified public fitted-export route in this delivery sequence | Deferred until Python/Rust, the one C++ emitter under both standards and JavaScript are qualified. A training GPU does not establish CUDA inference parity. |

Source anchors: `crates/jope-cli/src/kernel.rs`, `crates/jope-pipeline/src/emit.rs`, `crates/feat-math/src/{transpile,emit_py,emit_rs,emit_cpp,emit_js}.rs`, `crates/feat-math/src/advanced_v3/fitted_cpp.rs`, and the pending `advanced_v3/fitted_portable.rs` plus `crates/jope-bench/src/panel/inference.rs`, which are absent from merged main `a75e543d2` and exist only as stopped, unintegrated custody patch views ([§](#c-catalogue--pending-source-references)).

<a id="c-inference--complete-registered-recipe-matrix"></a>

## 9.3 Complete registered-recipe matrix

<a id="c-inference--coverage-by-recipe-and-backend"></a>

### 9.3.1 Coverage by recipe and backend

The table covers all 26 task-compatible native benchmark recipes. **Yes means the source admits the recipe/head under the additional state restrictions; it is not a measured parity result.** Every row has native Rust fit/save/reload support in the benchmark contract. Public fitted columns come from the pending product packet, which is not in merged main `a75e543d2`; they are unverified against merged source and require actual selected-state qualification. The private C++ column describes the head gate that exists at `a75e543d2` only. Legacy `jope emit` has no column because it can emit none of the 26 recipes: each carries an `input` stage (`crates/jope-bench/src/panel/methods.rs:20-21`, `:50`), which the legacy planner rejects (`crates/feat-math/src/spec.rs:383-389`, `:592`). External adapters are covered in [§](#c-inference--external-winners).

| Task | Registered method ID | Private C++ scalar head gate | Public Python/Rust fitted | Public C++17/20 fitted | Public JavaScript fitted |
|---|---|---|---|---|---|
| Regression | `rust.ridge` | Yes | Yes | Yes | Yes |
| Regression | `rust.ridge_gcv` | Yes | Unsupported | Unsupported | Unsupported |
| Regression | `rust.elasticnet` | Yes | Unsupported | Unsupported | Unsupported |
| Regression | `rust.histgbm` | Yes | Unsupported | Unsupported | Unsupported |
| Regression | `rust.histgbm2` | Yes | Unsupported | Unsupported | Unsupported |
| Regression | `rust.ksmooth` | Yes | Unsupported | Unsupported | Unsupported |
| Regression | `rust.ensemble` | Yes | Unsupported | Unsupported | Unsupported |
| Regression | `rust.krr` | Yes | Unsupported | Unsupported | Unsupported |
| Regression | `rust.gpr` | Yes | Unsupported | Unsupported | Unsupported |
| Regression | `rust.hurdle2` | Yes | Unsupported | Unsupported | Unsupported |
| Regression | `rust.stack` | Yes | Unsupported | Unsupported | Unsupported |
| Regression | `rust.tab_boost` | Yes | Yes | Unsupported | Unsupported |
| Regression | `rust.extra_trees` | Yes | Yes | Unsupported | Unsupported |
| Regression | `rust.gam_boost` | Yes | Yes | Unsupported | Unsupported |
| Regression | `rust.pls` | Yes | Yes | Unsupported | Unsupported |
| Regression | `rust.portfolio` (`core_v1`, `preset=core`) | Yes, members checked | Unsupported | Unsupported | Unsupported |
| Regression | `rust.gpr.rbf_linear` | Yes | Unsupported | Unsupported | Unsupported |
| Binary | `rust.logit_ridge` | Yes | Yes | Yes | Yes |
| Binary | `rust.tab_boost` | Yes | Yes | Unsupported | Unsupported |
| Binary | `rust.extra_trees` | Yes | Yes | Unsupported | Unsupported |
| Binary | `rust.gam_boost` | Yes | Yes | Unsupported | Unsupported |
| Binary | `rust.portfolio` | Yes, members checked | Yes | Unsupported | Unsupported |
| Multiclass | `rust.multiclass_ridge` | Unsupported matrix ABI | Yes | Unsupported | Yes |
| Multiclass | `rust.multiclass_ovr_logistic` | Unsupported matrix ABI | Yes | Unsupported | Yes |
| Multiclass | `rust.multiclass_diagonal_lda` | Unsupported matrix ABI | Yes | Unsupported | Yes |
| Multiclass | `rust.multiclass_nearest_centroid` | Unsupported matrix ABI | Yes | Unsupported | Yes |

This yields static public coverage of 14/26 recipes for Python/Rust, 2/26 for each C++ standard and 6/26 for JavaScript. The existing private C++ head gate admits 22/26 (every regression and binary recipe); it is neither the public bundle gate nor a parity result. Those counts are not percentages of datasets, winners or actual successful parity runs. Python/Rust's twelve unsupported registered recipes are legacy regression heads outside the public fitted lowering. Generated programs can introduce additional unsupported state even when their final head appears in a supported row.

The registry records aliases separately: regression ridge has the historical alias `logistic`, histogram boosting has `gbm`, the ensemble has `ensemble_ridge_gbm`, and binary logit ridge has `logistic`. Aliases do not consume extra fit slots. `gpr.rbf_linear` remains a distinct configuration. Do not infer task compatibility from an ambiguous alias; use task plus exact registered recipe identity.

Source anchors: `crates/jope-bench/src/panel/methods.rs::rust_methods`, `advanced_v3/fitted_cpp.rs::{supports_fitted_cpp17,check_head}`, pending `advanced_v3/fitted_portable.rs::{fitted_portable_unsupported,fitted_javascript_unsupported,fitted_public_cpp_unsupported}` and `advanced_v3/fitted_portable/scalar.rs::supports`.

<a id="c-inference--state-restrictions-beyond-the-head-table"></a>

### 9.3.2 State restrictions beyond the head table

The pending public lowering validates the full model binding and fitted input dimensions. It admits IID rows and fitted IID row order only. Historical state, other model forms, target mappings, nonempty general transform lists and calibration are currently unsupported. Categorical encoding must be the admitted fitted one-hot mode. Missing-aware state is limited to the supported tabular-boosting and extra-tree implementations. Invalid coefficients, schema bindings or dimensions are errors, not legitimate unsupported fallbacks.

The benchmark's registered recipes use numeric median imputation, categorical one-hot encoding, missing indicators, `min_count=1` and `max_onehot=256`. Their fitted dictionaries and imputation statistics are part of the state. A deployment must submit columns in the recorded order and use the recorded raw string/null representation. An unseen category must follow the fitted input policy; neither the host nor caller should invent a new category mapping. Binary probability outputs include the two original class names and order; multiclass outputs include every original class column.

The existing private C++ library has broader checks for target mappings, transforms, calibration and ordered/raw input than the public fitted gate. Its broader admission must not be used to skip public privacy, raw-boundary or numerical qualification. The selected public subset is a deliberate current boundary, not evidence that every other native model is intrinsically unexportable.

<a id="c-inference--external-winners"></a>

### 9.3.3 External winners

XGBoost, LightGBM, CatBoost and scikit-learn are executable benchmark adapters for the admitted tasks. Their native fitted artifact retains the estimator pickle and preprocessing/reload evidence. The pending public winner-export helper reports standalone Python, Rust, C++17, C++20 and JavaScript export of that external library state as **typed unsupported**. It preserves the external winner, all losses and native fitted artifact. Installing a library in Python does not make its pickle a dependency-free Python inference bundle.

<a id="c-inference--private-worked-examples-and-their-measured-execution"></a>

## 9.4 Private worked examples and their measured execution

Examples A and B are runbooks that can be executed at merged main. Their runbook summary:

| Runbook part | Examples A and B |
|---|---|
| Purpose and status | Show a legacy recipe fit, five-language recipe emission with refit and a bit comparison (A), and the shape of the legacy pipeline search (B). **available**: all fourteen packet commands passed on 23 September 2026 ([§](#c-inference--measured-execution-of-the-private-examples)). |
| Prerequisites and resources | A checksum-verified immutable CLI (`J`), the reviewed source checkout (`R`) for the drivers and helpers, `python3`, `rustc`, `g++` and `node`, and a new durable private output directory (`A`) outside Git and `/tmp`. Numerical work needs its own allocation; the executed run declared one CPU, 4 GiB memory, no swap or GPU, no network and 300 seconds in total. |
| Manifest-creation command | None. The packet `examples/commands.json` is a committed, hand-prepared file, and no committed runner executes it; the 23 September run used a private controller retained in custody. `examples/execution-receipt.json` points from the packet to that run's receipts. |
| Execution | The listings below, or the argument arrays of `examples/commands.json` with their placeholders filled in. |
| Artifacts and success | For A: `fit-report.json`, `emit-report.json`, five `pred-*.txt` files, `predictions.bits` and `recipe-parity.json`. For B: `pipeline-result.json` and the pipeline's `receipt.json`. Success means every command exits zero, the validator finds all five prediction files with exactly eight finite, bit-identical values, and receipt verification passes. |
| Failure and resume | A nonzero exit is a failure; keep its stderr. The packet has no resume step: rerun it into a new output directory. |
| Limitations | Recipe refits on one synthetic fixture and one legacy pipeline fixture only. No saved-fitted-state export, neural ASM, panel or training qualification. |

<a id="c-inference--example-a-a-tiny-private-recipe-fit-and-predictions"></a>

### 9.4.1 Example A: a tiny private recipe, fit and predictions

<a id="c-inference--the-synthetic-fixture"></a>

#### 9.4.1.1 The synthetic fixture

The supplied `examples/gp-synthetic.json` contains 24 explicitly synthetic rows with two integer, non-null features and target `3*x0 − 2*x1 + 5`: 16 training rows, four validation rows and four test rows. It is a maintainer example, not one of the frozen thirty datasets or the official ten generated fixtures. `train.csv` contains the same sixteen training rows as headerless `x0,x1,target`; `predict.csv` contains the eight validation/test feature rows with targets removed.

One training row is `-12,-5,-21`; the corresponding typed JSON row is:

```json
{"id":0,"features":[-12,-5],"target":-21,"split":"train","group":null,"timestamp":null,"series":null}
```

<a id="c-inference--fit-and-emit-the-recipe"></a>

#### 9.4.1.2 Fit and emit the recipe

Set `R` to the repository root, `E` to `$R/docs/internal/asm-manual/examples`, `J` to a checksum-verified immutable CLI and `A` to a new durable private output directory. Execute numerical work only after its allocation is available. Executed on 23 September 2026: all fourteen packet commands of Examples A and B passed ([§](#c-inference--measured-execution-of-the-private-examples)). The listings are reproducible instructions, not receipts; `mkdir` and the two `cat` concatenations are setup steps outside the fourteen:

```sh
mkdir "$A"
"$J" fit --input "$E/gp-synthetic.json" \
  --genome '(pipe X (standardize) (ridge))' --json > "$A/fit-report.json"
"$J" emit --genome '(pipe X (standardize) (ridge))' \
  --lang all --output "$A/recipe" --json > "$A/emit-report.json"
```

`fit-report.json` records training/holdout R² for this numeric legacy path. The holdout prefers validation rows; the fallback is test rows if validation cannot be loaded. Do not use this convenience path to establish the frozen-panel leakage or classification protocol. It does not persist a complete fitted artifact. `emit-report.json` lists language source sizes, paths and hashes, and each emitted program below fits its own model from `train.csv`.

<a id="c-inference--run-and-compare-the-emitted-programs"></a>

#### 9.4.1.3 Run and compare the emitted programs

```sh
python3 "$A/recipe/model.py" "$E/train.csv" "$E/predict.csv" "$A/recipe/pred-python.txt"
rustc -O "$A/recipe/model.rs" -o "$A/recipe/model-rs"
"$A/recipe/model-rs" "$E/train.csv" "$E/predict.csv" "$A/recipe/pred-rust.txt"

cat "$A/recipe/model.cpp" "$R/deploy/ci/emitters/driver.cpp" > "$A/recipe/program.cpp"
g++ -std=c++17 -O3 -ffp-contract=off -fno-fast-math \
  "$A/recipe/program.cpp" -o "$A/recipe/model-cpp17"
"$A/recipe/model-cpp17" "$E/train.csv" "$E/predict.csv" "$A/recipe/pred-cpp17.txt"
g++ -std=c++20 -O3 -ffp-contract=off -fno-fast-math \
  "$A/recipe/program.cpp" -o "$A/recipe/model-cpp20"
"$A/recipe/model-cpp20" "$E/train.csv" "$E/predict.csv" "$A/recipe/pred-cpp20.txt"

cat "$A/recipe/model.js" "$R/deploy/ci/emitters/driver.js" > "$A/recipe/program.js"
node "$A/recipe/program.js" "$E/train.csv" "$E/predict.csv" "$A/recipe/pred-js.txt"
python3 "$R/deploy/ci/emitters/compare-bits.py" "$A/recipe" > "$A/predictions.bits"
python3 "$E/validate-outputs.py" --recipe-output "$A/recipe" > "$A/recipe-parity.json"
```

The supplied `validate-outputs.py` requires all five prediction files to exist, each with eight finite values, and checks their complete bit patterns before producing a success receipt. The helper compares every existing `pred-*.txt`; by itself it does not enforce that the required five files were produced. Record each command's nonzero exit as failure and retain its stderr. A comparison of only one surviving file cannot qualify the language matrix. The observed exact bit agreement on this ridge fixture does not imply exact JavaScript agreement for arbitrary transcendental operations.

`examples/commands.json` supplies the exact argument arrays and placeholders for a maintainer runner. Compiler and interpreter versions, source hashes, fixture hashes, commands, exit codes and wall/CPU measurements belong in the execution receipt. Preparation and execution receipts are retained separately: `commands.json` keeps its original `prepared, not executed` status field as a historical record, and the execution is recorded in [§](#c-inference--measured-execution-of-the-private-examples). The packet is not edited after execution. Instead, `examples/execution-receipt.json` points from it to the execution: it names the packet and its SHA-256, the fourteen-command run of 23 September 2026, and the SHA-256 of each execution receipt under `data/example-acceptance/`.

<a id="c-inference--example-b-actual-legacy-pipelinesearch-shape"></a>

### 9.4.2 Example B: actual legacy pipeline/search shape

The existing pipeline takes a locally cached catalog dataset directory with `metadata.json` or `manifest.json` and verified files. A bare CSV and the synthetic JSON above are not that catalog format. Use the committed regression fixture for a reproducible offline command:

```sh
"$J" pipeline run \
  --dataset "$R/datasets/test/pmlb-229-pwlinear" \
  --budget evaluations=4,seconds=60 --seed 7 --rng-seed 7 --cuts 1 \
  --emit rust,python,cpp17,cpp20,js --policy skip \
  --output "$A/pipeline" --json > "$A/pipeline-result.json"
"$J" pipeline verify-receipt --receipt "$A/pipeline/receipt.json" --json
```

At this budget the command measures four legacy candidates on one prepared cut: generation zero's bootstrap genome `(pipe X (ridge))` plus three grammar-random draws. No evolution runs: generation zero has `POPULATION = 12` slots, and offspring are bred only in later generations, which need an evaluation budget above twelve (the default budget is `evaluations=64,seconds=300`). The command then selects the lowest mean validation RMSE, emits supported recipe sources and writes a pipeline receipt. Sources: `crates/jope-pipeline/src/search.rs:75-80`, `:466-484`; `crates/jope-pipeline/src/select.rs:18`; `crates/jope-pipeline/src/cli.rs:38`. `--policy skip` explicitly records the absent neural proposal stage; `auto` also records a skip and `require` fails closed in this legacy route. It is not the qualified neural ASM arm.

The legacy budget is per cut, combines exploration/comparison attempts and active seconds, and is nonpreemptive for an in-flight fit. Multiple cuts can consume multiple per-cut allowances. It is not the twenty-worker panel's hard isolated 30-second fit and original global deadline contract. Preserve the recorded `deadline_overrun` and stage counts rather than applying official panel claims to this command.

The emitted winner may not be covered by C++ or JavaScript. The receipt retains exclusions and their reasons; it never replaces the selected recipe solely to make all requested languages emit. Reusing the same C++ source under `-std=c++17` and `-std=c++20` establishes the intended single-emitter structure. The maintained `deploy/ci/emitters.sh` and `deploy/ci/e2e-smoke.sh` provide larger qualification workflows; running them builds and executes models and requires its own resource allocation.

<a id="c-inference--measured-execution-of-the-private-examples"></a>

### 9.4.3 Measured execution of the private examples

<a id="c-inference--what-passed-and-what-was-measured"></a>

#### 9.4.3.1 What passed and what was measured

On 23 September 2026, all **14 unchanged example commands passed**, with zero failed, unavailable, timed-out or unstarted commands. The original command packet and static-only preparation receipts are retained unchanged as historical inputs; this execution is separate evidence.

The immutable CLI came from reviewed source `68f56404c3e199e06413bfd645d3fe0803d7b96f`; merged main `a75e543d2166017cb0139e0a251d51d9a0c98e08` has the identical source tree `176b75bc36d76c7ce4eaeb7234677ec013468889`. The executable SHA256 was `eab2d5b1b05629020aab3d540d51eb24ba0db18ff0917da188f89a34b6dc5e2b`. The original fourteen-command packet SHA256 was `f34771938934d1360cbe054689393198a53929479ba119f5a0faf2b0da7a170a`.

The private synthetic fixture contains **24 rows: 16 training, four validation and four test**. The recipe `(pipe X (standardize) (ridge))` achieved training R² **0.9995230651** and validation R² **0.9918641493**. Its emitted Python, Rust, C++17, C++20 and JavaScript programs each fitted from the same training CSV and produced **eight finite predictions with identical IEEE-754 bits**. A separate validator required all five output files and all eight rows, so a missing backend could not pass by omission.

The packet also ran the separately identified committed **PMLB 229 pwlinear** legacy pipeline fixture (`1cb97b1d7314d43d`). It completed four successful candidate measurements in one generation with no failed measurements (`stopped_by: evaluations`), selected the bootstrap genome `(pipe X (ridge))`, reported validation RMSE **0.0775685574** and R² **0.8151398564**, and passed receipt verification. Its policy stage was explicitly skipped. This fixture is not the synthetic twenty-four-row example and this invocation is not the official thirty-dataset campaign.

<a id="c-inference--resources-accounting-and-scope-limits"></a>

#### 9.4.3.2 Resources, accounting and scope limits

| Measurement | Recorded value |
|---|---|
| Controller elapsed time | 10.407138 seconds |
| Launcher elapsed time | 10.506140 seconds |
| Exact systemd unit CPU | 9.382049 CPU-seconds |
| Fourteen-command child CPU subtotal | 2.952963 CPU-seconds |
| Python | 3.12.3 |
| Rust | 1.97.1 |
| G++ | 13.3.0 |
| Node.js | 26.1.0 |

The unit CPU includes overlapping command/controller work; **do not add it to those subtotals**. The separately granted allocation declared one CPU, 4 GiB memory, zero swap/GPU, network denial and 300 seconds total, with 30-second numerical-command and 60-second compiler limits. The shared exclusive CPU lock was held and released; all command process groups drained. CPU affinity `[0]` and the dedicated unit cgroup were observed. A live property readback raced unit collection and returned unloaded defaults, so independent readback of memory/swap/quota/network/device limits is **not established**. The unit reported 5,332,992 bytes peak memory and zero swap, while cumulative reaped-child maximum RSS was 168,808 KiB; retain these different accounting domains without claiming a reconciled total peak.

This result qualifies the documented **recipe refits and legacy pipeline example**. It does not qualify exact saved-fitted-state export, neural ASM consumption, an ASM under-sixty-second warm demonstration, GPU training or the full panel. No Cargo build or download ran. Full command outputs, tool/input/source/output hashes and the original unit journal are retained in the private `agent-maintainer-manual/examples-run/` evidence directory.

Evidence pins: `HANDOFF.json` SHA256 `a05e8a0e9f9430b0c71ff23c57e5e29532a61bbe118daac3fe358b45f2423346`; `audit.json` SHA256 `363fbe62a706b44030205da78411243ba206ad2366d546a277babe966b802c91`; `result.json` SHA256 `71395a038feeef768bd6c69bb43dcabbfced20333d989cd1161c07213f4f68d6`. Byte-identical copies of these three files are in `data/example-acceptance/`, and `examples/execution-receipt.json` records the same pins beside the packet's hash.

<a id="c-inference--example-c-exact-fitted-winner-export-after-pending-integration"></a>

## 9.5 Example C: exact fitted winner export after pending integration

This runbook is the route intended to meet the rule of [§](#c-inference--deployment-support-definition-priority-and-status). At the PR83 snapshot, its two export commands, `benchmark export-winner` and `benchmark qualify-export`, existed only in stopped custody patch views; `benchmark report` existed at `a75e543d2`, but the `fitted_inference` section through which it reports export results existed only in those views. PR88 later integrated these interfaces and passed CPU fixture checks ([§](#c-status--implementation-snapshot-and-this-manuals-own-identity)); real campaign winner export remains unqualified. Its runbook summary:

| Runbook part | Example C: exact fitted winner export |
|---|---|
| Purpose and status | Export the selected winner's saved fitted state to Python, Rust, C++17, C++20 and JavaScript, and qualify each backend against the native predictions, without refit. **pending integration**: `benchmark export-winner` and `benchmark qualify-export` are absent from merged main `a75e543d2`, and the pending view compiles them only with the `asm` feature on Linux. |
| Prerequisites and resources | The same integrated immutable installed CLI, approved plan and plan ID, manifest, development root and journal as the campaign, with complete confirmation; a pinned tools file ([§](#c-inference--tool-inventory-qualification-order-and-isolation)); and the original campaign's remaining allowance. Each job has one thread and at most the original 30-second fit limit, inside the campaign's existing supervisor and worker cgroup. |
| Manifest-creation command | None of its own: the export consumes the campaign's existing plan, manifest and journal. No manifest-creation command exists at `a75e543d2`, or in the pending export view, for the tools file; it must be prepared from the installed bytes, in the campaign `Runtime` schema. |
| Execution | The three commands of [§](#c-inference--export-commands-and-winner-selection), with backends qualified in the order Python, Rust, C++17, C++20, JavaScript. |
| Artifacts and success | The semantic prepared identity, the private `prepared.json`, and a public bundle root with each validated bundle and its public manifest. A backend succeeds only through an executed qualification that meets its numerical policy; `all_backends_qualified` is true only when all five pass. |
| Failure and resume | A compiler absence, timeout, crash or parity failure blocks the later backends, and a failed comparison stays `failed_parity`. Started work without a finished receipt stays unknown until shared terminal recovery. Repeating the command cannot give an expired campaign a new allowance, and report recovery is read-only. |
| Limitations | Static public coverage is 14 of 26 recipes for Python/Rust, 2 for each C++ standard and 6 for JavaScript ([§](#c-inference--coverage-by-recipe-and-backend)). External-library winners are typed unsupported. Fold-zero parity does not prove the other four folds. No separate post-panel validation stage exists. |

<a id="c-inference--export-commands-and-winner-selection"></a>

### 9.5.1 Export commands and winner selection

Use the final integrated installed CLI, original admitted plan and journal, unchanged panel/development inputs and exact selected recipe. The pending front door, absent from merged main `a75e543d2`, accepts a dataset and an existing fitted fold, rather than a replacement recipe:

```sh
"$J" benchmark export-winner \
  --plan "$PLAN" --plan-sha256 "$PLAN_ID" --manifest "$M" --datasets "$D" \
  --journal "$JOURNAL" --dataset "$DATASET_ID" --fold 0
"$J" benchmark qualify-export \
  --prepared "$PREPARED_EXPORT" --prepared-sha256 "$PREPARED_EXPORT_ID" \
  --tools "$TOOLS" --tools-sha256 "$TOOLS_FILE_SHA256"
"$J" benchmark report \
  --plan "$PLAN" --plan-sha256 "$PLAN_ID" --manifest "$M" --datasets "$D" \
  --journal "$JOURNAL"
```

The first command returns the semantic prepared identity, private `prepared.json` and public bundle root. It chooses the smallest complete five-fold mean loss, breaking ties by method identity. Incomplete confirmation has no final winner. Choose another existing fold only when that exact fitted state is the intended qualification target; fold-zero parity does not prove the other four states.

<a id="c-inference--tool-inventory-qualification-order-and-isolation"></a>

### 9.5.2 Tool inventory, qualification order and isolation

The tools file uses the existing private `Runtime` schema keyed by command name, normally `python3`, `rustc`, `g++` and `node`. Each entry pins the physical executable and SHA256, exact version, empty argument prefix, required toolchain/package artifacts, admitted environment and unavailable reason or null. Use the real compiler executable, not an unpinned selector such as a mutable rustup shim. Tool inventory must be prepared from actual installed bytes; an illustrative JSON snippet with invented digests is not admissible.

Qualification order is Python, Rust, the same C++ emitter under C++17 then C++20, then JavaScript. Unsupported settles that backend's inventory without replacing the winner. A compiler absence, crash, timeout or parity failure prevents silently proceeding as though that backend qualified. The complete probability matrix, row order and class order are compared, not just the winning class or a rounded score.

The child receives generated inference files and unlabeled raw prediction rows. Expected predictions, original scoring labels and dataset-root access stay with the controller. Each job uses the existing bounded worker supervisor. Report recovery checks source/tool/input/artifact/command/receipt identities and runs no compiler or predictor. Started work without a finished receipt remains unknown until shared terminal recovery; it is not a successful retry.

<a id="c-inference--backend-numerical-policies-public-metadata-and-the-export-clock"></a>

### 9.5.3 Backend numerical policies, public metadata and the export clock

| Backend | Public files | Pinned numerical comparison |
|---|---|---|
| Python | `model.py` | Exact IEEE-754 bits, host platform libm. |
| Rust | `model.rs` | Exact IEEE-754 bits, host platform libm. |
| C++17 / C++20 | `model.cpp`, `codec.json`, `boundary.py` | Exact IEEE-754 bits through the typed Python/ctypes boundary; no fast-math/FMA contraction drift. |
| JavaScript | `model.js` | Absolute/relative `1e-12` policy with explicit ECMAScript Math identity. |

Necessary public metadata records the task, ordered columns/classes, fitted raw-cell/input policy, emitter/template hashes, runtime, numerical policy and file checksums. The initial bundle is `emitted-unqualified`. `all_backends_qualified` becomes true only after five actual successful qualifications. An operation can complete with explicit unsupported outcomes while that all-backends flag remains false.

These jobs use the original campaign's remaining execution allowance; they do not open a new export clock. For the official two-hour panel, no new export job may start after original `t0 + 6900 seconds`, and the last five minutes remain reporting reserve. If confirmation consumes the window, preserve unattempted export statuses. A future separately bounded post-panel validation stage would need its own explicit identity and cost accounting; the current front door does not implement one.

<a id="c-inference--future-work-and-acceptance-boundaries"></a>

## 9.6 Future work and acceptance boundaries

Priority gaps are actual full-state parity for supported winners, broader fitted-state lowering, a typed multiclass C++ boundary, qualified external-library export where feasible, and only later CUDA. Each extension needs exact native-state replay, missing/unknown-category tests, numerical policy fixed before scoring, stale-evidence refusal and public-bundle canaries. New operators additionally need ablation and compatibility evidence; adding an emitter branch alone does not establish reusable learning value.

Two existing emitter options are not exposed by any command. The legacy emitters' default math policy, `platform-libm-v1`, lowers transcendentals to the host math library, so exact parity does not extend to JavaScript or CUDA. The reserved `portable-scalar-kernel-v1` policy makes the Python/Rust pair fail closed until every emitted transcendental is routed through the portable kernel (`crates/feat-math/src/transpile.rs:51-80`, `:176-187`), while the C++ and JavaScript emitters accept it and substitute portable math helpers (`crates/feat-math/src/emit_cpp.rs:48-52`, `:72-81`; `crates/feat-math/src/emit_js.rs:37-45`); only in-source tests select it. `EmitOptions.fast_runtime` exists, but neither `jope emit` nor `jope pipeline run --emit` exposes it (`crates/jope-cli/src/kernel.rs:78-81`; `crates/jope-pipeline/src/emit.rs:6-7`, `:122`).

Retain native winners even when every requested language is unsupported. Report the full loss/ranking and export gap together. Performance claims need measured creation/prediction time on the actual artifact and hardware; binary size, source-line count, emission throughput or a passed static support function cannot substitute for those measurements.


---

# Chapter 10 — Benchmark execution, ranking, Elo and interpretation

<a id="c-benchmark--three-evidence-collections-with-different-purposes"></a>

## 10.1 Three evidence collections with different purposes

The frozen ASM panel, the historical same-cut benchmark and the imported SOTArena ladders have different inputs and protocols. Keep their figures and identities separate. A strong historical Elo is not a result for the frozen thirty-dataset ASM panel, which has not yet been run. A rank computed on development labels is not a sealed-test estimate.

| Collection | Inputs and purpose | Current interpretation |
| --- | --- | --- |
| Frozen `asm-private-30-v1` | Ten regression, ten binary and ten multiclass datasets; original frozen row assignments | Current implementation target and adaptive development domain; one regression operational proof, full panel unrun |
| Historical same-cut v1/v2 | Ten committed regression datasets; all methods share each cut; screening and deepening | Reproducible historical comparisons; v1 and v2 are separate snapshots |
| Historical tabular-regression-v2 | Imported score inventory and corrected benchmark-valid/public-development views | Historical ladder with its own coverage and validity rules; not a current ASM qualification |

The plot source tables retain method, dataset, snapshot, coverage and source hashes. Missing cells stay missing. The book does not manufacture thirty current outcomes from the historical collections. Historical method names are identifiers; the word “jope” in an identifier does not prove the method is this exact product checkpoint or executable.

<a id="c-benchmark--frozen-panel-execution-contract"></a>

## 10.2 Frozen-panel execution contract

<a id="c-benchmark--the-screen-freeze-and-confirm-procedure"></a>

### 10.2.1 The screen, freeze and confirm procedure

For each dataset, screen every applicable native method, configured external adapter and admitted search arm on development fold zero. Freeze the top five recipe identities and confirm those recipes across all five original development folds. Additionally confirm `asm.neural` if it missed the top five; the current code forces only `asm.neural`, not `asm.generic`, into this extra confirmation (`crates/jope-bench/src/panel/campaign.rs:779-783`). A configuration such as `gpr.rbf_linear` is distinct; a spelling alias does not justify an extra independent fit. The recipe is frozen, while the fitted state for each fold is learned from that fold's permitted training rows.

For each dataset, applicable methods screen on development fold zero. The top five recipes freeze before confirmation across all five development folds. `asm.neural` is also confirmed when it misses that top five. Report every original outcome. Selection can use adaptive development folds; these are not untouched external generalization estimates. Preserve the explicit selection-exposure metadata, and keep sealed evaluation untouched.

<a id="c-benchmark--adaptive-development-exposure"></a>

### 10.2.2 Adaptive development exposure

This screen/freeze/confirm design uses development data adaptively. The selected recipe can already have learned from rows that occur in a later confirmation fold. Fold-zero labels influence screening. The implementation records these intersections; the five-fold summary remains adaptive development evidence. Sealed rows and sealed meta-test families stay outside learning, optimization and this documentation audit. The word “test” in an optimization-visible file does not make it a valid test set. Under the owner's family-level domain rule, the frozen panel remains an adaptive development domain even after a complete campaign ([§](#c-data--three-family-level-domains)).

The frozen protocol screens methods on development fold 0, freezes the top five recipes, and confirms those recipes on the same five development folds; `asm.neural` is additionally confirmed if it missed the top five. The synthesized recipe was learned from the outer fold 0 fitting set, containing development folds 1–4. Thus those later confirmation folds are exposed to recipe learning. Every frozen recipe's fold 0 labels influenced screen selection. The report derives row intersections and records them per recipe/fold; it labels the selected five-fold summary adaptive development and leaves sealed evaluation unopened. This is the chosen protocol, not an unbiased outer generalization estimate. Source: `crates/jope-bench/src/panel/selection_exposure.rs:1`.

The exposure is not symmetric, so the adaptive bias has a direction: it favours `asm.*` over fixed-configuration comparators. A synthesized `asm.*` recipe is recorded with the learning rule `inner_selection_on_cut_zero_training_rows`: it was chosen by a search of up to 1,000 evaluations per arm on the fold 0 fitting rows, which are development folds 1–4. Every `rust.*` and `external.*` recipe is recorded as `fixed_before_current_dataset_screen`. All methods share only the fold 0 screen selection. The five-fold mean therefore compares an optimistic ASM estimate with comparator estimates exposed only through that screen, and the difference can grow as search pressure grows; an apparent panel win can arise from selection alone. Sources: `crates/jope-bench/src/panel/selection_exposure.rs:71`, `:91-92`; `crates/jope-bench/src/panel/asm_campaign.rs:37`; `crates/jope-bench/src/panel/optimize.rs:132-133`. A nested variant, with an independent ASM search inside each outer fold, is not implemented; requiring it before any external claim is a review proposal pending owner decision.

<a id="c-benchmark--running-the-frozen-panel-campaign-plan-execute-and-report"></a>

## 10.3 Running the frozen-panel campaign: plan, execute and report

This runbook has seven parts: purpose and status, prerequisites and resources, manifest creation, execution, artifacts and success, failure and resume, and limitations. Status labels are as of 2026-09-23 about 23:55 UTC.

<a id="c-benchmark--purpose-status-and-prerequisites"></a>

### 10.3.1 Purpose, status and prerequisites

**Purpose.** Run the unchanged thirty-dataset protocol once, as roadmap item PANEL-01 ([§](#c-plan--compact-task-index)): screen every applicable method on development fold zero, freeze five recipes, confirm them on all five development folds, and report adaptive development evidence for every dataset.

**Status: implemented but unqualified.** `optimize plan`, `benchmark run` and `benchmark report` exist in the immutable CLI at `68f56404`, which has the same tree as main `a75e543d2` ([§](#c-installation--command-surfaces-and-the-stopped-integration-order)). No full-panel campaign has run. ASM execution so far is limited to separately bounded one-dataset regression proofs ([§](#c-status--canonical-proof-record-and-per-arm-facts)). A campaign with a newly trained policy is pending integration: at `a75e543d2` the configuration validator accepts only the accepted step-60000 policy bytes (`crates/jope-bench/src/panel/optimize.rs:161-165`), and PANEL-01 waits for the final trained configuration.

**Prerequisites and resources.**

| Prerequisite | Requirement |
|---|---|
| CLI `$J` | Linux build with the `asm` feature. `run` refuses a binary whose path or SHA-256 differs from the plan's pinned native runtime (`crates/jope-bench/src/panel/campaign_cli.rs:175-184`). |
| Panel `$M` and data `$D` | Unchanged manifest `ba7dace4…` and verified prepared development data ([§](#c-installation--prepare-and-verify-the-panel)) |
| Base plan and `BASE_PLAN_ID` | Runtime-aware comparator plan and its printed `plan_sha256`. `optimize plan` refuses a base plan whose scope is not `comparator-qualification` (`crates/jope-bench/src/panel/optimize_cli.rs:91-94`). |
| ASM configuration and its file SHA-256 | Complete `jope.asm-configuration/v1` file. It pins a `jope.asm-worker-build-inventory/v1` release manifest, source manifest, checkpoint receipt, policy weights, vocabulary, frozen transforms, policy and encoder runtimes, encoder assets and optional context qualification (`crates/jope-bench/src/panel/optimize.rs:99-122`). |
| Journal | Absolute durable path outside `/tmp` and Git working trees; `run` creates it with mode 0700 (`crates/jope-bench/src/panel/campaign_cli.rs:130-147`). |
| Resources | Fixed by the base plan: 20 workers and 20 CPUs, 64 GiB, zero swap, at most one GPU task, one thread per worker, 30 seconds per fit, 7,200 seconds total including a 300-second reporting reserve (`crates/jope-bench/src/panel/campaign_cli.rs:83-93`). Do not overlap exclusive full-CPU CI ([§](#c-scheduling--shared-compute)). |

**Manifest creation.** The panel manifest is committed and frozen: it is verified, never created ([§](#c-installation--prepare-and-verify-the-panel)). Prepared data, the runtime inventory and the base plan have creation commands in [§](#c-installation--prepare-the-panel-and-external-runtimes), and `optimize plan` below creates the campaign plan. No command at `a75e543d2` creates the ASM configuration or its `jope.asm-worker-build-inventory/v1` release manifest: the committed source only reads and verifies both schemas (`crates/jope-bench/src/panel/optimize.rs:29`, `:102`; `crates/jope-bench/src/panel/worker_build.rs:9`, `:81`). Assemble them from real build, asset and qualification receipts; a short hand-written configuration with invented hashes cannot replace those artifacts ([§](#c-search--required-inputs)).

<a id="c-benchmark--preregistered-arms-and-per-task-scope"></a>

### 10.3.2 Preregistered arms and per-task scope

Prepare the qualified configuration before opening an execution window. It binds the actual context producer, source/worker/model inputs, numerical qualification and policy configuration. The standard comparison preregisters a 64-slot policy pool plus 16 exploration candidates against 80 generic candidates, with equal declared arm allowances. Policy-pool shortfalls carry `neural_rejection_fallback`; only `neural_policy` slots count as genuine checkpoint proposals. Generation-zero neural slots must actually fit; a proposal file or generic-only result cannot demonstrate neural consumption.

That preregistration applies in full only to regression at the pinned source `68f56404`. The accepted policy is regression-only, so binary and multiclass datasets have no neural arm, and their generic arm searches ridge heads only (`logit_ridge` or `multiclass_ridge` with at most four transforms; see [§](#c-architecture--experimental-classification-context)):

| Task | Frozen datasets | Neural arm | Generic arm | Non-ASM comparators |
|---|---|---|---|---|
| Regression | 10 | 64 policy slots plus 16 exploration candidates | 80 grammar-uniform candidates | 17 native recipes plus 4 external engines |
| Binary | 10 | Unavailable | 80 `logit_ridge` candidates | 5 native recipes plus 4 external engines |
| Multiclass | 10 | Unavailable | 80 `multiclass_ridge` candidates | 4 native recipes plus 4 external engines |

At this source, a better policy therefore cannot change binary or multiclass results. Sources: `crates/jope-bench/src/panel/optimize.rs:34`, `:35`, `:493`; `crates/jope-pipeline/src/neural_search.rs:28`.

<a id="c-benchmark--execution-artifacts-and-success"></a>

### 10.3.3 Execution, artifacts and success

Plan first. Planning performs zero fits and reports `execution_budget_started: false` (`crates/jope-bench/src/panel/optimize_cli.rs:126`). Capture the printed semantic plan ID rather than hashing the file ([§](#c-installation--physical-layout-and-identities)), then execute and report:

```sh
"$J" optimize plan \
  --base-plan "$A/base.json" --base-plan-sha256 "$BASE_PLAN_ID" \
  --configuration "$ASM_CONFIGURATION" \
  --configuration-sha256 "$ASM_CONFIGURATION_FILE_SHA256" \
  --output "$A/panel.json" > "$A/panel-plan-receipt.json"
PANEL_PLAN_ID=$(jq -er '.plan_sha256' "$A/panel-plan-receipt.json")

"$J" benchmark run --plan "$A/panel.json" --plan-sha256 "$PANEL_PLAN_ID" \
  --manifest "$M" --datasets "$D" --journal "$A/panel-journal"

"$J" benchmark report --plan "$A/panel.json" --plan-sha256 "$PANEL_PLAN_ID" \
  --manifest "$M" --datasets "$D" --journal "$A/panel-journal"
```

**Artifacts.** `optimize plan` prints `plan_sha256`, `scope: "asm-panel"`, `fits: 0`, `neural_slots: 64`, `exploration_slots: 16`, `generic_slots: 80` and `checkpoint_training_exposure: "unknown"` (`crates/jope-bench/src/panel/optimize_cli.rs:126-129`). `benchmark run` writes the retained plan as immutable `campaign.json` in the journal (`crates/jope-bench/src/panel/campaign.rs:661`) and, when the campaign completes, an immutable report `reports/<report SHA-256>.json` with schema `jope.panel-report/v2` (`campaign.rs:890-900`). The base plan declares `finished`, `observations`, `frozen` and `reports` as expected artifacts (`crates/jope-bench/src/panel/campaign_cli.rs:94-99`). `benchmark run` returns `ok: true` only when its supervised launch exits with status 0; once `campaign.json` exists, its output also carries the read-only recovery report (`campaign_cli.rs:199-208`). Retain the evidence listed in [§](#c-scheduling--artifacts-what-evidence-to-retain).

**Success.** Success is a `jope.panel-report/v2` whose dataset list covers all thirty datasets, each with screen outcomes, frozen recipes, confirmation observations, recipe summaries and selection exposure (`campaign.rs:869-872`), and which meets the minimum complete report in [§](#c-benchmark--minimum-complete-future-campaign-report). Every report states `sealed_inputs_opened: false`, `optimization_visible_labels: "development"`, `confirmation_role: "adaptive_development"` and `five_fold_summary_is_unbiased_generalization_estimate: false` (`campaign.rs:893-894`). A successful campaign is therefore complete adaptive development evidence, not a sealed-test or unseen-family result.

<a id="c-benchmark--failure-resume-and-limitations"></a>

### 10.3.4 Failure, resume and limitations

`optimize run` and `optimize report` delegate to the same campaign contracts. Repeating run with the same plan and journal resumes the original authenticated state and absolute deadline. A restart does not grant another two hours. Do not repair journals by editing receipt fields or copying successes from another experiment. Read-only report recovery authenticates retained artifacts and recomputes scores; it does not fit another predictor.

**Failure and non-final outcomes.** When bounded execution cannot be admitted, the supervised campaign reports `jope.panel-admission/v2` with `ok: false`, and no selection or scientific result is frozen (`crates/jope-bench/src/panel/campaign.rs:728-733`). Resumable inventory is reported as `jope.asm-partial/v1` for the paired arms (`campaign.rs:754-756`) or as `jope.panel-partial/v2` with `ok: false` for confirmation (`campaign.rs:839-841`). Resume each by repeating `benchmark run` with the same plan, plan ID and journal inside the original window; stop reasons and recovery phases are in [§](#c-scheduling--reading-stop-reasons-and-resuming). A read-only `benchmark report` refuses a journal that belongs to another plan and returns `jope.panel-recovery-report/v2` with `fits_started: 0` and `clock_started: false` (`campaign.rs:974-978`, `:1216-1222`).

**Limitations.** Both panel report constructors at `a75e543d2` set `complete_asm_demonstration: false` (`campaign.rs:895`, `:1222`). Binary and multiclass datasets have no neural arm at this source. The five-fold summary is adaptive development evidence whose exposure favours `asm.*` ([§](#c-benchmark--adaptive-development-exposure)); comparators are fixed default configurations ([§](#c-benchmark--minimum-complete-future-campaign-report)); and the panel's policy and pretrained-encoder exposure are unknown ([§](#c-data--three-family-level-domains)). Selected-winner export and a newly trained policy are pending integration ([§](#c-installation--command-surfaces-and-the-stopped-integration-order)).

<a id="c-benchmark--metrics-and-class-order"></a>

## 10.4 Metrics and class order

| Task | Primary reported metric | Smaller-is-better search loss | Required prediction shape |
| --- | --- | --- | --- |
| Regression | RMSE | RMSE | One finite real prediction per row |
| Binary classification | AUROC | `1 − AUROC` | Probabilities with original two-class order |
| Multiclass classification | Log loss | Log loss | One probability for every original class on every row |

RMSE is the square root of the mean squared prediction error. It has the target's units, so a raw RMSE on machine CPU cannot be averaged meaningfully with an unrelated dataset's RMSE without an explicitly declared normalization. AUROC measures score ordering between positive and negative examples; the positive class and class order must be fixed by the task contract. A one-class evaluation fold cannot produce a meaningful AUROC. Multiclass log loss averages the negative logarithm of the probability assigned to the true class. Integer class predictions alone cannot supply that probability-matrix metric.

Probability shape, finiteness, class names and order belong to the authenticated prediction boundary. A backend that produces the same numbers in a different class order has failed parity. A backend with only scalar output cannot be advertised as supporting a multiclass winner. Any numerical clipping, normalization, accumulation, tie or tolerance rule must be fixed before scoring and recorded with the evaluator; changing the rule afterward changes comparison identity.

Regression is scored by RMSE. Binary classification uses AUROC, with the frozen second class as positive and loss `1 − AUROC`. Multiclass uses the full probability matrix and original class order for log loss. Predictions must be finite; class order and row count must match; probability rows are validated without silently renormalizing. The pinned log-loss epsilon is `1e-15`, and probability row-sum tolerance is `1e-6`.

Regression uses RMSE; binary uses AUROC with minimization loss `1−AUROC`; multiclass uses probability-matrix log loss with the frozen class ordering. Features retain numeric/categorical type and missing cells; original labels are not converted to manufacture a task. Screen all applicable methods on original fold zero, freeze five recipes, confirm across all five development folds, and confirm `asm.neural` additionally if it missed the five. These optimization-visible folds remain development evidence. Sealed evaluation is separate.

The historical same-cut command has an older contract. It constructs its own hashed cuts, uses regression R², binary AUROC and multiclass macro-F1 for screening, and applies its recorded robust-fitness summary during deepening. Its regression ladder uses RMSE; its classification ladder can maximize AUROC or macro-F1. Those historical definitions must not be silently copied into the frozen ASM panel's multiclass log-loss contract. Source: `docs/same-cut-bench.md` and `crates/jope-bench/src/eval.rs`; current panel code lives under `crates/jope-bench/src/panel/`.

<a id="c-benchmark--what-elo-means-here"></a>

## 10.5 What Elo means here

<a id="c-benchmark--the-bradleyterry-model-tie-rule-and-defaults"></a>

### 10.5.1 The Bradley–Terry model, tie rule and defaults

The arena implementation fits a regularized Bradley–Terry model on an Elo scale. For two ratings A and B, its modeled pairwise probability is:

`P(A beats B) = 1 / (1 + 10 ** ((rating_B − rating_A) / 400))`.

Each dataset with both eligible scores supplies one equally weighted comparison for that method pair. The observed outcome is 1 for a win, 0 for a loss and 0.5 for a tie. Metric direction is explicit. The relative tie condition is `abs(a − b) <= 1e-12 * max(abs(a), abs(b), 1)`. A missing or ineligible score supplies no invented numerical loss.

The default implementation starts and recenters ratings at 1500, runs at most 250 diagonal-Newton iterations, applies L2 regularization of `1e-4`, clips each iteration's update to ±40 Elo points and stops when the largest update is below `1e-8`. The ±40 setting is an iteration clamp, not an online match-by-match chess update. Source: `crates/jope-arena/src/elo.rs`.

Two source paths preserve historical arithmetic: one accumulates individual sorted dataset/pair matches in the original Python order, and the reporter aggregates each pair's sufficient statistics. They are equivalent in exact arithmetic but can differ slightly in floating-point accumulation. Reproduction must identify the path and snapshot, not merely say “Elo.”

<a id="c-benchmark--interpreting-a-rating"></a>

### 10.5.2 Interpreting a rating

The rating is relative to the included method universe, evidence eligibility and overlap graph. A high rating on a handful of datasets has different coverage from the same rating on every dataset. Report dataset coverage, full-coverage rank, average rank, wins, pending cells and protocol beside Elo. Adding methods, removing invalid rows or changing overlap can change ratings without any predictor changing. Elo does not measure the magnitude of an RMSE improvement, runtime, memory, statistical certainty or future unseen-data performance. No confidence interval or significance claim is implied by a plotted point.

<a id="c-benchmark--how-the-retained-elo-scale-is-constructed"></a>

### 10.5.3 How the retained Elo scale is constructed

`crates/jope-arena/src/elo.rs` fits regularized Bradley–Terry ratings. Each dataset contributes one equal-weight comparison for each pair of methods with available scores. The preferred direction is taken from the metric; RMSE comparisons minimize. A tie has outcome 0.5 when `|a−b| ≤ 1e−12 × max(|a|,|b|,1)`; otherwise the better score receives 1 and the other 0. Missing cells create no comparison and must not become invented defeats.

The expected pairwise outcome is `P(A beats B)=1/(1+10^((rating_B−rating_A)/400))`. The committed implementation centers at 1500, uses an L2 regularization coefficient of 1e−4, caps diagonal-Newton iterations at 250, clamps each iteration's rating update to ±40, and stops when the largest update falls below 1e−8. It preserves sorted dataset/method-pair accumulation and the historical numerical summation policy for reproduction. This describes the source algorithm; the audit did not rerun it or infer convergence where the snapshot does not report it.

A rating difference summarizes outcomes within one comparison pool. It does not encode an absolute RMSE difference, probability of unseen-dataset superiority, confidence interval, or universal ranking across tasks. A method observed only on its screened favorable subset needs its coverage displayed beside its rating. The historical `afm.lite-race.v1.resampled` row describes that search procedure; it does not prove the current neural ASM model.

<a id="c-benchmark--minimum-complete-future-campaign-report"></a>

## 10.6 Minimum complete future campaign report

For every one of the thirty immutable datasets, retain the input and cut identities; applicable, skipped and unavailable methods with reasons; screen outcomes; frozen finalists; all confirmation-fold outcomes; `asm.neural`'s extra confirmation when needed; original and fitted recipe identities; class order; selected loss and exposure labels; resource limits and original deadlines; failures and cancellations; actual fit/predict counts; and retained winner/backend outcomes. Report all losses and gaps, not only wins. A deadline-exhausted dataset remains in the denominator with an explicit incomplete outcome.

The owner's target of ten wins per task category is not yet a countable definition. The estimate that counts, the eligible comparator set, whether `asm.generic` is a comparator or an ASM control, the tie rule and whether an incomplete or unconfirmed dataset can count as a win are not fixed, and no code at merged main `a75e543d2` counts wins against the best non-ASM comparator. A review proposal, pending owner ratification and not adopted, defines PANEL-WINS(asm-panel-v1) as, per task, the number of the ten frozen datasets on which `asm.neural`'s confirmed five-fold mean development loss is strictly below the minimum over all eligible frozen non-ASM comparators (every `rust.*` and `external.*` recipe). Under that proposal, `asm.generic` is reported in its own column as an ASM control, ties and incomplete, failed or unconfirmed datasets count as not won, and the result is reported as (regression, binary, multiclass). The separate win columns are described in [§](#c-learning--win-counting-and-a-proposed-acceptance-record). With exactly ten frozen datasets per task, ten wins per category would be a 30/30 sweep. It remains a measured target; the current evidence does not promise that result. The full-panel report, newly trained policy consumption, selected fitted-winner parity and serving promotion remain separate acceptance milestones.

Panel comparators are fixed configurations. Each native `rust.*` recipe keeps its registered configuration, and the four external engines (XGBoost, LightGBM, CatBoost and scikit-learn) run as `external.{engine}.default`; the campaign performs no per-dataset hyperparameter search for any comparator (a head's own internal fitting, such as `ridge_gcv`'s generalized cross-validation, is part of its fixed recipe). Each ASM arm, by contrast, searches for up to 120 seconds and up to 1,000 evaluations. Sources: `crates/jope-bench/src/panel/methods.rs:60-63`, `crates/jope-bench/src/panel/campaign_cli.rs:67-72`, `crates/jope-bench/src/panel/optimize.rs:132-133`. The strongest historical benchmark-valid method, TabFM Ensemble (rank 1, Elo 1934.9), is not a panel comparator; on that historical ladder the `jope-dime-regressor` entry is rank 31 at Elo 1687.0 ([§](#c-targets--why-the-two-historical-ladders-differ)). Ten panel wins per category against these default comparators would therefore not show that ASM beats the strongest known tabular methods. Tuned or foundation-model comparator tiers for a future `asm-panel-v2`, never retroactive to v1, are a review proposal pending owner decision.

Classification search is also narrower than its comparators. At merged main `a75e543d2`, binary and multiclass datasets have no neural arm, because the accepted policy is regression-only, and their `asm.generic` search samples only ridge heads (`logit_ridge` for binary, `multiclass_ridge` for multiclass). Their comparators are five binary or four multiclass native recipes plus the four external engines. Sources: `crates/jope-bench/src/panel/optimize.rs:35`, `:492-493`; `crates/jope-bench/src/panel/methods.rs:90-104`; [§](#c-architecture--experimental-classification-context).


---

# Chapter 11 — Scheduling, cancellation, recovery and accounting

<a id="c-scheduling--arm-order-and-clocks"></a>

## 11.1 Arm order and clocks

The arm order is fixed: neural first, then generic. Fresh context encoding and policy production run inside the neural arm clock, and each arm is clipped to the original execution cutoff. In the origin proof at source `68f56404` the producer gate opened 21.094 seconds after the campaign start and its exit was observed at 41.581 seconds; its measured wall time was 20.486968041 seconds. The neural cutoff then lay 78.613 seconds ahead, before allowing for post-exit verification/setup, so this is remaining arm time rather than guaranteed fit execution time. The neural arm stopped at `original_deadline`. The generic arm began 0.553 seconds after that cutoff, retained its full declared 120-second allowance and stopped at `evaluation_limit`. Equal declared allowances are not equal fit-only time. Sources: `crates/jope-pipeline/src/neural_search.rs:654`, `:701`, `:729`, `:772`; `crates/jope-pipeline/src/execution.rs:592`; the accepted proposal execution and recovered report.

<a id="c-scheduling--resource-limits-shared-compute-and-supervision"></a>

## 11.2 Resource limits, shared compute and supervision

<a id="c-scheduling--twenty-bounded-workers-resource-limits-and-supervision"></a>

### 11.2.1 Twenty bounded workers: resource limits and supervision

| Constraint | Official panel policy |
|---|---|
| Worker/CPU count | Twenty isolated workers shared by search and benchmarking; at most twenty CPUs. |
| Thread count | One thread per worker, including external libraries. |
| Memory and swap | 64 GiB total; zero swap. The shared policy reserves 4 GiB for the controller and caps each fit at 3 GiB. |
| GPU capacity | At most one GPU task; the approved training hardware is one RTX 3090. GPU training has its own separate total hour. |
| Fit and search time | At most 30 seconds per fit, normally 120 seconds per search arm. |
| Panel wall time | 7,200 seconds total, including 300 seconds reserved for reporting. No new fit/qualification starts after the original execution cutoff at 6,900 seconds. |
| Coexistence | Do not overlap the experiment with exclusive full-CPU CI. |

The CLI uses the shared Linux systemd/cgroup-v2 supervisor and process-group cleanup. Missing host capability is unavailable execution. Do not put an additional outer allocation lock around a command that already acquires the shared allocation. A child crash, timeout, cancellation, launch failure and unavailable dependency have different recorded states; unknown work/cost is not zero.

The default admitted panel allocation is twenty CPUs/workers, 64 GiB memory, zero swap, one RTX 3090 when GPU work is explicitly allocated, thirty seconds per fit and 120 seconds per search arm. The full panel receives two hours total, including a five-minute reporting reserve. Training receives one GPU-hour total and twenty minutes per category. Do not overlap numerical campaigns with exclusive full-CPU CI. Resume continues the original campaign deadline and resource envelope.

<a id="c-scheduling--resource-policy-and-the-training-clock"></a>

### 11.2.2 Resource policy and the training clock

| Operation | Required policy |
|---|---|
| Official frozen30 panel | 20 workers/CPUs, 64GiB, zero swap, at most one GPU task, 30 seconds per fit, 7200 seconds total including300-second reporting reserve |
| Paired search | 64 policy-pool slots (true neural or explicit fallback)+16 exploration versus 80 generic; default 120 seconds per arm; same declared allowance and grammar/variation identity |
| Separate origin/teacher qualification | Explicitly preregistered resource manifest and separate journal; no inferred extra allowance and no reset of another campaign |
| Training | One visible RTX3090, one campaign of at most3600 seconds, at most1200 seconds per category, one active category worker, CPU allowance20,64GiB,zero swap |
| Training numerical runtime | One Torch thread, FP32 storage, CUDA TF32 enabled by declared policy, AMP disabled; explicit experimental lineage |
| CPU training | Explicit synthetic qualification only; never an automatic fallback from requested CUDA |
| Full-CPU CI | Exclusive; no overlapping campaign/training resource allocation |

The training device check requires exactly one visible CUDA device, an RTX3090 name and a usable GPU UUID. It records requested/resolved device identity. Resume requires the same identity. A missing GPU is an explicit failure; it must not become a successful CPU run under a CUDA claim.

Native planning does not open the training clock. First supervised admission creates the global window. Each category receives at most20 minutes, capped by the remaining global deadline minus its five-second reporting reserve. All passes, validation, reloads and checkpoint work must fit the original window. A pass is one supervised worker attempt: one call of `train_policy_v3_from_jsonl` with `epochs=1` and batch size 256 (2 for synthetic fixtures); every later pass resumes from the previous pass's `resume.pt`. Two passes are required for continuation evidence; they do not receive two category budgets. A fresh directory or modified plan is a new campaign, not a legitimate way to finish an expired approved campaign.

Sources: current `crates/jope-bench/src/panel/campaign_cli.rs:51`; pending `python/src/jope/policy/training_campaign.py:164`, `:704`, `:720`, `:1043`; `crates/jope-cli/src/train.rs:31`.

<a id="c-scheduling--shared-compute"></a>

### 11.2.3 Shared compute

Fast CI runs on xbabe1. The canonical Ubuntu machine is local xbabe2 for full CPU work. The user's SSH alias `xbabe2` points elsewhere; use the runbook's local form for this machine.

The exclusive full runner and numerical campaigns share a host lock. Drain owned listeners and verify no active full job before numerical allocation. Do not seize the host by stopping another job. Record allocation ownership and restore the runner after numerical work releases it. Keep full CPU CI separate from the one-GPU-hour training campaign and the frozen panel.

Missing cgroups or dependencies cause explicit failures/unavailable outcomes; they do not authorize an unbounded fallback. Worker control readbacks, not prose, establish the limits actually enforced.

<a id="c-scheduling--cancellation-resume-and-recovery"></a>

## 11.3 Cancellation, resume and recovery

<a id="c-scheduling--reading-stop-reasons-and-resuming"></a>

### 11.3.1 Reading stop reasons and resuming

Final search reasons include `evaluation_limit`, `original_deadline`, `search_space_exhausted` and `no_measured_parent`. `admission_pending_original_window` and `cancelled_original_window` are non-final: no winner may be frozen while either holds. The proof front door can return `ok:false, state:pending_original_window`. Resume the same plan and journal, preserving its original arm and campaign deadlines; resume never creates another allowance. Before cutoff, unresolved never-admitted inventory may be retried; at/after cutoff it can only be finalized under the existing evidence rules.

A setup failure before gate release becomes terminal `budget_exhausted` if its original job cutoff has been reached; setup consumption stays unknown. Before cutoff, ordinary pre-gate failure is `unavailable` and cancellation is `cancelled`, subject to the retryable-inventory rules. Interrupted started-only attempts are excluded from that category, as described in [§](#c-scheduling--interrupted-attempts-and-proof-interpretation). The earlier proof at source `3fd7d5b2` retained exit 1 after 245.93 seconds because an expired pre-gate attempt incorrectly remained retryable; its unchanged journal is evidence of the failure, not permission to reset the clock. Sources: `crates/jope-pipeline/src/neural_search.rs:1308`, `:1380`, `:1457`, `:845`; `crates/jope-pipeline/src/execution.rs:1209`; `crates/jope-bench/src/panel/origin_proof.rs:194`.

<a id="c-scheduling--interrupted-attempts-and-proof-interpretation"></a>

### 11.3.2 Interrupted attempts and proof interpretation

An attempt with a `started/` record and no `finished/` record is never rerun, and the pool does not recover its original result or final counters. After its process group is safely drained, the pool persists terminal `cancelled` with reason `controller_interrupted: original attempt not retried`; CPU, wall time and exit accounting remain unknown. If recovery cannot drain the group, no recovered terminal record is persisted and further admission is blocked (`recovery_not_drained`). Such attempts are excluded from retryable never-admitted inventory. Separately, a normally watched attempt persists its observed terminal outcome before potentially slow cleanup; that rule does not reconstruct interrupted accounting. The search uses immutable observed journals and refuses changed completed evidence. At cutoff, only proven never-admitted inventory may normalize; the per-fold replay digest preserves ordered real attempts and output identities. Both live PoolBackend feedback and recovered FitAdapter scoring compute the same shared digest.

A regression caught during acceptance illustrates the distinction: the proof at source `d0f5748b` finished 964 worker commands (834 completed, 130 failed), but its recovered report failed because the two feedback constructors disagreed about replay evidence. Those fit counts did not justify declaring the report passed. The `68f56404` correction restored one shared digest computation, followed by a fresh bounded proof and recovered report. The `d0f5748b` records were retained unchanged.

Never extend an expired original deadline, mutate a failed result into a success, reinterpret an unknown cost as zero, or cite syntax parsing as model execution. A report can honestly say “source implementation prepared; CPU fixture passed; live training pending.” That is more useful than collapsing all three into one completion flag.

<a id="c-scheduling--recovery-phases"></a>

### 11.3.3 Recovery phases

Identify whether failure occurred before admission, in setup, after the gate, during cleanup, in controller scoring or during report generation.

- A pre-admission dependency refusal may be retried only inside the original window.
- Setup can consume CPU without starting a fit. Unknown setup cost stays unknown.
- A started identity is durable. Controller death does not authorize silently repeating it.
- Completed folds inside a failed candidate still require artifact authentication.
- Original-cutoff inventory settles without extending the clock.
- A report can fail after successful fits. Retain both; do not call the whole operation successful.
- Saving a terminal outcome does not release cleanup authority before descendants drain.

Resume with the same plan and journal. A changed source, dataset, budget or protocol requires a separately identified experiment with the old evidence retained, not a relabelled continuation.

<a id="c-scheduling--artifacts-what-evidence-to-retain"></a>

## 11.4 Artifacts: what evidence to retain

| Evidence | What to retain and inspect |
|---|---|
| Source/build/install | Full commit, source manifest, Cargo messages, exact features/toolchain, dependency graph, binary and worker hashes, physical paths, cold timings and exits. |
| Plan and configuration | Frozen panel identity, all runtime/model/encoder pins, resource limits, semantic plan ID, seeds, numerical policy and original window. |
| Worker execution | Started/finished union, exact job/input identity, output inventory, exits/signals, timeouts, process-group drainage, memory/CPU counters where known. |
| Scientific results | All screen/confirmation losses, frozen recipe set, original row/class order, baseline, selection exposure and unsupported/unavailable outcomes. |
| Fitted predictor | Complete private state, fitted preprocessing, recipe, native reload proof, prediction hashes and exact artifact identity. |
| Export qualification | Exact winner/fold/state, public bundle, toolchain/input/class/policy hashes, actual commands and parity result. See [§](#c-inference--example-c-exact-fitted-winner-export-after-pending-integration). |

Retain source revision and input manifest; exact command, directory, exit and timestamps; executable/runtime/features/environment; data/split/evaluator/vocabulary/protocol identities; original limits/start/deadline/cutoff; collected/executed/failed/skipped/excluded/unavailable inventories; artifact hashes and worker receipts; report replay; limitations and next owner.

Verify a source manifest against the committed blobs it claims. The current bytes at `target/release/jope` need not be the bytes used by an earlier result. Preserve immutable executable copies with their build receipts before another build replaces the usual path.

Current ASM custody is outside Git at `/home/ubuntu/jope-rescue-20260921/asm-integration-20260923/`. This host location is not a portable runtime default.

<a id="c-scheduling--cost-accounting"></a>

## 11.5 Cost accounting

<a id="c-scheduling--counting-predictors-recipes-commands-and-export-cost"></a>

### 11.5.1 Counting predictors, recipes, commands and export cost

A recipe may be evaluated on multiple folds or re-used in later fits. Therefore proposal count, recipe count, unique fitted predictors, launched commands and physical fit starts are different denominators. A cache hit does not become another new predictor; a partial/unknown attempt does not become a zero-cost fit. [§](#c-scheduling--worked-example-cpu-wall-time-and-per-predictor-cost-of-the-one-dataset-proof) applies this accounting to the current proof.

Report elapsed wall time, summed worker wall time, worker CPU time, aggregate supervisor CPU, GPU time if measured, launched commands, successful fits, failures, unique fitted predictors and unique reusable recipes separately. Aggregate supervisor CPU contains worker CPU; adding them double-counts. Parallel worker durations do not equal elapsed wall time. Deduplicated/cache reuse is useful but is not another physical fitted predictor.

Count a unique fitted predictor by authenticated complete fitted-artifact identity, and a reusable recipe by the declared recipe identity. This does not establish behavioral uniqueness. Fit failures, policy/context work and preparation still incur cost. A worker CPU subtotal is not automatically total creation CPU. Validated systemd unit CPU can include setup, controller and children; it overlaps worker CPU and must not be added to it. Unknown counters stay null with a reason.

In the pending ASM export accounting, `fitted_export_commands` is a subset of global worker commands. Comparator-only accounting requires its separate export cost to be added. Export preparation is measured separately; final receipt publication and other unmeasured overhead remain explicit. Retain the original panel cutoff: backend qualification after a campaign expires is unattempted through the existing front door. A distinct post-panel bounded validation stage is not implemented and cannot be claimed inside the two-hour panel allowance.

<a id="c-scheduling--worked-example-cpu-wall-time-and-per-predictor-cost-of-the-one-dataset-proof"></a>

### 11.5.2 Worked example: CPU, wall time and per-predictor cost of the one-dataset proof

Exact supervisor CPU was **270.889403 CPU-seconds**; summed worker CPU was **181.220773 CPU-seconds**. The supervisor aggregate contains worker use: **do not add them**. Summed worker wall durations were **202.266164543 seconds**; concurrent durations are not elapsed campaign time.

Amortized over 832 unique predictors, these observations yield approximately 0.217813 worker CPU-seconds, 0.325588 aggregate CPU-seconds and 0.287432 run wall-seconds per predictor. Over 164 recipes, they yield approximately 1.105005, 1.651765 and 1.458190 seconds. These are campaign accounting ratios, not isolated latency predictions for arbitrary models.

The current one-dataset proof completed 964 worker commands: 834 completed and 130 failed. It counted 832 unique fitted predictors and 164 recipes, plus two final refits with 34 development-hold predictions each. Its run took 239.143121970 seconds; independent report recovery took 18.718997325 seconds. These values are measured at the qualified source and protocol, not projected full-panel throughput. The CPU totals are in the proof record above and the amortized denominators in the table below.

The supervisor figure is the exact `CPU_USAGE_NSEC=270889403000` event from local PID 1 systemd, linked to the precise supervisor unit, source, launch receipt and original experiment window. It includes controller/setup threads and child workers and excludes the outer launcher, immutable build and external preparation. It overlaps the worker CPU total; **never add them**. The rounded systemd stderr message is only diagnostic. No exact controller/setup overhead is derived by subtracting a rounded total. Isolated controller CPU, GPU seconds, token usage and currency cost remain unknown in the report.

| Denominator | Worker CPU seconds | Supervisor aggregate CPU seconds | Amortized run wall seconds |
|---|---:|---:|---:|
| One unique fitted predictor |0.2178134291|0.3255882248|0.2874316370|
| One unique recipe |1.1050047134|1.6517646524|1.4581897681|

These are accounting ratios for one retained run, not extrapolated throughput or equal-quality claims. A recipe and a fitted predictor differ: the same recipe can create distinct fitted artifacts on separate folds. Unique artifact bytes do not prove behavioral uniqueness. Every failed fit remains in the inventory; failed or unavailable commands are not silently counted as zero cost.

<a id="c-scheduling--the-one-hour-ten-million-token-reference-scenario"></a>

### 11.5.3 The one-hour, ten-million-token reference scenario

GPU work, tokens, money, isolated controller CPU, downloads and installation were not measured by this proof. The supplied one-hour/ten-million-token scenario provides a 3,600-second contextual reference: this proof's run wall time is approximately 6.64% of that duration. This is not a matched-workload speedup. No scenario hardware allocation, predictor count, token trace or cost ledger exists here, so ten million tokens cannot be converted into a measured productivity or currency comparison.

The supplied one-hour/ten-million-token scenario has 3,600 seconds and 10,000,000 tokens by stipulation. This run used 239.143 seconds of wall time but has no measured token or monetary receipt. A dollar comparison, token savings claim, or one-hour quality extrapolation would require actual token usage, contemporaneous input/output/cache rates and an equivalent task/quality protocol. None is invented here.

The supplied one-hour/ten-million-token scenario is a contextual scenario only. Its workload, model, hardware, predictor inventory and token/cost traces are unavailable. This proof used approximately 6.64% of an hour for its run, but that fraction is not a matched speedup or token-efficiency result. Dollar cost and token usage remain unmeasured.


---

# Chapter 12 — Maintenance, privacy, release and preservation

<a id="c-maintenance--questions-that-should-remain-answerable-after-a-maintainer-handoff"></a>

## 12.1 Questions that should remain answerable after a maintainer handoff

Which reviewed source and runtime produced this artifact? Which exact rows and original labels were available? Was this a recipe, a checkpoint or a saved fitted predictor? Which score selected it, under what comparison identity? Which failures, missing dependencies and cancelled jobs were retained? Did any nominal test labels influence selection? Did resume preserve the original deadline? Does the export predict from saved state, and which backend actually passed? Did training perform positive updates and change parameters? Can another maintainer locate and authenticate the receipt without reading credentials or sealed data?

The roadmap ([§](#c-plan--compact-task-index)) translates the unanswered parts into named acceptance items. Unknowns stay explicit until the corresponding evidence exists.

<a id="c-maintenance--repository-authority-and-coordination"></a>

## 12.2 Repository authority and coordination

The canonical product checkout is `/home/ubuntu/JopeDime`, upstream `veox-ai/JopeDime`. `/home/ubuntu/jain-split/jope-dime` is a different research repository; `/home/ubuntu/jope-dime` may be its symlink. Similar names are not interchangeable source, checkpoint or promotion authorities.

Before source changes, inspect the open PR queue, worktrees, branch/full HEAD, dirty state and latest `AGENT_CHAT.md`. Under `flock` on `AGENT_CHAT.md.lock`, reread overlapping claims and append checkout, branch, full HEAD, owner, paths and verification. The append lock does not permit two writers in one checkout. An idle session or old claim is not a stopped-head handoff.

Keep one writer per checkout and one implementation PR queue. Preserve pending work and authorship. A resumed campaign retains its resource limits and deadline.

Repository operation remains serialized: read the current queue/ledger, claim paths under `AGENT_CHAT.md.lock`, keep one writer per checkout, and preserve pending work. Changes require exact-head CI, independent eligible review/native approval, guarded rebase merge and verification of remote `main` plus `main-guard`. A changed head invalidates an earlier head's approval. Preserve evidence before deleting closed lanes or build directories; `target/` can contain irreplaceable receipts.

<a id="c-maintenance--worktrees-and-builds"></a>

## 12.3 Worktrees and builds

Extra copies belong under `<repo>/.agent/worktrees/`, at most four lanes. `.claude/worktrees` stays a symlink into that shared root. Verify it before worktree operations and keep `.agent/` locally excluded from Git status.

Each lane reuses its own `target/`. Before Cargo:

```sh
export CARGO_INCREMENTAL=0
export CARGO_PROFILE_DEV_DEBUG=line-tables-only
export CARGO_BUILD_JOBS=4
```

Do not set `CARGO_TARGET_DIR` globally or create a new target per task. CI selects persistent PR and trusted-main storage separately. Resolve binaries from Cargo artifact messages: guessing a hashed test executable can select an old binary built with different features.

Never put source, worktrees or builds in `/tmp`. A target directory can contain evidence. Cargo markers do not prove that every descendant is disposable.

<a id="c-maintenance--coverage-and-reviewed-lifecycle"></a>

## 12.4 Coverage and reviewed lifecycle

<a id="c-maintenance--coverage-inventories-and-ci-lanes"></a>

### 12.4.1 Coverage inventories and CI lanes

Read `deploy/ci/coverage.md`, mapping and generated inventory together. Discovery counts packages, features and families; execution receipts count what one invocation actually collected and ran. Invocations overlap. Path exclusions, ignored tests and unavailable dependencies do not become passing tests.

Fast CI has eight lanes plus aggregate. Full-main diagnostics cover additional families and have recorded failures/unavailable capabilities. A fast gate does not establish all-feature, GPU or live-S3 qualification. Regenerate inventories and scores through their owning proof scripts; never hand-edit generated evidence.

<a id="c-maintenance--writer-and-reviewer-identities"></a>

### 12.4.2 Writer and reviewer identities

The established writer is `jepsontaylor`; the designated independent Claude reviewer uses `neverhuman`. Verify them for every PR. The reviewer must differ from the opener and every associated commit author and committer. Use the full paginated inventory and record unassociated commits.

```sh
/home/ubuntu/.local/bin/gh-role writer api user --jq .login
/home/ubuntu/.local/bin/gh-role eligible reviewer veox-ai/JopeDime PR_NUMBER
```

The helper prevents identity mistakes but cannot prove independence. Changing credentials does not turn a writer into a reviewer. Never print tokens or credential-bearing environments.

<a id="c-maintenance--freeze-review-and-merge"></a>

### 12.4.3 Freeze, review and merge

Freeze the full head; dispatch fresh manual CI; retain source SHA, run ID, attempt and dispatch time. Verify current-attempt jobs/aggregate and nonzero meaningful coverage. Supply the independent reviewer with diff, acceptance commands and evidence. A native approval must explicitly bind the final head and remain undismissed. Changed source repeats checks and review.

```sh
/home/ubuntu/.local/bin/gh-role writer pr merge PR_NUMBER \
  --rebase --match-head-commit FULL_REVIEWED_SHA
```

Immediately recheck head, eligibility, CI and approval before merging. Afterwards verify `merged_at`, fetch remote main, compare the reviewed tree and verify main-guard. Preserve any ledger tail or other dirty work before restoring clean verified main. Protection enforcement has been procedural; a protection API 403 does not prove a preventive branch lock. Main-guard detects unreviewed pushes after they occur.

<a id="c-maintenance--main-guard-and-full-main-scheduling"></a>

### 12.4.4 Main-guard and full-main scheduling

Main-guard and the full-main diagnostic share the single xbabe2 `jope-full-cpu` slot and run serially. A main push queues both; full-main has a 120-minute timeout, so main-guard can wait behind it. Read back job/run states and retain the exact source/run/attempt before changing the queue. In the PR83 lifecycle the owner-authorized full-main diagnostic was cancelled while still queued, before any job started, and the owned full timer was restored to allow main-guard to run. That is a recorded unexecuted diagnostic, not passing coverage or general permission to cancel running work. Follow the current allocation owner and runner runbook; never stop another running job to accelerate a merge. Sources: `.github/workflows/main-guard.yml:17`, `.github/workflows/full-main.yml:24`, `docs/runbooks/runner.md:15`, `:131`; the PR83 lifecycle cancellation receipt.

<a id="c-maintenance--privacy-boundaries-the-private-manual-and-public-fitted-bundles"></a>

## 12.5 Privacy boundaries: the private manual and public fitted bundles

This is a maintainer document for the private product repository. Its explicitly requested GP examples and internal operational paths explain the implementation; they do not expand the allowlist for public inference bundles.

This is a maintainer document containing internal model and discovery design. The user's request authorizes this explanation and the companion private GP examples. It does not enlarge the allowlist for distributed inference artifacts or authorize publishing private weights, discovery histories, vocabulary assets or training corpora. Hashes identify dependencies; they are not the dependencies themselves.

Allowlisted public fitted bundles contain inference code, necessary fitted input dictionaries, class/schema metadata and their hashes. They exclude the private recipe, discovery history and scoring references. The private prepared plan and worker directory remain private. Canary/text-policy checks supplement the explicit file allowlist; passing static checks does not replace actual complete-prediction parity.

<a id="c-maintenance--release-and-installation-are-independent-from-model-promotion"></a>

## 12.6 Release and installation are independent from model promotion

The current tag workflow runs for `v*`, checks tag = workspace = Python version, builds a Linux binary, writes build provenance and checksums, builds the Python wheel, and records model/dataset references by hash. The committed build command selects default CLI features; it does **not** include `--features asm`. Therefore the existence of this workflow does not prove that a released binary can execute the private ASM path. No newly published release is asserted here. The current private source build in [§](#c-installation--development-build-of-the-cli) explicitly selects `asm`.

[Release workflow](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/.github/workflows/release.yml#L71) and [version consistency](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/deploy/ci/version-consistency.sh#L1) are source evidence. A future reviewed version bump can use `just bump X.Y.Z`; that recipe edits manifests and the lockfile and invokes Cargo. It is a developer mutation recipe, not a command executed for this manual or a decision to release any particular version. [Exact bump implementation](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/justfile#L63).

The checksum-verifying private binary installer and CLI-only source-build helper remain product28 follow-ons. Their concrete commands and cold/warm separation are in [§](#c-installation--build-and-install). Installing a new CLI does not admit new data, train a model, or promote a checkpoint. Likewise, `models pull` or a storage manifest does not change the accepted policy pins.

<a id="c-maintenance--preservation-and-lane-closure"></a>

## 12.7 Preservation and lane closure

Before deletion inspect `git status`, `git stash list`, `git log --all --not --remotes`, and local versus remote tags. A branch-only log misses commits elsewhere.

Preserve unique source/evidence for the tree being removed, restore independently into the reusable bare verification directory without alternates, and verify the exact commit, tree and connectivity. A valid bundle checksum does not prove the desired detached HEAD was included.

After preservation and a fresh consumer check, remove a closed lane with `git worktree remove`. Indefinite redundant copies are not an alternative to the required reclamation step. Unknown contents remain protected. Follow `deploy/maintenance/README.md`; its reporting timer proposes candidates and never authorizes deletion. Backup, replication and retention operations remain separately authorized: closing a lane does not authorize them, and neither does this manual.

<a id="c-maintenance--rebuilding-and-updating-this-manual"></a>

## 12.8 Rebuilding and updating this manual

Rebuilding the book is itself a runbook. Its summary:

| Runbook part | Rebuilding this manual |
|---|---|
| Purpose and status | Render both volumes from the committed Markdown, figures and data. **available**: the committed `render-receipt.json` records both edition-2 volumes, their input and output hashes and warnings; the receipt is written only after every renderer gate passes, and `publication.json` declares the build timestamp. The original PR87 build passed every renderer gate with no warnings. Each build is identified by its own `render-receipt.json`. |
| Prerequisites and resources | Python 3.12 with the direct versions in `tools/requirements-docs.txt`, Pango/Fontconfig, DejaVu fonts and Poppler; local CPU only. The build opens no network resource, model, dataset or sealed label. |
| Manifest-creation command | None. `book.json`, `publication.json` and `aliases.json` are reviewed inputs edited directly; no committed command generates them. |
| Execution | The figure scripts when their data changed, then the renderer, then `tools/verify_publication.py` ([§](#c-maintenance--rebuilding-the-book), [§](#c-maintenance--regenerating-the-training-and-target-figures), [§](#c-maintenance--the-publication-record-and-its-three-identities)). |
| Artifacts and success | Two PDFs under `docs/`, `manual.html`, `MANUAL.md`, `structure.json` and `render-receipt.json`. Success is a zero exit from the renderer with every gate passed and a zero exit from `tools/verify_publication.py`, followed by inspection of the rendered pages and extracted text. |
| Failure and resume | A failed gate exits nonzero and can leave a new PDF beside the previous build's `render-receipt.json`, which the renderer does not remove and which no longer describes it; such a PDF is not a build, and `verify_publication.py` rejects it as stale. Fix the input and rerun from the start: there is no partial resume. |
| Limitations | PDF bytes are not reproducible bit for bit; the renderer does not check that a figure is newer than its data; it runs no model. |

<a id="c-maintenance--rebuilding-the-book"></a>

### 12.8.1 Rebuilding the book

The 2026-09-23 builds used Python 3.12 and the observed direct versions in `tools/requirements-docs.txt`. That file is not a transitive lock or a verified cold installer. A fresh environment also needs working Pango/Fontconfig, DejaVu fonts, and Poppler's `pdfinfo`, `pdftotext` and `pdftoppm`. The render receipt records exact Python/library versions and font-file hashes. The following environment-creation recipe is provided for rebuilding; it was not executed as a fresh installation during this document audit:

```sh
MANUAL_ENV=/home/ubuntu/.venvs/jope-manual
python3 -m venv "$MANUAL_ENV"
"$MANUAL_ENV/bin/python" -m pip install \
  -r docs/internal/asm-manual/tools/requirements-docs.txt
. "$MANUAL_ENV/bin/activate"
```

From the repository root, render the audited retained inputs with these local-only commands:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 docs/internal/asm-manual/tools/make_inventory_figures.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 docs/internal/asm-manual/tools/make_figures.py
python3 docs/internal/asm-manual/tools/build_manual.py
pdfinfo docs/asm-maintainer-manual.pdf
pdfinfo docs/asm-evidence-reference.pdf
pdftotext -layout docs/asm-maintainer-manual.pdf /path/to/durable/review/volume-1.txt
pdftotext -layout docs/asm-evidence-reference.pdf /path/to/durable/review/volume-2.txt
```

The two figure scripts above do not regenerate the training and target figures. A complete rebuild also runs the four scripts of [§](#c-maintenance--regenerating-the-training-and-target-figures) before `build_manual.py`, because the renderer does not check that a figure is newer than its data. `make_figures.py` still defines a fixed host path to the ASM custody root (`tools/make_figures.py:39-41`), but never uses it.

Rendering uses Python Markdown, BeautifulSoup and WeasyPrint with system DejaVu fonts; their observed versions and font hashes are retained in the render receipt. Plotting uses Matplotlib and NumPy. The renderer rejects missing/nonlocal images and failed resource fetches, and checks that inputs did not change during rendering. It does not run models, fit predictors, fetch datasets or open sealed labels. Inspect the resulting pages and extracted text before treating a successful render as a complete document.

Byte-identical PDF rebuilding is not promised. `SOURCE_DATE_EPOCH` fixes font-subset timestamps, but the observed WeasyPrint 68.1 tagged-table identifiers contain process-specific values. Preserved first-edition repeat renders had identical normalized logical objects and decoded non-container streams after accounting for those identifiers, while their PDF hashes differed. Keep each output with its own render receipt and verify input hashes, content and layout; do not treat a changed PDF hash alone as changed measurements. This observation concerns the renderer, not numerical predictor reproducibility.

<a id="c-maintenance--what-the-renderer-reads-checks-and-writes"></a>

### 12.8.2 What the renderer reads, checks and writes

`tools/build_manual.py` takes the layout from `book.json` (schema `asm-manual.book/v1`):

- **Chapters and appendices.** `chapters` is a flat list of section files in reading order, numbered from 1 when rendered. `appendices` lists files whose letters must run A, B, C and onwards. The renderer refuses a file listed twice, a file in `sections/` that `book.json` omits, and a listed file that does not exist.
- **Volumes.** An appendix with a `volume` key is published in that volume's PDF. Edition 2 publishes chapters 1 to 13 as Volume 1, `docs/asm-maintainer-manual.pdf`, and appendices A to E as Volume 2, `docs/asm-evidence-reference.pdf`. After chapter 13, Volume 1 carries a signpost that keeps each appendix's letter, title and sections and links into Volume 2. It is followed by a legacy destination index, which gives every edition-1 destination now in Volume 2 a named landing linked to its new heading. Volume 2 has its own title page and two-level contents.
- **Position-independent anchors.** Section files are named by topic, not by position. A heading's anchor is `c-<file stem>--<GitHub heading slug>`, and a chapter title's is `c-<file stem>`, so moving a chapter does not change its anchors. Write a cross-reference as a Markdown link whose target is `file.md#slug`, or `#slug` within one file. The renderer fills a link whose text is exactly `§` with the target's number and title, and refuses any internal link it cannot resolve.
- **Aliases.** The required `aliases.json` maps all 266 published older destinations, such as edition 1's `chapter-NN` and `chapter-NN-M` anchors, to current headings. Each alias becomes an additional anchor on its heading. Destinations moved to Volume 2 also retain a named landing in the original Volume-1 PDF, with an explicit link to the exact semantic target. An alias that names no heading, or that repeats an existing heading anchor, is refused.
- **Strict heading check.** Each file must begin with exactly one `#` title. Headings go at most four levels deep and may not skip a level. Headings under one parent need distinct titles; a heading followed directly by a heading of the same or a shallower level, or by nothing, is refused as empty; and a heading may not have exactly one subdivision. The chapter title counts as a parent, so a file with exactly one section is refused; a file with no sections at all is only a warning. Headings are numbered by the renderer, never by hand, and heading-like lines inside fenced code are not treated as headings.
- **Size gate.** Each PDF must be at most 5 MiB (5,242,880 bytes), the repository's per-file cap. A larger volume stops the build.
- **Pinned inputs.** Every Markdown, JSON, JSON Lines, CSV, Python, CSS, PNG, SVG, property-list, gzip and text file under `docs/internal/asm-manual/`, except the generated outputs, is hashed together with `docs/asm-maintainer-status.plist` and the landing page `docs/asm-maintainer-manual.md`.
- **Outputs.** Besides the PDFs: `manual.html` (Volume 1), the combined `MANUAL.md`, `structure.json` with every numbered heading, figure and alias, and `render-receipt.json` (schema `jope.manual-render/v2`). The receipt's `volumes` list gives each PDF's name, page count, SHA-256 and bytes; its top-level `pages`, `pdf_sha256` and `pdf_bytes` describe Volume 1 only. It also records heading counts per level, the alias count, warnings, the Python, library and font identities and every input hash, and it is written only after every gate has passed.

<a id="c-maintenance--regenerating-the-training-and-target-figures"></a>

### 12.8.3 Regenerating the training and target figures

Regenerate the training and target figures before the PDF after changing an audited source CSV:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 docs/internal/asm-manual/tools/make_training_figures.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 docs/internal/asm-manual/tools/make_target_figures.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 docs/internal/asm-manual/tools/make_legacy_training_figure.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 docs/internal/asm-manual/tools/make_per_step_figure.py
python3 docs/internal/asm-manual/tools/build_manual.py
```

These commands render retained observations; they do not train a model or reconstruct unavailable metrics. The original-history verifier needs the separately preserved source files named in each CSV. Plot rebuilding uses the committed CSVs. No missing source or changed hash may be converted into a successful historical verification.

<a id="c-maintenance--the-publication-record-and-its-three-identities"></a>

### 12.8.4 The publication record and its three identities

Three identities describe a copy of this book. They change independently, and none implies the others:

| Identity | What it names | Rule and edition-2 value |
|---|---|---|
| Implementation snapshot | The reviewed application source that the book documents | Reviewed `68f56404c3e199e06413bfd645d3fe0803d7b96f`, merged 2026-09-23T16:34:47Z as main `a75e543d2166017cb0139e0a251d51d9a0c98e08`, tree `176b75bc36d76c7ce4eaeb7234677ec013468889` ([§](#c-status--implementation-snapshot-and-this-manuals-own-identity)). The publication manifest and covers pin this same application snapshot. Later documentation and implementation deliveries have separate dated observations. |
| Qualification observations | Each measurement the book reports | Each keeps its own observation time and receipt, such as the fourteen-command example run of 23 September 2026 ([§](#c-inference--measured-execution-of-the-private-examples)). Rebuilding the book repeats none of them. |
| Document build | The edition, its pinned inputs and its PDFs | Recorded by the build itself: the input hashes and each volume's PDF hash, bytes and pages in `render-receipt.json`. |

The renderer requires `docs/internal/asm-manual/publication.json` and prints it in the front matter as the publication record, with one subsection per identity. It validates the schema, UTC timestamps, implementation identity format and both PDF filenames, and refuses undeclared PDFs under `docs/`. The snapshot subsection gives source, main, merge time, tree and an optional note. The observations form a table of observation, observed UTC time and receipt. The document-build subsection states the edition and its date, `SOURCE_DATE_EPOCH`, the number of pinned inputs and a SHA-256 digest of their path and hash list. PDF hashes stay in `render-receipt.json`, because a PDF cannot contain its own hash. `publication.json` is a reviewed input edited directly and is itself pinned: the renderer derives none of it from Git and refuses to build without it. The declared `build_utc` sets `SOURCE_DATE_EPOCH`. The cover's edition line (edition, date and the manifest's implementation main) derives from the manifest, and its count of primary acceptance items derives from `data/roadmap.json`; the renderer computes no per-status counts. Run `python3 docs/internal/asm-manual/tools/verify_publication.py` after rendering to reject stale or missing input/output hashes, size violations and inconsistent task priorities or statuses across Markdown, JSON and plist.

<a id="c-maintenance--updating-the-book"></a>

### 12.8.5 Updating the book

Refresh inventories and receipts, then section Markdown, figures, checklist, `publication.json` and both PDFs. Keep a rendering input manifest. A checkpoint does not update performance figures by implication; merged training source does not prove actual learning. Historical Elo remains historical after regeneration.

A manual change lands through the same reviewed lifecycle as source ([§](#c-maintenance--freeze-review-and-merge)): fresh exact-head CI and an independent non-contributor native review of the final head. Every committed file, including the rendered PDF, must stay within the repository's 5 MiB (5,242,880-byte) file cap (`deploy/ci/check.sh:46`), and because the rebase merge lands every commit of the branch, the cap applies to intermediate commits as well. The earlier publication, PR84 (published head `e06c0551`), was closed as superseded for that reason: the independent non-contributor review 5295763455, a COMMENTED review at `e06c0551`, found two files above the cap in three ancestor commits. Its clean-history successor, commit `2022017a8` on `docs/asm-maintainer-manual-final`, is a single commit on top of main `a75e543d2166…` that carries a 203-page PDF (`69418c48…`). It was published as PR85 and merged at 2026-09-23T22:33:36Z as main `bdfc95c63d905c647e41175e79dc30cdc8600287`, after independent native approval 5297559032 at that exact head and full CI run 35927921851; the following main-guard run 35929010658 passed. This second edition restructures and corrects that published edition. The original authoring commits `7b9398b76` and `bda953f62` are preserved; their changes and the unfinished content pass were integrated on `docs/asm-manual-edition2-integration`, retaining the separately merged PR86 corrections. Edition 2 merged through PR87 at 2026-09-24T02:58:00Z as main `02c0bef6c81ab17e034daea308c1a0b5a7940467`, after exact-head CI 35948653955 attempt 1 and independent native approval 5299139743; main-guard 35949456287 passed. Later revisions need their own exact-head CI and independent native approval; local rendering alone does not establish publication.

Before delivery check local links, source identities, totals, metric directions, captions, capability status and rendered pages. Extract PDF text and inspect dense tables, examples and figures. Preserve unavailable statistics and measurements explicitly rather than filling them with zero.


---

# Chapter 13 — Prioritized implementation plan and extension interfaces

<a id="c-plan--prioritized-implementation-plan"></a>

## 13.1 Prioritized implementation plan

<a id="c-plan--purpose-and-scope"></a>

### 13.1.1 Purpose and scope

This checklist is the requested property-list of remaining work. `roadmap.json` is its machine-readable source, with stable IDs, priorities, evidence levels, dependencies, commands, required receipts, budgets and source references. It describes source 68f56404 and the stopped training64, product28, warm-target and winner-export follow-on patches ([§](#c-catalogue--names-and-glossary)). PR83 is merged: remote main `a75e543d2166017cb0139e0a251d51d9a0c98e08` has the reviewed source tree, and main-guard 35889731910 attempt 1 succeeded. That implementation snapshot is unchanged as of 2026-09-23 about 23:55 UTC. Edition 1 of this manual merged later, through PR85 at 22:33:36 UTC, as main `bdfc95c63d905c647e41175e79dc30cdc8600287`, after independent native approval 5297559032 and exact-head CI 35927921851; main-guard 35929010658 passed. That merge changed only documentation files (under `docs/`, plus `README.md` and `AGENT_CHAT.md`). The documentation-correction PR86 (head `0079ceddbaba739d736e80f6fe9b999955c4ad3a`, tree `01ecd51afc77e753ee4fe1f3fbe3c130803bbf43`) merged at 2026-09-24T00:04:12Z as main `2b375ef11fc9298ffa67d4d03e0bb83053201802`; main-guard 35936673729 passed. Edition 2 merged through PR87 at 2026-09-24T02:58:00Z as main `02c0bef6c81ab17e034daea308c1a0b5a7940467`, after exact-head CI 35948653955 attempt 1 and independent native approval 5299139743; main-guard 35949456287 passed; revisions repeat that lifecycle ([§](#c-status--implementation-snapshot-and-this-manuals-own-identity)). Clean canonical main was verified before the documentation branch. LIFE-01 records the completed readback; later source work still needs its own lifecycle.

P0 blocks the specified downstream proof or integration; it does not mean every item is a newly discovered defect. P1 is required for the full requested demonstration or manual. P2 is explicit subsequent work. Commands are unexecuted acceptance templates unless their named evidence says otherwise. Variables must resolve to actual final-source artifacts and emitted identities. An empty command list records a missing truthful interface rather than inventing one.

There are 43 primary items. Five of them (DOC-05, DOC-06, PACK-01, FOCUS-01 and LOOP-01) respond to the user’s training-history, continuation, flywheel, focused-search and packing questions. At the owner’s direction for edition 2, LOOP-01 is the central regression product milestone, and this roadmap therefore lists it as P1 ([§](#c-plan--critical-order)). Documentation work does not wait for implementation; implementation and numerical follow-ons preserve the original budgets and frozen protocol.

<a id="c-plan--critical-order"></a>

### 13.1.2 Critical order

Verified reviewed main → guarded source integration → actual tests, host qualification and final packaging → fresh qualified contexts and preregistered real teachers → all-eligible family-disjoint corpus → one GPU-hour training with continuation → explicit new-checkpoint consumption → real fitted warm demonstration → unchanged thirty-dataset campaign, selected-state backend qualification inside its remaining window, and report → eligible motif reuse.

**LOOP-01 is the central regression product milestone** (P1, not yet qualified). Its chain is: new eligible target → actual exposure → changed parameters → compatible reload → improved fitted search → another new generation. It builds on the order above: an eligible corpus (TEACH-04), a verified continuation and export (TRAIN-04) and live consumption of the newly trained checkpoint (NEURAL-03). Two passes through one corpus are continuation, not two generations. The accepted 512×6 step-60000 model stays separate from experimental regression lineage v1 (256×2, new initialization) ([§](#c-training--what-the-pending-campaign-trains)). LOOP-01’s two controlled rounds may remain unfunded or inconclusive within the existing 20-minute regression GPU cap; no new allocation is implied. The comparison of sibling fresh and replay-only branches (25/75 fresh/replay versus replay-only, at equal committed updates) and the preregistered decision rule are proposals, not achieved results ([§](#c-learning)).

Library parity fixtures precede the full panel. Actual selected-winner exports require complete confirmations and must still fit the original execution window. They cannot consume its reporting reserve or silently run after expiry. Documentation can proceed alongside source preparation; actual numerical examples need an allocated execution window.

Do not spend the GPU hour until the regression consumer and its actual fixture tests are ready. Do not freeze the final official panel configuration until its checkpoint is chosen and consumed. Separate proof allowances suggested in older plans are proposals, not new budget authorization.

<a id="c-plan--compact-task-index"></a>

### 13.1.3 Compact task index

| ID | Priority | Stage | Workstream | Current state | Dependencies |
|---|---|---:|---|---|---|
| LIFE-01 | P0 | 0 | coordination | complete with retained readback | None |
| DOC-01 | P1 | 0 | manual | canonical private manual assembled | LIFE-01 |
| DOC-02 | P1 | 0 | manual examples | complete at named recipe/refit scope | DOC-01 |
| DOC-03 | P1 | 0 | manual figures | figures rendered; local PDF inspection completed; publication pending | DOC-01 |
| DOC-04 | P1 | 0 | manual quality | rendered; final publication gates pending | DOC-01, DOC-02, DOC-03 |
| INT-01 | P0 | 0 | serial integration | stopped unmerged packets | LIFE-01 |
| INT-02 | P0 | 0 | integration qualification | pending | INT-01 |
| BENCH-01 | P1 | 1 | frozen benchmark foundation | implemented; retain during integration | INT-02 |
| BENCH-02 | P1 | 1 | external comparators | refresh at final environment | INT-02 |
| EXEC-01 | P1 | 2 | bounded workers | implemented; CI coverage gap retained | INT-02 |
| EXEC-02 | P1 | 2 | fault regression tests | test gap: drain-and-fixtures-F2 | INT-01 |
| EXEC-03 | P1 | 2 | receipt replay regression tests | test gap: resume-fix-F2 | INT-01 |
| EXEC-04 | P1 | 2 | cutoff semantics regression | test gap: regressions-F2 | INT-01 |
| EXEC-05 | P1 | 2 | accounting and unknown outcomes | pending execution | INT-02 |
| NEURAL-01 | P0 | 3 | context qualification | requires final-source binding | INT-02 |
| TEACH-01 | P0 | 6 | HWM comparison identity | not runtime qualified | INT-02 |
| TEACH-02 | P0 | 6 | label exposure readers | residual eligibility-B6 | TEACH-01 |
| TEACH-03 | P0 | 3 | real teacher collection | blocked on fresh admitted evidence | BENCH-01, BENCH-02, NEURAL-01, TEACH-01, TEACH-02 |
| TEACH-04 | P0 | 3 | real corpus materialization | blocked on two real families | TEACH-03 |
| TRAIN-01 | P0 | 3 | actual trainer tests | residual training-F3 | INT-02 |
| TRAIN-02 | P0 | 3 | native supervised training | host proof pending | TRAIN-01 |
| TRAIN-03 | P0 | 3 | bounded real GPU training | blocked on admitted data and consumption readiness | TEACH-04, TRAIN-02, NEURAL-02 |
| TRAIN-04 | P0 | 3 | checkpoint continuation and export | requires actual campaign | TRAIN-03 |
| NEURAL-02 | P0 | 3 | trained-checkpoint consumer readiness | compile/runtime fixture pending | INT-02, TRAIN-01 |
| NEURAL-03 | P0 | 3 | live newly trained consumption | newly trained consumption pending | TRAIN-04, NEURAL-01 |
| INFER-01 | P0 | 4 | fitted state and privacy | qualification pending | INT-02 |
| INFER-02 | P1 | 4 | exact selected-winner portability | actual export/parity unrun | INFER-01, PANEL-01 |
| INSTALL-01 | P1 | 5 | private installation | cold/live path pending | INT-02 |
| INSTALL-02 | P1 | 5 | final input/runtime preparation | pending final source | INSTALL-01, BENCH-01, BENCH-02 |
| DEMO-01 | P1 | 5 | measured warm fitted demonstration | real trained warm proof pending | NEURAL-03, INSTALL-02 |
| PANEL-01 | P1 | 5 | unchanged thirty-dataset campaign | pending final trained configuration | BENCH-01, BENCH-02, NEURAL-03, DEMO-01, INFER-01, INSTALL-02 |
| REPORT-01 | P1 | 5 | complete measured report | full campaign report pending | PANEL-01, EXEC-05 |
| MOTIF-01 | P1 | 6 | eligible reusable fragments | mining front door incomplete | TEACH-01, TEACH-02, TEACH-03 |
| MOTIF-02 | P1 | 6 | reuse validation | live reuse pending | MOTIF-01 |
| MOTIF-03 | P2 | 6 | new operators | future conditional work | MOTIF-02 |
| IP-01 | P1 | 0 | public privacy boundaries | residual D-IP4 and future PR text work | INT-01 |
| MAINT-01 | P2 | 0 | broad diagnostic qualification | not closed by fast CI | LIFE-01 |
| FUTURE-01 | P2 | 4 | CUDA inference and serving promotion | deferred | INFER-02, REPORT-01 |
| DOC-05 | P1 | 0 | real training and target history | Original training and target values verified; final composed-document inspection required | DOC-01 |
| DOC-06 | P1 | 0 | versions, KPIs and continuation runbook | Current and pending interfaces source-audited; independent chapter review complete | DOC-01 |
| PACK-01 | P2 | 3 | packed training consumer and incremental loading | Legacy binary store exists; pending Python category route still materializes JSONL | TEACH-04, TRAIN-01 |
| FOCUS-01 | P2 | 6 | sustained single-dataset improvement | Current bounded proof exists; no incumbent-seeded attack front door | TEACH-01, TEACH-02, TEACH-03 |
| LOOP-01 | P1 | 3 | central regression product milestone: repeatable measured learning | New complete improved-target-to-improved-search cycle not yet qualified | NEURAL-03, TEACH-04, TRAIN-04 |

<a id="c-plan--proposed-owner-decisions-and-external-dependencies"></a>

### 13.1.4 Proposed owner decisions and external dependencies

The property list is self-contained: every proposed dependency resolves to a primary item, a coordinated proposal, or the registry below. The original contributor proposal is preserved as `data/coordinated-roadmap-proposal.json`, SHA256 `334ae97446ef29963bce47582f1550c7d369f0b329ef143c128685991422ffd4`. Its recommendations are contributor suggestions, not fresh user instructions. Already explicit user decisions require no reconfirmation.

| Reference | Meaning | Current disposition |
| --- | --- | --- |
| OD-1 | Confirm the name ASM in writing (reported by Codex 16:07Z) | ASM naming is already explicit user direction; no reconfirmation needed. Any protocol rename remains a separately versioned change. |
| OD-2 | Confirm HWM train-side OOF admission and the test-to-development relabel with burn receipts (audit D3) | Train-side OOF admission and exposed-test-as-development treatment are already explicit user direction; no reconfirmation needed. |
| OD-3 | D-IP1: ban 'tree' in emitted code | Unresolved contributor proposal; not a newly imposed approval prerequisite and not execution authorization. |
| OD-4 | D-IP2: extra_trees naming | Unresolved contributor proposal; not a newly imposed approval prerequisite and not execution authorization. |
| OD-5 | D-IP3: publication boundary | This private manual is explicitly requested and authorized. Broader public-release boundary changes remain separate; this item does not block the manual. |
| OD-6 | D-IP4: PR 83 identifiers | Unresolved contributor proposal; not a newly imposed approval prerequisite and not execution authorization. |
| OD-7 | B6 fit fence and data rows | Unresolved contributor proposal; not a newly imposed approval prerequisite and not execution authorization. |
| OD-8 | Teachers from panel families 230_machine_cpu and 712_chscase_geyser1 | Prospective small families already selected from provenance; exposure labels require actual evidence. Unknown must not become known-seen solely from this recommendation. |
| OD-9 | W2 scale-out CPU (about 4 h) | Proposed additional compute outside the current allocation; no new hours or deadline are granted. |
| OD-10 | 3,755 unknown-licence catalog datasets | Unresolved data-use proposal. Unknown license metadata does not establish permission for any proposed use. |
| OD-11 | Lineage-B corpora | Unresolved contributor proposal; not a newly imposed approval prerequisite and not execution authorization. |
| OD-12 | Multiclass data (the catalog has none) | Unresolved contributor proposal; not a newly imposed approval prerequisite and not execution authorization. |
| OD-13 | Sealed one-shot evaluation | Outside current execution scope: sealed evaluation remains untouched. |
| OD-14 | Cost scenario constants | Unresolved contributor proposal; not a newly imposed approval prerequisite and not execution authorization. |
| OD-15 | CUDA timing | Unresolved contributor proposal; not a newly imposed approval prerequisite and not execution authorization. |
| OD-16 | Second GPU hour | Proposed additional compute outside the current allocation; no new hours or deadline are granted. |
| OD-17 | The proposed 300 s proof windows | Per-proof resource files must identify actual admission and remaining allowance; an example duration does not create a new budget. |
| product28 merged | Product28 stopped overlay has completed its own reviewed merge; part of INT-01, currently pending. | External prerequisite, explicitly unresolved unless a separate retained receipt closes it. |
| product28 on main | Same product28 reviewed-main prerequisite; currently pending. | External prerequisite, explicitly unresolved unless a separate retained receipt closes it. |
| training64 merged (eligibility API) | Training64 eligibility interface has completed its own reviewed merge; part of INT-01, currently pending. | External prerequisite, explicitly unresolved unless a separate retained receipt closes it. |
| freeze passed | Final implementation/runtime configuration and required acceptance are frozen; documentation freeze alone does not satisfy this. | External prerequisite, explicitly unresolved unless a separate retained receipt closes it. |
| compute window | A concrete finite resource allocation is admitted; no new allowance is implied. | External prerequisite, explicitly unresolved unless a separate retained receipt closes it. |
| DATA-01 for part b | The metadata inventory proposal DATA-01 is complete for the catalog-generation part of DOCFIX-01. | External prerequisite, explicitly unresolved unless a separate retained receipt closes it. |

The dispositions above are the roadmap data's own and are not edited here. Their state between the PR86 merge (2026-09-24T00:04:12Z) and the PR88 merge (2026-09-24T05:11:50Z):

- training64 and product28 remain stopped patches outside `main`, so the rows “product28 merged”, “product28 on main” and “training64 merged (eligibility API)” stay open.
- OD-8: both proposed teacher families, `230_machine_cpu` and `712_chscase_geyser1`, are regression datasets of the frozen panel. If they are used to manufacture teachers, later evaluations on them must be labelled exposed/adaptive ([§](#c-learning--freezing-the-experiment-and-choosing-families)).
- DATA-01: Claude's metadata-only packet was released at 19:30:23 UTC (manifest `SHA256SUMS`, `4bebee5a…`, 148 entries), but its sealed-name gate failed on one raw file. A later clarification (`RELEASE-CLARIFICATION.md`, `380f3402…`, with manifest `SHA256SUMS.release-2`, `a4b0c436…`) limits the release to eight aggregate files and keeps `raw/`, `scratch/` and the full HANDOFF in custody. Codex imported only an aggregate subset (stopped subset `558548be…`): the summary `e4dbe075…` and the release pointer `366fa000…`, present here as `data/data-estate-extension-summary.json` and `data/data-estate-extension-release.json`. No retained receipt closes “DATA-01 for part b”. The DOCFIX-01 correction landed through the separate documentation PR86 (head `0079ceddb`), merged as `2b375ef11` at 2026-09-24T00:04:12Z.

<a id="c-plan--stopping-and-completion-rules"></a>

### 13.1.5 Stopping and completion rules

An unsuccessful prerequisite remains a recorded refusal or failure. Do not substitute synthetic real-data teachers, another winner, easier datasets or a new deadline. Source completion does not close a runtime task. A plan, help output or checkpoint file is not the corresponding live receipt.

Keep four claims distinct: an eligible teacher, a new discovery, information actually consumed in training, and an improved policy. Evidence for one does not establish another, and a hash alone does not establish eligibility.

For LOOP-01 the owner’s stop rules are proposed: stop on no eligible discoveries, an incompatibility, no material gain, insufficient evidence or an exhausted budget; preserve the best checkpoint; never promote serving automatically. If its rounds are unfunded or the data insufficient, report the mechanical proof and descriptive results, and record the causal verdict as “inconclusive”, not as success. Gain over the parent checkpoint is a separate claim from advantage over generic search or the product; generic search and the accepted step-60000 model remain report baselines ([§](#c-learning)).

Full delivery requires all thirty outcomes, losses and gaps, backend limitations, measured unique-predictor/recipe costs, honest training category scope and actual newly trained consumption. Ten wins per task and an under-sixty-second warm result remain declared goals, not promises. No win-counting rule is ratified: [§](#c-catalogue--names-and-glossary) gives the proposed PANEL-WINS definition, pending owner ratification, and [§](#c-learning--win-counting-and-a-proposed-acceptance-record) the separate win columns. The one-hour/ten-million-token comparison must retain unknown hardware, workload, token and currency quantities.

The document gate requires one readable PDF plus matching section Markdown, reproducible figure inputs, source/lifecycle references and this maintained checklist. The maintainer-private examples are authorized documentation; their existence does not expand public inference bundle contents.

<a id="c-plan--extension-interfaces"></a>

## 13.2 Extension interfaces

<a id="c-plan--the-continuation-interfaces-that-exist-today"></a>

### 13.2.1 The continuation interfaces that exist today

The training chapter is the authoritative continuation runbook: its per-mode procedures, variables and limits govern ([§](#c-training)). This table is only an at-a-glance reference. Its status labels repeat that chapter's: available, implemented but unqualified, pending integration and proposed. An interface that is present at main `a75e543d2` but has no execution receipt is implemented but unqualified, not available, and no label by itself qualifies a campaign.

| Interface | Status | Current behavior | Appropriate interpretation |
|---|---|---|---|
| `jope train-step --local` | implemented but unqualified | Fits a centroid router from local labels | Small router training; not the deep policy model. |
| `jope learn ...` | proposed | Not exposed by the umbrella parser at `68f56404` (main `a75e543d2`) | ADR operating intent; do not present as a working command. |
| `jope-parallel train ...` with `neural` | implemented but unqualified (dispatch held) | Native regression trainer syntax exists; direct numerical dispatch is held | Retained native continuation machinery, not presently usable through this direct entry point. |
| `jope-py train-policy --init-from ...` | implemented but unqualified | Python policy weight warm start; CLI exposes no `--resume-from` | A new optimization run initialized from compatible weights, not exact optimizer continuation. |
| `train_policy_v3_from_jsonl(..., resume_from=..., resume_checkpoint=...)` | implemented but unqualified (Python API) | Existing Python API restores model, optimizer, scheduler, scaler and RNG subject to compatibility | A real continuation API, requiring a trusted matching Python resume artifact; not an accepted-native-checkpoint conversion. |
| Pending `jope train plan/run/resume/verify/export` | pending integration | Supervised, task-routed, bounded training proof and experimental export | Preferred integration target for the approved one-GPU-hour demonstration; still unintegrated and unexecuted. |

**Today no executable route produces a successor of the accepted step-60000 serving model.** Exact native `--resume` of step 60,000 is refused: its `state.json` has no `execution_partition`, and resume requires the original partition (`crates/jope-engine/src/training.rs:1106`). Every native `jope-parallel train` dispatch, including `--initialize` with a carried optimizer, stops at the dispatch hold ([§](#c-training--the-dispatch-hold-resource-incompatibility-and-valid-topologies)); a `parallel.checkpoint_continuation.v1` carry must in addition reproduce the parent run's recorded world/microbatch partition (`crates/jope-engine/src/checkpoint_recovery.rs:191`), and the accepted lineage ran at world 8. `jope-py train-policy --init-from` loads a Python policy payload and copies only shape-compatible tensors (`python/src/jope/policy/train_v3.py:1635`, `:1702`); it is not a loader for the native 512-wide, six-layer model ([§](#c-architecture--checkpoint-binding-and-configuration)). The Python resume API accepts only a `PrimePolicyTrainerResume.v1` artifact (`train_v3.py:1457`), a schema the native trainer does not write. The pending training64 campaign trains a new, smaller Python model family, experimental regression lineage v1 (256×2, new initialization), not a continuation ([§](#c-training--what-the-pending-campaign-trains)). Under the owner’s naming rule, its initial model (M_init) is named M0 only after it has actually trained on the first corpus (C0). The RTX step 60,100 run is a historical hardware-qualification continuation, and the later A100 step 69,000 remains unaccepted; neither is a serving successor.

Sources: [current umbrella commands](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-cli/src/main.rs#L26), [local train-step dispatch](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-cli/src/main.rs#L308), [Python CLI flags](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/python/src/jope/cli.py#L186), [Python API](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/python/src/jope/policy/train_v3.py#L1331).

<a id="c-plan--code-to-extend"></a>

### 13.2.2 Code to extend

These are the narrow extension points at reviewed source `68f56404`, whose tree is identical to merged main `a75e543d2`. Follow the exact-revision links; line numbers refer to that snapshot.

| File | Responsibility for the focused extension |
| --- | --- |
| [optimize_cli.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-bench/src/panel/optimize_cli.rs#L13) | Add the request/command only after defining its schema and budget semantics |
| [origin_proof.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-bench/src/panel/origin_proof.rs#L55) | Existing single-frozen-regression plan and resource binding; current dataset restriction is explicit |
| [optimize.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-bench/src/panel/optimize.rs#L111) | Configuration, source/assets, search allowance and fit adapter integration |
| [search_plan.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-pipeline/src/search_plan.rs#L1) | Exact train-side five-fold selection closure and original rows/groups |
| [neural_search.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-pipeline/src/neural_search.rs#L1180) | Initial population, measured ancestry, mutation/crossover, duplicate handling and original-arm resume |
| [campaign_cli.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-bench/src/panel/campaign_cli.rs#L130) | Immutable plan and private journal handling, execution/report front doors |
| [hwm.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-learning/src/hwm.rs) | Existing HWM admission and immutable event history; stricter identity/eligibility changes remain a pending overlay |
| Pending `jope-learning/src/eligibility.rs` and `jope-cli/src/train/corpus.rs` | New authenticated OOF admission and eligible context-to-program corpus export; integrate and qualify before using them as authority |

Required behavioral tests include an improving incumbent, a tie that retains the prior pointer, a worse challenger, a wrong evaluator/split/vocabulary, missing folds, changed categories/class order, unavailable dependencies, cancelled partial waves, concurrent incumbent changes, an exhausted resume window and an attempted sealed/burned-partition admission. Actual successful improvement and successful no-improvement completion are separate acceptance cases.

The owner’s rules for this extension are proposed, not implemented. Focused optimization is finite and incumbent-aware: one total window; exact identity for rows, groups, evaluator, vocabulary and numerics; a declared initial population, exploration and wave schedule; and an atomic incumbent update. A tie keeps the incumbent, and a race with a concurrent incumbent change requires a fresh decision. Ingesting an arbitrary new dataset is a separate interface from frozen30 ([§](#c-search--what-is-missing-for-a-sustained-incumbent-seeded-dataset-attack)).

<a id="c-plan--extending-this-version-without-losing-comparability"></a>

### 13.2.3 Extending this version without losing comparability

A new dataset requires a content/split/group identity, family and exposure classification, train-only preparation, task-correct scoring, and a retained unavailable outcome if any prerequisite fails. A changed encoder requires component/assembly qualification and a new input identity, even when it still outputs 2048 values. A vocabulary change requires explicit token compatibility, executable canonicalization and checkpoint migration decisions. A new learned policy requires optimizer/reload/continuation proof followed by actual downstream generation-and-fit evaluation. A new operator additionally requires typed grammar support, task applicability, ablation and fitted-state/export coverage. A faster scheduler or emitter requires unchanged input/loss semantics and actual cancellation/resume/parity evidence.

Do not close this loop by editing version strings, declaring an old snapshot newly eligible, treating a warmer loss curve as promotion, or retagging a recipe-refit example as a fitted export. Use the source interfaces that exist, integrate the bounded missing path through the serial reviewed queue, and retain every remaining acceptance gate in [§](#c-learning--the-closure-ledger).


---

# Volume 2 — Evidence reference

---

# Appendix A — All thirty datasets and all 522 feature records

All statistics below use development rows only. Numeric values use finite observations; missing cells are counted separately. Standard deviation is the population statistic. Medians and the additional 1%, 5%, 25%, 75%, 95%, 99% quantiles in JSON linearly interpolate sorted values at `(n−1)p`. Category counts retain exact stored strings. Original class ordering and fold class counts are preserved. Upstream metadata, provenance URLs, source byte/hash pins, group-size histograms and all file identities remain in JSON.

<a id="c-datasets--all-thirty-datasets-at-a-glance"></a>

## A.1 All thirty datasets at a glance

<a id="c-datasets--per-dataset-summary"></a>

### A.1.1 Per-dataset summary

| Dataset | Task | Total rows | Development | Sealed metadata | Numeric features | Categorical features | Missing development cells | Fold sizes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 197_cpu_act | regression | 8192 | 6553 | 1639 | 21 | 0 | 0 | 1311,1311,1311,1310,1310 |
| 225_puma8NH | regression | 8192 | 6553 | 1639 | 8 | 0 | 0 | 1311,1311,1311,1310,1310 |
| 230_machine_cpu | regression | 209 | 169 | 40 | 6 | 0 | 0 | 34,32,37,33,33 |
| 522_pm10 | regression | 500 | 402 | 98 | 7 | 0 | 0 | 81,81,91,67,82 |
| 529_pollen | regression | 3848 | 3078 | 770 | 4 | 0 | 0 | 616,616,616,615,615 |
| 579_fri_c0_250_5 | regression | 250 | 200 | 50 | 5 | 0 | 0 | 40,40,40,40,40 |
| auto_insurance_price | regression | 201 | 162 | 39 | 13 | 10 | 10 | 35,32,31,31,33 |
| 690_visualizing_galaxy | regression | 323 | 258 | 65 | 4 | 0 | 0 | 52,52,52,51,51 |
| 712_chscase_geyser1 | regression | 222 | 172 | 50 | 2 | 0 | 0 | 32,34,38,35,33 |
| 503_wind | regression | 6574 | 5238 | 1336 | 14 | 0 | 0 | 1066,1074,1030,1032,1036 |
| breast_cancer_wisconsin_diagnostic | binary | 569 | 455 | 114 | 30 | 0 | 0 | 91,91,91,91,91 |
| credit_approval_germany | binary | 1000 | 800 | 200 | 3 | 17 | 0 | 160,160,160,160,160 |
| congressional_voting_records | binary | 435 | 347 | 88 | 0 | 16 | 0 | 77,57,64,84,65 |
| haberman | binary | 306 | 245 | 61 | 3 | 0 | 0 | 49,49,46,50,51 |
| ionosphere | binary | 351 | 281 | 70 | 32 | 2 | 0 | 55,57,56,57,56 |
| sonar | binary | 208 | 166 | 42 | 60 | 0 | 0 | 34,33,33,33,33 |
| spambase | binary | 4601 | 3723 | 878 | 57 | 0 | 0 | 696,781,710,789,747 |
| phoneme | binary | 5404 | 4330 | 1074 | 5 | 0 | 0 | 866,867,866,870,861 |
| tic_tac_toe | binary | 958 | 766 | 192 | 0 | 9 | 0 | 154,153,153,153,153 |
| horse_colic_surgery | binary | 368 | 292 | 76 | 7 | 13 | 1583 | 58,61,58,58,57 |
| iris | multiclass | 150 | 122 | 28 | 4 | 0 | 0 | 25,23,25,25,24 |
| wine_recognition | multiclass | 178 | 142 | 36 | 13 | 0 | 0 | 29,29,28,28,28 |
| car_evaluation | multiclass | 1728 | 1382 | 346 | 0 | 6 | 0 | 277,277,276,276,276 |
| balance_scale | multiclass | 625 | 500 | 125 | 0 | 4 | 0 | 100,100,100,100,100 |
| dermatology | multiclass | 366 | 292 | 74 | 1 | 33 | 0 | 59,59,58,58,58 |
| segmentation | multiclass | 2310 | 1848 | 462 | 18 | 1 | 0 | 364,372,369,373,370 |
| vehicle | multiclass | 846 | 677 | 169 | 18 | 0 | 0 | 135,136,135,136,135 |
| optdigits | multiclass | 5620 | 4496 | 1124 | 51 | 13 | 0 | 900,899,899,899,899 |
| penguins | multiclass | 344 | 275 | 69 | 5 | 2 | 19 | 55,55,55,55,55 |
| tae | multiclass | 151 | 124 | 27 | 3 | 2 | 0 | 20,25,27,30,22 |

<a id="c-datasets--frozen-dataset-coverage"></a>

### A.1.2 Frozen dataset coverage

The following chart covers all thirty immutable datasets in task order: regression, binary, multiclass. It preserves the original feature-type metadata and row assignments. Sealed counts come from membership metadata; sealed target bodies were not opened. The per-dataset sections under [§](#c-datasets--regression-datasets), [§](#c-datasets--binary-classification-datasets) and [§](#c-datasets--multiclass-classification-datasets) give readable numeric tables, class/category frequencies and every feature record.

![All thirty datasets: frozen rows and feature types](figures/inventory-frozen30.png)

Totals are 55,029 rows, 44,048 development rows, 10,981 sealed membership records and 522 summed feature records. Feature counts are not a union of semantic variables across datasets. The chart and the detailed tables share the same pinned inventory.

<a id="c-datasets--regression-datasets"></a>

## A.2 Regression datasets

<a id="c-datasets--197_cpu_act"></a>

### A.2.1 197_cpu_act

Task `regression`; metric `rmse`; 6553 development rows; 1639 sealed rows counted from assignments; 21 numeric and 0 categorical features. Development groups: 6553; all groups: 8192; largest group: 1. Original fold sizes:[1311, 1311, 1311, 1310, 1310]; other-four-fold fit sizes:[5242, 5242, 5242, 5243, 5243]. Declared grouping columns: none.

Development file SHA-256: `c987b431681784b87f12f3c98e660fd0aa76f08d01fe005695f41470b3750019`. Assignment SHA-256: `624d44ee0a5e2214e8edf256f8d9c028ca3f3765237401e3c84cf9c232a60bc7`. Original source SHA-256: `d00fd6bac2eda0821a04ff10277663d211127745a19a474c0feee41a16914fdc`. Original license: unknown; repository license: MIT.

Development target: min 0, median 89, max 99, mean 84.01938, population SD 18.290468; missing 0.

| Feature | Type | Missing | Distinct nonmissing | Minimum | Median | Maximum | Mean | Population SD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| lread | numeric | 0 | 207 | 0 | 7 | 1845 | 19.349306 | 53.733421 |
| lwrite | numeric | 0 | 177 | 0 | 1 | 575 | 12.922173 | 29.496482 |
| scall | numeric | 0 | 3696 | 109 | 2053 | 12493 | 2307.3473 | 1631.4764 |
| sread | numeric | 0 | 752 | 6 | 166 | 5318 | 209.85686 | 199.84585 |
| swrite | numeric | 0 | 601 | 7 | 117 | 5456 | 149.81505 | 162.43571 |
| fork | numeric | 0 | 207 | 0 | 0.80000001 | 20.120001 | 1.8901007 | 2.4752517 |
| exec | numeric | 0 | 349 | 0 | 1.2 | 59.560001 | 2.7978773 | 5.2634111 |
| rchar | numeric | 0 | 6414 | 278 | 122879 | 2526649 | 195586.22 | 238487.78 |
| wchar | numeric | 0 | 6397 | 1498 | 46082 | 1801623 | 95334.943 | 140638.4 |
| pgout | numeric | 0 | 362 | 0 | 0 | 81.440002 | 2.2342881 | 5.2216254 |
| ppgout | numeric | 0 | 670 | 0 | 0 | 168.06 | 5.8141904 | 14.712169 |
| pgfree | numeric | 0 | 927 | 0 | 0 | 523 | 11.634935 | 31.594236 |
| pgscan | numeric | 0 | 1004 | 0 | 0 | 1237 | 21.142667 | 69.957532 |
| atch | numeric | 0 | 219 | 0 | 0 | 211.58 | 1.0821486 | 5.4985106 |
| pgin | numeric | 0 | 745 | 0 | 2.8 | 126.45 | 8.2493835 | 13.75562 |
| ppgin | numeric | 0 | 955 | 0 | 3.8099999 | 292.60999 | 12.334296 | 22.180491 |
| pflt | numeric | 0 | 2626 | 0 | 64.199997 | 899.79999 | 110.20706 | 114.35173 |
| vflt | numeric | 0 | 3310 | 0.2 | 121.56 | 1365 | 185.84375 | 190.4443 |
| runqsz | numeric | 0 | 253 | 1 | 2 | 2823 | 19.619182 | 128.21611 |
| freemem | numeric | 0 | 2757 | 55 | 579 | 12027 | 1768.3189 | 2497.6881 |
| freeswap | numeric | 0 | 6154 | 2 | 1137126 | 2042280 | 1328586 | 421591.38 |

<a id="c-datasets--225_puma8nh"></a>

### A.2.2 225_puma8NH

Task `regression`; metric `rmse`; 6553 development rows; 1639 sealed rows counted from assignments; 8 numeric and 0 categorical features. Development groups: 6553; all groups: 8192; largest group: 1. Original fold sizes:[1311, 1311, 1311, 1310, 1310]; other-four-fold fit sizes:[5242, 5242, 5242, 5243, 5243]. Declared grouping columns: none.

Development file SHA-256: `31a4ce74a20613e192cab948518ab391feb0b70b0863fbe263d288d4a5df9c6e`. Assignment SHA-256: `f901eda7d22c08a8e7d6b8eb8455f21de4fc58eb4cb497657d2a56bd035cc8e1`. Original source SHA-256: `0be6203e167cc5e7b038368dbfe0a7790d5dc423c9d7e42887907c5c03f81c27`. Original license: unknown; repository license: MIT.

Development target: min-12.415322, median 1.054127, max 11.87619, mean 1.1382282, population SD 5.6448802; missing 0.

| Feature | Type | Missing | Distinct nonmissing | Minimum | Median | Maximum | Mean | Population SD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| theta1 | numeric | 0 | 6547 | -1.884729 | 0.0060899998 | 1.884735 | -0.0022002574 | 1.0823712 |
| theta2 | numeric | 0 | 6543 | -1.8834831 | 0.016564 | 1.883306 | 0.010064784 | 1.0937572 |
| theta3 | numeric | 0 | 6548 | -1.884449 | 0.023234 | 1.884046 | 0.0076206044 | 1.0832023 |
| thetad1 | numeric | 0 | 6548 | -1.884632 | -0.013532 | 1.884923 | -0.007727921 | 1.0913692 |
| thetad2 | numeric | 0 | 6549 | -1.8847851 | -0.030689999 | 1.884077 | -0.013800457 | 1.09539 |
| thetad3 | numeric | 0 | 6546 | -1.884558 | 0.020455999 | 1.884746 | 0.0078690332 | 1.0864734 |
| tau1 | numeric | 0 | 6534 | -0.59999502 | 0.0047479998 | 0.599011 | 0.00079091185 | 0.34369033 |
| tau2 | numeric | 0 | 6533 | -0.59987199 | 0.0048799999 | 0.59991199 | 0.0012599463 | 0.34712905 |

<a id="c-datasets--230_machine_cpu"></a>

### A.2.3 230_machine_cpu

Task `regression`; metric `rmse`; 169 development rows; 40 sealed rows counted from assignments; 6 numeric and 0 categorical features. Development groups: 152; all groups: 190; largest group: 5. Original fold sizes:[34, 32, 37, 33, 33]; other-four-fold fit sizes:[135,137,132,136,136]. Declared grouping columns: none.

Development file SHA-256: `07c9b448d18c0479cb817235efff5ec242fd3723329dcd69582d39043d150c87`. Assignment SHA-256: `68d86fbc518c72ee20dc5feefac8ad2e5a2f7c36ea9613b2755c7177f6f1bf82`. Original source SHA-256: `937b0a23076f948343b8b7ac005bc70ea7080c391218fb4896462699b8cb59c4`. Original license: unknown; repository license: MIT.

Development target: min 6, median 50, max 1150, mean 111.53254, population SD 169.51452; missing 0.

| Feature | Type | Missing | Distinct nonmissing | Minimum | Median | Maximum | Mean | Population SD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MYCT | numeric | 0 | 56 | 17 | 110 | 1500 | 196.59172 | 248.93309 |
| MMIN | numeric | 0 | 20 | 64 | 2000 | 32000 | 2901.7751 | 3928.1762 |
| MMAX | numeric | 0 | 21 | 64 | 8000 | 64000 | 12481.811 | 12389.434 |
| CACH | numeric | 0 | 22 | 0 | 8 | 256 | 24.633136 | 38.093256 |
| CHMIN | numeric | 0 | 15 | 0 | 3 | 52 | 4.9940828 | 7.2483396 |
| CHMAX | numeric | 0 | 28 | 0 | 8 | 176 | 19.100592 | 27.884089 |

<a id="c-datasets--522_pm10"></a>

### A.2.4 522_pm10

Task `regression`; metric `rmse`; 402 development rows; 98 sealed rows counted from assignments; 7 numeric and 0 categorical features. Development groups: 218; all groups: 273; largest group: 6. Original fold sizes:[81, 81, 91, 67, 82]; other-four-fold fit sizes:[321,321,311,335,320]. Declared grouping columns:['day'].

Development file SHA-256: `a33090c6e33323c48afd74338f72b1caa12376b6af4af00f7a48f6c03168df4c`. Assignment SHA-256: `13a00e1a058422110f6ff4caabdac7f648e5da976e10ed0a27be077534e469dd`. Original source SHA-256: `ecbabaa5a989a8c11f5a57e874c135f386da5379d6689ae6d5d4264a652fa5e0`. Original license: unknown; repository license: MIT.

Development target: min 0.69314998, median 3.3322001, max 5.3845, mean 3.3171902, population SD 0.85366052; missing 0.

| Feature | Type | Missing | Distinct nonmissing | Minimum | Median | Maximum | Mean | Population SD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cars_per_hour | numeric | 0 | 374 | 3.8066599 | 7.4974751 | 8.3520803 | 7.0094372 | 1.0713765 |
| temperature_at_2m | numeric | 0 | 197 | -19 | 0.80000001 | 19.799999 | 0.51965173 | 6.3551238 |
| wind_speed | numeric | 0 | 71 | 0.40000001 | 2.8 | 9.3999996 | 3.1445274 | 1.7968553 |
| temperature_diff_2m_25m | numeric | 0 | 56 | -5 | 0 | 4 | 0.11666667 | 0.97697158 |
| wind_direction | numeric | 0 | 288 | 5 | 103.2 | 358 | 140.67065 | 84.378122 |
| hour_of_day | numeric | 0 | 24 | 1 | 12 | 24 | 12.266169 | 6.8120686 |
| day | numeric | 0 | 218 | 32 | 206.5 | 605 | 303.22139 | 197.99917 |

<a id="c-datasets--529_pollen"></a>

### A.2.5 529_pollen

Task `regression`; metric `rmse`; 3078 development rows; 770 sealed rows counted from assignments; 4 numeric and 0 categorical features. Development groups: 3078; all groups: 3848; largest group: 1. Original fold sizes:[616,616,616,615,615]; other-four-fold fit sizes:[2462, 2462, 2462, 2463, 2463]. Declared grouping columns: none.

Development file SHA-256: `4afdc668b7c7bdde004b651e639ee62542518fa9629a4b9a9d18f42d2bfe5cf5`. Assignment SHA-256: `225ab47f4b84b8b29f808751a567dc06e49af57f77d918402dc98929494b9aad`. Original source SHA-256: `2e9414f0216e75296ba4211c567dd8a60d269273b38f8e00bd8fea3a37d2e654`. Original license: unknown; repository license: MIT.

Development target: min-12.0391, median-0.0282, max 10.4934, mean 0.013061306, population SD 3.1652511; missing 0.

| Feature | Type | Missing | Distinct nonmissing | Minimum | Median | Maximum | Mean | Population SD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| RIDGE | numeric | 0 | 3052 | -23.283899 | -0.1736 | 21.406601 | -0.040160139 | 6.4118802 |
| NUB | numeric | 0 | 3047 | -16.393499 | -0.26875 | 17.258301 | -0.017482649 | 5.1239751 |
| CRACK | numeric | 0 | 3057 | -26.4652 | -0.12075 | 25.6434 | -0.07717969 | 7.9651552 |
| WEIGHT | numeric | 0 | 3067 | -34.035198 | -0.1534 | 35.802799 | 0.062779366 | 10.099992 |

<a id="c-datasets--579_fri_c0_250_5"></a>

### A.2.6 579_fri_c0_250_5

Task `regression`; metric `rmse`; 200 development rows; 50 sealed rows counted from assignments; 5 numeric and 0 categorical features. Development groups: 200; all groups: 250; largest group: 1. Original fold sizes:[40, 40, 40, 40, 40]; other-four-fold fit sizes:[160,160,160,160,160]. Declared grouping columns: none.

Development file SHA-256: `9c2608cc2a14cb52e26f174518606fbc4bb7ed48b04aa4c62d840a7dc6cbd365`. Assignment SHA-256: `fa08e1237360b16d98c581d6c62695cb6c8631cca9940ba1671a6b8678e60790`. Original source SHA-256: `d32597bb1043df296ac3be1fff926646b26f268f89185310ea898e59b5fff896`. Original license: unknown; repository license: MIT.

Development target: min-2.2208376, median-0.050367098, max 2.1689749, mean-0.045378203, population SD 1.0368873; missing 0.

| Feature | Type | Missing | Distinct nonmissing | Minimum | Median | Maximum | Mean | Population SD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| oz1 | numeric | 0 | 200 | -1.7117687 | -0.040313881 | 1.7466414 | -0.0097722925 | 0.99221231 |
| oz2 | numeric | 0 | 200 | -1.7675385 | 0.11001987 | 1.7166964 | 0.017360429 | 1.0225496 |
| oz3 | numeric | 0 | 200 | -1.798439 | -0.013159709 | 1.6682318 | 0.024478181 | 0.97661234 |
| oz4 | numeric | 0 | 200 | -1.7556303 | 0.029955601 | 1.6519692 | -0.027132471 | 0.97989126 |
| oz5 | numeric | 0 | 200 | -1.8355519 | 0.022033866 | 1.7011706 | -0.017172821 | 1.0096145 |

<a id="c-datasets--auto_insurance_price"></a>

### A.2.7 auto_insurance_price

<a id="c-datasets--summary-and-feature-statistics"></a>

#### A.2.7.1 Summary and feature statistics

Task `regression`; metric `rmse`; 162 development rows; 39 sealed rows counted from assignments; 13 numeric and 10 categorical features. Development groups: 154; all groups: 193; largest group: 2. Original fold sizes:[35, 32, 31, 31, 33]; other-four-fold fit sizes:[127,130,131,131,129]. Declared grouping columns: none.

Development file SHA-256: `6c0719a5e4433e653e8997fc1f19842fd3aae044531e7639484d24d879d7f971`. Assignment SHA-256: `9769b5f1510b1048afe81125f8f7dbb1272ab81825305af5414c8da6c52c7b1f`. Original source SHA-256: `244c05ca33384bcdd519ff88fad036ae7140a5438b76a8248401cdc065582ac7`. Original license: CC BY 4.0; repository license: MIT.

Development target: min 5118, median 10096.5, max 45400, mean 13127.222, population SD 8034.0417; missing 0.

| Feature | Type | Missing | Distinct nonmissing | Minimum | Median | Maximum | Mean | Population SD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| make | categorical | 0 | 21 | n/a | n/a | n/a | n/a | n/a |
| fuel_type | categorical | 0 | 2 | n/a | n/a | n/a | n/a | n/a |
| aspiration | categorical | 0 | 2 | n/a | n/a | n/a | n/a | n/a |
| number_of_doors | categorical | 0 | 2 | n/a | n/a | n/a | n/a | n/a |
| body_style | categorical | 0 | 5 | n/a | n/a | n/a | n/a | n/a |
| drive_wheels | categorical | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| engine_location | categorical | 0 | 2 | n/a | n/a | n/a | n/a | n/a |
| wheel_base | numeric | 0 | 49 | 86.6 | 96.9 | 120.9 | 98.664815 | 6.1873474 |
| length | numeric | 0 | 67 | 141.1 | 172.5 | 208.1 | 174.10123 | 12.321366 |
| width | numeric | 0 | 40 | 60.3 | 65.4 | 72 | 65.856173 | 2.1333027 |
| height | numeric | 0 | 46 | 47.8 | 54.1 | 59.8 | 53.730247 | 2.5110641 |
| curb_weight | numeric | 0 | 140 | 1488 | 2407.5 | 4066 | 2548.8457 | 518.36907 |
| engine_type | categorical | 0 | 6 | n/a | n/a | n/a | n/a | n/a |
| number_of_cylinders | categorical | 0 | 7 | n/a | n/a | n/a | n/a | n/a |
| engine_size | numeric | 0 | 40 | 61 | 110 | 326 | 126.39506 | 42.981816 |
| fuel_system | categorical | 0 | 7 | n/a | n/a | n/a | n/a | n/a |
| bore | numeric | 4 | 35 | 2.54 | 3.31 | 3.94 | 3.3218987 | 0.26890813 |
| stroke | numeric | 4 | 33 | 2.07 | 3.29 | 4.17 | 3.2515823 | 0.31914744 |
| compression_ratio | numeric | 0 | 28 | 7 | 9 | 23 | 10.003148 | 3.7645327 |
| horsepower | numeric | 1 | 52 | 48 | 95 | 262 | 103.68323 | 38.08872 |
| peak_rpm | numeric | 1 | 21 | 4150 | 5200 | 6600 | 5129.8137 | 480.66268 |
| city_mpg | numeric | 0 | 26 | 13 | 24.5 | 49 | 25.024691 | 6.2370365 |
| highway_mpg | numeric | 0 | 29 | 16 | 30 | 54 | 30.518519 | 6.6640942 |

<a id="c-datasets--category-frequencies"></a>

#### A.2.7.2 Category frequencies

Exact development category frequencies:

- `make`: `{"0": 3, "1": 6, "11": 11, "12": 16, "13": 9, "14": 5, "15": 4, "16": 1, "17": 3, "18": 12, "19": 25, "2": 5, "20": 11, "21": 8, "3": 2, "4": 6, "5": 11, "6": 2, "7": 3, "8": 13, "9": 6}`. Categories absent from the other four folds: `{"0": ["16"], "1": [], "2": [], "3": [], "4": []}`.
- `fuel_type`: `{"0": 14, "1": 148}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `aspiration`: `{"0": 134, "1": 28}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `number_of_doors`: `{"0": 93, "1": 69}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `body_style`: `{"0": 7, "1": 22, "2": 75, "3": 54, "4": 4}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `drive_wheels`: `{"0": 8, "1": 96, "2": 58}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `engine_location`: `{"0": 159, "1": 3}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `engine_type`: `{"0": 11, "1": 10, "2": 112, "3": 15, "4": 10, "5": 4}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `number_of_cylinders`: `{"0": 4, "1": 1, "2": 124, "3": 9, "4": 20, "5": 3, "6": 1}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": ["6"], "3": ["1"], "4": []}`.
- `fuel_system`: `{"0": 9, "1": 55, "2": 3, "3": 14, "5": 72, "6": 8, "7": 1}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": ["7"], "4": []}`.

<a id="c-datasets--690_visualizing_galaxy"></a>

### A.2.8 690_visualizing_galaxy

Task `regression`; metric `rmse`; 258 development rows; 65 sealed rows counted from assignments; 4 numeric and 0 categorical features. Development groups: 258; all groups: 323; largest group: 1. Original fold sizes:[52, 52, 52, 51, 51]; other-four-fold fit sizes:[206,206,206,207,207]. Declared grouping columns: none.

Development file SHA-256: `5c060a8e2459df5e8a533003febab98755a531f3bce0426ac1b617edf2445659`. Assignment SHA-256: `d3f63cf22ce07441cdb3e01d77160d3c36a6630c99c5e7533f9b5563e06db892`. Original source SHA-256: `5fe909e08a622224135a2dcfd409bbe10abab4919f917a07b771c3fb00580852`. Original license: unknown; repository license: MIT.

Development target: min 1409, median 1581, max 1769, mean 1588.6085, population SD 91.295658; missing 0.

| Feature | Type | Missing | Distinct nonmissing | Minimum | Median | Maximum | Mean | Population SD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| eastwest | numeric | 0 | 258 | -29.666929 | -0.42766218 | 29.48414 | -0.60256838 | 12.015002 |
| northsouth | numeric | 0 | 258 | -49.107689 | 1.8613807 | 49.888725 | 2.4837183 | 22.042518 |
| angle | numeric | 0 | 7 | 12.5 | 102.5 | 133 | 81.486434 | 38.60098 |
| radialposition | numeric | 0 | 245 | -51.099998 | -1.8 | 55.700001 | -1.5406977 | 25.187122 |

<a id="c-datasets--712_chscase_geyser1"></a>

### A.2.9 712_chscase_geyser1

Task `regression`; metric `rmse`; 172 development rows; 50 sealed rows counted from assignments; 2 numeric and 0 categorical features. Development groups: 135; all groups: 169; largest group: 4. Original fold sizes:[32, 34, 38, 35, 33]; other-four-fold fit sizes:[140,138,134,137,139]. Declared grouping columns: none.

Development file SHA-256: `a00344615df0ba8b22227753d6931e72c039bed451500c74c794a91038f63459`. Assignment SHA-256: `161ae799380a7919870d5ae493d2c026c6f36fda9fc2dae192ad5f9009fb205b`. Original source SHA-256: `9231e5c6bb08d6a7fab8509e4a8330063a5e3da77b75ff3b9dc412557ed3e334`. Original license: unknown; repository license: MIT.

Development target: min 43, median 75, max 95, mean 71.098837, population SD 12.668703; missing 0.

| Feature | Type | Missing | Distinct nonmissing | Minimum | Median | Maximum | Mean | Population SD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| col_1 | numeric | 0 | 16 | 1 | 16 | 23 | 12.209302 | 7.6502261 |
| col_2 | numeric | 0 | 33 | 1.7 | 4.05 | 5.1999998 | 3.5883721 | 1.0771861 |

<a id="c-datasets--503_wind"></a>

### A.2.10 503_wind

Task `regression`; metric `rmse`; 5238 development rows; 1336 sealed rows counted from assignments; 14 numeric and 0 categorical features. Development groups: 172; all groups: 216; largest group: 31. Original fold sizes:[1066, 1074, 1030, 1032, 1036]; other-four-fold fit sizes:[4172, 4164, 4208, 4206, 4202]. Declared grouping columns:['year', 'month'].

Development file SHA-256: `3bb9b2400ba47561c0df144b7a0d2b8583ad589425d3ada4d0c6313c331d4fd2`. Assignment SHA-256: `1cc0ecb4f63ebddab23f309899171f350164ff1b91b81c144e32ad57640cf281`. Original source SHA-256: `2cc1e040bdc20737674a00039cd80e16758c511e697e524ed17e4335ec93e9a6`. Original license: unknown; repository license: MIT.

Development target: min 1.75, median 14.88, max 42.540001, mean 15.517566, population SD 6.6603293; missing 0.

| Feature | Type | Missing | Distinct nonmissing | Minimum | Median | Maximum | Mean | Population SD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| year | numeric | 0 | 18 | 61 | 70 | 78 | 69.660939 | 5.171068 |
| month | numeric | 0 | 12 | 1 | 6 | 12 | 6.358152 | 3.3938973 |
| day | numeric | 0 | 31 | 1 | 16 | 31 | 15.737686 | 8.8047809 |
| RPT | numeric | 0 | 644 | 0.67000002 | 11.63 | 35.799999 | 12.284956 | 5.5789555 |
| VAL | numeric | 0 | 594 | 0.37 | 10.08 | 33.369999 | 10.579704 | 5.2260336 |
| ROS | numeric | 0 | 589 | 1.5 | 10.83 | 33.84 | 11.576588 | 4.9764516 |
| KIL | numeric | 0 | 439 | 0 | 5.71 | 28.459999 | 6.2592974 | 3.5713878 |
| SHA | numeric | 0 | 575 | 0.13 | 9.9200001 | 33.630001 | 10.369643 | 4.8910479 |
| BIR | numeric | 0 | 446 | 0 | 6.79 | 26.040001 | 7.0677301 | 3.9515225 |
| DUB | numeric | 0 | 565 | 0 | 9.0799999 | 30.370001 | 9.6849523 | 4.9379271 |
| CLA | numeric | 0 | 520 | 0 | 8.04 | 31.08 | 8.4148511 | 4.4694644 |
| MUL | numeric | 0 | 490 | 0 | 8.1199999 | 25.799999 | 8.4560596 | 4.1545162 |
| CLO | numeric | 0 | 520 | 0.039999999 | 8.21 | 28.209999 | 8.6498683 | 4.5013668 |
| BEL | numeric | 0 | 664 | 0.57999998 | 12.42 | 42.380001 | 13.044351 | 5.8033753 |

<a id="c-datasets--binary-classification-datasets"></a>

## A.3 Binary classification datasets

<a id="c-datasets--breast_cancer_wisconsin_diagnostic"></a>

### A.3.1 breast_cancer_wisconsin_diagnostic

Task `binary`; metric `auroc`; 455 development rows; 114 sealed rows counted from assignments; 30 numeric and 0 categorical features. Development groups: 455; all groups: 569; largest group: 1. Original fold sizes:[91, 91, 91, 91, 91]; other-four-fold fit sizes:[364,364,364,364,364]. Declared grouping columns: none.

Development file SHA-256: `cc8bdc81b49bef92ae2aee74a4e919ea5d2515465adef12b7ebae53733e33fc6`. Assignment SHA-256: `41c454f412740e2996d3b7894af2e6d7c75042bdae9db222d14427e1ca7fb64c`. Original source SHA-256: `b3845ddf3afff808cd6449daa546caf4d8b18f109b7ebb2344990e4c37fea291`. Original license: CC BY 4.0; repository license: MIT.

Frozen class order: `["0", "1"]`. Development counts: `{"0": 284, "1": 171}`. Counts by original fold: `{"0": {"0": 55, "1": 36}, "1": {"0": 58, "1": 33}, "2": {"0": 61, "1": 30}, "3": {"0": 57, "1": 34}, "4": {"0": 53, "1": 38}}`.

| Feature | Type | Missing | Distinct nonmissing | Minimum | Median | Maximum | Mean | Population SD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| radius_mean | numeric | 0 | 379 | 6.981 | 13.37 | 28.11 | 14.128822 | 3.5074831 |
| texture_mean | numeric | 0 | 395 | 9.71 | 18.83 | 39.28 | 19.337451 | 4.3409678 |
| perimeter_mean | numeric | 0 | 423 | 43.79 | 86.18 | 188.5 | 91.964747 | 24.188136 |
| area_mean | numeric | 0 | 437 | 143.5 | 551.1 | 2501 | 654.93604 | 347.93468 |
| smoothness_mean | numeric | 0 | 386 | 0.05263 | 0.09578 | 0.1634 | 0.096036352 | 0.014378444 |
| compactness_mean | numeric | 0 | 433 | 0.01938 | 0.09263 | 0.3454 | 0.10405275 | 0.053448375 |
| concavity_mean | numeric | 0 | 432 | 0 | 0.06154 | 0.4268 | 0.088762678 | 0.080580341 |
| n_concave_mean | numeric | 0 | 432 | 0 | 0.03326 | 0.2012 | 0.048554477 | 0.038664424 |
| symmetry_mean | numeric | 0 | 358 | 0.106 | 0.1784 | 0.304 | 0.18096022 | 0.028034982 |
| fractal_dimension_mean | numeric | 0 | 412 | 0.04996 | 0.06161 | 0.09744 | 0.062755077 | 0.00713061 |
| radius_se | numeric | 0 | 438 | 0.1144 | 0.3186 | 2.873 | 0.40416242 | 0.28494741 |
| texture_se | numeric | 0 | 421 | 0.3602 | 1.127 | 4.885 | 1.2224974 | 0.56657768 |
| perimeter_se | numeric | 0 | 432 | 0.757 | 2.244 | 21.98 | 2.8611211 | 2.0984043 |
| area_se | numeric | 0 | 428 | 6.802 | 24.25 | 542.2 | 40.441947 | 47.366591 |
| smoothness_se | numeric | 0 | 442 | 0.001713 | 0.006471 | 0.03113 | 0.0070618176 | 0.0030840353 |
| compactness_se | numeric | 0 | 438 | 0.002252 | 0.02048 | 0.1354 | 0.025522848 | 0.017892893 |
| concavity_se | numeric | 0 | 431 | 0 | 0.02586 | 0.396 | 0.03220903 | 0.031603666 |
| n_concave_se | numeric | 0 | 409 | 0 | 0.01083 | 0.05279 | 0.011810613 | 0.0063615603 |
| symmetry_se | numeric | 0 | 407 | 0.007882 | 0.01868 | 0.07895 | 0.020614919 | 0.0087221018 |
| fractal_dimension_se | numeric | 0 | 437 | 0.0008948 | 0.003224 | 0.02984 | 0.0037881501 | 0.0024496254 |
| radius_worst | numeric | 0 | 387 | 7.93 | 14.98 | 36.04 | 16.256743 | 4.7838454 |
| texture_worst | numeric | 0 | 420 | 12.02 | 25.4 | 49.54 | 25.688879 | 6.1068704 |
| perimeter_worst | numeric | 0 | 419 | 50.41 | 97.96 | 251.2 | 107.09422 | 33.240634 |
| area_worst | numeric | 0 | 440 | 185.2 | 688.6 | 4254 | 878.64 | 556.25614 |
| smoothness_worst | numeric | 0 | 352 | 0.07117 | 0.1312 | 0.2226 | 0.13195897 | 0.023257748 |
| compactness_worst | numeric | 0 | 429 | 0.02729 | 0.2113 | 1.058 | 0.25405004 | 0.15864571 |
| concavity_worst | numeric | 0 | 432 | 0 | 0.2267 | 1.252 | 0.27054573 | 0.20677617 |
| n_concave_worst | numeric | 0 | 396 | 0 | 0.09861 | 0.291 | 0.11400506 | 0.065796181 |
| symmetry_worst | numeric | 0 | 408 | 0.1565 | 0.281 | 0.6638 | 0.2903422 | 0.064265468 |
| fractal_dimension_worst | numeric | 0 | 430 | 0.05504 | 0.07999 | 0.2075 | 0.084133473 | 0.018414541 |

<a id="c-datasets--credit_approval_germany"></a>

### A.3.2 credit_approval_germany

<a id="c-datasets--summary-and-feature-statistics-1"></a>

#### A.3.2.1 Summary and feature statistics

Task `binary`; metric `auroc`; 800 development rows; 200 sealed rows counted from assignments; 3 numeric and 17 categorical features. Development groups: 800; all groups: 1000; largest group: 1. Original fold sizes:[160,160,160,160,160]; other-four-fold fit sizes:[640,640,640,640,640]. Declared grouping columns: none.

Development file SHA-256: `34ffaac6a51ea0ac2e153ba828ce6b2437f685f6d181bb175356a1285c36ef7d`. Assignment SHA-256: `42532d431f177698ca65270388a542d6a687c9634648d523d601ea7763e5a7c4`. Original source SHA-256: `93e819f4b2d26df1cd5eb9805bd2fe15767941c6c8d66c58eba86d9249cb8ce1`. Original license: CC BY 4.0; repository license: MIT.

Frozen class order: `["0", "1"]`. Development counts: `{"0": 552, "1": 248}`. Counts by original fold: `{"0": {"0": 109, "1": 51}, "1": {"0": 107, "1": 53}, "2": {"0": 118, "1": 42}, "3": {"0": 115, "1": 45}, "4": {"0": 103, "1": 57}}`.

| Feature | Type | Missing | Distinct nonmissing | Minimum | Median | Maximum | Mean | Population SD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| credit_height | numeric | 0 | 752 | 276 | 2330 | 18424 | 3317.4475 | 2867.9399 |
| credit_duration | numeric | 0 | 30 | 4 | 18 | 72 | 20.98125 | 12.204237 |
| purpose | categorical | 0 | 10 | n/a | n/a | n/a | n/a | n/a |
| credit_guarantee | categorical | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| rate_fraction_income | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| credit_history | categorical | 0 | 5 | n/a | n/a | n/a | n/a | n/a |
| existing_credits_at_bank | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| other_credits | categorical | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| age | numeric | 0 | 53 | 19 | 34 | 75 | 35.77375 | 11.346919 |
| personal_status | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| dependents | categorical | 0 | 2 | n/a | n/a | n/a | n/a | n/a |
| employment_type | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| employment_length | categorical | 0 | 5 | n/a | n/a | n/a | n/a | n/a |
| foreign_worker | categorical | 0 | 2 | n/a | n/a | n/a | n/a | n/a |
| housing_status | categorical | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| residence_duration | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| checking_account_status | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| savings_account_status | categorical | 0 | 5 | n/a | n/a | n/a | n/a | n/a |
| assets | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| telephone | categorical | 0 | 2 | n/a | n/a | n/a | n/a | n/a |

<a id="c-datasets--category-frequencies-1"></a>

#### A.3.2.2 Category frequencies

Exact development category frequencies:

- `purpose`: `{"0": 181, "1": 78, "2": 151, "3": 226, "4": 9, "5": 15, "6": 41, "7": 8, "8": 81, "9": 10}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `credit_guarantee`: `{"0": 726, "1": 37, "2": 37}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `rate_fraction_income`: `{"0": 108, "1": 177, "2": 131, "3": 384}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `credit_history`: `{"0": 31, "1": 38, "2": 424, "3": 72, "4": 235}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `existing_credits_at_bank`: `{"0": 499, "1": 274, "2": 23, "3": 4}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `other_credits`: `{"0": 112, "1": 38, "2": 650}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `personal_status`: `{"0": 41, "1": 240, "2": 447, "3": 72}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `dependents`: `{"0": 671, "1": 129}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `employment_type`: `{"0": 16, "1": 166, "2": 492, "3": 126}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `employment_length`: `{"0": 51, "1": 131, "2": 261, "3": 145, "4": 212}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `foreign_worker`: `{"0": 768, "1": 32}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `housing_status`: `{"0": 144, "1": 571, "2": 85}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `residence_duration`: `{"0": 104, "1": 243, "2": 113, "3": 340}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `checking_account_status`: `{"0": 212, "1": 211, "2": 58, "3": 319}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `savings_account_status`: `{"0": 485, "1": 84, "2": 54, "3": 33, "4": 144}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `assets`: `{"0": 226, "1": 187, "2": 264, "3": 123}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `telephone`: `{"0": 477, "1": 323}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.

<a id="c-datasets--congressional_voting_records"></a>

### A.3.3 congressional_voting_records

<a id="c-datasets--summary-and-feature-statistics-2"></a>

#### A.3.3.1 Summary and feature statistics

Task `binary`; metric `auroc`; 347 development rows; 88 sealed rows counted from assignments; 0 numeric and 16 categorical features. Development groups: 274; all groups: 342; largest group: 8. Original fold sizes:[77, 57, 64, 84, 65]; other-four-fold fit sizes:[270,290,283,263,282]. Declared grouping columns: none.

Development file SHA-256: `77382af6acc208516bcf44d9ff2911848b90c9893eca5499353e79dd7aa98c0c`. Assignment SHA-256: `3a38db7f711af084a16a69625d5ac782b2591fd2aa0682f0074afaa39bdc45c1`. Original source SHA-256: `fbee2fea90f1e1ce3251869a84b87723e37ab6a512c515a4fb6b9f9d72e5ee0c`. Original license: CC BY 4.0; repository license: MIT.

Frozen class order: `["0", "1"]`. Development counts: `{"0": 209, "1": 138}`. Counts by original fold: `{"0": {"0": 58, "1": 19}, "1": {"0": 33, "1": 24}, "2": {"0": 36, "1": 28}, "3": {"0": 44, "1": 40}, "4": {"0": 38, "1": 27}}`.

| Feature | Type | Missing | Distinct nonmissing | Minimum | Median | Maximum | Mean | Population SD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| handicapped_infants | categorical | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| water_project_cost_sharing | categorical | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| adoption_of_the_budget_resolution | categorical | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| physician_fee_freeze | categorical | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| el_salvador_aid | categorical | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| religious_groups_in_schools | categorical | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| anti_satellite_test_ban | categorical | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| aid_to_nicaraguan_contras | categorical | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| mx_missile | categorical | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| immigration | categorical | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| synfuels_corporation_cutback | categorical | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| education_spending | categorical | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| superfund_right_to_sue | categorical | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| crime | categorical | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| duty_free_exports | categorical | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| export_administation_act_south_africa | categorical | 0 | 3 | n/a | n/a | n/a | n/a | n/a |

<a id="c-datasets--category-frequencies-2"></a>

#### A.3.3.2 Category frequencies

Exact development category frequencies:

- `handicapped_infants`: `{"0": 142, "1": 194, "2": 11}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `water_project_cost_sharing`: `{"0": 157, "1": 160, "2": 30}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `adoption_of_the_budget_resolution`: `{"0": 200, "1": 138, "2": 9}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `physician_fee_freeze`: `{"0": 147, "1": 191, "2": 9}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `el_salvador_aid`: `{"0": 177, "1": 157, "2": 13}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `religious_groups_in_schools`: `{"0": 224, "1": 113, "2": 10}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `anti_satellite_test_ban`: `{"0": 185, "1": 151, "2": 11}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `aid_to_nicaraguan_contras`: `{"0": 192, "1": 144, "2": 11}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `mx_missile`: `{"0": 164, "1": 168, "2": 15}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `immigration`: `{"0": 176, "1": 166, "2": 5}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `synfuels_corporation_cutback`: `{"0": 123, "1": 211, "2": 13}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `education_spending`: `{"0": 142, "1": 182, "2": 23}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `superfund_right_to_sue`: `{"0": 169, "1": 159, "2": 19}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `crime`: `{"0": 207, "1": 127, "2": 13}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `duty_free_exports`: `{"0": 132, "1": 195, "2": 20}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `export_administation_act_south_africa`: `{"0": 219, "1": 53, "2": 75}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.

<a id="c-datasets--haberman"></a>

### A.3.4 haberman

Task `binary`; metric `auroc`; 245 development rows; 61 sealed rows counted from assignments; 3 numeric and 0 categorical features. Development groups: 226; all groups: 283; largest group: 3. Original fold sizes:[49, 49, 46, 50, 51]; other-four-fold fit sizes:[196,196,199,195,194]. Declared grouping columns: none.

Development file SHA-256: `6261779f619ecbdf9184c49da9fae7bb4f8bbcf1ffadffd51637db2673407b3a`. Assignment SHA-256: `705d871416d7b284d96331ead05b1bf138ba0669923cbe7e28668cbc63c6d56e`. Original source SHA-256: `2f44ac12e09d2fd09d9ac70231cecbfbc333356339dec0509ad1ed96e36e94c0`. Original license: unknown; repository license: MIT.

Frozen class order: `["1", "2"]`. Development counts: `{"1": 181, "2": 64}`. Counts by original fold: `{"0": {"1": 36, "2": 13}, "1": {"1": 36, "2": 13}, "2": {"1": 33, "2": 13}, "3": {"1": 38, "2": 12}, "4": {"1": 38, "2": 13}}`.

| Feature | Type | Missing | Distinct nonmissing | Minimum | Median | Maximum | Mean | Population SD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Age_of_patient_at_time_of_operation | numeric | 0 | 47 | 30 | 52 | 83 | 52.665306 | 10.685973 |
| Patients_year_of_operation | numeric | 0 | 12 | 58 | 63 | 69 | 62.857143 | 3.3036776 |
| Number_of_positive_axillary_nodes_detected | numeric | 0 | 28 | 0 | 1 | 46 | 3.9346939 | 6.6807537 |

<a id="c-datasets--ionosphere"></a>

### A.3.5 ionosphere

Task `binary`; metric `auroc`; 281 development rows; 70 sealed rows counted from assignments; 32 numeric and 2 categorical features. Development groups: 280; all groups: 350; largest group: 2. Original fold sizes:[55, 57, 56, 57, 56]; other-four-fold fit sizes:[226,224,225,224,225]. Declared grouping columns: none.

Development file SHA-256: `9dffeb4dac21ce8104c8bb98f1f96968b1d6262c0c47bd2676293a615ef4bf16`. Assignment SHA-256: `73798005d06153b5109154cc728c53e95fe027f65fd548079e364750c38e7f50`. Original source SHA-256: `adb6fc877404f2b55f6d59fc7322265c733f5cf0ae6d7c2e52c3468733630023`. Original license: unknown; repository license: MIT.

Frozen class order: `["0", "1"]`. Development counts: `{"0": 102, "1": 179}`. Counts by original fold: `{"0": {"0": 21, "1": 34}, "1": {"0": 26, "1": 31}, "2": {"0": 18, "1": 38}, "3": {"0": 19, "1": 38}, "4": {"0": 18, "1": 38}}`.

| Feature | Type | Missing | Distinct nonmissing | Minimum | Median | Maximum | Mean | Population SD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | categorical | 0 | 2 | n/a | n/a | n/a | n/a | n/a |
| 1 | categorical | 0 | 1 | n/a | n/a | n/a | n/a | n/a |
| 2 | numeric | 0 | 174 | -1 | 0.88103 | 1 | 0.66155744 | 0.47010906 |
| 3 | numeric | 0 | 215 | -1 | 0.00892 | 1 | 0.028568683 | 0.44112987 |
| 4 | numeric | 0 | 160 | -1 | 0.82683 | 1 | 0.60080947 | 0.53074655 |
| 5 | numeric | 0 | 206 | -1 | 0.0127 | 1 | 0.089482954 | 0.46367215 |
| 6 | numeric | 0 | 180 | -1 | 0.76001 | 1 | 0.55640886 | 0.49881365 |
| 7 | numeric | 0 | 204 | -1 | 0.01128 | 1 | 0.099825658 | 0.52851044 |
| 8 | numeric | 0 | 194 | -1 | 0.71875 | 1 | 0.54520381 | 0.48753108 |
| 9 | numeric | 0 | 211 | -1 | 0.01681 | 1 | 0.17432249 | 0.47797457 |
| 10 | numeric | 0 | 197 | -1 | 0.68627 | 1 | 0.49457996 | 0.55173223 |
| 11 | numeric | 0 | 212 | -1 | 0.02825 | 1 | 0.16022107 | 0.49491547 |
| 12 | numeric | 0 | 193 | -1 | 0.67028 | 1 | 0.4104532 | 0.61450019 |
| 13 | numeric | 0 | 215 | -1 | 0.02105 | 1 | 0.080409858 | 0.50169559 |
| 14 | numeric | 0 | 188 | -1 | 0.62936 | 1 | 0.35983819 | 0.64793776 |
| 15 | numeric | 0 | 214 | -1 | 0 | 1 | 0.077333772 | 0.47171614 |
| 16 | numeric | 0 | 203 | -1 | 0.62121 | 1 | 0.39399758 | 0.61580948 |
| 17 | numeric | 0 | 226 | -1 | 0 | 1 | 0.0025851957 | 0.49900626 |
| 18 | numeric | 0 | 203 | -1 | 0.61996 | 1 | 0.39353537 | 0.61746332 |
| 19 | numeric | 0 | 213 | -1 | 0 | 1 | -0.020650036 | 0.52258519 |
| 20 | numeric | 0 | 198 | -1 | 0.52494 | 1 | 0.3334358 | 0.61650624 |
| 21 | numeric | 0 | 211 | -1 | 0 | 1 | 0.014242954 | 0.51554817 |
| 22 | numeric | 0 | 200 | -1 | 0.53246 | 1 | 0.35669872 | 0.61030843 |
| 23 | numeric | 0 | 209 | -1 | 0 | 1 | -0.036835658 | 0.53328823 |
| 24 | numeric | 0 | 206 | -1 | 0.55389 | 1 | 0.39690103 | 0.57383195 |
| 25 | numeric | 0 | 218 | -1 | -0.01664 | 1 | -0.065681388 | 0.50686043 |
| 26 | numeric | 0 | 206 | -1 | 0.69811 | 1 | 0.53114655 | 0.51992115 |
| 27 | numeric | 0 | 227 | -1 | -0.02294 | 1 | -0.051712669 | 0.5482146 |
| 28 | numeric | 0 | 196 | -1 | 0.48348 | 1 | 0.36189975 | 0.58595804 |
| 29 | numeric | 0 | 212 | -1 | 0 | 1 | -0.03398274 | 0.51497855 |
| 30 | numeric | 0 | 199 | -1 | 0.45114 | 1 | 0.34863491 | 0.56760128 |
| 31 | numeric | 0 | 209 | -1 | 0 | 1 | -0.014466975 | 0.51454688 |
| 32 | numeric | 0 | 198 | -1 | 0.40445 | 1 | 0.33919872 | 0.52562623 |
| 33 | numeric | 0 | 210 | -1 | 0 | 1 | 0.0074272242 | 0.4767589 |

Exact development category frequencies:

- `0`: `{"0": 30, "1": 251}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `1`: `{"0": 281}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.

<a id="c-datasets--sonar"></a>

### A.3.6 sonar

<a id="c-datasets--summary-identities-and-class-counts"></a>

#### A.3.6.1 Summary, identities and class counts

Task `binary`; metric `auroc`; 166 development rows; 42 sealed rows counted from assignments; 60 numeric and 0 categorical features. Development groups: 166; all groups: 208; largest group: 1. Original fold sizes:[34, 33, 33, 33, 33]; other-four-fold fit sizes:[132,133,133,133,133]. Declared grouping columns: none.

Development file SHA-256: `413956c1d626d6671e4b2e99f06ef3c9dbeab52cb947aa2dda52c5eb4d590ec4`. Assignment SHA-256: `dce5e4eb3541c056cab75d72b0be5212d4788bfec249a54a6ffe6160e82853f9`. Original source SHA-256: `64378cabaecdb50e2e1e644c1d6bc359f0627ebe313870ad09e1e41ed972529b`. Original license: unknown; repository license: MIT.

Frozen class order: `["0", "1"]`. Development counts: `{"0": 88, "1": 78}`. Counts by original fold: `{"0": {"0": 19, "1": 15}, "1": {"0": 20, "1": 13}, "2": {"0": 16, "1": 17}, "3": {"0": 15, "1": 18}, "4": {"0": 18, "1": 15}}`.

<a id="c-datasets--feature-statistics"></a>

#### A.3.6.2 Feature statistics

| Feature | Type | Missing | Distinct nonmissing | Minimum | Median | Maximum | Mean | Population SD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A1 | numeric | 0 | 141 | 0.0015 | 0.0217 | 0.1371 | 0.028841566 | 0.023686115 |
| A2 | numeric | 0 | 151 | 0.0006 | 0.0286 | 0.2339 | 0.037833735 | 0.034101163 |
| A3 | numeric | 0 | 157 | 0.0015 | 0.0321 | 0.3059 | 0.042960843 | 0.037819594 |
| A4 | numeric | 0 | 152 | 0.0058 | 0.04405 | 0.4264 | 0.053654217 | 0.046923156 |
| A5 | numeric | 0 | 157 | 0.0067 | 0.0625 | 0.401 | 0.075328916 | 0.055578143 |
| A6 | numeric | 0 | 159 | 0.0102 | 0.0898 | 0.3823 | 0.10617169 | 0.061505341 |
| A7 | numeric | 0 | 159 | 0.0033 | 0.11245 | 0.3729 | 0.12627711 | 0.064775691 |
| A8 | numeric | 0 | 162 | 0.0055 | 0.1117 | 0.459 | 0.13576145 | 0.088328103 |
| A9 | numeric | 0 | 164 | 0.0075 | 0.15375 | 0.6828 | 0.18029096 | 0.12123601 |
| A10 | numeric | 0 | 166 | 0.0113 | 0.1809 | 0.7106 | 0.2100741 | 0.138032 |
| A11 | numeric | 0 | 162 | 0.0289 | 0.2273 | 0.7342 | 0.23626867 | 0.13285597 |
| A12 | numeric | 0 | 165 | 0.0259 | 0.24905 | 0.706 | 0.24793976 | 0.13719085 |
| A13 | numeric | 0 | 161 | 0.0184 | 0.25735 | 0.7131 | 0.27112349 | 0.13802798 |
| A14 | numeric | 0 | 165 | 0.0273 | 0.2811 | 0.997 | 0.29665482 | 0.16533271 |
| A15 | numeric | 0 | 163 | 0.0031 | 0.27805 | 0.9137 | 0.32257771 | 0.20514931 |
| A16 | numeric | 0 | 162 | 0.0162 | 0.31505 | 0.9751 | 0.38494157 | 0.23299201 |
| A17 | numeric | 0 | 161 | 0.0349 | 0.3242 | 1 | 0.42208735 | 0.26767643 |
| A18 | numeric | 0 | 163 | 0.0375 | 0.3712 | 1 | 0.45387711 | 0.26423326 |
| A19 | numeric | 0 | 164 | 0.0746 | 0.42215 | 1 | 0.49631566 | 0.25898032 |
| A20 | numeric | 0 | 162 | 0.0656 | 0.4977 | 1 | 0.54843916 | 0.26497105 |
| A21 | numeric | 0 | 162 | 0.0512 | 0.58605 | 1 | 0.59400783 | 0.25705453 |
| A22 | numeric | 0 | 162 | 0.0219 | 0.65335 | 1 | 0.61085241 | 0.2582013 |
| A23 | numeric | 0 | 157 | 0.0563 | 0.66805 | 1 | 0.63666627 | 0.24875836 |
| A24 | numeric | 0 | 159 | 0.0239 | 0.68315 | 1 | 0.6634747 | 0.23756303 |
| A25 | numeric | 0 | 159 | 0.024 | 0.7211 | 1 | 0.6719988 | 0.24365831 |
| A26 | numeric | 0 | 155 | 0.0921 | 0.7545 | 1 | 0.69583916 | 0.23788158 |
| A27 | numeric | 0 | 155 | 0.0481 | 0.7211 | 1 | 0.7011488 | 0.24248268 |
| A28 | numeric | 0 | 155 | 0.0284 | 0.7337 | 1 | 0.69982289 | 0.2339089 |
| A29 | numeric | 0 | 158 | 0.0144 | 0.6808 | 1 | 0.64515843 | 0.23642261 |
| A30 | numeric | 0 | 162 | 0.1354 | 0.61945 | 1 | 0.58621928 | 0.21518403 |
| A31 | numeric | 0 | 166 | 0.0482 | 0.5037 | 0.9657 | 0.51252169 | 0.21079284 |
| A32 | numeric | 0 | 164 | 0.0404 | 0.44535 | 0.9306 | 0.44586867 | 0.20698438 |
| A33 | numeric | 0 | 163 | 0.0477 | 0.39375 | 0.9708 | 0.42063614 | 0.19916545 |
| A34 | numeric | 0 | 165 | 0.0212 | 0.3682 | 0.9647 | 0.40769639 | 0.23003905 |
| A35 | numeric | 0 | 163 | 0.0223 | 0.3369 | 1 | 0.40306205 | 0.2611019 |
| A36 | numeric | 0 | 164 | 0.0185 | 0.32115 | 1 | 0.3937 | 0.27059015 |
| A37 | numeric | 0 | 166 | 0.0379 | 0.3103 | 0.9497 | 0.37263133 | 0.24345705 |
| A38 | numeric | 0 | 164 | 0.0383 | 0.3153 | 1 | 0.34736988 | 0.21649059 |
| A39 | numeric | 0 | 163 | 0.0407 | 0.28345 | 0.9857 | 0.33274458 | 0.20047974 |
| A40 | numeric | 0 | 164 | 0.0202 | 0.28225 | 0.9297 | 0.32078976 | 0.17907208 |
| A41 | numeric | 0 | 164 | 0.0438 | 0.26305 | 0.8995 | 0.29696867 | 0.17064681 |
| A42 | numeric | 0 | 166 | 0.0056 | 0.2534 | 0.8246 | 0.27926265 | 0.17015944 |
| A43 | numeric | 0 | 163 | 0.0159 | 0.2162 | 0.7733 | 0.24302651 | 0.14246534 |
| A44 | numeric | 0 | 157 | 0.0255 | 0.1751 | 0.7762 | 0.21003795 | 0.13362077 |
| A45 | numeric | 0 | 165 | 0.0046 | 0.148 | 0.7034 | 0.19582169 | 0.1514589 |
| A46 | numeric | 0 | 161 | 0.008 | 0.1179 | 0.7292 | 0.16208976 | 0.13726397 |
| A47 | numeric | 0 | 163 | 0.0073 | 0.10385 | 0.5522 | 0.12454398 | 0.089007856 |
| A48 | numeric | 0 | 162 | 0.0052 | 0.0803 | 0.3339 | 0.09448494 | 0.063893903 |
| A49 | numeric | 0 | 154 | 0.0073 | 0.0451 | 0.1981 | 0.053189157 | 0.035846172 |
| A50 | numeric | 0 | 132 | 0.0006 | 0.0179 | 0.0825 | 0.020369277 | 0.013156943 |
| A51 | numeric | 0 | 134 | 0.0009 | 0.014 | 0.1004 | 0.015896386 | 0.01129344 |
| A52 | numeric | 0 | 127 | 0.0008 | 0.01125 | 0.0709 | 0.013522289 | 0.0099364813 |
| A53 | numeric | 0 | 117 | 0.0005 | 0.00955 | 0.039 | 0.010813855 | 0.0070422897 |
| A54 | numeric | 0 | 114 | 0.0011 | 0.0094 | 0.0352 | 0.010825904 | 0.0070825505 |
| A55 | numeric | 0 | 113 | 0.0006 | 0.0076 | 0.0447 | 0.0095614458 | 0.0071552547 |
| A56 | numeric | 0 | 108 | 0.0004 | 0.00705 | 0.0394 | 0.0082789157 | 0.0057867194 |
| A57 | numeric | 0 | 105 | 0.0003 | 0.00575 | 0.0355 | 0.007623494 | 0.0057258021 |
| A58 | numeric | 0 | 115 | 0.0003 | 0.0059 | 0.044 | 0.0080210843 | 0.0065779644 |
| A59 | numeric | 0 | 103 | 0.0001 | 0.0061 | 0.0332 | 0.0079795181 | 0.0061732079 |
| A60 | numeric | 0 | 98 | 0.0007 | 0.0053 | 0.0439 | 0.0065289157 | 0.0052317043 |

<a id="c-datasets--spambase"></a>

### A.3.7 spambase

<a id="c-datasets--summary-identities-and-class-counts-1"></a>

#### A.3.7.1 Summary, identities and class counts

Task `binary`; metric `auroc`; 3723 development rows; 878 sealed rows counted from assignments; 57 numeric and 0 categorical features. Development groups: 3381; all groups: 4207; largest group: 69. Original fold sizes:[696,781,710,789,747]; other-four-fold fit sizes:[3027, 2942, 3013, 2934, 2976]. Declared grouping columns: none.

Development file SHA-256: `f02d0bf948106d71bdc4dd57a1acad05e36935396d3ce2450aa326e863d93748`. Assignment SHA-256: `d740e152f7c9942115e88fe36274779a4187bc3cdafa050d75d75ca207804991`. Original source SHA-256: `735e8d43f7ee78ac54ca17404510e95f4c8b364a721e4d9532a1b3221ed70961`. Original license: unknown; repository license: MIT.

Frozen class order: `["0", "1"]`. Development counts: `{"0": 2261, "1": 1462}`. Counts by original fold: `{"0": {"0": 413, "1": 283}, "1": {"0": 486, "1": 295}, "2": {"0": 422, "1": 288}, "3": {"0": 489, "1": 300}, "4": {"0": 451, "1": 296}}`.

<a id="c-datasets--feature-statistics-1"></a>

#### A.3.7.2 Feature statistics

| Feature | Type | Missing | Distinct nonmissing | Minimum | Median | Maximum | Mean | Population SD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | numeric | 0 | 136 | 0 | 0 | 4.34 | 0.10836691 | 0.31447626 |
| 1 | numeric | 0 | 158 | 0 | 0 | 14.28 | 0.23424926 | 1.4140581 |
| 2 | numeric | 0 | 208 | 0 | 0 | 5.1 | 0.28450175 | 0.50613967 |
| 3 | numeric | 0 | 39 | 0 | 0 | 42.81 | 0.076145582 | 1.53401 |
| 4 | numeric | 0 | 238 | 0 | 0 | 8.33 | 0.30181574 | 0.63247696 |
| 5 | numeric | 0 | 134 | 0 | 0 | 5.88 | 0.099076014 | 0.28457055 |
| 6 | numeric | 0 | 159 | 0 | 0 | 7.27 | 0.1084448 | 0.37299389 |
| 7 | numeric | 0 | 154 | 0 | 0 | 11.11 | 0.10242815 | 0.40766094 |
| 8 | numeric | 0 | 136 | 0 | 0 | 5.26 | 0.090405587 | 0.28227639 |
| 9 | numeric | 0 | 233 | 0 | 0 | 18.18 | 0.24231534 | 0.66893504 |
| 10 | numeric | 0 | 106 | 0 | 0 | 2.61 | 0.057700779 | 0.19908542 |
| 11 | numeric | 0 | 301 | 0 | 0.12 | 9.67 | 0.53767123 | 0.85542489 |
| 12 | numeric | 0 | 147 | 0 | 0 | 5.55 | 0.094139135 | 0.29447118 |
| 13 | numeric | 0 | 119 | 0 | 0 | 10 | 0.057160892 | 0.3424115 |
| 14 | numeric | 0 | 105 | 0 | 0 | 4.41 | 0.050021488 | 0.26409551 |
| 15 | numeric | 0 | 238 | 0 | 0 | 20 | 0.25256245 | 0.86568242 |
| 16 | numeric | 0 | 178 | 0 | 0 | 7.14 | 0.14358313 | 0.44016113 |
| 17 | numeric | 0 | 209 | 0 | 0 | 7.69 | 0.17846092 | 0.49965964 |
| 18 | numeric | 0 | 549 | 0 | 1.3 | 18.75 | 1.6304647 | 1.7444236 |
| 19 | numeric | 0 | 133 | 0 | 0 | 18.18 | 0.087045393 | 0.51437355 |
| 20 | numeric | 0 | 388 | 0 | 0.23 | 11.11 | 0.82221595 | 1.2353801 |
| 21 | numeric | 0 | 81 | 0 | 0 | 17.1 | 0.12898469 | 1.0758037 |
| 22 | numeric | 0 | 157 | 0 | 0 | 5.45 | 0.10662369 | 0.36628547 |
| 23 | numeric | 0 | 133 | 0 | 0 | 12.5 | 0.088568359 | 0.37568098 |
| 24 | numeric | 0 | 355 | 0 | 0 | 20.83 | 0.53969917 | 1.6402831 |
| 25 | numeric | 0 | 251 | 0 | 0 | 16.66 | 0.2710744 | 0.91077722 |
| 26 | numeric | 0 | 221 | 0 | 0 | 33.33 | 0.87795326 | 3.6989178 |
| 27 | numeric | 0 | 179 | 0 | 0 | 5.88 | 0.12094547 | 0.51186153 |
| 28 | numeric | 0 | 140 | 0 | 0 | 14.28 | 0.09994628 | 0.61141527 |
| 29 | numeric | 0 | 167 | 0 | 0 | 5.88 | 0.10789954 | 0.47740186 |
| 30 | numeric | 0 | 113 | 0 | 0 | 12.5 | 0.068775181 | 0.4277239 |
| 31 | numeric | 0 | 90 | 0 | 0 | 4.76 | 0.049648133 | 0.34482243 |
| 32 | numeric | 0 | 167 | 0 | 0 | 8.33 | 0.091326887 | 0.48361691 |
| 33 | numeric | 0 | 97 | 0 | 0 | 4.76 | 0.050577491 | 0.34585255 |
| 34 | numeric | 0 | 161 | 0 | 0 | 20 | 0.10762557 | 0.5614855 |
| 35 | numeric | 0 | 151 | 0 | 0 | 4.76 | 0.099151222 | 0.40425707 |
| 36 | numeric | 0 | 182 | 0 | 0 | 6.89 | 0.13856299 | 0.43320226 |
| 37 | numeric | 0 | 46 | 0 | 0 | 8.33 | 0.014488316 | 0.24219157 |
| 38 | numeric | 0 | 143 | 0 | 0 | 11.11 | 0.07558152 | 0.42405656 |
| 39 | numeric | 0 | 116 | 0 | 0 | 4.76 | 0.067894171 | 0.36984345 |
| 40 | numeric | 0 | 89 | 0 | 0 | 7.14 | 0.045412302 | 0.37629292 |
| 41 | numeric | 0 | 166 | 0 | 0 | 14.28 | 0.12858716 | 0.77234937 |
| 42 | numeric | 0 | 127 | 0 | 0 | 3.57 | 0.045417674 | 0.21994189 |
| 43 | numeric | 0 | 143 | 0 | 0 | 20 | 0.080515713 | 0.65927373 |
| 44 | numeric | 0 | 220 | 0 | 0 | 21.42 | 0.29374161 | 0.99304581 |
| 45 | numeric | 0 | 202 | 0 | 0 | 22.05 | 0.18154714 | 0.94847185 |
| 46 | numeric | 0 | 34 | 0 | 0 | 2.17 | 0.0059629331 | 0.082484296 |
| 47 | numeric | 0 | 93 | 0 | 0 | 5 | 0.025753425 | 0.19009714 |
| 48 | numeric | 0 | 280 | 0 | 0 | 4.385 | 0.03976014 | 0.24401967 |
| 49 | numeric | 0 | 594 | 0 | 0.065 | 9.752 | 0.14165941 | 0.28547001 |
| 50 | numeric | 0 | 203 | 0 | 0 | 4.081 | 0.017885039 | 0.11686678 |
| 51 | numeric | 0 | 867 | 0 | 0 | 32.478 | 0.26717996 | 0.80078459 |
| 52 | numeric | 0 | 456 | 0 | 0 | 6.003 | 0.07593285 | 0.24670478 |
| 53 | numeric | 0 | 279 | 0 | 0 | 19.829 | 0.046236637 | 0.46825088 |
| 54 | numeric | 0 | 1867 | 1 | 2.285 | 1102.5 | 5.2333 | 32.564591 |
| 55 | numeric | 0 | 244 | 1 | 15 | 9989 | 53.011281 | 208.61383 |
| 56 | numeric | 0 | 827 | 1 | 97 | 15841 | 282.47059 | 604.67923 |

<a id="c-datasets--phoneme"></a>

### A.3.8 phoneme

Task `binary`; metric `auroc`; 4330 development rows; 1074 sealed rows counted from assignments; 5 numeric and 0 categorical features. Development groups: 4279; all groups: 5349; largest group: 2. Original fold sizes:[866,867,866,870,861]; other-four-fold fit sizes:[3464, 3463, 3464, 3460, 3469]. Declared grouping columns: none.

Development file SHA-256: `bbb0adca013fc84e710901a2a7ad7523000c351e38c9e041cbd9bef125315c3e`. Assignment SHA-256: `0b377d73783b9e26995fe4a821dfefc745d84316f7f457bcecf4985bb832e0f2`. Original source SHA-256: `2e1c31a2c560f3539be5bda1084939765c80ddd814bd6919fd8a3465f9b8aee6`. Original license: unknown; repository license: MIT.

Frozen class order: `["0", "1"]`. Development counts: `{"0": 3048, "1": 1282}`. Counts by original fold: `{"0": {"0": 610, "1": 256}, "1": {"0": 623, "1": 244}, "2": {"0": 583, "1": 283}, "3": {"0": 623, "1": 247}, "4": {"0": 609, "1": 252}}`.

| Feature | Type | Missing | Distinct nonmissing | Minimum | Median | Maximum | Mean | Population SD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Aa | numeric | 0 | 1851 | -1.7 | 0.4865 | 4.107 | 0.82364804 | 0.86789971 |
| Ao | numeric | 0 | 2210 | -1.327 | 1.068 | 4.378 | 1.2516467 | 0.85489827 |
| Dcl | numeric | 0 | 2277 | -1.641 | 0.716 | 3.155 | 0.76292702 | 0.92614533 |
| Iy | numeric | 0 | 2152 | -1.494 | 0.286 | 2.785 | 0.40085681 | 0.79622039 |
| Sh | numeric | 0 | 1607 | -1.246 | -0.044 | 2.719 | 0.079781524 | 0.57446239 |

<a id="c-datasets--tic_tac_toe"></a>

### A.3.9 tic_tac_toe

<a id="c-datasets--summary-and-feature-statistics-3"></a>

#### A.3.9.1 Summary and feature statistics

Task `binary`; metric `auroc`; 766 development rows; 192 sealed rows counted from assignments; 0 numeric and 9 categorical features. Development groups: 766; all groups: 958; largest group: 1. Original fold sizes:[154,153,153,153,153]; other-four-fold fit sizes:[612,613,613,613,613]. Declared grouping columns: none.

Development file SHA-256: `3d2766839ac876c2c3342d3340b2b10d78aae59d0663ec4ced605836f86fc643`. Assignment SHA-256: `5f25ffba77126e2c010512008c353731e2c65de4d33d7b47d2cad20a9366cf1b`. Original source SHA-256: `d00d1522a06b5b67a0aea355c268ffe575e4a28651fa4fbf0e204705be3ce47b`. Original license: unknown; repository license: MIT.

Frozen class order: `["0", "1"]`. Development counts: `{"0": 252, "1": 514}`. Counts by original fold: `{"0": {"0": 54, "1": 100}, "1": {"0": 53, "1": 100}, "2": {"0": 51, "1": 102}, "3": {"0": 47, "1": 106}, "4": {"0": 47, "1": 106}}`.

| Feature | Type | Missing | Distinct nonmissing | Minimum | Median | Maximum | Mean | Population SD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| t1 | categorical | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| t2 | categorical | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| t3 | categorical | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| t4 | categorical | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| t5 | categorical | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| t6 | categorical | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| t7 | categorical | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| t8 | categorical | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| t9 | categorical | 0 | 3 | n/a | n/a | n/a | n/a | n/a |

<a id="c-datasets--category-frequencies-3"></a>

#### A.3.9.2 Category frequencies

Exact development category frequencies:

- `t1`: `{"0": 178, "1": 260, "2": 328}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `t2`: `{"0": 201, "1": 266, "2": 299}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `t3`: `{"0": 168, "1": 268, "2": 330}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `t4`: `{"0": 204, "1": 262, "2": 300}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `t5`: `{"0": 119, "1": 273, "2": 374}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `t6`: `{"0": 196, "1": 261, "2": 309}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `t7`: `{"0": 160, "1": 263, "2": 343}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `t8`: `{"0": 214, "1": 262, "2": 290}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `t9`: `{"0": 167, "1": 265, "2": 334}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.

<a id="c-datasets--horse_colic_surgery"></a>

### A.3.10 horse_colic_surgery

<a id="c-datasets--summary-and-feature-statistics-4"></a>

#### A.3.10.1 Summary and feature statistics

Task `binary`; metric `auroc`; 292 development rows; 76 sealed rows counted from assignments; 7 numeric and 13 categorical features. Development groups: 283; all groups: 354; largest group: 4. Original fold sizes:[58, 61, 58, 58, 57]; other-four-fold fit sizes:[234,231,234,234,235]. Declared grouping columns: none.

Development file SHA-256: `4b92b18271d7a289565d1845df42b8d903b22c321954cc78dd9fba90800f92ff`. Assignment SHA-256: `1cd242e37b4b64ebc39ba83e834f7822ebe1bc19746b8970c74574deb99f3539`. Original source SHA-256: `5079ad94ed5d03057e754b97692f1675603d515ebd663fcf34b87c45103e449c`. Original license: CC BY 4.0; repository license: MIT.

Frozen class order: `["0", "1"]`. Development counts: `{"0": 180, "1": 112}`. Counts by original fold: `{"0": {"0": 38, "1": 20}, "1": {"0": 35, "1": 26}, "2": {"0": 31, "1": 27}, "3": {"0": 38, "1": 20}, "4": {"0": 38, "1": 19}}`.

| Feature | Type | Missing | Distinct nonmissing | Minimum | Median | Maximum | Mean | Population SD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| age_category | categorical | 25 | 1 | n/a | n/a | n/a | n/a | n/a |
| rectal_temperature | numeric | 59 | 38 | 35.4 | 38.1 | 40.8 | 38.139056 | 0.73254976 |
| heart_rate | numeric | 20 | 53 | 30 | 62 | 184 | 71.330882 | 29.13295 |
| respiratory_rate | numeric | 62 | 38 | 8 | 28 | 96 | 31.617391 | 18.790878 |
| extremity_temperature | categorical | 53 | 4 | n/a | n/a | n/a | n/a | n/a |
| peripheral_pulse | categorical | 67 | 4 | n/a | n/a | n/a | n/a | n/a |
| mucous_membrane_appearance | categorical | 39 | 6 | n/a | n/a | n/a | n/a | n/a |
| capillary_refill_time | categorical | 36 | 2 | n/a | n/a | n/a | n/a | n/a |
| pain | categorical | 47 | 5 | n/a | n/a | n/a | n/a | n/a |
| peristalsis | categorical | 41 | 4 | n/a | n/a | n/a | n/a | n/a |
| abdominal_distension | categorical | 53 | 4 | n/a | n/a | n/a | n/a | n/a |
| nasogastric_tube_stomach_gas | categorical | 115 | 3 | n/a | n/a | n/a | n/a | n/a |
| nasogastric_tube_reflux_volume | categorical | 112 | 3 | n/a | n/a | n/a | n/a | n/a |
| nasogastric_tube_reflux_ph | numeric | 237 | 22 | 1 | 5 | 8 | 4.6490909 | 1.9697598 |
| rectal_feces_presence | categorical | 100 | 4 | n/a | n/a | n/a | n/a | n/a |
| rectal_examination | categorical | 112 | 5 | n/a | n/a | n/a | n/a | n/a |
| red_blood_cell_count | numeric | 31 | 50 | 4 | 44 | 75 | 44.961303 | 10.838738 |
| blood_protein | numeric | 37 | 78 | 3.3 | 7.5 | 86 | 25.715294 | 27.945219 |
| abdominocentesis_appearance | categorical | 154 | 3 | n/a | n/a | n/a | n/a | n/a |
| abdominocentesis_protein | numeric | 183 | 38 | 0.1 | 2 | 10.1 | 2.8963303 | 1.841366 |

<a id="c-datasets--category-frequencies-4"></a>

#### A.3.10.2 Category frequencies

Exact development category frequencies:

- `age_category`: `{"0": 267}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `extremity_temperature`: `{"0": 74, "1": 33, "2": 107, "3": 25}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `peripheral_pulse`: `{"0": 5, "1": 118, "2": 91, "3": 11}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `mucous_membrane_appearance`: `{"0": 87, "1": 25, "2": 61, "3": 41, "4": 20, "5": 19}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `capillary_refill_time`: `{"0": 180, "1": 76}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `pain`: `{"0": 44, "1": 62, "2": 65, "3": 37, "4": 37}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `peristalsis`: `{"0": 41, "1": 16, "2": 123, "3": 71}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `abdominal_distension`: `{"0": 82, "1": 60, "2": 64, "3": 33}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `nasogastric_tube_stomach_gas`: `{"0": 61, "1": 92, "2": 24}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `nasogastric_tube_reflux_volume`: `{"0": 109, "1": 37, "2": 34}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `rectal_feces_presence`: `{"0": 11, "1": 56, "2": 50, "3": 75}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `rectal_examination`: `{"0": 27, "1": 17, "2": 18, "3": 40, "4": 78}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `abdominocentesis_appearance`: `{"0": 43, "1": 43, "2": 52}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.

<a id="c-datasets--multiclass-classification-datasets"></a>

## A.4 Multiclass classification datasets

<a id="c-datasets--iris"></a>

### A.4.1 iris

Task `multiclass`; metric `log_loss`; 122 development rows; 28 sealed rows counted from assignments; 4 numeric and 0 categorical features. Development groups: 119; all groups: 147; largest group: 3. Original fold sizes:[25, 23, 25, 25, 24]; other-four-fold fit sizes:[97, 99, 97, 97, 98]. Declared grouping columns: none.

Development file SHA-256: `86b4b47bc14f4513838072b130aed1b9c4f5bd9056cc110215a8c66be3d9dcc3`. Assignment SHA-256: `4322393c97bb172340ad840055cbc0edb09b8321ece9d7ce5b0aed1934aadfad`. Original source SHA-256: `ce4982cba30bd5aaa352e51d55e1cf56c56ae6d573d644f36257de1177717228`. Original license: unknown; repository license: MIT.

Frozen class order: `["0", "1", "2"]`. Development counts: `{"0": 37, "1": 40, "2": 45}`. Counts by original fold: `{"0": {"0": 10, "1": 10, "2": 5}, "1": {"0": 5, "1": 11, "2": 7}, "2": {"0": 8, "1": 7, "2": 10}, "3": {"0": 7, "1": 8, "2": 10}, "4": {"0": 7, "1": 4, "2": 13}}`.

| Feature | Type | Missing | Distinct nonmissing | Minimum | Median | Maximum | Mean | Population SD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sepal-length | numeric | 0 | 34 | 4.3 | 5.8 | 7.9 | 5.9008197 | 0.8357284 |
| sepal-width | numeric | 0 | 23 | 2 | 3 | 4.4 | 3.0491803 | 0.4435782 |
| petal-length | numeric | 0 | 40 | 1 | 4.5 | 6.9 | 3.8901639 | 1.7489767 |
| petal-width | numeric | 0 | 21 | 0.1 | 1.4 | 2.5 | 1.2540984 | 0.76170081 |

<a id="c-datasets--wine_recognition"></a>

### A.4.2 wine_recognition

Task `multiclass`; metric `log_loss`; 142 development rows; 36 sealed rows counted from assignments; 13 numeric and 0 categorical features. Development groups: 142; all groups: 178; largest group: 1. Original fold sizes:[29, 29, 28, 28, 28]; other-four-fold fit sizes:[113,113,114,114,114]. Declared grouping columns: none.

Development file SHA-256: `ff8de7ed12ef935c0d42be31c08bae6ca18ae909e60dee1d9c1887ce333e253f`. Assignment SHA-256: `0a5a84bbb63803ef0b31e7fde40c3fcd2d0e2246b62033cc820be3c907917614`. Original source SHA-256: `56f5ba3f3ba78f6e522a13e5a97e03ff28cf9fa4107021b22b932f6d4064c145`. Original license: unknown; repository license: MIT.

Frozen class order: `["1", "2", "3"]`. Development counts: `{"1": 47, "2": 57, "3": 38}`. Counts by original fold: `{"0": {"1": 10, "2": 11, "3": 8}, "1": {"1": 9, "2": 12, "3": 8}, "2": {"1": 10, "2": 11, "3": 7}, "3": {"1": 9, "2": 12, "3": 7}, "4": {"1": 9, "2": 11, "3": 8}}`.

| Feature | Type | Missing | Distinct nonmissing | Minimum | Median | Maximum | Mean | Population SD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | numeric | 0 | 106 | 11.45 | 13.05 | 14.83 | 13.017746 | 0.78176704 |
| 2 | numeric | 0 | 114 | 0.89 | 1.88 | 5.8 | 2.3453521 | 1.140337 |
| 3 | numeric | 0 | 74 | 1.36 | 2.36 | 3.23 | 2.3671831 | 0.27320313 |
| 4 | numeric | 0 | 55 | 10.6 | 19.25 | 30 | 19.509859 | 3.3591227 |
| 5 | numeric | 0 | 48 | 70 | 98 | 162 | 99.43662 | 14.362247 |
| 6 | numeric | 0 | 80 | 0.98 | 2.38 | 3.88 | 2.3109155 | 0.62909045 |
| 7 | numeric | 0 | 111 | 0.34 | 2.135 | 5.08 | 2.0360563 | 1.0064125 |
| 8 | numeric | 0 | 39 | 0.13 | 0.335 | 0.66 | 0.36577465 | 0.12517213 |
| 9 | numeric | 0 | 89 | 0.42 | 1.545 | 3.58 | 1.6009859 | 0.58075874 |
| 10 | numeric | 0 | 111 | 1.28 | 4.69 | 13 | 5.0982394 | 2.3589459 |
| 11 | numeric | 0 | 71 | 0.48 | 0.96 | 1.45 | 0.94574648 | 0.21709943 |
| 12 | numeric | 0 | 106 | 1.27 | 2.78 | 3.82 | 2.6256338 | 0.69527932 |
| 13 | numeric | 0 | 106 | 278 | 673.5 | 1680 | 751.73239 | 317.35709 |

<a id="c-datasets--car_evaluation"></a>

### A.4.3 car_evaluation

<a id="c-datasets--summary-and-feature-statistics-5"></a>

#### A.4.3.1 Summary and feature statistics

Task `multiclass`; metric `log_loss`; 1382 development rows; 346 sealed rows counted from assignments; 0 numeric and 6 categorical features. Development groups: 1382; all groups: 1728; largest group: 1. Original fold sizes:[277,277,276,276,276]; other-four-fold fit sizes:[1105, 1105, 1106, 1106, 1106]. Declared grouping columns: none.

Development file SHA-256: `014c5c81b356e617cac7daa8901914c12837f5458c221882466f2536f047fc46`. Assignment SHA-256: `c36e08e3b50f2c560f4a0d06cf8d719da1a1352870246ad04f12ec7b7a855fab`. Original source SHA-256: `9c18fe91ba5401cab754456a9bfbfe54a6c307b0d4b880259fb4218edef77998`. Original license: CC BY 4.0; repository license: MIT.

Frozen class order: `["0", "1", "2", "3"]`. Development counts: `{"0": 967, "1": 310, "2": 49, "3": 56}`. Counts by original fold: `{"0": {"0": 194, "1": 64, "2": 10, "3": 9}, "1": {"0": 195, "1": 59, "2": 12, "3": 11}, "2": {"0": 200, "1": 53, "2": 9, "3": 14}, "3": {"0": 186, "1": 69, "2": 10, "3": 11}, "4": {"0": 192, "1": 65, "2": 8, "3": 11}}`.

| Feature | Type | Missing | Distinct nonmissing | Minimum | Median | Maximum | Mean | Population SD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| buying_price | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| maintenance_costs | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| doors | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| seats | categorical | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| luggage_boot | categorical | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| safety | categorical | 0 | 3 | n/a | n/a | n/a | n/a | n/a |

<a id="c-datasets--category-frequencies-5"></a>

#### A.4.3.2 Category frequencies

Exact development category frequencies:

- `buying_price`: `{"0": 345, "1": 346, "2": 346, "3": 345}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `maintenance_costs`: `{"0": 345, "1": 346, "2": 346, "3": 345}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `doors`: `{"0": 341, "1": 351, "2": 346, "3": 344}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `seats`: `{"0": 454, "1": 466, "2": 462}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `luggage_boot`: `{"0": 448, "1": 468, "2": 466}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `safety`: `{"0": 467, "1": 468, "2": 447}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.

<a id="c-datasets--balance_scale"></a>

### A.4.4 balance_scale

<a id="c-datasets--summary-and-feature-statistics-6"></a>

#### A.4.4.1 Summary and feature statistics

Task `multiclass`; metric `log_loss`; 500 development rows; 125 sealed rows counted from assignments; 0 numeric and 4 categorical features. Development groups: 500; all groups: 625; largest group: 1. Original fold sizes:[100,100,100,100,100]; other-four-fold fit sizes:[400,400,400,400,400]. Declared grouping columns: none.

Development file SHA-256: `9267117d816d008803477a791ca9cf346fefc17aba62afc9a9b5d444e9530329`. Assignment SHA-256: `40efdd8487cb7076e85a00d07e1d42d4902a84e1e9aef740d291e341b2509764`. Original source SHA-256: `8c87be0e86f362154b34bd608caf8c5804e357e437b7db91b14974b6afaffe32`. Original license: unknown; repository license: MIT.

Frozen class order: `["0", "1", "2"]`. Development counts: `{"0": 40, "1": 211, "2": 249}`. Counts by original fold: `{"0": {"0": 4, "1": 49, "2": 47}, "1": {"0": 11, "1": 42, "2": 47}, "2": {"0": 8, "1": 39, "2": 53}, "3": {"0": 11, "1": 43, "2": 46}, "4": {"0": 6, "1": 38, "2": 56}}`.

| Feature | Type | Missing | Distinct nonmissing | Minimum | Median | Maximum | Mean | Population SD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Left-Weight | categorical | 0 | 5 | n/a | n/a | n/a | n/a | n/a |
| Left-Distance | categorical | 0 | 5 | n/a | n/a | n/a | n/a | n/a |
| Right-Weight | categorical | 0 | 5 | n/a | n/a | n/a | n/a | n/a |
| Right-Distance | categorical | 0 | 5 | n/a | n/a | n/a | n/a | n/a |

<a id="c-datasets--category-frequencies-6"></a>

#### A.4.4.2 Category frequencies

Exact development category frequencies:

- `Left-Weight`: `{"1": 100, "2": 100, "3": 100, "4": 100, "5": 100}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `Left-Distance`: `{"1": 100, "2": 100, "3": 100, "4": 98, "5": 102}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `Right-Weight`: `{"1": 101, "2": 101, "3": 99, "4": 99, "5": 100}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `Right-Distance`: `{"1": 70, "2": 86, "3": 106, "4": 118, "5": 120}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.

<a id="c-datasets--dermatology"></a>

### A.4.5 dermatology

<a id="c-datasets--summary-identities-and-class-counts-2"></a>

#### A.4.5.1 Summary, identities and class counts

Task `multiclass`; metric `log_loss`; 292 development rows; 74 sealed rows counted from assignments; 1 numeric and 33 categorical features. Development groups: 292; all groups: 366; largest group: 1. Original fold sizes:[59, 59, 58, 58, 58]; other-four-fold fit sizes:[233,233,234,234,234]. Declared grouping columns: none.

Development file SHA-256: `a447d3a279d1f4c11f3b330935d362e3d8ca4c5e99afe5276f01e411a8e0795e`. Assignment SHA-256: `88ce0ea91c15e418c566d1a455a7b01b6cbc4335be5fbfca3efcf4549267a379`. Original source SHA-256: `c8852ba546e92249debecd9a897cda3f94fd10872fa13c62eb14ab40e279166e`. Original license: unknown; repository license: MIT.

Frozen class order: `["1", "2", "3", "4", "5", "6"]`. Development counts: `{"1": 89, "2": 48, "3": 61, "4": 39, "5": 40, "6": 15}`. Counts by original fold: `{"0": {"1": 16, "2": 9, "3": 11, "4": 10, "5": 8, "6": 5}, "1": {"1": 16, "2": 13, "3": 13, "4": 9, "5": 6, "6": 2}, "2": {"1": 15, "2": 9, "3": 16, "4": 8, "5": 7, "6": 3}, "3": {"1": 22, "2": 9, "3": 10, "4": 6, "5": 10, "6": 1}, "4": {"1": 20, "2": 8, "3": 11, "4": 6, "5": 9, "6": 4}}`.

<a id="c-datasets--feature-statistics-2"></a>

#### A.4.5.2 Feature statistics

| Feature | Type | Missing | Distinct nonmissing | Minimum | Median | Maximum | Mean | Population SD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| erythema | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| scaling | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| definite_borders | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| itching | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| koebner_phenomenon | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| polygonal_papules | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| follicular_papules | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| oral_mucosal_involvement | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| knee_and_elbow_involvement | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| scalp_involvement | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| family_history | categorical | 0 | 2 | n/a | n/a | n/a | n/a | n/a |
| melanin_incontinence | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| eosinophils_in_the_infiltrate | categorical | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| PNL_infiltrate | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| fibrosis_of_the_papillary_dermis | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| exocytosis | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| acanthosis | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| hyperkeratosis | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| parakeratosis | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| clubbing_of_the_rete_ridges | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| elongation_of_the_rete_ridges | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| thinning_of_the_suprapapillary_epidermis | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| spongiform_pustule | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| munro_microabcess | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| focal_hypergranulosis | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| disappearance_of_the_granular_layer | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| vacuolisation_and_damage_of_basal_layer | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| spongiosis | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| saw-tooth_appearance_of_retes | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| follicular_horn_plug | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| perifollicular_parakeratosis | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| inflammatory_monoluclear_inflitrate | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| band-like_infiltrate | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| Age | numeric | 0 | 58 | 0 | 26.5 | 60 | 27.921233 | 15.306591 |

<a id="c-datasets--category-frequencies-7"></a>

#### A.4.5.3 Category frequencies

Exact development category frequencies:

- `erythema`: `{"0": 3, "1": 45, "2": 175, "3": 69}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `scaling`: `{"0": 6, "1": 84, "2": 160, "3": 42}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `definite_borders`: `{"0": 45, "1": 72, "2": 133, "3": 42}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `itching`: `{"0": 90, "1": 62, "2": 80, "3": 60}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `koebner_phenomenon`: `{"0": 176, "1": 58, "2": 42, "3": 16}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `polygonal_papules`: `{"0": 234, "1": 1, "2": 32, "3": 25}`. Categories absent from the other four folds: `{"0": [], "1": ["1"], "2": [], "3": [], "4": []}`.
- `follicular_papules`: `{"0": 265, "1": 10, "2": 14, "3": 3}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `oral_mucosal_involvement`: `{"0": 236, "1": 9, "2": 36, "3": 11}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `knee_and_elbow_involvement`: `{"0": 201, "1": 21, "2": 50, "3": 20}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `scalp_involvement`: `{"0": 210, "1": 23, "2": 44, "3": 15}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `family_history`: `{"0": 257, "1": 35}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `melanin_incontinence`: `{"0": 233, "1": 8, "2": 40, "3": 11}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `eosinophils_in_the_infiltrate`: `{"0": 263, "1": 22, "2": 7}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `PNL_infiltrate`: `{"0": 190, "1": 57, "2": 39, "3": 6}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `fibrosis_of_the_papillary_dermis`: `{"0": 250, "1": 7, "2": 18, "3": 17}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `exocytosis`: `{"0": 92, "1": 47, "2": 109, "3": 44}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `acanthosis`: `{"0": 8, "1": 59, "2": 164, "3": 61}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `hyperkeratosis`: `{"0": 181, "1": 73, "2": 33, "3": 5}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `parakeratosis`: `{"0": 70, "1": 94, "2": 105, "3": 23}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `clubbing_of_the_rete_ridges`: `{"0": 202, "1": 17, "2": 44, "3": 29}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `elongation_of_the_rete_ridges`: `{"0": 161, "1": 20, "2": 74, "3": 37}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `thinning_of_the_suprapapillary_epidermis`: `{"0": 203, "1": 16, "2": 48, "3": 25}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `spongiform_pustule`: `{"0": 240, "1": 28, "2": 21, "3": 3}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `munro_microabcess`: `{"0": 231, "1": 26, "2": 26, "3": 9}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `focal_hypergranulosis`: `{"0": 232, "1": 12, "2": 36, "3": 12}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `disappearance_of_the_granular_layer`: `{"0": 216, "1": 23, "2": 40, "3": 13}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `vacuolisation_and_damage_of_basal_layer`: `{"0": 231, "1": 3, "2": 37, "3": 21}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `spongiosis`: `{"0": 155, "1": 23, "2": 81, "3": 33}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `saw-tooth_appearance_of_retes`: `{"0": 232, "1": 4, "2": 34, "3": 22}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `follicular_horn_plug`: `{"0": 276, "1": 8, "2": 5, "3": 3}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `perifollicular_parakeratosis`: `{"0": 276, "1": 4, "2": 9, "3": 3}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `inflammatory_monoluclear_inflitrate`: `{"0": 11, "1": 64, "2": 171, "3": 46}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `band-like_infiltrate`: `{"0": 226, "1": 3, "2": 19, "3": 44}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.

<a id="c-datasets--segmentation"></a>

### A.4.6 segmentation

Task `multiclass`; metric `log_loss`; 1848 development rows; 462 sealed rows counted from assignments; 18 numeric and 1 categorical features. Development groups: 1669; all groups: 2086; largest group: 3. Original fold sizes:[364,372,369,373,370]; other-four-fold fit sizes:[1484, 1476, 1479, 1475, 1478]. Declared grouping columns: none.

Development file SHA-256: `6691b654d75aa6ba4a66792fe7cbeef46738c9a02b5ca667cd41d9e141ecc67a`. Assignment SHA-256: `10bcfcc5634490f57e7b69f3b8071fd1e49d63e2a03a62e182772e6f3df321d6`. Original source SHA-256: `980ba4342d2fe489793cc3431678318f8d1467fe6bf6a687eb18bfc2267a049f`. Original license: unknown; repository license: MIT.

Frozen class order: `["0", "1", "2", "3", "4", "5", "6"]`. Development counts: `{"0": 261, "1": 262, "2": 265, "3": 262, "4": 269, "5": 260, "6": 269}`. Counts by original fold: `{"0": {"0": 47, "1": 49, "2": 51, "3": 54, "4": 51, "5": 59, "6": 53}, "1": {"0": 52, "1": 54, "2": 54, "3": 53, "4": 56, "5": 46, "6": 57}, "2": {"0": 51, "1": 52, "2": 51, "3": 53, "4": 56, "5": 53, "6": 53}, "3": {"0": 57, "1": 48, "2": 56, "3": 54, "4": 51, "5": 53, "6": 54}, "4": {"0": 54, "1": 59, "2": 53, "3": 48, "4": 55, "5": 49, "6": 52}}`.

| Feature | Type | Missing | Distinct nonmissing | Minimum | Median | Maximum | Mean | Population SD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| region-centroid-col | numeric | 0 | 253 | 1 | 121 | 254 | 124.82955 | 72.877771 |
| region-centroid-row | numeric | 0 | 237 | 11 | 122 | 251 | 123.68019 | 57.350115 |
| region-pixel-count | categorical | 0 | 1 | n/a | n/a | n/a | n/a | n/a |
| short-line-density-5 | numeric | 0 | 4 | 0 | 0 | 0.33333334 | 0.014490139 | 0.040503723 |
| short-line-density-2 | numeric | 0 | 3 | 0 | 0 | 0.22222222 | 0.0046897546 | 0.024343745 |
| vedge-mean | numeric | 0 | 582 | 0 | 1.2777773 | 27.944443 | 1.8955628 | 2.6719697 |
| vedge-sd | numeric | 0 | 1459 | 0 | 0.83148183 | 991.7184 | 5.9348586 | 45.97572 |
| hedge-mean | numeric | 0 | 637 | 0 | 1.4444447 | 44.722225 | 2.4776034 | 3.7718629 |
| hedge-sd | numeric | 0 | 1472 | -1.5894573e-08 | 0.99814636 | 1386.3292 | 8.7672312 | 61.930518 |
| intensity-mean | numeric | 0 | 1087 | 0 | 21.611111 | 143.44444 | 36.80293 | 38.072465 |
| rawred-mean | numeric | 0 | 622 | 0 | 19.555555 | 137.11111 | 32.616883 | 34.956813 |
| rawblue-mean | numeric | 0 | 714 | 0 | 27.277778 | 150.88889 | 43.863998 | 43.415929 |
| rawgreen-mean | numeric | 0 | 637 | 0 | 20.333334 | 142.55556 | 33.92791 | 36.242479 |
| exred-mean | numeric | 0 | 414 | -49.666668 | -10.666667 | 9.888889 | -12.558141 | 11.498634 |
| exblue-mean | numeric | 0 | 604 | -12.444445 | 19.444445 | 82 | 21.183201 | 19.520045 |
| exgreen-mean | numeric | 0 | 361 | -33.88889 | -10.666667 | 24.666666 | -8.6250601 | 11.569724 |
| value-mean | numeric | 0 | 715 | 0 | 28.555555 | 150.88889 | 44.813493 | 42.815875 |
| saturatoin-mean | numeric | 0 | 1519 | 0 | 0.37455574 | 1 | 0.42759796 | 0.23090285 |
| hue-mean | numeric | 0 | 1552 | -3.0441751 | -2.0481283 | 2.9124804 | -1.3641642 | 1.5416936 |

Exact development category frequencies:

- `region-pixel-count`: `{"9.0": 1848}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.

<a id="c-datasets--vehicle"></a>

### A.4.7 vehicle

Task `multiclass`; metric `log_loss`; 677 development rows; 169 sealed rows counted from assignments; 18 numeric and 0 categorical features. Development groups: 677; all groups: 846; largest group: 1. Original fold sizes:[135,136,135,136,135]; other-four-fold fit sizes:[542,541,542,541,542]. Declared grouping columns: none.

Development file SHA-256: `117699e46fbd73fdfaa9996fc671a7a88463cc768f0f1e859b77ad7120a49b18`. Assignment SHA-256: `44645392494e0be6c4a4d3e708db4af46b0d4c7af52bc6cf3666bf407109030b`. Original source SHA-256: `dad03273414ae7127e30b6e4e66df90319a148b37ead5dd376dd140be5baedc4`. Original license: unknown; repository license: MIT.

Frozen class order: `["1", "2", "3", "4"]`. Development counts: `{"1": 169, "2": 175, "3": 174, "4": 159}`. Counts by original fold: `{"0": {"1": 34, "2": 35, "3": 36, "4": 30}, "1": {"1": 33, "2": 36, "3": 33, "4": 34}, "2": {"1": 37, "2": 34, "3": 34, "4": 30}, "3": {"1": 34, "2": 36, "3": 32, "4": 34}, "4": {"1": 31, "2": 34, "3": 39, "4": 31}}`.

| Feature | Type | Missing | Distinct nonmissing | Minimum | Median | Maximum | Mean | Population SD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| COMPACTNESS | numeric | 0 | 43 | 73 | 93 | 119 | 93.949778 | 8.1342405 |
| CIRCULARITY | numeric | 0 | 27 | 33 | 44 | 59 | 44.909897 | 6.1908953 |
| DISTANCE CIRCULARITY | numeric | 0 | 63 | 40 | 80 | 112 | 82.490399 | 15.889816 |
| RADIUS RATIO | numeric | 0 | 134 | 104 | 169 | 333 | 169.83161 | 33.806618 |
| PR AXIS ASPECT RATIO | numeric | 0 | 36 | 47 | 61 | 138 | 61.805022 | 8.2894697 |
| MAX LENGTH ASPECT RATIO | numeric | 0 | 19 | 2 | 8 | 55 | 8.6070901 | 4.8232398 |
| SCATTER RATIO | numeric | 0 | 126 | 112 | 157 | 265 | 169.65731 | 33.487679 |
| ELONGATEDNESS | numeric | 0 | 35 | 26 | 43 | 61 | 40.748892 | 7.8641824 |
| PR AXISRECTANGULAR | numeric | 0 | 13 | 17 | 20 | 29 | 20.645495 | 2.6103189 |
| LENGTHRECTANGULAR | numeric | 0 | 65 | 118 | 146 | 186 | 148.14032 | 14.560102 |
| MAJORVARIANCE | numeric | 0 | 125 | 130 | 179 | 320 | 189.4195 | 31.887562 |
| MINORVARIANCE | numeric | 0 | 384 | 184 | 366 | 1018 | 444.34269 | 178.10982 |
| GYRATIONRADIUS | numeric | 0 | 138 | 112 | 174 | 268 | 174.82422 | 32.80978 |
| MAJORSKEWNESS | numeric | 0 | 39 | 59 | 71 | 135 | 72.462334 | 7.7825212 |
| MINORSKEWNESS | numeric | 0 | 23 | 0 | 5 | 22 | 6.3353028 | 4.9982104 |
| MINORKURTOSIS | numeric | 0 | 41 | 0 | 11 | 41 | 12.82127 | 9.0213413 |
| MAJORKURTOSIS | numeric | 0 | 29 | 176 | 189 | 204 | 189.05761 | 6.1604295 |
| HOLLOWS RATIO | numeric | 0 | 31 | 181 | 197 | 211 | 195.79173 | 7.4621283 |

<a id="c-datasets--optdigits"></a>

### A.4.8 optdigits

<a id="c-datasets--summary-identities-and-class-counts-3"></a>

#### A.4.8.1 Summary, identities and class counts

Task `multiclass`; metric `log_loss`; 4496 development rows; 1124 sealed rows counted from assignments; 51 numeric and 13 categorical features. Development groups: 4496; all groups: 5620; largest group: 1. Original fold sizes:[900,899,899,899,899]; other-four-fold fit sizes:[3596, 3597, 3597, 3597, 3597]. Declared grouping columns: none.

Development file SHA-256: `afaa9819c5ac7059a64f7457e73444b3ba7c28ccb4cd3de27cdf5fe13f7751f6`. Assignment SHA-256: `0407e498a26a4cea16fe28c170e626d1338dc4bf8148d0e1be17ba3d87056018`. Original source SHA-256: `b70b7ac68a54d1ccd41bd24f11324c3af96b67541791fb9d8684cbdf151533c3`. Original license: unknown; repository license: MIT.

Frozen class order: `["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]`. Development counts: `{"0": 444, "1": 456, "2": 444, "3": 476, "4": 462, "5": 433, "6": 437, "7": 444, "8": 439, "9": 461}`. Counts by original fold: `{"0": {"0": 84, "1": 90, "2": 96, "3": 82, "4": 100, "5": 96, "6": 86, "7": 85, "8": 87, "9": 94}, "1": {"0": 88, "1": 90, "2": 74, "3": 109, "4": 95, "5": 83, "6": 96, "7": 93, "8": 93, "9": 78}, "2": {"0": 77, "1": 99, "2": 106, "3": 92, "4": 89, "5": 83, "6": 95, "7": 80, "8": 72, "9": 106}, "3": {"0": 93, "1": 88, "2": 92, "3": 89, "4": 90, "5": 79, "6": 72, "7": 99, "8": 94, "9": 103}, "4": {"0": 102, "1": 89, "2": 76, "3": 104, "4": 88, "5": 92, "6": 88, "7": 87, "8": 93, "9": 80}}`.

<a id="c-datasets--feature-statistics-3"></a>

#### A.4.8.2 Feature statistics

| Feature | Type | Missing | Distinct nonmissing | Minimum | Median | Maximum | Mean | Population SD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| input1 | categorical | 0 | 1 | n/a | n/a | n/a | n/a | n/a |
| input2 | categorical | 0 | 9 | n/a | n/a | n/a | n/a | n/a |
| input3 | numeric | 0 | 17 | 0 | 5 | 16 | 5.3854537 | 4.6694807 |
| input4 | numeric | 0 | 17 | 0 | 13 | 16 | 11.763345 | 4.2575085 |
| input5 | numeric | 0 | 17 | 0 | 13 | 16 | 11.573176 | 4.4513964 |
| input6 | numeric | 0 | 17 | 0 | 4 | 16 | 5.5584964 | 5.6073441 |
| input7 | numeric | 0 | 17 | 0 | 0 | 16 | 1.3727758 | 3.349232 |
| input8 | numeric | 0 | 16 | 0 | 0 | 15 | 0.13122776 | 0.99381714 |
| input9 | categorical | 0 | 4 | n/a | n/a | n/a | n/a | n/a |
| input10 | numeric | 0 | 16 | 0 | 0 | 15 | 1.9730872 | 3.0879654 |
| input11 | numeric | 0 | 17 | 0 | 12 | 16 | 10.52758 | 5.4372549 |
| input12 | numeric | 0 | 17 | 0 | 13 | 16 | 11.79782 | 4.01247 |
| input13 | numeric | 0 | 17 | 0 | 11 | 16 | 10.540036 | 4.7693414 |
| input14 | numeric | 0 | 17 | 0 | 9 | 16 | 8.261121 | 5.9790084 |
| input15 | numeric | 0 | 17 | 0 | 0 | 16 | 2.0991993 | 3.9437706 |
| input16 | numeric | 0 | 14 | 0 | 0 | 13 | 0.13411922 | 0.91377922 |
| input17 | categorical | 0 | 5 | n/a | n/a | n/a | n/a | n/a |
| input18 | numeric | 0 | 17 | 0 | 1 | 16 | 2.5849644 | 3.4827212 |
| input19 | numeric | 0 | 17 | 0 | 11 | 16 | 9.6209964 | 5.8251162 |
| input20 | numeric | 0 | 17 | 0 | 6 | 16 | 6.8256228 | 5.8753604 |
| input21 | numeric | 0 | 17 | 0 | 6 | 16 | 7.1846085 | 6.1510996 |
| input22 | numeric | 0 | 17 | 0 | 9 | 16 | 7.9857651 | 6.2479372 |
| input23 | numeric | 0 | 17 | 0 | 0 | 16 | 1.9759786 | 3.498392 |
| input24 | categorical | 0 | 8 | n/a | n/a | n/a | n/a | n/a |
| input25 | categorical | 0 | 2 | n/a | n/a | n/a | n/a | n/a |
| input26 | numeric | 0 | 17 | 0 | 1 | 16 | 2.363879 | 3.0806441 |
| input27 | numeric | 0 | 17 | 0 | 11 | 16 | 9.1581406 | 6.1416147 |
| input28 | numeric | 0 | 17 | 0 | 10 | 16 | 9.0745107 | 5.9029827 |
| input29 | numeric | 0 | 17 | 0 | 12 | 16 | 9.7602313 | 6.2595395 |
| input30 | numeric | 0 | 17 | 0 | 8 | 16 | 7.7955961 | 5.9558196 |
| input31 | numeric | 0 | 17 | 0 | 0 | 16 | 2.3709964 | 3.6629328 |
| input32 | categorical | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| input33 | categorical | 0 | 2 | n/a | n/a | n/a | n/a | n/a |
| input34 | numeric | 0 | 16 | 0 | 0 | 15 | 2.1419039 | 3.3179478 |
| input35 | numeric | 0 | 17 | 0 | 8 | 16 | 7.67371 | 6.2749206 |
| input36 | numeric | 0 | 17 | 0 | 11 | 16 | 9.155694 | 6.1866184 |
| input37 | numeric | 0 | 17 | 0 | 12 | 16 | 10.287811 | 5.9272272 |
| input38 | numeric | 0 | 17 | 0 | 10 | 16 | 9.0538256 | 5.8700142 |
| input39 | numeric | 0 | 15 | 0 | 1 | 14 | 2.9219306 | 3.5062095 |
| input40 | categorical | 0 | 1 | n/a | n/a | n/a | n/a | n/a |
| input41 | categorical | 0 | 8 | n/a | n/a | n/a | n/a | n/a |
| input42 | numeric | 0 | 17 | 0 | 0 | 16 | 1.4971085 | 3.0053319 |
| input43 | numeric | 0 | 17 | 0 | 5 | 16 | 6.6127669 | 6.5197965 |
| input44 | numeric | 0 | 17 | 0 | 7 | 16 | 7.1683719 | 6.4851243 |
| input45 | numeric | 0 | 17 | 0 | 8 | 16 | 7.8153915 | 6.2973257 |
| input46 | numeric | 0 | 17 | 0 | 10 | 16 | 8.5473754 | 5.7808416 |
| input47 | numeric | 0 | 17 | 0 | 1 | 16 | 3.518016 | 4.3473655 |
| input48 | categorical | 0 | 7 | n/a | n/a | n/a | n/a | n/a |
| input49 | categorical | 0 | 9 | n/a | n/a | n/a | n/a | n/a |
| input50 | numeric | 0 | 17 | 0 | 0 | 16 | 0.80649466 | 1.9683692 |
| input51 | numeric | 0 | 17 | 0 | 8 | 16 | 7.7595641 | 5.6567184 |
| input52 | numeric | 0 | 17 | 0 | 10 | 16 | 9.7504448 | 5.1729862 |
| input53 | numeric | 0 | 17 | 0 | 10 | 16 | 9.6554715 | 5.2848009 |
| input54 | numeric | 0 | 17 | 0 | 11 | 16 | 9.1548043 | 5.9832257 |
| input55 | numeric | 0 | 17 | 0 | 1 | 16 | 3.7446619 | 4.8885914 |
| input56 | numeric | 0 | 13 | 0 | 0 | 13 | 0.15814057 | 0.81911142 |
| input57 | categorical | 0 | 2 | n/a | n/a | n/a | n/a | n/a |
| input58 | numeric | 0 | 11 | 0 | 0 | 10 | 0.27935943 | 0.92382518 |
| input59 | numeric | 0 | 17 | 0 | 5 | 16 | 5.7382117 | 5.007576 |
| input60 | numeric | 0 | 17 | 0 | 13 | 16 | 11.981317 | 4.3434726 |
| input61 | numeric | 0 | 17 | 0 | 13 | 16 | 11.577625 | 4.9533614 |
| input62 | numeric | 0 | 17 | 0 | 6 | 16 | 6.7164146 | 5.7925554 |
| input63 | numeric | 0 | 17 | 0 | 0 | 16 | 2.0507117 | 3.990158 |
| input64 | numeric | 0 | 17 | 0 | 0 | 16 | 0.24154804 | 1.3632569 |

<a id="c-datasets--category-frequencies-8"></a>

#### A.4.8.3 Category frequencies

Exact development category frequencies:

- `input1`: `{"0.0": 4496}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `input2`: `{"0.0": 3810, "1.0": 343, "2.0": 175, "3.0": 89, "4.0": 41, "5.0": 19, "6.0": 10, "7.0": 6, "8.0": 3}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `input9`: `{"0.0": 4489, "1.0": 4, "2.0": 2, "5.0": 1}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": ["5.0"], "3": [], "4": []}`.
- `input17`: `{"0.0": 4484, "1.0": 6, "2.0": 3, "3.0": 2, "5.0": 1}`. Categories absent from the other four folds: `{"0": [], "1": ["5.0"], "2": [], "3": [], "4": []}`.
- `input24`: `{"0.0": 4418, "1.0": 24, "2.0": 17, "3.0": 11, "4.0": 12, "5.0": 7, "6.0": 5, "8.0": 2}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": ["8.0"], "3": [], "4": []}`.
- `input25`: `{"0.0": 4491, "1.0": 5}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `input32`: `{"0.0": 4484, "1.0": 11, "2.0": 1}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": ["2.0"], "4": []}`.
- `input33`: `{"0.0": 4492, "1.0": 4}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `input40`: `{"0.0": 4496}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `input41`: `{"0.0": 4455, "1.0": 14, "2.0": 11, "3.0": 6, "4.0": 5, "5.0": 3, "6.0": 1, "7.0": 1}`. Categories absent from the other four folds: `{"0": [], "1": ["7.0"], "2": [], "3": ["6.0"], "4": []}`.
- `input48`: `{"0.0": 4441, "1.0": 36, "2.0": 11, "3.0": 2, "4.0": 3, "5.0": 1, "6.0": 2}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": ["5.0"]}`.
- `input49`: `{"0.0": 4465, "1.0": 18, "10.0": 1, "2.0": 4, "3.0": 4, "4.0": 1, "5.0": 1, "7.0": 1, "8.0": 1}`. Categories absent from the other four folds: `{"0": ["5.0"], "1": ["4.0"], "2": ["10.0", "7.0"], "3": [], "4": ["8.0"]}`.
- `input57`: `{"0.0": 4494, "1.0": 2}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.

<a id="c-datasets--penguins"></a>

### A.4.9 penguins

Task `multiclass`; metric `log_loss`; 275 development rows; 69 sealed rows counted from assignments; 5 numeric and 2 categorical features. Development groups: 275; all groups: 344; largest group: 1. Original fold sizes:[55, 55, 55, 55, 55]; other-four-fold fit sizes:[220,220,220,220,220]. Declared grouping columns: none.

Development file SHA-256: `cc9e242f3aa638cbaf2fc6951c37631b5928239aaa0c1dba03cc176629d0a4ed`. Assignment SHA-256: `81a892fa47114e6462ad030e74c1dad1e3936fdd3c0aadfa8cacd354db82b332`. Original source SHA-256: `416c13c9349eba384c5af4711f77a7761ffebcef23cb7e780f65172827b29087`. Original license: unknown; repository license: MIT.

Frozen class order: `["0", "1", "2"]`. Development counts: `{"0": 121, "1": 55, "2": 99}`. Counts by original fold: `{"0": {"0": 25, "1": 11, "2": 19}, "1": {"0": 24, "1": 11, "2": 20}, "2": {"0": 24, "1": 11, "2": 20}, "3": {"0": 24, "1": 11, "2": 20}, "4": {"0": 24, "1": 11, "2": 20}}`.

| Feature | Type | Missing | Distinct nonmissing | Minimum | Median | Maximum | Mean | Population SD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| island | categorical | 0 | 3 | n/a | n/a | n/a | n/a | n/a |
| bill_length_mm | numeric | 2 | 150 | 33.1 | 44.5 | 59.6 | 43.959341 | 5.5528954 |
| bill_depth_mm | numeric | 2 | 75 | 13.1 | 17.3 | 21.2 | 17.141026 | 1.9388646 |
| flipper_length_mm | numeric | 2 | 55 | 172 | 197 | 231 | 200.89744 | 14.092984 |
| body_mass_g | numeric | 2 | 89 | 2700 | 4050 | 6300 | 4188.4615 | 815.6716 |
| sex | categorical | 11 | 2 | n/a | n/a | n/a | n/a | n/a |
| year | numeric | 0 | 3 | 2007 | 2008 | 2009 | 2008.0218 | 0.81768874 |

Exact development category frequencies:

- `island`: `{"0": 135, "1": 100, "2": 40}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `sex`: `{"0": 130, "1": 134}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.

<a id="c-datasets--tae"></a>

### A.4.10 tae

Task `multiclass`; metric `log_loss`; 124 development rows; 27 sealed rows counted from assignments; 3 numeric and 2 categorical features. Development groups: 86; all groups: 106; largest group: 4. Original fold sizes:[20, 25, 27, 30, 22]; other-four-fold fit sizes:[104, 99, 97, 94,102]. Declared grouping columns: none.

Development file SHA-256: `96b5a504b8ee490a66b5d4b01f8fed9f7753885bfc0b49e17c42099cc441517e`. Assignment SHA-256: `7e83b2b19fc5acb93d2afc04ce0a6430f5609b7380c806978d603562186e08ea`. Original source SHA-256: `5eb60cdd947ecb4ea476ba854299dbe18d522df3ed684fb8f60657fdd4132910`. Original license: unknown; repository license: MIT.

Frozen class order: `["1", "2", "3"]`. Development counts: `{"1": 41, "2": 40, "3": 43}`. Counts by original fold: `{"0": {"1": 6, "2": 10, "3": 4}, "1": {"1": 9, "2": 9, "3": 7}, "2": {"1": 10, "2": 5, "3": 12}, "3": {"1": 10, "2": 9, "3": 11}, "4": {"1": 6, "2": 7, "3": 9}}`.

| Feature | Type | Missing | Distinct nonmissing | Minimum | Median | Maximum | Mean | Population SD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Whether_of_not_the_TA_is_a_native_English_speaker | categorical | 0 | 2 | n/a | n/a | n/a | n/a | n/a |
| Course_instructor | numeric | 0 | 24 | 1 | 13 | 25 | 13.822581 | 6.9607753 |
| Course | numeric | 0 | 23 | 1 | 3.5 | 25 | 8.0806452 | 6.9683579 |
| Summer_or_regular_semester | categorical | 0 | 2 | n/a | n/a | n/a | n/a | n/a |
| Class_size | numeric | 0 | 42 | 5 | 27 | 66 | 28.193548 | 12.968123 |

Exact development category frequencies:

- `Whether_of_not_the_TA_is_a_native_English_speaker`: `{"1": 23, "2": 101}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.
- `Summer_or_regular_semester`: `{"1": 19, "2": 105}`. Categories absent from the other four folds: `{"0": [], "1": [], "2": [], "3": [], "4": []}`.

<a id="c-datasets--explicitly-omitted-or-unknown-statistics"></a>

## A.5 Explicitly omitted or unknown statistics

Sealed target values, target distributions, class balance, feature distributions, missingness and category cardinalities are deliberately unread. Their row counts and expected file identities come from frozen metadata only. Raw full-dataset distribution statistics are therefore unavailable; development summaries must not be labeled full-dataset summaries. Checkpoint dataset exposure, unrecorded subject/time identifiers and original licenses marked unknown remain unknown. The broad S3 catalog is a stored metadata inventory, not fresh payload validation, admissible-teacher inventory or a count of trained models. Exact current panel ranks, per-category wins and Elo await the unchanged panel, whose matrix has no completed cells as of 2026-09-23 about 23:55 UTC ([§](#c-targets--retained-snapshots-and-plot-data)). Final cold installation, final warm target performance, actual GPU training hours and trained-checkpoint consumption await their own receipts. Under the owner's family-level domains (teacher/train, adaptive development, and evaluation outside training and selection), these thirty datasets remain adaptive-development data ([§](#c-benchmark--adaptive-development-exposure)); whether a pretrained encoder was exposed to them also remains unknown.


---

<a id="c-training-history"></a>

# Appendix B — Historical training, checkpoint lineage and original loss trajectories

The supplied CSVs retain original values, source file SHA-256, source path and line, and JSON Pointer where appropriate. There is no interpolation. Separate figures show different populations; joining them into one curve would erase changes in architecture, objective, corpus, sampler and validation protocol. The later recovery lineage has an independently audited, complete retained series of 63,000 per-step observations; the older 108-block capture retains only block endpoints.

<a id="c-training-history--dated-lineage-and-evidence-coverage"></a>

## B.1 Dated lineage and evidence coverage

| Observation or historical interval | Retained state | What is established; what is not |
|---|---|---|
| Jope Prime gen31, preserved by the provenance inventory dated 2026-07-12 | Fourteen complete epoch-summary rows, explicitly `policy_v3.2`; immutable warm-start checkpoint identified as `6a204434…`; 125,970 training and 2,169 validation rows | Real logged training and validation measurements. The log itself has no wall-clock timestamp or exact trainer-source hash; its event date is not inferred from file modification time. The provenance inventory's date is a historical declaration, not a newly verified execution date. |
| Lineage A, historical xbabe2 Python control plane | Four surviving frozen corpus snapshots and original warm-start state; substantial live control-plane state and later checkpoint descendants declared lost | Historical training and teaching data existed. The lost accepted-pointer and sealed-use history cannot be reconstructed merely from aggregate scores. |
| September 9–11 native training audit | 108 completed 3,000-step blocks, 108 different corpus identities and 62 starting checkpoint identities | Original initialization records explicitly reset AdamW. The capture retains each block's first and last metric row, not its complete per-step log. |
| September 12–13 recovery lineage (lineage B, [§](#c-data-estate--lineages-a-and-b-and-the-staging-store)), retained A100 metrics and checkpoint manifests | 63,000 complete metric rows at steps 6,001–69,000 in 21 blocks/corpora; 78 checkpoints at steps 4,000–69,000 across 22 corpus identities | This later lineage carries FP32 AdamW ancestry. Complete logged losses accompany the metadata. The accepted serving checkpoint remains 60,000; 69,000 is unaccepted. Validation remains adaptive and its corpus/panel changes. |
| September 14, 22:19:52–22:27:47 UTC | Complete 100-row RTX qualification trajectory, steps 60,001–60,100, interrupted and resumed after 60,006 | Original metadata records carried optimizer state and changed model/parameter identities. It explicitly describes unchanged-corpus hardware qualification, with zero fresh programs: a historical hardware continuation, not a production improvement or promotion run. |
| September 18, 20:17:48 UTC publication manifest | Product bundle `policy-step-60000`, source checkpoint `8ac80c269cc47c253b6f53fa3bc1eb75bb2fcda0bdace3ff8554be0175c400d6` | The current product's accepted source checkpoint is pinned. Publication time is distinct from training time. |
| Current bounded ASM work, as of 2026-09-23 about 23:55 UTC | Verified one-dataset regression neural-origin proof at source `68f56404`, merged to main as `a75e543d2` (PR83); the training64 follow-on remains a stopped, unmerged patch (pending integration), and no eligible real corpus has been demonstrated for its admission guard ([§](#c-teachers--present-teacher-readiness-and-the-smallest-real-path)) | New category training and subsequent consumption of its learned export remain pending; no checkpoint has been trained by this work. No new GPU-hour budget is inferred from historical training. Experimental regression lineage v1 (pending integration) is not part of this history. |

Sources: `docs/adr/ADR-0005-canonical-learning-loop.md:12-24,72`, `docs/learning.md:280-285`; historical native audit `learning-hpc.md:37`; original checkpoint manifests and metrics pinned in the CSVs; `models/bundles/policy-step-60000.json:7,73`; `/home/ubuntu/jope-prime/provenance.json:3,13,129`. The retained legacy source declares `policy_v3.2` at `/home/ubuntu/jope/src/jope/policy/model_v3.py:40`; it is useful for interpreting the old log, but the log does not attest that exact source build.

The accepted serving model is the native 512×6 decoder at step 60,000 ([§](#c-architecture--checkpoint-binding-and-configuration)). Experimental regression lineage v1 (256×2, new initialization) is the separate line that the pending bounded campaign would start: a Python `JopePolicyModelV3` with width 256 and two layers, a new, smaller model family rather than a continuation of step 60,000 ([§](#c-training--what-the-pending-campaign-trains)). Status at the PR83/stopped-overlay snapshot: pending integration; the training64 patch was then outside main. PR88 later integrated it ([§](#c-status--implementation-snapshot-and-this-manuals-own-identity)); no real campaign checkpoint is established. Its initial model `M_init` is named `M0` only after it has actually trained on its first corpus `C0`; as of 2026-09-23 about 23:55 UTC it has no trained checkpoint. None of the records in this appendix belongs to it, and its measurements must not be joined to these historical trajectories.

<a id="c-training-history--legacy-python-training-gen31-and-round-1"></a>

## B.2 Legacy Python training: gen31 and round 1

<a id="c-training-history--gen31-fitting-the-objectives-versus-using-the-context"></a>

### B.2.1 Gen31: fitting the objectives versus using the context

<a id="c-training-history--epoch-losses-and-the-routing-warning"></a>

#### B.2.1.1 Epoch losses and the routing warning

The gen31 log contains epochs 1–14, with phase A at epochs 1–6 and phase B at 7–14. Both phases train the ranking branch. The retained v3.2 trainer lowers the context-encoder learning rate at the phase boundary, while phase B adds reward-weighted sequence imitation. It logs two GPUs, requested batch size 2,048 and learning rate 0.00015. Its final elapsed timer is 1,135.158 seconds; that timer is a run-local measurement, not a cold-install or end-to-end campaign duration.

![Fourteen original gen31 epoch summaries, including training and validation components](figures/training-gen31-loss.png)

Every point is an original epoch summary. The training sequence component (`train_loss_components.seq_nll`) fell from 1.563002 to 0.918725, but 0.466185 of that fall occurred in one step at the phase A→B boundary (1.473704 at epoch 6, 1.007519 at epoch 7), where phase B adds reward-weighted sequence imitation; within each phase it fell by only about 0.089. Held-out sequence NLL (`val_seq_nll`, no label smoothing) rose in every phase-A epoch and overall from 2.055572 to 2.182772 (+6.2%). The full training composite fell from 2.926419 to 2.022394, while validation composite changed from 7.039105 to 6.926823 and reached its minimum of 6.671162 at epoch 3. The final epoch therefore was not the best by every monitored objective. The train/validation composite gap grew from 4.112686 to 4.904429. The model fitted its training objective while held-out likelihood got worse; these measurements do not show that a newly generated algorithm outperformed its ancestor. The figure's validation sequence line is the 0.05-smoothed `val_loss_components.seq_nll`, not `val_seq_nll`.

Routing is a particularly visible warning: training routing cross-entropy fell from 0.642189 to 0.005816, while validation routing cross-entropy remained 3.498959 at the end. The retained labels and grouping still need to be held constant before assigning causality, but a near-zero training classifier loss alone is plainly insufficient evidence of useful routing.

<a id="c-training-history--context-advantage-and-ranking-diagnostics"></a>

#### B.2.1.2 Context advantage and ranking diagnostics

![Gen31 recorded context advantage and ranking diagnostics](figures/training-gen31-quality.png)

Pairwise validation ordering accuracy rose from 0.533419 to 0.538499. The genome-only control rose from 0.501411 to 0.503346, so recorded conditional advantage changed from 0.032008 to 0.035153: approximately 3.20 to 3.52 percentage points. This is a modest ranking diagnostic on retained candidates, not a 53.8% downstream tournament win rate. Historical top-1 regret decreased from 0.017565 to 0.016676 in the retained target's units. Predicted-reward correlation remained small, 0.036390 to 0.025889. The uncertainty/error statistic was negative, −0.196674 to −0.228137; it does not establish calibrated uncertainty.

<a id="c-training-history--metric-definitions-and-the-v32-objective"></a>

#### B.2.1.3 Metric definitions and the v3.2 objective

The following definitions explain why similarly named numbers cannot be substituted for one another:

| Logged metric | Source meaning and limitation |
|---|---|
| `train_loss`, `val_loss` | Combined objectives, including learned uncertainty weights and regularizing log-variance terms. They are not sums of the plotted raw component values. Training aggregation averages executed batch losses across ranks. |
| `train_loss_components.seq_nll` / `val_loss_components.seq_nll` | Legal-token, 0.05-smoothed sequence cross-entropy with applicable provenance/phase row weights. |
| `val_seq_nll` | A separate validation evaluator uses zero label smoothing and averages batch NLL. It is deliberately distinct from `val_loss_components.seq_nll`; the first epoch logs 2.055572 versus 2.175046. |
| Pairwise ordering accuracy | Agreement on unequal-value pairs within a dataset. This implementation does not give explicit half-credit to tied predictions; its Boolean comparison has an order-dependent tie convention. |
| Conditional advantage | Conditional reward ranking accuracy minus the genome-only control's accuracy on the same retained population. It probes context use; it does not fit new programs. |
| Historical top-1 regret | Per dataset, best retained target value minus the target value of the retained program selected by predicted reward, averaged over datasets. It is not regret relative to an unknown optimum. |
| Reward CV correlation | Pearson correlation with retained `robust_delta` labels. The name does not make it a new downstream cross-validation score. |
| Head accuracy | Match at the first head-role grammar token under teacher-forced decoding, grouped by engine. It is not fitted-prediction accuracy. |
| Uncertainty/error Spearman | Correlation between sorted positional ranks of ensemble dispersion and absolute reward error. The retained implementation does not average tied ranks, so preserve the implementation identity when comparing it. |

The old v3.2 objective has seven components: sequence NLL, reward Bradley–Terry, reward listwise, runtime Huber, stability BCE, ablation Bradley–Terry and routing cross-entropy. Its profile scales are respectively 1, 1, 0.5, 0.2, 0.5, 1 and 0.5. Reward/ranking/ablation/routing terms use fixed profile scales; sequence, runtime and stability use the retained Kendall–Gal-style learned weighting. The log's `loss_weights` values are `exp(-log_var)`, not complete effective coefficients. This differs materially from the five-term native objective below. Source interpretation is grounded in retained v3.2 code; absence of an exact run-source binding remains explicit.

Sources: original `train_metrics.jsonl` SHA `d57de1b28a8678327c984a82732cd774520db2cb0994d893a45f8a5ea428e312`, lines 1–14; `/home/ubuntu/jope/src/jope/policy/train_v3.py:360,364,438,444,474,580,613`; `policy/model_v3.py:40,286,300,305,372,376`; `policy/losses.py:23,35,48,86,100`; `model.py:981,997,1015`. Full hashes and exact definitions are retained in this chapter's metadata.

<a id="c-training-history--round-1-and-the-reported-self-play-ladder"></a>

### B.2.2 Round 1 and the reported self-play ladder

<a id="c-training-history--round-1-epochs-and-the-gate-receipt"></a>

#### B.2.2.1 Round-1 epochs and the gate receipt

Four original `round1.train_metrics.jsonl` rows survive for the lineage-A Python model `policy_v3.3-fused-heads`. They log phase A at epochs 1–2 and phase B at 3–4, with learning rate 0.0005. This is a separate model/run from gen31 and the native recovery lineage; their composite losses must not be joined into one trajectory.

| Logged measure | Epoch 1 | Epoch 4 | Interpretation |
| --- | ---: | ---: | --- |
| Training composite | 5.191046 | 3.212379 | Better fit to the run's changing phase objective |
| Validation composite | 9.080722 | 8.241397 | Minimum 7.938042 at epoch 2; the final epoch is worse than that minimum |
| Validation minus training composite | 3.889675 | 5.029017 | The recorded gap widens; phase changes limit a simple overfitting interpretation |
| Validation pairwise accuracy | 0.577940 | 0.624193 | A ranking diagnostic, not downstream predictive performance |
| Validation sequence NLL | 1.686433 | 1.773742 | The final value is worse, despite the lower composite |

A separate gate receipt dated **2026-07-25 10:24:49.812 UTC** compares 557 validation rows per model: pairwise accuracy 0.534717 → 0.663611, sequence NLL 2.110434 → 1.838604, and reward/CV correlation −0.097427 → 0.445109. It records `pass: true` and explicitly says “gate only; nothing deployed or rotated.” These are real retained gate measurements, not proof of current teacher eligibility, an independent unseen-family evaluation, or a subsequent serving promotion. The gate's timestamp does not supply a missing event timestamp for every epoch.

<a id="c-training-history--the-reported-generation-level-loss-and-internal-elo"></a>

#### B.2.2.2 The reported generation-level loss and internal Elo

![Legacy round-1 epoch measurements and reported generation-level loss and internal Elo; the generation panels reproduce reported paper summaries, not a replayed tournament](figures/training-legacy-round1-generations.svg)

The two lower panels retain the **reported paper CSV summaries** from `/home/ubuntu/jope-prime/docs/paper/data/`, rather than replaying their historical tournaments. The loss table has generations 1–31; the internal self-play Elo table has generations 1–30. Across the loss summaries, validation sequence NLL changes from 1.783587 to 2.084099, and pairwise accuracy from 0.594856 to 0.529146. Internal top-1 Elo changes from 1,073.2 to 1,119.2. Over their thirty common generations, the descriptive Pearson correlation between Elo and sequence NLL is +0.728336: higher internal Elo coexists with worse likelihood. Correlation across this evolving history is not a causal result, and the underlying populations/objectives have not been frozen for a new matched evaluation.

The generation-31 summary exactly matches **epoch 3**, not epoch 14, on validation sequence NLL, validation pairwise accuracy and the training sequence component. That explains why a generation-level table and the final gen31 log row can disagree without either number being fabricated. There is no generation-31 Elo row in this retained table. Neither the internal ladder nor its `shipped` column establishes a current release or the benchmark-valid Elo described in [§](#c-targets--historical-benchmark-valid-elo). Full source-to-tournament provenance and exposure qualification remain separate from this arithmetic/source-file check.

The companion `data/legacy-training/` retains all four original files byte-for-byte, a hash-pinned recomputation receipt and the plot input/output hashes. Their source paths are `/home/ubuntu/jope-flywheel/training/round1/{round1.train_metrics.jsonl,gate-result.v1.json}` and `/home/ubuntu/jope-prime/docs/paper/data/{loss_components_by_generation.csv,elo_trajectory.csv}`. No model or dataset payload was opened to assemble this supplement.

<a id="c-training-history--native-training-the-108-block-audit-and-the-a100-recovery-lineage"></a>

## B.3 Native training: the 108-block audit and the A100 recovery lineage

<a id="c-training-history--the-september-11-native-objective-and-108-reset-blocks"></a>

### B.3.1 The September 11 native objective and 108 reset blocks

<a id="c-training-history--native-architecture-and-objective"></a>

#### B.3.1.1 Native architecture and objective

The audited native decoder consumes a 1,536-dimensional dataset vector and a 512-dimensional target vector. The configuration uses width 512, six decoder layers, four attention heads, FFN width 2,048, dropout 0.1, 401 vocabulary tokens, 657 embedding rows and maximum sequence length 128. The reward path has separate context processing and four reward heads. These are the observed historical dimensions, not a claim that every historical model version used identical architecture.

The active native objective was:

`L = sequence_nll + 0.25 × ranking_bt + 0.10 × ranking_listwise + 0.10 × runtime_huber + 0.05 × contextfree_bt`.

Sequence loss is legal-token cross-entropy with 0.05 smoothing and normalized program weights `0.05 + percentile^4`. Bradley–Terry uses weighted within-comparison winner/loser pairs. Listwise uses a `softmax(percentile / 0.25)` target; its nonzero entropy floor matters when interpreting an apparent plateau. Runtime loss compares `log1p` seconds. The program-only ablation term checks whether ranking can be achieved without dataset context. Stability and the seven stored log-variance parameters were inactive in this native build. Seven parameter names do not imply seven optimized losses.

<a id="c-training-history--endpoint-measurements-and-parent-regimes"></a>

#### B.3.1.2 Endpoint measurements and parent regimes

![First and last sampled minibatch measurements for each of 108 reset training blocks](figures/training-historical-endpoints.png)

The audit's original capture includes exactly two metric records per block: first and last, plus the original full log's path, SHA and reported 3,000-row count. The intermediate 2,998 rows per block are not reconstructed. These endpoints are sampled minibatches, not averages over the whole block. All 108 blocks reset AdamW, so their 324,000 summed steps are **block work**, not one optimizer's age. At global batch 2,048, the corresponding 663,552,000 nominal program-example presentations are not unique programs or discoveries. Replayed or uncommitted work outside this completed block census is not silently included.

The starting checkpoint changes in three regimes. Cycle 0 starts from the earlier campaign checkpoint `20dca670…`. Cycles 1–60 warm-start their model from the preceding block's checkpoint, while still resetting AdamW. Cycles 61–73 repeatedly start from cycle 59 (`61ba6043…`), and cycles 74–107 repeatedly start from the recovery checkpoint `c2f229c9…`. These are model-parent relationships, not optimizer-state continuation.

All 34 first-to-last reductions in composite, sequence NLL, Bradley–Terry and listwise loss occur exactly in cycles 74–107, the recovery-parent regime. Every block in cycles 0–73 has a higher final sampled value for those four losses than its first value. Runtime Huber decreases exactly in cycles 78–107. The counts therefore align with the parent regime; they are not scattered improvements across 108 comparable starts. The visible jumps at cycle 74 coincide with the rollback to the recovery parent and must not be interpreted as a continuous learning event. Endpoints still sample different minibatches on different corpora, so this is descriptive evidence, not a paired significance test or causal isolation of the parent change. The last block's training NLL was 0.561710, while its adaptive validation NLL was 1.410741; that gap is not a sealed generalization estimate.

<a id="c-training-history--adaptive-validation-observations"></a>

#### B.3.1.3 Adaptive validation observations

![Adaptive validation observations from changing corpus and panel identities](figures/training-adaptive-validation.png)

The plotted validation points are deliberately not a continuous generalization curve. Dashed boundaries identify the repeated-cycle-59 regime from cycle 61 and the recovery-parent regime from cycle 74. The validation Bradley–Terry jump near cycle 74 coincides with that rollback; it is not evidence of smooth accumulated improvement. Each source identity travels with its value. Old native telemetry exposed total loss and four component losses but omitted the active ablation component; its contribution cannot be inferred exactly by subtracting rounded or differently aggregated values. Per-term gradient norms were also unavailable. The later RTX log does expose `contextfree_bt` and weighted terms, so that historical limitation must not be presented as universal.

Sources: `docs/legacy/root-docs/audits/worldclass-redteam-20260911/learning-hpc.md:9,21,33,37`; retained `evidence/learning-native-v3/neural.rs:447,512` and `training.rs:692,712,720,863`; research capture `/home/ubuntu/jain-split/jope-dime/docs/audits/worldclass-redteam-20260911/learning-remote-metadata.json` SHA `1d2ab6ad9a8df4749bd1da000c6b1ea04d00395a777199907f3b599ad9af528b`.

<a id="c-training-history--fixed-monitoring-and-actual-promotion-evidence"></a>

### B.3.2 Fixed monitoring and actual promotion evidence

![Fixed-corpus and fixed-panel historical checkpoint comparison](figures/training-fixed-monitor.png)

The fixed monitor is the stronger loss comparison in this audit: all records bind corpus `11e60a43…`, panel `fbff7a048…` and 2,048 sampled examples. Recovery was evaluated by both relevant executable builds with identical recorded outputs. Cycle 25 can be paired with recovery under executable `8ca21024…`; cycle 107 can be paired with recovery under `3382cdcb…`. The CSV retains both recovery rows even where the visual combines their identical values.

| Checkpoint | Sequence NLL | Bradley–Terry | Listwise | Runtime Huber |
|---|---:|---:|---:|---:|
| Recovery checkpoint `c2f229c9…` (step 3,000) | 1.394498 | 0.533676 | 2.262484 | 0.076231 |
| Cycle 25 | 1.396643 | 0.659752 | 2.485061 | 0.065244 |
| Cycle 107 (one reset block started from recovery) | 1.408606 | 0.542086 | 2.277948 | 0.073749 |

Sequence and reward-ranking losses did not improve over recovery in these fixed-monitor observations; runtime error was lower. This is an objective tradeoff, not an overall win. The monitor explicitly marks itself adaptive teaching validation, not independent evaluation; `training_selection_used` is false, but that does not turn previously used teaching labels into sealed data.

All 48 promotion decisions indexed in the September 11 capture were **inconclusive**, with qualification evidence not supplied. This is not a measured 0/48 challenger win rate. It establishes missing evidence at that historical observation. The later accepted 60,000 checkpoint belongs to a subsequent recovery lineage; the September 11 result must not be extended to claim that no later checkpoint was ever accepted. A full joined chain of matched executable comparisons and successive qualified target improvements is still required for a quantitative “the flywheel kept improving” claim.

<a id="c-training-history--the-complete-a100-loss-series-steps-600169000"></a>

### B.3.3 The complete A100 loss series: steps 6,001–69,000

<a id="c-training-history--source-and-extraction"></a>

#### B.3.3.1 Source and extraction

A second retained source supplies the complete per-step loss series for this lineage. Original `metrics.jsonl` files in cycles `000001` through `000021` of `eight-a100-recovery-20260912/learning`, under `/home/ubuntu/jope-migration-evidence/claude/node-high-value/files/home/ubuntu/jope-dime/parallel/runs`, contain **63,000 contiguous reported optimizer steps**, exactly 3,000 per block. Their embedded timestamps span September 12, 18:43:27 through September 13, 18:09:49 UTC. These dates come from the rows, not filesystem modification time. All rows record rank 0, world 8, microbatch 256, accumulation 1 and global batch 2,048. Every row binds the block's corpus and a carried FP32 AdamW parent; parent identities match the retained checkpoint lineage. A100 attribution is historical run lineage, not a new device probe.

The compressed `data/per-step-history/a100-step-metrics.csv.gz` preserves all 63,000 rows and **378,000 exact numeric metric tokens**: `loss`, `sequence_nll`, `ranking_bt`, `ranking_listwise`, `runtime_huber` and `gradient_norm`. Step, local step and timestamp tokens are also preserved. Cycle plus one-based `source_line` resolves each value to a pinned original log in `source-manifest.json`. This is a selected-column extraction, not a replacement for all raw telemetry: timing breakdowns, counters and other nested fields remain in the source logs. A second read compares every selected token against its original and rechecks each full log hash.

<a id="c-training-history--within-block-and-boundary-patterns"></a>

#### B.3.3.2 Within-block and boundary patterns

![Complete retained A100 history: 100-step training averages (upper) and adaptive checkpoint validation on changing panels (lower); the lower panel's decline mixes learning with panel changes and is not a generalization curve](figures/training-a100-per-step.svg)

The upper panel shows 630 **non-overlapping 100-step arithmetic means**, with a separate line per corpus; it does not display every noisy minibatch point or interpolate absent data. The underlying CSV retains every selected point. The lower panel shows the original 78 checkpoint sequence-NLL observations without joining changed evaluation populations. Dashed orange marks the accepted step 60,000; shaded later steps remain unaccepted.

The within-block pattern is consistent, but its interpretation is bounded. All 21 composite-loss OLS slopes are negative, all 21 final sampled losses are below their first samples, and all 21 last-100 means are below their first-100 means. The median last-100 minus first-100 change is **−0.017066**, with range **−0.043771 to −0.008792**. Every single-point block boundary jumps upward; 19 of 20 boundary comparisons also rise when comparing adjacent 100-step means. Thus “sawtooth” describes the recorded pattern, not a claim that every individual optimization step improves. Each of the 21 blocks has a distinct corpus. Changed sampling and teaching data prevent attributing a boundary jump, or an across-block decline, to parameters alone.

| Block and optimizer range | First sampled loss | Last sampled loss | First-100 mean | Last-100 mean |
|---|---:|---:|---:|---:|
| 1: 6,001–9,000 | 0.855045 | 0.792636 | 0.831595 | 0.787825 |
| 18: 57,001–60,000, accepted endpoint | 0.819152 | 0.785951 | 0.801961 | 0.790811 |
| 21: 66,001–69,000, unaccepted endpoint | 0.830200 | 0.792011 | 0.809123 | 0.797115 |

A descriptive OLS over all 54,000 observed steps from 6,001 through 60,000 has slope **−3.3860 × 10⁻⁷ loss per step**, R² **0.1885**, and fitted first-to-last change **−0.018284**. These are arithmetic summaries of serially dependent observations from changing corpora, not confidence estimates or a measured generalization gain. `a100-block-statistics.csv` retains all 126 block/metric summaries, including each slope and endpoints; `a100-corpus-boundaries.csv` makes the boundary comparisons explicit.

<a id="c-training-history--telemetry-limits-and-the-recovery-boundary"></a>

#### B.3.3.3 Telemetry limits and the recovery boundary

The composite loss is retained as reported. These 63,000 rows expose four named objective components and gradient norm, but omit top-level `contextfree_bt`, effective objective weights and per-step learning rate. Absence of a telemetry field does not establish that its objective was inactive. Do not fill it with zero, reconstruct the full objective from an incomplete component list, or directly compare its composite scale with the separately recorded RTX qualification run. Loss reduction is evidence of objective fitting; it does not by itself establish better encoding-conditioned GP proposals or cheaper, better fitted algorithms.

The inspected cycle-`000000/train-2` directory has checkpoint/run metadata but no `metrics.jsonl`: steps 3,001–6,000 therefore lack per-step rows **in this inspected source**. The antecedent steps 1–3,000 are outside this per-step source set. This is a precise recovery boundary, not a claim that no other log could exist. It also does not erase the earlier fixed 2,048-example monitor: that separate recovery/cycle-25/cycle-107 comparison remains in the preceding section. The later 78-checkpoint series spans 22 corpora and 21 panels and does not provide one unchanged evaluation population.

<a id="c-training-history--carried-state-accepted-step-60000-and-rtx-step-60100"></a>

### B.3.4 Carried state, accepted step 60,000 and RTX step 60,100

<a id="c-training-history--a100-checkpoints-and-the-accepted-step-60000"></a>

#### B.3.4.1 A100 checkpoints and the accepted step 60,000

![Original A100 checkpoint validation measurements, with accepted step 60000 marked](figures/training-a100-checkpoints.png)

The 78 retained A100 state manifests live under `/home/ubuntu/jope-node-mirror/weights/runs/eight-a100-recovery-20260912/learning/cycle-*/train-2/checkpoints/step-*/state.json`. The retained metadata covers 78 checkpoints, all with “carried FP32 AdamW with bias-correction step.” This is materially different from the 108 reset blocks. It spans 22 corpus identities and recorded steps 4,000–69,000. The graph keeps these as adaptive validation observations. A higher retained step is not automatically the serving model. The accepted 60,000 state's exact hash is `8ac80c26…`, corpus `1c840597…`, model `053df4f5…`, and parameter digest `9c7057e7…`; it records ancestry from step 57,000 and requests fresh-balanced sampling. Its request for 50% fresh comparisons does not mean 50% fresh program slots or measured downstream improvement. The block-18 per-step log ends at this exact step/corpus; block 19 explicitly starts from accepted state `8ac80c26…`. Of the 78 retained checkpoints, 74 also have validation entries inside the per-step logs; all 296 overlapping metric cells match the separately extracted checkpoint CSV. Sequence NLL is 1.428209 at step 4,000, 1.387687 at accepted 60,000, and 1.395417 at unaccepted 69,000; different adaptive panels limit this to descriptive observations. The product explicitly refuses the retained step-69,000 state/model identities rather than selecting the highest numbered checkpoint. The accepted state records no `execution_partition`, while the native trainer's exact resume requires one equal to the requested topology (`crates/jope-engine/src/training.rs:1106`), so step 60,000 cannot be resumed as the same experiment; [§](#c-training--native-lineage-b-the-held-command-and-its-blockers) describes the held continuation routes.

<a id="c-training-history--the-rtx-step-60100-qualification-run"></a>

#### B.3.4.2 The RTX step 60,100 qualification run

![All 100 original step measurements of the world-1 qualification run, a historical hardware continuation rather than a promotion, with the recorded resume boundary; the RTX 3090 attribution comes from ADR-0005, not from the run metadata](figures/training-rtx60100-steps.png)

The RTX run retains every metric record at steps 60,001–60,100. The exact 100-row source is `/home/ubuntu/jope-evidence/xbabe2-migration-20260914/accepted60000-restored/parallel/runs/xbabe2-hardware-qualification-20260914/final-runtime-v2-eof/train/metrics.jsonl`, distinct from the earlier recovery attempts. RTX 3090 attribution comes from ADR-0005; the retained run metadata itself records world size 1. Its first attempt starts from 60,000; later rows record initial optimizer step 60,006 after recovery. Therefore the graph uses optimizer step, not the run-local elapsed timer, which resets at resume. The timestamp span includes interruption/recovery and differs from the final attempt's 73.561-second elapsed value; neither should replace the other without labeling its scope.

The final state records **world 1 × microbatch 256 × accumulation 8 = global batch 2,048**. ADR-0005 at `a75e543d2` describes this run as microbatch 32 × accumulation 8; the retained state corrects that. Documentation-correction PR86 (head `0079ceddb`), merged at 2026-09-24T00:04:12Z as main `2b375ef11`, appended the same correction to ADR-0005 as its “Implementation correction — 2026-09-23” section without rewriting its historical decision. Model digest changes from `053df4f5…` to `1f202cce…`; parameter digest changes from `9c7057e7…` to `762094a0…`; optimizer identity also changes. The final state is `745be4b5e998bce85615349a5b03b901cd8ad2cf605e6dce3b86fbf777cc7388`. These are retained training-update and continuation evidence, not a fresh numerical replay performed for this manual.

The receipt commits 204,800 program presentations and zero fresh programs. Its purpose is explicitly “isolated unchanged-corpus hardware qualification, not production continuation outcomes.” Final adaptive validation NLL is 1.409792 and Bradley–Terry 0.475421, on 256 examples. Step 60,000's retained validation used a different panel and 2,048 examples; comparing those numbers as a paired improvement would be invalid. The checkpoint is not promoted by being 100 steps newer. Training further on one unchanged corpus is continuation, not a new generation. The recorded interruption and recovery after step 60,006 is also not evidence of an exact interrupted resume as the owner now defines it: that must reject optimizer fallback before any update and prove that the interrupted and uninterrupted runs reach the same state, and this manual has no such parity comparison for this run.


<a id="c-training-history--current-publication-regeneration-and-provenance"></a>

## B.4 Current publication regeneration and provenance

Edition 3 regenerated the seven retained-history figures and the 63,000-step SVG from the
same authenticated observations. `data/historical-plot-provenance.json` records each figure’s
source hashes, extraction/replay and plotting commands, coverage, units and interpretation
limits. `tools/reextract_training_metrics.py --output NEW_CUSTODY_DIRECTORY` reproduces the
3,218 original metric values and 78 checkpoint lineage rows using frozen row-selection metadata;
it does not overwrite original records or invent missing observations. The A100 per-step
extraction command remains in `data/per-step-history/README.md`. This rebuild rechecked all
21 original A100 logs and all 82 retained historical sources. For the 108-block history,
the original remote logs are represented only by their retained endpoint capture and recorded
hashes; this is not a new independent recovery of those remote logs.

All 63,000 step rows and 378,000 six-metric tokens matched their retained original logs.
These are metadata reconstruction and plotting checks, not optimizer execution or new
measurements. The new observed-transferable-lift KPI remains **not yet measured**
([§](#c-learning--observed-transferable-lift-reporting-contract-not-yet-measured)).


---

# Appendix C — Historical GP targets, rankings, Elo and attribution limits

This chapter answers two questions: which historical corpora and target records survive, and how the historical target collections and their observed scores changed. What one training example contains is described in [§](#c-teachers--a-dataset-is-not-one-training-example), and how the implementation stores and loads packed records in [§](#c-data-estate--record-layout). This manual's audit read metadata, manifests and development-score receipts. It did not read sealed labels, model tensors, packed target-record bodies or the large audit-label sidecars, and it ran no training or fits. External packing, inventory and seed audits report results that depend on payloads this audit did not open; those results are attributed separately below and were checked here through retained metadata, scripts and arithmetic, without replaying the payload reads. File-metadata observations, externally reported payload checks and source-derived calculations remain distinct evidence levels.

<a id="c-targets--did-gp-targets-get-better"></a>

## C.1 Did GP targets get better?

<a id="c-targets--carried-forward-scores-and-the-admitted-receipt-inventory"></a>

### C.1.1 Carried-forward scores and the admitted receipt inventory

**Some recorded carried-forward best scores became lower.** These are historical development outcomes from searches sharing incumbent and teacher seeds. They are not independent discoveries by each arm, and their no-worsening pattern is partly structural. Corpus growth alone would not establish even this narrower result.

The audit enumerated 29,744 `search.json` paths under `/home/ubuntu/jope-node-mirror/results`, which resolves into `/home/ubuntu/jain-split/jain-datasets/.runtime/node-archive-mirror-20260914/results`. This source covers generations 0–70. The audit rejected 3,387 paths outside the accepted context index's `meta_train` dataset/cut allowlist **before opening their score files**. The remaining 26,357 receipts identify adaptive development-validation RMSE, prepared data, vocabulary, evaluator, execution protocol and completion metadata. Each source file and observation ledger has a distinct recorded digest. Their recorded completion times span September 12, 2026 18:37:13 UTC to September 14, 2026 13:37:43 UTC. These times are receipt metadata, not a new replay of the fits.

Only 3,530 of the 26,357 permitted receipts are also present in the separately preserved `accepted60000-restored` tree, all byte-identical; the other 22,827 receipts, including every permitted generation 10–70 receipt, are absent from that tree. A custody archive preserves all **26,357 original search receipts**, totaling **62,271,855 uncompressed bytes**. A separate streaming verification checked every member name, size and original SHA256 without reopening the source bodies. `original-target-receipts-custody.json` pins the archive and verification. The archive is a second local copy on the same filesystem, not an off-host backup. The separate seed bodies used by the external carry-forward analysis are outside that archive.

<a id="c-targets--first-to-last-comparison-at-1000-evaluations"></a>

### C.1.2 First-to-last comparison at 1,000 evaluations

For a readable comparison, the table restricts each repeated comparison to exactly 1,000 completed evaluations, matching its recorded dataset/family, cut, prepared SHA, vocabulary SHA, evaluator SHA/protocol, execution-order protocol and run. It compares the first retained generation with the last retained generation within that group. Lower RMSE is better. Improvement and worsening use exact `<` and `>` comparisons of the parsed reported scores; unchanged means exact equality, with no epsilon. The analysis script applies no additional cumulative minimum, but the underlying search already carries incumbent/teacher programs forward. Thus the reported per-generation best usually behaves like an inherited incumbent rather than a fresh independent draw, but it is not strictly monotone. In the 9,536 adjacent 1,000-evaluation transitions where the previous best was re-seeded, the recorded best stayed equal 9,406 times, fell 130 times and never rose; all 27 rises occur in the 76 transitions where it was not re-seeded. The score field is `best_validation_rmse` under the historical `featmath_training_fold_normalized_v1` evaluator; the plots do not claim original target units.

| Historical search arm | Repeated arm groups | Last lower | Same | Last higher | Median relative reduction, all groups (unchanged included) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Neural | 51 | 19 | 32 | 0 | 0% |
| Generic | 54 | 22 | 32 | 0 | 0% |
| Shuffled | 54 | 23 | 31 | 0 | 0% |

![First-to-last carried-forward best scores at 1,000 evaluations; the arms share seeds. The 64 lower arm groups cover 23 distinct dataset/cuts. No arm group ended higher, so the “Last higher” legend entry has no bar.](figures/target-quality-progress-counts.png)

The 159 arm groups cover **55 distinct dataset/cut pairs**. The 64 lower arm groups (19 + 22 + 23) cover only **23 distinct pairs**; 22 of those 23 end at the same score in every available arm. These counts were independently recomputed from the published comparison CSV. The denominators and shared seed pools make the arm counts dependent; one discovery can later appear in multiple arms.

The separately released external seed audit reports `seed_population_supplied=true` and `teacher_first_v1` for all 26,357 receipts. The previous recorded best appears in the next seed set in 25,790 of 25,985 transitions; 26,051 reported winners (98.84%) are already in that search's seed population. Incumbent, retained-teacher and exploration seed sets are identical across arms in all 8,785 multi-arm dataset/cut/generation groups checked. Those seed-body counts are externally reported and metadata-checked here, not a new local seed-payload replay. They explain why zero first-to-last worsening is not independent evidence of stability. A lower inherited score is also not a new discovery by the receiving arm. `carry-forward-audit.json` preserves these evidence boundaries.

This table is therefore not a paired policy-ranking result. Random seeds change between generations, elapsed time varies, and this audit did not requalify the actual CPU/memory/hardware enforcement or complete build closure of the historical searches. Matching these recorded fields is narrower than the complete current experiment identity. The recorded generation manifests declare 600 seconds and 1,000 evaluations for the illustrative example; those declarations do not establish equal consumed wall time. Checkpoint association is available by generation, but exact exposure of each improved target to later optimizer batches remains unjoined.

<a id="c-targets--the-pine-chromium-example-attribution-limits-and-published-data"></a>

### C.1.3 The pine-chromium example, attribution limits and published data

A concrete example is `mendeley-6bspjkt7bg-pine-chromium`, dataset `b68222ad60398ea8`, cut 1. Across generations 0–70 the recorded best development RMSE changes from **0.009749860799638275 to 0.003185684481133936**, a **67.3258% relative reduction**. All three arms finish at the same score. The neural arm first records that final value at generation 43; generic and shuffled do so at generation 42. The external seed audit identifies the final program outside the generic and shuffled seed sets at generation 42, then as an `incumbent_cut_teacher` seed in all three arms at generation 43. The neural arm's later equality is carry-forward, not evidence that it independently discovered that program. This identifies generation-level observations; it does not establish which of two same-generation arms discovered the program first in wall-clock time. The recorded best is not monotone here either: generic reports 0.007209 at generation 29 and 0.009750 at generation 30, and generic and shuffled both rise from 0.004698 at generation 39 to 0.006747 at generation 40; the neural arm first reports 0.007209 at generation 39. The plot data include all 213 generation/arm pairs, with no interpolated missing generation. This example was selected after inspection as the largest first-to-last relative improvement in the neural 1,000-evaluation subset; it is illustrative, not representative.

![Post-hoc pine-chromium cut 1 history with shared teachers: generic/shuffled record the final program outside their seeds at generation 42; all arms inherit it at generation 43.](figures/target-quality-pine-example.png)

Relative improvement is `(first_RMSE - last_RMSE) / first_RMSE`. The percentage is meaningful within that dataset/cut. Raw RMSEs from unlike target scales are not averaged to claim overall progress. A running-best curve, if separately shown, must be labelled as such: its monotonicity follows partly from selection. Better retained minima do not prove that a trained neural proposal policy caused the gain. Shared incumbent/teacher seed pools and carry-forward are established in the external audit; search history and ordinary exploration also prevent attribution to policy learning. Count a new search discovery separately from a copied incumbent: authenticate the full seed set, identify the first measured program absent from it, and retain the discovering arm, generation, context, program and parent identities. The aggregate table above does not provide that full per-program discovery ledger. The owner's acceptance rules (proposed) keep four claims distinct: an eligible teacher, a new discovery, information actually consumed by training, and an improved policy. Initial populations must be authenticated, and a discovery must not silently seed another arm. The historical records in this appendix support at most lower retained development scores and, for single programs such as the pine-chromium example, a generation-level first appearance outside a seed set. They do not establish teacher eligibility, consumption by a later optimizer or an improved policy, and a receipt hash alone does not make a program eligible.

The small publication files are `target-quality-1000-summary.csv`, `target-quality-1000-comparisons.csv` and `target-quality-pine-chromium-cut1.csv`. Full 26,357-row extracted evidence stays in custody, with its hash in `historical-target-quality-summary.json`; the manual need not embed the 19.5 MB full extraction. This analysis reads development scores only and creates no new teacher eligibility, promotion or benchmark-valid result.

<a id="c-targets--paired-final-search-outcomes-within-the-admitted-development-inventory"></a>

### C.1.4 Paired final-search outcomes within the admitted development inventory

A metadata-only comparison pairs arms **within the same generation**, keeping the recorded run, dataset/family, cut, prepared SHA, vocabulary SHA, evaluator SHA/protocol, execution-order protocol and exactly 1,000 completed evaluations equal. It uses the existing 26,357 admitted receipts, not a newly expanded input boundary. Scores are finite positive `best_validation_rmse`; lower is better. Each comparison counts exact score equality, with no epsilon.

| Neural compared with | Paired observations | Neural lower | Equal | Neural higher | Geometric mean neural/control RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Generic | 3,268 | 24 | 3,227 | 17 | 1.0001869094 |
| Shuffled | 3,253 | 24 | 3,213 | 16 | 1.0000854976 |

These are repeated adaptive **final-search** observations across generations 0–70, not measurements of initial neural seed quality. The generic comparison covers 52 dataset/cut pairs from 30 families; the shuffled comparison covers 52 pairs from 31 families. Reported wall times differ in every pair. Families, contexts and generations repeat; the arms share incumbent, retained-teacher and exploration seeds, and the search can recover from poor initial proposals. The paired rows are not independent arm discoveries. No independent-sample confidence interval, causal advantage or failure-inclusive benchmark claim follows. The ratios are descriptive and close to one; they do not establish that training improved routing.

The external report's broader routing extract has 42,863 rows. Only 3,530 match this chapter's admitted receipt inventory by retained path and all checked score, prepared-data, vocabulary, evaluator, protocol, completed-evaluation and wall-time fields. The other 39,333 are outside that existing inventory; their eligibility remains unqualified here, **not classified as all sealed**. No original score path was reopened for this join. The broader campaign ratios and p-values are therefore not imported. The compact `packing-routing-audit.json` pins inputs, matching fields, arithmetic and exclusions.

<a id="c-targets--existing-historical-rankings-and-elo"></a>

## C.2 Existing historical rankings and Elo

<a id="c-targets--retained-snapshots-and-plot-data"></a>

### C.2.1 Retained snapshots and plot data

Six committed snapshots are available and their ladder and score SHA-256 values were verified against retained bytes. No new Elo computation occurred in this audit. The snapshots use their own method pools and dataset protocols, so their Elo numbers cannot be compared across snapshots as a universal score.

| Snapshot | Rankable rows | Methods | Datasets | Interpretation |
|---|---:|---:|---:|---|
| same-cut-v1-screen |110|11|10|historical_same_cut_reproducible_snapshot_not_current_ASM30|
| same-cut-v1-deep |50|11|10|historical_same_cut_reproducible_snapshot_not_current_ASM30|
| same-cut-v2-screen |150|15|10|historical_same_cut_reproducible_snapshot_not_current_ASM30|
| same-cut-v2-deep |50|15|10|historical_same_cut_reproducible_snapshot_not_current_ASM30|
| historical-sotarena-published |8763|305|45|historical_published_best_known_test_selected_not_benchmark_valid|
| historical-sotarena-benchmark-valid |8718|304|45|historical_corrected_public_development|

For plots, use `performance-evidence.json.gz`: 6 complete ladder row sets; 520 same-cut score cells, including missing/rejected statuses; 20 per-dataset reports with exact cut identities, source heads and screen/deep timings; 8,763 historical SOTArena score rows; and the 2 current-origin refit loss rows. Do not blend these into one chart without explicit separate protocol panels. The frozen panel (asm-private-30-v1) matrix had no completed cells as of 2026-09-23 about 23:55 UTC; represent it as pending, not a fabricated ranking.

<a id="c-targets--ranking-figures-interpretation-contract"></a>

### C.2.2 Ranking figures: interpretation contract

Historical ladders retain their original snapshot names. Benchmark-valid and public-development views are separated. Same-cut screening includes all admitted methods; deepening includes only finalists, so empty cells have a deliberate meaning. A rank heatmap compares methods within a dataset and stage; it is not a plot of absolute loss across datasets. Any chart with only the top methods states that limit, while its CSV preserves the full underlying inventory.

The historical ranking plots that follow use their separately pinned snapshots. They are not new model runs and are not a frozen-panel (asm-private-30-v1) campaign. Benchmark-valid and public-development views retain their original eligibility distinctions. Same-cut deep-stage coverage varies because only finalists were deepened; absent cells are not zero losses. See [§](#c-benchmark--the-bradleyterry-model-tie-rule-and-defaults) for the exact Elo model and tie rule, [§](#c-benchmark--interpreting-a-rating) for coverage limits and [§](#c-benchmark--adaptive-development-exposure) for the adaptive-development interpretation.

All machine-readable rows are retained in `data/performance-evidence.json.gz` and `data/plot-historical-elo.csv`. A displayed top-method subset does not remove the other methods from those inventories. Dataset selection and historical protocol caveats travel with each figure.

<a id="c-targets--sotarena-published-and-corrected-ladders"></a>

### C.2.3 SOTArena published and corrected ladders

<a id="c-targets--historical-benchmark-valid-elo"></a>

#### C.2.3.1 Historical benchmark-valid Elo

This view retains 304 methods and the historical corrected eligibility filter. The historical `jope-dime-regressor` entrant is rank 31 with Elo 1687.0 in this snapshot. “Benchmark-valid” means no retained artifact contradicts a row's flag; comparator protocols were not independently verified. It does not certify that every comparator obeyed today's frozen ASM protocol.

![Historical benchmark-valid Elo ladder and complete rank distribution](figures/fig01-elo-benchmark-valid.png)

The top-method view is paired with the distribution of the full method universe. Dataset coverage and the identity of the historical entrant remain visible. These numbers describe the retained historical ladder, not the current accepted step-60000 model's thirty-dataset performance.

<a id="c-targets--why-the-two-historical-ladders-differ"></a>

#### C.2.3.2 Why the two historical ladders differ

The published SOTArena snapshot puts historical `jope-dime` at rank 1/Elo 2259.8 with 11 wins and 45/45 coverage. All 45 rows are test-selected: 42 best-test evolution-envelope rows plus 3 test-adaptive accepted replays. The corrected benchmark-valid snapshot excludes those 45; **that rank 1 is not a held-out ASM claim**. The corrected snapshot has 304 methods/8,718 rankable rows. Its fixed-seed historical `jope-dime-regressor` entry is rank 31/Elo 1687.0, 1 win, 42/45 coverage; TabFM Ensemble is rank 1/Elo 1934.9. It remains historical public-development evidence.

The published view includes 45 test-selected `jope-dime` rows; the corrected benchmark-valid view removes them. The historical `jope-dime` rank-one Elo of 2259.8 belongs to the former view. It is a best-known test-selected result, not a held-out claim. An eligibility change alters the overlap graph and the fitted ladder, so these ratings must not be combined into a single current score.

![Historical published and benchmark-valid ladders with their distinct eligibility](figures/fig02-ladders-valid-vs-adaptive.png)

<a id="c-targets--historical-same-cut-screening-and-deepening"></a>

### C.2.4 Historical same-cut screening and deepening

<a id="c-targets--same-cut-v1-and-v2-snapshots-with-screen-and-deep-elo"></a>

#### C.2.4.1 Same-cut v1 and v2 snapshots with screen and deep Elo

Same-cut v1 covers native heads plus resampled search; v2 adds XGBoost 2.1.4, LightGBM 4.6.0, CatBoost 1.2.10 and scikit-learn 1.7.2 random forest under Python 3.12.3. Both use the favorable historical test-10 regression selection, seed 20260907, five folds, top-five deepening and 32×8 search. Deep rows include only screened finalists; coverage must be printed with rankings. Their timing fields are then-recorded wall times under uncontrolled host load, not the present isolated-worker timing protocol.

Same-cut v2 compares all fifteen methods during screening on ten regression datasets. Only the five finalists per dataset are deepened across all five cuts, so the deep-stage denominator differs by method. Hollow markers preserve the earlier v1 snapshot; they are a separate run, not confidence intervals. The historical searched program is distinct from the current qualified neural proposal path.

![Historical same-cut screen and deep Elo with coverage](figures/fig03-samecut-deep-elo.png)

The ten historical datasets were a favorable selection for the earlier engine. Their ranking cannot establish performance on a broader corpus or substitute for the frozen panel (asm-private-30-v1). External engine versions printed in these plots are the historical run versions, not a claim about currently installed adapter versions.

<a id="c-targets--historical-relative-loss-and-per-dataset-rank"></a>

#### C.2.4.2 Historical relative loss and per-dataset rank

Each heatmap cell is a method's screen RMSE divided by the best screen RMSE on the same dataset. A ratio of one is best; the color saturates at three while displayed values are rounded. This normalization supports within-dataset visual comparison without averaging incompatible raw target units. Outlines show which cells reached the deep stage.

![Historical same-cut relative RMSE heatmap](figures/fig04-samecut-rmse-heatmap.png)

The rank-distribution plot shows all ten per-dataset screening ranks for each method, with mean and median. It reveals variability hidden by a single aggregate ladder. A rank difference conveys ordering, not the magnitude of a loss improvement or a confidence interval.

![Historical same-cut per-dataset rank distribution](figures/fig05-samecut-rank-distribution.png)

<a id="c-targets--historical-production-time-and-the-supplied-reference-scenario"></a>

### C.2.5 Historical production time and the supplied reference scenario

The retained same-cut reports record mean program-production wall time 2.1837062093 seconds, median 0.9870389355 seconds and maximum 8.299870126 seconds across ten datasets. These are recorded timings under the host's contemporary load, not isolated controlled latency measurements and not the current paired ASM proof's elapsed time.

![Historical program-production times on the favorable same-cut v2 test-10 selection; the one-hour/ten-million-token reference line is not measured](figures/fig06-seconds-per-program.png)

The one-hour/ten-million-token marker is the user's contextual scenario, drawn as a reference line; it is not measured. Workload, hardware, token trace, currency and result quality were not measured under matching conditions. The chart therefore establishes no matched speedup, cost saving, token-efficiency ratio or quality equivalence. The measured current proof and its per-predictor/per-recipe accounting remain separately reported.


---

<a id="c-data-estate"></a>

# Appendix D — Dated data estate, recovery evidence and binary formats

This appendix draws mainly on four dated metadata captures made on 2026-09-23; [§](#c-data-estate--recorded-copy-and-retention-gaps) also uses the DATA-01 author's copy-risk reconciliation, whose sizes partly come from a 2026-09-20 census. None of the four captures read a dataset payload, sealed label or model weight body. Compare figures within one capture; the captures differ in time and scope.

| Capture | Time (UTC) | Scope | Where |
|---|---|---|---|
| Product local roots | 16:29:27 | Six named product roots; `lstat` metadata plus two cached dataset manifests | [§](#c-data-estate--product-local-roots-and-the-local-dataset-cache) |
| Product S3 buckets | 16:39:11 | Three product buckets; current objects only | [§](#c-data-estate--product-s3-buckets) |
| DATA-01 bucket extension | configuration 17:30:13–17:30:19; learning-store prefix listings 17:30:32–17:33:50; bucket listings 17:34:15–17:37:10; assembled 17:37:54 | Configuration of nine buckets, targeted `veox-jopedime-learning-use2` prefixes, and current-object listings of seven additional or reconciled buckets | [§](#c-data-estate--additional-archive-replica-and-custody-metadata) |
| DATA-01 host roots | scans started 17:30:58–17:33:14; report 18:54:01 | Twelve named host roots; `find -xdev` metadata | [§](#c-data-estate--additional-host-data-roots) |

Both DATA-01 captures reach this manual only as an aggregate subset; [§](#c-data-estate--additional-and-replica-buckets) records its release status.

<a id="c-data-estate--configured-product-stores-and-local-roots"></a>

## D.1 Configured product stores and local roots

<a id="c-data-estate--access-method-store-defaults-and-path-conventions"></a>

### D.1.1 Access method, store defaults and path conventions

AWS metadata access used profile `veox-jepson-prod` and verified account `714044927654` with STS. No credentials are included in the receipts. Current-object listings exclude object versions and deleted objects. ETags are retained as concurrency guards for metadata downloads; they are not assumed to be SHA-256 content hashes. Catalog/collection/bundle manifest downloads use the exact listed ETag and verify size; their bytes receive local SHA-256 pins. No S3 dataset blobs, sealed tables, or model weight bodies were fetched.

`jope-datasets` defaults to `s3://veox-jopedime-datasets-use2`; `jope-models` defaults to `s3://veox-jopedime-models-use2`. The known learning store is `s3://veox-jopedime-learning-use2`; the current `jope-learning` mirror resolver uses `--store`, then `JOPE_LEARNING_URL`, and otherwise no remote mirror. It must not be described as an automatic remote default. No JOPE path/store overrides were set in this audit process.

There are two source-level path conventions to name explicitly: dataset cache resolution defaults to `${XDG_CACHE_HOME:-~/.cache}/jope`, while general `jope_paths::jope_home()` resolves a marked checkout. General mutable state defaults to `${XDG_STATE_HOME:-~/.local/state}/jope`. Use an explicit root or the owning command’s resolver; do not assume the shared variable name makes the defaults identical.

<a id="c-data-estate--product-s3-buckets"></a>

### D.1.2 Product S3 buckets

| S3 bucket / prefix | Current listed objects | Current listed logical bytes |
|---|---:|---:|
| `veox-jopedime-datasets-use2` | 15,442 | 3,234,922,515 |
| ↳ `blobs/` | 11,577 | 3,224,104,137 |
| ↳ `catalog/` | 2 | 5,363,942 |
| ↳ `collections/` | 4 | 94,356 |
| ↳ `datasets/` | 3,859 | 5,360,080 |
| `veox-jopedime-learning-use2` | 645,281 | 430,645,459,168 |
| ↳ `staging/` | 645,281 | 430,645,459,168 |
| `veox-jopedime-models-use2` | 967 | 3,929,105,273 |
| ↳ `blobs/` | 21 | 1,461,921,570 |
| ↳ `bundles/` | 3 | 32,045 |
| ↳ `staging/` | 943 | 2,467,151,658 |

The original three bucket listings completed with zero payload-body reads. The learning bucket contains current `staging/lineage-a/`, `staging/lineage-b/` and `staging/mirror-weights/` prefixes; the later DATA-01 extension below supplies measured and explicitly derived prefix counts. Current listed learning bytes are 430,645,459,168. A separate CloudWatch daily statistic at 2026-09-23T00:00:00Z reports 645,281 objects and 430,732,116,688 StandardStorage bytes. It is a dated metric with a different scope/capture time; no exact reconciliation with the current listing is claimed. Paginated S3 listing is an observation over an interval, not an atomic bucket snapshot.

<a id="c-data-estate--product-local-roots-and-the-local-dataset-cache"></a>

### D.1.3 Product local roots and the local dataset cache

| Local root | Present | Regular files | Logical bytes | Unique-inode logical bytes |
|---|---|---:|---:|---:|
| `/home/ubuntu/.cache/jope` | True | 170 | 36,754,881 | 35,585,244 |
| `/home/ubuntu/.local/state/jope` | False | 0 | 0 | 0 |
| `/home/ubuntu/JopeDime/datasets/test` | True | 40 | 1,096,413 | 1,096,413 |
| `/home/ubuntu/JopeDime/datasets/by-id` | False | 0 | 0 | 0 |
| `/home/ubuntu/JopeDime/models` | True | 5 | 33,687 | 33,687 |
| `/home/ubuntu/JopeDime/state` | False | 0 | 0 | 0 |

The local inventory uses `lstat`, does not follow symlinks, and counts hardlinked payload bytes once in the unique-inode column. It covers the named product roots, not every archive, research checkout or build directory on the host. The standard local state roots were absent. That does not imply the host has no learning evidence: campaign and recovery receipts live in explicit custody paths outside those defaults.

The local dataset cache has two cataloged regression datasets, `0014b9dad5725a1e` (2,000 rows, 16 features) and `006bca140f93e411` (2,000 rows, 32 features), each with 1,000 declared training and 1,000 declared test rows. Their payloads were not reopened for this inventory. The committed historical `datasets/test/` collection is 10 datasets/40 files/1,096,413 B; it is distinct from the frozen thirty-dataset cache.

<a id="c-data-estate--additional-archive-replica-and-custody-metadata"></a>

## D.2 Additional archive, replica and custody metadata

<a id="c-data-estate--additional-and-replica-buckets"></a>

### D.2.1 Additional and replica buckets

A separate DATA-01 metadata capture at 2026-09-23T17:37:54Z extends the earlier three-store observation. It used the same verified AWS account and read current-object listings and bucket configuration only. The frozen panel and its row counts are unchanged. The author released the packet at 19:30:23 UTC (manifest `4bebee5a…`, 148 entries), but its handoff records a failed sealed-name gate and `cleared_for_handoff: false`. The only gate hit is in one raw file. A clarification recorded at 21:05:48 UTC (`RELEASE-CLARIFICATION.md`, `380f3402…`, covered by manifest `SHA256SUMS.release-2`, `a4b0c436…`) limits the release to eight pinned files: seven aggregate files and the DOCFIX-01 patch, which is reserved for a separate follow-up documentation PR. Codex applied that patch, with bounded corrections, in documentation-correction PR86 (head `0079ceddb`), which merged at 2026-09-24T00:04:12Z as main `2b375ef11`. Everything under `raw/` and `scratch/`, and the full handoff, stays quarantined in custody. This edition imports only Codex's two-file aggregate projection (stopped subset `558548be…`), which the clarification confirms lies inside the released scope: `data-estate-extension-summary.json` (`e4dbe075…`) and its source/scope pointer `data-estate-extension-release.json` (`366fa000…`). This is an aggregate subset, not a claim that the whole packet passed its release gates.

| Additional or reconciled S3 bucket | Current objects | Current logical bytes |
|---|---:|---:|
| `veox-jopedime-archive-use2` | 180,547 | 33,078,118,944 |
| `veox-jopedime-learning-usw2` | 0 | 0 |
| `veox-jopedime-models-usw2` | 967 | 3,929,105,273 |
| `veox-jopedime-models-use2` | 967 | 3,929,105,273 |
| `jopedime-714044927654-us-east-2-dev` | 186,305 | 118,116,270,003 |
| `jopedime-714044927654-us-east-2-prod` | 186,304 | 117,847,288,926 |
| `jopedime-714044927654-us-east-2-sealed` | 8,679 | 246,060,447,894 |

The models buckets agree in total and reported prefix counts and bytes; this does not prove key-by-key or payload identity. The learning replica is empty at this capture. Its source's four enabled replication filters are `hwm/`, `waves/`, `checkpoints/serving/` and the prefix `training-head`; none covers the current `staging/` objects. This is an observed configuration/coverage gap, not evidence that a matching replication rule failed. A `HeadBucket` probe for `veox-jopedime-datasets-usw2` returned HTTP 404 in this account/context; no dataset replica listing was obtained.

<a id="c-data-estate--learning-store-prefixes"></a>

### D.2.2 Learning-store prefixes

| Learning-store prefix | Objects | Logical bytes | Basis |
|---|---:|---:|---|
| `staging/lineage-b/accepted60000-restored/` | 225,207 | 110,918,425,513 | measured |
| `staging/lineage-a/encodings/` | 419,644 | 149,840,032,105 | derived |
| `staging/lineage-a/preserved-archives-20260908-F-D-E/` | 34 | 141,775,927,725 | measured |
| `staging/lineage-a/genebank/` | 5 | 1,809,949,972 | measured |
| `staging/mirror-weights/` | 391 | 26,301,123,853 | measured |

The encodings row is a residual: the earlier full-bucket total minus four later measured sibling prefixes. It was not independently re-listed in full, and the sum closes by construction across capture times. Keep this distinction when comparing inventories. The accepted-60000 restored prefix's 225,207 objects and 110,918,425,513 B also match the retained local upload inventory by file count and size; that comparison is not a fresh content restore test.

<a id="c-data-estate--versioning-and-object-lock"></a>

### D.2.3 Versioning and Object Lock

All nine inspected buckets report versioning enabled. The six `veox-jopedime-*` buckets report Object Lock enabled with no default retention; the separate dev/prod/sealed family returned no Object Lock configuration. Neither result establishes retention for each object or preservation of every prior version. This listing excludes old versions and delete markers. Sealed storage contributes aggregate counts and bytes only; no sealed object body was fetched. Object/file counts are not counts of unique datasets, admitted teachers or recoverable checkpoints.

<a id="c-data-estate--additional-host-data-roots"></a>

## D.3 Additional host data roots

<a id="c-data-estate--root-inventory-and-totals"></a>

### D.3.1 Root inventory and totals

The DATA-01 report generated at 2026-09-23T18:54:01Z summarizes metadata scans starting at 17:30–17:33 UTC across twelve named roots with `find -xdev` metadata only. Its following apparent-byte totals count each regular-file path, including hardlinks. They are not additive with the earlier product-root table or the S3 copies, and do not measure memory needed to train.

| Local root | Regular-file paths | Apparent bytes | Allocated bytes, including directories | Scan |
|---|---:|---:|---:|---|
| `/home/ubuntu/remote_super/quant` | 6,127,909 | 740,214,173,579 | 755,552,710,656 | completed |
| `/home/ubuntu/jope-flywheel` | 465,739 | 161,547,128,345 | 162,954,956,800 | completed |
| `/home/ubuntu/preserved-archives` | 23 | 141,775,926,746 | 141,776,224,256 | completed |
| `/home/ubuntu/jope-preservation` | 390,889 | 250,487,278,093 | 250,592,124,928 | completed |
| `/home/ubuntu/jope-evidence` | 236,213 | 123,899,684,744 | 125,110,591,488 | completed |
| `/home/ubuntu/jope-migration-evidence` | 234,158 | 33,597,305,065 | 35,018,276,864 | partial: 15 unreadable directories |
| `/home/ubuntu/jope-dime-mirror` | 201,251 | 23,067,601,488 | 23,633,518,592 | completed |
| `/home/ubuntu/jope-rescue-20260921` | 70,442 | 12,508,351,289 | 12,767,522,816 | completed |
| `/home/ubuntu/golden42` | 37,612 | 1,749,309,751 | 1,849,475,072 | completed |
| `/home/ubuntu/sotarena` | 22,602 | 705,696,476 | 776,736,768 | completed |
| `/home/ubuntu/hyperion/weights` | 4 | 253,108,234 | 253,132,800 | completed |
| `/home/ubuntu/.cache/jope` | 170 | 36,754,881 | 37,158,912 | completed |

Across these roots the scan observed 7,787,012 regular-file paths, 1,489,842,318,691 apparent bytes and 1,510,322,429,952 allocated bytes. Removing the recorded within-root and cross-root hardlink overcount gives 1,472,779,971,117 apparent bytes; content-identical copies on different inodes remain counted. Fifteen permission-denied directories under `jope-migration-evidence` are missing from these totals. Several roots were changing during the scan, including the custody directory holding the inventory itself.

<a id="c-data-estate--filesystem-capacity-and-dataset-shaped-directories"></a>

### D.3.2 Filesystem capacity and dataset-shaped directories

All twelve roots resolve to the same filesystem. Its contemporaneous readback was 7,936,183,619,584 B total, 6,815,282,892,800 B used and 1,040,879,616,000 B available (969.395 GiB); these are dated observations, not current capacity guarantees. Multiple local paths on that filesystem do not provide independent failure-domain copies. Count/size matches with upload receipts identify recovery candidates; they do not authorize deletion or replace exact-content restore verification.

The `remote_super/quant` scan found 2,322 top-level 16-hex-ID directories with all three `train.csv`, `test.csv` and `meta.json` markers. The corrected `synth*` classification contains 25,197 `synth_<16hex>` dataset directories with those direct markers, plus two containers named `synthetic` and `synthetic_packed`. The dataset-directory class contains 86,923 files and 21,494,046,579 B; it includes more than the three payload markers per dataset. These are name/shape observations, not proof of unique, valid or train-admitted datasets. No rows, feature distributions or label bodies were opened. A pathname containing `sealed` is only a naming statistic and cannot identify every sealed/test-label payload.

<a id="c-data-estate--recorded-copy-and-retention-gaps"></a>

## D.4 Recorded copy and retention gaps

<a id="c-data-estate--material-without-an-established-remote-copy"></a>

### D.4.1 Material without an established remote copy

The author reconciled the surveyed product/custody listings, upload receipts and local indexes. The following sizes describe material for which that reconciliation did not establish a remote copy. They do not prove absence from every bucket, machine or archive. Counts overlap other tables and risk classes; never sum them as unique at-risk bytes.

| Material | Recorded files | Logical bytes | Evidence basis |
|---|---:|---:|---|
| T3 synthetic-regression train/meta/test payload | 75,591 | 21,471,473,310 | September 20 census; 25,197 dataset groups |
| T3 synthetic-binary payload | 305,284 | 135,798,140,876 | September 20 census |
| Synthetic source data | 101,687 | 126,856,964,874 | September 20 census |
| Binary generator script | 1 | 102,068 | September 23 file metadata |
| Node-archive mirror classes without a recorded S3 match | 394,460 | 75,989,854,135 | September 23 sizes joined to retained SHA-256 indexes |
| Quant model tree | 650,015 | 78,566,865,846 | September 23 file metadata; includes potentially replaceable cache |
| Migration evidence outside the uploaded node-high-value subtree | 53,638 | 22,715,493,488 | September 23 metadata; 15 unreadable directories excluded |

Payload sizes above include aggregate withheld/test-file sizes from historical metadata; this audit did not open those bodies. Documentation records the T3 regression generator as lost; this is a retained provenance claim, not a fresh exhaustive search. The binary generator has no second copy established in the surveyed records. The 21,471,473,310 B payload figure is the older three-file census, whereas the 21,494,046,579 B current directory class also includes other files: these are different scopes. Model, research, journal, evidence and other retained-root risks are summarized by all 19 source risk IDs in the companion JSON; missing counts remain null. Seven retained roots were not rescanned.

<a id="c-data-estate--recorded-custody-risks-and-disk-capacity"></a>

### D.4.2 Recorded custody risks and disk capacity

The empty surveyed learning replica and uncovered `staging/` prefix leave cross-region protection unestablished for the current learning objects. Counts matching between model buckets do not establish a successful content restore. Bucket Object Lock without default retention does not prove any object's retention or legal hold; custody buckets have no Object Lock configuration. The retained custody status also records a packer stop for exhausted disk reserve. At the dated local readback, `df` reported 87% used; capacity must be rechecked before admitting later workloads. These findings call for verified preservation and owner decisions about replication/retention. They authorize neither deletion nor new cloud writes, and they do not change the original training or panel budgets. Proposed rules: keep a single learning-store writer until an atomic server-side pointer update is qualified; backup, replication and retention operations remain separately authorized.

<a id="c-data-estate--recovery-evidence-what-is-preserved-and-what-was-lost"></a>

## D.5 Recovery evidence: what is preserved, and what was lost

<a id="c-data-estate--lineages-a-and-b-and-the-staging-store"></a>

### D.5.1 Lineages A and B and the staging store

Two historical lineages must remain separate. Lineage A contains July–September flywheel material, old winner/genebank state and preserved archives. Lineage B contains the native corpus/training chain used by the retained step-60000 checkpoint. `ADR-0005` and `docs/learning.md` describe their intended migration and their observed losses; those dated records do not authorize silently promoting old labels under the current ASM rules.

The learning S3 store is a staging estate: at the 2026-09-23 listings all 645,281 objects in `veox-jopedime-learning-use2` were under `staging/`. Targeted metadata listings found `staging/lineage-a/`, `staging/lineage-b/accepted60000-restored/` and `staging/mirror-weights/`. [§](#c-data-estate--product-s3-buckets) and [§](#c-data-estate--learning-store-prefixes) record the bucket census and the per-prefix counts. Object upload time is not the time a target was discovered or consumed. Five small metadata objects were fetched with an ETag precondition and hashed: the encoding caveats, frozen slice listing, archive catalog, genebank deprecation note and accepted60000 materialization intent. Their exact keys, byte lengths, ETags, upload times and fetch hashes are in `metadata/fetched-metadata.json` in custody.

<a id="c-data-estate--lineage-a-selections-archives-lost-records-and-exposed-score-tiers"></a>

### D.5.2 Lineage-A selections, archives, lost records and exposed score tiers

The frozen July 25 `slice500` listing chose the first 500 bytewise-sorted names from 25,031 directories, excluding 186 incoming entries and hidden packed storage. That is a reproducible selection of candidate datasets, not proof of 500 admitted teachers. The retained encoding caveat says the historical binary target encoder used a regression-flavored checkpoint: binary values were in range, but that did not establish classification training qualification. The same note describes historical target-vector fusion and encoder pins. It must not be recast as the current verified classification-context contract.

Lineage A preserves archives for snapshots `004754860`, `008903292`, `009658180` and `009915070`. The archives' compressed byte sizes and checksums identify preservation containers, not in-memory training size. This audit did not extract them. The historical records declare the strict-test-exact database, pair registry, later immutable checkpoint descendants, acceptance pointers, sealed-consumption receipts and task-Elo registry lost. A directory number or an old selected pointer cannot recreate that missing evidence. The old genebank has an improvement-event schema with task, dataset, score tier, metric, evaluator, source and ingestion time, but lacks the complete cut/vocabulary/protocol identity required by the current HWM. Its 22,798 improvement-event rows are not 22,798 currently admissible teachers; its timestamps include backfill ingestion.

The exact-current HWM identity and current admission requirements are explained in [§](#c-teachers--the-admission-guard-label-exposure-and-hwm-identity) and [§](#c-learning--what-better-and-better-must-mean). The historical `test_exact` score tier is optimization-exposed development evidence when used to choose programs. Renaming the tier does not restore independence.

<a id="c-data-estate--retained-corpus-sizes"></a>

## D.6 Retained corpus sizes

<a id="c-data-estate--six-retained-corpora"></a>

### D.6.1 Six retained corpora

The following table comes from six directly read local index/provenance files. Snapshot names are identifiers, not a reconstructed wall-clock axis. Several publications have no independently recorded timestamp in their index. These are retained historical corpora, not six steps in one proved monotonic update sequence. Counts must not be summed: corpora overlap.

| Corpus | Records | Contexts | Dataset/family counts | Groups | Record bytes | Audit-label bytes |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `qualification-v1` | 9,955 | 3,500 | 35 / 29 | 530 | 5,296,060 | 2,466,332 |
| `qualification-v2` | 85,498 | 3,500 | 35 / 29 | 834 | 45,484,936 | 25,540,156 |
| `qualification-v3` | 422,840 | 7,000 | 35 / 29 | 1,320 | 224,950,880 | 131,176,452 |
| `cohort64-qualification-v1` | 779,158 | 8,799 | 64 / 64 | 1,620 | 414,512,056 | 243,431,656 |
| `dataset-inclusion-20260909` | 5,538,508 | 8,904 | 82 / 82 | 6,815 | 2,946,486,256 | 1,753,858,051 |
| Accepted60000 closure, `snapshot-000097` | 10,214,332 | 8,894 | 80 / 80 | 7,877 | 5,434,024,624 | 3,227,289,461 |

All six use 532-byte packed records. The first five indexes use `parallel.teaching_corpus.v1`; the accepted closure uses `parallel.teaching_shards.v1`. Their indexes alone range from 1,807,418 to 98,330,379 bytes. The accepted sharded index is 60,589,288 bytes. Large JSON metadata remains a real part of startup work even when target records are binary. Full paths, byte counts, schemas and SHA-256 values are in `corpus-snapshots.csv` and `local-corpus-metadata.json`.

<a id="c-data-estate--snapshot-421-and-snapshot-97-accounting"></a>

### D.6.2 Snapshot 421 and snapshot 97 accounting

A separate September 11 audit captured snapshot `000421` with 8,899,150 records and 24,313,972 observations seen; its original index hash is `74d643be…`. That is a retained audit observation, not a seventh newly payload-verified corpus. The later accepted `000097` belongs to a different recovery run: 97 is not chronologically earlier than 421. The apparent count difference cannot be called a count of newly discovered unique teachers without joining actual program/context identities and admission events.

For accepted snapshot 97, metadata records 34,321,997 observations seen, 109,797 failed programs, 488,028 withheld meta-test observations, 1,338,630 observations without a matching context, 405,322 exact-parameter tokenization rejections and 112,903 out-of-vocabulary parameter rejections. These diagnostics describe the packer's accounting categories; do not assume they form a disjoint partition that can simply be subtracted from the observation total.

<a id="c-data-estate--binary-format-the-exact-packed-teaching-record"></a>

## D.7 Binary format: the exact packed teaching record

<a id="c-data-estate--record-layout"></a>

### D.7.1 Record layout

The Rust writer and reader agree on the following fixed-width layout. Offsets are byte offsets; all multibyte numbers are little-endian.

| Offset | Bytes | Stored field | Meaning |
| ---: | ---: | --- | --- |
| 0 | 256 | `128 × u16` token IDs | Exact representable grammar sequence, with padded slots |
| 256 | 256 | `128 × u16` legal-set IDs | In `teaching_shards.v1`, group-local index into that shard’s own `legal_sets` for each sequence position |
| 512 | 2 | `u16` length | Actual token sequence length |
| 514 | 2 | `u16` zero | Reserved space |
| 516 | 4 | `u32` context ID | Index into the shared context metadata/vector table |
| 520 | 4 | `f32` value | Score-derived training value |
| 524 | 4 | `f32` percentile | Ranking supervision under the declared label policy |
| 528 | 4 | `f32` log-runtime | Natural logarithm of one plus runtime |
| Total | **532** | | |

<a id="c-data-estate--packing-semantics-legal-sets-and-label-policies"></a>

### D.7.2 Packing semantics, legal sets and label policies

This is binary packing, not general-purpose compression. Every record reserves 128 token slots and 128 legal-state slots. A short program still occupies 532 bytes on disk. Context vectors are shared by context identity. In the accepted sharded format, each group stores its own legal-set table in `shard.json` and its index entry; legal IDs are local to that table. Loading remaps them into one global table (63 distinct sets in the external accepted-corpus audit), shared in host memory and on the GPU. The older monolithic `teaching_corpus.v1` format instead has a top-level legal table. Do not interpret a raw shard ID as a global ID. `groups.bin` stores two little-endian u64 values per group—start record and count—so a group entry is 16 bytes. The JSON index binds context, vocabulary, record hashes and offsets; each immutable shard also has a manifest and an audit-label sidecar.

Original floating-point measurements and exact program provenance are retained separately from the float32 training targets. Historical `reference_v1` orders original f64 RMSEs, uses a median-scaled score value, and assigns ordinal percentiles. The stored value is `1 / (1 + RMSE / max(upper_middle_RMSE, 1e-8))`; the percentile is `1 - i/(n-1)` in the ordered group. Here `upper_middle_RMSE` is the sorted original RMSE at zero-based index `floor(n/2)`, including for an even-sized group; it is not an average of the two middle scores. Its tie behavior is not the later midrank policy. `frozen_z_v1` preserves the first admitted value/percentile for an existing program while new programs receive their initial labels. `strict_midrank_v4` introduces exact-original-precision tie handling and the strict admission contract. These names are different target semantics, not interchangeable storage optimizations. Repacking cannot silently relabel historical targets or snap unsupported program parameters to convenient vocabulary tokens.

<a id="c-data-estate--verified-loading-into-seal-protected-shared-memory"></a>

### D.7.3 Verified loading into seal-protected shared memory

The training reader's verified path copies and hashes the actual record bytes into seal-protected shared memory (a `memfd` carrying write, grow, shrink and seal-lock seals), or obtains a verified shared file carrying those seals, then uses a read-only memory mapping. These Linux file seals, unrelated to sealed evaluation data, prevent later writes, growth and truncation. This protects a running trainer from source-file mutation and mmap truncation faults. It is **not** a promise of zero-copy disk startup: verification and the initial sealed-memory copy have costs. Index, record and context identities remain content-based; a filesystem timestamp is not a substitute for them.

<a id="c-data-estate--what-the-accepted-corpus-packing-audit-measured"></a>

### D.7.4 What the accepted-corpus packing audit measured

<a id="c-data-estate--occupancy-padding-and-index-size"></a>

#### D.7.4.1 Occupancy, padding and index size

The external verifier reports scanning all 10,214,332 records in 7,877 accepted-corpus shards, checking recorded payload/sidecar hashes and layout invariants. Its corpus identity is the same `1c840597…` bound in [§](#c-data--record-split-and-the-meta-test-closure-question). This manual independently verifies its selected metadata-file hashes and recomputes the following arithmetic; it does not claim a second scan of the original record or label bodies.

| Quantity | Result | Evidence boundary |
| --- | ---: | --- |
| Occupied token slots | 205,657,518 of 1,307,434,496 | Sum of the external length histogram |
| Mean / maximum sequence length | 20.1342 / 54 tokens | Accepted snapshot only; not a future grammar bound |
| Padding fraction | 84.2701% | `1 - occupied_slots / (records × 128)` |
| Pretty-printed `legal_sets` values | 37,691,003 bytes, 62.2074% of the 60,589,288-byte index | Raw stored JSON spans; compact serialization is a different 8,335,358 bytes |
| Original record arrays | 5,434,024,624 bytes | 10,214,332 × 532 |
| Proposed 180-byte record arrays | 1,838,579,760 bytes | Layout calculation; excludes indexes, contexts and runtime overhead |
| Proposed binary index | 1,906,735 bytes | Estimated table components; no qualified reader or writer |

<a id="c-data-estate--compression-the-proposed-180-byte-layout-and-label-precision"></a>

#### D.7.4.2 Compression, the proposed 180-byte layout and label precision

The verifier also reports an actual external `zstd -3 -T1` size experiment: original records compressed separately by group occupy **292,918,135 bytes**, audit-label group frames **331,375,483 bytes**, and one compressed index **5,371,779 bytes**. These named components total **629,665,397 bytes**, versus **8,721,903,373 bytes** uncompressed. This subtotal excludes shard manifests, `groups.bin`, contexts, vocabulary, checkpoint and other provenance; it is not a complete deployable-store size or a host/GPU memory measurement. Experimental 180-byte group records compressed to 259,362,502 bytes in that external experiment. No loader-throughput, optimizer or fitted-search benefit was measured by this metadata audit.

Status: proposed. The owner's packing order follows phase profiling: first a verified Python packed-reader bridge, then immutable shard reuse with bounded staging, then compression, and a narrower layout only if measurements justify it. The 180-byte layout is such a narrower layout: `56 × u16` tokens + `56 × u8` local legal-set IDs + three float32 labels: 112 + 56 + 12 bytes. It would reconstruct length and context from the token prefix and authenticated group metadata. The observed maximum length rounds to 56, and the current loss path already trims each batch to a multiple of eight. Those facts motivate the proposal; they **do not prove bit-identical training**. A production codec must preserve the existing 128-token grammar limit through a versioned extension or verified fallback layout; the observed 54-token maximum does not authorize a smaller grammar cap or truncation. It must reject inconsistent widths and legal-ID overflow, preserve group/sampler order, RNG sequence and all identity/provenance, reconstruct identical legal masks and every optimizer input, and pass record round trips plus fixed-batch token/mask/context/label, loss, gradient, update and reload/continuation parity. Changing physical bytes also changes the current raw-index corpus digest; compatibility needs an explicit verified migration, not an inherited old hash.

Keep the numerical policy separate from byte layout. The external all-pairs calculation reports that float16 creates new value ties in **0.377392%** of all within-group record pairs and **0.289905%** of teacher-versus-other pairs. These have different denominators from adjacent-score pairs. Quantizing labels therefore changes ranking supervision. Preserve float32 training labels and contexts, and original-precision audit observations; compression of their bytes does not require reducing their precision.


---

# Appendix E — Full task catalogue, source/schema index, receipts and glossary

<a id="c-catalogue--primary-acceptance-items-by-stage"></a>

## E.1 Primary acceptance items by stage

<a id="c-catalogue--stage-0-coordination-documentation-and-cross-cutting-work"></a>

### E.1.1 Stage 0: coordination, documentation and cross-cutting work

<a id="c-catalogue--life-01--coordination"></a>

#### E.1.1.1 LIFE-01 — coordination

**Priority:** P0. **Current evidence:** reviewed merge, exact-tree remote main and successful main-guard. **Status:** complete with retained readback.

**Action:** Retain the verified lifecycle receipt; future source changes repeat exact-head CI and eligible independent native review.

**Dependencies:** None.

**Acceptance commands:**

```sh
/home/ubuntu/.local/bin/gh-role writer pr view 83 --json mergedAt,mergeCommit,headRefOid,state
/home/ubuntu/.local/bin/gh-role writer run view 35889731910 --json headSha,attempt,status,conclusion,jobs
```

**Required evidence:** PR83 merged 2026-09-23T16:34:47Z; remote main a75e543d2166017cb0139e0a251d51d9a0c98e08 has tree 176b75bc36d76c7ce4eaeb7234677ec013468889, the same tree as reviewed head 68f56404c3e1. Main-guard 35889731910 attempt 1 succeeded; job 107278860508 finished 16:40:11Z. Clean canonical main verified before docs/asm-maintainer-manual branch claim.

**Resources:** Read-only/static custody work; no fitting, models, GPU or campaign admission.

**Source/evidence references:** `AGENTS.md:Pull requests`, [§](#c-status--implementation-snapshot-and-this-manuals-own-identity), [§](#c-maintenance--coverage-and-reviewed-lifecycle), `neural-integration/lifecycle-68f56404c3e199e06413bfd645d3fe0803d7b96f/remote-main-verification.json`.

**Qualification limit:** This task is complete; subsequent implementation remains pending. Keep the measured source identity 68f56404c3e1 distinct from the rebase-merge commit identity a75e543d2166.

<a id="c-catalogue--doc-01--manual"></a>

#### E.1.1.2 DOC-01 — manual

**Priority:** P1. **Current evidence:** assembled thirteen operating chapters and five evidence appendices with source-audited training and target histories. **Status:** canonical private manual assembled.

**Action:** Maintain the dated evidence inventory and current source/pending distinctions.

**Dependencies:** LIFE-01.

**Acceptance commands:**

```sh
python3 docs/internal/asm-manual/tools/build_manual.py
```

**Required evidence:** Input manifest pins every chapter, figure dataset, source receipt and render tool; duplicates resolved; current/pending/historical/unknown labels retained.

**Resources:** Read-only/static custody work; no fitting, models, GPU or campaign admission.

**Source/evidence references:** `architecture-training-evidence.json`, `product-language-source-pins.json`, `dataset-inventory.json`, `performance-evidence.json.gz`.

**Qualification limit:** Root integrated the book and renderer. Claude historical plot inputs were copied under stable before/after hashes; no stopped external renderer handoff is required.

<a id="c-catalogue--doc-02--manual-examples"></a>

#### E.1.1.3 DOC-02 — manual examples

**Priority:** P1. **Current evidence:** fourteen actual commands passed; five backend predictions bit-identical. **Status:** complete at named recipe/refit scope.

**Action:** Retain the completed fourteen-command receipt and rerun after meaningful recipe/runtime changes; qualify saved fitted winners separately.

**Dependencies:** DOC-01.

**Acceptance commands:**

```sh
python3 "$EXAMPLES/validate-outputs.py" --recipe-output "$EXAMPLE_RUN/recipe"
"$J" pipeline verify-receipt --receipt "$EXAMPLE_RUN/pipeline/receipt.json" --json
```

**Required evidence:** examples/commands.json instantiated with actual paths and pinned source/tool versions; five outputs with 8 finite values and complete-bit comparison; pipeline receipt verification. Stopped audit SHA256 363fbe62a706b44030205da78411243ba206ad2366d546a277babe966b802c91.

**Resources:** Completed separate 300-second allocation: one CPU, 4 GiB, zero swap/GPU, network denial declared; 30-second numerical and 60-second compiler caps. Actual receipt retained; independent live property readback limitation is disclosed. No Cargo or frozen-panel campaign; no deadline reset.

**Source/evidence references:** `examples/commands.json`, `examples/validate-outputs.py`, [§](#c-inference--example-a-a-tiny-private-recipe-fit-and-predictions), [§](#c-inference--example-b-actual-legacy-pipelinesearch-shape).

**Qualification limit:** Actual acceptance: data/example-acceptance/audit.json. Five recipe refits each predict eight synthetic rows; separate legacy pmlb229 pipeline has four successful measurements. No fitted-export, ASM demo, training or full-panel qualification. Resource-readback race retained.

<a id="c-catalogue--doc-03--manual-figures"></a>

#### E.1.1.4 DOC-03 — manual figures

**Priority:** P1. **Current evidence:** twenty-one local figure artifacts; original metrics, complete recovered per-step series and generation points retained. **Status:** figures rendered; local PDF inspection completed; publication pending.

**Action:** Retain historical/current labels, complete source CSVs and final PDF visual checks.

**Dependencies:** DOC-01.

**Acceptance commands:**

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 docs/internal/asm-manual/tools/make_inventory_figures.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 docs/internal/asm-manual/tools/make_figures.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 docs/internal/asm-manual/tools/make_training_figures.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 docs/internal/asm-manual/tools/make_target_figures.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 docs/internal/asm-manual/tools/make_legacy_training_figure.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 docs/internal/asm-manual/tools/make_per_step_figure.py
```

**Required evidence:** Plot source CSV/JSON, tool source, plot hash, precise captions, missing-stat flags and historical-vs-current labels; no invented sealed statistics or causal speedup.

**Resources:** Read-only/static custody work; no fitting, models, GPU or campaign admission.

**Source/evidence references:** [§](#c-datasets--frozen-dataset-coverage), [§](#c-targets--ranking-figures-interpretation-contract), `dataset-inventory.json`, `performance-evidence.json.gz`.

**Qualification limit:** Historical Elo belongs to its original ladder/pool/protocol; the current one-dataset proof is not a new Elo ladder. Source and figure checks are retained; authored architecture SVG is a repository source artifact.

<a id="c-catalogue--doc-04--manual-quality"></a>

#### E.1.1.5 DOC-04 — manual quality

**Priority:** P1. **Current evidence:** edition 2 rendered in two volumes; local content, navigation and visual checks completed. **Status:** rendered; final publication gates pending.

**Action:** Render docs/*.pdf, extract PDF text, inspect all dense tables/code/figures, verify local links and filenames, and produce a readback checklist.

**Dependencies:** DOC-01, DOC-02, DOC-03.

**Acceptance commands:**

```sh
pdftotext -layout "$MANUAL_PDF" "$MANUAL_TEXT"
```

**Required evidence:** PDF SHA/page count; extracted-text receipt; visual inspections of title/contents, widest tables, commands, plots and final checklist; no clipping/omitted rows; Markdown counterparts present.

**Resources:** Read-only/static custody work; no fitting, models, GPU or campaign admission.

**Source/evidence references:** [§](#c-maintenance--updating-the-book).

**Qualification limit:** A successful renderer exit is insufficient: the PDF must remain readable and complete.

<a id="c-catalogue--doc-05--real-training-and-target-history"></a>

#### E.1.1.6 DOC-05 — real training and target history

**Priority:** P1. **Status:** Original training and target values verified; final composed-document inspection required.

**Action:** Publish exact observed optimizer/corpus/encoding/target history, component-loss curves, checkpoint lineage, source hashes and explicit comparability/missing-history limits.

**Dependencies:** DOC-01.

**Required evidence:** Audited original JSON-pointer-to-CSV equality; every curve bound to model/corpus/monitor/objective; original failed/inconclusive observations retained.

**Commands:**

```sh
python3 docs/internal/asm-manual/tools/make_training_figures.py
python3 docs/internal/asm-manual/tools/make_target_figures.py
python3 docs/internal/asm-manual/tools/make_legacy_training_figure.py
python3 docs/internal/asm-manual/tools/make_per_step_figure.py
```

The recovered series preserves 63,000 rows and 378,000 exact metric tokens. Keep all 21 corpus blocks, 20 intervening boundaries and the legacy internal-ladder limitations explicit.

**Resources:** No new fits, training or model loads; plotting retained metrics only.

**Sources:** [§](#c-training-history--dated-lineage-and-evidence-coverage), [§](#c-learning--what-the-historical-training-and-target-evidence-establishes).

<a id="c-catalogue--doc-06--versions-kpis-and-continuation-runbook"></a>

#### E.1.1.7 DOC-06 — versions, KPIs and continuation runbook

**Priority:** P1. **Status:** Current and pending interfaces source-audited; independent chapter review complete.

**Action:** Publish the product/model/encoding/vocabulary/corpus/protocol version tuple, north-star and KPI definitions, real continuation evidence, and exact current-versus-pending commands and code.

**Dependencies:** DOC-01.

**Required evidence:** Source-bound command/flag/API audit; held native grant path disclosed; warm-start distinct from optimizer resume; full input closure and 64 GiB budget compatibility stated.

**Commands:** No complete executable front door is claimed; use the linked chapter’s current/pending interface map.

**Resources:** Documentation and metadata validation only; commands do not admit a new training window.

**Sources:** [§](#c-status--what-version-are-we-running), [§](#c-plan--the-continuation-interfaces-that-exist-today), [§](#c-learning--what-better-and-better-must-mean), [§](#c-search--focused-single-dataset-search).

<a id="c-catalogue--int-01--serial-integration"></a>

#### E.1.1.8 INT-01 — serial integration

**Priority:** P0. **Current evidence:** two exact custody replays + guarded hashes. **Status:** stopped unmerged packets.

**Action:** Refresh guards only after verified main; integrate training64 then product28 then warm/winner add-ons without replacing newer source corrections.

**Dependencies:** LIFE-01.

**Acceptance commands:**

```sh
python3 "$TRAINING_PACKET/replay.py" --expect-head "$SOURCE_SHA" --manifest-sha256 "$TRAINING_MANIFEST_SHA" --output "$TRAINING_PACKET/replay-NEW"
python3 "$PRODUCT_PACKET/replay.py" --expect-head "$SOURCE_SHA" --manifest-sha256 "$PRODUCT_MANIFEST_SHA" --output "$PRODUCT_PACKET/replay-NEW"
```

**Required evidence:** Exact-source guard receipts; unchanged original packet hashes/skips/resolutions; new applied diff; root scorer digest/cutoff/terminal persistence and current ledger hooks preserved.

**Resources:** Read-only/static custody work; no fitting, models, GPU or campaign admission.

**Source/evidence references:** `agent-training-integration/refresh-68f56404/HANDOFF.json`, `agent-product-final-68f56404/HANDOFF.json`, `agent-delivery-refresh-68f56404/HANDOFF.json`.

**Qualification limit:** NEW is an unused custody directory name. Add-on replay invocation must follow its own pinned helper. Never force stale guards or apply constituent accounting/coverage twice.

<a id="c-catalogue--int-02--integration-qualification"></a>

#### E.1.1.9 INT-02 — integration qualification

**Priority:** P0. **Current evidence:** static syntax/feature checks; no integrated compilation yet. **Status:** pending.

**Action:** Compile/type-check the actual combined source, format, run meaningful suites and strict Clippy, regenerate coverage/inventories through owners, then obtain exact-head CI/native approval.

**Dependencies:** INT-01.

**Acceptance commands:**

```sh
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-bench --features asm
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-cli --features asm
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-pipeline --features qualified-context
cargo fmt --all --check
python3 deploy/ci/coverage.py discover
python3 deploy/ci/coverage.py validate
python3 deploy/ci/test-training-proof.py
TMPDIR="$ACCEPTANCE/tmp" JOPE_BUILD_JOBS=4 bash deploy/ci/test-inventory.sh --update
```

**Required evidence:** Collection/exit logs, actual tests passed/failed/ignored/deselected, refreshed inventory diff, feature/dependency graph, build executable hashes, current-attemptCI and independent exact-head approval.

**Resources:** Existing exclusive CPU CI allocation; one reused lane target, jobs4 unless lane policy sets otherwise; no concurrent numerical campaign.

**Source/evidence references:** `deploy/ci/coverage.json`, `deploy/ci/training-proof.py (pending)`, `deploy/ci/fast-lane.sh`, `deploy/ci/full-lane.sh`.

**Qualification limit:** Custody syntax tests are not this gate. Required 18 Python / 27 Rust / 10 policy names are proof minima, not total coverage. New code needs actual test collection.

<a id="c-catalogue--ip-01--public-privacy-boundaries"></a>

#### E.1.1.10 IP-01 — public privacy boundaries

**Priority:** P1. **Current evidence:** allowlists/canaries prepared; historicallintblindspots. **Status:** residual D-IP4 and future PR text work.

**Action:** Resolve the approved public vocabulary/naming scope and test actual CLI help, receipts and NOTICE. Preserve the explicitly requested internal manual and the strict public inference allowlist.

**Dependencies:** INT-01.

**Acceptance commands:**

```sh
bash deploy/ci/ip-scrub.sh
```

**Required evidence:** Actual shipped-file inventory and help/receipt/NOTICE canaries; private provenance absent; allowlist negative cases. Keep the internal requested manual distinct from public bundles.

**Resources:** Existing exclusive CPU CI allocation; one reused lane target, jobs4 unless lane policy sets otherwise; no concurrent numerical campaign.

**Source/evidence references:** `AGENT_CHAT.md:D-IP4`, `deploy/ci/ip-scrub.sh`, `crates/feat-math/src/advanced_v3/fitted_portable.rs (pending)`.

**Qualification limit:** Current lint success does not prove unscanned surfaces clean. The user explicitly requested the private GP explanation, so ordinary creation of this manual needs no additional approval.

<a id="c-catalogue--maint-01--broad-diagnostic-qualification"></a>

#### E.1.1.11 MAINT-01 — broad diagnostic qualification

**Priority:** P2. **Current evidence:** historicalfull-default failures inventoried. **Status:** not closed by fast CI.

**Action:** Reproduce and repair recorded full-default, scratch, policy and neural families through their owning full lanes. Retain unavailable capabilities and resource limits.

**Dependencies:** LIFE-01.

**Acceptance commands:**

```sh
bash deploy/ci/full-lane.sh full-default
bash deploy/ci/full-lane.sh full-scratch
bash deploy/ci/full-lane.sh full-policy
bash deploy/ci/full-lane.sh full-neural
```

**Required evidence:** Source-bound full-suite collection and failures, reproductions and corrected receipts. Keep fast-CI and full-diagnostic totals separate; zero tests or skips cannot count as passes.

**Resources:** Existing exclusive CPU CI allocation; one reused lane target, jobs4 unless lane policy sets otherwise; no concurrent numerical campaign.

**Source/evidence references:** `deploy/ci/full-lane.sh`, [§](#c-maintenance--coverage-and-reviewed-lifecycle), `AGENT_CHAT.md:full-default failures`.

**Qualification limit:** Check lane environment requirements before invocation. The owning qualification supplies allocation and budget; this manual does not admit them.

<a id="c-catalogue--stage-1-correct-benchmark-integration"></a>

### E.1.2 Stage 1: correct benchmark integration

<a id="c-catalogue--bench-01--frozen-benchmark-foundation"></a>

#### E.1.2.1 BENCH-01 — frozen benchmark foundation

**Priority:** P1. **Current evidence:** current source + local/exact-head tests. **Status:** implemented; retain during integration.

**Action:** Reverify unchanged manifest and all prepared inputs, actual row/group assignments, original category/missing/label semantics, loss directions, full class probability ordering and task method inventory.

**Dependencies:** INT-02.

**Acceptance commands:**

```sh
"$J" benchmark verify-manifest --manifest "$M"
"$J" benchmark verify-inputs --manifest "$M" --datasets "$D"
"$J" benchmark methods
```

**Required evidence:** Original panel SHA ba7dace4…, 30 datasets, 10 per task, exact input-byte verification, native 17/5/4 recipes and 4 external families; alias/configuration inventory and unavailable reasons.

**Resources:** Read-only/static custody work; no fitting, models, GPU or campaign admission.

**Source/evidence references:** `crates/jope-bench/src/panel/data.rs`, `crates/jope-bench/src/panel/methods.rs`, `crates/jope-bench/src/panel/score.rs`, `benchmarks/panel.json`.

**Qualification limit:** Verification is metadata/data validation, no new fits. Do not replace a failed panel dataset.

<a id="c-catalogue--bench-02--external-comparators"></a>

#### E.1.2.2 BENCH-02 — external comparators

**Priority:** P1. **Current evidence:** four adapters implemented; earlier runtime inventories. **Status:** refresh at final environment.

**Action:** Pin all runtime/package/adapter bytes and actual constructors; execute relevant adapter contract tests and preserve unavailable dependencies.

**Dependencies:** INT-02.

**Acceptance commands:**

```sh
python3 deploy/prepare-panel-runtime.py --binary "$J" --native-command benchmark --build-receipt "$CLI_BUILD_RECEIPT" --source-sha "$SOURCE_SHA" --python "$EXTERNAL_PYTHON" --site-packages "$EXTERNAL_SITE_PACKAGES" --adapters crates/jope-bench/adapters --output "$ACCEPTANCE/runtimes.json"
"$EXTERNAL_PYTHON" -m pytest crates/jope-bench/adapters/test_panel.py -q
```

**Required evidence:** Current 4 engine version/source/package hashes, constructors for 3 tasks, actual contract-test receipt and explicit unavailability inventory.

**Resources:** Preparation owns sharedCPUallocation; tests use allocatedCPU; no panel/training window begins.

**Source/evidence references:** `deploy/prepare-panel-runtime.py`, `crates/jope-bench/adapters/_panel.py`, `crates/jope-bench/adapters/test_panel.py`.

**Qualification limit:** Do not nest an outer allocation lock around a front door already taking it. Environment mutation invalidates old inventory.

<a id="c-catalogue--stage-2-shared-execution-and-accounting"></a>

### E.1.3 Stage 2: shared execution and accounting

<a id="c-catalogue--exec-01--bounded-workers"></a>

#### E.1.3.1 EXEC-01 — bounded workers

**Priority:** P1. **Current evidence:** ten real synthetic host cases passed at 68. **Status:** implemented; CI coverage gap retained.

**Action:** Requalify final worker/controller bytes through ten current host cases and make required-lane coverage of critical async cleanup paths explicit.

**Dependencies:** INT-02.

**Acceptance commands:**

```sh
python3 deploy/ci/qualify-execution.py --test-binary "$PIPELINE_TEST_BINARY" --output "$ACCEPTANCE/host-execution"
```

**Required evidence:** Ten nonzero actual cases,20concurrentworkers,limit readbacks, process-group drainage, setup failures/cancel gates, original-window equality and fixture code reachability.

**Resources:** Existing exclusive CPU CI allocation; one reused lane target, jobs4 unless lane policy sets otherwise; no concurrent numerical campaign.

**Source/evidence references:** `deploy/ci/qualify-execution.py`, `crates/jope-pipeline/src/execution.rs`.

**Qualification limit:** Current host qualification is not automatically covered by the ordinary required unit lane; missing privileges fail, not skip.

<a id="c-catalogue--exec-02--fault-regression-tests"></a>

#### E.1.3.2 EXEC-02 — fault regression tests

**Priority:** P1. **Current evidence:** production terminal-persistence fix qualified indirectly. **Status:** test gap: drain-and-fixtures-F2.

**Action:** Add a real async-reap fixture with pre-gate BudgetExhausted and an undrained group. Prove terminal record survives before cleanup and admission remains blocked until drain.

**Dependencies:** INT-01.

**Acceptance commands:**

```sh
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-pipeline --features qualified-context execution::tests
```

**Required evidence:** Reproduction fails when persist is moved back under drained branch or admission_blocked handling is removed; corrected actual test passes with cleanup ordering evidence.

**Resources:** Existing exclusive CPU CI allocation; one reused lane target, jobs4 unless lane policy sets otherwise; no concurrent numerical campaign.

**Source/evidence references:** `crates/jope-pipeline/src/execution.rs:812`, `AGENT_CHAT.md:2026-09-23T16:17 delta review`.

**Qualification limit:** Exact new test name is not present yet; existing module command is the truthful current gate. Ten old host cases do not refute this coverage gap.

<a id="c-catalogue--exec-03--receipt-replay-regression-tests"></a>

#### E.1.3.3 EXEC-03 — receipt replay regression tests

**Priority:** P1. **Current evidence:** shared per-fold helper fixed and real 68 recovered report passed. **Status:** test gap: resume-fix-F2.

**Action:** Add PoolBackend-shaped candidate receipt roundtrip asserting live Feedback equals recovered FitAdapter Feedback, including digest and partial completed-fold evidence.

**Dependencies:** INT-01.

**Acceptance commands:**

```sh
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-bench --features asm
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-pipeline --features qualified-context neural_search::tests
```

**Required evidence:** Regression reproduces the d0f5748b7db1 fitness/replay mismatch; exercises Some(shared fold digest), changed completed outputs refuse, never-admitted normalization cannot gain evidence.

**Resources:** Existing exclusive CPU CI allocation; one reused lane target, jobs4 unless lane policy sets otherwise; no concurrent numerical campaign.

**Source/evidence references:** `crates/jope-pipeline/src/neural_search.rs:1735`, `crates/jope-bench/src/panel/optimize_fit.rs`, `AGENT_CHAT.md:resume-fix-F2`.

**Qualification limit:** The two original FixtureBackend reproductions use absent digests and do not cover the same_folds path.

<a id="c-catalogue--exec-04--cutoff-semantics-regression"></a>

#### E.1.3.4 EXEC-04 — cutoff semantics regression

**Priority:** P1. **Current evidence:** production cutoff ordering corrected; existing test too weak. **Status:** test gap: regressions-F2.

**Action:** Strengthen cancellation-at-cutoff test to assert original_deadline stop reason and persisted receipt, not only is_ok.

**Dependencies:** INT-01.

**Acceptance commands:**

```sh
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-pipeline --features qualified-context cancellation_at_cutoff_preserves_observed_generation_on_resume
```

**Required evidence:** Test fails if cancellation check moves before deadline check; successful resume emits no new fit and leaves observed bytes unchanged.

**Resources:** Existing exclusive CPU CI allocation; one reused lane target, jobs4 unless lane policy sets otherwise; no concurrent numerical campaign.

**Source/evidence references:** `crates/jope-pipeline/src/neural_search.rs:1380`, `AGENT_CHAT.md:regressions-F2`.

<a id="c-catalogue--exec-05--accounting-and-unknown-outcomes"></a>

#### E.1.3.5 EXEC-05 — accounting and unknown outcomes

**Priority:** P1. **Current evidence:** additive accounting union source + four authored Rust cases. **Status:** pending execution.

**Action:** Execute started/finished-union accounting tests; retain started-only unknowns separately from terminal refusals and qualify report replay without child retries.

**Dependencies:** INT-02.

**Acceptance commands:**

```sh
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-bench --features asm asm_campaign
```

**Required evidence:** Actual four new accounting tests collected; started-only attempts counted unknown, no inferredzero, exports counted once inASMworker totals, recovered reporting no launch.

**Resources:** Existing exclusive CPU CI allocation; one reused lane target, jobs4 unless lane policy sets otherwise; no concurrent numerical campaign.

**Source/evidence references:** `agent-accounting-union/training-bridge/HANDOFF.json`, `crates/jope-bench/src/panel/asm_campaign.rs (pending)`.

<a id="c-catalogue--stage-3-neural-consumption-and-training"></a>

### E.1.4 Stage 3: neural consumption and training

<a id="c-catalogue--neural-01--context-qualification"></a>

#### E.1.4.1 NEURAL-01 — context qualification

**Priority:** P0. **Current evidence:** eight fixed native cases passed at 68. **Status:** requires final-source binding.

**Action:** Build exact context/policy/proof workers; qualify unchanged native references for final source and preserve eight fixed-case plus actual dataset process evidence.

**Dependencies:** INT-02.

**Acceptance commands:**

```sh
"$CONTEXT_PROOF" prepare --sources "$REFERENCE_SOURCES" --sources-sha256 "$REFERENCE_SOURCES_SHA" --encoder-files "$ENCODER_FILES" --encoder-files-sha256 "$ENCODER_FILES_SHA" --worker "$CONTEXT_WORKER" --worker-sha256 "$CONTEXT_WORKER_SHA" --source-sha40 "$SOURCE_SHA" --resources "$CONTEXT_RESOURCES" --resources-sha256 "$CONTEXT_RESOURCES_SHA" --output "$ACCEPTANCE/context-proof"
"$CONTEXT_PROOF" run --plan "$CONTEXT_PLAN" --plan-sha256 "$CONTEXT_PLAN_SHA"
```

**Required evidence:** All eight fixed cases pass the unchanged component/context maximum absolute difference of 0.0003; source, assets and program identities match. Preserve actual bounded execution receipts and the qualified construction token, original labels and row order.

**Resources:** Separate explicit resource manifest and approved allowance required before admission;20 CPUs/workers maximum,64 GiB maximum,zero swap,one thread,30 seconds per fit,120 seconds per arm maximum. No new allowance granted by this checklist.

**Source/evidence references:** `crates/jope-policy/src/context_producer/admission.rs:4`, `crates/jope-pipeline/src/cpu_context_bridge.rs`.

**Qualification limit:** Metadata/path equality or checksum alone is not numerical qualification; no changed tolerance or checkpoint substitution.

<a id="c-catalogue--teach-03--real-teacher-collection"></a>

#### E.1.4.2 TEACH-03 — real teacher collection

**Priority:** P0. **Current evidence:** zero eligible current teachers; two prospective evidenced families. **Status:** blocked on fresh admitted evidence.

**Action:** Preregister teaching before two separately bounded proof plans for machineCPU and geyser. Obtain actual finite OOF improvement and improvements on four of five folds, or retain the rejection.

**Dependencies:** BENCH-01, BENCH-02, NEURAL-01, TEACH-01, TEACH-02.

**Acceptance commands:**

```sh
"$J" optimize proof-plan --base-plan "$BASE_PLAN" --base-plan-sha256 "$BASE_ID" --configuration "$TEACHING_CONFIGURATION" --configuration-sha256 "$TEACHING_CONFIGURATION_SHA" --dataset "$FAMILY_DATASET" --resources "$TEACHING_RESOURCES" --resources-sha256 "$TEACHING_RESOURCES_SHA" --output "$FAMILY_PLAN"
"$J" optimize run --plan "$FAMILY_PLAN" --plan-sha256 "$FAMILY_PLAN_ID" --manifest "$M" --datasets "$D" --journal "$FAMILY_JOURNAL"
"$J" optimize report --plan "$FAMILY_PLAN" --plan-sha256 "$FAMILY_PLAN_ID" --manifest "$M" --datasets "$D" --journal "$FAMILY_JOURNAL"
```

**Required evidence:** Before-fit freeze, vocabulary and source pins; two independent families and original windows; nine context proofs per dataset; complete five-fold predictions and trivial baselines. Apply the 1e-12 absolute plus 1e-6 relative gain margin and four-of-five rule. Retain no-signal refusals.

**Resources:** Separate explicit resource manifest and approved allowance required before admission;20 CPUs/workers maximum,64 GiB maximum,zero swap,one thread,30 seconds per fit,120 seconds per arm maximum. No new allowance granted by this checklist.

**Source/evidence references:** `agent-training-readiness/readiness-v3.json`, [§](#c-teachers--present-teacher-readiness-and-the-smallest-real-path), `crates/jope-bench/src/panel/teaching.rs (pending)`.

**Qualification limit:** Execute separately for 230_machine_cpu and 712_chscase_geyser1; the prospective choice predates losses. Both are frozen-panel regression datasets, so if either is used to manufacture teachers, later evaluations on it must be labelled exposed/adaptive ([§](#c-learning--freezing-the-experiment-and-choosing-families); OD-8 in [§](#c-plan--proposed-owner-decisions-and-external-dependencies)). Old operational receipts from sources 55e9ed688faf, 3fd7d5b2ebb6 and 68f56404c3e1 cannot be retrofitted. The older queue suggested 300 seconds per family without admitting that allowance.

<a id="c-catalogue--teach-04--real-corpus-materialization"></a>

#### E.1.4.3 TEACH-04 — real corpus materialization

**Priority:** P0. **Current evidence:** native exporter prepared. **Status:** blocked on two real families.

**Action:** Build the request from every eligible requested comparison and its original native-context proof; freeze a family-disjoint corpus and the exact vocabulary.

**Dependencies:** TEACH-03.

**Acceptance commands:**

```sh
"$J" train prepare-inputs --learning-state "$LEARNING_STATE" --request "$CORPUS_REQUEST" --request-sha256 "$CORPUS_REQUEST_SHA" --output "$CORPUS_OUTPUT"
```

**Required evidence:** At least two real families; every eligible requested comparison included; seed 7 partition derived from source families; exact 1536 + 512 context values. No invented context, parameter-bin snapping, label remapping or omission of difficult examples.

**Resources:** Read-only/static custody work; no fitting, models, GPU or campaign admission.

**Source/evidence references:** `crates/jope-cli/src/train/corpus.rs (pending)`, [§](#c-training--build-the-real-corpus-from-admitted-evidence).

**Qualification limit:** Only if exactly two families qualify does the recorded machineCPU-development/geyser-training assignment apply. More eligible families must all remain admitted.

<a id="c-catalogue--train-01--actual-trainer-tests"></a>

#### E.1.4.4 TRAIN-01 — actual trainer tests

**Priority:** P0. **Current evidence:** seven pure input tests passed; actual model cases deselected. **Status:** residual training-F3.

**Action:** Execute actual optimizer, reload and continuation tests across regression, binary and multiclass, plus task, vocabulary and runtime compatibility negatives.

**Dependencies:** INT-02.

**Acceptance commands:**

```sh
python/.venv/bin/python -m pytest python/tests/test_policy_training_contract.py python/tests/test_training_campaign.py python/tests/test_training_corpus_contract.py python/tests/test_experimental_proposal.py python/tests/test_experimental_runtime.py -q --junitxml="$ACCEPTANCE/training-fixtures.xml"
python3 deploy/ci/test-training-proof.py
```

**Required evidence:** Collected test IDs include all three parameters of test_actual_trained_classifier_reload_sampling_uses_task_head. Prove positive updates, changed parameters, task reward routing, reload and continuation; reject zero updates and missing architecture. The proof gate must see actual execution.

**Resources:** Existing exclusive CPU CI allocation; one reused lane target, jobs4 unless lane policy sets otherwise; no concurrent numerical campaign.

**Source/evidence references:** `python/tests/test_experimental_proposal.py (pending)`, `python/src/jope/policy/train_v3.py (pending)`, `deploy/ci/training-proof.py (pending)`.

**Qualification limit:** Required-name inventories and deselected tests do not count as execution. A synthetic CPU checkpoint remains fixture evidence.

<a id="c-catalogue--train-02--native-supervised-training"></a>

#### E.1.4.5 TRAIN-02 — native supervised training

**Priority:** P0. **Current evidence:** supervisor/cgroup orchestration prepared. **Status:** host proof pending.

**Action:** Exercise the native training front door with bounded synthetic normal and cancellation cases: subgroup entry, release, gate refusal, cleanup and original-window resume.

**Dependencies:** TRAIN-01.

**Acceptance commands:**

```sh
"$J" train plan --state "$FIXTURE_TRAIN_STATE" --python "$TRAINING_PYTHON" --synthetic-fixture --task all-admitted --device cpu --source-sha "$SOURCE_SHA" --seed 7 --passes 2
"$J" train run --state "$FIXTURE_TRAIN_STATE"
"$J" train verify --state "$FIXTURE_TRAIN_STATE"
```

**Required evidence:** Actual normal and interrupted CPU workers; controller-owned subgroup joined before execution; original deadline, device and runtime unchanged; children drained and resumed state compatible. The native wrapper must enforce admission.

**Resources:** Explicitbounded CPU fixture allocation; same planned 3600 seconds/1200scaps are maxima, not authorization to spend them twice; no GPU.

**Source/evidence references:** `crates/jope-cli/src/train.rs (pending)`, `python/src/jope/policy/training_campaign.py (pending)`.

**Qualification limit:** Author the cancellation harness against the actual front door and target only its owned processes. CPU mode requires explicit fixture provenance.

<a id="c-catalogue--train-03--bounded-real-gpu-training"></a>

#### E.1.4.6 TRAIN-03 — bounded real GPU training

**Priority:** P0. **Current evidence:** implemented follow-on; no GPU campaign started. **Status:** blocked on admitted data and consumption readiness.

**Action:** Run one all-admitted CUDA campaign with real regression and two passes. Retain synthetic classifier fallback separately and report every unavailable category.

**Dependencies:** TEACH-04, TRAIN-02, NEURAL-02.

**Acceptance commands:**

```sh
"$J" train plan --state "$TRAIN_STATE" --python "$TRAINING_PYTHON" --inputs "$CORPUS_OUTPUT/inputs.json" --task all-admitted --device cuda:0 --source-sha "$SOURCE_SHA" --learning-state "$LEARNING_STATE" --seed 7 --passes 2
"$J" train run --state "$TRAIN_STATE"
```

**Required evidence:** Original global/category windows; one RTX3090 UUID; positive updates, changed parameter hashes and task coverage with data_kind/fallback reasons. Save and reload model plus optimizer; record CPU/GPU cost where measured. Requested CUDA must not silently fall back to CPU.

**Resources:** One original 3600 seconds total window;≤1200 seconds per category including both passes;one RTX 3090,20CPU,64 GiB,zero swap,1Torch thread;no exclusiveCPU CI overlap. Resume never resets.

**Source/evidence references:** `python/src/jope/policy/training_campaign.py:780 (pending)`, [§](#c-training--plan-run-resume-verify-and-export).

**Qualification limit:** NEURAL-02 gates source and fixture readiness before the GPU hour. It does not depend on a newly trained live export, avoiding a circular prerequisite.

<a id="c-catalogue--train-04--checkpoint-continuation-and-export"></a>

#### E.1.4.7 TRAIN-04 — checkpoint continuation and export

**Priority:** P0. **Current evidence:** verification/export source prepared. **Status:** requires actual campaign.

**Action:** Verify two committed passes, optimizer counters and the immutable receipt chain; export complete architecture, vocabulary, task, source, encoder and runtime bindings.

**Dependencies:** TRAIN-03.

**Acceptance commands:**

```sh
"$J" train resume --state "$TRAIN_STATE"
"$J" train verify --state "$TRAIN_STATE"
"$J" train export --state "$TRAIN_STATE" --output "$TRAIN_EXPORT"
```

**Required evidence:** Resume only when needed inside the original window. Prove increased optimizer steps, identical parameters after reload, changed parameters relative to the prior checkpoint, matching export/resume proof and completed task coverage. Keep experimental non-serving status.

**Resources:** One original 3600 seconds total window;≤1200 seconds per category including both passes;one RTX 3090,20CPU,64 GiB,zero swap,1Torch thread;no exclusiveCPU CI overlap. Resume never resets.

**Source/evidence references:** `python/src/jope/policy/training_campaign.py:1203 (pending)`, `python/src/jope/policy/training_campaign.py:1303 (pending)`.

**Qualification limit:** Resume is conditional, not an instruction to start a third pass after completing the plan. A new state or plan cannot extend the approved hour.

<a id="c-catalogue--neural-02--trained-checkpoint-consumer-readiness"></a>

#### E.1.4.8 NEURAL-02 — trained-checkpoint consumer readiness

**Priority:** P0. **Current evidence:** independent source review + purechecks; regression packet appended. **Status:** compile/runtime fixture pending.

**Action:** Qualify the explicit experimental regression consumer before GPU use. Preserve accepted-baseline hard pins and strict architecture, task, vocabulary, source and encoder compatibility.

**Dependencies:** INT-02, TRAIN-01.

**Acceptance commands:**

```sh
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-pipeline --features qualified-context experimental_regression_requires_matching_task_and_executable_head
python/.venv/bin/python -m pytest python/tests/test_experimental_proposal.py -q
```

**Required evidence:** Execute all three actual task-model cases. Test accepted-versus-experimental decoding and refusal of incomplete architecture, missing state, changed vocabulary, source or encoder. The operational regression route requires a real admitted export.

**Resources:** Existing exclusive CPU CI allocation; one reused lane target, jobs4 unless lane policy sets otherwise; no concurrent numerical campaign.

**Source/evidence references:** `agent-experimental-regression/HANDOFF.json`, `crates/jope-pipeline/src/experimental_producer.rs (pending)`, `python/src/jope/policy/experimental_proposal.py (pending)`.

**Qualification limit:** The new regression consumer is not serving promotion. Synthetic regression may test mechanics but cannot enter the real operational route.

<a id="c-catalogue--neural-03--live-newly-trained-consumption"></a>

#### E.1.4.9 NEURAL-03 — live newly trained consumption

**Priority:** P0. **Current evidence:** current accepted checkpoint origin/evolution proved onone dataset. **Status:** newly trained consumption pending.

**Action:** Pin the actual exported checkpoints and runtime/canonicalizer into the final configuration. Prove new-checkpoint proposals reach real generation-zero fits and completed descendants.

**Dependencies:** TRAIN-04, NEURAL-01.

**Acceptance commands:**

```sh
"$J" optimize proof-plan --base-plan "$BASE_PLAN" --base-plan-sha256 "$BASE_ID" --configuration "$TRAINED_CONFIGURATION" --configuration-sha256 "$TRAINED_CONFIGURATION_SHA" --dataset 230_machine_cpu --resources "$CONSUMPTION_RESOURCES" --resources-sha256 "$CONSUMPTION_RESOURCES_SHA" --output "$CONSUMPTION_PLAN"
"$J" optimize run --plan "$CONSUMPTION_PLAN" --plan-sha256 "$CONSUMPTION_PLAN_ID" --manifest "$M" --datasets "$D" --journal "$CONSUMPTION_JOURNAL"
"$J" optimize report --plan "$CONSUMPTION_PLAN" --plan-sha256 "$CONSUMPTION_PLAN_ID" --manifest "$M" --datasets "$D" --journal "$CONSUMPTION_JOURNAL"
```

**Required evidence:** The exported checkpoint hash appears in actual proposal and fit evidence;64+16 versus 80 origins remain authenticated. Retain completed descendants where claimed, predictions and fitted states. Regression must be admitted and classifier fixture limits explicit; replay the original window successfully.

**Resources:** Separate explicit resource manifest and approved allowance required before admission;20 CPUs/workers maximum,64 GiB maximum,zero swap,one thread,30 seconds per fit,120 seconds per arm maximum. No new allowance granted by this checklist.

**Source/evidence references:** `crates/jope-bench/src/panel/experimental_proposal.rs (pending)`, `crates/jope-bench/src/panel/origin_proof.rs (pending)`.

**Qualification limit:** The current proof-plan scopes one regression dataset. Classification needs its implemented campaign/configuration route; this command alone cannot prove binary or multiclass live consumption.

<a id="c-catalogue--pack-01--packed-training-consumer-and-incremental-loading"></a>

#### E.1.4.10 PACK-01 — packed training consumer and incremental loading

**Priority:** P2. **Status:** Legacy binary store exists; pending Python category route still materializes JSONL.

**Action:** Design and qualify a lossless packed reader for the bounded trainer, sharing repeated contexts and immutable shards while excluding sealed-family context vectors; measure host/device loading and fresh-example exposure.

**Dependencies:** TEACH-04, TRAIN-01.

**Required evidence:** JSONL-versus-packed token/context/label/sampler equivalence; corruption/schema tests; finite complete labels; startup/RSS/I/O/padding measurements; incremental shard reuse and actual optimizer input counts.

**Commands:** No complete executable front door is claimed; use the linked chapter’s current/pending interface map.

**Resources:** Source/fixture planning first; any measured training remains inside a separately declared remaining authorized window.

**Sources:** [§](#c-data-estate--record-layout), `python/src/jope/policy/train_v3.py`, `crates/jope-engine/src/teaching_store.rs`.

<a id="c-catalogue--loop-01--repeatable-measured-learning-flywheel"></a>

#### E.1.4.11 LOOP-01 — repeatable measured learning flywheel

**Priority:** P1. **Status:** New complete improved-target-to-improved-search cycle not yet qualified.

**Action:** Demonstrate the regression chain new eligible target → actual exposure → changed parameters → compatible reload → improved fitted search → another new generation, over linked generations; two passes through one corpus are continuation, not two generations. Measure next-search benefit against both the parent checkpoint and generic search, keep the accepted 512×6 step-60000 model separate from experimental regression lineage v1 (256×2, new initialization), and never promote serving automatically.

**Dependencies:** NEURAL-03, TEACH-04, TRAIN-04.

**Required evidence:** Per-cycle target deltas, actual fresh exposures, component losses on fixed monitors, optimizer/parent/checkpoint hashes, full failures/cost and matched fitted-program outcomes; promotion separately decided. Decision: the proposed preregistered regression decision rule in learning.md, a proposal and not an achieved result: family-normalized gain versus the parent on at least 10 independent monitor families per round, with sibling fresh-versus-replay-only branches from one immutable parent. Unfunded or insufficient data yields mechanical proof and descriptive results with the causal verdict inconclusive, not an invented success.

**Commands:** No complete executable front door is claimed; use the linked chapter’s current/pending interface map.

**Resources:** Original one GPU-hour cap remains binding, with at most 20 minutes per category. Its two controlled rounds may remain unfunded or inconclusive within that existing 20-minute regression GPU cap; this item grants no new allocation. Further campaigns require their own explicit allocation.

**Sources:** [§](#c-training-history--dated-lineage-and-evidence-coverage), [§](#c-plan--the-continuation-interfaces-that-exist-today), [§](#c-learning--what-better-and-better-must-mean).

<a id="c-catalogue--stage-4-portable-fitted-inference"></a>

### E.1.5 Stage 4: portable fitted inference

<a id="c-catalogue--infer-01--fitted-state-and-privacy"></a>

#### E.1.5.1 INFER-01 — fitted state and privacy

**Priority:** P0. **Current evidence:** native fitted state exists; product28 source prepared. **Status:** qualification pending.

**Action:** Execute the lossless private codec, complete fitted preprocessing/state reload, and public allowlist/canary tests before export. Unsupported states remain explicit.

**Dependencies:** INT-02.

**Acceptance commands:**

```sh
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p feat-math fitted_portable
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-bench --features asm inference
bash deploy/ci/ip-scrub.sh
```

**Required evidence:** Complete prediction parity after reload; missing/unseen category, class order and full probability-matrix cases. Preserve the private codec identity; public bundles must omit recipe, discovery and provenance canaries.

**Resources:** Existing exclusive CPU CI allocation; one reused lane target, jobs4 unless lane policy sets otherwise; no concurrent numerical campaign.

**Source/evidence references:** `crates/feat-math/src/advanced_v3/fitted_portable.rs (pending)`, `crates/jope-bench/src/panel/inference.rs (pending)`, [§](#c-inference--state-restrictions-beyond-the-head-table).

**Qualification limit:** The private C++ head gate is broader than the public gate. Python/Rust 14 of 26, C++ 2 of 26 and JavaScript 6 of 26 are static head counts, not runtime qualification rates.

<a id="c-catalogue--infer-02--exact-selected-winner-portability"></a>

#### E.1.5.2 INFER-02 — exact selected-winner portability

**Priority:** P1. **Current evidence:** winner/add-on source + statictests. **Status:** actual export/parity unrun.

**Action:** Qualify the selected existing fold state in order: Python, Rust, the same C++ emitter under C++17 and C++20, then JavaScript. Preserve unsupported external winners and the complete ranking.

**Dependencies:** INFER-01, PANEL-01.

**Acceptance commands:**

```sh
"$J" benchmark export-winner --plan "$PLAN" --plan-sha256 "$PLAN_ID" --manifest "$M" --datasets "$D" --journal "$JOURNAL" --dataset "$DATASET_ID" --fold "$FOLD"
"$J" benchmark qualify-export --prepared "$PREPARED_EXPORT" --prepared-sha256 "$PREPARED_EXPORT_ID" --tools "$TOOLS" --tools-sha256 "$TOOLS_SHA"
"$J" benchmark report --plan "$PLAN" --plan-sha256 "$PLAN_ID" --manifest "$M" --datasets "$D" --journal "$JOURNAL"
```

**Required evidence:** Exact state, artifact, input, class, toolchain and numerical-policy hashes; actual five-backend outcomes. Python/Rust/C++ use bitwise comparison; JavaScript uses 1e-12 absolute and relative tolerance. Recover interrupted work without retrying its child; read-only reports launch nothing. Unsupported never replaces the winner.

**Resources:** Same original campaign execution window and worker caps; no new job after t0 + 6900 seconds for official panel. No post-panel allowance is implemented.

**Source/evidence references:** `crates/jope-bench/src/panel/fitted_export.rs (pending)`, `crates/jope-bench/src/panel/fitted_export_run.rs (pending)`, [§](#c-inference--example-c-exact-fitted-winner-export-after-pending-integration).

**Qualification limit:** Library fixtures precede the panel. Actual selected winners require complete confirmations from PANEL-01 and qualification before the original execution cutoff. No separately bounded post-panel qualification front door exists. One fitted fold does not qualify the other four states.

<a id="c-catalogue--future-01--cuda-inference-and-serving-promotion"></a>

#### E.1.5.3 FUTURE-01 — CUDA inference and serving promotion

**Priority:** P2. **Current evidence:** not qualified in this delivery. **Status:** deferred.

**Action:** After portable CPU backends, design CUDA fitted-inference qualification and independent experimental-to-serving promotion. Preserve the accepted baseline until promotion passes.

**Dependencies:** INFER-02, REPORT-01.

**Command boundary:** no truthful executable acceptance command yet; implement or obtain missing interface first.

**Required evidence:** New backend contract, fixed numerical policy, exact device/toolchain/state parity and promotion evaluation; reviewed pointer compare-and-swap and rollback where used.

**Resources:** NoCUDAinference/promotionbudgetadmitted bythisroadmap.

**Source/evidence references:** `docs/adr/ADR-0005-canonical-learning-loop.md`, [§](#c-training--what-the-pending-campaign-trains), [§](#c-training--compatibility-checks-and-fail-closed-rules).

**Qualification limit:** Training GPU availability does not prove CUDA inference parity. Export is not serving promotion.

<a id="c-catalogue--stage-5-installation-and-measured-demonstration"></a>

### E.1.6 Stage 5: installation and measured demonstration

<a id="c-catalogue--install-01--private-installation"></a>

#### E.1.6.1 INSTALL-01 — private installation

**Priority:** P1. **Current evidence:** installer source and 15 metadata tests passed. **Status:** cold/live path pending.

**Action:** Build the CLI from final source, measure cold build separately, checksum-install the exact Linux binary, and verify actual command help plus dependency/ELF boundaries.

**Dependencies:** INT-02.

**Acceptance commands:**

```sh
python3 deploy/install-private.py source --checkout "$R" --source-sha "$SOURCE_SHA" --output "$ACCEPTANCE/cli-build" --jobs 4
python3 deploy/install-private.py binary --binary "$PRIVATE_BINARY" --sha256 "$BINARY_SHA" --source-sha "$SOURCE_SHA" --prefix "$ACCEPTANCE/install"
"$J" doctor
```

**Required evidence:** Cold build/install durations, cache and already_present flags; Cargo executable messages, features, toolchain, dependency graph and ELF checks. Retain checksum/help readbacks and failure receipts. CLI dependencies exclude the engine, Torch and CUDA.

**Resources:** Existing exclusive CPU CI allocation; one reused lane target, jobs4 unless lane policy sets otherwise; no concurrent numerical campaign.

**Source/evidence references:** `deploy/install-private.py (pending)`, [§](#c-installation--build-and-install).

**Qualification limit:** An installer receipt is not policy/context-worker build authority. Do not include training credentials or model bodies in the public inference bundle.

<a id="c-catalogue--install-02--final-inputruntime-preparation"></a>

#### E.1.6.2 INSTALL-02 — final input/runtime preparation

**Priority:** P1. **Current evidence:** prepared panel existing; runtime refresh required. **Status:** pending final source.

**Action:** Install the pinned final Python wheel without changing CUDA dependencies; build worker inventories and verify prepared datasets before campaign planning.

**Dependencies:** INSTALL-01, BENCH-01, BENCH-02.

**Acceptance commands:**

```sh
"$J" benchmark verify-inputs --manifest "$M" --datasets "$D"
"$J" benchmark plan --manifest "$M" --datasets "$D" --source-sha "$SOURCE_SHA" --runtimes "$RUNTIMES" --output "$BASE_PLAN"
```

**Required evidence:** Wheel, source, runtime and dependency inventory; context/policy workers identified from actual Cargo messages; semantic base-plan identity; zero fits and no execution clock started.

**Resources:** Read-only/static custody work; no fitting, models, GPU or campaign admission.

**Source/evidence references:** [§](#c-installation--prepare-the-panel-and-external-runtimes), `agent-acceptance-queue/QUEUE.md`.

**Qualification limit:** Packaging commands depend on final reviewed artifacts and the actual virtual environment. Preserve versions and file identities; changing the environment invalidates a pinned campaign.

<a id="c-catalogue--demo-01--measured-warm-fitted-demonstration"></a>

#### E.1.6.3 DEMO-01 — measured warm fitted demonstration

**Priority:** P1. **Current evidence:** warmtargetadd-onprepared;historicalfirstfit≈50.4s. **Status:** real trained warm proof pending.

**Action:** Produce actual trained-neural and generic fitted artifacts, reloaded predictions, baseline/comparison and truthful target_met. Do not clip arm budgets to pass sixty seconds.

**Dependencies:** NEURAL-03, INSTALL-02.

**Acceptance commands:**

```sh
"$J" demo --asm-plan "$WARM_PLAN" --asm-plan-sha256 "$WARM_PLAN_ID" --manifest "$M" --datasets "$D" --output "$WARM_OUTPUT" --export python --export rust
python3 deploy/ci/qualify-demo.py --binary "$J" --sha256 "$J_SHA" --source-sha "$SOURCE_SHA" --output "$ACCEPTANCE/generic-three-task" --require-under-60
```

**Required evidence:** Actual fitted state, predictions and baseline; both full paired allowances; start/end, warm elapsed, execution_complete and target_met. Keep generic synthetic qualification separate from trained-neural proof.

**Resources:** Separate explicit resource manifest and approved allowance required before admission;20 CPUs/workers maximum,64 GiB maximum,zero swap,one thread,30 seconds per fit,120 seconds per arm maximum. No new allowance granted by this checklist.

**Source/evidence references:** `crates/jope-bench/src/panel/demo_asm.rs (pending)`, `docs/private-warm-target.md (pending)`, [§](#c-search--warm-fitted-demonstration-after-integration).

**Qualification limit:** Under sixty seconds is the declared goal, not a promise; no qualifying measurement exists yet. The existing twenty-second-arm example is unqualified. An explicitly admitted longer proof may satisfy execution while missing the target. This checklist grants no implicit 300-second allowance.

<a id="c-catalogue--panel-01--unchanged-thirty-dataset-campaign"></a>

#### E.1.6.4 PANEL-01 — unchanged thirty-dataset campaign

**Priority:** P1. **Current evidence:** protocol/currentfoundation qualified;full panel unstarted. **Status:** pending final trained configuration.

**Action:** Freeze the final roster/configuration and run the unchanged thirty datasets once. Screen fold0, freeze the top five, confirm five development folds, and add asm.neural diagnostically when absent. Retain every failure.

**Dependencies:** BENCH-01, BENCH-02, NEURAL-03, DEMO-01, INFER-01, INSTALL-02.

**Acceptance commands:**

```sh
"$J" optimize plan --base-plan "$BASE_PLAN" --base-plan-sha256 "$BASE_ID" --configuration "$TRAINED_CONFIGURATION" --configuration-sha256 "$TRAINED_CONFIGURATION_SHA" --output "$PANEL_PLAN"
"$J" benchmark run --plan "$PANEL_PLAN" --plan-sha256 "$PANEL_PLAN_ID" --manifest "$M" --datasets "$D" --journal "$PANEL_JOURNAL"
"$J" benchmark report --plan "$PANEL_PLAN" --plan-sha256 "$PANEL_PLAN_ID" --manifest "$M" --datasets "$D" --journal "$PANEL_JOURNAL"
```

**Required evidence:** All thirty dataset identities, ten per task; every applicable method slot; screen/frozen/confirmation losses; selection-exposure row intersection hashes; no sealed reads; winner backend outcomes and actual win counts.

**Resources:** Exactly 7200 seconds including 300 seconds reporting reserve;20 CPUs/workers,64 GiB,zero swap,one thread per fit,30 seconds per fit,120 seconds per search arm; sealed labels unopened; existing window never reset.

**Source/evidence references:** `benchmarks/panel.json`, `crates/jope-bench/src/panel/protocol.rs`, `crates/jope-bench/src/panel/selection_exposure.rs`.

**Qualification limit:** Current code forces asm.neural into extra confirmation, not asm.generic. Selected five-fold summaries are adaptive development, and ten wins per category are not promised. Qualify exports before the original 6900-second cutoff or preserve unattempted status.

<a id="c-catalogue--report-01--complete-measured-report"></a>

#### E.1.6.5 REPORT-01 — complete measured report

**Priority:** P1. **Current evidence:** one-dataset, 832 predictor/164recipe operational audit. **Status:** full campaign report pending.

**Action:** Publish all thirty losses/gaps, outcomes, time/cost per unique artifact and recipe, coverage and failure inventories. Compare the supplied one-hour/ten-million-token scenario only using supported quantities.

**Dependencies:** PANEL-01, EXEC-05.

**Acceptance commands:**

```sh
"$J" benchmark report --plan "$PANEL_PLAN" --plan-sha256 "$PANEL_PLAN_ID" --manifest "$M" --datasets "$D" --journal "$PANEL_JOURNAL"
```

**Required evidence:** Report recovery without fits; union of started/finished records; worker CPU distinguished from overlapping service CPU; known and unknown costs; training, installation, preparation and export accounted separately. Tokens and money stay unknown unless measured; retain win/loss/gap tables.

**Resources:** Original 300 secondsreportingreserve forpanel; laterreadonlyauditopensnofitbudget.

**Source/evidence references:** [§](#c-scheduling--worked-example-cpu-wall-time-and-per-predictor-cost-of-the-one-dataset-proof), [§](#c-status--canonical-proof-record-and-per-arm-facts), `crates/jope-bench/src/panel/asm_campaign.rs (pending)`.

**Qualification limit:** The one-hour comparison supplies wall-duration context, not a matched speedup. Unique artifact hashes do not establish behavioral uniqueness. Never add aggregate service CPU to worker CPU already included in it.

<a id="c-catalogue--stage-6-eligible-optimization-and-reuse"></a>

### E.1.7 Stage 6: eligible optimization and reuse

<a id="c-catalogue--teach-01--hwm-comparison-identity"></a>

#### E.1.7.1 TEACH-01 — HWM comparison identity

**Priority:** P0. **Current evidence:** pending full-identity source + test plan. **Status:** not runtime qualified.

**Action:** Apply and execute full comparison identity enforcement before feedback; ensure stale evaluator/vocab/preprocessing/protocol/class order cannot collide with an incumbent.

**Dependencies:** INT-02.

**Acceptance commands:**

```sh
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-learning
```

**Required evidence:** Changed identity negative cases, old record migration/refusal behavior, hash-chain integrity and ties/strict improvement semantics.

**Resources:** Existing exclusive CPU CI allocation; one reused lane target, jobs4 unless lane policy sets otherwise; no concurrent numerical campaign.

**Source/evidence references:** `crates/jope-learning/src/record.rs (pending)`, `crates/jope-learning/src/hwm.rs (pending)`, `crates/jope-learning/src/eligibility.rs (pending)`.

<a id="c-catalogue--teach-02--label-exposure-readers"></a>

#### E.1.7.2 TEACH-02 — label exposure readers

**Priority:** P0. **Current evidence:** burn writer and effective-role mapping prepared. **Status:** residual eligibility-B6.

**Action:** Wire durable label-burn checks into every path that may claim an untouched test score, including HWM ingest/freeze and report publication, or explicitly prohibit that claim.

**Dependencies:** TEACH-01.

**Acceptance commands:**

```sh
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-learning
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-bench --features asm
```

**Required evidence:** Production caller inventory for require_unburned_test_partition; an integration test proving a burned partition is refused by test-valid readers; attempted sealed teaching refused; adaptive development reporting remains truthful.

**Resources:** Existing exclusive CPU CI allocation; one reused lane target, jobs4 unless lane policy sets otherwise; no concurrent numerical campaign.

**Source/evidence references:** `crates/jope-learning/src/eligibility.rs:253 (pending)`, `AGENT_CHAT.md:eligibility-B6`, `crates/jope-bench/src/panel/selection_exposure.rs`.

**Qualification limit:** The panel at source 68f56404c3e1 has no benchmark-valid producer. Separate historical closure questions remain: indexed meta-test contexts and externally reported exposed high-water marks do not by themselves establish training-label leakage. A burn writer alone is not downstream enforcement.

<a id="c-catalogue--motif-01--eligible-reusable-fragments"></a>

#### E.1.7.3 MOTIF-01 — eligible reusable fragments

**Priority:** P1. **Current evidence:** separate typed library branch exists. **Status:** mining front door incomplete.

**Action:** Integrate the production eligible-State reader, typed fragment/macro miner and umbrella CLI. Preserve full comparison identity, provenance and private/public separation.

**Dependencies:** TEACH-01, TEACH-02, TEACH-03.

**Acceptance commands:**

```sh
"$J" motifs mine --help
"$J" motifs validate --help
```

**Required evidence:** Actual mine CLI contract/help; eligible-input identity inventory; deterministic typed fragment output. Exclude rejected, no-signal, unbound and sealed examples.

**Resources:** Read-only/static custody work; no fitting, models, GPU or campaign admission.

**Source/evidence references:** [§](#c-status--pending-acceptance), `agent-acceptance-queue/QUEUE.md:motifs`.

**Qualification limit:** These help commands are unavailable at source 68f56404c3e1 and at main a75e543d2166, which has the same tree; they are acceptance targets. No truthful exact mine invocation exists yet; record it after implementation.

<a id="c-catalogue--motif-02--reuse-validation"></a>

#### E.1.7.4 MOTIF-02 — reuse validation

**Priority:** P1. **Current evidence:** validationlibrary/preparation only. **Status:** live reuse pending.

**Action:** Validate fragments/macros on independent host contexts, lossless roundtrips, deterministic fit compatibility and privacy. Measure reuse benefit under the full comparison identity.

**Dependencies:** MOTIF-01.

**Command boundary:** no truthful executable acceptance command yet; implement or obtain missing interface first.

**Required evidence:** Exact implemented validate command and input table; at least three hosts per fragment where its existing contract requires that count; roundtrip and actual fit receipts, independent context identity and canary absence. Support frequency is separate from measured uplift.

**Resources:** Explicitnewboundedreusequalification allocation required; noresetofpanelortrainingwindows.

**Source/evidence references:** `agent-acceptance-queue/QUEUE.md:motifs`, `docs/adr/ADR-0005-canonical-learning-loop.md`.

**Qualification limit:** Do not invent mine flags. The historical validate argument shape alone does not establish umbrella wiring or production State admission.

<a id="c-catalogue--motif-03--new-operators"></a>

#### E.1.7.5 MOTIF-03 — new operators

**Priority:** P2. **Current evidence:** useracceptance requirement. **Status:** future conditional work.

**Action:** Admit new operators only after measured ablation and grammar, checkpoint and backend compatibility tests. Preserve the old vocabulary identity.

**Dependencies:** MOTIF-02.

**Command boundary:** no truthful executable acceptance command yet; implement or obtain missing interface first.

**Required evidence:** Preregistered ablation recipes, budget and contexts; measured marginal gain; task/type/codec/emitter/vocabulary compatibility; new identity and migration policy.

**Resources:** Separatelypreregistered ablationbudget; noneadmitted here.

**Source/evidence references:** `user approval plan:Eligible optimization and reuse`, [§](#c-inference--future-work-and-acceptance-boundaries).

**Qualification limit:** Typed reusable macros may stay within existing operators. Fragment frequency alone does not justify adding an operator.

<a id="c-catalogue--focus-01--sustained-single-dataset-improvement"></a>

#### E.1.7.6 FOCUS-01 — sustained single-dataset improvement

**Priority:** P2. **Status:** Current bounded proof exists; no incumbent-seeded attack front door.

**Action:** Add an authenticated incumbent-aware multi-wave request using full comparison identity, retained original deadline, bounded workers and durable improvement events. Preserve exploration and all non-improvement outcomes.

**Dependencies:** TEACH-01, TEACH-02, TEACH-03.

**Required evidence:** Improving, tied, worse, stale and concurrent incumbents; full OOF/baseline admission; no deadline reset or sealed input; successful no-improvement completion; actual cost per eligible improvement.

**Commands:** No complete executable front door is claimed; use the linked chapter’s current/pending interface map.

**Resources:** No indefinite loop or additional allowance implied; whole focused experiment has one predeclared finite budget.

**Sources:** [§](#c-search--focused-single-dataset-search), `crates/jope-bench/src/panel/origin_proof.rs`, `crates/jope-pipeline/src/neural_search.rs`.

<a id="c-catalogue--coordinated-future-backlog-24-proposals"></a>

## E.2 Coordinated future backlog: 24 proposals

These are Claude’s coordinated follow-on proposals, retained alongside their overlap with the primary checklist. They are not 24 additional completed features, new budgets, or modifications to the frozen panel. Proposed `OD-*` decisions have not been silently adopted. No sealed evaluation is authorized by this list. The JSON/property list retains each proposed owner and original acceptance description.

<a id="c-catalogue--documentation-onboarding-and-repository-hygiene"></a>

### E.2.1 Documentation, onboarding and repository hygiene

<a id="c-catalogue--proposal-handoff-01"></a>

#### E.2.1.1 Proposal HANDOFF-01

**Proposed owner:** Claude. **Priority:** P1. **Primary overlap:** DOC-01, DOC-05.

**Proposed acceptance:** track-a-20260922/manual-20260923/release/SHA256SUMS verifies; FINDINGS.json posted; sections 06/07/09/11 and figures/tools/data handed off

**Disposition:** Track with the overlapping primary item; exact implementation and acceptance remain pending.

<a id="c-catalogue--proposal-hyg-01"></a>

#### E.2.1.2 Proposal HYG-01

**Proposed owner:** Claude. **Priority:** P1. **Primary overlap:** INT-02.

**Proposed acceptance:** cargo metadata shows workspace default members; offline_cli cpp20 test passes; jope-bench run defaults to min(20, cores)

**Disposition:** Track with the overlapping primary item; exact implementation and acceptance remain pending.

<a id="c-catalogue--proposal-hello-01"></a>

#### E.2.1.3 Proposal HELLO-01

**Proposed owner:** Claude docs, Codex verbs. **Priority:** P1. **Primary overlap:** DEMO-01.

**Proposed acceptance:** a quickstart of at most 5 commands routing to jope demo runs top to bottom from a clean checkout; transcript retained

**Disposition:** Track with the overlapping primary item; exact implementation and acceptance remain pending.

<a id="c-catalogue--proposal-b6-01"></a>

#### E.2.1.4 Proposal B6-01

**Proposed owner:** Claude. **Priority:** P2. **Primary overlap:** MAINT-01.

**Proposed acceptance:** A2 seed diagnostic and retained-evidence asserts; spec_drift repointed (3 ids); sotarena repoint; scratch budget 900/1500 s; neural-env.sh resolves CUDA_HOME; 13 ids added to known-broken-tests.txt

**Disposition:** Track with the overlapping primary item; exact implementation and acceptance remain pending.

<a id="c-catalogue--proposal-docfix-01"></a>

#### E.2.1.5 Proposal DOCFIX-01

**Proposed owner:** Claude patch, Codex integrates. **Priority:** P1. **Primary overlap:** DOC-01.

**Proposed acceptance:** learning.md, architecture.md, quickstart.md and datasets.md corrected; AGENTS bucket table with sizes; catalogs/s3.json generated from DATA-01

**Disposition:** Track with the overlapping primary item; exact implementation and acceptance remain pending.

<a id="c-catalogue--proposal-layout-01"></a>

#### E.2.1.6 Proposal LAYOUT-01

**Proposed owner:** Claude. **Priority:** P2. **Primary overlap:** INT-02.

**Proposed acceptance:** crate package names equal directories; jope-contracts shared types; fewer duplicate Task enums/fold planners

**Disposition:** Track with the overlapping primary item; exact implementation and acceptance remain pending.

<a id="c-catalogue--privacy-and-naming"></a>

### E.2.2 Privacy and naming

<a id="c-catalogue--proposal-ip-q1"></a>

#### E.2.2.1 Proposal IP-Q1

**Proposed owner:** Claude. **Priority:** P1. **Primary overlap:** IP-01.

**Proposed acceptance:** CLI help/doctor/README/NOTICE/docs/Cargo descriptions clean; ip-scrub covers NOTICE and schemas; new lint tests fail on old main and pass after

**Disposition:** Track with the overlapping primary item; exact implementation and acceptance remain pending.

<a id="c-catalogue--proposal-ip-q2"></a>

#### E.2.2.2 Proposal IP-Q2

**Proposed owner:** Claude. **Priority:** P1. **Primary overlap:** IP-01.

**Proposed acceptance:** emitter text 2.3.0 with re-pinned fixtures; manifest key reader switch; codec serde aliases

**Disposition:** Track with the overlapping primary item; exact implementation and acceptance remain pending.

<a id="c-catalogue--proposal-ip-q3"></a>

#### E.2.2.3 Proposal IP-Q3

**Proposed owner:** Claude. **Priority:** P2. **Primary overlap:** IP-01.

**Proposed acceptance:** receipts v3 and PR 83 ids renamed with a dual-name journal loader

**Disposition:** Track with the overlapping primary item; exact implementation and acceptance remain pending.

<a id="c-catalogue--proposal-ip-q4"></a>

#### E.2.2.4 Proposal IP-Q4

**Proposed owner:** Claude. **Priority:** P2. **Primary overlap:** IP-01, PANEL-01.

**Proposed acceptance:** same-cut v3 evidence; afm.* method id renamed

**Disposition:** Track with the overlapping primary item; exact implementation and acceptance remain pending.

<a id="c-catalogue--proposal-ip-q5"></a>

#### E.2.2.5 Proposal IP-Q5

**Proposed owner:** Claude. **Priority:** P2. **Primary overlap:** IP-01.

**Proposed acceptance:** publication boundary: private-path rule for docs/internal/** and docs/asm.md plus a packaging exclusion check

**Disposition:** Track with the overlapping primary item; exact implementation and acceptance remain pending.

<a id="c-catalogue--proposal-name-01"></a>

#### E.2.2.6 Proposal NAME-01

**Proposed owner:** Claude. **Priority:** P1. **Primary overlap:** IP-01.

**Proposed acceptance:** no public AFM/ASPFM/Prime/Lime text; lint enforces

**Disposition:** Track with the overlapping primary item; exact implementation and acceptance remain pending.

<a id="c-catalogue--data-teachers-and-training"></a>

### E.2.3 Data, teachers and training

<a id="c-catalogue--proposal-data-01"></a>

#### E.2.3.1 Proposal DATA-01

**Proposed owner:** Claude. **Priority:** P1. **Primary overlap:** DOC-01, DOC-05.

**Proposed acceptance:** metadata-only inventory of archive-use2, usw2 replicas, custody dev/prod (sealed: counts only, after Codex agrees), bucket configuration, learning per-prefix reconciliation, replica parity, local roots, single-copy risk list, disk headroom; JSON in the s3-estate schema; zero body reads

**Disposition:** Author stopped the inventory at 19:30 UTC; this manual includes only the independently checked aggregate subset and 19 qualified risk summaries. The raw packet retains a failed sealed-name gate and is excluded. DOCFIX-01 remains pending; no object-body or numerical access.

<a id="c-catalogue--proposal-data-02"></a>

#### E.2.3.2 Proposal DATA-02

**Proposed owner:** Claude design, Codex prepares. **Priority:** P1. **Primary overlap:** TEACH-04.

**Proposed acceptance:** frozen teaching manifest: seeded, selected before outcomes, families disjoint from all 30 panel families

**Disposition:** Track with the overlapping primary item; exact implementation and acceptance remain pending.

<a id="c-catalogue--proposal-data-03"></a>

#### E.2.3.3 Proposal DATA-03

**Proposed owner:** Owner, Claude. **Priority:** P2. **Primary overlap:** TEACH-03.

**Proposed acceptance:** licence-known multiclass corpus with family registry and frozen splits

**Disposition:** Track with the overlapping primary item; exact implementation and acceptance remain pending.

<a id="c-catalogue--proposal-teach-05"></a>

#### E.2.3.4 Proposal TEACH-05

**Proposed owner:** Codex runs, Claude audits. **Priority:** P1. **Primary overlap:** TEACH-03.

**Proposed acceptance:** per-wave admission and refusal receipts; every admitted family counted

**Disposition:** Track with the overlapping primary item; exact implementation and acceptance remain pending.

<a id="c-catalogue--proposal-train-05"></a>

#### E.2.3.5 Proposal TRAIN-05

**Proposed owner:** Codex. **Priority:** P2. **Primary overlap:** TRAIN-03, LOOP-01.

**Proposed acceptance:** second GPU campaign; consumption reproven

**Disposition:** Second campaign is outside the existing one-GPU-hour allowance; no additional budget or deadline is granted.

<a id="c-catalogue--proposal-hwm-01"></a>

#### E.2.3.6 Proposal HWM-01

**Proposed owner:** Codex implements, Claude designs and tests. **Priority:** P2. **Primary overlap:** FOCUS-01.

**Proposed acceptance:** incumbent-seeded single-dataset attack; replacement only when the candidate beats the incumbent on at least 4 of 5 folds by the margin; negative tests for stale identity, tie, test-high/OOF-flat and burned partition

**Disposition:** FOCUS-01 covers the user’s focused-search need. Extra incumbent four-of-five-fold rule is a proposed policy, separate from required baseline admission.

<a id="c-catalogue--evaluation-export-reuse-and-cost"></a>

### E.2.4 Evaluation, export, reuse and cost

<a id="c-catalogue--proposal-sealed-01"></a>

#### E.2.4.1 Proposal SEALED-01

**Proposed owner:** Codex runs, owner authorizes. **Priority:** P1. **Primary overlap:** REPORT-01.

**Proposed acceptance:** preregistered one-shot sealed evaluation; burn receipts; no reselection

**Disposition:** Outside current execution scope: frozen sealed evaluation remains untouched. This proposal does not authorize opening it.

<a id="c-catalogue--proposal-emit-01"></a>

#### E.2.4.2 Proposal EMIT-01

**Proposed owner:** Claude. **Priority:** P1. **Primary overlap:** INFER-02.

**Proposed acceptance:** fitted C++17/20 and JS coverage for winning families with library parity fixtures

**Disposition:** Track with the overlapping primary item; exact implementation and acceptance remain pending.

<a id="c-catalogue--proposal-infer-03"></a>

#### E.2.4.3 Proposal INFER-03

**Proposed owner:** Codex. **Priority:** P2. **Primary overlap:** INFER-02.

**Proposed acceptance:** post-panel export qualification in its own admitted window

**Disposition:** Track with the overlapping primary item; exact implementation and acceptance remain pending.

<a id="c-catalogue--proposal-emit-03"></a>

#### E.2.4.4 Proposal EMIT-03

**Proposed owner:** Claude. **Priority:** P2. **Primary overlap:** INFER-01.

**Proposed acceptance:** language-neutral DSL semantics spec, golden corpus, numeric-policy conformance harness

**Disposition:** Track with the overlapping primary item; exact implementation and acceptance remain pending.

<a id="c-catalogue--proposal-motif-00"></a>

#### E.2.4.5 Proposal MOTIF-00

**Proposed owner:** Claude. **Priority:** P1. **Primary overlap:** MOTIF-01.

**Proposed acceptance:** jope motifs mine|validate --help exit 0; deterministic fixture output; --locked build and tests

**Disposition:** Track with the overlapping primary item; exact implementation and acceptance remain pending.

<a id="c-catalogue--proposal-cost-01"></a>

#### E.2.4.6 Proposal COST-01

**Proposed owner:** Owner constants, Claude table. **Priority:** P1. **Primary overlap:** REPORT-01.

**Proposed acceptance:** labelled scenario table separate from the measured table; no quality-parity claim

**Disposition:** Track with the overlapping primary item; exact implementation and acceptance remain pending.

<a id="c-catalogue--planned-execution-and-review-ownership"></a>

## E.3 Planned execution and review ownership

Codex owns canonical integration and numerical admission. Claude contributes source and evidence work in the serial queue. A contributor does not become the independent native reviewer of that same change; the final review is assigned to a separate eligible process. The table below is the current assignment; the original ownership proposal is retained unchanged as history in `data/coordinated-roadmap-proposal.json`.

| Item | Execution | Claude contribution / review scope |
| --- | --- | --- |
| LIFE-01 | Completed PR83 lifecycle | Independent native review retained in lifecycle evidence. |
| DOC-01 | Codex canonical integration | Contribution/source findings only; exact-head native review uses a separate non-contributor Claude process. |
| DOC-02 | Codex (done 16:46Z) | none |
| DOC-03 | Codex | Fresh non-contributing independent native reviewer |
| DOC-04 | Codex | Fresh non-contributing independent native reviewer |
| INT-01 | Codex | pre-review of each refreshed replay; native review at exact head |
| INT-02 | Codex | pre-review; native review |
| BENCH-01 | Codex | read-only receipt audit |
| BENCH-02 | Codex | read-only receipt audit |
| EXEC-01 | Codex | review |
| EXEC-02 | Claude writes the tests; Codex runs them under the CPU lock | custody patch (drain-and-fixtures-F2: async reap, pre-gate BudgetExhausted, undrained group) |
| EXEC-03 | Claude writes the tests; Codex runs them | custody patch (resume-fix-F2: PoolBackend receipt through verify_receipt, Some(shared fold digest)) |
| EXEC-04 | Claude writes the tests; Codex runs them | custody patch (regressions-F2: assert original_deadline stop reason and persisted search receipt) |
| EXEC-05 | Codex | review |
| NEURAL-01 | Codex | receipt audit |
| TEACH-01 | Codex | review |
| TEACH-02 | Codex | optional custody patch: production caller wiring for require_unburned_test_partition plus an integration test |
| TEACH-03 | Codex | audit that freeze/vocabulary pins precede fits; independently recompute per-fold gains and the 4-of-5 rule |
| TEACH-04 | Codex | audit that every eligible comparison is included |
| TRAIN-01 | Codex | review |
| TRAIN-02 | Codex | review |
| TRAIN-03 | Codex | review |
| TRAIN-04 | Codex | review |
| NEURAL-02 | Codex | receipt audit |
| NEURAL-03 | Codex | audit that the new checkpoint hash appears in proposal and fit evidence |
| INFER-01 | Codex | review of emitter/parity parts |
| INFER-02 | Codex | review of emitter/parity parts |
| INSTALL-01 | Codex | new-user walkthrough of quickstart and doctor output |
| INSTALL-02 | Codex | new-user walkthrough |
| DEMO-01 | Codex | wins/cost table spec; independent recomputation |
| PANEL-01 | Codex | independent read-only recomputation from the journal |
| REPORT-01 | Codex | independent wins/losses and cost recomputation; every loss published |
| MOTIF-01 | Claude code, Codex live run | via MOTIF-00 |
| MOTIF-02 | Claude design, Codex windows | preregistered ablation design |
| MOTIF-03 | Claude design, Codex windows | two-stage promotion design |
| IP-01 | Claude | IP-Q1..Q5 |
| MAINT-01 | Claude (B6-01); Codex runs full lanes | author |
| FUTURE-01 | deferred | CUDA design after EMIT-01 |
| DOC-05 | Codex integration | Independent source/data cross-check |
| DOC-06 | Codex integration | Independent source/data cross-check |
| PACK-01 | Codex integration | Independent source/data cross-check |
| FOCUS-01 | Codex integration | Independent source/data cross-check |
| LOOP-01 | Codex integration | Independent source/data cross-check |

<a id="c-catalogue--source-index"></a>

## E.4 Source index

<a id="c-catalogue--current-implementation-references"></a>

### E.4.1 Current implementation references

These links bind the reviewed implementation commit `68f56404c3e1`, not a moving branch. PR83 merged it on 2026-09-23 as main `a75e543d2166`, which has the same source tree (`176b75bc36d7`), so each pinned file is also the file at that main commit. Hashes below are SHA256 of file bytes; Git commit/tree identifiers use Git’s separate object identity. A link needs access to the private repository.

| Current source | SHA256 |
| --- | --- |
| [benchmarks/panel.json](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/benchmarks/panel.json#L1) | `ba7dace4cd535687002cc95538dbc7f7aaf77ec973177c7159b3c6d22471eee0` |
| [benchmarks/prepare.py](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/benchmarks/prepare.py#L1) | `7600de95f0769309d814d80c6723d65f0588124b3e0d1a9875d8155fcc4a91c9` |
| [crates/feat-math/src/advanced_v3/fitted_cpp.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/feat-math/src/advanced_v3/fitted_cpp.rs#L1) | `c20d7e9e7d530102435bdfd07a7c5275de041742a38d681842b87147e6c82fef` |
| [crates/feat-math/src/emit_cpp.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/feat-math/src/emit_cpp.rs#L1) | `531e4d006caac9c41f3934df8f0f84d1f62749a9380ab4b67d1ec21180e65157` |
| [crates/feat-math/src/emit_js.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/feat-math/src/emit_js.rs#L1) | `2e18c3a0fa85ff4fb0013d610be9404079956d9331c9988486b828bc7ac805cf` |
| [crates/feat-math/src/emit_py.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/feat-math/src/emit_py.rs#L1) | `138a8c2a279da1b9df7b3c155606ac7e18c145982400ca7ace40f120cf386ca6` |
| [crates/feat-math/src/emit_rs.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/feat-math/src/emit_rs.rs#L1) | `92d1277a90886073ba5b3aa19aa9fb461f97ad3290c2c4a91a1ccaeec7c1d405` |
| [crates/feat-math/src/transpile.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/feat-math/src/transpile.rs#L1) | `374d7c1e54c9199b932963f564306deec8ffcc1f0ac7add5554fdcee04e0892a` |
| [crates/jope-bench/src/panel/campaign.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-bench/src/panel/campaign.rs#L1) | `95e162538f7601f40bd2e7019639a6622259e516fa38cd1f92e83a0d4117b20c` |
| [crates/jope-bench/src/panel/campaign_cli.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-bench/src/panel/campaign_cli.rs#L18) | `419262eab7516ba0ede8432a5d1f162fd4d64047681764e55edab9e438bb2ad1` |
| [crates/jope-bench/src/panel/cli.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-bench/src/panel/cli.rs#L1) | `f1f9879d0b2de5473f8fcd9b3722e5c41147d5e33b26adac0948d0f4541a7b21` |
| [crates/jope-bench/src/panel/methods.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-bench/src/panel/methods.rs#L1) | `170b3fcd3fadc140c5d3006e34372a8042d6c8939395a3f633085b884e1cdaa6` |
| [crates/jope-bench/src/panel/optimize_cli.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-bench/src/panel/optimize_cli.rs#L1) | `17af561a039fd4832dd1bd23c226bbde5065b09b5273eb6e0b1353899042bd80` |
| [crates/jope-bench/src/panel/receipt_timing.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-bench/src/panel/receipt_timing.rs#L1) | `0a479d36f84c9c22381310e9dd83ca4cf4c48594a7541c666587c0226f4a7367` |
| [crates/jope-bench/src/panel/score.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-bench/src/panel/score.rs#L1) | `10093ed25a81940c7ab985843e742efc49990f3d361c92764d6d6dcae07e81e9` |
| [crates/jope-bench/src/panel/selection_exposure.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-bench/src/panel/selection_exposure.rs#L9) | `a591787e6eaa95bb2e11d4c46776707076016b06785c4fad1dbc71d1a4d9fc8e` |
| [crates/jope-cli/src/kernel.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-cli/src/kernel.rs#L1) | `d8f26b2eb49652d61600c699155e7b69664bd0e868e1fc76229da012e6e547c8` |
| [crates/jope-cli/src/main.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-cli/src/main.rs#L1) | `4b2990e5294d0668a8d5f7a6c1e809d91fbb93be1418fc3fb54f291c88251265` |
| [crates/jope-cli/src/schema.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-cli/src/schema.rs#L1) | `307ce6a4fa3c22bef7c15e7272bd834100ba78715eba7b1c0547a2e1b6b04d1d` |
| [crates/jope-paths/src/lib.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-paths/src/lib.rs#L1) | `abc95bfa0e258642e0101b2f82c07706492dfe1d6c8a6fd99043a475893aa120` |
| [crates/jope-pipeline/src/cli.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-pipeline/src/cli.rs#L1) | `1bd10526c2d58c233f5bb1e9bd575bb5c7afa82691b2238ad2f6993f88bd6ae8` |
| [crates/jope-pipeline/src/cpu_context_bridge.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-pipeline/src/cpu_context_bridge.rs#L18) | `48dbe30216bf42520df668ca5b3d4a4138d21ae2731ba07c349fe6ec232729ff` |
| [crates/jope-pipeline/src/dataset.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-pipeline/src/dataset.rs#L1) | `3f34436e6d0bbb747c846d63dfe3bcbc4e1ce4e34863caa37d7dce0c7183e7ba` |
| [crates/jope-pipeline/src/emit.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-pipeline/src/emit.rs#L1) | `bb351af0947653caf9bfb5d276efd8b2645599503a101f1100b9fc20bb07edb9` |
| [crates/jope-pipeline/src/execution.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-pipeline/src/execution.rs#L1) | `e38f68bbb765918f421e69d82fa697d822d60a53eb10c51e30cc9f1ec8066a76` |
| [crates/jope-pipeline/src/experiment.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-pipeline/src/experiment.rs#L1) | `cec0f36073f73de8c09f508512c18e0098bb3f4028af08081d1b6a8639ab2e1e` |
| [crates/jope-pipeline/src/front_door.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-pipeline/src/front_door.rs#L1) | `71dc0402ff413f020a25983a5ba98386324c16af03363f5deae4fafceafd856a` |
| [crates/jope-pipeline/src/neural_search.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-pipeline/src/neural_search.rs#L27) | `a8cd55c9c4f7b5a1060eab5979deb76db24484360389f2072f6b906c4bae0390` |
| [crates/jope-pipeline/src/search_context.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-pipeline/src/search_context.rs#L8) | `e423e9676a75426cdb6fdbbaf908e80c4bc1c89c5d93007aedcdfc0189cd382e` |
| [crates/jope-pipeline/src/supervisor.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-pipeline/src/supervisor.rs#L1) | `a61265692d61580fe174c44a5f1cc2971e6512b803d91bc480d38c90a73a16b6` |
| [crates/jope-policy/src/config.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-policy/src/config.rs#L17) | `a740ca64edde353c9624f5fdc3b63878c1200ee95c159a71bca1556abc28ed37` |
| [crates/jope-policy/src/context_producer.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-policy/src/context_producer.rs#L35) | `f218d0539e273df4f53263c8ca4f06639d68456ab2057a195bf3253e27aa32d2` |
| [crates/jope-policy/src/context_producer/admission.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-policy/src/context_producer/admission.rs#L4) | `2e98364dfe3994a21ccc2d6115adb7d7396c3de3a9babf15bcae473d4d59c441` |
| [crates/jope-policy/src/context_producer/input.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-policy/src/context_producer/input.rs#L5) | `0fc3df5ed519439dd47fd27022e4be11594924a46fff4c926901cb514f5e4609` |
| [crates/jope-policy/src/producer.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-policy/src/producer.rs#L12) | `bfa9ecfe92bc6224d0409e2a66048a9858958a4833102cd57c0201fc4b083895` |
| [crates/jope-policy/src/bin/jope-policy-producer.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-policy/src/bin/jope-policy-producer.rs#L13) | `73a4678ac5fa362396f943ba02e8bc68e1fd5d00d4610d4de5ba597463fb3e6b` |
| [crates/jope-policy/src/context_producer/worker.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-policy/src/context_producer/worker.rs#L9) | `c106275c81e706c3eef8ff4644aa74efaf0d8a3e26eb97eb36b9222d052d82c4` |
| [crates/jope-policy/src/seeds.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-policy/src/seeds.rs#L42) | `434d8a0cde2e34c57291ec3657bf92707de6fbaf99f53e0fbe6f1b4aee969c15` |
| [crates/jope-policy/src/lib.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-policy/src/lib.rs#L133) | `2b0da78dcd569de31fdd0922c90e56e19e404ba9f1e184a1eb61cf634ac733a8` |
| [crates/jope-policy/src/model.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-policy/src/model.rs#L28) | `7d0f464fe4bd51ec4b2afa2257678565f959760c74bc347a7d1cebff3efe1e25` |
| [crates/jope-policy/src/ranking.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-policy/src/ranking.rs#L22) | `92c31cc55f3943aea2246bc0191325fbd625d2019fd49ce33c54f505b751843c` |
| [deploy/ci/e2e-smoke.sh](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/deploy/ci/e2e-smoke.sh#L1) | `6becfc97830d87907bfb594aae247a117fb23162382d77bc75fbd770b4668354` |
| [deploy/ci/emitters.sh](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/deploy/ci/emitters.sh#L1) | `beb99faf01c60b4f39b0179eab46696bcd5cc10658b8cdd690962bbd35254e6b` |
| [deploy/ci/emitters/compare-bits.py](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/deploy/ci/emitters/compare-bits.py#L1) | `eaee5cc96d3d3161c85fefc02d636874adb4d399c340863534d83db5e8fe6249` |
| [deploy/ci/emitters/driver.cpp](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/deploy/ci/emitters/driver.cpp#L1) | `899ad86994c93ac37312a66acde6900948948b22bfae1dd0be1516c7e99ca416` |
| [deploy/ci/emitters/driver.js](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/deploy/ci/emitters/driver.js#L1) | `754b98bc8d694ec482e23826412e7dcbde76717e7d2f72d18a9adeb76ad12f75` |
| [deploy/prepare-panel-runtime.py](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/deploy/prepare-panel-runtime.py#L1) | `f9a0d1d9ca97dfd7664a39a8df6b1e364e738fdfaaa13a0e327402c073ee46f5` |
| [python/src/jope/policy/model_v3.py](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/python/src/jope/policy/model_v3.py#L93) | `11ab74d12bda8a2fb817b073a685d128eff06479f769e4301488339424314ffe` |

Current interface/semantics: `crates/jope-cli/src/main.rs`, `crates/jope-cli/src/kernel.rs`, `crates/jope-pipeline/src/cli.rs`, `crates/jope-pipeline/src/front_door.rs`, `crates/jope-paths/src/lib.rs`, `crates/jope-bench/src/panel/{cli,campaign_cli,campaign,optimize_cli,methods,score,receipt_timing}.rs`, `crates/jope-pipeline/src/{experiment,execution,supervisor,neural_search}.rs`, `deploy/prepare-panel-runtime.py`, `benchmarks/{panel.json,prepare.py}` and `AGENTS.md`.

<a id="c-catalogue--pending-source-references"></a>

### E.4.2 Pending source references

These are stopped custody patch views. They are not installed capabilities and are not linked as if they existed at the reviewed Git commit. Their patch identities and acceptance gates are named in [§](#c-training--the-pending-bounded-extension-path-sequence-and-limits), [§](#c-installation--command-surfaces-and-the-stopped-integration-order) and the integration card [§](#c-catalogue--int-01--serial-integration).

| Pending path | SHA256 of inspected view |
| --- | --- |
| `agent-delivery-refresh-68f56404/replay-001/source-text/crates/jope-bench/src/panel/demo_asm.rs` | `dd1a287c4ff3c1447a20e81d2c2990913923cd48c667f3be600f5d6b3c972f62` |
| `agent-delivery-refresh-68f56404/replay-001/source-text/crates/jope-bench/src/panel/fitted_export.rs` | `2f6e75d70e6d89bbfeac7db7ec67c585e6d7fc5c6287b134851db931ef5ffb30` |
| `agent-delivery-refresh-68f56404/replay-001/source-text/crates/jope-bench/src/panel/fitted_export_run.rs` | `3b007f44198aaa7b96fbca4936ed7ad551feea086f6018f1ff0458ddee6c7295` |
| `agent-delivery-refresh-68f56404/replay-001/source-text/docs/private-fitted-qualification.md` | `abf27b0dd7fabc0ee4a1f095d37d2ee4f77c5ffdfd2988699990521089f61ef8` |
| `agent-delivery-refresh-68f56404/replay-001/source-text/docs/private-warm-target.md` | `4234468c7f9b5348f7a264f815d10d6f62baf1f4b5397c6cf1fed644a12d97a8` |
| `agent-product-final-68f56404/replay-001/source-text/crates/feat-math/src/advanced_v3/fitted_portable.rs` | `f2f8998251f2e7e97900563a7032b7d8b8c80bf43e080b7b622b2e195f75737f` |
| `agent-product-final-68f56404/replay-001/source-text/crates/feat-math/src/advanced_v3/fitted_portable/scalar.rs` | `34823e72c9576c7025d6bbb31a44203f397e1d571642d516a63b5ef328413970` |
| `agent-product-final-68f56404/replay-001/source-text/crates/jope-bench/src/panel/demo.rs` | `00751a3094b42c6f742cffed5f7804803977f47d9f2837b161421e884f7efdfc` |
| `agent-product-final-68f56404/replay-001/source-text/crates/jope-bench/src/panel/demo_asm.rs` | `74c2c6164015bc65f53d021c77f78d9a9ae7238d6c68bb007da424328707064d` |
| `agent-product-final-68f56404/replay-001/source-text/crates/jope-bench/src/panel/inference.rs` | `c4929b553dd6893a48b836c2c3838b62b9d96fbd2c5f76634e235eaeed98e626` |
| `agent-product-final-68f56404/replay-001/source-text/deploy/install-private.py` | `3e4e16d5d7e224624c970aca7217952339c8e679b12752aa8317452c27120e9c` |
| `agent-product-final-68f56404/replay-001/source-text/docs/private-install-demo.md` | `05d728bba5fdc78c1568650ef8179d71ce19cf5c8860ad23317c8cea53dfebb6` |
| `crates/jope-bench/src/panel/experimental_proposal.rs` | `d03c765ac08d1ecc039a82f3dbea0c40bc5158baac042e8df348f0c91a90a018` |
| `crates/jope-bench/src/panel/optimize_cli.rs` | `487a299045d9a24604511cdecd8d4fa0b64ff055e4557339ce34120858549eca` |
| `crates/jope-bench/src/panel/origin_proof.rs` | `229d26685d60646887e5ec30ec4734aa78dfdb2f37788ffbf30f878ce593b675` |
| `crates/jope-cli/src/train.rs` | `486235575fc972306a6294233d94f8a37179945931e74bfb298e342752c634a2` |
| `crates/jope-cli/src/train/corpus.rs` | `a239fd863384817821427033963d8c0c5ebc54642130b087a9848aa51f2d93c5` |
| `crates/jope-learning/src/eligibility.rs` | `839e1de7b5f2c869817564ed39f3c30843208f7a00a1a277d27fe50857ce1e8d` |
| `crates/jope-learning/src/eligibility/inner.rs` | `b5fb468fcc554ec7da9227244d435d84593910c254678fb3f4174b5d7a4e9215` |
| `crates/jope-learning/src/record.rs` | `4360d96a4a6c0a27f5ba78fbcab4f5be10a3a800f56acee6379ef805b3815d61` |
| `crates/jope-pipeline/src/experimental_producer.rs` | `9a15e7077651ded2adf364a26a579720ae21aa92f0362c0122d132e34162411c` |
| `crates/jope-pipeline/src/search_plan.rs` | `fe8e402357b9b61359b03760401da9a568b8a1e7092b1222947fc30d2c8fb817` |
| `crates/jope-policy/src/context_producer/classification.rs` | `465af30abcdc7bcc02c046d92a71e0b7a22be910320e22158ef16b216adc4fa6` |
| `metadata-receipts/veox-jopedime-models-use2--bundles--hyperion-v7--manifest.json` | `fdaf938fc7f7957c06959f12c130b2bd81cdaef5e5bb2198ba75cf35da1c5233` |
| `metadata-receipts/veox-jopedime-models-use2--bundles--jope-prime-gen31-v2--manifest.json` | `c8d8d13a2f4cb2eeb21eea364c33e14429c84d344e6b50eb38a472173b97b6f2` |
| `metadata-receipts/veox-jopedime-models-use2--bundles--policy-step-60000--manifest.json` | `d8e259b26ef8d580ddb99ae5826c9501bcdc0c32045945eef9010926fd796e9e` |
| `python/src/jope/policy/experimental_proposal.py` | `f0338f0505a27c9f2f5eac7579e2c506da5c116b3a206f44c52a1cf87cf9c0aa` |
| `python/src/jope/policy/training_campaign.py` | `9d9cfb3257ecf2b6072cab8adc6d830f3df096605a06ff0677e6338c2aad5a2c` |

Pending interfaces: final product postimages of `deploy/install-private.py`, `docs/private-install-demo.md` and panel `{demo,demo_asm,inference}.rs`; stopped add-on postimages of panel `{fitted_export,fitted_export_run}.rs` and `docs/{private-warm-target,private-fitted-qualification}.md`. Exact source and receipt hashes accompany this appendix's custody handoff.

<a id="c-catalogue--machine-readable-companions"></a>

## E.5 Machine-readable companions

<a id="c-catalogue--companion-files"></a>

### E.5.1 Companion files

The manual directory, `docs/internal/asm-manual/`, contains the following locally reproducible evidence. Paths are relative to it, except the property list, which is `docs/asm-maintainer-status.plist` from the repository root. The original metadata/custody paths inside receipts describe where observations were obtained; use the copies under this directory when available. Historical preparation receipts keep their original pre-execution wording; the current state of the examples is the executed result in `data/example-acceptance/result.json` (14 commands, `ok: true`).

| Companion | Purpose |
| --- | --- |
| `data/dataset-inventory.json` | Thirty complete development dataset records and 522 feature records |
| `data/catalog-summary.json` | Corpus task, row, feature, origin and license metadata totals |
| `data/metadata-receipts/` | Catalog JSONL and collection/model metadata; no payload bodies |
| `data/s3-estate.json`, `data/local-estate.json` | Dated current-object and local logical-byte inventories |
| `data/data-estate-extension-summary.json`, `data/data-estate-extension-release.json` | Reconciled aggregate archive/replica/local-root inventories and 19 qualified recovery risks; exact source mappings and raw-packet exclusion boundary |
| `data/performance-evidence.json.gz` | Current proof, historical score cells, reports, ladder identities and scope |
| `data/plot-*.csv` | Compact plot inputs with snapshot and source pins |
| `data/roadmap.json` | Detailed 43-item dependency/acceptance checklist and 24 separately marked coordinated proposals |
| `docs/asm-maintainer-status.plist` | XML property-list representation of the same checklist |
| `data/example-acceptance/` | Actual fourteen-command result, audit and handoff |
| `examples/` | Private synthetic fixture, exact argument packet and complete-output validator |
| `figures/` | Reproducible charts and associated historical data inputs |
| `render-receipt.json` | PDF digest, pages, consumed input hashes and rendering environment |

The machine-readable companions are `dataset-inventory.json` (all thirty datasets, 522 feature records, complete available upstream metadata and detailed development statistics), `performance-evidence.json.gz` (measured current origin plus historical score cells, timing and ladders), `catalog-summary.json`, `local-estate.json`, `s3-estate.json`, and `metadata-receipts/`. The catalog JSONL retains every available per-dataset catalog field for all 3,859 corpus datasets. No claim is made to enumerate every mathematically possible statistic.

<a id="c-catalogue--lossless-evidence-packaging"></a>

### E.5.2 Lossless evidence packaging

The full performance inventory and original catalog JSONL are stored as `data/performance-evidence.json.gz` and `data/metadata-receipts/veox-jopedime-datasets-use2--catalog--v1--catalog.jsonl.gz` to respect the repository’s 5 MiB file limit. Gzip preserves every original byte; no rows, fields, numeric precision or metadata were removed. `data/evidence-packaging.json` records both compressed and original SHA256 values, sizes and preservation paths. Historical handoffs and the initial assembly manifest intentionally retain their original uncompressed paths and hashes; the packaging receipt supplies the current location mapping.

Read either companion directly with Python’s standard library:

```python
import gzip
import json
from pathlib import Path

root = Path("docs/internal/asm-manual/data")
with gzip.open(root / "performance-evidence.json.gz", "rt") as stream:
    performance = json.load(stream)
with gzip.open(root / "metadata-receipts/veox-jopedime-datasets-use2--catalog--v1--catalog.jsonl.gz", "rt") as stream:
    catalog = [json.loads(line) for line in stream if line.strip()]
assert len(catalog) == 3859
```

The plotting tools consume the separately retained CSVs and summary JSON, so this storage change does not change their inputs or measurements. The renderer hashes the compressed companions as well as the packaging receipt.

<a id="c-catalogue--training-history-and-continuation-companions"></a>

### E.5.3 Training-history and continuation companions

<a id="c-catalogue--companion-inventory"></a>

#### E.5.3.1 Companion inventory

The learning expansion is documented in [actual training history](#c-training-history--dated-lineage-and-evidence-coverage), [target history and packing](#c-learning--what-the-historical-training-and-target-evidence-establishes), [versions, KPIs and continuation](#c-plan--the-continuation-interfaces-that-exist-today), and [flywheel/focused search](#c-learning--what-better-and-better-must-mean).

| Companion | Purpose |
| --- | --- |
| `data/training-history/*.csv` | Original loss/diagnostic observations and checkpoint lineage, with source hashes and line/JSON Pointer anchors |
| `data/training-history/metric-definitions.json` | Exact historical loss and KPI meanings; objective changes and missing source-build binding |
| `data/training-history/verification.json` | Original-value verification of 3,218 metrics and 78 checkpoint records |
| `data/training-history/figure-receipt.json` | Input/output hashes for seven historical training figures |
| `data/legacy-training/` | Four original round-1 and reported self-play summary files, source hashes, recomputed diagnostics and a vector-plot receipt; internal Elo is not benchmark-valid |
| `data/per-step-history/` | All 63,000 retained recovery steps and 378,000 exact metric tokens in compressed CSV, 21 raw-log hashes, corpus boundaries and explicit 100-step means |
| `data/per-step-history/independent-review/` | Second complete comparison with original logs, checkpoint identities and all derived block/bin statistics; numerical/source review is separate from final-head CI |
| `figures/per-step-figure-receipt.json` | Source hashes and recomputation checks for all 630 plotted loss bins and 78 adaptive validation points |
| `data/packing-routing-audit.json` | Selected external packing metadata, recomputed sizes and strictly scoped paired final-search observations; excludes unjoined campaign rows |
| `data/original-target-receipts-custody.json` | Durable local archive and verification hashes for all 26,357 original development search receipts; archive stays outside Git, with seeds and off-host backup explicitly excluded |
| `data/target-history/*.csv` | All 159 repeated historical comparison summaries and all 213 illustrative arm/generation points; exact source identities and limits |
| `data/target-history/manifest.json`, `data/target-history/verification.json` | Six corpus inventories, binary-format source pins and original target-observation verification |
| `data/target-history/figure-receipt.json` | Input/output hashes for two historical GP-target figures |
| `data/extension-guide/` | Current-source version/command/API pins, stopped handoff and static/metadata-only verification |
| `examples/continue-python-policy.py` | Current Python resume API illustration; metadata-only by default, numerical branch unexecuted and externally supervised |
| `data/single-dataset-help-receipt.json` | Six actual metadata/help commands, with immutable CLI identity and zero fits |

<a id="c-catalogue--reusing-the-evidence"></a>

#### E.5.3.2 Reusing the evidence

The adjacent `data/training-history` CSVs are long-form original measurements. `block-endpoint-metrics.csv` has 1,296 rows; `block-adaptive-validation.csv` 432; `fixed-monitor-metrics.csv` 16; `gen31-epoch-metrics.csv` 462; `a100-checkpoint-lineage.csv` 78; `a100-checkpoint-validation.csv` 312; and `rtx60100-step-metrics.csv` 700. The two original small logs are retained alongside the normalized data. Source hashes and line/JSON Pointer anchors make a plotted point independently traceable. The initial extraction audit covers four CSVs. The shipped `verify_history.py` checks all 3,218 metric `value` cells across the measurement CSVs, plus seven fields (step, model/parameter/optimizer/corpus hashes, world and global batch) and the state-file hash for each of the 78 lineage rows. It does not check every auxiliary numeric column, such as elapsed time, parent/local step, requested fresh fraction or row counts. Running that helper writes `verification.json` beside the CSVs; use a custody copy when preserving a stopped evidence bundle. Model bodies and sealed target values were not read for this reconstruction.

The additional `data/per-step-history` package preserves the 3,809,970-byte compressed step CSV (SHA `aa431f5a2f5f4d6bfc2e8568ef57bb10b100356750d22eb615dd68cfa1647627`), 21 full source-log hashes, block statistics, explicit 100-step bins and boundary comparisons. `tools/extract_per_step.py` regenerates into a new output directory from the retained raw logs and previously audited checkpoint CSVs; it refuses to overwrite an existing extraction. `tools/make_per_step_figure.py` independently recomputes every plotted bin from the compressed row data before drawing it. Run `python3 docs/internal/asm-manual/tools/make_per_step_figure.py` from the repository root to regenerate this one figure without loading models. The original seven-figure builder remains separate. Both figure receipts bind their exact input hashes.

<a id="c-catalogue--receipts"></a>

## E.6 Receipts

<a id="c-catalogue--important-local-receipts"></a>

### E.6.1 Important local receipts

The custody root is the single host-local directory `/home/ubuntu/jope-rescue-20260921/asm-integration-20260923/`; this manual records no off-host copy of it. These are host-local receipt locations, not promises that every reader has the same files. The structured documentation evidence identifies the source observations copied into this edition. Weight bodies, source dataset bodies and sealed labels are not embedded in the manual.

| Relative custody path | Meaning |
| --- | --- |
| `neural-integration/lifecycle-68f56404c3e199e06413bfd645d3fe0803d7b96f/` | Full author/committer inventory, CI, native reviews, merge and main-guard readbacks |
| `neural-integration/pr83-final-review-68f56404c3e199e06413bfd645d3fe0803d7b96f/` | Independent review prompt, source/proof inspection and native approval receipt |
| `neural-integration/source-match-68f56404c3e199e06413bfd645d3fe0803d7b96f.json` | All 2,855 Git blobs match the immutable build input inventory |
| `neural-integration/native-context-68f56404c3e199e06413bfd645d3fe0803d7b96f/` | Eight native encoder qualification cases under unchanged numerical policy |
| `neural-integration/origin-run-68f56404.json` | Final operational run command, timing and exit status |
| `neural-integration/origin-report-68f56404.json` | Independent recovered-report command and timing |
| `agent-final-acceptance/measured-origin-68f56404/audit.json` | Source/report/worker inventory and independently bound supervisor accounting |
| `neural-integration/final-head-pipeline-bench-68f56404.json` | Local 53 benchmark and 106 pipeline test results |
| `neural-integration/final-head-host-ten-68f56404.json` | Ten actual privileged host cases and binary identity |
| `agent-final-acceptance/failed-origin-d0f5748b/audit.json` | Preserved failed prior run; not an accepted current score |
| `agent-maintainer-manual/` | Section drafts, structured data inventory, example preparation and documentation observations |

Treat a receipt's command, source SHA, input hashes, runtime and original deadline as one evidence unit. Copying a number out of that unit can lose its scope. A local path alone proves neither continued availability nor integrity: verify the retained hash and the contents required for the claim.

<a id="c-catalogue--the-pr83-lifecycle-record"></a>

### E.6.2 The PR83 lifecycle record

| Gate | Exact observation |
| --- | --- |
| PR | `veox-ai/JopeDime#83`, merged |
| Manual CI dispatch | Run 35887284772, attempt 1, source `68f56404…`, success |
| PR CI | Run 35887255898, attempt 1, source `68f56404…`, success |
| Required CI jobs | Eight lanes and aggregate succeeded in both runs |
| CI test inventory | 937 Rust passed, 14 skipped; 221 Python passed per run |
| First independent native review | 5293748183, `neverhuman`, APPROVED, exact full reviewed head, 16:29:01 UTC |
| Second independent native review | 5293788772, same eligible account, APPROVED, same head, 16:32:43 UTC; includes residual disclosures |
| Author eligibility | All twelve commits associated with `jepsontaylor` as author and committer; reviewer differs |
| Merge | Guarded rebase with `--match-head-commit` at the reviewed head, completed at 16:34:47 UTC as main `a75e543d2166…` with the reviewed tree `176b75bc36d7…` |
| Remote-main verification | Fetched main and compared complete Git tree; canonical clean main observed before documentation branch |
| Main guard | Run 35889731910, attempt 1, success at resulting main; guard job 107278860508 |

Two review processes overlapped during coordination. Both were independent of the source writer and both approvals are retained; they are not two different human reviewers. One process's supplementary disclosures identify outstanding work rather than silently treating it as fixed. The writer did not select another credential to approve its own source.

The full-main diagnostic run 35889732032 that the PR83 merge queued was cancelled while still queued to preserve the admitted local allocation and let main-guard run. It did not pass or execute. Earlier full-main run 35852022466 failed and remains historical diagnostic evidence requiring repair. Passing fast CI does not replace that outstanding broad qualification.

<a id="c-catalogue--reproduction-boundaries"></a>

## E.7 Reproduction boundaries

Rendering this manual uses its committed section Markdown, structured data, figure inputs and Python rendering tools. It should not fetch a corpus, load a model, run a fit or contact the sealed store. Plotting historical evidence is distinct from rerunning historical experiments. The included figure data keeps missingness, eligibility and snapshot identities.

The private toy example has its own synthetic input and output receipt. Even a successful five-language recipe/refit comparison cannot qualify a different selected fitted predictor. The pending fitted-inference acceptance must authenticate the retained winner and compare complete saved-state predictions with its original numerical policy.

Operational replay needs more than the document: the exact source/build, immutable workers, model/checkpoint/vocabulary, runtime dependencies, allowed data inputs, original plan and journals, and an available resource allocation. Reproduction that resumes an expired campaign cannot start fresh jobs by assigning a new deadline. A new experiment needs a separately recorded plan; it does not repair the old receipt.

<a id="c-catalogue--names-and-glossary"></a>

## E.8 Names and glossary

ASM is the current product name, settled by the user plan. Historical method IDs such as `afm.lite-race.v1` and `afm.lite-race.v1.resampled` remain exact evidence identifiers in committed same-cut results and the current legacy registry. Use ASM in new prose, but do not rename old identifiers or rewrite frozen reports in place. A method-identity change requires an explicitly versioned new comparison/evidence record. Sources: `crates/jope-bench/src/methods.rs:165`, `:168`; the 2026-09-23 16:07 coordination record.

| Term | Meaning in this repository |
| --- | --- |
| ASM | Algorithm Synthesis Model: learns to generate and optimize executable algorithms from data |
| Algorithm / recipe / program | Executable typed procedure and configuration; reusable across fits. The legacy CLI flag for a recipe is `--genome`. A recipe is not its native fitted predictor, an emitted recipe in another language or a fitted inference bundle ([§](#c-architecture--gp-representation-recipes-fitted-predictors-and-the-four-artifact-types)) |
| Method | A comparable benchmark entrant: a registered native recipe (`rust.*`), an external library adapter (`external.*`), an ASM entrant (`asm.neural`, `asm.generic`) or, in historical ladders, a whole system |
| Candidate | A recipe placed in one search slot |
| GP | Genetic programming: search over executable programs using variation and measured selection |
| Arm | One search population in a campaign, with a fixed proposal source and its own declared allowance: the neural arm (policy pool plus exploration slots) or the generic arm (no policy proposals). Historical searches also ran a shuffled-context arm. Unrelated to the Hyperion asset label `arm_b` (the v5 arm-B weights, `hyperion_v5_arm_b.safetensors`) |
| Fitted predictor | One recipe plus preprocessing and model state learned from exact training inputs |
| Checkpoint | Serialized neural proposal-model parameters and compatibility metadata; distinct from a fitted tabular predictor |
| Context | Authenticated numerical dataset/target representation supplied to the proposal policy |
| Dataset encoder | Learned or specified mapping from permitted dataset structure/rows to context features |
| Target encoder / target context | Representation of allowed target information; task-specific and restricted to permitted fitting rows. In the regression context it is the 512-float block computed from min–max-rescaled training targets. The 1,024 foundation values in the 1,536-float “dataset” block also use the training targets ([§](#c-architecture--component-dimensions-and-assembly)) |
| Vocabulary | Token identities, grammar and executable restrictions associated with a checkpoint |
| Proposal policy | The neural model that proposes program tokens for a context; the accepted regression policy is the lineage-B step-60000 checkpoint |
| Generation zero | Initial candidate slots before evolutionary descendants; 80 per arm in the current ASM search |
| Seed pool / RNG seed | Seed pool: the programs placed into generation zero of an arm (historically including incumbents and retained teachers). RNG seed: a random-number seed, such as seed 7 |
| Exploration candidate | An admitted generic candidate in the neural arm; not falsely labelled neural-origin |
| Offspring | Mutation or crossover descendant with authenticated parent identities and actual fit evidence |
| OOF | Out-of-fold: predict rows using a fit that excluded those rows; adaptive recipe selection can still expose them at another stage |
| Same-cut cuts (historical) | Hashed cuts built by the historical same-cut benchmark command; not comparable with the frozen panel's cuts ([§](#c-benchmark--metrics-and-class-order)) |
| Development-hold score | The score of a final refit on a cut's fitting rows, measured on that cut's held fold (report field `development_hold_score`, `crates/jope-bench/src/panel/origin_proof.rs:321`); the one-dataset proof's RMSE uses 34 held rows |
| Policy-validation families | In the pending teaching partition, the families held out to validate the policy; older text calls them “development” families. Not the panel's development rows |
| Exposure / adaptive | Exposure: use of a partition's labels for selection, optimization or reporting. Adaptive: measured on data that also helped choose what is measured, so not an independent generalization estimate ([§](#c-benchmark--adaptive-development-exposure)) |
| Cut identity | Frozen identity of the training/scoring row assignment and evaluator context |
| Inner folds | Five whole-group folds inside a cut's fitting rows, used for out-of-fold selection and, in the pending eligibility path, for teaching admission ([§](#c-teachers--the-preregistered-freeze-and-train-side-inner-split)) |
| Frozen panel | `asm-private-30-v1` under protocol `asm-panel-v1`: 30 PMLB datasets, ten per task, manifest `ba7dace4…` ([§](#c-data--the-frozen-thirty-dataset-panel-identity-totals-groups-and-immutability)). Older text also says official30 or the unchanged thirty-dataset panel. Training-side validation sets that change between checkpoints are not this panel |
| Task / category | Task: regression, binary classification or multiclass classification. Training text, `--task all-admitted` and the owner's goal of ten wins per category use “category” in this sense; elsewhere “category” means a categorical feature value |
| Label | The column a dataset asks a model to predict; one of the senses of “target” |
| Family | A group of related datasets from one source, such as the UCI computer-hardware source of `230_machine_cpu`. The teaching partition assigns whole families to training or validation. The north star (proposed; owner ratification pending) is measured on previously unseen families ([§](#c-status--the-proposed-north-star-and-matched-comparisons)) |
| Development data | Data visible to optimization or recipe/model selection, regardless of a file's name; in the panel, every non-sealed row |
| Development folds 0–4 | The five original folds of the panel's development rows; every group stays within one fold ([§](#c-data--the-frozen-thirty-dataset-panel-identity-totals-groups-and-immutability)) |
| Cut k | Fit on the development folds other than k and hold out fold k (`dataset.cut(k)`, `crates/jope-bench/src/panel/data.rs:314`). Older text calls cut 0 “fold0” or “development fold zero”. For `230_machine_cpu`, cut 0 has 135 fitting rows and 34 held rows |
| Sealed evaluation | Separately protected evaluation outside the learning and optimization closure. The “seal-protected shared memory” of the training reader (memfd seals) is unrelated |
| Burned partition / burn receipt | In the pending training64 patch: a dataset cut originally in the `test` role whose labels were made visible to optimization, so that they now count as development data, and the immutable `jope.label-burn/v1` receipt that records this relabel. Sealed and `meta_test` labels are refused rather than burned. `require_unburned_test_partition` refuses to present a burned partition as untouched; it has no production caller yet (TEACH-02) |
| HWM | High-water mark: the best retained result for a comparison identity. At `a75e543d2` the current-HWM pointer is keyed only by task, dataset hash and cut (`crates/jope-learning/src/state.rs:13`), so a bound result under a different evaluator, vocabulary or protocol with the same metric can still displace the incumbent (`crates/jope-learning/src/hwm.rs:64`). The pending eligibility/HWM correction (TEACH-01) uses the complete comparison identity (task, dataset, cut, evaluator, split, vocabulary, preprocessing and numerical protocol), so that a result under a different identity cannot displace an unrelated incumbent ([§](#c-teachers--the-admission-guard-label-exposure-and-hwm-identity)) |
| PANEL-WINS (proposed) | **Proposed definition, pending owner ratification; no code at `a75e543d2` computes it.** Per task, the number of the ten frozen datasets on which `asm.neural`'s confirmed five-fold mean development loss is strictly below the minimum over all eligible frozen non-ASM comparators (every `rust.*` and `external.*` recipe). Incomplete, failed or unconfirmed datasets count as not won. `asm.generic` is reported in its own column. Reported as (regression, binary, multiclass); the owner's goal of ten wins per task is (10, 10, 10), a 30/30 sweep ([§](#c-learning--win-counting-and-a-proposed-acceptance-record)) |
| Teaching admission | Eligibility decision permitting a measured example into policy training |
| Motif / macro | Typed reusable program fragment with eligible provenance; not automatically a new primitive operator |
| Numerical policy | Declared arithmetic, probability, ordering, tolerance and runtime assumptions fixed before scoring |
| Alias | Alternate spelling of an existing recipe; not another independent method fit. The regression `ridge` head carries the historical alias `logistic` (`crates/feat-math/src/advanced_v3/grammar.rs:1263`), a known misnomer |
| Front door | The supported CLI entry point for an operation, which enforces its admission rules, identities and budgets |
| Head | Name the sense: PR head commit, reward or attention head, estimator head (such as the ridge head), or serving/training pointer |
| Unsupported | A backend or task deliberately refuses a capability outside its admitted contract |
| Unavailable | A dependency or required runtime is missing; different from a model failure |
| Failed | An attempted operation ended unsuccessfully; retain its reason and measured/unknown costs |
| Skipped | A known test or operation did not execute; not passing coverage |
| Journal | Incremental durable started/finished/observation records for audit and original-window recovery |
| Receipt | Evidence binding command, source, inputs, runtime, resources, times, status and artifacts |
| Provenance | The chain of source, data, model, recipe and execution identities supporting an artifact |
| Custody | Host-local preservation directories outside Git, such as `/home/ubuntu/jope-rescue-20260921/asm-integration-20260923/`, that hold stopped patches, receipts and evidence. “Custody buckets” (also “custody dev/prod”) are something else: the account's separate `jopedime-714044927654-us-east-2-dev`, `-prod` and `-sealed` S3 buckets ([§](#c-data-estate--additional-and-replica-buckets)) |
| Elo | Relative Bradley–Terry rating on the arena's declared method universe and eligible pairwise scores |
| Warm demo | Demonstration after verified preparation; report separately from cold installation |
| Promotion | Explicit qualification of an exact artifact/model/backend for its intended serving use; a merge alone is insufficient |
