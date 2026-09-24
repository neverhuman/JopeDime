# Current status, versions, north star and measured limitations

An **Algorithm Synthesis Model (ASM) learns to generate and optimize executable algorithms from data.** JopeDime turns that objective into a bounded, inspectable workflow: validate original data and row assignments, encode a fitting-only context, propose executable recipes, fit them, measure them under a fixed protocol, evolve measured recipes, retain fitted artifacts, and eventually admit qualified examples into another round of model training. This chapter states the position on 23 September 2026 at about 22:00 UTC, at merged main `a75e543d2166` (reviewed head `68f56404c3e1`, same source tree).

**Later implementation observation — 2026-09-24.** PR88 integrated the training64, product28, warm-demonstration and winner-export packets, the mandatory regression-consumption gate, and preparation, input-closure, strict-resume and controlled-comparison changes. Reviewed source `8fcf5470cf90f8facb4fac598b67725660448864` merged as main `0fa1841834d1fc67535f021a91a1911a321e6363` at 05:11:50 UTC; exact-head CI 35957350158 attempt 1, independent native approval 5300062934 and main-guard 35958877242 passed. A source-bound runtime build and small CPU fixtures passed. No new real-teacher campaign, GPU window or serving promotion was admitted. The operational learning outcome remains inconclusive. The chapter and roadmap status labels below describe their named PR83/stopped-overlay snapshot, not later main. Use the [later implementation runbook](https://github.com/veox-ai/JopeDime/blob/8fcf5470cf90f8facb4fac598b67725660448864/docs/experimental-learning-preparation.md) for the integrated interfaces; historical commands and evidence are retained here.

## ASM at a glance

This table is the bottom line on 23 September 2026 at about 22:00 UTC. Each row links to the sections that hold its evidence. Development scores are adaptive, and historical results do not establish current capability.

| Area | Current state | Evidence |
| --- | --- | --- |
| **Works** | One bounded regression proof on one frozen-panel dataset, `230_machine_cpu`: verified context → 64 policy proposals and 16 exploration proposals → five-fold fits on 20 bounded workers → 16 offspring (8 mutation, 8 crossover) → an independently recovered report. Genuine neural-origin fits and fitted descendants are verified. | [§](#summary-of-the-one-dataset-proof), [§](#canonical-proof-record-and-per-arm-facts) |
| **Not better yet** | On that dataset the neural arm (policy-seeded search) had 1.071 times the development-hold RMSE of the generic arm (grammar-uniform search): 166.3 against 155.3 on 34 held rows, under equal declared allowances but unequal fit time. In retained historical final searches, whose arms shared their seed pools, the two arms tied exactly on 3,227 of 3,268 paired comparisons (98.7%). Parity of the product's CPU policy port with the trained model is not established. | [§](scheduling.md#arm-order-and-clocks), [§](targets.md#paired-final-search-outcomes-within-the-admitted-development-inventory), [§](#what-works-now) |
| **Not run** | The frozen thirty-dataset panel (0 of 30 datasets); teacher admission under the new contract (0 teachers); new training (no newly trained checkpoint); consumption of a newly trained checkpoint; exact fitted-winner export; installer and warm demonstration. | [§](#pending-acceptance), [§](training.md#what-can-be-claimed-now) |
| **Historical standing** | On the corrected, benchmark-valid historical SOTArena ladder the fixed-seed `jope-dime-regressor` entrant ranks 31 of 304 methods (Elo 1687.0; rank 1 is TabFM Ensemble at 1934.9). The published ladder's rank 1 for `jope-dime` rests on test-selected rows and is not a held-out claim. | [§](targets.md#why-the-two-historical-ladders-differ) |
| **Next** | The critical order, in short: integrate the guarded follow-on source; run the actual tests, host qualification and packaging; produce fresh qualified contexts and admit preregistered real teachers; build the all-eligible family-disjoint corpus; one GPU-hour of training with continuation; consume the new checkpoint; run the real fitted warm demonstration; then run the unchanged thirty-dataset campaign. Under the current coordination rule, source integration waits until the manual's pull-request lifecycle closes. | [§](plan.md#critical-order), [§](plan.md#proposed-owner-decisions-and-external-dependencies) |
| **Risks** | Cross-region protection of the current learning objects is not established: the surveyed learning replica is empty and the `staging/` prefix is not replicated. Several large data classes have no established remote copy, and the disk was 87% used at the dated readback. | [§](data-estate.md#recorded-copy-and-retention-gaps) |

## Reading this manual: audience, evidence levels and figures

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

## North star: better algorithms from accumulated experience

### The proposed north star and matched comparisons

The product goal is a repeatable learning cycle: verified dataset-and-split inputs become contexts; the policy proposes executable programs; bounded genetic-programming (GP) search fits and improves them; eligible measured programs become teachers; training produces a new checkpoint; and that checkpoint improves a subsequent bounded search. The north star (*proposed; owner ratification pending*) is **better predictive algorithms on previously unseen dataset families at the same declared compute budget**, with reproducible fitted artifacts; [§](#the-proposed-north-star-and-matched-comparisons) states it in full. More optimizer steps, a larger corpus, lower training loss or a higher historical Elo are supporting observations; each answers a different question.

Measure progress against both the generic search control and the previous accepted checkpoint, using the same data identities, folds, method set and resource envelope. Report task-specific loss, wins/ties/losses, failure and unavailable rates, and cost per unique fitted predictor. A trained checkpoint earns a promotion claim only through its bound qualification procedure. The frozen thirty-dataset panel is a development comparison; it cannot establish unseen-family performance when prior checkpoint exposure is unknown. Sealed evaluation remains untouched.

**ASM learns to generate and optimize executable algorithms from data.** Its north star (proposed; owner ratification pending) is better downstream predictive loss from generated, actually fitted algorithms on previously unseen dataset families, under the same compute allowance, with reproducible predictions and a usable fitted artifact. This is an operational definition for the next evaluation, not a ratified objective or a claim that the current model has already met it. The frozen-panel win count PANEL-WINS (proposed definition; owner ratification pending; [§](learning.md#kpi-definitions)) is a separate development KPI, not the north star: the panel is adaptive development data whose prior checkpoint exposure is unknown, so it cannot establish unseen-family performance.

Compare each challenger against both a matched generic search and the previous checkpoint, with the same grammar, task, rows, class ordering, preparation, seeds, resources, scoring policy and exposure rules. When the challenger belongs to a different model family, compare it with the previous checkpoint only through such matched downstream fits, never through loss values: the pending bounded campaign trains a width-256, two-layer Python `JopePolicyModelV3` ([§](training.md#what-the-pending-campaign-trains)), not a continuation of the accepted width-512, six-layer step-60,000 model ([§](architecture.md#the-accepted-regression-model-lineage-b-step-60000)). A lower imitation loss alone is insufficient: the generated programs must compile or interpret, fit, survive the resource limits, and improve predictions. Conversely, an honestly measured optimizer update is training progress even when the downstream candidate does not win.

The frozen thirty-dataset panel (asm-private-30-v1; not yet executed) defines an adaptive **development** comparison whose protocol records selection and confirmation exposure. It cannot establish the previously-unseen-family part of the north star merely because a dataset appears in that panel. That requires independently frozen family/exposure evidence and a separately authorized evaluation; this manual grants no access to sealed labels. Historical exposure that cannot be reconstructed stays unknown. See [§](learning.md#what-the-historical-training-and-target-evidence-establishes), [§](targets.md#did-gp-targets-get-better) and [§](learning.md#the-closure-ledger).

### Where the current evidence stands

The latest retained origin proof at source `68f56404` is a useful concrete boundary: it executed actual neural proposals, fitted generation zero, and fitted neural descendants, with an authenticated recovered report. Its neural development-hold RMSE (n = 34) was 166.3181717454848 versus generic 155.25909998168862 on one regression dataset, under equal declared arm allowances but unequal fit time; the neural arm's selected recipe descends only from random-exploration candidates, not from a `neural_policy` proposal. This demonstrates consumption, not a quality win, a quality loss or a thirty-dataset result. The fourteen private language-example commands establish recipe-refit prediction parity across five language configurations; they run no GP evolution and are not exact-state fitted-winner export. Historical trained corpora also must not be confused with the zero newly demonstrated teachers under the pending ASM admission contract. The measured origin report SHA is `6e0c20720256b0735a6969c45032f44082c92abacadf2c4eb21ef5c390f249dd`; see [§](#canonical-proof-record-and-per-arm-facts) and [§](inference.md#measured-execution-of-the-private-examples).

There **has been real historical policy training**. The accepted serving model is the recovered lineage-B step-60000 checkpoint. Retained logs cover 63,000 optimizer steps of that lineage (steps 6,001–69,000, in 21 blocks, each on a different corpus). Loss falls inside every block and jumps upward at every corpus change. The descriptive linear fit over steps 6,001–60,000 changes by −0.018284 in total, while the tabulated block losses lie between 0.785951 and 0.855045 ([§](training-history.md#within-block-and-boundary-patterns)). The 78 retained checkpoint observations of that lineage use changing evaluation populations. The retained fixed-monitor comparison covers three earlier checkpoints (recovery, cycle 25 and cycle 107), and there sequence and reward-ranking losses did not improve over recovery ([§](training-history.md#fixed-monitoring-and-actual-promotion-evidence)). In retained paired final searches, whose arms shared their seed pools, the neural arm tied the generic arm exactly on 3,227 of 3,268 comparisons ([§](targets.md#paired-final-search-outcomes-within-the-admitted-development-inventory)). Across all 26,357 historical search receipts, an external seed audit reports that 98.84% of reported winners were already in their search's seed population ([§](targets.md#first-to-last-comparison-at-1000-evaluations)). The optimizer fitted its objective; better proposals have not been demonstrated.

Historical optimizer evidence for the serving lineage: 21 original A100-recovery logs retain all 63,000 steps from 6,001 through 69,000, with 378,000 selected loss/component/gradient values preserved exactly. All 21 blocks show declining within-block composite loss, while each uses a different corpus. This is historical fitting evidence only, separate from better downstream GP targets or proposal quality, and it implies no new training or promotion. Accepted serving remains step 60,000; later A100 step 69,000 is explicitly refused, and the separately retained world-1 step 60,100 is hardware qualification. The source-bound CSV, corpus boundaries and explicit averaging are in [§](training-history.md#the-complete-a100-loss-series-steps-600169000).

The new ASM training path has not completed a real bounded acceptance. No eligible real corpus exists for it yet: zero teachers have been admitted under the new contract, and its two prospective teaching families, `230_machine_cpu` and `712_chscase_geyser1`, are both frozen-panel regression datasets ([§](teachers.md#present-teacher-readiness-and-the-smallest-real-path)). Its pending campaign would train a new, smaller Python model (width 256, two layers), not a continuation of step 60,000 ([§](training.md#what-the-pending-campaign-trains)). The native command that could continue step 60,000 is held: its GPU-grant check always refuses (`crates/jope-engine/src/task_contract.rs:88-92`; [§](training.md#the-dispatch-hold-resource-incompatibility-and-valid-topologies)). Zero newly admitted teachers must not be read as an empty historical corpus or as evidence that JopeDime never trained: the accepted checkpoint's retained closure corpus, `snapshot-000097`, holds 10,214,332 target records labelled under the historical `reference_v1` policy ([§](data-estate.md#six-retained-corpora)).

## What works now

The frozen benchmark foundation consumes actual row assignments, original labels, declared categorical columns, missing values, groups and class order. It uses regression RMSE, binary AUROC and multiclass log loss from a complete probability matrix. [§](inference.md#complete-registered-recipe-matrix) inventories the executable native recipes, and [§](installation.md#native-roster-and-external-comparator-runtimes) the four external adapter families, which run at their default configurations.

The shared execution layer has twenty bounded fit slots, resource accounting, independent deadline enforcement, process-tree cleanup, durable journals and original-window resume. Its ten real Linux process/cgroup tests use synthetic workloads to exercise completion, stalled controllers, cancellation, forced process death, memory failure, concurrency and setup failure.

The regression neural path has a qualified native context bridge and a measured proposal-to-fit-to-evolution proof. Policy-origin metadata is checked against pinned producer output, completed inner-fold measurements and measured ancestry. A candidate labelled neural without those links is insufficient. Canonical duplicates retain proposal slots and origins but do not justify fitting the same executable twice.

Numerical parity of the product's policy implementation with the trained model is not established. The product runs a Candle port of the policy on the CPU in float32. The crate states that no numeric parity with the trained tch forward pass is claimed, that its logits and rewards are unverified until reference fixtures exist, and that its `cuda` feature is declared but has never been compiled (`crates/jope-policy/src/lib.rs:84-95`; `crates/jope-policy/src/model.rs:18`). Until such fixtures establish parity, a neural-versus-generic result measures this port, not necessarily the trained model.

Observed generations survive cancellation at the arm cutoff and interruption before the final receipt. Resume reauthenticates completed folds, including completed work inside a partially failed candidate. Only jobs positively proven never admitted can acquire a different expiry status. That exception cannot create a new fit, replace fitted state or extend the clock.

Legacy fit, emit and pipeline surfaces also exist. Their recipe emitters write a program that trains the recipe again; they do not export a fitted predictor; that is the separate, pending portable fitted-state export work. At `a75e543d2` they also cannot emit any registered frozen-panel recipe or executable synthesized ASM recipe: every such recipe carries the raw-table `(input …)` stage, and the legacy planner has no case for it (`crates/jope-bench/src/panel/methods.rs:20-21`, `:50`; `crates/jope-pipeline/src/synthesized_recipe.rs:34-46`; `crates/feat-math/src/spec.rs:383`, `:443`, `:592`). [§](inference.md#language-interfaces-at-the-frozen-source) gives the detail and [§](inference.md#what-passed-and-what-was-measured) the worked-example result. In short:

| Language | Recipe emit and refit (legacy emitters) | Exact fitted-state export |
| --- | --- | --- |
| Python, Rust | The emitter's legacy program set, not panel or ASM recipes; in the private worked example the emitted programs refitted `(pipe X (standardize) (ridge))` with bit-identical predictions | Pending (INFER-02); the public fitted Python/Rust packet is unmerged |
| C++17, C++20 | One C++ emitter compiled under both standards; same worked-example result | Pending (INFER-02); the pending public bundle still needs Python `ctypes` at inference |
| JavaScript | ES2020 emitter; same worked-example result | Pending (INFER-02) |
| CUDA | None | Deferred (FUTURE-01) |

## Latest measured operational proof

The current source (reviewed `68f56404`, merged as main `a75e543d2` with the same tree) has a successful operational proof on `230_machine_cpu`: the verified encoder produced context; the neural producer supplied actual generation-zero fits; a fitted descendant followed; two fitted refits each returned 34 development-hold predictions. The neural refit did not beat the generic arm, which had the same declared allowance but more fit time; this is one budget-mismatched sample, not a matched quality comparison. This proof uses the accepted checkpoint, not a newly trained experimental export. As of 2026-09-23, new training, final installed warm demonstrations and the frozen thirty-dataset panel (asm-private-30-v1) had not started, and no current thirty-dataset ranking or Elo exists.

### Summary of the one-dataset proof

This is **one preselected regression-dataset operational qualification**, not the unchanged thirty-dataset campaign, which has not run. Its scores are development-hold scores: each final refit is scored on the 34 held rows of development fold 0. This is a short summary; the canonical proof record is [§](#canonical-proof-record-and-per-arm-facts).

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
| Interval, baseline and winner origin | Not computed: no confidence interval and no mean-predictor RMSE on the same 34 rows. The neural arm's selected recipe (`ad665e5b…`) is a generation-1 crossover of two generation-0 `random_exploration` candidates, with no `neural_policy` ancestor ([§](#canonical-proof-record-and-per-arm-facts)) |
| Genuine neural initial fits and fitted descendants | Verified |
| Teachers admitted from this proof | None |
| Full frozen-panel conclusion | Not established |

Table values are rounded; the exact values are in [§](#canonical-proof-record-and-per-arm-facts), and the exact losses and run wall time are also in `data/performance-evidence.json.gz`. The recovered report SHA256 is `6e0c20720256b0735a6969c45032f44082c92abacadf2c4eb21ef5c390f249dd`. The experiment identity is `646095badab6706168c7e869118d62fddd399faa2a9d6bc7f484c95a021a3944`; its original window ran from 2026-09-23T16:16:50.613Z to 16:26:50.613Z (Unix milliseconds `1790180210613` and `1790180810613`). Plan identity: `a7befb386344bd0cbffa866ad8793077dead185f64e2ae0fb4035a4ed3dcfcc2`.

This proves execution and evidence continuity. It does not prove ASM superiority, ten wins per task, learned improvement after training, an under-sixty-second warm demo, installation speed or portable inference for every winner. The arm clocks are explained in [§](scheduling.md#arm-order-and-clocks); the dense-width refusals (16 of 64 genuine neural-policy candidates against 4 of 80 generic candidates) are in [§](#failures-dense-width-refusals-and-the-earlier-failed-reproducer).

### Canonical proof record and per-arm facts

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
| Work inside the arm clock before any fit | Context encoding and policy production; producer exit observed 41.581 s after campaign start, leaving at most 78.613 s to the neural cutoff ([§](scheduling.md#arm-order-and-clocks)) | No context or proposal production (`proposal: null`) |
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

### The outcome figure and its limits

This is the current bounded operational proof, with one preselected regression dataset and equal declared search-arm allowances but unequal fit time. It demonstrates authenticated neural-origin fits and fitted descendants. The complete thirty-dataset ASM panel has not run, so this figure cannot establish its wins, ranking or Elo.

The current operational figure contains only the two development-hold RMSE values (n = 34) from the preselected `230_machine_cpu` proof: neural 166.3181717454848 and generic 155.25909998168862. Lower is better. Generic has the lower RMSE in this one budget-mismatched, one-dataset sample; it is not a matched quality comparison ([§](#canonical-proof-record-and-per-arm-facts)). It would be misleading to compute a new thirty-dataset Elo from these two values or describe the 832 fitted predictors as 832 independent datasets.

![Current operational proof: neural and generic development-hold RMSE (n = 34)](../figures/current-origin-loss.png)

The two underlying records are in `data/plot-current-origin.csv`; exact evidence and denominators are in `data/performance-evidence.json.gz`. There are 34 predictions per final refit. The proof's run wall time is 239.143 seconds; its independent recovered report takes another 18.719 seconds. CPU, wall-time and per-predictor accounting follows in [§](scheduling.md#worked-example-cpu-wall-time-and-per-predictor-cost-of-the-one-dataset-proof); the status summary is [§](#latest-measured-operational-proof).

### Failures: dense-width refusals and the earlier failed reproducer

Every one of those 130 failed commands exited 1 with a verified stderr message `generated dense width N exceeds 512`. They represent 26 distinct candidate recipes with five failed folds each. Generation zero included 16 of 64 genuine neural-policy candidates failing this way (25%), compared with 4 of 80 generic candidates (5%); the remaining failures came from exploration or offspring. These jobs reached their process gates, but the failed candidates lack complete physical-fit counters, so the corresponding per-origin total remains unknown rather than inferred as zero or five. The 512-column check occurs inside fit execution; a verified prelaunch width rejection remains an optimization opportunity. Sources: `crates/feat-math/src/advanced_v3/feature_block.rs:109`, `crates/feat-math/src/advanced_v3/grammar.rs:2299-2300`; retained 130 finished/stderr hashes and report origin counts.

The earlier failed reproducer at source `d0f5748b7db1` failed after 194.789382994 seconds at scorer replay. All 964 commands had terminal receipts, with 834 completed and 130 failed. Its exact supervisor CPU was 252.614611 seconds and worker CPU 181.863951 seconds. It remains a failed reproducer; the old journal and deadline were not repaired or reopened.

## Failure records are retained

Two failed cancellation/crash reproductions preceded the resume correction. A partial-fold fixture initially failed on a noncanonical path, then passed after fixing the fixture without weakening verification. A live run at `d0f5748b7db1ac419110289f912c7eface0022bc` completed its worker inventory but failed report verification because live and recovered scoring populated different fold-evidence fields. Its journal remains unchanged. The shared scorer correction and the successful proof at the corrected PR head commit `68f56404c3e1` are separate evidence.

Final local suites at the snapshot passed **53 benchmark and 106 pipeline tests**. One privileged test was ignored in that invocation and qualified separately through **ten actual host cases**. The numerical encoder proof passed **eight fixed cases** under unchanged tolerances and thirty-second per-job limits. [§](catalogue.md#the-pr83-lifecycle-record) records CI and merge state independently.

## Pending acceptance

| Workstream | Current boundary | Remaining acceptance |
| --- | --- | --- |
| Regression neural consumption | Qualified contexts, proposals, actual fits and descendants on one dataset; parity of the CPU policy port with the trained model not established | Reference-fixture parity with the trained model, full panel and newly trained checkpoint consumption |
| Classification neural consumption | No classification neural arm at the current source: the accepted policy is regression-only, and classification `asm.generic` searches ridge heads only ([§](architecture.md#experimental-classification-context)). Experimental context/invariance support; follow-on producer prepared | Task-correct live consumption and clearly labelled fixture scope |
| High-water mark (HWM) and teaching eligibility | Follow-on source prepared; zero teachers admitted under the new contract | Full identity, finite out-of-fold (OOF) improvement, four-of-five-fold gate and no-signal refusal |
| New ASM training path | Historical trained model exists; new guarded source and test plans prepared. The pending campaign trains a new, smaller Python model, not a continuation of step 60,000 | Positive updates, changed parameters, reload, continuation, resume compatibility and bounded GPU proof for this new path |
| Trained regression export use | Explicit experimental consumer prepared | Compile and execute exported-checkpoint-driven fits |
| Portable fitted inference | Native fitted artifacts; export packet prepared; legacy emitters cannot emit panel or ASM recipes | Full state parity and actual selected-winner backend qualification |
| Installation and warm demo | Source packets prepared | Checksum install, CLI-only build, preparation, fitted artifact/predictions and separate cold/warm timing |
| Thirty-dataset comparison | Frozen panel and protocol; not yet run (0 of 30 datasets) | Unchanged two-hour campaign, every loss/refusal/failure and gap |
| Motifs | Separate typed library branch | Eligible-state reader, mine command, umbrella wiring and validated reuse |
| Broad full-main tests | Historical failures retained; full-main runs 35929010741 and 35936673628 failed, each with eight families passed, two failed and one unavailable. The earlier PR83 record remains historical ([§](catalogue.md#the-pr83-lifecycle-record)) | Reproduce and repair; fast CI is not full qualification |
| New serving promotion | Recovered accepted step-60000 remains the model binding; the native command that could continue it is held | Exact release/backend/model acceptance and reviewed promotion for a new checkpoint |
| Manual publication | Edition 1 landed through PR85 at `bdfc95c63`; DOCFIX-01 landed through PR86 at `2b375ef11`. PR84 was superseded because its intermediate history exceeded the 5 MiB file cap. Edition 2 landed through PR87 at `02c0bef6c` after exact-head CI and independent native approval; main-guard passed | Each later revision needs fresh exact-head CI, independent native approval and reviewed merge |

[§](plan.md#compact-task-index) and the acceptance cards in [§](catalogue.md#primary-acceptance-items-by-stage) expand these rows into actionable items; the property list carries the same items. A prepared patch or planned command is not an installed or executed capability.

## What version are we running?

### Implementation snapshot and this manual's own identity

The implementation snapshot is **68f56404c3e199e06413bfd645d3fe0803d7b96f**, dated 23 September 2026. [§](catalogue.md#the-pr83-lifecycle-record) records the separately refreshed reviewed-main state. A rebase merge can produce a different commit with the same reviewed tree. Older plans and historical PASS entries do not establish current capability.

This edition describes the implementation reviewed at `68f56404c3e199e06413bfd645d3fe0803d7b96f`. PR83 merged on 23 September 2026 at 16:34:47 UTC. The resulting remote main is `a75e543d2166017cb0139e0a251d51d9a0c98e08`. Both revisions have Git tree `176b75bc36d76c7ce4eaeb7234677ec013468889`. Rebase merge changed commit identity; it did not change the reviewed source tree.

The manual itself is a later documentation change. Its publication identity must be read from its own Git history and CI receipt, not confused with the implementation snapshot. Follow-on patches described as pending are outside the implementation snapshot even if they are preserved on this host.

Publication history, refreshed after the PR87 merge; the application snapshot above stays pinned to PR83:

- **PR84 was closed as superseded** (coordination record of 22:22:38 UTC). Its head is `e06c0551e3ec5ee5260debb3d953c17f34d15093`. The independent non-contributor review 5295763455 (state COMMENTED, 19:30:12 UTC, at that head) withheld approval for two reasons: a blocking finding that three intermediate commits each contain two files larger than the repository's 5 MiB cap (`deploy/ci/check.sh:46`), so a rebase merge would put those commits on `main`; and exact-head CI that was not yet fully green.
- **PR85, the clean-history successor, merged at 22:33:36 UTC.** Codex, the manual owner, published the single commit `2022017a8c45cc25967ae99ae781ca26276140b8` on `a75e543d2` (branch `docs/asm-maintainer-manual-final`, pushed at 22:21:30 UTC) with a 203-page PDF (SHA-256 `69418c48…`). Exact-head CI run 35927921851 attempt 1 succeeded, and the independent non-contributor review 5297559032 approved that head at 22:32:39 UTC. The merge produced `main` `bdfc95c63d905c647e41175e79dc30cdc8600287`, whose Git tree `fbbed08a…` is identical to that of `2022017a8`.
- **Edition 2** restructures and corrects `2022017a8`. It was prepared in Claude's lane and integrated by the manual owner, preserving the authoring commits. Edition 2 merged through PR87 at 2026-09-24T02:58:00Z as main `02c0bef6c81ab17e034daea308c1a0b5a7940467`, after exact-head CI 35948653955 attempt 1 and independent native approval 5299139743; main-guard 35949456287 passed. Further corrections repeat the reviewed lifecycle.

### The identity tuple

There is no single version number that describes all of the following. Record the entire identity tuple whenever comparing results or continuing training.

| Dimension | Current evidenced identity | What changing it means |
|---|---|---|
| Product/package | Rust workspace and Python package `0.1.0`; immutable CLI reports `jope 0.1.0` | A packaging/API release number; not a checkpoint generation or proof of installed capability. |
| Implementation | Reviewed source `68f56404c3e199e06413bfd645d3fe0803d7b96f`, merged as `a75e543d2166017cb0139e0a251d51d9a0c98e08` with the same source tree | A later documentation commit does not rebuild the binary. Preserve binary/build/source-manifest hashes. |
| Accepted regression policy | Lineage B, step 60000, `parallel.checkpoint.v1`; state hash `8ac80c269cc47c253b6f53fa3bc1eb75bb2fcda0bdace3ff8554be0175c400d6` | The current accepted consumer hard-pins this identity and its model/configuration/parameters. Pinning does not establish numerical parity: the Candle CPU port claims no parity with the trained tch forward pass, its logits and rewards are unverified, and its `cuda` feature has never been compiled (`crates/jope-policy/src/lib.rs:84-95`). |
| Hardware-qualification continuation | Step 60100 is retained hardware-qualification continuation evidence, state SHA `745be4b5e998bce85615349a5b03b901cd8ad2cf605e6dce3b86fbf777cc7388`; verify the actual training pointer (the `training-head` key) independently | This is not the accepted serving checkpoint. Do not advance a pointer from a filename or step number. |
| Rejected policy | Step 69000 is explicitly refused by state or model hash | A numerically higher training step does not imply promotion. |
| Python model family | Current Python implementation declares `policy_v3.4-split-context`, with legacy `policy_v3.3-fused-heads`; trainer resume format `PrimePolicyTrainerResume.v1` | These identify a different trainer/model serialization surface from the accepted native `model.ot` lineage. |
| Experimental ASM policy (pending) | The stopped training64 patch would train a new Python `JopePolicyModelV3` (width 256, 2 layers, 4 heads, feed-forward 1024), initialized at most from a Python checkpoint; no real campaign has run with it | Experimental export only: not serving, not lineage B and not a continuation of step 60000, which the Python path cannot load. Compare the two only through matched downstream fits, never through loss values. The native route that would continue step 60000, `jope-parallel`, is held ([§](training.md#the-dispatch-hold-resource-incompatibility-and-valid-topologies)). |
| Encoder | Current CPU (Rust/Candle FP32) context schema `jope.cpu-context/v1`, separate from the held native GPU path; produced artifacts require pinned native-fixture replay before qualification. Dataset 1536 (Hyperion 512 from the features only + projected foundation 1024 from the features and training targets) + target 512 = 2048 floats, so “dataset 1536” is a historical label that includes target-conditioned values; legacy frozen encoder descriptor `parallel.frozen_encoder.v1` | Every component weight, transform, source/backend and assembly hash participates in identity. Dimensions alone do not establish compatibility. |
| Hyperion assets | Current CPU producer pins the v5 arm-B weight hash `88b9e8345e44d3ea609a8286215618a762f14e607fcdada6d27447b1f74668be`; a separately stored bundle is called `hyperion-v7` | A bundle name or Rust `V7Model` type is not permission to substitute newer-looking weights. “v5” here is asset lineage, not a global product schema version. |
| Action vocabulary | Accepted 401-token vocabulary SHA `8d55fee4366acd2c175cf55bfd769cebc904770f5227a6fd50a613723dc85787`; documented product restriction SHA `570b2e5e8d859a73b888f5d0bee27f59da0e2376536babce24c241604f818e81` | The restriction preserves the token table. Adding tokens or changing parameter bins requires a new compatibility and training decision. |
| Corpus | Native `parallel.teaching_shards.v1` / `parallel.teaching_corpus.v1`, plus immutable index, vocabulary, contexts, records, label policy and exposure identities | A corpus schema does not certify teacher eligibility, data freshness, or unseen-family status. |
| Dataset/catalog | `jope.dataset-catalog/v1`, `jope.dataset-collection/v1`; individual content and split hashes remain separate | Keeping the schema while changing rows, labels, categories, groups or splits produces a different comparison. |
| Model packaging | `jope.model-bundle/v2` | Package/license/inventory metadata is separate from accepted model identity and successful inference. |
| Frozen benchmark | `asm-panel-v1:screen-fold=0:top=5:confirm=all-5:mean-fold-loss`; evaluator `jope-bench.frozen-panel-f64/v1`; manifest SHA `ba7dace4cd535687002cc95538dbc7f7aaf77ec973177c7159b3c6d22471eee0` | Altering a method, split, deadline, evaluator or selection rule requires a new protocol identity. Do not relabel an altered run as the frozen panel. |

Source: [workspace version](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/Cargo.toml#L30), [Python package](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/python/pyproject.toml#L5), [accepted checkpoint and vocabulary pins](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-policy/src/lib.rs#L132), [Python model version](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/python/src/jope/policy/model_v3.py#L43), [resume schema](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/python/src/jope/policy/train_v3.py#L57), [context assets and identity](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-policy/src/context_producer.rs#L35), [legacy encoder descriptor](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-engine/src/encoding.rs#L47), [catalog schemas](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-datasets/src/catalog.rs#L9), [bundle schema](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-models/src/manifest.rs#L9), [panel identity](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-bench/src/panel/mod.rs#L36).
