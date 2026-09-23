# Current status: what works now and what is pending

An **Algorithm Synthesis Model (ASM) learns to generate and optimize executable algorithms from data.** JopeDime turns that objective into a bounded, inspectable workflow: validate original data and row assignments, encode a fitting-only context, propose executable recipes, fit them, measure them under a fixed protocol, evolve measured recipes, retain fitted artifacts, and eventually admit qualified examples into another round of model training. This chapter states the position on 23 September 2026 at about 22:00 UTC, at merged main `a75e543d2166` (reviewed head `68f56404c3e1`, same source tree).

## ASM at a glance

This table is the bottom line on 23 September 2026 at about 22:00 UTC. Each row links to the sections that hold its evidence. Development scores are adaptive, and historical results do not establish current capability.

| Area | Current state | Evidence |
| --- | --- | --- |
| **Works** | One bounded regression proof on one frozen-panel dataset, `230_machine_cpu`: verified context → 64 policy proposals and 16 exploration proposals → five-fold fits on 20 bounded workers → 16 offspring (8 mutation, 8 crossover) → an independently recovered report. Genuine neural-origin fits and fitted descendants are verified. | [§](#the-one-dataset-proof-record), [§](18-measured-performance.md#proof-record-and-outcome-figure) |
| **Not better yet** | On that dataset the neural arm (policy-seeded search) had 1.071 times the development-hold RMSE of the generic arm (grammar-uniform search): 166.3 against 155.3 on 34 held rows, under equal declared allowances but unequal fit time. In retained historical final searches, whose arms shared their seed pools, the two arms tied exactly on 3,227 of 3,268 paired comparisons (98.7%). Parity of the product's CPU policy port with the trained model is not established. | [§](07-policy-and-search.md#arm-order-and-clocks), [§](20-corpus-and-target-history.md#paired-final-search-outcomes-within-the-admitted-development-inventory), [§](#what-works-now) |
| **Not run** | The frozen thirty-dataset panel (0 of 30 datasets); teacher admission under the new contract (0 teachers); new training (no newly trained checkpoint); consumption of a newly trained checkpoint; exact fitted-winner export; installer and warm demonstration. | [§](#pending-acceptance), [§](12-training-the-policy.md#what-can-be-claimed-now) |
| **Historical standing** | On the corrected, benchmark-valid historical SOTArena ladder the fixed-seed `jope-dime-regressor` entrant ranks 31 of 304 methods (Elo 1687.0; rank 1 is TabFM Ensemble at 1934.9). The published ladder's rank 1 for `jope-dime` rests on test-selected rows and is not a held-out claim. | [§](18-measured-performance.md#why-the-two-historical-ladders-differ) |
| **Next** | The critical order, in short: integrate the guarded follow-on source; run the actual tests, host qualification and packaging; produce fresh qualified contexts and admit preregistered real teachers; build the all-eligible family-disjoint corpus; one GPU-hour of training with continuation; consume the new checkpoint; run the real fitted warm demonstration; then run the unchanged thirty-dataset campaign. Under the current coordination rule, source integration waits until the manual's pull-request lifecycle closes. | [§](03-roadmap.md#critical-order), [§](03-roadmap.md#proposed-owner-decisions-and-external-dependencies) |
| **Risks** | Cross-region protection of the current learning objects is not established: the surveyed learning replica is empty and the `staging/` prefix is not replicated. Several large data classes have no established remote copy, and the disk was 87% used at the dated readback. | [§](C-data-estate.md#recorded-copy-and-retention-gaps) |

## North star: better algorithms from accumulated experience

The product goal is a repeatable learning cycle: verified dataset-and-split inputs become contexts; the policy proposes executable programs; bounded genetic-programming (GP) search fits and improves them; eligible measured programs become teachers; training produces a new checkpoint; and that checkpoint improves a subsequent bounded search. The north star (*proposed; owner ratification pending*) is **better predictive algorithms on previously unseen dataset families at the same declared compute budget**, with reproducible fitted artifacts; [§](13-flywheel.md#the-proposed-north-star-and-matched-comparisons) states it in full. More optimizer steps, a larger corpus, lower training loss or a higher historical Elo are supporting observations; each answers a different question.

Measure progress against both the generic search control and the previous accepted checkpoint, using the same data identities, folds, method set and resource envelope. Report task-specific loss, wins/ties/losses, failure and unavailable rates, and cost per unique fitted predictor. A trained checkpoint earns a promotion claim only through its bound qualification procedure. The frozen thirty-dataset panel is a development comparison; it cannot establish unseen-family performance when prior checkpoint exposure is unknown. Sealed evaluation remains untouched.

There **has been real historical policy training**. The accepted serving model is the recovered lineage-B step-60000 checkpoint. Retained logs cover 63,000 optimizer steps of that lineage (steps 6,001–69,000, in 21 blocks, each on a different corpus). Loss falls inside every block and jumps upward at every corpus change. The descriptive linear fit over steps 6,001–60,000 changes by −0.018284 in total, while the tabulated block losses lie between 0.785951 and 0.855045 ([§](19-training-history.md#within-block-and-boundary-patterns)). The 78 retained checkpoint observations of that lineage use changing evaluation populations. The retained fixed-monitor comparison covers three earlier checkpoints (recovery, cycle 25 and cycle 107), and there sequence and reward-ranking losses did not improve over recovery ([§](19-training-history.md#fixed-monitoring-and-actual-promotion-evidence)). In retained paired final searches, whose arms shared their seed pools, the neural arm tied the generic arm exactly on 3,227 of 3,268 comparisons ([§](20-corpus-and-target-history.md#paired-final-search-outcomes-within-the-admitted-development-inventory)). Across all 26,357 historical search receipts, an external seed audit reports that 98.84% of reported winners were already in their search's seed population ([§](20-corpus-and-target-history.md#first-to-last-comparison-at-1000-evaluations)). The optimizer fitted its objective; better proposals have not been demonstrated.

The new ASM training path has not completed a real bounded acceptance. No eligible real corpus exists for it yet: zero teachers have been admitted under the new contract, and its two prospective teaching families, `230_machine_cpu` and `712_chscase_geyser1`, are both frozen-panel regression datasets ([§](10-teachers-and-targets.md#present-teacher-readiness-and-the-smallest-real-path)). Its pending campaign would train a new, smaller Python model (width 256, two layers), not a continuation of step 60,000 ([§](12-training-the-policy.md#what-the-pending-campaign-trains)). The native command that could continue step 60,000 is held: its GPU-grant check always refuses (`crates/jope-engine/src/task_contract.rs:88-92`; [§](12-training-the-policy.md#the-dispatch-hold-resource-incompatibility-and-valid-topologies)). Zero newly admitted teachers must not be read as an empty historical corpus or as evidence that JopeDime never trained: the accepted checkpoint's retained closure corpus, `snapshot-000097`, holds 10,214,332 target records labelled under the historical `reference_v1` policy ([§](20-corpus-and-target-history.md#six-retained-corpora)).

## What works now

The frozen benchmark foundation consumes actual row assignments, original labels, declared categorical columns, missing values, groups and class order. It uses regression RMSE, binary AUROC and multiclass log loss from a complete probability matrix. [§](09-programs-and-languages.md#complete-registered-recipe-matrix) inventories the executable native recipes, and [§](14-install-and-prepare.md#native-roster-and-external-comparator-runtimes) the four external adapter families, which run at their default configurations.

The shared execution layer has twenty bounded fit slots, resource accounting, independent deadline enforcement, process-tree cleanup, durable journals and original-window resume. Its ten real Linux process/cgroup tests use synthetic workloads to exercise completion, stalled controllers, cancellation, forced process death, memory failure, concurrency and setup failure.

The regression neural path has a qualified native context bridge and a measured proposal-to-fit-to-evolution proof. Policy-origin metadata is checked against pinned producer output, completed inner-fold measurements and measured ancestry. A candidate labelled neural without those links is insufficient. Canonical duplicates retain proposal slots and origins but do not justify fitting the same executable twice.

Numerical parity of the product's policy implementation with the trained model is not established. The product runs a Candle port of the policy on the CPU in float32. The crate states that no numeric parity with the trained tch forward pass is claimed, that its logits and rewards are unverified until reference fixtures exist, and that its `cuda` feature is declared but has never been compiled (`crates/jope-policy/src/lib.rs:84-95`; `crates/jope-policy/src/model.rs:18`). Until such fixtures establish parity, a neural-versus-generic result measures this port, not necessarily the trained model.

Observed generations survive cancellation at the arm cutoff and interruption before the final receipt. Resume reauthenticates completed folds, including completed work inside a partially failed candidate. Only jobs positively proven never admitted can acquire a different expiry status. That exception cannot create a new fit, replace fitted state or extend the clock.

Legacy fit, emit and pipeline surfaces also exist. Their recipe emitters write a program that trains the recipe again; they do not export a fitted predictor; that is the separate, pending portable fitted-state export work. At `a75e543d2` they also cannot emit any registered frozen-panel recipe or executable synthesized ASM recipe: every such recipe carries the raw-table `(input …)` stage, and the legacy planner has no case for it (`crates/jope-bench/src/panel/methods.rs:20-21`, `:50`; `crates/jope-pipeline/src/synthesized_recipe.rs:34-46`; `crates/feat-math/src/spec.rs:383`, `:443`, `:592`). [§](09-programs-and-languages.md#language-interfaces-at-the-frozen-source) gives the detail and [§](09-programs-and-languages.md#what-passed-and-what-was-measured) the worked-example result. In short:

| Language | Recipe emit and refit (legacy emitters) | Exact fitted-state export |
| --- | --- | --- |
| Python, Rust | The emitter's legacy program set, not panel or ASM recipes; in the private worked example the emitted programs refitted `(pipe X (standardize) (ridge))` with bit-identical predictions | Pending (INFER-02); the public fitted Python/Rust packet is unmerged |
| C++17, C++20 | One C++ emitter compiled under both standards; same worked-example result | Pending (INFER-02); the pending public bundle still needs Python `ctypes` at inference |
| JavaScript | ES2020 emitter; same worked-example result | Pending (INFER-02) |
| CUDA | None | Deferred (FUTURE-01) |

## Latest measured operational proof

### The one-dataset proof record

This is **one preselected regression-dataset operational qualification**, not the unchanged thirty-dataset campaign, which has not run. Its scores are development-hold scores: each final refit is scored on the 34 held rows of development fold 0. This is a short summary; the canonical proof record is [§](18-measured-performance.md#measured-current-source-operational-proof).

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
| Interval, baseline and winner origin | Not computed: no confidence interval and no mean-predictor RMSE on the same 34 rows. The neural arm's selected recipe (`ad665e5b…`) is a generation-1 crossover of two generation-0 `random_exploration` candidates, with no `neural_policy` ancestor ([§](18-measured-performance.md#proof-record-and-outcome-figure)) |
| Genuine neural initial fits and fitted descendants | Verified |
| Teachers admitted from this proof | None |
| Full frozen-panel conclusion | Not established |

Table values are rounded; the exact values are in [§](18-measured-performance.md#measured-current-source-operational-proof), and the exact losses and run wall time are also in `data/performance-evidence.json.gz`. The recovered report SHA256 is `6e0c20720256b0735a6969c45032f44082c92abacadf2c4eb21ef5c390f249dd`. The experiment identity is `646095badab6706168c7e869118d62fddd399faa2a9d6bc7f484c95a021a3944`; its original window ran from 2026-09-23T16:16:50.613Z to 16:26:50.613Z (Unix milliseconds `1790180210613` and `1790180810613`). Plan identity: `a7befb386344bd0cbffa866ad8793077dead185f64e2ae0fb4035a4ed3dcfcc2`.

This proves execution and evidence continuity. It does not prove ASM superiority, ten wins per task, learned improvement after training, an under-sixty-second warm demo, installation speed or portable inference for every winner. The arm clocks are explained in [§](07-policy-and-search.md#arm-order-and-clocks); the dense-width refusals (16 of 64 genuine neural-policy candidates against 4 of 80 generic candidates) are in [§](18-measured-performance.md#failures-dense-width-refusals-and-the-earlier-failed-reproducer).

### Cost: interpret the units first

Exact supervisor CPU was **270.889403 CPU-seconds**; summed worker CPU was **181.220773 CPU-seconds**. The supervisor aggregate contains worker use: **do not add them**. Summed worker wall durations were **202.266164543 seconds**; concurrent durations are not elapsed campaign time.

Amortized over 832 unique predictors, these observations yield approximately 0.217813 worker CPU-seconds, 0.325588 aggregate CPU-seconds and 0.287432 run wall-seconds per predictor. Over 164 recipes, they yield approximately 1.105005, 1.651765 and 1.458190 seconds. These are campaign accounting ratios, not isolated latency predictions for arbitrary models.

GPU work, tokens, money, isolated controller CPU, downloads and installation were not measured by this proof. The supplied one-hour/ten-million-token scenario provides a 3,600-second contextual reference: this proof's run wall time is approximately 6.64% of that duration. This is not a matched-workload speedup. No scenario hardware allocation, predictor count, token trace or cost ledger exists here, so ten million tokens cannot be converted into a measured productivity or currency comparison.

## Pending acceptance

| Workstream | Current boundary | Remaining acceptance |
| --- | --- | --- |
| Regression neural consumption | Qualified contexts, proposals, actual fits and descendants on one dataset; parity of the CPU policy port with the trained model not established | Reference-fixture parity with the trained model, full panel and newly trained checkpoint consumption |
| Classification neural consumption | No classification neural arm at the current source: the accepted policy is regression-only, and classification `asm.generic` searches ridge heads only ([§](06-encodings-and-context.md#experimental-classification-context)). Experimental context/invariance support; follow-on producer prepared | Task-correct live consumption and clearly labelled fixture scope |
| High-water mark (HWM) and teaching eligibility | Follow-on source prepared; zero teachers admitted under the new contract | Full identity, finite out-of-fold (OOF) improvement, four-of-five-fold gate and no-signal refusal |
| New ASM training path | Historical trained model exists; new guarded source and test plans prepared. The pending campaign trains a new, smaller Python model, not a continuation of step 60,000 | Positive updates, changed parameters, reload, continuation, resume compatibility and bounded GPU proof for this new path |
| Trained regression export use | Explicit experimental consumer prepared | Compile and execute exported-checkpoint-driven fits |
| Portable fitted inference | Native fitted artifacts; export packet prepared; legacy emitters cannot emit panel or ASM recipes | Full state parity and actual selected-winner backend qualification |
| Installation and warm demo | Source packets prepared | Checksum install, CLI-only build, preparation, fitted artifact/predictions and separate cold/warm timing |
| Thirty-dataset comparison | Frozen panel and protocol; not yet run (0 of 30 datasets) | Unchanged two-hour campaign, every loss/refusal/failure and gap |
| Motifs | Separate typed library branch | Eligible-state reader, mine command, umbrella wiring and validated reuse |
| Broad full-main tests | Historical failures inventoried; the last full-main run that executed (35852022466, at the PR82 merge `300e89c4f`) failed, and run 35889732032, queued by the PR83 merge at `a75e543d2`, was cancelled before it executed ([§](17-maintenance.md#worked-example-the-pr83-lifecycle-record)) | Reproduce and repair; fast CI is not full qualification |
| New serving promotion | Recovered accepted step-60000 remains the model binding; the native command that could continue it is held | Exact release/backend/model acceptance and reviewed promotion for a new checkpoint |
| Manual publication | Edition-1 successor `2022017a8` is local and unpublished; PR84 (head `e06c0551`) is being superseded because three of its intermediate commits contain files above the repository's 5 MiB cap; this edition 2 is prepared for the manual owner to integrate | One published successor with fresh exact-head CI, an independent non-contributor review and a reviewed merge; DOCFIX-01 follows as a separate docs PR |

[§](03-roadmap.md#compact-task-index) and the acceptance cards in [§](A-roadmap-catalogue.md#primary-acceptance-items-by-stage) expand these rows into actionable items; the property list carries the same items. A prepared patch or planned command is not an installed or executed capability.

## Failure records are retained

Two failed cancellation/crash reproductions preceded the resume correction. A partial-fold fixture initially failed on a noncanonical path, then passed after fixing the fixture without weakening verification. A live run at `d0f5748b7db1ac419110289f912c7eface0022bc` completed its worker inventory but failed report verification because live and recovered scoring populated different fold-evidence fields. Its journal remains unchanged. The shared scorer correction and the successful proof at the corrected PR head commit `68f56404c3e1` are separate evidence.

Final local suites at the snapshot passed **53 benchmark and 106 pipeline tests**. One privileged test was ignored in that invocation and qualified separately through **ten actual host cases**. The numerical encoder proof passed **eight fixed cases** under unchanged tolerances and thirty-second per-job limits. [§](17-maintenance.md#worked-example-the-pr83-lifecycle-record) records CI and merge state independently.
