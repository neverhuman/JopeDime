# Reading the system and its current status

## Purpose and audience

An **Algorithm Synthesis Model (ASM) learns to generate and optimize executable algorithms from data.** JopeDime turns that objective into a bounded, inspectable workflow: validate original data and row assignments, encode a fitting-only context, propose executable recipes, fit them, measure them under a fixed protocol, evolve measured recipes, retain fitted artifacts, and eventually admit qualified examples into another round of model training.

This manual serves operators, scientists, implementation maintainers and reviewers. Operators need installation, preparation, execution, reporting and recovery procedures. Scientists need dataset identities, split boundaries, metrics, model lineage, comparison budgets and limits of the evidence. Maintainers need the workspace map, command contracts, fitted-state formats, feature boundaries and tests. Reviewers need to distinguish code that exists, behavior that has been demonstrated, changes that have merged, and claims still requiring evidence.

The implementation snapshot is **68f56404c3e199e06413bfd645d3fe0803d7b96f**, dated 23 September 2026. The lifecycle appendix records the separately refreshed reviewed-main state. A rebase merge can produce a different commit with the same reviewed tree. Older plans and historical PASS entries do not establish current capability.

This is a maintainer document for the private product repository. Its explicitly requested GP examples and internal operational paths explain the implementation; they do not expand the allowlist for public inference bundles.

## North star: better algorithms from accumulated experience

The product target is a repeatable learning cycle: verified dataset-and-split inputs become contexts; the policy proposes executable programs; bounded GP fits and improves them; eligible measured programs become teachers; training produces a new checkpoint; and that checkpoint improves a subsequent bounded search. The north star is **better predictive algorithms on previously unseen dataset families at the same declared compute budget**, with reproducible fitted artifacts. More optimizer steps, a larger corpus, lower training loss or a higher historical Elo are supporting observations; each answers a different question.

Measure progress against both the generic search control and the previous accepted checkpoint, using the same data identities, folds, method set and resource envelope. Report task-specific loss, wins/ties/losses, failure and unavailable rates, and cost per unique fitted predictor. A trained checkpoint earns a promotion claim only through its bound qualification procedure. The frozen thirty-dataset panel is a development comparison; it cannot establish unseen-family performance when prior checkpoint exposure is unknown. Sealed evaluation remains untouched.

There **has been real historical policy training**. The accepted model is the recovered lineage-B step-60000 checkpoint, and retained historical records describe substantial optimizer work and multiple teaching-corpus versions. The newer ASM training/teaching integration has not yet completed its own real bounded acceptance. Its zero newly admitted teachers must not be read as an empty historical corpus or as evidence that JopeDime never trained. The training-history and extension chapters separate those timelines, show the actual available losses, and explain which version each result belongs to.

## How to read a completion claim

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

## Working implementation

The frozen benchmark foundation consumes actual row assignments, original labels, declared categorical columns, missing values, groups and class order. It uses regression RMSE, binary AUROC and multiclass log loss from a complete probability matrix. The method chapter inventories executable native recipes and the four external adapter families.

The shared execution layer has twenty bounded fit slots, resource accounting, independent deadline enforcement, process-tree cleanup, durable journals and original-window resume. Its ten real Linux process/cgroup tests use synthetic workloads to exercise completion, stalled controllers, cancellation, forced process death, memory failure, concurrency and setup failure.

The regression neural path has a qualified native context bridge and a measured proposal-to-fit-to-evolution proof. Policy-origin metadata is checked against pinned producer output, completed inner-fold measurements and measured ancestry. A candidate labelled neural without those links is insufficient. Canonical duplicates retain proposal slots and origins but do not justify fitting the same executable twice.

Observed generations survive cancellation at the arm cutoff and interruption before the final receipt. Resume reauthenticates completed folds, including completed work inside a partially failed candidate. Only jobs positively proven never admitted can acquire a different expiry status. That exception cannot create a new fit, replace fitted state or extend the clock.

Legacy fit, emit and pipeline surfaces also exist. Their recipe emitters can train a recipe again. That is distinct from the pending portable fitted-state export work; consult the language matrix before assuming an emitted artifact needs no refitting.

## Latest measured operational proof

This is **one preselected regression-dataset operational qualification**, not the unchanged thirty-dataset demonstration. Its scores are development scores.

| Item | Observation |
| --- | --- |
| Dataset | `230_machine_cpu` |
| Initial neural arm | 64 actual `neural_policy` slots, zero fallback slots, plus 16 `random_exploration` slots in this proof |
| Initial generic arm | 80 generic slots |
| Arm / fit allowance | 120 seconds per arm; 30 seconds per worker fit |
| Global proof window | 600 seconds, including 30 seconds for reporting |
| Host limits | 20 CPUs/workers, 64 GiB, zero swap, no GPU tasks |
| Started and finished worker commands | 964 |
| Completed / failed commands | 834 / 130 |
| Unique fitted predictors / reusable recipes | 832 / 164 |
| Final refits / predictions per refit | 2 / 34 |
| Run-command wall time | 239.143121970 seconds |
| Independent report replay | 18.718997325 seconds |
| Neural development RMSE | 166.3181717454848 |
| Generic development RMSE | 155.25909998168862 |
| Score ordering | Generic has lower RMSE in this comparison |
| Genuine neural initial fits and fitted descendants | Verified |
| Teachers admitted from this proof | None |
| Full frozen-panel conclusion | Not established |

The recovered report SHA256 is `6e0c20720256b0735a6969c45032f44082c92abacadf2c4eb21ef5c390f249dd`. The experiment identity is `646095badab6706168c7e869118d62fddd399faa2a9d6bc7f484c95a021a3944`; its original start/deadline are Unix milliseconds `1790180210613` and `1790180810613`. Plan identity: `a7befb386344bd0cbffa866ad8793077dead185f64e2ae0fb4035a4ed3dcfcc2`.

This proves execution and evidence continuity. It does not prove ASM superiority, ten wins per category, learned improvement after training, an under-sixty-second warm demo, installation speed or portable inference for every winner.

## Cost: interpret the units first

Exact supervisor CPU was **270.889403 CPU-seconds**; summed worker CPU was **181.220773 CPU-seconds**. The supervisor aggregate contains worker use: **do not add them**. Summed worker wall durations were **202.266164543 seconds**; concurrent durations are not elapsed campaign time.

Amortized over 832 unique predictors, these observations yield approximately 0.217813 worker CPU-seconds, 0.325588 aggregate CPU-seconds and 0.287432 run wall-seconds per predictor. Over 164 recipes, they yield approximately 1.105005, 1.651765 and 1.458190 seconds. These are campaign accounting ratios, not isolated latency predictions for arbitrary models.

GPU work, tokens, money, isolated controller CPU, downloads and installation were not measured by this proof. The supplied one-hour/ten-million-token scenario provides a 3,600-second contextual reference: this proof's run wall time is approximately 6.64% of that duration. This is not a matched-workload speedup. No scenario hardware allocation, predictor count, token trace or cost ledger exists here, so ten million tokens cannot be converted into a measured productivity or currency comparison.

## Pending acceptance

| Workstream | Current boundary | Remaining acceptance |
| --- | --- | --- |
| Regression neural consumption | Qualified contexts, proposals, actual fits and descendants | Full panel and newly trained checkpoint consumption |
| Classification neural consumption | Experimental context/invariance support; follow-on producer prepared | Task-correct live consumption and clearly labelled fixture scope |
| HWM and teaching eligibility | Follow-on source prepared | Full identity, finite OOF improvement, four-of-five-fold gate and no-signal refusal |
| New ASM training path | Historical trained model exists; new guarded source and test plans prepared | Positive updates, changed parameters, reload, continuation, resume compatibility and bounded GPU proof for this new path |
| Trained regression export use | Explicit experimental consumer prepared | Compile and execute exported-checkpoint-driven fits |
| Portable fitted inference | Native fitted artifacts; export packet prepared | Full state parity and actual selected-winner backend qualification |
| Installation and warm demo | Source packets prepared | Checksum install, CLI-only build, preparation, fitted artifact/predictions and separate cold/warm timing |
| Thirty-dataset comparison | Frozen panel and protocol | Unchanged two-hour campaign, every loss/refusal/failure and gap |
| Motifs | Separate typed library branch | Eligible-state reader, mine command, umbrella wiring and validated reuse |
| Broad full-main tests | Historical failures inventoried | Reproduce and repair; fast CI is not full qualification |
| New serving promotion | Recovered accepted step-60000 remains the model binding | Exact release/backend/model acceptance and reviewed promotion for a new checkpoint |

The detailed checklist and property-list expand these rows into actionable items. A prepared patch or planned command is not an installed or executed capability.

## Failure records are retained

Two failed cancellation/crash reproductions preceded the resume correction. A partial-fold fixture initially failed on a noncanonical path, then passed after fixing the fixture without weakening verification. A live run at `d0f5748b7db1ac419110289f912c7eface0022bc` completed its worker inventory but failed report verification because live and recovered scoring populated different fold-evidence fields. Its journal remains unchanged. The shared scorer correction and successful new-head proof are separate evidence.

Final local suites at the snapshot passed **53 benchmark and 106 pipeline tests**. One privileged test was ignored in that invocation and qualified separately through **ten actual host cases**. The numerical encoder proof passed **eight fixed cases** under unchanged tolerances and thirty-second per-job limits. The lifecycle appendix records CI and merge state independently.
