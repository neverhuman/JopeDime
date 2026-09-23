# Closing the flywheel and pushing one dataset further

## What “better and better” must mean

A genuine flywheel joins four measured improvements. GP finds a better executable teacher for an unchanged comparison. Training actually consumes that new information. The policy becomes better at proposing or ranking programs for the relevant dataset-and-split context. Finally, the next bounded GP search improves because of that policy. Each arrow needs its own evidence; a growing corpus, a saved checkpoint or a completed search cannot stand in for the whole chain.

The comparison identity is essential. A target belongs to a task, dataset content, exact split and groups, evaluator, vocabulary and numerical protocol. If any changes, the score is a new experiment rather than a better version of the old target. Context identity also binds the fitted preprocessing, encoder assets and allowed fitting rows. Two vectors with the same width need not represent the same problem.

An incumbent can improve monotonically under a fixed development objective while real generalization gets worse through repeated selection. We therefore need both a target-quality ratchet on development data and independent evidence that the learned search transfers to other families. The latter is the north star. Repeatedly attacking one dataset is useful specialization; it is not evidence of broad transfer.

## The closure ledger

| Link in the loop | Existing evidence | Remaining proof | Roadmap connection |
| --- | --- | --- | --- |
| Dataset and split → valid context | Current regression producer checks original rows, preprocessing, assets and eight reference cases | Classifier context/checkpoint consumption and newly trained context compatibility | NEURAL / TRAIN items |
| Context → real initial GP candidates | Current 80-slot path (64-slot policy pool plus 16 exploration, with explicit rejection fallback) has authenticated genuine `neural_policy` fits | Show the newly trained checkpoint produced the measured candidates | NEURAL / TRAIN items |
| GP evaluation → improved target | Current bounded search fits and evolves candidates; historical champion and corpus records exist | New admission needs complete finite OOF evidence, baseline improvement on at least 4/5 folds and the full identity | TEACH items |
| Improved target → frozen teaching corpus | Historical binary teaching stores and corpora survive; pending exporter authenticates inputs | Execute the new eligible-state export with complete admitted-family coverage; record fresh versus repeated examples | TEACH / TRAIN items |
| Corpus → parameter update | Historical optimizer logs and accepted step-60000 exist | New bounded run must prove positive updates, changed parameters, reload and true optimizer continuation | TRAIN items |
| Parameter update → better policy | Historical imitation/ranking diagnostics exist | Compare parent and challenger on unchanged diagnostic inputs; then measure fitted-program benefit under matched budgets | NEURAL / TRAIN items |
| Better policy → better next search | Current accepted model reaches actual evolution on one dataset | Run the exported challenger in a subsequent search and retain a causally interpretable comparison | NEURAL / PANEL items |
| Improved search → next useful training generation | Historical controller and target chains exist | Complete at least two linked new cycles with eligible new targets, actual exposure and measured downstream outcomes | End-to-end flywheel qualification |
| Qualified checkpoint → serving | Product is bound to recovered accepted step-60000 | Design and qualify the missing experimental-to-serving promotion contract, then require the new checkpoint to pass it; training-head movement alone does not move serving | FUTURE-01: deferred; no executable acceptance command yet |

The prefixes above refer to the detailed roadmap's existing workstreams; they are not new executable command names. The current one-dataset proof closes context→proposal→fit→evolution. It does not close the new teaching→training→improved-search cycle. The historical training chapter separately reports what earlier controllers actually achieved, including unavailable or inconclusive promotion evidence.

## A minimal experiment that can establish learning

Freeze the experiment before opening the run window: source and binaries, exact parent checkpoint, corpus membership, encoder and vocabulary, objective weights, optimizer carry policy, sampler, random streams, fit/search budgets and comparison metrics. Keep the accepted parent available as a control. A new objective, new encoding or incompatible optimizer state starts a new lineage; it must not inherit a misleading uninterrupted-step counter.

Use eligible teachers from independently identified training families. Keep family-disjoint development diagnostics for the model. When a dataset was used to manufacture teachers, label subsequent evaluations on that dataset as exposed/adaptive. Unknown historical checkpoint exposure remains unknown. A split name alone cannot prove an unseen family.

For each cycle, retain all of the following:

1. **Target creation:** incumbent and challenger recipe hashes, all five candidate and baseline OOF losses, complete fit outcomes and exact comparison identity. Record rejected, tied, failed and unavailable candidates too.
2. **Corpus delta:** previous/new corpus hashes; added, improved, removed and unchanged comparison/program identities; distinct contexts and families; rejection reasons. Distinguish logical rows from unique programs and unique contexts.
3. **Actual optimizer exposure:** how often newly admitted programs and contexts appeared in committed steps, their weights, sample repetition, padding fraction and fresh/broad sampling fractions. New rows in a manifest are insufficient if the sampler never reaches them.
4. **Training continuity:** initialization parent, parameter hashes before/after, optimizer initialization versus restored moments, local and cumulative committed steps, effective batch and actual clock/resource use. A failed attempt's compute stays in accounting.
5. **Policy diagnostics:** component losses on a fixed, identified monitor; valid-program rate, uniqueness, conditional ranking and proposal quality. Report corpus/monitor changes as boundaries rather than drawing a continuous evaluation curve through them.
6. **Next-search outcome:** execute both parent and challenger proposals as actual fits under the same candidate/compute allowance, plus the generic control. Retain neural-origin and ancestry counts, failure rates, selected program identities and the downstream losses.
7. **Decision:** accept, reject or inconclusive with the exact gate and original evidence. Preserve the old serving pointer unless promotion succeeds.

Two linked successful cycles are a practical demonstration of repeatability. They do not prove unlimited improvement. Plateaus, search-space ceilings, noisy estimates, sample reuse and model capacity remain real constraints. Report the improvement per cycle and cost per eligible improvement; do not promise a monotonic performance curve on unseen data.

## Distinguish routing, ranking and search recovery

The learned route is a conditional distribution over executable program tokens given the dataset-and-target context. It is richer than selecting a fixed named estimator. The same model also supplies learned ranking signals. Its historical reward representation mean-pools independent token embeddings, a bag of actions. It has no learned representation of token order: for the same context, programs with the same token multiset have the same pooled representation in exact arithmetic. This limits ranking distinctions between programs whose execution differs only through token order; it does not remove order from the autoregressive proposal decoder. See the [architecture chapter](02-architecture.md).

Measure the policy before GP recovery as well as after it. A long search can find the same winner despite poor neural proposals, while a short search may reveal useful initialization that disappears at a larger budget. Shared incumbents can create many ties. Equal time caps do not imply equal completed fit counts, especially when candidate runtimes differ.

Useful controls answer different questions: generic grammar sampling tests end-to-end benefit over the non-neural search; the previous accepted checkpoint tests whether training helped; shuffled contexts test conditional information; and random ranking on the same candidate pool tests ranking utility. These additional controls are proposed experiments, not extra runs hidden inside the approved panel. Declare their budget and scope before execution.

## Commands that exist today for one dataset

The current private Linux `asm` build supports a **separately bounded operational proof on one frozen regression dataset**. This is the closest current neural search front door to focused optimization. It is not an arbitrary-dataset ingestion command, a daemon, or an incumbent-seeded high-water-mark attack.

The following metadata commands are safe preparation and were checked against the immutable reviewed68 CLI:

```sh
"$JOPE" optimize capabilities
"$JOPE" optimize proof-plan --help
"$JOPE" optimize run --help
"$JOPE" optimize report --help
```

`compiled:true` means support was compiled, not that assets, context qualification or training have passed. Use a binary whose build inventory, source manifest and executable hashes match the configuration. The current implementation caps an arm at 120 seconds and its candidate allowance at 80–1000 evaluations. Per-fit time remains at most 30 seconds. Editing a JSON number beyond those caps does not extend the implementation.

### Required inputs

| Variable | Meaning |
| --- | --- |
| `JOPE` | Authenticated Linux CLI built with `asm`; not an unrelated binary found on `PATH` |
| `BASE_PLAN` / `BASE_PLAN_ID` | Verified comparator plan and its emitted semantic identity |
| `ASM_CONFIG` / `ASM_CONFIG_SHA` | Complete qualified ASM configuration and its file SHA256 |
| `FOCUS_RESOURCES` / `FOCUS_RESOURCES_SHA` | Explicit resource-window declaration and its file SHA256 |
| `PANEL` / `PREPARED` | Unchanged frozen panel manifest and verified prepared development data |
| `FOCUS_DIR` | New durable private output directory given as an absolute path, outside Git working trees and `/tmp`; `optimize run` refuses a relative journal path |
| `DATASET` | One regression dataset already present in the frozen panel, such as `230_machine_cpu` |

The base plan and complete configuration are prepared using the operations and architecture runbooks. They include real runtime/worker inventories, checkpoint/vocabulary/transform assets and context qualification. Hand-writing a short configuration with invented hashes cannot replace those artifacts. The resource declaration includes workers, CPUs, memory, zero swap, GPU task allowance, threads per worker, total/reporting seconds and the per-fit cap.

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

The command audit for this chapter executed only metadata/help operations. These example execution commands were not run as a new campaign while writing the manual. The separately measured68 proof described in chapter 1 remains the actual numerical example.

### Other current search surfaces

`jope search --input "$FIXTURE" --json` scores a tiny ridge-family candidate bank. It is useful for its documented fixture scope; it is not sustained neural GP exploration.

`jope pipeline run` accepts a catalog dataset or dataset directory, cuts, seeds and a budget, and can emit several languages. Its `--policy auto` and `skip` report a skip, while `require` fails closed. Its legacy budget description explicitly allows an in-flight fit to overrun. Therefore neither increasing its attempt count nor choosing `--policy require` supplies the current bounded neural learning path. See the language chapter for the actual legacy pipeline example and its receipt.

## What is missing for a sustained, incumbent-seeded dataset attack

There is no current `--incumbent` input on `optimize proof-plan`, and no shipped `jope optimize attack` command. The ongoing wave-by-wave attack interface is a proposed follow-on. Documenting a plausible command string would not make it available.

An implementation should reuse the current worker pool, selection plan and journals, and integrate the pending eligibility overlay (TEACH-01/TEACH-02) only after it is qualified, with these specific additions:

| Addition | Required behavior |
| --- | --- |
| Frozen focus request | Authenticate one dataset, task, exact fold/group/row identity, evaluator, vocabulary, numerical policy and declared total window before searching |
| Eligible incumbent reader | Read the incumbent under the complete comparison identity; refuse a stale, incompatible or unauthenticated record |
| Incumbent-aware initialization | Include the incumbent and bounded mutations/macros with explicit origins; preserve an exploration fraction and canonical duplicate accounting |
| Wave accounting | Reuse the original global deadline and journal; each bounded wave consumes remaining time rather than resetting it |
| Re-evaluation and admission | Use complete finite train-side OOF evidence; meet the required trivial-baseline/4-of-5-fold gate, then compare against the incumbent under the same identity |
| Atomic improvement | Strict improvement moves the pointer only after durable evidence; ties keep the incumbent; concurrent pointer changes trigger re-decision |
| Useful teaching delta | Export newly eligible programs and justified comparison records with original context/labels/exposure, rather than copying a winner's reported score |
| Honest completion | Report best retained incumbent, all waves and costs, improvement/stagnation, failures and unmet backend support even when no new winner appears |

The original acceptance requires four-of-five-fold improvement over a trivial baseline. A proposed extra four-of-five-fold comparison against the incumbent is a separate robustness policy, not a shipped requirement or retrospective reason to reject old evidence. Freeze that policy before running a focused extension. The current HWM strict-improvement rule also does not by itself prove the new teaching eligibility contract.

For a dataset outside the frozen panel, first implement or use a separate authenticated ingestion/preparation request with the same label, category, missingness and group guarantees. Do not edit the thirty-dataset manifest to sneak it into the official panel. Repeated comparisons use development labels; sealed evaluation remains outside search, teaching and motif mining.

## Code to extend

These are the narrow extension points at reviewed source68. Follow the exact-revision links; line numbers refer to that snapshot.

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

## Specialization without starving the general learner

One difficult dataset can produce many near-duplicate teachers. Count unique canonical programs, meaningful improvements and distinct contexts, not only serialized rows. Retain broader replay and family balance so a successful focused attack does not make the policy forget other problems. The binary-packing chapter describes how to share repeated contexts and immutable shards without losing original provenance.

Record the fresh-program exposure actually sampled. A nominal 50% fresh-comparison schedule need not yield 50% fresh program slots when each comparison also contains an incumbent and old comparators. Prefer measured exposure, a frozen baseline checkpoint, component-loss diagnostics and actual downstream fitting over assumptions based on the sampler's name.

If the target ratchet moves but fixed-monitor policy quality deteriorates, investigate the objective, label consistency, sampler and comparator support before simply training longer. If policy metrics improve but fitted searches do not, inspect proposal diversity, candidate runtimes, grammar coverage and GP recovery. If neither improves, report the plateau and preserve the best verified incumbent. The system earns a flywheel claim through this evidence, not through an endlessly running process.
