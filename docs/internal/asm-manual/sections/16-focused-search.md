# Focused single-dataset search

## Commands that exist today for one dataset

### Front door and safe metadata checks

The current private Linux `asm` build supports a **separately bounded operational proof on one frozen regression dataset**. This is the closest current neural search front door to focused optimization. It is not an arbitrary-dataset ingestion command, a daemon, or an incumbent-seeded high-water-mark attack.

The following metadata commands are safe preparation and were checked against the immutable reviewed 68 CLI:

```sh
"$JOPE" optimize capabilities
"$JOPE" optimize proof-plan --help
"$JOPE" optimize run --help
"$JOPE" optimize report --help
```

`compiled:true` means support was compiled, not that assets, context qualification or training have passed. Use a binary whose build inventory, source manifest and executable hashes match the configuration. The implementation at `a75e543d2` caps an arm at 120 seconds and its candidate allowance at 80–1000 evaluations (`crates/jope-bench/src/panel/optimize.rs:132-133`). Per-fit time remains at most 30 seconds (`crates/jope-bench/src/panel/origin_proof.rs:86`). Editing a JSON number beyond those caps does not extend the implementation.

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

The base plan and complete configuration are prepared as described in [§](14-install-and-prepare.md#native-roster-and-external-comparator-runtimes) and [§](06-encodings-and-context.md#qualification-and-admission). They include real runtime/worker inventories, checkpoint/vocabulary/transform assets and context qualification. Hand-writing a short configuration with invented hashes cannot replace those artifacts. The resource declaration includes workers, CPUs, memory, zero swap, GPU task allowance, threads per worker, total/reporting seconds and the per-fit cap.

### Plan, run and report a focused proof

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

The command audit for this chapter executed only metadata/help operations. These example execution commands were not run as a new campaign while writing the manual. The separately measured origin proof at source `68f56404` ([§](18-measured-performance.md#measured-current-source-operational-proof)) remains the actual numerical example.

### Other current search surfaces

`jope search --input "$FIXTURE" --json` scores a tiny ridge-family candidate bank. It is useful for its documented fixture scope; it is not sustained neural GP exploration.

`jope pipeline run` accepts a catalog dataset or dataset directory, cuts, seeds and a budget, and can emit several languages. Its `--policy auto` and `skip` report a skip, while `require` fails closed. Its legacy budget description explicitly allows an in-flight fit to overrun. Therefore neither increasing its attempt count nor choosing `--policy require` supplies the current bounded neural learning path. See [§](09-programs-and-languages.md#example-b-actual-legacy-pipelinesearch-shape) for the actual legacy pipeline example and its receipt.

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

For a dataset outside the frozen panel, first implement or use a separate authenticated ingestion/preparation request with the same label, category, missingness and group guarantees. Do not edit the frozen panel's manifest (asm-private-30-v1) to add it. Repeated comparisons use development labels; sealed evaluation remains outside search, teaching and motif mining.

## Code to extend

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

## Specialization without starving the general learner

One difficult dataset can produce many near-duplicate teachers. Count unique canonical programs, meaningful improvements and distinct contexts, not only serialized rows. Retain broader replay and family balance so a successful focused attack does not make the policy forget other problems. [§](11-corpora-and-packing.md#packing-semantics-legal-sets-and-label-policies) describes how to share repeated contexts and immutable shards without losing original provenance.

Record the fresh-program exposure actually sampled. A nominal 50% fresh-comparison schedule need not yield 50% fresh program slots when each comparison also contains an incumbent and old comparators. Prefer measured exposure, a frozen baseline checkpoint, component-loss diagnostics and actual downstream fitting over assumptions based on the sampler's name.

If the target ratchet moves but fixed-monitor policy quality deteriorates, investigate the objective, label consistency, sampler and comparator support before simply training longer. If policy metrics improve but fitted searches do not, inspect proposal diversity, candidate runtimes, grammar coverage and GP recovery. If neither improves, report the plateau and preserve the best verified incumbent. The system earns a flywheel claim through this evidence, not through an endlessly running process.
