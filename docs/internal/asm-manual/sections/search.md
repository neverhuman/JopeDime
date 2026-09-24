# Regression search, fitting, prediction and focused optimization

## Proposal generation and ranking

Accepted proposals are generated under grammar constraints, and the neural pool is the first 64 distinct valid programs in draw order; any shortfall after the bounded sampling rounds is filled with distinct grammar-uniform programs labelled `neural_rejection_fallback`. Ranking orders that pool by the mean of four reward heads, descending; exact ties use program SHA256 then text. Runtime estimates are reported without subtracting an undocumented runtime penalty. Ranking binds the checkpoint, context and vocabulary and records that scoring labels and fitted GP scores were not read. It does not select or reject programs: all 64 ranked programs and the 16 exploration programs enter generation zero. Sources: `crates/jope-policy/src/ranking.rs:22`; `crates/jope-policy/src/decode.rs:9-11`, `:212-225`, `:359`; `crates/jope-policy/src/seeds.rs:257-270`.

Each ranked policy-pool program carries `rank_log_weight`, the log-softmax of its four-head mean predicted reward over that pool. Historical files called this `policy_logprob`. It is a within-context ranking weight, not a sampling/selection probability or propensity; do not use it for importance weighting or off-policy correction. Exploration candidates carry no rank weight.

The four reward heads read a masked mean of the program's token embeddings, with no position information (`crates/jope-policy/src/model.rs:555-567`). Within one context, two programs with the same token multiset therefore receive the same rewards in exact arithmetic, so their relative order carries no learned signal; an exact tie falls to the program-hash tie-break. This does not remove token order from the autoregressive proposal decoder ([§](learning.md#distinguish-routing-ranking-and-search-recovery)). The owner's rule for this limitation (status: proposed): first measure how often such collisions occur and the ranking regret they cause, and defer any change to the reward architecture, so that the first causal training experiment keeps architecture, objective, corpus and sampler changes distinguishable.

## Paired arms, generation zero and evolution

ASM compares a 64-slot policy pool plus 16 exploration generation-zero slots with 80 generic slots. The pool retains `neural_policy` versus `neural_rejection_fallback` origins explicitly. Both arms receive the same declared search and fit allowances, but encoding and proposal work consume the neural arm's own clock; fitting time can differ. A producer refusal, duplicate or failed candidate is an outcome to record. It cannot silently become another neural success or a free replacement arm. Actual inner-fold fits determine selection and offspring ancestry. A model's predicted reward never substitutes for measured loss.

Each arm starts with 80 generation-zero slots. In the neural arm, slots 0–63 form the ranked policy pool: distinct valid checkpoint samples are labelled `neural_policy`; any shortfall after the bounded sampling rounds is filled by distinct grammar-uniform programs labelled `neural_rejection_fallback`. Slots 64–79 are `random_exploration`. Only `neural_policy` roots and completed descendants with verified neural ancestry count as genuine neural evidence; fallback and exploration roots do not. In the origin proof at source `68f56404` all 64 pool slots were `neural_policy`, 48 completed all five inner folds, and no fallback slots were needed. Receipts retain initial-slot origin and descendant ancestry. Actual completed inner OOF losses select the best four parents. New generations alternate mutation and crossover attempts, canonicalize recipes, reject duplicate executable identities, and retain variation attempts in the journal. Source: `crates/jope-pipeline/src/neural_search.rs:27`, `:1380`.

Both initial populations are authenticated, and neither arm is seeded from the other. At `a75e543d2` the neural arm's 80 generation-zero slots come only from the verified policy-producer artifact: the controller verifies the artifact against the worker request, requires exactly 80 slots and refuses any slot whose declared origin does not match its position (`crates/jope-pipeline/src/neural_search.rs:1066-1102`). The generic arm draws its 80 distinct candidates from its own grammar stream (`neural_search.rs:1147-1176`). Generation zero reads no candidate discovered by the other arm. Historical final searches differed: their arms shared incumbent, retained-teacher and exploration seed pools, and shared incumbents can create many ties ([§](learning.md#distinguish-routing-ranking-and-search-recovery)). The owner's rule for any future seeding (status: proposed): an initial population, including an incumbent, retained teachers or another arm's discovery, enters an arm only as an authenticated origin declared in the plan before execution; a discovery never silently seeds another arm.

## Dense-width refusals and the proposed feasibility preflight

At `a75e543d2` the 512-column cap applies to generated dense width, not to total schema or raw input width (`MAX_GENERATED_DENSE_WIDTH`, `crates/feat-math/src/advanced_v3/feature_block.rs:6`, `:107-110`). No check runs before launch: the recipe validation that precedes fitting checks task, schema, syntax and stage admissibility but does not predict generated widths, so an over-wide recipe fails inside a launched fit. In the origin proof those refusals were 130 failed commands from 26 recipes, including 16 of 64 `neural_policy` generation-zero candidates ([§](status.md#failures-dense-width-refusals-and-the-earlier-failed-reproducer)).

A pre-launch dense-width feasibility check is **proposed**; nothing below is implemented. The owner's requirements (relayed 2026-09-23T22:27:11Z) and Codex's read-only source investigation (`plan-execution-20260923/feasibility-phase-accounting.md`, SHA-256 `0154e6b0…`) bound it:

- **Permitted fitting information only.** The check may read each inner fold's task, exact schema, class order and training-side fitting rows, never prediction rows or validation labels. One-hot and hybrid categorical widths depend on the fitting-side category layout, which the compiler deliberately refuses to guess; an undecidable case stays unknown and goes to the actual worker rather than being rejected.
- **Unchanged limits and clocks.** A rejected candidate keeps its slot, origin and ancestry, counts against the arm's evaluation allowance and earns no replacement draw. The check runs inside the existing arm allowance.
- **Separate reporting.** A preflight rejection is reported separately from a launched fit failure, with its own receipt and zero physical fit starts.
- **No unlimited free retries.** Resume neither retries a proven rejection nor changes the deadline.

The accounting hazard is specific. `PoolBackend::evaluate` records every job-preparation error as `Unavailable`, with zero physical fit starts and `retryable_inventory: true` (`crates/jope-pipeline/src/neural_search.rs:1608-1624`). A naive preflight that returned an error from job preparation would therefore mark a deterministically infeasible recipe as retryable and could repeat preflight work on resume. The investigation instead proposes a typed terminal outcome with accounting: all five folds are checked before any fold of the candidate launches, and the candidate keeps one observation with zero physical fit starts, no worker receipt, no out-of-fold loss and `retryable_inventory: false`. Recovery then verifies the retained preflight receipt without fitting. The closest existing precedent is the zero-fit terminal record for canonical duplicates (`evaluate_unique`, `neural_search.rs:884-925`).

## Warm fitted demonstration after integration

Runbook summary (status as of 2026-09-23 about 23:55 UTC: **pending integration**):

| Runbook part | Warm fitted demonstration |
| --- | --- |
| Purpose and status | Fit, save, reload and predict in one timed invocation: a generic mode with one registered recipe, and an authenticated ASM mode with paired arms on one frozen regression dataset. **Pending integration:** `demo` belongs to the stopped product overlay and is absent from the immutable `68f56404` CLI ([§](installation.md#command-surfaces-and-the-stopped-integration-order)) |
| Prerequisites and resources | Generic mode: the allowance stated in [§](#generic-and-asm-demonstration-commands). ASM mode: the reviewed CLI built with `asm`, the worker build inventory, qualified context and policy workers, the runtime inventory for the installed executable, and a resource declaration; the product guide's example declares 20 workers and CPUs, 64 GiB, zero swap, no GPU tasks, 60 seconds in total with 5 for reporting, and 30 seconds per fit |
| Manifest-creation command | Generic mode writes its own private plan inside its output directory. ASM mode uses `benchmark plan` and then `optimize proof-plan`, both present at `a75e543d2`. No `jope` command writes the warm resource declaration or the warm-only ASM configuration |
| Execution | The `demo` commands in [§](#generic-and-asm-demonstration-commands) |
| Artifacts and success | Generic: the private plan, worker execution journal, `fitted.json`, `predictions.json`, `baseline.json`, `demo-result.json` and `invocation-receipt.json`. ASM: `asm-fitted.json`, `generic-fitted.json`, their predictions, `baseline.json`, the complete `origin-proof.json`, `asm-demo-result.json` and `invocation-receipt.json`. Success requires saved complete fitted state, unchanged predictions after reload and finite comparisons; ASM mode also requires two full equal arm allowances, genuine neural-origin fits and evolution, and both selected refits. `under_60_seconds` (and, with the warm-target add-on, `target_met`) is true only for a successful full invocation measured below 60 seconds |
| Failure and resume | Failed attempts keep their failure and execution evidence. The pending guide describes no resume of an interrupted demonstration: a new attempt uses a new output directory outside Git checkouts and `/tmp`, and cannot reuse an expired origin-proof journal. Missing or clipped arm time sets `matched_comparison: false` and fails the ASM demonstration |
| Limitations | One dataset: not a thirty-dataset result, a held-out quality claim or training evidence. Synthetic mode is labelled. Optional exports stay `emitted-unqualified` or explicitly unsupported. No speed claim exists until a run is measured |

Source: pending `docs/private-install-demo.md:90-203` and `crates/jope-bench/src/panel/demo_asm.rs:373-459` in the product overlay replay (`agent-product-final-68f56404/replay-001/source-text` under the ASM custody root).

### Generic and ASM demonstration commands

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

### The under-sixty-second target and historical timing

The warm-target add-on treats under 60 seconds as a declared, measured goal, not a promise and not a cap that prevents a required real fitted demonstration. Both complete equal arm allowances must fit within the declared original execution window. Unequal, missing or globally clipped arms fail the matched-comparison condition. An explicitly admitted longer allowance can produce a successful complete execution with `target_met: false`; it cannot claim under-60 acceptance. It does not change the official panel's 120-second arm defaults.

The retained proof at source `3fd7d5b2` started the policy about 22.1 seconds into its original window, spent about 20.8 seconds in policy production and started the first fit around 50.4 seconds. Those are historical measurements, not a forecast or a proof that equal 20-second warm arms work. A suggested longer configuration is a sizing proposal until explicitly admitted. Neither a generic substitute nor clipping the second arm solves this requirement.

## Focused single-dataset search

Status as of 2026-09-23 about 23:55 UTC: the bounded one-dataset proof front door below is **available** at `a75e543d2` for a regression dataset already in the frozen panel; the sustained, incumbent-aware attack is **proposed** ([§](#what-is-missing-for-a-sustained-incumbent-seeded-dataset-attack)).

### Commands that exist today for one dataset

Runbook summary:

| Runbook part | One-dataset focused proof |
| --- | --- |
| Purpose and status | Run one bounded paired neural and generic search on one frozen-panel regression dataset and recompute an authenticated development report. **Available** at `a75e543d2`; measured once, at `68f56404` on `230_machine_cpu` ([§](status.md#canonical-proof-record-and-per-arm-facts)) |
| Prerequisites and resources | The inputs in [§](#required-inputs); no exclusive full-CPU CI running; a configuration and resource declaration within the implementation caps (an arm of at most 120 seconds, 80–1000 evaluations, at most 30 seconds per fit). The measured proof declared 20 workers and CPUs, 64 GiB, zero swap, no GPU tasks and a 600-second window including 30 seconds for reporting |
| Manifest-creation command | `benchmark plan` writes the base (comparator) plan ([§](installation.md#native-roster-and-external-comparator-runtimes)); `optimize proof-plan` writes the focused plan and prints its semantic `plan_sha256`. No `jope optimize` subcommand writes the complete `jope.asm-configuration/v1` configuration or the resource declaration: the origin proof's configuration came from the custody helper `agent-neural/prepare-origin-config.py` (receipt `configuration-preparation.json` beside that proof), which is not a product command. The frozen panel manifest itself is fixed, not generated |
| Execution | `optimize run` ([§](#plan-run-and-report-a-focused-proof)); it refuses a running binary that differs from the pinned native runtime |
| Artifacts and success | `plan.json` and `plan-receipt.json`; the journal (window, campaign and per-arm records, started and finished worker receipts); the report returned by `run` and recomputed by `optimize report` without fitting. Success is an authenticated report of both arms inside the original window, not a neural win |
| Failure and resume | The same `run` resumes the journal within its original window; an expired window cannot be renewed. Failed and refused fits, such as dense-width refusals ([§](#dense-width-refusals-and-the-proposed-feasibility-preflight)), remain recorded outcomes |
| Limitations | One regression dataset already in the frozen panel; no `--incumbent` input; no arbitrary-dataset ingestion; development-hold scores only; the dense-width cap is checked only inside launched fits |

Sources: `crates/jope-bench/src/panel/optimize_cli.rs:48-89` and `crates/jope-bench/src/panel/campaign_cli.rs:130-208` at `a75e543d2`; custody `neural-integration/origin-proof-68f56404…/configuration-preparation.json`.

#### Front door and safe metadata checks

The current private Linux `asm` build supports a **separately bounded operational proof on one frozen regression dataset**. This is the closest current neural search front door to focused optimization. It is not an arbitrary-dataset ingestion command, a daemon, or an incumbent-seeded high-water-mark attack.

The following metadata commands are safe preparation and were checked against the immutable reviewed 68 CLI:

```sh
"$JOPE" optimize capabilities
"$JOPE" optimize proof-plan --help
"$JOPE" optimize run --help
"$JOPE" optimize report --help
```

`compiled:true` means support was compiled, not that assets, context qualification or training have passed. Use a binary whose build inventory, source manifest and executable hashes match the configuration. The implementation at `a75e543d2` caps an arm at 120 seconds and its candidate allowance at 80–1000 evaluations (`crates/jope-bench/src/panel/optimize.rs:132-133`). Per-fit time remains at most 30 seconds (`crates/jope-bench/src/panel/origin_proof.rs:86`). Editing a JSON number beyond those caps does not extend the implementation.

#### Required inputs

| Variable | Meaning |
| --- | --- |
| `JOPE` | Authenticated Linux CLI built with `asm`; not an unrelated binary found on `PATH` |
| `BASE_PLAN` / `BASE_PLAN_ID` | Verified comparator plan and its emitted semantic identity |
| `ASM_CONFIG` / `ASM_CONFIG_SHA` | Complete qualified ASM configuration and its file SHA256 |
| `FOCUS_RESOURCES` / `FOCUS_RESOURCES_SHA` | Explicit resource-window declaration and its file SHA256 |
| `PANEL` / `PREPARED` | Unchanged frozen panel manifest and verified prepared development data |
| `FOCUS_DIR` | New durable private output directory given as an absolute path, outside Git working trees and `/tmp`; `optimize run` refuses a relative journal path |
| `DATASET` | One regression dataset already present in the frozen panel, such as `230_machine_cpu` |

The base plan and complete configuration are prepared as described in [§](installation.md#native-roster-and-external-comparator-runtimes) and [§](architecture.md#qualification-and-admission). They include real runtime/worker inventories, checkpoint/vocabulary/transform assets and context qualification. Hand-writing a short configuration with invented hashes cannot replace those artifacts. The resource declaration includes workers, CPUs, memory, zero swap, GPU task allowance, threads per worker, total/reporting seconds and the per-fit cap.

#### Plan, run and report a focused proof

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

The command audit for this chapter executed only metadata/help operations. These example execution commands were not run as a new campaign while writing the manual. The separately measured origin proof at source `68f56404` ([§](status.md#canonical-proof-record-and-per-arm-facts)) remains the actual numerical example.

#### Other current search surfaces

`jope search --input "$FIXTURE" --json` scores a tiny ridge-family candidate bank. It is useful for its documented fixture scope; it is not sustained neural GP exploration.

`jope pipeline run` accepts a catalog dataset or dataset directory, cuts, seeds and a budget, and can emit several languages. Its `--policy auto` and `skip` report a skip, while `require` fails closed. Its legacy budget description explicitly allows an in-flight fit to overrun. Therefore neither increasing its attempt count nor choosing `--policy require` supplies the current bounded neural learning path. See [§](inference.md#example-b-actual-legacy-pipelinesearch-shape) for the actual legacy pipeline example and its receipt.

### What is missing for a sustained, incumbent-seeded dataset attack

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
