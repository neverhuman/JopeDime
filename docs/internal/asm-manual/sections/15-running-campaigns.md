# Operator runbook: run, resume, demonstrate and troubleshoot

## Plan, execute and report

Prepare the qualified configuration before opening an execution window. It binds the actual context producer, source/worker/model inputs, numerical qualification and policy configuration. The standard comparison preregisters a 64-slot policy pool plus 16 exploration candidates against 80 generic candidates, with equal declared arm allowances. Policy-pool shortfalls carry `neural_rejection_fallback`; only `neural_policy` slots count as genuine checkpoint proposals. Generation-zero neural slots must actually fit; a proposal file or generic-only result cannot demonstrate neural consumption.

That preregistration applies in full only to regression at the pinned source `68f56404`. The accepted policy is regression-only, so binary and multiclass datasets have no neural arm, and their generic arm searches ridge heads only (`logit_ridge` or `multiclass_ridge` with at most four transforms; see [§](06-encodings-and-context.md#experimental-classification-context)):

| Task | Frozen datasets | Neural arm | Generic arm | Non-ASM comparators |
|---|---|---|---|---|
| Regression | 10 | 64 policy slots plus 16 exploration candidates | 80 grammar-uniform candidates | 17 native recipes plus 4 external engines |
| Binary | 10 | Unavailable | 80 `logit_ridge` candidates | 5 native recipes plus 4 external engines |
| Multiclass | 10 | Unavailable | 80 `multiclass_ridge` candidates | 4 native recipes plus 4 external engines |

At this source, a better policy therefore cannot change binary or multiclass results. Sources: `crates/jope-bench/src/panel/optimize.rs:34`, `:35`, `:493`; `crates/jope-pipeline/src/neural_search.rs:28`.

```sh
"$J" optimize plan \
  --base-plan "$A/base.json" --base-plan-sha256 "$BASE_PLAN_ID" \
  --configuration "$ASM_CONFIGURATION" \
  --configuration-sha256 "$ASM_CONFIGURATION_FILE_SHA256" \
  --output "$A/panel.json"

"$J" benchmark run --plan "$A/panel.json" --plan-sha256 "$PANEL_PLAN_ID" \
  --manifest "$M" --datasets "$D" --journal "$A/panel-journal"

"$J" benchmark report --plan "$A/panel.json" --plan-sha256 "$PANEL_PLAN_ID" \
  --manifest "$M" --datasets "$D" --journal "$A/panel-journal"
```

`optimize run` and `optimize report` delegate to the same campaign contracts. Repeating run with the same plan and journal resumes the original authenticated state and absolute deadline. A restart does not grant another two hours. Do not repair journals by editing receipt fields or copying successes from another experiment. Read-only report recovery authenticates retained artifacts and recomputes scores; it does not fit another predictor.

## Resume and recovery

### Reading stop reasons and resuming

Final search reasons include `evaluation_limit`, `original_deadline`, `search_space_exhausted` and `no_measured_parent`. `admission_pending_original_window` and `cancelled_original_window` are non-final: no winner may be frozen while either holds. The proof front door can return `ok:false, state:pending_original_window`. Resume the same plan and journal, preserving its original arm and campaign deadlines; resume never creates another allowance. Before cutoff, unresolved never-admitted inventory may be retried; at/after cutoff it can only be finalized under the existing evidence rules.

A setup failure before gate release becomes terminal `budget_exhausted` if its original job cutoff has been reached; setup consumption stays unknown. Before cutoff, ordinary pre-gate failure is `unavailable` and cancellation is `cancelled`, subject to the retryable-inventory rules. Interrupted started-only attempts are excluded from that category, as described in [§](#interrupted-attempts-and-proof-interpretation). The earlier proof at source `3fd7d5b2` retained exit 1 after 245.93 seconds because an expired pre-gate attempt incorrectly remained retryable; its unchanged journal is evidence of the failure, not permission to reset the clock. Sources: `crates/jope-pipeline/src/neural_search.rs:1308`, `:1380`, `:1457`, `:845`; `crates/jope-pipeline/src/execution.rs:1209`; `crates/jope-bench/src/panel/origin_proof.rs:194`.

### Interrupted attempts and proof interpretation

An attempt with a `started/` record and no `finished/` record is never rerun, and the pool does not recover its original result or final counters. After its process group is safely drained, the pool persists terminal `cancelled` with reason `controller_interrupted: original attempt not retried`; CPU, wall time and exit accounting remain unknown. If recovery cannot drain the group, no recovered terminal record is persisted and further admission is blocked (`recovery_not_drained`). Such attempts are excluded from retryable never-admitted inventory. Separately, a normally watched attempt persists its observed terminal outcome before potentially slow cleanup; that rule does not reconstruct interrupted accounting. The search uses immutable observed journals and refuses changed completed evidence. At cutoff, only proven never-admitted inventory may normalize; the per-fold replay digest preserves ordered real attempts and output identities. Both live PoolBackend feedback and recovered FitAdapter scoring compute the same shared digest.

A regression caught during acceptance illustrates the distinction: the proof at source `d0f5748b` finished 964 worker commands (834 completed, 130 failed), but its recovered report failed because the two feedback constructors disagreed about replay evidence. Those fit counts did not justify declaring the report passed. The `68f56404` correction restored one shared digest computation, followed by a fresh bounded proof and recovered report. The `d0f5748b` records were retained unchanged.

Never extend an expired original deadline, mutate a failed result into a success, reinterpret an unknown cost as zero, or cite syntax parsing as model execution. A report can honestly say “source implementation prepared; CPU fixture passed; live training pending.” That is more useful than collapsing all three into one completion flag.

### Recovery phases

Identify whether failure occurred before admission, in setup, after the gate, during cleanup, in controller scoring or during report generation.

- A pre-admission dependency refusal may be retried only inside the original window.
- Setup can consume CPU without starting a fit. Unknown setup cost stays unknown.
- A started identity is durable. Controller death does not authorize silently repeating it.
- Completed folds inside a failed candidate still require artifact authentication.
- Original-cutoff inventory settles without extending the clock.
- A report can fail after successful fits. Retain both; do not call the whole operation successful.
- Saving a terminal outcome does not release cleanup authority before descendants drain.

Resume with the same plan and journal. A changed source, dataset, budget or protocol requires a separately identified experiment with the old evidence retained, not a relabelled continuation.

## Warm fitted demonstration after integration

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

## Artifact inventory, cost and privacy

### What evidence to retain

| Evidence | What to retain and inspect |
|---|---|
| Source/build/install | Full commit, source manifest, Cargo messages, exact features/toolchain, dependency graph, binary and worker hashes, physical paths, cold timings and exits. |
| Plan and configuration | Frozen panel identity, all runtime/model/encoder pins, resource limits, semantic plan ID, seeds, numerical policy and original window. |
| Worker execution | Started/finished union, exact job/input identity, output inventory, exits/signals, timeouts, process-group drainage, memory/CPU counters where known. |
| Scientific results | All screen/confirmation losses, frozen recipe set, original row/class order, baseline, selection exposure and unsupported/unavailable outcomes. |
| Fitted predictor | Complete private state, fitted preprocessing, recipe, native reload proof, prediction hashes and exact artifact identity. |
| Export qualification | Exact winner/fold/state, public bundle, toolchain/input/class/policy hashes, actual commands and parity result. See [§](09-programs-and-languages.md#example-c-exact-fitted-winner-export-after-pending-integration). |

Retain source revision and input manifest; exact command, directory, exit and timestamps; executable/runtime/features/environment; data/split/evaluator/vocabulary/protocol identities; original limits/start/deadline/cutoff; collected/executed/failed/skipped/excluded/unavailable inventories; artifact hashes and worker receipts; report replay; limitations and next owner.

Verify a source manifest against the committed blobs it claims. The current bytes at `target/release/jope` need not be the bytes used by an earlier result. Preserve immutable executable copies with their build receipts before another build replaces the usual path.

Current ASM custody is outside Git at `/home/ubuntu/jope-rescue-20260921/asm-integration-20260923/`. This host location is not a portable runtime default.

### Counting, export accounting and public-bundle privacy

Count a unique fitted predictor by authenticated complete fitted-artifact identity, and a reusable recipe by the declared recipe identity. This does not establish behavioral uniqueness. Fit failures, policy/context work and preparation still incur cost. A worker CPU subtotal is not automatically total creation CPU. Validated systemd unit CPU can include setup, controller and children; it overlaps worker CPU and must not be added to it. Unknown counters stay null with a reason.

In the pending ASM export accounting, `fitted_export_commands` is a subset of global worker commands. Comparator-only accounting requires its separate export cost to be added. Export preparation is measured separately; final receipt publication and other unmeasured overhead remain explicit. Retain the original panel cutoff: backend qualification after a campaign expires is unattempted through the existing front door. A distinct post-panel bounded validation stage is not implemented and cannot be claimed inside the two-hour panel allowance.

Allowlisted public fitted bundles contain inference code, necessary fitted input dictionaries, class/schema metadata and their hashes. They exclude the private recipe, discovery history and scoring references. The private prepared plan and worker directory remain private. Canary/text-policy checks supplement the explicit file allowlist; passing static checks does not replace actual complete-prediction parity.

## Troubleshooting and safe recovery

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
