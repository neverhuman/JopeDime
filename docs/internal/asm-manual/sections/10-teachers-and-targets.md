# From measured programs to teachers

## A dataset is not one training example

### Accounting units

A training context describes a particular dataset, split and encoding. Many executable programs can share that context. Each program supplies a separate target sequence and measured supervision. The useful accounting units are therefore:

| Unit | Meaning | Why the distinction matters |
| --- | --- | --- |
| Dataset/family | A problem and its related datasets | Family partitions control generalization; row counts alone do not identify independence. |
| Cut/context | A dataset split plus prepared-data and encoder identities | The same dataset can have many contexts. Reusing a context must preserve its actual split and fitting rows. |
| Program observation | A measured evaluation of a concrete executable recipe | Failed, duplicated, incompatible or exposed observations need not become training examples. |
| Admitted target record | A representable program and its supervision for one context. Under the pending contract, a **teacher** is such a record that passed the `jope.teaching-guard/v1` admission ([§](#the-admission-guard-label-exposure-and-hwm-identity)) and entered a frozen corpus | Repeated observations of the same program/cut do not automatically earn repeated votes. |
| Corpus snapshot | An immutable collection of target records and identities | Its record count is not the number of optimizer updates or unique datasets. |
| Presentation | A sampled training example used in an update | Sampling revisits records. Millions of presentations do not mean millions of newly discovered targets. |

### Context vectors and program targets

The legacy Rust context is 2,048 little-endian float32 values, or 8,192 bytes. The model receives a 1,536-value dataset lane and a separate 512-value target-statistics lane. The historical encoder recipe described the dataset lane as Hyperion 512 plus Foundation/PCA 1,024. Despite its name, the dataset lane is not label-free: in the current context producer the 1,024 Foundation values are computed from the features together with the standardized training targets, and only the 512 Hyperion values use the features alone ([§](06-encodings-and-context.md#component-dimensions-and-assembly)). These vectors are shared by context identity; they are not repeated inside every 532-byte target record. The program target is a grammar-token sequence, with the legal-token set for each step, context ID, score-derived value, percentile and log-runtime supervision.

A concrete metadata binding in the accepted step-60000 corpus connects dataset `031201c6895f724b`, family `rdatasets-agridat-kalamkar-potato-uniformity`, cut 0 and `meta_train` to a prepared view and context vector. Its context SHA begins `6f5fdd79…`, prepared-view SHA `811feabd…`, and content-addressed shard contains 24,231 records. The full paths and hashes are in `accepted-corpus-binding.json`. This audit verified that metadata relation without opening the encoded vector or target records.

### The learning chain

The learning chain is:

1. Pin the raw dataset, family partition and actual fitting/development row assignments.
2. Fit preprocessing and encoding only within the permitted input closure; pin their bytes and versions.
3. Generate a concrete executable program, fit it and retain its outcome, measurement, resource use and evaluator identity.
4. Apply the declared admission policy; reject failed, incompatible, exposed or unsupported targets.
5. Tokenize the exact program and retain original observations beside the training representation.
6. Sample the admitted record, update parameters, and prove checkpoint reload/continuation.
7. Evaluate the new policy through actual proposals and fits under the comparison protocol.

A file at step 5 does not prove steps 6–7. Conversely, a lower optimizer loss does not establish better executable proposals.

## How a fitted recipe becomes an eligible teacher

### The preregistered freeze and train-side inner split

The pending eligibility path freezes independent facts before optimization: original dataset and prepared-data identity, ordered rows and groups, target role, exact inner-fold assignment, evaluator and vocabulary identities, numerical policy, baseline and worker/resource pins. This freeze cannot be created retrospectively from a winning candidate's own output. A historical successful fit is not equivalent to preregistered teaching evidence.

The train-side inner split uses five whole-group folds. Groups are assigned deterministically by largest size first, with lexical tie-breaking and balancing. No group's rows are split simply to fill a fold. The cut's held-out development fold remains outside the search's inner fitting/selection closure.

### The admission guard, label exposure and HWM identity

Each candidate must produce complete authenticated OOF predictions for all five inner folds. Scores are recomputed from retained predictions, exact targets and ordered class probabilities. The trivial baseline is fitted from each fold's training rows: target mean for regression, class frequencies for classification. The guard uses mean fold loss and admits only if:

- all five fold losses and baseline losses are finite;
- the mean candidate loss is below the mean baseline loss by more than `1e-12 + 1e-6 * abs(mean_baseline_loss)`;
- at least four of five folds each improve by more than `1e-12 + 1e-6 * abs(that fold's baseline_loss)`;
- actual worker requests, process outcomes, output hashes, fitted artifacts, seeds, source, resources and deadlines match the original freeze.

The criteria are intentionally stronger than “a run completed” or “the score field is positive.” Constant/no-signal predictors, missing folds, reordered rows, changed class order, stale scores, unauthenticated prediction blobs and unknown execution accounting cannot provide teaching authority. A failed admission is retained as evidence rather than discarded.

Optimization-visible labels initially called `test` become development labels, with a durable record of that exposure. `sealed` and `meta_test` inputs are refused for optimization/teaching. The HWM identity includes the full comparison context, so an improvement under a different evaluator, split, vocabulary, preprocessing or numerical protocol cannot displace an unrelated incumbent. These details belong to the pending eligibility/HWM correction, not a claim that every historical HWM record already meets this rule.

Sources: pending `crates/jope-learning/src/eligibility.rs:170`, `:253`, `:277`, `:427`, `:700`, `:1023`; `crates/jope-learning/src/eligibility/inner.rs`; `crates/jope-learning/src/record.rs`; `crates/jope-pipeline/src/search_plan.rs`.

## From optimization to reusable knowledge

The pending strict HWM key includes task, dataset hash, cut identity, evaluator, vocabulary and protocol. Scores from different identities cannot overwrite each other or supply training feedback. A recipe must show finite train-side out-of-fold improvement over a trivial baseline and improve on four of five folds before teaching admission. No-signal data should be refused by this gate. An optimization-visible test label is development evidence even if its filename says otherwise.

Typed motifs can reuse eligible program fragments and macros while retaining provenance. The prepared motif library does not yet establish the production eligible-state reader, `motifs mine` integration or operational reuse. A new operator requires compatibility tests and an ablation under the unchanged comparison protocol. Repeatedly selecting the best development result can improve development performance without proving improved sealed generalization.

## Present teacher readiness and the smallest real path

The metadata readiness audit verified 28 source-family evidence files and the retained original process/context receipts. The inspected historical origin runs at `55e9ed688faf` and `3fd7d5b2ebb6` had no preregistered teaching freeze or vocabulary pins; they cannot be upgraded after the fact. The operational origin proof at `68f56404` (the reviewed PR83 head, merged on 2026-09-23 as main `a75e543d2` with the identical tree) also was not a teaching campaign. There is therefore no currently demonstrated eligible real corpus for the new ASM admission-and-training proof. This does not negate the historical training corpus or the optimizer work that produced the retained model.

The prospective small pair selected from frozen provenance/size, before observing losses, is:

| Dataset | Evidenced family | Cut 0 fitting rows (development fold 0 held out) |
|---|---|---:|
| `230_machine_cpu` | UCI computer-hardware relative-performance source | 135 |
| `712_chscase_geyser1` | Chatterjee casebook geyser source | 140 |

Both must independently produce eligible recipes. The minimum of ten completed inner fits is only the five-fold×two-family floor for one successful recipe per family. It is not a replacement for the specified matched search and does not promise that either family will improve.

If exactly these two families qualify, seed 7 assigns `230_machine_cpu` to the policy-validation families and `712_chscase_geyser1` to the policy-training families under the current partition rule, leaving a single training family. If more families qualify, use the entire admitted set and let the native family partition determine membership. Do not omit inconvenient eligible comparisons, force the two-family allocation, or reuse the frozen panel's (`asm-private-30-v1`) deadline for a separate trial. Historical checkpoint exposure remains a separately reported unknown; family provenance alone does not prove zero prior exposure.

Both prospective families are small regression datasets of the frozen panel itself (169 and 172 development rows, [§](05-datasets-and-panel.md#per-dataset-summary-and-limitations)); no real binary or multiclass family is prospective. With exactly these two families, the first training campaign is a **mechanics qualification**: it can show optimizer updates, reload, continuation and consumption of a new checkpoint, but it cannot test whether training improves proposals, and its export must not be compared with step 60,000 as a flywheel challenger ([§](12-training-the-policy.md#what-the-pending-campaign-trains)). If either dataset is used to manufacture teachers, later panel evaluations on it are labelled exposed/adaptive ([§](13-flywheel.md#freezing-the-experiment-and-choosing-families)); see also OD-8 in [§](03-roadmap.md#proposed-owner-decisions-and-external-dependencies).

Source evidence: custody `agent-training-readiness/readiness-v3.json`, SHA256 `9197900d4b696811909c4ba4e779e6c5d488569dd9c360165cc55fd43bf8c735`. Its metadata inspection does not itself admit training or re-run numerical qualification.

## Prepare a teaching campaign before it starts

### Final build and the teaching configuration

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

### Plan, run and recover a teaching campaign

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
