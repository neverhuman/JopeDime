# Reading this manual: evidence levels, vocabulary and identities

## Purpose, audience and private scope

This manual serves operators, scientists, implementation maintainers and reviewers. Operators need installation, preparation, execution, reporting and recovery procedures. Scientists need dataset identities, split boundaries, metrics, model lineage, comparison budgets and limits of the evidence. Maintainers need the workspace map, command contracts, fitted-state formats, feature boundaries and tests. Reviewers need to distinguish code that exists, behavior that has been demonstrated, changes that have merged, and claims still requiring evidence.

This is a maintainer document for the private product repository. Its explicitly requested GP examples and internal operational paths explain the implementation; they do not expand the allowlist for public inference bundles.

This is a maintainer document containing internal model and discovery design. The user's request authorizes this explanation and the companion private GP examples. It does not enlarge the allowlist for distributed inference artifacts or authorize publishing private weights, discovery histories, vocabulary assets or training corpora. Hashes identify dependencies; they are not the dependencies themselves.

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

## Names and glossary

ASM is the current product name, settled by the user plan. Historical method IDs such as `afm.lite-race.v1` and `afm.lite-race.v1.resampled` remain exact evidence identifiers in committed same-cut results and the current legacy registry. Use ASM in new prose, but do not rename old identifiers or rewrite frozen reports in place. A method-identity change requires an explicitly versioned new comparison/evidence record. Sources: `crates/jope-bench/src/methods.rs:165`, `:168`; the 2026-09-23 16:07 coordination record.

| Term | Meaning in this repository |
| --- | --- |
| ASM | Algorithm Synthesis Model: learns to generate and optimize executable algorithms from data |
| Algorithm / recipe / program | Executable typed procedure and configuration; reusable across fits. The legacy CLI flag for a recipe is `--genome`. A recipe is not its native fitted predictor, an emitted recipe in another language or a fitted inference bundle ([§](09-programs-and-languages.md#four-different-artifacts)) |
| Method | A comparable benchmark entrant: a registered native recipe (`rust.*`), an external library adapter (`external.*`), an ASM entrant (`asm.neural`, `asm.generic`) or, in historical ladders, a whole system |
| Candidate | A recipe placed in one search slot |
| GP | Genetic programming: search over executable programs using variation and measured selection |
| Arm | One search population in a campaign, with a fixed proposal source and its own declared allowance: the neural arm (policy pool plus exploration slots) or the generic arm (no policy proposals). Historical searches also ran a shuffled-context arm. Unrelated to the Hyperion asset label `arm_b` (the v5 arm-B weights, `hyperion_v5_arm_b.safetensors`) |
| Fitted predictor | One recipe plus preprocessing and model state learned from exact training inputs |
| Checkpoint | Serialized neural proposal-model parameters and compatibility metadata; distinct from a fitted tabular predictor |
| Context | Authenticated numerical dataset/target representation supplied to the proposal policy |
| Dataset encoder | Learned or specified mapping from permitted dataset structure/rows to context features |
| Target encoder / target context | Representation of allowed target information; task-specific and restricted to permitted fitting rows. In the regression context it is the 512-float block computed from min–max-rescaled training targets. The 1,024 foundation values in the 1,536-float “dataset” block also use the training targets ([§](06-encodings-and-context.md#component-dimensions-and-assembly)) |
| Vocabulary | Token identities, grammar and executable restrictions associated with a checkpoint |
| Proposal policy | The neural model that proposes program tokens for a context; the accepted regression policy is the lineage-B step-60000 checkpoint |
| Generation zero | Initial candidate slots before evolutionary descendants; 80 per arm in the current ASM search |
| Seed pool / RNG seed | Seed pool: the programs placed into generation zero of an arm (historically including incumbents and retained teachers). RNG seed: a random-number seed, such as seed 7 |
| Exploration candidate | An admitted generic candidate in the neural arm; not falsely labelled neural-origin |
| Offspring | Mutation or crossover descendant with authenticated parent identities and actual fit evidence |
| OOF | Out-of-fold: predict rows using a fit that excluded those rows; adaptive recipe selection can still expose them at another stage |
| Same-cut cuts (historical) | Hashed cuts built by the historical same-cut benchmark command; not comparable with the frozen panel's cuts ([§](08-evaluation-protocol.md#metrics-and-class-order)) |
| Development-hold score | The score of a final refit on a cut's fitting rows, measured on that cut's held fold (report field `development_hold_score`, `crates/jope-bench/src/panel/origin_proof.rs:321`); the one-dataset proof's RMSE uses 34 held rows |
| Policy-validation families | In the pending teaching partition, the families held out to validate the policy; older text calls them “development” families. Not the panel's development rows |
| Exposure / adaptive | Exposure: use of a partition's labels for selection, optimization or reporting. Adaptive: measured on data that also helped choose what is measured, so not an independent generalization estimate ([§](08-evaluation-protocol.md#adaptive-development-exposure)) |
| Cut identity | Frozen identity of the training/scoring row assignment and evaluator context |
| Inner folds | Five whole-group folds inside a cut's fitting rows, used for out-of-fold selection and, in the pending eligibility path, for teaching admission ([§](10-teachers-and-targets.md#the-preregistered-freeze-and-train-side-inner-split)) |
| Frozen panel | `asm-private-30-v1` under protocol `asm-panel-v1`: 30 PMLB datasets, ten per task, manifest `ba7dace4…` ([§](05-datasets-and-panel.md#frozen-thirty-dataset-panel)). Older text also says official30 or the unchanged thirty-dataset panel. Training-side validation sets that change between checkpoints are not this panel |
| Task / category | Task: regression, binary classification or multiclass classification. Training text, `--task all-admitted` and the owner's goal of ten wins per category use “category” in this sense; elsewhere “category” means a categorical feature value |
| Label | The column a dataset asks a model to predict; one of the senses of “target” |
| Family | A group of related datasets from one source, such as the UCI computer-hardware source of `230_machine_cpu`. The teaching partition assigns whole families to training or validation. The north star (proposed; owner ratification pending) is measured on previously unseen families ([§](13-flywheel.md#the-proposed-north-star-and-matched-comparisons)) |
| Development data | Data visible to optimization or recipe/model selection, regardless of a file's name; in the panel, every non-sealed row |
| Development folds 0–4 | The five original folds of the panel's development rows; every group stays within one fold ([§](05-datasets-and-panel.md#identity-selection-totals-and-immutability)) |
| Cut k | Fit on the development folds other than k and hold out fold k (`dataset.cut(k)`, `crates/jope-bench/src/panel/data.rs:314`). Older text calls cut 0 “fold0” or “development fold zero”. For `230_machine_cpu`, cut 0 has 135 fitting rows and 34 held rows |
| Sealed evaluation | Separately protected evaluation outside the learning and optimization closure. The “seal-protected shared memory” of the training reader (memfd seals) is unrelated |
| Burned partition / burn receipt | In the pending training64 patch: a dataset cut originally in the `test` role whose labels were made visible to optimization, so that they now count as development data, and the immutable `jope.label-burn/v1` receipt that records this relabel. Sealed and `meta_test` labels are refused rather than burned. `require_unburned_test_partition` refuses to present a burned partition as untouched; it has no production caller yet (TEACH-02) |
| HWM | High-water mark: the best retained result for a comparison identity. At `a75e543d2` the current-HWM pointer is keyed only by task, dataset hash and cut (`crates/jope-learning/src/state.rs:13`), so a bound result under a different evaluator, vocabulary or protocol with the same metric can still displace the incumbent (`crates/jope-learning/src/hwm.rs:64`). The pending eligibility/HWM correction (TEACH-01) uses the complete comparison identity (task, dataset, cut, evaluator, split, vocabulary, preprocessing and numerical protocol), so that a result under a different identity cannot displace an unrelated incumbent ([§](10-teachers-and-targets.md#the-admission-guard-label-exposure-and-hwm-identity)) |
| PANEL-WINS (proposed) | **Proposed definition, pending owner ratification; no code at `a75e543d2` computes it.** Per task, the number of the ten frozen datasets on which `asm.neural`'s confirmed five-fold mean development loss is strictly below the minimum over all eligible frozen non-ASM comparators (every `rust.*` and `external.*` recipe). Incomplete, failed or unconfirmed datasets count as not won. `asm.generic` is reported in its own column. Reported as (regression, binary, multiclass); the owner's goal of ten wins per task is (10, 10, 10), a 30/30 sweep ([§](13-flywheel.md#win-counting-and-a-proposed-acceptance-record)) |
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
| Custody | Host-local preservation directories outside Git, such as `/home/ubuntu/jope-rescue-20260921/asm-integration-20260923/`, that hold stopped patches, receipts and evidence. “Custody buckets” (also “custody dev/prod”) are something else: the account's separate `jopedime-714044927654-us-east-2-dev`, `-prod` and `-sealed` S3 buckets ([§](C-data-estate.md#additional-and-replica-buckets)) |
| Elo | Relative Bradley–Terry rating on the arena's declared method universe and eligible pairwise scores |
| Warm demo | Demonstration after verified preparation; report separately from cold installation |
| Promotion | Explicit qualification of an exact artifact/model/backend for its intended serving use; a merge alone is insufficient |

## What version are we running?

### Implementation snapshot and this manual's own identity

The implementation snapshot is **68f56404c3e199e06413bfd645d3fe0803d7b96f**, dated 23 September 2026. [§](17-maintenance.md#worked-example-the-pr83-lifecycle-record) records the separately refreshed reviewed-main state. A rebase merge can produce a different commit with the same reviewed tree. Older plans and historical PASS entries do not establish current capability.

This edition describes the implementation reviewed at `68f56404c3e199e06413bfd645d3fe0803d7b96f`. PR83 merged on 23 September 2026 at 16:34:47 UTC. The resulting remote main is `a75e543d2166017cb0139e0a251d51d9a0c98e08`. Both revisions have Git tree `176b75bc36d76c7ce4eaeb7234677ec013468889`. Rebase merge changed commit identity; it did not change the reviewed source tree.

The manual itself is a later documentation change. Its publication identity must be read from its own Git history and CI receipt, not confused with the implementation snapshot. Follow-on patches described as pending are outside the implementation snapshot even if they are preserved on this host.

Publication state at 2026-09-23 22:34 UTC: edition 1 of this manual is on `main`.

- **PR84 was closed as superseded** (coordination record of 22:22:38 UTC). Its head is `e06c0551e3ec5ee5260debb3d953c17f34d15093`. The independent non-contributor review 5295763455 (state COMMENTED, 19:30:12 UTC, at that head) withheld approval for two reasons: a blocking finding that three intermediate commits each contain two files larger than the repository's 5 MiB cap (`deploy/ci/check.sh:46`), so a rebase merge would put those commits on `main`; and exact-head CI that was not yet fully green.
- **PR85, the clean-history successor, merged at 22:33:36 UTC.** Codex, the manual owner, published the single commit `2022017a8c45cc25967ae99ae781ca26276140b8` on `a75e543d2` (branch `docs/asm-maintainer-manual-final`, pushed at 22:21:30 UTC) with a 203-page PDF (SHA-256 `69418c48…`). Exact-head CI run 35927921851 attempt 1 succeeded, and the independent non-contributor review 5297559032 approved that head at 22:32:39 UTC. The merge produced `main` `bdfc95c63d905c647e41175e79dc30cdc8600287`, whose Git tree `fbbed08a…` is identical to that of `2022017a8`.
- **This edition 2** restructures and corrects `2022017a8`. It was prepared in Claude's lane for the manual owner to integrate. Like any manual change, it needs fresh exact-head CI and an independent non-contributor review before it can merge.

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
| Experimental ASM policy (pending) | The stopped training64 patch would train a new Python `JopePolicyModelV3` (width 256, 2 layers, 4 heads, feed-forward 1024), initialized at most from a Python checkpoint; no real campaign has run with it | Experimental export only: not serving, not lineage B and not a continuation of step 60000, which the Python path cannot load. Compare the two only through matched downstream fits, never through loss values. The native route that would continue step 60000, `jope-parallel`, is held ([§](12-training-the-policy.md#the-dispatch-hold-resource-incompatibility-and-valid-topologies)). |
| Encoder | Current CPU (Rust/Candle FP32) context schema `jope.cpu-context/v1`, separate from the held native GPU path; produced artifacts require pinned native-fixture replay before qualification. Dataset 1536 (Hyperion 512 from the features only + projected foundation 1024 from the features and training targets) + target 512 = 2048 floats, so “dataset 1536” is a historical label that includes target-conditioned values; legacy frozen encoder descriptor `parallel.frozen_encoder.v1` | Every component weight, transform, source/backend and assembly hash participates in identity. Dimensions alone do not establish compatibility. |
| Hyperion assets | Current CPU producer pins the v5 arm-B weight hash `88b9e8345e44d3ea609a8286215618a762f14e607fcdada6d27447b1f74668be`; a separately stored bundle is called `hyperion-v7` | A bundle name or Rust `V7Model` type is not permission to substitute newer-looking weights. “v5” here is asset lineage, not a global product schema version. |
| Action vocabulary | Accepted 401-token vocabulary SHA `8d55fee4366acd2c175cf55bfd769cebc904770f5227a6fd50a613723dc85787`; documented product restriction SHA `570b2e5e8d859a73b888f5d0bee27f59da0e2376536babce24c241604f818e81` | The restriction preserves the token table. Adding tokens or changing parameter bins requires a new compatibility and training decision. |
| Corpus | Native `parallel.teaching_shards.v1` / `parallel.teaching_corpus.v1`, plus immutable index, vocabulary, contexts, records, label policy and exposure identities | A corpus schema does not certify teacher eligibility, data freshness, or unseen-family status. |
| Dataset/catalog | `jope.dataset-catalog/v1`, `jope.dataset-collection/v1`; individual content and split hashes remain separate | Keeping the schema while changing rows, labels, categories, groups or splits produces a different comparison. |
| Model packaging | `jope.model-bundle/v2` | Package/license/inventory metadata is separate from accepted model identity and successful inference. |
| Frozen benchmark | `asm-panel-v1:screen-fold=0:top=5:confirm=all-5:mean-fold-loss`; evaluator `jope-bench.frozen-panel-f64/v1`; manifest SHA `ba7dace4cd535687002cc95538dbc7f7aaf77ec973177c7159b3c6d22471eee0` | Altering a method, split, deadline, evaluator or selection rule requires a new protocol identity. Do not relabel an altered run as the frozen panel. |

Source: [workspace version](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/Cargo.toml#L30), [Python package](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/python/pyproject.toml#L5), [accepted checkpoint and vocabulary pins](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-policy/src/lib.rs#L132), [Python model version](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/python/src/jope/policy/model_v3.py#L43), [resume schema](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/python/src/jope/policy/train_v3.py#L57), [context assets and identity](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-policy/src/context_producer.rs#L35), [legacy encoder descriptor](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-engine/src/encoding.rs#L47), [catalog schemas](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-datasets/src/catalog.rs#L9), [bundle schema](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-models/src/manifest.rs#L9), [panel identity](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-bench/src/panel/mod.rs#L36).

### Reading identities from the immutable CLI

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

## Release and installation are independent from model promotion

The current tag workflow runs for `v*`, checks tag = workspace = Python version, builds a Linux binary, writes build provenance and checksums, builds the Python wheel, and records model/dataset references by hash. The committed build command selects default CLI features; it does **not** include `--features asm`. Therefore the existence of this workflow does not prove that a released binary can execute the private ASM path. No newly published release is asserted here. The current private source build in [§](14-install-and-prepare.md#development-build-of-the-cli) explicitly selects `asm`.

[Release workflow](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/.github/workflows/release.yml#L71) and [version consistency](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/deploy/ci/version-consistency.sh#L1) are source evidence. A future reviewed version bump can use `just bump X.Y.Z`; that recipe edits manifests and the lockfile and invokes Cargo. It is a developer mutation recipe, not a command executed for this manual or a decision to release any particular version. [Exact bump implementation](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/justfile#L63).

The checksum-verifying private binary installer and CLI-only source-build helper remain product28 follow-ons. Their concrete commands and cold/warm separation are in [§](14-install-and-prepare.md#build-and-install). Installing a new CLI does not admit new data, train a model, or promote a checkpoint. Likewise, `models pull` or a storage manifest does not change the accepted policy pins.
