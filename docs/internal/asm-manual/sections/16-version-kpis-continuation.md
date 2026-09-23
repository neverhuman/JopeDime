# Versions, measurable progress and continuing the deep model

## The north star and what it does not yet prove

**ASM learns to generate and optimize executable algorithms from data.** Its proposed north star is better downstream predictive loss from generated, actually fitted algorithms on previously unseen dataset families, under the same compute allowance, with reproducible predictions and a usable fitted artifact. This is an operational definition for the next evaluation, not a newly ratified release or a claim that the current model has already met it.

Compare each challenger against both a matched generic search and the previous checkpoint, with the same grammar, task, rows, class ordering, preparation, seeds, resources, scoring policy and exposure rules. A lower imitation loss alone is insufficient: the generated programs must compile or interpret, fit, survive the resource limits, and improve predictions. Conversely, an honestly measured optimizer update is training progress even when the downstream candidate does not win.

The current frozen thirty-dataset demonstration is an adaptive **development** comparison. Selection and confirmation exposure is recorded. It cannot establish the previously-unseen-family part of the north star merely because a dataset appears in that panel. That requires independently frozen family/exposure evidence and a separately authorized evaluation; this manual grants no access to sealed labels. Historical exposure that cannot be reconstructed stays unknown. See [training history](14-training-history.md), [target history and packing](15-target-history-packing.md), and [flywheel closure and focused search](17-flywheel-focused-search.md).

The latest retained source-68 origin proof is a useful concrete boundary: it executed actual neural proposals, fitted generation zero, and fitted neural descendants, with an authenticated recovered report. Its neural RMSE was 166.3181717454848 versus generic 155.25909998168862 on one regression dataset. This demonstrates consumption, not a quality win or a thirty-dataset result. The fourteen-command private GP example establishes recipe-refit prediction parity across five language configurations; it is not exact-state fitted-winner export. Historical trained corpora also must not be confused with the zero newly demonstrated teachers under the pending ASM admission contract. The measured origin report SHA is `6e0c20720256b0735a6969c45032f44082c92abacadf2c4eb21ef5c390f249dd`; see [measured visual evidence](08-visual-evidence.md) and [the private language example](05-languages.md).

Historical optimizer evidence is now more complete: 21 original A100-recovery logs retain all 63,000 steps from 6,001 through 69,000, with 378,000 selected loss/component/gradient values preserved exactly. All 21 blocks show declining within-block composite loss, while each uses a different corpus. This is actual historical fitting evidence, separate from better downstream GP targets or proposal quality. Accepted serving remains step 60,000; later A100 step 69,000 is explicitly refused, and the separately retained world-1 step 60,100 is hardware qualification. See the source-bound CSV, corpus boundaries and explicit averaging in [training history](14-training-history.md). No new training or promotion is implied by discovering these logs.

## What version are we running?

There is no single version number that describes all of the following. Record the entire identity tuple whenever comparing results or continuing training.

| Dimension | Current evidenced identity | What changing it means |
|---|---|---|
| Product/package | Rust workspace and Python package `0.1.0`; immutable CLI reports `jope 0.1.0` | A packaging/API release number; not a checkpoint generation or proof of installed capability. |
| Implementation | Reviewed source `68f56404c3e199e06413bfd645d3fe0803d7b96f`, merged as `a75e543d2166017cb0139e0a251d51d9a0c98e08` with the same source tree | A later documentation commit does not rebuild the binary. Preserve binary/build/source-manifest hashes. |
| Accepted regression policy | Lineage B, step 60000, `parallel.checkpoint.v1`; state hash `8ac80c269cc47c253b6f53fa3bc1eb75bb2fcda0bdace3ff8554be0175c400d6` | The current accepted consumer hard-pins this identity and its model/configuration/parameters. |
| Hardware-qualification continuation | Step 60100 is retained hardware-qualification continuation evidence, state SHA `745be4b5e998bce85615349a5b03b901cd8ad2cf605e6dce3b86fbf777cc7388`; verify the actual training-head pointer independently | This is not the accepted serving checkpoint. Do not advance a pointer from a filename or step number. |
| Rejected policy | Step 69000 is explicitly refused by state or model hash | A numerically higher training step does not imply promotion. |
| Python model family | Current Python implementation declares `policy_v3.4-split-context`, with legacy `policy_v3.3-fused-heads`; trainer resume format `PrimePolicyTrainerResume.v1` | These identify a different trainer/model serialization surface from the accepted native `model.ot` lineage. |
| Encoder | Current CPU (Rust/Candle FP32) context schema `jope.cpu-context/v1`, separate from the held native GPU path; produced artifacts require pinned native-fixture replay before qualification. Dataset 1536 (Hyperion 512 + projected foundation 1024) + target 512 = 2048 floats; legacy frozen encoder descriptor `parallel.frozen_encoder.v1` | Every component weight, transform, source/backend and assembly hash participates in identity. Dimensions alone do not establish compatibility. |
| Hyperion assets | Current CPU producer pins the v5 arm-B weight hash `88b9e8345e44d3ea609a8286215618a762f14e607fcdada6d27447b1f74668be`; a separately stored bundle is called `hyperion-v7` | A bundle name or Rust `V7Model` type is not permission to substitute newer-looking weights. “v5” here is asset lineage, not a global product schema version. |
| Action vocabulary | Accepted 401-token vocabulary SHA `8d55fee4366acd2c175cf55bfd769cebc904770f5227a6fd50a613723dc85787`; documented product restriction SHA `570b2e5e8d859a73b888f5d0bee27f59da0e2376536babce24c241604f818e81` | The restriction preserves the token table. Adding tokens or changing parameter bins requires a new compatibility and training decision. |
| Corpus | Native `parallel.teaching_shards.v1` / `parallel.teaching_corpus.v1`, plus immutable index, vocabulary, contexts, records, label policy and exposure identities | A corpus schema does not certify teacher eligibility, data freshness, or unseen-family status. |
| Dataset/catalog | `jope.dataset-catalog/v1`, `jope.dataset-collection/v1`; individual content and split hashes remain separate | Keeping the schema while changing rows, labels, categories, groups or splits produces a different comparison. |
| Model packaging | `jope.model-bundle/v2` | Package/license/inventory metadata is separate from accepted model identity and successful inference. |
| Frozen benchmark | `asm-panel-v1:screen-fold=0:top=5:confirm=all-5:mean-fold-loss`; evaluator `jope-bench.frozen-panel-f64/v1`; manifest SHA `ba7dace4cd535687002cc95538dbc7f7aaf77ec973177c7159b3c6d22471eee0` | Altering a method, split, deadline, evaluator or selection rule requires a new protocol identity. Do not relabel an altered run as the frozen panel. |

Source: [workspace version](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/Cargo.toml#L30), [Python package](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/python/pyproject.toml#L5), [accepted checkpoint and vocabulary pins](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-policy/src/lib.rs#L132), [Python model version](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/python/src/jope/policy/model_v3.py#L43), [resume schema](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/python/src/jope/policy/train_v3.py#L54), [context assets and identity](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-policy/src/context_producer.rs#L35), [legacy encoder descriptor](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-engine/src/encoding.rs#L13), [catalog schemas](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-datasets/src/catalog.rs#L9), [bundle schema](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-models/src/manifest.rs#L9), [panel identity](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-bench/src/panel/mod.rs#L36).

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

The current tag workflow runs for `v*`, checks tag = workspace = Python version, builds a Linux binary, writes build provenance and checksums, builds the Python wheel, and records model/dataset references by hash. The committed build command selects default CLI features; it does **not** include `--features asm`. Therefore the existence of this workflow does not prove that a released binary can execute the private ASM path. No newly published release is asserted here. The current private source build shown in the operations chapter explicitly selects `asm`.

[Release workflow](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/.github/workflows/release.yml#L61) and [version consistency](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/deploy/ci/version-consistency.sh#L1) are source evidence. A future reviewed version bump can use `just bump X.Y.Z`; that recipe edits manifests and the lockfile and invokes Cargo. It is a developer mutation recipe, not a command executed for this manual or a decision to release any particular version. [Exact bump implementation](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/justfile#L63).

The checksum-verifying private binary installer and CLI-only source-build helper remain product28 follow-ons. Their concrete commands and cold/warm separation are in [product operations](03-operations.md). Installing a new CLI does not admit new data, train a model, or promote a checkpoint. Likewise, `models pull` or a storage manifest does not change the accepted policy pins.

## The continuation interfaces that exist today

| Interface | Current behavior | Appropriate interpretation |
|---|---|---|
| `jope train-step --local` | Fits a centroid router from local labels | Small router training; not the deep policy model. |
| `jope learn ...` | Not exposed by the source-68 umbrella parser | ADR operating intent; do not present as a working command. |
| `jope-parallel train ...` with `neural` | Native regression trainer syntax exists; direct numerical dispatch is held | Retained native continuation machinery, not presently usable through this direct entry point. |
| `jope-py train-policy --init-from ...` | Python policy weight warm start; CLI exposes no `--resume-from` | A new optimization run initialized from compatible weights, not exact optimizer continuation. |
| `train_policy_v3_from_jsonl(..., resume_from=..., resume_checkpoint=...)` | Existing Python API restores model, optimizer, scheduler, scaler and RNG subject to compatibility | A real continuation API, requiring a trusted matching Python resume artifact; not an accepted-native-checkpoint conversion. |
| Pending `jope train plan/run/resume/verify/export` | Supervised, task-routed, bounded training proof and experimental export | Preferred integration target for the approved one-GPU-hour demonstration; still unintegrated and unexecuted. |

Sources: [current umbrella commands](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-cli/src/main.rs#L26), [local train-step dispatch](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-cli/src/main.rs#L308), [Python CLI flags](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/python/src/jope/cli.py#L186), [Python API](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/python/src/jope/policy/train_v3.py#L1331).

### Variables used in the continuation examples

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
| `NEW_RUN` | New private output directory for the Python example; the continuation helper requires it not to exist. This new output location does not renew a resource or time allowance. |
| `FINAL_JOPE`, `FINAL_SOURCE_SHA` | Future source-bound CLI and full source commit after the pending training integration and its required checks; not the current source-68 executable. |
| `LEARNING_STATE`, `CORPUS_REQUEST`, `CORPUS_REQUEST_SHA256`, `CORPUS_OUTPUT` | Admitted learning state, exact corpus-preparation request and its file SHA256, and private prepared-input output directory, as specified in [training operations](04-training.md). |
| `TRAINING_STATE`, `TRAINING_EXPORT` | Original supervised training state directory and private experimental export output directory; resume reuses `TRAINING_STATE`. |

### Native lineage B: the held command and its blockers

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

Here `--steps 100` declares the same increment for the whole original run; resume does not authorize another hundred steps. A completed run explicitly refuses resume. `--initialize` alone starts fresh FP32 AdamW and resets the new-corpus sampler; preserving optimizer ancestry requires the verified carry policy and compatible objective/partition evidence. Do not fabricate that policy JSON or relabel an incompatible recovered checkpoint. Corpus, vocabulary, model, optimizer, config, sampler, partition and deadline checks occur before CUDA.

The current direct controller calls `require_original_gpu_grant`, which **always returns an error**: “original native GPU grant verifier unavailable; numerical dispatch is held.” Environment variables cannot release this gate. A separate typed native funding-admission API exists, but this is not a working direct-trainer bypass. Even after a future proper dispatch integration, the controller's fresh-telemetry check requires at least 128 GiB available RAM and 50 GiB free disk, incompatible with the current approved 64 GiB limit. Valid normal microbatches are 64, 128 or 256, and global batch is 2048. With world 1/microbatch 64, accumulation is 32. The retained step 60100 hardware qualification actually used world 1, microbatch 256 and accumulation 8, carrying AdamW from 60000 to 60100 on the unchanged corpus with zero fresh program presentations. Its 100-step update is genuine optimizer work; it is not production continuation or promotion. The ADR's microbatch 32 example is inaccurate for that retained run and is not a currently accepted CLI topology.

Before any future continuation, audit the **entire** input closure, including every context descriptor and staged tensor, not only training records. The retained snapshot 97 index lists 8,894 contexts, including 878 `meta_test` descriptors, while record rows are train/validation only. Current legacy training stages all indexed contexts before sampling. Absence of sealed record rows therefore does not prove strict sealed-family exclusion. Do not blindly continue that corpus under the new closure policy; preserve the metadata observation and resolve it without opening sealed payloads. See [target history and packing](15-target-history-packing.md) and [context staging source](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-engine/src/training.rs#L1179).

Sources: [native command parser](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-engine/src/main.rs#L106), [controller admission and step window](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-engine/src/controller.rs#L478), [dispatch hold](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-engine/src/task_contract.rs#L85), [checkpoint checks and warm start](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-engine/src/training.rs#L923).

### Current Python training and a concrete continuation call

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

For **actual optimizer continuation**, call the existing API from a private maintainer script. The following is the essential code; `settings.json` must contain the original compatible `kwargs`, and all referenced inputs must be hash-pinned in the operator's original run inventory. The complete companion `continue-python-policy.py` includes those byte-hash checks, exclusive output creation, explicit GPU preflight, and refusal of distributed or non-current resume modes. It is an illustrative API caller, not a new installed product command or a qualified supervisor.

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

This current Python route has material limits. CUDA requests can silently fall back to CPU in the underlying code, so the companion refuses absent CUDA before calling it. Current task-routing/zero-update and resume-proof fixes belong to the pending training overlay. The loader can discard minority embedding widths with a warning; a qualified input inventory should reject such a mixed corpus earlier. Existing Python training alone does not provide the shared cgroup resource contract, original campaign deadline, teaching admission, parameter-change proof or serving promotion. Execute any diagnostic caller only inside a separately admitted resource/time supervisor, not as a way to bypass the pending product gates. See [current device fallback](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/python/src/jope/policy/ddp.py#L25) and [mixed-width handling](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/python/src/jope/policy/train_v3.py#L1485).

## The pending bounded extension path

The stopped training64 patch is `595824ef6eb8e15872be058668503cedad1dd9a5ff98ac4700c6f713251d384c`, with manifest `b86b550d7ee39fd93d5202accd5f77c975b7bcb4fd8f8e7706c69c5f279294a2`. It has not been integrated into source 68 or qualified by a fresh real GPU run. Its exact command contract is described in [training operations](04-training.md); pending files cannot honestly be linked as existing source 68 GitHub files.

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

Run and resume are alternatives according to state; do not blindly execute resume after a completed run. The whole campaign permits one RTX 3090 and one GPU-hour, at most 20 minutes per category, 20 CPUs, 64 GiB and zero swap. Two passes share each category window; they do not receive fresh time allowances. At least two real admitted families are necessary before claiming the real regression training path. Classification fixtures may demonstrate labeled mechanics when real admission fails, but remain synthetic. Unknown or unavailable categories are reported rather than filled with invented success.

Verification must establish positive updates, changed parameters, reload equality, actual optimizer continuation, full input/runtime/source/device/vocabulary compatibility and persisted accounting. Export is experimental. The pending `experimental.regression`, `experimental.binary` and `experimental.multiclass` consumer entries bind task-specific exported checkpoints; a later real fit must prove the new checkpoint generated and fitted candidates. None of these commands updates the accepted serving pointer. Product28, warm-target and winner-export follow training64; they do not fix teacher eligibility by themselves.

## A KPI ledger with defined denominators

These are proposed reporting definitions for acceptance, not a claim that every current report already emits every field. Preserve the fixed protocol where it already specifies a quantity. Additions must be preregistered before comparing challengers; do not change a definition after seeing results.

| KPI | Definition and denominator | Evidence and present gap |
|---|---|---|
| Task loss delta | For a valid matched pair, `loss_reference − loss_challenger`; positive means better. Regression loss is RMSE; binary loss is `1−AUROC`; multiclass is ordered probability-matrix log loss. Report both generic and previous-checkpoint references separately. | Exact predictions, rows/classes, evaluator, numerical policy and resource receipts. Never average raw RMSE with log loss across tasks. |
| Strict wins / planned comparisons | Count valid matched improvements over all preregistered comparisons, including failed/unavailable comparisons in the denominator. Report ties, losses, one-sided failures and two-sided failures separately. | Complete planned inventory. Failure is not a fabricated numerical loss; paired mean delta is separately labeled as complete-pair-only with coverage. |
| Family generalization | Per-task mean of family means under the preregistered split, with cluster uncertainty at family level when sample size permits | Frozen family provenance and prior-exposure evidence. No current thirty-panel unseen-family claim; historical unknown exposure stays unknown. |
| Relative improvement | `(Lref−Lnew)/Lref` only for finite, strictly positive reference loss; report absolute delta when zero | No hidden epsilon convention or cross-task aggregate introduced after results. |
| Teaching admission | Eligible comparisons / all requested comparisons, and eligible independent families / all requested families | Preregistered five-fold train-side OOF evidence, finite baseline improvement and improvement on four of five folds. Newly demonstrated ASM-qualified count currently zero; historical corpus counts are separate. |
| Genuine neural consumption | Completed `neural_policy` generation-zero candidates / actual `neural_policy` slots; also report completed genuine neural candidates /64 policy-pool slots, fallback slots and completions, and exploration completions /16. Completed descendants require verified neural ancestry. | Authenticated proposal→request→started→finished→artifact chains. A recipe file, neural tag, generic fit or started-only attempt is insufficient. |
| Validity and fit coverage | Canonical task-compatible proposals / all proposed draws; completed fits / all scheduled eligible fits | Distinguish grammar refusal, dependency unavailability, fit error, timeout, cancellation, never-started and unknown. State whether duplicate candidates are deduplicated. |
| Optimizer progress | Updates this run, cumulative optimizer step, start/end parameter digests, committed passes, and exact resumed-state identity | Batch/epoch metrics and checkpoint proofs. Positive epoch count or a checkpoint filename is insufficient; training loss denominator and objective must match before plotting together. |
| Latency | Cold build/install/preparation separately; warm end-to-end wall time and declared target `<60s`; actual per-arm planned/available/executed time | Original monotonic/absolute windows and launch receipts. A completed >60s demo can be execution-complete with target missed. |
| Resource and cost efficiency | Total known creation CPU/GPU/wall and monetary/token costs divided by authenticated distinct complete fitted artifacts; also cost per distinct reusable recipe | Distinct means declared hash identity, not proven functional uniqueness. Whole-unit counters overlap worker sums; do not add them. Unknown overhead, price or tokens remain null with reason. |
| Fitted portability | Qualified exact selected fitted predictor/backend pairs / all requested selected predictor/backend pairs | Actual emitted code execution and prediction parity, in order Python/Rust then C++17/C++20 then JS. Unsupported pairs stay in the inventory and never replace the winner. Current five-language toy parity is recipe refitting only. |
| Reproducibility | Successfully authenticated independent report/reload replays / all requested replays | Same source, model, input, plan, protocol and artifact identities. Stale outputs, missing rows or changed class order are failures. |
| Promotion | Accepted serving-head advance with independent matched evaluation, exact checkpoint identity and controlled pointer update | Separate from training-head advance and experimental export. No new promotion is claimed. |

Define a dataset-level comparison using the protocol's frozen fold summary; retain fold-level results separately. Keep separate win columns for generic search, the previous checkpoint, and the best eligible frozen non-ASM comparator. A win against one reference is not automatically a win against the others. For a 30-dataset target of ten wins per category, each reference's denominator remains ten per category and all thirty outcomes are listed, including unavailable ones. It is a measured target, not a promise. There is currently no completed full-panel result establishing those counts.

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

## Extending this version without losing comparability

A new dataset requires a content/split/group identity, family and exposure classification, train-only preparation, task-correct scoring, and a retained unavailable outcome if any prerequisite fails. A changed encoder requires component/assembly qualification and a new input identity, even when it still outputs2048 values. A vocabulary change requires explicit token compatibility, executable canonicalization and checkpoint migration decisions. A new learned policy requires optimizer/reload/continuation proof followed by actual downstream generation-and-fit evaluation. A new operator additionally requires typed grammar support, task applicability, ablation and fitted-state/export coverage. A faster scheduler or emitter requires unchanged input/loss semantics and actual cancellation/resume/parity evidence.

Do not close this loop by editing version strings, declaring an old snapshot newly eligible, treating a warmer loss curve as promotion, or retagging a recipe-refit example as a fitted export. Use the source interfaces that exist, integrate the bounded missing path through the serial reviewed queue, and retain every remaining acceptance gate in [flywheel closure and focused search](17-flywheel-focused-search.md).
