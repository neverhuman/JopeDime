# Training, continuation, export and compatibility

## What can be claimed now

At source `68f56404c3e199e06413bfd645d3fe0803d7b96f`, the latest bounded regression origin proof has actual neural proposals, fitted generation-zero candidates, fitted descendants and a successfully authenticated recovered report. The newly authored training path is still a stopped follow-on. Its guarded 64-file patch is `595824ef6eb8e15872be058668503cedad1dd9a5ff98ac4700c6f713251d384c`; manifest `b86b550d7ee39fd93d5202accd5f77c975b7bcb4fd8f8e7706c69c5f279294a2` binds every original packet, overlap resolution and current source/API guard.

The follow-on passed custody replay guards, source syntax, pure metadata checks and feature-graph inspection. That evidence does not establish Rust type checking, a GPU optimizer run, a real newly trained checkpoint, or a successful live trained-checkpoint consumption loop. The demonstrated teacher count under this pending ASM admission contract remains zero in the inspected evidence. Historical lineage-B teaching corpora and its accepted trained checkpoint are separate retained assets, described in [§](data-estate.md#retained-corpus-sizes) and [§](training-history.md#carried-state-accepted-step-60000-and-rtx-step-60100); they have not been retroactively certified under the new admission contract. Source support, fixture tests, exact-head CI, live qualification and serving promotion must remain separate status fields.

All commands below involving `jope train` or experimental consumer configuration describe the **pending overlay's implemented interface**. They are a maintainer runbook for integration and authorized acceptance, not claims that these commands were executed at main `a75e543d2` (PR83). Preserve this status notice when extracting examples.

**Status as of 2026-09-23 about 23:55 UTC.** The implementation snapshot is unchanged: reviewed `68f56404` merged as main `a75e543d2`. Main has since advanced to `bdfc95c63` (PR85), whose changes since `a75e543d2` touch only documentation, `README.md` and the coordination ledger. Training64 remains a stopped patch outside main; no training GPU window has been admitted and no training clock has started. Each runbook in this chapter carries exactly one status label per route: **available**, **implemented but unqualified**, **pending integration** or **proposed**. The owner's edition-2 design requirements appear here as rules or proposals, not as achieved results.

## The four continuation names and their commands

### Names, runbooks and status

Every way this chapter starts or continues training has exactly one of four names. Report a run by its name rather than by a command flag, and never describe one kind of run with another's name. [§](plan.md#the-continuation-interfaces-that-exist-today) lists the same interfaces at a glance; this chapter remains the operational runbook.

| Name | Definition | Runbooks in this chapter, with status |
|---|---|---|
| Cold initialization | New parameters from the declared architecture and seed, with a fresh optimizer; no parent weights or optimizer state | Native `jope-parallel train` with neither `--initialize` nor `--resume`, which records initialization kind `fresh`: implemented but unqualified (dispatch held). `jope-py train-policy` without `--init-from`: implemented but unqualified. The pending campaign's first pass when its inputs name no initialization checkpoint: pending integration. |
| Weight warm start | Parent weights only; the optimizer starts fresh | Native `--initialize` with no replay policy (recorded as fresh FP32 AdamW, with the new corpus starting at step zero) or with a policy whose `optimizer` is `reset`: implemented but unqualified (held). `jope-py train-policy --init-from` in [§](#weight-warm-start-with-the-existing-cli): implemented but unqualified. The pending campaign's first pass from an initialization checkpoint: pending integration. |
| Explicit new-corpus transition | Parent weights **and** optimizer state carried onto a different corpus, with the optimizer carry and the sampler initialization both declared | Native `--initialize` with a verified `parallel.replay_policy.v2` policy whose `optimizer` is `carry`; the run records "carried FP32 AdamW with bias-correction step" and a lineage-bound sampler counter whose local block step starts at zero: implemented but unqualified (held). No Python route exists: the pending trainer's strict input provenance rejects a changed corpus on resume. Sibling branches built on this transition are proposed ([§](#proposed-sibling-fresh-and-replay-branches)). |
| Exact interrupted resume | The original run continues from its own saved state, with its original inputs and original deadline; no allowance is renewed | Native `--resume` in [§](#the-implemented-command-shape): implemented but unqualified (held). The Python API and its companion helper in [§](#exact-optimizer-continuation-through-the-python-api): implemented but unqualified. Pending `train resume` in [§](#plan-run-resume-verify-and-export): pending integration. |

Two passes through one corpus are continuation, not two generations: the pending campaign's second pass continues the same run on the same corpus inside the same window ([§](#plan-run-resume-verify-and-export)).

**Historical instances.** Lineage B's A100 recovery blocks each trained on a distinct corpus while carrying FP32 AdamW, the historical form of a new-corpus transition. Their accepted 512x6 step 60,000 remains the serving model and cannot be exactly resumed, because its state records no `execution_partition`. RTX step 60,100 is a historical hardware continuation: carried AdamW on the unchanged corpus with zero fresh programs, not a production continuation or a promotion. Step 69,000 remains unaccepted. See [§](training-history.md#carried-state-accepted-step-60000-and-rtx-step-60100).

### Rules for exact interrupted resume

These are owner requirements, **pending integration**: no current or pending route meets both of them yet.

1. **Reject optimizer fallback before any update.** An exact resume whose optimizer state is missing or marked incompatible must stop before its first update. It must never continue with a fresh optimizer under the resume name.
2. **Prove state parity.** A deterministic run that is interrupted and resumed must reach the same parameters, optimizer moments, scheduler, RNG, sampler position and counters as the same run left uninterrupted.

Present behavior against these rules:

- The Python API starts a fresh optimizer, scheduler and scaler when a resume artifact marks its optimizer state incompatible (main `train_v3.py:1919–1930`; pending `train_v3.py:2095–2111`). The companion helper refuses such an artifact before it calls the trainer, so it meets rule 1 for its own narrow case; it provides no parity proof.
- The pending campaign checks `optimizer_state_resumed` only after a pass has trained (pending `training_campaign.py:771`), so a fallback can consume updates and window time before it is rejected.
- Resume artifacts are written at epoch end without a minibatch cursor, and the campaign commits whole passes: an interrupted pass restarts from the last committed head inside the original window (pending `training_campaign.py:799`, `:1065`). That re-runs the pass; it is not an update-level continuation.
- Native `--resume` checks the execution partition, sampler and objectives before CUDA (`crates/jope-engine/src/training.rs:1106–1110`). This manual has no parity evidence for it, and its dispatch is held.
- No existing test compares interrupted and uninterrupted trajectories.

The proposed follow-up is an explicit strict-resume mode that rejects before updates; update-boundary checkpoints that carry the sampler permutation and cursor, counters and all RNG state; and deterministic equality tests of interrupted against uninterrupted runs. Warm-start behavior keeps its own name. Source: Codex's read-only analysis `boundary-resume-siblings.md` in custody `plan-execution-20260923`.

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
| `NEW_RUN` | New private output directory for the Python example; the continuation helper requires it not to exist and refuses it under `/tmp` or inside a Git working tree. This new output location does not renew a resource or time allowance. |
| `FINAL_JOPE`, `FINAL_SOURCE_SHA` | Future source-bound CLI and full source commit after the pending training integration and its required checks; not an executable built from the current main `a75e543d2`. |
| `LEARNING_STATE`, `CORPUS_REQUEST`, `CORPUS_REQUEST_SHA256`, `CORPUS_OUTPUT` | Admitted learning state, exact corpus-preparation request and its file SHA256, and private prepared-input output directory, as specified in [§](#build-the-real-corpus-from-admitted-evidence). |
| `TRAINING_STATE`, `TRAINING_EXPORT` | Original supervised training state directory and private experimental export output directory; resume reuses `TRAINING_STATE`. |

### Native lineage B: the held command and its blockers

| Runbook part | Native lineage-B continuation |
|---|---|
| Purpose and status | Continue lineage B natively: a weight warm start or an explicit new-corpus transition with `--initialize`, an exact interrupted resume with `--resume`. **Implemented but unqualified**: every direct dispatch stops at the dispatch hold. |
| Prerequisites and resources | A source-bound `neural` build; a compatible corpus with verified allowed input closure; a verified parent checkpoint and carry policy. The controller's fresh-telemetry check requires at least 128 GiB available RAM and 50 GiB free disk, more than the approved 64 GiB limit. |
| Manifest-creation command | None is qualified in this manual. Main contains a corpus publisher (`jope-parallel build-corpus`) and code that writes `parallel.replay_policy.v2` policies, but none has been qualified for this runbook. Supply an already verified corpus and policy, never a fabricated one. |
| Execution | The commands in [§](#the-implemented-command-shape), which are not to be dispatched under the present allocation. |
| Artifacts and success | Checkpoint directories whose `state.json` binds model, optimizer, configuration, corpus and vocabulary hashes, plus the run's `initialization.json`. No success receipt exists for this held route. |
| Failure and resume | `--resume` applies only to an interrupted original run with the identical window and inputs. A completed run refuses resume, and accepted step 60,000 cannot be resumed ([§](#rules-for-exact-interrupted-resume)). |
| Limitations | The dispatch hold, the resource gate and the world-8 partition of the accepted lineage ([§](#the-dispatch-hold-resource-incompatibility-and-valid-topologies)). |

#### The implemented command shape

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

Here `--steps 100` declares the same increment for the whole original run; resume does not authorize another hundred steps. A completed run explicitly refuses resume. `--initialize` alone starts fresh FP32 AdamW and resets the new-corpus sampler; preserving optimizer ancestry requires the verified carry policy and compatible objective/partition evidence. Do not fabricate that policy JSON or relabel an incompatible recovered checkpoint. Corpus, vocabulary, model, optimizer, config, sampler, partition and deadline checks occur before CUDA. In this chapter's names, the first command is a weight warm start when the policy's `optimizer` is `reset`, and an explicit new-corpus transition when it is `carry` and the corpus differs from the parent's; the second command is an exact interrupted resume.

#### The dispatch hold, resource incompatibility and valid topologies

The current direct controller calls `require_original_gpu_grant`, which **always returns an error**: “original native GPU grant verifier unavailable; numerical dispatch is held.” Environment variables cannot release this gate. A separate typed native funding-admission API exists, but this is not a working direct-trainer bypass. Even after a future proper dispatch integration, the controller's fresh-telemetry check requires at least 128 GiB available RAM and 50 GiB free disk, incompatible with the current approved 64 GiB limit. Valid normal microbatches are 64, 128 or 256, and global batch is 2048. With world 1/microbatch 64, accumulation is 32. The retained step 60100 hardware qualification actually used world 1, microbatch 256 and accumulation 8, carrying AdamW from 60000 to 60100 on the unchanged corpus with zero fresh program presentations. Its 100-step update is genuine optimizer work; it is not production continuation or promotion. The ADR's microbatch 32 example is inaccurate for that retained run and is not a currently accepted CLI topology.

Sources: [native command parser](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-engine/src/main.rs#L106), [controller admission and step window](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-engine/src/controller.rs#L479), [dispatch hold](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-engine/src/task_contract.rs#L88), [checkpoint checks and warm start](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-engine/src/training.rs#L923).

### Current Python training and a concrete continuation call

#### Weight warm start with the existing CLI

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

| Runbook part | Python weight warm start |
|---|---|
| Purpose and status | Start a new Python optimization run from compatible weights: a weight warm start. **Implemented but unqualified**: the CLI exists at `a75e543d2` and was source-checked, not executed, for this chapter. |
| Prerequisites and resources | A verified Python environment, an admitted compatible JSONL corpus, a trusted weight artifact of the stated architecture, and one CUDA device inside a separately admitted resource/time supervisor. |
| Manifest-creation command | No manifest-creation command exists at `a75e543d2` for an admitted corpus: the admission contract and its exporter, `train prepare-inputs`, are pending integration ([§](#build-the-real-corpus-from-admitted-evidence)). |
| Execution | The command above. |
| Artifacts and success | `policy.pt`, `metrics.jsonl` and `progress.json` under `$NEW_RUN`. A completed run shows an optimization run only, not eligibility, optimizer continuation or improvement. |
| Failure and resume | The CLI exposes no `--resume-from`. A failed warm start is a failed run; repeating it is a new run in a new directory, without a renewed allowance. |
| Limitations | Shape-filtered or prefix-copied weights, defaults that do not reproduce the accepted model, and the device fallback in [§](#limits-of-the-current-python-route). |

#### Exact optimizer continuation through the Python API

For **actual optimizer continuation**, call the existing API from a private maintainer script. The following is the essential code; `settings.json` must contain the original compatible `kwargs`, and all referenced inputs must be hash-pinned in the operator's original run inventory. The complete companion `continue-python-policy.py` includes those byte-hash checks, exclusive output creation, explicit GPU preflight, refusal of distributed or non-current resume modes, and refusal of an `--output` under `/tmp` or inside a Git working tree (the path itself or any parent directory containing a `.git` entry). It is an illustrative API caller, not a new installed product command or a qualified supervisor.

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

#### The companion continuation helper and the resume artifact

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

| Runbook part | Python exact interrupted resume (companion helper) |
|---|---|
| Purpose and status | Continue an ordinary current-state Python run through the existing API from its epoch-end resume artifact. This is an exact interrupted resume only when the settings, corpus and resume artifact are the original run's and its original window is still open; the helper cannot check the window. **Implemented but unqualified**: only syntax/API and metadata-path checks were performed. |
| Prerequisites and resources | The trusted resume artifact, original settings, corpus and audited input closure, each with its expected SHA256; one visible RTX 3090; a separately admitted resource/time supervisor. |
| Manifest-creation command | No manifest-creation command exists: the operator writes `settings.json` in the illustrative `manual.python-resume-settings/v1` schema from the original run's configuration. |
| Execution | The helper without `--execute-diagnostic` (metadata only, no Torch import), then the same command with it inside the supervisor. |
| Artifacts and success | `policy.pt`, `resume.pt`, `metrics.jsonl`, `progress.json` and `api-result.json` in the new output directory. The result is scoped as a diagnostic API call, not product acceptance. |
| Failure and resume | Refuses changed input digests, missing or unknown settings, a resume artifact that is not an ordinary current-state resume with compatible optimizer state, more than one process, anything other than one visible RTX 3090, an existing output directory, and an output under `/tmp` or inside a Git working tree. A retry needs a new output directory and keeps the original window. |
| Limitations | No interrupted-versus-uninterrupted parity proof and no in-epoch cursor ([§](#rules-for-exact-interrupted-resume)). It fixes `amp=False` and compares only the four architecture fields with the artifact, so it does not detect a changed AMP or batch-size regime. |

#### Limits of the current Python route

This current Python route has material limits. CUDA requests can silently fall back to CPU in the underlying code, so the companion refuses absent CUDA before calling it. Current task-routing/zero-update and resume-proof fixes belong to the pending training overlay. The loader can discard minority embedding widths with a warning; a qualified input inventory should reject such a mixed corpus earlier. Existing Python training alone does not provide the shared cgroup resource contract, original campaign deadline, teaching admission, parameter-change proof or serving promotion. Execute any diagnostic caller only inside a separately admitted resource/time supervisor, not as a way to bypass the pending product gates. See [current device fallback](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/python/src/jope/policy/ddp.py#L25) and [mixed-width handling](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/python/src/jope/policy/train_v3.py#L1497).

## Evidence and identity before a bounded campaign

A training run has at least four separate identities:

| Identity | What it binds | Why it is separate |
|---|---|---|
| Learning state | Eligible HWM records and teaching evidence | Accumulated data authority is distinct from a single training attempt |
| Training plan | Task/category inputs, source, runtime, numerics, seed, passes and resource policy | Changing it invalidates an existing run's compatibility |
| Execution window | Actual first admission and immutable deadline | Resume must use elapsed budget rather than restart the clock |
| Checkpoint/export | Parameters, architecture, vocabulary, task, optimizer ancestry and completed proof | A file's existence is not evidence of an update or eligibility |

Use durable directories outside Git working trees and outside `/tmp`. Keep corpus and model assets out of Git. The native front door inventories the actual Python interpreter, preserves its virtual-environment invocation path, captures environment and installed runtime identity, and supervises workers rather than treating a bare Python subprocess as resource isolation.

The training runtime inventory hashes installed dependency bytes, excluding generated bytecode, and records all installed distributions rather than only a short library allowlist. Version strings alone do not protect against modified installed files. The full source commit and source-package hashes are part of compatibility. An incompatible environment should be rebuilt/replanned before admission, not silently accepted during resume.

Sources: pending `crates/jope-cli/src/train.rs:1`; `python/src/jope/policy/training_campaign.py:120`.

## The bounded training campaign chain

| Runbook part | Bounded training campaign chain |
|---|---|
| Purpose and status | Train, verify and export one bounded experimental policy from admitted evidence: experimental regression lineage v1 plus explicitly synthetic classifier fixtures. **Pending integration**: the training64 patch is outside main and has never run on a GPU. |
| Prerequisites and resources | Reviewed integration with required checks at the final head, a source-bound `jope` build, fresh real teacher admission covering at least two independent families, and an admitted training window. Resources: one RTX 3090 and one GPU-hour in total, at most 20 minutes per category, 20 CPUs, 64 GiB and zero swap. |
| Manifest-creation command | `train prepare-inputs` writes the inputs manifest (`inputs.json`) from a `jope.real-training-corpus-request/v1` request, and `train plan` freezes the plan without starting a clock. No command, at `a75e543d2` or in the pending overlay, writes the request itself; its comparison proofs must come from actual admission ([§](#build-the-real-corpus-from-admitted-evidence)). |
| Execution | `train run`, or `train resume` after an interruption inside the original windows ([§](#plan-run-resume-verify-and-export)). |
| Artifacts and success | Per-category attempts, receipts and heads in the state directory. `train verify` succeeds only with positive updates, changed parameters, exact reload, resumed optimizer state and at least two committed passes; `train export` then writes a private experimental bundle with a file inventory. |
| Failure and resume | An exhausted window is recorded as `budget_exhausted`. Resume keeps the plan, the state and the original global and category deadlines; an interrupted pass restarts from the last committed head ([§](#rules-for-exact-interrupted-resume)). |
| Limitations | Real corpus preparation supports regression only. With two prospective families it is a mechanics qualification, not a flywheel test. Input is JSONL without a packed reader, there are no sibling fresh/replay branches, and export never moves the serving pointer. |

### What the pending campaign trains

The pending training campaign uses the existing Python `JopePolicyModelV3` with task-routed reward/stability behavior. Real admitted training uses width 256, two layers, four heads and feed-forward 1024; synthetic qualification fixtures use width 8, one layer, two heads and feed-forward 16. Both consume the explicit context schema. Training uses one-epoch passes, learning rate 5e-4, batch 256 for admitted rows or 2 for fixtures, AMP disabled and automatic batch-size tuning disabled. These are the bounded campaign's settings, not the accepted step-60000 model's architecture (width 512, six layers, feed-forward 2048; [§](architecture.md#checkpoint-binding-and-configuration)). The campaign therefore produces a **new, smaller model family**, not a continuation of step 60,000: its optional initialization is a Python policy payload, and its checkpoint cannot be loaded in place of the accepted model, whose loader rejects incorrectly shaped tensors. Compare it with step 60,000 only through matched downstream fits under the same protocol, never through loss values.

**Naming.** The regression line that this campaign would start is called **experimental regression lineage v1 (256x2, new initialization)**. It is planned and has not been trained. Its initialization, M_init, is either a cold initialization or a weight warm start from a declared compatible Python payload. M_init is named M0 only after it has actually trained on the lineage's first admitted corpus, C0, and that run has passed verification; until then no M0 exists. The line stays separate from the accepted 512x6 step 60,000 of lineage B: it is neither its successor nor a replacement for it.

### Build the real corpus from admitted evidence

`train prepare-inputs` takes a request, not hand-written training examples. The request schema binds requested tasks, seed, family registry, vocabulary, encoder and optional initialization checkpoint. Its `contexts` map has one independently pinned proof for every eligible comparison. Each proof includes producer/source, original input, qualification/reference manifest, vector/artifact, frozen transforms, component bundle, actual context execution and eight case executions.

The exporter reopens the original exact fitting input, verifies feature schema/values/labels and row order, verifies complete context admission, and checks each recipe against exact executable vocabulary bins. It refuses parameter snapping, wrong-task heads and fabricated vectors. It then writes corpus rows with 1536 dataset plus 512 target dimensions, family-disjoint train/development metadata and retained source/context identities.

```sh
"$JOPE_FINAL_CLI" train prepare-inputs \
  --learning-state "$JOPE_LEARNING_STATE" \
  --request "$JOPE_CORPUS_REQUEST" --request-sha256 "$JOPE_CORPUS_REQUEST_SHA" \
  --output "$JOPE_CORPUS_OUTPUT"
```

Do not author a request using invented eligible comparison IDs: these must come from actual successful admission. The exported request requires all eligible requested comparisons. At least two independent families are necessary for a real train/development partition. Source: pending `crates/jope-cli/src/train/corpus.rs:43`, `:195`, `:362`, `:485`.

### Plan, run, resume, verify and export

The intended real all-category path preserves real admitted regression while allowing explicitly labeled synthetic fallback only for classification categories whose real admission fails:

```sh
"$JOPE_FINAL_CLI" train plan \
  --state "$JOPE_TRAINING_STATE" --python "$JOPE_TRAINING_PYTHON" \
  --inputs "$JOPE_CORPUS_OUTPUT/inputs.json" \
  --task all-admitted --device cuda:0 --source-sha "$JOPE_FINAL_SOURCE" \
  --learning-state "$JOPE_LEARNING_STATE" --seed 7 --passes 2

"$JOPE_FINAL_CLI" train run --state "$JOPE_TRAINING_STATE"
"$JOPE_FINAL_CLI" train verify --state "$JOPE_TRAINING_STATE"
"$JOPE_FINAL_CLI" train export \
  --state "$JOPE_TRAINING_STATE" --output "$JOPE_TRAINING_EXPORT"
```

If interrupted before the original deadlines and while all plan/runtime/input pins still agree, continue with:

```sh
"$JOPE_FINAL_CLI" train resume --state "$JOPE_TRAINING_STATE"
```

Do not use the top-level `--synthetic-fixture` switch in place of missing real regression evidence. For deliberately small mechanical CPU qualification only, the explicit interface is `train plan --synthetic-fixture --device cpu`, with the same source/runtime/state requirements. Such a result stays synthetic and does not satisfy the real-data task.

The planner records requested, admitted, unavailable and fallback categories. With `all-admitted`, a missing real regression category can remain explicitly unavailable while classifier fixtures proceed; that is partial coverage and cannot be reported as completed real-regression training. Current real-classification corpus admission is not qualified. Binary/multiclass fixture fallback retains its own synthetic provenance and task labels.

Verification must establish positive optimizer updates, changed parameter digests, exact reload, an actual resumed optimizer state and increasing step counters, at least two committed passes, complete planned category coverage where claimed, and compatible device/source/runtime/vocabulary/input identities. The worker reloads both `policy.pt` and `resume.pt` and requires equal parameter proof. Each pass belongs to the original category/global windows and contributes an immutable execution/receipt chain. A zero-update checkpoint or interrupted partial output cannot become a successful pass. Both planned passes read one corpus inside one category window, so the second pass is continuation within one generation, not a second LOOP-01 generation. It supplies the optimizer-continuation proof: verification refuses a category head with fewer than two committed passes (pending `training_campaign.py:1210`). Verification establishes training mechanics only. An eligible teacher, a new discovery, information actually consumed by the optimizer and an improved policy remain four distinct claims, each needing its own evidence.

Export first performs verification, then writes the exact plan, verification and category artifacts into a private output bundle with a file inventory. It creates experimental training artifacts; it does not update the accepted serving pointer. Sources: pending `python/src/jope/policy/training_campaign.py:744`, `:780`, `:1043`, `:1203`, `:1303`.

### The pending bounded extension path: sequence and limits

The stopped training64 patch is `595824ef6eb8e15872be058668503cedad1dd9a5ff98ac4700c6f713251d384c`, with manifest `b86b550d7ee39fd93d5202accd5f77c975b7bcb4fd8f8e7706c69c5f279294a2`. It has not been integrated into main (`a75e543d2`, PR83) or qualified by a fresh real GPU run. Its exact command contract is described in [§](#plan-run-resume-verify-and-export); pending files cannot honestly be linked as existing GitHub files at main.

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

Run and resume are alternatives according to state; do not blindly execute resume after a completed run. The whole campaign permits one RTX 3090 and one GPU-hour, at most 20 minutes per category, 20 CPUs, 64 GiB and zero swap. Two passes share each category window; they do not receive fresh time allowances. At least two real admitted families are necessary before claiming the real regression training path; with only the two prospective panel families of [§](teachers.md#present-teacher-readiness-and-the-smallest-real-path), the campaign is a mechanics qualification, not a flywheel test. Classification fixtures may demonstrate labeled mechanics when real admission fails, but remain synthetic. Unknown or unavailable categories are reported rather than filled with invented success.

Verification must establish positive updates, changed parameters, reload equality, actual optimizer continuation, full input/runtime/source/device/vocabulary compatibility and persisted accounting. Export is experimental. The pending `experimental.regression`, `experimental.binary` and `experimental.multiclass` consumer entries bind task-specific exported checkpoints; a later real fit must prove the new checkpoint generated and fitted candidates. None of these commands updates the accepted serving pointer. Product28, warm-target and winner-export follow training64; they do not fix teacher eligibility by themselves.

### Proposed sibling fresh and replay branches

**Status: proposed.** No implementation exists: the pending campaign advances one mutable head per category over one pinned corpus. It has no immutable sibling parent, branch identifier, committed-update target, 25/75 sampler allocation or per-update fresh-exposure record (pending `training_campaign.py:799`, `:1065`). The owner's design for one controlled round is:

- Two sibling branches, F (fresh) and R (replay only), start from one immutable parent model **and** optimizer state and share the original deadline; neither receives a new window.
- Both commit the same number of updates and batch slots. F samples 25% fresh and 75% replay examples; R samples replay only.
- Each records measured fresh exposure, duplicate and unique families, consumed example identities, non-padding tokens and GPU cost. The unchanged parent is kept as a comparison artifact.
- Moving onto the new corpus is an explicit new-corpus transition that declares optimizer carry and sampler initialization; it is not a resume.
- Neither branch is promoted to serving.

The first causal experiment keeps architecture, objective, corpus and sampler changes distinguishable, so it does not change the architecture or the reward representation alongside these branches. Within the existing 20-minute regression GPU cap, the two controlled LOOP-01 rounds may remain unfunded or inconclusive. An unfunded round, or one with insufficient data, yields mechanical proof and descriptive results, and its causal outcome is reported as inconclusive, not as success. [§](learning.md#the-closure-ledger) defines LOOP-01.

### Training input packing and the learning store

**Status: proposed.** The pending campaign still reads JSONL into memory and writes a per-attempt `corpus.jsonl`; no packed-reader scalability qualification exists ([§](teachers.md#python-training-input-and-the-pending-bounded-campaign)). Packing work follows phase profiling, in this order:

1. Profile the preparation phases, as in [§](teachers.md#concrete-corpus-improvements-with-acceptance-criteria).
2. Add a verified Python packed-reader bridge.
3. Reuse immutable shards, with bounded staging.
4. Compress.
5. Adopt a narrower record layout only if measurement justifies it.

Every step preserves float32 supervision, the legal-set mappings and the 128-token grammar, and must show record, batch, loss, gradient, update and resume parity ([§](data-estate.md#compression-the-proposed-180-byte-layout-and-label-precision)). Keep one learning-store writer until an atomic server-side pointer update is qualified. Backup, replication and retention operations remain separately authorized; nothing in this chapter authorizes them ([§](data-estate.md#recorded-custody-risks-and-disk-capacity)).

## Consuming a newly trained checkpoint

### The experimental.regression configuration entry

The pending final regression consumer is selected by the `experimental.regression` entry of the complete `jope.asm-configuration/v1` object. This entry is an object with:

| Field | Required meaning |
|---|---|
| `schema` | `jope.experimental-python-configuration/v1` |
| `task` | `regression`, `binary`, or `multiclass`, matching its configuration key |
| `runtime` | Full pinned PythonRuntime object, including program, manifest, source/package files and runtime digest |
| `export_manifest` | Exact exported training manifest artifact |
| `checkpoint` | Exact task's exported `policy.pt` artifact |
| `vocabulary` | Exact frozen action vocabulary artifact |
| `canonicalizer` / `canonicalizer_build` | Native executable and actual successful build evidence |
| `canonicalizer_features` | Exact feature set from Cargo artifact messages |
| `sampling` | Bounded count, temperature, top-p and maximum sequence length |
| `generic_draws` | Declared bounded generic bank allowance |

Artifacts use `{path, sha256}` with absolute paths and real file hashes. Runtime is not merely a Python executable/version: source files and installed package files must be inventoried. The native canonicalizer must have a successful build receipt whose source and source-manifest pins agree with the consumer. Do not copy an accepted checkpoint receipt into this export field.

### Compatibility checks and fail-closed rules

Regression additionally requires the export to identify real admitted regression training, the same source commit, and the exact native encoder identity. The consumer admits the complete native context and independently verifies architecture, task, vocabulary, checkpoint and proof. Classification uses its separate context lineage. Missing architecture fields, missing/extra model state, altered vocabulary, source changes, changed encoder assets or a different training task fail closed. No explicit experimental selection silently falls back to the accepted model.

The export records the complete architecture, vocabulary, task, source, input identities, model state and training proof. The pending experimental consumer requires all architecture fields explicitly; the general-purpose Python loader's backward-compatible defaults cannot reinterpret a saved experimental checkpoint. It compares the reconstructed model's architecture and full state to the export. Sources: pending `python/src/jope/policy/training_campaign.py:780`, `python/src/jope/policy/experimental_proposal.py`, `crates/jope-pipeline/src/experimental_producer.rs`.

The final regression follow-on closes a specific gap: `Configuration.experimental.regression` can explicitly select a newly trained regression export. Omission keeps the accepted baseline. This operational regression route requires **real admitted** training, the same source commit and the exact native encoder identity used by training. It first admits the qualified native2048 regression context, then wraps those exact values in the separate `asm.experimental-regression-context/v1` lineage with zero classes and `policy_compatible:false`. It does not promote the new checkpoint to accepted serving or weaken the accepted hard pins.

Binary and multiclass use their separate experimental context path; a synthetic classifier fixture can qualify mechanical training/consumption behavior only when explicitly reported as synthetic. Synthetic regression training is permitted as a CPU test but cannot be substituted into the operational real-regression consumer. There is no silent accepted-policy fallback after an explicitly chosen experimental checkpoint fails compatibility.

### Proving consumption in a new operational proof

After building a new configuration around the verified export, preregister a separately authorized operational proof using the same `optimize proof-plan` front door. Its report must bind the new checkpoint hash to actual generation-zero neural fits, retain fitted artifacts/predictions and, when evolution is claimed, show completed descendants with neural ancestry. Recover the report from the same original journal and deadline. Only after those facts are demonstrated should the final official panel configuration be frozen around that checkpoint.

Source: pending `crates/jope-bench/src/panel/experimental_proposal.rs:36`, `crates/jope-pipeline/src/experimental_producer.rs:36`, `python/src/jope/policy/experimental_proposal.py`; overlay documentation `docs/experimental-regression-consumption.md`.

| Runbook part | Consumption of a newly trained checkpoint |
|---|---|
| Purpose and status | Show that a verified experimental export generated and fitted real candidates. **Pending integration**: the `experimental.regression` consumer exists only in the stopped overlay. |
| Prerequisites and resources | A verified export, a complete configuration whose `experimental.regression` entry pins it ([§](#the-experimentalregression-configuration-entry)), and a separately authorized operational proof with its own resources. |
| Manifest-creation command | `optimize proof-plan` preregisters the proof; it exists at `a75e543d2`, without the experimental consumer. No command writes the configuration entry: in the pending source the schema string occurs only as the consumer's expected value, and the overlay documentation has the operator bind it. |
| Execution | The same front door's run and report steps, recovered from the original journal and deadline. |
| Artifacts and success | A report that binds the checkpoint hash to actual generation-zero neural fits, with retained fitted artifacts and predictions, and completed descendants with neural ancestry when evolution is claimed. |
| Failure and resume | Compatibility failures fail closed, with no fallback to the accepted model; recovery uses the original journal and deadline. |
| Limitations | Consumption is not improvement, and nothing here moves the serving pointer. |

## Remaining acceptance queue and maintenance risks

The required next integration/acceptance work is sequential: apply guarded training64 after PR83 lifecycle completion; compile and run all required tests at the final head; admit fresh genuine teachers; materialize and verify the corpus; execute the one approved GPU hour; verify/export; consume the new checkpoint in actual fits; qualify fitted portability and warm/cold delivery; execute the unchanged30 panel and report all outcomes. The product chapters cover the remaining deployment stages in detail.

LOOP-01 is the central regression product milestone ([§](learning.md#the-closure-ledger)): a new eligible target, actual exposure, changed parameters, compatible reload, improved fitted search, then another new generation. The training chain in this chapter can supply exposure, changed parameters and compatible reload. Improved fitted search and a further generation also need the consumption proof above and a preregistered decision. Under the owner's proposed rule, a round stops on no eligible discoveries, incompatibility, no material gain, insufficient evidence or an exhausted budget; the best verified checkpoint is preserved, and serving is never promoted automatically.

Maintain explicit checks for changed dependency bytes, an unsupported GPU/device, absent families, zero updates, wrong task heads, changed vocabulary bins, missing continuation, missing native component receipts, stale code hashes, counterfeit fitted outputs, incomplete process-group cleanup and partially written journals. The experimental route is intentionally separate from accepted serving. Any future promotion mechanism must add its own independent evaluation and pointer discipline rather than rename an experimental export.

The current source-only regression follow-on has an independent review with no confirmed production/API blocker. Seven pure Python checks passed; the three actual-model regression/binary/multiclass cases were explicitly deselected because custody work was barred from model execution. The next required CI/runtime acceptance must execute those cases and verify their collected inventory. This chapter records the gap rather than claiming it closed.
