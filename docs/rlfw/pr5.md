# Train-only measured targets and offline challenger

`rlfw.exit_targets` builds conservative targets only after fresh train-role and
native/Rust artifact-closure checks. Development and sealed rows fail before
eligibility filtering. Unqualified outcomes never supply a numeric reward.
The set form deduplicates programs and weights the measured raw q tilt by the
frozen anchor probability. Complete, unfiltered raw iid groups preserve repeated
draws and omit that probability factor. Both forms require two distinct supported
programs and two distinct measured grids; other contexts contribute zero actor
loss. Targets retain receipt identities, certified anchor support mass and
EOS-inclusive conditional prefix masses. Each supported token sequence must
match the trie catalog's canonical program bytes and SHA; its native/Rust proof
binds that same program SHA and lineage.

The guarantee concerns only the measured partial support. Distillation into a
shared network is approximate, and clipping changes the operator. Executed
matched games remain the acceptance test.

`rlfw.exit_objective.objective` returns the original six-term L6 result object
immediately when both new coefficients are zero. Policy terms use dropout-off,
float64 trie-child likelihood, summed through EOS, with numerical checks against
PR1. The training KL term uses fresh, unselected current-policy draws and an
unbiased score-function gradient; its logged value is a sample estimate. The
64-context qualification probe enumerates the full trie using the exact chain
rule. A timeout is an unavailable probe. It never becomes a passing approximation.
Physical equality of policy, anchor, support and device proves initial KL zero.

The explicit `rlfw.exit_offline_train` CLI defaults to CPU and coefficients zero.
It copies the sealed checkpoint, creates a new optimizer with the pinned L6
parameter grouping, and writes only under a new `BENCH/ckpt/challenger_*` run.
It does not resume the service optimizer. Limits are at most 500 updates,
learning rate at most `1e-4`, checkpoints every 100 updates and the unchanged
campaign deadline. An exact probe KL of `0.05` or more disqualifies the checkpoint.
Checkpoints store numeric parameters, optimizer moments and RNG state with the
unchanged support files. They do not carry the service's old control record.

Before each step the trainer checks every batch role again. It records a durable
reservation before the optimizer call and completed parameter hashes afterward.
An interrupted optimizer call stays explicitly unknown. Existing runs are never
automatically resumed. Neither a checkpoint nor a successful probe is a lift or
promotion claim.

## Runtime contract

The byte-pinned benchmark adapter supplies
`build_training_runtime(config, *, policy, anchor)`. It returns:

- `target_rows`, `verify_receipt(row)` and `anchor_logprob(context_id, tokens)`
  for a fresh target rebuild using the exact pinned artifact verifier and PR1;
- `role_check(context_id)` returning the current train role and role-map SHA;
- `contexts`, a train-only mapping of IDs to pinned normalized vectors;
- `probe_contexts`, exactly 64 frozen, distinct train context IDs;
- `next_batch(update)`, a complete context batch using the existing mixed L6
  supervision and sparse support masks;
- `base_objective`, the actual pinned `neural_native_batch.objective` function.

The CLI checks the L6 function's import origin and identity. It owns the optimizer,
parameter hashing and checkpoint writer. The adapter never takes an optimizer,
writes a checkpoint or requests a service action. Its module, all helpers, policy
manifest and target file are individually pinned in the runtime-file inventory.
Fresh verifier output must bind context, lineage, program SHA, comparison identity, exact raw/clipped/
grid quality, receipt SHA, closure and the unchanged `1e-9` parity rule.

The target builder is a Python API. An adapter with verified rows and the pinned
anchor can call it before training:

```python
from rlfw.exit_targets import TargetConfig, build_targets

targets = build_targets(
    qualified_train_rows, TargetConfig(**frozen_target_config),
    trie=anchor.trie, role_check=fresh_train_authority,
    verify_receipt=pinned_receipt_verifier,
    anchor_logprob=lambda cid, tokens:
        anchor.logprobs_many([normalized_train_vectors[cid]], [tokens])[0].total,
)
```

Persist the resulting JSON create-only under `BENCH/runs` and pin its bytes.
Native archive labels lacking Rust closure cannot construct these targets.

## Commands and evidence

From the reviewed checkout:

```sh
export TMPDIR="$HOME/.cache/jope-test-tmp/rlfw-l2"
mkdir -p "$TMPDIR"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=deploy/dl-flywheel/pipelines \
  nice -n 19 ionice -c3 python3 -B -m unittest \
  rlfw.tests.test_exit_targets rlfw.tests.test_exit_training -v
```

The fixture log `logs/l2/exit-trainer-identity-final.log` records 27 passing tests with no
skips. It includes measured tilt derivative/variance, set versus iid weighting,
exact full-trie KL and gradients, KL-estimator gradient, default-zero equality,
dev refusal, two synthetic optimizer updates with an unchanged anchor, the
threshold refusal, fixed deadline, path isolation, real numeric checkpoint serialization and unknown interrupted step.
It reproduces and refuses the draft program/token mismatch while accepting the
aligned canonical mapping and retaining zero support for an unsupported program.
These are data-free mechanics; synthetic updates are not sealed-model training.

`logs/l2/pinned-l6-parity.json` records bitwise-equal loss and gradients against
the actual pinned L6 implementation on a fixed synthetic batch supporting all
six terms. It binds the pinned function, code manifest, extension and proof
script bytes. The proof runs no model, optimizer or evaluator. The four absent
legacy suites remain unavailable-on-main.

After a qualified train corpus, reviewed adapter and targets are available,
compute the source bundle identity in the same checkout:

```sh
export PYTHONPATH=deploy/dl-flywheel/pipelines
SOURCE_SHA=$(python3 -B - <<'PY'
import hashlib, pathlib
from rlfw.arena.game_spec import digest
import rlfw.exit_offline_train as training
root = pathlib.Path(training.__file__).parent
print(digest({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(root.rglob("*.py"))}))
PY
)
TARGET_SHA=$(python3 -B -c 'import json,os; print(json.load(open(os.environ["BENCH"]+"/runs/exit_targets.json"))["target_sha256"])')
systemd-run --user --scope --unit=rlfw-l2-shadow \
  -p CPUQuota=400% -p MemoryMax=6G -p MemorySwapMax=0 -p TasksMax=96 \
  env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 nice -n 10 ionice -c3 \
  python3 -B -m rlfw.exit_offline_train \
  --bench-root "$BENCH" --out "$BENCH/ckpt/challenger_exit_v1" \
  --policy-code-root "$BENCH/code_pin" \
  --checkpoint-dir "$BENCH/ckpt/current_latest" \
  --targets "$BENCH/runs/exit_targets.json" --target-sha256 "$TARGET_SHA" \
  --runtime-binding "$BENCH/l2/training_binding.py" \
  --runtime-files "$BENCH/cache/training-runtime-files.json" \
  --source-sha256 "$SOURCE_SHA" --deadline-unix 1791550800 \
  --device cpu --max-updates 500 --learning-rate 0.0001 \
  --lambda-pi 0.25 --lambda-kl 0.02
```

The CPU default supports fixtures and explicitly authorized CPU runs. The current
campaign's device placement still follows the owner's ruling. CUDA needs a
byte-bound passed same-pin CPU/CUDA receipt and `--device-parity FILE`; absence
or failed parity refuses execution. The command does not imply that those live
inputs or CUDA qualification already exist. Actual challenger execution remains
unavailable until qualified train targets and the runtime adapter are provided.

A trained checkpoint still needs its own sampler/device qualification and a
paired arena against the baseline under [PR2](pr2.md). Promotion is a separate
research gate; this command changes no champion, teacher or service pointer.
