# Matched-budget arena and KPI harness

The additive `rlfw.arena` package freezes a game before execution, precommits all
arm slates for each context, and persists score disclosure and slot charging in an
isolated SQLite outbox. The comparator is the original program re-executed under
the pinned protocol. Uniform draws are uniform over whole programs; policy draws
use the deployed trie sampler with replacement. Duplicates, cache hits, program
failures and timeouts consume slots. Infrastructure and parity unknowns get two
retries and then remain nonnumeric, censored slots.
Attempt reservations are durable: an interrupted attempt consumes its retry
allowance, and a resumed slot cannot start a fourth attempt. Unobserved costs
stay explicitly unknown; only observed cost lower bounds are retained.

Scores require verified native execution, emitted Rust parity, artifact closure
and a receipt. The deployed integer `q_grid` is converted to q-units by dividing
by 1,000,000. Parity uses the pinned normalized error criterion
`abs(actual-reference)/max(1,abs(reference)) <= 1e-9`; raw absolute error is a
separate diagnostic. The evaluator is injected from the byte-pinned campaign
adapter; this package contains no evaluator implementation or service imports.

Every reported performance number carries this claim label:
**Lane A, task-label-aware inner-CV search, dev panel L=54, directional**.
The report also records the actually observed lineages and partial-panel status.
It includes lineage-macro win/tie/loss, strict win, observed empty validity,
conditional regret and positive regret, best-q and validity at 1/2/4/8, validity
and cost statistics, paired differences, headroom, tau sensitivity, and separate
qualified archive/fresh ranking diagnostics. Censored units are excluded from
point win rates; bounds assign every censored unit LOSS/WIN. Conditional best-q
and regret use measured valid results and retain the censoring rate beside them.
Parent-cluster bootstrap uses 2,000 draws and seed 7; siblings are never clusters.

## Acceptance and evidence

From the reviewed checkout, run:

```sh
export TMPDIR="$HOME/.cache/jope-test-tmp/rlfw-l2"
mkdir -p "$TMPDIR"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=deploy/dl-flywheel/pipelines \
  nice -n 19 ionice -c3 python3 -B -m unittest \
  rlfw.tests.test_arena_kpi rlfw.tests.test_arena_scheduler -v
```

The registered `run_rlfw_offline.py` runner and main's required gate/CI are the
landing checks. The four legacy runners named in the input plan are
`unavailable-on-main`; they are not reported as passing. Independent LR review
and exact-head CI remain required. Fixture logs are in the campaign's
`logs/l2/arena-*.log`; they establish mechanics, not a live baseline or model lift.
The 500 simulated panels test bootstrap coverage; another fixture duplicates one
lineage's siblings 1,000 times and requires the same interval.

## Runtime adapter and qualification

`build_runtime(spec, *, device="cpu")` returns `catalog`, `hwm_programs`, `sample`, `evaluate`,
`role_check`, and `frozen_cache`. Programs contain exactly `dsl`,
`program_sha256`, and complete BOS/EOS `tokens`. The sampler returns the exact
program, tokens, summed/per-token trie likelihood, checkpoint parameter SHA and
support SHA. The evaluator receives a score-free request and returns nullable
integer `q_grid`, outcome class, measured wall/CPU seconds, context/evaluator/
canonicalizer/protocol/role pins, and qualified closure/parity receipt fields.
Completed results require `closure_verified`, `parity_verified`,
`parity_max_normalized_error` and `receipt_sha256`. The fresh role callback must
return exactly the frozen `development` role and role-map SHA.
Requests include a `logical_slot_id` distinct from the exact execution key;
use it with the persisted attempt number for create-only evaluation destinations.
This avoids collisions when unknown outcomes require another logical disclosure
of the same program. A persistent worker adapter supplies `close()` to release
only its own pool. Sampler batch-protocol identity is retained in the PR1 trace,
and measured slate-preparation wall time is reported separately from slots.

The adapter and its execution helpers are individually hashed in
`GameSpec.runtime_files`. Their bytes are checked before every attempt and on
resume. The exact-result cache is an immutable mapping whose canonical digest
is frozen in the game. A new game uses its own outbox. Scores from the running
service, unqualified archives and predicted heads are never comparator rewards.

The CLI accepts `--device cpu|cuda`, default `cpu`; it must match the frozen
`GameSpec.device`. CUDA needs `--device-parity FILE`, whose SHA is frozen as
`device_parity_sha256` and included in `runtime_files`. Use PR1
`verify_checkpoint --device cpu --compare-device cuda` on identical complete
sequences. Every checkpoint needs passed CPU/CUDA parity for total/per-token
likelihood, q, completion probability and log1p runtime at `1e-6`. The
byte-bound file is either that PR1 receipt or `{"receipts": [receipt, ...]}`.
Unavailable CUDA or incomplete parity fails. D3 authorizes deterministic CPU
inference for this campaign; CUDA is unqualified under the unchanged tolerance.
Every executed game still needs its actual runtime qualification.

The policy qualification facade, itself byte-pinned by the game, is:

```json
{
  "passed": true,
  "draws": 1000,
  "determinism_passed": true,
  "maximum_logprob_error": 0.000001,
  "checkpoint_parameter_sha256s": ["FULL_QUALIFIED_PARAMETER_SHA"]
}
```

The facade must reference L1's real sealed-checkpoint replay evidence; it cannot
replace that evidence. Every selected checkpoint needs its own qualification.

## L0 baseline commands after reviewed merge

Set `BENCH` to the campaign directory and `DEVICE=cpu` under D3 on the verified xbabe1
CPU replica. Run from the reviewed main checkout
with `PYTHONPATH=deploy/dl-flywheel/pipelines`. After L0 freezes the adapter and
L1 provides `cache/policy-qualification.json`, prepare score-free specifications:

```sh
export PYTHONPATH=deploy/dl-flywheel/pipelines
python3 -B - <<'PY'
from collections import defaultdict
import hashlib, json, os, pathlib, random
from rlfw.arena.game_spec import ArmSpec, ContextSpec, GameSpec, canonical, digest

b = pathlib.Path(os.environ["BENCH"]).resolve()
device = os.environ.get("DEVICE", "cpu")
parity = b / "cache/device-parity.json" if device == "cuda" else None
def pin(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
prepared = json.loads((b / "cache/arena-contexts.json").read_text())
keys = ("context_id", "lineage_id", "role", "identity_sha256", "hwm_program_sha256")
contexts = tuple(ContextSpec(**{k: row[k] for k in keys}) for row in prepared)
checkpoint = b / "ckpt/current_latest"
manifest = json.loads((checkpoint / "manifest.json").read_text())
trie = json.loads((checkpoint / "trie-identity.json").read_text())
normalizer = pin(checkpoint / "preprocessing.json")
common = (trie["sha256"], normalizer, trie["vocabulary_sha256"])
arms = (ArmSpec("hwm", None, *common), ArmSpec("uniform_trie", None, *common),
        ArmSpec("trie_sample:current_latest", manifest["parameter_sha256"], *common))
dependencies = ("l0/arena_binding.py", "l0/execute_one.py", "l0/bench_binding.py",
                "evaluator_pin/MANIFEST.sha256", "evaluator_pin/config.json",
                "cache/INPUT_MANIFEST.sha256", "l0/cpu_policy.py",
                "cache/dev-normalized.npy", "cache/dev-normalized.json",
                "cache/arena-core54.json", "cache/dev-contexts.json",
                "cache/arena-contexts.json", "cache/dev-hwm-programs.json",
                "l0/CAMPAIGN_D3.json", "prompts/D3_GPU_PARITY_CPU_INFERENCE.md",
                "cache/policy-qualification.json", "replica-inputs/MAP_D1v2.json")
runtime_files = tuple((name, pin(b / name)) for name in dependencies)
if parity is not None:
    runtime_files += ((str(parity.relative_to(b)), pin(parity)),)
grouped = defaultdict(list)
for c in contexts:
    grouped[c.lineage_id].append(c)
rng = random.Random(20261009)
eligible = sorted(k for k, v in grouped.items() if len(v) >= 2)
rng.shuffle(eligible)
smoke = tuple(c for lineage in eligible[:10]
              for c in rng.sample(sorted(grouped[lineage], key=lambda c: c.context_id), 2))
assert len(smoke) == 20
core_rows = json.loads((b / "cache/arena-core54.json").read_text())["contexts"]
core = tuple(ContextSpec(**{k: row[k] for k in keys}) for row in core_rows)
assert len(core) == 54 and len({c.lineage_id for c in core}) == 54
assert all(c in contexts for c in core)
(b / "state/game_specs").mkdir(exist_ok=True)
for name, roster in (("baseline_T0", smoke), ("baseline_T1", core)):
    spec = GameSpec(name, str(b), str(b / "runs" / name),
        str(b / "state" / (name + "-outbox.sqlite")), roster, arms, (1,),
        "3b9caf36a3cde3f12d0337ebfc633b55a1e989a3f53410df578f6fc9a7aa0109",
        digest({}), "e3be938644722d14a1c1a0e82168518838c4d174cfe031076a339745c48870a8",
        "69975947982bcffb0a55fde96f85b68f043e107e0d946eadffcdac7311f026c1",
        slate_k=32, runtime_binding_sha256=pin(b / "l0/arena_binding.py"),
        policy_code_manifest_sha256=pin(b / "code_pin/MANIFEST.sha256"),
        policy_qualification_sha256=pin(b / "cache/policy-qualification.json"),
        deadline_unix=1791550800.0, runtime_files=runtime_files, device=device,
        device_parity_sha256=pin(parity) if parity else None)
    spec.freeze()
    with (b / "state/game_specs" / (name + ".json")).open("x") as f:
        f.write(canonical(spec.to_dict()) + "\n")
    print(name, spec.sha256)
PY
```

L0 starts these commands only after its health baseline qualifies and its U1
pool scope is no longer active. Under D1, while the protected service is
owner-stopped, its rate guard is N/A and both-host memory guards still apply.
The example uses the byte-verified xbabe1 CPU replica; L0 owns placement and
cross-host evaluator parity. The unit belongs to L0's existing kill switch:

```sh
systemd-run --user --scope --unit=rlfw-l0-bench-pool \
  -p CPUQuota=400% -p MemoryMax=6G -p MemorySwapMax=0 -p TasksMax=96 \
  env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  nice -n 10 ionice -c3 python3 -B -m rlfw.bench_cli arena \
  --spec "$BENCH/state/game_specs/baseline_T0.json" \
  --runtime-binding "$BENCH/l0/arena_binding.py" \
  --policy-code-root "$BENCH/code_pin" \
  --policy-qualification "$BENCH/cache/policy-qualification.json" \
  --device cpu \
  --pool-workers 4 --checkpoint-report-every 20
```

For the core, use the same command with `baseline_T1.json`. D3's score-free
roster contains one original context in each of all 54 lineages. The other 209
contexts remain unscored. A whole-panel extension requires a separately frozen
roster before execution; it cannot change the core game's identity on resume.
Label these results **dev panel, directional; CPU inference**.
The source-host 1,000-draw proof used Torch 2.14; the replica uses Torch 2.7.
Retain that distinction and qualify the actual runtime before a live baseline.
The current native parity STOP prevents game launch; D3 changes device placement
and does not clear that later STOP.
Parent sampling can be preregistered as an additional arm only after that
checkpoint's qualification, before any score is exposed.

Each run writes `game_spec.json`, `decisions.jsonl` and its PR1 manifest,
`slots.json`, `report.json`, `report.md`, `per_context.csv`, and
`outputs.manifest.json`. An interim trace remains open until all registered
contexts close. The output manifest detects changed bytes before comparison.
The size/coverage flag alone does not establish a valid baseline; retain G0/G1
and deterministic replay receipts separately.

Compare two verified runs using explicitly selected arms:

```sh
python3 -B -m rlfw.bench_cli compare \
  --a "$BENCH/runs/baseline_T1" --a-arm trie_sample:current_latest \
  --b "$BENCH/runs/baseline_T1" --b-arm uniform_trie \
  --out "$BENCH/runs/compare_sample_vs_uniform.json"
```

Paired runs must share the exact roster/order, seeds, budget, margin, cache,
support, evaluator, protocol and qualified incumbent values. The comparison
also binds device placement. Each checkpoint's sampler/device qualification is
bound to its full game SHA; those arm-specific certificate hashes are excluded
from the shared pairing identity. Execution helper and code pins remain shared.
The comparison
refuses missing counterparts and writes create-only. A later challenger can
reuse comparator receipts with `--reuse-hwm-from` only after identical family
identity and full original evidence validation. Promotion and teacher changes
are outside this command.
