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

Set `BENCH` to the campaign directory. D3 authorizes CPU inference; the source
route has the full 1,000-draw Torch 2.14.0+cu130 certificate. The replica route must have
its own full certificate matching its actual Torch version. The two routes use
different wrappers and cannot share a mismatched certificate. All native parity
and campaign STOP guards remain active.

L0's sealed `l0/CPU_ARENA_PREPARATION_V5.json` records the helper hashes, including
the corrected observed-slot denominator. Read the latest `INTEGRATION_DELTA.md`
before freezing a game. Older preparation manifests describe historical bytes.
After reviewed PR1/PR2 merge, L0 creates an immutable merged-package copy, then
freezes both score-free specifications using the explicit CPU route:

```sh
CPU_PYTHON=/home/ubuntu/JopeDime/.agent/worktrees/bt-implementation/python/.venv/bin/python
"$CPU_PYTHON" -B "$BENCH/l0/pin_merged_package.py" --commit "$MERGED_FULL_SHA"
PACKAGE_ROOT="$BENCH/merged/$MERGED_FULL_SHA"
QUALIFICATION_FILE="$BENCH/runs/l0-source-cpu-qualification-v5/qualification.json"
QUALIFICATION_SHA256=0af037cc256e1ad8c4a2ff888768313145c23f582d1cdc30b08bb6350da2db15
"$CPU_PYTHON" -B "$BENCH/l0/prepare_games.py" \
  --package-root "$PACKAGE_ROOT" --cpu-route source \
  --qualification "$QUALIFICATION_FILE" \
  --qualification-sha256 "$QUALIFICATION_SHA256"
```

`MERGED_FULL_SHA` must be verified in fetched `origin/main`; `QUALIFICATION_FILE`
is L0's immutable facade for L1's complete CPU receipt, and
`QUALIFICATION_SHA256` is its pinned byte hash. The source interpreter must report
Torch 2.14.0+cu130, exactly matching that certificate; the default system Python
and a CPU-only 2.14.0 wheel do not satisfy this identity. Preparation verifies those identities and pins the chosen
route, shared helpers, original inputs, normalization, code and qualification.
It refuses an existing specification with different bytes. No private candidate
package can become an official game through this command.

After owner resolution of the current native parity STOP, L0 verifies its health
baseline and starts the source route on xbabe2 with the original qualified
evaluator references. This launcher uses the immutable package, four workers,
400% CPU, 6 GiB memory, zero swap and nice 10 under L0's existing kill switch:

```sh
bash "$BENCH/l0/run_game_source_cpu.sh" "$PACKAGE_ROOT" baseline_T0
bash "$BENCH/l0/run_game_source_cpu.sh" "$PACKAGE_ROOT" baseline_T1
```

For a separately qualified replica, freeze with `--cpu-route replica` and use
L0's `run_game.sh` on xbabe1. It selects `arena_binding_replica_cpu.py`; the
source launcher selects `arena_binding_source_cpu.py`. Baseline and any paired
challenger must retain the same route and complete runtime family. The replica
must never consume the source's Torch 2.14 certificate as Torch 2.7 evidence.
No replica full-qualification or live launch is claimed by these instructions.

T0 has 20 contexts across 10 lineages. T1 uses the frozen score-free core with
one original context in each of all 54 lineages. Its other 209 development
contexts remain unscored. A whole-panel extension needs a separately frozen
roster before execution; it cannot change the core identity on resume. Label
results **dev panel, directional; CPU inference**. A parent checkpoint can be
preregistered only after its own qualification, before score exposure.

The protected service's rate guard is N/A while the owner has stopped it; fresh
rate and both-host memory guards apply when it is running. The current native
parity STOP prevents game launch. D3 changes device placement and does not
clear that later STOP. Preparation and contract tests are not live acceptance.

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
