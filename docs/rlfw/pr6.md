# Research checkpoint gate and teacher proposals

`rlfw.promotion_gate` reads matched, hash-verified arena outputs. It derives
utility from qualified slot rows and re-scored originals, rather than accepting
the report's summary values. The checkpoint arms must share the frozen panel,
seeds, support, preprocessing, budget, cache and evaluator conditions. Different
quality for one measured program/comparison, or different measurements under
one receipt SHA, refuses the input.

Promotion requires all 54 registered lineages and complete paired units. A
censored result, unknown evaluator cost or unqualified original produces an
`INCONCLUSIVE` decision. An incomplete run also stays inconclusive. Reduced
context rosters can qualify only when they were frozen before execution and
still contain all 54 lineages.

The module computes parent-macro paired positive gain at eight slots, with
2,000 parent-cluster bootstrap draws and seed 7. Both its ordinary 95% lower
bound and the look's stricter spent-alpha lower bound must exceed zero. The
paired win-rate bounds must be nonnegative. Validity at eight slots and qualified
completion may each drop at most two points. Median evaluator time may rise at
most 5% unless positive gain per second improves. Evaluator time is averaged
over the registered seeds within each context before taking the context median.

The six monthly look budgets use an O'Brien–Fleming-style cumulative spending
function, with total alpha 0.05. All games share one monthly ledger. Every look
is reserved durably before computing a result; an interruption consumes that
look. The first two look tails are smaller than 2,000 bootstrap draws can
resolve, so they are inconclusive. The code does not extrapolate those tails.

Reservations, decisions and pointer applications have immutable SHA-256
integrity receipts. This integrity hash does not authenticate a signer. Receipts
bind panel and game hashes, checkpoint identities, source bundle, bootstrap
seed, per-lineage differences, month, look, verdict and prior pointer version.
The pointer application gets a separate receipt after its actual commit; a
decision or application request alone does not prove a pointer moved.

Only `--apply-research-pointer` enables a qualifying pointer change. One atomic
`BENCH/state/champion.json` version contains both `search_champion_sha256` and
`anchor_sha256`. Each version is preserved create-only under
`BENCH/state/pointer_versions/`. Subsequent gates must compare against the
current research champion. The module never changes a serving checkpoint,
native ledger or teacher table. Campaign STOP/HOLD prevents pointer application,
and the owner-renewed 05:00 UTC October 10 deadline cannot be extended by this CLI.
D5 supersedes the earlier cutoffs; recorded attempts and monthly gate looks remain intact.

## Commands and evidence

D5 and D5.1 qualify the development panel through the benchmark lane's sealed
263-context screen and versioned arena custody. The gate reads their byte-bound
state, applies arena exclusions to both arms, and derives the retained lineage
inventory. Cumulative exclusion of more than 52 contexts, or retention of fewer
than 44 lineages, refuses the gate. The original 54-lineage protocol remains
required when no D5 custody exists.

Retained contexts keep their screened original program and measured comparator
quality. Gate intervals use the final retained lineages with the unchanged
2,000 bootstrap draws and seed 7. Reservations and decisions bind the custody
versions, coverage and exclusions; a changed custody snapshot prevents a pointer
commit. The gate never changes the benchmark custody or STOP records.

For arena exclusions it reports each arm's WIN/LOSS bounds and the paired bounds
over the original paired roster, including lineages lost entirely in the arena.
These are deterministic sensitivity bounds, separate from the retained-lineage CIs.
The panel protocol supplies no finite bound for an excluded raw gain, so that
gain remains unavailable and the promotion decision is inconclusive. No failed
or excluded gain becomes zero. These limits are separate from fixture calibration.

From the reviewed checkout:

```sh
export TMPDIR="$HOME/.cache/jope-test-tmp/rlfw-l2"
mkdir -p "$TMPDIR"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=deploy/dl-flywheel/pipelines \
  nice -n 19 ionice -c3 uv run --project python --no-sync python3 -B -m unittest \
  rlfw.tests.test_promotion_gate rlfw.tests.test_teacher_dry_run -v
```

The synthetic null test examines 1,000 entire six-look schedules with 54 parent
clusters, unchanged seed 7 and 2,000 bootstrap draws. It reports 29 promotions,
below the requested 5% limit, for that simulated null. This is fixture
calibration, not live development-panel evidence or a universal error guarantee.
The fixture suite also checks interruptions, immutable budget consumption,
receipt tampering, state links, pointer changes, a real arena-output loader and
teacher authority/integrity refusals. Logs live in
`logs/l2/promotion-teacher-final.log` and `logs/l2/shadow-gate-final.log`.

After qualified challenger and baseline games exist, compute the source bundle
identity and evaluate one explicit gate:

```sh
export PYTHONPATH=deploy/dl-flywheel/pipelines
SOURCE_SHA=$(python3 -B - <<'PY'
import hashlib, pathlib
from rlfw.arena.game_spec import digest
import rlfw.promotion_gate as gate
root = pathlib.Path(gate.__file__).parent
print(digest({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(root.rglob("*.py"))}))
PY
)
nice -n 19 ionice -c3 python3 -B -m rlfw.promotion_gate \
  --bench-root "$BENCH" \
  --challenger-run "$BENCH/runs/exit_T1" \
  --champion-run "$BENCH/runs/baseline_T1" \
  --challenger-arm trie_sample:challenger \
  --champion-arm trie_sample:current_latest \
  --deadline-unix 1791608400 --source-sha256 "$SOURCE_SHA"
```

This command consumes a monthly look and writes receipts. To enable a research
pointer application, add `--apply-research-pointer` to that same invocation;
the flag defaults off. Do not run a second invocation merely to apply a result:
it consumes another look. Keep command, source commit, exit status and environment
in the campaign operation receipt. No live challenger game or gate result is
claimed by this source change.

## Teacher dry-run contract

`rlfw.teacher_dry_run.dry_run` is a pure reporter. It takes registered incumbent /
challenger pairs and frozen `TeacherConfig` pins. Both measurements must be
qualified train-role rows under one exact comparison identity. A fresh authority
check precedes artifact access and runs again before returning a proposal.
Predicted values, unqualified outcomes and protected roles are hard errors.

The verifier returns the actual pinned native/Rust closure and binds context,
lineage, program SHA, comparison key, receipt, measured raw/clipped/grid quality,
the unchanged `1e-9` parity rule, epsilon and the two saved out-of-fold prediction
arrays. Original row IDs and target bytes must match between the programs.
The reporter computes each fold's log RMSE ratio and paired mean. Its margin is
the maximum of the registered numerical margin, 0.01, and 1.96 times the larger
of the paired row-bootstrap SE or half the fold disagreement.

A passing original margin needs fresh native execution, Rust recompile and
qualified parity evidence for both programs. Distinct fresh receipts must retain
bitwise-identical prediction arrays and original measured values. This is an
integrity check; the reporter still records only the original two statistical
fold observations. The fresh-verification callback must verify those actual
artifacts, rather than relabel a cache lookup.

At the registered minimum of 64 fitting rows, five preregistered refit splits
inside the training fitting rows must have a positive median paired difference.
Their qualified receipts belong in a separate research namespace and never
replace original quality. The callback verifies membership, both programs,
registration and full closure. Smaller contexts carry the specified
`promoted_within_protocol_only` proposal label. Nothing is actually promoted or
written by this reporter, and the owner's teacher-adoption decision remains
required.

```python
from rlfw.teacher_dry_run import TeacherConfig, dry_run

report = dry_run(
    registered_train_pairs, TeacherConfig(**frozen_teacher_pins),
    role_check=fresh_train_authority,
    verify_receipt=pinned_native_rust_verifier,
    verify_fresh=verified_fresh_execution_and_recompile,
    verify_refits=verified_registered_refit_closure,
)
```

The caller can preserve this report create-only under `BENCH/runs`, with its
source, command and input receipt inventory. Qualified train pairs and fresh
refit artifacts are runtime inputs; the synthetic fixtures do not provide them.
The additive package and its explicit APIs leave all existing default paths
unchanged. Both fixture families are registered in `run_rlfw_offline.py`, which
runs the complete additive fixture collection and writes source-bound identities,
counts and outcomes. Use the already synced, locked CPU environment through
`uv run --project python --no-sync` for the complete fixture collection and
`just required` for acceptance; full
exact-head CI and independent LR approval remain required.
