#!/usr/bin/env bash
# deploy/ci/e2e-smoke.sh — the `e2e-smoke` lane: one end-to-end `jope pipeline run` on a committed
# dataset, twice.
#
# What it proves:
#   * the pipeline runs dataset -> prepare -> search -> select -> emit -> receipt offline, with the
#     propose stage explicitly skipped (`--policy skip`) rather than silently absent;
#   * the search actually MEASURED: every cut reports successful measurements, the selection is
#     `min_mean_validation_rmse` and the receipt carries a champion with a real held-out RMSE and
#     R^2. This is the assertion that fails if the search ever regresses to the unmeasured
#     bootstrap fallback, which is what the engine-backed search did on every run;
#   * the receipt validates against the committed `jope.pipeline-receipt/v1` schema, records the
#     skip with a reason, and names the emitted files with their real sha256;
#   * two runs at the same seeds emit byte-identical sources;
#   * every emitted program compiles, runs on the committed split and agrees with its siblings on
#     the IEEE-754 bits of its predictions (the same comparison deploy/ci/emitters.sh makes, via
#     the same deploy/ci/emitters/compare-bits.py).
#
# There are no committed sha256 pins here: the selected genome depends on what the search measures,
# so this lane pins behaviour, not bytes. deploy/ci/emitters.sh is where the byte pins live, and it
# is also where the four-language C++/JavaScript bit agreement is proven on a pinned genome — the
# champion this lane finds is whatever measured best, and the C++/ES2020 emitters do not cover
# every head the search can reach (the exclusions are recorded in the receipt and printed).
# Nothing is skipped: a missing compiler fails the lane.
set -euo pipefail
cd "$(dirname "$0")/../.."

DATASET=${E2E_DATASET:-datasets/test/pmlb-229-pwlinear}
OUT=${E2E_OUT:-target/e2e-smoke}
BUDGET=${E2E_BUDGET:-evaluations=4,seconds=60}
RNG_SEED=${E2E_RNG_SEED:-7}
CXX_FLAGS=(-O3 -ffp-contract=off -fno-fast-math)
if [ -n "${JOPE_CXX_MARCH:-}" ]; then CXX_FLAGS+=("-march=${JOPE_CXX_MARCH}"); fi

for tool in rustc g++ clang++ node python3 sha256sum; do
  command -v "$tool" > /dev/null || { echo "missing required tool: $tool"; exit 1; }
done

rm -rf "$OUT"; mkdir -p "$OUT"
cargo build -q --locked --bin jope
JOPE=target/debug/jope

# One process per candidate, each pinned to one thread, as the engine expects.
export OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

for run in a b; do
  echo "== pipeline run $run"
  "$JOPE" pipeline run \
    --dataset "$DATASET" \
    --budget "$BUDGET" \
    --rng-seed "$RNG_SEED" \
    --emit rust,python,cpp17,cpp20,js \
    --policy skip \
    --output "$OUT/$run"
done

# Multi-cut selection. Nothing else in this script or in the unit tests passes --cuts, so the
# whole cross-cut path -- including the remeasurement that makes the ranking comparable -- used to
# early-return on every CI run and could be deleted without turning a single test red.
#
# The assertion is about the candidate pool, not about which genome wins, so it does not depend on
# what the search happens to find. Without remeasurement a genome discovered on one cut is only
# ever measured there, so NO non-bootstrap genome reaches every cut; with it, the top-k from each
# cut are measured on all of them.
#
# The COUNT scales with the budget, so the assertion deliberately requires only that the set is
# non-empty. At this lane's budget (evaluations=4,seconds=60) it yields 9; at the larger
# evaluations=24,seconds=120 used while developing the check it yielded 12. Without remeasurement
# it is 0 at either budget, which is the property being tested.
#
# The contrast that matters, measured at this dataset and seed 7: the champion degrades from
# `(pipe X (rank append=true) (gpr m=400 steps=60))` at hold RMSE 0.070814 to the bootstrap
# `(pipe X (ridge))` at 0.095582 -- 26 % worse, which is exactly the pathology cross-cut
# selection exists to avoid.
echo "== multi-cut selection ranks candidates measured on every cut"
"$JOPE" pipeline run \
  --dataset "$DATASET" \
  --budget "$BUDGET" \
  --rng-seed "$RNG_SEED" \
  --emit rust,python \
  --policy skip \
  --cuts 3 \
  --output "$OUT/cuts3"
python3 - "$OUT/cuts3" <<'PYEOF'
import collections, glob, json, os, sys

root = sys.argv[1]
files = sorted(glob.glob(os.path.join(root, "search", "cut-*", "observations.json")))
assert len(files) == 3, f"expected three per-cut observation files, found {len(files)}"

seen = collections.defaultdict(set)
for path in files:
    for observation in json.load(open(path)):
        seen[observation["genome"]].add(observation["cut"])

cuts = {cut for cuts_seen in seen.values() for cut in cuts_seen}
assert cuts == {0, 1, 2}, f"observations mention cuts {sorted(cuts)}, expected 0..2"

bootstrap = "(pipe X (ridge))"
covered = [g for g, c in seen.items() if c == cuts and g != bootstrap]
assert covered, (
    "no searched genome was measured on every cut, so the selection ranked means taken over "
    "different cuts -- remeasurement did not run"
)

receipt = json.load(open(os.path.join(root, "receipt.json")))
assert receipt["ok"], receipt

# Indexed unconditionally and with no `in` guard. The first version of this read
# receipt["select"]["best"] behind an `if "select" in receipt` -- but the receipt has no
# top-level "select" key, so `best` was always None, the assertion never ran, and the line
# below printed "champion ranked over 3" regardless. A guard that silently turns an assertion
# off is worse than no assertion: it reports success for a check that did not happen.
best = receipt["search"]["best"]
assert best["cuts_measured"] == 3, best

print(
    f"  ok: {len(covered)} searched genome(s) measured on all 3 cuts; "
    f"champion {best['genome']} ranked over {best['cuts_measured']}"
)
PYEOF

# A run that emits nothing must say so and fail, instead of reporting a passed emit stage over an
# empty directory. Which branch this takes depends on the champion the search happens to find, so
# both are asserted rather than assuming today's winner: if cpp17 covers it there must be an
# artifact, and if it does not the run must exit non-zero with `ok:false` and a failed emit stage.
echo "== a single uncovered target fails the run instead of passing with nothing emitted"
set +e
"$JOPE" pipeline run \
  --dataset "$DATASET" \
  --budget "$BUDGET" \
  --rng-seed "$RNG_SEED" \
  --emit cpp17 \
  --policy skip \
  --output "$OUT/cpp17-only" > "$OUT/cpp17-only.log" 2>&1
cpp17_rc=$?
set -e
python3 - "$OUT/cpp17-only/receipt.json" "$cpp17_rc" <<'PYEOF'
import json, sys
receipt, rc = json.load(open(sys.argv[1])), int(sys.argv[2])
emit = next(s for s in receipt["stages"] if s["name"] == "emit")
if receipt["artifacts"]:
    assert rc == 0, f"cpp17 was emitted but the run exited {rc}"
    assert emit["status"] == "passed" and receipt["ok"], receipt
    print(f"  ok: cpp17 covered this champion; emitted {len(receipt['artifacts'])} artifact(s), rc=0")
else:
    assert rc != 0, "no code was emitted and the run still exited 0"
    assert emit["status"] == "failed", emit
    assert receipt["ok"] is False, "ok survived a failed stage"
    assert receipt["emit_exclusions"], "nothing emitted and nothing excluded"
    print(f"  ok: cpp17 does not cover this champion; rc={rc}, emit failed: {emit['reason']}")
PYEOF

echo "== emitted sources are byte-identical across the two runs"
diff -r "$OUT/a/emit" "$OUT/b/emit"
echo "  ok"

echo "== receipt validates against the committed schema"
"$JOPE" pipeline verify-receipt --receipt "$OUT/a/receipt.json"

echo "== receipt is truthful about the skip and the artifacts"
python3 - "$OUT/a" <<'PY'
import hashlib, json, os, sys
run = sys.argv[1]
receipt = json.load(open(os.path.join(run, "receipt.json")))
stages = {s["name"]: s for s in receipt["stages"]}
propose = stages["propose"]
assert propose["status"] == "skipped", propose
assert propose.get("reason"), "the skipped propose stage records no reason"
for name in ("dataset", "prepare", "search", "select", "emit"):
    assert stages[name]["status"] == "passed", stages[name]
assert receipt["policy"] == "skip", receipt["policy"]

# The search measured. Without this the lane passes on a run that fitted nothing.
search = receipt["search"]
cuts = search["cuts"]
assert cuts, "no cut was searched"
measured = sum(c["successful_measurements"] for c in cuts)
assert measured > 0, f"the search reported {measured} successful measurements"
assert search["selection"] == "min_mean_validation_rmse", search["selection"]
best = search.get("best")
assert best, "a measured search recorded no champion"
assert best["genome"] == search["selected_genome"], best["genome"]
for key in ("validation_rmse", "validation_r2", "train_r2"):
    value = best[key]
    assert isinstance(value, (int, float)) and value == value, f"{key} is {value}"
assert best["validation_rmse"] >= 0.0, best["validation_rmse"]
assert best["cuts_measured"] >= 1, best["cuts_measured"]
for cut in cuts:
    assert cut["completed_evaluations"] == (
        cut["successful_measurements"] + cut["failed_or_unavailable_measurements"]
    ), cut
    # The fit rows and the scored rows are disjoint by construction; both must exist.
    assert cut["train_rows"] > 0 and cut["hold_rows"] > 0, cut
    assert cut["stopped_by"] in ("evaluations", "seconds"), cut["stopped_by"]
    if cut["successful_measurements"]:
        assert cut["best_program"], cut
print(f"  ok: search measured {measured} candidate(s); champion {search['selected_genome']} "
      f"hold RMSE {best['validation_rmse']:.6f} R2 {best['validation_r2']:.6f}")
assert receipt["artifacts"], "no artifact was emitted"
for artifact in receipt["artifacts"]:
    path = os.path.join(run, "emit", artifact["path"])
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    assert digest == artifact["sha256"], f"{path}: {digest} != {artifact['sha256']}"
    assert os.path.getsize(path) == artifact["bytes"], path
languages = sorted({a["language"] for a in receipt["artifacts"]} | {e["language"] for e in receipt["emit_exclusions"]})
assert languages == ["cpp17", "cpp20", "javascript", "python", "rust"], languages
print(f"  ok: {len(receipt['artifacts'])} artifact(s), propose skipped: {propose['reason'][:60]}...")
PY

DIR="$OUT/a/emit"
# The emitted programs take (train.csv, test_features.csv, predictions): derive the features-only
# test file from the committed split (the last column is the target).
ncols=$(head -1 "$DATASET/train.csv" | awk -F, '{print NF}')
cut -d, -f1-"$((ncols - 1))" "$DATASET/test.csv" > "$OUT/test_features.csv"

programs=()
execute() { local name=$1; shift; "$@" "$DATASET/train.csv" "$OUT/test_features.csv" "$DIR/pred-$name.txt"; programs+=("$name"); }

targets=$(cat "$DIR/targets.txt")
echo "== compile + run the emitted programs ($targets)"
case " $targets " in *" py "*)
  python3 -m py_compile "$DIR/model.py"; execute python python3 "$DIR/model.py";;
esac
case " $targets " in *" rs "*)
  rustc -O -o "$DIR/model-rs" "$DIR/model.rs"; execute rs "$DIR/model-rs";;
esac
# The C++ and JavaScript artifacts are libraries; the CSV drivers in deploy/ci/emitters/ are
# appended to make standalone programs with the same argv contract. One C++ source, two standards.
case " $targets " in *" cpp "*)
  cat "$DIR/model.cpp" deploy/ci/emitters/driver.cpp > "$DIR/program.cpp"
  for cxx in g++ clang++; do for std in 17 20; do
    "$cxx" "-std=c++$std" "${CXX_FLAGS[@]}" -o "$DIR/model-$cxx-$std" "$DIR/program.cpp"
    execute "$cxx-$std" "$DIR/model-$cxx-$std"
  done; done;;
esac
case " $targets " in *" js "*)
  cat "$DIR/model.js" deploy/ci/emitters/driver.js > "$DIR/program.js"
  node --check "$DIR/program.js"; execute js node "$DIR/program.js";;
esac

echo "== compare predictions of ${#programs[@]} programs (IEEE-754 bit patterns)"
# The champion decides which languages are covered, but a run that emitted nothing comparable is
# not a proof of anything.
[ "${#programs[@]}" -ge 2 ] || { echo "only ${#programs[@]} program(s) emitted; nothing to compare"; exit 1; }
python3 deploy/ci/emitters/compare-bits.py "$DIR" > "$DIR/predictions.bits"
echo "  $(wc -l < "$DIR/predictions.bits") rows; programs agree: ${programs[*]}"
echo "e2e-smoke ok"
