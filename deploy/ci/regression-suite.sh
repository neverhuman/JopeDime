#!/usr/bin/env bash
# deploy/ci/regression-suite.sh — plan WP12 "regression-suite" (weekly, self-hosted xbabe2):
#
#   golden42        one Golden42 panel cell: 42 datasets x 1 seed through the frozen
#                   `jope-dime-regressor` campaign; every RMSE within 1e-9 of
#                   assets/regression/golden42-baseline.json
#   encoder-parity  the four encoder parity suites of crates/jope-encoders (foundation, target
#                   encoder, JOPE policy, Starforge golden classifier/regressor/Chimera) against the
#                   pinned weights `jope models fetch --pinned` exported
#   policy-parity   the policy's proposals on the three committed seed-population fixtures
#                   (assets/regression/seed-population/*) must equal the committed files
#   bench           emitter throughput vs assets/regression/bench-baseline.json (-10 % fails);
#                   `bench --record` rewrites the baseline for this host
#   all             every step above, in that order
#
# Each step fails with the exact prerequisite that is missing and which work package produces it.
# Nothing is skipped silently. Receipts land in $RECEIPTS_DIR (default target/regression).
#
# Environment (all optional unless a step names it as required):
#   RECEIPTS_DIR              receipts and scratch state                 (target/regression)
#   JOPE_REGISTRY             dataset root with <hash>/{train,test}.csv  (WP7: `jope datasets fetch golden42`)
#   JOPE_CAMPAIGNS_BIN        prebuilt jope-dime-regressor               (else cargo build --locked)
#   JOPE_PANEL_ID             campaign run id                            (regression-<date>)
#   STARFORGE_WEIGHTS, STARFORGE_GOLDEN_JSON, STARFORGE_REGRESSOR_WEIGHTS,
#   STARFORGE_REGRESSOR_GOLDEN_JSON, STARFORGE_CHIMERA_WEIGHTS, STARFORGE_CHIMERA_GOLDEN_JSON
#                             the pinned encoder weights + golden vectors (WP7)
#   JOPE_POLICY_CHECKPOINT    checkpoint dir whose state.json hashes to the fixtures'
#                             checkpoint_sha256                          (WP11: `jope learn pull`)
#   JOPE_PROPOSE_CORPUS       the prepared corpus dir the fixtures were proposed from (WP7/WP11)
#   JOPE_ENGINE_BIN           prebuilt jope-parallel with the `neural` feature (else cargo build)
#   JOPE_BENCH_RUNS           bench repetitions                          (200)
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO=$(pwd)
RECEIPTS_DIR=${RECEIPTS_DIR:-"$REPO/target/regression"}
mkdir -p "$RECEIPTS_DIR"
ASSETS="$REPO/assets/regression"

step() { printf '\n== regression-suite: %s\n' "$*"; }
fail() { printf 'regression-suite FAILED [%s]: %s\n' "$CURRENT" "$*" >&2; exit 1; }
require_tool() { command -v "$1" > /dev/null || fail "required tool missing: $1"; }
require_file() { [ -f "$1" ] || fail "$2: $1 is missing"; }
require_dir() { [ -d "$1" ] || fail "$2: $1 is missing"; }
require_env() { # <name> <who-produces-it>
  if [ -z "${!1:-}" ]; then fail "$1 is unset; $2"; fi
}
sha256() { sha256sum "$1" | cut -c1-64; }
CURRENT=setup

# ---------------------------------------------------------------------------------------------
golden42() {
  CURRENT=golden42
  step "Golden42 panel cell (42 datasets x 1 seed)"
  require_tool jq; require_tool python3
  local baseline="$ASSETS/golden42-baseline.json"
  require_file "$baseline" "the committed baseline"
  [ "$(jq -r .schema "$baseline")" = "jope.golden42-baseline/v1" ] || fail "$baseline has an unexpected schema"
  local panel="$REPO/crates/jope-campaigns/manifests/panel.txt"
  require_file "$panel" "the pinned panel"
  [ "$(sha256 "$panel")" = "$(jq -r .panel.sha256 "$baseline")" ] || fail "panel.txt sha256 differs from the baseline's panel"
  local count; count=$(jq '.datasets | length' "$baseline")
  [ "$count" = 42 ] || fail "baseline lists $count datasets, expected 42"

  require_env JOPE_REGISTRY "run \`jope datasets fetch golden42\` (WP7) and point JOPE_REGISTRY at its by-id root"
  require_dir "$JOPE_REGISTRY" "JOPE_REGISTRY (WP7 dataset fetch output)"
  local cores; cores=$(nproc)
  [ "$cores" -ge 60 ] || fail "the frozen campaign plans 60 cores (require_core_plan); this host has $cores"

  local panel_id=${JOPE_PANEL_ID:-"regression-$(date -u +%Y%m%d)"}
  local scratch="$RECEIPTS_DIR/golden42/registry"
  local state="$RECEIPTS_DIR/golden42/state"
  if [ -e "$scratch" ] || [ -e "$state" ]; then
    fail "$RECEIPTS_DIR/golden42 already exists; the campaign refuses to resume — remove it or choose another RECEIPTS_DIR"
  fi
  local legacy_runs="$REPO/crates/jope-campaigns/runs/$panel_id"
  if [ -e "$legacy_runs" ]; then
    fail "$legacy_runs exists (a previous run under this panel id); remove it or set JOPE_PANEL_ID"
  fi
  mkdir -p "$scratch" "$state"
  export JOPE_STATE="$state"

  # The scratch registry holds only the pinned train/test files, verified against the baseline,
  # so the campaign can publish predictions/reports next to them without touching JOPE_REGISTRY.
  local hash train test
  while read -r hash; do
    train="$JOPE_REGISTRY/$hash/train.csv"; test="$JOPE_REGISTRY/$hash/test.csv"
    require_file "$train" "dataset $hash (WP7: jope datasets fetch golden42)"
    require_file "$test" "dataset $hash (WP7: jope datasets fetch golden42)"
    [ "$(sha256 "$train")" = "$(jq -r --arg h "$hash" '.datasets[] | select(.dataset_hash==$h) | .train_sha256' "$baseline")" ] \
      || fail "$train does not hash to the baseline's train_sha256 for $hash"
    [ "$(sha256 "$test")" = "$(jq -r --arg h "$hash" '.datasets[] | select(.dataset_hash==$h) | .test_sha256' "$baseline")" ] \
      || fail "$test does not hash to the baseline's test_sha256 for $hash"
    mkdir -p "$scratch/$hash"
    cp "$train" "$scratch/$hash/train.csv"; cp "$test" "$scratch/$hash/test.csv"
    [ ! -f "$JOPE_REGISTRY/$hash/meta.json" ] || cp "$JOPE_REGISTRY/$hash/meta.json" "$scratch/$hash/meta.json"
  done < <(jq -r '.datasets[].dataset_hash' "$baseline")

  local bin
  if [ -n "${JOPE_CAMPAIGNS_BIN:-}" ]; then
    bin=$JOPE_CAMPAIGNS_BIN; [ -x "$bin" ] || fail "JOPE_CAMPAIGNS_BIN=$bin is not executable"
  else
    bin=$(python3 "$REPO/deploy/ci/cargo-artifact.py" build --profile release \
      --package jope-dime-regressor --bin jope-dime-regressor) \
      || fail "cargo build -p jope-dime-regressor failed (WP5 campaigns crate)"
  fi
  local common=(--registry "$scratch" --panel-id "$panel_id" --kernel-version legacy-v2)
  local phase
  for phase in verify-inputs select-panel validate-panel score-panel; do
    printf -- '-- jope-dime-regressor %s\n' "$phase"
    "$bin" "${common[@]}" "$phase" 2>&1 | tee "$RECEIPTS_DIR/golden42/$phase.log" > /dev/null \
      || fail "jope-dime-regressor $phase failed (see $RECEIPTS_DIR/golden42/$phase.log)"
  done

  # Compare every published report with the baseline: |rmse - baseline| <= 1e-9.
  local report="$RECEIPTS_DIR/golden42/golden42-cell.json"
  python3 - "$baseline" "$scratch" "$report" "$panel_id" <<'PY' || fail "Golden42 RMSE comparison failed (see $RECEIPTS_DIR/golden42/golden42-cell.json)"
import json, sys, hashlib, socket, subprocess, datetime
baseline_path, scratch, out, panel_id = sys.argv[1:5]
baseline = json.load(open(baseline_path))
tol = float(baseline["tolerance"])
rows, bad = [], 0
for d in baseline["datasets"]:
    h = d["dataset_hash"]
    path = f"{scratch}/{h}/reports/jope-dime-regressor.json"
    try:
        r = json.load(open(path))
    except FileNotFoundError:
        rows.append({"dataset_hash": h, "status": "missing-report", "path": path}); bad += 1; continue
    if r.get("schema") != "JopeDimeDatasetReport.v3":
        rows.append({"dataset_hash": h, "status": "unexpected-schema", "schema": r.get("schema")}); bad += 1; continue
    rmse = float(r["metrics"]["rmse"]); delta = abs(rmse - float(d["rmse"]))
    ok = delta <= tol
    rows.append({"dataset_hash": h, "status": "ok" if ok else "rmse-drift", "rmse": rmse,
                 "baseline_rmse": d["rmse"], "delta": delta,
                 "semantic_sha256": r["algorithm"]["semantic_sha256"], "baseline_semantic_sha256": d["semantic_sha256"],
                 "prediction_sha256": r["prediction"]["sha256"], "baseline_prediction_sha256": d["prediction_sha256"],
                 "report_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()})
    bad += 0 if ok else 1
receipt = {"schema": "jope.golden42-cell-receipt/v1", "panel_id": panel_id, "tolerance": tol,
           "host": socket.gethostname(), "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
           "finished": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "datasets": len(rows), "failures": bad, "rows": rows}
json.dump(receipt, open(out, "w"), indent=2); open(out, "a").write("\n")
for row in rows:
    if row["status"] != "ok":
        print(f"{row['dataset_hash']}: {row['status']} " + (f"delta={row['delta']:.3e}" if "delta" in row else ""))
print(f"golden42 cell: {len(rows) - bad}/{len(rows)} within {tol}")
sys.exit(1 if bad else 0)
PY
  printf 'golden42 receipt: %s\n' "$report"
}

# ---------------------------------------------------------------------------------------------
encoder_parity() {
  CURRENT=encoder-parity
  step "encoder parity (foundation, target encoder, JOPE policy, Starforge golden)"
  local v
  for v in STARFORGE_WEIGHTS STARFORGE_GOLDEN_JSON STARFORGE_REGRESSOR_WEIGHTS \
           STARFORGE_REGRESSOR_GOLDEN_JSON STARFORGE_CHIMERA_WEIGHTS STARFORGE_CHIMERA_GOLDEN_JSON; do
    require_env "$v" "run \`jope models fetch --pinned\` (WP7) and export the paths it prints"
    [ -e "${!v}" ] || fail "$v=${!v} does not exist"
  done
  local log="$RECEIPTS_DIR/encoder-parity.log"
  if ! cargo test --locked -p starforge --test foundation_parity --test target_encoder_parity \
        --test jope_policy_parity --test golden_parity -- --nocapture 2>&1 | tee "$log" > /dev/null; then
    tail -40 "$log"
    fail "an encoder parity test failed (full log: $log)"
  fi
  if grep -q '^skipping' "$log"; then
    grep '^skipping' "$log"
    fail "a parity test skipped itself; every encoder must be exercised (log: $log)"
  fi
  grep -E '^test result' "$log" > "$RECEIPTS_DIR/encoder-parity.txt"
  printf 'encoder parity receipt: %s\n' "$RECEIPTS_DIR/encoder-parity.txt"
}

# ---------------------------------------------------------------------------------------------
policy_parity() {
  CURRENT=policy-parity
  step "policy parity on the committed seed-population fixtures"
  require_tool jq
  local fixtures=("$ASSETS"/seed-population/*/seeds.json)
  [ "${#fixtures[@]}" = 3 ] || fail "expected 3 seed-population fixtures, found ${#fixtures[@]}"
  require_env JOPE_POLICY_CHECKPOINT "point it at the checkpoint directory (state.json) the fixtures bind (WP11: jope learn pull)"
  require_file "$JOPE_POLICY_CHECKPOINT/state.json" "JOPE_POLICY_CHECKPOINT"
  require_env JOPE_PROPOSE_CORPUS "point it at the prepared corpus the fixtures were proposed from (WP7/WP11)"
  require_dir "$JOPE_PROPOSE_CORPUS" "JOPE_PROPOSE_CORPUS"
  local checkpoint_sha; checkpoint_sha=$(sha256 "$JOPE_POLICY_CHECKPOINT/state.json")
  local bin
  if [ -n "${JOPE_ENGINE_BIN:-}" ]; then
    bin=$JOPE_ENGINE_BIN; [ -x "$bin" ] || fail "JOPE_ENGINE_BIN=$bin is not executable"
  else
    bin=$(python3 "$REPO/deploy/ci/cargo-artifact.py" build --profile release \
      --package jope-parallel --bin jope-parallel --features neural) \
      || fail "cargo build -p jope-parallel --features neural failed (LibTorch per deploy/receipts/libtorch.json, WP8)"
  fi
  "$bin" --help 2>/dev/null | grep -q '^ *propose\b' || fail "$bin has no \`propose\` subcommand (built without the neural feature?)"

  local fixture dir want cut prepared out actual expected receipt failures=0
  receipt="$RECEIPTS_DIR/policy-parity.json"; : > "$receipt"
  for fixture in "${fixtures[@]}"; do
    dir=$(dirname "$fixture"); want=$(jq -r .checkpoint_sha256 "$fixture"); cut=$(jq -r .cut "$fixture")
    prepared=$(jq -r .prepared_sha256 "$fixture")
    [ "$want" = "$checkpoint_sha" ] || fail "$fixture binds checkpoint $want but JOPE_POLICY_CHECKPOINT/state.json hashes to $checkpoint_sha"
    [ "$(jq -r .seeds_sha256 "$dir/generation.json")" = "$(sha256 "$fixture")" ] || fail "$dir/generation.json does not vouch for its seeds.json"
    out="$RECEIPTS_DIR/policy-parity/$(basename "$dir")"
    rm -rf "$out"; mkdir -p "$out"
    "$bin" --root "$RECEIPTS_DIR/policy-parity" propose --corpus "$JOPE_PROPOSE_CORPUS" \
      --checkpoint "$JOPE_POLICY_CHECKPOINT" --output "$out" --cuts 1 --cut-start "$cut" \
      > "$out/propose.log" 2>&1 || fail "propose failed for $(basename "$dir") (see $out/propose.log)"
    actual=$(find "$out" -name seeds.json | head -1)
    [ -n "$actual" ] || fail "propose wrote no seeds.json under $out"
    [ "$(jq -r .prepared_sha256 "$actual")" = "$prepared" ] || fail "$(basename "$dir"): proposals were made from prepared corpus $(jq -r .prepared_sha256 "$actual"), fixture expects $prepared"
    expected=$(jq -S '{cut, prepared_sha256, context_sha256, checkpoint_sha256, candidates}' "$fixture")
    if [ "$(jq -S '{cut, prepared_sha256, context_sha256, checkpoint_sha256, candidates}' "$actual")" = "$expected" ]; then
      printf '{"fixture":"%s","status":"ok"}\n' "$(basename "$dir")" >> "$receipt"
    else
      printf '{"fixture":"%s","status":"mismatch","actual":"%s"}\n' "$(basename "$dir")" "$actual" >> "$receipt"
      printf 'policy parity mismatch: %s (actual %s)\n' "$(basename "$dir")" "$actual"
      failures=$((failures + 1))
    fi
  done
  [ "$failures" = 0 ] || fail "$failures of ${#fixtures[@]} fixtures differ from the policy's proposals (receipt: $receipt)"
  printf 'policy parity receipt: %s\n' "$receipt"
}

# ---------------------------------------------------------------------------------------------
bench() { # [--record]
  CURRENT=bench
  step "emitter throughput bench"
  require_tool python3
  local record=0; [ "${1:-}" = "--record" ] && record=1
  local genome="$REPO/crates/feat-math/assets/champions/8944cbdf.dsl"
  require_file "$genome" "the committed champion genome"
  local emit runs=${JOPE_BENCH_RUNS:-200} out="$RECEIPTS_DIR/bench-emit"
  emit=$(python3 "$REPO/deploy/ci/cargo-artifact.py" build --profile release --package feat-math --example emit) \
    || fail "cargo build -p feat-math --example emit failed"
  rm -rf "$out"; mkdir -p "$out"
  "$emit" "$genome" "$out/warm" > /dev/null 2>&1 || fail "$emit cannot emit the champion"
  local timings="$out/timings-ns.txt" i t0 t1
  : > "$timings"
  for ((i = 0; i < runs; i++)); do
    t0=$(date +%s%N); "$emit" "$genome" "$out/run" > /dev/null 2>&1; t1=$(date +%s%N)
    printf '%s\n' $((t1 - t0)) >> "$timings"
  done
  local receipt="$RECEIPTS_DIR/bench.json" baseline="$ASSETS/bench-baseline.json"
  python3 - "$timings" "$receipt" "$baseline" "$record" "$genome" "$runs" <<'PY' || fail "bench comparison failed (receipt: $RECEIPTS_DIR/bench.json)"
import json, sys, statistics, socket, subprocess, platform, os, datetime, shutil
timings, receipt_path, baseline_path, record, genome, runs = sys.argv[1:7]
ns = [int(l) for l in open(timings) if l.strip()]
ms = [n / 1e6 for n in ns]
sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
rustc = subprocess.check_output(["rustc", "--version"]).decode().strip()
receipt = {
    "schema": "jope.bench-baseline/v1",
    "bench": "emit-champion-8944cbdf",
    "what": "wall time of one `feat-math` release `emit` example run (process spawn included) on the committed champion; lower is better",
    "genome": os.path.relpath(genome, os.getcwd()),
    "genome_sha256": __import__("hashlib").sha256(open(genome, "rb").read()).hexdigest(),
    "runs": int(runs),
    "host": socket.gethostname(),
    "cpu": platform.processor() or platform.machine(),
    "nproc": os.cpu_count(),
    "kernel": platform.release(),
    "git_sha": sha,
    "rustc": rustc,
    "recorded": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "mean_ms": round(statistics.fmean(ms), 4),
    "median_ms": round(statistics.median(ms), 4),
    "min_ms": round(min(ms), 4),
    "p95_ms": round(sorted(ms)[int(0.95 * (len(ms) - 1))], 4),
    "alert_pct": 10.0,
}
json.dump(receipt, open(receipt_path, "w"), indent=2); open(receipt_path, "a").write("\n")
print(f"emit x{runs}: median {receipt['median_ms']} ms, mean {receipt['mean_ms']} ms, min {receipt['min_ms']} ms on {receipt['host']}")
if record == "1":
    shutil.copyfile(receipt_path, baseline_path); print(f"baseline recorded: {baseline_path}"); sys.exit(0)
if not os.path.exists(baseline_path):
    print(f"no baseline at {baseline_path}; run `regression-suite.sh bench --record` first"); sys.exit(1)
base = json.load(open(baseline_path))
if base.get("host") != receipt["host"]:
    print(f"baseline was recorded on {base.get('host')}, this is {receipt['host']}; record a baseline for this host"); sys.exit(1)
if base.get("bench") != receipt["bench"] or base.get("genome_sha256") != receipt["genome_sha256"]:
    print("baseline measures a different bench/genome; re-record it"); sys.exit(1)
limit = base["median_ms"] * (1 + base.get("alert_pct", 10.0) / 100)
if receipt["median_ms"] > limit:
    print(f"REGRESSION: median {receipt['median_ms']} ms > {limit:.4f} ms (baseline {base['median_ms']} ms + {base.get('alert_pct', 10.0)} %)"); sys.exit(1)
print(f"within budget: median {receipt['median_ms']} ms <= {limit:.4f} ms (baseline {base['median_ms']} ms at {base['git_sha'][:12]})")
PY
  printf 'bench receipt: %s\n' "$RECEIPTS_DIR/bench.json"
}

# ---------------------------------------------------------------------------------------------
usage() { sed -n '2,20p' "$0"; exit 2; }
case "${1:-all}" in
  golden42) golden42 ;;
  encoder-parity) encoder_parity ;;
  policy-parity) policy_parity ;;
  bench) bench "${2:-}" ;;
  all) golden42; encoder_parity; policy_parity; bench ;;
  *) usage ;;
esac
printf '\nregression-suite: %s PASSED (receipts in %s)\n' "${1:-all}" "$RECEIPTS_DIR"
