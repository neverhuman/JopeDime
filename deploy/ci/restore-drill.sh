#!/usr/bin/env bash
# deploy/ci/restore-drill.sh — plan WP12 "restore-drill": from a fresh directory, prove S3 is a
# complete backup: `jope models fetch --pinned`, `jope datasets fetch test-10`,
# `jope learn pull --serving`, then the e2e smoke. Every step either passes or fails with the
# exact prerequisite that is missing (which WP produces it); nothing is skipped.
#
#   bash deploy/ci/restore-drill.sh            # DRILL_DIR defaults to a fresh mktemp directory
#   DRILL_DIR=/scratch/drill RECEIPTS_DIR=target/restore-drill bash deploy/ci/restore-drill.sh
#
# Environment: AWS credentials for the JopeDimeCiRead role (OIDC in the workflow, a profile
# locally); JOPE_BIN to use a prebuilt `jope`; JOPE_OFFLINE must be unset (this is the online test).
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO=$(pwd)

RECEIPTS_DIR=${RECEIPTS_DIR:-"$REPO/target/restore-drill"}
DRILL_DIR=${DRILL_DIR:-$(mktemp -d -t jope-restore-drill.XXXXXX)}
mkdir -p "$RECEIPTS_DIR"
LOG="$RECEIPTS_DIR/restore-drill.log"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

STEP_STATUS=()
started=$(date -u +%Y-%m-%dT%H:%M:%SZ)

step() { printf '\n== restore-drill: %s\n' "$*"; }
fail() { printf 'restore-drill FAILED: %s\n' "$*" >&2; write_receipt failed "$*"; exit 1; }
record() { STEP_STATUS+=("$1=$2"); }

write_receipt() { # <status> <detail>
  local status=$1 detail=$2
  {
    printf '{\n  "schema": "jope.restore-drill-receipt/v1",\n'
    printf '  "status": "%s",\n  "detail": "%s",\n' "$status" "${detail//\"/\\\"}"
    printf '  "started": "%s",\n  "finished": "%s",\n' "$started" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '  "host": "%s",\n  "git_sha": "%s",\n' "$(hostname)" "$(git rev-parse HEAD)"
    printf '  "drill_dir": "%s",\n  "jope_home": "%s",\n  "jope_state": "%s",\n' "$DRILL_DIR" "$JOPE_HOME" "$JOPE_STATE"
    printf '  "steps": ['
    local first=1 s
    for s in "${STEP_STATUS[@]}"; do
      if [ $first = 1 ]; then first=0; else printf ','; fi
      printf '\n    {"step": "%s", "status": "%s"}' "${s%%=*}" "${s#*=}"
    done
    printf '\n  ]\n}\n'
  } > "$RECEIPTS_DIR/restore-drill.json"
  printf 'receipt: %s\n' "$RECEIPTS_DIR/restore-drill.json"
}

# ---- 0. a fresh home and state: nothing from the checkout except the source tree -------------
step "fresh JOPE_HOME / JOPE_STATE under $DRILL_DIR"
[ -d "$DRILL_DIR" ] || fail "DRILL_DIR $DRILL_DIR does not exist"
if [ -n "$(ls -A "$DRILL_DIR")" ]; then
  fail "DRILL_DIR $DRILL_DIR is not empty; the drill must start from nothing"
fi
export JOPE_HOME="$DRILL_DIR/home" JOPE_STATE="$DRILL_DIR/state"
mkdir -p "$JOPE_HOME" "$JOPE_STATE"
touch "$JOPE_HOME/.jope-home"
if [ -n "${JOPE_OFFLINE:-}" ]; then
  fail "JOPE_OFFLINE=$JOPE_OFFLINE is set; the restore drill must reach S3"
fi
record fresh-dir ok

# ---- 1. the `jope` umbrella binary (WP9 jope-cli) ---------------------------------------------
step "jope binary"
if [ -n "${JOPE_BIN:-}" ]; then
  [ -x "$JOPE_BIN" ] || fail "JOPE_BIN=$JOPE_BIN is not executable"
else
  cargo build -q --locked -p jope-cli --bin jope || fail "cargo build -p jope-cli failed"
  JOPE_BIN="$REPO/target/debug/jope"
fi
"$JOPE_BIN" --version || fail "$JOPE_BIN --version failed"
need_subcommand() { # <subcommand> <wp>
  if ! "$JOPE_BIN" "$1" --help > /dev/null 2>&1; then
    fail "\`jope $1\` is not implemented by $JOPE_BIN ($2 output missing); the drill cannot continue"
  fi
}
record jope-binary ok

# ---- 2. models: pinned bundles from S3, verified by sha256 (WP7) ------------------------------
step "jope models fetch --pinned"
need_subcommand models "WP7 (models catalog + S3 fetch)"
"$JOPE_BIN" models fetch --pinned || fail "jope models fetch --pinned failed (S3 access or manifest)"
"$JOPE_BIN" models verify || fail "jope models verify failed after fetch (a pinned bundle does not match its sha256)"
record models-fetch ok

# ---- 3. datasets: the ten test datasets (WP7) --------------------------------------------------
step "jope datasets fetch test-10"
need_subcommand datasets "WP7 (dataset catalog + S3 fetch)"
"$JOPE_BIN" datasets fetch test-10 || fail "jope datasets fetch test-10 failed"
"$JOPE_BIN" datasets verify || fail "jope datasets verify failed after fetch"
record datasets-fetch ok

# ---- 4. learning store: the serving wave (WP11) -----------------------------------------------
step "jope learn pull --serving"
need_subcommand learn "WP11 (learning store pull/push)"
"$JOPE_BIN" learn pull --serving || fail "jope learn pull --serving failed (serving pointer or checkpoint unreachable)"
record learn-pull ok

# ---- 5. end-to-end smoke on the restored state (WP8) -----------------------------------------
step "e2e smoke"
SMOKE="$REPO/deploy/ci/e2e-smoke.sh"
[ -f "$SMOKE" ] || fail "deploy/ci/e2e-smoke.sh is missing (WP8 output); the restored state was not exercised"
bash "$SMOKE" || fail "e2e smoke failed on the restored state"
record e2e-smoke ok

write_receipt passed "models, datasets, serving wave and the e2e smoke restored from S3 into $DRILL_DIR"
printf '\nrestore-drill PASSED\n'
