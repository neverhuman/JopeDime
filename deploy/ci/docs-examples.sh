#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
bin=${JOPEDIME_BIN:-"$repo_root/target/debug/jopedime"}
csv=${REL05_DATASET_CSV:?set REL05_DATASET_CSV to the permitted rel05-s3-catalog-30-row CSV}
home=${JOPEDIME_HOME:?set JOPEDIME_HOME to an isolated directory outside the repository}
server_pid=
server_started=0

fail() {
    printf 'docs-examples: %s\n' "$*" >&2
    exit 1
}

[[ -x "$bin" ]] || fail "candidate executable is missing or not executable: $bin"
[[ -f "$csv" ]] || fail "catalog CSV does not exist: $csv"
command -v curl >/dev/null || fail 'curl is required'
command -v python3 >/dev/null || fail 'python3 is required'

bin=$(realpath "$bin")
csv=$(realpath "$csv")
mkdir -p "$home"
home=$(realpath "$home")
case "$home/" in
    /tmp/*) fail 'JOPEDIME_HOME must not be under /tmp' ;;
    "$repo_root"/*) fail 'JOPEDIME_HOME must be outside the repository' ;;
esac
chmod 700 "$home"
export JOPEDIME_HOME="$home"
[[ ! -e "$home/daemon.json" ]] || fail 'JOPEDIME_HOME already has a daemon record; stop that engine or choose an isolated home'

python3 - "$csv" <<'PY'
import csv
import hashlib
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
data = path.read_bytes()
expected = "3fcf54fcaf3444a421c1f8de5c3649edc7f052f4bc50575fe621474cccd98c3e"
if hashlib.sha256(data).hexdigest() != expected:
    raise SystemExit("catalog CSV SHA-256 does not match the REL-05 fixture")
with path.open(newline="") as stream:
    rows = list(csv.reader(stream))
if len(rows) != 31 or len(rows[0]) != 13 or any(len(row) != 13 for row in rows):
    raise SystemExit("catalog CSV must contain 30 data rows and 13 columns")
print("catalog fixture: 30 rows x 13 columns; SHA-256 verified")
PY

"$bin" --version --json | tee "$home/version.json"
status=$("$bin" serve --status)
[[ "$status" == stopped ]] || fail "expected a stopped isolated engine, got: $status"
printf '%s\n' "$status"

cleanup() {
    if [[ "$server_started" == 1 ]]; then
        "$bin" serve --stop >/dev/null 2>&1 || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"$bin" serve --idle-exit 10m >"$home/engine.log" 2>&1 &
server_pid=$!
server_started=1
for ((attempt = 0; attempt < 100; attempt++)); do
    if status=$("$bin" serve --status 2>/dev/null) && [[ "$status" == running* ]]; then
        printf '%s\n' "$status"
        break
    fi
    kill -0 "$server_pid" 2>/dev/null || fail 'local engine exited before becoming ready'
    sleep 0.1
done
[[ "$server_started" == 1 ]] || fail 'local engine did not become ready within 10 seconds'

daemon_file="$home/daemon.json"
url=$(python3 - "$daemon_file" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["url"])
PY
)
python3 - "$daemon_file" <<'PY' | curl --fail-with-body --silent --show-error --config - "$url/v1/system" >"$home/system-info.json"
import json
import sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
print('header = "Authorization: Bearer ' + record["token"] + '"')
PY
python3 - "$home/system-info.json" <<'PY'
import json
import sys
snapshot = json.load(open(sys.argv[1], encoding="utf-8"))
assert snapshot["health"] == "ok" and snapshot["api_version"] == "1.0"
print("GET /v1/system: HTTP 200; health=ok; api_version=1.0")
PY

"$bin" dataset add "$csv" \
    --name rel05-s3-catalog-30-row \
    --idempotency-key rel05-s3-catalog-30-row-v1 \
    --json | tee "$home/dataset-add.json"
python3 - "$home/dataset-add.json" <<'PY'
import json
import sys
profile = json.load(open(sys.argv[1], encoding="utf-8"))
assert profile["rows"] == "30" and len(profile["columns"]) == 13
assert profile["idempotency_key"] == "rel05-s3-catalog-30-row-v1"
assert profile["name"] == "rel05-s3-catalog-30-row"
print("dataset add: " + profile["dataset_id"] + " (30 rows, 13 columns; requested name preserved)")
PY

http_status=$(python3 - "$daemon_file" <<'PY' | curl --fail-with-body --silent --show-error --config - \
    --output "$home/http-dataset-replay.json" --write-out '%{http_code}' \
    -X POST -H 'Content-Type: text/csv' \
    -H 'Idempotency-Key: rel05-s3-catalog-30-row-v1' \
    --data-binary "@$csv" "$url/v1/datasets?name=rel05-s3-catalog-30-row"
import json
import sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
print('header = "Authorization: Bearer ' + record["token"] + '"')
PY
)
[[ "$http_status" == 201 ]] || fail "expected idempotent HTTP registration replay to return 201, got $http_status"
python3 - "$home/http-dataset-replay.json" <<'PY'
import json
import sys
profile = json.load(open(sys.argv[1], encoding="utf-8"))
assert profile["rows"] == "30" and len(profile["columns"]) == 13
assert profile["name"] == "rel05-s3-catalog-30-row"
print("POST /v1/datasets CSV replay: HTTP 201; 30 rows, 13 columns; requested name preserved")
PY

set +e
"$bin" dataset list --json >"$home/dataset-list.stdout" 2>"$home/dataset-list.stderr"
list_status=$?
set -e
[[ "$list_status" == 2 ]] || fail "expected dataset list to report the unavailable dataset_inspect route with exit 2, got $list_status"
grep -Fq 'Operation dataset_inspect is not implemented in this build.' "$home/dataset-list.stderr" \
    || fail 'dataset list did not report the expected capability error'
printf 'dataset list: expected capability-unavailable error (exit 2)\n'

"$bin" serve --stop
wait "$server_pid"
server_started=0
status=$("$bin" serve --status)
[[ "$status" == stopped ]] || fail "expected stopped engine after cleanup, got: $status"
printf '%s\n' "$status"
