#!/usr/bin/env bash
# Controller regression fixtures only. These mocks never establish isolation or gate D1.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
helper=${1:-$script_dir/test-clean-host-prestage.sh}
base=${JOPE_CLEAN_HOST_CONTROL_TMP:-${XDG_CACHE_HOME:-$HOME/.cache}/jopedime-clean-host-controls}
case $base in /tmp | /tmp/*) echo "controls must not use /tmp" >&2; exit 2 ;; esac
mkdir -p "$base"
root=$(mktemp -d "$base/control.XXXXXX")
mkdir -p "$root/repo/deploy/appliance" "$root/repo/deploy/ci" "$root/tools"
cp -- "$helper" "$root/repo/deploy/appliance/test-clean-host-prestage.sh"

cat > "$root/tools/unshare" <<'SH'
#!/usr/bin/env bash
while [[ $1 == --* ]]; do shift; done
exec "$@"
SH
cat > "$root/tools/ip" <<'SH'
#!/usr/bin/env bash
if [[ $* == '-o link show' ]]; then printf '1: lo: <LOOPBACK,UP>\n'; fi
SH
cat > "$root/tools/git" <<'SH'
#!/usr/bin/env bash
printf 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n'
SH
cat > "$root/repo/deploy/ci/link-audit.sh" <<'SH'
#!/usr/bin/env bash
echo 'FIXTURE link scanner; no actual dependency/isolation claim'
SH
chmod +x "$root/tools/"* "$root/repo/deploy/ci/link-audit.sh"

cat > "$root/mock.template" <<'SH'
#!/bin/bash
set -euo pipefail
mode=FIXTURE_MODE
command=$1
shift
case $command in
  --version) printf '{"package":null}\n' ;;
  methods) printf '{"methods":[{"id":"ridge"}]}\n' ;;
  serve)
    case ${1:-} in
      --status) echo running ;;
      --stop) exit 0 ;;
      *) exec /bin/sleep 60 ;;
    esac
    ;;
  dataset)
    if [[ $mode == dataset-refusal ]]; then
      echo 'Operation dataset_register is not implemented in this build' >&2
      exit 2
    fi
    printf '{"dataset_id":"ds_fixture"}\n'
    ;;
  run)
    if [[ $mode == run-refusal ]]; then
      echo 'Operation run_start is not implemented in this build' >&2
      exit 2
    fi
    if [[ $mode == success ]]; then
      while [[ ${1:-} != --target ]]; do shift; done
      [[ $2 == training_label ]] || exit 9
    fi
    printf '{"state":"completed","run_id":"run_fixture","best":{}}\n'
    ;;
  export)
    language= output=
    while (($#)); do
      case $1 in
        --lang) language=$2; shift 2 ;;
        --out) output=$2; shift 2 ;;
        *) shift ;;
      esac
    done
    if [[ $mode == export-refusal && $language == rust ]]; then exit 3; fi
    printf 'fixture export\n' > "$output"
    printf '{}\n'
    ;;
  predict)
    while [[ $1 != --output ]]; do shift; done
    printf 'prediction\n1\n2\n' > "$2"
    if [[ $mode != short-prediction ]]; then printf '3\n' >> "$2"; fi
    ;;
  fit) echo 'unexpected local fit substitution' >&2; exit 91 ;;
  *) echo "unexpected fixture command $command" >&2; exit 92 ;;
esac
SH

passed=0
failed=0
check_case() {
  local mode=$1 expected_exit=$2 predicate=$3
  shift 3
  local work="$root/$mode"
  sed "s/FIXTURE_MODE/$mode/" "$root/mock.template" > "$root/mock-$mode"
  chmod +x "$root/mock-$mode"
  local status=0
  PATH="$root/tools:$PATH" bash "$root/repo/deploy/appliance/test-clean-host-prestage.sh" \
    "$root/mock-$mode" "$work" "$@" > "$root/$mode.stdout" 2> "$root/$mode.stderr" || status=$?
  if [[ $status == "$expected_exit" && -f $work/clean-host.json ]] && \
    jq -e "$predicate and (.release_gate_d1 == \"not qualified\")" "$work/clean-host.json" >/dev/null; then
    printf 'PASS %s (fixture only)\n' "$mode"
    passed=$((passed + 1))
  else
    printf 'FAIL %s: exit=%s expected=%s receipt=%s\n' "$mode" "$status" "$expected_exit" "$work/clean-host.json" >&2
    failed=$((failed + 1))
  fi
}

# First case reproduces the original lost-receipt failure on a refused run_start.
check_case run-refusal 2 '(.status == "failed" and .failed_step == "run_start" and
  .dataset_registration == "passed" and .run_discovery.status == "failed" and
  .prediction_rows == null and .export_status.python == "not-run" and
  .prestage_checks_passed == false and .link_audit == "passed")'
check_case export-refusal 3 '(.status == "failed" and .failed_step == "export_rust" and
  .run_discovery.status == "passed" and .export_status.python == "passed" and
  .export_status.rust == "failed" and .export_status.javascript == "not-run")'
check_case short-prediction 1 '(.status == "failed" and .failed_step == "prediction_rows" and
  .prediction_rows == 2 and .prestage_checks_passed == false)'
check_case dataset-refusal 1 '(.status == "failed" and .failed_step == "dataset_register" and
  .run_discovery.status == "not-run" and .export_mode == "run")' --require-golden-path
printf 'x,training_label\n1,2\n2,4\n3,6\n' > "$root/train.csv"
printf 'x\n1\n2\n3\n' > "$root/predict.csv"
check_case success 0 '(.status == "passed" and .failed_step == null and
  .prestage_checks_passed == true and .input.kind == "supplied" and
  .input.target == "training_label" and .prediction_rows == 3 and
  .run_discovery.status == "passed" and .export_status.javascript == "passed")' \
  --require-golden-path --csv "$root/train.csv" training_label "$root/predict.csv"
printf 'Controller fixture controls: %s passed, %s failed; retained at %s\n' "$passed" "$failed" "$root"
[[ $failed == 0 ]]
