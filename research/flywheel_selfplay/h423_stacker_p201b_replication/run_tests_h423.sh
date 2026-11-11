#!/bin/bash
# Run the two test files of deploy_h423.py from a staged code root, each in its own process, and write a summary.
#   run_tests_h423.sh STAGE_DIR OUT_JSON LOG
set -uo pipefail
ST=${1:?stage dir}; OUT=${2:?out json}; LOG=${3:?log}
[ ! -e "$OUT" ] || { echo "$OUT exists (create-once)" >&2; exit 3; }
cd "$ST/code" || exit 2
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
unset JOPE_H423_WINSOR_DESIGNS JOPE_H423_POOLED_ADMISSION
tot=0; fail=0; err=0; okall=true
for f in test_h423.py test_admission_h423.py; do
  echo "== $f $(date -u +%FT%TZ)" >> "$LOG"
  python3 -B "$f" -v >> "$LOG" 2>&1; rc=$?
  n=$(grep -E '^Ran [0-9]+ tests?' "$LOG" | tail -n 1 | gawk '{print $2}')
  last=$(tail -n 1 "$LOG")
  tot=$((tot + ${n:-0}))
  if [ "$rc" != 0 ] || [ "$last" != OK ]; then
    okall=false
    fail=$((fail + $(echo "$last" | grep -oE 'failures=[0-9]+' | cut -d= -f2 || echo 0)))
    err=$((err + $(echo "$last" | grep -oE 'errors=[0-9]+' | cut -d= -f2 || echo 0)))
    [ "$last" = OK ] || [ $((fail + err)) -gt 0 ] || err=$((err + 1))
  fi
done
jq -n --argjson ok "$okall" --argjson tests "$tot" --argjson failures "$fail" --argjson errors "$err" \
  --arg stage "$ST" --arg code "$(sha256sum "$ST/CODE.sha256" | cut -d' ' -f1)" --arg at "$(date -u +%FT%TZ)" \
  '{schema: "jope.h423.tests/v1", at: $at, stage: $stage, stage_code_sha256: $code, files: ["test_h423.py", "test_admission_h423.py"],
    ok: $ok, tests: $tests, failures: $failures, errors: $errors}' > "$OUT"
cat "$OUT"
