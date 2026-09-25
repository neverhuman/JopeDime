#!/usr/bin/env bash
# Keep pytest-created links inside a non-dereferenced archive, not loose CI evidence.
set -euo pipefail
evidence=${JOPE_LANE_EVIDENCE:?fresh receipt directory required}
fixtures="${evidence}-pytest-fixtures"
[[ ! -e $fixtures && ! -L $fixtures ]] || {
  echo "pytest fixture path already exists: $fixtures" >&2
  exit 2
}
status=0
"$@" --basetemp="$fixtures" --junitxml="$evidence/junit.xml" || status=$?
if [[ -d $fixtures ]]; then
  tar -czf "$evidence/pytest-fixtures.tar.gz" -C "$fixtures" . || {
    echo 'failed to retain pytest fixtures' >&2
    (( status != 0 )) || status=1
  }
fi
exit "$status"
