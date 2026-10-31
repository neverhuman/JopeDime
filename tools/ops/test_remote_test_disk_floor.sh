#!/usr/bin/env bash
# shellcheck disable=SC1090
set -euo pipefail
SCRIPT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/remote_test.sh"
source <(sed -n '/^min_free_gb(){/,/^}/p' "$SCRIPT")
test "$(min_free_gb xbabe1)" = 100
test "$(min_free_gb xbabe3)" = 60
test "$(min_free_gb xbabe2)" = 40
test "$(min_free_gb xbabe4)" = 40
printf 'remote_test disk floors: PASS\n'
