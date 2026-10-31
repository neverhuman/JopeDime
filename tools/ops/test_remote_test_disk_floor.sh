#!/usr/bin/env bash
# shellcheck disable=SC1090
set -euo pipefail
SCRIPT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/remote_test.sh"
source <(sed -n '/^min_free_gb(){/,/^}/p' "$SCRIPT")
source <(sed -n '/^auto_hosts(){/,/^}/p' "$SCRIPT")
source <(sed -n '/^host_is_disabled(){/,/^}/p' "$SCRIPT")
test "$(min_free_gb xbabe1)" = 100
test "$(min_free_gb xbabe3)" = 60
test "$(min_free_gb xbabe2)" = 40
test "$(min_free_gb xbabe4)" = 40
test "$(auto_hosts)" = $'xbabe1\nxbabe4'
host_is_disabled xbabe3
host_is_disabled xbabe1 && exit 1
host_is_disabled xbabe4 && exit 1
printf 'remote_test host config: PASS\n'
