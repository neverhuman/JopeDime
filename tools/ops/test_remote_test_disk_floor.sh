#!/usr/bin/env bash
# shellcheck disable=SC1090,SC2016
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
test "$(grep -Fc 'CARGO_INCREMENTAL=0' "$SCRIPT")" -eq 3
test "$(grep -Fc 'CARGO_PROFILE_DEV_DEBUG=line-tables-only' "$SCRIPT")" -eq 3
grep -Fq 'S=$HOME/remote_tests/src' "$SCRIPT"
grep -Fq 'CARGO_TARGET_DIR="$T"' "$SCRIPT"
grep -Fq 'CARGO_TARGET_DIR=/target' "$SCRIPT"
grep -Fq -- '-v jope_source:/workspace -v jope_target:/target' "$SCRIPT"
grep -Fq '.remote-test.lock' "$SCRIPT"
if grep -Fq 'remote_tests/src.$$' "$SCRIPT"; then exit 1; fi
grep -Fq 'commit-tree "$TREE"' "$SCRIPT"
grep -Fq 'config jope.sourceSha' "$SCRIPT"
grep -Fq 'fetch -q \"\$B\" $REF' "$SCRIPT"
grep -Fq 'checkout -q --force FETCH_HEAD' "$SCRIPT"
if grep -Fq 'git archive' "$SCRIPT"; then exit 1; fi
grep -Fq -- '-e CARGO_TARGET_DIR=/target -e CARGO_INCREMENTAL=0 -e CARGO_PROFILE_DEV_DEBUG=line-tables-only' "$SCRIPT"
printf 'remote_test host config: PASS\n'
printf 'remote_test Cargo write profile: PASS\n'
