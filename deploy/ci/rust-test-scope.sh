#!/usr/bin/env bash
# deploy/ci/rust-test-scope.sh — the crates whose tests the hosted `rust-test` lane gates.
#
#   (no args)   the -p flags for the hosted (product) scope
#   --deferred  the -p flags for the crates a hosted runner cannot build (see below)
#
# WHY A SCOPE AND NOT THE WHOLE WORKSPACE
# `cargo test` LINKS, unlike `cargo check`. Linking every test binary in this workspace produces a
# 72 GB target directory (measured 2026-09-21); a standard ubuntu-24.04 runner has roughly 14 GB
# free, so the link dies with `collect2: fatal error: ld terminated with signal 7 [Bus error]` and
# no test runs at all. The product scope below fits: measured on hosted run 35586778652, it links
# to 8.98 GB against 11.5 GB free (3.9 GB left afterwards) and executes 651 tests in 626 s. The
# full product inventory is 664 = 651 executed + 9 ignored + 4 filtered by the `pr` profile.
#
# The deferred crates (jope-parallel, jope-dime-regressor, the vendored smartcluster set) are the
# heavy ones, and they are also where every currently-broken and every race-sensitive test lives:
# deploy/ci/known-broken-tests.txt and deploy/ci/serial-test-binaries.txt describe them. They
# belong on the self-hosted runner lane, which has the disk for them. Until that lane exists they
# are genuinely untested on pull requests, which this file states rather than hides.
set -euo pipefail

product=(feat-math jope-cli jope-datasets jope-models starforge jope-policy hyperion numpy-pcg64 jope-paths battle-gpu jope-arena)
deferred=(jope-parallel jope-dime-regressor jain-smartcluster jain-smartcluster-core jain-smartcluster-fabric jain-smartcluster-daemon jain-worker jainnode scq)

# deploy/ci/check.sh asserts that product + deferred is exactly the workspace, so a crate added
# later cannot silently land in neither list (which is how jope-arena slipped through once).
case "${1:-}" in
  --deferred) set -- "${deferred[@]}" ;;
  "")         set -- "${product[@]}" ;;
  *) echo "usage: $0 [--deferred]" >&2; exit 2 ;;
esac
[ "$#" -gt 0 ] || { echo "$0: empty package set" >&2; exit 3; }
printf -- '-p %s ' "$@"
echo
