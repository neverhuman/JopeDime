#!/usr/bin/env bash
# deploy/ci/rust-test-filter.sh — one source of truth for the nextest filter expressions the
# `rust-test` CI lane and the justfile share, so the two can never drift.
#
#   (no args)   the blocking set: everything except the known-broken tests and the serial binaries
#   --broken    exactly the known-broken tests (deploy/ci/known-broken-tests.txt)
#   --serial    exactly the serial-only binaries (deploy/ci/serial-test-binaries.txt)
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

broken_terms() {
  while IFS= read -r line; do
    line=${line%%#*}; line=$(echo "$line" | xargs || true); [ -z "$line" ] && continue
    echo "(binary(${line%% *}) and test(=${line#* }))"
  done < "$here/known-broken-tests.txt"
}
serial_terms() {
  while IFS= read -r line; do
    line=${line%%#*}; line=$(echo "$line" | xargs || true); [ -z "$line" ] && continue
    echo "binary($line)"
  done < "$here/serial-test-binaries.txt"
}
join_or() { local out=""; while IFS= read -r t; do [ -z "$t" ] && continue; out="${out:+$out or }$t"; done; echo "$out"; }

case "${1:-}" in
  --broken) b=$(broken_terms | join_or); echo "${b:-none()}" ;;
  --serial) s=$(serial_terms | join_or); echo "${s:-none()}" ;;
  "") b=$(broken_terms | join_or); s=$(serial_terms | join_or)
      both=""; [ -n "$b" ] && both="$b"; [ -n "$s" ] && both="${both:+$both or }$s"
      [ -z "$both" ] && echo 'all()' || echo "not ($both)" ;;
  *) echo "usage: $0 [--broken|--serial]" >&2; exit 2 ;;
esac
