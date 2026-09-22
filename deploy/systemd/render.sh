#!/usr/bin/env bash
# deploy/systemd/render.sh — render the unit templates (deploy/systemd/**/*.in)
# into a directory, substituting @JOPE_HOME@ @JOPE_STATE@ @JOPE_BUNDLE@ @JOPE_USER@.
#
#   render.sh --home <dir> --state <dir> --bundle <dir> --user <name> --out <dir>
#
# Thin wrapper over `jope-units-render` (crates/jope-paths), which is the one
# implementation of the substitution; the test crates/jope-campaigns/tests/
# systemd_unit_paths.rs renders the same templates through the same code.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../.." && pwd)

usage() {
  echo "usage: $0 --home <dir> --state <dir> --bundle <dir> --user <name> --out <dir>" >&2
  exit 2
}

home="" state="" bundle="" user="" out=""
while [ $# -gt 0 ]; do
  case "$1" in
    --home) home=${2:-}; shift 2 ;;
    --state) state=${2:-}; shift 2 ;;
    --bundle) bundle=${2:-}; shift 2 ;;
    --user) user=${2:-}; shift 2 ;;
    --out) out=${2:-}; shift 2 ;;
    -h|--help) usage ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done
[ -n "$home" ] && [ -n "$state" ] && [ -n "$bundle" ] && [ -n "$user" ] && [ -n "$out" ] || usage

# The renderer: an explicitly named binary, one on PATH, or Cargo
# (which reuses a cached build when fresh, regardless of its target directory).
renderer=${JOPE_UNITS_RENDER:-}
if [ -z "$renderer" ]; then
  if command -v jope-units-render >/dev/null 2>&1; then
    renderer=$(command -v jope-units-render)
  else
    renderer=$(python3 "$repo/deploy/ci/cargo-artifact.py" build \
      --manifest-path "$repo/Cargo.toml" --package jope-paths --bin jope-units-render)
  fi
fi

exec "$renderer" --templates "$here" --home "$home" --state "$state" --bundle "$bundle" --user "$user" --out "$out"
