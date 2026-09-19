#!/usr/bin/env bash
# deploy/systemd/render.sh — render the unit templates (deploy/systemd/**/*.in)
# into a directory, substituting @JOPE_HOME@ @JOPE_STATE@ @JOPE_BUNDLE@ @JOPE_USER@.
#
#   render.sh --home <dir> --state <dir> --bundle <dir> --user <name> --out <dir>
#   render.sh --install-symlinks <bindir>
#
# `--install-symlinks` creates, in <bindir>, one link per program of the
# `jope-selfplay` and `jope-campaigns` binaries found there (their `--list`
# output; see BINARIES.md), so the historical names the units execute
# (`@JOPE_BUNDLE@/jope_selfplay_queue …`) resolve.
#
# Thin wrapper over `jope-units-render` (crates/jope-paths), which is the one
# implementation of the substitution; the test crates/jope-selfplay/tests/
# systemd_unit_paths.rs renders the same templates through the same code.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../.." && pwd)

usage() {
  echo "usage: $0 --home <dir> --state <dir> --bundle <dir> --user <name> --out <dir>" >&2
  echo "       $0 --install-symlinks <bindir>" >&2
  exit 2
}

install_symlinks() {
  local bindir=$1 binary name
  [ -d "$bindir" ] || { echo "not a directory: $bindir" >&2; exit 2; }
  for binary in jope-selfplay jope-campaigns; do
    [ -x "$bindir/$binary" ] || { echo "missing $bindir/$binary (cargo build --release -p $binary)" >&2; exit 2; }
    "$bindir/$binary" --list | while IFS= read -r name; do
      [ -n "$name" ] || continue
      if [ -e "$bindir/$name" ] && [ ! -L "$bindir/$name" ]; then
        echo "refusing to replace a regular file: $bindir/$name" >&2
        exit 2
      fi
      ln -sfn "$binary" "$bindir/$name"
      echo "$bindir/$name -> $binary"
    done
  done
}

if [ "${1:-}" = "--install-symlinks" ]; then
  [ $# -eq 2 ] || usage
  install_symlinks "$2"
  exit 0
fi

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

# The renderer: an explicitly named binary, one on PATH, one built in this
# checkout, or a fresh build (needs the workspace toolchain).
renderer=${JOPE_UNITS_RENDER:-}
if [ -z "$renderer" ]; then
  if command -v jope-units-render >/dev/null 2>&1; then
    renderer=$(command -v jope-units-render)
  elif [ -x "$repo/target/release/jope-units-render" ]; then
    renderer=$repo/target/release/jope-units-render
  elif [ -x "$repo/target/debug/jope-units-render" ]; then
    renderer=$repo/target/debug/jope-units-render
  else
    cargo build -q --locked --manifest-path "$repo/Cargo.toml" -p jope-paths --bin jope-units-render
    renderer=$repo/target/debug/jope-units-render
  fi
fi

exec "$renderer" --templates "$here" --home "$home" --state "$state" --bundle "$bundle" --user "$user" --out "$out"
