#!/usr/bin/env bash
# deploy/ci/parity-matrix.sh — capability-parity guard (plan: red-team pass 2 item 6, WP10).
#
# Fails when docs/parity/capability-matrix.md has a row with no target and no dropped/lost/pending
# status, when a binary, clap subcommand, example, Python tool, jope-py command or systemd unit
# exists in this tree without a row, when a pending-WP<N> number is outside 4..12, or when a
# moved row points at a missing path. Prints counts by status and by kind.
#
# The clap subcommands are taken from the source enum (crates/jope-engine/src/main.rs, which also
# lists the `neural`-gated and hidden variants) and, unless --no-build is given, cross-checked
# against the parsed `cargo run -q -p jope-parallel -- --help` output. The hygiene lane passes
# --no-build; the test lanes run the full form once the engine is built.
set -euo pipefail
cd "$(dirname "$0")/../.."

build=1
for arg in "$@"; do
  case "$arg" in
    --no-build) build=0 ;;
    -h|--help) echo "usage: $0 [--no-build]"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

args=(check --matrix docs/parity/capability-matrix.md --root .)
help_file=""
if [ "$build" = 1 ]; then
  help_file=$(mktemp)
  trap 'rm -f "$help_file"' EXIT
  if ! cargo run -q -p jope-parallel --bin jope-parallel --locked -- --help > "$help_file"; then
    echo "cargo run -q -p jope-parallel -- --help failed" >&2
    exit 1
  fi
  echo "jope-parallel --help lists $(awk '/^Commands:/{f=1;next} /^Options:/{f=0} f && /^  [a-z]/ && $1!="help"{n++} END{print n+0}' "$help_file") visible subcommands"
  args+=(--help-text "$help_file")
fi

python3 deploy/ci/parity-census.py "${args[@]}"
