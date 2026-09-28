#!/usr/bin/env bash
# deploy/ci/ip-scrub.sh — the shipped-text rule, applied to prose.
#
# The product story is "the model reads your data and writes a program". Nothing public may say
# how the program is found. Emitted code, manifests and every backend are covered by
# `crates/feat-math/tests/ip_scrub.rs` (`lint_shipped_text`); this script covers the prose the
# repository publishes: README.md, docs/*.md, the runbooks, SECURITY.md, CONTRIBUTING.md and the
# `description =` line of every product crate.
#
# Word-bounded and case-insensitive. Inline code spans (`...`) and fenced code blocks are stripped
# first: identifiers, file names and commands are not prose and legacy module names must keep
# their spelling. Exemptions are listed in deploy/ci/ip-scrub-exempt.txt, one path per line, each
# with the reason it is exempt; historical records (docs/adr, docs/legacy, docs/parity,
# docs/coordination) are never scanned.
set -euo pipefail
cd "$(dirname "$0")/../.."

pattern='\bGP\b|genome|genetic|evolut|evolv|\bpopulation|chromosome|mutation|crossover|lexicase|fitness|keijzer|koza|expression tree|parse tree|syntax tree|symbolic regression'

exempt=deploy/ci/ip-scrub-exempt.txt
is_exempt() {
  [ -f "$exempt" ] && grep -vE '^\s*(#|$)' "$exempt" | cut -d' ' -f1 | grep -qxF "$1"
}

files=()
for f in README.md SECURITY.md CONTRIBUTING.md docs/*.md docs/runbooks/*.md; do
  [ -f "$f" ] && files+=("$f")
done
# Product crate descriptions (the list deploy/ci/rust-test-scope.sh maintains).
while read -r pkg; do
  manifest=$(cargo metadata --locked --format-version 1 --no-deps 2>/dev/null \
    | python3 -c 'import json,sys; pkg=sys.argv[1]; m=json.load(sys.stdin); print(next((p["manifest_path"] for p in m["packages"] if p["name"]==pkg), ""))' "$pkg")
  [ -n "$manifest" ] && files+=("$(realpath --relative-to=. "$manifest")")
done < <(bash deploy/ci/rust-test-scope.sh | tr ' ' '\n' | grep -v '^-p$' | grep -v '^$')

fail=0
for f in "${files[@]}"; do
  if is_exempt "$f"; then continue; fi
  case "$f" in
    *Cargo.toml) text=$(grep -n '^description' "$f" || true) ;;
    *)
      # Drop fenced code blocks, then inline code spans, keeping line numbers.
      text=$(awk 'BEGIN{fence=0} /^```/{fence=!fence; print NR":"; next} {if(fence){print NR":"} else {print NR":"$0}}' "$f" \
        | sed -E 's/`[^`]*`//g') ;;
  esac
  hits=$(printf '%s\n' "$text" | grep -inE "$pattern" || true)
  if [ -n "$hits" ]; then
    echo "$f:"; printf '%s\n' "$hits" | sed 's/^/  /' | cut -c1-200
    fail=1
  fi
done
if [ "$fail" = 1 ]; then
  echo "ip-scrub: the prose above reveals how programs are found; reword it (see deploy/ci/ip-scrub.sh)" >&2
  exit 1
fi
echo "ip-scrub: ${#files[@]} files clean"
