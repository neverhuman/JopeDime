#!/usr/bin/env bash
# Report missing prerequisites; never install or mutate a host.
set -euo pipefail
cd "$(dirname "$0")/.."
failed=0
for tool in git bash python3 jq rustup cargo rustc node npm uv g++ clang++ actionlint gitleaks shellcheck; do
  if command -v "$tool" >/dev/null; then printf 'present %s\n' "$tool"; else printf 'missing %s\n' "$tool" >&2; failed=1; fi
done
for tool in nextest deny; do
  if ! cargo "$tool" --version; then failed=1; fi
done
pinned=$(sed -n 's/^channel = "\(.*\)"/\1/p' rust-toolchain.toml)
actual=$(rustc --version 2>/dev/null || true)
if [[ "$actual" != "rustc $pinned "* ]]; then printf 'expected rustc %s, found %s\n' "$pinned" "$actual" >&2; failed=1; fi
exit "$failed"
