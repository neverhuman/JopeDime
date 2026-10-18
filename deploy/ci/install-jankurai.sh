#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p target/jankurai
archive="target/jankurai/jankurai-1.7.1-x86_64-unknown-linux-gnu.tar.gz"
if [[ ! -f "$archive" ]]; then
  curl --fail --location --silent --show-error --retry 3 \
    https://github.com/neverhuman/jankurai-audit/releases/download/v1.7.1/jankurai-1.7.1-x86_64-unknown-linux-gnu.tar.gz \
    --output "$archive"
fi
echo "d628d9dfc67ce5968889d3bd2beb4636c3c8f1647a1a2a123422df50e2cf9a98  $archive" | sha256sum --check --status
tar -xzf "$archive" -C target/jankurai
test "$(target/jankurai/jankurai-1.7.1-x86_64-unknown-linux-gnu/jankurai --version)" = "jankurai 1.7.1"
