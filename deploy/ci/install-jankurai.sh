#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p target/jankurai
archive="target/jankurai/jankurai-1.7.5-x86_64-unknown-linux-gnu.tar.gz"
if [[ ! -f "$archive" ]]; then
  curl --fail --location --silent --show-error --retry 3 \
    https://github.com/neverhuman/jankurai-audit/releases/download/v1.7.5/jankurai-1.7.5-x86_64-unknown-linux-gnu.tar.gz \
    --output "$archive"
fi
echo "792c16465c274a266f64541152eb5c796fdf22113120133789d28bf99bcf915c  $archive" | sha256sum --check --status
tar -xzf "$archive" -C target/jankurai
test "$(target/jankurai/jankurai-1.7.5-x86_64-unknown-linux-gnu/jankurai --version)" = "jankurai 1.7.5"
