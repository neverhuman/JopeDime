#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p target/jankurai
version=1.7.5
digest=792c16465c274a266f64541152eb5c796fdf22113120133789d28bf99bcf915c
archive="target/jankurai/jankurai-$version-x86_64-unknown-linux-gnu.tar.gz"
if [[ ! -f "$archive" ]]; then
  # Interrupted downloads must not become a permanent, corrupt cached release.
  download=$(mktemp "$archive.XXXXXX")
  trap 'rm -f "$download"' EXIT
  curl --proto '=https' --tlsv1.2 --fail --location --silent --show-error --retry 3 \
    "https://github.com/neverhuman/jankurai-audit/releases/download/v$version/jankurai-$version-x86_64-unknown-linux-gnu.tar.gz" \
    --output "$download"
  echo "$digest  $download" | sha256sum --check --status
  mv "$download" "$archive"
  trap - EXIT
fi
echo "$digest  $archive" | sha256sum --check --status
# Release archives carry the builder's uid/gid; root in a user namespace cannot chown to them.
tar --no-same-owner -xzf "$archive" -C target/jankurai
binary="target/jankurai/jankurai-$version-x86_64-unknown-linux-gnu/jankurai"
test "$("$binary" --version)" = "jankurai $version"
