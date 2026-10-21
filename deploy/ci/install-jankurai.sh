#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p target/jankurai
version=1.7.5
digest=792c16465c274a266f64541152eb5c796fdf22113120133789d28bf99bcf915c
archive="target/jankurai/jankurai-$version-x86_64-unknown-linux-gnu.tar.gz"
if [[ -e "$archive" || -L "$archive" ]]; then
  # Cached bytes must be a regular file, never a borrowed path or another object type.
  [[ -f "$archive" && ! -L "$archive" ]] || { echo 'refusing non-regular archive cache entry' >&2; exit 1; }
else
  # Interrupted downloads must not become a permanent, corrupt cached release.
  download=$(mktemp "$archive.XXXXXX")
  trap 'rm -f "$download"' EXIT
  curl --proto '=https' --tlsv1.2 --fail --location --silent --show-error --retry 3 \
    "https://github.com/neverhuman/jankurai-audit/releases/download/v$version/jankurai-$version-x86_64-unknown-linux-gnu.tar.gz" \
    --output "$download"
  echo "$digest  $download" | sha256sum --check --status
  # Both names are on the same filesystem. Publish verified bytes without replacing any
  # directory, FIFO, symlink or competing cache file created while the download ran.
  ln -T -- "$download" "$archive"
  rm -f -- "$download"
  trap - EXIT
fi
echo "$digest  $archive" | sha256sum --check --status
# Release archives carry the builder's uid/gid; root in a user namespace cannot chown to them.
tar --no-same-owner -xzf "$archive" -C target/jankurai
binary="target/jankurai/jankurai-$version-x86_64-unknown-linux-gnu/jankurai"
test "$("$binary" --version)" = "jankurai $version"
