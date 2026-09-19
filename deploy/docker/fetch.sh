#!/usr/bin/env bash
# jope-fetch <url> <sha256> <dest> — download one file and verify its sha256 before it is used.
# Used by the Dockerfiles in deploy/docker; a mismatch removes the file and fails the build.
set -euo pipefail
url=${1:?url}; expected=${2:?sha256}; dest=${3:?dest}
[[ "$expected" =~ ^[0-9a-f]{64}$ ]] || { echo "jope-fetch: bad sha256 '$expected'" >&2; exit 2; }
curl -fsSL --retry 5 --retry-all-errors --proto '=https' --tlsv1.2 -o "$dest" "$url"
actual=$(sha256sum "$dest" | cut -c1-64)
if [[ "$actual" != "$expected" ]]; then
  rm -f "$dest"
  echo "jope-fetch: sha256 mismatch for $url: expected $expected, got $actual" >&2
  exit 1
fi
echo "jope-fetch: ok $dest $actual"
