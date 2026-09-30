#!/usr/bin/env bash
# One encrypted POSIX tar shard, streamed directly from source to the USB archive.
# The source is never removed here. A failed transfer retains its .partial for inspection.
set -euo pipefail
root=${1:?usage: jope-usb-shard.sh ROOT RELATIVE DESTINATION KEY_FILE}
relative=${2:?usage: jope-usb-shard.sh ROOT RELATIVE DESTINATION KEY_FILE}
destination=${3:?usage: jope-usb-shard.sh ROOT RELATIVE DESTINATION KEY_FILE}
key_file=${4:?usage: jope-usb-shard.sh ROOT RELATIVE DESTINATION KEY_FILE}
test -d "$root"
test -e "$root/$relative"
test -f "$key_file"
[[ "$relative" != /* && "$relative" != *..* && "$relative" != *$'\n'* ]]
mkdir -p -- "$destination"
identity=$(printf '%s\0%s' "$root" "$relative" | sha256sum | cut -c1-16)
name=$(printf '%s' "$relative" | tr '/' '_' | tr -cd 'A-Za-z0-9._-')
archive="$destination/$name-$identity.tar.zst.gpg"
partial="$archive.partial"
test ! -e "$archive"
test ! -e "$partial"
started=$(date -u +%FT%TZ)
before=$(df -B1 --output=avail / | tail -1 | tr -d ' ')
tar --acls --xattrs --selinux --numeric-owner -C "$root" -cf - -- "$relative" |
  zstd -T2 -3 -q |
  gpg --batch --yes --pinentry-mode loopback --no-symkey-cache --compress-algo none \
    --cipher-algo AES256 --passphrase-file "$key_file" --symmetric --output "$partial"
mv -- "$partial" "$archive"
sha256sum "$archive" > "$archive.sha256"
gpg --batch --quiet --pinentry-mode loopback --passphrase-file "$key_file" \
  --decrypt --output - "$archive" | zstd -dc -q | tar -tf - > "$archive.entries"
after=$(df -B1 --output=avail / | tail -1 | tr -d ' ')
jq -n --arg root "$root" --arg relative "$relative" --arg archive "$archive" \
  --arg started "$started" --arg finished "$(date -u +%FT%TZ)" \
  --argjson bytes "$(stat -c %s "$archive")" \
  --argjson entries "$(wc -l < "$archive.entries")" \
  --argjson root_free_before "$before" --argjson root_free_after "$after" \
  '{schema:"jope.usb-shard/v1",root:$root,relative:$relative,archive:$archive,
    started:$started,finished:$finished,archive_bytes:$bytes,entries:$entries,
    root_free_before:$root_free_before,root_free_after:$root_free_after,
    full_readback:true,source_deleted:false}' > "$archive.receipt.json"
cat "$archive.receipt.json"
