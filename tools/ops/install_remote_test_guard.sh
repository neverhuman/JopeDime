#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "$ROOT" rev-parse --show-toplevel)
TARGET="$HOME/bin/remote_test.sh"

if [[ "$(hostname -s)" != xbabe2 ]]; then
  printf 'Refusing to update the host-local runner on %s; expected xbabe2.\n' "$(hostname -s)" >&2
  exit 1
fi
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain -- "$ROOT")" ]] || ! git -C "$REPO_ROOT" merge-base --is-ancestor HEAD origin/main; then
  printf 'Refusing to update the runner: tools/ops is dirty or HEAD is not merged into origin/main.\n' >&2
  exit 1
fi
if [[ -L "$HOME/bin" || -L "$TARGET" || ! -d "$HOME/bin" ]]; then
  printf 'Refusing to update through a symlink or missing bin directory.\n' >&2
  exit 1
fi

BACKUP_DIR="$HOME/.local/state/jope-diskd/remote-test-backups/$(date -u +%Y%m%dT%H%M%SZ)"
install -d -m 0700 "$BACKUP_DIR"
if [[ -e "$TARGET" ]]; then
  cp -p -- "$TARGET" "$BACKUP_DIR/remote_test.sh"
fi
TEMP="$HOME/bin/.remote_test.sh.$$"
trap 'rm -f -- "$TEMP"' EXIT
install -m 0755 "$ROOT/remote_test.sh" "$TEMP"
mv -f -- "$TEMP" "$TARGET"
trap - EXIT
cmp -s "$ROOT/remote_test.sh" "$TARGET"
printf 'Installed tracked runner to %s; previous copy backed up at %s\n' "$TARGET" "$BACKUP_DIR"
