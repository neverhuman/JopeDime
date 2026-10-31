#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ENABLE=0
if [[ "${1:-}" == "--enable" ]]; then
  ENABLE=1
elif [[ $# -gt 0 ]]; then
  printf 'Usage: %s [--enable]\n' "$0" >&2
  exit 2
fi

BIN_DIR="$HOME/.local/bin"
CONFIG_DIR="$HOME/.config/jope-diskd"
STATE_DIR="$HOME/.local/state/jope-diskd"
BIN="$BIN_DIR/jope-diskd"
CONFIG="$CONFIG_DIR/policy.yaml"

printf 'Install jope-diskd to %s and policy to %s\n' "$BIN" "$CONFIG"
if [[ $ENABLE -eq 0 ]]; then
  printf 'Dry run only. Re-run with --enable after the reviewed PR is merged and all host dry-runs pass.\n'
  exit 0
fi

REPO_ROOT=$(git -C "$ROOT" rev-parse --show-toplevel) || {
  printf 'Refusing enable: installer must run from the merged JopeDime repository.\n' >&2
  exit 1
}
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain -- "$ROOT")" ]] || ! git -C "$REPO_ROOT" merge-base --is-ancestor HEAD origin/main; then
  printf 'Refusing enable: tools/ops is dirty or HEAD is not contained in origin/main. Deploy only the merged implementation.\n' >&2
  exit 1
fi

case "$(uname -s)" in
  Linux)
    command -v systemctl >/dev/null
    command -v loginctl >/dev/null
    linger=$(loginctl show-user "$(id -un)" -p Linger --value)
    if [[ "$linger" != yes ]]; then
      printf 'Refusing to enable timer: user linger is not enabled. Ask the host owner to provision linger.\n' >&2
      exit 1
    fi
    ;;
  Darwin)
    command -v crontab >/dev/null
    ;;
  *)
    printf 'Unsupported scheduler host: %s\n' "$(uname -s)" >&2
    exit 1
    ;;
esac

install -d -m 0700 "$BIN_DIR" "$CONFIG_DIR" "$STATE_DIR"
backup_existing() {
  local path="$1"
  if [[ -e "$path" ]]; then
    local backup_dir
    backup_dir="$STATE_DIR/install-backup-$(date -u +%Y%m%dT%H%M%SZ)"
    install -d -m 0700 "$backup_dir"
    cp -p "$path" "$backup_dir/$(basename "$path")"
  fi
}
backup_existing "$BIN"
backup_existing "$CONFIG"
install -m 0755 "$ROOT/jope_diskd.py" "$BIN"
install -m 0600 "$ROOT/disk-policy.yaml" "$CONFIG"

case "$(uname -s)" in
  Linux)
    UNIT_DIR="$HOME/.config/systemd/user"
    install -d -m 0700 "$UNIT_DIR"
    backup_existing "$UNIT_DIR/jope-diskd.service"
    backup_existing "$UNIT_DIR/jope-diskd.timer"
    install -m 0644 "$ROOT/systemd/jope-diskd.service" "$UNIT_DIR/jope-diskd.service"
    install -m 0644 "$ROOT/systemd/jope-diskd.timer" "$UNIT_DIR/jope-diskd.timer"
    systemctl --user daemon-reload
    systemctl --user enable --now jope-diskd.timer
    systemctl --user status jope-diskd.timer --no-pager
    ;;
  Darwin)
    LOG_DIR="$HOME/Library/Logs"
    install -d -m 0700 "$LOG_DIR"
    old_crontab=$(crontab -l 2>/dev/null || true)
    if ! printf '%s\n' "$old_crontab" | grep -Fq "$BIN --config $CONFIG --apply"; then
      line="*/10 * * * * /usr/bin/nice -n 10 $BIN --config $CONFIG --apply >> $LOG_DIR/jope-diskd.log 2>&1 # jope-diskd managed"
      { printf '%s\n' "$old_crontab"; printf '%s\n' "$line"; } | crontab -
    fi
    crontab -l
    ;;
esac
