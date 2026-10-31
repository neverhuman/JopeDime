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
  printf 'Dry run only. Re-run with --enable on xbabe1, xbabe2, or xbabe3 after the reviewed PR merges and its local dry-run passes.\n'
  exit 0
fi

REPO_ROOT=$(git -C "$ROOT" rev-parse --show-toplevel) || {
  printf 'Refusing enable: installer must run from the merged JopeDime repository.\n' >&2
  exit 1
}
for path in "$HOME/.local" "$HOME/.local/bin" "$HOME/.local/state" "$STATE_DIR" "$HOME/.config" "$CONFIG_DIR" "$BIN" "$CONFIG" "$HOME/.config/environment.d" "$HOME/.config/environment.d/99-jope-sccache.conf" "$HOME/.profile" "$HOME/.config/systemd" "$HOME/.config/systemd/user" "$HOME/.config/systemd/user/jope-diskd.service" "$HOME/.config/systemd/user/jope-diskd.timer"; do
  if [[ -L "$path" ]]; then
    printf 'Refusing to install through symlinked path %s.\n' "$path" >&2
    exit 1
  fi
done
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain -- "$ROOT")" ]] || ! git -C "$REPO_ROOT" merge-base --is-ancestor HEAD origin/main; then
  printf 'Refusing enable: tools/ops is dirty or HEAD is not contained in origin/main. Deploy only the merged implementation.\n' >&2
  exit 1
fi

case "$(uname -s)" in
  Linux)
    case "$(hostname -s)" in
      xbabe1|xbabe2|xbabe3) ;;
      *)
        printf 'Refusing to enable local-only disk policy on unsupported host %s.\n' "$(hostname -s)" >&2
        exit 1
        ;;
    esac
    command -v systemctl >/dev/null
    command -v loginctl >/dev/null
    python3 - "$ROOT/disk-policy.yaml" "$(hostname -s)" <<'PY'
import json
import sys
from pathlib import Path

policy = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if sys.argv[2] not in policy.get("allowed_hosts", []):
    raise SystemExit(f"Refusing to install policy that does not include this host: {sys.argv[2]}")
PY
    linger=$(loginctl show-user "$(id -un)" -p Linger --value)
    if [[ "$linger" != yes ]]; then
      printf 'Refusing to enable timer: user linger is not enabled. Ask the host owner to provision linger.\n' >&2
      exit 1
    fi
    ;;
  Darwin)
    printf 'Refusing to enable the Linux-only local disk policy on Darwin.\n' >&2
    exit 1
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
    ENV_DIR="$HOME/.config/environment.d"
    ENV_FILE="$ENV_DIR/99-jope-sccache.conf"
    if [[ -L "$HOME/.config" || -L "$ENV_DIR" || -L "$ENV_FILE" ]]; then
      printf 'Refusing to write through a symlinked sccache environment path.\n' >&2
      exit 1
    fi
    if [[ ! -d "$ENV_DIR" ]]; then
      install -d -m 0700 "$ENV_DIR"
    fi
    if [[ ! -f "$ENV_FILE" ]] || [[ "$(cat -- "$ENV_FILE")" != 'SCCACHE_CACHE_SIZE=60G' ]]; then
      backup_existing "$ENV_FILE"
      printf 'SCCACHE_CACHE_SIZE=60G\n' > "$ENV_FILE"
      chmod 0644 "$ENV_FILE"
    fi
    PROFILE="$HOME/.profile"
    if [[ -L "$PROFILE" ]]; then
      printf 'Refusing to write through a symlinked %s.\n' "$PROFILE" >&2
      exit 1
    fi
    PROFILE_LINE='export SCCACHE_CACHE_SIZE=60G'
    if ! grep -Fqx -- "$PROFILE_LINE" "$PROFILE" 2>/dev/null; then
      backup_existing "$PROFILE"
      printf '\n# JopeDime DISK-01 sccache cap\n%s\n' "$PROFILE_LINE" >> "$PROFILE"
    fi
    systemctl --user set-environment SCCACHE_CACHE_SIZE=60G
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
esac
