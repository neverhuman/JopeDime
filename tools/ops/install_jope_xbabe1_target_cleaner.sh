#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "$ROOT" rev-parse --show-toplevel)
USER_NAME=$(id -un)

if [[ "$(hostname -s)" != xbabe1 || "$USER_NAME" != ubuntu ]]; then
  printf 'Refusing x1 target-cleaner install on host/user %s/%s.\n' "$(hostname -s)" "$USER_NAME" >&2
  exit 1
fi
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain -- "$ROOT")" ]] || ! git -C "$REPO_ROOT" merge-base --is-ancestor HEAD origin/main; then
  printf 'Refusing install: tools/ops is dirty or HEAD is not merged into origin/main.\n' >&2
  exit 1
fi
sudo -n true
for path in /usr/local /usr/local/sbin /etc/systemd /etc/systemd/system /var/log /var/log/jope-diskd-xbabe1 /usr/local/sbin/jope-remote-target-cleaner /etc/systemd/system/jope-xbabe1-target-cleaner.service /etc/systemd/system/jope-xbabe1-target-cleaner.timer; do
  if [[ -L "$path" ]]; then
    printf 'Refusing to install through symlinked path %s.\n' "$path" >&2
    exit 1
  fi
done

sudo install -d -m 0755 /usr/local/sbin
sudo install -d -m 0755 /etc/systemd/system
sudo install -d -m 0750 /var/log/jope-diskd-xbabe1
BACKUP_DIR="/var/lib/jope-diskd-xbabe1/backups/$(date -u +%Y%m%dT%H%M%SZ)"
for path in /usr/local/sbin/jope-remote-target-cleaner /etc/systemd/system/jope-xbabe1-target-cleaner.service /etc/systemd/system/jope-xbabe1-target-cleaner.timer; do
  if [[ -e "$path" ]]; then
    sudo install -d -m 0700 "$BACKUP_DIR"
    sudo cp -a -- "$path" "$BACKUP_DIR/$(basename "$path")"
  fi
done
sudo install -m 0755 "$ROOT/jope_remote_target_cleaner.py" /usr/local/sbin/jope-remote-target-cleaner
sudo install -m 0644 "$ROOT/systemd/jope-xbabe1-target-cleaner.service" /etc/systemd/system/jope-xbabe1-target-cleaner.service
sudo install -m 0644 "$ROOT/systemd/jope-xbabe1-target-cleaner.timer" /etc/systemd/system/jope-xbabe1-target-cleaner.timer
sudo systemctl daemon-reload
sudo systemctl enable --now jope-xbabe1-target-cleaner.timer
sudo systemctl status jope-xbabe1-target-cleaner.timer --no-pager
