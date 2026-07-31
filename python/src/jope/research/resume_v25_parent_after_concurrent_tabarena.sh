#!/usr/bin/env bash
set -euo pipefail

parent_pid="${1:?parent PID is required}"
golden_pid="${2:?Golden42 child PID is required}"
tab_service="${3:?TabArena service name is required}"
tab_summary="${4:?TabArena summary path is required}"

if [[ ! "$parent_pid" =~ ^[0-9]+$ || ! "$golden_pid" =~ ^[0-9]+$ ]]; then
  echo "controller PIDs must be numeric" >&2
  exit 2
fi

while kill -0 "$golden_pid" 2>/dev/null; do
  sleep 10
done

while systemctl --user is-active --quiet "$tab_service"; do
  sleep 10
done

if [[ ! -f "$tab_summary" ]]; then
  echo "TabArena service stopped without producing $tab_summary; leaving parent stopped" >&2
  exit 1
fi
if ! kill -0 "$parent_pid" 2>/dev/null; then
  echo "original parent controller $parent_pid is no longer alive" >&2
  exit 1
fi

kill -CONT "$parent_pid"
