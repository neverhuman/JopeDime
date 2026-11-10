#!/bin/bash
# xbabe4 guard for the H419 addendum-4 jobs: stop them if the host's free memory falls below 15% (the Mac also runs a
# VM and user applications that must not be disturbed).   x4_guard.sh LOG PIDFILE...
LOG=${1:?}; shift
while :; do
  free=$(memory_pressure 2>/dev/null | awk '/free percentage/ {gsub("%","",$NF); print $NF}')
  alive=0; for f in "$@"; do p=$(cat "$f" 2>/dev/null); [ -n "$p" ] && kill -0 "$p" 2>/dev/null && alive=1; done
  [ "$alive" = 1 ] || { echo "$(date -u +%FT%TZ) no job alive: guard exits" >> "$LOG"; exit 0; }
  if [ -n "$free" ] && [ "$free" -lt 15 ]; then
    for f in "$@"; do p=$(cat "$f" 2>/dev/null); [ -n "$p" ] && kill -TERM "$p" 2>/dev/null; done
    echo "$(date -u +%FT%TZ) free memory $free% < 15%: jobs stopped" >> "$LOG"; exit 0
  fi
  sleep 60
done
