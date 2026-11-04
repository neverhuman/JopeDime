#!/bin/bash
# H335 deny-by-default bubblewrap sandbox on xbabe2 (= H317 sandbox_h317.sh + the H335 code dir and data root).
# Usage: sandbox_h335.sh [--net] <command...>     env: H335_VIEWS_RW=1 only for the P201b SEARCH-view encoding;
#        H335_CODE=<frozen read-only copy> is bound at the H335 source path (Phase T runs from a staged copy)
#   * /home/ubuntu, /tmp, /mnt, /media, /srv, /app, /review, /Users, /opt/work are empty tmpfs;
#   * bound back read-only: ~/.local, the FROZEN H317 source dir, the H335 source dir, JopeDimeMicro (benchmark_data
#     hidden), the TRAIN-only res-01 view at the canonical res-01/input path (1,219 TRAIN tables/pools + redacted
#     selected.json), and inside the rw data root $H317_DATA (= h335-cl): neural/ (screener models, hard links to H317's),
#     views-search/ (unless H335_VIEWS_RW=1) and bin/ (superset worker); redteam/ hidden;
#   * network unshared unless --net.
set -euo pipefail
SRC317=/home/ubuntu/jope_plan/evidence/H317/source/research/flywheel_selfplay/h317_island10_multiscale_curriculum
SRC=/home/ubuntu/jope_plan/evidence/H335/source/research/flywheel_selfplay/h335_aggressive_locked_elo
DATA=${H335_DATA:-/home/ubuntu/jope-data/h335-cl}
VIEW=/home/ubuntu/jope-data/h317-cl/res01-train-view
MICRO=/home/ubuntu/JopeDimeMicro
NET=(--unshare-net)
if [ "${1:-}" = "--net" ]; then NET=(); shift; fi
[ -f "$VIEW/selected.json" ] || { echo "missing TRAIN view $VIEW" >&2; exit 2; }
[ "$(ls "$VIEW/tables" | wc -l)" = 1219 ] || { echo "TRAIN view tables != 1219" >&2; exit 2; }
mkdir -p "$DATA/redteam" "$DATA/views-search" "$DATA/neural" "$DATA/bin"
RO=(--ro-bind "$DATA/neural" "$DATA/neural" --ro-bind "$DATA/bin" "$DATA/bin")
[ "${H335_VIEWS_RW:-0}" = 1 ] || RO+=(--ro-bind "$DATA/views-search" "$DATA/views-search")
exec bwrap \
  --ro-bind / / \
  --tmpfs /home/ubuntu --tmpfs /tmp --tmpfs /mnt --tmpfs /media --tmpfs /srv \
  --tmpfs /app --tmpfs /review --tmpfs /Users --tmpfs /opt/work \
  --dev /dev --proc /proc \
  --ro-bind /home/ubuntu/.local /home/ubuntu/.local \
  --ro-bind "$SRC317" "$SRC317" \
  --ro-bind "${H335_CODE:-$SRC}" "$SRC" \
  --bind "$DATA" "$DATA" --tmpfs "$DATA/redteam" "${RO[@]}" \
  --ro-bind "$MICRO" "$MICRO" --tmpfs "$MICRO/assets/benchmark_data" \
  --ro-bind "$VIEW" /home/ubuntu/jope-data/res-01/input \
  --setenv HOME /home/ubuntu \
  --setenv H317_RES01 /home/ubuntu/jope-data/res-01/input \
  --setenv H317_DATA "$DATA" \
  --setenv H317_SANDBOX 1 \
  --setenv PYTHONDONTWRITEBYTECODE 1 \
  --chdir "$SRC" \
  "${NET[@]}" --die-with-parent \
  "$@"
