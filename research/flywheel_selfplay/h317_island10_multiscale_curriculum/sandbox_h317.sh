#!/bin/bash
# H317 deny-by-default sandbox for every Gate H0 process on xbabe2 (leakage red team BLOCKER-1, fix (b)).
# The in-process audit hook cannot see native-code readers, dir_fd opens or child processes (the Rust worker); this
# sandbox makes everything except the TRAIN-only view invisible. Usage: sandbox_h317.sh [--net] <command...>
#   * /home/ubuntu, /tmp, /mnt, /media, /srv, /app, /review, /Users, /opt/work are empty tmpfs;
#   * bound back: ~/.local (python user site, ro), the H317 source dir (ro), $H317_DATA (rw, with redteam/ hidden),
#     JopeDimeMicro (ro, assets/benchmark_data hidden), and the TRAIN-only res-01 view mounted AT the canonical
#     res-01/input path (ro): hard links to the 1,219 TRAIN tables/pools + champions/winners/source_manifest +
#     the redacted selected.json (dev rows = identifiers only);
#   * network is unshared unless --net is given.
set -euo pipefail
SRC=/home/ubuntu/jope_plan/evidence/H317/source/research/flywheel_selfplay/h317_island10_multiscale_curriculum
DATA=${H317_DATA:-/home/ubuntu/jope-data/h317-cl}
VIEW=/home/ubuntu/jope-data/h317-cl/res01-train-view
MICRO=/home/ubuntu/JopeDimeMicro
NET=(--unshare-net)
if [ "${1:-}" = "--net" ]; then NET=(); shift; fi
[ -f "$VIEW/selected.json" ] || { echo "missing TRAIN view $VIEW" >&2; exit 2; }
[ "$(ls "$VIEW/tables" | wc -l)" = 1219 ] || { echo "TRAIN view tables != 1219" >&2; exit 2; }
exec bwrap \
  --ro-bind / / \
  --tmpfs /home/ubuntu --tmpfs /tmp --tmpfs /mnt --tmpfs /media --tmpfs /srv \
  --tmpfs /app --tmpfs /review --tmpfs /Users --tmpfs /opt/work \
  --dev /dev --proc /proc \
  --ro-bind /home/ubuntu/.local /home/ubuntu/.local \
  --ro-bind "$SRC" "$SRC" \
  --bind "$DATA" "$DATA" --tmpfs "$DATA/redteam" \
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
