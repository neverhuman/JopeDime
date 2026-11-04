#!/bin/bash
# H317 deny-by-default docker sandbox for xbabe0 (equivalent of sandbox_h317.sh / bwrap on xbabe2).
# Usage (on xbabe0):  sandbox_h317_docker.sh [--stage-v] <command...>
#   * image h317-empty:v1 = an EMPTY rootfs (docker import of an empty tar); host /usr /lib /lib64 /bin /sbin are bound
#     read-only, /etc is only passwd, group, nsswitch.conf, ld.so.cache, ld.so.conf(.d), localtime, python3.12/,
#     alternatives/ (system numpy links libblas/liblapack through /etc/alternatives);
#   * --network none, uid:gid 1000:1000, --cap-drop ALL, no-new-privileges, read-only rootfs (no --init: /sbin is the host bind);
#   * /home and /tmp are empty tmpfs; bound back: ~/.local (ro, python libs incl. torch), the staged H317 code (ro, at
#     the canonical source path), the staged TRAIN view (ro, at /home/ubuntu/jope-data/res-01/input), the staged
#     JopeDimeMicro subset (ro: worker + 3 python modules + feature names; no assets/benchmark_data, no results),
#     $H317_DATA (rw) with the staged view files and screener models (neural/runs) re-bound ro and redteam/ hidden;
#   * CPU set / memory: H317_CPUSET (default 16-63,80-127), H317_MEMORY (default 64g); /proc/stat and /proc/meminfo
#     stay host-wide, so the in-process resource guard counts OTHER load on the host too (pause above 85 % CPU).
#   * Stage V: `--stage-v` mounts $H317_STAGE_V/dlval-view (TRAIN + DL-Val tables/pools, built only after the LOCK) in
#     place of the TRAIN view. It is refused unless the create-once LOCK receipt exists and matches LOCK.json. Gate H0
#     invocations never mount any DL-Val directory.
set -euo pipefail
SRC=/home/ubuntu/jope_plan/evidence/H317/source/research/flywheel_selfplay/h317_island10_multiscale_curriculum
STAGE=${H317_STAGE:-/home/ubuntu/h317-stage}
STAGE_V=${H317_STAGE_V:-/home/ubuntu/h317-stage-v}
DATA=${H317_DATA:-/home/ubuntu/jope-data/h317-cl}
IMAGE=h317-empty:v1
CPUSET=${H317_CPUSET:-16-63,80-127}
MEM=${H317_MEMORY:-64g}
VIEW=$STAGE/res01-train-view
if [ "${1:-}" = "--stage-v" ]; then
  shift
  RECEIPT=/home/ubuntu/jope_plan/ns/receipts/H317_LOCK_sha256.txt
  [ -s "$RECEIPT" ] || { echo "Stage V refused: no LOCK receipt" >&2; exit 3; }
  [ "$(sha256sum "$DATA/stage_v/LOCK.json" | cut -d' ' -f1)" = "$(cut -d' ' -f1 "$RECEIPT")" ] \
    || { echo "Stage V refused: LOCK.json does not match the receipt" >&2; exit 3; }
  VIEW=$STAGE_V/dlval-view
fi
[ "$(ls "$VIEW/tables" | wc -l)" -ge 1219 ] || { echo "view $VIEW incomplete" >&2; exit 2; }
[ "$(sha256sum "$STAGE/micro/bin/bootstrap.s1main" | cut -c1-64)" = \
  e3be938644722d14a1c1a0e82168518838c4d174cfe031076a339745c48870a8 ] || { echo "worker sha mismatch" >&2; exit 2; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || tar -cf - -T /dev/null | docker import - "$IMAGE" >/dev/null
mkdir -p "$DATA/redteam" "$DATA/neural/runs" "$DATA/views-search"
ETC=()
for f in passwd group nsswitch.conf ld.so.cache ld.so.conf localtime; do ETC+=(-v "/etc/$f:/etc/$f:ro"); done
for d in ld.so.conf.d python3.12 alternatives; do [ -d "/etc/$d" ] && ETC+=(-v "/etc/$d:/etc/$d:ro"); done
TTY=(); [ -t 0 ] && TTY=(-t)
exec docker run --rm -i "${TTY[@]}" --network none --user 1000:1000 --cap-drop ALL \
  --security-opt no-new-privileges --read-only --cpuset-cpus "$CPUSET" --memory "$MEM" --pids-limit 8192 \
  --tmpfs /tmp:rw,size=256m --tmpfs /home:rw,size=64m,mode=0755 --tmpfs /run:rw,size=16m \
  -v /usr:/usr:ro -v /lib:/lib:ro -v /lib64:/lib64:ro -v /bin:/bin:ro -v /sbin:/sbin:ro "${ETC[@]}" \
  -v /home/ubuntu/.local:/home/ubuntu/.local:ro \
  -v "$STAGE/code:$SRC:ro" \
  -v "$VIEW:/home/ubuntu/jope-data/res-01/input:ro" \
  -v "$STAGE/micro:/home/ubuntu/JopeDimeMicro:ro" \
  -v "$DATA:$DATA:rw" \
  -v "$DATA/neural/views_c32_train1219.npz:$DATA/neural/views_c32_train1219.npz:ro" \
  -v "$DATA/views-search:$DATA/views-search:ro" \
  -v "$DATA/neural/runs:$DATA/neural/runs:ro" \
  --tmpfs "$DATA/redteam:rw,size=1m" \
  -e HOME=/home/ubuntu -e PATH=/usr/bin:/bin -e H317_RES01=/home/ubuntu/jope-data/res-01/input \
  -e H317_DATA="$DATA" -e H317_SANDBOX=docker -e PYTHONDONTWRITEBYTECODE=1 \
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  ${H317_GUARD_LOG:+-e H317_GUARD_LOG="$H317_GUARD_LOG"} \
  -w "$SRC" "$IMAGE" "$@"
