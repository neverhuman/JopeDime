#!/bin/bash
# H335 deny-by-default docker sandbox on xbabe0 (= H317 sandbox_h317_docker.sh, Gate-H0 mode, + the H335 code dir,
# the h335-cl data root and the superset worker), plus a `--stage-v` mode (= sandbox_h317_docker.sh --stage-v with H335
# paths; stage_v_mounts_h335.md). Usage (on xbabe0): sandbox_h335_docker.sh [--stage-v] <command...>
#   * image h317-empty:v1 (empty rootfs); host /usr /lib /lib64 /bin /sbin ro; minimal /etc; --network none;
#     uid 1000:1000; --cap-drop ALL; no-new-privileges; read-only rootfs; /home and /tmp empty tmpfs;
#   * bound back ro: ~/.local, the staged FROZEN H317 code (at the H317 source path; sha-pinned by h335_core),
#     the staged H335 code (at the H335 source path), the staged TRAIN view (at res-01/input), the staged
#     JopeDimeMicro subset; rw: $DATA (h335-cl) with neural/, views-search/, bin/ re-bound ro and redteam/ hidden;
#   * CPU set / memory: H335_CPUSET (default 16-63,80-127), H335_MEMORY (default 96g).
#   * --stage-v (only after the unlock): $H335_STAGE_V/dlval-view (the staged TRAIN view + the 91 DL-Val tables, built
#     only after the unlock; no pools) replaces the TRAIN view at res-01/input; plus the H335 LOCK receipt (ro, fixed
#     path), the Stage V root $H335_STAGE_V/stagev-root at /home/ubuntu/jope-data/h335-stagev (rw, H335_STAGEV_ROOT)
#     with views-search/ re-bound ro, $DATA/stage_v ro (LOCK.json, STAGE_V_STARTED, train_inputs_manifest.json) and
#     $DATA/build ro (the frontier binary is a LOCK file). Refused unless the receipt names sha256(LOCK.json), the
#     marker names the same LOCK, the Stage V dirs and the frontier binary are staged and the view holds 1,310 tables.
set -euo pipefail
SRC317=/home/ubuntu/jope_plan/evidence/H317/source/research/flywheel_selfplay/h317_island10_multiscale_curriculum
SRC=/home/ubuntu/jope_plan/evidence/H335/source/research/flywheel_selfplay/h335_aggressive_locked_elo
STAGE317=${H317_STAGE:-/home/ubuntu/h317-stage}
STAGE=${H335_STAGE:-/home/ubuntu/h335-stage}
DATA=${H335_DATA:-/home/ubuntu/jope-data/h335-cl}
IMAGE=h317-empty:v1
CPUSET=${H335_CPUSET:-16-63,80-127}
MEM=${H335_MEMORY:-96g}
VIEW=$STAGE317/res01-train-view
SVM=()
if [ "${1:-}" = "--stage-v" ]; then
  shift
  RECEIPT=/home/ubuntu/jope_plan/ns/receipts/H335_LOCK_sha256.txt
  STAGE_V=${H335_STAGE_V:-/home/ubuntu/h335-stage-v}
  SVROOT=$STAGE_V/stagev-root
  SVC=/home/ubuntu/jope-data/h335-stagev
  sv_refuse() { echo "Stage V refused: $*" >&2; exit 3; }
  [ -s "$RECEIPT" ] || sv_refuse "no LOCK receipt"
  R=$(cut -d' ' -f1 "$RECEIPT")
  [ "$(sha256sum "$DATA/stage_v/LOCK.json" | cut -d' ' -f1)" = "$R" ] || sv_refuse "LOCK.json does not match the receipt"
  [ -s "$DATA/stage_v/STAGE_V_STARTED" ] || sv_refuse "no STAGE_V_STARTED marker"
  [ "$(cut -d' ' -f2 "$DATA/stage_v/STAGE_V_STARTED")" = "$R" ] || sv_refuse "STAGE_V_STARTED does not name this LOCK"
  for d in "$SVROOT/views-search" "$SVROOT/run" "$SVROOT/gp" "$DATA/build"; do [ -d "$d" ] || sv_refuse "missing $d"; done
  [ -s "$DATA/build/frontier_audit_h335" ] || sv_refuse "LOCK file $DATA/build/frontier_audit_h335 not staged"
  VIEW=$STAGE_V/dlval-view
  [ "$(ls "$VIEW/tables" | wc -l)" = 1310 ] || sv_refuse "$VIEW/tables != 1,219 TRAIN + 91 DL-Val"
  SVM=(-v "$RECEIPT:$RECEIPT:ro"
       -v "$SVROOT:$SVC:rw"
       -v "$SVROOT/views-search:$SVC/views-search:ro"
       -v "$DATA/stage_v:$DATA/stage_v:ro"
       -v "$DATA/build:$DATA/build:ro"
       -e H335_STAGEV_ROOT="$SVC")
fi
[ "$(ls "$VIEW/tables" | wc -l)" -ge 1219 ] || { echo "view $VIEW incomplete" >&2; exit 2; }
[ "$(sha256sum "$DATA/bin/bootstrap.s1main.superset-2a848c8f8085" | cut -c1-64)" = \
  2a848c8f8085a39210c9714b3d3b80448fea66afecb54a951345f2926c26d7a2 ] || { echo "superset worker sha mismatch" >&2; exit 2; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || tar -cf - -T /dev/null | docker import - "$IMAGE" >/dev/null
mkdir -p "$DATA/redteam" "$DATA/neural" "$DATA/views-search" "$DATA/bin"
ETC=()
for f in passwd group nsswitch.conf ld.so.cache ld.so.conf localtime; do ETC+=(-v "/etc/$f:/etc/$f:ro"); done
for d in ld.so.conf.d python3.12 alternatives; do [ -d "/etc/$d" ] && ETC+=(-v "/etc/$d:/etc/$d:ro"); done
TTY=(); [ -t 0 ] && TTY=(-t)
exec docker run --rm -i "${TTY[@]}" --network none --user 1000:1000 --cap-drop ALL \
  --security-opt no-new-privileges --read-only --cpuset-cpus "$CPUSET" --memory "$MEM" --pids-limit 16384 \
  --tmpfs /tmp:rw,size=256m --tmpfs /home:rw,size=64m,mode=0755 --tmpfs /run:rw,size=16m \
  -v /usr:/usr:ro -v /lib:/lib:ro -v /lib64:/lib64:ro -v /bin:/bin:ro -v /sbin:/sbin:ro "${ETC[@]}" \
  -v /home/ubuntu/.local:/home/ubuntu/.local:ro \
  -v "$STAGE317/code:$SRC317:ro" \
  -v "$STAGE/code:$SRC:ro" \
  -v "$VIEW:/home/ubuntu/jope-data/res-01/input:ro" \
  -v "$STAGE317/micro:/home/ubuntu/JopeDimeMicro:ro" \
  -v "$DATA:$DATA:rw" \
  -v "$DATA/neural:$DATA/neural:ro" \
  -v "$DATA/views-search:$DATA/views-search:ro" \
  -v "$DATA/bin:$DATA/bin:ro" \
  --tmpfs "$DATA/redteam:rw,size=1m" "${SVM[@]}" \
  -e HOME=/home/ubuntu -e PATH=/usr/bin:/bin -e H317_RES01=/home/ubuntu/jope-data/res-01/input \
  -e H317_DATA="$DATA" -e H317_SANDBOX=docker -e PYTHONDONTWRITEBYTECODE=1 \
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  ${H335_GROUP:+-e H335_GROUP="$H335_GROUP"} \
  -w "$SRC" "$IMAGE" "$@"
