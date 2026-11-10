#!/bin/bash
# H410 dev91 docker sandbox on xbabe0 (= sandbox_h335_docker.sh with the H410 data root, receipt and mounts).
#   sandbox_h410_d91.sh <prelock|unlock|run> <command...>     env: H410_STAGE (holds code/), H410_CPUSET, H410_MEMORY
#   prelock  TRAIN-only res-01 view, no DL-Val byte visible; $DATA/stage_v rw (writes LOCK.json)
#   unlock   TRAIN-only view, no DL-Val byte visible; $DATA/stage_v rw (creates STAGE_V_STARTED); receipt required
#   run      DL-Val view (H335 Stage V dlval-view, ro) at res-01/input; H335 stagev-root/views-search ro at
#            $DATA/views-search; $DATA/stage_v ro; receipt required and must name sha256(LOCK.json); marker required
# Always: image h317-empty:v1, --network none, uid 1000, --cap-drop ALL, no-new-privileges, read-only rootfs, empty
# /home and /tmp tmpfs; ro: ~/.local, frozen H317 code, the staged H410 code (at the H335 source path), JopeDimeMicro,
# h335-cl/bin (superset worker) and h335-cl/neural (screener models); rw: $DATA only. H335's Stage V root is never
# mounted rw.
set -euo pipefail
MODE=${1:?prelock|unlock|run}; shift
SRC317=/home/ubuntu/jope_plan/evidence/H317/source/research/flywheel_selfplay/h317_island10_multiscale_curriculum
SRC=/home/ubuntu/jope_plan/evidence/H335/source/research/flywheel_selfplay/h335_aggressive_locked_elo
STAGE317=/home/ubuntu/h317-stage
STAGE=${H410_STAGE:?H410_STAGE}
DATA=/home/ubuntu/jope-data/h410-d91
H335D=/home/ubuntu/jope-data/h335-cl
SV=/home/ubuntu/h335-stage-v
RECEIPT=/home/ubuntu/jope_plan/ns/receipts/H410_LOCK_sha256.txt
CPUSET=${H410_CPUSET:-16-63,80-127}
MEM=${H410_MEMORY:-64g}
IMAGE=h317-empty:v1
refuse() { echo "H410 sandbox refused: $*" >&2; exit 3; }
(cd "$STAGE/code" && sha256sum --quiet -c ../CODE.sha256) || refuse "staged code differs from CODE.sha256"
[ ! -w "$STAGE/code/d91_h410.py" ] || refuse "$STAGE/code is not read-only"
mkdir -p "$DATA/stage_v" "$DATA/views-search"
VIEW=$STAGE317/res01-train-view
EXTRA=()
case $MODE in
  prelock) EXTRA=(-v "$DATA/stage_v:$DATA/stage_v:rw") ;;
  unlock)
    [ -s "$RECEIPT" ] || refuse "no receipt"
    [ "$(sha256sum "$DATA/stage_v/LOCK.json" | cut -d' ' -f1)" = "$(cut -d' ' -f1 "$RECEIPT")" ] || refuse "LOCK != receipt"
    EXTRA=(-v "$DATA/stage_v:$DATA/stage_v:rw" -v "$RECEIPT:$RECEIPT:ro") ;;
  run)
    [ -s "$RECEIPT" ] || refuse "no receipt"
    R=$(cut -d' ' -f1 "$RECEIPT")
    [ "$(sha256sum "$DATA/stage_v/LOCK.json" | cut -d' ' -f1)" = "$R" ] || refuse "LOCK != receipt"
    [ "$(cut -d' ' -f2 "$DATA/stage_v/STAGE_V_STARTED")" = "$R" ] || refuse "STAGE_V_STARTED does not name this LOCK"
    VIEW=$SV/dlval-view
    [ "$(ls "$VIEW/tables" | wc -l)" = 1310 ] || refuse "$VIEW/tables != 1,310"
    EXTRA=(-v "$DATA/stage_v:$DATA/stage_v:ro" -v "$RECEIPT:$RECEIPT:ro"
           -v "$SV/stagev-root/views-search:$DATA/views-search:ro") ;;
  *) refuse "mode $MODE" ;;
esac
[ "$(ls "$VIEW/tables" | wc -l)" -ge 1219 ] || refuse "view $VIEW incomplete"
[ "$(sha256sum "$H335D/bin/bootstrap.s1main.superset-2a848c8f8085" | cut -c1-64)" = \
  2a848c8f8085a39210c9714b3d3b80448fea66afecb54a951345f2926c26d7a2 ] || refuse "superset worker sha mismatch"
ETC=()
for f in passwd group nsswitch.conf ld.so.cache ld.so.conf localtime; do ETC+=(-v "/etc/$f:/etc/$f:ro"); done
for d in ld.so.conf.d python3.12 alternatives; do [ -d "/etc/$d" ] && ETC+=(-v "/etc/$d:/etc/$d:ro"); done
exec docker run --rm -i --network none --user 1000:1000 --cap-drop ALL \
  --security-opt no-new-privileges --read-only --cpuset-cpus "$CPUSET" --memory "$MEM" --pids-limit 16384 \
  --tmpfs /tmp:rw,size=256m --tmpfs /home:rw,size=64m,mode=0755 --tmpfs /run:rw,size=16m \
  -v /usr:/usr:ro -v /lib:/lib:ro -v /lib64:/lib64:ro -v /bin:/bin:ro -v /sbin:/sbin:ro "${ETC[@]}" \
  -v /home/ubuntu/.local:/home/ubuntu/.local:ro \
  -v "$STAGE317/code:$SRC317:ro" \
  -v "$STAGE/code:$SRC:ro" \
  -v "$VIEW:/home/ubuntu/jope-data/res-01/input:ro" \
  -v "$STAGE317/micro:/home/ubuntu/JopeDimeMicro:ro" \
  -v "$DATA:$DATA:rw" "${EXTRA[@]}" \
  -v "$H335D/bin:$H335D/bin:ro" \
  -v "$H335D/neural:$H335D/neural:ro" \
  -e HOME=/home/ubuntu -e PATH=/usr/bin:/bin -e H317_RES01=/home/ubuntu/jope-data/res-01/input \
  -e H317_DATA="$DATA" -e H410_D91="$DATA" -e H317_SANDBOX=docker -e PYTHONDONTWRITEBYTECODE=1 \
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e H335_GROUP=AGG \
  -w "$SRC" "$IMAGE" "$@"
