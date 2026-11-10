#!/bin/bash
# H335 Stage V staging between xbabe2 (run it here) and xbabe0 (stage_v_mounts_h335.md). Every transfer is checked with
# sha256sum -c on arrival and every check is fatal. Refused while any h335-* unit is active on xbabe0.
# Usage: stage_v_xbabe0.sh lock|prep|pullback|first-outcome
#   lock          after the receipt, BEFORE the unlock: the LOCK's H335 runtime code + the S6 script run_audit_v.sh
#                 (its sha from the LOCK's docs_sha256), both from the frozen copy $H335_CODE,
#                 $DATA/bin/frontier_audit_h335, LOCK.json, train_inputs_manifest.json, the receipt; then every LOCK
#                 file is checked at its xbabe0 host path AND inside the docker sandbox (stage_v_h335.py
#                 verify-lock-files; it never unlocks)
#   prep          after $SV/run/PREP_DONE.json: STAGE_V_STARTED, UNLOCK_RECEIPT.json, the DL-Val view (hard links of the
#                 xbabe0 TRAIN view + the 91 DL-Val tables; no pools), stagev-root/{views-search (91),
#                 run/PREP_DONE.json, run/targets_manifest.json, empty gp/}; then a dry run of the S6 layout:
#                 `sandbox_h335_docker.sh --stage-v ... verify-lock-files` (every LOCK file inside the Stage V container)
#   pullback      after the audit (refused once FIRST_OUTCOME_READ exists on xbabe2): gp/stages, run/{AUDIT_*,
#                 unit_attempts.*, STOPPED_*, shards/*} and run/events.jsonl -> $SV/run/events.xbabe0.jsonl
#   first-outcome after stats: $SV/run/FIRST_OUTCOME_READ -> xbabe0, so no audit unit can run there any more
set -euo pipefail
SRC=/home/ubuntu/jope_plan/evidence/H335/source/research/flywheel_selfplay/h335_aggressive_locked_elo
SRC317=/home/ubuntu/jope_plan/evidence/H317/source/research/flywheel_selfplay/h317_island10_multiscale_curriculum
CODE=${H335_CODE:-/home/ubuntu/h335-stage/code}       # xbabe2 frozen read-only copy (bound at $SRC by the sandboxes)
DATA=/home/ubuntu/jope-data/h335-cl
SV=/home/ubuntu/jope-data/h335-stagev
MICRO=/home/ubuntu/JopeDimeMicro
RES01=/home/ubuntu/jope-data/res-01/input
RECEIPT=/home/ubuntu/jope_plan/ns/receipts/H335_LOCK_sha256.txt
X0=${H335_STAGE_HOST:-xbabe0}
X0STAGE=/home/ubuntu/h335-stage                        # $X0STAGE/code is mounted at $SRC in the docker sandbox
X0STAGE317=/home/ubuntu/h317-stage                     # code / micro / res01-train-view mounted at SRC317 / MICRO / RES01
STAGE_V=/home/ubuntu/h335-stage-v
SVROOT=$STAGE_V/stagev-root                            # mounted at $SV in the --stage-v docker sandbox
LOCK=$DATA/stage_v/LOCK.json
S6_FILES=(run_audit_v.sh)                              # scripts S6 runs on xbabe0 (recorded in docs_sha256)
MODE=${1:-}
TAG=$(date -u +%Y%m%dT%H%M%SZ)
die() { echo "stage_v_xbabe0.sh $MODE FAILED: $*" >&2; exit 5; }
x0() { ssh -n -o BatchMode=yes "$X0" "$@"; }
[ "$(hostname)" = xbabe2 ] || die "run on xbabe2"
active=$(x0 "systemctl --user list-units --state=active --no-legend 'h335-*'") || die "ssh $X0 failed"
[ -z "$active" ] || die "an h335-* unit is active on $X0: $active"
[ -s "$LOCK" ] && [ -s "$RECEIPT" ] || die "LOCK.json or the receipt is missing"
R=$(cut -d' ' -f1 "$RECEIPT")
[ "$(sha256sum "$LOCK" | cut -d' ' -f1)" = "$R" ] || die "LOCK.json does not match the receipt"
verify_in_docker() {  # $1 = "" (Gate-H0 layout) or "--stage-v" (the exact S6 container layout)
  x0 "cd $X0STAGE/code && H335_STAGE_V=$STAGE_V ./sandbox_h335_docker.sh ${1:-} python3 stage_v_h335.py \
      verify-lock-files" || die "verify-lock-files refused inside the $X0 docker sandbox ${1:-}"
}

lock_mode() {
  mkdir -p "$DATA/staging"
  local M=$DATA/staging/x0_lock_$TAG
  jq -r '.code_sha256 | to_entries[] | "\(.value)  \(.key)"' "$LOCK" > "$M.lock.sha256" || die "jq LOCK"
  for f in "${S6_FILES[@]}"; do  # the S6 scripts, sha-pinned by the LOCK's (non-verified) docs record
    jq -r --arg f "$SRC/$f" '"\(.docs_sha256[$f] // error("no \($f) in docs_sha256"))  \($f)"' "$LOCK" \
      >> "$M.lock.sha256" || die "$f is not recorded in the LOCK docs_sha256"
  done
  grep -F "  $SRC/" "$M.lock.sha256" | sed "s#  $SRC/#  #" > "$M.code.sha256" || die "no H335 code in the LOCK"
  (cd "$CODE" && sha256sum --quiet -c "$M.code.sha256") || die "$CODE differs from the LOCK"
  cut -c67- "$M.code.sha256" > "$M.code.files"
  sed -e "s#  $SRC/#  $X0STAGE/code/#" -e "s#  $SRC317/#  $X0STAGE317/code/#" -e "s#  $MICRO/#  $X0STAGE317/micro/#" \
      -e "s#  $RES01/#  $X0STAGE317/res01-train-view/#" "$M.lock.sha256" > "$M.x0.sha256"
  x0 "mkdir -p $X0STAGE/code $DATA/bin $DATA/stage_v $DATA/staging $(dirname "$RECEIPT")" || die "mkdir on $X0"
  rsync -a --chmod=Fa-w --files-from="$M.code.files" "$CODE/" "$X0:$X0STAGE/code/" || die "code rsync"
  rsync -a --chmod=Fa-w "$DATA/bin/frontier_audit_h335" "$X0:$DATA/bin/" || die "frontier binary rsync"
  rsync -a --chmod=Fa-w "$LOCK" "$DATA/stage_v/train_inputs_manifest.json" "$X0:$DATA/stage_v/" || die "LOCK rsync"
  rsync -a --chmod=Fa-w "$RECEIPT" "$X0:$RECEIPT" || die "receipt rsync"
  scp -q "$M.x0.sha256" "$X0:$DATA/staging/" || die "manifest scp"
  x0 "sha256sum --quiet -c $DATA/staging/$(basename "$M").x0.sha256" || die "a LOCK file differs at its $X0 host path"
  x0 "[ \"\$(sha256sum $LOCK | cut -d' ' -f1)\" = $R ] && [ \"\$(cut -d' ' -f1 $RECEIPT)\" = $R ]" \
    || die "LOCK.json / receipt on $X0 != $R"
  verify_in_docker ""
  echo "lock staging OK on $X0 ($(wc -l < "$M.lock.sha256") files verified on the host; LOCK files also in docker)"
}

prep_mode() {
  [ "$(cut -d' ' -f2 "$DATA/stage_v/STAGE_V_STARTED")" = "$R" ] || die "STAGE_V_STARTED does not name this LOCK"
  [ -s "$SV/run/PREP_DONE.json" ] || die "no $SV/run/PREP_DONE.json"
  for f in STOPPED_REGRESSION_FAIL STOPPED_INTEGRITY FIRST_OUTCOME_READ; do
    [ ! -e "$SV/run/$f" ] || die "$f exists"
  done
  mkdir -p "$SV/staging"
  local M=$SV/staging/x0_prep_$TAG b
  b=$(basename "$M")
  jq -r '.targets | keys[]' "$SV/run/targets_manifest.json" > "$M.cids" || die "jq targets"
  [ "$(wc -l < "$M.cids")" = 91 ] || die "targets != 91"
  sed 's#.*#tables/&.jtt1#' "$M.cids" > "$M.tables.files"
  sed 's#.*#views-search/&.npz#' "$M.cids" > "$M.views.files"
  (cd "$SV/res01-v-view" && xargs -a "$M.tables.files" sha256sum) > "$M.tables.sha256" || die "table shas"
  (cd "$SV" && xargs -a "$M.views.files" sha256sum && sha256sum run/PREP_DONE.json run/targets_manifest.json) \
    > "$M.root.sha256" || die "view shas"
  (cd "$DATA/stage_v" && sha256sum LOCK.json STAGE_V_STARTED UNLOCK_RECEIPT.json train_inputs_manifest.json) \
    > "$M.sv.sha256" || die "stage_v shas"
  x0 "mkdir -p $STAGE_V/dlval-view $SVROOT/run $SVROOT/gp $SVROOT/views-search $DATA/stage_v && \
      { [ -d $STAGE_V/dlval-view/tables ] || cp -al $X0STAGE317/res01-train-view/. $STAGE_V/dlval-view/; }" \
    || die "dlval-view base on $X0"
  rsync -a --chmod=Fa-w --files-from="$M.tables.files" "$SV/res01-v-view/" "$X0:$STAGE_V/dlval-view/" || die "tables"
  rsync -a --chmod=Fa-w --files-from="$M.views.files" "$SV/" "$X0:$SVROOT/" || die "views"
  rsync -a --chmod=Fa-w "$SV/run/PREP_DONE.json" "$SV/run/targets_manifest.json" "$X0:$SVROOT/run/" || die "run files"
  rsync -a --chmod=Fa-w "$DATA/stage_v/STAGE_V_STARTED" "$DATA/stage_v/UNLOCK_RECEIPT.json" "$X0:$DATA/stage_v/" \
    || die "marker"
  scp -q "$M.tables.sha256" "$M.root.sha256" "$M.sv.sha256" "$X0:$STAGE_V/" || die "manifest scp"
  x0 "cd $STAGE_V/dlval-view && sha256sum --quiet -c $STAGE_V/$b.tables.sha256" || die "DL-Val tables differ"
  x0 "cd $SVROOT && sha256sum --quiet -c $STAGE_V/$b.root.sha256" || die "views / run files differ"
  x0 "cd $DATA/stage_v && sha256sum --quiet -c $STAGE_V/$b.sv.sha256" || die "stage_v files differ"
  n=$(x0 "ls $STAGE_V/dlval-view/tables | wc -l") || die "count"
  [ "$n" = 1310 ] || die "dlval-view tables $n != 1310"
  verify_in_docker --stage-v  # dry run of the S6 container: receipt / marker / views / frontier checks + LOCK hashes
  echo "prep staging OK on $X0 (S6 layout dry run passed)"
}

pullback_mode() {
  [ ! -e "$SV/run/FIRST_OUTCOME_READ" ] || die "FIRST_OUTCOME_READ exists: the record set is frozen"
  mkdir -p "$SV/staging"
  local M=$SV/staging/x0_pull_$TAG b
  b=$(basename "$M")
  x0 "cd $SVROOT && find gp/stages run -type f \\( -path 'gp/stages/*' -o -name 'AUDIT_*' -o -name 'unit_attempts.*' \
      -o -name 'STOPPED_*' -o -path 'run/shards/*' -o -path run/events.jsonl \\) -print0 | sort -z \
      | xargs -0 -r sha256sum > $STAGE_V/$b.sha256" || die "remote manifest"
  scp -q "$X0:$STAGE_V/$b.sha256" "$M.sha256" || die "manifest scp"
  [ -s "$M.sha256" ] || die "nothing to pull"
  grep -v '  run/events.jsonl$' "$M.sha256" > "$M.main.sha256" || true
  cut -c67- "$M.main.sha256" > "$M.files"
  rsync -a --files-from="$M.files" "$X0:$SVROOT/" "$SV/" || die "records rsync"
  (cd "$SV" && sha256sum --quiet -c "$M.main.sha256") || die "pulled records differ"
  if grep -q '  run/events.jsonl$' "$M.sha256"; then
    rsync -a "$X0:$SVROOT/run/events.jsonl" "$SV/run/events.xbabe0.jsonl" || die "events rsync"
    grep '  run/events.jsonl$' "$M.sha256" | sed 's#  run/events.jsonl$#  run/events.xbabe0.jsonl#' > "$M.ev.sha256"
    (cd "$SV" && sha256sum --quiet -c "$M.ev.sha256") || die "events differ"
  fi
  echo "pullback OK: $(wc -l < "$M.sha256") files (manifest $M.sha256)"
}

first_outcome_mode() {
  [ -s "$SV/run/FIRST_OUTCOME_READ" ] || die "no FIRST_OUTCOME_READ on xbabe2 (run stats first)"
  rsync -a --ignore-existing --chmod=Fa-w "$SV/run/FIRST_OUTCOME_READ" "$X0:$SVROOT/run/" || die "rsync"
  want=$(sha256sum "$SV/run/FIRST_OUTCOME_READ" | cut -d' ' -f1)
  got=$(x0 "sha256sum $SVROOT/run/FIRST_OUTCOME_READ" | cut -d' ' -f1) || die "remote sha"
  [ "$want" = "$got" ] || die "FIRST_OUTCOME_READ on $X0 differs"
  echo "FIRST_OUTCOME_READ is on $X0: the audit refuses every further unit there"
}

case "$MODE" in
  lock) lock_mode ;;
  prep) prep_mode ;;
  pullback) pullback_mode ;;
  first-outcome) first_outcome_mode ;;
  *) echo "usage: $0 lock|prep|pullback|first-outcome" >&2; exit 2 ;;
esac
