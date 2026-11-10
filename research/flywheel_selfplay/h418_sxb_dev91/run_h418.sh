#!/bin/bash
# H418 orchestration, run on xbabe2 (PREREG §3). Every scoring step refuses unless the stage is a `pub` stage whose
# H418 files are byte-identical to origin/v1/next (PREREG first). Steps marked [no dev91] read no dev91 byte.
#   H418_STAGE=/home/ubuntu/h418-stage/pub-TAG run_h418.sh STEP [args]
#   selftest DIR        [no dev91] analyze_h418.py on synthetic fixtures (selftest_h418.py) in a new scratch DIR
#   import-check        [no dev91] xbabe0 docker, prelock (TRAIN-only view): `d91_h418.py --help` (dev or pub stage)
#   gate                [no dev91] every PR in $EV/gate_prs.txt is MERGED (the owner's merge-all gate)
#   precheck            [no dev91] no h416.* score on the dev91 board (main, v1/next, local root), no H418 LOCK yet
#   inputs              create-once: DATA/h335-prep (H335 prep manifests, cmp vs the H381/H410 copies), the sha-only
#                       manifest of the stored GS files (91 memos + 182 V_AGG records), the board export at the PREREG
#                       commit + its sha manifest, spec.json. Hashes only; no value is read.
#   lock                xbabe0 prelock sandbox: d91_h418.py lock; receipt create-once (noclobber) on xbabe0 + xbabe2
#   announce CPUS HOURS post the xbabe0 core window on AGENT_CHAT.md (required before `gp` / `deploy`)
#   unlock              gate re-checked; the one-shot Stage V unlock, then `select` (cids91.json)
#   gp ARM CPUS JOBS MEM          xbabe0 unit h418-gp-ARM (ARM in GE GL): d91_h418.py gp, 91 contexts x 2 seeds
#   deploy-prep         create-once DATA/runs/D1, D2 memos = per context cat GS GE GL (GS checked against the LOCKed
#                       manifest; both arms complete: one record per context and seed)
#   memo-conflicts      per context, the memo keys whose stored results differ between GS, GE and GL (H416 §10 scan)
#   deploy SEED CPUS JOBS MEM     xbabe0 unit h418-deploy-D<n> (D1 = 335101, D2 = 335102): d91_h418.py deploy
#   status              unit states and record counts (counts only)
#   pull                create-once sha-compared read-only copies on xbabe2: $EV/pulled
#   analyze             analyze_h418.py: gates, then the h416.SXB board rows (create-once) and the readout
#   board               the rows into the board export copy and the local board root; tools/ladder.py --write;
#                       LADDER_focus_SXB.md (pre-H418 ladders kept as *.pre-h418.*)
# After an xbabe0 reboot or crash (PREREG §4): re-run the same `gp` / `deploy` step with the same arguments. Stored
# records are returned unchanged, unfinished units resume from their memo, and units.log records each (re)start.
set -euo pipefail
STEP=${1:?step}; shift
STAGE=${H418_STAGE:?H418_STAGE}
REPO=/home/ubuntu/JopeDime
WDIR=research/flywheel_selfplay/h418_sxb_dev91
BRD=research/elo/reference-board/boards/dev91-audit-v1
TOOLS=research/elo/reference-board/tools
EV=/home/ubuntu/jope_plan/evidence/H418
DATA=/home/ubuntu/jope-data/h418-d91            # xbabe0
SV=/home/ubuntu/h335-stage-v                    # xbabe0: H335 Stage V root (read-only use)
GSR=$SV/stagev-root/gp                          # xbabe0: the stored GS run (gproot_AGG/memo, stages/V_AGG)
PREP=x0_prep_20261105T005754Z
RECEIPT=/home/ubuntu/jope_plan/ns/receipts/H418_LOCK_sha256.txt
GATE=$EV/gate_prs.txt
LOCALBOARD=/home/ubuntu/jope-data/h335-stagev/refboard/dev91-audit-v1   # xbabe2 board root
H335CODE=/home/ubuntu/h335-stage/code
SEEDS=(335101 335102)
H418_FILES=(d91_h418.py sandbox_h418_d91.sh analyze_h418.py selftest_h418.py run_h418.sh stage_h418.sh)
GH=/home/ubuntu/.local/bin/gh-role
CRED=(-c credential.helper= -c "credential.helper=!$GH writer auth git-credential")
URL=https://github.com/neverhuman/JopeDime.git
x0() { ssh -o BatchMode=yes xbabe0 "$@"; }
die() { echo "run_h418 refused: $*" >&2; exit 3; }
[ "$(hostname)" = xbabe2 ] || die "run on xbabe2"
(cd "$STAGE/code" && sha256sum --quiet -c ../CODE.sha256) || die "staged code differs from CODE.sha256"
MODE=$(cat "$STAGE/mode")
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$EV/logs" "$EV/results" "$EV/inputs"
case $STEP in
  selftest|import-check|gate|precheck) ;;
  *)
    [ "$MODE" = pub ] || die "$STAGE is not a pub stage (scoring steps stage the published bytes)"
    git -C "$REPO" "${CRED[@]}" fetch -q "$URL" +refs/heads/v1/next:refs/remotes/origin/v1/next
    cmp -s "$STAGE/code/PREREG_H418.md" <(git -C "$REPO" show "origin/v1/next:$WDIR/PREREG.md") \
      || die "staged PREREG is not the H418 PREREG on origin/v1/next: nothing runs before publication"
    for f in "${H418_FILES[@]}"; do
      cmp -s "$STAGE/code/$f" <(git -C "$REPO" show "origin/v1/next:$WDIR/$f") || die "$f differs from origin/v1/next"
    done ;;
esac

gate_ok() {  # every gate PR merged; prints the state of each and the other open PRs
  [ -s "$GATE" ] || { echo "no gate file $GATE" >&2; return 1; }
  local bad=0 pr st
  for pr in $(grep -o '^[0-9]\+' "$GATE"); do
    st=$("$GH" writer pr view "$pr" --repo neverhuman/JopeDime --json state,mergedAt --jq '.state+" "+(.mergedAt // "")')
    echo "gate PR $pr: $st"
    [ "${st%% *}" = MERGED ] || bad=1
  done
  echo "open PRs now: $("$GH" writer pr list --repo neverhuman/JopeDime --state open --json number \
    --jq '[.[].number] | map(tostring) | join(" ")')"
  return "$bad"
}
sandbox() {  # sandbox MODE CPUS MEM CMD...: a short foreground docker sandbox call on xbabe0
  local m=$1 c=$2 mem=$3; shift 3
  x0 "H418_STAGE=$STAGE H418_CPUSET=$c H418_MEMORY=$mem bash $STAGE/code/sandbox_h418_d91.sh $m $*"
}
unit() {  # unit NAME CPUS MEM CMD...: a long xbabe0 systemd-run user unit (docker sandbox, run mode), not waited for
  local name=$1 c=$2 mem=$3; shift 3
  x0 "mkdir -p $DATA/logs && ! systemctl --user is-active --quiet $name && systemd-run --user --unit $name --collect \
    -p MemoryMax=${mem^^} -p MemorySwapMax=0 -p Nice=19 --setenv=H418_STAGE=$STAGE --setenv=H418_CPUSET=$c \
    --setenv=H418_MEMORY=$mem -p StandardOutput=append:$DATA/logs/$name.log \
    -p StandardError=append:$DATA/logs/$name.err bash $STAGE/code/sandbox_h418_d91.sh run nice -n 19 $*"
}
seed_tag() { case $1 in 335101) echo D1 ;; 335102) echo D2 ;; *) die "seed $1" ;; esac; }
prereg_commit() { git -C "$REPO" log -1 --format=%H origin/v1/next -- "$WDIR/PREREG.md"; }

case $STEP in
  selftest)
    DIR=${1:?scratch dir}
    [ ! -e "$DIR" ] || die "$DIR exists"
    mkdir -p "$DIR"  # the published tools/ladder.py, for the ladder cross-check on the synthetic board
    git -C "$REPO" show "origin/v1/next:$TOOLS/ladder.py" > "$DIR/ladder.py"
    REFBOARD_H335_DIR="$H335CODE" nice -n 19 python3 -B "$STAGE/code/selftest_h418.py" --dir "$DIR/cases" \
      --ladder "$DIR/ladder.py" | tee "$DIR/selftest.txt"
    # a pass record for this exact stage (CODE.sha256), required by `inputs` (PREREG §3.1)
    printf '%s %s %s\n' "$(sha256sum "$STAGE/CODE.sha256" | cut -d' ' -f1)" "$(date -u +%FT%TZ)" "$DIR" \
      >> "$EV/logs/selftest_pass.txt" ;;
  import-check)
    sandbox prelock 16-17 4g python3 -u d91_h418.py --help | head -3 ;;
  gate) gate_ok ;;
  precheck)
    git -C "$REPO" "${CRED[@]}" fetch -q "$URL" +refs/heads/v1/next:refs/remotes/origin/v1/next \
      +refs/heads/main:refs/remotes/origin/main
    for ref in origin/main origin/v1/next; do
      n=$(git -C "$REPO" ls-tree --name-only "$ref:$BRD/scores/" 2>/dev/null | grep -c '^h416\.' || true)
      echo "$ref: $(git -C "$REPO" ls-tree --name-only "$ref:$BRD/scores/" 2>/dev/null | wc -l) score files, h416.* = $n"
      [ "$n" = 0 ] || die "an h416 score exists on $ref"
    done
    ! compgen -G "$LOCALBOARD/scores/h416.*" > /dev/null || die "an h416 score exists in $LOCALBOARD"
    [ ! -e "$RECEIPT" ] || die "$RECEIPT exists on xbabe2"
    x0 "[ ! -e $RECEIPT ] && [ ! -e $DATA/stage_v/LOCK.json ]" || die "an H418 receipt or LOCK exists on xbabe0"
    echo "precheck OK $(date -u +%FT%TZ): no h416.* dev91 score, no H418 LOCK" | tee -a "$EV/logs/precheck.log" ;;
  inputs)
    grep -q "^$(sha256sum "$STAGE/CODE.sha256" | cut -d' ' -f1) " "$EV/logs/selftest_pass.txt" 2>/dev/null \
      || die "no passing selftest for this stage (run_h418.sh selftest DIR first)"
    PC=$(prereg_commit); [ ${#PC} = 40 ] || die "no PREREG commit on origin/v1/next"
    B=$EV/board/dev91-audit-v1
    [ ! -e "$B" ] && [ ! -e "$EV/inputs/board_audit.sha256" ] || die "board export exists (create-once)"
    x0 "[ ! -e $DATA/h335-prep ] && [ ! -e $DATA/inputs ]" || die "xbabe0 inputs exist (create-once)"
    # H335 prep manifests (table + SEARCH-view shas): the copies H381 and H410 used
    x0 "set -euo pipefail; mkdir -p $DATA/h335-prep $DATA/inputs
      for k in tables root; do cp $SV/$PREP.\$k.sha256 $DATA/h335-prep/
        cmp $DATA/h335-prep/$PREP.\$k.sha256 /home/ubuntu/jope-data/h410-d91/h335-prep/$PREP.\$k.sha256; done
      chmod a-w $DATA/h335-prep/*"
    # the stored GS run: sha256 of every memo and V_AGG record, by path (no file is parsed)
    x0 "set -euo pipefail; cd $GSR; ls gproot_AGG/memo/*.split.jsonl stages/V_AGG/*__335101.jsonl \
      stages/V_AGG/*__335102.jsonl | sed 's#^#$GSR/#' | xargs sha256sum > $DATA/inputs/gs_stage_v.sha256
      [ \$(grep -c '/gproot_AGG/memo/' $DATA/inputs/gs_stage_v.sha256) = 91 ]
      [ \$(grep -c '/stages/V_AGG/' $DATA/inputs/gs_stage_v.sha256) = 182 ]
      chmod a-w $DATA/inputs/gs_stage_v.sha256"
    rsync -a "xbabe0:$DATA/inputs/gs_stage_v.sha256" "$EV/inputs/gs_stage_v.sha256"
    # the board at the PREREG commit: BOARD.json, ladder.json, scores/*.jsonl (31 entrants, no h416.*)
    mkdir -p "$B/scores"
    git -C "$REPO" show "$PC:$BRD/BOARD.json" > "$B/BOARD.json"
    git -C "$REPO" show "$PC:$BRD/ladder.json" > "$B/ladder.json"
    git -C "$REPO" ls-tree --name-only "$PC:$BRD/scores/" | while read -r s; do
      git -C "$REPO" show "$PC:$BRD/scores/$s" > "$B/scores/$s"
    done
    [ "$(find "$B/scores" -maxdepth 1 -type f | wc -l)" = 31 ] && ! compgen -G "$B/scores/h416.*" > /dev/null \
      || die "board export is not the 31-entrant board without h416"
    for s in "$B"/scores/*.jsonl; do  # the local board root holds the same score bytes
      cmp -s "$s" "$LOCALBOARD/scores/$(basename "$s")" || echo "note: $(basename "$s") differs from $LOCALBOARD"
    done | tee "$EV/logs/board_local_cmp.txt"
    (cd "$B" && find . -type f | sort | sed 's#^\./##' | xargs sha256sum) > "$EV/inputs/board_audit.sha256"
    chmod -R a-w "$B" "$EV/inputs/board_audit.sha256"
    rsync -a "$EV/inputs/board_audit.sha256" "xbabe0:$DATA/inputs/board_audit.sha256"
    jq -n --arg p "$PC" --arg s "$(cat "$STAGE/source_commit")" '{prereg_commit: $p, stage_commit: $s}' \
      > "$EV/inputs/spec.json"
    rsync -a "$EV/inputs/spec.json" "xbabe0:$DATA/inputs/spec.json"
    x0 "chmod a-w $DATA/inputs/*"
    echo "inputs: PREREG $PC; GS manifest $(sha256sum "$EV/inputs/gs_stage_v.sha256" | cut -c1-16); board manifest" \
      "$(sha256sum "$EV/inputs/board_audit.sha256" | cut -c1-16) ($(wc -l < "$EV/inputs/board_audit.sha256") files)" \
      | tee -a "$EV/logs/inputs.log" ;;
  lock)
    [ ! -e "$RECEIPT" ] || die "$RECEIPT exists"
    x0 "[ ! -e $RECEIPT ] && [ ! -e $DATA/stage_v/LOCK.json ] && [ -s $DATA/inputs/spec.json ]" \
      || die "xbabe0: receipt / LOCK exists or no spec"
    sandbox prelock 16-23 8g python3 -u d91_h418.py lock --spec "$DATA/inputs/spec.json" | tee "$EV/logs/lock.log"
    S=$(x0 "sha256sum $DATA/stage_v/LOCK.json" | cut -d' ' -f1)
    [ "$(tail -n 1 "$EV/logs/lock.log")" = "$S" ] || die "LOCK sha differs from the driver's output"
    x0 "set -o noclobber; echo '$S  $DATA/stage_v/LOCK.json' > $RECEIPT && chmod 0444 $RECEIPT"
    ( set -o noclobber; echo "$S  $DATA/stage_v/LOCK.json" > "$RECEIPT" ) && chmod 0444 "$RECEIPT"
    mkdir -p "$EV/stage_v"
    rsync -a "xbabe0:$DATA/stage_v/LOCK.json" "$EV/stage_v/LOCK.json"
    [ "$(jq '[.code_sha256 | keys[] | ascii_downcase | select(test("dev91|dl-val|dl_val|dlval|hold_out|hold-out"))]
      | length' "$EV/stage_v/LOCK.json")" = 0 ] || die "a DL-Val token in LOCK code paths"
    echo "LOCK $S; receipt $RECEIPT (xbabe0 + xbabe2)" | tee -a "$EV/logs/lock.log" ;;
  announce)
    CPUS=${1:?CPUS}; HOURS=${2:?HOURS}
    LINE="$(date -u +%FT%TZ) cl_fwriter (Claude): H418 taking xbabe0 cpus $CPUS (docker, nice 19) for about $HOURS h: \
the h416.SXB dev91-audit-v1 once-score (GE/GL GP arms, then the D1/D2 deploys). @cl_apex: please pause your xbabe0 \
units on those cpus now; I will post here when they are free."
    printf '%s\n' "$LINE" | (cd "$REPO" && flock AGENT_CHAT.md.lock tee -a AGENT_CHAT.md >/dev/null)
    printf '%s\n' "$LINE" >> "$EV/logs/announce.txt"; echo "$LINE" ;;
  unlock)
    if ! gate_ok | tee -a "$EV/logs/gate.log"; then die "merge-all gate not met (nothing unlocked)"; fi
    x0 "[ -s $RECEIPT ] && [ ! -e $DATA/stage_v/STAGE_V_STARTED ]" || die "no receipt, or already unlocked"
    sandbox unlock 16-23 8g python3 -u d91_h418.py unlock | tee "$EV/logs/unlock.log"
    sandbox run 16-23 8g python3 -u d91_h418.py select | tee "$EV/logs/select.log"
    rsync -a "xbabe0:$DATA/cids91.json" "xbabe0:$DATA/stage_v/STAGE_V_STARTED" "$EV/stage_v/"
    chmod a-w "$EV"/stage_v/* ;;
  gp)
    ARM=${1:?ARM}; CPUS=${2:?CPUS}; JOBS=${3:?JOBS}; MEM=${4:?MEM}
    case $ARM in GE|GL) ;; *) die "GP arms are GE and GL (GS = the stored H335 Stage V run)" ;; esac
    [ -s "$EV/logs/announce.txt" ] || die "post the xbabe0 window first (announce)"
    x0 "[ -s $DATA/cids91.json ]" || die "unlock / select first"
    unit "h418-gp-$ARM" "$CPUS" "$MEM" python3 -u d91_h418.py gp --arm "$ARM" --run-dir "$DATA/runs/$ARM" \
      --cores "$CPUS" --jobs "$JOBS"
    echo "started h418-gp-$ARM on xbabe0 cpus $CPUS $(date -u +%FT%TZ)" | tee -a "$EV/logs/units.log" ;;
  deploy-prep)
    MAN=$EV/logs/memo_deploy.sha256
    [ ! -e "$MAN" ] || die "$MAN exists (create-once)"
    x0 "! systemctl --user is-active --quiet h418-gp-GE && ! systemctl --user is-active --quiet h418-gp-GL" \
      || die "a GP arm is still running"
    # GS sources unchanged since the LOCK; every arm unit recorded; then the per-context concatenation GS GE GL
    x0 "set -euo pipefail; sha256sum --quiet -c $DATA/inputs/gs_stage_v.sha256
      for d in D1 D2; do [ ! -e $DATA/runs/\$d ] || { echo \"\$d exists (create-once)\" >&2; exit 3; }; done
      for c in \$(jq -r '.cids[]' $DATA/cids91.json); do
        src=$GSR/gproot_AGG/memo/\$c.split.jsonl
        for a in GE GL; do
          for s in ${SEEDS[*]}; do
            [ \$(grep -c '\"type\": \"result\"' $DATA/runs/\$a/stages/E416_AGG/\${c}__\$s.jsonl) = 1 ] \
              || { echo \"arm \$a \${c:0:12} seed \$s: no single record\" >&2; exit 4; }
          done
          [ -s $DATA/runs/\$a/gproot_AGG/memo/\$c.split.jsonl ] || { echo \"arm \$a \${c:0:12}: no memo\" >&2; exit 4; }
          src=\"\$src $DATA/runs/\$a/gproot_AGG/memo/\$c.split.jsonl\"
        done
        for d in D1 D2; do
          mkdir -p $DATA/runs/\$d/gproot_AGG/memo
          cat \$src > $DATA/runs/\$d/gproot_AGG/memo/\$c.split.jsonl
        done
        sha256sum \$src $DATA/runs/D1/gproot_AGG/memo/\$c.split.jsonl $DATA/runs/D2/gproot_AGG/memo/\$c.split.jsonl
      done" > "$MAN"
    chmod a-w "$MAN"
    [ "$(grep -c '/runs/D1/gproot_AGG/memo/' "$MAN")" = 91 ] || die "deploy-prep: not 91 memos"
    echo "deploy-prep: 91 contexts x (D1, D2); manifest $MAN $(sha256sum "$MAN" | cut -c1-16)" ;;
  memo-conflicts)
    OUT=$EV/results/memo_conflicts.jsonl
    [ -f "$EV/logs/memo_deploy.sha256" ] || die "deploy-prep first"
    [ ! -e "$OUT" ] || die "$OUT exists (create-once)"
    grep -v " $DATA/runs/D[12]/" "$EV/logs/memo_deploy.sha256" | x0 "sha256sum --quiet -c"  # sources unchanged
    x0 "set -euo pipefail; for c in \$(jq -r '.cids[]' $DATA/cids91.json); do
        for a in GS=$GSR GE=$DATA/runs/GE GL=$DATA/runs/GL; do
          jq -c --arg a \${a%%=*} '{a: \$a, k: (.kind + \"|\" + .env_sha256 + \"|\" + .sha), r: [.fitness, .error, .status]}' \
            \${a#*=}/gproot_AGG/memo/\$c.split.jsonl
        done | jq -s -c --arg c \$c '[group_by(.k)[] | select(length > 1)] as \$d
          | [\$d[] | select((map(.r) | unique | length) > 1)] as \$x
          | {cid: \$c, shared_keys: (\$d | length),
             gs_conflicts: ([\$x[] | select(any(.[]; .a == \"GS\"))] | length),
             arm_conflicts: ([\$x[] | select(all(.[]; .a != \"GS\"))] | length),
             conflicts: [\$x[] | {key: .[0].k, results: map({run: .a, fitness: .r[0], error: .r[1], status: .r[2]})}]}'
      done" > "$OUT"
    chmod a-w "$OUT"
    [ "$(wc -l < "$OUT")" = 91 ] || die "memo-conflicts: not 91 contexts"
    jq -s -c '{contexts: length, shared_key_contexts: (map(select(.shared_keys > 0)) | length),
      gs_conflict_contexts: [.[] | select(.gs_conflicts > 0) | .cid[0:12]],
      arm_conflict_contexts: [.[] | select(.arm_conflicts > 0) | .cid[0:12]]}' "$OUT" | tee -a "$EV/logs/memo_conflicts.log" ;;
  deploy)
    SEED=${1:?SEED}; CPUS=${2:?CPUS}; JOBS=${3:?JOBS}; MEM=${4:?MEM}
    TAG=$(seed_tag "$SEED")
    [ -s "$EV/logs/announce.txt" ] || die "post the xbabe0 window first (announce)"
    [ -f "$EV/results/memo_conflicts.jsonl" ] || die "memo-conflicts first (pre-registered before the deploy)"
    unit "h418-deploy-$TAG" "$CPUS" "$MEM" python3 -u d91_h418.py deploy --seed "$SEED" --tag "$TAG" \
      --run-dir "$DATA/runs/$TAG" --cores "$CPUS" --jobs "$JOBS" --inner-jobs 3
    echo "started h418-deploy-$TAG on xbabe0 cpus $CPUS $(date -u +%FT%TZ)" | tee -a "$EV/logs/units.log" ;;
  status)
    x0 "for u in h418-gp-GE h418-gp-GL h418-deploy-D1 h418-deploy-D2; do
        echo \"\$u \$(systemctl --user is-active \$u 2>/dev/null || true)\"; done
      for a in GE GL; do [ -d $DATA/runs/\$a/stages/E416_AGG ] && \
        echo \"\$a records \$(ls $DATA/runs/\$a/stages/E416_AGG | wc -l)/182\"; done
      for d in D1 D2; do [ -d $DATA/runs/\$d/stages/H416_\$d ] && \
        echo \"\$d records \$(ls $DATA/runs/\$d/stages/H416_\$d | wc -l)/91\"; done; uptime" ;;
  pull)
    P=$EV/pulled
    [ ! -e "$P" ] || die "$P exists (create-once)"
    x0 "! systemctl --user is-active --quiet h418-deploy-D1 && ! systemctl --user is-active --quiet h418-deploy-D2" \
      || die "a deploy is still running"
    mkdir -p "$P/GS/stages"
    for d in D1 D2; do
      mkdir -p "$P/$d"
      rsync -a "xbabe0:$DATA/runs/$d/config_$d.json" "xbabe0:$DATA/runs/$d/stages" "xbabe0:$DATA/runs/$d/h416_${d}_npz" \
        "$P/$d/"
      rsync -a "xbabe0:$DATA/runs/$d/" --include='status_*.json' --exclude='*' "$P/$d/"
    done
    for a in GE GL; do
      mkdir -p "$P/$a"
      rsync -a "xbabe0:$DATA/runs/$a/config.json" "xbabe0:$DATA/runs/$a/stages" "$P/$a/"
      rsync -a "xbabe0:$DATA/runs/$a/" --include='status_*.json' --exclude='*' "$P/$a/"
    done
    rsync -a "xbabe0:$GSR/stages/V_AGG" "$P/GS/stages/"
    rsync -a "xbabe0:$DATA/cids91.json" "xbabe0:$DATA/stage_v/LOCK.json" "$P/"
    for d in D1 D2 GE GL; do
      (cd "$P/$d" && find . -type f | sort | xargs sha256sum) > "$P/$d.sha256"
      x0 "cd $DATA/runs/$d && sha256sum --quiet -c" < "$P/$d.sha256"
    done
    (cd "$P/GS" && find . -type f | sort | xargs sha256sum) > "$P/GS.sha256"
    x0 "cd $GSR && sha256sum --quiet -c" < "$P/GS.sha256"
    # the pulled GS records are the LOCKed bytes
    join -1 2 -2 2 <(sed "s#  $GSR/#  #" "$EV/inputs/gs_stage_v.sha256" | grep ' stages/' | sort -k2) \
      <(sed 's#  \./#  #' "$P/GS.sha256" | sort -k2) | awk '$2 != $3 {bad++} END {exit ((bad > 0) || (NR != 182))}' \
      || die "pulled GS records differ from the LOCKed manifest"
    cmp -s "$P/LOCK.json" "$EV/stage_v/LOCK.json" || die "LOCK.json differs from the lock-time copy"
    chmod -R a-w "$P"
    echo "pulled $(find "$P" -type f | wc -l) files to $P (sha-compared with xbabe0)" | tee -a "$EV/logs/pull.log" ;;
  analyze)
    P=$EV/pulled
    [ -d "$P" ] || die "pull first"
    for d in D1 D2 GE GL GS; do (cd "$P/$d" && sha256sum --quiet -c "$P/$d.sha256"); done  # unchanged since pull
    rc=0
    nice -n 19 python3 -B "$STAGE/code/analyze_h418.py" --pulled "$P" --board "$EV/board/dev91-audit-v1" \
      --board-manifest "$EV/inputs/board_audit.sha256" --lock "$EV/stage_v/LOCK.json" \
      --design "$STAGE/code/DESIGN.json" --conflicts "$EV/results/memo_conflicts.jsonl" \
      --out "$EV/results/H418_DEV91.json" --rows "$EV/results/h416.SXB.jsonl" > "$EV/results/H418_DEV91.txt" 2>&1 || rc=$?
    chmod a-w "$EV/results/H418_DEV91.txt"
    for f in H418_DEV91.json h416.SXB.jsonl; do [ -e "$EV/results/$f" ] && chmod a-w "$EV/results/$f"; done
    cat "$EV/results/H418_DEV91.txt"; echo "analyze exit $rc"; exit "$rc" ;;
  board)
    ROWS=$EV/results/h416.SXB.jsonl
    [ -s "$ROWS" ] || die "no rows (analyze first)"
    PC=$(jq -r .prereg_commit "$EV/inputs/spec.json")
    BN=$EV/board-new/dev91-audit-v1
    [ ! -e "$BN" ] || die "$BN exists (create-once)"
    [ ! -e "$LOCALBOARD/scores/h416.SXB.jsonl" ] || die "$LOCALBOARD already holds h416.SXB"
    mkdir -p "$EV/tools" "$BN"
    git -C "$REPO" show "$PC:$TOOLS/ladder.py" > "$EV/tools/ladder.py"
    cp -r "$EV/board/dev91-audit-v1/." "$BN/" && chmod -R u+w "$BN"
    cp "$ROWS" "$BN/scores/h416.SXB.jsonl"
    LAD=(env REFBOARD_H335_DIR="$H335CODE" nice -n 19 python3 -B "$EV/tools/ladder.py")
    "${LAD[@]}" "$BN" --write > "$EV/results/LADDER_dev91_h418.md"
    "${LAD[@]}" "$BN" --focus h416.SXB > "$EV/results/LADDER_focus_SXB.md"
    jq -e --slurpfile a "$EV/results/H418_DEV91.json" \
      '(.elo.elo["h416.SXB"].elo - $a[0].readout.board_elo["h416.SXB"].elo) | fabs < 1e-9' "$BN/ladder.json" >/dev/null \
      || die "ladder.py's h416.SXB Elo differs from analyze_h418's"
    # the local board root: keep the pre-H418 ladder, add the rows (0444), regenerate
    ( set -o noclobber; cat "$LOCALBOARD/ladder.json" > "$LOCALBOARD/ladder.pre-h418.json"
      cat "$LOCALBOARD/LADDER.md" > "$LOCALBOARD/LADDER.pre-h418.md" )
    install -m 0444 "$ROWS" "$LOCALBOARD/scores/h416.SXB.jsonl"
    "${LAD[@]}" "$LOCALBOARD" --write > /dev/null
    cmp -s <(jq -S .elo "$LOCALBOARD/ladder.json") <(jq -S .elo "$BN/ladder.json") \
      || echo "note: the local board root ladder differs from the export's (extra local files?)"
    head -n 40 "$EV/results/LADDER_dev91_h418.md" ;;
  *) die "unknown step $STEP" ;;
esac
