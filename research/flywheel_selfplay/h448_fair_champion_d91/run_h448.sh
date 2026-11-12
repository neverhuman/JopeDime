#!/bin/bash
# H448 orchestration on xbabe2 (PREREG §3-§4). Every scoring step refuses unless the stage is a `pub` stage whose
# H448 files + FROZEN_FAIR_CHAMPION.json are byte-identical to origin/v1/next.
#   H448_STAGE=/home/ubuntu/h448-stage/pub-TAG run_h448.sh STEP [args]
#   selftest DIR        [no d91] selftest_h448.py on synthetic fixtures + published tools/ladder.py cross-check
#   import-check        [no d91] xbabe0 docker prelock (TRAIN-only view): `d91_h448.py --help`
#   precheck            [no d91] no h448.* score on main, v1/next, or local board; H418 D1/D2 shas verified; no LOCK
#   inputs              [no d91] create-once: DATA/h335-prep, H418 D1/D2 manifests, 32-entrant board export + manifest,
#                       ICL checkpoint & training_bank assets + icl_assets.sha256, spec.json. Hashes only; 0 values read.
#   prelock             [no d91] xbabe0 prelock sandbox: d91_h448.py lock -> LOCK.json; prints LOCK sha256 for owner
#                       Stage-V authorization (does NOT write ns/receipts/H448_LOCK_sha256.txt itself).
#   announce CPUS HOURS post the xbabe0 core window on AGENT_CHAT.md (required before `cols`)
#   unlock              verifies H445 precondition (or OWNER_DIRECT_D91 in receipt), syncs receipt to xbabe0 if needed,
#                       runs one-shot Stage V unlock + `bank_and_screen` (cids91.json, bank_d91_search.npz, H425 screen)
#   cols                xbabe0 docker `run` sandbox: M7 (cols_matern.py), K3 (cols_h430.py), F4 (cols_h433.py), and
#                       frozen ICL (cols_r4_e8.py, 64 single-thread LPT workers on 16-63,80-95); posts release when done
#   pull                create-once sha-verified read-only copies on xbabe2: $EV/pulled
#   analyze             analyze_h448.py: pre-readout gates (incl. H418 SXB replay <= 1e-9), then create-once rows + JSON
#   board               adds h448.APEX_FAIR_CHAMPION.jsonl to board-new & local board; tools/ladder.py --write (< 1e-9)
set -euo pipefail
STEP=${1:?step}; shift
STAGE=${H448_STAGE:?H448_STAGE}
REPO=/home/ubuntu/JopeDime
WDIR=research/flywheel_selfplay/h448_fair_champion_d91
G447=research/flywheel_selfplay/h447_legal_icl_104m_24m_r4e8
BRD=research/elo/reference-board/boards/dev91-audit-v1
TOOLS=research/elo/reference-board/tools
EV=/home/ubuntu/jope_plan/evidence/H448
P418=/home/ubuntu/jope_plan/evidence/H418/pulled
H445_RO=/home/ubuntu/jope_plan/evidence/H445/results/READOUT_DEV.json
DATA=/home/ubuntu/jope-data/h448-d91            # xbabe0
SV=/home/ubuntu/h335-stage-v                    # xbabe0: H335 Stage V root (read-only use)
PREP=x0_prep_20261105T005754Z
RECEIPT=/home/ubuntu/jope_plan/ns/receipts/H448_LOCK_sha256.txt
LOCALBOARD=/home/ubuntu/jope-data/h335-stagev/refboard/dev91-audit-v1
H335CODE=/home/ubuntu/h335-stage/code
H448_FILES=(d91_h448.py sandbox_h448_d91.sh analyze_h448.py selftest_h448.py run_h448.sh stage_h448.sh)
GH=/home/ubuntu/.local/bin/gh-role
CRED=(-c credential.helper= -c "credential.helper=!$GH writer auth git-credential")
URL=https://github.com/neverhuman/JopeDime.git
SSH_OPTS=(-i /home/ubuntu/.ssh/xbabe2_ed25519 -o BatchMode=yes)
x0() { ssh "${SSH_OPTS[@]}" xbabe0 "$@"; }
rs0() { rsync -a -e "ssh ${SSH_OPTS[*]}" "$@"; }
die() { echo "run_h448 refused: $*" >&2; exit 3; }

[ "$(hostname)" = xbabe2 ] || die "run on xbabe2"
(cd "$STAGE/code" && sha256sum --quiet -c ../CODE.sha256) || die "staged code differs from CODE.sha256"
MODE=$(cat "$STAGE/mode")
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$EV/logs" "$EV/results" "$EV/inputs"

case $STEP in
  selftest|import-check|precheck) ;;
  *)
    [ "$MODE" = pub ] || die "$STAGE is not a pub stage (scoring steps stage the published bytes)"
    git -C "$REPO" "${CRED[@]}" fetch -q "$URL" +refs/heads/v1/next:refs/remotes/origin/v1/next
    cmp -s "$STAGE/code/PREREG_H448.md" <(git -C "$REPO" show "origin/v1/next:$WDIR/PREREG.md") \
      || die "staged PREREG is not the H448 PREREG on origin/v1/next"
    cmp -s "$STAGE/code/FROZEN_FAIR_CHAMPION.json" <(git -C "$REPO" show "origin/v1/next:$G447/FROZEN_FAIR_CHAMPION.json") \
      || die "staged FROZEN_FAIR_CHAMPION.json differs from origin/v1/next"
    for f in "${H448_FILES[@]}"; do
      cmp -s "$STAGE/code/$f" <(git -C "$REPO" show "origin/v1/next:$WDIR/$f") || die "$f differs from origin/v1/next"
    done ;;
esac

sandbox() {
  local m=$1 c=$2 mem=$3; shift 3
  x0 "H448_STAGE=$STAGE H448_CPUSET=$c H448_MEMORY=$mem bash $STAGE/code/sandbox_h448_d91.sh $m $*"
}
prereg_commit() { git -C "$REPO" log -1 --format=%H origin/v1/next -- "$WDIR/PREREG.md"; }

case $STEP in
  selftest)
    DIR=${1:?scratch dir}
    [ ! -e "$DIR" ] || die "$DIR exists"
    mkdir -p "$DIR"
    git -C "$REPO" show "origin/v1/next:$TOOLS/ladder.py" > "$DIR/ladder.py"
    REFBOARD_H335_DIR="$H335CODE" nice -n 19 python3 -B "$STAGE/code/selftest_h448.py" --dir "$DIR/cases" \
      --ladder "$DIR/ladder.py" | tee "$DIR/selftest.txt"
    printf '%s %s %s\n' "$(sha256sum "$STAGE/CODE.sha256" | cut -d' ' -f1)" "$(date -u +%FT%TZ)" "$DIR" \
      >> "$EV/logs/selftest_pass.txt" ;;

  import-check)
    sandbox prelock 16-17 4g python3 -u d91_h448.py --help | head -n 5 ;;

  precheck)
    git -C "$REPO" "${CRED[@]}" fetch -q "$URL" +refs/heads/v1/next:refs/remotes/origin/v1/next \
      +refs/heads/main:refs/remotes/origin/main
    for ref in origin/main origin/v1/next; do
      n=$(git -C "$REPO" ls-tree --name-only "$ref:$BRD/scores/" 2>/dev/null | grep -c '^h448\.' || true)
      echo "$ref: $(git -C "$REPO" ls-tree --name-only "$ref:$BRD/scores/" 2>/dev/null | wc -l) score files, h448.* = $n"
      [ "$n" = 0 ] || die "an h448 score exists on $ref"
    done
    ! compgen -G "$LOCALBOARD/scores/h448.*" > /dev/null || die "an h448 score exists in $LOCALBOARD"
    for d in D1 D2; do
      (cd "$P418/$d" && sha256sum --quiet -c "$P418/$d.sha256") || die "H418 pulled $d sha check failed"
    done
    [ ! -e "$RECEIPT" ] || die "$RECEIPT exists on xbabe2"
    x0 "[ ! -e $RECEIPT ] && [ ! -e $DATA/stage_v/LOCK.json ]" || die "an H448 receipt or LOCK exists on xbabe0"
    echo "precheck OK $(date -u +%FT%TZ): no h448.* score, H418 D1/D2 shas verified, no H448 LOCK" \
      | tee -a "$EV/logs/precheck.log" ;;

  inputs)
    grep -q "^$(sha256sum "$STAGE/CODE.sha256" | cut -d' ' -f1) " "$EV/logs/selftest_pass.txt" 2>/dev/null \
      || die "no passing selftest for this stage (run_h448.sh selftest DIR first)"
    PC=$(prereg_commit); [ ${#PC} = 40 ] || die "no PREREG commit on origin/v1/next"
    B=$EV/board/dev91-audit-v1
    [ ! -e "$B" ] && [ ! -e "$EV/inputs/board_audit.sha256" ] || die "board export exists (create-once)"
    x0 "[ ! -e $DATA/h335-prep ] && [ ! -e $DATA/inputs ]" || die "xbabe0 inputs exist (create-once)"

    # 1. H335 prep manifests on xbabe0
    x0 "set -euo pipefail; mkdir -p $DATA/h335-prep $DATA/inputs
      for k in tables root; do cp $SV/$PREP.\$k.sha256 $DATA/h335-prep/
        cmp $DATA/h335-prep/$PREP.\$k.sha256 /home/ubuntu/jope-data/h418-d91/h335-prep/$PREP.\$k.sha256; done
      chmod a-w $DATA/h335-prep/*"

    # 2. H418 D1 & D2 pulled manifests
    for d in D1 D2; do
      (cd "$P418/$d" && sha256sum --quiet -c "$P418/$d.sha256") || die "H418 pulled $d sha mismatch"
    done
    cp "$P418/D1.sha256" "$EV/inputs/d1_pulled.sha256"
    cp "$P418/D2.sha256" "$EV/inputs/d2_pulled.sha256"
    chmod 0444 "$EV/inputs/d1_pulled.sha256" "$EV/inputs/d2_pulled.sha256"
    rs0 "$EV/inputs/d1_pulled.sha256" "$EV/inputs/d2_pulled.sha256" "xbabe0:$DATA/inputs/"

    # 3. Board at PREREG commit (32 entrants including h416.SXB & kumo_tabular_large_e8.v1, 0 h448.*)
    mkdir -p "$B/scores"
    git -C "$REPO" show "$PC:$BRD/BOARD.json" > "$B/BOARD.json"
    git -C "$REPO" show "$PC:$BRD/ladder.json" > "$B/ladder.json"
    git -C "$REPO" ls-tree --name-only "$PC:$BRD/scores/" | while read -r s; do
      git -C "$REPO" show "$PC:$BRD/scores/$s" > "$B/scores/$s"
    done
    [ "$(find "$B/scores" -maxdepth 1 -type f | wc -l)" = 32 ] && ! compgen -G "$B/scores/h448.*" > /dev/null \
      || die "board export is not the 32-entrant board without h448"
    (cd "$B" && find . -type f | sort | sed 's#^\./##' | xargs sha256sum) > "$EV/inputs/board_audit.sha256"
    chmod -R a-w "$B" "$EV/inputs/board_audit.sha256"
    rs0 "$EV/inputs/board_audit.sha256" "xbabe0:$DATA/inputs/board_audit.sha256"

    # 4. Frozen champion training_bank + ICL checkpoints -> xbabe0:$DATA/inputs + icl_assets.sha256
    TB_NAME=$(jq -r '.blocks.icl.training_bank // "fit_icl_bank.npz"' "$STAGE/code/FROZEN_FAIR_CHAMPION.json")
    if [ "$TB_NAME" = "all_legal_excl_p201b_bank.npz" ]; then
      TB_SRC=/home/ubuntu/jope-data/h335-cl/h446/pub-r6/banks/all_legal_excl_p201b_bank.npz
    elif [ "$TB_NAME" = "all_legal_bank.npz" ]; then
      TB_SRC=/home/ubuntu/jope-data/h335-cl/h446/pub-r6/banks/all_legal_bank.npz
    else
      TB_SRC=/home/ubuntu/jope-data/apex-icl/bank/fit_icl_bank.npz
    fi
    CK_L_BASE=/home/ubuntu/jope-data/h447/inputs/ICL_L_g1_w2_ckpt_h030.pt
    CK_M_BASE=/home/ubuntu/jope-data/h447/inputs/ICL_M_r3_ckpt_h040.pt
    CK_M20_BASE=/home/ubuntu/jope-data/h424/ICL_M_r3_ckpt_h020.pt
    ASSETS=("$TB_SRC" "$CK_L_BASE" "$CK_M_BASE" "$CK_M20_BASE")
    rs0 "${ASSETS[@]}" "xbabe0:$DATA/inputs/"
    P_L=$(jq -r '.blocks.icl.ckpt_l // empty' "$STAGE/code/FROZEN_FAIR_CHAMPION.json")
    P_M=$(jq -r '.blocks.icl.ckpt_m // empty' "$STAGE/code/FROZEN_FAIR_CHAMPION.json")
    if [ -n "$P_L" ] && [ -f "$P_L" ]; then rs0 "$P_L" "xbabe0:$DATA/inputs/ckpt_frozen_L.pt"; fi
    if [ -n "$P_M" ] && [ -f "$P_M" ]; then rs0 "$P_M" "xbabe0:$DATA/inputs/ckpt_frozen_M.pt"; fi
    x0 "set -euo pipefail; cd $DATA/inputs
      ls -1 *.npz *.pt | sort -u | xargs sha256sum > icl_assets.sha256
      chmod 0444 *.npz *.pt icl_assets.sha256"
    rs0 "xbabe0:$DATA/inputs/icl_assets.sha256" "$EV/inputs/icl_assets.sha256"
    chmod 0444 "$EV/inputs/icl_assets.sha256"

    # 5. spec.json
    jq -n --arg p "$PC" --arg s "$(cat "$STAGE/source_commit")" '{prereg_commit: $p, stage_commit: $s}' \
      > "$EV/inputs/spec.json"
    chmod 0444 "$EV/inputs/spec.json"
    rs0 "$EV/inputs/spec.json" "xbabe0:$DATA/inputs/spec.json"
    x0 "chmod a-w $DATA/inputs/*"
    echo "inputs OK: PREREG $PC; board $(wc -l < "$EV/inputs/board_audit.sha256") files;" \
      "icl_assets $(wc -l < "$EV/inputs/icl_assets.sha256") files" | tee -a "$EV/logs/inputs.log" ;;

  prelock)
    [ ! -e "$RECEIPT" ] || die "$RECEIPT already exists"
    x0 "[ ! -e $RECEIPT ] && [ ! -e $DATA/stage_v/LOCK.json ] && [ -s $DATA/inputs/spec.json ]" \
      || die "xbabe0: receipt / LOCK exists or no spec"
    sandbox prelock 16-23 8g python3 -u d91_h448.py lock --spec "$DATA/inputs/spec.json" | tee "$EV/logs/prelock.log"
    S=$(x0 "sha256sum $DATA/stage_v/LOCK.json" | cut -d' ' -f1)
    [ "$(tail -n 1 "$EV/logs/prelock.log")" = "$S" ] || die "LOCK sha differs from driver output"
    mkdir -p "$EV/stage_v"
    rs0 "xbabe0:$DATA/stage_v/LOCK.json" "$EV/stage_v/LOCK.json"
    chmod 0444 "$EV/stage_v/LOCK.json"
    [ "$(jq '[.code_sha256 | keys[] | ascii_downcase | select(test("dev91|dl-val|dl_val|dlval|hold_out|hold-out"))]
      | length' "$EV/stage_v/LOCK.json")" = 0 ] || die "a DL-Val token in LOCK code paths"
    echo "PRELOCK_READY sha256=$S lock_path=$DATA/stage_v/LOCK.json (awaiting owner Stage-V receipt $RECEIPT)" \
      | tee -a "$EV/logs/prelock.log" ;;

  announce)
    CPUS=${1:?CPUS}; HOURS=${2:?HOURS}
    LINE="$(date -u +%FT%TZ) agy_apex_w10: H448 taking xbabe0 cpus $CPUS (docker --network none, nice 19) for ~${HOURS}h: official fair dev91-audit-v1 once-score of h448.APEX_FAIR_CHAMPION (M7 + K3 + F4 + frozen ICL columns on SEARCH rows; GP pools reused from H418 D1/D2). Will post release when done."
    printf '%s\n' "$LINE" | (cd "$REPO" && flock AGENT_CHAT.md.lock tee -a AGENT_CHAT.md >/dev/null)
    printf '%s\n' "$LINE" >> "$EV/logs/announce.txt"; echo "$LINE" ;;

  unlock)
    [ -s "$RECEIPT" ] || die "missing owner Stage-V receipt $RECEIPT"
    S_LOCK=$(sha256sum "$EV/stage_v/LOCK.json" | cut -d' ' -f1)
    [ "$(awk '{print $1}' "$RECEIPT")" = "$S_LOCK" ] || die "receipt sha does not match $EV/stage_v/LOCK.json"
    # Check H445 precondition (PREREG §4.1) BEFORE unlocking Stage V
    if [ -s "$H445_RO" ]; then
      V445=$(jq -r '.verdict // ""' "$H445_RO")
      if [ "$V445" = HARM ] && ! grep -qw H445_OVERRIDE "$RECEIPT"; then
        die "H445 verdict is HARM and receipt lacks H445_OVERRIDE (nothing unlocked)"
      fi
      echo "H445 precondition satisfied: verdict=$V445" | tee -a "$EV/logs/unlock.log"
    elif grep -qw OWNER_DIRECT_D91 "$RECEIPT"; then
      echo "H445 precondition: OWNER_DIRECT_D91 present in owner receipt" | tee -a "$EV/logs/unlock.log"
    else
      die "missing $H445_RO and receipt lacks OWNER_DIRECT_D91 (nothing unlocked)"
    fi
    # Sync owner receipt to xbabe0 if not already present there
    if ! x0 "[ -s $RECEIPT ]"; then
      R_CONTENT=$(cat "$RECEIPT")
      x0 "mkdir -p $(dirname "$RECEIPT") && set -o noclobber && printf '%s\n' '$R_CONTENT' > $RECEIPT && chmod 0444 $RECEIPT"
    fi
    x0 "[ -s $RECEIPT ] && [ ! -e $DATA/stage_v/STAGE_V_STARTED ]" || die "xbabe0 missing receipt or already unlocked"
    sandbox unlock 16-23 8g python3 -u d91_h448.py unlock | tee -a "$EV/logs/unlock.log"
    sandbox run 16-23 16g python3 -u d91_h448.py bank_and_screen | tee "$EV/logs/bank_and_screen.log"
    rs0 "xbabe0:$DATA/cids91.json" "xbabe0:$DATA/SCREEN_D91_SEARCH.json" "xbabe0:$DATA/stage_v/STAGE_V_STARTED" \
      "$EV/stage_v/"
    chmod a-w "$EV"/stage_v/* ;;

  cols)
    [ -s "$EV/logs/announce.txt" ] || die "post the xbabe0 window first (announce)"
    x0 "[ -s $DATA/bank_d91_search.npz ]" || die "unlock / bank_and_screen first"
    ICL_EN=$(jq -r '.blocks.icl.enabled // false' "$STAGE/code/FROZEN_FAIR_CHAMPION.json")
    ICL_REC=$(jq -r '.blocks.icl.recipe // "NONE"' "$STAGE/code/FROZEN_FAIR_CHAMPION.json")
    CK_L_BN="ICL_L_g1_w2_ckpt_h030.pt"
    SHA_L_DEF="637bc123db18ca027fb46f5113a8f778f3aafb703fecb8c0edc91da7faeebffc"
    CK_M_BN="ICL_M_r3_ckpt_h040.pt"
    SHA_M_DEF="eb3a0903a7da136e2cdedd1fa3f1863fd351121d49f50a4b49e5b0e6d57e9ae4"
    CK_M20_ARG=""
    if [ "$ICL_EN" = true ]; then
      if [ "$ICL_REC" = "CW5_R4_M20" ]; then
        CK_M20_ARG="--ckpt-m20 $DATA/inputs/ICL_M_r3_ckpt_h020.pt --sha-m20 e3c8fd9972ce0c1684630ff06b159e950f6f498c4461d0454e0c6a6f44d38dd7"
      else
        CK_L_BN="ckpt_frozen_L.pt"
        SHA_L_DEF=$(jq -r '.blocks.icl.sha_l' "$STAGE/code/FROZEN_FAIR_CHAMPION.json")
        CK_M_BN="ckpt_frozen_M.pt"
        SHA_M_DEF=$(jq -r '.blocks.icl.sha_m' "$STAGE/code/FROZEN_FAIR_CHAMPION.json")
      fi
    fi

    echo "$(date -u +%FT%TZ) starting H448 cols inside xbabe0 docker sandbox" | tee -a "$EV/logs/cols.log"
    sandbox run "16-39,41-63,80-127" 96g bash -c "'
      set -euo pipefail
      mkdir -p $DATA/logs $DATA/cols_matern_D91 $DATA/cols_k3_D91 $DATA/cols_f4_D91 $DATA/cols_icl_D91
      CPUS=( \$(seq 16 39) \$(seq 41 63) \$(seq 80 127) )

      echo \"\$(date -u +%FT%TZ) [1/4] cols_matern.py (91 shards)\"
      rc=0
      for k in \$(seq 0 90); do
        c=\${CPUS[\$k]}
        taskset -c \"\$c\" nice -n 19 python3 -B cols_matern.py --bank $DATA/bank_d91_search.npz \
          --out $DATA/cols_matern_D91 --shard \"\$k/91\" --ctx-s 900 > $DATA/logs/matern_\$k.log 2>&1 &
      done
      for j in \$(jobs -p); do wait \"\$j\" || rc=\$?; done
      [ \"\$rc\" = 0 ]

      echo \"\$(date -u +%FT%TZ) [2/4] cols_h430.py (91 shards)\"
      for k in \$(seq 0 90); do
        c=\${CPUS[\$k]}
        taskset -c \"\$c\" nice -n 19 python3 -B cols_h430.py --bank $DATA/bank_d91_search.npz \
          --out $DATA/cols_k3_D91 --shard \"\$k/91\" --ctx-s 900 > $DATA/logs/k3_\$k.log 2>&1 &
      done
      for j in \$(jobs -p); do wait \"\$j\" || rc=\$?; done
      [ \"\$rc\" = 0 ]

      echo \"\$(date -u +%FT%TZ) [3/4] cols_h433.py (91 shards)\"
      for k in \$(seq 0 90); do
        c=\${CPUS[\$k]}
        taskset -c \"\$c\" nice -n 19 python3 -B cols_h433.py --bank $DATA/bank_d91_search.npz \
          --out $DATA/cols_f4_D91 --shard \"\$k/91\" --ctx-s 1200 > $DATA/logs/f4_\$k.log 2>&1 &
      done
      for j in \$(jobs -p); do wait \"\$j\" || rc=\$?; done
      [ \"\$rc\" = 0 ]

      if [ \"$ICL_EN\" = true ]; then
        echo \"\$(date -u +%FT%TZ) [4/4] cols_r4_e8.py (91 single-thread LPT workers on 16-39,41-63,80-123 — 1 wave)\"
        ICL_CPUS=( \$(seq 16 39) \$(seq 41 63) \$(seq 80 123) )
        for w in \$(seq 0 90); do
          c=\${ICL_CPUS[\$w]}
          taskset -c \"\$c\" nice -n 19 python3 -B cols_r4_e8.py run \
            --bank $DATA/bank_d91_search.npz \
            --ckpt-l $DATA/inputs/$CK_L_BN --sha-l $SHA_L_DEF --size-l L \
            --ckpt-m $DATA/inputs/$CK_M_BN --sha-m $SHA_M_DEF --size-m M \
            $CK_M20_ARG \
            --out $DATA/cols_icl_D91 --device cpu --threads 1 \
            --queue --worker-id \"\$w\" --ctx-s 3600 > $DATA/logs/icl_q\$w.log 2>&1 &
        done
        for j in \$(jobs -p); do wait \"\$j\" || rc=\$?; done
        [ \"\$rc\" = 0 ]
      fi
      echo \"\$(date -u +%FT%TZ) ALL H448 COLS DONE\"
    '" | tee -a "$EV/logs/cols.log"

    REL="$(date -u +%FT%TZ) agy_apex_w10: H448 xbabe0 columns complete; releasing xbabe0 cpus 16-63,80-127."
    printf '%s\n' "$REL" | (cd "$REPO" && flock AGENT_CHAT.md.lock tee -a AGENT_CHAT.md >/dev/null)
    printf '%s\n' "$REL" >> "$EV/logs/announce.txt"; echo "$REL" ;;

  pull)
    P=$EV/pulled
    [ ! -e "$P" ] || die "$P exists (create-once)"
    mkdir -p "$P"
    rs0 "xbabe0:$DATA/cids91.json" "xbabe0:$DATA/SCREEN_D91_SEARCH.json" \
        "xbabe0:$DATA/bank_d91_search.npz" "xbabe0:$DATA/bank_d91_search.npz.json" \
        "xbabe0:$DATA/stage_v/LOCK.json" \
        "xbabe0:$DATA/cols_matern_D91" "xbabe0:$DATA/cols_k3_D91" \
        "xbabe0:$DATA/cols_f4_D91" "xbabe0:$DATA/cols_icl_D91" "$P/"
    (cd "$P" && find . -type f ! -name 'LOCK.json' | sort | xargs sha256sum) > "$P/pulled.sha256"
    x0 "cd $DATA && sha256sum --quiet -c" < "$P/pulled.sha256"
    cmp -s "$P/LOCK.json" "$EV/stage_v/LOCK.json" || die "LOCK.json differs from prelock copy"
    chmod -R a-w "$P"
    echo "pulled $(find "$P" -type f | wc -l) files to $P (sha-verified with xbabe0)" | tee -a "$EV/logs/pull.log" ;;

  analyze)
    P=$EV/pulled
    [ -d "$P" ] || die "pull first"
    (cd "$P" && sha256sum --quiet -c "$P/pulled.sha256")
    for d in D1 D2; do (cd "$P418/$d" && sha256sum --quiet -c "$P418/$d.sha256"); done
    H445_ARG=()
    [ -s "$H445_RO" ] && H445_ARG=(--h445-readout "$H445_RO")
    rc=0
    nice -n 19 python3 -B "$STAGE/code/analyze_h448.py" \
      --pulled "$P" --h418-pulled "$P418" --board "$EV/board/dev91-audit-v1" \
      --board-manifest "$EV/inputs/board_audit.sha256" --lock "$EV/stage_v/LOCK.json" \
      --receipt "$RECEIPT" --frozen "$STAGE/code/FROZEN_FAIR_CHAMPION.json" \
      --design "$STAGE/code/DESIGN.json" --a2-cells "$STAGE/code/a2_cells.json" \
      "${H445_ARG[@]}" \
      --out "$EV/results/H448_DEV91.json" --rows "$EV/results/h448.APEX_FAIR_CHAMPION.jsonl" \
      > "$EV/results/H448_DEV91.txt" 2>&1 || rc=$?
    chmod a-w "$EV/results/H448_DEV91.txt"
    for f in H448_DEV91.json h448.APEX_FAIR_CHAMPION.jsonl; do
      [ -e "$EV/results/$f" ] && chmod a-w "$EV/results/$f"
    done
    cat "$EV/results/H448_DEV91.txt"; echo "analyze exit $rc"; exit "$rc" ;;

  board)
    ROWS=$EV/results/h448.APEX_FAIR_CHAMPION.jsonl
    [ -s "$ROWS" ] || die "no rows (analyze first)"
    PC=$(jq -r .prereg_commit "$EV/inputs/spec.json")
    BN=$EV/board-new/dev91-audit-v1
    [ ! -e "$BN" ] || die "$BN exists (create-once)"
    [ ! -e "$LOCALBOARD/scores/h448.APEX_FAIR_CHAMPION.jsonl" ] || die "$LOCALBOARD already holds h448.APEX_FAIR_CHAMPION"
    mkdir -p "$EV/tools" "$BN"
    git -C "$REPO" show "$PC:$TOOLS/ladder.py" > "$EV/tools/ladder.py"
    cp -r "$EV/board/dev91-audit-v1/." "$BN/" && chmod -R u+w "$BN"
    cp "$ROWS" "$BN/scores/h448.APEX_FAIR_CHAMPION.jsonl"
    LAD=(env REFBOARD_H335_DIR="$H335CODE" nice -n 19 python3 -B "$EV/tools/ladder.py")
    "${LAD[@]}" "$BN" --write > "$EV/results/LADDER_dev91_h448.md"
    "${LAD[@]}" "$BN" --focus h448.APEX_FAIR_CHAMPION > "$EV/results/LADDER_focus_H448.md"
    jq -e --slurpfile a "$EV/results/H448_DEV91.json" \
      '(.elo.elo["h448.APEX_FAIR_CHAMPION"].elo - $a[0].readout.canonical_board_91.entrant_elo.elo) | fabs < 1e-9' \
      "$BN/ladder.json" >/dev/null || die "ladder.py Elo differs from analyze_h448"
    ( set -o noclobber; cat "$LOCALBOARD/ladder.json" > "$LOCALBOARD/ladder.pre-h448.json"
      cat "$LOCALBOARD/LADDER.md" > "$LOCALBOARD/LADDER.pre-h448.md" )
    install -m 0444 "$ROWS" "$LOCALBOARD/scores/h448.APEX_FAIR_CHAMPION.jsonl"
    "${LAD[@]}" "$LOCALBOARD" --write > /dev/null
    cmp -s <(jq -S .elo "$LOCALBOARD/ladder.json") <(jq -S .elo "$BN/ladder.json") \
      || echo "note: local board root ladder differs from export"
    head -n 45 "$EV/results/LADDER_dev91_h448.md" ;;

  *) die "unknown step $STEP" ;;
esac
