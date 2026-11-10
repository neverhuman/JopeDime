#!/bin/bash
# H418 AMENDMENT_2 driver, run on xbabe2 (AMENDMENT_2.md §5). No xbabe0 step and no new dev91 compute: the analysis reads
# the pulled, sha-checked records ($EV/pulled) and the LOCKed board export only.
#   run_h418_a2.sh stage          create-once /home/ubuntu/h418-stage/a2: the pub-p1 code (checked against its
#                                 CODE.sha256) + the A2 files at their origin/v1/next bytes
#   run_h418_a2.sh selftest DIR   [no dev91] selftest_h418_a2.py on synthetic fixtures in a new DIR + tools/ladder.py
#   run_h418_a2.sh analyze        THE one amended analysis: noclobber receipt ns/receipts/H418_A2_ANALYZE.txt first;
#                                 results/H418_DEV91_A2.{json,txt} + results/h416.SXB.jsonl (create-once); the
#                                 INCOMPLETE output results/H418_DEV91.{json,txt} is sha-checked before and after
#   run_h418_a2.sh board          run_h418.sh's board step, read against results/H418_DEV91_A2.json
set -euo pipefail
STEP=${1:?step}; shift
REPO=/home/ubuntu/JopeDime
WDIR=research/flywheel_selfplay/h418_sxb_dev91
TOOLS=research/elo/reference-board/tools
EV=/home/ubuntu/jope_plan/evidence/H418
P1=/home/ubuntu/h418-stage/pub-p1
STAGE=/home/ubuntu/h418-stage/a2
H335CODE=/home/ubuntu/h335-stage/code
LOCALBOARD=/home/ubuntu/jope-data/h335-stagev/refboard/dev91-audit-v1   # xbabe2 board root
RECEIPT=/home/ubuntu/jope_plan/ns/receipts/H418_A2_ANALYZE.txt
A2_FILES=(AMENDMENT_2.md a2_cells.json analyze_h418_a2.py selftest_h418_a2.py run_h418_a2.sh)
INC=("results/H418_DEV91.json ee9652bf293672ecc7c28bec460dea417e5477b72c734de359fb701ad572ce6a"
     "results/H418_DEV91.txt bc58581d578006e8e5cd2cd08641b43b1e0240bcc53b91643198d97bfc6d7f59")
OUT=$EV/results/H418_DEV91_A2.json
TXT=$EV/results/H418_DEV91_A2.txt
ROWS=$EV/results/h416.SXB.jsonl
CRED=(-c credential.helper= -c "credential.helper=!/home/ubuntu/.local/bin/gh-role writer auth git-credential")
URL=https://github.com/neverhuman/JopeDime.git
die() { echo "run_h418_a2 refused: $*" >&2; exit 3; }
[ "$(hostname)" = xbabe2 ] || die "run on xbabe2"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
fetch() { git -C "$REPO" "${CRED[@]}" fetch -q "$URL" +refs/heads/v1/next:refs/remotes/origin/v1/next; }
inc_ok() {  # the INCOMPLETE output is unchanged
  local x f h
  for x in "${INC[@]}"; do
    f=${x% *}; h=${x#* }
    [ "$(sha256sum "$EV/$f" | cut -d' ' -f1)" = "$h" ] || return 1
  done
}
stage_ok() {  # the stage is intact and its A2 files are the published bytes
  [ -d "$STAGE/code" ] || die "no stage $STAGE (run: stage)"
  (cd "$STAGE/code" && sha256sum --quiet -c ../CODE.sha256) || die "staged code differs from its CODE.sha256"
  (cd "$STAGE/code" && sha256sum --quiet -c ../CODE.p1.sha256) || die "staged pub-p1 files differ from pub-p1"
  fetch
  local f
  for f in "${A2_FILES[@]}"; do
    cmp -s "$STAGE/code/$f" <(git -C "$REPO" show "origin/v1/next:$WDIR/$f") || die "$f differs from origin/v1/next"
  done
}

case $STEP in
  stage)
    [ ! -e "$STAGE" ] || die "$STAGE exists (create-once)"
    (cd "$P1/code" && sha256sum --quiet -c ../CODE.sha256) || die "pub-p1 code differs from its CODE.sha256"
    fetch
    mkdir -p "$STAGE/code"
    cp -p "$P1"/code/* "$STAGE/code/"
    for f in "${A2_FILES[@]}"; do
      [ ! -e "$STAGE/code/$f" ] || die "$f is also a pub-p1 file"
      git -C "$REPO" show "origin/v1/next:$WDIR/$f" > "$STAGE/code/$f" || die "$f is not published on origin/v1/next"
      chmod 0555 "$STAGE/code/$f"
    done
    cp "$P1/CODE.sha256" "$STAGE/CODE.p1.sha256"
    git -C "$REPO" rev-parse origin/v1/next > "$STAGE/source_commit"
    (cd "$STAGE/code" && find . -maxdepth 1 -type f | sort | sed 's#^\./##' | xargs sha256sum) > "$STAGE/CODE.sha256"
    chmod a-w "$STAGE/CODE.sha256" "$STAGE/CODE.p1.sha256" "$STAGE/source_commit" "$STAGE/code"
    echo "staged $(wc -l < "$STAGE/CODE.sha256") files to $STAGE from $(cut -c1-10 "$STAGE/source_commit");" \
      "CODE.sha256 $(sha256sum "$STAGE/CODE.sha256" | cut -c1-16)" | tee -a "$EV/logs/stage_a2.log" ;;
  selftest)
    DIR=${1:?new scratch dir}
    [ ! -e "$DIR" ] || die "$DIR exists"
    stage_ok
    mkdir -p "$DIR"
    git -C "$REPO" show "origin/v1/next:$TOOLS/ladder.py" > "$DIR/ladder.py"
    REFBOARD_H335_DIR="$H335CODE" nice -n 19 python3 -B "$STAGE/code/selftest_h418_a2.py" --dir "$DIR/cases" \
      --ladder "$DIR/ladder.py" | tee "$DIR/selftest.txt"
    printf '%s %s %s\n' "$(sha256sum "$STAGE/CODE.sha256" | cut -d' ' -f1)" "$(date -u +%FT%TZ)" "$DIR" \
      >> "$EV/logs/selftest_a2_pass.txt" ;;
  analyze)
    stage_ok
    CS=$(sha256sum "$STAGE/CODE.sha256" | cut -d' ' -f1)
    grep -q "^$CS " "$EV/logs/selftest_a2_pass.txt" 2>/dev/null || die "no selftest pass for this stage"
    inc_ok || die "the INCOMPLETE output differs from AMENDMENT_2 §2"
    P=$EV/pulled
    for d in D1 D2 GE GL GS; do (cd "$P/$d" && sha256sum --quiet -c "$P/$d.sha256") || die "pulled $d changed"; done
    for f in "$OUT" "$TXT" "$ROWS"; do [ ! -e "$f" ] || die "$f exists (create-once)"; done
    ( set -o noclobber
      printf '%s H418 AMENDMENT_2 analyze: stage %s code %s source_commit %s\n' "$(date -u +%FT%TZ)" "$STAGE" "$CS" \
        "$(cat "$STAGE/source_commit")" > "$RECEIPT" ) 2>/dev/null || die "the amended analysis already ran ($RECEIPT)"
    chmod a-w "$RECEIPT"
    rc=0
    nice -n 19 python3 -B "$STAGE/code/analyze_h418_a2.py" --pulled "$P" --board "$EV/board/dev91-audit-v1" \
      --board-manifest "$EV/inputs/board_audit.sha256" --lock "$EV/stage_v/LOCK.json" \
      --design "$STAGE/code/DESIGN.json" --conflicts "$EV/results/memo_conflicts.jsonl" \
      --a2-cells "$STAGE/code/a2_cells.json" --out "$OUT" --rows "$ROWS" > "$TXT" 2>&1 || rc=$?
    chmod a-w "$TXT"
    for f in "$OUT" "$ROWS"; do [ -e "$f" ] && chmod a-w "$f"; done
    echo "$(date -u +%FT%TZ) analyze-a2 exit $rc code $CS" >> "$EV/logs/analyze_a2.log"
    inc_ok || { echo "the INCOMPLETE output changed during analyze" >&2; rc=9; }
    cat "$TXT"; echo "analyze exit $rc"; exit "$rc" ;;
  board)
    stage_ok
    [ -s "$ROWS" ] || die "no rows (analyze first)"
    [ "$(jq -r .status "$OUT")" = SCORED ] || die "$OUT is not SCORED"
    [ "$(sha256sum "$ROWS" | cut -d' ' -f1)" = "$(jq -r .rows.sha256 "$OUT")" ] || die "rows differ from $OUT"
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
    jq -e --slurpfile a "$OUT" \
      '(.elo.elo["h416.SXB"].elo - $a[0].readout.board_elo["h416.SXB"].elo) | fabs < 1e-9' "$BN/ladder.json" >/dev/null \
      || die "ladder.py's h416.SXB Elo differs from analyze_h418_a2's"
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
