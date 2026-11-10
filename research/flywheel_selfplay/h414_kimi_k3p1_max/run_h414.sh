#!/bin/bash
# H414 steps (xbabe2). Usage: run_h414.sh STEP   with H414_CODE=~/h414-stage/code-TAG for every step after `stage`.
#   stage TAG     frozen read-only copy of this dir's H414 files, taken from origin/v1/next (outside the evidence tree:
#                 the H317 guard denies that tree before the unlock)
#   lock          LOCK.json (code: staged files, both binaries + their rlibs + every entrant source, h335_core + H317
#                 pins; inputs: dev91 + P201b BOARD.json and the two TRAIN parity record files; twofold: the two-fold
#                 BOARD.json + build tree, checked here in bash) + create-once receipt
#   unlock        one-shot Stage V unlock (refuses unless the staged files equal origin/v1/next)
#   score | rows  dev91-audit-v1, guarded python driver d91_h414.py
#   twofold       dev91-s1-two-fold-v1, ONE run of twofold_eval_h414 (bash: the H317 guard refuses ~/jope-data/h257/ at
#                 every stage). Requires the receipt == sha256(LOCK.json), the unlock marker naming that sha, every
#                 LOCK code/twofold hash and every envelope sha against the two-fold BOARD.json. Create-once STARTED.
#   twofold_rows  create-once two-fold board rows; a context with no record is failed, u = -2.0 (board rule 2)
#   p201b_rows    create-once P201b rows: h414.kimik3p1_max (pre-LOCK TRAIN run, same binary) and the optional
#                 h412.grok4p7_xhigh (H412 pre-LOCK TRAIN records, imported). TRAIN only; no unlock needed.
set -euo pipefail
STEP=${1:?step}
REPO=/home/ubuntu/JopeDime
DATA=${H414_DATA:-/home/ubuntu/jope-data/h414/d91}
WDIR=research/flywheel_selfplay/h414_kimi_k3p1_max
FILES=(PREREG.md d91_h414.py frontier_audit_h414.py frontier_audit_h414.rs build_h414.sh twofold_eval_h414.rs
       build_twofold_h414.sh run_h414.sh)
RECEIPT=/home/ubuntu/jope_plan/ns/receipts/H414_LOCK_sha256.txt
BOARD_ROOT=/home/ubuntu/jope-data/h335-stagev/refboard/dev91-audit-v1
P201B_ROOT=/home/ubuntu/jope-data/h335-cl/refboard/p201b-train-audit-v1
TF_ROOT=/home/ubuntu/jope-data/h257/refboard/dev91-s1-two-fold-v1
TF_ENV=/home/ubuntu/jope-data/h257/cache/val91_jtt1_envelopes
BIN=/home/ubuntu/jope-data/h414/bin/frontier_audit_h414
TF_BIN=/home/ubuntu/jope-data/h414/bin/twofold_eval_h414
BT=/home/ubuntu/jope-data/h414/build/frontier
TT=/home/ubuntu/jope-data/h414/build/twofold
KIMI_T=/home/ubuntu/jope-data/h414/parity/train_T_h414.jsonl
GROK_T=/home/ubuntu/jope-data/h412/parity/train_T_h412.jsonl
H335C=/home/ubuntu/h335-stage/code
H317=/home/ubuntu/jope_plan/evidence/H317/source/research/flywheel_selfplay/h317_island10_multiscale_curriculum
KIMI_SHA=0c211694ec66605b994595fd8f4cf2b8894df6f156caa3b80f1fcc75f5d6ceb3
KIMI_ORIGIN="owner file ~/JopeDime/kimik3p1_max.rs (xbabe2, mtime 2026-11-07T16:43:30Z, 47156 B)"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
shamap() { jq -R -n '[inputs | split("  ") | {(.[1]): .[0]}] | add'; }
if [ "$STEP" = stage ]; then
  TAG=${2:?tag}; OUT=/home/ubuntu/h414-stage/code-$TAG
  [ -e "$OUT" ] && { echo "$OUT exists" >&2; exit 2; }
  mkdir -p "$OUT"
  for f in "${FILES[@]}"; do git -C "$REPO" show "origin/v1/next:$WDIR/$f" > "$OUT/$f"; done
  chmod 0555 "$OUT"/*.sh; chmod 0444 "$OUT"/*.py "$OUT"/*.md "$OUT"/*.rs
  (cd "$OUT" && sha256sum -- "${FILES[@]}" > CODE.sha256 && chmod 0444 CODE.sha256)
  chmod 0555 "$OUT"; echo "staged $OUT $(sha256sum "$OUT/CODE.sha256" | cut -d' ' -f1)"; exit 0
fi
CODE=${H414_CODE:?H414_CODE}
cd "$CODE" && sha256sum --quiet -c CODE.sha256
for f in "${FILES[@]}"; do
  cmp -s "$CODE/$f" <(git -C "$REPO" show "origin/v1/next:$WDIR/$f") \
    || { echo "$f is not equal to origin/v1/next: refusing (publish the PREREG first)" >&2; exit 3; }
done
cmp -s "$CODE/frontier_audit_h414.rs" "$BT/harness/frontier_audit_h414.rs" || { echo "harness source != build tree" >&2; exit 3; }
cmp -s "$CODE/frontier_audit_h414.py" "$BT/harness/frontier_audit_h414.py" || { echo "wrapper != build tree" >&2; exit 3; }
cmp -s "$CODE/build_h414.sh" "$BT/build.sh" || { echo "build_h414.sh != build tree" >&2; exit 3; }
cmp -s "$CODE/twofold_eval_h414.rs" "$TT/harness/eval_frontier_dl_val91_h414.rs" || { echo "two-fold source != build tree" >&2; exit 3; }
cmp -s "$CODE/build_twofold_h414.sh" "$TT/build.sh" || { echo "build_twofold_h414.sh != build tree" >&2; exit 3; }
LK=$DATA/stage_v/LOCK.json
# bash equivalent of resume_stage_v for the steps the python guard cannot host
lock_ok() {
  local got posted named
  got=$(sha256sum "$LK" | cut -d' ' -f1); posted=$(cut -d' ' -f1 "$RECEIPT")
  [ "$got" = "$posted" ] || { echo "LOCK.json sha $got != receipt $posted" >&2; exit 5; }
  jq -r '.code_sha256 + .twofold_sha256 | to_entries[] | "\(.value)  \(.key)"' "$LK" | sha256sum --quiet -c \
    || { echo "a LOCK-pinned file changed" >&2; exit 5; }
  if [ "${1:-}" = unlocked ]; then
    named=$(cut -d' ' -f2 "$DATA/stage_v/STAGE_V_STARTED" 2>/dev/null || true)
    [ "$named" = "$got" ] || { echo "no unlock marker naming LOCK $got" >&2; exit 5; }
  fi
}
PY=(nice -n 19 python3 -B "$CODE/d91_h414.py")
TFD=$DATA/twofold
TF_OUT=$TFD/kimi_twofold.jsonl
TF_ROWS=$TFD/h414.kimik3p1_max.twofold.jsonl
P_ROWS=$DATA/p201b
case $STEP in
  lock)
    [ -e "$LK" ] && { echo "LOCK.json exists" >&2; exit 2; }
    [ "$(sha256sum "$TF_BIN" | cut -d' ' -f1)" = 97bf86530bef4c61c837883721d570e78cdd7aabecb6713b740cd02cad6cd871 ] \
      || { echo "two-fold binary sha drift" >&2; exit 3; }
    [ "$(sha256sum "$BIN" | cut -d' ' -f1)" = c34025df88cc3ec681ac186e5a0ad911b0caaa84a41f0ee68c2f31476a627162 ] \
      || { echo "frontier binary sha drift" >&2; exit 3; }
    mkdir -p "$DATA/stage_v"
    { for f in "${FILES[@]}"; do sha256sum "$CODE/$f"; done
      sha256sum "$BIN" "$TF_BIN" "$BT"/lib/*.rlib "$TT"/lib/*.rlib
      find "$BT/src" "$TT/src" -type f -name '*.rs' | sort | xargs sha256sum
      sha256sum "$H335C/h335_core.py" "$H317/common_h317.py"; } | shamap > "$DATA/stage_v/code.json"
    # dev91-path inputs are verified after the unlock (score / rows), never by unlock_stage_v (the guard refuses them)
    sha256sum "$BOARD_ROOT/BOARD.json" "$P201B_ROOT/BOARD.json" "$KIMI_T" "$GROK_T" | shamap > "$DATA/stage_v/inputs.json"
    # two-fold: bash-checked only (the guard refuses ~/jope-data/h257/ and dl_val names at every stage)
    sha256sum "$TF_ROOT/BOARD.json" "$TT/harness/eval_frontier_dl_val91_h414.rs" "$TT/build.sh" \
      | shamap > "$DATA/stage_v/twofold.json"
    jq -n --slurpfile c "$DATA/stage_v/code.json" --slurpfile i "$DATA/stage_v/inputs.json" \
      --slurpfile t "$DATA/stage_v/twofold.json" --arg utc "$(date -u +%FT%TZ)" --arg bin "$BIN" \
      --arg bs "$(sha256sum "$BIN" | cut -d' ' -f1)" --arg tb "$TF_BIN" --arg ts "$(sha256sum "$TF_BIN" | cut -d' ' -f1)" \
      --arg ks "$KIMI_SHA" --arg ko "$KIMI_ORIGIN" \
      '{wave:"H414", entrant:"kimik3p1_max", model_id:"h414.kimik3p1_max", entrant_source_sha256:$ks,
        entrant_origin:$ko,
        boards:{"dev91-audit-v1":"40cb4828cf5c9a0d2d9cc0f32b7900819c992ec9a58c5d18f83ac84b29fa4069",
                "dev91-s1-two-fold-v1":"b8836033c0540e1c7d6a488cb33248372cab02e4a16776cd57b08ede269ea190",
                "p201b-train-audit-v1":"ad20a65e309e5fc670b7f646f9aa6f2ad725064368b36cd6bbdf202ba6355c19"},
        seed:42, cpu_limit_s:300, wall_limit_s:900, binary:$bin, binary_sha256:$bs,
        twofold_binary:$tb, twofold_binary_sha256:$ts, twofold_threads:8, utc:$utc,
        code_sha256:$c[0], inputs_sha256:$i[0], twofold_sha256:$t[0]}' > "$LK"
    chmod 0444 "$LK"
    S=$(sha256sum "$LK" | cut -d' ' -f1)
    ( set -o noclobber; echo "$S  $LK" > "$RECEIPT" ) && chmod 0444 "$RECEIPT"
    echo "LOCK $S" ;;
  unlock|score|rows) "${PY[@]}" "$STEP" ;;
  twofold)
    lock_ok unlocked
    mkdir -p "$TFD"
    ( set -o noclobber; echo "$(date -u +%FT%TZ) $(sha256sum "$LK" | cut -d' ' -f1)" > "$TFD/STARTED" ) \
      || { echo "two-fold run already started once ($TFD/STARTED)" >&2; exit 2; }
    jq -r '.contexts[].cid' "$TF_ROOT/BOARD.json" > "$TFD/cids.txt"
    [ "$(wc -l < "$TFD/cids.txt")" = 91 ] || { echo "two-fold board is not 91 contexts" >&2; exit 3; }
    jq -r '.contexts[] | "\(.envelope_sha256)  \(.cid).jtt1"' "$TF_ROOT/BOARD.json" > "$TFD/envelopes.sha256"
    (cd "$TF_ENV" && sha256sum --quiet -c "$TFD/envelopes.sha256") || { echo "envelope sha != two-fold BOARD.json" >&2; exit 3; }
    t0=$(date -u +%FT%TZ)
    set +e
    nice -n 19 timeout 7200 "$TF_BIN" "$TF_ENV" "$TFD/cids.txt" "$TF_OUT" 8 kimik3p1_max 2> "$TFD/run.err"
    rc=$?
    set -e
    jq -n --arg t0 "$t0" --arg t1 "$(date -u +%FT%TZ)" --argjson rc "$rc" --arg host "$(uname -n)" \
      --arg lk "$(sha256sum "$LK" | cut -d' ' -f1)" '{start:$t0, end:$t1, exit_code:$rc, host:$host, lock_sha256:$lk}' \
      > "$TFD/run_status.json"
    echo "two-fold exit $rc: $(jq -r .status "$TF_OUT" 2>/dev/null | sort | uniq -c | tr -s ' ' | tr '\n' ' ')" ;;
  twofold_rows)
    lock_ok unlocked
    [ -e "$TF_ROWS" ] && { echo "$TF_ROWS exists (create-once)" >&2; exit 2; }
    [ -s "$TFD/run_status.json" ] || { echo "two-fold run not finished" >&2; exit 2; }
    BSHA=$(jq -r .board_sha256 "$TF_ROOT/BOARD.json")
    [ "$BSHA" = b8836033c0540e1c7d6a488cb33248372cab02e4a16776cd57b08ede269ea190 ] || { echo "two-fold board sha" >&2; exit 3; }
    # a truncated last line (a run stopped by the timeout) is not a record: its context counts as failed
    touch "$TF_OUT"; jq -cR 'fromjson? | select(type == "object")' "$TF_OUT" > "$TFD/parsed.jsonl"
    n=$(jq -r .cid "$TFD/parsed.jsonl" | wc -l); u=$(jq -r .cid "$TFD/parsed.jsonl" | sort -u | wc -l)
    [ "$n" = "$u" ] || { echo "duplicate two-fold records" >&2; exit 4; }
    [ "$n" = 0 ] || [ "$(jq -r .entrant "$TFD/parsed.jsonl" | sort -u)" = kimik3p1_max ] \
      || { echo "unexpected entrant in two-fold output" >&2; exit 4; }
    jq -c '[.contexts[] | {key:.cid, value:.envelope_sha256}] | from_entries' "$TF_ROOT/BOARD.json" > "$TFD/env.json"
    jq -c '{key:.cid, value:(del(.fold0_preds, .fold1_preds))}' "$TFD/parsed.jsonl" | jq -cs from_entries > "$TFD/recs.json"
    jq -e --slurpfile e "$TFD/env.json" 'keys - ($e[0] | keys) | length == 0' "$TFD/recs.json" >/dev/null \
      || { echo "a two-fold record names a context that is not on the board" >&2; exit 4; }
    jq -cn --slurpfile e "$TFD/env.json" --slurpfile r "$TFD/recs.json" --arg bs "$BSHA" \
      --arg lk "$(sha256sum "$LK" | cut -d' ' -f1)" --arg raw "$(sha256sum "$TF_OUT" | cut -d' ' -f1)" \
      --arg tb "$(sha256sum "$TF_BIN" | cut -d' ' -f1)" --arg ks "$KIMI_SHA" --arg ko "$KIMI_ORIGIN" \
      --arg utc "$(jq -r .end "$TFD/run_status.json")" '
      $e[0] | to_entries | sort_by(.key)[] | .key as $c | .value as $s | ($r[0][$c]) as $x
      | {board:"dev91-s1-two-fold-v1", board_sha256:$bs, cid:$c, envelope_sha256:$s, model_id:"h414.kimik3p1_max",
         status:(if $x == null then "failed" else $x.status end),
         utility:(if $x == null then -2.0 else $x.s1_utility_zero_comp end),
         utility_definition:"s1_utility_zero_comp (jope.s1-two-fold/v1, H257 S1 gated; -2.0 on gate fail)",
         produced_by:{wave:"H414", producer:"twofold_eval_h414 (H257 eval_frontier_dl_val91.rs + kimik3p1_max)",
                      binary_sha256:$tb, entrant_source_sha256:$ks, entrant_origin:$ko, threads:8,
                      lock_sha256:$lk, raw_jsonl_sha256:$raw, produced_utc:$utc}}
      + (if $x == null then {note:"no record from the single run: failed (board rule 2)"} else {wall_s:($x.wall_ms/1000), h414_record:$x} end)' \
      > "$TFD/rows.tmp"
    [ "$(wc -l < "$TFD/rows.tmp")" = 91 ] || { echo "expected 91 rows" >&2; exit 4; }
    ( set -o noclobber; cat "$TFD/rows.tmp" > "$TF_ROWS" ) && chmod 0444 "$TF_ROWS"
    echo "$TF_ROWS $(sha256sum "$TF_ROWS" | cut -d' ' -f1) $(jq -r .status "$TF_ROWS" | sort | uniq -c | tr -s ' ' | tr '\n' ' ')" ;;
  p201b_rows)
    lock_ok
    mkdir -p "$P_ROWS"
    PB=$(jq -r .board_sha256 "$P201B_ROOT/BOARD.json")
    [ "$PB" = ad20a65e309e5fc670b7f646f9aa6f2ad725064368b36cd6bbdf202ba6355c19 ] || { echo "P201b board sha" >&2; exit 3; }
    jq -c '[.contexts[] | {key:.cid, value:.envelope_sha256}] | from_entries' "$P201B_ROOT/BOARD.json" > "$P_ROWS/env.json"
    for spec in "kimik3p1_max:h414.kimik3p1_max:$KIMI_T:H414:$KIMI_SHA" \
                "grok4p7_xhigh:h412.grok4p7_xhigh:$GROK_T:H412:8e648cef93186aca9d10868ed1ed2a62407950169082aa1f4fab87af379664f8"; do
      IFS=: read -r ent id src wave esha <<< "$spec"
      dst=$P_ROWS/$id.jsonl
      [ -e "$dst" ] && { echo "$dst exists (create-once)" >&2; exit 2; }
      jq -c --arg e "$ent" 'select(.entrant == $e)' "$src" > "$P_ROWS/$id.src.jsonl"
      [ "$(wc -l < "$P_ROWS/$id.src.jsonl")" = 201 ] || { echo "$id: expected 201 TRAIN records" >&2; exit 4; }
      [ "$(jq -r .cid "$P_ROWS/$id.src.jsonl" | sort -u | wc -l)" = 201 ] || { echo "$id: duplicate contexts" >&2; exit 4; }
      jq -c --slurpfile env "$P_ROWS/env.json" --arg pb "$PB" --arg id "$id" --arg w "$wave" --arg es "$esha" \
         --arg ss "$(sha256sum "$src" | cut -d' ' -f1)" --arg sp "$src" \
         --arg lk "$(sha256sum "$LK" | cut -d' ' -f1)" '
        if $env[0][.cid] == null or $env[0][.cid] != .envelope_sha256 then error("envelope sha != P201b BOARD.json: \(.cid)") else . end
        | {board:"p201b-train-audit-v1", board_sha256:$pb, cid, envelope_sha256, model_id:$id, status,
           utility:(.utility|tonumber), rmse_audit, base_rmse_audit, wall_s:.runner_wall_s, cpu_s, audit_preds_sha256,
           harness_version:(if $w == "H414" then "h414-frontier-audit-v1" else "h412-frontier-audit-v1" end),
           source:{wave:$w, entrant:.entrant, entrant_source_sha256:$es, seed:42, binary_sha256, wrapper_version,
                   phase:"pre-LOCK TRAIN run (deterministic: a second run is identical)",
                   train_records:$sp, train_records_sha256:$ss, boarded_by:"H414", h414_lock_sha256:$lk}}' \
        "$P_ROWS/$id.src.jsonl" | sort > "$P_ROWS/$id.tmp"
      [ "$(wc -l < "$P_ROWS/$id.tmp")" = 201 ] || { echo "$id: expected 201 rows" >&2; exit 4; }
      ( set -o noclobber; cat "$P_ROWS/$id.tmp" > "$dst" ) && chmod 0444 "$dst"
      echo "$dst $(sha256sum "$dst" | cut -d' ' -f1) $(jq -r .status "$dst" | sort | uniq -c | tr -s ' ' | tr '\n' ' ')"
    done ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
