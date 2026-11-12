#!/bin/bash
# H451 readout (jq only). Metric definitions follow CW-4 readout_cw4.metrics without the monitor metric M:
#   K   mean of the family means of cap over B3, B4, B5 (all noise levels)
#   B9  B10  family means of cap
#   L   mean cap over B1 tasks with noise 0
#   P   mean u8 over noise-free tasks, B12 excluded
#   S6  mean of (L, B3, B4, B5, B9, B10)
# cap is null when the reference skill is below 0.05; nulls are dropped from every mean.
#
#   summarize_h451.sh metrics DIR READOUT SUBSET       READOUT full|ridgeres, SUBSET all|rep0  -> one JSON line
#   summarize_h451.sh noise SUBSET READOUT             |M_h020 - M_h024| per metric (stored H431 runs)
#   summarize_h451.sh verdict INIT_DIR FINAL_DIR SIZE  the pre-registered comparison, both readouts -> JSON
set -euo pipefail
STORED=/home/ubuntu/jope-data/h335-cl/h431/pub-r2
METRICS=(K B9 B10 L P S6)
JQ_LIB='
def mean: map(select(. != null)) | if length == 0 then null else add / length end;
def fam($r; $f): map(select(.family == $f) | .cap[$r]) | mean;
def metrics($r):
  . as $t
  | ($t | map(select(.family == "B1" and .noise == 0) | .cap[$r]) | mean) as $L
  | ($t | fam($r; "B3")) as $b3 | ($t | fam($r; "B4")) as $b4 | ($t | fam($r; "B5")) as $b5
  | ($t | fam($r; "B9")) as $b9 | ($t | fam($r; "B10")) as $b10
  | {K: ([$b3, $b4, $b5] | mean), B9: $b9, B10: $b10, L: $L,
     P: ($t | map(select(.noise == 0 and .family != "B12") | .u8[$r]) | mean),
     S6: ([$L, $b3, $b4, $b5, $b9, $b10] | mean),
     n_tasks: ($t | length), not_ok: ($t | map(select(.ok | not)) | length)};
'

tasks() {  # DIR SUBSET -> JSON array of task records
  local d=$1 s=$2
  find "$d" -maxdepth 1 -name 'task_[0-9][0-9][0-9][0-9].json' -print0 | sort -z | xargs -0 cat \
    | jq -s --arg s "$s" 'if $s == "rep0" then map(select(.rep == 0)) else . end'
}

metrics() {  # DIR READOUT SUBSET
  tasks "$1" "$3" | jq -c --arg r "$2" "$JQ_LIB metrics(\$r)"
}

noise() {  # SUBSET READOUT
  jq -cn --argjson a "$(metrics "$STORED/bat_M_h020" "$2" "$1")" --argjson b "$(metrics "$STORED/bat_M_h024" "$2" "$1")" \
    '[("K","B9","B10","L","P","S6") as $k | {($k): (($a[$k] - $b[$k]) | fabs)}] | add'
}

case ${1:?metrics|noise|verdict} in
metrics) metrics "${2:?DIR}" "${3:?full|ridgeres}" "${4:?all|rep0}" ;;
noise) noise "${2:?all|rep0}" "${3:?full|ridgeres}" ;;
verdict)
  INIT=${2:?INIT_DIR}; FIN=${3:?FINAL_DIR}; SIZE=${4:?M|L}
  case $SIZE in M) SUB=all; WANT=340 ;; L) SUB=rep0; WANT=68 ;; *) echo "SIZE must be M or L" >&2; exit 2 ;; esac
  out='{}'
  for r in ridgeres full; do
    mi=$(metrics "$INIT" "$r" "$SUB"); mf=$(metrics "$FIN" "$r" "$SUB"); nz=$(noise "$SUB" "$r")
    one=$(jq -cn --argjson i "$mi" --argjson f "$mf" --argjson n "$nz" --argjson want "$WANT" '
      def thr($k): ([0.05, 2 * $n[$k]] | max);
      ([("K","B9","B10","L","P","S6") as $k | {($k): ($f[$k] - $i[$k])}] | add) as $d
      | ($i.n_tasks == $want and $f.n_tasks == $want and $i.not_ok == 0 and $f.not_ok == 0) as $complete
      | ([("K","L","S6") as $k | select($d[$k] < -thr($k)) | $k]) as $lost
      | {init: $i, final: $f, noise: $n, delta: $d,
         thr: ([("K","L","S6") as $k | {($k): thr($k)}] | add),
         complete: $complete, lost_on: $lost,
         verdict: (if ($complete | not) then "INCOMPLETE"
                   elif ($lost | length) > 0 then "CAPABILITY_LOST"
                   elif $d.S6 >= thr("S6") then "CAPABILITY_KEPT_S6_GAIN"
                   else "CAPABILITY_KEPT" end)}')
    out=$(jq -cn --argjson o "$out" --argjson x "$one" --arg r "$r" '$o + {($r): $x}')
  done
  jq -n --argjson o "$out" --arg size "$SIZE" --arg sub "$SUB" --arg init "$INIT" --arg fin "$FIN" \
    '{schema: "jope.h451.verdict/v1", size: $size, subset: $sub, init_dir: $init, final_dir: $fin,
      primary_readout: "ridgeres", verdict: $o.ridgeres.verdict, readouts: $o}'
  ;;
*) echo "unknown command $1" >&2; exit 2 ;;
esac
