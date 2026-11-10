#!/bin/bash
# Import the H257 two-fold DL-Val91 tournament (2026-11-04) as the stored board dev91-s1-two-fold-v1.
# Copies records only: nothing is fitted, predicted or re-scored.
#
# Source: H257 phase 2 (origin/main 571873a38c, research/flywheel_selfplay/h257_phase2_subsample_gp_curriculum/):
#   evaluate_tabular_foundation_dl_val91.py wrote  tabular_foundation_dl_val91_results.jsonl  (9 configs x 91)
#   evaluate_frontier_vs_h257_elo.py         wrote  frontier_dl_val91_results.jsonl            (4 LLM regressors x 91)
#   and the S3 single-split lookup             s3_precomputed_tabular_dl_val91_cache.json      (6 models x 91 S3 keys)
#   all under /home/ubuntu/jope-data/h257/cache on xbabe2. Envelopes: .../cache/val91_jtt1_envelopes/<cid>.jtt1.
#
# Usage: import_h257_two_fold.sh H257_CACHE_DIR DEV91_AUDIT_BOARD_JSON OUT_ROOT
#   e.g. import_h257_two_fold.sh /home/ubuntu/jope-data/h257/cache \
#          /home/ubuntu/jope-data/h335-stagev/refboard/dev91-audit-v1/BOARD.json \
#          /home/ubuntu/jope-data/h257/refboard/dev91-s1-two-fold-v1
# Writes OUT_ROOT/BOARD.json, OUT_ROOT/scores/h257.<entrant>.jsonl (predictions stripped; utility =
# s1_utility_zero_comp, the "S1 gated" utility of the H257 tournament) and OUT_ROOT/sources/ (verbatim copies of the
# source files + SHA256SUMS). Refuses to overwrite a different existing board. Prints counts and hashes, never data.
set -euo pipefail
SRC=$(readlink -f "$1"); AUD=$(readlink -f "$2"); OUT=$3
BOARD=dev91-s1-two-fold-v1
ENV=$SRC/val91_jtt1_envelopes
TF=$SRC/tabular_foundation_dl_val91_results.jsonl
FR=$SRC/frontier_dl_val91_results.jsonl
mkdir -p "$OUT/scores" "$OUT/sources"
OUT=$(readlink -f "$OUT")
W=$(mktemp -d "${TMPDIR:-/tmp}/import_h257.XXXXXX")  # scratch only; never deleted by this tool

# contexts: cid, family (from the AUDIT board: same 91 DL-Val91 contexts), envelope sha256 and header counts
jq -r '.contexts[] | "\(.cid) \(.family_id)"' "$AUD" | sort > "$W/fam.txt"
[ "$(ls "$ENV"/*.jtt1 | wc -l)" = 91 ] || { echo "expected 91 envelopes in $ENV" >&2; exit 2; }
while read -r cid fam; do
  f=$ENV/$cid.jtt1
  [ -f "$f" ] || { echo "missing envelope $cid" >&2; exit 2; }
  read -r magic nt nq width < <(od -A n -t u4 -N 16 "$f")
  [ "$magic" = 827610186 ] || { echo "not JTT1: $cid" >&2; exit 2; }
  n0=$(dd if="$f" bs=1 skip=17 count="$nt" status=none | od -A n -t u1 -v | tr -s ' ' '\n' | grep -cx 0 || true)
  jq -cn --arg c "$cid" --arg fam "$fam" --arg s "$(sha256sum < "$f" | cut -c1-64)" \
    --argjson nt "$nt" --argjson nq "$nq" --argjson w "$width" --argjson n0 "$n0" \
    '{cid:$c, family_id:$fam, envelope_sha256:$s, n_rows:$nt, n_query:$nq, width:$w, n_fold0:$n0, n_fold1:($nt-$n0)}'
done < "$W/fam.txt" > "$W/ctx.jsonl"
jq -s --arg b "$BOARD" '
  {schema:"jope.reference-board/v1", board:$b,
   description:("DL-Val91 (the same 91 dev contexts / 19 families as dev91-audit-v1) under the H257 two-fold split. "
     + "Stored scores of the H257 phase-2 tournament (2026-11-04): TabFM, TabFM ensemble, Kumo-Tabular L/S, TabPFN 3.5 "
     + "(n_est 8 and 1), TabICL v2, TabDPT 1.3, Mitra v2 and 4 frozen LLM regressors. Not comparable with dev91-audit-v1 "
     + "(different row split)."),
   split:("jope.s1-two-fold/v1 (H257): per context the same row count and family as dev91-audit-v1, split in half "
     + "(fold 0 / fold 1); fit on one fold, predict the other, both directions. ratio_k = RMSE_k / RMSE of the "
     + "training-fold mean; logratio = mean of the two log ratios; sd_logratio = half their absolute difference; "
     + "gate = exp(logratio) <= 0.90 and both folds beat the baseline; u = s1_utility_zero_comp = "
     + "-(0.9 * logratio + 0.2 * sd_logratio) if the gate passes, else -2.0 (also -2.0 on a fit failure). "
     + "Definition: evaluate_tabular_foundation_dl_val91.py docstring, origin/main 571873a38c."),
   envelope_root:"/home/ubuntu/jope-data/h257/cache/val91_jtt1_envelopes (xbabe2)",
   n_contexts:length, n_families:([.[].family_id]|unique|length), contexts:.,
   board_sha256:(map(.cid + .envelope_sha256) | join(""))}' "$W/ctx.jsonl" > "$W/BOARD.pre.json"
BSHA=$(jq -j '.board_sha256' "$W/BOARD.pre.json" | sha256sum | cut -c1-64)
jq --arg s "$BSHA" '.board_sha256=$s' "$W/BOARD.pre.json" > "$W/BOARD.json"
if [ -f "$OUT/BOARD.json" ] && ! cmp -s "$OUT/BOARD.json" "$W/BOARD.json"; then
  echo "REFUSING: $OUT/BOARD.json exists and differs" >&2; exit 3
fi
mv "$W/BOARD.json" "$OUT/BOARD.json"

# verbatim sources + their hashes
for f in tabular_foundation_dl_val91_results.jsonl tabular_foundation_dl_val91_cache.json \
         frontier_dl_val91_results.jsonl frontier_dl_val91_cache.json s3_precomputed_tabular_dl_val91_cache.json; do
  if [ -f "$OUT/sources/$f" ]; then cmp -s "$SRC/$f" "$OUT/sources/$f" || { echo "REFUSING: sources/$f differs" >&2; exit 3; }
  else cp -p "$SRC/$f" "$OUT/sources/$f"; fi
done
(cd "$OUT/sources" && sha256sum -- *.json *.jsonl) > "$OUT/sources/SHA256SUMS"

# score records: one file per entrant, sorted by cid, predictions stripped, bound to board + envelope sha256
jq -c '[.contexts[] | {key:.cid, value:.envelope_sha256}] | from_entries' "$OUT/BOARD.json" > "$W/env.json"
for pair in "$TF:evaluate_tabular_foundation_dl_val91.py" "$FR:evaluate_frontier_vs_h257_elo.py"; do
  file=${pair%%:*}; prog=${pair##*:}
  fsha=$(sha256sum < "$file" | cut -c1-64)
  for ent in $(jq -r .entrant "$file" | sort -u); do
    id=h257.$ent; dst=$OUT/scores/$id.jsonl
    jq -c --arg e "$ent" --arg b "$BOARD" --arg bs "$BSHA" --arg src "$(basename "$file")" --arg fs "$fsha" \
       --arg prog "$prog" --slurpfile env "$W/env.json" '
      select(.entrant == $e)
      | del(.fold0_preds, .fold1_preds, .preds, .predictions)
      | {board:$b, board_sha256:$bs, cid, envelope_sha256:$env[0][.cid], model_id:("h257." + .entrant),
         status, utility:.s1_utility_zero_comp,
         utility_definition:"s1_utility_zero_comp (jope.s1-two-fold/v1, H257 S1 gated; -2.0 on gate fail)",
         imported_from:{file:$src, file_sha256:$fs, producer:$prog,
                        producer_commit:"571873a38c (origin/main, PR #2120)", produced_utc:"2026-11-04"},
         h257_record:.}' "$file" | sort > "$W/$id.jsonl"
    [ "$(wc -l < "$W/$id.jsonl")" = 91 ] || { echo "$id: expected 91 records" >&2; exit 4; }
    if [ -f "$dst" ]; then cmp -s "$dst" "$W/$id.jsonl" || { echo "REFUSING: $dst differs" >&2; exit 3; }
    else mv "$W/$id.jsonl" "$dst"; fi
  done
done
echo "scratch: $W"
echo "board=$BOARD board_sha256=$BSHA contexts=$(jq .n_contexts "$OUT/BOARD.json") families=$(jq .n_families "$OUT/BOARD.json")"
for f in "$OUT"/scores/*.jsonl; do echo "$(basename "$f" .jsonl): $(jq -r .status "$f" | sort | uniq -c | tr -s ' ' | tr '\n' ' ')"; done
