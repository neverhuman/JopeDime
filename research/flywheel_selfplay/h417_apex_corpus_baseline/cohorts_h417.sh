#!/bin/bash
# H417 APEX cohort manifest (metadata + table headers only; TRAIN-side data).
# Sources: the four fresh-cohort rosters built by H286/H288/H292/H294 (one train-only JTT1 context per family).
# Split rule (frozen): d = sha256("APEX-split-v1" NUL family_key); k = int(d[0:8], 16) mod 5
#   k=0 CONFIRM (never trained, read once by the champion wave), k=1 DEV, k=2 FIT-SEL, k=3,4 FIT-ICL.
# Checks (any failure stops with no output file): unique family_key / context / dataset across cohorts; table sha256 =
# roster train_only_jtt1_sha256; JTT1 nq = 0 (no query rows); dataset_id and family_key disjoint from the pinned
# TRAIN-only roster view (1,219 DL-Train rows + the 91 DL-Val ids it lists; ids only, no DL-Val table is opened).
# usage: cohorts_h417.sh OUT.json     (create-once; refuses to overwrite)
set -euo pipefail
OUT=${1:?OUT.json}
[ ! -e "$OUT" ] || { echo "$OUT exists (create-once)" >&2; exit 3; }
D=/home/ubuntu/jope-data
VIEW=$D/h317-cl/res01-train-view/selected.json
[ "$(sha256sum "$VIEW" | cut -d' ' -f1)" = 7d64774ad9dacb5fec79190edbe0c674e778556db84dcf29c474e239b68e8c79 ] \
  || { echo "TRAIN roster view pin" >&2; exit 4; }
T=$(mktemp -d "$(dirname "$OUT")/.cohorts.XXXXXX")
declare -A PIN=( [h286]=64b8925faa4e [h288]=646b9e516eca [h292]=a614fa4d503a [h294]=72531e1317ae )
: > "$T/rows.tsv"; : > "$T/src.jsonl"
for h in h286 h288 h292 h294; do
  f=$D/$h/fresh/selected.json; s=$(sha256sum "$f" | cut -d' ' -f1)
  [ "${s:0:12}" = "${PIN[$h]}" ] || { echo "$h roster pin" >&2; exit 4; }
  jq -c --arg h "$h" --arg s "$s" '{wave:$h, roster_sha256:$s, protocol:.protocol, n:(.fresh|length)}' "$f" >> "$T/src.jsonl"
  jq -r --arg h "$h" '.fresh[] | select(.task=="regression" and .source_split=="train") |
     [$h, .context_sha256, .dataset_id, .family_key, .family_id, .view_cut_id, .train_only_jtt1_sha256] | @tsv' "$f" >> "$T/rows.tsv"
done
n=$(wc -l < "$T/rows.tsv")
for col in 2 3 4; do
  u=$(cut -f$col "$T/rows.tsv" | sort -u | wc -l)
  [ "$u" = "$n" ] || { echo "column $col not unique ($u of $n)" >&2; exit 5; }
done
jq -r '(.train[], .dev[]) | .dataset_id' "$VIEW" | sort -u > "$T/roster_ds.txt"
jq -r '(.train[], .dev[]) | .family_id | sub("^fam-";"")' "$VIEW" | sort -u > "$T/roster_fam.txt"
[ "$(cut -f3 "$T/rows.tsv" | sort -u | comm -12 - "$T/roster_ds.txt" | wc -l)" = 0 ] || { echo "dataset overlap with roster" >&2; exit 5; }
[ "$(cut -f4 "$T/rows.tsv" | sort -u | comm -12 - "$T/roster_fam.txt" | wc -l)" = 0 ] || { echo "family overlap with roster" >&2; exit 5; }
names=(CONFIRM DEV FIT-SEL FIT-ICL FIT-ICL)
: > "$T/ctx.jsonl"
while IFS=$'\t' read -r h cid ds fk fid cut tsha; do
  tab=$D/$h/fresh/tables/$cid.jtt1
  [ "$(sha256sum "$tab" | cut -d' ' -f1)" = "$tsha" ] || { echo "table sha $cid" >&2; exit 6; }
  read -r nt nq w < <(od -An -tu4 -j4 -N12 "$tab")
  [ "$nq" = 0 ] || { echo "nq != 0 for $cid" >&2; exit 6; }
  d=$(printf 'APEX-split-v1\0%s' "$fk" | sha256sum | cut -c1-8)
  k=$(( 16#$d % 5 ))
  printf '{"context_sha256":"%s","dataset_id":"%s","family_key":"%s","family_id":"%s","view_cut_id":"%s","train_only_jtt1_sha256":"%s","source_wave":"%s","rows":%d,"width":%d,"split_digest8":"%s","cohort":"%s"}\n' \
    "$cid" "$ds" "$fk" "$fid" "$cut" "$tsha" "$h" "$nt" "$w" "$d" "${names[$k]}" >> "$T/ctx.jsonl"
done < "$T/rows.tsv"
jq -s -c --slurpfile src "$T/src.jsonl" --arg made "$(date -u +%FT%TZ)" '
  (sort_by(.context_sha256)) as $c
  | {schema:"jope.h417.apex_cohorts/v1", made_utc:$made,
     rule:"d=sha256(\"APEX-split-v1\" NUL family_key); k=int(d[0:8],16) mod 5; 0 CONFIRM, 1 DEV, 2 FIT-SEL, 3-4 FIT-ICL",
     train_roster_view_sha256:"7d64774ad9dacb5fec79190edbe0c674e778556db84dcf29c474e239b68e8c79",
     sources:$src, n:($c|length),
     counts:($c|group_by(.cohort)|map({(.[0].cohort):length})|add),
     cids_sha256_input:"sha256 of the newline-joined sorted context ids is in the .sha256 sidecar",
     contexts:$c}' "$T/ctx.jsonl" > "$T/out.json"
set -o noclobber
cat "$T/out.json" > "$OUT"
chmod 0444 "$OUT"
jq -r '.contexts[].context_sha256' "$OUT" | sha256sum | cut -d' ' -f1 > "$T/cids.sha"
printf '%s  %s\ncids_sha256 %s\n' "$(sha256sum "$OUT" | cut -d' ' -f1)" "$(basename "$OUT")" "$(cat "$T/cids.sha")" > "$OUT.sha256"
chmod 0444 "$OUT.sha256"
jq -c '{n, counts}' "$OUT"
