#!/bin/bash
# H335 Stage V S3 (stage_v_mounts_h335.md): build the xbabe2 Stage V root $SV ONLY after the pinned unlock (S2).
# Hard links and read-only copies only; nothing is decoded. Every check is fatal (it stops the script, never the operator
# shell). Idempotent: an existing link must already be the same file (-ef) and an existing copy must have its pinned
# sha256; nothing is overwritten or removed. Run on the xbabe2 host shell: ./build_stagev_view.sh
set -euo pipefail
DATA=/home/ubuntu/jope-data/h335-cl
SV=/home/ubuntu/jope-data/h335-stagev
VIEW=/home/ubuntu/jope-data/h335-view/res01-train-view   # TRAIN-only view (+ cpu_exact_fused.py), = sandbox_h335.sh
RES=/home/ubuntu/jope-data/res-01/input                  # canonical res-01 (TRAIN + DL-Val)
H258=/home/ubuntu/jope_plan/evidence/H258/argon-t1405/inputs
RECEIPT=/home/ubuntu/jope_plan/ns/receipts/H335_LOCK_sha256.txt
SEL_SHA=981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1     # ORIGINAL selected.json
FUSED_SHA=b9f1468a61aa4dc888c8f2931f9c759f7d4f057793a77ccb05eed6300fb4f9ab   # cpu_exact_fused.py
REF_SHA=30853cad78d860870338316dfb5a4e869e46dc5c22d42fe4ab9d5f24f31c373a     # best_dev_checkpoint.json (H258)
COSTS_SHA=0e812aa7ea4ce46a85cbe17cb7eba3a58678cf13ec3b215d48a334dd5626f11f   # r882-secondary-costs.npz (H258)
die() { echo "build_stagev_view.sh FAILED: $*" >&2; exit 5; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
lnk() {  # hard link $1 -> $2, or verify that $2 already is $1
  if [ -e "$2" ]; then [ "$1" -ef "$2" ] || die "$2 exists and is not a link of $1"
  else ln "$1" "$2" || die "ln $1 $2"; fi
}
cpy() {  # read-only copy $1 -> dir $2 with pinned sha $3, or verify the existing copy
  local dst
  dst=$2/$(basename "$1")
  [ -e "$dst" ] || install -m 0444 "$1" "$2/" || die "install $1"
  [ "$(sha "$dst")" = "$3" ] || die "$dst sha != pinned ${3:0:12}"
}
[ "$(hostname)" = xbabe2 ] || die "run on xbabe2"
[ -s "$RECEIPT" ] || die "no LOCK receipt"
R=$(cut -d' ' -f1 "$RECEIPT")
[ "$(sha "$DATA/stage_v/LOCK.json")" = "$R" ] || die "LOCK.json does not match the receipt"
[ -s "$DATA/stage_v/STAGE_V_STARTED" ] || die "not unlocked (no STAGE_V_STARTED): DL-Val stays unread"
[ "$(cut -d' ' -f2 "$DATA/stage_v/STAGE_V_STARTED")" = "$R" ] || die "STAGE_V_STARTED does not name this LOCK"
[ "$(find "$VIEW/tables" -name '*.jtt1' | wc -l)" = 1219 ] || die "TRAIN view tables != 1219"
[ "$(find "$VIEW/pools" -name '*.npz' | wc -l)" = 1219 ] || die "TRAIN view pools != 1219"
mkdir -p "$SV/res01-v-view/tables" "$SV/res01-v-view/pools" "$SV/inputs" "$SV/run" || die "mkdir $SV"
for f in "$VIEW"/tables/*.jtt1; do lnk "$f" "$SV/res01-v-view/tables/${f##*/}"; done
for f in "$VIEW"/pools/*.npz; do lnk "$f" "$SV/res01-v-view/pools/${f##*/}"; done
for f in champions.json winners.json source_manifest.json cpu_exact_fused.py; do lnk "$VIEW/$f" "$SV/res01-v-view/$f"; done
T=$SV/inputs/target_cids.txt
[ -e "$T" ] || { jq -r '.dev[].context_sha256' "$VIEW/selected.json" > "$T.tmp" && mv "$T.tmp" "$T"; } \
  || die "target cids from the redacted selected.json"
[ "$(wc -l < "$T")" = 91 ] || die "$T does not hold 91 DL-Val ids"
while read -r c; do
  [[ "$c" =~ ^[0-9a-f]{64}$ ]] || die "malformed id in $T"
  lnk "$RES/tables/$c.jtt1" "$SV/res01-v-view/tables/$c.jtt1"
  lnk "$RES/pools/$c.npz" "$SV/res01-v-view/pools/$c.npz"
done < "$T"
lnk "$RES/selected.json" "$SV/res01-v-view/selected.json"
cpy "$H258/r882/best_dev_checkpoint.json" "$SV/inputs" "$REF_SHA"
cpy "$H258/r882-secondary-costs.npz" "$SV/inputs" "$COSTS_SHA"
[ "$(sha "$SV/res01-v-view/selected.json")" = "$SEL_SHA" ] || die "V view selected.json is not the ORIGINAL"
[ "$(sha "$SV/res01-v-view/cpu_exact_fused.py")" = "$FUSED_SHA" ] || die "cpu_exact_fused.py sha"
[ "$(find "$SV/res01-v-view/tables" -name '*.jtt1' | wc -l)" = 1310 ] || die "V view tables != 1,310"
[ "$(find "$SV/res01-v-view/pools" -name '*.npz' | wc -l)" = 1310 ] || die "V view pools != 1,310"
echo "V root OK: $SV (1,310 tables / pools, original selected.json, R882 inputs pinned)"
