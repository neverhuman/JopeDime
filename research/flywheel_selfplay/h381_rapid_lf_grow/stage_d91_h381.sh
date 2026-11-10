#!/bin/bash
# Stage the H381 part-B (dev91) code root on xbabe2 and copy it to xbabe0 (create-once): the frozen H335 files (each
# byte-identical to origin/v1/next) + gp_h381.py + d91_h381.py + sandbox_h381_d91.sh, each byte-identical to
# origin/v1/next too, read-only, with CODE.sha256 next to code/ (sandbox_h381_d91.sh checks it on every call).
#   stage_d91_h381.sh TAG   ->  /home/ubuntu/h381-stage/d91-TAG on xbabe2 and xbabe0
set -euo pipefail
TAG=${1:?TAG}
REPO=/home/ubuntu/JopeDime
H335=/home/ubuntu/h335-stage/code
G335=research/flywheel_selfplay/h335_aggressive_locked_elo
G381=research/flywheel_selfplay/h381_rapid_lf_grow
DST=/home/ubuntu/h381-stage/d91-$TAG
SSH=(ssh -o BatchMode=yes xbabe0)
[ -e "$DST" ] && { echo "$DST exists (create-once)" >&2; exit 3; }
"${SSH[@]}" "[ ! -e $DST ]" || { echo "xbabe0:$DST exists (create-once)" >&2; exit 3; }
git -C "$REPO" -c credential.helper= -c "credential.helper=!/home/ubuntu/.local/bin/gh-role writer auth git-credential" \
  fetch -q https://github.com/neverhuman/JopeDime.git +refs/heads/v1/next:refs/remotes/origin/v1/next
mkdir -p "$DST/code"
(cd "$H335" && find . -maxdepth 1 -type f ! -name '.*' | sort | sed 's#^\./##') > "$DST/h335.files"
while read -r f; do
  cmp -s "$H335/$f" <(git -C "$REPO" show "origin/v1/next:$G335/$f") \
    || { echo "H335 file $f differs from origin/v1/next" >&2; exit 4; }
done < "$DST/h335.files"
rsync -a --chmod=Fa-w --files-from="$DST/h335.files" "$H335/" "$DST/code/"
for f in gp_h381.py d91_h381.py sandbox_h381_d91.sh; do
  git -C "$REPO" show "origin/v1/next:$G381/$f" > "$DST/code/$f"   # the published bytes, not the working copy
  chmod 0555 "$DST/code/$f"
done
git -C "$REPO" rev-parse origin/v1/next > "$DST/source_commit"
(cd "$DST/code" && find . -maxdepth 1 -type f | sort | sed 's#^\./##' | xargs sha256sum) > "$DST/CODE.sha256"
chmod a-w "$DST/CODE.sha256" "$DST/h335.files" "$DST/source_commit"
"${SSH[@]}" "mkdir -p /home/ubuntu/h381-stage"
rsync -a "$DST/" "xbabe0:$DST/"
"${SSH[@]}" "cd $DST/code && sha256sum --quiet -c ../CODE.sha256"
echo "staged $(wc -l < "$DST/CODE.sha256") files to $DST (xbabe2 + xbabe0) from $(cut -c1-10 "$DST/source_commit"); CODE.sha256 $(sha256sum "$DST/CODE.sha256" | cut -c1-16)"
