#!/bin/bash
# Stage the H354 code to a frozen read-only copy outside ~/jope_plan/evidence/ (the H317 guard refuses every other
# wave's evidence tree before Stage V; H335 ran the same way from ~/h335-stage/code).
#   stage_h354.sh TAG   ->  /home/ubuntu/h354-stage/code-TAG  (create-once; every file sha256-listed in CODE.sha256)
# The run scripts take the staged dir as their code root and check CODE.sha256 before starting.
set -euo pipefail
SRC=$(dirname "$(readlink -f "$0")")
TAG=${1:?TAG}
DST=/home/ubuntu/h354-stage/code-$TAG
[ -e "$DST" ] && { echo "$DST exists (create-once)" >&2; exit 3; }
mkdir -p "$DST"
(cd "$SRC" && find . -maxdepth 1 -type f \( -name '*.py' -o -name '*.sh' -o -name '*.md' \) ! -name '.*' | sort \
  | sed 's#^\./##' > "$DST/.files")
rsync -a --chmod=Fa-w --files-from="$DST/.files" "$SRC/" "$DST/"
(cd "$DST" && xargs -a .files sha256sum > CODE.sha256 && chmod a-w CODE.sha256 .files)
echo "staged $(wc -l < "$DST/.files") files to $DST; CODE.sha256 $(sha256sum "$DST/CODE.sha256" | cut -c1-16)"
