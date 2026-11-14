#!/usr/bin/env bash
# Stages the H458 runtime copy: create-once, read-only, with the source commit, CODE.sha256 and pinned dependencies.
# usage: stage_h458.sh [<lane dir>] [<stage dir>]   (default stage dir: /home/ubuntu/h458-stage/code-<commit>)
set -euo pipefail
LANE="${1:-/home/ubuntu/JopeDime/.agent/worktrees/h458-flywheel}"
STAGE="${2:-/home/ubuntu/h458-stage/code-$(git -C "$LANE" rev-parse --short=10 HEAD)}"
WAVE="research/flywheel_selfplay/h458_test_informed_flywheel"
H452="/home/ubuntu/h452-stage/code"
VENDOR_FILES="cols_104m_e8_gpresid_h452.py stack_guard_cv7_h452.py"

[ -e "$STAGE" ] && { echo "stage dir $STAGE exists (create-once)"; exit 2; }
if [ -n "$(git -C "$LANE" status --porcelain -- "$WAVE")" ]; then
  echo "uncommitted changes under $WAVE; commit first"; exit 2
fi
SRC="$(git -C "$LANE" rev-parse HEAD)"
mkdir -p "$STAGE/vendor"
for f in $(git -C "$LANE" ls-tree --name-only "$SRC" "$WAVE/" | xargs -n1 basename); do
  git -C "$LANE" show "$SRC:$WAVE/$f" > "$STAGE/$f"
done
chmod +x "$STAGE"/*.sh
for f in $VENDOR_FILES; do cp --preserve=timestamps "$H452/$f" "$STAGE/vendor/$f"; done
echo "$SRC" > "$STAGE/source_commit"

# pin the bytes of every dependency module the wave imports
( cd "$STAGE" && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 /usr/bin/python3 -B - <<'PY'
import json
import paths_h458 as P
P.setup_paths()
import episodes_h458, probe_h458, train_h458  # noqa: F401  (imports every dependency)
assert P.dep_dir() == P.VENDOR, "vendor folder not in use"
shas = P.verify_deps(strict=False)
with open("deps_h458.json", "x") as fh:
    fh.write(json.dumps({"schema": "jope.h458.deps/v1", "sha256": shas}, indent=1, sort_keys=True) + "\n")
print("pinned", len(shas), "dependency files")
PY
)
( cd "$STAGE" && find . -type f ! -name CODE.sha256 -print0 | sort -z | xargs -0 sha256sum > CODE.sha256 )
chmod -R a-w "$STAGE"
echo "staged $STAGE at $SRC"
sha256sum "$STAGE/CODE.sha256"
