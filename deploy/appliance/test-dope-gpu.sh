#!/usr/bin/env bash
# Execute the sealed GPU appliance with its own released checkpoint; outputs stay under the lane.
set -euo pipefail
package=$(realpath "$1")
proof=$(realpath -m "$2")
case $proof in /tmp|/tmp/*) echo 'proof directory must not be under /tmp' >&2; exit 2;; esac
mkdir -p "$proof"
readelf -d "$package" > "$proof/dynamic.txt"
if rg 'NEEDED.*(libcuda|libcublas|libcurand|libnvrtc)' "$proof/dynamic.txt"; then
  echo 'GPU libraries must load only after device selection' >&2; exit 1
fi
"$package" version --json > "$proof/version.json"
python3 - "$proof" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); version=json.loads((p/'version.json').read_text())
assert version['features']['dope_cuda'] is True
with (p/'train.csv').open('w') as f:
 f.write('a,b,y\n')
 for i in range(80): f.write(f'{i/80},{(i%7)/7},{i/40+(i%3)/10}\n')
PY
"$package" fit --train "$proof/train.csv" --target y --method ridge --output "$proof/model.jope" \
  --landmark-mode none --device cuda:0 --landmark-events "$proof/events.jsonl" \
  --landmark-receipt "$proof/encoding.json" --receipt "$proof/run.json" --json > "$proof/stdout.json"
python3 - "$proof" <<'PY'
import json,math,sys
from pathlib import Path
p=Path(sys.argv[1]); r=json.loads((p/'run.json').read_text()); e=json.loads((p/'encoding.json').read_text())
events=[json.loads(line) for line in (p/'events.jsonl').read_text().splitlines()]
assert r['device']['selected']=='cuda'
assert r['landmark_encoding']['sha256']
assert e['status']=='none' and not any(e['mask']) and e['version']==0
assert 'Cuda' in e['device'] and e['outcome']=='complete'
assert len(events)==1 and len(events[0]['embedding'])==512
assert all(math.isfinite(v) for v in events[0]['embedding'])
PY
CUDA_VISIBLE_DEVICES='' "$package" version --json > "$proof/version-device-hidden.json"
set +e
CUDA_VISIBLE_DEVICES='' "$package" fit --train "$proof/train.csv" --target y --output "$proof/refused.jope" \
  --landmark-mode none --device cuda:0 --landmark-receipt "$proof/refused-encoding.json" --json > "$proof/refused.json"
status=$?
set -e
[ "$status" = 4 ] && [ ! -e "$proof/refused.jope" ]
echo "Dope GPU package smoke passed: $proof"
