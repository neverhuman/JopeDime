#!/usr/bin/env bash
# Run the sealed GPU appliance with its own released checkpoint; retain each proof artifact.
set -euo pipefail
package=$(realpath "$1")
proof=$(realpath -m "$2")
repo=$(cd "$(dirname "$0")/../.." && pwd)
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
(p/'predict.csv').write_text('a,b\n0.2,0.1\n0.3,0.5\n')
for kind,classes in [('binary',2),('multiclass',3)]:
 with (p/f'{kind}.csv').open('w') as f:
  f.write('a,color,y\n')
  for i in range(90):
   a='' if i%11==0 else str(i/90)
   color='' if i%13==0 else ['red','blue','green'][i%3]
   f.write(f'{a},{color},class{i%classes}\n')
PY
for mode in none progressive full; do
  "$package" fit --train "$proof/train.csv" --target y --method ridge --output "$proof/$mode.jope" \
    --landmark-mode "$mode" --landmark-subsample 40 --device cuda:0 \
    --landmark-events "$proof/$mode-events.jsonl" --landmark-receipt "$proof/$mode-encoding.json" \
    --receipt "$proof/$mode-run.json" --json > "$proof/$mode-stdout.json"
done
"$package" fit-predict --train "$proof/train.csv" --target y --method ridge \
  --input "$proof/predict.csv" --output "$proof/predictions.csv" --landmark-mode fast --device cuda:0 \
  --landmark-receipt "$proof/fit-predict-encoding.json" --receipt "$proof/fit-predict-run.json" \
  --json > "$proof/fit-predict-stdout.json"
for kind in binary multiclass; do
  method=logistic
  [ "$kind" != multiclass ] || method=multiclass_ridge
  "$package" fit --train "$proof/$kind.csv" --target y --task "$kind" --method "$method" \
    --output "$proof/$kind.jope" --landmark-mode none --device cuda:0 \
    --landmark-receipt "$proof/$kind-encoding.json" --receipt "$proof/$kind-run.json" \
    --json > "$proof/$kind-stdout.json"
done
python3 - "$proof" "$repo" <<'PY'
import json,math,sys
from pathlib import Path
import jsonschema
p,repo=map(Path,sys.argv[1:]); enc_schema=json.loads((repo/'schemas/jope.landmark-encoding.v1.schema.json').read_text())
run_schema=json.loads((repo/'schemas/jope.appliance-run.v1.schema.json').read_text())
for name in ['none','progressive','full','fit-predict','binary','multiclass']:
 r=json.loads((p/f'{name}-run.json').read_text()); e=json.loads((p/f'{name}-encoding.json').read_text())
 jsonschema.Draft202012Validator(run_schema).validate(r)
 jsonschema.Draft202012Validator(enc_schema).validate(e)
 assert r['device']['selected']=='cuda' and r['landmark_encoding']['role']=='receipt'
 assert r['landmark_encoding']['sha256'] and e['outcome']=='complete'
 assert 'Cuda' in e['device'] and e['work_budget']==2_000_000_000_000
 assert len(e['embedding'])==512 and all(math.isfinite(v) for v in e['embedding'])
 if name in ['none','fit-predict','binary','multiclass']:
  assert e['status']=='none' and not any(e['mask']) and e['version']==0
for name in ['none','progressive','full']:
 events=[json.loads(line) for line in (p/f'{name}-events.jsonl').read_text().splitlines()]
 assert events[0]['receipt']['version']==0 and not any(events[0]['receipt']['mask'])
 assert [e['receipt']['version'] for e in events]==sorted(e['receipt']['version'] for e in events)
 for e in events:
  jsonschema.Draft202012Validator(enc_schema).validate(e['receipt'])
  assert len(e['embedding'])==512 and all(math.isfinite(v) for v in e['embedding'])
 if name=='none': assert len(events)==1
 if name=='progressive': assert len(events)>=3
prog=json.loads((p/'progressive-encoding.json').read_text()); full=json.loads((p/'full-encoding.json').read_text())
assert prog['conditioning_digest']==full['conditioning_digest']
assert prog['embedding']==full['embedding']
PY
python3 - "$package" "$proof" <<'PY'
import json,struct,sys
from pathlib import Path
raw=bytearray(Path(sys.argv[1]).read_bytes()); footer=len(raw)-128
base,offset,length=struct.unpack_from('<QQQ',raw,footer+24)
manifest=json.loads(raw[offset:offset+length])
entry=next(e for e in manifest['entries'] if e['id']=='model/dope-mlc-v3/model.safetensors')
frame=entry['frames'][0]; raw[((base+4095)//4096)*4096+frame['offset']+frame['len']//2]^=1
bad=Path(sys.argv[2])/'tampered-jopedime'; bad.write_bytes(raw); bad.chmod(0o755)
PY
set +e
"$proof/tampered-jopedime" fit --train "$proof/train.csv" --target y --method ridge \
  --output "$proof/tampered.jope" --landmark-mode none --device cuda:0 \
  --landmark-receipt "$proof/tampered-encoding.json" --json > "$proof/tampered.json"
tamper_status=$?
CUDA_VISIBLE_DEVICES='' "$package" version --json > "$proof/version-device-hidden.json"
CUDA_VISIBLE_DEVICES='' "$package" fit --train "$proof/train.csv" --target y --output "$proof/refused.jope" \
  --landmark-mode none --device cuda:0 --landmark-receipt "$proof/refused-encoding.json" --json > "$proof/refused.json"
status=$?
set -e
[ "$tamper_status" = 3 ] && [ ! -e "$proof/tampered.jope" ]
[ "$status" = 4 ] && [ ! -e "$proof/refused.jope" ]
echo "Dope GPU package smoke passed: $proof"
