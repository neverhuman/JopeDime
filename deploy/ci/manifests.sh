#!/usr/bin/env bash
# deploy/ci/manifests.sh — manifests are consistent with what they describe; no weights are downloaded.
#   models/manifest.json (jope.model-index/v1) ↔ models/bundles/<id>.json (jope.model-bundle/v2) ↔ schemas/jope.model-bundle.v2.schema.json
#   models/PINNED names only bundles the index knows.
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 - <<'PY'
import json, hashlib, pathlib, re, sys
fail = []
schema = json.loads(pathlib.Path('schemas/jope.model-bundle.v2.schema.json').read_text())
index = json.loads(pathlib.Path('models/manifest.json').read_text())
assert index['schema'] == 'jope.model-index/v1', index['schema']
known = set()
def check_schema(node, spec, where):
    # Minimal dependency-free validator for the subset of JSON Schema used by our schema files:
    # type / const / enum / required / properties / items / pattern / minimum / minItems / minLength.
    t = spec.get('type')
    if t:
        types = t if isinstance(t, list) else [t]
        ok = any({'object': isinstance(node, dict), 'array': isinstance(node, list), 'string': isinstance(node, str),
                  'integer': isinstance(node, int) and not isinstance(node, bool), 'null': node is None}[x] for x in types)
        if not ok: fail.append(f"{where}: expected {t}"); return
    if 'const' in spec and node != spec['const']: fail.append(f"{where}: expected {spec['const']!r}, got {node!r}")
    if 'enum' in spec and node not in spec['enum']: fail.append(f"{where}: {node!r} not in {spec['enum']}")
    if isinstance(node, str):
        if 'pattern' in spec and not re.search(spec['pattern'], node): fail.append(f"{where}: {node!r} !~ {spec['pattern']}")
        if 'minLength' in spec and len(node) < spec['minLength']: fail.append(f"{where}: too short")
    if isinstance(node, int) and not isinstance(node, bool) and 'minimum' in spec and node < spec['minimum']: fail.append(f"{where}: below minimum")
    if isinstance(node, dict):
        for r in spec.get('required', []):
            if r not in node: fail.append(f"{where}: missing required {r}")
        for k, sub in spec.get('properties', {}).items():
            if k in node: check_schema(node[k], sub, f"{where}.{k}")
    if isinstance(node, list):
        if 'minItems' in spec and len(node) < spec['minItems']: fail.append(f"{where}: fewer than {spec['minItems']} items")
        if 'items' in spec:
            for i, item in enumerate(node): check_schema(item, spec['items'], f"{where}[{i}]")
for row in index['bundles']:
    p = pathlib.Path(row['manifest']); raw = p.read_bytes(); m = json.loads(raw)
    known.add(m['bundle_id'])
    if hashlib.sha256(raw).hexdigest() != row['manifest_sha256']: fail.append(f"{p}: sha256 differs from models/manifest.json")
    if m['bundle_id'] != row['bundle_id']: fail.append(f"{p}: bundle_id {m['bundle_id']} != index {row['bundle_id']}")
    if row['s3_key'] != f"bundles/{m['bundle_id']}/manifest.json": fail.append(f"{p}: s3_key")
    if row['files'] != len(m['files']) or row['bytes'] != sum(f['bytes'] for f in m['files']): fail.append(f"{p}: files/bytes totals differ from the index")
    check_schema(m, schema, str(p))
    for f in m['files']:
        if f['blob'] != f"blobs/sha256/{f['sha256']}": fail.append(f"{p}: {f['path']} blob key != sha256")
    seen = [f['path'] for f in m['files']]
    if len(seen) != len(set(seen)): fail.append(f"{p}: duplicate paths")
    print(f"ok {p} ({len(m['files'])} files, {sum(f['bytes'] for f in m['files'])} bytes)")
for line in pathlib.Path('models/PINNED').read_text().splitlines():
    line = line.strip()
    if line and not line.startswith('#') and line not in known: fail.append(f"models/PINNED names unknown bundle {line}")
for p in pathlib.Path('models/bundles').glob('*.json'):
    if json.loads(p.read_text())['bundle_id'] not in known: fail.append(f"{p} is not listed in models/manifest.json")

compat_path = pathlib.Path('models/dope-compat.json')
compat = json.loads(compat_path.read_text())
if compat.get('schema') != 'jope.dope-compat/v1':
    fail.append("models/dope-compat.json: invalid schema")
eval_id = json.loads(pathlib.Path('crates/feat-math/tests/fixtures/compat_freeze/evaluator_identity.v1.json').read_text())
if compat.get('evaluator_cargo_lock_subgraph_sha256') != eval_id['cargo_lock_sha256']:
    fail.append("models/dope-compat.json: evaluator_cargo_lock_subgraph_sha256 differs from evaluator_identity.v1.json")
if compat.get('evaluator_identity_digest') != eval_id['identity_digest']:
    fail.append("models/dope-compat.json: evaluator_identity_digest differs from evaluator_identity.v1.json")
r882 = json.loads(pathlib.Path('models/bundles/r882.json').read_text())
r882_dope = next((f['sha256'] for f in r882['files'] if f['path'] == 'dope.safetensors'), None)
uicl = json.loads(pathlib.Path('models/bundles/unified-icl-regbin-v2.json').read_text())
if compat.get('fused_bundle_dope_safetensors_sha256') != r882_dope:
    fail.append("models/dope-compat.json: fused_bundle_dope_safetensors_sha256 differs from r882.json")
uicl_consumer = compat.get('consumers', {}).get('unified_icl_regbin_v2', {})
if uicl_consumer.get('dope_git_rev') != uicl['dope_pin']['git_rev'] or uicl_consumer.get('dope_safetensors_sha256') != uicl['dope_pin']['weights_sha256']:
    fail.append("models/dope-compat.json: unified_icl_regbin_v2 consumer differs from models/bundles/unified-icl-regbin-v2.json")
enc_cargo = pathlib.Path('crates/jope-encoders/Cargo.toml').read_text()
if f'rev = "{compat.get("engine_source_commit")}"' not in enc_cargo or compat.get('preparation_source') != compat.get('engine_source_commit'):
    fail.append("models/dope-compat.json: engine_source_commit / preparation_source differs from crates/jope-encoders/Cargo.toml")
tok_contract = compat.get('contracts', {}).get('h256_token_layout', {})
tok_path = pathlib.Path(tok_contract.get('path', ''))
if not tok_path.is_file():
    fail.append(f"models/dope-compat.json: missing {tok_path}")
else:
    tok_raw = tok_path.read_bytes()
    tok_doc = json.loads(tok_raw)
    if hashlib.sha256(tok_raw).hexdigest() != tok_contract.get('sha256') or tok_doc.get('schema') != tok_contract.get('schema'):
        fail.append("models/dope-compat.json: h256_token_layout sha256 or schema mismatch")
print("ok models/dope-compat.json")

if fail:
    print('\n'.join(fail)); sys.exit(1)
print("manifests ok")
PY
