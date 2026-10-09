"""Verify imported source hashes and syntax without writing bytecode."""
import ast,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
repo=root.parents[1]
manifest=json.loads((root/'SOURCE_MANIFEST.json').read_bytes())
for item in manifest['files']:
    p=repo/item['path']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],p
for p in root.rglob('*.py'):
    ast.parse(p.read_bytes(),filename=str(p))
print(f"Verified {len(manifest['files'])} imported files and Python syntax")
