#!/usr/bin/env python3
"""Create an explicit runtime allowlist; data/model transfers use separate manifests."""
import hashlib
import json
import sys
from pathlib import Path

from kernel_release import runtime_files

root=Path(__file__).resolve().parents[1]
paths=list(runtime_files(root))
files=[dict(source=str(p),destination=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,dataset=None,role='parallel_source') for p in sorted(paths)]
out=Path(sys.argv[1]);out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(dict(schema='parallel.transfer.v1',files=files),indent=2)+'\n')
print(f'{len(files)} runtime files in {out}')
