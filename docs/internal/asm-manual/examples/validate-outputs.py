#!/usr/bin/env python3
"""Check the private five-runtime recipe example; this never executes a model."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--recipe-output', type=Path, required=True)
args = parser.parse_args()
names = ('python', 'rust', 'cpp17', 'cpp20', 'js')
rows = {}
artifacts = {}
for name in names:
    path = args.recipe_output / ('pred-' + name + '.txt')
    raw = path.read_bytes()
    values = [float(line) for line in raw.decode().splitlines() if line.strip()]
    if len(values) != 8 or not all(math.isfinite(value) for value in values):
        raise SystemExit('Expected exactly eight finite predictions: ' + str(path))
    rows[name] = [struct.pack('<d', value).hex() for value in values]
    artifacts[name] = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}
if any(value != rows['python'] for value in rows.values()):
    raise SystemExit('Required language prediction bits differ')
print(json.dumps({'schema': 'jope.maintainer-recipe-parity/v1', 'ok': True,
                  'required_backends': list(names), 'prediction_rows': 8,
                  'artifacts': artifacts, 'prediction_bits': rows['python'],
                  'scope': 'synthetic recipe refits; not frozen fitted-state export qualification'},
                 indent=2))
