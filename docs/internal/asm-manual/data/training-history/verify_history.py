#!/usr/bin/env python3
"""Verify retained measurements against original metadata, without loading models."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILES = [
    'block-endpoint-metrics.csv', 'block-adaptive-validation.csv',
    'fixed-monitor-metrics.csv', 'gen31-epoch-metrics.csv',
    'a100-checkpoint-lineage.csv', 'a100-checkpoint-validation.csv',
    'rtx60100-step-metrics.csv',
]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    grouped = defaultdict(list)
    inventory = []
    for name in FILES:
        path = ROOT / name
        rows = list(csv.DictReader(path.open()))
        inventory.append({'file': name, 'rows': len(rows), 'sha256': digest(path)})
        for row in rows:
            grouped[row['source_path']].append((name, row))
    numeric = 0
    lineage = 0
    sources = []
    for source, rows in sorted(grouped.items()):
        path = Path(source)
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        payload = [json.loads(line) for line in raw.splitlines()] if source.endswith('.jsonl') else json.loads(raw)
        for name, row in rows:
            assert row['source_sha256'] == sha, (name, source, 'source changed')
            if 'source_json_pointer' in row:
                value = payload[int(row['source_line']) - 1] if source.endswith('.jsonl') else payload
                for key in row['source_json_pointer'].strip('/').split('/'):
                    value = value[int(key)] if isinstance(value, list) else value[key]
                assert math.isfinite(float(row['value'])) and float(row['value']) == value, (name, row)
                assert int(row['source_line']) > 0
                numeric += 1
            else:
                assert name == 'a100-checkpoint-lineage.csv'
                for key in ['step', 'model_sha256', 'parameter_sha256', 'optimizer_sha256', 'corpus_sha256', 'world', 'global_batch']:
                    assert str(payload[key]) == row[key], (source, key)
                assert row['state_sha256'] == sha
                lineage += 1
        sources.append({'path': source, 'sha256': sha})
    assert numeric == 3218 and lineage == 78, (numeric, lineage)
    rtx = list(csv.DictReader((ROOT/'rtx60100-step-metrics.csv').open()))
    assert sorted({int(row['step']) for row in rtx}) == list(range(60001, 60101))
    result = {'schema': 'asm.training_history.verification.v1', 'status': 'PASS',
              'numeric_values_exactly_recovered': numeric, 'checkpoint_lineages_verified': lineage,
              'model_or_corpus_payloads_loaded': False, 'artifacts': inventory, 'sources': sources,
              'limitations': ['JSON line anchors identify source record or original-log reference; JSON Pointer identifies the exact nested metric.',
                             'Original remote metrics referenced by the108-block capture were not independently rehashed: only the captured endpoints and recorded original digest are available.',
                             'This is a metadata reconstruction, not numerical re-execution of training.']}
    (ROOT/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ['status','numeric_values_exactly_recovered','checkpoint_lineages_verified']}))

if __name__ == '__main__':
    main()
