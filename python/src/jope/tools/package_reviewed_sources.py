#!/usr/bin/env python3
"""Package one immutable copy of newly reviewed originals; never materialize cuts."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
review_path = root / 'manifests/generated/provenance-review/review-v2.json'
candidate_path = root / 'manifests/generated/provenance-review/pilot-candidates-v2.json'
review = json.loads(review_path.read_text())
candidates = json.loads(candidate_path.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert candidates['review_sha256'] == sha(review_path)
selected = {row['id'] for row in candidates['datasets']}
files = {}
backing = {}
withheld = []
def entry(source, destination, dataset, role):
    assert source.is_file() and not source.is_symlink(), source
    value = dict(source=str(source), destination=str(destination), sha256=sha(source),
                 bytes=source.stat().st_size, dataset=dataset, role=role)
    previous = files.setdefault(str(destination), value)
    assert previous['sha256'] == value['sha256'], destination
    return value

for row in review['datasets']:
    if row['id'] not in selected:
        continue
    assert row['observational_origin'] == 'verified_documented_measurements'
    assert row['pilot_candidate_status'] == 'eligible_for_fresh_cohort_preparation'
    metadata = Path(row['metadata']['path'])
    assert sha(metadata) == row['metadata']['sha256']
    assert sha(Path(row['documentation']['path'])) == row['documentation']['sha256']
    directory = metadata.parent
    assert sha(directory / 'train.csv') == row['reconstruction']['training_sha256']
    destination = Path('data/originals/review-v2/regression') / row['id']
    for source in sorted(directory.rglob('*')):
        if not source.is_file():
            continue
        assert not any(p in ('.git', '__pycache__', '.venv', 'target') for p in source.parts)
        if source.suffix == '.csvx':
            withheld.append(dict(source=str(source), sha256=sha(source), bytes=source.stat().st_size,
                                 dataset=row['id'], admission='unresolved',
                                 reason='Alternate numeric table lacks a reconstruction/transform receipt; source pointer retained, payload withheld.'))
            continue
        assert source.suffix in ('.csv', '.json', '.jsonl', '.vs', '.txt', '.log', '.md', '.png', '.svg', '.pdf'), source
        relative = source.relative_to(directory)
        role = 'real_dataset_original' if relative.name in ('train.csv', 'test.csv', 'meta.json') else 'real_dataset_legacy_sidecar'
        entry(source, destination / relative, row['id'], role)
    source = Path(row['reconstruction']['source'])
    assert sha(source) == row['reconstruction']['source_sha256']
    relative = source.relative_to('/home/ubuntu')
    value = entry(source, Path('data/originals/backing') / relative, row['id'], 'real_backing_source')
    backing[value['destination']] = value

manifest_dir = root / 'manifests/generated'
for name, payload in [('review-v2-transfer.json', list(files.values())), ('backing-review-v2-transfer.json', list(backing.values()))]:
    path = manifest_dir / name
    path.write_text(json.dumps(dict(schema='parallel.transfer.v1', files=payload), indent=2) + '\n')
(manifest_dir / 'review-v2-withheld-artifacts.json').write_text(json.dumps(withheld, indent=2) + '\n')
print(json.dumps(dict(reviewed_datasets=len(selected), files=len(files), bytes=sum(f['bytes'] for f in files.values()),
                      withheld_artifacts=len(withheld), copied_cut_csv_files=0, cohort_activated=False)))
