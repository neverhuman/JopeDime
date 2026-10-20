"""Read immutable result records; never rank incompatible identities together."""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from protocol import validate_result


def scoreboard(paths):
    groups = {}
    for path in paths:
        r = validate_result(json.loads(Path(path).read_text()))
        groups.setdefault(r['comparison_identity_sha256'], []).append({
            'run_id': r['run_id'], 'branch': r['branch'], 'concept': r['concept'],
            'state': r['state'], 'metrics': r.get('metrics', {}), 'primary': r.get('primary'),
            'counts': r['counts'], 'cost': r.get('cost'), 'source': str(path)})
    return {'schema': 'jope.research-scoreboard/v1', 'comparison_groups': groups,
            'rule': 'Separate identities/roles/cuts/budgets are separate groups. Failed/unfinished rows retained.'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('results', nargs='+', type=Path)
    a = ap.parse_args()
    print(json.dumps(scoreboard(a.results), indent=2, allow_nan=False))
