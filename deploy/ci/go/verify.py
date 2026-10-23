"""Require nonzero execution, closed registry accounting and raw facade coverage."""
import json
from pathlib import Path
import re


def verify(root, profile):
    logs = {}
    # Checked minimums prevent a filtered one-test run from qualifying a lane.
    minimums = dict(recipes=90, fitted=2, bindings=21, pipeline=19, cli=2, census=5)
    for name in ('recipes', 'fitted', 'bindings', 'pipeline', 'cli', 'census'):
        text = (root / (name + '.log')).read_text()
        counts = re.findall(r'test result: ok\. (\d+) passed; (\d+) failed', text)
        assert counts and sum(int(p) for p, _ in counts) >= minimums[name], name + ': incomplete tests'
        if name == 'recipes':
            assert 'test go_runtime_is_required ... ok' in text, 'recipes: missing Go presence gate'
            full = re.findall(r'test result: ok\. (\d+) passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;', text)
            assert any(int(n) >= minimums[name] for n in full), 'recipes: filtered or partial suite'
        assert all(int(f) == 0 for _, f in counts) and 'test result: FAILED' not in text
        if name in ('recipes', 'fitted', 'census'):
            assert text.count('Go 0:') == text.count('Go 1:') > 0, name + ': missing engine'
        logs[name] = text
    census = [json.loads(s) for s in re.findall(r'Go census: (\{[^\n]+\})', logs['census'])]
    assert sorted(c['kind'] for c in census) == ['bound', 'legacy-bound', 'recipe', 'v2']
    for c in census:
        assert c['total'] > 0 and c['unaccounted'] == 0
        if c['kind'] == 'v2':
            assert c['executed'] == c['total']
        else:
            assert len(set(c['executed'])) == len(c['executed'])
            assert len(set(c['refused'])) == len(c['refused'])
            assert len(c['executed']) + len(c['refused']) == c['total']
            assert not set(c['executed']) & set(c['refused'])
    assert logs['census'].count('Go metadata: strict heading qualified') == 1
    expected = sum(c['executed'] if c['kind'] == 'v2' else len(c['executed']) for c in census) + 6
    assert logs['census'].count('Go 0:') == expected
    contracts = (root / 'contracts.log').read_text()
    assert re.search(r'^PASS$', contracts, re.M) and '--- FAIL:' not in contracts
    rows = profile.read_text().splitlines()
    assert rows[0] == 'mode: atomic'
    blocks = [r.split() for r in rows[1:]]
    assert blocks and all(len(b) == 3 and b[0].startswith('jope.generated/model/model.go:') for b in blocks)
    assert sum(int(b[1]) for b in blocks) > 0
    assert all(int(b[1]) == 0 or int(b[2]) > 0 for b in blocks), 'uncovered facade statements'
    return census


if __name__ == '__main__':
    root = Path('target/go-evidence')
    census = verify(root, Path('target/go-contract/coverage.out'))
    (root / 'census.json').write_text(json.dumps(census, indent=2) + '\n')
    print('Complete Go censuses, both engines and 100% facade statements verified')
