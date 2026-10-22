"""Fail closed on absent tests, incomplete censuses, or absent coverage files."""
import json
import pathlib
import re

root = pathlib.Path('target/typescript-evidence')
for name in ('recipes', 'fitted', 'bindings', 'pipeline', 'cli', 'census'):
    text = (root / (name + '.log')).read_text()
    counts = re.findall(r'test result: ok\. (\d+) passed; (\d+) failed', text)
    assert counts and sum(int(p) for p, _ in counts) > 0, name + ': zero tests'
    assert all(int(f) == 0 for _, f in counts), name + ': failed tests'
    if name in ('recipes', 'fitted', 'census'):
        assert text.count('TypeScript cjs:') == text.count('TypeScript esm:') > 0, name
censuses = [json.loads(s) for s in re.findall(r'TypeScript census: (\{[^\n]+\})', (root/'census.log').read_text())]
assert sorted(c['kind'] for c in censuses) == ['bound', 'legacy-bound', 'recipe', 'v2']
for c in censuses:
    assert c['total'] > 0 and c['unaccounted'] == 0
    if c['kind'] == 'v2':
        assert c['executed'] == c['total']
    else:
        assert len(c['executed']) + len(c['refused']) == c['total']
        assert not set(c['executed']) & set(c['refused'])
expected_executions = sum(c['executed'] if c['kind'] == 'v2' else len(c['executed']) for c in censuses) + 5  # categorical codec variants
assert (root/'census.log').read_text().count('TypeScript cjs:') == expected_executions
contracts = (root/'contracts.log').read_text()
assert re.search(r'pass [1-9]\d*', contracts) and re.search(r'fail 0\b', contracts)
assert re.search(r'model\.js\s+\|\s+100\.00\s+\|\s+100\.00\s+\|\s+100\.00', contracts), 'facade coverage absent or below 100%'
(root/'census.json').write_text(json.dumps(censuses, indent=2)+'\n')
print('Complete TypeScript censuses and 100% facade line/branch/function coverage verified')
