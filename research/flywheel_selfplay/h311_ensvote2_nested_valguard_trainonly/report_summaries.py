"""Add explicitly descriptive medians of already-scored count columns."""
from pathlib import Path
import json
from statistics import median

P = Path(__file__).resolve().parent.parent
D = Path('/home/ubuntu/jope-data/h311-astra')

def main():
    path=P/'source/metrics.json'
    raw=path.read_bytes()
    with (D/'metrics_before_count_summaries.json').open('xb') as f:f.write(raw)
    m=json.loads(raw)
    for arm in m['arms'].values():
        arm['median_wtl_componentwise']=[median([c['wtl'][i] for c in arm['cells']]) for i in range(3)]
        arm['median_invalid_excess']=median([c['invalid_excess'] for c in arm['cells']])
    m['descriptive_count_note']='WTL pooled counts are componentwise medians across nine cells, not counts from a pooled independent sample.'
    out=json.dumps(m,separators=(',',':'),allow_nan=False)+'\n'
    assert len(out.encode())<50000
    path.write_text(out)
    print('Descriptive count medians added; no fitting, scoring, bootstrap or decision change.')

if __name__=='__main__':main()
