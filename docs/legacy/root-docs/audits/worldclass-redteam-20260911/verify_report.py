#!/usr/bin/env python3
"""Verify audit arithmetic, local references, action dependencies and current artifact hashes.
Does not execute application code, fits, benchmarks, model loads or protected-data reads.
"""
import csv
import datetime
import hashlib
import json
import re
import statistics
from pathlib import Path

BASE = Path(__file__).resolve().parent
checks = []

def record(name, ok, detail):
    checks.append({'name': name, 'passed': bool(ok), 'detail': detail})

def read(name):
    return json.loads((BASE / name).read_text())

c = read('data-folds-evidence/metadata-census.json')
record('active partition arithmetic', c['active']['rows'] == 8894 and sum(c['active']['partitions'].values()) == 8894 and c['active']['eligible_rows'] == 8016, c['active'])
rows = [x['train_rows'] for x in c['geometry']]
record('active geometry denominator', len(rows) == 8894 and statistics.median(rows) == 275 and sum(x > 1000 for x in rows) == 10, {'contexts': len(rows), 'median': statistics.median(rows), 'above1000': sum(x > 1000 for x in rows)})
record('empty bounded eligible expansion', c['expansion']['metadata_eligible_additions'] == 0 and c['expansion']['registered_eligible'] == 0 and c['expansion']['component_metadata_complete'] == 10, c['expansion'])
record('declared repeated coverage', c['coverage']['declared_jobs'] == 28560 and c['coverage']['distinct_cuts'] == 140, {'declared': c['coverage']['declared_jobs'], 'distinct': c['coverage']['distinct_cuts']})
t = read('learning-derived-summary.json')
record('training reset census', (t['blocks'], t['parent_count'], t['corpora'], t['optimizer_reset_blocks']) == (108,62,108,108), {k:t[k] for k in ['blocks','parent_count','corpora','optimizer_reset_blocks']})
record('nominal presentation arithmetic', t['nominal_steps'] == 108*3000 and t['nominal_global_examples'] == 108*3000*2048, {'steps': t['nominal_steps'], 'presentations':t['nominal_global_examples']})
p = read('learning-performance-summary.json')
controller = sum(x['seconds'] for x in p['controller_attempts']['passed'])
ratio = t['measured_step_seconds']['sum'] / controller
record('step/controller scope arithmetic', abs(controller-25206.182609528) < 0.01 and round(ratio*100,2) == 79.37, {'controller_seconds':controller,'measured_step_fraction':ratio,'not_gpu_arithmetic_occupancy':True})
paired = read('evidence/paired-diagnostic.json')
v = read('evidence/diagnostic-verification.json')
record('paired endpoint counts', paired['rows'] == 420 and paired['groups'] == 140 and v['summary_receipts_matched'] == 420 and paired['baseline_available'] == 0, {'rows':paired['rows'],'groups':paired['groups'],'search_summary_matches':v['summary_receipts_matched'],'native_admission_verified':v['native_admission_verified']})
a = paired['partitions']['meta_train']['by_opponent']['generic']
dev = paired['partitions']['meta_validation']['by_opponent']['generic']
record('adaptive wins/ties/losses', (a['win'],a['tie'],a['loss']) == (1,120,3) and (dev['win'],dev['tie'],dev['loss']) == (0,16,0), {'train':[a['win'],a['tie'],a['loss']],'development':[dev['win'],dev['tie'],dev['loss']]})
with (BASE/'paired-outcomes.csv').open() as f:
    count = sum(1 for _ in csv.DictReader(f))
record('paired CSV row count', count == 280, {'rows':count})
register = read('gap-register.json')
ids = [x['id'] for x in register['items']]
index = {x['id']: x for x in register['items']}
seen, active, errors = set(), set(), []
def visit(rid):
    if rid in active:
        errors.append('cycle at '+rid)
        return
    if rid in seen: return
    if rid not in index:
        errors.append('missing '+rid)
        return
    active.add(rid)
    for dep in index[rid]['depends_on']: visit(dep)
    active.remove(rid)
    seen.add(rid)
for rid in ids: visit(rid)
record('action dependency graph', len(ids) == len(set(ids)) == 34 and not errors, {'actions':len(ids),'errors':errors})
record('status separation', all(all(k in x for k in ['implementation_status','verification_status','deployment_status','scientific_status']) and not x['closed'] for x in register['items']) and not register['overall_scientific_objective_complete'], {'overall_complete':register['overall_scientific_objective_complete']})
missing, refs, footnote_errors = [], 0, []
for name in ['REPORT.md','PRECISION_DESIGN.md','data-folds.md','learning-hpc.md','authority-gp.md','precision-layout.md','precision-addendum.md']:
    text = (BASE/name).read_text()
    for label, target in re.findall(r'\[([^\]]+)\]\(([^)]+)\)', text):
        if target.startswith(('https://','http://','#')): continue
        refs += 1
        if not (BASE/target.split('#')[0]).resolve().exists():
            missing.append({'file':name,'target':target})
    definitions = set(re.findall(r'^\[\^([^\]]+)\]:', text, re.M))
    uses = set(re.findall(r'\[\^([^\]]+)\](?!:)', text))
    if uses-definitions: footnote_errors.append({'file':name,'undefined':sorted(uses-definitions)})
for item in register['items']:
    for target in item['evidence']:
        refs += 1
        if not (BASE/target).resolve().exists(): missing.append({'action':item['id'],'target':target})
# The index is emitted below and intentionally does not hash itself.
missing = [x for x in missing if x.get('target') != 'evidence-index.json']
record('local artifact links', not missing, {'references':refs,'missing':missing})
record('footnote definitions', not footnote_errors, {'errors':footnote_errors})
policy = (BASE/'evidence/promotion_gate.py').read_text()
record('frozen primary promotion rule', 'statistics.fmean(means) < .01 or ci[0] <= 0' in policy and "for reference in ('recovery', 'original_serving', 'serving')" in policy, {'rule':'mean improvement >=1%; confidence lower bound >0; three primary references'})
artifacts = []
for path in sorted(BASE.rglob('*')):
    if not path.is_file() or path.is_symlink() or '__pycache__' in path.parts: continue
    rel = str(path.relative_to(BASE))
    if rel in ('evidence-index.json','verification.json'): continue
    data = path.read_bytes()
    artifacts.append({'path':rel,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
manifest = {'schema':'jope.redteam-evidence-index.v1','captured_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Hashes of local audit artifacts; not independent raw-data or numerical qualification. Original source hashes and remote paths are within individual captures.','excludes':['evidence-index.json','verification.json'],'artifacts':artifacts}
(BASE/'evidence-index.json').write_text(json.dumps(manifest,indent=2)+'\n')
record('artifact manifest emitted', bool(artifacts), {'files':len(artifacts),'bytes':sum(x['bytes'] for x in artifacts)})
result={'schema':'jope.redteam-report-verification.v1','checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':all(x['passed'] for x in checks),'checks':checks,'application_tests_run':False,'fits_or_benchmarks_run':False,'scientific_objective_qualified':False,'evidence_index_sha256':hashlib.sha256((BASE/'evidence-index.json').read_bytes()).hexdigest()}
(BASE/'verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'passed':result['passed'],'checks':len(checks),'failed':[x for x in checks if not x['passed']],'indexed_files':len(artifacts)},ensure_ascii=False))
raise SystemExit(0 if result['passed'] else 1)
