"""A1 resource-only continuation: frozen scientific task, disjoint missing cells."""
from pathlib import Path
import sys,json,hashlib,os
from concurrent.futures import ProcessPoolExecutor
import run_h325 as r
P=Path(__file__).resolve().parent.parent;D=r.DATA

def inventory_hashes(td):return {str(p.relative_to(td)):r.sha(p) for p in td.rglob('*') if p.is_file()}
def main():
 assert json.loads((P/'amend-published.json').read_text())['published']
 handoff=json.loads((P/'resource-handoff.json').read_text());assert handoff['prior_cgroup_stopped'] and handoff['outer_score_markers']==0
 initial=json.loads((P/'runtime-source-manifest.json').read_text())
 for name,dig in initial['source_sha256'].items():assert r.sha(P/'source'/name)==dig,name
 assert r.sha(D/'regression.json')==handoff['regression_sha256']
 assert r.sha(P/'input_identity_validation.json')==handoff['input_identity_validation_sha256']
 r.initialize();r.install_xt();completed={tuple(x['task']) for x in handoff['completed']}
 for x in handoff['completed']:
  p,k,s=x['task'];td=D/p/f'fold_{k}'/f'root_{s}';assert inventory_hashes(td)==x['files_sha256']
 tasks=[(p,k,s) for p in r.PARTS for k in range(4) for s in r.SEEDS if (p,k,s) not in completed]
 assert tasks==[tuple(x) for x in handoff['remaining']] and len(tasks)+len(completed)==36
 assert not any((D/p/f'fold_{k}'/f'root_{s}').exists() for p,k,s in tasks)
 with ProcessPoolExecutor(max_workers=4) as pool:list(pool.map(r.task,tasks))
 for x in handoff['completed']:
  p,k,s=x['task'];td=D/p/f'fold_{k}'/f'root_{s}';assert inventory_hashes(td)==x['files_sha256']
 r.verify_locks();r.durable(D/'fit_done.json',dict(status='PASS',locks=36,completed_before_A1=len(completed),new_complete_cells=len(tasks),resource_amendment_commit=json.loads((P/'amend-published.json').read_text())['commit']))
if __name__=='__main__':main()
