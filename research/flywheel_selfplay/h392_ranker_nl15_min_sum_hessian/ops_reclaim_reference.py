"""Rehash and prune only H392-owned historical copies after independent S3 proof."""
from pathlib import Path
from collections import Counter
import sys,json,hashlib,subprocess,datetime
P=Path(__file__).parent;D=Path('/home/ubuntu/jope-data/h392-hwv');sys.path.insert(0,str(P/'source'));from native_audit import parse
from tree_structure import parse_trees,validate
r=json.loads((P/'reference-archive-verification.json').read_text());assert r['status']=='PASS' and r['record']['all_members_exact_version_verified'] and r['record']['exact_version_get_sha256_verified'];assert json.loads((P/'reference-archive-unit-exit.json').read_text())['EXIT_STATUS']=='0'
assert subprocess.run(['systemctl','--user','is-active','h392-run'],capture_output=True,text=True).stdout.strip() not in ('active','activating')
assert not list(D.glob('P*/fold_*/root_*/fit_lock.json'))
profiles={}
for name,root,leaves in [('BASE',D/'base_native',7),('NL15',D/'nl15',15)]:
 depths=Counter();counts=Counter();models=0
 for p in sorted(root.glob('P*/fold_*/root_*/refit/*.txt')):
  if p.name.endswith('-veto.txt'):continue
  native=parse(p.read_text());assert native['max_bin']=='255' and float(native['lambda_l1'])==0
  trees=validate(parse_trees(p.read_text()),3,leaves);depths.update(t[0] for t in trees);counts.update(t[1] for t in trees);models+=1
 assert models==972;profiles[name]=dict(models=models,trees=sum(depths.values()),depth_histogram={str(k):v for k,v in sorted(depths.items())},leaf_histogram={str(k):v for k,v in sorted(counts.items())},max_actual_depth=max(depths),max_actual_leaves=max(counts))
(P/'reference-tree-summary.json').write_text(json.dumps(dict(status='PASS',profiles=profiles,new_fits=0,new_predictions=0,source_archive_version=r['record']['version_id']),indent=2)+'\n')
for n,v in r['members'].items():
 p=D/n;assert p.is_file() and not p.is_symlink() and p.stat().st_size==v['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256'],n
for n in r['members']:(D/n).unlink()
record=dict(status='PASS',host='xbabe2',files=len(r['members']),bytes=sum(v['bytes'] for v in r['members'].values()),archive=r['record'],sha_all_reverified_before_delete=True,only_validated_historical_native_copies=True,current_treatment_data_removed=False,retained='All comparator arrays/locks/scores/TRAINinputs/source/nativeconfigJSON',utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(P/'reference-copy-reclamation.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(dict(status='PASS',files=record['files'],bytes=record['bytes'])))
