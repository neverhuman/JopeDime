from pathlib import Path
from collections import Counter
import json,hashlib,subprocess,shlex,sys,datetime
P=Path(__file__).parent;D=Path('/home/ubuntu/jope-data/h368-x1');r=json.loads((P/'reference-archive-verification.json').read_text());assert r['status']=='PASS' and r['record']['all_members_exact_version_verified'] and r['record']['exact_version_get_sha256_verified']
assert json.loads((P/'reference-archive-unit-exit.json').read_text())['EXIT_STATUS']=='0'
sys.path.insert(0,str(P/'source'));from tree_structure import parse_trees,validate,summarize
from native_audit import parse
profiles={}
for name,root in [('BASE',D/'base_native'),('NL15',D/'nl15')]:
 depths=Counter();leaves=Counter();models=0
 for p in sorted(root.glob('P*/fold_*/root_*/refit/*.txt')):
  if p.name.endswith('-veto.txt'):continue
  native=parse(p.read_text());trees=validate(parse_trees(p.read_text()),3,7 if name=='BASE' else 15);depths.update(t[0] for t in trees);leaves.update(t[1] for t in trees);models+=1
 assert models==972;profiles[name]=dict(models=models,trees=sum(depths.values()),depth_histogram={str(k):v for k,v in sorted(depths.items())},leaf_histogram={str(k):v for k,v in sorted(leaves.items())},max_actual_depth=max(depths),max_actual_leaves=max(leaves))
(P/'reference-tree-summary.json').write_text(json.dumps(dict(status='PASS',profiles=profiles,new_fits=0,new_predictions=0,source_archive_version=r['record']['version_id']),indent=2)+'\n')
code='''from pathlib import Path
import json,hashlib,subprocess,sys,datetime
D=Path('/home/ubuntu/jope-data/h368-x1');r=json.load(sys.stdin);P=D/'control';assert D.resolve()==D
units={n:subprocess.run(['systemctl','--user','is-active',n],capture_output=True,text=True).stdout.strip() for n in ['h368-x1-prepare','h368-x1-run']};assert all(s not in ('active','activating','deactivating') for s in units.values())
assert json.loads((P/'prepare-unit-exit.json').read_text())['EXIT_STATUS']=='0'
for n,v in r['members'].items():
 p=D/n;assert p.is_file() and not p.is_symlink();h=hashlib.sha256(p.read_bytes()).hexdigest();assert h==v['sha256'] and p.stat().st_size==v['bytes'],n
for n in r['members']:(D/n).unlink()
record=dict(status='PASS',host='xbabe1',files=len(r['members']),bytes=sum(v['bytes'] for v in r['members'].values()),archive=r['record'],sha_all_reverified_before_delete=True,only_validated_historical_native_copies=True,retained='All arrays/fitlocks/scoremarkers/TRAINinputs/source/nativeconfigJSON; currenttreatment untouched',own_units=units,utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(P/'reference-copy-reclamation.json').write_text(json.dumps(record,indent=2)+'\\n');print(json.dumps(record))
'''
remote=json.loads(subprocess.check_output(['ssh','-T','-o','BatchMode=yes','xbabe1','nice -n 19 taskset -c 35 python3 -c '+shlex.quote(code)],input=json.dumps(r).encode()));assert remote['status']=='PASS'
for n,v in r['members'].items():
 p=D/n;assert p.is_file() and not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256'],n
for n in r['members']:(D/n).unlink()
record=dict(status='PASS',remote=remote,local_files_removed=len(r['members']),local_bytes_removed=sum(v['bytes'] for v in r['members'].values()),archive=r['record'],current_treatment_fits=0,current_treatment_data_removed=False,utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(P/'reference-copy-reclamation.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(dict(reclaimed_each_host=record['local_bytes_removed'],files_each=record['local_files_removed'],source_recovery_version=r['record']['version_id'])))
