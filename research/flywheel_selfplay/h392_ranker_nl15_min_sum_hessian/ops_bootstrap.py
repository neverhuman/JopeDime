"""Recover committed H383's frozen harness into this owned H392 custody tree."""
from pathlib import Path
import hashlib,json,subprocess,shutil,ast
P=Path(__file__).parent;O=P.with_name('h383-solhwv');R=Path('/home/ubuntu/JopeDime');S=P/'source';T=P/'template'
REF='ca89ada6b78bf91439118d2d330de4385a5bde15';PRE='research/flywheel_selfplay/h383_ranker_nl15_reg_alpha_05_xb2/'
def adapt(t):
 return t.replace('h383_ranker_nl15_reg_alpha_05_xb2','h392_ranker_nl15_min_sum_hessian').replace('h383-solhwv','h392-solhwv').replace('h383-hwv','h392-hwv').replace('383191','392191').replace('H348-H383','H348-H392').replace('H348–H383','H348–H392').replace('H383','H392').replace('h383','h392').replace('L1A05','MINHESS').replace('l1a05','minhess').replace('hwv-xb2-20261105T170500Z','hwv-20261105T203000Z').replace('20261105T170500Z','20261105T203000Z').replace('hwv-xb2-20261105-t1705','hwv-20261105-t2030').replace('2026,11,5,20,45','2026,11,6,0,45').replace('2026,11,5,21,0','2026,11,6,1,0').replace('2026-11-05T20:45:00','2026-11-06T00:45:00').replace('2026-11-05T21:00:00','2026-11-06T01:00:00').replace('rd/h392-solhwv','rd/h392-hwv')
S.mkdir();T.mkdir();frozen=json.loads((O/'runtime-source-manifest.json').read_text());inventory={}
for name,h in frozen.items():
 raw=subprocess.check_output(['git','--no-optional-locks','show',REF+':'+PRE+name],cwd=R);assert hashlib.sha256(raw).hexdigest()==h,name
 (T/name).write_bytes(raw);inventory[name]=h
 if name in ('PREREG.md','manifest.json','test_wait_gate.py'):continue
 (S/name.replace('383','392')).write_text(adapt(raw.decode()))
shutil.copytree(O/'vendor',P/'vendor')
for n in ['prepare_inputs.py','stage_inputs.py','fetch_nl15.py','launch_prepare.py','prepare.py','launch.py','publish.py','gitcredential.py','reference_custody.py','reclaim_reference.py','close_archive.py','launch_archive.py','archive_guard.py','post_archive.py','pre_archive.py','reclaim.py']:
 (P/n).write_text(adapt((O/n).read_text()))
m=json.loads((T/'manifest.json').read_text());m.update(wave='H392',authority='/home/ubuntu/nudges/hw_t2030_hwv_h392_nl15_minhess.md',compute_stop='2026-11-06T00:45:00Z',close_stop='2026-11-06T01:00:00Z',paired_bootstrap_rng=392191,reg_alpha=0.0,harness_lineage='H383 qualified harness; only ranker native min_sum_hessian_in_leaf differs from H352 NL15',source_H383_commit=REF,guard='MemAvail20GiB/scratch2GB;CPU>85avg60/load>108three60;fixed00:45ZNov6',intervention='ranker Hessian floor from archived H352 ranker leaf_weight Q25 rounded3sigfig; one value before prereg',core_pools={'H392_UNION':m['cores']},start_condition='ONE fresh union CPU admission after preparation/prereg, no foreign-unit dependency')
m.pop('source_sha256',None);m.pop('prereg_sha256',None);(S/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
c=S/'controller_h392.py';t=c.read_text();a=t.index('def wait_for_h377():');b=t.index('def main():',a);t=t[:a]+t[b:];a=t.index(' if not wait_for_h377():');b=t.index(' eligible,idle,raw,host=probe()',a);t=t[:a]+t[b:];t=t.replace('assert all(v==eligible for v in after.values())','assert all(set(v)==set(eligible) for v in after.values())');c.write_text(t)
(P/'template-copy.json').write_text(json.dumps(dict(commit=REF,source=PRE,files=inventory,science='Only MINHESS arm substituted; single-parameter inverse and native checks mandatory'),indent=2)+'\n')
for p in S.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
print('Recovered',len(inventory),'frozen source blobs from',REF,'without changing canonical Git state')
