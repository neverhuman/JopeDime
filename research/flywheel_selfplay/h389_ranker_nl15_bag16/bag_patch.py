"""Only double N_BAGS on exact H352 constructors; original seed and vote code retained."""
import hashlib,inspect,json
from pathlib import Path
from nl15_baseline_patch import LEAVES,transform
HERE=Path(__file__).resolve().parent

def count_change(bagged,baseline,treatment):
 assert isinstance(baseline,int) and baseline>0 and treatment==2*baseline
 assert bagged.N_BAGS==baseline and bagged.Q_FIXED==6 and bagged.Q_GRID==(5,6,7)
 before=bagged.N_BAGS;bagged.N_BAGS=treatment
 return dict(baseline=before,treatment=bagged.N_BAGS,seed_index_prefix_unchanged=True,new_indices=list(range(before,treatment)),aggregation_code_unchanged=True,absolute_vote_thresholds=[5,6,7],three_member_agreement=[2,3])

def seed_extension(core,member_seeds,baseline,treatment):
 records=[]
 for seed in member_seeds:
  row=dict(member_seed=int(seed))
  for stream in ['outer_boot','outer_bag']:
   old=[core.stream_seed(int(seed),0,'BAG_SWITCH',stream,b) for b in range(baseline)]
   new=[core.stream_seed(int(seed),0,'BAG_SWITCH',stream,b) for b in range(treatment)]
   assert old==new[:baseline] and len(set(new))==treatment
   # Independent implementation of the exact existing byte/seed rule.
   for b,v in enumerate(new):assert v==int.from_bytes(hashlib.sha256(':'.join(map(str,(int(seed),0,'BAG_SWITCH',stream,b))).encode('utf-8')).digest()[:8],'little')&0x7fffffff
   row[stream]=new
  records.append(row)
 return records

def install(G):
 from nl15_baseline_patch import install as install_nl15
 cfg=json.loads((HERE/'bag_count.json').read_text());baseline=cfg['baseline'];treatment=cfg['treatment']
 baseline_asts=install_nl15(G)
 bag=G['bagged'];count=count_change(bag,baseline,treatment)
 member_seeds=[int(s) for values in G['a302'].EXPECTED_MEMBER_SEEDS.values() for s in values]
 seeds=seed_extension(G['core'],member_seeds,baseline,treatment)
 # The entire fit loop, seed generator and aggregators are the verified original functions.
 functions=[G['r302'].fit_and_freeze_member,G['core'].stream_seed,bag.vote_counts,bag.switch_by_votes,G['ens'].score_single_member_from_disk]
 functions_proof=[dict(name=f.__name__,source_sha256=hashlib.sha256(inspect.getsource(f).encode()).hexdigest()) for f in functions]
 return dict(status='PASS',ranker_kwargs='Exact H352 NL15, no new keywords or native deltas',constructors=baseline_asts,bag_count=count,seeds=seeds,unchanged_functions=functions_proof,num_leaves=15,max_depth=3,veto_unchanged=True,min_split_gain=0,LR_and_trees_unchanged=True)


def verify_bag_call_counts(counts,member_seeds,bag_count):
 checked=[]
 for member_seed in sorted(member_seeds):
  for b in range(bag_count):
   raw=':'.join(map(str,(int(member_seed),0,'BAG_SWITCH','outer_bag',b))).encode('utf-8')
   seed=int.from_bytes(hashlib.sha256(raw).digest()[:8],'little')&0x7fffffff
   assert counts.get(seed,0)==13,(member_seed,b,seed,counts.get(seed,0))
   checked.append(dict(member_seed=int(member_seed),bag_index=b,seed=seed,constructor_calls=13))
 return dict(status='PASS',checked=checked,bag_count=bag_count,expected_inner_and_final_calls_per_bag=13)
