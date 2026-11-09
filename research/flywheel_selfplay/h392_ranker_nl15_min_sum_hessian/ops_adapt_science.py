"""Apply only the H392 ranker Hessian-floor intervention to the qualified harness."""
from pathlib import Path
import ast
P=Path(__file__).parent;S=P/'source'
def edit(name,changes):
 p=S/name;t=p.read_text()
 for old,new in changes:
  assert old in t,(name,old);t=t.replace(old,new)
 p.write_text(t)
edit('depth_patch.py',[
 ('ranker reg_alpha0->0.5 keyword','ranker min_child_weight keyword (native min_sum_hessian_in_leaf)'),
 ('import ast,copy,hashlib,inspect','import ast,copy,hashlib,inspect,json\nfrom pathlib import Path\n_VALUE=json.loads((Path(__file__).parent/"hessian_value.json").read_text());assert _VALUE["status"]=="PASS"\nMINHESS=_VALUE["value"];BASE_HESS=_VALUE["baseline"];assert MINHESS>BASE_HESS'),
 (';REG_ALPHA=.5',''),("k.arg!='reg_alpha'","k.arg!='min_child_weight'"),
 ("assert 'reg_alpha' not in kw and 'max_bin' not in kw","assert 'reg_alpha' not in kw and 'max_bin' not in kw and 'min_child_weight' not in kw and 'min_sum_hessian_in_leaf' not in kw"),
 ("arg='reg_alpha',value=ast.Constant(REG_ALPHA)","arg='min_child_weight',value=ast.Constant(MINHESS)"),
 ("{'reg_alpha':[0,.5]}","{'min_child_weight':[BASE_HESS,MINHESS]}"),
 ("added_keywords=['reg_alpha']","added_keywords=['min_child_weight']"),
 ('reg_alpha=.5,veto_unchanged','reg_alpha=0,min_sum_hessian_in_leaf=MINHESS,veto_unchanged')])
edit('native_audit.py',[
 ("assert float(native['min_sum_hessian_in_leaf'])==.001\n target=(.5 if check_treatment else 0.0) if kind=='LGBMRanker' else 1.0", "from depth_patch import MINHESS,BASE_HESS\n assert float(native['min_sum_hessian_in_leaf'])==(MINHESS if check_treatment and kind=='LGBMRanker' else BASE_HESS)\n target=0.0 if kind=='LGBMRanker' else 1.0"),
 (" for key,want in constants.items():\n  target='0.5' if key=='lambda_l1' and kind=='LGBMRanker' else want\n  assert native[key]==target,(kind,key,native[key],target,'H352 constant native field drift')", " from depth_patch import MINHESS,BASE_HESS\n for key,want in constants.items():\n  if key=='min_sum_hessian_in_leaf' and kind=='LGBMRanker':\n   assert float(want)==BASE_HESS and float(native[key])==MINHESS,(kind,key,native[key],MINHESS)\n  else:assert native[key]==want,(kind,key,native[key],want,'H352 constant native field drift')"),
 ("{'lambda_l1':[0,.5]}","{'min_sum_hessian_in_leaf':[BASE_HESS,MINHESS]}")])
edit('run_h392.py',[
 ('max_bin=255/reg_alpha=0.5','max_bin=255/ranker Hessian floor'),
 ('import numpy as np','import numpy as np\nfrom depth_patch import MINHESS,BASE_HESS'),
 ("params['reg_alpha']==.5 and params['min_child_weight']==.001", "params['reg_alpha']==0 and params['min_child_weight']==MINHESS"),
 ("'lambda_l1','lambda_l2','min_child_weight','min_sum_hessian_in_leaf'", "'lambda_l1','lambda_l2','reg_alpha','min_sum_hessian_in_leaf'"),
 ("assert 'path_smooth' not in kwargs", "assert 'min_child_weight' in kwargs and 'path_smooth' not in kwargs"),
 ("{'reg_alpha':[0,.5]}","{'min_child_weight':[BASE_HESS,MINHESS]}"),
 ("p['reg_alpha']==.5 and p['min_child_weight']==.001", "p['reg_alpha']==0 and p['min_child_weight']==MINHESS")])
edit('evaluate_h392.py',[("Versus H352 rankers-only reg_alpha0->0.5 (native lambda_l1); max_bin255 retained in fresh Dataset; max_depth3/num_leaves15 fixed/realized_binary_bound8; BASEreg_lambda1/rankerreg_alpha0.5/minchild20/LambdaRank/sigmoid1/labels/defaultgain/groups/GBDT/all other args/veto unchanged", "Versus H352 rankers-only min_sum_hessian_in_leaf from archived native baseline to fixed preregistered MINHESS via min_child_weight; max_bin255/depth3/leaves15/L2one/L1zero/minchild20/LambdaRank/defaults/seeds/labels/groups/veto unchanged")])
edit('check_nl15_native.py',[("only lambda_l1 0->0.5 may change in rankers","only min_sum_hessian_in_leaf may change in rankers")])
edit('test_depth.py',[
 ('import depth_patch as p','import depth_patch as p'),
 ("if k!='reg_alpha'","if k!='min_child_weight'"),
 ("set(before)|{'reg_alpha'}","set(before)|{'min_child_weight'}"),
 ("after['reg_alpha'],ast.dump(ast.Constant(.5)","after['min_child_weight'],ast.dump(ast.Constant(p.MINHESS)"),
 ("['min_child_weight','path_smooth'","['reg_alpha','path_smooth'")])
edit('test_native_audit.py',[
 ('import native_audit as n','import native_audit as n\nfrom depth_patch import MINHESS,BASE_HESS'),
 ("reg_alpha=.5),dict(self.r['native_params'],num_leaves='15',max_depth='3',lambda_l1='0.5')", "reg_alpha=0,min_child_weight=MINHESS),dict(self.r['native_params'],num_leaves='15',max_depth='3',lambda_l1='0',min_sum_hessian_in_leaf=str(MINHESS))"),
 ("('lambda_l1','0')","('lambda_l1','0.5')"),
 ("('min_sum_hessian_in_leaf','9.87')","('min_sum_hessian_in_leaf',str(BASE_HESS))")])
edit('test_h392.py',[
 ('import run_h392 as r','import run_h392 as r\nfrom depth_patch import MINHESS'),
 ('reg_alpha=.5,min_child_samples=20','reg_alpha=0,min_child_weight=MINHESS,min_child_samples=20')])
edit('test_secondary.py',[
 ('import native_audit as n','import native_audit as n\nfrom depth_patch import MINHESS,BASE_HESS'),
 ("native['lambda_l1']='0.5'","native['lambda_l1']='0';native['min_sum_hessian_in_leaf']=str(MINHESS)"),
 ('only_L1_change','only_Hessian_floor_change'),
 ("{'lambda_l1':[0,.5]}","{'min_sum_hessian_in_leaf':[BASE_HESS,MINHESS]}"),
 ("('min_data_in_bin','9')","('min_data_in_bin','9'),('lambda_l1','0.5'),('min_sum_hessian_in_leaf',str(BASE_HESS))")])
edit('test_evaluator_report.py',[("self.assertIn('reg_alpha0->0.5'","self.assertIn('min_sum_hessian_in_leaf'")])
edit('test_dataset_audit.py',[
 ('reg_alpha=.5','reg_alpha=0,min_child_weight=MINHESS'),
 ('import unittest','from depth_patch import MINHESS\nimport unittest')])
edit('test_resource_forecast.py',[("2026,11,5,20","2026,11,6,0")])
for p in S.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
print('H392 Hessian-only source adaptation parsed; no optimizer fits')
