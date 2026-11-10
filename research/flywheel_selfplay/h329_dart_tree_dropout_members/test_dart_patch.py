import ast
import hashlib
import subprocess
import unittest
from pathlib import Path
import numpy as np
import dart_patch as d

ROOT=Path('/home/ubuntu/JopeDime')
COMMIT='213b30ad52e1ce48927bba960ed609bd98ab4dfe'
PREFIX='research/flywheel_selfplay/h287_bagswitch_seed_ensemble_export/'


def function_source(name, fun):
    raw=subprocess.check_output(['git','-C',str(ROOT),'show',COMMIT+':'+PREFIX+name],text=True)
    node=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name==fun)
    return ast.get_source_segment(raw,node)


class DartTests(unittest.TestCase):
    def test_both_original_ranker_asts_restore_exactly(self):
        for file,fun in [('topk.py','fit_ranker'),('bagged.py','fit_ranker_n')]:
            old,new=d.transform(function_source(file,fun))
            self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(d.strip(new),include_attributes=False))
            call=next(n for n in ast.walk(new) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='LGBMRanker')
            p={k.arg:k.value for k in call.keywords}
            self.assertEqual(ast.literal_eval(p['boosting_type']),'dart')
            for k,v in d.EXTRA.items():self.assertEqual(ast.literal_eval(p[k]),v)
            self.assertEqual(ast.unparse(p['drop_seed']),'int(random_state)')
    def test_ranker_rounds_and_every_other_keyword_unchanged(self):
        for file,fun in [('topk.py','fit_ranker'),('bagged.py','fit_ranker_n')]:
            old,new=d.transform(function_source(file,fun))
            def kwargs(tree):
                c=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='LGBMRanker')
                return {k.arg:ast.dump(k.value,include_attributes=False) for k in c.keywords}
            before,after=kwargs(old),kwargs(new)
            self.assertEqual({k:v for k,v in before.items() if k!='boosting_type'}, {k:v for k,v in after.items() if k not in d.ADDED|{'boosting_type'}})
            self.assertNotIn('early_stopping_rounds',after)
    def test_unexpected_base_boosting_rejected(self):
        source=function_source('topk.py','fit_ranker').replace('boosting_type="gbdt"','boosting_type="rf"')
        with self.assertRaises(AssertionError):d.transform(source)
    def test_duplicate_dart_override_rejected(self):
        _,tree=d.transform(function_source('topk.py','fit_ranker'))
        with self.assertRaises(AssertionError):d.transform(ast.unparse(tree))


if __name__=='__main__':unittest.main()
