import unittest
import numpy as np
from overfit import family_gain, summarize


class OverfitTests(unittest.TestCase):
    def test_equal_family_not_equal_context(self):
        packets={'a':{'utility_k':[0,1]},'b':{'utility_k':[0,1]},'c':{'utility_k':[0,-1]}}
        self.assertEqual(family_gain(['a','b','c'],[1,1,1],packets,{'a':'A','b':'A','c':'B'}),0)

    def test_paired_gap_and_full_cells(self):
        rows=[]
        for arm in ['BASE','COS']:
            for p in ['P0','P1','P2']:
                for fold in range(4):
                    for seed in [30001,30002,30003]:
                        for member in range(3):
                            gap=.25 if arm=='BASE' else .125
                            rows.append(dict(arm=arm,partition=p,fold=fold,root_seed=seed,member_index=member,train_gain=.5,inner_val_gain=.5-gap,gap=gap))
        result=summarize(rows)
        self.assertEqual(len(result['cells']),18)
        self.assertEqual(result['paired_gap_change']['mean'],-.125)
        self.assertEqual(result['arms']['BASE']['mean_gap'],.25)

    def test_missing_member_fails_closed(self):
        with self.assertRaises(AssertionError):summarize([])


if __name__=='__main__':unittest.main()
