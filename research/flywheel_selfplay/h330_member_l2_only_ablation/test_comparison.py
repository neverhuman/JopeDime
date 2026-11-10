import copy,json,unittest
from comparison import compare

def fixture():
    cells=[dict(partition='P'+str(i//3),root_seed=30001+i%3,family_mean_gain=.1,harmed_rate=.1,drop_top={'10':.01},elo=1700) for i in range(9)]
    aggregate=dict(median_gain=.1,ci95=[0,.2],median_harmed_rate=.1,median_elo=1700,drop_top={'10':.01})
    arm=dict(cells=cells,aggregate=aggregate)
    paired=dict(median=.003,ci95=[-.01,.02],lcb95=-.005,per_seed_medians=[.001,.002,.003])
    old=dict(wave_id='H327',verdict='REG_NOT_ADVANCED',arms={'BASE':copy.deepcopy(arm),'REG':copy.deepcopy(arm)},paired_drop_top10=paired,member_overfit_diagnostic={'arms':{'REG':{'mean_gap':.08}}})
    return old,{'BASE':copy.deepcopy(arm),'L2ONLY':copy.deepcopy(arm)},paired,{'arms':{'L2ONLY':{'mean_gap':.07}}}

class ComparisonTests(unittest.TestCase):
    def test_descriptive_only_and_inputs_unchanged(self):
        args=fixture();before=json.dumps(args,sort_keys=True);out=compare(*args)
        self.assertEqual(json.dumps(args,sort_keys=True),before)
        self.assertEqual(out['scope'],'DESCRIPTIVE_ONLY_NO_EXTRA_DECISION')
        self.assertEqual(out['historical_verdict'],'REG_NOT_ADVANCED')
        self.assertEqual(len(out['cells']),9)
        self.assertNotIn('verdict',out)
        self.assertNotIn('criteria',out)

    def test_baseline_or_cell_identity_mismatch_rejected(self):
        for part in ['BASE','L2ONLY']:
            args=fixture();args[1][part]['cells'][0]['root_seed']=99
            with self.assertRaises(AssertionError):compare(*args)

if __name__=='__main__':unittest.main()
