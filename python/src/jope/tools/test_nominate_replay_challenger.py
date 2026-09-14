import unittest
from nominate_replay_challenger import select


class NominationTests(unittest.TestCase):
    def test_failures_remain_in_panel_and_both_references_must_improve(self):
        rows=lambda values:[dict(status='ok',rmse=v) for v in values]
        scores=dict(recovery=rows([1,2]),serving=rows([.8,1.6]),a=rows([.85,1.7]),b=rows([.7,1.4]))
        winner,means,_=select(scores,['a','b'],['recovery','serving'])
        self.assertEqual(winner,'b');self.assertGreater(means['a']['serving'],1)
        scores['b'][0]=dict(status='failed',rmse=None)
        self.assertIsNone(select(scores,['a','b'],['recovery','serving'])[0])
        scores['recovery'][0]=dict(status='failed',rmse=None)
        self.assertIn('reference fit failed',select(scores,['a','b'],['recovery','serving'])[2])
