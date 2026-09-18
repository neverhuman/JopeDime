import json
import tempfile
import unittest
from pathlib import Path

import population_judge as pj
from resume_expired_cycle import atomic, sha

BINDING = dict(corpus_sha256='c' * 64, panel_sha256='p' * 64, executable_sha256='e' * 64)


def monitor(root, name, bt, nll=1.39, listwise=2.2, checkpoint=None):
    path = Path(root) / ('%s.json' % name)
    atomic(path, dict(schema='parallel.frozen_monitoring.v1', checkpoint='ckpt/' + name, evaluated_step=69000, timestamp=1789300000,
                      binding=dict(BINDING, checkpoint_sha256=checkpoint or (name * 16)[:64]), sequence_nll=nll, ranking_bt=bt, ranking_listwise=listwise,
                      runtime_huber=0.07, sampled_examples=2048, family_balanced=True, independent_evaluation=False, final_tests_opened=False))
    return str(path)


def baseline(root, g_bt=0.04, g_nll=0.02):
    path = Path(root) / 'baseline.json'
    atomic(path, dict(schema='parallel.flywheel_baseline.v1', metrics=dict(panel_null=dict(guards=dict(g_bt=g_bt, g_nll=g_nll), guards_borrowed=False))))
    return path


def tournament(root, arms):
    path = Path(root) / 'tournament.json'
    atomic(path, dict(schema='parallel.population_tournament.v1', contexts=16, arms={str(k): dict(wins=w, ties=t, losses=l) for k, (w, t, l) in arms.items()}, final_tests_opened=False))
    return path


class PopulationJudge(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.parent = monitor(self.root, 'parent', 0.520)
        self.baseline = baseline(self.root)

    def run_judge(self, arms, tournament_path=None, history=None):
        argv = ['--parent', self.parent, '--baseline', str(self.baseline), '--dry-run'] + [x for a in arms for x in ('--arm', a)]
        if tournament_path:
            argv += ['--tournament', str(tournament_path)]
        if history:
            atomic(self.root / 'history.json', history)
            argv += ['--history', str(self.root / 'history.json')]
        return pj.main(argv)

    def test_incumbent_eligible_is_kept_and_eta_bounds_the_margin(self):
        arms = ['0=' + monitor(self.root, 'a0', 0.518), '1=' + monitor(self.root, 'a1', 0.522), '2=' + monitor(self.root, 'a2', 0.5175)]
        judgment = self.run_judge(arms)
        self.assertEqual((judgment['outcome'], judgment['adopted_arm']), ('population_adopted', 0))
        self.assertAlmostEqual(judgment['eta'], 0.004, places=12)
        self.assertAlmostEqual(judgment['margin'], 0.01, places=12)   # 0.25 * g_bt = 0.01 > eta
        self.assertTrue(all(a['eligible'] for a in judgment['arms'].values()))
        self.assertEqual(judgment['selected_of_k'], 3)
        self.assertIs(judgment['candidate_won'], False)

    def test_variant_replaces_incumbent_only_beyond_the_margin_and_with_tournament_wins(self):
        arms = ['0=' + monitor(self.root, 'a0', 0.518), '1=' + monitor(self.root, 'a1', 0.519), '3=' + monitor(self.root, 'a3', 0.505)]
        judgment = self.run_judge(arms)
        self.assertEqual((judgment['outcome'], judgment['adopted_arm']), ('population_variant_adopted', 3))
        self.assertIn('beats the incumbent', judgment['reason'])
        blocked = self.run_judge(arms, tournament(self.root, {3: (6, 2, 8)}))
        self.assertEqual((blocked['outcome'], blocked['adopted_arm']), ('population_adopted', 0))
        allowed = self.run_judge(arms, tournament(self.root, {3: (8, 0, 8)}))
        self.assertEqual(allowed['adopted_arm'], 3)
        self.assertEqual(allowed['arms']['3']['tournament']['wins'], 8)

    def test_rescue_and_rejection(self):
        arms = ['0=' + monitor(self.root, 'a0', 0.560), '1=' + monitor(self.root, 'a1', 0.561), '4=' + monitor(self.root, 'a4', 0.530)]
        judgment = self.run_judge(arms)
        self.assertEqual((judgment['outcome'], judgment['adopted_arm']), ('population_rescue', 4))
        self.assertFalse(judgment['arms']['0']['eligible'])
        self.assertIn('ranking_bt', judgment['arms']['0']['exceeded'])
        nothing = self.run_judge(['0=' + monitor(self.root, 'b0', 0.560), '1=' + monitor(self.root, 'b1', 0.561)])
        self.assertEqual((nothing['outcome'], nothing['adopted_arm']), ('population_rejected', None))
        self.assertIn('diagnostic_rejected', nothing['reason'])

    def test_identity_changing_arm_needs_two_of_three_wins(self):
        arms = ['0=' + monitor(self.root, 'a0', 0.518), '1=' + monitor(self.root, 'a1', 0.519), '7=' + monitor(self.root, 'a7', 0.500) + ':dense_pairwise_v2+seed_value_scaled_bt_v1']
        first = self.run_judge(arms)
        self.assertEqual((first['outcome'], first['adopted_arm']), ('population_adopted', 0))   # one win is not enough
        self.assertEqual(first['arms']['7']['hysteresis'], dict(required=True, satisfied=False, wins=1, window=3, needed=2))
        history = [dict(candidate_arm=7, candidate_identity='dense_pairwise_v2+seed_value_scaled_bt_v1', candidate_won=True)]
        second = self.run_judge(arms, history=history)
        self.assertEqual((second['outcome'], second['adopted_arm']), ('population_variant_adopted', 7))
        self.assertEqual(second['arms']['7']['hysteresis']['wins'], 2)

    def test_refusals(self):
        with self.assertRaisesRegex(RuntimeError, 'arm 0'):
            self.run_judge(['2=' + monitor(self.root, 'a2', 0.51)])
        foreign = Path(self.root) / 'foreign.json'
        doc = json.loads(Path(monitor(self.root, 'f', 0.51)).read_bytes())
        doc['binding']['panel_sha256'] = 'q' * 64
        atomic(foreign, doc)
        with self.assertRaisesRegex(RuntimeError, 'different panel'):
            self.run_judge(['0=' + monitor(self.root, 'a0', 0.518), '1=' + str(foreign)])
        with self.assertRaisesRegex(RuntimeError, 'is the parent checkpoint'):
            self.run_judge(['0=' + monitor(self.root, 'same', 0.518, checkpoint=('parent' * 16)[:64])])

    def test_immutable_output(self):
        arms = ['0=' + monitor(self.root, 'a0', 0.518), '1=' + monitor(self.root, 'a1', 0.519)]
        output = self.root / 'judgments' / 'j.json'
        argv = ['--parent', self.parent, '--baseline', str(self.baseline), '--arm', arms[0], '--arm', arms[1], '--cycle', '21', '--output', str(output)]
        pj.main(argv)
        self.assertEqual(json.loads(output.read_bytes())['cycle'], 21)
        digest = sha(output)
        pj.main(argv)
        self.assertEqual(sha(output), digest)


if __name__ == '__main__':
    unittest.main()
