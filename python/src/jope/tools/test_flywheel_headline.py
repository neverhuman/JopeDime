import json
import os
from pathlib import Path
import tempfile
import time
import unittest

import flywheel_headline as headline
from resume_expired_cycle import atomic, sha


def generation_row(number, wins=10, losses=6):
    block = dict(wins=wins, ties=0, losses=losses, mean_log_ratio=-0.02, ci90=[-0.05, 0.01])
    return dict(schema='parallel.flywheel_series.v1', kind='generation', id='generation-%06d' % number, generation=number, timestamp=1789300000 + number,
                checkpoint_sha256='c' * 64, report_sha256='r' * 64, contexts=140,
                matched=dict(val=dict(seed_only=dict(block, vs_shuffled=dict(wins=9, ties=1, losses=6)), trailing10=dict(block, wins=100, losses=60, generations=[number - 9, number])),
                             train=dict(seed_only=dict(wins=120, ties=0, losses=4, mean_log_ratio=-0.3), final=dict(wins=1, ties=123, losses=0))),
                best_initial_arm_derived=dict(counts={'observations_ledger': 420}, rule='derived'), final_tests_opened=False)


def monitor_row(step):
    return dict(schema='parallel.flywheel_series.v1', kind='monitor', id='m%d' % step, evaluated_step=step, timestamp=1789290000 + step, checkpoint_sha256='k%d' % step,
                panel=dict(sequence_nll=1.39, ranking_bt=0.51 + step * 1e-6, ranking_listwise=2.2, runtime_huber=0.07, nontie_pairwise_accuracy=None),
                baseline_delta=dict(ranking_bt=0.01), lineage='learning')


class Headline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'parallel'
        self.run = self.root / 'runs/run'
        (self.run / 'learning').mkdir(parents=True)
        atomic(self.root / 'manifest.json', dict(run_identity='fixture', cpu_executable='bin/gp', bindings={'bin/gp': 'e' * 64}, vocabulary_sha256='v' * 64))
        atomic(self.root / 'baseline.json', dict(schema='parallel.flywheel_baseline.v1', metrics=dict(matched_heldout=dict(seed_only=dict(wins=60, ties=0, losses=100, mean_log_ratio=0.02, ci90=[-0.015, 0.055]), mde=0.077),
                                                                                                  matched_train=dict(seed_only=dict(wins=1230, ties=0, losses=10)))))
        self.series = self.run / 'successor/scoreboard/flywheel-series.jsonl'
        self.series.parent.mkdir(parents=True)
        with open(self.series, 'w') as stream:
            for row in [monitor_row(3000), monitor_row(6000), generation_row(30), generation_row(31, wins=7, losses=9), generation_row(31, wins=8, losses=8)]:
                stream.write(json.dumps(row) + '\n')
        atomic(self.run / 'learning/progress.json', dict(phase='running', automatic_training=False, training_diagnostic=dict(held=True), training_hold_reason='regression',
                                                         completed_training_blocks=21, training_growth=dict(eligible=False), last_committed_checkpoint='l/step-66000',
                                                         serving_checkpoint='s/step-3000', serving_status='provisional_unqualified', gpu_occupancy=dict(task_occupancy=0.03)))
        store = self.root / 'runs/champions'
        for name, partition, evaluator, ages in (('a', 'meta_train', 'e' * 64, (100, 5000)), ('b', 'meta_validation', 'e' * 64, (100000,)), ('c', 'meta_test', 'e' * 64, (10,)), ('d', 'meta_train', 'f' * 64, (10,))):
            directory = store / name
            directory.mkdir(parents=True)
            atomic(directory / 'current.json', dict(identity=dict(partition=partition, evaluator_sha256=evaluator)))
            for i, age in enumerate(ages):
                path = directory / ('event-%s.json' % (str(i) * 64))
                atomic(path, dict(x=i))
                os.utime(path, (time.time() - age, time.time() - age))

    def test_receipt_rows_sources_and_claims(self):
        receipt = headline.build(self.root, self.run, self.root / 'manifest.json', self.root / 'baseline.json', self.series, 31)
        rows = receipt['rows']
        self.assertEqual(rows['primary_heldout']['seed_only']['wins'], 8)   # the newest row for generation 31 wins
        self.assertEqual(rows['primary_heldout']['trailing10']['n_pairs'], 160)
        self.assertEqual(rows['primary_heldout']['baseline']['wins'], 60)
        self.assertEqual(rows['primary_heldout']['mde'], 0.077)
        self.assertIn('NOT held out', rows['training_win_rate']['label'])
        self.assertAlmostEqual(rows['training_win_rate']['seed_only']['win_rate'], 120 / 124)
        self.assertEqual([m['evaluated_step'] for m in rows['frozen_panel']], [3000, 6000])
        self.assertEqual((rows['high_marks']['events_24h'], rows['high_marks']['events_1h'], rows['high_marks']['identities'], rows['high_marks']['all_time']), (2, 1, 2, 3))
        self.assertEqual(rows['high_marks']['by_partition'], dict(meta_train=2, meta_validation=0))   # meta_test and foreign evaluators never counted
        self.assertEqual((rows['learning']['hold'], rows['learning']['completed_training_blocks']), (True, 21))
        self.assertEqual(rows['publication']['status'], 'no_repository')
        self.assertEqual(receipt['claims']['independent_improvement'], 'not_established')
        self.assertIs(receipt['final_tests_opened'], False)
        with self.assertRaises(RuntimeError):
            headline.build(self.root, self.run, self.root / 'manifest.json', self.root / 'baseline.json', self.series, 99)

    def test_cli_writes_immutable_receipt_and_latest(self):
        output = self.run / 'successor/scoreboard/headline'
        latest = self.run / 'successor/scoreboard/latest.json'
        argv = ['--root', str(self.root), '--run', 'runs/run', '--manifest', str(self.root / 'manifest.json'), '--baseline', str(self.root / 'baseline.json'), '--series', str(self.series),
                '--output', str(output), '--latest', str(latest)]
        first = headline.main(argv)
        path = output / 'generation-000031.json'
        self.assertEqual(json.loads(path.read_bytes())['generation'], 31)
        digest = sha(path)
        self.assertEqual(json.loads(latest.read_bytes())['schema'], headline.LATEST_SCHEMA)
        self.assertEqual(json.loads(latest.read_bytes())['receipt']['sha256'], digest)
        # the receipt embeds a generation timestamp, so a re-run must not change the immutable file: it refuses instead
        with self.assertRaises(RuntimeError):
            headline.main(argv)
        self.assertEqual(sha(path), digest)
        verdict = self.run / 'verdict.json'
        atomic(verdict, dict(schema='parallel.flywheel_verdict.v1', outcome='smarter_confirmed'))
        confirmed = headline.build(self.root, self.run, self.root / 'manifest.json', self.root / 'baseline.json', self.series, 30, verdict)
        self.assertTrue(confirmed['claims']['independent_improvement'].startswith('established_by_verdict:'))


if __name__ == '__main__':
    unittest.main()
