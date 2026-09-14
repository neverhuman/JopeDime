import json
from pathlib import Path
import tempfile
import unittest

import population_arm_policy as pa
from resume_expired_cycle import atomic, sha


def base_policy():
    return dict(schema='parallel.replay_policy.v2', replay='fresh_balanced', optimizer='carry', sampling_seed=7527005550759306971, fresh_fraction=0.5,
                fresh_groups=[[1, 2, 3]], fresh_records=[[4, 5]], corpus_sha256='c' * 64, vocabulary_sha256='v' * 64, baseline_corpus_sha256='b' * 64,
                parent_checkpoint_sha256='p' * 64, eligible=True, final_tests_opened=False)


class PopulationArmPolicy(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name) / 'policy.json'
        atomic(self.base, base_policy())

    def test_arm_policies_change_only_seed_optimizer_replay(self):
        p0, r0 = pa.derive(self.base, 0)
        p1, r1 = pa.derive(self.base, 1)
        p2, r2 = pa.derive(self.base, 2)
        self.assertNotEqual(p0['sampling_seed'], p1['sampling_seed'])
        self.assertEqual(p2['sampling_seed'], pa.arm_seed(sha(self.base), 2, 0))   # the arm index is part of the pre-registered seed rule
        self.assertNotEqual(p0['sampling_seed'], p2['sampling_seed'])
        self.assertEqual((p2['optimizer'], p0['optimizer']), ('reset', 'carry'))
        self.assertEqual(p0['sampling_seed'], pa.arm_seed(sha(self.base), 0, 0))
        for p in (p0, p1, p2):
            self.assertEqual({k: p[k] for k in pa.MEMBERSHIP}, {k: base_policy()[k] for k in pa.MEMBERSHIP})
        self.assertEqual(r0['changed'].keys() if r0['changed'] else set(), {'sampling_seed'})
        self.assertEqual(sorted(r2['changed']), ['optimizer', 'sampling_seed'])
        self.assertEqual((r0['membership_sha256'], r1['membership_identical'], pa.derive(self.base, 3)[1]['needs_sampler_support']), (r1['membership_sha256'], True, True))
        with self.assertRaises(RuntimeError):
            pa.derive(self.base, 9)

    def test_cli_writes_policy_and_immutable_receipt(self):
        out = Path(self.tmp.name) / 'arms/policy-arm-1.json'
        receipt = Path(self.tmp.name) / 'arms/receipt-arm-1.json'
        pa.main(['--base', str(self.base), '--arm', '1', '--output', str(out), '--receipt', str(receipt)])
        digest = sha(receipt)
        pa.main(['--base', str(self.base), '--arm', '1', '--output', str(out), '--receipt', str(receipt)])
        self.assertEqual(sha(receipt), digest)
        self.assertEqual(json.loads(receipt.read_bytes())['policy']['sha256'], sha(out))


if __name__ == '__main__':
    unittest.main()
