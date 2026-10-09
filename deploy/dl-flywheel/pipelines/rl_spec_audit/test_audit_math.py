"""Finite-action estimator checks; no trained model, private data or services."""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import audit_math as a


class MathAuditTests(unittest.TestCase):
    def setUp(self):
        self.z = [-.4, .8, -.1]
        self.p = a.softmax(self.z)
        self.q = [-.7, 1.2, .3]
        self.h = .2

    def marginal(self, count):
        _, gradient = a.enumerated(self.p, self.q, self.h, count)
        self.assertLess(a.error(gradient, a.finite(self.z, self.q, self.h, count)), 1e-9)
        _, averaged = a.enumerated(self.p, self.q, self.h, count, average=True)
        self.assertLess(a.error(averaged, [v / count for v in gradient]), 1e-12)

    def test_marginal_best_of_one(self):
        self.marginal(1)

    def test_marginal_best_of_two(self):
        self.marginal(2)

    def test_marginal_best_of_four(self):
        self.marginal(4)

    def dense(self, count):
        mean = sum(p * q for p, q in zip(self.p, self.q))
        expected = [p * (q - mean) for p, q in zip(self.p, self.q)]
        _, measured = a.enumerated(self.p, self.q, self.h, count, average=True, dense=True)
        self.assertLess(a.error(measured, expected), 1e-12)

    def test_dense_rloo_two(self):
        self.dense(2)

    def test_dense_rloo_four(self):
        self.dense(4)

    def test_stale_best_of_group_needs_joint_not_item_weight(self):
        behavior = a.softmax([1., -.3, .4])
        _, target = a.enumerated(self.p, self.q, self.h, 4)
        _, item = a.enumerated(self.p, self.q, self.h, 4, mu=behavior, weight='item')
        _, joint = a.enumerated(self.p, self.q, self.h, 4, mu=behavior, weight='joint')
        self.assertGreater(a.error(item, target), .01)
        self.assertLess(a.error(joint, target), 1e-12)

    def test_raw_incumbent_and_clipped_sample_can_erase_gradient(self):
        values = [0., 8., 10.]
        cap = math.log(1000)
        _, raw = a.enumerated(self.p, values, 8.5, 4)
        _, mixed = a.enumerated(self.p, [min(cap, v) for v in values], 8.5, 4)
        self.assertGreater(max(abs(v) for v in raw), 0.)
        self.assertEqual(mixed, [0., 0., 0.])

    def test_conditional_quality_is_not_unconditional_failure_utility(self):
        p = [.4, .6]
        values = [1., -1.]
        mean = sum(x * y for x, y in zip(p, values))
        gradient = [p[i] * (values[i] - mean) for i in range(2)]
        self.assertEqual(gradient, [.48, -.48])
        # Conditioning on the only successful action gives constant quality.
        self.assertNotEqual(gradient, [0., 0.])
        cases = a.audit_cases()
        self.assertEqual(len(cases), 8)
        self.assertEqual(cases[-1]['utility_gradient'], gradient)

    def test_cli_portable_and_create_only_matches_retained_numerical_cases(self):
        source = Path(a.__file__).resolve()
        historical = json.loads((source.parent / 'provenance/math-audit.json').read_bytes())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'audit.json'
            command = [sys.executable, '-B', str(source), '--output', str(path)]
            result = subprocess.run(command, cwd=tmp, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            raw = path.read_bytes()
            recorded = json.loads(raw)
            def same_cases(actual, expected):
                if isinstance(expected, dict):
                    self.assertEqual(set(actual), set(expected))
                    for key in expected:
                        same_cases(actual[key], expected[key])
                elif isinstance(expected, list):
                    self.assertEqual(len(actual), len(expected))
                    for left, right in zip(actual, expected):
                        same_cases(left, right)
                elif isinstance(expected, float):
                    self.assertAlmostEqual(actual, expected, delta=1e-9)
                else:
                    self.assertEqual(actual, expected)
            same_cases(recorded['tests'], historical['tests'])
            self.assertEqual(recorded['attachment_sha256'], historical['attachment_sha256'])
            self.assertIn('not read or reverified', recorded['attachment_provenance'])
            self.assertEqual((recorded['optimizer_updates'], recorded['GP_executions'], recorded['cloud_requests']), (0, 0, 0))
            repeat = subprocess.run(command, cwd=tmp, capture_output=True, text=True, timeout=20)
            self.assertNotEqual(repeat.returncode, 0)
            self.assertEqual(path.read_bytes(), raw)


if __name__ == '__main__':
    unittest.main()
