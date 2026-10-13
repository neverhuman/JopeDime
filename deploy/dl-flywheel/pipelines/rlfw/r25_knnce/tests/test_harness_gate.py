"""R25 harness v7 refuses configs it was not written for, before it creates any output (no data, no torch)."""
import json
import tempfile
import unittest
from pathlib import Path

from rlfw.r25_knnce import lpi_scratch_train_v7 as H


class HarnessGateTest(unittest.TestCase):
    def _main_refuses(self, body):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / 'config.json'
            cfg.write_text(json.dumps(body))
            out = Path(tmp) / 'run'
            with self.assertRaises(ValueError):
                H.main(str(cfg), str(out))
            self.assertFalse(out.exists(), 'a refused config must not create the run directory')

    def test_revision_6_config_is_refused(self):
        self._main_refuses({'schema': 'jope.r22-lpi-scratch-config/v6', 'wall_seconds': 60,
                            'cuda_memory_fraction': 0.4, 'max_tokens_per_chunk': 16384})

    def test_unbounded_wall_is_refused(self):
        self._main_refuses({'schema': 'jope.r22-lpi-scratch-config/v7', 'wall_seconds': 7021,
                            'cuda_memory_fraction': 0.4, 'max_tokens_per_chunk': 16384})

    def test_pinned_r25_config_is_the_signed_off_arm(self):
        cfg = json.loads((Path(H.__file__).parent / 'configs' / 'r25.json').read_text())
        self.assertEqual(cfg['schema'], 'jope.r22-lpi-scratch-config/v7')
        self.assertEqual(cfg['arms'], [['r25_knnce', 0.0, 0.2, 0.25]])
        self.assertEqual(cfg['knn'], {'k': 4, 'metric': 'cosine_frozen_normalised_encoding',
                                      'expect_table_digest': 'e9feb0ce846dda2ad2cb6a0143601b3be24cfc56496fc20a'
                                                             '5709cd5c33f0fcc3'})
        self.assertEqual((cfg['max_updates'], cfg['max_epochs'], cfg['learning_rate'], cfg['batch_contexts']),
                         (1408, 4, 3.75e-5, 16))  # option A, ENG:watcher OWNER DECISION 2026-10-13T08:23:05Z
        self.assertTrue(cfg['resume_v1_moments'])
        self.assertEqual(cfg['l6_roster'], 'full_minus_heldout_lineages')
        self.assertEqual((cfg['kl_alarm'], cfg['kl_disqualify']), (0.03, 0.05))


if __name__ == '__main__':
    unittest.main()
