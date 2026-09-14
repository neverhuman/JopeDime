import unittest

import useful_hour as u


class UsefulHourTests(unittest.TestCase):
    def test_training_and_monitoring_are_classified_without_fabricating_proof(self):
        for flags, kind in ((dict(train_active=True), 'train'), (dict(monitor_active=True), 'monitor')):
            row = u.classify(False, False, False, False, **flags)
            self.assertEqual(row['kind'], kind)
            self.assertFalse(row['verified_useful'])
            evidence = u.join({}, 10, 8, True, kind, [{'verified': True}], 99)
            self.assertIsNone(evidence['verified_useful_gpu_seconds'])
            self.assertEqual(evidence['acceptance'], 'inconclusive')
            self.assertTrue(evidence['missing_evidence'])

    def test_cpu_pack_and_search_do_not_imply_useful_gpu_work(self):
        self.assertFalse(u.classify(False, True, False, False)['useful'])
        self.assertFalse(u.classify(False, False, True, False)['useful'])

    def test_vram_residency_without_outputs_cannot_pass(self):
        self.assertEqual(u.classify(False, False, False, False)["kind"], "unexplained")
        history = {}
        row = u.join(history, 0, 8, True, "unexplained", [], 0)
        row = u.join(history, 3600, 8, True, "unexplained", [], 0)
        self.assertTrue(row["vram_residency_is_not_occupancy"])
        self.assertNotEqual(row["acceptance"], "pass")

    def test_declared_output_path_is_not_committed_evidence(self):
        kind = u.classify(True, False, False, False, 80)
        self.assertEqual(kind["kind"], "propose")
        self.assertTrue(kind["useful"])
        history = {}
        row = u.join(history, 10, 8, True, "propose", ["complete.json"], 80)
        self.assertFalse(row["committed_outputs"])
        self.assertEqual(row["declared_outputs"], ["complete.json"])
        self.assertEqual(row["acceptance"], "inconclusive")

    def test_fabricated_full_hour_and_receipt_boolean_cannot_qualify(self):
        history = {}
        for t in range(0, 3601, 10):
            row = u.join(history, t, 8, True, "propose", [{"verified": True}], 100)
        self.assertFalse(row["committed_outputs"])
        self.assertEqual(row["observed_gpu_task_seconds"], 0)
        self.assertNotEqual(row["acceptance"], "pass")


if __name__ == "__main__":
    unittest.main()
