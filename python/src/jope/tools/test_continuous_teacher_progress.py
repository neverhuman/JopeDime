import copy
import unittest

import continuous_teacher_progress as teacher
from test_continuous_status import ContinuousStatusTests


class TeacherTests(unittest.TestCase):
    def test_duplicate_generations_ties_and_teachers_cannot_manufacture_progress(self):
        report, jobs = ContinuousStatusTests().quality()
        report.update(timestamp=100, jobs_sha256="jobs")
        for row in report["rows"]:
            row["search_sha256"] = row["output"]
        generations = [
            (dict(number=i, checkpoint="checkpoint"), copy.deepcopy(report), jobs) for i in range(3)
        ]
        result = teacher.summarize(generations, 200)
        self.assertEqual(result["new_champions"], 1)
        self.assertEqual(result["unique_improved_families"], 1)
        self.assertEqual(result["discovery_origins"], {"unavailable": 1})
        self.assertEqual(result["seconds_since_meaningful_improvement"], 100)
        for row in report["rows"]:
            row["best_final_gp"] = row["best_retained_teacher"]
        result = teacher.summarize([(dict(number=0, checkpoint="checkpoint"), report, jobs)], 200)
        self.assertEqual(result["new_champions"], 0)
        self.assertIsNone(result["last_meaningful_improvement_at"])

    def test_no_receipts_is_explicitly_no_evidence(self):
        result = teacher.summarize([], 200)
        self.assertIsNone(result["source_timestamp"])
        self.assertEqual(result["new_champions"], 0)


if __name__ == "__main__":
    unittest.main()
