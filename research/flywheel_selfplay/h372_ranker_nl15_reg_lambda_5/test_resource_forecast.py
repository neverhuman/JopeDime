import datetime, unittest
import resource_probe as p
import forecast_h372 as f


class ResourceTests(unittest.TestCase):
    def cap(self, n):
        return dict(eligible_cores=list(p.POOLS["A"][:n]))

    def report(self):
        return {
            "sysstat": {
                "hosts": [
                    {
                        "statistics": [
                            {
                                "cpu-load": [
                                    {"cpu": str(c), "idle": 90.0}
                                    for c in (
                                        *p.ASSIGNED,
                                        *(x + 64 for x in p.ASSIGNED),
                                    )
                                ]
                            }
                        ]
                    }
                ]
            }
        }

    def test_pool_A_only(self):
        self.assertEqual(p.POOLS, {"A": tuple(range(15))})

    def test_first8(self):
        self.assertEqual(p.eligible_from_report(self.report())[0], list(range(8)))

    def test_no_second_pool(self):
        r = self.report()
        for x in r["sysstat"]["hosts"][0]["statistics"][0]["cpu-load"]:
            x["idle"] = 84.99
        self.assertEqual(p.selection(r)[0], [])

    def test_exact85_inclusive(self):
        r = self.report()
        for x in r["sysstat"]["hosts"][0]["statistics"][0]["cpu-load"]:
            x["idle"] = 85
        self.assertEqual(p.selection(r)[0], list(range(8)))

    def test_second_interval_rejected(self):
        r = self.report()
        r["sysstat"]["hosts"][0]["statistics"].append({})
        with self.assertRaises(AssertionError):
            p.eligible_from_report(r)

    def test_CPU70_min4_foreign_rejected(self):
        self.assertTrue(p.admission_pass([0, 1, 2, 3], 70))
        self.assertFalse(p.admission_pass([0, 1, 2], 10))
        self.assertFalse(p.admission_pass([0, 1, 2, 3], 70.01))
        self.assertFalse(p.admission_pass([0, 1, 27, 28], 1))

    def test_H352_reference_and_deadline(self):
        early = datetime.datetime(2026, 11, 5, 12, tzinfo=datetime.timezone.utc)
        late = datetime.datetime(2026, 11, 5, 16, 30, tzinfo=datetime.timezone.utc)
        self.assertEqual(f.cpu_reference(), 26189.516914)
        self.assertFalse(f.estimate(self.cap(3), early)["launch_full_36"])
        self.assertTrue(f.estimate(self.cap(4), early)["launch_full_36"])
        self.assertFalse(f.estimate(self.cap(8), late)["launch_full_36"])

    def test_first_task_max_reference_zero_ongoing_credit(self):
        t = datetime.datetime(2026, 11, 5, 12, tzinfo=datetime.timezone.utc)
        r = f.after_first_cell(dict(cpu_seconds=1000), 8, t)
        self.assertEqual(r["per_task_cpu_reference_seconds"], 1000)
        self.assertEqual(r["remaining_tasks"], 35)
        self.assertEqual(r["forecast_seconds"], 35 * 1000 / (8 * 0.9) * 1.15 + 300)

    def test_first_task_late_rejects(self):
        self.assertFalse(
            f.after_first_cell(
                dict(cpu_seconds=1),
                8,
                datetime.datetime(2026, 11, 5, 16, 30, tzinfo=datetime.timezone.utc),
            )["launch_full_36"]
        )


if __name__ == "__main__":
    unittest.main()
