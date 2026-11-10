import unittest
from resource_guard import Window, DEADLINE


class GuardTests(unittest.TestCase):
    def row(
        self, w, t, cpu=20, load=1, mem=20 * 2**30, scratch=2_000_000_000, utc=None
    ):
        return w.sample(t, int(t * 100), int(t * (100 - cpu)), load, mem, scratch, utc)

    def test_no_instant_CPU_or_load_gate(self):
        w = Window(0, 0, 0)
        self.assertEqual(self.row(w, 5, cpu=100, load=120)["reason"], [])

    def test_full60_cpu_gate_strict(self):
        self.assertEqual(self.row(Window(0, 0, 0), 60, cpu=85)["reason"], [])
        self.assertTrue(self.row(Window(0, 0, 0), 60, cpu=86)["reason"])

    def test_three_full60_load_samples(self):
        w = Window(0, 0, 0)
        self.assertEqual(self.row(w, 59, load=120)["load_high_streak_60s"], 0)
        self.assertEqual(self.row(w, 60, load=109)["reason"], [])
        self.assertEqual(self.row(w, 120, load=109)["reason"], [])
        self.assertTrue(self.row(w, 180, load=109)["reason"])

    def test_load108_resets(self):
        w = Window(0, 0, 0)
        self.row(w, 60, load=109)
        self.row(w, 120, load=108)
        self.assertEqual(self.row(w, 180, load=109)["load_high_streak_60s"], 1)

    def test_immediate_mem_scratch_deadline(self):
        self.assertTrue(self.row(Window(0, 0, 0), 1, mem=20 * 2**30 - 1)["reason"])
        self.assertTrue(self.row(Window(0, 0, 0), 1, scratch=2_000_000_001)["reason"])
        self.assertTrue(self.row(Window(0, 0, 0), 1, utc=DEADLINE)["reason"])


if __name__ == "__main__":
    unittest.main()
