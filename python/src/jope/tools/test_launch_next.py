import unittest
from unittest.mock import Mock

import fence_honesty as fence
import launch_next as L


class LaunchNextTests(unittest.TestCase):
    def test_refuses_legacy_units_and_pid(self):
        with self.assertRaisesRegex(ValueError, "fenced|forbidden|774452"):
            L.refuse("jope-parallel-gp.service")
        with self.assertRaisesRegex(ValueError, "fenced|forbidden"):
            L.refuse("jope-parallel-campaign")
        with self.assertRaisesRegex(ValueError, "774452"):
            L.refuse(fence.NEXT_GP, pid=774452)

    def test_renaming_a_service_cannot_bypass_authority(self):
        runner = Mock()
        for unit in (fence.NEXT_GP, fence.NEXT_CONTROLLER):
            with self.assertRaisesRegex(ValueError, "native namespace activation"):
                L.start(unit, runner=runner)
        runner.assert_not_called()
        with self.assertRaisesRegex(ValueError, "only"):
            L.refuse("jope-parallel-monitor.service")

    def test_stops_resident_only_without_dispatcher(self):
        runner = Mock()
        self.assertFalse(L.stop_resident_if_no_dispatcher(True, runner=runner)["stopped"])
        runner.assert_not_called()
        self.assertFalse(L.stop_resident_if_no_dispatcher(False, runner=runner)["stopped"])
        runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
