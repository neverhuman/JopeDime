import unittest

import fence_honesty as f


class FenceHonestyTests(unittest.TestCase):
    def test_refuses_restart_774452(self):
        with self.assertRaisesRegex(ValueError, "774452"):
            f.refuse_legacy_restart("restart", pid=774452)
        with self.assertRaisesRegex(ValueError, "forbidden"):
            f.refuse_legacy_restart("unfence")
        self.assertTrue(f.refuse_legacy_restart("observe"))
        with self.assertRaisesRegex(ValueError, "fenced"):
            f.refuse_legacy_restart("start", unit="jope-parallel-gp.service")

    def test_idle_gpus_do_not_authorize_train(self):
        self.assertFalse(f.idle_gpus_authorize_train(0, True))
        self.assertFalse(f.idle_gpus_authorize_train(90, False))

    def test_planned_migration_is_not_a_104_crash(self):
        status = f.inspect_fence(
            dict(
                legacy_dispatch_fenced=True, historical_import_complete=False, registry_dispatch_enabled=False
            ),
            override_exists=True,
            handoff_exists=False,
            gp_main_pid=0,
            restart="no",
        )
        self.assertEqual(status["phase"], "planned_migration")
        self.assertTrue(status["verified"])
        self.assertFalse(status["historical_import_complete"])
        self.assertFalse(status["restart_legacy_gp"])
        self.assertTrue(status["historical_104_is_not_a_crash"])

    def test_handoff_present_or_live_pid_is_unverified(self):
        self.assertFalse(
            f.inspect_fence(
                dict(
                    legacy_dispatch_fenced=True,
                    historical_import_complete=False,
                    registry_dispatch_enabled=False,
                ),
                override_exists=True,
                handoff_exists=True,
            )["verified"]
        )
        self.assertFalse(
            f.inspect_fence(
                dict(
                    legacy_dispatch_fenced=True,
                    historical_import_complete=False,
                    registry_dispatch_enabled=False,
                ),
                override_exists=True,
                handoff_exists=False,
                gp_main_pid=774452,
            )["verified"]
        )


if __name__ == "__main__":
    unittest.main()
