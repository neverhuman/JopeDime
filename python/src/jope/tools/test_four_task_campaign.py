import unittest

import four_task_campaign as c


class FourTaskCampaignTests(unittest.TestCase):
    def test_binary_cannot_inherit_recovery(self):
        inv = c.inventory(["a", "b"], "binary")
        with self.assertRaisesRegex(ValueError, "recovery"):
            c.isolate("binary", inv["sha256"], parent_campaign=c.RECOVERY_CAMPAIGN)
        with self.assertRaisesRegex(ValueError, "recovery"):
            c.isolate("binary", inv["sha256"], parent_checkpoint_sha=c.RECOVERY_SHA)

    def test_campaigns_have_distinct_identities(self):
        ids = [c.isolate(task, "inv")["campaign_id"] for task in c.TASKS]
        self.assertEqual(len(set(ids)), 4)
        for task in c.TASKS:
            camp = c.isolate(task, "inv")
            self.assertFalse(camp["may_merge_into_recovery"])
            self.assertEqual(camp["dispatcher"], "prepared.task")
            self.assertFalse(camp["routing_ce"])

    def test_timeseries_forbids_expand_cuts(self):
        camp = c.isolate("timeseries", "inv")
        self.assertFalse(camp["expand_cuts"])
        self.assertTrue(camp["rolling_origin"])
        self.assertEqual(camp["primary_metric"], "mase")

    def test_multiclass_is_not_multi_hot(self):
        camp = c.isolate("multiclass", "inv")
        self.assertTrue(camp["multi_label_means_multiclass"])
        self.assertEqual(camp["quant_dir"], "multi_label")

    def test_specialist_cannot_bind_recovery_for_other_tasks(self):
        camp = c.isolate("binary", "inv")
        with self.assertRaisesRegex(ValueError, "recovery"):
            c.specialist_bind(camp, c.RECOVERY_SHA)
        bind = c.specialist_bind(camp, "abc" * 21 + "ab")
        self.assertEqual(bind["task"], "binary")


if __name__ == "__main__":
    unittest.main()
