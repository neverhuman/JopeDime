import unittest

import league_episode as league


def arms(neural=1.0, generic=1.1, random=1.2, teacher=1.05, logprob=None):
    out = {
        "neural": league.arm(neural, program_sha256="n", evaluations=64, policy_logprob=logprob),
        "generic": league.arm(generic, program_sha256="g", evaluations=64),
        "random": league.arm(random, program_sha256="r", evaluations=64),
        "teacher": league.arm(teacher, program_sha256="t", evaluations=1),
    }
    if logprob is None:
        out["neural"].pop("policy_logprob", None)
    return out


class LeagueEpisodeTests(unittest.TestCase):
    def test_required_arms_and_optional_logprob(self):
        board = dict(family="islr-auto", cut=0, prepared_sha256="p" * 64)
        row = league.episode(board, arms(), independent_evaluation=False)
        self.assertEqual(row["schema"], "parallel.league_episode.v1")
        self.assertEqual(set(row["arms"]), {"neural", "generic", "random", "teacher"})
        self.assertEqual(row["neural_vs_teacher"], "win")
        self.assertEqual(row["neural_vs_random"], "win")
        self.assertEqual(row["neural_vs_generic"], "win")
        self.assertFalse(row["independent_evaluation"])
        self.assertFalse(row["serving_written"])
        self.assertFalse(row["shuffled_minted"])
        self.assertNotIn("policy_logprob", row["arms"]["neural"])
        league.validate(row)
        tagged = league.episode(board, arms(logprob=-2.5), independent_evaluation=False,
                                policy_logprob=-2.5)
        self.assertEqual(tagged["arms"]["neural"]["policy_logprob"], -2.5)
        self.assertFalse(tagged["independent_evaluation"])
        league.validate(tagged)

    def test_self_declared_independence_and_nonfinite_metrics_reject(self):
        for flag in (True, 1, 0, "false"):
            with self.assertRaisesRegex(ValueError, "independent"):
                league.episode({}, arms(), flag)
        row = league.episode({}, arms(), False)
        row["independent_evaluation"] = True
        with self.assertRaisesRegex(ValueError, "independent"):
            league.validate(row)
        for score in (True, False, float("nan"), float("inf"), -1):
            with self.assertRaises(ValueError):
                league.episode({}, arms(neural=score), False)

    def test_missing_random_and_shuffled_are_rejected(self):
        board = dict(family="islr-auto", cut=0)
        incomplete = arms()
        del incomplete["random"]
        with self.assertRaisesRegex(ValueError, "required arms"):
            league.episode(board, incomplete, False)
        extra = dict(arms(), shuffled=league.arm(9.0))
        with self.assertRaisesRegex(ValueError, "required arms"):
            league.episode(board, extra, False)
        row = league.episode(board, arms(neural=1.0, teacher=1.0, generic=0.9, random=2.0), False)
        self.assertEqual(row["neural_vs_teacher"], "tie")
        self.assertEqual(row["neural_vs_generic"], "loss")
        self.assertEqual(row["neural_vs_random"], "win")

    def test_serving_write_cannot_validate(self):
        row = league.episode(dict(family="x"), arms(), False)
        row["serving_written"] = True
        with self.assertRaisesRegex(ValueError, "must not write serving"):
            league.validate(row)
        row = league.episode(dict(family="x"), arms(), False)
        row["shuffled_minted"] = True
        with self.assertRaisesRegex(ValueError, "shuffled stays off"):
            league.validate(row)
        row = league.episode(dict(family="x"), arms(), False)
        row["neural_vs_teacher"] = "loss"
        with self.assertRaisesRegex(ValueError, "disagrees"):
            league.validate(row)


if __name__ == "__main__":
    unittest.main()
