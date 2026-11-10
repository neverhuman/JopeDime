"""Exercise failed-family coverage and stable reservation without touching S3/data."""

import hashlib
import unittest
from scout import scanned_partitions, set_sha, split_family


def fixture():
    eligible = [
        dict(family_key=f"{half}{i}", half=half)
        for half in ["EVEN", "ODD"]
        for i in range(5)
    ]
    return {
        1: dict(
            eligible=eligible,
            assignment={
                "H283": {"families": ["EVEN1"]},
                "H285": {"families": ["ODD1"]},
            },
        ),
        2: dict(assignment={"H286": {"families": ["EVEN3"]}}),
        3: dict(assignment={"H288": {"families": ["ODD3"]}}),
    }


class ScoutTests(unittest.TestCase):
    def test_failed_families_before_anchor_stay_excluded(self):
        sets = scanned_partitions(fixture())
        self.assertEqual(
            sets["H286_scanned_including_failures"], {"EVEN0", "EVEN2", "EVEN3"}
        )
        self.assertEqual(
            sets["H288_scanned_including_failures"], {"ODD0", "ODD2", "ODD3"}
        )

    def test_final_reserves_include_unbuildable_families(self):
        sets = scanned_partitions(fixture())
        self.assertEqual(sets["H294_scanned_including_failures"], {"EVEN4"})
        self.assertEqual(sets["H292_scanned_including_failures"], {"ODD4"})
        self.assertEqual(len(set().union(*sets.values())), 10)

    def test_original_first_wave_scanned_prefix(self):
        sets = scanned_partitions(fixture())
        self.assertEqual(sets["H283_initial_scanned_prefix"], {"EVEN0", "EVEN1"})

    def test_missing_anchor_refuses_exhaustion_claim(self):
        cohorts = fixture()
        cohorts[2]["assignment"]["H286"]["families"] = ["not-in-catalog"]
        with self.assertRaises(ValueError):
            scanned_partitions(cohorts)

    def test_set_hash_is_order_and_duplicate_invariant(self):
        self.assertEqual(set_sha(["b", "a", "a"]), set_sha(["a", "b"]))
        self.assertNotEqual(set_sha(["a", "b"]), set_sha(["a", "c"]))

    def test_split_uses_domain_and_null_separator(self):
        for key in ["srcfile:alpha", "srcfile:beta", "dedupe-fingerprint"]:
            expected = (
                "v6A"
                if int(
                    hashlib.sha256(b"H299-v6-split-v1\0" + key.encode()).hexdigest(), 16
                )
                % 2
                == 0
                else "v6B"
            )
            self.assertEqual(split_family(key), expected)
        choices = [split_family(f"family-{i}") for i in range(100)]
        self.assertEqual(set(choices), {"v6A", "v6B"})


if __name__ == "__main__":
    unittest.main()
