import unittest

from prepare_folds import (
    PACKET_FILE_SHA256,
    PACKET_IDENTITY_SHA256,
    RosterError,
    build_manifest,
)


def sha(value: int) -> str:
    return f"{value:064x}"


def roster():
    rows = []
    for family_index in range(205):
        for _ in range(6 if family_index < 194 else 5):
            table_group = family_index // 2
            rows.append(
                {
                    "context_sha256": sha(1000 + len(rows)),
                    "family_id": f"family-{family_index:03d}",
                    "physical_table_sha256": sha(2000 + table_group),
                    "task": "regression",
                    "dl_split": "DL-Train",
                }
            )
    assert len(rows) == 1219
    return {
        "schema": "jope.w6.train-roster/v1",
        "packet_manifest_file_sha256": PACKET_FILE_SHA256,
        "packet_identity_sha256": PACKET_IDENTITY_SHA256,
        "rows": rows,
    }


class W6FoldPreparationTests(unittest.TestCase):
    def test_assignments_are_deterministic_and_complete(self):
        first = build_manifest(roster(), sha(9000))
        second = build_manifest(roster(), sha(9000))
        self.assertEqual(first, second)
        self.assertEqual(first["context_count"], 1219)
        self.assertEqual(first["family_count"], 205)
        self.assertEqual(first["fold_count"], 5)
        self.assertEqual({row["fold"] for row in first["assignments"]}, set(range(5)))
        self.assertEqual(len(first["assignments"]), 1219)
        self.assertEqual(
            {row["context_sha256"] for row in first["assignments"]},
            {row["context_sha256"] for row in roster()["rows"]},
        )

    def test_family_and_physical_table_aliases_stay_in_one_fold(self):
        result = build_manifest(roster(), sha(9000))
        fold_by_family = {row["family_id"]: row["fold"] for row in result["assignments"]}
        fold_by_table = {}
        for row in result["assignments"]:
            self.assertEqual(fold_by_family[row["family_id"]], row["fold"])
            prior = fold_by_table.setdefault(row["physical_table_sha256"], row["fold"])
            self.assertEqual(prior, row["fold"])

    def test_nontraining_rows_and_outcome_fields_are_rejected(self):
        outside = roster()
        outside["rows"][0]["dl_split"] = "DL-Val"
        with self.assertRaisesRegex(RosterError, "outside the DL-Train"):
            build_manifest(outside, sha(9000))

        leaked = roster()
        leaked["rows"][0]["fitness"] = 1.23
        with self.assertRaisesRegex(RosterError, "unrecognized fields"):
            build_manifest(leaked, sha(9000))

    def test_duplicate_contexts_and_wrong_packet_identity_are_rejected(self):
        duplicate = roster()
        duplicate["rows"][1]["context_sha256"] = duplicate["rows"][0]["context_sha256"]
        with self.assertRaisesRegex(RosterError, "duplicate context"):
            build_manifest(duplicate, sha(9000))

        wrong_pin = roster()
        wrong_pin["packet_identity_sha256"] = sha(42)
        with self.assertRaisesRegex(RosterError, "packet identity mismatch"):
            build_manifest(wrong_pin, sha(9000))


if __name__ == "__main__":
    unittest.main()
