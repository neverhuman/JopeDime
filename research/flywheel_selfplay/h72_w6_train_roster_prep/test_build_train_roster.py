from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from research.flywheel_selfplay.h72_w6_train_roster_prep.build_train_roster import (
    JsonStream,
    RosterError,
    read_selected_metadata,
    read_train_rows,
)


class TrainOnlyStreamingTests(unittest.TestCase):
    def test_train_reader_stops_before_unparseable_dev_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selected.json"
            path.write_text('{"train":[{"context_sha256":"train-only"}],"dev":THIS MUST NOT PARSE')
            self.assertEqual(read_train_rows(path), [{"context_sha256": "train-only"}])

    def test_metadata_reader_skips_unselected_values_without_decoding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.json"
            path.write_text(
                '{"train-id":{"canonical_sha256":"' + "a" * 64 +
                '"},"dev-id":{"not-json":NaN}}'
            )
            rows = read_selected_metadata(path, {"train-id"})
            self.assertEqual(set(rows), {"train-id"})
            self.assertEqual(rows["train-id"]["canonical_sha256"], "a" * 64)

    def test_train_reader_requires_train_partition_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selected.json"
            path.write_text('{"dev":[],"train":[]}')
            with self.assertRaisesRegex(RosterError, "first"):
                read_train_rows(path)

    def test_value_scanner_skips_nested_json_structure(self) -> None:
        import io

        reader = JsonStream(io.StringIO('{"nested":[{"s":"}\\\"{"}]} ,"next":1'))
        reader.get()
        self.assertEqual(reader.raw_string(), '"nested"')
        reader.skip_ws()
        self.assertEqual(reader.get(), ":")
        self.assertIsNone(reader.value(decode=False))
        reader.skip_ws()
        self.assertEqual(reader.get(), "}")
        reader.skip_ws()
        self.assertEqual(reader.get(), ",")
        reader.skip_ws()
        self.assertEqual(reader.raw_string(), '"next"')


if __name__ == "__main__":
    unittest.main()
