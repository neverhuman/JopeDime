"""Focused regressions for the r882 custody exporter."""

import hashlib
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save as save_safetensors

import prepare_release


class PinnedSafetensorsTests(unittest.TestCase):
    def test_path_substitution_after_verification_does_not_change_parsed_bytes(self):
        verified = save_safetensors({"weight": torch.tensor([1.0, 2.0])})
        substituted = save_safetensors({"weight": torch.tensor([9.0, 9.0])})
        expected = hashlib.sha256(verified).hexdigest()

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "base.safetensors"
            path.write_bytes(verified)
            original_pinned = prepare_release.pinned

            def substitute_after_verification(asset_path, digest, size=None):
                data = original_pinned(asset_path, digest, size)
                asset_path.write_bytes(substituted)
                return data

            prepare_release.pinned = substitute_after_verification
            try:
                tensors = prepare_release.pinned_safetensors(
                    path, expected, len(verified)
                )
            finally:
                prepare_release.pinned = original_pinned

            self.assertNotEqual(path.read_bytes(), verified)
            self.assertTrue(torch.equal(tensors["weight"], torch.tensor([1.0, 2.0])))


if __name__ == "__main__":
    unittest.main()
