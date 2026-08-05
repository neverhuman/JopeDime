from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import jope_spec_policy  # noqa: E402
import test_exact_train as trainer  # noqa: E402


class TrainerSpecPolicyTests(unittest.TestCase):
    def test_renderer_consumes_the_complete_central_trainer_policy(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            prime = root / "prime"
            train_source = prime / "jope" / "policy" / "train_v3.py"
            train_source.parent.mkdir(parents=True)
            train_source.write_text("# immutable trainer provenance\n")
            warm = root / "warm.model.pt"
            warm.write_bytes(b"checkpoint")
            train = root / "train.jsonl"
            validation = root / "validation.jsonl"
            train.write_text("{}\n")
            validation.write_text("{}\n")
            manifest = root / "corpus-manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "source_label_watermark": 7,
                        "train": str(train),
                        "validation": str(validation),
                    }
                )
            )

            with (
                mock.patch.object(trainer, "ROOT", root),
                mock.patch.object(trainer, "prime_source_root", return_value=prime),
                mock.patch.object(
                    trainer,
                    "accepted_checkpoint",
                    return_value=(warm, "a" * 64),
                ),
            ):
                rendered = trainer.render_spec(manifest, task="regression")

            body = Path(rendered["spec"]).read_text()
            self.assertIn(jope_spec_policy.render_resource_block("trainer"), body)
            self.assertIn(jope_spec_policy.render_placement_block("trainer"), body)
            self.assertIn(jope_spec_policy.render_limits_block("trainer"), body)
            self.assertIn("cpus_reserved = 4", body)
            self.assertIn("ram_mb = 16384", body)
            self.assertIn("preemptible = false", body)
            self.assertNotIn("/etc/hostname", body)
            self.assertNotIn("hostname =", body)


if __name__ == "__main__":
    unittest.main()
