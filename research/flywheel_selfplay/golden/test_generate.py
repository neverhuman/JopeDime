"""CPU pre-stage proof; no qualification capture generation or GPU use."""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import generate
import numpy as np
import torch
from reference import Reference
from safetensors.torch import load as load_tensors


class Contracts(unittest.TestCase):
    def test_canonical_json_reorders_objects_but_preserves_lists_and_refuses_ambiguous_input(self):
        self.assertEqual(generate.canonical(dict(b=2, a=1)), generate.canonical(dict(a=1, b=2)))
        self.assertNotEqual(generate.canonical([1, 2]), generate.canonical([2, 1]))
        for data in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}'):
            with self.assertRaises(ValueError):
                generate.read_json(data)

    def test_pinned_definition_does_not_execute_module_side_effects_and_refuses_hash_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "files/unit/reference.py"
            path.parent.mkdir(parents=True)
            source = b'raise RuntimeError("module body must not run")\nVALUE=7\ndef pure(x): return x+VALUE\n'
            path.write_bytes(source)
            ref = Reference.__new__(Reference)
            ref.custody, ref.source_hashes, ref.env = root, {}, {}
            ref.pins = {"/unit/reference.py": {"sha256": hashlib.sha256(source).hexdigest()}}
            ref.define("unit/reference.py", ["pure"], ["VALUE"])
            self.assertEqual(ref.env["pure"](2), 9)
            path.write_bytes(source + b"# changed\n")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                ref.define("unit/reference.py", ["pure"])

    def test_capture_paths_and_hashes_refuse_before_reading_outside_the_packet(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("../outside", "/outside"):
                with self.assertRaises(ValueError):
                    generate.captured(root, dict(path=name, sha256="0" * 64))
            path = root / "metadata.json"
            path.write_bytes(b"{}")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                generate.captured(root, dict(path=path.name, sha256="0" * 64))
            self.assertEqual(generate.captured(root, dict(path=path.name, sha256=generate.sha(b"{}"))), b"{}")

    def test_lease_excludes_another_process_and_releases_without_touching_the_real_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            lock = Path(temp) / "owned-unit.lock"
            lock.touch()
            code = "import fcntl,sys; f=open(sys.argv[1],'r+'); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)"
            with generate.qualification_lease(lock):
                p = subprocess.run([sys.executable, "-c", code, str(lock)], capture_output=True)
                self.assertNotEqual(p.returncode, 0)
            p = subprocess.run([sys.executable, "-c", code, str(lock)], capture_output=True)
            self.assertEqual(p.returncode, 0, p.stderr)

    def test_generation_requires_explicit_reviewed_capture_pin_before_lease_or_output(self):
        with mock.patch.object(generate, "generate", side_effect=AssertionError("generation reached")):
            with contextlib.redirect_stderr(io.StringIO()):
                code = generate.main(
                    [
                        "--bundle",
                        "/unused",
                        "--expected-bundle-digest",
                        "sha256:unused",
                        "--captures",
                        "/missing",
                        "--output",
                        "/unused-output",
                    ]
                )
            self.assertEqual(code, 2)


@unittest.skipUnless(
    os.environ.get("JOPE_RES07_BUNDLE"), "set explicit RES-06 bundle and digest for actual CPU proof"
)
class RealBundle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.bundle = Path(os.environ["JOPE_RES07_BUNDLE"])
        cls.digest = os.environ["JOPE_RES07_BUNDLE_DIGEST"]
        cls.before = {p.name: generate.sha(p.read_bytes()) for p in cls.bundle.iterdir()}
        cls.manifest, cls.custody, cls.reference, cls.report = generate.validate_bundle(
            cls.bundle, cls.digest
        )

    @classmethod
    def tearDownClass(cls):
        after = {p.name: generate.sha(p.read_bytes()) for p in cls.bundle.iterdir()}
        if cls.before != after:
            raise AssertionError("original RES-06 bundle changed")

    def test_real_all_tensor_and_source_contracts_validate_without_gpu_or_fixture_claim(self):
        self.assertEqual(self.report["tensors"], 338)
        self.assertEqual(self.report["validated_prior_rows"], 881)
        self.assertEqual(self.report["qualification"], dict(M2=False, M3=False, M5=False))
        self.assertEqual(self.report["fixture_files_written"], 0)
        self.assertFalse(torch.cuda.is_initialized())
        # Constructor/alias wiring cannot modify even the in-memory frozen state.
        generate.exporter().validate_state(self.reference.state)

    def test_canonical_safetensors_preserves_real_program_projection_f64_and_integer_shape(self):
        bank = [
            generate.read_json(row)
            for row in generate.decompress((self.bundle / "bank.jsonl.zst").read_bytes()).splitlines()
        ]
        features, projection = self.reference.featurize(bank[0]["dsl"])
        values = dict(
            feat412=features,
            proj16=projection,
            scalar=np.asarray(7, dtype=np.int64),
            indices=np.arange(4, dtype=np.int64),
        )
        encoded = generate.tensor_bytes(values)
        reversed_values = dict(reversed(list(values.items())))
        self.assertEqual(encoded, generate.tensor_bytes(reversed_values))
        decoded = load_tensors(encoded)
        self.assertEqual(decoded["proj16"].dtype, torch.float64)
        self.assertEqual(decoded["scalar"].shape, torch.Size([]))
        self.assertTrue(np.array_equal(decoded["proj16"].numpy(), projection))
        broken = projection.copy()
        broken[0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            generate.tensor_bytes(dict(proj16=broken))

    def test_payload_tamper_and_missing_frozen_buffer_are_refused_without_changing_original(self):
        with tempfile.TemporaryDirectory() as temp:
            copied = Path(temp) / "bundle"
            shutil.copytree(self.bundle, copied)
            path = copied / "featurizer.json"
            path.chmod(0o600)  # Deliberate corruption only in our private copied fixture.
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "BundleDigest mismatch"):
                generate.validate_bundle(copied, self.digest)
        state = dict(self.reference.state)
        state.pop("mu_lm48")
        with self.assertRaisesRegex(ValueError, "complete tensor manifest"):
            generate.exporter().validate_state(state)


if __name__ == "__main__":
    unittest.main()
