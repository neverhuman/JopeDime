"""CPU pre-stage proof; no qualification capture generation or GPU use."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import struct
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import capture_cpu
import generate
import numpy as np
import torch
from reference import Reference
from safetensors.torch import load as load_tensors


class Contracts(unittest.TestCase):
    def test_canonical_json_reorders_objects_but_preserves_lists_and_refuses_ambiguous_input(
        self,
    ):
        self.assertEqual(
            generate.canonical(dict(b=2, a=1)), generate.canonical(dict(a=1, b=2))
        )
        self.assertNotEqual(generate.canonical([1, 2]), generate.canonical([2, 1]))
        for data in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}'):
            with self.assertRaises(ValueError):
                generate.read_json(data)

    def test_pinned_definition_does_not_execute_module_side_effects_and_refuses_hash_drift(
        self,
    ):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "files/unit/reference.py"
            path.parent.mkdir(parents=True)
            source = b'raise RuntimeError("module body must not run")\nVALUE=7\ndef pure(x): return x+VALUE\n'
            path.write_bytes(source)
            ref = Reference.__new__(Reference)
            ref.custody, ref.source_hashes, ref.env = root, {}, {}
            ref.pins = {
                "/unit/reference.py": {"sha256": hashlib.sha256(source).hexdigest()}
            }
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
            self.assertEqual(
                generate.captured(
                    root, dict(path=path.name, sha256=generate.sha(b"{}"))
                ),
                b"{}",
            )

    def test_lease_excludes_another_process_and_releases_without_touching_the_real_lock(
        self,
    ):
        with tempfile.TemporaryDirectory() as temp:
            lock = Path(temp) / "owned-unit.lock"
            lock.touch()
            code = "import fcntl,sys; f=open(sys.argv[1],'r+'); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)"
            with generate.qualification_lease(lock):
                p = subprocess.run(
                    [sys.executable, "-c", code, str(lock)], capture_output=True
                )
                self.assertNotEqual(p.returncode, 0)
            p = subprocess.run(
                [sys.executable, "-c", code, str(lock)], capture_output=True
            )
            self.assertEqual(p.returncode, 0, p.stderr)

    def test_generation_requires_explicit_reviewed_capture_pin_before_lease_or_output(
        self,
    ):
        with mock.patch.object(
            generate, "generate", side_effect=AssertionError("generation reached")
        ):
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
        # The independent CPU replay mode also refuses partial authentication
        # before opening a plan, touching the native capture path or publishing.
        with mock.patch.object(capture_cpu, "capture") as native:
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as refused:
                    capture_cpu.main(
                        [
                            "--plan",
                            "/missing",
                            "--expected-plan-sha256",
                            "0" * 64,
                            "--output",
                            "/unused-output",
                            "--regenerate-captures",
                            "/missing",
                            "--expected-captures-sha256",
                            "0" * 64,
                        ]
                    )
            self.assertEqual(refused.exception.code, 2)
            native.assert_not_called()


@unittest.skipUnless(
    os.environ.get("JOPE_RES07_BUNDLE"),
    "set explicit RES-06 bundle and digest for actual CPU proof",
)
class RealBundle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.bundle = Path(os.environ["JOPE_RES07_BUNDLE"])
        cls.digest = os.environ["JOPE_RES07_BUNDLE_DIGEST"]
        cls.before = {
            p.name: generate.sha(p.read_bytes()) for p in cls.bundle.iterdir()
        }
        cls.manifest, cls.custody, cls.reference, cls.report = generate.validate_bundle(
            cls.bundle, cls.digest
        )

    @classmethod
    def tearDownClass(cls):
        after = {p.name: generate.sha(p.read_bytes()) for p in cls.bundle.iterdir()}
        if cls.before != after:
            raise AssertionError("original RES-06 bundle changed")

    def test_real_all_tensor_and_source_contracts_validate_without_gpu_or_fixture_claim(
        self,
    ):
        self.assertEqual(self.report["tensors"], 338)
        self.assertEqual(self.report["validated_prior_rows"], 881)
        self.assertEqual(
            self.report["qualification"], dict(M2=False, M3=False, M5=False)
        )
        self.assertEqual(self.report["fixture_files_written"], 0)
        self.assertFalse(torch.cuda.is_initialized())
        # Constructor/alias wiring cannot modify even the in-memory frozen state.
        generate.exporter().validate_state(self.reference.state)

    def test_model_prior_population_is_preserved_and_cannot_be_relabelled_as_proposer(
        self,
    ):
        bank = [
            generate.read_json(row)
            for row in generate.decompress(
                (self.bundle / "bank.jsonl.zst").read_bytes()
            ).splitlines()
        ]
        original = generate.canonical(bank)
        programs = generate.model_prior_programs(bank)
        self.assertEqual(len(programs), 872)
        self.assertEqual({p["dsl"] for p in programs}, {row["dsl"] for row in bank})
        self.assertEqual(generate.canonical(bank), original)
        self.assertFalse(self.report["subset"]["historical_corpus_complete"])
        self.assertFalse(self.report["subset"]["proposer_retrieval_qualified"])
        for fields in (
            {"role": "proposer"},
            {"coordinate_space": "raw_svd_lm48_first12"},
            {"role": None},
        ):
            changed = [dict(bank[0], **fields), *bank[1:]]
            with self.assertRaisesRegex(ValueError, "only the tagged model_prior"):
                generate.model_prior_programs(changed)

    def test_canonical_safetensors_preserves_real_program_projection_f64_and_integer_shape(
        self,
    ):
        bank = [
            generate.read_json(row)
            for row in generate.decompress(
                (self.bundle / "bank.jsonl.zst").read_bytes()
            ).splitlines()
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

    def test_payload_tamper_and_missing_frozen_buffer_are_refused_without_changing_original(
        self,
    ):
        with tempfile.TemporaryDirectory() as temp:
            copied = Path(temp) / "bundle"
            shutil.copytree(self.bundle, copied)
            path = copied / "featurizer.json"
            path.chmod(
                0o600
            )  # Deliberate corruption only in our private copied fixture.
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "BundleDigest mismatch"):
                generate.validate_bundle(copied, self.digest)
        state = dict(self.reference.state)
        state.pop("mu_lm48")
        with self.assertRaisesRegex(ValueError, "complete tensor manifest"):
            generate.exporter().validate_state(state)


class CpuCaptureGuards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Row-free metadata exercises refusal paths; it never reaches inference.
        contexts = [
            dict(
                context_sha256=f"{i:064x}",
                dataset_id=str(i),
                family_id=str(i),
                source_split="inner" if i == 0 else "train",
                dl_split="DL-Train",
                train_only_jtt1_sha256="0" * 64,
                rows=64,
                columns=2,
                table_path="must-not-be-opened",
                program_sha256=[str(j) for j in range(20)],
            )
            for i in range(50)
        ]
        selected = self.root / "selected.json"
        selected.write_text(json.dumps(dict(train=contexts)))
        self.plan = dict(
            contexts=contexts,
            selected=str(selected),
            selected_sha256=hashlib.sha256(selected.read_bytes()).hexdigest(),
            budget=dict(
                after_lock_wall_seconds=1799,
                cpu_affinity_slots=4,
                math_threads=1,
                max_native_replays=1000,
                max_native_fold_fits=2000,
                max_output_bytes=2**31,
            ),
        )

    def test_training_and_inner_roles_require_matching_pinned_selection(self):
        self.assertEqual(len(capture_cpu.validate_plan(self.plan)), 50)
        for value in ("test", "DL-Val", "sealed"):
            plan = copy.deepcopy(self.plan)
            plan["contexts"][0]["source_split"] = value
            with self.assertRaises(ValueError):
                capture_cpu.validate_plan(plan)

    def test_context_identity_duplicate_and_oversize_refuse_before_tables(self):
        for field, value in [
            ("dl_split", "DL-Val"),
            ("dataset_id", "different"),
            ("family_id", "different"),
            ("train_only_jtt1_sha256", "f" * 64),
            ("rows", 1025),
        ]:
            plan = copy.deepcopy(self.plan)
            plan["contexts"][0][field] = value
            with self.assertRaises(ValueError):
                capture_cpu.validate_plan(plan)
        plan = copy.deepcopy(self.plan)
        plan["contexts"][1] = plan["contexts"][0]
        with self.assertRaises(ValueError):
            capture_cpu.validate_plan(plan)
        # Refuse drift and a different captured population before model loading.
        packet = dict(contexts=self.plan["contexts"], bundle_digest="sha256:unused")
        captures = self.root / "captures.json"
        captures.write_bytes(generate.canonical(packet))
        self.plan.update(bundle_digest=packet["bundle_digest"], custody=str(self.root))
        with mock.patch.dict(os.environ, CUDA_VISIBLE_DEVICES=""):
            with mock.patch.object(os, "getpriority", return_value=10):
                with mock.patch.object(
                    torch.cuda, "is_initialized", return_value=False
                ):
                    with mock.patch.object(generate, "_generate_locked") as numerical:
                        with self.assertRaisesRegex(
                            ValueError, "input identity differs"
                        ):
                            capture_cpu.regenerate(
                                self.plan,
                                captures,
                                "0" * 64,
                                "f" * 64,
                                self.root / "output",
                            )
                        packet["contexts"] = copy.deepcopy(packet["contexts"])
                        packet["contexts"][0]["family_id"] = "different"
                        captures.write_bytes(generate.canonical(packet))
                        with self.assertRaisesRegex(
                            ValueError, "reviewed plan population"
                        ):
                            capture_cpu.regenerate(
                                self.plan,
                                captures,
                                generate.sha(captures.read_bytes()),
                                "f" * 64,
                                self.root / "output",
                            )
                        numerical.assert_not_called()
        self.assertFalse((self.root / "output").exists())

    def test_original_resource_limits_cannot_expand(self):
        for field, value in [
            ("after_lock_wall_seconds", 1800),
            ("cpu_affinity_slots", 5),
            ("math_threads", 2),
            ("max_native_replays", 1001),
            ("max_native_fold_fits", 2001),
            ("max_output_bytes", 2**31 + 1),
        ]:
            plan = copy.deepcopy(self.plan)
            plan["budget"][field] = value
            with self.assertRaises(ValueError):
                capture_cpu.validate_plan(plan)

        # Re-packaging does not re-arm the native deadline or call its evaluator.
        plan = dict(self.plan, cpu_deadline_epoch_seconds=0)
        path = self.root / "plan.json"
        path.write_bytes(generate.canonical(plan))
        with mock.patch.object(capture_cpu, "regenerate", return_value={}) as replay:
            with mock.patch.object(capture_cpu, "capture") as native:
                with mock.patch.object(capture_cpu.signal, "alarm") as alarm:
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(
                            capture_cpu.main(
                                [
                                    "--plan",
                                    str(path),
                                    "--expected-plan-sha256",
                                    generate.sha(path.read_bytes()),
                                    "--output",
                                    str(self.root / "output"),
                                    "--regenerate-captures",
                                    str(self.root / "captures.json"),
                                    "--expected-captures-sha256",
                                    "a" * 64,
                                    "--expected-index-sha256",
                                    "b" * 64,
                                ]
                            ),
                            0,
                        )
                    replay.assert_called_once()
                    native.assert_not_called()
                    alarm.assert_not_called()
        self.assertEqual(path.read_bytes(), generate.canonical(plan))

    def test_native_transport_preserves_real_table_prefix_folds_and_labels(self):
        # Use the actual approved training table when running the real CPU lane.
        import os

        path = os.environ.get("JOPE_CPU_TRAINING_TABLE")
        if path is None:
            self.skipTest(
                "set JOPE_CPU_TRAINING_TABLE for real JTT byte-preservation proof"
            )
        original = Path(path).read_bytes()
        transport = capture_cpu.native_training_transport(original)
        nt, nq, width = struct.unpack_from("<III", transport, 4)
        self.assertEqual(nq, 10)
        self.assertEqual(transport[:8], original[:8])
        self.assertEqual(transport[12 : len(original)], original[12:])
        self.assertEqual(
            transport[len(original) :],
            original[17 + nt : 17 + nt + 10 * (width + 1) * 8],
        )
        self.assertEqual(struct.unpack_from("<I", original, 8)[0], 0)
        self.assertEqual(len(transport), len(original) + 10 * (width + 1) * 8)
        with self.assertRaises(ValueError):
            capture_cpu.native_training_transport(transport)
        with self.assertRaises(ValueError):
            capture_cpu.native_training_transport(original[:-8])


if __name__ == "__main__":
    unittest.main()
