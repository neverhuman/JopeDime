import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import materialize_g0 as g0

ROOT = Path(__file__).resolve().parents[4]
CONTEXT = "c" * 64


def fixture():
    return (
        b"JTT1"
        + struct.pack("<III", 70, 14, 2)
        + b"\x02"
        + bytes(i % 2 for i in range(70))
        + b"".join(struct.pack("<ddd", i, i % 5, 3 * i + 1) for i in range(84))
    )


def programs():
    out = []
    for i in range(96):
        dsl = f"(pipe X (ridge lambda_frac={(i + 1) / 10000}))"
        out.append(
            {
                "dsl": dsl,
                "program_sha256": g0.sha(dsl.encode()),
                "birth_seq": i,
                "parents": [],
                "operator": "grammar_random",
            }
        )
    return out


class G0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = os.environ | {
            "CARGO_INCREMENTAL": "0",
            "CARGO_PROFILE_DEV_DEBUG": "line-tables-only",
            "CARGO_BUILD_JOBS": "4",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        cls.target = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
        cls.binaries = {}
        for crate in ("lambda-worker", "island-engine", "proposal-engine"):
            command = [
                "cargo",
                "build",
                "--locked",
                "--manifest-path",
                str(
                    ROOT
                    / "deploy/lambda/training/teacher-v2/rust"
                    / crate
                    / "Cargo.toml"
                ),
                "--target-dir",
                str(cls.target),
                "--message-format=json",
            ]
            build = subprocess.run(
                command,
                cwd=ROOT,
                env=cls.env,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if build.returncode:
                raise RuntimeError(f"{crate} failed to build: {build.stderr}")
            artifacts = [
                j["executable"]
                for line in build.stdout.splitlines()
                if (j := json.loads(line)).get("reason") == "compiler-artifact"
                and j.get("executable")
                and "bin" in j["target"]["kind"]
            ]
            if len(artifacts) != 1:
                raise RuntimeError(
                    f"ambiguous Cargo binary artifacts for {crate}: {artifacts}"
                )
            cls.binaries[crate] = artifacts[0]
        fixtures = cls.target / "s1-g0-fixtures"
        fixtures.mkdir(exist_ok=True)
        cls.work = Path(
            tempfile.mkdtemp(prefix="g0-", dir=fixtures)
        )  # Retain fixture evidence.

    def test_prior_is_fresh_bounded_and_replays_without_labels(self):
        binary = self.binaries["proposal-engine"]
        pin = g0.sha(Path(binary).read_bytes())
        a = g0.seed_source(CONTEXT, binary, pin)
        self.assertEqual(len(a), 96)
        self.assertEqual(a, g0.seed_source(CONTEXT, binary, pin))
        self.assertTrue(
            all(not p["parents"] and p["operator"] == "grammar_random" for p in a)
        )
        with self.assertRaises(ValueError):
            g0.seed_source(CONTEXT, binary, "0" * 64)

    def test_real_worker_and_g0_files_are_identical_under_outer_target_flip(self):
        raw = fixture()
        changed = bytearray(raw)
        start = 17 + 70 + 70 * 3 * 8
        for i in range(14):
            struct.pack_into("<d", changed, start + i * 3 * 8 + 16, float("nan"))
        sources = programs()
        trees = []
        for tag, data in (("original", raw), ("test-flip", changed)):
            prepared = self.work / tag
            g0.prepare(CONTEXT, data, sources, prepared, "a" * 64)
            data_path = self.work / f"{tag}.jtt1"
            data_path.write_bytes(data)
            records = []
            for n, program in enumerate(sources):
                candidate = self.work / f"{tag}-{n}.candidate.json"
                output = self.work / f"{tag}-{n}.evaluation.json"
                candidate.write_bytes(g0.canonical(program))
                subprocess.run(
                    [
                        self.binaries["lambda-worker"],
                        "--fit-s1",
                        str(data_path),
                        str(candidate),
                        str(output),
                    ],
                    env=self.env,
                    check=True,
                    capture_output=True,
                    timeout=30,
                )
                record = json.loads(output.read_bytes())
                self.assertNotIn("official_test", record)
                record["context_sha256"] = CONTEXT
                records.append(record)
            evaluations = self.work / f"{tag}-evaluations.json"
            evaluations.write_bytes(g0.canonical(records))
            scored = self.work / f"{tag}-scored"
            subprocess.run(
                [
                    self.binaries["island-engine"],
                    "--score-g0",
                    "--context",
                    CONTEXT,
                    "--sample",
                    str(prepared / "sample.json"),
                    "--projection",
                    str(prepared / "projection.f32"),
                    "--seeds",
                    str(prepared / "seeds.json"),
                    "--evaluations",
                    str(evaluations),
                    "--out",
                    str(scored),
                ],
                env=self.env,
                check=True,
                capture_output=True,
                timeout=30,
            )
            command = [
                self.binaries["island-engine"],
                "--s1-two-fold",
                "--context",
                CONTEXT,
                "--sample",
                str(prepared / "sample.json"),
                "--projection",
                str(prepared / "projection.f32"),
                "--control-index",
                str(scored / "seed-index.jsonl.gz"),
                "--seen-jsonl",
                str(scored / "seen.jsonl"),
                "--proposal-binary",
                self.binaries["proposal-engine"],
            ]
            loaded = subprocess.run(
                command,
                input='{"cmd":"shutdown"}\n',
                env=self.env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(loaded.returncode, 0, loaded.stderr)
            missing = subprocess.run(
                command + ["--resume-jsonl", str(self.work / "absent-resume")],
                input="",
                env=self.env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn("requested resume file", missing.stderr)
            trees.append(
                {"prepared/" + p.name: p.read_bytes() for p in prepared.iterdir()}
                | {"scored/" + p.name: p.read_bytes() for p in scored.iterdir()}
            )
        self.assertEqual(trees[0], trees[1], "S1_ASSERT_G0_TARGET_FLIP")
        authorized = bytearray(raw)
        struct.pack_into("<d", authorized, 17 + 70 + 16, -12345)
        self.assertNotEqual(g0.sample(CONTEXT, raw), g0.sample(CONTEXT, authorized))
        # A test-dependent materializer is caught by this identity predicate.
        self.assertNotEqual(g0.sha(raw), g0.sha(changed))
        self.assertEqual(
            len((self.work / "original-scored/seen.jsonl").read_text().splitlines()), 96
        )

    def test_shape_and_publication_refusals(self):
        for count in (1, 7):
            raw = bytearray(fixture())
            raw[16] = count
            with self.assertRaises(ValueError):
                g0.sample(CONTEXT, raw)
        raw = bytearray(fixture())
        raw[17] = 2
        with self.assertRaises(ValueError):
            g0.sample(CONTEXT, raw)
        with self.assertRaises(ValueError):
            g0.prepare(
                CONTEXT, fixture(), programs()[:-1], self.work / "incomplete", "a" * 64
            )
        out = self.work / "create-only"
        g0.prepare(CONTEXT, fixture(), programs(), out, "a" * 64)
        before = {p.name: p.read_bytes() for p in out.iterdir()}
        with self.assertRaises(FileExistsError):
            g0.prepare(CONTEXT, fixture(), programs(), out, "a" * 64)
        self.assertEqual(before, {p.name: p.read_bytes() for p in out.iterdir()})

    def test_preparation_target_flip_catches_test_dependent_identity_mutant(self):
        raw = fixture()
        flipped = bytearray(raw)
        struct.pack_into("<d", flipped, 17 + 70 + 70 * 3 * 8 + 16, -12345)
        self.assertEqual(g0.sample(CONTEXT, raw), g0.sample(CONTEXT, flipped))
        original = g0.training_view

        def leaking_view(data):
            targets, folds, _ = original(data)
            return targets, folds, g0.sha(data)

        # Execute the leaking transformation through the actual sample entrypoint.
        with patch.object(g0, "training_view", leaking_view):
            self.assertNotEqual(g0.sample(CONTEXT, raw), g0.sample(CONTEXT, flipped))
        allowed = bytearray(raw)
        struct.pack_into("<d", allowed, 17 + 70 + 16, -12345)
        self.assertNotEqual(g0.sample(CONTEXT, raw), g0.sample(CONTEXT, allowed))


if __name__ == "__main__":
    unittest.main()
