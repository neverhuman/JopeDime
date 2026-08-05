#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from scripts import absolute_path_audit as audit_mod


def file_row(path: Path) -> dict[str, object]:
    link_info = path.lstat()
    symlink = stat.S_ISLNK(link_info.st_mode)
    info = path.stat() if symlink else link_info
    kind = "file" if stat.S_ISREG(info.st_mode) else "directory"
    row: dict[str, object] = {
        "exists": True,
        "type": kind,
        "mode": stat.S_IMODE(info.st_mode),
        "uid": info.st_uid,
        "gid": info.st_gid,
        "executable": os.access(path, os.X_OK),
        "writable": os.access(path, os.W_OK),
        "symlink": symlink,
    }
    if symlink:
        row["symlink_target"] = os.readlink(path)
    if kind == "file":
        row["size"] = info.st_size
        row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return row


class BundleFixture:
    def __init__(self, root: Path) -> None:
        self.digest = "a" * 64
        self.bundle = root / self.digest
        self.binary = self.bundle / "runtime/bin/worker"
        self.unit = self.bundle / "runtime/systemd/worker.service"
        self.binary.parent.mkdir(parents=True)
        self.unit.parent.mkdir(parents=True)
        self.binary.write_bytes(b"worker-binary")
        self.binary.chmod(0o755)
        current_binary = self.bundle.parent / "current/runtime/bin/worker"
        self.unit.write_text(
            "[Service]\n"
            "ExecStartPre=/bin/sh -c true\n"
            f"ExecStart={current_binary} \\\n  --runtime-path {self.bundle.parent}/current/runtime/bin\n"
        )
        self.unit.chmod(0o644)
        inventory = []
        for target, path in (
            ("runtime/bin/worker", self.binary),
            ("runtime/systemd/worker.service", self.unit),
        ):
            inventory.append(
                {
                    "target": target,
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        service_manifest = {
            "schema": "JopeSelfplayServiceManifest.v3",
            "binaries": ["worker"],
            "services": [
                {
                    "unit": "worker.service",
                    "binary": "worker",
                    "owner": "test",
                    "active": True,
                }
            ],
        }
        self.services = root / "services.json"
        self.services.write_text(json.dumps(service_manifest))
        (self.bundle / "bundle.json").write_text(
            json.dumps(
                {
                    "schema": "RuntimeBundle.v3",
                    "bundle_sha256": self.digest,
                    "files": inventory,
                    "service_manifest": service_manifest,
                    "service_manifest_sha256": hashlib.sha256(
                        self.services.read_bytes()
                    ).hexdigest(),
                }
            )
        )

    def probe(self, nodes: list[str], paths: list[str]):
        return (
            {node: {path: file_row(Path(path)) for path in paths} for node in nodes},
            {},
        )


class AbsolutePathAuditTests(unittest.TestCase):
    def test_logical_continuation_and_colon_list_are_visible(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            bundle = root / ("b" * 64)
            unit = root / "joined.service"
            unit.write_text(
                "ExecStart=/usr/bin/env \\\n PYTHONPATH="
                f"{root}/current/pytorch:{root}/current/shared \\\n /usr/bin/python3\n"
            )
            uses = audit_mod.collect_unit(unit, bundle)
            paths = {use.path for use in uses}
            self.assertIn(str(bundle / "pytorch"), paths)
            self.assertIn(str(bundle / "shared"), paths)
            self.assertIn("/usr/bin/env", paths)
            self.assertIn("/usr/bin/python3", paths)
            self.assertTrue(all("current" not in path for path in paths if str(root) in path))

    def test_dangling_continuation_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "dangling"):
            audit_mod.logical_systemd_entries("ExecStart=/usr/bin/env \\")

    def test_exact_root_rewrite_does_not_move_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            exact = root / ("c" * 64)
            current = root / "current/runtime/bin/tool"
            rewritten, changed = audit_mod.resolve_current(str(current), exact)
            self.assertTrue(changed)
            self.assertEqual(rewritten, str(exact / "runtime/bin/tool"))
            self.assertFalse((root / "current").exists())

    def test_only_real_input_source_is_staged_and_workdir_is_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            exact = root / ("d" * 64)
            spec = root / "job.toml"
            pinned = "1" * 64
            spec.write_text(
                "[job]\n"
                'command = "/usr/bin/python3"\n'
                'workdir = "/var/lib/jope/runtime-work"\n'
                'source = "/var/lib/jope/objects/input.bin"\n\n'
                "[[input]]\n"
                'source = "/var/lib/jope/objects/input.bin"\n'
                f'sha256 = "{pinned}"\n'
            )
            uses = audit_mod.collect_spec(spec, exact)
            by_original = {use.original: use for use in uses}
            self.assertEqual(
                by_original["/var/lib/jope/objects/input.bin"].scope,
                "staged_submission_node",
            )
            self.assertEqual(
                by_original["/var/lib/jope/objects/input.bin"].pinned_sha256,
                pinned,
            )
            self.assertEqual(
                by_original["/var/lib/jope/runtime-work"].scope,
                "job_eligible_nodes",
            )
            self.assertEqual(
                [
                    use.scope
                    for use in uses
                    if use.original == "/var/lib/jope/objects/input.bin"
                ][0],
                "job_eligible_nodes",
            )
            self.assertEqual(
                sorted(
                    use.scope
                    for use in uses
                    if use.original == "/var/lib/jope/objects/input.bin"
                ),
                ["job_eligible_nodes", "staged_submission_node"],
            )

    def test_mutable_checkout_and_traversal_fail_closed(self) -> None:
        exact = Path("/var/lib/jope") / ("e" * 64)
        use = audit_mod.path_use(
            "/home/ubuntu/jope-prime/registry/operator_registry.json",
            source=Path("job.toml"),
            field="dependencies.smoke.args[1]",
            scope="job_eligible_nodes",
            bundle_root=exact,
        )
        classification, error = audit_mod.classify(use, exact)
        self.assertIsNone(classification)
        self.assertIn("mutable checkout", error or "")
        with self.assertRaisesRegex(ValueError, "traversal"):
            audit_mod.path_use(
                "/var/lib/jope/../secret",
                source=Path("job.toml"),
                field="job.workdir",
                scope="job_eligible_nodes",
                bundle_root=exact,
            )

    def test_exact_content_addressed_bundle_path_is_not_excluded(self) -> None:
        exact = Path("/var/lib/jope") / ("e" * 64)
        rollback = audit_mod.RUNTIME_BUNDLES_ROOT / ("f" * 64) / "runtime/bin/tool"
        use = audit_mod.path_use(
            str(rollback),
            source=Path("job.toml"),
            field="job.args[0]",
            scope="job_eligible_nodes",
            bundle_root=exact,
        )
        classification, error = audit_mod.classify(use, exact)
        self.assertIsNone(error)
        self.assertEqual(classification, "content_addressed_immutable_external")

    def test_host_executable_symlink_is_verified_by_resolved_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            target = root / "tool"
            link = root / "tool-link"
            target.write_bytes(b"tool")
            target.chmod(0o755)
            link.symlink_to(target)
            contract, error = audit_mod._local_contract(
                link,
                "host_executable_or_capability",
                {},
                root / ("1" * 64),
                set(),
            )
            self.assertIsNone(error)
            self.assertEqual(contract["type"], "file")
            self.assertTrue(contract["symlink"])
            self.assertEqual(contract["symlink_target"], str(target))
            link.unlink()
            link.symlink_to(root / "absent")
            _, error = audit_mod._local_contract(
                link,
                "host_executable_or_capability",
                {},
                root / ("1" * 64),
                set(),
            )
            self.assertIn("dangling", error or "")

    def test_symlink_target_mismatch_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = BundleFixture(Path(raw))

            def mismatch(nodes: list[str], paths: list[str]):
                results, failures = fixture.probe(nodes, paths)
                results["node-a"]["/bin/sh"]["symlink_target"] = "different-shell"
                return results, failures

            report = audit_mod.audit(
                bundle_root=fixture.bundle,
                service_manifest_path=fixture.services,
                specs=[],
                units=[],
                nodes=["node-a", "node-b"],
                node_probe=mismatch,
            )
            self.assertFalse(report["healthy"])
            self.assertIn(
                "symlink_target_mismatch",
                {finding["code"] for finding in report["findings"]},
            )

    def test_cli_rejects_a_node_subset(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly xbabe0"):
            audit_mod.validate_cli_nodes(["xbabe2"])
        audit_mod.validate_cli_nodes(list(reversed(audit_mod.DEFAULT_NODES)))

    def test_unreachable_mandatory_node_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = BundleFixture(Path(raw))

            def unreachable(nodes: list[str], paths: list[str]):
                return (
                    {"node-a": {path: file_row(Path(path)) for path in paths}},
                    {"node-b": "TimeoutExpired: probe timed out"},
                )

            report = audit_mod.audit(
                bundle_root=fixture.bundle,
                service_manifest_path=fixture.services,
                specs=[],
                units=[],
                nodes=["node-a", "node-b"],
                node_probe=unreachable,
            )
            self.assertFalse(report["healthy"])
            self.assertIn(
                "mandatory_node_unreachable",
                {finding["code"] for finding in report["findings"]},
            )
            self.assertEqual(report["nodes"]["node-b"]["status"], "unreachable")

    def test_cross_node_sha_mismatch_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = BundleFixture(Path(raw))

            def mismatch(nodes: list[str], paths: list[str]):
                results, failures = fixture.probe(nodes, paths)
                results["node-b"][str(fixture.binary)]["sha256"] = "f" * 64
                return results, failures

            report = audit_mod.audit(
                bundle_root=fixture.bundle,
                service_manifest_path=fixture.services,
                specs=[],
                units=[],
                nodes=["node-a", "node-b"],
                node_probe=mismatch,
            )
            self.assertFalse(report["healthy"])
            mismatches = [
                finding
                for finding in report["findings"]
                if finding["code"] == "sha256_mismatch"
            ]
            self.assertEqual(len(mismatches), 1)
            self.assertEqual(mismatches[0]["node"], "node-b")

    def test_green_receipt_is_typed_and_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = BundleFixture(root)
            report = audit_mod.audit(
                bundle_root=fixture.bundle,
                service_manifest_path=fixture.services,
                specs=[],
                units=[],
                nodes=["node-a", "node-b"],
                node_probe=fixture.probe,
            )
            self.assertTrue(report["healthy"], report["findings"])
            self.assertEqual(report["schema"], "JopeAbsolutePathAudit.v1")
            output = root / "receipts/path-audit.json"
            audit_mod.atomic_json(output, report)
            self.assertEqual(json.loads(output.read_text())["healthy"], True)
            self.assertEqual(list(output.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
