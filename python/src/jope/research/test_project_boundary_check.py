from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("project_boundary_check.py")
SPEC = importlib.util.spec_from_file_location("project_boundary_check", MODULE_PATH)
assert SPEC and SPEC.loader
boundary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(boundary)


def install_service_manifest(root: Path, unit_text: str) -> None:
    (root / "manifests/selfplay").mkdir(parents=True)
    (root / "systemd").mkdir()
    (root / "manifests/selfplay/service-manifest.v3.json").write_text(
        json.dumps(
            {
                "schema": "JopeSelfplayServiceManifest.v3",
                "binaries": ["jope_selfplay_queue"],
                "services": [
                    {
                        "unit": "jope-selfplay-queue.service",
                        "binary": "jope_selfplay_queue",
                        "owner": "queue",
                        "active": True,
                    }
                ],
            }
        )
    )
    (root / "systemd/jope-selfplay-queue.service").write_text(unit_text)


def clean_unit() -> str:
    return "\n".join(
        [
            "[Service]",
            "WorkingDirectory=/home/ubuntu/jope-flywheel/test_exact_gen0",
            "ExecStart=/home/ubuntu/jope-flywheel/test_exact_gen0/runtime-bundles/current/runtime/bin/jope_selfplay_queue --once",
        ]
    )


def test_manifest_declares_complete_authority_boundary():
    manifest = boundary.load_manifest()
    assert manifest["authority"]["source_control"] == "."
    assert manifest["authority"]["runtime_state"].endswith("jope-flywheel/test_exact_gen0")
    assert manifest["task_vocabulary"] == ["regression", "binary", "multiclass", "timeseries"]
    assert manifest["service_manifest"] == "manifests/selfplay/service-manifest.v3.json"
    assert set(manifest["permitted_dependencies"]) == {"jope_prime", "scq"}


def test_clean_tree_passes(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/main.rs").write_text('const TASK: &str = "binary";\n')
    install_service_manifest(tmp_path, clean_unit())
    assert boundary.check(tmp_path, boundary.load_manifest()) == []


def test_sibling_import_fails_closed(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/main.rs").write_text('include!("/home/ubuntu/jope-slime/src/lib.rs");\n')
    install_service_manifest(tmp_path, clean_unit())
    failures = boundary.check(tmp_path, boundary.load_manifest())
    assert any("jope-slime" in failure for failure in failures)


def test_documented_lesson_is_allowed_but_executable_is_not(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/lesson.md").write_text("A jope-slime checkpoint taught a prose-only lesson.\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src/main.rs").write_text('const FOREIGN: &str = "jope-slime checkpoint";\n')
    install_service_manifest(tmp_path, clean_unit())
    failures = boundary.check(tmp_path, boundary.load_manifest())
    assert failures == ["src/main.rs: forbidden project reference jope-slime checkpoint"]


def test_tracked_symlink_and_mutable_unit_fail_closed(tmp_path):
    (tmp_path / "src").mkdir()
    os.symlink("/home/ubuntu/jope-slime/src/lib.rs", tmp_path / "src/foreign.rs")
    install_service_manifest(
        tmp_path,
        "\n".join(
            [
                "[Service]",
                "WorkingDirectory=/home/ubuntu/jope-dime",
                "ExecStart=/home/ubuntu/jope-dime/target/release/jope_selfplay_queue /home/ubuntu/jope-dime/scripts/worker.py",
            ]
        ),
    )
    failures = boundary.check(tmp_path, boundary.load_manifest())
    assert any("absolute target" in failure for failure in failures)
    assert any("mutable runtime dependency" in failure for failure in failures)
    assert any("unbundled script" in failure for failure in failures)
