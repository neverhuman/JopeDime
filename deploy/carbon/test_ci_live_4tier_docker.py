#!/usr/bin/env python3
"""Stdlib unit tests for `ci-live-4tier-docker.py` (Option B, Option A, and secret refusal)."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve().with_name("ci-live-4tier-docker.py")


def load_driver_module() -> Any:
    """Import `ci-live-4tier-docker.py` cleanly under stdlib importlib."""
    spec = importlib.util.spec_from_file_location("ci_live_4tier_docker", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load ci-live-4tier-docker.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DRIVER = load_driver_module()


def make_fake_suite_output(mode: str, leak_marker: str = "") -> dict[str, Any]:
    """Construct a deterministic 8-run container suite output payload."""
    runs: list[dict[str, Any]] = []
    for effort, waves, shards in (
        ("fast", 1, 0),
        ("medium", 4, 0),
        ("high", 12, 120),
        ("ultra", 32, 480),
    ):
        target = (
            "aws_live"
            if (mode == "live_pro_secrets" or effort in ("fast", "medium"))
            else "ci_emulator"
        )
        tier = "pro" if mode == "live_pro_secrets" or effort in ("high", "ultra") else "free"
        exec_mode = "SoloSingleLambda" if shards == 0 else "HybridFanOut"
        for cloud in ("dev", "prod"):
            usd = round(0.00035 + waves * 0.00014, 6)
            runs.append(
                {
                    "cloud": cloud,
                    "effort": effort,
                    "execution_target": target,
                    "tier_applied": tier,
                    "execution_mode": exec_mode,
                    "region": "us-east-2",
                    "run_id": f"run_{cloud}_{effort}_01{leak_marker}",
                    "final_state": "completed",
                    "encrypted_cuen": True,
                    "shredded_after_run": True,
                    "waves_completed": waves,
                    "waves_target": waves,
                    "candidates_evaluated": 25 + waves * 4,
                    "eval_shards": shards,
                    "poll_requests": waves,
                    "prep_ms": 15.0,
                    "upload_ms": 180.0,
                    "search_ms": 3800.0 + waves * 800.0,
                    "predict_ms": 45.0,
                    "total_wall_ms": 4040.0 + waves * 800.0,
                    "container_wall_ms": 4090.0 + waves * 800.0,
                    "holdout_rmse": 0.0975,
                    "holdout_r2": 0.99961,
                    "mean_baseline_rmse": 6.21,
                    "linear_baseline_rmse": 5.08,
                    "linear_baseline_r2": 0.331,
                    "rmse_reduction_vs_linear_pct": 98.08,
                    "cost_estimate": {
                        "apigw_http_api_usd": 0.000008,
                        "lambda_edge_usd": 0.000004,
                        "lambda_engine_usd": round(usd * 0.65, 6),
                        "lambda_eval_usd": round(usd * 0.30, 6),
                        "s3_requests_usd": 0.000015,
                        "s3_storage_and_transfer_usd": 0.000001,
                        "total_estimated_usd": usd,
                        "runs_per_1_usd": round(1.0 / usd, 1),
                    },
                }
            )
    return {
        "container": {
            "os_pretty_name": "Ubuntu 24.04.5 LTS",
            "uname": "Linux 6.8.0 x86_64",
            "zero_credentials_pre_install": True,
            "post_logout_credentials_absent": True,
            "verified_absent_env_vars": ["AWS_ACCESS_KEY_ID", "GITHUB_TOKEN"],
            "verified_absent_paths_pre_install": ["/root/.aws", "/root/.config/jopedime"],
        },
        "install": {
            "command": "curl -fsSL https://downloads.neverhuman.org/carbon/install.sh | sh",
            "install_wall_ms": 1620.0,
            "installed_binary_path": "/root/.local/bin/carbon",
            "installed_binary_bytes": 3763432,
            "installed_binary_sha256": "b" * 64,
            "installed_clouds_json_sha256": "e" * 64,
            "config_dir_mode": "0o700",
            "release_tag": "carbon-cli-v1.1.0",
            "self_verify_verified": True,
            "manifest_ecdsa_verified": True,
            "kek_pins_verified": 6,
            "doctor_healthy": True,
            "doctor_dev_healthy_regions": 1,
            "doctor_prod_healthy_regions": 5,
            "quickstart_passed": True,
            "quickstart_run_id": "run_qs_01",
            "quickstart_wall_ms": 4890.0,
            "quickstart_holdout_r2": 0.99982,
            "quickstart_encrypted_cuen": True,
            "quickstart_shredded": True,
            "live_anon_high_gate_status": 403,
        },
        "dataset": {
            "name": "nonlinear_240x6.csv",
            "rows": 240,
            "fit_rows": 192,
            "holdout_rows": 48,
            "features": 6,
            "bytes": 15938,
            "sha256": "3" * 64,
        },
        "runs": runs,
    }


def write_mock_scripts(base: Path) -> tuple[Path, Path]:
    """Create stub `docker` and `carbon-emu` executables inside `base`."""
    docker_stub = base / "mock-docker"
    docker_stub.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "if sys.argv[1:3] == ['image', 'inspect']:\n"
        "    print(json.dumps([{'Id': 'sha256:' + 'a'*64, 'RepoDigests': ['ubuntu@sha256:' + 'a'*64]}]))\n"
        "elif sys.argv[1] == 'build':\n"
        "    sys.stdin.read(); print('sha256:' + 'c'*64)\n"
        "elif sys.argv[1] == 'run':\n"
        "    payload = json.loads(sys.stdin.read())\n"
        "    print(json.dumps({'received_mode': payload['cfg']['pro_secrets_mode']}))\n",
        encoding="utf-8",
    )
    docker_stub.chmod(0o755)

    emu_stub = base / "mock-carbon-emu"
    emu_stub.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys, time\n"
        "wd = pathlib.Path(sys.argv[sys.argv.index('--work-dir') + 1])\n"
        "(wd / 'emu-clouds.json').write_text('{\"schema\":\"jopedime.carbon.clouds.v1\"}')\n"
        "(wd / 'emu-ready.json').write_text(json.dumps({'dev_pro_token': 'emu_dev_tok_xyz', 'prod_pro_token': 'emu_prod_tok_xyz'}))\n"
        "while not (wd / 'emu-stop').exists():\n"
        "    time.sleep(0.02)\n"
        "(wd / 'emu-ready.json').unlink(missing_ok=True)\n"
        "(wd / 'emu-telemetry.json').write_text(json.dumps({'total_residual_cas_blobs': 0}))\n",
        encoding="utf-8",
    )
    emu_stub.chmod(0o755)
    return docker_stub, emu_stub


class CiLive4TierDockerTests(unittest.TestCase):
    """Conformance tests for Option B, Option A, and secret/hostname leak refusal."""

    def setUp(self) -> None:
        scratch = ROOT / "target" / "tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix="ci-live-4tier-", dir=scratch)
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.docker_stub, self.emu_stub = write_mock_scripts(self.base)

    def test_option_b_zero_secrets_hybrid_succeeds_and_writes_manifest(self) -> None:
        out_dir = self.base / "out-b"
        sha = "f" * 40
        fake_suite = make_fake_suite_output("zero_secrets_hybrid_live_and_emulator")
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("CARBON_CI_")
        }
        env["GITHUB_RUN_ID"] = "99001"
        env["GITHUB_RUN_ATTEMPT"] = "2"
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(DRIVER, "run_cleanroom_container", return_value=fake_suite) as run_mock,
        ):
            rc = DRIVER.main(
                [
                    "--source-sha",
                    sha,
                    "--out-dir",
                    str(out_dir),
                    "--carbon-emu-bin",
                    str(self.emu_stub),
                    "--docker-bin",
                    str(self.docker_stub),
                ]
            )
        self.assertEqual(rc, 0)
        cfg_passed = run_mock.call_args.args[2]
        self.assertEqual(
            cfg_passed["pro_secrets_mode"], "zero_secrets_hybrid_live_and_emulator"
        )
        self.assertEqual(cfg_passed["dev_pro_token"], "emu_dev_tok_xyz")
        receipt = json.loads((out_dir / "live-ci-4tier-docker.json").read_text())
        self.assertEqual(receipt["source_sha"], sha)
        self.assertEqual(receipt["github_run_id"], "99001")
        self.assertEqual(receipt["github_run_attempt"], 2)
        self.assertEqual(len(receipt["runs"]), 8)
        self.assertEqual(len(receipt["per_tier_comparison"]), 4)
        self.assertEqual(
            receipt["per_tier_comparison"][0]["execution_target"], "aws_live"
        )
        self.assertEqual(
            receipt["per_tier_comparison"][3]["execution_target"], "ci_emulator"
        )
        self.assertTrue((out_dir / "MANIFEST.sha256").is_file())

    def test_option_a_live_pro_secrets_and_missing_secret_refusal(self) -> None:
        out_dir = self.base / "out-a"
        sha = "e" * 40
        env_missing = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("CARBON_CI_")
        }
        with patch.dict(os.environ, env_missing, clear=True):
            rc_missing = DRIVER.main(
                [
                    "--source-sha",
                    sha,
                    "--out-dir",
                    str(out_dir),
                    "--require-live-pro",
                    "--docker-bin",
                    str(self.docker_stub),
                ]
            )
        self.assertEqual(rc_missing, 2)

        env_live = dict(
            env_missing,
            CARBON_CI_DEV_PRO_TOKEN="live_dev_secret_token_99",
            CARBON_CI_PROD_PRO_TOKEN="live_prod_secret_token_99",
        )
        fake_suite = make_fake_suite_output("live_pro_secrets")
        with (
            patch.dict(os.environ, env_live, clear=True),
            patch.object(DRIVER, "run_cleanroom_container", return_value=fake_suite),
        ):
            rc_ok = DRIVER.main(
                [
                    "--source-sha",
                    sha,
                    "--out-dir",
                    str(out_dir),
                    "--require-live-pro",
                    "--docker-bin",
                    str(self.docker_stub),
                ]
            )
        self.assertEqual(rc_ok, 0)
        receipt = json.loads((out_dir / "live-ci-4tier-docker.json").read_text())
        self.assertEqual(receipt["pro_secrets_mode"], "live_pro_secrets")
        self.assertTrue(
            all(r["execution_target"] == "aws_live" for r in receipt["runs"])
        )

    def test_secret_or_hostname_leak_fails_closed_before_disk_write(self) -> None:
        out_dir = self.base / "out-leak"
        sha = "d" * 40
        leaked = make_fake_suite_output(
            "zero_secrets_hybrid_live_and_emulator", leak_marker="_emu_dev_tok_xyz"
        )
        with patch.object(DRIVER, "run_cleanroom_container", return_value=leaked):
            rc = DRIVER.main(
                [
                    "--source-sha",
                    sha,
                    "--out-dir",
                    str(out_dir),
                    "--carbon-emu-bin",
                    str(self.emu_stub),
                    "--docker-bin",
                    str(self.docker_stub),
                ]
            )
        self.assertEqual(rc, 1)
        self.assertFalse((out_dir / "live-ci-4tier-docker.json").exists())


if __name__ == "__main__":
    unittest.main()
