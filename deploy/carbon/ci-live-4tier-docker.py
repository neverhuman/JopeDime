#!/usr/bin/env python3
"""Host orchestrator for Carbon Fresh-Docker production CLI install + dev/prod 4-tier CI E2E."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any

SCHEMA = "jopedime.carbon.live-ci-4tier-docker.v1"
CLEANROOM_TAG = "carbon-ci-cleanroom:ubuntu2404"
CLEANROOM_DOCKERFILE = """\
FROM ubuntu:24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \\
    ca-certificates curl openssl python3 zstd && rm -rf /var/lib/apt/lists/*
"""

FORBIDDEN_RECEIPT_MARKERS: tuple[str, ...] = (
    "ctk" + "1.",
    "AK" + "IA",
    "AS" + "IA",
    "BEGIN " + "PRIVATE KEY",
    "BEGIN EC " + "PRIVATE KEY",
    "CARBON_AUTH_SECRET",
    "CARBON_SEAL_SECRET",
    "CARBON_PRESIGN_SECRET",
    "xbabe",
    "atomicsoul",
    "/" + "home/",
)

STRIPPED_CHILD_ENV_PREFIXES: tuple[str, ...] = (
    "AWS_",
    "CARBON_",
    "GH_",
    "GITHUB_TOKEN",
)


def sanitized_host_env() -> dict[str, str]:
    """Return a copy of os.environ with AWS, GitHub, and Carbon credentials removed."""
    return {
        k: v
        for k, v in os.environ.items()
        if not any(k == p or k.startswith(p) for p in STRIPPED_CHILD_ENV_PREFIXES)
    }


def run_checked(
    args: list[str],
    timeout_s: int = 900,
    stdin_text: str | None = None,
) -> str:
    """Run a child command without a shell and return stdout on zero exit."""
    proc = subprocess.run(
        args,
        input=stdin_text,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_s,
        env=sanitized_host_env(),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"command {args[0]} failed (rc={proc.returncode}): {proc.stderr.strip()}"
        )
    return proc.stdout


def inspect_and_build_cleanroom_image(docker_bin: str) -> tuple[str, str]:
    """Ensure ubuntu:24.04 is pulled, inspect its immutable digest, and build clean-room layer."""
    probe = subprocess.run(
        [docker_bin, "image", "inspect", "ubuntu:24.04"],
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
        env=sanitized_host_env(),
    )
    if probe.returncode != 0:
        run_checked([docker_bin, "pull", "ubuntu:24.04"], timeout_s=300)
        raw_inspect = run_checked(
            [docker_bin, "image", "inspect", "ubuntu:24.04"], timeout_s=60
        )
    else:
        raw_inspect = probe.stdout

    info_list = json.loads(raw_inspect)
    if not isinstance(info_list, list) or not info_list:
        raise RuntimeError("docker image inspect ubuntu:24.04 returned empty payload")
    info = info_list[0]
    image_id = str(info.get("Id", ""))
    digests = info.get("RepoDigests") or []
    repo_digest = str(digests[0]) if digests else f"ubuntu@{image_id}"
    if not image_id.startswith("sha256:"):
        raise RuntimeError(f"unexpected ubuntu:24.04 image Id: {image_id}")

    run_checked(
        [docker_bin, "build", "-q", "-t", CLEANROOM_TAG, "-"],
        timeout_s=300,
        stdin_text=CLEANROOM_DOCKERFILE,
    )
    return image_id, repo_digest


class DualCellEmulatorContext:
    """Context manager for isolated dev + prod CarbonLocalLambdaEmulator cells on localhost."""

    def __init__(self, carbon_emu_bin: Path, work_dir: Path) -> None:
        self.carbon_emu_bin = carbon_emu_bin
        self.work_dir = work_dir
        self.proc: subprocess.Popen[str] | None = None
        self.ready: dict[str, Any] = {}
        self.emu_clouds_json = ""
        self.telemetry: dict[str, Any] = {"total_residual_cas_blobs": 0}

    def start(self) -> None:
        """Spawn `carbon emu --serve-dual-cells` and wait for `emu-ready.json`."""
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            [
                str(self.carbon_emu_bin),
                "emu",
                "--work-dir",
                str(self.work_dir),
                "--serve-dual-cells",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=sanitized_host_env(),
        )
        ready_path = self.work_dir / "emu-ready.json"
        clouds_path = self.work_dir / "emu-clouds.json"
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            if ready_path.is_file() and clouds_path.is_file():
                self.ready = json.loads(ready_path.read_text(encoding="utf-8"))
                self.emu_clouds_json = clouds_path.read_text(encoding="utf-8")
                return
            if self.proc.poll() is not None:
                _out, err = self.proc.communicate(timeout=5)
                raise RuntimeError(
                    f"dual-cell emulator exited early (rc={self.proc.returncode}): {err.strip()}"
                )
            time.sleep(0.05)
        raise RuntimeError("timed out waiting for dual-cell emulator emu-ready.json")

    def stop(self) -> dict[str, Any]:
        """Signal shutdown via `emu-stop`, collect `emu-telemetry.json`, and clean scratch."""
        try:
            if self.proc is not None:
                if self.proc.poll() is None:
                    (self.work_dir / "emu-stop").write_text("stop\n", encoding="utf-8")
                    try:
                        self.proc.communicate(timeout=10)
                    except subprocess.TimeoutExpired:
                        self.proc.kill()
                        self.proc.communicate(timeout=5)
                else:
                    self.proc.communicate(timeout=2)
            telem_path = self.work_dir / "emu-telemetry.json"
            if telem_path.is_file():
                self.telemetry = json.loads(telem_path.read_text(encoding="utf-8"))
        finally:
            if self.work_dir.exists():
                shutil.rmtree(self.work_dir, ignore_errors=True)
        return self.telemetry


def run_cleanroom_container(
    docker_bin: str,
    driver_script: Path,
    driver_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Execute `container_4tier_driver.py` inside an unmounted `env -i` Ubuntu 24.04 container."""
    bootstrap = (
        "import json,sys; p=json.loads(sys.stdin.read()); "
        "g={'__name__':'__main__','DRIVER_CFG':p['cfg']}; "
        "exec(compile(p['code'],'container_4tier_driver.py','exec'),g)"
    )
    payload = json.dumps(
        {"code": driver_script.read_text(encoding="utf-8"), "cfg": driver_cfg}
    )
    cmd = [
        docker_bin,
        "run",
        "--rm",
        "--network",
        "host",
        "-i",
        CLEANROOM_TAG,
        "/usr/bin/env",
        "-i",
        "PATH=/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME=/root",
        "LC_ALL=C.UTF-8",
        "python3",
        "-c",
        bootstrap,
    ]
    out = run_checked(cmd, timeout_s=1200, stdin_text=payload)
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError("clean-room container produced empty stdout")
    return json.loads(lines[-1])


def summarize_tier_side(entry: dict[str, Any]) -> dict[str, Any]:
    """Project a run entry into a compact per-tier comparison cell."""
    cost = entry["cost_estimate"]
    return {
        "run_id": entry["run_id"],
        "region": entry["region"],
        "tier_applied": entry["tier_applied"],
        "execution_mode": entry["execution_mode"],
        "waves_completed": entry["waves_completed"],
        "eval_shards": entry["eval_shards"],
        "total_wall_ms": entry["total_wall_ms"],
        "holdout_r2": entry["holdout_r2"],
        "holdout_rmse": entry["holdout_rmse"],
        "total_estimated_usd": cost["total_estimated_usd"],
        "runs_per_1_usd": cost["runs_per_1_usd"],
    }


def build_per_tier_comparison(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair dev and prod runs across fast, medium, high, and ultra."""
    by_key = {(r["cloud"], r["effort"]): r for r in runs}
    rows: list[dict[str, Any]] = []
    for effort in ("fast", "medium", "high", "ultra"):
        dev = by_key.get(("dev", effort))
        prod = by_key.get(("prod", effort))
        if dev is None or prod is None:
            raise RuntimeError(f"missing dev or prod run for effort={effort}")
        if dev["execution_target"] != prod["execution_target"]:
            raise RuntimeError(f"mismatched execution_target for effort={effort}")
        rows.append(
            {
                "effort": effort,
                "execution_target": dev["execution_target"],
                "waves_target": dev["waves_target"],
                "dev": summarize_tier_side(dev),
                "prod": summarize_tier_side(prod),
            }
        )
    return rows


def verify_receipt_and_scrub(
    receipt: dict[str, Any], sensitive_tokens: list[str]
) -> str:
    """Validate receipt pass gates and assert zero secret or host-path substrings."""
    c = receipt["container"]
    inst = receipt["install"]
    if not c.get("zero_credentials_pre_install") or not c.get(
        "post_logout_credentials_absent"
    ):
        raise RuntimeError(f"container credential hygiene check failed: {c}")
    if (
        not inst.get("self_verify_verified")
        or not inst.get("manifest_ecdsa_verified")
        or int(inst.get("kek_pins_verified", 0)) < 6
    ):
        raise RuntimeError(f"self-verify checks failed: {inst}")
    if (
        not inst.get("doctor_healthy")
        or int(inst.get("doctor_dev_healthy_regions", 0)) < 1
        or int(inst.get("doctor_prod_healthy_regions", 0)) < 5
    ):
        raise RuntimeError(f"doctor checks failed: {inst}")
    if not inst.get("quickstart_passed") or inst.get("live_anon_high_gate_status") != 403:
        raise RuntimeError(f"quickstart or anonymous Pro-gate check failed: {inst}")
    for r in receipt["runs"]:
        if (
            r.get("final_state") != "completed"
            or not r.get("encrypted_cuen")
            or not r.get("shredded_after_run")
            or float(r.get("holdout_r2", 0.0)) <= 0.90
        ):
            raise RuntimeError(f"tier run qualification failed: {r}")
    if int(receipt["emulator_cells"].get("total_residual_cas_blobs", 1)) != 0:
        raise RuntimeError(f"residual CAS blobs detected: {receipt['emulator_cells']}")

    rendered = json.dumps(receipt, indent=2) + "\n"
    for marker in FORBIDDEN_RECEIPT_MARKERS:
        if marker in rendered:
            raise RuntimeError(f"forbidden substring {marker!r} detected in receipt")
    for tok in sensitive_tokens:
        if tok and tok in rendered:
            raise RuntimeError("sensitive capability token leaked into receipt")
    return rendered


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI flags for `ci-live-4tier-docker.py`."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-sha", required=True, help="Full 40-char Git commit SHA")
    p.add_argument("--out-dir", required=True, type=Path, help="Output directory")
    p.add_argument(
        "--carbon-emu-bin",
        type=Path,
        default=None,
        help="Path to `carbon` binary for Option B dual-cell emulator",
    )
    p.add_argument(
        "--require-live-pro",
        action="store_true",
        help="Require CARBON_CI_DEV_PRO_TOKEN and CARBON_CI_PROD_PRO_TOKEN (Option A)",
    )
    p.add_argument("--docker-bin", default="docker", help="Docker CLI binary")
    p.add_argument(
        "--driver-script",
        type=Path,
        default=Path(__file__).resolve().with_name("container_4tier_driver.py"),
    )
    return p.parse_args(argv)


def execute(args: argparse.Namespace) -> Path:
    """Run Fresh-Docker install + 8-run dev/prod 4-tier suite and emit SHA-256-bound receipt."""
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_sha):
        raise ValueError(f"--source-sha must be 40 lowercase hex chars: {args.source_sha}")

    dev_pro = os.environ.get("CARBON_CI_DEV_PRO_TOKEN", "").strip()
    prod_pro = os.environ.get("CARBON_CI_PROD_PRO_TOKEN", "").strip()
    if args.require_live_pro and not (dev_pro and prod_pro):
        raise ValueError(
            "--require-live-pro requires both CARBON_CI_DEV_PRO_TOKEN and CARBON_CI_PROD_PRO_TOKEN"
        )

    use_live_pro = bool(dev_pro and prod_pro)
    mode = (
        "live_pro_secrets"
        if use_live_pro
        else "zero_secrets_hybrid_live_and_emulator"
    )
    if not use_live_pro and (
        args.carbon_emu_bin is None or not args.carbon_emu_bin.is_file()
    ):
        raise ValueError(
            "Option B zero-secret mode requires --carbon-emu-bin pointing to a built carbon binary"
        )

    t0 = time.monotonic()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    image_id, repo_digest = inspect_and_build_cleanroom_image(args.docker_bin)

    sensitive_tokens = [t for t in (dev_pro, prod_pro) if t]
    emu_ctx: DualCellEmulatorContext | None = None
    emu_telemetry: dict[str, Any] = {"mode": "not_used", "total_residual_cas_blobs": 0}
    try:
        if use_live_pro:
            driver_cfg: dict[str, Any] = {
                "pro_secrets_mode": mode,
                "dev_pro_token": dev_pro,
                "prod_pro_token": prod_pro,
            }
        else:
            if args.carbon_emu_bin is None:
                raise ValueError("--carbon-emu-bin is required in Option B mode")
            scratch = out_dir.parent / f".carbon-emu-scratch-{os.getpid()}"
            emu_ctx = DualCellEmulatorContext(args.carbon_emu_bin.resolve(), scratch)
            emu_ctx.start()
            emu_dev_tok = str(emu_ctx.ready["dev_pro_token"])
            emu_prod_tok = str(emu_ctx.ready["prod_pro_token"])
            sensitive_tokens.extend([emu_dev_tok, emu_prod_tok])
            driver_cfg = {
                "pro_secrets_mode": mode,
                "dev_pro_token": emu_dev_tok,
                "prod_pro_token": emu_prod_tok,
                "emu_clouds_json": emu_ctx.emu_clouds_json,
            }

        suite_out = run_cleanroom_container(
            args.docker_bin, args.driver_script.resolve(), driver_cfg
        )
    finally:
        if emu_ctx is not None:
            emu_telemetry = emu_ctx.stop()

    total_wall_ms = round((time.monotonic() - t0) * 1000.0, 2)
    container_info = dict(suite_out["container"])
    container_info["base_image"] = "ubuntu:24.04"
    container_info["image_id"] = image_id
    container_info["repo_digest"] = repo_digest

    runs = list(suite_out["runs"])
    cmps = build_per_tier_comparison(runs)
    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    attempt_raw = os.environ.get("GITHUB_RUN_ATTEMPT", "1")
    attempt = int(attempt_raw) if attempt_raw.isdigit() and int(attempt_raw) >= 1 else 1

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "captured_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_sha": args.source_sha,
        "github_run_id": str(run_id),
        "github_run_attempt": attempt,
        "pro_secrets_mode": mode,
        "all_passed": True,
        "total_wall_ms": total_wall_ms,
        "container": container_info,
        "install": suite_out["install"],
        "dataset": suite_out["dataset"],
        "runs": runs,
        "per_tier_comparison": cmps,
        "emulator_cells": emu_telemetry,
    }
    rendered = verify_receipt_and_scrub(receipt, sensitive_tokens)
    receipt_path = out_dir / "live-ci-4tier-docker.json"
    receipt_path.write_text(rendered, encoding="utf-8")
    digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    (out_dir / "MANIFEST.sha256").write_text(
        f"{digest}  live-ci-4tier-docker.json\n", encoding="utf-8"
    )
    return receipt_path


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    try:
        args = parse_args(argv)
        path = execute(args)
    except ValueError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    except (RuntimeError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    sys.stdout.write(f"wrote verified receipt: {path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
