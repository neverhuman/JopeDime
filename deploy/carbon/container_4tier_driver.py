#!/usr/bin/env python3
"""Clean-room container driver executed inside unmounted ubuntu:24.04 for Carbon CI E2E."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

FORBIDDEN_ENV_VARS: tuple[str, ...] = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_PROFILE",
    "AWS_DEFAULT_REGION",
    "AWS_REGION",
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "CARBON_TOKEN",
    "CARBON_AUTH_TOKEN",
    "CARBON_AUTH_SECRET_HEX",
    "CARBON_SEAL_SECRET_HEX",
    "CARBON_PRESIGN_SECRET_HEX",
)

FORBIDDEN_PRE_PATHS: tuple[str, ...] = (
    "/root/.aws",
    "/root/.ssh",
    "/root/.config/gh",
    "/root/.config/jopedime",
    "/root/.netrc",
)

INSTALL_CMD = "curl -fsSL https://downloads.neverhuman.org/carbon/install.sh | sh"
CARBON_BIN = "/root/.local/bin/carbon"
CONFIG_DIR = Path("/root/.config/jopedime")
CSV_PATH = Path("/root/nonlinear_240x6.csv")
EMU_CLOUDS_PATH = Path("/root/emu-clouds.json")


def sha256_file(path: Path) -> str:
    """Compute lowercase SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def read_os_pretty_name() -> str:
    """Read PRETTY_NAME from /etc/os-release inside the container."""
    os_rel = Path("/etc/os-release")
    if not os_rel.is_file():
        return "Ubuntu 24.04 LTS"
    for line in os_rel.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("PRETTY_NAME="):
            return line.split("=", 1)[1].strip().strip('"')
    return "Ubuntu 24.04 LTS"


def run_cmd(
    args: list[str],
    timeout_s: int = 300,
    stdin_text: str | None = None,
) -> tuple[int, str, str, float]:
    """Execute a subprocess without a shell and return (rc, stdout, stderr, wall_ms)."""
    t0 = time.monotonic()
    proc = subprocess.run(
        args,
        input=stdin_text,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_s,
    )
    wall_ms = (time.monotonic() - t0) * 1000.0
    return proc.returncode, proc.stdout, proc.stderr, wall_ms


def verify_clean_room_pre_install() -> dict[str, Any]:
    """Assert zero AWS, GitHub, or Carbon credentials exist before install.sh runs."""
    leaked_vars = [k for k in FORBIDDEN_ENV_VARS if os.environ.get(k)]
    present_paths = [p for p in FORBIDDEN_PRE_PATHS if Path(p).exists()]
    clean = len(leaked_vars) == 0 and len(present_paths) == 0
    return {
        "os_pretty_name": read_os_pretty_name(),
        "uname": f"{platform.system()} {platform.release()} {platform.machine()}",
        "zero_credentials_pre_install": clean,
        "leaked_env_vars": leaked_vars,
        "present_paths_pre_install": present_paths,
        "verified_absent_env_vars": [
            k for k in FORBIDDEN_ENV_VARS if not k.endswith("_SECRET_HEX")
        ],
        "verified_absent_paths_pre_install": list(FORBIDDEN_PRE_PATHS),
    }


def execute_production_install_and_checks() -> dict[str, Any]:
    """Install production Carbon CLI via downloads.neverhuman.org and verify onboarding."""
    rc_dl, out_dl, err_dl, dl_ms = run_cmd(
        ["curl", "-fsSL", "https://downloads.neverhuman.org/carbon/install.sh"],
        timeout_s=60,
    )
    if rc_dl != 0 or not out_dl.strip():
        raise RuntimeError(f"download install.sh failed (rc={rc_dl}): {err_dl.strip()}")
    rc_sh, _out_sh, err_sh, sh_ms = run_cmd(["sh"], timeout_s=120, stdin_text=out_dl)
    if rc_sh != 0:
        raise RuntimeError(f"install.sh execution failed (rc={rc_sh}): {err_sh.strip()}")

    bin_path = Path(CARBON_BIN)
    clouds_path = CONFIG_DIR / "carbon-clouds.json"
    if not bin_path.is_file() or not clouds_path.is_file():
        raise RuntimeError("install.sh did not produce binary and carbon-clouds.json")

    cfg_mode = oct(CONFIG_DIR.stat().st_mode & 0o777)
    rc_sv, out_sv, err_sv, _ = run_cmd([CARBON_BIN, "self-verify"], timeout_s=30)
    if rc_sv != 0:
        raise RuntimeError(f"carbon self-verify failed: {err_sv.strip()}")
    sv = json.loads(out_sv)

    rc_doc, out_doc, err_doc, _ = run_cmd(
        [CARBON_BIN, "doctor", "--cloud", "all"], timeout_s=60
    )
    if rc_doc != 0:
        raise RuntimeError(f"carbon doctor failed: {err_doc.strip()}")
    doc = json.loads(out_doc)

    rc_qs, out_qs, err_qs, qs_wall_ms = run_cmd([CARBON_BIN, "quickstart"], timeout_s=180)
    if rc_qs != 0:
        raise RuntimeError(f"carbon quickstart failed: {err_qs.strip()}")
    qs = json.loads(out_qs)

    rc_csv, _, err_csv, _ = run_cmd(
        [CARBON_BIN, "quickstart", "--emit-csv", str(CSV_PATH), "--csv-only"],
        timeout_s=30,
    )
    if rc_csv != 0 or not CSV_PATH.is_file():
        raise RuntimeError(f"emit nonlinear_240x6.csv failed: {err_csv.strip()}")

    rc_gate, out_gate, err_gate, _ = run_cmd(
        [
            CARBON_BIN,
            "run",
            "--csv",
            str(CSV_PATH),
            "--target",
            "target",
            "--cloud",
            "prod",
            "--free",
            "--effort",
            "high",
            "--shred",
        ],
        timeout_s=60,
    )
    gate_combined = f"{out_gate}\n{err_gate}"
    gate_status = 403 if (rc_gate != 0 and ("403" in gate_combined or "Pro" in gate_combined or "pro" in gate_combined)) else 0

    return {
        "command": INSTALL_CMD,
        "install_wall_ms": round(dl_ms + sh_ms, 2),
        "installed_binary_path": CARBON_BIN,
        "installed_binary_bytes": bin_path.stat().st_size,
        "installed_binary_sha256": sha256_file(bin_path),
        "installed_clouds_json_sha256": sha256_file(clouds_path),
        "config_dir_mode": cfg_mode,
        "release_tag": str(sv.get("release_tag", "")),
        "self_verify_verified": bool(sv.get("verified")),
        "manifest_ecdsa_verified": bool(sv.get("manifest_ecdsa_verified")),
        "kek_pins_verified": int(sv.get("kek_pins_verified", 0)),
        "doctor_healthy": bool(doc.get("healthy")),
        "doctor_dev_healthy_regions": int(
            doc.get("clouds", {}).get("dev", {}).get("healthy_regions", 0)
        ),
        "doctor_prod_healthy_regions": int(
            doc.get("clouds", {}).get("prod", {}).get("healthy_regions", 0)
        ),
        "quickstart_passed": qs.get("final_state") == "completed",
        "quickstart_run_id": str(qs.get("run_id", "")),
        "quickstart_wall_ms": round(float(qs.get("total_wall_ms", qs_wall_ms)), 2),
        "quickstart_holdout_r2": float(qs.get("holdout_r2", 0.0)),
        "quickstart_encrypted_cuen": bool(qs.get("encrypted_cuen")),
        "quickstart_shredded": bool(qs.get("shredded_after_run")),
        "live_anon_high_gate_status": gate_status,
    }


def login_stdin_token(cloud: str, token: str, clouds_manifest: str | None) -> None:
    """Store a capability token via stdin without placing secrets in argv."""
    cmd = [CARBON_BIN, "login", "--cloud", cloud, "--token-file", "-"]
    if clouds_manifest:
        cmd.extend(["--clouds-manifest", clouds_manifest])
    rc, _out, err, _ = run_cmd(cmd, timeout_s=30, stdin_text=token.strip() + "\n")
    if rc != 0:
        raise RuntimeError(f"carbon login --cloud {cloud} failed (rc={rc}): {err.strip()}")


def execute_single_run(
    cloud: str,
    effort: str,
    tier_flag: str,
    execution_target: str,
    clouds_manifest: str | None,
) -> dict[str, Any]:
    """Run one encrypted E2E Carbon job with --shred and normalize cost + timing telemetry."""
    cmd = [
        CARBON_BIN,
        "run",
        "--csv",
        str(CSV_PATH),
        "--target",
        "target",
        "--cloud",
        cloud,
        tier_flag,
        "--effort",
        effort,
        "--seed",
        "42",
        "--predict-rows",
        "5",
        "--shred",
    ]
    if clouds_manifest:
        cmd.extend(["--clouds-manifest", clouds_manifest])
    rc, out, err, container_wall_ms = run_cmd(cmd, timeout_s=900)
    if rc != 0:
        raise RuntimeError(
            f"carbon run {cloud}/{effort} ({execution_target}) failed (rc={rc}): {err.strip()}"
        )
    r = json.loads(out)
    c = r.get("cost_estimate") or {}
    eval_calls = max(int(r.get("eval_invocations", 0)), int(r.get("eval_shards", 0)))
    return {
        "cloud": cloud,
        "effort": effort,
        "execution_target": execution_target,
        "tier_applied": str(r.get("tier_applied", "")),
        "execution_mode": str(r.get("execution_mode", "")),
        "region": str(r.get("selected_region", "")),
        "run_id": str(r.get("run_id", "")),
        "final_state": str(r.get("final_state", "")),
        "encrypted_cuen": bool(r.get("encrypted_cuen")),
        "shredded_after_run": bool(r.get("shredded_after_run")),
        "waves_completed": int(r.get("waves_completed", 0)),
        "waves_target": int(r.get("waves_target", 0)),
        "candidates_evaluated": int(r.get("candidates_evaluated", 0)),
        "eval_shards": eval_calls,
        "poll_requests": int(r.get("poll_requests", 0)),
        "prep_ms": float(r.get("prep_ms", 0.0)),
        "upload_ms": float(r.get("upload_ms", 0.0)),
        "search_ms": float(r.get("search_ms", 0.0)),
        "predict_ms": float(r.get("predict_ms", 0.0)),
        "total_wall_ms": float(r.get("total_wall_ms", 0.0)),
        "container_wall_ms": round(container_wall_ms, 2),
        "holdout_rmse": float(r.get("outer_holdout_score", 0.0)),
        "holdout_r2": float(r.get("holdout_r2", 0.0)),
        "mean_baseline_rmse": float(r.get("mean_baseline_rmse", 0.0)),
        "linear_baseline_rmse": float(r.get("linear_baseline_rmse", 0.0)),
        "linear_baseline_r2": float(r.get("linear_baseline_r2", 0.0)),
        "rmse_reduction_vs_linear_pct": float(r.get("rmse_reduction_vs_linear_pct", 0.0)),
        "cost_estimate": {
            "apigw_http_api_usd": float(c.get("apigw_http_api_usd", 0.0)),
            "lambda_edge_usd": float(c.get("lambda_edge_usd", 0.0)),
            "lambda_engine_usd": float(c.get("lambda_engine_usd", 0.0)),
            "lambda_eval_usd": float(c.get("lambda_eval_usd", 0.0)),
            "s3_requests_usd": float(c.get("s3_requests_usd", 0.0)),
            "s3_storage_and_transfer_usd": float(
                c.get("s3_storage_and_transfer_usd", 0.0)
            ),
            "total_estimated_usd": float(c.get("total_estimated_usd", 0.0)),
            "runs_per_1_usd": float(c.get("runs_per_1_usd", 0.0)),
        },
    }


def execute_all_eight_tiers(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Execute dev & prod across fast, medium, high, and ultra under Option A or Option B."""
    mode = str(cfg["pro_secrets_mode"])
    results: list[dict[str, Any]] = []
    if mode == "live_pro_secrets":
        login_stdin_token("dev", str(cfg["dev_pro_token"]), None)
        login_stdin_token("prod", str(cfg["prod_pro_token"]), None)
        for effort in ("fast", "medium", "high", "ultra"):
            with ThreadPoolExecutor(max_workers=2) as pool:
                f_dev = pool.submit(
                    execute_single_run, "dev", effort, "--pro", "aws_live", None
                )
                f_prod = pool.submit(
                    execute_single_run, "prod", effort, "--pro", "aws_live", None
                )
                results.extend([f_dev.result(), f_prod.result()])
        return results

    for cloud in ("dev", "prod"):
        rc, _o, err, _ = run_cmd(
            [CARBON_BIN, "register", "--free", "--cloud", cloud, "--label", f"ci_{cloud}"],
            timeout_s=60,
        )
        if rc != 0:
            raise RuntimeError(f"carbon register --free --cloud {cloud} failed: {err.strip()}")

    for effort in ("fast", "medium"):
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_dev = pool.submit(
                execute_single_run, "dev", effort, "--free", "aws_live", None
            )
            f_prod = pool.submit(
                execute_single_run, "prod", effort, "--free", "aws_live", None
            )
            results.extend([f_dev.result(), f_prod.result()])

    EMU_CLOUDS_PATH.write_text(str(cfg["emu_clouds_json"]), encoding="utf-8")
    os.chmod(EMU_CLOUDS_PATH, 0o600)
    login_stdin_token("dev", str(cfg["dev_pro_token"]), str(EMU_CLOUDS_PATH))
    login_stdin_token("prod", str(cfg["prod_pro_token"]), str(EMU_CLOUDS_PATH))
    for effort in ("high", "ultra"):
        for cloud in ("dev", "prod"):
            results.append(
                execute_single_run(
                    cloud, effort, "--pro", "ci_emulator", str(EMU_CLOUDS_PATH)
                )
            )
    return results


def run_container_suite(cfg: dict[str, Any]) -> dict[str, Any]:
    """Top-level clean-room container entrypoint."""
    pre = verify_clean_room_pre_install()
    if not pre["zero_credentials_pre_install"]:
        raise RuntimeError(f"clean-room pre-check failed: {pre}")
    inst = execute_production_install_and_checks()
    try:
        runs = execute_all_eight_tiers(cfg)
    finally:
        if Path(CARBON_BIN).is_file():
            run_cmd([CARBON_BIN, "logout", "--all"], timeout_s=20)
        if EMU_CLOUDS_PATH.exists():
            EMU_CLOUDS_PATH.unlink()
    cred_file = CONFIG_DIR / "carbon-credentials.json"
    pre["post_logout_credentials_absent"] = not cred_file.exists()
    first = runs[0]
    dataset = {
        "name": "nonlinear_240x6.csv",
        "rows": 240,
        "fit_rows": 192,
        "holdout_rows": 48,
        "features": 6,
        "bytes": CSV_PATH.stat().st_size,
        "sha256": sha256_file(CSV_PATH),
        "mean_baseline_rmse": first["mean_baseline_rmse"],
        "ridge_linear_baseline_rmse": first["linear_baseline_rmse"],
        "ridge_linear_baseline_r2": first["linear_baseline_r2"],
    }
    return {"container": pre, "install": inst, "dataset": dataset, "runs": runs}


if __name__ == "__main__":
    active_cfg = globals().get("DRIVER_CFG")
    if active_cfg is None:
        active_cfg = json.loads(sys.stdin.read())
    sys.stdout.write(json.dumps(run_container_suite(active_cfg)) + "\n")
