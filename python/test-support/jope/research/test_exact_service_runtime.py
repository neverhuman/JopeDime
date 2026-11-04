from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from test_exact_common import atomic_json, dynamic_queue_targets, sha256_file

ROOT = Path("/home/ubuntu/jope-flywheel/test_exact_gen0")
SCRIPTS = Path(__file__).resolve().parent
SCQ = Path("/home/ubuntu/smart_cluster/target/release/scq")
SCQD = Path("/home/ubuntu/smart_cluster/scripts/scqd.sh")
_wave_process: subprocess.Popen[str] | None = None
_status_process: subprocess.Popen[str] | None = None

def loaded_bundle(root: Path = ROOT) -> str | None:
    process_bundle = os.environ.get("JOPE_LOADED_BUNDLE_SHA256")
    if process_bundle:
        return process_bundle
    pointer = root / "runtime-bundles" / "active.json"
    if not pointer.is_file():
        return None
    value = json.loads(pointer.read_text())
    manifest = Path(value["bundle"])
    if (
        not manifest.is_file()
        or sha256_file(manifest) != value["bundle_manifest_sha256"]
    ):
        raise RuntimeError("active runtime bundle pointer failed verification")
    return str(value["bundle_sha256"])


def run(args: list[str], timeout: int = 3_600) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", f"/home/ubuntu/jope-prime/src:{SCRIPTS}")
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        env=env,
    )


def wave_generation_state() -> dict | None:
    global _wave_process

    if _wave_process is None:
        return None
    returncode = _wave_process.poll()
    if returncode is None:
        temporary = (
            ROOT
            / "manifests"
            / f".strict-wave-*.tmp-{_wave_process.pid}"
        )
        return {
            "pending": True,
            "phase": "generating_wave",
            "pid": _wave_process.pid,
            "manifests_written": sum(
                1
                for path in temporary.parent.glob(temporary.name)
                for _ in path.glob("*.json")
            ),
        }
    stdout, stderr = _wave_process.communicate()
    _wave_process = None
    if returncode:
        raise RuntimeError((stderr or stdout).strip()[-4_000:])
    lines = [line for line in stdout.splitlines() if line.strip()]
    return {
        "pending": False,
        "phase": "wave_committed",
        **(json.loads(lines[-1]) if lines else {}),
    }


def start_wave_generation(proposals: str, cohort: str | None = None) -> dict:
    global _wave_process

    if _wave_process is not None:
        raise RuntimeError("wave generator is already active")
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", f"/home/ubuntu/jope-prime/src:{SCRIPTS}")
    command = [
            sys.executable,
            str(SCRIPTS / "test_exact_wave.py"),
            "--candidates-per-dataset",
            "256",
            "--max-shard",
            "100",
            "--proposals",
            proposals,
        ]
    if cohort is not None:
        command.extend(["--cohort", cohort])
    _wave_process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    return {
        "pending": True,
        "phase": "generating_wave",
        "pid": _wave_process.pid,
        "manifests_written": 0,
    }


def pending_micro_wave_request() -> dict | None:
    request_root = ROOT / "ops" / "micro-wave-requests"
    state_path = ROOT / "ops" / "micro-wave-request-state.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    handled = set(state.get("handled_request_sha256s") or [])
    for path in reversed(sorted(request_root.glob("*.json"))):
        digest = sha256_file(path)
        if digest not in handled:
            return {
                **json.loads(path.read_text()),
                "request": str(path),
                "request_sha256": digest,
            }
    return None


def mark_micro_wave_request(request: dict, disposition: str) -> None:
    state_path = ROOT / "ops" / "micro-wave-request-state.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    handled = list(state.get("handled_request_sha256s") or [])
    digest = request["request_sha256"]
    if digest not in handled:
        handled.append(digest)
    atomic_json(
        state_path,
        {
            "schema": "JopeMicroWaveRequestState.v1",
            "handled_request_sha256s": handled,
            "latest_request": request["request"],
            "latest_request_sha256": digest,
            "latest_disposition": disposition,
            "updated_at_ms": int(time.time() * 1000),
        },
    )


def refresh_status_async(root: Path) -> None:
    global _status_process

    if _status_process is not None:
        if _status_process.poll() is None:
            return
        _status_process = None
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", f"/home/ubuntu/jope-prime/src:{SCRIPTS}")
    _status_process = subprocess.Popen(
        [
            sys.executable,
            str(SCRIPTS / "test_exact_status.py"),
            "--root",
            str(root),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )


def parse_result(result: subprocess.CompletedProcess) -> dict:
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip()[-4_000:])
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1]) if lines else {}


def burn_targets(root: Path = ROOT) -> tuple[int, int, float]:
    state = root / "ops" / "backlog.sqlite"
    values = []
    if state.is_file():
        database = sqlite3.connect(state)
        try:
            exists = database.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='burn_samples'"
            ).fetchone()
            if exists:
                cutoff = int(time.time() * 1000) - 60 * 60_000
                values = [
                    float(row[0])
                    for row in database.execute(
                        "SELECT jobs_per_minute FROM burn_samples "
                        "WHERE observed_at_ms>=? ORDER BY observed_at_ms",
                        (cutoff,),
                    )
                ]
        finally:
            database.close()
    from test_exact_backlog import cluster_slot_inventory

    capacity = cluster_slot_inventory()
    return dynamic_queue_targets(values, capacity["effective_cluster_slots"])


