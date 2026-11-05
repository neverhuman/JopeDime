#!/usr/bin/env python3
"""Stage and verify the Dendrite preview or full OptiX plate render."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time
import uuid


from common import artifact_family, validate_configuration, require_render_enabled, configured_hosts, external_path

ROOT = artifact_family("variable-attribution-v1/dendrite-final-v1")
REMOTE = "/home/ubuntu/jope-data/jain-render-20261103/dendrite-final-v1"
SSH = ("ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=6",
       "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3")
HOSTS = configured_hosts()
CASE_IDS = tuple(f"case-{letter}" for letter in "abcdefghij")
CASE_SPANS = {cid: (index * 120 + 1, (index + 1) * 120) for index, cid in enumerate(CASE_IDS)}
PREVIEW_FRAMES = {"case-a": 89, "case-g": 810}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(message):
    print(f"{now()} {message}", flush=True)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    temporary = path.with_suffix(".tmp")
    external_path(temporary).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    external_path(temporary).replace(external_path(path))


def inputs():
    blend = external_path(ROOT / "scene/scene.blend")
    receipt_path = external_path(ROOT / "scene/scene.receipt.json")
    receipt = read_json(receipt_path)
    ids = receipt.get("case_ids") or [table["case_id"] for table in receipt["tables"]]
    if (tuple(ids) != CASE_IDS or [tuple(span) for span in receipt["case_frames"]] != list(CASE_SPANS.values())
            or receipt["network"]["spatial_layout"] != "dendrite" or receipt["fps"] != 30):
        raise ValueError("Expected the ten-case, 30 fps Dendrite scene")
    return {"scene_sha256": digest(blend), "scene_receipt_sha256": digest(receipt_path),
            "guard_sha256": digest(Path(__file__).with_name("gpu_case_worker.sh"))}


def run(command, *, timeout, retry=(), attempts=1):
    for attempt in range(attempts):
        try:
            result = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            if attempt + 1 == attempts:
                raise
        else:
            if result.returncode == 0:
                return result.stdout.strip()
            if result.returncode not in retry or attempt + 1 == attempts:
                raise RuntimeError(f"{shlex.join(command)} returned {result.returncode}:\n"
                                   f"{result.stdout}{result.stderr}")
        time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable retry state")


def ssh(host, command):
    require_render_enabled()
    return run((*SSH, host, command), timeout=40, retry=(255,), attempts=3)


def rsync(source, destination):
    require_render_enabled()
    return run(("rsync", "-a", "--partial", "--partial-dir=.rsync-partial", "--timeout=60",
                "-e", shlex.join(SSH), *source, str(destination)),
               timeout=330, retry=(10, 12, 23, 24, 30, 35, 255), attempts=3)


def revision(binding):
    return f"{binding['scene_sha256'][:12]}-{binding['scene_receipt_sha256'][:12]}"


def remote_stage(binding):
    return f"{REMOTE}/revisions/{revision(binding)}"


def stage(host, binding):
    remote = remote_stage(binding)
    ssh(host, f"umask 077; mkdir -p -- {remote}/scene {remote}/jobs")
    rsync([str(external_path(ROOT / "scene/scene.blend")), str(external_path(ROOT / "scene/scene.receipt.json"))], f"{host}:{remote}/scene/")
    rsync([str(Path(__file__).with_name("gpu_case_worker.sh"))], f"{host}:{remote}/")
    checks = "\n".join(f"{binding[key]}  {remote}/{path}" for key, path in (
        ("scene_sha256", "scene/scene.blend"), ("scene_receipt_sha256", "scene/scene.receipt.json"),
        ("guard_sha256", "gpu_case_worker.sh"),
    ))
    ssh(host, f"printf '%s\\n' {shlex.quote(checks)} | sha256sum -c")
    log(f"{host} scene and guard staged and verified")


def verify_plate(directory, mode, cid, binding):
    path = directory / "render.receipt.json"
    receipt = read_json(path)
    first, last = (PREVIEW_FRAMES[cid],) * 2 if mode == "preview" else CASE_SPANS[cid]
    required = {
        "schema": "jain.gpu-scene-loop/v1", "status": "PASS", "case_id": cid,
        "scene_sha256": binding["scene_sha256"], "scene_receipt_sha256": binding["scene_receipt_sha256"],
        "backend": "OPTIX", "gpu_only": True, "cpu_fallback": False,
        "width": 1920, "height": 1080, "fps": 30, "samples": 8,
        "frame_start": first, "frame_end": last, "complete_case": mode == "loops",
    }
    for key, value in required.items():
        if receipt.get(key) != value:
            raise ValueError(f"{path}: expected {key}={value!r}, found {receipt.get(key)!r}")
    if [entry["frame"] for entry in receipt["frames"]] != list(range(first, last + 1)):
        raise ValueError(f"{path}: frames missing or reordered")
    for frame in receipt["frames"]:
        name = f"frame-{frame['frame']:06d}.png"
        if frame["file"] != name or digest(directory / name) != frame["sha256"]:
            raise ValueError(f"{directory / name}: receipt mismatch")
    return {"case_id": cid, "frame_count": last - first + 1, "render_receipt_sha256": digest(path),
            "devices": receipt["devices"], "render_elapsed_ns": receipt["elapsed_ns"]}


def destination(mode, cid):
    if mode == "preview":
        return external_path(ROOT / ("preview-plate" if cid == "case-a" else "preview-plate-g"))
    return external_path(ROOT / f"loops/{cid}")


def launch(host, mode, cid, binding):
    job = f"{mode}-{cid}-{uuid.uuid4().hex[:12]}"
    remote = remote_stage(binding)
    remote_output = f"{remote}/{mode}/{cid}"
    job_dir = f"{remote}/jobs/{job}"
    guard = shlex.join(("env", "JAIN_ENABLE_REMOTE_RENDER=1", "bash", f"{remote}/gpu_case_worker.sh", mode, cid,
                       binding["scene_sha256"], binding["scene_receipt_sha256"], job))
    result = ssh(host, f"""set -eu
if test -f {remote_output}/render.receipt.json; then
    python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"]=="PASS"' \
        {remote_output}/render.receipt.json
    echo complete
elif mkdir -m 700 -- {job_dir} 2>/dev/null; then
    test ! -e {remote_output}
    nohup {guard} </dev/null >{job_dir}/render.log 2>&1 &
    echo started
else
    test -d {job_dir}
    echo claimed
fi""")
    if result == "complete":
        log(f"{host} {cid} reusing its completed remote render")
        return None
    log(f"{host} {cid} {result} ({job})")
    return job_dir


def wait_case(host, cid, job_dir):
    if job_dir is None:
        return
    deadline = time.monotonic() + 9 * 60
    while time.monotonic() < deadline:
        status = ssh(host, f"if test -s {job_dir}/status; then cat -- {job_dir}/status; fi")
        if status == "0":
            log(f"{host} {cid} rendered")
            return
        if status:
            last_log = ssh(host, f"tail -50 -- {job_dir}/render.log")
            raise RuntimeError(f"{host} {cid} failed with exit {status}:\n{last_log}")
        time.sleep(10)
    raise TimeoutError(f"{host} {cid} did not finish in 9 minutes. Check {job_dir}/render.log")


def transfer(host, mode, cid, binding):
    output = destination(mode, cid)
    incoming = external_path(ROOT / "transfer" / revision(binding) / host / mode / cid)
    external_path(incoming).mkdir(parents=True, exist_ok=True)
    rsync([f"{host}:{remote_stage(binding)}/{mode}/{cid}/"], incoming)
    proof = verify_plate(incoming, mode, cid, binding)
    external_path(output.parent).mkdir(parents=True, exist_ok=True)
    external_path(incoming).rename(external_path(output))
    log(f"{host} {cid} transferred and verified ({proof['frame_count']} frames)")
    return {**proof, "host": host, "path": str(output)}


def render_host(host, mode, cases, binding, transfer_pool):
    verified = []
    pending = [cid for cid in cases if not destination(mode, cid).exists()]
    for cid in cases:
        path = destination(mode, cid)
        if cid not in pending:
            verified.append({**verify_plate(path, mode, cid, binding), "host": host, "path": str(path)})
    if not pending:
        return verified
    stage(host, binding)
    transfers = []
    for cid in pending:
        job_dir = launch(host, mode, cid, binding)
        wait_case(host, cid, job_dir)
        transfers.append(transfer_pool.submit(transfer, host, mode, cid, binding))
    verified.extend(future.result() for future in transfers)
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check-inputs", "preview", "full"))
    args = parser.parse_args()
    validate_configuration()
    binding = inputs()
    if args.mode == "check-inputs":
        print(json.dumps(binding, indent=2))
        return
    require_render_enabled()
    mode = "preview" if args.mode == "preview" else "loops"
    assignments = (dict(zip(HOSTS, ((cid,) for cid in PREVIEW_FRAMES), strict=True)) if mode == "preview"
                   else {host: CASE_IDS[index::2] for index, host in enumerate(HOSTS)})
    started = now()
    with (external_path(ROOT / ".render-orchestrator.lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with ThreadPoolExecutor(max_workers=len(assignments)) as hosts, ThreadPoolExecutor(max_workers=2) as transfers:
            futures = [hosts.submit(render_host, host, mode, cases, binding, transfers)
                       for host, cases in assignments.items()]
            results = [item for future in futures for item in future.result()]
        if inputs() != binding:
            raise ValueError("Scene, receipt, or worker guard changed during the render")
        summary = {"schema": "jain.dendrite-render-transfer/v1", "status": "PASS", "mode": mode, **binding,
                   "remote_stage": remote_stage(binding),
                   "started_utc": started, "completed_utc": now(),
                   "frames": sum(item["frame_count"] for item in results),
                   "cases": sorted(results, key=lambda item: item["case_id"])}
        write_json(external_path(ROOT / f"{mode}-transfer.receipt.json"), summary)
        log(f"{mode} complete, {summary['frames']} frames verified")


if __name__ == "__main__":
    main()
