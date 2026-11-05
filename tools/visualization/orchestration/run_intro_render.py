#!/usr/bin/env python3
"""Stage the packed JAIN intro scene, render guarded GPU shards, and verify PNGs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import struct
import subprocess
import time
import uuid


from common import REPO, artifact_family, validate_configuration, require_render_enabled, configured_hosts, external_path

ROOT = artifact_family("jain-intro-v2")
SOURCE = REPO / "tools/visualization/blender"
LOCAL_RUNTIME = Path(os.environ.get("JAIN_BLENDER", "/__JAIN_BLENDER_NOT_CONFIGURED__/blender")).expanduser().resolve()
LOCAL_RUNTIME_SHA256 = "de1e04556f03716d25682dd01cc776d94e512f7ff8673049a486c3e8b7f9b619"
BASE = "/home/ubuntu/jope-data/jain-render-20261103"
RUNTIME = f"{BASE}/runtime/blender-5.2.3-linux-x64/blender"
RUNTIME_SHA256 = "b8b07a190fb25db76aae6de08dbc6c2b3d64e22e82bed6ba90de9545859f3431"
SSH = ("ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=6",
       "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3")
HOSTS = configured_hosts()
PREVIEW = ((28, 145, 198, 249, 306, 329), (83, 168, 220, 270, 316, 330))
FULL = (tuple(range(1, 166)), tuple(range(166, 331)))


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(message):
    print(f"{now()} {message}", flush=True)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(argv, timeout, retry):
    for attempt in range(3):
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            if attempt == 2:
                raise
        else:
            if result.returncode == 0:
                return result.stdout.strip()
            if result.returncode not in retry or attempt == 2:
                raise RuntimeError(f"{shlex.join(argv)} exited {result.returncode}:\n{result.stdout}{result.stderr}")
        time.sleep(2 * (attempt + 1))
    raise RuntimeError("retry loop exited without a result")


def ssh(host, command):
    require_render_enabled()
    return run((*SSH, host, command), 40, {255})


def rsync(sources, target):
    require_render_enabled()
    return run(("rsync", "-a", "--partial", "--partial-dir=.rsync-partial", "--timeout=60",
                "-e", shlex.join(SSH), *map(str, sources), str(target)), 330,
               {10, 12, 23, 24, 30, 35, 255})


def get_inputs():
    paths = {
        "scene_sha256": external_path(ROOT / "scene/intro.blend"),
        "scene_receipt_sha256": external_path(ROOT / "scene/intro.receipt.json"),
        "worker_sha256": SOURCE / "intro_render.py",
        "shared_worker_sha256": SOURCE / "render_animation.py",
    }
    receipt = json.loads(paths["scene_receipt_sha256"].read_text())
    if receipt.get("schema") != "jain.intro-scene/v1":
        raise ValueError("missing the final JAIN intro scene receipt")
    return paths, {key: sha(path) for key, path in paths.items()}


def frame_options(frames):
    if frames == tuple(range(frames[0], frames[-1] + 1)) and len(frames) > 1:
        return ("--frame", f"{frames[0]}:{frames[-1]}")
    return tuple(item for frame in frames for item in ("--frame", str(frame)))


def stage_and_launch(host, paths, binding, remote, job, frames, samples, scale):
    staged = f"{remote}/inputs"
    code = f"{remote}/code/tools/visualization/blender"
    ssh(host, f"umask 077; mkdir -p -- {staged} {code} {remote}/jobs")
    rsync([path for key, path in paths.items() if key.startswith("scene_")], f"{host}:{staged}/")
    rsync([path for key, path in paths.items() if not key.startswith("scene_")], f"{host}:{code}/")
    checks = "\n".join((f"{RUNTIME_SHA256}  {RUNTIME}", *(
        f"{binding[key]}  {staged if key.startswith('scene_') else code}/{path.name}"
        for key, path in paths.items()
    )))
    command = shlex.join(("timeout", "--kill-after=30s", "45m", RUNTIME, "--background",
                          "--factory-startup", "--disable-autoexec", "--threads", "8",
                          "--python-exit-code", "1", "--python", f"{code}/intro_render.py", "--",
                          "--scene", f"{staged}/intro.blend", "--receipt", f"{staged}/intro.receipt.json",
                          "--output-dir", f"{job}/frames", "--samples", str(samples), "--scale", str(scale),
                          *frame_options(frames)))
    guard = f"""set -euo pipefail
umask 077
finish() {{
    code=$?
    printf '%s\\n' "$code" > {job}/status.tmp
    mv -- {job}/status.tmp {job}/status
    exit "$code"
}}
trap finish EXIT
exec 9>{BASE}/jain-intro-gpu.lock
flock -n 9
printf '%s\\n' {shlex.quote(checks)} | sha256sum -c
free_mib="$(timeout 15s nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
compute_pids="$(timeout 15s nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d ' ')"
printf 'host=%s free_mib=%s existing_compute_pids=%s\\n' "$(hostname)" "$free_mib" "$compute_pids"
[[ "$free_mib" =~ ^[0-9]+$ ]]
((free_mib >= 6144))
[[ -z "$compute_pids" ]]
{command}
"""
    result = ssh(host, f"""set -eu
if mkdir -m 700 -- {job} 2>/dev/null; then
    nohup bash -c {shlex.quote(guard)} </dev/null >{job}/render.log 2>&1 &
    echo started
else
    test -d {job}
    echo already-staged
fi""")
    log(f"{host} {result} {len(frames)} frames in {job}")


def wait_job(host, job, total):
    deadline = time.monotonic() + 48 * 60
    reported = 0
    while time.monotonic() < deadline:
        state = ssh(host, f"if test -s {job}/status; then echo done; cat {job}/status; "
                          f"elif test -f {job}/frames/frames.jsonl; then echo running; "
                          f"wc -l < {job}/frames/frames.jsonl; else printf 'running\\n0\\n'; fi")
        status, count = state.splitlines()
        if status == "done":
            if count == "0":
                return
            raise RuntimeError(f"{host} Blender guard/worker exit {count}:\n"
                               + ssh(host, f"tail -50 -- {job}/render.log"))
        if int(count) >= reported + 15:
            log(f"{host} rendered {count}/{total}")
            reported = int(count)
        time.sleep(3 if total < 10 else 10)
    raise TimeoutError(f"{host} render didn't report completion: {job}/render.log")


def verify_shard(path, frames, binding, samples, scale, backend):
    receipt_path = path / "render.receipt.json"
    receipt = json.loads(receipt_path.read_text())
    width, height = 1920 * scale // 100, 1080 * scale // 100
    required = {key: binding[key] for key in ("scene_sha256", "scene_receipt_sha256", "worker_sha256")}
    required.update(schema="jain.intro-gpu-frames/v1", status="PASS", gpu_only=True, cpu_fallback=False,
                    backend=backend, width=width, height=height, fps=30, samples=samples,
                    requested_frames=list(frames))
    if any(receipt.get(key) != value for key, value in required.items()):
        raise ValueError(f"{receipt_path}: invalid render provenance or settings")
    if [record["frame"] for record in receipt["frames"]] != list(frames):
        raise ValueError(f"{receipt_path}: frame set differs")
    for item in receipt["frames"]:
        name = f"frame-{item['frame']:06d}.png"
        png = path / name
        if item["file"] != name or sha(png) != item["sha256"]:
            raise ValueError(f"{png}: wrong filename or SHA-256")
        with png.open("rb") as stream:
            header = stream.read(24)
        if header[:16] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" or struct.unpack(">II", header[16:24]) != (width, height):
            raise ValueError(f"{png}: wrong PNG dimensions")
    return receipt, sha(receipt_path)


def host_render(host, paths, binding, remote, job_key, frames, samples, scale):
    job = f"{remote}/jobs/{job_key}-{host}"
    local = external_path(ROOT / "shards" / job_key / host)
    stage_and_launch(host, paths, binding, remote, job, frames, samples, scale)
    wait_job(host, job, len(frames))
    external_path(local).mkdir(parents=True, exist_ok=True)
    rsync([f"{host}:{job}/frames/"], local)
    receipt, receipt_sha = verify_shard(local, frames, binding, samples, scale, "OPTIX")
    log(f"{host} transferred and verified {len(frames)} frames")
    return local, receipt, {"host": host, "remote_job": job, "receipt_sha256": receipt_sha,
                            "devices": receipt["devices"], "elapsed_ns": receipt["elapsed_ns"],
                            "runtime_sha256": RUNTIME_SHA256}


def local_render(paths, binding, job_key, frames, samples, scale):
    if sha(LOCAL_RUNTIME) != LOCAL_RUNTIME_SHA256:
        raise ValueError("local Blender does not match the pinned Metal runtime")
    local = external_path(ROOT / "shards" / job_key / "local-metal")
    log(f"local Metal started {len(frames)} frames")
    subprocess.run((str(LOCAL_RUNTIME), "--background", "--factory-startup", "--disable-autoexec",
                    "--python-exit-code", "1", "--python", str(paths["worker_sha256"]), "--",
                    "--scene", str(paths["scene_sha256"]), "--receipt", str(paths["scene_receipt_sha256"]),
                    "--output-dir", str(local), "--samples", str(samples), "--scale", str(scale),
                    "--device", "METAL", *frame_options(frames)), check=True)
    receipt, receipt_sha = verify_shard(local, frames, binding, samples, scale, "METAL")
    return local, receipt, {"host": "local-metal", "receipt_sha256": receipt_sha,
                            "devices": receipt["devices"], "elapsed_ns": receipt["elapsed_ns"],
                            "runtime_sha256": LOCAL_RUNTIME_SHA256}


def publish(name, shards, binding, samples, scale, started):
    output = external_path(ROOT / name)
    pending = external_path(ROOT / f".{name}-{uuid.uuid4().hex[:8]}.pending")
    external_path(pending).mkdir(exist_ok=False)
    frames = []
    for path, receipt, _ in shards:
        for item in receipt["frames"]:
            filename = item["file"]
            os.link(path / filename, external_path(pending / filename))
            frames.append(item)
    external_path(pending).rename(external_path(output))
    receipt = {"schema": "jain.intro-plates/v1", "status": "PASS", **binding,
               "samples": samples,
               "width": 1920 * scale // 100, "height": 1080 * scale // 100, "fps": 30,
               "started_utc": started, "completed_utc": now(), "directory": str(output),
               "shards": [proof for _, _, proof in shards],
               "frames": sorted(frames, key=lambda item: item["frame"])}
    receipt_path = external_path(ROOT / f"{name}.receipt.json")
    temporary = pending.with_suffix(".json")
    with external_path(temporary).open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    external_path(temporary).rename(external_path(receipt_path))
    log(f"published {len(frames)} verified plates at {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preview", "full", "hero"))
    parser.add_argument("--samples", type=int)
    parser.add_argument("--expected-scene-sha256")
    parser.add_argument("--expected-scene-receipt-sha256")
    parser.add_argument("--local-metal", action="store_true")
    args = parser.parse_args()
    validate_configuration()
    require_render_enabled(local=args.local_metal, uhd=args.mode == "hero")
    paths, binding = get_inputs()
    if args.expected_scene_sha256 and binding["scene_sha256"] != args.expected_scene_sha256:
        raise ValueError("saved scene changed after preview approval")
    if (args.expected_scene_receipt_sha256
            and binding["scene_receipt_sha256"] != args.expected_scene_receipt_sha256):
        raise ValueError("scene receipt changed after preview approval")
    name, samples, scale, groups = {
        "preview": ("previews", 12, 100, PREVIEW),
        "full": ("frames", 16, 100, FULL),
        "hero": ("heroes-4k", 48, 200, ((28,), (270,))),
    }[args.mode]
    if args.samples is not None:
        samples = args.samples
    if not 1 <= samples <= 4096:
        parser.error("--samples must be in [1, 4096]")
    revision = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()[:20]
    remote = f"{BASE}/jain-intro-v2/revisions/{revision}"
    job_key = f"{revision}-{name}-{samples}s"
    started = now()
    with (external_path(ROOT / ".intro-render.lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (external_path(ROOT / name)).exists() or (external_path(ROOT / f"{name}.receipt.json")).exists():
            raise FileExistsError(f"Preserve the existing {name} before publishing another render")
        if args.local_metal:
            frames = tuple(sorted(frame for group in groups for frame in group))
            shards = [local_render(paths, binding, job_key, frames, samples, scale)]
        else:
            with ThreadPoolExecutor(max_workers=2) as pool:
                tasks = [pool.submit(host_render, host, paths, binding, remote, job_key, frames, samples, scale)
                         for host, frames in zip(HOSTS, groups, strict=True)]
                shards = [task.result() for task in tasks]
        if get_inputs()[1] != binding:
            raise ValueError("intro scene, receipt or worker changed during rendering")
        publish(name, shards, binding, samples, scale, started)


if __name__ == "__main__":
    main()
