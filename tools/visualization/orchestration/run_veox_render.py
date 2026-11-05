#!/usr/bin/env python3
"""Render the sealed 1080p VEOX endcap on idle, pinned OptiX hosts."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import struct
import subprocess
import sys
import time
import uuid


from common import REPO, artifact_family, validate_configuration, require_render_enabled, external_path

ROOT = artifact_family("jain-veox-endcap-v1")
WORKER = REPO / "tools/visualization/blender/veox_endcap.py"
GPU_HELPER = WORKER.with_name("render_animation.py")
SIZE = (1920, 1080)
FPS = 30
PREVIEW = (1, 28, 55, 95, 170, 180)
FULL = (tuple(range(1, 91)), tuple(range(91, 181)))

from run_intro_render import (  # noqa: E402
    BASE, HOSTS, LOCAL_RUNTIME, LOCAL_RUNTIME_SHA256, RUNTIME, RUNTIME_SHA256, log, now, rsync, sha, ssh,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def inputs(scene_dir):
    scene_file = scene_dir / "veox-endcap.blend"
    scene_receipt = scene_file.with_suffix(".receipt.json")
    paths = {"scene_sha256": scene_file, "scene_receipt_sha256": scene_receipt, "worker_sha256": WORKER,
             "shared_worker_sha256": GPU_HELPER}
    scene = load(scene_receipt)
    require(scene["schema"] == "jain.veox-endcap-scene/v1" and scene["status"] == "PASS",
            "expected passing sealed endcap scene")
    binding = {key: sha(path) for key, path in paths.items()}
    require(scene["worker_sha256"] == binding["worker_sha256"], "scene and worker revision differ")
    return paths, binding


def request_frames(frames):
    return tuple(item for frame in frames for item in ("--frame", str(frame)))


def worker_cli(code, scene, scene_receipt, plates, frames, samples, *, runtime=RUNTIME, device="OPTIX"):
    return (str(runtime), "--background", "--factory-startup", "--disable-autoexec", "--threads", "8",
            "--python-exit-code", "1", "--python", code, "--", "render", "--scene", scene,
            "--receipt", scene_receipt, "--output-dir", plates, "--samples", str(samples),
            "--device", device, *request_frames(frames))


def stage(host, paths, binding, root):
    scene_dir = f"{root}/scene"
    code_dir = f"{root}/code/tools/visualization/blender"
    ssh(host, f"umask 077; mkdir -p -- {scene_dir} {code_dir} {root}/jobs")
    rsync((paths["scene_sha256"], paths["scene_receipt_sha256"]), f"{host}:{scene_dir}/")
    rsync((WORKER, GPU_HELPER), f"{host}:{code_dir}/")
    remote = {key: f"{code_dir if key.endswith('worker_sha256') else scene_dir}/{path.name}"
              for key, path in paths.items()}
    check = "\n".join((f"{RUNTIME_SHA256}  {RUNTIME}", *(f"{binding[key]}  {name}" for key, name in remote.items())))
    ssh(host, f"printf '%s\\n' {shlex.quote(check)} | sha256sum -c")
    return remote, check


def launch(host, remote, check, job, frames, samples):
    cmd = shlex.join(("timeout", "--kill-after=20s", "20m", *worker_cli(
        remote["worker_sha256"], remote["scene_sha256"], remote["scene_receipt_sha256"],
        f"{job}/plates", frames, samples,
    )))
    guard = f"""set -euo pipefail
umask 077
finish() {{
    code=$?
    printf '%s\\n' "$code" > {job}/status.tmp
    mv -- {job}/status.tmp {job}/status
    exit "$code"
}}
trap finish EXIT
exec 9>{BASE}/jain-veox-endcap-gpu.lock
flock -n 9
printf '%s\\n' {shlex.quote(check)} | sha256sum -c
free_mib="$(timeout 15s nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
compute_pids="$(timeout 15s nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d ' ')"
printf 'host=%s free_mib=%s existing_compute_pids=%s\\n' "$(hostname)" "$free_mib" "$compute_pids"
[[ "$free_mib" =~ ^[0-9]+$ ]]
((free_mib >= 6144))
[[ -z "$compute_pids" ]]
{cmd}
"""
    status = ssh(host, f"""set -eu
if mkdir -m 700 -- {job} 2>/dev/null; then
    nohup bash -c {shlex.quote(guard)} </dev/null >{job}/render.log 2>&1 &
    echo started
else
    test -d {job}
    echo existing
fi""")
    log(f"{host} {status} VEOX {len(frames)} frames at {samples} samples ({job})")


def wait(host, job, count):
    deadline = time.monotonic() + 21 * 60
    last = 0
    while time.monotonic() < deadline:
        state = ssh(host, f"if test -s {job}/status; then echo done; cat {job}/status; "
                          f"elif test -f {job}/plates/frames.jsonl; then echo running; "
                          f"wc -l < {job}/plates/frames.jsonl; else printf 'running\\n0\\n'; fi")
        status, value = state.splitlines()
        if status == "done":
            if value == "0":
                return
            raise RuntimeError(f"{host} VEOX Blender exited {value}:\n" + ssh(host, f"tail -35 -- {job}/render.log"))
        if int(value) >= last + 15:
            log(f"{host} VEOX {value}/{count} frames")
            last = int(value)
        time.sleep(3 if count < 10 else 10)
    raise TimeoutError(f"{host} VEOX timeout: {job}/render.log")


def verify(directory, binding, frames, samples, backend="OPTIX"):
    receipt_path = directory / "render.receipt.json"
    record = load(receipt_path)
    expected = {"schema": "jain.veox-endcap-plates/v1", "status": "PASS", **binding,
                "width": SIZE[0], "height": SIZE[1], "fps": FPS, "backend": backend,
                "samples": samples, "gpu_only": True, "cpu_fallback": False}
    for key, value in expected.items():
        require(record.get(key) == value, f"{receipt_path}: {key} expected {value!r}, found {record.get(key)!r}")
    require([entry["frame"] for entry in record["frames"]] == list(frames), "endcap frames missing or reordered")
    for entry in record["frames"]:
        filename = f"frame-{entry['frame']:06d}.png"
        path = directory / filename
        require(entry["file"] == filename and sha(path) == entry["sha256"], f"unexpected frame or digest: {path}")
        with path.open("rb") as stream:
            header = stream.read(24)
        require(header[:16] == b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" and struct.unpack(">II", header[16:24]) == SIZE,
                f"unexpected PNG dimensions: {path}")
    return record, sha(receipt_path)


def render_host(host, root, job_name, frames, paths, binding, samples):
    directory = external_path(ROOT / "shards" / job_name / host)
    remote, check = stage(host, paths, binding, root)
    job = f"{root}/jobs/{job_name}-{host}"
    launch(host, remote, check, job, frames, samples)
    wait(host, job, len(frames))
    external_path(directory).mkdir(parents=True, exist_ok=True)
    rsync([f"{host}:{job}/plates/"], directory)
    record, digest = verify(directory, binding, frames, samples)
    log(f"{host} VEOX transferred and verified {len(frames)} plates")
    return directory, record, {"host": host, "receipt_sha256": digest, "remote_job": job,
                               "devices": record["devices"], "frames": len(frames),
                               "backend": "OPTIX", "runtime_sha256": RUNTIME_SHA256}


def render_local(job_name, frames, paths, binding, samples):
    require(sha(LOCAL_RUNTIME) == LOCAL_RUNTIME_SHA256, "unpinned local Blender runtime")
    directory = external_path(ROOT / "shards" / job_name / "local-metal")
    if not (directory / "render.receipt.json").exists():
        external_path(directory.parent).mkdir(parents=True, exist_ok=True)
        log(f"local Metal started VEOX {len(frames)} frames at {samples} samples")
        with (external_path(directory.parent / "local-metal.log")).open("x") as stream:
            subprocess.run(worker_cli(str(paths["worker_sha256"]), str(paths["scene_sha256"]),
                                       str(paths["scene_receipt_sha256"]), str(directory), frames, samples,
                                       runtime=LOCAL_RUNTIME, device="METAL"), timeout=20 * 60, check=True,
                           stdout=stream, stderr=subprocess.STDOUT)
    record, digest = verify(directory, binding, frames, samples, "METAL")
    log(f"local Metal verified {len(frames)} VEOX plates")
    return directory, record, {"host": "local-metal", "receipt_sha256": digest,
                               "devices": record["devices"], "frames": len(frames),
                               "backend": "METAL", "runtime_sha256": LOCAL_RUNTIME_SHA256}


def publish(name, shards, frames, binding, samples, started):
    output = external_path(ROOT / name)
    staging = external_path(ROOT / f".{name}-{uuid.uuid4().hex[:8]}.pending")
    external_path(staging).mkdir()
    records = []
    for directory, record, _ in shards:
        for entry in record["frames"]:
            filename = entry["file"]
            os.link(directory / filename, external_path(staging / filename))
            records.append(entry)
    records.sort(key=lambda entry: entry["frame"])
    require(tuple(entry["frame"] for entry in records) == frames, "published set differs from requested frames")
    external_path(staging).rename(external_path(output))
    runtimes = {proof["backend"]: proof["runtime_sha256"] for _, _, proof in shards}
    manifest = {"schema": "jain.veox-endcap-plates/v1", "status": "PASS", **binding,
                "width": SIZE[0], "height": SIZE[1], "fps": FPS,
                "backend": next(iter(runtimes)) if len(runtimes) == 1 else "MIXED",
                "samples": samples, "gpu_only": True, "runtime_sha256s": runtimes,
                "started_utc": started, "completed_utc": now(), "directory": str(output),
                "shards": [proof for _, _, proof in shards], "frames": records}
    if len(runtimes) == 1:
        manifest["runtime_sha256"] = next(iter(runtimes.values()))
    temporary = external_path(ROOT / f".{name}.receipt.pending")
    with external_path(temporary).open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    external_path(temporary).rename(external_path(ROOT / f"{name}.receipt.json"))
    log(f"published {len(records)} verified VEOX endcap plates at {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "preview", "full"))
    parser.add_argument("--preview-frames", type=int, nargs="+", default=PREVIEW)
    parser.add_argument("--preview-host", choices=HOSTS, default=HOSTS[1])
    parser.add_argument("--split-preview", action="store_true")
    host_mode = parser.add_mutually_exclusive_group()
    host_mode.add_argument("--full-host", choices=HOSTS)
    host_mode.add_argument("--metal-tail", action="store_true")
    parser.add_argument("--scene-dir", type=Path, default=external_path(ROOT / "scene"))
    parser.add_argument("--preview-output", choices=("previews", "previews-v2", "previews-v3", "previews-v4", "previews-v5", "previews-v6", "previews-v7"),
                        default="previews")
    parser.add_argument("--full-output", choices=("frames", "frames-v5", "frames-v6", "frames-v7"), default="frames")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--expected-scene-sha256")
    parser.add_argument("--expected-worker-sha256")
    args = parser.parse_args()
    validate_configuration()
    paths, binding = inputs(args.scene_dir.resolve())
    for key in ("scene_sha256", "worker_sha256"):
        expected = getattr(args, f"expected_{key}")
        if expected:
            require(binding[key] == expected, f"{key} differs from approved bytes")
    if args.mode == "check":
        print(json.dumps(binding, indent=2))
        return
    preview = tuple(sorted(set(args.preview_frames)))
    require(len(preview) == len(args.preview_frames) and 1 <= preview[0] <= preview[-1] <= 180,
            "preview frames must be unique and in 1–180")
    if args.mode == "preview":
        if args.split_preview:
            require(len(preview) >= 2, "split preview needs at least two frames")
            middle = len(preview) // 2
            groups, hosts = (preview[:middle], preview[middle:]), HOSTS
        else:
            groups, hosts = (preview,), (args.preview_host,)
    elif args.full_host:
        groups, hosts = (FULL[0] + FULL[1],), (args.full_host,)
    elif args.metal_tail:
        groups, hosts = FULL, (HOSTS[0], "local-metal")
    else:
        groups, hosts = FULL, HOSTS
    require_render_enabled()
    if args.metal_tail:
        require_render_enabled(local=True)
    name = args.preview_output if args.mode == "preview" else args.full_output
    samples = args.samples or 96
    require(16 <= samples <= 4096, "invalid sample count")
    frames = tuple(sorted(frame for group in groups for frame in group))
    revision = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()[:20]
    remote = f"{BASE}/jain-veox-endcap-v1/revisions/{revision}"
    started = now()
    job_name = f"{revision}-{name}-{samples}s-{hashlib.sha256(str(frames).encode()).hexdigest()[:8]}"
    with (external_path(ROOT / ".veox-render.lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (external_path(ROOT / name)).exists() and not (external_path(ROOT / f"{name}.receipt.json")).exists(), f"{name} already published")
        with ThreadPoolExecutor(max_workers=len(hosts)) as pool:
            tasks = [pool.submit(render_local, job_name, group, paths, binding, samples) if host == "local-metal"
                     else pool.submit(render_host, host, remote, job_name, group, paths, binding, samples)
                     for host, group in zip(hosts, groups, strict=True)]
            shards = [task.result() for task in tasks]
        require(inputs(args.scene_dir.resolve())[1] == binding, "endcap source or worker changed during rendering")
        publish(name, shards, frames, binding, samples, started)


if __name__ == "__main__":
    main()
