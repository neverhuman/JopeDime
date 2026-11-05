#!/usr/bin/env python3
"""Stage, render, verify, and assemble native UHD Blender plates."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
from queue import Empty, Queue
import shlex
import shutil
import struct
import subprocess
import sys
import time
import uuid

from build_render_plan import ROOT, digest, read_json, require
from common import REPO, validate_configuration, require_render_enabled, external_path


from run_intro_render import (  # noqa: E402
    BASE, HOSTS, LOCAL_RUNTIME, LOCAL_RUNTIME_SHA256, RUNTIME, RUNTIME_SHA256, log, now, rsync, ssh,
)


WORKERS = REPO / "tools/visualization/blender"
PLAN = external_path(ROOT / "render-plan.receipt.json")
WIDTH, HEIGHT, FPS = 3840, 2160, 30
CHUNK = 24
LOCAL_HEADROOM = 25 * 1024**3
QUALITY = {"quality_profile": "uhd-master", "width": WIDTH, "height": HEIGHT, "fps": FPS,
           "native_render": True, "resolution_percentage": 200, "file_format": "PNG",
           "color_mode": "RGB", "color_depth": 16, "adaptive_sampling": True,
           "adaptive_threshold": .003, "adaptive_min_samples": 32, "png_compression": 35,
           "denoising_use_gpu": True, "denoising_quality": "HIGH", "denoising_prefilter": "ACCURATE",
           "denoising_input_passes": "RGB_ALBEDO_NORMAL", "gpu_only": True, "cpu_fallback": False}


@dataclass(frozen=True)
class Job:
    family: str
    name: str
    frames: tuple[int, ...]
    samples: int = 256
    case_id: str | None = None

    def __post_init__(self):
        require(self.family in ("intro", "demo"), "invalid job family")
        require(isinstance(self.name, str) and re.fullmatch(r"[a-z0-9][a-z0-9-]*", self.name), "unsafe job name")
        require(type(self.samples) is int and 1 <= self.samples <= 4096, "invalid sample count")
        require(bool(self.frames) and all(type(frame) is int for frame in self.frames), "invalid frame types")
        require(list(self.frames) == sorted(set(self.frames)), "frames must be ordered and unique")
        if self.family == "intro":
            require(self.case_id is None and 1 <= self.frames[0] <= self.frames[-1] <= 330, "invalid intro frames")
        else:
            require(self.case_id in tuple(f"case-{letter}" for letter in "abcdefghij"), "invalid demo case")
            start = (ord(self.case_id[-1]) - ord("a")) * 120 + 1
            require(start <= self.frames[0] <= self.frames[-1] < start + 120, "invalid demo frame range")


class Busy(RuntimeError):
    pass


def receipt(path, payload):
    pending = path.with_name(f".{path.name}-{uuid.uuid4().hex[:8]}.pending")
    with external_path(pending).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    require(not path.exists(), f"refusing to replace {path}")
    external_path(pending).rename(external_path(path))


def inputs():
    plan = read_json(PLAN)
    require(plan.get("schema") == "jain.final-4k-render-plan/v1"
            and plan.get("status") == "PASS" and plan.get("native_frames_total") == 1290, "invalid UHD plan")
    require(plan.get("intro_frames") == list(range(1, 331)), "invalid intro plan")
    require([case["case_id"] for case in plan["demo_cases"]] == [f"case-{c}" for c in "abcdefghij"], "invalid case order")
    full_jobs(plan)  # Validate names, IDs and frame ranges before transport admission.
    for artifact in plan["inputs"].values():
        require(digest(Path(artifact["path"])) == artifact["sha256"], f"approved input changed: {artifact['path']}")
    paths = {key: Path(plan["inputs"][key]["path"]) for key in (
        "intro_scene", "intro_scene_receipt", "demo_scene", "demo_scene_receipt",
    )}
    expected_names = {"intro_scene": "intro.blend", "intro_scene_receipt": "intro.receipt.json",
                      "demo_scene": "scene.blend", "demo_scene_receipt": "scene.receipt.json"}
    require(all(path.name == expected_names[key] for key, path in paths.items()), "unsafe input artifact name")
    paths.update({"intro_worker": WORKERS / "intro_render.py", "demo_worker": WORKERS / "render_animation.py"})
    binding = {key: digest(path) for key, path in paths.items()}
    binding["plan"] = digest(PLAN)
    revision = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()[:20]
    return plan, paths, binding, revision


def remote_paths(revision, paths):
    root = f"{BASE}/jain-final-4k-v1/revisions/{revision}"
    result = {key: (f"{root}/code/tools/visualization/blender/{path.name}" if key.endswith("_worker")
                    else f"{root}/inputs/{key.split('_')[0]}/{path.name}") for key, path in paths.items()}
    return root, result


def stage_host(host, revision, paths, binding):
    root, remote = remote_paths(revision, paths)
    folders = {str(Path(path).parent) for path in remote.values()}
    ssh(host, f"umask 077; mkdir -p -- {shlex.join((*sorted(folders), f'{root}/jobs'))}")
    for folder in sorted(folders):
        rsync([path for key, path in paths.items() if str(Path(remote[key]).parent) == folder], f"{host}:{folder}/")
    checks = "\n".join((f"{RUNTIME_SHA256}  {RUNTIME}", *(f"{binding[key]}  {path}" for key, path in remote.items())))
    ssh(host, f"printf '%s\\n' {shlex.quote(checks)} | sha256sum -c")
    log(f"{host} pinned UHD inputs staged under {root}")
    return root, remote, checks


def device_status(host):
    return ssh(host, "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits; "
                     "nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits")


def can_render(host):
    lines = device_status(host).splitlines()
    return len(lines) == 1 and int(lines[0].strip()) >= 12_288


def worker_args(job, paths, output, device):
    prefix = job.family
    args = ["--scene", str(paths[f"{prefix}_scene"]), "--receipt", str(paths[f"{prefix}_scene_receipt"]),
            "--output-dir", str(output), "--samples", str(job.samples), "--scale", "200", "--quality", "uhd-master",
            "--device", device]
    if job.case_id is not None:
        args.extend(("--case-id", job.case_id))
    if prefix == "intro" and job.frames == tuple(range(job.frames[0], job.frames[-1] + 1)):
        args.extend(("--frame", f"{job.frames[0]}:{job.frames[-1]}"))
    else:
        args.extend(item for frame in job.frames for item in ("--frame", str(frame)))
    return args


def blender_cmd(runtime, job, paths, output, device):
    return (str(runtime), "--background", "--factory-startup", "--disable-autoexec", "--threads", "8",
            "--python-exit-code", "1", "--python", str(paths[f"{job.family}_worker"]), "--",
            *worker_args(job, paths, output, device))


def launch(host, job, staged, location):
    _, paths, checks = staged
    command = shlex.join(("timeout", "--kill-after=30s", "120m", *blender_cmd(
        RUNTIME, job, paths, f"{location}/plates", "OPTIX",
    )))
    guard = f"""set -euo pipefail
umask 077
finish() {{
    code=$?
    printf '%s\\n' "$code" > {location}/status.tmp
    mv -- {location}/status.tmp {location}/status
    exit "$code"
}}
trap finish EXIT
exec 9>{BASE}/jain-final-4k-gpu.lock
flock -n 9 || exit 72
printf '%s\\n' {shlex.quote(checks)} | sha256sum -c
free_mib="$(timeout 15s nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
compute_pids="$(timeout 15s nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d ' ')"
printf 'host=%s free_mib=%s existing_compute_pids=%s\\n' "$(hostname)" "$free_mib" "$compute_pids"
[[ "$free_mib" =~ ^[0-9]+$ ]] && ((free_mib >= 12288)) && [[ -z "$compute_pids" ]] || exit 72
[[ $(df -Pm -- {location} | awk 'NR==2{{print $4}}') -ge 20480 ]] || exit 73
{command}
"""
    result = ssh(host, f"""set -eu
if mkdir -m 700 -- {location} 2>/dev/null; then
    nohup bash -c {shlex.quote(guard)} </dev/null >{location}/render.log 2>&1 &
    echo started
else
    test -d {location}
    echo existing
fi""")
    log(f"{host} {result} {job.name}: {len(job.frames)} frames at max {job.samples} spp")


def job_state(host, location):
    return ssh(host, f"if test -s {location}/status; then echo done; cat {location}/status; "
                     f"elif test -f {location}/plates/frames.jsonl; then echo running; "
                     f"wc -l < {location}/plates/frames.jsonl; elif test -d {location}; then printf 'running\\n0\\n'; "
                     "else printf 'missing\\n0\\n'; fi").splitlines()


def wait_job(host, job, location):
    deadline = time.monotonic() + 123 * 60
    last = 0
    while time.monotonic() < deadline:
        state, count = job_state(host, location)
        if state == "done":
            if count == "0":
                return
            if count == "72":
                raise Busy(f"{host} admitted other work before {job.name}")
            raise RuntimeError(f"{host} {job.name} exit {count}:\n" + ssh(host, f"tail -45 -- {location}/render.log"))
        if int(count) >= last + 6:
            log(f"{host} {job.name}: {count}/{len(job.frames)}")
            last = int(count)
        time.sleep(4 if len(job.frames) == 1 else 15)
    raise TimeoutError(f"{host} {job.name} timed out at {location}")


def verify_job(path, job, binding):
    receipt_path = path / "render.receipt.json"
    record = read_json(receipt_path)
    expected = {**QUALITY, "status": "PASS", "samples": job.samples, "requested_frames": list(job.frames),
                "scene_sha256": binding[f"{job.family}_scene"],
                "scene_receipt_sha256": binding[f"{job.family}_scene_receipt"],
                "worker_sha256": binding[f"{job.family}_worker"]}
    if job.case_id is not None:
        expected["case_id"] = job.case_id
    for key, value in expected.items():
        require(record.get(key) == value, f"{receipt_path}: {key} expected {value!r}, got {record.get(key)!r}")
    devices = record.get("devices", [])
    require(record["backend"] in ("OPTIX", "METAL") and len(devices) == 1
            and devices[0]["type"] == record["backend"], f"invalid GPU binding in {receipt_path}")
    if job.family == "intro":
        require(record.get("shared_worker_sha256") == binding["demo_worker"], "intro shared worker hash differs")
    require([row["frame"] for row in record["frames"]] == list(job.frames), f"wrong frames for {job.name}")
    for entry in record["frames"]:
        filename = f"frame-{entry['frame']:06d}.png"
        png = path / filename
        require(entry["file"] == filename and digest(png) == entry["sha256"], f"plate SHA differs: {png}")
        with png.open("rb") as stream:
            header = stream.read(26)
        require(header[:16] == b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
                and struct.unpack(">IIBB", header[16:26]) == (WIDTH, HEIGHT, 16, 2), f"not native RGB16: {png}")
    return record


def job_output(revision, job):
    return external_path(ROOT / "shards" / revision / job.name)


def render_remote(host, job, staged, revision, binding):
    output = job_output(revision, job)
    if (output / "render.receipt.json").is_file():
        try:
            return verify_job(output, job, binding)
        except FileNotFoundError:
            log(f"{job.name} needs a partial transfer resumed")
    root, _, _ = staged
    for attempt in range(20):
        location = f"{root}/jobs/{job.name}-{host}-{attempt:02d}"
        state, code = job_state(host, location)
        # A future explicit resume leaves cancelled and admission-denied
        # attempts intact and starts the shard in a fresh attempt directory.
        if state == "done" and code in ("72", "143"):
            continue
        if state == "missing":
            if not can_render(host):
                raise Busy(f"{host} already has a GPU process or insufficient headroom")
            launch(host, job, staged, location)
        wait_job(host, job, location)
        require(shutil.disk_usage(ROOT).free >= LOCAL_HEADROOM + 2 * 1024**3,
                "local transfer would violate the reserved 25 GiB disk headroom")
        external_path(output).mkdir(parents=True, exist_ok=True)
        rsync([f"{host}:{location}/plates/"], output)
        record = verify_job(output, job, binding)
        log(f"{host} verified {job.name}: {len(job.frames)} native RGB16 plates")
        return record
    raise RuntimeError(f"too many GPU admission rejections for {job.name}")


def render_local(job, paths, revision, binding):
    output = job_output(revision, job)
    if not (output / "render.receipt.json").is_file():
        require(digest(LOCAL_RUNTIME) == LOCAL_RUNTIME_SHA256, "unpinned Metal runtime")
        log(f"Metal starting {job.name}: {len(job.frames)} frames at max {job.samples} spp")
        with (external_path(ROOT / f"{job.name}-metal.log")).open("x") as stream:
            subprocess.run(blender_cmd(LOCAL_RUNTIME, job, paths, output, "METAL"), timeout=7200,
                           check=True, stdout=stream, stderr=subprocess.STDOUT)
    return verify_job(output, job, binding)


def probes():
    by_host = {
        HOSTS[1]: [Job("demo", f"probe-demo-{samples}", (89,), samples, "case-a")
                   for samples in (32, 128, 256)],
        HOSTS[0]: [Job("intro", f"probe-intro-{samples}", (270,), samples) for samples in (32, 128, 256)]
                  + [Job("intro", "probe-board-256", (28,))],
    }
    return by_host


def decode_16(path):
    import numpy as np
    pixels = subprocess.check_output(("ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-frames:v", "1",
                                      "-pix_fmt", "rgb48le", "-f", "rawvideo", "pipe:1"))
    return np.frombuffer(pixels, dtype="<u2").reshape(HEIGHT, WIDTH, 3)


def quality_comparison(revision, family, samples):
    import math
    import numpy as np
    frame = 89 if family == "demo" else 270
    def pixels(count):
        return decode_16(external_path(ROOT / "shards" / revision / f"probe-{family}-{count}" / f"frame-{frame:06d}.png"))
    difference = pixels(samples).astype(np.float32) - pixels(256)
    central = difference[450:1940, 1240:2618]
    result = {}
    for label, crop in (("full", difference), ("network_roi", central)):
        mse = float(np.mean((crop / 65535.) ** 2))
        result[label] = {"psnr_db": -10 * math.log10(mse) if mse else None,
                         "equal": mse == 0, "mean_absolute_rgb16_levels": float(np.abs(crop).mean())}
    return result


def benchmark(paths, binding, revision, local):
    tasks = probes()
    started = now()

    def host_jobs(host, jobs):
        staged = stage_host(host, revision, paths, binding)
        return [render_remote(host, job, staged, revision, binding) for job in jobs]

    if local:
        results = {"local-metal": [render_local(job, paths, revision, binding)
                                   for jobs in tasks.values() for job in jobs]}
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = {host: pool.submit(host_jobs, host, jobs) for host, jobs in tasks.items()}
            results = {host: future.result() for host, future in pending.items()}
    comparisons = {family: {samples: quality_comparison(revision, family, samples) for samples in (32, 128)}
                   for family in ("intro", "demo")}
    jobs = list(job for group in tasks.values() for job in group)
    measurements = []
    for job in jobs:
        folder = job_output(revision, job)
        item = verify_job(folder, job, binding)
        for frame in item["frames"]:
            measurements.append({"job": job.name, "samples": job.samples, "devices": item["devices"],
                                 "seconds": frame["elapsed_ns"] / 1e9, "bytes": (folder / frame["file"]).stat().st_size,
                                 "receipt_sha256": digest(folder / "render.receipt.json"), "plate_sha256": frame["sha256"]})
    receipt(external_path(ROOT / "probe.receipt.json"), {"schema": "jain.final-4k-probe/v1", "status": "PASS", "binding": binding,
            "revision": revision, "started_utc": started, "completed_utc": now(), "hosts": list(results),
            "quality": QUALITY, "measurements": measurements, "comparison_to_256_spp": comparisons})
    log("UHD render probes verified")


def chunks(frames):
    return [tuple(frames[index:index + CHUNK]) for index in range(0, len(frames), CHUNK)]


def full_jobs(plan):
    intro = [Job("intro", f"intro-{frames[0]:06d}-{frames[-1]:06d}", frames)
             for frames in chunks(plan["intro_frames"])]
    demo = [Job("demo", f"demo-{case['case_id']}-{frames[0]:06d}-{frames[-1]:06d}", frames, case_id=case["case_id"])
            for case in plan["demo_cases"] for frames in chunks(case["frames"])]
    mixed = [job for pair in zip(intro, demo) for job in pair]
    return mixed + intro[len(demo):] + demo[len(intro):]


def publish_group(jobs, revision, binding, group, expected_frames, case_id=None):
    if case_id:
        output = external_path(ROOT / "demo-plates" / group)
        manifest = output / "render.receipt.json"
        schema = "jain.final-4k-case-plates/v1"
    else:
        output = external_path(ROOT / group)
        manifest = external_path(ROOT / f"{group}.receipt.json")
        schema = "jain.intro-plates/v1"
    family, = {job.family for job in jobs}
    shards = []
    frames = []
    for job in jobs:
        directory = job_output(revision, job)
        record = verify_job(directory, job, binding)
        for row in record["frames"]:
            frames.append((directory / row["file"], row))
        path = directory / "render.receipt.json"
        shards.append({"job": job.name, "path": str(path), "receipt_sha256": digest(path),
                       "backend": record["backend"], "devices": record["devices"], "elapsed_ns": record["elapsed_ns"]})
    frames.sort(key=lambda item: item[1]["frame"])
    records = [row for _, row in frames]
    require([row["frame"] for row in records] == expected_frames, f"frame coverage differs for {group}")
    result = {"schema": schema, "status": "PASS", **QUALITY, "samples": 256,
              "scene_sha256": binding[f"{family}_scene"], "scene_receipt_sha256": binding[f"{family}_scene_receipt"],
              "worker_sha256": binding[f"{family}_worker"], "render_plan_sha256": binding["plan"],
              "requested_frames": expected_frames, "frames": records, "shards": shards, "directory": str(output)}
    if case_id:
        result.update(case_id=case_id, complete_case=False)

    if manifest.is_file():
        require(read_json(manifest) == result, f"existing {group} receipt differs")
        for row in records:
            require(digest(output / row["file"]) == row["sha256"], f"published {group} PNG differs")
    else:
        require(not output.exists(), f"refusing to replace {output}")
        external_path(output.parent).mkdir(parents=True, exist_ok=True)
        pending = output.with_name(f".{group}-{uuid.uuid4().hex[:8]}.pending")
        external_path(pending).mkdir()
        for source, row in frames:
            os.link(source, external_path(pending / row["file"]))
        if case_id:
            receipt(pending / manifest.name, result)
        external_path(pending).rename(external_path(output))
        if case_id is None:
            receipt(manifest, result)
        log(f"published {len(records)} verified native UHD plates for {group}")
    return {"receipt": str(manifest), "sha256": digest(manifest), "frames": len(records)}


def run_full(plan, paths, binding, revision, local):
    probe_path = external_path(ROOT / "probe.receipt.json")
    probe = read_json(probe_path)
    require(probe["status"] == "PASS" and probe["binding"] == binding, "UHD probes are absent or stale")
    sizes = [item["bytes"] for item in probe["measurements"]]
    jobs = full_jobs(plan)
    pending = [job for job in jobs if not (job_output(revision, job) / "render.receipt.json").is_file()]
    remaining = sum(len(job.frames) for job in pending)
    require(shutil.disk_usage(ROOT).free >= LOCAL_HEADROOM + int(max(sizes) * remaining * 1.20),
            "not enough local headroom for conservative 16-bit plate estimate")
    started = now()
    schedule = Queue()
    for job in jobs:
        schedule.put(job)

    def on_gpu(host):
        staged = stage_host(host, revision, paths, binding)
        while True:
            try:
                job = schedule.get_nowait()
            except Empty:
                return
            try:
                render_remote(host, job, staged, revision, binding)
            except Busy as error:
                schedule.put(job)
                log(f"{error}; leaving it alone and checking again in 120s")
                time.sleep(120)

    if local:
        for job in jobs:
            render_local(job, paths, revision, binding)
    else:
        with ThreadPoolExecutor(max_workers=len(HOSTS)) as pool:
            tasks = [pool.submit(on_gpu, host) for host in HOSTS]
            for task in tasks:
                task.result()
    require(inputs()[2] == binding, "approved scenes or UHD worker sources changed during production")
    intro = publish_group([job for job in jobs if job.family == "intro"], revision, binding,
                          "intro-plates", plan["intro_frames"])
    demo = {case["case_id"]: publish_group([job for job in jobs if job.case_id == case["case_id"]], revision,
            binding, case["case_id"], case["frames"], case["case_id"]) for case in plan["demo_cases"]}
    total = intro["frames"] + sum(item["frames"] for item in demo.values())
    require(total == plan["native_frames_total"], "published UHD plate total differs from the plan")
    receipt(external_path(ROOT / "native-plates.receipt.json"), {"schema": "jain.final-4k-native-plates/v1", "status": "PASS",
             "binding": binding, "revision": revision, "started_utc": started, "completed_utc": now(),
             "quality": QUALITY, "samples": 256, "runtime_sha256": LOCAL_RUNTIME_SHA256 if local else RUNTIME_SHA256,
             "probe_receipt_sha256": digest(probe_path), "native_frames_total": total,
             "intro_plates": intro, "demo_case_plates": demo})
    log(f"all {total} native UHD plates ready for compositing")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "probe", "full"))
    parser.add_argument("--local-metal", action="store_true")
    args = parser.parse_args()
    validate_configuration()
    plan, paths, binding, revision = inputs()
    if args.mode == "check":
        jobs = full_jobs(plan)
        print(json.dumps({"binding": binding, "revision": revision, "native_plates": sum(len(job.frames) for job in jobs),
                          "jobs": len(jobs), "quality": QUALITY}, indent=2))
        return
    require_render_enabled(local=args.local_metal, uhd=True)
    with (external_path(ROOT / ".uhd-render.lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode == "probe":
            benchmark(paths, binding, revision, args.local_metal)
        else:
            run_full(plan, paths, binding, revision, args.local_metal)


if __name__ == "__main__":
    main()
