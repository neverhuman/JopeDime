#!/usr/bin/env python3
"""Render only the bounded closing intro chunks on the local Metal GPU."""
from __future__ import annotations

import argparse
import shutil
import subprocess


import render_uhd as uhd
from common import require_render_enabled, external_path


def already_on_remote(root, job):
    for host in uhd.HOSTS:
        if uhd.ssh(host, f"find {root}/jobs -mindepth 1 -maxdepth 1 -type d -name '{job.name}-{host}-*' -print -quit"):
            return True
    return False


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    import numpy as np
    require_render_enabled(local=True, uhd=True)
    require_render_enabled(uhd=True)
    plan, paths, binding, revision = uhd.inputs()
    root, _ = uhd.remote_paths(revision, paths)
    probe = uhd.Job("intro", "probe-metal-intro-256", (270,))
    local_probe = uhd.job_output(revision, probe)
    evidence = uhd.verify_job(local_probe, probe, binding)
    elapsed = evidence["frames"][0]["elapsed_ns"] / 1e9
    uhd.require(elapsed <= 30., "Metal probe too slow for a bounded sidecar")
    uhd.require(uhd.digest(uhd.LOCAL_RUNTIME) == uhd.LOCAL_RUNTIME_SHA256, "unpinned Metal runtime")
    remote_probe = uhd.job_output(revision, uhd.Job("intro", "probe-intro-256", (270,)))
    diff = (uhd.decode_16(local_probe / "frame-000270.png").astype(np.float32)
            - uhd.decode_16(remote_probe / "frame-000270.png"))
    full_psnr = float(-10 * np.log10(np.mean((diff / 65535.) ** 2)))
    roi = diff[450:1940, 1240:2618]
    roi_psnr = float(-10 * np.log10(np.mean((roi / 65535.) ** 2)))
    uhd.require(roi_psnr >= 50., "Metal/OptiX probe differences too large for a mid-intro backend change")
    started = uhd.now()
    results = []
    jobs = [job for job in uhd.full_jobs(plan) if job.family == "intro" and job.frames[0] >= 265]
    for job in reversed(jobs):
        final = uhd.job_output(revision, job)
        if final.exists() or already_on_remote(root, job):
            uhd.log(f"Metal leaves {job.name} to the active remote schedule")
            continue
        uhd.require(shutil.disk_usage(uhd.ROOT).free >= uhd.LOCAL_HEADROOM + 2 * 1024**3, "disk reserve reached")
        scratch = uhd.ROOT / "metal-sidecar" / revision / job.name
        external_path(scratch.parent).mkdir(parents=True, exist_ok=True)
        log_path = scratch.parent / f"{job.name}.log"
        uhd.log(f"Metal rendering far-tail {job.name}")
        with external_path(log_path).open("x") as log:
            subprocess.run(uhd.blender_cmd(uhd.LOCAL_RUNTIME, job, paths, scratch, "METAL"), timeout=1800,
                           check=True, stdout=log, stderr=subprocess.STDOUT)
        result = uhd.verify_job(scratch, job, binding)
        if final.exists() or already_on_remote(root, job):
            uhd.log(f"Metal preserves its alternate {job.name}; remote owns the canonical slot")
            continue
        external_path(final.parent).mkdir(parents=True, exist_ok=True)
        external_path(scratch).rename(external_path(final))
        results.append({"job": job.name, "frames": list(job.frames),
                        "receipt_sha256": uhd.digest(final / "render.receipt.json"),
                        "elapsed_ns": result["elapsed_ns"]})
        uhd.log(f"Metal published {len(job.frames)} verified native RGB16 plates for {job.name}")
    uhd.receipt(uhd.ROOT / "metal-sidecar.receipt.json", {
        "schema": "jain.final-4k-metal-sidecar/v1", "status": "PASS", "binding": binding,
        "started_utc": started, "completed_utc": uhd.now(), "runtime_sha256": uhd.LOCAL_RUNTIME_SHA256,
        "probe_seconds": elapsed, "metal_probe_receipt_sha256": uhd.digest(local_probe / "render.receipt.json"),
        "optix_probe_receipt_sha256": uhd.digest(remote_probe / "render.receipt.json"),
        "probe_full_psnr_db": full_psnr, "probe_roi_psnr_db": roi_psnr, "chunks": results,
        "frames_published": sum(len(job["frames"]) for job in results),
    })


if __name__ == "__main__":
    main()
