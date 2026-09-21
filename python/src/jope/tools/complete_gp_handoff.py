"""Recover a health launch argument and verify an already activated GP handoff."""

import argparse
import shlex
import subprocess
import time
from pathlib import Path

from activate_promotion_safeguards import verify_effective_launches
from promotion_gate import immutable
from resume_expired_cycle import read, require, sha


def main(root, deployment, baseline):
    controller = deployment / "controller"
    manifest = controller / "manifest.json"
    m = read(manifest)
    old = read(controller / "state.before.json")
    state_path = root / m["output"] / "state.json"
    require(
        m["bindings"][str(baseline.relative_to(root))] == sha(baseline),
        "recovery baseline is not manifest-bound",
    )
    b = read(baseline)["binding"]
    panel = read(root / m["monitor_panel"])
    require(
        b["panel_sha256"] == panel["panel_sha256"]
        and b["corpus_sha256"] == panel["corpus_sha256"]
        and b["executable_sha256"] == m["bindings"][m["monitor_executable"]],
        "recovery baseline identity mismatch",
    )
    gp_pid = subprocess.check_output(
        ["systemctl", "show", "jope-parallel-gp.service", "-p", "MainPID", "--value"], text=True
    ).strip()
    repair = deployment / "health-repair"
    repair.mkdir(exist_ok=True)
    source = controller / "jope-parallel-health.service.conf"
    content = source.read_text()
    lines = content.splitlines()
    line = next(i for i, text in enumerate(lines) if text.startswith("ExecStart=") and "--baseline" in text)
    command = shlex.split(lines[line].partition("=")[2])
    command[command.index("--baseline") + 1] = str(baseline)
    require(
        command[command.index("--manifest") + 1] == str(manifest),
        "health repair would change active manifest",
    )
    lines[line] = "ExecStart=" + shlex.join(command)
    new = "\n".join(lines) + "\n"
    immutable(
        repair / "request.json",
        dict(
            before=content,
            after=new,
            manifest_sha256=sha(manifest),
            baseline_sha256=sha(baseline),
            driver_sha256=sha(Path(__file__)),
            gp_pid=gp_pid,
        ),
    )
    path = repair / "health.conf"
    path.write_text(new)
    target = Path("/etc/systemd/system/jope-parallel-health.service.d/zz-promotion-safeguards.conf")
    subprocess.run(["sudo", "-n", "install", "-m", "644", str(path), str(target)], check=True)
    subprocess.run(["sudo", "-n", "systemctl", "daemon-reload"], check=True)
    verify_effective_launches({"jope-parallel-health.service": new})
    require(
        read(root / "runs/continuous-progress.json")["manifest_sha256"] == sha(manifest),
        "controller has not published the new manifest",
    )
    subprocess.run(["sudo", "-n", "systemctl", "start", "jope-parallel-health.service"], check=True)
    state = read(state_path)
    for key in ("checkpoint", "serving_checkpoint", "serving_checkpoint_sha256", "serving_status"):
        require(state[key] == old[key], "handoff changed serving: " + key)
    for key in ("candidates", "promotion_decisions", "consumed_training_snapshots"):
        require(all(row in state[key] for row in old[key]), "handoff lost durable state: " + key)
    require(
        state["manifest_sha256"]
        == read(root / "runs/continuous-control.json")["manifest_sha256"]
        == sha(manifest),
        "active controller identity mismatch",
    )
    gp = read(root / "runs/gp-service.json")
    require(
        gp["executable_sha256"] == m["bindings"][m["cpu_executable"]]
        and gp["workers"] == 104
        and gp["threads_each"] == 1
        and gp["slot_leases"],
        "GP release/allocation mismatch",
    )
    require(
        subprocess.check_output(
            ["systemctl", "show", "jope-parallel-gp.service", "-p", "MainPID", "--value"], text=True
        ).strip()
        == gp_pid,
        "GP restarted during health repair",
    )
    immutable(
        repair / "complete.json",
        dict(timestamp=time.time(), request_sha256=sha(repair / "request.json"), status="passed"),
    )
    immutable(
        controller / "complete.json",
        dict(
            timestamp=time.time(),
            manifest_sha256=sha(manifest),
            serving_checkpoint=state["serving_checkpoint"],
            serving_checkpoint_sha256=state["serving_checkpoint_sha256"],
            serving_status=state["serving_status"],
            gp_uninterrupted=True,
            health_repair_sha256=sha(repair / "complete.json"),
            final_tests_opened=False,
        ),
    )
    immutable(
        deployment / "complete.json",
        dict(
            timestamp=time.time(),
            request_sha256=sha(deployment / "request.json"),
            drained_sha256=sha(deployment / "drained.json"),
            controller_sha256=sha(controller / "complete.json"),
            executable_sha256=gp["executable_sha256"],
            workers=104,
            threads_each=1,
            recovered_attempts=["cold startup observer deadline", "inherited health baseline"],
            final_tests_opened=False,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "deployment", "baseline"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    main(a.root.resolve(), a.deployment.resolve(), a.baseline.resolve())
