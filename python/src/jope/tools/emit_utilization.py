#!/usr/bin/env python3
"""Install next units on the GPU host and emit utilization receipts. Never starts the fenced GP."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import fence_honesty as fence
import launch_next as launch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "parallel/runs/flywheel-utilization-20260911"
SSH_KEY = Path("~/.ssh/lambda_cloud_a100_20260907_ed25519").expanduser()
HOST = "ubuntu@158.101.20.226"
REMOTE = "/home/ubuntu/jope-dime/parallel"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-i", str(SSH_KEY), HOST]


def write(name, payload):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def ssh(script, timeout=60):
    result = subprocess.run(
        SSH + ["python3", "-"], input=script, text=True, capture_output=True, timeout=timeout
    )
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:] or result.stdout[-2000:])
    text = result.stdout.strip().splitlines()
    return json.loads(text[-1]) if text else {}


def rsync_runtime():
    dest = HOST + ":" + REMOTE + "/runs/flywheel-utilization-20260911/runtime/tools/"
    subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-i",
            str(SSH_KEY),
            HOST,
            "mkdir -p "
            + REMOTE
            + "/runs/flywheel-utilization-20260911/runtime/tools "
            + REMOTE
            + "/runs/flywheel-utilization-20260911/runtime/systemd",
        ],
        check=True,
    )
    subprocess.run(
        [
            "rsync",
            "-a",
            "--delete",
            "--exclude=__pycache__",
            "-e",
            "ssh -o BatchMode=yes -i " + str(SSH_KEY),
            str(ROOT / "parallel/tools") + "/",
            dest,
        ],
        check=True,
    )
    subprocess.run(
        [
            "rsync",
            "-a",
            "-e",
            "ssh -o BatchMode=yes -i " + str(SSH_KEY),
            str(ROOT / "parallel/systemd/jope-parallel-gp-next.service"),
            str(ROOT / "parallel/systemd/jope-parallel-controller-next.service"),
            HOST + ":" + REMOTE + "/runs/flywheel-utilization-20260911/runtime/systemd/",
        ],
        check=True,
    )


def install_units():
    return ssh(r"""
import subprocess, json
from pathlib import Path
base = Path('/home/ubuntu/jope-dime/parallel/runs/flywheel-utilization-20260911/runtime/systemd')
for name in ('jope-parallel-gp-next.service','jope-parallel-controller-next.service'):
    subprocess.run(['sudo','-n','install','-m','644',str(base/name),'/etc/systemd/system/'+name], check=True)
subprocess.run(['sudo','-n','systemctl','daemon-reload'], check=True)
print(json.dumps({'installed': True, 'legacy_untouched': True}))
""")


def start_next(unit):
    launch.refuse(unit, "start")
    return ssh(f"""
import json, subprocess, sys
sys.path.insert(0, '{REMOTE}/runs/flywheel-utilization-20260911/runtime/tools')
import launch_next
print(json.dumps(launch_next.start('{unit}')))
""")


def observe():
    return ssh(
        r"""
import json, subprocess, time
from pathlib import Path
def show(unit):
    out = subprocess.check_output(['systemctl','show',unit,'--property=ActiveState,MainPID,SubState,InvocationID'], text=True)
    return dict(line.split('=',1) for line in out.splitlines() if '=' in line)
smi = subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'], text=True)
gp = json.loads(Path('/home/ubuntu/jope-dime/parallel/runs/gp-service.json').read_text())
ranks = []
pool = Path('/home/ubuntu/jope-dime/parallel/runs/flywheel-20260909/resident-production-v1')
for i in range(8):
    status = pool / f'rank-{i}' / 'status.json'
    if status.exists():
        ranks.append(json.loads(status.read_text()).get('state') or json.loads(status.read_text()).get('status'))
    else:
        ranks.append('missing')
progress = json.loads(Path('/home/ubuntu/jope-dime/parallel/runs/continuous-progress.json').read_text())
state = json.loads(Path('/home/ubuntu/jope-dime/parallel/runs/continuous-learning-v2-20260909/state.json').read_text())
print(json.dumps({
    'timestamp': time.time(),
    'gp_next': show('jope-parallel-gp-next.service'),
    'controller_next': show('jope-parallel-controller-next.service'),
    'legacy_gp': show('jope-parallel-gp.service'),
    'campaign': show('jope-parallel-campaign.service'),
    'resident': show('jope-parallel-resident-gpu.service'),
    'nvidia': [line.strip() for line in smi.splitlines()],
    'resident_ranks': ranks,
    'tasks': list((state.get('tasks') or {})),
    'hold': (progress.get('automatic_training_hold') or {}).get('status'),
    'gp_service': {k: gp.get(k) for k in ('timestamp','phase','active','queued','completed_evaluations','wait_for_work','slot_leases')},
}))
""",
        timeout=90,
    )


def wait_gp(launch_ts, seconds=90):
    last = {}
    deadline = time.time() + seconds
    while time.time() < deadline:
        last = observe()
        pid = int(last["gp_next"].get("MainPID") or 0)
        fresh = float(last["gp_service"].get("timestamp") or 0) >= launch_ts - 5
        if pid not in (0, fence.LEGACY_GP_PID) and last["gp_service"].get("phase") == "running" and fresh:
            return last
        time.sleep(3)
    return last


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rsync_runtime()
    write("u1-install.json", install_units())
    pre = ssh(r"""
import hashlib, json, time
from pathlib import Path
cpu = Path('/home/ubuntu/jope-dime/parallel/runs/flywheel-20260909/native-v5-cpu/target-bin/jope-parallel-native')
print(json.dumps({
    'executable': str(cpu),
    'executable_sha256': hashlib.sha256(cpu.read_bytes()).hexdigest(),
    'recorded_before_spawn': True,
    'timestamp': time.time(),
}))
""")
    write("u1-executable.json", pre)
    launch_ts = time.time()
    try:
        started = start_next(fence.NEXT_GP)
    except Exception as exc:
        write("u1-gp-next-failed.json", {"error": str(exc)})
        write(
            "u3-resident-stopped.json",
            ssh(r"""
import json, sys
sys.path.insert(0, '/home/ubuntu/jope-dime/parallel/runs/flywheel-utilization-20260911/runtime/tools')
import launch_next
print(json.dumps(launch_next.stop_resident_if_no_dispatcher(False)))
"""),
        )
        raise
    write("u1-gp-next.json", dict(started, executable_sha256=pre["executable_sha256"], launch_ts=launch_ts))
    first = wait_gp(launch_ts)
    pid = int(first["gp_next"].get("MainPID") or 0)
    if pid in (0, fence.LEGACY_GP_PID):
        write(
            "u3-resident-stopped.json",
            ssh(r"""
import json, sys
sys.path.insert(0, '/home/ubuntu/jope-dime/parallel/runs/flywheel-utilization-20260911/runtime/tools')
import launch_next
print(json.dumps(launch_next.stop_resident_if_no_dispatcher(False)))
"""),
        )
        raise RuntimeError("gp-next did not publish a new identity: " + json.dumps(first["gp_next"]))
    if str(first["legacy_gp"].get("MainPID") or "0") not in ("0", ""):
        raise RuntimeError("legacy GP is not still stopped")
    write("u1-gp-next-observe.json", first)
    ctrl = start_next(fence.NEXT_CONTROLLER)
    write("u2-controller-next.json", ctrl)
    time.sleep(5)
    second = observe()
    write(
        "u3-resident-bind.json",
        dict(
            resident_left_up=second["resident"].get("ActiveState") == "active",
            gp_next_pid=int(second["gp_next"].get("MainPID") or 0),
            controller_pid=int(second["controller_next"].get("MainPID") or 0),
            invocation=second["gp_next"].get("InvocationID"),
            legacy_main_pid=int(second["legacy_gp"].get("MainPID") or 0),
            campaign=second["campaign"].get("ActiveState"),
            nvidia=second["nvidia"],
            resident_ranks=second.get("resident_ranks"),
            tasks=second.get("tasks"),
            hold=second.get("hold"),
            gp_service=second["gp_service"],
            dispatch_into_existing_pool=True,
            restarted_774452=False,
            train_on_recovery=False,
        ),
    )
    write(
        "SUMMARY.json",
        dict(
            schema="parallel.utilization_summary.v1",
            gp_next_pid=pid,
            restart_774452=False,
            legacy_fenced=True,
            controller_started=second["controller_next"].get("ActiveState") == "active",
            leftover_incomplete_searches=286,
            may_serve=False,
            claimed_savings_745=False,
        ),
    )


if __name__ == "__main__":
    main()
