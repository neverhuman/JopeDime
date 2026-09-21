#!/usr/bin/env python3
"""Poll the GPU host until propose kernels leave 0% or the sample budget ends."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import useful_hour as useful

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "parallel/runs/flywheel-utilization-20260911"
SSH_KEY = Path("~/.ssh/lambda_cloud_a100_20260907_ed25519").expanduser()
HOST = "ubuntu@158.101.20.226"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-i", str(SSH_KEY), HOST]
LAUNCH = 1789097627.4435108


def ssh():
    script = r"""
import json, subprocess, time
from pathlib import Path
root = Path('/home/ubuntu/jope-dime/parallel')
gp = json.loads((root/'runs/gp-service.json').read_text())
state = json.loads((root/'runs/continuous-learning-v2-20260909/state.json').read_text())
progress = json.loads((root/'runs/continuous-progress.json').read_text())
smi = subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'], text=True)
ranks = []
pool = root/'runs/flywheel-20260909/resident-production-v1'
for i in range(8):
    row = json.loads((pool/f'rank-{i}.json').read_text())
    ranks.append(row.get('status'))
last = (state.get('generations') or [{}])[-1]
remaining = 0
for generation in reversed(state.get('generations') or []):
    if generation.get('stage') != 4:
        continue
    for job in json.loads((root/generation['path']/'jobs.json').read_text())['jobs']:
        remaining += not (root/job['output']/'search.json').exists()
        if remaining >= 400:
            break
    if remaining >= 400:
        break
search_after = 0
base = root/'runs/continuous-learning-v2-20260909'
for marker in base.glob('generation-00005[3-9]/**/complete.json'):
    if marker.stat().st_mtime >= 1789097627:
        search_after += 1
for marker in base.glob('generation-000054/**/complete.json'):
    if marker.stat().st_mtime >= 1789097627:
        search_after += 1
print(json.dumps({
    'now': time.time(),
    'gp': {k: gp.get(k) for k in ('timestamp','phase','active','queued','completed_evaluations')},
    'tasks': list(state.get('tasks') or {}),
    'hold': bool(state.get('automatic_training_hold')),
    'automatic_training': progress.get('automatic_training'),
    'idle_explanation': progress.get('idle_explanation'),
    'last_generation': {'number': last.get('number'), 'stage': last.get('stage')},
    'ranks': ranks,
    'nvidia': [line.strip() for line in smi.splitlines()],
    'legacy_pid': subprocess.check_output(['systemctl','show','jope-parallel-gp.service','-p','MainPID','--value'], text=True).strip(),
    'gp_next_pid': subprocess.check_output(['systemctl','show','jope-parallel-gp-next.service','-p','MainPID','--value'], text=True).strip(),
    'controller_pid': subprocess.check_output(['systemctl','show','jope-parallel-controller-next.service','-p','MainPID','--value'], text=True).strip(),
    'new_propose_complete': search_after,
    'remaining_incomplete': remaining,
}))
"""
    result = subprocess.run(SSH + ["python3", "-"], input=script, text=True, capture_output=True, timeout=90)
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:] or result.stdout[-2000:])
    return json.loads(result.stdout.strip().splitlines()[-1])


def classify(row):
    kernels = [int(x.split(",")[0]) for x in row["nvidia"] if "," in x]
    mean = sum(kernels) / len(kernels) if kernels else 0
    tasks = set(row["tasks"])
    propose = "proposal" in tasks and any(k > 0 for k in kernels)
    gp_eval = int(row["gp"].get("active") or 0) > 0
    pack = "pack" in tasks
    kind = useful.classify(
        propose,
        gp_eval,
        pack,
        bool(row["hold"]) and not (propose or gp_eval or pack),
        mean,
    )
    outputs = []
    if kind["kind"] == "gp-eval" and row["gp"].get("completed_evaluations", 0) > 26381957:
        outputs.append("search.json")
    if kind["kind"] == "pack":
        outputs.append("provenance.json")
    if kind["kind"] == "propose":
        outputs.append("complete.json")
    return kind, kernels, outputs


def main():
    history = {}
    samples = []
    first_kernel = None
    deadline = time.time() + 2400
    while time.time() < deadline:
        row = ssh()
        kind, kernels, outputs = classify(row)
        snap = useful.join(history, row["now"], 8, True, kind["kind"], outputs, kind["kernel_mean"])
        sample = dict(row, kind=kind["kind"], kernels=kernels, outputs=outputs, occupancy=snap)
        samples.append(sample)
        (OUT / "u4-samples.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))
        if first_kernel is None and any(k > 0 for k in kernels):
            first_kernel = sample
            (OUT / "u4-first-kernel.json").write_text(
                json.dumps(first_kernel, indent=2, sort_keys=True) + "\n"
            )
            break
        if "train" in row["tasks"]:
            raise RuntimeError("train task scheduled on recovery; abort")
        if int(row["legacy_pid"] or 0) != 0:
            raise RuntimeError("legacy GP pid is not 0")
        time.sleep(20)
    last = samples[-1] if samples else {}
    (OUT / "u4-useful-hour.json").write_text(
        json.dumps(
            {
                "schema": "parallel.useful_hour.v1",
                "launch_ts": LAUNCH,
                "samples": len(samples),
                "first_kernel_nonzero": first_kernel is not None,
                "first_kernel": None
                if first_kernel is None
                else {
                    "now": first_kernel["now"],
                    "kernels": first_kernel["kernels"],
                    "kind": first_kernel["kind"],
                },
                "last_kind": last.get("kind"),
                "last_tasks": last.get("tasks"),
                "completed_evaluations": (last.get("gp") or {}).get("completed_evaluations"),
                "gp_active": (last.get("gp") or {}).get("active"),
                "occupancy": last.get("occupancy"),
                "vram_residency_is_not_occupancy": True,
                "acceptance": (last.get("occupancy") or {}).get("acceptance", "inconclusive"),
                "may_serve": False,
                "claimed_savings_745": False,
                "train_on_recovery": False,
                "restart_774452": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
