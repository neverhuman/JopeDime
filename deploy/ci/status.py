#!/usr/bin/env python3
"""Inspect one exact Actions run without needing an available Actions runner."""
import argparse
import datetime
import json
import re
import subprocess
import sys


def api(path):
    result = subprocess.run(["gh", "api", path], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"GitHub read failed: {path}: {result.stderr.strip()}")
    return json.loads(result.stdout)


def pages(path, key):
    values = []
    for page in range(1, 1000):
        response = api(f"{path}?per_page=100&page={page}")
        values.extend(response[key])
        if len(response[key]) < 100:
            return values
    raise RuntimeError("pagination exceeded explicit bound")


def describe(run, jobs, runners, now):
    queued = []
    for job in jobs:
        if job["status"] != "queued":
            continue
        since = job.get("created_at") or job.get("started_at")
        seconds = None if not since else max(0, (now - datetime.datetime.fromisoformat(since.replace("Z", "+00:00"))).total_seconds())
        labels = job.get("labels", [])
        matching = [runner for runner in runners if set(label.casefold() for label in labels).issubset({label["name"].casefold() for label in runner["labels"]})]
        queued.append({"job_id": job["id"], "name": job["name"], "requested_labels": labels,
                       "queued_seconds": seconds, "queue_time_source": "job" if since else "unavailable",
                       "over_ten_minutes": None if seconds is None else seconds >= 600,
                       "matching_online_runners": [{"name": runner["name"], "busy": runner["busy"]} for runner in matching if runner["status"] == "online"],
                       "matching_offline_runners": [runner["name"] for runner in matching if runner["status"] != "online"]})
    return {"run_id": run["id"], "attempt": run["run_attempt"], "source_sha": run["head_sha"],
            "status": run["status"], "conclusion": run["conclusion"], "url": run["html_url"],
            "observed_at": now.isoformat(), "queued_jobs": queued,
            "jobs": [{"id": job["id"], "name": job["name"], "status": job["status"], "conclusion": job["conclusion"]} for job in jobs]}


def probe(host):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", host) or host.startswith("-"):
        raise ValueError("host must be a plain configured SSH host name")
    # Fixed read-only commands; never inspect environment, process arguments or logs
    # that may contain JIT registration secrets.
    script = '''import json,subprocess
commands=[['systemctl','show','jope-runner@ci.service','jope-runner@ci2.service','jope-runner@full.service','-p','Id','-p','LoadState','-p','ActiveState','-p','SubState','-p','Result','-p','ExecMainStatus'],['df','-B1','--output=target,avail','/','/var/lib/docker']]
rows=[]
for command in commands:
 r=subprocess.run(command,capture_output=True,text=True,timeout=10)
 rows.append({'command':command,'exit_status':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
print(json.dumps(rows))
'''
    command = ["python3", "-"] if host == "local" else ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host, "python3", "-"]
    try:
        result = subprocess.run(command, input=script, text=True, capture_output=True, timeout=30)
        return {"host": host, "status": "observed" if result.returncode == 0 else "unavailable", "exit_status": result.returncode,
                "observations": json.loads(result.stdout) if result.returncode == 0 else [], "diagnostic": result.stderr.strip()}
    except (subprocess.TimeoutExpired, ValueError) as exc:
        return {"host": host, "status": "unavailable", "diagnostic": str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", type=int)
    parser.add_argument("--repo", default="veox-ai/JopeDime")
    parser.add_argument("--expected-sha")
    parser.add_argument("--host", action="append", help="SSH host to observe after ten minutes queued; default xbabe1 and xbabe2")
    parser.add_argument("--no-host-probes", action="store_true")
    args = parser.parse_args()
    try:
        run = api(f"repos/{args.repo}/actions/runs/{args.run_id}")
        if args.expected_sha and run["head_sha"] != args.expected_sha:
            raise ValueError("requested source SHA does not match this run")
        jobs = pages(f"repos/{args.repo}/actions/runs/{args.run_id}/attempts/{run['run_attempt']}/jobs", "jobs")
        runners = pages(f"repos/{args.repo}/actions/runners", "runners")
        report = describe(run, jobs, runners, datetime.datetime.now(datetime.timezone.utc))
        report["host_observations"] = []
        if any(job["over_ten_minutes"] for job in report["queued_jobs"]) and not args.no_host_probes:
            report["host_observations"] = [probe(host) for host in (args.host or ["xbabe1", "xbabe2"])]
        report["host_probe_status"] = "observed" if report["host_observations"] else "not-requested-or-under-threshold"
        print(json.dumps(report, indent=2))
        return 1 if run["status"] == "completed" and run["conclusion"] != "success" else 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(json.dumps({"run_id": args.run_id, "status": "unavailable", "diagnostic": str(exc)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
