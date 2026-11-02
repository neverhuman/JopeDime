"""H229 single-attempt source-bound supervisor; load pauses happen between context units."""

from pathlib import Path
import argparse
import datetime as dt
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import types

REPO = Path("/home/ubuntu/JopeDime")
PREFIX = "research/flywheel_selfplay/h229_pool_headroom_replication/"
DATA = Path.home() / "jope-data/h229"


def source_module(source, path, name):
    raw = subprocess.check_output(
        ["git", "show", source + ":" + PREFIX + path], cwd=REPO
    )
    m = types.ModuleType(name)
    m.__file__ = str(REPO / PREFIX / path)
    sys.modules[name] = m
    exec(compile(raw, m.__file__, "exec"), m.__dict__)
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--phase", choices=["smoke", "run"], required=True)
    args = parser.parse_args()
    os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    os.environ["H229_SOURCE"] = args.source
    b = source_module(args.source, "backend.py", "backend")
    protocol = source_module(args.source, "protocol.py", "h229_protocol")
    doc = b.registration()
    subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            args.source,
            "refs/agent/res08/h229/main",
        ],
        cwd=REPO,
        check=True,
    )
    DATA.mkdir(parents=True, exist_ok=True)
    owner_lock = (DATA / "H229.owner.lock").open("a+")
    fcntl.flock(owner_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.phase == "smoke":
        protocol.DEADLINE = min(
            protocol.DEADLINE,
            dt.datetime.fromisoformat("2026-11-02T22:30:00+00:00").timestamp(),
        )
    if args.phase == "smoke":
        roots = [DATA / "smoke-001"]
        root = next((r for r in roots if not r.exists()), None)
        if (
            root is None
            or time.time()
            >= dt.datetime.fromisoformat("2026-11-02T22:30:00+00:00").timestamp()
        ):
            raise RuntimeError("one smoke attempt or smoke deadline exhausted")
        smoke = None
    else:
        root = DATA / "attempt-001"
        assert not root.exists(), "one primary attempt; no retry/resume"
        passed = [
            r / "metrics.json"
            for r in sorted(DATA.glob("smoke-*"))
            if (r / "metrics.json").exists()
            and json.loads((r / "metrics.json").read_text()).get("status")
            == "SMOKE_PASS"
            and json.loads((r / "metrics.json").read_text()).get("source_commit")
            == args.source
        ]
        if not passed:
            raise RuntimeError("exact-source full-path smoke missing")
        smoke = passed[-1]
    # Read-only admission before consuming an attempt or copying a model.
    protocol.wait(DATA)
    root.mkdir()
    (DATA / "input").mkdir(exist_ok=True)
    if not b.R882.exists():
        shutil.copyfile(doc["checkpoint_copy_source"], b.R882)
    if b.digest(b.R882) != b.R882_SHA:
        raise RuntimeError("owned checkpoint COPY changed")
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        RAYON_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
        H229_SOURCE=args.source,
    )
    bootstrap = "import subprocess,sys,types; s=sys.argv.pop(1); p='research/flywheel_selfplay/h229_pool_headroom_replication/'; r='/home/ubuntu/JopeDime'; raw=subprocess.check_output(['git','show',s+':'+p+'backend.py'],cwd=r); m=types.ModuleType('backend');m.__file__=r+'/'+p+'backend.py';sys.modules['backend']=m;exec(compile(raw,m.__file__,'exec'),m.__dict__);raw=subprocess.check_output(['git','show',s+':'+p+'runner.py'],cwd=r);exec(compile(raw,r+'/'+p+'runner.py','exec'),{'__name__':'__main__','__file__':r+'/'+p+'runner.py'})"
    argv = [
        sys.executable,
        "-B",
        "-c",
        bootstrap,
        args.source,
        str(root),
        args.phase,
    ] + ([str(smoke)] if smoke else [])
    with (
        (root / "worker.stdout").open("x") as out,
        (root / "worker.stderr").open("x") as err,
    ):
        child = subprocess.Popen(
            argv, cwd=REPO, env=env, stdout=out, stderr=err, start_new_session=True
        )
    b.write(
        root / "launch.json",
        dict(
            source_commit=args.source,
            source_tree=subprocess.check_output(
                ["git", "rev-parse", args.source + "^{tree}"], cwd=REPO, text=True
            ).strip(),
            phase=args.phase,
            worker_pid=child.pid,
            supervisor_pid=os.getpid(),
            started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            smoke_sha256=b.digest(smoke) if smoke else None,
            compute_deadline_utc=doc["resources"]["compute_deadline_utc"],
        ),
    )
    import psutil

    process = psutil.Process(child.pid)
    started = process.create_time()
    failure = None
    try:
        last = 0
        soft_paused = False
        last_pause_log = 0.0
        while child.poll() is None:
            try:
                o = protocol.observation(root, process)
            except psutil.NoSuchProcess:
                if child.poll() is not None:
                    break
                raise
            try:
                # Hard limits checked every second; soft hysteresis rechecked every60s.
                protocol.admission(o, soft_paused)
                if not soft_paused or time.monotonic() - last_pause_log >= 60:
                    next_paused, pause_fields = protocol.shared_admission(root, o)
                else:
                    next_paused = soft_paused
            except protocol.HardStop as guard_error:
                protocol.record(root, "HARD_STOP", o, guard_error.fields)
                raise
            if next_paused and (
                not soft_paused or time.monotonic() - last_pause_log >= 60
            ):
                protocol.record(
                    root,
                    "PAUSE_REQUEST" if not soft_paused else "PAUSE_RECHECK",
                    o,
                    pause_fields,
                )
                last_pause_log = time.monotonic()
            elif soft_paused and not next_paused:
                protocol.record(root, "RESUME_ALLOWED", o, [])
            soft_paused = next_paused
            for p in [process, *process.children(recursive=True)]:
                try:
                    if p.nice() != 19:
                        raise RuntimeError("child nice changed")
                    for tid in p.threads():
                        if not set(psutil.Process(tid.id).cpu_affinity()) <= set(
                            os.sched_getaffinity(0)
                        ):
                            raise RuntimeError(
                                "child thread affinity escaped four-core allocation"
                            )
                except psutil.NoSuchProcess:
                    pass
            if time.monotonic() - last > 10:
                with (root / "resources.jsonl").open("a") as f:
                    f.write(json.dumps(o) + "\n")
                last = time.monotonic()
            time.sleep(1)
    except BaseException as exc:
        failure = str(exc)
        if not isinstance(exc, protocol.HardStop):
            protocol.record(
                root,
                "SUPERVISOR_ERROR",
                locals().get("o", {}),
                [{"field": "supervisor_exception", "value": str(exc)}],
            )
        if process.is_running() and process.create_time() == started:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
    code = child.wait()
    b.write(
        root / "terminal.json",
        dict(
            at=dt.datetime.now(dt.timezone.utc).isoformat(),
            worker_exit=code,
            error=failure,
            source_commit=args.source,
            phase=args.phase,
            scientific_retry=False,
        ),
    )
    if failure:
        path = root / "metrics.json"
        metrics = (
            json.loads(path.read_text()) if path.exists() else dict(optimizer_updates=0)
        )
        metrics.update(
            status="INCOMPLETE",
            verdict="INCOMPLETE",
            supervisor_error=failure,
            actual_worker_exit=code,
        )
        b.write(path, metrics)
    return code if code else int(failure is not None)


if __name__ == "__main__":
    raise SystemExit(main())
