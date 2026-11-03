"""Shared fixtures and helpers for capture-client tests."""

"""Data-free tests for the A5 custody packager."""

import ast

import importlib.util

import argparse

import hashlib

import inspect

import json

import os

from pathlib import Path

import signal

import subprocess

import sys

import tempfile

import textwrap

import time

import pytest

SCRIPT = Path(__file__).with_name("capture_clients.py")

SPEC = importlib.util.spec_from_file_location("capture_clients", SCRIPT)

MODULE = importlib.util.module_from_spec(SPEC)

assert SPEC and SPEC.loader

SPEC.loader.exec_module(MODULE)

THREE_CLIENTS_SCRIPT = Path(__file__).with_name("capture_three_clients.py")

THREE_CLIENTS_SPEC = importlib.util.spec_from_file_location(
    "capture_three_clients", THREE_CLIENTS_SCRIPT
)

assert THREE_CLIENTS_SPEC and THREE_CLIENTS_SPEC.loader

THREE_CLIENTS_MODULE = importlib.util.module_from_spec(THREE_CLIENTS_SPEC)

THREE_CLIENTS_SPEC.loader.exec_module(THREE_CLIENTS_MODULE)

def _runner_fixture(
    tmp_path,
    *,
    sleep_key=None,
    descendant_key=None,
    drain_key=None,
    transport_key=None,
    flush_seconds=0.2,
    recorder_seconds=2,
    async_recorder_key=None,
    recorder_group_key=None,
):
    repo = tmp_path / "source"
    repo.mkdir()
    capture_script = (
        repo / "deploy/qualification/agents/transcripts/run-prep/capture_stdio.py"
    )
    capture_script.parent.mkdir(parents=True)
    if transport_key is None:
        capture_script.write_text(
            "# recorder identity fixture for packaging controls\n", encoding="utf-8"
        )
    else:
        capture_script.write_bytes(
            Path(__file__).with_name("capture_stdio.py").read_bytes()
        )
        capture_script.with_name("verify_capture.py").write_bytes(
            Path(__file__).with_name("verify_capture.py").read_bytes()
        )
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "fixture"], check=True
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "fixture@example.invalid"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
    source = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()

    candidate = tmp_path / "inert-candidate.py"
    candidate.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env python3
            import json, os, signal, sys, time
            from pathlib import Path
            home = Path(os.environ['JOPEDIME_HOME'])
            args = sys.argv[1:]
            if args == ['--version', '--json']:
                print(json.dumps({{'source_sha40': {source!r}}}))
            elif args == ['serve', '--idle-exit', '15m']:
                home.mkdir(parents=True, exist_ok=True)
                def stop(*unused):
                    (home / 'stopped').write_text('yes')
                    # jankurai:allow HLT-008-FALSE-GREEN-RISK reason="legitimate exit/skip check" expires=2026-12-31
                    raise SystemExit(0)
                signal.signal(signal.SIGTERM, stop)
                (home / 'pid').write_text(str(os.getpid()))
                while True:
                    time.sleep(0.05)
            elif args == ['serve', '--status']:
                print('running' if (home / 'pid').is_file() else 'stopped')
                # jankurai:allow HLT-008-FALSE-GREEN-RISK reason="legitimate exit/skip check" expires=2026-12-31
                raise SystemExit(0 if (home / 'pid').is_file() else 1)
            elif args == ['serve', '--stop']:
                try:
                    os.kill(int((home / 'pid').read_text()), signal.SIGTERM)
                except (OSError, ValueError):
                    # jankurai:allow HLT-008-FALSE-GREEN-RISK reason="legitimate exit/skip check" expires=2026-12-31
                    raise SystemExit(1)
            elif args == ['mcp']:
                for line in sys.stdin:
                    request = json.loads(line)
                    print(json.dumps({{'jsonrpc': '2.0', 'id': request['id'], 'result': {{}}}}), flush=True)
            else:
                # jankurai:allow HLT-008-FALSE-GREEN-RISK reason="legitimate exit/skip check" expires=2026-12-31
                raise SystemExit(2)
            """
        ),
        encoding="utf-8",
    )
    candidate.chmod(0o700)

    client = tmp_path / "inert-client.py"
    client.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env python3
            import hashlib, json, os, signal, subprocess, sys, time
            from pathlib import Path
            _, capture_stdio, capture_dir, candidate, engine_home, key = sys.argv
            Path('client.pid').write_text(str(os.getpid()))
            if key == {drain_key!r}:
                # jankurai:allow HLT-008-FALSE-GREEN-RISK reason="legitimate exit/skip check" expires=2026-12-31
                child_code = "import hashlib, json, os, signal, sys, time; from pathlib import Path; signal.signal(signal.SIGTERM, lambda *_: sys.exit(0)); Path('drain-child.pid').write_text(str(os.getpid())); Path('drain-child.ready').write_text('ready'); deadline=time.monotonic()+5\\nwhile not Path('drain-child.release').exists() and time.monotonic()<deadline: time.sleep(0.01)\\nif Path('drain-child.release').exists():\\n output=Path(sys.argv[1]); candidate=Path(sys.argv[2]); output.mkdir(parents=True, exist_ok=True); (output/'wire.jsonl').write_text(json.dumps(dict(inventory_counts=[0,1,1]))); receipt=dict(complete=True,producer_sha256=hashlib.sha256(Path(sys.argv[3]).read_bytes()).hexdigest(),argv=[str(candidate),'mcp'],executable_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),server_exit_code=0,relay_exit_code=0,limits=dict(seconds=120,stream_bytes=16777216,frame_bytes=1048576),extra_environment_names=[]); (output/'capture.json').write_text(json.dumps(receipt))"
                subprocess.Popen([sys.executable, '-c', child_code, capture_dir, candidate, capture_stdio])
                deadline = time.monotonic() + 5
                while not Path('drain-child.ready').exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                if not Path('drain-child.ready').exists():
                    # jankurai:allow HLT-008-FALSE-GREEN-RISK reason="legitimate exit/skip check" expires=2026-12-31
                    raise SystemExit('drain descendant did not start')
                # jankurai:allow HLT-008-FALSE-GREEN-RISK reason="legitimate exit/skip check" expires=2026-12-31
                raise SystemExit(0)
            if key == {async_recorder_key!r}:
                child_code = "import hashlib, json, os, sys, time; from pathlib import Path; Path('async-recorder.pid').write_text(str(os.getpid())); time.sleep(0.4); output=Path(sys.argv[1]); candidate=Path(sys.argv[2]); output.mkdir(parents=True, exist_ok=True); (output/'wire.jsonl').write_text(json.dumps(dict(inventory_counts=[0,1,1]))); receipt=dict(complete=True,producer_sha256=hashlib.sha256(Path(sys.argv[3]).read_bytes()).hexdigest(),argv=[str(candidate),'mcp'],executable_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),server_exit_code=0,relay_exit_code=0,limits=dict(seconds=120,stream_bytes=16777216,frame_bytes=1048576),extra_environment_names=[]); (output/'capture.json').write_text(json.dumps(receipt))"
                subprocess.Popen(
                    [sys.executable, '-c', child_code, capture_dir, candidate, capture_stdio],
                    start_new_session=True,
                )
                # jankurai:allow HLT-008-FALSE-GREEN-RISK reason="legitimate exit/skip check" expires=2026-12-31
                raise SystemExit(0)
            if key == {descendant_key!r}:
                child_code = "import os, signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); open('descendant.pid', 'w').write(str(os.getpid())); time.sleep(120)"
                subprocess.Popen([sys.executable, '-c', child_code])
                deadline = time.monotonic() + 5
                while not Path('descendant.pid').exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                if not Path('descendant.pid').exists():
                    # jankurai:allow HLT-008-FALSE-GREEN-RISK reason="legitimate exit/skip check" expires=2026-12-31
                    raise SystemExit('descendant did not start')
            if key == {transport_key!r}:
                recorder_options = {{'process_group': 0}} if key == {recorder_group_key!r} else {{}}
                recorder = subprocess.Popen([sys.executable, '-B', capture_stdio, '--out-dir', capture_dir, '--seconds', {str(recorder_seconds)!r}, '--max-bytes', '1048576', '--', candidate, 'mcp'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, **recorder_options)
                recorder.stdin.write(b'{{"jsonrpc":"2.0","id":1,"method":"ping"}}\\n')
                recorder.stdin.flush()
                if not recorder.stdout.readline():
                    # jankurai:allow HLT-008-FALSE-GREEN-RISK reason="legitimate exit/skip check" expires=2026-12-31
                    raise SystemExit('recorder produced no response')
                Path('recorder.pid').write_text(str(recorder.pid))
                if key == {sleep_key!r}:
                    time.sleep(30)
                flush_code = "import os,time;from pathlib import Path;time.sleep({flush_seconds!r});Path('recorder-flushed').write_text('yes');os.close(" + str(recorder.stdin.fileno()) + ")"
                subprocess.Popen([sys.executable, '-B', '-c', flush_code], pass_fds=(recorder.stdin.fileno(),), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # jankurai:allow HLT-008-FALSE-GREEN-RISK reason="legitimate exit/skip check" expires=2026-12-31
                raise SystemExit(0)
            if key == {sleep_key!r}:
                time.sleep(30)
            home = Path(engine_home)
            state = home / 'runs.json'
            runs = json.loads(state.read_text()) if state.exists() else []
            before = len(runs)
            runs.append(key)
            state.write_text(json.dumps(runs))
            output = Path(capture_dir)
            output.mkdir(parents=True)
            (output / 'wire.jsonl').write_text(json.dumps({{'inventory_counts': [before, len(runs), len(runs)]}}) + '\\n')
            receipt = {{'complete': True, 'argv': [candidate, 'mcp'], 'producer_sha256': hashlib.sha256(Path(capture_stdio).read_bytes()).hexdigest(),
                'executable_sha256': hashlib.sha256(Path(candidate).read_bytes()).hexdigest(),
                'server_exit_code': 0, 'relay_exit_code': 0,
                'limits': {{'seconds': 120, 'stream_bytes': 16777216, 'frame_bytes': 1048576}},
                'extra_environment_names': []}}
            (output / 'capture.json').write_text(json.dumps(receipt))
            """
        ),
        encoding="utf-8",
    )
    client.chmod(0o700)

    candidate_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
    client_sha = hashlib.sha256(client.read_bytes()).hexdigest()
    handoff = tmp_path / "handoff.json"
    handoff.write_text(
        json.dumps({"split": "TRAIN", "dataset_id": "synthetic-fixture-only"}),
        encoding="utf-8",
    )
    train = tmp_path / "train.txt"
    train.write_text("synthetic fixture only", encoding="utf-8")
    plan = {
        "schema": "jopedime.a5-three-client-command-plan/v1",
        "source_sha": source,
        "candidate_artifact_sha256": candidate_sha,
        "clients": {},
    }
    for key, (_slug, implementation) in THREE_CLIENTS_MODULE.CLIENTS.items():
        plan["clients"][key] = {
            "implementation": implementation,
            "version": "SYNTHETIC_ONLY",
            "executable": str(client),
            "executable_sha256": client_sha,
            "argv": [
                str(client),
                "{capture_stdio}",
                "{capture_dir}",
                "{candidate}",
                "{engine_home}",
                key,
            ],
        }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    args = argparse.Namespace(
        repo=repo,
        source_sha=source,
        candidate=candidate,
        candidate_sha256=candidate_sha,
        input_handoff=handoff,
        input_handoff_sha256=hashlib.sha256(handoff.read_bytes()).hexdigest(),
        train_path=train,
        client_plan=plan_path,
        engine_home=tmp_path / "engine-home",
        out_dir=tmp_path / "capture-output",
        preflight_only=False,
        execute_named_clients=True,
    )
    return args, plan, candidate, client

def _processes_with_paths_under(root):
    root = Path(root).resolve()
    root_text = str(root)
    processes = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="ascii")
            stat_fields = stat.rsplit(")", 1)[1].split()
            state, pgid = stat_fields[0], int(stat_fields[2])
            cwd = Path(os.readlink(entry / "cwd"))
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except (OSError, IndexError, ValueError):
            continue
        if state in {"Z", "X"}:
            continue
        cwd_match = cwd == root or root in cwd.parents
        argv_match = any(
            arg == root_text.encode() or arg.startswith((root_text + os.sep).encode())
            for arg in argv
            if arg
        )
        if cwd_match or argv_match:
            processes.append({"pid": int(entry.name), "pgid": pgid, "cwd": str(cwd)})
    return processes

def _kill_fixture_processes(root):
    for process in _processes_with_paths_under(root):
        try:
            os.killpg(process["pgid"], signal.SIGKILL)
        except ProcessLookupError:
            pass

@pytest.fixture
def a5_tmp_path():
    with tempfile.TemporaryDirectory(prefix="a5-runner-test-", dir=Path.home()) as root:
        path = Path(root)
        try:
            yield path
        finally:
            _kill_fixture_processes(path)

def _pid_is_live(pid):
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as stat_file:
            state = stat_file.read().rsplit(")", 1)[1].split()[0]
        return state != "Z"
    except FileNotFoundError:
        return False

def _start_sigterm_ignoring_engine(tmp_path):
    ready = tmp_path / "engine-ready.txt"
    code = (
        "import signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(ready)!r}).write_text('ready'); "
        "print('READY', flush=True); time.sleep(60)"
    )
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", code],
        cwd=tmp_path,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert process.stdout.readline().strip() == "READY"
    return process

def _cleanup_test_engine(process):
    if process.poll() is None:
        assert os.getpgid(process.pid) == process.pid
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=3)
    for stream in (process.stdout, process.stderr):
        if stream is not None:
            stream.close()

def _sigint_blocked():
    return signal.SIGINT in signal.pthread_sigmask(signal.SIG_BLOCK, ())

def _engine_cleanup_finally_lines():
    """Line span of run_capture's engine cleanup `finally:`, found structurally.

    The innermost try statement whose body launches the engine owns the
    cleanup; the first line executed inside its `finally:` is the earliest
    point an interrupt can reach the teardown.
    """
    source, start = inspect.getsourcelines(THREE_CLIENTS_MODULE.run_capture)
    tree = ast.parse("".join(source))
    owners = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        and node.finalbody
        and "--idle-exit" in "\n".join(ast.unparse(stmt) for stmt in node.body)
    ]
    assert owners, "run_capture has no engine cleanup finally"
    owner = max(owners, key=lambda node: node.lineno)
    return (
        start - 1 + owner.finalbody[0].lineno,
        start - 1 + owner.finalbody[-1].end_lineno,
    )


__all__ = [name for name in globals() if not name.startswith("__")]
