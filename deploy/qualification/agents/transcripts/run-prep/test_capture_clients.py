"""Data-free tests for the A5 custody packager."""

import importlib.util
import argparse
import hashlib
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


def _runner_fixture(tmp_path, *, sleep_key=None, descendant_key=None):
    repo = tmp_path / "source"
    repo.mkdir()
    capture_script = (
        repo / "deploy/qualification/agents/transcripts/run-prep/capture_stdio.py"
    )
    capture_script.parent.mkdir(parents=True)
    capture_script.write_text("# committed synthetic fixture stub\n", encoding="utf-8")
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
                    raise SystemExit(0)
                signal.signal(signal.SIGTERM, stop)
                (home / 'pid').write_text(str(os.getpid()))
                while True:
                    time.sleep(0.05)
            elif args == ['serve', '--status']:
                print('running' if (home / 'pid').is_file() else 'stopped')
                raise SystemExit(0 if (home / 'pid').is_file() else 1)
            elif args == ['serve', '--stop']:
                try:
                    os.kill(int((home / 'pid').read_text()), signal.SIGTERM)
                except (OSError, ValueError):
                    raise SystemExit(1)
            else:
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
            if key == {descendant_key!r}:
                child_code = "import os, signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); open('descendant.pid', 'w').write(str(os.getpid())); time.sleep(120)"
                subprocess.Popen([sys.executable, '-c', child_code])
                deadline = time.monotonic() + 5
                while not Path('descendant.pid').exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                if not Path('descendant.pid').exists():
                    raise SystemExit('descendant did not start')
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
            receipt = {{'complete': True, 'argv': [candidate, 'mcp'],
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


def test_synthetic_package_contract():
    MODULE.self_test()


def test_three_client_capture_plan_self_test():
    THREE_CLIENTS_MODULE.self_test()


def test_invalid_placeholder_refuses_before_any_candidate_process(monkeypatch):
    with tempfile.TemporaryDirectory(
        prefix="a5-plan-preflight-", dir=Path.home()
    ) as temp:
        args, plan, _candidate, _client = _runner_fixture(Path(temp))
        plan["clients"]["claude_code"]["argv"].append("{not_supported}")
        args.client_plan.write_text(json.dumps(plan), encoding="utf-8")
        args.preflight_only = True
        original_run = subprocess.run

        def reject_candidate(command, *positional, **kwargs):
            if command[0] == str(args.candidate):
                pytest.fail("preflight invoked the candidate")
            return original_run(command, *positional, **kwargs)

        monkeypatch.setattr(THREE_CLIENTS_MODULE.subprocess, "run", reject_candidate)
        with pytest.raises(
            THREE_CLIENTS_MODULE.CaptureError, match="unsupported placeholder"
        ):
            THREE_CLIENTS_MODULE.run_capture(args)
        assert not args.out_dir.exists()
        assert not args.engine_home.exists()


def test_runner_isolates_each_client_engine_home_and_run_inventory():
    with tempfile.TemporaryDirectory(
        prefix="a5-isolated-homes-", dir=Path.home()
    ) as temp:
        args, _plan, _candidate, _client = _runner_fixture(Path(temp))
        result = THREE_CLIENTS_MODULE.run_capture(args)
        assert result["status"] == "CAPTURES_COMPLETE_PENDING_SEMANTIC_REVIEW"
        assert result["a5_status"] == "NOT_QUALIFIED"
        assert len(set(result["engine_homes"].values())) == 3
        for key, (slug, _) in THREE_CLIENTS_MODULE.CLIENTS.items():
            assert Path(result["engine_homes"][key]).joinpath("stopped").is_file()
            wire = Path(result["sidecars"]) / f"{slug}.mcp.jsonl"
            assert json.loads(wire.read_text())["inventory_counts"] == [0, 1, 1]
            execution = json.loads(
                (Path(result["sidecars"]) / f"{slug}.execution.json").read_text()
            )
            assert execution["engine_exit"] == execution["engine_stop_exit"] == 0


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


def test_timeout_kills_and_reaps_client_and_stops_its_engine(a5_tmp_path):
    args, plan, _candidate, _client = _runner_fixture(
        a5_tmp_path, sleep_key="claude_code", descendant_key="claude_code"
    )
    plan["clients"]["claude_code"]["timeout_seconds"] = 1
    args.client_plan.write_text(json.dumps(plan), encoding="utf-8")
    try:
        result = THREE_CLIENTS_MODULE.run_capture(args)
        row = result["clients"]["claude_code"]
        assert result["status"] == "INCOMPLETE"
        assert row["timed_out"] is True
        assert row["client_exit"] is not None
        assert Path(row["engine_home"], "stopped").is_file()
        assert _processes_with_paths_under(a5_tmp_path) == []
    finally:
        _kill_fixture_processes(a5_tmp_path)


def _pid_is_live(pid):
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as stat_file:
            state = stat_file.read().rsplit(")", 1)[1].split()[0]
        return state != "Z"
    except FileNotFoundError:
        return False


@pytest.mark.parametrize("timed_out", [False, True], ids=["normal-exit", "timeout"])
def test_client_process_group_descendant_is_gone_after_teardown(timed_out, a5_tmp_path):
    args, plan, _candidate, _client = _runner_fixture(
        a5_tmp_path,
        sleep_key="claude_code" if timed_out else None,
        descendant_key="claude_code",
    )
    case = args.out_dir / "clients" / "claude_code"
    plan["clients"]["claude_code"]["timeout_seconds"] = 1
    args.client_plan.write_text(json.dumps(plan), encoding="utf-8")
    try:
        result = THREE_CLIENTS_MODULE.run_capture(args)
        child_pid = int((case / "descendant.pid").read_text())
        assert not _pid_is_live(child_pid), (
            f"same-group descendant {child_pid} survived {result['status']}"
        )
        assert result["clients"]["claude_code"]["timed_out"] is timed_out
        assert _processes_with_paths_under(a5_tmp_path) == []
    finally:
        _kill_fixture_processes(a5_tmp_path)


def test_interrupt_kills_client_process_group_descendant(monkeypatch, a5_tmp_path):
    args, _plan, _candidate, client = _runner_fixture(
        a5_tmp_path, descendant_key="claude_code"
    )
    case = args.out_dir / "clients" / "claude_code"
    original_popen = subprocess.Popen

    class InterruptAfterDescendant:
        def __init__(self, process):
            self.process = process
            self.interrupted = False

        def __getattr__(self, name):
            return getattr(self.process, name)

        def wait(self, *positional, **kwargs):
            deadline = time.monotonic() + 5
            marker = case / "descendant.pid"
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert marker.exists(), "fixture descendant did not start"
            if not self.interrupted:
                self.interrupted = True
                raise KeyboardInterrupt
            return self.process.wait(*positional, **kwargs)

    def interrupt_client(command, *positional, **kwargs):
        process = original_popen(command, *positional, **kwargs)
        if command[0] == str(client) and command[-1] == "claude_code":
            return InterruptAfterDescendant(process)
        return process

    monkeypatch.setattr(THREE_CLIENTS_MODULE.subprocess, "Popen", interrupt_client)
    try:
        with pytest.raises(KeyboardInterrupt):
            THREE_CLIENTS_MODULE.run_capture(args)
        child_pid = int((case / "descendant.pid").read_text())
        assert not _pid_is_live(child_pid), (
            f"same-group descendant {child_pid} survived interrupt teardown"
        )
        assert _processes_with_paths_under(a5_tmp_path) == []
    finally:
        _kill_fixture_processes(a5_tmp_path)


def test_launch_failure_stops_engine_and_records_incomplete_client(
    monkeypatch, a5_tmp_path
):
    args, _plan, _candidate, client = _runner_fixture(a5_tmp_path)
    original_popen = subprocess.Popen

    def fail_first_client(command, *positional, **kwargs):
        if command[0] == str(client) and command[-1] == "claude_code":
            raise FileNotFoundError("synthetic launch failure")
        return original_popen(command, *positional, **kwargs)

    monkeypatch.setattr(THREE_CLIENTS_MODULE.subprocess, "Popen", fail_first_client)
    try:
        result = THREE_CLIENTS_MODULE.run_capture(args)
        row = result["clients"]["claude_code"]
        assert result["status"] == "INCOMPLETE"
        assert row["client_exit"] is None
        assert row["capture_error"].startswith("client launch failed:")
        assert Path(row["engine_home"], "stopped").is_file()
        assert _processes_with_paths_under(a5_tmp_path) == []
    finally:
        _kill_fixture_processes(a5_tmp_path)


def test_interrupt_terminates_and_reaps_client_and_engine(monkeypatch, a5_tmp_path):
    args, _plan, _candidate, client = _runner_fixture(
        a5_tmp_path, sleep_key="claude_code", descendant_key="claude_code"
    )
    case = args.out_dir / "clients" / "claude_code"
    original_popen = subprocess.Popen
    started_clients = []

    class InterruptOnce:
        def __init__(self, process):
            self.process = process
            self.interrupted = False

        def __getattr__(self, name):
            return getattr(self.process, name)

        def wait(self, *positional, **kwargs):
            deadline = time.monotonic() + 5
            marker = case / "descendant.pid"
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert marker.exists(), "fixture descendant did not start"
            if not self.interrupted:
                self.interrupted = True
                raise KeyboardInterrupt
            return self.process.wait(*positional, **kwargs)

    def interrupt_first_client(command, *positional, **kwargs):
        process = original_popen(command, *positional, **kwargs)
        if command[0] == str(client) and command[-1] == "claude_code":
            wrapper = InterruptOnce(process)
            started_clients.append(wrapper)
            return wrapper
        return process

    monkeypatch.setattr(
        THREE_CLIENTS_MODULE.subprocess, "Popen", interrupt_first_client
    )
    try:
        with pytest.raises(KeyboardInterrupt):
            THREE_CLIENTS_MODULE.run_capture(args)
        assert started_clients and started_clients[0].poll() is not None
        assert (args.engine_home / "claude_code" / "stopped").is_file()
        assert _processes_with_paths_under(a5_tmp_path) == []
    finally:
        _kill_fixture_processes(a5_tmp_path)


@pytest.mark.parametrize(
    "failure", ["cleanup-error", "cleanup-interrupt"], ids=["error", "interrupt"]
)
def test_engine_cleanup_runs_after_client_cleanup_failure(
    failure, monkeypatch, a5_tmp_path
):
    args, _plan, _candidate, _client = _runner_fixture(
        a5_tmp_path, descendant_key="claude_code"
    )
    case = args.out_dir / "clients" / "claude_code"
    original_terminate = THREE_CLIENTS_MODULE.terminate_group

    def cleanup_then_fail(process, *positional, **kwargs):
        result = original_terminate(process, *positional, **kwargs)
        if process.pid == int((case / "client.pid").read_text()):
            if failure == "cleanup-interrupt":
                raise KeyboardInterrupt("injected at client cleanup boundary")
            raise THREE_CLIENTS_MODULE.CaptureError(
                "injected client cleanup boundary failure"
            )
        return result

    monkeypatch.setattr(THREE_CLIENTS_MODULE, "terminate_group", cleanup_then_fail)
    expected = (
        KeyboardInterrupt
        if failure == "cleanup-interrupt"
        else THREE_CLIENTS_MODULE.CaptureError
    )
    try:
        with pytest.raises(expected):
            THREE_CLIENTS_MODULE.run_capture(args)
        assert Path(args.engine_home, "claude_code", "stopped").is_file()
        assert _processes_with_paths_under(a5_tmp_path) == []
    finally:
        _kill_fixture_processes(a5_tmp_path)


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


@pytest.mark.parametrize(
    "wait_error", ["sigint", "oserror"], ids=["actual-sigint", "wait-error"]
)
def test_stop_engine_reaps_group_when_wait_raises(wait_error, tmp_path):
    engine = _start_sigterm_ignoring_engine(tmp_path)

    class InterruptWaitOnce:
        def __init__(self, process):
            self.process = process
            self.injected = False

        def __getattr__(self, name):
            return getattr(self.process, name)

        def wait(self, *positional, **kwargs):
            if kwargs.get("timeout") == 10 and not self.injected:
                self.injected = True
                if wait_error == "sigint":
                    os.kill(os.getpid(), signal.SIGINT)
                    return self.process.wait(*positional, **kwargs)
                raise OSError("injected engine wait failure")
            return self.process.wait(*positional, **kwargs)

    try:
        with pytest.raises(KeyboardInterrupt if wait_error == "sigint" else OSError):
            THREE_CLIENTS_MODULE.stop_engine(
                Path("/usr/bin/false"),
                tmp_path,
                os.environ.copy(),
                InterruptWaitOnce(engine),
            )
        assert engine.poll() is not None, "engine survived interrupted wait cleanup"
    finally:
        _cleanup_test_engine(engine)


@pytest.mark.parametrize(
    "boundary",
    [
        "grace-wait",
        "grace-sleep",
        "kill-sleep",
        "reap-wait",
        "inventory-iterdir",
        "inventory-stat",
    ],
    ids=[
        "grace-wait",
        "grace-sleep",
        "kill-sleep",
        "reap-wait",
        "inventory-iterdir",
        "inventory-stat",
    ],
)
def test_stop_engine_finishes_teardown_when_cleanup_is_interrupted(
    boundary, monkeypatch, tmp_path
):
    engine = _start_sigterm_ignoring_engine(tmp_path)
    original_terminate = THREE_CLIENTS_MODULE.terminate_group
    original_killpg = os.killpg
    original_sleep = time.sleep
    original_path = THREE_CLIENTS_MODULE.Path
    state = {"injected": False, "kill_requested": False, "deferred": False}

    def inject_sigint():
        state["injected"] = True
        os.kill(os.getpid(), signal.SIGINT)
        # Only reached when the teardown mask held the signal pending.
        state["deferred"] = signal.SIGINT in signal.sigpending()

    class InterruptAtCleanupBoundary:
        def __init__(self, process):
            self.process = process

        def __getattr__(self, name):
            return getattr(self.process, name)

        def wait(self, *positional, **kwargs):
            timeout = kwargs.get("timeout")
            if timeout == 10:
                raise subprocess.TimeoutExpired("initial engine wait", 10)
            if boundary == "grace-wait" and timeout == 5:
                inject_sigint()
            if (
                boundary
                in {
                    "grace-sleep",
                    "kill-sleep",
                    "inventory-iterdir",
                    "inventory-stat",
                }
                and timeout == 0.1
            ):
                raise subprocess.TimeoutExpired("short grace wait", 0.1)
            if boundary == "reap-wait" and state["kill_requested"]:
                inject_sigint()
            return self.process.wait(*positional, **kwargs)

    def short_grace(process, *positional, **kwargs):
        grace_seconds = 5 if boundary == "grace-wait" else 0.1
        return original_terminate(process, grace_seconds=grace_seconds)

    def track_kill(pgid, sig):
        if sig == signal.SIGKILL:
            state["kill_requested"] = True
            if boundary == "kill-sleep":
                return
        original_killpg(pgid, sig)

    def interrupt_sleep(seconds):
        if boundary == "grace-sleep" and not state["injected"]:
            inject_sigint()
        if (
            boundary == "kill-sleep"
            and state["kill_requested"]
            and not state["injected"]
        ):
            try:
                inject_sigint()
            finally:
                original_killpg(engine.pid, signal.SIGKILL)
        original_sleep(seconds)

    class OwnStat:
        def read_text(self, *args, **kwargs):
            if boundary == "inventory-stat" and not state["injected"]:
                inject_sigint()
            return original_path(f"/proc/{engine.pid}/stat").read_text(*args, **kwargs)

    class OwnEntry:
        name = str(engine.pid)

        def __truediv__(self, child):
            assert child == "stat"
            return OwnStat()

    class ProcDirectory:
        def is_dir(self):
            return True

        def iterdir(self):
            if boundary == "inventory-iterdir" and not state["injected"]:
                inject_sigint()
                return original_path("/proc").iterdir()
            return [OwnEntry()]

    def path_with_owned_proc(path):
        if str(path) == "/proc" and boundary.startswith("inventory-"):
            return ProcDirectory()
        return original_path(path)

    monkeypatch.setattr(THREE_CLIENTS_MODULE, "terminate_group", short_grace)
    monkeypatch.setattr(THREE_CLIENTS_MODULE.os, "killpg", track_kill)
    monkeypatch.setattr(THREE_CLIENTS_MODULE.time, "sleep", interrupt_sleep)
    monkeypatch.setattr(THREE_CLIENTS_MODULE, "Path", path_with_owned_proc)
    try:
        with pytest.raises(KeyboardInterrupt):
            THREE_CLIENTS_MODULE.stop_engine(
                Path("/usr/bin/false"),
                tmp_path,
                os.environ.copy(),
                InterruptAtCleanupBoundary(engine),
            )
        assert state["injected"] is True
        assert state["deferred"] is True, f"SIGINT was not deferred at {boundary}"
        assert engine.poll() is not None, f"engine survived {boundary} interrupt"
        assert _processes_with_paths_under(tmp_path) == []
    finally:
        monkeypatch.undo()
        _cleanup_test_engine(engine)


def _sigint_blocked():
    return signal.SIGINT in signal.pthread_sigmask(signal.SIG_BLOCK, ())


def test_block_sigint_restores_mask_when_blocking_surfaces_pending_interrupt(
    monkeypatch, tmp_path
):
    engine = _start_sigterm_ignoring_engine(tmp_path)
    real_sigmask = signal.pthread_sigmask
    blocked_before = _sigint_blocked()
    state = {"surfaced": False}

    def apply_then_surface(how, mask):
        previous = real_sigmask(how, mask)
        if (
            how == signal.SIG_BLOCK
            and signal.SIGINT in set(mask)
            and not state["surfaced"]
        ):
            state["surfaced"] = True
            # CPython applies the new mask before it runs pending handlers.
            raise KeyboardInterrupt("pending SIGINT surfaced by pthread_sigmask")
        return previous

    monkeypatch.setattr(
        THREE_CLIENTS_MODULE.signal, "pthread_sigmask", apply_then_surface
    )
    try:
        with pytest.raises(KeyboardInterrupt):
            THREE_CLIENTS_MODULE.terminate_group(engine, grace_seconds=0.1)
        monkeypatch.undo()
        assert state["surfaced"] is True
        assert engine.poll() is not None, "engine survived surfaced interrupt"
        assert _sigint_blocked() is blocked_before, "SIGINT mask was not restored"
    finally:
        monkeypatch.undo()
        if _sigint_blocked() and not blocked_before:
            real_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT})
        _cleanup_test_engine(engine)


def test_run_capture_defers_interrupt_at_engine_stop_entry(monkeypatch, a5_tmp_path):
    args, _plan, _candidate, _client = _runner_fixture(a5_tmp_path)
    original_stop = THREE_CLIENTS_MODULE.stop_engine
    stopped = []
    blocked_before = _sigint_blocked()

    def interrupt_at_entry(candidate, repo, environment, process):
        if not stopped:
            os.kill(os.getpid(), signal.SIGINT)
        stopped.append(process)
        return original_stop(candidate, repo, environment, process)

    monkeypatch.setattr(THREE_CLIENTS_MODULE, "stop_engine", interrupt_at_entry)
    try:
        with pytest.raises(KeyboardInterrupt):
            THREE_CLIENTS_MODULE.run_capture(args)
        assert len(stopped) == 1, "engine stop was skipped by the entry interrupt"
        assert stopped[0] is not None and stopped[0].poll() is not None
        assert _sigint_blocked() is blocked_before, "SIGINT mask was not restored"
        assert _processes_with_paths_under(a5_tmp_path) == []
    finally:
        _kill_fixture_processes(a5_tmp_path)


def test_stop_engine_preserves_stop_error_after_group_cleanup(monkeypatch, tmp_path):
    engine = _start_sigterm_ignoring_engine(tmp_path)
    original_wait = engine.wait
    first_wait = True

    def fail_stop(*args, **kwargs):
        raise FileNotFoundError("synthetic stop command failure")

    def timeout_initial_wait(*args, **kwargs):
        nonlocal first_wait
        if first_wait and kwargs.get("timeout") == 10:
            first_wait = False
            raise subprocess.TimeoutExpired("engine", 10)
        return original_wait(*args, **kwargs)

    monkeypatch.setattr(THREE_CLIENTS_MODULE.subprocess, "run", fail_stop)
    monkeypatch.setattr(engine, "wait", timeout_initial_wait)
    try:
        with pytest.raises(FileNotFoundError, match="synthetic stop command failure"):
            THREE_CLIENTS_MODULE.stop_engine(
                Path("/missing/inert-candidate"),
                tmp_path,
                os.environ.copy(),
                engine,
            )
        assert engine.poll() is not None, "engine survived stop-command failure cleanup"
    finally:
        _cleanup_test_engine(engine)


def test_roster_names_are_exact():
    assert set(MODULE.CLIENTS) == {"claude_code", "gemini_cli", "codex_cli"}


def test_digest_is_lowercase_sha256():
    assert (
        MODULE.digest(b"abc")
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_copy_drift_cannot_publish_a_manifest(monkeypatch):
    original = MODULE.shutil.copyfile

    def tamper(source, destination, **kwargs):
        result = original(source, destination, **kwargs)
        Path(destination).write_bytes(b"changed after sidecar validation")
        return result

    monkeypatch.setattr(MODULE.shutil, "copyfile", tamper)
    with pytest.raises(MODULE.CapturePackageError, match="sidecar.*changed"):
        MODULE.self_test()


def test_candidate_drift_during_copy_cannot_publish(monkeypatch):
    original = MODULE.shutil.copyfile

    def tamper(source, destination, **kwargs):
        result = original(source, destination, **kwargs)
        (Path(source).parent.parent / "candidate.bin").write_bytes(b"changed candidate")
        return result

    monkeypatch.setattr(MODULE.shutil, "copyfile", tamper)
    with pytest.raises(MODULE.CapturePackageError, match="candidate"):
        MODULE.self_test()
