"""A5 capture-client tests."""

from test_capture_clients_support import *

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
    args, _plan, _candidate, _client = _runner_fixture(
        a5_tmp_path, descendant_key="claude_code"
    )
    case = args.out_dir / "clients/claude_code"
    original = THREE_CLIENTS_MODULE.observe_client_exit
    interrupted = []

    def interrupt_after_descendant(process):
        deadline = time.monotonic() + 5
        marker = case / "descendant.pid"
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists(), "fixture descendant did not start"
        if not interrupted:
            interrupted.append(process)
            raise KeyboardInterrupt
        return original(process)

    monkeypatch.setattr(
        THREE_CLIENTS_MODULE, "observe_client_exit", interrupt_after_descendant
    )
    try:
        with pytest.raises(KeyboardInterrupt):
            THREE_CLIENTS_MODULE.run_capture(args)
        assert not _pid_is_live(int((case / "descendant.pid").read_text()))
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
    args, _plan, _candidate, _client = _runner_fixture(
        a5_tmp_path, sleep_key="claude_code", descendant_key="claude_code"
    )
    case = args.out_dir / "clients/claude_code"
    original = THREE_CLIENTS_MODULE.observe_client_exit
    observed = []

    def interrupt_first_client(process):
        deadline = time.monotonic() + 5
        marker = case / "descendant.pid"
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists(), "fixture descendant did not start"
        if not observed:
            observed.append(process)
            raise KeyboardInterrupt
        return original(process)

    monkeypatch.setattr(
        THREE_CLIENTS_MODULE, "observe_client_exit", interrupt_first_client
    )
    try:
        with pytest.raises(KeyboardInterrupt):
            THREE_CLIENTS_MODULE.run_capture(args)
        assert observed and observed[0].poll() is not None
        assert (args.engine_home / "claude_code/stopped").is_file()
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
