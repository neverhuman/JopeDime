"""A5 capture-client tests."""

from test_capture_clients_support import *

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
def test_run_capture_defers_interrupt_at_outer_cleanup_entry(monkeypatch, a5_tmp_path):
    args, _plan, _candidate, _client = _runner_fixture(a5_tmp_path)
    original_stop = THREE_CLIENTS_MODULE.stop_engine
    original_handler = signal.getsignal(signal.SIGINT)
    stopped = []
    injected = {"value": False}

    source, first_line = inspect.getsourcelines(THREE_CLIENTS_MODULE.run_capture)
    function = ast.parse("".join(source)).body[0]
    cleanup_try = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Try)
        and node.finalbody
        and any(
            isinstance(call.func, ast.Name) and call.func.id == "stop_engine"
            for statement in node.finalbody
            for call in ast.walk(statement)
            if isinstance(call, ast.Call)
        )
    )
    cleanup_entry_line = first_line + cleanup_try.finalbody[0].lineno - 1

    def track_stop(candidate, repo, environment, process):
        stopped.append(process)
        return original_stop(candidate, repo, environment, process)

    def interrupt_at_finally_entry(frame, event, _arg):
        if (
            not injected["value"]
            and frame.f_code is THREE_CLIENTS_MODULE.run_capture.__code__
            and event == "line"
            and frame.f_lineno == cleanup_entry_line
        ):
            injected["value"] = True
            sys.settrace(None)
            os.kill(os.getpid(), signal.SIGINT)
            return None
        return interrupt_at_finally_entry

    monkeypatch.setattr(THREE_CLIENTS_MODULE, "stop_engine", track_stop)
    try:
        sys.settrace(interrupt_at_finally_entry)
        with pytest.raises(KeyboardInterrupt):
            THREE_CLIENTS_MODULE.run_capture(args)
        assert injected["value"] is True, "SIGINT missed the outer cleanup entry"
        assert len(stopped) == 1, "engine stop was skipped by outer-finally SIGINT"
        assert stopped[0] is not None and stopped[0].poll() is not None
        assert _processes_with_paths_under(a5_tmp_path) == []
        assert signal.getsignal(signal.SIGINT) is original_handler
    finally:
        sys.settrace(None)
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
def test_run_capture_defers_traced_interrupt_at_teardown_entry(a5_tmp_path):
    args, _plan, _candidate, _client = _runner_fixture(a5_tmp_path)
    first, last = _engine_cleanup_finally_lines()
    code = THREE_CLIENTS_MODULE.run_capture.__code__
    state = {"sent": False}

    def interrupt_at_teardown_line(frame, event, arg):
        if event == "line" and first <= frame.f_lineno <= last and not state["sent"]:
            state["sent"] = True
            os.kill(os.getpid(), signal.SIGINT)
        return interrupt_at_teardown_line

    def trace_run_capture(frame, event, arg):
        return interrupt_at_teardown_line if frame.f_code is code else None

    try:
        sys.settrace(trace_run_capture)
        try:
            with pytest.raises(KeyboardInterrupt):
                THREE_CLIENTS_MODULE.run_capture(args)
        finally:
            sys.settrace(None)
        assert state["sent"] is True
        assert list(Path(args.engine_home).glob("*/stopped")), "engine stop skipped"
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
        assert _processes_with_paths_under(a5_tmp_path) == []
    finally:
        _kill_fixture_processes(a5_tmp_path)
def test_run_capture_honors_interrupt_deferred_during_engine_launch(
    monkeypatch, a5_tmp_path
):
    args, _plan, _candidate, client = _runner_fixture(a5_tmp_path)
    original_popen = subprocess.Popen
    launched = {"engines": [], "clients": 0}

    def interrupt_after_engine_fork(command, *positional, **kwargs):
        process = original_popen(command, *positional, **kwargs)
        if "--idle-exit" in command and not launched["engines"]:
            launched["engines"].append(process)
            # Arrives after the fork but before run_capture owns the handle.
            os.kill(os.getpid(), signal.SIGINT)
        elif command[0] == str(client):
            launched["clients"] += 1
        return process

    monkeypatch.setattr(
        THREE_CLIENTS_MODULE.subprocess, "Popen", interrupt_after_engine_fork
    )
    try:
        with pytest.raises(KeyboardInterrupt):
            THREE_CLIENTS_MODULE.run_capture(args)
        assert launched["engines"] and launched["engines"][0].poll() is not None
        assert launched["clients"] == 0, "deferred interrupt was not honored promptly"
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
        assert _processes_with_paths_under(a5_tmp_path) == []
    finally:
        _kill_fixture_processes(a5_tmp_path)
def test_normal_cli_exit_preserves_real_claude_recorder_flush(a5_tmp_path):
    args, _plan, _candidate, _client = _runner_fixture(
        a5_tmp_path, transport_key="claude_code"
    )
    result = THREE_CLIENTS_MODULE.run_capture(args)
    case = args.out_dir / "clients/claude_code"
    assert (case / "recorder-flushed").is_file(), (
        "normal CLI exit killed the pending recorder flush"
    )
    assert result["status"] == "CAPTURES_COMPLETE_PENDING_SEMANTIC_REVIEW"
    receipt_path = case / "transport/capture.json"
    assert receipt_path.is_file() and not receipt_path.is_symlink()
    receipt = json.loads(receipt_path.read_text())
    assert receipt["complete"] is True
    assert receipt["relay_exit_code"] == receipt["server_exit_code"] == 0
    assert result["clients"]["claude_code"]["client_exit"] == 0
    assert _processes_with_paths_under(a5_tmp_path) == []
def test_process_group_leader_recorder_is_reisolated(a5_tmp_path):
    args, _plan, _candidate, _client = _runner_fixture(
        a5_tmp_path,
        transport_key="claude_code",
        recorder_group_key="claude_code",
    )
    result = THREE_CLIENTS_MODULE.run_capture(args)

    assert result["status"] == "CAPTURES_COMPLETE_PENDING_SEMANTIC_REVIEW"
    assert result["clients"]["claude_code"]["client_exit"] == 0
    transport = args.out_dir / "clients/claude_code/transport"
    receipt = json.loads((transport / "capture.json").read_text())
    assert receipt["complete"] is True
    assert receipt["server_exit_code"] == receipt["relay_exit_code"] == 0
    assert _processes_with_paths_under(a5_tmp_path) == []
def test_timeout_cleans_isolated_recorder_and_server(monkeypatch, a5_tmp_path):
    args, plan, _candidate, _client = _runner_fixture(
        a5_tmp_path, transport_key="claude_code", sleep_key="claude_code"
    )
    plan["clients"]["claude_code"]["timeout_seconds"] = 1
    args.client_plan.write_text(json.dumps(plan))
    original = THREE_CLIENTS_MODULE.RecorderSession.groups
    isolated = []

    def observe_groups(recorder):
        groups = original(recorder)
        if recorder.identity is not None:
            current = THREE_CLIENTS_MODULE._process_info(recorder.identity["pid"])
            if current is not None:
                isolated.append((current, recorder.client_pid))
        return groups

    monkeypatch.setattr(THREE_CLIENTS_MODULE.RecorderSession, "groups", observe_groups)
    result = THREE_CLIENTS_MODULE.run_capture(args)
    assert result["status"] == "INCOMPLETE"
    assert result["clients"]["claude_code"]["timed_out"] is True
    assert any(
        p["pid"] == p["session"] == p["group"] and p["pid"] != client
        for p, client in isolated
    )
    assert _processes_with_paths_under(a5_tmp_path) == []
def test_drain_timeout_preserves_client_exit_and_refuses_success(
    monkeypatch, a5_tmp_path
):
    args, _plan, _candidate, _client = _runner_fixture(
        a5_tmp_path, transport_key="claude_code", flush_seconds=1
    )
    monkeypatch.setattr(THREE_CLIENTS_MODULE, "RECORDER_DRAIN_SECONDS", 0.1)
    result = THREE_CLIENTS_MODULE.run_capture(args)
    row = result["clients"]["claude_code"]
    assert result["status"] == "INCOMPLETE"
    assert row["client_exit"] == 0
    assert "drain deadline" in row["capture_error"]
    assert row["timed_out"] is False
    assert _processes_with_paths_under(a5_tmp_path) == []
def test_foreign_recorder_and_reused_pid_are_never_signaled(monkeypatch, a5_tmp_path):
    args, _plan, _candidate, _client = _runner_fixture(a5_tmp_path)
    producer = (
        args.repo / "deploy/qualification/agents/transcripts/run-prep/capture_stdio.py"
    )
    directory = a5_tmp_path / "foreign-lease"
    directory.mkdir()
    foreign = subprocess.Popen(
        [
            sys.executable,
            "-B",
            "-c",
            "import time;time.sleep(30)",
            str(producer),
            "--out-dir",
            str(directory),
        ],
        start_new_session=True,
    )
    info = THREE_CLIENTS_MODULE._process_info(foreign.pid)
    lease = dict(
        schema="jopedime.mcp-recorder-session/v2",
        pid=foreign.pid,
        owner_session=12345,
        start_ticks=info["start_ticks"],
        producer_sha256=hashlib.sha256(producer.read_bytes()).hexdigest(),
        argv=[str(args.candidate), "mcp"],
    )
    (directory / "recorder.json").write_text(json.dumps(lease))
    recorder = THREE_CLIENTS_MODULE.RecorderSession(
        directory, 12345, producer, args.candidate
    )
    signals = []
    monkeypatch.setattr(
        THREE_CLIENTS_MODULE.os, "killpg", lambda *args: signals.append(args)
    )
    try:
        with pytest.raises(
            THREE_CLIENTS_MODULE.CaptureError, match="outside the owned capture"
        ):
            recorder.stop()
        assert signals == [] and foreign.poll() is None
        recorder.identity = dict(lease, start_ticks=lease["start_ticks"] - 1)
        with pytest.raises(THREE_CLIENTS_MODULE.CaptureError, match="PID was reused"):
            recorder.stop()
        assert signals == [] and foreign.poll() is None
    finally:
        foreign.kill()
        foreign.wait(timeout=3)
@pytest.mark.parametrize(
    "pin_plan", [False, True], ids=["missing-pin-refused", "matching-pin-admitted"]
)
def test_capture_tool_source_is_separate_from_frozen_candidate(pin_plan, a5_tmp_path):
    args, plan, _candidate, _client = _runner_fixture(a5_tmp_path)
    frozen = args.source_sha
    (args.repo / "tool-version.txt").write_text("new reviewed tools")
    subprocess.run(["git", "-C", str(args.repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(args.repo), "commit", "-qm", "tool metadata source"],
        check=True,
    )
    tools = subprocess.check_output(
        ["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True
    ).strip()
    args.capture_source_sha = tools
    args.preflight_only = True
    if pin_plan:
        plan["capture_source_sha"] = tools
        args.client_plan.write_text(json.dumps(plan))
        result = THREE_CLIENTS_MODULE.run_capture(args)
        assert result["status"] == "PREFLIGHT_READY_NO_CLIENTS_LAUNCHED"
        assert result["source_sha"] == frozen
        assert result["capture_source_sha"] == tools != frozen
    else:
        with pytest.raises(
            THREE_CLIENTS_MODULE.CaptureError, match="capture-tool source SHA mismatch"
        ):
            THREE_CLIENTS_MODULE.run_capture(args)
    assert not args.out_dir.exists() and not args.engine_home.exists()
def test_recorder_can_finish_during_process_identity_observation(
    monkeypatch, a5_tmp_path
):
    args, _plan, _candidate, _client = _runner_fixture(
        a5_tmp_path, transport_key="claude_code"
    )
    original = Path.read_bytes
    observed = []

    def finish_before_command_read(path):
        if not observed and path.name == "cmdline" and path.parent.name.isdecimal():
            pid = int(path.parent.name)
            observed.append(pid)
            deadline = time.monotonic() + 3
            while _pid_is_live(pid) and time.monotonic() < deadline:
                time.sleep(0.01)
            assert not _pid_is_live(pid), "recorder did not finish its delayed EOF"
            return b""  # Linux exposes empty cmdline for the now-zombie recorder.
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", finish_before_command_read)
    result = THREE_CLIENTS_MODULE.run_capture(args)
    assert observed, "ownership transition was not exercised"
    assert result["status"] == "CAPTURES_COMPLETE_PENDING_SEMANTIC_REVIEW"
    assert (args.out_dir / "clients/claude_code/transport/capture.json").is_file()
    assert _processes_with_paths_under(a5_tmp_path) == []
