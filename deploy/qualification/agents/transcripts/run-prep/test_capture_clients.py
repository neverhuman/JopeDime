"""A5 capture-client tests."""

from test_capture_clients_support import *

def test_reviewed_transport_bounds_admit_v14_and_refuse_above_limit():
    actual_v14 = {
        "seconds": 390.0,
        "stream_bytes": 16_777_216,
        "frame_bytes": 1_048_576,
    }
    assert THREE_CLIENTS_MODULE.MAX_TRANSPORT_SECONDS == actual_v14["seconds"]
    assert THREE_CLIENTS_MODULE.transport_limits_within_reviewed_bounds(actual_v14)

    above_seconds = {**actual_v14, "seconds": 390.0001}
    assert not THREE_CLIENTS_MODULE.transport_limits_within_reviewed_bounds(
        above_seconds
    )
@pytest.mark.parametrize(
    ("seconds", "expected_status"),
    [
        (390, "CAPTURES_COMPLETE_PENDING_SEMANTIC_REVIEW"),
        (391, "INCOMPLETE"),
    ],
)
def test_real_recorder_transport_limit_matches_consumer(
    seconds, expected_status, a5_tmp_path
):
    args, _plan, _candidate, _client = _runner_fixture(
        a5_tmp_path,
        transport_key="claude_code",
        recorder_seconds=seconds,
    )

    result = THREE_CLIENTS_MODULE.run_capture(args)

    assert result["status"] == expected_status
    row = result["clients"]["claude_code"]
    if seconds == 390:
        receipt = json.loads(
            (args.out_dir / "clients/claude_code/transport/capture.json").read_text()
        )
        assert receipt["limits"]["seconds"] == 390
        assert row["capture_error"] is None
    else:
        assert "transport limits exceed the reviewed bounds" in row["capture_error"]
@pytest.mark.parametrize(
    "changes",
    [
        {"stream_bytes": 16_777_217},
        {"frame_bytes": 1_048_575},
        {"seconds": True},
        {"stream_bytes": True},
    ],
)
def test_reviewed_transport_bounds_keep_byte_frame_and_type_guards(changes):
    limits = {
        "seconds": 390.0,
        "stream_bytes": 16_777_216,
        "frame_bytes": 1_048_576,
        **changes,
    }
    assert not THREE_CLIENTS_MODULE.transport_limits_within_reviewed_bounds(limits)
def test_synthetic_package_contract():
    MODULE.self_test()
def test_three_client_capture_plan_self_test():
    THREE_CLIENTS_MODULE.self_test()
def test_partial_client_selection_requires_exclusion_reason():
    with pytest.raises(THREE_CLIENTS_MODULE.CaptureError, match="exclude-reason"):
        THREE_CLIENTS_MODULE.client_selection(["claude_code", "codex_cli"], None)
def test_partial_client_capture_records_excluded_client(a5_tmp_path):
    original_mask = signal.pthread_sigmask(signal.SIG_BLOCK, ())
    original_handler = signal.getsignal(signal.SIGINT)
    args, _plan, _candidate, _client = _runner_fixture(a5_tmp_path)
    args.clients = ["claude_code", "codex_cli"]
    args.exclude_reason = "Gemini CLI MCP configuration unavailable"

    result = THREE_CLIENTS_MODULE.run_capture(args)

    assert signal.pthread_sigmask(signal.SIG_BLOCK, ()) == original_mask
    assert signal.getsignal(signal.SIGINT) is original_handler

    assert result["status"] == "PARTIAL_CAPTURES_PENDING_SEMANTIC_REVIEW"
    assert result["selected_clients"] == ["claude_code", "codex_cli"]
    assert result["excluded_clients"] == {
        "gemini_cli": "Gemini CLI MCP configuration unavailable"
    }
    assert result["clients"]["gemini_cli"]["status"] == "EXCLUDED"
    assert not (args.out_dir / "clients/gemini_cli").exists()
    excluded_execution = json.loads(
        (args.out_dir / "sidecars/gemini-cli.execution.json").read_text(
            encoding="utf-8"
        )
    )
    assert excluded_execution["status"] == "EXCLUDED"
    assert excluded_execution["exclusion_reason"] == args.exclude_reason
def test_normal_exit_drains_owned_recorder_descendant(monkeypatch, a5_tmp_path):
    red = a5_tmp_path / "immediate-cleanup"
    red.mkdir()
    red_args, _plan, _candidate, _client = _runner_fixture(red, drain_key="claude_code")
    with monkeypatch.context() as patch:
        patch.setattr(THREE_CLIENTS_MODULE, "drain_group", lambda *_args: None)
        red_result = THREE_CLIENTS_MODULE.run_capture(red_args)
    assert red_result["status"] == "INCOMPLETE"
    assert not (
        red_args.out_dir / "clients/claude_code/transport/capture.json"
    ).exists()

    green = a5_tmp_path / "bounded-natural-drain"
    green.mkdir()
    green_args, _plan, _candidate, _client = _runner_fixture(
        green, drain_key="claude_code"
    )
    real_drain = THREE_CLIENTS_MODULE.drain_group

    def release_then_drain(process, deadline, check_interrupted, receipt_path=None):
        (green_args.out_dir / "clients/claude_code/drain-child.release").write_text(
            "go", encoding="utf-8"
        )
        return real_drain(process, deadline, check_interrupted, receipt_path)

    with monkeypatch.context() as patch:
        patch.setattr(THREE_CLIENTS_MODULE, "drain_group", release_then_drain)
        green_result = THREE_CLIENTS_MODULE.run_capture(green_args)
    assert green_result["status"] == "CAPTURES_COMPLETE_PENDING_SEMANTIC_REVIEW"
    assert green_result["clients"]["claude_code"]["client_exit"] == 0
def test_normal_exit_waits_for_isolated_recorder_receipt(a5_tmp_path):
    args, _plan, _candidate, _client = _runner_fixture(
        a5_tmp_path, async_recorder_key="claude_code"
    )
    result = THREE_CLIENTS_MODULE.run_capture(args)
    receipt = args.out_dir / "clients/claude_code/transport/capture.json"
    deadline = time.monotonic() + 3
    while not receipt.is_file() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert receipt.is_file(), "isolated MCP recorder did not finish its receipt"
    assert result["status"] == "CAPTURES_COMPLETE_PENDING_SEMANTIC_REVIEW"
    assert result["clients"]["claude_code"]["client_exit"] == 0
    assert result["clients"]["claude_code"]["capture_error"] is None
def test_deadline_poll_does_not_extend_recorder_drain(monkeypatch, a5_tmp_path):
    args, plan, _candidate, client = _runner_fixture(
        a5_tmp_path, drain_key="claude_code"
    )
    plan["clients"]["claude_code"]["timeout_seconds"] = 1
    args.client_plan.write_text(json.dumps(plan), encoding="utf-8")

    class ShiftedTime:
        client_finished = False
        phase = 0

        @staticmethod
        def monotonic():
            now = time.monotonic()
            if not ShiftedTime.client_finished:
                return now
            if ShiftedTime.phase == 0:
                ShiftedTime.phase = 1
                return now - 2
            ShiftedTime.phase = 2
            return now + 2

        @staticmethod
        def sleep(seconds):
            return time.sleep(seconds)

    original_popen = THREE_CLIENTS_MODULE.subprocess.Popen

    def exit_client_before_deadline_poll(command, *positional, **kwargs):
        process = original_popen(command, *positional, **kwargs)
        if command[0] == str(client):
            assert process.wait(timeout=5) == 0
            ShiftedTime.client_finished = True
        return process

    drain_calls = []
    real_drain = THREE_CLIENTS_MODULE.drain_group

    def drain_must_not_run(process, deadline, check_interrupted, *extra):
        if process.args[-1] == "claude_code":
            drain_calls.append(True)
            raise AssertionError(
                "no recorder drain budget remains at the client deadline"
            )
        return real_drain(process, deadline, check_interrupted, *extra)

    with monkeypatch.context() as patch:
        patch.setattr(THREE_CLIENTS_MODULE, "time", ShiftedTime)
        patch.setattr(
            THREE_CLIENTS_MODULE.subprocess,
            "Popen",
            exit_client_before_deadline_poll,
        )
        patch.setattr(THREE_CLIENTS_MODULE, "drain_group", drain_must_not_run)
        result = THREE_CLIENTS_MODULE.run_capture(args)

    assert result["status"] == "INCOMPLETE"
    assert result["clients"]["claude_code"]["client_exit"] == 0
    assert not result["clients"]["claude_code"]["timed_out"]
    assert drain_calls == []
    assert not (args.out_dir / "clients/claude_code/transport/capture.json").exists()
def test_normal_exit_drain_deadline_is_not_replaced_with_grace(
    monkeypatch, a5_tmp_path
):
    args, plan, _candidate, client = _runner_fixture(
        a5_tmp_path, drain_key="claude_code"
    )
    plan["clients"]["claude_code"]["timeout_seconds"] = 1
    args.client_plan.write_text(json.dumps(plan), encoding="utf-8")

    class ShiftedTime:
        offset = 0.0

        @staticmethod
        def monotonic():
            return time.monotonic() + ShiftedTime.offset

        @staticmethod
        def sleep(seconds):
            return time.sleep(seconds)

    original_popen = THREE_CLIENTS_MODULE.subprocess.Popen

    def client_exits_after_deadline(command, *positional, **kwargs):
        process = original_popen(command, *positional, **kwargs)
        if command[0] == str(client):
            original_wait = process.wait

            def wait_then_cross_deadline(timeout=None):
                exit_code = original_wait(timeout=5)
                assert exit_code == 0
                # Model a normal exit observed only after the original deadline.
                ShiftedTime.offset = 2.0
                return exit_code

            process.wait = wait_then_cross_deadline
        return process

    drain_deadlines = []

    def record_drain_deadline(process, deadline, *_args, **_kwargs):
        if process.args[-1] == "claude_code":
            drain_deadlines.append(deadline)

    with monkeypatch.context() as patch:
        patch.setattr(THREE_CLIENTS_MODULE, "time", ShiftedTime)
        patch.setattr(
            THREE_CLIENTS_MODULE.subprocess,
            "Popen",
            client_exits_after_deadline,
        )
        patch.setattr(THREE_CLIENTS_MODULE, "drain_group", record_drain_deadline)
        result = THREE_CLIENTS_MODULE.run_capture(args)

    assert result["clients"]["claude_code"]["client_exit"] == 0
    assert not result["clients"]["claude_code"]["timed_out"]
    assert all(deadline <= ShiftedTime.monotonic() for deadline in drain_deadlines), (
        "recorder drain deadline must not extend beyond the client_deadline"
    )
def test_slow_eof_flush_fits_post_exit_recorder_budget(a5_tmp_path):
    args, _plan, _candidate, _client = _runner_fixture(
        a5_tmp_path,
        transport_key="claude_code",
        flush_seconds=6,
        recorder_seconds=30,
    )
    result = THREE_CLIENTS_MODULE.run_capture(args)
    row = result["clients"]["claude_code"]
    receipt = args.out_dir / "clients/claude_code/transport/capture.json"
    assert result["status"] == "CAPTURES_COMPLETE_PENDING_SEMANTIC_REVIEW"
    assert row["client_exit"] == 0
    assert row["capture_error"] is None
    assert json.loads(receipt.read_text())["complete"] is True
def test_normal_exit_drain_is_bounded_before_forced_cleanup(monkeypatch, a5_tmp_path):
    args, _plan, _candidate, _client = _runner_fixture(
        a5_tmp_path, drain_key="claude_code"
    )
    original_terminate = THREE_CLIENTS_MODULE.terminate_group

    def short_forced_cleanup(process, grace_seconds=5):
        return original_terminate(process, grace_seconds=0.1)

    monkeypatch.setattr(THREE_CLIENTS_MODULE, "NORMAL_EXIT_DRAIN_MAX_SECONDS", 0.05)
    monkeypatch.setattr(
        THREE_CLIENTS_MODULE, "NORMAL_EXIT_RECEIPT_DRAIN_MAX_SECONDS", 0.1
    )
    monkeypatch.setattr(THREE_CLIENTS_MODULE, "terminate_group", short_forced_cleanup)
    original_drain = THREE_CLIENTS_MODULE.drain_group
    drain_elapsed = []

    def measure_drain(*positional):
        drain_started = time.monotonic()
        try:
            return original_drain(*positional)
        finally:
            drain_elapsed.append(time.monotonic() - drain_started)

    monkeypatch.setattr(THREE_CLIENTS_MODULE, "drain_group", measure_drain)
    started = time.monotonic()
    result = THREE_CLIENTS_MODULE.run_capture(args)
    elapsed = time.monotonic() - started
    assert elapsed < 8
    assert drain_elapsed and drain_elapsed[0] <= 0.3
    assert result["status"] == "INCOMPLETE"
    assert result["clients"]["claude_code"]["client_exit"] == 0
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
