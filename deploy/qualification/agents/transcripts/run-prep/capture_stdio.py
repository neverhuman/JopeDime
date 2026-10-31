#!/usr/bin/env python3
"""Bounded POSIX MCP stdio recording; never synthesize replies or qualify A5."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import subprocess
import sys
import time

from verify_capture import CaptureError, decode_json, require


_DEFAULT_ENV = frozenset(
    {
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "TMPDIR",
        "TMP",
        "TEMP",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_STATE_HOME",
        "XDG_CACHE_HOME",
        "JOPEDIME_HOME",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "CUDA_VISIBLE_DEVICES",
    }
)


def child_environment(extra_names):
    """Select names explicitly; never return credentials through prefix matching."""
    requested = set()
    for name in extra_names:
        require(
            isinstance(name, str)
            and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is not None,
            "--env-allow requires a variable name, not an assignment",
        )
        require(name in os.environ, f"requested environment variable is absent: {name}")
        requested.add(name)
    selected = {
        name: os.environ[name]
        for name in _DEFAULT_ENV | requested
        if name in os.environ
    }
    selected.setdefault("PATH", os.defpath)
    return selected


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record(command, directory, seconds, max_bytes, env_allow=()):
    require(
        hasattr(os, "WNOWAIT")
        and hasattr(os, "waitid")
        and hasattr(selectors, "PollSelector"),
        "capture requires POSIX poll/waitid/WNOWAIT",
    )
    require(
        command and math.isfinite(seconds) and seconds > 0 and max_bytes > 0,
        "a server command and positive finite bounds are required",
    )
    environment = child_environment(env_allow)
    directory.mkdir(mode=0o700)  # Exclusive: never reuse or delete capture evidence.
    paths = {
        name: directory / name
        for name in ("wire.jsonl", "client.raw", "server.raw", "server.stderr")
    }
    streams = {
        name: os.fdopen(
            os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb"
        )
        for name, path in paths.items()
    }
    started = datetime.now(timezone.utc).isoformat()
    deadline = time.monotonic() + seconds
    child = None
    server_exit = None
    error = None
    retained = 0
    frames = {"client_to_server": 0, "server_to_client": 0}
    fragments = {name: bytearray() for name in frames}
    outgoing = {"to_server": bytearray(), "to_client": bytearray()}
    interrupted = False
    selector = selectors.PollSelector()
    blocking = {fd: os.get_blocking(fd) for fd in (0, 1)}
    old_handlers = {}
    executable = shutil.which(command[0], path=environment["PATH"])
    executable_pin = None
    source_paths = {
        "producer_sha256": Path(__file__),
        "verifier_sha256": Path(__file__).with_name("verify_capture.py"),
    }
    source_pins = {}

    def interrupt(_number, _frame):
        nonlocal interrupted
        interrupted = True

    def retain(name, data):
        nonlocal retained
        require(retained + len(data) <= max_bytes, "stream byte limit exceeded")
        streams[name].write(data)
        retained += len(data)

    def observed(direction, chunk, eof):
        buffer = fragments[direction]
        buffer.extend(chunk)
        while b"\n" in buffer:
            end = buffer.index(b"\n") + 1
            require(end <= 1024 * 1024, "JSON-RPC frame exceeds 1 MiB")
            raw = bytes(buffer[:end])
            del buffer[:end]
            message = decode_json(raw)
            require(
                isinstance(message, dict) and message.get("jsonrpc") == "2.0",
                "stream contains a non-JSON-RPC frame",
            )
            row = dict(
                at=datetime.now(timezone.utc).isoformat(),
                direction=direction,
                message=message,
            )
            retain(
                "wire.jsonl",
                (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"),
            )
            frames[direction] += 1
            outgoing[
                "to_server" if direction == "client_to_server" else "to_client"
            ].extend(raw)
        require(len(buffer) <= 1024 * 1024, "JSON-RPC frame exceeds 1 MiB")
        require(not eof or not buffer, "unterminated JSON-RPC frame at EOF")

    try:
        source_pins = {name: digest(path) for name, path in source_paths.items()}
        if executable:
            executable_pin = digest(executable)
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, interrupt)
        child = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            bufsize=0,
            env=environment,
        )
        for fd in (
            0,
            1,
            child.stdin.fileno(),
            child.stdout.fileno(),
            child.stderr.fileno(),
        ):
            os.set_blocking(fd, False)
        readers = {
            0: "client_to_server",
            child.stdout.fileno(): "server_to_client",
            child.stderr.fileno(): "stderr",
        }
        for fd, label in readers.items():
            selector.register(fd, selectors.EVENT_READ, label)
        input_eof = False
        writers = {"to_server": child.stdin.fileno(), "to_client": 1}
        while True:
            require(not interrupted, "capture interrupted")
            require(time.monotonic() < deadline, "capture deadline exceeded")
            # Observe without reaping: reserve the leader PID/PGID through group cleanup.
            exited = (
                os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                is not None
            )
            if (
                exited
                and 0 in readers
                and not any(key.fd == 0 for key, _ in selector.select(0))
            ):
                selector.unregister(0)
                del readers[0]
                observed("client_to_server", b"", True)
                input_eof = True
            for label, fd in list(writers.items()):
                registered = fd in selector.get_map()
                if outgoing[label] and not registered:
                    selector.register(fd, selectors.EVENT_WRITE, label)
                elif not outgoing[label] and registered:
                    selector.unregister(fd)
            if input_eof and not outgoing["to_server"] and "to_server" in writers:
                child.stdin.close()
                del writers["to_server"]
            if exited and not readers and not any(outgoing.values()):
                break
            for key, _events in selector.select(
                min(0.05, max(0, deadline - time.monotonic()))
            ):
                fd, label = key.fd, key.data
                try:
                    if label in outgoing:
                        count = os.write(fd, outgoing[label])
                        del outgoing[label][:count]
                    else:
                        chunk = os.read(fd, 65536)
                        if chunk:
                            name = {
                                "client_to_server": "client.raw",
                                "server_to_client": "server.raw",
                                "stderr": "server.stderr",
                            }[label]
                            retain(name, chunk)
                        else:
                            selector.unregister(fd)
                            del readers[fd]
                            if fd == 0:
                                input_eof = True
                        if label != "stderr":
                            observed(label, chunk, not chunk)
                except BlockingIOError:
                    continue
        if executable_pin:
            require(
                digest(executable) == executable_pin,
                "server executable changed during capture",
            )
        require(
            all(digest(source_paths[name]) == pin for name, pin in source_pins.items()),
            "capture source changed during execution",
        )
    except (CaptureError, OSError) as exc:
        error = str(exc) if isinstance(exc, CaptureError) else type(exc).__name__
    finally:
        if child is not None:
            # Only this fresh session's process group, while its leader PID is reserved.
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(child.pid, sig)
                except ProcessLookupError:
                    pass
                if sig == signal.SIGTERM:
                    time.sleep(0.05)
            try:
                server_exit = child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                error = "owned server did not exit after group cleanup"
            for pipe in (child.stdin, child.stdout, child.stderr):
                pipe.close()
        selector.close()
        for fd, value in blocking.items():
            os.set_blocking(fd, value)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        for stream in streams.values():
            stream.close()

    if interrupted:
        error = "capture interrupted"
    code = 125 if error else (server_exit if server_exit >= 0 else 128 - server_exit)
    receipt = dict(
        schema="jopedime.mcp-stdio-capture/v1",
        started_at=started,
        ended_at=datetime.now(timezone.utc).isoformat(),
        argv=command,
        **source_pins,
        python_version=sys.version.split()[0],
        platform=sys.platform,
        nice=os.getpriority(os.PRIO_PROCESS, 0),
        environment_names=sorted(environment),
        extra_environment_names=sorted(set(env_allow)),
        executable=executable,
        executable_sha256=executable_pin,
        server_pid=child.pid if child else None,
        elapsed_seconds=time.monotonic() - deadline + seconds,
        server_exit_code=server_exit,
        relay_exit_code=code,
        client_exit_code=None,
        complete=error is None,
        error=error,
        a5_status="UNVERIFIED",
        limits=dict(seconds=seconds, stream_bytes=max_bytes, frame_bytes=1024 * 1024),
        frames=frames,
        retained_stream_bytes=retained,
        files={
            name: dict(bytes=path.stat().st_size, sha256=digest(path))
            for name, path in paths.items()
        },
    )
    fd = os.open(
        directory / "capture.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(fd, "w") as out:
        json.dump(receipt, out, indent=2)
        out.write("\n")
    if error:
        print(f"MCP capture refused: {error}", file=sys.stderr)
    return code


def self_test():
    """Offline process/transport controls; no named client or engine runs."""
    import tempfile
    import unittest

    require(sys.platform == "linux", "process-custody self-tests require Linux /proc")
    request = b' {"jsonrpc":"2.0", "id":"one", "method":"tools/call", "params":{"name":"system_info","arguments":{}}} \n'
    reply = ' {"jsonrpc":"2.0","id":"one","result":{"structuredContent":{"note":"\u03bb"}}} \n'.encode()

    class Controls(unittest.TestCase):
        def setUp(self):
            self.temp = tempfile.TemporaryDirectory(prefix="mcp-capture-control-")
            self.addCleanup(self.temp.cleanup)
            self.out = Path(self.temp.name) / "capture"

        def run_capture(self, code, data=b"", *options):
            args = [
                sys.executable,
                "-B",
                __file__,
                "--out-dir",
                str(self.out),
                "--seconds",
                "2",
                *options,
                "--",
                sys.executable,
                "-B",
                "-c",
                code,
            ]
            return subprocess.run(args, input=data, capture_output=True, timeout=5)

        def receipt(self):
            return json.loads((self.out / "capture.json").read_text())

        def test_exact_bytes_stderr_private_files_and_pure_wire(self):
            code = (
                f'import os,sys,time;sys.stdin.buffer.readline();os.write(2,b"diagnostic\\n");'
                f"os.write(1,{reply[:17]!r});time.sleep(0.03);os.write(1,{reply[17:]!r})"
            )
            result = self.run_capture(code, request)
            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stdout, reply)
            self.assertEqual((self.out / "client.raw").read_bytes(), request)
            self.assertEqual((self.out / "server.raw").read_bytes(), reply)
            self.assertEqual((self.out / "server.stderr").read_bytes(), b"diagnostic\n")
            from verify_capture import tool_calls

            self.assertEqual(len(tool_calls((self.out / "wire.jsonl").read_bytes())), 1)
            receipt = self.receipt()
            self.assertTrue(receipt["complete"])
            self.assertIsNone(receipt["client_exit_code"])
            self.assertEqual(receipt["a5_status"], "UNVERIFIED")
            self.assertEqual(self.out.stat().st_mode & 0o777, 0o700)
            for path in self.out.iterdir():
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)

        def test_nonzero_server_exit_preserved(self):
            result = self.run_capture("import sys;sys.exit(7)")
            self.assertEqual(result.returncode, 7)
            self.assertEqual(self.receipt()["server_exit_code"], 7)

        def test_existing_directory_never_overwritten(self):
            self.out.mkdir()
            keep = self.out / "keep"
            keep.write_text("original")
            result = self.run_capture('raise RuntimeError("must not start")')
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(keep.read_text(), "original")
            self.assertEqual(list(self.out.iterdir()), [keep])

        def test_bad_or_unterminated_server_frame_refuses_without_reply(self):
            for raw in [
                b"not-json\n",
                b'{"jsonrpc":"2.0"}',
                b'{"jsonrpc":"2.0","x":NaN}\n',
            ]:
                with self.subTest(raw=raw):
                    self.out = Path(self.temp.name) / str(
                        len(list(Path(self.temp.name).iterdir()))
                    )
                    result = self.run_capture(f"import os;os.write(1,{raw!r})")
                    self.assertEqual(result.returncode, 125)
                    self.assertEqual(result.stdout, b"")
                    self.assertFalse(self.receipt()["complete"])
                    self.assertEqual((self.out / "server.raw").read_bytes(), raw)

        def test_byte_and_wall_bounds(self):
            result = self.run_capture(
                'import os;os.write(2,b"x"*65536)', b"", "--max-bytes", "1024"
            )
            self.assertEqual(result.returncode, 125)
            self.assertLessEqual(self.receipt()["retained_stream_bytes"], 1024)
            self.out = Path(self.temp.name) / "timeout"
            result = self.run_capture(
                "import time;time.sleep(60)", b"", "--seconds", "0.15"
            )
            self.assertEqual(result.returncode, 125)
            self.assertIn("deadline", self.receipt()["error"])

        def test_normal_exit_descendant_group_is_removed(self):
            code = 'import subprocess,sys; p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]);print(p.pid,file=sys.stderr,flush=True)'
            result = self.run_capture(code, b"", "--seconds", "0.2")
            pid = int((self.out / "server.stderr").read_text().strip())
            state = Path(f"/proc/{pid}/stat")
            self.assertTrue(not state.exists() or state.read_text().split()[2] == "Z")
            self.assertEqual(self.receipt()["server_exit_code"], 0)
            self.assertEqual(result.returncode, 125)

        def test_interruption_removes_only_owned_server(self):
            ready = Path(self.temp.name) / "ready"
            code = f'import os,time;open({str(ready)!r},"w").write(str(os.getpid()));time.sleep(60)'
            args = [
                sys.executable,
                "-B",
                __file__,
                "--out-dir",
                str(self.out),
                "--",
                sys.executable,
                "-B",
                "-c",
                code,
            ]
            process = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                end = time.monotonic() + 2
                while not ready.exists() and time.monotonic() < end:
                    time.sleep(0.01)
                self.assertTrue(ready.exists())
                process.send_signal(signal.SIGTERM)
                process.communicate(timeout=5)
                self.assertEqual(process.returncode, 125)
                self.assertIn("interrupted", self.receipt()["error"])
                self.assertFalse(Path(f"/proc/{ready.read_text()}/stat").exists())
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate(timeout=5)

        def test_closed_output_consumer_refuses(self):
            code = f"import os,sys;sys.stdin.buffer.readline();os.write(1,{reply!r})"
            args = [
                sys.executable,
                "-B",
                __file__,
                "--out-dir",
                str(self.out),
                "--",
                sys.executable,
                "-B",
                "-c",
                code,
            ]
            process = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            process.stdout.close()
            process.stdin.write(request)
            process.stdin.close()
            try:
                self.assertEqual(process.wait(timeout=5), 125)
                self.assertIn("BrokenPipe", self.receipt()["error"])
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
                process.stderr.close()

        def test_frame_bound_and_launch_failure_are_receipted(self):
            result = self.run_capture('import os;os.write(1,b"x"*(1024*1024+1))')
            self.assertEqual(result.returncode, 125)
            self.assertIn("frame exceeds", self.receipt()["error"])
            self.out = Path(self.temp.name) / "missing-command"
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    __file__,
                    "--out-dir",
                    str(self.out),
                    "--",
                    str(Path(self.temp.name) / "absent"),
                ],
                input=b"",
                capture_output=True,
                timeout=5,
            )
            self.assertEqual(result.returncode, 125)
            self.assertIsNone(self.receipt()["server_exit_code"])
            self.assertFalse(self.receipt()["complete"])

        def test_early_server_exit_retains_pending_input_and_refuses(self):
            # Delay returning the real child handle until the child has exited.
            # This exposes the drain-before-exit boundary deterministically.
            code = (
                "import sys,time;sys.path.insert(0,sys.argv[1]);import capture_stdio as m;"
                "real=m.subprocess.Popen\n"
                "def delayed(*a,**k):\n p=real(*a,**k);time.sleep(.1);return p\n"
                "m.subprocess.Popen=delayed\n"
                "sys.exit(m.record([sys.argv[3]],m.Path(sys.argv[2]),2,1048576))"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-c",
                    code,
                    str(Path(__file__).parent),
                    str(self.out),
                    shutil.which("true"),
                ],
                input=request,
                capture_output=True,
                timeout=5,
            )
            self.assertEqual(result.returncode, 125)
            self.assertFalse(self.receipt()["complete"])
            self.assertEqual((self.out / "client.raw").read_bytes(), request)

        def test_source_change_is_not_labelled_as_complete(self):
            copy = Path(self.temp.name) / "producer"
            copy.mkdir()
            for name in ("capture_stdio.py", "verify_capture.py"):
                shutil.copyfile(Path(__file__).with_name(name), copy / name)
            script = copy / "capture_stdio.py"
            expected = digest(script)
            code = f'from pathlib import Path;p=Path({str(script)!r});p.write_bytes(p.read_bytes()+b"\\n")'
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(script),
                    "--out-dir",
                    str(self.out),
                    "--",
                    sys.executable,
                    "-B",
                    "-c",
                    code,
                ],
                input=b"",
                capture_output=True,
                timeout=5,
            )
            self.assertEqual(result.returncode, 125)
            self.assertEqual(self.receipt()["producer_sha256"], expected)
            self.assertIn("source changed", self.receipt()["error"])

        def test_default_environment_strips_secrets_and_keeps_runtime_controls(self):
            from unittest.mock import patch

            secret_names = [
                "GH_TOKEN",
                "AWS_SECRET_ACCESS_KEY",
                "OPENAI_API_KEY",
                "MCP_CAPTURE_TEST_SECRET",
            ]
            runtime = dict(
                PATH=os.defpath,
                HOME=self.temp.name,
                LANG="C.UTF-8",
                TMPDIR=self.temp.name,
                JOPEDIME_HOME=str(Path(self.temp.name) / "isolated-store"),
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                RAYON_NUM_THREADS="1",
                CUDA_VISIBLE_DEVICES="",
            )
            parent = {
                **runtime,
                **dict.fromkeys(secret_names, "synthetic-sentinel-not-a-credential"),
            }
            names = list(parent)
            code = (
                "import os,json;names="
                + repr(names)
                + ';print(json.dumps({"jsonrpc":"2.0","id":"env","result":{n:os.environ.get(n) for n in names}}))'
            )
            with patch.dict(os.environ, parent, clear=True):
                result = self.run_capture(code)
            self.assertEqual(result.returncode, 0)
            observed = json.loads(result.stdout)["result"]
            self.assertTrue(all(observed[name] is None for name in secret_names))
            self.assertEqual({name: observed[name] for name in runtime}, runtime)
            self.assertEqual(set(self.receipt()["environment_names"]), set(runtime))
            with patch.dict(
                os.environ,
                {
                    "LD_PRELOAD": "sentinel",
                    "PYTHONPATH": "sentinel",
                    "HTTPS_PROXY": "sentinel",
                    "AWS_PROFILE": "sentinel",
                },
                clear=True,
            ):
                self.assertEqual(child_environment([]), {"PATH": os.defpath})

        def test_explicit_environment_opt_in_records_names_only(self):
            from unittest.mock import patch

            name = "MCP_CAPTURE_TEST_CONFIG"
            value = "synthetic-value-not-for-receipt"
            # Use only synthetic configuration; no host credential values enter the child.
            code = (
                'import os,json;print(json.dumps(dict(jsonrpc="2.0",id="env",result=os.environ.get("'
                + name
                + '"))))'
            )
            with patch.dict(
                os.environ,
                {"PATH": os.defpath, "HOME": self.temp.name, name: value},
                clear=True,
            ):
                result = self.run_capture(
                    code, b"", "--env-allow", name, "--env-allow", name
                )
            self.assertEqual(result.returncode, 0)
            self.assertEqual(json.loads(result.stdout)["result"], value)
            self.assertEqual(self.receipt()["extra_environment_names"], [name])
            self.assertIn(name, self.receipt()["environment_names"])
            self.assertNotIn(value, (self.out / "capture.json").read_text())

        def test_missing_or_invalid_environment_opt_in_refuses_before_launch(self):
            from unittest.mock import patch

            for name in ["UNSET_CAPTURE_CONFIG", "", "KEY=synthetic-value", "0BAD"]:
                with (
                    self.subTest(name=name),
                    patch.dict(os.environ, {"PATH": os.defpath}, clear=True),
                ):
                    result = self.run_capture(
                        'raise RuntimeError("must not launch")',
                        b"",
                        "--env-allow",
                        name,
                    )
                    self.assertEqual(result.returncode, 125)
                    self.assertFalse(self.out.exists())
                    self.assertNotIn(b"synthetic-value", result.stderr)

    return (
        0
        if unittest.TextTestRunner(verbosity=2)
        .run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
        .wasSuccessful()
        else 1
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--max-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument(
        "--env-allow",
        action="append",
        default=[],
        metavar="NAME",
        help="forward one additional existing environment variable by name (repeatable)",
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.self_test:
        if args.out_dir or args.command or args.env_allow:
            parser.error("--self-test cannot record a server")
        return self_test()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not args.out_dir or not command:
        parser.error("--out-dir and -- SERVER [ARGS] are required")
    try:
        return record(
            command, args.out_dir, args.seconds, args.max_bytes, args.env_allow
        )
    except (CaptureError, OSError) as exc:
        detail = str(exc) if isinstance(exc, CaptureError) else type(exc).__name__
        print(f"MCP capture did not start: {detail}", file=sys.stderr)
        return 125


if __name__ == "__main__":
    sys.exit(main())
