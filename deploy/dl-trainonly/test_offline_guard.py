#!/usr/bin/env python3
"""Python audit guard refusals, with no network traffic or numerical jobs."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

from unittest import mock

from run_lane_offline import HOST_CASES, PRIVATE_INPUT_ROOTS, case_ids, make_audit, offline_suite, private_inputs_unavailable


class AuditGuard(unittest.TestCase):
    def setUp(self):
        self.denied, self.probes = [], []
        self.audit = make_audit(self.denied, self.probes, Path.home())

    def test_socket_audit_events_all_refused(self):
        for event in ("socket.__new__", "socket.connect", "socket.sendto", "socket.sendmsg", "socket.bind",
                      "socket.getaddrinfo", "socket.gethostbyname", "socket.unknown_future_event"):
            with self.assertRaisesRegex(RuntimeError, "offline fixtures forbid socket"):
                self.audit(event, ())
        self.assertEqual(len(self.denied), 8)

    def test_direct_process_audit_events_all_refused(self):
        for event in ("os.system", "os.posix_spawn", "os.fork", "os.forkpty", "os.exec", "os.spawnv", "subprocess.Popen"):
            with self.assertRaisesRegex(RuntimeError, "offline fixtures forbid"):
                self.audit(event, ("/never-executed", ["/never-executed"], None, {}))
        self.assertEqual(len(self.denied), 7)

    def test_only_exact_read_only_library_probe_permitted(self):
        for event in ("subprocess.Popen", "os.posix_spawn"):
            self.audit(event, ("/sbin/ldconfig", ["/sbin/ldconfig", "-p"], None, {}))
        self.assertEqual(len(self.probes), 2)
        self.assertEqual(self.denied, [])
        for event in ("subprocess.Popen", "os.posix_spawn", "os.exec"):
            for command in (["/sbin/ldconfig"], ["/sbin/ldconfig", "-p", "-v"], ["/sbin/ldconfig", "-n"]):
                with self.assertRaises(RuntimeError):
                    self.audit(event, ("/sbin/ldconfig", command, None, {}))
        with self.assertRaises(RuntimeError):
            self.audit("os.exec", ("/sbin/ldconfig", ["/sbin/ldconfig", "-p"], {}))

    def test_python_credential_reads_refused(self):
        for name in (".aws/config", ".ssh/key", ".config/gh/hosts.yml", ".config/jopedime/secrets/aws.env", ".claude/key"):
            with self.assertRaisesRegex(RuntimeError, "credential-file reads"):
                self.audit("open", (str(Path.home() / name), "r", 0))
        self.assertEqual(len(self.denied), 5)
        self.audit("open", ("/dev/null", "r", 0))

    def test_private_lane_input_reads_refused(self):
        for root in PRIVATE_INPUT_ROOTS:
            with self.assertRaisesRegex(RuntimeError, "private-lane input reads"):
                self.audit("open", (str(root / "fixture-not-read.json"), "r", 0))
        self.assertEqual(self.denied, ["private-lane input read"] * 2)

    def test_read_guard_refuses_lexical_and_resolved_symlink_escapes(self):
        private = PRIVATE_INPUT_ROOTS[0] / "fixture-not-read.json"
        with mock.patch.object(Path, "resolve", return_value=Path("/dev/null")) as resolve:
            with self.assertRaisesRegex(RuntimeError, "private-lane input reads"):
                self.audit("open", (str(private), "r", 0))
            with self.assertRaisesRegex(RuntimeError, "credential-file reads"):
                self.audit("open", (str(Path.home() / ".aws/fixture"), "r", 0))
            resolve.assert_not_called()
        with mock.patch.object(Path, "resolve", return_value=private):
            with self.assertRaisesRegex(RuntimeError, "private-lane input reads"):
                self.audit("open", ("/fixture/alias-not-opened.json", "r", 0))

    def test_only_declared_host_cases_excluded_without_private_stat(self):
        class Named(unittest.TestCase):
            def __init__(self, name):
                super().__init__()
                self.name = name

            def id(self):
                return self.name

        stat = Path.stat
        private = PRIVATE_INPUT_ROOTS[0] / "fixture-not-statted.json"

        def checked_stat(path, *args, **kwargs):
            if path == private:
                raise AssertionError("private host stat attempted")
            return stat(path, *args, **kwargs)

        class Loader:
            def loadTestsFromNames(loader, modules):
                self.assertFalse(private.exists())
                self.assertTrue(Path("/dev/null").exists())
                return unittest.TestSuite(Named(name) for name in modules)

        names = list(HOST_CASES) + ["fixture.authorized"]
        unavailable = []
        with mock.patch.object(Path, "stat", checked_stat):
            suite, collected = offline_suite(Loader(), names, unavailable)
        self.assertEqual(collected, names)
        self.assertEqual(list(case_ids(suite)), ["fixture.authorized"])
        self.assertEqual(unavailable, [str(private)])
        for malformed in (names[1:], names + [names[0]]):
            with self.assertRaisesRegex(ValueError, "unique declared host cases"):
                offline_suite(Loader(), malformed, [])

    def test_unrelated_events_do_not_create_false_failures(self):
        self.audit("import", ("fixture",))
        self.audit("os.listdir", ("/fixture",))
        self.assertEqual(self.denied, [])
        self.assertEqual(self.probes, [])

    def test_private_existence_unavailable_during_execution_and_restored(self):
        exists, stat = Path.exists, Path.stat
        unavailable, attempted = [], []
        private_paths = [root / "fixture-not-statted/context.json" for root in PRIVATE_INPUT_ROOTS]

        def checked_stat(path, *args, **kwargs):
            if path in private_paths:
                attempted.append(str(path))
                raise PermissionError("host private path is inaccessible")
            return stat(path, *args, **kwargs)

        class Fixture(unittest.TestCase):
            def runTest(inner):
                for path in private_paths:
                    inner.assertFalse(path.exists())
                inner.assertTrue(Path("/dev/null").exists())

        with mock.patch.object(Path, "stat", checked_stat):
            with private_inputs_unavailable(unavailable):
                result = unittest.TestResult()
                Fixture().run(result)
            self.assertEqual(result.testsRun, 1)
            self.assertEqual(result.errors, [])
            self.assertEqual(result.failures, [])
            self.assertEqual(attempted, [])
            self.assertEqual(unavailable, [str(path) for path in private_paths])
            self.assertIs(Path.exists, exists)
            with self.assertRaises(PermissionError):
                private_paths[0].exists()
        self.assertEqual(attempted, [str(private_paths[0])])

    def child(self, body):
        script = "import os, socket, subprocess, sys\nfrom pathlib import Path\nfrom run_lane_offline import make_audit\n"
        script += "denied=[];probes=[]\nsys.addaudithook(make_audit(denied,probes,Path.home()))\n" + body
        env = {k: value for k, value in os.environ.items() if not k.startswith(("AWS_", "JOPE_AWS_", "GH_", "GITHUB_"))}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).parent, env=env,
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_actual_tcp_udp_unix_creation_refused_before_any_traffic(self):
        self.child("""for family, kind in ((socket.AF_INET,socket.SOCK_STREAM), (socket.AF_INET,socket.SOCK_DGRAM), (socket.AF_UNIX,socket.SOCK_STREAM), (socket.AF_INET6,socket.SOCK_STREAM)):
 try:
  s=socket.socket(family,kind)
 except RuntimeError:
  pass
 else:
  s.close();raise AssertionError('socket creation was permitted')
assert denied == ['socket.__new__']*4
""")

    def test_actual_spawn_and_exec_calls_refused_before_launch(self):
        self.child("""for operation in (
 lambda:subprocess.Popen(['/never-executed']),
 lambda:os.posix_spawn('/never-executed',['/never-executed'],{}),
 lambda:os.execv('/never-executed',['/never-executed'])):
 try:
  operation()
 except RuntimeError:
  pass
 else:
  raise AssertionError('process launch was permitted')
assert denied == ['subprocess.Popen','os.posix_spawn','os.exec']
""")

    def test_actual_fork_refused_before_creating_child(self):
        self.child("""try:
 pid=os.fork()
except RuntimeError:
 pass
else:
 if pid==0:os._exit(93)
 os.waitpid(pid,0)
 raise AssertionError('fork was permitted')
assert denied == ['os.fork']
""")


if __name__ == '__main__':
    unittest.main()
