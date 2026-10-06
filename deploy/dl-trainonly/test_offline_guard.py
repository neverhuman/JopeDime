#!/usr/bin/env python3
"""Python audit guard refusals, with no network traffic or numerical jobs."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

from run_lane_offline import make_audit


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

    def test_unrelated_events_do_not_create_false_failures(self):
        self.audit("import", ("fixture",))
        self.audit("os.listdir", ("/fixture",))
        self.assertEqual(self.denied, [])
        self.assertEqual(self.probes, [])

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
