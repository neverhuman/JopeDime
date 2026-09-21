import fcntl
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import resident_gpu_pool
from resident_gpu_pool import (
    idle_worker,
    native_request_id,
    process_identity,
    rank_observation,
    reconcile_startup,
    reconcile_tasks,
    verify_pool_ready,
    worker_matches,
)
from resume_expired_cycle import atomic, read, sha


def compile_worker(root):
    """A real ELF process with a held rank lock and Unix socket; no CUDA linkage."""
    source = root / "synthetic-worker.c"
    binary = root / "synthetic-worker"
    source.write_text(r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <time.h>
int main(int argc,char **argv) {
    char *pool=NULL; int rank=-1;
    for(int i=1;i+1<argc;i++) {
        if(!strcmp(argv[i],"--pool"))pool=argv[++i];
        else if(!strcmp(argv[i],"--rank"))rank=atoi(argv[++i]);
    }
    if(!pool || rank<0)return 2;
    char path[4096],line[4096],boot[128],digest[128],command[256];
    snprintf(path,sizeof(path),"%s/rank-%d.lock",pool,rank);
    int lock=open(path,O_RDWR|O_CREAT,0600);if(lock<0 || flock(lock,LOCK_EX|LOCK_NB))return 3;
    int sock=socket(AF_UNIX,SOCK_STREAM,0);struct sockaddr_un address={.sun_family=AF_UNIX};
    snprintf(address.sun_path,sizeof(address.sun_path),"%s/rank-%d.sock",pool,rank);
    unlink(address.sun_path);
    if(bind(sock,(void*)&address,sizeof(address)) || listen(sock,8))return 4;
    struct stat endpoint;if(stat(address.sun_path,&endpoint))return 5;
    FILE *file=fopen("/proc/self/stat","r");if(!file || !fgets(line,sizeof(line),file))return 6;fclose(file);
    char *field=strrchr(line,')')+2;unsigned long ticks=0;
    for(int n=0;n<=19;n++) {char *next=strchr(field,' ');if(n==19)ticks=strtoul(field,NULL,10);if(next)field=next+1;}
    file=fopen("/proc/sys/kernel/random/boot_id","r");if(!file || !fgets(boot,sizeof(boot),file))return 7;fclose(file);
    boot[strcspn(boot,"\n")]=0;
    snprintf(command,sizeof(command),"/usr/bin/sha256sum /proc/%d/exe",getpid());
    file=popen(command,"r");if(!file || !fgets(digest,sizeof(digest),file))return 8;pclose(file);digest[64]=0;
    snprintf(path,sizeof(path),"%s/rank-%d.json.tmp",pool,rank);file=fopen(path,"w");if(!file)return 9;
    fprintf(file,"{\"pid\":%d,\"rank\":%d,\"timestamp\":%ld,\"status\":\"idle\",\"worker\":{\"process\":{\"pid\":%d,\"boot_id\":\"%s\",\"start_ticks\":%lu},\"executable_sha256\":\"%s\",\"socket_device\":%lu,\"socket_inode\":%lu}}",getpid(),rank,time(NULL),getpid(),boot,ticks,digest,(unsigned long)endpoint.st_dev,(unsigned long)endpoint.st_ino);
    fflush(file);fsync(fileno(file));fclose(file);
    snprintf(line,sizeof(line),"%s/rank-%d.json",pool,rank);if(rename(path,line))return 10;
    while(1)sleep(1);
}
""")
    subprocess.run(["cc", "-O0", "-o", str(binary), str(source)], check=True, capture_output=True, timeout=10)
    return binary


def publish_task(root, pool, executable, terminal=None):
    request = dict(
        root=str(root),
        rank=0,
        executable_sha256=executable,
        task=dict(kind="Monitor", output=str(root / "original-output.json")),
        env={},
    )
    identity = native_request_id(request)
    task = pool / "tasks" / identity
    task.mkdir(parents=True)
    (pool / "dispatches").mkdir(exist_ok=True)
    atomic(
        pool / "dispatches" / identity, dict(request_sha256=identity, client=process_identity(os.getpid()))
    )
    atomic(task / "request.json", request)
    atomic(
        task / "accepted.json",
        dict(
            schema="parallel.resident_task_acceptance.v2",
            request_sha256=identity,
            worker=dict(process=process_identity(os.getpid()), executable_sha256=executable),
        ),
    )
    if terminal:
        atomic(root / "original-output.json", dict(original="output"))
        success = terminal == "complete"
        atomic(
            task / (terminal + ".json"),
            dict(
                schema="parallel.resident_task_completion.v2",
                request_sha256=identity,
                accepted_sha256=sha(task / "accepted.json"),
                success=success,
                receipt_sha256=sha(root / "original-output.json") if success else None,
                error=None if success else "injected",
            ),
        )
    return identity


class ResidentCustodyTests(unittest.TestCase):
    def setUp(self):
        self.identity = process_identity(os.getpid())
        self.executable = sha(Path("/proc/self/exe"))
        self.state = dict(
            pid=os.getpid(),
            status="idle",
            worker=dict(process=self.identity, executable_sha256=self.executable),
        )

    def test_current_process_requires_birth_boot_and_executable(self):
        self.assertTrue(worker_matches(self.state, self.identity, self.executable))
        self.assertTrue(idle_worker(self.state, os.getpid(), self.identity, self.executable))
        self.assertFalse(idle_worker(self.state, os.getpid()))
        self.assertFalse(worker_matches(self.state, self.identity, "b" * 64))
        for key, value in [("start_ticks", self.identity["start_ticks"] + 1), ("boot_id", "previous boot")]:
            self.assertFalse(worker_matches(self.state, dict(self.identity, **{key: value}), self.executable))
        self.state["status"] = "busy"
        self.assertFalse(idle_worker(self.state, os.getpid(), self.identity, self.executable))

    def test_pid_reuse_and_disappeared_process_do_not_authorize_dispatch(self):
        with patch("resident_gpu_pool.process_identity", return_value=None):
            self.assertFalse(worker_matches(self.state, self.identity, self.executable))
        with patch("resident_gpu_pool.process_identity", return_value=dict(self.identity, start_ticks=0)):
            self.assertFalse(worker_matches(self.state, self.identity, self.executable))

    def test_executable_claim_cannot_replace_loaded_process_evidence(self):
        self.state["worker"]["executable_sha256"] = "c" * 64
        self.assertFalse(worker_matches(self.state, self.identity, "c" * 64))

    def live_rank(self, root, pool):
        fixture = compile_worker(root)
        child = subprocess.Popen([str(fixture), "--pool", str(pool), "--rank", "0"])

        def cleanup():
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)

        self.addCleanup(cleanup)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            self.assertIsNone(child.poll())
            if (pool / "rank-0.json").is_file():
                return child, fixture
            time.sleep(0.01)
        self.fail("synthetic ELF worker did not publish readiness")

    def test_real_socket_replacement_is_visible_without_connecting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool = root / "pool"
            pool.mkdir()
            child, fixture = self.live_rank(root, pool)
            identity = process_identity(child.pid)
            before = rank_observation(pool, 0, identity, sha(fixture))
            (pool / "rank-0.sock").unlink()
            with socket.socket(socket.AF_UNIX) as replacement:
                replacement.bind(str(pool / "rank-0.sock"))
                with self.assertRaisesRegex(RuntimeError, "socket identity"):
                    rank_observation(pool, 0, identity, sha(fixture))
            self.assertIsNone(child.poll())
            self.assertEqual(before["process"], identity)

    def test_replaced_lock_does_not_hide_surviving_worker_on_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool = root / "pool"
            pool.mkdir()
            child, fixture = self.live_rank(root, pool)
            (pool / "rank-0.lock").unlink()
            (pool / "rank-0.lock").touch()
            with self.assertRaisesRegex(RuntimeError, "surviving worker"):
                reconcile_startup(pool, 2)
            with self.assertRaisesRegex(RuntimeError, "rank lock descriptor"):
                rank_observation(pool, 0, process_identity(child.pid), sha(fixture))
            self.assertIsNone(child.poll())

    def test_pool_readiness_requires_fresh_supervisor_and_all_original_ranks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool = root / "pool"
            pool.mkdir()
            child, fixture = self.live_rank(root, pool)
            ownership = pool / "supervisors/test/start.json"
            ownership.parent.mkdir(parents=True)
            binding = dict(
                schema="parallel.resident_supervisor_ownership.v1",
                supervisor=self.identity,
                worker_executable_sha256=sha(fixture),
                supervisor_executable_sha256=self.executable,
                sources={
                    str(Path(resident_gpu_pool.__file__).absolute()): sha(Path(resident_gpu_pool.__file__))
                },
            )
            atomic(ownership, binding)
            state = dict(
                schema="parallel.resident_gpu_pool.v3",
                phase="running",
                world=1,
                executable_sha256=sha(fixture),
                supervisor=self.identity,
                timestamp=time.time(),
                ownership=str(ownership),
                ownership_sha256=sha(ownership),
                processes={"0": process_identity(child.pid)},
            )
            atomic(pool / "pool.json", state)
            verify_pool_ready(pool, 1, sha(fixture), self.identity)
            state["timestamp"] -= 10
            atomic(pool / "pool.json", state)
            with self.assertRaisesRegex(RuntimeError, "stale"):
                verify_pool_ready(pool, 1, sha(fixture), self.identity)
            state["timestamp"] = time.time()
            state["processes"] = {}
            atomic(pool / "pool.json", state)
            with self.assertRaisesRegex(RuntimeError, "omits rank"):
                verify_pool_ready(pool, 1, sha(fixture), self.identity)
            state["processes"] = {"0": process_identity(child.pid)}
            binding["sources"][str(Path(resident_gpu_pool.__file__).absolute())] = "a" * 64
            atomic(ownership, binding)
            state["ownership_sha256"] = sha(ownership)
            atomic(pool / "pool.json", state)
            with self.assertRaisesRegex(RuntimeError, "source binding"):
                verify_pool_ready(pool, 1, sha(fixture), self.identity)

    def test_supervisor_restart_preserves_surviving_rank_custody(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = Path(tmp)
            with (pool / "rank-1.lock").open("a") as original:
                fcntl.flock(original, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, "execution custody"):
                    reconcile_startup(pool, 2)
            reconcile_startup(pool, 2)

    def test_real_supervisor_restarts_synthetic_rank_and_drains_on_hold(self):
        self.assertTrue(
            {0, 1, 2, 3}.issubset(os.sched_getaffinity(0)),
            "synthetic supervisor test requires its configured CPU fixture",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool = root / "runs/pool"
            pool.mkdir(parents=True)
            (root / "runs/encoding-service.lock").touch()
            fixture = compile_worker(root)
            command = [
                sys.executable,
                "-B",
                str(Path(__file__).with_name("resident_gpu_pool.py")),
                "--root",
                str(root),
                "--binary",
                str(fixture),
                "--pool",
                str(pool),
                "--world",
                "2",
            ]
            with (root / "supervisor.log").open("w") as log:
                supervisor = subprocess.Popen(command, stdout=log, stderr=log)
                try:

                    def wait_for(predicate):
                        deadline = time.monotonic() + 10
                        while time.monotonic() < deadline:
                            self.assertIsNone(supervisor.poll(), (root / "supervisor.log").read_text())
                            if (pool / "pool.json").is_file():
                                state = read(pool / "pool.json")
                                if predicate(state):
                                    return state
                            time.sleep(0.05)
                        self.fail("real supervisor did not reach expected synthetic state")

                    before = wait_for(lambda state: state["phase"] == "running")
                    os.kill(before["pids"]["0"], signal.SIGKILL)
                    after = wait_for(
                        lambda state: (
                            state["phase"] == "running" and state["pids"]["0"] != before["pids"]["0"]
                        )
                    )
                    self.assertNotEqual(before["processes"]["0"], after["processes"]["0"])
                    with (root / "runs/encoding-service.lock").open("r+b") as lease:
                        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        atomic(root / "runs/continuous-control.json", dict(paused=True))
                        draining = wait_for(lambda state: state["phase"] == "draining")
                        self.assertEqual(draining["pids"], after["pids"])
                        self.assertTrue(all(process_identity(pid) for pid in after["pids"].values()))
                    held = wait_for(lambda state: state["phase"] == "suspended")
                    self.assertEqual(held["pids"], {})
                finally:
                    supervisor.terminate()
                    try:
                        supervisor.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        supervisor.kill()
                        supervisor.wait()
                        raise
                self.assertEqual(supervisor.returncode, 0, (root / "supervisor.log").read_text())
                cleanup = read(next((pool / "supervisors").glob("*/complete.json")))
                self.assertEqual(cleanup["remaining_owned_processes"], [])
                self.assertEqual(cleanup["reconciliation"]["unknown"], 0)

    def test_unknown_acceptance_survives_two_supervisor_restarts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool = root / "runs/pool"
            pool.mkdir(parents=True)
            (root / "runs/encoding-service.lock").touch()
            fixture = compile_worker(root)
            identity = publish_task(root, pool, sha(fixture))
            before = {str(p): sha(p) for p in (pool / "tasks" / identity).iterdir()}
            command = [
                sys.executable,
                "-B",
                str(Path(__file__).with_name("resident_gpu_pool.py")),
                "--root",
                str(root),
                "--binary",
                str(fixture),
                "--pool",
                str(pool),
                "--world",
                "2",
            ]
            for _ in range(2):
                result = subprocess.run(command, capture_output=True, timeout=10)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(b"requires reconciliation", result.stderr)
                self.assertEqual(read(pool / "pool.json")["phase"], "reconciliation_required")
            self.assertEqual(before, {str(p): sha(p) for p in (pool / "tasks" / identity).iterdir()})
            self.assertFalse((pool / "rank-0.json").exists())

    def test_cache_exit_stops_borrowing_ranks_without_starting_a_new_cache_epoch(self):
        self.assertTrue({0, 1, 2, 3}.issubset(os.sched_getaffinity(0)))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool = root / "runs/pool"
            pool.mkdir(parents=True)
            (root / "runs/encoding-service.lock").touch()
            fixture = compile_worker(root)
            command = [
                sys.executable,
                "-B",
                str(Path(__file__).with_name("resident_gpu_pool.py")),
                "--root",
                str(root),
                "--binary",
                str(fixture),
                "--pool",
                str(pool),
                "--world",
                "2",
            ]
            with (root / "supervisor.log").open("w") as log:
                supervisor = subprocess.Popen(command, stdout=log, stderr=log)
                try:
                    deadline = time.monotonic() + 10
                    before = cache = None
                    while time.monotonic() < deadline:
                        self.assertIsNone(supervisor.poll(), (root / "supervisor.log").read_text())
                        if (pool / "pool.json").is_file() and (pool / "teaching-cache.json").is_file():
                            before = read(pool / "pool.json")
                            cache = read(pool / "teaching-cache.json")
                            if before["phase"] == "running":
                                break
                        time.sleep(0.05)
                    self.assertIsNotNone(before)
                    self.assertEqual(before["phase"], "running")
                    original_cache = process_identity(cache["pid"])
                    self.assertIsNotNone(original_cache)
                    os.kill(cache["pid"], signal.SIGKILL)
                    supervisor.wait(timeout=10)
                    self.assertNotEqual(supervisor.returncode, 0)
                    self.assertEqual(read(pool / "pool.json")["phase"], "reconciliation_required")
                    cleanup = read(next((pool / "supervisors").glob("*/complete.json")))
                    self.assertIn("cache epoch exited", cleanup["failure"])
                    self.assertEqual(cleanup["remaining_owned_processes"], [])
                    self.assertEqual(cleanup["reconciliation"]["unknown"], 0)
                    for identity in before["processes"].values():
                        self.assertNotEqual(process_identity(identity["pid"]), identity)
                    # A restarted cache would publish a different epoch. The
                    # original observation remains the only cache identity.
                    self.assertEqual(read(pool / "teaching-cache.json")["epoch"], cache["epoch"])
                    self.assertEqual(len(list((pool / "supervisors").glob("*/start.json"))), 1)
                finally:
                    if supervisor.poll() is None:
                        supervisor.terminate()
                        try:
                            supervisor.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            supervisor.kill()
                            supervisor.wait()
                            raise


class RetainedTaskEvidenceTests(unittest.TestCase):
    def test_unknown_dispatch_torn_acceptance_and_corrupt_evidence_stay_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool = root / "pool"
            pool.mkdir()
            identity = publish_task(root, pool, "a" * 64)
            snapshot = reconcile_tasks(root, pool, "a" * 64)
            self.assertEqual(snapshot["unknown"], 1)
            self.assertFalse(snapshot["records"][0]["automatic_retry_permitted"])
            (pool / "tasks" / identity / "accepted.json").write_text("{")
            self.assertEqual(reconcile_tasks(root, pool, "a" * 64)["unknown"], 1)
            (pool / "dispatches" / identity).write_text("{")
            self.assertEqual(reconcile_tasks(root, pool, "a" * 64)["unknown"], 1)

    def test_original_completion_can_be_observed_without_restarting_its_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool = root / "pool"
            pool.mkdir()
            identity = publish_task(root, pool, "a" * 64, "complete")
            snapshot = reconcile_tasks(root, pool, "a" * 64)
            self.assertEqual(snapshot["unknown"], 0)
            self.assertEqual(snapshot["records"][0]["status"], "committed_success")
            self.assertFalse(snapshot["execution_authorized"])
            (root / "original-output.json").write_text("changed")
            self.assertEqual(reconcile_tasks(root, pool, "a" * 64)["unknown"], 1)
            self.assertTrue((pool / "tasks" / identity).exists())

    def test_failed_output_cannot_become_reusable_and_symlink_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool = root / "pool"
            pool.mkdir()
            identity = publish_task(root, pool, "a" * 64, "failure")
            snapshot = reconcile_tasks(root, pool, "a" * 64)
            self.assertEqual(snapshot["records"][0]["status"], "committed_failure")
            self.assertNotIn("output", snapshot["records"][0])
            dispatch = pool / "dispatches" / identity
            outside = root / "outside"
            outside.write_bytes(dispatch.read_bytes())
            dispatch.unlink()
            dispatch.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "symlink"):
                reconcile_tasks(root, pool, "a" * 64)


if __name__ == "__main__":
    unittest.main()
