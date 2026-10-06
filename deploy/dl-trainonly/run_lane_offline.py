#!/usr/bin/env python3
"""Run only audited DL source fixtures, offline and with GPUs hidden."""
import argparse
import contextlib
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import time
import unittest

from verify_lane import verify

MODULES = (
    "tests.test_rowsel_raw_split_v2", "tests.test_v2_rowsel_raw", "tests.test_v2_rowsel_raw_helper",
    "tests.test_v2_publish", "tests.test_rowsel", "tests.test_nonnumeric", "tests.test_t1raw",
    "tests.test_t1prep", "tests.test_trainonly", "tests.test_s3io", "tests.test_v2_rowsel_build",
)


def case_ids(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from case_ids(test)
        else:
            yield test.id()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    scratch = out / "scratch"
    scratch.mkdir()
    os.environ["TMPDIR"] = str(scratch)
    for name in list(os.environ):
        if name.startswith(("AWS_", "JOPE_AWS_", "GH_", "GITHUB_")):
            os.environ.pop(name)
    os.environ.update({"AWS_EC2_METADATA_DISABLED": "true", "AWS_CONFIG_FILE": "/dev/null",
                       "AWS_SHARED_CREDENTIALS_FILE": "/dev/null", "AWS_ACCESS_KEY_ID": "offline-fixture",
                       "AWS_SECRET_ACCESS_KEY": "offline-fixture", "AWS_DEFAULT_REGION": "us-east-2",
                       "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                       "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""})
    sys.dont_write_bytecode = True
    cpus = sorted(os.sched_getaffinity(0))[:4]
    os.sched_setaffinity(0, cpus)
    limit = 16 * 1024**3
    old_soft, old_hard = resource.getrlimit(resource.RLIMIT_AS)
    memory = min(limit, old_hard) if old_hard != resource.RLIM_INFINITY else limit
    resource.setrlimit(resource.RLIMIT_AS, (memory, old_hard))
    forbidden, probes = [], []
    secret_roots = tuple(Path.home() / name for name in (".aws", ".ssh", ".config/gh", ".config/jopedime/secrets", ".claude"))

    def audit(event, values):
        if event == "open" and isinstance(values[0], (str, bytes)):
            path = Path(os.fsdecode(values[0])).resolve()
            if any(path == prefix or path.is_relative_to(prefix) for prefix in secret_roots):
                forbidden.append("credential-file read")
                raise RuntimeError("offline fixtures forbid credential-file reads")
        if event == "subprocess.Popen" and values[0] == "/sbin/ldconfig" and values[1] == ["/sbin/ldconfig", "-p"]:
            probes.append({"executable": values[0], "argv": values[1]})
            return
        if event in ("socket.connect", "socket.getaddrinfo", "subprocess.Popen", "os.system"):
            forbidden.append(event)
            raise RuntimeError("offline fixtures forbid " + event)

    sys.addaudithook(audit)
    custody = Path(__file__).with_name("lane")
    before = verify(custody)
    root = custody / "src/dl_training_v1"
    os.chdir(root)
    sys.path.insert(0, str(root))
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "boto3", "botocore")}
    suite = unittest.defaultTestLoader.loadTestsFromNames(MODULES)
    names = list(case_ids(suite))
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    tick = time.monotonic()
    with (out / "unittest.log").open("x") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    after = verify(custody)
    report = {"schema": "jope.dl-offline-fixtures/v1", "start_utc": start,
              "end_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "seconds": time.monotonic() - tick,
              "argv": sys.argv, "python": sys.executable, "cwd": str(root), "dependencies": versions,
              "handoff_sha256": before["handoff_sha256"], "source_unchanged": before == after,
              "modules": list(MODULES), "case_ids": names, "collected": len(names), "executed": result.testsRun,
              "failed": len(result.failures) + len(result.errors), "skipped": len(result.skipped),
              "expected_failures": len(result.expectedFailures), "unexpected_successes": len(result.unexpectedSuccesses),
              "forbidden_attempts": forbidden, "library_cache_probes": probes,
              "resource_limits": {"affinity_cpus": cpus, "address_space_bytes": memory, "threads": 1},
              "log_sha256": hashlib.sha256((out / "unittest.log").read_bytes()).hexdigest(),
              "scope": "audited synthetic offline CPU fixtures; no live corpus, pilot, training, GPU, Lambda, S3 or runtime adoption"}
    with (out / "receipt.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("collected", "executed", "failed", "skipped", "seconds", "handoff_sha256", "source_unchanged")}))
    return not (result.wasSuccessful() and not forbidden and before == after and len(names) == result.testsRun == 329
                and not result.skipped and not result.expectedFailures and not result.unexpectedSuccesses)


if __name__ == "__main__":
    raise SystemExit(main())
