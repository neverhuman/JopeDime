import math
import os
import signal
import subprocess
import time


def child_environment(folder, target, jobs, source):
    forwarded = (
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "CARGO_HOME",
        "RUSTUP_HOME",
        "CC",
        "CXX",
        "AR",
        "LD_LIBRARY_PATH",
        "LIBTORCH",
        "LIBTORCH_USE_PYTORCH",
        "LIBTORCH_BYPASS_VERSION_CHECK",
        "CMAKE_PREFIX_PATH",
        "PKG_CONFIG_PATH",
        "UV_CACHE_DIR",
    )
    environment = {key: os.environ[key] for key in forwarded if key in os.environ}
    environment.update(
        CUDA_VISIBLE_DEVICES="",
        AWS_EC2_METADATA_DISABLED="true",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        RAYON_NUM_THREADS="1",
        CARGO_BUILD_JOBS=str(jobs),
        CARGO_INCREMENTAL="0",
        CARGO_PROFILE_DEV_DEBUG="line-tables-only",
        CARGO_TARGET_DIR=str(target),
        JOPE_LANE_EVIDENCE=str(folder),
        JOPE_SOURCE_SHA=source,
        JOPE_PARITY_STRICT="1",
        JOPE_CPU_TEST_THREADS=str(jobs),
        JOPE_BUILD_JOBS=str(jobs),
        JOPE_CPU_BUILD_JOBS=str(jobs),
        PYTHONDONTWRITEBYTECODE="1",
        TMPDIR=str(folder / "scratch"),
    )
    return environment


def validate_timeout(seconds):
    if not math.isfinite(seconds) or seconds <= 0 or seconds > 3600:
        raise ValueError(
            "gate timeout must be finite, positive and at most 3600 seconds"
        )


def _wait_unreaped(process, seconds):
    # Keep the leader's PID reserved until cleanup has signalled its whole group.
    deadline = time.monotonic() + seconds
    while (
        os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None
    ):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, seconds)
        time.sleep(min(0.05, remaining))


def _stop_owned_group(process):
    try:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        time.sleep(5)
    finally:
        # The leader may have exited, but a descendant can still ignore SIGTERM.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def invoke(command, root, log, environment, seconds):
    validate_timeout(seconds)
    with log.open("xb") as stream:
        process = subprocess.Popen(
            ["nice", "-n", "10", *command],
            cwd=root,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_unreaped(process, seconds)
        except subprocess.TimeoutExpired:
            _stop_owned_group(process)
            return 124, True
        except BaseException:
            _stop_owned_group(process)
            raise
        # The exited leader remains unreaped, reserving its PID/PGID while
        # owned descendants are stopped before receipt/log hashing. Preserve
        # the leader's original exit status rather than treating cleanup as success.
        _stop_owned_group(process)
        return process.returncode, False


