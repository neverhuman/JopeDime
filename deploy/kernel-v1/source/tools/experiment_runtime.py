"""Shared filesystem, journal and process boundary for isolated experiments.

Native children must run under the transient service: path validation alone
cannot prevent a concurrent rename from redirecting a native output write.
"""
from contextlib import contextmanager
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid


def require(condition, message="invalid experiment artifact"):
    if not condition:
        raise ValueError(message)


def checked_path(path):
    path = Path(os.path.abspath(path))
    for part in (*reversed(path.parents), path):
        require(not part.is_symlink(), f"symlink traversal rejected: {part}")
    return path


def inside(root, path):
    root = checked_path(root)
    result = checked_path(root / path)
    require(result.is_relative_to(root), f"path escapes root: {path}")
    return result


def output_root(root, output):
    root, output = checked_path(root), inside(root, output)
    require(output != root and output != root / "runs", "shared directory is not an experiment output")
    # Experiments occupy a dedicated runs child; production stores never do.
    require(output.is_relative_to(root / "runs"), "experiment output must be a dedicated runs directory")
    protected = {root / p for p in ("runs/gp", "runs/champions", "runs/campaign", "runs/checkpoints")}
    registries = list((root / "manifests").rglob("*.json")) + [root / "runs/gp-priority-jobs.json"]
    for path in registries:
        try:
            doc = json.loads(path.read_text())
            if isinstance(doc, dict) and "campaign" in doc.get("schema", "") and isinstance(doc.get("output"), str):
                protected.add(inside(root, doc["output"]))
            if isinstance(doc, dict) and doc.get("schema") == "parallel.gp_jobs.v1":
                # GP stores also live outside campaign directories. Reserve the
                # entire runs child, including cuts a later queue may admit.
                for job in doc.get("jobs", []):
                    if not isinstance(job, dict) or not isinstance(job.get("output"), str):
                        continue
                    destination = Path(os.path.abspath(root / job["output"]))
                    if destination.is_relative_to(root / "runs"):
                        parts = destination.relative_to(root / "runs").parts
                        protected.add(root / "runs" / parts[0] if parts else root / "runs")
        except (OSError, ValueError, TypeError):
            continue
    for name in ("campaign-progress.json", "cycle-progress.json"):
        path = root / "runs" / name
        if path.exists():
            doc = json.loads(path.read_text())
            for key in ("output", "current_cycle", "campaign_output"):
                if isinstance(doc.get(key), str):
                    protected.add(inside(root, doc[key]))
    for path in protected:
        require(not (output.is_relative_to(path) or path.is_relative_to(output)),
                f"experiment output overlaps production: {path}")
    if output.exists():
        for directory, dirs, files in os.walk(output, followlinks=False):
            for name in dirs + files:
                checked_path(Path(directory) / name)
    return output


@contextmanager
def parent_fd(path, create=False):
    """Walk every directory using O_NOFOLLOW, retaining the destination dirfd."""
    path = checked_path(path)
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for name in path.parts[1:-1]:
            if create:
                try:
                    os.mkdir(name, mode=0o700, dir_fd=fd)
                except FileExistsError:
                    pass
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd, path.name
    finally:
        os.close(fd)


def atomic_bytes(path, data):
    with parent_fd(path, create=True) as (fd, name):
        temporary = "." + name + "." + uuid.uuid4().hex + ".tmp"
        try:
            out = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
            with os.fdopen(out, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, name, src_dir_fd=fd, dst_dir_fd=fd)
            os.fsync(fd)
        finally:
            try:
                os.unlink(temporary, dir_fd=fd)
            except FileNotFoundError:
                pass


def write(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def journal(path, repair=False):
    path = checked_path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    end = data.rfind(b"\n") + 1
    rows = []
    for number, line in enumerate(data[:end].splitlines(), 1):
        try:
            value = json.loads(line)
            require(isinstance(value, dict), "record must be an object")
            rows.append(value)
        except (ValueError, UnicodeError) as error:
            raise ValueError(f"corrupt complete journal record {path}:{number}") from error
    if end != len(data) and repair:
        # Preserve evidence before trimming only the uncommitted final suffix.
        evidence = path.with_name(path.name + ".torn-" + hashlib.sha256(data[end:]).hexdigest())
        atomic_bytes(evidence, data[end:])
        atomic_bytes(path, data[:end])
    return rows


def append(path, value):
    data = (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode()
    with parent_fd(path, create=True) as (directory, name):
        fd = os.open(name, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600, dir_fd=directory)
        with os.fdopen(fd, "ab") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(directory)


@contextmanager
def controller(root, output, name="experiment"):
    global _controller_fds
    output = output_root(root, output)
    with parent_fd(output / f"{name}-controller.lock", create=True) as (directory, namefile):
        fd = os.open(namefile, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600, dir_fd=directory)
        previous = _controller_fds
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"another {name} controller owns this experiment") from error
            _controller_fds = previous + (fd,)
            yield output
        finally:
            _controller_fds = previous
            os.close(fd)


def child_env(output=None, **updates):
    semantic = [k for k in os.environ if k.startswith("JAIN_HYPERION_") and k != "JAIN_HYPERION_CUDA_MAX_LOAD_RATIO"]
    require(not semantic, "semantic Hyperion overrides rejected: " + ", ".join(sorted(semantic)))
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "RAYON_NUM_THREADS": "1"}
    if output is not None:
        env["JOPE_EXPERIMENT_DIR"] = str(output)
    env.update(updates)
    return env


_lease_fds = ()
_controller_fds = ()


def popen(command, **kwargs):
    """Inherit the lease and die if the launching controller disappears."""
    parent = os.getpid()
    limits = kwargs.pop("preexec_fn", None)
    def prepare():
        if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
            os._exit(127)
        if os.getppid() != parent:
            os._exit(127)
        if limits:
            limits()
    kwargs["pass_fds"] = tuple(set(kwargs.pop("pass_fds", ())) | set(_lease_fds) | set(_controller_fds))
    kwargs["preexec_fn"] = prepare
    kwargs["start_new_session"] = True
    return subprocess.Popen(command, **kwargs)


def run(command, *, root=None, timeout=60, **kwargs):
    """Synchronous owned work; callers with a lease still honor handoff."""
    import time
    children = []
    with owned_ranks(children):
        child = popen(command, **kwargs)
        children.append(child)
        deadline = time.monotonic() + timeout
        while True:
            if root is not None:
                from experiment_scheduler import admission_reason
                reason = admission_reason(root)
                if reason:
                    raise BlockingIOError(reason)
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(command, timeout)
            try:
                stdout, stderr = child.communicate(timeout=min(.2, max(.001, deadline - time.monotonic())))
                return subprocess.CompletedProcess(command, child.returncode, stdout, stderr)
            except subprocess.TimeoutExpired:
                pass


def stop(children):
    # Kill the group even if the leader has exited but a descendant remains.
    for child in children:
        # Never signal a different process that has reused a reaped leader's PID.
        current = Path(f"/proc/{child.pid}/stat")
        if child.returncode is not None and current.exists():
            continue
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for child in children:
        child.wait(timeout=10)


def retire(children, child):
    stop([child])
    if child in children:
        children.remove(child)


@contextmanager
def owned_ranks(children):
    try:
        yield
    finally:
        stop(children)


def supervise(root, output):
    """Re-enter CLI in a transient system service; never edit production units."""
    output = output_root(root, output)
    child_env()  # Reject semantic overrides before any resource admission.
    if os.environ.get("JOPE_EXPERIMENT_SERVICE") == "1":
        require("jope-experiment-" in Path("/proc/self/cgroup").read_text(), "experiment service cgroup missing")
        return
    output.mkdir(parents=True, exist_ok=True)
    locks = [checked_path(Path(root) / "runs" / name) for name in ("experiment-spare-cpus.lock", "encoding-service.lock")]
    require(all(p.is_file() for p in locks), "existing scheduler lock files are required")
    unit = "jope-experiment-" + uuid.uuid4().hex
    command = ["sudo", "-n", "systemd-run", "--quiet", "--wait", "--pipe", "--collect", "--unit=" + unit,
               "-p", "User=" + str(os.getuid()), "-p", "KillMode=control-group", "-p", "SendSIGKILL=yes",
               "-p", "TimeoutStopSec=10", "-p", "ProtectSystem=strict", "-p", "ProtectHome=read-only",
               "-p", "PrivateTmp=yes", "-p", "NoNewPrivileges=yes", "-p", "RestrictSUIDSGID=yes",
               "-p", "ReadOnlyPaths=" + str(Path(root).resolve()),
               "-p", "ReadWritePaths=" + " ".join(str(p) for p in [output, *locks]),
               "-p", "WorkingDirectory=" + str(Path(root).resolve()), "-p", "UMask=0077",
               "-p", "Nice=10", "-p", "AllowedCPUs=0-7 62-69", "--setenv=JOPE_EXPERIMENT_SERVICE=1", "--setenv=PYTHONDONTWRITEBYTECODE=1",
               sys.executable, str(Path(sys.argv[0]).resolve()), *sys.argv[1:]]
    raise SystemExit(subprocess.call(command))


def entrypoint(main, output_flag, name="experiment"):
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--root", type=Path)
    parser.add_argument(output_flag, type=Path, dest="output")
    args, _ = parser.parse_known_args()
    if "--help" in sys.argv or not args.root or not args.output:
        return main()
    output = inside(args.root, args.output)
    if output_flag == "--manifest":
        output = output.parent
    supervise(args.root, output)
    with controller(args.root, output, name):
        return main()
