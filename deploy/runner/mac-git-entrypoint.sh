#!/usr/bin/env bash
# Git-only emulation for the approved Mac Linux guest; .NET/Rust/Node retain Rosetta.
# Keep the reviewed image entrypoint and JIT stdin protocol. Only Git uses QEMU.
set -euo pipefail

jope_mac_git_tool() { # inspect-file|inspect-volume|provision-volume, reviewed policy, optional public input
  [[ $# == 3 ]] || return 2
  /usr/bin/python3 -I - "$@" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

action, config, supplied = sys.argv[1:]
policy = json.loads(Path(config).read_text())["git_emulation"]
keys = {"schema", "image", "version", "binary_sha256", "binary_bytes", "volume"}
if set(policy) != keys or policy["schema"] != "jope.mac-git/v1":
    raise ValueError("Unexpected Git-only emulation policy")
digest, size, volume = policy["binary_sha256"], policy["binary_bytes"], policy["volume"]
if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
    raise ValueError("Invalid Git emulator digest")
if type(size) is not int or not 20 <= size <= 64 * 1024**2:
    raise ValueError("Invalid Git emulator size")
if volume != "jope-git-emulation-" + digest[:12]:
    raise ValueError("Tool volume must be content addressed")
if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", policy["version"]):
    raise ValueError("Invalid Git emulator version")
if not re.fullmatch(r"tonistiigi/binfmt:qemu-v[0-9.]+-[0-9]+@sha256:[0-9a-f]{64}", policy["image"]):
    raise ValueError("Invalid public emulator image pin")

def checked_bytes(path, installed):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path or path.is_symlink() or not path.is_file():
        raise ValueError("Emulator must be a direct regular absolute file")
    info = path.stat()
    if info.st_size != size or info.st_nlink != 1 or info.st_mode & 0o022 or not os.access(path, os.X_OK):
        raise ValueError("Emulator size, links or permissions differ")
    if installed and (info.st_uid, info.st_gid, info.st_mode & 0o7777) != (0, 0, 0o555):
        raise ValueError("Installed emulator must be root:root 0555")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("Emulator hash differs from reviewed policy")
    if data[:6] != b"\x7fELF\x02\x01" or int.from_bytes(data[18:20], "little") != 183:
        raise ValueError("Emulator must be a native ARM64 ELF")
    return data

def checked_volume(root):
    if root.resolve() != root or root.is_symlink() or not root.is_dir():
        raise ValueError("Tool volume must be a direct directory")
    info = root.stat()
    if (info.st_uid, info.st_gid, info.st_mode & 0o7777) != (0, 0, 0o555):
        raise ValueError("Tool volume must be root:root 0555")
    if sorted(p.name for p in root.iterdir()) != ["qemu-x86_64"]:
        raise ValueError("Tool volume has unexpected contents; preserve it")
    checked_bytes(root / "qemu-x86_64", True)

def docker(*args):
    return subprocess.run(["/usr/bin/docker", *args], text=True, capture_output=True, timeout=30)

if action == "inspect-file":
    binary = Path(supplied)
    checked_bytes(binary, True)
    if not os.statvfs(binary).f_flag & os.ST_RDONLY:
        raise ValueError("Job emulator mount must be read-only")
elif action in ("inspect-volume", "provision-volume"):
    if os.geteuid() != 0:
        raise ValueError("Tool volume management requires root")
    # Validate and retain the public input bytes before the first possible mutation.
    payload = checked_bytes(supplied, False) if supplied else None
    result = docker("volume", "inspect", "--format", "{{.Mountpoint}}", volume)
    if result.returncode:
        if action != "provision-volume" or payload is None:
            raise ValueError("Verified tool volume is absent; supply the pinned public binary")
        # An unavailable daemon must never be interpreted as a missing volume.
        probe = docker("volume", "ls", "--format", "{{.Name}}")
        if probe.returncode or volume in probe.stdout.splitlines():
            raise ValueError("Cannot establish tool-volume absence")
        created = docker("volume", "create", volume)
        if created.returncode or created.stdout.strip() != volume:
            raise ValueError("Tool-volume creation outcome uncertain; inspect before retry")
        result = docker("volume", "inspect", "--format", "{{.Mountpoint}}", volume)
        if result.returncode:
            raise ValueError("Cannot inspect newly created tool volume")
        root = Path(result.stdout.strip())
        if not root.is_absolute() or root.resolve() != root or root.is_symlink() or list(root.iterdir()):
            raise ValueError("New tool volume is not an empty direct directory; preserve it")
        if (root.stat().st_uid, root.stat().st_gid) != (0, 0):
            raise ValueError("New tool volume has unexpected ownership")
        fd = os.open(root / "qemu-x86_64", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o555)
        try:
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("Short emulator write")
                view = view[written:]
            os.fchown(fd, 0, 0)
            os.fchmod(fd, 0o555)
            os.fsync(fd)
        finally:
            os.close(fd)
        root.chmod(0o555)
        directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    root = Path(result.stdout.strip())
    checked_volume(root)
    binary = root / "qemu-x86_64"
else:
    raise ValueError("Unsupported Git tool operation")

version = subprocess.check_output([str(binary), "--version"], text=True, timeout=10).splitlines()[0]
if not version.startswith("qemu-x86_64 version " + policy["version"] + " "):
    raise ValueError("Pinned emulator reports an unexpected version")
print(json.dumps({"binary_sha256": digest, "binary_bytes": size, "architecture": "aarch64",
                  "uid": 0, "gid": 0, "mode": "0555", "volume": volume,
                  "version": policy["version"]}, sort_keys=True))
PY
}

jope_mac_git_prepare() {
  [[ $# == 3 ]] || return 2
  local emulator=$1 expected=$2 shim_root=$3
  [[ $expected =~ ^[0-9a-f]{64}$ && -n $PATH ]] || return 2
  # The heredoc is the child's stdin; the caller's opaque JIT stream stays unread.
  if ! /usr/bin/python3 -I - "$emulator" "$expected" "$shim_root" <<'PY'
import hashlib
import os
from pathlib import Path
import re
import shlex
import sys

emulator, expected, destination = sys.argv[1:]
tool = Path(emulator)
root = Path(destination)
if not tool.is_absolute() or tool.is_symlink() or not tool.is_file():
    raise ValueError("Git emulator must be a regular absolute file")
if tool.resolve() != tool or not os.access(tool, os.X_OK):
    raise ValueError("Git emulator path must be direct and executable")
if tool.stat().st_mode & 0o022:
    raise ValueError("Git emulator cannot be group or world writable")
data = tool.read_bytes()
if hashlib.sha256(data).hexdigest() != expected:
    raise ValueError("Git emulator digest differs from reviewed policy")
if data[:6] != b"\x7fELF\x02\x01" or int.from_bytes(data[18:20], "little") != 183:
    raise ValueError("Git emulator must be a native ARM64 ELF")
if not root.is_absolute() or root.resolve() != root or root.exists() or root.is_symlink():
    raise ValueError("Git shim directory must be a new direct absolute path")

git = Path("/usr/bin/git")
core = Path("/usr/lib/git-core")
if not git.is_file() or not os.access(git, os.X_OK) or not core.is_dir():
    raise ValueError("Reviewed image Git installation is unavailable")

root.mkdir(mode=0o700)
binary = root / "bin"
helpers = root / "libexec"
binary.mkdir(mode=0o700)
helpers.mkdir(mode=0o700)

def wrap(target, output):
    # The pinned tonistiigi binary expects the preserved original argv[0].
    # All caller arguments remain separate words; neither eval nor a shell reparse.
    line = ("#!/bin/sh\nexec " + shlex.quote(str(tool))
            + " -0 " + shlex.quote(target.name)
            + " " + shlex.quote(str(target))
            + " " + shlex.quote(target.name) + ' "$@"\n')
    output.write_text(line)
    output.chmod(0o700)

wrap(git, binary / "git")
count = 0
for target in sorted(core.iterdir()):
    if not target.is_file():
        continue
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]*", target.name):
        raise ValueError("Unexpected Git helper name in reviewed image")
    with target.open("rb") as stream:
        header = stream.read(20)
    if header[:4] == b"\x7fELF":
        if header[4:6] != b"\x02\x01" or int.from_bytes(header[18:20], "little") != 62:
            raise ValueError("Unexpected Git helper architecture")
        wrap(target, helpers / target.name)
        count += 1
    else:
        (helpers / target.name).symlink_to(target)
if count == 0:
    raise ValueError("No image Git ELF helpers were wrapped")
PY
  then
    return 2
  fi
  export PATH="$shim_root/bin:$PATH"
  export GIT_EXEC_PATH="$shim_root/libexec"
}

jope_mac_git_main() {
  [[ $# == 0 && ${JOPE_RUNNER_HOST:-} == xbabe4 ]] || return 2
  local expected
  jope_mac_git_tool inspect-file /opt/jope-runner/mac-capacity.json \
    /opt/jope-git-emulation/qemu-x86_64 >/dev/null || return 2
  expected=$(/usr/bin/jq -er '.git_emulation.binary_sha256
    | select(type == "string" and test("^[0-9a-f]{64}$"))' \
    /opt/jope-runner/mac-capacity.json) || return 2
  jope_mac_git_prepare /opt/jope-git-emulation/qemu-x86_64 "$expected" \
    /home/runner/git-emulation || return 2
  exec /usr/local/bin/jope-runner-entrypoint
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  case ${1:-} in
    --inspect-tool) [[ $# == 3 ]] && jope_mac_git_tool inspect-file "$2" "$3" ;;
    --inspect-volume) [[ $# == 2 ]] && jope_mac_git_tool inspect-volume "$2" '' ;;
    --provision-tool) [[ $# == 3 ]] && jope_mac_git_tool provision-volume "$2" "$3" ;;
    *) jope_mac_git_main "$@" ;;
  esac
fi
