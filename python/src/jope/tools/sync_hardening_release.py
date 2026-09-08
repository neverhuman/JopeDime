"""Sync only a release inventory to a fresh remote staging directory."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tempfile

from experiment_runtime import require


def sync(release, host, key, remote_root):
    manifest = json.loads((release / "release-manifest.json").read_text())
    require(release.name == "hardening-v2-" + manifest["source_sha256"], "invalid release name")
    files = manifest["files"]
    for name, expected in files.items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "unsafe inventory path")
        require(hashlib.sha256((release / name).read_bytes()).hexdigest() == expected, f"local hash mismatch: {name}")
    require(hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == manifest["source_sha256"], "source hash mismatch")
    staging = remote_root + "/builds/." + release.name + ".staging"
    final = remote_root + "/builds/" + release.name
    ssh = ["ssh", "-o", "BatchMode=yes", "-i", str(key), host]
    def remote(code, *args):
        return subprocess.run(ssh + [shlex.join(["python3", "-", *args])], input=code, text=True,
                              capture_output=True, check=True).stdout
    remote("import pathlib,sys\ns,f=map(pathlib.Path,sys.argv[1:]);\nif s.exists() or f.exists(): raise SystemExit('release/staging exists; inspect before retry')\ns.mkdir()\n", staging, final)
    with tempfile.NamedTemporaryFile(mode="w") as inventory:
        inventory.write("\n".join([*files, "release-manifest.json"]) + "\n")
        inventory.flush()
        subprocess.run(["rsync", "-a", "--files-from=" + inventory.name, "-e", shlex.join(ssh[:-1]),
                        str(release) + "/", host + ":" + shlex.quote(staging + "/")], check=True)
    verification = remote('''import pathlib,sys,json,hashlib,os
s,f=map(pathlib.Path,sys.argv[1:3]); manifest=s/'release-manifest.json'
if hashlib.sha256(manifest.read_bytes()).hexdigest()!=sys.argv[3]: raise SystemExit('remote release manifest differs from local')
m=json.loads(manifest.read_text())
if hashlib.sha256(json.dumps(m['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()!=m['source_sha256']: raise SystemExit('remote source identity mismatch')
actual={str(p.relative_to(s)) for p in s.rglob('*') if p.is_file()}
if actual != set(m['files'])|{'release-manifest.json'}: raise SystemExit('remote inventory mismatch')
for name,h in m['files'].items():
 p=s/name
 if p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest()!=h: raise SystemExit('remote hash mismatch: '+name)
 with p.open('rb') as stream: os.fsync(stream.fileno())
with manifest.open('rb') as stream: os.fsync(stream.fileno())
if f.exists(): raise SystemExit('published release already exists')
os.rename(s,f)
fd=os.open(f.parent,os.O_RDONLY); os.fsync(fd); os.close(fd)
for p in f.rglob('*'):
 if p.is_dir(): p.chmod(0o555)
f.chmod(0o555)
print(json.dumps({'release':str(f),'source_sha256':m['source_sha256'],'verified_files':len(m['files']),'published_atomically':True}))
''', staging, final, hashlib.sha256((release / "release-manifest.json").read_bytes()).hexdigest())
    return json.loads(verification)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--host", default="ubuntu@158.101.20.226")
    parser.add_argument("--key", type=Path, default=Path.home() / ".ssh/lambda_cloud_a100_20260907_ed25519")
    parser.add_argument("--remote-root", default="/home/ubuntu/jope-dime/parallel")
    args = parser.parse_args()
    print(json.dumps(sync(args.release.resolve(), args.host, args.key, args.remote_root), indent=2))
