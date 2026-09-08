#!/usr/bin/env python3
"""Four disjoint, manifest-driven streams; reuse matching remote originals."""
import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import subprocess
import tempfile


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--identity", required=True)
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    manifest_path = root / "manifests/generated/review-v3-transfer.json"
    manifest = json.loads(manifest_path.read_text())
    ssh = ["rtk", "proxy", "ssh", "-i", args.identity, "-o", "BatchMode=yes", args.host, "python3 -"]

    def remote(script):
        result = subprocess.run(ssh, input=script, text=True, capture_output=True, check=True)
        return json.loads(result.stdout)

    # A destination is immutable. Existing content is reused only after SHA256
    # verification; mismatching files are never overwritten by rsync.
    preflight = """import pathlib,hashlib,json,os
root=pathlib.Path(ROOT)
files=FILES
missing=[];reused=[]
for f in files:
 p=root/f['destination']
 assert not p.is_symlink(),str(p)
 if p.exists():
  assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256'],str(p)
  reused.append(f['destination'])
 else:
  p.parent.mkdir(parents=True,exist_ok=True)
  missing.append(f['destination'])
print(json.dumps({'missing':missing,'reused':reused}))
""".replace("ROOT", repr(str(root))).replace("FILES", repr(manifest["files"]))
    state = remote(preflight)
    missing = set(state["missing"])
    streams = []
    for dataset in sorted({f["dataset"] for f in manifest["files"]}):
        rows = [f for f in manifest["files"] if f["dataset"] == dataset and f["role"] != "real_backing_source" and f["destination"] in missing]
        if rows:
            source = pathlib.Path(rows[0]["source"])
            while source.name != dataset:
                source = source.parent
            streams.append((source, pathlib.Path("data/originals/review-v3/regression") / dataset, rows))
    backing = [f for f in manifest["files"] if f["role"] == "real_backing_source" and f["destination"] in missing]
    if backing:
        streams.append((pathlib.Path("/home/ubuntu"), pathlib.Path("data/originals/backing"), backing))
    assert len(streams) <= 4

    def transfer(stream):
        source, destination, rows = stream
        with tempfile.NamedTemporaryFile(mode="w", prefix="review-v3-rsync-", delete=True) as listing:
            for row in rows:
                path = pathlib.Path(row["source"])
                assert sha(path) == row["sha256"]
                listing.write(str(path.relative_to(source)) + "\n")
            listing.flush()
            command = ["rtk", "proxy", "rsync", "-a", "--ignore-existing", "--partial", "--partial-dir=.rsync-partial",
                       "--files-from", listing.name, "-e", f"ssh -i {args.identity} -o BatchMode=yes",
                       str(source) + "/", f"{args.host}:{root / destination}/"]
            subprocess.run(command, check=True)
        return len(rows)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        transferred = sum(pool.map(transfer, streams))
    checked = remote(preflight)
    assert not checked["missing"]
    receipt = dict(schema="parallel.reviewed_source_staging.v1", passed=True, host=args.host,
                   manifest_sha256=sha(manifest_path), verified_files=len(checked["reused"]),
                   transferred_files=transferred, reused_files=len(state["reused"]), streams=len(streams),
                   transferred_bytes=sum(f["bytes"] for f in manifest["files"] if f["destination"] in missing),
                   copied_cut_csv_files=0, original_final_test_targets_parsed=False)
    output = root / "manifests/generated/review-v3-node-staging.json"
    with output.open("x") as target:
        json.dump(receipt, target, indent=2)
        target.write("\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
