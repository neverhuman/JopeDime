#!/usr/bin/env python3
"""Verify pinned DL source custody; this command grants no runtime authority."""
import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

HANDOFF_SHA = "009532038cee11ea20075a7291e4b247c7df8e97806946a61bd8dce7dd635802"
PREDECESSOR_SHA = "4b8c54da7e4b11cc2f30b02aa88f159d5240134bf4962e670d2bfcddbcd699ad"
SUPPLEMENT_NAME = "test_v2_rowsel_raw.py.2ae3cf61cdbe29e15ecaa642c4f5e5edc78db32211f4337083ad943afc550f09"
HEX = re.compile(r"[0-9a-f]{64}\Z")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def read_regular(root, name):
    rel = PurePosixPath(name)
    if rel.is_absolute() or not rel.parts or str(rel) != name or any(x in (".", "..") for x in rel.parts):
        raise ValueError("unsafe source path: " + name)
    path = root
    for part in rel.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("symlink in source path: " + name)
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("missing/escaping source file: " + name)
    return path.read_bytes()


def pinned_json(root, name, expected):
    data = read_regular(root, name)
    if not HEX.fullmatch(expected) or sha(data) != expected:
        raise ValueError("trusted manifest pin mismatch: " + name)
    return json.loads(data, object_pairs_hook=unique)


def verify(root, handoff_sha=HANDOFF_SHA, predecessor_sha=PREDECESSOR_SHA, supplement_name=SUPPLEMENT_NAME):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("invalid custody root")
    current = pinned_json(root, "MANIFEST.json", handoff_sha)
    prior = pinned_json(root, "predecessor/MANIFEST.json", predecessor_sha)
    if current.get("schema") != "jope.dl-lane-handoff/v2" or prior.get("schema") != "jope.dl-lane-handoff/v1":
        raise ValueError("unsupported custody schema")
    if current.get("supersedes_sha256") != predecessor_sha:
        raise ValueError("predecessor identity mismatch")
    records = {}
    for name, record in current["files"].items():
        if not name.startswith("src/dl_training_v1/"):
            raise ValueError("current source outside declared tree")
        records[name] = record
    for name, record in current["historical_versions"].items():
        if "/" in name or name.rsplit(".", 1)[-1] != record["sha256"]:
            raise ValueError("historical filename identity mismatch")
        records["historical/" + name] = record
    captured = {}
    for name, record in records.items():
        expected, size = record["sha256"], record["bytes"]
        if not isinstance(expected, str) or not HEX.fullmatch(expected) or type(size) is not int or size < 0:
            raise ValueError("invalid source binding: " + name)
        data = read_regular(root, name)
        if len(data) != size or sha(data) != expected:
            raise ValueError("source bytes mismatch: " + name)
        captured[name] = data
    discovered = set()
    for directory in ("src", "historical"):
        for path in (root / directory).rglob("*"):
            if path.is_symlink():
                raise ValueError("unlisted source symlink")
            if path.is_file():
                discovered.add(path.relative_to(root).as_posix())
    if discovered != set(records):
        raise ValueError("source inventory mismatch")
    supplement = read_regular(root, "predecessor/" + supplement_name)
    supplement_sha = supplement_name.rsplit(".", 1)[-1]
    if not HEX.fullmatch(supplement_sha) or sha(supplement) != supplement_sha:
        raise ValueError("supplemental predecessor bytes mismatch")
    if any(p.is_symlink() for p in (root / "predecessor").rglob("*")):
        raise ValueError("predecessor symlink")
    predecessor_paths = {"predecessor/MANIFEST.json", "predecessor/" + supplement_name}
    if {p.relative_to(root).as_posix() for p in (root / "predecessor").rglob("*") if p.is_file()} != predecessor_paths:
        raise ValueError("predecessor inventory mismatch")
    available = {sha(data): name for name, data in captured.items()}
    available[supplement_sha] = "predecessor/" + supplement_name
    reconstruction = {}
    for name, record in prior["files"].items():
        if record["sha256"] not in available:
            raise ValueError("predecessor version lost: " + name)
        reconstruction[name] = available[record["sha256"]]
    for name, record in prior["historical_versions"].items():
        retained = current["historical_versions"].get(name, {})
        if any(retained.get(field) != record[field] for field in ("sha256", "bytes")):
            raise ValueError("historical predecessor version lost: " + name)
    python_files = [name for name in captured if name.endswith(".py")]
    for name in python_files:
        ast.parse(captured[name], filename=name)
    return {"schema": "jope.dl-source-custody/v1", "handoff_sha256": handoff_sha,
            "predecessor_sha256": predecessor_sha, "current_files": len(current["files"]),
            "historical_versions": len(current["historical_versions"]), "supplemental_versions": 1,
            "python_syntax_files": len(python_files), "predecessor_restoration": reconstruction,
            "scope": "exact source custody and syntax only; no imported code, execution, runtime or training authority"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).with_name("lane"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.root)
    if args.output:
        with args.output.open("x") as out:
            json.dump(report, out, indent=2)
            out.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "predecessor_restoration"}))


if __name__ == "__main__":
    main()
