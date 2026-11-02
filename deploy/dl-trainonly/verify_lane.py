#!/usr/bin/env python3
"""Verify pinned DL source custody; this command grants no runtime authority."""
import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

HANDOFF_SHA = "9afa6c81dc7e4ce0df2d45b76b7d871fe0ea5f660c932f4bb1e6f4553ca49ef7"
# Ordered newest to oldest: each manifest's supersedes_sha256 names the next pin; the oldest names none.
PREDECESSORS = (
    ("predecessor/source-closure-correction-20261102/MANIFEST.json", "cc9609693ada637ed1cc3308c514ecb52de9d35aa8ed1d45a17103c432a76414"),
    ("predecessor/root-pr251-resume-20261007T1640Z/MANIFEST.json", "e88de3c63c574a133b0742f7c179a217f26f2dee5cffc14f7b82189c63a06143"),
    ("predecessor/restage-20261007T080755Z/MANIFEST.json", "f7caed54a6d78a1a7da92b61db3c47bc79e86e15d5ce50e1228aa6396a747f43"),
    ("predecessor/restage-20261007T054519Z/MANIFEST.json", "fb2140bc50deb66057f009f42cf6690cca20e3862e8f552199e292068d9ba754"),
    ("predecessor/restage-20261007T034806Z/MANIFEST.json", "985cb6288a26565ae22e331f92f1d28aeb459dd2c72e31fa26a60b6e8f96dacc"),
    ("predecessor/restage-20261006T153320Z/MANIFEST.json", "4419ff430ec81a079829e67b935da8b16633e238c76887bd209c6ca9c4d0712d"),
    ("predecessor/restage-20261006T131001Z/MANIFEST.json", "23d3a8a7fd6d15f056e6d5465a9e524ae46f5b69fd9d0e6c6f4721251ce70413"),
    ("predecessor/restage-20261006T114844Z/MANIFEST.json", "253e593b7537cdecf75a89be24a83804bc63d7e2596e02cc0c28d780a50f2b9e"),
    ("predecessor/restage-20261006T082918Z/MANIFEST.json", "365ec3cecab4e647a160c1f5c2a6baec1f45c5ea102e3180922173ed463f628f"),
    ("predecessor/restage-20261006T073946Z/MANIFEST.json", "6314ef6f33f01878a149cdb36823a78c96d6ae3e2c45cf409d756711c71d6619"),
    ("predecessor/restage-20261006T035515Z/MANIFEST.json", "009532038cee11ea20075a7291e4b247c7df8e97806946a61bd8dce7dd635802"),
    ("predecessor/MANIFEST.json", "4b8c54da7e4b11cc2f30b02aa88f159d5240134bf4962e670d2bfcddbcd699ad"),
)
# Predecessor current versions that no later manifest retains, preserved byte for byte.
SUPPLEMENTS = (
    "predecessor/test_v2_rowsel_raw.py.2ae3cf61cdbe29e15ecaa642c4f5e5edc78db32211f4337083ad943afc550f09",
    "predecessor/restage-20261006T035515Z/test_v2_publish.py.bbb8e5deb2e68846022c25de2971f9fc6ce22845c7d49ca1cb8a0922a4ad0780",
    "predecessor/restage-20261006T035515Z/test_v2_rowsel_raw.py.f33f3d2b270e533060374d4a6a442d5c6010fda727f062ac3758b9bf9ea9ace7",
)
# Stagings a producer manifest records as abandoned and never pushed; they must never be a chain member.
ABANDONED = "abandoned_unpushed_stagings"
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


def verify(root, handoff_sha=HANDOFF_SHA, predecessors=PREDECESSORS, supplements=SUPPLEMENTS):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("invalid custody root")
    current = pinned_json(root, "MANIFEST.json", handoff_sha)
    if not predecessors:
        raise ValueError("predecessor chain missing")
    chain = [(name, expected, pinned_json(root, name, expected)) for name, expected in predecessors]
    schemas = [current.get("schema")] + [manifest.get("schema") for _, _, manifest in chain]
    if any(s != "jope.dl-lane-handoff/v2" for s in schemas[:-1]) or schemas[-1] != "jope.dl-lane-handoff/v1":
        raise ValueError("unsupported custody schema")
    newer = current
    for _, expected, manifest in chain:
        if newer.get("supersedes_sha256") != expected:
            raise ValueError("predecessor identity mismatch")
        newer = manifest
    if "supersedes_sha256" in newer:
        raise ValueError("predecessor chain incomplete")
    abandoned = {}
    for manifest in [current] + [manifest for _, _, manifest in chain]:
        listed = manifest.get(ABANDONED, [])
        if not isinstance(listed, list):
            raise ValueError("invalid abandoned staging record")
        for record in listed:
            if (not isinstance(record, dict) or not isinstance(record.get("manifest"), str)
                    or not isinstance(record.get("manifest_sha256"), str) or not HEX.fullmatch(record["manifest_sha256"])):
                raise ValueError("invalid abandoned staging record")
            abandoned[record["manifest_sha256"]] = record["manifest"]
    members = {handoff_sha} | {expected for _, expected in predecessors}
    if set(abandoned) & members or {"predecessor/" + name for name in abandoned.values()} & {name for name, _ in predecessors}:
        raise ValueError("abandoned staging inside custody chain")
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
                relative = path.relative_to(root)
                # Python may leave ignored interpreter caches beside an otherwise
                # exact custody tree. They are not source bytes and are never
                # admitted as manifest entries; keep rejecting every other extra.
                if "__pycache__" in relative.parts and path.suffix in {".pyc", ".pyo"}:
                    continue
                discovered.add(relative.as_posix())
    if discovered != set(records):
        raise ValueError("source inventory mismatch")
    available = {sha(data): name for name, data in captured.items()}
    for name in supplements:
        if not name.startswith("predecessor/"):
            raise ValueError("supplemental predecessor outside custody")
        supplement = read_regular(root, name)
        supplement_sha = name.rsplit(".", 1)[-1]
        if not HEX.fullmatch(supplement_sha) or sha(supplement) != supplement_sha:
            raise ValueError("supplemental predecessor bytes mismatch")
        if supplement_sha in available:
            raise ValueError("redundant supplemental predecessor")
        available[supplement_sha] = name
    if any(p.is_symlink() for p in (root / "predecessor").rglob("*")):
        raise ValueError("predecessor symlink")
    predecessor_paths = {name for name, _, _ in chain} | set(supplements)
    if {p.relative_to(root).as_posix() for p in (root / "predecessor").rglob("*") if p.is_file()} != predecessor_paths:
        raise ValueError("predecessor inventory mismatch")
    reconstruction = {}
    for _, expected, prior in chain:
        restored = reconstruction[expected] = {}
        for name, record in prior["files"].items():
            if record["sha256"] not in available:
                raise ValueError("predecessor version lost: " + name)
            restored[name] = available[record["sha256"]]
        for name, record in prior["historical_versions"].items():
            retained = current["historical_versions"].get(name, {})
            if any(retained.get(field) != record[field] for field in ("sha256", "bytes")):
                raise ValueError("historical predecessor version lost: " + name)
    referenced = {record["sha256"] for _, _, prior in chain for record in prior["files"].values()}
    if any(name.rsplit(".", 1)[-1] not in referenced for name in supplements):
        raise ValueError("unreferenced supplemental predecessor")
    python_files = [name for name in captured if name.endswith(".py")]
    for name in python_files:
        ast.parse(captured[name], filename=name)
    return {"schema": "jope.dl-source-custody/v2", "handoff_sha256": handoff_sha,
            "predecessor_sha256": [expected for _, expected in predecessors], "current_files": len(current["files"]),
            "historical_versions": len(current["historical_versions"]), "supplemental_versions": len(supplements),
            "abandoned_unpushed_stagings": sorted(abandoned),
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
