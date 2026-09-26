#!/usr/bin/env python3
"""Check source custody and coverage; this does not qualify product behavior."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1] / "findings"
COUNTS = (8, 9, 10, 10, 8, 12, 10, 10, 10, 10)
PREFIXES = ("K", "I", "K", "I", "I", "K", "K", "K", "I", "K")
STAGES = {"source_implementation", "fixture_proof", "hosted_ci",
          "live_qualification", "promotion"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root=ROOT):
    record = json.loads((root / "register.json").read_text())
    require(record["schema"] == "jope.regression-findings/v1", "schema")
    require(re.fullmatch(r"[0-9a-f]{40}", record["baseline_source"]), "baseline SHA")
    require(set(record["packages"]) == {f"W{i}" for i in range(13)}, "packages")
    require(len(record["sources"]) == 10, "source count")
    source_entries = {}
    for shard, source in enumerate(record["sources"], 1):
        require(source["shard"] == shard, "source order/duplicate")
        require(source["path"] == f"sources/shard-{shard:02d}.md", "source path")
        path = root / source["path"]
        require(not path.is_symlink(), "source symlink")
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == source["sha256"], "source digest")
        text = raw.decode()
        headings = list(re.finditer(r"^#{1,3} .*$", text, re.M))
        ids = set()
        for index, heading in enumerate(headings):
            match = re.match(r"### ([GIK]\d+)\s", heading.group())
            if not match:
                continue
            original_id = match[1]
            key = f"S{shard:02d}-{original_id}"
            require(key not in source_entries, "duplicate source ID")
            ids.add((original_id[0], int(original_id[1:])))
            end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
            section = text[heading.start():end]
            fields = list(re.finditer(r"^(?:- )?\*\*([^*]+)\*\*", section, re.M))
            predicates = []
            for n, field in enumerate(fields):
                if re.search(
                    "detection|acceptance|closure|close with|completion evidence|"
                    "stop condition|stop / dependency|harden", field[1], re.I
                ):
                    stop = fields[n + 1].start() if n + 1 < len(fields) else len(section)
                    predicates.append(section[field.start():stop].strip())
            source_entries[key] = {
                "original_id": original_id, "shard": shard,
                "title": heading.group()[4:], "source_sha256": source["sha256"],
                "section_sha256": hashlib.sha256(section.encode()).hexdigest(),
                "source_lines": [text[:heading.start()].count("\n") + 1,
                                 text[:end].count("\n")],
                "kind": "gap" if original_id.startswith("G") else "improvement",
                "source_predicates": predicates,
            }
        count = COUNTS[shard - 1]
        expected = {("G", i) for i in range(1, count + 1)}
        expected |= {(PREFIXES[shard - 1], i)
                     for i in range(1, count + (2 if shard == 2 else 1))}
        require(ids == expected, f"shard {shard} coverage")
    entries = record["entries"]
    require(len(entries) == len(source_entries) == 195, "entry count")
    require({e["id"] for e in entries} == set(source_entries), "entry IDs")
    for entry in entries:
        key = entry["id"]
        for field, value in source_entries[key].items():
            actual = entry['closure_route'][field] if field == 'source_predicates' else entry[field]
            require(actual == value, f"{key}: {field}")
        package = record["packages"][entry["work_package"]]
        for field in ("subsystem", "receiving_boundary", "paths"):
            require(bool(package[field]), f"{key}: {field}")
        require(all(dep in record["packages"] for dep in package["dependencies"]), "dependency")
        require(bool(entry["scope"]), f"{key}: scope")
        route = entry["closure_route"]
        require(route["specification"] ==
                f"sources/shard-{entry['shard']:02d}.md#L{entry['source_lines'][0]}",
                f"{key}: specification")
        require(route["evidence_location"] == f"evidence/{key}.json", f"{key}: evidence path")
        require(bool(route["acceptance"]), f"{key}: acceptance")
        # V1 is an open-work inventory. Qualifying evidence needs a reviewed schema
        # extension, not free-form strings that silently upgrade this inventory.
        require(entry["disposition"] == ("unverified-source-finding" if entry["kind"] == "gap"
                                        else "proposed-improvement"), f"{key}: disposition")
        require(route["reproduction_command"] is None and
                route["reproduction_status"] == "not-yet-established-through-receiving-code",
                f"{key}: reproduction")
        require(route["evidence_status"] == "not-produced", f"{key}: evidence status")
        require(set(entry["attestations"]) == STAGES and
                all(value == [] for value in entry["attestations"].values()),
                f"{key}: unsupported attestation")
    return {"entries": len(entries), **Counter(e["kind"] for e in entries),
            "scope": "inventory integrity only; no product finding closed"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), sort_keys=True))
