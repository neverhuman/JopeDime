"""H01: head shares from existing observations. No new fits. No invented core-seconds."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

SCHEMA = "parallel.head_cost_census.v1"
HEAD = re.compile(r"\(([A-Za-z][A-Za-z0-9_]*)")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def program_head(genome):
    names = HEAD.findall(genome or "")
    return names[-1] if names else "unknown"


def is_timeout(row):
    failure = row.get("failure")
    if not failure:
        return False
    text = str(failure).lower()
    return "timeout" in text or "deadline" in text


def freeze_paths(roots):
    paths = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        paths.extend(sorted(p for p in root.rglob("observations.json") if p.is_file()))
    return [str(p) for p in paths]


def scan(paths):
    heads = Counter()
    timeouts = Counter()
    programs = {}
    walls = Counter()
    wall_n = Counter()
    files = 0
    rows = 0
    unread = []
    for raw in paths:
        path = Path(raw)
        try:
            data = json.loads(path.read_bytes())
        except (OSError, ValueError, UnicodeError):
            unread.append(str(path))
            continue
        if not isinstance(data, list):
            unread.append(str(path))
            continue
        files += 1
        for row in data:
            if not isinstance(row, dict):
                continue
            rows += 1
            head = program_head(row.get("genome"))
            heads[head] += 1
            programs.setdefault(head, set()).add(row.get("program_sha256") or row.get("genome"))
            if is_timeout(row):
                timeouts[head] += 1
            seconds = row.get("runtime_seconds")
            if seconds is None and isinstance(row.get("metrics"), dict):
                seconds = row["metrics"].get("seconds")
            if isinstance(seconds, (int, float)):
                walls[head] += float(seconds)
                wall_n[head] += 1
    shares = []
    total = sum(heads.values()) or 1
    for head, count in heads.most_common():
        shares.append(dict(
            head=head,
            observations=count,
            observation_share=count / total,
            unique_programs=len(programs.get(head, ())),
            timeouts=timeouts[head],
            wall_seconds_sum=walls[head] if wall_n[head] else None,
            wall_rows=wall_n[head],
            core_seconds=None,
        ))
    return dict(files=files, observations=rows, unread=unread, shares=shares,
                observation_total=sum(heads.values()))


def census(roots, output):
    declared = freeze_paths(roots)
    result = scan(declared)
    receipt = dict(
        schema=SCHEMA,
        declared_roots=[str(Path(r)) for r in roots],
        declared_paths=declared,
        declared_before_scan=True,
        new_fits=0,
        core_seconds_invented=False,
        unverified=not declared,
        **result,
    )
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    target.write_text(payload)
    receipt["receipt_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    return receipt
