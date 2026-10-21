#!/usr/bin/env python3
"""Create/verify exact-run rendered UI evidence; no network, Git or browser execution."""
import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import struct
import sys
import xml.etree.ElementTree as ET
import zlib

SCHEMA = "jope.rendered-ux-ci/v2"
RECEIPT = "rendered-receipt.json"
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024
MAX_FILES = 10000


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def document(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            need(key not in out, "duplicate JSON field")
            out[key] = value
        return out

    value = json.loads(raw, object_pairs_hook=pairs)
    # Also reject parsed Infinity/NaN and overflowing JSON numbers.
    json.dumps(value, allow_nan=False)
    return value


def read_file(path):
    need(path.is_file() and not path.is_symlink(), "evidence must be a regular non-symlink file")
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        need(before.st_size <= MAX_FILE_BYTES, "evidence file exceeds byte bound")
        raw = handle.read(MAX_FILE_BYTES + 1)
        after = os.fstat(handle.fileno())
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    need(len(raw) == before.st_size and identity(before) == identity(after)
         == identity(path.stat(follow_symlinks=False)), "evidence changed during read")
    return raw


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def geometry(raw):
    value = document(raw)
    need(isinstance(value, dict), "geometry object required")
    width, height, total = (value.get(k) for k in ("viewport", "viewportHeight", "documentWidth"))
    need(all(finite(n) and n > 0 for n in (width, height, total)) and total <= width + 1,
         "invalid viewport or overflowing document")
    for key in ("main", "heading", "navigation"):
        box = value.get(key)
        need(isinstance(box, dict) and all(finite(box.get(k)) for k in ("x", "y", "width", "height", "right")),
             "missing or nonfinite landmark geometry")
        need(box["width"] > 0 and box["height"] > 0 and box["x"] >= -1
             and box["right"] <= width + 1 and abs(box["x"] + box["width"] - box["right"]) <= 1,
             "invalid or clipped landmark geometry")
    panels = value.get("panels")
    need(isinstance(panels, list), "panel geometry required")
    for panel in panels:
        need(isinstance(panel, dict) and all(finite(panel.get(k)) for k in ("x", "right", "width", "height")),
             "nonfinite panel geometry")
        need(panel["width"] > 0 and panel["height"] > 0 and panel["x"] >= -1
             and panel["right"] <= width + 1 and abs(panel["x"] + panel["width"] - panel["right"]) <= 1,
             "invalid or clipped panel geometry")
    dialog = value.get("dialog")
    if dialog is not None:
        need(isinstance(dialog, dict) and all(finite(dialog.get(k)) for k in ("x", "y", "right", "bottom")),
             "invalid dialog geometry")
        need(0 <= dialog["x"] < dialog["right"] <= width
             and 0 <= dialog["y"] < dialog["bottom"] <= height, "dialog exceeds viewport")


def accessibility(raw):
    value = document(raw)
    need(isinstance(value, dict) and isinstance(value.get("testEngine"), dict)
         and value["testEngine"].get("name") == "axe-core"
         and isinstance(value["testEngine"].get("version"), str)
         and value.get("violations") == [] and isinstance(value.get("passes"), list)
         and len(value["passes"]) > 0, "missing axe results or accessibility violations")


def screenshot(raw):
    need(raw.startswith(b"\x89PNG\r\n\x1a\n"), "screenshot must be PNG")
    offset, kinds = 8, []
    while offset < len(raw):
        need(offset + 12 <= len(raw), "truncated PNG chunk")
        size = struct.unpack(">I", raw[offset:offset + 4])[0]
        kind = raw[offset + 4:offset + 8]
        end = offset + 12 + size
        need(end <= len(raw), "truncated PNG data")
        data = raw[offset + 8:offset + 8 + size]
        crc = struct.unpack(">I", raw[offset + 8 + size:end])[0]
        need(zlib.crc32(kind + data) & 0xffffffff == crc, "PNG checksum mismatch")
        kinds.append(kind)
        if kind == b"IHDR":
            need(len(kinds) == 1 and size == 13, "invalid PNG header")
            need(all(n > 0 for n in struct.unpack(">II", data[:8])), "empty screenshot")
        if kind == b"IEND":
            need(size == 0 and end == len(raw), "invalid PNG end")
        offset = end
    need(kinds and kinds[0] == b"IHDR" and kinds[-1] == b"IEND" and b"IDAT" in kinds,
         "incomplete PNG screenshot")


def attachment_bytes(item, root):
    need(isinstance(item, dict) and isinstance(item.get("name"), str)
         and 0 < len(item["name"]) <= 200, "attachment name required")
    if "body" in item:
        need("path" not in item and isinstance(item["body"], str), "ambiguous attachment storage")
        need(len(item["body"]) <= MAX_FILE_BYTES * 4 // 3 + 8, "attachment exceeds byte bound")
        return base64.b64decode(item["body"], validate=True)
    path = PurePosixPath(item.get("path", ""))
    need(str(path) not in ("", ".") and not path.is_absolute() and ".." not in path.parts,
         "attachment paths must be relative to evidence root or embedded")
    target = root.joinpath(*path.parts)
    need(target.resolve().is_relative_to(root)
         and all(not parent.is_symlink() for parent in (target, *target.parents) if parent.is_relative_to(root)),
         "attachment escapes evidence root")
    return read_file(target)


def specs(suites):
    need(isinstance(suites, list), "suite list required")
    for suite in suites:
        need(isinstance(suite, dict), "suite object required")
        yield from suite.get("specs", [])
        yield from specs(suite.get("suites", []))


def artifact_inventory(root):
    paths = [root / "report.json", root / "junit.xml"]
    for name in ("html", "results"):
        directory = root / name
        need(directory.is_dir() and not directory.is_symlink(), "HTML report and result directories required")
        for path in directory.rglob("*"):
            need(not path.is_symlink(), "symlink in evidence tree")
            if path.is_file():
                need(len(paths) < MAX_FILES, "too many evidence files")
                paths.append(path)
    need((root / "html/index.html").is_file(), "portable HTML report required")
    need(len(paths) <= MAX_FILES, "too many evidence files")
    entries, total = [], 0
    for path in sorted(paths):
        raw = read_file(path)
        total += len(raw)
        need(total <= MAX_TOTAL_BYTES, "evidence exceeds total byte bound")
        entries.append({"path": path.relative_to(root).as_posix(), "bytes": len(raw), "sha256": digest(raw)})
    return entries


def report_identity(config, source_sha, run_id, attempt, checkout_sha=None, event_sha=None):
    """Check caller-bound identities; Playwright annotations never replace core metadata."""
    checkout_sha = source_sha if checkout_sha is None else checkout_sha
    event_sha = source_sha if event_sha is None else event_sha
    for value in (source_sha, checkout_sha, event_sha):
        need(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None,
             "full source, checkout and event SHAs required")
    need(checkout_sha == source_sha, "browser checkout must equal tested source")
    need(re.fullmatch(r"[1-9][0-9]*", run_id) is not None
         and re.fullmatch(r"[1-9][0-9]*", attempt) is not None, "positive run/attempt required")
    core = {"source_sha": source_sha, "run_id": run_id, "attempt": attempt}
    need(isinstance(config, dict), "report configuration required")
    metadata = config.get("metadata")
    need(isinstance(metadata, dict) and all(metadata.get(k) == v for k, v in core.items())
         and set(metadata) <= set(core) | {"actualWorkers", "ci", "gitCommit", "gitDiff"},
         "stale, missing or unknown report source/run/attempt metadata")
    # These are known Playwright annotations, validated separately from provenance.
    if "workers" in config:
        need(type(config["workers"]) is int and config["workers"] > 0,
             "invalid configured browser worker count")
    if "actualWorkers" in metadata:
        need(type(metadata["actualWorkers"]) is int and metadata["actualWorkers"] > 0
             and ("workers" not in config or metadata["actualWorkers"] <= config["workers"]),
             "invalid actual browser worker count")

    def text(value, maximum):
        return isinstance(value, str) and len(value) <= maximum and "\x00" not in value

    repository = "https://github.com/neverhuman/JopeDime"
    if "ci" in metadata:
        ci = metadata["ci"]
        need(isinstance(ci, dict) and set(ci) <= {
            "commitHref", "commitHash", "buildHref", "prHref", "prTitle", "prBaseHash", "branch"},
            "unknown or invalid Playwright CI annotations")
        need(ci.get("commitHash") == event_sha
             and ci.get("commitHref") == f"{repository}/commit/{event_sha}"
             and ci.get("buildHref") in (f"{repository}/actions/runs/{run_id}",
                                        f"{repository}/actions/runs/{run_id}/attempts/{attempt}"),
             "Playwright CI annotation does not match event/run/attempt")
        if "prHref" in ci:
            need(isinstance(ci["prHref"], str) and re.fullmatch(
                re.escape(repository) + r"/pull/[1-9][0-9]*", ci["prHref"]) is not None,
                "invalid pull request annotation")
        if "prBaseHash" in ci:
            need(isinstance(ci["prBaseHash"], str)
                 and re.fullmatch(r"[0-9a-f]{40}", ci["prBaseHash"]) is not None,
                 "invalid pull request base annotation")
        for key in ("prTitle", "branch"):
            if key in ci:
                need(text(ci[key], 4096), "invalid Playwright CI text annotation")
    if "gitCommit" in metadata:
        commit = metadata["gitCommit"]
        need(isinstance(commit, dict) and set(commit) <= {
            "shortHash", "hash", "subject", "body", "author", "committer", "branch"}
             and commit.get("hash") == checkout_sha,
             "Playwright Git annotation does not match checkout")
        if "shortHash" in commit:
            short = commit["shortHash"]
            need(isinstance(short, str) and re.fullmatch(r"[0-9a-f]{4,40}", short) is not None
                 and checkout_sha.startswith(short), "invalid short checkout annotation")
        for key, maximum in (("subject", 16384), ("body", 1000000), ("branch", 4096)):
            if key in commit:
                need(text(commit[key], maximum), "invalid Playwright Git text annotation")
        for key in ("author", "committer"):
            if key in commit:
                person = commit[key]
                need(isinstance(person, dict) and set(person) == {"name", "email", "time"}
                     and text(person["name"], 4096) and text(person["email"], 4096)
                     and type(person["time"]) is int and person["time"] >= 0,
                     "invalid Playwright commit person annotation")
    if "gitDiff" in metadata:
        need(text(metadata["gitDiff"], 100000), "invalid Playwright diff annotation")
    return {**core, "checkout_sha": checkout_sha, "event_sha": event_sha}


def build(root, source_sha, run_id, attempt, checkout_sha=None, event_sha=None):
    need(root.is_dir() and not root.is_symlink(), "real evidence root required")
    root = root.resolve()
    raw_report = read_file(root / "report.json")
    report = document(raw_report)
    need(isinstance(report, dict) and report.get("errors") == [], "report has global errors")
    config = report.get("config", {})
    identity = report_identity(config, source_sha, run_id, attempt, checkout_sha, event_sha)
    projects = config.get("projects")
    need(isinstance(projects, list) and projects, "declared browser projects required")
    project_ids = [p.get("id") for p in projects if isinstance(p, dict)]
    need(len(project_ids) == len(projects) and all(isinstance(p, str) and p for p in project_ids)
         and len(set(project_ids)) == len(project_ids), "invalid project inventory")
    stats = report.get("stats", {})
    need(isinstance(stats, dict) and type(stats.get("expected")) is int and stats["expected"] > 0
         and all(type(stats.get(k)) is int and stats[k] == 0 for k in ("skipped", "unexpected", "flaky")),
         "nonzero unskipped unflaky passing browser tests required")
    tests, seen, executed_projects = [], set(), set()
    for spec in specs(report.get("suites")):
        need(isinstance(spec, dict) and spec.get("ok") is True
             and isinstance(spec.get("id"), str) and isinstance(spec.get("tests"), list)
             and spec["tests"], "invalid or failed test specification")
        for test in spec["tests"]:
            need(isinstance(test, dict) and test.get("expectedStatus") == "passed"
                 and test.get("status") == "expected", "unexpected or skipped test")
            project = test.get("projectId")
            key = (project, spec["id"])
            need(project in project_ids and key not in seen, "duplicate or undeclared browser test")
            seen.add(key)
            executed_projects.add(project)
            results = test.get("results")
            need(isinstance(results, list) and len(results) == 1, "missing or retried browser result")
            result = results[0]
            need(isinstance(result, dict) and result.get("status") == "passed"
                 and type(result.get("retry")) is int and result["retry"] == 0 and result.get("errors") == []
                 and result.get("error") is None, "failed, retried or errored browser result")
            attachments = result.get("attachments")
            need(isinstance(attachments, list) and attachments, "test attachments required")
            groups, pins, names = {}, [], set()
            for item in attachments:
                raw = attachment_bytes(item, root)
                name = item["name"]
                need(name not in names, "duplicate attachment name")
                names.add(name)
                pins.append({"name": name, "bytes": len(raw), "sha256": digest(raw)})
                for suffix, kind, content, validate in (
                    ("-geometry.json", "geometry", "application/json", geometry),
                    ("-accessibility.json", "accessibility", "application/json", accessibility),
                    (".png", "screenshot", "image/png", screenshot),
                ):
                    if name.endswith(suffix):
                        stem = name[:-len(suffix)]
                        need(stem and item.get("contentType") == content, "invalid attachment content type")
                        validate(raw)
                        groups.setdefault(stem, set()).add(kind)
                        break
            need(groups and all(parts == {"geometry", "accessibility", "screenshot"} for parts in groups.values()),
                 "each rendered check needs screenshot, geometry and axe evidence")
            tests.append({"project": project, "id": spec["id"], "checks": sorted(groups), "attachments": pins})
    need(len(tests) == stats["expected"] and executed_projects == set(project_ids),
         "report counts or executed projects do not match inventory")
    xml = read_file(root / "junit.xml")
    need(b"<!DOCTYPE" not in xml and b"<!ENTITY" not in xml, "XML declarations forbidden")
    junit = ET.fromstring(xml)
    need(junit.tag == "testsuites" and int(junit.get("tests", "-1")) == len(tests)
         and all(int(junit.get(k, "-1")) == 0 for k in ("failures", "errors", "skipped"))
         and len(junit.findall(".//testcase")) == len(tests)
         and not any(junit.findall(".//" + k) for k in ("failure", "error", "skipped")),
         "JUnit does not confirm every browser test passed")
    inventory = artifact_inventory(root)
    image_hashes = {a["sha256"] for t in tests for a in t["attachments"] if a["name"].endswith(".png")}
    retained_images = {p["sha256"] for p in inventory if p["path"].startswith("html/data/") and p["path"].endswith(".png")}
    need(image_hashes <= retained_images, "portable HTML screenshot assets missing")
    need(next(p for p in inventory if p["path"] == "html/index.html")["bytes"] > 0,
         "portable HTML report is empty")
    report_pin = next(p for p in inventory if p["path"] == "report.json")
    need(report_pin["sha256"] == digest(raw_report), "report changed during validation")
    return {"schema": SCHEMA, **identity,
            "scope": "synthetic frontend fixtures; not live training qualification",
            "executed": len(tests), "passed": len(tests), "failed": 0,
            "checks": sum(len(t["checks"]) for t in tests), "projects": sorted(project_ids),
            "tests": tests, "artifacts": inventory}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("create", "verify"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--checkout-sha", help="trusted browser checkout; defaults to source")
    parser.add_argument("--event-sha", help="trusted workflow event SHA; defaults to source")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt", required=True)
    args = parser.parse_args(argv)
    try:
        expected = build(args.root, args.source_sha, args.run_id, args.attempt,
                         args.checkout_sha, args.event_sha)
        receipt = args.root / RECEIPT
        if args.mode == "create":
            raw = (json.dumps(expected, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
            descriptor = os.open(receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
        else:
            need(document(read_file(receipt)) == expected, "missing, stale or tampered rendered receipt/evidence")
        print(json.dumps({"schema": SCHEMA, "mode": args.mode, "executed": expected["executed"],
                          "checks": expected["checks"], "source_sha": args.source_sha,
                          "checkout_sha": expected["checkout_sha"], "event_sha": expected["event_sha"],
                          "run_id": args.run_id, "attempt": args.attempt}))
        return 0
    except (OSError, ValueError, TypeError, KeyError, ET.ParseError) as error:
        print(f"rendered UX evidence refused: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
