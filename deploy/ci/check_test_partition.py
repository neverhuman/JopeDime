#!/usr/bin/env python3
"""Prove that split CI jobs run exactly the tests of the single job each one replaced.

rust-test (ci.yml): every `slice:M/N` matrix leg lists the unsharded selection, which is the
pre-split rust-test command, as collection-unsharded.json next to its own partitioned
collection.json. The shards' selections must cover the unsharded selection exactly once, differ
from it only by the partition, and each shard's JUnit must execute exactly its own selection.

full-main-cpu (full-main.yml): deploy/ci/full-main.py GROUPS must cover every declared `full-*`
family exactly once (the families the single pre-split job ran), and the workflow must run every
group. Given --base, nothing the base declared may be missing.

Commands:
  static [--base REV]                     workflow, group and scope declarations (hygiene lane)
  rust-shards --receipts DIR --output F   downloaded rust-test shard evidence (ci.yml aggregate)
  collections --unsharded F --shard F...  any listings, e.g. the base command against the shards
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
RUST_TEST_SHARDS = 2
SHARD = re.compile(r"slice:([1-9][0-9]*)/([1-9][0-9]*)")


def shard_unit(receipt):
    """The execution unit of a CI receipt: its lane, or rust-test-M for slice M of rust-test."""
    lane = receipt["lane"]
    if lane != "rust-test":
        return lane
    match = SHARD.fullmatch(str(receipt.get("environment", {}).get("JOPE_RUST_TEST_PARTITION", "")))
    if not match:
        raise ValueError("rust-test receipt does not record its slice partition")
    return f"rust-test-{match[1]}"


def listing(path):
    """{(binary-id, test): (filter status, mismatch reason, ignored)} of a nextest JSON listing."""
    data = json.loads(Path(path).read_text())
    tests = {}
    for binary, suite in data["rust-suites"].items():
        for name, case in suite.get("testcases", {}).items():
            match = case.get("filter-match", {})
            tests[(binary, name)] = (match.get("status"), match.get("reason"), case.get("ignored"))
    if len(tests) != data["test-count"]:
        raise ValueError(f"{path}: test-count {data['test-count']} differs from {len(tests)} listed tests")
    return tests


def executed(path):
    """Executed (not skipped) JUnit cases as [((classname, name), passed)]."""
    rows = []
    for case in ET.parse(path).getroot().iter("testcase"):
        if case.find("skipped") is None:
            rows.append(((case.get("classname"), case.get("name")), case.find("failure") is None and case.find("error") is None))
    return rows


def digest(tests):
    return hashlib.sha256("".join(f"{b}\t{n}\n" for b, n in sorted(tests)).encode()).hexdigest()


def named(tests, limit=50):
    return [f"{b} {n}" for b, n in sorted(tests)[:limit]]


def compare(unsharded, shards):
    """unsharded: listing paths (one per shard run, all identical); shards: [(name, collection, junit or None)]."""
    errors = []
    old = listing(unsharded[0])
    for other in unsharded[1:]:
        if listing(other) != old:
            errors.append(f"{other}: unsharded listing differs between shards")
    selected = {test for test, (status, _, _) in old.items() if status == "matches"}
    union, seen, report = set(), {}, {}
    for name, collection, junit in shards:
        current = listing(collection)
        if set(current) != set(old):
            errors.append(f"{name}: lists {len(set(current) ^ set(old))} tests the unsharded listing does not")
        mine = set()
        for test, row in current.items():
            if test not in old:
                continue
            if old[test][0] == "matches" and row[0] == "matches":
                mine.add(test)
            elif not (old[test][0] == "matches" and row[:2] == ("mismatch", "partition")) and row != old[test]:
                errors.append(f"{name}: {test[0]} {test[1]} selection differs beyond the partition: {row} vs {old[test]}")
        for test in mine:
            seen.setdefault(test, []).append(name)
        union |= mine
        row = {"selected": len(mine), "selected_sha256": digest(mine)}
        if junit is not None:
            ran = executed(junit)
            ids = [test for test, _ in ran]
            twice = {test for test in ids if ids.count(test) > 1}
            row.update(executed=len(ids), failed=sum(not ok for _, ok in ran))
            if set(ids) != mine or twice:
                errors.append(f"{name}: executed {len(set(ids))} tests, selected {len(mine)}; "
                              f"unexecuted {named(mine - set(ids), 5)}, unselected {named(set(ids) - mine, 5)}, repeated {named(twice, 5)}")
            if row["failed"]:
                errors.append(f"{name}: {row['failed']} executed tests did not pass")
        report[name] = row
    duplicated = {test for test, owners in seen.items() if len(owners) > 1}
    missing, extra = selected - union, union - selected
    errors += [f"{len(missing)} unsharded tests in no shard: {named(missing, 10)}"] if missing else []
    errors += [f"{len(duplicated)} tests in more than one shard: {named(duplicated, 10)}"] if duplicated else []
    errors += [f"{len(extra)} shard tests outside the unsharded selection: {named(extra, 10)}"] if extra else []
    return {"schema": "jope.test-partition/v1", "kind": "rust-test",
            "status": "failed" if errors else "passed",
            "unsharded": {"listed": len(old), "selected": len(selected), "selected_sha256": digest(selected)},
            "shards": report, "union": {"selected": len(union), "selected_sha256": digest(union)},
            "missing": named(missing), "duplicated": named(duplicated), "extra": named(extra), "errors": errors}


def rust_shard_dirs(receipts):
    """The evidence directory of each rust-test-M unit; publication retries share bytes (aggregate.py)."""
    units = {}
    for artifact in sorted(Path(receipts).iterdir()):
        path = artifact / "receipt.json"
        if artifact.is_dir() and not artifact.is_symlink() and path.is_file() and not path.is_symlink():
            receipt = json.loads(path.read_text())
            if receipt.get("lane") == "rust-test":
                units.setdefault(shard_unit(receipt), artifact)
    expected = [f"rust-test-{shard}" for shard in range(1, RUST_TEST_SHARDS + 1)]
    if sorted(units) != sorted(expected):
        raise ValueError(f"rust-test shard receipts {sorted(units)}, expected {expected}")
    return [(unit, units[unit]) for unit in expected]


def verify_shards(dirs):
    """dirs: [(unit, evidence directory)] of every rust-test shard."""
    return compare([folder / "collection-unsharded.json" for _, folder in dirs],
                   [(unit, folder / "collection.json", folder / "junit.xml") for unit, folder in dirs])


def load_full_main():
    spec = importlib.util.spec_from_file_location("full_main", ROOT / "deploy/ci/full-main.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "deploy/ci"))
    spec.loader.exec_module(module)
    return module


def declared_families(text):
    return sorted(name for name in json.loads(text)["lanes"] if name.startswith("full-"))


def static(base=None):
    errors, full = [], load_full_main()
    families = declared_families((ROOT / "deploy/ci/coverage.json").read_text())
    groups = {name: list(members) for name, members in full.GROUPS.items()}
    errors += full.partition_errors(families, full.GROUPS)
    workflow = (ROOT / ".github/workflows/full-main.yml").read_text()
    matrix = re.findall(r"^ +group: \[([^\]]*)\]$", workflow, re.M)
    if len(matrix) != 1 or [g.strip() for g in matrix[0].split(",")] != list(groups):
        errors.append(f"full-main.yml cpu matrix {matrix} does not run exactly the groups {list(groups)}")
    for step in ("run", "summarize"):
        if workflow.count(f'full-main.py {step} --group "$GROUP" ') != 1:
            errors.append(f"full-main.yml does not {step} exactly its matrix group")
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    shards = re.findall(r"^ +shard: \[([^\]]*)\]$", ci, re.M)
    expected = [str(n) for n in range(1, RUST_TEST_SHARDS + 1)]
    if len(shards) != 1 or [s.strip() for s in shards[0].split(",")] != expected:
        errors.append(f"ci.yml rust-test matrix {shards} is not shards {expected}")
    if ci.count(f"JOPE_RUST_TEST_PARTITION: slice:${{{{ matrix.shard }}}}/{RUST_TEST_SHARDS}\n") != 1:
        errors.append(f"ci.yml rust-test does not partition by slice:<shard>/{RUST_TEST_SHARDS}")
    scope = subprocess.check_output(["bash", "deploy/ci/rust-test-scope.sh"], cwd=ROOT, text=True).split()
    report = {"schema": "jope.test-partition/v1", "kind": "static",
              "full_main": {"declared": families, "groups": groups}, "rust_test": {"shards": RUST_TEST_SHARDS, "scope": scope[1::2]}}
    if base:
        old = declared_families(subprocess.check_output(["git", "show", f"{base}:deploy/ci/coverage.json"], cwd=ROOT, text=True))
        assigned = {name for members in groups.values() for name in members}
        old_scope = subprocess.run(["bash", "-s"], cwd=ROOT, text=True, capture_output=True, check=True,
                                   input=subprocess.check_output(["git", "show", f"{base}:deploy/ci/rust-test-scope.sh"], cwd=ROOT, text=True)).stdout.split()
        report["base"] = {"rev": base, "full_main_families": old, "missing_families": sorted(set(old) - assigned),
                          "added_families": sorted(assigned - set(old)), "rust_test_scope": old_scope[1::2],
                          "missing_scope": sorted(set(old_scope[1::2]) - set(scope[1::2]))}
        errors += [f"family {name} ran on {base} but is in no group" for name in report["base"]["missing_families"]]
        errors += [f"crate {name} was in the {base} rust-test scope but is not now" for name in report["base"]["missing_scope"]]
    report.update(status="failed" if errors else "passed", errors=errors)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["static", "rust-shards", "collections"])
    parser.add_argument("--base", help="static: also require nothing declared at this revision is missing")
    parser.add_argument("--receipts", type=Path, help="rust-shards: downloaded CI artifacts of this run and attempt")
    parser.add_argument("--unsharded", type=Path, action="append", default=[])
    parser.add_argument("--shard", type=Path, action="append", default=[], help="collections: partitioned listing")
    parser.add_argument("--output", type=Path, help="write the report here (exclusive create)")
    args = parser.parse_args()
    try:
        if args.command == "static":
            report = static(args.base or None)
        elif args.command == "rust-shards":
            if args.receipts is None:
                parser.error("rust-shards requires --receipts")
            report = verify_shards(rust_shard_dirs(args.receipts))
        else:
            if not args.unsharded or not args.shard:
                parser.error("collections requires --unsharded and --shard")
            report = compare(args.unsharded, [(str(path), path, None) for path in args.shard])
    except (OSError, ValueError, KeyError, TypeError, ET.ParseError, subprocess.CalledProcessError) as exc:
        report = {"schema": "jope.test-partition/v1", "kind": args.command, "status": "failed", "errors": [str(exc)]}
    text = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(text)
    print(text, end="")
    return report["status"] != "passed"


if __name__ == "__main__":
    raise SystemExit(main())
