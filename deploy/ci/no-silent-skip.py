#!/usr/bin/env python3
"""no-silent-skip.py — fail a Rust test lane when a test did not run and nothing accounts for it.

    no-silent-skip.py --lane <name> --list <nextest-list.json> --junit <junit.xml> \
                      [--log <nextest-run.log>] [--allowlist deploy/ci/ignored-tests.txt]

Inputs are produced by the same lane with the same profile, packages and features:
  --list   `cargo nextest list -P <profile> ... --message-format json` (every test, ignored flag)
  --junit  the profile's junit report (only tests that ran)
  --log    the run's stdout with `--success-output final` (per-test captured output)

Checks (each violation is printed; the exit code is 1 if any):
  1. every test the list marks `ignored` is an `ignored` line of the allowlist;
  2. every test the list does not mark ignored appears in the junit exactly once and is not
     <skipped/>; the junit contains no <skipped/> at all;
  3. every `gated` allowlist line whose lane is this lane has all of its env variables set in this
     process (the workflow step runs the check with the lane's env) and, when --log is given, that
     test's captured output holds no skip marker ("SKIP", "skipping");
  4. with --log: a skip marker in the output of any test that is not a `gated` line for another
     lane is a violation (the test returned early in the one lane that must run it for real).
Allowlist lines that name a test absent from the list are reported as notes (feature-gated files
such as the neural-only tests are not compiled in every lane) and are never failures.
"""
import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

MARKER = re.compile(r"\bskip(ped|ping)?\b", re.IGNORECASE)
SUMMARY = re.compile(r"^\s+Summary\s+\[")
SECTION = re.compile(r"^\s+(PASS|FAIL|TIMEOUT|SIGSEGV|ABORT|LEAK|FLAKY)\s+\[[^\]]*\]\s+\(\s*\d+/\d+\)\s+(\S+)\s+(\S+)\s*$")


def read_allowlist(path):
    ignored, gated = {}, {}
    with open(path) as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            kind = parts[0]
            if kind == "ignored" and len(parts) >= 3:
                ignored[(parts[1], parts[2])] = " ".join(parts[3:])
            elif kind == "gated" and len(parts) >= 5:
                gated[(parts[1], parts[2])] = {"lane": parts[3], "env": [] if parts[4] == "-" else parts[4].split(","),
                                               "reason": " ".join(parts[5:])}
            else:
                sys.exit(f"{path}:{lineno}: unreadable allowlist line: {line}")
    return ignored, gated


def read_list(path):
    with open(path) as f:
        data = json.load(f)
    tests = {}
    for binary_id, suite in data.get("rust-suites", {}).items():
        for name, case in suite.get("testcases", {}).items():
            tests[(binary_id, name)] = bool(case.get("ignored"))
    return tests


def read_junit(path):
    root = ET.parse(path).getroot()
    seen, skipped = {}, []
    for case in root.iter("testcase"):
        key = (case.get("classname", ""), case.get("name", ""))
        seen[key] = seen.get(key, 0) + 1
        if case.find("skipped") is not None:
            skipped.append(key)
    return seen, skipped, int(root.get("tests", "0"))


def read_log(path):
    """Map (binary-id, test) -> captured output text (sections after the Summary line)."""
    sections, current, after_summary = {}, None, False
    with open(path, errors="replace") as f:
        for line in f:
            if not after_summary:
                after_summary = SUMMARY.match(line) is not None
                continue
            m = SECTION.match(line)
            if m:
                current = (m.group(2), m.group(3))
                sections.setdefault(current, [])
                continue
            if current is not None:
                sections[current].append(line)
    return {k: "".join(v) for k, v in sections.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lane", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--junit", required=True)
    ap.add_argument("--log")
    ap.add_argument("--allowlist", default=os.path.join(os.path.dirname(__file__), "ignored-tests.txt"))
    ap.add_argument("--verbose", action="store_true", help="print every note (default: counts only)")
    args = ap.parse_args()

    allowed_ignored, gated = read_allowlist(args.allowlist)
    tests = read_list(args.list)
    seen, junit_skipped, junit_total = read_junit(args.junit)
    log = read_log(args.log) if args.log else None
    violations, notes = [], []

    # 1. ignored tests must be allowlisted
    listed_ignored = {k for k, ig in tests.items() if ig}
    for key in sorted(listed_ignored - set(allowed_ignored)):
        violations.append(f"ignored test not in the allowlist: {key[0]} {key[1]}")
    for key in sorted(set(allowed_ignored) - set(tests)):
        notes.append(f"allowlisted ignored test not compiled in this lane: {key[0]} {key[1]}")
    for key in sorted(set(allowed_ignored) & set(tests)):
        if not tests[key]:
            violations.append(f"allowlisted as ignored but the list says it runs (update the allowlist): {key[0]} {key[1]}")

    # 2. every runnable test appears in the junit exactly once and is not skipped
    runnable = {k for k, ig in tests.items() if not ig}
    for key in sorted(runnable):
        n = seen.get(key, 0)
        if n == 0:
            violations.append(f"test listed but absent from the junit (never ran): {key[0]} {key[1]}")
        elif n > 1:
            notes.append(f"test appears {n} times in the junit (retries): {key[0]} {key[1]}")
    for key in sorted(set(seen) - set(tests)):
        notes.append(f"junit names a test the list does not (different filters?): {key[0]} {key[1]}")
    for key in junit_skipped:
        violations.append(f"junit reports <skipped/>: {key[0]} {key[1]}")
    if junit_total != len(runnable):
        notes.append(f"junit tests={junit_total}, list runnable={len(runnable)}")

    # 3./4. env gates and skip markers
    for key, gate in sorted(gated.items()):
        if gate["lane"] != args.lane:
            continue
        missing = [v for v in gate["env"] if not os.environ.get(v)]
        if key not in tests:
            notes.append(f"gated test for this lane is not compiled here: {key[0]} {key[1]}")
            continue
        if missing:
            violations.append(f"env gate unset for {key[0]} {key[1]} in lane {args.lane}: {','.join(missing)} ({gate['reason']})")
    if log is not None:
        for key, text in sorted(log.items()):
            hits = [l.strip() for l in text.splitlines() if MARKER.search(l)]
            if not hits:
                continue
            gate = gated.get(key)
            if gate and gate["lane"] != args.lane:
                notes.append(f"skip marker allowed (runs for real in lane {gate['lane']}): {key[0]} {key[1]}: {hits[0]}")
            else:
                violations.append(f"skip marker in the output of {key[0]} {key[1]}: {hits[0]}")

    print(f"no-silent-skip lane={args.lane}: {len(tests)} listed, {len(runnable)} runnable, "
          f"{len(listed_ignored)} ignored, junit tests={junit_total}, skipped={len(junit_skipped)}, "
          f"log sections={len(log) if log is not None else 'n/a'}")
    if args.verbose:
        for n in notes:
            print(f"  note: {n}")
    elif notes:
        kinds = {}
        for n in notes:
            kinds[n.split(":")[0]] = kinds.get(n.split(":")[0], 0) + 1
        for kind, count in sorted(kinds.items()):
            print(f"  notes: {count} x {kind} (--verbose lists them)")
    for v in violations:
        print(f"  VIOLATION: {v}")
    if violations:
        print(f"no-silent-skip: {len(violations)} violation(s)")
        return 1
    print("no-silent-skip: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
