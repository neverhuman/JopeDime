#!/usr/bin/env python3
"""Reconcile runner summaries with JUnit without inventing omitted skip counts."""
import re
import xml.etree.ElementTree as ET

FIELDS = ("collected", "executed", "passed", "failed", "skipped", "excluded")


def counts(passed, failed, skipped, excluded=0):
    return dict(collected=passed + failed + skipped + excluded, executed=passed + failed,
                passed=passed, failed=failed, skipped=skipped, excluded=excluded)


def junit_counts(path):
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"invalid JUnit: {exc}") from exc
    cases = list(root.iter("testcase"))
    skipped = sum(case.find("skipped") is not None for case in cases)
    failed = sum(case.find("failure") is not None or case.find("error") is not None for case in cases)
    # A missing skipped attribute in nextest 0.9.133 is unknown, even if no skipped
    # testcase was serialized. Never use len(cases) as the collected count.
    suites = list(root.iter("testsuite"))
    skip_count = sum(int(s.attrib["skipped"]) for s in suites) if suites and all("skipped" in s.attrib for s in suites) else None
    if skip_count is not None and skip_count != skipped:
        raise ValueError("JUnit skipped attributes disagree with serialized cases")
    return dict(collected=None, executed=len(cases) - skipped, passed=len(cases) - skipped - failed,
                failed=failed, skipped=skip_count, excluded=None)


def summarize(log, backend, junit=None):
    log = re.sub(r"\x1b\[[0-9;]*m", "", log)
    if backend == "none":
        return dict.fromkeys(FIELDS)
    if backend == "nextest":
        summaries = re.findall(r"Summary\s+\[[^\]]+\]\s+(\d+) tests run:\s*([^\n]+)", log)
        if len(summaries) != 1:
            raise ValueError("expected exactly one nextest execution summary")
        total, detail = summaries[0]
        values = {name: int(n) for n, name in re.findall(r"(\d+) (passed|failed|skipped)", detail)}
        if "passed" not in values or any(word in detail for word in ("timed out", "leaked", "aborted")):
            raise ValueError("incomplete or abnormal nextest summary")
        result = counts(values["passed"], values.get("failed", 0), values.get("skipped", 0))
        if result["executed"] != int(total):
            raise ValueError("nextest execution total mismatch")
    elif backend == "pytest":
        summaries = [line for line in log.splitlines() if re.search(r"\b\d+ (?:passed|failed|skipped|error|errors)\b.*\bin [0-9.]+s", line)]
        if len(summaries) != 1:
            raise ValueError("expected exactly one pytest execution summary")
        values = {name: int(n) for n, name in re.findall(r"(\d+) (passed|failed|skipped|deselected|errors?|xfailed|xpassed)", summaries[0])}
        if any(values.get(name, 0) for name in ("error", "errors", "xfailed", "xpassed")):
            raise ValueError("pytest errors/xfail/xpass require an explicit disposition")
        result = counts(values.get("passed", 0), values.get("failed", 0), values.get("skipped", 0), values.get("deselected", 0))
    elif backend == "rust":
        matches = re.findall(r"test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored; \d+ measured; (\d+) filtered out", log)
        if not matches:
            raise ValueError("missing Rust harness summaries")
        result = counts(*(sum(int(row[i]) for row in matches) for i in range(4)))
    elif backend == "unittest":
        matches = re.findall(r"Ran (\d+) tests? in [^\n]+\n\s*\n(OK(?: \(skipped=(\d+)\))?|FAILED[^\n]*)", log)
        if not matches or any(not status.startswith("OK") for _, status, _ in matches):
            raise ValueError("missing or failed unittest summaries")
        total = sum(int(n) for n, _, _ in matches)
        skipped = sum(int(n or 0) for _, _, n in matches)
        result = counts(total - skipped, 0, skipped)
    else:
        raise ValueError(f"unsupported result backend: {backend}")
    if junit is not None:
        report = junit_counts(junit)
        for field in ("executed", "passed", "failed"):
            if report[field] != result[field]:
                raise ValueError(f"JUnit and execution summary disagree: {field}")
        if report["skipped"] is not None and report["skipped"] != result["skipped"]:
            raise ValueError("JUnit and execution summary disagree: skipped")
    return result
