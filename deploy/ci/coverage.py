#!/usr/bin/env python3
"""Static CI execution map guard. Discovery is neither collection nor execution."""
import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def workflow_jobs(text):
    """Read block-mapping job keys; fail closed on unsupported YAML structure.

    actionlint validates full YAML independently. Keeping the jobs mapping in block
    form avoids introducing a second YAML dependency into the minimal CI image.
    Comments, arbitrary consistent indentation and flow-style job values are supported.
    """
    headers = list(re.finditer(r"^jobs\s*:(.*)$", text, re.M))
    if len(headers) != 1 or headers[0][1].strip().split("#", 1)[0].strip():
        raise ValueError("workflow requires one top-level block jobs mapping")
    names = []
    indentation = None
    for line in text[headers[0].end():].splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith((" ", "\t")):
            break
        if "\t" in line[:len(line) - len(line.lstrip())]:
            raise ValueError("workflow job indentation cannot contain tabs")
        depth = len(line) - len(line.lstrip(" "))
        if indentation is None:
            indentation = depth
        if depth < indentation:
            raise ValueError("inconsistent workflow job indentation")
        if depth == indentation:
            match = re.match(r'''(?:([A-Za-z_][\w-]*)|"([A-Za-z_][\w-]*)"|'([A-Za-z_][\w-]*)')\s*:(?:\s|$)''', line.strip())
            if not match:
                raise ValueError(f"unsupported workflow job key: {line.strip()}")
            names.append(next(value for value in match.groups() if value))
    if not names or len(names) != len(set(names)):
        raise ValueError("workflow has missing or duplicate job keys")
    return names


def default_features(discovered, package):
    prefix = f"feature:{package}/"
    features = {key[len(prefix):]: value["dependencies"] for key, value in discovered.items() if key.startswith(prefix)}
    active = set()
    pending = ["default"] if "default" in features else []
    while pending:
        feature = pending.pop()
        if feature not in active:
            active.add(feature)
            pending.extend(dependency for dependency in features.get(feature, []) if dependency in features)
    return active


def discover(root=ROOT):
    metadata = json.loads(subprocess.check_output(
        ["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1"], cwd=root, text=True))
    files = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root
    ).decode().split("\0")
    items = {}
    for package in metadata["packages"]:
        if package["id"] not in metadata["workspace_members"]:
            continue
        name = package["name"]
        items[f"package:{name}"] = {"kind": "package", "manifest": str(Path(package["manifest_path"]).relative_to(root))}
        for feature, dependencies in sorted(package["features"].items()):
            items[f"feature:{name}/{feature}"] = {"kind": "feature", "dependencies": sorted(dependencies)}
        for target in package["targets"]:
            key = f"target:{name}/{'+'.join(target['kind'])}/{target['name']}"
            items[key] = {"kind": "target", "source": str(Path(target["src_path"]).relative_to(root)),
                          "test": target["test"], "doctest": target["doctest"],
                          "required_features": sorted(target.get("required-features", []))}
    for filename in sorted(set(files) - {""}):
        path = Path(filename)
        if path.suffix == ".py" and (path.name.startswith(("test_", "test-")) or path.stem.endswith("_test")):
            items[f"python:{filename}"] = {"kind": "python-family"}
        if path.parent == Path(".github/workflows") and path.suffix in (".yaml", ".yml"):
            items[f"workflow:{filename}"] = {"kind": "workflow"}
            for name in workflow_jobs((root / filename).read_text()):
                items[f"job:{filename}/{name}"] = {"kind": "workflow-job"}
        if path.suffix in (".rs", ".py", ".sh") and "bench" in path.name.lower():
            items[f"benchmark:{filename}"] = {"kind": "benchmark-source"}
    return dict(sorted(items.items()))


def validate(mapping, discovered):
    errors = []
    if mapping.get("schema") != "jope.coverage/v1":
        errors.append("unsupported coverage schema")
    recorded = mapping.get("items", {})
    for key in sorted(discovered.keys() - recorded.keys()):
        errors.append(f"unassigned addition: {key}")
    for key in sorted(recorded.keys() - discovered.keys()):
        errors.append(f"stale assignment: {key}")
    lanes = mapping.get("lanes", {})
    dispositions = mapping.get("dispositions", {})
    for name, lane in lanes.items():
        for field in ("command", "environment", "inputs", "path_applicability", "evidence", "requires_nonzero_tests", "stage"):
            if field not in lane:
                errors.append(f"lane {name} missing {field}")
        if not lane.get("command") or not lane.get("inputs") or not lane.get("evidence"):
            errors.append(f"lane {name} has empty command/inputs/evidence")
        if type(lane.get("requires_nonzero_tests")) is not bool:
            errors.append(f"lane {name} requires_nonzero_tests must be boolean")
        if lane.get("stage") not in ("static", "collection", "execution"):
            errors.append(f"lane {name} has invalid stage")
    for name, disposition in dispositions.items():
        if disposition.get("status") not in ("deferred", "historical"):
            errors.append(f"disposition {name} has invalid status")
        for field in ("reason", "dependencies", "owner_phase"):
            if not disposition.get(field):
                errors.append(f"disposition {name} missing {field}")
    for key in sorted(recorded.keys() & discovered.keys()):
        item = recorded[key]
        if item.get("discovery") != discovered[key]:
            errors.append(f"changed discovery: {key}; review its assignment")
        assigned = item.get("lanes", [])
        deferred = item.get("dispositions", [])
        if key.startswith("target:"):
            package = key.split(":", 1)[1].split("/", 1)[0]
            missing_features = set(discovered[key]["required_features"]) - default_features(discovered, package)
            default_lanes = {name for name, lane in lanes.items() if lane.get("feature_selection") == "package-defaults"}
            if missing_features and (default_lanes.intersection(assigned) or "full-cpu-default" in deferred):
                errors.append(f"non-default required features {sorted(missing_features)} assigned to default execution: {key}")
        if not assigned and not deferred:
            errors.append(f"unassigned item: {key}")
        for lane in assigned:
            if lane not in lanes:
                errors.append(f"unknown lane {lane} for {key}")
        for disposition in deferred:
            if disposition not in dispositions:
                errors.append(f"unknown disposition {disposition} for {key}")
        if not item.get("scope"):
            errors.append(f"missing execution scope: {key}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["discover", "validate"])
    parser.add_argument("--mapping", type=Path, default=ROOT / "deploy/ci/coverage.json")
    args = parser.parse_args()
    discovered = discover()
    if args.command == "discover":
        print(json.dumps({"stage": "static", "items": discovered}, indent=2))
        return 0
    errors = validate(json.loads(args.mapping.read_text()), discovered)
    for error in errors:
        print(error)
    counts = {kind: sum(i["kind"] == kind for i in discovered.values()) for kind in sorted({i["kind"] for i in discovered.values()})}
    print(json.dumps({"stage": "static", "counts": counts, "unassigned_or_invalid": len(errors)}, sort_keys=True))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
