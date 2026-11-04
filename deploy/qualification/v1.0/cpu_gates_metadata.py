import ast
import hashlib
import importlib.util
import json
import re


def inventory(root, revision=None, *, git, controls):
    def read(name):
        return (
            git(root, "show", revision + ":" + name)
            if revision
            else (root / name).read_text()
        )

    mapping = json.loads(read("deploy/ci/coverage.json"))
    syntax = ast.parse(read("deploy/ci/full-main.py"))
    definitions = [
        n
        for n in syntax.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "GROUPS" for t in n.targets)
    ]
    if len(definitions) != 1:
        raise ValueError("expected one literal full CPU GROUPS declaration")
    groups = ast.literal_eval(definitions[0].value)
    names = [name for entries in groups.values() for name in entries]
    families = {
        name: lane
        for name, lane in mapping["lanes"].items()
        if name.startswith("full-")
    }
    if not names or len(names) != len(set(names)) or set(names) != set(families):
        raise ValueError("full CPU groups and coverage families must partition exactly")
    result = [
        dict(
            id=name,
            command=command,
            result_backend="none",
            requires_nonzero_tests=False,
        )
        for name, command in controls.items()
    ]
    for name in names:
        if not re.fullmatch(r"full-[a-z0-9-]+", name):
            raise ValueError("invalid CPU family identifier")
        lane = families[name]
        command = lane["command"]
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(x, str) or not x for x in command)
        ):
            raise ValueError("invalid CPU command: " + name)
        backend = lane["result_backend"]
        if backend not in {"none", "nextest", "rust", "pytest", "unittest"}:
            raise ValueError("unsupported result backend: " + name)
        if lane["requires_nonzero_tests"] and backend == "none":
            raise ValueError("nonzero test family requires a result backend: " + name)
        result.append(
            dict(
                id=name,
                command=command,
                result_backend=backend,
                requires_nonzero_tests=lane["requires_nonzero_tests"],
            )
        )
    return result


def result_parser(root):
    spec = importlib.util.spec_from_file_location(
        "qualification_cpu_results", root / "deploy/ci/results.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def release_inventory(plan_root, *, digest, gate_ids):
    """Inventory dated plan/status metadata; never infer signed-byte qualification."""
    plan_root = plan_root.resolve()
    sources = {}
    texts = {}

    def read(relative, required=True):
        path = plan_root / relative
        if not path.resolve().is_relative_to(plan_root):
            raise ValueError("plan metadata escapes its declared root")
        if relative not in sources:
            exists = path.is_file()
            if required and not exists:
                raise ValueError("required plan metadata missing: " + relative)
            data = path.read_bytes() if exists else None
            sources[relative] = dict(
                path=relative,
                exists=exists,
                sha256=hashlib.sha256(data).hexdigest() if data is not None else None,
            )
            texts[relative] = data.decode("utf-8") if data is not None else None
        return texts[relative], sources[relative]

    master, _ = read("master_plan.md")
    specification, _ = read("source/spec-v1.md")
    section = specification.split("## 14. Release gates", 1)[-1].split("\n## ", 1)[0]
    rows = re.findall(r"^\| \*\*([A-Z][0-9]+)\*\* \| ([^|]+)\| (.+) \|$", section, re.M)
    if [row[0] for row in rows] != gate_ids[:-1]:
        raise ValueError("specification must enumerate every release gate exactly once")
    q1 = re.search(r"\*\*Capability gate Q1 — learned guidance\.\*\* (.+)", section)
    if q1 is None or "28" not in master or "Q1" not in master:
        raise ValueError("master plan/specification capability-gate mapping missing")
    rows.append(("Q1", "Learned guidance default", q1[1]))
    gates = {
        gate: dict(
            status="not-qualified",
            name=name.strip(),
            criterion=criterion.strip(),
            reason="Signed candidate and qualifying rerun receipts are unavailable; task notes are not qualification",
            release_blocking=gate != "Q1",
            rerun_status="not-run",
            rerun_command=None,
            qualification_evidence=None,
            evidence=[],
        )
        for gate, name, criterion in rows
    }
    task_rows = re.findall(
        r"^\| P[0-9]+ \| \[([A-Z]+-[0-9]+)\]\((tasks/[^/)]+\.md)\).*?\| ([^|]+) \|$",
        master,
        re.M,
    )
    if not task_rows:
        raise ValueError("master plan task index missing")
    for task_id, card_path, plan_status in task_rows:
        card, card_source = read(card_path)
        field = re.search(r"^\| Gates closed \| ([^|]+)\|", card, re.M)
        if field is None:
            raise ValueError("task gate mapping missing: " + task_id)
        mapped = set(re.findall(r"\b[A-Z][0-9]+\b", field[1])) & set(gate_ids)
        if not mapped:  # REL-07's aggregate 'all' is not per-gate proof.
            continue
        note, note_source = read("status/" + task_id + ".md", required=False)
        evidence = dict(
            task=task_id,
            task_card=card_source,
            reported_plan_status=plan_status.strip(),
            status_note=dict(
                **note_source,
                reported_status=next(
                    (x.strip() for x in (note or "").splitlines() if x.strip()), None
                ),
            ),
            qualifies_signed_bytes=False,
        )
        for gate in mapped:
            gates[gate]["evidence"].append(evidence)
    # Catch a concurrently rewritten status/plan during this inventory, including
    # an absent note appearing. Later changes require a fresh dated inventory.
    for relative, source in sources.items():
        path = plan_root / relative
        if path.is_file() != source["exists"] or (
            source["exists"] and digest(path) != source["sha256"]
        ):
            raise ValueError(
                "plan/status metadata changed during inventory: " + relative
            )
    return dict(
        plan_root=str(plan_root),
        plan_sources=list(sources.values()),
        release_gates=gates,
    )


