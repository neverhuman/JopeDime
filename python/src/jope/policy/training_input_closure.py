"""Authenticate the projected data a learning worker may read, before model loading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

COMMON = {
    "corpus",
    "context-index",
    "encoder-identity",
    "vocabulary",
    "teaching-admission",
    "family-partition",
    "checkpoint",
}
CONTEXT = {"context-values", "context-producer", "fitting-input", "teaching-recipe"}
EXCLUDED = {"outer-test", "sealed-meta-test", "outside-selection-evaluation"}
INPUT_ROLES = {
    "corpus": "corpus",
    "contexts": "context-index",
    "encoder": "encoder-identity",
    "vocabulary": "vocabulary",
    "admission": "teaching-admission",
    "checkpoint": "checkpoint",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def reference(ref):
    require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "closure artifact reference required")
    path = Path(ref["path"])
    require(
        path.is_absolute() and path.resolve(strict=True) == path and path.is_file(),
        "closure path must be physical",
    )
    with path.open("rb") as stream:
        require(
            hashlib.file_digest(stream, "sha256").hexdigest() == ref["sha256"], "closure artifact changed"
        )
    return path


def admit(inputs):
    """Return a verified allowlist. Unlisted references are rejected without opening them."""
    require("input_closure" in inputs, "projected input closure is required")
    manifest_path = reference(inputs["input_closure"])
    manifest = json.loads(manifest_path.read_bytes())
    require(manifest.get("schema") == "jope.training-input-closure/v1", "unsupported input closure")
    root = Path(manifest["root"])
    require(
        root.is_absolute() and root.resolve(strict=True) == root and root == manifest_path.parent,
        "closure root differs",
    )
    require(not root.is_relative_to("/tmp"), "closure root must be durable")
    require(set(manifest["excluded_roles"]) == EXCLUDED, "closure exclusion domains differ")
    families = manifest["families"]
    require(
        families and set(families.values()) <= {"train", "development"}, "closure family role is forbidden"
    )
    contexts = manifest["contexts"]
    require(contexts and len(contexts) == len(set(contexts)), "closure contexts absent or duplicated")
    inventory = {}
    for item in manifest["artifacts"]:
        ref, role = item["artifact"], item["role"]
        require(role in COMMON | CONTEXT, "forbidden closure artifact role")
        path = Path(ref["path"])
        require(
            path.parent == root and str(path) not in inventory,
            "closure artifact escapes root or is duplicated",
        )
        if role in CONTEXT:
            require(
                item.get("split") in {"train", "development"}
                and families.get(item.get("source_family")) == item["split"]
                and item.get("context_id") in contexts,
                "closure context family/split differs",
            )
        inventory[str(path)] = item
    require(set(inputs) <= set(INPUT_ROLES) | {"input_closure"}, "unlisted input kind")
    for name, ref in inputs.items():
        if name == "input_closure":
            continue
        item = inventory.get(ref["path"])
        require(
            item is not None and item["artifact"] == ref and item["role"] == INPUT_ROLES[name],
            "input is outside its declared closure role",
        )
    expected = set(inventory) | {str(manifest_path), str(root / "inputs.json")}
    require({str(p) for p in root.iterdir()} <= expected, "undeclared file in worker input root")

    pending = [ref["path"] for name, ref in inputs.items() if name != "input_closure"]
    visited = set()

    def scan(value):
        if isinstance(value, dict):
            if "path" in value:
                require(set(value) == {"path", "sha256"}, "unlisted path-bearing metadata")
                item = inventory.get(value["path"])
                require(item is not None and item["artifact"] == value, "unlisted transitive artifact")
                pending.append(value["path"])
            else:
                for child in value.values():
                    scan(child)
        elif isinstance(value, list):
            for child in value:
                scan(child)

    # Establish the entire allowlist before opening its data, and never follow a
    # metadata reference merely because it supplies a matching checksum.
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        item = inventory[name]
        path = reference(item["artifact"])
        if item["role"] == "checkpoint":
            continue
        if item["role"] == "corpus":
            with path.open() as stream:
                for line in stream:
                    if line.strip():
                        scan(json.loads(line))
        else:
            scan(json.loads(path.read_bytes()))
    require(visited == set(inventory), "unused declared artifact in worker closure")
    return manifest
