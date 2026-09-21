#!/usr/bin/env python3
"""Freeze membership and known contamination limits from preparation metadata."""

import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path

from promotion_gate import immutable
from resume_expired_cycle import read, require, sha


def freeze(root, contexts, output):
    index = read(contexts)
    rows = {}
    preparations = {}
    families = defaultdict(set)
    sources = defaultdict(set)
    for context in index["encodings"]:
        name = context["prepared"]
        path = (root / name / "prepared.json").resolve()
        require(path.is_relative_to(root), "prepared metadata outside workspace")
        if name not in rows:
            require(sha(path) == context["prepared_sha256"], "prepared metadata identity changed")
            prepared = read(path)
            preparations[name] = prepared
            registry = prepared.get("family_registry_binding", {})
            reasons = []
            if not prepared.get("family_partition_frozen"):
                reasons.append("family partition is provisional")
            if not registry.get("full_inventory_complete"):
                reasons.append("family inventory is incomplete")
            if not prepared.get("clean_evaluation"):
                reasons.append("clean evaluation is not established")
            if "unqualified" in prepared.get("clean_status", ""):
                reasons.append(prepared["clean_status"])
            source = prepared.get("storage", {}).get("source_sha256")
            if not source:
                reasons.append("original source digest is absent")
            row = dict(
                prepared=name,
                prepared_sha256=context["prepared_sha256"],
                dataset=prepared["dataset"],
                family=prepared["family"],
                partition=prepared["meta_partition"],
                source_sha256=source,
                family_registry_binding=registry,
                clean_status=prepared.get("clean_status"),
                independent_benchmark_eligible=not reasons,
                limitations=reasons,
                cuts=[],
            )
            rows[name] = row
        row = rows[name]
        require(
            (row["dataset"], row["family"], row["partition"])
            == (context["dataset"], context["family"], context["partition"]),
            "context family/partition conflicts with preparation",
        )
        cut = next(c for c in preparations[name]["cuts"] if c["index"] == context["cut"])
        require(
            not set(cut["train_indices"]) & set(cut["validation_indices"]),
            "training/evaluation rows overlap within cut",
        )
        row["cuts"].append(
            {k: cut[k] for k in ("index", "training_sha256", "validation_sha256", "preprocessing_sha256")}
        )
        families[row["family"]].add(row["partition"])
        if row["source_sha256"]:
            sources[row["source_sha256"]].add((row["family"], row["partition"]))
    conflicts = [
        dict(kind="family crosses partitions", family=f, partitions=sorted(p))
        for f, p in families.items()
        if len(p) > 1
    ]
    conflicts.extend(
        dict(kind="source shared across families or partitions", source_sha256=s, members=sorted(m))
        for s, m in sources.items()
        if len(m) > 1
    )
    value = dict(
        schema="parallel.family_contamination_inventory.v1",
        timestamp=time.time(),
        context_index_sha256=sha(contexts),
        prepared=list(rows.values()),
        families=len(families),
        partitions=dict(Counter(row["partition"] for row in rows.values())),
        known_conflicts=conflicts,
        unresolved_memberships=sum(bool(row["limitations"]) for row in rows.values()),
        scope="Declared prepared/context membership and source-digest overlap; undisclosed ancestry or semantic overlap remains unresolved.",
        csv_or_target_values_opened=False,
        final_tests_opened=False,
    )
    immutable(output, value)
    return value


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "contexts", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    value = freeze(a.root.resolve(), a.contexts.resolve(), a.output.resolve())
    print({k: v for k, v in value.items() if k != "prepared"})
