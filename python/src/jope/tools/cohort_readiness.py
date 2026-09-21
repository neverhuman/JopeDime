#!/usr/bin/env python3
"""Count completed selected-family GP views using compact durable receipts only."""

import argparse
import hashlib
import json
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    root = args.root
    selection_path = root / "manifests/generated/cohort64-qualification-v1.json"
    selection = read(selection_path)
    assert selection["qualification_only"] and len(selection["selected"]) == 64
    by_prepared, groups = {}, {}
    for c in selection["selected"]:
        groups[c["family"]] = {k: c[k] for k in ["dataset", "family", "partition"]}
        groups[c["family"]].update(completed_views=0, completed_evaluations=0, pending_views=0)
        groups[c["family"]]["completed"] = []
        for p in c["prepared_versions"]:
            assert digest(root / p["path"]) == p["sha256"]
            by_prepared[str(Path(p["path"]).parent)] = c
    known, pending, errors = set(), {}, []
    for binding in selection["job_manifests"]:
        path = root / binding["path"]
        assert digest(path) == binding["sha256"]
        for job in read(path)["jobs"]:
            if job["prepared"] not in by_prepared or job["output"] in known:
                continue
            known.add(job["output"])
            c = by_prepared[job["prepared"]]
            group = groups[c["family"]]
            summary_path = root / job["output"] / "search.json"
            progress_path = root / job["output"] / "progress.json"
            if summary_path.exists():
                try:
                    summary, progress = read(summary_path), read(progress_path)
                    assert summary["schema"] == "parallel.search.v1"
                    assert progress["prepared_sha256"] == job["prepared_sha256"]
                    assert progress["cut"] == job["cut"]
                    assert summary.get("final_tests_opened") is False
                    assert summary["vocabulary_sha256"] == selection["vocabulary"]["sha256"]
                    count = summary["completed_evaluations"]
                    assert isinstance(count, int) and count >= 0
                    group["completed_views"] += 1
                    group["completed_evaluations"] += count
                    group["completed"].append(
                        {
                            "cut": job["cut"],
                            "output": job["output"],
                            "completed_evaluations": count,
                            "summary_sha256": digest(summary_path),
                            "progress_sha256": digest(progress_path),
                        }
                    )
                    continue
                except (AssertionError, KeyError, OSError, ValueError) as error:
                    errors.append(
                        {"output": job["output"], "reason": str(error) or "receipt identity mismatch"}
                    )
            group["pending_views"] += 1
            pending.setdefault(c["family"], []).append(job)
    ordered = sorted(groups.values(), key=lambda g: (g["partition"], g["family"]))
    counts = {}
    for partition in ["meta_train", "meta_validation", "meta_test"]:
        subset = [g for g in ordered if g["partition"] == partition]
        counts[partition] = {
            "selected_families": len(subset),
            "families_with_completed_view": sum(g["completed_views"] > 0 for g in subset),
            "families_with_two_completed_views": sum(g["completed_views"] >= 2 for g in subset),
            "completed_views": sum(g["completed_views"] for g in subset),
            "completed_evaluations": sum(g["completed_evaluations"] for g in subset),
        }
    priority = []
    for partition in ["meta_validation", "meta_train", "meta_test"]:
        needed = []
        for g in ordered:
            if g["partition"] != partition:
                continue
            candidates = sorted(pending.get(g["family"], []), key=lambda j: (j["cut"], j["output"]))
            needed.append(candidates[: max(0, 2 - g["completed_views"])])
        for index in range(2):
            priority.extend(rows[index] for rows in needed if len(rows) > index)
    priority_path = root / "runs/cohort64-bootstrap-priority-jobs.json"
    save(
        priority_path,
        {
            "schema": "parallel.gp_jobs.v1",
            "qualification": True,
            "vocabulary": selection["vocabulary"]["path"],
            "jobs": priority,
        },
    )
    result = {
        "schema": "parallel.cohort_bootstrap_readiness.v1",
        "timestamp": int(time.time()),
        "selection_sha256": digest(selection_path),
        "qualification_only": True,
        "compact_receipt_ready": counts["meta_train"]["families_with_completed_view"] == 48
        and counts["meta_validation"]["families_with_completed_view"] == 8,
        "requires_corpus_validation": True,
        "limitation": "Completed search receipts establish work completion; finite reward groups and usable programs require the subsequent corpus validation.",
        "partitions": counts,
        "families": ordered,
        "invalid_receipts": errors,
        "pending_first_two_views": {
            "path": str(priority_path.relative_to(root)),
            "sha256": digest(priority_path),
            "jobs": len(priority),
        },
        "priority_enqueued": False,
        "full_observation_ledgers_read": False,
        "original_final_test_targets_opened": False,
    }
    out = root / "runs/cohort64-bootstrap-readiness.json"
    save(out, result)
    print(
        json.dumps(
            {
                "receipt": str(out),
                "partitions": counts,
                "priority_jobs": len(priority),
                "invalid_receipts": len(errors),
                "compact_receipt_ready": result["compact_receipt_ready"],
            }
        )
    )


if __name__ == "__main__":
    main()
