#!/usr/bin/env python3
"""Read-only extraction of the real training series used by the deep-stack chapter.

F1  fixed-panel monitor across the A100 lineage: every parallel.frozen_monitoring.v1 record in
    the retained learning-receipts monitoring directory (panel fbff7a04, corpus 11e60a43,
    2,048 sampled examples), one row per record.
F3  A100 throughput and timing: per-step examples_per_second, end_to_end_examples_per_second,
    step_seconds, communication_seconds from the 21 original metrics.jsonl logs, as
    non-overlapping 100-step means plus per-block medians.

Every output row carries its source path and SHA-256; a manifest pins every input. Nothing is
written outside --out, and no sealed, meta_test or test path is opened.
"""
import argparse
import csv
import hashlib
import json
import pathlib
import statistics
import sys

MONITOR = pathlib.Path("/home/ubuntu/jain-split/jope-dime/parallel/recovery/eight-a100-20260912/"
                       "learning-receipts/learning/monitoring")
A100 = pathlib.Path("/home/ubuntu/jope-migration-evidence/claude/node-high-value/files/home/ubuntu/"
                    "jope-dime/parallel/runs/eight-a100-recovery-20260912/learning")
FORBIDDEN = ("sealed", "meta_test", "meta-test", "test.csv", "test_features", "test_labels")
TIMING = ("examples_per_second", "end_to_end_examples_per_second", "step_seconds",
          "communication_seconds")


def read(path):
    s = str(path).lower()
    if any(w in s for w in FORBIDDEN):
        sys.exit(f"refused forbidden path {path}")
    raw = pathlib.Path(path).read_bytes()
    return raw, hashlib.sha256(raw).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = pathlib.Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "deep-stack.extraction/v1", "inputs": []}

    # F1
    rows = []
    for f in sorted(MONITOR.glob("*.json")):
        raw, sha = read(f)
        o = json.loads(raw)
        assert o["schema"] == "parallel.frozen_monitoring.v1"
        assert o["final_tests_opened"] is False and o["training_selection_used"] is False
        manifest["inputs"].append({"path": str(f), "sha256": sha})
        rows.append({"evaluated_step": o["evaluated_step"], "checkpoint": o["checkpoint"],
                     "checkpoint_sha256": o["binding"]["checkpoint_sha256"],
                     "executable_sha256": o["binding"]["executable_sha256"],
                     "panel_sha256": o["binding"]["panel_sha256"],
                     "corpus_sha256": o["binding"]["corpus_sha256"],
                     "sampled_examples": o["sampled_examples"], "timestamp": o["timestamp"],
                     "sequence_nll": repr(o["sequence_nll"]), "ranking_bt": repr(o["ranking_bt"]),
                     "ranking_listwise": repr(o["ranking_listwise"]),
                     "runtime_huber": repr(o["runtime_huber"]),
                     "evaluation_bias": o["evaluation_bias"],
                     "independent_evaluation": o["independent_evaluation"],
                     "source_path": str(f), "source_sha256": sha})
    rows.sort(key=lambda r: (r["evaluated_step"], r["executable_sha256"]))
    assert len({r["panel_sha256"] for r in rows}) == 1 and len({r["corpus_sha256"] for r in rows}) == 1
    with open(out / "a100-fixed-monitor.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    manifest["f1"] = {"records": len(rows), "steps": sorted({r["evaluated_step"] for r in rows}),
                      "panel_sha256": rows[0]["panel_sha256"], "corpus_sha256": rows[0]["corpus_sha256"],
                      "executables": sorted({r["executable_sha256"] for r in rows})}

    # F3
    bins, blocks = [], []
    for cycle in range(1, 22):
        path = A100 / f"cycle-{cycle:06d}" / "train-2" / "metrics.jsonl"
        raw, sha = read(path)
        manifest["inputs"].append({"path": str(path), "sha256": sha})
        recs = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        recs = [r for r in recs if "step" in r and all(k in r for k in TIMING)]
        assert len(recs) == 3000, (cycle, len(recs))
        assert all(r["world"] == 8 and r["global_batch"] == 2048 for r in recs)
        for i in range(0, 3000, 100):
            chunk = recs[i:i + 100]
            row = {"cycle": cycle, "first_step": chunk[0]["step"], "last_step": chunk[-1]["step"]}
            for k in TIMING:
                row[k] = repr(sum(float(r[k]) for r in chunk) / len(chunk))
            row["source_sha256"] = sha
            bins.append(row)
        block = {"cycle": cycle, "first_step": recs[0]["step"], "last_step": recs[-1]["step"],
                 "corpus_sha256": recs[0].get("corpus_sha256")}
        for k in TIMING:
            block[f"median_{k}"] = repr(statistics.median(float(r[k]) for r in recs))
        block["source_path"] = str(path)
        block["source_sha256"] = sha
        blocks.append(block)
    for name, data in (("a100-throughput-bins.csv", bins), ("a100-throughput-blocks.csv", blocks)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    manifest["f3"] = {"bins": len(bins), "blocks": len(blocks),
                      "overall_median_of_block_medians": {
                          k: statistics.median(float(b[f"median_{k}"]) for b in blocks) for k in TIMING}}
    (out / "extraction-manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "inputs"}, indent=1))


if __name__ == "__main__":
    main()
