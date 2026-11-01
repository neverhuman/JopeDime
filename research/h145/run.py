#!/usr/bin/env python3
"""Execute the merged H145 protocol using source-bound, TRAIN-only frozen outputs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = Path("/home/ubuntu/jope-data/h145")
EVIDENCE = Path("/home/ubuntu/jope_plan/evidence/H145")
PREREG_MERGE = "6fc5cd8c74736cc91d32a5fb9f5d36a72fc3f6e9"
BUCKET = "veox-jopedime-learning-use2"
PREFIX = "experiments/rd_breakthrough_20261030/"
DOPE = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
SELECTED = Path("/home/ubuntu/jope-data/h50/input/selected.json")
SELECTED_SHA = "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1"
CHECKPOINTS = {
    "R782": ("427b0d4608e94f60f5f97d79b5c1c4ce9a7efd59d17e7853f714e901f3a4af0c", None),
    "R785": (
        "ef6696f6369122a162d58b9b522fd83cae110a5576ef3760e3b43ba85d129e0a",
        "/home/ubuntu/jope-data/h50/ckpt/round-785.pt",
    ),
    "R859": ("5802245f447b1852610576cde08b8a299a9e1e99eb82991aac63c958284fa715", None),
    "R882": (
        "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64",
        "/home/ubuntu/jope-data/h51/ckpt/round-882.pt",
    ),
    "LATEST": (
        "f566a2f08ef215f5afc328c0d94374498a06d4f9a8c35b07ae5f4d4ebd6249a6",
        None,
    ),
}
ROUNDS = {"R782": 782, "R785": 785, "R859": 859, "R882": 882, "LATEST": 2555}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def value_end(text, start):
    if text[start] not in '{["':
        end = start
        while end < len(text) and text[end] not in ",}]":
            end += 1
        return end
    quoted, depth, end = text[start] == '"', 0 if text[start] == '"' else 1, start + 1
    while end < len(text):
        char = text[end]
        if quoted:
            if char == "\\":
                end += 2
                continue
            if char == '"':
                quoted = False
                if not depth:
                    return end + 1
        elif char == '"':
            quoted = True
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
            if not depth:
                return end + 1
        end += 1
    raise ValueError("incomplete JSON value")


def raw_field(text, wanted):
    """Return one raw subtree. Unrequested values, including DL-Val, are never decoded."""
    pos = 0
    while text[pos].isspace():
        pos += 1
    assert text[pos] == "{"
    pos += 1
    while True:
        while text[pos].isspace() or text[pos] == ",":
            pos += 1
        if text[pos] == "}":
            raise KeyError(wanted)
        end = value_end(text, pos)
        key = json.loads(text[pos:end])
        pos = end
        while text[pos].isspace():
            pos += 1
        assert text[pos] == ":"
        pos += 1
        while text[pos].isspace():
            pos += 1
        end = value_end(text, pos)
        if key == wanted:
            return text[pos:end]
        pos = end


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def guard(start):
    now = datetime.now(timezone.utc)
    if (
        now >= datetime(2026, 11, 1, 3, 45, tzinfo=timezone.utc)
        or time.time() - start > 7200
    ):
        raise RuntimeError("H145 wall/deadline guard")
    if os.getloadavg()[0] > 0.85 * len(os.sched_getaffinity(0)):
        raise RuntimeError("H145 host CPU guard")
    usage, children = (
        resource.getrusage(resource.RUSAGE_SELF),
        resource.getrusage(resource.RUSAGE_CHILDREN),
    )
    if (
        usage.ru_maxrss * 1024 > 6_000_000_000
        or usage.ru_utime + usage.ru_stime + children.ru_utime + children.ru_stime
        > 8 * 3600
    ):
        raise RuntimeError("H145 RSS/CPU budget guard")
    if sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file()) > 500_000_000:
        raise RuntimeError("H145 scratch guard")


def main():
    assert os.getpriority(os.PRIO_PROCESS, 0) >= 10
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", PREREG_MERGE, "HEAD"],
        cwd=REPO,
        check=True,
    )
    prereg = subprocess.check_output(
        ["git", "show", PREREG_MERGE + ":research/h145/PREREG.md"], cwd=REPO
    )
    assert prereg == (HERE / "PREREG.md").read_bytes()
    ROOT.mkdir(exist_ok=True)
    (ROOT / "inputs").mkdir(exist_ok=True)
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    started = time.time()
    start_record = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "started_epoch": started,
        "source_commit": source,
        "prereg_merge": PREREG_MERGE,
        "prereg_sha256": sha(prereg),
        "runner_sha256": sha(Path(__file__).read_bytes()),
        "statistics_sha256": sha((HERE / "analysis.py").read_bytes()),
        "hard_deadline": "2026-11-01T03:45:00Z",
        "wall_budget_seconds": 7200,
    }
    with (ROOT / "RUN_STARTED.json").open("x") as stream:
        json.dump(start_record, stream, indent=2)
    guard(started)
    stats = module("h145_statistics", HERE / "analysis.py")
    raw = SELECTED.read_bytes()
    assert sha(raw) == SELECTED_SHA
    rows = json.loads(raw_field(raw.decode(), "train"))
    catalog_path = Path("/home/ubuntu/jope_plan/master/source/S3_DATASET_CATALOG.json")
    catalog_raw = catalog_path.read_bytes()
    catalog_ids = {r["id"] for r in json.loads(catalog_raw)["datasets"]}
    assert len(rows) == 1219 and len({r["family_id"] for r in rows}) == 205
    assert all(
        r["dl_split"] == "DL-Train"
        and r["source_split"] in ("train", "inner")
        and r["task"] == "regression"
        and r["dataset_id"] in catalog_ids
        for r in rows
    )
    folds = stats.family_folds(rows)
    write(ROOT / "assessment-roster.json", rows)
    write(
        ROOT / "family-folds.json", folds
    )  # frozen before score-object downloads/decoding
    input_receipts = []

    def get(key, expected, name):
        guard(started)
        dest = ROOT / "inputs" / name
        assert not dest.exists()
        response = subprocess.check_output(
            [
                "aws",
                "s3api",
                "get-object",
                "--profile",
                "veox-jepson-prod",
                "--bucket",
                BUCKET,
                "--key",
                PREFIX + key,
                str(dest),
            ],
            text=True,
        )
        data = dest.read_bytes()
        assert sha(data) == expected, key
        metadata = json.loads(response)
        input_receipts.append(
            {
                "key": PREFIX + key,
                "sha256": expected,
                "version_id": metadata.get("VersionId"),
                "bytes": len(data),
            }
        )
        write(ROOT / "input-readbacks.json", input_receipts)
        return data

    # A metadata-only manifest was inventoried before registration. These exact hashes bind cached outputs.
    h139 = json.loads((EVIDENCE / "source-custody-0.json").read_text())
    h143 = json.loads((EVIDENCE / "source-custody-1.json").read_text())
    assert h139["source_commit"] == "78f9d9c1de25bbce49978edff8b2a97eade786ef"
    assert h143["source_commit"] == "77446bbcef958bacc6c907d0265f48c31b2a8dd9"
    assert (
        h139["files"]["per_context.json"]["sha256"]
        == "86c6a5623ed5aa1636be0cc8f69f5befb29f1b6262b5e3dfeb5cf5f5a2b9c194"
    )
    assert (
        h143["files"]["per_incumbent.json"]["sha256"]
        == "c444a67d6fcd8a06d5f02f46be988cf84a7b847586a1cd6dd8a8698b19c8eec1"
    )
    states, checkpoints = {}, {}
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for arm, (expected, existing) in CHECKPOINTS.items():
        if existing:
            path = Path(existing)
        elif arm == "LATEST":
            path = ROOT / "inputs/LATEST-R2555.pt"
            with path.open("xb") as stream:
                subprocess.run(
                    [
                        "ssh",
                        "-o",
                        "BatchMode=yes",
                        "xbabe1",
                        "cat /home/ubuntu/jope-data/h145/inputs/LATEST-R2555.pt",
                    ],
                    stdout=stream,
                    check=True,
                )
        else:
            get(f"H139/opus-h139-inputs-20261031/{arm}.pt", expected, arm + ".pt")
            path = ROOT / "inputs" / (arm + ".pt")
        assert sha(path.read_bytes()) == expected
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        assert checkpoint["dope_sha256"] == DOPE
        states[arm] = checkpoint["state_dict"]
        checkpoints[arm] = {"sha256": expected, "path": str(path)}
        guard(started)

    def tensor_bytes(tensor):
        return (
            tensor.detach()
            .cpu()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )

    ref = states["R882"]
    aliases, digest_groups = {}, {}
    parameter_receipts = {}
    for arm in ["R882", "R782", "R785", "R859", "LATEST"]:
        state = states[arm]
        assert set(state) == set(ref)
        h = hashlib.sha256()
        non_adapter = 0
        for name in sorted(state):
            value, baseline = state[name], ref[name]
            assert value.shape == baseline.shape and value.dtype == baseline.dtype
            data = tensor_bytes(value)
            if not name.startswith("adapter_10tok."):
                assert data == tensor_bytes(baseline), (arm, name)
                non_adapter += 1
            h.update(
                json.dumps([name, str(value.dtype), list(value.shape)]).encode() + data
            )
        digest = h.hexdigest()
        aliases[arm] = digest_groups.setdefault(digest, arm)
        parameter_receipts[arm] = {
            "state_sha256": digest,
            "non_adapter_tensors_equal": non_adapter,
            "dope": DOPE,
        }
    write(
        ROOT / "checkpoint-identity.json",
        {
            "checkpoints": checkpoints,
            "parameters": parameter_receipts,
            "aliases": aliases,
        },
    )
    if aliases["LATEST"] == "LATEST":
        raise RuntimeError(
            "LATEST differs: registered fresh CPU forward required; never substitute cached R882"
        )
    del states, ref, checkpoint
    guard(started)

    raw139 = get(
        "H139/opus-h139-20261101a/per_context.json",
        h139["files"]["per_context.json"]["sha256"],
        "h139-mixed-custody.json",
    ).decode()
    assert json.loads(raw_field(raw139, "train_cids")) == [
        r["context_sha256"] for r in rows
    ]
    raw_train = raw_field(raw139, "native_train")
    values = {
        arm: np.asarray(json.loads(raw_field(raw_train, arm)), dtype=np.float64)
        for arm in ("R782", "R859", "R882")
    }
    raw143 = get(
        "H143/opus-h143-20261101a/per_incumbent.json",
        h143["files"]["per_incumbent.json"]["sha256"],
        "h143-mixed-custody.json",
    ).decode()
    values["R785"] = np.asarray(
        json.loads(raw_field(raw_field(raw143, "785"), "train")), dtype=np.float64
    )
    np.testing.assert_array_equal(
        values["R882"], json.loads(raw_field(raw_field(raw143, "882"), "train"))
    )
    values["LATEST"] = values[aliases["LATEST"]].copy()
    assert all(v.shape == (1219,) and np.isfinite(v).all() for v in values.values())
    for arm in ("R782", "R785", "R859", "R882"):
        manifest, prefix = (
            (h143, "H143/opus-h143-20261101a/")
            if arm == "R785"
            else (h139, "H139/opus-h139-20261101a/")
        )
        load_name, dope_name = (
            f"{arm}/checkpoint-load.json",
            f"{arm}/dope-secondary.json",
        )
        load = json.loads(
            get(
                prefix + load_name,
                manifest["files"][load_name]["sha256"],
                arm + "-load.json",
            )
        )
        dope = json.loads(
            get(
                prefix + dope_name,
                manifest["files"][dope_name]["sha256"],
                arm + "-dope.json",
            )
        )
        assert (
            load["checkpoint_sha256"] == CHECKPOINTS[arm][0] and not load["unexpected"]
        )
        assert dope["before"] == dope["after"] == DOPE
    guard(started)
    result, arrays = stats.analyze(rows, values, aliases, ROUNDS)
    elo = module(
        "h145_canonical_elo", REPO / "python/src/jope/north_star/elo_tracker.py"
    )
    records = [
        elo.DatasetEvalRecord(
            dataset_id=row["context_sha256"],
            family_id=row["family_id"],
            task_type="regression",
            scores_by_contender={
                a: float(values[a][i]) for a in result["unique_functions"]
            },
        )
        for i, row in enumerate(rows)
    ]
    canonical = elo.compute_family_grouped_bradley_terry_elo(
        records, anchor_contender="R882", prior=0.01
    )["ratings"]
    assert all(
        round(result["arms"][a]["elo"], 2) == canonical[a]
        for a in result["unique_functions"]
    )
    old = {
        "R782": 1.1073772627425547,
        "R785": 1.137595,
        "R859": 1.1377941898164525,
        "R882": 1.1377941898164525,
    }
    old["LATEST"] = old[aliases["LATEST"]]

    def rank(vector):
        v = np.asarray(vector)
        return (
            1
            + (v[None, :] > v[:, None]).sum(1)
            + 0.5 * ((v[None, :] == v[:, None]).sum(1) - 1)
        )

    unique = result["unique_functions"]
    a, b = (
        rank([old[k] for k in unique]),
        rank([result["arms"][k]["equal_family_mean_utility"] for k in unique]),
    )
    result["published_91_context_comparison"] = {
        "aggregate_values": old,
        "R785_value_precision": "published rounded 1.137595; no raw DL-Val reads",
        "ranking": sorted(old, key=lambda k: (-old[k], ROUNDS[k])),
        "spearman_unique_functions": float(np.corrcoef(a, b)[0, 1])
        if np.ptp(a) and np.ptp(b)
        else None,
        "source": "merged H139/H143 published aggregate results; old context-weighted vs new equal-family estimands differ",
    }
    np.savez_compressed(ROOT / "analysis-arrays.npz", **arrays)
    write(
        ROOT / "per-context-train.json",
        {
            "context_sha256": [r["context_sha256"] for r in rows],
            "families": [r["family_id"] for r in rows],
            "values": {a: v.tolist() for a, v in values.items()},
        },
    )
    guard(started)
    result["execution"] = dict(
        start_record,
        completed_at=datetime.now(timezone.utc).isoformat(),
        host=platform.node(),
        python=platform.python_version(),
        numpy=np.__version__,
        torch=torch.__version__,
        wall_seconds=time.time() - started,
        cpu_seconds=time.process_time(),
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        nice=os.getpriority(os.PRIO_PROCESS, 0),
        math_threads=1,
        CUDA_initialized=torch.cuda.is_initialized(),
        optimizer_updates=0,
        DOPE_modified=False,
        DLVal_outcome_subtrees_decoded=0,
        sealed_holdout_payload_reads=0,
        cache_replayed=True,
        new_model_forwards=0,
        catalog_sha256=sha(catalog_raw),
        selected_sha256=SELECTED_SHA,
        canonical_elo_point_check="PASS",
        aliases=aliases,
        checkpoint_parameters=parameter_receipts,
    )
    write(ROOT / "metrics.json", result)
    assert (ROOT / "metrics.json").stat().st_size < 100_000
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "proposal": result["proposal"],
                "ranking": result["ranking"],
                "aliases": aliases,
                "gates": result["decision_gates"],
            }
        )
    )


if __name__ == "__main__":
    main()
