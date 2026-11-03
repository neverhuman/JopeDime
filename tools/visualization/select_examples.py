#!/usr/bin/env python3
"""Predeclare a bounded fixture pool and select diverse observed DOPE contexts.

Selection uses only native task+distribution embeddings, never output quality,
holdout predictions, generated-code structure, or the descriptor passthrough.
All catalog, trace, selection and copied CSV artifacts are private diagnostics.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import struct
import subprocess
import sys

CATALOG_SCHEMA = "jope.embedding-candidate-catalog/v1"
TRACE_SCHEMA = "jope.visual-trace-diagnostic/v1"
RECEIPT_SCHEMA = "jope.embedding-diversity-selection/v1"
BUNDLE = "sha256:057e50a9a741d30d541ed8af8339cec84a5157af621100cb70857052ef14dacc"
RECIPES = [
    "independent Gaussian inputs; unrelated uniform target",
    "nearly collinear Gaussian inputs; linear target",
    "anticorrelated trend and squared input; step target",
    "binary inputs; XOR target with small noise",
    "sparse positive input spikes; rare-event target",
    "clipped Cauchy inputs; log and square-root target",
    "lognormal inputs; log-ratio and interaction target",
    "separated correlated Gaussian clusters; cluster-conditioned target",
    "uniform inputs; cubic and quartic target",
    "uniform angular inputs; high-frequency periodic target",
    "quantized categorical inputs; staircase target",
    "uniform inputs; heteroskedastic linear target",
    "uniform inputs; narrow rational peak target",
    "uniform inputs; four-way product target",
    "constant zero inputs; varying Gaussian target",
    "one-hot inputs; four-valued target",
    "Gaussian inputs; strongly skewed exponential target",
    "Gaussian inputs; two-regime nonlinear target",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path):
    require(path.stat().st_size <= 256 * 1024 * 1024, "JSON exceeds 256 MiB cap")
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def write_new(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def source_identity():
    script = Path(__file__).resolve()
    result = subprocess.run(["rtk", "proxy", "git", "rev-parse", "HEAD"], cwd=script.parent,
                            capture_output=True, text=True, check=True)
    return {"script_sha256": sha(script.read_bytes()), "repository_head": result.stdout.strip(),
            "argv": sys.argv, "python": platform.python_version()}


def f32(value):
    value = struct.unpack("<f", struct.pack("<f", float(value)))[0]
    require(math.isfinite(value), "fixture contains a nonfinite value")
    return value


def varied_row(recipe, i, rng):
    x = [rng.uniform(-2, 2) for _ in range(4)]
    g = lambda: rng.gauss(0, 1)
    t = (i - 47.5) / 24
    if recipe == 0:
        x = [g() for _ in range(4)]; y = rng.uniform(-20, 20)
    elif recipe == 1:
        z = g(); x = [z + .001 * g() for _ in range(4)]; y = 3 * x[0] - .1
    elif recipe == 2:
        x = [t, -t, t * t, 1.]; y = 5 if t > 0 else -5
    elif recipe == 3:
        x = [float(rng.randrange(2)) for _ in range(4)]; y = 4 * (int(x[0]) ^ int(x[1])) + .01 * g()
    elif recipe == 4:
        x = [float((i + j * 7) % 23 == 0) * (j + 1) * 10 for j in range(4)]; y = 8 * float(x[0] > 0) + .001 * g()
    elif recipe == 5:
        x = [max(-30, min(30, math.tan(math.pi * (rng.random() - .5)))) for _ in range(4)]
        y = math.log1p(abs(x[0])) - math.sqrt(abs(x[2]))
    elif recipe == 6:
        x = [math.exp(max(-5, min(5, 1.7 * g()))) for _ in range(4)]
        y = math.log(x[0]) - math.log(x[1]) + .02 * x[2] * x[3]
    elif recipe == 7:
        cluster = -1 if i % 2 else 1; z = 5 * cluster + .2 * g()
        x = [z + .1 * g(), -z + .2 * g(), cluster + .1 * g(), g()]
        y = cluster * (x[0] ** 2 + .1 * x[3])
    elif recipe == 8:
        y = x[0] ** 3 - .6 * x[1] ** 4 + x[2] * x[3]
    elif recipe == 9:
        x = [rng.uniform(-math.pi, math.pi) for _ in range(4)]
        y = math.sin(13 * x[0]) + math.cos(17 * x[1]) + .2 * math.sin(x[2] * x[3])
    elif recipe == 10:
        x = [float(rng.randrange(6)) for _ in range(4)]
        y = float(int(x[0]) // 2) - 2 * float(x[1] > 3) + .1 * x[2]
    elif recipe == 11:
        y = 2 * x[0] - x[1] + abs(x[0]) * 2 * g()
    elif recipe == 12:
        y = 1 / (.02 + x[0] ** 2) + .5 / (.03 + (x[1] - .2) ** 2)
    elif recipe == 13:
        y = 5 * math.prod(x)
    elif recipe == 14:
        x = [0.] * 4; y = g()
    elif recipe == 15:
        category = rng.randrange(4); x = [float(j == category) for j in range(4)]
        y = [-6., 2., 10., -1.][category]
    elif recipe == 16:
        x = [g() for _ in range(4)]; y = math.exp(max(-6, min(6, 2.5 * x[0])))
    else:
        x = [g() for _ in range(4)]
        y = math.sin(5 * x[0]) + x[1] ** 2 if x[3] > 0 else -3 * x[0] * x[2]
    return [f32(v) for v in (*x, y)]


def generate(root: Path, baseline: Path):
    require(not (root / "catalog.json").exists(), "refuse to replace a predeclared pool")
    candidates = []
    for index in range(24):
        batch, case = f"batch-{index // 8 + 1:02d}", f"case-{chr(97 + index % 8)}"
        path = root / batch / f"{case}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        if index < 6:
            data = (baseline / f"case-{chr(97 + index)}.csv").read_bytes()
            description = f"unchanged prior six-case baseline {chr(97 + index)}"
        else:
            rng = random.Random(20261103 + index * 1543)
            rows = [varied_row(index - 6, row, rng) for row in range(96)]
            data = ("x0,x1,x2,x3,y\n" + "".join(",".join(f"{value:.9f}" for value in row) + "\n" for row in rows)).encode()
            description = RECIPES[index - 6]
        with path.open("xb") as stream:
            stream.write(data)
        candidates.append({"id": f"pool-{index + 1:03d}", "batch": batch, "batch_case_id": case,
                           "csv_path": str(path.resolve()), "dataset_sha256": sha(data),
                           "description": description, "baseline": index < 6, "rows": 96, "features": 4})
    require(len({item["dataset_sha256"] for item in candidates}) == 24, "duplicate candidate data")
    catalog = {"schema": CATALOG_SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
               "predeclared_before_inference": True, "selection_uses_holdout_quality": False,
               "candidate_count": 24, "seed_recipe": "random.Random(20261103 + candidate_zero_based_index * 1543)",
               "source": source_identity(), "split": {"seed": 20261103, "fit_rows": 76,
               "holdout_rows": 20, "label": "public-run/fit-v1"}, "candidates": candidates}
    write_new(root / "catalog.json", catalog)
    print(json.dumps({"catalog": str(root / "catalog.json"), "candidates": 24,
                      "batches": [str(root / f"batch-{i:02d}") for i in range(1, 4)]}))


def load_embeddings(catalog, traces):
    candidates = catalog["candidates"]
    by_hash = {item["dataset_sha256"]: item for item in candidates}
    require(len(by_hash) == len(candidates), "catalog has duplicate data identities")
    embeddings, observations, trace_ids = {}, {}, []
    asset_identity = None
    for path in traces:
        data = read_json(path)
        require(data["schema"] == TRACE_SCHEMA and data["bundle_digest"] == BUNDLE, "wrong native trace route")
        assets = data["assets"]
        if asset_identity is None:
            asset_identity = assets
        require(assets == asset_identity, "checkpoint assets changed between batches")
        trace_id = {"path": str(path.resolve()), "sha256": sha(path.read_bytes()),
                    "native_source_commit": data["source_commit"], "bundle_digest": data["bundle_digest"]}
        trace_ids.append(trace_id)
        for case in data["cases"]:
            item = by_hash.get(case["dataset_sha256"])
            require(item is not None and case["id"] == item["batch_case_id"], "unregistered native data/case identity")
            require(item["id"] not in embeddings, "duplicate native candidate observation")
            require(sha(Path(item["csv_path"]).read_bytes()) == item["dataset_sha256"], "candidate bytes changed")
            require(case["rows"] == 76 and case["input_rows"] == 96 and len(case["holdout_row_ids"]) == 20,
                    "native FIT scope differs from preregistration")
            require(case["validation"]["trace_on_off_bitwise_equal"] and case["validation"]["closure_negative_control_rejected"],
                    "native observation controls failed")
            taps = [tap for tap in case["trace"]["tensors"] if tap["network"] == "dope" and tap["name"] == "dope629"]
            require(len(taps) == 1 and taps[0]["shape"] == [1, 629], "native DOPE readout unavailable")
            values = case["outputs"]["dope629"]
            require(len(values) == 629 and values == taps[0]["values"] and all(math.isfinite(v) for v in values),
                    "DOPE output differs from native readout")
            vector = values[256:512]
            require(math.fsum(v * v for v in vector) > 0, "zero learned embedding has undefined cosine distance")
            embeddings[item["id"]] = vector
            observations[item["id"]] = {"trace_sha256": trace_id["sha256"], "native_case_id": case["id"],
                "dataset_sha256": case["dataset_sha256"], "embedding_f32_sha256": sha(struct.pack("<256f", *vector)),
                "fit_row_ids": case["fit_row_ids"], "holdout_row_ids": case["holdout_row_ids"]}
    require(set(embeddings) == {item["id"] for item in candidates}, "pool has missing native observations")
    return embeddings, observations, trace_ids, asset_identity


def distances(vectors):
    norms = [math.sqrt(math.fsum(v * v for v in vector)) for vector in vectors]
    matrix = [[0.] * len(vectors) for _ in vectors]
    for a in range(len(vectors)):
        for b in range(a + 1, len(vectors)):
            cosine = math.fsum(x * y for x, y in zip(vectors[a], vectors[b])) / (norms[a] * norms[b])
            matrix[a][b] = matrix[b][a] = 1 - max(-1., min(1., cosine))
    return matrix


def spread(matrix, indices):
    values = [matrix[a][b] for i, a in enumerate(indices) for b in indices[i + 1:]]
    return {"min": min(values), "max": max(values), "mean": math.fsum(values) / len(values)}


def select(catalog_path: Path, traces: list[Path], output: Path, engine_dir: Path):
    catalog = read_json(catalog_path)
    require(catalog["schema"] == CATALOG_SCHEMA and catalog["candidate_count"] == 24, "expected predeclared 24-case catalog")
    require(catalog["predeclared_before_inference"] is True and catalog["selection_uses_holdout_quality"] is False,
            "catalog does not declare the embedding-only selection rule")
    embeddings, observations, trace_ids, assets = load_embeddings(catalog, traces)
    ids = sorted(embeddings)
    matrix = distances([embeddings[cid] for cid in ids])
    pair = min(((a, b) for a in range(len(ids)) for b in range(a + 1, len(ids))),
               key=lambda pair: (-matrix[pair[0]][pair[1]], ids[pair[0]], ids[pair[1]]))
    selected, steps = list(pair), [{"kind": "farthest_pair", "ids": [ids[i] for i in pair], "distance": matrix[pair[0]][pair[1]]}]
    while len(selected) < 6:
        chosen = min((i for i in range(len(ids)) if i not in selected),
                     key=lambda i: (-min(matrix[i][j] for j in selected), ids[i]))
        steps.append({"kind": "farthest_point", "id": ids[chosen],
                      "minimum_distance_to_previously_selected": min(matrix[chosen][j] for j in selected)})
        selected.append(chosen)
    by_id = {item["id"]: item for item in catalog["candidates"]}
    baseline = [i for i, cid in enumerate(ids) if by_id[cid]["baseline"]]
    require(len(baseline) == 6, "expected six registered baseline cases")
    engine_dir.mkdir(parents=True, exist_ok=True)
    chosen_records = []
    for alias, index in zip("abcdef", selected):
        item = by_id[ids[index]]
        destination = engine_dir / f"case-{alias}.csv"
        data = Path(item["csv_path"]).read_bytes()
        require(sha(data) == item["dataset_sha256"], "selected CSV bytes changed")
        with destination.open("xb") as stream:
            stream.write(data)
        chosen_records.append({**item, "engine_case_id": f"case-{alias}",
                               "engine_csv_path": str(destination.resolve()), "observation": observations[item["id"]]})
    receipt = {"schema": RECEIPT_SCHEMA, "catalog_sha256": sha(catalog_path.read_bytes()),
               "selection_source": source_identity(), "traces": trace_ids, "assets": assets,
               "embedding": {"native_tensor": "DOPE629", "slice_half_open": [256, 512],
               "dimensions": 256, "meaning": "learned task128 plus distribution128; excludes copied raw descriptor bypass"},
               "distance": "1 - cosine; deterministic math.fsum accumulation in Python f64",
               "algorithm": "farthest pair, then farthest point by max-min distance; lexicographic ID tie-break",
               "pool_ids": ids, "pool_pairwise_cosine_distances": matrix, "steps": steps,
               "pool_spread": spread(matrix, list(range(len(ids)))), "baseline_spread": spread(matrix, baseline),
               "selected_spread": spread(matrix, selected), "selected": chosen_records,
               "selected_pairwise_cosine_distances": [[matrix[a][b] for b in selected] for a in selected],
               "scope": "most diverse greedy subset of this tested 24-case pool; not a global optimum, quality result, performance claim or guarantee of different generated algorithms",
               "holdout_quality_used_for_selection": False, "all_24_candidates_observed": True}
    write_new(output, receipt)
    print(json.dumps({"selection": str(output), "selected_ids": [ids[i] for i in selected],
                      "baseline_spread": receipt["baseline_spread"], "selected_spread": receipt["selected_spread"],
                      "pool_spread": receipt["pool_spread"], "engine_csv_directory": str(engine_dir)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--root", type=Path, required=True)
    gen.add_argument("--baseline", type=Path, required=True)
    choose = sub.add_parser("select")
    choose.add_argument("--catalog", type=Path, required=True)
    choose.add_argument("--trace", type=Path, action="append", required=True)
    choose.add_argument("--output", type=Path, required=True)
    choose.add_argument("--engine-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate":
        generate(args.root, args.baseline)
    else:
        select(args.catalog, args.trace, args.output, args.engine_dir)


if __name__ == "__main__":
    main()
