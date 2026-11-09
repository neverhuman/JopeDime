#!/usr/bin/env python3
"""Measure raw FIT-cell sensitivity of the frozen H256 proposal margin."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
from statistics import median
import struct
import subprocess

from round_trip_metrics import ResidentInference, case_catalog, read_json, require, sha

SCHEMA = "jain.neural-cell-influence/v1"
ROWS = 17
COLUMNS = 5
FIRST_PROPOSAL = 4
SLATE = 12
LANDMARK_COMMIT = "d002d68112e611b6ee9ac3a722c47631669f963d"
LANDMARK_PATH = "crates/jope-service/src/ranker/landmarks.rs"
DRIVER = r"""#[path = "landmarks.rs"]
mod landmarks;

use std::io::{self, Read, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = io::stdin().lock();
    let mut output = io::stdout().lock();
    let mut header = [0u8; 8];
    while input.read(&mut header[..1])? != 0 {
        input.read_exact(&mut header[1..])?;
        let rows = u32::from_le_bytes(header[..4].try_into()?) as usize;
        let columns = u32::from_le_bytes(header[4..].try_into()?) as usize;
        if rows == 0 || rows > 1024 || columns != 4 {
            return Err("unexpected FIT dimensions".into());
        }
        let x_count = rows * columns;
        let mut bytes = vec![0u8; 4 * (x_count + rows)];
        input.read_exact(&mut bytes)?;
        let values: Vec<f32> = bytes.chunks_exact(4)
            .map(|cell| f32::from_le_bytes(cell.try_into().unwrap()))
            .collect();
        if values.iter().any(|v| !v.is_finite()) {
            return Err("non-finite FIT value".into());
        }
        let bundle = landmarks::fit_transform_and_extract_bundle_high(
            &values[..x_count], rows, columns, &values[x_count..],
        )?;
        for value in bundle.vector().iter().chain(bundle.columns().iter().flatten()) {
            output.write_all(&value.to_le_bytes())?;
        }
        output.flush()?;
    }
    Ok(())
}
"""


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def create(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)


def relative_file(root, name):
    path = (root / name).resolve(strict=True)
    require(not Path(name).is_absolute() and path.is_relative_to(root) and path.is_file(),
            "artifact path escapes its root")
    return path


def bounded_matrix(values, columns, label):
    require(isinstance(values, list) and len(values) == ROWS
            and all(isinstance(row, list) and len(row) == columns for row in values),
            f"{label}: matrix dimensions differ")
    return values


def proposal_margin(costs, winner):
    require(len(costs) == SLATE and all(math.isfinite(value) for value in costs)
            and FIRST_PROPOSAL <= winner < SLATE, "invalid final proposal costs")
    return min(costs[i] for i in range(FIRST_PROPOSAL, SLATE) if i != winner) - costs[winner]


def signed_effects(base, costs, winner):
    baseline = proposal_margin(base, winner)
    margin = proposal_margin(costs, winner) - baseline
    winner_cost = costs[winner] - base[winner]
    counterfactual = min(range(FIRST_PROPOSAL, SLATE), key=costs.__getitem__)
    require(math.isfinite(margin) and math.isfinite(winner_cost), "non-finite score delta")
    return margin, winner_cost, counterfactual


def scores_of(deltas):
    largest = max(abs(value) for row in deltas for value in row)
    return largest, [[abs(value) / largest if largest else 0. for value in row] for row in deltas]


class NativeLandmarks:
    """One resident CPU process linked only to the archived native landmark module."""

    def __init__(self, work, rustc, rlib, revision):
        self.work = work
        work.mkdir()
        repo = Path(__file__).resolve().parents[2]
        require(re.fullmatch("[0-9a-f]{40}", revision) is not None, "need exact landmark revision")
        source_command = ["git", "-C", str(repo), "show", f"{revision}:{LANDMARK_PATH}"]
        source = subprocess.run(source_command, check=True, capture_output=True, timeout=15).stdout
        create(work / "landmarks.rs", source)
        create(work / "driver.rs", DRIVER.encode())
        binary = work / "landmark-input"
        argv = [str(rustc), "--edition=2021", "-O", "-C", "lto=fat", "-C", "codegen-units=1",
                "--extern", f"feat_math={rlib}", "-L", f"dependency={rlib.parent}",
                str(work / "driver.rs"), "-o", str(binary)]
        build = subprocess.run(argv, capture_output=True, timeout=90)
        create(work / "build.stdout", build.stdout)
        create(work / "build.stderr", build.stderr)
        require(build.returncode == 0, f"native landmark build failed: {work / 'build.stderr'}")
        self.receipt = {
            "revision": revision, "source_command": source_command, "landmarks_sha256": digest(source),
            "driver_sha256": digest(DRIVER.encode()), "binary_sha256": sha(binary),
            "feat_math_rlib_sha256": sha(rlib),
            "command": argv, "device": "cpu",
            "rustc_version": subprocess.check_output([rustc, "-vV"], text=True, timeout=10),
        }
        self.log = (work / "driver.stderr").open("xb")
        try:
            self.process = subprocess.Popen([binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=self.log)
        except BaseException:
            self.log.close()
            raise

    def encode(self, raw_x, raw_y):
        nx = raw_x.astype("<f4", copy=False)
        ny = raw_y.astype("<f4", copy=False)
        frame = struct.pack("<II", *nx.shape) + nx.tobytes() + ny.tobytes()
        self.process.stdin.write(frame)
        self.process.stdin.flush()
        output = self.process.stdout.read(4 * (256 + 1088))
        require(len(output) == 4 * (256 + 1088), "native landmarks ended before returning a complete frame")
        v, col = output[:1024], output[1024:]
        return {
            "v256": struct.unpack("<256f", v), "col64x17": struct.unpack("<1088f", col),
            "v256_f32_sha256": digest(v), "col64x17_f32_sha256": digest(col),
            "raw_fit_x_f32_sha256": digest(nx.tobytes()), "raw_fit_y_f32_sha256": digest(ny.tobytes()),
        }

    def close(self):
        try:
            self.process.stdin.close()
            require(self.process.wait(timeout=10) == 0, f"native helper failed: {self.work / 'driver.stderr'}")
        finally:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait()
            self.process.stdout.close()
            self.log.close()


def state_hash(module):
    hashed = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        header = json.dumps([name, list(tensor.shape), str(tensor.dtype)]).encode()
        hashed.update(struct.pack("<I", len(header)))
        hashed.update(header)
        hashed.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return hashed.hexdigest()


def frozen_state(resident):
    return {
        "problem_encoder": state_hash(resident.model),
        "base_ranker": state_hash(resident.base),
        "g_folds": [state_hash(model) for model in resident.bridges],
    }


def table_context(case, original, model, trace_root):
    cid = case["id"]
    source = case["source"]
    proof = None
    path = original / f"{cid}.csv"
    split_path = original / cid / "proof/split.json"
    csv_sha = sha(path)
    split_sha = sha(split_path)
    require(csv_sha == case["dataset_sha256"], "raw CSV changed")
    selected_program = digest(case["selection"]["selected_program"].encode())
    if source is not None:
        proof_path = relative_file(trace_root, source["proof_path"])
        require(sha(proof_path) == source["proof_sha256"], "native GP proof changed")
        proof = read_json(proof_path)
        require(csv_sha == proof["input_csv"]["sha256"] and split_sha == proof["split"]["sha256"]
                and selected_program == source["selected_program_sha256"],
                "GP proof, CSV, split, or selected recipe differs")
    else:
        summary = read_json(split_path.with_name("summary.json"))
        require(summary["input_sha256"] == csv_sha, "raw neural CSV differs from native split proof")
    split = read_json(split_path)
    fit, holdout = split["fit"], split["holdout"]
    require(fit == case["fit_row_ids"] and holdout == case["holdout_row_ids"]
            and len(fit) >= ROWS, "FIT/holdout roster differs")
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        rows = [[float(value) for value in row] for row in reader]
    require(header == case["input_header"] and case["features"] == 4 and len(set(header)) == COLUMNS
            and len(rows) == case["rows"] and sorted(fit + holdout) == list(range(len(rows)))
            and all(len(row) == COLUMNS and all(math.isfinite(v) for v in row) for row in rows),
            "expected complete finite four-feature regression CSV")
    model = Path(model["model_directory"]).resolve(strict=True)
    manifest_path = model / "manifest.json"
    require(model.name == "sha256-" + sha(manifest_path), "original contract model identity differs")
    parts = {part["path"]: part["sha256"] for part in read_json(manifest_path)["files"]}
    contract_path = model / "input.json"
    require(sha(contract_path) == parts["input.json"]
            and (proof is None or parts["input.json"] == proof["model"]["input_sha256"]),
            "original input contract differs")
    contract = read_json(contract_path)
    names = [col["name"] for col in contract["columns"]]
    require(contract["prediction_version"] == 2 and sorted(names) == sorted(header[:-1])
            and contract["target"]["name"] == header[-1], "raw descriptor schema differs")
    medians = [median(rows[i][j] for i in fit) for j in range(COLUMNS)]
    require(all(math.isfinite(value) for value in medians), "FIT median overflow")
    return {
        "rows": rows, "fit": fit, "holdout": holdout, "header": header, "medians": medians,
        "feature_indices": [header.index(name) for name in names],
        "bindings": {
            "csv_path": str(path), "csv_sha256": case["dataset_sha256"], "split_path": str(split_path),
            "split_sha256": split_sha, "input_contract_sha256": parts["input.json"],
            "original_contract_model_id": model.name, "model_id": source["model_id"] if source else None,
            "gp_proof_sha256": source["proof_sha256"] if source else None,
            "selected_program_sha256": selected_program,
        },
    }


def check_baseline(case, resident, encoded, result):
    """Verify exact descriptor and score bits, even for an unfitted neural capture."""
    cid = case["id"]
    fit = resident.fit_inputs[cid]
    for field in ("v256", "col64x17"):
        actual = struct.pack(f"<{len(encoded[field])}f", *encoded[field])
        expected = struct.pack(f"<{len(fit[field])}f", *fit[field])
        require(actual == expected and digest(actual) == encoded[field + "_f32_sha256"],
                f"{cid}: baseline descriptor bits differ for {field}")
    require(result["selected_index"] == case["selection"]["selected_slate_index"]
            and result["selected_dsl"] == case["selection"]["selected_program"], f"{cid}: baseline proposal differs")
    actual = result["final_cost"].tolist()
    expected = case["outputs"]["final_k0_cost"]
    require(struct.pack("<12d", *actual) == struct.pack("<12d", *expected),
            f"{cid}: baseline fused score bits differ")
    reference = resident.reference_critics[cid]["arrays"]
    for name in ("gate_risk", "sp_prior", "h_cost"):
        require(struct.pack("<12d", *result[name]) == struct.pack("<12d", *reference[name]),
                f"{cid}: baseline critic score bits differ for {name}")


def run_case(case, context, resident, native):
    np = resident.np
    cid = case["id"]
    row_ids = context["fit"][:ROWS]
    raw_table = [context["rows"][i] for i in row_ids]
    medians = context["medians"]
    fit = np.asarray([context["rows"][i] for i in context["fit"]], dtype=np.float32)
    indices = context["feature_indices"]
    raw_x = np.ascontiguousarray(fit[:, indices])
    raw_y = np.ascontiguousarray(fit[:, -1])
    encoded = native.encode(raw_x, raw_y)
    baseline = resident.forward(cid, encoded)
    check_baseline(case, resident, encoded, baseline)
    winner = baseline["selected_index"]
    base_costs = baseline["final_cost"].tolist()
    deltas, winner_deltas, winners, evidence = [], [], [], []
    for fit_row in range(ROWS):
        delta_row, winner_delta_row, winner_row, evidence_row = [], [], [], []
        for column in range(COLUMNS):
            x, y = raw_x.copy(), raw_y.copy()
            replacement = np.float32(medians[column])
            if column == COLUMNS - 1:
                y[fit_row] = replacement
            else:
                x[fit_row, indices.index(column)] = replacement
            changed = native.encode(x, y)
            counterfactual = resident.forward(cid, changed)
            costs = counterfactual["final_cost"].tolist()
            delta, winner_delta, new_winner = signed_effects(base_costs, costs, winner)
            require(new_winner == counterfactual["selected_index"], "counterfactual winner differs")
            same_input = (changed["raw_fit_x_f32_sha256"] == encoded["raw_fit_x_f32_sha256"]
                          and changed["raw_fit_y_f32_sha256"] == encoded["raw_fit_y_f32_sha256"])
            require(not same_input or (costs == base_costs and delta == winner_delta == 0.),
                    "no-op intervention changed deterministic output")
            delta_row.append(delta)
            winner_delta_row.append(winner_delta)
            winner_row.append(new_winner)
            evidence_row.append({
                "final_k0_cost": costs,
                "v256_f32_sha256": changed["v256_f32_sha256"],
                "col64x17_f32_sha256": changed["col64x17_f32_sha256"],
                "raw_fit_x_f32_sha256": changed["raw_fit_x_f32_sha256"],
                "raw_fit_y_f32_sha256": changed["raw_fit_y_f32_sha256"],
                "no_op_f32": same_input,
            })
        deltas.append(delta_row)
        winner_deltas.append(winner_delta_row)
        winners.append(winner_row)
        evidence.append(evidence_row)
    maximum, scores = scores_of(deltas)
    return {
        "case_id": cid, "row_ids": row_ids, "raw_table": raw_table, "visible_rows": ROWS,
        "feature_names": context["header"][:-1], "target_name": context["header"][-1],
        "target_column_index": COLUMNS - 1, "fit_medians": medians,
        "baseline_winner_index": winner, "baseline_margin": proposal_margin(base_costs, winner),
        "baseline_final_k0_cost": base_costs, "maximum_absolute_margin_delta": maximum,
        "scores": scores, "signed_margin_deltas": deltas, "signed_winner_cost_deltas": winner_deltas,
        "counterfactual_winners": winners, "counterfactuals": evidence,
        "baseline": {
            **{key: value for key, value in encoded.items() if key.endswith("_sha256")},
            "descriptor_bits_equal": True, "final_cost_max_abs_error": 0.,
            "critic_max_abs_error": 0.,
        },
        "bindings": context["bindings"],
    }


def validate_case(record, case, context):
    require(record["bindings"] == context["bindings"] and record["case_id"] == case["id"],
            "cell case provenance differs")
    row_ids = context["fit"][:ROWS]
    require(record["row_ids"] == row_ids and record["visible_rows"] == ROWS
            and record["raw_table"] == [context["rows"][i] for i in row_ids]
            and record["feature_names"] == context["header"][:-1]
            and record["target_name"] == context["header"][-1] and record["target_column_index"] == COLUMNS - 1
            and record["fit_medians"] == context["medians"], "visible FIT table or reference values differ")
    base = case["outputs"]["final_k0_cost"]
    winner = case["selection"]["selected_slate_index"]
    require(record["baseline_final_k0_cost"] == base and record["baseline_winner_index"] == winner
            and record["baseline_margin"] == proposal_margin(base, winner),
            "baseline final neural ranking differs")
    require(record["baseline"]["descriptor_bits_equal"] is True
            and record["baseline"]["final_cost_max_abs_error"] == 0.
            and record["baseline"]["critic_max_abs_error"] == 0., "baseline parity failed")
    keys = ("signed_margin_deltas", "signed_winner_cost_deltas", "counterfactual_winners",
            "counterfactuals", "scores")
    for key in keys:
        bounded_matrix(record[key], COLUMNS, key)
    rows = [context["rows"][index] for index in context["fit"]]
    indices = context["feature_indices"]
    feature_values = [row[j] for row in rows for j in indices]
    raw_x = struct.pack(f"<{len(feature_values)}f", *feature_values)
    raw_y = struct.pack(f"<{len(rows)}f", *(row[-1] for row in rows))
    require(digest(raw_x) == record["baseline"]["raw_fit_x_f32_sha256"]
            and digest(raw_y) == record["baseline"]["raw_fit_y_f32_sha256"],
            "baseline raw FIT hashes differ")
    for i in range(ROWS):
        for j in range(COLUMNS):
            entry = record["counterfactuals"][i][j]
            x, y = bytearray(raw_x), bytearray(raw_y)
            if j == COLUMNS - 1:
                struct.pack_into("<f", y, 4 * i, context["medians"][j])
            else:
                struct.pack_into("<f", x, 4 * (len(indices) * i + indices.index(j)), context["medians"][j])
            require(digest(x) == entry["raw_fit_x_f32_sha256"]
                    and digest(y) == entry["raw_fit_y_f32_sha256"], "single-cell FIT intervention differs")
            costs = entry["final_k0_cost"]
            delta, winner_delta, new_winner = signed_effects(base, costs, winner)
            require(delta == record["signed_margin_deltas"][i][j]
                    and winner_delta == record["signed_winner_cost_deltas"][i][j]
                    and new_winner == record["counterfactual_winners"][i][j], "counterfactual delta differs")
            for key in ("v256_f32_sha256", "col64x17_f32_sha256", "raw_fit_x_f32_sha256", "raw_fit_y_f32_sha256"):
                require(re.fullmatch("[0-9a-f]{64}", entry[key]) is not None, "invalid counterfactual digest")
            unchanged = ("raw_fit_x_f32_sha256", "raw_fit_y_f32_sha256")
            require(entry["no_op_f32"] == all(entry[key] == record["baseline"][key] for key in unchanged)
                    and (not entry["no_op_f32"] or costs == base), "invalid FP32 no-op evidence")
    maximum, scores = scores_of(record["signed_margin_deltas"])
    require(record["maximum_absolute_margin_delta"] == maximum and record["scores"] == scores,
            "normalized cell scores differ")


def load_influence(path, trace_path, data=None):
    """Validate cells against original CSV, FIT roster, selected output, and score arithmetic."""
    trace_path = Path(trace_path).resolve(strict=True)
    data = read_json(trace_path) if data is None else data
    result = read_json(path)
    require(result["schema"] == SCHEMA and result["status"] == "PASS"
            and result["trace_sha256"] == sha(trace_path)
            and result["weights_before"] == result["weights_after"]
            and result["cpu_only"] is True and result["method"]["target_included"] is True
            and result["method"]["eligible_slate_indices"] == list(range(FIRST_PROPOSAL, SLATE))
            and result["interventions_per_case"] == ROWS * COLUMNS, "invalid attribution envelope")
    cases = case_catalog(data["cases"], "neural attribution trace")
    require(set(cases) == set(result["cases"]) and result["case_count"] == len(cases),
            "attribution case inventory differs")
    original = Path(result["original_run"]).resolve(strict=True)
    models = {model["case_id"]: model for model in read_json(original / "model-paths.json")}
    require(set(cases) == set(models), "original model inventory differs")
    for cid, case in cases.items():
        context = table_context(case, original, models[cid], trace_path.parent)
        validate_case(result["cases"][cid], case, context)
    return result


def capture(args):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    trace_path, original, root, custody = (path.resolve(strict=True) for path in
                                           (args.trace, args.original_run, args.root, args.stage2_custody))
    fit_inputs, features, critics = (path.resolve(strict=True) for path in
                                     (args.fit_inputs, args.native_features, args.critic_inputs))
    output = args.output.resolve()
    require(not output.exists(), "attribution output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    data = read_json(trace_path)
    require(data["schema"] == "jain.h256-stage2-capture/v1", "expected actual saved neural capture")
    cases = case_catalog(data["cases"], "neural attribution trace")
    model_list = read_json(original / "model-paths.json")
    models = {model["case_id"]: model for model in model_list}
    require(set(cases) == set(models) and len(models) == len(model_list), "original model inventory differs")
    native = NativeLandmarks(output.with_suffix(".native"), args.rustc.resolve(strict=True),
                             args.feat_math_rlib.resolve(strict=True), args.landmark_commit)
    resident = None
    try:
        resident = ResidentInference(root, custody, data, features, fit_inputs, critics)
        before = frozen_state(resident)
        records = {}
        for cid, case in cases.items():
            context = table_context(case, original, models[cid], trace_path.parent)
            record = run_case(case, context, resident, native)
            validate_case(record, case, context)
            records[cid] = record
            switches = sum(value != record["baseline_winner_index"]
                           for row in record["counterfactual_winners"] for value in row)
            print(cid, "max |margin delta|", record["maximum_absolute_margin_delta"],
                  "winner switches", switches, flush=True)
        after = frozen_state(resident)
        require(before == after, "frozen inference parameters changed")
        result = {
            "schema": SCHEMA, "status": "PASS", "trace_sha256": sha(trace_path),
            "trace_path": str(trace_path),
            "original_run": str(original), "case_count": len(cases), "interventions_per_case": ROWS * COLUMNS,
            "cpu_only": True, "weights_before": before, "weights_after": after,
            "inputs": {str(path): sha(path) for path in
                       (fit_inputs, features, critics, root / "assets/round-882.pt",
                        custody / "assets/stage2-ensemble.pt", custody / "assets/unified.pt",
                        custody / "assets/critic.pt", Path(__file__).resolve(),
                        Path(__file__).with_name("round_trip_metrics.py").resolve())},
            "landmarks": native.receipt,
            "method": {
                "intervention": "replace one raw visible FIT cell with its column's baseline FIT median",
                "scope": "unchanged FIT roster and candidate bank; fresh native descriptors and frozen fused inference",
                "scalar": "lowest alternative proposal cost minus the original winner's cost",
                "magnitude": "absolute margin change divided by maximum absolute margin change over the 85 visible cells",
                "eligible_slate_indices": list(range(FIRST_PROPOSAL, SLATE)),
                "target_included": True,
            },
            "cases": records,
        }
    finally:
        try:
            if resident is not None:
                resident.close()
        finally:
            native.close()
    create(output, (json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("root", "stage2-custody", "original-run", "trace", "fit-inputs",
                   "native-features", "critic-inputs", "output"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--rustc", type=Path, required=True)
    parser.add_argument("--feat-math-rlib", type=Path, required=True)
    parser.add_argument("--landmark-commit", default=LANDMARK_COMMIT)
    args = parser.parse_args()
    capture(args)


if __name__ == "__main__":
    main()
