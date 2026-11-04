#!/usr/bin/env python3
"""Emit and qualify the scheduled language artifacts for the captured GP selections."""

import argparse
import itertools
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile

from short_recipe_source import (
    CALLS, CASE_IDS, STATELESS, checked_bytes, require, sha, source_spans, write_new,
)

SCHEMA = "jain.multilingual-wave-source/v1"
# Target identities and visible files. The v3 manifest binds the entire build closure.
SCHEDULE = (
    ("cpp17,cpp20", (("model.cpp", "cpp"),)),
    ("python", (("model.py", "python"),)),
    ("rust", (("model.rs", "rust"),)),
    ("typescript", (("model.ts", "typescript"), ("model.kernel.rs", "rust"))),
    ("go", (("model.go", "go"), ("model.go.kernel.rs", "rust"))),
    ("javascript", (("model.js", "javascript"),)),
    ("cuda", (("cuda/model.cu", "cuda"), ("cuda/model_core.rs", "rust"))),
    ("python", (("model.py", "python"),)),
    ("rust", (("model.rs", "rust"),)),
    ("c17", (("c17/model.h", "c"), ("c17/model_core.rs", "rust"))),
)
EMITTER = """use feat_math::{
    artifact_manifest::{ArtifactDomain, Target},
    transpile::{self, EmitOptions},
    GenomeIr,
};
use std::{fs, io::{self, Read, Write}, path::PathBuf};

fn write_new(path: &std::path::Path, bytes: &[u8]) -> io::Result<()> {
    fs::create_dir_all(path.parent().expect("artifact parent"))?;
    fs::OpenOptions::new().write(true).create_new(true).open(path)?.write_all(bytes)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let targets = Target::parse_list(&args[1])?;
    let root = PathBuf::from(&args[2]);
    let mut text = String::new();
    io::stdin().read_to_string(&mut text)?;
    let genome = GenomeIr::parse(&text)?;
    let result = transpile::emit_targets_v3(&genome, &EmitOptions::default(), &targets)?;
    write_new(&root.join("manifest-v3.json"), &result.manifest.envelope_bytes(ArtifactDomain::Native)?)?;
    if !result.manifest.all_supported() {
        return Err(format!("target admission refused: {:?}", result.manifest.targets).into());
    }
    for (name, source) in result.sources {
        write_new(&root.join(name), source.as_bytes())?;
    }
    Ok(())
}
"""
JS_DRIVER = """
const fs = require('node:fs'), vm = require('node:vm');
vm.runInThisContext(fs.readFileSync(process.argv[2], 'utf8'));
const csv = path => fs.readFileSync(path,'utf8').trim().split('\\n').map(row=>row.split(',').map(Number));
const fit = csv(process.argv[3]), probe = csv(process.argv[4]);
const width = fit[0].length - 1;
const labels = Float64Array.from(fit.map(row=>row[width]));
const x = Float64Array.from(fit.flatMap(row=>row.slice(0,width)));
const p = Float64Array.from(probe.flat()), out = new Float64Array(probe.length);
const {status, handle} = modelFitNumeric(x, x.length, fit.length, width, labels, labels.length);
if (status !== 0 || modelPredictNumericBatch(handle,p,p.length,probe.length,width,null,0,out,out.length)!==0)
    throw Error("JavaScript model fit or prediction failed");
modelUnload(handle);
fs.writeFileSync(process.argv[5], Array.from(out).join('\\n')+'\\n', {flag:'wx'});
"""


def execute(argv, stdin=None):
    args = list(map(str, argv))
    result = subprocess.run(args, input=stdin, capture_output=True, timeout=180)
    if result.returncode:
        raise RuntimeError(f"{args[0]} exited {result.returncode}: {result.stderr.decode(errors='replace')}")
    return result.stdout


def audited(argv):
    execute(argv)
    return list(map(str, argv))


def compiled_spans(raw, recipe, language, rust_ops):
    """Match Python/C++ call sites and JavaScript operation specs by ordinal."""
    if language == "rust":
        operations, excerpt = source_spans(raw, recipe)
        return operations, excerpt
    if language not in {"python", "cpp", "javascript"}:
        return [], None
    lines = raw.splitlines(keepends=True)
    offsets = [0, *itertools.accumulate(map(len, lines))]
    names = [op["label"] for op in rust_ops]
    if language == "javascript":
        start = next(i for i, line in enumerate(lines) if line.startswith(b"const MODEL_STAGES="))
        require(lines[start + 1].startswith(b"const MODEL_HEAD="), "JavaScript head declaration absent")
        operations = []
        for i, (line_no, name) in enumerate(zip(
            [start] * (len(names) - 1) + [start + 1], names,
        )):
            specs = list(re.finditer(rb'\{kind:"([^"]+)"[^{}]*\}', lines[line_no]))
            spec = specs[i if line_no == start else 0]
            require(spec[1].decode() == name, "JavaScript operation spec differs")
            operations.append({
                "ordinal": i + 1, "label": name,
                "spans": [[offsets[line_no] + spec.start(), offsets[line_no] + spec.end()]],
            })
        return operations, {
            "span": [offsets[start], offsets[start + 2]], "lines": [start + 1, start + 2],
        }

    fit_begin, predict_begin, end_prefix = {
        "python": (b"    def fit(self, x_rows:", b"    def predict(self, x_rows:", b"    def score("),
        "cpp": (b"Model Model::fit(", b"std::vector<double> Model::predict(", b"}"),
    }[language]
    fit_start = next(i for i, line in enumerate(lines) if line.startswith(fit_begin))
    predict_start = next(i for i in range(fit_start + 1, len(lines)) if lines[i].startswith(predict_begin))
    predict_end = next(i for i in range(predict_start + 1, len(lines)) if lines[i].startswith(end_prefix))
    fit_calls = [i for i in range(fit_start, predict_start) if b"_apply(" in lines[i]]
    predict_calls = [i for i in range(predict_start, predict_end) if b"_apply(" in lines[i]]
    require(len(fit_calls) == len(predict_calls) == len(names) - 1, "direct emitter stage count differs")
    operations = []
    for ordinal, name in enumerate(names, 1):
        family = CALLS[name].encode()
        if ordinal == len(names):
            fit = next(i for i in range(fit_start, predict_start) if family + b"_fit(" in lines[i])
            predict = next(i for i in range(predict_start, predict_end) if family + b"_predict(" in lines[i])
            first = fit
        else:
            fit, predict = fit_calls[ordinal - 1], predict_calls[ordinal - 1]
            require(all(family + b"_apply(" in lines[i] for i in (fit, predict)),
                    f"{language} stage {ordinal} differs")
            first = fit - (name not in STATELESS)
            if first != fit:
                stage = f"stage{ordinal}".encode()
                require(stage in lines[first] and family + b"_fit(" in lines[first]
                        and all(stage in lines[i] for i in (fit, predict)),
                        f"{language} stage {ordinal} state differs")
        operations.append({
            "ordinal": ordinal, "label": name,
            "spans": [[offsets[first], offsets[fit + 1]], [offsets[predict], offsets[predict + 1]]],
        })
    if language == "cpp":
        predict_end += 1
    return operations, {
        "span": [offsets[fit_start], offsets[predict_end]],
        "lines": [fit_start + 1, predict_end],
    }


def c_driver(fit_path, probe_path, *, cpp=False):
    fit = [line.split(",") for line in fit_path.read_text().splitlines()]
    probe = [line.split(",") for line in probe_path.read_text().splitlines()]
    cols = len(probe[0])
    require(all(len(row) == cols + 1 for row in fit) and all(len(row) == cols for row in probe),
            "C17 proof width differs")
    x = ",".join(value for row in fit for value in row[:-1])
    y = ",".join(row[-1] for row in fit)
    p = ",".join(value for row in probe for value in row)
    rows, probe_rows = len(fit), len(probe)
    if cpp:
        include = "model.cpp"
        calls = f"""    model_handle *handle = nullptr;
    if (model_abi_version() != 1) return 1;
    if (model_fit_numeric(x, {rows*cols}, {rows}, {cols}, y, {rows}, &handle)) return 2;
    if (model_predict_numeric_batch(handle, p, {probe_rows*cols}, {probe_rows}, {cols},
                                    nullptr, 0, out, {probe_rows})) return 3;
    model_unload(handle);"""
    else:
        include = "model.h"
        calls = f"""    uint64_t handle = 0;
    if (jope_model_abi_version() != 1) return 1;
    if (jope_model_fit(x, {rows*cols}, y, {rows}, {rows}, {cols}, &handle)) return 2;
    if (jope_model_predict(handle, p, {probe_rows*cols}, {probe_rows}, {cols}, out, {probe_rows})) return 3;
    if (jope_model_free(handle)) return 4;"""
    return f"""#include <stdio.h>
#include "{include}"
int main(void) {{
    const double x[] = {{{x}}};
    const double y[] = {{{y}}};
    const double p[] = {{{p}}};
    double out[{probe_rows}];
{calls}
    for (size_t i = 0; i < {probe_rows}; ++i) printf("%.17g\\n", out[i]);
    return 0;
}}
"""


def check_output(raw, target, reference):
    scale, low = target["scale"], target["min"]
    predicted = [low if scale <= 1e-12 else low + float(value) * scale for value in raw.splitlines()]
    require(len(predicted) == len(reference), "prediction row count differs")
    errors = [abs(actual - expected) for actual, expected in zip(predicted, reference)]
    passed = all(math.isfinite(actual) and error <= 1e-12 + 1e-10 * abs(expected)
                 for actual, expected, error in zip(predicted, reference, errors))
    return {"status": "PASS" if passed else "FAIL", "prediction_rows": len(reference),
            "max_abs_error": max(errors), "normalized_prediction_sha256": sha(raw)}


def qualify(targets, directory, primary, kernel, inputs, rustc, target, reference):
    fit, probe = inputs
    proof_dir = directory / "proof"
    proof_dir.mkdir()
    results = []
    variants = targets.split(",")
    for variant in variants:
        predictions = proof_dir / f"{variant}.predictions.txt"
        binary = proof_dir / variant
        builds = []
        scope = "direct target fit and prediction"
        if variant in {"cpp17", "cpp20"}:
            driver = proof_dir / f"driver-{variant}.cpp"
            write_new(driver, c_driver(fit, probe, cpp=True).encode())
            builds.append(audited([
                shutil.which("clang++") or "clang++", "-O2", f"-std=c++{variant[-2:]}",
                "-ffp-contract=off", "-fno-fast-math", "-I", primary.parent, driver, "-o", binary,
            ]))
            argv = [binary]
        elif variant == "python":
            argv = [sys.executable, "-B", primary, fit, probe, predictions]
        elif variant == "javascript":
            node = shutil.which("node") or "node"
            driver = proof_dir / "driver.cjs"
            write_new(driver, JS_DRIVER.encode())
            builds.append(audited([node, "--check", primary]))
            argv = [node, driver, primary, fit, probe, predictions]
        elif variant == "c17":
            library = proof_dir / "libmodel.a"
            driver = proof_dir / "driver.c"
            write_new(driver, c_driver(fit, probe).encode())
            builds.append(audited([
                rustc, "--edition=2021", "-O", "-C", "panic=unwind", "--crate-type=staticlib",
                primary.with_name("model_ffi.rs"), "-o", library,
            ]))
            builds.append(audited([
                shutil.which("clang") or "clang", "-O2", "-std=c17", "-ffp-contract=off",
                "-I", primary.parent, driver, library, "-o", binary,
            ]))
            argv = [binary]
            scope = "C17 consumer through emitted Rust C ABI"
        else:
            actual_source = primary if variant == "rust" else kernel
            builds.append(audited([
                rustc, "--edition=2021", "-O", "--crate-name=model_kernel", actual_source, "-o", binary,
            ]))
            argv = [binary, fit, probe, predictions]
            if variant != "rust":
                scope = "emitted Rust numerical kernel, compiled and run as CPU host recipe"
            if variant == "cuda":
                library = proof_dir / "libmodel_native.a"
                builds.append(audited([
                    rustc, "--edition=2021", "-O", "--crate-type=staticlib",
                    primary.with_name("model_native.rs"), "-o", library,
                ]))
        stdout = execute(argv)
        if variant in {"c17", "cpp17", "cpp20"}:
            write_new(predictions, stdout)
        result = check_output(predictions.read_bytes(), target, reference)
        results.append({
            "target": variant, "scope": scope, "build_commands": builds,
            "command": list(map(str, argv)), **result,
        })
    return results


def prepare(gp_source, rust_catalog, output, rustc, rlib):
    gp_source, rust_catalog, rustc, rlib = (
        path.resolve(strict=True) for path in (gp_source, rust_catalog, rustc, rlib)
    )
    output = output.resolve()
    base_bytes = rust_catalog.read_bytes()
    base = json.loads(base_bytes)
    require(base["schema"] == "jain.short-recipe-source/v1" and base["status"] == "PASS",
            "passing Rust source catalog required")
    require(tuple(case["case_id"] for case in base["cases"]) == CASE_IDS,
            "ordered ten source cases required")
    gp_controller = json.loads(checked_bytes(gp_source / "controller.receipt.json", base["gp_controller_sha256"]))
    selections = {case["case_id"]: case["selection"] for case in gp_controller["cases"]}
    output.mkdir(parents=True)
    build = output / "build"
    build.mkdir()
    emitter_source, emitter = build / "emit_targets.rs", build / "emit_targets"
    write_new(emitter_source, EMITTER.encode())
    emitter_build = audited([
        rustc, "--edition=2021", "-O", "-C", "lto=fat", "-C", "codegen-units=1",
        "--extern", f"feat_math={rlib}", "-L", f"dependency={rlib.parent}",
        emitter_source, "-o", emitter,
    ])
    cases = []
    for original, (targets, visible) in zip(base["cases"], SCHEDULE):
        case_id = original["case_id"]
        old = gp_source / case_id
        selection = selections[case_id]
        require(all(original[key] == selection[key] for key in ("model_id", "selected_program_sha256"))
                and original["original_fitted_source_sha256"] == selection["rust_sha256"],
                f"{case_id}: recipe/model/source identity differs")
        require(sha(original["selected_program"].encode()) == selection["selected_program_sha256"],
                f"{case_id}: selected recipe bytes differ")
        old_receipt = json.loads(checked_bytes(old / "receipt.json", selection["proof_sha256"]))
        checked_bytes(old / "predictor.rs", selection["rust_sha256"])
        recipe = checked_bytes(old.with_suffix(".selected.dsl"), selection["selected_program_sha256"])
        out = output / case_id
        execute([emitter, targets, out], recipe)
        manifest_path = out / "manifest-v3.json"
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)["payload"]
        require(manifest["genome_sha256"] == selection["selected_program_sha256"]
                and manifest["dsl"].encode() == recipe,
                f"{case_id}: emitted v3 recipe digest differs")
        closure = {}
        for item in manifest["targets"]:
            require(item["state"]["status"] == "supported" and item["state"]["fitted"] is None,
                    f"{case_id}: target refused or has fitted state")
            for part in item["state"]["files"]:
                path = out / part["path"]
                checked_bytes(path, part["sha256"])
                closure[part["path"]] = part["sha256"]
        files = []
        for name, language in visible:
            path = out / name
            raw = checked_bytes(path, closure[name])
            require(not any(marker in raw for marker in
                            (b"fn fitted_model(", b"INPUT_SCALERS", b"TARGET_SCALE", b"SELF_CHECK")),
                    f"{case_id}: fitted-state adapter unexpectedly present")
            operations, excerpt = compiled_spans(raw, recipe, language, original["operations"])
            files.append({
                "source_path": str(path.relative_to(output)), "source_sha256": sha(raw),
                "source_bytes": len(raw), "language": language, "operations": operations,
                "excerpt": excerpt,
            })
        fit = rust_catalog.parent / case_id / "fit.csv"
        probe = rust_catalog.parent / case_id / "probe.csv"
        proof = original["prediction_proof"]
        checked_bytes(fit, proof["encoded_fit_sha256"])
        checked_bytes(probe, proof["encoded_probe_sha256"])
        contract_path = old / "models" / original["model_id"] / "input.json"
        contract = json.loads(checked_bytes(contract_path, proof["input_contract_sha256"]))
        reference = json.loads(checked_bytes(old / "native-predictions.json",
                                             old_receipt["native_predictions_sha256"]))
        primary = out / visible[0][0]
        kernel = out / visible[1][0] if len(visible) == 2 else None
        qualifiers = qualify(targets, out, primary, kernel, (fit, probe), rustc,
                             contract["target"], reference)
        target_execution = {
            "typescript": "source bound; TypeScript/Wasm host execution unverified",
            "go": "source bound; Go/Wasm host execution unverified",
            "cuda": "source bound; Ridge device path emitted; CUDA execution unverified",
        }.get(targets, "local target executed")
        cases.append({
            "case_id": case_id, "model_id": original["model_id"],
            "selected_program_sha256": selection["selected_program_sha256"],
            "original_fitted_source_sha256": selection["rust_sha256"],
            "original_gp_proof_sha256": selection["proof_sha256"],
            "target_execution": target_execution, "files": files,
            "artifact_manifest": str(manifest_path.relative_to(output)),
            "artifact_manifest_sha256": sha(manifest_bytes), "source_closure": closure,
            "parity": qualifiers, "fit_rows": proof["fit_rows"], "holdout_rows": proof["holdout_rows"],
            "input_contract_sha256": proof["input_contract_sha256"],
            "encoded_fit_sha256": proof["encoded_fit_sha256"], "encoded_probe_sha256": proof["encoded_probe_sha256"],
        })
        print(case_id, targets, [(p["status"], p["max_abs_error"]) for p in qualifiers], flush=True)
    result = {
        "schema": SCHEMA, "status": "PASS", "cases": cases, "trace_sha256": base["trace_sha256"],
        "rust_catalog_sha256": sha(base_bytes), "gp_controller_sha256": base["gp_controller_sha256"],
        "scope": "Canonical GP recipe source closure and measured per-case CPU proofs. "
                 "See target_execution and parity records for backend coverage.",
        "span_contract": base["span_contract"],
        "emitter": {
            "entry_point": "feat_math::transpile::emit_targets_v3, default options",
            "helper_source_sha256": sha(EMITTER.encode()), "helper_binary_sha256": sha(emitter.read_bytes()),
            "feat_math_rlib_sha256": sha(rlib.read_bytes()), "build_command": emitter_build,
            "rustc_version": execute([rustc, "-vV"]).decode(),
        },
    }
    catalog = output / "catalog.json"
    write_new(catalog, (json.dumps(result, indent=2) + "\n").encode())
    with ZipFile(output / "JAIN-multilingual-recipe-sources.zip", "x", compression=ZIP_DEFLATED) as archive:
        archive.write(catalog, "catalog.json")
        for case in cases:
            cid = case["case_id"]
            for name in case["source_closure"]:
                relative = f"{cid}/{name}"
                archive.write(output / relative, relative)
            archive.write(output / case["artifact_manifest"], case["artifact_manifest"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("gp-source", "rust-catalog", "output", "rustc", "feat-math-rlib"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.gp_source, args.rust_catalog, args.output, args.rustc, args.feat_math_rlib)


if __name__ == "__main__":
    main()
