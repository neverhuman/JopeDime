#!/usr/bin/env bash
# Local, fail-closed spec 10.5 qualification; no GitHub dispatch or new Cargo workspace.
set -euo pipefail
cd "$(dirname "$0")/../.."
fault=0
if [[ ${1:-} == --fault-emitter ]]; then fault=1; shift; fi
[[ $# -le 1 ]] || { echo 'usage: export-parity.sh [--fault-emitter] [NEW_REPORT_DIR]' >&2; exit 2; }
report=${1:-"$PWD/target/export-parity/$(date -u +%Y%m%dT%H%M%SZ)-$$"}
[[ ! -e "$report" && ! -L "$report" ]] || { echo 'report directory must be new' >&2; exit 2; }
mkdir -p -- "$report"
report=$(realpath -- "$report")
cp -- deploy/ci/export-parity.sh "$report/harness.sh"
export CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only
export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-8} RAYON_NUM_THREADS=1
[[ $CARGO_BUILD_JOBS =~ ^[1-8]$ ]] || { echo 'build jobs must be 1..8' >&2; exit 2; }
export TMPDIR="${CARGO_TARGET_DIR:-$PWD/target}/tmp"
mkdir -p -- "$TMPDIR"
export JOPE_EXPORT_PARITY_REPORT="$report" JOPE_EXPORT_PARITY_FAULT="$fault"
cat > "$report/generate.rs" <<'RUST'
use feat_math::{
    evolve::api::capability::{CapabilityKind, Language, Registry},
    genome::GenomeIr,
    model::{fit, Model},
};
use jope_runtime::{fitted_js, fitted_py, fitted_rs};
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs, path::Path};
const WIDTH: usize = 12;
fn rows(n: usize, mut seed: u64) -> Vec<Vec<f64>> {
    (0..n)
        .map(|_| {
            (0..WIDTH)
                .map(|j| {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    if j == WIDTH - 1 {
                        0.25
                    } else {
                        (seed >> 11) as f64 / (1u64 << 53) as f64
                    }
                })
                .collect()
        })
        .collect()
}
fn probe(kind: CapabilityKind, name: &str) -> String {
    if matches!(
        name,
        "gbm" | "histgbm" | "histgbm2" | "ensemble" | "ensemble_ridge_gbm"
    ) {
        return format!("(pipe X ({name} rounds=4 depth=2 leaves=4 min_leaf=2 bags=1))");
    }
    let op = match name {
        "target" => "target kind=winsor",
        "map" | "derive" => return format!("(pipe X ({name} expr=\"col(0)\") (ridge))"),
        "gpr" => "gpr m=50 steps=1",
        "krr" => "krr m=50",
        "portfolio" => "portfolio preset=core",
        _ => name,
    };
    if matches!(kind, CapabilityKind::Calibration | CapabilityKind::Encoding) {
        return "(pipe X (ridge))".into();
    }
    if kind == CapabilityKind::Head {
        format!("(pipe X ({op}))")
    } else {
        format!("(pipe X ({op}) (ridge))")
    }
}
fn save_case(
    out: &Path,
    cases: &mut Vec<Value>,
    key: String,
    dsl: &str,
    genome: &GenomeIr,
    model: &Model,
    test: &[Vec<f64>],
    languages: &[Language],
    decision: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let index = cases.len();
    let dir = out.join(format!("case-{index}"));
    fs::create_dir(&dir)?;
    let expected = model.predict(&test.to_vec())?;
    assert!(
        expected.len() == test.len() && expected.iter().all(|v| v.is_finite()),
        "non-finite native golden: {key}"
    );
    fs::write(dir.join("model.json"), serde_json::to_vec(model)?)?;
    let mut emissions = Vec::new();
    for language in languages {
        let (name, ext) = match language {
            Language::Rust => ("rust", "rs"),
            Language::Python => ("python", "py"),
            Language::Javascript => ("javascript", "mjs"),
            _ => unreachable!(),
        };
        let emit = || -> Result<String, String> {
            match language {
                Language::Rust => fitted_rs::emit_with_checks(
                    genome,
                    model,
                    WIDTH,
                    &test[..64]
                        .iter()
                        .zip(&expected)
                        .map(|(r, y)| fitted_rs::CheckRow {
                            inputs: r.clone(),
                            expected: *y,
                        })
                        .collect::<Vec<_>>(),
                )
                .map_err(|e| format!("{e:#}")),
                Language::Python => fitted_py::emit_with_checks(
                    genome,
                    model,
                    WIDTH,
                    &test[..64]
                        .iter()
                        .zip(&expected)
                        .map(|(r, y)| fitted_py::CheckRow {
                            inputs: r.clone(),
                            expected: *y,
                        })
                        .collect::<Vec<_>>(),
                )
                .map_err(|e| format!("{e:#}")),
                Language::Javascript => fitted_js::emit_with_checks(
                    genome,
                    model,
                    WIDTH,
                    &test[..64]
                        .iter()
                        .zip(&expected)
                        .map(|(r, y)| fitted_js::CheckRow {
                            inputs: r.clone(),
                            expected: *y,
                        })
                        .collect::<Vec<_>>(),
                )
                .map_err(|e| format!("{e:#}")),
                _ => unreachable!(),
            }
        };
        match emit() {
            Ok(source) => {
                assert_eq!(source, emit()?, "nondeterministic emitter: {key}");
                fs::write(dir.join(format!("predictor.{ext}")), source)?;
                emissions.push(json!({"language":name,"status":"generated","file":format!("case-{index}/predictor.{ext}")}));
            }
            Err(error) => {
                emissions.push(json!({"language":name,"status":"emit_error","error":error}))
            }
        }
    }
    cases.push(json!({"key":key,"dsl":dsl,"directory":format!("case-{index}"),"rows":test,"expected":expected,"decision":decision,"golden_rows":64,"emissions":emissions}));
    Ok(())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let directory = std::env::args().nth(1).expect("report directory");
    let out = Path::new(&directory);
    let registry = Registry::current();
    fs::write(
        out.join("registry.json"),
        serde_json::to_vec_pretty(&registry)?,
    )?;
    let train = rows(160, 7);
    let y: Vec<f64> = train
        .iter()
        .map(|r| 2.0 * r[0] - r[1] * r[2] + (3.0 * r[3]).sin() + 0.5 * r[4] + 2.0)
        .collect();
    let mut golden = rows(57, 99);
    golden.extend([
        vec![0.0; WIDTH],
        vec![-0.0; WIDTH],
        vec![-f64::from_bits(1); WIDTH],
        vec![0.25; WIDTH],
        vec![-0.5; WIDTH],
        vec![1.5; WIDTH],
        vec![1e6; WIDTH],
    ]);
    let mut cases = Vec::new();
    let mut cells = Vec::new();
    for entry in &registry.entries {
        let key = format!("{:?}/{}", entry.kind, entry.name);
        for language in Language::ALL {
            cells.push(json!({"key":key,"language":language,"supported":entry.supported[&language],"status":if entry.supported[&language]{"pending"}else{"unsupported_registry"}}));
        }
        let languages: Vec<_> = [Language::Rust, Language::Python, Language::Javascript]
            .into_iter()
            .filter(|l| entry.supported[l])
            .collect();
        if languages.is_empty() {
            continue;
        }
        assert!(
            matches!(entry.kind, CapabilityKind::Operation | CapabilityKind::Head)
                || (entry.kind == CapabilityKind::Calibration && entry.name == "none")
                || (entry.kind == CapabilityKind::Encoding && entry.name == "dense_numeric"),
            "new supported entry needs a golden: {key}"
        );
        let dsl = probe(entry.kind, entry.name);
        let genome = GenomeIr::parse(&dsl)?;
        let model = fit(&genome, &train, &y)?;
        save_case(
            out, &mut cases, key, &dsl, &genome, &model, &golden, &languages, false,
        )?;
    }
    // Observe real routing: label each actual fitted tree's leaves with node IDs.
    let mut decision_rows = golden.clone();
    decision_rows.extend(
        rows(2000, 2026)
            .into_iter()
            .map(|r| r.into_iter().map(|v| v * 1.5 - 0.25).collect()),
    );
    for dsl in [
        "(pipe X (gbm rounds=4 depth=2 min_leaf=2))",
        "(pipe X (histgbm2 rounds=4 leaves=4 min_leaf=2 bags=1))",
    ] {
        use feat_math::{
            boost2::BoostNode,
            heads::{FittedHead, GbmNode},
        };
        let genome = GenomeIr::parse(dsl)?;
        let model = fit(&genome, &train, &y)?;
        let Model::Pipeline { head, .. } = &model else {
            unreachable!()
        };
        let probes: Vec<_> = match head.as_ref() {
            FittedHead::Gbm(h) => h
                .trees
                .iter()
                .map(|tree| {
                    let mut h = h.clone();
                    let mut tree = tree.clone();
                    for (i, node) in tree.iter_mut().enumerate() {
                        if let GbmNode::Leaf(v) = node {
                            *v = i as f64;
                        }
                    }
                    h.init = 0.0;
                    h.lr = 1.0;
                    h.trees = vec![tree];
                    FittedHead::Gbm(h)
                })
                .collect(),
            FittedHead::HistGbm2(h) => h
                .bags
                .iter()
                .flatten()
                .map(|tree| {
                    let mut h = h.clone();
                    let mut tree = tree.clone();
                    for (i, node) in tree.iter_mut().enumerate() {
                        if let BoostNode::Leaf(v) = node {
                            *v = i as f64;
                        }
                    }
                    h.init = 0.0;
                    h.lr = 1.0;
                    h.bags = vec![vec![tree]];
                    FittedHead::HistGbm2(h)
                })
                .collect(),
            _ => unreachable!(),
        };
        assert!(!probes.is_empty());
        for (i, head) in probes.into_iter().enumerate() {
            let boundaries: Vec<(usize, f64)> = match &head {
                FittedHead::Gbm(h) => h
                    .trees
                    .iter()
                    .flatten()
                    .filter_map(|n| {
                        if let GbmNode::Split {
                            feature, thr_bin, ..
                        } = n
                        {
                            Some((*feature, h.edges[*feature][*thr_bin as usize]))
                        } else {
                            None
                        }
                    })
                    .collect(),
                FittedHead::HistGbm2(h) => h
                    .bags
                    .iter()
                    .flatten()
                    .flatten()
                    .filter_map(|n| {
                        if let BoostNode::Split {
                            feature, threshold, ..
                        } = n
                        {
                            Some((*feature, *threshold))
                        } else {
                            None
                        }
                    })
                    .collect(),
                _ => unreachable!(),
            };
            let margin = boundaries
                .iter()
                .flat_map(|(f, t)| golden.iter().map(move |r| (r[*f] - *t).abs()))
                .fold(f64::INFINITY, f64::min);
            assert!(
                margin.is_finite() && margin > 1e-9,
                "golden lies on a split threshold"
            );
            let model = Model::Pipeline {
                target: None,
                ops: vec![],
                head: Box::new(head),
            };
            save_case(
                out,
                &mut cases,
                format!("routing/{dsl}/{i}"),
                dsl,
                &genome,
                &model,
                &decision_rows,
                &[Language::Rust, Language::Python],
                true,
            )?;
            cases.last_mut().unwrap()["golden_threshold_margin"] = json!(margin);
        }
    }
    // Raw numeric encoding is a distinct contract, not categorical support.
    use jope_runtime::math_pipeline::{
        InputContract, MathPipeline, NumericColumn, PipelineState, TargetScale, INPUT_PART,
        PIPELINE_PART,
    };
    let dsl = "(pipe X (robust_scale append=true) (ridge))";
    let genome = GenomeIr::parse(dsl)?;
    let model = fit(&genome, &train, &y)?;
    let columns = (0..WIDTH)
        .map(|i| NumericColumn {
            name: format!("c{i}"),
            min: 0.0,
            max: 1.0,
            constant: i == WIDTH - 1,
        })
        .collect();
    let input = InputContract::new(
        columns,
        (0..WIDTH).rev().collect(),
        TargetScale {
            name: "target".into(),
            min: -2.0,
            scale: 3.0,
        },
    )?;
    let parts = BTreeMap::from([
        (INPUT_PART.to_owned(), serde_json::to_vec(&input)?),
        (
            PIPELINE_PART.to_owned(),
            serde_json::to_vec(&PipelineState {
                dsl: dsl.into(),
                model,
            })?,
        ),
    ]);
    let pipeline = MathPipeline::decode(&parts, PIPELINE_PART)?;
    let mut raw: Vec<Vec<Option<String>>> = golden
        .iter()
        .map(|r| r.iter().map(|v| Some(v.to_string())).collect())
        .collect();
    for value in [
        None,
        Some(""),
        Some("NaN"),
        Some("inf"),
        Some("-inf"),
        Some("unknown-category"),
        Some("true"),
        Some("false"),
    ] {
        raw.push(vec![value.map(str::to_string); WIDTH]);
    }
    let borrowed: Vec<Vec<Option<&str>>> = raw
        .iter()
        .map(|r| r.iter().map(|v| v.as_deref()).collect())
        .collect();
    let expected = pipeline.predict_cells(&borrowed)?;
    fs::write(out.join("raw.rs"), fitted_rs::emit_pipeline(&pipeline)?)?;
    fs::write(
        out.join("raw.py"),
        fitted_py::emit_math_pipeline(&pipeline)?,
    )?;
    fs::write(out.join("raw.mjs"), pipeline.export_javascript()?)?;
    fs::write(
        out.join("fixtures.json"),
        serde_json::to_vec(
            &json!({"cells":cells,"cases":cases,"raw":{"rows":raw,"expected":expected,"columns":pipeline.input.columns.iter().map(|c|&c.name).collect::<Vec<_>>()}}),
        )?,
    )?;
    Ok(())
}
RUST
nice -n 10 python3 -I -B - "$report" <<'PY'
import datetime, hashlib, json, math, os, pathlib, platform, subprocess, sys, time, traceback

root = pathlib.Path.cwd()
out = pathlib.Path(sys.argv[1])
started = time.monotonic()
report = {
    "schema": "jope.export-parity/v1",
    "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "status": "running",
    "passed": False,
    "cases": [],
    "cells": [],
    "toolchains": {},
    "fixture_kind": "deterministic numerical contract fixtures, not quality/training qualification",
    "host": platform.platform(),
    "environment": {
        "CARGO_BUILD_JOBS": os.environ["CARGO_BUILD_JOBS"],
        "CARGO_INCREMENTAL": os.environ["CARGO_INCREMENTAL"],
        "CARGO_PROFILE_DEV_DEBUG": os.environ["CARGO_PROFILE_DEV_DEBUG"],
        "RAYON_NUM_THREADS": os.environ["RAYON_NUM_THREADS"],
        "niceness": os.getpriority(os.PRIO_PROCESS, 0),
    },
    "ci_executed": os.environ.get("GITHUB_ACTIONS") == "true",
    "ci_context": {
        key: os.environ.get(key)
        for key in [
            "GITHUB_RUN_ID",
            "GITHUB_RUN_ATTEMPT",
            "GITHUB_WORKFLOW_REF",
            "GITHUB_REPOSITORY",
            "GITHUB_EVENT_NAME",
        ]
    },
    "scope": "Advertised primary-language registry cells. Unsupported calibration, categorical and class-label contracts remain unqualified.",
}


def digest(p):
    with pathlib.Path(p).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def command(args, name, timeout=900):
    begin = time.monotonic()
    with (
        (out / (name + ".stdout")).open("w") as stdout,
        (out / (name + ".stderr")).open("w") as stderr,
    ):
        result = subprocess.run(
            args, stdout=stdout, stderr=stderr, text=True, timeout=timeout
        )
    report.setdefault("commands", []).append(
        {
            "argv": [str(x) for x in args],
            "exit": result.returncode,
            "seconds": time.monotonic() - begin,
            "stdout": name + ".stdout",
            "stderr": name + ".stderr",
        }
    )
    if result.returncode:
        raise RuntimeError(f"{name} exited {result.returncode}; see {name}.stderr")
    return (out / (name + ".stdout")).read_text()


def compare(case, language, actual):
    expected = case["expected"]
    assert len(actual) == len(expected), f"{case['key']}: shape mismatch"
    absolute, relative = (1e-9, 1e-7) if language == "javascript" else (1e-12, 1e-10)
    differences = []
    for i, (a, b) in enumerate(zip(actual, expected)):
        if (
            isinstance(a, bool)
            or not isinstance(a, (int, float))
            or not math.isfinite(a)
            or not math.isfinite(b)
            or abs(a - b) > absolute + relative * abs(b)
        ):
            differences.append(i)
    if case.get("decision"):
        flips = [i for i, (a, b) in enumerate(zip(actual, expected)) if a != b]
        golden = sum(i < 64 for i in flips)
        random = len(flips) - golden
        assert not golden and random * 1000 <= len(expected) - 64, (
            f"{case['key']}: decision flips golden={golden} randomized={random}"
        )
    else:
        flips = []
        assert not differences, (
            f"{case['key']}/{language}: {len(differences)} numeric mismatches, first row {differences[:1]}"
        )
    return {
        "status": "pass",
        "language": language,
        "rows": len(expected),
        "atol": absolute,
        "rtol": relative,
        "max_abs_error": max(abs(a - b) for a, b in zip(actual, expected)),
        "decision_flip_rows": flips,
        "golden_flip_rate": sum(i < 64 for i in flips) / 64
        if case.get("decision")
        else None,
        "randomized_flip_rate": sum(i >= 64 for i in flips) / (len(expected) - 64)
        if case.get("decision")
        else None,
        "decision_measurement": "actual fitted leaf node ID"
        if case.get("decision")
        else None,
        "golden_threshold_margin": case.get("golden_threshold_margin"),
        "numeric_mismatch_rows": differences,
        "self_check_rows": 64,
    }


try:
    report["source_sha"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    report["source_dirty"] = subprocess.check_output(
        ["git", "status", "--porcelain"], text=True
    )
    report["harness_sha256"] = digest(out / "harness.sh")
    report["cargo_lock_sha256"] = digest(root / "Cargo.lock")
    for name, args, pin in [
        ("rustc", ["rustc", "--version"], "rustc 1.97.1"),
        ("python", ["python3", "--version"], "Python 3.12.3"),
        ("node", ["node", "--version"], "v26.1.0"),
    ]:
        version = command(args, "version-" + name).strip()
        assert version == pin or (name == "rustc" and version.startswith(pin + " ")), (
            f"{name}: expected {pin}, got {version}"
        )
        report["toolchains"][name] = version
    assert sys.version_info[:3] == (3, 12, 3), (
        "driver interpreter differs from pinned Python"
    )
    report["load_ceiling"] = min(85, 0.85 * len(os.sched_getaffinity(0)))
    while os.getloadavg()[0] > report["load_ceiling"]:
        time.sleep(10)
    build = command(
        [
            "cargo",
            "build",
            "--locked",
            "-p",
            "jope-runtime",
            "--lib",
            "--message-format=json",
        ],
        "cargo-build",
    )
    libraries = {}
    for line in build.splitlines():
        message = json.loads(line)
        if message.get("reason") == "compiler-artifact":
            for filename in message["filenames"]:
                if filename.endswith(".rlib"):
                    libraries[message["target"]["name"]] = filename
    library_dir = pathlib.Path(libraries["jope_runtime"]).parent
    dependency_dir = library_dir if library_dir.name == "deps" else library_dir / "deps"
    args = [
        "rustc",
        "--edition=2021",
        "-C",
        "debuginfo=line-tables-only",
        "-L",
        "dependency=" + str(dependency_dir),
    ]
    for name in ["jope_runtime", "feat_math", "serde_json"]:
        args += ["--extern", name + "=" + libraries[name]]
    command(
        args + [str(out / "generate.rs"), "-o", str(out / "generate")],
        "compile-generator",
    )
    command([str(out / "generate"), str(out)], "generate-fixtures")
    fixtures = json.loads((out / "fixtures.json").read_text())
    report["cells"] = fixtures["cells"]
    report["registry_sha256"] = digest(out / "registry.json")
    report["fixture_sha256"] = digest(out / "fixtures.json")
    cases = fixtures["cases"]
    assert cases, "zero collected golden models"
    expected_cells = sum(c["supported"] for c in report["cells"])
    report["expected_supported_cells"] = expected_cells
    errors = [
        (c["key"], emission)
        for c in cases
        for emission in c["emissions"]
        if emission["status"] != "generated"
    ]
    assert not errors, f"advertised support lacks fitted export: {errors}"
    fault = os.environ["JOPE_EXPORT_PARITY_FAULT"] == "1"
    report["fault_injected"] = fault
    if fault:
        victim = next(
            e for c in cases for e in c["emissions"] if e["language"] == "python"
        )
        p = out / victim["file"]
        text = p.read_text()
        old = "return fitted_model().predict(rows)"
        assert text.count(old) == 1
        report["fault_original_sha256"] = digest(p)
        p.write_text(
            text.replace(
                old, "return [value + 1.0 for value in fitted_model().predict(rows)]"
            )
        )
        report["fault_file"] = victim["file"]
        report["fault_mutated_sha256"] = digest(p)
    # Execute Python and JavaScript separately from the native reference generator.
    for language in ["python", "javascript"]:
        for i, case in enumerate(cases):
            emission = next(
                (e for e in case["emissions"] if e["language"] == language), None
            )
            if not emission:
                continue
            file = out / emission["file"]
            stem = f"{language}-{i}"
            try:
                command(
                    (["python3", "-I", "-B"] if language == "python" else ["node"])
                    + [str(file), "--self-check"],
                    stem + "-self-check",
                )
                if language == "python":
                    driver = file.parent / "run.py"
                    driver.write_text(
                        'import runpy,json,sys\nm=runpy.run_path(sys.argv[1]); rows=json.load(open(sys.argv[2]))["cases"][int(sys.argv[3])]["rows"];print(json.dumps(m["fitted_predict"](rows),allow_nan=False))\n'
                    )
                    values = command(
                        [
                            "python3",
                            "-I",
                            "-B",
                            str(driver),
                            str(file),
                            str(out / "fixtures.json"),
                            str(i),
                        ],
                        stem,
                    )
                else:
                    driver = file.parent / "run.mjs"
                    driver.write_text(
                        file.read_text()
                        + "\nconst rows="
                        + json.dumps(case["rows"])
                        + ';const h=modelFittedHandle(),out=new Float64Array(rows.length),flat=Float64Array.from(rows.flat());if(modelPredictNumericBatch(h,flat,flat.length,rows.length,12,null,0,out,out.length)!==0)throw Error("status");console.log(JSON.stringify(Array.from(out)));\n'
                    )
                    values = command(["node", str(driver)], stem)
                actual = json.loads(values)
                record = compare(case, language, actual)
                record.update(key=case["key"], emitted_sha256=digest(file))
                report["cases"].append(record)
                (file.parent / (language + "-rows.json")).write_text(
                    json.dumps(
                        {
                            "expected": case["expected"],
                            "actual": actual,
                            "decision": case["decision"],
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
            except Exception as error:
                report["cases"].append(
                    {
                        "key": case["key"],
                        "language": language,
                        "status": "fail",
                        "error": str(error),
                        "emitted_sha256": digest(file),
                    }
                )
    # Batch real std-only Rust modules; every module includes a populated self_check test.
    rust_cases = [
        (i, c)
        for i, c in enumerate(cases)
        if any(e["language"] == "rust" for e in c["emissions"])
    ]
    program = out / "rust-predictors.rs"
    chunks = []
    calls = []
    for i, c in rust_cases:
        source = (out / c["directory"] / "predictor.rs").read_text()
        (out / c["directory"] / "rows.tsv").write_text(
            "".join(" ".join(repr(v) for v in row) + "\n" for row in c["rows"])
        )
        chunks.append(f"mod c{i} {{\n{source}\n}}\n")
        calls.append(
            f'let rows=read_rows(&format!("{{root}}/{c["directory"]}/rows.tsv"));for value in c{i}::predict_numeric(&rows).unwrap() {{println!("{{:.17e}}",value);}}'
        )
    loader = "fn read_rows(path:&str)->Vec<Vec<f64>>{std::fs::read_to_string(path).unwrap().lines().map(|line|line.split_whitespace().map(|v|v.parse::<f64>().unwrap()).collect()).collect()}\n"
    program.write_text(
        "".join(chunks)
        + "\n"
        + loader
        + 'fn main(){let root=std::env::args().nth(1).expect("fixture directory");'
        + "".join(calls)
        + "}\n"
    )
    command(
        [
            "rustc",
            "--edition=2021",
            "-C",
            "codegen-units=8",
            str(program),
            "-o",
            str(out / "rust-predictors"),
        ],
        "compile-predictors",
    )
    actual = [
        float(v)
        for v in command(
            [str(out / "rust-predictors"), str(out)], "rust-predictors"
        ).splitlines()
    ]
    offset = 0
    for i, c in rust_cases:
        n = len(c["expected"])
        values = actual[offset : offset + n]
        offset += n
        record = compare(c, "rust", values)
        record.update(
            key=c["key"], emitted_sha256=digest(out / c["directory"] / "predictor.rs")
        )
        report["cases"].append(record)
        (out / c["directory"] / "rust-rows.json").write_text(
            json.dumps(
                {
                    "expected": c["expected"],
                    "actual": values,
                    "decision": c["decision"],
                },
                allow_nan=False,
            )
            + "\n"
        )
    assert offset == len(actual), "unexpected Rust output rows"
    command(
        [
            "rustc",
            "--edition=2021",
            "--test",
            "-C",
            "codegen-units=8",
            str(program),
            "-o",
            str(out / "rust-self-checks"),
        ],
        "compile-self-checks",
    )
    self_check = command(
        [str(out / "rust-self-checks"), "--test-threads=1"], "rust-self-checks"
    )
    assert f"{len(rust_cases)} passed; 0 failed" in self_check, (
        "zero/missing Rust self-checks"
    )
    # Raw-cell encoding, non-finite/missing tokens, constant columns and explicit column order.
    raw = fixtures["raw"]
    raw_case = {"key": "raw_numeric_contract", "expected": raw["expected"]}
    raw_results = []
    for language in ["python", "javascript", "rust"]:
        if language == "python":
            p = out / "raw-run.py"
            p.write_text(
                'import runpy,json,sys\nm=runpy.run_path(sys.argv[1]);f=json.load(open(sys.argv[2]))["raw"];assert list(m["MODEL_COLUMNS"])==f["columns"];print(json.dumps(m["predict_rows"](f["rows"]),allow_nan=False))\n'
            )
            text = command(
                [
                    "python3",
                    "-I",
                    "-B",
                    str(p),
                    str(out / "raw.py"),
                    str(out / "fixtures.json"),
                ],
                "raw-python",
            )
        elif language == "javascript":
            p = out / "raw-run.mjs"
            p.write_text(
                (out / "raw.mjs").read_text()
                + "\nif(JSON.stringify(MODEL_COLUMNS)!=="
                + json.dumps(json.dumps(raw["columns"], separators=(",", ":")))
                + ')throw Error("column order");console.log(JSON.stringify(predictRows('
                + json.dumps(raw["rows"])
                + ")));\n"
            )
            text = command(["node", str(p)], "raw-javascript")
        else:
            rows = (
                "vec!["
                + ",".join(
                    "vec!["
                    + ",".join(
                        "None" if v is None else "Some(" + json.dumps(v) + ")"
                        for v in row
                    )
                    + "]"
                    for row in raw["rows"]
                )
                + "]"
            )
            columns = "&[" + ",".join(json.dumps(v) for v in raw["columns"]) + "]"
            p = out / "raw-run.rs"
            p.write_text(
                (out / "raw.rs").read_text()
                + f'\nfn main(){{let rows={rows};let headers={columns};let actual=predict_cells(&rows).unwrap();assert_eq!(actual,predict_table(headers,&rows).unwrap());let reversed_headers:Vec<_>=headers.iter().rev().copied().collect();let reversed_rows:Vec<Vec<_>>=rows.iter().map(|r|r.iter().rev().copied().collect()).collect();assert_eq!(actual,predict_table(&reversed_headers,&reversed_rows).unwrap());for value in actual{{println!("{{:.17e}}",value);}}}}\n'
            )
            command(
                ["rustc", "--edition=2021", str(p), "-o", str(out / "raw-run")],
                "compile-raw",
            )
            text = command([str(out / "raw-run")], "raw-rust")
            text = json.dumps([float(v) for v in text.splitlines()])
        record = compare(raw_case, language, json.loads(text))
        record["self_check_rows"] = 0
        raw_results.append(record)
    report["raw_numeric_contract"] = raw_results
    for cell in report["cells"]:
        if cell["supported"]:
            matches = [
                c
                for c in report["cases"]
                if c["key"] == cell["key"] and c["language"] == cell["language"]
            ]
            assert len(matches) == 1, cell
            cell["status"] = matches[0]["status"]
    executed = sum(c["status"] == "pass" for c in report["cells"])
    report["executed_supported_cells"] = executed
    assert expected_cells > 0 and executed == expected_cells, (
        f"{executed}/{expected_cells} supported cells passed; see failed case records"
    )
    report.update(
        status="pass",
        passed=True,
        executed_supported_cells=executed,
        unsupported_cells=sum(not c["supported"] for c in report["cells"]),
    )
    identity = {
        key: report[key]
        for key in [
            "source_sha",
            "harness_sha256",
            "registry_sha256",
            "fixture_sha256",
            "toolchains",
        ]
    }
    report["qualification_id"] = (
        "sha256:"
        + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    )
except Exception as error:
    report.update(status="fail", passed=False, error=str(error))
    (out / "failure.txt").write_text(traceback.format_exc())
finally:
    report["seconds"] = time.monotonic() - started
    report["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report["source_end"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    report["source_dirty_end"] = subprocess.check_output(
        ["git", "status", "--porcelain"], text=True
    )
    report["source_stable"] = (
        report["source_end"] == report.get("source_sha")
        and report["source_dirty_end"] == report.get("source_dirty")
        and digest(root / "deploy/ci/export-parity.sh") == report.get("harness_sha256")
    )
    if not report["source_stable"]:
        report.update(
            status="source_changed",
            passed=False,
            error="source changed during qualification",
        )
    (out / "summary.md").write_text(
        f"Export parity: **{report['status']}**\n\nSource: {report.get('source_sha', 'unavailable')}\n\nSupported cells: {report.get('executed_supported_cells', 0)}/{report.get('expected_supported_cells', 'unavailable')}\n\n{report.get('error', 'All advertised supported cells executed with real pinned toolchains. Unsupported registry cells remain unqualified; hosted E2 is not claimed.')}\n"
    )
    report["files"] = {
        str(p.relative_to(out)): {"bytes": p.stat().st_size, "sha256": digest(p)}
        for p in sorted(out.rglob("*"))
        if p.is_file() and p.name != "report.json"
    }
    (out / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps({k: report[k] for k in ["status", "passed", "seconds"]})
        + " report="
        + str(out / "report.json")
    )
sys.exit(0 if report["passed"] else 1)
PY
