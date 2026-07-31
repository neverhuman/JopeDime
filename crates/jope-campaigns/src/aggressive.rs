//! Additive exploratory Golden42 speed/campaign contracts.
//!
//! This module deliberately does not alter the frozen V1/V2 campaign paths.
//! It provides the first executable gate for the new test-adaptive study:
//! cross-language inference timing and an immutable, hash-bound receipt.

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

const NODES: [&str; 3] = ["xbabe1", "xbabe2", "xbabe3"];
const WORKERS_PER_NODE: usize = 80;
const REPETITIONS: usize = 30;
const FITTED_ADMISSION_SCHEMA: &str = "JopeFittedBackendAdmission.v1";

fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn run(program: &str, args: &[&str]) -> Result<Vec<u8>> {
    let output = Command::new(program)
        .args(args)
        .output()
        .with_context(|| format!("launch {program}"))?;
    if !output.status.success() {
        bail!(
            "{program} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(output.stdout)
}

fn tool_exists(program: &str) -> bool {
    Command::new(program).arg("--version").output().is_ok()
}

/// Benchmark the committed four-language portable inference fixtures.
///
/// The fixture is intentionally the existing dependency-free scalar kernel;
/// this gate measures executable language overhead without changing model
/// semantics. Fitted-model emitters remain a subsequent admission gate.
pub(crate) fn bench_languages(output: &Path) -> Result<()> {
    if output.exists() {
        bail!("output path already exists: {}", output.display());
    }
    fs::create_dir_all(output)?;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("vendor/feat-math/tests/portable");
    let work = output.join("build");
    fs::create_dir_all(&work)?;
    let mut backends = Vec::new();

    if tool_exists("rustc") {
        let bin = work.join("portable-rust");
        run(
            "rustc",
            &[
                "-O",
                "-o",
                bin.to_str().unwrap(),
                root.join("full_ref.rs").to_str().unwrap(),
            ],
        )?;
        backends.push(("rust", bin));
    }
    if tool_exists("g++") {
        let bin = work.join("portable-cpp");
        run(
            "g++",
            &[
                "-O2",
                "-ffp-contract=off",
                "-fno-fast-math",
                "-std=c++17",
                "-o",
                bin.to_str().unwrap(),
                root.join("full_ref.cpp").to_str().unwrap(),
            ],
        )?;
        backends.push(("cpp17", bin));
    }
    if tool_exists("python3") {
        backends.push(("python", root.join("full_ref.py")));
    }
    if tool_exists("node") {
        backends.push(("javascript", root.join("full_ref.js")));
    }
    if backends.is_empty() {
        bail!("no language toolchain is available");
    }

    let mut rows = Vec::new();
    let mut reference: Option<Vec<u8>> = None;
    for (language, executable) in backends {
        let (program, args): (&str, Vec<&str>) = match language {
            "python" => ("python3", vec![executable.to_str().unwrap()]),
            "javascript" => ("node", vec![executable.to_str().unwrap()]),
            _ => (executable.to_str().unwrap(), Vec::new()),
        };
        let first = run(program, &args)?;
        if let Some(expected) = &reference {
            if expected != &first {
                bail!("{language} output differs from Rust reference");
            }
        } else {
            reference = Some(first.clone());
        }
        let start = Instant::now();
        let mut digest = String::new();
        for _ in 0..REPETITIONS {
            let bytes = run(program, &args)?;
            digest = sha256(&bytes);
        }
        let elapsed = start.elapsed().as_nanos() as u64;
        rows.push(json!({
            "language": language,
            "repetitions": REPETITIONS,
            "total_ns": elapsed,
            "median_proxy_ns": elapsed / REPETITIONS as u64,
            "output_sha256": digest,
            "parity": true,
        }));
    }
    rows.sort_by(|a, b| {
        a["median_proxy_ns"]
            .as_u64()
            .cmp(&b["median_proxy_ns"].as_u64())
    });
    let value = json!({
        "schema": "JopeGolden42LanguageBenchmark.v1",
        "benchmark_valid": false,
        "purpose": "exploratory inference-language timing; not fitted-model parity admission",
        "nodes": NODES,
        "workers_per_node": WORKERS_PER_NODE,
        "repetitions": REPETITIONS,
        "fixture_root": root,
        "fixture_sha256": sha256(&fs::read(root.join("full_ref.rs"))?),
        "fastest_language": rows.first().and_then(|v| v["language"].as_str()),
        "backends": rows,
    });
    let bytes = serde_json::to_vec_pretty(&value)?;
    fs::write(
        output.join("language-benchmark.v1.json"),
        [bytes.as_slice(), b"\n"].concat(),
    )?;
    Ok(())
}

pub(crate) fn preflight_contract(output: &Path) -> Result<()> {
    if output.exists() {
        bail!("output path already exists: {}", output.display());
    }
    fs::create_dir_all(output)?;
    let value: Value = json!({
        "schema": "JopeGolden42AggressiveContract.v1",
        "benchmark_valid": false,
        "test_adaptive": true,
        "dataset_panel": "/home/ubuntu/golden42/golden42_foundry/data/panel.json",
        "baseline_manifest": "manifests/panel-analysis.v1.json",
        "nodes": NODES,
        "workers_per_node": WORKERS_PER_NODE,
        "population": 240,
        "max_generations": 100,
        "stop_rule": "strict test RMSE below best declared external baseline, confirmed by three replays",
        "backend_admission": "backend-admission.v1.json; full fitted-registry correctness precedes latency and freezes Rust when C++17 is incomplete",
    });
    let bytes = serde_json::to_vec_pretty(&value)?;
    fs::write(
        output.join("run-contract.v1.json"),
        [bytes.as_slice(), b"\n"].concat(),
    )?;
    Ok(())
}

fn median(values: &mut [u128]) -> u128 {
    values.sort_unstable();
    values[values.len() / 2]
}

/// Produce the correctness-first fitted-backend decision used by SCQ80.
///
/// The committed standalone C++ emitter consumes an unfitted `Plan`; it is not
/// a `BoundModel` runtime.  Until the latter covers the complete live registry,
/// the C++ candidate is rejected before the performance threshold is reached.
pub(crate) fn fitted_backend_admission() -> Result<Value> {
    use feat_math::{
        emit_bound, fit_table, ColumnSpec, DataSchema, GenomeIr, Objective, RawTable, RowSemantics,
        TaskSpec,
    };

    let schema = DataSchema {
        columns: vec![
            ColumnSpec::numeric("x0"),
            ColumnSpec::numeric("x1"),
            ColumnSpec::categorical("kind"),
        ],
        row_semantics: RowSemantics::Iid,
    };
    let mut csv = String::from("x0,x1,kind\n");
    let mut target = Vec::new();
    for row in 0..96usize {
        let x0 = row as f64 / 17.0 - 2.0;
        let x1 = ((row * 7) % 19) as f64 / 5.0;
        let kind = if row % 3 == 0 { "a" } else { "b" };
        csv.push_str(&format!("{x0:.17},{x1:.17},{kind}\n"));
        target.push(0.75 * x0 - 0.2 * x1 + f64::from(kind == "a"));
    }
    let table = RawTable::from_csv(&csv)?;
    let genome = GenomeIr::parse(
        "(pipe X (input numeric=median missing=indicator categorical=hybrid) (standardize) (ridge lambda_frac=0.02))",
    )?;
    let task = TaskSpec::regression(Objective::Rmse)?;
    let mut fit_ns = Vec::with_capacity(REPETITIONS);
    let mut predict_ns = Vec::with_capacity(REPETITIONS);
    let mut lower_emit_ns = Vec::with_capacity(REPETITIONS);
    let mut prediction_bits = Vec::new();
    let mut emitted_rust_sha256 = String::new();
    let mut emitted_python_sha256 = String::new();
    for _ in 0..REPETITIONS {
        let started = Instant::now();
        let model = fit_table(&genome, &task, &schema, &table, &target)?;
        fit_ns.push(started.elapsed().as_nanos());
        let started = Instant::now();
        let predictions = model.predict_table(&table)?;
        predict_ns.push(started.elapsed().as_nanos());
        prediction_bits.clear();
        prediction_bits.extend(
            predictions
                .iter()
                .flat_map(|value| value.to_bits().to_le_bytes()),
        );
        let started = Instant::now();
        let emitted = emit_bound(&model, &feat_math::transpile::EmitOptions::default())?;
        lower_emit_ns.push(started.elapsed().as_nanos());
        emitted_rust_sha256 = sha256(emitted.rust.as_bytes());
        emitted_python_sha256 = sha256(emitted.python.as_bytes());
    }
    let source_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("vendor/feat-math/src");
    let registry_files = [
        "ops.rs",
        "heads.rs",
        "model.rs",
        "target_transform.rs",
        "advanced_v3/input.rs",
        "advanced_v3/transforms/mod.rs",
        "advanced_v3/heads/mod.rs",
        "advanced_v3/calibration.rs",
        "advanced_v3/pipeline.rs",
    ];
    let identities = registry_files
        .iter()
        .map(|relative| {
            let bytes = fs::read(source_root.join(relative))?;
            Ok(json!({"file": relative, "sha256": sha256(&bytes)}))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(json!({
        "schema": FITTED_ADMISSION_SCHEMA,
        "benchmark_valid": false,
        "purpose": "correctness-first fitted backend admission for the test-adaptive SCQ80 study",
        "repetitions": REPETITIONS,
        "full_live_registry_required": true,
        "registry_identities": identities,
        "rust_probe": {
            "kind": "fitted schema-bound mixed-input regression pipeline",
            "rows": table.rows.len(),
            "fit_median_ns": median(&mut fit_ns),
            "warm_prediction_median_ns": median(&mut predict_ns),
            "fitted_rust_python_emit_median_ns": median(&mut lower_emit_ns),
            "prediction_bits_sha256": sha256(&prediction_bits),
            "emitted_rust_sha256": emitted_rust_sha256,
            "emitted_python_sha256": emitted_python_sha256,
        },
        "cpp17": {
            "admitted": false,
            "coverage_complete": false,
            "latency_threshold_evaluated": false,
            "reason": "P-1C emits standalone unfitted Plan programs; no once-compiled C++17 runtime lowers both BoundPredictor::Pipeline and Legacy across the complete live registry",
        },
        "selected_backend": "rust",
        "selection_frozen": true,
        "fallback_during_run": false,
    }))
}

pub(crate) fn write_fitted_backend_admission(output: &Path) -> Result<()> {
    if output.exists() {
        bail!("output path already exists: {}", output.display());
    }
    let value = fitted_backend_admission()?;
    let bytes = serde_json::to_vec_pretty(&value)?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(output, [bytes.as_slice(), b"\n"].concat())?;
    Ok(())
}
