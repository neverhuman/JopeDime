//! Registry-derived conversion witnesses. The historical corpus stays read-only.

use feat_math::kernel::capabilities::Manifest;
use feat_math::kernel::expression_recipe::{emit_cpp17 as emit_expression_cpp17, PROFILE};
use feat_math::kernel::Program;
use feat_math::registry::{generatable_operators, operator_registry};
use feat_math::transpile::{emit, emit_cpp17, supports, supports_cpp17, EmitOptions};
use feat_math::{fit_predict, GenomeIr, Pcg32};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use std::fs;
use std::path::Path;
use std::process::Command;

const TRAINING_ROWS: usize = 64;

const HISTORICAL: &str = include_str!(
    "../fixtures/compat_freeze/historical_v1/objects/sha256/b5e66d14bdde73e4ff7d69ad65d6e7a8b0772ebc7c532f6a60bb775f4c2ef5c3.json"
);
const EMITTED: &str = include_str!("fixtures/compat_freeze/emitted_bytes.v1.json");
const MODELS: &str = include_str!("fixtures/compat_freeze/model_serde.v1.json");

fn rows_by_name(rows: &[Value]) -> BTreeMap<&str, &Value> {
    rows.iter()
        .map(|row| (row["name"].as_str().unwrap(), row))
        .collect()
}

fn witness_sources() -> BTreeMap<String, String> {
    let historical: Value = serde_json::from_str(HISTORICAL).unwrap();
    let emitted: Value = serde_json::from_str(EMITTED).unwrap();
    let programs = rows_by_name(emitted["programs"].as_array().unwrap());
    historical["registry"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|row| row["evidence"]["kind"] == "fitted_existing")
        .map(|row| {
            let name = row["evidence"]["name"].as_str().unwrap();
            (
                name.to_owned(),
                programs[name]["canonical"].as_str().unwrap().to_owned(),
            )
        })
        .collect()
}

#[test]
fn all_frozen_wire_ids_have_current_conversion_witnesses_or_explicit_refusals() {
    let historical: Value = serde_json::from_str(HISTORICAL).unwrap();
    let emitted: Value = serde_json::from_str(EMITTED).unwrap();
    let models: Value = serde_json::from_str(MODELS).unwrap();
    let frozen = historical["registry"].as_array().unwrap();
    let programs = rows_by_name(emitted["programs"].as_array().unwrap());
    let model_rows = rows_by_name(models["models"].as_array().unwrap());
    let manifest = Manifest::current();
    let registry = operator_registry();
    let generatable: BTreeSet<_> = generatable_operators(false).map(|op| op.name).collect();
    let mut witness_names = BTreeSet::new();
    let mut kinds = BTreeMap::<&str, usize>::new();

    assert_eq!(registry.len(), 53, "frozen wire IDs 0–52");
    assert_eq!(frozen.len(), registry.len());
    for (id, (op, old)) in registry.iter().zip(frozen).enumerate() {
        assert_eq!(usize::from(op.id.0), id, "wire order");
        assert_eq!(old["id"], id, "wire {id} frozen ID");
        assert_eq!(old["name"], op.name, "wire {id} spelling");
        assert_eq!(old["semantic_key"], op.semantic_key, "wire {id} semantics");
        for (field, actual) in [
            ("parseable", op.lifecycle.parseable),
            ("executable", op.lifecycle.executable),
            ("generatable", op.lifecycle.generatable),
            ("quarantined", op.lifecycle.quarantined),
        ] {
            assert_eq!(old[field], actual, "wire {id} lifecycle {field}");
        }
        assert_eq!(generatable.contains(op.name), op.lifecycle.generatable);
        let cap = manifest
            .op(op.name)
            .unwrap_or_else(|| panic!("wire {id} missing manifest"));
        assert_eq!(cap.wire_id, Some(op.id.0));
        let kind = old["evidence"]["kind"].as_str().unwrap();
        if op.lifecycle.generatable {
            assert_eq!(kind, "fitted_existing", "wire {id}: search-visible witness");
        }
        *kinds.entry(kind).or_default() += 1;
        match kind {
            "fitted_existing" => {
                let name = old["evidence"]["name"].as_str().unwrap();
                let source = programs[name]["canonical"].as_str().unwrap();
                assert!(
                    source.contains(&format!("({} ", op.name))
                        || source.contains(&format!("({})", op.name)),
                    "wire {id}: witness must actually contain {}",
                    op.name
                );
                assert!(
                    model_rows.contains_key(name),
                    "wire {id}: missing fitted witness"
                );
                let parsed = GenomeIr::parse(source).unwrap();
                let canonical = parsed.canonical();
                let reparsed = GenomeIr::parse(&canonical).unwrap();
                assert_eq!(
                    reparsed.canonical(),
                    canonical,
                    "wire {id}: canonical fixed point"
                );
                assert_eq!(
                    Program::import_legacy(&parsed).to_bytes().unwrap(),
                    Program::import_legacy(&reparsed).to_bytes().unwrap(),
                    "wire {id}: executable codec round trip"
                );
                assert!(
                    supports(&reparsed).is_ok(),
                    "wire {id}: paired recipe refused"
                );
                let artifact = emit(&reparsed, &EmitOptions::default()).unwrap();
                let sidecar: Value = serde_json::from_str(&artifact.manifest).unwrap();
                assert_eq!(sidecar["program"], canonical);
                witness_names.insert(name);
            }
            "alias_case" => {
                assert!(op.lifecycle.executable && !op.lifecycle.generatable);
                assert_ne!(op.name, op.semantic_key, "wire {id}: alias key");
            }
            "quarantined_refusal" => {
                assert!(op.lifecycle.quarantined && !op.lifecycle.executable);
            }
            other => panic!("wire {id}: unclassified witness {other}"),
        }
    }
    assert_eq!(kinds["fitted_existing"], 42);
    assert_eq!(kinds["alias_case"], 4);
    assert_eq!(kinds["quarantined_refusal"], 7);
    assert_eq!(witness_names.len(), 19, "deduplicated executable programs");
    assert_eq!(generatable.len(), 35, "search-visible frozen wire subset");
}

fn toy_data() -> (Vec<Vec<f64>>, Vec<f64>) {
    toy_data_seed(42)
}

fn toy_data_seed(seed: u64) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = Pcg32::new(seed, 17);
    let mut x = Vec::new();
    let mut y = Vec::new();
    for _ in 0..80 {
        let row: Vec<f64> = (0..5).map(|_| rng.next_f64() * 4.0 - 2.0).collect();
        let target = 1.7 * row[0] - 0.9 * row[1]
            + 0.5 * (3.0 * row[2]).sin()
            + 0.3 * row[3] * row[4]
            + 0.1 * rng.gaussian();
        x.push(row);
        y.push(target);
    }
    (x, y)
}

fn write_recipe_csvs(dir: &Path, x: &[Vec<f64>], y: &[f64]) {
    let mut train = String::new();
    for (row, target) in x[..TRAINING_ROWS].iter().zip(&y[..TRAINING_ROWS]) {
        for value in row {
            train.push_str(&format!("{value:?},"));
        }
        train.push_str(&format!("{target:?}\n"));
    }
    fs::write(dir.join("train.csv"), train).unwrap();
    let mut test = String::new();
    for row in &x[TRAINING_ROWS..] {
        test.push_str(
            &row.iter()
                .map(|value| format!("{value:?}"))
                .collect::<Vec<_>>()
                .join(","),
        );
        test.push('\n');
    }
    fs::write(dir.join("test.csv"), test).unwrap();
    assert_eq!(
        fs::read_to_string(dir.join("train.csv"))
            .unwrap()
            .lines()
            .count(),
        TRAINING_ROWS
    );
    assert_eq!(
        fs::read_to_string(dir.join("test.csv"))
            .unwrap()
            .lines()
            .count(),
        x.len() - TRAINING_ROWS
    );
}

fn require_success(what: &str, command: &mut Command) {
    let output = command
        .output()
        .unwrap_or_else(|error| panic!("{what}: {error}"));
    assert!(
        output.status.success(),
        "{what}: exit {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn recipe_bits(path: &Path) -> Vec<u64> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| line.trim().parse::<f64>().unwrap().to_bits())
        .collect()
}

fn run_emitted_pair(
    name: &str,
    genome: &GenomeIr,
    dir: &Path,
    x: &[Vec<f64>],
    y: &[f64],
) -> (Vec<u64>, Vec<u64>) {
    fs::create_dir_all(dir).unwrap();
    write_recipe_csvs(dir, x, y);
    let artifact = emit(genome, &EmitOptions::default()).unwrap();
    let python = dir.join("model.py");
    let python_out = dir.join("python.txt");
    fs::write(&python, artifact.python).unwrap();
    require_success(
        &format!("{name}: emitted Python"),
        Command::new("python3")
            .arg(&python)
            .arg(dir.join("train.csv"))
            .arg(dir.join("test.csv"))
            .arg(&python_out)
            .env("LC_ALL", "C")
            .env("PYTHONDONTWRITEBYTECODE", "1"),
    );
    let rust = dir.join("model.rs");
    let binary = dir.join("model_bin");
    let rust_out = dir.join("rust.txt");
    fs::write(&rust, artifact.rust).unwrap();
    require_success(
        &format!("{name}: rustc -O"),
        Command::new("rustc")
            .arg("-O")
            .arg("-o")
            .arg(&binary)
            .arg(&rust),
    );
    require_success(
        &format!("{name}: emitted Rust"),
        Command::new(&binary)
            .arg(dir.join("train.csv"))
            .arg(dir.join("test.csv"))
            .arg(&rust_out),
    );
    (recipe_bits(&python_out), recipe_bits(&rust_out))
}

fn run_cpp_recipe(
    name: &str,
    mut source: String,
    dir: &Path,
    x: &[Vec<f64>],
    y: &[f64],
) -> Vec<u64> {
    assert_eq!(x.len(), 80);
    assert_eq!(y.len(), 80);
    assert_eq!(TRAINING_ROWS, 64, "C++ numeric ABI training rows");
    assert!(x.iter().all(|row| row.len() == 5));
    fs::create_dir_all(dir).unwrap();
    source.push_str("\n#include <cstdio>\nint main() {\n");
    // Decode exact input bits, including signed zero and nonfinite feature payloads.
    // The input reaches the model's numeric ABI rather than the finite-only CSV CLI.
    for (label, values) in [
        (
            "train",
            x[..TRAINING_ROWS]
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>(),
        ),
        ("targets", y[..TRAINING_ROWS].to_vec()),
        (
            "test",
            x[TRAINING_ROWS..]
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>(),
        ),
    ] {
        write!(source, "const std::uint64_t {label}_bits[] = {{").unwrap();
        for value in &values {
            write!(source, "0x{:016x}ULL,", value.to_bits()).unwrap();
        }
        writeln!(
            source,
            "}}; double {label}[{}]; for (std::size_t i=0; i<{}; ++i) std::memcpy(&{label}[i], &{label}_bits[i], 8);",
            values.len(), values.len()
        )
        .unwrap();
    }
    source.push_str(
        "double out[16] = {}; model_handle* model = nullptr;\n\
         if (model_fit_numeric(train, 320, 64, 5, targets, 64, &model) != 0) return 10;\n\
         if (model_predict_numeric_batch(model, test, 80, 16, 5, nullptr, 0, out, 16) != 0) return 11;\n\
         for (double value : out) { std::uint64_t bits; std::memcpy(&bits, &value, 8); std::printf(\"%016llx\\n\", static_cast<unsigned long long>(bits)); }\n\
         model_unload(model); return 0; }\n",
    );
    let path = dir.join("model.cpp");
    let binary = dir.join("model.bin");
    fs::write(&path, source).unwrap();
    require_success(
        &format!("{name}: g++17 O2"),
        Command::new("g++")
            .args(["-std=c++17", "-O2", "-ffp-contract=off", "-fno-fast-math"])
            .arg(&path)
            .arg("-o")
            .arg(&binary),
    );
    let output = Command::new(&binary).output().unwrap();
    assert!(
        output.status.success(),
        "{name}: C++17 recipe exit {}: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let bits: Vec<u64> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|word| u64::from_str_radix(word, 16).unwrap())
        .collect();
    assert_eq!(bits.len(), 16, "{name}: C++17 output row count");
    bits
}

fn round_trip(source: &str) -> GenomeIr {
    let original = GenomeIr::parse(source).unwrap();
    let canonical = original.canonical();
    let genome = GenomeIr::parse(&canonical).unwrap();
    assert_eq!(canonical, genome.canonical());
    let bytes = Program::import_legacy(&genome).to_bytes().unwrap();
    assert_eq!(
        bytes,
        Program::from_bytes(&bytes).unwrap().to_bytes().unwrap()
    );
    genome
}

#[test]
fn every_direct_wire_witness_evaluates_to_its_frozen_model_bits() {
    let models: Value = serde_json::from_str(MODELS).unwrap();
    let frozen = rows_by_name(models["models"].as_array().unwrap());
    let (x, y) = toy_data();
    for (name, source) in witness_sources() {
        let genome = GenomeIr::parse(&source).unwrap();
        let expected: Vec<u64> = frozen[name.as_str()]["predictions"]
            .as_str()
            .unwrap()
            .split_whitespace()
            .map(|word| u64::from_str_radix(word, 16).unwrap())
            .collect();
        let actual: Vec<u64> = fit_predict(
            &genome,
            &x[..TRAINING_ROWS].to_vec(),
            &y[..TRAINING_ROWS],
            &x[TRAINING_ROWS..].to_vec(),
        )
        .unwrap_or_else(|error| panic!("{name}: evaluator: {error:#}"))
        .into_iter()
        .map(f64::to_bits)
        .collect();
        assert_eq!(actual, expected, "{name}: frozen evaluator bits");
    }
}

#[test]
fn every_direct_wire_witness_runs_emitted_rust_and_python_on_the_frozen_input() {
    let models: Value = serde_json::from_str(MODELS).unwrap();
    let frozen = rows_by_name(models["models"].as_array().unwrap());
    let (x, y) = toy_data();
    let scratch_root = std::env::var_os("TMPDIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")));
    fs::create_dir_all(&scratch_root).unwrap();
    let scratch = tempfile::Builder::new()
        .prefix("conversion-wire-matrix-")
        .tempdir_in(scratch_root)
        .unwrap();
    for (name, source) in witness_sources() {
        let dir = scratch.path().join(&name);
        let genome = GenomeIr::parse(&source).unwrap();
        let expected: Vec<u64> = frozen[name.as_str()]["predictions"]
            .as_str()
            .unwrap()
            .split_whitespace()
            .map(|word| u64::from_str_radix(word, 16).unwrap())
            .collect();
        assert_eq!(expected.len(), 16, "{name}: frozen row count");
        let (python, rust) = run_emitted_pair(&name, &genome, &dir, &x, &y);
        assert_eq!(python, expected, "{name}: Python bits");
        assert_eq!(rust, expected, "{name}: Rust bits");
    }
}

#[test]
fn standalone_cpp17_recipe_executes_every_admitted_direct_wire_witness() {
    let models: Value = serde_json::from_str(MODELS).unwrap();
    let frozen = rows_by_name(models["models"].as_array().unwrap());
    let emitted: Value = serde_json::from_str(EMITTED).unwrap();
    let frozen_emissions = rows_by_name(emitted["programs"].as_array().unwrap());
    let (x, y) = toy_data();
    let scratch_root = std::env::var_os("TMPDIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")));
    fs::create_dir_all(&scratch_root).unwrap();
    let scratch = tempfile::Builder::new()
        .prefix("conversion-wire-cpp17-")
        .tempdir_in(scratch_root)
        .unwrap();
    let mut admitted = 0;
    let mut refused = 0;
    for (name, source) in witness_sources() {
        let genome = GenomeIr::parse(&source).unwrap();
        let support = supports_cpp17(&genome);
        let emission = emit_cpp17(&genome, &EmitOptions::default());
        assert_eq!(support.is_ok(), emission.is_ok(), "{name}: C++17 admission");
        let frozen_route = frozen_emissions[name.as_str()]["cpp17"].as_str().unwrap();
        assert_eq!(
            support.is_err(),
            frozen_route.starts_with("unsupported:"),
            "{name}: frozen C++17 route"
        );
        let Ok(driver) = emission else {
            let reasons = support
                .unwrap_err()
                .into_iter()
                .map(|reason| {
                    reason
                        .split(['(', '{'])
                        .next()
                        .unwrap()
                        .trim_end()
                        .to_owned()
                })
                .collect::<Vec<_>>()
                .join("; ");
            assert_eq!(
                reasons,
                frozen_route.strip_prefix("unsupported: ").unwrap(),
                "{name}: frozen C++17 refusal reasons"
            );
            refused += 1;
            continue;
        };
        admitted += 1;
        let actual = run_cpp_recipe(&name, driver, &scratch.path().join(&name), &x, &y);
        let expected: Vec<u64> = frozen[name.as_str()]["predictions"]
            .as_str()
            .unwrap()
            .split_whitespace()
            .map(|word| u64::from_str_radix(word, 16).unwrap())
            .collect();
        assert_eq!(actual, expected, "{name}: standalone C++17 recipe bits");
    }
    assert_eq!(admitted, 6, "standalone C++17 route inventory");
    assert_eq!(refused, 13, "each other direct wire witness classified");
}

#[test]
fn expression_cpp17_recipe_executes_admitted_wire_witnesses_and_retains_refusals() {
    let models: Value = serde_json::from_str(MODELS).unwrap();
    let frozen = rows_by_name(models["models"].as_array().unwrap());
    let emitted: Value = serde_json::from_str(EMITTED).unwrap();
    let frozen_emissions = rows_by_name(emitted["programs"].as_array().unwrap());
    let (x, y) = toy_data();
    let scratch = tempfile::tempdir().unwrap();
    let mut admitted = 0;
    let mut refused = 0;
    for (name, source) in witness_sources() {
        let genome = round_trip(&source);
        let legacy = frozen_emissions[name.as_str()]["cpp17"].as_str().unwrap();
        match emit_expression_cpp17(&genome, &EmitOptions::default()) {
            Ok(recipe) => {
                admitted += 1;
                assert_eq!(recipe.profile, PROFILE);
                assert!(recipe.source.starts_with(&format!("// {PROFILE}\n")));
                let actual =
                    run_cpp_recipe(&name, recipe.source, &scratch.path().join(&name), &x, &y);
                let expected: Vec<u64> = frozen[name.as_str()]["predictions"]
                    .as_str()
                    .unwrap()
                    .split_whitespace()
                    .map(|word| u64::from_str_radix(word, 16).unwrap())
                    .collect();
                assert_eq!(actual, expected, "{name}: expression recipe frozen bits");
                if legacy.starts_with("unsupported:") {
                    assert!(matches!(
                        name.as_str(),
                        "golden_map_derive_ridge" | "golden_map_pow_elasticnet"
                    ));
                    assert!(emit_cpp17(&genome, &EmitOptions::default()).is_err());
                }
            }
            Err(feat_math::kernel::Error::Unsupported(reason)) => {
                refused += 1;
                // As with supports_cpp17 above, compare element kinds; debug payloads
                // add parameter context but do not change the refused route.
                let labels = reason
                    .split(", ")
                    .filter(|part| {
                        part.starts_with("head ")
                            || part.starts_with("transform ")
                            || *part == "target transforms"
                    })
                    .map(|part| part.split(['(', '{']).next().unwrap().trim_end())
                    .collect::<Vec<_>>()
                    .join("; ");
                assert_eq!(
                    labels,
                    legacy.strip_prefix("unsupported: ").unwrap(),
                    "{name}: structured expression recipe refusal: {reason}"
                );
            }
            Err(error) => panic!("{name}: unexpected expression recipe error: {error}"),
        }
    }
    assert_eq!(admitted, 8, "six legacy plus two expression wire witnesses");
    assert_eq!(refused, 11, "unsupported heads/stages/targets stay refused");
}

#[test]
fn seeded_composed_expression_recipes_match_rust_python_and_evaluator_bits() {
    let programs = [
        "(pipe X (map expr=\"round(x)\" append=true) (derive expr=\"c0+c5\" append=true) (ridge))",
        "(pipe X (derive expr=\"c1/c0\" append=true) (map expr=\"clip(x,-2,2)\") (ridge))",
        "(pipe X (map expr=\"abs(x)\" append=true) (derive expr=\"row_sum-row_mean+c5\" append=true) (elasticnet alpha=0.01 l1_ratio=0.5 sweeps=30))",
    ];
    let scratch = tempfile::tempdir().unwrap();
    for seed in [41, 43, 47] {
        let (mut x, y) = toy_data_seed(seed);
        // Targets remain finite and unchanged; the pipeline sanitizes features before fit.
        for (index, value) in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY]
            .into_iter()
            .enumerate()
        {
            x[index][index] = value;
            x[66 + index][index] = value;
        }
        x[64][0] = -0.0;
        x[65][1] = f64::from_bits(1);
        x[69] = vec![0.0, -0.0, 0.0, -0.0, 0.0];
        x[70][0] = f64::from_bits(1e-12_f64.to_bits() - 1);
        x[71][0] = f64::from_bits(1e-12_f64.to_bits() + 1);
        assert!(y.iter().all(|value| value.is_finite()));
        for (index, source) in programs.into_iter().enumerate() {
            let name = format!("seed-{seed}-program-{index}");
            let genome = round_trip(source);
            let expected: Vec<u64> = fit_predict(
                &genome,
                &x[..TRAINING_ROWS].to_vec(),
                &y[..TRAINING_ROWS],
                &x[TRAINING_ROWS..].to_vec(),
            )
            .unwrap()
            .into_iter()
            .map(f64::to_bits)
            .collect();
            let dir = scratch.path().join(&name);
            let (python, rust) = run_emitted_pair(&name, &genome, &dir, &x, &y);
            assert_eq!(python, expected, "{name}: Python bits");
            assert_eq!(rust, expected, "{name}: Rust bits");
            let recipe = emit_expression_cpp17(&genome, &EmitOptions::default()).unwrap();
            assert_eq!(recipe.profile, PROFILE);
            let cpp = run_cpp_recipe(&name, recipe.source, &dir, &x, &y);
            assert_eq!(cpp, expected, "{name}: C++17 expression recipe bits");
        }
    }
}

#[test]
fn seeded_stage_parameter_recipes_match_three_languages_and_evaluator_bits() {
    let programs = [
        "(pipe X (quantile_bins bins=2) (ridge lambda_frac=0.1))",
        "(pipe X (quantile_bins bins=16 append=true) (elasticnet alpha=0.02 l1_ratio=0.25 sweeps=17))",
        "(pipe X (cheb degree=1) (ridge lambda_frac=0.03))",
        "(pipe X (cheb degree=6 append=true) (ridge lambda_frac=2))",
        "(pipe X (step t=-0.5) (ridge lambda_frac=0.2))",
        "(pipe X (step t=0.75 append=true) (elasticnet alpha=0.03 l1_ratio=0.75 sweeps=23))",
        "(pipe X (pair_prod topk=2) (ridge lambda_frac=0.04))",
        "(pipe X (pair_prod topk=4 append=true) (quantile_bins bins=3) (elasticnet alpha=0.03 l1_ratio=0.5 sweeps=19))",
    ];
    let mutations = [
        ("quantile_bins.bins", 0, "bins=2", "bins=3"),
        ("ridge.lambda_frac", 0, "lambda_frac=0.1", "lambda_frac=0.2"),
        ("cheb.degree", 2, "degree=1", "degree=2"),
        ("step.t", 4, "t=-0.5", "t=-0.25"),
        ("pair_prod.topk", 6, "topk=2", "topk=3"),
        ("elasticnet.alpha", 1, "alpha=0.02", "alpha=0.04"),
        ("elasticnet.l1_ratio", 1, "l1_ratio=0.25", "l1_ratio=0.6"),
        ("elasticnet.sweeps", 1, "sweeps=17", "sweeps=3"),
    ];
    let scratch = tempfile::tempdir().unwrap();
    let mut cases = 0;
    let mut sensitivities = 0;
    for seed in [53, 59] {
        let (mut x, y) = toy_data_seed(seed);
        // Fixed boundary features supplement each seed. Targets remain finite;
        // the same training split feeds native fit, CSV recipes and the C++ ABI.
        x[0][0] = -0.0;
        x[1][1] = f64::from_bits(1);
        x[64][0] = -0.0;
        x[65][1] = f64::from_bits(1);
        x[66][2] = -0.5;
        x[67][2] = 0.75;
        let check = |name: &str, source: &str| {
            let genome = round_trip(source);
            assert!(supports(&genome).is_ok(), "{name}: paired admission");
            assert!(
                supports_cpp17(&genome).is_ok(),
                "{name}: default C++17 admission"
            );
            let expected: Vec<u64> = fit_predict(
                &genome,
                &x[..TRAINING_ROWS].to_vec(),
                &y[..TRAINING_ROWS],
                &x[TRAINING_ROWS..].to_vec(),
            )
            .unwrap()
            .into_iter()
            .map(f64::to_bits)
            .collect();
            assert_eq!(expected.len(), 16, "{name}: evaluator output count");
            let dir = scratch.path().join(name);
            let (python, rust) = run_emitted_pair(name, &genome, &dir, &x, &y);
            assert_eq!(python, expected, "{name}: Python parameter recipe bits");
            assert_eq!(rust, expected, "{name}: Rust parameter recipe bits");
            let source = emit_cpp17(&genome, &EmitOptions::default()).unwrap();
            let cpp = run_cpp_recipe(name, source, &dir, &x, &y);
            assert_eq!(cpp, expected, "{name}: C++17 parameter recipe bits");
            expected
        };
        let originals: Vec<_> = programs
            .iter()
            .enumerate()
            .map(|(index, source)| {
                check(&format!("parameters-seed-{seed}-program-{index}"), source)
            })
            .collect();
        cases += originals.len();
        for (parameter, index, from, to) in mutations {
            assert_eq!(
                programs[index].matches(from).count(),
                1,
                "{parameter}: one consumed key"
            );
            let changed = programs[index].replacen(from, to, 1);
            let words = check(
                &format!("parameters-seed-{seed}-changed-{parameter}"),
                &changed,
            );
            assert_ne!(
                words, originals[index],
                "seed{seed}/{parameter}: parameter must affect native held-out predictions"
            );
            sensitivities += 1;
            cases += 1;
        }
    }
    assert_eq!(
        cases, 32,
        "eight originals and eight mutations across two seeds"
    );
    assert_eq!(
        sensitivities, 16,
        "each consumed key changes each seed's predictions"
    );
}

#[test]
fn symbolic_and_portfolio_forms_execute_or_keep_their_structured_refusal() {
    let cases = [
        ("symbolic_raw", "(sym raw \"sin(c0)+row_mean\")", true),
        ("symbolic_affine", "(sym affine \"c0+0.5\")", true),
        ("compact_portfolio", "(pipe X (portfolio preset=core))", true),
        (
            "nested_portfolio",
            "(portfolio (member (pipe X (ridge))) (member (pipe X (robust_scale) (ridge))) (blend))",
            false,
        ),
    ];
    let (x, y) = toy_data();
    let scratch_root = std::env::var_os("TMPDIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")));
    fs::create_dir_all(&scratch_root).unwrap();
    let scratch = tempfile::Builder::new()
        .prefix("conversion-genome-forms-")
        .tempdir_in(scratch_root)
        .unwrap();
    for (name, source, admitted) in cases {
        let genome = GenomeIr::parse(source).unwrap();
        let canonical = genome.canonical();
        let parsed = GenomeIr::parse(&canonical).unwrap();
        assert_eq!(
            parsed.canonical(),
            canonical,
            "{name}: canonical fixed point"
        );
        let program = Program::import_legacy(&genome);
        let bytes = program.to_bytes().unwrap();
        assert_eq!(
            Program::import_legacy(&parsed).to_bytes().unwrap(),
            bytes,
            "{name}: executable tree"
        );
        assert_eq!(
            Program::parse_lossless_text(&program.to_lossless_text().unwrap())
                .unwrap()
                .to_bytes()
                .unwrap(),
            bytes,
            "{name}: typed envelope"
        );
        let paired = supports(&parsed);
        assert_eq!(paired.is_ok(), admitted, "{name}: paired route admission");
        let cpp = supports_cpp17(&parsed);
        assert!(cpp.is_err(), "{name}: standalone C++17 must refuse");
        assert_eq!(
            emit_cpp17(&parsed, &EmitOptions::default()).is_ok(),
            cpp.is_ok(),
            "{name}: C++17 public route"
        );
        match name {
            "symbolic_raw" | "symbolic_affine" => {
                assert_eq!(cpp.unwrap_err(), vec!["symbolic formula programs"]);
            }
            "nested_portfolio" => {
                let refusal = "portfolio: explicit nested member emission is not supported yet; use the compact head form (pipe X ... (portfolio preset=core ...))";
                assert_eq!(paired.unwrap_err(), vec![refusal]);
                assert_eq!(cpp.unwrap_err(), vec![refusal]);
                assert_eq!(
                    emit(&parsed, &EmitOptions::default())
                        .unwrap_err()
                        .to_string(),
                    format!("unsupported program elements: {refusal}")
                );
                continue;
            }
            "compact_portfolio" => assert!(!cpp.unwrap_err().is_empty()),
            _ => unreachable!(),
        }
        let expected: Vec<u64> = fit_predict(
            &parsed,
            &x[..TRAINING_ROWS].to_vec(),
            &y[..TRAINING_ROWS],
            &x[TRAINING_ROWS..].to_vec(),
        )
        .unwrap_or_else(|error| panic!("{name}: evaluator: {error:#}"))
        .into_iter()
        .map(f64::to_bits)
        .collect();
        let (python, rust) = run_emitted_pair(name, &parsed, &scratch.path().join(name), &x, &y);
        assert_eq!(python, expected, "{name}: Python recipe bits");
        assert_eq!(rust, expected, "{name}: Rust recipe bits");
    }
}
