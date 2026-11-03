//! Bounded local refit/export of a supplied DSL, without discovery or ranker admission.
//! Selection/checkpoint causality belongs to the external caller's receipt.
use anyhow::{ensure, Context, Result};
use jope_runtime::{
    limits::Limits,
    math_pipeline::{InputContract, NumericColumn, PipelineState, INPUT_PART, PIPELINE_PART},
    model_object::{Manifest, ModelObject, Part, Producer, MATH_PIPELINE, SCHEMA},
};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::Command,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const MAX_MODEL: u64 = 256 * 1024 * 1024;
const MAX_CSV: u64 = 8 * 1024 * 1024;
const MAX_SOURCE: usize = 16 * 1024 * 1024;
const ABS_TOL: f64 = 1e-12;
const REL_TOL: f64 = 1e-10;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Split {
    fit: Vec<usize>,
    holdout: Vec<usize>,
    seed: u64,
    label: String,
    #[serde(default)]
    input_sha256: Option<String>,
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn read_bounded(path: &Path, maximum: u64) -> Result<Vec<u8>> {
    let meta = fs::symlink_metadata(path)?;
    ensure!(
        meta.is_file() && meta.len() <= maximum,
        "invalid/oversized input: {}",
        path.display()
    );
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take(maximum + 1)
        .read_to_end(&mut bytes)?;
    ensure!(bytes.len() as u64 <= maximum, "input grew past bound");
    Ok(bytes)
}

fn write_json(path: &Path, value: &impl serde::Serialize) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}

fn numeric(cell: Option<&str>) -> Result<f64> {
    let value = cell
        .context("missing numeric cell")?
        .trim()
        .parse::<f64>()?;
    ensure!(value.is_finite(), "nonfinite numeric cell");
    Ok(value)
}

fn encode(value: f64, column: &NumericColumn) -> Result<f64> {
    ensure!(value.is_finite(), "nonfinite feature");
    let result = if column.constant {
        0.0
    } else {
        ((value - column.min) / (column.max - column.min)).clamp(0.0, 1.0)
    };
    ensure!(result.is_finite(), "normalization overflow");
    Ok(result)
}

fn check_split(split: &Split, rows: usize) -> Result<()> {
    ensure!(rows >= 2 && rows <= 4096, "row count outside 2..4096");
    ensure!(
        !split.fit.is_empty() && !split.holdout.is_empty(),
        "FIT/HOLDOUT must both be nonempty"
    );
    let all: BTreeSet<_> = split.fit.iter().chain(&split.holdout).copied().collect();
    ensure!(
        split.fit.len() + split.holdout.len() == rows
            && all.len() == rows
            && all.iter().copied().eq(0..rows),
        "split is not a disjoint complete partition"
    );
    let expected = jope_service::discover::splits::random_split(
        rows,
        split.fit.len(),
        split.seed,
        &split.label,
    )?;
    ensure!(
        expected.0 == split.fit && expected.1 == split.holdout,
        "split differs from production seed/label algorithm or row order"
    );
    Ok(())
}

fn predictions(bytes: &[u8]) -> Result<Vec<f64>> {
    std::str::from_utf8(bytes)?
        .lines()
        .map(|line| Ok(line.parse::<f64>()?))
        .collect()
}

fn parity(actual: &[f64], native: &[f64], scope: &str) -> Result<f64> {
    ensure!(
        actual.len() == native.len(),
        "{scope}: prediction row count differs"
    );
    let mut maximum = 0.0_f64;
    for (row, (&value, &expected)) in actual.iter().zip(native).enumerate() {
        ensure!(
            value.is_finite() && expected.is_finite(),
            "{scope}: nonfinite prediction at {row}"
        );
        let error = (value - expected).abs();
        ensure!(
            error <= ABS_TOL + REL_TOL * expected.abs(),
            "{scope}: row {row}: {value} versus {expected}"
        );
        maximum = maximum.max(error);
    }
    Ok(maximum)
}

fn command(program: &str, args: &[String], name: &str, directory: &Path) -> Result<Vec<u8>> {
    let program = if program == "rustc" {
        option_env!("RUSTC").unwrap_or("rustc")
    } else {
        program
    };
    let begin = Instant::now();
    let result = Command::new(program)
        .args(args)
        .env("RAYON_NUM_THREADS", "1")
        .env("OMP_NUM_THREADS", "1")
        .current_dir(directory)
        .output()?;
    fs::write(directory.join(format!("{name}.stdout")), &result.stdout)?;
    fs::write(directory.join(format!("{name}.stderr")), &result.stderr)?;
    write_json(
        &directory.join(format!("{name}.receipt.json")),
        &json!({
            "program":program,"args":args,"cwd":directory,"exit_code":result.status.code(),
            "process_wall_ns":begin.elapsed().as_nanos().to_string(),
            "stdout_sha256":sha(&result.stdout),"stderr_sha256":sha(&result.stderr),
            "scope":"compiler/self-check/prediction process; not neural or emission throughput"
        }),
    )?;
    ensure!(
        result.status.success(),
        "{name} failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    Ok(result.stdout)
}

fn rust_rows(rows: &[Vec<Option<String>>]) -> Result<String> {
    rows.iter()
        .map(|row| {
            let cells = row
                .iter()
                .map(|cell| -> Result<String> {
                    Ok(match cell {
                        Some(value) => format!("Some({})", serde_json::to_string(value)?),
                        None => "None".into(),
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(format!("vec![{}]", cells.join(",")))
        })
        .collect::<Result<Vec<_>>>()
        .map(|rows| rows.join(",\n"))
}

fn main() -> Result<()> {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    if args.len() == 3 && args[0] == "--features" {
        return export_features(Path::new(&args[1]), Path::new(&args[2]));
    }
    if args.len() == 1 && args[0] == "--help" {
        println!("usage: visual_fit_dsl SELECTED_DSL CSV ORIGINAL_SAMECASE_MODEL_DIR SPLIT_JSON NEW_EXTERNAL_OUTPUT_DIR");
        return Ok(());
    }
    ensure!(args.len() == 5, "usage: visual_fit_dsl SELECTED_DSL CSV ORIGINAL_SAMECASE_MODEL_DIR SPLIT_JSON NEW_EXTERNAL_OUTPUT_DIR");
    ensure!(
        !cfg!(debug_assertions),
        "working training parity requires a native --release reference to match emitted Rust -O; a debug/release comparison is not qualification"
    );
    let dsl_path = fs::canonicalize(&args[0])?;
    let csv_path = fs::canonicalize(&args[1])?;
    let original_path = fs::canonicalize(&args[2])?;
    let split_path = fs::canonicalize(&args[3])?;
    let output = PathBuf::from(&args[4]);
    ensure!(
        output.is_absolute() && !output.exists(),
        "output must be a new absolute external directory"
    );
    let worktree = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .context("worktree root missing")?
        .canonicalize()?;
    let parent = output
        .parent()
        .context("output parent missing")?
        .canonicalize()?;
    ensure!(
        !parent.starts_with(&worktree),
        "generated model/source evidence must stay outside worktree"
    );

    let started_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_millis()
        .to_string();
    let started = Instant::now();
    let dsl_bytes = read_bounded(&dsl_path, 64 * 1024)?;
    let csv_bytes = read_bounded(&csv_path, MAX_CSV)?;
    let split_bytes = read_bounded(&split_path, 1024 * 1024)?;
    let split: Split = serde_json::from_slice(&split_bytes)?;
    let csv_sha = sha(&csv_bytes);
    // The old production split has no CSV digest; its adjacent executed proof does.
    let (expected_csv_sha, split_binding) = if let Some(expected) = &split.input_sha256 {
        (
            expected.clone(),
            json!({"path":split_path,"sha256":sha(&split_bytes),"field":"input_sha256"}),
        )
    } else {
        let path = split_path.with_file_name("summary.json");
        let bytes = read_bounded(&path, 1024 * 1024)?;
        let proof: Value = serde_json::from_slice(&bytes)?;
        let expected = proof["input_sha256"]
            .as_str()
            .context("split needs input_sha256 or adjacent summary.json input_sha256")?
            .to_owned();
        (
            expected,
            json!({"path":path,"sha256":sha(&bytes),"field":"input_sha256"}),
        )
    };
    ensure!(
        csv_sha == expected_csv_sha,
        "CSV bytes differ from split proof"
    );
    let table = jope_runtime::table::parse_csv(&csv_bytes, &Limits::default())?;
    check_split(&split, table.rows.len())?;
    let original = ModelObject::read(&original_path, MAX_MODEL)?;
    original.verify_executable()?;
    let old = original.math_pipeline()?.input;
    ensure!(
        old.prediction_version == 2,
        "require original f64 prediction_version=2 contract"
    );
    let input = InputContract::new(old.columns, old.selected, old.target)?;
    ensure!(
        input.columns.len() <= 256 && input.selected.len() <= 256,
        "feature count exceeds 256"
    );
    let at = input
        .columns
        .iter()
        .map(|column| {
            table
                .headers
                .iter()
                .position(|name| name == &column.name)
                .with_context(|| format!("missing feature {:?}", column.name))
        })
        .collect::<Result<Vec<_>>>()?;
    let target_at = table
        .headers
        .iter()
        .position(|name| name == &input.target.name)
        .context("missing target")?;
    ensure!(
        !input
            .columns
            .iter()
            .any(|column| column.name == input.target.name),
        "target leaked into features"
    );
    let raw_x = table
        .rows
        .iter()
        .map(|row| {
            at.iter()
                .map(|&j| numeric(row[j].as_deref()))
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    let raw_y = table
        .rows
        .iter()
        .map(|row| numeric(row[target_at].as_deref()))
        .collect::<Result<Vec<_>>>()?;
    // Exact FIT-only range checks prevent borrowing another dataset's old input envelope.
    for (j, column) in input.columns.iter().enumerate() {
        let lo = split
            .fit
            .iter()
            .map(|&i| raw_x[i][j])
            .reduce(f64::min)
            .context("empty FIT")?;
        let hi = split
            .fit
            .iter()
            .map(|&i| raw_x[i][j])
            .reduce(f64::max)
            .context("empty FIT")?;
        ensure!(
            lo.to_bits() == column.min.to_bits()
                && hi.to_bits() == column.max.to_bits()
                && column.constant == (lo == hi),
            "FIT-only feature range mismatch for {:?}",
            column.name
        );
    }
    let target_min = split
        .fit
        .iter()
        .map(|&i| raw_y[i])
        .reduce(f64::min)
        .context("empty target")?;
    let target_max = split
        .fit
        .iter()
        .map(|&i| raw_y[i])
        .reduce(f64::max)
        .context("empty target")?;
    let target_scale = (target_max - target_min).max(1e-12);
    ensure!(
        target_min.to_bits() == input.target.min.to_bits()
            && target_scale.to_bits() == input.target.scale.to_bits(),
        "FIT-only target range mismatch"
    );
    let encoded = raw_x
        .iter()
        .map(|row| {
            input
                .selected
                .iter()
                .map(|&j| encode(row[j], &input.columns[j]))
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    let fit_x = split
        .fit
        .iter()
        .map(|&i| encoded[i].clone())
        .collect::<Vec<_>>();
    let fit_y = split
        .fit
        .iter()
        .map(|&i| (raw_y[i] - input.target.min) / input.target.scale)
        .collect::<Vec<_>>();
    ensure!(
        fit_y.iter().all(|v| v.is_finite()),
        "target normalization overflow"
    );
    let genome = feat_math::GenomeIr::parse(std::str::from_utf8(&dsl_bytes)?.trim())?;
    let canonical = genome.canonical();
    let fit_begin = Instant::now();
    let fitted = feat_math::model::fit(&genome, &fit_x, &fit_y)?;
    let fit_ns = fit_begin.elapsed().as_nanos().to_string();
    let parts: BTreeMap<String, Vec<u8>> = BTreeMap::from([
        (
            PIPELINE_PART.into(),
            serde_json::to_vec(&PipelineState {
                dsl: canonical.clone(),
                model: fitted,
            })?,
        ),
        (INPUT_PART.into(), serde_json::to_vec(&input)?),
    ]);
    let object = ModelObject::new(
        Manifest {
            schema: SCHEMA.into(),
            minimum_reader_version: 1,
            format: MATH_PIPELINE.into(),
            producer: Producer::default(),
            primary: PIPELINE_PART.into(),
            required_capabilities: vec![],
            files: parts
                .iter()
                .map(|(path, bytes)| Part::new(path, "application/json", bytes))
                .collect(),
            metadata: json!({"fitting":"local_selected_dsl_fit_only","prediction_version":2,"training_rows":split.fit.len(),"missing_input_policy":"refused by this numeric proof helper"}),
        },
        parts,
        MAX_MODEL,
    )?;
    object.verify_executable()?;
    let (native, warnings) = object.math_pipeline()?.predict_table(&table)?;
    ensure!(
        native.len() == table.rows.len() && native.iter().all(|v| v.is_finite()),
        "invalid native predictions"
    );
    let rust = object.export_source("rust")?;
    let python = object.export_source("python")?;
    ensure!(
        !rust.is_empty()
            && !python.is_empty()
            && rust.len() <= MAX_SOURCE
            && python.len() <= MAX_SOURCE,
        "source outside bound"
    );

    fs::create_dir(&output)?;
    let output = output.canonicalize()?;
    let model_dir = object.save(&output.join("models"), MAX_MODEL)?;
    let rust_path = output.join("predictor.rs");
    let python_path = output.join("predictor.py");
    fs::write(&rust_path, &rust)?;
    fs::write(&python_path, &python)?;
    fs::write(output.join("selected.dsl"), &dsl_bytes)?;
    fs::write(output.join("pipeline.dsl"), &canonical)?;
    fs::write(output.join("input.csv"), &csv_bytes)?;
    fs::write(output.join("split.json"), &split_bytes)?;
    write_json(&output.join("native-predictions.json"), &native)?;
    let proof = output.join("proof");
    fs::create_dir(&proof)?;
    command(
        "rustc",
        &[
            "--edition=2021".into(),
            "--test".into(),
            rust_path.display().to_string(),
            "-o".into(),
            proof.join("self-check").display().to_string(),
        ],
        "rust-selfcheck-compile",
        &proof,
    )?;
    command(
        &proof.join("self-check").display().to_string(),
        &[],
        "rust-selfcheck",
        &proof,
    )?;
    command(
        "python3",
        &[python_path.display().to_string(), "--self-check".into()],
        "python-selfcheck",
        &proof,
    )?;

    let model_cells = table
        .rows
        .iter()
        .map(|row| at.iter().map(|&j| row[j].clone()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let rows_literal = rust_rows(&model_cells)?;
    write_json(&proof.join("feature-rows.json"), &model_cells)?;
    let driver = proof.join("driver.rs");
    fs::write(&driver, format!("#[path = {}]\nmod predictor;\nfn main() {{\n let rows:Vec<Vec<Option<&str>>>=vec![{}];\n for value in predictor::predict_cells(&rows).unwrap() {{ println!(\"{{:.17e}}\",value); }}\n}}\n", serde_json::to_string(&rust_path.display().to_string())?, rows_literal))?;
    command(
        "rustc",
        &[
            "--edition=2021".into(),
            "-O".into(),
            driver.display().to_string(),
            "-o".into(),
            proof.join("predictor").display().to_string(),
        ],
        "rust-compile",
        &proof,
    )?;
    let rust_predictions = predictions(&command(
        &proof.join("predictor").display().to_string(),
        &[],
        "rust-predictions",
        &proof,
    )?)?;
    let py_script = "import importlib.util,json,sys; s=importlib.util.spec_from_file_location('verified_predictor',sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); rows=json.load(open(sys.argv[2])); print(json.dumps(m.predict_rows(rows),allow_nan=False))";
    let py_stdout = command(
        "python3",
        &[
            "-c".into(),
            py_script.into(),
            python_path.display().to_string(),
            proof.join("feature-rows.json").display().to_string(),
        ],
        "python-predictions",
        &proof,
    )?;
    let python_predictions: Vec<f64> = serde_json::from_slice(&py_stdout)?;
    let rust_error = parity(&rust_predictions, &native, "Rust/native")?;
    let python_error = parity(&python_predictions, &native, "Python/native")?;
    let cross_error = parity(&rust_predictions, &python_predictions, "Rust/Python")?;
    write_json(&proof.join("rust-predictions.json"), &rust_predictions)?;
    write_json(&proof.join("python-predictions.json"), &python_predictions)?;

    // Append a proof-only main after every unchanged byte of the complete export.
    let training_driver = format!(
        r#"
// Proof-only driver; every preceding byte is the unchanged exported artifact.
fn main() {{
 let rows:Vec<Vec<Option<&str>>>=vec![{}];
 let labels:&[f64]=&{:?};
 let fit_ids:&[usize]=&{:?};
 let encoded:Vec<Vec<f64>>=rows.iter().map(|row| INPUT_SELECTED.iter().map(|&j|encode_cell(row[j],j).unwrap()).collect()).collect();
 let x:Vec<Vec<f64>>=fit_ids.iter().map(|&i|encoded[i].clone()).collect();
 let y:Vec<f64>=fit_ids.iter().map(|&i|(labels[i]-TARGET_MIN)/TARGET_SCALE).collect();
 let model=Model::fit(&x,&y);
 for p in model.predict(&encoded) {{
  let value=if TARGET_SCALE<=1e-12 {{TARGET_MIN}} else {{TARGET_MIN+p*TARGET_SCALE}};
  println!("{{:.17e}}",value);
 }}
}}
"#,
        rows_literal, raw_y, split.fit
    );
    let mut training_bytes = rust.clone();
    training_bytes.extend_from_slice(training_driver.as_bytes());
    let training_path = proof.join("train-driver.rs");
    fs::write(&training_path, &training_bytes)?;
    ensure!(
        fs::read(&training_path)?.starts_with(&rust),
        "training proof lost exported prefix"
    );
    command(
        "rustc",
        &[
            "--edition=2021".into(),
            "-O".into(),
            training_path.display().to_string(),
            "-o".into(),
            proof.join("train-predictor").display().to_string(),
        ],
        "rust-training-compile",
        &proof,
    )?;
    let trained = predictions(&command(
        &proof.join("train-predictor").display().to_string(),
        &[],
        "rust-training-predictions",
        &proof,
    )?)?;
    let training_error = parity(&trained, &native, "emitted Rust Model::fit/native")?;
    write_json(&proof.join("trained-predictions.json"), &trained)?;
    let training_proof = json!({"schema":"jope.emitted-training-proof/v1","artifact_sha256":sha(&rust),"derivative_source_sha256":sha(&training_bytes),"unchanged_artifact_prefix_bytes":rust.len(),"fit_rows":split.fit.len(),"prediction_rows":native.len(),"heldout_rows":split.holdout.len(),"max_abs_error_vs_native_gp":training_error,"parity_abs_tolerance":ABS_TOL,"parity_rel_tolerance":REL_TOL,"scope":"emitted Model::fit on exact FIT rows under exported input/target contract, then all-row prediction versus this supplied-DSL native refit"});
    write_json(&proof.join("training-summary.json"), &training_proof)?;
    ensure!(
        fs::read(&rust_path)? == rust && fs::read(&python_path)? == python,
        "executed source changed"
    );
    ensure!(
        object.export_source("rust")? == rust && object.export_source("python")? == python,
        "fresh source emission bytes differ"
    );
    ensure!(
        read_bounded(&dsl_path, 64 * 1024)? == dsl_bytes
            && read_bounded(&csv_path, MAX_CSV)? == csv_bytes
            && read_bounded(&split_path, 1024 * 1024)? == split_bytes,
        "input changed during proof"
    );
    let saved = ModelObject::read(&model_dir, MAX_MODEL)?;
    ensure!(
        saved.id() == object.id() && saved.parts() == object.parts(),
        "saved model differs"
    );
    let input_bytes = &object.parts()[INPUT_PART];
    let pipeline_bytes = &object.parts()[PIPELINE_PART];
    let receipt = json!({
        "schema":"jope.visual-selected-dsl-fit/v1","state":"pass","scope":"local supplied-DSL native refit and working source proof; selection/checkpoint causality belongs to caller; no registered Engine champion or ranker admission claim",
        "started_unix_ms":started_ms,"wall_ns":started.elapsed().as_nanos().to_string(),"native_fit_ns":fit_ns,
        "command_argv":std::env::args().collect::<Vec<_>>(),"cwd":std::env::current_dir()?,
        "helper_source_sha256":sha(include_bytes!("visual_fit_dsl.rs")),
        "native_build_profile":"release","native_debug_assertions":cfg!(debug_assertions),
        "emitted_rust_training_profile":"-O","rustc_program":option_env!("RUSTC").unwrap_or("rustc"),
        "input_csv":{"path":csv_path,"sha256":csv_sha,"bytes":csv_bytes.len()},
        "selected_dsl":{"path":dsl_path,"sha256":sha(&dsl_bytes),"canonical_sha256":sha(canonical.as_bytes()),"canonical":canonical},
        "split":{"path":split_path,"sha256":sha(&split_bytes),"binding":split_binding,"seed":split.seed,"label":split.label,"fit":split.fit,"holdout":split.holdout},
        "original_contract_model":{"path":original_path,"model_id":original.id(),"input_sha256":sha(&original.parts()[INPUT_PART])},
        "model":{"path":model_dir,"model_id":object.id(),"input_sha256":sha(input_bytes),"pipeline_sha256":sha(pipeline_bytes),"prediction_version":2},
        "normalized_fit":{"features_sha256":sha(&serde_json::to_vec(&fit_x)?),"targets_sha256":sha(&serde_json::to_vec(&fit_y)?),"columns":input.columns,"selected":input.selected,"fit_ranges_exact":true},
        "prediction_rows":native.len(),"fit_rows":split.fit.len(),"heldout_rows":split.holdout.len(),"ignored_table_columns":warnings,
        "native_predictions_sha256":sha(&fs::read(output.join("native-predictions.json"))?),
        "source_sha256":{"rust":sha(&rust),"python":sha(&python)},"source_bytes":{"rust":rust.len(),"python":python.len()},
        "rust_self_check":"pass","python_self_check":"pass","self_check_rows_per_language":64,"source_unchanged":true,
        "parity_abs_tolerance":ABS_TOL,"parity_rel_tolerance":REL_TOL,
        "max_abs_errors":{"rust_vs_native":rust_error,"python_vs_native":python_error,"rust_vs_python":cross_error},
        "training_parity":training_proof,"neural_decoding_tokens_per_second":null,"emission_tokens_per_second":null
    });
    write_json(&output.join("receipt.json"), &receipt)?;
    println!(
        "PASS supplied-DSL refit/export: {} FIT, {} HOLDOUT, {} predictions; {}",
        split.fit.len(),
        split.holdout.len(),
        native.len(),
        output.display()
    );
    Ok(())
}

fn export_features(request_path: &Path, output: &Path) -> Result<()> {
    use jope_service::ranker::{featurizer::Featurizer, proj16::proj16};
    let bytes = read_bounded(request_path, 1024 * 1024)?;
    let request: Value = serde_json::from_slice(&bytes)?;
    ensure!(
        request["schema"] == "jain.h256-program-slate-request/v1",
        "unsupported feature request"
    );
    ensure!(
        output.is_absolute() && !output.exists(),
        "feature output must be a new absolute file"
    );
    let cases = request["cases"]
        .as_array()
        .context("missing feature cases")?;
    ensure!(
        !cases.is_empty() && cases.len() <= 8,
        "expected 1..8 feature cases"
    );
    let featurizer = Featurizer::default();
    let mut results = Vec::new();
    for case in cases {
        let programs = case["programs"]
            .as_array()
            .context("missing ordered programs")?;
        ensure!(
            !programs.is_empty() && programs.len() <= 64,
            "expected 1..64 ordered programs"
        );
        let mut raw = Vec::new();
        let mut projections = Vec::new();
        let mut canonical = Vec::new();
        for program in programs {
            let dsl = program.as_str().context("program is not text")?;
            ensure!(dsl.len() <= 16384, "DSL exceeds feature limit");
            let genome = feat_math::GenomeIr::parse(dsl)?;
            let text = genome.canonical();
            let features = featurizer.featurize(&text)?;
            projections.extend(proj16(&features.map(f64::from)));
            raw.extend(features);
            canonical.push(text);
        }
        ensure!(
            canonical
                .iter()
                .zip(programs)
                .all(|(a, b)| b.as_str() == Some(a.as_str())),
            "slate programs were not canonical; refuse an implicit identity change"
        );
        results.push(
            json!({"id":case["id"],"dataset_sha256":case["dataset_sha256"],
            "native_call_id":case["native_call_id"],"programs":canonical,
            "raw_program_features412":raw,"projection16_f64":projections,
            "provenance":case["provenance"]}),
        );
    }
    write_json(
        output,
        &json!({"schema":"jain.h256-native-slate-features/v1",
        "request_sha256":sha(&bytes),"helper_source_sha256":sha(include_bytes!("visual_fit_dsl.rs")),
        "feature_recipe_sha256":jope_service::ranker::featurizer::REFERENCE_SOURCE_SHA256,
        "feature_names_sha256":jope_service::ranker::featurizer::FEATURE_NAMES_SHA256,
        "scope":"native DSL-only raw412 and frozen proj16; no cost prediction, fitting or GP selection",
        "cases":results}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f64_contract_retains_precision_clamps_and_refuses_bad_cells() {
        let column = NumericColumn {
            name: "x".into(),
            min: 0.0,
            max: 1.0,
            constant: false,
        };
        let precise = 0.123456789012345_f64;
        assert_eq!(
            encode(numeric(Some("0.123456789012345")).unwrap(), &column)
                .unwrap()
                .to_bits(),
            precise.to_bits()
        );
        assert_eq!(encode(-1.0, &column).unwrap(), 0.0);
        assert_eq!(encode(2.0, &column).unwrap(), 1.0);
        assert!(numeric(None).is_err());
        assert!(numeric(Some("NaN")).is_err());
        assert!(numeric(Some("yes")).is_err());
    }

    #[test]
    fn split_refuses_duplicate_members_and_wrong_seed_order() {
        let (fit, holdout) =
            jope_service::discover::splits::random_split(8, 6, 12, "test").unwrap();
        let mut split = Split {
            fit,
            holdout,
            seed: 12,
            label: "test".into(),
            input_sha256: None,
        };
        check_split(&split, 8).unwrap();
        split.fit.swap(0, 1);
        assert!(check_split(&split, 8).is_err());
        split.fit.swap(0, 1);
        split.holdout[0] = split.fit[0];
        assert!(check_split(&split, 8).is_err());
    }
}
