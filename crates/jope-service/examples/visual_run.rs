//! One actual CPU Engine run: native conditioning/admission observations, its
//! committed champion, the production export route, and executed source parity.
//! This trusted-host example starts no listener and accepts only signed assets.
use anyhow::{ensure, Context, Result};
use axum::{
    body::{to_bytes, Body},
    http::Request,
    Router,
};
use jope_runtime::model_object::ModelObject;
use jope_service::{
    dataset::register,
    engine::InProcessEngine,
    http::auth::{BearerToken, LocalAuth},
    ranker::cuda::release,
    run::learned::AdmittedRanker,
    store::RecordKey,
    visual_trace::begin_run_journal,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{Cursor, Read},
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant},
};
use tower::ServiceExt;

const TOKEN: &str = "abababababababababababababababababababababababababababababababab";

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn write_json(path: &Path, value: &impl serde::Serialize) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}

async fn request(
    app: &Router,
    method: &str,
    path: &str,
    body: Option<Value>,
    key: &str,
) -> Result<Vec<u8>> {
    let mut builder = Request::builder()
        .method(method)
        .uri(path)
        .header("host", "127.0.0.1:31234")
        .header("authorization", format!("Bearer {TOKEN}"))
        .header("accept", "application/json");
    if body.is_some() {
        builder = builder
            .header("content-type", "application/json")
            .header("idempotency-key", key);
    }
    let response = app
        .clone()
        .oneshot(builder.body(match body {
            Some(value) => Body::from(serde_json::to_vec(&value)?),
            None => Body::empty(),
        })?)
        .await?;
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 64 * 1024 * 1024)
        .await?
        .to_vec();
    ensure!(
        status.is_success(),
        "{method} {path}: {status}: {}",
        String::from_utf8_lossy(&bytes)
    );
    Ok(bytes)
}

fn command(program: &str, args: &[String], name: &str, directory: &Path) -> Result<Vec<u8>> {
    let begin = Instant::now();
    let result = Command::new(program)
        .args(args)
        .env("RAYON_NUM_THREADS", "1")
        .env("OMP_NUM_THREADS", "1")
        .output()?;
    fs::write(directory.join(format!("{name}.stdout")), &result.stdout)?;
    fs::write(directory.join(format!("{name}.stderr")), &result.stderr)?;
    write_json(
        &directory.join(format!("{name}.receipt.json")),
        &json!({
            "program":program,"args":args,"exit_code":result.status.code(),
            "process_wall_ns":begin.elapsed().as_nanos().to_string(),
            "scope":"compiler or predictor process; not emission timing"
        }),
    )?;
    ensure!(
        result.status.success(),
        "{name} failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    Ok(result.stdout)
}

#[tokio::main(flavor = "multi_thread", worker_threads = 1)]
async fn main() -> Result<()> {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    ensure!(
        args.len() == 4,
        "usage: visual_run SIGNED_APPLIANCE SIGNED_DARWIN_WORKER INPUT_CSV_OR_DIRECTORY NEW_EXTERNAL_OUT"
    );
    let signed = Path::new(&args[0]);
    let worker = Path::new(&args[1]);
    let input_path = Path::new(&args[2]);
    let out = Path::new(&args[3]);
    ensure!(
        out.is_absolute() && !out.exists(),
        "output must be a new absolute external directory"
    );
    ensure!(
        !out.starts_with(std::env::current_dir()?),
        "bulk evidence must be outside the checkout"
    );
    let inputs: Vec<PathBuf> = if input_path.is_file() {
        vec![input_path.to_path_buf()]
    } else {
        let mut entries = fs::read_dir(input_path)?
            .filter_map(|v| v.ok())
            .map(|v| v.path())
            .filter(|v| {
                v.is_file()
                    && v.extension().is_some_and(|s| s == "csv")
                    && v.file_stem()
                        .and_then(|s| s.to_str())
                        .is_some_and(|s| s.starts_with("case-") && s.len() == 6)
            })
            .collect::<Vec<_>>();
        entries.sort();
        entries
    };
    ensure!(
        (1..=8).contains(&inputs.len()),
        "expected one to eight declared case CSV files"
    );
    fs::create_dir_all(out)?;
    let mut cases = Vec::new();
    let mut proofs = Vec::new();
    let mut models = Vec::new();
    let mut assets = Value::Null;
    for csv_path in inputs {
        let (case, proof, model, case_assets) =
            capture_case(signed, worker, &csv_path, out).await?;
        if assets.is_null() {
            assets = case_assets;
        } else {
            ensure!(
                assets == case_assets,
                "bundle assets changed across the catalog"
            );
        }
        cases.push(case);
        proofs.push(proof);
        models.push(model);
    }
    write_json(
        &out.join("export-proof-summary.json"),
        &json!({"cases":proofs}),
    )?;
    write_json(&out.join("model-paths.json"), &models)?;
    let runtime_head = Command::new("git").args(["rev-parse", "HEAD"]).output()?;
    let runtime_status = Command::new("git")
        .args([
            "status",
            "--porcelain",
            "--untracked-files=normal",
            "--",
            "crates/jope-service",
        ])
        .output()?;
    let runtime_clean = runtime_status.status.success() && runtime_status.stdout.is_empty();
    let runtime_head_text = String::from_utf8_lossy(&runtime_head.stdout)
        .trim()
        .to_owned();
    let source_bound =
        runtime_clean && option_env!("JOPE_BUILD_SOURCE_SHA") == Some(runtime_head_text.as_str());
    let trace = json!({"schema":"jain.visual-run-trace/v1","bundle_digest":release::R882,
        "source_commit":if source_bound {option_env!("JOPE_BUILD_SOURCE_SHA")} else {None},
        "build_base_commit":option_env!("JOPE_BUILD_SOURCE_SHA"),
        "runtime_checkout_head":runtime_head_text,"source_identity_verified":source_bound,
        "runtime_service_source_dirty":!runtime_clean,
        "capture_executable_sha256":sha(&fs::read(std::env::current_exe()?)?),
        "assets":assets,"cases":cases,"public_release":false,
        "placement":{"dope":"cpu","ranker":"cpu","evaluation":"cpu","workers":1},
        "capture_authority":"actual in-process RunService conditioning and selected admission calls, followed by its committed champion and production source export",
        "diagnostic_slate_substitution":false});
    write_json(&out.join("trace.json.partial"), &trace)?;
    fs::rename(out.join("trace.json.partial"), out.join("trace.json"))?;
    Ok(())
}

async fn capture_case(
    signed: &Path,
    worker: &Path,
    csv_path: &Path,
    out: &Path,
) -> Result<(Value, Value, Value, Value)> {
    let package = jope_assets::Package::open(signed)?.context("signed appliance required")?;
    package.verify_full()?;
    let bundle = release::load_r882_embedded(&package)?;
    let mut assets = serde_json::Map::new();
    for name in [
        "dope.safetensors",
        "jope.safetensors",
        "manifest.json",
        "featurizer.json",
    ] {
        let bytes = bundle.asset(name).context("authenticated asset missing")?;
        assets.insert(
            name.into(),
            json!({"sha256":sha(bytes),"bytes":bytes.len()}),
        );
    }
    let worker_package =
        jope_assets::Package::open(worker)?.context("signed worker appliance required")?;
    worker_package.verify_full()?;
    let case_id = csv_path
        .file_stem()
        .and_then(|v| v.to_str())
        .context("CSV stem required")?;
    ensure!(
        case_id.starts_with("case-") && case_id.len() == 6,
        "expected case-a..case-z input name"
    );
    let case = out.join(case_id);
    fs::create_dir(&case)?;
    let input = fs::read(csv_path)?;
    fs::write(out.join(format!("{case_id}.csv")), &input)?;
    let table = jope_runtime::table::parse_csv(&input, &Default::default())?;
    ensure!(
        table.rows.len() >= 64 && table.headers.len() == 5,
        "at least 64 rows and four numeric features required"
    );
    let target = table
        .headers
        .last()
        .context("target header missing")?
        .clone();
    let ranker = AdmittedRanker::r882(bundle)?;
    let engine = InProcessEngine::open_with_ranker(case.join("home"), worker, ranker)?;
    let dataset = register::register(engine.store(), input.as_slice(), &Default::default())?;
    let app = engine.router(LocalAuth::new(BearerToken::from_hex(TOKEN)?, 31234)?)?;
    let spec = json!({"dataset":dataset.id,"target":target,"task":"regression","ignore_columns":[],
        "test_dataset":null,"split":{"kind":"random","holdout_fraction":0.2,"group_column":null,"time_column":null},
        "objective":"rmse","effort":"fast","budget":{"evaluations":"24","wall_seconds":"120"},
        "landmarks":"full","guidance":"learned","languages":["rust","python"],"require_all_languages":true,
        "device":"cpu","seed":"20261103","dry_run":false,"plan_digest":null});
    write_json(&case.join("run-spec.json"), &spec)?;
    let journal = begin_run_journal()?;
    // A single start request creates the actual Plan and FIT conditioning once;
    // no unrelated dry-run conditioning or diagnostic slate is substituted.
    let started: Value = serde_json::from_slice(
        &request(&app, "POST", "/v1/runs", Some(spec), "visual-single-start").await?,
    )?;
    write_json(&case.join("run-start.json"), &started)?;
    let run_id = started["run_id"]
        .as_str()
        .context("run ID missing")?
        .to_owned();
    let begin = Instant::now();
    let finished = loop {
        let status: Value = serde_json::from_slice(
            &request(&app, "GET", &format!("/v1/runs/{run_id}"), None, "").await?,
        )?;
        if matches!(
            status["state"].as_str(),
            Some("completed" | "failed" | "cancelled")
        ) {
            break status;
        }
        ensure!(
            begin.elapsed() < Duration::from_secs(180),
            "run did not drain within its outer bound"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    };
    write_json(&case.join("run-finished.json"), &finished)?;
    let observed = journal.finish()?;
    write_json(&case.join("native-journal.json"), &observed)?;
    ensure!(
        observed.run_id.as_deref() == Some(&run_id),
        "journal is bound to another run"
    );
    ensure!(
        observed.dropped_records == 0,
        "native observations were dropped"
    );
    ensure!(
        observed
            .calls
            .iter()
            .any(|v| v.identity["phase"] == "ranker_admission"),
        "actual run never invoked learned admission"
    );
    ensure!(
        finished["state"] == "completed",
        "actual run did not complete: {finished}"
    );
    let parsed_run = run_id.parse()?;
    let committed = engine
        .store()
        .get(&RecordKey::Run(parsed_run))?
        .context("committed run missing")?;
    write_json(&case.join("committed-run.json"), &committed)?;
    let mut events = Vec::<Value>::new();
    let mut after = "0".to_owned();
    loop {
        let page: Value = serde_json::from_slice(
            &request(
                &app,
                "GET",
                &format!("/v1/runs/{run_id}/events?after={after}&limit=256"),
                None,
                "",
            )
            .await?,
        )?;
        let values = page["events"].as_array().context("event page missing")?;
        events.extend(values.iter().cloned());
        let next = page["next_after"]
            .as_str()
            .context("event cursor missing")?
            .to_owned();
        if next == after || next == finished["event_watermark"].as_str().unwrap_or("") {
            break;
        }
        after = next;
    }
    write_json(&case.join("events.json"), &events)?;
    let algorithm = finished["best"]["algorithm_id"]
        .as_str()
        .context("champion missing")?;
    let export: Value = serde_json::from_slice(&request(&app,"POST",&format!("/v1/runs/{run_id}/exports"),
        Some(json!({"algorithm_id":algorithm,"kind":"bundle","languages":["rust","python"],"require_all_languages":true,"refit":"fit"})),"visual-single-export").await?)?;
    write_json(&case.join("export-manifest.json"), &export)?;
    let archive = request(
        &app,
        "GET",
        &format!(
            "/v1/artifacts/{}",
            export["artifact_id"]
                .as_str()
                .context("export artifact missing")?
        ),
        None,
        "",
    )
    .await?;
    fs::write(case.join("export.zip"), &archive)?;
    let export_dir = case.join("export-rust-python");
    fs::create_dir(&export_dir)?;
    let mut zip = zip::ZipArchive::new(Cursor::new(&archive))?;
    for index in 0..zip.len() {
        let mut entry = zip.by_index(index)?;
        let relative = entry
            .enclosed_name()
            .context("unsafe archive path")?
            .to_path_buf();
        if entry.is_dir() {
            continue;
        }
        ensure!(entry.size() <= 64 * 1024 * 1024, "oversized export entry");
        let path = export_dir.join(relative);
        fs::create_dir_all(path.parent().context("export parent missing")?)?;
        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes)?;
        fs::write(path, bytes)?;
    }
    let model_id = export["model_digest"]
        .as_str()
        .context("model identity missing")?;
    let model_dir = case.join("home/models").join(model_id);
    let model = ModelObject::read(&model_dir, 256 * 1024 * 1024)?;
    ensure!(
        model.id() == model_id,
        "committed model differs from export model"
    );
    let pipeline = model.math_pipeline()?;
    let (native, input_warnings) = pipeline.predict_table(&table)?;
    ensure!(native.len() >= 64, "insufficient executed parity rows");
    let rust_path = export_dir.join("src/rust/predictor.rs");
    let python_path = export_dir.join("src/python/predictor.py");
    ensure!(
        model.export_source("rust")? == fs::read(&rust_path)?,
        "Rust artifact differs from this run's GP model"
    );
    ensure!(
        model.export_source("python")? == fs::read(&python_path)?,
        "Python artifact differs from this run's GP model"
    );
    let original_rust_sha = sha(&fs::read(&rust_path)?);
    let original_python_sha = sha(&fs::read(&python_path)?);
    let proof = case.join("proof");
    fs::create_dir(&proof)?;
    let fit_count = ((table.rows.len() as f64) * 0.8).floor() as usize;
    let (fit_ids, holdout_ids) = jope_service::discover::splits::random_split(
        table.rows.len(),
        fit_count,
        20261103,
        "public-run/fit-v1",
    )?;
    let mut fit_table = table.clone();
    fit_table.rows = fit_ids.iter().map(|&i| table.rows[i].clone()).collect();
    let fit_bytes = jope_service::dataset::jmat::encode(&fit_table)?;
    let conditioning = observed
        .calls
        .iter()
        .find(|call| call.identity["phase"] == "fit_conditioning")
        .context("actual FIT conditioning was not recorded")?;
    ensure!(
        conditioning.identity["fit_matrix_sha256"] == sha(&fit_bytes),
        "recorded conditioning does not use this run's actual FIT partition"
    );
    write_json(
        &proof.join("split.json"),
        &json!({"seed":20261103,"label":"public-run/fit-v1","fit":fit_ids,"holdout":holdout_ids}),
    )?;
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
    let driver = proof.join("driver.rs");
    fs::write(&driver, format!("#[path = {}]\nmod predictor;\nfn main() {{\n let text=std::fs::read_to_string(std::env::args().nth(1).unwrap()).unwrap();\n let rows:Vec<Vec<Option<&str>>>=text.lines().skip(1).map(|line|line.split(',').take(4).map(Some).collect()).collect();\n for value in predictor::predict_cells(&rows).unwrap() {{ println!(\"{{:.17}}\",value); }}\n}}\n",serde_json::to_string(&rust_path.display().to_string())?))?;
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
    let rs = command(
        &proof.join("predictor").display().to_string(),
        &[csv_path.display().to_string()],
        "rust-predictions",
        &proof,
    )?;
    let rust = std::str::from_utf8(&rs)?
        .lines()
        .map(str::parse::<f64>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let py_script = "import csv,importlib.util,json,sys; s=importlib.util.spec_from_file_location('verified_predictor',sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); rows=list(csv.reader(open(sys.argv[2])))[1:]; print(json.dumps(m.predict_rows([row[:4] for row in rows])))";
    let py = command(
        "python3",
        &[
            "-c".into(),
            py_script.into(),
            python_path.display().to_string(),
            csv_path.display().to_string(),
        ],
        "python-predictions",
        &proof,
    )?;
    let python: Vec<f64> = serde_json::from_slice(&py)?;
    ensure!(
        rust.len() == native.len() && python.len() == native.len(),
        "source prediction row count differs"
    );
    let mut errors = [0_f64; 3];
    for ((r, p), n) in rust.iter().zip(&python).zip(&native) {
        ensure!(
            r.is_finite() && p.is_finite() && n.is_finite(),
            "nonfinite source prediction"
        );
        errors[0] = errors[0].max((r - n).abs());
        errors[1] = errors[1].max((p - n).abs());
        errors[2] = errors[2].max((r - p).abs());
        ensure!(
            (r - n).abs() <= 1e-12 + 1e-10 * n.abs() && (p - n).abs() <= 1e-12 + 1e-10 * n.abs(),
            "source/engine parity failed"
        );
    }
    // The downloaded artifact stays byte-for-byte unchanged. A separate proof
    // crate appends only a driver, so its private emitted training implementation
    // is exercised in the same crate without modifying the product export.
    let original_rust = fs::read(&rust_path)?;
    let row_literals = table
        .rows
        .iter()
        .map(|row| {
            Ok(format!(
                "vec![{}]",
                row.iter()
                    .map(|cell| -> Result<String> {
                        Ok(match cell {
                            Some(value) => format!("Some({})", serde_json::to_string(value)?),
                            None => "None".into(),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?
                    .join(",")
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let target_at = table
        .headers
        .iter()
        .position(|name| name == &pipeline.input.target.name)
        .context("training target column missing")?;
    let training_driver = format!(
        r#"
// Proof-only driver; every preceding byte is the unchanged production export.
fn main() {{
    let rows:Vec<Vec<Option<&str>>> = vec![{}];
    let fit_ids:&[usize] = &{:?};
    let encoded:Vec<Vec<f64>> = rows.iter().map(|row|
        INPUT_SELECTED.iter().map(|&j| encode_cell(row[j], j).unwrap()).collect()).collect();
    let x:Vec<Vec<f64>> = fit_ids.iter().map(|&i| encoded[i].clone()).collect();
    let y:Vec<f64> = fit_ids.iter().map(|&i|
        (rows[i][{}].unwrap().parse::<f64>().unwrap()-TARGET_MIN)/TARGET_SCALE).collect();
    let model = Model::fit(&x, &y);
    for p in model.predict(&encoded) {{
        let value = if TARGET_SCALE <= 1e-12 {{ TARGET_MIN }} else {{ TARGET_MIN+p*TARGET_SCALE }};
        println!("{{:.17}}", value);
    }}
}}
"#,
        row_literals.join(",\n"),
        fit_ids,
        target_at
    );
    let training_path = proof.join("train-driver.rs");
    let mut training_bytes = original_rust.clone();
    training_bytes.extend_from_slice(training_driver.as_bytes());
    fs::write(&training_path, &training_bytes)?;
    ensure!(
        fs::read(&training_path)?.starts_with(&original_rust),
        "training proof lost the unchanged exported source prefix"
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
    let trained_stdout = command(
        &proof.join("train-predictor").display().to_string(),
        &[],
        "rust-training-predictions",
        &proof,
    )?;
    let trained = std::str::from_utf8(&trained_stdout)?
        .lines()
        .map(str::parse::<f64>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    ensure!(
        trained.len() == native.len(),
        "trained source row count differs"
    );
    let mut training_error = 0_f64;
    for (actual, expected) in trained.iter().zip(&native) {
        ensure!(
            actual.is_finite() && (actual - expected).abs() <= 1e-12 + 1e-10 * expected.abs(),
            "emitted Rust training/native GP parity failed: {actual} versus {expected}"
        );
        training_error = training_error.max((actual - expected).abs());
    }
    let training_proof = json!({"schema":"jope.emitted-training-proof/v1",
        "artifact_sha256":original_rust_sha,"derivative_source_sha256":sha(&training_bytes),
        "unchanged_artifact_prefix_bytes":original_rust.len(),"fit_rows":fit_ids.len(),
        "prediction_rows":native.len(),"heldout_rows":holdout_ids.len(),
        "max_abs_error_vs_native_gp":training_error,"parity_abs_tolerance":1e-12,"parity_rel_tolerance":1e-10,
        "scope":"emitted Model::fit on actual FIT rows using emitted input/target contract, then prediction versus the same actual run's fitted native GP"});
    write_json(&proof.join("training-summary.json"), &training_proof)?;
    write_json(&proof.join("native-predictions.json"), &native)?;
    ensure!(
        sha(&fs::read(&rust_path)?) == original_rust_sha
            && sha(&fs::read(&python_path)?) == original_python_sha,
        "executed source bytes changed"
    );
    ensure!(
        fs::read(csv_path)? == input,
        "input CSV changed during execution proof"
    );
    let source_proof = json!({"schema":"jope-viz.export-proof/v1","case":case_id.trim_start_matches("case-"),
        "run_id":run_id,"input_sha256":sha(&input),"fit_rows":fit_ids.len(),"holdout_rows":holdout_ids.len(),
        "source_sha256":{"rust":sha(&fs::read(&rust_path)?),"python":sha(&fs::read(&python_path)?)},
        "rust_self_check":"pass","python_self_check":"pass","source_unchanged":true,
        "max_abs_errors":{"rust_vs_engine":errors[0],"python_vs_engine":errors[1],"rust_vs_python":errors[2]},
        "parity_rows":native.len(),"engine_parity_rows":native.len(),"training_parity":training_proof,
        "neural_trace_scope":"same actual Engine run and recorded admission slates; no separate inference capture"});
    write_json(&proof.join("summary.json"), &source_proof)?;
    let champion_dsl = pipeline.state.dsl.clone();
    let canonical = feat_math::GenomeIr::parse(&champion_dsl)?.canonical();
    let admitted_rounds = observed
        .calls
        .iter()
        .filter(|call| call.identity["phase"] == "ranker_admission")
        .filter(|call| {
            call.outputs["selected_programs"]
                .as_array()
                .is_some_and(|p| p.iter().any(|v| v.as_str() == Some(&canonical)))
        })
        .map(|call| call.identity["round"].clone())
        .collect::<Vec<_>>();
    use jope_service::discover::r#loop::{NativeSpace, SearchSpace};
    let seed_match = NativeSpace.seeds()?.iter().any(|v| v == &canonical);
    let provenance = json!({"run_id":run_id,"champion_algorithm_id":algorithm,"champion_program_sha256":sha(canonical.as_bytes()),
        "champion_canonical_program":canonical,"selected_in_observed_learned_rounds":admitted_rounds,
        "native_fixed_seed_roster_match":seed_match,
        "credit":if seed_match {"winner matches a native seed; do not credit the neural ranker with writing or selecting it"} else {"use the recorded selected-round identity; the neural component ranks programs and does not emit source"}});
    write_json(&case.join("causal-provenance.json"), &provenance)?;
    let receipt = json!({"schema":"jope.single-run-visual-proof/v1","run_id":run_id,"dataset_id":dataset.id,
        "dataset_sha256":sha(&input),"journal_sha256":sha(&fs::read(case.join("native-journal.json"))?),
        "worker_sha256":sha(&fs::read(worker)?),"authenticated_bundle":release::R882,
        "champion":finished["best"],"model_id":model.id(),"model_manifest_sha256":sha(model.manifest_bytes()),
        "rust_sha256":sha(&fs::read(&rust_path)?),"python_sha256":sha(&fs::read(&python_path)?),
        "parity_rows":native.len(),"heldout_rows":holdout_ids.len(),"fit_rows":fit_ids.len(),"input_warnings":input_warnings,"max_abs_errors":{"rust_vs_engine":errors[0],"python_vs_engine":errors[1],"rust_vs_python":errors[2]},
        "source_self_checks":"both executed successfully","training_parity":training_proof,"causal_provenance":provenance,
        "native_calls":observed.calls.len(),"dropped_records":observed.dropped_records,
        "scope":"one actual in-process RunService run; native conditioning and real admission slates from that run; its champion and production export; no diagnostic-slate splice",
        "inference_device":"cpu","evaluation_workers":1,"listeners_started":0,"token_rate":null});
    write_json(&case.join("single-run-proof.json"), &receipt)?;
    let ranked = observed
        .calls
        .iter()
        .find(|call| call.identity["phase"] == "ranker_admission")
        .context("actual ranking call missing")?;
    let mut affine = Vec::new();
    let mut tensors = Vec::new();
    for call in [conditioning, ranked] {
        for record in &call.trace.affine {
            let mut record = serde_json::to_value(record)?;
            record["call_id"] = json!(call.call_id);
            affine.push(record);
        }
        for record in &call.trace.tensors {
            let mut record = serde_json::to_value(record)?;
            record["call_id"] = json!(call.call_id);
            tensors.push(record);
        }
    }
    let preview = fit_table
        .rows
        .iter()
        .take(5)
        .map(|row| {
            row.iter()
                .map(|value| -> Result<f32> {
                    Ok(value
                        .as_deref()
                        .context("missing numeric preview cell")?
                        .parse()?)
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    let display = json!({"id":case_id,"run_id":run_id,"dataset_id":dataset.id,
        "dataset_sha256":sha(&input),"rows":fit_ids.len(),"input_rows":table.rows.len(),"features":4,
        "fit_row_ids":fit_ids,"holdout_row_ids":holdout_ids,"split_seed":20261103,
        "input_header":table.headers,"table_preview":preview,"diagnostic":false,
        "scope":"one actual production Engine run, its native conditioning, observed real ranking slate, committed champion, export, and source execution",
        "trace":{"schema":"jain.actual-selected-native-calls/v1","affine":affine,"tensors":tensors},
        "display_call_ids":[conditioning.call_id,ranked.call_id],
        "outputs":{"dope629":conditioning.outputs["dope629"],"tokens":conditioning.outputs["tokens"],
            "mask":conditioning.outputs["mask"],"costs":ranked.outputs["total_cost"]},
        "validation":{"actual_run_binding":true,"observed_affine_closure_passed":true,
            "source_execution_parity":true,"source_self_checks_passed":true,"dropped_records":0},
        "source":{"run_id":run_id,"model_id":model.id(),"champion_program_sha256":sha(canonical.as_bytes()),
            "rust_sha256":original_rust_sha,"python_sha256":original_python_sha,
            "proof_path":format!("{case_id}/single-run-proof.json"),
            "proof_sha256":sha(&fs::read(case.join("single-run-proof.json"))?),
            "parity_rows":native.len(),"engine_parity_rows":native.len(),"heldout_rows":holdout_ids.len(),
            "training_proof_path":format!("{case_id}/proof/training-summary.json"),
            "training_proof_sha256":sha(&fs::read(proof.join("training-summary.json"))?)},
        "native_journal_path":format!("{case_id}/native-journal.json"),
        "native_journal_sha256":sha(&fs::read(case.join("native-journal.json"))?),
        "causal_provenance":provenance,"tokens_per_second":null,
        "candidate_input_scope":"actual first recorded production ranking slate; full real rounds retained in the same-run journal"});
    println!(
        "PASS actual run {run_id}: {} native calls, {} source parity rows, max error {:.3e}; {}",
        observed.calls.len(),
        native.len(),
        errors.into_iter().fold(0_f64, f64::max),
        case.display()
    );
    Ok((
        display,
        source_proof,
        json!({"case_id":case_id,"model_directory":model_dir,"source":rust_path,"run_id":run_id}),
        Value::Object(assets),
    ))
}
