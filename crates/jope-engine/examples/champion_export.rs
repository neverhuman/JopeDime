//! Export one per-dataset-fold high mark (parallel.cut_champion.v1) as standalone code with a parity receipt.
//!
//! Deterministic GP-to-code without an LLM: the champion's genome is emitted through feat_math::transpile as
//! predict.py / predict.rs (emit), predict.cpp (C++17) and predict.js (ES2020) when each rung supports it; the native
//! predictor is re-fitted on the bound training view (gp::fit_view) and must reproduce the observation's semantic
//! sha and RMSE (as predictor_parity does); the emitted Python program is then executed on the same normalized rows and
//! its predictions must agree with the native predictor elementwise (|d| <= 1e-9 * max(1, |y|)) and in RMSE
//! (<= 1e-9 * max(1, rmse)). Everything is written under --output; nothing under runs/ or data/ is modified; the
//! sealed test partition is never read (a champion identity is meta_train or meta_validation by construction).
//!
//!   champion_export --root ROOT --event runs/champions/<key>/event-<sha>.json --prepared data/prepared/<view> \
//!                   --output <dir> [--skip-python]
use anyhow::{bail, ensure, Context, Result};
use clap::Parser;
use feat_math::{transpile, GenomeIr};
use jope_parallel::{
    champions::Champion,
    gp,
    manifest::{file_hash, hash, read_json, resolve, write_json},
    prepare::{load_view, Prepared},
};
use serde_json::json;
use std::{fs, path::PathBuf, process::Command, time::Instant};

const PARITY_SCHEMA: &str = "parallel.champion_parity.v1";
const EXPORT_SCHEMA: &str = "parallel.champion_export.v1";
const TOLERANCE: f64 = 1e-9;
const CLAIM: &str = "local record high on the adaptive cut-heldout teacher; standalone emitted code with a replay check; not a held-out win, not an independent evaluation, not a serving promotion";

#[derive(Parser)]
struct Cli {
    #[arg(long)]
    root: PathBuf,
    /// One immutable champion event file (parallel.cut_champion.v1).
    #[arg(long)]
    event: PathBuf,
    /// The prepared view directory whose prepared.json hashes to the champion's identity.prepared_sha256.
    #[arg(long)]
    prepared: PathBuf,
    #[arg(long)]
    output: PathBuf,
    /// Emit only; do not execute the emitted Python program.
    #[arg(long, default_value_t = false)]
    skip_python: bool,
    /// Wall-clock cap for the emitted Python run, seconds.
    #[arg(long, default_value_t = 120)]
    python_timeout: u64,
}

fn write(path: &PathBuf, text: &str) -> Result<String> {
    fs::write(path, text.as_bytes())?;
    Ok(hash(text.as_bytes()))
}

fn csv_rows(rows: &[Vec<f64>], drop_last: bool) -> String {
    let mut out = String::new();
    for row in rows {
        let cells: Vec<String> = row[..row.len() - usize::from(drop_last)]
            .iter()
            .map(|v| format!("{v:?}"))
            .collect();
        out.push_str(&cells.join(","));
        out.push('\n');
    }
    out
}

fn rmse(actual: &[f64], predicted: &[f64]) -> f64 {
    let n = actual.len().max(1) as f64;
    (actual
        .iter()
        .zip(predicted)
        .map(|(a, p)| (a - p) * (a - p))
        .sum::<f64>()
        / n)
        .sqrt()
}

fn main() -> Result<()> {
    let args = Cli::parse();
    let started = Instant::now();
    let root = args.root.canonicalize()?;
    let event_path = resolve(&root, &args.event)?;
    let prepared = resolve(&root, &args.prepared)?;
    let output = resolve(&root, &args.output)?;
    ensure!(
        !output.exists(),
        "output exists; exports are immutable ({})",
        output.display()
    );
    let event_bytes = fs::read(&event_path)?;
    let champion: Champion = serde_json::from_slice(&event_bytes)?;
    champion.validate()?;
    let event_sha = hash(&serde_json::to_vec(&champion)?);
    ensure!(
        event_path.file_name().and_then(|n| n.to_str()) == Some(&format!("event-{event_sha}.json")),
        "event file name does not match its content hash"
    );
    ensure!(
        matches!(
            champion.identity.partition.as_str(),
            "meta_train" | "meta_validation"
        ),
        "only meta_train / meta_validation champions are exported"
    );
    ensure!(
        file_hash(&prepared.join("prepared.json"))? == champion.identity.prepared_sha256,
        "prepared view does not hash to the champion identity"
    );
    let p: Prepared = read_json(&prepared.join("prepared.json"))?;
    ensure!(
        p.task == "regression",
        "regression champions only in this export"
    );
    let cut = champion.identity.cut;
    let genome = champion.genome().to_owned();
    let program_sha = hash(genome.as_bytes());
    ensure!(
        champion.observation["program_sha256"] == program_sha,
        "genome bytes do not hash to program_sha256"
    );
    fs::create_dir_all(&output)?;

    // 1. Emit every supported rung.
    let ir = GenomeIr::parse(&genome)?;
    let opts = transpile::EmitOptions::default();
    let mut rungs = serde_json::Map::new();
    let supports = transpile::supports(&ir);
    let (py_sha, rs_sha) = match supports {
        Ok(()) => {
            let emitted = transpile::emit(&ir, &opts)?;
            let py = write(&output.join("predict.py"), &emitted.python)?;
            let rs = write(&output.join("predict.rs"), &emitted.rust)?;
            write(&output.join("artifact-manifest.json"), &emitted.manifest)?;
            rungs.insert(
                "python".into(),
                json!({"status":"emitted","sha256":py,"tokens":emitted.py_tokens}),
            );
            rungs.insert(
                "rust".into(),
                json!({"status":"emitted","sha256":rs,"tokens":emitted.rs_tokens}),
            );
            (Some(py), Some(rs))
        }
        Err(missing) => {
            rungs.insert(
                "python".into(),
                json!({"status":"unsupported","missing":missing}),
            );
            rungs.insert(
                "rust".into(),
                json!({"status":"unsupported","missing":missing}),
            );
            (None, None)
        }
    };
    match transpile::supports_cpp17(&ir) {
        Ok(()) => {
            let cpp = transpile::emit_cpp17_counted(&ir, &opts)?;
            let sha = write(&output.join("predict.cpp"), &cpp.source)?;
            rungs.insert("cpp17".into(), json!({"status":"emitted","sha256":sha,"math_policy":cpp.math_policy,"executed":false}));
        }
        Err(missing) => {
            rungs.insert(
                "cpp17".into(),
                json!({"status":"unsupported","missing":missing}),
            );
        }
    }
    match transpile::supports_javascript(&ir) {
        Ok(()) => {
            let js = transpile::emit_javascript_counted(&ir, &opts)?;
            let sha = write(&output.join("predict.js"), &js.source)?;
            rungs.insert("javascript".into(), json!({"status":"emitted","sha256":sha,"math_policy":js.math_policy,"executed":false,"reason":"no JavaScript runtime on the node"}));
        }
        Err(missing) => {
            rungs.insert(
                "javascript".into(),
                json!({"status":"unsupported","missing":missing}),
            );
        }
    }

    // 2. Native replay: re-fit the genome on the bound training view and reproduce the observation.
    let model_path = output.join("predictor.json");
    gp::fit_view(&prepared, cut, &genome, &model_path)?;
    let model: gp::Predictor = read_json(&model_path)?;
    let train = load_view(&prepared, cut, true)?;
    let validation = load_view(&prepared, cut, false)?;
    let x_val: Vec<Vec<f64>> = validation
        .iter()
        .map(|r| r[..r.len() - 1].to_vec())
        .collect();
    let y_val: Vec<f64> = validation.iter().map(|r| r[r.len() - 1]).collect();
    let native = model.model.predict(&x_val);
    let semantic_sha = hash(&serde_json::to_vec(&native)?);
    let observed = champion.observation["metrics"]["rmse"]
        .as_f64()
        .context("observation rmse")?;
    let target_scale = champion.observation["target_scale"]
        .as_f64()
        .context("target_scale")?;
    let native_rmse = rmse(&y_val, &native);
    let semantic_ok = champion.observation["metrics"]["semantic_sha256"]
        .as_str()
        .map(|s| s == semantic_sha);
    let native_ok = (native_rmse - observed).abs() <= TOLERANCE * observed.max(1.0);
    ensure!(
        (native_rmse * target_scale - champion.rmse_original_units).abs()
            <= TOLERANCE * champion.rmse_original_units.max(1.0),
        "native replay RMSE in original units differs from the champion record"
    );

    // 3. Emitted Python replay on the same normalized rows.
    let mut python = json!({"status":"skipped"});
    if let (Some(_), false) = (py_sha.as_ref(), args.skip_python) {
        let train_csv = output.join("train.csv");
        let test_csv = output.join("test_features.csv");
        let pred_csv = output.join("predictions.py.csv");
        fs::write(&train_csv, csv_rows(&train, false))?;
        fs::write(&test_csv, csv_rows(&x_val, false))?;
        let run = Command::new("timeout")
            .arg(args.python_timeout.to_string())
            .arg("python3")
            .arg(output.join("predict.py"))
            .arg(&train_csv)
            .arg(&test_csv)
            .arg(&pred_csv)
            .env("OMP_NUM_THREADS", "1")
            .output()?;
        if !run.status.success() {
            python = json!({"status":"failed","exit":run.status.code(),"stderr":String::from_utf8_lossy(&run.stderr).chars().take(2000).collect::<String>()});
        } else {
            let text = fs::read_to_string(&pred_csv)?;
            let predictions: Vec<f64> = text
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| l.trim().parse::<f64>())
                .collect::<std::result::Result<_, _>>()?;
            ensure!(
                predictions.len() == native.len(),
                "emitted python produced {} predictions for {} rows",
                predictions.len(),
                native.len()
            );
            let max_abs = predictions
                .iter()
                .zip(&native)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            let elementwise = predictions
                .iter()
                .zip(&native)
                .all(|(a, b)| (a - b).abs() <= TOLERANCE * b.abs().max(1.0));
            let py_rmse = rmse(&y_val, &predictions);
            let rmse_ok = (py_rmse - native_rmse).abs() <= TOLERANCE * native_rmse.max(1.0);
            python = json!({"status": if elementwise && rmse_ok {"pass"} else {"fail"}, "rows":predictions.len(), "max_abs_difference":max_abs,
                            "rmse":py_rmse, "native_rmse":native_rmse, "elementwise_within_tolerance":elementwise, "rmse_within_tolerance":rmse_ok});
        }
    }
    let parity_status = if native_ok
        && semantic_ok != Some(false)
        && python["status"] != "fail"
        && python["status"] != "failed"
    {
        "pass"
    } else {
        "fail"
    };
    let parity = json!({"schema":PARITY_SCHEMA,"status":parity_status,"tolerance":TOLERANCE,"native":{"status": if native_ok {"pass"} else {"fail"},"rmse_normalized":native_rmse,
        "observation_rmse_normalized":observed,"rmse_original_units":native_rmse*target_scale,"champion_rmse_original_units":champion.rmse_original_units,
        "semantic_sha256":semantic_sha,"observation_semantic_sha256":champion.observation["metrics"]["semantic_sha256"],"semantic_equal":semantic_ok,
        "validation_rows":y_val.len(),"training_rows":train.len()},"python":python,"prediction_input":"bound cut validation rows within the champion's own partition",
        "final_tests_opened":false});
    write_json(&output.join("parity.json"), &parity)?;
    let export = json!({"schema":EXPORT_SCHEMA,"event_sha256":event_sha,"program_sha256":program_sha,"identity":champion.identity,
        "rmse_original_units":champion.rmse_original_units,"improvement_number":champion.improvement_number,"source":champion.source,
        "prepared":{"path":args.prepared,"prepared_sha256":champion.identity.prepared_sha256},"rungs":rungs,"parity_status":parity_status,
        "python_sha256":py_sha,"rust_sha256":rs_sha,"emitter":{"math_kernel":opts.math_kernel.policy_id()},"llm_calls":0,"claim":CLAIM,
        "seconds":started.elapsed().as_secs_f64(),"final_tests_opened":false});
    write_json(&output.join("export.json"), &export)?;
    fs::write(output.join("champion.json"), &event_bytes)?;
    if parity_status != "pass" {
        // The record is still exported; the receipt says the emitter or replay disagreed. Exit non-zero so a publisher counts it.
        println!("{}", serde_json::to_string(&export)?);
        bail!("parity failed for {}", event_sha);
    }
    println!("{}", serde_json::to_string(&export)?);
    Ok(())
}
