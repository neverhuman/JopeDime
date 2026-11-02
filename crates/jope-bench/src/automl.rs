//! Original Rust submissions registered with the same-cut AutoML comparison path.
//!
//! The learner is frozen source; this module owns transport and registration only.
//! No fitted artifact or official challenge qualification is implied by registration.

use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};

use crate::adapter::{parse_spec, AdapterSpec};

// Preserve the submitted bytes, including formatting. Style diagnostics in the
// immutable entrant are not integration edits; rustc still forbids unsafe code.
#[rustfmt::skip]
#[allow(clippy::needless_range_loop, clippy::unnecessary_map_or)]
#[path = "../submissions/GPT6_Astra_Max.rs"]
pub mod submission;

use submission::{Regressor, RegressorApi};

/// Stable CLI name for this original learner.
pub const NAME: &str = "gpt6-astra-max";
/// Identity embedded in adapter version output and checked by the contract tests.
pub const SOURCE_SHA256: &str = "b51ad3291c6e433388a49ed8338d069a245300d634c9da76be1fb1f1454cd0eb";

/// Machine-readable, additive AutoML catalog. Historical program catalogs are unchanged.
pub fn registry() -> serde_json::Value {
    serde_json::json!({
        "schema": "jope.original-rust-automl/v1",
        "methods": [{
            "name": NAME,
            "algorithm_name": "GPT6_Astra_Max",
            "method_id": "adapter.GPT6_Astra_Max",
            "tasks": ["regression"],
            "language": "rust",
            "source_sha256": SOURCE_SHA256,
            "input_contract": "last_target_features_only_prediction",
            "single_threaded_learner": true,
            "primary_challenge_eligibility": "not_asserted",
            "predictive_quality": "unmeasured"
        }]
    })
}

/// Register an in-binary adapter using argv boundaries (executable paths may contain spaces).
/// The seed is explicit, recorded in the command and reused for every independent fitting job.
pub fn adapter_spec(
    name: &str,
    executable: &Path,
    seed: u64,
    timeout_seconds: f64,
) -> Result<AdapterSpec> {
    ensure!(name == NAME, "unknown AutoML method {name:?}");
    let mut spec = parse_spec("GPT6_Astra_Max=unused", timeout_seconds)?;
    spec.command = vec![
        executable
            .to_str()
            .context("non-UTF-8 executable path")?
            .to_owned(),
        "automl-adapter".to_owned(),
        name.to_owned(),
        seed.to_string(),
    ];
    Ok(spec)
}

struct NumericTable {
    values: Vec<f64>,
    rows: usize,
    columns: usize,
}

/// Headerless numeric CSV. For zero-width predictions each empty line is one row;
/// an empty file is an empty batch. Other empty rows are errors, never silently dropped.
fn read_table(path: &Path, expected_width: Option<usize>) -> Result<NumericTable> {
    let reader = BufReader::new(std::fs::File::open(path).context("open numeric input")?);
    let mut values = Vec::new();
    let mut rows = 0usize;
    let mut width = expected_width;
    for line in reader.lines() {
        let line = line.context("read numeric input")?;
        let line = line.trim();
        if width == Some(0) {
            ensure!(
                line.is_empty(),
                "zero-feature prediction row contains a field"
            );
        } else {
            ensure!(!line.is_empty(), "empty numeric row at line {}", rows + 1);
            let cells = line.split(',').collect::<Vec<_>>();
            let columns = *width.get_or_insert(cells.len());
            ensure!(
                cells.len() == columns,
                "inconsistent numeric row width at line {}",
                rows + 1
            );
            values
                .try_reserve(cells.len())
                .context("numeric input allocation")?;
            for cell in cells {
                values.push(cell.trim().parse::<f64>().context("invalid numeric cell")?);
            }
        }
        rows = rows.checked_add(1).context("numeric row count overflow")?;
    }
    Ok(NumericTable {
        values,
        rows,
        columns: width.unwrap_or(0),
    })
}

fn execute_adapter(args: &[String]) -> Result<()> {
    ensure!(args.len() >= 2 && args[0] == NAME,
        "usage: automl-adapter gpt6-astra-max <seed> --version | <train.csv> <predict.csv> <out.csv> regression");
    let seed = args[1].parse::<u64>().context("invalid AutoML seed")?;
    if args.len() == 3 && args[2] == "--version" {
        println!("GPT6_Astra_Max 1 source-sha256={SOURCE_SHA256} seed={seed}");
        return Ok(());
    }
    ensure!(
        args.len() == 6,
        "expected train, prediction, output paths and regression task"
    );
    ensure!(
        args[5] == "regression",
        "GPT6_Astra_Max supports regression only"
    );
    let train = read_table(Path::new(&args[2]), None)?;
    ensure!(
        train.columns >= 1 && train.rows >= 1,
        "training table is empty"
    );
    let mut model = Regressor::new(seed);
    model
        .fit(&train.values, train.rows, train.columns)
        .map_err(anyhow::Error::msg)?;
    let predict = read_table(Path::new(&args[3]), Some(train.columns - 1))?;
    let predictions = model
        .predict(&predict.values, predict.rows, predict.columns)
        .map_err(anyhow::Error::msg)?;
    ensure!(predictions.len() == predict.rows, "wrong prediction count");
    if predictions.iter().any(|value| !value.is_finite()) {
        bail!("nonfinite prediction");
    }
    let mut writer =
        BufWriter::new(std::fs::File::create(&args[4]).context("create prediction output")?);
    for value in predictions {
        writeln!(writer, "{value:?}").context("write prediction output")?;
    }
    writer.flush().context("flush prediction output")
}

/// Process transport for the existing `jope.bench-adapter/v1` controller.
pub fn adapter_main(args: &[String]) -> i32 {
    match execute_adapter(args) {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("GPT6_Astra_Max: {error:#}");
            1
        }
    }
}

#[cfg(test)]
#[path = "automl_contract_tests.rs"]
mod contract_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn frozen_source_and_catalog_identity_agree() {
        let source = include_bytes!("../submissions/GPT6_Astra_Max.rs");
        assert_eq!(format!("{:x}", Sha256::digest(source)), SOURCE_SHA256);
        let provenance: serde_json::Value = serde_json::from_str(include_str!(
            "../submissions/gpt6-astra-max/provenance.json"
        ))
        .unwrap();
        assert_eq!(provenance["source_sha256"], SOURCE_SHA256);
        assert_eq!(registry()["methods"][0]["source_sha256"], SOURCE_SHA256);
    }

    #[test]
    fn registration_preserves_argv_and_rejects_invalid_requests() {
        let spec = adapter_spec(NAME, Path::new("path with spaces/jope-bench"), 42, 600.0).unwrap();
        assert_eq!(spec.id(), "adapter.GPT6_Astra_Max");
        assert_eq!(
            spec.command,
            ["path with spaces/jope-bench", "automl-adapter", NAME, "42"]
        );
        assert!(adapter_spec("unknown", Path::new("jope-bench"), 42, 600.0).is_err());
        assert!(adapter_spec(NAME, Path::new("jope-bench"), 42, f64::NAN).is_err());
        assert!(execute_adapter(&[
            NAME.into(),
            "42".into(),
            "a".into(),
            "b".into(),
            "c".into(),
            "binary".into()
        ])
        .is_err());
    }
}
