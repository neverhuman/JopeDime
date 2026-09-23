//! A single fit's private input/output boundary. Resource enforcement belongs to the
//! shared controller, which starts this entry point in its admitted worker cgroup.

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    fs::OpenOptions,
    io::{Read, Write},
    path::Path,
};

use super::{
    methods::{fit_rust, FittedRustArtifact, RustFitRequest},
    score::Predictions,
};

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RustWorkerOutput {
    pub schema: String,
    pub input_sha256: String,
    pub artifact_sha256: String,
    pub predictions: Predictions,
    pub artifact: FittedRustArtifact,
}

pub fn fit(input: &Path, output: &Path) -> Result<()> {
    ensure!(
        std::fs::metadata(input)?.len() <= 128 * 1024 * 1024,
        "fit input exceeds 128 MiB"
    );
    let mut bytes = Vec::new();
    std::fs::File::open(input)
        .context("read private fit input")?
        .take(128 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() <= 128 * 1024 * 1024,
        "fit input exceeds 128 MiB"
    );
    let request: RustFitRequest = serde_json::from_slice(&bytes)?;
    let fitted = fit_rust(&request)?;
    let artifact_bytes = fitted.artifact.bytes()?;
    let artifact_sha256 = jope_pipeline::sha256_hex(&artifact_bytes);
    // Prove that the serialized, complete fitted state can reproduce the output before
    // publishing it. This never refits and has no access to held-out targets.
    let restored = FittedRustArtifact::from_bytes(&artifact_bytes, &artifact_sha256)?;
    ensure!(
        serde_json::to_vec(&restored.predict(&request.predict)?)?
            == serde_json::to_vec(&fitted.predictions)?,
        "fitted reload changed predictions"
    );
    write_private(
        output,
        &RustWorkerOutput {
            schema: "jope.panel-rust-worker-output/v1".into(),
            input_sha256: jope_pipeline::sha256_hex(&bytes),
            artifact_sha256,
            predictions: fitted.predictions,
            artifact: fitted.artifact,
        },
    )
}

/// Immutable private artifacts; an existing path is never silently overwritten on resume.
pub fn write_private(path: &Path, value: &impl Serialize) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)?;
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(path)
        .with_context(|| format!("create {}", path.display()))?;
    file.write_all(&bytes)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use feat_math::advanced_v3::{ColumnSpec, DataSchema, RawTable, RowSemantics};
    use jope_datasets::catalog::Task;

    #[test]
    fn worker_retains_fitted_state_and_never_overwrites_a_previous_result() {
        let tmp = tempfile::tempdir().unwrap();
        let input = tmp.path().join("input.json");
        let output = tmp.path().join("output.json");
        let request = RustFitRequest {
            task: Task::Regression,
            schema: DataSchema {
                columns: vec![ColumnSpec::numeric("value")],
                row_semantics: RowSemantics::Iid,
            },
            classes: vec![],
            train: RawTable::new(
                vec!["value".into()],
                (0..12).map(|i| vec![i.to_string()]).collect(),
            )
            .unwrap(),
            targets: (0..12).map(|i| (2 * i) as f64).collect(),
            predict: RawTable::new(
                vec!["value".into()],
                vec![vec!["1.5".into()], vec!["".into()]],
            )
            .unwrap(),
            method: super::super::methods::rust_methods(Task::Regression)
                .unwrap()
                .remove(0),
        };
        write_private(&input, &request).unwrap();
        fit(&input, &output).unwrap();
        let original = std::fs::read(&output).unwrap();
        let result: RustWorkerOutput = serde_json::from_slice(&original).unwrap();
        assert_eq!(
            result.input_sha256,
            jope_pipeline::sha256_hex(&std::fs::read(&input).unwrap())
        );
        assert_eq!(
            result.artifact_sha256,
            jope_pipeline::sha256_hex(&result.artifact.bytes().unwrap())
        );
        assert_eq!(
            result.predictions,
            result.artifact.predict(&request.predict).unwrap()
        );
        assert!(fit(&input, &output).is_err());
        assert_eq!(original, std::fs::read(&output).unwrap());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&output).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
    }
}
