//! Private worker build provenance, separate from the CLI installation receipt.
//! The controller pins this inventory before a pair starts. It authenticates
//! recorded builds of trusted source; it is not a reproducible-build claim.
use anyhow::{ensure, Context, Result};
use jope_pipeline::execution::Artifact;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, fs};

pub const SCHEMA: &str = "jope.asm-worker-build-inventory/v1";
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerBuildInventory {
    pub schema: String,
    pub source_sha40: String,
    pub source_manifest: Artifact,
    pub workers: Vec<WorkerBuild>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerBuild {
    pub role: String,
    pub executable: Artifact,
    pub features: Vec<String>,
    pub build_receipt: Artifact,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuildReceipt {
    pub schema: String,
    pub source_sha40: String,
    pub source_manifest_sha256: String,
    pub command: Vec<String>,
    pub working_directory: String,
    pub started_unix_ms: u64,
    pub finished_unix_ms: u64,
    pub exit_code: i32,
    /// The unmodified stdout produced with --message-format=json-render-diagnostics.
    pub cargo_messages: Artifact,
    /// Paths from Cargo messages and checksums captured immediately after build.
    /// The immutable deployed copy must retain these exact executable bytes.
    pub executables: Vec<BuiltExecutable>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuiltExecutable {
    pub target_name: String,
    pub path: std::path::PathBuf,
    pub sha256: String,
}
impl WorkerBuildInventory {
    pub fn verify(&self, config: &super::optimize::Configuration) -> Result<Vec<Artifact>> {
        ensure!(
            self.schema == SCHEMA && self.source_sha40 == config.source_sha40,
            "worker inventory source or schema mismatch"
        );
        ensure!(
            self.source_manifest.path == config.source_manifest.path
                && self.source_manifest.sha256 == config.source_manifest.sha256
                && self.workers.len() == 2,
            "worker inventory requires exact source manifest and two roles"
        );
        self.source_manifest.verify()?;
        let mut closure = vec![self.source_manifest.clone()];
        for (role, binary, runtime) in [
            ("policy", "jope-policy-producer", &config.policy),
            ("context", "jope-context-worker", &config.encoder),
        ] {
            let matches: Vec<_> = self.workers.iter().filter(|w| w.role == role).collect();
            ensure!(matches.len() == 1, "missing or duplicate worker build role");
            let worker = matches[0];
            ensure!(
                worker.executable.path == runtime.program.path
                    && worker.executable.sha256 == runtime.program.sha256,
                "worker build executable differs from frozen runtime"
            );
            worker.executable.verify()?;
            worker.build_receipt.verify()?;
            let build: BuildReceipt =
                serde_json::from_slice(&fs::read(&worker.build_receipt.path)?)?;
            ensure!(
                build.schema == "jope.asm-cargo-build/v1"
                    && build.exit_code == 0
                    && build.source_sha40 == self.source_sha40
                    && build.source_manifest_sha256 == self.source_manifest.sha256
                    && build.finished_unix_ms >= build.started_unix_ms
                    && build.started_unix_ms > 0
                    && std::path::Path::new(&build.working_directory).is_absolute(),
                "unsuccessful or unbound worker build"
            );
            ensure!(
                build.command.iter().any(|a| a == "build")
                    && build
                        .command
                        .iter()
                        .any(|a| a.starts_with("--message-format=json")),
                "worker build lacks Cargo artifact capture"
            );
            build.cargo_messages.verify()?;
            let expected: BTreeSet<_> = worker.features.iter().cloned().collect();
            ensure!(
                expected.len() == worker.features.len() && expected.contains("context-producer"),
                "worker features must include qualified CPU context consumption"
            );
            let built: Vec<_> = build
                .executables
                .iter()
                .filter(|e| e.target_name == binary)
                .collect();
            ensure!(
                built.len() == 1
                    && built[0].path.is_absolute()
                    && built[0].sha256 == worker.executable.sha256,
                "immutable worker copy differs from captured build artifact"
            );
            let mut matching = 0;
            let mut successful = false;
            for line in fs::read_to_string(&build.cargo_messages.path)?.lines() {
                let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
                    continue;
                };
                if value["reason"] == "build-finished" && value["success"] == true {
                    successful = true;
                }
                if value["reason"] != "compiler-artifact" || value["target"]["name"] != binary {
                    continue;
                }
                let Some(path) = value["executable"].as_str() else {
                    continue;
                };
                let features: BTreeSet<String> = serde_json::from_value(value["features"].clone())
                    .context("Cargo artifact features absent")?;
                ensure!(
                    features == expected && std::path::Path::new(path) == built[0].path,
                    "Cargo artifact executable or feature identity mismatch"
                );
                matching += 1;
            }
            ensure!(
                matching == 1 && successful,
                "worker build lacks one actual matching executable and successful completion"
            );
            closure.extend([
                worker.executable.clone(),
                worker.build_receipt.clone(),
                build.cargo_messages,
            ]);
        }
        Ok(closure)
    }
}
