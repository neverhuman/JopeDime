//! Small front door. Report the measured scope; unsupported capabilities remain
//! explicit until their own implementation and qualification have landed.
use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::{path::PathBuf, process::ExitCode};

#[derive(Parser)]
pub struct BenchmarkCli {
    #[command(subcommand)]
    pub command: BenchmarkCommand,
}
#[derive(Subcommand)]
pub enum BenchmarkCommand {
    /// Verify the fixed panel's task, metric, split and input identities (no fits).
    VerifyManifest {
        #[arg(long)]
        manifest: PathBuf,
    },
}
pub fn benchmark(cli: BenchmarkCli) -> ExitCode {
    let result = match cli.command {
        BenchmarkCommand::VerifyManifest { manifest } => verify_manifest(&manifest),
    };
    match result {
        Ok(value) => {
            println!("{value}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("jope benchmark: {error:#}");
            ExitCode::FAILURE
        }
    }
}
fn verify_manifest(path: &std::path::Path) -> Result<serde_json::Value> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let panel: crate::experiment::Panel = serde_json::from_slice(&bytes)?;
    panel.validate()?;
    Ok(
        serde_json::json!({"schema":"jope.panel-check/v1", "ok":true,
        "panel":panel.panel_id, "manifest_sha256":crate::sha256_hex(&bytes),
        "datasets":panel.datasets.len(), "input_bytes_verified":false,
        "note":"Verify materialized inputs with benchmarks/prepare.py before execution; no fits performed."}),
    )
}
pub fn doctor() -> serde_json::Value {
    let heads: Vec<_> = feat_math::registry::operator_registry()
        .iter()
        .filter(|op| op.role == feat_math::registry::OperatorRole::Head)
        .map(|op| {
            serde_json::json!({"name":op.name, "algorithm":op.semantic_key,
            "lifecycle":op.lifecycle, "declared_tasks":op.tasks})
        })
        .collect();
    serde_json::json!({
        "schema":"jope.doctor/v1", "name":"Algorithm Synthesis Model (ASM)",
        "description":"learns to generate and optimize executable algorithms from data",
        "tool_version":crate::TOOL_VERSION, "panel_ready":false,
        "capabilities":[
            {"name":"regression pipeline", "status":"implemented", "command":"jope pipeline run", "limit":"generic GP seeds; numerical finite inputs; recipe source emission"},
            {"name":"binary and multiclass pipeline", "status":"not_integrated", "limit":"task-aware kernels exist; the regression pipeline refuses these tasks"},
            {"name":"bounded 30-dataset benchmark", "status":"contract_only", "limit":"execution and method adapters are pending"},
            {"name":"neural generation-zero proposals", "status":"not_integrated", "limit":"accepted regression policy exists; pipeline does not consume it"},
            {"name":"policy training", "status":"not_qualified", "limit":"existing machinery needs a bounded execution proof; train-step is a centroid router"},
            {"name":"fitted predictor export across five backends", "status":"not_qualified", "limit":"existing emitters have different coverage; recipe emission is not fitted inference parity"}
        ],
        "native_head_inventory":heads,
        "inventory_note":"Governed registry metadata, including aliases and unavailable entries; not a runnable or qualified benchmark roster. Advanced-v3 heads are inventoried separately in docs/asm.md.",
        "legacy_fixture_limit":"fit/search/router use numeric R2 diagnostics; non-regression outputs are not classification evidence and are excluded from the panel",
        "serving_state_changed":false
    })
}
