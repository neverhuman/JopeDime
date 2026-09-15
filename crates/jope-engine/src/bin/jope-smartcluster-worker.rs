use clap::Parser;
use std::path::PathBuf;
#[derive(Parser)]
struct Args {
    #[arg(long, required_unless_present = "inspect_runtime")]
    task_manifest: Option<PathBuf>,
    #[arg(long, required_unless_present = "inspect_runtime")]
    manifest_sha256: Option<String>,
    /// CPU-only loaded-image observation; never grants numerical execution.
    #[arg(long, conflicts_with_all=["task_manifest","manifest_sha256"])]
    inspect_runtime: bool,
}
fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if args.inspect_runtime {
        println!(
            "{}",
            serde_json::to_string_pretty(
                &jope_parallel::smartcluster_worker::runtime_observation()?
            )?
        );
        return Ok(());
    }
    jope_parallel::smartcluster_worker::serve(
        &args.task_manifest.expect("required argument"),
        &args.manifest_sha256.expect("required argument"),
    )
}
